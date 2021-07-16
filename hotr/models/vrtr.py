# ------------------------------------------------------------------------
# HOTR official code : hotr/models/hotr.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from hotr.util.misc import NestedTensor, nested_tensor_from_tensor_list
from .feed_forward import MLP
from torchvision.ops import roi_align

# sequential visual relationship transformer
class VRTR(nn.Module):
    def __init__(self, args, detr):
        super().__init__()

        # * Instance Transformer ---------------
        self.detr = detr
        if args.frozen_weights is not None:
            # if this flag is given, freeze the object detection related parameters of DETR
            for p in self.parameters():
                p.requires_grad_(False)
        # --------------------------------------

        # relation proposal
        rel_rep_dim = 1024
        self.union_box_feature_extractor = RelationFeatureExtractor(in_channels=2048, resolution=7, out_dim=rel_rep_dim)
        self.relation_proposal_mlp = nn.Sequential(
            make_fc(rel_rep_dim, rel_rep_dim // 2), nn.ReLU(),
            make_fc(rel_rep_dim // 2, 1)
        )

        # relation classification

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # >>>>>>>>>>>>  BACKBONE LAYERS  <<<<<<<<<<<<<<<
        features, pos = self.detr.backbone(samples)
        bs = features[-1].tensors.shape[0]
        src, mask = features[-1].decompose()
        assert mask is not None
        # ----------------------------------------------

        # >>>>>>>>>>>> OBJECT DETECTION LAYERS <<<<<<<<<<
        hs, _ = self.detr.transformer(self.detr.input_proj(src), mask, self.detr.query_embed.weight, pos[-1])
        inst_repr = F.normalize(hs[-1], p=2, dim=2) # instance representations
        num_nodes = inst_repr.shape[1]

        # Prediction Heads for Object Detection
        outputs_class = self.detr.class_embed(hs)[-1]
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()[-1]
        # -----------------------------------------------

        # >>>>>>>>>>>> HOI DETECTION LAYERS <<<<<<<<<<<<<<<
        for imgid in range(bs):
            # >>>>>>>>>>>> relation proposal <<<<<<<<<<<<<<<
            inst_labels = outputs_class[imgid].max(-1)[-1]
            rel_mat = torch.zeros((num_nodes, num_nodes))
            rel_mat[inst_labels==1] = 1
            rel_pairs = rel_mat.nonzero(as_tuple=False)
            rel_reps = self.union_box_feature_extractor(features[-1], outputs_coord[imgid], rel_pairs, idx=imgid)
            p_relation_exist_logits = self.relation_proposal_mlp(rel_reps)

            # >>>>>>>>>>>> relation classification <<<<<<<<<<<<<<<
            # outputs_action =

        out = {
            "pred_logits": outputs_class,
            "pred_boxes": outputs_coord,
            # "pred_rel_pairs": outputs_action[-1],
            # "pred_actions": outputs_action[-1],
        }
        return out

class VRTRCriterion(nn.Module):
    """ This class computes the loss for VRTR.
    1. proposal loss
    2. relation classification loss
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, outputs, targets):
        pass


class VRTRPostProcess(nn.Module):
    def __init__(self, args, model):
        super().__init__()
        self.args = args

    @torch.no_grad()
    def forward(self, outputs, target_sizes, threshold=0, dataset='coco'):
       pass


def make_fc(dim_in, hidden_dim, a=1):
    '''
        Caffe2 implementation uses XavierFill, which in fact
        corresponds to kaiming_uniform_ in PyTorch
        a: negative slope
    '''
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=a)
    nn.init.constant_(fc.bias, 0)
    return fc


def make_conv3x3(
    in_channels,
    out_channels,
    padding=1,
    dilation=1,
    stride=1,
    kaiming_init=True
):
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if kaiming_init:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    else:
        torch.nn.init.normal_(conv.weight, std=0.01)
        nn.init.constant_(conv.bias, 0)
    return conv

# todo: add semantic feature
class RelationFeatureExtractor(nn.Module):
    def __init__(self, in_channels, resolution=7, out_dim=1024):
        super(RelationFeatureExtractor, self).__init__()
        self.resolution = resolution

        # reduce channel size before pooling
        out_ch = 256
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_ch, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.visual_proj = make_fc(out_ch * (resolution**2), out_dim, a=0)

        # rectangle
        spatial_out_ch = out_ch // 4
        self.rect_size = resolution * 4 -1
        self.rect_conv = nn.Sequential(
            nn.Conv2d(2, spatial_out_ch, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(spatial_out_ch, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(spatial_out_ch, spatial_out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(spatial_out_ch, momentum=0.01),
            nn.Flatten(),
            make_fc(spatial_out_ch*(resolution**2), out_dim, a=0)
        )

        # fusion
        self.fusion_fc = make_fc(out_dim, out_dim, a=0)

    def forward(self, features, boxes, rel_pairs, idx):
        """pool feature for boxes on one image
            features: dxhxw
            boxes: Nx4 (xyxy, nomalized to 0-1)
            rel_pairs: Nx2
        """
        # union boxes
        head_boxes = boxes[rel_pairs[:,0]]
        tail_boxes = boxes[rel_pairs[:,1]]
        union_boxes = torch.cat([
            torch.min(head_boxes[:,:2], tail_boxes[:,:2]),
            torch.max(head_boxes[:,2:], tail_boxes[:,2:])
        ], dim=1)

        # visual extractor
        h, w = (~features.mask[idx]).nonzero(as_tuple=False).max(dim=0)[0] + 1 # image area
        proj_feature = self.input_proj(features.tensors[idx:idx+1, :, :h, :w])
        # boxes
        scaled_union_boxes = torch.cat(
            [
                torch.zeros((len(union_boxes),1)).to(device=union_boxes.device),
                union_boxes * torch.tensor([w,h,w,h]).to(device=union_boxes.device, dtype=union_boxes.dtype).unsqueeze(0),
            ], dim=-1
        )
        union_visual_feats = roi_align(proj_feature, scaled_union_boxes, output_size=self.resolution, sampling_ratio=2)
        visual_feats = self.visual_proj(union_visual_feats.flatten(start_dim=1))

        # spatial extractor
        num_rel = len(rel_pairs)
        dummy_x_range = torch.arange(self.rect_size, device=head_boxes.device).view(1, 1, -1).expand(num_rel, self.rect_size, self.rect_size)
        dummy_y_range = torch.arange(self.rect_size, device=head_boxes.device).view(1, -1, 1).expand(num_rel, self.rect_size, self.rect_size)
        head_proposal = head_boxes * self.rect_size # resize bbox to the scale rect_size
        tail_proposal = tail_boxes * self.rect_size
        head_rect = ((dummy_x_range >= head_proposal[:,0].floor().view(-1,1,1).long()) & \
                        (dummy_x_range <= head_proposal[:,2].ceil().view(-1,1,1).long()) & \
                        (dummy_y_range >= head_proposal[:,1].floor().view(-1,1,1).long()) & \
                        (dummy_y_range <= head_proposal[:,3].ceil().view(-1,1,1).long())).float()
        tail_rect = ((dummy_x_range >= tail_proposal[:,0].floor().view(-1,1,1).long()) & \
                        (dummy_x_range <= tail_proposal[:,2].ceil().view(-1,1,1).long()) & \
                        (dummy_y_range >= tail_proposal[:,1].floor().view(-1,1,1).long()) & \
                        (dummy_y_range <= tail_proposal[:,3].ceil().view(-1,1,1).long())).float()
        rect_input = torch.stack((head_rect, tail_rect), dim=1) # (num_rel, 4, rect_size, rect_size)
        rect_features = self.rect_conv(rect_input)

        # fusion
        x = self.fusion_fc(visual_feats + rect_features)
        return x

