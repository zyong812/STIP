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
from .transformer import TransformerDecoderLayer, TransformerDecoder
from hotr.util import box_ops
import numpy as np

# sequential visual relationship transformer
class VRTR(nn.Module):
    def __init__(self, args, detr, detr_matcher):
        super().__init__()
        self.args = args
        self.detr_matcher = detr_matcher
        backbone_out_ch = 2048

        # * Instance Transformer ---------------
        self.detr = detr
        if args.frozen_weights is not None:
            # if this flag is given, freeze the object detection related parameters of DETR
            for p in self.parameters():
                p.requires_grad_(False)
        # --------------------------------------

        # relation proposal
        rel_rep_dim = 1024
        self.union_box_feature_extractor = RelationFeatureExtractor(in_channels=backbone_out_ch, resolution=7, out_dim=rel_rep_dim)
        self.relation_proposal_mlp = nn.Sequential(
            make_fc(rel_rep_dim, rel_rep_dim // 2), nn.ReLU(),
            make_fc(rel_rep_dim // 2, 1)
        )

        # relation classification
        self.memory_input_proj = nn.Conv2d(backbone_out_ch, self.args.hidden_dim, kernel_size=1)
        self.rel_query_pre_proj = make_fc(rel_rep_dim, self.args.hidden_dim)

        decoder_layer = TransformerDecoderLayer(d_model=self.args.hidden_dim, nhead=self.args.hoi_nheads)
        decoder_norm = nn.LayerNorm(self.args.hidden_dim)
        self.interaction_decoder = TransformerDecoder(decoder_layer, self.args.hoi_dec_layers, decoder_norm, return_intermediate=True)
        self.action_embed = nn.Linear(self.args.hidden_dim, self.args.num_actions+1)

    def forward(self, samples: NestedTensor, targets=None):
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
        inst_repr = hs[-1] # instance representations
        num_nodes = inst_repr.shape[1]

        # Prediction Heads for Object Detection
        outputs_class = self.detr.class_embed(inst_repr)
        outputs_coord = self.detr.bbox_embed(inst_repr).sigmoid()
        # -----------------------------------------------
        det2gt_indices = None
        if self.training:
            detr_outs = {"pred_logits": outputs_class, "pred_boxes": outputs_coord}
            det2gt_indices = self.detr_matcher(detr_outs, targets)
            gt_rel_pairs = []
            for (ds, gs), t in zip(det2gt_indices, targets):
                gt2det_map = torch.zeros(len(gs)).to(device=ds.device, dtype=ds.dtype)
                gt2det_map[gs] = ds
                gt_rel_pairs.append(gt2det_map[t['relation_map'].sum(-1).nonzero(as_tuple=False)])
                if len(gt_rel_pairs[-1]) > self.args.num_hoi_queries: print(t['image_id'])

        # >>>>>>>>>>>> HOI DETECTION LAYERS <<<<<<<<<<<<<<<
        pred_rel_exists, pred_rel_pairs, pred_actions = [], [], []
        for imgid in range(bs):
            # >>>>>>>>>>>> relation proposal <<<<<<<<<<<<<<<
            inst_labels = outputs_class[imgid].max(-1)[-1]
            rel_mat = torch.zeros((num_nodes, num_nodes))
            rel_mat[inst_labels==1] = 1

            if self.training:
                rel_mat[gt_rel_pairs[imgid][:,0].unique()] = 1
                rel_mat[gt_rel_pairs[imgid][:,0], gt_rel_pairs[imgid][:, 1]] = 0
                rel_pairs = rel_mat.nonzero(as_tuple=False) # neg pairs

                if self.args.hard_negative_relation_sampling:
                    # hard negative sampling
                    all_pairs = torch.cat([gt_rel_pairs[imgid], rel_pairs], dim=0)
                    gt_pair_count = len(gt_rel_pairs[imgid])
                    all_rel_reps = self.union_box_feature_extractor(all_pairs, features[-1], outputs_coord[imgid], inst_repr[imgid], idx=imgid)
                    p_relation_exist_logits = self.relation_proposal_mlp(all_rel_reps).squeeze()

                    gt_inds = torch.arange(gt_pair_count).to(p_relation_exist_logits.device)
                    _, sort_rel_inds = p_relation_exist_logits.squeeze()[gt_pair_count:].sort(descending=True)
                    sampled_rel_inds = torch.cat([gt_inds, sort_rel_inds+gt_pair_count])[:self.args.num_hoi_queries]

                    sampled_rel_pairs = all_pairs[sampled_rel_inds]
                    sampled_rel_reps = all_rel_reps[sampled_rel_inds]
                    sampled_rel_pred_exists = p_relation_exist_logits[sampled_rel_inds]
                else:
                    # random sampling
                    sampled_neg_inds = torch.randperm(len(rel_pairs))
                    sampled_rel_pairs = torch.cat([gt_rel_pairs[imgid], rel_pairs[sampled_neg_inds]], dim=0)[:self.args.num_hoi_queries]
                    sampled_rel_reps = self.union_box_feature_extractor(sampled_rel_pairs, features[-1], outputs_coord[imgid], inst_repr[imgid], idx=imgid)
                    sampled_rel_pred_exists = self.relation_proposal_mlp(sampled_rel_reps).squeeze()
            else:
                rel_pairs = rel_mat.nonzero(as_tuple=False)
                if len(rel_pairs) == 0:
                    print('xxxx')
                    rel_pairs = (rel_mat == 0).nonzero(as_tuple=False) # todo: ??
                rel_reps = self.union_box_feature_extractor(rel_pairs, features[-1], outputs_coord[imgid], inst_repr[imgid], idx=imgid)
                p_relation_exist_logits = self.relation_proposal_mlp(rel_reps).squeeze()

                _, sort_rel_inds = p_relation_exist_logits.squeeze().sort(descending=True)
                sampled_rel_inds = sort_rel_inds[:self.args.num_hoi_queries] # todo: 可以调大试试

                sampled_rel_pairs = rel_pairs[sampled_rel_inds]
                sampled_rel_reps = rel_reps[sampled_rel_inds]
                sampled_rel_pred_exists = p_relation_exist_logits[sampled_rel_inds]

            # >>>>>>>>>>>> relation classification <<<<<<<<<<<<<<<
            outs = self.interaction_decoder(tgt=self.rel_query_pre_proj(sampled_rel_reps).unsqueeze(1),
                                            memory=self.memory_input_proj(src[imgid:imgid+1]).flatten(2).permute(2,0,1),
                                            memory_key_padding_mask=mask[imgid:imgid+1].flatten(1),
                                            pos=pos[-1][imgid:imgid+1].flatten(2).permute(2, 0, 1)) # todo: union mask, pos embedding etc.
            action_logits = self.action_embed(outs)

            pred_rel_pairs.append(sampled_rel_pairs)
            pred_actions.append(action_logits)
            pred_rel_exists.append(sampled_rel_pred_exists)

        pred_actions = torch.cat(pred_actions, dim=2).transpose(1,2)
        out = {
            "pred_logits": outputs_class,
            "pred_boxes": outputs_coord,
            "pred_rel_pairs": torch.stack(pred_rel_pairs, dim=0),
            "pred_actions": pred_actions[-1],
            "pred_action_exists": torch.stack(pred_rel_exists, dim=0),
            "det2gt_indices": det2gt_indices,
            "hoi_recognition_time": 0,
        }

        if self.args.hoi_aux_loss: # auxiliary loss
            out['hoi_aux_outputs'] = self._set_aux_loss_with_tgt(pred_actions)

        return out

    @torch.jit.unused
    def _set_aux_loss_with_tgt(self, outputs_action):
        return [{'pred_actions': x} for x in outputs_action[:-1]]

class VRTRCriterion(nn.Module):
    """ This class computes the loss for VRTR.
    1. proposal loss
    2. relation classification loss
    """
    def __init__(self, args, matcher):
        super().__init__()
        self.args = args
        self.matcher = matcher
        self.weight_dict = {
            'loss_proposal': 1,
            'loss_act': 1
        }
        if args.hoi_aux_loss:
            for i in range(args.hoi_dec_layers):
                self.weight_dict.update({f'loss_act_{i}': self.weight_dict['loss_act']})

        if args.dataset_file == 'vcoco':
            self.invalid_ids = args.invalid_ids
            self.valid_ids = np.concatenate((args.valid_ids,[-1]), axis=0) # no interaction

    def forward(self, outputs, targets, log=False):
        # instance matching
        if outputs['det2gt_indices'] is None:
            outputs_without_aux = {k: v for k, v in outputs.items() if (k != 'aux_outputs' and k != 'hoi_aux_outputs')}
            indices = self.matcher(outputs_without_aux, targets)
        else:
            indices = outputs['det2gt_indices']

        # generate relation targets
        all_rel_pair_targets = []
        for imgid, (tgt, (det_idxs, gtbox_idxs)) in enumerate(zip(targets, indices)):
            det2gt_map = {int(d): int(g) for d, g in zip(det_idxs, gtbox_idxs)}
            gt_relation_map = tgt['relation_map']
            rel_pairs = outputs['pred_rel_pairs'][imgid]
            rel_pair_targets = torch.zeros((len(rel_pairs), gt_relation_map.shape[-1])).to(gt_relation_map.device)
            for idx, rel in enumerate(rel_pairs):
                if (int(rel[0]) in det2gt_map) and (int(rel[1]) in det2gt_map):
                    rel_pair_targets[idx] = gt_relation_map[det2gt_map[int(rel[0])], det2gt_map[int(rel[1])]]
            all_rel_pair_targets.append(rel_pair_targets)
        all_rel_pair_targets = torch.stack(all_rel_pair_targets, dim=0)

        rel_proposal_targets = (all_rel_pair_targets[..., self.valid_ids].sum(-1) > 0).float()
        all_rel_pair_targets = torch.cat([all_rel_pair_targets, rel_proposal_targets.unsqueeze(-1)], dim=-1)

        # loss_proposal = F.binary_cross_entropy_with_logits(outputs['pred_action_exists'], rel_proposal_targets) # loss proposals
        loss_proposal = focal_loss(outputs['pred_action_exists'], rel_proposal_targets) # loss proposals
        loss_action = focal_loss(outputs['pred_actions'][..., self.valid_ids], all_rel_pair_targets[..., self.valid_ids]) # loss action classification

        loss_dict = {'loss_proposal': loss_proposal, 'loss_act': loss_action}
        if 'hoi_aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['hoi_aux_outputs']):
                aux_loss = {f'loss_act_{i}': focal_loss(aux_outputs['pred_actions'][..., self.valid_ids], all_rel_pair_targets[..., self.valid_ids])}
                loss_dict.update(aux_loss)

        return loss_dict

class VRTRPostProcess(nn.Module):
    def __init__(self, args, model):
        super().__init__()
        self.args = args

    @torch.no_grad()
    def forward(self, outputs, target_sizes, threshold=0, dataset='coco'):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # for relationship post-processing
        pair_actions = outputs['pred_actions'].sigmoid() * outputs['pred_action_exists'].sigmoid().unsqueeze(-1)
        h_indices = outputs['pred_rel_pairs'][:,:,0]
        o_indices = outputs['pred_rel_pairs'][:,:,1]

        # 和 HOTR 保持一致的排序方式 (todo: 我们方法该怎么更加合理排序)
        results = []
        for batch_idx, (s, l, b)  in enumerate(zip(scores, labels, boxes)):
            h_inds = (l == 1) & (s > threshold)
            o_inds = (s > threshold)

            h_box, h_cat = b[h_inds], s[h_inds]
            o_box, o_cat = b[o_inds], s[o_inds]

            # for scenario 1 in v-coco dataset
            o_inds = torch.cat((o_inds, torch.ones(1).type(torch.bool).to(o_inds.device)))
            o_box = torch.cat((o_box, torch.Tensor([0, 0, 0, 0]).unsqueeze(0).to(o_box.device))) # 增加一个空的 box

            result_dict = {
                'h_box': h_box, 'h_cat': h_cat,
                'o_box': o_box, 'o_cat': o_cat,
                'scores': s, 'labels': l, 'boxes': b
            }

            K = boxes.shape[1]
            n_act = pair_actions[batch_idx][:, :-1].shape[-1]
            score = torch.zeros((n_act, K, K+1)).to(pair_actions[batch_idx].device)
            sorted_score = torch.zeros((n_act, K, K+1)).to(pair_actions[batch_idx].device)
            id_score = torch.zeros((K, K+1)).to(pair_actions[batch_idx].device)

            # Score function: 所有 query 的结果加起来. 为什么要这么排序？
            for h_idx, o_idx, pair_action in zip(h_indices[batch_idx], o_indices[batch_idx], pair_actions[batch_idx]):
                matching_score = (1-pair_action[-1]) # no interaction score
                if h_idx == o_idx: o_idx = -1 # 特殊情况处理，主语和宾语相同
                if matching_score > id_score[h_idx, o_idx]:
                    id_score[h_idx, o_idx] = matching_score
                    sorted_score[:, h_idx, o_idx] = matching_score * pair_action[:-1]
                score[:, h_idx, o_idx] += matching_score * pair_action[:-1]

            score += sorted_score
            score = score[:, h_inds, :]
            score = score[:, :, o_inds]

            result_dict.update({
                'pair_score': score,
                'hoi_recognition_time': 0,
            })

            results.append(result_dict)

        return results


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
        self.visual_proj = make_fc(out_ch * (resolution**2), out_dim)

        # head & tail feature
        instr_hidden_dim = 256

        # spatial feature
        spatial_in_dim, spatial_out_dim = 8, 256
        self.spatial_proj = make_fc(spatial_in_dim, spatial_out_dim)

        # fusion
        self.fusion_fc = nn.Sequential(
            make_fc(out_dim+instr_hidden_dim*2+spatial_out_dim, out_dim), nn.ReLU(),
            make_fc(out_dim, out_dim), nn.ReLU()
        )

    def forward(self, rel_pairs, features, boxes, inst_reprs, idx):
        """pool feature for boxes on one image
            features: dxhxw
            boxes: Nx4 (cx_cy_wh, nomalized to 0-1)
            rel_pairs: Nx2
        """
        # union feature
        xyxy_boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        head_boxes = xyxy_boxes[rel_pairs[:, 0]]
        tail_boxes = xyxy_boxes[rel_pairs[:, 1]]
        union_boxes = torch.cat([
            torch.min(head_boxes[:,:2], tail_boxes[:,:2]),
            torch.max(head_boxes[:,2:], tail_boxes[:,2:])
        ], dim=1)

        # H, W = features.tensors.shape[-2:] # stacked image size
        h, w = (~features.mask[idx]).nonzero(as_tuple=False).max(dim=0)[0] + 1 # image area
        proj_feature = self.input_proj(features.tensors[idx:idx+1])
        scaled_union_boxes = torch.cat(
            [
                torch.zeros((len(union_boxes),1)).to(device=union_boxes.device),
                union_boxes * torch.tensor([w,h,w,h]).to(device=union_boxes.device, dtype=union_boxes.dtype).unsqueeze(0),
            ], dim=-1
        )
        union_visual_feats = roi_align(proj_feature, scaled_union_boxes, output_size=self.resolution, sampling_ratio=2)
        union_visual_feats = self.visual_proj(union_visual_feats.flatten(start_dim=1))

        # head & tail features
        head_feats = inst_reprs[rel_pairs[:,0]]
        tail_feats = inst_reprs[rel_pairs[:,1]]
        tail_feats[rel_pairs[:,0]==rel_pairs[:,1]] = 0 # set to 0 when head==tail (i.e., tail overlapped)

        # spatial layout feats
        box_layout_feats = self.extract_spatial_layout_feats(xyxy_boxes)
        rel_spatial_feats = self.spatial_proj(box_layout_feats[rel_pairs[:,0], rel_pairs[:,1]])

        relation_feats = torch.cat([union_visual_feats, head_feats, tail_feats, rel_spatial_feats], dim=-1)
        x = self.fusion_fc(relation_feats)
        return x

    def extract_spatial_layout_feats(self, xyxy_boxes):
        box_center = torch.stack([(xyxy_boxes[:, 0] + xyxy_boxes[:, 2]) / 2, (xyxy_boxes[:, 1] + xyxy_boxes[:, 3]) / 2], dim=1)
        dxdy = box_center.unsqueeze(1) - box_center.unsqueeze(0) # distances
        theta = (torch.atan2(dxdy[...,1], dxdy[...,0]) / np.pi).unsqueeze(-1)
        dis = dxdy.norm(dim=-1, keepdim=True)

        box_area = (xyxy_boxes[:, 2:] - xyxy_boxes[:, :2]).prod(dim=1) # areas
        intersec_lt = torch.max(xyxy_boxes.unsqueeze(1)[...,:2], xyxy_boxes.unsqueeze(0)[...,:2])
        intersec_rb = torch.min(xyxy_boxes.unsqueeze(1)[...,2:], xyxy_boxes.unsqueeze(0)[...,2:])
        overlap = (intersec_rb - intersec_lt).clamp(min=0).prod(dim=-1, keepdim=True)
        union_lt = torch.min(xyxy_boxes.unsqueeze(1)[...,:2], xyxy_boxes.unsqueeze(0)[...,:2])
        union_rb = torch.max(xyxy_boxes.unsqueeze(1)[...,2:], xyxy_boxes.unsqueeze(0)[...,2:])
        union = (union_rb - union_lt).clamp(min=0).prod(dim=-1, keepdim=True)
        spatial_feats = torch.cat([
            dxdy, dis, theta, # dx, dy, distance, theta
            overlap, union, box_area[:,None,None].expand(*union.shape), box_area[None,:,None].expand(*union.shape) # overlap, union, subj, obj
        ], dim=-1)
        return spatial_feats


def focal_loss(blogits, target_classes, alpha=0.25, gamma=2):
    probs = blogits.sigmoid() # prob(positive)
    loss_bce = F.binary_cross_entropy_with_logits(blogits, target_classes, reduction='none')
    p_t = probs * target_classes + (1 - probs) * (1 - target_classes)
    loss_bce = ((1-p_t)**gamma * loss_bce)

    alpha_t = alpha * target_classes + (1 - alpha) * (1 - target_classes)
    loss_focal = alpha_t * loss_bce

    loss = loss_focal.sum() / max(target_classes.sum(), 1)
    return loss

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
