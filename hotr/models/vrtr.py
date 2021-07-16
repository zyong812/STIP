# ------------------------------------------------------------------------
# HOTR official code : hotr/models/hotr.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import datetime

from hotr.util.misc import NestedTensor, nested_tensor_from_tensor_list
from .feed_forward import MLP

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
        hidden_dim = detr.transformer.d_model
        # --------------------------------------

        # relation proposal

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

        # Prediction Heads for Object Detection
        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        # -----------------------------------------------

        # >>>>>>>>>>>> HOI DETECTION LAYERS <<<<<<<<<<<<<<<
        # relation proposal

        # relation classification

        return


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
