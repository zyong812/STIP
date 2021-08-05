import math
import os
import sys
from typing import Iterable
import numpy as np
import copy
import itertools

import torch

import hotr.util.misc as utils
import hotr.util.logger as loggers
from hotr.data.evaluators.hico_eval import HICOEvaluator
from hotr.models.vrtr_utils import check_annotation, plot_cross_attention

@torch.no_grad()
def hico_evaluate(model, postprocessors, data_loader, device, thr):
    model.eval()

    metric_logger = loggers.MetricLogger(mode="test", delimiter="  ")
    header = 'Evaluation Inference (HICO-DET)'

    preds = []
    gts = []
    indices = []
    hoi_recognition_time = []

    for samples, targets in metric_logger.log_every(data_loader, 50, header):
        samples = samples.to(device)
        targets = [{k: (v.to(device) if k != 'id' else v) for k, v in t.items()} for t in targets]

        # dec_selfattn_weights, dec_crossattn_weights = [], []
        # hook = model.interaction_decoder.layers[-1].multihead_attn.register_forward_hook(lambda self, input, output: dec_crossattn_weights.append(output[1]))

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes, threshold=thr, dataset='hico-det')
        hoi_recognition_time.append(results[0]['hoi_recognition_time'] * 1000)

        # check_annotation(samples, targets, mode='eval', rel_num=20)
        # plot_cross_attention(samples, outputs, targets, dec_crossattn_weights); hook.remove()
        # if targets[0]['image_id'].item() in [48, 88]:
        #     print('xxx')

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))


    print(f"[stats] HOI Recognition Time (avg) : {sum(hoi_recognition_time)/len(hoi_recognition_time):.4f} ms")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]
    
    evaluator = HICOEvaluator(preds, gts, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat)

    stats = evaluator.evaluate()

    return stats

