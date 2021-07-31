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
from hotr.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

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

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes, threshold=thr, dataset='hico-det')
        hoi_recognition_time.append(results[0]['hoi_recognition_time'] * 1000)

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

        # check_res(samples, targets, type='annotation', rel_num=20)
        # check_res(samples, results, type='prediction', rel_num=20)

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


# visualize results
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
def check_res(samples, res, type='annotation', rel_num=20):
    img_tensors, img_masks = samples.decompose()
    h, w = (img_masks[0].float() < 1).nonzero().max(0)[0].cpu() + 1

    img_tensor = img_tensors[0,:,:h,:w].cpu().permute(1,2,0)
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())

    res = res[0]
    org_h, org_w = res['orig_size'].cpu().float()
    boxes = res['boxes'].cpu()
    boxes = boxes * torch.tensor([w/org_w, h/org_h, w/org_w, h/org_h]).unsqueeze(0)
    vg_obj_names = []
    for ind, x in enumerate(res['labels']):
        # vg_obj_names.append(f"{dataset.ind_to_classes[x]}({ind})")
        vg_obj_names.append(f"class{x}({ind})")

    rel_pairs = []
    if type == 'annotation':
        res = res[0]
        org_h, org_w = res['orig_size'].cpu().float()
        boxes = res['boxes'].cpu()
        boxes = boxes * torch.tensor([w/org_w, h/org_h, w/org_w, h/org_h]).unsqueeze(0)
        vg_obj_names = []
        for ind, x in enumerate(res['labels']):
            # vg_obj_names.append(f"{dataset.ind_to_classes[x]}({ind})")
            vg_obj_names.append(f"class{x}({ind})")

        rel_pairs = res['hois'][:rel_num, :2].cpu()
        rel_labels = res['hois'][:rel_num, 2].cpu()
    elif type == 'prediction':
        rel_pairs = torch.stack([res['sub_ids'], res['obj_ids']], dim=0).cpu()
        rel_labels = res['hois'][:rel_num, 2].cpu()

    # list relations
    rel_strs = ''
    for i, rel in enumerate(rel_pairs): # print relation triplets
        rel_strs += (f"{vg_obj_names[rel[0]]} ---{rel_labels[i]}----> {vg_obj_names[rel[1]]}\n")

    # draw images
    plt.imshow(img_tensor)
    for ind, bbox in enumerate(boxes):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1,y1), x2-x1+1, y2-y1+1, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        txt = plt.text(x1-10, y1-10, vg_obj_names[ind], color='black')
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    plt.title(type)
    plt.gca().yaxis.set_label_position("right")
    plt.ylabel(rel_strs, rotation=0, labelpad=140, fontsize=8, loc='top')
    plt.show()
