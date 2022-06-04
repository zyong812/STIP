import numpy as np
import copy
import itertools

import torch

import src.util.misc as utils
import src.util.logger as loggers
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# evaluate detection on HICO-DET
@torch.no_grad()
def hico_det_evaluate(model, postprocessors, data_loader, device, args):
    hico_valid_obj_ids = torch.tensor(args.valid_obj_ids) # id -> coco id

    model.eval()

    metric_logger = loggers.MetricLogger(mode="test", delimiter="  ")
    header = 'Evaluation Inference (HICO-DET)'

    preds = []
    gts = []

    all_predictions = {}
    for samples, targets in metric_logger.log_every(data_loader, 50, header):
        samples = samples.to(device)
        targets = [{k: (v.to(device) if k != 'id' else v) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, dataset='hico-det')

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = {i: img_preds for i, img_preds in enumerate(preds) if i in indices}
    gts = {i: img_gts for i, img_gts in enumerate(gts) if i in indices}

    stats = do_detection_evaluation(preds, gts, hico_valid_obj_ids)
    return stats

def do_detection_evaluation(predictions, groundtruths, hico_valid_obj_ids):
    # create a Coco-like object that we can use to evaluate detection!
    anns = []
    for image_id, gt in groundtruths.items():
        labels = gt['labels'].tolist() # map to coco like ids
        boxes = gt['boxes'].tolist() # xyxy
        for cls, box in zip(labels, boxes):
            anns.append({
                'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1], # xywh
                'category_id': cls,
                'id': len(anns),
                'image_id': image_id,
                'iscrowd': 0,
            })
    fauxcoco = COCO()
    fauxcoco.dataset = {
        'info': {'description': 'use coco script for vg detection evaluation'},
        'images': [{'id': i} for i in range(len(groundtruths))],
        'categories': [
            {'supercategory': 'person', 'id': i, 'name': str(i)} for i in hico_valid_obj_ids.tolist()
        ],
        'annotations': anns,
    }
    fauxcoco.createIndex()

    # format predictions to coco-like
    cocolike_predictions = []
    for image_id, prediction in predictions.items():
        box = torch.stack((prediction['boxes'][:,0], prediction['boxes'][:,1], prediction['boxes'][:,2]-prediction['boxes'][:,0]+1, prediction['boxes'][:,3]-prediction['boxes'][:,1]+1), dim=1).detach().cpu().numpy() # xywh
        label = prediction['labels'].cpu().numpy() # (#objs,)
        score = prediction['scores'].cpu().numpy() # (#objs,)

        image_id = np.asarray([image_id]*len(box))
        cocolike_predictions.append(
            np.column_stack((image_id, box, score, label))
        )
    cocolike_predictions = np.concatenate(cocolike_predictions, 0)
    # evaluate via coco API
    res = fauxcoco.loadRes(cocolike_predictions)
    coco_eval = COCOeval(fauxcoco, res, 'bbox')
    coco_eval.params.imgIds = list(range(len(groundtruths)))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mAp = coco_eval.stats[1]

    return mAp