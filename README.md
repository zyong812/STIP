## [CVPR 2022] Exploring Structure-aware Transformer over Interaction Proposals for Human-object Interaction Detection

* [PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Exploring_Structure-Aware_Transformer_Over_Interaction_Proposals_for_Human-Object_Interaction_Detection_CVPR_2022_paper.pdf)
* [Supplementary](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Zhang_Exploring_Structure-Aware_Transformer_CVPR_2022_supplemental.pdf)

## Paper introduction

Recent high-performing Human-Object Interaction (HOI) detection techniques have been highly influenced by Transformer-based object detector (i.e., DETR). Nevertheless, most of them directly map parametric interaction queries into a set of HOI predictions through vanilla Transformer in a one-stage manner. This leaves rich inter- or intra-interaction structure under-exploited. In this work, we design a novel Transformer-style HOI detector, i.e., Structure-aware Transformer over Interaction Proposals (STIP), for HOI detection. Such design decomposes the process of HOI set prediction into two subsequent phases, i.e., an interaction proposal generation is first performed, and then followed by transforming the non-parametric interaction proposals into HOI predictions via a structure-aware Transformer. The structure-aware Transformer upgrades vanilla Transformer by encoding additionally the holistically semantic structure among interaction proposals as well as the locally spatial structure of human/object within each interaction proposal, so as to strengthen HOI predictions. Extensive experiments conducted on V-COCO and HICO-DET benchmarks have demonstrated the effectiveness of STIP, and superior results are reported when comparing with the state-of-the-art HOI detectors.

<p align="center"><img width="80%" src="./imgs/STIP.png"></p>

## Installation

### 1. Environmental Setup
```bash
$ conda create -n STIP python=3.7
$ conda install -c pytorch pytorch torchvision # PyTorch 1.7.1, torchvision 0.8.2, CUDA=11.0
$ conda install cython scipy
$ pip install pycocotools
$ pip install opencv-python
$ pip install wandb
```

### 2. HOI dataset setup
Our current version supports the experiments for both [V-COCO](https://github.com/s-gupta/v-coco) and [HICO-DET](https://drive.google.com/file/d/1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk/view) dataset.
Download the dataset under the pulled directory.
For HICO-DET, we use the [annotation files](https://drive.google.com/file/d/1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk/view) provided by the PPDM authors.
Download the [list of actions](https://drive.google.com/open?id=1EeHNHuYyJI-qqDk_-5nay7Mb07tzZLsl) as `list_action.txt` and place them under the unballed hico-det directory.
Below we present how you should place the files.
```bash
# V-COCO setup
$ git clone https://github.com/s-gupta/v-coco.git
$ cd v-coco
$ ln -s [:COCO_DIR] coco/images # COCO_DIR contains images of train2014 & val2014
$ python script_pick_annotations.py [:COCO_DIR]/annotations

# HICO-DET setup
$ tar -zxvf hico_20160224_det.tar.gz # move the unballed folder under the pulled repository

# dataset setup
STIP
 │─ v-coco
 │   │─ data
 │   │   │─ instances_vcoco_all_2014.json
 │   │   :
 │   └─ coco
 │       │─ images
 │       │   │─ train2014
 │       │   │   │─ COCO_train2014_000000000009.jpg
 │       │   │   :
 │       │   └─ val2014
 │       │       │─ COCO_val2014_000000000042.jpg
 :       :       :
 │─ hico_20160224_det
 │       │─ list_action.txt
 │       │─ annotations
 │       │   │─ trainval_hico.json
 │       │   │─ test_hico.json
 │       │   └─ corre_hico.npy
 :       :
```

If you wish to download the datasets on our own directory, simply change the 'data_path' argument to the directory you have downloaded the datasets.
```bash
--data_path [:your_own_directory]/[v-coco/hico_20160224_det]
``` 

### 3. Training/Testing on V-COCO

```shell
python STIP_main.py --validate \
    --num_hoi_queries 32 --batch_size 4 --lr 5e-5 --HOIDet --hoi_aux_loss --no_aux_loss \ 
    --dataset_file vcoco  --data_path v-coco --detr_weights https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --output_dir checkpoints/vcoco --group_name STIP_debug --run_name vcoco_run1
```

* Add `--eval` option for evaluation

### 4. Training/Testing on HICO-DET

Training with pretrained DETR detector on COCO.
```shell
python STIP_main.py --validate \ 
    --num_hoi_queries 32 --batch_size 4 --lr 5e-5 --HOIDet --hoi_aux_loss --no_aux_loss \
    --dataset_file hico-det --data_path hico_20160224_det --detr_weights https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --output_dir checkpoints/hico-det --group_name STIP_debug --run_name hicodet_run1
```

Jointly fine-tune object detector & HOI detector
```shell
python STIP_main.py --validate \
    --num_hoi_queries 32 --batch_size 2 --lr 1e-5 --HOIDet --hoi_aux_loss \
    --dataset_file hico-det --data_path hico_20160224_det \
    --output_dir checkpoints/hico-det --group_name STIP_debug --run_name hicodet_run1/jointly-tune \
    --resume checkpoints/hico-det/STIP_debug/best.pth --train_detr
```

## Citation

If you find this code helpful for your research, please cite our paper.
```
@InProceedings{Zhang_2022_CVPR,
    author    = {Zhang, Yong and Pan, Yingwei and Yao, Ting and Huang, Rui and Mei, Tao and Chen, Chang-Wen},
    title     = {Exploring Structure-Aware Transformer Over Interaction Proposals for Human-Object Interaction Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {19548-19557}
}
```

## Others

Credit: [DETR](https://github.com/facebookresearch/detr), [HOTR](https://github.com/kakaobrain/HOTR)
