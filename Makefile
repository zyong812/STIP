# [HICO-DET] multi-gpu train (runs in 1 GPU)
# 1-1. Training with pretrained DETR detector on COCO.
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env STIP_main.py \
    --validate \
    --num_hoi_queries 32 --batch_size 8 --lr 2e-4 --HOIDet --hoi_aux_loss --no_aux_loss \
    --dataset_file hico-det --data_path /dengkunyuan/Data/HICO-DET/hico_20160224_det \
    --detr_weights https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --output_dir checkpoints/hico-det --group_name 2stage_pre-trained_on_MS-COCO --run_name 20241220200000

# 1-2. Training with pretrained DETR detector fine-tuned on HICO-DET.
# 这里使用HOTR提供的hico_ft_q16.pth作为frozen DETR weights，和使用STIP提供的hico-det_928a85d_best_noFT.pth作为frozen DETR weights的效果应该是类似的，只需要修改STIP_main.py中的if 'hico_ft_q16.pth' in args这一行代码。
# 它们的区别在于前者在HICO-DET上进行了fine-tune然后再冻结进行了HOTR训练，而后者不能保证在HICO-DET上进行了fine-tune（因为STIP的README中只提到它没有联合微调）。所以优先使用前者。
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env STIP_main.py \
    --validate \
    --num_hoi_queries 32 --batch_size 8 --lr 2e-4  --HOIDet --hoi_aux_loss --no_aux_loss \
    --dataset_file hico-det --data_path /dengkunyuan/Data/HICO-DET/hico_20160224_det \
    --detr_weights /dengkunyuan/Project/IL-HOID/Analytic-continual-learning-main/STIP/checkpoints/hico_ft_q16.pth \
    --output_dir checkpoints/hico-det --group_name 2stage_fine-tuned_on_HICO-DET --run_name 20241220210000

# 2. Jointly fine-tune object detector & HOI detector
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env STIP_main.py \
    --validate \
    --num_hoi_queries 32 --batch_size 2 --lr 1e-5 --HOIDet --hoi_aux_loss \
    --dataset_file hico-det --data_path /dengkunyuan/Data/HICO-DET/hico_20160224_det \
    --output_dir checkpoints/hico-det --group_name Jointly_fine-tune --run_name hicodet_run1/jointly-tune \
    --resume checkpoints/hico-det/STIP_debug/best.pth --train_detr

# [HICO-DET] single-gpu test: original batch size is 4
python STIP_main.py --eval \
    --num_hoi_queries 32 --batch_size 8 --lr 5e-5 --HOIDet --hoi_aux_loss --no_aux_loss \
    --dataset_file hico-det --data_path /dengkunyuan/Data/HICO-DET/hico_20160224_det \
    --resume /dengkunyuan/Project/IL-HOID/Analytic-continual-learning-main/STIP/checkpoints/hico-det_928a85d_best_noFT.pth \
    --output_dir checkpoints/hico-det --group_name raw --run_name test