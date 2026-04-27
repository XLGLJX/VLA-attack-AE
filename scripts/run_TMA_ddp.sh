#!/bin/bash
current_dir=$(pwd)
echo $current_dir
torchrun --nproc_per_node=2 --master_port=29501 VLAAttacker/TMA_wrapper_ddp.py \
    --maskidx 0 \
    --lr 2e-3 \
    --server $current_dir \
    --device 0 \
    --iter 2000 \
    --accumulate 4 \
    --bs 2 \
    --warmup 20 \
    --tags "debug testrun" \
    --filterGripTrainTo1 false \
    --geometry true \
    --patch_size "3,50,50" \
    --swanlab_project "VLA-Attack" \
    --innerLoop 50 \
    --dataset "libero_spatial_no_noops" \
    --targetAction 0
