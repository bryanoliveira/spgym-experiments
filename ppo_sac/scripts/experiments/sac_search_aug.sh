#!/bin/bash
for seed in 1 2 3
do
    for aug in "crop" "shift" "channel_shuffle" "grayscale" "inversion" "grayscale,channel_shuffle" "grayscale,inversion,channel_shuffle"
    do
        # data augmentation (rad)
        python train_sac.py --env-configs "{\"image_pool_size\": 5}" \
        --independent-encoder --encoder-tau 0.025 \
        --apply-data-augmentation --augmentations $aug
    done
done