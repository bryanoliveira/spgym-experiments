#!/bin/bash
for seed in 1 2 3 4 5
do
    for i in 1 3 5
    do
        python train_sac.py --env-configs "{\"w\": 4, \"image_pool_size\": $i}"
    done
done