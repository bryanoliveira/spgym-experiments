#!/bin/bash
for pool_size in 1 3 5 7 10 15 20 30 50 100 500
do
    for seed in 1 2 3 4 5
    do
        python ppo.py \
        --env_configs "{\"w\": 3, \"variation\": \"image\", \"image_folder\": \"imagenet-1k\", \"image_pool_size\": ${pool_size}}"
    done
done