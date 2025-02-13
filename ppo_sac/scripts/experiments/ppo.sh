#!/bin/bash
for i in 1 5 10 20 30 50 100
do
    for seed in 1 2 3 4 5
    do
        python ppo.py \
        --env_configs "{\"w\": 4, \"image_pool_size\": $i}"
done
