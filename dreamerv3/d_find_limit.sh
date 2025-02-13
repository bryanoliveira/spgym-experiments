#!/bin/bash
for pool_size in 1 2 3 4 5 6 7 8 9 10; do
    python dreamerv3/main.py --logdir ./logdir/$(date "+%Y%m%d_%H%M%S")-sldp_imagenet_w3_p${pool_size} \
        --configs sldp_image \
        --env.sldp.w 3 \
        --env.sldp.image_folder imagenet-1k \
        --env.sldp.image_pool_size ${pool_size} \
        --run.steps 10000000
done