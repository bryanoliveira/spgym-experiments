#!/bin/bash
for pool_size in 1 3 5 7 10; do
    for seed in 1 2 3 4 5; do
        python dreamerv3/main.py --logdir ./logdir/$(date "+%Y%m%d_%H%M%S")-dreamer_nodec_w3_imagenet_p${pool_size} \
            --configs sldp_image_nodec \
            --env.sldp.image_pool_size ${pool_size}
    done
done