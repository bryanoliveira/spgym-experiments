#!/bin/bash
for pool_size in 1 3 5; do
    for seed in 1 2; do
        logdir="./logdir/$(date "+%Y%m%d_%H%M%S")-dreamer_w4_imagenet_p${pool_size}"
        echo "logdir: $logdir"
        mkdir -p "$logdir"
        python dreamerv3/main.py --logdir "$logdir" \
            --configs sldp_image \
            --env.sldp.w 4 \
            --env.sldp.image_pool_size ${pool_size} \
            2>&1 | tee "$logdir/run.log"
    done
done
