#!/bin/bash
for pool_size in 1 5 10 20 30 50 100 500; do
    for seed in 1 2 3 4 5; do
        logdir="./logdir/$(date "+%Y%m%d_%H%M%S")-dreamer_w3_imagenet_p${pool_size}"
        echo "logdir: $logdir"
        mkdir -p "$logdir"
        python dreamerv3/main.py --logdir "$logdir" \
            --configs sldp_image \
            --env.sldp.image_pool_size ${pool_size} \
            2>&1 | tee "$logdir/run.log"
    done
done
