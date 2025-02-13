#!/bin/bash

for seed in 1 2 3 4 5; do
    for env in \
        "dm_control/cartpole-swingup-v0" \
        "dm_control/cheetah-run-v0" \
        "dm_control/finger-spin-v0" \
        "dm_control/reacher-easy-v0" \
        "dm_control/walker-walk-v0" \
        "dm_control/hopper-hop-v0"; do
        python sac_continuous_torchcompile.py --env_id ${env} --seed ${seed}
    done
done