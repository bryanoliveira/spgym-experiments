#!/bin/bash
for seed in 1 # 2 3
do
    for i in 1 # 5 10 20
    do
        # # baseline
        # python train_sac.py --track True --total-timesteps 10000000 \
        # --env-configs "{\"image_pool_size\": $i}" --num-envs 512 --async-envs True

        # # reward & transition model
        # python train_sac.py --track True --total-timesteps 10000000 \
        # --env-configs "{\"image_pool_size\": $i}" --num-envs 512 --async-envs True
        # --repr-loss-coef 1 --use-reward-loss True --use-transition-loss True

        # reconstruction (Ghosh et al. (2019))
        python train_sac.py --track True --total-timesteps 10000000 \
        --env-configs "{\"image_pool_size\": $i}" --num-envs 512 --async-envs True \
        --repr-loss-coef 1 --use-reconstruction-loss True --decoder-decay-weight 1e-5 --repr-decay-weight 1e-6

        # variational reconstruction
        python train_sac.py --track True --total-timesteps 10000000 \
        --env-configs "{\"image_pool_size\": $i}" --num-envs 512 --async-envs True \
        --repr-loss-coef 1 --use-reconstruction-loss True --variational-reconstruction True

        # contrastive augmented
        python train_sac.py --track True --total-timesteps 10000000 \
        --env-configs "{\"image_pool_size\": $i}" --num-envs 512 --async-envs True \
        --use-curl-loss True --contrastive-positives augmented --augmentations "channel_shuffle,brightness,color_jitter,inversion,grayscale" \
        --repr-loss-coef 1 --target-network-frequency 2 --tau 0.01 --encoder-tau 0.001 # + curl hypers

        # # contrastive temporal
        # python train_sac.py --track True --total-timesteps 10000000 \
        # --env-configs "{\"image_pool_size\": $i}" --num-envs 512 --async-envs True \
        # --repr-loss-coef 1 --use-curl-loss True --contrastive-positives temporal --target-network-frequency 2 --tau 0.01 --encoder-tau 0.001   # + curl hypers

        # data augmentation (rad?)
        python train_sac.py --track True --total-timesteps 10000000 \
        --env-configs "{\"image_pool_size\": $i}" --num-envs 512 --async-envs True \
        --repr-loss-coef 1 --apply-data-augmentation True

        # data augmentation loss
        python train_sac.py --track True --total-timesteps 10000000 \
        --env-configs "{\"image_pool_size\": $i}" --num-envs 512 --async-envs True \
        --repr-loss-coef 1 --use-data-augmentation-loss True

        # # state metric
        # python train_sac.py --track True --total-timesteps 10000000 \
        # --env-configs "{\"image_pool_size\": $i}" --num-envs 512 --async-envs True \
        # --repr-loss-coef 1 --use-dbc-loss True

        # # self supervised (augmentation)
        python train_sac.py --track True --total-timesteps 10000000 \
        --env-configs "{\"image_pool_size\": $i}" --num-envs 512 --async-envs True \
        --repr-loss-coef 1 --use-spr-loss True --apply-data-augmentation True --batch-size 3500

        # # self supervised (dropout)
        # python train_sac.py --track True --total-timesteps 10000000 \
        # --env-configs "{\"image_pool_size\": $i}" --num-envs 512 --async-envs True \
        # --repr-loss-coef 1 --use-spr-loss True --encoder-dropout 0.5 --batch-size 3500
    done
done