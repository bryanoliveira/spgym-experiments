#!/bin/bash
for seed in 1 2 3 4 5
do
    for i in 1 5 10
    do
        # baseline
        python train_sac.py --env-configs "{\"image_pool_size\": $i}"

        # reward & transition model (tomar 2021)
        python train_sac.py --env-configs "{\"image_pool_size\": $i}" \
        --independent-encoder --encoder-tau 0.025 \
        --use-reward-loss --use-transition-loss

        # reconstruction (sac-ae - Ghosh et al. (2019))
        python train_sac.py --env-configs "{\"image_pool_size\": $i}" \
        --independent-encoder --encoder-tau 0.025 \
        --use-reconstruction-loss --decoder-decay-weight 1e-7 --repr-decay-weight 1e-6

        # variational reconstruction (sac-vae)
        python train_sac.py --env-configs "{\"image_pool_size\": $i}" \
        --independent-encoder --encoder-tau 0.025 \
        --use-reconstruction-loss --variational-reconstruction

        # contrastive augmented
        python train_sac.py --env-configs "{\"image_pool_size\": $i}" \
        --independent-encoder --encoder-tau 0.025 \
        --use-curl-loss --contrastive-positives augmented

        # # contrastive temporal
        # python train_sac.py --env-configs "{\"image_pool_size\": $i}" \
        # --independent-encoder --encoder-tau 0.025 \
        # --use-curl-loss --contrastive-positives temporal

        # data augmentation (rad)
        python train_sac.py --env-configs "{\"image_pool_size\": $i}" \
        --independent-encoder --encoder-tau 0.025 \
        --apply-data-augmentation

        # # data augmentation loss
        # python train_sac.py --env-configs "{\"image_pool_size\": $i}" \
        # --independent-encoder --encoder-tau 0.025 \
        # --use-data-augmentation-loss

        # state metric (dbc)
        python train_sac.py --env-configs "{\"image_pool_size\": $i}" \
        --independent-encoder --encoder-tau 0.025 \
        --use-dbc-loss

        # self supervised (augmentation)
        python train_sac.py --env-configs "{\"image_pool_size\": $i}" \
        --independent-encoder --encoder-tau 0.025 \
        --use-spr-loss --apply-data-augmentation

        # # self supervised (dropout)
        # python train_sac.py --env-configs "{\"image_pool_size\": $i}" \
        # --independent-encoder --encoder-tau 0.025 \
        # --use-spr-loss --encoder-dropout 0.5
    done
done

for seed in 1 2 3 4 5
do
    for i in 20 30 50 100
    do
        python train_sac.py --env-configs "{\"image_pool_size\": $i}"
    done
done