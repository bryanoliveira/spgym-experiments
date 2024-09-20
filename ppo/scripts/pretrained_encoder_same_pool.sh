#!/bin/bash

# Scan the runs folder for experiments matching the pattern
for experiment in runs/[0-9]*-ppo_w3_*; do
    if [ -d "$experiment" ]; then
        echo "Found experiment: $experiment"
        
        # Extract the datetime and experiment name
        datetime=$(basename "$experiment" | cut -d'-' -f1)
        exp_name=$(basename "$experiment" | cut -d'-' -f2-)
        
        echo "  Datetime: $datetime"
        echo "  Experiment name: $exp_name"
        
        # Extract image_pool_size and seed from the experiment name
        if [[ $exp_name =~ ppo_w3_image_imagenet1k_p([0-9]+)_([0-9]+)$ ]]; then
            image_pool_size="${BASH_REMATCH[1]}"
            seed="${BASH_REMATCH[2]}"
            echo "  Image pool size: $image_pool_size"
            echo "  Seed: $seed"
        else
            echo "  Image pool size or seed not found in experiment name"
            exit 1
        fi

        # Find the latest checkpoint
        latest_checkpoint=$(find "$experiment" -name "checkpoint_*.pth" | sort -V | tail -n 1)
        
        if [ -n "$latest_checkpoint" ]; then
            echo "  Latest checkpoint: $latest_checkpoint"
            checkpoint_step=$(basename "$latest_checkpoint" | sed 's/checkpoint_\(.*\)\.pth/\1/')
            echo "  Checkpoint step: $checkpoint_step"
        else
            echo "  No checkpoint found"
            exit 1
        fi

        command="python ppo.py \
            --exp_name \"ppo_senc\" \
            --env_configs '{\"w\": 3, \"variation\": \"image\", \"image_folder\": \"imagenet-1k\", \"image_pool_size\": $image_pool_size, \"image_pool_seed\": $seed}' \
            --checkpoint-load-path \"$latest_checkpoint\" \
            --checkpoint-param-filter \"^encoder\.\"
        "

        echo "Executing command:"
        echo "$command"
        eval "$command"

        echo "------------------------"
    fi
done

echo "Scan complete."
