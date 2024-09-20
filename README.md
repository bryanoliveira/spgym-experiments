# Sliding Puzzles Gym: A Scalable Benchmark for State Representation in Visual Reinforcement Learning

This repository is the official implementation of Sliding Puzzles Gym: A Scalable Benchmark for State Representation in Visual Reinforcement Learning. 

## Requirements

1. Setup the benchmark

Download the ImageNet-1k validation split from this URL:
https://huggingface.co/datasets/ILSVRC/imagenet-1k/blob/main/data/val_images.tar.gz

Extract the images to `spgym/imgs/imagenet-1k`

2. Prepare Python environments

Create separate Python environments for Dreamer and PPO. `requirements.txt` can be found inside each correspondent folder, and Dockerfiles are also available. See `dreamer/README.md` and `ppo/README.md` for more details.

## Training

Scripts for reproducing the paper results can be found in `dreamer/scripts` and `ppo/scripts`. Seeds are generated automatically by the algorithms if not specified. For example, inside `dreamer` or `ppo` folders, run:

```train
bash scripts/scale.sh
```

## Results

Results can be found in our paper.