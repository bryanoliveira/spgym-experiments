# 1. Test setup:
# docker run -it --rm --gpus all pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime nvidia-smi
#
# If the above does not work, try adding the --privileged flag
# and changing the command to `sh -c 'ldconfig -v && nvidia-smi'`.
#
# 2. Build:
# sudo docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) --build-arg WANDB_KEY=<wandb_key> --build-arg HF_TOKEN=<hf_token> -f Dockerfile -t bryanlincoln/spgym_cleanrl:latest .
# sudo docker push bryanlincoln/spgym_cleanrl:latest

# 3. Run:
# sudo docker run -d --rm --gpus '"device=0"' --cpus=16.0 --memory=16g -v ./runs_spgym:/sliding-puzzle/runs -v ./dataset_spgym:/opt/conda/lib/python3.10/site-packages/sliding_puzzles/imgs bryanlincoln/spgym_cleanrl bash scripts/experiments/sac.sh

# 3. See results:
# tensorboard --logdir ~/logdir

# System
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco
ENV PYTHONUNBUFFERED 1
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PIP_NO_CACHE_DIR 1
RUN apt-get update && apt-get install -y \
    libglew2.1 libgl1-mesa-glx libosmesa6 \
    wget unrar cmake g++ libgl1-mesa-dev \
    libx11-6 openjdk-8-jdk x11-xserver-utils xvfb \
    bc git curl dos2unix \
    && apt-get clean
RUN pip3 install --upgrade pip
RUN git config --global --add safe.directory "*"

# Envs
ENV NUMBA_CACHE_DIR=/tmp

COPY requirements_spgym.txt requirements.txt
RUN pip3 install swig
RUN pip3 install -r requirements.txt

ARG GIT_COMMIT
ENV GIT_COMMIT=$GIT_COMMIT
ARG WANDB_KEY
ENV WANDB_API_KEY=$WANDB_KEY
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

RUN mkdir /sliding-puzzle
COPY . /sliding-puzzle
RUN chown -R 1000:root /sliding-puzzle && chmod -R 775 /sliding-puzzle
WORKDIR /sliding-puzzle

RUN dos2unix /sliding-puzzle/*.sh
RUN dos2unix /sliding-puzzle/scripts/experiments/*.sh

ENTRYPOINT ["/sliding-puzzle/entrypoint_dataset.sh"]
CMD ["bash", "scripts/experiments/sac.sh"]