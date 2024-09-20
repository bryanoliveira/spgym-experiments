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
    bc \
    && apt-get clean
RUN pip3 install --upgrade pip

# Envs
ENV NUMBA_CACHE_DIR=/tmp

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip install "gymnasium[atari, accept-rom-license]"

ARG GIT_COMMIT
ENV GIT_COMMIT=$GIT_COMMIT
ARG WANDB_KEY
ENV WANDB_API_KEY=$WANDB_KEY

RUN mkdir /sliding-puzzle
COPY . /sliding-puzzle
RUN chown -R 1000:root /sliding-puzzle && chmod -R 775 /sliding-puzzle
WORKDIR /sliding-puzzle

RUN chmod +x /sliding-puzzle/entrypoint.sh
ENTRYPOINT ["/sliding-puzzle/entrypoint.sh"]
CMD ["bash", "scripts/scale.sh"]