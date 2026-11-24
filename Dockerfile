# Training environment for large-scale-DRL-exploration (ARiADNE, ground-truth critic).
# The repo is mounted at /workspace at run time (see run_train.sh), so code edits need no rebuild;
# rebuild only when requirements.txt changes. torch from PyPI ships its own CUDA libraries: the host
# needs only the NVIDIA driver and the container toolkit (docker run --gpus all).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt
# run_train.sh runs as the host user (so outputs are not owned by root) with HOME=/tmp/home
RUN mkdir -m 1777 /tmp/home
WORKDIR /workspace
