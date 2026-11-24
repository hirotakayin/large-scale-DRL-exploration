#!/bin/bash
# One training run in the ariadne-train image, writing model/, train/ and gifs/<name>/ in this repo.
# usage: ./run_train.sh <run_name> [time_limit]     e.g. ./run_train.sh ariadne1_docker_test 10m
#   run_name    becomes FOLDER_NAME in parameter.py; must not exist yet under model/ (a new run overwrites it)
#   time_limit  timeout(1) duration (10m, 2h, ...); omit to run until Ctrl-C. Either way the driver gets SIGINT,
#               stops its workers and exits; it does not save on exit, so the last checkpoint is the last
#               multiple of 32 episodes.
# Start it inside tmux so it survives logging out:  tmux new -s train ./run_train.sh my_run 2h
# Build the image first:  docker build -t ariadne-train:v1 .
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
name=${1:?usage: run_train.sh <run_name> [time_limit]}
limit=${2:-0}  # 0 = no limit
image=${IMAGE:-ariadne-train:v1}
if [ -e "model/$name" ] || [ -e "train/$name" ]; then
    echo "model/$name or train/$name already exists: back it up to ~/rl/runs/ or pick another name" >&2; exit 1
fi
if docker ps -q --filter name=^ariadne-train$ | grep -q .; then
    echo "a container named ariadne-train is already running" >&2; exit 1
fi
echo "run $name, image $image, limit $limit, log train_$name.log"
timeout -s INT "$limit" docker run --rm --init --gpus all --shm-size=16g --name ariadne-train \
    --user "$(id -u):$(id -g)" -e HOME=/tmp/home -e FOLDER_NAME="$name" \
    -v "$PWD":/workspace -w /workspace \
    "$image" python -u driver.py 2>&1 | tee "train_$name.log"
