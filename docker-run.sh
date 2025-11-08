#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Docker run script for Cosmos-Transfer2.5
# This script runs the Docker container with proper GPU support and volume mounts

set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if Docker image exists
if ! docker image inspect cosmos-transfer2.5:latest > /dev/null 2>&1; then
    echo "Docker image 'cosmos-transfer2.5:latest' not found. Building it now..."
    ./docker-build.sh
fi

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Check if NVIDIA GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. GPU support may not work."
    GPU_FLAG=""
else
    # Allow specifying GPU via CUDA_VISIBLE_DEVICES or use all GPUs by default
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        GPU_FLAG="--gpus '\"device=$CUDA_VISIBLE_DEVICES\"'"
        echo "NVIDIA GPU detected. Using GPU(s): $CUDA_VISIBLE_DEVICES"
    else
        GPU_FLAG="--gpus all"
        echo "NVIDIA GPU detected. Using --gpus all (use CUDA_VISIBLE_DEVICES=1 to use GPU 1, etc.)"
    fi
fi

# Run the Docker container
# -v $(pwd):/workspace: Mount current directory to /workspace
# -v cosmos-transfer2.5-venv:/workspace/.venv: Use a named volume for .venv to persist it
# --rm: Remove container when it exits
# -it: Interactive terminal (only if TTY is available)
echo "Starting Docker container..."
echo "Current directory will be mounted to /workspace in the container"
if [ ! -d "$HOME/.cache/huggingface" ]; then
    echo ""
    echo "⚠️  Warning: Hugging Face cache not found at $HOME/.cache/huggingface"
    echo "   You may need to log in to Hugging Face first."
    echo "   Run: ./docker-setup-hf.sh"
fi
echo ""

# Check if TTY is available
if [ -t 0 ] && [ -t 1 ]; then
    DOCKER_TTY_FLAG="-it"
else
    DOCKER_TTY_FLAG=""
fi

# Mount Hugging Face cache if it exists
HF_CACHE_MOUNT=""
if [ -d "$HOME/.cache/huggingface" ]; then
    HF_CACHE_MOUNT="-v $HOME/.cache/huggingface:/root/.cache/huggingface"
    echo "Mounting Hugging Face cache from $HOME/.cache/huggingface"
fi

docker run \
    $GPU_FLAG \
    --rm \
    $DOCKER_TTY_FLAG \
    -v "$(pwd):/workspace" \
    -v cosmos-transfer2.5-venv:/workspace/.venv \
    $HF_CACHE_MOUNT \
    -w /workspace \
    --name cosmos-transfer2.5-run \
    cosmos-transfer2.5:latest \
    "$@"

