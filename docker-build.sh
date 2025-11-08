#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Docker build script for Cosmos-Transfer2.5
# This script builds the Docker image with proper configuration

set -e

echo "Building Cosmos-Transfer2.5 Docker image..."

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Build the Docker image
# Using --ulimit to avoid file descriptor issues during build
docker build \
    --ulimit nofile=131071:131071 \
    -f Dockerfile \
    -t cosmos-transfer2.5:latest \
    .

echo "Docker image built successfully!"
echo "Image name: cosmos-transfer2.5:latest"
echo ""
echo "To run the container, use:"
echo "  ./docker-run.sh"
echo "or"
echo "  docker run --gpus all --rm -v \$(pwd):/workspace -v cosmos-transfer2.5-venv:/workspace/.venv -it cosmos-transfer2.5:latest"


