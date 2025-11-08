#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Setup script to authenticate with Hugging Face in Docker
# This script helps you log in to Hugging Face inside the Docker container

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Hugging Face Setup for Cosmos-Transfer2.5"
echo "=========================================="
echo ""
echo "Before running inference, you need to:"
echo "1. Accept the NVIDIA Open Model License at:"
echo "   https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B"
echo ""
echo "2. Get a Hugging Face token from:"
echo "   https://huggingface.co/settings/tokens"
echo ""
echo "3. Log in to Hugging Face"
echo ""
read -p "Press Enter to continue after completing steps 1-2..."

# Check if HF_TOKEN environment variable is set
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "Please enter your Hugging Face token:"
    read -s HF_TOKEN
    echo ""
fi

# Run Docker container to login
echo "Logging in to Hugging Face in Docker container..."

docker run \
    --rm \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e HF_TOKEN="$HF_TOKEN" \
    cosmos-transfer2.5:latest \
    bash -c "pip install -q huggingface_hub[cli] && huggingface-cli login --token \$HF_TOKEN"

echo ""
echo "✅ Hugging Face authentication completed!"
echo ""
echo "You can now run inference with:"
echo "  ./docker-run-inference.sh -i assets/robot_example/params_learnable.json -o outputs/robot_test_learnable"


