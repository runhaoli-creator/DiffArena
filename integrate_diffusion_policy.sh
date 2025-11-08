#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Script to integrate Diffusion Policy into Cosmos Transfer2.5 project
# This script helps set up the unified codebase structure

set -e

echo "=========================================="
echo "Diffusion Policy Integration Script"
echo "=========================================="
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if diffusion_policy already exists
if [ -d "diffusion_policy" ]; then
    echo "⚠ Warning: diffusion_policy directory already exists."
    read -p "Do you want to remove it and re-integrate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing diffusion_policy directory..."
        rm -rf diffusion_policy
    else
        echo "Keeping existing directory. Exiting."
        exit 0
    fi
fi

# Ask for Diffusion Policy path
echo "Please provide the path to your Diffusion Policy repository:"
read -p "Path: " DIFFUSION_POLICY_PATH

# Expand user home directory if ~ is used
DIFFUSION_POLICY_PATH="${DIFFUSION_POLICY_PATH/#\~/$HOME}"

# Check if path exists
if [ ! -d "$DIFFUSION_POLICY_PATH" ]; then
    echo "❌ Error: Path does not exist: $DIFFUSION_POLICY_PATH"
    exit 1
fi

# Check if it looks like a Diffusion Policy repository
if [ ! -f "$DIFFUSION_POLICY_PATH/diffusion_policy/__init__.py" ]; then
    echo "⚠ Warning: This doesn't look like a Diffusion Policy repository."
    echo "Expected: $DIFFUSION_POLICY_PATH/diffusion_policy/__init__.py"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Ask for integration method
echo ""
echo "Choose integration method:"
echo "1. Copy (recommended for Docker, creates independent copy)"
echo "2. Symbolic link (saves space, but requires both repos to stay together)"
read -p "Enter choice (1 or 2): " INTEGRATION_METHOD

case $INTEGRATION_METHOD in
    1)
        echo "Copying Diffusion Policy directory..."
        cp -r "$DIFFUSION_POLICY_PATH" ./diffusion_policy
        echo "✓ Copied successfully"
        ;;
    2)
        echo "Creating symbolic link..."
        ln -s "$DIFFUSION_POLICY_PATH" ./diffusion_policy
        echo "✓ Symbolic link created"
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac

# Verify integration
if [ -d "diffusion_policy" ] && [ -f "diffusion_policy/diffusion_policy/__init__.py" ]; then
    echo ""
    echo "✅ Integration successful!"
    echo ""
    echo "Next steps:"
    echo "1. Run: ./setup_unified_env.sh"
    echo "2. Run: python test_unified_env.py"
    echo "3. If using Docker, build with: docker build -f Dockerfile.unified -t cosmos-transfer-diffusion-policy ."
else
    echo ""
    echo "❌ Integration verification failed. Please check manually."
    exit 1
fi

