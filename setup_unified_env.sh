#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unified environment setup script
# This script sets up the environment for both Cosmos Transfer2.5 and Diffusion Policy

set -e

echo "=========================================="
echo "Unified Environment Setup"
echo "=========================================="

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if we're in a Docker container
if [ -f /.dockerenv ]; then
    echo "Running inside Docker container"
    IN_DOCKER=true
else
    echo "Running on host system"
    IN_DOCKER=false
fi

# Step 1: Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Step 2: Install Cosmos Transfer2.5 dependencies
echo ""
echo "Step 1: Installing Cosmos Transfer2.5 dependencies..."
uv sync

# Step 3: Activate the virtual environment
echo ""
echo "Step 2: Activating virtual environment..."
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "Virtual environment activated"
else
    echo "Warning: .venv not found. Creating..."
    uv venv
    source .venv/bin/activate
fi

# Step 4: Install Diffusion Policy dependencies
echo ""
echo "Step 3: Installing Diffusion Policy dependencies..."
if [ -f requirements-diffusion-policy.txt ]; then
    uv pip install -r requirements-diffusion-policy.txt
    echo "Diffusion Policy dependencies installed"
else
    echo "Warning: requirements-diffusion-policy.txt not found. Skipping..."
fi

# Step 5: Install Diffusion Policy package (if directory exists)
echo ""
echo "Step 4: Installing Diffusion Policy package..."
if [ -d "diffusion_policy" ]; then
    uv pip install -e ./diffusion_policy
    echo "Diffusion Policy package installed in development mode"
else
    echo "Warning: diffusion_policy directory not found."
    echo "Please copy or link the diffusion_policy directory to this location."
fi

# Step 6: Set environment variables
echo ""
echo "Step 5: Setting environment variables..."
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/diffusion_policy:${PYTHONPATH}"
export LEARN_NOISE=1

# Create a .env file for easy sourcing
cat > .env << EOF
# Unified Environment Variables
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/diffusion_policy:\${PYTHONPATH}"
export LEARN_NOISE=1

# Activate virtual environment
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi
EOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source .env"
echo ""
echo "Or manually activate:"
echo "  source .venv/bin/activate"
echo "  export PYTHONPATH=\"$SCRIPT_DIR:$SCRIPT_DIR/diffusion_policy:\$PYTHONPATH\""
echo "  export LEARN_NOISE=1"
echo ""
echo "To test the environment, run:"
echo "  python test_unified_env.py"
echo ""

