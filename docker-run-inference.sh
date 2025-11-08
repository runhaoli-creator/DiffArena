#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convenience script to run inference in Docker
# Usage: ./docker-run-inference.sh [inference_args...]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default inference command if no arguments provided
if [ $# -eq 0 ]; then
    echo "No arguments provided. Running example inference with robot_example..."
    echo "Usage: $0 [inference_args...]"
    echo "Example: $0 -i assets/robot_example/edge/robot_edge_spec.json -o outputs/robot_test"
    echo ""
    echo "Running default example..."
    INFERENCE_CMD="python examples/inference.py -i assets/robot_example/edge/robot_edge_spec.json -o outputs/robot_test"
else
    INFERENCE_CMD="python examples/inference.py $@"
fi

# Run in Docker
cd "$SCRIPT_DIR"
./docker-run.sh bash -c "$INFERENCE_CMD"


