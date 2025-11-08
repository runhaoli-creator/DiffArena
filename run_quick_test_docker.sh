#!/bin/bash
# 在Docker中快速测试运行

set -e

echo "=========================================="
echo "快速测试 - Docker环境"
echo "使用GPU 1和2"
echo "=========================================="

cd /home/runhaoli/runhaoli/cosmos-transfer2.5/cosmos-transfer2.5

# 运行Docker容器
docker run --gpus '"device=1,2"' --rm \
    -v $(pwd):/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN\
    -e LEARN_NOISE=1 \
    -e CUDA_VISIBLE_DEVICES=0,1 \
    cosmos-transfer-diffusion-policy \
    bash -c "
        source .venv/bin/activate
        export PYTHONPATH=/workspace:/workspace/diffusion_policy:\$PYTHONPATH
        
        # 安装numba
        uv pip install numba > /dev/null 2>&1
        
        # 登录HuggingFace
        python -c 'from huggingface_hub import login; import os; login(token=os.environ[\"HF_TOKEN\"])' > /dev/null 2>&1
        
        # 运行快速测试
        cd /workspace
        python quick_test.py
    "

