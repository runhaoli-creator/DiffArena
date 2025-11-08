#!/bin/bash
# 简单的梯度流测试脚本
# 可以在 CPU 或 GPU 上运行，使用很小的张量

set -e

echo "=========================================="
echo "梯度流测试脚本"
echo "=========================================="
echo ""

# 检查是否在 Docker 中
if [ -f /.dockerenv ]; then
    echo "✅ 在 Docker 容器中运行"
    PYTHON_CMD="python"
else
    echo "ℹ️  在本地环境运行"
    PYTHON_CMD="python3"
fi

# 检查是否有 GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_AVAILABLE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$GPU_AVAILABLE" -gt 0 ]; then
        echo "✅ GPU 可用，显存: ${GPU_AVAILABLE}MB"
        DEVICE="cuda"
    else
        echo "⚠️  GPU 显存不足，将使用 CPU"
        DEVICE="cpu"
    fi
else
    echo "ℹ️  未检测到 GPU，将使用 CPU"
    DEVICE="cpu"
fi

echo ""
echo "测试模式 1: 默认模式（关闭随机噪声注入）"
echo "----------------------------------------"
$PYTHON_CMD test_gradient_flow.py

echo ""
echo "测试模式 2: 启用可学习的每步噪声（从 z0 派生）"
echo "----------------------------------------"
LEARNABLE_STEP_NOISE=1 $PYTHON_CMD test_gradient_flow.py

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="

