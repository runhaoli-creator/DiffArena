#!/bin/bash
# Docker环境测试脚本
# 用于验证Docker容器中的环境是否配置正确

set -e

echo "=========================================="
echo "Docker Environment Test"
echo "=========================================="

# 激活环境
source .venv/bin/activate
export PYTHONPATH="/workspace:/workspace/diffusion_policy:$PYTHONPATH"
export LEARN_NOISE=1

# 安装缺失的依赖（如果需要）
echo ""
echo "Installing missing dependencies..."
if command -v uv &> /dev/null; then
    uv pip install numba>=0.60.0 || /workspace/.venv/bin/pip install -q numba>=0.60.0 || true
else
    /workspace/.venv/bin/pip install -q numba>=0.60.0 || true
fi

# 测试导入
echo ""
echo "Testing imports..."
echo ""

# 测试Cosmos Transfer
echo "1. Testing Cosmos Transfer2.5..."
python -c "from cosmos_transfer2.config import SetupArguments; print('   ✓ Cosmos Transfer2.5 config imported')" || echo "   ✗ Failed"

# 测试Diffusion Policy
echo "2. Testing Diffusion Policy..."
python -c "from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy; print('   ✓ DiffusionUnetImagePolicy imported')" || echo "   ✗ Failed"
python -c "from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset; print('   ✓ PushTImageDataset imported')" || echo "   ✗ Failed"

# 测试PyTorch
echo "3. Testing PyTorch..."
python -c "import torch; print(f'   ✓ PyTorch {torch.__version__}'); print(f'   ✓ CUDA available: {torch.cuda.is_available()}')" || echo "   ✗ Failed"

# 测试adversarial_attack脚本
echo "4. Testing adversarial_attack.py..."
python -c "import sys; sys.path.insert(0, '/workspace'); from adversarial_attack import AdversarialAttack; print('   ✓ AdversarialAttack class imported')" || echo "   ✗ Failed"

echo ""
echo "=========================================="
echo "Docker environment test completed!"
echo "=========================================="

