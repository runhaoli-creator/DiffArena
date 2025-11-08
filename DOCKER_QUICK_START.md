# Docker快速启动指南

## ✅ Docker环境已配置完成

统一环境已成功配置，可以在Docker中运行。

## 快速启动

### 方法1: 使用Docker Compose（推荐）

```bash
# 启动容器
docker-compose -f docker-compose.unified.yml up -d

# 进入容器
docker-compose -f docker-compose.unified.yml exec cosmos-transfer-diffusion-policy bash

# 在容器内
source .venv/bin/activate
export PYTHONPATH=/workspace:/workspace/diffusion_policy:$PYTHONPATH
uv pip install numba  # 首次运行需要安装numba
```

### 方法2: 直接使用Docker

```bash
# 启动交互式容器
docker run --gpus all -it \
    -v $(pwd):/workspace \
    -v cosmos-transfer-venv:/workspace/.venv \
    -e LEARN_NOISE=1 \
    cosmos-transfer-diffusion-policy

# 在容器内
source .venv/bin/activate
export PYTHONPATH=/workspace:/workspace/diffusion_policy:$PYTHONPATH
uv pip install numba  # 首次运行需要安装numba
```

## 验证环境

在容器内运行：

```bash
# 安装numba（首次运行）
uv pip install numba

# 运行测试
python test_unified_env.py

# 或快速测试
python -c "
import torch
import cosmos_transfer2
import diffusion_policy
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
print('✅ All imports successful!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"
```

## 运行对抗性攻击

在容器内：

```bash
python adversarial_attack.py \
    --cosmos_checkpoint edge \
    --input_video assets/robot_example/robot_input.mp4 \
    --prompt "A robot arm manipulating objects" \
    --diffusion_policy_checkpoint /path/to/checkpoint.ckpt \
    --diffusion_policy_config /path/to/config.yaml \
    --dataset_path /path/to/dataset.zarr \
    --output_dir ./adversarial_output \
    --num_iterations 10 \
    --lr 1e-3
```

## 注意事项

1. **numba安装**: 首次运行需要 `uv pip install numba`
2. **HuggingFace认证**: 如果需要guardrails，需要设置HF token
3. **GPU支持**: 确保安装了nvidia-docker2
4. **数据持久化**: 使用volumes保存.venv和数据

## 环境变量

容器内已设置：
- `PYTHONPATH=/workspace:/workspace/diffusion_policy`
- `LEARN_NOISE=1`
- `PATH=/workspace/.venv/bin:$PATH`

## 故障排除

### 问题: numba未安装
```bash
uv pip install numba
```

### 问题: 权限错误
```bash
# 检查volume权限
docker volume inspect cosmos-transfer-diffusion-policy-venv
```

### 问题: GPU不可用
```bash
# 检查nvidia-docker
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

