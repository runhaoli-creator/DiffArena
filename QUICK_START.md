# 快速开始指南 / Quick Start Guide

本指南将帮助你快速将Cosmos Transfer2.5和Diffusion Policy整合到同一个环境中。

## 前提条件 / Prerequisites

- Docker (推荐) 或 本地Python环境
- NVIDIA GPU 和 NVIDIA Driver
- 已下载的Cosmos Transfer2.5代码
- 已下载的Diffusion Policy代码

## 方法1: 使用Docker (推荐)

### 步骤1: 整合Diffusion Policy

```bash
cd /path/to/cosmos-transfer2.5
./integrate_diffusion_policy.sh
```

按照提示输入Diffusion Policy的路径，选择复制或符号链接。

### 步骤2: 构建Docker镜像

```bash
docker build --ulimit nofile=131071:131071 \
    -f Dockerfile.unified \
    -t cosmos-transfer-diffusion-policy .
```

### 步骤3: 运行Docker容器

```bash
docker run --gpus all -it \
    -v $(pwd):/workspace \
    -v cosmos-transfer-venv:/workspace/.venv \
    -e LEARN_NOISE=1 \
    cosmos-transfer-diffusion-policy
```

或者使用docker-compose:

```bash
docker-compose -f docker-compose.unified.yml up -d
docker-compose -f docker-compose.unified.yml exec cosmos-transfer-diffusion-policy bash
```

### 步骤4: 测试环境

在容器内运行:

```bash
python test_unified_env.py
```

## 方法2: 本地环境 (不使用Docker)

### 步骤1: 整合Diffusion Policy

```bash
cd /path/to/cosmos-transfer2.5
./integrate_diffusion_policy.sh
```

### 步骤2: 设置环境

```bash
./setup_unified_env.sh
```

### 步骤3: 激活环境

```bash
source .env
```

或者手动激活:

```bash
source .venv/bin/activate
export PYTHONPATH="$(pwd):$(pwd)/diffusion_policy:$PYTHONPATH"
export LEARN_NOISE=1
```

### 步骤4: 测试环境

```bash
python test_unified_env.py
```

## 运行对抗性攻击

环境配置完成后，可以运行对抗性攻击脚本:

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

## 常见问题

### Q: 如何在现有的Docker容器中添加Diffusion Policy?

A: 如果你已经有Cosmos Transfer的Docker容器在运行:

1. 在容器内运行:
   ```bash
   pip install -r requirements-diffusion-policy.txt
   pip install -e ./diffusion_policy
   ```

2. 设置环境变量:
   ```bash
   export PYTHONPATH="/workspace:/workspace/diffusion_policy:$PYTHONPATH"
   ```

3. 重启容器或重新构建镜像

### Q: 依赖冲突怎么办?

A: 
1. 优先使用Cosmos Transfer的依赖版本
2. 检查`requirements-diffusion-policy.txt`中的版本号
3. 如果仍有冲突，可能需要手动调整版本

### Q: 如何验证两个项目都能正常工作?

A: 运行测试脚本:
```bash
python test_unified_env.py
```

如果所有测试通过，说明环境配置正确。

## 目录结构

整合后的目录结构应该是:

```
cosmos-transfer2.5/
├── cosmos_transfer2/          # Cosmos Transfer2.5
├── diffusion_policy/          # Diffusion Policy (新添加)
├── adversarial_attack.py      # 对抗性攻击脚本
├── Dockerfile.unified         # 统一的Dockerfile
├── docker-compose.unified.yml # 统一的docker-compose
├── requirements-diffusion-policy.txt  # Diffusion Policy依赖
├── setup_unified_env.sh       # 环境设置脚本
├── test_unified_env.py        # 环境测试脚本
└── integrate_diffusion_policy.sh  # 整合脚本
```

## 下一步

- 阅读 `ENVIRONMENT_SETUP.md` 了解详细配置
- 阅读 `ADVERSARIAL_ATTACK_README.md` 了解对抗性攻击的使用方法
- 查看 `example_adversarial_attack.py` 查看示例代码

