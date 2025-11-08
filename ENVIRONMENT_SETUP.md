# 统一环境配置指南

本文档说明如何将Cosmos Transfer2.5和Diffusion Policy整合到同一个环境中，并确保它们在Docker和本地环境中都能正常运行。

## 目录结构

推荐的项目结构：

```
cosmos-transfer2.5/
├── cosmos_transfer2/          # Cosmos Transfer2.5 主代码
├── diffusion_policy/          # Diffusion Policy 子目录（从原项目复制）
├── adversarial_attack.py      # 对抗性攻击脚本
├── Dockerfile                 # 统一的Dockerfile
├── docker-compose.yml         # Docker Compose配置
├── pyproject.toml             # Cosmos Transfer依赖（已存在）
├── requirements-diffusion-policy.txt  # Diffusion Policy额外依赖
└── setup_unified_env.sh       # 统一环境安装脚本
```

## 步骤1: 整合Diffusion Policy到Cosmos Transfer项目

### 1.1 复制Diffusion Policy代码

```bash
# 在cosmos-transfer2.5目录下执行
cd /path/to/cosmos-transfer2.5
cp -r /path/to/diffusion_policy ./diffusion_policy
```

### 1.2 创建符号链接（可选，如果不想复制）

```bash
# 如果两个项目在同一父目录下
cd /path/to/cosmos-transfer2.5
ln -s ../diffusion_policy ./diffusion_policy
```

## 步骤2: 创建统一的环境配置文件

### 2.1 创建Diffusion Policy的依赖文件

创建 `requirements-diffusion-policy.txt`，包含Diffusion Policy需要的依赖（兼容Python 3.10和PyTorch 2.x）：

```bash
# Diffusion Policy依赖（适配到PyTorch 2.x和Python 3.10）
# 注意：原conda环境中的某些包需要替换为新版本

# 核心依赖
einops>=0.7.0
hydra-core>=1.3.0
omegaconf>=2.3.0
wandb>=0.15.0

# 数据处理
zarr>=2.14.0
numcodecs>=0.11.0
h5py>=3.8.0
imageio>=2.31.0
imageio-ffmpeg>=0.4.9

# 机器人仿真（可选，如果不需要可以注释掉）
# gym>=0.21.0
# pymunk>=6.5.0
# free-mujoco-py>=2.3.0  # 替代原mujoco-py

# 视觉处理
opencv-python>=4.8.0
scikit-image>=0.21.0
scikit-video>=1.1.11

# 工具库
tqdm>=4.66.0
dill>=0.3.7
termcolor>=2.3.0
psutil>=5.9.0
click>=8.1.0

# Diffusers（兼容PyTorch 2.x）
diffusers>=0.24.0  # 升级到支持PyTorch 2.x的版本

# 其他
matplotlib>=3.7.0
tensorboard>=2.14.0
tensorboardx>=2.6.0
```

## 步骤3: 修改Dockerfile以支持两个项目

我将在下一步创建修改后的Dockerfile。

## 步骤4: 修改pyproject.toml添加Diffusion Policy依赖

在`pyproject.toml`的`[project]`部分添加：

```toml
dependencies = [
  # ... 原有依赖 ...
  # Diffusion Policy依赖
  "einops>=0.7.0",
  "hydra-core>=1.3.0",
  "omegaconf>=2.3.0",
  "wandb>=0.15.0",
  "zarr>=2.14.0",
  "numcodecs>=0.11.0",
  "h5py>=3.8.0",
  "imageio>=2.31.0",
  "imageio-ffmpeg>=0.4.9",
  "opencv-python>=4.8.0",
  "scikit-image>=0.21.0",
  "scikit-video>=1.1.11",
  "tqdm>=4.66.0",
  "dill>=0.3.7",
  "termcolor>=2.3.0",
  "psutil>=5.9.0",
  "diffusers>=0.24.0",
  "matplotlib>=3.7.0",
  "tensorboard>=2.14.0",
  "tensorboardx>=2.6.0",
]
```

## 步骤5: 修改adversarial_attack.py的导入路径

由于两个项目现在在同一个目录下，导入路径会更简单：

```python
# 修改后的导入
from cosmos_transfer2.inference import Control2WorldInference
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
```

## 步骤6: 环境变量配置

在Docker或本地环境中设置：

```bash
export PYTHONPATH="/workspace:/workspace/diffusion_policy:$PYTHONPATH"
export LEARN_NOISE=1  # 启用可学习噪声
```

## 使用Docker

### 构建镜像

```bash
docker build --ulimit nofile=131071:131071 -f Dockerfile . -t cosmos-transfer-diffusion-policy
```

### 运行容器

```bash
docker run --gpus all -it \
  -v $(pwd):/workspace \
  -v cosmos-transfer-venv:/workspace/.venv \
  -e LEARN_NOISE=1 \
  cosmos-transfer-diffusion-policy
```

## 使用Docker Compose

使用提供的`docker-compose.yml`（已更新）：

```bash
docker-compose up -d
docker-compose exec cosmos-transfer2.5 bash
```

## 本地环境（不使用Docker）

### 安装uv（如果还没有）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 安装依赖

```bash
cd cosmos-transfer2.5
uv sync
source .venv/bin/activate

# 安装Diffusion Policy的额外依赖
pip install -r requirements-diffusion-policy.txt

# 安装Diffusion Policy包（开发模式）
pip install -e ./diffusion_policy
```

### 设置环境变量

```bash
export PYTHONPATH="$(pwd):$(pwd)/diffusion_policy:$PYTHONPATH"
export LEARN_NOISE=1
```

## 验证环境

创建测试脚本`test_unified_env.py`：

```python
#!/usr/bin/env python3
"""测试统一环境是否配置正确"""

import sys
print(f"Python version: {sys.version}")

# 测试Cosmos Transfer导入
try:
    from cosmos_transfer2.inference import Control2WorldInference
    print("✓ Cosmos Transfer2.5 imported successfully")
except Exception as e:
    print(f"✗ Cosmos Transfer2.5 import failed: {e}")
    sys.exit(1)

# 测试Diffusion Policy导入
try:
    from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
    from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
    print("✓ Diffusion Policy imported successfully")
except Exception as e:
    print(f"✗ Diffusion Policy import failed: {e}")
    sys.exit(1)

# 测试PyTorch版本
import torch
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch.version.cuda}")

print("\n✅ All imports successful! Environment is ready.")
```

运行测试：

```bash
python test_unified_env.py
```

## 常见问题

### Q1: Diffusion Policy需要PyTorch 1.12，但Cosmos Transfer需要PyTorch 2.7，怎么办？

A: 我们需要升级Diffusion Policy的代码以兼容PyTorch 2.x。主要修改点：
- `diffusers`库升级到0.24.0+（支持PyTorch 2.x）
- 检查代码中的API变化并适配
- 某些旧API可能需要替换

### Q2: 某些Diffusion Policy的依赖与Cosmos Transfer冲突怎么办？

A: 优先级规则：
1. Cosmos Transfer的依赖优先（保持其环境稳定）
2. 对于冲突的包，尝试升级到兼容版本
3. 如果无法兼容，考虑修改Diffusion Policy代码以使用Cosmos Transfer的版本

### Q3: 如何在Docker中安装conda依赖？

A: 我们可以在Docker中安装miniconda，然后使用conda安装某些特殊包，但推荐尽量使用pip/uv管理所有依赖。

### Q4: 内存不够怎么办？

A: 
- 使用Docker的volume缓存
- 减少不必要的依赖
- 使用`--no-install-project`选项安装

## 下一步

1. 运行`test_unified_env.py`验证环境
2. 运行`adversarial_attack.py`测试完整流程
3. 根据实际情况调整依赖版本

