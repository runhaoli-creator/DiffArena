# Docker 使用指南

本文档说明如何在 Docker 中运行 Cosmos-Transfer2.5 的 inference 代码。

## 前置要求

1. **Docker**: 已安装 Docker
2. **NVIDIA Container Toolkit**: 已安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 以支持 GPU
3. **NVIDIA GPU**: 支持 CUDA 12.8.1 的 NVIDIA GPU（Ampere 架构或更新）

## 快速开始

### 方法 1: 使用便捷脚本（推荐）

#### 1. 构建 Docker 镜像

```bash
chmod +x docker-build.sh
./docker-build.sh
```

#### 2. 运行容器（交互式）

```bash
chmod +x docker-run.sh
./docker-run.sh
```

这会在容器内启动一个 bash shell，你可以在其中运行命令。

#### 3. 在容器内运行 inference

```bash
# 进入容器后，运行 inference
python examples/inference.py -i assets/robot_example/edge/robot_edge_spec.json -o outputs/robot_test

# 或者使用 learnable noise 配置
python examples/inference.py -i assets/robot_example/params.json -o outputs/robot_test
```

#### 4. 直接运行 inference（不进入交互式 shell）

```bash
chmod +x docker-run-inference.sh
./docker-run-inference.sh -i assets/robot_example/edge/robot_edge_spec.json -o outputs/robot_test
```

### 方法 2: 使用 Docker Compose

#### 1. 构建并启动容器

```bash
docker-compose up -d --build
```

#### 2. 进入容器

```bash
docker-compose exec cosmos-transfer2.5 bash
```

#### 3. 运行 inference

```bash
python examples/inference.py -i assets/robot_example/edge/robot_edge_spec.json -o outputs/robot_test
```

#### 4. 停止容器

```bash
docker-compose down
```

### 方法 3: 手动使用 Docker 命令

#### 1. 构建镜像

```bash
docker build --ulimit nofile=131071:131071 -f Dockerfile . -t cosmos-transfer2.5:latest
```

#### 2. 运行容器

```bash
docker run --gpus all --rm -it \
  -v $(pwd):/workspace \
  -v cosmos-transfer2.5-venv:/workspace/.venv \
  -w /workspace \
  cosmos-transfer2.5:latest \
  bash
```

#### 3. 在容器内运行 inference

```bash
python examples/inference.py -i assets/robot_example/edge/robot_edge_spec.json -o outputs/robot_test
```

## 运行 Learnable Noise 测试

要测试 learnable noise 功能，确保在配置文件中设置了 `learn_noise: true`，或者使用环境变量：

```bash
# 在容器内
export LEARN_NOISE=1
python examples/inference.py -i assets/robot_example/params.json -o outputs/robot_test_learnable
```

或者在 Docker Compose 中已经默认设置了 `LEARN_NOISE=1`。

## 数据持久化

- **代码**: 当前目录挂载到容器的 `/workspace`，代码修改会立即生效
- **输出**: `outputs/` 目录在容器内和宿主机之间共享
- **虚拟环境**: `.venv` 使用命名卷 `cosmos-transfer2.5-venv` 持久化，避免每次重建

## 常见问题

### 1. GPU 不可用

如果遇到 GPU 相关错误：

```bash
# 检查 NVIDIA Container Toolkit 是否安装
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

### 2. 文件描述符限制

如果构建时遇到文件描述符错误，使用 `--ulimit` 参数（已包含在 `docker-build.sh` 中）。

### 3. 内存不足

如果遇到内存不足，可以：
- 减少 batch size
- 使用较小的分辨率
- 关闭 guardrails（如果不需要）

### 4. 检查点下载

首次运行 inference 时，模型检查点会自动从 Hugging Face 下载。确保：
- 已登录 Hugging Face: `hf auth login`
- 已接受 [NVIDIA Open Model License](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B)

## 验证环境

在容器内运行以下命令验证环境：

```bash
# 检查 Python 版本
python --version

# 检查 GPU
nvidia-smi

# 检查 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 检查安装的包
uv pip list
```

## 清理

删除容器和镜像：

```bash
# 删除容器
docker rm cosmos-transfer2.5-run

# 删除镜像（可选）
docker rmi cosmos-transfer2.5:latest

# 删除 Docker Compose 相关（如果使用）
docker-compose down -v
```

## 更多信息

- [Setup Guide](docs/setup.md)
- [Inference Guide](docs/inference.md)
- [Docker 官方文档](https://docs.docker.com/)


