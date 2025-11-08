#!/bin/bash
# 运行对抗攻击脚本 - 使用PushT数据集进行验证

set -e

echo "=========================================="
echo "Running Adversarial Attack on PushT"
echo "=========================================="

# 使用GPU 2和3（显存最空闲的）
export CUDA_VISIBLE_DEVICES=2,3

# 设置路径
WORKSPACE="/home/runhaoli/runhaoli/cosmos-transfer2.5/cosmos-transfer2.5"
cd "$WORKSPACE"

# HuggingFace token
: "${HF_TOKEN:?Please set HF_TOKEN environment variable before running this script}"
export LEARN_NOISE=1

# 数据集和checkpoint路径
DATASET_PATH="diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"
CHECKPOINT_PATH="diffusion_policy/data/outputs/2025.10.11/19.23.26_train_diffusion_unet_hybrid_pusht_image/checkpoints/epoch=0400-test_mean_score=0.901.ckpt"
# Checkpoint已包含cfg，不需要单独的config文件
CONFIG_PATH=""

# 输入视频（使用robot示例）
INPUT_VIDEO="assets/robot_example/robot_input.mp4"

# 输出目录
OUTPUT_DIR="./adversarial_output_pusht_$(date +%Y%m%d_%H%M%S)"

# 优化参数（小规模测试）
NUM_ITERATIONS=1  # 只运行3次迭代进行快速验证
LR=1e-3

echo "Configuration:"
echo "  GPU: 2,3"
echo "  Dataset: $DATASET_PATH"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Config: $CONFIG_PATH"
echo "  Input Video: $INPUT_VIDEO"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Iterations: $NUM_ITERATIONS"
echo "  Learning Rate: $LR"
echo ""

# 检查文件是否存在
if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Error: Input video not found: $INPUT_VIDEO"
    exit 1
fi

if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset not found: $DATASET_PATH"
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

# 在Docker中运行
docker run --gpus '"device=2,3"' --rm \
    -v "$WORKSPACE:/workspace" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e HF_TOKEN="$HF_TOKEN" \
    -e LEARN_NOISE=1 \
    -e CUDA_VISIBLE_DEVICES=0,1 \
    cosmos-transfer-diffusion-policy \
    bash -c "
        source .venv/bin/activate
        export PYTHONPATH=/workspace:/workspace/diffusion_policy:\$PYTHONPATH
        
        # 安装依赖（如果需要）
        # 安装系统依赖（CMake for robomimic）
        apt-get update -qq > /dev/null 2>&1 && apt-get install -y -qq cmake build-essential > /dev/null 2>&1 || true
        
        # 安装Python依赖
        uv pip install numba dill 'robomimic>=0.3.0' > /dev/null 2>&1 || true
        
        # 登录HuggingFace
        python -c 'from huggingface_hub import login; import os; login(token=os.environ[\"HF_TOKEN\"])' > /dev/null 2>&1
        
        cd /workspace
        
        # 运行对抗攻击
        CMD="python adversarial_attack.py \
            --cosmos_checkpoint edge \
            --input_video $INPUT_VIDEO \
            --prompt 'A robot arm manipulating objects on a table' \
            --diffusion_policy_checkpoint $CHECKPOINT_PATH \
            --dataset_path $DATASET_PATH \
            --output_dir $OUTPUT_DIR \
            --num_iterations $NUM_ITERATIONS \
            --lr $LR \
            --device cuda"
        
        # 如果CONFIG_PATH不为空，添加config参数
        if [ -n "$CONFIG_PATH" ]; then
            CMD=\"\$CMD --diffusion_policy_config $CONFIG_PATH\"
        fi
        
        eval \$CMD
    "

echo ""
echo "=========================================="
echo "Adversarial attack completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

