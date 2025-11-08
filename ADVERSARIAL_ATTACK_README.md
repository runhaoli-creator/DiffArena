# 对抗性攻击实现说明

本实现将Cosmos Transfer2.5的生成视频与Diffusion Policy模型连接，通过优化初始噪声来最大化Diffusion Policy的action loss。

## 核心思路

1. **Cosmos Transfer2.5生成视频**：从可学习的初始噪声开始生成视频
2. **转换为Diffusion Policy输入**：将生成的视频帧转换为Diffusion Policy需要的观测格式
3. **计算Action Loss**：Diffusion Policy根据输入图像预测动作，计算与ground truth的loss
4. **反向传播优化**：将loss反向传播到Cosmos Transfer的初始噪声，最大化该loss

## 文件说明

### 主要文件

- `adversarial_attack.py`: 主攻击脚本，包含完整的优化流程
- `cosmos_transfer2/_src/common/modules/res_sampler.py`: 修改了采样器以支持可学习的噪声

### 关键修改

1. **res_sampler.py**: 
   - 添加了对`GLOBAL_Z0_ANCHOR`的支持，允许外部传入可学习的噪声
   - 当`LEARN_NOISE`环境变量为"1"时，启用可学习噪声模式

2. **adversarial_attack.py**:
   - `AdversarialAttack`类：实现完整的对抗性攻击流程
   - `video_frames_to_obs_dict`: 将Cosmos生成的视频转换为Diffusion Policy的观测格式
   - `compute_action_loss`: 计算action loss
   - `optimize_noise`: 优化噪声的主循环

## 使用方法

### 基本用法

```bash
python adversarial_attack.py \
    --cosmos_checkpoint <cosmos_checkpoint_or_variant> \
    --input_video <path_to_input_video> \
    --prompt "your prompt here" \
    --diffusion_policy_checkpoint <path_to_policy_checkpoint> \
    --diffusion_policy_config <path_to_policy_config.yaml> \
    --dataset_path <path_to_dataset> \
    --output_dir ./adversarial_output \
    --num_iterations 10 \
    --lr 1e-3 \
    --device cuda
```

### 参数说明

- `--cosmos_checkpoint`: Cosmos Transfer的checkpoint路径或variant名称
- `--input_video`: 输入视频路径
- `--prompt`: 文本提示词
- `--diffusion_policy_checkpoint`: Diffusion Policy的checkpoint路径
- `--diffusion_policy_config`: Diffusion Policy的配置文件路径（YAML格式）
- `--dataset_path`: 数据集路径（用于获取ground truth）
- `--output_dir`: 输出目录
- `--num_iterations`: 优化迭代次数（默认10）
- `--lr`: 学习率（默认1e-3）
- `--device`: 运行设备（默认cuda）

## 工作流程

1. **初始化**：
   - 加载Cosmos Transfer2.5模型
   - 加载Diffusion Policy模型
   - 从数据集获取ground truth样本

2. **首次生成**：
   - 启用`LEARN_NOISE`环境变量
   - 生成视频以获取初始噪声的形状
   - 将初始噪声转换为可学习参数

3. **优化循环**：
   - 使用当前噪声参数生成视频
   - 将视频转换为Diffusion Policy的观测格式
   - Diffusion Policy预测动作
   - 计算与ground truth的loss
   - 反向传播（最大化loss = 最小化负loss）
   - 更新噪声参数

4. **保存结果**：
   - 保存每次迭代的视频和噪声
   - 保存优化历史记录

## 代码示例

```python
from adversarial_attack import AdversarialAttack, load_diffusion_policy
from cosmos_transfer2.inference import Control2WorldInference, SetupArguments, InferenceArguments
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset

# 初始化Cosmos Transfer
setup_args = SetupArguments(
    output_dir="./output",
    context_parallel_size=1,
    enable_guardrails=False,
)
cosmos_inference = Control2WorldInference(
    args=setup_args,
    batch_hint_keys=["edge"],
)

# 加载Diffusion Policy
diffusion_policy = load_diffusion_policy(
    checkpoint_path="path/to/checkpoint.ckpt",
    config_path="path/to/config.yaml"
)

# 加载数据集
dataset = PushTImageDataset("path/to/dataset", horizon=16)
normalizer = dataset.get_normalizer()
diffusion_policy.set_normalizer(normalizer)

# 获取ground truth
sample_batch = dataset[0]
ground_truth_batch = {
    'obs': {k: v.unsqueeze(0) for k, v in sample_batch['obs'].items()},
    'action': sample_batch['action'].unsqueeze(0),
}

# 创建攻击实例
attack = AdversarialAttack(
    cosmos_inference=cosmos_inference,
    diffusion_policy=diffusion_policy,
    ground_truth_batch=ground_truth_batch,
    device="cuda",
)

# 创建推理参数
sample_args = InferenceArguments(
    name="attack_sample",
    video_path="input_video.mp4",
    prompt="your prompt",
    hint_keys=["edge"],
    control_weight_dict={"edge": "1.0"},
    guidance=7,
    seed=42,
    resolution="720",
    num_steps=35,
)

# 运行优化
results = attack.optimize_noise(
    sample_args=sample_args,
    num_iterations=10,
    lr=1e-3,
    output_dir="./output/iterations",
)
```

## 注意事项

1. **内存使用**：生成过程需要大量GPU内存，建议使用大显存的GPU
2. **梯度流**：确保在整个流程中梯度能够正确反向传播
3. **噪声形状**：噪声的形状必须与模型的latent空间匹配
4. **数据集格式**：确保数据集格式与代码中期望的格式一致

## 故障排除

### 问题：GLOBAL_Z0_ANCHOR为None
- 确保`LEARN_NOISE`环境变量设置为"1"
- 检查`res_sampler.py`中的条件判断逻辑

### 问题：形状不匹配
- 检查输入视频的分辨率
- 检查Cosmos Transfer和Diffusion Policy的输入要求

### 问题：梯度为None
- 确保在整个流程中启用了梯度计算
- 检查模型是否处于eval模式（某些层在eval模式下可能不计算梯度）

## 未来改进

1. 支持批量优化
2. 添加更多的损失函数选项
3. 支持不同的噪声初始化策略
4. 添加可视化工具
5. 优化内存使用

