# Learnable Noise 代码修改检查清单

## ✅ 已完成的修改

### 1. `cosmos_transfer2/inference.py`
- ✅ 添加了 learnable noise 参数支持（`learn_noise`, `noise_steps`, `noise_lr`）
- ✅ 使用 `_generate_img2world_impl` 而不是 `generate_img2world`（绕过 @torch.no_grad()）
- ✅ 使用 `(-loss).backward()` 实现梯度上升（最大化 loss）
- ✅ 在 backward() 之后、step() 之前检查梯度
- ✅ 优化器顺序正确：先 step() 再 zero_grad()
- ✅ 添加了详细的日志输出

### 2. `cosmos_transfer2/_src/common/modules/res_sampler.py`
- ✅ 添加了 `GLOBAL_Z0_ANCHOR` 模块级全局变量
- ✅ 在 `_forward_impl` 中创建可学习的 z0（`z0.requires_grad_(True)`）
- ✅ 正确设置 `GLOBAL_Z0_ANCHOR`（使用 sys.modules）
- ✅ 实现了 learnable step noise 功能（从 z0 派生每步噪声）
- ✅ 在 `forward` 方法中设置 `learn_noise=True` 当 `allow_grad=True`
- ✅ 支持环境变量 `LEARNABLE_STEP_NOISE` 来控制是否启用可学习的每步噪声
- ✅ 在 `step_fn` 中正确处理三种情况：
  - `learnable_step_noise=True`: 使用从 z0 派生的噪声
  - `learn_noise=True` 但 `learnable_step_noise=False`: 关闭随机噪声注入
  - 正常情况: 使用独立随机噪声

### 3. `cosmos_transfer2/_src/transfer2/inference/inference_pipeline.py`
- ✅ 创建了 `_generate_img2world_impl` 方法（不带 @torch.no_grad()）
- ✅ `generate_img2world` 作为包装器调用 `_generate_img2world_impl`

### 4. `cosmos_transfer2/config.py`
- ✅ `InferenceArguments` 允许额外字段（`extra="allow"`）
- ✅ 添加了 `learn_noise`, `noise_steps`, `noise_lr` 参数

### 5. 配置文件
- ✅ 创建了 `assets/robot_example/params_learnable.json`（正确的路径格式）

## 关键代码路径验证

### 梯度流路径
```
output_video (带梯度)
  ← denoised_output
  ← ... (采样过程)
  ← step_noise (从 z0 派生，或不存在)
  ← start_state (sigma_max * z0)
  ← z0 (requires_grad=True)
```

### 关键检查点
1. ✅ `z0.requires_grad_(True)` - 已设置
2. ✅ `torch.set_grad_enabled(True)` - 在第一次生成时启用
3. ✅ `(-loss).backward()` - 梯度上升
4. ✅ `GLOBAL_Z0_ANCHOR` - 正确设置和访问
5. ✅ 优化器更新 - 顺序正确

## 使用方式

### 默认模式（关闭随机噪声注入）
```bash
python examples/inference.py -i assets/robot_example/params_learnable.json -o outputs/test
```

### 启用可学习的每步噪声（保持随机噪声注入）
```bash
export LEARNABLE_STEP_NOISE=1
python examples/inference.py -i assets/robot_example/params_learnable.json -o outputs/test
```

## 预期输出

运行成功后，日志中应该看到：
1. `Learnable noise config: learn_noise=True, noise_steps=1, noise_lr=0.005`
2. `Starting learnable noise optimization...`
3. `Initial loss: <数值>`
4. `✅ Gradient successfully backpropagated to z0!`
5. `z0 grad mean: <数值>`
6. `Noise optimization step 1/1 completed`

## 注意事项

- 如果 `learnable_step_noise=True`，会看到：`Generated N learnable step noises from z0 (gradient-preserving)`
- 如果梯度没有反传成功，会看到详细的错误提示和检查项

