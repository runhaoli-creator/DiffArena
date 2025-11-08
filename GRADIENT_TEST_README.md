# 梯度流测试指南

由于 GPU 显存紧张，我们提供了一个简单的测试脚本来验证梯度反传是否成功。这个脚本使用很小的张量（约 0.0005 MB），不需要加载完整的模型，可以在 CPU 或 GPU 上快速运行。

## 快速开始

### 方法 1: 直接运行测试脚本（推荐）

```bash
# 在本地环境
python test_gradient_flow.py

# 或在 Docker 中
docker run --rm -v $(pwd):/workspace -w /workspace cosmos-transfer2.5:latest python test_gradient_flow.py
```

### 方法 2: 使用测试脚本（自动测试两种模式）

```bash
# 在本地环境
./run_gradient_test.sh

# 或在 Docker 中
docker run --rm -v $(pwd):/workspace -w /workspace cosmos-transfer2.5:latest bash run_gradient_test.sh
```

## 测试模式

### 模式 1: 默认模式（关闭随机噪声注入）

```bash
python test_gradient_flow.py
```

这种模式会：
- 关闭每步的随机噪声注入
- 使用 `z0` 作为唯一的噪声源
- 梯度直接从 `output` 反传到 `z0`

### 模式 2: 可学习的每步噪声（从 z0 派生）

```bash
LEARNABLE_STEP_NOISE=1 python test_gradient_flow.py
```

这种模式会：
- 每步注入从 `z0` 派生的噪声（保持梯度流）
- 使用可微分变换从 `z0` 生成每步噪声
- 梯度可以从每步噪声传递回 `z0`

## 预期输出

如果测试成功，你应该看到：

```
============================================================
开始测试梯度反传...
============================================================

1. 运行前向传播（带梯度）...
   输出形状: torch.Size([1, 2, 2, 8, 8])
   output.requires_grad: True

2. 检查 GLOBAL_Z0_ANCHOR...
   ✅ GLOBAL_Z0_ANCHOR 已设置
   形状: torch.Size([1, 2, 2, 8, 8])
   requires_grad: True

3. 计算 loss 并反向传播...
   Loss: 0.xxxxxx
   ✅ 反向传播完成

4. 检查梯度...
   ==================================================
   ✅ 梯度成功反传到 z0！
   ==================================================
   z0 grad mean:  x.xxxxxxe-xx
   z0 grad max:   x.xxxxxxe-xx
   z0 grad min:   x.xxxxxxe-xx

   ✅✅✅ 测试通过！梯度成功反传到 learnable noise！

5. 测试优化器更新...
   z0 更新量 (mean abs diff): x.xxxxxxe-xx
   ✅ 优化器成功更新了 z0

============================================================
测试完成！
============================================================
```

## 如果测试失败

如果看到 `❌ 没有梯度！`，请检查：

1. **`res_sampler._forward_impl` 中的 `allow_grad` 是否为 `True`**
   - 检查 `res_sampler.py` 第 202 行

2. **`z0.requires_grad_(True)` 是否被调用**
   - 检查 `res_sampler.py` 第 219 行

3. **前向传播是否在 `torch.set_grad_enabled(True)` 中**
   - 测试脚本已经确保这一点

4. **`solver_cfg.learn_noise` 是否设置为 `True`**
   - 测试脚本已经设置

5. **如果使用了 `learnable_step_noise`，检查是否正确设置**
   - 检查 `res_sampler.py` 第 231-245 行

## 资源占用

- **显存**: ~0.0005 MB（非常小）
- **运行时间**: < 1 秒（CPU 或 GPU）
- **不需要**: 完整的模型权重、视频文件等

## 与完整 inference 的区别

这个测试脚本：
- ✅ 使用很小的张量（1×2×2×8×8）
- ✅ 使用简单的线性 denoiser（`x0 = 0.7*x + 0.1`）
- ✅ 只运行 4 步采样
- ✅ 不需要加载模型权重
- ✅ 不需要视频文件

完整 inference：
- 使用真实的视频尺寸（例如 720p）
- 使用真实的 diffusion 模型
- 运行更多采样步数
- 需要加载模型权重（几 GB）

## 测试原理

测试脚本模拟了完整的梯度流：

```
z0 (requires_grad=True)
  ↓
sigma_max * z0 = start_state
  ↓
采样过程（可能包含 step_noise）
  ↓
denoised_output
  ↓
loss = MSE(output, target)
  ↓
backward(-loss)  # 梯度上升
  ↓
z0.grad ✅
```

如果 `z0.grad` 不为 `None` 且梯度值合理，说明梯度反传成功！

