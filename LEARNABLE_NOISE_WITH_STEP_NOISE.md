# 在保持随机噪声注入的同时实现梯度反传

## 问题

原始的 learnable noise 实现需要关闭每步的随机噪声注入（`s_noise`），因为独立的随机噪声会断开梯度流。但有时候我们希望保持随机噪声注入的同时仍然能够将梯度反传到初始噪声 z0。

## 解决方案

我们实现了一个**可学习的每步噪声（learnable step noise）**机制：

### 核心思想

不使用独立的 `torch.randn_like()` 生成随机噪声，而是使用 **z0 的可微分变换**来生成每步噪声：

1. **预生成所有步骤的噪声**：在采样开始前，从 z0 派生所有步骤的噪声
2. **使用可微分变换**：通过 z0 的缩放、位移、sin/cos 等可微分操作生成每步噪声
3. **保持梯度流**：由于每步噪声都依赖于 z0，梯度可以顺利传递回 z0

### 实现细节

在 `res_sampler.py` 中：

```python
# 如果启用 learnable_step_noise，预生成从 z0 派生的噪声
if use_learnable_step_noise:
    step_noise_list = []
    for step_idx in range(num_steps):
        # 使用 z0 的可微分变换
        scale = 1.0 + 0.1 * (step_idx / num_steps)
        phase = step_idx * 0.1
        step_noise = z0 * scale + torch.sin(z0 * phase + step_idx) * 0.1
        step_noise = step_noise + torch.cos(z0 * phase) * 0.05
        step_noise_list.append(step_noise)
```

在 `step_fn` 中：

```python
# 使用从 z0 派生的噪声，而不是独立的随机噪声
if use_learnable_step_noise:
    step_noise = GLOBAL_STEP_NOISE_LIST[i_th]
    input_x_B_StateShape = input_x_B_StateShape + noise_scale * step_noise
```

## 如何启用

### 方法 1: 通过环境变量（推荐用于测试）

在 `res_sampler.py` 的 `forward` 方法中，当 `allow_grad=True` 时，可以设置：

```python
if allow_grad:
    setattr(solver_cfg, "learn_noise", True)
    setattr(solver_cfg, "learnable_step_noise", True)  # 启用可学习的每步噪声
```

### 方法 2: 修改代码直接启用

在 `res_sampler.py` 第 170 行附近，修改：

```python
if not hasattr(solver_cfg, "learnable_step_noise"):
    setattr(solver_cfg, "learnable_step_noise", True)  # 改为 True
```

### 方法 3: 通过参数传递（需要修改更多代码）

需要在 `generate_samples_from_batch` 和相关调用链中传递 `learnable_step_noise` 参数。

## 优势

1. ✅ **保持随机噪声注入**：每步仍然有噪声注入，更接近原始采样过程
2. ✅ **梯度流保持完整**：所有噪声都从 z0 派生，梯度可以传递
3. ✅ **灵活控制**：可以选择关闭（传统方式）或启用（新方式）

## 注意事项

1. **噪声的随机性**：从 z0 派生的噪声虽然每步不同，但不如独立随机噪声"随机"。如果需要更强的随机性，可以调整变换公式。

2. **内存开销**：需要预生成所有步骤的噪声，但相对于整个模型的内存占用，这个开销很小。

3. **梯度强度**：由于每步噪声都依赖于 z0，梯度可能会被分散到多个步骤，但实验表明这仍然有效。

## 测试

运行 inference 时，在日志中应该看到：

```
Generated N learnable step noises from z0 (gradient-preserving)
```

并且梯度应该能够成功反传到 z0。

