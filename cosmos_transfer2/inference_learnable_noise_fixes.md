# Learnable Noise 代码检查与修复建议

## 发现的问题

### 1. **优化器顺序问题** ⚠️
当前代码：
```python
opt = torch.optim.Adam([_rs.GLOBAL_Z0_ANCHOR], lr=noise_lr)
for _ in range(noise_steps):
    opt.step()
    opt.zero_grad()
```
**问题**：在第一次迭代中，应该在 backward() 之后立即调用 step()，但循环中先 step() 再 zero_grad() 的顺序不对。

**修复**：应该先 zero_grad() 再 step()，或者对于单步优化，直接在 backward() 后 step()。

### 2. **最大化 Loss 的目标** ⚠️⚠️⚠️
当前代码：
```python
loss = F.mse_loss(output_video, baseline_out)
loss.backward()
```
**问题**：如果目标是**最大化 loss**（让 loss 越来越大），当前实现会**最小化 loss**。

**修复**：应该使用梯度上升，即对 `-loss` 做 backward，或者使用梯度上升的优化器（负学习率）。

### 3. **solver_cfg.learn_noise 传递问题** ⚠️
当前代码在 `res_sampler.py` 中：
```python
if allow_grad or getattr(sampler_cfg.solver, "learn_noise", False) or LEARN_NOISE_ENV:
```
**问题**：需要确保在第一次生成时，solver_cfg.solver.learn_noise 被设置为 True，这样 step_fn 中才会关闭随机噪声注入。

### 4. **梯度检查时机问题**
当前代码在 step() 之后检查梯度，但此时梯度可能已经被清零。


