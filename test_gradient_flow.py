#!/usr/bin/env python3
"""
简单的梯度流测试脚本
用于验证 learnable noise 的梯度反传是否正常工作
不需要完整的模型，只需要很小的张量即可测试（节省显存）

使用方法：
  # 默认模式（关闭随机噪声注入）
  python test_gradient_flow.py

  # 启用可学习的每步噪声（从 z0 派生）
  LEARNABLE_STEP_NOISE=1 python test_gradient_flow.py
"""

import os
import sys

# 设置环境变量（必须在导入之前）
os.environ["LEARN_NOISE"] = "1"  # 启用 learnable noise

import torch
import torch.nn.functional as F

# 导入 res_sampler
try:
    from cosmos_transfer2._src.common.modules.res_sampler import (
        Sampler, SamplerConfig, SolverConfig, SolverTimestampConfig
    )
    # 导入 GLOBAL_Z0_ANCHOR
    import cosmos_transfer2._src.common.modules.res_sampler as _rs
except ImportError as e:
    print(f"❌ 无法导入 res_sampler: {e}")
    print("   请确保在正确的环境中运行（可能需要激活虚拟环境或使用 Docker）")
    sys.exit(1)

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 设置随机种子
torch.manual_seed(42)

# 1. 创建很小的测试张量（节省显存）
# B=1, C=2, T=2, H=8, W=8 - 非常小的尺寸
B, C, T, H, W = 1, 2, 2, 8, 8
x_sigma_max = torch.zeros((B, C, T, H, W), dtype=torch.float32, device=device)

print(f"测试张量形状: {x_sigma_max.shape}")
print(f"显存占用: {x_sigma_max.numel() * 4 / 1024 / 1024:.2f} MB")

# 2. 创建一个简单的 denoiser 函数（确保对输入有梯度依赖）
# 这个函数应该模拟真实的 denoiser，但非常简单
# 使用可学习的参数，确保梯度流
a = torch.tensor(0.7, dtype=torch.float32, device=device, requires_grad=False)
b = torch.tensor(0.1, dtype=torch.float32, device=device, requires_grad=False)

def simple_denoiser(x, sigma):
    """
    简单的 denoiser：x0 = a * x + b
    确保输出依赖于输入，这样梯度可以传递
    """
    # x: [B, C, T, H, W], sigma: [B] 或标量（这里不使用 sigma）
    return a * x + b

# 3. 配置 Sampler
# 使用很少的步数以减少计算量
solver_cfg = SolverConfig(
    is_multi=False,  # 使用 Runge-Kutta 方法
    rk="1euler",     # 最简单的 Euler 方法
    multistep="2ab",
    s_churn=0.0,     # 关闭随机噪声注入（或者测试 learnable_step_noise）
    s_t_max=float("inf"),
    s_t_min=0.05,
    s_noise=1.0
)

# 设置 learn_noise
setattr(solver_cfg, "learn_noise", True)

# 可选：测试 learnable_step_noise
# 如果设置了环境变量 LEARNABLE_STEP_NOISE=1，会启用可学习的每步噪声
use_learnable_step_noise = os.getenv("LEARNABLE_STEP_NOISE", "0") == "1"
if use_learnable_step_noise:
    setattr(solver_cfg, "learnable_step_noise", True)
    print("✅ 启用 learnable_step_noise（从 z0 派生每步噪声）")
else:
    setattr(solver_cfg, "learnable_step_noise", False)
    print("ℹ️  使用默认模式（关闭随机噪声注入）")

ts_cfg = SolverTimestampConfig(
    nfe=4,      # 只使用 4 步（减少计算量）
    t_min=0.5,
    t_max=5.0,
    order=3.0,
    is_forward=False
)
sampler_cfg = SamplerConfig(solver=solver_cfg, timestamps=ts_cfg, sample_clean=False)

sampler = Sampler(sampler_cfg).to(device)

print("\n" + "="*60)
print("开始测试梯度反传...")
print("="*60)

# 4. 前向传播（带梯度）
# 确保梯度计算是开启的
torch.enable_grad()

print("\n1. 运行前向传播（带梯度）...")
with torch.set_grad_enabled(True):
    output = sampler._forward_impl(simple_denoiser, x_sigma_max, sampler_cfg)

print(f"   输出形状: {output.shape}")
print(f"   output.requires_grad: {output.requires_grad}")

# 5. 检查 GLOBAL_Z0_ANCHOR 是否设置
print("\n2. 检查 GLOBAL_Z0_ANCHOR...")
GLOBAL_Z0_ANCHOR = getattr(_rs, "GLOBAL_Z0_ANCHOR", None)
if GLOBAL_Z0_ANCHOR is not None:
    print(f"   ✅ GLOBAL_Z0_ANCHOR 已设置")
    print(f"   形状: {GLOBAL_Z0_ANCHOR.shape}")
    print(f"   requires_grad: {GLOBAL_Z0_ANCHOR.requires_grad}")
else:
    print("   ❌ GLOBAL_Z0_ANCHOR 未设置！")
    print("   请检查 res_sampler 的代码是否正确设置了这个变量")
    exit(1)

# 6. 计算 loss 并反向传播
print("\n3. 计算 loss 并反向传播...")
target = torch.zeros_like(output)
loss = F.mse_loss(output, target)
print(f"   Loss: {loss.item():.6f}")

# 使用 -loss 进行梯度上升（最大化 loss）
(-loss).backward()
print("   ✅ 反向传播完成")

# 7. 检查梯度
print("\n4. 检查梯度...")
if GLOBAL_Z0_ANCHOR.grad is not None:
    grad_mean = GLOBAL_Z0_ANCHOR.grad.abs().mean().item()
    grad_max = GLOBAL_Z0_ANCHOR.grad.abs().max().item()
    grad_min = GLOBAL_Z0_ANCHOR.grad.abs().min().item()
    
    print("   " + "="*50)
    print("   ✅ 梯度成功反传到 z0！")
    print("   " + "="*50)
    print(f"   z0 grad mean:  {grad_mean:.6e}")
    print(f"   z0 grad max:   {grad_max:.6e}")
    print(f"   z0 grad min:   {grad_min:.6e}")
    
    if grad_mean > 0:
        print("\n   ✅✅✅ 测试通过！梯度成功反传到 learnable noise！")
    else:
        print("\n   ⚠️  警告：梯度均值接近 0，可能有问题")
else:
    print("   " + "="*50)
    print("   ❌ 没有梯度！")
    print("   " + "="*50)
    print("   可能的原因：")
    print("   1. res_sampler._forward_impl 中的 allow_grad=False")
    print("   2. z0.requires_grad_(True) 未调用")
    print("   3. 前向传播不在 torch.set_grad_enabled(True) 中")
    print("   4. 如果使用了 learnable_step_noise，检查是否正确设置")
    exit(1)

# 8. 测试优化器更新
print("\n5. 测试优化器更新...")
if GLOBAL_Z0_ANCHOR.grad is not None:
    # 保存原始值
    z0_before = GLOBAL_Z0_ANCHOR.clone().detach()
    
    # 创建优化器并更新
    opt = torch.optim.Adam([GLOBAL_Z0_ANCHOR], lr=0.01)
    opt.step()
    opt.zero_grad()
    
    # 检查是否更新
    z0_after = GLOBAL_Z0_ANCHOR.clone().detach()
    diff = (z0_after - z0_before).abs().mean().item()
    
    print(f"   z0 更新量 (mean abs diff): {diff:.6e}")
    if diff > 0:
        print("   ✅ 优化器成功更新了 z0")
    else:
        print("   ⚠️  警告：z0 没有更新")

print("\n" + "="*60)
print("测试完成！")
print("="*60)

