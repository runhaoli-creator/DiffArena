import os
os.environ["LEARN_NOISE"] = "1"  # 再保险一遍（首选还是在 bash 里 export）

import torch
from cosmos_transfer2._src.common.modules.res_sampler import (
    Sampler, SamplerConfig, SolverConfig, SolverTimestampConfig, GLOBAL_Z0_ANCHOR
)

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) 小形状，跑得快
B, C, T, H, W = 1, 2, 2, 8, 8
x_sigma_max = torch.zeros((B, C, T, H, W), dtype=torch.float64, device=device)

# 2) 假 denoiser：x0 = a*x + b（确保对输入有梯度依赖）
a = torch.tensor(0.7, dtype=torch.float64, device=device)
b = torch.tensor(0.1, dtype=torch.float64, device=device)
def x0_fn(x, sigma):
    # x: [B,C,T,H,W], sigma: [B]（这里不用 sigma）
    return a * x + b

# 3) Sampler 配置：少步数；学习噪声时关闭每步新增随机注入（在你改过的 res_sampler 中由 learn_noise 控制）
solver_cfg = SolverConfig(
    is_multi=False, rk="1euler", multistep="2ab",
    s_churn=0.0, s_t_max=float("inf"), s_t_min=0.05, s_noise=1.0
)
setattr(solver_cfg, "learn_noise", True)  # 学习噪声：res_sampler.step_fn 内不再注入新随机噪声

ts_cfg = SolverTimestampConfig(nfe=4, t_min=0.5, t_max=5.0, order=3.0, is_forward=False)
sampler_cfg = SamplerConfig(solver=solver_cfg, timestamps=ts_cfg, sample_clean=False)

sampler = Sampler(sampler_cfg).to(device)

# 4) 前向（要有梯度，确保你已移除了 res_sampler 里的 @torch.no_grad 装饰器）
out = sampler._forward_impl(x0_fn, torch.randn_like(x_sigma_max), sampler_cfg)
target = torch.zeros_like(out)
loss = torch.nn.functional.mse_loss(out, target)
loss.backward()

assert GLOBAL_Z0_ANCHOR is not None, "GLOBAL_Z0_ANCHOR 未设置 —— 请检查 res_sampler 初始化 z0 的代码是否赋值到了全局变量"
print("z0.grad is None? ", GLOBAL_Z0_ANCHOR.grad is None)
if GLOBAL_Z0_ANCHOR.grad is not None:
    print("z0.grad mean abs:", GLOBAL_Z0_ANCHOR.grad.abs().mean().item())
else:
    raise RuntimeError("没有梯度回到 z0：1) res_sampler 里是否仍有 @torch.no_grad；2) step_fn 是否还在每步注入随机噪声；3) 初始噪声是否是 sigma_max * z0 且 z0.requires_grad_(True)")
