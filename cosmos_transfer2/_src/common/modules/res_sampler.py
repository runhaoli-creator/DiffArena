# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A general framework for various sampling algorithm from a diffusion model.
Impl based on
* Refined Exponential Solver (RES) in https://arxiv.org/pdf/2308.02157
* also clude other impl, DDIM, DEIS, DPM-Solver, EDM sampler.
Most of sampling algorihtm, Runge-Kutta, Multi-step, etc, can be impl in this framework by \
    adding new step function in get_runge_kutta_fn or get_multi_step_fn.
"""

import math
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import attrs
import torch
import torch.nn.functional as F

from cosmos_transfer2._src.common.functional.multi_step import get_multi_step_fn, is_multi_step_fn_supported
from cosmos_transfer2._src.common.functional.runge_kutta import get_runge_kutta_fn, is_runge_kutta_fn_supported
from cosmos_transfer2._src.imaginaire.config import make_freezable
from cosmos_transfer2._src.imaginaire.utils import log

# === Learnable-noise support (ADD) ===========================================
import os

# 外部可通过环境变量控制是否学习噪声（也可以稍后从 params/solver_cfg 里传递）
LEARN_NOISE_ENV = os.getenv("LEARN_NOISE", "0") == "1"
# 控制是否使用可学习的每步噪声（从 z0 派生，保持梯度流）
LEARNABLE_STEP_NOISE_ENV = os.getenv("LEARNABLE_STEP_NOISE", "0") == "1"

# 让外层能拿到 z0 的引用（比如在 control2world.infer 里做 backward/opt.step）
GLOBAL_Z0_ANCHOR = None
# ============================================================================ 


COMMON_SOLVER_OPTIONS = Literal["2ab", "2mid", "1euler"]


@make_freezable
@attrs.define(slots=False)
class SolverConfig:
    is_multi: bool = False
    rk: str = "2mid"
    multistep: str = "2ab"
    # following parameters control stochasticity, see EDM paper
    # BY default, we use deterministic with no stochasticity
    s_churn: float = 0.0
    s_t_max: float = float("inf")
    s_t_min: float = 0.05
    s_noise: float = 1.0


@make_freezable
@attrs.define(slots=False)
class SolverTimestampConfig:
    nfe: int = 50
    t_min: float = 0.002
    t_max: float = 80.0
    order: float = 7.0
    is_forward: bool = False  # whether generate forward or backward timestamps


@make_freezable
@attrs.define(slots=False)
class SamplerConfig:
    solver: SolverConfig = attrs.field(factory=SolverConfig)
    timestamps: SolverTimestampConfig = attrs.field(factory=SolverTimestampConfig)
    sample_clean: bool = True  # whether run one last step to generate clean image


def get_rev_ts(
    t_min: float, t_max: float, num_steps: int, ts_order: Union[int, float], is_forward: bool = False
) -> torch.Tensor:
    """
    Generate a sequence of reverse time steps.

    Args:
        t_min (float): The minimum time value.
        t_max (float): The maximum time value.
        num_steps (int): The number of time steps to generate.
        ts_order (Union[int, float]): The order of the time step progression.
        is_forward (bool, optional): If True, returns the sequence in forward order. Defaults to False.

    Returns:
        torch.Tensor: A tensor containing the generated time steps in reverse or forward order.

    Raises:
        ValueError: If `t_min` is not less than `t_max`.
        TypeError: If `ts_order` is not an integer or float.
    """
    if t_min >= t_max:
        raise ValueError("t_min must be less than t_max")

    if not isinstance(ts_order, (int, float)):
        raise TypeError("ts_order must be an integer or float")

    step_indices = torch.arange(num_steps + 1, dtype=torch.float64)
    time_steps = (
        t_max ** (1 / ts_order) + step_indices / num_steps * (t_min ** (1 / ts_order) - t_max ** (1 / ts_order))
    ) ** ts_order

    if is_forward:
        return time_steps.flip(dims=(0,))

    return time_steps


class Sampler(torch.nn.Module):
    def __init__(self, cfg: Optional[SamplerConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = SamplerConfig()
        self.cfg = cfg

    #@torch.no_grad()
    def forward(
        self,
        x0_fn: Callable,
        x_sigma_max: torch.Tensor,
        num_steps: int = 35,
        sigma_min: float = 0.002,
        sigma_max: float = 80,
        rho: float = 7,
        S_churn: float = 0,
        S_min: float = 0,
        S_max: float = float("inf"),
        S_noise: float = 1,
        solver_option: str = "2ab",
    ) -> torch.Tensor:

         # 允许通过 solver_cfg.learn_noise 或环境变量开启梯度
        allow_grad = True  # 默认 True 更方便你现在调试；上线可改 False
        # allow_grad = bool(getattr(self.cfg.solver, "learn_noise", False) or LEARN_NOISE_ENV)

        in_dtype = x_sigma_max.dtype

        def float64_x0_fn(x_B_StateShape: torch.Tensor, t_B: torch.Tensor) -> torch.Tensor:
            return x0_fn(x_B_StateShape.to(in_dtype), t_B.to(in_dtype)).to(torch.float64)

        is_multistep = is_multi_step_fn_supported(solver_option)
        is_rk = is_runge_kutta_fn_supported(solver_option)
        assert is_multistep or is_rk, f"Only support multistep or Runge-Kutta method, got {solver_option}"

        solver_cfg = SolverConfig(
            s_churn=S_churn,
            s_t_max=S_max,
            s_t_min=S_min,
            s_noise=S_noise,
            is_multi=is_multistep,
            rk=solver_option,
            multistep=solver_option,
        )
        # 如果 allow_grad=True，设置 learn_noise=True
        # 可以选择是否使用 learnable_step_noise（从 z0 派生每步噪声，保持梯度流）
        if allow_grad:
            setattr(solver_cfg, "learn_noise", True)
            # 默认关闭 learnable_step_noise，使用传统的关闭随机噪声注入方式
            # 如果需要保持随机噪声注入但又要梯度流，设置 learnable_step_noise=True
            # 可以通过环境变量 LEARNABLE_STEP_NOISE=1 来启用
            if not hasattr(solver_cfg, "learnable_step_noise"):
                learnable_step_noise_env = os.getenv("LEARNABLE_STEP_NOISE", "0") == "1"
                setattr(solver_cfg, "learnable_step_noise", learnable_step_noise_env)
        timestamps_cfg = SolverTimestampConfig(nfe=num_steps, t_min=sigma_min, t_max=sigma_max, order=rho)
        sampler_cfg = SamplerConfig(solver=solver_cfg, timestamps=timestamps_cfg, sample_clean=True)

        return self._forward_impl(float64_x0_fn, x_sigma_max, sampler_cfg).to(in_dtype)

    #@torch.no_grad()
    def _forward_impl(
        self,
        denoiser_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        noisy_input_B_StateShape: torch.Tensor,
        sampler_cfg: Optional[SamplerConfig] = None,
        callback_fns: Optional[List[Callable]] = None,
    ) -> torch.Tensor:
        """
        Internal implementation of the forward pass.

        Args:
            denoiser_fn: Function to denoise the input.
            noisy_input_B_StateShape: Input tensor with noise.
            sampler_cfg: Configuration for the sampler.
            callback_fns: List of callback functions to be called during sampling.

        Returns:
            torch.Tensor: Denoised output tensor.
        """
        # 与 forward 一致的 allow_grad 开关（可以只从 sampler_cfg.solver.learn_noise 或环境获取）
        allow_grad = True  # 调试阶段默认开梯度
        # allow_grad = bool(getattr(sampler_cfg.solver, "learn_noise", False) or LEARN_NOISE_ENV)

        sampler_cfg = self.cfg if sampler_cfg is None else sampler_cfg
        solver_order = 1 if sampler_cfg.solver.is_multi else int(sampler_cfg.solver.rk[0])
        num_timestamps = sampler_cfg.timestamps.nfe // solver_order

        sigmas_L = get_rev_ts(
            sampler_cfg.timestamps.t_min, sampler_cfg.timestamps.t_max, num_timestamps, sampler_cfg.timestamps.order
        ).to(noisy_input_B_StateShape.device)

        # === Learnable z0 (ADD) =============================================
        # 条件：当 allow_grad=True 或 solver.learn_noise=True 时启用可学习噪声
        if allow_grad or getattr(sampler_cfg.solver, "learn_noise", False) or LEARN_NOISE_ENV:
            sigma_max_0 = sigmas_L[0]
            
            # 检查是否已经存在可学习的noise（从外部传入）
            import sys
            current_module = sys.modules[__name__]
            if hasattr(current_module, 'GLOBAL_Z0_ANCHOR') and current_module.GLOBAL_Z0_ANCHOR is not None:
                # 使用外部提供的learnable noise
                z0 = current_module.GLOBAL_Z0_ANCHOR
                # 确保形状匹配
                if z0.shape != noisy_input_B_StateShape.shape:
                    log.warning(f"GLOBAL_Z0_ANCHOR shape {z0.shape} doesn't match input shape {noisy_input_B_StateShape.shape}, resizing...")
                    # Handle different dimensional cases
                    if len(z0.shape) == 5:  # (B, C, T, H, W)
                        z0 = F.interpolate(z0, size=noisy_input_B_StateShape.shape[2:], mode='trilinear', align_corners=False)
                    elif len(z0.shape) == 4:  # (B, C, H, W)
                        z0 = F.interpolate(z0, size=noisy_input_B_StateShape.shape[2:], mode='bilinear', align_corners=False)
                    else:
                        # Fallback: reshape or pad
                        if z0.numel() == noisy_input_B_StateShape.numel():
                            z0 = z0.reshape(noisy_input_B_StateShape.shape)
                        else:
                            log.error(f"Cannot reshape z0 from {z0.shape} to {noisy_input_B_StateShape.shape}")
                            z0 = torch.randn_like(noisy_input_B_StateShape)
                # 确保requires_grad
                if not z0.requires_grad:
                    z0 = z0.requires_grad_(True)
                    current_module.GLOBAL_Z0_ANCHOR = z0
            else:
                # 以传入的 noisy_input 的形状为准，初始化 z0 为标准正态
                z0 = torch.randn_like(noisy_input_B_StateShape)
                z0.requires_grad_(True)
                # 暴露给外层，便于读取梯度 / 用 optimizer 更新
                current_module.GLOBAL_Z0_ANCHOR = z0
            
            input_xT_B_StateShape = sigma_max_0 * z0  # x_T = sigma_max * z0
            
            # 如果允许每步噪声注入但仍然保持梯度流，预生成所有步骤的噪声
            # 关键：使用 z0 的可微分变换来生成每步噪声，而不是独立的随机噪声
            # 这样梯度可以从每步噪声传递回 z0
            use_learnable_step_noise = getattr(sampler_cfg.solver, "learnable_step_noise", False)
            if use_learnable_step_noise:
                # 预生成所有步骤的噪声，通过 z0 的可微分变换来生成（保持梯度连接）
                num_steps = len(sigmas_L) - 1
                step_noise_list = []
                for step_idx in range(num_steps):
                    # 方法：使用 z0 的可微分变换来生成每步噪声
                    # 1. 使用 z0 的缩放和位移（保持可微分）
                    scale = 1.0 + 0.1 * (step_idx / num_steps)  # 每步略有不同
                    # 2. 使用 z0 的周期性变换（sin/cos，保持可微分）
                    phase = step_idx * 0.1  # 每步的相位偏移
                    step_noise = z0 * scale + torch.sin(z0 * phase + step_idx) * 0.1
                    # 3. 添加 z0 的旋转/翻转（通过可微分的矩阵变换）
                    # 使用简单的线性组合确保梯度流
                    step_noise = step_noise + torch.cos(z0 * phase) * 0.05
                    step_noise_list.append(step_noise)
                # 存储到全局变量供 step_fn 使用
                current_module.GLOBAL_STEP_NOISE_LIST = step_noise_list
                log.debug(f"Generated {num_steps} learnable step noises from z0 (gradient-preserving)")
            else:
                current_module.GLOBAL_STEP_NOISE_LIST = None

            start_state = input_xT_B_StateShape
        else:
            # 保持旧行为：使用上游传入的初始噪声张量
            start_state = noisy_input_B_StateShape
            import sys
            current_module = sys.modules[__name__]
            current_module.GLOBAL_STEP_NOISE_LIST = None
        # ====================================================================

        # 用上面的 start_state 进入采样
        denoised_output = differential_equation_solver(
            denoiser_fn, sigmas_L, sampler_cfg.solver, callback_fns=callback_fns
        )(start_state)


        if sampler_cfg.sample_clean:
            # Override denoised_output with fully denoised version
            ones = torch.ones(denoised_output.size(0), device=denoised_output.device, dtype=denoised_output.dtype)
            denoised_output = denoiser_fn(denoised_output, sigmas_L[-1] * ones)

        return denoised_output


def fori_loop(lower: int, upper: int, body_fun: Callable[[int, Any], Any], init_val: Any) -> Any:
    """
    Implements a for loop with a function.

    Args:
        lower: Lower bound of the loop (inclusive).
        upper: Upper bound of the loop (exclusive).
        body_fun: Function to be applied in each iteration.
        init_val: Initial value for the loop.

    Returns:
        The final result after all iterations.
    """
    val = init_val
    for i in range(lower, upper):
        # Add log during sampling to meet APS job health requirement of one log every 2mins
        if i % 10 == 0:
            log.info(f"fori_loop: {i}")
        val = body_fun(i, val)
    return val


def differential_equation_solver(
    x0_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sigmas_L: torch.Tensor,
    solver_cfg: SolverConfig,
    callback_fns: Optional[List[Callable]] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates a differential equation solver function.

    Args:
        x0_fn: Function to compute x0 prediction.
        sigmas_L: Tensor of sigma values with shape [L,].
        solver_cfg: Configuration for the solver.
        callback_fns: Optional list of callback functions.

    Returns:
        A function that solves the differential equation.
    """
    num_step = len(sigmas_L) - 1

    if solver_cfg.is_multi:
        update_step_fn = get_multi_step_fn(solver_cfg.multistep)
    else:
        update_step_fn = get_runge_kutta_fn(solver_cfg.rk)

    eta = min(solver_cfg.s_churn / (num_step + 1), math.sqrt(1.2) - 1)

    def sample_fn(input_xT_B_StateShape: torch.Tensor) -> torch.Tensor:
        """
        Samples from the differential equation.

        Args:
            input_xT_B_StateShape: Input tensor with shape [B, StateShape].

        Returns:
            Output tensor with shape [B, StateShape].
        """
        ones_B = torch.ones(input_xT_B_StateShape.size(0), device=input_xT_B_StateShape.device, dtype=torch.float64)

        def step_fn(
            i_th: int, state: Tuple[torch.Tensor, Optional[List[torch.Tensor]]]
        ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
            input_x_B_StateShape, x0_preds = state
            sigma_cur_0, sigma_next_0 = sigmas_L[i_th], sigmas_L[i_th + 1]

            # algorithm 2: line 4-6
            if solver_cfg.s_t_min < sigma_cur_0 < solver_cfg.s_t_max:
                hat_sigma_cur_0 = sigma_cur_0 + eta * sigma_cur_0

                # 保证 solver_cfg 有 learn_noise 字段
                if not hasattr(solver_cfg, "learn_noise"):
                    setattr(solver_cfg, "learn_noise", False)

                # 检查是否使用可学习的每步噪声（从 z0 派生，保持梯度流）
                use_learnable_step_noise = getattr(solver_cfg, "learnable_step_noise", False)
                
                # 如果启用 learnable_step_noise，使用从 z0 派生的噪声（保持梯度流）
                if use_learnable_step_noise and solver_cfg.s_noise > 0.0:
                    import sys
                    current_module = sys.modules[__name__]
                    if hasattr(current_module, "GLOBAL_STEP_NOISE_LIST") and current_module.GLOBAL_STEP_NOISE_LIST is not None:
                        # 使用预生成的从 z0 派生的噪声
                        step_noise = current_module.GLOBAL_STEP_NOISE_LIST[i_th]
                        noise_scale = (hat_sigma_cur_0**2 - sigma_cur_0**2).sqrt() * solver_cfg.s_noise
                        input_x_B_StateShape = input_x_B_StateShape + noise_scale * step_noise
                    else:
                        # 回退到关闭随机噪声注入
                        pass  # 不添加噪声
                # 如果 learn_noise=True 但 learnable_step_noise=False，关闭随机噪声注入
                elif solver_cfg.learn_noise and not use_learnable_step_noise:
                # 学习噪声阶段：关闭每步新随机注入，避免稀释梯度到 z0 的路径
                    pass  # 不添加随机噪声
                # 正常情况：使用随机噪声注入
                elif not solver_cfg.learn_noise and solver_cfg.s_noise > 0.0:
                    input_x_B_StateShape = input_x_B_StateShape + (
                        hat_sigma_cur_0**2 - sigma_cur_0**2
                    ).sqrt() * solver_cfg.s_noise * torch.randn_like(input_x_B_StateShape)

                sigma_cur_0 = hat_sigma_cur_0


            if solver_cfg.is_multi:
                x0_pred_B_StateShape = x0_fn(input_x_B_StateShape, sigma_cur_0 * ones_B)
                output_x_B_StateShape, x0_preds = update_step_fn(
                    input_x_B_StateShape, sigma_cur_0 * ones_B, sigma_next_0 * ones_B, x0_pred_B_StateShape, x0_preds
                )
            else:
                output_x_B_StateShape, x0_preds = update_step_fn(
                    input_x_B_StateShape, sigma_cur_0 * ones_B, sigma_next_0 * ones_B, x0_fn
                )

            if callback_fns:
                for callback_fn in callback_fns:
                    callback_fn(**locals())

            return output_x_B_StateShape, x0_preds

        x_at_eps, _ = fori_loop(0, num_step, step_fn, [input_xT_B_StateShape, None])
        return x_at_eps

    return sample_fn
