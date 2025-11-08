#!/usr/bin/env python3
"""
简化的对抗性攻击示例脚本
用于快速测试和演示
"""

import sys
from pathlib import Path

# 添加项目路径
COSMOS_ROOT = Path(__file__).parent
DIFFUSION_POLICY_ROOT = COSMOS_ROOT.parent / "diffusion_policy"
sys.path.insert(0, str(COSMOS_ROOT))
sys.path.insert(0, str(DIFFUSION_POLICY_ROOT))

from adversarial_attack import AdversarialAttack
from cosmos_transfer2.inference import Control2WorldInference, SetupArguments, InferenceArguments
from cosmos_transfer2.config import path_to_str
from cosmos_transfer2._src.imaginaire.utils import log


def main():
    """示例主函数"""
    
    # ===== 配置参数 =====
    # 请根据你的实际情况修改这些路径
    COSMOS_CHECKPOINT = "edge"  # 或实际的checkpoint路径
    INPUT_VIDEO = "assets/robot_example/robot_input.mp4"
    PROMPT = "A robot arm manipulating objects on a table"
    
    DIFFUSION_POLICY_CHECKPOINT = "path/to/diffusion_policy/checkpoint.ckpt"
    DIFFUSION_POLICY_CONFIG = "path/to/diffusion_policy/config.yaml"
    DATASET_PATH = "path/to/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr"
    
    OUTPUT_DIR = "./adversarial_output_example"
    NUM_ITERATIONS = 5  # 示例中减少迭代次数
    LR = 1e-3
    DEVICE = "cuda"
    
    # ===== 初始化Cosmos Transfer =====
    log.info("=" * 60)
    log.info("初始化 Cosmos Transfer2.5...")
    log.info("=" * 60)
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_args = SetupArguments(
        output_dir=output_dir,
        context_parallel_size=1,
        enable_guardrails=False,  # 禁用guardrails以加快速度
        enable_parallel_tokenizer=False,
    )
    
    cosmos_inference = Control2WorldInference(
        args=setup_args,
        batch_hint_keys=["edge"],  # 使用edge作为控制信号
    )
    
    # ===== 加载Diffusion Policy =====
    log.info("=" * 60)
    log.info("加载 Diffusion Policy...")
    log.info("=" * 60)
    
    try:
        from adversarial_attack import load_diffusion_policy
        diffusion_policy = load_diffusion_policy(
            DIFFUSION_POLICY_CHECKPOINT,
            DIFFUSION_POLICY_CONFIG
        )
    except Exception as e:
        log.error(f"加载Diffusion Policy失败: {e}")
        log.error("请确保checkpoint和config路径正确")
        return
    
    # ===== 加载数据集 =====
    log.info("=" * 60)
    log.info("加载数据集...")
    log.info("=" * 60)
    
    try:
        from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
        dataset = PushTImageDataset(DATASET_PATH, horizon=16)
        normalizer = dataset.get_normalizer()
        diffusion_policy.set_normalizer(normalizer)
        
        # 获取一个ground truth样本
        sample_batch = dataset[0]
        ground_truth_batch = {
            'obs': {k: v.unsqueeze(0) for k, v in sample_batch['obs'].items()},
            'action': sample_batch['action'].unsqueeze(0),
        }
        
        log.info(f"Ground truth action shape: {ground_truth_batch['action'].shape}")
        
    except Exception as e:
        log.error(f"加载数据集失败: {e}")
        log.error("请确保数据集路径正确")
        return
    
    # ===== 创建对抗性攻击实例 =====
    log.info("=" * 60)
    log.info("创建对抗性攻击实例...")
    log.info("=" * 60)
    
    attack = AdversarialAttack(
        cosmos_inference=cosmos_inference,
        diffusion_policy=diffusion_policy,
        ground_truth_batch=ground_truth_batch,
        device=DEVICE,
    )
    
    # ===== 创建推理参数 =====
    sample_args = InferenceArguments(
        name="example_adversarial_sample",
        video_path=INPUT_VIDEO,
        prompt=PROMPT,
        negative_prompt="",
        hint_keys=["edge"],
        control_weight_dict={"edge": "1.0"},
        guidance=7,
        seed=42,
        resolution="720",
        num_steps=35,
        num_conditional_frames=1,
        num_video_frames_per_chunk=93,
    )
    
    # ===== 运行优化 =====
    log.info("=" * 60)
    log.info("开始对抗性优化...")
    log.info("=" * 60)
    
    try:
        results = attack.optimize_noise(
            sample_args=sample_args,
            num_iterations=NUM_ITERATIONS,
            lr=LR,
            output_dir=output_dir / "iterations",
        )
        
        # ===== 保存结果 =====
        import torch
        results_path = output_dir / "optimization_results.pt"
        torch.save(results, results_path)
        log.info(f"优化结果已保存到: {results_path}")
        
        log.info("=" * 60)
        log.info("优化完成！")
        log.info("=" * 60)
        log.info(f"最佳Loss: {results['best_loss']:.6f}")
        log.info(f"初始Loss: {results['loss_history'][0] if results['loss_history'] else 'N/A'}")
        log.info(f"最终Loss: {results['loss_history'][-1] if results['loss_history'] else 'N/A'}")
        
    except Exception as e:
        log.error(f"优化过程出错: {e}")
        import traceback
        log.error(traceback.format_exc())


if __name__ == "__main__":
    main()

