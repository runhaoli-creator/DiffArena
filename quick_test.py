#!/usr/bin/env python3
"""
快速测试脚本 - 验证代码能否运行并生成结果
使用最小配置进行快速测试
"""

import sys
import os
from pathlib import Path

# 添加路径
COSMOS_ROOT = Path(__file__).parent
sys.path.insert(0, str(COSMOS_ROOT))
DIFFUSION_POLICY_ROOT = COSMOS_ROOT / "diffusion_policy"
if DIFFUSION_POLICY_ROOT.exists():
    sys.path.insert(0, str(DIFFUSION_POLICY_ROOT))

import torch
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2.inference import Control2WorldInference, SetupArguments, InferenceArguments
from cosmos_transfer2.config import path_to_str, EdgeConfig

# 设置HuggingFace token
HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ["LEARN_NOISE"] = "1"

# 设置使用GPU 1和2
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

def main():
    print("=" * 60)
    print("快速测试 - 验证代码能否运行")
    print("=" * 60)
    
    # 检查GPU
    print(f"\n使用GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    if torch.cuda.is_available():
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️ CUDA不可用")
        return
    
    # 检查输入视频
    input_video = "assets/robot_example/robot_input.mp4"
    if not Path(input_video).exists():
        print(f"⚠️ 输入视频不存在: {input_video}")
        print("请提供有效的输入视频路径")
        return
    
    print(f"\n✓ 输入视频: {input_video}")
    
    # 初始化Cosmos Transfer（最小配置）
    print("\n初始化 Cosmos Transfer2.5...")
    try:
        output_dir = Path("./test_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        setup_args = SetupArguments(
            output_dir=output_dir,
            model="edge",  # 使用edge模型
            context_parallel_size=1,
            keep_going=True,
        )
        
        cosmos_inference = Control2WorldInference(
            args=setup_args,
            batch_hint_keys=["edge"],
        )
        print("✓ Cosmos Transfer2.5 初始化成功")
    except Exception as e:
        print(f"✗ Cosmos Transfer2.5 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试生成一个简单的视频（不运行完整优化）
    print("\n测试视频生成...")
    try:
        sample_args = InferenceArguments(
            name="quick_test",
            video_path=input_video,
            prompt="A robot arm manipulating objects on a table",
            negative_prompt="",
            edge=EdgeConfig(control_weight=1.0),  # 提供edge配置
            guidance=7,
            seed=42,
            resolution="480",  # 使用较低分辨率加快速度
            num_steps=10,  # 减少步数加快速度
            num_conditional_frames=1,
            num_video_frames_per_chunk=10,  # 减少帧数
        )
        
        print("开始生成视频...")
        # 使用generate方法（wrapper，会自动处理no_grad）
        # 如果negative_prompt为空，设置为None
        negative_prompt = sample_args.negative_prompt if sample_args.negative_prompt else None
        
        # 获取control_modalities，如果所有值都是None，则传入空字典
        control_modalities = sample_args.control_modalities
        if control_modalities and not any(v for v in control_modalities.values() if v):
            control_modalities = {}
        
        output_video, control_video_dict, fps, _ = (
            cosmos_inference.inference_pipeline.generate_img2world(
                video_path=path_to_str(sample_args.video_path),
                prompt=sample_args.prompt,
                negative_prompt=negative_prompt,
                guidance=sample_args.guidance,
                seed=sample_args.seed,
                resolution=sample_args.resolution,
                control_weight=",".join([str(sample_args.control_weight_dict.get(k, 1.0)) for k in sample_args.hint_keys]),
                hint_key=sample_args.hint_keys,
                input_control_video_paths=control_modalities if control_modalities else {},
                num_steps=sample_args.num_steps,
                num_conditional_frames=sample_args.num_conditional_frames,
                num_video_frames_per_chunk=sample_args.num_video_frames_per_chunk,
            )
        )
        
        print(f"✓ 视频生成成功!")
        print(f"  输出形状: {output_video.shape}")
        print(f"  FPS: {fps}")
        
        # 保存测试结果
        output_path = output_dir / "quick_test_output.pt"
        torch.save(output_video.cpu(), output_path)
        print(f"  保存到: {output_path}")
        
    except Exception as e:
        print(f"✗ 视频生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("✅ 快速测试完成！代码可以正常运行。")
    print("=" * 60)
    print("\n如果要运行完整的对抗性攻击，需要：")
    print("1. Diffusion Policy checkpoint")
    print("2. Diffusion Policy config文件")
    print("3. 数据集路径")
    print("\n然后运行: python adversarial_attack.py [参数]")

if __name__ == "__main__":
    main()

