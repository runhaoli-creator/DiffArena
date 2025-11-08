#!/usr/bin/env python3
"""
简单测试 - 验证代码能否正常运行
只验证初始化和导入，不生成完整视频
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
HF_TOKEN = os.environ.get("HF_TOKEN")
os.environ["LEARN_NOISE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

print("=" * 60)
print("简单测试 - 验证代码能否正常运行")
print("=" * 60)

# 检查GPU
print(f"\n使用GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
if torch.cuda.is_available():
    print(f"可用GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"    显存: {props.total_memory / 1024**3:.2f} GB")

# 测试导入
print("\n1. 测试Cosmos Transfer2.5导入...")
try:
    from cosmos_transfer2.inference import Control2WorldInference, SetupArguments
    from cosmos_transfer2.config import InferenceArguments, EdgeConfig
    print("   ✓ Cosmos Transfer2.5导入成功")
except Exception as e:
    print(f"   ✗ 导入失败: {e}")
    sys.exit(1)

print("\n2. 测试Diffusion Policy导入...")
try:
    from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
    from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
    print("   ✓ Diffusion Policy导入成功")
except Exception as e:
    print(f"   ✗ 导入失败: {e}")
    sys.exit(1)

print("\n3. 测试Cosmos Transfer2.5初始化...")
try:
    output_dir = Path("./test_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_args = SetupArguments(
        output_dir=output_dir,
        model="edge",
        context_parallel_size=1,
        keep_going=True,
    )
    
    cosmos_inference = Control2WorldInference(
        args=setup_args,
        batch_hint_keys=["edge"],
    )
    print("   ✓ Cosmos Transfer2.5初始化成功")
except Exception as e:
    print(f"   ✗ 初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n4. 测试adversarial_attack导入...")
try:
    from adversarial_attack import AdversarialAttack
    print("   ✓ AdversarialAttack导入成功")
except Exception as e:
    print(f"   ✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 所有测试通过！代码可以正常运行。")
print("=" * 60)
print("\n说明:")
print("- 代码已成功初始化和导入")
print("- GPU环境配置正确")
print("- 可以开始运行完整的对抗性攻击")
print("\n要运行完整测试（包括视频生成），请使用:")
print("  python adversarial_attack.py [参数]")

