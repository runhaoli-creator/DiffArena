#!/usr/bin/env python3
"""
测试统一环境是否配置正确
Test if the unified environment is configured correctly
"""

import sys
import os

def test_import(module_name, display_name=None):
    """测试导入模块"""
    if display_name is None:
        display_name = module_name
    try:
        __import__(module_name)
        print(f"✓ {display_name} imported successfully")
        return True
    except Exception as e:
        print(f"✗ {display_name} import failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Testing Unified Environment")
    print("=" * 60)
    print()
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print()
    
    if sys.version_info < (3, 10):
        print("✗ Python 3.10+ is required")
        return False
    else:
        print("✓ Python version check passed")
    print()
    
    # Check PYTHONPATH
    pythonpath = os.environ.get('PYTHONPATH', '')
    print(f"PYTHONPATH: {pythonpath}")
    print()
    
    all_passed = True
    
    # Test Cosmos Transfer2.5 imports
    print("Testing Cosmos Transfer2.5 imports...")
    print("-" * 60)
    all_passed &= test_import("cosmos_transfer2", "Cosmos Transfer2.5")
    all_passed &= test_import("cosmos_transfer2.inference", "Cosmos Transfer2.5 inference")
    all_passed &= test_import("cosmos_transfer2.config", "Cosmos Transfer2.5 config")
    print()
    
    # Test Diffusion Policy imports
    print("Testing Diffusion Policy imports...")
    print("-" * 60)
    all_passed &= test_import("diffusion_policy", "Diffusion Policy")
    all_passed &= test_import("diffusion_policy.policy", "Diffusion Policy policy")
    all_passed &= test_import("diffusion_policy.dataset", "Diffusion Policy dataset")
    try:
        from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
        print("✓ DiffusionUnetImagePolicy imported successfully")
    except Exception as e:
        print(f"✗ DiffusionUnetImagePolicy import failed: {e}")
        all_passed = False
    try:
        from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
        print("✓ PushTImageDataset imported successfully")
    except Exception as e:
        print(f"✗ PushTImageDataset import failed: {e}")
        all_passed = False
    print()
    
    # Test PyTorch
    print("Testing PyTorch...")
    print("-" * 60)
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        all_passed = False
    print()
    
    # Test key dependencies
    print("Testing key dependencies...")
    print("-" * 60)
    test_import("diffusers", "Diffusers")
    test_import("einops", "Einops")
    test_import("hydra", "Hydra")
    test_import("wandb", "Wandb")
    test_import("zarr", "Zarr")
    test_import("cv2", "OpenCV")
    print()
    
    # Test environment variables
    print("Testing environment variables...")
    print("-" * 60)
    learn_noise = os.environ.get('LEARN_NOISE', '0')
    print(f"LEARN_NOISE: {learn_noise}")
    if learn_noise == '1':
        print("✓ LEARN_NOISE is enabled")
    else:
        print("⚠ LEARN_NOISE is not enabled (optional)")
    print()
    
    # Final result
    print("=" * 60)
    if all_passed:
        print("✅ All critical tests passed! Environment is ready.")
        print()
        print("You can now run:")
        print("  python adversarial_attack.py [arguments]")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print()
        print("Common fixes:")
        print("  1. Run: source .env  (or source .venv/bin/activate)")
        print("  2. Run: export PYTHONPATH=\"$(pwd):$(pwd)/diffusion_policy:\$PYTHONPATH\"")
        print("  3. Run: ./setup_unified_env.sh")
        return 1

if __name__ == "__main__":
    sys.exit(main())

