#!/usr/bin/env python3
"""快速测试checkpoint加载"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "diffusion_policy"))

import torch
import dill

checkpoint_path = "diffusion_policy/data/outputs/2025.10.11/19.23.26_train_diffusion_unet_hybrid_pusht_image/checkpoints/epoch=0400-test_mean_score=0.901.ckpt"

print("Loading checkpoint...")
payload = torch.load(open(checkpoint_path, 'rb'), map_location='cpu', pickle_module=dill)

print(f"Keys in checkpoint: {list(payload.keys())}")
if 'cfg' in payload:
    cfg = payload['cfg']
    print(f"Config _target_: {cfg._target_}")
    print(f"Policy _target_: {cfg.policy._target_}")
    print(f"Has state_dicts: {'state_dicts' in payload}")
    if 'state_dicts' in payload:
        print(f"State dict keys: {list(payload['state_dicts'].keys())}")


