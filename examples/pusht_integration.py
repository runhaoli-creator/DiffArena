#!/usr/bin/env python3
"""
Minimal Push-T integration test:
1. Load a Push-T sample from Diffusion Policy dataset.
2. Export the observation frames to a temporary video file.
3. Run Cosmos Transfer2.5 inference on that video with gradient enabled.
4. Feed the generated video back into the Diffusion Policy model together with
   the ground-truth actions and print the resulting loss.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import imageio.v2 as imageio
import numpy as np
import torch

# Resolve repository root and add to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
COSMOS_ROOT = REPO_ROOT
DIFFUSION_POLICY_ROOT = REPO_ROOT / "diffusion_policy"

sys.path.insert(0, str(COSMOS_ROOT))
sys.path.insert(0, str(DIFFUSION_POLICY_ROOT))

from adversarial_attack import load_diffusion_policy
from cosmos_transfer2.config import EdgeConfig, path_to_str
from cosmos_transfer2.inference import Control2WorldInference, InferenceArguments, SetupArguments
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset


def convert_obs_to_video(
    images: torch.Tensor,
    output_path: Path,
    fps: int = 10,
) -> None:
    """
    Write observation frames to an mp4 file for Cosmos inference.
    Args:
        images: Tensor with shape (T, C, H, W) in [0, 1].
        output_path: Destination path for the mp4 file.
        fps: Frames per second used when saving the video.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = images.detach().cpu().numpy()  # (T, C, H, W)
    frames = np.clip(frames, 0.0, 1.0)
    frames = (frames * 255).astype(np.uint8)
    frames = np.transpose(frames, (0, 2, 3, 1))  # (T, H, W, C)
    imageio.mimwrite(output_path, frames, fps=fps)


def prepare_cosmos(
    output_dir: Path,
) -> Control2WorldInference:
    """
    Instantiate Cosmos Transfer2.5 inference pipeline with default Edge model.
    """
    setup_args = SetupArguments(
        output_dir=output_dir,
        model="edge",
        context_parallel_size=1,
        disable_guardrails=True,
        keep_going=True,
    )
    return Control2WorldInference(args=setup_args, batch_hint_keys=["edge"])


def run_cosmos_generation(
    cosmos: Control2WorldInference,
    sample_args: InferenceArguments,
) -> Tuple[torch.Tensor, int]:
    """
    Execute Cosmos inference with gradient enabled to obtain the generated video.
    Returns:
        generated_video: Tensor with shape (B, C, T, H, W) in [-1, 1].
        fps: Frames per second of the generated video.
    """
    kwargs = dict(
        video_path=path_to_str(sample_args.video_path),
        prompt=sample_args.prompt,
        negative_prompt=sample_args.negative_prompt,
        guidance=sample_args.guidance,
        seed=sample_args.seed,
        resolution=sample_args.resolution,
        control_weight=",".join(
            str(sample_args.control_weight_dict.get(k, 1.0))
            for k in sample_args.hint_keys
        ),
        sigma_max=sample_args.sigma_max,
        hint_key=sample_args.hint_keys,
        input_control_video_paths={
            k: v for k, v in sample_args.control_modalities.items() if v is not None
        },
        show_control_condition=sample_args.show_control_condition,
        seg_control_prompt=sample_args.seg_control_prompt,
        show_input=sample_args.show_input,
        keep_input_resolution=not sample_args.not_keep_input_resolution,
        preset_blur_strength=sample_args.preset_blur_strength,
        preset_edge_threshold=sample_args.preset_edge_threshold,
        num_conditional_frames=sample_args.num_conditional_frames,
        num_video_frames_per_chunk=sample_args.num_video_frames_per_chunk,
        num_steps=sample_args.num_steps,
    )

    with torch.set_grad_enabled(True):
        generated_video, _, fps, _ = cosmos.inference_pipeline._generate_img2world_impl(
            **kwargs
        )
    generated_video = generated_video.float()
    return generated_video, fps


def prepare_diffusion_policy(
    checkpoint_path: Path,
    dataset: PushTImageDataset,
) -> torch.nn.Module:
    """
    Load Diffusion Policy from checkpoint and attach dataset normalizer.
    """
    policy = load_diffusion_policy(str(checkpoint_path))
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)
    policy.eval()
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Push-T integration pipeline")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DIFFUSION_POLICY_ROOT / "data/pusht/pusht_cchi_v7_replay.zarr",
        help="Path to Push-T zarr dataset.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to Diffusion Policy checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Dataset index used for the integration test.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "outputs/pusht_integration",
        help="Directory used for temporary artifacts.",
    )
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default=None,
        help="Comma-separated list of GPU indices to expose (e.g. '1,2').",
    )
    args = parser.parse_args()

    os.environ.setdefault("LEARN_NOISE", "1")
    if args.cuda_devices:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    dataset_path = args.dataset.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")

    dataset = PushTImageDataset(
        zarr_path=str(dataset_path),
        horizon=16,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
    )
    sample = dataset[args.sample_index]
    obs_images = sample["obs"]["image"]  # (T, C, H, W), float32 in [0, 1]
    obs_agent_pos = sample["obs"]["agent_pos"]  # (T, 2)
    gt_actions = sample["action"]  # (T, 2)

    print(f"Loaded sample {args.sample_index}")
    print(f"  image shape: {tuple(obs_images.shape)}")
    print(f"  agent_pos shape: {tuple(obs_agent_pos.shape)}")
    print(f"  action shape: {tuple(gt_actions.shape)}")

    temp_video_path = output_dir / f"pusht_sample_{args.sample_index:04d}.mp4"
    convert_obs_to_video(obs_images, temp_video_path, fps=10)
    print(f"Wrote temporary video to {temp_video_path}")

    cosmos = prepare_cosmos(output_dir=output_dir / "cosmos_logs")
    sample_args = InferenceArguments(
        name=f"pusht_sample_{args.sample_index:04d}",
        video_path=temp_video_path,
        prompt="Push-T dataset sample rendered as video.",
        negative_prompt="",
        edge=EdgeConfig(control_weight=1.0),
        guidance=3,
        seed=2025 + args.sample_index,
        resolution="480",
        num_steps=20,
        num_conditional_frames=1,
        num_video_frames_per_chunk=obs_images.shape[0],
    )

    generated_video, fps = run_cosmos_generation(cosmos, sample_args)
    print(f"Cosmos output shape: {tuple(generated_video.shape)}, fps={fps}")

    dp_policy = prepare_diffusion_policy(checkpoint_path, dataset)

    # Convert Cosmos output to Diffusion Policy observation format
    dp_images = (generated_video + 1.0) / 2.0  # [-1,1] -> [0,1]
    dp_images = dp_images.clamp(0.0, 1.0)
    dp_images = dp_images.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)

    # Align sequence length with agent_pos
    target_len = obs_agent_pos.shape[0]
    if dp_images.shape[1] != target_len:
        if dp_images.shape[1] > target_len:
            dp_images = dp_images[:, :target_len]
        else:
            pad_frames = target_len - dp_images.shape[1]
            pad_tensor = dp_images[:, -1:].repeat(1, pad_frames, 1, 1, 1)
            dp_images = torch.cat([dp_images, pad_tensor], dim=1)

    obs_dict = {
        "image": dp_images,
        "agent_pos": obs_agent_pos.unsqueeze(0),
    }
    actions = gt_actions.unsqueeze(0)

    with torch.no_grad():
        pred_dict = dp_policy.predict_action(obs_dict)
        pred_actions = pred_dict["action"]

    Ta = pred_actions.shape[1]
    aligned_gt = actions[:, :Ta]
    loss = torch.nn.functional.mse_loss(pred_actions, aligned_gt)

    print(f"Predicted action shape: {tuple(pred_actions.shape)}")
    print(f"Ground truth (aligned) shape: {tuple(aligned_gt.shape)}")
    print(f"MSE loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()

