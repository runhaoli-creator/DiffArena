# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adversarial Attack Script: Optimize Cosmos Transfer2.5 noise to maximize Diffusion Policy action loss
#
# This script implements an adversarial attack where:
# 1. Cosmos Transfer2.5 generates videos from learnable initial noise
# 2. Generated video frames are fed to Diffusion Policy
# 3. Action loss between predicted and ground truth actions is computed
# 4. Loss is backpropagated to optimize the initial noise to maximize the loss

import sys
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import argparse
from tqdm import tqdm

# Add paths for imports
# In unified environment, both projects are in the same directory
COSMOS_ROOT = Path(__file__).parent
DIFFUSION_POLICY_ROOT = COSMOS_ROOT / "diffusion_policy"

# Try unified structure first, then fallback to separate structure
if DIFFUSION_POLICY_ROOT.exists():
    sys.path.insert(0, str(COSMOS_ROOT))
    sys.path.insert(0, str(DIFFUSION_POLICY_ROOT))
else:
    # Fallback: try parent directory (separate repos)
    DIFFUSION_POLICY_ROOT = COSMOS_ROOT.parent / "diffusion_policy"
    sys.path.insert(0, str(COSMOS_ROOT))
    if DIFFUSION_POLICY_ROOT.exists():
        sys.path.insert(0, str(DIFFUSION_POLICY_ROOT))

from cosmos_transfer2.inference import Control2WorldInference, SetupArguments, InferenceArguments
from cosmos_transfer2.config import path_to_str, EdgeConfig
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.common import presets as guardrail_presets

# Import Diffusion Policy components
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
from diffusion_policy.common.pytorch_util import dict_apply


class AdversarialAttack:
    """
    Adversarial attack that optimizes Cosmos Transfer2.5 initial noise
    to maximize Diffusion Policy action loss.
    """
    
    def __init__(
        self,
        cosmos_inference: Control2WorldInference,
        diffusion_policy: DiffusionUnetImagePolicy,
        ground_truth_batch: Dict[str, torch.Tensor],
        device: str = "cuda",
    ):
        """
        Args:
            cosmos_inference: Initialized Cosmos Transfer2.5 inference object
            diffusion_policy: Loaded Diffusion Policy model
            ground_truth_batch: Ground truth batch with 'obs' and 'action' keys
            device: Device to run on
        """
        self.cosmos_inference = cosmos_inference
        self.diffusion_policy = diffusion_policy
        self.ground_truth_batch = ground_truth_batch
        self.device = device
        
        # Move diffusion policy to device and set to eval mode
        self.diffusion_policy.to(device)
        self.diffusion_policy.eval()
        
        # Extract ground truth actions
        self.gt_actions = ground_truth_batch['action'].to(device)
        
        log.info(f"Initialized AdversarialAttack on device {device}")
        log.info(f"Ground truth action shape: {self.gt_actions.shape}")
    
    def video_frames_to_obs_dict(
        self, 
        video_tensor: torch.Tensor,
        target_shape: Tuple[int, int] = (96, 96),
        num_frames: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Convert Cosmos generated video tensor to Diffusion Policy observation format.
        
        Args:
            video_tensor: Video tensor from Cosmos (B, C, T, H, W) in range [-1, 1]
            target_shape: Target image size (H, W) for Diffusion Policy
            num_frames: Number of frames to use (None = use all)
            
        Returns:
            Dictionary with 'image' key containing (B, T, C, H, W) tensor in [0, 1]
        """
        # video_tensor is (B, C, T, H, W) in range [-1, 1]
        B, C, T, H, W = video_tensor.shape
        
        if num_frames is not None:
            T = min(T, num_frames)
            video_tensor = video_tensor[:, :, :T]
        
        # Convert from [-1, 1] to [0, 1]
        video_tensor = (video_tensor + 1.0) / 2.0
        
        # Reshape to (B, T, C, H, W)
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4)
        
        # Resize to target shape if needed
        if (H, W) != target_shape:
            video_tensor = F.interpolate(
                video_tensor.reshape(B * T, C, H, W),
                size=target_shape,
                mode='bilinear',
                align_corners=False
            ).reshape(B, T, C, *target_shape)
        
        # Ensure values are in [0, 1]
        video_tensor = torch.clamp(video_tensor, 0.0, 1.0)
        
        # Create obs dict (Diffusion Policy expects 'image' key)
        obs_dict = {
            'image': video_tensor
        }
        
        # If ground truth has 'agent_pos', we might need to include it
        # For now, we'll use the image only and let the policy handle missing state
        if 'agent_pos' in self.ground_truth_batch['obs']:
            # Use ground truth agent_pos for now (could be made learnable too)
            obs_dict['agent_pos'] = self.ground_truth_batch['obs']['agent_pos'].to(self.device)
        
        return obs_dict
    
    def compute_action_loss(
        self,
        generated_video: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute action loss between Diffusion Policy predictions and ground truth.
        
        Args:
            generated_video: Generated video tensor from Cosmos (B, C, T, H, W)
            
        Returns:
            loss: Action loss (scalar tensor)
            metrics: Dictionary with additional metrics
        """
        # Convert video to observation format
        obs_dict = self.video_frames_to_obs_dict(generated_video)
        
        # Get Diffusion Policy action predictions
        with torch.enable_grad():
            # Ensure gradients can flow through the policy
            pred_dict = self.diffusion_policy.predict_action(obs_dict)
            pred_actions = pred_dict['action']  # (B, Ta, Da)
        
        # Align dimensions: ground truth might be (B, T, Da), pred is (B, Ta, Da)
        # Use the first Ta steps of ground truth
        Ta = pred_actions.shape[1]
        gt_actions_aligned = self.gt_actions[:, :Ta]
        
        # Compute MSE loss
        action_loss = F.mse_loss(pred_actions, gt_actions_aligned)
        
        # Additional metrics
        metrics = {
            'action_loss': action_loss.item(),
            'pred_actions': pred_actions.detach().cpu(),
            'gt_actions': gt_actions_aligned.detach().cpu(),
            'action_diff': (pred_actions - gt_actions_aligned).abs().mean().item(),
        }
        
        return action_loss, metrics
    
    def optimize_noise(
        self,
        sample_args: InferenceArguments,
        num_iterations: int = 10,
        lr: float = 1e-3,
        output_dir: Optional[Path] = None,
    ) -> Dict:
        """
        Optimize initial noise to maximize action loss.
        
        Args:
            sample_args: Inference arguments for Cosmos Transfer
            num_iterations: Number of optimization iterations
            lr: Learning rate for noise optimization
            output_dir: Directory to save intermediate results
            
        Returns:
            Dictionary with optimization results
        """
        log.info("=" * 60)
        log.info("Starting Adversarial Noise Optimization")
        log.info("=" * 60)
        log.info(f"Iterations: {num_iterations}")
        log.info(f"Learning rate: {lr}")
        
        # Setup optimizer for learnable noise
        # We'll need to access the noise from the res_sampler
        try:
            from cosmos_transfer2._src.common.modules import res_sampler as _rs
        except Exception:
            from cosmos_transfer2._src.predict2.samplers import res_sampler as _rs
        
        # Enable learnable noise via environment variable
        import os
        os.environ['LEARN_NOISE'] = '1'
        
        # Initialize noise parameter (will be set during first forward pass)
        noise_param = None
        optimizer = None
        
        best_loss = float('-inf')
        best_noise = None
        loss_history = []
        
        # First pass: generate video to get noise shape
        log.info("Initial pass to determine noise shape...")
        
        # Ensure LEARN_NOISE environment variable is set
        import os
        os.environ['LEARN_NOISE'] = '1'
        
        with torch.set_grad_enabled(True):
            # Clear previous noise anchor
            _rs.GLOBAL_Z0_ANCHOR = None
            
            # Generate video - this will create noise internally
            output_video, control_video_dict, fps, _ = (
                self.cosmos_inference.inference_pipeline._generate_img2world_impl(
                    video_path=path_to_str(sample_args.video_path),
                    prompt=sample_args.prompt,
                    negative_prompt=sample_args.negative_prompt,
                    image_context_path=path_to_str(sample_args.image_context_path) if hasattr(sample_args, 'image_context_path') else None,
                    guidance=sample_args.guidance,
                    seed=sample_args.seed,
                    resolution=sample_args.resolution,
                    control_weight=",".join([str(sample_args.control_weight_dict.get(k, 1.0)) for k in self.cosmos_inference.batch_hint_keys]),
                    sigma_max=sample_args.sigma_max,
                    hint_key=sample_args.hint_keys,
                    input_control_video_paths=sample_args.control_modalities,
                    show_control_condition=getattr(sample_args, 'show_control_condition', False),
                    seg_control_prompt=sample_args.seg_control_prompt,
                    show_input=getattr(sample_args, 'show_input', False),
                    keep_input_resolution=not getattr(sample_args, 'not_keep_input_resolution', True),
                    preset_blur_strength=getattr(sample_args, 'preset_blur_strength', 'medium'),
                    preset_edge_threshold=getattr(sample_args, 'preset_edge_threshold', 'medium'),
                    num_conditional_frames=sample_args.num_conditional_frames,
                    num_video_frames_per_chunk=sample_args.num_video_frames_per_chunk,
                    num_steps=sample_args.num_steps,
                )
            )
            
            # Check if noise was created, if not create it manually from the output video shape
            if hasattr(_rs, 'GLOBAL_Z0_ANCHOR') and _rs.GLOBAL_Z0_ANCHOR is not None:
                # Initialize learnable noise parameter from the generated noise
                noise_param = _rs.GLOBAL_Z0_ANCHOR.clone().detach().requires_grad_(True)
            else:
                # If GLOBAL_Z0_ANCHOR was not created, manually create it from output video shape
                # The noise shape should match the latent space shape used during generation
                # We need to infer the latent shape from the output video
                log.warning("GLOBAL_Z0_ANCHOR was not created during initial pass. Creating manually...")
                # The latent space is typically smaller than the output (due to VAE compression)
                # For Cosmos Transfer, we need to encode the video to get the latent shape
                # For now, let's try to get the shape from the model's state space
                # We'll create noise with the same shape as the output video for now
                # This might need adjustment based on the actual VAE latent space
                latent_shape = output_video.shape  # (B, C, T, H, W)
                # Create random noise with the same shape
                noise_param = torch.randn_like(output_video).requires_grad_(True)
                # Store it in GLOBAL_Z0_ANCHOR for future use
                _rs.GLOBAL_Z0_ANCHOR = noise_param
                log.info(f"Manually created noise parameter with shape: {noise_param.shape}")
            
            optimizer = torch.optim.Adam([noise_param], lr=lr)
            log.info(f"Initialized noise parameter with shape: {noise_param.shape}")
            
            # Compute initial loss
            output_video = output_video.to(self.device)
            initial_loss, initial_metrics = self.compute_action_loss(output_video)
            loss_history.append(initial_loss.item())
            log.info(f"Initial Action Loss: {initial_loss.item():.6f}")
        
        # Optimization loop
        for iteration in tqdm(range(num_iterations), desc="Optimizing noise"):
            # Enable gradients for this iteration
            with torch.set_grad_enabled(True):
                try:
                    # Set the learnable noise parameter
                    _rs.GLOBAL_Z0_ANCHOR = noise_param
                    
                    # Generate video with learnable noise
                    output_video, control_video_dict, fps, _ = (
                        self.cosmos_inference.inference_pipeline._generate_img2world_impl(
                            video_path=path_to_str(sample_args.video_path),
                            prompt=sample_args.prompt,
                            negative_prompt=sample_args.negative_prompt,
                            image_context_path=path_to_str(sample_args.image_context_path) if hasattr(sample_args, 'image_context_path') else None,
                            guidance=sample_args.guidance,
                            seed=sample_args.seed,  # Keep same seed for consistency
                            resolution=sample_args.resolution,
                            control_weight=",".join([str(sample_args.control_weight_dict.get(k, 1.0)) for k in self.cosmos_inference.batch_hint_keys]),
                            sigma_max=sample_args.sigma_max,
                            hint_key=sample_args.hint_keys,
                            input_control_video_paths=sample_args.control_modalities,
                            show_control_condition=getattr(sample_args, 'show_control_condition', False),
                            seg_control_prompt=sample_args.seg_control_prompt,
                            show_input=getattr(sample_args, 'show_input', False),
                            keep_input_resolution=not getattr(sample_args, 'not_keep_input_resolution', True),
                            preset_blur_strength=getattr(sample_args, 'preset_blur_strength', 'medium'),
                            preset_edge_threshold=getattr(sample_args, 'preset_edge_threshold', 'medium'),
                            num_conditional_frames=sample_args.num_conditional_frames,
                            num_video_frames_per_chunk=sample_args.num_video_frames_per_chunk,
                            num_steps=sample_args.num_steps,
                        )
                    )
                    
                    # Move video to device
                    output_video = output_video.to(self.device)
                    
                    # Compute action loss
                    action_loss, metrics = self.compute_action_loss(output_video)
                    
                    # We want to MAXIMIZE the loss (adversarial attack)
                    # So we minimize the negative loss
                    loss_to_minimize = -action_loss
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss_to_minimize.backward()
                    
                    # Clip gradients for stability
                    if noise_param.grad is not None:
                        torch.nn.utils.clip_grad_norm_([noise_param], max_norm=1.0)
                    
                    # Update noise
                    optimizer.step()
                    
                    # Track best loss (we want maximum loss)
                    current_loss = action_loss.item()
                    loss_history.append(current_loss)
                    
                    if current_loss > best_loss:
                        best_loss = current_loss
                        best_noise = noise_param.clone().detach()
                    
                    # Log progress
                    if iteration % 1 == 0:
                        log.info(f"Iteration {iteration+1}/{num_iterations}")
                        log.info(f"  Action Loss: {current_loss:.6f}")
                        log.info(f"  Action Diff: {metrics['action_diff']:.6f}")
                        if noise_param.grad is not None:
                            grad_norm = noise_param.grad.norm().item()
                            log.info(f"  Noise Grad Norm: {grad_norm:.6e}")
                            log.info(f"  Noise Param Norm: {noise_param.norm().item():.6f}")
                    
                    # Save intermediate results
                    if output_dir and (iteration % max(1, num_iterations // 5) == 0 or iteration == num_iterations - 1):
                        output_dir.mkdir(parents=True, exist_ok=True)
                        # Save video
                        video_path = output_dir / f"iter_{iteration:03d}_video.pt"
                        torch.save(output_video.cpu(), video_path)
                        # Save noise
                        noise_path = output_dir / f"iter_{iteration:03d}_noise.pt"
                        torch.save(noise_param.detach().cpu(), noise_path)
                        
                except Exception as e:
                    log.error(f"Error in iteration {iteration}: {e}")
                    import traceback
                    log.error(traceback.format_exc())
                    break
        
        # Generate final video with best noise
        log.info("=" * 60)
        log.info("Generating final video with optimized noise...")
        log.info("=" * 60)
        
        final_video = None
        initial_video = None
        final_video_path = None
        initial_video_path = None
        
        if best_noise is not None and output_dir:
            # Generate video with best noise
            with torch.set_grad_enabled(False):
                _rs.GLOBAL_Z0_ANCHOR = best_noise
                
                log.info("Generating video with best (maximized loss) noise...")
                final_video, control_video_dict, fps, _ = (
                    self.cosmos_inference.inference_pipeline._generate_img2world_impl(
                        video_path=path_to_str(sample_args.video_path),
                        prompt=sample_args.prompt,
                        negative_prompt=sample_args.negative_prompt,
                        image_context_path=path_to_str(sample_args.image_context_path) if hasattr(sample_args, 'image_context_path') else None,
                        guidance=sample_args.guidance,
                        seed=sample_args.seed,
                        resolution=sample_args.resolution,
                        control_weight=",".join([str(sample_args.control_weight_dict.get(k, 1.0)) for k in self.cosmos_inference.batch_hint_keys]),
                        sigma_max=sample_args.sigma_max,
                        hint_key=sample_args.hint_keys,
                        input_control_video_paths=sample_args.control_modalities,
                        show_control_condition=getattr(sample_args, 'show_control_condition', False),
                        seg_control_prompt=sample_args.seg_control_prompt,
                        show_input=getattr(sample_args, 'show_input', False),
                        keep_input_resolution=not getattr(sample_args, 'not_keep_input_resolution', True),
                        preset_blur_strength=getattr(sample_args, 'preset_blur_strength', 'medium'),
                        preset_edge_threshold=getattr(sample_args, 'preset_edge_threshold', 'medium'),
                        num_conditional_frames=sample_args.num_conditional_frames,
                        num_video_frames_per_chunk=sample_args.num_video_frames_per_chunk,
                        num_steps=sample_args.num_steps,
                    )
                )
                final_video = final_video.to(self.device)
                
                # Compute final action loss
                final_loss, final_metrics = self.compute_action_loss(final_video)
                log.info(f"Final video Action Loss: {final_loss.item():.6f}")
                
                # Save final video
                from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video
                
                # Convert from (B, C, T, H, W) to (C, T, H, W) and normalize to [0, 1]
                final_video_save = final_video[0]  # Remove batch dimension: (C, T, H, W)
                final_video_save = (final_video_save + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]
                final_video_save = torch.clamp(final_video_save, 0.0, 1.0)
                
                final_video_path = output_dir / "final_optimized_video"
                save_img_or_video(final_video_save.cpu(), str(final_video_path), fps=fps)
                log.info(f"✅ Final optimized video saved to: {final_video_path}.mp4")
                
                # Optionally generate and save initial video for comparison
                log.info("Generating video with initial (unoptimized) noise for comparison...")
                _rs.GLOBAL_Z0_ANCHOR = None  # Reset to None to generate with random noise
                
                initial_video, _, _, _ = (
                    self.cosmos_inference.inference_pipeline._generate_img2world_impl(
                        video_path=path_to_str(sample_args.video_path),
                        prompt=sample_args.prompt,
                        negative_prompt=sample_args.negative_prompt,
                        image_context_path=path_to_str(sample_args.image_context_path) if hasattr(sample_args, 'image_context_path') else None,
                        guidance=sample_args.guidance,
                        seed=sample_args.seed,  # Same seed for fair comparison
                        resolution=sample_args.resolution,
                        control_weight=",".join([str(sample_args.control_weight_dict.get(k, 1.0)) for k in self.cosmos_inference.batch_hint_keys]),
                        sigma_max=sample_args.sigma_max,
                        hint_key=sample_args.hint_keys,
                        input_control_video_paths=sample_args.control_modalities,
                        show_control_condition=getattr(sample_args, 'show_control_condition', False),
                        seg_control_prompt=sample_args.seg_control_prompt,
                        show_input=getattr(sample_args, 'show_input', False),
                        keep_input_resolution=not getattr(sample_args, 'not_keep_input_resolution', True),
                        preset_blur_strength=getattr(sample_args, 'preset_blur_strength', 'medium'),
                        preset_edge_threshold=getattr(sample_args, 'preset_edge_threshold', 'medium'),
                        num_conditional_frames=sample_args.num_conditional_frames,
                        num_video_frames_per_chunk=sample_args.num_video_frames_per_chunk,
                        num_steps=sample_args.num_steps,
                    )
                )
                initial_video = initial_video.to(self.device)
                
                # Compute initial action loss
                initial_loss, initial_metrics = self.compute_action_loss(initial_video)
                log.info(f"Initial video Action Loss: {initial_loss.item():.6f}")
                
                # Save initial video
                initial_video_save = initial_video[0]  # Remove batch dimension
                initial_video_save = (initial_video_save + 1.0) / 2.0
                initial_video_save = torch.clamp(initial_video_save, 0.0, 1.0)
                
                initial_video_path = output_dir / "initial_unoptimized_video"
                save_img_or_video(initial_video_save.cpu(), str(initial_video_path), fps=fps)
                log.info(f"✅ Initial (unoptimized) video saved to: {initial_video_path}.mp4")
                
                # Save comparison summary
                comparison_summary = {
                    'initial_loss': initial_loss.item(),
                    'final_loss': final_loss.item(),
                    'loss_increase': final_loss.item() - initial_loss.item(),
                    'loss_increase_percent': ((final_loss.item() - initial_loss.item()) / initial_loss.item() * 100) if initial_loss.item() > 0 else 0,
                }
                comparison_path = output_dir / "comparison_summary.txt"
                with open(comparison_path, 'w') as f:
                    f.write("=" * 60 + "\n")
                    f.write("Adversarial Attack Comparison Summary\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Initial (Unoptimized) Video:\n")
                    f.write(f"  Action Loss: {initial_loss.item():.6f}\n")
                    f.write(f"  Video Path: {initial_video_path}.mp4\n\n")
                    f.write(f"Final (Optimized) Video:\n")
                    f.write(f"  Action Loss: {final_loss.item():.6f}\n")
                    f.write(f"  Video Path: {final_video_path}.mp4\n\n")
                    f.write(f"Loss Increase: {comparison_summary['loss_increase']:.6f}\n")
                    f.write(f"Loss Increase (%): {comparison_summary['loss_increase_percent']:.2f}%\n")
                log.info(f"✅ Comparison summary saved to: {comparison_path}")
        
        results = {
            'best_loss': best_loss,
            'loss_history': loss_history,
            'best_noise': best_noise,
            'final_noise': noise_param.detach().clone() if noise_param is not None else None,
            'final_video_path': str(final_video_path) + ".mp4" if final_video_path is not None else None,
            'initial_video_path': str(initial_video_path) + ".mp4" if initial_video_path is not None else None,
        }
        
        log.info("=" * 60)
        log.info("Optimization Complete")
        log.info("=" * 60)
        log.info(f"Best Loss: {best_loss:.6f}")
        log.info(f"Initial Loss: {loss_history[0] if loss_history else 'N/A'}")
        log.info(f"Final Loss: {loss_history[-1] if loss_history else 'N/A'}")
        if final_video is not None and initial_video is not None:
            log.info(f"Final Video: {results['final_video_path']}")
            log.info(f"Initial Video: {results['initial_video_path']}")
        
        return results


def load_diffusion_policy(checkpoint_path: str, config_path: Optional[str] = None) -> DiffusionUnetImagePolicy:
    """Load Diffusion Policy model from checkpoint."""
    import dill
    import hydra
    from omegaconf import OmegaConf
    from diffusion_policy.workspace.base_workspace import BaseWorkspace
    
    log.info(f"Loading Diffusion Policy from {checkpoint_path}")
    
    # Load checkpoint (using dill for Diffusion Policy checkpoints)
    try:
        payload = torch.load(open(checkpoint_path, 'rb'), map_location='cpu', pickle_module=dill)
    except Exception as e:
        log.warning(f"Failed to load with dill, trying standard torch.load: {e}")
        payload = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if checkpoint has cfg (workspace format)
    if 'cfg' in payload:
        # Use workspace format - directly instantiate policy from cfg.policy
        cfg = payload['cfg']
        
        # Try to instantiate policy directly from cfg.policy to avoid workspace dependencies
        try:
            policy = hydra.utils.instantiate(cfg.policy)
            
            # Load state dict
            if 'state_dicts' in payload and 'model' in payload['state_dicts']:
                policy.load_state_dict(payload['state_dicts']['model'])
            elif 'state_dicts' in payload and 'ema_model' in payload['state_dicts']:
                # Use EMA model if available and use_ema is True
                if cfg.training.get('use_ema', False):
                    policy.load_state_dict(payload['state_dicts']['ema_model'])
                else:
                    policy.load_state_dict(payload['state_dicts']['model'])
            else:
                raise ValueError("Cannot find model state_dict in checkpoint")
            
            log.info("Successfully loaded Diffusion Policy directly from cfg.policy")
        except Exception as e:
            log.warning(f"Failed to load policy directly, trying workspace approach: {e}")
            # Fallback to workspace approach (requires robomimic)
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg, output_dir=None)
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)
            
            # Get policy from workspace
            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            
            log.info("Successfully loaded Diffusion Policy from workspace checkpoint")
    elif config_path:
        # Load from config file
        cfg = OmegaConf.load(config_path)
        
        # If config has policy section, instantiate it
        if 'policy' in cfg:
            policy = hydra.utils.instantiate(cfg.policy)
            if 'state_dict' in payload:
                policy.load_state_dict(payload['state_dict'])
            elif 'state_dicts' in payload and 'model' in payload['state_dicts']:
                policy.load_state_dict(payload['state_dicts']['model'])
            else:
                raise ValueError("Cannot find state_dict in checkpoint")
        else:
            raise ValueError(f"Config file {config_path} does not contain 'policy' section")
        
        log.info("Successfully loaded Diffusion Policy from config file")
    else:
        raise ValueError("Checkpoint does not contain 'cfg' and no config_path provided")
    
    return policy


def main():
    parser = argparse.ArgumentParser(description="Adversarial Attack: Optimize Cosmos Transfer noise to maximize Diffusion Policy loss")
    
    # Cosmos Transfer arguments
    parser.add_argument("--cosmos_checkpoint", type=str, required=True, help="Cosmos Transfer checkpoint path or variant")
    parser.add_argument("--input_video", type=str, required=True, help="Input video path")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--output_dir", type=str, default="./adversarial_output", help="Output directory")
    
    # Diffusion Policy arguments
    parser.add_argument("--diffusion_policy_checkpoint", type=str, required=True, help="Diffusion Policy checkpoint path")
    parser.add_argument("--diffusion_policy_config", type=str, default="", help="Diffusion Policy config YAML path (optional if checkpoint has cfg)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset for getting ground truth")
    
    # Optimization arguments
    parser.add_argument("--num_iterations", type=int, default=10, help="Number of optimization iterations")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for noise optimization")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Cosmos Transfer
    log.info("Initializing Cosmos Transfer2.5...")
    # Parse cosmos_checkpoint (could be "edge", "depth", etc. or a path)
    model_name = args.cosmos_checkpoint if args.cosmos_checkpoint in ["edge", "depth", "seg", "vis"] else "edge"
    
    setup_args = SetupArguments(
        output_dir=output_dir,
        model=model_name,
        context_parallel_size=1,
        keep_going=True,
    )
    cosmos_inference = Control2WorldInference(
        args=setup_args,
        batch_hint_keys=["edge"],  # Adjust based on your needs
    )
    
    # Load Diffusion Policy
    log.info("Loading Diffusion Policy...")
    config_path = args.diffusion_policy_config if args.diffusion_policy_config else None
    diffusion_policy = load_diffusion_policy(
        args.diffusion_policy_checkpoint,
        config_path
    )
    
    # Load dataset and get a ground truth batch
    log.info(f"Loading dataset from {args.dataset_path}...")
    dataset = PushTImageDataset(args.dataset_path, horizon=16)
    normalizer = dataset.get_normalizer()
    diffusion_policy.set_normalizer(normalizer)
    
    # Get a sample batch
    sample_batch = dataset[0]  # Get first sample
    # Add batch dimension
    ground_truth_batch = {
        'obs': {k: v.unsqueeze(0) for k, v in sample_batch['obs'].items()},
        'action': sample_batch['action'].unsqueeze(0),
    }
    
    # Create inference arguments
    sample_args = InferenceArguments(
        name="adversarial_sample",
        video_path=args.input_video,
        prompt=args.prompt,
        negative_prompt="",
        edge=EdgeConfig(control_weight=1.0),  # Provide edge config
        guidance=7,
        seed=42,
        resolution="720",  # Default resolution
        num_steps=35,  # Default number of steps
        # num_video_frames_per_chunk will use default value (93)
    )
    
    # Initialize adversarial attack
    attack = AdversarialAttack(
        cosmos_inference=cosmos_inference,
        diffusion_policy=diffusion_policy,
        ground_truth_batch=ground_truth_batch,
        device=args.device,
    )
    
    # Run optimization
    results = attack.optimize_noise(
        sample_args=sample_args,
        num_iterations=args.num_iterations,
        lr=args.lr,
        output_dir=output_dir / "iterations",
    )
    
    # Save results
    results_path = output_dir / "optimization_results.pt"
    torch.save(results, results_path)
    log.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()

