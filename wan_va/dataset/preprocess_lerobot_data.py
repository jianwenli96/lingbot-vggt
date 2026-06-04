#!/usr/bin/env python3
"""
Processing script using Xi0 VAE and text encoder initialization logic.

Supports:
- Step 1: Generate action_config in episodes.jsonl
- Step 2: Extract latents using Wan2.2 VAE
- Batch processing multiple datasets in a directory
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch
import torch_npu
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

# Add Xi0 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from Xi0 utils
from wan_va.modules.utils import load_vae, load_text_encoder, load_tokenizer


# ==================== Step 1: Generate action_config ====================

def generate_action_config(episode: Dict) -> List[Dict]:
    """Generate default action_config for an episode.

    Default: treat entire episode as single segment with task description.
    """
    episode_length = episode.get("length", 0)
    tasks = episode.get("tasks", [])
    task_text = tasks[0] if tasks else ""

    action_config = [
        {
            "start_frame": 0,
            "end_frame": episode_length,
            "action_text": task_text,
            "skill": ""  # Optional field
        }
    ]
    return action_config


def update_episodes_jsonl(dataset_path: Path) -> None:
    """Step 1: Add or update action_config in episodes.jsonl.

    Args:
        dataset_path: Path to LeRobot dataset root
    """
    episodes_path = dataset_path / "meta" / "episodes.jsonl"

    if not episodes_path.exists():
        raise FileNotFoundError(f"episodes.jsonl not found at {episodes_path}")

    # Load existing episodes
    episodes = []
    with open(episodes_path, 'r') as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))

    # Update action_config
    updated_count = 0
    for episode in episodes:
        episode['action_config'] = generate_action_config(episode)
        updated_count += 1

    # Save updated episodes
    with open(episodes_path, 'w') as f:
        for episode in episodes:
            f.write(json.dumps(episode) + '\n')

    print(f"  ✓ Updated {updated_count}/{len(episodes)} episodes in {episodes_path}")


# ==================== Step 2: Extract latents ====================

def normalize_latents(latents, latents_mean, latents_std):
    """Normalize latents using VAE config."""
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
    latents = ((latents.float() - latents_mean) * latents_std).to(latents)
    return latents


def extract_latents_from_video(
    vae,
    video_path: str,
    action_text: str,
    text_encoder,
    tokenizer,
    fps: int = 10,
    height: int = 256,
    width: int = 256,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    dtype=torch.bfloat16,
    device: str = "npu",
):
    """Extract latent features from video using Xi0 VAE (AutoencoderKLWan)."""
    import av  # Use PyAV for better codec support (including AV1)

    # Read video with PyAV
    container = av.open(video_path)
    stream = container.streams.video[0]

    ori_fps = float(stream.average_rate)
    total_frames = stream.frames
    video_height = stream.height
    video_width = stream.width

    if end_frame is None:
        end_frame = total_frames

    # Calculate frame indices to sample
    frame_indices = []
    frame_idx = start_frame
    frame_stride = max(1, int(ori_fps / fps))
    while frame_idx < end_frame:
        frame_indices.append(frame_idx)
        frame_idx += frame_stride

    if len(frame_indices) == 0:
        raise ValueError(f"No frames to extract from {video_path}")

    # Adjust frame count to satisfy constraint: (num_frames - 1) % 4 == 0
    # This is required by Wan VAE encoder
    num_frames = len(frame_indices)
    if (num_frames - 1) % 4 != 0:
        # Find the largest valid frame count <= current count
        # num_frames = 4k + 1, where k is integer
        # => k = (num_frames - 1) // 4
        # => valid_num_frames = 4k + 1
        valid_num_frames = ((num_frames - 1) // 4) * 4 + 1
        if valid_num_frames > 0:
            frame_indices = frame_indices[:valid_num_frames]
            print(f"Warning: Adjusted frame count from {num_frames} to {valid_num_frames} to satisfy VAE constraint")
        else:
            raise ValueError(f"Frame count {num_frames} cannot satisfy VAE constraint (need at least 1 frame)")

    # Read frames using PyAV (supports software AV1 decoding)
    frames = []
    frame_count = 0
    target_indices = set(frame_indices)

    container.seek(0)
    for frame in container.decode(video=0):
        if frame_count in target_indices:
            # Convert to numpy array (RGB format)
            img = frame.to_ndarray(format='rgb24')
            frames.append(img)
        frame_count += 1
        if frame_count >= end_frame:
            break
    container.close()

    if len(frames) == 0:
        raise ValueError(f"No frames read from {video_path}")

    # Convert to tensor
    frames = np.stack(frames)
    frames = torch.from_numpy(frames).float().permute(3, 0, 1, 2).unsqueeze(0)
    # Shape: (1, C, F, H, W) - batch, channels, frames, height, width

    # Resize spatial dimensions (height and width) for each frame
    frames = frames.squeeze(0).permute(1, 0, 2, 3)  # (F, C, H, W)
    frames = F.interpolate(frames, size=(height, width), mode='bilinear', align_corners=False)
    # Reshape back to (1, C, F, H, W)
    frames = frames.permute(1, 0, 2, 3).unsqueeze(0)

    # Normalize to [-1, 1]
    frames = frames / 255.0 * 2.0 - 1.0
    frames = frames.to(device).to(dtype)

    # Encode with VAE (AutoencoderKLWan)
    with torch.no_grad():
        # Use vae.encode() directly - it handles patchify and encoder internally
        posterior = vae.encode(frames).latent_dist
        # Use mean (mu) instead of sample for deterministic latent representation
        latents = posterior.mean

        # Normalize latents using VAE config
        latents_mean = torch.tensor(vae.config.latents_mean).to(latents.device)
        latents_std = torch.tensor(vae.config.latents_std).to(latents.device)
        latents = normalize_latents(latents, latents_mean, 1.0 / latents_std)

    # Get latent shape
    latent_num_frames = latents.shape[2]
    latent_height = latents.shape[3]
    latent_width = latents.shape[4]

    # Flatten latents: (B, C, F, H, W) -> (B*F*H*W, C)
    # Note: batch dimension is included in the flattening
    latents_flat = rearrange(latents, 'b c f h w -> (b f h w) c')

    # Encode text
    prompt = action_text.strip()
    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    with torch.no_grad():
        text_encoder_device = next(text_encoder.parameters()).device
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder_device),
            mask.to(text_encoder_device)
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        text_emb = prompt_embeds[0]

    # Prepare output
    output = {
        'latent': latents_flat.cpu().to(dtype),
        'latent_num_frames': latent_num_frames,
        'latent_height': latent_height,
        'latent_width': latent_width,
        'video_num_frames': len(frame_indices),
        'video_height': video_height,
        'video_width': video_width,
        'text_emb': text_emb.cpu().to(dtype),
        'text': action_text,
        'frame_ids': frame_indices,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'fps': fps,
        'ori_fps': int(ori_fps),
    }

    return output


def get_target_resolution(video_key: str, base_height: int, base_width: int, env_type: str = None) -> tuple:
    """Get target resolution for a video key based on environment type.

    For robotwin_tshape mode:
    - cam_high uses base resolution (base_height, base_width)
    - wrist cameras use half resolution (base_height // 2, base_width // 2)

    Args:
        video_key: Video key (e.g., 'observation.images.cam_high')
        base_height: Base height resolution
        base_width: Base width resolution
        env_type: Environment type (e.g., 'robotwin_tshape')

    Returns:
        tuple: (height, width) target resolution
    """
    if env_type == 'robotwin_tshape':
        # Check if this is a wrist camera
        is_wrist = any(wrist_key in video_key for wrist_key in ['left_wrist', 'right_wrist'])
        if is_wrist:
            # Use half resolution for wrist cameras
            return base_height // 2, base_width // 2

    # Default: use base resolution
    return base_height, base_width


def main():
    parser = argparse.ArgumentParser(
        description="Process dataset using Wan2.2 VAE and text encoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Process multiple datasets in a directory:
            python preprocess_lerobot_data.py --input-dir /path/to/datasets
        """
    )
    parser.add_argument("--input-dir", type=str, help="Path to directory containing multiple datasets")
    parser.add_argument("--pretrained-model-path", type=str,
                        default="/efs-gy1/ckpts/lingbot-va-base",
                        help="Path to pretrained model root directory (containing vae/, text_encoder/, tokenizer/)")
    parser.add_argument("--fps", type=float, default=12.5, help="Target FPS")
    parser.add_argument("--height", type=int, default=256, help="Target height")
    parser.add_argument("--width", type=int, default=320, help="Target width")
    parser.add_argument("--env-type", type=str, default=None,
                        help="Environment type (e.g., 'robotwin_tshape'). If not specified, auto-detect from dataset.")
    parser.add_argument("--video-keys", type=str, nargs='+', help="Video keys to process")
    parser.add_argument("--device", type=str, default="npu:0", help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument("--skip-step1", action="store_true", help="Skip Step 1 (action_config generation)")
    parser.add_argument("--skip-step2", action="store_true", help="Skip Step 2 (latent extraction)")

    args = parser.parse_args()

    # Validate input arguments
    if not args.input_dir:
        parser.error("Argument --input-dir must be specified")

    # Get list of datasets to process
    dataset_paths = []

    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Find all LeRobot datasets (directories with meta/info.json)
        for subdir in sorted(input_dir.iterdir()):
            if subdir.is_dir() and (subdir / "meta" / "info.json").exists():
                dataset_paths.append(subdir)

        print(f"Found {len(dataset_paths)} datasets in {input_dir}:")
        for p in dataset_paths:
            print(f"  - {p.name}")

    if not dataset_paths:
        raise ValueError("No datasets found to process")

    # Build model paths from pretrained-model-path
    vae_path = Path(args.pretrained_model_path) / "vae"
    text_encoder_path = Path(args.pretrained_model_path) / "text_encoder"
    tokenizer_path = Path(args.pretrained_model_path) / "tokenizer"

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # Check device availability
    if args.device.startswith('npu'):
        if not (hasattr(torch, 'npu') and torch.npu.is_available()):
            print("NPU not available, falling back to CPU")
            args.device = 'cpu'

    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"  Device: {args.device}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Target resolution: {args.height}x{args.width} @ {args.fps} fps")
    print(f"  Pretrained model: {args.pretrained_model_path}")
    print(f"  Steps: Step1={not args.skip_step1}, Step2={not args.skip_step2}")
    print(f"{'='*60}\n")

    # Load models (only if Step 2 is enabled)
    vae, text_encoder, tokenizer = None, None, None
    if not args.skip_step2:
        print("Loading models for Step 2...")
        print(f"  VAE: {vae_path}")
        print(f"  Text encoder: {text_encoder_path}")
        print(f"  Tokenizer: {tokenizer_path}")

        vae = load_vae(str(vae_path), dtype, args.device)
        text_encoder = load_text_encoder(str(text_encoder_path), dtype, args.device)
        tokenizer = load_tokenizer(str(tokenizer_path))
        print("  ✓ Models loaded\n")

    # Process each dataset
    for dataset_idx, dataset_path in enumerate(dataset_paths, 1):
        print(f"\n{'='*60}")
        print(f"[{dataset_idx}/{len(dataset_paths)}] Processing: {dataset_path.name}")
        print(f"{'='*60}\n")

        # Step 1: Generate action_config
        if not args.skip_step1:
            print("Step 1: Generating action_config...")
            update_episodes_jsonl(dataset_path)

        # Step 2: Extract latents
        if not args.skip_step2:
            print("\nStep 2: Extracting latents...")

            # Auto-detect video keys
            video_keys = args.video_keys
            if video_keys is None:
                videos_dir = dataset_path / "videos"
                if videos_dir.exists():
                    chunk_dirs = list(videos_dir.glob("chunk-*"))
                    if chunk_dirs:
                        video_keys = [d.name for d in chunk_dirs[0].iterdir() if d.is_dir()]
                        print(f"  Auto-detected video keys: {video_keys}")

            if not video_keys:
                print("  Warning: No video keys found, skipping Step 2")
                continue

            # Detect or use specified env_type
            env_type = args.env_type
            if env_type is None:
                # Try to auto-detect from dataset structure
                # If video keys contain 'left_wrist' and 'right_wrist', it's likely robotwin_tshape
                if any('left_wrist' in vk for vk in video_keys) and any('right_wrist' in vk for vk in video_keys):
                    env_type = 'robotwin_tshape'
                    print(f"  Auto-detected env_type: {env_type}")
                else:
                    env_type = None
                    print(f"  No env_type detected, using base resolution for all cameras")
            else:
                print(f"  Using env_type: {env_type}")

            # Read episodes
            episodes_path = dataset_path / "meta" / "episodes.jsonl"
            episodes = []
            with open(episodes_path, 'r') as f:
                for line in f:
                    if line.strip():
                        episodes.append(json.loads(line))

            # Create latents directory
            latents_dir = dataset_path / "latents"
            latents_dir.mkdir(exist_ok=True)

            # Process episodes
            for episode in tqdm(episodes, desc=f"  Processing episodes"):
                episode_index = episode['episode_index']
                episode_chunk = episode.get('episode_chunk', 0)
                length = episode['length']
                action_configs = episode['action_config']

                for video_key in video_keys:
                    video_file = (
                        dataset_path / "videos" / f"chunk-{episode_chunk:03d}" /
                        video_key / f"episode_{episode_index:06d}.mp4"
                    )

                    if not video_file.exists():
                        continue

                    latent_key_dir = latents_dir / f"chunk-{episode_chunk:03d}" / video_key
                    latent_key_dir.mkdir(parents=True, exist_ok=True)

                    # Get target resolution for this video key
                    target_height, target_width = get_target_resolution(
                        video_key, args.height, args.width, env_type
                    )

                    # Print resolution info for first episode
                    if episode_index == 0 and action_configs and action_configs[0]['start_frame'] == 0:
                        print(f"    {video_key}: target resolution {target_height}×{target_width}")

                    for acfg in action_configs:
                        start_frame = acfg['start_frame']
                        end_frame = acfg['end_frame']
                        action_text = acfg['action_text']

                        latent_file = latent_key_dir / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"

                        try:
                            latent_data = extract_latents_from_video(
                                vae, str(video_file), action_text, text_encoder, tokenizer,
                                args.fps, target_height, target_width,  # Use computed resolution
                                start_frame, end_frame, dtype, args.device
                            )
                            torch.save(latent_data, latent_file)
                        except Exception as e:
                            print(f"\n  Error processing {video_file}: {e}")
                            import traceback
                            traceback.print_exc()

            print(f"  ✓ Latents saved to {latents_dir}")

    print(f"\n{'='*60}")
    print(f"✓ All datasets processed successfully!")
    print(f"{'='*60}\n")

    # Generate empty_emb.pt in input_dir
    if args.input_dir:
        input_dir = Path(args.input_dir)
        empty_emb_path = input_dir / "empty_emb.pt"

        # Create a zero tensor with shape (512, 4096) and dtype bfloat16
        # This matches the content in /efs-gy1/dgh/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_clean_50/empty_emb.pt
        empty_emb = torch.zeros(512, 4096, dtype=torch.bfloat16)
        torch.save(empty_emb, empty_emb_path)

        print(f"✓ Generated empty_emb.pt at {empty_emb_path}")


if __name__ == "__main__":
    main()
