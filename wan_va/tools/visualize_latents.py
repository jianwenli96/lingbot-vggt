# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import sys
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

import re
import glob
import torch
import torch_npu
import numpy as np
from einops import rearrange
from torch_npu.contrib import transfer_to_npu

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model
from distributed.util import _configure_model
from modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from utils import (
    FlowMatchScheduler,
    init_logger,
    logger
)


class VA_Server:

    def __init__(self, job_config):
        self.job_config = job_config
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        self.enable_offload = getattr(job_config, 'enable_offload', True)  # offload vae & text_encoder to save vram

        self.vae = load_vae(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'vae'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )

        self.video_processor = VideoProcessor(vae_scale_factor=1)

    def decode_one_video(self, latents, output_type):
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video.detach(), output_type=output_type)
        return video

    # 定义一个提取数字的函数
    def extract_number(self, path):
        # 1. 获取最后的文件名部分 (e.g., 'latents_62.pt')
        filename = path.split('/')[-1]
        # 2. 用正则表达式找到文件名中所有的数字组合
        numbers = re.findall(r'\d+', filename)
        # 3. 取最后一个匹配的数字并转为整数，如果没有数字则默认返回 0
        return int(numbers[-1]) if numbers else 0

    def decode_single_pth(self, pth_path, output_path=None, fps=10):
        """Decode a single .pth file to video.

        Args:
            pth_path: Path to .pth file containing flattened latent tensor
            output_path: Output video path. If None, saves to same directory as .pth file
            fps: FPS for output video
        """
        # Load .pth file
        logger.info(f"Loading .pth file: {pth_path}")
        data = torch.load(pth_path, weights_only=False)

        # Extract latent and metadata
        latent_flat = data['latent']  # Shape: (B*F*H*W, C)
        latent_num_frames = data['latent_num_frames']
        latent_height = data['latent_height']
        latent_width = data['latent_width']

        logger.info(f"Latent shape: {latent_flat.shape}, frames: {latent_num_frames}, "
                   f"height: {latent_height}, width: {latent_width}")

        # Reshape flattened latent back to (B, C, F, H, W)
        # latent_flat shape: (B*F*H*W, C) -> (1, C, F, H, W)
        latent_channels = latent_flat.shape[1]  # Should be 48
        latents = rearrange(latent_flat, '(b f h w) c -> b c f h w',
                           b=1, f=latent_num_frames, h=latent_height, w=latent_width)

        # Move VAE to GPU for decoding
        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)

        # Decode video
        latents = latents.to(self.device)
        decoded_video = self.decode_one_video(latents, 'np')[0]

        # Determine output path

        if output_path is None:
            pth_dir = os.path.dirname(pth_path)
            pth_name = os.path.basename(pth_path).replace('.pth', '')
            output_path = os.path.join(pth_dir, f"{pth_name}_decoded.mp4")

        # Export video
        logger.info(f"Exporting video with {len(decoded_video)} frames to {output_path}")
        export_to_video(decoded_video, output_path, fps=fps)
        logger.info("Video export completed!")

        return output_path

    def decode_video_latent(self, latent_root, output_path=None):
        pt_pathes = [pt_path for pt_path in glob.glob(os.path.join(latent_root, "latents*.pt"))]
        pt_pathes = sorted(pt_pathes, key=self.extract_number)
        chunk_size = 5
        chunked_pathes = [pt_pathes[i: i + chunk_size] for i in range(0, len(pt_pathes), chunk_size)]

        # Move VAE to GPU for decoding
        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)

        # Collect all video frames from all chunks
        all_frames = []
        for index, chunk in enumerate(chunked_pathes):
            logger.info(f"Decoding chunk {index + 1}/{len(chunked_pathes)}")
            pred_latent_lst = []
            for pt_path in chunk:
                pred_latent = torch.load(pt_path, weights_only=False)
                pred_latent_lst.append(pred_latent.to(self.device))
            pred_latent_lst = torch.cat(pred_latent_lst, dim=2)
            decoded_video = self.decode_one_video(pred_latent_lst, 'np')[0]
            all_frames.append(decoded_video)

        # Concatenate all frames and export to a single video
        all_frames = np.concatenate(all_frames, axis=0)
        if output_path is None:
            output_path = os.path.join(latent_root, "merged_video.mp4")
        logger.info(f"Exporting video with {len(all_frames)} frames to {output_path}")
        export_to_video(all_frames, output_path, fps=10)
        logger.info("Video export completed!")


def run(args):
    config = VA_CONFIGS[args.config_name]
    if args.latent_root is not None:
        config.latent_root = args.latent_root
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    model = VA_Server(config)

    # Check if decoding single .pth file
    if args.pth_path:
        model.decode_single_pth(args.pth_path, args.output_path, args.fps)
    else:
        model.decode_video_latent(args.latent_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        required=False,
        default='robotwin_i2av',
        help="config name.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help='(start) port'
    )
    parser.add_argument(
        "--latent-root",
        type=str,
        default=None,
        help='save root'
    )
    parser.add_argument(
        "--pth-path",
        type=str,
        default=None,
        help='Path to a single .pth file to decode. If specified, --latent-root is ignored.'
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help='Output video path for single .pth decoding. If not specified, saves to same directory as .pth file.'
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help='FPS for output video when decoding single .pth file.'
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    # setup_debugger()
    init_logger()
    main()
