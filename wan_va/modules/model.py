# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import math
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange
from typing import Callable, ClassVar
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
    flex_attention,
    and_masks,
    or_masks
)
from functools import partial

__all__ = [
    'WanTransformer3DModel',
    'build_training_token_metadata',
    'pack_visual_tokens',
    'unpack_visual_tokens',
]


@dataclass(frozen=True)
class VisualTokenLayout:
    num_frames: int
    video_tokens_per_frame: int
    vggt_tokens_per_frame: int

    @property
    def tokens_per_frame(self):
        return self.video_tokens_per_frame + self.vggt_tokens_per_frame


def pack_visual_tokens(video_tokens, vggt_tokens):
    """Concatenate video and VGGT tokens within each frame."""
    if video_tokens.ndim != 4 or vggt_tokens.ndim != 4:
        raise ValueError(
            "visual tokens must have shape [B, F, N, D], got "
            f"{tuple(video_tokens.shape)} and {tuple(vggt_tokens.shape)}"
        )
    if video_tokens.shape[:2] != vggt_tokens.shape[:2]:
        raise ValueError(
            "video and VGGT batch and frame dimensions must match, got "
            f"{tuple(video_tokens.shape[:2])} and {tuple(vggt_tokens.shape[:2])}"
        )
    if video_tokens.shape[-1] != vggt_tokens.shape[-1]:
        raise ValueError(
            "video and VGGT hidden dimensions must match, got "
            f"{video_tokens.shape[-1]} and {vggt_tokens.shape[-1]}"
        )
    return torch.cat([video_tokens, vggt_tokens], dim=2)


def unpack_visual_tokens(
    joint_tokens,
    video_tokens_per_frame,
    vggt_tokens_per_frame,
):
    """Split a per-frame joint visual sequence back into its modalities."""
    if joint_tokens.ndim != 4:
        raise ValueError(
            "joint visual tokens must have shape [B, F, N, D], got "
            f"{tuple(joint_tokens.shape)}"
        )
    expected_tokens = video_tokens_per_frame + vggt_tokens_per_frame
    if joint_tokens.shape[2] != expected_tokens:
        raise ValueError(
            f"expected {expected_tokens} tokens per frame, got "
            f"{joint_tokens.shape[2]}"
        )
    return torch.split(
        joint_tokens,
        [video_tokens_per_frame, vggt_tokens_per_frame],
        dim=2,
    )


def build_training_token_metadata(
    batch_size,
    num_visual_frames,
    visual_tokens_per_frame,
    action_shape,
    chunk_size,
):
    """Build sequence, frame, and noise IDs for joint visual/action training."""
    action_batch, action_frames, action_height, action_width = action_shape
    if action_batch != batch_size:
        raise ValueError(
            f"visual and action batch sizes must match, got {batch_size} and {action_batch}"
        )
    visual_seq_ids = torch.arange(batch_size)[:, None, None].expand(
        -1, num_visual_frames, visual_tokens_per_frame
    ).flatten()
    action_seq_ids = torch.arange(batch_size)[:, None, None, None].expand(
        -1, action_frames, action_height, action_width
    ).flatten()
    seq_ids = torch.cat(
        [visual_seq_ids, visual_seq_ids, action_seq_ids, action_seq_ids]
    )

    visual_frame_ids = torch.arange(num_visual_frames)[None, :, None].expand(
        batch_size, -1, visual_tokens_per_frame
    ).flatten()
    action_frame_ids = torch.arange(action_frames)[None, :, None, None].expand(
        batch_size, -1, action_height, action_width
    ).flatten()
    frame_ids = torch.cat(
        [
            visual_frame_ids // chunk_size * 2,
            visual_frame_ids // chunk_size * 2,
            action_frame_ids // chunk_size * 2 + 1,
            action_frame_ids // chunk_size * 2 + 1,
        ]
    )
    noise_ids = torch.cat(
        [
            torch.zeros_like(visual_frame_ids),
            torch.ones_like(visual_frame_ids),
            torch.zeros_like(action_frame_ids),
            torch.ones_like(action_frame_ids),
        ]
    )
    return seq_ids, frame_ids, noise_ids


def custom_sdpa(q, k, v, **kwargs):
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2))
    return out.transpose(1, 2)


class SDPAAttnFunc(nn.Module):
    """
    使用PyTorch原生SDPA实现FlexAttnFunc的attention mask功能
    将FlexAttention的BlockMask转换为标准的attention mask张量
    """
    # attention_mask: ClassVar[torch.Tensor] = None
    # cross_attention_mask: ClassVar[torch.Tensor] = None
    seq_len: ClassVar[int] = None
    text_seq_len: ClassVar[int] = None

    def __init__(
        self,
        is_cross=False,
    ) -> None:
        super().__init__()
        self.is_cross = is_cross

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        """
        前向传播
        query, key, value: [B, S, N, D] 格式
        """
        q_varlen = rearrange(query[0], "s n d -> 1 n s d")
        k_varlen = rearrange(key[0], "s n d -> 1 n s d")
        v_varlen = rearrange(value[0], "s n d -> 1 n s d")

        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes
        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        q_varlen = half(q_varlen)
        k_varlen = half(k_varlen)
        v_varlen = half(v_varlen)
        q_varlen = q_varlen.to(v_varlen.dtype)
        k_varlen = k_varlen.to(v_varlen.dtype)

        # 选择对应的attention mask
        # attn_mask = SDPAAttnFunc.cross_attention_mask if self.is_cross else SDPAAttnFunc.attention_mask
        attn_mask = cross_attention_mask if self.is_cross else attention_mask
        attn_mask = attn_mask.to(query.device).bool() # 放到设备上 + 设置dtype
        # 使用SDPA计算注意力
        x_out = F.scaled_dot_product_attention(
            q_varlen, k_varlen, v_varlen, attn_mask=attn_mask)

        x_out = rearrange(x_out, "b n s d -> b s n d")
        return x_out

    @staticmethod
    @torch.no_grad()
    def init_mask(
        batch_size,
        num_visual_frames,
        visual_tokens_per_frame,
        action_shape,
        padded_length,
        chunk_size,
        window_size,
        device,
        dtype,
    ):
        """
        初始化attention mask
        将FlexAttention的掩码逻辑转换为标准的布尔或浮点掩码张量
        """
        _, _, action_frames, action_height, action_width = action_shape
        seq_ids, frame_ids, noise_ids = build_training_token_metadata(
            batch_size=batch_size,
            num_visual_frames=num_visual_frames,
            visual_tokens_per_frame=visual_tokens_per_frame,
            action_shape=(batch_size, action_frames, action_height, action_width),
            chunk_size=chunk_size,
        )

        # Padding
        seq_ids = F.pad(seq_ids, (0, padded_length), value=-1)
        frame_ids = F.pad(frame_ids, (0, padded_length), value=-1)
        noise_ids = F.pad(noise_ids, (0, padded_length), value=-1)

        seq_ids = seq_ids.long().to(device)
        frame_ids = frame_ids.long().to(device)
        noise_ids = noise_ids.long().to(device)

        # 生成自注意力掩码
        attention_mask = SDPAAttnFunc._create_attention_mask(
            seq_ids, frame_ids, noise_ids, window_size, device
        )
        # visualize_npu_mask(attention_mask, tokens=None, title="Causal Attention", save_path="Causal_attention_mask.png")
        # SDPAAttnFunc.attention_mask = attention_mask.to(dtype)
        SDPAAttnFunc.seq_len = len(seq_ids)

        # 生成交叉注意力掩码
        text_seq_ids = torch.arange(batch_size)[:, None].expand(-1, 512).flatten()
        text_seq_ids = text_seq_ids.long().to(device)
        cross_attention_mask = SDPAAttnFunc._create_cross_attention_mask(
            seq_ids, text_seq_ids, device
        )
        # visualize_npu_mask(cross_attention_mask, tokens=None, title="Cross Attention", save_path="cross_attention_mask.png")
        # SDPAAttnFunc.cross_attention_mask = cross_attention_mask.to(dtype)
        SDPAAttnFunc.text_seq_len = len(text_seq_ids)
        return attention_mask, cross_attention_mask

    @staticmethod
    @torch.no_grad()
    def _create_cross_attention_mask(seq_ids, text_seq_ids, device):
        """
        创建交叉注意力掩码
        返回: [S_q, S_kv] 的布尔掩码张量
        """
        seq_len_q = len(seq_ids)
        seq_len_kv = len(text_seq_ids)

        # 创建索引网格
        q_idx = torch.arange(seq_len_q, device=device).unsqueeze(1)
        kv_idx = torch.arange(seq_len_kv, device=device).unsqueeze(0)

        # 掩码逻辑: seq_ids[q_idx] == text_seq_ids[kv_idx] 且都 >= 0
        mask = (seq_ids[q_idx] == text_seq_ids[kv_idx]) & \
               (seq_ids[q_idx] >= 0) & \
               (text_seq_ids[kv_idx] >= 0)

        # 扩展到 [1, 1, S_q, S_kv] 以匹配 [B, N, S_q, D] @ [B, N, S_kv, D]
        # SDPA期望的掩码格式是 [S_q, S_kv] 或 [B, N, S_q, S_kv]
        mask = mask.unsqueeze(0).unsqueeze(0).bool()
        # 将True转换为1.0, False转换为-inf (或使用additive mask)
        mask = mask.masked_fill(mask == 0, False) # float('-inf')
        mask = mask.masked_fill(mask == 1, True) # 0.0

        return mask

    @staticmethod
    @torch.no_grad()
    def _create_attention_mask(seq_ids, frame_ids, noise_ids, window_size, device):
        """
        创建自注意力掩码
        返回: [S_q, S_kv] 的掩码张量
        """
        seq_len = len(seq_ids)

        # 创建索引网格
        q_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        kv_idx = torch.arange(seq_len, device=device).unsqueeze(0)

        # 1. 序列掩码: 相同序列ID且都 >= 0
        seq_mask = (seq_ids[q_idx] == seq_ids[kv_idx]) & \
                   (seq_ids[q_idx] >= 0) & \
                   (seq_ids[kv_idx] >= 0)

        # 2. 帧级因果掩码: kv帧 <= q帧
        block_causal_mask = frame_ids[kv_idx] <= frame_ids[q_idx]

        # 3. 排除自身的因果掩码: kv帧 < q帧
        block_causal_mask_exclude_self = frame_ids[kv_idx] < frame_ids[q_idx]

        # 4. 自身帧掩码: kv帧 == q帧
        block_self_mask = frame_ids[kv_idx] == frame_ids[q_idx]

        # 5. clean到clean掩码
        clean2clean_mask = (noise_ids[q_idx] == 1) & (noise_ids[kv_idx] == 1)

        # 6. noise到clean掩码
        noise2clean_mask = (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 1)

        # 7. noise到noise掩码
        noise2noise_mask = (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 0)

        # 8. 窗口掩码: 帧ID差异 <= window_size
        block_window_mask = (frame_ids[q_idx] - frame_ids[kv_idx]).abs() <= window_size

        # 组合掩码逻辑
        # mask_list = [
        #     clean2clean AND block_causal,
        #     noise2clean AND block_causal_exclude_self,
        #     noise2noise AND block_self
        # ]
        mask1 = clean2clean_mask & block_causal_mask
        mask2 = noise2clean_mask & block_causal_mask_exclude_self
        mask3 = noise2noise_mask & block_self_mask

        # OR组合三个条件
        combined_mask = mask1 | mask2 | mask3

        # AND序列掩码和窗口掩码
        final_mask = combined_mask & seq_mask & block_window_mask

        # 转换为SDPA期望的格式: [1, 1, S, S]
        # 布尔掩码: True表示参与注意力, False表示屏蔽
        # SDPA使用additive mask: 0表示参与, -inf表示屏蔽
        mask = final_mask.unsqueeze(0).unsqueeze(0).bool()
        mask = mask.masked_fill(mask == 0, False) # float('-inf')
        mask = mask.masked_fill(mask == 1, True) # 0.0
        return mask


class FlexAttnFunc(nn.Module):
    flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention, dynamic=True, 
    )
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(create_block_mask)
    attention_mask: ClassVar[BlockMask] = None
    cross_attention_mask: ClassVar[BlockMask] = None

    def __init__(
        self, 
        is_cross=False,
    ) -> None:
        super().__init__()
        self.is_cross = is_cross
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        q_varlen = rearrange(query[0], "s n d -> 1 n s d")
        k_varlen = rearrange(key[0], "s n d -> 1 n s d")
        v_varlen = rearrange(value[0], "s n d -> 1 n s d")

        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes
        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)
        
        q_varlen = half(q_varlen)
        k_varlen = half(k_varlen)
        v_varlen = half(v_varlen)
        q_varlen = q_varlen.to(v_varlen.dtype)
        k_varlen = k_varlen.to(v_varlen.dtype)

        block_mask = FlexAttnFunc.cross_attention_mask if self.is_cross else FlexAttnFunc.attention_mask

        x_out = FlexAttnFunc.flex_attn(q_varlen, k_varlen, v_varlen, block_mask=block_mask, kernel_options = {
                                                    "BLOCK_M": 64,
                                                    "BLOCK_N": 64,
                                                    "BLOCK_M1": 32,
                                                    "BLOCK_N1": 64,
                                                    "BLOCK_M2": 64,
                                                    "BLOCK_N2": 32,
                                                })

        x_out = rearrange(x_out, "b n s d -> b s n d")
        return x_out

    @staticmethod
    @torch.no_grad()
    def init_mask(
        batch_size,
        num_visual_frames,
        visual_tokens_per_frame,
        action_shape, 
        padded_length, 
        chunk_size,
        window_size,
        device,
    ):
        torch._inductor.config.realize_opcount_threshold = 100
        _, _, action_frames, action_height, action_width = action_shape
        seq_ids, frame_ids, noise_ids = build_training_token_metadata(
            batch_size=batch_size,
            num_visual_frames=num_visual_frames,
            visual_tokens_per_frame=visual_tokens_per_frame,
            action_shape=(batch_size, action_frames, action_height, action_width),
            chunk_size=chunk_size,
        )

        seq_ids = F.pad(seq_ids, (0, padded_length), value=-1)
        frame_ids = F.pad(frame_ids, (0, padded_length), value=-1)
        noise_ids = F.pad(noise_ids, (0, padded_length), value=-1)

        mask_mod = FlexAttnFunc._get_mask_mod(seq_ids.long().to(device), frame_ids.long().to(device), noise_ids.long().to(device), window_size)
        block_mask = FlexAttnFunc.compiled_create_block_mask(
                mask_mod, 1, 1, len(seq_ids), len(seq_ids), device=device, _compile=True
            )
        FlexAttnFunc.attention_mask = block_mask

        text_seq_ids = torch.arange(batch_size)[:, None].expand(-1, 512).flatten()
        mask_mod_cross = FlexAttnFunc._get_cross_mask_mod(seq_ids.long().to(device), text_seq_ids.long().to(device))
        block_mask_cross = FlexAttnFunc.compiled_create_block_mask(
                mask_mod_cross, 1, 1, len(seq_ids), len(text_seq_ids), device=device, _compile=True
            )
        FlexAttnFunc.cross_attention_mask = block_mask_cross
    
    @staticmethod
    @torch.no_grad()
    def _get_cross_mask_mod(seq_ids, text_seq_ids):
        def seq_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (seq_ids[q_idx] == text_seq_ids[kv_idx]) & (seq_ids[q_idx] >=0 ) & (text_seq_ids[kv_idx] >= 0)
        return seq_mask
    
    @staticmethod
    @torch.no_grad()
    def _get_mask_mod(seq_ids, frame_ids, noise_ids, window_size):
        def seq_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (seq_ids[q_idx] == seq_ids[kv_idx]) & (seq_ids[q_idx] >=0 ) & (seq_ids[kv_idx] >= 0)
        
        def block_causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (frame_ids[kv_idx] <= frame_ids[q_idx])
        
        def block_causal_mask_exclude_self(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (frame_ids[kv_idx] < frame_ids[q_idx])
        
        def block_self_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (frame_ids[kv_idx] == frame_ids[q_idx])
        
        def clean2clean_mask(
                b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 1) & (noise_ids[kv_idx] == 1)
        
        def noise2clean_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 1)
        def noise2noise_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 0)
        
        def block_window_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor, window_size: int
        ):
            return ((frame_ids[q_idx] - frame_ids[kv_idx]).abs() <= window_size)

        mask_list = []
        mask_list.append(and_masks(clean2clean_mask, block_causal_mask))
        mask_list.append(and_masks(noise2clean_mask, block_causal_mask_exclude_self))
        mask_list.append(and_masks(noise2noise_mask, block_self_mask))
        mask = or_masks(*mask_list)
        mask = and_masks(mask, seq_mask)
        mask = and_masks(mask, partial(block_window_mask, window_size=window_size))
        return mask
       
class WanTimeTextImageEmbedding(nn.Module):

    def __init__(
        self,
        dim,
        time_freq_dim,
        time_proj_dim,
        text_embed_dim,
        pos_embed_seq_len,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim,
                                        flip_sin_to_cos=True,
                                        downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim,
                                               time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim,
                                                       dim,
                                                       act_fn="gelu_tanh")

    def forward(
        self,
        timestep: torch.Tensor,
        dtype=None,
    ):
        B, L = timestep.shape
        timestep = timestep.reshape(-1)
        timestep = self.timesteps_proj(timestep)
        # time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        time_embedder_dtype = self.time_embedder.linear_1.weight.dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).to(dtype=dtype)
        timestep_proj = self.time_proj(self.act_fn(temb))
        return temb.reshape(B, L, -1), timestep_proj.reshape(B, L, -1)


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.f_dim = self.attention_head_dim - 2 * (self.attention_head_dim // 3)
        self.h_dim = self.attention_head_dim // 3
        self.w_dim = self.attention_head_dim // 3

        # Precompute and register buffers
        f_freqs_base, h_freqs_base, w_freqs_base = self._precompute_freqs_base()
        self.f_freqs_base = f_freqs_base
        self.h_freqs_base = h_freqs_base
        self.w_freqs_base = w_freqs_base

    def _precompute_freqs_base(self):
        # freqs_base = 1.0 / (theta ** (2k / dim))
        f_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.f_dim, 2)[:(self.f_dim // 2)].double() / self.f_dim))
        h_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.h_dim, 2)[:(self.h_dim // 2)].double() / self.h_dim))
        w_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.w_dim, 2)[:(self.w_dim // 2)].double() / self.w_dim))
        return f_freqs_base, h_freqs_base, w_freqs_base

    def forward(self, grid_ids):
        with torch.no_grad():
            f_freqs = grid_ids[:, 0, :].unsqueeze(-1) * self.f_freqs_base.to(grid_ids.device)
            h_freqs = grid_ids[:, 1, :].unsqueeze(-1) * self.h_freqs_base.to(grid_ids.device)
            w_freqs = grid_ids[:, 2, :].unsqueeze(-1) * self.w_freqs_base.to(grid_ids.device)
            freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        return freqs_cis


class WanAttention(torch.nn.Module):

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        eps=1e-5,
        dropout=0.0,
        cross_attention_dim_head=None,
        attn_mode='torch',
    ):
        super().__init__()
        if attn_mode == 'torch':
            self.attn_op = custom_sdpa
        elif attn_mode == 'flashattn':
            try:
                from flash_attn_interface import flash_attn_func
            except:
                from flash_attn import flash_attn_func
            self.attn_op = flash_attn_func
        elif attn_mode == 'torch_flex':
            self.attn_op = SDPAAttnFunc(cross_attention_dim_head is not None)
        elif attn_mode == 'flex':
            self.attn_op = FlexAttnFunc(cross_attention_dim_head is not None)
        else:
            raise ValueError(
                f"Unsupported attention mode: {attn_mode}, only support torch and flashattn"
            )

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList([
            torch.nn.Linear(self.inner_dim, dim, bias=True),
            torch.nn.Dropout(dropout),
        ])
        self.norm_q = torch.nn.RMSNorm(dim_head * heads,
                                       eps=eps,
                                       elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads,
                                       eps=eps,
                                       elementwise_affine=True)
        self.attn_caches = {} if cross_attention_dim_head is None else None

    def clear_pred_cache(self, cache_name):
        if self.attn_caches is None:
            return
        cache = self.attn_caches[cache_name]
        is_pred = cache['is_pred']
        cache['mask'][is_pred] = False

    def clear_cache(self, cache_name):
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = None

    def init_kv_cache(self, cache_name, total_tolen, num_head, head_dim,
                      device, dtype, batch_size):
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = {
            'k':
            torch.empty([batch_size, total_tolen, num_head, head_dim],
                        device=device,
                        dtype=dtype),
            'v':
            torch.empty([batch_size, total_tolen, num_head, head_dim],
                        device=device,
                        dtype=dtype),
            'id':
            torch.full((total_tolen, ), -1, device=device),
            "mask":
            torch.zeros((total_tolen, ), dtype=torch.bool, device=device),
            "is_pred":
            torch.zeros((total_tolen, ), dtype=torch.bool, device=device),
        }

    def allocate_slots(self, cache_name, key_size):
        cache = self.attn_caches[cache_name]
        mask = cache["mask"]
        ids = cache["id"]
        free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        if free.numel() < key_size:
            used = mask.nonzero(as_tuple=False).squeeze(-1)

            used_ids = ids[used]
            order = torch.argsort(used_ids)
            need = key_size - free.numel()
            to_free = used[order[:need]]

            mask[to_free] = False
            ids[to_free] = -1
            free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        assert free.numel() >= key_size
        return free[:key_size]

    def _next_cache_id(self, cache_name):
        ids = self.attn_caches[cache_name]['id']
        mask = self.attn_caches[cache_name]['mask']

        if mask.any():
            return ids[mask].max() + 1
        else:
            return torch.tensor(0, device=ids.device, dtype=ids.dtype)

    def update_cache(self, cache_name, key, value, is_pred):
        cache = self.attn_caches[cache_name]

        key_size = key.shape[1]
        slots = self.allocate_slots(cache_name, key_size)

        new_id = self._next_cache_id(cache_name)

        cache['k'][:, slots] = key
        cache['v'][:, slots] = value
        cache['mask'][slots] = True
        cache['id'][slots] = new_id
        cache['is_pred'][slots] = is_pred
        return slots

    def restore_cache(self, cache_name, slots):
        self.attn_caches[cache_name]['mask'][slots] = False

    def forward(
        self,
        q,
        k,
        v,
        rotary_emb,
        attention_mask,
        cross_attention_mask,
        update_cache=0,
        cache_name='pos',
    ):
        kv_cache = self.attn_caches[
            cache_name] if (self.attn_caches is not None) and (cache_name in self.attn_caches) else None

        query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
        query = self.norm_q(query)
        query = query.unflatten(2, (self.heads, -1))
        key = self.norm_k(key)
        key = key.unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))
        if rotary_emb is not None:

            def apply_rotary_emb(x, freqs):
                x_out = torch.view_as_complex(
                    x.to(torch.float64).reshape(x.shape[0], x.shape[1],
                                                x.shape[2], -1, 2))
                x_out = torch.view_as_real(x_out * freqs).flatten(3)
                return x_out.to(x.dtype)
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        slots = None
        if kv_cache is not None and kv_cache['k'] is not None:
            slots = self.update_cache(cache_name,
                                      key,
                                      value,
                                      is_pred=(update_cache == 1))
            key_pool = self.attn_caches[cache_name]['k']
            value_pool = self.attn_caches[cache_name]['v']
            mask = self.attn_caches[cache_name]['mask']
            valid = mask.nonzero(as_tuple=False).squeeze(-1)
            key = key_pool[:, valid]
            value = value_pool[:, valid]

        hidden_states = self.attn_op(query, key, value, attention_mask=attention_mask, cross_attention_mask=cross_attention_mask)

        if update_cache == 0:
            if kv_cache is not None and kv_cache['k'] is not None:
                self.restore_cache(cache_name, slots)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class WanTransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        cross_attn_norm=False,
        eps=1e-6,
        attn_mode: str = "flashattn",
    ):
        super().__init__()
        self.attn_mode = attn_mode

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            attn_mode=attn_mode,
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            attn_mode=attn_mode,
        )
        self.norm2 = FP32LayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim,
                               inner_dim=ffn_dim,
                               activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 6, dim) / dim**0.5)

    def compute_adaln_modulation(
        self,
        temb,
    ):
        temb_scale_shift_table = self.scale_shift_table[None] + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(6, dim=1)
        )
        shift_msa = shift_msa.squeeze(1)
        scale_msa = scale_msa.squeeze(1)
        gate_msa = gate_msa.squeeze(1)
        c_shift_msa = c_shift_msa.squeeze(1)
        c_scale_msa = c_scale_msa.squeeze(1)
        c_gate_msa = c_gate_msa.squeeze(1)

        return shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa

    def compute_cross_attn_and_ffn(
        self,
        hidden_states,
        encoder_hidden_states,
        c_shift_msa,
        c_scale_msa,
        c_gate_msa,
        attention_mask=None,
        cross_attention_mask=None,
        update_cache=0,
        cache_name="pos",
    ):
        # Cross-attention
        norm_hidden_states = self.norm2(
            hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states,
                                 encoder_hidden_states,
                                 encoder_hidden_states,
                                 None,
                                 attention_mask,
                                 cross_attention_mask,
                                 update_cache=0,
                                 cache_name=cache_name)
        hidden_states = hidden_states + attn_output

        # Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) *
                              (1. + c_scale_msa) +
                              c_shift_msa).type_as(hidden_states)

        ff_output = self.ffn(norm_hidden_states)

        hidden_states = (hidden_states.float() +
                         ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb,
        attention_mask=None,
        cross_attention_mask=None,
        update_cache=0,
        cache_name='pos',
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.compute_adaln_modulation(temb)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) *
                              (1. + scale_msa) +
                              shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states,
                                 norm_hidden_states,
                                 norm_hidden_states,
                                 rotary_emb,
                                 attention_mask,
                                 cross_attention_mask,
                                 update_cache=update_cache,
                                 cache_name=cache_name)
        hidden_states = (hidden_states.float() +
                         attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention and FFN
        hidden_states = self.compute_cross_attn_and_ffn(hidden_states,
                                                        encoder_hidden_states,
                                                        c_shift_msa,
                                                        c_scale_msa,
                                                        c_gate_msa,
                                                        attention_mask,
                                                        cross_attention_mask,
                                                        update_cache=0,
                                                        cache_name=cache_name)
        return hidden_states


class WanTransformer3DModel(ModelMixin, ConfigMixin):
    r"""
    TODO
    """
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
                                        # "patch_embedding", 
                                        "patch_embedding_mlp",
                                        "vggt_patch_embedding_mlp",
                                        "condition_embedder", 
                                        'condition_embedder_action',
                                        "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", 
                             "scale_shift_table", 
                             "scale_shift_table_action",
                             "norm1", 
                             'action_norm1',
                             'text_norm1',
                             "norm2", 
                             'action_norm2',
                             'text_norm2',
                             "norm3",
                             'action_norm3',
                             'text_norm3'
                             ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(self,
                 patch_size=[1, 2, 2],
                 vggt_patch_size=[1, 1, 1],
                 num_attention_heads=24,
                 attention_head_dim=128,
                 in_channels=48,
                 out_channels=48,
                 vggt_in_channels=2048,
                 vggt_out_channels=2048,
                 action_dim=30,
                 text_dim=4096,
                 freq_dim=256,
                 ffn_dim=14336,
                 num_layers=30,
                 cross_attn_norm=True,
                 eps=1e-06,
                 rope_max_seq_len=1024,
                 pos_embed_seq_len=None,
                 attn_mode="torch"):
        r"""
        TODO
        """
        super().__init__()
        self.patch_size = patch_size
        self.vggt_patch_size = vggt_patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size,
                                      rope_max_seq_len)
        self.patch_embedding_mlp = nn.Linear(
            in_channels * patch_size[0] * patch_size[1] * patch_size[2],
            inner_dim)
        self.vggt_patch_embedding_mlp = nn.Linear(
            vggt_in_channels * vggt_patch_size[0] * vggt_patch_size[1] * vggt_patch_size[2],
            inner_dim)
        self.video_modality_embedding = nn.Parameter(
            torch.zeros(1, 1, 1, inner_dim)
        )
        self.vggt_modality_embedding = nn.Parameter(
            torch.zeros(1, 1, 1, inner_dim)
        )
        self.action_embedder = nn.Linear(action_dim, inner_dim)
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        self.condition_embedder_action = deepcopy(self.condition_embedder)

        self.blocks = nn.ModuleList([
            WanTransformerBlock(inner_dim,
                                ffn_dim,
                                num_attention_heads,
                                cross_attn_norm,
                                eps,
                                attn_mode=attn_mode) for _ in range(num_layers)
        ])

        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim,
                                  out_channels * math.prod(patch_size))
        self.vggt_proj_out = nn.Linear(inner_dim,
                                       vggt_out_channels * math.prod(vggt_patch_size))
        self.action_proj_out = nn.Linear(inner_dim, action_dim)
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5)

    def clear_cache(self, cache_name):
        for block in self.blocks:
            block.attn1.clear_cache(cache_name)

    def clear_pred_cache(self, cache_name):
        for block in self.blocks:
            block.attn1.clear_pred_cache(cache_name)

    @staticmethod
    def cache_token_capacity(
        attn_window,
        visual_token_per_chunk,
        action_token_per_chunk,
    ):
        return (attn_window // 2) * (
            visual_token_per_chunk + action_token_per_chunk
        )

    def create_empty_cache(self, cache_name, attn_window,
                           visual_token_per_chunk, action_token_per_chunk,
                           device, dtype, batch_size):
        total_tolen = self.cache_token_capacity(
            attn_window,
            visual_token_per_chunk,
            action_token_per_chunk,
        )
        for block in self.blocks:
            block.attn1.init_kv_cache(cache_name, total_tolen,
                                      self.num_attention_heads,
                                      self.attention_head_dim, device, dtype, batch_size)
    
    def _input_embed(self, latents, input_type='latent'):
        if input_type == 'latent':
            hidden_states = rearrange(
                latents,
                'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2])
            hidden_states = self.patch_embedding_mlp(hidden_states)
        elif input_type == 'vggt_latent':
            hidden_states = rearrange(
                latents,
                'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                p1=self.vggt_patch_size[0],
                p2=self.vggt_patch_size[1],
                p3=self.vggt_patch_size[2])
            hidden_states = self.vggt_patch_embedding_mlp(hidden_states)
        elif input_type == 'action':
            hidden_states = rearrange(latents, 'b c f h w -> b (f h w) c')
            hidden_states = self.action_embedder(hidden_states)
        elif input_type == 'text':
            hidden_states = self.condition_embedder.text_embedder(latents)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
        return hidden_states

    def _pack_visual_hidden_states(self, video_latents, vggt_latents):
        if video_latents.shape[2] % self.patch_size[0] != 0:
            raise ValueError(
                f"video frame count {video_latents.shape[2]} is not divisible by "
                f"temporal patch size {self.patch_size[0]}"
            )
        if vggt_latents.shape[2] % self.vggt_patch_size[0] != 0:
            raise ValueError(
                f"VGGT frame count {vggt_latents.shape[2]} is not divisible by "
                f"temporal patch size {self.vggt_patch_size[0]}"
            )
        video_frames = video_latents.shape[2] // self.patch_size[0]
        vggt_frames = vggt_latents.shape[2] // self.vggt_patch_size[0]
        if video_frames != vggt_frames:
            raise ValueError(
                "video and VGGT must produce the same number of token frames, got "
                f"{video_frames} and {vggt_frames}"
            )

        video_tokens = self._input_embed(video_latents, input_type='latent')
        vggt_tokens = self._input_embed(vggt_latents, input_type='vggt_latent')
        if video_tokens.shape[1] % video_frames != 0:
            raise ValueError("video token count is not divisible by its frame count")
        if vggt_tokens.shape[1] % vggt_frames != 0:
            raise ValueError("VGGT token count is not divisible by its frame count")

        video_tokens_per_frame = video_tokens.shape[1] // video_frames
        vggt_tokens_per_frame = vggt_tokens.shape[1] // vggt_frames
        video_tokens = video_tokens.reshape(
            video_tokens.shape[0], video_frames, video_tokens_per_frame, -1
        )
        vggt_tokens = vggt_tokens.reshape(
            vggt_tokens.shape[0], vggt_frames, vggt_tokens_per_frame, -1
        )
        video_tokens = video_tokens + self.video_modality_embedding
        vggt_tokens = vggt_tokens + self.vggt_modality_embedding
        layout = VisualTokenLayout(
            num_frames=video_frames,
            video_tokens_per_frame=video_tokens_per_frame,
            vggt_tokens_per_frame=vggt_tokens_per_frame,
        )
        return pack_visual_tokens(video_tokens, vggt_tokens), layout

    def _pack_visual_grid_ids(
        self,
        video_grid_ids,
        vggt_grid_ids,
        num_frames,
        video_tokens_per_frame,
        vggt_tokens_per_frame,
    ):
        if (
            video_grid_ids.ndim != 3
            or vggt_grid_ids.ndim != 3
            or video_grid_ids.shape[1] != 4
            or vggt_grid_ids.shape[1] != 4
            or video_grid_ids.shape[0] != vggt_grid_ids.shape[0]
        ):
            raise ValueError(
                "video and VGGT grid IDs must have matching batch size and shape [B, 4, L]"
            )
        expected_video_tokens = num_frames * video_tokens_per_frame
        expected_vggt_tokens = num_frames * vggt_tokens_per_frame
        if video_grid_ids.shape[2] != expected_video_tokens:
            raise ValueError(
                f"expected {expected_video_tokens} video grid IDs, got {video_grid_ids.shape[2]}"
            )
        if vggt_grid_ids.shape[2] != expected_vggt_tokens:
            raise ValueError(
                f"expected {expected_vggt_tokens} VGGT grid IDs, got {vggt_grid_ids.shape[2]}"
            )
        video_grid_ids = video_grid_ids.transpose(1, 2).reshape(
            video_grid_ids.shape[0], num_frames, video_tokens_per_frame, 4
        )
        vggt_grid_ids = vggt_grid_ids.transpose(1, 2).reshape(
            vggt_grid_ids.shape[0], num_frames, vggt_tokens_per_frame, 4
        )
        return pack_visual_tokens(video_grid_ids, vggt_grid_ids).flatten(1, 2).transpose(1, 2)

    def _pack_visual_values(self, video_values, vggt_values, layout):
        expected_video = layout.num_frames * layout.video_tokens_per_frame
        expected_vggt = layout.num_frames * layout.vggt_tokens_per_frame
        if video_values.shape[1] != expected_video:
            raise ValueError(
                f"expected {expected_video} video values, got {video_values.shape[1]}"
            )
        if vggt_values.shape[1] != expected_vggt:
            raise ValueError(
                f"expected {expected_vggt} VGGT values, got {vggt_values.shape[1]}"
            )
        video_values = video_values.reshape(
            video_values.shape[0],
            layout.num_frames,
            layout.video_tokens_per_frame,
            *video_values.shape[2:],
        )
        vggt_values = vggt_values.reshape(
            vggt_values.shape[0],
            layout.num_frames,
            layout.vggt_tokens_per_frame,
            *vggt_values.shape[2:],
        )
        return torch.cat([video_values, vggt_values], dim=2).flatten(1, 2)

    def _unpack_visual_values(self, joint_values, layout):
        expected_tokens = layout.num_frames * layout.tokens_per_frame
        if joint_values.shape[1] != expected_tokens:
            raise ValueError(
                f"expected {expected_tokens} joint visual values, got {joint_values.shape[1]}"
            )
        joint_values = joint_values.reshape(
            joint_values.shape[0],
            layout.num_frames,
            layout.tokens_per_frame,
            *joint_values.shape[2:],
        )
        video_values, vggt_values = unpack_visual_tokens(
            joint_values,
            layout.video_tokens_per_frame,
            layout.vggt_tokens_per_frame,
        )
        return video_values.flatten(1, 2), vggt_values.flatten(1, 2)

    def _pack_visual_time_embeddings(
        self,
        video_timesteps,
        vggt_timesteps,
        latent_dict,
        vggt_dict,
        layout,
        dtype,
    ):
        if (
            video_timesteps.shape != vggt_timesteps.shape
            or not torch.equal(video_timesteps, vggt_timesteps)
        ):
            raise ValueError(
                "video and VGGT must share identical timesteps in joint visual diffusion"
            )
        video_temb, video_timestep_proj = self._time_embed(
            video_timesteps,
            latent_dict['noisy_latents'].shape[-2],
            latent_dict['noisy_latents'].shape[-1],
            dtype=dtype,
        )
        vggt_temb, vggt_timestep_proj = self._time_embed(
            vggt_timesteps,
            vggt_dict['noisy_latents'].shape[-2],
            vggt_dict['noisy_latents'].shape[-1],
            dtype=dtype,
            vggt_mode=True,
        )
        return (
            self._pack_visual_values(video_temb, vggt_temb, layout),
            self._pack_visual_values(
                video_timestep_proj, vggt_timestep_proj, layout
            ),
        )

    def _time_embed(self, timesteps, H, W, dtype, vggt_mode=False, action_mode=False):
        if action_mode:
            pach_scale_h, pach_scale_w = (1, 1)
        elif vggt_mode:
            pach_scale_h, pach_scale_w = (self.vggt_patch_size[1], self.vggt_patch_size[2])
        else:
            pach_scale_h, pach_scale_w = (self.patch_size[1], self.patch_size[2])
        latent_time_steps = torch.repeat_interleave(
            timesteps,
            (H // pach_scale_h) *
            (W // pach_scale_w), dim=1)  # L
        current_condition_embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        temb, timestep_proj = current_condition_embedder(
            latent_time_steps, dtype=dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C
        return temb, timestep_proj

    def forward_train(self, input_dict):
        input_dict['latent_dict']['noisy_latents'] = input_dict['latent_dict']['noisy_latents'].to(torch.bfloat16)
        input_dict['latent_dict']['latent'] = input_dict['latent_dict']['latent'].to(torch.bfloat16)
        input_dict['vggt_dict']['noisy_latents'] = input_dict['vggt_dict']['noisy_latents'].to(torch.bfloat16)
        input_dict['vggt_dict']['latent'] = input_dict['vggt_dict']['latent'].to(torch.bfloat16)
        input_dict['action_dict']['noisy_latents'] = input_dict['action_dict']['noisy_latents'].to(torch.bfloat16)
        input_dict['action_dict']['latent'] = input_dict['action_dict']['latent'].to(torch.bfloat16)

        latent_dict = input_dict['latent_dict']
        vggt_dict = input_dict['vggt_dict']
        action_dict = input_dict['action_dict']
        batch_size = latent_dict['noisy_latents'].shape[0]

        joint_visual_states, layout = self._pack_visual_hidden_states(
            latent_dict['noisy_latents'], vggt_dict['noisy_latents']
        )
        condition_joint_visual_states, condition_layout = self._pack_visual_hidden_states(
            latent_dict['latent'], vggt_dict['latent']
        )
        if condition_layout != layout:
            raise ValueError("noisy and clean visual layouts must match")
        joint_visual_states = joint_visual_states.flatten(1, 2).flatten(0, 1)[None]
        condition_joint_visual_states = condition_joint_visual_states.flatten(1, 2).flatten(0, 1)[None]
        action_hidden_states = self._input_embed(action_dict['noisy_latents'], input_type='action').flatten(0, 1)[None]
        text_hidden_states = self._input_embed(latent_dict["text_emb"], input_type='text')

        text_hidden_states = text_hidden_states.flatten(0, 1)[None]

        condition_action_hidden_states = self._input_embed(action_dict['latent'], input_type='action').flatten(0, 1)[None]

        hidden_states = torch.cat([joint_visual_states,
                                   condition_joint_visual_states,
                                   action_hidden_states, 
                                   condition_action_hidden_states], dim=1)

        visual_grid_id = self._pack_visual_grid_ids(
            latent_dict['grid_id'],
            vggt_dict['grid_id'],
            layout.num_frames,
            layout.video_tokens_per_frame,
            layout.vggt_tokens_per_frame,
        ).permute(1, 0, 2).flatten(1)[None]
        action_grid_id = action_dict['grid_id'].permute(1, 0, 2).flatten(1)[None]
        full_grid_id = torch.cat([visual_grid_id] * 2 + [action_grid_id] * 2, dim=2)

        rotary_emb = self.rope(full_grid_id)[:, :, None] 

        visual_temb, visual_timestep_proj = self._pack_visual_time_embeddings(
            latent_dict['timesteps'],
            vggt_dict['timesteps'],
            latent_dict,
            vggt_dict,
            layout,
            hidden_states.dtype,
        )
        condition_visual_temb, condition_visual_timestep_proj = self._pack_visual_time_embeddings(
            latent_dict['cond_timesteps'],
            vggt_dict['cond_timesteps'],
            latent_dict,
            vggt_dict,
            layout,
            hidden_states.dtype,
        )
        action_temb, action_timestep_proj = self._time_embed(action_dict['timesteps'],
                        action_dict['noisy_latents'].shape[-2], 
                        action_dict['noisy_latents'].shape[-1], 
                        dtype=hidden_states.dtype, 
                        action_mode=True)
        condition_action_temb, condition_action_timestep_proj = self._time_embed(
                        action_dict['cond_timesteps'],
                        action_dict['noisy_latents'].shape[-2],
                        action_dict['noisy_latents'].shape[-1],
                        dtype=hidden_states.dtype,
                        action_mode=True)
        temb = torch.cat([
            visual_temb.flatten(0, 1)[None],
            condition_visual_temb.flatten(0, 1)[None],
            action_temb.flatten(0, 1)[None],
            condition_action_temb.flatten(0, 1)[None],
        ], dim=1)
        timestep_proj = torch.cat([
            visual_timestep_proj.flatten(0, 1)[None],
            condition_visual_timestep_proj.flatten(0, 1)[None],
            action_timestep_proj.flatten(0, 1)[None],
            condition_action_timestep_proj.flatten(0, 1)[None],
        ], dim=1)

        total_length = hidden_states.shape[1]
        padded_length = (128 - total_length % 128) % 128
        hidden_states = F.pad(hidden_states, (0, 0, 0, padded_length))
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        temb = F.pad(temb, (0, 0, 0, padded_length))
        timestep_proj = F.pad(timestep_proj, (0, 0, 0, 0, 0, padded_length))

        split_list = [joint_visual_states.shape[1],
                      condition_joint_visual_states.shape[1],
                      action_hidden_states.shape[1], 
                      condition_action_hidden_states.shape[1],
                      padded_length]

        attn_mask, cross_attn_mask = SDPAAttnFunc.init_mask(
                               batch_size,
                               layout.num_frames,
                               layout.tokens_per_frame,
                               action_dict['noisy_latents'].shape, 
                               padded_length, 
                               input_dict["chunk_size"],
                               window_size=input_dict['window_size'],
                               device=hidden_states.device,
                               dtype=hidden_states.dtype)

        for block in self.blocks:
            hidden_states = block(hidden_states,
                                  text_hidden_states,
                                  timestep_proj,
                                  rotary_emb,
                                  attention_mask=attn_mask,
                                  cross_attention_mask=cross_attn_mask,
                                  update_cache=False)
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table,
                                 'b l n c -> b n l c').chunk(2, dim=1)
        shift = shift.to(hidden_states.device).squeeze(1)
        scale = scale.to(hidden_states.device).squeeze(1)
        hidden_states = (self.norm_out(hidden_states.float()) *
                                (1. + scale) +
                                shift).type_as(hidden_states)
        joint_visual_states, _, action_hidden_states, _, _ = torch.split(hidden_states, split_list, dim=1)
        joint_visual_states = joint_visual_states.reshape(
            batch_size, layout.num_frames * layout.tokens_per_frame, -1
        )
        latent_hidden_states, vggt_hidden_states = self._unpack_visual_values(
            joint_visual_states, layout
        )
        latent_hidden_states = self.proj_out(latent_hidden_states)
        latent_hidden_states = rearrange(latent_hidden_states,
                                             'b l (n c) -> b (l n) c',
                                             n=math.prod(self.patch_size))  #
        vggt_hidden_states = self.vggt_proj_out(vggt_hidden_states)
        vggt_hidden_states = rearrange(vggt_hidden_states,
                                             'b l (n c) -> b (l n) c',
                                             n=math.prod(self.vggt_patch_size))  #
        action_hidden_states = self.action_proj_out(action_hidden_states)
        action_hidden_states = rearrange(action_hidden_states,
                                             '1 (b l) c -> b l c',
                                             b=batch_size)  #

        return latent_hidden_states, vggt_hidden_states, action_hidden_states

    def _infer_joint(
        self,
        latent_dict,
        vggt_dict,
        update_cache=0,
        cache_name="pos",
    ):
        hidden_states, layout = self._pack_visual_hidden_states(
            latent_dict['noisy_latents'], vggt_dict['noisy_latents']
        )
        hidden_states = hidden_states.flatten(1, 2)

        text_hidden_states = self.condition_embedder.text_embedder(latent_dict["text_emb"])
        full_grid_id = self._pack_visual_grid_ids(
            latent_dict['grid_id'],
            vggt_dict['grid_id'],
            layout.num_frames,
            layout.video_tokens_per_frame,
            layout.vggt_tokens_per_frame,
        )
        rotary_emb = self.rope(full_grid_id)[:, :, None]

        temb, timestep_proj = self._pack_visual_time_embeddings(
            latent_dict['timesteps'],
            vggt_dict['timesteps'],
            latent_dict,
            vggt_dict,
            layout,
            hidden_states.dtype,
        )

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                text_hidden_states,
                timestep_proj,
                rotary_emb,
                update_cache=update_cache,
                cache_name=cache_name,
            )

        temb_scale_shift_table = (self.scale_shift_table[None] + temb[:, :, None])
        shift, scale = rearrange(temb_scale_shift_table, 'b l n c -> b n l c').chunk(2, dim=1)
        hidden_states = (self.norm_out(hidden_states.float()) *
            (1. + scale.squeeze(1).to(hidden_states.device)) +
            shift.squeeze(1).to(hidden_states.device)).type_as(hidden_states)

        latent_hidden_states, vggt_hidden_states = self._unpack_visual_values(
            hidden_states, layout
        )
        latent_hidden_states = self.proj_out(latent_hidden_states)
        latent_hidden_states = rearrange(
            latent_hidden_states,
            'b l (n c) -> b (l n) c',
            n=math.prod(self.patch_size),
        )
        vggt_hidden_states = self.vggt_proj_out(vggt_hidden_states)
        vggt_hidden_states = rearrange(
            vggt_hidden_states,
            'b l (n c) -> b (l n) c',
            n=math.prod(self.vggt_patch_size),
        )
        return latent_hidden_states, vggt_hidden_states

    def forward(
        self,
        input_dict,
        vggt_dict=None,
        update_cache=0,
        cache_name="pos",
        action_mode=False,
        train_mode=False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if train_mode:
            return self.forward_train(input_dict)
        if vggt_dict is not None:
            return self._infer_joint(
                input_dict,
                vggt_dict,
                update_cache=update_cache,
                cache_name=cache_name,
            )
        if action_mode:  # action input emb
            latent_hidden_states = rearrange(input_dict['noisy_latents'],
                                             'b c f h w -> b (f h w) c')
            latent_hidden_states = self.action_embedder(
                latent_hidden_states)  # B L1 C
        else:  # latent input emb
            latent_hidden_states = rearrange(
                input_dict['noisy_latents'],
                'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2])
            latent_hidden_states = self.patch_embedding_mlp(
                latent_hidden_states)
        text_hidden_states = self.condition_embedder.text_embedder(
            input_dict["text_emb"])  # B L2 C

        latent_grid_id = input_dict['grid_id']
        rotary_emb = self.rope(latent_grid_id)[:, :, None]  # 1 L 1 C
        pach_scale_h, pach_scale_w = (1, 1) if action_mode else (
            self.patch_size[1], self.patch_size[2])

        latent_time_steps = torch.repeat_interleave(
            input_dict['timesteps'],
            (input_dict['noisy_latents'].shape[-2] // pach_scale_h) *
            (input_dict['noisy_latents'].shape[-1] // pach_scale_w), dim=1)  # L
        current_condition_embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        temb, timestep_proj = current_condition_embedder(
            latent_time_steps, dtype=latent_hidden_states.dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C

        for block in self.blocks:
            latent_hidden_states = block(latent_hidden_states,
                                         text_hidden_states,
                                         timestep_proj,
                                         rotary_emb,
                                         update_cache=update_cache,
                                         cache_name=cache_name)
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table,
                                 'b l n c -> b n l c').chunk(2, dim=1)
        shift = shift.to(latent_hidden_states.device).squeeze(1)
        scale = scale.to(latent_hidden_states.device).squeeze(1)
        latent_hidden_states = (self.norm_out(latent_hidden_states.float()) *
                                (1. + scale) +
                                shift).type_as(latent_hidden_states)

        if action_mode:
            latent_hidden_states = self.action_proj_out(latent_hidden_states)
        else:
            latent_hidden_states = self.proj_out(latent_hidden_states)
            latent_hidden_states = rearrange(latent_hidden_states,
                                             'b l (n c) -> b (l n) c',
                                             n=math.prod(self.patch_size))  #

        return latent_hidden_states


if __name__ == '__main__':
    model = WanTransformer3DModel(patch_size=[1, 2, 2],
                                  vggt_patch_size=[1, 1, 1],
                                  num_attention_heads=24,
                                  attention_head_dim=128,
                                  in_channels=48,
                                  out_channels=48,
                                  vggt_in_channels=2048,
                                  vggt_out_channels=2048,
                                  action_dim=30,
                                  text_dim=4096,
                                  freq_dim=256,
                                  ffn_dim=14336,
                                  num_layers=30,
                                  cross_attn_norm=True,
                                  eps=1e-6,
                                  rope_max_seq_len=1024,
                                  pos_embed_seq_len=None,
                                  attn_mode="torch")
    print(model)
