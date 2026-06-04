# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import concurrent.futures

import numpy as np
import torch
import matplotlib.pyplot as plt

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

__all__ = ['get_mesh_id', 'save_async', 'data_seq_to_patch', 'visualize_attn_mask']


def data_seq_to_patch(
    patch_size,
    data_seq,
    latent_num_frames,
    latent_height,
    latent_width,
    batch_size=1,
):
    p_t, p_h, p_w = patch_size
    post_patch_num_frames = latent_num_frames // p_t
    post_patch_height = latent_height // p_h
    post_patch_width = latent_width // p_w

    data_patch = data_seq.reshape(batch_size, post_patch_num_frames,
                                  post_patch_height, post_patch_width, p_t,
                                  p_h, p_w, -1)
    data_patch = data_patch.permute(0, 7, 1, 4, 2, 5, 3, 6)
    data_patch = data_patch.flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return data_patch


def get_mesh_id(f, h, w, t, f_w=1, f_shift=0, action=False):
    f_idx = torch.arange(f_shift, f + f_shift) * f_w
    h_idx = torch.arange(h)
    w_idx = torch.arange(w)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing='ij')
    if action:
        ff_offset = (torch.ones([h]).cumsum(0) / (h + 1)).view(1, -1, 1)
        ff = ff + ff_offset
        hh = torch.ones_like(hh) * -1
        ww = torch.ones_like(ww) * -1

    grid_id = torch.cat(
        [
            ff.unsqueeze(0),
            hh.unsqueeze(0),
            ww.unsqueeze(0),
        ],
        dim=0,
    ).flatten(1)
    grid_id = torch.cat([grid_id, torch.full_like(grid_id[:1], t)], dim=0)
    return grid_id


def save_async(obj, file_path):
    """
    todo
    """
    if torch.is_tensor(obj) or (isinstance(obj, dict) and any(
            torch.is_tensor(v) for v in obj.values())):
        if torch.is_tensor(obj):
            if obj.is_cuda:
                obj = obj.cpu()
        elif isinstance(obj, dict):
            obj = {
                k: v.cpu() if torch.is_tensor(v) else v
                for k, v in obj.items()
            }
        executor.submit(torch.save, obj, file_path)
    elif isinstance(obj, np.ndarray):
        obj_copy = obj.copy()
        executor.submit(np.save, file_path, obj_copy)
    else:
        executor.submit(torch.save, obj, file_path)

def sample_timestep_id(
    batch_size: int = 1,
    min_timestep_bd: float = 0.0,
    max_timestep_bd: float = 1.0,
    num_train_timesteps: int = 1000,
):
    u = torch.rand(size=[batch_size])
    u = u * (max_timestep_bd - min_timestep_bd) + min_timestep_bd
    timestep_id = (u * num_train_timesteps).clamp(min=0, max=num_train_timesteps - 1).to(torch.int64)
    return timestep_id


def warmup_constant_lambda(current_step, warmup_steps=1000):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0


def visualize_attn_mask(mask, tokens=None, title="Attention Mask", save_path="attention_mask.png"):
    """
    将attention mask可视化为热力图并保存为图片

    Args:
        mask: attention mask张量，形状可以是 [S, S], [B, S, S], 或 [1, 1, S, S]
              True/1表示参与注意力计算，False/0表示屏蔽
        tokens: 可选的token标签列表，用于坐标轴标注
        title: 图片标题
        save_path: 保存路径
    """
    # 转换为numpy数组
    if isinstance(mask, torch.Tensor):
        # 处理不同的形状
        if mask.dim() == 4:
            # [1, 1, S, S] -> [S, S]
            mask_np = mask.squeeze().cpu().numpy()
        elif mask.dim() == 3:
            # [B, S, S] -> 取第一个batch
            mask_np = mask[0].cpu().numpy()
        else:
            # [S, S]
            mask_np = mask.cpu().numpy()
    else:
        mask_np = np.array(mask)

    # 确保是2D数组
    if mask_np.ndim != 2:
        print(f"Warning: mask shape {mask_np.shape} is not 2D, skipping visualization")
        return

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))

    # 将布尔值转换为数值（True->1, False->0）用于可视化
    if mask_np.dtype == bool:
        mask_vis = mask_np.astype(np.float32)
    else:
        mask_vis = mask_np.astype(np.float32)

    # 绘制热力图
    im = ax.imshow(mask_vis, cmap='binary', aspect='auto', vmin=0, vmax=1)

    # 设置标题和标签
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xlabel('Key/Value Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)

    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention (1=Active, 0=Masked)', fontsize=10)

    # 如果提供了token标签，设置坐标轴刻度
    if tokens is not None and len(tokens) > 0:
        num_tokens = len(tokens)
        # 如果token数量太多，只显示部分刻度
        if num_tokens > 50:
            step = num_tokens // 20
            tick_positions = list(range(0, num_tokens, step))
            tick_labels = [tokens[i] if i < len(tokens) else '' for i in tick_positions]
        else:
            tick_positions = list(range(num_tokens))
            tick_labels = tokens

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=8)

    # 添加网格线（可选）
    if mask_vis.shape[0] <= 100:  # 只在较小尺寸时显示网格
        ax.set_xticks(np.arange(-0.5, mask_vis.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, mask_vis.shape[0], 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Attention mask visualization saved to: {save_path}")
    print(f"Mask shape: {mask_vis.shape}, Active positions: {np.sum(mask_vis > 0.5)}, Masked positions: {np.sum(mask_vis < 0.5)}")
