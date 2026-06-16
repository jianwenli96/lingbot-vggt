import torch

from wan_va.utils.utils import get_mesh_id, sample_timestep_id


def _prepare_visual_branch(
    latent,
    scheduler,
    timesteps,
    cond_timesteps,
    patch_size,
    noise,
    cond_noise,
):
    batch_size, _, num_frames, height, width = latent.shape
    patch_f, patch_h, patch_w = patch_size
    if num_frames % patch_f or height % patch_h or width % patch_w:
        raise ValueError(
            f"visual latent shape {tuple(latent.shape)} is not divisible by patch size {tuple(patch_size)}"
        )

    noisy_latents = scheduler.add_noise(latent, noise, timesteps, t_dim=2)
    targets = scheduler.training_target(latent, noise, timesteps)
    clean_latents = latent
    if cond_noise is not None:
        clean_latents = scheduler.add_noise(
            latent, cond_noise, cond_timesteps, t_dim=2
        )

    grid_id = get_mesh_id(
        num_frames // patch_f,
        height // patch_h,
        width // patch_w,
        t=0,
        f_w=1,
        f_shift=0,
        action=False,
    ).to(latent.device)
    return {
        "timesteps": timesteps[None].repeat(batch_size, 1),
        "noisy_latents": noisy_latents,
        "targets": targets,
        "latent": clean_latents,
        "cond_timesteps": cond_timesteps[None].repeat(batch_size, 1),
        "grid_id": grid_id[None].repeat(batch_size, 1, 1),
    }


@torch.no_grad()
def prepare_joint_visual_noise(
    video_latents,
    vggt_latents,
    scheduler,
    video_patch_size,
    vggt_patch_size,
    device,
    noisy_cond_prob=0.0,
    timestep_ids=None,
    cond_timestep_ids=None,
    add_condition_noise=None,
):
    """Prepare synchronized video/VGGT diffusion inputs with independent noise."""
    if video_latents.shape[0] != vggt_latents.shape[0]:
        raise ValueError("video and VGGT batch sizes must match")
    if video_latents.shape[2] != vggt_latents.shape[2]:
        raise ValueError(
            "video and VGGT frame counts must match, got "
            f"{video_latents.shape[2]} and {vggt_latents.shape[2]}"
        )

    num_frames = video_latents.shape[2]
    if timestep_ids is None:
        timestep_ids = sample_timestep_id(
            batch_size=num_frames,
            num_train_timesteps=scheduler.num_train_timesteps,
        )
    if timestep_ids.numel() != num_frames:
        raise ValueError(f"expected {num_frames} timestep IDs, got {timestep_ids.numel()}")
    timesteps = scheduler.timesteps[timestep_ids].to(device=device)

    if add_condition_noise is None:
        add_condition_noise = torch.rand(1).item() < noisy_cond_prob
    if add_condition_noise:
        if cond_timestep_ids is None:
            cond_timestep_ids = sample_timestep_id(
                batch_size=num_frames,
                min_timestep_bd=0.5,
                max_timestep_bd=1.0,
                num_train_timesteps=scheduler.num_train_timesteps,
            )
        if cond_timestep_ids.numel() != num_frames:
            raise ValueError(
                f"expected {num_frames} condition timestep IDs, got {cond_timestep_ids.numel()}"
            )
        cond_timesteps = scheduler.timesteps[cond_timestep_ids].to(device=device)
        video_cond_noise = torch.randn_like(video_latents)
        vggt_cond_noise = torch.randn_like(vggt_latents)
    else:
        cond_timesteps = torch.zeros_like(timesteps)
        video_cond_noise = None
        vggt_cond_noise = None

    video_dict = _prepare_visual_branch(
        video_latents,
        scheduler,
        timesteps,
        cond_timesteps,
        video_patch_size,
        torch.randn_like(video_latents),
        video_cond_noise,
    )
    vggt_dict = _prepare_visual_branch(
        vggt_latents,
        scheduler,
        timesteps,
        cond_timesteps,
        vggt_patch_size,
        torch.randn_like(vggt_latents),
        vggt_cond_noise,
    )
    return video_dict, vggt_dict
