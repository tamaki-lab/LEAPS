import torch


class BatchedVideo3DPatches:
    def __init__(self, tensor):
        self.cubes: torch.Tensor = tensor
        assert tensor.dim() == 8, (
            "tensor should be 7D tensor (B, T_num_cube, H_num_cube, W_num_cube, t_cube_len, h_cube_len, w_cube_len, C)"
        )

    def get_num_cubes(self):
        return self.cubes.size(1) * self.cubes.size(2) * self.cubes.size(3)

    def get_num_cubes_per_frame(self):
        return self.cubes.size(2) * self.cubes.size(3)

    def get_temporal_num_cubes(self):
        return self.cubes.size(1)

    def get_flatten_cubes(self):
        return self.cubes.reshape(
            self.cubes.size(0),
            self.get_num_cubes(),
            self.cubes.size(4),
            self.cubes.size(5),
            self.cubes.size(6),
            self.cubes.size(7),
        )


def create_3d_patches(video_tensor: torch.Tensor, t_patch: int = 2, s_patch: int = 16):
    """
    convert video tensor to 3D patches.
    Args:
        video_tensor: (B, T, C, H, W)
        t_patch: temporal patch size (default: 2).
        s_patch: spatial patch size (default: 16).

    Returns:
        patches: (B, num_t_patches, num_h_patches, num_w_patches, t_patch, s_patch, s_patch, C)
    """
    B, T, C, H, W = video_tensor.shape

    assert T % t_patch == 0, "frames should be divisible by t_patch."
    num_t_patches = T // t_patch

    assert H % s_patch == 0 and W % s_patch == 0, "height and width should be divisible by s_patch."

    video_tensor = video_tensor.view(B, num_t_patches, t_patch, C, H, W)
    patches = video_tensor.unfold(4, s_patch, s_patch).unfold(5, s_patch, s_patch)

    # (B, num_t_patches, num_h_patches, num_w_patches, t_patch, s_patch, s_patch, C)
    patches = patches.permute(0, 1, 4, 5, 2, 6, 7, 3)

    return BatchedVideo3DPatches(patches)


def reconstruct_from_3d_patch(patches: BatchedVideo3DPatches, t_patch: int = 2, s_patch: int = 16):
    """
    convert 3D patches back to original video tensor.

    Args:
        patches: (B, num_t_patches, num_h_patches, num_w_patches, t_patch, s_patch, s_patch, C)
        t_patch: tubelet_size。
        s_patch: spatial_patch_size。

    Returns:
        video_tensor: (B, T, C, H, W)。
    """
    B, num_t_patches, num_h_patches, num_w_patches, _, _, _, C = patches.cubes.shape

    #
    T = num_t_patches * t_patch
    H = num_h_patches * s_patch
    W = num_w_patches * s_patch

    # reconstruct
    patches.cubes = patches.cubes.permute(0, 1, 4, 7, 2, 5, 3, 6)  # B, num_t_patches, t_patch, C, num_h, s, num_w, s)
    video = patches.cubes.reshape(B, T, C, H, W)

    return video


def make_mask_video(video_tensor, cube_mask, mean, std):
    video_tensor = video_tensor * std + mean
    video_cubes = create_3d_patches(video_tensor, t_patch=2, s_patch=16)
    flatten_video_cubes = video_cubes.get_flatten_cubes()

    yellow_green = torch.tensor([0.7, 0.8, 0.2], device=video_tensor.device)
    cube_mask = cube_mask[..., None, None, None, None]
    flatten_video_cubes = torch.where(cube_mask, yellow_green[None, None, None, None, None, ...], flatten_video_cubes)
    video_cubes = BatchedVideo3DPatches(flatten_video_cubes.reshape(video_cubes.cubes.shape))
    video_tensor = reconstruct_from_3d_patch(video_cubes, t_patch=2, s_patch=16)
    return video_tensor


def make_entropy_heatmap_video(video_tensor, entropy_per_cube, mean, std, tubelet_size=2, patch_size=16):
    mean = torch.tensor(mean).to(device=video_tensor.device)
    std = torch.tensor(std).to(device=video_tensor.device)
    mean = mean[None, None, :, None, None]  # (1, 1, 3, 1, 1)
    std = std[None, None, :, None, None]  # (1, 1, 3, 1, 1)
    video_tensor = video_tensor * std + mean
    video_tensor = torch.clamp(video_tensor, 0.0, 1.0)

    video_cubes = create_3d_patches(video_tensor, t_patch=tubelet_size, s_patch=patch_size)
    flatten_video_cubes = video_cubes.get_flatten_cubes()

    r = torch.where(
        entropy_per_cube < 0.5,
        entropy_per_cube * 2.0,  # 0〜0.5 red increases
        torch.ones_like(entropy_per_cube),
    )  # 0.5〜1.0 red=1
    g = torch.where(
        entropy_per_cube < 0.5,
        entropy_per_cube * 2.0,  # 0〜0.5 green increases (towards yellow)
        (1.0 - (entropy_per_cube - 0.5) * 2.0).clamp(0.0, 1.0),
    )  # 0.5〜1.0 green decreases
    b = torch.where(
        entropy_per_cube < 0.5,
        (1.0 - entropy_per_cube * 2.0).clamp(0.0, 1.0),  # 0〜0.5 blue decreases
        torch.zeros_like(entropy_per_cube),
    )  # 0.5〜1.0 blue=0

    # bs, seq_len = entropy_per_cube.shape

    # Stack the RGB channels
    heatmap_rgb = torch.stack([r, g, b], dim=-1)
    heatmap_rgb = heatmap_rgb[..., None, None, None].permute(0, 1, 3, 4, 5, 2)
    alpha = 0.5
    flatten_video_cubes = (1.0 - alpha) * flatten_video_cubes + alpha * heatmap_rgb

    video_cubes = BatchedVideo3DPatches(flatten_video_cubes.reshape(video_cubes.cubes.shape))
    video_tensor = reconstruct_from_3d_patch(video_cubes, t_patch=tubelet_size, s_patch=patch_size)
    return video_tensor


def make_entropy_divide_map_video(video_tensor, mask_per_tokens, mean, std, tubelet_size=2, patch_size=16):
    mean = torch.tensor(mean).to(device=video_tensor.device)
    std = torch.tensor(std).to(device=video_tensor.device)
    mean = mean[None, None, :, None, None]  # (1, 1, 3, 1, 1)
    std = std[None, None, :, None, None]  # (1, 1, 3, 1, 1)
    video_tensor = video_tensor * std + mean
    video_cubes = create_3d_patches(video_tensor, t_patch=tubelet_size, s_patch=patch_size)
    flatten_video_cubes = video_cubes.get_flatten_cubes()

    if mask_per_tokens is not None:
        opacity = 1.0
        mask_per_tokens = (~mask_per_tokens).to(torch.float16)
        white = torch.ones_like(flatten_video_cubes).float() * opacity * mask_per_tokens[..., None, None, None, None]

        opacity_mask = torch.ones_like(flatten_video_cubes).float() * (
            1.0 - opacity * mask_per_tokens[..., None, None, None, None]
        )
        flatten_video_cubes = flatten_video_cubes * opacity_mask + white

    video_cubes = BatchedVideo3DPatches(flatten_video_cubes.reshape(video_cubes.cubes.shape))
    video_tensor = reconstruct_from_3d_patch(video_cubes, t_patch=tubelet_size, s_patch=patch_size)
    return video_tensor
