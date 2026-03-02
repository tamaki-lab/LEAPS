import os
from functools import partial
from glob import glob

import torch
import torchvision.io as io
from tqdm import tqdm

from args import ArgParse
from dataloader.clip_sampler.clip_sampler import ClipSamplerArgs, sparse_video_sampler
from dataloader.transforms import TransformVideoInfo, transform_video
from model.core.video_voxel_heatmap import make_entropy_divide_map_video, make_entropy_heatmap_video
from model.leaps.leaps import LEAPS, LeapsOutput, create_leaps_128_4h_6l_pretrained


def write_heatmap_video(
    pixel_values: torch.Tensor,
    leap_mask: torch.Tensor | None = None,
    entropy_score: torch.Tensor | None = None,
    avg_color=(0.45, 0.45, 0.45),
    std_color=(0.225, 0.225, 0.225),
    tubelet_size=2,
    patch_size=16,
    output_path="temp_heatmap_video.mp4",
    fps=7.5,
):
    video = pixel_values

    if entropy_score is not None:

        def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
            min_val = torch.min(tensor)
            max_val = torch.max(tensor)
            normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-8)
            return normalized_tensor

        entropy = normalize_tensor(entropy_score)

        masked_video = make_entropy_heatmap_video(
            video,
            entropy,
            mean=tuple(avg_color),
            std=tuple(std_color),
            tubelet_size=tubelet_size,  # type: ignore
            patch_size=patch_size,  # type: ignore
        )
    else:
        assert leap_mask is not None, "Either entropy_score or leap_mask must be provided."
        masked_video = make_entropy_divide_map_video(
            video,
            leap_mask,
            mean=tuple(avg_color),
            std=tuple(std_color),
            tubelet_size=tubelet_size,  # type: ignore
            patch_size=patch_size,  # type: ignore
        )
    masked_video = (masked_video * 255).permute(0, 1, 3, 4, 2).squeeze(0).detach().cpu().to(dtype=torch.uint8)
    io.write_video(output_path, masked_video, fps=fps, video_codec="libx264")


def main():
    assert torch.cuda.is_available()

    args = ArgParse.get()

    num_frames = args.frames_per_clip

    leaps: LEAPS = create_leaps_128_4h_6l_pretrained(args.leaps_pth, num_frames=num_frames).to("cuda")

    src_videos_dir = args.visualize_videos_src
    video_paths = glob(os.path.join(src_videos_dir, "*.mp4"))

    for video_path in tqdm(video_paths):
        video_name = os.path.basename(video_path).split(".")[0]
        dst_video_dir = os.path.join(args.visualize_videos_dest, video_name)
        os.makedirs(dst_video_dir, exist_ok=True)

        # load video to tensor
        video, _, info = io.read_video(video_path, pts_unit="sec", output_format="TCHW")

        fps = info["video_fps"]
        frame_sec_list = [i / fps for i in range(video.shape[0])]

        sample_frame_indices = sparse_video_sampler(
            frame_sec_list,
            clip_sampler_args=ClipSamplerArgs(
                num_frames=num_frames,
                num_views_for_sequencial=1,
                target_fps=7.5,
            ),
        )[0]

        clip = video[sample_frame_indices]  # (T, C, H, W)
        _, transform = transform_video(TransformVideoInfo(resize_size=224, crop_size=224))
        clip = transform(clip).to("cuda")  # (num_frames, C, H, W)

        output: LeapsOutput = leaps(clip.unsqueeze(0), remain_token_rate=args.remain_token_rate)

        entropy_st = output.entropy_st  # (1, num_patches)
        entropy_s = output.entropy_space  # (1, num_patches)
        leap_mask = output.leap_mask  # (1, num_patches)

        writer = partial(
            write_heatmap_video,
            avg_color=tuple(args.avg_color),
            std_color=tuple(args.std_color),
            tubelet_size=2,  # type: ignore
            patch_size=16,  # type: ignore
            fps=15,
        )

        writer(
            pixel_values=clip,
            entropy_score=entropy_st,
            output_path=os.path.join(dst_video_dir, f"{video_name}_entropy_st.mp4"),
        )
        writer(
            pixel_values=clip,
            entropy_score=entropy_s,
            output_path=os.path.join(dst_video_dir, f"{video_name}_entropy_s.mp4"),
        )
        writer(
            pixel_values=clip,
            leap_mask=leap_mask,
            output_path=os.path.join(dst_video_dir, f"{video_name}_leap_mask_r{args.remain_token_rate}.mp4"),
        )


if __name__ == "__main__":
    main()
