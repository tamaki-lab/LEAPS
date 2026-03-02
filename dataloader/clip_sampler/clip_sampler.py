import random
import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class ClipSamplerArgs:
    num_frames: int
    num_views_for_sequencial: int
    target_fps: float


def uniform_clip_sampler(frame_sec_list, clip_sampler_args: ClipSamplerArgs):
    n_frames = clip_sampler_args.num_frames
    if n_frames <= len(frame_sec_list):
        return np.linspace(0, len(frame_sec_list) - 1, num=n_frames).astype(int)
    else:
        diff = n_frames - len(frame_sec_list)
        is_even = diff % 2 == 0
        pre = np.zeros(diff // 2 if is_even else diff // 2 + 1, dtype=np.int8)
        mid = np.array(range(len(frame_sec_list)))
        suf = np.ones((diff // 2), dtype=np.int8) * (len(frame_sec_list) - 1)
        return np.concatenate([pre, mid, suf])


def center_frame_clip_sampler(frame_sec_list, clip_sampler_args: ClipSamplerArgs):
    n_frames = clip_sampler_args.num_frames
    if n_frames != 1:
        warnings.warn(
            "when center_frame clip sampler is selected, args.clip_frames is ignored and only one frame is fetched."
        )
    return np.array([len(frame_sec_list) // 2])


def sample_frames_within_duration(frame_sec_list, start_time: float, target_duration: float, n_frames: int):
    end_time = start_time + target_duration
    frame_indices = [i for i, t in enumerate(frame_sec_list) if start_time <= t <= end_time]

    if len(frame_indices) >= n_frames:
        sampled_indices = np.linspace(0, len(frame_indices) - 1, n_frames)
        return np.round(np.array(frame_indices)[sampled_indices.astype(int)]).astype(int)
    else:
        sampled_indices = np.linspace(0, len(frame_indices) - 1, n_frames)
        return np.round(
            np.array(frame_indices)[np.clip(sampled_indices.astype(int), 0, len(frame_indices) - 1)]
        ).astype(int)


def random_clip_sampler(frame_sec_list, clip_sampler_args: ClipSamplerArgs):
    target_fps = clip_sampler_args.target_fps

    if isinstance(frame_sec_list, int):
        # create dummy frame_sec_list
        input_num_frames = frame_sec_list
        frame_sec_list = [frame_idx / target_fps for frame_idx in range(input_num_frames)]

    n_frames = clip_sampler_args.num_frames

    if len(frame_sec_list) == 0:
        return np.array([], dtype=int)

    target_duration = n_frames / target_fps
    max_start_time = frame_sec_list[-1] - target_duration
    if max_start_time <= 0:
        indices = np.linspace(0, len(frame_sec_list) - 1, n_frames)
        return np.round(indices).astype(int)

    start_time = random.uniform(0, max_start_time)
    return sample_frames_within_duration(frame_sec_list, start_time, target_duration, n_frames)


def multi_sequential_clip_sampler(frame_sec_list, clip_sampler_args: ClipSamplerArgs) -> list[np.ndarray]:
    n_frames = clip_sampler_args.num_frames
    num_views = clip_sampler_args.num_views_for_sequencial
    target_fps = clip_sampler_args.target_fps

    if isinstance(frame_sec_list, int):
        # create dummy frame_sec_list
        input_num_frames = frame_sec_list
        frame_sec_list = [frame_idx / target_fps for frame_idx in range(input_num_frames)]

    if len(frame_sec_list) == 0:
        return [np.array([], dtype=int) for _ in range(num_views)]

    target_duration = n_frames / target_fps
    total_duration = frame_sec_list[-1]
    available_duration = total_duration - target_duration

    if available_duration <= 0:
        indices = np.linspace(0, len(frame_sec_list) - 1, n_frames)
        return [np.round(indices).astype(int) for _ in range(num_views)]

    step = available_duration / (num_views + 1)
    res = []
    for i in range(num_views):
        start_time = step * (i + 1)
        indices = sample_frames_within_duration(frame_sec_list, start_time, target_duration, n_frames)
        res.append(indices)

    return res


def space_evenly_multi_view_clip_sampler(frame_sec_list, clip_sampler_args: ClipSamplerArgs) -> list[np.ndarray]:
    n_frames = clip_sampler_args.num_frames
    num_views = clip_sampler_args.num_views_for_sequencial
    target_fps = clip_sampler_args.target_fps

    if isinstance(frame_sec_list, int):
        # create dummy frame_sec_list
        input_num_frames = frame_sec_list
        frame_sec_list = [frame_idx / target_fps for frame_idx in range(input_num_frames)]

    if len(frame_sec_list) == 0:
        return [np.array([], dtype=int) for _ in range(num_views)]

    t0 = frame_sec_list[0]
    tN = frame_sec_list[-1]
    seg_start_times = []
    seg_end_times = []

    clip_duration = n_frames / target_fps
    video_duration = tN - t0

    segment_duraiton = video_duration / num_views

    result = []
    if video_duration < clip_duration:
        indices = np.linspace(0, len(frame_sec_list) - 1, n_frames)
        return [np.round(indices).astype(int) for _ in range(num_views)]
    elif segment_duraiton < clip_duration:
        available_duration = video_duration - clip_duration
        seg_start_times = np.linspace(t0, t0 + available_duration, num_views)
        seg_end_times = seg_start_times + clip_duration
    else:
        for i in range(num_views):
            seg_start_times.append(t0 + i * segment_duraiton)
            seg_end_times.append(t0 + (i + 1) * segment_duraiton)
    seg_end_times[-1] = tN

    for i in range(num_views):
        seg_t0 = seg_start_times[i]
        seg_tN = seg_end_times[i]
        seg_duration = seg_tN - seg_t0
        center_time = (seg_duration / 2.0) + seg_t0
        start_time = center_time - (clip_duration / 2.0)

        result.append(sample_frames_within_duration(frame_sec_list, start_time, clip_duration, n_frames))
    return result


def sparse_video_sampler(frame_sec_list, clip_sampler_args: ClipSamplerArgs) -> list[np.ndarray]:
    model_input_base_n_frames = clip_sampler_args.num_frames

    times = len(frame_sec_list) // model_input_base_n_frames
    n_frames = times * model_input_base_n_frames

    clip_sampler_args.num_views_for_sequencial = 1
    clip_sampler_args.num_frames = n_frames

    return space_evenly_multi_view_clip_sampler(frame_sec_list, clip_sampler_args)


def configure_sampler(is_train: bool):
    return random_clip_sampler if is_train else space_evenly_multi_view_clip_sampler
