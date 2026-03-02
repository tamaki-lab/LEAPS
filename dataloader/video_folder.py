import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pytorchvideo.data import labeled_video_dataset
from pytorchvideo.data.clip_sampling import (
    ConstantClipsPerVideoSampler,
    RandomClipSampler,
)
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
from torchvision.transforms import v2 as transforms


@dataclass
class VideoFolderInfo:
    root: str
    train_dir: str
    val_dir: str
    batch_size: int
    num_workers: int
    train_transform: transforms
    val_transform: transforms
    clip_duration: float
    clips_per_video: int


def collate_for_video(batch: Any):
    batch_dict = torch.utils.data.default_collate(batch)
    # return batch_dict["video"], batch_dict["label"]
    return {
        "video": batch_dict["video"],
        "label": batch_dict["label"],
    }


def video_folder_loader(
    video_dir_path: str,
    num_workers: int,
    batch_size: int,
    transform,
    num_frames: int,
    target_fps: float,
    clips_per_video: int,
    is_val: bool = False,
) -> tuple[DataLoader, int]:
    clip_duration = num_frames / target_fps

    if is_val:
        clip_sampler = ConstantClipsPerVideoSampler(clip_duration=clip_duration, clips_per_video=clips_per_video)
        video_sampler = SequentialSampler
    else:
        clip_sampler = RandomClipSampler(clip_duration=clip_duration)
        video_sampler = RandomSampler

    dataset = labeled_video_dataset(
        data_path=video_dir_path,
        clip_sampler=clip_sampler,
        video_sampler=video_sampler,
        transform=transform,
        decode_audio=False,
        decoder="pyav",
    )

    dataset.classes = sorted([d.name for d in Path(video_dir_path).iterdir()])
    dataset.n_classes = len(dataset.classes)

    loader = DataLoader(
        LimitDataset(dataset),
        batch_size=batch_size,
        drop_last=not is_val,
        num_workers=num_workers,
        collate_fn=collate_for_video,
    )

    return loader, dataset.n_classes


class LimitDataset(torch.utils.data.Dataset):
    """
    To ensure a constant number of samples are retrieved from the dataset we use this
    LimitDataset wrapper. This is necessary because several of the underlying videos
    may be corrupted while fetching or decoding, however, we always want the same
    number of steps per epoch.

    https://github.com/facebookresearch/pytorchvideo/blob/f7e7a88a9a04b70cb65a564acfc38538fe71ff7b/tutorials/video_classification_example/train.py#L341
    https://github.com/facebookresearch/pytorchvideo/issues/96
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(itertools.repeat(iter(dataset), 2))

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos
