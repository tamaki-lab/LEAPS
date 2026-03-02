import argparse
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from dataloader.transforms import TransformVideoInfo
from dataloader.wds_folder import (
    wds_train_video_folder,
    wds_val_video_folder,
)


@dataclass
class DataloadersInfo:
    """DataloadersInfo

    train_loader (torch.utils.data.DataLoader): training set loader
    val_loader (torch.utils.data.DataLoader): validation set loader
    n_classes (int): number of classes
    """

    train_loader: DataLoader
    val_loader: list[DataLoader]
    n_classes: int


def configure_dataloader(
    command_line_args: argparse.Namespace,
):
    """dataloader factory

    Args:
        command_line_args (argparse.Namespace): command line args
        dataset_name (SupportedDatasets): dataset name (str).
            ["VideoFolder", "ZeroImages"]

    Raises:
        ValueError: invalid dataset_name is given

    Returns:
        (DataloadersInfo): dataset information
    """

    args = command_line_args

    transform_info = TransformVideoInfo(
        resize_size=256,
        crop_size=224,
        mean=tuple(args.avg_color),
        std=tuple(args.std_color),
        rand_augmentation_magnitude=args.rand_aug_mag,
        rand_augmentation_magnitude_std=args.rand_aug_mag_std,
        rand_augmentation_num_ops=args.rand_aug_num_ops,
        multi_crop=args.multi_crop,
    )

    # train_loader n_classes
    shards_path_list = [str(path) for path in Path(args.train_dir).glob("*.tar") if not path.is_dir()]
    num_devices = args.devices
    num_workers = min(args.num_workers, len(shards_path_list) // num_devices)

    train_loader, n_classes = wds_train_video_folder(
        train_dir=args.train_dir,
        clip_frames=args.frames_per_clip,
        target_fps=args.target_fps,
        clips_per_video=args.clips_per_video,
        batch_size=args.batch_size,
        num_workers=num_workers,
        gpus=args.devices,
        transform_info=transform_info,
    )

    # val_loader
    val_loaders = []
    val_batch_size = args.val_batch_size

    for i, val_dir in enumerate(args.val_dir.split(",")):
        shards_path_list = [str(path) for path in Path(val_dir).glob("*.tar") if not path.is_dir()]
        num_devices = args.devices
        num_workers = min(args.num_workers_val, len(shards_path_list) // num_devices)
        #
        val_loader = wds_val_video_folder(
            val_dir=val_dir,
            clip_frames=args.frames_per_clip,
            target_fps=args.target_fps,
            clips_per_video=args.clips_per_video,
            batch_size=val_batch_size,
            num_workers=num_workers,
            gpus=args.devices,
            transform_info=transform_info,
        )
        val_loaders.append(val_loader)

    return DataloadersInfo(train_loader=train_loader, val_loader=val_loaders, n_classes=n_classes)
