import random
from functools import partial
from pathlib import Path

import webdataset as wds

from ..clip_sampler import (
    ClipSamplerArgs,
    configure_sampler,
)
from ..transforms import transform_video
from .info_from_json import info_from_json
from .util.shards_mapper import (
    StandardWdsPipeline,
)
from .util.video_decoder import (
    VideoDecoder,
)


def wds_train_video_folder(
    train_dir,
    clip_frames,
    target_fps,
    clips_per_video,
    batch_size,
    num_workers,
    gpus,
    transform_info,
):
    clip_sampler_args = ClipSamplerArgs(
        num_frames=clip_frames,
        target_fps=target_fps,
        num_views_for_sequencial=clips_per_video,
    )

    make_loader = partial(
        wds_video_dataloader,
        clip_sampler_args=clip_sampler_args,
        num_workers=num_workers,
    )

    train_clip_sampler = configure_sampler(is_train=True)

    train_loader = make_loader(
        train_dir,
        is_train=True,
        gpus=gpus,
        transform_info=transform_info,
        clip_sampler=train_clip_sampler,
        batch_size=batch_size,
    )

    _, n_classes, _ = info_from_json(train_dir)

    return train_loader, n_classes


def wds_val_video_folder(
    val_dir,
    clip_frames,
    target_fps,
    clips_per_video,
    batch_size,
    num_workers,
    gpus,
    transform_info,
    do_shuffle=False,
):
    clip_sampler_args = ClipSamplerArgs(
        num_frames=clip_frames,
        target_fps=target_fps,
        num_views_for_sequencial=clips_per_video,
    )

    make_loader = partial(
        wds_video_dataloader,
        clip_sampler_args=clip_sampler_args,
        num_workers=num_workers,
        do_shuffle=do_shuffle,
        transform_info=transform_info,
        is_train=False,
        gpus=gpus,
        batch_size=batch_size,
    )

    val_clip_sampler = configure_sampler(is_train=False)

    val_loader = make_loader(
        val_dir,
        clip_sampler=val_clip_sampler,
    )

    return val_loader


def wds_video_folder(
    train_dir,
    val_dir,
    clip_frames,
    target_fps,
    clips_per_video,
    batch_size,
    num_workers,
    gpus,
    transform_info,
    val_info_jsons=None,
):
    clip_sampler_args = ClipSamplerArgs(
        num_frames=clip_frames,
        target_fps=target_fps,
        num_views_for_sequencial=clips_per_video,
    )
    make_loader = partial(
        wds_video_dataloader,
        clip_sampler_args=clip_sampler_args,
        # batch_size=batch_size,
        num_workers=num_workers,
        transform_info=transform_info,
    )
    val_paths = val_dir.split(",")
    val_info_jsons = val_info_jsons.split(",") if val_info_jsons is not None else None

    train_clip_sampler = configure_sampler(is_train=True)
    val_clip_sampler = configure_sampler(is_train=False)

    train_loader = make_loader(
        train_dir,
        is_train=True,
        gpus=gpus,
        clip_sampler=train_clip_sampler,
        batch_size=batch_size,
    )
    if val_info_jsons is None:
        val_loaders = [
            make_loader(p, is_train=False, gpus=gpus, clip_sampler=val_clip_sampler, batch_size=1) for p in val_paths
        ]
    else:
        val_loaders = [
            make_loader(p, is_train=False, gpus=gpus, clip_sampler=val_clip_sampler, batch_size=1)
            for p, j in zip(val_paths, val_info_jsons, strict=False)
        ]

    _, n_classes, _ = info_from_json(train_dir)

    return train_loader, val_loaders, n_classes


def wds_video_dataloader(
    shards_path,
    is_train,
    clip_sampler,
    clip_sampler_args,
    batch_size,
    num_workers,
    gpus,
    transform_info,
    do_shuffle: bool = True,
):
    shards_path_list = [str(path) for path in Path(shards_path).glob("*.tar") if not path.is_dir()]

    dataset_size, _, _ = info_from_json(shards_path)

    dataset_name = shards_path.split("/")[-2]

    dataset, collate_fn = make_dataset(
        dataset_name=dataset_name,
        is_train=is_train,
        do_shuffle=do_shuffle,
        shards_url=shards_path_list,
        clip_sampler=clip_sampler,
        clip_sampler_args=clip_sampler_args,
        shuffle_buffer_size=100 if is_train else 1,
        is_ddp=gpus > 1,
        transform_info=transform_info,
    )
    dataset = dataset.batched(batch_size, partial=False)

    loader = wds.WebLoader(
        dataset,
        num_workers=num_workers,
        batch_size=None,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=2,
    )

    num_batches = dataset_size // (batch_size * gpus)
    loader.length = num_batches
    loader = loader.with_length(num_batches)
    loader = loader.repeat(nbatches=num_batches)
    loader = loader.slice(num_batches)  # pylint: disable=E1101

    return loader


def make_dataset(
    dataset_name,
    is_train,
    shards_url,
    clip_sampler,
    clip_sampler_args,
    shuffle_buffer_size,
    # caption_text,
    is_ddp,
    transform_info,
    do_shuffle: bool = True,
):
    nodesplitter = wds.split_by_node if is_ddp else wds.single_node_only
    if is_train:
        do_shuffle = True
    if do_shuffle:
        random.shuffle(shards_url)

    dataset = wds.WebDataset(shards_url, nodesplitter=nodesplitter, shardshuffle=24)

    wds_pipeline = StandardWdsPipeline(
        dataset_name=dataset_name,
    )
    train_t, val_t = transform_video(transform_info)
    transform = train_t if is_train else val_t
    video_decoder = VideoDecoder(
        clip_sampler=clip_sampler,
        clip_sampler_args=clip_sampler_args,
        transform=transform,
    )

    dataset = wds_pipeline(dataset, video_decoder)
    collate_fn = wds_pipeline.collate_fn

    return dataset, collate_fn
