from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Union

import torch
import torchvision.transforms.v2 as image_transform
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2._auto_augment import _AutoAugmentBase, get_size


@dataclass
class TransformVideoInfo:
    resize_size: int = 256
    crop_size: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    rand_augmentation_magnitude: float = 9.0
    rand_augmentation_magnitude_std: float = 0.5
    rand_augmentation_num_ops: int = 4
    multi_crop: bool = False
    crop_only: bool = False


_FillType = Union[int, tuple[int, ...], dict[type | str, int | tuple[int, ...]]]


class RandAugmentWithMagStd(_AutoAugmentBase):
    r"""Continuous RandAugment with magnitude noise.

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (float): Mean magnitude level (0 <= magnitude <= max_levels).
        max_levels (int): Maximum magnitude level (controls scale of transforms).
        magnitude_std (float): Standard deviation for magnitude sampling.
        interpolation (InterpolationMode | int): Interpolation mode.
        fill (_FillType): Fill value for area outside transformed image.
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: float = 0.0,
        max_levels: int = 10,
        magnitude_std: float = 0.5,
        interpolation: InterpolationMode | int = InterpolationMode.NEAREST,
        fill: _FillType | dict[type | str, _FillType] = None,
    ) -> None:
        super().__init__(interpolation=interpolation, fill=fill)
        self.num_ops = num_ops
        self.magnitude = torch.tensor(magnitude).float()
        self.max_levels = max_levels
        self.magnitude_std = torch.tensor(magnitude_std).float()
        self._augmentation_space = self._create_augmentation_space()

    def _create_augmentation_space(self) -> dict[str, tuple[Any, bool]]:
        # parameter_fn(mag_level, height, width) -> actual parameter
        return {
            "Identity": (lambda mag, h, w: None, False),
            "ShearX": (lambda mag, h, w: (mag / self.max_levels) * 0.3, True),
            "ShearY": (lambda mag, h, w: (mag / self.max_levels) * 0.3, True),
            "TranslateX": (lambda mag, h, w: (mag / self.max_levels) * (150.0 / 331.0 * w), True),
            "TranslateY": (lambda mag, h, w: (mag / self.max_levels) * (150.0 / 331.0 * h), True),
            "Rotate": (lambda mag, h, w: (mag / self.max_levels) * 30.0, True),
            "Brightness": (lambda mag, h, w: mag / self.max_levels * 0.9, False),
            "Color": (lambda mag, h, w: mag / self.max_levels * 0.9, False),
            "Contrast": (lambda mag, h, w: mag / self.max_levels * 0.9, False),
            "Sharpness": (lambda mag, h, w: mag / self.max_levels * 0.9, False),
            # "Posterize": (lambda mag, h, w: int(round(8 - (mag / self.max_levels) * 4)), False),
            "Solarize": (lambda mag, h, w: 1.0 - (mag / self.max_levels), False),
            "AutoContrast": (lambda mag, h, w: None, False),
            "Equalize": (lambda mag, h, w: None, False),
        }

    def forward(self, *inputs: Any) -> Any:
        flat_inputs_with_spec, image_or_video = self._flatten_and_extract_image_or_video(inputs)
        height, width = get_size(image_or_video)

        for _ in range(self.num_ops):
            transform_id, (param_fn, signed) = self._get_random_item(self._augmentation_space)
            # sample continuous magnitude with noise
            mag = torch.normal(mean=self.magnitude, std=self.magnitude_std)
            # clamp to valid range
            mag = mag.clamp(0.0, float(self.max_levels))
            if signed and torch.rand(()) <= 0.5:
                mag = -mag
            mag = mag.cpu().item()
            # get actual transform parameter
            magnitude = param_fn(mag, height, width)

            image_or_video = self._apply_image_or_video_transform(
                image_or_video,
                transform_id,
                magnitude,
                interpolation=self.interpolation,
                fill=self._fill,
            )

        return self._unflatten_and_insert_image_or_video(flat_inputs_with_spec, image_or_video)


def three_horizontal_crops_center_vertical(clip: torch.Tensor, crop_size: int | tuple[int, int]) -> list[torch.Tensor]:
    """
    clip: TCHW (float/uint8)
    returns: [3 x (TCHW)]  Left, Center, Right crops
    """
    if isinstance(crop_size, int):
        ch = cw = crop_size
    else:
        ch, cw = crop_size

    # TCHW
    assert clip.ndim == 4 and clip.shape[1] in (1, 3), "expect TCHW"
    T, C, H, W = clip.shape
    if H < ch or W < cw:
        raise ValueError(
            f"Input spatial size ({H},{W}) is smaller than crop_size ({ch},{cw}). Resizeを十分大きくしてください。"
        )

    top = (H - ch) // 2
    coords = [
        (top, 0),  # Left
        (top, (W - cw) // 2),  # Center
        (top, W - cw),  # Right
    ]

    return [clip[:, :, t : t + ch, l : l + cw] for t, l in coords]


def transform_video(trans_image_info: TransformVideoInfo) -> tuple[Callable, Callable]:
    """transform for images

    Args:
        trans_image_info (TransformImageInfo): information for image transform

    Returns:
        Tuple[torchvision.transforms]: train and val transforms
    """

    def train_transform(img) -> torch.Tensor:
        if trans_image_info.crop_only:
            transform = image_transform.Compose(
                [
                    image_transform.ToImage(),  # HWC ndarray --> CHW tensor (Image)
                    image_transform.ToDtype(torch.float32, scale=True),  # [0,255] --> [0,1]
                    image_transform.RandomResizedCrop(
                        trans_image_info.crop_size,
                        scale=(0.75, 1.0),
                        antialias=True,
                    ),
                ]
            )
        else:
            transform = image_transform.Compose(
                [
                    image_transform.ToImage(),  # HWC ndarray --> CHW tensor (Image)
                    image_transform.ToDtype(torch.uint8),  # Must be uint8 for RandAugment
                    image_transform.RandomResizedCrop(
                        trans_image_info.crop_size,
                        scale=(0.75, 1.0),
                        antialias=True,
                    ),
                    RandAugmentWithMagStd(
                        num_ops=trans_image_info.rand_augmentation_num_ops,
                        magnitude=trans_image_info.rand_augmentation_magnitude,
                        magnitude_std=trans_image_info.rand_augmentation_magnitude_std,
                        interpolation=InterpolationMode.BILINEAR,
                        fill=(128, 128, 128),  # RGB mean,
                    ),
                    image_transform.ToDtype(torch.float32, scale=True),  # [0,255] --> [0,1]
                    image_transform.RandomHorizontalFlip(),
                    image_transform.Normalize(trans_image_info.mean, trans_image_info.std),
                ]
            )

        return transform(img)

    def val_transform(img) -> torch.Tensor:
        val_ops = [
            image_transform.ToImage(),
            image_transform.ToDtype(torch.float32, scale=True),
            image_transform.Resize(trans_image_info.resize_size, antialias=True),
        ]
        if not trans_image_info.multi_crop:
            val_ops.append(image_transform.CenterCrop(trans_image_info.crop_size))
        val_ops.append(image_transform.Normalize(trans_image_info.mean, trans_image_info.std))

        transform = image_transform.Compose(val_ops)

        output = transform(img)
        if trans_image_info.multi_crop:
            output = three_horizontal_crops_center_vertical(output, trans_image_info.crop_size)
            output = torch.stack(output, dim=0)  # [3, T, C, H, W]
        return output

    return train_transform, val_transform
