from abc import abstractmethod
from typing import Any

import torch
from torch import nn

from .model_config import BaseModelConfig
from .model_output_dataclass import VideoRecogOutput, average_video_recog_outputs


class BaseModelInterface(nn.Module):
    @abstractmethod
    def forward_feature(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def multiview_forward(self, *args, **kwargs):
        raise NotImplementedError


class ClassificationBaseModel(BaseModelInterface):
    def __init__(self, model_config: BaseModelConfig):
        super().__init__()
        self.model_config = model_config

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
        verb_labels: torch.Tensor | None = None,
        noun_labels: torch.Tensor | None = None,
        **kwargs,
    ) -> VideoRecogOutput:
        raise NotImplementedError

    def multiview_forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None,
        verb_labels: torch.Tensor | None = None,
        noun_labels: torch.Tensor | None = None,
        **kwargs,
    ) -> VideoRecogOutput:
        """
        Args:
            pixel_values (torch.Tensor): shape (batch_size, num_views, ...)
        Returns:
            torch.Tensor: logits from the model
        """

        outputs = []
        for i in range(pixel_values.size(1)):
            output = self.forward(
                pixel_values=pixel_values[:, i],
                labels=labels,
                verb_labels=verb_labels,
                noun_labels=noun_labels,
            )
            outputs.append(output)
        return average_video_recog_outputs(outputs)


def get_device(model: ClassificationBaseModel | nn.DataParallel | nn.Module) -> torch.device | Any:
    """get acutual device
    taken from https://github.com/pytorch/pytorch/issues/7460

    Returns:
        torch.device: device on which the model is loaded
    """

    if isinstance(model, nn.DataParallel):
        return next(model.module.parameters()).device

    return next(model.parameters()).device
