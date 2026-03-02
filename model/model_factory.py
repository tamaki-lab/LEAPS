import os

from args import ArgLiteral

from .core.base_model import ClassificationBaseModel
from .core.model_config import BaseModelConfig
from .leaps import (
    LeapsVideoMAEClassifier,
    LeapsVivitClassifier,
    VideoMAEAttentionDistillation,
)


def set_torch_home(model_info: BaseModelConfig) -> None:
    """Specity the directory where a pre-trained model is stored.
    Otherwise, by default, models are stored in users home dir `~/.torch`
    """
    os.environ["TORCH_HOME"] = model_info.torch_home


def configure_model_class(
    model_name: ArgLiteral.SupportedModels,
) -> type[ClassificationBaseModel]:
    cls_dict = {
        # TODO : add LEAPS only model class here
        ArgLiteral.SupportedModels.LEAPS_VIDEOMAE: LeapsVideoMAEClassifier,
        ArgLiteral.SupportedModels.LEAPS_VIVIT: LeapsVivitClassifier,
        ArgLiteral.SupportedModels.Distill_LEAPS: VideoMAEAttentionDistillation,
    }
    assert model_name in cls_dict, f"Model {model_name} not found in cls_dict"
    model = cls_dict[model_name]

    return model


def configure_model(
    model_info: BaseModelConfig,
) -> ClassificationBaseModel:
    """model factory

    model_info:
        model_info (ModelInfo): information for model

    Raises:
        ValueError: invalide model name given by command line

    Returns:
        ClassificationBaseModel: model
    """
    set_torch_home(model_info)
    model_class = configure_model_class(model_info.model_name)
    return model_class(model_info)  # type: ignore
