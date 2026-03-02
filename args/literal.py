from dataclasses import dataclass
from enum import Enum


class Choice(Enum):
    @classmethod
    def choices(cls):
        return tuple([choice.value for choice in cls])


@dataclass
class ArgParam:
    choices: tuple
    default: object


class ArgLiteral:
    #  Model names
    class SupportedModels(Choice):
        LEAPS_VIDEOMAE = "leaps_videomae"
        LEAPS_VIDEOMAE_Ensemble = "leaps_videomae_ensemble"
        LEAPS_VIVIT = "leaps_vivit"
        LEAPS_InternVideo2 = "leaps_internvideo2"
        Distill_LEAPS = "attn_distill"

        def is_debias(self) -> bool:
            return self.value.startswith("leaps")

        def is_attn_distill(self) -> bool:
            return self.value == "attn_distill"

        def is_internvideo2(self) -> bool:
            return self.value.endswith("internvideo2")

    model_name = ArgParam(SupportedModels.choices(), SupportedModels.LEAPS_VIDEOMAE.value)

    class DatasetType(Choice):
        ImageFolder = "ImageFolder"
        VideoFolder = "VideoFolder"
        WdsVideo = "WdsVideo"
        FramesVideo = "FramesVideo"

    dataset_type = ArgParam(DatasetType.choices(), DatasetType.WdsVideo.value)

    class OptimizerName(Choice):
        SGD = "SGD"
        ADAM = "Adam"
        ADAMW = "AdamW"

    optimizer_name = ArgParam(OptimizerName.choices(), OptimizerName.ADAMW.value)

    class SchedulerName(Choice):
        CONSTANT = "constant"
        COSINE_DECAY = "cosine_decay"

    scheduler_name = ArgParam(SchedulerName.choices(), SchedulerName.COSINE_DECAY.value)
