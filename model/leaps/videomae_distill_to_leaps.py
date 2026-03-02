from dataclasses import dataclass

import torch
from torch import nn

from model.core.model_config import AttnDistillConfig

from .leaps import LEAPS, LeapsOutput
from .leaps_teacher import TeacherLEAPS


@dataclass
class AttnDistillOutput:
    acc: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    ent_st_mse: torch.Tensor | None = None
    ent_s_mse: torch.Tensor | None = None
    leap_mask: torch.Tensor | None = None


class VideoMAEAttentionDistillation(nn.Module):
    def __init__(self, model_config: AttnDistillConfig):
        super().__init__()
        self.model_config = model_config

        self.teacher: LEAPS | TeacherLEAPS = TeacherLEAPS(
            model_config.teacher_model_name,
            do_attn_calibration=model_config.teacher_attn_calibration,
        ).requires_grad_(False)

        self.leaps = LEAPS(
            hidden_dim=model_config.leaps_hidden_dim,
            num_frames=self.model_config.num_frames,
            feedforward_dim=model_config.leaps_feedforward_dim,
            num_heads=model_config.leaps_num_heads,
            num_layers=model_config.leaps_num_layers,
        )

        self.mse_fct = nn.MSELoss()

    def forward(self, pixel_values: torch.Tensor) -> AttnDistillOutput:
        output: LeapsOutput = self.leaps(pixel_values, remain_token_rate=0.3)
        pred_leap_mask = output.leap_mask
        pred_ent_st = output.entropy_st
        pred_ent_s = output.entropy_space

        with torch.no_grad():
            teacher_output: LeapsOutput = self.teacher(
                pixel_values=pixel_values,
                remain_token_rate=0.3,
            )
            teacher_ent_st = teacher_output.entropy_st
            teacher_ent_s = teacher_output.entropy_space
            teacher_leap_mask = teacher_output.leap_mask

        loss = 0.0
        weight = 1e8
        all_ent_mse = weight * self.mse_fct(pred_ent_st, teacher_ent_st).mean()
        sp_ent_mse = weight * self.mse_fct(pred_ent_s, teacher_ent_s).mean()

        loss += all_ent_mse + sp_ent_mse

        acc = (pred_leap_mask & teacher_leap_mask).float().sum(-1)
        acc = (acc / teacher_leap_mask.float().sum(-1)).mean()

        return AttnDistillOutput(
            acc=acc,
            loss=loss,
            ent_st_mse=all_ent_mse,
            ent_s_mse=sp_ent_mse,
            leap_mask=pred_leap_mask,
        )
