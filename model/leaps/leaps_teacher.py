from enum import Enum

import torch
from torch import nn

from args.arg_parse_for_leaps import LocalityScoreFormulation

from .components.leaps_backbone_interface import LeapsBackboneInterface
from .components.teacher_backbone.videomae import VideoMAEForLeapsBackbone
from .leaps import LeapsOutput, calc_leap_mask_stats


class AttnCalibState(Enum):
    INITIAL = 0
    READY = 1


class TeacherLEAPS(nn.Module):
    def __init__(self, model_name: str, use_layer_idx: int | None = None, do_attn_calibration: bool = False):
        super().__init__()
        if model_name in [
            "MCG-NJU/videomae-base",
            "MCG-NJU/videomae-large",
            "MCG-NJU/videomae-base-finetuned-kinetics",
            "MCG-NJU/videomae-large-finetuned-kinetics",
        ]:
            self.model = VideoMAEForLeapsBackbone.from_pretrained(model_name, use_mean_pooling=False).requires_grad_(
                False
            )
            if use_layer_idx is not None:
                self.model.set_use_layer_idx(use_layer_idx)  # type: ignore
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        if hasattr(self.model, "config"):
            self.Sp: int = (self.model.config.image_size // self.model.config.patch_size) ** 2  # type: ignore
            self.Tp: int = self.model.config.num_frames // self.model.config.tubelet_size  # type: ignore
        else:
            self.Sp: int = (self.model.image_size // self.model.patch_size) ** 2  # type: ignore
            self.Tp: int = self.model.num_frames // self.model.tubelet_size  # type: ignore

        self.calib_state = AttnCalibState.INITIAL if do_attn_calibration else None
        self.calibration_src_attn_score = None

        assert isinstance(self.model, LeapsBackboneInterface), "Model must implement LeapsBackboneInterface"

    def forward(
        self,
        pixel_values: torch.Tensor,
        remain_token_rate: float = 0.3,
        tau_st: float = 1.0,
        tau_s: float = 1e-3,
        locality_score_formulation: LocalityScoreFormulation = LocalityScoreFormulation.HYBRID,
        **kwargs,
    ) -> LeapsOutput:
        """
        Args:
            pixel_values: (B, C, T, H, W)
            remain_token_rate: ratio of tokens to keep.
            calibration_src_attn_score: attention scores for calibration (optional).

        Returns:
            remain_token_mask: (B, N) bool
            remain_indices:    (B, K) long
            all_entropy:       (B, N) spatio-temporal entropy
            space_entropy:     (B, N) spatial entropy
            attn_scores:       (B, H, N, N) attention scores used for the final outputs
        """

        attn_scores = self.model(pixel_values=pixel_values).attention_score
        if attn_scores.dim() != 4:
            raise ValueError(f"attn_scores must be 4D (B,H,N,N), got shape {tuple(attn_scores.shape)}")

        if self.calib_state == AttnCalibState.INITIAL:
            self.calib_state = AttnCalibState.READY
            zeros = torch.zeros_like(pixel_values).float().type_as(pixel_values)[0].unsqueeze(0)
            zero_output: LeapsOutput = self.forward(pixel_values=zeros, remain_token_rate=0.5)
            self.calibration_src_attn_score = zero_output.attn_scores

        stats = calc_leap_mask_stats(
            attn_scores=attn_scores,
            locality_score_formulation=locality_score_formulation,
            tau_st=tau_st,
            tau_s=tau_s,
            remain_rate=remain_token_rate,
            src_Tp=self.Tp,
            src_Sp=self.Sp,
            dest_Tp=self.Tp,
            dest_Sp=self.Sp,
            attn_score_calib_src=self.calibration_src_attn_score,
        )

        return LeapsOutput(
            leap_mask=stats.leap_mask,
            leap_indices=stats.leap_indices,
            attn_scores=attn_scores,
            entropy_st=stats.ent_stats.ent_st,
            entropy_space=stats.ent_stats.ent_s,
        )
