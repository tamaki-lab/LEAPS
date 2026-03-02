from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from transformers import VideoMAEConfig, VivitConfig

from model.core.base_model import ClassificationBaseModel
from model.core.model_config import LEAPSModelConfig
from model.core.model_output_dataclass import (
    VideoRecogOutput,
    VideoRecogPatchSelOutput,
)
from model.core.utils import load_weight_from_pth

from .components.base_models.base_custom_model_interface import VideoRecogBaseModelInterface
from .components.base_models.videomae import CustomVideoMAEForVideoClassification, VideoMAEForVideoClassficationVerbNoun
from .components.base_models.vivit import CustomVivitForVideoClassification
from .leaps import (
    LEAPS,
    LeapsOutput,
    create_leaps_128_4h_6l_pretrained,
)
from .leaps_teacher import TeacherLEAPS

# ==============
# Core Model Class
# specific model classes are defined after this
# ==============


class LeapsVideoClassificationModel(ClassificationBaseModel):
    @dataclass
    class PatchSelectionResult:
        remain_embs: torch.Tensor
        remain_token_mask: torch.Tensor
        remain_indices: torch.Tensor
        entropy_st: torch.Tensor | None = None
        entropy_s: torch.Tensor | None = None
        locality_score: torch.Tensor | None = None

    def __init__(self, model_config: LEAPSModelConfig):
        super().__init__(model_config)
        self.model_config = model_config
        self.teacher = model_config.use_teacher

        if model_config.use_teacher:
            self.leaps: TeacherLEAPS | LEAPS = TeacherLEAPS(
                model_config.teacher_model_name,
                use_layer_idx=model_config.use_teacher_layer_idx,
                do_attn_calibration=model_config.teacher_attn_calibration,
            )
        else:
            self.leaps: TeacherLEAPS | LEAPS = create_leaps_128_4h_6l_pretrained(
                num_frames=model_config.num_frames,
                pth_path=model_config.leaps_pth,
            )
        self.leaps_r = model_config.remain_token_rate

        self.model: VideoRecogBaseModelInterface = self.initialize_model()

        self.patch_selector: Callable[
            [torch.Tensor, torch.Tensor], LeapsVideoClassificationModel.PatchSelectionResult
        ] = self.random_patch_selector if model_config.random_patch_selection else self.leaps_patch_selector

        self.cls_fct = nn.CrossEntropyLoss(label_smoothing=self.model_config.label_smoothing)

    def set_leaps_r(self, r: float):
        self.leaps_r = r
        self.model_config.remain_token_rate = r

    def initialize_model(self) -> VideoRecogBaseModelInterface:
        raise NotImplementedError

    def renormalize_input(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # unnormalize pixel_values and re-normalize with leaps mean and std
        if (
            self.model_config.base_model_color_mean != self.model_config.leaps_color_mean
            or self.model_config.base_model_color_std != self.model_config.leaps_color_std
        ):
            mean = torch.tensor(self.model_config.base_model_color_mean).view(1, 1, 3, 1, 1).to(pixel_values.device)
            std = torch.tensor(self.model_config.base_model_color_std).view(1, 1, 3, 1, 1).to(pixel_values.device)
            pixel_values = pixel_values * std + mean  # unnormalize
            leaps_mean = torch.tensor(self.model_config.leaps_color_mean).view(1, 1, 3, 1, 1).to(pixel_values.device)
            leaps_std = torch.tensor(self.model_config.leaps_color_std).view(1, 1, 3, 1, 1).to(pixel_values.device)
            pixel_values = (pixel_values - leaps_mean) / leaps_std  # re-normalize
        return pixel_values

    @torch.no_grad()
    def leaps_patch_selector(self, pixel_values: torch.Tensor, input_embs: torch.Tensor) -> PatchSelectionResult:
        assert hasattr(self.model, "tubelet_size")
        assert hasattr(self.model, "patch_size")

        leaps_output: LeapsOutput = self.leaps(
            pixel_values,
            remain_token_rate=self.leaps_r,
            locality_score_formulation=self.model_config.locality_score_formulation,
            tau_st=self.model_config.tau_st,
            tau_s=self.model_config.tau_s,
        )

        input_embs = torch.gather(
            input_embs,
            dim=1,
            index=leaps_output.leap_indices.unsqueeze(-1).expand(-1, -1, input_embs.shape[-1]),
        )

        return LeapsVideoClassificationModel.PatchSelectionResult(
            remain_embs=input_embs,
            remain_token_mask=leaps_output.leap_mask,
            remain_indices=leaps_output.leap_indices,
            entropy_st=leaps_output.entropy_st,
            entropy_s=leaps_output.entropy_space,
            locality_score=leaps_output.locality_score,
        )

    @torch.no_grad()
    def random_patch_selector(self, _: torch.Tensor, input_embs: torch.Tensor) -> PatchSelectionResult:
        select_indices = torch.randperm(input_embs.shape[1], device=input_embs.device)

        remain_token_mask = torch.zeros(input_embs.shape[1], dtype=torch.bool, device=input_embs.device)
        remain_token_num = int(input_embs.shape[1] * self.model_config.remain_token_rate)

        remain_indices = select_indices[:remain_token_num]
        remain_token_mask[remain_indices] = True

        input_embs = input_embs[:, select_indices, :]
        input_embs = input_embs[:, :remain_token_num, :]

        return LeapsVideoClassificationModel.PatchSelectionResult(
            remain_embs=input_embs,
            remain_token_mask=remain_token_mask,
            remain_indices=remain_indices,
        )

    def forward_feature(
        self,
        pixel_values: torch.Tensor,
    ) -> VideoRecogPatchSelOutput:
        assert isinstance(self.model, nn.Module)
        input_embs = self.model.forward_embeddings(pixel_values)

        pixel_values = self.renormalize_input(pixel_values)

        cls_token = None
        if self.model.has_cls_token:
            cls_token = input_embs[:, 0].unsqueeze(1)  # (B, 1, d)
            input_embs = input_embs[:, 1:]  # (B, N, d)

        sel_res = self.patch_selector(
            pixel_values,
            input_embs,
        )
        input_embs = sel_res.remain_embs

        if cls_token is not None:
            input_embs = torch.cat([cls_token, input_embs], dim=1)

        output = self.model(input_embs)
        assert isinstance(output, VideoRecogOutput)
        act_logits = output.logits

        if self.model_config.pred_verb_noun:
            verb_logits = output.verb_logits
            noun_logits = output.noun_logits
        else:
            verb_logits = None
            noun_logits = None

        return VideoRecogPatchSelOutput(
            logits=act_logits,
            remain_token_mask=sel_res.remain_token_mask,
            entropy_st=sel_res.entropy_st,
            entropy_s=sel_res.entropy_s,
            verb_logits=verb_logits,
            noun_logits=noun_logits,
            locality_score=sel_res.locality_score,
        )

    def forward(
        self, pixel_values: torch.Tensor, labels=None, verb_labels=None, noun_labels=None, **kwargs
    ) -> VideoRecogPatchSelOutput:
        output = self.forward_feature(pixel_values)

        loss = None
        if labels is not None:
            loss = 0.0
            loss = self.cls_fct(output.logits, labels).mean()
        verb_loss, noun_loss = None, None
        if self.model_config.pred_verb_noun:
            if verb_labels is not None:
                verb_loss = self.cls_fct(output.verb_logits, verb_labels).mean()
                loss = loss + verb_loss
            if noun_labels is not None:
                noun_loss = self.cls_fct(output.noun_logits, noun_labels).mean()
                loss = loss + noun_loss

        return VideoRecogPatchSelOutput(
            logits=output.logits,
            loss=loss,
            remain_token_mask=output.remain_token_mask,
            entropy_st=output.entropy_st,
            entropy_s=output.entropy_s,
            verb_logits=output.verb_logits,
            noun_logits=output.noun_logits,
            verb_loss=verb_loss,
            noun_loss=noun_loss,
            locality_score=output.locality_score,
        )


# ==============
# Specific Model Classes
# ==============


# LEAPS + VideoMAE
class LeapsVideoMAEClassifier(LeapsVideoClassificationModel):
    def initialize_model(self):
        config = VideoMAEConfig.from_pretrained(
            self.model_config.classification_model_name,  # type: ignore
            num_labels=self.model_config.n_classes,
            num_frames=self.model_config.num_frames,
            use_mean_pooling=True,
        )

        if self.model_config.classification_scratch:
            if not self.model_config.pred_verb_noun:
                model = CustomVideoMAEForVideoClassification(config=config)  # type: ignore
            else:
                model = VideoMAEForVideoClassficationVerbNoun(
                    config=config,  # type: ignore
                    verb_num_classes=self.model_config.verb_num_classes,
                    noun_num_classes=self.model_config.noun_num_classes,
                )
        else:
            if not self.model_config.pred_verb_noun:
                model = CustomVideoMAEForVideoClassification.from_pretrained(
                    self.model_config.classification_model_name,
                    config=config,
                    ignore_mismatched_sizes=True,
                )
            else:
                model = VideoMAEForVideoClassficationVerbNoun.from_pretrained(
                    self.model_config.classification_model_name,
                    config=config,
                    verb_num_classes=self.model_config.verb_num_classes,
                    noun_num_classes=self.model_config.noun_num_classes,
                    ignore_mismatched_sizes=True,
                )

            if self.model_config.classification_model_pth is not None:
                load_weight_from_pth(model, self.model_config.classification_model_pth)
        # model.train()
        assert isinstance(model, CustomVideoMAEForVideoClassification)
        return model


# LEAPS + VIVIT
class LeapsVivitClassifier(LeapsVideoClassificationModel):
    def initialize_model(self):
        pth = None

        config = VivitConfig.from_pretrained(
            self.model_config.classification_model_name,  # type: ignore
            num_labels=self.model_config.n_classes,
            num_frames=self.model_config.num_frames,
        )
        if self.model_config.classification_scratch:
            model = CustomVivitForVideoClassification(config=config)  # type: ignore
        else:
            model = CustomVivitForVideoClassification.from_pretrained(
                self.model_config.classification_model_name,
                config=config,
                ignore_mismatched_sizes=True,
            )
            if self.model_config.classification_model_pth is not None:
                load_weight_from_pth(model, self.model_config.classification_model_pth)
            if pth is not None:
                load_weight_from_pth(model.vivit, pth)  # type: ignore
        # model.train()
        assert isinstance(model, CustomVivitForVideoClassification)
        return model
