# This file is based on the implementation of
# transformers.models.clip.modeling_videomae from the Hugging Face Transformers library.
# Original source: https://github.com/huggingface/transformers


import torch
import torch.nn as nn
from transformers import VideoMAEConfig, VideoMAEForVideoClassification, VideoMAEModel
from transformers.modeling_outputs import BaseModelOutput

from model.core.model_output_dataclass import VideoRecogOutput

from ..base_custom_model_interface import VideoRecogBaseModelInterface


class CustomVideoMAE(VideoMAEModel):
    def __init__(self, config: VideoMAEConfig) -> None:
        super().__init__(config)

        self.config = config
        self.use_layer_idx = None

    def forward_embeddings(self, pixel_values: torch.FloatTensor):
        return self.embeddings(pixel_values, None)

    def forward(
        self,
        embeddings: torch.FloatTensor = None,
        pixel_values: torch.FloatTensor | None = None,
        output_attentions: bool = False,
    ):
        if pixel_values is None and embeddings is None or (pixel_values is not None and embeddings is not None):
            raise ValueError("Either pixel_values or embeddings should be provided.")

        if pixel_values is not None:
            embeddings = self.forward_embeddings(pixel_values)

        encoder_outputs = self.encoder(embeddings, output_attentions=output_attentions)

        sequence_output = encoder_outputs[0]
        if self.layernorm is not None:
            sequence_output = self.layernorm(sequence_output)

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CustomVideoMAEForVideoClassification(VideoRecogBaseModelInterface, VideoMAEForVideoClassification):
    has_cls_token: bool = False

    def __init__(self, config: VideoMAEConfig) -> None:
        super().__init__(config)
        self.videomae = CustomVideoMAE(config)
        self.tubelet_size = config.tubelet_size
        self.patch_size = config.patch_size

    def forward_embeddings(self, pixel_values: torch.Tensor, **kwargs):
        return self.videomae.forward_embeddings(pixel_values)

    def forward_with_selected_embeddings(
        self,
        embeddings: torch.Tensor,
        **kwargs,
    ):
        outputs = self.videomae(
            embeddings,
        )

        sequence_output = outputs[0]

        if self.fc_norm is not None:
            sequence_output = self.fc_norm(sequence_output.mean(1))
        else:
            sequence_output = sequence_output[:, 0]

        logits = self.classifier(sequence_output)

        return VideoRecogOutput(
            logits=logits,
        )


class VideoMAEForVideoClassficationVerbNoun(CustomVideoMAEForVideoClassification):
    def __init__(self, config: VideoMAEConfig, verb_num_classes: int, noun_num_classes: int) -> None:
        super().__init__(config)
        self.verb_classifier = nn.Linear(config.hidden_size, verb_num_classes)
        self.noun_classifier = nn.Linear(config.hidden_size, noun_num_classes)

    def forward_with_selected_embeddings(
        self,
        embeddings: torch.Tensor,
        **kwargs,
    ):
        outputs = self.videomae(
            embeddings,
        )

        sequence_output = outputs[0]

        if self.fc_norm is not None:
            sequence_output = self.fc_norm(sequence_output.mean(1))
        else:
            sequence_output = sequence_output[:, 0]

        logits = self.classifier(sequence_output)

        verb_logits = self.verb_classifier(sequence_output)
        noun_logits = self.noun_classifier(sequence_output)

        return VideoRecogOutput(
            logits=logits,
            verb_logits=verb_logits,
            noun_logits=noun_logits,
        )
