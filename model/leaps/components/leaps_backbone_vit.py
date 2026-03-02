# This file is based on the implementation of
# transformers.models.clip.modeling_videomae from the Hugging Face Transformers library.
# Original source: https://github.com/huggingface/transformers

import collections.abc
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.videomae.modeling_videomae import get_sinusoid_encoding_table
from transformers.models.vit.modeling_vit import ViTLayer, ViTSelfAttention

from .leaps_backbone_interface import LeapsBackboneInterface, LeapsBackboneOutput


class LeapsViTConfig(ViTConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tubelet_size: int = kwargs.get("tubelet_size", 2)  # Default tubelet size for video input
        self.num_frames: int = kwargs.get("num_frames", 16)  # Default number of frames for video input


@dataclass
class LeapsViTOutput(BaseModelOutput):
    last_attention_scores: torch.FloatTensor | None = None


class LeapsViTPatchEmbeddingsForVideoInput(nn.Module):
    def __init__(self, config):
        super().__init__()

        image_size = config.image_size
        patch_size = config.patch_size
        num_channels = config.num_channels
        hidden_size = config.hidden_size
        num_frames = config.num_frames
        tubelet_size = config.tubelet_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.tubelet_size = int(tubelet_size)
        num_patches = (
            (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)  # type: ignore
        )
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.projection = nn.Conv3d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),  # type: ignore
            stride=(self.tubelet_size, patch_size[0], patch_size[1]),  # type: ignore
        )

    def forward(self, pixel_values, **kwargs):
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height != self.image_size[0] or width != self.image_size[1]:  # type: ignore
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."  # type: ignore
            )
        # permute to (batch_size, num_channels, num_frames, height, width)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class LeapsViTEmbeddings(nn.Module):
    def __init__(self, config: LeapsViTConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = LeapsViTPatchEmbeddingsForVideoInput(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = get_sinusoid_encoding_table(num_patches, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, T, C, H, W = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        embeddings = embeddings + self.position_embeddings.type_as(embeddings).to(embeddings.device).clone().detach()

        embeddings = self.dropout(embeddings)

        return embeddings


class LeapsViTLastSelfAttention(ViTSelfAttention):
    def __init__(
        self,
        config: LeapsViTConfig,
    ) -> None:
        super().__init__(config)

        self.last_attention_scores_callback = None

    def forward(
        self, hidden_states: torch.Tensor, head_mask: torch.Tensor | None = None, output_attentions: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if self.last_attention_scores_callback is not None:
            # callback to get the last attention scores
            self.last_attention_scores_callback(attention_scores)
        else:
            assert False, "last_attention_scores_callback is not set."

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class LeapsViTLastAttention(nn.Module):
    def __init__(self, config: LeapsViTConfig) -> None:
        super().__init__()
        self.attention = LeapsViTLastSelfAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
        attention_output = self_outputs[0]

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class LeapsViTLastLayer(ViTLayer):
    def __init__(self, config: LeapsViTConfig):
        super().__init__(config)
        self.attention = LeapsViTLastAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        outputs = (attention_output,) + outputs

        return outputs


class LeapsViT(LeapsBackboneInterface, ViTModel):
    def __init__(self, config: LeapsViTConfig) -> None:
        super().__init__(config)

        self.embeddings = LeapsViTEmbeddings(config)
        self.encoder.layer[-1] = LeapsViTLastLayer(config)

    def forward_embeddings(self, pixel_values: torch.Tensor):
        return self.embeddings(pixel_values, None)

    def extract_attention_scores(
        self,
        pixel_values: torch.Tensor,
    ):
        embeddings = self.forward_embeddings(pixel_values)

        last_attn_scores = None

        def last_attn_scores_callback(attention_scores: torch.Tensor):
            nonlocal last_attn_scores
            last_attn_scores = attention_scores

        self.encoder.layer[-1].attention.attention.last_attention_scores_callback = last_attn_scores_callback  # type: ignore[method-assign]
        _ = self.encoder(embeddings)

        assert last_attn_scores is not None
        return LeapsBackboneOutput(last_attn_scores)
