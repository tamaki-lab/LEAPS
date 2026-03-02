# This file is based on the implementation of
# transformers.models.clip.modeling_videomae from the Hugging Face Transformers library.
# Original source: https://github.com/huggingface/transformers

import math

import torch
import torch.nn as nn
from transformers import VideoMAEConfig, VideoMAEModel
from transformers.models.videomae.modeling_videomae import VideoMAESelfAttention

from ...leaps_backbone_interface import LeapsBackboneInterface, LeapsBackboneOutput


class MyVideoMAESelfAttention(VideoMAESelfAttention):
    def __init__(
        self,
        config: VideoMAEConfig,
    ) -> None:
        super().__init__(config)

        self.last_attention_scores_callback = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
        k_bias = torch.zeros_like(self.v_bias, requires_grad=False) if self.q_bias is not None else None
        keys = nn.functional.linear(hidden_states, self.key.weight, bias=k_bias)
        values = nn.functional.linear(hidden_states, self.value.weight, bias=self.v_bias)
        queries = nn.functional.linear(hidden_states, self.query.weight, bias=self.q_bias)

        # reshape for multi-head
        query_layer = self.transpose_for_scores(queries)
        key_layer = self.transpose_for_scores(keys)
        value_layer = self.transpose_for_scores(values)

        # attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if self.last_attention_scores_callback is not None:
            # callback to get the last attention scores
            self.last_attention_scores_callback(attention_scores)

        # probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # context
        context = torch.matmul(attention_probs, value_layer)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(hidden_states.size(0), hidden_states.size(1), self.all_head_size)

        outputs = (context, attention_probs) if output_attentions else (context,)
        return outputs


class VideoMAEForLeapsBackbone(LeapsBackboneInterface, VideoMAEModel):
    def __init__(self, config: VideoMAEConfig) -> None:
        super().__init__(config)

        self.encoder.layer[-1].attention.attention = MyVideoMAESelfAttention(config)

        self.config = config
        self.use_layer_idx = None

    def set_use_layer_idx(self, layer_idx: int):
        self.use_layer_idx = layer_idx
        self.encoder.layer[-1].attention.attention = VideoMAESelfAttention(self.config)
        self.encoder.layer[layer_idx].attention.attention = MyVideoMAESelfAttention(self.config)

    def forward_embeddings(self, pixel_values: torch.FloatTensor):
        return self.embeddings(pixel_values, None)

    def extract_attention_scores(
        self,
        pixel_values: torch.Tensor | None = None,
        output_attentions: bool = False,
    ):
        embeddings = self.forward_embeddings(pixel_values)

        last_attn_scores = None

        def last_attn_scores_callback(attention_scores: torch.Tensor):
            nonlocal last_attn_scores
            last_attn_scores = attention_scores

        if self.use_layer_idx is None:
            self.encoder.layer[-1].attention.attention.last_attention_scores_callback = last_attn_scores_callback
        else:
            self.encoder.layer[
                self.use_layer_idx
            ].attention.attention.last_attention_scores_callback = last_attn_scores_callback

        encoder_outputs = self.encoder(embeddings, output_attentions=output_attentions)

        sequence_output = encoder_outputs[0]
        if self.layernorm is not None:
            sequence_output = self.layernorm(sequence_output)

        assert last_attn_scores is not None
        return LeapsBackboneOutput(
            attention_score=last_attn_scores,
        )
