import torch
from transformers import VivitForVideoClassification, VivitModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
)
from transformers.models.vivit.modeling_vivit import VivitLayer, VivitSelfAttention

from model.core.model_output_dataclass import VideoRecogOutput

from ..base_custom_model_interface import VideoRecogBaseModelInterface


class CustomVivitSdpaSelfAttention(VivitSelfAttention):
    def forward(
        self,
        hidden_states,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        ####### Faster attention than original implementation ###
        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            self.dropout.p if self.training else 0.0,
            is_causal=False,
            scale=None,
        )
        #########################################################

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, None) if output_attentions else (context_layer,)

        return outputs  # type: ignore


class CustomVivitModel(VivitModel):
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        embedding_output: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool = False,
        return_dict: bool | None = None,
    ) -> tuple[torch.FloatTensor] | BaseModelOutputWithPooling:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        ############## MODIFIED HERE ##############
        if (
            pixel_values is None
            and embedding_output is None
            or (pixel_values is not None and embedding_output is not None)
        ):
            raise ValueError("Either pixel_values or embeddings should be provided.")

        if pixel_values is not None:
            embedding_output = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        ##########################################

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CustomVivitForVideoClassification(VideoRecogBaseModelInterface, VivitForVideoClassification):
    has_cls_token: bool = True

    def __init__(self, config) -> None:
        super().__init__(config)
        self.vivit.__class__ = CustomVivitModel
        for layer in self.vivit.encoder.layer:
            assert isinstance(layer, VivitLayer)
            layer.attention.attention.__class__ = CustomVivitSdpaSelfAttention
        tubelet_size = config.tubelet_size.copy()
        self.tubelet_size = tubelet_size[0]
        self.patch_size = tubelet_size[1]

    def forward_embeddings(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False, **kwargs):
        # this model has cls_token
        return self.vivit.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

    def forward_with_selected_embeddings(
        self,
        embeddings: torch.Tensor,
        head_mask: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool = False,
        return_dict: bool | None = None,
        **kwargs,
    ) -> VideoRecogOutput:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vivit(
            embedding_output=embeddings,  # MODIFIED HERE
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        return VideoRecogOutput(
            logits=logits,
        )
