from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

# =========================
# utils
# =========================


def stack_mean(tensors: Sequence[torch.Tensor | None]) -> torch.Tensor | None:
    """
    Mean over view dimension (dim=0) for a list of tensors.
    - If any element is None => return None (safe default for optional fields).
    """
    if not tensors or any(t is None for t in tensors):
        return None
    return torch.stack(list(tensors), dim=0).mean(dim=0)  # type: ignore


def mean_list(xs: Sequence[float | int | None]) -> float | None:
    """
    Mean for numeric scalars.
    - If any element is None or empty => return None.
    """
    if not xs or any(x is None for x in xs):
        return None
    return float(sum(xs)) / len(xs)  # type: ignore


@dataclass
class VideoRecogOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    pixel_values: torch.Tensor | None = None
    verb_logits: torch.Tensor | None = None
    noun_logits: torch.Tensor | None = None
    verb_loss: torch.Tensor | None = None
    noun_loss: torch.Tensor | None = None

    @classmethod
    def average(cls, outputs: Sequence[VideoRecogOutput]) -> VideoRecogOutput:
        """Average outputs of multiple views. Used for multi-view validation."""
        if not outputs:
            raise ValueError("outputs must be non-empty")

        # keep "view-invariant / debug" tensors as first element (original behavior)
        first = outputs[0]

        return cls(
            logits=stack_mean([o.logits for o in outputs]),  # type: ignore
            loss=stack_mean([o.loss for o in outputs]),
            pixel_values=first.pixel_values,
            verb_logits=stack_mean([o.verb_logits for o in outputs]),
            noun_logits=stack_mean([o.noun_logits for o in outputs]),
            verb_loss=stack_mean([o.verb_loss for o in outputs]),
            noun_loss=stack_mean([o.noun_loss for o in outputs]),
        )


@dataclass
class VideoRecogPatchSelOutput(VideoRecogOutput):
    remain_token_mask: torch.Tensor | None = None
    entropy_st: torch.Tensor | None = None
    entropy_s: torch.Tensor | None = None
    locality_score: torch.Tensor | None = None
    token_length: float | None = None

    @classmethod
    def average(cls, outputs: Sequence[VideoRecogPatchSelOutput]) -> VideoRecogPatchSelOutput:
        """Average outputs of multiple views. Used for multi-view validation."""
        if not outputs:
            raise ValueError("outputs must be non-empty")
        if not all(isinstance(o, VideoRecogPatchSelOutput) for o in outputs):
            raise TypeError("all outputs must be DebiasOutput")

        first = outputs[0]

        return cls(
            logits=stack_mean([o.logits for o in outputs]),  # type: ignore
            loss=stack_mean([o.loss for o in outputs]),
            pixel_values=first.pixel_values,
            verb_logits=stack_mean([o.verb_logits for o in outputs]),
            noun_logits=stack_mean([o.noun_logits for o in outputs]),
            verb_loss=stack_mean([o.verb_loss for o in outputs]),
            noun_loss=stack_mean([o.noun_loss for o in outputs]),
            remain_token_mask=first.remain_token_mask,
            entropy_st=first.entropy_st,
            entropy_s=first.entropy_s,
            locality_score=first.locality_score,
            token_length=mean_list([o.token_length for o in outputs]),
        )


def average_video_recog_outputs(
    outputs: Sequence[VideoRecogOutput | VideoRecogPatchSelOutput],
) -> VideoRecogOutput | VideoRecogPatchSelOutput:
    """Average outputs of multiple views. Used for multi-view validation."""
    if not outputs:
        raise ValueError("outputs must be non-empty")
    if all(isinstance(o, VideoRecogPatchSelOutput) for o in outputs):
        return VideoRecogPatchSelOutput.average(outputs)  # type: ignore
    return VideoRecogOutput.average(outputs)  # type: ignore
