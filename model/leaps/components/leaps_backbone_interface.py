from dataclasses import dataclass

import torch


@dataclass
class LeapsBackboneOutput:
    attention_score: torch.Tensor


class LeapsBackboneInterface:
    def extract_attention_scores(self, pixel_values: torch.Tensor) -> LeapsBackboneOutput:
        raise NotImplementedError("Subclasses should implement this method.")

    def __call__(self, *args, **kwds):
        return self.extract_attention_scores(*args, **kwds)
