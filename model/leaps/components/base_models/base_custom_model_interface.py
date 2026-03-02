import torch

from model.core.model_output_dataclass import VideoRecogOutput


class VideoRecogBaseModelInterface:
    has_cls_token: bool = False

    def forward_embeddings(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")

    def forward_with_selected_embeddings(self, embeddings: torch.Tensor, **kwargs) -> VideoRecogOutput:
        raise NotImplementedError("Subclasses should implement this method.")

    def __call__(self, *args, **kwargs):
        return self.forward_with_selected_embeddings(*args, **kwargs)
