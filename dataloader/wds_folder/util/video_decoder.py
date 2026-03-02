import torch
from torch import as_tensor
from torchvision.io import decode_jpeg


class VideoDecoder:
    def __init__(self, clip_sampler, clip_sampler_args, transform) -> None:
        self.clip_sampler = clip_sampler
        self.clip_sampler_args = clip_sampler_args
        self.transform = transform
        self.frame_indices = None

    def decode(self, jpg_byte_list, frame_indices):
        clip = [decode_jpeg(as_tensor(list(jpg_byte_list[i]), dtype=torch.uint8)) for i in frame_indices]
        clip = torch.stack(clip, 0)  # TCHW
        if self.transform is not None:
            clip = self.transform(clip)
        return clip

    def video_decoder(
        self,
        video_pickle,
        source_fps=None,
    ):
        if len(video_pickle) == 2:
            jpg_byte_list, frame_sec_list = video_pickle
        else:
            jpg_byte_list = video_pickle
            frame_sec_list = len(jpg_byte_list)

        if source_fps is not None:
            frame_sec_list = [frame_idx / source_fps for frame_idx in range(len(jpg_byte_list))]

        frame_indices = self.clip_sampler(frame_sec_list, self.clip_sampler_args)

        if isinstance(frame_indices, list):
            # Multi-view
            multi_views = []
            for indices in frame_indices:
                clip = self.decode(jpg_byte_list, indices)
                if clip.dim() == 5:
                    multi_views += list(clip.unbind(dim=0))
                else:
                    multi_views.append(clip)
            clips = torch.stack(multi_views, 0)
            return clips
        else:
            # Single view
            clip = self.decode(jpg_byte_list, frame_indices)
            return clip
