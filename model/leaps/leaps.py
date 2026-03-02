from dataclasses import dataclass, field

import torch
from torch import nn

from args.arg_parse_for_leaps import LocalityScoreFormulation

from .components.leaps_backbone_interface import LeapsBackboneOutput
from .components.leaps_backbone_vit import LeapsViT, LeapsViTConfig


@dataclass
class LeapsOutput:
    leap_mask: torch.Tensor
    leap_indices: torch.Tensor
    attn_scores: torch.Tensor
    entropy_st: torch.Tensor
    entropy_space: torch.Tensor
    locality_score: torch.Tensor | None = None


def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
    # Shannon entropy
    entropy = -(probs * probs.log()).sum(dim=-1)
    return entropy


def attn_score_to_head_avg_probs(attn_score: torch.Tensor) -> torch.Tensor:
    """
    Convert attention scores to probabilities using softmax.
    Args:
        attn_score (torch.Tensor): Attention scores of shape (bs, num_heads, num_tokens, num_tokens).
    Returns:
        torch.Tensor: Averaged over heads attention probabilities of shape (bs, num_tokens, num_tokens).
    """
    attn_probs = torch.nn.functional.softmax(attn_score, dim=-1)
    return attn_probs.mean(dim=1)  # average over heads


def compute_spatial_entropy(attn_score: torch.Tensor, tp: int, sp: int) -> torch.Tensor:
    """
    attn_map: (bs, attn_head, N, N) where N = Tp * Sp
    output: (bs, N) spatial entropy per query
    """
    bs, h, N, _ = attn_score.shape
    assert N == tp * sp, "Patch size mismatch"

    spatial_attn_scores = torch.zeros(bs, h, N, sp, device=attn_score.device).to(attn_score.dtype)

    for t in range(tp):
        query_indices = torch.arange(t * sp, (t + 1) * sp)
        key_indices = query_indices
        spatial_attn_scores[:, :, query_indices, :] = attn_score[:, :, query_indices][:, :, :, key_indices]

    spatial_attn_probs = attn_score_to_head_avg_probs(spatial_attn_scores)
    entropy = compute_entropy(spatial_attn_probs)

    return entropy


def compute_spatio_temporal_entropy(attn_scores: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        attn_scores (torch.Tensor): shape (bs, num_heads, num_tokens, num_tokens)
        q (float, optional): tsallis parameter. Defaults to 1.0.
    Returns:
        torch.Tensor: _description_
    """
    attn_probs = attn_score_to_head_avg_probs(attn_scores)
    return compute_entropy(probs=attn_probs)


@dataclass
class EntropyStats:
    ent_st: torch.Tensor
    ent_s: torch.Tensor
    std_st: torch.Tensor = field(init=False)
    std_s: torch.Tensor = field(init=False)
    mean_st: torch.Tensor = field(init=False)
    mean_s: torch.Tensor = field(init=False)
    ent_st_standardized: torch.Tensor = field(init=False)
    ent_s_standardized: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.recalc_stats()

    def recalc_stats(self):
        self.std_s, self.mean_s = torch.std_mean(self.ent_s, dim=1, keepdim=True)
        self.std_st, self.mean_st = torch.std_mean(self.ent_st, dim=1, keepdim=True)
        self.ent_s_standardized = (self.ent_s - self.mean_s) / self.std_s
        self.ent_st_standardized = (self.ent_st - self.mean_st) / self.std_st


def compute_entropy_stats(attn_score, tp, sp):
    spatio_temporal_entropy = compute_spatio_temporal_entropy(attn_scores=attn_score)
    spatial_entropy = compute_spatial_entropy(
        attn_score=attn_score,
        tp=tp,
        sp=sp,
    )

    return EntropyStats(
        ent_s=spatial_entropy,
        ent_st=spatio_temporal_entropy,
    )


def compute_locality_score(
    attn_score,
    tau_st=1.0,
    tau_s=1e-3,
    temporal_num_patches=8,
    spatial_num_patches=196,
    attn_score_calib_src=None,
    locality_score_formulation: LocalityScoreFormulation = LocalityScoreFormulation.HYBRID,
):
    ent_stats = compute_entropy_stats(
        attn_score=attn_score,
        tp=temporal_num_patches,
        sp=spatial_num_patches,
    )

    if attn_score_calib_src is not None:
        calib_stats = compute_entropy_stats(
            attn_score=attn_score_calib_src,
            tp=temporal_num_patches,
            sp=spatial_num_patches,
        )

        calib_st_weight = torch.clamp(calib_stats.mean_st / calib_stats.ent_st, min=1.0)
        calib_s_weight = torch.clamp(calib_stats.mean_s / calib_stats.ent_s, min=1.0)

        ent_stats.ent_st = ent_stats.ent_st * (calib_st_weight ** (ent_stats.std_st / calib_stats.std_st))
        ent_stats.ent_s = ent_stats.ent_s * (calib_s_weight ** (ent_stats.std_s / calib_stats.std_s))

        ent_stats.recalc_stats()

    spatio_temporal_static_score = torch.sigmoid(ent_stats.ent_st_standardized / tau_st)
    spatial_score = torch.sigmoid(ent_stats.ent_s_standardized / tau_s)

    if locality_score_formulation == LocalityScoreFormulation.HYBRID:
        locality_score = 1 - (spatio_temporal_static_score + spatial_score) / 2
    elif locality_score_formulation == LocalityScoreFormulation.ST:
        raw_spatio_temporal_static_score = torch.sigmoid(ent_stats.ent_st_standardized)
        locality_score = 1 - raw_spatio_temporal_static_score
    elif locality_score_formulation == LocalityScoreFormulation.SPACE:
        raw_spatial_score = torch.sigmoid(ent_stats.ent_s_standardized)
        locality_score = 1 - raw_spatial_score
    else:
        raise ValueError(f"Unknown locality_score_formulation: {locality_score_formulation}")

    return locality_score, ent_stats


def generate_leap_mask(
    locality_score: torch.Tensor,
    src_Sp: int,
    src_Tp: int,
    dest_Sp: int,
    dest_Tp: int,
    remain_rate: float,
):
    """_summary_

    Args:
        locality_score (torch.Tensor): shape (batch_size, num_tokens)
        remain_rate (float, optional): _description_. Defaults to 0.3.
    """

    B, _ = locality_score.shape

    if src_Sp != dest_Sp or src_Tp != dest_Tp:
        src_edge_Sp = int(float(src_Sp) ** 0.5)
        dest_edge_Sp = int(float(dest_Sp) ** 0.5)
        assert dest_edge_Sp * dest_edge_Sp == dest_Sp, "dest_Sp is not a perfect square"

        locality_score = locality_score.reshape(B, src_Tp, src_Sp).reshape(B, src_Tp, src_edge_Sp, src_edge_Sp)
        locality_score = torch.nn.functional.interpolate(
            locality_score.unsqueeze(1),  # (B, 1, sTp, seS, seS), this is necessary for interpolate
            size=(
                dest_Tp,
                dest_edge_Sp,
                dest_edge_Sp,
            ),
            mode="trilinear",
            align_corners=False,
        ).squeeze(1)  # B, dTp, deS, deS

        locality_score = locality_score.reshape(B, dest_Tp, dest_Sp)
        locality_score = locality_score.reshape(B, dest_Tp * dest_Sp)

    num_remain_tokens: int = locality_score.shape[1]

    num_remain_tokens = int(num_remain_tokens * remain_rate)

    _, remain_indices = locality_score.topk(k=num_remain_tokens, dim=1)

    remain_token_mask = torch.zeros_like(locality_score, dtype=torch.bool)
    remain_token_mask.scatter_(
        dim=1,
        index=remain_indices,
        src=torch.ones_like(remain_indices).to(locality_score).bool(),
    )  # type: ignore
    return remain_token_mask, remain_indices


@dataclass
class LeapMaskStats:
    leap_mask: torch.Tensor
    leap_indices: torch.Tensor
    locality_score: torch.Tensor
    ent_stats: EntropyStats


def calc_leap_mask_stats(
    attn_scores,
    locality_score_formulation,
    tau_st,
    tau_s,
    remain_rate,
    src_Tp,
    src_Sp,
    dest_Tp,
    dest_Sp,
    attn_score_calib_src=None,
) -> LeapMaskStats:
    locality_score, ent_stats = compute_locality_score(
        attn_scores,
        locality_score_formulation=locality_score_formulation,
        tau_st=tau_st,
        tau_s=tau_s,
        temporal_num_patches=src_Tp,
        spatial_num_patches=src_Sp,
        attn_score_calib_src=attn_score_calib_src,
    )
    leap_mask, leap_indices = generate_leap_mask(
        locality_score,
        remain_rate=remain_rate,
        src_Tp=src_Tp,
        src_Sp=src_Sp,
        dest_Tp=dest_Tp,
        dest_Sp=dest_Sp,
    )
    return LeapMaskStats(
        leap_mask=leap_mask,
        leap_indices=leap_indices,
        locality_score=locality_score,
        ent_stats=ent_stats,
    )


class LEAPS(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_frames: int,
        feedforward_dim: int,
        num_heads: int,
        num_layers: int,
        image_size: int = 224,
        num_channels: int = 3,
        patch_size: int = 16,
        tubelet_size: int = 2,
    ):
        super().__init__()
        config = LeapsViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            intermediate_size=feedforward_dim,
            num_attention_heads=num_heads,
            tubelet_size=tubelet_size,
            num_frames=num_frames,
        )
        self.model = LeapsViT(config)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        remain_token_rate: float,
        tau_st: float = 1.0,
        tau_s: float = 1e-3,
        locality_score_formulation: LocalityScoreFormulation = LocalityScoreFormulation.HYBRID,
        dest_tubelet_size: int | None = None,
        dest_patch_size: int | None = None,
    ) -> LeapsOutput:
        _, T, _, _, _ = pixel_values.shape
        tubelet_size = self.model.config.tubelet_size
        src_Sp = (self.model.config.image_size // self.model.config.patch_size) ** 2
        if dest_patch_size is None:
            dest_Sp = src_Sp
        else:
            dest_Sp = (self.model.config.image_size // dest_patch_size) ** 2

        if T == self.model.config.num_frames:
            src_Tp = T // tubelet_size
            dest_Tp = T // dest_tubelet_size if dest_tubelet_size is not None else src_Tp
            backbone_output: LeapsBackboneOutput = self.model(pixel_values)
            attn_scores = backbone_output.attention_score

            leap_stats = calc_leap_mask_stats(
                attn_scores=attn_scores,
                locality_score_formulation=locality_score_formulation,
                tau_st=tau_st,
                tau_s=tau_s,
                remain_rate=remain_token_rate,
                src_Tp=src_Tp,
                src_Sp=src_Sp,
                dest_Tp=dest_Tp,
                dest_Sp=dest_Sp,
            )

            return LeapsOutput(
                leap_mask=leap_stats.leap_mask,
                leap_indices=leap_stats.leap_indices,
                attn_scores=attn_scores,
                entropy_st=leap_stats.ent_stats.ent_st,
                entropy_space=leap_stats.ent_stats.ent_s,
                locality_score=leap_stats.locality_score,
            )
        elif T % self.model.config.num_frames == 0:
            chunk_size = self.model.config.num_frames
            num_chunks = T // chunk_size
            all_attn_scores, all_locality_score = [], []
            all_entropy_st, all_entropy_space = [], []
            all_leap_mask, all_leap_indices = [], []

            spatial_seq_len = None
            for i in range(num_chunks):
                chunk_pixels = pixel_values[:, i * chunk_size : (i + 1) * chunk_size]
                src_Tp = chunk_pixels.shape[1] // tubelet_size
                dest_Tp = chunk_pixels.shape[1] // dest_tubelet_size if dest_tubelet_size is not None else src_Tp

                backbone_output: LeapsBackboneOutput = self.model(chunk_pixels)
                attn_scores = backbone_output.attention_score

                leap_stats = calc_leap_mask_stats(
                    attn_scores=attn_scores,
                    locality_score_formulation=locality_score_formulation,
                    tau_st=tau_st,
                    tau_s=tau_s,
                    remain_rate=remain_token_rate,
                    src_Tp=src_Tp,
                    src_Sp=src_Sp,
                    dest_Tp=dest_Tp,
                    dest_Sp=dest_Sp,
                )

                spatial_seq_len = leap_stats.locality_score.shape[1] if spatial_seq_len is None else spatial_seq_len
                leap_indices = leap_stats.leap_indices + i * spatial_seq_len

                all_attn_scores.append(attn_scores)
                all_locality_score.append(leap_stats.locality_score)
                all_entropy_st.append(leap_stats.ent_stats.ent_st)
                all_entropy_space.append(leap_stats.ent_stats.ent_s)
                all_leap_mask.append(leap_stats.leap_mask)
                all_leap_indices.append(leap_indices)

            leap_mask = torch.cat(all_leap_mask, dim=1)
            leap_indices = torch.cat(all_leap_indices, dim=1)

            return LeapsOutput(
                leap_mask=leap_mask,
                leap_indices=leap_indices,
                attn_scores=torch.cat(all_attn_scores, dim=2),
                entropy_st=torch.cat(all_entropy_st, dim=1),
                entropy_space=torch.cat(all_entropy_space, dim=1),
                locality_score=torch.cat(all_locality_score, dim=1),
            )
        else:
            raise NotImplementedError(f"Unsupported number of frames: {T}")


def create_leaps_128_4h_6l_pretrained(
    pth_path: str = "pth/leaps/leaps_k400_128_4h_6l_v1L.pth",
    num_frames: int = 16,
) -> LEAPS:
    if num_frames > 16:
        num_frames = 16

    leaps = LEAPS(
        hidden_dim=128,
        num_frames=num_frames,
        feedforward_dim=512,
        num_heads=4,
        num_layers=6,
    ).requires_grad_(False)
    leaps.load_state_dict(torch.load(pth_path, map_location="cpu"))
    return leaps
