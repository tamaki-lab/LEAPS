from dataclasses import dataclass

from args import ArgLiteral
from args.arg_parse_for_leaps import LocalityScoreFormulation


def check_values(*args, min_val=None, max_val=None):
    for arg in args:
        if min_val is not None:
            assert arg >= min_val, f"Invalid value: {arg}, min_val: {min_val}"
        if max_val is not None:
            assert arg <= max_val, f"Invalid value: {arg}, max_val: {max_val}"


@dataclass
class BaseModelConfig:
    model_name: ArgLiteral.SupportedModels
    torch_home: str = "./"
    n_classes: int = 10
    num_frames: int = 16
    classification_model_name: str | None = None
    classification_model_pth: str | None = None
    classification_scratch: bool = False
    pred_verb_noun: bool = False
    verb_num_classes: int = 24
    noun_num_classes: int = 90
    train_only_head: bool = False


@dataclass
class AttnDistillConfig(BaseModelConfig):
    leaps_hidden_dim: int = 256
    leaps_feedforward_dim: int = 1024
    leaps_num_heads: int = 8
    leaps_num_layers: int = 12
    teacher_model_name: str = "MCG-NJU/videomae-base"
    teacher_attn_calibration: bool = False


@dataclass
class LEAPSModelConfig(BaseModelConfig):
    locality_score_formulation: LocalityScoreFormulation = LocalityScoreFormulation.HYBRID
    tau_st: float = 1.0
    tau_s: float = 1e-3
    label_smoothing: float = 0.0
    remain_token_rate: float = 0.5  # Used when token_num_strategy is FIX_RATE
    use_teacher: bool = False
    teacher_model_name: str = "MCG-NJU/videomae-base"
    use_teacher_layer_idx: int | None = None
    teacher_attn_calibration: bool = False
    leaps_pth: str = ""
    base_model_color_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    base_model_color_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    leaps_color_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    leaps_color_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    random_patch_selection: bool = False
    leaps_hidden_dim: int = 128
    leaps_feedforward_dim: int = 512
    leaps_num_heads: int = 4
    leaps_num_layers: int = 6


def config_factory(
    model_name: ArgLiteral.SupportedModels,
    n_classes,
    args,
) -> BaseModelConfig:
    base_config = BaseModelConfig(
        model_name=model_name,
        torch_home=args.torch_home,
        n_classes=n_classes,
        num_frames=args.frames_per_clip,
        classification_model_name=args.classification_model_name,
        classification_model_pth=args.classification_model_pth,
        classification_scratch=args.classification_scratch,
        pred_verb_noun=args.pred_verb_noun,
        verb_num_classes=args.verb_num_classes,
        noun_num_classes=args.noun_num_classes,
        train_only_head=args.train_only_head,
    )
    if model_name.is_attn_distill():
        return AttnDistillConfig(
            **base_config.__dict__,
            leaps_hidden_dim=args.leaps_hidden_dim,
            leaps_feedforward_dim=args.leaps_feedforward_dim,
            leaps_num_heads=args.leaps_num_heads,
            leaps_num_layers=args.leaps_num_layers,
            teacher_model_name=args.teacher_model_name,
            teacher_attn_calibration=args.teacher_attn_calibration,
        )
    if model_name.is_debias():
        return LEAPSModelConfig(
            **base_config.__dict__,
            locality_score_formulation=LocalityScoreFormulation(args.locality_score_formulation),
            tau_st=args.tau_st,
            tau_s=args.tau_s,
            label_smoothing=args.label_smoothing,
            remain_token_rate=args.remain_token_rate,
            use_teacher=args.use_teacher,
            teacher_model_name=args.teacher_model_name,
            use_teacher_layer_idx=args.use_teacher_layer_idx,
            teacher_attn_calibration=args.teacher_attn_calibration,
            leaps_pth=args.leaps_pth,
            base_model_color_mean=tuple(args.avg_color),
            base_model_color_std=tuple(args.std_color),
            leaps_color_mean=tuple(args.leaps_avg_color),
            leaps_color_std=tuple(args.leaps_std_color),
            random_patch_selection=args.random_patch_selection,
            leaps_hidden_dim=args.leaps_hidden_dim,
            leaps_feedforward_dim=args.leaps_feedforward_dim,
            leaps_num_heads=args.leaps_num_heads,
            leaps_num_layers=args.leaps_num_layers,
        )
    else:
        return base_config
