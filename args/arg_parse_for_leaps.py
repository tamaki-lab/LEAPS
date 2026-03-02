import argparse

from .literal import ArgParam, Choice


class LocalityScoreFormulation(Choice):
    HYBRID = "hybrid"  # default
    ST = "st"
    SPACE = "space"


locality_score_formulation = ArgParam(LocalityScoreFormulation.choices(), LocalityScoreFormulation.HYBRID.value)


def add_leaps_argument(parser: argparse.ArgumentParser):
    """
    hyperparams:
    --initial_action_sim_thres (-ini_ast)
    --min_action_sim_thres (-m_ast)
    --initial_bias_diff_coeff (-ini_bdc)
    --max_bias_diff_coeff (-m_bdc)
    --initial_act_bias_attn_sim_thres (-ini_abas)
    --min_act_bias_attn_sim_thres (-m_abas)

    loss:
    --action_token_sim_loss (-no_atsl)
    --bias_token_dist_loss (-no_btdl)
    --act_bias_attn_sim_loss (-no_abasl)
    """

    parser.add_argument(
        "-lst",
        "--locality_score_formulation",
        type=str,
        dest="locality_score_formulation",
        default=locality_score_formulation.default,
        choices=locality_score_formulation.choices,
        help="Static score type.",
    )

    parser.add_argument(
        "-tau_st",
        "--tau_st",
        type=float,
        dest="tau_st",
        default=1.0,
        help="Tau_st for locality score calculation.",
    )

    parser.add_argument(
        "-tau_s",
        "--tau_s",
        type=float,
        dest="tau_s",
        default=1e-3,
        help="Tau_s for locality score calculation.",
    )

    parser.add_argument(
        "-leaps_r",
        "--token_retention_rate",
        "--remain_token_rate",
        dest="remain_token_rate",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--use_teacher",
        dest="use_teacher",
        action="store_true",
        help="Whether to use teacher LEAPS. For inference.py, this means using the teacher model as LEAPS.",
    )
    parser.set_defaults(use_teacher=False)

    parser.add_argument(
        "-teacher_layer_idx",
        "--use_teacher_layer_idx",
        dest="use_teacher_layer_idx",
        type=int,
        default=None,
        help="Layer index to use for teacher attention calibration (if None, use last layer).",
    )

    parser.add_argument(
        "-teacher_name",
        "--teacher_model_name",
        dest="teacher_model_name",
        type=str,
        choices=[
            "MCG-NJU/videomae-base",
            "MCG-NJU/videomae-large",
        ],
        default="MCG-NJU/videomae-large",
        help="Oracle model name from HuggingFace.",
    )

    parser.add_argument(
        "-no_teacher_calib",
        "--no_teacher_attn_calibration",
        action="store_false",
        dest="teacher_attn_calibration",
        help="Whether to use attention calibration with teacher.",
    )
    parser.set_defaults(teacher_attn_calibration=True)

    parser.add_argument(
        "-cls_model",
        "-classif_model",
        "--classification_model_name",
        dest="classification_model_name",
        type=str,
        choices=[
            "MCG-NJU/videomae-base-finetuned-kinetics",
            "MCG-NJU/videomae-large-finetuned-kinetics",
            "MCG-NJU/videomae-huge-finetuned-kinetics",
            "google/vivit-b-16x2-kinetics400",  # avg: 0.5 0.5 0.5, std: 0.5 0.5 0.5, fs: 32
        ],
        default="MCG-NJU/videomae-base-finetuned-kinetics",
        help="Classification model name from HuggingFace.",
    )

    parser.add_argument(
        "-cls_scratch",
        "--cls_scratch",
        action="store_true",
        dest="classification_scratch",
        help="Whether to train the classification model from scratch.",
    )
    parser.set_defaults(classification_scratch=False)

    parser.add_argument(
        "-cls_pth",
        "--cls_model_pth",
        dest="classification_model_pth",
        type=str,
        default=None,
        help="Path to the classification model .pth file (if None, use the pretrained model",
    )

    parser.add_argument(
        "-leaps_pth",
        "--leaps_model_pth",
        dest="leaps_pth",
        type=str,
        default="pth/leaps/leaps_k400_128_4h_6l_v1L.pth",
        help="Path to the leaps model .pth file.",
    )

    parser.add_argument(
        "-leaps_avg_color",
        "--leaps_avg_color",
        dest="leaps_avg_color",
        type=float,
        nargs=3,
        default=[0.485, 0.456, 0.406],
        help="Mean for input normalization for leaps model.",
    )

    parser.add_argument(
        "-leaps_std_color",
        "--leaps_std_color",
        dest="leaps_std_color",
        type=float,
        nargs=3,
        default=[0.229, 0.224, 0.225],
        help="Std for input normalization for leaps model.",
    )

    parser.add_argument(
        "-random_patch_selection",
        "--random_patch_selection",
        action="store_true",
        dest="random_patch_selection",
        help="Whether to use random patch selection instead of LEAPS.",
    )
    parser.set_defaults(random_patch_selection=False)
