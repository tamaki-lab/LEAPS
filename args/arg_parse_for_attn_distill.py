import argparse


def add_distillation_argument(parser: argparse.ArgumentParser):
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
        "-s_hdim",
        "--leaps_hidden_dim",
        type=int,
        default=128,
        dest="leaps_hidden_dim",
    )
    parser.add_argument(
        "-s_fdim",
        "--leaps_feedforward_dim",
        type=int,
        default=512,
        dest="leaps_feedforward_dim",
    )
    parser.add_argument(
        "-s_head",
        "--leaps_num_heads",
        type=int,
        default=4,
        dest="leaps_num_heads",
    )
    parser.add_argument(
        "-s_layer",
        "--leaps_num_layers",
        type=int,
        default=6,
        dest="leaps_num_layers",
    )
    parser.add_argument(
        "-t_model",
        "--teacher_model",
        type=str,
        default="MCG-NJU/videomae-large",
        dest="teacher_model",
    )
