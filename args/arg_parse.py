import argparse

from .arg_parse_for_attn_distill import add_distillation_argument
from .arg_parse_for_leaps import add_leaps_argument
from .literal import ArgLiteral


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter):
    """show default values of argparse.
    see
    https://stackoverflow.com/questions/18462610/argumentparser-epilog-and-description-formatting-in-conjunction-with-argumentdef
    for details.
    """


class ArgParse:
    @staticmethod
    def get() -> argparse.Namespace:
        """generate argparse object

        Returns:
            args (argparse.Namespace): object of command line arguments
        """
        parser = argparse.ArgumentParser(
            description="simple image/video classification",
            formatter_class=CustomFormatter,
        )

        add_leaps_argument(parser)
        add_distillation_argument(parser)

        parser.add_argument(
            "--debug",
            dest="debug",
            action="store_true",
            help="debug mode (not default)",
        )

        parser.add_argument(
            "--val_only",
            dest="val_only",
            action="store_true",
            help="debug mode (not default)",
        )

        # dataset
        parser.add_argument(
            "-td",
            "--train_dir",
            type=str,
            default="train",
            help="train dataset directory.",
        )
        parser.add_argument(
            "-vd",
            "--val_dir",
            type=str,
            default="val",
            help="validation dataset directory. if multiple, split by ','.",
        )

        parser.add_argument(
            "--visualize_videos_src",
            type=str,
            default="visualize_videos/src",
            help="source directory for videos to visualize.",
        )

        parser.add_argument(
            "--visualize_videos_dest",
            type=str,
            default="visualize_videos/dest",
            help="destination directory for videos to visualize.",
        )

        # model
        parser.add_argument(
            "--torch_home",
            type=str,
            default="./pretrained_models",
            help="TORCH_HOME environment variable where pre-trained model weights are stored.",
        )
        parser.add_argument(
            "-m",
            "--model_name",
            type=str,
            default=ArgLiteral.model_name.default,
            choices=ArgLiteral.model_name.choices,
            help="name of the model",
        )

        # video
        parser.add_argument(
            "-fs",
            "--frames_per_clip",
            type=int,
            dest="frames_per_clip",
            default=16,
            help="frames per clip.",
        )
        parser.add_argument(
            "-fps",
            "--target_fps",
            type=float,
            dest="target_fps",
            default=7.5,
            help="temporal stride for sampling frames.",
        )
        parser.add_argument(
            "--clips_per_video",
            type=int,
            default=4,
            help="sampling clips per video for validation, when clip_sampler is 'random'.",
        )
        parser.add_argument(
            "-multi_crop",
            "--multi_crop",
            action="store_true",
            default=False,
            help="use multi-crop for validation (default: False).",
        )

        parser.add_argument(
            "-avg_color",
            "--avg_color",
            type=float,
            nargs=3,
            default=[0.485, 0.456, 0.406],
            dest="avg_color",
            help="mean RGB values for normalization. [0,1] range.",
        )
        parser.add_argument(
            "-std_color",
            "--std_color",
            type=float,
            nargs=3,
            default=[0.229, 0.224, 0.225],
            dest="std_color",
            help="standard deviation RGB values for normalization. For [0,1] range.",
        )
        parser.add_argument(
            "--rand_aug_mag",
            type=float,
            default=0.0,
            dest="rand_aug_mag",
            help="magnitude of RandAugment.",
        )
        parser.add_argument(
            "--rand_aug_mag_std",
            type=float,
            default=0.5,
            dest="rand_aug_mag_std",
            help="standard deviation of magnitude of RandAugment.",
        )
        parser.add_argument(
            "--rand_aug_num_ops",
            type=int,
            default=4,
            dest="rand_aug_num_ops",
            help="number of operations of RandAugment.",
        )

        # training
        parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch size.")
        parser.add_argument("-b_v", "--val_batch_size", type=int, default=1, help="val_batch size.")
        parser.add_argument(
            "-max_val",
            "--max_val_step",
            type=int,
            default=None,
            dest="max_val_step",
            help="max_val_step",
        )
        parser.add_argument("-w", "--num_workers", type=int, default=2, help="number of workers.")
        parser.add_argument(
            "-w_v",
            "-w_val",
            "--num_workers_val",
            type=int,
            default=12,
            dest="num_workers_val",
            help="number of workers for validation.",
        )
        parser.add_argument("-e", "--num_epochs", type=int, default=25, help="number of epochs.")

        parser.add_argument(
            "-li",
            "--log_interval_steps",
            type=int,
            default=1,
            help="logging interval in steps.",
        )

        # optimizer
        parser.add_argument(
            "-opt",
            "--optimizer_name",
            type=str,
            dest="optimizer_name",
            default=ArgLiteral.optimizer_name.default,
            choices=ArgLiteral.optimizer_name.choices,
            help="optimizer name.",
        )
        parser.add_argument(
            "--grad_accum",
            type=int,
            default=1,
            help="steps to accumlate gradients.",
        )
        parser.add_argument("-lr", type=float, default=5e-4, help="learning rate.")
        parser.add_argument(
            "-lrd",
            "--lr_decay_rate",
            dest="lr_decay_rate",
            type=float,
            default=0.75,
            help="learning rate decay rate per layer.",
        )
        parser.add_argument("--momentum", type=float, default=0.9, help="momentum of SGD.")
        parser.add_argument(
            "-wd",
            "--weight_decay",
            type=float,
            default=5e-4,
            dest="weight_decay",
            help="weight decay.",
        )
        parser.add_argument(
            "--use_scheduler",
            dest="use_scheduler",
            action="store_true",
            help="use scheduler (not default)",
        )
        parser.add_argument(
            "--no_scheduler",
            dest="use_scheduler",
            action="store_false",
            help="do not use scheduler (default)",
        )
        parser.set_defaults(use_scheduler=False)

        parser.add_argument(
            "-sch",
            "--scheduler_name",
            type=str,
            dest="scheduler_name",
            default=ArgLiteral.scheduler_name.default,
            choices=ArgLiteral.scheduler_name.choices,
            help="scheduler name.",
        )

        parser.add_argument(
            "--warmup_epoch",
            type=int,
            default=5,
            help="number of warmup epochs.",
        )

        # multi-GPU strategy
        parser.add_argument(
            "--use_dp",
            dest="use_dp",
            action="store_true",
            help="GPUs with data parallel (dp); not for lightning",
        )
        parser.set_defaults(use_dp=False)

        parser.add_argument(
            "--devices",
            "--gpus",
            type=int,
            default=1,
            help="the number of gpus.",
        )

        parser.add_argument(
            "-l_smooth",
            "--label_smoothing",
            dest="label_smoothing",
            type=float,
            default=0.05,
        )

        # log dirs
        parser.add_argument(
            "--comet_log_dir",
            type=str,
            default="./comet_logs/",
            help="dir to comet log files.",
        )
        parser.add_argument(
            "--tf_log_dir",
            type=str,
            default="./tf_logs/",
            help="dir to TensorBoard log files.",
        )
        parser.add_argument(
            "-exp_key",
            "-ck",
            "--comet_exp_key",
            type=str,
            dest="comet_exp_key",
            default=None,
            help="comet experiment key to resume from.",
        )

        # checkpoint files
        parser.add_argument(
            "--save_checkpoint_dir",
            type=str,
            default="./log",
            help="dir to save checkpoint files.",
        )
        parser.add_argument(
            "-ckpt",
            "-ckpt_path",
            "--checkpoint_to_resume",
            type=str,
            default=None,
            dest="checkpoint_to_resume",
            help="path to the checkpoint file to resume from.",
        )

        parser.add_argument(
            "-etag",
            "--experiment_tag",
            type=str,
            default=None,
            dest="experiment_tag",
            help="experiment tag for comet.ml.",
        )

        parser.add_argument(
            "-swap_pickle",
            "--swap_pickle_path",
            type=str,
            default="data/frames/actionswap/swap_pickles/actionswap_rand_1.pickle",
            dest="swap_pickle_path",
        )

        parser.add_argument(
            "-class2id",
            "--class2id_json_path",
            type=str,
            default=None,
            dest="class2id_json_path",
        )

        # disabling comet for debugging
        parser.add_argument(
            "--disable_comet",
            "--no_comet",
            dest="disable_comet",
            action="store_true",
            help="do not use comet.ml (default: use comet)",
        )
        parser.set_defaults(disable_comet=False)

        parser.add_argument(
            "--remap_label_id_for_internvideo2",
            dest="remap_label_id_for_internvideo2",
            action="store_true",
        )

        parser.add_argument(
            "--train_only_head",
            dest="train_only_head",
            action="store_true",
        )

        parser.add_argument(
            "--pred_verb_noun",
            dest="pred_verb_noun",
            action="store_true",
            help="predict verb and noun classes (not default), for datasets like Assembly101.",
        )

        parser.add_argument(
            "--noun_num_classes",
            dest="noun_num_classes",
            type=int,
            default=90,
        )

        parser.add_argument(
            "--verb_num_classes",
            dest="verb_num_classes",
            type=int,
            default=24,
        )

        parser.add_argument(
            "--videomae_mask_ratio", type=float, default=0.9, help="mask ratio for VideoMAE pretraining."
        )

        args = parser.parse_args()

        print(args)

        return args
