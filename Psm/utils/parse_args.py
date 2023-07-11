import os
import sys
import warnings
from argparse import ArgumentParser, Namespace

import torch
import yaml
from src.data.rotate_images_by_quarters import QUARTER_DEGREES
from src.paths import CKPT_DIR_PATH, LOG_DIR_PATH

from .computation.get_all_devices import get_all_devices

_NUM_DEBUG_EPOCHS = 5


def parse_args():
    """Command-line argument parser for training."""
    _parser = ArgumentParser(
        description="Continual Learning "
        "with Representation Learning and Out-of-Distribution Detection."
    )

    _global_args = _parser.add_argument_group(
        title="Global Arguments",
        description="Global arguments for the whole training process, "
        "shared by all the tasks/learners/processes.",
    )
    _global_args.add_argument(
        "--name",
        help="Name of the experiment, used for logging and checkpointing.",
        required=True,
        type=str,
    )
    _global_args.add_argument(
        "--random_seed",
        help="Random seed for reproducibility.",
        default=0,
        type=int,
    )
    _global_args.add_argument(
        "--debug",
        help="Whether to run in debug mode.",
        action="store_true",
    )
    _global_args.add_argument(
        "--verbose",
        help="Whether to print/log verbose messages.",
        action="store_true",
    )
    _global_args.add_argument(
        "--devices",
        help="Device to use for training.",
        default="cuda",
        type=str,
    )
    _global_args.add_argument(
        "--multi_gpu",
        help="Whether to use multiple (all available) GPUs.",
        action="store_true",
    )
    _global_args.add_argument(
        "--amp",
        help="Whether to use mixed precision training.",
        action="store_true",
    )
    _global_args.add_argument(
        "--log_dir",
        help="Directory to save the logs.",
        type=str,
    )
    _global_args.add_argument(
        "--ckpt_dir",
        help="Directory to save the model checkpoints.",
        type=str,
    )

    _data_args = _parser.add_argument_group(
        title="Data Arguments",
        description="Arguments for the data (scenario, datasets, etc.).",
    )
    _data_args.add_argument(
        "--cl_scenario_name",
        help="Continual learning scenario.",
        type=str,
        required=True,
    )
    _data_args.add_argument(
        "--cl_scenario_fixed_class_ordering",
        help="Whether to use a fixed ordering (0, 1, ...) for the "
        "classes streamed to the continual learning process. Otherwise, "
        "the classes are randomly shuffled, depending on the seed.",
        action="store_true",
    )
    _data_args.add_argument(
        "--cl_scenario_n_experiences",
        help="Number of experiences in the scenario.",
        type=int,
        required=True,
    )
    _data_args.add_argument(
        "--use_pretrained_img_size",
        help="Whether to use the image size of the pretrained backbone.",
        action="store_true",
    )

    _backbone_args = _parser.add_argument_group(
        title="Backbone Arguments",
        description="Arguments for the backbone of representation learning "
        "and downstream tasks (e.g. classification).",
    )
    _backbone_args.add_argument(
        "--backbone_name",
        help="Backbone for representation learning and downstream tasks.",
        required=True,
        type=str,
    )
    _backbone_args.add_argument(
        "--backbone_pretrained",
        help="Whether to use pretrained backbone.",
        action="store_true",
    )
    _backbone_args.add_argument(
        "--backbone_sharing",
        help="Whether to share the backbone between different tasks.",
        action="store_true",
    )

    _hat_args = _parser.add_argument_group(
        title="Hard Attention Mask Arguments",
        description="Arguments for the hard attention mask.",
    )
    _hat_args.add_argument(
        "--hat_max_scale",
        help="Maximum scale of the hard attention mask.",
        default=700.0,
        type=float,
    )
    _hat_args.add_argument(
        "--hat_scale_annealing_on",
        help="The annealing factor for the scale hard attention mask during "
        "training. No annealing if set to None.",
        choices=[None, "epoch", "step"],
        default=None,
        type=str,
    )
    _hat_args.add_argument(
        "--hat_reg_lambdas",
        help="Regularization lambdas for the hard attention mask.",
        nargs="+",
        default=[1.0],
        type=float,
    )

    _representation_learning_args = _parser.add_argument_group(
        title="Representation Learning Arguments",
        description="Arguments for the representation learning process.",
    )
    _representation_learning_args.add_argument(
        "--rep_learner_name",
        help="Representation learning name (method).",
        required=True,
        type=str,
    )
    _add_backbone_head_learner_args(
        _representation_learning_args,
        name="representation learner",
        prefix="rep",
    )

    _classification_learning_args = _parser.add_argument_group(
        title="Classification Learning Arguments",
        description="Arguments for the classification learning process.",
    )
    _classification_learning_args.add_argument(
        "--clf_learner_name",
        help="Classification learning name (method).",
        required=True,
        type=str,
    )
    _add_backbone_head_learner_args(
        _classification_learning_args,
        name="classification learner",
        prefix="clf",
    )
    _classification_learning_args.add_argument(
        "--clf_dropout_rate",
        help="Dropout rate for the classification head.",
        default=0.0,
        type=float,
    )

    _ood_detection_args = _parser.add_argument_group(
        title="Out-of-Distribution Detection Arguments",
        description="Arguments for the out-of-distribution detection process.",
    )
    _ood_detection_args.add_argument(
        "--ood_detector_name",
        help="Out-of-distribution detection method.",
        required=True,
        type=str,
    )

    _classification_testing_args = _parser.add_argument_group(
        title="Classification Testing Arguments",
        description="Arguments for the classification testing process.",
    )
    _classification_testing_args.add_argument(
        "--report_cil_acc",
        help="Whether to report CIL (or TIL) prediction results as metric.",
        action="store_true",
    )
    _classification_testing_args.add_argument(
        "--load_last_clf_ckpt",
        help="Whether to load the last models for testing. Loading the best "
        "models by default.",
        action="store_true",
    )
    _classification_learning_args.add_argument(
        "--tst_num_passes",
        help="Number of test passes. If set to 1 (default), the test will "
        "ber performed once without augmentation. Otherwise, the test "
        "will be performed multiple times with augmentations.",
        default=1,
        type=int,
    )

    _args = _parser.parse_args()
    _resolve_args(_args)
    return _args


def _add_backbone_head_learner_args(
    parser: ArgumentParser,
    name: str,
    prefix: str,
):
    assert not prefix.endswith("_")
    parser.add_argument(
        f"--{prefix}_backbone_optimizer_name",
        help=f"Optimizer for the {name} backbone.",
        type=str,
    )
    parser.add_argument(
        f"--{prefix}_backbone_optimizer_lr",
        help=f"Learning rate for the {name} backbone optimizer.",
        type=float,
    )
    parser.add_argument(
        f"--{prefix}_backbone_lr_scheduler_name",
        help=f"Learning rate scheduler for the {name} backbone.",
        type=str,
    )
    parser.add_argument(
        f"--{prefix}_head_optimizer_name",
        help=f"Optimizer for the {name} head.",
        type=str,
    )
    parser.add_argument(
        f"--{prefix}_head_optimizer_lr",
        help=f"Learning rate for the {name} head optimizer.",
        type=float,
    )
    parser.add_argument(
        f"--{prefix}_head_lr_scheduler_name",
        help=f"Learning rate scheduler for the {name} head.",
        type=str,
    )
    parser.add_argument(
        f"--{prefix}_gradient_clip_val",
        help=f"Gradient clipping value for the {name} learner.",
        default=None,
        type=float,
    )
    parser.add_argument(
        f"--{prefix}_max_epochs",
        help=f"Maximum number of epochs for {name} training.",
        type=int,
    )
    parser.add_argument(
        f"--{prefix}_batch_size",
        help=f"Batch size for {name} training.",
        type=int,
    )
    parser.add_argument(
        f"--{prefix}_early_stopping_patience",
        help=f"Early stopping patience for {name} training.",
        type=int,
    )
    parser.add_argument(
        f"--{prefix}_resume_from_ckpt",
        help=f"Whether to resume from checkpoint for {name} training.",
        action="store_true",
    )


def _resolve_args(args: Namespace):
    """Resolve the arguments."""
    _resolve_debug(args)
    _resolve_devices(args)
    _resolve_amp(args)
    _resolve_log_dir(args)
    _resolve_ckpt_dir(args)
    _resolve_data(args)
    _resolve_backbone(args)
    _resolve_rep_learner(args)
    _resolve_clf_learner(args)
    _resolve_ood_detector(args)
    os.makedirs(args.log_dir, exist_ok=True)
    _args_file_path = os.path.join(args.log_dir, "args.yaml")
    with open(_args_file_path, "w") as __f:
        yaml.dump(args, __f, default_flow_style=False)


def _resolve_debug(args: Namespace):
    if args.debug:
        args.rep_max_epochs = min(_NUM_DEBUG_EPOCHS, args.rep_max_epochs)
        args.clf_max_epochs = min(_NUM_DEBUG_EPOCHS, args.clf_max_epochs)
        # args.rep_early_stopping_patience = 1
        # args.clf_early_stopping_patience = 1
        args.name = f"{args.name}_debug"
        args.verbose = True


def _resolve_devices(args: Namespace):
    if args.devices == "cpu":
        args.devices = [torch.device("cpu")]
    elif args.devices in ["cuda", "gpu"]:
        __interactive_mode = bool(getattr(sys, "ps1", sys.flags.interactive))
        if args.multi_gpu and not __interactive_mode:
            args.devices = get_all_devices()
        else:
            # PyTorch Lightning parallelism does not work in interactive mode.
            args.devices = get_all_devices()[:1]
    else:
        raise ValueError(f"Unknown device: {args.devices}")


def _resolve_amp(args: Namespace):
    args.precision = 16 if args.amp else 32


def _resolve_log_dir(args: Namespace):
    if args.log_dir is None:
        args.log_dir = os.path.join(LOG_DIR_PATH, args.name)


def _resolve_ckpt_dir(args: Namespace):
    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join(CKPT_DIR_PATH, args.name)


def _resolve_data(args: Namespace):
    pass


def _resolve_backbone(args: Namespace):
    if args.backbone_pretrained:
        args.use_pretrained_img_size = True


def _resolve_rep_learner(args: Namespace):
    _resolve_backbone_head_optimization(args, prefix="rep")
    # Reduce batch size to prevent OOM due to rotation augmentation.
    if "Rotated" in args.rep_learner_name:
        args.rep_batch_size = args.rep_batch_size // len(QUARTER_DEGREES)


def _resolve_clf_learner(args: Namespace):
    _resolve_backbone_head_optimization(args, prefix="clf")
    # Reduce batch size to prevent OOM due to rotation augmentation.
    if "Rotated" in args.clf_learner_name:
        args.clf_batch_size = args.clf_batch_size // len(QUARTER_DEGREES)


def _resolve_ood_detector(args: Namespace):
    args.ood_detector_kwargs = {}


def _resolve_backbone_head_optimization(
    args: Namespace,
    prefix: str,
):
    assert prefix in ["rep", "clf"]
    for _prefix in [f"{prefix}_backbone", f"{prefix}_head"]:
        _optimizer_name = getattr(args, f"{_prefix}_optimizer_name")
        _optimizer_lr = getattr(args, f"{_prefix}_optimizer_lr")
        # Effective learning rate = base learning rate * number of GPUs.
        if _optimizer_lr is not None:
            _optimizer_lr *= len(args.devices)
        _lr_scheduler_name = getattr(args, f"{_prefix}_lr_scheduler_name")
        _optimizer_kwargs = {"lr": _optimizer_lr}
        _lr_scheduler_kwargs = {}
        _max_epochs = getattr(args, f"{prefix}_max_epochs")

        if _optimizer_name in ["SGD", "LARS"]:
            _optimizer_kwargs["momentum"] = 0.9
        elif _optimizer_name in ["Adam"]:
            pass
        elif _optimizer_name is None:
            pass
        else:
            _warning_msg = (
                f"Cannot resolve the kwargs for {prefix} optimizer"
                f" {_optimizer_name}. Please consider adding more kwargs "
                f"in argument resolution."
            )
            warnings.warn(_warning_msg)

        if _lr_scheduler_name in ["LinearWarmupCosineAnnealingLR"]:
            _lr_scheduler_kwargs["warmup_epochs"] = 1 if args.debug else 10
            _lr_scheduler_kwargs["max_epochs"] = _max_epochs
        elif _lr_scheduler_name in ["ReduceLROnPlateau"]:
            _lr_scheduler_kwargs["patience"] = _max_epochs // 10
            _lr_scheduler_kwargs["factor"] = 0.1
        elif _lr_scheduler_name in ["MultiStepLR"]:
            _lr_scheduler_kwargs["milestones"] = [
                int(_max_epochs * 0.60),
                int(_max_epochs * 0.75),
                int(_max_epochs * 0.90),
            ]
            _lr_scheduler_kwargs["gamma"] = 0.1
        elif _lr_scheduler_name is None:
            pass
        else:
            _warning_msg = (
                f"Cannot resolve the kwargs for {prefix} lr scheduler "
                f"{_lr_scheduler_name}. Please consider adding more kwargs "
                f"in argument resolution."
            )
            warnings.warn(_warning_msg)

        setattr(args, f"{_prefix}_optimizer_kwargs", _optimizer_kwargs)
        setattr(args, f"{_prefix}_lr_scheduler_kwargs", _lr_scheduler_kwargs)
        delattr(args, f"{_prefix}_optimizer_lr")
