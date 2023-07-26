"""This module implements miscellaneous utility functions and classes."""
from .computation import get_all_devices, get_rank, set_random_seed
from .debug import Tee, debug_wrapper, tee_output
from .lightning_utils import (
    configure_optimizer,
    load_best_model_from_trainer,
    load_last_model_from_trainer,
)
from .params import merge_params, print_params
from .parse_args import parse_args
from .python_utils import (
    assign_attr,
    assign_init_args,
    get_class_from_module,
    get_closest_match,
    get_function_from_module,
    get_object_from_module,
    get_unique_new_file_path,
    is_subclass,
)
from .torch_utils import (
    ToTensor,
    compare_modules,
    get_activation,
    get_lr_scheduler,
    get_optimizer,
    is_training_with_grad,
)

__all__ = [
    "get_all_devices",
    "get_rank",
    "set_random_seed",
    "debug_wrapper",
    "Tee",
    "tee_output",
    "configure_optimizer",
    "load_best_model_from_trainer",
    "load_last_model_from_trainer",
    "merge_params",
    "print_params",
    "assign_attr",
    "assign_init_args",
    "get_closest_match",
    "get_object_from_module",
    "get_class_from_module",
    "get_function_from_module",
    "get_unique_new_file_path",
    "is_subclass",
    "compare_modules",
    "get_activation",
    "get_lr_scheduler",
    "get_optimizer",
    "is_training_with_grad",
    "ToTensor",
    # Project specific
    "parse_args",
]
