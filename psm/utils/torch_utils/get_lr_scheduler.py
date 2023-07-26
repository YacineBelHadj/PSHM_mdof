import warnings
from typing import Any, Dict

import torch
from src.utils.python_utils.get_object_from_module import get_class_from_module
from src.utils.python_utils.is_subclass import is_subclass

# noinspection PyProtectedMember
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.optimizer import Optimizer


def _is_lr_scheduler_class(
    lr_scheduler_class: Any,
) -> bool:
    """Check if the given class is a PyTorch LR scheduler class."""
    return is_subclass(lr_scheduler_class, LRScheduler)


def get_lr_scheduler(
    lr_scheduler_name: str,
    lr_scheduler_optim: Optimizer,
    lr_scheduler_kwargs: Dict[str, Any],
    loose_match: bool = False,
) -> LRScheduler:
    """Get a PyTorch learning rate (lr) scheduler by name and then
    initialize it.

    Args:
        lr_scheduler_name: A string as the name of target lr scheduler.
        lr_scheduler_optim: The optimizer whose lr to be scheduled.
        lr_scheduler_kwargs: A dictionary of keyword arguments for the
            returned lr scheduler.
        loose_match: An boolean indicator for loose matching of name.

    Returns:
        A PyTorch lr scheduler instance with a name equal or similar to
        the target name, initialized with the optimizer whose lr to be
        scheduled and keyword arguments.

    Raises:
        ValueError: No valid optimizer found in `torch.optim.lr_scheduler`
            or `flash.core.optimizers` with the given name.

    """
    try:
        _lr_scheduler_class: type = get_class_from_module(
            class_name=lr_scheduler_name,
            module=torch.optim.lr_scheduler,
            loose_match=loose_match,
        )
    except ValueError:
        try:
            import flash

            _lr_scheduler_class: type = get_class_from_module(
                class_name=lr_scheduler_name,
                module=flash.core.optimizers,
                loose_match=loose_match,
            )
        except ImportError:
            _error_msg = (
                f"Cannot find an learning rate scheduler with a name "
                f"equal (or similar) to '{lr_scheduler_name}' in "
                f"module `torch.optim.lr_scheduler`."
            )
            raise ValueError(_error_msg)
        except ValueError:
            _error_msg = (
                f"Cannot find an learning rate scheduler with a name "
                f"equal (or similar) to '{lr_scheduler_name}' in "
                f"module `torch.optim.lr_scheduler` or "
                f"`flash.core.optimizers`."
            )
            raise ValueError(_error_msg)

    # ReduceLROnPlateau is NOT a subclass of LRScheduler and should be
    # permitted without raising a warning.
    if (
        not _is_lr_scheduler_class(_lr_scheduler_class)
        and _lr_scheduler_class is not ReduceLROnPlateau
    ):
        _warning_msg = (
            f"Class '{_lr_scheduler_class}' found by the name "
            f"'{lr_scheduler_name}' is not a subclass of "
            f"'torch.optim.lr_scheduler._LRScheduler'."
        )
        warnings.warn(_warning_msg)

    try:
        return _lr_scheduler_class(
            optimizer=lr_scheduler_optim,
            **lr_scheduler_kwargs,
        )
    except TypeError:
        return _lr_scheduler_class(
            **lr_scheduler_kwargs,
        )
