import warnings
from typing import Any, Dict, Iterable, Union

import torch
from src.utils.python_utils.get_object_from_module import get_class_from_module
from src.utils.python_utils.is_subclass import is_subclass
from torch import Tensor
from torch.optim.optimizer import Optimizer


def _is_optimizer_class(
    optimizer_class: type,
) -> bool:
    """Check if the given class is a PyTorch optimizer class."""
    return is_subclass(optimizer_class, Optimizer)


def get_optimizer(
    optimizer_name: str,
    optimizer_params: Iterable[Union[Tensor, Dict]],
    optimizer_kwargs: Dict[str, Any],
    loose_match: bool = False,
) -> Optimizer:
    """Get a PyTorch optimizer by name and then initialize it.

    Args:
        optimizer_name: A string as the name of target optimizer.
        optimizer_params: An iterable structure of parameters
            to be optimized or dictionaries defining parameter groups.
        optimizer_kwargs: A dictionary of keyword arguments for the
            returned optimizer.
        loose_match: An boolean indicator for loose matching of name.

    Returns:
        A PyTorch optimizer instance with a name equal or similar to
        the target name, initialized with the parameters to be
        optimized and keyword arguments.

    Raises:
        ValueError: No valid optimizer found in the `torch.optim`
            or `flash.core.optimizers` with the given name.

    """
    try:
        _optimizer_class: type = get_class_from_module(
            class_name=optimizer_name,
            module=torch.optim,
            loose_match=loose_match,
        )
    except ValueError:
        try:
            import flash

            _optimizer_class: type = get_class_from_module(
                class_name=optimizer_name,
                module=flash.core.optimizers,
                loose_match=loose_match,
            )
        except ImportError:
            _error_msg = (
                f"Cannot find an optimizer with a name equal "
                f"(or similar) to '{optimizer_name}' in "
                f"module 'torch.optim'."
            )
            raise ValueError(_error_msg)
        except ValueError:
            _error_msg = (
                f"Cannot find an optimizer with a name equal "
                f"(or similar) to '{optimizer_name}' in "
                f"module `torch.optim` or `flash.core.optimizers`."
            )
            raise ValueError(_error_msg)

    if not _is_optimizer_class(_optimizer_class):
        _warning_msg = (
            f"Class '{_optimizer_class}' found by the name "
            f"'{optimizer_name}' is not a subclass of "
            f"'torch.optim.optimizer.Optimizer'."
        )
        warnings.warn(_warning_msg)

    return _optimizer_class(
        params=optimizer_params,
        **optimizer_kwargs,
    )
