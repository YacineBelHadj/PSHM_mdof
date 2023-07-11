from typing import Any, Dict

import torch
from src.utils.python_utils.get_object_from_module import get_class_from_module


def get_activation(
    activation_name: str,
    activation_kwargs: Dict[str, Any],
    loose_match: bool = False,
) -> Any:
    """Get a PyTorch activation module by name and then initialize it.

    Args:
        activation_name: A string as the name of target activation module.
        activation_kwargs: A dictionary of keyword arguments for the
            returned activation module.
        loose_match: An boolean indicator for loose matching of name.

    Returns:
        A PyTorch activation module instance with a name equal or
        similar to the target name, initialized with the keyword
        arguments.

    """
    try:
        _activation_class: type = get_class_from_module(
            class_name=activation_name,
            module=torch.nn.modules.activation,
            loose_match=loose_match,
        )
    except ValueError:
        _error_msg = (
            f"Cannot find an activation module with a name "
            f"equal (or similar) to '{activation_name}' in "
            f"module 'torch.nn.modules.activation'."
        )
        raise ValueError(_error_msg)

    return _activation_class(**activation_kwargs)
