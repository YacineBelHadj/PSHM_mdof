import torch
from torch import nn


def is_training_with_grad(module: nn.Module) -> bool:
    """Check if the module is in training mode and requires gradients.

    Args:
        module: The module to check.

    Returns:
        True if the module is in training mode and requires gradients.

    """
    return (
        module.training
        and any(p.requires_grad for p in module.parameters())
        and torch.is_grad_enabled()
    )
