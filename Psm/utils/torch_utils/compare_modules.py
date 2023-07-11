import torch
from torch import nn


def compare_modules(
    source_module: nn.Module,
    target_module: nn.Module,
    weight_only: bool,
    verbose: bool = False,
) -> bool:
    """Compare two modules and return whether they are the same.

    Args:
        source_module: The source PyTorch module.
        target_module: The target PyTorch module.
        weight_only: Whether to compare only the weights (and bias) of the
            modules.
        verbose: Boolean indicating whether to print the details of the
            comparison (e.g. the names of the modules that are different).

    Returns:
        A boolean indicating whether the modules are the same.

    """
    _differences, _total = [], []
    for __i, __j in zip(
        source_module.state_dict().items(), target_module.state_dict().items()
    ):
        __n1, __p1 = __i
        __n2, __p2 = __j

        if weight_only:
            if "weight" not in __n1 and "bias" not in __n1:
                continue

        _total.append(__n1)
        if not torch.equal(__p1, __p2):
            if __n1 == __n2:
                _differences.append(__n1)
            else:
                _error_msg = (
                    "Parameter ame mismatch found. Please check "
                    "if these two modules share the same architecture."
                )
                raise ValueError(_error_msg)

    if len(_differences) != 0 and verbose:
        print(
            f"Found the following {len(_differences)} out of "
            f"{len(_total)} differences:"
        )
        for __i in _differences:
            print(f"{__i}")
    return len(_differences) == 0
