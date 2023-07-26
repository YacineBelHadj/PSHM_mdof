from typing import Any, Dict, Optional

from src.utils.torch_utils.get_lr_scheduler import get_lr_scheduler
from src.utils.torch_utils.get_optimizer import get_optimizer
from torch import nn


def configure_optimizer(
    model: nn.Module,
    model_name: str,
    optimizer_name: Optional[str],
    optimizer_kwargs: Optional[Dict[str, Any]],
    lr_scheduler_name: Optional[str],
    lr_scheduler_kwargs: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Configure optimizer and lr scheduler for a PyTorch model.

    This boilerplate function is meant to be used by LightningModule during
    `configure_optimizers` function calling.

    Args:
        model: A PyTorch model.
        model_name: String name of the model (parameters).
        optimizer_name: String name of the optimizer.
        optimizer_kwargs: Keyword arguments for the optimizer.
        lr_scheduler_name: String name of the lr scheduler.
        lr_scheduler_kwargs: Keyword arguments for the lr scheduler.

    Returns:
        A dictionary of optimizer configuration for pytorch lightning module
        if the learning rate is not 0, otherwise None.

    """
    if (
        optimizer_name is None
        or (optimizer_kwargs is not None and optimizer_kwargs["lr"]) == 0.0
    ):
        return None
    _optimizer = get_optimizer(
        optimizer_name=optimizer_name,
        optimizer_params=[
            {
                "params": model.parameters(),
                "name": model_name,
            }
        ],
        optimizer_kwargs=optimizer_kwargs,
    )
    if lr_scheduler_name is None:
        return {"optimizer": _optimizer}
    else:
        _lr_scheduler = get_lr_scheduler(
            lr_scheduler_name=lr_scheduler_name,
            lr_scheduler_optim=_optimizer,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        return {
            "optimizer": _optimizer,
            # Note that the scheduler parameters such as intervals, monitor,
            # etc. are not included in the returned configuration, because one
            # might not use automated optimization offered by LightningModule
            # by executing the optimizer/scheduler steps manually.
            "lr_scheduler": {"scheduler": _lr_scheduler},
        }
