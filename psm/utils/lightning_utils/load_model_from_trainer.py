from copy import deepcopy
from typing import Optional

import pytorch_lightning as pl
import torch


def load_best_model_from_trainer(
    trainer: pl.Trainer,
    monitor: Optional[str] = None,
    verbose: bool = False,
) -> pl.LightningModule:
    """Load the best model from a PyTorch Lightning Trainer.

    Args:
        trainer: A PyTorch Lightning Trainer with checkpoint callback after
            training for at least one epoch.
        monitor: The metric to monitor. If None, the default metric of the
            Trainer is used.
        verbose: Whether to print the path of the best model.

    Returns:
        A PyTorch Lightning Module loaded from the best model path.

    """
    _lightning_module = deepcopy(trainer.lightning_module)

    if monitor is None:
        _best_model_path = trainer.checkpoint_callback.best_model_path
    else:
        _best_model_path = None
        for __cb in trainer.checkpoint_callbacks:
            if __cb.monitor == monitor:
                _best_model_path = __cb.best_model_path
                break

    if _best_model_path is None:
        raise ValueError(
            f"Could not find a checkpoint callback with monitor {monitor}"
        )

    if verbose:
        print(f"Loading best model from {_best_model_path}")
    _best_model_state_dict = torch.load(_best_model_path)["state_dict"]
    _lightning_module.load_state_dict(_best_model_state_dict)
    return _lightning_module


def load_last_model_from_trainer(
    trainer: pl.Trainer,
    verbose: bool = False,
) -> pl.LightningModule:
    """Load the last model from a PyTorch Lightning Trainer.

    Args:
        trainer: A PyTorch Lightning Trainer with checkpoint callback after
            training for at least one epoch.
        verbose: Whether to print the path of the last model.

    Returns:
        A PyTorch Lightning Module loaded from the last model path.

    """
    _lightning_module = deepcopy(trainer.lightning_module)
    _last_model_path = None
    for __cb in trainer.checkpoint_callbacks:
        print(__cb.monitor)
        if __cb.monitor is None:
            _last_model_path = __cb.last_model_path
            break

    if _last_model_path is None:
        raise ValueError(
            "Could not find a checkpoint callback with valid last model path"
        )

    if verbose:
        print(f"Loading last model from {_last_model_path}")
    _last_model_state_dict = torch.load(_last_model_path)["state_dict"]
    _lightning_module.load_state_dict(_last_model_state_dict)
    return _lightning_module
