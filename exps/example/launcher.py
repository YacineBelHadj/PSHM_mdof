from argparse import Namespace

import nni

from src.utils import (
    set_random_seed,
    merge_params,
    print_params,
)


_default_global_params = Namespace(
    random_seed=0,
)
_default_data_params = Namespace(
    dataset_name="CIFAR10",
)
_default_model_params = Namespace(
    model_name="ResNet18",
)
_default_optimization_params = Namespace(
    batch_size=32,
    learning_rate=1e-3,
    optimizer_name="Adam",
    optimizer_kwargs={
        "lr": 1e-3,
    },
    lr_scheduler_name="CosineAnnealing",
    lr_scheduler_kwargs={
        "T_max": 10,
        "eta_min": 1e-5,
    },
    early_stop_patience=10,
)
default_params = Namespace(
    **vars(_default_global_params),
    **vars(_default_data_params),
    **vars(_default_model_params),
    **vars(_default_optimization_params),
)


if __name__ == "__main__":

    params = merge_params(
        default_params,
        nni.get_next_parameter(),
    )
    print_params(params)
    set_random_seed(params.random_seed)

    # Prepare data
    # Construct model
    # Train and evaluate model with checkpointing
