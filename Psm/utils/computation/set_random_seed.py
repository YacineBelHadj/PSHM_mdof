import random
import warnings

import numpy as np
import torch


# noinspection PyUnresolvedReferences
def set_random_seed(
    random_seed: int,
    deterministic: bool = False,
) -> None:
    """Seed the random generators of random, NumPy, PyTorch, etc.

    Note that this function will not only set the random state of
    all three packages, but also make the CuDNN strictly deterministic
    for reproducible results.

    References:
        - https://pytorch.org/docs/stable/notes/randomness.html#cudnn

    Args:
        random_seed: A seed/state for random generators.
        deterministic: Whether to make the CuDNN strictly deterministic.
            This could lead to performance degradation or slower convergence.

    Returns:
        None

    """
    try:
        # noinspection PyUnresolvedReferences
        from lightning_lite.utilities.seed import seed_everything

        seed_everything(random_seed)
    except ImportError:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    if torch.cuda.is_available() and deterministic:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except AttributeError:
            _warning_msg = (
                "Failed to configure CuDNN for deterministic "
                "computation and reproducible results."
            )
            warnings.warn(_warning_msg)
