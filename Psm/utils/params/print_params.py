from argparse import Namespace
from typing import Dict, Union


def print_params(params: Union[Namespace, Dict]):
    """Print the parameters with proper indentation.

    Args:
        params: Namespace or dictionary base parameters.

    """
    _params = params if isinstance(params, dict) else vars(params)
    _max_key_len = max([len(__k) for __k in _params.keys()])
    for __k, __v in _params.items():
        print(f"{__k:{_max_key_len + 4}s}: {__v}")
