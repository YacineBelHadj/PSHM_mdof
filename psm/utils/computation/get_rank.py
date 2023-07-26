import os


def get_rank() -> int:
    """Get the rank of the current process.

    Return:
        The rank of the current process. 0 if the process is not distributed.

    """
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0
