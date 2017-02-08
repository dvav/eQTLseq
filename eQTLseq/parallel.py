"""Pool of workers for parallel processing."""

import multiprocessing as _mlp

POOL = None


def init(n_threads):
    """TODO."""
    global POOL
    if n_threads == 1:
        POOL = None
    elif n_threads is None or n_threads < 1:
        POOL = _mlp.Pool(processes=_mlp.cpu_count())
    else:
        POOL = _mlp.Pool(processes=n_threads)


def close():
    """TODO."""
    if POOL is not None:
        POOL.close()
        POOL.join()
