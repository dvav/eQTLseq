"""Implements run()."""

import multiprocessing as _mlp
import sys as _sys

import numpy as _nmp

from eQTLseq import parallel as _prl

from eQTLseq.ModelBinomGibbs import ModelBinomGibbs as _ModelBinomGibbs
from eQTLseq.ModelNBinomGibbs import ModelNBinomGibbs as _ModelNBinomGibbs
from eQTLseq.ModelNormalGibbs import ModelNormalGibbs as _ModelNormalGibbs
from eQTLseq.ModelPoissonGibbs import ModelPoissonGibbs as _ModelPoissonGibbs


def run(Z, G, mdl='Normal', scale=True, n_iters=1000, burnin=0.5, beta_thr=1e-6, s2_lims=(1e-20, 1e3), n_threads=1,
        progress=True, **extra):
    """Run an estimation algorithm for a specified number of iterations."""
    Z = Z.T
    n_threads = _mlp.cpu_count() if n_threads is None else n_threads
    n_burnin = round(n_iters * burnin)
    assert mdl in ('Normal', 'Poisson', 'Binomial', 'NBinomial')

    n_samples1, n_genes = Z.shape
    n_samples2, n_markers = G.shape

    assert n_samples1 == n_samples2

    # arguments
    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0)
    GTG = G.T.dot(G)

    args = {
        'n_samples': n_samples1,
        'n_markers': n_markers,
        'n_genes': n_genes,
        'n_iters': n_iters,
        'n_burnin': n_burnin,
        'beta_thr': beta_thr,
        's2_lims': s2_lims,
        'scale': scale,
        'Z': Z,
        'G': G,
        'GTG': GTG,
        **extra
    }

    if mdl == 'Normal':
        args['Y'] = (Z - _nmp.mean(Z, 0)) / _nmp.std(Z, 0) if scale else Z - _nmp.mean(Z, 0)
        args['YTY'] = _nmp.sum(args['Y']**2, 0)
        args['GTY'] = G.T.dot(args['Y'])

    # prepare model
    Model = {
        'Poisson': _ModelPoissonGibbs,
        'Binomial': _ModelBinomGibbs,
        'NBinomial': _ModelNBinomGibbs,
        'Normal': _ModelNormalGibbs,
    }[mdl]
    mdl = Model(**args)

    # loop
    state = _nmp.empty(n_iters + 1)
    state.fill(_nmp.nan)
    state[0] = 0
    print('Starting...', file=_sys.stderr)
    _prl.Pool = None if n_threads <= 1 else _mlp.Pool(processes=n_threads)
    for itr in range(1, n_iters + 1):
        mdl.update(itr, **args)
        state[itr] = mdl.get_state(**args)
        if progress:
            print('\r' + 'Iteration {0} of {1}'.format(itr, n_iters), end='', file=_sys.stderr)

    if _prl.Pool is not None:
        _prl.Pool.close()
        _prl.Pool.join()

    print('\nDone!', file=_sys.stderr)

    #
    return {
        'state': state,
        **mdl.get_estimates(**args)
    }
