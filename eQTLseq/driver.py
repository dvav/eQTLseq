"""Implements run()."""

import sys as _sys

import numpy as _nmp
import multiprocessing as _mlp

from eQTLseq.ModelBinomGibbs import ModelBinomGibbs as _ModelBinomGibbs
from eQTLseq.ModelNBinomGibbs import ModelNBinomGibbs as _ModelNBinomGibbs
from eQTLseq.ModelNormalGibbs import ModelNormalGibbs as _ModelNormalGibbs
from eQTLseq.ModelPoissonGibbs import ModelPoissonGibbs as _ModelPoissonGibbs

import eQTLseq.utils as _utils


def run(Z, G, mdl='Normal', trans=None, std=True, norm_factors=None, n_iters=1000, n_burnin=None, beta_thr=1e-6,
        s2_lims=(1e-20, 1e3), n_threads=_mlp.cpu_count(), **extra):
    """Run an estimation algorithm for a specified number of iterations."""
    n_burnin = round(n_iters * 0.5) if n_burnin is None else n_burnin
    assert mdl in ('Normal', 'Poisson', 'Binomial', 'NBinomial')

    n_samples1, n_genes = Z.shape
    n_samples2, n_markers = G.shape

    assert n_samples1 == n_samples2

    # normalise and transform data
    print('Preparing data...', file=_sys.stderr)
    norm_factors = _utils.calculate_norm_factors(Z.T) if norm_factors is None else norm_factors
    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0)
    GTG = G.T.dot(G)

    # arguments
    args = {
        'n_samples': n_samples1,
        'n_markers': n_markers,
        'n_genes': n_genes,
        'n_iters': n_iters,
        'n_burnin': n_burnin,
        'beta_thr': beta_thr,
        's2_lims': s2_lims,
        'Z': Z,
        'G': G,
        'norm_factors': norm_factors,
        'GTG': GTG,
        **extra
    }

    if mdl in ('Normal', 'Normal2'):
        Y = _utils.transform_data(Z, norm_factors, kind=trans) if trans is not None else Z
        args['Y'] = (Y - _nmp.mean(Y, 0)) / _nmp.std(Y, 0) if std else Y - _nmp.mean(Y, 0)
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
    print('\r' + 'Iteration {0} of {1}'.format(0, n_iters), end='', file=_sys.stderr)
    with _mlp.Pool(processes=n_threads) as parallel:
        for itr in range(1, n_iters + 1):
            mdl.update(itr, parallel=parallel, **args)
            state[itr] = mdl.get_state(**args)
            print('\r' + 'Iteration {0} of {1}'.format(itr, n_iters), end='', file=_sys.stderr)
    print('\nDone!', file=_sys.stderr)

    #
    return state, mdl.get_estimates(**args)
