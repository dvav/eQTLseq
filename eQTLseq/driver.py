"""Implements run()."""

import sys as _sys

import numpy as _nmp
import multiprocessing as _mlp

from eQTLseq.ModelBinomGibbs import ModelBinomGibbs as _ModelBinomGibbs
from eQTLseq.ModelNBinomGibbs import ModelNBinomGibbs as _ModelNBinomGibbs
from eQTLseq.ModelNBinomGibbs2 import ModelNBinomGibbs2 as _ModelNBinomGibbs2
from eQTLseq.ModelNBinomGibbs3 import ModelNBinomGibbs3 as _ModelNBinomGibbs3
from eQTLseq.ModelNBinomGibbs4 import ModelNBinomGibbs4 as _ModelNBinomGibbs4
from eQTLseq.ModelNormalGibbs import ModelNormalGibbs as _ModelNormalGibbs
from eQTLseq.ModelNormalGibbs2 import ModelNormalGibbs2 as _ModelNormalGibbs2
from eQTLseq.ModelPoissonGibbs import ModelPoissonGibbs as _ModelPoissonGibbs

from eQTLseq.ModelNormalEM import ModelNormalEM as _ModelNormalEM

from eQTLseq.ModelNormalVB import ModelNormalVB as _ModelNormalVB

import eQTLseq.utils as _utils


def run(Z, G, mdl='Normal', alg='Gibbs', trans=None, std=True, norm_factors=None, n_iters=1000, n_burnin=None,
        beta_thr=1e-6, s2_lims=(1e-20, 1e3), tol_abs=1e-6, tol_rel=1e-3, n_threads=1, **extra):
    """Run an estimation algorithm for a specified number of iterations."""
    n_burnin = round(n_iters * 0.5) if n_burnin is None else n_burnin
    assert mdl in ('Normal', 'Normal2', 'Poisson', 'Binomial', 'NBinomial', 'NBinomial2', 'NBinomial3', 'NBinomial4')
    assert alg in ('Gibbs', 'VB', 'EM')

    n_samples1, n_genes = Z.shape
    n_samples2, n_markers = G.shape

    assert n_samples1 == n_samples2

    # normalise and transform data
    print('Preparing data...', file=_sys.stderr)
    norm_factors = _utils.calculate_norm_factors(Z.T) if norm_factors is None else norm_factors
    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0)
    GTG = G.T.dot(G)

    if mdl in ('Normal', 'Normal2'):
        Y = _utils.transform_data(Z, norm_factors, kind=trans) if trans is not None else Z
        Y = (Y - _nmp.mean(Y, 0)) / _nmp.std(Y, 0) if std else Y - _nmp.mean(Y, 0)
        YTY = _nmp.sum(Y**2, 0)
        GTY = G.T.dot(Y)
    else:
        Y = None
        YTY = None
        GTY = None

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
        'Y': Y,
        'G': G,
        'YTY': YTY,
        'GTG': GTG,
        'GTY': GTY,
        'norm_factors': norm_factors,
        'mu': extra['mu'],
        'phi': extra['phi'],
        'YY': extra['YY'],
        'beta': extra['beta']
    }

    # prepare model
    Model = {
        'Poisson': {'Gibbs': _ModelPoissonGibbs},
        'Binomial': {'Gibbs': _ModelBinomGibbs},
        'NBinomial': {'Gibbs': _ModelNBinomGibbs},
        'NBinomial2': {'Gibbs': _ModelNBinomGibbs2},
        'NBinomial3': {'Gibbs': _ModelNBinomGibbs3},
        'NBinomial4': {'Gibbs': _ModelNBinomGibbs4},
        'Normal': {'Gibbs': _ModelNormalGibbs, 'VB': _ModelNormalVB, 'EM': _ModelNormalEM},
        'Normal2': {'Gibbs': _ModelNormalGibbs2}
    }[mdl][alg]
    mdl = Model(**args)

    # loop
    state = _nmp.empty(n_iters + 1)
    state.fill(_nmp.nan)
    state[0] = 0
    print('\r' + 'Iteration {0} of {1}'.format(0, n_iters), end='', file=_sys.stderr)
    with _mlp.Pool(processes=n_threads) as parallel:
        for itr in range(1, n_iters + 1):
            # update
            mdl.update(itr, parallel=parallel, **args)
            state[itr] = mdl.get_state(**args)

            # log
            print('\r' + 'Iteration {0} of {1}'.format(itr, n_iters), end='', file=_sys.stderr)

            # # error
            # err_abs = _nmp.abs(state[itr] - state[itr-1])
            # err_rel = _nmp.abs((state[itr] - state[itr-1])/state[itr-1])
            # if err_abs < tol_abs and err_rel < tol_rel:
            #     break

    print('\nDone!', file=_sys.stderr)

    #
    return state, mdl.get_estimates(**args)
