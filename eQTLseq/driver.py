"""Implements run()."""

import sys as _sys

import numpy as _nmp

from eQTLseq.ModelBinomGibbs import ModelBinomGibbs as _ModelBinomGibbs
from eQTLseq.ModelNBinomGibbs import ModelNBinomGibbs as _ModelNBinomGibbs
from eQTLseq.ModelNBinomGibbs2 import ModelNBinomGibbs2 as _ModelNBinomGibbs2
from eQTLseq.ModelNormalGibbs import ModelNormalGibbs as _ModelNormalGibbs
from eQTLseq.ModelPoissonGibbs import ModelPoissonGibbs as _ModelPoissonGibbs

import eQTLseq.utils as _utils


def run(Z, G, mdl='Poisson', trans=None, norm_factors=None, n_iters=1000, n_burnin=None,
        beta_thr=1e-6, s2_lims=(1e-20, 1e3), tol=1e-6):
    """Run an estimation algorithm for a specified number of iterations."""
    n_burnin = round(n_iters * 0.5) if n_burnin is None else n_burnin
    assert mdl in ('Normal', 'Poisson', 'Binomial', 'NBinomial', 'NBinomial2')

    n_samples1, n_genes = Z.shape
    n_samples2, n_markers = G.shape

    assert n_samples1 == n_samples2

    # normalise and transform data
    print('Preparing data...', file=_sys.stderr)
    norm_factors = _utils.calculate_norm_factors(Z.T) if norm_factors is None else norm_factors
    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0)
    GTG = G.T.dot(G)

    if mdl == 'Normal':
        Y = _utils.transform_data(Z, norm_factors, kind=trans) if trans is not None else Z
        Y = Y - _nmp.mean(Y, 0)
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
        'norm_factors': norm_factors
    }

    # prepare model
    Model = {
        'Poisson': _ModelPoissonGibbs,
        'Binomial': _ModelBinomGibbs,
        'NBinomial': _ModelNBinomGibbs,
        'NBinomial2': _ModelNBinomGibbs2,
        'Normal': _ModelNormalGibbs
    }[mdl]
    mdl = Model(**args)

    # loop
    loglik = _nmp.empty(n_iters + 1)
    loglik.fill(_nmp.nan)
    loglik[0] = 0
    for itr in range(1, n_iters + 1):
        mdl.update(itr, **args)
        loglik[itr] = mdl.get_log_likelihood(**args)

        print('\r' + 'Iteration {0} of {1}'.format(itr, n_iters), end='', file=_sys.stderr)

    print('\nDone!', file=_sys.stderr)

    #
    return loglik, mdl.get_estimates(**args)
