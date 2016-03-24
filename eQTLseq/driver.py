"""Implements run()."""

import sys as _sys

import numpy as _nmp

from eQTLseq.ModelNBinomGibbs import ModelNBinomGibbs as _ModelNBinomGibbs
from eQTLseq.ModelNBinomGibbs2 import ModelNBinomGibbs2 as _ModelNBinomGibbs2
from eQTLseq.ModelNormalGibbs import ModelNormalGibbs as _ModelNormalGibbs
from eQTLseq.ModelPoissonGibbs import ModelPoissonGibbs as _ModelPoissonGibbs

import eQTLseq.utils as _utils


def run(D, G, mdl='Normal', norm=True, n_iters=1000, n_burnin=None, beta_thr=1e-6, s2_lims=(1e-20, 1e3), tol=1e-6, mu=None, phi=None, YY=None):
    """Run an estimation algorithm for a specified number of iterations."""
    n_burnin = round(n_iters * 0.5) if n_burnin is None else n_burnin
    assert mdl in ('Normal', 'Poisson', 'NBinom', 'NBinom2')

    n_samples1 = D.shape[0]
    n_genes = D.shape[1] if _nmp.ndim(D) == 2 else 1
    n_samples2, n_markers = G.shape

    assert n_samples1 == n_samples2

    # normalize data
    if mdl in ('NBinom', 'NBinom2', 'Poisson') and norm:
        norm_factors = _utils.normalise_RNAseq_data(D.T)
    else:
        norm_factors = _nmp.ones(n_samples1)

    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0)
    GTG = G.T.dot(G)

    if mdl == 'Normal':
        Z = None
        Y = D - _nmp.mean(D, 0)
        YTY = _nmp.sum(Y**2, 0)
        GTY = G.T.dot(Y)
    else:
        Z = D
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
        'YY': YY,
        'mu': mu,
        'phi': phi,
        'norm_factors': norm_factors
    }

    # prepare model
    Model = {
        'Normal': _ModelNormalGibbs,
        'Poisson': _ModelPoissonGibbs,
        'NBinom': _ModelNBinomGibbs,
        'NBinom2': _ModelNBinomGibbs2,
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
