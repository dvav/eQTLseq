"""Implements run()."""

import numpy as _nmp

from eQTLseq.ModelNBinomGibbs import ModelNBinomGibbs as _ModelNBinomGibbs
from eQTLseq.ModelNBinomGibbs2 import ModelNBinomGibbs2 as _ModelNBinomGibbs2
from eQTLseq.ModelNormalGibbs import ModelNormalGibbs as _ModelNormalGibbs

import eQTLseq.utils as _utils


def run(Y, G, mdl='Normal', norm=True, n_iters=1000, n_burnin=None, beta_thr=1e-6, s2_lims=(1e-20, 1e3), tol=1e-6):
    """Run an estimation algorithm for a specified number of iterations."""
    n_burnin = round(n_iters * 0.5) if n_burnin is None else n_burnin
    assert mdl in ('Normal', 'NBinom', 'NBinom2')

    n_samples1 = Y.shape[0]
    n_genes = Y.shape[1] if _nmp.ndim(Y) == 2 else 1
    n_samples2, n_markers = G.shape

    assert n_samples1 == n_samples2

    # normalize data
    if mdl in ('NBinom', 'NBinom2') and norm:
        Y = _utils.normalise_RNAseq_data(Y.T)[0].T

    if mdl == 'Normal':
        Y = Y - _nmp.mean(Y, 0)
    else:
        Y = Y / _nmp.mean(Y, 0)

    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0)

    # used later in calculations
    YTY = _nmp.sum(Y**2, 0)
    GTG = G.T.dot(G)
    GTY = G.T.dot(Y)

    # arguments
    args = {
        'n_samples': n_samples1,
        'n_markers': n_markers,
        'n_genes': n_genes,
        'n_iters': n_iters,
        'n_burnin': n_burnin,
        'beta_thr': beta_thr,
        's2_lims': s2_lims,
        'Y': Y,
        'G': G,
        'YTY': YTY,
        'GTG': GTG,
        'GTY': GTY
    }

    # prepare model
    Model = {
        'Normal': _ModelNormalGibbs,
        'NBinom': _ModelNBinomGibbs,
        'NBinom2': _ModelNBinomGibbs2
    }[mdl]
    mdl = Model(**args)

    # loop
    loglik = _nmp.empty(n_iters + 1)
    loglik.fill(_nmp.nan)
    loglik[0] = 0
    print('{} iterations (max):'.format(n_iters), end='')
    for itr in range(1, n_iters + 1):
        mdl.update(itr, **args)
        loglik[itr] = mdl.get_joint_log_likelihood(**args)

        # print('Iteration {0} of {1}'.format(itr, n_iters), end='\r')
        print('.', end='')
    print('Done!')

    #
    return loglik, mdl.get_estimates(**args)
