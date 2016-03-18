"""Implements run()."""

import numpy as _nmp

from eQTLseq.ModelNBinomGibbs import ModelNBinomGibbs as _ModelNBinomGibbs
from eQTLseq.ModelNormalGibbs import ModelNormalGibbs as _ModelNormalGibbs
from eQTLseq.ModelTraitNormalGibbs import ModelTraitNormalGibbs as _ModelTraitNormalGibbs

from eQTLseq.ModelNormalVB import ModelNormalVB as _ModelNormalVB
from eQTLseq.ModelTraitNormalEM import ModelTraitNormalEM as _ModelTraitNormalEM
from eQTLseq.ModelTraitNormalVB import ModelTraitNormalVB as _ModelTraitNormalVB

import eQTLseq.utils as _utils


def run(Y, G, kind='eQTLs', mdl='Normal', alg='Gibbs', norm=True, n_iters=1000, n_burnin=None, s2_lims=(1e-12, 1e12),
        tol=1e-6):
    """Run an estimation algorithm for a specified number of iterations."""
    n_burnin = round(n_iters * 0.5) if n_burnin is None else n_burnin
    assert kind in ('eQTLs', 'Trait')
    assert mdl in ('Normal', 'NBinom')
    assert alg in ('Gibbs', 'EM', 'VB')

    n_samples1 = Y.shape[0]
    n_genes = Y.shape[1] if _nmp.ndim(Y) == 2 else 1
    n_samples2, n_markers = G.shape

    assert n_samples1 == n_samples2

    # normalize data
    if mdl == 'NBinom' and norm:
        Y = _utils.normalise_RNAseq_data(Y.T)[0].T

    if mdl == 'Normal':
        Y = Y - _nmp.mean(Y, 0)

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
        's2_lims': s2_lims,
        'Y': Y,
        'G': G,
        'YTY': YTY,
        'GTG': GTG,
        'GTY': GTY
    }

    # prepare model
    Model = {
        'eQTLs': {
            'Normal': {'Gibbs': _ModelNormalGibbs, 'VB': _ModelNormalVB},
            'NBinom': {'Gibbs': _ModelNBinomGibbs}
        },
        'Trait': {
            'Normal': {'Gibbs': _ModelTraitNormalGibbs, 'EM': _ModelTraitNormalEM, 'VB': _ModelTraitNormalVB}
        }
    }[kind][mdl][alg]
    mdl = Model(**args)

    # loop
    itr, err, trace0 = 0, 1, mdl.trace[0]
    print('{} iterations (max):'.format(n_iters), end='')
    while itr < n_iters and _nmp.any(err > tol):
        itr = itr + 1
        mdl.update(itr, **args)

        trace1 = mdl.trace[itr]
        err = _nmp.abs(trace1 - trace0)
        trace0 = trace1

        # print('Iteration {0} of {1}'.format(itr, n_iters), end='\b')
        print('.', end='')
    print('Done!')

    #
    return mdl.trace, mdl.get_estimates(**args)
