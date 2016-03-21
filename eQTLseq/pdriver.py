"""Implements run()."""

import joblib as _jbl
import numpy as _nmp

from eQTLseq.ModelTraitNormalEM import ModelTraitNormalEM as _ModelTraitNormalEM
from eQTLseq.ModelTraitNormalGibbs import ModelTraitNormalGibbs as _ModelTraitNormalGibbs
from eQTLseq.ModelTraitNormalVB import ModelTraitNormalVB as _ModelTraitNormalVB

import eQTLseq.utils as _utils


def prun(Y, G, mdl='Normal', alg='Gibbs', norm=True, n_iters=1000, n_burnin=None, s2_lims=(1e-12, 1e12), tol=1e-6):
    """Run an estimation algorithm for a specified number of iterations."""
    n_burnin = round(n_iters * 0.5) if n_burnin is None else n_burnin
    assert mdl in ('Normal', 'NBinom')
    assert alg in ('Gibbs', 'EM', 'VB')

    n_samples1, n_genes = Y.shape
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
    args = [{
        'n_samples': n_samples1,
        'n_markers': n_markers,
        'n_iters': n_iters,
        'n_burnin': n_burnin,
        's2_lims': s2_lims,
        'Y': Y[:, _],
        'G': G,
        'YTY': YTY[_],
        'GTG': GTG,
        'GTY': GTY[:, _]
    } for _ in range(n_genes)]

    # prepare model
    Model = {
        'Normal': {
            'Gibbs': _ModelTraitNormalGibbs,
            'EM': _ModelTraitNormalEM,
            'VB': _ModelTraitNormalVB
        }
    }[mdl][alg]
    mdls = [Model(**arg) for arg in args]

    # loop
    itr, err, trace0 = 0, 1, _nmp.sum([mdl.trace[0] for mdl in mdls])
    print('{} iterations (max): '.format(n_iters), end='')
    with _jbl.Parallel(n_jobs=8) as parallel:
        while itr < n_iters and err > tol:
            itr = itr + 1
            # for mdl, arg in zip(mdls, args):
            #     mdl.update(itr, **arg)
            parallel(_jbl.delayed(mdl.update)(itr, **arg) for mdl, arg in zip(mdls, args))

            trace1 = _nmp.sum([mdl.trace[itr] for mdl in mdls])
            err = _nmp.abs(trace1 - trace0)
            trace0 = trace1

            # print('Iteration {0} of {1}'.format(itr, n_iters), end='\b')
            print('.', end='')
    print('Done!')

    trace = _nmp.asarray([mdl.trace for mdl in mdls]).sum(0)
    est = [mdl.get_estimates(**arg) for mdl, arg in zip(mdls, args)]

    #
    return trace, est
