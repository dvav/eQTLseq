"""Implements run()."""

import numpy as _nmp
import tqdm as _tqdm

from eQTLseq import parallel as _prl

from eQTLseq.ModelBinomGibbs import ModelBinomGibbs as _ModelBinomGibbs
from eQTLseq.ModelNBinomGibbs import ModelNBinomGibbs as _ModelNBinomGibbs
from eQTLseq.ModelNormalGibbs import ModelNormalGibbs as _ModelNormalGibbs
from eQTLseq.ModelPoissonGibbs import ModelPoissonGibbs as _ModelPoissonGibbs


def run(Z, G, model='Normal', scale=True, n_iters=1000, burnin=0.5, beta_thr=1e-6, s2_lims=(1e-20, 1e3), n_threads=1,
        progress=True):
    """Run an estimation algorithm for a specified number of iterations."""
    Z = Z.T
    n_samples1, n_genes = Z.shape
    n_samples2, n_markers = G.shape

    assert n_samples1 == n_samples2
    assert model in ('Normal', 'Poisson', 'Binomial', 'NBinomial')
    assert n_iters > 0
    assert 0.5 <= burnin < 1
    assert 0 < beta_thr < 1e-6

    # data
    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0)
    GTG = G.T.dot(G)

    if model == 'Normal':
        Y = (Z - _nmp.mean(Z, 0)) / _nmp.std(Z, 0) if scale else Z - _nmp.mean(Z, 0)
        YTY = _nmp.sum(Y**2, 0)
        GTY = G.T.dot(Y)
    else:
        Y = YTY = GTY = None

    # args
    args = {
        'model': model,
        'n_samples': n_samples1,
        'n_markers': n_markers,
        'n_genes': n_genes,
        'n_iters': n_iters,
        'n_burnin': round(n_iters * burnin),
        'beta_thr': beta_thr,
        's2_lims': s2_lims,
        'scale': scale,
        'Z': Z,
        'Y': Y,
        'G': G,
        'GTG': GTG,
        'YTY': YTY,
        'GTY': GTY
    }

    # loop
    _prl.init(n_threads)
    mdl = _generate_model(**args)
    state = _nmp.r_[0, [_nmp.nan] * n_iters]
    for itr in _tqdm.tqdm(range(1, n_iters + 1), disable=not progress):
        mdl.update(itr, **args)
        state[itr] = mdl.get_state(**args)
    _prl.clean()

    #
    return {
        'state': state,
        **mdl.get_estimates(**args)
    }


def _generate_model(**args):
    """TODO."""
    Model = {
        'Poisson': _ModelPoissonGibbs,
        'Binomial': _ModelBinomGibbs,
        'NBinomial': _ModelNBinomGibbs,
        'Normal': _ModelNormalGibbs,
    }[args['model']]

    ##
    return Model(**args)
