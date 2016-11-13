"""Implements run()."""

import numpy as _nmp
import tqdm as _tqdm

from eQTLseq import parallel as _prl

from eQTLseq.ModelBinomGibbs import ModelBinomGibbs as _ModelBinomGibbs
from eQTLseq.ModelNBinomGibbs import ModelNBinomGibbs as _ModelNBinomGibbs
from eQTLseq.ModelNormalGibbs import ModelNormalGibbs as _ModelNormalGibbs
from eQTLseq.ModelPoissonGibbs import ModelPoissonGibbs as _ModelPoissonGibbs

_MODELS = {
    'Poisson': _ModelPoissonGibbs,
    'Binomial': _ModelBinomGibbs,
    'NBinomial': _ModelNBinomGibbs,
    'Normal': _ModelNormalGibbs
}


def run(Z, G, model='Normal', scaleG=True, n_iters=1000, burnin=0.5, beta_thr=1e-6, s2_lims=(1e-20, 1e3),
        n_threads=1, progress=True):
    """Run an estimation algorithm for a specified number of iterations."""
    n_samples1, n_genes = Z.shape
    n_samples2, n_markers = G.shape

    assert n_samples1 == n_samples2
    assert model in ('Normal', 'Poisson', 'Binomial', 'NBinomial')
    assert n_iters > 0
    assert 0.5 <= burnin < 1
    assert 0 < beta_thr <= 1e-6
    assert s2_lims[0] > 0 and s2_lims[1] > 0 and s2_lims[0] < s2_lims[1]

    # data
    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0) if scaleG else G

    # args
    args = {
        'n_iters': n_iters,
        'n_burnin': round(n_iters * burnin),
        'beta_thr': beta_thr,
        's2_lims': s2_lims,
        'Z': Z,
        'G': G
    }

    # loop
    _prl.init(n_threads)
    mdl = _MODELS[model](**args)
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


def get_metrics(Z, G, res, model='Normal', scaleG=True):
    """TODO."""
    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0) if scaleG else G

    ##
    return _MODELS[model].get_metrics(Z, G, res)
