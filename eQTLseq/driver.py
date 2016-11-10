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


def run(Z, G, model='Normal', scaleG=True, centreZ=True, n_iters=1000, burnin=0.5, beta_thr=1e-6, s2_lims=(1e-20, 1e3),
        n_threads=1, progress=True):
    """Run an estimation algorithm for a specified number of iterations."""
    n_samples1, n_genes = Z.shape
    n_samples2, n_markers = G.shape

    assert n_samples1 == n_samples2
    assert model in ('Normal', 'Poisson', 'Binomial', 'NBinomial')
    assert n_iters > 0
    assert 0.5 <= burnin < 1
    assert 0 < beta_thr <= 1e-6

    # data
    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0) if scaleG else G
    GTG = G.T.dot(G)

    if model == 'Normal':
        Y = Z - _nmp.mean(Z, 0) if centreZ else Z
        YTY = _nmp.sum(Y**2, 0)
        GTY = G.T.dot(Y)
    else:
        Y = YTY = GTY = None

    # args
    args = {
        'n_samples': n_samples1,
        'n_markers': n_markers,
        'n_genes': n_genes,
        'n_iters': n_iters,
        'n_burnin': round(n_iters * burnin),
        'beta_thr': beta_thr,
        's2_lims': s2_lims,
        'scaleG': scaleG,
        'centreZ': centreZ,
        'Z': Z,
        'Y': Y,
        'G': G,
        'GTG': GTG,
        'YTY': YTY,
        'GTY': GTY
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


def get_metrics(Z, G, res, model='Normal', scaleG=True, centreZ=True):
    """TODO."""
    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0) if scaleG else G
    if model == 'Normal':
        Z = Z - _nmp.mean(Z, 0) if centreZ else Z

    ##
    return {
        'R2': _MODELS[model].get_R2(Z, G, res),
        'X2': _MODELS[model].get_X2(Z, G, res),
        'X2p': _MODELS[model].get_X2p(Z, G, res),
        'X2c': _MODELS[model].get_X2c(Z, G, res),
        'nMSE': _MODELS[model].get_nMSE(Z, G, res),
        'PCC': _MODELS[model].get_PCC(Z, G, res),
        'RHO': _MODELS[model].get_RHO(Z, G, res)
    }
