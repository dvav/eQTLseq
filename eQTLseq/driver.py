"""Implements run()."""

import numpy as _nmp

from eQTLseq import parallel as _prl
from eQTLseq import utils as _utils
from eQTLseq import alg as _alg

from eQTLseq.ModelBinom import ModelBinom as _ModelBinom
from eQTLseq.ModelNBinom import ModelNBinom as _ModelNBinom
from eQTLseq.ModelNormal import ModelNormal as _ModelNormal
from eQTLseq.ModelPoisson import ModelPoisson as _ModelPoisson

_MODELS = {
    'Poisson': _ModelPoisson,
    'Binomial': _ModelBinom,
    'NBinomial': _ModelNBinom,
    'Normal': _ModelNormal
}

_EPS = _nmp.finfo('float').eps


def run(Z, G, model='Normal', scaleG=True, n_iters=1000, burnin=0.8, beta_thr=1e-6, n_threads=1, hide_progress=False):
    """Run an estimation algorithm for a specified number of iterations."""
    n_samples1, n_genes = Z.shape
    n_samples2, n_markers = G.shape

    assert n_samples1 == n_samples2
    assert model in ('Normal', 'Poisson', 'Binomial', 'NBinomial')
    assert n_iters > 0
    assert 0.5 <= burnin < 1
    assert 0 <= beta_thr

    # data
    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0) if scaleG else G

    # args
    args = {
        'n_iters': n_iters,
        'n_burnin': round(n_iters * burnin),
        'hide_progress': hide_progress,
        'beta_thr': beta_thr,
        'Z': Z,
        'G': G
    }

    # main
    _prl.init(n_threads)

    mdl = _MODELS[model](**args)
    res = _alg.gibbs(mdl, **args)

    _prl.close()

    #
    return res


def get_metrics(Z, G, res, model='Normal', scaleG=True):
    """Auxilliary function for computing predicitive metrics, after estimation has been completed."""
    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0) if scaleG else G

    Zhat = _MODELS[model].get_pred(Z, G, res)
    Z, Zhat = (Z, Zhat) if model == 'Normal' else (_nmp.log(Z + 1), _nmp.log(Zhat + 1))

    CCC = _utils.compute_ccc(Z, Zhat)
    MSE = _nmp.mean((Z - Zhat) ** 2)
    RMSE = _nmp.sqrt(MSE)
    NRMSE = RMSE / (_nmp.std(Z) + _EPS)
    N2RMSE = RMSE / (_nmp.max(Z) - _nmp.min(Z) + _EPS)

    ##
    return {
        'CCC': CCC,
        'RMSE': RMSE,
        'NRMSE': NRMSE,
        'N2RMSE': N2RMSE
    }
