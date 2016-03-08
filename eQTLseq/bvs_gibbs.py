"""Implements Bayesian variable selection through shrinkage."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.utils as _utils


def bvs_gibbs(Y, G, n_iters=1000, s2_lims=(1e-6, 1e6)):
    """Do Bayesian variable selection through shrinkage, similarly to the LASSO."""
    # prepare data
    Y = (Y - _nmp.mean(Y)) / _nmp.std(Y)
    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0)

    GTG = _nmp.dot(G.T, G)
    GTY = _nmp.dot(G.T, Y)

    n_samples, n_markers = G.shape

    # output variables
    s2_out = _nmp.empty(n_iters + 1)
    beta_out = _nmp.empty((n_iters + 1, n_markers))
    s2_beta_out = _nmp.empty((n_iters + 1, n_markers))

    # initial conditions
    tau = _rnd.rand()
    tau_beta = _rnd.rand(n_markers)
    beta = _rnd.normal(0, 1 / _nmp.sqrt(tau_beta))

    s2_out[0] = 1 / tau
    beta_out[0, :] = beta
    s2_beta_out[0, :] = 1 / tau_beta

    # loop
    for itr in range(1, n_iters + 1):
        beta = _sample_beta(GTG, GTY, tau, tau_beta)
        tau = _sample_tau(Y, G, beta, tau_beta)
        tau_beta = _sample_tau_beta(beta, tau)

        tau_beta = _nmp.clip(tau_beta, 1 / s2_lims[1], 1 / s2_lims[0])

        s2_out[itr] = 1 / tau
        beta_out[itr, :] = beta
        s2_beta_out[itr, :] = 1 / tau_beta

        print('Iteration {0} of {1}'.format(itr, n_iters), end='\r')

    ##
    return {'beta': beta_out, 's2_beta': s2_beta_out, 's2': s2_out}


def _sample_beta(GTG, GTY, tau, tau_beta):
    # sample beta
    A = tau * (GTG + _nmp.diag(tau_beta))
    b = tau * GTY
    beta = _utils.sample_multivariate_normal(b, A)

    ##
    return beta


def _sample_tau(Y, G, beta, tau_beta):
    n_samples, n_markers = G.shape

    # sample tau
    resid = Y - _nmp.dot(G, beta)
    shape = 0.5 * (n_markers + n_samples)
    rate = 0.5 * _nmp.sum(resid ** 2) + 0.5 * _nmp.sum(beta ** 2 * tau_beta)
    tau = _rnd.gamma(shape, 1 / rate)

    ##
    return tau


def _sample_tau_beta(beta, tau):
    # sample tau_beta
    shape = 0.5
    rate = 0.5 * beta**2 * tau
    tau_beta = shape / rate  # _rnd.gamma(shape, 1 / rate)

    ##
    return tau_beta
