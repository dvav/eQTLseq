"""Implements mdl_normal_gibbs."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.utils as _utils


def mdl_normal_gibbs(Y, G, n_iters=1000, n_burnin=500, s2_lims=(1e-6, 1e6)):
    """Search for eQTLs in normally distributed gene expression data."""
    # prepare data
    Y = (Y - _nmp.mean(Y, 0)) / _nmp.std(Y, 0)
    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0)

    GTG = G.T.dot(G)
    GTY = G.T.dot(Y)

    n_samples1, n_genes = Y.shape
    n_samples2, n_markers = G.shape

    assert n_samples1 == n_samples2

    # initial conditions
    tau = _nmp.ones(n_genes)
    zeta = _nmp.ones((n_genes, n_markers))
    beta = _rnd.randn(n_genes, n_markers)

    traces = _nmp.zeros((n_iters + 1, 3))
    traces[0, :] = [_nmp.sqrt(_nmp.sum(tau**2)), 1 / _nmp.sqrt(_nmp.sum(zeta**2)), _nmp.sqrt(_nmp.sum(beta**2))]

    # loop
    tau_sum, tau2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)
    zeta_sum, zeta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))
    beta_sum, beta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))

    for itr in range(1, n_iters + 1):
        beta = _sample_beta(GTG, GTY, tau, zeta)
        tau = _sample_tau(Y, G, beta, zeta)
        zeta = _sample_zeta(beta, tau)

        zeta = _nmp.clip(zeta, 1 / s2_lims[1], 1 / s2_lims[0])

        # output
        tau2, zeta2, beta2 = tau**2, zeta**2, beta**2
        traces[itr, :] = [_nmp.sqrt(_nmp.sum(tau2)), 1 / _nmp.sqrt(_nmp.sum(zeta2)), _nmp.sqrt(_nmp.sum(beta2))]

        if(itr > n_burnin):
            tau_sum += tau
            zeta_sum += zeta
            beta_sum += beta

            tau2_sum += tau2
            zeta2_sum += zeta2
            beta2_sum += beta2

        # log
        print('Iteration {0} of {1}'.format(itr, n_iters), end='\r')

    ##
    N = n_iters - n_burnin
    tau_mean, zeta_mean, beta_mean = tau_sum / N, zeta_sum / N, beta_sum / N
    tau_var, zeta_var, beta_var = tau2_sum / N - tau_mean**2, zeta2_sum / N - zeta_mean**2, beta2_sum / N - beta_mean**2
    return {'traces': traces,
            'tau_mean': tau_mean, 'tau_var': tau_var,
            'zeta_mean': zeta_mean, 'zeta_var': zeta_var,
            'beta_mean': beta_mean, 'beta_var': beta_var}


def _sample_beta(GTG, GTY, tau, zeta):
    _, n_markers = zeta.shape

    # sample beta
    A = tau[:, None, None] * (GTG + zeta[:, :, None] * _nmp.identity(n_markers))
    b = tau * GTY
    beta = _utils.sample_multivariate_normal2(b.T, A)

    ##
    return beta


def _sample_tau(Y, G, beta, zeta):
    n_samples, n_markers = G.shape

    # sample tau
    resid = Y - G.dot(beta.T)
    shape = 0.5 * (n_samples + n_markers)
    rate = 0.5 * _nmp.sum(resid ** 2, 0) + 0.5 * _nmp.sum(beta ** 2 * zeta, 1)
    tau = _rnd.gamma(shape, 1 / rate)

    ##
    return tau


def _sample_zeta(beta, tau):
    # sample zeta
    shape = 0.5
    rate = 0.5 * beta**2 * tau[:, None]
    zeta = shape / rate  # _rnd.gamma(shape, 1 / rate)

    ##
    return zeta
