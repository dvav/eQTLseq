"""Implements mdl_poisson_gibbs."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.utils as _utils


def mdl_poisson_gibbs(Y, G, n_iters=1000, n_burnin=500, s2_lims=(1e-6, 1e6)):
    """Search for eQTLs in normalised gene expression data, assuming a Poisson model."""
    # prepare data
    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0)

    GTG = G.T.dot(G)

    n_samples1, n_genes = Y.shape
    n_samples2, n_markers = G.shape

    assert n_samples1 == n_samples2
    n_samples = n_samples1

    # initial conditions
    mu = _nmp.mean(Y, 0)
    zeta = _nmp.ones((n_genes, n_markers))
    tau_psi = _nmp.ones(n_genes)
    beta = _rnd.randn(n_genes, n_markers)
    psi = _rnd.randn(n_samples, n_genes)

    traces = _nmp.zeros((n_iters + 1, 5))
    traces[0, :] = [
        _nmp.sqrt(_nmp.sum(_nmp.log(mu)**2)),
        1 / _nmp.sqrt(_nmp.sum(tau_psi**2)), _nmp.sqrt(_nmp.sum(psi**2)),
        1 / _nmp.sqrt(_nmp.sum(zeta**2)), _nmp.sqrt(_nmp.sum(beta**2))
    ]

    # loop
    beta_sum, beta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))
    mu_sum, mu2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)
    for itr in range(1, n_iters + 1):
        # sample psi
        if(_rnd.rand() < 0.5):
            psi = _sample_psi_local(Y, G, mu, psi, beta, tau_psi)
        else:
            psi = _sample_psi_global(Y, G, mu, psi, beta, tau_psi)

        # sample mu
        mu = _sample_mu2(Y, None, psi)

        # sample beta
        beta = _sample_beta(GTG, G, psi, tau_psi, zeta)

        # sample the rest
        tau_psi = _sample_tau_psi(G, psi, beta, zeta)
        zeta = _sample_zeta(beta, tau_psi)

        zeta = _nmp.clip(zeta, 1 / s2_lims[1], 1 / s2_lims[0])
        tau_psi = _nmp.clip(tau_psi, 1 / s2_lims[1], 1 / s2_lims[0])

        # output
        beta2 = beta**2
        traces[itr, :] = [
            _nmp.sqrt(_nmp.sum(_nmp.log(mu)**2)),
            1 / _nmp.sqrt(_nmp.sum(tau_psi**2)), _nmp.sqrt(_nmp.sum(psi**2)),
            1 / _nmp.sqrt(_nmp.sum(zeta**2)), _nmp.sqrt(_nmp.sum(beta2))
        ]

        if(itr > n_burnin):
            beta_sum += beta
            mu_sum += mu

            beta2_sum += beta2
            mu2_sum += mu**2

        # log
        print('Iteration {0} of {1}'.format(itr, n_iters), end='\r')

    ##
    N = n_iters - n_burnin
    beta_mean, mu_mean = beta_sum / N, mu_sum / N
    beta_var, mu_var = beta2_sum / N - beta_mean**2, mu2_sum / N - mu_mean**2
    return {
        'traces': traces,
        'beta_mean': beta_mean, 'beta_var': beta_var,
        'mu_mean': mu_mean, 'mu_var': mu_var
    }


def _sample_mu(Y, mu, psi, scale=0.1):
    _, n_genes = Y.shape

    ysum = Y.sum(0)
    psum = _nmp.exp(psi).sum(0)

    # proposal
    mu_ = mu * _nmp.exp(scale * _rnd.randn(n_genes))

    # loglik
    loglik = ysum * _nmp.log(mu) - mu * psum
    loglik_ = ysum * _nmp.log(mu_) - mu_ * psum

    # do Metropolis step
    idxs = _rnd.rand(n_genes) < _nmp.exp(loglik_ - loglik)
    mu[idxs] = mu_[idxs]

    #
    return mu


def _sample_mu2(Y, _, psi):
    n_samples, _ = Y.shape

    Y = Y / _nmp.exp(psi)

    shape = Y.sum(0)
    rate = n_samples

    mu = _rnd.gamma(shape, 1 / rate)

    #
    return mu


def _sample_psi_global(Y, G, mu, psi, beta, tau_psi):
    n_samples, n_genes = Y.shape

    # sample proposals from a normal prior
    psi_ = _rnd.normal(G.dot(beta.T), 1 / _nmp.sqrt(tau_psi))

    # compute loglik
    loglik = Y * psi - mu * _nmp.exp(psi)
    loglik_ = Y * psi_ - mu * _nmp.exp(psi_)

    # do Metropolis step
    idxs = _rnd.rand(n_samples, n_genes) < _nmp.exp(loglik_ - loglik)
    psi[idxs] = psi_[idxs]

    #
    return psi


def _sample_psi_local(Y, G, mu, psi, beta, tau_psi, scale=0.01):
    n_samples, n_genes = Y.shape
    GBT = G.dot(beta.T)

    # sample proposals from a log-normal with small scale
    psi_ = psi * _nmp.exp(scale * _rnd.randn(n_samples, n_genes))

    # compute loglik
    loglik = Y * psi - mu * _nmp.exp(psi) - 0.5 * tau_psi * (psi - GBT)**2
    loglik_ = Y * psi_ - mu * _nmp.exp(psi_) - 0.5 * tau_psi * (psi_ - GBT)**2

    # do Metropolis step
    idxs = _rnd.rand(n_samples, n_genes) < _nmp.exp(loglik_ - loglik)
    psi[idxs] = psi_[idxs]

    #
    return psi


def _sample_tau_psi(G, psi, beta, zeta):
    n_samples, n_markers = G.shape

    # sample tau
    resid = psi - G.dot(beta.T)
    shape = 0.5 * (n_samples + n_markers)
    rate = 0.5 * _nmp.sum(resid ** 2, 0) + 0.5 * _nmp.sum(beta ** 2 * zeta, 1)
    tau_psi = _rnd.gamma(shape, 1 / rate)

    #
    return tau_psi


def _sample_beta(GTG, G, psi, tau_psi, zeta):
    _, n_markers = zeta.shape

    # sample beta
    A = tau_psi[:, None, None] * (GTG + zeta[:, :, None] * _nmp.identity(n_markers))
    b = tau_psi * G.T.dot(psi)
    beta = _utils.sample_multivariate_normal2(b.T, A)

    ##
    return beta


def _sample_zeta(beta, tau_psi):
    # sample zeta
    shape = 0.5
    rate = 0.5 * beta**2 * tau_psi[:, None]
    zeta = shape / rate  # _rnd.gamma(shape, 1 / rate)

    ##
    return zeta
