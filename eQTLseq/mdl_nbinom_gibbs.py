"""Implements mdl_nbinom_gibbs."""

import numpy as _nmp
import numpy.random as _rnd
import scipy.special as _spc

import eQTLseq.utils as _utils


def mdl_nbinom_gibbs(Y, G, n_iters=1000, n_burnin=500, s2_lims=(1e-6, 1e6), phi_prior='LN'):
    """Search for eQTLs in normalised gene expression data, assuming a negative binomial model."""
    # prepare data
    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0)

    GTG = G.T.dot(G)

    n_samples1, n_genes = Y.shape
    n_samples2, n_markers = G.shape

    assert n_samples1 == n_samples2
    n_samples = n_samples1

    # phi prior
    _sample_phi_local, _sample_phi_global, _sample_mu_tau_phi = {
        'LN': (_sample_phi_local_ln, _sample_phi_global_ln, _sample_mu_tau_phi_ln),
        'EXP': (_sample_phi_local_exp, _sample_phi_global_exp, _sample_mu_tau_phi_exp),
        'JEF': (_sample_phi_local_jef, _sample_phi_global_jef, _sample_mu_tau_phi_jef)
    }[phi_prior]

    # initial conditions
    mu_phi, tau_phi = 0, 1
    phi, mu = _nmp.exp(_rnd.randn(n_genes)), _nmp.mean(Y, 0)
    zeta = _nmp.ones((n_genes, n_markers))
    tau_psi = _nmp.ones(n_genes)
    beta = _rnd.randn(n_genes, n_markers)
    psi = _rnd.randn(n_samples, n_genes)

    traces = _nmp.zeros((n_iters + 1, 8))
    traces[0, :] = [
        mu_phi, tau_phi, _nmp.sqrt(_nmp.sum(_nmp.log(phi)**2)), _nmp.sqrt(_nmp.sum(_nmp.log(mu)**2)),
        1 / _nmp.sqrt(_nmp.sum(tau_psi**2)), _nmp.sqrt(_nmp.sum(psi**2)),
        1 / _nmp.sqrt(_nmp.sum(zeta**2)), _nmp.sqrt(_nmp.sum(beta**2))
    ]

    # loop
    beta_sum, beta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))
    phi_sum, phi2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)
    mu_sum, mu2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)
    for itr in range(1, n_iters + 1):
        # sample phi and mu
        if(_rnd.rand() < 0.5):
            phi = _sample_phi_local(Y, mu, phi, psi, mu_phi, tau_phi)
        else:
            phi = _sample_phi_global(Y, mu, phi, psi, mu_phi, tau_phi)
        mu = _sample_mu(Y, phi, psi)

        # sample psi
        if(_rnd.rand() < 0.5):
            psi = _sample_psi_local(Y, G, mu, phi, psi, beta, tau_psi)
        else:
            psi = _sample_psi_global(Y, G, mu, phi, psi, beta, tau_psi)

        # sample beta
        beta = _sample_beta(GTG, G, psi, tau_psi, zeta)

        # sample the rest
        mu_phi, tau_phi = _sample_mu_tau_phi(phi, tau_phi)
        tau_psi = _sample_tau_psi(G, psi, beta, zeta)
        zeta = _sample_zeta(beta, tau_psi)

        zeta = _nmp.clip(zeta, 1 / s2_lims[1], 1 / s2_lims[0])
        tau_psi = _nmp.clip(tau_psi, 1 / s2_lims[1], 1 / s2_lims[0])

        # output
        beta2 = beta**2
        traces[itr, :] = [
            mu_phi, tau_phi, _nmp.sqrt(_nmp.sum(_nmp.log(phi)**2)), _nmp.sqrt(_nmp.sum(_nmp.log(mu)**2)),
            1 / _nmp.sqrt(_nmp.sum(tau_psi**2)), _nmp.sqrt(_nmp.sum(psi**2)),
            1 / _nmp.sqrt(_nmp.sum(zeta**2)), _nmp.sqrt(_nmp.sum(beta2))
        ]

        if(itr > n_burnin):
            beta_sum += beta
            phi_sum += phi
            mu_sum += mu

            beta2_sum += beta2
            phi2_sum += phi**2
            mu2_sum += mu**2

        # log
        print('Iteration {0} of {1}'.format(itr, n_iters), end='\r')

    ##
    N = n_iters - n_burnin
    beta_mean, mu_mean, phi_mean = beta_sum / N, mu_sum / N, phi_sum / N
    beta_var, mu_var, phi_var = beta2_sum / N - beta_mean**2, mu2_sum / N - mu_mean**2, phi2_sum / N - phi_mean**2
    return {
        'traces': traces,
        'beta_mean': beta_mean, 'beta_var': beta_var,
        'phi_mean': phi_mean, 'phi_var': phi_var,
        'mu_mean': mu_mean, 'mu_var': mu_var
    }


def _sample_phi_global_ln(Y, mu, phi, psi, mu_phi, tau_phi):
    n_samples, n_genes = Y.shape

    means = mu * _nmp.exp(psi)

    # sample proposals from the prior
    alpha = 1 / phi
    pi = alpha / (alpha + means)

    phi_ = _nmp.exp(_rnd.normal(mu_phi, 1 / _nmp.sqrt(tau_phi), n_genes))
    alpha_ = 1 / phi_
    pi_ = alpha_ / (alpha_ + means)

    # compute loglik
    loglik = _spc.gammaln(Y + alpha).sum(0) - n_samples * _spc.gammaln(alpha) \
        + alpha * _nmp.log(pi).sum(0) + _nmp.sum(Y * _nmp.log1p(-pi), 0)
    loglik_ = _spc.gammaln(Y + alpha_).sum(0) - n_samples * _spc.gammaln(alpha_) \
        + alpha_ * _nmp.log(pi_).sum(0) + _nmp.sum(Y * _nmp.log1p(-pi_), 0)

    # Metropolis step
    idxs = _rnd.rand(n_genes) < _nmp.exp(loglik_ - loglik)
    phi[idxs] = phi_[idxs]

    #
    return phi


def _sample_phi_local_ln(Y, mu, phi, psi, mu_phi, tau_phi, scale=0.01):
    n_samples, n_genes = Y.shape

    means = mu * _nmp.exp(psi)

    # sample proposals from a log-normal with small scale
    alpha = 1 / phi
    pi = alpha / (alpha + means)
    log_phi = _nmp.log(phi)

    phi_ = phi * _nmp.exp(scale * _rnd.randn(n_genes))
    alpha_ = 1 / phi_
    pi_ = alpha_ / (alpha_ + means)
    log_phi_ = _nmp.log(phi_)

    # compute loglik
    loglik = _spc.gammaln(Y + alpha).sum(0) - n_samples * _spc.gammaln(alpha) \
        + alpha * _nmp.log(pi).sum(0) + _nmp.sum(Y * _nmp.log1p(-pi), 0) \
        - log_phi - 0.5 * tau_phi * (log_phi - mu_phi)**2
    loglik_ = _spc.gammaln(Y + alpha_).sum(0) - n_samples * _spc.gammaln(alpha_) \
        + alpha_ * _nmp.log(pi_).sum(0) + _nmp.sum(Y * _nmp.log1p(-pi_), 0) \
        - log_phi_ - 0.5 * tau_phi * (log_phi_ - mu_phi)**2

    # Metropolis step
    idxs = _rnd.rand(n_genes) < _nmp.exp(loglik_ - loglik)
    phi[idxs] = phi_[idxs]

    #
    return phi


def _sample_mu_tau_phi_ln(phi, tau_phi):
    n_genes = phi.size
    log_phi = _nmp.log(phi)

    # sample mu_phi
    mean = log_phi.sum() / n_genes
    prec = n_genes * tau_phi
    mu_phi = _rnd.normal(mean, 1 / _nmp.sqrt(prec))

    # sample tau_phi
    shape = 0.5 * n_genes
    rate = 0.5 * _nmp.sum((log_phi - mu_phi)**2)
    tau_phi = _rnd.gamma(shape, 1 / rate)

    #
    return mu_phi, tau_phi


def _sample_phi_global_exp(Y, mu, phi, psi, mu_phi, tau_phi):
    n_samples, n_genes = Y.shape

    means = mu * _nmp.exp(psi)

    # sample proposals from the prior
    alpha = 1 / phi
    pi = alpha / (alpha + means)

    phi_ = _rnd.exponential(1 / tau_phi, n_genes)
    alpha_ = 1 / phi_
    pi_ = alpha_ / (alpha_ + means)

    # compute loglik
    loglik = _spc.gammaln(Y + alpha).sum(0) - n_samples * _spc.gammaln(alpha) \
        + alpha * _nmp.log(pi).sum(0) + _nmp.sum(Y * _nmp.log1p(-pi), 0)
    loglik_ = _spc.gammaln(Y + alpha_).sum(0) - n_samples * _spc.gammaln(alpha_) \
        + alpha_ * _nmp.log(pi_).sum(0) + _nmp.sum(Y * _nmp.log1p(-pi_), 0)

    # Metropolis step
    idxs = _rnd.rand(n_genes) < _nmp.exp(loglik_ - loglik)
    phi[idxs] = phi_[idxs]

    #
    return phi


def _sample_phi_local_exp(Y, mu, phi, psi, mu_phi, tau_phi, scale=0.01):
    n_samples, n_genes = Y.shape

    means = mu * _nmp.exp(psi)

    # sample proposals from a log-normal with small scale
    alpha = 1 / phi
    pi = alpha / (alpha + means)

    phi_ = phi * _nmp.exp(scale * _rnd.randn(n_genes))
    alpha_ = 1 / phi_
    pi_ = alpha_ / (alpha_ + means)

    # compute loglik
    loglik = _spc.gammaln(Y + alpha).sum(0) - n_samples * _spc.gammaln(alpha) \
        + alpha * _nmp.log(pi).sum(0) + _nmp.sum(Y * _nmp.log1p(-pi), 0) \
        - tau_phi * phi
    loglik_ = _spc.gammaln(Y + alpha_).sum(0) - n_samples * _spc.gammaln(alpha_) \
        + alpha_ * _nmp.log(pi_).sum(0) + _nmp.sum(Y * _nmp.log1p(-pi_), 0) \
        - tau_phi * phi_

    # Metropolis step
    idxs = _rnd.rand(n_genes) < _nmp.exp(loglik_ - loglik)
    phi[idxs] = phi_[idxs]

    #
    return phi


def _sample_mu_tau_phi_exp(phi, tau_phi):
    n_genes = phi.size

    # sample mu_phi
    mu_phi = _nmp.nan

    # sample tau_phi
    shape = n_genes
    rate = _nmp.sum(phi)
    tau_phi = _rnd.gamma(shape, 1 / rate)

    #
    return mu_phi, tau_phi


def _sample_phi_global_jef(Y, mu, phi, psi, mu_phi, tau_phi):
    n_samples, n_genes = Y.shape

    means = mu * _nmp.exp(psi)

    # sample proposals from the prior
    alpha = 1 / phi
    pi = alpha / (alpha + means)

    phi_ = _nmp.exp(_rnd.uniform(-20, 10, n_genes))
    alpha_ = 1 / phi_
    pi_ = alpha_ / (alpha_ + means)

    # compute loglik
    loglik = _spc.gammaln(Y + alpha).sum(0) - n_samples * _spc.gammaln(alpha) \
        + alpha * _nmp.log(pi).sum(0) + _nmp.sum(Y * _nmp.log1p(-pi), 0)
    loglik_ = _spc.gammaln(Y + alpha_).sum(0) - n_samples * _spc.gammaln(alpha_) \
        + alpha_ * _nmp.log(pi_).sum(0) + _nmp.sum(Y * _nmp.log1p(-pi_), 0)

    # Metropolis step
    idxs = _rnd.rand(n_genes) < _nmp.exp(loglik_ - loglik)
    phi[idxs] = phi_[idxs]

    #
    return phi


def _sample_phi_local_jef(Y, mu, phi, psi, mu_phi, tau_phi, scale=0.01):
    n_samples, n_genes = Y.shape

    means = mu * _nmp.exp(psi)

    # sample proposals from a log-normal with small scale
    alpha = 1 / phi
    pi = alpha / (alpha + means)

    phi_ = phi * _nmp.exp(scale * _rnd.randn(n_genes))
    alpha_ = 1 / phi_
    pi_ = alpha_ / (alpha_ + means)

    # compute loglik
    loglik = _spc.gammaln(Y + alpha).sum(0) - n_samples * _spc.gammaln(alpha) \
        + alpha * _nmp.log(pi).sum(0) + _nmp.sum(Y * _nmp.log1p(-pi), 0) \
        - _nmp.log(phi)
    loglik_ = _spc.gammaln(Y + alpha_).sum(0) - n_samples * _spc.gammaln(alpha_) \
        + alpha_ * _nmp.log(pi_).sum(0) + _nmp.sum(Y * _nmp.log1p(-pi_), 0) \
        - _nmp.log(phi_)

    # Metropolis step
    idxs = _rnd.rand(n_genes) < _nmp.exp(loglik_ - loglik)
    phi[idxs] = phi_[idxs]

    #
    return phi


def _sample_mu_tau_phi_jef(phi, tau_phi):
    return _nmp.nan, _nmp.nan


def _sample_mu(Y, phi, psi, a=0.5, b=0.5):
    n_samples, _ = Y.shape

    Y = Y / _nmp.exp(psi)
    alpha = 1 / phi

    c1 = a + n_samples * alpha
    c2 = b + Y.sum(0)

    pi = _rnd.beta(c1, c2)
    mu = alpha * (1 - pi) / pi

    #
    return mu


def _sample_psi_global(Y, G, mu, phi, psi, beta, tau_psi):
    n_samples, n_genes = Y.shape
    alpha = 1 / phi

    # sample proposals from a normal prior
    pi = alpha / (alpha + mu * _nmp.exp(psi))

    psi_ = _rnd.normal(G.dot(beta.T), 1 / _nmp.sqrt(tau_psi))
    pi_ = alpha / (alpha + mu * _nmp.exp(psi_))

    # compute loglik
    loglik = alpha * _nmp.log(pi) + Y * _nmp.log1p(-pi)
    loglik_ = alpha * _nmp.log(pi_) + Y * _nmp.log1p(-pi_)

    # do Metropolis step
    idxs = _rnd.rand(n_samples, n_genes) < _nmp.exp(loglik_ - loglik)
    psi[idxs] = psi_[idxs]

    #
    return psi


def _sample_psi_local(Y, G, mu, phi, psi, beta, tau_psi, scale=0.01):
    n_samples, n_genes = Y.shape
    alpha = 1 / phi
    GBT = G.dot(beta.T)

    # sample proposals from a log-normal with small scale
    pi = alpha / (alpha + _nmp.exp(psi))

    psi_ = psi * _nmp.exp(scale * _rnd.randn(n_samples, n_genes))
    pi_ = alpha / (alpha + _nmp.exp(psi_))

    # compute loglik
    loglik = alpha * _nmp.log(pi) + Y * _nmp.log1p(-pi) - 0.5 * tau_psi * (psi - GBT)**2
    loglik_ = alpha * _nmp.log(pi_) + Y * _nmp.log1p(-pi_) - 0.5 * tau_psi * (psi_ - GBT)**2

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
