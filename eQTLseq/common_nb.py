"""Implements functions for the Negative Binomial model."""

import numpy as _nmp
import numpy.random as _rnd
import scipy.special as _spc

import eQTLseq.utils as _utils

_EPS = _nmp.finfo('float').eps


def sample_mu(Z, G, phi, beta, a=0.5, b=0.5):
    """Sample average expression level `mu` for each gene."""
    n_samples, _ = Z.shape

    Z = Z * _nmp.exp(-G.dot(beta.T))
    alpha = 1 / phi

    c1 = a + Z.sum(0)
    c2 = b + n_samples * alpha

    pi = _rnd.beta(c1, c2)
    pi = _nmp.clip(pi, _EPS, 1 - _EPS)

    mu = alpha * pi / (1 - pi)

    #
    return mu


def sample_phi(Z, G, mu, phi, beta, mu_phi, tau_phi):
    """Sample dispersion `phi` for each gene."""
    n_samples, n_genes = Z.shape
    means = mu * _nmp.exp(G.dot(beta.T))

    # sample proposals from the prior
    alpha = 1 / phi
    pi = means / (alpha + means)

    phi_ = _nmp.exp(mu_phi + _rnd.randn(n_genes) / _nmp.sqrt(tau_phi))
    alpha_ = 1 / phi_
    pi_ = means / (alpha_ + means)

    pi = _nmp.clip(pi, _EPS, 1 - _EPS)    # bound pi/pi_ between (0,1) to avoid ...
    pi_ = _nmp.clip(pi_, _EPS, 1 - _EPS)   # divide-by-zero errors

    # compute loglik
    loglik = (_spc.gammaln(Z + alpha) - _spc.gammaln(alpha) + alpha * _nmp.log1p(-pi) + Z * _nmp.log(pi)).sum(0)
    loglik_ = (_spc.gammaln(Z + alpha_) - _spc.gammaln(alpha_) + alpha_ * _nmp.log1p(-pi_) + Z * _nmp.log(pi_)).sum(0)

    # Metropolis step
    idxs = _nmp.log(_rnd.rand(n_genes)) < loglik_ - loglik
    phi[idxs] = phi_[idxs]


def sample_beta(Z, G, mu, phi, beta, tau, zeta, eta, idxs_genes, idxs_markers):
    """Sample matrix of coefficients `B` from a multivariate Normal distribution."""
    # get reduced parameters and data
    Z = Z[:, idxs_genes]
    G = G[:, idxs_markers]

    mu = mu[idxs_genes]
    phi = phi[idxs_genes]
    tau = tau[idxs_genes]
    eta = eta[idxs_markers]
    zeta = zeta[idxs_genes, :][:, idxs_markers]
    beta = beta[idxs_genes, :][:, idxs_markers]

    _, n_markers = G.shape

    # sample beta
    alpha = 1 / phi
    x0 = _nmp.log(mu) - _nmp.log(alpha)

    omega = _utils.sample_PG(Z + alpha, x0 + G.dot(beta.T))

    theta = tau[:, None] * zeta * eta
    A1 = _nmp.dot(omega.T[:, None, :] * G.T, G)
    A2 = theta[:, :, None] * _nmp.identity(n_markers)
    A = A1 + A2
    b = 0.5 * G.T.dot(Z - alpha - 2 * omega * x0)
    beta = _utils.sample_multivariate_normal_many(b.T, A)

    ##
    return beta


def sample_tau(beta, zeta, eta):
    """Sample gene-specific precision parameters `tau`."""
    _, n_markers = beta.shape

    # sample zeta
    shape = 0.5 * n_markers
    rate = 0.5 * (eta * beta**2 * zeta).sum(1)
    tau = _rnd.gamma(shape, 1 / rate)

    ##
    return _nmp.clip(tau, _EPS, 1/_EPS)


def sample_mu_tau_phi(phi):
    """Sample parameters of the log-normal prior of `phi`."""
    n_genes = phi.size
    log_phi = _nmp.log(phi)
    log_phi2 = log_phi**2

    # sample tau_phi
    shape = 0.5 * n_genes
    rate = 0.5 * log_phi2.sum()
    tau_phi = _rnd.gamma(shape, 1 / rate)

    # sample mu_phi, given tau_phi
    mean = log_phi.sum() / n_genes
    prec = n_genes * tau_phi
    mu_phi = _rnd.normal(mean, 1 / _nmp.sqrt(prec))

    #
    return mu_phi, tau_phi
