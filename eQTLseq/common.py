"""Implements common model functions."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.utils as _utils

_EPS = _nmp.finfo('float').eps


def get_idxs_redux(beta, tau, zeta, eta, beta_thr):
    """Identifies relevant genes and markers."""
    idxs = _nmp.abs(beta) > beta_thr

    idxs_markers = _nmp.any(idxs, 0)
    idxs_genes = _nmp.any(idxs, 1)

    ##
    return idxs_genes, idxs_markers, idxs


def sample_beta(Z, G, mu, tau, zeta, eta, idxs_genes, idxs_markers):
    """Sample matrix of coefficients `B` from a multivariate Normal distribution."""
    # get reduced parameters and data
    Z = Z[:, idxs_genes]
    G = G[:, idxs_markers]

    mu = mu[idxs_genes]
    tau = tau[idxs_genes]
    eta = eta[idxs_markers]
    zeta = zeta[idxs_genes, :][:, idxs_markers]

    _, n_markers = G.shape

    # sample beta
    theta = zeta * eta
    A = tau[:, None, None] * (G.T.dot(G) + theta[:, :, None] * _nmp.identity(n_markers))
    b = tau * G.T.dot(Z - mu)
    beta = _utils.sample_multivariate_normal_many(b.T, A)

    ##
    return beta


def sample_mu(Z, G, beta, tau):
    """Sample average expression level `mu` for each gene."""
    n_samples, _ = G.shape

    resid = Z - G.dot(beta.T)
    mean = resid.sum(0) / n_samples
    prec = n_samples * tau
    mu = _rnd.normal(mean, 1 / _nmp.sqrt(prec))

    ##
    return mu


def sample_tau(Z, G, beta, mu, zeta, eta):
    """Sample gene-specific precision parameters `tau`."""
    n_samples, n_markers = G.shape

    # sample tau
    A = (Z - mu - G.dot(beta.T))**2
    B = zeta * eta * beta**2

    shape = 0.5 * (n_samples + n_markers)
    rate = 0.5 * (A.sum(0) + B.sum(1))
    tau = _rnd.gamma(shape, 1 / rate)

    ##
    return _nmp.clip(tau, _EPS, 1/_EPS)


def sample_zeta(beta, tau, eta):
    """Sample gene- and variant-specific parameters `zeta`."""
    # sample zeta
    shape = 0.5
    rate = 0.5 * eta * beta**2 * tau[:, None]
    zeta = shape / rate  # _rnd.gamma(shape, 1 / rate)

    ##
    return _nmp.clip(zeta, _EPS, 1/_EPS)


def sample_eta(beta, tau, zeta):
    """Sample variant-specific parameters `eta`."""
    n_genes, _ = zeta.shape

    # sample zeta
    A = zeta * beta**2 * tau[:, None]
    shape = 0.5 * n_genes
    rate = 0.5 * A.sum(0)
    eta = _rnd.gamma(shape, 1 / rate)

    ##
    return _nmp.clip(eta, _EPS, 1/_EPS)
