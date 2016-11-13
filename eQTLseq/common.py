"""Implements common model functions."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.utils as _utils


def get_estimates(tags, sums, sums2, N):
    """TODO."""
    means = [s / N for s in sums]
    varrs = [s2 / N - mu**2 for s2, mu, in zip(sums2, means)]
    tags_vars = [_ + '_var' for _ in tags]

    return {
        **dict(zip(tags, means)),
        **dict(zip(tags_vars, varrs))
    }


def get_idxs_redux(beta, tau, zeta, eta, beta_thr, s2_lims):
    """Identifies relevant genes and markers."""
    idxs = (_nmp.abs(beta) > beta_thr) & (tau[:, None] * zeta * eta < 1 / s2_lims[0])
    idxs[[0, 1], [0, 1]] = True  # just a precaution
    idxs_markers = _nmp.any(idxs, 0)
    idxs_genes = _nmp.any(idxs, 1)

    ##
    return idxs_genes, idxs_markers


def sample_beta(Z, G, mu, tau, zeta, eta, idxs_genes, idxs_markers):
    """TODO."""
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
    """TODO."""
    n_samples, _ = G.shape

    resid = Z - G.dot(beta.T)
    mean = resid.sum(0) / n_samples
    prec = n_samples * tau
    mu = _rnd.normal(mean, 1 / _nmp.sqrt(prec))

    ##
    return mu


def sample_tau(Z, G, beta, mu, zeta, eta, s2_lims):
    """TODO."""
    n_samples, n_markers = G.shape

    # sample tau
    A = (Z - mu - G.dot(beta.T))**2
    B = zeta * eta * beta**2

    shape = 0.5 * (n_samples + n_markers)
    rate = 0.5 * (A.sum(0) + B.sum(1))
    tau = _rnd.gamma(shape, 1 / rate)

    ##
    return _nmp.clip(tau, 1 / s2_lims[1], 1 / s2_lims[0])


def sample_zeta(beta, tau, eta, s2_lims):
    """TODO."""
    # sample zeta
    shape = 0.5
    rate = 0.5 * eta * beta**2 * tau[:, None]
    zeta = shape / rate  # _rnd.gamma(shape, 1 / rate)

    ##
    return _nmp.clip(zeta, 1 / s2_lims[1], 1 / s2_lims[0])


def sample_eta(beta, tau, zeta, s2_lims):
    """TODO."""
    n_genes, _ = zeta.shape

    # sample zeta
    A = zeta * beta**2 * tau[:, None]
    shape = 0.5 * n_genes
    rate = 0.5 * A.sum(0)
    eta = _rnd.gamma(shape, 1 / rate)

    ##
    return _nmp.clip(eta, 1 / s2_lims[1], 1 / s2_lims[0])
