"""Implements posterior samplers used by more than one model."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.utils as _utils


def sample_beta_tau(YTY, GTG, GTY, zeta, eta, n_samples):
    """TODO."""
    _, n_markers = zeta.shape

    # sample tau
    shape = 0.5 * (n_samples + n_markers)
    rate = 0.5 * YTY
    tau = _rnd.gamma(shape, 1 / rate)

    # sample beta
    A = tau[:, None, None] * (GTG + zeta[:, :, None] * _nmp.diag(eta))
    b = tau * GTY
    beta = _utils.sample_multivariate_normal_many(b.T, A)
    # beta = _nmp.asarray([_utils.sample_multivariate_normal(b_, A_) for b_, A_ in zip(b.T, A)])
    # beta = _utils.sample_multivariate_normal2(b.T, A)

    ##
    return beta, tau


def sample_zeta(beta, tau, eta):
    """TODO."""
    # sample zeta
    shape = 0.5
    rate = 0.5 * eta * beta**2 * tau[:, None]
    zeta = shape / rate  # _rnd.gamma(shape, 1 / rate)

    ##
    return zeta


def sample_eta(beta, tau, zeta):
    """TODO."""
    n_genes, _ = zeta.shape

    # sample zeta
    shape = 0.5 * n_genes
    rate = 0.5 * (zeta * beta**2 * tau[:, None]).sum(0)
    eta = _rnd.gamma(shape, 1 / rate)

    ##
    return eta

##


def update_beta_tau(YTY, GTG, GTY, zeta, n_samples, n_markers):
    """TODO."""
    # update tau
    shape = 0.5 * (n_samples + n_markers)
    rate = 0.5 * YTY
    tau = shape / rate

    # update beta
    A = GTG + zeta[:, :, None] * _nmp.identity(n_markers)
    beta = _utils.chol_solve_many(A, GTY.T)

    ##
    return beta, tau


def update_zeta(beta, tau):
    """TODO."""
    # sample zeta
    shape = 0.5
    # rate = 0.5 * (beta**2 + beta_var) * tau[:, None]
    rate = 0.5 * beta**2 * tau[:, None]
    zeta = shape / rate

    ##
    return zeta
