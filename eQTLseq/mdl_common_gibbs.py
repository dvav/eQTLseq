"""Implements posterior samplers used by more than one model."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.utils as _utils


def sample_beta(GTG, GTY, tau, zeta):
    """TODO."""
    _, n_markers = zeta.shape

    # sample beta
    A = tau[:, None, None] * (GTG + zeta[:, :, None] * _nmp.identity(n_markers))
    b = tau * GTY
    beta = _utils.sample_multivariate_normal2(b.T, A)

    ##
    return beta


def sample_tau(Y, G, beta, zeta):
    """TODO."""
    n_samples, n_markers = G.shape

    # sample tau
    resid = Y - G.dot(beta.T)
    shape = 0.5 * (n_samples + n_markers)
    rate = 0.5 * _nmp.sum(resid ** 2, 0) + 0.5 * _nmp.sum(beta ** 2 * zeta, 1)
    tau = _rnd.gamma(shape, 1 / rate)

    ##
    return tau


def sample_zeta(beta, tau):
    """TODO."""
    # sample zeta
    shape = 0.5
    rate = 0.5 * beta**2 * tau[:, None]
    zeta = shape / rate  # _rnd.gamma(shape, 1 / rate)

    ##
    return zeta
