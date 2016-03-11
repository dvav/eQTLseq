"""Implements ModelTraitNormalEM."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.utils as _utils


class ModelTraitNormalEM(object):
    """A normal model of Bayesian variable selection through shrinkage for a single trait estimated using EM."""

    def __init__(self, **args):
        """TODO."""
        Y, G, n_iters, s2_lims = args['Y'], args['G'], args['n_iters'], args['s2_lims']

        # standardize data
        self.Y = (Y - _nmp.mean(Y)) / _nmp.std(Y)
        self.G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0)

        # used later in calculations
        self.GTG = self.G.T.dot(self.G)
        self.GTY = self.G.T.dot(self.Y)

        # number of samples and genetic markers
        n_samples, n_markers = self.G.shape

        # initial conditions
        self.tau = _rnd.rand()
        self.zeta = _rnd.rand(n_markers)
        self.beta = _rnd.normal(0, 1 / _nmp.sqrt(self.zeta))

        self.traces = _nmp.empty((n_iters + 1, 3))
        self.traces.fill(_nmp.nan)
        self.traces[0, :] = [
            1 / self.tau,
            1 / _utils.norm(self.zeta),
            _utils.norm(self.beta)
        ]

        # other parameters
        self.s2_min = s2_lims[0]
        self.s2_max = s2_lims[1]

    def update(self, itr):
        """TODO."""
        # E-step
        self.zeta = _update_zeta(self.beta, self.tau)
        self.zeta = _nmp.clip(self.zeta, 1 / self.s2_max, 1 / self.s2_min)

        # M-step
        self.beta = _update_beta(self.GTG, self.GTY, self.tau, self.zeta)
        self.tau = _update_tau(self.Y, self.G, self.beta, self.zeta)

        # update the rest
        self.traces[itr, :] = [
            1 / self.tau,
            1 / _utils.norm(self.zeta),
            _utils.norm(self.beta)
        ]

    def stats(self):
        """TODO."""
        return {
            'traces': self.traces,
            'tau': self.tau,
            'zeta': self.zeta,
            'beta': self.beta
        }


def _update_beta(GTG, GTY, tau, zeta):
    # sample beta
    A = tau * (GTG + _nmp.diag(zeta))
    b = tau * GTY
    beta = _utils.chol_solve(A, b)

    ##
    return beta


def _update_tau(Y, G, beta, zeta):
    n_samples, n_markers = G.shape

    # sample tau
    resid = Y - G.dot(beta)
    shape = 0.5 * (n_markers + n_samples)
    rate = 0.5 * _nmp.sum(resid ** 2) + 0.5 * _nmp.sum(beta ** 2 * zeta)
    tau = (shape - 1) / rate

    ##
    return tau


def _update_zeta(beta, tau):
    # sample tau_beta
    shape = 0.5
    rate = 0.5 * beta**2 * tau
    zeta = shape / rate

    ##
    return zeta
