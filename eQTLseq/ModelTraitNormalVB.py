"""Implements ModelTraitNormalVB."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.utils as _utils


class ModelTraitNormalVB(object):
    """A normal model of Bayesian variable selection through shrinkage for a single trait estimated using VB."""

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
        self.tau_mean, self.tau_var = _rnd.rand(), 1
        self.zeta_mean, self.zeta_var = _rnd.rand(n_markers), _nmp.ones(n_markers)
        self.beta_mean, self.beta_var = _rnd.normal(0, 1 / _nmp.sqrt(self.zeta_mean)), _nmp.ones(n_markers)

        self._traces = _nmp.empty((n_iters + 1, 3))
        self._traces.fill(_nmp.nan)
        self._traces[0, :] = [
            1 / self.tau_mean,
            1 / _utils.norm(self.zeta_mean),
            _utils.norm(self.beta_mean)
        ]

        # other parameters
        self.s2_min = s2_lims[0]
        self.s2_max = s2_lims[1]

    def update(self, itr):
        """TODO."""
        # update beta, tau and zeta
        self.beta_mean, self.beta_var = _update_beta(self.GTG, self.GTY, self.tau_mean, self.zeta_mean)
        self.tau_mean, self.tau_var = _update_tau(self.Y, self.G, self.beta_mean, self.tau_mean, self.zeta_mean)
        self.zeta_mean, self.zeta_var = _update_zeta(self.beta_mean, self.beta_var, self.tau_mean)

        self.zeta_mean = _nmp.clip(self.zeta_mean, 1 / self.s2_max, 1 / self.s2_min)

        # update the rest
        self._traces[itr, :] = [
            1 / self.tau_mean,
            1 / _utils.norm(self.zeta_mean),
            _utils.norm(self.beta_mean)
        ]

    @property
    def traces(self):
        """TODO."""
        return self._traces

    @property
    def estimates(self):
        """TODO."""
        return {
            'tau': self.tau_mean, 'tau_var': self.tau_var,
            'zeta': self.zeta_mean, 'zeta_var': self.zeta_var,
            'beta': self.beta_mean, 'beta_var': self.beta_var
        }


def _update_beta(GTG, GTY, tau_mean, zeta_mean):
    # sample beta
    A = tau_mean * (GTG + _nmp.diag(zeta_mean))
    b = tau_mean * GTY
    beta_mean = _utils.chol_solve(A, b)
    beta_var = _nmp.diag(_nmp.linalg.inv(A))

    ##
    return beta_mean, beta_var


def _update_tau(Y, G, beta_mean, tau_mean, zeta_mean):
    n_samples, n_markers = G.shape

    # sample tau
    resid = Y - G.dot(beta_mean)
    shape = 0.5 * (n_markers + n_samples)
    rate = 0.5 * _nmp.sum(resid ** 2) + 0.5 * _nmp.sum(beta_mean ** 2 * zeta_mean) + 0.5 * n_markers / tau_mean
    tau_mean = shape / rate
    tau_var = shape / rate**2

    ##
    return tau_mean, tau_var


def _update_zeta(beta_mean, beta_var, tau_mean):
    # sample tau_beta
    shape = 0.5
    rate = 0.5 * tau_mean * (beta_mean**2 + beta_var)
    zeta_mean = shape / rate
    zeta_var = shape / rate**2

    ##
    return zeta_mean, zeta_var
