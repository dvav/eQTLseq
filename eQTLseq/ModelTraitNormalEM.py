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
        self.Y = Y - _nmp.mean(Y)
        self.G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0)

        # used later in calculations
        self.GTG = self.G.T.dot(self.G)
        self.GTY = self.G.T.dot(self.Y)

        # number of samples and genetic markers
        n_samples, n_markers = self.G.shape

        # initial conditions
        self.tau = _rnd.rand()
        self.zeta = _rnd.rand(n_markers)
        self.beta = _rnd.normal(0, _nmp.ones(n_markers))

        self._traces = _nmp.empty(n_iters + 1)
        self._traces.fill(_nmp.nan)
        self._traces[0] = 0

        # other parameters
        self.s2_min = s2_lims[0]
        self.s2_max = s2_lims[1]

    def update(self, itr):
        """TODO."""
        # E step
        self.zeta = _update_zeta(self.beta, self.tau)
        self.zeta = _nmp.clip(self.zeta, 1 / self.s2_max, 1 / self.s2_min)

        # M step
        self.beta, self.tau = _update_beta_tau(self.Y, self.G, self.GTG, self.GTY, self.zeta)

        # update the rest
        self._traces[itr] = _calculate_lower_bound(self.Y, self.G, self.beta, self.tau, self.zeta)

    @property
    def traces(self):
        """TODO."""
        return self._traces

    @property
    def estimates(self):
        """TODO."""
        return {'tau': self.tau, 'zeta': self.zeta, 'beta': self.beta}


def _update_beta_tau(Y, G, GTG, GTY, zeta):
    n_samples, n_markers = G.shape

    # calculate beta
    A = GTG + _nmp.diag(zeta)
    beta = _utils.chol_solve(A, GTY)

    # calculate tau
    resid = Y - G.dot(beta)
    shape = 0.5 * (n_markers + n_samples)
    rate = 0.5 * _nmp.sum(resid ** 2) + 0.5 * _nmp.sum(beta ** 2 * zeta)
    tau = (shape - 1) / rate

    ##
    return beta, tau


def _update_zeta(beta, tau):
    # sample tau_beta
    shape = 0.5
    rate = 0.5 * tau * beta**2
    zeta = shape / rate

    ##
    return zeta


def _calculate_lower_bound(Y, G, beta, tau, zeta):
    # number of samples and markers
    n_samples, n_markers = G.shape

    #
    resid1 = Y - G.dot(beta)
    resid1 = (resid1**2).sum()
    resid2 = (beta**2 * zeta).sum()
    energy = (0.5 * n_samples + 0.5 * n_markers - 1) * _nmp.log(tau) - 0.5 * tau * (resid1 + resid2)
    entropy = 0.5 * _nmp.log(zeta).sum() + 0.5 * tau * resid2

    #
    return energy + entropy
