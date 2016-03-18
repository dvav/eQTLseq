"""Implements ModelTraitNormalEM."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.utils as _utils


class ModelTraitNormalEM(object):
    """A normal model of Bayesian variable selection through shrinkage for a single trait estimated using EM."""

    def __init__(self, **args):
        """TODO."""
        n_markers, n_iters = args['n_markers'], args['n_iters']

        # initial conditions
        self.tau = _rnd.rand()
        self.zeta = _rnd.rand(n_markers)
        self.beta = _rnd.normal(0, _nmp.ones(n_markers))

        self._trace = _nmp.empty(n_iters + 1)
        self._trace.fill(_nmp.nan)
        self._trace[0] = 0

    def update(self, itr, **args):
        """TODO."""
        Y, G, GTG, GTY, s2_lims = args['Y'], args['G'], args['GTG'], args['GTY'], args['s2_lims']

        # E step
        self.zeta = _update_zeta(self.beta, self.tau)
        self.zeta = _nmp.clip(self.zeta, 1 / s2_lims[1], 1 / s2_lims[0])

        # M step
        self.beta, self.tau = _update_beta_tau(Y, G, GTG, GTY, self.zeta)

        # update the rest
        self._trace[itr] = _calculate_lower_bound(Y, G, self.beta, self.tau, self.zeta)

    @property
    def trace(self):
        """TODO."""
        return self._trace

    def get_estimates(self, **args):
        """TODO."""
        return {'tau': self.tau, 'zeta': self.zeta, 'beta': self.beta}


def _update_beta_tau(Y, G, GTG, GTY, zeta):
    n_samples, n_markers = G.shape

    # calculate beta
    A = GTG + _nmp.diag(zeta)
    beta = _utils.chol_solve(A, GTY)

    # calculate tau
    resid = Y - (G * beta).sum(1)
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
    resid1 = Y - (G * beta).sum(1)
    resid1 = (resid1**2).sum()
    resid2 = (beta**2 * zeta).sum()
    energy = (0.5 * n_samples + 0.5 * n_markers - 1) * _nmp.log(tau) - 0.5 * tau * (resid1 + resid2)
    entropy = 0.5 * _nmp.log(zeta).sum() + 0.5 * tau * resid2

    #
    return energy + entropy
