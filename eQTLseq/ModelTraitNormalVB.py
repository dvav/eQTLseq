"""Implements ModelTraitNormalVB."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.utils as _utils


class ModelTraitNormalVB(object):
    """A normal model of Bayesian variable selection through shrinkage for a single trait estimated using VB."""

    def __init__(self, **args):
        """TODO."""
        n_markers, n_iters = args['n_markers'], args['n_iters']

        # initial conditions
        self.tau = _rnd.rand()
        self.zeta = _rnd.rand(n_markers)
        self.beta = _rnd.normal(0, _nmp.ones(n_markers))
        self.beta_var = _rnd.normal(0, _nmp.ones(n_markers))

        self._trace = _nmp.empty(n_iters + 1)
        self._trace.fill(_nmp.nan)
        self._trace[0] = 0

    def update(self, itr, **args):
        """TODO."""
        Y, G, YTY, GTG, GTY, n_samples, n_markers, s2_lims = args['Y'], args['G'], args['YTY'], args['GTG'], \
            args['GTY'], args['n_samples'], args['n_markers'], args['s2_lims']

        # update beta, tau and zeta
        self.beta, self.tau = _update_beta_tau(YTY, GTG, GTY, self.zeta, n_samples, n_markers)
        self.zeta = _update_zeta(self.beta, self.tau)
        self.zeta = _nmp.clip(self.zeta, 1 / s2_lims[1], 1 / s2_lims[0])

        # update the rest
        self._trace[itr] = _calculate_joint_log_likelihood(Y, G, self.beta, self.tau, self.zeta)

    @property
    def trace(self):
        """TODO."""
        return self._trace

    def get_estimates(self, **args):
        """TODO."""
        n_samples, n_markers, YTY, GTG = args['n_samples'], args['n_markers'], args['YTY'], args['GTG']

        # tau_var and beta_var
        shape = 0.5 * (n_samples + n_markers)
        rate = 0.5 * YTY
        tau_var = shape / rate**2

        A = GTG + _nmp.diag(self.zeta)
        r = rate / (shape - 1)
        beta_cov = _utils.chol_solve(A,  _nmp.diag(r * _nmp.ones(n_markers)))

        # zeta_var
        shape = 0.5
        rate = 0.5 * self.tau * self.beta**2
        zeta_var = shape / rate**2

        return {
            'tau': self.tau, 'tau_var': tau_var,
            'zeta': self.zeta, 'zeta_var': zeta_var,
            'beta': self.beta, 'beta_var': _nmp.diag(beta_cov)
        }


def _update_beta_tau(YTY, GTG, GTY, zeta, n_samples, n_markers):
    shape = 0.5 * (n_markers + n_samples)
    rate = 0.5 * YTY
    tau = shape / rate

    # sample beta
    A = GTG + _nmp.diag(zeta)
    beta = _utils.chol_solve(A, GTY)

    ##
    return beta, tau


def _update_zeta(beta, tau):
    # sample tau_beta
    shape = 0.5
    # rate = 0.5 * tau * (beta**2 + beta_var)
    rate = 0.5 * tau * beta**2
    zeta = shape / rate

    ##
    return zeta


def _calculate_joint_log_likelihood(Y, G, beta, tau, zeta):
    # number of samples and markers
    n_samples, n_markers = G.shape

    #
    resid = Y - (G * beta).sum(1)

    A = (0.5 * n_samples + 0.5 * n_markers - 1) * _nmp.log(tau)
    B = 0.5 * tau * (resid**2).sum()
    C = 0.5 * tau * (beta**2 * zeta).sum()
    D = 0.5 * _nmp.log(zeta).sum()

    #
    return A - B - C - D
