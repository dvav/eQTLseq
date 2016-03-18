"""Implements ModelTraitNormalGibbs."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.utils as _utils


class ModelTraitNormalGibbs(object):
    """A normal model of Bayesian variable selection through shrinkage for a single trait estimated using Gibbs."""

    def __init__(self, **args):
        """TODO."""
        n_markers, n_iters = args['n_markers'], args['n_iters']

        # initial conditions
        self.tau = _rnd.rand()
        self.zeta = _rnd.rand(n_markers)
        self.beta = _rnd.normal(0, 1, size=n_markers)

        self._trace = _nmp.empty(n_iters + 1)
        self._trace.fill(_nmp.nan)
        self._trace[0] = 0

        self.tau_sum, self.tau2_sum = 0, 0
        self.zeta_sum, self.zeta2_sum = _nmp.zeros(n_markers), _nmp.zeros(n_markers)
        self.beta_sum, self.beta2_sum = _nmp.zeros(n_markers), _nmp.zeros(n_markers)

    def update(self, itr, **args):
        """TODO."""
        Y, G, YTY, GTG, GTY, s2_lims, n_burnin, n_samples, n_markers = args['Y'], args['G'], args['YTY'], args['GTG'], \
            args['GTY'], args['s2_lims'], args['n_burnin'], args['n_samples'], args['n_markers']

        # sample beta, tau and zeta
        self.beta, self.tau = _sample_beta_tau(YTY, GTG, GTY, self.zeta, n_samples, n_markers)
        self.zeta = _sample_zeta(self.beta, self.tau)
        self.zeta = _nmp.clip(self.zeta, 1 / s2_lims[1], 1 / s2_lims[0])

        self._trace[itr] = _calculate_joint_log_likelihood(Y, G, self.beta, self.tau, self.zeta)

        if(itr > n_burnin):
            self.tau_sum += self.tau
            self.zeta_sum += self.zeta
            self.beta_sum += self.beta

            self.tau2_sum += self.tau**2
            self.zeta2_sum += self.zeta**2
            self.beta2_sum += self.beta**2

    @property
    def trace(self):
        """TODO."""
        return self._trace

    def get_estimates(self, **args):
        """TODO."""
        n_iters, n_burnin = args['n_iters'], args['n_burnin']

        N = n_iters - n_burnin
        tau_mean, zeta_mean, beta_mean = self.tau_sum / N, self.zeta_sum / N, self.beta_sum / N
        tau_var, zeta_var, beta_var = self.tau2_sum / N - tau_mean**2, self.zeta2_sum / N - zeta_mean**2, \
            self.beta2_sum / N - beta_mean**2

        return {
            'tau': tau_mean, 'tau_var': tau_var,
            'zeta': zeta_mean, 'zeta_var': zeta_var,
            'beta': beta_mean, 'beta_var': beta_var
        }


def _sample_beta_tau(YTY, GTG, GTY, zeta, n_samples, n_markers):
    # sample tau
    shape = 0.5 * (n_markers + n_samples)
    rate = 0.5 * YTY
    tau = _rnd.gamma(shape, 1 / rate)

    # sample beta
    A = tau * (GTG + _nmp.diag(zeta))
    b = tau * GTY
    beta = _utils.sample_multivariate_normal(b, A)

    ##
    return beta, tau


def _sample_zeta(beta, tau):
    # sample tau_beta
    shape = 0.5
    rate = 0.5 * beta**2 * tau
    zeta = shape / rate  # _rnd.gamma(shape, 1 / rate)

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
