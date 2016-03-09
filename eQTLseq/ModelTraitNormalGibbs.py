"""Implements ModelTraitNormalGibbs."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.utils as _utils


class ModelTraitNormalGibbs(object):
    """A normal model of Bayesian variable selection through shrinkage for a single trait estimated using Gibbs."""

    def __init__(self, Y, G, n_iters, n_burnin, s2_lims):
        """TODO."""
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

        self.traces = _nmp.zeros((n_iters + 1, 3))
        self.traces[0, :] = [
            1 / self.tau,
            1 / _nmp.sqrt(_nmp.sum(self.zeta**2)),
            _nmp.sqrt(_nmp.sum(self.beta**2))
        ]
        self.tau_sum, self.tau2_sum = 0, 0
        self.zeta_sum, self.zeta2_sum = _nmp.zeros(n_markers), _nmp.zeros(n_markers)
        self.beta_sum, self.beta2_sum = _nmp.zeros(n_markers), _nmp.zeros(n_markers)

        # other parameters
        self.n_iters = n_iters
        self.n_burnin = n_burnin
        self.s2_min = s2_lims[0]
        self.s2_max = s2_lims[1]

    def update(self, itr):
        """TODO."""
        # sample beta, tau and zeta
        self.beta = _sample_beta(self.GTG, self.GTY, self.tau, self.zeta)
        self.tau = _sample_tau(self.Y, self.G, self.beta, self.zeta)
        self.zeta = _sample_zeta(self.beta, self.tau)

        self.zeta = _nmp.clip(self.zeta, 1 / self.s2_max, 1 / self.s2_min)

        # update the rest
        beta2, zeta2 = self.beta**2, self.zeta**2
        self.traces[itr, :] = [
            1 / self.tau,
            1 / _nmp.sqrt(_nmp.sum(zeta2)),
            _nmp.sqrt(_nmp.sum(beta2))
        ]

        if(itr > self.n_burnin):
            self.tau_sum += self.tau
            self.zeta_sum += self.zeta
            self.beta_sum += self.beta

            self.tau2_sum += self.tau**2
            self.zeta2_sum += zeta2
            self.beta2_sum += beta2

    def stats(self):
        """TODO."""
        N = self.n_iters - self.n_burnin
        tau_mean, zeta_mean, beta_mean = self.tau_sum / N, self.zeta_sum / N, self.beta_sum / N
        tau_var, zeta_var, beta_var = self.tau2_sum / N - tau_mean**2, self.zeta2_sum / N - zeta_mean**2, \
            self.beta2_sum / N - beta_mean**2

        return {'traces': self.traces,
                'tau_mean': tau_mean, 'tau_var': tau_var,
                'zeta_mean': zeta_mean, 'zeta_var': zeta_var,
                'beta_mean': beta_mean, 'beta_var': beta_var}


def _sample_beta(GTG, GTY, tau, zeta):
    # sample beta
    A = tau * (GTG + _nmp.diag(zeta))
    b = tau * GTY
    beta = _utils.sample_multivariate_normal(b, A)

    ##
    return beta


def _sample_tau(Y, G, beta, zeta):
    n_samples, n_markers = G.shape

    # sample tau
    resid = Y - G.dot(beta)
    shape = 0.5 * (n_markers + n_samples)
    rate = 0.5 * _nmp.sum(resid ** 2) + 0.5 * _nmp.sum(beta ** 2 * zeta)
    tau = _rnd.gamma(shape, 1 / rate)

    ##
    return tau


def _sample_zeta(beta, tau):
    # sample tau_beta
    shape = 0.5
    rate = 0.5 * beta**2 * tau
    zeta = shape / rate  # _rnd.gamma(shape, 1 / rate)

    ##
    return zeta
