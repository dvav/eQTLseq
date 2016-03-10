"""Implements ModelNormalGibbs."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.mdl_common_gibbs as _common
import eQTLseq.utils as _utils


class ModelNormalGibbs(object):
    """A normal model estimated using Gibbs sampling."""

    def __init__(self, Y, G, n_iters, n_burnin, s2_lims):
        """TODO."""
        # standardize data
        self.Y = (Y - _nmp.mean(Y, 0)) / _nmp.std(Y, 0)
        self.G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0)

        # used later in calculations
        self.GTG = self.G.T.dot(self.G)
        self.GTY = self.G.T.dot(self.Y)

        # number of samples, genes and genetic markers
        n_samples1, n_genes = Y.shape
        n_samples2, n_markers = G.shape

        assert n_samples1 == n_samples2

        # initial conditions
        self.tau = _nmp.ones(n_genes)
        self.zeta = _nmp.ones((n_genes, n_markers))
        self.beta = _rnd.randn(n_genes, n_markers)

        self.traces = _nmp.zeros((n_iters + 1, 3))
        self.traces[0, :] = [
            1 / _utils.norm(self.tau),
            1 / _utils.norm(self.zeta),
            _utils.norm(self.beta)
        ]

        self.tau_sum, self.tau2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)
        self.zeta_sum, self.zeta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))
        self.beta_sum, self.beta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))

        # other parameters
        self.n_iters = n_iters
        self.n_burnin = n_burnin
        self.s2_min = s2_lims[0]
        self.s2_max = s2_lims[1]

    def update(self, itr):
        """TODO."""
        # sample beta, tau and zeta
        self.beta = _common.sample_beta(self.GTG, self.GTY, self.tau, self.zeta)
        self.tau = _common.sample_tau(self.Y, self.G, self.beta, self.zeta)
        self.zeta = _common.sample_zeta(self.beta, self.tau)

        self.zeta = _nmp.clip(self.zeta, 1 / self.s2_max, 1 / self.s2_min)

        # update the rest
        self.traces[itr, :] = [
            1 / _utils.norm(self.tau),
            1 / _utils.norm(self.zeta),
            _utils.norm(self.beta)
        ]

        if(itr > self.n_burnin):
            self.tau_sum += self.tau
            self.zeta_sum += self.zeta
            self.beta_sum += self.beta

            self.tau2_sum += self.tau**2
            self.zeta2_sum += self.zeta**2
            self.beta2_sum += self.beta**2

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
