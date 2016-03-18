"""Implements ModelNormalGibbs."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.mdl_common as _common


class ModelNormalGibbs(object):
    """A normal model estimated using Gibbs sampling."""

    def __init__(self, **args):
        """TODO."""
        n_iters, n_genes, n_markers = args['n_iters'], args['n_genes'], args['n_markers']

        # initial conditions
        self.tau = _nmp.ones(n_genes)
        self.zeta = _nmp.ones((n_genes, n_markers))
        self.beta = _rnd.randn(n_genes, n_markers)

        self._trace = _nmp.empty(n_iters + 1)
        self._trace.fill(_nmp.nan)
        self._trace[0] = 0

        self.tau_sum, self.tau2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)
        self.zeta_sum, self.zeta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))
        self.beta_sum, self.beta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))

    def update(self, itr, **args):
        """TODO."""
        Y, G, GTG, GTY, s2_lims, n_burnin = args['Y'], args['G'], args['GTG'], args['GTY'], args['s2_lims'], \
            args['n_burnin']

        # sample beta, tau and zeta
        self.beta = _common.sample_beta(GTG, GTY, self.tau, self.zeta)
        self.tau = _common.sample_tau(Y, G, self.beta, self.zeta)
        self.zeta = _common.sample_zeta(self.beta, self.tau)
        self.zeta = _nmp.clip(self.zeta, 1 / s2_lims[1], 1 / s2_lims[0])

        # update the rest
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

        #
        N = n_iters - n_burnin
        tau_mean, zeta_mean, beta_mean = self.tau_sum / N, self.zeta_sum / N, self.beta_sum / N
        tau_var, zeta_var, beta_var = self.tau2_sum / N - tau_mean**2, self.zeta2_sum / N - zeta_mean**2, \
            self.beta2_sum / N - beta_mean**2

        return {
            'tau': tau_mean, 'tau_var': tau_var,
            'zeta': zeta_mean, 'zeta_var': zeta_var,
            'beta': beta_mean, 'beta_var': beta_var
        }


def _calculate_joint_log_likelihood(Y, G, beta, tau, zeta):
    # number of samples and markers
    n_samples, n_markers = G.shape

    #
    resid = Y - G.dot(beta.T)

    A = (0.5 * n_samples + 0.5 * n_markers - 1) * _nmp.log(tau).sum()
    B = 0.5 * (tau * resid**2).sum()
    C = 0.5 * (tau[:, None] * beta**2 * zeta).sum()
    D = 0.5 * _nmp.log(zeta).sum()

    #
    return A - B - C - D
