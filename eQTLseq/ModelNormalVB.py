"""Implements ModelNormalEM."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.mdl_common as _common
import eQTLseq.utils as _utils


class ModelNormalVB(object):
    """A normal model estimated using Variational Bayes."""

    def __init__(self, **args):
        """TODO."""
        Y, G, n_iters, s2_lims = args['Y'], args['G'], args['n_iters'], args['s2_lims']

        # standardize data
        self.Y = Y - _nmp.mean(Y, 0)
        self.G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0)

        # used later in calculations
        self.YTY = _nmp.sum(self.Y**2, 0)
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

        self._traces = _nmp.empty(n_iters + 1)
        self._traces.fill(_nmp.nan)
        self._traces[0] = 0

        # other parameters
        self.s2_min = s2_lims[0]
        self.s2_max = s2_lims[1]

    def update(self, itr):
        """TODO."""
        # update beta, eta, zeta
        self.beta, self.tau = _common.update_beta_tau(self.YTY, self.GTG, self.GTY, self.zeta, *self.G.shape)
        self.zeta = _common.update_zeta(self.beta, self.tau)
        self.zeta = _nmp.clip(self.zeta, 1 / self.s2_max, 1 / self.s2_min)

        # update the rest
        self._traces[itr] = _calculate_joint_log_likelihood(self.Y, self.G, self.beta, self.tau, self.zeta)

    @property
    def traces(self):
        """TODO."""
        return self._traces

    @property
    def estimates(self):
        """TODO."""
        n_samples, n_markers = self.G.shape

        # tau_var and beta_var
        shape = 0.5 * (n_samples + n_markers)
        rate = 0.5 * self.YTY
        tau_var = shape / rate**2

        # A = self.GTG + _nmp.diag(self.zeta)
        # r = rate / (shape - 1)
        # beta_cov = _utils.chol_solve(A,  _nmp.diag(r * _nmp.ones(n_markers)))

        # zeta_var
        shape = 0.5
        rate = 0.5 * self.beta**2 * self.tau[:, None]
        zeta_var = shape / rate**2

        return {
            'tau': self.tau, 'tau_var': tau_var,
            'zeta': self.zeta, 'zeta_var': zeta_var,
            'beta': self.beta, 'beta_var': _nmp.nan
        }


def _calculate_joint_log_likelihood(Y, G, beta, tau, zeta):    # not ready yet!!!!
    # number of samples and markers
    n_samples, n_markers = G.shape

    #
    resid = Y - G.dot(beta.T)
    A = (0.5 * n_samples + 0.5 * n_markers - 1) * _nmp.log(tau).sum()
    B = 0.5 * (tau * resid**2).sum()
    C = 0.5 * (tau[:, None] * beta**2 * zeta).sum()
    D = 0.5 * _nmp.log(zeta).sum()

    loglik = A - B - C - D

    #
    return loglik
