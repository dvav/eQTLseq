"""Implements ModelNormalEM."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.mdl_common as _common
import eQTLseq.utils as _utils


class ModelNormalVB(object):
    """A normal model estimated using Variational Bayes."""

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

    def update(self, itr, **args):
        """TODO."""
        Y, G, YTY, GTG, GTY, s2_lims = args['Y'], args['G'], args['YTY'], args['GTG'], args['GTY'], args['s2_lims']

        # update beta, eta, zeta
        self.beta, self.tau = _common.update_beta_tau(YTY, GTG, GTY, self.zeta, *G.shape)
        self.zeta = _common.update_zeta(self.beta, self.tau)
        self.zeta = _nmp.clip(self.zeta, 1 / s2_lims[1], 1 / s2_lims[0])

        # update the rest
        self._trace[itr] = _calculate_joint_log_likelihood(Y, G, self.beta, self.tau, self.zeta)

    @property
    def trace(self):
        """TODO."""
        return self._trace

    def get_estimates(self, **args):
        """TODO."""
        n_samples, n_markers, YTY = args['n_samples'], args['n_markers'], args['YTY']

        # tau_var and beta_var
        shape = 0.5 * (n_samples + n_markers)
        rate = 0.5 * YTY
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
