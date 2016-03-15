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
        self.tau, self.tau_var = _nmp.ones(n_genes), _nmp.ones(n_genes)
        self.zeta, self.zeta_var = _nmp.ones((n_genes, n_markers)), _nmp.ones((n_genes, n_markers))
        self.beta, self.beta_var = _rnd.randn(n_genes, n_markers), _nmp.ones((n_genes, n_markers))

        self.traces = _nmp.empty((n_iters + 1, 3))
        self.traces.fill(_nmp.nan)
        self.traces[0, :] = [
            1 / _utils.norm(self.tau),
            1 / _utils.norm(self.zeta),
            _utils.norm(self.beta)
        ]

        # other parameters
        self.s2_min = s2_lims[0]
        self.s2_max = s2_lims[1]

    def update(self, itr):
        """TODO."""
        # update beta, eta, zeta
        self.beta, self.beta_var = _common.update_beta(self.GTG, self.GTY, self.tau, self.zeta)
        self.tau, self.tau_var = _common.update_tau(self.Y, self.G, self.beta, self.tau, self.zeta)
        self.zeta, self.zeta_var = _common.update_zeta(self.beta, self.beta_var, self.tau)
        self.zeta = _nmp.clip(self.zeta, 1 / self.s2_max, 1 / self.s2_min)

        # update the rest
        self.traces[itr, :] = [
            1 / _utils.norm(self.tau),
            1 / _utils.norm(self.zeta),
            _utils.norm(self.beta)
        ]

    def stats(self):
        """TODO."""
        return {
            'traces': self.traces,
            'tau': self.tau, 'tau_var': self.tau_var,
            'zeta': self.zeta, 'zeta_var': self.zeta_var,
            'beta': self.beta, 'beta_var': self.beta_var
        }
