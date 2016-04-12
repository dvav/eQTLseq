"""Implements ModelNormalGibbs."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.utils as _utils


class ModelNormalEM(object):
    """A normal model estimated using expectation-maximisation."""

    def __init__(self, **args):
        """TODO."""
        n_genes, n_markers = args['n_genes'], args['n_markers']

        # initial conditions
        self.tau = _nmp.ones(n_genes)
        self.eta = _nmp.ones(n_markers)
        self.zeta = _nmp.ones((n_genes, n_markers))
        self.beta = _rnd.randn(n_genes, n_markers)

        self.idxs_markers = _nmp.ones(n_markers, dtype='bool')
        self.idxs_genes = _nmp.ones(n_genes, dtype='bool')

    def update(self, itr, **args):
        """TODO."""
        YTY, GTG, GTY = args['YTY'], args['GTG'], args['GTY']
        beta_thr, s2_lims = args['beta_thr'], args['s2_lims']

        # E step: update zeta
        self.zeta = _update_zeta(self.beta, self.tau, self.eta)
        self.zeta = _nmp.clip(self.zeta, 1 / s2_lims[1], 1 / s2_lims[0])

        # M step: update eta
        self.eta = _update_eta(self.beta, self.tau, self.zeta)
        self.eta = _nmp.clip(self.eta, 1 / s2_lims[1], 1 / s2_lims[0])

        # identify irrelevant genes and markers and exclude them
        idxs = (_nmp.abs(self.beta) > beta_thr) & (self.zeta * self.eta * self.tau[:, None] < 1 / s2_lims[0])
        idxs[[0, 1], [0, 1]] = True  # just a precaution
        self.idxs_markers = _nmp.any(idxs, 0)
        self.idxs_genes = _nmp.any(idxs, 1)

        YTY = YTY[self.idxs_genes]
        GTG = GTG[:, self.idxs_markers][self.idxs_markers, :]
        GTY = GTY[self.idxs_markers, :][:, self.idxs_genes]

        zeta = self.zeta[self.idxs_genes, :][:, self.idxs_markers]
        eta = self.eta[self.idxs_markers]

        # M step: update beta and tau
        beta, tau = _update_beta_tau(YTY, GTG, GTY, zeta, eta, args['n_samples'], s2_lims)

        self.beta[_nmp.ix_(self.idxs_genes, self.idxs_markers)] = beta
        self.tau[self.idxs_genes] = tau

    def get_estimates(self, **args):
        """TODO."""
        return {'tau': self.tau, 'zeta': self.zeta, 'eta': self.eta, 'beta': self.beta}

    def get_state(self, **args):
        """TODO."""
        return _nmp.sqrt((self.beta**2).sum())


def _update_beta_tau(YTY, GTG, GTY, zeta, eta, n_samples, s2_lims):
    """TODO."""
    _, n_markers = zeta.shape

    # update tau
    shape = 0.5 * (n_samples + n_markers)
    rate = 0.5 * YTY
    tau = (shape - 1) / rate
    tau = _nmp.clip(tau, 1 / s2_lims[1], 1 / s2_lims[0])

    # sample beta
    A = GTG + zeta[:, :, None] * _nmp.diag(eta)
    beta = _utils.solve_chol_many(A, GTY.T)

    ##
    return beta, tau


def _update_zeta(beta, tau, eta):
    """TODO."""
    # sample zeta
    shape = 0.5
    rate = 0.5 * eta * tau[:, None] * beta**2
    zeta = shape / rate

    ##
    return zeta


def _update_eta(beta, tau, zeta):
    """TODO."""
    n_genes, _ = zeta.shape

    # sample zeta
    shape = 0.5 * n_genes
    rate = 0.5 * (zeta * tau[:, None] * beta**2).sum(0)
    eta = (shape - 1) / rate

    ##
    return eta
