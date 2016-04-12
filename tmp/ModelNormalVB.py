"""Implements ModelNormalGibbs."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.utils as _utils


class ModelNormalVB(object):
    """A normal model estimated using variational Bayes."""

    def __init__(self, **args):
        """TODO."""
        n_genes, n_markers = args['n_genes'], args['n_markers']

        # initial conditions
        self.tau, self.tau_var = _nmp.ones(n_genes), _nmp.ones(n_genes)
        self.eta, self.eta_var = _nmp.ones(n_markers), _nmp.ones(n_markers)
        self.zeta, self.zeta_var = _nmp.ones((n_genes, n_markers)), _nmp.ones((n_genes, n_markers))
        self.beta, self.beta_var = _rnd.randn(n_genes, n_markers), _rnd.randn(n_genes, n_markers)

        self.idxs_markers = _nmp.ones(n_markers, dtype='bool')
        self.idxs_genes = _nmp.ones(n_genes, dtype='bool')

    def update(self, itr, **args):
        """TODO."""
        YTY, GTG, GTY = args['YTY'], args['GTG'], args['GTY']
        beta_thr, s2_lims = args['beta_thr'], args['s2_lims']

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

        # sample beta and tau
        beta, beta_var, tau, tau_var = _update_beta_tau(YTY, GTG, GTY, zeta, eta, args['n_samples'], s2_lims)

        self.beta[_nmp.ix_(self.idxs_genes, self.idxs_markers)] = beta
        self.beta_var[_nmp.ix_(self.idxs_genes, self.idxs_markers)] = beta_var

        self.tau[self.idxs_genes] = tau
        self.tau_var[self.idxs_genes] = tau_var

        # sample eta and zeta
        self.zeta, self.zeta_var = _update_zeta(self.beta, self.beta_var, self.tau, self.eta)
        self.zeta = _nmp.clip(self.zeta, 1 / s2_lims[1], 1 / s2_lims[0])

        self.eta, self.eta_var = _update_eta(self.beta, self.beta_var, self.tau, self.zeta)
        self.eta = _nmp.clip(self.eta, 1 / s2_lims[1], 1 / s2_lims[0])

    def get_estimates(self, **args):
        """TODO."""
        return {
            'tau': self.tau, 'tau_var': self.tau_var,
            'zeta': self.zeta, 'zeta_var': self.zeta_var,
            'eta': self.eta, 'eta_var': self.eta_var,
            'beta': self.beta, 'beta_var': self.beta_var
        }

    def get_state(self, **args):
        """TODO."""
        return _nmp.sqrt((self.beta**2).sum())


def _update_beta_tau(YTY, GTG, GTY, zeta, eta, n_samples, s2_lims):
    """TODO."""
    _, n_markers = zeta.shape

    # update tau
    shape = 0.5 * (n_samples + n_markers)
    rate = 0.5 * YTY
    tau = shape / rate
    tau_var = shape / rate**2
    tau = _nmp.clip(tau, 1 / s2_lims[1], 1 / s2_lims[0])

    # sample beta
    A = GTG + zeta[:, :, None] * _nmp.diag(eta)
    beta = _utils.solve_chol_many(A, GTY.T)
    beta_var = rate[:, None] / (shape - 1) / _nmp.diagonal(A, axis1=1, axis2=2)   # ??????????

    # S = _nmp.linalg.inv(A)
    # beta = (S * GTY.T[:, :, None]).sum(1)
    # beta_var = rate[:, None] / (shape - 1) * _nmp.diagonal(S, axis1=1, axis2=2)

    ##
    return beta, beta_var, tau, tau_var


def _update_zeta(beta, beta_var, tau, eta):
    """TODO."""
    # sample zeta
    shape = 0.5
    rate = 0.5 * eta * tau[:, None] * (beta**2 + beta_var)
    zeta = shape / rate
    zeta_var = shape / rate**2

    ##
    return zeta, zeta_var


def _update_eta(beta, beta_var, tau, zeta):
    """TODO."""
    n_genes, _ = zeta.shape

    # sample zeta
    shape = 0.5 * n_genes
    rate = 0.5 * (zeta * tau[:, None] * (beta**2 + beta_var)).sum(0)
    eta = shape / rate
    eta_var = shape / rate**2

    ##
    return eta, eta_var
