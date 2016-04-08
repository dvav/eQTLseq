"""Implements ModelNormalStdGibbs."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.utils as _utils


class ModelNormalStdGibbs(object):
    """A normal model estimated using Gibbs sampling (assuming standardised input data)."""

    def __init__(self, **args):
        """TODO."""
        n_genes, n_markers = args['n_genes'], args['n_markers']

        # initial conditions
        self.eta = _nmp.ones(n_markers)
        self.zeta = _nmp.ones((n_genes, n_markers))
        self.beta = _rnd.randn(n_genes, n_markers)

        self.idxs_markers = _nmp.ones(n_markers, dtype='bool')
        self.idxs_genes = _nmp.ones(n_genes, dtype='bool')

        self.zeta_sum, self.zeta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))
        self.eta_sum, self.eta2_sum = _nmp.zeros(n_markers), _nmp.zeros(n_markers)
        self.beta_sum, self.beta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))

    def update(self, itr, **args):
        """TODO."""
        GTG, GTY = args['GTG'], args['GTY']
        n_burnin, beta_thr, s2_lims = args['n_burnin'], args['beta_thr'], args['s2_lims']

        # identify irrelevant genes and markers and exclude them
        idxs = (_nmp.abs(self.beta) > beta_thr) & (self.zeta * self.eta < 1 / s2_lims[0])
        idxs[[0, 1], [0, 1]] = True  # just a precaution
        self.idxs_markers = _nmp.any(idxs, 0)
        self.idxs_genes = _nmp.any(idxs, 1)

        GTG = GTG[:, self.idxs_markers][self.idxs_markers, :]
        GTY = GTY[self.idxs_markers, :][:, self.idxs_genes]

        zeta = self.zeta[self.idxs_genes, :][:, self.idxs_markers]
        eta = self.eta[self.idxs_markers]

        # sample beta, tau and zeta
        beta = _sample_beta(GTG, GTY, zeta, eta)

        zeta = _sample_zeta(beta, eta)
        zeta = _nmp.clip(zeta, 1 / s2_lims[1], 1 / s2_lims[0])

        eta = _sample_eta(beta, zeta)
        eta = _nmp.clip(eta, 1 / s2_lims[1], 1 / s2_lims[0])

        self.beta[_nmp.ix_(self.idxs_genes, self.idxs_markers)] = beta
        self.zeta[_nmp.ix_(self.idxs_genes, self.idxs_markers)] = zeta
        self.eta[self.idxs_markers] = eta

        if(itr > n_burnin):
            self.zeta_sum += self.zeta
            self.eta_sum += self.eta
            self.beta_sum += self.beta

            self.zeta2_sum += self.zeta**2
            self.eta2_sum += self.eta**2
            self.beta2_sum += self.beta**2

    def get_estimates(self, **args):
        """TODO."""
        n_iters, n_burnin = args['n_iters'], args['n_burnin']

        #
        N = n_iters - n_burnin
        zeta_mean, eta_mean, beta_mean = self.zeta_sum / N, self.eta_sum / N, self.beta_sum / N
        zeta_var, eta_var, beta_var = self.zeta2_sum / N - zeta_mean**2, self.eta2_sum / N - eta_mean**2, \
            self.beta2_sum / N - beta_mean**2

        return {
            'zeta': zeta_mean, 'zeta_var': zeta_var,
            'eta': eta_mean, 'eta_var': eta_var,
            'beta': beta_mean, 'beta_var': beta_var
        }

    def get_state(self, **args):
        """TODO."""
        return _nmp.sqrt((self.beta**2).sum())


def _sample_beta(GTG, GTY, zeta, eta):
    """TODO."""
    # sample beta
    A = GTG + zeta[:, :, None] * _nmp.diag(eta)
    beta = _utils.sample_multivariate_normal_many(GTY.T, A)

    ##
    return beta


def _sample_zeta(beta, eta):
    """TODO."""
    # sample zeta
    shape = 0.5
    rate = 0.5 * eta * beta**2
    zeta = shape / rate  # _rnd.gamma(shape, 1 / rate)

    ##
    return zeta


def _sample_eta(beta, zeta):
    """TODO."""
    n_genes, _ = zeta.shape

    # sample zeta
    shape = 0.5 * n_genes
    rate = 0.5 * (zeta * beta**2).sum(0)
    eta = _rnd.gamma(shape, 1 / rate)

    ##
    return eta
