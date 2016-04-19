"""Implements ModelNormalStdGibbs."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.utils as _utils


class ModelNormalStdGibbs(object):
    """A normal model estimated using Gibbs sampling."""

    def __init__(self, **args):
        """TODO."""
        n_genes, n_markers = args['n_genes'], args['n_markers']

        # initial conditions
        self.eta = _nmp.ones(n_markers)
        self.zeta = _nmp.ones((n_genes, n_markers))
        self.beta = _rnd.randn(n_genes, n_markers)

        self.zeta_sum, self.zeta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))
        self.eta_sum, self.eta2_sum = _nmp.zeros(n_markers), _nmp.zeros(n_markers)
        self.beta_sum, self.beta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))

    def update(self, itr, **args):
        """TODO."""
        GTG, GTY = args['GTG'], args['GTY']
        beta_thr, s2_lims, n_samples, n_genes = args['beta_thr'], args['s2_lims'], args['n_samples'], args['n_genes']
        parallel = args['parallel']
        phi = args['phi'] if 'phi' in args else _nmp.ones(n_genes)

        # identify irrelevant genes and markers and exclude them
        idxs = (_nmp.abs(self.beta) > beta_thr) & (self.zeta * self.eta / phi[:, None] < 1 / s2_lims[0])
        idxs[[0, 1], [0, 1]] = True  # just a precaution
        idxs_markers = _nmp.any(idxs, 0)
        idxs_genes = _nmp.any(idxs, 1)

        GTG = GTG[:, idxs_markers][idxs_markers, :]
        GTY = GTY[idxs_markers, :][:, idxs_genes]

        zeta = self.zeta[idxs_genes, :][:, idxs_markers]
        eta = self.eta[idxs_markers]

        # sample beta and tau
        beta = _sample_beta(GTG, GTY, zeta, eta, phi[idxs_genes], n_samples, s2_lims, parallel)
        self.beta[_nmp.ix_(idxs_genes, idxs_markers)] = beta

        # sample eta and zeta
        self.zeta = _sample_zeta(self.beta, self.eta, phi)
        self.zeta = _nmp.clip(self.zeta, 1 / s2_lims[1], 1 / s2_lims[0])

        self.eta = _sample_eta(self.beta, self.zeta, phi)
        self.eta = _nmp.clip(self.eta, 1 / s2_lims[1], 1 / s2_lims[0])

        if(itr > args['n_burnin']):
            self.zeta_sum += self.zeta
            self.eta_sum += self.eta
            self.beta_sum += self.beta

            self.zeta2_sum += self.zeta**2
            self.eta2_sum += self.eta**2
            self.beta2_sum += self.beta**2

    def get_estimates(self, **args):
        """TODO."""
        N = args['n_iters'] - args['n_burnin']
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


def _sample_beta(GTG, GTY, zeta, eta, phi, n_samples, s2_lims, parallel):
    """TODO."""
    _, n_markers = zeta.shape

    # sample beta
    theta = 1 / phi
    A = theta[:, None, None] * (GTG + zeta[:, :, None] * _nmp.diag(eta))
    b = theta * GTY
    beta = _utils.sample_multivariate_normal_many(b.T, A, parallel)

    ##
    return beta


def _sample_zeta(beta, eta, phi):
    """TODO."""
    # sample zeta
    theta = 1 / phi
    shape = 0.5
    rate = 0.5 * eta * beta**2 * theta[:, None]
    zeta = shape / rate  # _rnd.gamma(shape, 1 / rate)

    ##
    return zeta


def _sample_eta(beta, zeta, phi):
    """TODO."""
    n_genes, _ = zeta.shape

    # sample zeta
    theta = 1 / phi
    shape = 0.5 * n_genes
    rate = 0.5 * (zeta * beta**2 * theta[:, None]).sum(0)
    eta = _rnd.gamma(shape, 1 / rate)

    ##
    return eta
