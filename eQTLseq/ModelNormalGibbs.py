"""Implements ModelNormalGibbs."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.utils as _utils


class ModelNormalGibbs(object):
    """A normal model estimated using Gibbs sampling."""

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

        self.tau_sum, self.tau2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)
        self.zeta_sum, self.zeta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))
        self.eta_sum, self.eta2_sum = _nmp.zeros(n_markers), _nmp.zeros(n_markers)
        self.beta_sum, self.beta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))

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
        beta, tau = _sample_beta_tau(YTY, GTG, GTY, zeta, eta, args['n_samples'], s2_lims, args['parallel'])

        self.beta[_nmp.ix_(self.idxs_genes, self.idxs_markers)] = beta
        self.tau[self.idxs_genes] = tau

        # sample eta and zeta
        self.zeta = _sample_zeta(self.beta, self.tau, self.eta)
        self.zeta = _nmp.clip(self.zeta, 1 / s2_lims[1], 1 / s2_lims[0])

        self.eta = _sample_eta(self.beta, self.tau, self.zeta)
        self.eta = _nmp.clip(self.eta, 1 / s2_lims[1], 1 / s2_lims[0])

        if(itr > args['n_burnin']):
            self.tau_sum += self.tau
            self.zeta_sum += self.zeta
            self.eta_sum += self.eta
            self.beta_sum += self.beta

            self.tau2_sum += self.tau**2
            self.zeta2_sum += self.zeta**2
            self.eta2_sum += self.eta**2
            self.beta2_sum += self.beta**2

    def get_estimates(self, **args):
        """TODO."""
        N = args['n_iters'] - args['n_burnin']
        tau_mean, zeta_mean, eta_mean, beta_mean = self.tau_sum / N, self.zeta_sum / N, self.eta_sum / N, \
            self.beta_sum / N
        tau_var, zeta_var, eta_var, beta_var = self.tau2_sum / N - tau_mean**2, self.zeta2_sum / N - zeta_mean**2, \
            self.eta2_sum / N - eta_mean**2, self.beta2_sum / N - beta_mean**2

        return {
            'tau': tau_mean, 'tau_var': tau_var,
            'zeta': zeta_mean, 'zeta_var': zeta_var,
            'eta': eta_mean, 'eta_var': eta_var,
            'beta': beta_mean, 'beta_var': beta_var
        }

    def get_state(self, **args):
        """TODO."""
        return _nmp.sqrt((self.beta**2).sum())


def _sample_beta_tau(YTY, GTG, GTY, zeta, eta, n_samples, s2_lims, parallel):
    """TODO."""
    _, n_markers = zeta.shape

    # sample tau
    shape = 0.5 * (n_samples + n_markers)
    rate = 0.5 * YTY
    tau = _rnd.gamma(shape, 1 / rate)
    tau = _nmp.clip(tau, 1 / s2_lims[1], 1 / s2_lims[0])

    # sample beta
    A = tau[:, None, None] * (GTG + zeta[:, :, None] * _nmp.diag(eta))
    b = tau * GTY
    beta = _utils.sample_multivariate_normal_many(b.T, A, parallel)

    ##
    return beta, tau


def _sample_zeta(beta, tau, eta):
    """TODO."""
    # sample zeta
    shape = 0.5
    rate = 0.5 * eta * beta**2 * tau[:, None]
    zeta = shape / rate  # _rnd.gamma(shape, 1 / rate)

    ##
    return zeta


def _sample_eta(beta, tau, zeta):
    """TODO."""
    n_genes, _ = zeta.shape

    # sample zeta
    shape = 0.5 * n_genes
    rate = 0.5 * (zeta * beta**2 * tau[:, None]).sum(0)
    eta = _rnd.gamma(shape, 1 / rate)

    ##
    return eta
