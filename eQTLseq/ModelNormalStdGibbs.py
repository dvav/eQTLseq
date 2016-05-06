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
        self.tau = _nmp.ones(n_genes)
        self.eta = _nmp.ones(n_markers)
        self.zeta = _nmp.ones((n_genes, n_markers))
        self.beta = _rnd.randn(n_genes, n_markers) / n_markers

        self.zeta_sum, self.zeta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))
        self.tau_sum, self.tau2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)
        self.eta_sum, self.eta2_sum = _nmp.zeros(n_markers), _nmp.zeros(n_markers)
        self.beta_sum, self.beta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))

    def update(self, itr, **args):
        """TODO."""
        GTG, GTY = args['GTG'], args['GTY']
        beta_thr, s2_lims = args['beta_thr'], args['s2_lims']
        parallel = args['parallel']

        # identify irrelevant genes and markers and exclude them
        idxs = (_nmp.abs(self.beta) > beta_thr) & (self.tau[:, None] * self.zeta * self.eta < 1 / args['s2_lims'][0])
        idxs[[0, 1], [0, 1]] = True  # just a precaution
        idxs_markers = _nmp.any(idxs, 0)
        idxs_genes = _nmp.any(idxs, 1)

        GTG = GTG[:, idxs_markers][idxs_markers, :]
        GTY = GTY[idxs_markers, :][:, idxs_genes]

        tau = self.tau[idxs_genes]
        zeta = self.zeta[idxs_genes, :][:, idxs_markers]
        eta = self.eta[idxs_markers]

        # sample beta and tau
        beta = _sample_beta(GTG, GTY, tau, zeta, eta, parallel)
        self.beta[_nmp.ix_(idxs_genes, idxs_markers)] = beta

        # sample eta and zeta
        self.tau = _sample_tau(self.beta, self.zeta, self.eta)
        self.tau = _nmp.clip(self.tau, 1 / s2_lims[1], 1 / s2_lims[0])

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


def _sample_beta(GTG, GTY, tau, zeta, eta, parallel):
    """TODO."""
    A = GTG + tau[:, None, None] * zeta[:, :, None] * _nmp.diag(eta)
    beta = _utils.sample_multivariate_normal_many(GTY.T, A, parallel)

    ##
    return beta


def _sample_tau(beta, zeta, eta):
    """TODO."""
    _, n_markers = zeta.shape

    # sample zeta
    shape = 0.5 * n_markers
    rate = 0.5 * (eta * zeta * beta**2).sum(1)
    tau = _rnd.gamma(shape, 1 / rate)

    ##
    return tau


def _sample_zeta(beta, tau, eta):
    """TODO."""
    # sample zeta
    shape = 0.5
    rate = 0.5 * tau[:, None] * eta * beta**2
    zeta = shape / rate  # _rnd.gamma(shape, 1 / rate)

    ##
    return zeta


def _sample_eta(beta, tau, zeta):
    """TODO."""
    n_genes, _ = zeta.shape

    # sample zeta
    shape = 0.5 * n_genes
    rate = 0.5 * (tau[:, None] * zeta * beta**2).sum(0)
    eta = _rnd.gamma(shape, 1 / rate)

    ##
    return eta
