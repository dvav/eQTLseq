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

        self.tau_sum, self.tau2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)
        self.zeta_sum, self.zeta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))
        self.eta_sum, self.eta2_sum = _nmp.zeros(n_markers), _nmp.zeros(n_markers)
        self.beta_sum, self.beta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))

    def update(self, itr, **args):
        """TODO."""
        YTY, GTG, GTY = args['YTY'], args['GTG'], args['GTY']
        beta_thr, s2_lims, n_samples = args['beta_thr'], args['s2_lims'], args['n_samples']

        # sample beta and tau
        self.beta, self.tau = _sample_beta_tau(YTY, GTG, GTY, self.beta, self.tau, self.zeta, self.eta, n_samples,
                                               beta_thr, s2_lims)

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

    @staticmethod
    def get_error(Y, G, res):
        """TODO."""
        _, n_genes = Y.shape

        beta = res['beta']

        Yhat = G.dot(beta.T)

        # Y = _nmp.c_[Y, Yhat]
        # Y = _utils.blom(Y.T).T
        # Y, Yhat = Y[:, :n_genes], Y[:, n_genes:]

        ##
        return ((Y - Yhat)**2).sum() / Y.size

    # @staticmethod
    # def get_error(Y, G, res):
    #     """TODO."""
    #     _, n_genes = Y.shape
    #
    #     beta = res['beta']
    #     tau = res['tau']
    #
    #     Yhat = G.dot(beta.T)
    #     s2 = 1 / tau
    #
    #     ##
    #     return ((Y - Yhat)**2 / s2).sum() / Y.size


def _sample_beta_tau_(YTY, GTG, GTY, zeta, eta, n_samples, s2_lims):
    """TODO."""
    _, n_markers = zeta.shape

    # sample tau
    shape = 0.5 * (n_samples + n_markers)
    rate = 0.5 * YTY
    tau = _rnd.gamma(shape, 1 / rate)
    tau = _nmp.clip(tau, 1 / s2_lims[1], 1 / s2_lims[0])

    # sample beta
    theta = zeta * eta
    A = tau[:, None, None] * (GTG + theta[:, :, None] * _nmp.identity(n_markers))
    b = tau * GTY
    beta = _utils.sample_multivariate_normal_many(b.T, A)

    ##
    return beta, tau


def _sample_beta_tau(YTY, GTG, GTY, beta, tau, zeta, eta, n_samples, beta_thr, s2_lims):
    """TODO."""
    # identify irrelevant genes and markers and exclude them
    idxs = (_nmp.abs(beta) > beta_thr) & (tau[:, None] * zeta * eta < 1 / s2_lims[0])
    idxs[[0, 1], [0, 1]] = True  # just a precaution
    idxs_markers = _nmp.any(idxs, 0)
    idxs_genes = _nmp.any(idxs, 1)

    YTY = YTY[idxs_genes]
    GTG = GTG[:, idxs_markers][idxs_markers, :]
    GTY = GTY[idxs_markers, :][:, idxs_genes]

    zeta = zeta[idxs_genes, :][:, idxs_markers]
    eta = eta[idxs_markers]

    beta[_nmp.ix_(idxs_genes, idxs_markers)], tau[idxs_genes] = \
        _sample_beta_tau_(YTY, GTG, GTY, zeta, eta, n_samples, s2_lims)

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
