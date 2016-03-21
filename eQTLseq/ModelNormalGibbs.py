"""Implements ModelNormalGibbs."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.mdl_common as _common


class ModelNormalGibbs(object):
    """A normal model estimated using Gibbs sampling."""

    def __init__(self, **args):
        """TODO."""
        n_iters, n_genes, n_markers = args['n_iters'], args['n_genes'], args['n_markers']

        # initial conditions
        self.tau = _nmp.ones(n_genes)
        self.eta = _nmp.ones(n_markers)
        self.zeta = _nmp.ones((n_genes, n_markers))
        self.beta = _rnd.randn(n_genes, n_markers)

        self.idxs_markers = _nmp.ones(n_markers, dtype='bool')
        self.idxs_genes = _nmp.ones(n_genes, dtype='bool')

        self._trace = _nmp.empty(n_iters + 1)
        self._trace.fill(_nmp.nan)
        self._trace[0] = 0

        self.tau_sum, self.tau2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)
        self.zeta_sum, self.zeta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))
        self.eta_sum, self.eta2_sum = _nmp.zeros(n_markers), _nmp.zeros(n_markers)
        self.beta_sum, self.beta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))

    def update(self, itr, **args):
        """TODO."""
        Y, G, YTY, GTG, GTY = args['Y'], args['G'], args['YTY'], args['GTG'], args['GTY']
        n_burnin, beta_thr, s2_lims = args['n_burnin'], args['beta_thr'], args['s2_lims']
        n_samples, _ = G.shape

        idxs = _nmp.abs(self.beta) > beta_thr
        idxs[[0, 1], [0, 1]] = True  # just a precaution
        self.idxs_markers = _nmp.any(idxs, 0)
        self.idxs_genes = _nmp.any(idxs, 1)

        Y = Y[:, self.idxs_genes]
        G = G[:, self.idxs_markers]
        YTY = YTY[self.idxs_genes]
        GTG = GTG[:, self.idxs_markers][self.idxs_markers, :]
        GTY = GTY[self.idxs_markers, :][:, self.idxs_genes]

        zeta = self.zeta[self.idxs_genes, :][:, self.idxs_markers]
        eta = self.eta[self.idxs_markers]

        # sample beta, tau and zeta
        beta, tau = _common.sample_beta_tau(YTY, GTG, GTY, zeta, eta, n_samples)

        zeta = _common.sample_zeta(beta, tau, eta)
        zeta = _nmp.clip(zeta, 1 / s2_lims[1], 1 / s2_lims[0])

        eta = _common.sample_eta(beta, tau, zeta)
        eta = _nmp.clip(eta, 1 / s2_lims[1], 1 / s2_lims[0])

        # update the rest
        self._trace[itr] = _calculate_joint_log_likelihood(Y, G, beta, tau, zeta, eta)

        self.beta[_nmp.ix_(self.idxs_genes, self.idxs_markers)] = beta
        self.zeta[_nmp.ix_(self.idxs_genes, self.idxs_markers)] = zeta
        self.tau[self.idxs_genes] = tau
        self.eta[self.idxs_markers] = eta

        if(itr > n_burnin):
            self.tau_sum += self.tau
            self.zeta_sum += self.zeta
            self.eta_sum += self.eta
            self.beta_sum += self.beta

            self.tau2_sum += self.tau**2
            self.zeta2_sum += self.zeta**2
            self.eta2_sum += self.eta**2
            self.beta2_sum += self.beta**2

    @property
    def trace(self):
        """TODO."""
        return self._trace

    def get_estimates(self, **args):
        """TODO."""
        n_iters, n_burnin = args['n_iters'], args['n_burnin']

        #
        N = n_iters - n_burnin
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


def _calculate_joint_log_likelihood(Y, G, beta, tau, zeta, eta):
    # number of samples and markers
    n_samples, n_genes = Y.shape
    _, n_markers = G.shape

    #
    resid = Y - G.dot(beta.T)

    A = (0.5 * n_samples + 0.5 * n_markers - 1) * _nmp.log(tau).sum()
    B = 0.5 * (tau * resid**2).sum()
    C = 0.5 * (tau[:, None] * eta * beta**2 * zeta).sum()
    D = 0.5 * _nmp.log(zeta).sum()

    #
    return (A - B - C - D) / (n_markers * n_genes)
