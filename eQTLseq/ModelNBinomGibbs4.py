"""Implements ModelNBinomGibbs4."""

import numpy as _nmp
import numpy.random as _rnd
import scipy.special as _spc

import eQTLseq.utils as _utils


class ModelNBinomGibbs4(object):
    """A negative binomial model estimated using Gibbs sampling."""

    def __init__(self, **args):
        """TODO."""
        Z, c, n_markers = args['Z'], args['norm_factors'], args['n_markers']
        n_samples, n_genes = Z.shape

        # initial conditions
        self.tau = _nmp.ones(n_genes)
        self.eta = _nmp.ones(n_markers)
        self.zeta = _nmp.ones((n_genes, n_markers))
        self.beta = _rnd.randn(n_genes, n_markers)

        self.Y = _rnd.randn(n_samples, n_genes)

        self.mu_phi, self.tau_phi, self.phi = 0, 1, _nmp.exp(_rnd.randn(n_genes))
        self.mu = _nmp.mean(Z / c[:, None] * _nmp.exp(-self.Y), 0)

        self.idxs_markers = _nmp.ones(n_markers, dtype='bool')
        self.idxs_genes = _nmp.ones(n_genes, dtype='bool')

        self.tau_sum, self.tau2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)
        self.zeta_sum, self.zeta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))
        self.eta_sum, self.eta2_sum = _nmp.zeros(n_markers), _nmp.zeros(n_markers)
        self.beta_sum, self.beta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))
        self.phi_sum, self.phi2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)
        self.mu_sum, self.mu2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)

    def update(self, itr, **args):
        """TODO."""
        Z, G, GTG, norm_factors = args['Z'], args['G'], args['GTG'], args['norm_factors']
        beta_thr, s2_lims = args['beta_thr'], args['s2_lims']

        # sample mu and phi
        self.mu = _sample_mu(Z, norm_factors, self.phi, self.Y)
        self.phi = _sample_phi(Z, G, norm_factors, self.mu, self.phi, self.Y, self.beta, self.tau,
                               self.mu_phi, self.tau_phi)

        # sample mu_phi and tau_phi
        self.mu_phi, self.tau_phi = _sample_mu_tau_phi(self.phi)

        # sample Y
        # self.Y = args['YY']
        self.Y = _sample_Y(Z, G, norm_factors, self.mu, self.phi, self.Y, self.beta, self.tau)
        self.Y = (self.Y - _nmp.mean(self.Y, 0)) / _nmp.std(self.Y, 0)

        # identify relevant genes and markers and include
        idxs = (_nmp.abs(self.beta) > beta_thr) & \
            (self.zeta * self.eta * self.tau[:, None] / self.phi[:, None] < 1 / s2_lims[0])
        idxs[[0, 1], [0, 1]] = True  # just a precaution
        self.idxs_markers = _nmp.any(idxs, 0)
        self.idxs_genes = _nmp.any(idxs, 1)

        Y = self.Y[:, self.idxs_genes]
        G = G[:, self.idxs_markers]
        GTG = GTG[:, self.idxs_markers][self.idxs_markers, :]
        zeta = self.zeta[self.idxs_genes, :][:, self.idxs_markers]
        eta = self.eta[self.idxs_markers]

        # update beta and tau
        YTY = _nmp.sum(Y**2, 0)
        GTY = G.T.dot(Y)
        beta, tau = _sample_beta_tau(YTY, GTG, GTY, zeta, eta, args['n_samples'], s2_lims)

        self.beta[_nmp.ix_(self.idxs_genes, self.idxs_markers)] = beta
        self.tau[self.idxs_genes] = tau

        # sample eta and zeta
        self.zeta = _sample_zeta(self.beta, self.tau, self.eta)
        self.zeta = _nmp.clip(self.zeta, 1 / s2_lims[1], 1 / s2_lims[0])

        self.eta = _sample_eta(self.beta, self.tau, self.zeta)
        self.eta = _nmp.clip(self.eta, 1 / s2_lims[1], 1 / s2_lims[0])

        if(itr > args['n_burnin']):
            self.phi_sum += self.phi
            self.mu_sum += self.mu
            self.tau_sum += self.tau
            self.zeta_sum += self.zeta
            self.eta_sum += self.eta
            self.beta_sum += self.beta

            self.phi2_sum += self.phi**2
            self.mu2_sum += self.mu**2
            self.tau2_sum += self.tau**2
            self.zeta2_sum += self.zeta**2
            self.eta2_sum += self.eta**2
            self.beta2_sum += self.beta**2

    def get_estimates(self, **args):
        """TODO."""
        N = args['n_iters'] - args['n_burnin']
        phi_mean, mu_mean = self.phi_sum / N, self.mu_sum / N
        phi_var, mu_var = self.phi2_sum / N - phi_mean**2, self.mu2_sum / N - mu_mean**2

        tau_mean, zeta_mean, eta_mean, beta_mean = self.tau_sum / N, self.zeta_sum / N, self.eta_sum / N, \
            self.beta_sum / N
        tau_var, zeta_var, eta_var, beta_var = self.tau2_sum / N - tau_mean**2, self.zeta2_sum / N - zeta_mean**2, \
            self.eta2_sum / N - eta_mean**2, self.beta2_sum / N - beta_mean**2

        return {
            'phi': phi_mean, 'phi_var': phi_var,
            'mu': mu_mean, 'mu_var': mu_var,
            'tau': tau_mean, 'tau_var': tau_var,
            'zeta': zeta_mean, 'zeta_var': zeta_var,
            'eta': eta_mean, 'eta_var': eta_var,
            'beta': beta_mean, 'beta_var': beta_var

        }

    def get_log_likelihood(self, **args):
        """TODO."""
        G = args['G']
        n_samples, n_genes = self.Y.shape

        #
        theta = self.tau / self.phi
        resid = self.Y - G.dot(self.beta.T)
        loglik = (0.5 * _nmp.log(theta) - 0.5 * theta * resid**2).sum() / (n_samples * n_genes)

        #
        return loglik


def _sample_phi_local(Z, G, c, mu, phi, Y, beta, tau, mu_phi, tau_phi, scale=0.01):
    n_samples, n_genes = Z.shape
    means = c[:, None] * mu * _nmp.exp(Y)
    GBT = G.dot(beta.T)
    resid2 = (Y - GBT)**2

    # sample proposals from the prior
    alpha = 1 / phi
    log_phi = _nmp.log(phi)
    pi = alpha / (alpha + means)
    theta = tau / phi

    phi_ = phi * _nmp.exp(scale * _rnd.randn(n_genes))
    alpha_ = 1 / phi_
    log_phi_ = _nmp.log(phi_)
    pi_ = alpha_ / (alpha_ + means)
    theta_ = tau / phi_

    # compute logpost
    loglik = (_spc.gammaln(Z + alpha) - _spc.gammaln(alpha) + alpha * _nmp.log(pi) + Z * _nmp.log1p(-pi)).sum(0) \
        - 0.5 * (log_phi + theta * resid2).sum(0)
    loglik_ = (_spc.gammaln(Z + alpha_) - _spc.gammaln(alpha_) + alpha_ * _nmp.log(pi_) + Z * _nmp.log1p(-pi_)).sum(0) \
        - 0.5 * (log_phi_ + theta_ * resid2).sum(0)

    logprior = - log_phi - 0.5 * tau_phi * (log_phi - mu_phi)**2
    logprior_ = - log_phi_ - 0.5 * tau_phi * (log_phi_ - mu_phi)**2

    logpost = loglik + logprior
    logpost_ = loglik_ + logprior_

    # Metropolis step
    idxs = _rnd.rand(n_genes) < _nmp.exp(logpost_ - logpost)
    phi[idxs] = phi_[idxs]

    #
    return phi


def _sample_phi_global(Z, G, c, mu, phi, Y, beta, tau, mu_phi, tau_phi):
    n_samples, n_genes = Z.shape
    means = c[:, None] * mu * _nmp.exp(Y)
    GBT = G.dot(beta.T)
    resid2 = (Y - GBT)**2

    # sample proposals from the prior
    alpha = 1 / phi
    log_phi = _nmp.log(phi)
    pi = alpha / (alpha + means)
    theta = tau / phi

    phi_ = _nmp.exp(mu_phi + _rnd.randn(n_genes) / _nmp.sqrt(tau_phi))
    alpha_ = 1 / phi_
    log_phi_ = _nmp.log(phi_)
    pi_ = alpha_ / (alpha_ + means)
    theta_ = tau / phi_

    # compute loglik
    loglik = (_spc.gammaln(Z + alpha) - _spc.gammaln(alpha) + alpha * _nmp.log(pi) + Z * _nmp.log1p(-pi)).sum(0) \
        - 0.5 * (log_phi + theta * resid2).sum(0)
    loglik_ = (_spc.gammaln(Z + alpha_) - _spc.gammaln(alpha_) + alpha_ * _nmp.log(pi_) + Z * _nmp.log1p(-pi_)).sum(0) \
        - 0.5 * (log_phi_ + theta_ * resid2).sum(0)

    # Metropolis step
    idxs = _rnd.rand(n_genes) < _nmp.exp(loglik_ - loglik)
    phi[idxs] = phi_[idxs]

    #
    return phi


def _sample_phi(Z, G, norm_factors, mu, phi, Y, beta, tau, mu_phi, tau_phi):
    """TODO."""
    # if _rnd.rand() < 0.5:
    #     phi = _sample_phi_local(Z, G, norm_factors, mu, phi, Y, beta, tau, mu_phi, tau_phi)
    # else:
    phi = _sample_phi_global(Z, G, norm_factors, mu, phi, Y, beta, tau, mu_phi, tau_phi)

    #
    return phi


def _sample_mu(Z, c, phi, Y, a=0.5, b=0.5):
    n_samples, _ = Z.shape

    Z = Z / c[:, None] * _nmp.exp(-Y)
    alpha = 1 / phi

    c1 = a + n_samples * alpha
    c2 = b + Z.sum(0)

    pi = _rnd.beta(c1, c2)
    mu = alpha * (1 - pi) / pi

    #
    return mu


def _sample_mu_tau_phi(phi):
    n_genes = phi.size
    log_phi = _nmp.log(phi)
    log_phi2 = log_phi**2

    # sample tau_phi
    shape = 0.5 * n_genes
    rate = 0.5 * log_phi2.sum()
    tau_phi = _rnd.gamma(shape, 1 / rate)

    # sample mu_phi, given tau_phi
    mean = log_phi.sum() / n_genes
    prec = n_genes * tau_phi
    mu_phi = _rnd.normal(mean, 1 / _nmp.sqrt(prec))

    #
    return mu_phi, tau_phi


def _sample_Y_local(Z, G, c, mu, phi, Y, beta, tau, scale=0.01):
    n_samples, n_genes = Z.shape
    alpha = 1 / phi
    GBT = G.dot(beta.T)
    theta = tau / phi

    # sample proposals from a normal prior
    pi = alpha / (alpha + c[:, None] * mu * _nmp.exp(Y))

    Y_ = Y * _nmp.exp(scale * _rnd.randn(n_samples, n_genes))
    pi_ = alpha / (alpha + c[:, None] * mu * _nmp.exp(Y_))

    # compute loglik
    loglik = alpha * _nmp.log(pi) + Z * _nmp.log1p(-pi)
    loglik_ = alpha * _nmp.log(pi_) + Z * _nmp.log1p(-pi_)

    logprior = -0.5 * theta * (Y - GBT)**2
    logprior_ = -0.5 * theta * (Y_ - GBT)**2

    logpost = loglik + logprior
    logpost_ = loglik_ + logprior_

    # do Metropolis step
    idxs = _rnd.rand(n_samples, n_genes) < _nmp.exp(logpost_ - logpost)
    Y[idxs] = Y_[idxs]

    #
    return Y


def _sample_Y_global(Z, G, c, mu, phi, Y, beta, tau):
    n_samples, n_genes = Z.shape
    alpha = 1 / phi

    # sample proposals from a normal prior
    pi = alpha / (alpha + c[:, None] * mu * _nmp.exp(Y))

    Y_ = _rnd.normal(G.dot(beta.T), _nmp.sqrt(phi / tau))
    pi_ = alpha / (alpha + c[:, None] * mu * _nmp.exp(Y_))

    # compute loglik
    loglik = alpha * _nmp.log(pi) + Z * _nmp.log1p(-pi)
    loglik_ = alpha * _nmp.log(pi_) + Z * _nmp.log1p(-pi_)

    # do Metropolis step
    idxs = _rnd.rand(n_samples, n_genes) < _nmp.exp(loglik_ - loglik)
    Y[idxs] = Y_[idxs]

    #
    return Y


def _sample_Y(Z, G, norm_factors, mu, phi, Y, beta, tau):
    """TODO."""
    # if _rnd.rand() < 0.5:
    #     Y = _sample_Y_local(Z, G, norm_factors, mu, phi, Y, beta, tau)
    # else:
    Y = _sample_Y_global(Z, G, norm_factors, mu, phi, Y, beta, tau)

    #
    return Y


def _sample_beta_tau(YTY, GTG, GTY, zeta, eta, n_samples, s2_lims):
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
    beta = _utils.sample_multivariate_normal_many(b.T, A)

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
