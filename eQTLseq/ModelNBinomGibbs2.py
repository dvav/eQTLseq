"""Implements ModelNBinomGibbs2."""

import numpy as _nmp
import numpy.random as _rnd
import scipy.special as _spc


class ModelNBinomGibbs2(object):
    """A negative binomial model estimated using Gibbs sampling."""

    def __init__(self, **args):
        """TODO."""
        Z, n_markers = args['Z'], args['n_markers']
        n_samples, n_genes = Z.shape

        # initial conditions
        self.tau = _nmp.ones(n_genes)
        self.eta = _nmp.ones(n_markers)
        self.zeta = _nmp.ones((n_genes, n_markers))
        self.beta = _rnd.randn(n_genes, n_markers)
        self.phi = _nmp.exp(_rnd.randn(n_genes))
        self.mu = _nmp.mean(Z, 0)
        self.mu_phi, self.tau_phi = 0, 1

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
        Z, G, norm_factors = args['Z'], args['G'], args['norm_factors']
        beta_thr, s2_lims = args['beta_thr'], args['s2_lims']

        # sample mu and phi
        self.mu = _sample_mu(Z, G, norm_factors, self.phi, self.beta)
        self.phi = _sample_phi(Z, G, norm_factors, self.mu, self.phi, self.beta, self.mu_phi, self.tau_phi)

        # sample mu_phi and tau_phi
        self.mu_phi, self.tau_phi = _sample_mu_tau_phi(self.phi)

        # identify irrelevant genes and markers
        idxs = (_nmp.abs(self.beta) > beta_thr) & (self.zeta * self.eta * self.tau[:, None] < 1 / s2_lims[0])
        idxs[[0, 1], [0, 1]] = True  # just a precaution
        self.idxs_markers = _nmp.any(idxs, 0)
        self.idxs_genes = _nmp.any(idxs, 1)

        Z = Z[:, self.idxs_genes]
        G = G[:, self.idxs_markers]
        mu = self.mu[self.idxs_genes]
        phi = self.phi[self.idxs_genes]
        beta = self.beta[self.idxs_genes, :][:, self.idxs_markers]
        zeta = self.zeta[self.idxs_genes, :][:, self.idxs_markers]
        eta = self.eta[self.idxs_markers]
        tau = self.tau[self.idxs_genes]

        # update beta
        beta = _sample_beta(Z, G, norm_factors, mu, phi, beta, tau, zeta, eta)
        self.beta[_nmp.ix_(self.idxs_genes, self.idxs_markers)] = beta

        # update tau, zeta and eta
        self.tau = _sample_tau(self.beta, self.zeta, self.eta)
        self.tau = _nmp.clip(self.tau, 1 / s2_lims[1], 1 / s2_lims[0])

        self.zeta = _sample_zeta(self.beta, self.tau, self.eta)
        self.zeta = _nmp.clip(self.zeta, 1 / s2_lims[1], 1 / s2_lims[0])

        self.eta = _sample_eta(self.beta, self.tau, self.zeta)
        self.eta = _nmp.clip(self.eta, 1 / s2_lims[1], 1 / s2_lims[0])

        if(itr > args['n_burnin']):
            self.phi_sum += self.phi
            self.mu_sum += self.mu
            self.beta_sum += self.beta
            self.tau_sum += self.tau
            self.zeta_sum += self.zeta
            self.eta_sum += self.eta

            self.phi2_sum += self.phi**2
            self.mu2_sum += self.mu**2
            self.beta2_sum += self.beta**2
            self.tau2_sum += self.tau**2
            self.zeta2_sum += self.zeta**2
            self.eta2_sum += self.eta**2

    def get_estimates(self, **args):
        """TODO."""
        n_iters, n_burnin = args['n_iters'], args['n_burnin']

        #
        N = n_iters - n_burnin
        phi_mean, mu_mean = self.phi_sum / N, self.mu_sum / N
        tau_mean, zeta_mean, eta_mean, beta_mean = self.tau_sum / N, self.zeta_sum / N, self.eta_sum / N, \
            self.beta_sum / N

        phi_var, mu_var = self.phi2_sum / N - phi_mean**2, self.mu2_sum / N - mu_mean**2
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
        G, Z, c = args['G'], args['Z'], args['norm_factors']

        alpha = 1 / self.phi
        pi = alpha / (alpha + c[:, None] * self.mu * _nmp.exp(G.dot(self.beta.T)))

        loglik = (_spc.gammaln(Z + alpha) - _spc.gammaln(alpha) + alpha * _nmp.log(pi) + Z * _nmp.log1p(-pi)).sum()

        #
        return loglik


def _sample_phi_local(Z, G, c, mu, phi, beta, mu_phi, tau_phi, scale=0.01):
    n_samples, n_genes = Z.shape
    means = c[:, None] * mu * _nmp.exp(G.dot(beta.T))

    # sample proposals from the prior
    alpha = 1 / phi
    log_phi = _nmp.log(phi)
    pi = alpha / (alpha + means)

    phi_ = phi * _nmp.exp(scale * _rnd.randn(n_genes))
    alpha_ = 1 / phi_
    log_phi_ = _nmp.log(phi_)
    pi_ = alpha_ / (alpha_ + means)

    # compute logpost
    loglik = (_spc.gammaln(Z + alpha) - _spc.gammaln(alpha) + alpha * _nmp.log(pi) + Z * _nmp.log1p(-pi)).sum(0)
    loglik_ = (_spc.gammaln(Z + alpha_) - _spc.gammaln(alpha_) + alpha_ * _nmp.log(pi_) + Z * _nmp.log1p(-pi_)).sum(0)

    logprior = - log_phi - 0.5 * tau_phi * (log_phi - mu_phi)**2
    logprior_ = - log_phi_ - 0.5 * tau_phi * (log_phi_ - mu_phi)**2

    logpost = loglik + logprior
    logpost_ = loglik_ + logprior_

    # Metropolis step
    idxs = _rnd.rand(n_genes) < _nmp.exp(logpost_ - logpost)
    phi[idxs] = phi_[idxs]

    #
    return phi


def _sample_phi_global(Z, G, c, mu, phi, beta, mu_phi, tau_phi):
    n_samples, n_genes = Z.shape
    means = c[:, None] * mu * _nmp.exp(G.dot(beta.T))

    # sample proposals from the prior
    alpha = 1 / phi
    pi = alpha / (alpha + means)

    phi_ = _nmp.exp(mu_phi + _rnd.randn(n_genes) / _nmp.sqrt(tau_phi))
    alpha_ = 1 / phi_
    pi_ = alpha_ / (alpha_ + means)

    # compute loglik
    loglik = (_spc.gammaln(Z + alpha) - _spc.gammaln(alpha) + alpha * _nmp.log(pi) + Z * _nmp.log1p(-pi)).sum(0)
    loglik_ = (_spc.gammaln(Z + alpha_) - _spc.gammaln(alpha_) + alpha_ * _nmp.log(pi_) + Z * _nmp.log1p(-pi_)).sum(0)

    # Metropolis step
    idxs = _rnd.rand(n_genes) < _nmp.exp(loglik_ - loglik)
    phi[idxs] = phi_[idxs]

    #
    return phi


def _sample_phi(Z, G, norm_factors, mu, phi, beta, mu_phi, tau_phi):
    """TODO."""
    if _rnd.rand() < 0.5:
        phi = _sample_phi_local(Z, G, norm_factors, mu, phi, beta, mu_phi, tau_phi)
    else:
        phi = _sample_phi_global(Z, G, norm_factors, mu, phi, beta, mu_phi, tau_phi)

    #
    return phi


def _sample_mu(Z, G, c, phi, beta, a=0.5, b=0.5):
    n_samples, _ = Z.shape

    Z = Z / (_nmp.exp(G.dot(beta.T)) * c[:, None])
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


def _sample_beta_one_local(Z, G_i, G, c, mu, phi, beta_i, beta, tau, zeta_i, eta_i, scale=0.01):
    """TODO."""
    _, n_genes = Z.shape

    alpha = 1 / phi
    GBT = G.dot(beta.T)

    # sample proposals from a normal prior
    pi = alpha / (alpha + c[:, None] * mu * _nmp.exp(GBT + G_i[:, None] * beta_i))

    beta_i_ = beta_i * _nmp.exp(scale * _rnd.randn(n_genes))
    pi_ = alpha / (alpha + c[:, None] * mu * _nmp.exp(GBT + G_i[:, None] * beta_i_))

    # compute loglik
    loglik = (alpha * _nmp.log(pi) + Z * _nmp.log1p(-pi)).sum(0)
    loglik_ = (alpha * _nmp.log(pi_) + Z * _nmp.log1p(-pi_)).sum(0)

    logprior = -0.5 * tau * zeta_i * eta_i * beta_i**2
    logprior_ = -0.5 * tau * zeta_i * eta_i * beta_i_**2

    logpost = loglik + logprior
    logpost_ = loglik_ + logprior_

    # do Metropolis step
    idxs = _rnd.rand(n_genes) < _nmp.exp(logpost_ - logpost)
    beta_i[idxs] = beta_i_[idxs]

    #
    return beta_i


def _sample_beta_one_global(Z, G_i, G, c, mu, phi, beta_i, beta, tau, zeta_i, eta_i):
    """TODO."""
    _, n_genes = Z.shape

    alpha = 1 / phi
    GBT = G.dot(beta.T)

    # sample proposals from a normal prior
    pi = alpha / (alpha + c[:, None] * mu * _nmp.exp(GBT + G_i[:, None] * beta_i))

    beta_i_ = _rnd.normal(0, 1 / _nmp.sqrt(tau * eta_i * zeta_i))
    pi_ = alpha / (alpha + c[:, None] * mu * _nmp.exp(GBT + G_i[:, None] * beta_i_))

    # compute loglik
    loglik = (alpha * _nmp.log(pi) + Z * _nmp.log1p(-pi)).sum(0)
    loglik_ = (alpha * _nmp.log(pi_) + Z * _nmp.log1p(-pi_)).sum(0)

    # do Metropolis step
    idxs = _rnd.rand(n_genes) < _nmp.exp(loglik_ - loglik)
    beta_i[idxs] = beta_i_[idxs]

    #
    return beta_i


def _sample_beta_one(Z, G, norm_factors, mu, phi, beta, tau, zeta, eta, idx):
    """TODO."""
    _, n_markers = G.shape

    idxs = idx != _nmp.arange(n_markers)

    G_i, G = G[:, idx], G[:, idxs]
    beta_i, beta = beta[:, idx], beta[:, idxs]
    zeta_i = zeta[:, idx]
    eta_i = eta[idx]

    if _rnd.rand() < 0.5:
        beta_one = _sample_beta_one_local(Z, G_i, G, norm_factors, mu, phi, beta_i, beta, tau, zeta_i, eta_i)
    else:
        beta_one = _sample_beta_one_global(Z, G_i, G, norm_factors, mu, phi, beta_i, beta, tau, zeta_i, eta_i)

    #
    return beta_one


def _sample_beta(Z, G, norm_factors, mu, phi, beta, tau, zeta, eta):
    """TODO."""
    _, n_markers = G.shape

    idxs = _rnd.permutation(n_markers)
    for idx in idxs:
        beta[:, idx] = _sample_beta_one(Z, G, norm_factors, mu, phi, beta, tau, zeta, eta, idx)
    # beta_ = _nmp.asarray([_sample_beta_one(Z, G, norm_factors, mu, phi, beta, tau, zeta, eta, idx) for idx in idxs]).T
    # beta[:, idxs] = beta_
    #
    return beta


def _sample_tau(beta, zeta, eta):
    """TODO."""
    _, n_markers = zeta.shape

    # sample zeta
    shape = 0.5 * n_markers
    rate = 0.5 * (eta * beta**2 * zeta).sum(1)
    tau = _rnd.gamma(shape, 1 / rate)

    ##
    return tau


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
