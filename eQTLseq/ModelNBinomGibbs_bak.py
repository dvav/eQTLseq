"""Implements ModelNBinomGibbs."""

import numpy as _nmp
import numpy.random as _rnd
import scipy.special as _spc

import eQTLseq.mdl_common as _common


class ModelNBinomGibbs(object):
    """A negative binomial model estimated using Gibbs sampling."""

    def __init__(self, **args):
        """TODO."""
        Z, n_markers = args['Y'], args['n_markers']
        n_samples, n_genes = Z.shape

        # initial conditions
        self.mu_phi, self.tau_phi = 0, 1
        self.phi, self.mu = _nmp.exp(_rnd.randn(n_genes)), _nmp.mean(Z, 0)
        self.zeta = _nmp.ones((n_genes, n_markers))
        self.eta = _nmp.ones(n_markers)
        self.tau = _nmp.ones(n_genes)
        self.beta = _rnd.randn(n_genes, n_markers)
        self.Y = _rnd.randn(n_samples, n_genes)

        self.idxs_markers = _nmp.ones(n_markers, dtype='bool')
        self.idxs_genes = _nmp.ones(n_genes, dtype='bool')

        self.beta_sum, self.beta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))
        self.phi_sum, self.phi2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)
        self.mu_sum, self.mu2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)

    def update(self, itr, **args):
        """TODO."""
        Z, G, GTG = args['Y'], args['G'], args['GTG']
        n_burnin, beta_thr, s2_lims = args['n_burnin'], args['beta_thr'], args['s2_lims']
        n_samples, _ = G.shape

        # sample phi and mu
        if(_rnd.rand() < 0.5):
            self.phi = _sample_phi_local(Z, self.mu, self.phi, self.Y, self.mu_phi, self.tau_phi)
        else:
            self.phi = _sample_phi_global(Z, self.mu, self.phi, self.Y, self.mu_phi, self.tau_phi)
        self.mu = _sample_mu(Z, self.phi, self.Y)

        # sample psi
        if(_rnd.rand() < 0.5):
            self.Y = _sample_Y_local(Z, G, self.mu, self.phi, self.Y, self.beta, self.tau)
        else:
            self.Y = _sample_Y_global(Z, G, self.mu, self.phi, self.Y, self.beta, self.tau)

        # sample mu_phi and tau_phi
        self.mu_phi, self.tau_phi = _sample_mu_tau_phi(self.phi, self.tau_phi)

        # identify irrelevant genes and markers and exclude them
        idxs = (_nmp.abs(self.beta) > beta_thr) & (self.zeta * self.eta * self.tau[:, None] < 1 / s2_lims[0])
        idxs[[0, 1], [0, 1]] = True  # just a precaution
        self.idxs_markers = _nmp.any(idxs, 0)
        self.idxs_genes = _nmp.any(idxs, 1)

        Y = self.Y[:, self.idxs_genes]
        G = G[:, self.idxs_markers]
        GTG = GTG[:, self.idxs_markers][self.idxs_markers, :]
        YTY = _nmp.sum(Y**2, 0)
        GTY = G.T.dot(Y)

        zeta = self.zeta[self.idxs_genes, :][:, self.idxs_markers]
        eta = self.eta[self.idxs_markers]

        # sample beta, tau and zeta
        beta, tau = _common.sample_beta_tau(YTY, GTG, GTY, zeta, eta, n_samples)

        zeta = _common.sample_zeta(beta, tau, eta)
        zeta = _nmp.clip(zeta, 1 / s2_lims[1], 1 / s2_lims[0])

        eta = _common.sample_eta(beta, tau, zeta)
        eta = _nmp.clip(eta, 1 / s2_lims[1], 1 / s2_lims[0])

        self.beta[_nmp.ix_(self.idxs_genes, self.idxs_markers)] = beta
        self.zeta[_nmp.ix_(self.idxs_genes, self.idxs_markers)] = zeta
        self.tau[self.idxs_genes] = tau
        self.eta[self.idxs_markers] = eta

        if(itr > n_burnin):
            self.beta_sum += self.beta
            self.phi_sum += self.phi
            self.mu_sum += self.mu

            self.beta2_sum += self.beta**2
            self.phi2_sum += self.phi**2
            self.mu2_sum += self.mu**2

    def get_estimates(self, **args):
        """TODO."""
        n_iters, n_burnin = args['n_iters'], args['n_burnin']

        #
        N = n_iters - n_burnin
        beta_mean, mu_mean, phi_mean = self.beta_sum / N, self.mu_sum / N, self.phi_sum / N
        beta_var, mu_var, phi_var = self.beta2_sum / N - beta_mean**2, self.mu2_sum / N - mu_mean**2, \
            self.phi2_sum / N - phi_mean**2

        return {
            'beta': beta_mean, 'beta_var': beta_var,
            'phi': phi_mean, 'phi_var': phi_var,
            'mu': mu_mean, 'mu_var': mu_var
        }

    def get_joint_log_likelihood(self, **args):
        """TODO."""
        Z, G = args['Y'], args['G']

        Y = self.Y[:, self.idxs_genes]
        G = G[:, self.idxs_markers]
        beta = self.beta[self.idxs_genes, :][:, self.idxs_markers]
        zeta = self.zeta[self.idxs_genes, :][:, self.idxs_markers]
        eta = self.eta[self.idxs_markers]
        tau = self.tau[self.idxs_genes]

        # number of samples and markers
        n_samples, n_markers = G.shape

        #
        resid = Y - G.dot(beta.T)

        A = (0.5 * n_samples + 0.5 * n_markers - 1) * _nmp.log(tau).sum()
        B = 0.5 * (tau * resid**2).sum()
        C = 0.5 * (tau[:, None] * eta * beta**2 * zeta).sum()
        D = 0.5 * _nmp.log(zeta).sum()

        #
        return A - B - C - D


def _sample_phi_global(Z, mu, phi, Y, mu_phi, tau_phi):
    n_samples, n_genes = Z.shape

    means = mu * _nmp.exp(Y)

    # sample proposals from the prior
    alpha = 1 / phi
    pi = alpha / (alpha + means)

    phi_ = _nmp.exp(_rnd.normal(mu_phi, 1 / _nmp.sqrt(tau_phi), n_genes))
    alpha_ = 1 / phi_
    pi_ = alpha_ / (alpha_ + means)

    # compute loglik
    loglik = _spc.gammaln(Z + alpha).sum(0) - n_samples * _spc.gammaln(alpha) \
        + alpha * _nmp.log(pi).sum(0) + _nmp.sum(Z * _nmp.log1p(-pi), 0)
    loglik_ = _spc.gammaln(Z + alpha_).sum(0) - n_samples * _spc.gammaln(alpha_) \
        + alpha_ * _nmp.log(pi_).sum(0) + _nmp.sum(Z * _nmp.log1p(-pi_), 0)

    # Metropolis step
    idxs = _rnd.rand(n_genes) < _nmp.exp(loglik_ - loglik)
    phi[idxs] = phi_[idxs]

    #
    return phi


def _sample_phi_local(Z, mu, phi, Y, mu_phi, tau_phi, scale=0.01):
    n_samples, n_genes = Z.shape

    means = mu * _nmp.exp(Y)

    # sample proposals from a log-normal with small scale
    alpha = 1 / phi
    pi = alpha / (alpha + means)
    log_phi = _nmp.log(phi)

    phi_ = phi * _nmp.exp(scale * _rnd.randn(n_genes))
    alpha_ = 1 / phi_
    pi_ = alpha_ / (alpha_ + means)
    log_phi_ = _nmp.log(phi_)

    # compute loglik
    loglik = _spc.gammaln(Z + alpha).sum(0) - n_samples * _spc.gammaln(alpha) \
        + alpha * _nmp.log(pi).sum(0) + _nmp.sum(Z * _nmp.log1p(-pi), 0) \
        - log_phi - 0.5 * tau_phi * (log_phi - mu_phi)**2
    loglik_ = _spc.gammaln(Z + alpha_).sum(0) - n_samples * _spc.gammaln(alpha_) \
        + alpha_ * _nmp.log(pi_).sum(0) + _nmp.sum(Z * _nmp.log1p(-pi_), 0) \
        - log_phi_ - 0.5 * tau_phi * (log_phi_ - mu_phi)**2

    # Metropolis step
    idxs = _rnd.rand(n_genes) < _nmp.exp(loglik_ - loglik)
    phi[idxs] = phi_[idxs]

    #
    return phi


def _sample_mu_tau_phi(phi, tau_phi):
    n_genes = phi.size
    log_phi = _nmp.log(phi)

    # sample mu_phi
    mean = log_phi.sum() / n_genes
    prec = n_genes * tau_phi
    mu_phi = _rnd.normal(mean, 1 / _nmp.sqrt(prec))

    # sample tau_phi
    shape = 0.5 * n_genes
    rate = 0.5 * _nmp.sum((log_phi - mu_phi)**2)
    tau_phi = _rnd.gamma(shape, 1 / rate)

    #
    return mu_phi, tau_phi


def _sample_mu(Z, phi, Y, a=0.5, b=0.5):
    n_samples, _ = Z.shape

    Z = Z / _nmp.exp(Y)
    alpha = 1 / phi

    c1 = a + n_samples * alpha
    c2 = b + Z.sum(0)

    pi = _rnd.beta(c1, c2)
    mu = alpha * (1 - pi) / pi

    #
    return mu


def _sample_Y_global(Z, G, mu, phi, Y, beta, tau):
    n_samples, n_genes = Z.shape
    alpha = 1 / phi

    # sample proposals from a normal prior
    pi = alpha / (alpha + mu * _nmp.exp(Y))

    Y_ = _rnd.normal(G.dot(beta.T), 1 / _nmp.sqrt(tau))
    pi_ = alpha / (alpha + mu * _nmp.exp(Y_))

    # compute loglik
    loglik = alpha * _nmp.log(pi) + Z * _nmp.log1p(-pi)
    loglik_ = alpha * _nmp.log(pi_) + Z * _nmp.log1p(-pi_)

    # do Metropolis step
    idxs = _rnd.rand(n_samples, n_genes) < _nmp.exp(loglik_ - loglik)
    Y[idxs] = Y_[idxs]

    #
    return Y


def _sample_Y_local(Z, G, mu, phi, Y, beta, tau, scale=0.01):
    n_samples, n_genes = Z.shape
    alpha = 1 / phi
    GBT = G.dot(beta.T)

    # sample proposals from a log-normal with small scale
    pi = alpha / (alpha + _nmp.exp(Y))

    Y_ = Y * _nmp.exp(scale * _rnd.randn(n_samples, n_genes))
    pi_ = alpha / (alpha + _nmp.exp(Y_))

    # compute loglik
    loglik = alpha * _nmp.log(pi) + Z * _nmp.log1p(-pi) - 0.5 * tau * (Y - GBT)**2
    loglik_ = alpha * _nmp.log(pi_) + Z * _nmp.log1p(-pi_) - 0.5 * tau * (Y_ - GBT)**2

    # do Metropolis step
    idxs = _rnd.rand(n_samples, n_genes) < _nmp.exp(loglik_ - loglik)
    Y[idxs] = Y_[idxs]

    #
    return Y
