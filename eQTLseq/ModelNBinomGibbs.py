"""Implements ModelNBinomGibbs."""

import numpy as _nmp
import numpy.random as _rnd
import scipy.special as _spc

import eQTLseq.mdl_common_gibbs as _common


class ModelNBinomGibbs:
    """A negative binomial model estimated using Gibbs sampling."""

    def __init__(self, Y, G, n_iters, n_burnin, s2_lims):
        """TODO."""
        # standarize genotypes
        self.Y = Y
        self.G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0)

        # used later in calculations
        self.GTG = self.G.T.dot(self.G)

        # number of samples, genes and genetic markers
        n_samples1, n_genes = Y.shape
        n_samples2, n_markers = G.shape

        assert n_samples1 == n_samples2
        n_samples = n_samples1

        # initial conditions
        self.mu_phi, self.tau_phi = 0, 1
        self.phi, self.mu = _nmp.exp(_rnd.randn(n_genes)), _nmp.mean(self.Y, 0)
        self.zeta = _nmp.ones((n_genes, n_markers))
        self.tau_psi = _nmp.ones(n_genes)
        self.beta = _rnd.randn(n_genes, n_markers)
        self.psi = _rnd.randn(n_samples, n_genes)

        self.traces = _nmp.zeros((n_iters + 1, 8))
        self.traces[0, :] = [
            self.mu_phi, self.tau_phi,
            _nmp.sqrt(_nmp.sum(_nmp.log(self.phi)**2)), _nmp.sqrt(_nmp.sum(_nmp.log(self.mu)**2)),
            1 / _nmp.sqrt(_nmp.sum(self.tau_psi**2)), _nmp.sqrt(_nmp.sum(self.psi**2)),
            1 / _nmp.sqrt(_nmp.sum(self.zeta**2)), _nmp.sqrt(_nmp.sum(self.beta**2))
        ]

        self.beta_sum, self.beta2_sum = _nmp.zeros((n_genes, n_markers)), _nmp.zeros((n_genes, n_markers))
        self.phi_sum, self.phi2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)
        self.mu_sum, self.mu2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)

        # other parameters
        self.n_iters = n_iters
        self.n_burnin = n_burnin
        self.s2_min = s2_lims[0]
        self.s2_max = s2_lims[1]

    def update(self, itr):
        """TODO."""
        # sample phi and mu
        if(_rnd.rand() < 0.5):
            self.phi = _sample_phi_local(self.Y, self.mu, self.phi, self.psi, self.mu_phi, self.tau_phi)
        else:
            self.phi = _sample_phi_global(self.Y, self.mu, self.phi, self.psi, self.mu_phi, self.tau_phi)
        self.mu = _sample_mu(self.Y, self.phi, self.psi)

        # sample psi
        if(_rnd.rand() < 0.5):
            self.psi = _sample_psi_local(self.Y, self.G, self.mu, self.phi, self.psi, self.beta, self.tau_psi)
        else:
            self.psi = _sample_psi_global(self.Y, self.G, self.mu, self.phi, self.psi, self.beta, self.tau_psi)

        # sample mu_phi and tau_phi
        self.mu_phi, self.tau_phi = _sample_mu_tau_phi(self.phi, self.tau_phi)

        # sample beta, tau_psi and zeta
        self.beta = _common.sample_beta(self.GTG, self.G.T.dot(self.psi), self.tau_psi, self.zeta)

        self.tau_psi = _common.sample_tau(self.psi, self.G, self.beta, self.zeta)
        self.tau_psi = _nmp.clip(self.tau_psi, 1 / self.s2_max, 1 / self.s2_min)

        self.zeta = _common.sample_zeta(self.beta, self.tau_psi)
        self.zeta = _nmp.clip(self.zeta, 1 / self.s2_max, 1 / self.s2_min)

        # update the rest
        phi2, mu2, beta2 = self.phi**2, self.mu**2, self.beta**2
        self.traces[itr, :] = [
            self.mu_phi, self.tau_phi,
            _nmp.sqrt(_nmp.sum(phi2)), _nmp.sqrt(_nmp.sum(mu2)),
            1 / _nmp.sqrt(_nmp.sum(self.tau_psi**2)), _nmp.sqrt(_nmp.sum(self.psi**2)),
            1 / _nmp.sqrt(_nmp.sum(self.zeta**2)), _nmp.sqrt(_nmp.sum(beta2))
        ]

        if(itr > self.n_burnin):
            self.beta_sum += self.beta
            self.phi_sum += self.phi
            self.mu_sum += self.mu

            self.beta2_sum += beta2
            self.phi2_sum += phi2
            self.mu2_sum += mu2

    def stats(self):
        """TODO."""
        N = self.n_iters - self.n_burnin
        beta_mean, mu_mean, phi_mean = self.beta_sum / N, self.mu_sum / N, self.phi_sum / N
        beta_var, mu_var, phi_var = self.beta2_sum / N - beta_mean**2, self.mu2_sum / N - mu_mean**2, \
            self.phi2_sum / N - phi_mean**2

        return {
            'traces': self.traces,
            'beta_mean': beta_mean, 'beta_var': beta_var,
            'phi_mean': phi_mean, 'phi_var': phi_var,
            'mu_mean': mu_mean, 'mu_var': mu_var
        }


def _sample_phi_global(Y, mu, phi, psi, mu_phi, tau_phi):
    n_samples, n_genes = Y.shape

    means = mu * _nmp.exp(psi)

    # sample proposals from the prior
    alpha = 1 / phi
    pi = alpha / (alpha + means)

    phi_ = _nmp.exp(_rnd.normal(mu_phi, 1 / _nmp.sqrt(tau_phi), n_genes))
    alpha_ = 1 / phi_
    pi_ = alpha_ / (alpha_ + means)

    # compute loglik
    loglik = _spc.gammaln(Y + alpha).sum(0) - n_samples * _spc.gammaln(alpha) \
        + alpha * _nmp.log(pi).sum(0) + _nmp.sum(Y * _nmp.log1p(-pi), 0)
    loglik_ = _spc.gammaln(Y + alpha_).sum(0) - n_samples * _spc.gammaln(alpha_) \
        + alpha_ * _nmp.log(pi_).sum(0) + _nmp.sum(Y * _nmp.log1p(-pi_), 0)

    # Metropolis step
    idxs = _rnd.rand(n_genes) < _nmp.exp(loglik_ - loglik)
    phi[idxs] = phi_[idxs]

    #
    return phi


def _sample_phi_local(Y, mu, phi, psi, mu_phi, tau_phi, scale=0.01):
    n_samples, n_genes = Y.shape

    means = mu * _nmp.exp(psi)

    # sample proposals from a log-normal with small scale
    alpha = 1 / phi
    pi = alpha / (alpha + means)
    log_phi = _nmp.log(phi)

    phi_ = phi * _nmp.exp(scale * _rnd.randn(n_genes))
    alpha_ = 1 / phi_
    pi_ = alpha_ / (alpha_ + means)
    log_phi_ = _nmp.log(phi_)

    # compute loglik
    loglik = _spc.gammaln(Y + alpha).sum(0) - n_samples * _spc.gammaln(alpha) \
        + alpha * _nmp.log(pi).sum(0) + _nmp.sum(Y * _nmp.log1p(-pi), 0) \
        - log_phi - 0.5 * tau_phi * (log_phi - mu_phi)**2
    loglik_ = _spc.gammaln(Y + alpha_).sum(0) - n_samples * _spc.gammaln(alpha_) \
        + alpha_ * _nmp.log(pi_).sum(0) + _nmp.sum(Y * _nmp.log1p(-pi_), 0) \
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


def _sample_mu(Y, phi, psi, a=0.5, b=0.5):
    n_samples, _ = Y.shape

    Y = Y / _nmp.exp(psi)
    alpha = 1 / phi

    c1 = a + n_samples * alpha
    c2 = b + Y.sum(0)

    pi = _rnd.beta(c1, c2)
    mu = alpha * (1 - pi) / pi

    #
    return mu


def _sample_psi_global(Y, G, mu, phi, psi, beta, tau_psi):
    n_samples, n_genes = Y.shape
    alpha = 1 / phi

    # sample proposals from a normal prior
    pi = alpha / (alpha + mu * _nmp.exp(psi))

    psi_ = _rnd.normal(G.dot(beta.T), 1 / _nmp.sqrt(tau_psi))
    pi_ = alpha / (alpha + mu * _nmp.exp(psi_))

    # compute loglik
    loglik = alpha * _nmp.log(pi) + Y * _nmp.log1p(-pi)
    loglik_ = alpha * _nmp.log(pi_) + Y * _nmp.log1p(-pi_)

    # do Metropolis step
    idxs = _rnd.rand(n_samples, n_genes) < _nmp.exp(loglik_ - loglik)
    psi[idxs] = psi_[idxs]

    #
    return psi


def _sample_psi_local(Y, G, mu, phi, psi, beta, tau_psi, scale=0.01):
    n_samples, n_genes = Y.shape
    alpha = 1 / phi
    GBT = G.dot(beta.T)

    # sample proposals from a log-normal with small scale
    pi = alpha / (alpha + _nmp.exp(psi))

    psi_ = psi * _nmp.exp(scale * _rnd.randn(n_samples, n_genes))
    pi_ = alpha / (alpha + _nmp.exp(psi_))

    # compute loglik
    loglik = alpha * _nmp.log(pi) + Y * _nmp.log1p(-pi) - 0.5 * tau_psi * (psi - GBT)**2
    loglik_ = alpha * _nmp.log(pi_) + Y * _nmp.log1p(-pi_) - 0.5 * tau_psi * (psi_ - GBT)**2

    # do Metropolis step
    idxs = _rnd.rand(n_samples, n_genes) < _nmp.exp(loglik_ - loglik)
    psi[idxs] = psi_[idxs]

    #
    return psi
