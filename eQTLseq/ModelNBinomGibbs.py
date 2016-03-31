"""Implements ModelNBinomGibbs."""

import numpy as _nmp
import numpy.random as _rnd
import scipy.special as _spc

from eQTLseq.ModelNormalGibbs import ModelNormalGibbs as _ModelNormalGibbs


class ModelNBinomGibbs(_ModelNormalGibbs):
    """A negative binomial model estimated using Gibbs sampling."""

    def __init__(self, **args):
        """TODO."""
        Z, n_markers = args['Z'], args['n_markers']
        n_samples, n_genes = Z.shape

        super().__init__(n_genes=n_genes, n_markers=n_markers)

        # initial conditions
        self.mu_phi, self.tau_phi, self.phi, self.mu = 0, 1, _nmp.exp(_rnd.randn(n_genes)), _nmp.mean(Z, 0)

        self.Y = _rnd.randn(n_samples, n_genes)

        self.phi_sum, self.phi2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)
        self.mu_sum, self.mu2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)

    def update(self, itr, **args):
        """TODO."""
        Z, G, GTG, norm_factors = args['Z'], args['G'], args['GTG'], args['norm_factors']

        # sample mu and phi
        # self.mu = args['mu']
        # self.phi = args['phi']
        self.mu = _sample_mu(Z, norm_factors, self.phi, self.Y)
        self.phi = _sample_phi(Z, norm_factors, self.mu, self.phi, self.Y, self.mu_phi, self.tau_phi)

        # sample Y
        self.Y = _sample_Y(Z, G, norm_factors, self.mu, self.phi, self.Y, self.beta, self.tau)
        self.Y = self.Y - _nmp.mean(self.Y, 0)

        # update beta, tau, zeta and eta
        YTY = _nmp.sum(self.Y**2, 0)
        GTY = G.T.dot(self.Y)
        super().update(itr, YTY=YTY, GTG=GTG, GTY=GTY, n_burnin=args['n_burnin'], beta_thr=args['beta_thr'],
                       s2_lims=args['s2_lims'], n_samples=args['n_samples'])

        # sample mu_phi and tau_phi
        self.mu_phi, self.tau_phi = _sample_mu_tau_phi(self.phi)

        if(itr > args['n_burnin']):
            self.phi_sum += self.phi
            self.phi2_sum += self.phi**2
            self.mu_sum += self.mu
            self.mu2_sum += self.mu**2

    def get_estimates(self, **args):
        """TODO."""
        n_iters, n_burnin = args['n_iters'], args['n_burnin']

        #
        N = n_iters - n_burnin
        phi_mean, mu_mean = self.phi_sum / N, self.mu_sum / N
        phi_var, mu_var = self.phi2_sum / N - phi_mean**2, self.mu2_sum / N - mu_mean**2

        extra = super().get_estimates(n_iters=n_iters, n_burnin=n_burnin)

        return dict(
            phi=phi_mean, phi_var=phi_var,
            mu=mu_mean, mu_var=mu_var,
            **extra
        )

    def get_log_likelihood(self, **args):
        """TODO."""
        loglik = super().get_log_likelihood(Y=self.Y, G=args['G'])

        #
        return loglik


def _sample_phi_local(Z, c, mu, phi, Y, mu_phi, tau_phi, scale=0.01):
    n_samples, n_genes = Z.shape
    means = c[:, None] * mu * _nmp.exp(Y)

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


def _sample_phi_global(Z, c, mu, phi, Y, mu_phi, tau_phi):
    n_samples, n_genes = Z.shape
    means = c[:, None] * mu * _nmp.exp(Y)

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


def _sample_phi(Z, norm_factors, mu, phi, Y, mu_phi, tau_phi):
    """TODO."""
    if _rnd.rand() < 0.5:
        phi = _sample_phi_local(Z, norm_factors, mu, phi, Y, mu_phi, tau_phi)
    else:
        phi = _sample_phi_global(Z, norm_factors, mu, phi, Y, mu_phi, tau_phi)

    #
    return phi


def _sample_mu(Z, c, phi, Y, a=0.5, b=0.5):
    n_samples, _ = Z.shape

    Z = Z / (_nmp.exp(Y) * c[:, None])
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

    # sample proposals from a normal prior
    pi = alpha / (alpha + c[:, None] * mu * _nmp.exp(Y))

    Y_ = Y * _nmp.exp(scale * _rnd.randn(n_samples, n_genes))
    pi_ = alpha / (alpha + c[:, None] * mu * _nmp.exp(Y_))

    # compute loglik
    loglik = alpha * _nmp.log(pi) + Z * _nmp.log1p(-pi)
    loglik_ = alpha * _nmp.log(pi_) + Z * _nmp.log1p(-pi_)

    logprior = -0.5 * tau * (Y - GBT)**2
    logprior_ = -0.5 * tau * (Y_ - GBT)**2

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

    Y_ = _rnd.normal(G.dot(beta.T), 1 / _nmp.sqrt(tau))
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
    if _rnd.rand() < 0.5:
        Y = _sample_Y_local(Z, G, norm_factors, mu, phi, Y, beta, tau)
    else:
        Y = _sample_Y_global(Z, G, norm_factors, mu, phi, Y, beta, tau)

    #
    return Y


def _update_mu_phi(Z, c, phi, Y):
    Z = Z / _nmp.exp(Y) / c[:, None]
    mu = _nmp.mean(Z, 0)
    alpha = 1 / phi
    pi = alpha / (alpha + mu)

    f1 = (_spc.polygamma(0, Z + alpha) - _spc.polygamma(0, alpha) + _nmp.log(pi)).sum(0)
    f2 = (_spc.polygamma(1, Z + alpha) - _spc.polygamma(1, alpha) + phi * (1 - pi)).sum(0)

    # idxs = _nmp.abs(f2) < 1e-20
    # f2[idxs] = _nmp.sign(f2[idxs]) * 1e-20
    alpha = alpha - f1 / f2
    alpha = _nmp.clip(alpha, 1e-20, 1e20)

    #
    return mu, 1 / alpha


def _update_Y(Z, G, c, mu, phi, Y, beta, tau):
    alpha = 1 / phi
    GBT = G.dot(beta.T)
    pi = alpha / (alpha + c[:, None] * mu * _nmp.exp(Y))

    # compute logpost
    f1 = Z * pi - alpha * (1 - pi) - tau * (Y - GBT)
    f2 = -(alpha + Z) * pi * (1 - pi) - tau
    # f2[_nmp.abs(f2)<1e-3] = -1e-3
    # do Metropolis step
    Y = Y - f1 / f2

    #
    return Y
