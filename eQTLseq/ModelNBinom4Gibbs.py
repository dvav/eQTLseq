"""Implements ModelNBinomGibbs."""

import numpy as _nmp
import numpy.random as _rnd
import scipy.special as _spc

from eQTLseq.ModelNormalGibbs import ModelNormalGibbs as _ModelNormalGibbs


class ModelNBinom4Gibbs(_ModelNormalGibbs):
    """A negative binomial model estimated using Gibbs sampling."""

    def __init__(self, **args):
        """TODO."""
        super().__init__(**args)

        Z = args['Z']
        n_samples, n_genes = Z.shape

        # initial conditions
        self.Y = _rnd.randn(n_samples, n_genes)

        self.mu_phi, self.tau_phi, self.phi = 0, 1, _nmp.exp(_rnd.randn(n_genes))
        self.mu = _nmp.mean(Z * _nmp.exp(-self.Y), 0)

        self.phi_sum, self.phi2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)
        self.mu_sum, self.mu2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)

    def update(self, itr, **args):
        """TODO."""
        Z, G = args['Z'], args['G']

        # sample mu and phi
        self.mu = _sample_mu(Z, self.phi, self.Y)
        self.phi = _sample_phi(Z, G, self.mu, self.phi, self.Y, self.mu_phi, self.tau_phi)

        # sample mu_phi and tau_phi
        self.mu_phi, self.tau_phi = _sample_mu_tau_phi(self.phi)

        # sample Y
        self.Y = _sample_Y(Z, G, self.mu, self.phi, self.Y, self.beta, self.tau)
        self.Y = (self.Y - _nmp.mean(self.Y, 0)) / _nmp.std(self.Y, 0) if args['scale'] \
            else (self.Y - _nmp.mean(self.Y, 0))

        # update beta, tau, zeta and eta
        YTY = _nmp.sum(self.Y**2, 0)
        GTY = G.T.dot(self.Y)
        super().update(itr, YTY=YTY, GTY=GTY, **args)

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

        return {'phi': phi_mean, 'phi_var': phi_var, 'mu': mu_mean, 'mu_var': mu_var, **extra}

    def get_state(self, **args):
        """TODO."""
        return super().get_state()


def _sample_phi(Z, G, mu, phi, Y, mu_phi, tau_phi):
    n_samples, n_genes = Z.shape
    means = mu * _nmp.exp(Y)

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
    diff = loglik_ - loglik
    diff[diff > 100] = 100  # avoid overflows in exp below
    idxs = _rnd.rand(n_genes) < _nmp.exp(diff)
    phi[idxs] = phi_[idxs]

    #
    return phi


def _sample_mu(Z, phi, Y, a=0.5, b=0.5):
    n_samples, _ = Z.shape

    Z = Z * _nmp.exp(-Y)
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


def _sample_Y(Z, G, mu, phi, Y, beta, tau):
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
    diff = loglik_ - loglik
    diff[diff > 100] = 100  # avoid overflows in exp below
    idxs = _rnd.rand(n_samples, n_genes) < _nmp.exp(diff)
    Y[idxs] = Y_[idxs]

    #
    return Y
