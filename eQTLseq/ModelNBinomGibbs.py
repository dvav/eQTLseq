"""Implements ModelNBinomGibbs."""

import numpy as _nmp
import numpy.random as _rnd
import scipy.special as _spc

from eQTLseq.ModelNormalGibbs import ModelNormalGibbs as _ModelNormalGibbs


class ModelNBinomGibbs(_ModelNormalGibbs):
    """A negative binomial model estimated using Gibbs sampling."""

    def __init__(self, **args):
        """TODO."""
        Z, n_markers = args['Y'], args['n_markers']
        n_samples, n_genes = Z.shape

        super().__init__(n_genes=n_genes, n_markers=n_markers)

        # initial conditions
        self.mu_phi, self.tau_phi, self.phi = 0, 1, _nmp.exp(_rnd.randn(n_genes))
        self.Y = _rnd.randn(n_samples, n_genes)

        self.phi_sum, self.phi2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)

    def update(self, itr, **args):
        """TODO."""
        Z, G, GTG, n_burnin, beta_thr, s2_lims, n_samples = args['Y'], args['G'], args['GTG'], args['n_burnin'], \
            args['beta_thr'], args['s2_lims'], args['n_samples']

        # update beta, tau, zeta and eta
        YTY = _nmp.sum(self.Y**2, 0)
        GTY = G.T.dot(self.Y)
        super().update(itr, YTY=YTY, GTG=GTG, GTY=GTY,
                       n_burnin=n_burnin, beta_thr=beta_thr, s2_lims=s2_lims, n_samples=n_samples)

        # sample phi
        if(_rnd.rand() < 0.5):
            self.phi = _sample_phi_local(Z, self.phi, self.Y, self.mu_phi, self.tau_phi)
        else:
            self.phi = _sample_phi_global(Z, self.phi, self.Y, self.mu_phi, self.tau_phi)

        # sample Y
        if(_rnd.rand() < 0.5):
            self.Y = _sample_Y_local(Z, G, self.phi, self.Y, self.beta, self.tau)
        else:
            self.Y = _sample_Y_global(Z, G, self.phi, self.Y, self.beta, self.tau)

        # sample mu_phi and tau_phi
        self.mu_phi, self.tau_phi = _sample_mu_tau_phi(self.phi)

        if(itr > n_burnin):
            self.phi_sum += self.phi
            self.phi2_sum += self.phi**2

    def get_estimates(self, **args):
        """TODO."""
        n_iters, n_burnin = args['n_iters'], args['n_burnin']

        #
        N = n_iters - n_burnin
        phi_mean = self.phi_sum / N
        phi_var = self.phi2_sum / N - phi_mean**2

        extra = super().get_estimates(n_iters=n_iters, n_burnin=n_burnin)

        return dict(
            phi=phi_mean, phi_var=phi_var,
            **extra
        )

    def get_joint_log_likelihood(self, a=0.5, b=0.5, **args):
        """TODO."""
        Z, G = args['Y'], args['G']

        # number of samples and markers
        n_samples, n_markers = G.shape
        _, n_genes = Z.shape

        #
        log_phi = _nmp.log(self.phi)
        alpha = 1 / self.phi
        pi = alpha / (alpha + _nmp.exp(self.Y))

        A = (_spc.gammaln(Z + alpha) - _spc.gammaln(alpha) + alpha * _nmp.log(pi) + Z * _nmp.log1p(-pi)).sum()
        B = -(log_phi + 0.5 * self.tau_phi * (log_phi - self.mu_phi)**2).sum()
        C = super().get_joint_log_likelihood(Y=self.Y, G=G)

        #
        return A + B + C


def _sample_phi_global(Z, phi, Y, mu_phi, tau_phi):
    n_samples, n_genes = Z.shape
    means = _nmp.exp(Y)

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


def _sample_phi_local(Z, phi, Y, mu_phi, tau_phi, scale=0.01):
    n_samples, n_genes = Z.shape
    means = _nmp.exp(Y)

    # sample proposals from a log-normal with small scale
    alpha = 1 / phi
    pi = alpha / (alpha + means)
    log_phi = _nmp.log(phi)

    phi_ = phi * _nmp.exp(scale * _rnd.randn(n_genes))
    alpha_ = 1 / phi_
    pi_ = alpha_ / (alpha_ + means)
    log_phi_ = _nmp.log(phi_)

    # compute logpost
    loglik = (_spc.gammaln(Z + alpha) - _spc.gammaln(alpha) + alpha * _nmp.log(pi) + Z * _nmp.log1p(-pi)).sum(0)
    loglik_ = (_spc.gammaln(Z + alpha_) - _spc.gammaln(alpha_) + alpha_ * _nmp.log(pi_) + Z * _nmp.log1p(-pi_)).sum(0)

    logprior = -log_phi - 0.5 * tau_phi * (log_phi - mu_phi)**2
    logprior_ = -log_phi_ - 0.5 * tau_phi * (log_phi_ - mu_phi)**2

    logpost = loglik + logprior
    logpost_ = loglik_ + logprior_

    # Metropolis step
    idxs = _rnd.rand(n_genes) < _nmp.exp(logpost_ - logpost)
    phi[idxs] = phi_[idxs]

    #
    return phi


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


def _sample_Y_global(Z, G, phi, Y, beta, tau):
    n_samples, n_genes = Z.shape
    alpha = 1 / phi

    # sample proposals from a normal prior
    pi = alpha / (alpha + _nmp.exp(Y))

    Y_ = _rnd.normal(G.dot(beta.T), 1 / _nmp.sqrt(tau))
    pi_ = alpha / (alpha + _nmp.exp(Y_))

    # compute loglik
    loglik = alpha * _nmp.log(pi) + Z * _nmp.log1p(-pi)
    loglik_ = alpha * _nmp.log(pi_) + Z * _nmp.log1p(-pi_)

    # do Metropolis step
    idxs = _rnd.rand(n_samples, n_genes) < _nmp.exp(loglik_ - loglik)
    Y[idxs] = Y_[idxs]

    #
    return Y


def _sample_Y_local(Z, G, phi, Y, beta, tau, scale=0.01):
    n_samples, n_genes = Z.shape
    alpha = 1 / phi
    GBT = G.dot(beta.T)

    # sample proposals from a log-normal with small scale
    pi = alpha / (alpha + _nmp.exp(Y))

    Y_ = Y * _nmp.exp(scale * _rnd.randn(n_samples, n_genes))
    pi_ = alpha / (alpha + _nmp.exp(Y_))

    # compute logpost
    logpost = alpha * _nmp.log(pi) + Z * _nmp.log1p(-pi) - 0.5 * tau * (Y - GBT)**2
    logpost_ = alpha * _nmp.log(pi_) + Z * _nmp.log1p(-pi_) - 0.5 * tau * (Y_ - GBT)**2

    # do Metropolis step
    idxs = _rnd.rand(n_samples, n_genes) < _nmp.exp(logpost_ - logpost)
    Y[idxs] = Y_[idxs]

    #
    return Y
