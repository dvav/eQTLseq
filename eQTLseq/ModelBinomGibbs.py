"""Implements ModelBinomGibbs."""

import numpy as _nmp
import numpy.random as _rnd

from eQTLseq.ModelNormalStdGibbs import ModelNormalStdGibbs as _ModelNormalStdGibbs


class ModelBinomGibbs(_ModelNormalStdGibbs):
    """An overdispersed Binomial model estimated using Gibbs sampling."""

    def __init__(self, **args):
        """TODO."""
        Z, n_markers = args['Z'], args['n_markers']
        n_samples, n_genes = Z.shape

        super().__init__(n_genes=n_genes, n_markers=n_markers)

        # initial conditions
        self.mu = _nmp.mean(Z, 0)
        self.Y = _rnd.randn(n_samples, n_genes)
        self.mu_sum, self.mu2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)

    def update(self, itr, **args):
        """TODO."""
        Z, G, GTG, lib_sizes, n_burnin, beta_thr, s2_lims = args['Z'], args['G'], args['GTG'], \
            args['lib_sizes'], args['n_burnin'], args['beta_thr'], args['s2_lims']

        # sample mu
        self.mu = _sample_mu(Z, self.Y)

        # sample Y
        # self.Y = args['YY']
        # if _rnd.rand() < 0.5:
        #     self.Y = _sample_Y_local(Z, G, norm_factors, self.mu, self.Y, self.beta)
        # else:
        self.Y = _sample_Y_global(Z, G, lib_sizes, self.mu, self.Y, self.beta)
        self.Y = self.Y - _nmp.mean(self.Y, 0)

        # update beta, tau, zeta and eta
        YTY = _nmp.sum(self.Y**2, 0)
        GTY = G.T.dot(self.Y)
        super().update(itr, YTY=YTY, GTG=GTG, GTY=GTY, n_burnin=n_burnin, beta_thr=beta_thr, s2_lims=s2_lims)

        if(itr > n_burnin):
            self.mu_sum += self.mu
            self.mu2_sum += self.mu**2

    def get_estimates(self, **args):
        """TODO."""
        n_iters, n_burnin = args['n_iters'], args['n_burnin']

        #
        N = n_iters - n_burnin
        mu_mean = self.mu_sum / N
        mu_var = self.mu2_sum / N - mu_mean**2

        extra = super().get_estimates(n_iters=n_iters, n_burnin=n_burnin)

        return dict(
            mu=mu_mean, mu_var=mu_var,
            **extra
        )

    def get_log_likelihood(self, **args):
        """TODO."""
        loglik = super().get_log_likelihood(Y=self.Y, G=args['G'])

        #
        return loglik


def _sample_mu(Z, Y, a0=0.5, b0=0.5):
    Z = Z * _nmp.exp(-Y)
    n = Z.sum(1)
    s = Z.sum(0)

    a = a0 + s
    b = b0 + (n[:, None] - Z).sum(0)
    pi = _rnd.beta(a, b)
    mu = pi / (1 - pi)

    #
    return mu


def _sample_Y_local(Z, G, c, mu, Y, beta, scale=0.01):
    n_samples, n_genes = Z.shape
    GBT = G.dot(beta.T)

    # sample proposals from a normal prior
    means = c[:, None] * mu * _nmp.exp(Y)

    Y_ = Y * _nmp.exp(scale * _rnd.randn(n_samples, n_genes))
    means_ = c[:, None] * mu * _nmp.exp(Y_)

    # compute loglik
    logpost = Z * _nmp.log(means) - means - 0.5 * (Y - GBT)**2
    logpost_ = Z * _nmp.log(means_) - means_ - 0.5 * (Y_ - GBT)**2

    # do Metropolis step
    idxs = _rnd.rand(n_samples, n_genes) < _nmp.exp(logpost_ - logpost)
    Y[idxs] = Y_[idxs]

    #
    return Y


def _sample_Y_global(Z, G, n, mu, Y, beta):
    n_samples, n_genes = Z.shape

    # sample proposals from a normal prior
    pi = mu / (mu + _nmp.exp(-Y))

    Y_ = _rnd.normal(G.dot(beta.T), 1)
    pi_ = mu / (mu + _nmp.exp(-Y_))

    # compute loglik
    loglik = Z * _nmp.log(pi) + (n[:, None] - Z) * _nmp.log(1 - pi)
    loglik_ = Z * _nmp.log(pi_) + (n[:, None] - Z) * _nmp.log(1 - pi_)

    # do Metropolis step
    idxs = _rnd.rand(n_samples, n_genes) < _nmp.exp(loglik_ - loglik)
    Y[idxs] = Y_[idxs]

    #
    return Y
