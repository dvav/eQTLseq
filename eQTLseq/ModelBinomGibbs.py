"""Implements ModelBinomGibbs."""

import numpy as _nmp
import numpy.random as _rnd

from eQTLseq.ModelNormalGibbs import ModelNormalGibbs as _ModelNormalGibbs


class ModelBinomGibbs(_ModelNormalGibbs):
    """An overdispersed Binomial model estimated using Gibbs sampling."""

    def __init__(self, **args):
        """TODO."""
        super().__init__(**args)

        Z, c = args['Z'], args['norm_factors']
        n_samples, n_genes = Z.shape

        # initial conditions
        self.Y = _rnd.randn(n_samples, n_genes)

        self.mu = _nmp.mean(Z / c[:, None] * _nmp.exp(-self.Y), 0)
        self.mu_sum, self.mu2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)

    def update(self, itr, **args):
        """TODO."""
        Z, G, norm_factors = args['Z'], args['G'], args['norm_factors']

        # sample mu
        self.mu = _sample_mu(Z, norm_factors, self.Y)

        # sample Y
        self.Y = _sample_Y(Z, G, norm_factors, self.mu, self.Y, self.beta, self.tau)
        self.Y = self.Y - _nmp.mean(self.Y, 0)

        # update beta, tau, zeta and eta
        YTY = _nmp.sum(self.Y**2, 0)
        GTY = G.T.dot(self.Y)
        super().update(itr, YTY=YTY, GTY=GTY, **args)

        if(itr > args['n_burnin']):
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

        return {'mu': mu_mean, 'mu_var': mu_var, **extra}

    def get_log_likelihood(self, **args):
        """TODO."""
        return super().get_state()


def _sample_mu(Z, c, Y, a0=0.5, b0=0.5):
    Z = Z / c[:, None] * _nmp.exp(-Y)
    n = Z.sum(1)
    s = Z.sum(0)

    a = a0 + s
    b = b0 + (n[:, None] - Z).sum(0)
    pi = _rnd.beta(a, b)
    mu = pi / (1 - pi)

    #
    return mu


def _sample_Y(Z, G, c, mu, Y, beta, tau):
    n_samples, n_genes = Z.shape
    Z = Z / c[:, None]
    n = Z.sum(1)

    # sample proposals from a normal prior
    pi = mu / (mu + _nmp.exp(-Y))

    Y_ = _rnd.normal(G.dot(beta.T), 1 / _nmp.sqrt(tau))
    pi_ = mu / (mu + _nmp.exp(-Y_))

    # compute loglik
    loglik = Z * _nmp.log(pi) + (n[:, None] - Z) * _nmp.log1p(-pi)
    loglik_ = Z * _nmp.log(pi_) + (n[:, None] - Z) * _nmp.log1p(-pi_)

    # do Metropolis step
    idxs = _rnd.rand(n_samples, n_genes) < _nmp.exp(loglik_ - loglik)
    Y[idxs] = Y_[idxs]

    #
    return Y
