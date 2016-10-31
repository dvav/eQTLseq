"""Implements ModelPoissonGibbs."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.utils as _utils

from eQTLseq.ModelNormalGibbs import ModelNormalGibbs as _ModelNormalGibbs

_EPS = _nmp.finfo('float').eps


class ModelPoissonGibbs(_ModelNormalGibbs):
    """An overdispersed Poisson model estimated using Gibbs sampling."""

    def __init__(self, **args):
        """TODO."""
        super().__init__(**args)

        Z = args['Z']
        n_samples, n_genes = Z.shape

        # initial conditions
        self.Y = _rnd.randn(n_samples, n_genes)

        self.mu = _nmp.mean(Z * _nmp.exp(-self.Y), 0)
        self.mu_sum, self.mu2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)
        self.Y_sum, self.Y2_sum = _nmp.zeros((n_samples, n_genes)), _nmp.zeros((n_samples, n_genes))

    def update(self, itr, **args):
        """TODO."""
        Z, G = args['Z'], args['G']

        # update beta, tau, zeta and eta
        YTY = _nmp.sum(self.Y**2, 0)
        GTY = G.T.dot(self.Y)
        super().update(itr, **{**args, 'YTY': YTY, 'GTY': GTY})

        # sample Y
        self.Y = _sample_Y(Z, G, self.mu, self.Y, self.beta, self.tau)
        self.Y = self.Y - _nmp.mean(self.Y, 0)

        # sample mu
        self.mu = _sample_mu(Z, self.Y)

        if(itr > args['n_burnin']):
            self.Y_sum += self.Y
            self.Y2_sum += self.Y**2
            self.mu_sum += self.mu
            self.mu2_sum += self.mu**2

    def get_estimates(self, **args):
        """TODO."""
        n_iters, n_burnin = args['n_iters'], args['n_burnin']

        #
        N = n_iters - n_burnin
        mu_mean, Y_mean = self.mu_sum / N, self.Y_sum / N
        mu_var, Y_var = self.mu2_sum / N - mu_mean**2, self.Y2_sum / N - Y_mean**2

        extra = super().get_estimates(n_iters=n_iters, n_burnin=n_burnin)

        return {
            'mu': mu_mean, 'mu_var': mu_var,
            'Y': Y_mean.T, 'Y_var': Y_var.T,
            **extra
        }

    @staticmethod
    def get_error(Z, G, res):
        """TODO."""
        _, n_genes = Z.shape

        beta = res['beta']
        mu = res['mu']

        Yhat = G.dot(beta.T)
        Yhat = Yhat - _nmp.mean(Yhat, 0)
        Zhat = mu * _nmp.exp(Yhat)

        # Z = _nmp.c_[Z, Zhat]
        # Z = _utils.blom(Z.T).T
        # Z, Zhat = Z[:, :n_genes], Z[:, n_genes:]

        ##
        return ((_nmp.log(Z + 1) - _nmp.log(Zhat + 1))**2).sum() / Z.size

    # @staticmethod
    # def get_error(Z, G, res):
    #     """TODO."""
    #     _, n_genes = Z.shape
    #
    #     beta = res['beta']
    #     mu = res['mu']
    #
    #     Yhat = G.dot(beta.T)
    #     Yhat = Yhat - _nmp.mean(Yhat, 0)
    #     Zhat = mu * _nmp.exp(Yhat)
    #     s2 = Zhat
    #
    #     ##
    #     return ((Z - Zhat)**2 / s2).sum() / Z.size


def _sample_mu(Z, Y):
    n_samples, _ = Z.shape
    Z = Z * _nmp.exp(-Y)
    mu = _rnd.gamma(Z.sum(0), 1 / n_samples)

    #
    return mu


def _sample_Y(Z, G, mu, Y, beta, tau):
    n_samples, n_genes = Z.shape

    # sample proposals from a normal prior
    means = mu * _nmp.exp(Y)

    Y_ = _rnd.normal(G.dot(beta.T), 1 / _nmp.sqrt(tau))
    means_ = mu * _nmp.exp(Y_)

    # compute loglik
    loglik = Z * _nmp.log(means + _EPS) - means     # add a small number to avoid ...
    loglik_ = Z * _nmp.log(means_ + _EPS) - means_  # ... division-by-zero errors in log

    # do Metropolis step
    idxs = _nmp.log(_rnd.rand(n_samples, n_genes)) < loglik_ - loglik
    Y[idxs] = Y_[idxs]

    #
    return Y
