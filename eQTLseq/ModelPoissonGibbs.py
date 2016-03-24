"""Implements ModelNBinomGibbs."""

import numpy as _nmp
import numpy.random as _rnd

from eQTLseq.ModelNormalGibbs import ModelNormalGibbs as _ModelNormalGibbs


class ModelPoissonGibbs(_ModelNormalGibbs):
    """A Poisson model estimated using Gibbs sampling."""

    def __init__(self, **args):
        """TODO."""
        Z, n_markers = args['Z'], args['n_markers']
        n_samples, n_genes = Z.shape

        super().__init__(n_genes=n_genes, n_markers=n_markers)

        # initial conditions
        self.mu = _nmp.mean(Z, 0)

        self.Y = _rnd.randn(n_samples, n_genes)
        self.Y = self.Y - _nmp.mean(self.Y, 0)

        self.mu_sum, self.mu2_sum = _nmp.zeros(n_genes), _nmp.zeros(n_genes)

    def update(self, itr, **args):
        """TODO."""
        Z, G, GTG, norm_factors, n_burnin, beta_thr, s2_lims, n_samples = args['Z'], args['G'], args['GTG'], \
            args['norm_factors'], args['n_burnin'], args['beta_thr'], args['s2_lims'], args['n_samples']

        # sample mu
        self.mu = _sample_mu(Z, norm_factors, self.Y)
        # self.mu = _update_mu(Z, norm_factors, self.Y)

        # sample Y
        # self.Y = args['YY']
        self.Y = _sample_Y(Z, G, norm_factors, self.mu, self.Y, self.beta, self.tau)
        self.Y = self.Y - _nmp.mean(self.Y, 0)

        # update beta, tau, zeta and eta
        YTY = _nmp.sum(self.Y**2, 0)
        GTY = G.T.dot(self.Y)
        super().update(itr, YTY=YTY, GTG=GTG, GTY=GTY,
                       n_burnin=n_burnin, beta_thr=beta_thr, s2_lims=s2_lims, n_samples=n_samples)

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
        Z, c = args['Z'], args['norm_factors']

        #
        means = c[:, None] * self.mu * _nmp.exp(self.Y)
        loglik = (Z * _nmp.log(means) - means).sum()

        #
        return loglik


def _sample_mu(Z, c, Y):
    n_samples, _ = Z.shape

    Z = Z / (_nmp.exp(Y) * c[:, None])
    shape = Z.sum(0) + 1e-6
    mu = _rnd.gamma(shape, 1 / n_samples)

    #
    return mu


def _sample_Y(Z, G, c, mu, Y, beta, tau):
    n_samples, n_genes = Z.shape

    # sample proposals from a normal prior
    means = c[:, None] * mu * _nmp.exp(Y)

    Y_ = _rnd.normal(G.dot(beta.T), 1 / _nmp.sqrt(tau))
    means_ = c[:, None] * mu * _nmp.exp(Y_)

    # compute loglik
    loglik = Z * _nmp.log(means) - means
    loglik_ = Z * _nmp.log(means_) - means_

    # do Metropolis step
    idxs = _rnd.rand(n_samples, n_genes) < _nmp.exp(loglik_ - loglik)
    Y[idxs] = Y_[idxs]

    #
    return Y


def _update_mu(Z, c, Y):
    Z = Z / (_nmp.exp(Y) * c[:, None])
    mu = _nmp.mean(Z, 0)

    #
    return mu
