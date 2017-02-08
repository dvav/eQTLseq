"""Implements Poisson model."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.common as _cmn

_EPS = _nmp.finfo('float').eps


class ModelPoisson(object):
    """An overdispersed Poisson model estimated using Gibbs sampling."""

    def __init__(self, **args):
        n_samples, n_genes = args['Z'].shape
        _, n_markers = args['G'].shape

        # initial conditions
        logZ = _nmp.log(args['Z'] + 1)
        mu = _nmp.mean(logZ, 0)
        beta = _nmp.linalg.lstsq(args['G'], logZ - mu)[0].T
        Y = mu + args['G'].dot(beta.T)
        self.state = {
            'mu': mu,
            'tau': _nmp.ones(n_genes),
            'eta': _nmp.ones(n_markers),
            'zeta': _nmp.ones((n_genes, n_markers)),
            'Y': Y,
            'beta': beta
        }

        self.samplers = (
            self._sample_beta,
            self._sample_mu,
            self._sample_tau,
            self._sample_zeta,
            self._sample_eta,
            self._sample_Y
        )

    def _sample_beta(self, **args):
        st = self.state
        idxs_genes, idxs_markers, idxs = _cmn.get_idxs_redux(st['beta'], st['tau'], st['zeta'], st['eta'], args['beta_thr'])
        if _nmp.any(idxs):
            st['beta'][_nmp.ix_(idxs_genes, idxs_markers)] = _cmn.sample_beta(st['Y'], args['G'], st['mu'], st['tau'],
                                                                              st['zeta'], st['eta'], idxs_genes,
                                                                              idxs_markers)
        st['beta'][~idxs] = _EPS

    def _sample_mu(self, **args):
        st = self.state
        st['mu'] = _cmn.sample_mu(st['Y'], args['G'], st['beta'], st['tau'])

    def _sample_tau(self, **args):
        st = self.state
        st['tau'] = _cmn.sample_tau(st['Y'], args['G'], st['beta'], st['mu'], st['zeta'], st['eta'])

    def _sample_zeta(self, **args):
        st = self.state
        st['zeta'] = _cmn.sample_zeta(st['beta'], st['tau'], st['eta'])

    def _sample_eta(self, **args):
        st = self.state
        st['eta'] = _cmn.sample_eta(st['beta'], st['tau'], st['zeta'])

    def _sample_Y(self, **args):
        st = self.state
        _sample_Y(args['Z'], args['G'], st['Y'], st['beta'], st['mu'], st['tau'])

    def get_state(self):
        """TODO."""
        return _nmp.sqrt(_nmp.sum(self.state['beta']**2))

    @staticmethod
    def get_pred(Z, G, res):
        """TODO."""
        beta, mu = res['beta'], res['mu']
        Zhat = _nmp.exp(mu + G.dot(beta.T))

        ##
        return Zhat


def _sample_Y(Z, G, Y, beta, mu, tau):
    n_samples, n_genes = Z.shape

    # sample proposals from a normal prior
    means = _nmp.exp(Y)

    Y_ = _rnd.normal(mu + G.dot(beta.T), 1 / _nmp.sqrt(tau))
    means_ = _nmp.exp(Y_)
    means_ = _nmp.clip(means_, _EPS, 1 / _EPS)   # do this to avoid division by 0 in the log below

    # compute loglik
    loglik = Z * _nmp.log(means) - means
    loglik_ = Z * _nmp.log(means_) - means_

    # do Metropolis step
    idxs = _nmp.log(_rnd.rand(n_samples, n_genes)) < loglik_ - loglik
    Y[idxs] = Y_[idxs]
