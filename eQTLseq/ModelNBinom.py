"""Implements Negative Binomial model."""

import numpy as _nmp

import eQTLseq.common as _cmn
import eQTLseq.common_nb as _cnb

_EPS = _nmp.finfo('float').eps


class ModelNBinom(object):
    """A negative binomial model estimated using Gibbs sampling."""

    def __init__(self, **args):
        Z, G = args['Z'], args['G']
        _, n_genes = Z.shape
        _, n_markers = G.shape

        mu = _nmp.mean(Z, 0)
        s2 = _nmp.var(Z, 0)
        phi = _nmp.maximum(_EPS, (s2 - mu) / mu**2)

        # initial conditions
        b = _nmp.log(args['Z'] + 1) - _nmp.log(mu)
        beta = _nmp.linalg.lstsq(args['G'], b)[0].T
        self.state = {
            'tau': _nmp.ones(n_genes),
            'eta': _nmp.ones(n_markers),
            'zeta': _nmp.ones((n_genes, n_markers)),
            'beta': beta,
            'mu': mu,
            'phi': phi,
            'mu_phi': _nmp.mean(_nmp.log(phi)),
            'tau_phi': 1/_nmp.var(_nmp.log(phi))
        }

        self.samplers = (
            self._sample_beta,
            self._sample_mu,
            self._sample_phi,
            self._sample_tau,
            self._sample_zeta,
            self._sample_eta,
        )

    def _sample_beta(self, **args):
        st = self.state
        idxs_genes, idxs_markers, idxs = _cmn.get_idxs_redux(st['beta'], st['tau'], st['zeta'], st['eta'], args['beta_thr'])
        if _nmp.any(idxs):
            st['beta'][_nmp.ix_(idxs_genes, idxs_markers)] = _cnb.sample_beta(args['Z'], args['G'], st['mu'], st['phi'],
                                                                              st['beta'], st['tau'], st['zeta'],
                                                                              st['eta'], idxs_genes, idxs_markers)
        st['beta'][~idxs] = _EPS

    def _sample_tau(self, **args):
        st = self.state
        st['tau'] = _cnb.sample_tau(st['beta'], st['zeta'], st['eta'])

    def _sample_zeta(self, **args):
        st = self.state
        st['zeta'] = _cmn.sample_zeta(st['beta'], st['tau'], st['eta'])

    def _sample_eta(self, **args):
        st = self.state
        st['eta'] = _cmn.sample_eta(st['beta'], st['tau'], st['zeta'])

    def _sample_mu(self, **args):
        st = self.state
        st['mu'] = _cnb.sample_mu(args['Z'], args['G'], st['phi'], st['beta'])

    def _sample_phi(self, **args):
        st = self.state
        _cnb.sample_phi(args['Z'], args['G'], st['mu'], st['phi'], st['beta'], st['mu_phi'], st['tau_phi'])
        st['mu_phi'], st['tau_phi'] = _cnb.sample_mu_tau_phi(st['phi'])

    def get_state(self):
        """TODO."""
        return _nmp.sqrt(_nmp.sum(self.state['beta']**2))

    @staticmethod
    def get_pred(Z, G, res):
        """TODO."""
        beta, mu = res['beta'], res['mu']
        Zhat = mu * _nmp.exp(G.dot(beta.T))

        #
        return Zhat
