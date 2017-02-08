"""Implements Normal model."""

import numpy as _nmp

import eQTLseq.common as _cmn

_EPS = _nmp.finfo('float').eps


class ModelNormal(object):
    """A normal model estimated using Gibbs sampling."""

    def __init__(self, **args):
        _, n_genes = args['Z'].shape
        _, n_markers = args['G'].shape

        # initial conditions
        mu = _nmp.mean(args['Z'], 0)
        beta = _nmp.linalg.lstsq(args['G'], args['Z'] - mu)[0].T
        self.state = {
            'mu': mu,
            'tau': 1/_nmp.var(args['Z'], 0),
            'eta': _nmp.ones(n_markers),
            'zeta': _nmp.ones((n_genes, n_markers)),
            'beta': beta
        }

        # samplers
        self.samplers = (
            self._sample_beta,
            self._sample_mu,
            self._sample_tau,
            self._sample_zeta,
            self._sample_eta
        )

    def _sample_beta(self, **args):
        st = self.state
        idxs_genes, idxs_markers, idxs = _cmn.get_idxs_redux(st['beta'], st['tau'], st['zeta'], st['eta'], args['beta_thr'])
        if _nmp.any(idxs):
            st['beta'][_nmp.ix_(idxs_genes, idxs_markers)] = _cmn.sample_beta(args['Z'], args['G'], st['mu'], st['tau'],
                                                                              st['zeta'], st['eta'], idxs_genes,
                                                                              idxs_markers)
        st['beta'][~idxs] = _EPS

    def _sample_mu(self, **args):
        st = self.state
        st['mu'] = _cmn.sample_mu(args['Z'], args['G'], st['beta'], st['tau'])

    def _sample_tau(self, **args):
        st = self.state
        st['tau'] = _cmn.sample_tau(args['Z'], args['G'], st['beta'], st['mu'], st['zeta'], st['eta'])

    def _sample_zeta(self, **args):
        st = self.state
        st['zeta'] = _cmn.sample_zeta(st['beta'], st['tau'], st['eta'])

    def _sample_eta(self, **args):
        st = self.state
        st['eta'] = _cmn.sample_eta(st['beta'], st['tau'], st['zeta'])

    def get_state(self):
        """TODO."""
        return _nmp.sqrt(_nmp.sum(self.state['beta']**2))

    @staticmethod
    def get_pred(Z, G, res):
        """TODO."""
        beta, mu = res['beta'], res['mu']
        Zhat = mu + G.dot(beta.T)

        #
        return Zhat
