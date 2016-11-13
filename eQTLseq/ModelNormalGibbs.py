"""Implements ModelNormalGibbs."""

import collections as _clt

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.utils as _utils
import eQTLseq.common as _cmn

_EPS = _nmp.finfo('float').eps


class ModelNormalGibbs(object):
    """A normal model estimated using Gibbs sampling."""
    State = _clt.namedtuple('State', ('mu', 'tau', 'eta', 'zeta', 'beta'))

    def __init__(self, **args):
        """TODO."""
        _, n_genes = args['Z'].shape
        _, n_markers = args['G'].shape

        # initial conditions
        self.state = ModelNormalGibbs.State(
            mu=_nmp.mean(args['Z'], 0),
            tau=_nmp.ones(n_genes),
            eta=_nmp.ones(n_markers),
            zeta=_nmp.ones((n_genes, n_markers)),
            beta=_rnd.randn(n_genes, n_markers)
        )

        self.sums = [_nmp.zeros_like(_) for _ in self.state]
        self.sums2 = [_nmp.zeros_like(_) for _ in self.state]

    def update(self, itr, **args):
        """TODO."""
        Z, G, beta_thr, s2_lims = args['Z'], args['G'], args['beta_thr'], args['s2_lims']
        st = self.state

        # sample beta
        idxs_genes, idxs_markers = _cmn.get_idxs_redux(st.beta, st.tau, st.zeta, st.eta, beta_thr, s2_lims)
        st.beta[_nmp.ix_(idxs_genes, idxs_markers)] = \
            _cmn.sample_beta(Z, G, st.mu, st.tau, st.zeta, st.eta, idxs_genes, idxs_markers)
        st.mu[:] = _cmn.sample_mu(Z, G, st.beta, st.tau)
        st.tau[:] = _cmn.sample_tau(Z, G, st.beta, st.mu, st.zeta, st.eta, s2_lims)
        st.zeta[:, :] = _cmn.sample_zeta(st.beta, st.tau, st.eta, s2_lims)
        st.eta[:] = _cmn.sample_eta(st.beta, st.tau, st.zeta, s2_lims)

        if(itr > args['n_burnin']):
            self.sums = [s + st for s, st in zip(self.sums, self.state)]
            self.sums2 = [s2 + st**2 for s2, st in zip(self.sums2, self.state)]

    def get_estimates(self, **args):
        """TODO."""
        return _cmn.get_estimates(self.state._fields, self.sums, self.sums2, args['n_iters'] - args['n_burnin'])

    def get_state(self, **args):
        """TODO."""
        return _nmp.sqrt((self.state.beta**2).sum())

    @staticmethod
    def get_metrics(Z, G, res):
        """TODO."""
        _, n_genes = Z.shape
        beta, mu, tau = res['beta'], res['mu'], res['tau']

        # various calcs
        Zhat = mu + G.dot(beta.T)

        C = - 0.5 * _nmp.log(tau) - 0.5 * _nmp.log(2 * _nmp.pi)
        loglik = -0.5 * (Z - Zhat)**2 * tau + C
        loglikF = C
        loglik0 = -0.5 * (Z - mu)**2 * tau + C

        # metrics
        CCC = _utils.compute_ccc(Z, Zhat)
        R2 = 1 - loglik.sum() / loglik0.sum()
        NRMSD = _nmp.sqrt(_nmp.sum((Z - Zhat)**2) / Z.size) / (_nmp.max(Z) - _nmp.min(Z))
        DEV = _nmp.sum(- 2 * (loglik - loglikF)) / Z.size

        ##
        return {
            'CCC': CCC,
            'NRMSD': NRMSD,
            'R2': R2,
            'DEV': DEV
        }
