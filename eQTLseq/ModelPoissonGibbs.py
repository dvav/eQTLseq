"""Implements ModelPoissonGibbs."""

import collections as _clt

import numpy as _nmp
import numpy.random as _rnd
import scipy.special as _spc

import eQTLseq.utils as _utils
import eQTLseq.common as _cmn

_EPS = _nmp.finfo('float').eps


class ModelPoissonGibbs(object):
    """An overdispersed Poisson model estimated using Gibbs sampling."""
    State = _clt.namedtuple('State', ('mu', 'tau', 'eta', 'zeta', 'beta', 'Y'))

    def __init__(self, **args):
        """TODO."""
        n_samples, n_genes = args['Z'].shape
        _, n_markers = args['G'].shape

        # initial conditions
        mu = _nmp.mean(_nmp.log(args['Z'] + 1), 0)
        self.state = ModelPoissonGibbs.State(
            mu=mu,
            tau=_nmp.ones(n_genes),
            eta=_nmp.ones(n_markers),
            zeta=_nmp.ones((n_genes, n_markers)),
            beta=_rnd.randn(n_genes, n_markers),
            Y=mu + _rnd.randn(n_samples, n_genes)
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
            _cmn.sample_beta(st.Y, G, st.mu, st.tau, st.zeta, st.eta, idxs_genes, idxs_markers)
        st.mu[:] = _cmn.sample_mu(st.Y, G, st.beta, st.tau)
        st.tau[:] = _cmn.sample_tau(st.Y, G, st.beta, st.mu, st.zeta, st.eta, s2_lims)
        st.zeta[:, :] = _cmn.sample_zeta(st.beta, st.tau, st.eta, s2_lims)
        st.eta[:] = _cmn.sample_eta(st.beta, st.tau, st.zeta, s2_lims)
        st.Y[:, :] = _sample_Y(Z, G, st.Y, st.beta, st.mu, st.tau)

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
        beta, mu = res['beta'], res['mu']

        # various calcs
        Zhat = _nmp.exp(mu + G.dot(beta.T))

        loglik = Z * _nmp.log(Zhat + _EPS) - Zhat - _spc.gammaln(Z + 1)
        loglikF = Z * _nmp.log(Z + _EPS) - Z - _spc.gammaln(Z + 1)
        loglik0 = Z * mu - _nmp.exp(mu) - _spc.gammaln(Z + 1)

        # metrics
        lZ, lZhat = _nmp.log(Z + 1), _nmp.log(Zhat + 1)
        CCC = _utils.compute_ccc(lZ, lZhat)
        R2 = 1 - loglik.sum() / loglik0.sum()
        NRMSD = _nmp.sqrt(_nmp.sum((lZ - lZhat)**2) / Z.size) / (_nmp.max(lZ) - _nmp.min(lZ))
        DEV = _nmp.sum(- 2 * (loglik - loglikF)) / Z.size

        ##
        return {
            'CCC': CCC,
            'NRMSD': NRMSD,
            'R2': R2,
            'DEV': DEV
        }


def _sample_Y(Z, G, Y, beta, mu, tau):
    n_samples, n_genes = Z.shape
    Y = Y.copy()

    # sample proposals from a normal prior
    means = _nmp.exp(Y)

    Y_ = _rnd.normal(mu + G.dot(beta.T), 1 / _nmp.sqrt(tau))
    means_ = _nmp.exp(Y_)

    # compute loglik
    loglik = Z * _nmp.log(means + _EPS) - means     # add a small number to avoid ...
    loglik_ = Z * _nmp.log(means_ + _EPS) - means_  # ... division-by-zero errors in log

    # do Metropolis step
    idxs = _nmp.log(_rnd.rand(n_samples, n_genes)) < loglik_ - loglik
    Y[idxs] = Y_[idxs]

    #
    return Y
