"""Implements ModelBinomGibbs."""

import collections as _clt

import numpy as _nmp
import numpy.random as _rnd
import scipy.special as _spc

import eQTLseq.utils as _utils
import eQTLseq.common as _cmn

_EPS = _nmp.finfo('float').eps


class ModelBinomGibbs(object):
    """An overdispersed Binomial model estimated using Gibbs sampling."""
    State = _clt.namedtuple('State', ('mu', 'tau', 'eta', 'zeta', 'beta', 'Y'))

    def __init__(self, **args):
        """TODO."""
        n_samples, n_genes = args['Z'].shape
        _, n_markers = args['G'].shape

        # initial conditions
        p = args['Z'] / args['Z'].sum(1)[:, None]
        phat = p.sum(0) / n_samples
        mu = _nmp.log(phat) - _nmp.log1p(-phat)
        self.state = ModelBinomGibbs.State(
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
        N = Z.sum(1)
        pi = 1 / (1 + _nmp.exp(-mu - G.dot(beta.T)))
        pi = _nmp.clip(pi, _EPS, 1 - _EPS)
        Zhat = N[:, None] * pi

        pi0 = 1 / (1 + _nmp.exp(-mu))
        piF = Z / N[:, None]

        pi0 = _nmp.clip(pi0, _EPS, 1 - _EPS)
        piF = _nmp.clip(piF, _EPS, 1 - _EPS)

        C = _spc.gammaln(N + 1)[:, None] - _spc.gammaln(Z + 1) - _spc.gammaln(N[:, None] - Z + 1)
        loglik = Z * _nmp.log(pi) + (N[:, None] - Z) * _nmp.log1p(-pi) + C
        loglikF = Z * _nmp.log(piF) + (N[:, None] - Z) * _nmp.log1p(-piF) + C
        loglik0 = Z * _nmp.log(pi0) + (N[:, None] - Z) * _nmp.log1p(-pi0) + C

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
    N = Z.sum(1)

    # sample proposals from a normal prior
    pi = 1 / (1 + _nmp.exp(-Y))

    Y_ = _rnd.normal(mu + G.dot(beta.T), 1 / _nmp.sqrt(tau))
    pi_ = 1 / (1 + _nmp.exp(-Y_))

    pi = _nmp.clip(pi, _EPS, 1 - _EPS)    # bound pi/pi_ between (0,1) to avoid ...
    pi_ = _nmp.clip(pi_, _EPS, 1 - _EPS)   # divide-by-zero errors

    # compute loglik
    loglik = Z * _nmp.log(pi) + (N[:, None] - Z) * _nmp.log1p(-pi)
    loglik_ = Z * _nmp.log(pi_) + (N[:, None] - Z) * _nmp.log1p(-pi_)

    # do Metropolis step
    idxs = _nmp.log(_rnd.rand(n_samples, n_genes)) < loglik_ - loglik
    Y[idxs] = Y_[idxs]

    #
    return Y
