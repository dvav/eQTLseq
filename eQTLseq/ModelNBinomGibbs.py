"""Implements ModelNBinomGibbs."""

import collections as _clt

import numpy as _nmp
import numpy.random as _rnd
import scipy.special as _spc

import eQTLseq.utils as _utils
import eQTLseq.common as _cmn

_EPS = _nmp.finfo('float').eps


class ModelNBinomGibbs(object):
    """A negative binomial model estimated using Gibbs sampling."""
    State = _clt.namedtuple('State', ('tau', 'eta', 'zeta', 'beta', 'mu', 'phi', 'phi_pars'))

    def __init__(self, **args):
        """TODO."""
        Z, G = args['Z'], args['G']
        _, n_genes = Z.shape
        _, n_markers = G.shape

        # initial conditions
        self.state = ModelNBinomGibbs.State(
            tau=_nmp.ones(n_genes),
            eta=_nmp.ones(n_markers),
            zeta=_nmp.ones((n_genes, n_markers)),
            beta=_rnd.randn(n_genes, n_markers) / n_markers,
            mu=_nmp.mean(Z, 0),
            phi=_nmp.exp(_rnd.randn(n_genes)),
            phi_pars=_nmp.r_[0., 1.]
        )

        self.sums = [_nmp.zeros_like(_) for _ in self.state]
        self.sums2 = [_nmp.zeros_like(_) for _ in self.state]

    def update(self, itr, **args):
        """TODO."""
        Z, G, beta_thr, s2_lims = args['Z'], args['G'], args['beta_thr'], args['s2_lims']
        st = self.state

        # update beta
        idxs_genes, idxs_markers = _cmn.get_idxs_redux(st.beta, st.tau, st.zeta, st.eta, beta_thr, s2_lims)
        st.beta[_nmp.ix_(idxs_genes, idxs_markers)] = \
            _sample_beta(Z, G, st.mu, st.phi, st.beta, st.tau, st.zeta, st.eta, idxs_genes, idxs_markers)
        st.tau[:] = _sample_tau(st.beta, st.zeta, st.eta, s2_lims)
        st.zeta[:, :] = _cmn.sample_zeta(st.beta, st.tau, st.eta, s2_lims)
        st.eta[:] = _cmn.sample_eta(st.beta, st.tau, st.zeta, s2_lims)
        st.mu[:] = _sample_mu(Z, G, st.phi, st.beta)
        st.phi[:] = _sample_phi(Z, G, st.mu, st.phi, st.beta, st.phi_pars)
        st.phi_pars[:] = _sample_phi_pars(st.phi)

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
        beta, mu, phi = res['beta'], res['mu'], res['phi']
        alpha = 1 / phi

        # various calcs
        Zhat = mu * _nmp.exp(G.dot(beta.T))

        pi = Zhat / (alpha + Zhat)
        piF = Z / (alpha + Z)
        pi0 = mu / (alpha + mu)

        pi = _nmp.clip(pi, _EPS, 1 - _EPS)
        piF = _nmp.clip(piF, _EPS, 1 - _EPS)
        pi0 = _nmp.clip(pi0, _EPS, 1 - _EPS)

        C = _spc.gammaln(Z + alpha) - _spc.gammaln(alpha) - _spc.gammaln(Z + 1)
        loglik = alpha * _nmp.log1p(-pi) + Z * _nmp.log(pi) + C
        loglikF = alpha * _nmp.log1p(-piF) + Z * _nmp.log(piF) + C
        loglik0 = alpha * _nmp.log1p(-pi0) + Z * _nmp.log(pi0) + C

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


def _sample_phi(Z, G, mu, phi, beta, phi_pars):
    n_samples, n_genes = Z.shape
    mu_phi, tau_phi = phi_pars
    phi = phi.copy()
    means = mu * _nmp.exp(G.dot(beta.T))

    # sample proposals from the prior
    alpha = 1 / phi
    pi = means / (alpha + means)

    phi_ = _nmp.exp(mu_phi + _rnd.randn(n_genes) / _nmp.sqrt(tau_phi))
    alpha_ = 1 / phi_
    pi_ = means / (alpha_ + means)

    pi = _nmp.clip(pi, _EPS, 1 - _EPS)    # bound pi/pi_ between (0,1) to avoid ...
    pi_ = _nmp.clip(pi_, _EPS, 1 - _EPS)   # divide-by-zero errors

    # compute loglik
    loglik = (_spc.gammaln(Z + alpha) - _spc.gammaln(alpha) + alpha * _nmp.log1p(-pi) + Z * _nmp.log(pi)).sum(0)
    loglik_ = (_spc.gammaln(Z + alpha_) - _spc.gammaln(alpha_) + alpha_ * _nmp.log1p(-pi_) + Z * _nmp.log(pi_)).sum(0)

    # Metropolis step
    idxs = _nmp.log(_rnd.rand(n_genes)) < loglik_ - loglik
    phi[idxs] = phi_[idxs]

    #
    return phi


def _sample_mu(Z, G, phi, beta, a=0.5, b=0.5):
    n_samples, _ = Z.shape

    Z = Z * _nmp.exp(-G.dot(beta.T))
    alpha = 1 / phi

    c1 = a + Z.sum(0)
    c2 = b + n_samples * alpha

    pi = _rnd.beta(c1, c2)
    mu = alpha * pi / (1 - pi)

    #
    return mu


def _sample_phi_pars(phi):
    n_genes = phi.size
    log_phi = _nmp.log(phi)
    log_phi2 = log_phi**2

    # sample tau_phi
    shape = 0.5 * n_genes
    rate = 0.5 * log_phi2.sum()
    tau_phi = _rnd.gamma(shape, 1 / rate)

    # sample mu_phi, given tau_phi
    mean = log_phi.sum() / n_genes
    prec = n_genes * tau_phi
    mu_phi = _rnd.normal(mean, 1 / _nmp.sqrt(prec))

    #
    return mu_phi, tau_phi


def _sample_beta(Z, G, mu, phi, beta, tau, zeta, eta, idxs_genes, idxs_markers):
    """TODO."""
    # get reduced parameters and data
    Z = Z[:, idxs_genes]
    G = G[:, idxs_markers]

    mu = mu[idxs_genes]
    phi = phi[idxs_genes]
    tau = tau[idxs_genes]
    eta = eta[idxs_markers]
    zeta = zeta[idxs_genes, :][:, idxs_markers]
    beta = beta[idxs_genes, :][:, idxs_markers]

    _, n_markers = G.shape

    # sample beta
    alpha = 1 / phi
    x0 = _nmp.log(mu) - _nmp.log(alpha)

    omega = _utils.sample_PG(Z + alpha, x0 + G.dot(beta.T))

    theta = tau[:, None] * zeta * eta
    A1 = _nmp.dot(omega.T[:, None, :] * G.T, G)
    A2 = theta[:, :, None] * _nmp.identity(n_markers)
    A = A1 + A2
    b = 0.5 * G.T.dot(Z - alpha - 2 * omega * x0)
    beta = _utils.sample_multivariate_normal_many(b.T, A)

    ##
    return beta


def _sample_tau(beta, zeta, eta, s2_lims):
    """TODO."""
    _, n_markers = beta.shape

    # sample zeta
    shape = 0.5 * n_markers
    rate = 0.5 * (eta * beta**2 * zeta).sum(1)
    tau = _rnd.gamma(shape, 1 / rate)

    ##
    return _nmp.clip(tau, 1 / s2_lims[1], 1 / s2_lims[0])
