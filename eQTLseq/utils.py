"""Implements various utility functions."""

import sys as _sys

import numpy as _nmp
import numpy.random as _rnd
import scipy.linalg as _lin
import scipy.optimize as _opt
import scipy.special as _spc
import scipy.stats as _stats


def sample_multivariate_normal_one(U, b, z):
    """TODO."""
    y = _lin.solve_triangular(U, z)  # U * y = z
    mu_ = _lin.solve_triangular(U, b, trans='T')  # U.T * mu_ = b, where mu_ = U * mu
    mu = _lin.solve_triangular(U, mu_)  # U * mu = mu_

    #
    return y + mu


def sample_multivariate_normal_many(b, A):
    """Sample from the multivariate normal distribution with multiple precision matrices A and mu = A^-1 b."""
    z = _rnd.normal(size=b.shape)

    L = _nmp.linalg.cholesky(A)
    U = _nmp.transpose(L, axes=(0, 2, 1))

    y = [sample_multivariate_normal_one(U_, b_, z_) for U_, b_, z_ in zip(U, b, z)]

    # return
    return _nmp.asarray(y)


def sample_nbinom(mu, phi, size=None):
    """Sample from the Negative Binomial distribution with mean `mu` and dispersion `phi`."""
    # sample lambdas from gamma and, then, counts from Poisson
    shape = 1 / phi
    scale = mu * phi
    lam = _rnd.gamma(shape, scale, size=size)
    counts = _rnd.poisson(lam, size=size)

    #
    return counts


def calculate_norm_factors(read_counts, locfcn=_nmp.median):
    """Normalise RNA-seq counts data using the Relative Log Expression (RLE) method, as in DESeq."""
    # compute geometric mean of each row in log-scale
    logcounts = _nmp.log(read_counts)
    logmeans = _nmp.mean(logcounts, 1)

    # take the ratios
    logcounts -= logmeans[:, None]

    # get median (or other central tendency metric) of ratios excluding rows with 0 mean
    logcounts = logcounts[_nmp.isfinite(logmeans), :]
    norm_factors = _nmp.exp(locfcn(logcounts, 0))

    #
    return norm_factors


def fit_nbinom_model(read_counts, normalised=False):
    """Fit a Negative Binomial model to a table of RNA-seq count data using maximum likelihood estimation."""
    # prepare data
    n_genes, n_samples = read_counts.shape

    def fcn(alpha, ydata, ymean):
        return _spc.psi(ydata + alpha).sum() - n_samples * _spc.psi(alpha) + n_samples * _nmp.log(alpha) \
            - n_samples * _nmp.log(ymean + alpha)

    # iterate over genes and fit across samples
    ydata = read_counts if normalised else read_counts / calculate_norm_factors(read_counts)
    ymean = _nmp.mean(ydata, 1)
    alpha = _nmp.zeros(n_genes)
    converged = _nmp.zeros(n_genes, dtype=bool)
    for i in range(n_genes):
        try:  # find a better way to set a and b
            tmp = _opt.brentq(fcn, 1e-6, 1e6, args=(ydata[i, :], ymean[i]), full_output=True)
        except Exception:
            alpha[i] = _nmp.nan
            converged[i] = False
        else:
            alpha[i] = tmp[0]
            converged[i] = tmp[1].converged

        print('\rFitting gene {0} of {1}'.format(i, n_genes), end='', file=_sys.stderr)
    print('', file=_sys.stderr)

    #
    return {
        'mu': ymean,
        'phi': 1 / alpha,
        'converged': converged
    }


def blom(Z, c=3/8):
    """TODO."""
    _, N = Z.shape
    R = _nmp.asarray([_stats.rankdata(_) for _ in Z])
    P = (R - c) / (N - 2 * c + 1)
    Y = _nmp.sqrt(2) * _spc.erfinv(2 * P - 1)    # probit function

    #
    return Y


def transform_data(Z, norm_factors, kind='Blom'):
    """TODO."""
    assert kind in ('Blom', 'BoxCox', 'Log')

    fcn = {
        'Log': lambda x: _nmp.log(x + 1),
        'BoxCox': lambda x: _nmp.asarray([_stats.boxcox(_ + 1)[0] for _ in x]),
        'Blom': lambda x: blom(x + _rnd.rand(*x.shape)*1e-6)  # add small random numbers to avoid spurious ties
    }[kind]

    Z = Z / norm_factors[:, None]
    Y = fcn(Z)

    #
    return Y


def simulate_genotypes(n_samples=1000, n_markers=100, MAF_range=(0.05, 0.5)):
    """Generate a matrix of genotypes, using a binomial model."""
    # compute MAFs for each genetic marker and compute genotypes
    MAF = _rnd.uniform(MAF_range[0], MAF_range[1], n_markers)
    G = _rnd.binomial(2, MAF, (n_samples, n_markers))   # assume ploidy=2

    # drop mono-morphic markers
    G = G[:, _nmp.std(G, 0) > 0]

    #
    return {'G': G, 'MAF': MAF}


def simulate_eQTLs_normal(G, n_markers_causal, n_genes, n_genes_affected, s2e, h2):
    """Simulate eQTLs with normally distributed gene expression data."""
    _, n_markers = G.shape

    # sample causal markers and affected genes
    idxs_markers_causal = _rnd.choice(n_markers, n_markers_causal, replace=False)
    idxs_genes_affected = _nmp.hstack([
        _rnd.choice(n_genes, (n_genes_affected, 1), replace=False) for _ in range(n_markers_causal)
    ])

    # compute causal coefficients
    s2g = h2 * s2e / (1 - h2)
    beta = _nmp.zeros((n_genes, n_markers))
    beta[idxs_genes_affected, idxs_markers_causal] = \
        _rnd.normal(0, _nmp.sqrt(s2g / n_markers_causal), (n_genes_affected, n_markers_causal))

    # compute phenotype
    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0)
    Y = _rnd.normal(G.dot(beta.T), _nmp.sqrt(s2e))

    #
    return {'Y': Y, 'beta': beta}


def simulate_eQTLs_nbinom(G, mu, phi, n_markers_causal=2, n_genes=None, n_genes_affected=10, s2e=1, h2=0.5):
    """Simulate eQTLs with negative binomially distributed gene expression data."""
    _, n_markers = G.shape
    n_genes = phi.size if n_genes is None else n_genes

    assert n_markers > n_markers_causal
    assert n_genes > n_genes_affected
    assert n_genes <= phi.size

    idxs = _rnd.choice(phi.size, n_genes, replace=False)
    mu, phi = mu[idxs], phi[idxs]

    # compute phenotype
    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0)
    res = simulate_eQTLs_normal(G, n_markers_causal, n_genes, n_genes_affected, s2e, h2)
    # Z = _utils.sample_nbinom(mu * _nmp.exp(res['Y']), phi)
    Z = sample_nbinom(mu * _nmp.exp(G.dot(res['beta'].T)), phi)

    #
    return {'Z': Z, 'mu': mu, 'phi': phi, 'beta': res['beta'], 'Y': res['Y']}
