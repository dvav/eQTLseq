"""Simulate genotypes and RNA-seq data."""

import sys as _sys

import numpy as _nmp
import numpy.random as _rnd
import scipy.optimize as _opt
import scipy.special as _spc

import eQTLseq.utils as _utils


def fit_nbinom_model(read_counts, normalised=False):
    """Fit a Negative Binomial model to a table of RNA-seq count data using maximum likelihood estimation."""
    # prepare data
    n_genes, n_samples = read_counts.shape

    def fcn(alpha, ydata, ymean):
        return _spc.psi(ydata + alpha).sum() - n_samples * _spc.psi(alpha) + n_samples * _nmp.log(alpha) \
            - n_samples * _nmp.log(ymean + alpha)

    # iterate over genes and fit across samples
    ydata = read_counts if normalised else read_counts / _utils.calculate_norm_factors(read_counts)
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


def simulate_genotypes(MAF, n_samples=1000, n_markers=100):
    """Generate a matrix of genotypes, using a binomial model."""
    assert MAF.size >= n_markers

    # compute MAFs for each genetic marker and compute genotypes
    MAF = _rnd.choice(MAF, n_markers, replace=False)
    G = _rnd.binomial(2, MAF, (n_samples, n_markers))   # assume ploidy=2

    # drop mono-morphic markers
    G = G[:, _nmp.std(G, 0) > 0]

    #
    return {'G': G, 'MAF': MAF}


def simulate_eQTLs(G, mu, phi, pattern=(1, 10, 0, 0), size=4, pois=0.5, out=('S', 0.05, 5, 10)):
    """Simulate eQTLs with negative binomially distributed gene expression data."""
    _, n_markers = G.shape
    n_genes = phi.size
    n_markers_hot, n_genes_hot, n_genes_poly, n_markers_poly = pattern

    assert (n_markers > n_markers_hot) & (n_markers > n_markers_poly)
    assert (n_genes > n_genes_hot) & (n_genes > n_genes_poly)
    assert _nmp.all(_nmp.std(G, 0) > 0) and _nmp.all(_nmp.std(G, 1) > 0)
    assert size > 1
    assert 0 <= pois < 1
    assert out[0] in ['R', 'S'] and 0 <= out[1] < 1 and out[2] > 0 and out[2] < out[3]

    # poisson distributed genes
    poisson = _nmp.zeros(n_genes, dtype='bool')
    poisson[_rnd.choice(n_genes, int(n_genes * pois), replace=False)] = True
    phi[poisson] = 1e-20

    # coefficients
    beta = _nmp.zeros((n_genes, n_markers))

    # random gene/variant associations
    if n_markers_hot > 0 and n_genes_hot == 0 and n_markers_poly == 0 and n_genes_poly == 0:
        idxs = _rnd.choice(n_genes * n_markers, n_markers_hot, replace=False)
        beta[_nmp.unravel_index(idxs, (n_genes, n_markers))] = 1 + _rnd.exponential(size=n_markers_hot)

    # hotspots
    if n_markers_hot > 0:
        hot_idxs_markers = _rnd.choice(n_markers, n_markers_hot, replace=False)
        hot_idxs_genes = _nmp.hstack([_rnd.choice(n_genes, (n_genes_hot, 1), replace=False) for _ in hot_idxs_markers])
        beta[hot_idxs_genes, hot_idxs_markers] = 1 + _rnd.exponential(size=(n_genes_hot, n_markers_hot))

    # polygenic effects
    if n_genes_poly > 0:
        poly_idxs_genes = _rnd.choice(n_genes, (n_genes_poly, 1), replace=False)
        poly_idxs_markers = _nmp.vstack([_rnd.choice(n_markers, n_markers_poly, replace=False) for _ in poly_idxs_genes])
        beta[poly_idxs_genes, poly_idxs_markers] = 1 + _rnd.exponential(size=(n_genes_poly, n_markers_poly))

    beta = beta * _rnd.choice([-1, 1], size=beta.shape)

    # scale coefficients
    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0)

    GBT = G.dot(beta.T)
    mx = _nmp.max(_nmp.abs(GBT), 0)
    idxs = mx != 0
    beta[idxs, :] = beta[idxs, :] / mx[idxs, None] * _nmp.log(size)

    # compute phenotype
    GBT = G.dot(beta.T)
    Y = _rnd.normal(GBT, 1)
    Z = _utils.sample_nbinom(mu * _nmp.exp(GBT), phi)

    # outliers
    n_samples, _ = G.shape
    if out[0] == 'R':  # random outlier simulation
        outliers = _rnd.choice((True, False), size=(n_samples, n_genes), p=(out[1], 1 - out[1]))
    else:  # single outlier simulation
        n = int(n_genes * out[1])
        outliers = _nmp.zeros((n_samples, n_genes), dtype='bool')
        gene_idxs = _rnd.choice(n_genes, size=n, replace=False)
        sample_idxs = _rnd.choice(n_samples, size=n, replace=True)
        outliers[sample_idxs, gene_idxs] = True
    Z[outliers] = Z[outliers] * _rnd.uniform(out[2], out[3], size=_nmp.count_nonzero(outliers))

    # remove genes with zero variance
    idxs = _nmp.std(Z, 0) > 0
    Z = Z[:, idxs]
    mu = mu[idxs]
    phi = phi[idxs]
    beta = beta[idxs, :]
    poisson = poisson[idxs]
    outliers = outliers[:, idxs]

    #
    return {'Z': Z.T, 'Y': Y.T, 'mu': mu, 'phi': phi, 'beta': beta, 'poisson': poisson, 'outliers': outliers.T}
