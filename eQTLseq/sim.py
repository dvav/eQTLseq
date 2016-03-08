"""Includes various functions for simulating genotypic and phenotypic data."""

import numpy as _nmp
import numpy.random as _rnd

import eQTLseq.utils as _utils


def simulate_genotypes(n_samples=1000, n_markers=100, MAF_range=(0.05, 0.5)):
    """Generate a matrix of genotypes, using a binomial model."""
    # compute MAFs for each genetic marker and compute genotypes
    MAF = _rnd.uniform(MAF_range[0], MAF_range[1], n_markers)
    G = _rnd.binomial(2, MAF, (n_samples, n_markers))   # assume ploidy=2

    # drop mono-morphic markers
    G = G[:, _nmp.std(G, 0) > 0]

    #
    return {'G': G, 'MAF': MAF}


def simulate_trait_normal(G, n_markers_causal=10, s2e=1, h2=0.5):
    """Simulate a quantitative, normally distributed trait."""
    n_samples, n_markers = G.shape
    assert n_markers >= n_markers_causal

    # sample causal markers
    idxs_causal = _rnd.choice(n_markers, n_markers_causal, replace=False)

    # compute causal coefficients
    s2g = h2 * s2e / (1 - h2)
    coefs = _nmp.zeros(n_markers)
    coefs[idxs_causal] = _rnd.normal(0, _nmp.sqrt(s2g / n_markers_causal), n_markers_causal)

    # compute phenotype
    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0)
    Y = _rnd.normal(G.dot(coefs), _nmp.sqrt(s2e))

    #
    return {'Y': Y, 'coefs': coefs}


def simulate_eQTL_normal(G, n_markers_causal=2, n_genes=100, n_genes_affected=10, s2e=1, h2=0.5):
    """Simulate eQTLs with normally distributed gene expression data."""
    n_samples, n_markers = G.shape
    assert n_markers >= n_markers_causal
    assert n_genes >= n_genes_affected

    # sample causal markers and affected genes
    idxs_markers_causal = _rnd.choice(n_markers, n_markers_causal, replace=False)
    idxs_genes_affected = _nmp.hstack([
        _rnd.choice(n_genes, (n_genes_affected, 1), replace=False) for _ in range(n_markers_causal)
    ])

    # compute causal coefficients
    s2g = h2 * s2e / (1 - h2)
    coefs = _nmp.zeros((n_genes, n_markers))
    coefs[idxs_genes_affected, idxs_markers_causal] = \
        _rnd.normal(0, _nmp.sqrt(s2g / n_markers_causal), (n_genes_affected, n_markers_causal))

    # compute phenotype
    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0)
    Y = _rnd.normal(G.dot(coefs.T), _nmp.sqrt(s2e))

    #
    return {'Y': Y, 'coefs': coefs}


def simulate_eQTL_nbinom(G, mu, phi, n_markers_causal=2, n_genes=None, n_genes_affected=10, h2=0.5):
    """Simulate eQTLs with normally distributed gene expression data."""
    n_samples, n_markers = G.shape
    n_genes = phi.size if n_genes is None else n_genes

    assert n_markers >= n_markers_causal
    assert n_genes <= phi.size
    assert n_genes >= n_genes_affected

    idxs = _rnd.choice(phi.size, n_genes, replace=False)
    mu, phi = mu[idxs], phi[idxs]

    # sample causal markers and affected genes
    idxs_markers_causal = _rnd.choice(n_markers, n_markers_causal, replace=False)
    idxs_genes_affected = _nmp.hstack([
        _rnd.choice(n_genes, (n_genes_affected, 1), replace=False) for _ in range(n_markers_causal)
    ])

    # compute causal coefficients
    s2e = 1  # mu[idxs_genes_affected] + mu[idxs_genes_affected]**2 / phi[idxs_genes_affected]
    s2g = h2 * s2e / (1 - h2)
    coefs = _nmp.zeros((n_genes, n_markers))
    coefs[idxs_genes_affected, idxs_markers_causal] = \
        _rnd.normal(0, _nmp.sqrt(s2g / n_markers_causal), (n_genes_affected, n_markers_causal))

    # compute phenotype
    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0)
    Y = _utils.sample_nbinom(mu * _nmp.exp(G.dot(coefs.T)), phi)

    #
    return {'Y': Y, 'coefs': coefs, 'mu': mu, 'phi': phi}
