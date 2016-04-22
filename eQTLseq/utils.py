"""Implements various utility functions."""

import sys as _sys

import numpy as _nmp
import numpy.random as _rnd
import scipy.linalg as _lin
import scipy.optimize as _opt
import scipy.special as _spc
import scipy.stats as _stats

import rpy2.robjects as _R
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()


def solve_chol_one(L, b):
    """TODO."""
    x_ = _lin.solve_triangular(L, b, lower=True)  # L * x_ = b, where x_ = L.T * x
    x = _lin.solve_triangular(L.T, x_, lower=False)  # L.T * x = x_

    #
    return x


def solve_chol_many(A, b):
    """TODO."""
    L = _nmp.linalg.cholesky(A)
    x = [solve_chol_one(L_, b_) for L_, b_ in zip(L, b)]

    #
    return _nmp.asarray(x)


def sample_multivariate_normal_one(L, b, z):
    """TODO."""
    y = _lin.solve_triangular(L.T, z, lower=False)  # L.T * y = z
    mu_ = _lin.solve_triangular(L, b, lower=True)  # L * mu_ = b, where mu_ = L.T * mu
    mu = _lin.solve_triangular(L.T, mu_, lower=False)  # L.T * mu = mu_

    #
    return y + mu


def sample_multivariate_normal_many(b, A, parallel):
    """Sample from the multivariate normal distribution with multiple precision matrices A and mu = A^-1 b."""
    z = _rnd.normal(size=b.shape)
    L = _nmp.linalg.cholesky(A)
    y = [sample_multivariate_normal_one(L_, b_, z_) for L_, b_, z_ in zip(L, b, z)] if parallel is None else \
        parallel.starmap(sample_multivariate_normal_one, zip(L, b, z))

    # return
    return _nmp.asarray(y)


# def sample_multivariate_normal_many(b, A):
#     """Sample from the multivariate normal distribution with multiple precision matrices A and mu = A^-1 b."""
#     S = _nmp.linalg.inv(A)
#     L = _nmp.linalg.cholesky(S)
#     mu = _nmp.sum(S * b[:, None, :], 2)
#     z = _rnd.normal(size=b.shape)
#
#     y = mu + _nmp.sum(L * z[:, None, :], 2)
#
#     # return
#     return y


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
    logcounts = _nmp.log(read_counts + _nmp.finfo('float').tiny)  # add a tiny number to avoid division by zero
    logmeans = _nmp.mean(logcounts, 1)

    # take the ratios
    logcounts -= logmeans[:, None]

    # get median (or other central tendency metric) of ratios excluding rows with at least one zero entry
    idxs = _nmp.all(read_counts > 0, 1)
    logcounts = logcounts[idxs, :]
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
    N, _ = Z.shape
    R = _nmp.asarray([_stats.rankdata(_) for _ in Z.T])
    P = (R - c) / (N - 2 * c + 1)
    Y = _nmp.sqrt(2) * _spc.erfinv(2 * P - 1)    # probit function

    #
    return Y.T


def vst(Z):
    """TODO."""
    vst = _R.r('DESeq2::varianceStabilizingTransformation')
    res = vst(Z)
    Y = _nmp.asarray(res)

    #
    return Y


def voom(Z):
    """TODO."""
    voom = _R.r('limma::voom')
    res = voom(Z)
    Y = _nmp.asarray(res[0])

    #
    return Y


def transform_data(Z, kind='log'):
    """TODO."""
    assert kind in ('blom', 'boxcox', 'log', 'vst', 'voom')

    fcn = {
        'log': lambda x: _nmp.log(x + 1),
        'boxcox': lambda x: _nmp.asarray([_stats.boxcox(_ + 1)[0] for _ in x.T]).T,
        'blom': lambda x: blom(x + _rnd.rand(*x.shape)*1e-6),  # add small random numbers to avoid spurious ties
        'vst': lambda x: vst(x),
        'voom': lambda x: voom(x)
    }[kind]

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


def simulate_eQTLs(G, mu, phi, n_markers_causal=2, n_genes=None, n_genes_affected=10, s2=1):
    """Simulate eQTLs with negative binomially distributed gene expression data."""
    _, n_markers = G.shape
    n_genes = phi.size if n_genes is None else n_genes

    assert n_markers > n_markers_causal
    assert n_genes > n_genes_affected
    assert n_genes <= phi.size

    idxs = _rnd.choice(phi.size, n_genes, replace=False)
    mu, phi = mu[idxs], phi[idxs]

    # sample causal markers and affected genes
    idxs_markers_causal = _rnd.choice(n_markers, n_markers_causal, replace=False)
    idxs_genes_affected = _rnd.choice(n_genes, n_genes_affected, replace=False)

    # compute causal coefficients
    beta = _nmp.zeros((n_genes, n_markers))
    beta[_nmp.ix_(idxs_genes_affected, idxs_markers_causal)] = \
        _rnd.normal(0, _nmp.sqrt(s2 / n_markers_causal), (n_genes_affected, n_markers_causal))

    # compute phenotype
    G = (G - _nmp.mean(G, 0)) / _nmp.std(G, 0)
    Y = _rnd.normal(G.dot(beta.T), 1)
    Z = sample_nbinom(mu * _nmp.exp(G.dot(beta.T)), phi)
    # Z = sample_nbinom(mu * _nmp.exp(Y), phi)

    # remove genes with zero variance
    idxs = _nmp.std(Z, 0) > 0
    Z = Z[:, idxs]
    mu = mu[idxs]
    phi = phi[idxs]
    beta = beta[idxs, :]

    #
    return {'Z': Z.T, 'Y': Y.T, 'mu': mu, 'phi': phi, 'beta': beta}


def calculate_metrics(beta, beta_true, beta_thr=1e-6):
    """Calculate errors between estimated and true matrices of coefficients."""
    beta[_nmp.abs(beta) < beta_thr] = 0
    beta_true[_nmp.abs(beta_true) < beta_thr] = 0

    beta = beta / _nmp.abs(beta).sum()
    beta_true = beta_true / _nmp.abs(beta_true).sum()

    # matrix of hits
    hits = _nmp.abs(_nmp.sign(beta))
    hits_true = _nmp.abs(_nmp.sign(beta_true))

    # true and false positives/negatives
    TP = _nmp.sum((hits == 1) & (hits_true == 1))
    TN = _nmp.sum((hits == 0) & (hits_true == 0))
    FP = _nmp.sum((hits == 1) & (hits_true == 0))
    FN = _nmp.sum((hits == 0) & (hits_true == 1))

    # assert TP + TN + FP + FN == beta.size

    # various metrics
    TPR = TP / (TP + FN)  # true positive rate
    TNR = TN / (TN + FP)  # true negative rate
    PPV = TP / (TP + FP)  # positive predictive value
    NPV = TN / (TN + FN)  # negative predictive value
    FPR = FP / (FP + TN)  # false positive rate
    FDR = FP / (FP + TP)  # false discovery rate
    FNR = FN / (FN + TP)  # false negative rate

    MCC = (TP * TN - FP * FN) / _nmp.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))  # Matthew's correlation coefficient
    ACC = (TP + TN) / (TP + FP + FN + TN)  # accuracy
    F1 = 2 * TPR * PPV / (TPR + PPV)  # F1 score
    G = _nmp.sqrt(TPR * PPV)  # G score

    # sum of squared residuals
    idxs = (hits == 1) & (hits_true == 1)
    RSS = ((beta[idxs] - beta_true[idxs])**2).sum() / TP

    #
    return {
        'RSS': RSS,
        'MCC': MCC,
        'ACC': ACC,
        'F1': F1,
        'G': G,
        'TPR': TPR,
        'TNR': TNR,
        'PPV': PPV,
        'NPV': NPV,
        'FPR': FPR,
        'FDR': FDR,
        'FNR': FNR,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }
