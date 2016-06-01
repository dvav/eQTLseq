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


def sample_PG(a, b, K=10):
    """TODO."""
    assert a.shape == b.shape
    pi = _nmp.pi

    k = _nmp.r_[1:K+1][:, None, None]
    denom = (k - 0.5)**2 + 0.25 * (b / pi)**2

    g = _rnd.gamma(a, 1, size=(K,) + a.shape)
    x = 0.5 / pi**2 * (g / denom).sum(0)

    c1 = 0.5 * a / b * _nmp.tanh(0.5 * b)
    c2 = 0.5 / pi**2 * (a / denom).sum(0)
    x = c1 / c2 * x

    # return
    return x


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
        'blom': lambda x: blom(x),  # add small random numbers to avoid spurious ties  + _rnd.rand(*x.shape)*1e-6
        'vst': lambda x: vst(x),
        'voom': lambda x: voom(x)
    }[kind]

    Y = fcn(Z)

    #
    return Y


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


def simulate_eQTLs(G, mu, phi, pattern=(1, 10, 0, 0), size=4, pois=0.5, out=(0.05, 5, 10)):
    """Simulate eQTLs with negative binomially distributed gene expression data."""
    _, n_markers = G.shape
    n_genes = phi.size
    n_markers_hot, n_genes_hot, n_genes_poly, n_markers_poly = pattern

    assert (n_markers > n_markers_hot) & (n_markers > n_markers_poly)
    assert (n_genes > n_genes_hot) & (n_genes > n_genes_poly)
    assert _nmp.all(_nmp.std(G, 0) > 0) and _nmp.all(_nmp.std(G, 1) > 0)
    assert size > 1

    # poisson distributed genes
    poisson = _nmp.zeros(n_genes, dtype='bool')
    poisson[_rnd.choice(n_genes, int(n_genes * pois), replace=False)] = True
    phi[poisson] = 1e-20

    # coefficients
    beta = _nmp.zeros((n_genes, n_markers))

    # hotspots
    if n_markers_hot > 0:
        hot_idxs_markers = _rnd.choice(n_markers, n_markers_hot, replace=False)
        hot_idxs_genes = _nmp.hstack([_rnd.choice(n_genes, (n_genes_hot, 1), replace=False) for _ in hot_idxs_markers])
        beta[hot_idxs_genes, hot_idxs_markers] = 1 + _rnd.exponential(size=(n_genes_hot, n_markers_hot))

    # polymarker effects
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
    Z = sample_nbinom(mu * _nmp.exp(GBT), phi)

    # outliers
    n_samples, _ = G.shape
    outliers = _nmp.zeros((n_samples, n_genes), dtype='bool')
    for i in range(n_samples):
        outliers[i, _rnd.choice(n_genes, size=int(out[0] * n_genes), replace=False)] = True
    Z[outliers] = Z[outliers] * _rnd.uniform(out[1], out[2], size=_nmp.count_nonzero(outliers))

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


def calculate_metrics(beta, beta_true, beta_thr=1e-6):
    """Calculate errors between estimated and true matrices of coefficients."""
    beta[_nmp.abs(beta) < beta_thr] = 0
    beta_true[_nmp.abs(beta_true) < beta_thr] = 0

    # standardize
    beta = beta if _nmp.all(beta == 0) else beta / _nmp.abs(beta).sum()
    beta_true = beta_true if _nmp.all(beta_true == 0) else beta_true / _nmp.abs(beta_true).sum()

    # matrix of hits
    hits = _nmp.abs(_nmp.sign(beta))
    hits_true = _nmp.abs(_nmp.sign(beta_true))

    # true and false positives/negatives
    TP = _nmp.sum((hits == 1) & (hits_true == 1))
    TN = _nmp.sum((hits == 0) & (hits_true == 0))
    FP = _nmp.sum((hits == 1) & (hits_true == 0))
    FN = _nmp.sum((hits == 0) & (hits_true == 1))

    assert TP + TN + FP + FN == beta.size

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

    # average standardised residual among true positives
    idxs = (hits == 1) & (hits_true == 1)
    RSS = _nmp.mean(((beta[idxs] - beta_true[idxs]) / beta_true[idxs])**2) if _nmp.any(idxs) else _nmp.nan

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
