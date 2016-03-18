"""Implements various utility functions."""

import joblib as _jbl

import numpy as _nmp
import numpy.random as _rnd
import scipy.linalg as _lin
import scipy.optimize as _opt
import scipy.special as _spc


def norm(x):
    """Calculate euclidean norm of x."""
    return _nmp.sqrt(_nmp.sum(x**2))


def chol_solve(A, b):
    """Solve A * x = b, where A is symmetric positive definite, using the Cholesky factorisation of A."""
    a = _lin.cho_factor(A)
    x = _lin.cho_solve(a, b)

    # return
    return x


def sample_multivariate_normal(b, A):
    """Sample from the multivariate normal distribution with precision matrix A and mu = A^-1 b."""
    z = _rnd.normal(size=b.shape)

    U = _lin.cholesky(A)
    y = _lin.solve_triangular(U, z)  # U * y = z

    mu_ = _lin.solve_triangular(U, b, trans='T')  # U.T * mu_ = b, where mu_ = U * mu
    mu = _lin.solve_triangular(U, mu_)  # U * mu = mu_

    # return
    return y + mu


# _PARALLEL = _jbl.Parallel(n_jobs=8, backend='threading')
#
#
# def chol_solve_many(A, b):
#     """Solve A * x = b, where A are multiple symmetric positive definite matrices, using Cholesky."""
#     L = _nmp.linalg.cholesky(A)
#     X = _PARALLEL(_jbl.delayed(_lin.cho_solve)((L_, True), b_) for L_, b_ in zip(L, b))
#
#     #
#     return _nmp.asarray(X)

def chol_solve_many(A, b):
    """Solve A * x = b, where A are multiple symmetric positive definite matrices, using Cholesky."""
    L = _nmp.linalg.cholesky(A)
    X = [_lin.cho_solve((L_, True), b_) for L_, b_ in zip(L, b)]

    # return
    return _nmp.asarray(X)


def sample_multivariate_normal_many(b, A):
    """Sample from the multivariate normal distribution with multiple precision matrices A and mu = A^-1 b."""
    z = _rnd.normal(size=b.shape)

    L = _nmp.linalg.cholesky(A)
    U = _nmp.transpose(L, axes=(0, 2, 1))

    y = [_lin.solve_triangular(U_, z_) for U_, z_ in zip(U, z)]   # U * y = z

    m = [_lin.solve_triangular(U_, b_, trans='T') for U_, b_ in zip(U, b)]   # U.T * m = b, where m = U * mu
    mu = [_lin.solve_triangular(U_, m_) for U_, m_ in zip(U, m)]  # U * mu = mu_

    # return
    return _nmp.asarray(y) + _nmp.asarray(mu)


def sample_multivariate_normal2(b, A):
    """Sample from the multivariate normal distribution with multiple precision matrices A and mu = A^-1 b."""
    z = _rnd.normal(size=b.shape)

    L = _nmp.linalg.cholesky(A)
    U = _nmp.transpose(L, axes=(0, 2, 1))

    y = _nmp.linalg.solve(U, z)  # U * y = z

    mu_ = _nmp.linalg.solve(L, b)  # U.T * mu_ = b, where mu_ = U * mu
    mu = _nmp.linalg.solve(U, mu_)  # U * mu = mu_

    # return
    return y + mu


def sample_nbinom(mu, phi, size=None):
    """Sample from the Negative Binomial distribution with mean `mu` and dispersion `phi`."""
    # sample lambdas from gamma and, then, counts from Poisson
    shape = 1 / phi
    scale = mu * phi
    lam = _rnd.gamma(shape, scale, size=size)
    counts = _rnd.poisson(lam, size=size)

    #
    return counts


def normalise_RNAseq_data(read_counts, locfcn=_nmp.median):
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
    return read_counts / norm_factors, norm_factors


def fit_nbinom_model(read_counts, normalised=False):
    """Fit a Negative Binomial model to a table of RNA-seq count data using maximum likelihood estimation."""
    # prepare data
    n_genes, n_samples = read_counts.shape

    def fcn(alpha, ydata, ymean):
        return _spc.psi(ydata + alpha).sum() - n_samples * _spc.psi(alpha) + n_samples * _nmp.log(alpha) \
            - n_samples * _nmp.log(ymean + alpha)

    # iterate over genes and fit across samples
    ydata = read_counts if normalised else normalise_RNAseq_data(read_counts)[0]
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

        print('Fitting gene {0} of {1}'.format(i, n_genes), end='\r')

    #
    return {
        'mu': ymean,
        'phi': 1 / alpha,
        'converged': converged
    }
