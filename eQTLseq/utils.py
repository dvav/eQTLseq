"""Implements various utility functions."""

import numpy as _nmp
import numpy.random as _rnd
import scipy.linalg as _lin

from eQTLseq import parallel as _prl

_EPS = _nmp.finfo('float').eps


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


def sample_multivariate_normal_many(b, A):
    """Sample from the multivariate normal distribution with multiple precision matrices A and mu = A^-1 b."""
    z = _rnd.normal(size=b.shape)
    L = _nmp.linalg.cholesky(A)
    y = [sample_multivariate_normal_one(L_, b_, z_) for L_, b_, z_ in zip(L, b, z)] if _prl.POOL is None else \
        _prl.POOL.starmap(sample_multivariate_normal_one, zip(L, b, z))

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


def compute_ccc(x, y):
    """Calculates the Concordance Correlation Coefficient between vectors x and y."""
    x_hat = x.sum() / x.size
    y_hat = y.sum() / y.size

    s2x = _nmp.sum((x - x_hat)**2) / x.size
    s2y = _nmp.sum((y - y_hat)**2) / y.size
    sxy = _nmp.sum((x - x_hat)*(y - y_hat)) / x.size

    ##
    return 2 * sxy / (s2x + s2y + (x_hat - y_hat)**2)


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
    TPR = TP / (TP + FN + _EPS)  # true positive rate
    TNR = TN / (TN + FP + _EPS)  # true negative rate
    PPV = TP / (TP + FP + _EPS)  # positive predictive value
    NPV = TN / (TN + FN + _EPS)  # negative predictive value
    FPR = FP / (FP + TN + _EPS)  # false positive rate
    FDR = FP / (FP + TP + _EPS)  # false discovery rate
    FNR = FN / (FN + TP + _EPS)  # false negative rate

    MCC = (TP * TN - FP * FN) / _nmp.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN) + _EPS)  # Matthew's correlation coef.
    ACC = (TP + TN) / (TP + FP + FN + TN + _EPS)  # accuracy
    F1 = 2 * TPR * PPV / (TPR + PPV + _EPS)  # F1 score
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
