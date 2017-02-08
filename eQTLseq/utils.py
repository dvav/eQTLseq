"""Implements various utility functions."""

import numpy as _nmp
import numpy.random as _rnd
import scipy.linalg as _lin
import scipy.stats as _sts

from eQTLseq import parallel as _prl

_EPS = _nmp.finfo('float').eps


def solve_chol_one(L, b):
    """Solve one triangular system."""
    x_ = _lin.solve_triangular(L, b, lower=True)  # L * x_ = b, where x_ = L.T * x
    x = _lin.solve_triangular(L.T, x_, lower=False)  # L.T * x = x_

    #
    return x


def solve_chol_many(A, b):
    """Solve many linear systems, using Cholesky."""
    L = _nmp.linalg.cholesky(A)
    x = [solve_chol_one(L_, b_) for L_, b_ in zip(L, b)]

    #
    return _nmp.asarray(x)


def sample_multivariate_normal_one(L, b, z):
    """Sample from a single multivariate normal distribution."""
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
    """Sample from a Polya-Gamma distribution, as in Proc Int Conf Mach Learn. 2012; 2012: 1343–1350."""
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


def calculate_metrics(beta, beta_true, beta_thr=1e-6, collapse='none'):
    """Calculate errors between estimated and true matrices of coefficients."""
    assert collapse in ('none', 'genes', 'variants')

    # matrix of hits
    hits_2d = _nmp.abs(beta) > beta_thr * _nmp.max(_nmp.abs(beta))
    hits_true_2d = _nmp.abs(beta_true) > beta_thr * _nmp.max(_nmp.abs(beta_true))

    if collapse == 'genes':
        hits = _nmp.any(hits_2d, 0)
        hits_true = _nmp.any(hits_true_2d, 0)
    elif collapse == 'variants':
        hits = _nmp.any(hits_2d, 1)
        hits_true = _nmp.any(hits_true_2d, 1)
    else:
        hits = hits_2d
        hits_true = hits_true_2d

    # true and false positives/negatives
    TP = _nmp.sum(hits & hits_true)
    TN = _nmp.sum(~hits & ~hits_true)
    FP = _nmp.sum(hits & ~hits_true)
    FN = _nmp.sum(~hits & hits_true)

    assert TP + TN + FP + FN == hits.size

    # various metrics
    TPR = TP / (TP + FN + _EPS)  # true positive rate
    TNR = TN / (TN + FP + _EPS)  # true negative rate
    PPV = TP / (TP + FP + _EPS)  # positive predictive value
    NPV = TN / (TN + FN + _EPS)  # negative predictive value
    FPR = 1 - TNR                # false positive rate
    FDR = 1 - PPV                # false discovery rate
    FNR = 1 - TPR                # false negative rate

    MCC = (TP * TN - FP * FN) / _nmp.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN) + _EPS)  # Matthew's correlation coef.
    ACC = (TP + TN) / (TP + FP + FN + TN + _EPS)    # accuracy
    F1 = 2 * TPR * PPV / (TPR + PPV + _EPS)  # F1 score
    G = _nmp.sqrt(TPR * PPV)  # G score
    BM = TPR + TNR - 1        # informedness
    MK = PPV + NPV - 1        # markedness

    beta = beta / _nmp.abs(beta).sum()
    beta_true = beta_true / _nmp.abs(beta_true).sum()

    CCC = compute_ccc(beta, beta_true)
    MSE = _nmp.mean((beta - beta_true) ** 2)
    RMSE = _nmp.sqrt(MSE)
    NRMSE = RMSE / (_nmp.std(beta_true) + _EPS)

    beta_TP = beta[hits_2d & hits_true_2d]
    beta_true_TP = beta_true[hits_2d & hits_true_2d]
    CCC_TP = compute_ccc(beta_TP, beta_true_TP)
    MSE_TP = _nmp.mean((beta_TP - beta_true_TP) ** 2)
    RMSE_TP = _nmp.sqrt(MSE_TP)
    NRMSE_TP = RMSE_TP / (_nmp.std(beta_true_TP) + _EPS)

    #
    return {
        'CCC': CCC,
        'RMSE': RMSE,
        'NRMSE': NRMSE,
        'CCC_TP': CCC_TP,
        'RMSE_TP': RMSE_TP,
        'NRMSE_TP': NRMSE_TP,
        'MCC': MCC,
        'ACC': ACC,
        'F1': F1,
        'G': G,
        'BM': BM,
        'MK': MK,
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
