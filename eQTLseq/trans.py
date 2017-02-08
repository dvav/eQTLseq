"""Various data transformation methods."""

import numpy as _nmp
import scipy.stats as _stats
import scipy.special as _spc

from rpy2 import robjects as _ro
from rpy2.robjects import numpy2ri as _numpy2ri
_numpy2ri.activate()

_EPS = _nmp.finfo('float').eps


def calculate_norm_factors(read_counts, locfcn=_nmp.median):
    """Normalise RNA-seq counts data using the Relative Log Expression (RLE) method, as in DESeq."""
    # compute geometric mean of each row in log-scale
    idxs = _nmp.all(read_counts > 0, 1)
    logcounts = _nmp.log(read_counts[idxs, :])
    logmeans = _nmp.mean(logcounts, 1)

    # take the ratios
    logcounts = logcounts - logmeans[:, None]

    # get median (or other central tendency metric) of ratios excluding rows with at least one zero entry
    norm_factors = locfcn(_nmp.exp(logcounts), 0)

    #
    return norm_factors


def blom(Z, c, method):
    """Blom-transform matrix Z."""
    N, _ = Z.shape
    R = _nmp.asarray([_stats.rankdata(_, method) for _ in Z.T]).T
    P = (R - c) / (N - 2 * c + 1)
    Y = _nmp.sqrt(2) * _spc.erfinv(2 * P - 1)    # probit function

    #
    return Y


def vst(Z):
    """TODO."""
    vst = _ro.r('DESeq2::varianceStabilizingTransformation')
    res = vst(Z)
    Y = _nmp.asarray(res)

    #
    return Y


def rlog(Z):
    """TODO."""
    rlog = _ro.r('DESeq2::rlogTransformation')
    res = rlog(Z, fitType='local')
    Y = _nmp.asarray(res)

    #
    return Y


def voom(Z):
    """TODO."""
    voom = _ro.r('limma::voom')
    res = voom(Z)
    Y = _nmp.asarray(res[0])

    #
    return Y


def arcsin(Z):
    """Arcsine-transform matrix Z."""
    n_genes, _ = Z.shape
    p = (Z + 1) / (Z.sum(0) + n_genes)
    Y = _nmp.arcsin(_nmp.sqrt(p))

    #
    return Y


def logit(Z):
    """Logit-transform matrix Z."""
    n_genes, _ = Z.shape
    p = (Z + 1) / (Z.sum(0) + n_genes)
    Y = _nmp.log(p) - _nmp.log1p(-p)

    #
    return Y


def transform_data(Z, kind='log', c=3/8, method='average'):
    """Various data transformations."""
    assert kind in ('blom', 'boxcox', 'log', 'logcpm', 'arcsin', 'logit', 'vst', 'rlog', 'voom')

    trans = {
        'log': lambda X: _nmp.log(X + 1),
        'boxcox': lambda X: _nmp.asarray([_stats.boxcox(_ + 1)[0] for _ in X.T]).T,
        'blom': lambda X: blom(X, c, method),
        'vst': lambda X: vst(X),
        'rlog': lambda X: rlog(X),
        'voom': lambda X: voom(X),
        'logcpm': lambda X: _nmp.log(X + 0.5) - _nmp.log(X.sum(0) + 1) + 6 * _nmp.log(10),
        'arcsin': lambda X: arcsin(X),
        'logit': lambda X: logit(X)
    }[kind]

    Y = trans(Z)

    #
    return Y
