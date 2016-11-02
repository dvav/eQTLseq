"""TODO."""

import numpy as _nmp
import scipy.stats as _stats
import scipy.special as _spc

import rpy2.robjects as _R
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()


def calculate_norm_factors(read_counts, locfcn=_nmp.median):
    """Normalise RNA-seq counts data using the Relative Log Expression (RLE) method, as in DESeq."""
    # compute geometric mean of each row in log-scale
    idxs = _nmp.all(read_counts > 0, 1)
    logcounts = _nmp.log(read_counts[idxs, :])
    logmeans = _nmp.mean(logcounts, 1)

    # take the ratios
    logcounts = logcounts - logmeans[:, None]

    # get median (or other central tendency metric) of ratios excluding rows with at least one zero entry
    norm_factors = _nmp.exp(locfcn(logcounts, 0))

    #
    return norm_factors


def blom(Z, c=3/8):
    """TODO."""
    N, _ = Z.shape
    R = _nmp.asarray([_stats.rankdata(_) for _ in Z.T]).T
    P = (R - c) / (N - 2 * c + 1)
    Y = _nmp.sqrt(2) * _spc.erfinv(2 * P - 1)    # probit function

    #
    return Y


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

    trans = {
        'log': lambda X: _nmp.log(X + 1),
        'boxcox': lambda X: _nmp.asarray([_stats.boxcox(_ + 1)[0] for _ in X.T]).T,
        'blom': lambda X: blom(X),  # add small random numbers to avoid spurious ties  + _rnd.rand(*x.shape)*1e-6
        'vst': lambda X: vst(X),
        'voom': lambda X: voom(X)
    }[kind]

    Y = trans(Z)

    #
    return Y
