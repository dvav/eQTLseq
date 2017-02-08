"""Implements various algorithms."""

import numpy as _nmp
import tqdm as _tqdm


def gibbs(mdl, **args):
    """A Gibbs sampler."""
    n_iters, n_burnin, hide_progress = args['n_iters'], args['n_burnin'], args['hide_progress']

    state = _nmp.r_[0, [_nmp.nan] * n_iters]
    sums = {_: _nmp.zeros_like(mdl.state[_]) for _ in mdl.state}
    sums2 = {_: _nmp.zeros_like(mdl.state[_]) for _ in mdl.state}

    ##
    for itr in _tqdm.tqdm(range(1, n_iters + 1), disable=hide_progress):
        # update state by calling samplers sequentially
        for f in mdl.samplers:
            f(**args)

        state[itr] = mdl.get_state()

        # compute sufficient statistics, if necessary
        if(itr > n_burnin):
            sums = {_: sums[_] + mdl.state[_] for _ in mdl.state}
            sums2 = {_: sums2[_] + mdl.state[_]**2 for _ in mdl.state}

    ##
    return {
        'state': state,
        **_get_estimates(sums, sums2, n_iters - n_burnin)
    }


def _get_estimates(sums, sums2, N):
    """TODO."""
    means = {_: sums[_] / N for _ in sums}
    varrs = {_ + '_var': sums2[_] / N - means[_]**2 for _ in sums}

    return {
        'N': N,
        **means,
        **varrs
    }
