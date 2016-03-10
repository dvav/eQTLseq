"""Implements run()."""

from eQTLseq.ModelNBinomGibbs import ModelNBinomGibbs as _ModelNBinomGibbs
from eQTLseq.ModelNormalGibbs import ModelNormalGibbs as _ModelNormalGibbs
from eQTLseq.ModelTraitNormalGibbs import ModelTraitNormalGibbs as _ModelTraitNormalGibbs

import eQTLseq.utils as _utils


def run(Y, G, kind='eQTLs', mdl='Normal', alg='Gibbs', norm=False, n_iters=1000, n_burnin=None, s2_lims=(1e-6, 1e6)):
    """Run an estimation algorithm for a specified number of iterations."""
    n_burnin = round(n_iters * 0.5) if n_burnin is None else n_burnin
    assert kind in ('eQTLs', 'Trait')
    assert mdl in ('Normal', 'NBinom')
    assert alg in ('Gibbs',)

    # normalize data if necessary
    if mdl == 'NBinom' and not norm:
        Y = _utils.normalise_RNAseq_data(Y.T)[0].T

    # prepare model
    Model = {
        'eQTLs': {
            'Normal': {'Gibbs': _ModelNormalGibbs},
            'NBinom': {'Gibbs': _ModelNBinomGibbs}
        },
        'Trait': {
            'Normal': {'Gibbs': _ModelTraitNormalGibbs}
        }
    }[kind][mdl][alg]
    mdl = Model(Y, G, n_iters, n_burnin, s2_lims)

    # loop
    for itr in range(1, n_iters + 1):
        mdl.update(itr)
        print('Iteration {0} of {1}'.format(itr, n_iters), end='\r')

    #
    return mdl.stats()
