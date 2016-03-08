"""Implements run()."""

from eQTLseq.ModelNormalGibbs import ModelNormalGibbs as _ModelNormalGibbs
from eQTLseq.ModelNBinomGibbs import ModelNBinomGibbs as _ModelNBinomGibbs


def run(Y, G, mdl='Normal', alg='Gibbs', n_iters=1000, n_burnin=None, s2_lims=(1e-6, 1e6)):
    """Run an estimation algorithm for a specified number of iterations."""
    n_burnin = round(n_iters * 0.5) if n_burnin is None else n_burnin
    assert mdl in ('Normal', 'NBinom')
    assert alg in ('Gibbs',)

    # prepare model
    Model = {
        'Normal': {'Gibbs': _ModelNormalGibbs},
        'NBinom': {'Gibbs': _ModelNBinomGibbs}
    }[mdl][alg]
    mdl = Model(Y, G, n_iters, n_burnin, s2_lims)

    # loop
    for itr in range(1, n_iters + 1):
        mdl.update(itr)
        print('Iteration {0} of {1}'.format(itr, n_iters), end='\r')

    #
    return mdl.stats()
