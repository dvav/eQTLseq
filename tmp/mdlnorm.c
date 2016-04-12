#include <stdio.h>
#include <math.h>

static void _sample_beta(const double * y, const double * G, double * beta, const double tau,
    const double * zeta, const double * eta, const double * rnds, const ssize_t * idxs, const ssize_t n_samples,
    const ssize_t n_markers) {

    // iterate over markers
    for(ssize_t j = 0; j < n_markers; j++) {
        const ssize_t idx = idxs[j];

        // iterate over samples
        double sum1 = 0.0;
        double sum2 = 0.0;
        for(ssize_t i = 0; i < n_samples; i++) {
            const double G_ji = G[idx * n_samples + i];
            // compute dot product
            double dot = 0.0;
            for(ssize_t jj = 0; jj < n_markers; jj++) {
                ssize_t not_idx = idxs[jj];
                if(not_idx != idx) {
                    dot += G[not_idx * n_samples + i] * beta[not_idx];
                }
            }
            // compute auxilliary sums
            sum1 += G_ji * G_ji;
            sum2 += (y[i] - dot) * G_ji;
        }

        const double rho = sum1 + 0.5 * zeta[idx] * eta[idx];
        const double mu  = sum2 / rho;

        // compute beta[k, j]
        beta[idx] = mu + rnds[idx] / sqrt(tau * rho);
    }
}

void sample_beta(const double * Y, const double * G, double * beta, const double * tau, const double * zeta,
    const double * eta, const double * rnds, const ssize_t * idxs, const ssize_t n_samples, const ssize_t n_genes,
    const ssize_t n_markers) {

    // iterate over genes
#pragma omp parallel for
{
    for(ssize_t k = 0; k < n_genes; k++) {
        const double * y_k    = Y    + k * n_samples;
              double * beta_k = beta + k * n_markers;
        const double * zeta_k = zeta + k * n_markers;
        const double * rnds_k = rnds + k * n_markers;
        const double   tau_k  = tau[k];

        _sample_beta(y_k, G, beta_k, tau_k, zeta_k, eta, rnds_k, idxs, n_samples, n_markers);
    }
}
}
