#include <stddef.h>

double cg_step(size_t n, double *restrict x, double *restrict r,
               const double *restrict p, const double *restrict ap, double num) {
    double den = 0.;
    for (size_t i = 0; i < n; ++i) den += p[i] * ap[i];
    if (!(den > 0.)) return -1.;
    double alpha = num / den;
    double rr = 0.;
    for (size_t i = 0; i < n; ++i) {
        x[i] += alpha * p[i];
        r[i] -= alpha * ap[i];
        rr += r[i] * r[i];
    }
    return rr;
}

void cg_direction(size_t n, double *restrict p, const double *restrict r,
                  const double *restrict correction, double beta) {
    if (correction) {
        for (size_t i = 0; i < n; ++i)
            p[i] = beta * p[i] + r[i] - correction[i];
    } else {
        for (size_t i = 0; i < n; ++i)
            p[i] = beta * p[i] + r[i];
    }
}
