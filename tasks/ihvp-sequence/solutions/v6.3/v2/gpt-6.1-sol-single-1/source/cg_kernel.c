#include <stddef.h>
#include <math.h>

double cg_step(size_t n, double *restrict x, double *restrict r,
               double *restrict p, const double *restrict ap,
               double rr, double mu, double *bound)
{
    double curvature = 0.0;
    #pragma omp simd reduction(+:curvature)
    for (size_t i = 0; i < n; ++i)
        curvature += p[i] * ap[i];
    if (!(curvature > 0.0) || !isfinite(curvature))
        return -1.0;
    double alpha = rr / curvature;
    if (!isfinite(alpha))
        return -1.0;
    double next_rr = 0.0;
    #pragma omp simd reduction(+:next_rr)
    for (size_t i = 0; i < n; ++i) {
        x[i] += alpha * p[i];
        double next_r = r[i] - alpha * ap[i];
        r[i] = next_r;
        next_rr += next_r * next_r;
    }
    if (!isfinite(next_rr))
        return -1.0;
    double beta = next_rr / rr;
    #pragma omp simd
    for (size_t i = 0; i < n; ++i)
        p[i] = beta * p[i] + r[i];
    double delta = *bound - alpha;
    *bound = delta > 0.0 ? delta / (mu * delta + beta) : 1.0 / mu;
    if (!isfinite(*bound))
        *bound = 1.0 / mu;
    return next_rr;
}
