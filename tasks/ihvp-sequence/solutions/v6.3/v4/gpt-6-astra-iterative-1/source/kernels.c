#include <math.h>
#include <stddef.h>

int cg_step(size_t n, int m, double *x, double *r, double *p,
            const double *hp, const double *w, const double *aw,
            double rr, double *out) {
    double php = 0.0;
    #pragma omp simd reduction(+:php)
    for (size_t i = 0; i < n; ++i)
        php += p[i] * hp[i];
    if (!(php > 0.0) || !isfinite(php) || !(rr > 0.0))
        return 0;

    const double alpha = rr / php;
    double next_rr = 0.0;
    double coefficients[64] = {0.0};
    if (m == 2) {
        double c0 = 0.0, c1 = 0.0;
        #pragma omp simd reduction(+:next_rr,c0,c1)
        for (size_t i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            const double v = r[i] - alpha * hp[i];
            r[i] = v;
            next_rr += v * v;
            c0 += aw[i] * v;
            c1 += aw[n + i] * v;
        }
        coefficients[0] = c0;
        coefficients[1] = c1;
    } else if (m == 1) {
        double c0 = 0.0;
        #pragma omp simd reduction(+:next_rr,c0)
        for (size_t i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            const double v = r[i] - alpha * hp[i];
            r[i] = v;
            next_rr += v * v;
            c0 += aw[i] * v;
        }
        coefficients[0] = c0;
    } else {
        #pragma omp simd reduction(+:next_rr)
        for (size_t i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            const double v = r[i] - alpha * hp[i];
            r[i] = v;
            next_rr += v * v;
        }
        for (int j = 0; j < m; ++j) {
            double c = 0.0;
            const double *column = aw + (size_t)j * n;
            #pragma omp simd reduction(+:c)
            for (size_t i = 0; i < n; ++i)
                c += column[i] * r[i];
            coefficients[j] = c;
        }
    }
    const double beta = next_rr / rr;
    if (m == 2) {
        const double c0 = coefficients[0], c1 = coefficients[1];
        #pragma omp simd
        for (size_t i = 0; i < n; ++i)
            p[i] = beta * p[i] + r[i] - c0 * w[i] - c1 * w[n + i];
    } else if (m == 1) {
        const double c0 = coefficients[0];
        #pragma omp simd
        for (size_t i = 0; i < n; ++i)
            p[i] = beta * p[i] + r[i] - c0 * w[i];
    } else {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i)
            p[i] = beta * p[i] + r[i];
        for (int j = 0; j < m; ++j) {
            const double c = coefficients[j];
            const double *column = w + (size_t)j * n;
            #pragma omp simd
            for (size_t i = 0; i < n; ++i)
                p[i] -= c * column[i];
        }
    }
    out[0] = next_rr;
    out[1] = alpha;
    out[2] = php;
    return 1;
}
