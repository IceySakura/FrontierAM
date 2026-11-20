#include <math.h>
#include <stddef.h>
#include <immintrin.h>

static inline double reduce(__m256d x) {
    double a[4];
    _mm256_storeu_pd(a, x);
    return (a[0] + a[1]) + (a[2] + a[3]);
}

static double dot_across(const double *a, const double *b, size_t n) {
    __m256d s0 = _mm256_setzero_pd(), s1 = s0, s2 = s0, s3 = s0;
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm256_fmadd_pd(_mm256_loadu_pd(a+i), _mm256_loadu_pd(b+i), s0);
        s1 = _mm256_fmadd_pd(_mm256_loadu_pd(a+i+4), _mm256_loadu_pd(b+i+4), s1);
        s2 = _mm256_fmadd_pd(_mm256_loadu_pd(a+i+8), _mm256_loadu_pd(b+i+8), s2);
        s3 = _mm256_fmadd_pd(_mm256_loadu_pd(a+i+12), _mm256_loadu_pd(b+i+12), s3);
    }
    double sum = reduce(_mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3)));
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

double pcg_update(size_t n, size_t k, double *restrict y, double *restrict r,
                  double *restrict p, const double *restrict ap, const double *restrict v,
                  double rr, double inv_l, double *restrict gain) {
    if (k > 64) {
        return -1.;
    }
    double weights[64];
    __m256d w[64];
    double full_pap = dot_across(p, ap, n) * inv_l;
    double pap = full_pap;
    for (size_t j = 0; j < k; j++) {
        double c = dot_across(v + j*n, p, n);
        weights[j] = c;
        w[j] = _mm256_set1_pd(c);
        pap -= c * c;
    }
    if (!(pap > 1.e-10 * full_pap) || !(pap > 0.) || !isfinite(pap)) {
        return -1.;
    }
    double alpha = rr / pap;
    if (!isfinite(alpha)) {
        return -1.;
    }
    *gain = alpha * rr;
    __m256d ra = _mm256_set1_pd(alpha);
    __m256d inv = _mm256_set1_pd(inv_l);
    __m256d sum0 = _mm256_setzero_pd(), sum1 = sum0;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256d h0 = _mm256_mul_pd(_mm256_loadu_pd(ap+i), inv);
        __m256d h1 = _mm256_mul_pd(_mm256_loadu_pd(ap+i+4), inv);
        for (size_t j = 0; j < k; j++) {
            h0 = _mm256_fnmadd_pd(w[j], _mm256_loadu_pd(v+j*n+i), h0);
            h1 = _mm256_fnmadd_pd(w[j], _mm256_loadu_pd(v+j*n+i+4), h1);
        }
        __m256d r0 = _mm256_fnmadd_pd(ra, h0, _mm256_loadu_pd(r+i));
        __m256d r1 = _mm256_fnmadd_pd(ra, h1, _mm256_loadu_pd(r+i+4));
        _mm256_storeu_pd(r+i, r0);
        _mm256_storeu_pd(r+i+4, r1);
        _mm256_storeu_pd(y+i, _mm256_fmadd_pd(ra, _mm256_loadu_pd(p+i), _mm256_loadu_pd(y+i)));
        _mm256_storeu_pd(y+i+4, _mm256_fmadd_pd(ra, _mm256_loadu_pd(p+i+4), _mm256_loadu_pd(y+i+4)));
        sum0 = _mm256_fmadd_pd(r0, r0, sum0);
        sum1 = _mm256_fmadd_pd(r1, r1, sum1);
    }
    double newrr = reduce(_mm256_add_pd(sum0, sum1));
    for (; i < n; i++) {
        double h = inv_l * ap[i];
        for (size_t j = 0; j < k; j++) {
            h -= weights[j] * v[j*n+i];
        }
        r[i] -= alpha * h;
        y[i] += alpha * p[i];
        newrr += r[i]*r[i];
    }
    if (!isfinite(newrr)) {
        return -2.;
    }
    const double beta = newrr/rr;
    __m256d rb = _mm256_set1_pd(beta);
    for (i = 0; i + 4 <= n; i += 4) {
        _mm256_storeu_pd(p+i, _mm256_fmadd_pd(rb, _mm256_loadu_pd(p+i), _mm256_loadu_pd(r+i)));
    }
    for (; i < n; i++) {
        p[i] = r[i] + beta*p[i];
    }
    return newrr;
}
