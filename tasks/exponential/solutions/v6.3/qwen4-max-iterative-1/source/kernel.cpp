#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>

extern "C" {

static inline void spmv_default(int n, const int64_t* __restrict__ ap, const int32_t* __restrict__ aj,
                                const double* __restrict__ ax, const double* __restrict__ x,
                                double* __restrict__ y) {
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        const int64_t end = ap[i + 1];
        for (int64_t p = ap[i]; p < end; ++p) s += ax[p] * x[aj[p]];
        y[i] = s;
    }
}

__attribute__((target("avx2,fma")))
static inline void spmv_avx2(int n, const int64_t* __restrict__ ap, const int32_t* __restrict__ aj,
                             const double* __restrict__ ax, const double* __restrict__ x,
                             double* __restrict__ y) {
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        const int64_t end = ap[i + 1];
        for (int64_t p = ap[i]; p < end; ++p) s += ax[p] * x[aj[p]];
        y[i] = s;
    }
}

static inline int have_avx2() {
    static int mode = -1;
    if (mode < 0) mode = (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) ? 1 : 0;
    return mode;
}

static inline void spmv(int n, const int64_t* ap, const int32_t* aj,
                        const double* ax, const double* x, double* y) {
    if (have_avx2()) spmv_avx2(n, ap, aj, ax, x, y);
    else spmv_default(n, ap, aj, ax, x, y);
}

void fam_spmv(int n, const int64_t* ap, const int32_t* aj, const double* ax,
              const double* x, double* y) {
    spmv(n, ap, aj, ax, x, y);
}

double fam_inf_norm(int n, const int64_t* ap, const int32_t* aj, const double* ax) {
    double m = 0.0;
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int64_t p = ap[i]; p < ap[i + 1]; ++p) s += std::fabs(ax[p]);
        if (s > m) m = s;
    }
    return m;
}

double fam_gersh_lower(int n, const int64_t* ap, const int32_t* aj, const double* ax) {
    double lower = 1e300;
    for (int i = 0; i < n; ++i) {
        double diag = 0.0, off = 0.0;
        for (int64_t p = ap[i]; p < ap[i + 1]; ++p) {
            if (aj[p] == i) diag += ax[p];
            else off += std::fabs(ax[p]);
        }
        double v = diag - off;
        if (v < lower) lower = v;
    }
    return n ? lower : 0.0;
}

double fam_abs_upper(int n, const int64_t* ap, const int32_t* aj, const double* ax,
                     double inf_norm, int iters, double* work) {
    if (n <= 0 || inf_norm == 0.0) return 0.0;
    double* d = work;
    double* w = work + n;
    for (int i = 0; i < n; ++i) d[i] = 1.0;
    for (int it = 0; it < iters; ++it) {
        double mx = 0.0;
        for (int i = 0; i < n; ++i) {
            double s = 0.0;
            for (int64_t p = ap[i]; p < ap[i + 1]; ++p) s += std::fabs(ax[p]) * d[aj[p]];
            w[i] = s;
            if (s > mx) mx = s;
        }
        if (mx == 0.0) return 0.0;
        for (int i = 0; i < n; ++i) d[i] = (w[i] > 0.0) ? w[i] / mx : 1.0;
    }
    double upper = 0.0;
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int64_t p = ap[i]; p < ap[i + 1]; ++p) s += std::fabs(ax[p]) * d[aj[p]];
        double r = s / d[i];
        if (r > upper) upper = r;
    }
    return std::min(inf_norm, upper * (1.0 + 1e-10) + 1e-12 * inf_norm);
}

int fam_is_diagonal(int n, const int64_t* ap, const int32_t* aj, const double* ax) {
    for (int i = 0; i < n; ++i) {
        for (int64_t p = ap[i]; p < ap[i + 1]; ++p) {
            if (aj[p] != i && ax[p] != 0.0) return 0;
        }
    }
    return 1;
}

double fam_power_upper(int n, const int64_t* ap, const int32_t* aj, const double* ax,
                       double inf_norm, int iters, double* work) {
    if (n <= 0 || inf_norm == 0.0) return 0.0;
    double* v = work;
    double* w = work + n;
    uint64_t rng = 0x9e3779b97f4a7c15ULL ^ (uint64_t)n;
    for (int i = 0; i < n; ++i) {
        rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
        v[i] = (double)(rng * 2685821657736338717ULL) / 18446744073709551616.0 - 0.5;
    }
    double nv = 0.0;
    for (int i = 0; i < n; ++i) nv += v[i] * v[i];
    nv = std::sqrt(nv);
    if (nv == 0.0) return inf_norm;
    for (int i = 0; i < n; ++i) v[i] /= nv;
    double lambda = 0.0;
    for (int it = 0; it < iters; ++it) {
        spmv(n, ap, aj, ax, v, w);
        double dot = 0.0, norm = 0.0;
        for (int i = 0; i < n; ++i) { dot += v[i] * w[i]; norm += w[i] * w[i]; }
        lambda = dot;
        norm = std::sqrt(norm);
        if (norm == 0.0) return 0.0;
        for (int i = 0; i < n; ++i) v[i] = w[i] / norm;
    }
    spmv(n, ap, aj, ax, v, w);
    double dot = 0.0, res = 0.0;
    for (int i = 0; i < n; ++i) dot += v[i] * w[i];
    for (int i = 0; i < n; ++i) {
        double r = w[i] - dot * v[i];
        res += r * r;
    }
    res = std::sqrt(res);
    double upper = std::max(dot + res, dot * (1.0 + 1e-3)) + 1e-12 * inf_norm;
    return std::min(inf_norm, std::max(upper, 0.0));
}

static inline void cheb_step_default(int n, const int64_t* __restrict__ ap, const int32_t* __restrict__ aj,
                                     const double* __restrict__ ax, const double* __restrict__ u1,
                                     const double* __restrict__ u2, const double* __restrict__ b0,
                                     const double* __restrict__ b1, double ca, double tcb,
                                     double two_alpha, double* __restrict__ u0) {
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        const int64_t end = ap[i + 1];
        for (int64_t p = ap[i]; p < end; ++p) s += ax[p] * u1[aj[p]];
        u0[i] = ca * b0[i] + tcb * b1[i] + 2.0 * u1[i] - two_alpha * s - u2[i];
    }
}

__attribute__((target("avx2,fma")))
static inline void cheb_step_avx2(int n, const int64_t* __restrict__ ap, const int32_t* __restrict__ aj,
                                  const double* __restrict__ ax, const double* __restrict__ u1,
                                  const double* __restrict__ u2, const double* __restrict__ b0,
                                  const double* __restrict__ b1, double ca, double tcb,
                                  double two_alpha, double* __restrict__ u0) {
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        const int64_t end = ap[i + 1];
        for (int64_t p = ap[i]; p < end; ++p) s += ax[p] * u1[aj[p]];
        u0[i] = ca * b0[i] + tcb * b1[i] + 2.0 * u1[i] - two_alpha * s - u2[i];
    }
}

static inline void cheb_final_default(int n, const int64_t* __restrict__ ap, const int32_t* __restrict__ aj,
                                      const double* __restrict__ ax, const double* __restrict__ u1,
                                      const double* __restrict__ u2, const double* __restrict__ b0,
                                      const double* __restrict__ b1, double ca, double tcb,
                                      double alpha, double* __restrict__ y) {
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        const int64_t end = ap[i + 1];
        for (int64_t p = ap[i]; p < end; ++p) s += ax[p] * u1[aj[p]];
        y[i] = ca * b0[i] + tcb * b1[i] + u1[i] - alpha * s - u2[i];
    }
}

__attribute__((target("avx2,fma")))
static inline void cheb_final_avx2(int n, const int64_t* __restrict__ ap, const int32_t* __restrict__ aj,
                                   const double* __restrict__ ax, const double* __restrict__ u1,
                                   const double* __restrict__ u2, const double* __restrict__ b0,
                                   const double* __restrict__ b1, double ca, double tcb,
                                   double alpha, double* __restrict__ y) {
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        const int64_t end = ap[i + 1];
        for (int64_t p = ap[i]; p < end; ++p) s += ax[p] * u1[aj[p]];
        y[i] = ca * b0[i] + tcb * b1[i] + u1[i] - alpha * s - u2[i];
    }
}

void fam_cheb(int n, const int64_t* ap, const int32_t* aj, const double* ax,
              double t, double eta, const double* ca, const double* cb, int N,
              const double* b0, const double* b1, double* y, double* work) {
    if (N <= 0 || eta == 0.0) {
        for (int i = 0; i < n; ++i) y[i] = b0[i] + t * b1[i];
        return;
    }
    double alpha = 2.0 * t / eta;
    double* u0 = work;
    double* u1 = work + n;
    double* u2 = work + 2 * n;
    std::memset(u1, 0, n * sizeof(double));
    std::memset(u2, 0, n * sizeof(double));
    for (int i = 0; i < n; ++i) u1[i] = ca[N] * b0[i] + t * cb[N] * b1[i];
    for (int k = N - 1; k >= 1; --k) {
        if (have_avx2())
            cheb_step_avx2(n, ap, aj, ax, u1, u2, b0, b1, ca[k], t * cb[k], 2.0 * alpha, u0);
        else
            cheb_step_default(n, ap, aj, ax, u1, u2, b0, b1, ca[k], t * cb[k], 2.0 * alpha, u0);
        double* tmp = u2; u2 = u1; u1 = u0; u0 = tmp;
    }
    if (have_avx2())
        cheb_final_avx2(n, ap, aj, ax, u1, u2, b0, b1, ca[0], t * cb[0], alpha, y);
    else
        cheb_final_default(n, ap, aj, ax, u1, u2, b0, b1, ca[0], t * cb[0], alpha, y);
}

static inline void taylor_step_default(int n, const int64_t* __restrict__ ap, const int32_t* __restrict__ aj,
                                       const double* __restrict__ ax, const double* __restrict__ u,
                                       const double* __restrict__ b0, const double* __restrict__ b1,
                                       double c, double d, double t, double* __restrict__ v) {
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        const int64_t end = ap[i + 1];
        for (int64_t p = ap[i]; p < end; ++p) s += ax[p] * u[aj[p]];
        v[i] = c * b0[i] + d * b1[i] - t * s;
    }
}

__attribute__((target("avx2,fma")))
static inline void taylor_step_avx2(int n, const int64_t* __restrict__ ap, const int32_t* __restrict__ aj,
                                    const double* __restrict__ ax, const double* __restrict__ u,
                                    const double* __restrict__ b0, const double* __restrict__ b1,
                                    double c, double d, double t, double* __restrict__ v) {
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        const int64_t end = ap[i + 1];
        for (int64_t p = ap[i]; p < end; ++p) s += ax[p] * u[aj[p]];
        v[i] = c * b0[i] + d * b1[i] - t * s;
    }
}

void fam_taylor(int n, const int64_t* ap, const int32_t* aj, const double* ax,
                double t, const double* c, const double* d, int K,
                const double* b0, const double* b1, double* y, double* work) {
    double* u = work;
    double* v = work + n;
    for (int i = 0; i < n; ++i) u[i] = c[K] * b0[i] + d[K] * b1[i];
    for (int k = K - 1; k >= 0; --k) {
        if (have_avx2()) taylor_step_avx2(n, ap, aj, ax, u, b0, b1, c[k], d[k], t, v);
        else taylor_step_default(n, ap, aj, ax, u, b0, b1, c[k], d[k], t, v);
        double* tmp = u; u = v; v = tmp;
    }
    std::memcpy(y, u, n * sizeof(double));
}

void fam_to_ell(int n, const int64_t* ap, const int32_t* aj, const double* ax,
                int maxk, int32_t* eaj, double* eax) {
    for (int i = 0; i < n; ++i) {
        int64_t p = ap[i], end = ap[i + 1];
        int32_t* dstj = eaj + (int64_t)i * maxk;
        double* dstx = eax + (int64_t)i * maxk;
        int k = 0;
        for (; p < end; ++p, ++k) { dstj[k] = aj[p]; dstx[k] = ax[p]; }
        for (; k < maxk; ++k) { dstj[k] = 0; dstx[k] = 0.0; }
    }
}

static inline void ell_cheb_step_default(int n, int maxk, const int32_t* __restrict__ aj,
                                         const double* __restrict__ ax, const double* __restrict__ u1,
                                         const double* __restrict__ u2, const double* __restrict__ b0,
                                         const double* __restrict__ b1, double ca, double tcb,
                                         double two_alpha, double* __restrict__ u0) {
    for (int i = 0; i < n; ++i) {
        const double* v = ax + (int64_t)i * maxk;
        const int32_t* c = aj + (int64_t)i * maxk;
        double s = 0.0;
        for (int k = 0; k < maxk; ++k) s += v[k] * u1[c[k]];
        u0[i] = ca * b0[i] + tcb * b1[i] + 2.0 * u1[i] - two_alpha * s - u2[i];
    }
}

__attribute__((target("avx2,fma")))
static inline void ell_cheb_step_avx2(int n, int maxk, const int32_t* __restrict__ aj,
                                      const double* __restrict__ ax, const double* __restrict__ u1,
                                      const double* __restrict__ u2, const double* __restrict__ b0,
                                      const double* __restrict__ b1, double ca, double tcb,
                                      double two_alpha, double* __restrict__ u0) {
    for (int i = 0; i < n; ++i) {
        const double* v = ax + (int64_t)i * maxk;
        const int32_t* c = aj + (int64_t)i * maxk;
        double s = 0.0;
        for (int k = 0; k < maxk; ++k) s += v[k] * u1[c[k]];
        u0[i] = ca * b0[i] + tcb * b1[i] + 2.0 * u1[i] - two_alpha * s - u2[i];
    }
}

static inline void ell_cheb_final_default(int n, int maxk, const int32_t* __restrict__ aj,
                                          const double* __restrict__ ax, const double* __restrict__ u1,
                                          const double* __restrict__ u2, const double* __restrict__ b0,
                                          const double* __restrict__ b1, double ca, double tcb,
                                          double alpha, double* __restrict__ y) {
    for (int i = 0; i < n; ++i) {
        const double* v = ax + (int64_t)i * maxk;
        const int32_t* c = aj + (int64_t)i * maxk;
        double s = 0.0;
        for (int k = 0; k < maxk; ++k) s += v[k] * u1[c[k]];
        y[i] = ca * b0[i] + tcb * b1[i] + u1[i] - alpha * s - u2[i];
    }
}

__attribute__((target("avx2,fma")))
static inline void ell_cheb_final_avx2(int n, int maxk, const int32_t* __restrict__ aj,
                                       const double* __restrict__ ax, const double* __restrict__ u1,
                                       const double* __restrict__ u2, const double* __restrict__ b0,
                                       const double* __restrict__ b1, double ca, double tcb,
                                       double alpha, double* __restrict__ y) {
    for (int i = 0; i < n; ++i) {
        const double* v = ax + (int64_t)i * maxk;
        const int32_t* c = aj + (int64_t)i * maxk;
        double s = 0.0;
        for (int k = 0; k < maxk; ++k) s += v[k] * u1[c[k]];
        y[i] = ca * b0[i] + tcb * b1[i] + u1[i] - alpha * s - u2[i];
    }
}

void fam_cheb_ell(int n, int maxk, const int32_t* aj, const double* ax,
                  double t, double eta, const double* ca, const double* cb, int N,
                  const double* b0, const double* b1, double* y, double* work) {
    if (N <= 0 || eta == 0.0) {
        for (int i = 0; i < n; ++i) y[i] = b0[i] + t * b1[i];
        return;
    }
    double alpha = 2.0 * t / eta;
    double* u0 = work;
    double* u1 = work + n;
    double* u2 = work + 2 * n;
    std::memset(u1, 0, n * sizeof(double));
    std::memset(u2, 0, n * sizeof(double));
    for (int i = 0; i < n; ++i) u1[i] = ca[N] * b0[i] + t * cb[N] * b1[i];
    for (int k = N - 1; k >= 1; --k) {
        if (have_avx2())
            ell_cheb_step_avx2(n, maxk, aj, ax, u1, u2, b0, b1, ca[k], t * cb[k], 2.0 * alpha, u0);
        else
            ell_cheb_step_default(n, maxk, aj, ax, u1, u2, b0, b1, ca[k], t * cb[k], 2.0 * alpha, u0);
        double* tmp = u2; u2 = u1; u1 = u0; u0 = tmp;
    }
    if (have_avx2())
        ell_cheb_final_avx2(n, maxk, aj, ax, u1, u2, b0, b1, ca[0], t * cb[0], alpha, y);
    else
        ell_cheb_final_default(n, maxk, aj, ax, u1, u2, b0, b1, ca[0], t * cb[0], alpha, y);
}

static inline void ell_taylor_step_default(int n, int maxk, const int32_t* __restrict__ aj,
                                           const double* __restrict__ ax, const double* __restrict__ u,
                                           const double* __restrict__ b0, const double* __restrict__ b1,
                                           double c, double d, double t, double* __restrict__ v) {
    for (int i = 0; i < n; ++i) {
        const double* vv = ax + (int64_t)i * maxk;
        const int32_t* cc = aj + (int64_t)i * maxk;
        double s = 0.0;
        for (int k = 0; k < maxk; ++k) s += vv[k] * u[cc[k]];
        v[i] = c * b0[i] + d * b1[i] - t * s;
    }
}

__attribute__((target("avx2,fma")))
static inline void ell_taylor_step_avx2(int n, int maxk, const int32_t* __restrict__ aj,
                                        const double* __restrict__ ax, const double* __restrict__ u,
                                        const double* __restrict__ b0, const double* __restrict__ b1,
                                        double c, double d, double t, double* __restrict__ v) {
    for (int i = 0; i < n; ++i) {
        const double* vv = ax + (int64_t)i * maxk;
        const int32_t* cc = aj + (int64_t)i * maxk;
        double s = 0.0;
        for (int k = 0; k < maxk; ++k) s += vv[k] * u[cc[k]];
        v[i] = c * b0[i] + d * b1[i] - t * s;
    }
}

void fam_taylor_ell(int n, int maxk, const int32_t* aj, const double* ax,
                    double t, const double* c, const double* d, int K,
                    const double* b0, const double* b1, double* y, double* work) {
    double* u = work;
    double* v = work + n;
    for (int i = 0; i < n; ++i) u[i] = c[K] * b0[i] + d[K] * b1[i];
    for (int k = K - 1; k >= 0; --k) {
        if (have_avx2()) ell_taylor_step_avx2(n, maxk, aj, ax, u, b0, b1, c[k], d[k], t, v);
        else ell_taylor_step_default(n, maxk, aj, ax, u, b0, b1, c[k], d[k], t, v);
        double* tmp = u; u = v; v = tmp;
    }
    std::memcpy(y, u, n * sizeof(double));
}

static inline void inv_step_default(int n, const int64_t* __restrict__ ap, const int32_t* __restrict__ aj,
                                    const double* __restrict__ ax, const double* __restrict__ u1,
                                    const double* __restrict__ u2, const double* __restrict__ b,
                                    double c, double beta, double two_gamma, double* __restrict__ u0) {
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        const int64_t end = ap[i + 1];
        for (int64_t p = ap[i]; p < end; ++p) s += ax[p] * u1[aj[p]];
        u0[i] = c * b[i] + 2.0 * beta * s - two_gamma * u1[i] - u2[i];
    }
}

__attribute__((target("avx2,fma")))
static inline void inv_step_avx2(int n, const int64_t* __restrict__ ap, const int32_t* __restrict__ aj,
                                 const double* __restrict__ ax, const double* __restrict__ u1,
                                 const double* __restrict__ u2, const double* __restrict__ b,
                                 double c, double beta, double two_gamma, double* __restrict__ u0) {
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        const int64_t end = ap[i + 1];
        for (int64_t p = ap[i]; p < end; ++p) s += ax[p] * u1[aj[p]];
        u0[i] = c * b[i] + 2.0 * beta * s - two_gamma * u1[i] - u2[i];
    }
}

static inline void inv_final_default(int n, const int64_t* __restrict__ ap, const int32_t* __restrict__ aj,
                                     const double* __restrict__ ax, const double* __restrict__ u1,
                                     const double* __restrict__ u2, const double* __restrict__ b,
                                     double c, double beta, double gamma, double* __restrict__ y) {
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        const int64_t end = ap[i + 1];
        for (int64_t p = ap[i]; p < end; ++p) s += ax[p] * u1[aj[p]];
        y[i] = c * b[i] + beta * s - gamma * u1[i] - u2[i];
    }
}

__attribute__((target("avx2,fma")))
static inline void inv_final_avx2(int n, const int64_t* __restrict__ ap, const int32_t* __restrict__ aj,
                                  const double* __restrict__ ax, const double* __restrict__ u1,
                                  const double* __restrict__ u2, const double* __restrict__ b,
                                  double c, double beta, double gamma, double* __restrict__ y) {
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        const int64_t end = ap[i + 1];
        for (int64_t p = ap[i]; p < end; ++p) s += ax[p] * u1[aj[p]];
        y[i] = c * b[i] + beta * s - gamma * u1[i] - u2[i];
    }
}

void fam_inv_cheb(int n, const int64_t* ap, const int32_t* aj, const double* ax,
                  double lo, double hi, const double* c, int M,
                  const double* b, double* y, double* work) {
    if (M <= 0) {
        for (int i = 0; i < n; ++i) y[i] = c[0] * b[i];
        return;
    }
    double beta = 2.0 / (hi - lo);
    double gamma = (hi + lo) / (hi - lo);
    double* u0 = work;
    double* u1 = work + n;
    double* u2 = work + 2 * n;
    std::memset(u1, 0, n * sizeof(double));
    std::memset(u2, 0, n * sizeof(double));
    for (int i = 0; i < n; ++i) u1[i] = c[M] * b[i];
    for (int k = M - 1; k >= 1; --k) {
        if (have_avx2()) inv_step_avx2(n, ap, aj, ax, u1, u2, b, c[k], beta, 2.0 * gamma, u0);
        else inv_step_default(n, ap, aj, ax, u1, u2, b, c[k], beta, 2.0 * gamma, u0);
        double* tmp = u2; u2 = u1; u1 = u0; u0 = tmp;
    }
    if (have_avx2()) inv_final_avx2(n, ap, aj, ax, u1, u2, b, c[0], beta, gamma, y);
    else inv_final_default(n, ap, aj, ax, u1, u2, b, c[0], beta, gamma, y);
}

static inline void inv_ell_step_default(int n, int maxk, const int32_t* __restrict__ aj,
                                        const double* __restrict__ ax, const double* __restrict__ u1,
                                        const double* __restrict__ u2, const double* __restrict__ b,
                                        double c, double beta, double two_gamma, double* __restrict__ u0) {
    for (int i = 0; i < n; ++i) {
        const double* v = ax + (int64_t)i * maxk;
        const int32_t* cc = aj + (int64_t)i * maxk;
        double s = 0.0;
        for (int k = 0; k < maxk; ++k) s += v[k] * u1[cc[k]];
        u0[i] = c * b[i] + 2.0 * beta * s - two_gamma * u1[i] - u2[i];
    }
}

__attribute__((target("avx2,fma")))
static inline void inv_ell_step_avx2(int n, int maxk, const int32_t* __restrict__ aj,
                                     const double* __restrict__ ax, const double* __restrict__ u1,
                                     const double* __restrict__ u2, const double* __restrict__ b,
                                     double c, double beta, double two_gamma, double* __restrict__ u0) {
    for (int i = 0; i < n; ++i) {
        const double* v = ax + (int64_t)i * maxk;
        const int32_t* cc = aj + (int64_t)i * maxk;
        double s = 0.0;
        for (int k = 0; k < maxk; ++k) s += v[k] * u1[cc[k]];
        u0[i] = c * b[i] + 2.0 * beta * s - two_gamma * u1[i] - u2[i];
    }
}

static inline void inv_ell_final_default(int n, int maxk, const int32_t* __restrict__ aj,
                                         const double* __restrict__ ax, const double* __restrict__ u1,
                                         const double* __restrict__ u2, const double* __restrict__ b,
                                         double c, double beta, double gamma, double* __restrict__ y) {
    for (int i = 0; i < n; ++i) {
        const double* v = ax + (int64_t)i * maxk;
        const int32_t* cc = aj + (int64_t)i * maxk;
        double s = 0.0;
        for (int k = 0; k < maxk; ++k) s += v[k] * u1[cc[k]];
        y[i] = c * b[i] + beta * s - gamma * u1[i] - u2[i];
    }
}

__attribute__((target("avx2,fma")))
static inline void inv_ell_final_avx2(int n, int maxk, const int32_t* __restrict__ aj,
                                      const double* __restrict__ ax, const double* __restrict__ u1,
                                      const double* __restrict__ u2, const double* __restrict__ b,
                                      double c, double beta, double gamma, double* __restrict__ y) {
    for (int i = 0; i < n; ++i) {
        const double* v = ax + (int64_t)i * maxk;
        const int32_t* cc = aj + (int64_t)i * maxk;
        double s = 0.0;
        for (int k = 0; k < maxk; ++k) s += v[k] * u1[cc[k]];
        y[i] = c * b[i] + beta * s - gamma * u1[i] - u2[i];
    }
}

void fam_inv_cheb_ell(int n, int maxk, const int32_t* aj, const double* ax,
                      double lo, double hi, const double* c, int M,
                      const double* b, double* y, double* work) {
    if (M <= 0) {
        for (int i = 0; i < n; ++i) y[i] = c[0] * b[i];
        return;
    }
    double beta = 2.0 / (hi - lo);
    double gamma = (hi + lo) / (hi - lo);
    double* u0 = work;
    double* u1 = work + n;
    double* u2 = work + 2 * n;
    std::memset(u1, 0, n * sizeof(double));
    std::memset(u2, 0, n * sizeof(double));
    for (int i = 0; i < n; ++i) u1[i] = c[M] * b[i];
    for (int k = M - 1; k >= 1; --k) {
        if (have_avx2()) inv_ell_step_avx2(n, maxk, aj, ax, u1, u2, b, c[k], beta, 2.0 * gamma, u0);
        else inv_ell_step_default(n, maxk, aj, ax, u1, u2, b, c[k], beta, 2.0 * gamma, u0);
        double* tmp = u2; u2 = u1; u1 = u0; u0 = tmp;
    }
    if (have_avx2()) inv_ell_final_avx2(n, maxk, aj, ax, u1, u2, b, c[0], beta, gamma, y);
    else inv_ell_final_default(n, maxk, aj, ax, u1, u2, b, c[0], beta, gamma, y);
}

int64_t fam_sell_plan(int n, const int64_t* ap, const int32_t* aj, const double* ax,
                      int sigma, int32_t* rowperm, int32_t* chunk_max, int64_t* offsets) {
    const int C = 32;
    int nch = (n + C - 1) / C;
    for (int st = 0; st < n; st += sigma) {
        int en = std::min(n, st + sigma);
        for (int i = st; i < en; ++i) rowperm[i] = i;
        std::sort(rowperm + st, rowperm + en, [&](int32_t a, int32_t b) {
            return (ap[a + 1] - ap[a]) < (ap[b + 1] - ap[b]);
        });
    }
    int64_t total = 0;
    offsets[0] = 0;
    for (int ch = 0; ch < nch; ++ch) {
        int base = ch * C, rows = std::min(C, n - base), mk = 0;
        for (int r = 0; r < rows; ++r) {
            int i = rowperm[base + r];
            int l = (int)(ap[i + 1] - ap[i]);
            if (l > mk) mk = l;
        }
        chunk_max[ch] = mk;
        total += (int64_t)C * mk;
        offsets[ch + 1] = total;
    }
    return total;
}

void fam_sell_fill(int n, const int64_t* ap, const int32_t* aj, const double* ax,
                   const int32_t* rowperm, const int32_t* chunk_max, const int64_t* offsets,
                   int32_t* saj, double* sax) {
    const int C = 32;
    int nch = (n + C - 1) / C;
    for (int ch = 0; ch < nch; ++ch) {
        int base = ch * C, rows = std::min(C, n - base), mk = chunk_max[ch];
        int64_t off = offsets[ch];
        for (int k = 0; k < mk; ++k) {
            for (int r = 0; r < C; ++r) { saj[off + k * C + r] = 0; sax[off + k * C + r] = 0.0; }
        }
        for (int r = 0; r < rows; ++r) {
            int i = rowperm[base + r];
            int64_t p = ap[i], end = ap[i + 1];
            for (int k = 0; p < end; ++p, ++k) {
                saj[off + k * C + r] = aj[p];
                sax[off + k * C + r] = ax[p];
            }
        }
    }
}

static inline void sell_cheb_step_default(int n, const int32_t* __restrict__ rowperm,
                                          const int32_t* __restrict__ saj, const double* __restrict__ sax,
                                          const int32_t* __restrict__ chunk_max, const int64_t* __restrict__ offsets,
                                          const double* __restrict__ u1, const double* __restrict__ u2,
                                          const double* __restrict__ b0, const double* __restrict__ b1,
                                          double ca, double tcb, double two_alpha, double* __restrict__ u0) {
    const int C = 32;
    int nch = (n + C - 1) / C;
    for (int ch = 0; ch < nch; ++ch) {
        int base = ch * C, rows = std::min(C, n - base), mk = chunk_max[ch];
        const int32_t* ajc = saj + offsets[ch];
        const double* axc = sax + offsets[ch];
        double s[32] = {0};
        for (int k = 0; k < mk; ++k) {
            for (int r = 0; r < rows; ++r) s[r] += axc[k * C + r] * u1[ajc[k * C + r]];
        }
        for (int r = 0; r < rows; ++r) {
            int i = rowperm[base + r];
            u0[i] = ca * b0[i] + tcb * b1[i] + 2.0 * u1[i] - two_alpha * s[r] - u2[i];
        }
    }
}

__attribute__((target("avx2,fma")))
static inline void sell_cheb_step_avx2(int n, const int32_t* __restrict__ rowperm,
                                       const int32_t* __restrict__ saj, const double* __restrict__ sax,
                                       const int32_t* __restrict__ chunk_max, const int64_t* __restrict__ offsets,
                                       const double* __restrict__ u1, const double* __restrict__ u2,
                                       const double* __restrict__ b0, const double* __restrict__ b1,
                                       double ca, double tcb, double two_alpha, double* __restrict__ u0) {
    const int C = 32;
    int nch = (n + C - 1) / C;
    for (int ch = 0; ch < nch; ++ch) {
        int base = ch * C, rows = std::min(C, n - base), mk = chunk_max[ch];
        const int32_t* ajc = saj + offsets[ch];
        const double* axc = sax + offsets[ch];
        double s[32] = {0};
        for (int k = 0; k < mk; ++k) {
            for (int r = 0; r < rows; ++r) s[r] += axc[k * C + r] * u1[ajc[k * C + r]];
        }
        for (int r = 0; r < rows; ++r) {
            int i = rowperm[base + r];
            u0[i] = ca * b0[i] + tcb * b1[i] + 2.0 * u1[i] - two_alpha * s[r] - u2[i];
        }
    }
}

static inline void sell_cheb_final_default(int n, const int32_t* __restrict__ rowperm,
                                           const int32_t* __restrict__ saj, const double* __restrict__ sax,
                                           const int32_t* __restrict__ chunk_max, const int64_t* __restrict__ offsets,
                                           const double* __restrict__ u1, const double* __restrict__ u2,
                                           const double* __restrict__ b0, const double* __restrict__ b1,
                                           double ca, double tcb, double alpha, double* __restrict__ y) {
    const int C = 32;
    int nch = (n + C - 1) / C;
    for (int ch = 0; ch < nch; ++ch) {
        int base = ch * C, rows = std::min(C, n - base), mk = chunk_max[ch];
        const int32_t* ajc = saj + offsets[ch];
        const double* axc = sax + offsets[ch];
        double s[32] = {0};
        for (int k = 0; k < mk; ++k) {
            for (int r = 0; r < rows; ++r) s[r] += axc[k * C + r] * u1[ajc[k * C + r]];
        }
        for (int r = 0; r < rows; ++r) {
            int i = rowperm[base + r];
            y[i] = ca * b0[i] + tcb * b1[i] + u1[i] - alpha * s[r] - u2[i];
        }
    }
}

__attribute__((target("avx2,fma")))
static inline void sell_cheb_final_avx2(int n, const int32_t* __restrict__ rowperm,
                                        const int32_t* __restrict__ saj, const double* __restrict__ sax,
                                        const int32_t* __restrict__ chunk_max, const int64_t* __restrict__ offsets,
                                        const double* __restrict__ u1, const double* __restrict__ u2,
                                        const double* __restrict__ b0, const double* __restrict__ b1,
                                        double ca, double tcb, double alpha, double* __restrict__ y) {
    const int C = 32;
    int nch = (n + C - 1) / C;
    for (int ch = 0; ch < nch; ++ch) {
        int base = ch * C, rows = std::min(C, n - base), mk = chunk_max[ch];
        const int32_t* ajc = saj + offsets[ch];
        const double* axc = sax + offsets[ch];
        double s[32] = {0};
        for (int k = 0; k < mk; ++k) {
            for (int r = 0; r < rows; ++r) s[r] += axc[k * C + r] * u1[ajc[k * C + r]];
        }
        for (int r = 0; r < rows; ++r) {
            int i = rowperm[base + r];
            y[i] = ca * b0[i] + tcb * b1[i] + u1[i] - alpha * s[r] - u2[i];
        }
    }
}

void fam_cheb_sell(int n, const int32_t* rowperm, const int32_t* saj, const double* sax,
                   const int32_t* chunk_max, const int64_t* offsets,
                   double t, double eta, const double* ca, const double* cb, int N,
                   const double* b0, const double* b1, double* y, double* work) {
    if (N <= 0 || eta == 0.0) {
        for (int i = 0; i < n; ++i) y[i] = b0[i] + t * b1[i];
        return;
    }
    double alpha = 2.0 * t / eta;
    double* u0 = work;
    double* u1 = work + n;
    double* u2 = work + 2 * n;
    std::memset(u1, 0, n * sizeof(double));
    std::memset(u2, 0, n * sizeof(double));
    for (int i = 0; i < n; ++i) u1[i] = ca[N] * b0[i] + t * cb[N] * b1[i];
    for (int k = N - 1; k >= 1; --k) {
        if (have_avx2())
            sell_cheb_step_avx2(n, rowperm, saj, sax, chunk_max, offsets, u1, u2, b0, b1, ca[k], t * cb[k], 2.0 * alpha, u0);
        else
            sell_cheb_step_default(n, rowperm, saj, sax, chunk_max, offsets, u1, u2, b0, b1, ca[k], t * cb[k], 2.0 * alpha, u0);
        double* tmp = u2; u2 = u1; u1 = u0; u0 = tmp;
    }
    if (have_avx2())
        sell_cheb_final_avx2(n, rowperm, saj, sax, chunk_max, offsets, u1, u2, b0, b1, ca[0], t * cb[0], alpha, y);
    else
        sell_cheb_final_default(n, rowperm, saj, sax, chunk_max, offsets, u1, u2, b0, b1, ca[0], t * cb[0], alpha, y);
}

}
