/* FAMEXP2 solver kernel: CSR SpMV and Chebyshev-Clenshaw propagation.
 *
 * We approximate y = exp(-t L) b0 + t phi1(-t L) b1 by a single Chebyshev
 * polynomial of S = 2 L / lmax - I.  The two scalar functions share the same
 * Clenshaw recurrence (their source vectors are added), so one sparse matvec
 * per degree serves both right-hand sides.
 *
 * Build: gcc -O3 -shared -fPIC -o libfamexp.so kernel.c -lm
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

/* ------------------------------------------------------------------ */
/* Spectral upper bound.                                               */
/*                                                                     */
/* For symmetric L, lambda_max(L) <= rho(|L|).  Collatz-Wielandt says    */
/* rho(|L|) <= max_i (|L| d)_i / d_i for any d > 0.  Power iterations on  */
/* |L| tighten that rigorous bound well beyond the raw row-sum bound.    */
/* w must point to 2*n scratch doubles.                                  */
/* ------------------------------------------------------------------ */
EXPORT double famexp_lmax_bound(const int64_t *ip, const uint32_t *idx,
                                const double *dat, int n, int iters, double *w) {
    double *d = w, *r = w + n;
    double gersh = 0.0;
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int64_t j = ip[i]; j < ip[i + 1]; ++j) s += fabs(dat[j]);
        if (s > gersh) gersh = s;
    }
    if (!(gersh > 0.0)) return 0.0;
    for (int i = 0; i < n; ++i) d[i] = 1.0;
    double best = gersh;
    double prev = gersh;
    for (int it = 0; it < iters; ++it) {
        double mx = 0.0, rm = 0.0;
        for (int i = 0; i < n; ++i) {
            double s = 0.0;
            for (int64_t j = ip[i]; j < ip[i + 1]; ++j)
                s += fabs(dat[j]) * d[idx[j]];
            r[i] = s;
            double ratio = s / d[i];
            if (ratio > mx) mx = ratio;
            if (s > rm) rm = s;
        }
        if (mx < best) best = mx;
        if (!(rm > 0.0)) break;
        if (it >= 2 && prev - mx <= 1e-3 * mx) break;  /* converged */
        prev = mx;
        double inv = 1.0 / rm;
        for (int i = 0; i < n; ++i) d[i] = r[i] * inv + 1e-12;
    }
    return best;
}

/* ------------------------------------------------------------------ */
/* Cheap structural checks and connected components of the sparsity     */
/* graph.  Matrices made of many tiny independent blocks admit an exact */
/* dense block-diagonal solution with no polynomial iterations at all.  */
/* ------------------------------------------------------------------ */
EXPORT void famexp_pattern_stats(const int64_t *ip, const uint32_t *idx,
                                 int n, int *max_row, int *max_band) {
    int mr = 0, mb = 0;
    for (int i = 0; i < n; ++i) {
        int64_t a = ip[i], e = ip[i + 1];
        int cnt = (int)(e - a);
        if (cnt > mr) mr = cnt;
        if (cnt > 1) {
            int64_t band = (int64_t)idx[e - 1] - (int64_t)idx[a];
            if (band > mb) mb = (int)band;
        }
    }
    *max_row = mr;
    *max_band = mb;
}

static int uf_find(int *p, int x) {
    while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; }
    return x;
}

/* label[i] = compact component id, sizes[k] = block size.               */
/* Returns number of components; *maxsize is the largest block.          */
EXPORT int famexp_components(const int64_t *ip, const uint32_t *idx,
                             int n, int32_t *label, int32_t *sizes,
                             int *maxsize) {
    int *p = (int *)malloc(sizeof(int) * (size_t)n);
    int *rank = (int *)malloc(sizeof(int) * (size_t)n);
    if (!p || !rank) { free(p); free(rank); return -1; }
    for (int i = 0; i < n; ++i) { p[i] = i; rank[i] = 0; }
    for (int i = 0; i < n; ++i) {
        for (int64_t j = ip[i]; j < ip[i + 1]; ++j) {
            int ri = uf_find(p, i), rj = uf_find(p, (int)idx[j]);
            if (ri == rj) continue;
            if (rank[ri] < rank[rj]) { int t = ri; ri = rj; rj = t; }
            p[rj] = ri;
            if (rank[ri] == rank[rj]) ++rank[ri];
        }
    }
    int *root2id = (int *)malloc(sizeof(int) * (size_t)n);
    if (!root2id) { free(p); free(rank); return -1; }
    for (int i = 0; i < n; ++i) root2id[i] = -1;
    int nc = 0;
    for (int i = 0; i < n; ++i) {
        int r = uf_find(p, i);
        if (root2id[r] < 0) root2id[r] = nc++;
        label[i] = root2id[r];
    }
    for (int k = 0; k < nc; ++k) sizes[k] = 0;
    int mx = 0;
    for (int i = 0; i < n; ++i) {
        int s = ++sizes[label[i]];
        if (s > mx) mx = s;
    }
    *maxsize = mx;
    free(root2id); free(p); free(rank);
    return nc;
}

/* ------------------------------------------------------------------ */
/* Double-precision Clenshaw.                                          */
/*                                                                     */
/* dat must be pre-scaled by 4/lmax (so 2*S*v = dat*v - 2*v).            */
/* c,d: (m+1) coefficients for exp and phi1.                            */
/* work: 3*n doubles.  Result in y (n doubles).                         */
/* ------------------------------------------------------------------ */
EXPORT void famexp_clenshaw_d(const int64_t *ip, const uint32_t *idx,
                              const double *dat, int n, int m,
                              const double *c, const double *d,
                              const double *b0, const double *b1,
                              double *y, double *w) {
    double *p0 = w, *p1 = w + n, *p2 = w + 2 * n;
    for (int i = 0; i < n; ++i) p0[i] = p1[i] = p2[i] = 0.0;
    /* p1 = b_{k+1}, p2 = b_{k+2}, p0 = scratch / new state */
    for (int k = m; k >= 1; --k) {
        double ck = c[k], dk = d[k];
        for (int i = 0; i < n; ++i) {
            double s = 0.0;
            for (int64_t jj = ip[i]; jj < ip[i + 1]; ++jj)
                s += dat[jj] * p1[idx[jj]];
            p0[i] = s - 2.0 * p1[i] - p2[i] + ck * b0[i] + dk * b1[i];
        }
        double *tmp = p2; p2 = p1; p1 = p0; p0 = tmp;
    }
    double c0 = c[0], d0 = d[0];
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int64_t jj = ip[i]; jj < ip[i + 1]; ++jj)
            s += dat[jj] * p1[idx[jj]];
        y[i] = c0 * b0[i] + d0 * b1[i] + 0.5 * s - p1[i] - p2[i];
    }
}

/* ------------------------------------------------------------------ */
/* Extended-precision (80-bit on x86) Clenshaw.                        */
/* w: 3*n long doubles.                                                */
/* ------------------------------------------------------------------ */
EXPORT void famexp_clenshaw_l(const int64_t *ip, const uint32_t *idx,
                              const double *dat, int n, int m,
                              const double *c, const double *d,
                              const double *b0, const double *b1,
                              double *y, long double *w) {
    long double *p0 = w, *p1 = w + n, *p2 = w + 2 * n;
    for (int i = 0; i < n; ++i) p0[i] = p1[i] = p2[i] = 0.0L;
    for (int k = m; k >= 1; --k) {
        long double ck = (long double)c[k], dk = (long double)d[k];
        for (int i = 0; i < n; ++i) {
            long double s = 0.0L;
            for (int64_t jj = ip[i]; jj < ip[i + 1]; ++jj)
                s += (long double)dat[jj] * p1[idx[jj]];
            p0[i] = s - 2.0L * p1[i] - p2[i]
                    + ck * (long double)b0[i] + dk * (long double)b1[i];
        }
        long double *tmp = p2; p2 = p1; p1 = p0; p0 = tmp;
    }
    long double c0 = (long double)c[0], d0 = (long double)d[0];
    for (int i = 0; i < n; ++i) {
        long double s = 0.0L;
        for (int64_t jj = ip[i]; jj < ip[i + 1]; ++jj)
            s += (long double)dat[jj] * p1[idx[jj]];
        y[i] = (double)(c0 * (long double)b0[i] + d0 * (long double)b1[i]
                        + 0.5L * s - p1[i] - p2[i]);
    }
}

/* Scale CSR data in place by a factor (used for 4/lmax). */
EXPORT void famexp_scale(double *dat, int nnz, double factor) {
    for (int j = 0; j < nnz; ++j) dat[j] *= factor;
}


/* ------------------------------------------------------------------ */
/* Interleaved-row variants: several independent accumulation chains     */
/* hide gather latency on irregular sparsity patterns.                   */
/* ------------------------------------------------------------------ */
EXPORT void famexp_clenshaw_d4(const int64_t *ip, const uint32_t *idx,
                        const double *dat, int n, int m,
                        const double *c, const double *d,
                        const double *b0, const double *b1,
                        double *y, double *w) {
    double *p0 = w, *p1 = w + n, *p2 = w + 2 * n;
    for (int i = 0; i < n; ++i) p0[i] = p1[i] = p2[i] = 0.0;
    for (int k = m; k >= 1; --k) {
        double ck = c[k], dk = d[k];
        int i = 0;
        for (; i + 1 < n; i += 2) {
            int64_t j0 = ip[i], e0 = ip[i+1], j1 = e0, e1 = ip[i+2];
            double s0 = 0.0, s1 = 0.0;
            while (j0 < e0 || j1 < e1) {
                if (j0 < e0) { s0 += dat[j0] * p1[idx[j0]]; ++j0; }
                if (j1 < e1) { s1 += dat[j1] * p1[idx[j1]]; ++j1; }
            }
            p0[i]   = s0 - 2.0 * p1[i]   - p2[i]   + ck * b0[i]   + dk * b1[i];
            p0[i+1] = s1 - 2.0 * p1[i+1] - p2[i+1] + ck * b0[i+1] + dk * b1[i+1];
        }
        for (; i < n; ++i) {
            double s = 0.0;
            for (int64_t jj = ip[i]; jj < ip[i + 1]; ++jj) s += dat[jj] * p1[idx[jj]];
            p0[i] = s - 2.0 * p1[i] - p2[i] + ck * b0[i] + dk * b1[i];
        }
        double *tmp = p2; p2 = p1; p1 = p0; p0 = tmp;
    }
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int64_t jj = ip[i]; jj < ip[i + 1]; ++jj) s += dat[jj] * p1[idx[jj]];
        y[i] = c[0] * b0[i] + d[0] * b1[i] + 0.5 * s - p1[i] - p2[i];
    }
}

EXPORT void famexp_clenshaw_d5(const int64_t *ip, const uint32_t *idx,
                        const double *dat, int n, int m,
                        const double *c, const double *d,
                        const double *b0, const double *b1,
                        double *y, double *w) {
    double *p0 = w, *p1 = w + n, *p2 = w + 2 * n;
    for (int i = 0; i < n; ++i) p0[i] = p1[i] = p2[i] = 0.0;
    for (int k = m; k >= 1; --k) {
        double ck = c[k], dk = d[k];
        int i = 0;
        for (; i + 3 < n; i += 4) {
            int64_t j0=ip[i], e0=ip[i+1], j1=e0, e1=ip[i+2];
            int64_t j2=e1, e2=ip[i+3], j3=e2, e3=ip[i+4];
            double s0=0,s1=0,s2=0,s3=0;
            while (j0<e0 || j1<e1 || j2<e2 || j3<e3) {
                if (j0<e0) { s0 += dat[j0]*p1[idx[j0]]; ++j0; }
                if (j1<e1) { s1 += dat[j1]*p1[idx[j1]]; ++j1; }
                if (j2<e2) { s2 += dat[j2]*p1[idx[j2]]; ++j2; }
                if (j3<e3) { s3 += dat[j3]*p1[idx[j3]]; ++j3; }
            }
            p0[i]   = s0 - 2.0*p1[i]   - p2[i]   + ck*b0[i]   + dk*b1[i];
            p0[i+1] = s1 - 2.0*p1[i+1] - p2[i+1] + ck*b0[i+1] + dk*b1[i+1];
            p0[i+2] = s2 - 2.0*p1[i+2] - p2[i+2] + ck*b0[i+2] + dk*b1[i+2];
            p0[i+3] = s3 - 2.0*p1[i+3] - p2[i+3] + ck*b0[i+3] + dk*b1[i+3];
        }
        for (; i < n; ++i) {
            double s = 0.0;
            for (int64_t jj = ip[i]; jj < ip[i + 1]; ++jj) s += dat[jj] * p1[idx[jj]];
            p0[i] = s - 2.0 * p1[i] - p2[i] + ck * b0[i] + dk * b1[i];
        }
        double *tmp = p2; p2 = p1; p1 = p0; p0 = tmp;
    }
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int64_t jj = ip[i]; jj < ip[i + 1]; ++jj) s += dat[jj] * p1[idx[jj]];
        y[i] = c[0] * b0[i] + d[0] * b1[i] + 0.5 * s - p1[i] - p2[i];
    }
}
