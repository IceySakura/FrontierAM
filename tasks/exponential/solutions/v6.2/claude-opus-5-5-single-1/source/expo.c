/* FAMEXP2 exponential propagation solver:
 *   y = exp(-tL) b0 + t*phi1(-tL) b1,   L symmetric PSD (CSR).
 * Method: Chebyshev expansion on [0, lam_hi] evaluated with the forward
 * three-term recurrence, exp and phi1 coefficients applied to b0 and b1,
 * adaptive truncation from coefficient tail sums and the running ||y||_inf.
 * lam_hi: Gershgorin bound, tightened by a short Lanczos run when the
 * expansion is long; a 2-norm growth monitor restarts with Gershgorin if the
 * Lanczos-based interval was too small.  See README.md. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <sys/mman.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static int verbose = 0;

static void die(const char *m) { fprintf(stderr, "fatal: %s\n", m); _exit(3); }

static void read_exact(void *buf, size_t len) {
    char *p = (char *)buf;
    while (len) {
        size_t chunk = len > ((size_t)1 << 30) ? ((size_t)1 << 30) : len;
        ssize_t r = read(0, p, chunk);
        if (r < 0) { if (errno == EINTR) continue; die("read"); }
        if (r == 0) die("unexpected eof");
        p += r; len -= (size_t)r;
    }
}

static void write_all(const void *buf, size_t len) {
    const char *p = (const char *)buf;
    while (len) {
        ssize_t w = write(1, p, len);
        if (w < 0) { if (errno == EINTR) continue; die("write"); }
        p += w; len -= (size_t)w;
    }
}

static double jnum(const char *js, const char *key) {
    const char *p = strstr(js, key);
    if (!p) die("json key missing");
    p += strlen(key);
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ':') p++;
    return strtod(p, NULL);
}

static void *xmalloc(size_t sz) {
    void *p = NULL;
    if (sz == 0) sz = 64;
    if (posix_memalign(&p, 64, sz)) die("oom");
    return p;
}

/* Bump arena, mapped with huge pages and pre-faulted before READY so that the
   timed session does not pay page faults for its main buffers. */
static char *arena; static size_t arena_sz, arena_off;
static void arena_init(size_t sz) {
    size_t HP = (size_t)2 << 20;
    void *p = mmap(NULL, sz + HP, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { arena = NULL; return; }
    char *a = (char *)(((uintptr_t)p + HP - 1) & ~(uintptr_t)(HP - 1));
#ifdef MADV_HUGEPAGE
    madvise(a, sz, MADV_HUGEPAGE);
#endif
    memset(a, 0, sz);
    arena = a; arena_sz = sz; arena_off = 0;
}
static void *amalloc(size_t sz) {
    sz = (sz + 63) & ~(size_t)63;
    if (sz == 0) sz = 64;
    if (arena && arena_off + sz <= arena_sz) { void *p = arena + arena_off; arena_off += sz; return p; }
    return xmalloc(sz);
}
static void *acalloc(size_t sz) { void *p = amalloc(sz); memset(p, 0, sz); return p; }
static void afree(void *p) {
    if (!p) return;
    if (arena && (char *)p >= arena && (char *)p < arena + arena_sz) return;
    free(p);
}

/* ---------------- matrix ---------------- */
static int n;
static long nnz;
static uint32_t *rp;     /* n+1 */
static uint32_t *ci;     /* nnz */
static double *va;       /* nnz */
static int nk;           /* vector length used by the kernels (n, or n rounded up to 4 for SELL) */
static int nrows_real;   /* = n */
static int use_sell = 0, nch = 0, use_avx = 1;
static uint32_t *scs, *sci;
static double *sva;
/* SELL-4 copy of the symmetrically permuted matrix P L P^T */
static int *perm;        /* perm[new] = old */

/* mean |row-col| over every 8th row in a given numbering (sampled: cheap) */
static double order_cost(const int *ip) {
    double s = 0; long cnt = 0;
    for (int i = 0; i < n; i += 8) {
        long ri = ip ? ip[i] : i;
        for (uint32_t k = rp[i]; k < rp[i + 1]; k++) {
            long cj = ip ? ip[ci[k]] : (long)ci[k];
            s += (double)labs(ri - cj);
        }
        cnt += rp[i + 1] - rp[i];
    }
    return cnt > 0 ? s / (double)cnt : 0;
}

/* Cuthill-McKee-like order: plain BFS from the lowest-numbered unvisited
   node of each component (no degree sorting; only locality matters here) */
static void cm_order(int *ord) {
    char *vis = acalloc(n);
    int pos = 0;
    for (int s0 = 0; s0 < n; s0++) {
        if (vis[s0]) continue;
        vis[s0] = 1; ord[pos++] = s0;
        int h = pos - 1;
        while (h < pos) {
            if (h + 16 < pos) __builtin_prefetch(&rp[ord[h + 16]]);
            if (h + 8 < pos) __builtin_prefetch(&ci[rp[ord[h + 8]]]);
            int u = ord[h++];
            for (uint32_t k = rp[u]; k < rp[u + 1]; k++) {
                int v = (int)ci[k];
                if (!vis[v]) { vis[v] = 1; ord[pos++] = v; }
            }
        }
    }
    afree(vis);
}

static double build_sell(int try_cm, double c0) {
    double tb0 = now_sec();
    int *ord = amalloc(sizeof(int) * n), *ip = amalloc(sizeof(int) * n);
    double c1 = c0;
    double tb1 = now_sec(), tb2 = tb1;
    if (try_cm && c0 > 1000.0) {
        cm_order(ord);
        tb2 = now_sec();
        for (int i = 0; i < n; i++) ip[ord[i]] = i;
        c1 = order_cost(ip);
    }
    double tb3 = now_sec();
    if (!(c1 < 0.5 * c0)) for (int i = 0; i < n; i++) ord[i] = i;
    /* sort rows by length (descending) inside windows of 32 to limit padding */
    enum { SIG = 32 };
    for (int w0 = 0; w0 < n; w0 += SIG) {
        int w1 = w0 + SIG < n ? w0 + SIG : n;
        for (int a = w0 + 1; a < w1; a++) {
            int v = ord[a]; uint32_t lv = rp[v + 1] - rp[v];
            int b = a - 1;
            while (b >= w0 && rp[ord[b] + 1] - rp[ord[b]] < lv) { ord[b + 1] = ord[b]; b--; }
            ord[b + 1] = v;
        }
    }
    for (int i = 0; i < n; i++) ip[ord[i]] = i;
    int n4 = (n + 3) & ~3;
    nch = n4 / 4;
    scs = amalloc(sizeof(uint32_t) * (nch + 1));
    scs[0] = 0;
    for (int c = 0; c < nch; c++) {
        uint32_t w = 0;
        for (int r = 0; r < 4; r++) {
            int row = 4 * c + r;
            if (row < n) { int o = ord[row]; uint32_t l = rp[o + 1] - rp[o]; if (l > w) w = l; }
        }
        scs[c + 1] = scs[c] + 4 * w;
    }
    size_t tot = scs[nch];
    sci = amalloc(sizeof(uint32_t) * (tot + 4));
    sva = amalloc(sizeof(double) * (tot + 4));
    for (int c = 0; c < nch; c++) {
        uint32_t w = (scs[c + 1] - scs[c]) / 4;
        if (4 * c + 12 < n) {
            for (int r = 0; r < 4; r++) {
                int o2 = ord[4 * c + 8 + r];
                __builtin_prefetch(&ci[rp[o2]]); __builtin_prefetch(&va[rp[o2]]);
            }
            for (int r = 0; r < 4 && 4 * c + 16 + r < n; r++) __builtin_prefetch(&rp[ord[4 * c + 16 + r]]);
        }
        for (int r = 0; r < 4; r++) {
            int row = 4 * c + r;
            uint32_t l = 0, b0 = 0;
            if (row < n) { int o = ord[row]; b0 = rp[o]; l = rp[o + 1] - rp[o]; }
            for (uint32_t j = 0; j < w; j++) {
                size_t k = scs[c] + 4 * (size_t)j + r;
                if (j < l) { sci[k] = (uint32_t)ip[ci[b0 + j]]; sva[k] = va[b0 + j]; }
                else { sci[k] = (uint32_t)row; sva[k] = 0.0; }
            }
        }
    }
    perm = ord;
    afree(ip);
    use_sell = 1;
    nk = n4;
    if (verbose) fprintf(stderr, "sell: order cost orig %.1f cm %.1f (%s), padded %zu / %ld; times cost0 %.4f cm %.4f cost1 %.4f rest %.4f\n", c0, c1, c1 < 0.5 * c0 ? "cm" : "orig", tot, nnz, tb1 - tb0, tb2 - tb1, tb3 - tb2, now_sec() - tb3);
    return c1 < 0.5 * c0 ? c1 : c0;
}

/* y = L x in kernel numbering (CSR, or SELL-4 in the permuted numbering) */
static void spmv(const double *restrict x, double *restrict y) {
    if (use_sell) {
        for (int ch = 0; ch < nch; ch++) {
            double a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            for (uint32_t k = scs[ch]; k < scs[ch + 1]; k += 4) {
                a0 += sva[k] * x[sci[k]]; a1 += sva[k + 1] * x[sci[k + 1]];
                a2 += sva[k + 2] * x[sci[k + 2]]; a3 += sva[k + 3] * x[sci[k + 3]];
            }
            y[4 * ch] = a0; y[4 * ch + 1] = a1; y[4 * ch + 2] = a2; y[4 * ch + 3] = a3;
        }
        return;
    }
    for (int i = 0; i < n; i++) {
        double s = 0;
        for (uint32_t k = rp[i]; k < rp[i + 1]; k++) s += va[k] * x[ci[k]];
        y[i] = s;
    }
}

/* ---------------- spectral bound ---------------- */
/* largest eigenvalue of symmetric tridiagonal (al[0..k-1], be[0..k-2]) by bisection */
static double tri_maxeig(const double *al, const double *be, int k) {
    double lo = 1e300, hi = -1e300;
    for (int i = 0; i < k; i++) {
        double rad = (i > 0 ? fabs(be[i - 1]) : 0) + (i < k - 1 ? fabs(be[i]) : 0);
        if (al[i] - rad < lo) lo = al[i] - rad;
        if (al[i] + rad > hi) hi = al[i] + rad;
    }
    for (int it = 0; it < 200; it++) {
        double mid = 0.5 * (lo + hi);
        if (mid <= lo || mid >= hi) break;
        /* count eigenvalues > mid : number of negative pivots of (mid I - T) ... use Sturm on T - mid */
        int cnt = 0; double d = 1.0;
        for (int i = 0; i < k; i++) {
            double b2 = i > 0 ? be[i - 1] * be[i - 1] : 0;
            d = (al[i] - mid) - (i > 0 ? b2 / d : 0);
            if (d == 0) d = -1e-300;
            if (d > 0) cnt++;   /* pivots >0 count eigenvalues > mid */
        }
        if (cnt > 0) lo = mid; else hi = mid;
    }
    return hi;
}

/* last component of normalized eigenvector for eigenvalue theta (largest) by inverse iteration */
static double tri_lastcomp(const double *al, const double *be, int k, double theta) {
    double *x = malloc(sizeof(double) * k), *dd = malloc(sizeof(double) * k), *rhs = malloc(sizeof(double) * k);
    double shift = theta + 1e-9 * (fabs(theta) + 1e-300);
    for (int i = 0; i < k; i++) x[i] = 1.0;
    for (int it = 0; it < 3; it++) {
        /* solve (T - shift I) z = x ; T - shift I is negative definite -> Thomas */
        for (int i = 0; i < k; i++) rhs[i] = x[i];
        dd[0] = al[0] - shift;
        for (int i = 1; i < k; i++) {
            double m = be[i - 1] / dd[i - 1];
            dd[i] = (al[i] - shift) - m * be[i - 1];
            rhs[i] -= m * rhs[i - 1];
        }
        x[k - 1] = rhs[k - 1] / dd[k - 1];
        for (int i = k - 2; i >= 0; i--) x[i] = (rhs[i] - be[i] * x[i + 1]) / dd[i];
        double nr = 0; for (int i = 0; i < k; i++) nr += x[i] * x[i];
        nr = sqrt(nr); if (!(nr > 0) || !isfinite(nr)) { free(x); free(dd); free(rhs); return 1.0; }
        for (int i = 0; i < k; i++) x[i] /= nr;
    }
    double r = fabs(x[k - 1]);
    free(x); free(dd); free(rhs);
    return r;
}

static double lanczos_lmax(double G, int *steps_used) {
    int n = nk;   /* kernel numbering; padded rows are empty and stay zero */
    double *v = amalloc(sizeof(double) * n), *vo = amalloc(sizeof(double) * n), *w = amalloc(sizeof(double) * n);
    uint64_t s = 0x9E3779B97F4A7C15ULL;
    double nr = 0;
    for (int i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        double u = ((double)(s >> 11) * (1.0 / 9007199254740992.0)) - 0.5;
        if (use_sell && i >= nrows_real) u = 0;
        v[i] = u; nr += u * u;
    }
    nr = sqrt(nr);
    for (int i = 0; i < n; i++) { v[i] /= nr; vo[i] = 0; }
    enum { KMAX = 80 };
    double al[KMAX], be[KMAX];
    double beta = 0, theta = 0, thprev = -1, res = 0;
    int k;
    for (k = 0; k < KMAX; k++) {
        spmv(v, w);
        double a = 0;
        for (int i = 0; i < n; i++) { double wi = w[i] - beta * vo[i]; w[i] = wi; a += wi * v[i]; }
        double b2 = 0;
        for (int i = 0; i < n; i++) { double wi = w[i] - a * v[i]; w[i] = wi; b2 += wi * wi; }
        al[k] = a; beta = sqrt(b2 > 0 ? b2 : 0); be[k] = beta;
        int kk = k + 1;
        if (beta <= 1e-14 * G) {  /* invariant subspace: exact */
            theta = tri_maxeig(al, be, kk); res = 0; k = kk; goto done;
        }
        double inv = 1.0 / beta;
        for (int i = 0; i < n; i++) { double t = v[i]; v[i] = w[i] * inv; vo[i] = t; }
        if (kk >= 12 && kk % 3 == 0) {
            theta = tri_maxeig(al, be, kk);
            res = beta * tri_lastcomp(al, be, kk, theta);
            if (thprev > 0 && fabs(theta - thprev) <= 2e-4 * theta && res <= 0.01 * theta) { k = kk; goto done; }
            thprev = theta;
        }
    }
    theta = tri_maxeig(al, be, KMAX);
    res = beta * tri_lastcomp(al, be, KMAX, theta);
    k = KMAX;
done:
    afree(v); afree(vo); afree(w);
    *steps_used = k;
    double lh = theta + fmax(0.02 * theta, 1.5 * res);
    if (verbose) fprintf(stderr, "lanczos k=%d theta=%.10g res=%.3g G=%.10g -> lh=%.10g\n", k, theta, res, G, lh);
    if (!(lh > 0) || !isfinite(lh) || lh > G) lh = G;
    return lh;
}

/* ---------------- coefficients ---------------- */
typedef struct {
    double lh;        /* interval [0, lh] */
    int M;            /* number of coefficients available (0..M-1) */
    double *c, *d;    /* exp and phi1 coefficients */
    double *tc, *td;  /* tails: tc[k] = sum_{j>k} |c_j| */
} Coef;

static void coef_free(Coef *C) { free(C->c); free(C->d); free(C->tc); free(C->td); }

static void coef_build(Coef *C, double t, double lh) {
    double beta = 0.5 * t * lh;
    /* number of terms: ive_k ~ exp(-k^2/(2 beta)) ; need below ~1e-22 */
    int Mneed = (int)(sqrt(2.0 * beta * 55.0) + 3.0 * pow(beta, 1.0 / 3.0) + 40.0);
    int N = Mneed + (int)(0.25 * Mneed) + 60;
    double *r = malloc(sizeof(double) * (N + 2));
    r[N + 1] = 0;
    for (int k = N; k >= 1; k--) r[k] = 1.0 / (2.0 * k / beta + r[k + 1]);
    double *iv = malloc(sizeof(double) * (N + 1));
    iv[0] = 1.0;
    double sum = 0;
    for (int k = 1; k <= N; k++) { iv[k] = iv[k - 1] * r[k]; sum += iv[k]; }
    double i0 = 1.0 / (1.0 + 2.0 * sum);
    int M = Mneed;
    C->M = M; C->lh = lh;
    C->c = malloc(sizeof(double) * (M + 2)); C->d = malloc(sizeof(double) * (M + 2));
    C->tc = malloc(sizeof(double) * (M + 2)); C->td = malloc(sizeof(double) * (M + 2));
    double *e = malloc(sizeof(double) * (N + 3));
    for (int k = 0; k <= N; k++) {
        double ck = (k == 0 ? 1.0 : 2.0) * iv[k] * i0 * ((k & 1) ? -1.0 : 1.0);
        if (k < M + 2) C->c[k] = ck;
        e[k] = -(2.0 / lh) * ck;
    }
    e[0] += 2.0 / lh;
    e[N + 1] = 0; e[N + 2] = 0;
    /* backward recurrence for phi1 coefficients:  (x+1) g = e */
    double *dd = calloc(N + 3, sizeof(double));
    for (int k = N; k >= 2; k--) dd[k - 1] = 2.0 * (e[k] - dd[k]) - dd[k + 1];
    dd[0] = e[1] - dd[1] - 0.5 * dd[2];
    for (int k = 0; k < M + 2; k++) C->d[k] = dd[k];
    /* tails */
    double sc = 0, sd = 0;
    for (int k = N; k >= M + 2; k--) { sc += fabs(k < M + 2 ? 0 : ((k == 0 ? 1.0 : 2.0) * iv[k] * i0)); sd += fabs(dd[k]); }
    for (int k = M + 1; k >= 0; k--) {
        C->tc[k] = sc; C->td[k] = sd;
        sc += fabs(C->c[k]); sd += fabs(C->d[k]);
    }
    free(r); free(iv); free(e); free(dd);
}

/* ---------------- main solve ---------------- */
static double tval, rtol, atol_;
static double G;

#if defined(__AVX2__) && defined(__FMA__) && !defined(EXPO_NO_AVX)
#include <immintrin.h>
#define HAVE_AVX_KERNEL 1
/* AVX2 version of step1s_d: one 4-wide FMA per 4 chunk entries (two
   independent accumulators), vectorized row updates. */
static void step1s_avx(const double *restrict x, double *restrict xm, double *restrict y,
                       double p, double q, double r, double c,
                       double *nrm2, double *vmax, double *ymax) {
    const __m256d P = _mm256_set1_pd(p), Q = _mm256_set1_pd(q), R = _mm256_set1_pd(r), Cc = _mm256_set1_pd(c);
    const __m256d SIGN = _mm256_set1_pd(-0.0);
    __m256d S2 = _mm256_setzero_pd(), VM = _mm256_setzero_pd(), YM = _mm256_setzero_pd();
    const uint32_t *restrict CI = sci; const double *restrict VA = sva;
    for (int ch = 0; ch < nch; ch++) {
        uint32_t k = scs[ch], k1 = scs[ch + 1];
        __m256d a0 = _mm256_setzero_pd(), a1 = _mm256_setzero_pd();
        for (; k + 8 <= k1; k += 8) {
            __m256d x0 = _mm256_set_pd(x[CI[k + 3]], x[CI[k + 2]], x[CI[k + 1]], x[CI[k]]);
            __m256d x1 = _mm256_set_pd(x[CI[k + 7]], x[CI[k + 6]], x[CI[k + 5]], x[CI[k + 4]]);
            a0 = _mm256_fmadd_pd(_mm256_loadu_pd(VA + k), x0, a0);
            a1 = _mm256_fmadd_pd(_mm256_loadu_pd(VA + k + 4), x1, a1);
        }
        if (k < k1) {
            __m256d x0 = _mm256_set_pd(x[CI[k + 3]], x[CI[k + 2]], x[CI[k + 1]], x[CI[k]]);
            a0 = _mm256_fmadd_pd(_mm256_loadu_pd(VA + k), x0, a0);
        }
        __m256d acc = _mm256_add_pd(a0, a1);
        size_t i0 = 4 * (size_t)ch;
        __m256d xi = _mm256_loadu_pd(x + i0), xmi = _mm256_loadu_pd(xm + i0), yi = _mm256_loadu_pd(y + i0);
        __m256d vn = _mm256_fmadd_pd(P, acc, _mm256_fmadd_pd(Q, xi, _mm256_mul_pd(R, xmi)));
        _mm256_storeu_pd(xm + i0, vn);
        yi = _mm256_fmadd_pd(Cc, vn, yi);
        _mm256_storeu_pd(y + i0, yi);
        S2 = _mm256_fmadd_pd(vn, vn, S2);
        VM = _mm256_max_pd(VM, _mm256_andnot_pd(SIGN, vn));
        YM = _mm256_max_pd(YM, _mm256_andnot_pd(SIGN, yi));
    }
    double t[4];
    _mm256_storeu_pd(t, S2); *nrm2 = (t[0] + t[1]) + (t[2] + t[3]);
    _mm256_storeu_pd(t, VM); *vmax = fmax(fmax(t[0], t[1]), fmax(t[2], t[3]));
    _mm256_storeu_pd(t, YM); *ymax = fmax(fmax(t[0], t[1]), fmax(t[2], t[3]));
}

/* AVX2 version of step2s_d (x interleaved as (a,b) pairs).  Accumulators hold
   rows (0,2) and (1,3) of a chunk as (a,b,a,b) so that the value vector only
   needs in-lane unpacking. */
static void step2s_avx(const double *restrict x, double *restrict xm, double *restrict y,
                       double p, double q, double r, double c0, double c1,
                       double *nrm2a, double *nrm2b, double *vmaxa, double *vmaxb, double *ymax) {
    const __m256d P = _mm256_set1_pd(p), Q = _mm256_set1_pd(q), R = _mm256_set1_pd(r);
    const __m256d CC = _mm256_set_pd(c1, c0, c1, c0);
    const __m256d SIGN = _mm256_set1_pd(-0.0);
    __m256d SQ = _mm256_setzero_pd(), VM = _mm256_setzero_pd(), YM = _mm256_setzero_pd();
    const uint32_t *restrict CI = sci; const double *restrict VA = sva;
    for (int ch = 0; ch < nch; ch++) {
        uint32_t k = scs[ch], k1 = scs[ch + 1];
        __m256d e02 = _mm256_setzero_pd(), e13 = _mm256_setzero_pd();
        __m256d f02 = _mm256_setzero_pd(), f13 = _mm256_setzero_pd();
        for (; k + 8 <= k1; k += 8) {
            __m256d V = _mm256_loadu_pd(VA + k), W = _mm256_loadu_pd(VA + k + 4);
            __m256d X02 = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(x + 2 * (size_t)CI[k])), _mm_loadu_pd(x + 2 * (size_t)CI[k + 2]), 1);
            __m256d X13 = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(x + 2 * (size_t)CI[k + 1])), _mm_loadu_pd(x + 2 * (size_t)CI[k + 3]), 1);
            __m256d Y02 = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(x + 2 * (size_t)CI[k + 4])), _mm_loadu_pd(x + 2 * (size_t)CI[k + 6]), 1);
            __m256d Y13 = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(x + 2 * (size_t)CI[k + 5])), _mm_loadu_pd(x + 2 * (size_t)CI[k + 7]), 1);
            e02 = _mm256_fmadd_pd(_mm256_unpacklo_pd(V, V), X02, e02);
            e13 = _mm256_fmadd_pd(_mm256_unpackhi_pd(V, V), X13, e13);
            f02 = _mm256_fmadd_pd(_mm256_unpacklo_pd(W, W), Y02, f02);
            f13 = _mm256_fmadd_pd(_mm256_unpackhi_pd(W, W), Y13, f13);
        }
        if (k < k1) {
            __m256d V = _mm256_loadu_pd(VA + k);
            __m256d X02 = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(x + 2 * (size_t)CI[k])), _mm_loadu_pd(x + 2 * (size_t)CI[k + 2]), 1);
            __m256d X13 = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(x + 2 * (size_t)CI[k + 1])), _mm_loadu_pd(x + 2 * (size_t)CI[k + 3]), 1);
            e02 = _mm256_fmadd_pd(_mm256_unpacklo_pd(V, V), X02, e02);
            e13 = _mm256_fmadd_pd(_mm256_unpackhi_pd(V, V), X13, e13);
        }
        e02 = _mm256_add_pd(e02, f02); e13 = _mm256_add_pd(e13, f13);
        __m256d SA = _mm256_permute2f128_pd(e02, e13, 0x20);   /* rows 0,1 as (a,b,a,b) */
        __m256d SB = _mm256_permute2f128_pd(e02, e13, 0x31);   /* rows 2,3 */
        size_t i0 = 4 * (size_t)ch, j0 = 2 * i0;
        __m256d XA = _mm256_loadu_pd(x + j0), XB = _mm256_loadu_pd(x + j0 + 4);
        __m256d MA = _mm256_loadu_pd(xm + j0), MB = _mm256_loadu_pd(xm + j0 + 4);
        __m256d VA4 = _mm256_fmadd_pd(P, SA, _mm256_fmadd_pd(Q, XA, _mm256_mul_pd(R, MA)));
        __m256d VB4 = _mm256_fmadd_pd(P, SB, _mm256_fmadd_pd(Q, XB, _mm256_mul_pd(R, MB)));
        _mm256_storeu_pd(xm + j0, VA4); _mm256_storeu_pd(xm + j0 + 4, VB4);
        __m256d h = _mm256_hadd_pd(_mm256_mul_pd(CC, VA4), _mm256_mul_pd(CC, VB4));  /* rows 0,2,1,3 */
        __m256d yi = _mm256_add_pd(_mm256_loadu_pd(y + i0), _mm256_permute4x64_pd(h, 0xD8));
        _mm256_storeu_pd(y + i0, yi);
        SQ = _mm256_fmadd_pd(VA4, VA4, _mm256_fmadd_pd(VB4, VB4, SQ));
        VM = _mm256_max_pd(VM, _mm256_max_pd(_mm256_andnot_pd(SIGN, VA4), _mm256_andnot_pd(SIGN, VB4)));
        YM = _mm256_max_pd(YM, _mm256_andnot_pd(SIGN, yi));
    }
    double t[4];
    _mm256_storeu_pd(t, SQ); *nrm2a = t[0] + t[2]; *nrm2b = t[1] + t[3];
    _mm256_storeu_pd(t, VM); *vmaxa = fmax(t[0], t[2]); *vmaxb = fmax(t[1], t[3]);
    _mm256_storeu_pd(t, YM); *ymax = fmax(fmax(t[0], t[1]), fmax(t[2], t[3]));
}
#endif

#define REAL double
#define RABS(x) fabs(x)
#define SUF(x) x##_d
#ifdef HAVE_AVX_KERNEL
#define PICK1 (use_sell ? (use_avx ? step1s_avx : step1s_d) : step1_d)
#else
#define PICK1 (use_sell ? step1s_d : step1_d)
#endif
#ifdef HAVE_AVX_KERNEL
#define PICK2 (use_sell ? (use_avx ? step2s_avx : step2s_d) : step2_d)
#else
#define PICK2 (use_sell ? step2s_d : step2_d)
#endif
#include "cheb_body.h"
#undef PICK1
#undef PICK2
#undef REAL
#undef RABS
#undef SUF
#define REAL long double
#define RABS(x) fabsl(x)
#define SUF(x) x##_l
#define PICK1 (use_sell ? step1s_l : step1_l)
#define PICK2 (use_sell ? step2s_l : step2_l)
#include "cheb_body.h"
#undef PICK1
#undef PICK2
#undef REAL
#undef RABS
#undef SUF

/* estimated degree for the generic tolerance (used to decide on Lanczos) */
static int est_degree(const Coef *C, double tol) {
    for (int k = 1; k < C->M; k++) if (C->tc[k] <= tol) return k;
    return C->M;
}

/* --check: exercise every numerical kernel (CSR, SELL-4 scalar and AVX2,
   double and long double) on a small weighted path graph, without touching
   stdin; detects an unusable binary (e.g. unsupported instructions). */
static int self_check(void) {
    enum { CN = 11 };
    static uint32_t crp[CN + 1], cci[3 * CN];
    static double cva[3 * CN];
    uint32_t k = 0;
    for (int i = 0; i < CN; i++) {
        crp[i] = k;
        double wl = i > 0 ? 1.0 + 0.1 * i : 0, wr = i < CN - 1 ? 1.0 + 0.1 * (i + 1) : 0;
        if (i > 0) { cci[k] = i - 1; cva[k++] = -wl; }
        cci[k] = i; cva[k++] = wl + wr + 0.05 * (i % 3);
        if (i < CN - 1) { cci[k] = i + 1; cva[k++] = -wr; }
    }
    crp[CN] = k;
    n = CN; nnz = k; rp = crp; ci = cci; va = cva; nk = n; nrows_real = n;
    tval = 3.0; rtol = 1e-10; atol_ = 1e-13; G = 0;
    for (int i = 0; i < n; i++) { double s = 0; for (uint32_t j = rp[i]; j < rp[i + 1]; j++) s += fabs(va[j]); if (s > G) G = s; }
    Coef C; coef_build(&C, tval, G);
    double b0[CN + 4], b1[CN + 4], ref[2][CN + 4], y[CN + 4], kb[2 * (CN + 4)], ky[CN + 4];
    double w1[2 * (CN + 4)], w2[2 * (CN + 4)];
    long double l1[2 * (CN + 4)], l2[2 * (CN + 4)], ly[CN + 4];
    for (int i = 0; i < CN; i++) { b0[i] = (i % 2 ? 0.5 : -1.0) / (1 + i); b1[i] = 0.25 * (i - 5) / 5.0; }
    int deg; double ym;
    /* references: CSR double, both-vector and one-vector paths */
    cheb_solve_d(&C, b0, b1, ref[0], w1, w2, ref[0], 1, &deg, &ym);
    double zero[CN + 4] = {0};
    cheb_solve_d(&C, b0, zero, ref[1], w1, w2, ref[1], 1, &deg, &ym);
    cheb_solve_l(&C, b0, b1, y, l1, l2, ly, 0, &deg, &ym);
    double z = 0, z0 = 0;
    for (int i = 0; i < CN; i++) {
        if (!isfinite(ref[0][i]) || fabs(ref[0][i] - y[i]) > 1e-9) return 1;
        z += ref[0][i]; z0 += b0[i];
    }
    (void)z; (void)z0;
    /* SELL-4 (scalar and, if compiled, AVX2) must agree with CSR */
    build_sell(0, 0.0);
    for (int pass = 0; pass < 2; pass++) {
        use_avx = pass;
        for (int v = 0; v < 2; v++) {
            for (int i = 0; i < nk; i++) {
                int o = i < n ? perm[i] : -1;
                kb[i] = o >= 0 ? b0[o] : 0; kb[nk + i] = (o >= 0 && v == 0) ? b1[o] : 0;
            }
            cheb_solve_d(&C, kb, kb + nk, ky, w1, w2, ky, 1, &deg, &ym);
            for (int i = 0; i < n; i++) if (!isfinite(ky[i]) || fabs(ky[i] - ref[v][perm[i]]) > 1e-12) return 1;
            cheb_solve_l(&C, kb, kb + nk, ky, l1, l2, ly, 0, &deg, &ym);
            for (int i = 0; i < n; i++) if (!isfinite(ky[i]) || fabs(ky[i] - ref[v][perm[i]]) > 1e-9) return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--check") == 0) return self_check();
    if (getenv("EXPO_VERBOSE")) verbose = 1;
    if (getenv("EXPO_NO_AVXK")) use_avx = 0;
    { int ps = getenv("EXPO_PIPE_KB") ? atoi(getenv("EXPO_PIPE_KB")) * 1024 : 1 << 20; if (ps > 0) fcntl(0, F_SETPIPE_SZ, ps); }
    fcntl(1, F_SETPIPE_SZ, 1 << 20);
    /* input-independent preparation: pre-faulted arena for the largest legal
       session (n <= 120000, nnz <= 1e6), including SELL padding headroom */
    arena_init((size_t)(getenv("EXPO_ARENA_MB") ? atoi(getenv("EXPO_ARENA_MB")) : 64) << 20);
    write_all("READY\n", 6);

    char magic[8];
    read_exact(magic, 8);
    double t_start = now_sec();
    if (memcmp(magic, "FAMEXP2\n", 8) != 0) die("bad magic");
    uint32_t jl;
    read_exact(&jl, 4);
    if (jl == 0 || jl > 1 << 20) die("bad json len");
    char *js = malloc(jl + 1);
    read_exact(js, jl); js[jl] = 0;
    n = (int)jnum(js, "\"n\"");
    nnz = (long)jnum(js, "\"nnz\"");
    int requests = (int)jnum(js, "\"requests\"");
    tval = jnum(js, "\"t\"");
    rtol = jnum(js, "\"rtol\"");
    atol_ = jnum(js, "\"atol\"");
    if (n < 1 || nnz < 0 || requests < 1) die("bad header");
    nrows_real = n;

    uint64_t *ip64 = amalloc(sizeof(uint64_t) * (n + 1));
    read_exact(ip64, sizeof(uint64_t) * (n + 1));
    ci = amalloc(sizeof(uint32_t) * (nnz + 1));
    read_exact(ci, sizeof(uint32_t) * nnz);
    va = amalloc(sizeof(double) * (nnz + 1));
    rp = amalloc(sizeof(uint32_t) * (n + 1));
    /* read the values while computing Gershgorin row sums of complete rows */
    G = 0;
    {
        size_t total = sizeof(double) * (size_t)nnz, got = 0;
        int row = 0;
        for (;;) {
            size_t ent = got / sizeof(double);
            while (row < n && ip64[row + 1] <= ent) {
                rp[row] = (uint32_t)ip64[row];
                double s = 0;
                for (uint64_t k = ip64[row]; k < ip64[row + 1]; k++) s += fabs(va[k]);
                if (s > G) G = s;
                row++;
            }
            if (got >= total) break;
            ssize_t r = read(0, (char *)va + got, total - got);
            if (r < 0) { if (errno == EINTR) continue; die("read"); }
            if (r == 0) die("unexpected eof");
            got += (size_t)r;
        }
        if (row != n) die("bad indptr");
    }
    rp[n] = (uint32_t)ip64[n];
    afree(ip64);
    nk = n;
    if (verbose) fprintf(stderr, "time read+gersh %.4f\n", now_sec() - t_start);

    double *b = amalloc(sizeof(double) * 2 * (size_t)n);
    double *y = amalloc(sizeof(double) * (size_t)n);
    /* t*G < 1e-30: exp(-tL) = I and t*phi1(-tL) = t*I to far below any tolerance */
    int trivial = !(tval > 0) || !(G > 0) || tval * G < 1e-30;
    if (trivial) {
        for (int r = 0; r < requests; r++) {
            read_exact(b, sizeof(double) * 2 * (size_t)n);
            for (int i = 0; i < n; i++) y[i] = b[i] + tval * b[n + i];
            write_all(y, sizeof(double) * (size_t)n);
        }
        _exit(0);
    }
    long double *l1 = NULL, *l2 = NULL, *ly = NULL;
    double *ky2 = NULL;
    int force_ld = getenv("EXPO_FORCE_LD") != NULL, never_ld = getenv("EXPO_NEVER_LD") != NULL;

    Coef CG, CL; int haveL = 0;
    double Gs = G * (1.0 + 1e-12);
    coef_build(&CG, tval, Gs);
    int mG = est_degree(&CG, 1e-9 * rtol + atol_);
    /* SELL-4 + reordering pays off once enough recurrence steps are expected */
    long steps_total = (long)mG * requests;
    /* SELL-4 pays off sooner for local orderings (cheap, cache-friendly gathers);
       the AVX2 kernels only win on local orderings, so random graphs keep the
       scalar 4-row kernel. */
    if (n >= 64 && !getenv("EXPO_NO_SELL")) {
        double c0 = order_cost(NULL);
        int local0 = c0 < 2000.0;
        int try_cm = steps_total >= 150 && !local0;
        if (try_cm || (local0 && steps_total >= 8) || (!local0 && steps_total >= 30)) {
            double cf = build_sell(try_cm, c0);
            use_avx = use_avx && cf < 2000.0;
            if (verbose) fprintf(stderr, "time after sell build %.4f (avx %d)\n", now_sec() - t_start, use_avx);
        }
    }
    /* interval: Gershgorin, optionally Lanczos-tightened */
    int lanczos_thresh = requests > 1 ? 25 : 90;
    if (mG > lanczos_thresh && n > 1) {
        int ks;
        double lh = lanczos_lmax(G, &ks);
        if (getenv("EXPO_LH_SCALE")) lh *= atof(getenv("EXPO_LH_SCALE"));   /* testing only */
        if (lh < 0.97 * Gs) { coef_build(&CL, tval, lh); haveL = 1; }
        if (verbose) fprintf(stderr, "time after lanczos %.4f\n", now_sec() - t_start);
    }
    if (verbose) fprintf(stderr, "n=%d nnz=%ld t=%g rtol=%g G=%g mG=%d haveL=%d\n", n, nnz, tval, rtol, G, mG, haveL);
    double *w1 = amalloc(sizeof(double) * 2 * (size_t)nk), *w2 = amalloc(sizeof(double) * 2 * (size_t)nk);
    double *kb = b, *ky = y;
    if (use_sell) {
        kb = amalloc(sizeof(double) * 2 * (size_t)nk);
        ky = amalloc(sizeof(double) * (size_t)nk);
    }
    for (int r = 0; r < requests; r++) {
        read_exact(b, sizeof(double) * 2 * (size_t)n);
        double treq = now_sec();
        const double *b0 = b, *b1 = b + n;
        if (use_sell) {
            for (int i = 0; i < n; i++) { int o = perm[i]; kb[i] = b0[o]; kb[nk + i] = b1[o]; }
            for (int i = n; i < nk; i++) { kb[i] = 0; kb[nk + i] = 0; }
        }
        const double *kb0 = kb, *kb1 = use_sell ? kb + nk : b1;
        int deg = 0, bad = 1;
        double ym = 0;
        const Coef *C = &CG;
        if (haveL) {
            C = &CL;
            bad = cheb_solve_d(C, kb0, kb1, ky, w1, w2, ky, 1, &deg, &ym);
            if (bad) {
                if (verbose) fprintf(stderr, "growth detected at k=%d; fallback to Gershgorin\n", deg);
                haveL = 0; C = &CG;
            }
        }
        if (bad) cheb_solve_d(C, kb0, kb1, ky, w1, w2, ky, 0, &deg, &ym);
        /* a-posteriori rounding-risk estimate: errors injected at each step are
           amplified by at most ~max|f| (=1) for b0 and max|g| (=t) for b1. */
        double m0 = 0, m1 = 0;
        for (int i = 0; i < n; i++) {
            double a0 = fabs(b0[i]), a1 = fabs(b1[i]);
            m0 = a0 > m0 ? a0 : m0; m1 = a1 > m1 ? a1 : m1;
        }
        double risk = 1.1e-16 * (deg + 10) * (m0 + tval * m1) * fmax(1.0, G / C->lh);
        double tol = atol_ + rtol * ym;
        /* time guard: an extended-precision redo costs about 2.5x (one vector)
           or 4x (two vectors) of the double solve; keep the session budget. */
        double tnow = now_sec();
        double tsolve = tnow - treq;
        double ld_est = tsolve * ((m0 > 0 && m1 > 0) ? 4.5 : 3.0);
        double budget = requests > 1 ? 195.0 : 48.0;
        int remaining = requests - 1 - r;
        int ld_ok = (tnow - t_start) + ld_est + remaining * tsolve * 1.3 < budget;
        int need = !never_ld && (force_ld || risk > 0.25 * tol);
        if (need && !force_ld && (tnow - t_start) + 1.2 * tsolve + remaining * tsolve * 1.3 < budget) {
            /* Twin check: same polynomial and degree, inputs scaled by pi/4 so
               that every rounding differs.  Truncation is identical, so the
               difference measures rounding; accept the average if small. */
            if (!ky2) ky2 = amalloc(sizeof(double) * (size_t)nk);
            int deg2 = 0; double ym2 = 0;
            cheb_solve_x_d(C, kb0, kb1, 0.78539816339744828, deg, ky2, w1, w2, ky2, 0, &deg2, &ym2);
            double dmax = 0;
            for (int i = 0; i < nk; i++) { double dd = fabs(ky[i] - ky2[i]); dmax = dd > dmax ? dd : dmax; }
            if (verbose) fprintf(stderr, "req %d: twin check diff %.3g tol %.3g (risk %.3g) -> %s\n", r, dmax, tol, risk, dmax <= 0.1 * tol ? "accept" : "extended");
            if (dmax <= 0.1 * tol && deg2 == deg) {
                for (int i = 0; i < nk; i++) ky[i] = 0.5 * (ky[i] + ky2[i]);
                need = 0;
            }
            tnow = now_sec();
            ld_ok = (tnow - t_start) + ld_est + remaining * tsolve * 1.3 < budget;
        }
        if (need && !ld_ok && verbose)
            fprintf(stderr, "req %d: extended precision redo skipped by time guard\n", r);
        if (need && ld_ok) {
            if (!l1) {
                l1 = amalloc(sizeof(long double) * 2 * (size_t)nk);
                l2 = amalloc(sizeof(long double) * 2 * (size_t)nk);
                ly = amalloc(sizeof(long double) * (size_t)nk);
            }
            int deg2 = 0; double ym2 = 0;
            cheb_solve_l(C, kb0, kb1, ky, l1, l2, ly, 0, &deg2, &ym2);
            if (verbose) fprintf(stderr, "req %d: extended precision redo (risk %.3g tol %.3g) degree %d -> %d\n", r, risk, tol, deg, deg2);
            deg = deg2;
        }
        if (use_sell) for (int i = 0; i < n; i++) y[perm[i]] = ky[i];
        if (verbose) fprintf(stderr, "req %d degree %d ymax %.3g risk/tol %.3g  t_req %.4f (since start %.4f)\n", r, deg, ym, risk / tol, now_sec() - treq, now_sec() - t_start);
        write_all(y, sizeof(double) * (size_t)n);
    }
    _exit(0);
}
