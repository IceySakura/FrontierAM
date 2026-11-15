/* famexp2.c -- fast exponential propagation for FAMEXP2 sessions.
 *
 *   y_j = exp(-t L) b0_j + t*phi1(-t L) b1_j ,   phi1(z) = (e^z - 1)/z
 *
 * L is a real symmetric positive semidefinite sparse matrix given in full CSR.
 *
 * Method: Chebyshev expansion of exp(z) and phi1(z) on an interval [a,b] that
 * contains the spectrum of A = -t*L.  Both right hand sides of a request are
 * propagated simultaneously as the two columns of one block, so a Chebyshev
 * step costs a single sweep over the matrix; the three term recurrence, the
 * series accumulation and the divergence guard are fused into that sweep.
 *
 * The interval is t*||L||_inf (rigorous) or, when it pays off, a tighter value
 * obtained from a short Lanczos run; the latter is verified at run time by
 * monitoring ||T_k(Ahat) b||, which can only exceed ||b|| if the interval fails
 * to contain the spectrum.  On detection the request is redone with the
 * rigorous interval.
 *
 * Dependencies: POSIX, libm.  stdout carries protocol bytes only.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define HAVE_SSE2 1
#ifdef __FMA__
#define FMADD(a, b, c) _mm_fmadd_pd((a), (b), (c))
#else
#define FMADD(a, b, c) _mm_add_pd((c), _mm_mul_pd((a), (b)))
#endif
#endif

typedef long double ld;

static int g_verbose = 0;
static double g_t0 = 0.0;
static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
#define TRACE(tag) do { if (g_verbose) { double _n = now_s(); \
    fprintf(stderr, "TRACE %-14s %8.3f ms\n", tag, (_n - g_t0) * 1e3); g_t0 = _n; } } while (0)
#define LOG(...) do { if (g_verbose) fprintf(stderr, __VA_ARGS__); } while (0)

/* ------------------------------------------------------------------ */
/* raw I/O and allocation                                              */
/* ------------------------------------------------------------------ */
static int rd(int fd, void *b, size_t n) {
    size_t o = 0;
    while (o < n) {
        ssize_t r = read(fd, (char *)b + o, n - o);
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        if (r == 0) return -1;
        o += (size_t)r;
    }
    return 0;
}
static int wr(int fd, const void *b, size_t n) {
    size_t o = 0;
    while (o < n) {
        ssize_t r = write(fd, (const char *)b + o, n - o);
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        if (r == 0) return -1;
        o += (size_t)r;
    }
    return 0;
}
static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "out of memory\n"); _exit(70); }
    return p;
}

/* ------------------------------------------------------------------ */
/* JSON header parsing (tiny, tolerant)                                */
/* ------------------------------------------------------------------ */
static const char *jkey(const char *j, const char *key) {
    char pat[64];
    int n = snprintf(pat, sizeof pat, "\"%s\"", key);
    if (n < 0 || n >= (int)sizeof pat) return NULL;
    const char *p = j;
    while ((p = strstr(p, pat)) != NULL) {
        const char *q = p + strlen(pat);
        while (*q == ' ' || *q == '\t') q++;
        if (*q == ':') return q + 1;
        p = q;
    }
    return NULL;
}
static double jnum(const char *j, const char *key, double dflt) {
    const char *p = jkey(j, key);
    if (!p) return dflt;
    while (*p == ' ' || *p == '\t') p++;
    return strtod(p, NULL);
}

/* ------------------------------------------------------------------ */
/* long double FFT (iterative radix-2) used for the coefficient DCT    */
/* ------------------------------------------------------------------ */
#define PI_LD 3.14159265358979323846264338327950288L

static void fft_ld(ld *re, ld *im, int n) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            ld t;
            t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        ld ang = -2.0L * PI_LD / (ld)len;
        ld wr = cosl(ang), wi = sinl(ang);
        int half = len >> 1;
        for (int i = 0; i < n; i += len) {
            ld cr = 1.0L, ci = 0.0L;
            for (int k = 0; k < half; k++) {
                int a = i + k, b = a + half;
                ld ur = re[a], ui = im[a];
                ld vr = re[b] * cr - im[b] * ci;
                ld vi = re[b] * ci + im[b] * cr;
                re[a] = ur + vr; im[a] = ui + vi;
                re[b] = ur - vr; im[b] = ui - vi;
                ld ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
        }
    }
}


/* double precision FFT (iterative radix-2) for the coefficient DCT */
static void fft_d(double *re, double *im, int n) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            double t;
            t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * 3.14159265358979323846 / (double)len;
        double wr = cos(ang), wi = sin(ang);
        int half = len >> 1;
        for (int i = 0; i < n; i += len) {
            double cr = 1.0, ci = 0.0;
            for (int k = 0; k < half; k++) {
                int a = i + k, b = a + half;
                double ur = re[a], ui = im[a];
                double vr = re[b] * cr - im[b] * ci;
                double vi = re[b] * ci + im[b] * cr;
                re[a] = ur + vr; im[a] = ui + vi;
                re[b] = ur - vr; im[b] = ui - vi;
                double ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
        }
    }
}

/* Chebyshev coefficients from f at the N Chebyshev-Gauss nodes (double FFT). */
static void cheb_coeff_d(const double *fv, int N, int m, double *c) {
    int M = 2 * N;
    double *re = (double *)xmalloc(sizeof(double) * (size_t)M);
    double *im = (double *)xmalloc(sizeof(double) * (size_t)M);
    for (int j = 0; j < N; j++) { re[j] = fv[j]; im[j] = 0.0; }
    memset(re + N, 0, sizeof(double) * (size_t)N);
    memset(im + N, 0, sizeof(double) * (size_t)N);
    fft_d(re, im, M);
    for (int k = 0; k <= m; k++) {
        double ang = -3.14159265358979323846 * (double)k / (double)(2 * N);
        double x = re[k] * cos(ang) - im[k] * sin(ang);
        c[k] = 2.0 * x / (double)N;
    }
    c[0] *= 0.5;
    free(re); free(im);
}

/* Chebyshev coefficients of f (given at the N Chebyshev-Gauss nodes
 * x_j = cos(pi (j+.5)/N)) in the convention f = c_0/2 + sum_{k>=1} c_k T_k. */
static void cheb_coeff(const ld *fv, int N, int m, double *c) {
    int M = 2 * N;
    ld *re = (ld *)xmalloc(sizeof(ld) * (size_t)M);
    ld *im = (ld *)xmalloc(sizeof(ld) * (size_t)M);
    for (int j = 0; j < N; j++) { re[j] = fv[j]; im[j] = 0.0L; }
    memset(re + N, 0, sizeof(ld) * (size_t)N);
    memset(im + N, 0, sizeof(ld) * (size_t)N);
    fft_ld(re, im, M);
    /* X_k = sum_j f_j cos(pi k (2j+1)/(2N)) = Re( e^{-i pi k/(2N)} S_k )
     * with S_k = sum_j f_j e^{-i pi k j/N} = DFT_{2N}(f,0)_k                */
    for (int k = 0; k <= m; k++) {
        ld ang = -PI_LD * (ld)k / (ld)(2 * N);
        ld x = re[k] * cosl(ang) - im[k] * sinl(ang);
        c[k] = (double)(2.0L * x / (ld)N);
    }
    c[0] *= 0.5;
    free(re); free(im);
}

/* coefficients of exp(z) and phi1(z) for z in [a,b]
 *
 * Nodes are z_j = b - 2*dd*sin^2(theta_j/2) (cancellation free, unlike
 * cc + dd*cos(theta_j)), the sines come from an angle addition recurrence with
 * an exact restart every 128 nodes, and exp() is only evaluated where it does
 * not underflow (elsewhere exp(z)=0 and phi1(z)=-1/z to machine accuracy).
 * The DCT itself runs in double, whose ~1e-16 relative input error is far
 * below the accuracy the truncated series can represent.
 */
static void make_coeffs(double a, double b, int m, double *ae, double *ap) {
    ld dd = ((ld)b - (ld)a) / 2.0L;
    int N = 64;
    while (N < m + 128) N <<= 1;
    double *fe = (double *)xmalloc(sizeof(double) * (size_t)N);
    double *fp = (double *)xmalloc(sizeof(double) * (size_t)N);
    /* node angle phi_j = pi*(j+0.5)/(2N) = (j+0.5)*dlt, x_j = 1 - 2 sin^2 phi_j */
    const ld dlt = PI_LD / (ld)(2 * N);
    const ld Cd = cosl(dlt), Sd = sinl(dlt);
    ld ch = cosl(0.5L * dlt), sh = sinl(0.5L * dlt);
    for (int j = 0; j < N; j++) {
        if (j > 0 && (j & 127) == 0) {       /* exact restart bounds the drift */
            ld th = ((ld)j + 0.5L) * dlt;
            ch = cosl(th); sh = sinl(th);
        }
        ld z = (ld)b - 2.0L * dd * sh * sh;
        if (z > -60.0L) {
            ld ez = expl(z);
            fe[j] = (double)ez;
            fp[j] = (fabsl(z) < 1e-6L)
                        ? (double)(1.0L + z / 2.0L + z * z / 6.0L + z * z * z / 24.0L)
                        : (double)((ez - 1.0L) / z);
        } else {
            fe[j] = 0.0;
            fp[j] = (double)(-1.0L / z);
        }
        ld nch = ch * Cd - sh * Sd;
        sh = sh * Cd + ch * Sd;
        ch = nch;
    }
    cheb_coeff_d(fe, N, m, ae);
    cheb_coeff_d(fp, N, m, ap);
    free(fe); free(fp);
}

/* ------------------------------------------------------------------ */
/* largest eigenvalue of a symmetric tridiagonal matrix                */
/* ------------------------------------------------------------------ */
static int sturm_neg(int m, const double *al, const double *be, double mu) {
    int cnt = 0;
    ld q = (ld)al[0] - (ld)mu;
    if (q < 0.0L) cnt++;
    if (q == 0.0L) q = -1e-300L;
    for (int i = 1; i < m; i++) {
        q = (ld)al[i] - (ld)mu - (ld)be[i - 1] * (ld)be[i - 1] / q;
        if (q < 0.0L) cnt++;
        if (q == 0.0L) q = -1e-300L;
    }
    return cnt;
}
static double tridiag_max(int m, const double *al, const double *be) {
    if (m <= 0) return 0.0;
    if (m == 1) return al[0];
    double lo = al[0], bb = 0.0, hi = al[0];
    for (int i = 0; i < m; i++) {
        if (al[i] < lo) lo = al[i];
        if (al[i] > hi) hi = al[i];
        if (i > 0) bb += fabs(be[i - 1]);
    }
    double l = lo - bb - 1.0, r = hi + bb + 1.0;
    for (int it = 0; it < 90; it++) {
        double mid = 0.5 * (l + r);
        if (mid <= l || mid >= r) break;
        if (sturm_neg(m, al, be, mid) < m) l = mid; else r = mid;
    }
    return r;
}

/* plain single vector SpMV: w = L v */
static void spmv1(int n, const uint32_t *ip, const uint32_t *ix, const double *va,
                  const double *v, double *w) {
    for (int i = 0; i < n; i++) {
        const uint32_t s = ip[i], e = ip[i + 1];
        double s0 = 0.0;
        for (uint32_t q = s; q < e; q++) s0 += va[q] * v[ix[q]];
        w[i] = s0;
    }
}

/* Lanczos without reorthogonalisation; returns an approximation of lambda_max
 * from below (the largest Ritz value of the Krylov tridiagonal matrix). */
static double lanczos_max(int n, const uint32_t *ip, const uint32_t *ix,
                          const double *va, int m, double Linf) {
    if (n <= 0) return 0.0;
    if (m > n) m = n;
    if (m < 2) m = 2;
    double *al = (double *)xmalloc(sizeof(double) * (size_t)m);
    double *be = (double *)xmalloc(sizeof(double) * (size_t)m);
    double *v = (double *)xmalloc(sizeof(double) * (size_t)n);
    double *vp = (double *)xmalloc(sizeof(double) * (size_t)n);
    double *w = (double *)xmalloc(sizeof(double) * (size_t)n);
    uint64_t st = 0x243F6A8885A308D3ULL ^ ((uint64_t)n * 0x9E3779B97F4A7C15ULL);
    double nrm = 0.0;
    for (int i = 0; i < n; i++) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        v[i] = (double)((st >> 11) & 0x1FFFFFFFFFFFFFULL) * (1.0 / 9007199254740992.0) - 0.5;
        nrm += v[i] * v[i];
    }
    nrm = sqrt(nrm);
    if (!(nrm > 0.0)) { free(al);free(be);free(v);free(vp);free(w); return 0.0; }
    for (int i = 0; i < n; i++) v[i] /= nrm;
    int done = 0;
    double beta = 0.0;
    int early = (m > 16) ? 12 : m;    /* cheap checkpoint before paying for more */
    for (int k = 0; k < m; k++) {
        spmv1(n, ip, ix, va, v, w);
        double alpha = 0.0;
        for (int i = 0; i < n; i++) alpha += v[i] * w[i];
        al[k] = alpha;
        for (int i = 0; i < n; i++) w[i] -= alpha * v[i];
        if (k > 0) for (int i = 0; i < n; i++) w[i] -= beta * vp[i];
        double nb = 0.0;
        for (int i = 0; i < n; i++) nb += w[i] * w[i];
        nb = sqrt(nb);
        if (!(nb > 1e-11 * (fabs(alpha) + 1e-300)) || !(nb > 0.0)) { done = k + 1; break; }
        be[k] = nb;
        memcpy(vp, v, sizeof(double) * (size_t)n);
        for (int i = 0; i < n; i++) v[i] = w[i] / nb;
        beta = nb;
        if (k + 1 == early && Linf > 0.0) {
            /* no room for a tighter interval: ||L||_inf is already near optimal */
            if (tridiag_max(k + 1, al, be) >= 0.97 * Linf) { done = k + 1; break; }
        }
    }
    int mm = done ? done : m;
    double r = tridiag_max(mm, al, be);
    free(al); free(be); free(v); free(vp); free(w);
    return r;
}

/* ------------------------------------------------------------------ */
/* fused Chebyshev step over a two column block                        */
/* ------------------------------------------------------------------ */
/* U = c1*(L V) + c2*V - W   (elementwise on 2 interleaved columns)
 * Y[i] += ca*U[i,0] + cb*U[i,1]
 * returns max_i max(|U[i,0]|,|U[i,1]|); *pmaxY = max_i |Y[i]|          */
static double cheb_step(int n, const uint32_t *ip, const uint32_t *ix, const double *va,
                        const double *V, const double *W, double *U,
                        double c1, double c2, double *Y, double ca, double cb,
                        double *pmaxY, int mon) {
    double mx = 0.0, my = 0.0;
    for (int i = 0; i < n; i++) {
        const size_t s = ip[i], e = ip[i + 1];
        double s0, s1;
        size_t q = s;
#if HAVE_SSE2
        /* four independent accumulator chains: with a 4 cycle FMA latency a
         * single chain would throttle the row loop to 4 cycles per non-zero */
        __m128d a0 = _mm_setzero_pd(), a1 = _mm_setzero_pd();
        for (; q + 4 <= e; q += 4) {
            a0 = FMADD(_mm_set1_pd(va[q]), _mm_loadu_pd(&V[2u * (size_t)ix[q]]), a0);
            a1 = FMADD(_mm_set1_pd(va[q + 1]), _mm_loadu_pd(&V[2u * (size_t)ix[q + 1]]), a1);
            a0 = FMADD(_mm_set1_pd(va[q + 2]), _mm_loadu_pd(&V[2u * (size_t)ix[q + 2]]), a0);
            a1 = FMADD(_mm_set1_pd(va[q + 3]), _mm_loadu_pd(&V[2u * (size_t)ix[q + 3]]), a1);
        }
        for (; q + 2 <= e; q += 2) {
            a0 = FMADD(_mm_set1_pd(va[q]), _mm_loadu_pd(&V[2u * (size_t)ix[q]]), a0);
            a1 = FMADD(_mm_set1_pd(va[q + 1]), _mm_loadu_pd(&V[2u * (size_t)ix[q + 1]]), a1);
        }
        for (; q < e; q++)
            a0 = FMADD(_mm_set1_pd(va[q]), _mm_loadu_pd(&V[2u * (size_t)ix[q]]), a0);
        a0 = _mm_add_pd(a0, a1);
        s0 = _mm_cvtsd_f64(a0);
        s1 = _mm_cvtsd_f64(_mm_unpackhi_pd(a0, a0));
#else
        double t0 = 0.0, t1 = 0.0, u0 = 0.0, u1 = 0.0;
        for (; q + 2 <= e; q += 2) {
            const double *x = &V[2u * (size_t)ix[q]];
            const double *y = &V[2u * (size_t)ix[q + 1]];
            t0 += va[q] * x[0]; t1 += va[q] * x[1];
            u0 += va[q + 1] * y[0]; u1 += va[q + 1] * y[1];
        }
        for (; q < e; q++) {
            const double *x = &V[2u * (size_t)ix[q]];
            t0 += va[q] * x[0]; t1 += va[q] * x[1];
        }
        s0 = t0 + u0; s1 = t1 + u1;
#endif
        size_t o = 2u * (size_t)i;
        double r0 = c1 * s0 + c2 * V[o] - W[o];
        double r1 = c1 * s1 + c2 * V[o + 1] - W[o + 1];
        U[o] = r0; U[o + 1] = r1;
        double yv = Y[i] + ca * r0 + cb * r1;
        Y[i] = yv;
        if (mon) {   /* divergence guard + tolerance estimate: only every few steps */
            double ay = fabs(r0) > fabs(r1) ? fabs(r0) : fabs(r1);
            if (ay > mx) mx = ay;
            ay = fabs(yv);
            if (ay > my) my = ay;
        }
    }
    if (mon) *pmaxY = my;
    return mx;
}


/* ------------------------------------------------------------------ */
/* Reverse Cuthill-McKee reordering                                    */
/* ------------------------------------------------------------------ */
typedef struct { uint32_t i; double v; } Ent;
static int entcmp(const void *pa, const void *pb) {
    const Ent *a = (const Ent *)pa, *b = (const Ent *)pb;
    return (a->i > b->i) - (a->i < b->i);
}
static void ent_sort(Ent *e, int len) {
    if (len <= 24) {
        for (int a = 1; a < len; a++) {
            Ent t = e[a];
            int b = a - 1;
            while (b >= 0 && e[b].i > t.i) { e[b + 1] = e[b]; b--; }
            e[b + 1] = t;
        }
    } else {
        qsort(e, (size_t)len, sizeof(Ent), entcmp);
    }
}

/* BFS from root, neighbours enqueued in ascending degree order (Cuthill-McKee).
 * order[0..n-1] receives the visiting order; disconnected components are
 * appended, each started from its lowest degree untouched node. */
static void cm_order(int n, const uint32_t *ip, const uint32_t *ix,
                     const int32_t *deg, int root, int *order) {
    char *seen = (char *)calloc((size_t)n, 1);
    int *queue = (int *)xmalloc(sizeof(int) * (size_t)n);
    int *nb = (int *)xmalloc(sizeof(int) * (size_t)n);
    if (!seen) { free(queue); free(nb); for (int i = 0; i < n; i++) order[i] = i; return; }
    int cnt = 0;
    for (int start = root; start >= 0; ) {
        int head = 0, tail = 0;
        queue[tail++] = start;
        seen[start] = 1;
        while (head < tail) {
            int v = queue[head++];
            order[cnt++] = v;
            int c = 0;
            for (uint32_t q = ip[v]; q < ip[v + 1]; q++) {
                int j = (int)ix[q];
                if (!seen[j]) { seen[j] = 1; nb[c++] = j; }
            }
            /* insertion sort neighbours by ascending degree */
            for (int a = 1; a < c; a++) {
                int t = nb[a], d = deg[t], b = a - 1;
                while (b >= 0 && deg[nb[b]] > d) { nb[b + 1] = nb[b]; b--; }
                nb[b + 1] = t;
            }
            for (int a = 0; a < c; a++) queue[tail++] = nb[a];
        }
        start = -1;
        if (cnt < n) {
            int best = -1;
            for (int i = 0; i < n; i++)
                if (!seen[i] && (best < 0 || deg[i] < deg[best])) best = i;
            start = best;
        }
    }
    free(seen); free(queue); free(nb);
}

/* sum over all nonzeros of |row - col|: cheap measure of ordering locality */
static double profile_cost(int n, const uint32_t *ip, const uint32_t *ix, const int *map) {
    double p = 0.0;
    for (int i = 0; i < n; i++) {
        int ni = map ? map[i] : i;
        for (uint32_t q = ip[i]; q < ip[i + 1]; q++) {
            int nj = map ? map[(int)ix[q]] : (int)ix[q];
            int d = ni - nj;
            p += (d < 0) ? -d : d;
        }
    }
    return p;
}

/* Builds perm (new index -> old index) using RCM.  Returns 1 on success. */
static int build_rcm(int n, const uint32_t *ip, const uint32_t *ix, int *perm) {
    int32_t *deg = (int32_t *)xmalloc(sizeof(int32_t) * (size_t)n);
    int root = 0;
    for (int i = 0; i < n; i++) {
        deg[i] = (int32_t)(ip[i + 1] - ip[i]);
        if (deg[i] < deg[root]) root = i;
    }
    /* two pseudo-peripheral refinements: BFS depth from a min degree node */
    int *order = (int *)xmalloc(sizeof(int) * (size_t)n);
    cm_order(n, ip, ix, deg, root, order);
    root = order[n - 1];
    cm_order(n, ip, ix, deg, root, order);
    root = order[n - 1];
    cm_order(n, ip, ix, deg, root, order);
    for (int i = 0; i < n; i++) perm[i] = order[n - 1 - i];   /* reverse => RCM */
    free(deg); free(order);
    return 1;
}

/* Rebuild the CSR of P L P^T where perm[new] = old. */
static void permute_csr(int n, size_t nnz, const uint32_t *ip, const uint32_t *ix,
                        const double *va, const int *perm, uint32_t *nip,
                        uint32_t *nix, double *nva) {
    int *inv = (int *)xmalloc(sizeof(int) * (size_t)n);
    for (int i = 0; i < n; i++) inv[perm[i]] = i;
    Ent *tmp = NULL;
    size_t tmpcap = 0;
    nip[0] = 0;
    for (int i = 0; i < n; i++) {
        int o = perm[i];
        nip[i + 1] = nip[i] + (ip[o + 1] - ip[o]);
    }
    for (int i = 0; i < n; i++) {
        int o = perm[i];
        uint32_t s = ip[o], e = ip[o + 1];
        size_t len = (size_t)(e - s);
        uint32_t d = nip[i];
        if (len > tmpcap) {
            tmpcap = len;
            free(tmp);
            tmp = (Ent *)xmalloc(sizeof(Ent) * tmpcap);
        }
        for (size_t k = 0; k < len; k++) {
            tmp[k].i = (uint32_t)inv[ix[s + k]];
            tmp[k].v = va[s + k];
        }
        ent_sort(tmp, (int)len);
        for (size_t k = 0; k < len; k++) {
            nix[d + (uint32_t)k] = tmp[k].i;
            nva[d + (uint32_t)k] = tmp[k].v;
        }
    }
    free(inv); free(tmp);
    (void)nnz;
}

static double phi1_of(double z) {
    if (fabs(z) < 1e-8) return 1.0 + z / 2.0 + z * z / 6.0;
    return expm1(z) / z;
}

/* ------------------------------------------------------------------ */
int main(int argc, char **argv) {
    (void)argc; (void)argv;
    if (getenv("FAMEXP2_VERBOSE")) g_verbose = 1;
    g_t0 = now_s();

    if (wr(1, "READY\n", 6) != 0) return 1;

    char magic[8];
    if (rd(0, magic, 8) != 0 || memcmp(magic, "FAMEXP2\n", 8) != 0) return 2;
    uint32_t jlen = 0;
    if (rd(0, &jlen, 4) != 0 || jlen < 1 || jlen > 4096) return 2;
    char json[4200];
    if (rd(0, json, jlen) != 0) return 2;
    json[jlen] = 0;

    long long n_ll = (long long)jnum(json, "n", -1.0);
    long long nnz_ll = (long long)jnum(json, "nnz", -1.0);
    long long req_ll = (long long)jnum(json, "requests", 1.0);
    double t = jnum(json, "t", 0.0);
    double rtol = jnum(json, "rtol", 1e-4);
    double atol = jnum(json, "atol", rtol * 1e-3);
    if (n_ll < 1 || n_ll > 120000 || nnz_ll < 0 || nnz_ll > 1000000) return 2;
    int n = (int)n_ll;
    size_t nnz = (size_t)nnz_ll;
    int requests = (int)req_ll;
    if (requests < 1) requests = 1;
    if (!(atol >= 0.0)) atol = rtol * 1e-3;

    TRACE("ready+hdr");
    uint64_t *ip64 = (uint64_t *)xmalloc(sizeof(uint64_t) * ((size_t)n + 1));
    if (rd(0, ip64, sizeof(uint64_t) * ((size_t)n + 1)) != 0) return 2;
    uint32_t *ix = nnz ? (uint32_t *)xmalloc(sizeof(uint32_t) * nnz) : NULL;
    double *va = nnz ? (double *)xmalloc(sizeof(double) * nnz) : NULL;
    if (nnz) {
        if (rd(0, ix, sizeof(uint32_t) * nnz) != 0) return 2;
        if (rd(0, va, sizeof(double) * nnz) != 0) return 2;
    }
    TRACE("csr-read");
    uint32_t *ip = (uint32_t *)xmalloc(sizeof(uint32_t) * ((size_t)n + 1));
    for (int i = 0; i <= n; i++) ip[i] = (uint32_t)ip64[i];
    free(ip64);

    double Linf = 0.0;
    double prof = 0.0;
    int isdiag = 1;
    for (int i = 0; i < n; i++) {
        uint32_t s = ip[i], e = ip[i + 1];
        double rs = 0.0;
        for (uint32_t q = s; q < e; q++) {
            rs += fabs(va[q]);
            int d = (int)ix[q] - i;
            prof += (d < 0) ? (double)-d : (double)d;
            if (ix[q] != (uint32_t)i) isdiag = 0;
        }
        if (rs > Linf) Linf = rs;
    }
    TRACE("scan");
    double Rinf = t * Linf;
    LOG("n=%d nnz=%zu t=%.8g rtol=%.3g Linf=%.8g Rinf=%.8g diag=%d req=%d\n",
        n, nnz, t, rtol, Linf, Rinf, isdiag, requests);

    double *diagv = NULL;
    if (isdiag && nnz) {
        diagv = (double *)xmalloc(sizeof(double) * (size_t)n);
        memset(diagv, 0, sizeof(double) * (size_t)n);
        for (int i = 0; i < n; i++)
            for (uint32_t q = ip[i]; q < ip[i + 1]; q++) diagv[i] += va[q];
    }

    /* calibrated degree rule: m = Cdeg*sqrt(R) + Kdeg  (err/tol ~ 1e-2 margin) */
    double Cdeg, Kdeg;
    if (rtol >= 1e-5)      { Cdeg = 3.45; Kdeg = 12.0; }
    else if (rtol >= 1e-8) { Cdeg = 4.25; Kdeg = 16.0; }
    else                   { Cdeg = 5.05; Kdeg = 22.0; }
    {
        const char *e = getenv("FAMEXP2_CDEG");
        if (e) Cdeg = atof(e);
        e = getenv("FAMEXP2_KDEG");
        if (e) Kdeg = atof(e);
    }

    int mode;                       /* 0: b0 | 1: b0+t b1 | 2: diagonal | 3: chebyshev */
    if (t == 0.0) mode = 0;
    else if (!(Rinf > 0.0)) mode = 1;
    else if (isdiag) mode = 2;
    else mode = 3;

    double R = Rinf, cc = 0.0, dd = 1.0;
    int m = 0, mcap = 0, mcap_alloc = 0, minf = 0;
    double *ae = NULL, *ap = NULL;

    int *perm = NULL;
    if (mode == 3) {
        m = (int)(Cdeg * sqrt(Rinf) + Kdeg) + 1;
        if (m < 8) m = 8;
        /* Reorder only when the given numbering really is local: the sweep is
         * gather bound, and RCM turns a random ordering into a banded one. */
        if (m >= 100 && nnz > 0 && prof > 0.0625 * (double)n * (double)nnz) {
            int *cand = (int *)xmalloc(sizeof(int) * (size_t)n);
            build_rcm(n, ip, ix, cand);
            int *inv = (int *)xmalloc(sizeof(int) * (size_t)n);
            for (int i = 0; i < n; i++) inv[cand[i]] = i;
            double pn = profile_cost(n, ip, ix, inv);
            free(inv);
            LOG("rcm profile %.4g -> %.4g\n", prof, pn);
            if (pn < 0.7 * prof) {
                uint32_t *nip = (uint32_t *)xmalloc(sizeof(uint32_t) * ((size_t)n + 1));
                uint32_t *nix = nnz ? (uint32_t *)xmalloc(sizeof(uint32_t) * nnz) : NULL;
                double *nva = nnz ? (double *)xmalloc(sizeof(double) * nnz) : NULL;
                permute_csr(n, nnz, ip, ix, va, cand, nip, nix, nva);
                free(ip); free(ix); free(va);
                ip = nip; ix = nix; va = nva;
                perm = cand;
            } else {
                free(cand);
            }
        }
        int use_lanczos = (requests > 1 && m >= 24) || m >= 200;
        if (use_lanczos) {
            double est = lanczos_max(n, ip, ix, va, 36, Linf);
            LOG("lanczos lmax=%.10g  Linf=%.10g  ratio=%.5f\n", est, Linf, est / Linf);
            /* only worth using when it actually narrows the interval */
            if (est > 0.0 && est < 0.97 * Linf) {
                double Rtry = t * est * 1.01;
                if (Rtry > 0.0 && Rtry < Rinf) R = Rtry;
            }
        }
        m = (int)(Cdeg * sqrt(R) + Kdeg) + 1;
        if (m < 8) m = 8;
        mcap = (int)(m * 1.30) + 48;
        /* allocation covers the rigorous interval too, so a retry never has to
         * truncate the series */
        minf = (int)(Cdeg * sqrt(Rinf) + Kdeg) + 1;
        mcap_alloc = (int)(minf * 1.30) + 64;
        if (mcap_alloc < mcap) mcap_alloc = mcap;
        double aa = -R, bb = R * 1e-8;   /* slack for round-off negative eigenvalues */
        cc = 0.5 * (aa + bb);
        dd = 0.5 * (bb - aa);
        ae = (double *)xmalloc(sizeof(double) * (size_t)(mcap_alloc + 2));
        ap = (double *)xmalloc(sizeof(double) * (size_t)(mcap_alloc + 2));
        make_coeffs(aa, bb, mcap, ae, ap);
        TRACE("coeffs");
        LOG("R=%.8g m=%d mcap=%d\n", R, m, mcap);
    }

    double c1 = -2.0 * t / dd, c2 = -2.0 * cc / dd;

    size_t n2 = 2u * (size_t)n;
    double *b0 = (double *)xmalloc(sizeof(double) * (size_t)n);
    double *b1 = (double *)xmalloc(sizeof(double) * (size_t)n);
    double *Y = (double *)xmalloc(sizeof(double) * (size_t)n);
    double *out = (double *)xmalloc(sizeof(double) * (size_t)n);
    double *V1 = NULL, *V2 = NULL, *V3 = NULL;
    if (mode == 3) {
        V1 = (double *)xmalloc(sizeof(double) * n2);
        V2 = (double *)xmalloc(sizeof(double) * n2);
        V3 = (double *)xmalloc(sizeof(double) * n2);
    }

    for (int rq = 0; rq < requests; rq++) {
        if (rd(0, b0, sizeof(double) * (size_t)n) != 0) return 2;
        if (rd(0, b1, sizeof(double) * (size_t)n) != 0) return 2;
        TRACE("rhs-read");

        if (mode == 0) {
            memcpy(out, b0, sizeof(double) * (size_t)n);
        } else if (mode == 1) {
            for (int i = 0; i < n; i++) out[i] = b0[i] + t * b1[i];
        } else if (mode == 2) {
            for (int i = 0; i < n; i++) {
                double z = -t * diagv[i];
                out[i] = exp(z) * b0[i] + t * phi1_of(z) * b1[i];
            }
        } else {
            double mb0 = 0.0, mb1 = 0.0;
            for (int i = 0; i < n; i++) {
                int o = perm ? perm[i] : i;
                double x0 = b0[o], x1 = b1[o];
                V1[2u * (size_t)i] = x0;
                V1[2u * (size_t)i + 1] = x1;
                Y[i] = ae[0] * x0 + t * ap[0] * x1;
                double x = fabs(x0); if (x > mb0) mb0 = x;
                x = fabs(x1); if (x > mb1) mb1 = x;
            }
            double nb0 = 0.0, nb1 = 0.0;
            for (int i = 0; i < n; i++) { nb0 += b0[i] * b0[i]; nb1 += b1[i] * b1[i]; }
            double bn = sqrt(nb0) > sqrt(nb1) ? sqrt(nb0) : sqrt(nb1);
            (void)mb0; (void)mb1;
            double blow = 1.5 * bn + 1e-300;
            double maxY = 0.0, mx;

            memset(V3, 0, sizeof(double) * n2);
            /* V2 = T_1 = Ahat V1  (half of the general step) */
            cheb_step(n, ip, ix, va, V1, V3, V2, 0.5 * c1, 0.5 * c2, Y, 0.0, 0.0, &maxY, 1);
            for (int i = 0; i < n; i++)
                Y[i] += ae[1] * V2[2u * (size_t)i] + t * ap[1] * V2[2u * (size_t)i + 1];

            double *Vp = V1, *Vc = V2, *Vn = V3;
            int bad = 0, k = 2, mon = 1;
            for (; k <= m; k++) {
                mon = ((k & 7) == 0) || (k == m);
                mx = cheb_step(n, ip, ix, va, Vc, Vp, Vn, c1, c2, Y, ae[k], t * ap[k], &maxY, mon);
                double *tmp = Vp; Vp = Vc; Vc = Vn; Vn = tmp;
                if (mon && mx > blow) { bad = 1; break; }
            }

            if (bad) {
                LOG("interval violation at k=%d (mx=%.4g blow=%.4g); retry with Rinf\n", k, mx, blow);
                R = Rinf;
                double aa = -R, bb = R * 1e-8;   /* slack for round-off negative eigenvalues */
                cc = 0.5 * (aa + bb); dd = 0.5 * (bb - aa);
                c1 = -2.0 * t / dd; c2 = -2.0 * cc / dd;
                m = (int)(Cdeg * sqrt(R) + Kdeg) + 1;
                if (m < 8) m = 8;
                mcap = (int)(m * 1.30) + 48;
                if (mcap > mcap_alloc) mcap = mcap_alloc;
                make_coeffs(aa, bb, mcap, ae, ap);
                for (int i = 0; i < n; i++) {
                    int o = perm ? perm[i] : i;
                    V1[2u * (size_t)i] = b0[o];
                    V1[2u * (size_t)i + 1] = b1[o];
                    Y[i] = ae[0] * b0[o] + t * ap[0] * b1[o];
                }
                memset(V3, 0, sizeof(double) * n2);
                cheb_step(n, ip, ix, va, V1, V3, V2, 0.5 * c1, 0.5 * c2, Y, 0.0, 0.0, &maxY, 1);
                for (int i = 0; i < n; i++)
                    Y[i] += ae[1] * V2[2u * (size_t)i] + t * ap[1] * V2[2u * (size_t)i + 1];
                Vp = V1; Vc = V2; Vn = V3;
                for (int kk = 2; kk <= m; kk++) {
                    cheb_step(n, ip, ix, va, Vc, Vp, Vn, c1, c2, Y, ae[kk], t * ap[kk], &maxY, 0);
                    double *tmp = Vp; Vp = Vc; Vc = Vn; Vn = tmp;
                }
            } else if (m < mcap) {
                /* safety extension: if the last term is not negligible relative to
                 * the acceptance threshold keep going (covers underestimates of the
                 * needed degree, e.g. when max|y| is much smaller than ||b||). */
                int extra = 0;
                double term = (fabs(ae[m]) + t * fabs(ap[m])) * mx;
                double tol = atol + rtol * maxY;
                while (term > 0.02 * tol && m < mcap && extra < 600) {
                    m++; extra++;
                    mx = cheb_step(n, ip, ix, va, Vc, Vp, Vn, c1, c2, Y, ae[m], t * ap[m], &maxY, 1);
                    double *tmp = Vp; Vp = Vc; Vc = Vn; Vn = tmp;
                    if (mx > blow) break;
                    term = (fabs(ae[m]) + t * fabs(ap[m])) * mx;
                    tol = atol + rtol * maxY;
                }
                if (extra) LOG("extended %d steps to m=%d (maxY=%.6g)\n", extra, m, maxY);
                m = (int)(Cdeg * sqrt(R) + Kdeg) + 1;   /* restore for next request */
                if (m < 8) m = 8;
            }
            if (perm) for (int i = 0; i < n; i++) out[perm[i]] = Y[i];
            else memcpy(out, Y, sizeof(double) * (size_t)n);
        }

        TRACE("solve");
        for (int i = 0; i < n; i++) if (!isfinite(out[i])) out[i] = 0.0;
        if (wr(1, out, sizeof(double) * (size_t)n) != 0) return 2;
        TRACE("write");
    }

    if (g_verbose) fflush(stderr);
    _exit(0);
    return 0;
}
