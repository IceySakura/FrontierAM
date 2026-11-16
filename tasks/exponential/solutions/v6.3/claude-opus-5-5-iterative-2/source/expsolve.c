/* Standalone FAMEXP2 session solver (no Python in the timed process).
 * y = exp(-tL) b0 + t*phi1(-tL) b1,  L sparse symmetric PSD (CSR).
 * Strategy mirrors solver.py (the Python fallback); see README.md. */
#define _GNU_SOURCE
#include "expkern.c"
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <errno.h>
#include <xmmintrin.h>

static int DEBUG = 0;
static double T0 = 0.0;
#include <time.h>
static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
#define LOG(...) do { if (DEBUG) { fprintf(stderr, "%.2fms ", 1e3 * (now() - T0)); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

/* ------------------------------------------------------------------ memory / io */
static void *xalloc(size_t bytes)
{
    if (bytes < 64) bytes = 64;
    void *p = NULL;
    if (bytes >= (1u << 21)) {
        size_t al = (size_t)1 << 21;
        size_t sz = (bytes + al - 1) & ~(al - 1);
        if (posix_memalign(&p, al, sz) != 0) p = NULL;
        else madvise(p, sz, MADV_HUGEPAGE);
    } else {
        if (posix_memalign(&p, 64, bytes) != 0) p = NULL;
    }
    if (!p) { fprintf(stderr, "out of memory\n"); _exit(3); }
    return p;
}

static void read_exact(void *buf, size_t len)
{
    char *p = (char *)buf;
    while (len > 0) {
        ssize_t r = read(0, p, len);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) { fprintf(stderr, "unexpected end of input\n"); _exit(2); }
        p += r; len -= (size_t)r;
    }
}

static void write_all(const void *buf, size_t len)
{
    const char *p = (const char *)buf;
    while (len > 0) {
        ssize_t w = write(1, p, len);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) { _exit(4); }
        p += w; len -= (size_t)w;
    }
}

static double json_num(const char *js, const char *key, int *ok)
{
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(js, pat);
    if (!p) { *ok = 0; return 0.0; }
    p += strlen(pat);
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p != ':') { *ok = 0; return 0.0; }
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    char *end;
    double v = strtod(p, &end);
    if (end == p) *ok = 0;
    return v;
}

/* ------------------------------------------------------------------ special functions */
static double gfun(double x, double t)
{
    if (x == 0.0) return t;
    return -expm1(-t * x) / x;
}

/* ive[k] = exp(-beta) I_k(beta), k = 0..K, by Miller's backward recurrence normalised
 * with I_0 + 2 sum_{k>=1} I_k = exp(beta). */
static void ive_all(double beta, i64 K, double *out)
{
    if (beta <= 0.0) {
        for (i64 k = 0; k <= K; k++) out[k] = (k == 0) ? 1.0 : 0.0;
        return;
    }
    i64 Ns = K + 60 + (i64)(2.0 * sqrt(beta));
    double *I = (double *)malloc(sizeof(double) * (size_t)(Ns + 2));
    I[Ns + 1] = 0.0;
    I[Ns] = 1e-280;
    double inv = 2.0 / beta;
    for (i64 k = Ns; k >= 1; k--) {
        I[k - 1] = (double)k * inv * I[k] + I[k + 1];
        if (I[k - 1] > 1e250)
            for (i64 q = k - 1; q <= Ns + 1; q++) I[q] *= 1e-250;
    }
    double s = I[0];
    for (i64 k = 1; k <= Ns; k++) s += 2.0 * I[k];
    for (i64 k = 0; k <= K; k++) out[k] = I[k] / s;
    free(I);
}

/* in-place iterative radix-2 complex FFT (forward), n power of two */
static void fft(i64 n, double *re, double *im)
{
    for (i64 i = 1, j = 0; i < n; i++) {
        i64 bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            double tr = re[i]; re[i] = re[j]; re[j] = tr;
            double ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    for (i64 len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * M_PI / (double)len;
        i64 half = len >> 1;
        for (i64 k = 0; k < half; k++) {
            double wr = cos(ang * (double)k), wi = sin(ang * (double)k);
            for (i64 i = k; i < n; i += len) {
                i64 j = i + half;
                double xr = re[j] * wr - im[j] * wi;
                double xi = re[j] * wi + im[j] * wr;
                re[j] = re[i] - xr; im[j] = im[i] - xi;
                re[i] += xr; im[i] += xi;
            }
        }
    }
}

/* Chebyshev coefficients (first kind points, N = power of two) of samples x[j] = f(cos(pi(j+.5)/N)):
 * a_k = (2/N) sum_j x_j cos(pi k (j+.5)/N), a_0 halved.  Uses the Makhoul DCT-II. */
static void dct_coeffs(i64 N, const double *x, double *a, i64 K)
{
    double *re = (double *)malloc(sizeof(double) * (size_t)N);
    double *im = (double *)calloc((size_t)N, sizeof(double));
    for (i64 j = 0; j < N / 2; j++) {
        re[j] = x[2 * j];
        re[N - 1 - j] = x[2 * j + 1];
    }
    fft(N, re, im);
    for (i64 k = 0; k <= K && k < N; k++) {
        double ang = -M_PI * (double)k / (2.0 * (double)N);
        double v = 2.0 * (re[k] * cos(ang) - im[k] * sin(ang));
        a[k] = v / (double)N;
    }
    for (i64 k = N; k <= K; k++) a[k] = 0.0;
    a[0] *= 0.5;
    free(re); free(im);
}

/* ------------------------------------------------------------------ session state */
typedef struct {
    i64 n, nnz, requests;
    double t, rtol, atol;
    i64 *ip; i32 *ix; double *a;
    double G, glo;
    int mode; /* 0 identity, 1 zero, 2 blocks, 3 cheb, 4 pairs */
    i32 *partner; double *pairF;
    /* cheb */
    double lo, hi, sa, sb;
    i64 K;
    double *c, *tc, *d, *td;
    double *wa, *wb, *y;
    i64 lag; int twostep;
    i64 *perm; double *bp0, *bp1, *yp;
    int pat; i64 *pip; i32 *pix; double *dg, cval;
    i64 bw; int16_t *ox; int16_t *pox;
    double chg, lmin_bound;   /* lmin_bound: a lower bound on lambda_max */
    i64 bw0;                  /* bandwidth of the input ordering */
    double *pre0, *pre1;      /* one-shot: L b0, L b1 from the Gershgorin pass (input order) */
    int prez0, prez1;
    /* SELL-4 */
    int sell; i64 nch, lagc; i64 *scp; i32 *scl, *scol; double *sval, *sdg;
    double *A0, *B0, *A1, *B1, *sb0, *sb1, *sy;
    double *I2a, *I2b;   /* interleaved two-vector buffers (own zero slot at 2*n4) */
    /* blocks */
    i64 nc; i64 *order, *coff, *voff; double *vec, *bf, *bg, *bwork;
} Sess;

static void cheb_coeffs(Sess *S)
{
    double t = S->t, lo = S->lo, hi = S->hi;
    double h = 0.5 * (hi - lo), m = 0.5 * (hi + lo);
    double beta = t * h;
    i64 K = (i64)(sqrt(110.0 * beta) + 40.0);
    double *c = (double *)malloc(sizeof(double) * (size_t)(K + 2));
    double *d = (double *)malloc(sizeof(double) * (size_t)(K + 2));
    ive_all(beta, K, c);
    double el = exp(-t * lo);
    for (i64 k = 0; k <= K; k++) {
        double v = (k == 0 ? 1.0 : 2.0) * c[k] * el;
        if (k & 1) v = -v;
        if (!isfinite(v)) v = 0.0;
        c[k] = v;
    }
    i64 N = 64;
    while (N < 2 * K + 2) N *= 2;
    double *gv = (double *)malloc(sizeof(double) * (size_t)N);
    double gmax = 0.0;
    for (i64 j = 0; j < N; j++) {
        double x = m + h * cos(M_PI * ((double)j + 0.5) / (double)N);
        gv[j] = gfun(x, t);
        if (fabs(gv[j]) > gmax) gmax = fabs(gv[j]);
    }
    dct_coeffs(N, gv, d, K);
    free(gv);
    double floor_ = 32.0 * 2.220446049250313e-16 * gmax;
    i64 lastd = 0;
    for (i64 k = 0; k <= K; k++) if (fabs(d[k]) > floor_) lastd = k;
    for (i64 k = lastd + 1; k <= K; k++) d[k] = 0.0;
    double sc = 0.0;
    for (i64 k = 0; k <= K; k++) sc += fabs(c[k]);
    i64 lastc = 0;
    for (i64 k = 0; k <= K; k++) if (fabs(c[k]) > 1e-30 * (sc > 1e-300 ? sc : 1e-300)) lastc = k;
    i64 last = lastc > lastd ? lastc : lastd;
    i64 KK = last + 1;
    if (KK > K) KK = K;
    if (S->c) { free(S->c); free(S->d); free(S->tc); free(S->td); }
    S->K = KK;
    S->c = (double *)calloc((size_t)(KK + 2), sizeof(double));
    S->d = (double *)calloc((size_t)(KK + 2), sizeof(double));
    S->tc = (double *)calloc((size_t)(KK + 3), sizeof(double));
    S->td = (double *)calloc((size_t)(KK + 3), sizeof(double));
    for (i64 k = 0; k <= KK; k++) { S->c[k] = c[k]; S->d[k] = d[k]; }
    S->tc[KK + 1] = 0.0; S->td[KK + 1] = 0.0;
    for (i64 k = KK; k >= 0; k--) {
        S->tc[k] = S->tc[k + 1] + fabs(S->c[k]);
        S->td[k] = S->td[k + 1] + fabs(S->d[k]);
    }
    free(c); free(d);
    S->sa = 1.0 / h;
    S->sb = -m / h;
    LOG("interval %.17g %.17g K %ld", lo, hi, (long)KK);
}

static i64 degree_estimate(const Sess *S)
{
    /* first k with tail below 1e-3*rtol*tail[0] (planning only) */
    i64 k1 = 0, k2 = 0;
    double th1 = 1e-3 * S->rtol * S->tc[0], th2 = 1e-3 * S->rtol * S->td[0];
    while (k1 <= S->K && S->tc[k1] > th1) k1++;
    while (k2 <= S->K && S->td[k2] > th2) k2++;
    i64 k = k1 > k2 ? k1 : k2;
    return k < 1 ? 1 : k;
}

/* Sturm count: number of eigenvalues of T less than x */
static i64 sturm(i64 m, const double *al, const double *be, double x)
{
    i64 cnt = 0;
    double q = al[0] - x;
    if (q < 0) cnt++;
    for (i64 i = 1; i < m; i++) {
        if (q == 0.0) q = 1e-300;
        q = al[i] - x - be[i - 1] * be[i - 1] / q;
        if (q < 0) cnt++;
    }
    return cnt;
}

typedef struct { double *v0, *work, *al, *be; i64 done, cap; int dead; } LCtx;

/* Largest Ritz value and K-W style upper bound from the first mm Lanczos coefficients. */
static double lanczos_eval(Sess *S, LCtx *L, i64 mm, i64 mreq, double *theta_out)
{
    i64 n = S->n;
    double *al = L->al, *be = L->be;
    double glo = INFINITY, ghi = -INFINITY;
    for (i64 i = 0; i < mm; i++) {
        double r = (i > 0 ? fabs(be[i - 1]) : 0.0) + (i < mm - 1 ? fabs(be[i]) : 0.0);
        if (al[i] - r < glo) glo = al[i] - r;
        if (al[i] + r > ghi) ghi = al[i] + r;
    }
    double lo = glo, hi = ghi;
    for (int it = 0; it < 200 && hi - lo > 1e-15 * fabs(hi); it++) {
        double mid = 0.5 * (lo + hi);
        if (sturm(mm, al, be, mid) >= mm) hi = mid; else lo = mid;
    }
    double th = hi;
    double est;
    if (mm < mreq && fabs(be[mm - 1]) <= 1e-300) est = th * (1.0 + 1e-10);   /* invariant subspace */
    else {
        /* Kuczynski-Wozniakowski: random start, P(lmax > theta1/(1-eps)) <= 1.648 sqrt(n) e^{-sqrt(eps)(2m-1)} */
        double C = log(1.648 * sqrt((double)n) / 1e-2) / (double)(2 * mm - 1);
        double eps = C * C;
        if (eps > 0.5) eps = 0.5;
        est = th / (1.0 - eps) * (1.0 + 1e-9);
    }
    LOG("lanczos m %ld theta1 %.10g G %.10g est %.10g", (long)mm, th, S->G, est);
    if (theta_out) *theta_out = th;
    if (!isfinite(est)) return S->G;
    return est < S->G ? est : S->G;
}

/* Run (or continue) Lanczos up to m steps; returns the bound. */
static double lanczos_run(Sess *S, LCtx *L, i64 m, double *theta_out)
{
    i64 n = S->n;
    if (!L->v0) {
        L->cap = 128;
        L->v0 = (double *)xalloc(sizeof(double) * (size_t)n);
        L->work = (double *)xalloc(sizeof(double) * 3 * (size_t)n);
        L->al = (double *)calloc((size_t)L->cap, sizeof(double));
        L->be = (double *)calloc((size_t)L->cap, sizeof(double));
        uint64_t st = 0x9E3779B97F4A7C15ull;
        double nrm = 0.0;
        for (i64 i = 0; i < n; i++) {
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            double u = ((double)(st >> 11) + 0.5) * (1.0 / 9007199254740992.0) - 0.5;
            L->v0[i] = u;
            nrm += u * u;
        }
        nrm = 1.0 / sqrt(nrm);
        for (i64 i = 0; i < n; i++) L->v0[i] *= nrm;
        L->done = 0;
    }
    if (m > L->cap) m = L->cap;
    if (!L->dead && m > L->done) {
        i64 mm = lanczos_from(n, S->ip, S->ix, S->a, L->done, m, L->v0, L->al, L->be, L->work);
        if (mm < m) L->dead = 1;   /* breakdown: exact invariant subspace */
        L->done = mm;
    }
    return lanczos_eval(S, L, L->done, m, theta_out);
}

static void set_interval(Sess *S, double hi)
{
    S->lo = S->glo > 0 ? S->glo * (1.0 - 1e-12) : -1e-13 * hi;
    S->hi = hi * (1.0 + 1e-12) + 1e-300;
}

static void prepare_blocks(Sess *S, i64 nc, i64 *order, i64 *coff, i64 smax)
{
    i64 n = S->n;
    i64 *voff = (i64 *)xalloc(sizeof(i64) * (size_t)(nc + 1));
    voff[0] = 0;
    for (i64 c = 0; c < nc; c++) {
        i64 s = coff[c + 1] - coff[c];
        voff[c + 1] = voff[c] + s * s;
    }
    i64 *loc = (i64 *)xalloc(sizeof(i64) * (size_t)n);
    double *lam = (double *)xalloc(sizeof(double) * (size_t)n);
    double *vec = (double *)xalloc(sizeof(double) * (size_t)voff[nc]);
    double *work = (double *)xalloc(sizeof(double) * (size_t)(2 * smax * smax + 16));
    LOG("  block alloc");
    block_eig(n, S->ip, S->ix, S->a, nc, order, coff, voff, loc, lam, vec, work);
    LOG("  block eig");
    S->bf = (double *)xalloc(sizeof(double) * (size_t)n);
    S->bg = (double *)xalloc(sizeof(double) * (size_t)n);
    for (i64 i = 0; i < n; i++) {
        double x = lam[i], tx = S->t * x;
        double e = exp(-tx);
        S->bf[i] = e;
        /* (1-e)/x is cancellation-free once tx > 0.25; expm1 below that */
        S->bg[i] = tx > 0.25 ? (1.0 - e) / x : gfun(x, S->t);
    }
    LOG("  block f/g");
    S->nc = nc; S->order = order; S->coff = coff; S->voff = voff; S->vec = vec;
    S->bwork = (double *)xalloc(sizeof(double) * (size_t)(smax + 16));
    free(loc); free(lam); free(work);
    S->mode = 2;
}

/* BFS components that also returns the node order grouped by component.
 * Returns -1 as soon as a component exceeds limit. */
static i64 components_order(i64 n, const i64 *ip, const i32 *ix, const double *a,
                            i64 *order, i64 *coff, i64 limit, i64 *smax, char *mark)
{
    memset(mark, 0, (size_t)n);
    i64 nc = 0, tl = 0, mx = 0;
    for (i64 s = 0; s < n; s++) {
        if (mark[s]) continue;
        i64 start = tl, head = tl;
        order[tl++] = s; mark[s] = 1;
        while (head < tl) {
            i64 u = order[head++];
            for (i64 k = ip[u]; k < ip[u + 1]; k++) {
                i64 v = ix[k];
                if (a[k] != 0.0 && !mark[v]) { mark[v] = 1; order[tl++] = v; }
            }
            if (tl - start > limit) { *smax = tl - start; return -1; }
        }
        coff[nc++] = start;
        if (tl - start > mx) mx = tl - start;
    }
    coff[nc] = n;
    *smax = mx;
    return nc;
}

static int cmp_u64(const void *x, const void *y)
{
    uint64_t a = *(const uint64_t *)x, b = *(const uint64_t *)y;
    return a < b ? -1 : (a > b);
}

/* BFS (Cuthill-McKee style) order, then rows sorted by decreasing length inside windows of
 * W rows: keeps locality while making row lengths predictable for the branch predictor. */
static void reorder(Sess *S)
{
    i64 n = S->n, nnz = S->nnz;
    i64 *perm = (i64 *)xalloc(sizeof(i64) * (size_t)n);
    i64 *mark = (i64 *)xalloc(sizeof(i64) * (size_t)n);
    /* an input ordering that is already banded is kept (only the window sort is applied) */
    int keep = S->bw0 < n / 16;
    bfs_order(n, keep ? NULL : S->ip, S->ix, perm, mark);
    LOG("  bfs done (input bandwidth %ld, kept %d)", (long)S->bw0, keep);
    /* stable counting sort by decreasing row length (capped at 63) inside each window */
    const i64 W = 1024;
    i64 *tmp = (i64 *)malloc(sizeof(i64) * (size_t)W);
    i64 *ltmp = (i64 *)malloc(sizeof(i64) * (size_t)W);
    uint64_t *lkeys = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)W);
    for (i64 s0 = 0; s0 < n; s0 += W) {
        i64 e = s0 + W < n ? s0 + W : n;
        i64 cnt[65];
        memset(cnt, 0, sizeof cnt);
        for (i64 q = s0; q < e; q++) {
            i64 u = perm[q], len = S->ip[u + 1] - S->ip[u];
            cnt[63 - (len > 63 ? 63 : len) + 1]++;
        }
        for (int b = 1; b < 65; b++) cnt[b] += cnt[b - 1];
        i64 nlong = cnt[1];   /* rows with length >= 63 occupy tmp[0..nlong) */
        for (i64 q = s0; q < e; q++) {
            i64 u = perm[q], len = S->ip[u + 1] - S->ip[u];
            tmp[cnt[63 - (len > 63 ? 63 : len)]++] = u;
        }
        if (nlong > 1) {   /* long (hub) rows: exact decreasing length */
            for (i64 q = 0; q < nlong; q++) {
                uint64_t len = (uint64_t)(S->ip[tmp[q] + 1] - S->ip[tmp[q]]);
                if (len > 0xffffffffull) len = 0xffffffffull;
                lkeys[q] = ((0xffffffffull - len) << 32) | (uint64_t)q;
            }
            qsort(lkeys, (size_t)nlong, sizeof(uint64_t), cmp_u64);
            for (i64 q = 0; q < nlong; q++) ltmp[q] = tmp[lkeys[q] & 0xffffffffull];
            memcpy(tmp, ltmp, sizeof(i64) * (size_t)nlong);
        }
        memcpy(perm + s0, tmp, sizeof(i64) * (size_t)(e - s0));
    }
    free(tmp); free(ltmp); free(lkeys);
    LOG("  window sort done");
    i64 *iperm = mark;
    for (i64 i = 0; i < n; i++) iperm[perm[i]] = i;
    i64 *nip = (i64 *)xalloc(sizeof(i64) * (size_t)(n + 1));
    i32 *nix = (i32 *)xalloc(sizeof(i32) * (size_t)(nnz + 1));
    double *na = (double *)xalloc(sizeof(double) * (size_t)(nnz + 1));
    S->bw = permute_csr(n, S->ip, S->ix, S->a, perm, iperm, nip, nix, na);
    LOG("  permute done");
    free(S->ip); free(S->ix); free(S->a); free(iperm);
    S->ip = nip; S->ix = nix; S->a = na;
    S->perm = perm;
}

static int all_zero(const double *v, i64 n);

static void prepare(Sess *S, const double *b0, const double *b1)
{
    i64 n = S->n;
    if (S->t == 0.0) { S->mode = 0; return; }
    double gi[6];
    if (b0) {
        /* one-shot: the request is already known; fuse the first products into this pass */
        S->prez0 = all_zero(b0, n);
        S->prez1 = all_zero(b1, n);
        if (!S->prez0) S->pre0 = (double *)xalloc(sizeof(double) * (size_t)n);
        if (!S->prez1) S->pre1 = (double *)xalloc(sizeof(double) * (size_t)n);
        gersh_spmv(n, S->ip, S->ix, S->a, gi, S->prez0 ? NULL : b0, S->prez1 ? NULL : b1, S->pre0, S->pre1);
    } else {
        gersh(n, S->ip, S->ix, S->a, gi);
    }
    S->G = gi[0]; S->glo = gi[1]; S->chg = gi[2]; S->lmin_bound = gi[4]; S->bw0 = (i64)gi[5];
    LOG("G %.10g glo %.10g", S->G, S->glo);
    if (!(S->G > 0.0)) { S->mode = 1; return; }
    set_interval(S, S->G);
    cheb_coeffs(S);
    i64 kest = degree_estimate(S);
    double cheb_cost = (double)S->requests * (double)kest * ((double)S->nnz * 0.9 + (double)n * 3.0);
    {
        /* blocks of size <= 2 (every row has at most one off-diagonal partner): closed form */
        double pair_cost = (double)n * 45.0 + (double)S->requests * (double)n * 4.0;
        if (n > 64 && pair_cost < cheb_cost) {
            i32 *partner = (i32 *)xalloc(sizeof(i32) * (size_t)n);
            if (pair_structure(n, S->ip, S->ix, S->a, partner)) {
                S->partner = partner;
                S->pairF = (double *)xalloc(sizeof(double) * 4 * (size_t)n);
                pair_setup(n, S->ip, S->ix, S->a, partner, S->t, S->pairF);
                S->mode = 4;
                LOG("pair path (pair_cost %g cheb_cost %g)", pair_cost, cheb_cost);
                return;
            }
            free(partner);
        }
    }
    i64 *order = (i64 *)xalloc(sizeof(i64) * (size_t)n);
    i64 *coff = (i64 *)xalloc(sizeof(i64) * (size_t)(n + 1));
    char *mark = (char *)xalloc((size_t)n);
    i64 smax = 0, nc = -1;
    if (n <= 64) {
        for (i64 i = 0; i < n; i++) order[i] = i;
        coff[0] = 0; coff[1] = n; nc = 1; smax = n;
    } else if (cheb_cost > (double)n * 60.0) {
        /* a block path can only win when the polynomial work is substantial */
        nc = components_order(n, S->ip, S->ix, S->a, order, coff, 24, &smax, mark);
    }
    LOG("components %ld max %ld", (long)nc, (long)smax);
    if (nc > 0) {
        /* rough nanosecond models: blocks = eigen + exp/expm1 per eigenvalue + applies;
         * Chebyshev ~0.9 ns per stored entry and 3 ns per row and step */
        double block_cost = (double)n * ((smax <= 2 ? 10.0 : (double)(smax * smax) * 40.0) + 50.0)
                          + (double)S->requests * (double)n * ((double)smax * 4.0 + 6.0);
        LOG("block_cost %g cheb_cost %g", block_cost, cheb_cost);
        if (n <= 64 || block_cost < cheb_cost) {
            free(mark);
            prepare_blocks(S, nc, order, coff, smax);
            return;
        }
    }
    free(order); free(coff); free(mark);
    double hi = S->G;
    double work_units = (double)S->requests * (double)kest;
    double chg = S->chg;
    LOG("kest %ld len_changes %.3f", (long)kest, chg);
    if (work_units >= 35 && chg > 0.2 && n > 2048) {
        reorder(S);
        LOG("reordered");
    }
    /* lambda_max >= ell, so no upper bound can save more than a fraction 1 - sqrt(ell/hi) */
    double ell = S->lmin_bound;
    double gain_g = ell > 0.0 ? 1.0 - sqrt(ell / hi) : 1.0;
    LOG("lower bound %.10g gain<=%.4f", ell, gain_g);
    if (work_units >= 40 && n > 64 && work_units * gain_g > 4.0) {
        /* rigorous Collatz-Wielandt bound on rho(|L|) >= lambda_max(L) */
        double *cw = (double *)xalloc(sizeof(double) * 2 * (size_t)n);
        int iters = work_units < 100 ? 5 : (work_units < 400 ? 8 : (work_units < 2000 ? 10 : (work_units < 8000 ? 16 : 24)));
        double hcw = cw_bound(n, S->ip, S->ix, S->a, iters, cw);
        free(cw);
        if (hcw < hi) hi = hcw * (1.0 + 1e-10);
        LOG("cw bound %.10g (G %.10g, %d steps)", hcw, S->G, iters);
        double gain_cw = ell > 0.0 ? 1.0 - sqrt(ell / hi) : 1.0;
        if (work_units >= 60 && work_units * gain_cw > 1.2 * 8.0) {
            /* short Lanczos: theta1 <= lambda_max tells whether the CW bound can be beaten;
             * if so, continue the same Lanczos run to m steps */
            LCtx lc;
            memset(&lc, 0, sizeof lc);
            double th = 0.0;
            lanczos_run(S, &lc, 8, &th);
            double C = log(164.8 * sqrt((double)n));
            i64 m = (i64)cbrt(work_units * C * C / 4.0);
            if (m < 12) m = 12;
            if (m > 120) m = 120;
            double eps = (C / (double)(2 * m - 1)) * (C / (double)(2 * m - 1));
            if (eps > 0.5) eps = 0.5;
            LOG("theta8 %.10g ratio %.4f kw-factor(m=%ld) %.4f", th, hi / th, (long)m, 1.0 / (1.0 - eps));
            /* best case: lambda_max ~ theta8, so the K-W bound could reach theta8/(1-eps);
             * run the remaining steps only if that saving can repay them */
            double gain = th > 0.0 ? 1.0 - sqrt(th / (1.0 - eps) / hi) : 0.0;
            if (gain > 0.0 && work_units * gain > 1.2 * (double)(m - 8)) {
                double hkw = lanczos_run(S, &lc, m, NULL);
                if (hkw < hi) hi = hkw;
            }
            if (lc.v0) { free(lc.v0); free(lc.work); free(lc.al); free(lc.be); }
        }
    }
    {
        /* testing knob: deliberately shrink the interval to exercise the divergence fallback */
        const char *fs = getenv("EXPSOLVER_FORCE_HI_SCALE");
        if (fs) hi *= atof(fs);
    }
    set_interval(S, hi);
    cheb_coeffs(S);
    S->y = (double *)xalloc(sizeof(double) * (size_t)n);
    S->wa = (double *)xalloc(sizeof(double) * 2 * (size_t)n);
    S->wb = (double *)xalloc(sizeof(double) * 2 * (size_t)n);
    if (S->perm) {
        S->bp0 = (double *)xalloc(sizeof(double) * (size_t)n);
        S->bp1 = (double *)xalloc(sizeof(double) * (size_t)n);
        S->yp = (double *)xalloc(sizeof(double) * (size_t)n);
    }
    i64 far = 0;
    i64 bw = S->perm ? S->bw : (work_units >= 20 ? bandwidth(n, S->ip, S->ix, 4096, &far) : n);
    S->lag = bw < n ? bw : n;
    LOG("bandwidth %ld", (long)bw);
    {
        i64 *pip = (i64 *)xalloc(sizeof(i64) * (size_t)(n + 1));
        i32 *pix = (i32 *)xalloc(sizeof(i32) * (size_t)(S->nnz + 1));
        double *dg = (double *)xalloc(sizeof(double) * (size_t)n);
        double cval = 0.0;
        if (split_pattern(n, S->ip, S->ix, S->a, pip, pix, dg, &cval)) {
            S->pat = 1; S->pip = pip; S->pix = pix; S->dg = dg; S->cval = cval;
            LOG("pattern matrix, off-diagonal value %.17g", cval);
        } else {
            free(pip); free(pix); free(dg);
        }
    }
#ifdef HAVE_SELL
    if (S->lag < n / 8 && work_units >= 40) {   /* SELL-4 pays off only with locality and enough steps */
        i64 nch = (n + 3) / 4, n4 = 4 * nch;
        const i64 *sip = S->pat ? S->pip : S->ip;
        const i32 *six = S->pat ? S->pix : S->ix;
        i64 tot = sell_build(n, sip, six, NULL, NULL, NULL, NULL, NULL);
        S->nch = nch;
        S->scp = (i64 *)xalloc(sizeof(i64) * (size_t)(nch + 1));
        S->scl = (i32 *)xalloc(sizeof(i32) * (size_t)(nch + 1));
        S->scol = (i32 *)xalloc(sizeof(i32) * (size_t)(4 * tot + 4));
        S->sval = S->pat ? NULL : (double *)xalloc(sizeof(double) * (size_t)(4 * tot + 4));
        sell_build(n, sip, six, S->pat ? NULL : S->a, S->scp, S->scl, S->scol, S->sval);
        if (S->pat) {
            S->sdg = (double *)xalloc(sizeof(double) * (size_t)(n4 + 8));
            memset(S->sdg, 0, sizeof(double) * (size_t)(n4 + 8));
            memcpy(S->sdg, S->dg, sizeof(double) * (size_t)n);
        }
        double **bufs[7] = {&S->A0, &S->B0, &S->A1, &S->B1, &S->sb0, &S->sb1, &S->sy};
        for (int q = 0; q < 7; q++) {
            *bufs[q] = (double *)xalloc(sizeof(double) * (size_t)(n4 + 8));
            memset(*bufs[q], 0, sizeof(double) * (size_t)(n4 + 8));
        }
        /* interleaved two-vector kernel: wins for rows of >= 5 entries (shuffles amortised) */
        if ((double)tot >= 5.0 * (double)nch && !getenv("EXPSOLVER_NO_I2")) {
            S->I2a = (double *)xalloc(sizeof(double) * (size_t)(2 * n4 + 16));
            S->I2b = (double *)xalloc(sizeof(double) * (size_t)(2 * n4 + 16));
            memset(S->I2a, 0, sizeof(double) * (size_t)(2 * n4 + 16));
            memset(S->I2b, 0, sizeof(double) * (size_t)(2 * n4 + 16));
        }
        S->lagc = S->lag >= n ? nch : (S->lag + 3) / 4 + 1;
        S->sell = 1;
        LOG("sell-4 built: %ld groups (padding %.3f) lagc %ld", (long)tot, 4.0 * (double)tot / (double)(sip[n] > 0 ? sip[n] : 1), (long)S->lagc);
    }
#endif
    if (!S->sell && S->lag < 32767) {
        /* 16-bit relative column offsets: 2 bytes less per stored entry */
        if (S->pat) {
            S->pox = (int16_t *)xalloc(sizeof(int16_t) * (size_t)(S->pip[n] + 1));
            to_offsets16(n, S->pip, S->pix, S->pox);
        } else {
            S->ox = (int16_t *)xalloc(sizeof(int16_t) * (size_t)(S->nnz + 1));
            to_offsets16(n, S->ip, S->ix, S->ox);
        }
        LOG("16-bit offsets");
    }
    S->mode = 3;
}

/* Divergence fallback: progressively wider intervals.  hi: x1.1 (if below Gershgorin), then
 * Gershgorin; lo: round-off margins growing to -1e-6 G (exp(-t lo) <= e^0.1 since t G <= 1e5). */
static void widen_interval(Sess *S, int attempt)
{
    double G = S->G;
    if (attempt == 0 && S->hi * 1.1 < G) {
        S->hi *= 1.1;
    } else if (attempt <= 1) {
        S->lo = -1e-13 * G;
        S->hi = G * (1.0 + 1e-10) + 1e-300;
    } else if (attempt == 2) {
        S->lo = -1e-8 * G;
        S->hi = G * 1.01 + 1e-300;
    } else {
        S->lo = -1e-6 * G;
        S->hi = G * 1.02 + 1e-300;
    }
    LOG("divergence: widened interval to [%.6g, %.6g]", S->lo, S->hi);
}

static int all_zero(const double *v, i64 n)
{
    for (i64 i = 0; i < n; i++) if (v[i] != 0.0) return 0;
    return 1;
}

static double *solve(Sess *S, double *b0, double *b1, double *yout)
{
    i64 n = S->n;
    if (S->mode == 0) return b0;
    if (S->mode == 1) {
        for (i64 i = 0; i < n; i++) yout[i] = b0[i] + S->t * b1[i];
        return yout;
    }
    if (S->mode == 4) {
        pair_apply(n, S->partner, S->pairF, b0, b1, yout);
        return yout;
    }
    if (S->mode == 2) {
        block_apply(S->nc, S->order, S->coff, S->voff, S->vec, S->bf, S->bg, b0, b1, yout, S->bwork);
        return yout;
    }
    int z0 = all_zero(b0, n), z1 = all_zero(b1, n);
    if (z0 && z1) {
        memset(yout, 0, sizeof(double) * (size_t)n);
        return yout;
    }
#ifdef HAVE_SELL
    if (S->sell) {
        double safety = 10.0, growlim = 1.25, info[4];
        if (S->perm) {
            if (!z0) gather(n, S->perm, b0, S->sb0);
            if (!z1) gather(n, S->perm, b1, S->sb1);
        } else {
            if (!z0) memcpy(S->sb0, b0, sizeof(double) * (size_t)n);
            if (!z1) memcpy(S->sb1, b1, sizeof(double) * (size_t)n);
        }
        for (int attempt = 0; attempt < 5; attempt++) {
            i64 k;
#define SELLARGS(SFX) S->nch, S->scp, S->scl, S->scol, SFX
            if (S->pat) {
                if (z0) k = sell1_pat(SELLARGS(S->sdg), S->cval, S->sa, S->sb, S->d, S->td, S->d, S->td, S->K, S->sb1, S->sb1,
                                      S->sy, S->A0, S->B0, S->A1, S->B1, S->atol, S->rtol, safety, growlim, info, S->lagc);
                else if (z1) k = sell1_pat(SELLARGS(S->sdg), S->cval, S->sa, S->sb, S->c, S->tc, S->c, S->tc, S->K, S->sb0, S->sb0,
                                           S->sy, S->A0, S->B0, S->A1, S->B1, S->atol, S->rtol, safety, growlim, info, S->lagc);
                else if (S->I2a) k = sell2i_pat(SELLARGS(S->sdg), S->cval, S->sa, S->sb, S->c, S->tc, S->d, S->td, S->K,
                                                S->sb0, S->sb1, S->sy, S->I2a, S->I2b, S->atol, S->rtol, safety, growlim,
                                                info, S->lagc);
                else k = sell2_pat(SELLARGS(S->sdg), S->cval, S->sa, S->sb, S->c, S->tc, S->d, S->td, S->K, S->sb0, S->sb1,
                                   S->sy, S->A0, S->B0, S->A1, S->B1, S->atol, S->rtol, safety, growlim, info, S->lagc);
            } else {
                if (z0) k = sell1_csr(SELLARGS(S->sval), S->sa, S->sb, S->d, S->td, S->d, S->td, S->K, S->sb1, S->sb1,
                                      S->sy, S->A0, S->B0, S->A1, S->B1, S->atol, S->rtol, safety, growlim, info, S->lagc);
                else if (z1) k = sell1_csr(SELLARGS(S->sval), S->sa, S->sb, S->c, S->tc, S->c, S->tc, S->K, S->sb0, S->sb0,
                                           S->sy, S->A0, S->B0, S->A1, S->B1, S->atol, S->rtol, safety, growlim, info, S->lagc);
                else if (S->I2a) k = sell2i_csr(SELLARGS(S->sval), S->sa, S->sb, S->c, S->tc, S->d, S->td, S->K,
                                                S->sb0, S->sb1, S->sy, S->I2a, S->I2b, S->atol, S->rtol, safety, growlim,
                                                info, S->lagc);
                else k = sell2_csr(SELLARGS(S->sval), S->sa, S->sb, S->c, S->tc, S->d, S->td, S->K, S->sb0, S->sb1,
                                   S->sy, S->A0, S->B0, S->A1, S->B1, S->atol, S->rtol, safety, growlim, info, S->lagc);
            }
#undef SELLARGS
            LOG("sell degree %ld of %ld tail %.3g M %.4g ymax %.6g", (long)k, (long)S->K, info[0], info[1], info[2]);
            if (k >= 0) break;
            widen_interval(S, attempt);
            cheb_coeffs(S);
        }
        if (S->perm) scatter(n, S->perm, S->sy, yout);
        else memcpy(yout, S->sy, sizeof(double) * (size_t)n);
        return yout;
    }
#endif
    double *ydst = yout;
    if (S->perm) {
        if (!z0) gather(n, S->perm, b0, S->bp0);
        if (!z1) gather(n, S->perm, b1, S->bp1);
        b0 = S->bp0; b1 = S->bp1; yout = S->yp;
    }
    double safety = 10.0, growlim = 1.25, info[4];
    for (int attempt = 0; attempt < 5; attempt++) {
        i64 k;
        /* precomputed L b (input ordering, one-shot) when it matches this request's vectors */
        int prev = !S->perm && (S->pre0 || S->pre1) && S->prez0 == z0 && S->prez1 == z1;
        const double *p0 = prev ? S->pre0 : NULL, *p1 = prev ? S->pre1 : NULL;
#define RUN3(SFX, ...) \
            (z0 ? cheb1s##SFX(n, __VA_ARGS__, S->sa, S->sb, S->d, S->td, S->K, b1, yout, S->wa, S->wb, \
                              S->atol, S->rtol, safety, growlim, info, S->lag, p1) \
             : z1 ? cheb1s##SFX(n, __VA_ARGS__, S->sa, S->sb, S->c, S->tc, S->K, b0, yout, S->wa, S->wb, \
                                S->atol, S->rtol, safety, growlim, info, S->lag, p0) \
             : cheb2s##SFX(n, __VA_ARGS__, S->sa, S->sb, S->c, S->tc, S->d, S->td, S->K, b0, b1, yout, \
                           S->wa, S->wb, S->atol, S->rtol, safety, growlim, info, S->lag, p0, p1))
        if (S->pat && S->pox) k = RUN3(_p16, S->pip, S->pox, S->dg, S->cval);
        else if (S->pat) k = RUN3(_pat, S->pip, S->pix, S->dg, S->cval);
        else if (S->ox) k = RUN3(_c16, S->ip, S->ox, S->a);
        else k = RUN3(_csr, S->ip, S->ix, S->a);
#undef RUN3
        LOG("cheb degree %ld of %ld tail %.3g M %.4g ymax %.6g", (long)k, (long)S->K, info[0], info[1], info[2]);
        if (k >= 0) {
            if (S->perm) { scatter(n, S->perm, yout, ydst); return ydst; }
            return yout;
        }
        /* diverged: fall back to the guaranteed Gershgorin interval (PSD => lo ~ 0) */
        widen_interval(S, attempt);
        cheb_coeffs(S);
    }
    if (S->perm) { scatter(n, S->perm, yout, ydst); return ydst; }
    return yout;
}

int main(int argc, char **argv)
{
    if (argc > 1 && strcmp(argv[1], "--has-v3") == 0) {
        /* used by run.sh (with the generic build) to pick the AVX2/FMA build via CPUID */
        __builtin_cpu_init();
        return (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) ? 0 : 1;
    }
    DEBUG = getenv("EXPSOLVER_DEBUG") != NULL;
    /* flush-to-zero + denormals-are-zero: only values below 2.2e-308 (far under atol) are
     * affected, and slow denormal arithmetic can never occur */
    _mm_setcsr(_mm_getcsr() | 0x8040);
    fcntl(0, 1031, 1 << 20); /* F_SETPIPE_SZ */
    fcntl(1, 1031, 1 << 20);
    write_all("READY\n", 6);
    char head[12];
    read_exact(head, 12);
    T0 = now();
    if (memcmp(head, "FAMEXP2\n", 8) != 0) _exit(2);
    uint32_t jl;
    memcpy(&jl, head + 8, 4);
    char *js = (char *)malloc((size_t)jl + 1);
    read_exact(js, jl);
    js[jl] = 0;
    int ok = 1;
    Sess S;
    memset(&S, 0, sizeof S);
    S.n = (i64)json_num(js, "n", &ok);
    S.nnz = (i64)json_num(js, "nnz", &ok);
    S.requests = (i64)json_num(js, "requests", &ok);
    S.t = json_num(js, "t", &ok);
    S.rtol = json_num(js, "rtol", &ok);
    S.atol = json_num(js, "atol", &ok);
    if (!ok || S.n < 1) _exit(2);
    i64 n = S.n, nnz = S.nnz;
    S.ip = (i64 *)xalloc(sizeof(i64) * (size_t)(n + 1));
    S.ix = (i32 *)xalloc(sizeof(i32) * (size_t)(nnz + 1));
    S.a = (double *)xalloc(sizeof(double) * (size_t)(nnz + 1));
    read_exact(S.ip, sizeof(i64) * (size_t)(n + 1));
    read_exact(S.ix, sizeof(i32) * (size_t)nnz);
    read_exact(S.a, sizeof(double) * (size_t)nnz);
    LOG("matrix read %ld %ld", (long)n, (long)nnz);
    double *b = (double *)xalloc(sizeof(double) * 2 * (size_t)n);
    double *yb = (double *)xalloc(sizeof(double) * (size_t)n);
    int have = 0;
    /* Reading the one-shot request early and fusing L b into the Gershgorin pass was measured
     * to be neutral overall (wasted gathers on paths that do not use it), so it is disabled. */
    if (0 && S.requests == 1 && S.t != 0.0) {
        /* one-shot: the request follows the matrix in the stream; read it now */
        read_exact(b, sizeof(double) * 2 * (size_t)n);
        have = 1;
        LOG("request read (early)");
    }
    prepare(&S, have ? b : NULL, have ? b + n : NULL);
    LOG("prepared mode %d", S.mode);
    for (i64 r = 0; r < S.requests; r++) {
        if (!(r == 0 && have)) {
            read_exact(b, sizeof(double) * 2 * (size_t)n);
            LOG("request read");
        }
        double *y = solve(&S, b, b + n, yb);
        write_all(y, sizeof(double) * (size_t)n);
        LOG("written");
    }
    _exit(0);
}
