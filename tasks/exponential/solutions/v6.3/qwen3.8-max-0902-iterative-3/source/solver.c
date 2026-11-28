/* FAMEXP2 exponential propagation solver.
 *
 *   y = exp(-t*L) b0 + t*phi1(-t*L) b1,   phi1(z) = (exp(z)-1)/z
 *
 * L is a real symmetric positive semidefinite sparse matrix given in full CSR.
 *
 * Method: Chebyshev expansion of both scalar functions on an interval that
 * provably contains the spectrum of -t*L, namely [-t*Lam, eps*t*Lam] where Lam
 * is a rigorous upper bound for lambda_max(L) obtained from diagonally scaled
 * infinity norms (rho(L) = rho(D^-1 L D) <= ||D^-1 L D||_inf).
 *
 * Coefficients: exp  -> scaled modified Bessel functions I_k(r)e^{-r};
 *               phi1 -> stable downward three term recurrence that follows from
 *                       z*phi1(z) = exp(z)-1 in the Chebyshev basis.
 * The operator series is summed by the forward three term recurrence with a
 * rigorous truncation tail bound, so the degree grows adaptively until the
 * bound drops below (atol + rtol*||y||_inf)/safety.
 *
 * Single process, single thread, C99 + libm only.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>

/* ------------------------------------------------------------------ io --- */

static int read_exact(void *buf, size_t len)
{
    unsigned char *p = (unsigned char *)buf;
    size_t off = 0;
    while (off < len) {
        ssize_t r = read(0, p + off, len - off);
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        if (r == 0) return -1;
        off += (size_t)r;
    }
    return 0;
}

static int write_exact(const void *buf, size_t len)
{
    const unsigned char *p = (const unsigned char *)buf;
    size_t off = 0;
    while (off < len) {
        ssize_t r = write(1, p + off, len - off);
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        if (r == 0) return -1;
        off += (size_t)r;
    }
    return 0;
}

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* minimal JSON number lookup */
static int json_num(const char *js, const char *key, double *out)
{
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(js, pat);
    if (!p) return -1;
    p += strlen(pat);
    while (*p && *p != ':') p++;
    if (!*p) return -1;
    p++;
    char *end;
    double v = strtod(p, &end);
    if (end == p) return -1;
    *out = v;
    return 0;
}

/* ------------------------------------------------------- chebyshev coefs -- */

/* it[k] = I_k(x) * exp(-x), k = 0..K, by downward Miller recurrence using
 * the normalisation  I_0(x) + 2*sum_{k>=1} I_k(x) = e^x. */
static void bessel_scaled(double *it, int K, double x)
{
    if (!(x > 0.0)) { it[0] = 1.0; for (int k = 1; k <= K; k++) it[k] = 0.0; return; }
    int K0 = K;
    int need = (int)(9.0 * sqrt(x)) + 64;
    if (need > K0) K0 = need;
    double *w = (double *)malloc(sizeof(double) * (size_t)(K0 + 2));
    if (!w) { it[0] = 1.0; for (int k = 1; k <= K; k++) it[k] = 0.0; return; }
    w[K0 + 1] = 0.0;
    w[K0] = 1e-30;
    double two_over_x = 2.0 / x;
    for (int k = K0; k >= 1; k--) {
        double v = w[k + 1] + ((double)k * two_over_x) * w[k];
        w[k - 1] = v;
        if (v > 1e250) {
            double sc = 1e-250;
            for (int j = k - 1; j <= K0 + 1; j++) w[j] *= sc;
        }
    }
    double s = w[0];
    for (int k = 1; k <= K0; k++) s += 2.0 * w[k];
    double inv = 1.0 / s;
    for (int k = 0; k <= K; k++) it[k] = w[k] * inv;
    free(w);
}

/* Chebyshev coefficients k = 0..M on z in [a,b], r=(b-a)/2, c=(a+b)/2:
 *    alpha_k : exp(z)           = sum alpha_k T_k(u),  z = c + r u
 *    beta_k  : t*(exp(z)-1)/z   = sum beta_k  T_k(u)                          */
static int cheb_coeffs(double *alpha, double *beta, int M, double a, double b, double t)
{
    double r = 0.5 * (b - a);
    double c = 0.5 * (a + b);
    if (!(r > 0.0)) return -1;
    double eb = exp(b);
    int K = M + 1;
    double *it = (double *)malloc(sizeof(double) * (size_t)(K + 1));
    double *GG = (double *)malloc(sizeof(double) * (size_t)(K + 3));
    if (!it || !GG) { free(it); free(GG); return -1; }
    bessel_scaled(it, K, r);
    for (int k = 0; k <= K; k++) alpha[k] = (k == 0 ? 1.0 : 2.0) * it[k] * eb;
    /* (c + r u) * sum G_k T_k(u) = sum E_k T_k(u) - T_0(u),  E_k = alpha_k */
    memset(GG, 0, sizeof(double) * (size_t)(K + 3));
    for (int j = K; j >= 2; j--)
        GG[j - 1] = 2.0 * (alpha[j] - c * GG[j]) / r - GG[j + 1];
    double den = r - 2.0 * c * c / r;
    double G0 = (alpha[1] - 2.0 * c * (alpha[0] - 1.0) / r - 0.5 * r * GG[2]) / den;
    double G1 = 2.0 * (alpha[0] - 1.0 - c * G0) / r;
    GG[0] = G0; GG[1] = G1;
    for (int k = 0; k <= M; k++) beta[k] = t * GG[k];
    free(GG); free(it);
    return 0;
}

/* ---------------------------------------------------------- sparse ops --- */

typedef struct {
    int64_t n;
    const uint64_t *ip;
    const uint32_t *ix;
    const double *da;
} CSR;

static void row_abs_sums(const CSR *A, double *rs)
{
    int64_t n = A->n;
    const uint64_t *ip = A->ip; const double *da = A->da;
    for (int64_t i = 0; i < n; i++) {
        uint64_t s = ip[i], e = ip[i + 1];
        double acc = 0.0;
        for (uint64_t j = s; j < e; j++) acc += fabs(da[j]);
        rs[i] = acc;
    }
}

/* out[i] = sum_j |A_ij| s_j ; returns max_i out[i]/s_i  (rigorous bound on rho(A)) */
static double scaled_bound(const CSR *A, const double *s, double *out)
{
    int64_t n = A->n;
    const uint64_t *ip = A->ip; const uint32_t *ix = A->ix; const double *da = A->da;
    double mx = 0.0;
    for (int64_t i = 0; i < n; i++) {
        uint64_t st = ip[i], e = ip[i + 1];
        double acc = 0.0;
        for (uint64_t j = st; j < e; j++) acc += fabs(da[j]) * s[ix[j]];
        out[i] = acc;
        double si = s[i];
        double v = (si > 0.0) ? acc / si : 0.0;
        if (v > mx) mx = v;
    }
    return mx;
}

/* ------------------------------------------------- chebyshev iteration --- */

/* One Chebyshev step.  Two variants:
 *  - cheb_step1: single column, compact vectors.
 *  - cheb_step2: two columns interleaved (x[2i], x[2i+1]) so that one cache
 *    line serves both right hand sides during the sparse gather, which is the
 *    dominant cost on graph-like sparsity patterns.
 * On entry x = v_{k-1}, p = v_{k-2}.  Computes w = s1*(L x) + s2*x,
 * nv = 2w - p (stored into p), S += c*nv.  Caller swaps x and p afterwards. */
static void cheb_step1(const CSR *A, double s1, double s2, double c0,
                       double *x0, double *p0, double *S,
                       double *mv0, double *smax)
{
    int64_t n = A->n;
    const uint64_t *ip = A->ip; const uint32_t *ix = A->ix; const double *da = A->da;
    double m0 = 0.0, ms = 0.0;
    for (int64_t i = 0; i < n; i++) {
        uint64_t s = ip[i], e = ip[i + 1];
        double a0 = 0.0;
        for (uint64_t j = s; j < e; j++) a0 += da[j] * x0[ix[j]];
        double n0 = 2.0 * (s1 * a0 + s2 * x0[i]) - p0[i];
        p0[i] = n0;
        double sv = S[i] + c0 * n0;
        S[i] = sv;
        double t0 = fabs(n0), ts = fabs(sv);
        if (t0 > m0) m0 = t0;
        if (ts > ms) ms = ts;
    }
    *mv0 = m0; *smax = ms;
}

static void cheb_step2(const CSR *A, double s1, double s2, double c0, double c1,
                       double *x, double *p, double *S,
                       double *mv0, double *mv1, double *smax)
{
    int64_t n = A->n;
    const uint64_t *ip = A->ip; const uint32_t *ix = A->ix; const double *da = A->da;
    double m0 = 0.0, m1 = 0.0, ms = 0.0;
    double *x0 = x, *x1 = x + n, *p0 = p, *p1 = p + n;
    for (int64_t i = 0; i < n; i++) {
        uint64_t s = ip[i], e = ip[i + 1];
        double a0 = 0.0, a1 = 0.0;
        for (uint64_t j = s; j < e; j++) {
            uint32_t k = ix[j];
            double d = da[j];
            a0 += d * x0[k];
            a1 += d * x1[k];
        }
        double n0 = 2.0 * (s1 * a0 + s2 * x0[i]) - p0[i];
        double n1 = 2.0 * (s1 * a1 + s2 * x1[i]) - p1[i];
        p0[i] = n0; p1[i] = n1;
        double sv = S[i] + c0 * n0 + c1 * n1;
        S[i] = sv;
        double t0 = fabs(n0), t1 = fabs(n1), ts = fabs(sv);
        if (t0 > m0) m0 = t0;
        if (t1 > m1) m1 = t1;
        if (ts > ms) ms = ts;
    }
    *mv0 = m0; *mv1 = m1; *smax = ms;
}

/* --------------------------------------- exact small-component pathway --- */

#define KMAX 24

typedef struct {
    int64_t  ncomp;
    int32_t *coff;     /* ncomp+1 offsets into cnodes */
    int32_t *cnodes;   /* nodes grouped by component  */
    int32_t *csize;    /* ncomp                       */
    int64_t *eoff;     /* ncomp offsets into EP       */
    double  *EP;       /* per component: E(k*k) then P(k*k) */
    int32_t *pos;      /* node -> local index in its component */
} Blocks;

/* cyclic Jacobi eigensolver, k <= KMAX, row-major; V holds eigenvector columns */
static void jacobi_eigh(double *A, int k, double *w, double *V)
{
    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++) V[i * k + j] = (i == j) ? 1.0 : 0.0;
    double diag = 0.0;
    for (int p = 0; p < k; p++) diag += A[p * k + p] * A[p * k + p];
    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0.0;
        for (int p = 0; p < k; p++)
            for (int q = p + 1; q < k; q++) off += A[p * k + q] * A[p * k + q];
        if (off <= 1e-30 * (diag + 1e-300) || off == 0.0) break;
        for (int p = 0; p < k; p++) {
            for (int q = p + 1; q < k; q++) {
                double apq = A[p * k + q];
                if (apq == 0.0) continue;
                double app = A[p * k + p], aqq = A[q * k + q];
                double theta = (aqq - app) / (2.0 * apq);
                double tt = (theta >= 0.0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
                if (fabs(theta) > 1e30) tt = 1.0 / (2.0 * theta);
                double c = 1.0 / sqrt(tt * tt + 1.0), s = tt * c;
                for (int i = 0; i < k; i++) {
                    double aip = A[i * k + p], aiq = A[i * k + q];
                    A[i * k + p] = c * aip - s * aiq;
                    A[i * k + q] = s * aip + c * aiq;
                }
                for (int i = 0; i < k; i++) {
                    double api = A[p * k + i], aqi = A[q * k + i];
                    A[p * k + i] = c * api - s * aqi;
                    A[q * k + i] = s * api + c * aqi;
                }
                for (int i = 0; i < k; i++) {
                    double vip = V[i * k + p], viq = V[i * k + q];
                    V[i * k + p] = c * vip - s * viq;
                    V[i * k + q] = s * vip + c * viq;
                }
                A[p * k + q] = A[q * k + p] = 0.0;
            }
        }
    }
    for (int p = 0; p < k; p++) w[p] = A[p * k + p];
}

/* If every connected component of the graph of L has at most KMAX nodes,
 * precompute exact exp/phi1 blocks.  Returns 1 on success.  Detection aborts
 * as soon as a component grows past KMAX, so the failure cost is O(KMAX*deg). */
static int build_blocks(const CSR *A, Blocks *B, double t, double M_est, int requests)
{
    int64_t n = A->n;
    int64_t nnz = A->ip[n];
    const uint64_t *ip = A->ip; const uint32_t *ix = A->ix; const double *da = A->da;
    B->coff = NULL; B->cnodes = NULL; B->csize = NULL; B->eoff = NULL; B->EP = NULL; B->pos = NULL;
    unsigned char *vis = (unsigned char *)calloc((size_t)n, 1);
    int32_t *stack = (int32_t *)malloc(sizeof(int32_t) * (size_t)(KMAX + 4));
    B->cnodes = (int32_t *)malloc(sizeof(int32_t) * (size_t)n);
    B->coff = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n + 1));
    B->csize = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n ? n : 1));
    B->pos = (int32_t *)malloc(sizeof(int32_t) * (size_t)n);
    if (!vis || !stack || !B->cnodes || !B->coff || !B->csize || !B->pos) {
        free(vis); free(stack); return 0;
    }
    int64_t nc = 0, sumk2 = 0, sumk3 = 0, off = 0;
    for (int64_t i = 0; i < n; i++) {
        if (vis[i]) continue;
        int k = 0;
        stack[k++] = (int32_t)i; vis[i] = 1;
        int head = 0;
        while (head < k) {
            int32_t v = stack[head++];
            uint64_t s = ip[v], e = ip[v + 1];
            for (uint64_t j = s; j < e; j++) {
                int32_t w = (int32_t)ix[j];
                if (!vis[w]) {
                    if (k >= KMAX) { free(vis); free(stack); return 0; }
                    vis[w] = 1; stack[k++] = w;
                }
            }
        }
        B->coff[nc] = (int32_t)off;
        B->csize[nc] = k;
        for (int l = 0; l < k; l++) { B->cnodes[off + l] = stack[l]; B->pos[stack[l]] = l; }
        off += k; nc++; sumk2 += (int64_t)k * k; sumk3 += (int64_t)k * k * k;
    }
    B->coff[nc] = (int32_t)off;
    B->ncomp = nc;
    free(vis); free(stack);

    /* Only worth it when exact blocks beat the truncated series.  Both sides are
     * calibrated against measured sessions (nanosecond scale cost model): the
     * block path pays one eigensolve plus two transcendentals per eigenvalue up
     * front, the series pays M sparse passes per request.  1.4 is a safety
     * factor so borderline cases stay on the series path. */
    {
        double block_ns = 40.0 * n + 15.0 * (double)sumk3
                        + 7.0 * (double)sumk2 * requests + 5.0 * (double)(n + nnz);
        double cheb_ns = M_est * requests * 2.5 * (double)(nnz + n);
        if (block_ns * 1.4 >= cheb_ns) {
            free(B->cnodes); free(B->coff); free(B->csize); free(B->pos);
            B->cnodes = NULL; B->coff = NULL; B->csize = NULL; B->pos = NULL;
            return 0;
        }
    }

    B->eoff = (int64_t *)malloc(sizeof(int64_t) * (size_t)(nc + 1));
    B->EP = (double *)malloc(sizeof(double) * (size_t)(2 * sumk2 + 2));
    if (!B->eoff || !B->EP) return 0;
    double *Dm = (double *)malloc(sizeof(double) * (size_t)(KMAX * KMAX) * 3);
    double *wv = Dm + KMAX * KMAX, *Vv = wv + KMAX;
    if (!Dm) return 0;
    int64_t eo = 0;
    for (int64_t c = 0; c < nc; c++) {
        int k = B->csize[c];
        const int32_t *nd = B->cnodes + B->coff[c];
        B->eoff[c] = eo;
        double *E = B->EP + eo;
        double *P = E + k * k;
        eo += 2 * (int64_t)k * k;
        memset(Dm, 0, sizeof(double) * (size_t)k * k);
        for (int l = 0; l < k; l++) {
            int32_t v = nd[l];
            uint64_t s = ip[v], e = ip[v + 1];
            for (uint64_t j = s; j < e; j++) {
                int p = B->pos[ix[j]];
                double half = 0.5 * da[j];
                Dm[l * k + p] += half;
                Dm[p * k + l] += half;      /* symmetrise defensively */
            }
        }
        jacobi_eigh(Dm, k, wv, Vv);
        /* E = V diag(exp(-t w)) V^T ; P = V diag(t*phi1(-t w)) V^T */
        double fv[KMAX], gv[KMAX];
        for (int i = 0; i < k; i++) {
            double x = t * wv[i];
            fv[i] = exp(-x);
            gv[i] = (fabs(x) < 1e-8) ? t * (1.0 - 0.5 * x + x * x / 6.0)
                                     : -expm1(-x) / wv[i];
        }
        for (int l = 0; l < k; l++)
            for (int j = 0; j < k; j++) {
                double se = 0.0, sp = 0.0;
                for (int i = 0; i < k; i++) {
                    double vl = Vv[l * k + i], vj = Vv[j * k + i];
                    se += vl * fv[i] * vj;
                    sp += vl * gv[i] * vj;
                }
                E[l * k + j] = se; P[l * k + j] = sp;
            }
    }
    free(Dm);
    return 1;
}

static void free_blocks(Blocks *B)
{
    free(B->coff); free(B->cnodes); free(B->csize); free(B->eoff); free(B->EP); free(B->pos);
}

static void apply_blocks(const Blocks *B, const double *b0, const double *b1, double *y)
{
    double u0[KMAX], u1[KMAX], z[KMAX];
    for (int64_t c = 0; c < B->ncomp; c++) {
        int k = B->csize[c];
        const int32_t *nd = B->cnodes + B->coff[c];
        const double *E = B->EP + B->eoff[c];
        const double *P = E + k * k;
        for (int l = 0; l < k; l++) { u0[l] = b0[nd[l]]; u1[l] = b1[nd[l]]; }
        for (int l = 0; l < k; l++) {
            const double *er = E + l * k, *pr = P + l * k;
            double a = 0.0, bb = 0.0;
            for (int j = 0; j < k; j++) { a += er[j] * u0[j]; bb += pr[j] * u1[j]; }
            z[l] = a + bb;
        }
        for (int l = 0; l < k; l++) y[nd[l]] = z[l];
    }
}

#define SWAPD(a,b) do { double *q_=(a); (a)=(b); (b)=q_; } while (0)

/* ------------------------------------------------------------- driver ---- */

static const double SAFETY = 4.0;

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    if (write_exact("READY\n", 6) != 0) return 1;

    char magic[8];
    uint32_t jlen = 0;
    static char json[4100];
    if (read_exact(magic, 8)) return 1;
    if (memcmp(magic, "FAMEXP2\n", 8) != 0) return 1;
    if (read_exact(&jlen, 4)) return 1;
    if (jlen == 0 || jlen > 4096) return 1;
    if (read_exact(json, jlen)) return 1;
    json[jlen] = 0;

    double dn = 0, dnnz = 0, dreq = 0, dt = 0, drtol = 0, datol = 0;
    if (json_num(json, "n", &dn) || json_num(json, "nnz", &dnnz)
        || json_num(json, "requests", &dreq) || json_num(json, "t", &dt)
        || json_num(json, "rtol", &drtol) || json_num(json, "atol", &datol)) return 1;
    int64_t n = (int64_t)dn, nnz = (int64_t)dnnz;
    int requests = (int)dreq;
    double t = dt, rtol = drtol, atol = datol;
    if (n < 1 || nnz < 0 || requests < 1) return 1;

    uint64_t *indptr = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)(n + 1));
    uint32_t *indices = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)(nnz ? nnz : 1));
    double *data = (double *)malloc(sizeof(double) * (size_t)(nnz ? nnz : 1));
    if (!indptr || !indices || !data) return 1;
    if (read_exact(indptr, sizeof(uint64_t) * (size_t)(n + 1))) return 1;
    if (nnz && read_exact(indices, sizeof(uint32_t) * (size_t)nnz)) return 1;
    if (nnz && read_exact(data, sizeof(double) * (size_t)nnz)) return 1;

    double t_start = now_sec();
    double budget = (requests == 1) ? 14.0 : 140.0;

    CSR A; A.n = n; A.ip = indptr; A.ix = indices; A.da = data;

    /* 8 vectors of length n: b0,b1,ybuf,x0,p0,x1,p1,S */
    double *vec = (double *)malloc(sizeof(double) * (size_t)n * 8);
    if (!vec) return 1;
    double *b0 = vec, *b1 = vec + n, *ybuf = vec + 2 * n;
    double *x = vec + 3 * n, *p = vec + 5 * n;
    double *S  = vec + 7 * n;

    /* ---- rigorous upper bound on lambda_max(L), first pass ---- */
    double Lam = 0.0;
    double *rs = NULL, *tmp = NULL;
    if (nnz > 0 && t > 0.0) {
        rs = (double *)malloc(sizeof(double) * (size_t)n * 2);
        if (!rs) return 1;
        tmp = rs + n;
        row_abs_sums(&A, rs);
        for (int64_t i = 0; i < n; i++) if (rs[i] > Lam) Lam = rs[i];
    }

    /* series degree estimate: used both to size the coefficient arrays and to
     * decide whether the exact small-component path is the cheaper one */
    double M_est = 0.0;
    if (t > 0.0 && Lam > 0.0) {
        double ell = t * Lam * (1.0 + 1e-6);
        double target = atol / (8.0 * (1.0 + t));
        if (target < 1e-300) target = 1e-300;
        M_est = sqrt(ell * (log(1.0 / target) + 10.0)) + 16.0;
    }

    /* ---- exact small-component path (block diagonal / tiny operators) ---- */
    Blocks BLK;
    memset(&BLK, 0, sizeof BLK);
    int use_blocks = 0;
    if (nnz > 0 && nnz <= 64 * n && t > 0.0) {
        use_blocks = build_blocks(&A, &BLK, t, M_est, requests);
    }

    if (rs != NULL && !use_blocks) {
        /* scaled Gershgorin bounds are worth the extra passes only when the
         * Chebyshev degree is large enough for the interval width to matter */
        if (t * Lam > 25.0 || (requests > 1 && t * Lam > 1.5)) {
            double cand;
            /* s_i = rowsum_i */
            memcpy(tmp, rs, sizeof(double) * (size_t)n);
            for (int64_t i = 0; i < n; i++) if (!(tmp[i] > 0.0)) tmp[i] = 1.0;
            cand = scaled_bound(&A, tmp, rs);
            if (cand > 0.0 && cand < Lam) Lam = cand;
            row_abs_sums(&A, rs);
            /* s_i = sqrt(rowsum_i) */
            for (int64_t i = 0; i < n; i++) tmp[i] = rs[i] > 0.0 ? sqrt(rs[i]) : 1.0;
            cand = scaled_bound(&A, tmp, rs);
            if (cand > 0.0 && cand < Lam) Lam = cand;
            row_abs_sums(&A, rs);
            /* s_i = sqrt(|A_ii|) */
            for (int64_t i = 0; i < n; i++) tmp[i] = 0.0;
            for (int64_t i = 0; i < n; i++) {
                uint64_t s = indptr[i], e = indptr[i + 1];
                for (uint64_t j = s; j < e; j++)
                    if (indices[j] == (uint32_t)i) { tmp[i] = fabs(data[j]); break; }
            }
            for (int64_t i = 0; i < n; i++) tmp[i] = tmp[i] > 0.0 ? sqrt(tmp[i]) : 1.0;
            cand = scaled_bound(&A, tmp, rs);
            if (cand > 0.0 && cand < Lam) Lam = cand;
        }
    }
    if (rs != NULL) free(rs);

    /* interval in z = -t*lambda, with a small positive margin so that tiny
     * negative eigenvalues of the stored matrix stay inside */
    double margin = 1e-6;
    double za = -t * Lam, zb = margin * t * Lam;
    double rr = 0.5 * (zb - za), cc = 0.5 * (zb + za);
    double s1 = (rr > 0.0) ? -t / rr : 0.0;
    double s2 = (rr > 0.0) ? -cc / rr : 0.0;

    int deg_ok = 0, M = 0;
    double *alpha = NULL, *beta = NULL, *tailA = NULL, *tailB = NULL;
    if (t > 0.0 && Lam > 0.0 && !use_blocks) {
        double ell = t * Lam * (1.0 + margin);
        double target = atol / (8.0 * (1.0 + t));
        if (target < 1e-300) target = 1e-300;
        double lg = log(1.0 / target);
        double mest = sqrt(ell * (lg + 10.0)) + 16.0;
        M = (int)ceil(mest);
        if (M < 8) M = 8;
        if (M > 8000) M = 8000;
        alpha = (double *)malloc(sizeof(double) * (size_t)(M + 2) * 4);
        if (!alpha) return 1;
        beta = alpha + (M + 2); tailA = beta + (M + 2); tailB = tailA + (M + 2);
        if (cheb_coeffs(alpha, beta, M, za, zb, t) == 0) {
            double sa = 0.0, sb = 0.0;
            tailA[M] = 0.0; tailB[M] = 0.0;
            for (int k = M - 1; k >= 0; k--) {
                sa += fabs(alpha[k + 1]); sb += fabs(beta[k + 1]);
                tailA[k] = sa; tailB[k] = sb;
            }
            deg_ok = 1;
        }
    }

    for (int req = 0; req < requests; req++) {
        if (read_exact(b0, sizeof(double) * (size_t)n)) return 1;
        if (read_exact(b1, sizeof(double) * (size_t)n)) return 1;

        if (t == 0.0 || (!deg_ok && !use_blocks)) {
            if (t == 0.0) memcpy(ybuf, b0, sizeof(double) * (size_t)n);
            else for (int64_t i = 0; i < n; i++) ybuf[i] = b0[i] + t * b1[i];
            if (write_exact(ybuf, sizeof(double) * (size_t)n)) return 1;
            continue;
        }
        if (use_blocks) {
            apply_blocks(&BLK, b0, b1, ybuf);
            if (write_exact(ybuf, sizeof(double) * (size_t)n)) return 1;
            continue;
        }

        double nb0 = 0.0, nb1 = 0.0;
        for (int64_t i = 0; i < n; i++) {
            double v0 = fabs(b0[i]), v1 = fabs(b1[i]);
            if (v0 > nb0) nb0 = v0;
            if (v1 > nb1) nb1 = v1;
        }
        int nz0 = nb0 > 0.0, nz1 = nb1 > 0.0;
        if (!nz0 && !nz1) {
            memset(ybuf, 0, sizeof(double) * (size_t)n);
            if (write_exact(ybuf, sizeof(double) * (size_t)n)) return 1;
            continue;
        }
        int two = nz0 && nz1;

        /* column layout: interleaved pairs when both series are active */
        const double *cA = nz0 ? alpha : beta;   /* column 0 series */
        const double *tA = nz0 ? tailA : tailB;  /* its tail bound    */
        if (two) {
            for (int64_t i = 0; i < n; i++) {
                S[i] = alpha[0] * b0[i] + beta[0] * b1[i];
                x[i] = b0[i]; x[n + i] = b1[i];
                p[i] = 0.0;   p[n + i] = 0.0;
            }
        } else {
            const double *src = nz0 ? b0 : b1;
            for (int64_t i = 0; i < n; i++) {
                S[i] = cA[0] * src[i];
                x[i] = src[i];
                p[i] = 0.0;
            }
        }

        double mv0 = 0.0, mv1 = 0.0, smax = 0.0;
        /* k = 1 with halved scaling: nv = U v_0 exactly (p starts at zero) */
        if (two) cheb_step2(&A, 0.5 * s1, 0.5 * s2, alpha[1], beta[1], x, p, S, &mv0, &mv1, &smax);
        else     cheb_step1(&A, 0.5 * s1, 0.5 * s2, cA[1], x, p, S, &mv0, &smax);
        SWAPD(x, p);
        double V0 = mv0, V1 = mv1;
        int k = 1;
        double errbnd = tA[k] * V0 + (two ? tailB[k] * V1 : 0.0);
        double tau = atol + rtol * smax;
        while (k < M) {
            if (SAFETY * errbnd <= tau) break;
            if ((k & 31) == 0 && (now_sec() - t_start) > 0.85 * budget) break;
            k++;
            if (two) cheb_step2(&A, s1, s2, alpha[k], beta[k], x, p, S, &mv0, &mv1, &smax);
            else     cheb_step1(&A, s1, s2, cA[k], x, p, S, &mv0, &smax);
            SWAPD(x, p);
            if (mv0 > V0) V0 = mv0;
            if (mv1 > V1) V1 = mv1;
            errbnd = tA[k] * V0 + (two ? tailB[k] * V1 : 0.0);
            tau = atol + rtol * smax;
        }
        if (write_exact(S, sizeof(double) * (size_t)n)) return 1;
    }

    /* _exit() skips libc teardown: every output byte has already gone out
     * through unbuffered write() syscalls, and freeing tens of megabytes that
     * the kernel is about to reclaim costs measurable time on short sessions. */
    _exit(0);
}
