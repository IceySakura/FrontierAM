/* Homogeneous self-dual embedding interior-point method for
 *   min 1/2 x'Px + q'x  s.t.  Ax + s = b, s in K
 * K = product of zero, nonnegative, second-order, exponential and PSD cones.
 * Follows the Clarabel algorithm (Goulart & Chen 2024): HSDE with quadratic
 * objective, Mehrotra predictor-corrector, Nesterov-Todd scaling for
 * symmetric cones, Dahl-Andersen primal-dual scaling (with dual-scaling
 * fallback) and third-order correction for exponential cones, quasi-definite
 * KKT system with sparse LDL', static+dynamic regularisation and iterative
 * refinement, Ruiz equilibration. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include "npzio.h"
#include "ldl.h"
#include <dlfcn.h>
#include <dirent.h>
#include <unistd.h>

/* optional optimized BLAS (scipy's bundled OpenBLAS), resolved at run time */
typedef void (*dgemm_fn)(const char *, const char *, const int *, const int *, const int *, const double *,
                         const double *, const int *, const double *, const int *, const double *, double *,
                         const int *, size_t, size_t);
typedef void (*dtrsm_fn)(const char *, const char *, const char *, const char *, const int *, const int *,
                         const double *, const double *, const int *, double *, const int *, size_t, size_t,
                         size_t, size_t);
static dgemm_fn g_dgemm = NULL;
static dtrsm_fn g_dtrsm = NULL;
static void load_blas(void) {
    if (getenv("CONIC_NOBLAS")) return;
    const char *dirs[] = {"/usr/local/lib/python3.12/site-packages/scipy.libs",
                          "/usr/lib/python3/dist-packages/scipy.libs", NULL};
    for (int d = 0; dirs[d] && !g_dgemm; d++) {
        DIR *dp = opendir(dirs[d]);
        if (!dp) continue;
        struct dirent *e;
        while ((e = readdir(dp)) && !g_dgemm) {
            const char *nm = e->d_name;
            size_t L = strlen(nm);
            if (strncmp(nm, "libscipy_openblas", 17) != 0 || strstr(nm, "64_") || L < 4 || strcmp(nm + L - 3, ".so") != 0) continue;
            char path[1024];
            snprintf(path, sizeof path, "%s/%s", dirs[d], nm);
            void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
            if (!h) continue;
            dgemm_fn g = (dgemm_fn)dlsym(h, "scipy_dgemm_");
            dtrsm_fn t = (dtrsm_fn)dlsym(h, "scipy_dtrsm_");
            if (g && t) { g_dgemm = g; g_dtrsm = t; }
        }
        closedir(dp);
    }
}

typedef int64_t I;

extern void dpotrf_(const char *, const int *, double *, const int *, int *, size_t);
extern void dgesvd_(const char *, const char *, const int *, const int *, double *, const int *,
                    double *, double *, const int *, double *, const int *, double *, const int *,
                    int *, size_t, size_t);
extern void dsyev_(const char *, const char *, const int *, double *, const int *, double *,
                   double *, const int *, int *, size_t, size_t);

enum { K_ZERO = 0, K_NN = 1, K_SOC = 2, K_EXP = 3, K_PSD = 4 };
#define SOC_DENSE_MAX 1
#define TOL 1e-8
#define TOLA 0.9e-8   /* own acceptance margin for solved */

static const char *g_out = NULL;
static int g_verbose = 0;
static double g_theta = 0.0;
static double g_eps0 = 1e-8;
static double g_refatol = 0.0;
static int g_exp_limited = 0;
static int g_adaptive_ref = 1;
static double g_dropthr = 1e20;
static int g_drop_violated = 0;    /* absolute refinement target (adaptive, from current residuals) */
static int g_pd0 = 1;
static double g_t0;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static void fail_unresolved(const char *msg) {
    if (g_verbose) fprintf(stderr, "fatal: %s\n", msg);
    if (g_out) npz_write_result(g_out, "unresolved", NULL, 0, NULL, 0, NULL, 0, 0, 0, 0);
    exit(0);
}
static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) fail_unresolved("out of memory");
    return p;
}
static void *xcalloc(size_t n, size_t s) {
    void *p = calloc(n ? n : 1, s ? s : 1);
    if (!p) fail_unresolved("out of memory");
    return p;
}
#define NEW(T, n) ((T *)xmalloc(sizeof(T) * (size_t)(n)))
#define ZNEW(T, n) ((T *)xcalloc((size_t)(n), sizeof(T)))

static double ninf(const double *v, I n) {
    double r = 0;
    for (I i = 0; i < n; i++) { double a = fabs(v[i]); if (a > r) r = a; }
    return r;
}
static double dot(const double *a, const double *b, I n) {
    double r = 0;
    for (I i = 0; i < n; i++) r += a[i] * b[i];
    return r;
}

/* ---------------------------------------------------------------- CSC utils */
static void csc_transpose(I m, I n, const I *Ap, const I *Ai, const double *Ax,
                          I **Bp_, I **Bi_, double **Bx_) {
    I nnz = Ap[n];
    I *Bp = ZNEW(I, m + 1), *Bi = NEW(I, nnz), *nx = NEW(I, m + 1);
    double *Bx = NEW(double, nnz);
    for (I k = 0; k < nnz; k++) Bp[Ai[k] + 1]++;
    for (I i = 0; i < m; i++) Bp[i + 1] += Bp[i];
    memcpy(nx, Bp, sizeof(I) * (m + 1));
    for (I j = 0; j < n; j++)
        for (I k = Ap[j]; k < Ap[j + 1]; k++) {
            I p = nx[Ai[k]]++;
            Bi[p] = j;
            Bx[p] = Ax[k];
        }
    free(nx);
    *Bp_ = Bp; *Bi_ = Bi; *Bx_ = Bx;
}
/* sorted, duplicates summed, zeros dropped */
static void csc_clean(I m, I n, const I *Ap, const I *Ai, const double *Ax,
                      I **Cp, I **Ci, double **Cx) {
    I *Tp, *Ti, *Bp, *Bi; double *Tx, *Bx;
    csc_transpose(m, n, Ap, Ai, Ax, &Tp, &Ti, &Tx);
    csc_transpose(n, m, Tp, Ti, Tx, &Bp, &Bi, &Bx);
    free(Tp); free(Ti); free(Tx);
    I w = 0;
    for (I j = 0; j < n; j++) {
        I st = Bp[j], en = Bp[j + 1], c0 = w;
        Bp[j] = w;
        for (I k = st; k < en; k++) {
            if (w > c0 && Bi[w - 1] == Bi[k]) Bx[w - 1] += Bx[k];
            else { Bi[w] = Bi[k]; Bx[w] = Bx[k]; w++; }
        }
        I w2 = c0;
        for (I k = c0; k < w; k++) if (Bx[k] != 0.0) { Bi[w2] = Bi[k]; Bx[w2] = Bx[k]; w2++; }
        w = w2;
    }
    Bp[n] = w;
    *Cp = Bp; *Ci = Bi; *Cx = Bx;
}
/* y += A x (CSC m x n) */
static void csc_mv(I n, const I *Ap, const I *Ai, const double *Ax, const double *x, double *y) {
    for (I j = 0; j < n; j++) {
        double xj = x[j];
        if (xj == 0.0) continue;
        for (I k = Ap[j]; k < Ap[j + 1]; k++) y[Ai[k]] += Ax[k] * xj;
    }
}
/* y += A' z */
static void csc_mtv(I n, const I *Ap, const I *Ai, const double *Ax, const double *z, double *y) {
    for (I j = 0; j < n; j++) {
        double acc = 0;
        for (I k = Ap[j]; k < Ap[j + 1]; k++) acc += Ax[k] * z[Ai[k]];
        y[j] += acc;
    }
}
/* y += P x, P symmetric stored as upper triangle */
static void sym_mv(I n, const I *Pp, const I *Pi, const double *Px, const double *x, double *y) {
    for (I j = 0; j < n; j++) {
        double xj = x[j], acc = 0;
        for (I k = Pp[j]; k < Pp[j + 1]; k++) {
            I i = Pi[k];
            if (i == j) acc += Px[k] * xj;
            else { y[i] += Px[k] * xj; acc += Px[k] * x[i]; }
        }
        y[j] += acc;
    }
}

/* ---------------------------------------------------------------- dense helpers (column-major) */
/* C = op(A) * op(B), all d x d */
static void mm(int d, const double *A, int ta, const double *B, int tb, double *C) {
    if (g_dgemm && d >= 24) {
        double one = 1.0, zero = 0.0;
        g_dgemm(ta ? "T" : "N", tb ? "T" : "N", &d, &d, &d, &one, A, &d, B, &d, &zero, C, &d, 1, 1);
        return;
    }
    for (int j = 0; j < d; j++)
        for (int i = 0; i < d; i++) {
            double acc = 0;
            for (int k = 0; k < d; k++) {
                double a = ta ? A[k + (size_t)i * d] : A[i + (size_t)k * d];
                double b = tb ? B[j + (size_t)k * d] : B[k + (size_t)j * d];
                acc += a * b;
            }
            C[i + (size_t)j * d] = acc;
        }
}
static void smat(int d, const double *v, double *X) {
    const double ir2 = 1.0 / sqrt(2.0);
    I p = 0;
    for (int j = 0; j < d; j++)
        for (int i = 0; i <= j; i++, p++) {
            double val = (i == j) ? v[p] : v[p] * ir2;
            X[i + (size_t)j * d] = val;
            X[j + (size_t)i * d] = val;
        }
}
static void svec(int d, const double *X, double *v) {
    const double r2 = sqrt(2.0);
    I p = 0;
    for (int j = 0; j < d; j++)
        for (int i = 0; i <= j; i++, p++)
            v[p] = (i == j) ? X[i + (size_t)j * d]
                            : r2 * 0.5 * (X[i + (size_t)j * d] + X[j + (size_t)i * d]);
}
/* min eigenvalue of symmetric d x d matrix X (destroyed) */
static double min_eig(int d, double *X, double *wk, int lwk) {
    int info = 0;
    double *w = wk;
    int lw = lwk - d;
    dsyev_("N", "L", &d, X, &d, w, wk + d, &lw, &info, 1, 1);
    if (info != 0) return -1e300;
    return w[0];
}

/* ---------------------------------------------------------------- solver structures */
typedef struct {
    int type;
    I start, len, d;
    /* SOC */
    double eta; int expanded; I ecol, pu, pv, pd1, pd2;
    double D0, alpha, beta;
    /* PSD */
    double *R, *Ri, *M, *lam, *wk; int lwk;
    I nU, *U, *bp, *bi; double *bx;   /* PSD: touched columns of A and block CSC (local rows) */
    double *At;                        /* PSD: W^{-T} A_block, len x nU column-major */
    /* EXP */
    double Hs[9], Hd[9], g[3];
} cone_t;

typedef struct {
    I n, m, N;
    I *Pp, *Pi; double *Px;           /* scaled P, upper triangle */
    I *Ap, *Ai; double *Ax;           /* scaled A, CSC */
    I *Rp, *Rj; double *Rx;           /* scaled A, CSR */
    double *q, *b, *D, *E, c;
    int nc; cone_t *K; double nu; int nexp, npsd;
    I *Kp, *Ki; double *Kx; signed char *sgn; double *regd; I *pdiag, *hpos;
    ldl_t *F;
    double *lam, *wb;                 /* per-row NT data for NN / SOC */
    int pd_scaling;
    double eps_reg;
    double *t1, *t2;                  /* N-length work */
    int maxd;                         /* max PSD order */
} S_t;

/* ---------------------------------------------------------------- exp cone helpers */
static int exp_primal_feas(const double *s) {
    if (s[2] > 0 && s[1] > 0) {
        double r = s[1] * log(s[2] / s[1]) - s[0];
        if (r > 0 && isfinite(r)) return 1;
    }
    return 0;
}
static int exp_dual_feas(const double *z) {
    if (z[2] > 0 && z[0] < 0) {
        double r = z[1] - z[0] - z[0] * log(-z[2] / z[0]);
        if (r > 0 && isfinite(r)) return 1;
    }
    return 0;
}
/* gradient and Hessian of dual barrier f*(z) = -log(r) - log(-z0) - log(z2) */
static void exp_dual_grad_hess(const double *z, double *g, double *H) {
    double l = log(-z[2] / z[0]);
    double r = -z[0] * l - z[0] + z[1];
    double c2 = 1.0 / r;
    g[0] = c2 * l - 1.0 / z[0];
    g[1] = -c2;
    g[2] = (c2 * z[0] - 1.0) / z[2];
    double rr = r * r;
    H[0] = (rr - z[0] * r + l * l * z[0] * z[0]) / (rr * z[0] * z[0]);
    H[1] = H[3] = -l / rr;
    H[4] = 1.0 / rr;
    H[2] = H[6] = (z[1] - z[0]) / (rr * z[2]);
    H[5] = H[7] = -z[0] / (rr * z[2]);
    H[8] = (rr - z[0] * r + z[0] * z[0]) / (rr * z[2] * z[2]);
}
static double wright_omega(double z) {
    double w;
    if (z < 1.0 + M_PI) {
        double zm1 = z - 1.0, p = zm1;
        w = 1.0 + 0.5 * p;
        p *= zm1; w += p / 16.0;
        p *= zm1; w -= p / 192.0;
        p *= zm1; w -= p / 3072.0;
        p *= zm1; w += 13.0 * p / 61440.0;
    } else {
        double lz = log(z), zi = 1.0 / z;
        w = z - lz;
        double qq = lz * zi;
        w += qq;
        qq *= zi; w += qq * (lz / 2.0 - 1.0);
        qq *= zi; w += qq * (lz * lz / 3.0 - 1.5 * lz + 1.0);
    }
    if (w <= 0) w = 1e-300;
    for (int it = 0; it < 20; it++) {
        double f = w + log(w) - z;
        double fp = 1.0 + 1.0 / w;
        double fpp = -1.0 / (w * w);
        double dw = f / fp;
        dw = f / (fp - 0.5 * dw * fpp);
        double wn = w - dw;
        if (wn <= 0) wn = 0.5 * w;
        if (fabs(wn - w) <= 1e-15 * fabs(w)) { w = wn; break; }
        w = wn;
    }
    return w;
}
/* gradient of the conjugate (primal) barrier at s */
static void exp_primal_grad(const double *s, double *g) {
    double om = wright_omega(1.0 - s[0] / s[1] - log(s[1] / s[2]));
    g[0] = 1.0 / ((om - 1.0) * s[1]);
    g[1] = g[0] + g[0] * log(om * s[1] / s[2]) - 1.0 / s[1];
    g[2] = om / ((1.0 - om) * s[2]);
}
static int chol3_solve(const double *H, const double *b, double *x) {
    double L00 = H[0];
    if (!(L00 > 0)) return 0;
    L00 = sqrt(L00);
    double L10 = H[1] / L00, L20 = H[2] / L00;
    double t = H[4] - L10 * L10;
    if (!(t > 0)) return 0;
    double L11 = sqrt(t);
    double L21 = (H[5] - L20 * L10) / L11;
    t = H[8] - L20 * L20 - L21 * L21;
    if (!(t > 0)) return 0;
    double L22 = sqrt(t);
    double y0 = b[0] / L00, y1 = (b[1] - L10 * y0) / L11, y2 = (b[2] - L20 * y0 - L21 * y1) / L22;
    x[2] = y2 / L22;
    x[1] = (y1 - L21 * x[2]) / L11;
    x[0] = (y0 - L10 * x[1] - L20 * x[2]) / L00;
    return 1;
}
/* eta = 0.5 * D^3 f*(z)[a, b] */
static void exp_third(const double *z, const double *a, const double *b, double *eta) {
    double z1 = z[0], z3 = z[2];
    double l = log(-z3 / z1);
    double r = z[1] - z1 - z1 * l;
    double gr[3] = {-l, 1.0, -z1 / z3};
    double Hr[9] = {1.0 / z1, 0, -1.0 / z3, 0, 0, 0, -1.0 / z3, 0, z1 / (z3 * z3)};
    double T[3];
    T[0] = -a[0] * b[0] / (z1 * z1) + a[2] * b[2] / (z3 * z3);
    T[1] = 0;
    T[2] = (a[0] * b[2] + a[2] * b[0]) / (z3 * z3) - 2.0 * z1 * a[2] * b[2] / (z3 * z3 * z3);
    double Ha[3], Hb[3];
    for (int i = 0; i < 3; i++) {
        Ha[i] = Hr[i] * a[0] + Hr[i + 3] * a[1] + Hr[i + 6] * a[2];
        Hb[i] = Hr[i] * b[0] + Hr[i + 3] * b[1] + Hr[i + 6] * b[2];
    }
    double aHb = a[0] * Hb[0] + a[1] * Hb[1] + a[2] * Hb[2];
    double ga = gr[0] * a[0] + gr[1] * a[1] + gr[2] * a[2];
    double gb = gr[0] * b[0] + gr[1] * b[1] + gr[2] * b[2];
    double r2 = r * r, r3 = r2 * r;
    for (int i = 0; i < 3; i++)
        eta[i] = -T[i] / r + aHb * gr[i] / r2 + (Ha[i] * gb + ga * Hb[i]) / r2 - 2.0 * ga * gb * gr[i] / r3;
    eta[0] += -2.0 * a[0] * b[0] / (z1 * z1 * z1);
    eta[2] += -2.0 * a[2] * b[2] / (z3 * z3 * z3);
    for (int i = 0; i < 3; i++) eta[i] *= 0.5;
}
static void exp_update_scaling(cone_t *c, const double *s, const double *z, double mu_glob, int pd) {
    exp_dual_grad_hess(z, c->g, c->Hd);
    if (pd) {
        double zt[3];
        exp_primal_grad(s, zt);
        const double *st = c->g;
        double dsz = s[0] * z[0] + s[1] * z[1] + s[2] * z[2];
        double mu = dsz / 3.0;
        double mut = (st[0] * zt[0] + st[1] * zt[1] + st[2] * zt[2]) / 3.0;
        double ds[3], dz[3];
        for (int i = 0; i < 3; i++) { ds[i] = s[i] + mu * st[i]; dz[i] = z[i] + mu * zt[i]; }
        double dsdz = ds[0] * dz[0] + ds[1] * dz[1] + ds[2] * dz[2];
        double Hzt[3];
        for (int i = 0; i < 3; i++) Hzt[i] = c->Hd[i] * zt[0] + c->Hd[i + 3] * zt[1] + c->Hd[i + 6] * zt[2];
        double de1 = mu * mut - 1.0;
        double de2 = zt[0] * Hzt[0] + zt[1] * Hzt[1] + zt[2] * Hzt[2] - 3.0 * mut * mut;
        if (fabs(de1) > sqrt(2.2e-16) && fabs(de2) > 2.2e-16 && dsz > 0 && dsdz > 0 &&
            isfinite(de1) && isfinite(de2) && isfinite(dsdz)) {
            double tmp[3];
            for (int i = 0; i < 3; i++) tmp[i] = mut * st[i] - Hzt[i];
            double fro = 0;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++) {
                    double v = c->Hd[i + 3 * j] - st[i] * st[j] / 3.0 - tmp[i] * tmp[j] / de2;
                    fro += v * v;
                }
            double t = mu * sqrt(fro);
            double ax[3] = {z[1] * zt[2] - z[2] * zt[1], z[2] * zt[0] - z[0] * zt[2], z[0] * zt[1] - z[1] * zt[0]};
            double an = sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]);
            if (an > 0 && isfinite(t)) {
                for (int i = 0; i < 3; i++) ax[i] /= an;
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        c->Hs[i + 3 * j] = s[i] * s[j] / dsz + ds[i] * ds[j] / dsdz + t * ax[i] * ax[j];
                return;
            }
        }
    }
    for (int i = 0; i < 9; i++) c->Hs[i] = mu_glob * c->Hd[i];
}

/* ---------------------------------------------------------------- SOC helpers */
static double soc_res(const double *x, I d) {
    double nr = 0;
    for (I i = 1; i < d; i++) nr += x[i] * x[i];
    return (x[0] - sqrt(nr)) * (x[0] + sqrt(nr));
}
/* W v (NT scaling of SOC) */
static void soc_W(const double *w, double eta, I d, const double *v, double *y) {
    double w1v1 = 0;
    for (I i = 1; i < d; i++) w1v1 += w[i] * v[i];
    double c = v[0] + w1v1 / (1.0 + w[0]);
    y[0] = eta * (w[0] * v[0] + w1v1);
    for (I i = 1; i < d; i++) y[i] = eta * (v[i] + c * w[i]);
}
static void soc_Winv(const double *w, double eta, I d, const double *v, double *y) {
    double w1v1 = 0;
    for (I i = 1; i < d; i++) w1v1 += w[i] * v[i];
    double c = -v[0] + w1v1 / (1.0 + w[0]);
    y[0] = (w[0] * v[0] - w1v1) / eta;
    for (I i = 1; i < d; i++) y[i] = (v[i] + c * w[i]) / eta;
}
/* max step a >= 0 with x + a y in SOC (x interior) */
static double soc_step(const double *x, const double *y, I d) {
    double a = y[0] * y[0], b = x[0] * y[0], c = x[0] * x[0];
    for (I i = 1; i < d; i++) { a -= y[i] * y[i]; b -= x[i] * y[i]; c -= x[i] * x[i]; }
    b *= 2.0;
    if (c <= 0) return 0.0;
    double best = INFINITY;
    double sc = fabs(a) + fabs(b) + fabs(c);
    if (fabs(a) <= 1e-14 * sc) {
        if (b < 0) best = -c / b;
    } else {
        double disc = b * b - 4 * a * c;
        if (disc >= 0) {
            double sq = sqrt(disc);
            double qq = -0.5 * (b + (b >= 0 ? sq : -sq));
            double r1 = qq / a, r2 = (qq != 0) ? c / qq : INFINITY;
            if (r1 > 0 && r1 < best) best = r1;
            if (r2 > 0 && r2 < best) best = r2;
        }
    }
    /* also keep x0 + a y0 >= 0 */
    if (y[0] < 0) { double t = -x[0] / y[0]; if (t < best) best = t; }
    return best;
}

/* ---------------------------------------------------------------- cone operations */
static void cone_unit_init(S_t *S, double *s, double *z) {
    for (int k = 0; k < S->nc; k++) {
        cone_t *c = &S->K[k];
        double *sc = s + c->start, *zc = z + c->start;
        switch (c->type) {
        case K_ZERO: for (I i = 0; i < c->len; i++) sc[i] = zc[i] = 0; break;
        case K_NN: for (I i = 0; i < c->len; i++) sc[i] = zc[i] = 1; break;
        case K_SOC: for (I i = 0; i < c->len; i++) sc[i] = zc[i] = 0; sc[0] = zc[0] = 1; break;
        case K_EXP:
            sc[0] = zc[0] = -1.051383945322714;
            sc[1] = zc[1] = 0.556409619469370;
            sc[2] = zc[2] = 1.258967884768947;
            break;
        case K_PSD: {
            I p = 0;
            for (I j = 0; j < c->d; j++) for (I i = 0; i <= j; i++, p++) sc[p] = zc[p] = (i == j);
        } break;
        }
    }
}
/* shift a vector into the interior of the (self-dual) cones; per cone */
static void cone_shift(S_t *S, double *v) {
    for (int k = 0; k < S->nc; k++) {
        cone_t *c = &S->K[k];
        double *vc = v + c->start;
        switch (c->type) {
        case K_ZERO: for (I i = 0; i < c->len; i++) vc[i] = 0; break;
        case K_NN: {
            double mn = INFINITY;
            for (I i = 0; i < c->len; i++) if (vc[i] < mn) mn = vc[i];
            if (mn < 1e-8) for (I i = 0; i < c->len; i++) vc[i] += 1.0 - mn;
        } break;
        case K_SOC: {
            double nr = 0;
            for (I i = 1; i < c->len; i++) nr += vc[i] * vc[i];
            double mn = vc[0] - sqrt(nr);
            if (mn < 1e-8) vc[0] += 1.0 - mn;
        } break;
        case K_PSD: {
            int d = (int)c->d;
            smat(d, vc, c->M);
            double mn = min_eig(d, c->M, c->wk, c->lwk);
            if (mn < 1e-8) {
                I p = 0;
                for (I j = 0; j < d; j++) for (I i = 0; i <= j; i++, p++) if (i == j) vc[p] += 1.0 - mn;
            }
        } break;
        default: break;
        }
    }
}
/* returns 0 on failure */
static int cone_update_scaling(S_t *S, const double *s, const double *z, double mu) {
    for (int k = 0; k < S->nc; k++) {
        cone_t *c = &S->K[k];
        const double *sc = s + c->start, *zc = z + c->start;
        double *lam = S->lam + c->start, *wb = S->wb + c->start;
        switch (c->type) {
        case K_ZERO: break;
        case K_NN:
            for (I i = 0; i < c->len; i++) {
                if (!(sc[i] > 0 && zc[i] > 0)) return 0;
                wb[i] = sqrt(sc[i] / zc[i]);
                lam[i] = sqrt(sc[i] * zc[i]);
            }
            break;
        case K_SOC: {
            I d = c->len;
            double ds = soc_res(sc, d), dz = soc_res(zc, d);
            if (!(ds > 0 && dz > 0 && sc[0] > 0 && zc[0] > 0)) return 0;
            double sqs = sqrt(ds), sqz = sqrt(dz);
            double sz = 0;
            for (I i = 0; i < d; i++) sz += sc[i] * zc[i];
            sz /= (sqs * sqz);
            double gam = sqrt(0.5 * (1.0 + sz));
            wb[0] = (sc[0] / sqs + zc[0] / sqz) / (2 * gam);
            for (I i = 1; i < d; i++) wb[i] = (sc[i] / sqs - zc[i] / sqz) / (2 * gam);
            /* renormalise w so that det(w)=1 */
            double w1 = 0;
            for (I i = 1; i < d; i++) w1 += wb[i] * wb[i];
            wb[0] = sqrt(1.0 + w1);
            c->eta = sqrt(sqs / sqz);
            soc_W(wb, c->eta, d, zc, lam);
            if (c->expanded) {
                double del = w1;                       /* ||w1||^2 = w0^2 - 1 */
                double L = 2.0 / (2.0 * del + 1.0);
                double t = (del > 0) ? 1.0 / (2.0 * del * (2.0 * del + 1.0)) : 1.0;
                if (t > 1.0) t = 1.0;
                double b2 = L + t;
                c->beta = sqrt(b2);
                c->alpha = sqrt(2.0 + b2);
                c->D0 = (2.0 * del + 1.0) * t / (2.0 + b2);
            }
        } break;
        case K_EXP:
            if (!exp_primal_feas(sc) || !exp_dual_feas(zc)) return 0;
            exp_update_scaling(c, sc, zc, mu, S->pd_scaling);
            break;
        case K_PSD: {
            int d = (int)c->d, info = 0;
            double *Ls = c->wk, *Lz = c->wk + d * d, *Mt = c->wk + 2 * d * d, *U = c->wk + 3 * d * d,
                   *VT = c->wk + 4 * d * d, *sv = c->wk + 5 * d * d, *work = c->wk + 5 * d * d + d;
            int lw = c->lwk - 5 * d * d - d;
            smat(d, sc, Ls);
            smat(d, zc, Lz);
            dpotrf_("L", &d, Ls, &d, &info, 1);
            if (info != 0) return 0;
            dpotrf_("L", &d, Lz, &d, &info, 1);
            if (info != 0) return 0;
            for (int j = 0; j < d; j++) for (int i = 0; i < j; i++) { Ls[i + j * d] = 0; Lz[i + j * d] = 0; }
            mm(d, Lz, 1, Ls, 0, Mt); /* Lz' Ls */
            dgesvd_("A", "A", &d, &d, Mt, &d, sv, U, &d, VT, &d, work, &lw, &info, 1, 1);
            if (info != 0) return 0;
            for (int i = 0; i < d; i++) if (!(sv[i] > 0)) return 0;
            /* R = Ls V diag(sv^-1/2), Ri = diag(sv^-1/2) U' Lz' */
            for (int j = 0; j < d; j++) {
                double f = 1.0 / sqrt(sv[j]);
                for (int i = 0; i < d; i++) {
                    double acc = 0;
                    for (int k2 = 0; k2 < d; k2++) acc += Ls[i + k2 * d] * VT[j + k2 * d];
                    c->R[i + j * d] = acc * f;
                }
            }
            for (int j = 0; j < d; j++)
                for (int i = 0; i < d; i++) {
                    double acc = 0;
                    for (int k2 = 0; k2 < d; k2++) acc += U[k2 + i * d] * Lz[j + k2 * d];
                    c->Ri[i + j * d] = acc / sqrt(sv[i]);
                }
            for (int i = 0; i < d; i++) c->lam[i] = sv[i];
            mm(d, c->R, 0, c->R, 1, c->M);
        } break;
        }
    }
    return 1;
}
/* y = H v over all cones (rows of z) */
static void cone_Hmul(S_t *S, const double *v, double *y) {
    for (int k = 0; k < S->nc; k++) {
        cone_t *c = &S->K[k];
        const double *vc = v + c->start;
        double *yc = y + c->start, *wb = S->wb + c->start;
        switch (c->type) {
        case K_ZERO: for (I i = 0; i < c->len; i++) yc[i] = 0; break;
        case K_NN: for (I i = 0; i < c->len; i++) yc[i] = wb[i] * wb[i] * vc[i]; break;
        case K_SOC: {
            double e2 = c->eta * c->eta, wv = dot(wb, vc, c->len);
            yc[0] = e2 * (2 * wv * wb[0] - vc[0]);
            for (I i = 1; i < c->len; i++) yc[i] = e2 * (2 * wv * wb[i] + vc[i]);
        } break;
        case K_EXP:
            for (int i = 0; i < 3; i++) yc[i] = c->Hs[i] * vc[0] + c->Hs[i + 3] * vc[1] + c->Hs[i + 6] * vc[2];
            break;
        case K_PSD: {
            int d = (int)c->d;
            double *X = c->wk, *T = c->wk + d * d, *Y = c->wk + 2 * d * d;
            smat(d, vc, X);
            mm(d, c->M, 0, X, 0, T);
            mm(d, T, 0, c->M, 0, Y);
            svec(d, Y, yc);
        } break;
        }
    }
}
/* fill -H - eps into KKT */
static void kkt_fill_H(S_t *S) {
    double eps = S->eps_reg, *Kx = S->Kx;
    for (int k = 0; k < S->nc; k++) {
        cone_t *c = &S->K[k];
        I st = c->start;
        double *wb = S->wb + st;
        switch (c->type) {
        case K_ZERO: for (I i = 0; i < c->len; i++) Kx[S->hpos[st + i]] = -eps; break;
        case K_NN: for (I i = 0; i < c->len; i++) Kx[S->hpos[st + i]] = -wb[i] * wb[i] - eps; break;
        case K_SOC: {
            double e2 = c->eta * c->eta;
            if (!c->expanded) {
                for (I i = 0; i < c->len; i++) {
                    I p = S->hpos[st + i];
                    for (I j = 0; j <= i; j++) {
                        double h = 2.0 * wb[i] * wb[j] * e2;
                        if (i == j) h += (i == 0 ? -e2 : e2);
                        Kx[p + j] = -h - (i == j ? eps : 0.0);
                    }
                }
            } else {
                Kx[S->hpos[st]] = -e2 * c->D0 - eps;
                for (I i = 1; i < c->len; i++) Kx[S->hpos[st + i]] = -e2 - eps;
                double eta = c->eta;
                Kx[c->pu] = eta * 2.0 * wb[0] / c->alpha;
                Kx[c->pv] = 0.0;
                for (I i = 1; i < c->len; i++) {
                    Kx[c->pu + i] = eta * c->alpha * wb[i];
                    Kx[c->pv + i] = eta * c->beta * wb[i];
                }
                Kx[c->pd1] = 1.0 + eps;
                Kx[c->pd2] = -1.0 - eps;
            }
        } break;
        case K_EXP:
            for (I i = 0; i < 3; i++) {
                I p = S->hpos[st + i];
                for (I j = 0; j <= i; j++) Kx[p + j] = -c->Hs[j + 3 * i] - (i == j ? eps : 0.0);
            }
            break;
        case K_PSD: {
            int d = (int)c->d;
            I len = c->len;
            double *X = c->wk, *T = c->wk + d * d, *Y = c->wk + 2 * d * d, *v = c->wk + 3 * d * d;
            for (I u = 0; u < c->nU; u++) {
                I nzc = c->bp[u + 1] - c->bp[u];
                if (4 * nzc > d) {
                    for (I i = 0; i < len; i++) v[i] = 0;
                    for (I k2 = c->bp[u]; k2 < c->bp[u + 1]; k2++) v[c->bi[k2]] = c->bx[k2];
                    smat(d, v, X);
                    mm(d, c->Ri, 0, X, 0, T);
                    mm(d, T, 0, c->Ri, 1, Y);
                    svec(d, Y, c->At + u * len);
                } else {
                    /* sparse congruence: Y = sum_k x_k Ri E_k Ri' (upper triangle) */
                    double *out = c->At + u * len;
                    for (I i = 0; i < len; i++) out[i] = 0;
                    for (I k2 = c->bp[u]; k2 < c->bp[u + 1]; k2++) {
                        I li = c->bi[k2];
                        int jj = (int)((sqrt(8.0 * li + 1.0) - 1.0) / 2.0);
                        while ((I)jj * (jj + 1) / 2 > li) jj--;
                        while ((I)(jj + 1) * (jj + 2) / 2 <= li) jj++;
                        int ii = (int)(li - (I)jj * (jj + 1) / 2);
                        const double *ra = c->Ri + (size_t)ii * d, *rb = c->Ri + (size_t)jj * d;
                        double xv = c->bx[k2];
                        /* E = e_ii e_ii' (diag) or (e_ii e_jj' + e_jj e_ii')/sqrt2 ; Ri E Ri' entries (p,q) */
                        I pos = 0;
                        if (ii == jj) {
                            for (int qq = 0; qq < d; qq++) {
                                double f = xv * ra[qq];
                                for (int pp = 0; pp < qq; pp++, pos++) out[pos] += sqrt(2.0) * f * ra[pp];
                                out[pos++] += f * ra[qq];
                            }
                        } else {
                            for (int qq = 0; qq < d; qq++) {
                                double fa = xv * ra[qq], fb = xv * rb[qq];
                                for (int pp = 0; pp < qq; pp++, pos++) out[pos] += (fb * ra[pp] + fa * rb[pp]);
                                out[pos++] += sqrt(2.0) * fa * rb[qq];
                            }
                        }
                    }
                }
            }
            for (I ii = 0; ii < len; ii++) {
                I p = S->Kp[S->n + st + ii];
                for (I u = 0; u < c->nU; u++) Kx[p + u] = c->At[u * len + ii];
                Kx[S->hpos[st + ii]] = -1.0 - eps;
            }
        } break;
        }
    }
}

/* combined-step d_s; for affine step pass sigmu=0 and ds_a=dz_a=NULL (gives d_s = s) */
static void cone_ds(S_t *S, const double *s, const double *z, const double *ds_a, const double *dz_a,
                    double sigmu, double *out) {
    for (int k = 0; k < S->nc; k++) {
        cone_t *c = &S->K[k];
        I st = c->start, d = c->len;
        const double *sc = s + st, *zc = z + st;
        double *o = out + st, *lam = S->lam + st, *wb = S->wb + st;
        if (!ds_a) {
            for (I i = 0; i < d; i++) o[i] = (c->type == K_ZERO) ? 0.0 : sc[i];
            continue;
        }
        const double *dsa = ds_a + st, *dza = dz_a + st;
        switch (c->type) {
        case K_ZERO: for (I i = 0; i < d; i++) o[i] = 0; break;
        case K_NN:
            for (I i = 0; i < d; i++) o[i] = (sc[i] * zc[i] + dsa[i] * dza[i] - sigmu) / zc[i];
            break;
        case K_SOC: {
            double *a = S->t1, *bb = S->t2;     /* scratch, length >= d */
            soc_Winv(wb, c->eta, d, dsa, a);
            soc_W(wb, c->eta, d, dza, bb);
            /* V = lam o lam + a o bb - sigmu e */
            double *V = S->t1 + d, *U = S->t2 + d;
            V[0] = dot(lam, lam, d) + dot(a, bb, d) - sigmu;
            for (I i = 1; i < d; i++) V[i] = 2 * lam[0] * lam[i] + a[0] * bb[i] + bb[0] * a[i];
            /* U = lam \ V */
            double l1v1 = 0, l1l1 = 0;
            for (I i = 1; i < d; i++) { l1v1 += lam[i] * V[i]; l1l1 += lam[i] * lam[i]; }
            double det = (lam[0] - sqrt(l1l1)) * (lam[0] + sqrt(l1l1));
            U[0] = (lam[0] * V[0] - l1v1) / det;
            for (I i = 1; i < d; i++) U[i] = (V[i] - U[0] * lam[i]) / lam[0];
            soc_W(wb, c->eta, d, U, o);
        } break;
        case K_EXP: {
            double u[3], eta[3] = {0, 0, 0};
            if (chol3_solve(c->Hd, dsa, u)) exp_third(zc, dza, u, eta);
            for (int i = 0; i < 3; i++) {
                double v = sc[i] + sigmu * c->g[i] - eta[i];
                o[i] = isfinite(v) ? v : sc[i] + sigmu * c->g[i];
            }
        } break;
        case K_PSD: {
            int dd = (int)c->d;
            double *X = c->wk, *T = c->wk + dd * dd, *A1 = c->wk + 2 * dd * dd, *B1 = c->wk + 3 * dd * dd,
                   *V = c->wk + 4 * dd * dd;
            smat(dd, dsa, X);
            mm(dd, c->Ri, 0, X, 0, T);
            mm(dd, T, 0, c->Ri, 1, A1);
            smat(dd, dza, X);
            mm(dd, c->R, 1, X, 0, T);
            mm(dd, T, 0, c->R, 0, B1);
            mm(dd, A1, 0, B1, 0, X);
            for (int j = 0; j < dd; j++)
                for (int i = 0; i < dd; i++) {
                    double v = 0.5 * (X[i + j * dd] + X[j + i * dd]);
                    if (i == j) v += c->lam[i] * c->lam[i] - sigmu;
                    V[i + j * dd] = 2.0 * v / (c->lam[i] + c->lam[j]);
                }
            mm(dd, c->R, 0, V, 0, T);
            mm(dd, T, 0, c->R, 1, X);
            svec(dd, X, o);
        } break;
        }
    }
}
/* max step for s+a*ds in K, z+a*dz in K* (exp: backtracking), capped at amax */
static double cone_step(S_t *S, const double *s, const double *z, const double *ds, const double *dz, double amax) {
    double a = amax;
    for (int k = 0; k < S->nc; k++) {
        cone_t *c = &S->K[k];
        I st = c->start, d = c->len;
        const double *sc = s + st, *zc = z + st, *dsc = ds + st, *dzc = dz + st;
        switch (c->type) {
        case K_NN:
            for (I i = 0; i < d; i++) {
                if (dsc[i] < 0) { double t = -sc[i] / dsc[i]; if (t < a) a = t; }
                if (dzc[i] < 0) { double t = -zc[i] / dzc[i]; if (t < a) a = t; }
            }
            break;
        case K_SOC: {
            double t = soc_step(sc, dsc, d); if (t < a) a = t;
            t = soc_step(zc, dzc, d); if (t < a) a = t;
        } break;
        case K_PSD: {
            int dd = (int)c->d;
            double *X = c->wk, *T = c->wk + dd * dd, *A1 = c->wk + 2 * dd * dd;
            double *ew = c->wk + 3 * dd * dd;
            int lw = c->lwk - 3 * dd * dd;
            for (int pass = 0; pass < 2; pass++) {
                if (pass == 0) {
                    smat(dd, dsc, X);
                    mm(dd, c->Ri, 0, X, 0, T);
                    mm(dd, T, 0, c->Ri, 1, A1);
                } else {
                    smat(dd, dzc, X);
                    mm(dd, c->R, 1, X, 0, T);
                    mm(dd, T, 0, c->R, 0, A1);
                }
                for (int j = 0; j < dd; j++)
                    for (int i = 0; i < dd; i++)
                        A1[i + j * dd] /= sqrt(c->lam[i] * c->lam[j]);
                double mn = min_eig(dd, A1, ew, lw);
                if (mn < 0) { double t = -1.0 / mn; if (t < a) a = t; }
            }
        } break;
        default: break;
        }
    }
    if (S->nexp) {
        if (a > 1.0) a = 1.0;
        double a_sym = a;
        for (int it = 0; it < 200; it++) {
            int ok = 1;
            for (int k = 0; k < S->nc && ok; k++) {
                cone_t *c = &S->K[k];
                if (c->type != K_EXP) continue;
                I st = c->start;
                double ts[3], tz[3];
                for (int i = 0; i < 3; i++) { ts[i] = s[st + i] + a * ds[st + i]; tz[i] = z[st + i] + a * dz[st + i]; }
                if (!exp_primal_feas(ts) || !exp_dual_feas(tz)) ok = 0;
            }
            if (ok) break;
            a *= 0.8;
            if (a < 1e-12) { a = 0; break; }
        }
        if (g_verbose > 3) fprintf(stderr, "      step: sym=%.3e exp=%.3e\n", a_sym, a);
        g_exp_limited = (a < a_sym);
    }
    return a;
}

/* ---------------------------------------------------------------- KKT */
static void kkt_build(S_t *S) {
    I n = S->n, m = S->m;
    I N = n + m;
    for (int k = 0; k < S->nc; k++) {
        cone_t *c = &S->K[k];
        c->expanded = 0;
        if (c->type == K_SOC && c->len > SOC_DENSE_MAX) { c->expanded = 1; c->ecol = N; N += 2; }
    }
    S->N = N;
    I nnz = 0;
    for (I j = 0; j < n; j++) {
        for (I k = S->Pp[j]; k < S->Pp[j + 1]; k++) if (S->Pi[k] < j) nnz++;
        nnz++;
    }
    for (int k = 0; k < S->nc; k++) {
        cone_t *c = &S->K[k];
        if (c->type == K_PSD) {
            /* union of columns touched by the block rows, and block CSC */
            I *mark = ZNEW(I, n > 0 ? n : 1), cnt = 0;
            for (I i = c->start; i < c->start + c->len; i++)
                for (I k2 = S->Rp[i]; k2 < S->Rp[i + 1]; k2++)
                    if (!mark[S->Rj[k2]]) { mark[S->Rj[k2]] = 1; cnt++; }
            c->nU = cnt; c->U = NEW(I, cnt + 1);
            I *pos = NEW(I, n > 0 ? n : 1);
            cnt = 0;
            for (I j = 0; j < n; j++) if (mark[j]) { pos[j] = cnt; c->U[cnt++] = j; }
            c->bp = ZNEW(I, c->nU + 1);
            I nb = 0;
            for (I i = c->start; i < c->start + c->len; i++)
                for (I k2 = S->Rp[i]; k2 < S->Rp[i + 1]; k2++) { c->bp[pos[S->Rj[k2]] + 1]++; nb++; }
            for (I u = 0; u < c->nU; u++) c->bp[u + 1] += c->bp[u];
            c->bi = NEW(I, nb + 1); c->bx = NEW(double, nb + 1);
            I *nx = NEW(I, c->nU + 1);
            memcpy(nx, c->bp, sizeof(I) * (c->nU + 1));
            for (I i = c->start; i < c->start + c->len; i++)
                for (I k2 = S->Rp[i]; k2 < S->Rp[i + 1]; k2++) {
                    I u = pos[S->Rj[k2]], q = nx[u]++;
                    c->bi[q] = i - c->start; c->bx[q] = S->Rx[k2];
                }
            free(nx); free(pos); free(mark);
            c->At = NEW(double, c->len * c->nU + 1);
            nnz += c->len * (c->nU + 1);
            continue;
        }
        for (I i = c->start; i < c->start + c->len; i++) nnz += S->Rp[i + 1] - S->Rp[i];
        int dense = (c->type == K_EXP || (c->type == K_SOC && !c->expanded));
        if (dense) nnz += c->len * (c->len + 1) / 2;
        else nnz += c->len;
        if (c->expanded) nnz += 2 * (c->len + 1);
    }
    S->Kp = NEW(I, N + 1); S->Ki = NEW(I, nnz); S->Kx = ZNEW(double, nnz);
    S->sgn = NEW(signed char, N); S->regd = NEW(double, N);
    S->pdiag = NEW(I, n > 0 ? n : 1); S->hpos = NEW(I, m > 0 ? m : 1);
    I p = 0;
    for (I j = 0; j < n; j++) {
        S->Kp[j] = p;
        double dg = 0;
        for (I k = S->Pp[j]; k < S->Pp[j + 1]; k++) {
            if (S->Pi[k] < j) { S->Ki[p] = S->Pi[k]; S->Kx[p] = S->Px[k]; p++; }
            else if (S->Pi[k] == j) dg = S->Px[k];
        }
        S->pdiag[j] = p; S->Ki[p] = j; S->Kx[p] = dg; p++;
        S->sgn[j] = 1;
    }
    for (int k = 0; k < S->nc; k++) {
        cone_t *c = &S->K[k];
        int dense = (c->type == K_EXP || (c->type == K_SOC && !c->expanded));
        if (c->type == K_PSD) {
            for (I ii = 0; ii < c->len; ii++) {
                I i = c->start + ii;
                S->Kp[n + i] = p;
                for (I u = 0; u < c->nU; u++) S->Ki[p++] = c->U[u];
                S->hpos[i] = p;
                S->Ki[p++] = n + i;
                S->sgn[n + i] = -1;
            }
            continue;
        }
        for (I ii = 0; ii < c->len; ii++) {
            I i = c->start + ii;
            S->Kp[n + i] = p;
            for (I k2 = S->Rp[i]; k2 < S->Rp[i + 1]; k2++) { S->Ki[p] = S->Rj[k2]; S->Kx[p] = S->Rx[k2]; p++; }
            S->hpos[i] = p;
            if (dense) for (I jj = 0; jj <= ii; jj++) S->Ki[p++] = n + c->start + jj;
            else S->Ki[p++] = n + i;
            S->sgn[n + i] = -1;
        }
    }
    for (int k = 0; k < S->nc; k++) {
        cone_t *c = &S->K[k];
        if (!c->expanded) continue;
        S->Kp[c->ecol] = p; c->pu = p;
        for (I ii = 0; ii < c->len; ii++) S->Ki[p++] = n + c->start + ii;
        c->pd1 = p; S->Ki[p++] = c->ecol;
        S->Kp[c->ecol + 1] = p; c->pv = p;
        for (I ii = 0; ii < c->len; ii++) S->Ki[p++] = n + c->start + ii;
        c->pd2 = p; S->Ki[p++] = c->ecol + 1;
        S->sgn[c->ecol] = 1; S->sgn[c->ecol + 1] = -1;
    }
    S->Kp[N] = p;
}
static void kkt_set_reg(S_t *S, double eps) {
    double old = S->eps_reg;
    S->eps_reg = eps;
    for (I j = 0; j < S->n; j++) S->Kx[S->pdiag[j]] += eps - old;
    for (I j = 0; j < S->N; j++) S->regd[j] = S->sgn[j] * eps;
}
static void kkt_mv(S_t *S, const double *v, double *y) {
    I N = S->N;
    memset(y, 0, sizeof(double) * N);
    const I *Kp = S->Kp, *Ki = S->Ki; const double *Kx = S->Kx;
    for (I j = 0; j < N; j++) {
        double vj = v[j], acc = 0;
        for (I k = Kp[j]; k < Kp[j + 1]; k++) {
            I i = Ki[k];
            if (i == j) acc += Kx[k] * vj;
            else { y[i] += Kx[k] * vj; acc += Kx[k] * v[i]; }
        }
        y[j] += acc;
    }
    for (I j = 0; j < N; j++) y[j] -= S->regd[j] * v[j];
}
/* solve true KKT system: rhs (N) -> sol (N) */
static void kkt_solve(S_t *S, const double *rhs, double *sol) {
    I N = S->N;
    double *r = S->t1, *dx = S->t2;
    memcpy(sol, rhs, sizeof(double) * N);
    ldl_solve(S->F, sol);
    double bn = ninf(rhs, N);
    double prev = INFINITY;
    for (int it = 0; it < 10; it++) {
        kkt_mv(S, sol, r);
        for (I i = 0; i < N; i++) r[i] = rhs[i] - r[i];
        double rn = ninf(r, N);
        if (!(rn > fmax(g_refatol, 1e-12 + 1e-13 * bn))) break;
        if (it > 0 && rn > prev / 5.0) {
            if (rn > prev) { for (I i = 0; i < N; i++) sol[i] -= dx[i]; }
            break;
        }
        prev = rn;
        if (g_verbose > 2) fprintf(stderr, "      refine it=%d rn=%.2e bn=%.2e\n", it, rn, bn);
        memcpy(dx, r, sizeof(double) * N);
        ldl_solve(S->F, dx);
        for (I i = 0; i < N; i++) sol[i] += dx[i];
    }
}
/* PSD blocks use the scaled form: rows W^{-T}A, z~ = W z */
static void psd_apply(cone_t *c, const double *v, double *out, int inv) {
    int d = (int)c->d;
    double *X = c->wk, *T = c->wk + d * d, *Y = c->wk + 2 * d * d;
    smat(d, v, X);
    if (!inv) { mm(d, c->Ri, 0, X, 0, T); mm(d, T, 0, c->Ri, 1, Y); }   /* W^{-T} v = Ri V Ri' */
    else { mm(d, c->Ri, 1, X, 0, T); mm(d, T, 0, c->Ri, 0, Y); }        /* W^{-1} v = Ri' V Ri */
    svec(d, Y, out);
}
static void kkt_solve_T(S_t *S, double *rhs, double *sol) {
    if (S->npsd) {
        for (int k = 0; k < S->nc; k++) {
            cone_t *c = &S->K[k];
            if (c->type != K_PSD) continue;
            double *r = rhs + S->n + c->start;
            psd_apply(c, r, c->wk + 4 * c->d * c->d, 0);
            memcpy(r, c->wk + 4 * c->d * c->d, sizeof(double) * c->len);
        }
    }
    if (S->npsd && S->N == S->n + S->m) { memcpy(sol, rhs, sizeof(double) * S->N); ldl_solve(S->F, sol); }   /* outer loop refines */
    else kkt_solve(S, rhs, sol);
    if (S->npsd) {
        for (int k = 0; k < S->nc; k++) {
            cone_t *c = &S->K[k];
            if (c->type != K_PSD) continue;
            double *zz = sol + S->n + c->start;
            psd_apply(c, zz, c->wk + 4 * c->d * c->d, 1);
            memcpy(zz, c->wk + 4 * c->d * c->d, sizeof(double) * c->len);
        }
    }
}
static void cone_Hmul(S_t *S, const double *v, double *y);
/* reduced original system y = [P A'; A -H] v (x,z parts only) */
static void kred_mv(S_t *S, const double *v, double *y, double *Hw) {
    I n = S->n, m = S->m;
    memset(y, 0, sizeof(double) * (n + m));
    sym_mv(n, S->Pp, S->Pi, S->Px, v, y);
    csc_mtv(n, S->Ap, S->Ai, S->Ax, v + n, y);
    csc_mv(n, S->Ap, S->Ai, S->Ax, v, y + n);
    cone_Hmul(S, v + n, Hw);
    for (I i = 0; i < m; i++) y[n + i] -= Hw[i];
}
static double *g_ref_r = NULL, *g_ref_d = NULL, *g_ref_y = NULL, *g_ref_h = NULL, *g_ref_b = NULL;
static void kkt_solve_xz(S_t *S, double *rhs, double *sol) {
    I N = S->N, nm = S->n + S->m;
    if (!S->npsd) { kkt_solve_T(S, rhs, sol); return; }
    if (!g_ref_r) {
        g_ref_r = NEW(double, N); g_ref_d = NEW(double, N); g_ref_y = NEW(double, N);
        g_ref_h = NEW(double, S->m + 1); g_ref_b = NEW(double, N);
    }
    memcpy(g_ref_b, rhs, sizeof(double) * N);
    kkt_solve_T(S, rhs, sol);
    double bn = ninf(g_ref_b, nm), prev = INFINITY;
    for (int it = 0; it < 12; it++) {
        kred_mv(S, sol, g_ref_y, g_ref_h);
        for (I i = 0; i < nm; i++) g_ref_r[i] = g_ref_b[i] - g_ref_y[i];
        for (I i = nm; i < N; i++) g_ref_r[i] = 0;
        double rn = ninf(g_ref_r, nm);
        if (g_verbose > 2) fprintf(stderr, "      outer it=%d rn=%.2e bn=%.2e\n", it, rn, bn);
        if (!(rn > 1e-13 * bn + 1e-14)) break;
        if (it > 0 && rn > prev * 0.5) {
            if (rn > prev) for (I i = 0; i < N; i++) sol[i] -= g_ref_d[i];
            break;
        }
        prev = rn;
        kkt_solve_T(S, g_ref_r, g_ref_d);
        for (I i = 0; i < N; i++) sol[i] += g_ref_d[i];
    }
}
static void cone_identity_scaling(S_t *S) {
    for (int k = 0; k < S->nc; k++) {
        cone_t *c = &S->K[k];
        double *wb = S->wb + c->start;
        switch (c->type) {
        case K_NN: for (I i = 0; i < c->len; i++) wb[i] = 1.0; break;
        case K_SOC:
            for (I i = 0; i < c->len; i++) wb[i] = 0.0;
            wb[0] = 1.0; c->eta = 1.0;
            c->beta = sqrt(3.0); c->alpha = sqrt(5.0); c->D0 = 0.2;
            break;
        case K_EXP:
            for (int i = 0; i < 9; i++) c->Hs[i] = (i % 4 == 0);
            break;
        case K_PSD: {
            int d = (int)c->d;
            for (int i = 0; i < d * d; i++) c->R[i] = c->Ri[i] = c->M[i] = (i % (d + 1) == 0);
            for (int i = 0; i < d; i++) c->lam[i] = 1.0;
        } break;
        default: break;
        }
    }
}
static int kkt_factor(S_t *S) {
    int64_t r = ldl_factor(S->F, S->Kx, S->sgn, 1e-13, 2e-7);
    if (g_verbose > 1) fprintf(stderr, "    factor: %lld dynamic regs\n", (long long)r);
    return r >= 0;
}

/* ---------------------------------------------------------------- checker (original data) */
typedef struct {
    I n, m;
    I *Pp, *Pi; double *Px;   /* full symmetric P (cleaned) */
    I *Ap, *Ai; double *Ax;
    double *q, *b;
    double ascale, hscale;
    int nc; cone_t *K;      /* original cone list (merged zero/nonneg) */
    I *rmap;                /* reduced row -> original row (NULL if identity) */
    char *keep;             /* kept-row mask (original rows) */
} orig_t;

static double exp_viol(double x, double y, double z) {
    double bnd = fmax(fabs(y), fmax(fmax(x, 0.0), fmax(-z, 0.0)));
    if (y > 0 && z > 0) {
        double in = fmax(0.0, x - y * (log(z) - log(y)));
        return fmin(in, bnd);
    }
    return bnd;
}
static double cone_viol(const cone_t *K, int nc, const double *v, int dual) {
    double worst = 0;
    for (int k = 0; k < nc; k++) {
        const cone_t *c = &K[k];
        const double *b = v + c->start;
        I d = c->len;
        double sc = fmax(1.0, ninf(b, d)), viol = 0;
        switch (c->type) {
        case K_ZERO: viol = dual ? 0.0 : ninf(b, d) / sc; break;
        case K_NN: { double mn = INFINITY; for (I i = 0; i < d; i++) if (b[i] / sc < mn) mn = b[i] / sc; viol = fmax(0.0, -mn); } break;
        case K_SOC: { double nr = 0; for (I i = 1; i < d; i++) nr += (b[i] / sc) * (b[i] / sc); viol = fmax(0.0, sqrt(nr) - b[0] / sc); } break;
        case K_EXP:
            if (dual) viol = exp_viol(-b[1] / sc, -b[0] / sc, M_E * (b[2] / sc));
            else viol = exp_viol(b[0] / sc, b[1] / sc, b[2] / sc);
            break;
        case K_PSD: {
            int dd = (int)c->d;
            double *u = NEW(double, d);
            for (I i = 0; i < d; i++) u[i] = b[i] / sc;
            double *X = NEW(double, (size_t)dd * dd);
            smat(dd, u, X);
            int lw = 3 * dd + 64;
            double *wk = NEW(double, lw + dd);
            double mn = min_eig(dd, X, wk, lw + dd);
            viol = fmax(0.0, -mn);
            free(u); free(X); free(wk);
        } break;
        }
        if (!(viol <= worst)) worst = viol;   /* propagates NaN as worst */
        if (isnan(viol)) return INFINITY;
    }
    return worst;
}
/* returns 1 if (x,s,z) passes solved checks */
static int check_solved(orig_t *O, const double *x, const double *s, const double *z, double *info) {
    I n = O->n, m = O->m;
    double *Ax = ZNEW(double, m), *Px = ZNEW(double, n), *Atz = ZNEW(double, n), *r = NEW(double, m > n ? m : n);
    csc_mv(n, O->Ap, O->Ai, O->Ax, x, Ax);
    csc_mv(n, O->Pp, O->Pi, O->Px, x, Px);
    csc_mtv(n, O->Ap, O->Ai, O->Ax, z, Atz);
    for (I i = 0; i < m; i++) r[i] = Ax[i] + s[i] - O->b[i];
    double rp = ninf(r, m) / (1.0 + fmax(ninf(Ax, m), fmax(ninf(s, m), ninf(O->b, m))));
    for (I j = 0; j < n; j++) r[j] = Px[j] + O->q[j] + Atz[j];
    double rd = ninf(r, n) / (1.0 + fmax(ninf(Px, n), fmax(ninf(O->q, n), ninf(Atz, n))));
    double xPx = dot(x, Px, n), qx = dot(O->q, x, n), bz = dot(O->b, z, m);
    double p0 = 0.5 * xPx + qx, d0 = -0.5 * xPx - bz;
    double gap = fabs(p0 - d0), scl = fmax(fabs(p0), fabs(d0));
    double rel = scl > 0 ? gap / scl : 0.0;
    double cp = cone_viol(O->K, O->nc, s, 0), cd = cone_viol(O->K, O->nc, z, 1);
    free(Ax); free(Px); free(Atz); free(r);
    if (info) { info[0] = rp; info[1] = rd; info[2] = gap; info[3] = rel; info[4] = cp; info[5] = cd; }
    int ok = rp <= TOLA && rd <= TOLA && (gap <= TOLA || rel <= TOLA) && cp <= TOLA && cd <= TOLA;
    return ok && isfinite(rp) && isfinite(rd) && isfinite(gap);
}
static int check_pinf(orig_t *O, const double *z) {
    I n = O->n, m = O->m;
    double zn = ninf(z, m), bn = ninf(O->b, m);
    if (!(zn > 0 && bn > 0)) return 0;
    double *zh = NEW(double, m), *Atz = ZNEW(double, n);
    for (I i = 0; i < m; i++) zh[i] = z[i] / zn;
    double beta = 0;
    for (I i = 0; i < m; i++) beta -= (O->b[i] / bn) * zh[i];
    int ok = 0;
    if (beta > TOL) {
        csc_mtv(n, O->Ap, O->Ai, O->Ax, zh, Atz);
        double lin = 0;
        for (I j = 0; j < n; j++) { double v = fabs(Atz[j] / O->ascale); if (v > lin) lin = v; }
        double cv = cone_viol(O->K, O->nc, zh, 1);
        ok = (lin / beta <= TOLA) && (cv / beta <= TOLA);
        if (g_verbose > 1) fprintf(stderr, "    pinf check: beta=%.3e lin/beta=%.3e cone/beta=%.3e\n", beta, lin / beta, cv / beta);
    }
    free(zh); free(Atz);
    return ok;
}
static int check_dinf(orig_t *O, const double *x, const double *s) {
    I n = O->n, m = O->m;
    double xn = ninf(x, n), qn = ninf(O->q, n);
    if (!(xn > 0 && qn > 0)) return 0;
    double *xh = NEW(double, n), *sh = NEW(double, m), *Px = ZNEW(double, n), *Ax = ZNEW(double, m);
    for (I j = 0; j < n; j++) xh[j] = x[j] / xn;
    for (I i = 0; i < m; i++) sh[i] = s[i] / xn;
    double alpha = 0;
    for (I j = 0; j < n; j++) alpha -= (O->q[j] / qn) * xh[j];
    int ok = 0;
    if (alpha > TOL) {
        csc_mv(n, O->Pp, O->Pi, O->Px, xh, Px);
        csc_mv(n, O->Ap, O->Ai, O->Ax, xh, Ax);
        double pq = 0, lin = 0;
        for (I j = 0; j < n; j++) { double v = fabs(Px[j] / O->hscale); if (v > pq) pq = v; }
        for (I i = 0; i < m; i++) {
            double v = fabs(Ax[i] / O->ascale + sh[i] / O->ascale);
            if (v > lin) lin = v;
            sh[i] /= O->ascale;
        }
        double cv = cone_viol(O->K, O->nc, sh, 0);
        ok = (pq / alpha <= TOLA) && (lin / alpha <= TOLA) && (cv / alpha <= TOLA);
    }
    free(xh); free(sh); free(Px); free(Ax);
    return ok;
}

/* ---------------------------------------------------------------- equilibration */
static void equilibrate(S_t *S) {
    I n = S->n, m = S->m;
    double *D = S->D, *E = S->E;
    for (I j = 0; j < n; j++) D[j] = 1;
    for (I i = 0; i < m; i++) E[i] = 1;
    double *cn = NEW(double, n > 0 ? n : 1), *rn = NEW(double, m > 0 ? m : 1);
    double *dl = NEW(double, n > 0 ? n : 1), *el = NEW(double, m > 0 ? m : 1);
    const double mins = 1e-4, maxs = 1e4;
    for (int it = 0; it < 10; it++) {
        for (I j = 0; j < n; j++) cn[j] = 0;
        for (I i = 0; i < m; i++) rn[i] = 0;
        for (I j = 0; j < n; j++)
            for (I k = S->Pp[j]; k < S->Pp[j + 1]; k++) {
                double v = fabs(S->Px[k]);
                I i = S->Pi[k];
                if (v > cn[j]) cn[j] = v;
                if (v > cn[i]) cn[i] = v;
            }
        for (I j = 0; j < n; j++)
            for (I k = S->Ap[j]; k < S->Ap[j + 1]; k++) {
                double v = fabs(S->Ax[k]);
                if (v > cn[j]) cn[j] = v;
                if (v > rn[S->Ai[k]]) rn[S->Ai[k]] = v;
            }
        double dev = 0;
        for (I j = 0; j < n; j++) {
            double v = cn[j];
            if (v < mins) v = 1.0; else if (v > maxs) v = maxs;
            dl[j] = 1.0 / sqrt(v);
            dev = fmax(dev, fabs(1 - v));
        }
        for (I i = 0; i < m; i++) {
            double v = rn[i];
            if (v < mins) v = 1.0; else if (v > maxs) v = maxs;
            el[i] = 1.0 / sqrt(v);
            dev = fmax(dev, fabs(1 - v));
        }
        for (int k = 0; k < S->nc; k++) {
            cone_t *c = &S->K[k];
            if (c->type == K_SOC || c->type == K_EXP || c->type == K_PSD) {
                double mean = 0;
                for (I i = 0; i < c->len; i++) mean += el[c->start + i];
                mean /= c->len;
                for (I i = 0; i < c->len; i++) el[c->start + i] = mean;
            }
        }
        if (dev < 1e-3) break;
        /* apply */
        for (I j = 0; j < n; j++) {
            for (I k = S->Pp[j]; k < S->Pp[j + 1]; k++) S->Px[k] *= dl[j] * dl[S->Pi[k]];
            for (I k = S->Ap[j]; k < S->Ap[j + 1]; k++) S->Ax[k] *= dl[j] * el[S->Ai[k]];
        }
        for (I j = 0; j < n; j++) { D[j] *= dl[j]; }
        for (I i = 0; i < m; i++) { E[i] *= el[i]; }
    }
    /* cost scaling */
    for (I j = 0; j < n; j++) cn[j] = 0;
    for (I j = 0; j < n; j++)
        for (I k = S->Pp[j]; k < S->Pp[j + 1]; k++) {
            double v = fabs(S->Px[k]);
            I i = S->Pi[k];
            if (v > cn[j]) cn[j] = v;
            if (v > cn[i]) cn[i] = v;
        }
    double mean = 0;
    for (I j = 0; j < n; j++) mean += cn[j];
    mean = n > 0 ? mean / n : 0;
    for (I j = 0; j < n; j++) S->q[j] *= D[j];
    for (I i = 0; i < m; i++) S->b[i] *= E[i];
    double sc = fmax(mean, ninf(S->q, n));
    if (sc < mins) sc = 1.0; else if (sc > maxs) sc = maxs;
    S->c = 1.0 / sc;
    for (I k = 0; k < S->Pp[n]; k++) S->Px[k] *= S->c;
    for (I j = 0; j < n; j++) S->q[j] *= S->c;
    free(cn); free(rn); free(dl); free(el);
}

/* ---------------------------------------------------------------- main solve */
typedef struct {
    int status;   /* 0 unresolved, 1 solved, 2 pinf, 3 dinf */
    double *x, *s, *z;
} result_t;

static void unscale(S_t *S, const double *x, const double *s, const double *z, double tau,
                    double *xo, double *so, double *zo) {
    for (I j = 0; j < S->n; j++) xo[j] = S->D[j] * x[j] / tau;
    for (I i = 0; i < S->m; i++) { so[i] = s[i] / (S->E[i] * tau); zo[i] = S->E[i] * z[i] / (S->c * tau); }
}

/* expand reduced s,z to original rows; mode 0: s = b - Ax for dropped rows, mode 1: s = -Ax */
static void expand_rows(S_t *S, orig_t *O, const double *xo, double *sr, double *zr, double *sf, double *zf, int mode) {
    if (!O->rmap) {
        if (sr && sf != sr) memcpy(sf, sr, sizeof(double) * S->m);
        if (zr && zf != zr) memcpy(zf, zr, sizeof(double) * S->m);
        return;
    }
    I mf = O->m;
    if (sf) {
        double *Ax = ZNEW(double, mf);
        csc_mv(O->n, O->Ap, O->Ai, O->Ax, xo, Ax);
        for (I i = 0; i < mf; i++) sf[i] = (mode == 0 ? O->b[i] : 0.0) - Ax[i];
        free(Ax);
        for (I i = 0; i < S->m; i++) sf[O->rmap[i]] = sr[i];
    }
    if (zf) {
        for (I i = 0; i < mf; i++) zf[i] = 0;
        for (I i = 0; i < S->m; i++) zf[O->rmap[i]] = zr[i];
    }
}
static void solve(S_t *S, orig_t *O, result_t *res) {
    I n = S->n, m = S->m, N = S->N;
    double *x = ZNEW(double, n), *s = ZNEW(double, m), *z = ZNEW(double, m);
    double tau = 1, kap = 1;
    double *rx = NEW(double, n), *rz = NEW(double, m);
    double *Px = NEW(double, n), *Axv = NEW(double, m), *Atz = NEW(double, n);
    double *rhs = ZNEW(double, N), *sol = ZNEW(double, N), *sol1 = ZNEW(double, N);
    double *x1 = NEW(double, n), *z1 = NEW(double, m);
    double *dx = NEW(double, n), *dz = NEW(double, m), *dsv = NEW(double, m), *dsd = NEW(double, m);
    double *dxa = NEW(double, n), *dza = NEW(double, m), *dsa = NEW(double, m);
    double *Hv = NEW(double, m), *xi = NEW(double, n), *tmpn = NEW(double, n);
    double *xo = NEW(double, n), *so = NEW(double, O->m + 1), *zo = NEW(double, O->m + 1);
    double *sr = NEW(double, m + 1), *zr = NEW(double, m + 1);
    res->status = 0;
    res->x = xo; res->s = so; res->z = zo;

    S->pd_scaling = getenv("CONIC_DUALSC") ? 0 : g_pd0;
    double eps = getenv("CONIC_EPSREG") ? atof(getenv("CONIC_EPSREG")) : g_eps0;
    kkt_set_reg(S, eps);
    /* initialisation */
    if (S->nexp == 0) {
        cone_identity_scaling(S);
        kkt_fill_H(S);
        if (!kkt_factor(S)) {
            if (g_verbose) {
                I bad = 0;
                for (I k = 0; k < S->Kp[S->N]; k++) if (!isfinite(S->Kx[k])) bad++;
                fprintf(stderr, "init factor failed: %lld nonfinite Kx entries; c=%g\n", (long long)bad, S->c);
            }
            if (getenv("CONIC_DUMPK")) {
                FILE *fp = fopen("/tmp/K.bin", "wb");
                fwrite(&S->N, sizeof(I), 1, fp); fwrite(S->Kp, sizeof(I), S->N + 1, fp);
                fwrite(S->Ki, sizeof(I), S->Kp[S->N], fp); fwrite(S->Kx, sizeof(double), S->Kp[S->N], fp);
                fwrite(S->sgn, 1, S->N, fp); fclose(fp);
            }
            if (g_verbose) fprintf(stderr, "init factor failed\n");
            return;
        }
        /* temporarily the "true" matrix in kkt_mv uses Kx - regd, consistent with H = I */
        for (I j = 0; j < n; j++) rhs[j] = -S->q[j];
        for (I i = 0; i < m; i++) rhs[n + i] = S->b[i];
        for (I i = n + m; i < N; i++) rhs[i] = 0;
        kkt_solve_xz(S, rhs, sol);
        for (I j = 0; j < n; j++) x[j] = sol[j];
        for (I i = 0; i < m; i++) { z[i] = sol[n + i]; s[i] = -sol[n + i]; }
        cone_shift(S, s);
        cone_shift(S, z);
    } else {
        cone_unit_init(S, s, z);
    }

    int maxit = 200;
    double tol_in = TOLA;
    int iter;
    int nfail_small = 0;
    double best_merit = INFINITY, best_ratio = INFINITY;
    int last_imp = 0;
    for (iter = 0; iter < maxit; iter++) {
        /* residuals */
        memset(Px, 0, sizeof(double) * n); sym_mv(n, S->Pp, S->Pi, S->Px, x, Px);
        memset(Axv, 0, sizeof(double) * m); csc_mv(n, S->Ap, S->Ai, S->Ax, x, Axv);
        memset(Atz, 0, sizeof(double) * n); csc_mtv(n, S->Ap, S->Ai, S->Ax, z, Atz);
        for (I j = 0; j < n; j++) rx[j] = Px[j] + Atz[j] + S->q[j] * tau;
        for (I i = 0; i < m; i++) rz[i] = Axv[i] + s[i] - S->b[i] * tau;
        double xPx = dot(x, Px, n), qx = dot(S->q, x, n), bz = dot(S->b, z, m);
        double rtau = qx + bz + xPx / tau + kap;
        double sz = dot(s, z, m);
        double mu = (sz + tau * kap) / (S->nu + 1.0);

        /* termination checks in original units */
        {
            double pres = 0, nAx = 0, ns = 0, nb = 0;
            for (I i = 0; i < m; i++) {
                double ie = 1.0 / S->E[i];
                pres = fmax(pres, fabs(rz[i] * ie));
                nAx = fmax(nAx, fabs(Axv[i] * ie));
                ns = fmax(ns, fabs(s[i] * ie));
                nb = fmax(nb, fabs(S->b[i] * ie));
            }
            pres /= tau; nAx /= tau; ns /= tau;
            double rp = pres / (1 + fmax(nAx, fmax(ns, nb)));
            double dres = 0, nPx = 0, nq = 0, nAtz = 0;
            for (I j = 0; j < n; j++) {
                double id = 1.0 / S->D[j];
                dres = fmax(dres, fabs(rx[j] * id));
                nPx = fmax(nPx, fabs(Px[j] * id));
                nq = fmax(nq, fabs(S->q[j] * id));
                nAtz = fmax(nAtz, fabs(Atz[j] * id));
            }
            double ct = S->c * tau;
            dres /= ct; nPx /= ct; nq /= S->c; nAtz /= ct;
            double rd = dres / (1 + fmax(nPx, fmax(nq, nAtz)));
            double p0 = (0.5 * xPx / (tau * tau) + qx / tau) / S->c;
            double d0 = (-0.5 * xPx / (tau * tau) - bz / tau) / S->c;
            double gap = fabs(p0 - d0), scl = fmax(fabs(p0), fabs(d0));
            double rel = scl > 0 ? gap / scl : 0;
            if (g_verbose)
                fprintf(stderr, "%3d p0=% .8e d0=% .8e rp=%.2e rd=%.2e gap=%.2e rel=%.2e tau=%.2e kap=%.2e mu=%.2e t=%.3f\n",
                        iter, p0, d0, rp, rd, gap, rel, tau, kap, mu, now() - g_t0);
            g_refatol = (getenv("CONIC_REFTOL_FIXED") || !g_adaptive_ref) ? 0.0 : 1e-3 * fmin(ninf(rx, n), ninf(rz, m));
            {
                /* stall detection: no 30% progress in merit or toward a certificate for 30 iterations */
                double merit = fmax(rp, fmax(rd, fmin(rel, gap)));
                double ratio = tau / fmax(kap, 1e-300);
                if (merit < 0.7 * best_merit) { best_merit = merit; last_imp = iter; }
                if (ratio < 0.5 * best_ratio) { best_ratio = ratio; last_imp = iter; }
                if (iter - last_imp > 30 && getenv("CONIC_STALL")) {
                    if (g_verbose) fprintf(stderr, "stall detected\n");
                    break;
                }
            }
            if (rp <= tol_in && rd <= tol_in && (gap <= tol_in || rel <= tol_in)) {
                unscale(S, x, s, z, tau, xo, sr, zr);
                expand_rows(S, O, xo, sr, zr, so, zo, 0);
                double info[6];
                if (check_solved(O, xo, so, zo, info)) { res->status = 1; break; }
                if (g_verbose) fprintf(stderr, "  final check failed: %.2e %.2e %.2e %.2e %.2e %.2e\n",
                                       info[0], info[1], info[2], info[3], info[4], info[5]);
                if (O->rmap) {
                    /* optimistic presolve: a dropped row is violated -> give up quickly, rerun with it */
                    for (I i = 0; i < O->m; i++)
                        if (!O->keep[i] && so[i] < -1e-9 * fmax(1.0, fabs(O->b[i]))) { g_drop_violated = 1; break; }
                    if (g_drop_violated) { if (g_verbose) fprintf(stderr, "dropped row violated\n"); break; }
                }
                tol_in *= 0.3;
            }
            /* infeasibility */
            if (bz < 0) {
                for (I i = 0; i < m; i++) zr[i] = S->E[i] * z[i];
                expand_rows(S, O, xo, NULL, zr, NULL, zo, 0);
                if (check_pinf(O, zo)) { res->status = 2; break; }
            }
            if (qx < 0) {
                for (I j = 0; j < n; j++) xo[j] = S->D[j] * x[j];
                for (I i = 0; i < m; i++) sr[i] = s[i] / S->E[i];
                expand_rows(S, O, xo, sr, NULL, so, NULL, 1);
                if (check_dinf(O, xo, so)) { res->status = 3; break; }
            }
            if (O->rmap && qx < 0 && tau < 1e-8 * fmax(1.0, kap)) {
                /* reduced problem looks unbounded: dropped large-bound rows may be binding -> rerun with them */
                g_drop_violated = 1;
                if (g_verbose) fprintf(stderr, "reduced problem unbounded; rerun with dropped rows\n");
                break;
            }
        }

        /* scaling + factorisation */
        int ok = 0;
        for (int attempt = 0; attempt < 2 && !ok; attempt++) {
            if (!cone_update_scaling(S, s, z, mu)) {
                if (S->nexp && S->pd_scaling) { S->pd_scaling = 0; continue; }
                break;
            }
            kkt_fill_H(S);
            ok = kkt_factor(S);
        }
        if (!ok) { if (g_verbose) fprintf(stderr, "scaling/factor failure (pd=%d)\n", S->pd_scaling); break; }

        /* constant system */
        for (I j = 0; j < n; j++) rhs[j] = -S->q[j];
        for (I i = 0; i < m; i++) rhs[n + i] = S->b[i];
        for (I i = n + m; i < N; i++) rhs[i] = 0;
        kkt_solve_xz(S, rhs, sol1);
        memcpy(x1, sol1, sizeof(double) * n);
        memcpy(z1, sol1 + n, sizeof(double) * m);
        for (I j = 0; j < n; j++) xi[j] = x[j] / tau;
        /* denominator: kap/tau + (x1-xi)'P(x1-xi) + z1'Hz1 */
        double den;
        {
            for (I j = 0; j < n; j++) tmpn[j] = x1[j] - xi[j];
            double *Pt = dxa; memset(Pt, 0, sizeof(double) * n);
            sym_mv(n, S->Pp, S->Pi, S->Px, tmpn, Pt);
            cone_Hmul(S, z1, Hv);
            den = kap / tau + dot(tmpn, Pt, n) + dot(z1, Hv, m);
        }
        /* (q + 2 P xi) */
        double *q2 = tmpn;
        for (I j = 0; j < n; j++) q2[j] = S->q[j] + 2.0 * Px[j] / tau;
        double xiPxi = xPx / (tau * tau);
        /* include the regularisation terms the (possibly singular) regularised solve carries */
        if (!getenv("CONIC_DEN_NOEPS")) den += S->eps_reg * (dot(x1, x1, n) + dot(z1, z1, m));
        if (!getenv("CONIC_DEN_POSITIVE")) { double d2 = kap / tau + xPx / (tau * tau) - dot(S->q, x1, n) - 2.0 * dot(Px, x1, n) / tau - dot(S->b, z1, m); if (d2 > 0) den = d2; }

        double sigma = 0, alpha = 0, dtau = 0, dkap = 0;
        double dtau_a = 0, dkap_a = 0;
        for (int phase = 0; phase < 2; phase++) {
            double fac = (phase == 0) ? 1.0 : (1.0 - sigma);
            double dkapv;
            if (phase == 0) {
                cone_ds(S, s, z, NULL, NULL, 0.0, dsd);
                dkapv = tau * kap;
            } else {
                cone_ds(S, s, z, dsa, dza, sigma * mu, dsd);
                dkapv = tau * kap + dtau_a * dkap_a - sigma * mu;
            }
            for (I j = 0; j < n; j++) rhs[j] = -fac * rx[j];
            for (I i = 0; i < m; i++) rhs[n + i] = -fac * rz[i] + dsd[i];
            for (I i = n + m; i < N; i++) rhs[i] = 0;
            kkt_solve_xz(S, rhs, sol);
            double num = fac * rtau - dkapv / tau + dot(q2, sol, n) + dot(S->b, sol + n, m);
            dtau = num / den;
            for (I j = 0; j < n; j++) dx[j] = sol[j] + dtau * x1[j];
            for (I i = 0; i < m; i++) dz[i] = sol[n + i] + dtau * z1[i];
            /* ds = -d_s - H dz, using the KKT's own representation of H */
            if (S->npsd) {
                memset(Hv, 0, sizeof(double) * m);
                csc_mv(n, S->Ap, S->Ai, S->Ax, dx, Hv);
            }
            for (int k = 0; k < S->nc; k++) {
                cone_t *c = &S->K[k];
                I st = c->start, len = c->len;
                double *o = dsv + st; const double *dzc = dz + st, *dd = dsd + st;
                switch (c->type) {
                case K_ZERO: for (I i = 0; i < len; i++) o[i] = 0; break;
                case K_NN: { const double *w = S->wb + st; for (I i = 0; i < len; i++) o[i] = -dd[i] - w[i] * w[i] * dzc[i]; } break;
                case K_EXP:
                    for (int i = 0; i < 3; i++) o[i] = -dd[i] - (c->Hs[i] * dzc[0] + c->Hs[i + 3] * dzc[1] + c->Hs[i + 6] * dzc[2]);
                    break;
                case K_SOC: {
                    const double *w = S->wb + st;
                    double e2 = c->eta * c->eta;
                    if (c->expanded) {
                        double p1 = sol[c->ecol] + dtau * sol1[c->ecol], p2 = sol[c->ecol + 1] + dtau * sol1[c->ecol + 1];
                        o[0] = -dd[0] - (e2 * c->D0 * dzc[0] - c->eta * (2.0 * w[0] / c->alpha) * p1);
                        for (I i = 1; i < len; i++)
                            o[i] = -dd[i] - (e2 * dzc[i] - c->eta * w[i] * (c->alpha * p1 + c->beta * p2));
                    } else {
                        double wv = dot(w, dzc, len);
                        o[0] = -dd[0] - e2 * (2 * wv * w[0] - dzc[0]);
                        for (I i = 1; i < len; i++) o[i] = -dd[i] - e2 * (2 * wv * w[i] + dzc[i]);
                    }
                } break;
                case K_PSD:
                    for (I i = 0; i < len; i++) o[i] = -fac * rz[st + i] + S->b[st + i] * dtau - Hv[st + i];
                    break;
                }
            }
            dkap = -(dkapv + kap * dtau) / tau;
            /* step length */
            double amax = INFINITY;
            if (dtau < 0) amax = fmin(amax, -tau / dtau);
            if (dkap < 0) amax = fmin(amax, -kap / dkap);
            amax = cone_step(S, s, z, dsv, dz, amax);
            if (S->nexp && g_theta > 0 && phase == 1) {
                /* neighbourhood safeguard: each exp cone keeps s'z/3 >= theta * mu(alpha) */
                double a = fmin(1.0, 0.99 * amax);
                for (int bt = 0; bt < 30; bt++) {
                    double t2 = tau + a * dtau, k2 = kap + a * dkap, sz2 = 0;
                    for (I i = 0; i < m; i++) sz2 += (s[i] + a * dsv[i]) * (z[i] + a * dz[i]);
                    double mu2 = (sz2 + t2 * k2) / (S->nu + 1.0);
                    int ok = 1;
                    for (int k = 0; k < S->nc && ok; k++) {
                        cone_t *c = &S->K[k];
                        if (c->type != K_EXP) continue;
                        double loc = 0;
                        for (int i = 0; i < 3; i++) loc += (s[c->start + i] + a * dsv[c->start + i]) * (z[c->start + i] + a * dz[c->start + i]);
                        if (loc / 3.0 < g_theta * mu2) ok = 0;
                    }
                    if (ok) break;
                    a *= 0.8;
                }
                amax = a / 0.99;
            }
            if (phase == 0) {
                alpha = fmin(1.0, amax);
                sigma = pow(1.0 - alpha, 3);
                if (getenv("CONIC_SIGMIN")) sigma = fmax(sigma, atof(getenv("CONIC_SIGMIN")));
                memcpy(dxa, dx, sizeof(double) * n);
                memcpy(dza, dz, sizeof(double) * m);
                memcpy(dsa, dsv, sizeof(double) * m);
                dtau_a = dtau; dkap_a = dkap;
            } else {
                alpha = fmin(1.0, 0.99 * amax);
            }
        }
        if (g_verbose > 1) fprintf(stderr, "    sigma=%.3e alpha=%.3e pd=%d\n", sigma, alpha, S->pd_scaling);
        if (S->nexp && !S->pd_scaling && alpha < 0.1 && g_exp_limited && getenv("CONIC_TOGGLE")) {
            S->pd_scaling = 1;   /* dual scaling stuck as well: go back to primal-dual for the next iteration */
        } else
        if (S->nexp && S->pd_scaling && alpha < 0.1 && (g_exp_limited || getenv("CONIC_SWITCH_ANY"))) {
            S->pd_scaling = 0;
            continue;
        }
        if (alpha < 1e-8) {
            if (++nfail_small > 3) break;
        }
        for (I j = 0; j < n; j++) x[j] += alpha * dx[j];
        for (I i = 0; i < m; i++) { s[i] += alpha * dsv[i]; z[i] += alpha * dz[i]; }
        tau += alpha * dtau;
        kap += alpha * dkap;
        for (int k = 0; k < S->nc; k++) if (S->K[k].type == K_ZERO)
            for (I i = 0; i < S->K[k].len; i++) s[S->K[k].start + i] = 0;
        if (!(tau > 1e-200) || !(kap > 0) || !isfinite(tau) || !isfinite(kap)) break;
    }
    if (g_verbose) fprintf(stderr, "status %d after %d iterations, %.3fs\n", res->status, iter, now() - g_t0);
    if (res->status == 2) {
        double zn = ninf(zo, O->m);
        for (I i = 0; i < O->m; i++) zo[i] /= zn;
    } else if (res->status == 3) {
        double xn = ninf(xo, n);
        for (I j = 0; j < n; j++) xo[j] /= xn;
        for (I i = 0; i < O->m; i++) so[i] /= xn;
    }
}

/* ---------------------------------------------------------------- setup */
int main(int argc, char **argv) {
    const char *in = NULL, *out = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--input") && i + 1 < argc) in = argv[++i];
        else if (!strcmp(argv[i], "--output") && i + 1 < argc) out = argv[++i];
        else if (!strncmp(argv[i], "--input=", 8)) in = argv[i] + 8;
        else if (!strncmp(argv[i], "--output=", 9)) out = argv[i] + 9;
    }
    g_t0 = now();
    const char *ve = getenv("CONIC_VERBOSE");
    if (ve) g_verbose = atoi(ve);
    if (getenv("CONIC_THETA")) g_theta = atof(getenv("CONIC_THETA"));
    load_blas();
    g_dropthr = 1e20;
    if (getenv("CONIC_DROPTHR")) g_dropthr = atof(getenv("CONIC_DROPTHR"));
    if (getenv("CONIC_DROPTHR")) g_dropthr = atof(getenv("CONIC_DROPTHR"));
    if (g_dgemm && g_dtrsm) ldl_set_blas((ldl_dgemm_fn)g_dgemm, (ldl_dtrsm_fn)g_dtrsm);
    if (!in || !out) { fprintf(stderr, "usage: solver --input IN --output OUT\n"); return 0; }
    g_out = out;
    npz_problem P;
    char err[512];
    if (npz_read_problem(in, &P, err, sizeof err) != 0) fail_unresolved(err);
    I n = P.n, m = P.m;

    orig_t O;
    memset(&O, 0, sizeof O);
    O.n = n; O.m = m;
    csc_clean(n, n, P.P_colptr, P.P_rowind, P.P_val, &O.Pp, &O.Pi, &O.Px);
    csc_clean(m, n, P.A_colptr, P.A_rowind, P.A_val, &O.Ap, &O.Ai, &O.Ax);
    O.q = P.q; O.b = P.b;
    O.ascale = ninf(O.Ax, O.Ap[n]); if (O.ascale == 0) O.ascale = 1;
    O.hscale = ninf(O.Px, O.Pp[n]); if (O.hscale == 0) O.hscale = 1;

    /* original cones (merge consecutive zero / nonneg) */
    O.K = ZNEW(cone_t, P.ncones + 1);
    I row = 0;
    O.nc = 0;
    for (int k = 0; k < P.ncones; k++) {
        int t = P.cone_type[k];
        I d = P.cone_dim[k];
        I len = (t == K_PSD) ? d * (d + 1) / 2 : d;
        if ((t == K_ZERO || t == K_NN) && O.nc > 0 && O.K[O.nc - 1].type == t) {
            O.K[O.nc - 1].len += len;
        } else {
            cone_t *c = &O.K[O.nc++];
            c->type = t; c->start = row; c->len = len; c->d = d;
        }
        row += len;
    }
    if (row != m) fail_unresolved("cone dims mismatch");
    /* presolve: drop nonnegative rows with infinite upper bound b >= 1e20 */
    char *keep = NEW(char, m + 1);
    I mr = 0;
    for (I i = 0; i < m; i++) keep[i] = 1;
    if (!getenv("CONIC_NOPRESOLVE")) {
        I *rcnt = ZNEW(I, m + 1);
        for (I k = 0; k < O.Ap[n]; k++) rcnt[O.Ai[k]]++;
        for (int k = 0; k < O.nc; k++)
            if (O.K[k].type == K_NN)
                for (I i = O.K[k].start; i < O.K[k].start + O.K[k].len; i++)
                    if (O.b[i] >= 1e20 || (O.b[i] >= g_dropthr && rcnt[i] <= 1)) keep[i] = 0;
        free(rcnt);
    }
    for (I i = 0; i < m; i++) mr += keep[i];
    O.keep = keep;
    S_t S;
    memset(&S, 0, sizeof S);
    S.n = n; S.m = mr;
    S.K = ZNEW(cone_t, O.nc + 1);
    S.nc = 0; S.nu = 0;
    {
        I r2 = 0;
        for (int k = 0; k < O.nc; k++) {
            cone_t *c0 = &O.K[k];
            I len = c0->len;
            if (c0->type == K_NN && mr < m) { len = 0; for (I i = c0->start; i < c0->start + c0->len; i++) len += keep[i]; }
            if (len == 0) continue;
            cone_t *c = &S.K[S.nc++];
            c->type = c0->type; c->start = r2; c->len = len; c->d = c0->d;
            r2 += len;
        }
    }
    I *rnew = NULL;
    if (mr < m) {
        O.rmap = NEW(I, mr + 1);
        rnew = NEW(I, m + 1);
        I r2 = 0;
        for (I i = 0; i < m; i++) { if (keep[i]) { O.rmap[r2] = i; rnew[i] = r2++; } else rnew[i] = -1; }
    }
    for (int k = 0; k < S.nc; k++) {
        cone_t *c = &S.K[k];
        switch (c->type) {
        case K_ZERO: break;
        case K_NN: S.nu += c->len; break;
        case K_SOC: S.nu += 1; break;
        case K_EXP: S.nu += 3; S.nexp++; break;
        case K_PSD: {
            S.nu += c->d; S.npsd++;
            int d = (int)c->d;
            if (d > S.maxd) S.maxd = d;
            c->R = NEW(double, (size_t)d * d); c->Ri = NEW(double, (size_t)d * d);
            c->M = NEW(double, (size_t)d * d); c->lam = NEW(double, d);
            c->lwk = 6 * d * d + 10 * d + 256;
            c->wk = NEW(double, c->lwk);
        } break;
        }
    }
    /* scaled copies */
    I *Pu_p = NEW(I, n + 1), *Pu_i = NEW(I, O.Pp[n]); double *Pu_x = NEW(double, O.Pp[n]);
    {
        I p = 0;
        for (I j = 0; j < n; j++) {
            Pu_p[j] = p;
            for (I k = O.Pp[j]; k < O.Pp[j + 1]; k++)
                if (O.Pi[k] <= j) { Pu_i[p] = O.Pi[k]; Pu_x[p] = O.Px[k]; p++; }
        }
        Pu_p[n] = p;
    }
    S.Pp = Pu_p; S.Pi = Pu_i; S.Px = Pu_x;
    S.Ap = NEW(I, n + 1); S.Ai = NEW(I, O.Ap[n] + 1); S.Ax = NEW(double, O.Ap[n] + 1);
    {
        I p = 0;
        for (I j = 0; j < n; j++) {
            S.Ap[j] = p;
            for (I k = O.Ap[j]; k < O.Ap[j + 1]; k++) {
                I i = O.Ai[k];
                if (rnew) { if (rnew[i] < 0) continue; i = rnew[i]; }
                S.Ai[p] = i; S.Ax[p] = O.Ax[k]; p++;
            }
        }
        S.Ap[n] = p;
    }
    S.q = NEW(double, n); memcpy(S.q, O.q, sizeof(double) * n);
    S.b = NEW(double, mr + 1);
    for (I i = 0; i < m; i++) if (keep[i]) S.b[rnew ? rnew[i] : i] = O.b[i];
    m = mr;
    S.D = NEW(double, n > 0 ? n : 1); S.E = NEW(double, m > 0 ? m : 1);
    equilibrate(&S);
    csc_transpose(m, n, S.Ap, S.Ai, S.Ax, &S.Rp, &S.Rj, &S.Rx);
    S.lam = ZNEW(double, m > 0 ? m : 1); S.wb = ZNEW(double, m > 0 ? m : 1);
    kkt_build(&S);
    I maxlen = 1;
    for (int k = 0; k < S.nc; k++) if (S.K[k].len > maxlen) maxlen = S.K[k].len;
    I tl = S.N > 2 * maxlen ? S.N : 2 * maxlen;
    S.t1 = ZNEW(double, tl + 1); S.t2 = ZNEW(double, tl + 1);
    double t_pre = now();
    if (getenv("CONIC_DUMPK0")) {
        FILE *fp = fopen(getenv("CONIC_DUMPK0"), "wb");
        fwrite(&S.N, sizeof(I), 1, fp); fwrite(S.Kp, sizeof(I), S.N + 1, fp);
        fwrite(S.Ki, sizeof(I), S.Kp[S.N], fp); fwrite(S.Kx, sizeof(double), S.Kp[S.N], fp);
        fwrite(S.sgn, 1, S.N, fp); fclose(fp);
    }
    S.F = ldl_create(S.N, S.Kp, S.Ki, NULL);
    if (g_verbose) fprintf(stderr, "pre-ldl %.3fs, ldl_create %.3fs\n", t_pre - g_t0, now() - t_pre);
    if (!S.F) fail_unresolved("ldl_create failed");
    if (g_verbose) fprintf(stderr, "n=%lld m=%lld N=%lld nnzK=%lld nnzL=%lld setup %.3fs\n", (long long)n, (long long)m,
                           (long long)S.N, (long long)S.Kp[S.N], (long long)ldl_nnz_L(S.F), now() - g_t0);
    result_t R;
    const double eps_try[4] = {1e-8, 1e-7, 1e-8, 1e-10};
    const int pd_try[4] = {1, 1, 0, 1};
    for (int att = 0; att < 4; att++) {
        if (!S.nexp && att == 2) continue;          /* scaling choice only matters for exp cones */
        g_eps0 = eps_try[att];
        g_pd0 = pd_try[att];
        g_adaptive_ref = (att == 0);
        g_refatol = 0.0;
        if (g_verbose && att) fprintf(stderr, "retry %d: eps=%g pd=%d\n", att, g_eps0, g_pd0);
        solve(&S, &O, &R);
        if (R.status != 0) break;
        if (g_drop_violated || (mr < O.m && g_dropthr < 1e20)) break;   /* rerun on the fuller problem instead */
        if (now() - g_t0 > 25.0) break;
    }
    if (R.status == 0 && mr < O.m && !getenv("CONIC_NOPRESOLVE")) {
        /* dropped large-bound rows may have been binding: rerun keeping more rows */
        if (g_dropthr < 1e20) setenv("CONIC_DROPTHR", "1e20", 1);
        else setenv("CONIC_NOPRESOLVE", "1", 1);
        if (g_verbose) fprintf(stderr, "rerun with more rows kept\n");
        execv("/proc/self/exe", argv);
    }
    const char *st = R.status == 1 ? "solved" : R.status == 2 ? "primal_infeasible" : R.status == 3 ? "dual_infeasible" : "unresolved";
    int rc;
    I mo = O.m;
    if (R.status == 1) rc = npz_write_result(out, st, R.x, n, R.s, mo, R.z, mo, 0, 0, 0);
    else if (R.status == 2) rc = npz_write_result(out, st, NULL, 0, NULL, 0, R.z, mo, 0, 0, 0);
    else if (R.status == 3) rc = npz_write_result(out, st, R.x, n, R.s, mo, NULL, 0, 0, 0, 0);
    else rc = npz_write_result(out, st, NULL, 0, NULL, 0, NULL, 0, 0, 0, 0);
    if (rc != 0) fprintf(stderr, "write failed\n");
    if (g_verbose) fprintf(stderr, "total %.3fs\n", now() - g_t0);
    return 0;
}
