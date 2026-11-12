/* Conic interior-point solver: homogeneous embedding (Clarabel-style) with
 * quadratic objective, Ruiz equilibration, sparse quasi-definite LDL' KKT
 * solves with iterative refinement, NT scaling for symmetric cones and
 * dual-barrier scaling for exponential cones.  Termination uses exactly the
 * independent checker's criteria evaluated in the original coordinates. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <dlfcn.h>
#include <dirent.h>

#include "npzio.h"
#include "ldl.h"

/* ---------------- LAPACK ---------------- */
extern void dpotrf_(const char *uplo, const int *n, double *a, const int *lda, int *info, size_t);
extern void dgesvd_(const char *jobu, const char *jobvt, const int *m, const int *n, double *a,
                    const int *lda, double *s, double *u, const int *ldu, double *vt, const int *ldvt,
                    double *work, const int *lwork, int *info, size_t, size_t);
extern void dsyev_(const char *jobz, const char *uplo, const int *n, double *a, const int *lda,
                   double *w, double *work, const int *lwork, int *info, size_t, size_t);

static double wall0;
static double wtime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
static int verbose = 0;
#define LOG(...) do { if (verbose) fprintf(stderr, __VA_ARGS__); } while (0)


static void *xcalloc(size_t n, size_t s) { void *p = calloc(n ? n : 1, s ? s : 1); if (!p) { fprintf(stderr, "oom\n"); exit(3); } return p; }
#define DALLOC(k) ((double *)xcalloc((size_t)(k), sizeof(double)))
#define IALLOC(k) ((int *)xcalloc((size_t)(k), sizeof(int)))

static double nrminf(const double *v, int n) { double r = 0; for (int i = 0; i < n; i++) { double a = fabs(v[i]); if (a > r) r = a; } return r; }
static double dot(const double *a, const double *b, int n) { double r = 0; for (int i = 0; i < n; i++) r += a[i] * b[i]; return r; }

/* ---------------- cones ---------------- */
#define EXP_C0 (-1.051383945322714)
#define EXP_C1 (0.556409619469370)
#define EXP_C2 (1.258967884768947)
#define SOC_DENSE_MAX 24

typedef struct {
    int type, start, dim, ord;
    int dense, expand, pq;
    int wsc;            /* small SOC handled in W-scaled form: rows [W^-1 A_b, -I] */
    int nu_cols; int *ucols; /* union of column indices of the block rows */
    /* SOC */
    double eta, d0, u0, u1, v1;
    /* PSD */
    double *R, *Rinv, *G;   /* ord x ord column major */
    /* EXP */
    double H[9], g[3];
} cone_t;

typedef struct {
    int n, m, nc;
    cone_t *K;
    /* scaled data */
    int *Pp, *Pi; double *Px;         /* full P scaled (CSC) */
    int *Ap, *Ai; double *Ax;         /* scaled A */
    int *Rp, *Rj; double *Rx;         /* scaled A in CSR */
    double *q, *b;
    double *D, *E, c;
    int nu;
    int has_exp;
    /* original */
    problem_t *orig;
    double Amax, Pmax;
    /* workspace for SOC/PSD */
    double *wsoc;     /* m: w-bar per SOC rows */
    double *lam;      /* m: lambda (soc) / eigen-like (psd uses first ord) */
} prob_t;

/* ---------------- sparse helpers ---------------- */
/* y += alpha * M x for CSC M (nrows unused) */
static void csc_mv(int ncol, const int *Mp, const int *Mi, const double *Mx, const double *x, double *y, double alpha)
{
    for (int j = 0; j < ncol; j++) {
        double xj = alpha * x[j];
        if (xj == 0.0) continue;
        for (int k = Mp[j]; k < Mp[j + 1]; k++) y[Mi[k]] += Mx[k] * xj;
    }
}
/* y += alpha * M' x for CSC M */
static void csc_mtv(int ncol, const int *Mp, const int *Mi, const double *Mx, const double *x, double *y, double alpha)
{
    for (int j = 0; j < ncol; j++) {
        double s = 0;
        for (int k = Mp[j]; k < Mp[j + 1]; k++) s += Mx[k] * x[Mi[k]];
        y[j] += alpha * s;
    }
}

/* ---------------- checker replica ---------------- */
static double exp_violation(double x, double y, double z)
{
    double a = y > 0 ? 0 : 0; (void)a;
    double boundary = fabs(y);
    if (x > boundary) boundary = x;
    if (-z > boundary) boundary = -z;
    if (y > 0.0 && z > 0.0) {
        double interior = x - y * (log(z) - log(y));
        if (interior < 0) interior = 0;
        return interior < boundary ? interior : boundary;
    }
    return boundary;
}

static double psd_minev(int d, const double *u)
{
    double *M = DALLOC((size_t)d * d);
    int pos = 0;
    for (int j = 0; j < d; j++) {
        for (int i = 0; i <= j; i++) {
            double v = u[pos++];
            if (i < j) v /= sqrt(2.0);
            M[i + (size_t)j * d] = v; M[j + (size_t)i * d] = v;
        }
    }
    double *w = DALLOC(d);
    int lwork = -1, info = 0; double wq;
    dsyev_("N", "L", &d, M, &d, w, &wq, &lwork, &info, 1, 1);
    lwork = (int)wq + 1;
    double *work = DALLOC(lwork);
    dsyev_("N", "L", &d, M, &d, w, work, &lwork, &info, 1, 1);
    double r = info == 0 ? w[0] : -1e300;
    free(work); free(w); free(M);
    return r;
}

/* cone violation as in the checker; dual != 0 for dual cone */
static double cone_violation(const prob_t *pr, const double *v, int dual)
{
    double worst = 0;
    for (int k = 0; k < pr->nc; k++) {
        const cone_t *c = &pr->K[k];
        const double *blk = v + c->start;
        int d = c->dim;
        double nb = nrminf(blk, d);
        double sc = nb > 1.0 ? nb : 1.0;
        double viol = 0;
        if (c->type == CONE_ZERO) {
            viol = dual ? 0.0 : nb / sc;
        } else if (c->type == CONE_NONNEG) {
            double mn = 0;
            for (int i = 0; i < d; i++) { double u = blk[i] / sc; if (-u > mn) mn = -u; }
            viol = mn;
        } else if (c->type == CONE_SOC) {
            double s2 = 0;
            for (int i = 1; i < d; i++) { double u = blk[i] / sc; s2 += u * u; }
            viol = sqrt(s2) - blk[0] / sc;
            if (viol < 0) viol = 0;
        } else if (c->type == CONE_EXP) {
            double u0 = blk[0] / sc, u1 = blk[1] / sc, u2 = blk[2] / sc;
            viol = dual ? exp_violation(-u1, -u0, M_E * u2) : exp_violation(u0, u1, u2);
        } else {
            double *u = DALLOC(d);
            for (int i = 0; i < d; i++) u[i] = blk[i] / sc;
            double ev = psd_minev(c->ord, u);
            free(u);
            viol = -ev > 0 ? -ev : 0;
        }
        if (!(viol <= worst)) worst = viol;
    }
    return worst;
}

typedef struct { double rp, rd, gap, gaprel, conep, coned, score; double p0, d0; } solcheck_t;

/* full checker metrics for a candidate solution in original space */
static void check_solved(const prob_t *pr, const double *x, const double *s, const double *z, solcheck_t *ck)
{
    const problem_t *P = pr->orig;
    int n = P->n, m = P->m;
    double *Ax = DALLOC(m), *Px = DALLOC(n), *Atz = DALLOC(n), *r = DALLOC(m > n ? m : n);
    csc_mv(n, P->Ap, P->Ai, P->Ax, x, Ax, 1.0);
    csc_mv(n, P->Pp, P->Pi, P->Px, x, Px, 1.0);
    csc_mtv(n, P->Ap, P->Ai, P->Ax, z, Atz, 1.0);
    for (int i = 0; i < m; i++) r[i] = Ax[i] + s[i] - P->b[i];
    double den = nrminf(Ax, m); double t = nrminf(s, m); if (t > den) den = t; t = nrminf(P->b, m); if (t > den) den = t;
    ck->rp = nrminf(r, m) / (1.0 + den);
    for (int j = 0; j < n; j++) r[j] = Px[j] + P->q[j] + Atz[j];
    den = nrminf(Px, n); t = nrminf(P->q, n); if (t > den) den = t; t = nrminf(Atz, n); if (t > den) den = t;
    ck->rd = nrminf(r, n) / (1.0 + den);
    double xPx = dot(x, Px, n), qx = dot(P->q, x, n), bz = dot(P->b, z, m);
    double p0 = 0.5 * xPx + qx, d0 = -0.5 * xPx - bz;
    ck->p0 = p0; ck->d0 = d0;
    ck->gap = fabs(p0 - d0);
    double sc = fabs(p0) > fabs(d0) ? fabs(p0) : fabs(d0);
    ck->gaprel = sc > 0 ? ck->gap / sc : 0.0;
    ck->conep = cone_violation(pr, s, 0);
    ck->coned = cone_violation(pr, z, 1);
    double g = ck->gap < ck->gaprel ? ck->gap : ck->gaprel;
    double sco = ck->rp;
    if (ck->rd > sco) sco = ck->rd;
    if (g > sco) sco = g;
    if (ck->conep > sco) sco = ck->conep;
    if (ck->coned > sco) sco = ck->coned;
    if (!isfinite(sco)) sco = 1e300;
    ck->score = sco;
    free(Ax); free(Px); free(Atz); free(r);
}

/* primal infeasibility certificate score (<= tol passes); z in original space */
static double check_pinf(const prob_t *pr, const double *zin)
{
    const problem_t *P = pr->orig;
    int n = P->n, m = P->m;
    double nz = nrminf(zin, m), nb = nrminf(P->b, m);
    if (!(nz > 0) || !(nb > 0)) return 1e300;
    double *z = DALLOC(m);
    for (int i = 0; i < m; i++) z[i] = zin[i] / nz;
    double beta = 0;
    for (int i = 0; i < m; i++) beta -= (P->b[i] / nb) * z[i];
    double res = 1e300;
    if (beta > 1e-8) {
        double *Atz = DALLOC(n);
        double *Asc = NULL; (void)Asc;
        csc_mtv(n, P->Ap, P->Ai, P->Ax, z, Atz, 1.0);
        double lin = nrminf(Atz, n) / pr->Amax / beta;
        double cv = cone_violation(pr, z, 1) / beta;
        res = lin > cv ? lin : cv;
        free(Atz);
    }
    free(z);
    return isfinite(res) ? res : 1e300;
}

static double check_dinf(const prob_t *pr, const double *xin, const double *sin)
{
    const problem_t *P = pr->orig;
    int n = P->n, m = P->m;
    double nx = nrminf(xin, n), nq = nrminf(P->q, n);
    if (!(nx > 0) || !(nq > 0)) return 1e300;
    double *x = DALLOC(n), *s = DALLOC(m);
    for (int j = 0; j < n; j++) x[j] = xin[j] / nx;
    for (int i = 0; i < m; i++) s[i] = sin[i] / nx;
    double alpha = 0;
    for (int j = 0; j < n; j++) alpha -= (P->q[j] / nq) * x[j];
    double res = 1e300;
    if (alpha > 1e-8) {
        double *Px = DALLOC(n), *Ax = DALLOC(m);
        csc_mv(n, P->Pp, P->Pi, P->Px, x, Px, 1.0);
        csc_mv(n, P->Ap, P->Ai, P->Ax, x, Ax, 1.0);
        double quad = nrminf(Px, n) / pr->Pmax / alpha;
        for (int i = 0; i < m; i++) { Ax[i] = Ax[i] / pr->Amax + s[i] / pr->Amax; }
        double lin = nrminf(Ax, m) / alpha;
        for (int i = 0; i < m; i++) s[i] /= pr->Amax;
        double cv = cone_violation(pr, s, 0) / alpha;
        res = quad;
        if (lin > res) res = lin;
        if (cv > res) res = cv;
        free(Px); free(Ax);
    }
    free(x); free(s);
    return isfinite(res) ? res : 1e300;
}

/* ---------------- setup & equilibration ---------------- */
static void setup(prob_t *pr, problem_t *P)
{
    int n = P->n, m = P->m;
    pr->orig = P; pr->n = n; pr->m = m;
    pr->nc = P->ncones;
    pr->K = (cone_t *)xcalloc(pr->nc, sizeof(cone_t));
    int row = 0; pr->nu = 0; pr->has_exp = 0;
    for (int k = 0; k < pr->nc; k++) {
        cone_t *c = &pr->K[k];
        c->type = P->cone_type[k];
        c->start = row;
        if (c->type == CONE_PSD) { c->ord = P->cone_dim[k]; c->dim = c->ord * (c->ord + 1) / 2; }
        else { c->dim = P->cone_dim[k]; c->ord = 0; }
        row += c->dim;
        switch (c->type) {
        case CONE_ZERO: break;
        case CONE_NONNEG: pr->nu += c->dim; break;
        case CONE_SOC: pr->nu += 1; break;
        case CONE_EXP: pr->nu += 3; pr->has_exp = 1; break;
        case CONE_PSD: pr->nu += c->ord; break;
        }
        c->dense = (c->type == CONE_EXP) || (c->type == CONE_PSD) || (c->type == CONE_SOC && c->dim <= SOC_DENSE_MAX);
        c->expand = (c->type == CONE_SOC && c->dim > SOC_DENSE_MAX);
        c->wsc = (c->type == CONE_SOC && !c->expand && !getenv("CONIC_NOWSC"));
        if (c->type == CONE_PSD) {
            size_t d2 = (size_t)c->ord * c->ord;
            c->R = DALLOC(d2); c->Rinv = DALLOC(d2); c->G = DALLOC(d2);
        }
    }
    /* copy matrices */
    int nnzP = P->Pp[n], nnzA = P->Ap[n];
    pr->Pp = IALLOC(n + 1); pr->Pi = IALLOC(nnzP); pr->Px = DALLOC(nnzP);
    memcpy(pr->Pp, P->Pp, sizeof(int) * (n + 1)); memcpy(pr->Pi, P->Pi, sizeof(int) * nnzP); memcpy(pr->Px, P->Px, sizeof(double) * nnzP);
    pr->Ap = IALLOC(n + 1); pr->Ai = IALLOC(nnzA); pr->Ax = DALLOC(nnzA);
    memcpy(pr->Ap, P->Ap, sizeof(int) * (n + 1)); memcpy(pr->Ai, P->Ai, sizeof(int) * nnzA); memcpy(pr->Ax, P->Ax, sizeof(double) * nnzA);
    pr->q = DALLOC(n); memcpy(pr->q, P->q, sizeof(double) * n);
    pr->b = DALLOC(m); memcpy(pr->b, P->b, sizeof(double) * m);
    pr->D = DALLOC(n); pr->E = DALLOC(m);
    for (int j = 0; j < n; j++) pr->D[j] = 1.0;
    for (int i = 0; i < m; i++) pr->E[i] = 1.0;
    pr->Amax = nrminf(P->Ax, nnzA); if (pr->Amax == 0) pr->Amax = 1.0;
    pr->Pmax = nrminf(P->Px, nnzP); if (pr->Pmax == 0) pr->Pmax = 1.0;

    /* Ruiz equilibration */
    double *dw = DALLOC(n), *ew = DALLOC(m);
    for (int it = 0; it < 10; it++) {
        for (int j = 0; j < n; j++) {
            double mx = 0;
            for (int k = pr->Pp[j]; k < pr->Pp[j + 1]; k++) { double a = fabs(pr->Px[k]); if (a > mx) mx = a; }
            for (int k = pr->Ap[j]; k < pr->Ap[j + 1]; k++) { double a = fabs(pr->Ax[k]); if (a > mx) mx = a; }
            dw[j] = mx;
        }
        for (int i = 0; i < m; i++) ew[i] = 0;
        for (int j = 0; j < n; j++)
            for (int k = pr->Ap[j]; k < pr->Ap[j + 1]; k++) { double a = fabs(pr->Ax[k]); if (a > ew[pr->Ai[k]]) ew[pr->Ai[k]] = a; }
        double maxdev = 0;
        for (int j = 0; j < n; j++) {
            double v = dw[j]; v = (v < 1e-4) ? 1.0 : v; /* empty columns untouched */
            v = 1.0 / sqrt(v); if (v < 1e-4) v = 1e-4; if (v > 1e4) v = 1e4;
            dw[j] = v; if (fabs(1 - v) > maxdev) maxdev = fabs(1 - v);
        }
        for (int i = 0; i < m; i++) {
            double v = ew[i]; v = (v < 1e-4) ? 1.0 : v;
            v = 1.0 / sqrt(v); if (v < 1e-4) v = 1e-4; if (v > 1e4) v = 1e4;
            ew[i] = v;
        }
        for (int k = 0; k < pr->nc; k++) {
            cone_t *c = &pr->K[k];
            if (c->type == CONE_SOC || c->type == CONE_EXP || c->type == CONE_PSD) {
                double s = 0;
                for (int i = 0; i < c->dim; i++) s += ew[c->start + i];
                s /= c->dim;
                for (int i = 0; i < c->dim; i++) ew[c->start + i] = s;
            }
        }
        for (int i = 0; i < m; i++) if (fabs(1 - ew[i]) > maxdev) maxdev = fabs(1 - ew[i]);
        for (int j = 0; j < n; j++) {
            for (int k = pr->Pp[j]; k < pr->Pp[j + 1]; k++) pr->Px[k] *= dw[j] * dw[pr->Pi[k]];
            for (int k = pr->Ap[j]; k < pr->Ap[j + 1]; k++) pr->Ax[k] *= dw[j] * ew[pr->Ai[k]];
            pr->D[j] *= dw[j];
        }
        for (int i = 0; i < m; i++) pr->E[i] *= ew[i];
        if (maxdev < 1e-3) break;
    }
    for (int j = 0; j < n; j++) pr->q[j] *= pr->D[j];
    for (int i = 0; i < m; i++) pr->b[i] *= pr->E[i];
    /* cost scaling */
    double meanP = 0;
    for (int j = 0; j < n; j++) {
        double mx = 0;
        for (int k = pr->Pp[j]; k < pr->Pp[j + 1]; k++) { double a = fabs(pr->Px[k]); if (a > mx) mx = a; }
        meanP += mx;
    }
    meanP = n > 0 ? meanP / n : 0;
    double qn = nrminf(pr->q, n);
    double sc = meanP > qn ? meanP : qn;
    if (sc < 1e-4) sc = 1.0;
    if (sc > 1e4) sc = 1e4;
    if (sc < 1e-4) sc = 1e-4;
    pr->c = 1.0 / sc;
    for (int k = 0; k < pr->Pp[n]; k++) pr->Px[k] *= pr->c;
    for (int j = 0; j < n; j++) pr->q[j] *= pr->c;
    free(dw); free(ew);
    /* CSR of A */
    pr->Rp = IALLOC(m + 1); pr->Rj = IALLOC(nnzA); pr->Rx = DALLOC(nnzA);
    for (int k = 0; k < nnzA; k++) pr->Rp[pr->Ai[k] + 1]++;
    for (int i = 0; i < m; i++) pr->Rp[i + 1] += pr->Rp[i];
    int *nx = IALLOC(m); memcpy(nx, pr->Rp, sizeof(int) * m);
    for (int j = 0; j < n; j++)
        for (int k = pr->Ap[j]; k < pr->Ap[j + 1]; k++) { int p = nx[pr->Ai[k]]++; pr->Rj[p] = j; pr->Rx[p] = pr->Ax[k]; }
    free(nx);
    pr->wsoc = DALLOC(m); pr->lam = DALLOC(m);
}


/* ---------------- dense LDL' (OpenBLAS dgemm via dlopen) ---------------- */
typedef void (*dgemm_t)(const char *, const char *, const int *, const int *, const int *, const double *,
                        const double *, const int *, const double *, const int *, const double *, double *, const int *, size_t, size_t);
static dgemm_t p_dgemm = NULL;
static void load_blas(void)
{
    if (getenv("CONIC_NODENSE")) return;
    const char *dirs[] = { "/usr/local/lib/python3.12/site-packages/scipy.libs", NULL };
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    setenv("OMP_NUM_THREADS", "1", 1);
    for (int d = 0; dirs[d]; d++) {
        DIR *dir = opendir(dirs[d]);
        if (!dir) continue;
        struct dirent *e;
        while ((e = readdir(dir))) {
            if (strncmp(e->d_name, "libscipy_openblas-", 18) == 0 && strstr(e->d_name, ".so")) {
                char path[1024];
                snprintf(path, sizeof(path), "%s/%s", dirs[d], e->d_name);
                void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
                if (h) { p_dgemm = (dgemm_t)dlsym(h, "scipy_dgemm_"); if (p_dgemm) break; }
            }
        }
        closedir(dir);
        if (p_dgemm) break;
    }
}

/* in-place LDL' of column-major lower triangle A (N x N), no pivoting, Clarabel-style dynamic reg */
static int dense_ldl(int N, double *A, double *D, const signed char *sign, double eps, double delta)
{
    const int NB = 64;
    int nreg = 0;
    for (int k0 = 0; k0 < N; k0 += NB) {
        int nb = (N - k0 < NB) ? N - k0 : NB;
        for (int j = k0; j < k0 + nb; j++) {
            double *cj = A + (size_t)j * N;
            double dj = cj[j];
            if (!isfinite(dj)) return -1;
            if (sign[j] * dj <= eps) { dj = sign[j] * delta; nreg++; }
            D[j] = dj;
            double inv = 1.0 / dj;
            for (int i = j + 1; i < N; i++) cj[i] *= inv;
            /* update remaining columns of the block */
            for (int c = j + 1; c < k0 + nb; c++) {
                double f = cj[c] * dj;
                if (f == 0.0) continue;
                double *cc = A + (size_t)c * N;
                for (int i = c; i < N; i++) cc[i] -= cj[i] * f;
            }
        }
        int r0 = k0 + nb, m2 = N - r0;
        if (m2 > 0) {
            double *W = (double *)malloc(sizeof(double) * (size_t)m2 * nb);
            for (int j = 0; j < nb; j++) {
                double dj = D[k0 + j];
                const double *src = A + (size_t)(k0 + j) * N + r0;
                for (int i = 0; i < m2; i++) W[i + (size_t)j * m2] = src[i] * dj;
            }
            double mone = -1.0, one = 1.0;
            int lda = N;
            /* lower-trapezoidal update, one block column at a time */
            for (int c0 = 0; c0 < m2; c0 += 256) {
                int nc2 = (m2 - c0 < 256) ? m2 - c0 : 256;
                int mr = m2 - c0;
                p_dgemm("N", "T", &mr, &nc2, &nb, &mone, W + c0, &m2, A + (size_t)k0 * N + r0 + c0, &lda, &one,
                        A + (size_t)(r0 + c0) * N + r0 + c0, &lda, 1, 1);
            }
            free(W);
        }
    }
    return nreg;
}
static void dense_solve(int N, const double *A, const double *D, double *x)
{
    for (int j = 0; j < N; j++) { double xj = x[j]; if (xj == 0) continue; const double *cj = A + (size_t)j * N; for (int i = j + 1; i < N; i++) x[i] -= cj[i] * xj; }
    for (int j = 0; j < N; j++) x[j] /= D[j];
    for (int j = N - 1; j >= 0; j--) { const double *cj = A + (size_t)j * N; double s = 0; for (int i = j + 1; i < N; i++) s += cj[i] * x[i]; x[j] -= s; }
}

/* ---------------- KKT ---------------- */
typedef struct {
    int N, nnz;
    int *Kp, *Ki; double *Kx;
    double *reg;
    signed char *sign;
    int *hpos;     /* for cone row i: position of first H-block entry in column n+i */
    int *xdiag;    /* position of diagonal of x columns */
    ldl_t *f;
    int dense; double *DA, *DD;
    double *r, *dx, *work;
    double delta;
} kkt_t;

static int kkt_build(prob_t *pr, kkt_t *kk)
{
    int n = pr->n, m = pr->m;
    int nexp = 0;
    for (int k = 0; k < pr->nc; k++) if (pr->K[k].expand) { pr->K[k].pq = n + m + 2 * nexp; nexp++; }
    int N = n + m + 2 * nexp;
    kk->N = N;
    /* count */
    long cnt = 0;
    int *Kp = IALLOC(N + 1);
    for (int j = 0; j < n; j++) {
        int c = 0;
        for (int k = pr->Pp[j]; k < pr->Pp[j + 1]; k++) if (pr->Pi[k] < j) c++;
        Kp[j + 1] = c + 1;
    }
    for (int k = 0; k < pr->nc; k++) {
        cone_t *c = &pr->K[k];
        if (c->wsc) {
            /* union of the block rows' column patterns */
            int tot = 0;
            for (int li = 0; li < c->dim; li++) tot += pr->Rp[c->start + li + 1] - pr->Rp[c->start + li];
            c->ucols = IALLOC(tot);
            int nu = 0;
            for (int li = 0; li < c->dim; li++) for (int t = pr->Rp[c->start + li]; t < pr->Rp[c->start + li + 1]; t++) c->ucols[nu++] = pr->Rj[t];
            /* sort + unique (small) */
            for (int a1 = 1; a1 < nu; a1++) { int key = c->ucols[a1], b1 = a1 - 1; while (b1 >= 0 && c->ucols[b1] > key) { c->ucols[b1 + 1] = c->ucols[b1]; b1--; } c->ucols[b1 + 1] = key; }
            int u2 = 0; for (int a1 = 0; a1 < nu; a1++) if (u2 == 0 || c->ucols[u2 - 1] != c->ucols[a1]) c->ucols[u2++] = c->ucols[a1];
            c->nu_cols = u2;
        }
        for (int li = 0; li < c->dim; li++) {
            int i = c->start + li;
            int na = c->wsc ? c->nu_cols : pr->Rp[i + 1] - pr->Rp[i];
            Kp[n + i + 1] = na + (c->dense ? li + 1 : 1);
        }
        if (c->expand) { Kp[c->pq + 1] = c->dim + 1; Kp[c->pq + 2] = c->dim + 1; }
    }
    for (int j = 0; j < N; j++) { cnt += Kp[j + 1]; }
    if (cnt > 2000000000L) return -1;
    for (int j = 0; j < N; j++) Kp[j + 1] += Kp[j];
    int nnz = Kp[N];
    kk->nnz = nnz;
    int *Ki = IALLOC(nnz); double *Kx = DALLOC(nnz);
    kk->hpos = IALLOC(m > 0 ? m : 1); kk->xdiag = IALLOC(n > 0 ? n : 1);
    kk->reg = DALLOC(N); kk->sign = (signed char *)xcalloc(N, 1);
    double delta = 1e-8;
    kk->delta = delta;
    for (int j = 0; j < n; j++) {
        int p = Kp[j]; double dg = 0;
        for (int k = pr->Pp[j]; k < pr->Pp[j + 1]; k++) {
            if (pr->Pi[k] < j) { Ki[p] = pr->Pi[k]; Kx[p] = pr->Px[k]; p++; }
            else if (pr->Pi[k] == j) dg = pr->Px[k];
        }
        Ki[p] = j; Kx[p] = dg + delta; kk->xdiag[j] = p;
        kk->reg[j] = delta; kk->sign[j] = 1;
    }
    for (int k = 0; k < pr->nc; k++) {
        cone_t *c = &pr->K[k];
        for (int li = 0; li < c->dim; li++) {
            int i = c->start + li, col = n + i, p = Kp[col];
            if (c->wsc) { for (int t = 0; t < c->nu_cols; t++) { Ki[p] = c->ucols[t]; Kx[p] = 0.0; p++; } }
            else for (int t = pr->Rp[i]; t < pr->Rp[i + 1]; t++) { Ki[p] = pr->Rj[t]; Kx[p] = pr->Rx[t]; p++; }
            kk->hpos[i] = p;
            if (c->dense) { for (int r = 0; r <= li; r++) { Ki[p] = n + c->start + r; Kx[p] = (r == li) ? -1.0 : 0.0; p++; } }
            else { Ki[p] = col; Kx[p] = -1.0; p++; }
            kk->reg[col] = -delta; kk->sign[col] = -1;
        }
        if (c->expand) {
            for (int t = 0; t < 2; t++) {
                int col = c->pq + t, p = Kp[col];
                for (int r = 0; r < c->dim; r++) { Ki[p] = n + c->start + r; Kx[p] = 0.0; p++; }
                Ki[p] = col; Kx[p] = t == 0 ? 1.0 + delta : -1.0 - delta;
                kk->reg[col] = t == 0 ? delta : -delta; kk->sign[col] = t == 0 ? 1 : -1;
            }
        }
    }
    kk->Kp = Kp; kk->Ki = Ki; kk->Kx = Kx;
    double t0 = wtime();
    kk->f = ldl_symbolic(N, Kp, Ki, NULL);
    if (!kk->f) return -1;
    LOG("kkt N=%d nnz=%d nnzL=%ld symbolic %.3fs\n", N, nnz, ldl_nnz_L(kk->f), wtime() - t0);
    kk->dense = 0;
    if (p_dgemm && N >= 200 && N <= 6000 && (double)N < 3.0 * sqrt(2.0 * (double)ldl_nnz_L(kk->f))) {
        kk->DA = (double *)malloc(sizeof(double) * (size_t)N * N);
        kk->DD = DALLOC(N);
        if (kk->DA) { kk->dense = 1; LOG("dense KKT path\n"); }
    }
    kk->r = DALLOC(N); kk->dx = DALLOC(N); kk->work = DALLOC(N);
    return 0;
}

static void kkt_set_delta(kkt_t *kk, prob_t *pr, double nd)
{
    double od = kk->delta;
    for (int j = 0; j < pr->n; j++) kk->Kx[kk->xdiag[j]] += nd - od;
    for (int k = 0; k < pr->nc; k++) if (pr->K[k].expand) {
        int c = pr->K[k].pq;
        kk->Kx[kk->Kp[c + 1] - 1] = 1.0 + nd;
        kk->Kx[kk->Kp[c + 2] - 1] = -1.0 - nd;
    }
    for (int i = 0; i < kk->N; i++) kk->reg[i] = kk->sign[i] > 0 ? nd : -nd;
    kk->delta = nd;
}
static int kkt_factor(kkt_t *kk)
{
    if (!kk->dense) return ldl_numeric(kk->f, kk->Kx, kk->sign, 1e-13, 2e-7);
    int N = kk->N;
    double *A = kk->DA;
    memset(A, 0, sizeof(double) * (size_t)N * N);
    for (int j = 0; j < N; j++)
        for (int k = kk->Kp[j]; k < kk->Kp[j + 1]; k++) {
            int i = kk->Ki[k];
            /* upper entry (i,j), i <= j  -> lower (j,i) */
            A[j + (size_t)i * N] += kk->Kx[k];
        }
    return dense_ldl(N, A, kk->DD, kk->sign, 1e-13, 2e-7);
}
static void kkt_lsolve(kkt_t *kk, double *x)
{
    if (kk->dense) dense_solve(kk->N, kk->DA, kk->DD, x);
    else ldl_solve(kk->f, x);
}
/* y = K x (true matrix, i.e. without static regularization) */
static void kkt_mv(const kkt_t *kk, const double *x, double *y)
{
    int N = kk->N;
    for (int i = 0; i < N; i++) y[i] = -kk->reg[i] * x[i];
    for (int j = 0; j < N; j++) {
        double xj = x[j], s = 0;
        for (int k = kk->Kp[j]; k < kk->Kp[j + 1]; k++) {
            int i = kk->Ki[k]; double v = kk->Kx[k];
            if (i == j) s += v * xj;
            else { y[i] += v * xj; s += v * x[i]; }
        }
        y[j] += s;
    }
}

/* solve K x = b with iterative refinement; b preserved, result in x */
static void kkt_solve(kkt_t *kk, const double *b, double *x)
{
    int N = kk->N;
    memcpy(x, b, sizeof(double) * N);
    kkt_lsolve(kk, x);
    double bn = nrminf(b, N);
    double lastn = 1e300;
    for (int it = 0; it < 10; it++) {
        kkt_mv(kk, x, kk->r);
        for (int i = 0; i < N; i++) kk->r[i] = b[i] - kk->r[i];
        double rn = nrminf(kk->r, N);
        if (!(rn > 1e-12 + 1e-13 * bn)) break;
        if (it > 0 && lastn / rn < 2.0) {
            if (rn > lastn) { /* got worse: undo */
                for (int i = 0; i < N; i++) x[i] -= kk->dx[i];
            }
            break;
        }
        lastn = rn;
        if (verbose > 1) fprintf(stderr, "   ref it %d rn %.3e bn %.3e\n", it, rn, bn);
        memcpy(kk->dx, kk->r, sizeof(double) * N);
        kkt_lsolve(kk, kk->dx);
        for (int i = 0; i < N; i++) x[i] += kk->dx[i];
    }
}

/* ---------------- cone kernels ---------------- */
static inline int pidx(int i, int j) { return j * (j + 1) / 2 + i; } /* i <= j */

static void smat(int d, const double *v, double *M)
{
    int p = 0;
    for (int j = 0; j < d; j++) for (int i = 0; i <= j; i++) {
        double a = v[p++]; if (i < j) a *= M_SQRT1_2;
        M[i + (size_t)j * d] = a; M[j + (size_t)i * d] = a;
    }
}
static void svec(int d, const double *M, double *v)
{
    int p = 0;
    for (int j = 0; j < d; j++) for (int i = 0; i <= j; i++) {
        double a = 0.5 * (M[i + (size_t)j * d] + M[j + (size_t)i * d]);
        v[p++] = i < j ? a * M_SQRT2 : a;
    }
}
/* C = op(A) * op(B), column-major d x d; ta/tb transpose flags */
static void mm(int d, const double *A, int ta, const double *B, int tb, double *C)
{
    memset(C, 0, sizeof(double) * (size_t)d * d);
    for (int j = 0; j < d; j++) {
        for (int k = 0; k < d; k++) {
            double bkj = tb ? B[j + (size_t)k * d] : B[k + (size_t)j * d];
            if (bkj == 0.0) continue;
            double *cj = C + (size_t)j * d;
            if (!ta) { const double *ak = A + (size_t)k * d; for (int i = 0; i < d; i++) cj[i] += ak[i] * bkj; }
            else { for (int i = 0; i < d; i++) cj[i] += A[k + (size_t)i * d] * bkj; }
        }
    }
}

static double soc_res(const double *x, int d)
{
    double s = 0; for (int i = 1; i < d; i++) s += x[i] * x[i];
    double nr = sqrt(s);
    return (x[0] - nr) * (x[0] + nr);
}
/* y = Wbar x */
static void soc_wbar(const double *w, int d, const double *x, double *y, int inv)
{
    double t = 0; for (int i = 1; i < d; i++) t += w[i] * x[i];
    double w0 = w[0];
    double x0 = x[0];
    if (!inv) {
        y[0] = w0 * x0 + t;
        double cf = x0 + t / (1 + w0);
        for (int i = 1; i < d; i++) y[i] = x[i] + cf * w[i];
    } else {
        y[0] = w0 * x0 - t;
        double cf = -x0 + t / (1 + w0);
        for (int i = 1; i < d; i++) y[i] = x[i] + cf * w[i];
    }
}
/* jordan product */
static void soc_circ(const double *u, const double *v, int d, double *y)
{
    double y0 = dot(u, v, d);
    for (int i = 1; i < d; i++) y[i] = u[0] * v[i] + v[0] * u[i];
    y[0] = y0;
}
/* y = lam \ r */
static void soc_div(const double *l, const double *r, int d, double *y)
{
    double l1r1 = 0, l1l1 = 0;
    for (int i = 1; i < d; i++) { l1r1 += l[i] * r[i]; l1l1 += l[i] * l[i]; }
    double det = l[0] * l[0] - l1l1;
    double x0 = (l[0] * r[0] - l1r1) / det;
    for (int i = 1; i < d; i++) y[i] = (r[i] - x0 * l[i]) / l[0];
    y[0] = x0;
}

static double soc_step(const double *x, const double *dx, int d)
{
    double a = dx[0] * dx[0], b = x[0] * dx[0], c = x[0] * x[0];
    for (int i = 1; i < d; i++) { a -= dx[i] * dx[i]; b -= x[i] * dx[i]; c -= x[i] * x[i]; }
    if (c < 0) c = 0;
    double disc = b * b - a * c;
    if (a >= 0 && (b >= 0 || disc < 0)) return 1e300;
    if (disc < 0) disc = 0;
    double den = sqrt(disc) - b;
    if (den <= 0) return 0;
    return c / den;
}

/* exp cone dual barrier pieces */
static int exp_dual_in(const double *z)
{
    double u = z[0], v = z[1], w = z[2];
    if (!(u < 0) || !(w > 0)) return 0;
    double psi = v - u - u * log(-w / u);
    return psi > 0 && isfinite(psi);
}
static int exp_primal_in(const double *s)
{
    double u = s[0], v = s[1], w = s[2];
    if (!(v > 0) || !(w > 0)) return 0;
    double r = v * log(w / v) - u;
    return r > 0 && isfinite(r);
}
static void exp_dual_gh(const double *z, double *g, double *H)
{
    double u = z[0], v = z[1], w = z[2];
    double l = log(-w / u);
    double psi = v - u - u * l;
    g[0] = -1.0 / u + l / psi;
    g[1] = -1.0 / psi;
    g[2] = -1.0 / w + (u / w) / psi;
    if (H) {
        double p2 = psi * psi;
        H[0] = 1.0 / (u * u) - 1.0 / (u * psi) + l * l / p2;
        H[1] = -l / p2;
        H[2] = 1.0 / (w * psi) + l * u / (w * p2);
        H[4] = 1.0 / p2;
        H[5] = -u / (w * p2);
        H[8] = 1.0 / (w * w) - u / (w * w * psi) + u * u / (w * w * p2);
        H[3] = H[1]; H[6] = H[2]; H[7] = H[5];
    }
}

/* apply W^{-1} (scaled-KKT SOC blocks) to the constraint part of a KKT vector */
static void wsc_transform(prob_t *pr, double *vz)
{
    for (int k = 0; k < pr->nc; k++) {
        cone_t *c = &pr->K[k];
        if (!c->wsc) continue;
        int d = c->dim; double tmp[SOC_DENSE_MAX + 1];
        soc_wbar(pr->wsoc + c->start, d, vz + c->start, tmp, 1);
        for (int i = 0; i < d; i++) vz[c->start + i] = tmp[i] / c->eta;
    }
}
/* ---------------- solver state ---------------- */
typedef struct {
    double *x, *z, *s; double tau, kap;
} iter_t;

static void iter_alloc(iter_t *v, int n, int m) { v->x = DALLOC(n); v->z = DALLOC(m); v->s = DALLOC(m); v->tau = v->kap = 1; }

/* update scaling; returns 0 on success */
static int update_scaling(prob_t *pr, iter_t *v, double mu)
{
    for (int k = 0; k < pr->nc; k++) {
        cone_t *c = &pr->K[k];
        int d = c->dim;
        double *s = v->s + c->start, *z = v->z + c->start;
        if (c->type == CONE_SOC) {
            double sr = soc_res(s, d), zr = soc_res(z, d);
            if (!(sr > 0) || !(zr > 0)) return -1;
            double ss = sqrt(sr), zs = sqrt(zr);
            double *w = pr->wsoc + c->start;
            double sz = 0;
            for (int i = 0; i < d; i++) sz += (s[i] / ss) * (z[i] / zs);
            double gam = sqrt(0.5 * (1 + sz));
            w[0] = (s[0] / ss + z[0] / zs) / (2 * gam);
            for (int i = 1; i < d; i++) w[i] = (s[i] / ss - z[i] / zs) / (2 * gam);
            /* renormalize w0 */
            double w1 = 0; for (int i = 1; i < d; i++) w1 += w[i] * w[i];
            w[0] = sqrt(1 + w1);
            c->eta = sqrt(ss / zs);
            double *lam = pr->lam + c->start;
            /* stable lambda (CVXOPT): sqrt(ss*zs) * [gam; ((gam+z0)s1 + (gam+s0)z1)/(s0+z0+2gam)] */
            {
                double s0 = s[0] / ss, z0 = z[0] / zs, sc = sqrt(ss * zs);
                double dd = s0 + z0 + 2 * gam;
                lam[0] = sc * gam;
                for (int i = 1; i < d; i++) lam[i] = sc * ((gam + z0) * (s[i] / ss) + (gam + s0) * (z[i] / zs)) / dd;
            }
            if (c->expand) {
                double a = w1;
                double U = 2 * a + 1, L = 4 * a * (1 + a) / (2 * a + 1);
                double u02 = 0.5 * (L + U);
                c->u0 = sqrt(u02);
                c->d0 = 0.5 / (2 * a + 1);
                c->u1 = (a > 0) ? 2 * w[0] * sqrt(a) / c->u0 : 0;
                double v12 = c->u1 * c->u1 - 2 * a;
                c->v1 = v12 > 0 ? sqrt(v12) : 0;
            }
        } else if (c->type == CONE_EXP) {
            if (!exp_dual_in(z) || !exp_primal_in(s)) return -1;
            exp_dual_gh(z, c->g, c->H);
            for (int i = 0; i < 9; i++) c->H[i] *= mu;
        } else if (c->type == CONE_PSD) {
            int o = c->ord, info = 0;
            size_t d2 = (size_t)o * o;
            double *Ls = DALLOC(d2), *Lz = DALLOC(d2), *M = DALLOC(d2), *U = DALLOC(d2), *VT = DALLOC(d2), *sv = DALLOC(o);
            smat(o, s, Ls); smat(o, z, Lz);
            dpotrf_("L", &o, Ls, &o, &info, 1);
            if (info) { free(Ls); free(Lz); free(M); free(U); free(VT); free(sv); return -1; }
            dpotrf_("L", &o, Lz, &o, &info, 1);
            if (info) { free(Ls); free(Lz); free(M); free(U); free(VT); free(sv); return -1; }
            for (int j = 0; j < o; j++) for (int i = 0; i < j; i++) { Ls[i + (size_t)j * o] = 0; Lz[i + (size_t)j * o] = 0; }
            mm(o, Lz, 1, Ls, 0, M); /* Lz' Ls */
            int lwork = -1; double wq;
            dgesvd_("A", "A", &o, &o, M, &o, sv, U, &o, VT, &o, &wq, &lwork, &info, 1, 1);
            lwork = (int)wq + 1;
            double *work = DALLOC(lwork);
            dgesvd_("A", "A", &o, &o, M, &o, sv, U, &o, VT, &o, work, &lwork, &info, 1, 1);
            free(work);
            if (info) { free(Ls); free(Lz); free(M); free(U); free(VT); free(sv); return -1; }
            double *lam = pr->lam + c->start;
            for (int i = 0; i < o; i++) { if (!(sv[i] > 0)) { free(Ls); free(Lz); free(M); free(U); free(VT); free(sv); return -1; } lam[i] = sv[i]; }
            /* R = Ls V diag(sv^-1/2); V = VT' */
            for (int j = 0; j < o; j++) {
                double f = 1.0 / sqrt(sv[j]);
                for (int i = 0; i < o; i++) {
                    double acc = 0;
                    for (int k2 = 0; k2 < o; k2++) acc += Ls[i + (size_t)k2 * o] * VT[j + (size_t)k2 * o];
                    c->R[i + (size_t)j * o] = acc * f;
                }
            }
            /* Rinv = diag(sv^-1/2) U' Lz' */
            for (int j = 0; j < o; j++) for (int i = 0; i < o; i++) {
                double acc = 0;
                for (int k2 = 0; k2 < o; k2++) acc += U[k2 + (size_t)i * o] * Lz[j + (size_t)k2 * o];
                c->Rinv[i + (size_t)j * o] = acc / sqrt(sv[i]);
            }
            mm(o, c->R, 0, c->R, 1, c->G);
            free(Ls); free(Lz); free(M); free(U); free(VT); free(sv);
        }
    }
    return 0;
}

/* write -H (and -reg) into KKT values */
static void fill_kkt(prob_t *pr, kkt_t *kk, iter_t *v, int unitH)
{
    int n = pr->n;
    double dl = kk->delta;
    for (int k = 0; k < pr->nc; k++) {
        cone_t *c = &pr->K[k];
        int d = c->dim, st = c->start;
        double *Kx = kk->Kx;
        if (c->type == CONE_ZERO) {
            for (int i = 0; i < d; i++) Kx[kk->hpos[st + i]] = -dl;
        } else if (c->type == CONE_NONNEG) {
            for (int i = 0; i < d; i++) Kx[kk->hpos[st + i]] = -(unitH ? 1.0 : v->s[st + i] / v->z[st + i]) - dl;
        } else if (c->type == CONE_SOC && c->wsc) {
            double *w = pr->wsoc + st;
            int nu = c->nu_cols;
            double *M = DALLOC((size_t)d * nu), *col = DALLOC(d), *out = DALLOC(d);
            for (int li = 0; li < d; li++) {
                int i = st + li, t2 = 0;
                for (int t = pr->Rp[i]; t < pr->Rp[i + 1]; t++) {
                    while (c->ucols[t2] != pr->Rj[t]) t2++;
                    M[li + (size_t)d * t2] = pr->Rx[t];
                }
            }
            for (int t = 0; t < nu; t++) {
                for (int li = 0; li < d; li++) col[li] = M[li + (size_t)d * t];
                if (!unitH) { soc_wbar(w, d, col, out, 1); for (int li = 0; li < d; li++) out[li] /= c->eta; }
                else memcpy(out, col, sizeof(double) * d);
                for (int li = 0; li < d; li++) Kx[kk->Kp[n + st + li] + t] = out[li];
            }
            for (int j = 0; j < d; j++) for (int i = 0; i <= j; i++) Kx[kk->hpos[st + j] + i] = (i == j) ? -1.0 - dl : 0.0;
            free(M); free(col); free(out);
        } else if (c->type == CONE_SOC) {
            double *w = pr->wsoc + st;
            double e2 = c->eta * c->eta;
            if (c->dense) {
                for (int j = 0; j < d; j++) for (int i = 0; i <= j; i++) {
                    double h;
                    if (unitH) h = (i == j) ? 1.0 : 0.0;
                    else {
                        h = 2 * w[i] * w[j];
                        if (i == j) h += (i == 0) ? -1.0 : 1.0;
                        h *= e2;
                    }
                    Kx[kk->hpos[st + j] + i] = -h - (i == j ? dl : 0);
                }
            } else {
                double wn = 0; for (int i = 1; i < d; i++) wn += w[i] * w[i];
                wn = sqrt(wn);
                int pp = kk->Kp[c->pq], qp = kk->Kp[c->pq + 1];
                for (int i = 0; i < d; i++) {
                    double Di = (i == 0) ? c->d0 : 1.0;
                    if (unitH) Di = 1.0;
                    Kx[kk->hpos[st + i]] = -(unitH ? 1.0 : e2) * Di - dl;
                    double ui, vi;
                    if (unitH) { ui = 0; vi = 0; }
                    else if (i == 0) { ui = c->u0; vi = 0; }
                    else { double wh = wn > 0 ? w[i] / wn : 0; ui = c->u1 * wh; vi = c->v1 * wh; }
                    Kx[pp + i] = c->eta * ui * (unitH ? 0 : 1);
                    Kx[qp + i] = c->eta * vi * (unitH ? 0 : 1);
                }
                (void)n;
            }
        } else if (c->type == CONE_EXP) {
            for (int j = 0; j < 3; j++) for (int i = 0; i <= j; i++) {
                double h = unitH ? (i == j ? 1.0 : 0.0) : c->H[i + 3 * j];
                Kx[kk->hpos[st + j] + i] = -h - (i == j ? dl : 0);
            }
        } else { /* PSD */
            int o = c->ord;
            const double *G = c->G;
            for (int l = 0; l < o; l++) for (int kq = 0; kq <= l; kq++) {
                int col = pidx(kq, l);
                double bkl = (kq == l) ? 0.5 : M_SQRT1_2;
                int base = kk->hpos[st + col];
                for (int jj = 0; jj < o; jj++) for (int ii = 0; ii <= jj; ii++) {
                    int row = pidx(ii, jj);
                    if (row > col) continue;
                    double h;
                    if (unitH) h = (row == col) ? 1.0 : 0.0;
                    else {
                        double aij = (ii == jj) ? 1.0 : M_SQRT2;
                        h = aij * bkl * (G[ii + kq * o] * G[jj + l * o] + G[ii + l * o] * G[jj + kq * o]);
                    }
                    Kx[base + row] = -h - (row == col ? dl : 0);
                }
            }
        }
    }
}

/* y = H x for cone rows (scaled), dense per cone */
static void apply_H(prob_t *pr, iter_t *v, const double *x, double *y)
{
    for (int k = 0; k < pr->nc; k++) {
        cone_t *c = &pr->K[k];
        int d = c->dim, st = c->start;
        const double *xx = x + st; double *yy = y + st;
        if (c->type == CONE_ZERO) { for (int i = 0; i < d; i++) yy[i] = 0; }
        else if (c->type == CONE_NONNEG) { for (int i = 0; i < d; i++) yy[i] = v->s[st + i] / v->z[st + i] * xx[i]; }
        else if (c->type == CONE_SOC) {
            double *w = pr->wsoc + st;
            double t = dot(w, xx, d);
            double e2 = c->eta * c->eta;
            /* (2ww' - J) x */
            for (int i = 0; i < d; i++) yy[i] = e2 * (2 * w[i] * t + (i == 0 ? -xx[0] : xx[i]));
        } else if (c->type == CONE_EXP) {
            for (int i = 0; i < 3; i++) yy[i] = c->H[i] * xx[0] + c->H[i + 3] * xx[1] + c->H[i + 6] * xx[2];
        } else {
            int o = c->ord; size_t d2 = (size_t)o * o;
            double *X = DALLOC(d2), *T = DALLOC(d2), *Y = DALLOC(d2);
            smat(o, xx, X); mm(o, c->G, 0, X, 0, T); mm(o, T, 0, c->G, 0, Y); svec(o, Y, yy);
            free(X); free(T); free(Y);
        }
    }
}

/* d_s for the combined step (symmetric: Mehrotra; exp: s + sigma mu g) */
static void combined_ds(prob_t *pr, iter_t *v, const double *dsa, const double *dza, double sigmu, double *ds, int affine)
{
    for (int k = 0; k < pr->nc; k++) {
        cone_t *c = &pr->K[k];
        int d = c->dim, st = c->start;
        double *s = v->s + st, *z = v->z + st, *out = ds + st;
        if (c->type == CONE_ZERO) { for (int i = 0; i < d; i++) out[i] = 0; continue; }
        if (affine) { for (int i = 0; i < d; i++) out[i] = s[i]; continue; }
        if (c->type == CONE_NONNEG) {
            for (int i = 0; i < d; i++) out[i] = (s[i] * z[i] + dsa[st + i] * dza[st + i] - sigmu) / z[i];
        } else if (c->type == CONE_SOC) {
            double *w = pr->wsoc + st, *lam = pr->lam + st;
            double *a = DALLOC(d), *bb = DALLOC(d), *t = DALLOC(d), *u = DALLOC(d);
            soc_wbar(w, d, dsa + st, a, 1); for (int i = 0; i < d; i++) a[i] /= c->eta;   /* W^{-T} ds */
            soc_wbar(w, d, dza + st, bb, 0); for (int i = 0; i < d; i++) bb[i] *= c->eta; /* W dz */
            soc_circ(a, bb, d, t);
            soc_circ(lam, lam, d, u);
            for (int i = 0; i < d; i++) u[i] += t[i];
            u[0] -= sigmu;
            soc_div(lam, u, d, t);
            soc_wbar(w, d, t, out, 0); for (int i = 0; i < d; i++) out[i] *= c->eta;
            free(a); free(bb); free(t); free(u);
        } else if (c->type == CONE_EXP) {
            for (int i = 0; i < 3; i++) out[i] = s[i] + sigmu * c->g[i];
        } else {
            int o = c->ord; size_t d2 = (size_t)o * o;
            double *X = DALLOC(d2), *T = DALLOC(d2), *A1 = DALLOC(d2), *B1 = DALLOC(d2), *C1 = DALLOC(d2);
            double *lam = pr->lam + st;
            /* A1 = Rinv dS Rinv' */
            smat(o, dsa + st, X); mm(o, c->Rinv, 0, X, 0, T); mm(o, T, 0, c->Rinv, 1, A1);
            /* B1 = R' dZ R */
            smat(o, dza + st, X); mm(o, c->R, 1, X, 0, T); mm(o, T, 0, c->R, 0, B1);
            mm(o, A1, 0, B1, 0, C1);
            for (int j = 0; j < o; j++) for (int i = 0; i < o; i++) {
                double val = 0.5 * (C1[i + (size_t)j * o] + C1[j + (size_t)i * o]);
                if (i == j) val += lam[i] * lam[i] - sigmu;
                X[i + (size_t)j * o] = 2.0 * val / (lam[i] + lam[j]);
            }
            mm(o, c->R, 0, X, 0, T); mm(o, T, 0, c->R, 1, A1);
            svec(o, A1, out);
            free(X); free(T); free(A1); free(B1); free(C1);
        }
    }
}

static double psd_step2(cone_t *c, const double *lam, const double *dx, int isz)
{
    int o = c->ord; size_t d2 = (size_t)o * o;
    double *X = DALLOC(d2), *T = DALLOC(d2), *Y = DALLOC(d2), *w = DALLOC(o);
    smat(o, dx, X);
    if (isz) { mm(o, c->R, 1, X, 0, T); mm(o, T, 0, c->R, 0, Y); }
    else { mm(o, c->Rinv, 0, X, 0, T); mm(o, T, 0, c->Rinv, 1, Y); }
    for (int j = 0; j < o; j++) for (int i = 0; i < o; i++) Y[i + (size_t)j * o] /= sqrt(lam[i] * lam[j]);
    int lwork = -1, info = 0; double wq;
    dsyev_("N", "L", &o, Y, &o, w, &wq, &lwork, &info, 1, 1);
    lwork = (int)wq + 1;
    double *work = DALLOC(lwork);
    dsyev_("N", "L", &o, Y, &o, w, work, &lwork, &info, 1, 1);
    double r = 1e300;
    if (info == 0) { if (w[0] < 0) r = -1.0 / w[0]; }
    else r = 0;
    free(work); free(X); free(T); free(Y); free(w);
    return r;
}

/* maximum step for symmetric cones (and tau/kappa); exp handled by backtracking */
static double max_step(prob_t *pr, iter_t *v, const double *dx, const double *dz, const double *ds, double dtau, double dkap)
{
    double a = 1e300;
    (void)dx;
    if (dtau < 0) { double t = -v->tau / dtau; if (t < a) a = t; }
    if (dkap < 0) { double t = -v->kap / dkap; if (t < a) a = t; }
    for (int k = 0; k < pr->nc; k++) {
        cone_t *c = &pr->K[k];
        int d = c->dim, st = c->start;
        if (c->type == CONE_NONNEG) {
            for (int i = 0; i < d; i++) {
                if (ds[st + i] < 0) { double t = -v->s[st + i] / ds[st + i]; if (t < a) a = t; }
                if (dz[st + i] < 0) { double t = -v->z[st + i] / dz[st + i]; if (t < a) a = t; }
            }
        } else if (c->type == CONE_SOC) {
            double t = soc_step(v->s + st, ds + st, d); if (t < a) a = t;
            t = soc_step(v->z + st, dz + st, d); if (t < a) a = t;
        } else if (c->type == CONE_PSD) {
            double t = psd_step2(c, pr->lam + st, ds + st, 0); if (t < a) a = t;
            t = psd_step2(c, pr->lam + st, dz + st, 1); if (t < a) a = t;
        }
    }
    return a;
}

static int exp_ok(prob_t *pr, iter_t *v, const double *dz, const double *ds, double a, double mu_new, int central)
{
    for (int k = 0; k < pr->nc; k++) {
        cone_t *c = &pr->K[k];
        if (c->type != CONE_EXP) continue;
        int st = c->start;
        double s[3], z[3];
        for (int i = 0; i < 3; i++) { s[i] = v->s[st + i] + a * ds[st + i]; z[i] = v->z[st + i] + a * dz[st + i]; }
        if (!exp_primal_in(s) || !exp_dual_in(z)) return 0;
        if (central) {
            double sz = s[0] * z[0] + s[1] * z[1] + s[2] * z[2];
            if (sz < 0.1 * 3 * mu_new) return 0;
            /* neighborhood: || s + mu g(z) ||_{H^{-1}} <= beta mu */
            double g[3], H[9];
            exp_dual_gh(z, g, H);
            double r[3] = { s[0] + mu_new * g[0], s[1] + mu_new * g[1], s[2] + mu_new * g[2] };
            /* solve H y = r via Cholesky 3x3 */
            double l00 = H[0]; if (!(l00 > 0)) return 0; l00 = sqrt(l00);
            double l10 = H[1] / l00, l20 = H[2] / l00;
            double t11 = H[4] - l10 * l10; if (!(t11 > 0)) return 0; double l11 = sqrt(t11);
            double l21 = (H[5] - l20 * l10) / l11;
            double t22 = H[8] - l20 * l20 - l21 * l21; if (!(t22 > 0)) return 0; double l22 = sqrt(t22);
            double y0 = r[0] / l00, y1 = (r[1] - l10 * y0) / l11, y2 = (r[2] - l20 * y0 - l21 * y1) / l22;
            double nrm = sqrt(y0 * y0 + y1 * y1 + y2 * y2);
            if (nrm > 0.95 * mu_new * 10) return 0;
        }
    }
    return 1;
}

/* ---------------- main IPM ---------------- */
typedef struct { int status; double *x, *s, *z; } result_t; /* 0 unresolved, 1 solved, 2 pinf, 3 dinf */

static void unscale_sol(prob_t *pr, iter_t *v, double *x, double *s, double *z)
{
    int n = pr->n, m = pr->m;
    double t = v->tau;
    for (int j = 0; j < n; j++) x[j] = pr->D[j] * v->x[j] / t;
    for (int i = 0; i < m; i++) { s[i] = v->s[i] / pr->E[i] / t; z[i] = pr->E[i] * v->z[i] / (pr->c * t); }
    for (int k = 0; k < pr->nc; k++) if (pr->K[k].type == CONE_ZERO) for (int i = 0; i < pr->K[k].dim; i++) s[pr->K[k].start + i] = 0;
}

static void solve(prob_t *pr, result_t *res)
{
    int n = pr->n, m = pr->m;
    kkt_t kk; memset(&kk, 0, sizeof(kk));
    res->status = 0;
    res->x = DALLOC(n); res->s = DALLOC(m); res->z = DALLOC(m);
    if (kkt_build(pr, &kk)) { LOG("kkt build failed\n"); return; }
    int N = kk.N;
    iter_t v; iter_alloc(&v, n, m);
    double *rhs1 = DALLOC(N), *sol1 = DALLOC(N), *rhs2 = DALLOC(N), *sol2 = DALLOC(N);
    double *dx = DALLOC(n), *dz = DALLOC(m), *dsv = DALLOC(m);
    double *dxa = DALLOC(n), *dza = DALLOC(m), *dsa = DALLOC(m), *dsrhs = DALLOC(m);
    double *rx = DALLOC(n), *rz = DALLOC(m), *Px = DALLOC(n), *tmpm = DALLOC(m);
    double *ux = DALLOC(n), *us = DALLOC(m), *uz = DALLOC(m);
    double best = 1e300;
    double *bestx = DALLOC(n), *bests = DALLOC(m), *bestz = DALLOC(m);
    const double tol = 1e-8 * 0.95;
    int use_resds = getenv("CONIC_NORESDS") ? 0 : 1;

    /* ---- initialization ---- */
    if (!pr->has_exp) {
        fill_kkt(pr, &kk, &v, 1);
        if (kkt_factor(&kk) < 0) { LOG("init factor failed\n"); return; }
        for (int j = 0; j < n; j++) rhs1[j] = -pr->q[j];
        for (int i = 0; i < m; i++) rhs1[n + i] = pr->b[i];
        for (int i = n + m; i < N; i++) rhs1[i] = 0;
        kkt_solve(&kk, rhs1, sol1);
        memcpy(v.x, sol1, sizeof(double) * n);
        for (int i = 0; i < m; i++) { v.z[i] = sol1[n + i]; v.s[i] = -sol1[n + i]; }
        /* shift into cones */
        for (int pass = 0; pass < 2; pass++) {
            double *w = pass ? v.z : v.s;
            for (int k = 0; k < pr->nc; k++) {
                cone_t *c = &pr->K[k]; int d = c->dim, st = c->start;
                if (c->type == CONE_ZERO) { if (!pass) for (int i = 0; i < d; i++) w[st + i] = 0; continue; }
                if (c->type == CONE_NONNEG) {
                    double mn = 1e300; for (int i = 0; i < d; i++) if (w[st + i] < mn) mn = w[st + i];
                    if (mn < 1.4901161193847656e-08) for (int i = 0; i < d; i++) w[st + i] = (w[st + i] - mn) + 1.0;
                } else if (c->type == CONE_SOC) {
                    double nr = 0; for (int i = 1; i < d; i++) nr += w[st + i] * w[st + i];
                    double mn = w[st] - sqrt(nr);
                    if (mn < 1.4901161193847656e-08) w[st] = (w[st] - mn) + 1.0;
                } else if (c->type == CONE_PSD) {
                    double ev = psd_minev(c->ord, w + st);
                    if (ev < 1.4901161193847656e-08) for (int j = 0; j < c->ord; j++) w[st + pidx(j, j)] += (1.0 - ev);
                }
            }
        }
    } else {
        for (int k = 0; k < pr->nc; k++) {
            cone_t *c = &pr->K[k]; int d = c->dim, st = c->start;
            for (int i = 0; i < d; i++) { v.s[st + i] = 0; v.z[st + i] = 0; }
            if (c->type == CONE_ZERO) continue;
            if (c->type == CONE_NONNEG) for (int i = 0; i < d; i++) { v.s[st + i] = 1; v.z[st + i] = 1; }
            else if (c->type == CONE_SOC) { v.s[st] = 1; v.z[st] = 1; }
            else if (c->type == CONE_PSD) for (int j = 0; j < c->ord; j++) { v.s[st + pidx(j, j)] = 1; v.z[st + pidx(j, j)] = 1; }
            else { v.s[st] = EXP_C0; v.s[st + 1] = EXP_C1; v.s[st + 2] = EXP_C2; v.z[st] = EXP_C0; v.z[st + 1] = EXP_C1; v.z[st + 2] = EXP_C2; }
        }
    }
    v.tau = 1; v.kap = 1;

    int maxit = 200;
    double alpha = 0;
    int stall = 0;
    double prev_score = 1e300;
    for (int it = 0; it < maxit; it++) {
        /* residuals */
        memset(Px, 0, sizeof(double) * n);
        csc_mv(n, pr->Pp, pr->Pi, pr->Px, v.x, Px, 1.0);
        for (int j = 0; j < n; j++) rx[j] = Px[j] + pr->q[j] * v.tau;
        csc_mtv(n, pr->Ap, pr->Ai, pr->Ax, v.z, rx, 1.0);
        for (int i = 0; i < m; i++) rz[i] = v.s[i] - pr->b[i] * v.tau;
        csc_mv(n, pr->Ap, pr->Ai, pr->Ax, v.x, rz, 1.0);
        double xPx = dot(v.x, Px, n);
        double qx = dot(pr->q, v.x, n), bz = dot(pr->b, v.z, m);
        double rtau = qx + bz + v.kap + xPx / v.tau;
        double sz = dot(v.s, v.z, m);
        double mu = (sz + v.tau * v.kap) / (pr->nu + 1);

        /* checks in original space */
        unscale_sol(pr, &v, ux, us, uz);
        solcheck_t ck; check_solved(pr, ux, us, uz, &ck);
        LOG("it %3d mu %.3e tau %.3e kap %.3e rp %.2e rd %.2e gap %.2e/%.2e cones %.1e %.1e a %.3f t %.2f\n",
            it, mu, v.tau, v.kap, ck.rp, ck.rd, ck.gap, ck.gaprel, ck.conep, ck.coned, alpha, wtime() - wall0);
        if (ck.score < best) { best = ck.score; memcpy(bestx, ux, sizeof(double) * n); memcpy(bests, us, sizeof(double) * m); memcpy(bestz, uz, sizeof(double) * m); }
        if (ck.score <= tol) { res->status = 1; break; }
        /* infeasibility */
        {
            double bzo = 0; for (int i = 0; i < m; i++) bzo += pr->orig->b[i] * pr->E[i] * v.z[i];
            if (bzo < 0) {
                for (int i = 0; i < m; i++) tmpm[i] = pr->E[i] * v.z[i];
                double sc = check_pinf(pr, tmpm);
                if (sc <= tol) {
                    double nz = nrminf(tmpm, m);
                    for (int i = 0; i < m; i++) res->z[i] = tmpm[i] / nz;
                    res->status = 2; break;
                }
            }
            double qxo = 0; for (int j = 0; j < n; j++) qxo += pr->orig->q[j] * pr->D[j] * v.x[j];
            if (qxo < 0) {
                for (int j = 0; j < n; j++) ux[j] = pr->D[j] * v.x[j];
                for (int i = 0; i < m; i++) us[i] = v.s[i] / pr->E[i];
                for (int k = 0; k < pr->nc; k++) if (pr->K[k].type == CONE_ZERO) for (int i = 0; i < pr->K[k].dim; i++) us[pr->K[k].start + i] = 0;
                double sc = check_dinf(pr, ux, us);
                if (sc <= tol) {
                    double nx = nrminf(ux, n);
                    for (int j = 0; j < n; j++) res->x[j] = ux[j] / nx;
                    for (int i = 0; i < m; i++) res->s[i] = us[i] / nx;
                    res->status = 3; break;
                }
            }
        }
        if (wtime() - wall0 > 280) break;
        if (ck.score > prev_score * 0.999 && alpha < 1e-3) stall++; else stall = 0;
        prev_score = ck.score < prev_score ? ck.score : prev_score;
        if (stall > 10) { LOG("stall\n"); break; }

        /* scaling & factor */
        if (update_scaling(pr, &v, mu)) { LOG("scaling failed\n"); break; }
        fill_kkt(pr, &kk, &v, 0);
        int nreg = kkt_factor(&kk);
        for (int rt = 0; rt < 3 && nreg < 0; rt++) {
            kkt_set_delta(&kk, pr, kk.delta * 100);
            LOG("factor failed; delta -> %.1e\n", kk.delta);
            fill_kkt(pr, &kk, &v, 0);
            nreg = kkt_factor(&kk);
        }
        if (nreg < 0) { LOG("factor failed\n"); break; }
        if (nreg > 0 && verbose && !kk.dense) {
            const double *Dd = ldl_D(kk.f); const int *pm = ldl_perm(kk.f);
            int cx = 0, cz0 = 0, czn = 0, czs = 0, cpq = 0;
            for (int t = 0; t < N; t++) if (fabs(fabs(Dd[t]) - 2e-7) < 1e-20) {
                int o = pm[t];
                if (o < n) cx++; else if (o < n + m) { int r = o - n; int kk2 = 0; while (!(r >= pr->K[kk2].start && r < pr->K[kk2].start + pr->K[kk2].dim)) kk2++;
                    if (pr->K[kk2].type == CONE_ZERO) cz0++; else if (pr->K[kk2].type == CONE_NONNEG) czn++; else czs++; } else cpq++;
            }
            double dmin = 1e300, dmax = 0; for (int t = 0; t < N; t++) { double a = fabs(Dd[t]); if (a < dmin) dmin = a; if (a > dmax) dmax = a; }
            LOG("  nreg %d: x %d zero %d nonneg %d other %d pq %d  |D| in [%.1e, %.1e]\n", nreg, cx, cz0, czn, czs, cpq, dmin, dmax);
        }

        /* constant rhs */
        for (int j = 0; j < n; j++) rhs1[j] = -pr->q[j];
        for (int i = 0; i < m; i++) rhs1[n + i] = pr->b[i];
        for (int i = n + m; i < N; i++) rhs1[i] = 0;
        wsc_transform(pr, rhs1 + n);
        kkt_solve(&kk, rhs1, sol1);
        wsc_transform(pr, sol1 + n);
        for (int i = 0; i < m; i++) rhs1[n + i] = pr->b[i];
        /* tau denominator pieces */
        double *x1 = sol1, *z1 = sol1 + n;
        double *xi = dx; /* temp: xi = x/tau */
        for (int j = 0; j < n; j++) xi[j] = v.x[j] / v.tau;
        double *Pxi = rx == NULL ? NULL : Px; /* Px holds P x */
        double xiPxi = xPx / (v.tau * v.tau);
        double den = 0;
        for (int j = 0; j < n; j++) den += (pr->q[j] + 2 * Pxi[j] / v.tau) * x1[j];
        den += dot(pr->b, z1, m) - xiPxi - v.kap / v.tau;

        double sigma = 0, dtau = 0, dkap = 0;
        double dtau_a = 0, dkap_a = 0;
        for (int pass = 0; pass < 2; pass++) {
            int affine = (pass == 0);
            double eta = affine ? 1.0 : 1.0 - sigma;
            double sigmu = sigma * mu;
            combined_ds(pr, &v, dsa, dza, sigmu, dsrhs, affine);
            double dkap_rhs = affine ? v.tau * v.kap : v.tau * v.kap + dtau_a * dkap_a - sigmu;
            for (int j = 0; j < n; j++) rhs2[j] = -eta * rx[j];
            for (int i = 0; i < m; i++) rhs2[n + i] = -eta * rz[i] + dsrhs[i];
            for (int i = n + m; i < N; i++) rhs2[i] = 0;
            wsc_transform(pr, rhs2 + n);
            kkt_solve(&kk, rhs2, sol2);
            wsc_transform(pr, sol2 + n);
            double *x2 = sol2, *z2 = sol2 + n;
            double num = -eta * rtau + dkap_rhs / v.tau;
            for (int j = 0; j < n; j++) num -= (pr->q[j] + 2 * Pxi[j] / v.tau) * x2[j];
            num -= dot(pr->b, z2, m);
            dtau = num / den;
            for (int j = 0; j < n; j++) dx[j] = x2[j] + dtau * x1[j];
            for (int i = 0; i < m; i++) dz[i] = z2[i] + dtau * z1[i];
            apply_H(pr, &v, dz, dsv);
            for (int i = 0; i < m; i++) dsv[i] = -dsrhs[i] - dsv[i];
            if (use_resds) {
                /* ds from the linearized primal equation for SOC blocks: A dx + ds - b dtau = -eta rz */
                for (int i = 0; i < m; i++) tmpm[i] = 0;
                csc_mv(n, pr->Ap, pr->Ai, pr->Ax, dx, tmpm, 1.0);
                for (int k = 0; k < pr->nc; k++) {
                    cone_t *c = &pr->K[k];
                    if (c->type != CONE_SOC) continue;
                    for (int i = c->start; i < c->start + c->dim; i++) dsv[i] = -eta * rz[i] + pr->b[i] * dtau - tmpm[i];
                }
            }
            dkap = -(dkap_rhs + v.kap * dtau) / v.tau;
            double amax = max_step(pr, &v, dx, dz, dsv, dtau, dkap);
            if (!isfinite(dtau) || !(amax >= 0)) amax = 0;
            double frac = getenv("CONIC_FRAC") ? atof(getenv("CONIC_FRAC")) : 0.99;
            double a = affine ? (amax < 1.0 ? amax : 1.0) : (frac * amax < 1.0 ? frac * amax : 1.0);
            if (pr->has_exp) {
                int tries = 0;
                double a0 = a;
                int central = !affine;
              retry_ls:
                tries = 0; a = a0;
                while (tries < 60) {
                    double mu_new = 0;
                    if (!affine) {
                        double s2 = 0;
                        for (int i = 0; i < m; i++) s2 += (v.s[i] + a * dsv[i]) * (v.z[i] + a * dz[i]);
                        mu_new = (s2 + (v.tau + a * dtau) * (v.kap + a * dkap)) / (pr->nu + 1);
                    }
                    if (exp_ok(pr, &v, dz, dsv, a, mu_new, central)) break;
                    a *= 0.8; tries++;
                }
                if (tries >= 60) a = 0;
                if (central && a < 1e-3 * a0) { central = 0; goto retry_ls; }
            }
            if (affine) {
                sigma = (1 - a); sigma = sigma * sigma * sigma;
                memcpy(dxa, dx, sizeof(double) * n); memcpy(dza, dz, sizeof(double) * m); memcpy(dsa, dsv, sizeof(double) * m);
                dtau_a = dtau; dkap_a = dkap;
                if (pr->has_exp) { memset(dza, 0, sizeof(double) * m); memset(dsa, 0, sizeof(double) * m); dtau_a = 0; dkap_a = 0; }
            } else alpha = a;
        }
        (void)xi;
        /* update */
        for (int j = 0; j < n; j++) v.x[j] += alpha * dx[j];
        for (int i = 0; i < m; i++) { v.z[i] += alpha * dz[i]; v.s[i] += alpha * dsv[i]; }
        v.tau += alpha * dtau; v.kap += alpha * dkap;
        if (!(v.tau > 0) || !(v.kap > 0)) { LOG("tau/kap nonpositive\n"); break; }
        /* keep zero cone s exactly zero */
        for (int k = 0; k < pr->nc; k++) if (pr->K[k].type == CONE_ZERO) for (int i = 0; i < pr->K[k].dim; i++) v.s[pr->K[k].start + i] = 0;
    }
    if (res->status == 1) {
        unscale_sol(pr, &v, res->x, res->s, res->z);
    } else if (res->status == 0 && best <= tol) {
        memcpy(res->x, bestx, sizeof(double) * n); memcpy(res->s, bests, sizeof(double) * m); memcpy(res->z, bestz, sizeof(double) * m);
        res->status = 1;
    }
    LOG("best score %.3e\n", best);
}

int main(int argc, char **argv)
{
    wall0 = wtime();
    load_blas();
    const char *in = NULL, *out = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--input") && i + 1 < argc) in = argv[++i];
        else if (!strcmp(argv[i], "--output") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "-v")) verbose = 1;
    }
    if (getenv("CONIC_VERBOSE")) verbose = atoi(getenv("CONIC_VERBOSE"));
    if (!in || !out) { fprintf(stderr, "usage: %s --input IN --output OUT\n", argv[0]); return 2; }
    problem_t P; char err[512];
    if (npz_read_problem(in, &P, err, sizeof(err))) {
        fprintf(stderr, "read error: %s\n", err);
        npz_write_result(out, "unresolved", NULL, 0, NULL, NULL, 0, 0, 0, 0);
        return 0;
    }
    LOG("read %.3fs n=%d m=%d nnzA=%d nnzP=%d cones=%d\n", wtime() - wall0, P.n, P.m, P.Ap[P.n], P.Pp[P.n], P.ncones);
    prob_t pr; memset(&pr, 0, sizeof(pr));
    setup(&pr, &P);
    result_t res;
    solve(&pr, &res);
    const char *st = res.status == 1 ? "solved" : res.status == 2 ? "primal_infeasible" : res.status == 3 ? "dual_infeasible" : "unresolved";
    LOG("status %s time %.3f\n", st, wtime() - wall0);
    if (res.status == 1) npz_write_result(out, st, res.x, P.n, res.s, res.z, P.m, 0, 0, 0);
    else if (res.status == 2) npz_write_result(out, st, NULL, P.n, NULL, res.z, P.m, 0, 0, 0);
    else if (res.status == 3) npz_write_result(out, st, res.x, P.n, res.s, NULL, P.m, 0, 0, 0);
    else npz_write_result(out, st, NULL, P.n, NULL, NULL, P.m, 0, 0, 0);
    return 0;
}
