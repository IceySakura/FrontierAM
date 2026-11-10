/* Sparse LDL^T kernel (QDLDL-style up-looking factorization) with dynamic
 * regularization, iterative refinement and CSC mat-vec helpers.
 * All matrices are CSC with int32 indices; K is stored as its upper triangle
 * (diagonal included) in an already fill-reducing permuted order. */
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* y += A x */
void csc_matvec(int ncol, const int *Ap, const int *Ai, const double *Ax,
                const double *x, double *y) {
    for (int j = 0; j < ncol; j++) {
        double xj = x[j];
        if (xj == 0.0) continue;
        for (int p = Ap[j]; p < Ap[j + 1]; p++) y[Ai[p]] += Ax[p] * xj;
    }
}

/* y += A^T x */
void csc_matvec_t(int ncol, const int *Ap, const int *Ai, const double *Ax,
                  const double *x, double *y) {
    for (int j = 0; j < ncol; j++) {
        double s = 0.0;
        for (int p = Ap[j]; p < Ap[j + 1]; p++) s += Ax[p] * x[Ai[p]];
        y[j] += s;
    }
}

/* y = K x for symmetric K given by upper triangle */
static void symv_upper(int n, const int *Ap, const int *Ai, const double *Ax,
                       const double *x, double *y) {
    memset(y, 0, sizeof(double) * (size_t)n);
    for (int j = 0; j < n; j++) {
        double xj = x[j], s = 0.0;
        for (int p = Ap[j]; p < Ap[j + 1]; p++) {
            int i = Ai[p];
            double v = Ax[p];
            if (i == j) { s += v * xj; }
            else { y[i] += v * xj; s += v * x[i]; }
        }
        y[j] += s;
    }
}

/* elimination tree + column counts; returns nnz(L) or -1 */
long ldl_etree(int n, const int *Ap, const int *Ai, int *work, int *Lnz, int *etree) {
    for (int i = 0; i < n; i++) { work[i] = -1; Lnz[i] = 0; etree[i] = -1; }
    for (int j = 0; j < n; j++) {
        work[j] = j;
        for (int p = Ap[j]; p < Ap[j + 1]; p++) {
            int i = Ai[p];
            if (i > j) return -1;
            while (work[i] != j) {
                if (etree[i] == -1) etree[i] = j;
                Lnz[i]++;
                work[i] = j;
                i = etree[i];
            }
        }
    }
    long s = 0;
    for (int i = 0; i < n; i++) s += Lnz[i];
    return s;
}

/* numeric factorization; returns number of dynamically regularized pivots.
 * iwork: 3n ints, fwork: n doubles */
int ldl_factor(int n, const int *Ap, const int *Ai, const double *Ax,
               int *Lp, int *Li, double *Lx, double *D, double *Dinv,
               const int *Lnz, const int *etree, const signed char *sign,
               double eps, double delta, int *iwork, double *fwork) {
    int *yMarkers = iwork, *yIdx = iwork + n, *elimBuffer = iwork + 2 * n;
    int *LNext = (int *)0;
    double *yVals = fwork;
    int nreg = 0;
    /* LNext stored in Lp-based scratch: reuse etree-sized buffer is not allowed,
       so we keep it inside the yIdx region trick: allocate on Lp copy below */
    static int *lnext_buf = 0; static int lnext_cap = 0;
    if (lnext_cap < n) {
        if (lnext_buf) free(lnext_buf);
        lnext_buf = (int *)malloc(sizeof(int) * (size_t)(n + 1));
        lnext_cap = n;
    }
    LNext = lnext_buf;
    Lp[0] = 0;
    for (int i = 0; i < n; i++) {
        Lp[i + 1] = Lp[i] + Lnz[i];
        yMarkers[i] = 0; yVals[i] = 0.0; D[i] = 0.0; LNext[i] = Lp[i];
    }
    for (int k = 0; k < n; k++) {
        int nnzY = 0;
        double dk = 0.0;
        for (int p = Ap[k]; p < Ap[k + 1]; p++) {
            int bidx = Ai[p];
            if (bidx == k) { dk += Ax[p]; continue; }
            yVals[bidx] += Ax[p];
            int next = bidx;
            if (!yMarkers[next]) {
                yMarkers[next] = 1;
                elimBuffer[0] = next;
                int nnzE = 1;
                next = etree[bidx];
                while (next != -1 && next < k) {
                    if (yMarkers[next]) break;
                    yMarkers[next] = 1;
                    elimBuffer[nnzE++] = next;
                    next = etree[next];
                }
                while (nnzE) yIdx[nnzY++] = elimBuffer[--nnzE];
            }
        }
        for (int i = nnzY - 1; i >= 0; i--) {
            int c = yIdx[i];
            int tmp = LNext[c];
            double yc = yVals[c];
            for (int j = Lp[c]; j < tmp; j++) yVals[Li[j]] -= Lx[j] * yc;
            Li[tmp] = k;
            double l = yc * Dinv[c];
            Lx[tmp] = l;
            dk -= yc * l;
            LNext[c]++;
            yVals[c] = 0.0;
            yMarkers[c] = 0;
        }
        if (sign[k] * dk <= eps) { dk = sign[k] * delta; nreg++; }
        D[k] = dk;
        Dinv[k] = 1.0 / dk;
    }
    return nreg;
}

static void ldl_solve(int n, const int *Lp, const int *Li, const double *Lx,
                      const double *Dinv, double *x) {
    for (int i = 0; i < n; i++) {
        double xi = x[i];
        if (xi != 0.0)
            for (int j = Lp[i]; j < Lp[i + 1]; j++) x[Li[j]] -= Lx[j] * xi;
    }
    for (int i = 0; i < n; i++) x[i] *= Dinv[i];
    for (int i = n - 1; i >= 0; i--) {
        double s = x[i];
        for (int j = Lp[i]; j < Lp[i + 1]; j++) s -= Lx[j] * x[Li[j]];
        x[i] = s;
    }
}

/* Solve (K - diag(reg)) x = b with LDL of K, refining against the unregularized
 * matrix. work: 3n doubles. Returns number of refinement steps. */
int ldl_solve_refine(int n, const int *Ap, const int *Ai, const double *Ax,
                     const double *reg, const int *Lp, const int *Li,
                     const double *Lx, const double *Dinv, const double *b,
                     double *x, double *work, int maxiter, double abstol,
                     double reltol) {
    double *r = work, *dx = work + n, *Kx = work + 2 * n;
    double bnorm = 0.0;
    for (int i = 0; i < n; i++) { x[i] = b[i]; double a = fabs(b[i]); if (a > bnorm) bnorm = a; }
    ldl_solve(n, Lp, Li, Lx, Dinv, x);
    double tol = abstol + reltol * bnorm;
    double prev = INFINITY;
    int it = 0;
    for (; it < maxiter; it++) {
        symv_upper(n, Ap, Ai, Ax, x, Kx);
        double rn = 0.0;
        for (int i = 0; i < n; i++) {
            double v = b[i] - (Kx[i] - reg[i] * x[i]);
            r[i] = v;
            double a = fabs(v);
            if (a > rn || v != v) rn = (v != v) ? INFINITY : a;
        }
        if (rn <= tol) break;
        if (it > 0 && rn > prev) { for (int i = 0; i < n; i++) x[i] -= dx[i]; break; }
        if (it > 0 && rn > prev * 0.5) break; /* stalled */
        prev = rn;
        memcpy(dx, r, sizeof(double) * (size_t)n);
        ldl_solve(n, Lp, Li, Lx, Dinv, dx);
        for (int i = 0; i < n; i++) x[i] += dx[i];
    }
    return it;
}

/* counting sort of COO (row,col) pairs by column: Kp (N+1), Ki (nent),
 * order (nent): order[k] = source index of k-th CSC entry. */
void coo_to_csc(int N, long nent, const long *r, const long *c, int *Kp, int *Ki, long *order) {
    long *pos = (long *)malloc(sizeof(long) * (size_t)(N + 1));
    for (int j = 0; j <= N; j++) pos[j] = 0;
    for (long k = 0; k < nent; k++) pos[c[k] + 1]++;
    for (int j = 0; j < N; j++) pos[j + 1] += pos[j];
    for (int j = 0; j <= N; j++) Kp[j] = (int)pos[j];
    for (long k = 0; k < nent; k++) {
        long p = pos[c[k]]++;
        Ki[p] = (int)r[k];
        order[p] = k;
    }
    free(pos);
}

/* like ldl_etree but aborts (returns -2) once nnz(L) exceeds cap */
long ldl_count(int n, const int *Ap, const int *Ai, int *work, int *Lnz, int *etree, long cap) {
    long s = 0;
    for (int i = 0; i < n; i++) { work[i] = -1; Lnz[i] = 0; etree[i] = -1; }
    for (int j = 0; j < n; j++) {
        work[j] = j;
        for (int p = Ap[j]; p < Ap[j + 1]; p++) {
            int i = Ai[p];
            if (i > j) return -1;
            while (work[i] != j) {
                if (etree[i] == -1) etree[i] = j;
                Lnz[i]++;
                s++;
                work[i] = j;
                i = etree[i];
            }
        }
        if (s > cap) return -2;
    }
    return s;
}
