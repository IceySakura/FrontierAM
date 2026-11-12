/* Sparse LDL' for quasi-definite matrices, QDLDL algorithm (etree + column counts,
   up-looking row-by-row numeric factorization) applied to the permuted upper triangle
   C = (P K P')_upper. The values of C are scattered from the caller's Ax through a map
   computed once in the symbolic phase, so refactorization needs no sorting and no
   reallocation. */
#include "ldl.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "amd.h"

struct ldl_s {
    int n;
    int p0, nzK;       /* Ap[0], Ap[n] of the input pattern */
    int nzC;           /* strictly upper entries of C (duplicates merged) */
    int direct;        /* 1: map is injective and covers every diagonal => plain copy scatter */
    long nzL;          /* nnz of strictly lower L */
    int *perm;         /* n */
    int *Cp, *Ci;      /* strict upper part of C: n+1, nzC */
    int *map;          /* nzK: original position -> index in Cx (diag k -> nzC + k) */
    double *Cx;        /* nzC + n */
    int *etree, *Lnz;  /* n */
    int *Lp, *Li;      /* n+1, nzL */
    double *Lx;        /* nzL */
    double *D, *Dinv;  /* n */
    /* numeric / solve workspace */
    unsigned char *yMarkers; /* n */
    int *yIdx, *elimBuffer, *LNext; /* n each */
    double *yVals;     /* n */
    double *xwork;     /* n */
};

static void *xmalloc(size_t nmemb, size_t size)
{
    if (nmemb == 0) nmemb = 1;
    return malloc(nmemb * size);
}
static void *xcalloc(size_t nmemb, size_t size)
{
    if (nmemb == 0) nmemb = 1;
    return calloc(nmemb, size);
}

void ldl_free(ldl_t *f)
{
    if (!f) return;
    free(f->perm);
    free(f->Cp);
    free(f->Ci);
    free(f->map);
    free(f->Cx);
    free(f->etree);
    free(f->Lnz);
    free(f->Lp);
    free(f->Li);
    free(f->Lx);
    free(f->D);
    free(f->Dinv);
    free(f->yMarkers);
    free(f->yIdx);
    free(f->elimBuffer);
    free(f->LNext);
    free(f->yVals);
    free(f->xwork);
    free(f);
}

long ldl_nnz_L(const ldl_t *f) { return f ? f->nzL : 0; }
int ldl_n(const ldl_t *f) { return f ? f->n : 0; }
const int *ldl_perm(const ldl_t *f) { return f ? f->perm : NULL; }
const double *ldl_D(const ldl_t *f) { return f ? f->D : NULL; }

ldl_t *ldl_symbolic(int n, const int *Ap, const int *Ai, const int *perm_in)
{
    ldl_t *f;
    int *pinv = NULL, *Tp = NULL, *wmark = NULL, *newpos = NULL, *diagcnt = NULL;
    int i, j, k, p, q, nzK, p0, nstrict;
    int64_t sum;

    if (n < 0 || !Ap) return NULL;
    if (n > INT_MAX / 4) return NULL;
    f = (ldl_t *)calloc(1, sizeof(ldl_t));
    if (!f) return NULL;
    f->n = n;
    p0 = Ap[0];
    nzK = Ap[n];
    if (p0 < 0 || nzK < p0) goto fail;
    if (nzK > p0 && !Ai) goto fail;
    f->p0 = p0;
    f->nzK = nzK;

    /* ---- ordering ---- */
    f->perm = (int *)xmalloc((size_t)n, sizeof(int));
    pinv = (int *)xmalloc((size_t)n, sizeof(int));
    if (!f->perm || !pinv) goto fail;
    if (perm_in) {
        memcpy(f->perm, perm_in, (size_t)n * sizeof(int));
    } else {
        if (amd_order(n, Ap, Ai, f->perm) != 0) goto fail;
    }
    for (i = 0; i < n; i++) pinv[i] = -1;
    for (k = 0; k < n; k++) {
        i = f->perm[k];
        if (i < 0 || i >= n || pinv[i] != -1) goto fail; /* not a permutation */
        pinv[i] = k;
    }

    /* ---- permuted upper triangle C (strict part) + map ---- */
    f->Cp = (int *)xcalloc((size_t)n + 1, sizeof(int));
    Tp = (int *)xcalloc((size_t)n + 1, sizeof(int));
    diagcnt = (int *)xcalloc((size_t)n, sizeof(int));
    f->map = (int *)xmalloc((size_t)nzK, sizeof(int));
    if (!f->Cp || !Tp || !diagcnt || !f->map) goto fail;
    nstrict = 0;
    for (j = 0; j < n; j++) {
        int pj = pinv[j];
        for (p = Ap[j]; p < Ap[j + 1]; p++) {
            int a;
            i = Ai[p];
            if (i < 0 || i >= n) goto fail;
            a = pinv[i];
            if (a != pj) {
                Tp[(a > pj ? a : pj) + 1]++;
                nstrict++;
            }
        }
    }
    for (k = 0; k < n; k++) Tp[k + 1] += Tp[k];
    f->Ci = (int *)xmalloc((size_t)nstrict, sizeof(int));
    newpos = (int *)xmalloc((size_t)nstrict, sizeof(int));
    wmark = (int *)xmalloc((size_t)n, sizeof(int));
    if (!f->Ci || !newpos || !wmark) goto fail;
    for (k = 0; k < n; k++) wmark[k] = Tp[k]; /* next free slot per column */
    for (j = 0; j < n; j++) {
        int pj = pinv[j];
        for (p = Ap[j]; p < Ap[j + 1]; p++) {
            int a = pinv[Ai[p]];
            if (a == pj) {
                f->map[p] = -(a + 1); /* diagonal, fixed below */
                diagcnt[a]++;
            } else {
                int c = a > pj ? a : pj, r = a > pj ? pj : a;
                q = wmark[c]++;
                f->Ci[q] = r;
                f->map[p] = q;
            }
        }
    }
    /* merge duplicates in place (column by column), record new positions */
    for (k = 0; k < n; k++) wmark[k] = -1;
    q = 0;
    for (k = 0; k < n; k++) {
        int start = q;
        f->Cp[k] = start;
        for (p = Tp[k]; p < Tp[k + 1]; p++) {
            int r = f->Ci[p];
            if (wmark[r] >= start) {
                newpos[p] = wmark[r];
            } else {
                wmark[r] = q;
                f->Ci[q] = r;
                newpos[p] = q;
                q++;
            }
        }
    }
    f->Cp[n] = q;
    f->nzC = q;
    f->direct = (q == nstrict);
    for (k = 0; k < n; k++)
        if (diagcnt[k] != 1) f->direct = 0;
    for (p = p0; p < nzK; p++) {
        int m = f->map[p];
        f->map[p] = (m >= 0) ? newpos[m] : f->nzC + (-m - 1);
    }
    free(newpos);
    newpos = NULL;
    free(Tp);
    Tp = NULL;
    free(diagcnt);
    diagcnt = NULL;
    if (q < nstrict && q > 0) {
        int *tmp = (int *)realloc(f->Ci, (size_t)q * sizeof(int));
        if (tmp) f->Ci = tmp;
    }

    /* ---- elimination tree and column counts (QDLDL_etree) ---- */
    f->etree = (int *)xmalloc((size_t)n, sizeof(int));
    f->Lnz = (int *)xmalloc((size_t)n, sizeof(int));
    if (!f->etree || !f->Lnz) goto fail;
    {
        int *restrict work = wmark;
        int *restrict etree = f->etree;
        int *restrict Lnz = f->Lnz;
        const int *restrict Cp = f->Cp;
        const int *restrict Ci = f->Ci;
        for (i = 0; i < n; i++) {
            work[i] = -1;
            Lnz[i] = 0;
            etree[i] = -1;
        }
        for (j = 0; j < n; j++) {
            work[j] = j;
            for (p = Cp[j]; p < Cp[j + 1]; p++) {
                i = Ci[p];
                while (work[i] != j) {
                    if (etree[i] == -1) etree[i] = j;
                    Lnz[i]++;
                    work[i] = j;
                    i = etree[i];
                }
            }
        }
    }
    free(wmark);
    wmark = NULL;
    sum = 0;
    for (i = 0; i < n; i++) sum += f->Lnz[i];
    if (sum > (int64_t)INT_MAX) goto fail;
    f->nzL = (long)sum;

    /* ---- allocate L, D and numeric workspace ---- */
    f->Lp = (int *)xmalloc((size_t)n + 1, sizeof(int));
    f->Li = (int *)xmalloc((size_t)sum, sizeof(int));
    f->Lx = (double *)xmalloc((size_t)sum, sizeof(double));
    f->D = (double *)xcalloc((size_t)n, sizeof(double));
    f->Dinv = (double *)xcalloc((size_t)n, sizeof(double));
    f->Cx = (double *)xcalloc((size_t)f->nzC + (size_t)n, sizeof(double));
    f->yMarkers = (unsigned char *)xcalloc((size_t)n, 1);
    f->yIdx = (int *)xmalloc((size_t)n, sizeof(int));
    f->elimBuffer = (int *)xmalloc((size_t)n, sizeof(int));
    f->LNext = (int *)xmalloc((size_t)n, sizeof(int));
    f->yVals = (double *)xcalloc((size_t)n, sizeof(double));
    f->xwork = (double *)xmalloc((size_t)n, sizeof(double));
    if (!f->Lp || !f->Li || !f->Lx || !f->D || !f->Dinv || !f->Cx || !f->yMarkers || !f->yIdx ||
        !f->elimBuffer || !f->LNext || !f->yVals || !f->xwork)
        goto fail;
    f->Lp[0] = 0;
    for (i = 0; i < n; i++) f->Lp[i + 1] = f->Lp[i] + f->Lnz[i];
    free(pinv);
    return f;

fail:
    free(pinv);
    free(Tp);
    free(wmark);
    free(newpos);
    free(diagcnt);
    ldl_free(f);
    return NULL;
}

int ldl_numeric(ldl_t *f, const double *Ax, const signed char *sign, double eps, double delta)
{
    const int n = f->n;
    double *restrict Cx = f->Cx;
    const int *restrict map = f->map;
    const int *restrict Cp = f->Cp;
    const int *restrict Ci = f->Ci;
    const int *restrict etree = f->etree;
    const int *restrict Lp = f->Lp;
    const int *restrict perm = f->perm;
    int *restrict Li = f->Li;
    int *restrict yIdx = f->yIdx;
    int *restrict elimBuffer = f->elimBuffer;
    int *restrict LNext = f->LNext;
    double *restrict Lx = f->Lx;
    double *restrict D = f->D;
    double *restrict Dinv = f->Dinv;
    double *restrict y = f->yVals;
    unsigned char *restrict marked = f->yMarkers;
    const double *restrict Cd = Cx + f->nzC;
    int k, p, nreg = 0;

    /* ---- scatter values into the permuted upper triangle ---- */
    if (f->direct) {
        for (p = f->p0; p < f->nzK; p++) Cx[map[p]] = Ax[p];
    } else {
        memset(Cx, 0, ((size_t)f->nzC + (size_t)n) * sizeof(double));
        for (p = f->p0; p < f->nzK; p++) Cx[map[p]] += Ax[p];
    }
    for (k = 0; k < n; k++) {
        LNext[k] = Lp[k];
        y[k] = 0.0;
        marked[k] = 0;
    }

    /* ---- up-looking factorization, row k of L at a time ---- */
    for (k = 0; k < n; k++) {
        double dk = Cd[k];
        int nnzY = 0, i;
        const int pend = Cp[k + 1];
        for (p = Cp[k]; p < pend; p++) {
            int bidx = Ci[p];
            y[bidx] = Cx[p];
            if (!marked[bidx]) {
                int nnzE = 1, nextIdx;
                marked[bidx] = 1;
                elimBuffer[0] = bidx;
                nextIdx = etree[bidx];
                while (nextIdx != -1 && nextIdx < k) {
                    if (marked[nextIdx]) break;
                    marked[nextIdx] = 1;
                    elimBuffer[nnzE++] = nextIdx;
                    nextIdx = etree[nextIdx];
                }
                while (nnzE) yIdx[nnzY++] = elimBuffer[--nnzE];
            }
        }
        for (i = nnzY - 1; i >= 0; i--) {
            const int cidx = yIdx[i];
            const int tmpIdx = LNext[cidx];
            const double yc = y[cidx];
            double l;
            int jj;
            for (jj = Lp[cidx]; jj < tmpIdx; jj++) y[Li[jj]] -= Lx[jj] * yc;
            l = yc * Dinv[cidx];
            Li[tmpIdx] = k;
            Lx[tmpIdx] = l;
            dk -= yc * l;
            LNext[cidx] = tmpIdx + 1;
            y[cidx] = 0.0;
            marked[cidx] = 0;
        }
        if (!(fabs(dk) <= DBL_MAX)) return -1; /* NaN or Inf */
        if (sign) {
            const double s = (sign[perm[k]] < 0) ? -1.0 : 1.0;
            if (s * dk <= eps) {
                dk = s * delta;
                nreg++;
            }
        } else if (dk == 0.0) {
            return -1;
        }
        D[k] = dk;
        Dinv[k] = 1.0 / dk;
    }
    return nreg;
}

void ldl_solve(ldl_t *f, double *b)
{
    const int n = f->n;
    const int *restrict Lp = f->Lp;
    const int *restrict Li = f->Li;
    const double *restrict Lx = f->Lx;
    const double *restrict Dinv = f->Dinv;
    const int *restrict perm = f->perm;
    double *restrict x = f->xwork;
    int i, j;

    for (i = 0; i < n; i++) x[i] = b[perm[i]];
    /* L x = b (L unit lower, stored by columns) */
    for (i = 0; i < n; i++) {
        const double xi = x[i];
        const int pe = Lp[i + 1];
        if (xi != 0.0)
            for (j = Lp[i]; j < pe; j++) x[Li[j]] -= Lx[j] * xi;
    }
    for (i = 0; i < n; i++) x[i] *= Dinv[i];
    /* L' x = y */
    for (i = n - 1; i >= 0; i--) {
        double s = x[i];
        const int pe = Lp[i + 1];
        for (j = Lp[i]; j < pe; j++) s -= Lx[j] * x[Li[j]];
        x[i] = s;
    }
    for (i = 0; i < n; i++) b[perm[i]] = x[i];
}
