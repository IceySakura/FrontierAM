#include <math.h>
#include <stdint.h>
#include <stddef.h>

/* Left-looking LDL^T for a preordered symmetric quasi-definite matrix.
   Only the upper triangle is read. The symbolic structure is reusable. */
int64_t conic_ldl_symbolic(int n, const int *ap, const int *ai,
                          int *lp, int *parent, int *count, int *flag) {
    int64_t total = 0;
    int64_t entries = 0;
    for (int k = 0; k < n; ++k) {
        parent[k] = -1;
        flag[k] = k;
        count[k] = 0;
        for (int p = ap[k]; p < ap[k+1]; ++p) {
            int i = ai[p];
            if (i >= k) continue;
            while (flag[i] != k) {
                if (parent[i] == -1) parent[i] = k;
                ++count[i];
                if (++entries > 10000000) return -1;
                flag[i] = k;
                i = parent[i];
            }
        }
    }
    lp[0] = 0;
    for (int k = 0; k < n; ++k) {
        total += count[k];
        if (total > 10000000) return -1;
        lp[k+1] = (int)total;
    }
    return total;
}

int conic_ldl_numeric(int n, const int *ap, const int *ai, const double *ax,
                     const int *lp, const int *parent, int *li, double *lx,
                     double *d, double *y, int *count, int *pattern, int *flag,
                     const int *sign, double threshold, double shift) {
    int regularized = 0;
    for (int k = 0; k < n; ++k) {
        y[k] = 0.0;
        count[k] = 0;
        flag[k] = -1;
    }
    for (int k = 0; k < n; ++k) {
        int top = n;
        flag[k] = k;
        for (int p = ap[k]; p < ap[k+1]; ++p) {
            int i = ai[p];
            if (i > k) continue;
            y[i] += ax[p];
            int len = 0;
            while (flag[i] != k) {
                pattern[len++] = i;
                flag[i] = k;
                i = parent[i];
                if (i < 0) return -1;
            }
            while (len) pattern[--top] = pattern[--len];
        }
        double pivot = y[k];
        y[k] = 0.0;
        for (; top < n; ++top) {
            int i = pattern[top];
            double yi = y[i];
            y[i] = 0.0;
            int end = lp[i] + count[i];
            for (int p = lp[i]; p < end; ++p) {
                y[li[p]] -= lx[p]*yi;
            }
            double lki = yi/d[i];
            pivot -= yi*lki;
            if (end >= lp[i+1]) return -2;
            li[end] = k;
            lx[end] = lki;
            ++count[i];
        }
        if (!isfinite(pivot)) return -3;
        if ((sign[k] && sign[k]*pivot < threshold) ||
                (!sign[k] && fabs(pivot) < threshold)) {
            pivot = sign[k] ? sign[k]*shift : copysign(shift, pivot);
            ++regularized;
        }
        d[k] = pivot;
    }
    return regularized;
}

void conic_ldl_solve(int n, const int *lp, const int *li, const double *lx,
                    const double *d, double *x) {
    for (int j = 0; j < n; ++j) {
        double xj = x[j];
        for (int p = lp[j]; p < lp[j+1]; ++p) x[li[p]] -= lx[p]*xj;
    }
    for (int j = 0; j < n; ++j) x[j] /= d[j];
    for (int j = n-1; j >= 0; --j) {
        double xj = x[j];
        for (int p = lp[j]; p < lp[j+1]; ++p) xj -= lx[p]*x[li[p]];
        x[j] = xj;
    }
}

int conic_csc_sorted(int n, const int *ap, const int *ai, const double *ax) {
    for (int j = 0; j < n; ++j) {
        int previous = -1;
        for (int p = ap[j]; p < ap[j+1]; ++p) {
            if (ai[p] <= previous || ax[p] == 0.0 || !isfinite(ax[p])) return 0;
            previous = ai[p];
        }
    }
    return 1;
}

void conic_csc_matvec(int m, int n, const int *ap, const int *ai,
                     const double *ax, const double *x, double *y, int transpose) {
    if (transpose) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int p = ap[j]; p < ap[j+1]; ++p) sum += ax[p]*x[ai[p]];
            y[j] = sum;
        }
    } else {
        for (int i = 0; i < m; ++i) y[i] = 0.0;
        for (int j = 0; j < n; ++j) {
            double value = x[j];
            for (int p = ap[j]; p < ap[j+1]; ++p) y[ai[p]] += ax[p]*value;
        }
    }
}

void conic_csc_transpose(int m, int n, const int *ap, const int *ai, const double *ax,
                         int *bp, int *bi, double *bx, int *work) {
    for (int i = 0; i <= m; ++i) bp[i] = 0;
    for (int p = 0; p < ap[n]; ++p) ++bp[ai[p]+1];
    for (int i = 0; i < m; ++i) bp[i+1] += bp[i];
    for (int i = 0; i < m; ++i) work[i] = bp[i];
    for (int j = 0; j < n; ++j) {
        for (int p = ap[j]; p < ap[j+1]; ++p) {
            int next = work[ai[p]]++;
            bi[next] = j;
            bx[next] = ax[p];
        }
    }
}
