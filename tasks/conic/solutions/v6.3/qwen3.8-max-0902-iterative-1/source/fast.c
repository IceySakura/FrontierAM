/* Small helper routines for the conic solver: sparse matvecs and dense
 * Schur-complement accumulation that exploit sparsity.
 * Compiled by build.sh into libfast.so and used through ctypes. */
#include <math.h>
#include <stddef.h>

/* y = A x   (A stored CSR, m rows) */
void csr_mv(const double *data, const int *idx, const int *indptr,
            const double *x, double *y, int m)
{
    for (int r = 0; r < m; ++r) {
        double acc = 0.0;
        int s = indptr[r], e = indptr[r + 1];
        for (int k = s; k < e; ++k) acc += data[k] * x[idx[k]];
        y[r] = acc;
    }
}

/* y += A x   (A stored CSR, m rows) */
void csr_mv_add(const double *data, const int *idx, const int *indptr,
                const double *x, double *y, int m)
{
    for (int r = 0; r < m; ++r) {
        double acc = 0.0;
        int s = indptr[r], e = indptr[r + 1];
        for (int k = s; k < e; ++k) acc += data[k] * x[idx[k]];
        y[r] += acc;
    }
}

/* y = A^T x  (A stored CSC, n columns: y[j] = sum_k data[k]*x[idx[k]]) */
void csc_mvT(const double *data, const int *idx, const int *indptr,
             const double *x, double *y, int n)
{
    for (int j = 0; j < n; ++j) {
        double acc = 0.0;
        int s = indptr[j], e = indptr[j + 1];
        for (int k = s; k < e; ++k) acc += data[k] * x[idx[k]];
        y[j] = acc;
    }
}

/* y = A x with A given in CSC (scatter form), n columns, m rows */
void csc_mv(const double *data, const int *idx, const int *indptr,
            const double *x, double *y, int n)
{
    for (int j = 0; j < n; ++j) {
        double xj = x[j];
        if (xj == 0.0) continue;
        int s = indptr[j], e = indptr[j + 1];
        for (int k = s; k < e; ++k) y[idx[k]] += data[k] * xj;
    }
}

/* M += sum_{r=r0}^{r1-1} d[r] * a_r a_r^T   (a_r = row r of CSR A), M is n x n
 * row-major dense, full (both triangles updated). d is indexed by row. */
void schur_diag(const double *data, const int *idx, const int *indptr,
                const double *d, int r0, int r1, double *M, int n)
{
    for (int r = r0; r < r1; ++r) {
        double dr = d[r];
        if (dr == 0.0) continue;
        int s = indptr[r], e = indptr[r + 1];
        for (int p = s; p < e; ++p) {
            int j = idx[p];
            double v = dr * data[p];
            double *Mj = M + (size_t)j * n;
            for (int q = s; q < e; ++q) Mj[idx[q]] += v * data[q];
        }
    }
}

/* M += A_blk^T H A_blk where A_blk are rows [r0,r1) of the CSR matrix and
 * H is k x k symmetric dense (k = r1-r0), row-major. Only the lower triangle
 * of H is read. M is n x n row-major dense. */
void schur_gen(const double *data, const int *idx, const int *indptr,
               int r0, int k, const double *H, double *M, int n)
{
    for (int a = 0; a < k; ++a) {
        int as = indptr[r0 + a], ae = indptr[r0 + a + 1];
        if (as == ae) continue;
        for (int b = 0; b <= a; ++b) {
            double h = H[(size_t)a * k + b];
            if (h == 0.0) continue;
            int bs = indptr[r0 + b], be = indptr[r0 + b + 1];
            if (bs == be) continue;
            if (a == b) {
                for (int p = as; p < ae; ++p) {
                    int j = idx[p];
                    double dp = data[p];
                    double *Mj = M + (size_t)j * n;
                    for (int q = bs; q < be; ++q) Mj[idx[q]] += h * dp * data[q];
                }
            } else {
                for (int p = as; p < ae; ++p) {
                    int j = idx[p];
                    double dp = h * data[p];
                    double *Mj = M + (size_t)j * n;
                    for (int q = bs; q < be; ++q) {
                        int l = idx[q];
                        double t = dp * data[q];
                        Mj[l] += t;
                        M[(size_t)l * n + j] += t;
                    }
                }
            }
        }
    }
}
