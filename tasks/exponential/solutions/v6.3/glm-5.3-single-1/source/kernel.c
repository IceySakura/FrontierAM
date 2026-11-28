/* FAMEXP2 kernels: CSR block products and fused Chebyshev recurrences.
 *
 * The wire delivers a full symmetric CSR matrix (indptr uint64, indices uint32,
 * data float64) with sorted, unique column indices per row.  Because the matrix
 * is symmetric the same triple describes both the CSR and the CSC layout, but
 * all kernels below use the row oriented (gather) form which fuses cleanly with
 * the three term Chebyshev recurrence.
 *
 * Everything is compiled without OpenMP: the process group owns one CPU.
 */

#include <stdint.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* max_i sum_j |L_ij| : rigorous upper bound of lambda_max (PSD input) */
double k_inf_norm(int64_t n, const uint64_t *indptr, const double *data)
{
    double best = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const uint64_t p1 = indptr[i + 1];
        double s = 0.0;
        for (uint64_t p = indptr[i]; p < p1; ++p) s += fabs(data[p]);
        if (s > best) best = s;
    }
    return best;
}

/* ------------------------------------------------------------------ */
/* y = L x  (x, y are n x k row major, k = 2)                          */
void k_matvec2(int64_t n, const uint64_t *restrict indptr,
               const uint32_t *restrict indices, const double *restrict data,
               const double *restrict x, double *restrict y)
{
    for (int64_t i = 0; i < n; ++i) {
        const uint64_t p1 = indptr[i + 1];
        double s0 = 0.0, s1 = 0.0;
        for (uint64_t p = indptr[i]; p < p1; ++p) {
            const int64_t j = 2 * (int64_t)indices[p];
            const double a = data[p];
            s0 += a * x[j];
            s1 += a * x[j + 1];
        }
        y[2 * i] = s0;
        y[2 * i + 1] = s1;
    }
}

/* y = L x (single column) */
void k_matvec1(int64_t n, const uint64_t *restrict indptr,
               const uint32_t *restrict indices, const double *restrict data,
               const double *restrict x, double *restrict y)
{
    for (int64_t i = 0; i < n; ++i) {
        const uint64_t p1 = indptr[i + 1];
        double s = 0.0;
        for (uint64_t p = indptr[i]; p < p1; ++p) {
            s += data[p] * x[indices[p]];
        }
        y[i] = s;
    }
}

/* ------------------------------------------------------------------ */
/* Fused Chebyshev recurrence, block of two columns.
 *
 *   W_{k+1} = 2*(c1*L + c0*I) W_k - W_{k-1}
 *   Y      += ce[s]*W_{k+1}[col0] + cp[s]*W_{k+1}[col1]
 *
 * On entry bufA holds W_{k-1} and bufB holds W_k (each n x 2, row major).
 * nsteps recurrence steps are performed.  With keep_state != 0 the final state
 * is returned canonically: bufA = W_{k+nsteps-1}, bufB = W_{k+nsteps}.
 */
void k_cheb2(int64_t n, const uint64_t *restrict indptr,
             const uint32_t *restrict indices, const double *restrict data,
             double c1, double c0, int64_t nsteps,
             const double *restrict ce, const double *restrict cp,
             double *restrict bufA, double *restrict bufB,
             double *restrict Y, int32_t keep_state)
{
    double *Wp = bufA, *Wc = bufB;
    const double t1 = 2.0 * c1, t0 = 2.0 * c0;
    for (int64_t s = 0; s < nsteps; ++s) {
        const double e = (ce != 0) ? ce[s] : 0.0;
        const double p = (cp != 0) ? cp[s] : 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const uint64_t q1 = indptr[i + 1];
            double s0 = 0.0, s1 = 0.0;
            for (uint64_t q = indptr[i]; q < q1; ++q) {
                const int64_t j = 2 * (int64_t)indices[q];
                const double a = data[q];
                s0 += a * Wc[j];
                s1 += a * Wc[j + 1];
            }
            const double n0 = t1 * s0 + t0 * Wc[2 * i] - Wp[2 * i];
            const double n1 = t1 * s1 + t0 * Wc[2 * i + 1] - Wp[2 * i + 1];
            Wp[2 * i] = n0;
            Wp[2 * i + 1] = n1;
            Y[i] += e * n0 + p * n1;
        }
        double *tmp = Wp;
        Wp = Wc;
        Wc = tmp;
    }
    if (keep_state && (nsteps & 1)) {
        for (int64_t i = 0; i < 2 * n; ++i) {
            const double tt = bufA[i];
            bufA[i] = bufB[i];
            bufB[i] = tt;
        }
    }
}

/* Same recurrence for a single column. */
void k_cheb1(int64_t n, const uint64_t *restrict indptr,
             const uint32_t *restrict indices, const double *restrict data,
             double c1, double c0, int64_t nsteps,
             const double *restrict c,
             double *restrict bufA, double *restrict bufB,
             double *restrict Y)
{
    double *Wp = bufA, *Wc = bufB;
    const double t1 = 2.0 * c1, t0 = 2.0 * c0;
    for (int64_t s = 0; s < nsteps; ++s) {
        const double e = c[s];
        for (int64_t i = 0; i < n; ++i) {
            const uint64_t q1 = indptr[i + 1];
            double sm = 0.0;
            for (uint64_t q = indptr[i]; q < q1; ++q) {
                sm += data[q] * Wc[indices[q]];
            }
            const double nv = t1 * sm + t0 * Wc[i] - Wp[i];
            Wp[i] = nv;
            Y[i] += e * nv;
        }
        double *tmp = Wp;
        Wp = Wc;
        Wc = tmp;
    }
}

/* ------------------------------------------------------------------ */
/* Symmetric (half stored) triangular product used by the Krylov path:
 * uptr holds, per row i, only entries with column index > i, and a separate
 * diagonal array.  y = (D + U + U^T) x for a single column.
 */
void k_symatv(int64_t n, const uint64_t *restrict uptr, const uint32_t *restrict uind,
              const double *restrict uval, const double *restrict diag,
              const double *restrict x, double *restrict y)
{
    for (int64_t i = 0; i < n; ++i) y[i] = diag[i] * x[i];
    for (int64_t i = 0; i < n; ++i) {
        const uint64_t q1 = uptr[i + 1];
        const double xi = x[i];
        double s = 0.0;
        for (uint64_t q = uptr[i]; q < q1; ++q) {
            const int64_t j = uind[q];
            const double a = uval[q];
            s += a * x[j];
            y[j] += a * xi;
        }
        y[i] += s;
    }
}

/* block-of-two version of the same product */
void k_symatv2(int64_t n, const uint64_t *restrict uptr, const uint32_t *restrict uind,
               const double *restrict uval, const double *restrict diag,
               const double *restrict x, double *restrict y)
{
    for (int64_t i = 0; i < n; ++i) {
        y[2 * i] = diag[i] * x[2 * i];
        y[2 * i + 1] = diag[i] * x[2 * i + 1];
    }
    for (int64_t i = 0; i < n; ++i) {
        const uint64_t q1 = uptr[i + 1];
        const double x0 = x[2 * i], x1 = x[2 * i + 1];
        double s0 = 0.0, s1 = 0.0;
        for (uint64_t q = uptr[i]; q < q1; ++q) {
            const int64_t j = 2 * (int64_t)uind[q];
            const double a = uval[q];
            s0 += a * x[j];
            s1 += a * x[j + 1];
            y[j] += a * x0;
            y[j + 1] += a * x1;
        }
        y[2 * i] += s0;
        y[2 * i + 1] += s1;
    }
}

/* Extract the strict upper triangle (column > row) plus the diagonal of a
 * symmetric CSR matrix into a compact CSR-like structure.  Used once per
 * session, before the first request. */
int64_t k_split_upper(int64_t n, const uint64_t *restrict indptr,
                      const uint32_t *restrict indices, const double *restrict data,
                      uint64_t *restrict uptr, uint32_t *restrict uind,
                      double *restrict uval, double *restrict diag)
{
    int64_t cnt = 0;
    for (int64_t i = 0; i < n; ++i) diag[i] = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        uptr[i] = (uint64_t)cnt;
        const uint64_t p1 = indptr[i + 1];
        for (uint64_t p = indptr[i]; p < p1; ++p) {
            const uint32_t j = indices[p];
            if ((int64_t)j > i) {
                uind[cnt] = j;
                uval[cnt] = data[p];
                ++cnt;
            } else if ((int64_t)j == i) {
                diag[i] = data[p];
            }
        }
    }
    uptr[n] = (uint64_t)cnt;
    return cnt;
}
