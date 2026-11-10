/* Sparse kernels for the exponential-propagation solver (loaded via ctypes). */
#include <stdint.h>
#include <math.h>
#include <stddef.h>

/* y = A x for a full CSR matrix. */
void spmv(int64_t n, const int64_t *ip, const uint32_t *ix, const double *a,
          const double *x, double *y)
{
    for (int64_t i = 0; i < n; i++) {
        double s = 0.0;
        for (int64_t j = ip[i]; j < ip[i + 1]; j++) s += a[j] * x[ix[j]];
        y[i] = s;
    }
}

static inline double *row_ptr(double *V, int64_t n, int64_t j, int64_t lim, double *ring)
{
    return j < lim ? V + j * n : ring + (j % 3) * n;
}

/* Lanczos steps k0..k1-1.  Row j of the basis lives in V (j < lim) or in a
 * three-slot ring buffer.  alpha[k], beta[k] receive the recurrence
 * coefficients; v_{k+1} is normalised in place.  If acc != NULL the rows
 * k0+1..k1 are accumulated: acc += coef[j] * v_j (and acc2 += coef2[j] * v_j
 * when acc2 != NULL).  Returns the number of
 * completed steps' end index (k1, or k+1 at breakdown). */
int64_t lanczos_steps(int64_t n, const int64_t *ip, const uint32_t *ix, const double *a,
                      double *V, int64_t lim, double *ring, int64_t k0, int64_t k1,
                      double *alpha, double *beta, double brk,
                      double *acc, const double *coef, double *acc2, const double *coef2)
{
    for (int64_t k = k0; k < k1; k++) {
        double *v = row_ptr(V, n, k, lim, ring);
        double *w = row_ptr(V, n, k + 1, lim, ring);
        double al = 0.0;
        if (k > 0) {
            const double *vp = row_ptr(V, n, k - 1, lim, ring);
            const double bp = beta[k - 1];
            for (int64_t i = 0; i < n; i++) {
                double s = 0.0;
                for (int64_t j = ip[i]; j < ip[i + 1]; j++) s += a[j] * v[ix[j]];
                s -= bp * vp[i];
                w[i] = s;
                al += v[i] * s;
            }
        } else {
            for (int64_t i = 0; i < n; i++) {
                double s = 0.0;
                for (int64_t j = ip[i]; j < ip[i + 1]; j++) s += a[j] * v[ix[j]];
                w[i] = s;
                al += v[i] * s;
            }
        }
        double nr = 0.0;
        for (int64_t i = 0; i < n; i++) {
            double s = w[i] - al * v[i];
            w[i] = s;
            nr += s * s;
        }
        nr = sqrt(nr);
        alpha[k] = al;
        beta[k] = nr;
        if (!(nr > brk)) return k + 1;
        const double inv = 1.0 / nr;
        if (acc && acc2) {
            const double c = coef[k + 1], c2 = coef2[k + 1];
            for (int64_t i = 0; i < n; i++) {
                double s = w[i] * inv; w[i] = s; acc[i] += c * s; acc2[i] += c2 * s;
            }
        } else if (acc) {
            const double c = coef[k + 1];
            for (int64_t i = 0; i < n; i++) { double s = w[i] * inv; w[i] = s; acc[i] += c * s; }
        } else {
            for (int64_t i = 0; i < n; i++) w[i] *= inv;
        }
    }
    return k1;
}

/* Maximum absolute row sum (infinity norm) of a CSR matrix. */
double csr_inf_norm(int64_t n, const int64_t *ip, const double *a)
{
    double m = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double s = 0.0;
        for (int64_t j = ip[i]; j < ip[i + 1]; j++) s += fabs(a[j]);
        if (s > m) m = s;
    }
    return m;
}

/* One pass over the matrix: out[0] = infinity norm, out[1] = Gershgorin lower
 * bound min_i (a_ii - sum_{j!=i}|a_ij|), out[2] = mean |i-j| / n over entries,
 * out[3] = 1 if the matrix is diagonal. */
void csr_stats(int64_t n, const int64_t *ip, const uint32_t *ix, const double *a, double *out)
{
    double nrm = 0.0, low = INFINITY, dist = 0.0;
    int diag = 1;
    for (int64_t i = 0; i < n; i++) {
        double d = 0.0, off = 0.0;
        for (int64_t j = ip[i]; j < ip[i + 1]; j++) {
            const int64_t c = (int64_t)ix[j];
            if (c == i) d += a[j];
            else { off += fabs(a[j]); diag = 0; dist += (double)(c > i ? c - i : i - c); }
        }
        const double r = fabs(d) + off;
        if (r > nrm) nrm = r;
        if (d - off < low) low = d - off;
    }
    out[0] = nrm;
    out[1] = low;
    out[2] = ip[n] > 0 ? dist / ((double)ip[n] * (double)n) : 0.0;
    out[3] = diag;
}

/* Gershgorin lower bound min_i (a_ii - sum_{j!=i} |a_ij|) of the spectrum. */
double csr_gersh_low(int64_t n, const int64_t *ip, const uint32_t *ix, const double *a)
{
    double m = INFINITY;
    for (int64_t i = 0; i < n; i++) {
        double d = 0.0, off = 0.0;
        for (int64_t j = ip[i]; j < ip[i + 1]; j++) {
            if ((int64_t)ix[j] == i) d += a[j]; else off += fabs(a[j]);
        }
        double v = d - off;
        if (v < m) m = v;
    }
    return m;
}

/* 1 if every stored entry is on the diagonal, else 0. */
int csr_is_diagonal(int64_t n, const int64_t *ip, const uint32_t *ix)
{
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = ip[i]; j < ip[i + 1]; j++)
            if ((int64_t)ix[j] != i) return 0;
    return 1;
}

/* out = sum_{j<m} g[j] T_j(X) e1,  X = (2T - (hi+lo)I)/(hi-lo), T = tridiag(be, al, be)
 * of order k.  work must hold 3k doubles. */
void cheb_tridiag(int64_t k, const double *al, const double *be, double lo, double hi,
                  const double *g, int64_t m, double *out, double *work)
{
    const double s = 2.0 / (hi - lo), sh = (hi + lo) / (hi - lo);
    double *u0 = work, *u1 = work + k, *u2 = work + 2 * k;
    for (int64_t i = 0; i < k; i++) { u0[i] = 0.0; u1[i] = 0.0; u2[i] = 0.0; out[i] = 0.0; }
    u0[0] = 1.0;
    out[0] = g[0];
    if (m < 2) return;
    /* u1 = X e1 */
    u1[0] = s * al[0] - sh;
    if (k > 1) u1[1] = s * be[0];
    out[0] += g[1] * u1[0];
    if (k > 1) out[1] += g[1] * u1[1];
    for (int64_t j = 2; j < m; j++) {
        int64_t len = j + 1 < k ? j + 1 : k;
        const double gj = g[j];
        for (int64_t i = 0; i < len; i++) {
            double xu = al[i] * u1[i];
            if (i > 0) xu += be[i - 1] * u1[i - 1];
            if (i + 1 < k) xu += be[i] * u1[i + 1];
            double v = 2.0 * (s * xu - sh * u1[i]) - u0[i];
            u2[i] = v;
            out[i] += gj * v;
        }
        double *tmp = u0; u0 = u1; u1 = u2; u2 = tmp;
    }
}

/* Two independent Lanczos recurrences advanced in lockstep with one pass
 * over the matrix per step.  The current vectors are kept interleaved in X2
 * (2n doubles) so one cache miss serves both gathers.  Arithmetic per vector
 * is identical to lanczos_steps.  Stops early (returning k+1) if either
 * recurrence breaks down. */
int64_t lanczos2_steps(int64_t n, const int64_t *ip, const uint32_t *ix, const double *a,
                       double *V0, int64_t lim0, double *ring0, double *alpha0, double *beta0,
                       double *V1, int64_t lim1, double *ring1, double *alpha1, double *beta1,
                       int64_t k0, int64_t k1, double brk, double *X2)
{
    {
        const double *v0 = row_ptr(V0, n, k0, lim0, ring0), *v1 = row_ptr(V1, n, k0, lim1, ring1);
        for (int64_t i = 0; i < n; i++) { X2[2 * i] = v0[i]; X2[2 * i + 1] = v1[i]; }
    }
    for (int64_t k = k0; k < k1; k++) {
        double *v0 = row_ptr(V0, n, k, lim0, ring0), *w0 = row_ptr(V0, n, k + 1, lim0, ring0);
        double *v1 = row_ptr(V1, n, k, lim1, ring1), *w1 = row_ptr(V1, n, k + 1, lim1, ring1);
        double al0 = 0.0, al1 = 0.0;
        const double *p0 = k > 0 ? row_ptr(V0, n, k - 1, lim0, ring0) : NULL;
        const double *p1 = k > 0 ? row_ptr(V1, n, k - 1, lim1, ring1) : NULL;
        const double b0 = k > 0 ? beta0[k - 1] : 0.0, b1 = k > 0 ? beta1[k - 1] : 0.0;
        for (int64_t i = 0; i < n; i++) {
            double s0 = 0.0, s1 = 0.0;
            for (int64_t j = ip[i]; j < ip[i + 1]; j++) {
                const double aj = a[j];
                const double *xc = X2 + 2 * (int64_t)ix[j];
                s0 += aj * xc[0];
                s1 += aj * xc[1];
            }
            if (p0) { s0 -= b0 * p0[i]; s1 -= b1 * p1[i]; }
            w0[i] = s0; w1[i] = s1;
            al0 += v0[i] * s0; al1 += v1[i] * s1;
        }
        double n0 = 0.0, n1 = 0.0;
        for (int64_t i = 0; i < n; i++) {
            double s0 = w0[i] - al0 * v0[i];
            double s1 = w1[i] - al1 * v1[i];
            w0[i] = s0; w1[i] = s1;
            n0 += s0 * s0; n1 += s1 * s1;
        }
        n0 = sqrt(n0); n1 = sqrt(n1);
        alpha0[k] = al0; beta0[k] = n0;
        alpha1[k] = al1; beta1[k] = n1;
        if (!(n0 > brk) || !(n1 > brk)) {
            /* normalise the surviving recurrence exactly as lanczos_steps would */
            if (n0 > brk) { const double i0 = 1.0 / n0; for (int64_t i = 0; i < n; i++) w0[i] *= i0; }
            if (n1 > brk) { const double i1 = 1.0 / n1; for (int64_t i = 0; i < n; i++) w1[i] *= i1; }
            return k + 1;
        }
        const double i0 = 1.0 / n0, i1 = 1.0 / n1;
        for (int64_t i = 0; i < n; i++) {
            double s0 = w0[i] * i0, s1 = w1[i] * i1;
            w0[i] = s0; w1[i] = s1;
            X2[2 * i] = s0; X2[2 * i + 1] = s1;
        }
    }
    return k1;
}
