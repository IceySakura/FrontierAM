/* expkernel.c -- hot kernels for sparse exponential propagation (FAMEXP2).
 *
 * Compiled to a shared object and driven through ctypes by solve.py.
 * All matrices are CSR with 64-bit row pointers and 32-bit column indices.
 *
 *   exp_prep   : ||L||_inf and the Gershgorin lower spectral bound.
 *   exp_spmv   : y = L x, used by the power iteration for lambda_max.
 *   exp_cheb   : forward Chebyshev evaluation of
 *                y = f0(L) b0 + f1(L) b1,  f0(x)=exp(-t x),
 *                f1(x)=(1-exp(-t x))/x,
 *                with an a-posteriori truncation test on the coefficient tail.
 */
#include <math.h>
#include <stdint.h>
#include <stddef.h>

typedef int64_t i64;
typedef int32_t i32;

void exp_prep(const i64 *indptr, const i32 *indices, const double *data,
              i64 n, double *out)
{
    double mx = 0.0, mn = INFINITY;
    for (i64 i = 0; i < n; i++) {
        i64 s = indptr[i], e = indptr[i + 1];
        double rs = 0.0, dg = 0.0;
        for (i64 k = s; k < e; k++) {
            double v = data[k];
            rs += fabs(v);
            if (indices[k] == (i32)i) dg = v;
        }
        if (rs > mx) mx = rs;
        {
            double g = dg - (rs - fabs(dg));
            if (g < mn) mn = g;
        }
    }
    out[0] = mx;
    out[1] = (isfinite(mn) ? mn : 0.0);
}

void exp_spmv(const i64 *indptr, const i32 *indices, const double *data,
              i64 n, const double *x, double *y)
{
    for (i64 i = 0; i < n; i++) {
        i64 s = indptr[i], e = indptr[i + 1];
        double a = 0.0;
        for (i64 k = s; k < e; k++) a += data[k] * x[indices[k]];
        y[i] = a;
    }
}

/* work must hold 4*n doubles per right-hand side (A,B,C,W). */
i64 exp_cheb(const i64 *indptr, const i32 *indices, const double *data, i64 n,
             double scale, double shift,
             const double *c0, const double *c1, i64 dmax,
             const double *t0, const double *t1,
             const double *b0, const double *b1,
             double nb0, double nb1,
             double sa, double sr, double tfloor,
             double *y, double *work, i64 *pused, double *pymax)
{
    const i64 nrhs = (b1 != NULL) ? 2 : 1;
    double *A[2], *B[2], *C[2], *W[2];
    const double *bb[2], *cc[2], *tt[2];
    double nb[2];
    i64 r, i, k;
    double ym = 0.0;

    for (r = 0; r < nrhs; r++) {
        A[r] = work + (4 * r + 0) * n;
        B[r] = work + (4 * r + 1) * n;
        C[r] = work + (4 * r + 2) * n;
        W[r] = work + (4 * r + 3) * n;
    }
    bb[0] = b0; bb[1] = b1;
    cc[0] = c0; cc[1] = c1;
    tt[0] = t0; tt[1] = t1;
    nb[0] = nb0; nb[1] = nb1;

    for (i = 0; i < n; i++) y[i] = 0.0;

    /* k = 0 */
    for (r = 0; r < nrhs; r++) {
        const double *v = bb[r];
        double g = cc[r][0];
        double *a = A[r];
        if (g != 0.0) {
            for (i = 0; i < n; i++) { a[i] = v[i]; y[i] += g * v[i]; }
        } else {
            for (i = 0; i < n; i++) a[i] = v[i];
        }
    }
    if (dmax <= 0) { *pused = 0; *pymax = ym; return 0; }

    /* k = 1 :  B = scale*L*A - shift*A */
    for (r = 0; r < nrhs; r++) {
        const double *a = A[r];
        double *w = W[r], *b = B[r];
        double g = cc[r][1];
        for (i = 0; i < n; i++) w[i] = -shift * a[i];
        for (i = 0; i < n; i++) {
            i64 s = indptr[i], e = indptr[i + 1];
            double acc = 0.0;
            for (k = s; k < e; k++) acc += data[k] * a[indices[k]];
            w[i] += scale * acc;
        }
        for (i = 0; i < n; i++) {
            double v = 2.0 * w[i];
            b[i] = v;
            y[i] += g * v;
        }
    }
    k = 1;

    for (;;) {
        double thr, e0, e1, est;
        for (i = 0; i < n; i++) { double ay = fabs(y[i]); if (ay > ym) ym = ay; }
        thr = sa + sr * ym;
        e0 = t0[k + 1] * nb0;
        e1 = (nrhs > 1) ? (t1[k + 1] * nb1) : 0.0;
        est = e0 + e1;
        if (k >= dmax || est <= thr || est <= tfloor) break;

        /* T_{k+1} = 2*H*T_k - T_{k-1} */
        for (r = 0; r < nrhs; r++) {
            const double *a = A[r], *b = B[r];
            double *w = W[r], *c = C[r];
            double g = cc[r][k + 1];
            for (i = 0; i < n; i++) w[i] = -shift * b[i];
            for (i = 0; i < n; i++) {
                i64 s = indptr[i], e = indptr[i + 1];
                double acc = 0.0;
                i64 q;
                for (q = s; q < e; q++) acc += data[q] * b[indices[q]];
                w[i] += scale * acc;
            }
            for (i = 0; i < n; i++) {
                double v = 2.0 * w[i] - a[i];
                c[i] = v;
                y[i] += g * v;
            }
            { double *tmp = A[r]; A[r] = B[r]; B[r] = C[r]; C[r] = tmp; }
        }
        k++;
    }
    *pused = k;
    *pymax = ym;
    return k;
}
