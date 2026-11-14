/* fastcore.c -- hot loops for the FAMEXP2 exponential propagation task.
 *
 * Works directly on the CSR arrays as they arrive on the wire (int64 indptr,
 * int32 indices, float64 data): no conversion or copy of the operator.
 * Single threaded by construction (the session gets one CPU).
 */
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

typedef int64_t i64;
typedef int32_t i32;

/* max_i sum_j |L_ij| : rigorous upper bound on lambda_max for symmetric L. */
double rowabsmax(i64 n, const i64 *ip, const double *d)
{
    double best = 0.0;
    for (i64 i = 0; i < n; i++) {
        const i64 s = ip[i], e = ip[i + 1];
        double acc = 0.0;
        for (i64 j = s; j < e; j++) acc += fabs(d[j]);
        if (acc > best) best = acc;
    }
    return best;
}

double vnorm2(i64 n, const double *v)
{
    double a = 0.0;
    for (i64 i = 0; i < n; i++) a += v[i] * v[i];
    return sqrt(a);
}

double vnominf(i64 n, const double *v)
{
    double a = 0.0;
    for (i64 i = 0; i < n; i++) { double t = fabs(v[i]); if (t > a) a = t; }
    return a;
}

/* y <- a*x + y  (in place) */
void axpy(i64 n, double a, const double *x, double *y)
{
    for (i64 i = 0; i < n; i++) y[i] += a * x[i];
}

/* v <- a*v */
void scal(i64 n, double a, double *v)
{
    for (i64 i = 0; i < n; i++) v[i] *= a;
}

/* w <- L v */
void csr_mv(i64 n, const i64 *ip, const i32 *ij, const double *da,
            const double *v, double *w)
{
    for (i64 i = 0; i < n; i++) {
        double s = 0.0;
        for (i64 j = ip[i], e = ip[i + 1]; j < e; j++) s += da[j] * v[ij[j]];
        w[i] = s;
    }
}

/* ---- lambda_max estimate: `steps` Lanczos steps, largest Ritz value --------
 * No reorthogonalisation: the value is only used as a *suggested* upper end of
 * the Chebyshev interval, and the caller keeps a margin plus a blow-up monitor.
 * The largest Ritz value of the tridiagonal is found by Sturm bisection.
 * work must hold 3*n + 2*steps + 2 doubles.
 */
double lam_max_est(i64 n, const i64 *ip, const i32 *ij, const double *da,
                   i64 steps, double *work)
{
    double *v = work, *w = work + n, *q = work + 2 * n;
    double *alpha = work + 3 * n, *beta = alpha + steps + 1;
    i64 i, k;
    for (i = 0; i < n; i++) v[i] = sin((double)(i * 2 + 1) * 0.3781) + 0.5;
    double nrm = 0.0;
    for (i = 0; i < n; i++) nrm += v[i] * v[i];
    nrm = sqrt(nrm);
    if (!(nrm > 0.0)) return 0.0;
    for (i = 0; i < n; i++) v[i] /= nrm;
    double bprev = 0.0;
    i64 m = 0;
    for (k = 0; k < steps; k++) {
        csr_mv(n, ip, ij, da, v, w);
        double a = 0.0;
        for (i = 0; i < n; i++) a += v[i] * w[i];
        for (i = 0; i < n; i++) w[i] -= a * v[i];
        if (k > 0) for (i = 0; i < n; i++) w[i] -= bprev * q[i];
        double b = 0.0;
        for (i = 0; i < n; i++) b += w[i] * w[i];
        b = sqrt(b);
        alpha[m] = a;
        if (k + 1 < steps) beta[m] = b;
        m++;
        if (!(b > 1e-290)) break;
        memcpy(q, v, sizeof(double) * (size_t)n);
        for (i = 0; i < n; i++) v[i] = w[i] / b;
        bprev = b;
    }
    double lo = 0.0, hi = 0.0;
    for (i = 0; i < m; i++) {
        double r = (i > 0 ? fabs(beta[i - 1]) : 0.0) + (i + 1 < m ? fabs(beta[i]) : 0.0);
        if (alpha[i] - r < lo) lo = alpha[i] - r;
        if (alpha[i] + r > hi) hi = alpha[i] + r;
    }
    if (hi <= lo) return hi;
    for (int it = 0; it < 120; it++) {
        double mid = 0.5 * (lo + hi);
        i64 cnt = 0;
        double d = alpha[0] - mid;
        if (d < 0.0) cnt++;
        if (fabs(d) < 1e-300) d = (d < 0 ? -1e-300 : 1e-300);
        for (i = 1; i < m; i++) {
            d = alpha[i] - mid - beta[i - 1] * beta[i - 1] / d;
            if (d < 0.0) cnt++;
            if (fabs(d) < 1e-300) d = (d < 0 ? -1e-300 : 1e-300);
        }
        if (cnt > 0) lo = mid; else hi = mid;
    }
    return hi;
}

/* ------------------------------------------------------------------------ */
/* Chebyshev evaluation of y = f1(L) b0 + f2(L) b1.                          */
/*   xhat v = ca1*(L v) + cb1*v                (k = 1)                       */
/*   new    = ca*(L v) + cb*v - prev           (k >= 2)                      */
/* tail1[k], tail2[k] are rigorous bounds on sum_{j>k}|c_j|; the loop stops   */
/* as soon as  tail1[k]*||b0||_2 + tail2[k]*||b1||_2 <= safety*(atol +       */
/* rtol*||y_k||_inf).                                                        */
/* Returns the number of steps performed, or -1 (blow-up: interval too        */
/* small) or -2 (non-finite).                                                */
/* ------------------------------------------------------------------------ */
/* Accept the current degree when either the rigorous 2-norm tail bound or the
 * sharper infinity-norm estimate (observed max |T_j(Lhat) b| so far, times a
 * safety factor) falls below safety*(atol + rtol*||y||_inf). */
static int stop_ok(double tl1, double tl2, double nb0, double nb1,
                   double mx0, double mx1, double atol, double rtol,
                   double ynrm, double safety, double kap)
{
    double lim = safety * (atol + rtol * ynrm);
    if (tl1 * nb0 + tl2 * nb1 <= lim) return 1;
    if (mx0 <= 0.0) mx0 = 1.0;
    if (mx1 <= 0.0) mx1 = 1.0;
    if ((tl1 * mx0 + tl2 * mx1) * kap <= lim) return 1;
    return 0;
}

#ifndef ROWBLOCK
#define ROWBLOCK 2048
#endif

static i64 cheb_run(i64 n, const i64 *ip, const i32 *ij, const double *da,
                    double ca1, double cb1, double ca, double cb,
                    i64 m, const double *c1, const double *c2,
                    const double *tail1, const double *tail2,
                    double nb0, double nb1, double atol, double rtol,
                    double safety, const double *b0, const double *b1,
                    double *y, double *ws, const int use0, const int use1,
                    double growth, i64 rb, double *ynorm_out)
{
    /* three rotating buffers per column: prev, cur, new */
    double *A0 = ws, *B0 = ws + n, *C0 = ws + 2 * n;
    double *A1 = ws + 3 * n, *B1 = ws + 4 * n, *C1 = ws + 5 * n;
    double ynrm = 0.0, mx = 0.0, mx0 = 0.0, mx1 = 0.0;
    i64 i, k, r0;
    const double kap = 6.0;

    for (i = 0; i < n; i++) {
        double v0 = use0 ? b0[i] : 0.0, v1 = use1 ? b1[i] : 0.0;
        double acc = c1[0] * v0 + c2[0] * v1;
        y[i] = acc;
        A0[i] = v0; A1[i] = v1;
        double t = fabs(acc); if (t > ynrm) ynrm = t;
    }
    if (m < 1) { *ynorm_out = ynrm; return 0; }

    /* ---- k = 1 : B = xhat * A ---- */
    {
        double local = 0.0, accmax = 0.0;
        const double ck1 = c1[1], ck2 = c2[1];
        for (r0 = 0; r0 < n; r0 += rb) {
            i64 r1 = r0 + rb; if (r1 > n) r1 = n;
            for (i = r0; i < r1; i++) {
                double s0 = 0.0, s1 = 0.0;
                for (i64 j = ip[i], e = ip[i + 1]; j < e; j++) {
                    i32 c = ij[j]; double v = da[j];
                    if (use0) s0 += v * A0[c];
                    if (use1) s1 += v * A1[c];
                }
                double n0 = use0 ? (ca1 * s0 + cb1 * A0[i]) : 0.0;
                double n1 = use1 ? (ca1 * s1 + cb1 * A1[i]) : 0.0;
                B0[i] = n0; B1[i] = n1;
                double acc = y[i] + ck1 * n0 + ck2 * n1;
                y[i] = acc;
                double t = fabs(n0); if (t > mx0) mx0 = t;
                t = fabs(n1); if (t > mx1) mx1 = t;
                t = mx0 > mx1 ? mx0 : mx1; if (t > local) local = t;
                t = fabs(acc); if (t > accmax) accmax = t;
            }
        }
        if (local > mx) mx = local;
        if (mx > growth) return -1;
        ynrm = accmax;
        if (stop_ok(tail1[1], tail2[1], nb0, nb1, mx0, mx1, atol, rtol, ynrm,
                    safety, kap)) {
            *ynorm_out = ynrm; return 1;
        }
    }

    for (k = 2; k <= m; k++) {
        const double ck1 = c1[k], ck2 = c2[k];
        double local = 0.0, accmax = 0.0;
        for (r0 = 0; r0 < n; r0 += rb) {
            i64 r1 = r0 + rb; if (r1 > n) r1 = n;
            for (i = r0; i < r1; i++) {
                double s0 = 0.0, s1 = 0.0;
                for (i64 j = ip[i], e = ip[i + 1]; j < e; j++) {
                    i32 c = ij[j]; double v = da[j];
                    if (use0) s0 += v * B0[c];
                    if (use1) s1 += v * B1[c];
                }
                double n0 = use0 ? (ca * s0 + cb * B0[i] - A0[i]) : 0.0;
                double n1 = use1 ? (ca * s1 + cb * B1[i] - A1[i]) : 0.0;
                C0[i] = n0; C1[i] = n1;
                double acc = y[i] + ck1 * n0 + ck2 * n1;
                y[i] = acc;
                double t = fabs(n0); if (t > mx0) mx0 = t;
                t = fabs(n1); if (t > mx1) mx1 = t;
                t = mx0 > mx1 ? mx0 : mx1; if (t > local) local = t;
                t = fabs(acc); if (t > accmax) accmax = t;
            }
        }
        if (local > mx) mx = local;
        if (mx > growth) return -1;
        { double *tp = A0; A0 = B0; B0 = C0; C0 = tp;
          tp = A1; A1 = B1; B1 = C1; C1 = tp; }
        ynrm = accmax;
        if (stop_ok(tail1[k], tail2[k], nb0, nb1, mx0, mx1, atol, rtol, ynrm,
                    safety, kap)) {
            for (i = 0; i < n; i++) if (!isfinite(y[i])) return -2;
            *ynorm_out = ynrm;
            return k;
        }
    }
    for (i = 0; i < n; i++) if (!isfinite(y[i])) return -2;
    *ynorm_out = ynrm;
    return m;
}

i64 cheb2(i64 n, const i64 *ip, const i32 *ij, const double *da,
          double ca1, double cb1, double ca, double cb,
          i64 m, const double *c1, const double *c2,
          const double *tail1, const double *tail2,
          double nb0, double nb1, double atol, double rtol, double safety,
          const double *b0, const double *b1, double *y, double *ws,
          int use0, int use1, double growth, i64 rb, double *ynorm_out)
{
    if (rb < 256) rb = 256;
    if (use0 && use1)
        return cheb_run(n, ip, ij, da, ca1, cb1, ca, cb, m, c1, c2, tail1, tail2,
                        nb0, nb1, atol, rtol, safety, b0, b1, y, ws, 1, 1,
                        growth, rb, ynorm_out);
    if (use0)
        return cheb_run(n, ip, ij, da, ca1, cb1, ca, cb, m, c1, c2, tail1, tail2,
                        nb0, 0.0, atol, rtol, safety, b0, b1, y, ws, 1, 0,
                        growth, rb, ynorm_out);
    return cheb_run(n, ip, ij, da, ca1, cb1, ca, cb, m, c1, c2, tail1, tail2,
                    0.0, nb1, atol, rtol, safety, b0, b1, y, ws, 0, 1,
                    growth, rb, ynorm_out);
}
