/* Fused Chebyshev kernels for y = exp(-tL) b0 + t*phi1(-tL) b1, L sparse symmetric PSD (CSR). */
#include <stdint.h>
#include <math.h>
#include <string.h>

/* y = L x */
void spmv1(int64_t n, const uint64_t *ip, const uint32_t *ix, const double *a,
           const double *x, double *y)
{
    for (int64_t i = 0; i < n; i++) {
        double s = 0.0;
        for (uint64_t p = ip[i]; p < ip[i + 1]; p++) s += a[p] * x[ix[p]];
        y[i] = s;
    }
}

/* maximum absolute row sum (Gershgorin bound for symmetric L) */
double rowabsmax(int64_t n, const uint64_t *ip, const double *a)
{
    double m = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double s = 0.0;
        for (uint64_t p = ip[i]; p < ip[i + 1]; p++) s += fabs(a[p]);
        if (s > m) m = s;
    }
    return m;
}

/* One step on interleaved pairs: vn = ca*(L vc) + cb*vc + cc*vp ; y += c0*vn[:,0] + c1*vn[:,1].
   out[0]=max|vn0|, out[1]=max|vn1|, out[2]=max|y| */
static void step2_f64(int64_t n, const uint64_t *ip, const uint32_t *ix, const double *a,
                  double ca, double cb, double cc,
                  const double *vp, const double *vc, double *vn,
                  double c0, double c1, double *y, double *out)
{
    double m0 = 0.0, m1 = 0.0, my = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double s0 = 0.0, s1 = 0.0;
        uint64_t p = ip[i], e = ip[i + 1];
        for (; p < e; p++) {
            double v = a[p];
            const double *xx = vc + 2 * (uint64_t)ix[p];
            s0 += v * xx[0];
            s1 += v * xx[1];
        }
        double w0 = ca * s0 + cb * vc[2 * i] + cc * vp[2 * i];
        double w1 = ca * s1 + cb * vc[2 * i + 1] + cc * vp[2 * i + 1];
        vn[2 * i] = w0;
        vn[2 * i + 1] = w1;
        double yy = y[i] + c0 * w0 + c1 * w1;
        y[i] = yy;
        w0 = fabs(w0); w1 = fabs(w1); yy = fabs(yy);
        if (w0 > m0) m0 = w0;
        if (w1 > m1) m1 = w1;
        if (yy > my) my = yy;
    }
    out[0] = m0; out[1] = m1; out[2] = my;
}

/* Single-vector variant (used when one of b0/b1 is identically zero). */
static void step1_f64(int64_t n, const uint64_t *ip, const uint32_t *ix, const double *a,
                  double ca, double cb, double cc,
                  const double *vp, const double *vc, double *vn,
                  double c0, double *y, double *out)
{
    double m0 = 0.0, my = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double s0 = 0.0;
        uint64_t p = ip[i], e = ip[i + 1];
        for (; p < e; p++) s0 += a[p] * vc[ix[p]];
        double w0 = ca * s0 + cb * vc[i] + cc * vp[i];
        vn[i] = w0;
        double yy = y[i] + c0 * w0;
        y[i] = yy;
        w0 = fabs(w0); yy = fabs(yy);
        if (w0 > m0) m0 = w0;
        if (yy > my) my = yy;
    }
    out[0] = m0; out[1] = 0.0; out[2] = my;
}

/* One step on interleaved pairs: vn = ca*(L vc) + cb*vc + cc*vp ; y += c0*vn[:,0] + c1*vn[:,1].
   out[0]=max|vn0|, out[1]=max|vn1|, out[2]=max|y| */
static void step2_f32(int64_t n, const uint64_t *ip, const uint32_t *ix, const float *a,
                  double ca, double cb, double cc,
                  const double *vp, const double *vc, double *vn,
                  double c0, double c1, double *y, double *out)
{
    double m0 = 0.0, m1 = 0.0, my = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double s0 = 0.0, s1 = 0.0;
        uint64_t p = ip[i], e = ip[i + 1];
        for (; p < e; p++) {
            double v = (double)a[p];
            const double *xx = vc + 2 * (uint64_t)ix[p];
            s0 += v * xx[0];
            s1 += v * xx[1];
        }
        double w0 = ca * s0 + cb * vc[2 * i] + cc * vp[2 * i];
        double w1 = ca * s1 + cb * vc[2 * i + 1] + cc * vp[2 * i + 1];
        vn[2 * i] = w0;
        vn[2 * i + 1] = w1;
        double yy = y[i] + c0 * w0 + c1 * w1;
        y[i] = yy;
        w0 = fabs(w0); w1 = fabs(w1); yy = fabs(yy);
        if (w0 > m0) m0 = w0;
        if (w1 > m1) m1 = w1;
        if (yy > my) my = yy;
    }
    out[0] = m0; out[1] = m1; out[2] = my;
}

/* Single-vector variant (used when one of b0/b1 is identically zero). */
static void step1_f32(int64_t n, const uint64_t *ip, const uint32_t *ix, const float *a,
                  double ca, double cb, double cc,
                  const double *vp, const double *vc, double *vn,
                  double c0, double *y, double *out)
{
    double m0 = 0.0, my = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double s0 = 0.0;
        uint64_t p = ip[i], e = ip[i + 1];
        for (; p < e; p++) s0 += (double)a[p] * vc[ix[p]];
        double w0 = ca * s0 + cb * vc[i] + cc * vp[i];
        vn[i] = w0;
        double yy = y[i] + c0 * w0;
        y[i] = yy;
        w0 = fabs(w0); yy = fabs(yy);
        if (w0 > m0) m0 = w0;
        if (yy > my) my = yy;
    }
    out[0] = m0; out[1] = 0.0; out[2] = my;
}

/* Chebyshev series on [0,ub].  c0/c1: coefficients (length K+1, c[0] already halved),
   S0/S1: tail sums S[k] = sum_{j>k} |c[j]|.  Stops at the first k with
   safety*(S0[k]*M0 + S1[k]*M1) <= frac*(atol + rtol*max|y|), M = running max of |T_j b|.
   work: 6n doubles.  nvec: 2 (b0,b1 both used), or 1 with only b (coefs c0) used.
   Returns number of terms used (k), or -1 on detected divergence, -2 if K reached. */
int64_t cheb_run(int64_t n, const uint64_t *ip, const uint32_t *ix, const void *av, int f32,
                 double lb, double ub, int64_t K, const double *c0, const double *c1,
                 const double *S0, const double *S1,
                 double atol, double rtol, double frac, double safety, double lim0, double lim1,
                 int nvec, const double *b0, const double *b1, double *y, double *work,
                 double *info)
{
    double out[3];
    const double *a = (const double *)av;
    const float *a32 = (const float *)av;
    double *A = work, *B = work + 2 * n, *C = work + 4 * n;
    double M0 = 0.0, M1 = 0.0, my = 0.0;
    if (nvec == 2) {
        for (int64_t i = 0; i < n; i++) {
            A[2 * i] = b0[i]; A[2 * i + 1] = b1[i];
            y[i] = c0[0] * b0[i] + c1[0] * b1[i];
            double u = fabs(b0[i]), v = fabs(b1[i]), w = fabs(y[i]);
            if (u > M0) M0 = u;
            if (v > M1) M1 = v;
            if (w > my) my = w;
        }
    } else {
        for (int64_t i = 0; i < n; i++) {
            A[i] = b0[i];
            y[i] = c0[0] * b0[i];
            double u = fabs(b0[i]), w = fabs(y[i]);
            if (u > M0) M0 = u;
            if (w > my) my = w;
        }
    }
    double tol;
    if (K <= 0) { info[0] = 0; return 0; }
    tol = atol + rtol * my;
    if (safety * (S0[0] * M0 + S1[0] * M1) <= frac * tol) { info[0] = 0; return 0; }
    double wid = ub - lb, sh = 1.0 + 2.0 * lb / wid;
    /* T1 = (2/w) L b - (1+2lb/w) b */
    if (nvec == 2) { if (f32) step2_f32(n, ip, ix, a32, 2.0 / wid, -sh, 0.0, A, A, B, c0[1], c1[1], y, out); else step2_f64(n, ip, ix, a, 2.0 / wid, -sh, 0.0, A, A, B, c0[1], c1[1], y, out); }
    else { if (f32) step1_f32(n, ip, ix, a32, 2.0 / wid, -sh, 0.0, A, A, B, c0[1], y, out); else step1_f64(n, ip, ix, a, 2.0 / wid, -sh, 0.0, A, A, B, c0[1], y, out); }
    if (out[0] > M0) M0 = out[0];
    if (out[1] > M1) M1 = out[1];
    my = out[2];
    int64_t k = 1;
    double ca = 4.0 / wid, cb = -2.0 * sh;
    for (;;) {
        tol = atol + rtol * my;
        if (safety * (S0[k] * M0 + S1[k] * M1) <= frac * tol) break;
        if (k >= K) { info[0] = (double)k; info[1] = M0; info[2] = M1; return -2; }
        if (M0 > lim0 || M1 > lim1) {  /* |T_k b|_inf <= |b|_2 on a valid interval */
            info[0] = (double)k; info[1] = M0; info[2] = M1; return -1;
        }
        k++;
        if (nvec == 2) { if (f32) step2_f32(n, ip, ix, a32, ca, cb, -1.0, A, B, C, c0[k], c1[k], y, out); else step2_f64(n, ip, ix, a, ca, cb, -1.0, A, B, C, c0[k], c1[k], y, out); }
        else { if (f32) step1_f32(n, ip, ix, a32, ca, cb, -1.0, A, B, C, c0[k], y, out); else step1_f64(n, ip, ix, a, ca, cb, -1.0, A, B, C, c0[k], y, out); }
        if (out[0] > M0) M0 = out[0];
        if (out[1] > M1) M1 = out[1];
        my = out[2];
        double *T = A; A = B; B = C; C = T;
    }
    info[0] = (double)k; info[1] = M0; info[2] = M1; info[3] = my;
    return k;
}

/* Spectral bounds from the stored entries.  out[0]=Gershgorin upper, out[1]=Gershgorin lower,
   out[2]=max over stored off-diagonal (i,j) of L_ii+L_jj and max L_ii (Anderson-Morley type,
   valid when L is diagonally dominant), out[3]=min_i (L_ii - R_i)/max(R_i,tiny) (dominance margin). */
void bounds(int64_t n, const uint64_t *ip, const uint32_t *ix, const double *a, double *diag, double *out)
{
    double gu = 0.0, gl = INFINITY, dm = INFINITY, am = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double d = 0.0, r = 0.0;
        for (uint64_t p = ip[i]; p < ip[i + 1]; p++) {
            if ((int64_t)ix[p] == i) d += a[p]; else r += fabs(a[p]);
        }
        diag[i] = d;
        if (d + r > gu) gu = d + r;
        if (fabs(d) + r > gu) gu = fabs(d) + r;
        if (d - r < gl) gl = d - r;
        double marg = (d - r) / (r > 1e-300 ? r : 1e-300);
        if (r == 0.0) marg = (d >= 0.0) ? 1.0 : -1.0;
        if (marg < dm) dm = marg;
        if (d > am) am = d;
    }
    for (int64_t i = 0; i < n; i++) {
        for (uint64_t p = ip[i]; p < ip[i + 1]; p++) {
            int64_t j = ix[p];
            if (j != i && a[p] != 0.0) {
                double s = diag[i] + diag[j];
                if (s > am) am = s;
            }
        }
    }
    if (n == 0) gl = 0.0;
    out[0] = gu; out[1] = gl; out[2] = am; out[3] = dm;
}

/* Lanczos steps without reorthogonalisation.  state: v (n), vold (n), w (n); st[0] = beta (previous).
   Writes alpha[s], beta[s] for s < steps; returns steps done (stops early on breakdown). */
int64_t lanczos_steps(int64_t n, const uint64_t *ip, const uint32_t *ix, const double *a,
                      int64_t steps, double *v, double *vold, double *w, double *st,
                      double *alpha, double *beta)
{
    double b = st[0];
    for (int64_t s = 0; s < steps; s++) {
        double al = 0.0;
        for (int64_t i = 0; i < n; i++) {
            double acc = 0.0;
            for (uint64_t p = ip[i]; p < ip[i + 1]; p++) acc += a[p] * v[ix[p]];
            w[i] = acc;
            al += acc * v[i];
        }
        double nr = 0.0;
        for (int64_t i = 0; i < n; i++) {
            double x = w[i] - al * v[i] - b * vold[i];
            w[i] = x;
            nr += x * x;
        }
        nr = sqrt(nr);
        alpha[s] = al;
        beta[s] = nr;
        if (!(nr > 1e-13 * (fabs(al) > 1e-300 ? fabs(al) : 1e-300))) { st[0] = nr; return s + 1; }
        double inv = 1.0 / nr;
        for (int64_t i = 0; i < n; i++) {
            vold[i] = v[i];
            v[i] = w[i] * inv;
        }
        b = nr;
    }
    st[0] = b;
    return steps;
}
