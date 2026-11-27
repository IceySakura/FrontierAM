/* Kernels for the exponential-propagation solver (loaded through ctypes).
 *
 * CSR input is used as delivered: uint64 row pointers, uint32 column indices,
 * float64 values.  Lanczos chains are stored interleaved: with c chains
 * (c = 1 or 2) vector element i of chain j lives at v[c*i + j].
 */
#include <stdint.h>
#include <math.h>
#include <string.h>

typedef double v2d __attribute__((vector_size(16)));

/* y = A x for c=1 */
static void spmv1(int64_t n, const uint64_t *ip, const uint32_t *ix,
                  const double *a, const double *x, double *y)
{
    for (int64_t i = 0; i < n; i++) {
        uint64_t k = ip[i], e = ip[i + 1];
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        for (; k + 4 <= e; k += 4) {
            s0 += a[k] * x[ix[k]];
            s1 += a[k + 1] * x[ix[k + 1]];
            s2 += a[k + 2] * x[ix[k + 2]];
            s3 += a[k + 3] * x[ix[k + 3]];
        }
        for (; k < e; k++) s0 += a[k] * x[ix[k]];
        y[i] = (s0 + s1) + (s2 + s3);
    }
}

void csr_spmv(int64_t n, const uint64_t *ip, const uint32_t *ix,
              const double *a, const double *x, double *y)
{
    spmv1(n, ip, ix, a, x, y);
}

/* One Lanczos step for c interleaved chains.
 * vp = v_{k-1}, vc = v_k, vn = output v_{k+1};  bprev[j] = beta_{k-1}.
 * If dots != 0: computes alpha[j], beta[j] (norm of w) and stores v_{k+1}.
 * If dots == 0: uses the given alpha/beta (replay pass, bitwise identical).
 * vmax[j] receives max |v_{k+1}| (only when dots != 0).
 * Returns nothing; beta may be 0 (breakdown), then vn is left unnormalized. */
void lanczos_step(int64_t n, int c, const uint64_t *ip, const uint32_t *ix,
                  const double *a, const double *vp, const double *vc,
                  double *vn, const double *bprev, double *alpha, double *beta,
                  double *vmax, int dots)
{
    if (c == 1) {
        double bp = bprev[0], al = 0.0;
        spmv1(n, ip, ix, a, vc, vn);
        if (dots) {
            for (int64_t i = 0; i < n; i++) {
                double w = vn[i] - bp * vp[i];
                vn[i] = w;
                al += w * vc[i];
            }
            alpha[0] = al;
        } else {
            for (int64_t i = 0; i < n; i++) vn[i] = vn[i] - bp * vp[i];
            al = alpha[0];
        }
        double nr = 0.0;
        for (int64_t i = 0; i < n; i++) {
            double w = vn[i] - al * vc[i];
            vn[i] = w;
            nr += w * w;
        }
        double b;
        if (dots) { b = sqrt(nr); beta[0] = b; } else b = beta[0];
        if (b > 0.0) {
            double inv = 1.0 / b, mx = 0.0;
            for (int64_t i = 0; i < n; i++) {
                double w = vn[i] * inv;
                vn[i] = w;
                double aw = fabs(w);
                mx = aw > mx ? aw : mx;
            }
            if (dots) vmax[0] = mx;
        } else if (dots) vmax[0] = 0.0;
        return;
    }
    /* c == 2 */
    const v2d *x2 = (const v2d *)vc;
    const v2d *p2 = (const v2d *)vp;
    v2d *y2 = (v2d *)vn;
    v2d bp = {bprev[0], bprev[1]};
    v2d al = {0.0, 0.0};
    for (int64_t i = 0; i < n; i++) {
        uint64_t k = ip[i], e = ip[i + 1];
        v2d s0 = {0.0, 0.0}, s1 = {0.0, 0.0};
        for (; k + 2 <= e; k += 2) {
            v2d a0 = {a[k], a[k]}, a1 = {a[k + 1], a[k + 1]};
            s0 += a0 * x2[ix[k]];
            s1 += a1 * x2[ix[k + 1]];
        }
        if (k < e) { v2d a0 = {a[k], a[k]}; s0 += a0 * x2[ix[k]]; }
        v2d w = (s0 + s1) - bp * p2[i];
        y2[i] = w;
        if (dots) al += w * x2[i];
    }
    if (dots) { alpha[0] = al[0]; alpha[1] = al[1]; }
    else { al[0] = alpha[0]; al[1] = alpha[1]; }
    v2d nr = {0.0, 0.0};
    for (int64_t i = 0; i < n; i++) {
        v2d w = y2[i] - al * x2[i];
        y2[i] = w;
        nr += w * w;
    }
    double b0, b1;
    if (dots) { b0 = sqrt(nr[0]); b1 = sqrt(nr[1]); beta[0] = b0; beta[1] = b1; }
    else { b0 = beta[0]; b1 = beta[1]; }
    v2d inv = {b0 > 0.0 ? 1.0 / b0 : 1.0, b1 > 0.0 ? 1.0 / b1 : 1.0};
    double m0 = 0.0, m1 = 0.0;
    for (int64_t i = 0; i < n; i++) {
        v2d w = y2[i] * inv;
        y2[i] = w;
        double a0 = fabs(w[0]), a1 = fabs(w[1]);
        m0 = a0 > m0 ? a0 : m0;
        m1 = a1 > m1 ? a1 : m1;
    }
    if (dots) { vmax[0] = b0 > 0.0 ? m0 : 0.0; vmax[1] = b1 > 0.0 ? m1 : 0.0; }
}

/* Run Lanczos steps k = k0 .. k1-1 with the basis stored contiguously:
 * V has room for (k1+1) vectors of length c*n; V[0] and (k0>0: V[k0-1], V[k0])
 * already hold the starting vectors.  alpha/beta/vmax are arrays of shape
 * [k][c].  samp (ns indices) gathers v_{k+1}[samp] into S[(k+1)][ns][c].
 * Stops early at breakdown (beta below brk*|alpha| scale); returns the
 * number of steps completed. */
int64_t lanczos_run(int64_t n, int c, const uint64_t *ip, const uint32_t *ix,
                    const double *a, double *V, int64_t k0, int64_t k1,
                    double *alpha, double *beta, double *vmax,
                    const int64_t *samp, int64_t ns, double *S)
{
    const int64_t len = (int64_t)c * n;
    double zero[2] = {0.0, 0.0};
    for (int64_t k = k0; k < k1; k++) {
        const double *vp = k > 0 ? V + (k - 1) * len : V; /* beta_{-1}=0 */
        const double *bp = k > 0 ? beta + (k - 1) * c : zero;
        lanczos_step(n, c, ip, ix, a, vp, V + k * len, V + (k + 1) * len,
                     bp, alpha + k * c, beta + k * c, vmax + (k + 1) * c, 1);
        const double *vn = V + (k + 1) * len;
        double *Sk = S + (k + 1) * ns * c;
        for (int64_t s = 0; s < ns; s++)
            for (int j = 0; j < c; j++) Sk[s * c + j] = vn[samp[s] * c + j];
        int brk = 1;
        for (int j = 0; j < c; j++) if (beta[k * c + j] > 0.0) brk = 0;
        if (brk) return k - k0 + 1;
    }
    return k1 - k0;
}

/* Same as lanczos_run but with only three rotating buffers (no stored basis).
 * W holds 3 vectors of length c*n; on entry slot (k0-1)%3 and k0%3 are valid. */
int64_t lanczos_run3(int64_t n, int c, const uint64_t *ip, const uint32_t *ix,
                     const double *a, double *W, int64_t k0, int64_t k1,
                     double *alpha, double *beta, double *vmax,
                     const int64_t *samp, int64_t ns, double *S)
{
    const int64_t len = (int64_t)c * n;
    double zero[2] = {0.0, 0.0};
    for (int64_t k = k0; k < k1; k++) {
        const double *vp = k > 0 ? W + ((k - 1) % 3) * len : W;
        const double *bp = k > 0 ? beta + (k - 1) * c : zero;
        double *vn = W + ((k + 1) % 3) * len;
        lanczos_step(n, c, ip, ix, a, vp, W + (k % 3) * len, vn,
                     bp, alpha + k * c, beta + k * c, vmax + (k + 1) * c, 1);
        double *Sk = S + (k + 1) * ns * c;
        for (int64_t s = 0; s < ns; s++)
            for (int j = 0; j < c; j++) Sk[s * c + j] = vn[samp[s] * c + j];
        int brk = 1;
        for (int j = 0; j < c; j++) if (beta[k * c + j] > 0.0) brk = 0;
        if (brk) return k - k0 + 1;
    }
    return k1 - k0;
}

/* Replay pass: W slots (k0-1)%3 and k0%3 hold v_{k0-1}, v_{k0} (k0 >= 1).
 * Regenerates v_{k0+1}..v_{m-1} with the recorded alpha/beta (bitwise equal
 * to the first pass) and accumulates y += sum_k sum_j coef[k][j] v_k[c*i+j]. */
void lanczos_replay_from(int64_t n, int c, const uint64_t *ip, const uint32_t *ix,
                         const double *a, double *W, int64_t k0, int64_t m,
                         double *alpha, double *beta, const double *coef, double *y)
{
    const int64_t len = (int64_t)c * n;
    double dummy[2];
    for (int64_t k = k0; k + 1 < m; k++) {
        const double *vp = W + ((k - 1) % 3) * len;
        double *vn = W + ((k + 1) % 3) * len;
        lanczos_step(n, c, ip, ix, a, vp, W + (k % 3) * len, vn,
                     beta + (k - 1) * c, alpha + k * c, beta + k * c, dummy, 0);
        const double *cf = coef + (k + 1) * c;
        if (c == 1) {
            double c0 = cf[0];
            for (int64_t i = 0; i < n; i++) y[i] += c0 * vn[i];
        } else {
            double c0 = cf[0], c1 = cf[1];
            for (int64_t i = 0; i < n; i++) y[i] += c0 * vn[2 * i] + c1 * vn[2 * i + 1];
        }
    }
}

/* y[i] = sum_{k<m} sum_j coef[k][j] V[k][c*i+j], blocked over i for cache reuse. */
void combine(int64_t n, int c, const double *V, int64_t m, const double *coef, double *y)
{
    const int64_t len = (int64_t)c * n;
    const int64_t B = 2048;
    for (int64_t i0 = 0; i0 < n; i0 += B) {
        int64_t i1 = i0 + B < n ? i0 + B : n;
        for (int64_t i = i0; i < i1; i++) y[i] = 0.0;
        for (int64_t k = 0; k < m; k++) {
            const double *v = V + k * len;
            if (c == 1) {
                double c0 = coef[k];
                if (c0 == 0.0) continue;
                for (int64_t i = i0; i < i1; i++) y[i] += c0 * v[i];
            } else {
                double c0 = coef[2 * k], c1 = coef[2 * k + 1];
                for (int64_t i = i0; i < i1; i++) y[i] += c0 * v[2 * i] + c1 * v[2 * i + 1];
            }
        }
    }
}
