/* Kernels for y = exp(-tL) b0 + t*phi1(-tL) b1 with L sparse symmetric PSD (CSR).
 * Called from solver.py through ctypes.  All index arrays: row pointers int64,
 * column indices int32.  Single threaded by design (one CPU per session). */
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

typedef int64_t i64;
typedef int32_t i32;

/* out[0] = max_i sum_j |a_ij| (Gershgorin upper bound)
 * out[1] = min_i (a_ii - sum_{j!=i} |a_ij|) (Gershgorin lower bound)
 * out[2] = number of nonzero off-diagonal entries
 * out[3] = max_i |a_ii| */
void gersh(i64 n, const i64 *ip, const i32 *ix, const double *a, double *out)
{
    /* branch-free: row abs-sum s (= |d| + radius), diagonal d */
    double gmax = 0.0, glo = INFINITY, dmax = 0.0, lbest = 0.0;
    i64 ch = 0, prevlen = -1, bw = 0;
    for (i64 i = 0; i < n; i++) {
        double s = 0.0, d = 0.0, q = 0.0;
        i64 b = ip[i], e = ip[i + 1];
        for (i64 k = b; k < e; k++) {
            double v = a[k];
            s += fabs(v);
            q += v * v;
            d += (ix[k] == (i32)i) ? v : 0.0;
        }
        if (e > b) {   /* columns are sorted: the extremes give the row bandwidth */
            i64 d0 = i - ix[b], d1 = ix[e - 1] - i;
            d0 = d0 > d1 ? d0 : d1;
            bw = d0 > bw ? d0 : bw;
        }
        /* PSD: L^2 <= lambda_max L, so lambda_max >= ||L e_i||^2 / L_ii */
        if (d > 0.0 && q / d > lbest) lbest = q / d;
        double ad = fabs(d);
        gmax = s > gmax ? s : gmax;
        double lo = d - (s - ad);
        glo = lo < glo ? lo : glo;
        dmax = ad > dmax ? ad : dmax;
        ch += (e - b) != prevlen;
        prevlen = e - b;
    }
    if (n == 0) glo = 0.0;
    out[0] = gmax; out[1] = glo; out[2] = n > 1 ? (double)(ch - 1) / (double)(n - 1) : 0.0; out[3] = dmax;
    out[4] = lbest;
    out[5] = (double)bw;
}

static inline double rowdot(const i64 *ip, const i32 *ix, const double *a, const double *x, i64 i)
{
    i64 k = ip[i], e = ip[i + 1];
    double s0 = 0.0, s1 = 0.0;
    for (; k + 1 < e; k += 2) {
        s0 += a[k] * x[ix[k]];
        s1 += a[k + 1] * x[ix[k + 1]];
    }
    if (k < e) s0 += a[k] * x[ix[k]];
    return s0 + s1;
}

/* Chebyshev series for one vector:  y = sum_j coef[j] T_j(S) b, S = sa*L + sb*I.
 * tail[j] = sum_{i>=j} |coef[i]|  (length K+2, tail[K+1] = 0).
 * Stops at the first degree j with safety*tail[j+1]*M <= atol + rtol*||y||_inf,
 * M = max observed ||T_i(S)b||_inf.  Returns the degree used, or -(j+1) when
 * ||T_j(S) b||_2 exceeded growlim*||b||_2 (spectral interval too small).
 * info[0] = final tail estimate, info[1] = M, info[2] = ||y||_inf */
i64 cheb1(i64 n, const i64 *ip, const i32 *ix, const double *a,
          double sa, double sb, const double *coef, const double *tail, i64 K,
          const double *b, double *y, double *wa, double *wb,
          double atol, double rtol, double safety, double growlim, double *info)
{
    double bn2 = 0.0, binf = 0.0, ymax = 0.0;
    double c0 = coef[0];
    for (i64 i = 0; i < n; i++) {
        double v = b[i];
        wa[i] = v;
        bn2 += v * v;
        double av = fabs(v);
        if (av > binf) binf = av;
        double yy = c0 * v;
        y[i] = yy;
        if (fabs(yy) > ymax) ymax = fabs(yy);
    }
    info[0] = 0.0; info[1] = binf; info[2] = ymax;
    if (bn2 == 0.0) return 0;
    double M = binf;
    double glim2 = growlim * growlim * bn2;
    if (K < 1 || safety * tail[1] * M <= atol + rtol * ymax) {
        info[0] = tail[1] * M; info[2] = ymax;
        return 0;
    }
    /* step 1: wb = S b */
    {
        double c1 = coef[1], wn2 = 0.0, wmax = 0.0;
        ymax = 0.0;
        for (i64 i = 0; i < n; i++) {
            double s = rowdot(ip, ix, a, wa, i);
            double w = sa * s + sb * wa[i];
            wb[i] = w;
            wn2 += w * w;
            if (fabs(w) > wmax) wmax = fabs(w);
            double yy = y[i] + c1 * w;
            y[i] = yy;
            if (fabs(yy) > ymax) ymax = fabs(yy);
        }
        if (wn2 > glim2) return -2;
        if (wmax > M) M = wmax;
        if (K < 2 || safety * tail[2] * M <= atol + rtol * ymax) {
            info[0] = tail[2] * M; info[1] = M; info[2] = ymax;
            return 1;
        }
    }
    double sa2 = 2.0 * sa, sb2 = 2.0 * sb;
    for (i64 j = 1; j < K; j++) {
        /* wa <- 2 S wb - wa  (= T_{j+1}) */
        double cj = coef[j + 1], wn2 = 0.0, wmax = 0.0;
        ymax = 0.0;
        for (i64 i = 0; i < n; i++) {
            double s = rowdot(ip, ix, a, wb, i);
            double w = sa2 * s + sb2 * wb[i] - wa[i];
            wa[i] = w;
            wn2 += w * w;
            if (fabs(w) > wmax) wmax = fabs(w);
            double yy = y[i] + cj * w;
            y[i] = yy;
            if (fabs(yy) > ymax) ymax = fabs(yy);
        }
        double *tmp = wa; wa = wb; wb = tmp;
        if (!(wn2 <= glim2)) return -(j + 2);
        if (wmax > M) M = wmax;
        if (safety * tail[j + 2] * M <= atol + rtol * ymax) {
            info[0] = tail[j + 2] * M; info[1] = M; info[2] = ymax;
            return j + 1;
        }
    }
    info[0] = tail[K + 1] * M; info[1] = M; info[2] = ymax;
    return K;
}

/* Two vectors with separate coefficient sets, same polynomial basis:
 * y = sum_j c[j] T_j(S) b0 + d[j] T_j(S) b1.  W, Wp: workspaces of 2n. */
i64 cheb2(i64 n, const i64 *ip, const i32 *ix, const double *a,
          double sa, double sb, const double *c, const double *tc,
          const double *d, const double *td, i64 K,
          const double *b0, const double *b1, double *y, double *Wa, double *Wb,
          double atol, double rtol, double safety, double growlim, double *info)
{
    double n0 = 0.0, n1 = 0.0, m0 = 0.0, m1 = 0.0, ymax = 0.0;
    for (i64 i = 0; i < n; i++) {
        double u = b0[i], v = b1[i];
        Wa[2 * i] = u; Wa[2 * i + 1] = v;
        n0 += u * u; n1 += v * v;
        if (fabs(u) > m0) m0 = fabs(u);
        if (fabs(v) > m1) m1 = fabs(v);
        double yy = c[0] * u + d[0] * v;
        y[i] = yy;
        if (fabs(yy) > ymax) ymax = fabs(yy);
    }
    info[0] = 0.0; info[1] = m0; info[2] = ymax;
    double M0 = m0, M1 = m1;
    double g0 = growlim * growlim * n0, g1 = growlim * growlim * n1;
    if (K < 1 || safety * (tc[1] * M0 + td[1] * M1) <= atol + rtol * ymax) {
        info[0] = tc[1] * M0 + td[1] * M1; info[2] = ymax;
        return 0;
    }
    {
        double c1 = c[1], d1 = d[1], q0 = 0.0, q1 = 0.0, x0 = 0.0, x1 = 0.0;
        ymax = 0.0;
        for (i64 i = 0; i < n; i++) {
            double s0 = 0.0, s1 = 0.0;
            for (i64 k = ip[i]; k < ip[i + 1]; k++) {
                double av = a[k];
                const double *xp = Wa + 2 * (i64)ix[k];
                s0 += av * xp[0];
                s1 += av * xp[1];
            }
            double w0 = sa * s0 + sb * Wa[2 * i];
            double w1 = sa * s1 + sb * Wa[2 * i + 1];
            Wb[2 * i] = w0; Wb[2 * i + 1] = w1;
            q0 += w0 * w0; q1 += w1 * w1;
            if (fabs(w0) > x0) x0 = fabs(w0);
            if (fabs(w1) > x1) x1 = fabs(w1);
            double yy = y[i] + c1 * w0 + d1 * w1;
            y[i] = yy;
            if (fabs(yy) > ymax) ymax = fabs(yy);
        }
        if (q0 > g0 || q1 > g1) return -2;
        if (x0 > M0) M0 = x0;
        if (x1 > M1) M1 = x1;
        if (K < 2 || safety * (tc[2] * M0 + td[2] * M1) <= atol + rtol * ymax) {
            info[0] = tc[2] * M0 + td[2] * M1; info[1] = M0; info[2] = ymax;
            return 1;
        }
    }
    double sa2 = 2.0 * sa, sb2 = 2.0 * sb;
    for (i64 j = 1; j < K; j++) {
        double cj = c[j + 1], dj = d[j + 1], q0 = 0.0, q1 = 0.0, x0 = 0.0, x1 = 0.0;
        ymax = 0.0;
        for (i64 i = 0; i < n; i++) {
            double s0 = 0.0, s1 = 0.0;
            for (i64 k = ip[i]; k < ip[i + 1]; k++) {
                double av = a[k];
                const double *xp = Wb + 2 * (i64)ix[k];
                s0 += av * xp[0];
                s1 += av * xp[1];
            }
            double w0 = sa2 * s0 + sb2 * Wb[2 * i] - Wa[2 * i];
            double w1 = sa2 * s1 + sb2 * Wb[2 * i + 1] - Wa[2 * i + 1];
            Wa[2 * i] = w0; Wa[2 * i + 1] = w1;
            q0 += w0 * w0; q1 += w1 * w1;
            if (fabs(w0) > x0) x0 = fabs(w0);
            if (fabs(w1) > x1) x1 = fabs(w1);
            double yy = y[i] + cj * w0 + dj * w1;
            y[i] = yy;
            if (fabs(yy) > ymax) ymax = fabs(yy);
        }
        double *tmp = Wa; Wa = Wb; Wb = tmp;
        if (!(q0 <= g0) || !(q1 <= g1)) return -(j + 2);
        if (x0 > M0) M0 = x0;
        if (x1 > M1) M1 = x1;
        if (safety * (tc[j + 2] * M0 + td[j + 2] * M1) <= atol + rtol * ymax) {
            info[0] = tc[j + 2] * M0 + td[j + 2] * M1; info[1] = M0; info[2] = ymax;
            return j + 1;
        }
    }
    info[0] = tc[K + 1] * M0 + td[K + 1] * M1; info[1] = M0; info[2] = ymax;
    return K;
}

/* Plain Lanczos (no reorthogonalisation) from unit vector v0; m steps.
 * alpha[0..m-1], beta[0..m-1] (beta[j] couples step j and j+1).  Returns the
 * number of steps completed (stops early on breakdown).  work: 3n doubles. */
i64 lanczos_from(i64 n, const i64 *ip, const i32 *ix, const double *a, i64 j0, i64 m,
                 const double *v0, double *alpha, double *beta, double *work)
{
    /* steps j0..m-1; for j0 > 0 the state (vp, v) left in work by a previous call is resumed */
    double *vp = work, *v = work + n, *w = work + 2 * n;
    double bprev = 0.0;
    if (j0 == 0) {
        memcpy(v, v0, (size_t)n * sizeof(double));
        memset(vp, 0, (size_t)n * sizeof(double));
    } else {
        bprev = beta[j0 - 1];
    }
    for (i64 j = j0; j < m; j++) {
        double al = 0.0;
        for (i64 i = 0; i < n; i++) {
            double s = rowdot(ip, ix, a, v, i) - bprev * vp[i];
            w[i] = s;
            al += s * v[i];
        }
        double bn = 0.0;
        for (i64 i = 0; i < n; i++) {
            double s = w[i] - al * v[i];
            w[i] = s;
            bn += s * s;
        }
        bn = sqrt(bn);
        alpha[j] = al; beta[j] = bn;
        if (!(bn > 1e-300)) return j + 1;
        double inv = 1.0 / bn;
        for (i64 i = 0; i < n; i++) {
            vp[i] = v[i];
            v[i] = w[i] * inv;
        }
        bprev = bn;
    }
    return m;
}

/* Connected components of the off-diagonal nonzero pattern (BFS).
 * label[i] = component id, returns number of components; maxsize[0] = largest. */
i64 components(i64 n, const i64 *ip, const i32 *ix, const double *a,
               i64 *label, i64 *queue, i64 *maxsize, i64 limit)
{
    for (i64 i = 0; i < n; i++) label[i] = -1;
    i64 nc = 0, mx = 0;
    for (i64 s = 0; s < n; s++) {
        if (label[s] >= 0) continue;
        i64 head = 0, tailq = 0;
        queue[tailq++] = s; label[s] = nc;
        while (head < tailq) {
            i64 u = queue[head++];
            for (i64 k = ip[u]; k < ip[u + 1]; k++) {
                i64 v = ix[k];
                if (a[k] != 0.0 && label[v] < 0) { label[v] = nc; queue[tailq++] = v; }
            }
            if (tailq > limit) { maxsize[0] = tailq; return -1; }
        }
        if (tailq > mx) mx = tailq;
        nc++;
    }
    maxsize[0] = mx;
    return nc;
}

/* Cyclic Jacobi eigen-decomposition of a dense symmetric s x s matrix A (row major,
 * destroyed).  V (row major) receives eigenvectors as columns, w eigenvalues. */
static void jacobi(int s, double *A, double *V, double *w)
{
    for (int i = 0; i < s; i++)
        for (int j = 0; j < s; j++) V[i * s + j] = (i == j) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0.0, tot = 0.0;
        for (int i = 0; i < s; i++)
            for (int j = 0; j < s; j++) {
                double v = A[i * s + j] * A[i * s + j];
                tot += v;
                if (i != j) off += v;
            }
        if (off <= 1e-34 * tot || off == 0.0) break;
        for (int p = 0; p < s - 1; p++)
            for (int q = p + 1; q < s; q++) {
                double apq = A[p * s + q];
                if (apq == 0.0) continue;
                double app = A[p * s + p], aqq = A[q * s + q];
                double theta = (aqq - app) / (2.0 * apq);
                double tt = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
                double cs = 1.0 / sqrt(tt * tt + 1.0), sn = tt * cs;
                for (int k = 0; k < s; k++) {
                    double akp = A[k * s + p], akq = A[k * s + q];
                    A[k * s + p] = cs * akp - sn * akq;
                    A[k * s + q] = sn * akp + cs * akq;
                }
                for (int k = 0; k < s; k++) {
                    double apk = A[p * s + k], aqk = A[q * s + k];
                    A[p * s + k] = cs * apk - sn * aqk;
                    A[q * s + k] = sn * apk + cs * aqk;
                }
                A[p * s + q] = 0.0; A[q * s + p] = 0.0;
                for (int k = 0; k < s; k++) {
                    double vkp = V[k * s + p], vkq = V[k * s + q];
                    V[k * s + p] = cs * vkp - sn * vkq;
                    V[k * s + q] = sn * vkp + cs * vkq;
                }
            }
    }
    for (int i = 0; i < s; i++) w[i] = A[i * s + i];
}

/* Block eigen-decomposition.  order[]: nodes grouped by component, comp_off[c]..comp_off[c+1]
 * the nodes of component c.  loc[node] = local index inside its component (computed here).
 * Output: lam[n] (eigenvalues, laid out like order), vec: concatenated row-major blocks
 * (offset vec_off[c]).  work: >= 2*smax*smax doubles. */
void block_eig(i64 n, const i64 *ip, const i32 *ix, const double *a, i64 nc,
               const i64 *order, const i64 *comp_off, const i64 *vec_off, i64 *loc,
               double *lam, double *vec, double *work)
{
    /* local indices are only needed by the Jacobi (s > 2) blocks */
    for (i64 c = 0; c < nc; c++)
        if (comp_off[c + 1] - comp_off[c] > 2)
            for (i64 p = comp_off[c]; p < comp_off[c + 1]; p++) loc[order[p]] = p - comp_off[c];
    for (i64 c = 0; c < nc; c++) {
        i64 o = comp_off[c];
        int s = (int)(comp_off[c + 1] - o);
        double *V = vec + vec_off[c];
        if (s == 1) {
            i64 u = order[o];
            double d = 0.0;
            for (i64 k = ip[u]; k < ip[u + 1]; k++) if (ix[k] == u) d += a[k];
            lam[o] = d; V[0] = 1.0;
            continue;
        }
        if (s == 2) {
            /* closed form: rotation diagonalising [[p, q], [q, r]] */
            i64 u0 = order[o], u1 = order[o + 1];
            double p = 0.0, q0 = 0.0, q1 = 0.0, r = 0.0;
            for (i64 k = ip[u0]; k < ip[u0 + 1]; k++) { if (ix[k] == u0) p += a[k]; else if (ix[k] == u1) q0 += a[k]; }
            for (i64 k = ip[u1]; k < ip[u1 + 1]; k++) { if (ix[k] == u1) r += a[k]; else if (ix[k] == u0) q1 += a[k]; }
            double q = 0.5 * (q0 + q1);
            double cs = 1.0, sn = 0.0;
            if (q != 0.0) {
                double theta = (r - p) / (2.0 * q);
                double tt = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
                cs = 1.0 / sqrt(tt * tt + 1.0); sn = tt * cs;
                lam[o] = p - tt * q; lam[o + 1] = r + tt * q;
            } else {
                lam[o] = p; lam[o + 1] = r;
            }
            /* columns: eigenvectors (cs, -sn) for lam[o] and (sn, cs) for lam[o+1] */
            V[0] = cs; V[1] = sn; V[2] = -sn; V[3] = cs;
            continue;
        }
        double *A = work;
        memset(A, 0, sizeof(double) * (size_t)s * s);
        for (int r = 0; r < s; r++) {
            i64 u = order[o + r];
            for (i64 k = ip[u]; k < ip[u + 1]; k++) A[r * s + loc[ix[k]]] += a[k];
        }
        /* symmetrise exactly */
        for (int r = 0; r < s; r++)
            for (int q = r + 1; q < s; q++) {
                double v = 0.5 * (A[r * s + q] + A[q * s + r]);
                A[r * s + q] = v; A[q * s + r] = v;
            }
        jacobi(s, A, V, lam + o);
    }
}

/* y = V diag(fl) V^T b0 + V diag(gl) V^T b1 per block (b, y in original node numbering). */
void block_apply(i64 nc, const i64 *order, const i64 *comp_off, const i64 *vec_off,
                 const double *vec, const double *fl, const double *gl,
                 const double *b0, const double *b1, double *y, double *work)
{
    for (i64 c = 0; c < nc; c++) {
        i64 o = comp_off[c];
        int s = (int)(comp_off[c + 1] - o);
        const double *V = vec + vec_off[c];
        if (s == 1) {
            i64 u = order[o];
            y[u] = fl[o] * b0[u] + gl[o] * b1[u];
            continue;
        }
        double *z = work;
        for (int q = 0; q < s; q++) {
            double u0 = 0.0, u1 = 0.0;
            for (int r = 0; r < s; r++) {
                i64 u = order[o + r];
                u0 += V[r * s + q] * b0[u];
                u1 += V[r * s + q] * b1[u];
            }
            z[q] = fl[o + q] * u0 + gl[o + q] * u1;
        }
        for (int r = 0; r < s; r++) {
            double acc = 0.0;
            for (int q = 0; q < s; q++) acc += V[r * s + q] * z[q];
            y[order[o + r]] = acc;
        }
    }
}

/* Symmetric permutation: new row r = old row perm[r]; iperm = inverse of perm. */
i64 permute_csr(i64 n, const i64 *ip, const i32 *ix, const double *a,
                const i64 *perm, const i64 *iperm, i64 *nip, i32 *nix, double *na)
{
    /* 32-bit inverse permutation (half the cache footprint) + prefetch of upcoming rows */
    i32 *ip32 = (i32 *)malloc(sizeof(i32) * (size_t)(n > 0 ? n : 1));
    for (i64 i = 0; i < n; i++) ip32[i] = (i32)iperm[i];
    i64 pos = 0, bw = 0;
    nip[0] = 0;
    for (i64 r = 0; r < n; r++) {
        if (r + 8 < n) {
            i64 q = perm[r + 8];
            __builtin_prefetch(ix + ip[q]);
            __builtin_prefetch(a + ip[q]);
        }
        i64 o = perm[r];
        for (i64 k = ip[o]; k < ip[o + 1]; k++) {
            i64 c = ip32[ix[k]];
            nix[pos] = (i32)c;
            na[pos] = a[k];
            pos++;
            i64 dd = c > r ? c - r : r - c;
            bw = dd > bw ? dd : bw;
        }
        nip[r + 1] = pos;
    }
    free(ip32);
    return bw;
}

void gather(i64 n, const i64 *perm, const double *src, double *dst)
{
    for (i64 i = 0; i < n; i++) dst[i] = src[perm[i]];
}

void scatter(i64 n, const i64 *perm, const double *src, double *dst)
{
    for (i64 i = 0; i < n; i++) dst[perm[i]] = src[i];
}

/* ------------------------------------------------------------------ reordering */
/* Cuthill-McKee style BFS ordering (neighbours in index order), each component started
 * from a minimum-degree vertex.  perm[new] = old.  work: n i64. */
void bfs_order(i64 n, const i64 *ip, const i32 *ix, i64 *perm, i64 *markbuf)
{
    if (!ip) {   /* identity order (caller only wants the window sort) */
        for (i64 i = 0; i < n; i++) perm[i] = i;
        return;
    }
    unsigned char *mark = (unsigned char *)markbuf;
    memset(mark, 0, (size_t)n);
    i64 head = 0, tl = 0;
    /* process components in order of a min-degree start (scan once for global min) */
    i64 scan = 0;
    while (tl < n) {
        /* start: minimum-degree vertex among the next 4096 indices (bounded scan, O(n) total) */
        while (scan < n && mark[scan]) scan++;
        i64 best = scan, bdeg = ip[scan + 1] - ip[scan];
        i64 qend = scan + 4096 < n ? scan + 4096 : n;
        for (i64 q = scan + 1; q < qend; q++) {
            if (mark[q]) continue;
            i64 dg = ip[q + 1] - ip[q];
            if (dg < bdeg) { bdeg = dg; best = q; }
        }
        perm[tl++] = best; mark[best] = 1;
        while (head < tl) {
            i64 u = perm[head++];
            for (i64 k = ip[u]; k < ip[u + 1]; k++) {
                i64 v = ix[k];
                if (!mark[v]) { mark[v] = 1; perm[tl++] = v; }
            }
        }
    }
}

/* max |i - j| over stored entries, and count of entries with |i-j| > win */
i64 bandwidth(i64 n, const i64 *ip, const i32 *ix, i64 win, i64 *far)
{
    i64 bw = 0, f = 0;
    for (i64 i = 0; i < n; i++)
        for (i64 k = ip[i]; k < ip[i + 1]; k++) {
            i64 d = ix[k] - i;
            if (d < 0) d = -d;
            if (d > bw) bw = d;
            f += (d > win);
        }
    far[0] = f;
    return bw;
}

static inline void rowdot2(const i64 *ip, const i32 *ix, const double *a, const double *X, i64 i,
                           double *o0, double *o1)
{
    double s0 = 0.0, s1 = 0.0;
    for (i64 k = ip[i]; k < ip[i + 1]; k++) {
        double av = a[k];
        const double *xp = X + 2 * (i64)ix[k];
        s0 += av * xp[0];
        s1 += av * xp[1];
    }
    *o0 = s0; *o1 = s1;
}


/* ------------------------------------------------------------------ two steps per sweep */
/* general CSR values */
#define SUFFIX _csr
#define MATARGS const i64 *ip, const i32 *ix, const double *a
#define DOT1(X, i) rowdot(ip, ix, a, X, i)
#define DOT2(X, i, s0, s1) rowdot2(ip, ix, a, X, i, &(s0), &(s1))
#include "chebtmpl.h"
#undef SUFFIX
#undef MATARGS
#undef DOT1
#undef DOT2

/* pattern-only: L = diag(dg) + cval * pattern (off-diagonal pattern without values) */
static inline double patdot(const i64 *ip, const i32 *ix, const double *dg, double cval,
                            const double *x, i64 i)
{
    i64 k = ip[i], e = ip[i + 1];
    double s0 = 0.0, s1 = 0.0;
    for (; k + 1 < e; k += 2) {
        s0 += x[ix[k]];
        s1 += x[ix[k + 1]];
    }
    if (k < e) s0 += x[ix[k]];
    return dg[i] * x[i] + cval * (s0 + s1);
}

static inline void patdot2(const i64 *ip, const i32 *ix, const double *dg, double cval,
                           const double *X, i64 i, double *o0, double *o1)
{
    double s0 = 0.0, s1 = 0.0;
    for (i64 k = ip[i]; k < ip[i + 1]; k++) {
        const double *xp = X + 2 * (i64)ix[k];
        s0 += xp[0];
        s1 += xp[1];
    }
    *o0 = dg[i] * X[2 * i] + cval * s0;
    *o1 = dg[i] * X[2 * i + 1] + cval * s1;
}

#define SUFFIX _pat
#define MATARGS const i64 *ip, const i32 *ix, const double *dg, double cval
#define DOT1(X, i) patdot(ip, ix, dg, cval, X, i)
#define DOT2(X, i, s0, s1) patdot2(ip, ix, dg, cval, X, i, &(s0), &(s1))
#include "chebtmpl.h"
#undef SUFFIX
#undef MATARGS
#undef DOT1
#undef DOT2

/* If every stored off-diagonal entry equals one constant, split L into diag + pattern.
 * Returns 1 and fills (pip, pix, dg, *cval); 0 otherwise. */
int split_pattern(i64 n, const i64 *ip, const i32 *ix, const double *a,
                  i64 *pip, i32 *pix, double *dg, double *cval)
{
    int have = 0;
    double cv = 0.0;
    for (i64 i = 0; i < n; i++)
        for (i64 k = ip[i]; k < ip[i + 1]; k++)
            if (ix[k] != i) {
                if (!have) { cv = a[k]; have = 1; }
                else if (a[k] != cv) return 0;
            }
    if (!have || cv == 0.0) return 0;
    i64 pos = 0;
    pip[0] = 0;
    for (i64 i = 0; i < n; i++) {
        double d = 0.0;
        for (i64 k = ip[i]; k < ip[i + 1]; k++) {
            if (ix[k] == i) d += a[k];
            else pix[pos++] = ix[k];
        }
        dg[i] = d;
        pip[i + 1] = pos;
    }
    *cval = cv;
    return 1;
}

/* ------------------------------------------------------------------ 16-bit column offsets */
/* column = row + (int)ox[k]; valid when bandwidth < 32768 */
static inline double rowdot16(const i64 *ip, const int16_t *ox, const double *a, const double *x, i64 i)
{
    i64 k = ip[i], e = ip[i + 1];
    const double *xi = x + i;
    double s0 = 0.0, s1 = 0.0;
    for (; k + 1 < e; k += 2) {
        s0 += a[k] * xi[ox[k]];
        s1 += a[k + 1] * xi[ox[k + 1]];
    }
    if (k < e) s0 += a[k] * xi[ox[k]];
    return s0 + s1;
}
static inline void rowdot16_2(const i64 *ip, const int16_t *ox, const double *a, const double *X, i64 i,
                              double *o0, double *o1)
{
    double s0 = 0.0, s1 = 0.0;
    const double *xi = X + 2 * i;
    for (i64 k = ip[i]; k < ip[i + 1]; k++) {
        double av = a[k];
        const double *xp = xi + 2 * (i64)ox[k];
        s0 += av * xp[0];
        s1 += av * xp[1];
    }
    *o0 = s0; *o1 = s1;
}
#define SUFFIX _c16
#define MATARGS const i64 *ip, const int16_t *ox, const double *a
#define DOT1(X, i) rowdot16(ip, ox, a, X, i)
#define DOT2(X, i, s0, s1) rowdot16_2(ip, ox, a, X, i, &(s0), &(s1))
#include "chebtmpl.h"
#undef SUFFIX
#undef MATARGS
#undef DOT1
#undef DOT2

static inline double patdot16(const i64 *ip, const int16_t *ox, const double *dg, double cval,
                              const double *x, i64 i)
{
    i64 k = ip[i], e = ip[i + 1];
    const double *xi = x + i;
    double s0 = 0.0, s1 = 0.0;
    for (; k + 1 < e; k += 2) {
        s0 += xi[ox[k]];
        s1 += xi[ox[k + 1]];
    }
    if (k < e) s0 += xi[ox[k]];
    return dg[i] * x[i] + cval * (s0 + s1);
}
static inline void patdot16_2(const i64 *ip, const int16_t *ox, const double *dg, double cval,
                              const double *X, i64 i, double *o0, double *o1)
{
    double s0 = 0.0, s1 = 0.0;
    const double *xi = X + 2 * i;
    for (i64 k = ip[i]; k < ip[i + 1]; k++) {
        const double *xp = xi + 2 * (i64)ox[k];
        s0 += xp[0];
        s1 += xp[1];
    }
    *o0 = dg[i] * X[2 * i] + cval * s0;
    *o1 = dg[i] * X[2 * i + 1] + cval * s1;
}
#define SUFFIX _p16
#define MATARGS const i64 *ip, const int16_t *ox, const double *dg, double cval
#define DOT1(X, i) patdot16(ip, ox, dg, cval, X, i)
#define DOT2(X, i, s0, s1) patdot16_2(ip, ox, dg, cval, X, i, &(s0), &(s1))
#include "chebtmpl.h"
#undef SUFFIX
#undef MATARGS
#undef DOT1
#undef DOT2

/* relative 16-bit offsets ox[k] = ix[k] - row; caller guarantees bandwidth < 32768 */
void to_offsets16(i64 n, const i64 *ip, const i32 *ix, int16_t *ox)
{
    for (i64 i = 0; i < n; i++)
        for (i64 k = ip[i]; k < ip[i + 1]; k++) ox[k] = (int16_t)((i64)ix[k] - i);
}

/* ------------------------------------------------------------------ SELL-4 (AVX2) */
#ifdef __AVX2__
#include <immintrin.h>
static inline __m256d gath4(const double *x, const i32 *ci)
{
    __m128d lo = _mm_loadh_pd(_mm_load_sd(x + ci[0]), x + ci[1]);
    __m128d hi = _mm_loadh_pd(_mm_load_sd(x + ci[2]), x + ci[3]);
    return _mm256_insertf128_pd(_mm256_castpd128_pd256(lo), hi, 1);
}
#define SPAT 0
#define SNV 1
#include "selltmpl.h"
#undef SNV
#define SNV 2
#include "selltmpl.h"
#undef SNV
#undef SPAT
#define SPAT 1
#define SNV 1
#include "selltmpl.h"
#undef SNV
#define SNV 2
#include "selltmpl.h"
#undef SNV
#undef SPAT
#define SPAT 0
#include "sell2i.h"
#undef SPAT
#define SPAT 1
#include "sell2i.h"
#undef SPAT
#define HAVE_SELL 1
#endif

/* SELL-4 layout of a CSR matrix (a == NULL: pattern only).  Sizing pass when col == NULL:
 * returns the number of 4-entry groups.  Padding entries point at the zero slot 4*nch. */
i64 sell_build(i64 n, const i64 *ip, const i32 *ix, const double *a,
               i64 *cp, i32 *cl, i32 *col, double *val)
{
    i64 nch = (n + 3) / 4, tot = 0;
    const i32 zslot = (i32)(4 * nch);
    for (i64 c = 0; c < nch; c++) {
        i64 st[4], len[4], L = 0;
        for (int r = 0; r < 4; r++) {
            i64 i = 4 * c + r;
            st[r] = i < n ? ip[i] : 0;
            len[r] = i < n ? ip[i + 1] - ip[i] : 0;
            if (len[r] > L) L = len[r];
        }
        if (col) {
            cp[c] = tot;
            cl[c] = (i32)L;
            i32 *cq = col + 4 * tot;
            for (i64 k = 0; k < L; k++)
                for (int r = 0; r < 4; r++)
                    cq[4 * k + r] = k < len[r] ? ix[st[r] + k] : zslot;
            if (val) {
                double *vq = val + 4 * tot;
                for (i64 k = 0; k < L; k++)
                    for (int r = 0; r < 4; r++)
                        vq[4 * k + r] = k < len[r] ? a[st[r] + k] : 0.0;
            }
        }
        tot += L;
    }
    return tot;
}

/* Collatz-Wielandt: for any x > 0, lambda_max(L) <= rho(|L|) <= max_i (|L| x)_i / x_i.
 * Power steps on |L| from x = 1; returns the smallest bound seen (rigorous up to rounding).
 * work: 2n doubles. */
double cw_bound(i64 n, const i64 *ip, const i32 *ix, const double *a, int iters, double *work)
{
    double *x = work, *y = work + n;
    for (i64 i = 0; i < n; i++) x[i] = 1.0;
    double best = INFINITY;
    for (int it = 0; it < iters; it++) {
        double mx = 0.0, ratio = 0.0;
        for (i64 i = 0; i < n; i++) {
            double s = 0.0;
            for (i64 k = ip[i]; k < ip[i + 1]; k++) s += fabs(a[k]) * x[ix[k]];
            y[i] = s;
            double r = s / x[i];
            ratio = r > ratio ? r : ratio;
            mx = s > mx ? s : mx;
        }
        if (ratio < best) best = ratio;
        if (!(mx > 0.0)) break;
        double inv = 1.0 / mx;
        for (i64 i = 0; i < n; i++) x[i] = y[i] * inv + 1e-200;
    }
    return best;
}

i64 lanczos(i64 n, const i64 *ip, const i32 *ix, const double *a, i64 m,
            const double *v0, double *alpha, double *beta, double *work)
{
    return lanczos_from(n, ip, ix, a, 0, m, v0, alpha, beta, work);
}

/* ------------------------------------------------------------------ pairs (blocks of size <= 2) */
/* 1 if every row has at most one nonzero off-diagonal entry and partners are mutual;
 * partner[i] = j (or -1).  Cheap: exits at the first row with two off-diagonal nonzeros. */
int pair_structure(i64 n, const i64 *ip, const i32 *ix, const double *a, i32 *partner)
{
    for (i64 i = 0; i < n; i++) {
        i32 p = -1;
        for (i64 k = ip[i]; k < ip[i + 1]; k++) {
            if (ix[k] != (i32)i && a[k] != 0.0) {
                if (p >= 0) return 0;
                p = ix[k];
            }
        }
        partner[i] = p;
    }
    for (i64 i = 0; i < n; i++)
        if (partner[i] >= 0 && partner[partner[i]] != (i32)i) return 0;
    return 1;
}

static inline double pair_g(double x, double t, double e)
{
    double tx = t * x;
    if (x == 0.0) return t;
    return tx > 0.25 ? (1.0 - e) / x : -expm1(-tx) / x;
}

/* Per-row coefficients: y_i = F[4i] b0_i + F[4i+1] b0_p + F[4i+2] b1_i + F[4i+3] b1_p. */
void pair_setup(i64 n, const i64 *ip, const i32 *ix, const double *a, const i32 *partner,
                double t, double *F)
{
    for (i64 i = 0; i < n; i++) {
        i32 j = partner[i];
        if (j >= 0 && j < (i32)i) continue;   /* handled with its partner */
        double p = 0.0, r = 0.0, q0 = 0.0, q1 = 0.0;
        for (i64 k = ip[i]; k < ip[i + 1]; k++) {
            if (ix[k] == (i32)i) p += a[k];
            else if (ix[k] == j) q0 += a[k];
        }
        if (j < 0) {
            double e = exp(-t * p);
            F[4 * i] = e; F[4 * i + 1] = 0.0;
            F[4 * i + 2] = pair_g(p, t, e); F[4 * i + 3] = 0.0;
            continue;
        }
        for (i64 k = ip[j]; k < ip[j + 1]; k++) {
            if (ix[k] == j) r += a[k];
            else if (ix[k] == (i32)i) q1 += a[k];
        }
        double q = 0.5 * (q0 + q1), cs = 1.0, sn = 0.0, l1 = p, l2 = r;
        if (q != 0.0) {
            double theta = (r - p) / (2.0 * q);
            double tt = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
            cs = 1.0 / sqrt(tt * tt + 1.0); sn = tt * cs;
            l1 = p - tt * q; l2 = r + tt * q;
        }
        /* eigenvectors (cs, -sn) for l1 and (sn, cs) for l2 */
        double e1 = exp(-t * l1), e2 = exp(-t * l2);
        double g1 = pair_g(l1, t, e1), g2 = pair_g(l2, t, e2);
        double c2 = cs * cs, s2 = sn * sn, csn = cs * sn;
        F[4 * i] = c2 * e1 + s2 * e2;      F[4 * i + 1] = csn * (e2 - e1);
        F[4 * i + 2] = c2 * g1 + s2 * g2;  F[4 * i + 3] = csn * (g2 - g1);
        F[4 * (i64)j] = s2 * e1 + c2 * e2; F[4 * (i64)j + 1] = csn * (e2 - e1);
        F[4 * (i64)j + 2] = s2 * g1 + c2 * g2; F[4 * (i64)j + 3] = csn * (g2 - g1);
    }
}

void pair_apply(i64 n, const i32 *partner, const double *F, const double *b0, const double *b1, double *y)
{
    for (i64 i = 0; i < n; i++) {
        i32 j = partner[i];
        double v = F[4 * i] * b0[i] + F[4 * i + 2] * b1[i];
        if (j >= 0) v += F[4 * i + 1] * b0[j] + F[4 * i + 3] * b1[j];
        y[i] = v;
    }
}

/* gersh() plus u0 = L b0, u1 = L b1 in the same pass (b0/b1 may be NULL). */
void gersh_spmv(i64 n, const i64 *ip, const i32 *ix, const double *a, double *out,
                const double *b0, const double *b1, double *u0, double *u1)
{
    double gmax = 0.0, glo = INFINITY, dmax = 0.0, lbest = 0.0;
    i64 ch = 0, prevlen = -1, bw = 0;
    const int h0 = b0 != NULL, h1 = b1 != NULL;
    for (i64 i = 0; i < n; i++) {
        double s = 0.0, d = 0.0, q = 0.0, p0 = 0.0, p1 = 0.0;
        i64 b = ip[i], e = ip[i + 1];
        for (i64 k = b; k < e; k++) {
            double v = a[k];
            i32 c = ix[k];
            s += fabs(v);
            q += v * v;
            d += (c == (i32)i) ? v : 0.0;
            if (h0) p0 += v * b0[c];
            if (h1) p1 += v * b1[c];
        }
        if (h0) u0[i] = p0;
        if (h1) u1[i] = p1;
        if (d > 0.0 && q / d > lbest) lbest = q / d;
        if (e > b) {
            i64 d0 = i - ix[b], d1 = ix[e - 1] - i;
            d0 = d0 > d1 ? d0 : d1;
            bw = d0 > bw ? d0 : bw;
        }
        double ad = fabs(d);
        gmax = s > gmax ? s : gmax;
        double lo = d - (s - ad);
        glo = lo < glo ? lo : glo;
        dmax = ad > dmax ? ad : dmax;
        ch += (e - b) != prevlen;
        prevlen = e - b;
    }
    if (n == 0) glo = 0.0;
    out[0] = gmax; out[1] = glo; out[2] = n > 1 ? (double)(ch - 1) / (double)(n - 1) : 0.0; out[3] = dmax;
    out[4] = lbest;
    out[5] = (double)bw;
}
