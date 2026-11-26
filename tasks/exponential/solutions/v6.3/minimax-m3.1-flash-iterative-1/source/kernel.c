/* Chebyshev kernel for FAMEXP2: y = exp(-tL) b0 + t*phi1(-tL) b1.
 *
 * The matrix is a symmetric PSD sparse matrix given in CSR form with
 * uint64 row pointers, uint32 column indices and double values.  No
 * conversion or copying of the matrix is performed: the arrays are used
 * in place.  The two right hand sides are carried together in one
 * interleaved array so that a single pass over the matrix serves both.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef double dbl;

#define EXPORT __attribute__((visibility("default")))

/* ------------------------------------------------------------------ */
/* one pass over the matrix, two right hand sides at once              */
/* z holds interleaved pairs (component 0, component 1)                  */
static void spmv_pair(const uint64_t *ip, const uint32_t *ix, const dbl *a,
                      int64_t n, const dbl *z, dbl *out)
{
    for (int64_t i = 0; i < n; i++) {
        dbl s0 = 0.0, s1 = 0.0;
        uint64_t p = ip[i], q = ip[i + 1];
        for (; p < q; p++) {
            dbl av = a[p];
            uint32_t j = ix[p];
            s0 += av * z[2 * (size_t)j];
            s1 += av * z[2 * (size_t)j + 1];
        }
        out[2 * i] = s0;
        out[2 * i + 1] = s1;
    }
}

/* ------------------------------------------------------------------ */
/* Rigorous upper bound for lambda_max(L).                              */
/* Gershgorin on L plus the Collatz-Wielandt iteration on |L|: for a   */
/* non-negative matrix B and a strictly positive vector x,            */
/*     max_i (Bx)_i / x_i  >=  rho(B),                               */
/* and rho(|L|) >= lambda_max(L) because x^T |L| x >= |x^T L x|.      */
EXPORT double famexp_spectral_bound(const uint64_t *ip, const uint32_t *ix,
                                    const dbl *a, int64_t n, int maxiters,
                                    dbl *out_lo)
{
    dbl gersh = 0.0;
    dbl gersh_lo = INFINITY;
    for (int64_t i = 0; i < n; i++) {
        dbl rs = 0.0, diag = 0.0;
        uint64_t p = ip[i], q = ip[i + 1];
        for (; p < q; p++) {
            dbl av = a[p];
            rs += fabs(av);
            if (ix[p] == (uint32_t)i) diag = av;
        }
        dbl v = diag + rs - fabs(diag);
        if (v > gersh) gersh = v;
        dbl w = diag - rs + fabs(diag);
        if (w < gersh_lo) gersh_lo = w;
    }
    if (out_lo) *out_lo = (gersh_lo == INFINITY) ? 0.0 : gersh_lo;
    double best = gersh;
    if (n <= 0 || maxiters <= 0) return best;


    dbl *x = (dbl *)malloc(sizeof(dbl) * (size_t)n);
    dbl *y = (dbl *)malloc(sizeof(dbl) * (size_t)n);
    if (!x || !y) { free(x); free(y); return best; }
    for (int64_t i = 0; i < n; i++) {
        dbl u = sin((double)i * 12.9898 + 78.233) * 43758.5453;
        u = u - floor(u);
        x[i] = 1.0 + u;
    }
    double cw_best = INFINITY;
    double prev_cw = -1.0;
    for (int it = 0; it < maxiters; it++) {
        dbl yn = 0.0;
        for (int64_t i = 0; i < n; i++) {
            dbl s = 0.0;
            uint64_t p = ip[i], q = ip[i + 1];
            for (; p < q; p++) s += fabs(a[p]) * x[ix[p]];
            y[i] = s;
            if (s > yn) yn = s;
        }
        if (!(yn > 0.0)) break;
        dbl cw = 0.0;
        for (int64_t i = 0; i < n; i++) {
            if (x[i] <= 0.0) { cw = INFINITY; break; }
            dbl r = y[i] / x[i];
            if (r > cw) cw = r;
        }
        if (cw > 0.0 && isfinite(cw) && cw < cw_best) cw_best = cw;
        dbl floorv = yn * 1e-250;
        for (int64_t i = 0; i < n; i++) {
            dbl v = y[i];
            x[i] = (v > floorv ? v : floorv) / yn;
        }
        /* the Collatz-Wielandt iteration converges geometrically; two percent
           of improvement is enough for the degree of the Chebyshev series */
        if (prev_cw > 0.0 && (prev_cw - cw_best) < 0.02 * cw_best) break;
        prev_cw = cw_best;
    }
    if (cw_best < best) best = cw_best;
    free(x); free(y);
    return best;
}

/* ------------------------------------------------------------------ */
/* Chebyshev recurrence.  Returns 0 on success, 1 if the coefficient  */
/* array was exhausted before the accuracy criterion was met.           */
EXPORT int famexp_cheb(const uint64_t *ip, const uint32_t *ix, const dbl *a,
                       int64_t n, dbl lo, dbl hi,
                       const dbl *ac, int64_t na, const dbl *dc, int64_t nd,
                       const dbl *b0, const dbl *b1, dbl *y,
                       dbl atol, dbl rtol, int64_t *out_deg, dbl *dbg)
{
    int64_t nc = na < nd ? na : nd;
    if (nc < 2) return 1;

    dbl *tailA = (dbl *)malloc(sizeof(dbl) * (size_t)(nc + 1));
    dbl *tailD = (dbl *)malloc(sizeof(dbl) * (size_t)(nc + 1));
    dbl *z0 = (dbl *)malloc(sizeof(dbl) * (size_t)(2 * n));
    dbl *z1 = (dbl *)malloc(sizeof(dbl) * (size_t)(2 * n));
    dbl *z2 = (dbl *)malloc(sizeof(dbl) * (size_t)(2 * n));
    if (!tailA || !tailD || !z0 || !z1 || !z2) {
        free(tailA); free(tailD); free(z0); free(z1); free(z2);
        return 2;
    }
    tailA[nc] = 0.0; tailD[nc] = 0.0;
    for (int64_t k = nc - 1; k >= 0; k--) {
        tailA[k] = tailA[k + 1] + fabs(ac[k]);
        tailD[k] = tailD[k + 1] + fabs(dc[k]);
    }
    /* Euclidean norms of the right hand sides.  Since the spectrum of the
       scaled operator lies in [-1,1], every T_k has spectral norm at most
       one, hence ||sum_k c_k T_k(Ltilde) b||_2 <= (sum_k |c_k|) ||b||_2 and
       the infinity norm of the error is bounded by the same quantity. */
    dbl nb0 = 0.0, nb1 = 0.0;
    for (int64_t i = 0; i < n; i++) {
        nb0 += b0[i] * b0[i];
        nb1 += b1[i] * b1[i];
    }
    nb0 = sqrt(nb0);
    nb1 = sqrt(nb1);
    dbl c = 0.5 * (lo + hi), h = 0.5 * (hi - lo), s = 2.0 / h;
    for (int64_t i = 0; i < n; i++) {
        z0[2 * i] = b0[i]; z0[2 * i + 1] = b1[i];
    }
    spmv_pair(ip, ix, a, n, z0, z2);
    for (int64_t i = 0; i < n; i++) {
        z1[2 * i] = 0.5 * s * (z2[2 * i] - c * z0[2 * i]);
        z1[2 * i + 1] = 0.5 * s * (z2[2 * i + 1] - c * z0[2 * i + 1]);
    }
    dbl ymax = 0.0;
    for (int64_t i = 0; i < n; i++) {
        dbl v = ac[0] * z0[2 * i] + ac[1] * z1[2 * i]
              + dc[0] * z0[2 * i + 1] + dc[1] * z1[2 * i + 1];
        y[i] = v;
        dbl av = fabs(v); if (av > ymax) ymax = av;
    }
    int64_t deg = 1;
    dbl prev_rem = tailA[1] * nb0 + tailD[1] * nb1;
    int stall = 0;
    /* accuracy test after the first degree */
    for (int64_t j = 1; j < nc; j++) {
        dbl rem = tailA[j + 1] * nb0 + tailD[j + 1] * nb1;
        if (dbg) { dbg[0] = (dbl)j; dbg[1] = rem; dbg[2] = ymax; }
        if (rem * (1.0 + rtol) <= atol + rtol * ymax) { deg = j; goto done; }
        /* the tail of the series cannot fall below the rounding level of the
           coefficients; when it stops shrinking, further degrees are useless */
        if (prev_rem > 0.0 && (prev_rem - rem) <= 1e-4 * rem) {
            if (++stall >= 3) { deg = j; if (dbg) dbg[3] = 1.0; goto done; }
        } else {
            stall = 0;
        }
        prev_rem = rem;
        if (j + 1 >= nc) break;
        spmv_pair(ip, ix, a, n, z1, z2);
        dbl ymax_now = 0.0;
        for (int64_t i = 0; i < n; i++) {
            dbl u2 = s * (z2[2 * i] - c * z1[2 * i]) - z0[2 * i];
            dbl v2 = s * (z2[2 * i + 1] - c * z1[2 * i + 1]) - z0[2 * i + 1];
            z2[2 * i] = u2; z2[2 * i + 1] = v2;
            dbl val = y[i] + ac[j + 1] * u2 + dc[j + 1] * v2;
            y[i] = val;
            dbl av = fabs(val); if (av > ymax_now) ymax_now = av;
        }
        /* the norm of the current partial sum; the partial sums of a
           Chebyshev series oscillate, so the norm of an earlier partial sum
           would overestimate the norm of the answer by a large factor */
        ymax = ymax_now;
        dbl *tmp = z0; z0 = z1; z1 = z2; z2 = tmp;
        deg = j + 1;
    }
    /* coefficient array exhausted */
    *out_deg = deg;
    free(tailA); free(tailD); free(z0); free(z1); free(z2);
    return 1;
done:
    *out_deg = deg;
    free(tailA); free(tailD); free(z0); free(z1); free(z2);
    return 0;
}

/* ------------------------------------------------------------------ */
/* connected components of the sparsity pattern (union-find)            */
EXPORT int famexp_components(const uint64_t *ip, const uint32_t *ix, int64_t n,
                             int64_t *label, int64_t *out_maxsize,
                             int64_t *out_ncomp)
{
    int64_t *parent = (int64_t *)malloc(sizeof(int64_t) * (size_t)n);
    if (!parent) return -1;
    for (int64_t i = 0; i < n; i++) parent[i] = i;
    int64_t *size = (int64_t *)malloc(sizeof(int64_t) * (size_t)n);
    if (!size) { free(parent); return -1; }
    for (int64_t i = 0; i < n; i++) size[i] = 1;
    for (int64_t i = 0; i < n; i++) {
        uint64_t p = ip[i], q = ip[i + 1];
        for (; p < q; p++) {
            /* the matrix is symmetric, so only one triangle has to be walked */
            if (ix[p] <= (uint32_t)i) continue;
            int64_t a = i, b = ix[p];
            while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
            while (parent[b] != b) { parent[b] = parent[parent[b]]; b = parent[b]; }
            if (a == b) continue;
            if (size[a] < size[b]) { int64_t tmp = a; a = b; b = tmp; }
            parent[b] = a; size[a] += size[b];
        }
    }
    int64_t ncomp = 0, maxsize = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t r = i;
        while (parent[r] != r) { parent[r] = parent[parent[r]]; r = parent[r]; }
        label[i] = r;
    }
    for (int64_t i = 0; i < n; i++) size[i] = 0;
    for (int64_t i = 0; i < n; i++) size[label[i]]++;
    for (int64_t i = 0; i < n; i++) {
        if (size[i] > 0) { ncomp++; if (size[i] > maxsize) maxsize = size[i]; }
    }
    *out_maxsize = maxsize; *out_ncomp = ncomp;
    free(parent); free(size);
    return 0;
}
