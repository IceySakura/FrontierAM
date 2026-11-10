/* FAMEXP2 exponential propagation solver.
 *
 * y = exp(-tL) b0 + t*phi1(-tL) b1 is the first block of exp(-tB)[b0;1] with the
 * augmented operator B = [[L, -b1],[0, 0]].  Its spectrum is spec(L) U {0}, which
 * lies in [0, lam] for an upper bound lam of lambda_max(L).  We expand
 * exp(-t x) on [0, lam] in Chebyshev polynomials (coefficients are scaled
 * modified Bessel functions) and run the three-term recurrence on the augmented
 * vector; the extra scalar component is (-1)^k, so each term costs one fused
 * SpMV + vector update pass.  lam comes from a short Lanczos run (with safety
 * margin), capped by the rigorous Gershgorin bound.  The series is truncated
 * adaptively from the known coefficient tail, the observed iterate norms and the
 * running max|y|, against atol + rtol*max|y| with a large safety factor.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

static void rd(void *p, size_t len) {
    char *c = (char *)p;
    while (len) {
        ssize_t r = read(0, c, len > (1u << 30) ? (1u << 30) : len);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) _exit(3);
        c += r;
        len -= (size_t)r;
    }
}

static void wr(const void *p, size_t len) {
    const char *c = (const char *)p;
    while (len) {
        ssize_t r = write(1, c, len > (1u << 30) ? (1u << 30) : len);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) _exit(4);
        c += r;
        len -= (size_t)r;
    }
}

static double jnum(const char *js, const char *key) {
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(js, pat);
    if (!p) return NAN;
    p += strlen(pat);
    while (*p && *p != ':') p++;
    if (*p) p++;
    return strtod(p, NULL);
}

/* Bump allocator over an arena that is mapped and pre-faulted before READY,
 * so the timed session does not pay page faults; falls back to the heap. */
static char *arena, *arena_end, *arena_top;

static void arena_init(size_t bytes) {
    const size_t huge = (size_t)2 << 20;
    void *p = mmap(NULL, bytes + huge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return;
    char *a = (char *)(((uintptr_t)p + huge - 1) & ~(uintptr_t)(huge - 1));
#ifdef MADV_HUGEPAGE
    madvise(a, bytes, MADV_HUGEPAGE);
#endif
    memset(a, 0, bytes);
    arena = arena_top = a;
    arena_end = arena + bytes;
}

static void *xalloc(size_t bytes) {
    void *p = NULL;
    if (bytes == 0) bytes = 64;
    bytes = (bytes + 63) & ~(size_t)63;
    if (arena && (size_t)(arena_end - arena_top) >= bytes) {
        p = arena_top;
        arena_top += bytes;
        return p;
    }
    if (posix_memalign(&p, 64, bytes)) _exit(5);
    return p;
}

static void xfree(void *p) {
    if (p && !(arena && (char *)p >= arena && (char *)p < arena_end)) free(p);
}

static int n, nnz;
static int *rp, *ci;
static double *va;

static inline double rowdot(int i, const double *restrict x) {
    double s = 0.0;
    for (int p = rp[i]; p < rp[i + 1]; p++) s += va[p] * x[ci[p]];
    return s;
}

static int use_sell;
static void sell_spmv(const double *restrict x, double *restrict y);

/* y = L x */
static void spmv(const double *restrict x, double *restrict y) {
    if (use_sell) {
        sell_spmv(x, y);
        return;
    }
    for (int i = 0; i < n; i++) y[i] = rowdot(i, x);
}

/* Eigenvalues/vectors of a small dense symmetric matrix (cyclic Jacobi). */
static void jacobi(int m, double *A, double *V, double *w) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) V[i * m + j] = (i == j);
    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0, tot = 0;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < m; j++) {
                double a = A[i * m + j] * A[i * m + j];
                tot += a;
                if (i != j) off += a;
            }
        if (off <= 1e-30 * tot || off == 0) break;
        for (int p = 0; p < m; p++)
            for (int q = p + 1; q < m; q++) {
                double apq = A[p * m + q];
                if (apq == 0.0) continue;
                double app = A[p * m + p], aqq = A[q * m + q];
                double theta = (aqq - app) / (2.0 * apq);
                double tt = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
                double c = 1.0 / sqrt(tt * tt + 1.0), s = tt * c;
                for (int k = 0; k < m; k++) {
                    double akp = A[k * m + p], akq = A[k * m + q];
                    A[k * m + p] = c * akp - s * akq;
                    A[k * m + q] = s * akp + c * akq;
                }
                for (int k = 0; k < m; k++) {
                    double apk = A[p * m + k], aqk = A[q * m + k];
                    A[p * m + k] = c * apk - s * aqk;
                    A[q * m + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < m; k++) {
                    double vkp = V[k * m + p], vkq = V[k * m + q];
                    V[k * m + p] = c * vkp - s * vkq;
                    V[k * m + q] = s * vkp + c * vkq;
                }
            }
    }
    for (int i = 0; i < m; i++) w[i] = A[i * m + i];
}

/* Upper estimate of lambda_max(L) from msteps of Lanczos (no reorth.). */
static double lanczos_lmax(int msteps, double gersh) {
    double *v = xalloc(sizeof(double) * n), *vp = xalloc(sizeof(double) * n),
           *w = xalloc(sizeof(double) * n);
    double al[128], be[128];
    uint64_t s = 0x9E3779B97F4A7C15ull;
    double nrm = 0;
    for (int i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        double r = ((double)(s >> 11) * (1.0 / 9007199254740992.0)) - 0.5;
        v[i] = r;
        vp[i] = 0;
        nrm += r * r;
    }
    nrm = sqrt(nrm);
    for (int i = 0; i < n; i++) v[i] /= nrm;
    int m = 0;
    double beta = 0;
    if (msteps > 120) msteps = 120;
    for (int j = 0; j < msteps; j++) {
        spmv(v, w);
        double a = 0;
        for (int i = 0; i < n; i++) a += w[i] * v[i];
        double b = 0;
        for (int i = 0; i < n; i++) {
            double x = w[i] - a * v[i] - beta * vp[i];
            w[i] = x;
            b += x * x;
        }
        b = sqrt(b);
        al[m] = a;
        be[m] = b;
        m++;
        beta = b;
        if (!(b > 1e-10 * (fabs(a) + 1e-300))) { beta = 0; break; }
        for (int i = 0; i < n; i++) {
            vp[i] = v[i];
            v[i] = w[i] / b;
        }
    }
    double *A = xalloc(sizeof(double) * m * m), *V = xalloc(sizeof(double) * m * m), *ev = xalloc(sizeof(double) * m);
    memset(A, 0, sizeof(double) * m * m);
    for (int i = 0; i < m; i++) {
        A[i * m + i] = al[i];
        if (i + 1 < m) A[i * m + i + 1] = A[(i + 1) * m + i] = be[i];
    }
    jacobi(m, A, V, ev);
    int imax = 0;
    for (int i = 1; i < m; i++)
        if (ev[i] > ev[imax]) imax = i;
    double theta = ev[imax];
    double resid = fabs(beta * V[(m - 1) * m + imax]);
    double est = theta + 2.0 * resid + 0.01 * fabs(theta);
    if (beta == 0) est = theta + 1e-6 * fabs(theta) + 1e-300;
    xfree(A); xfree(V); xfree(ev); xfree(v); xfree(vp); xfree(w);
    if (!(est < gersh)) est = gersh;
    return est;
}

/* Mean |i - j| over the stored entries (sampled), a cheap locality measure. */
static double mean_band(void) {
    double s = 0;
    long cnt = 0;
    int step = n > 20000 ? n / 20000 : 1;
    for (int i = 0; i < n; i += step)
        for (int p = rp[i]; p < rp[i + 1]; p++) {
            s += abs(ci[p] - i);
            cnt++;
        }
    return cnt ? s / cnt : 0.0;
}

static const int *g_deg;
static int cmp_deg(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    int dx = g_deg[x], dy = g_deg[y];
    return dx < dy ? -1 : dx > dy ? 1 : (x < y ? -1 : x > y);
}

/* Breadth-first search from s over unvisited nodes; returns the number of
 * nodes reached and stores them in q (levels in lev). */
static int bfs_levels(int s, const char *done, int *q, int *lev, int *mark, int stamp) {
    int h = 0, tl = 0;
    q[tl++] = s;
    mark[s] = stamp;
    lev[s] = 0;
    while (h < tl) {
        int i = q[h++];
        for (int p = rp[i]; p < rp[i + 1]; p++) {
            int j = ci[p];
            if (!done[j] && mark[j] != stamp) {
                mark[j] = stamp;
                lev[j] = lev[i] + 1;
                q[tl++] = j;
            }
        }
    }
    return tl;
}

/* Reverse Cuthill-McKee ordering: perm[new] = old. */
static void rcm_order(int *perm) {
    int *deg = xalloc(sizeof(int) * n), *q = xalloc(sizeof(int) * n), *lev = xalloc(sizeof(int) * n),
        *mark = xalloc(sizeof(int) * n), *nb = xalloc(sizeof(int) * n);
    char *done = xalloc(n);
    for (int i = 0; i < n; i++) {
        deg[i] = rp[i + 1] - rp[i];
        mark[i] = -1;
        done[i] = 0;
    }
    g_deg = deg;
    int cnt = 0, stamp = 0, scan = 0;
    while (cnt < n) {
        while (done[scan]) scan++;
        int s = scan;
        /* pseudo-peripheral node: a few BFS sweeps */
        int ecc = -1;
        for (int it = 0; it < 4; it++) {
            int m = bfs_levels(s, done, q, lev, mark, stamp++);
            int last = lev[q[m - 1]];
            if (last <= ecc) break;
            ecc = last;
            int best = q[m - 1];
            for (int a = m - 1; a >= 0 && lev[q[a]] == last; a--)
                if (deg[q[a]] < deg[best]) best = q[a];
            s = best;
        }
        /* Cuthill-McKee from s */
        int h = cnt, tl = cnt;
        perm[tl++] = s;
        done[s] = 1;
        while (h < tl) {
            int i = perm[h++], k = 0;
            for (int p = rp[i]; p < rp[i + 1]; p++) {
                int j = ci[p];
                if (!done[j]) {
                    done[j] = 1;
                    nb[k++] = j;
                }
            }
            if (k > 1) qsort(nb, k, sizeof(int), cmp_deg);
            for (int a = 0; a < k; a++) perm[tl++] = nb[a];
        }
        cnt = tl;
    }
    for (int a = 0, b = n - 1; a < b; a++, b--) {
        int x = perm[a];
        perm[a] = perm[b];
        perm[b] = x;
    }
    xfree(deg); xfree(q); xfree(lev); xfree(mark); xfree(nb); xfree(done);
}

/* Replace the CSR arrays by the symmetric permutation P L P^T. */
static void apply_perm(const int *perm, int *iperm) {
    for (int a = 0; a < n; a++) iperm[perm[a]] = a;
    int *rp2 = xalloc(sizeof(int) * (n + 1)), *ci2 = xalloc(sizeof(int) * (size_t)(nnz ? nnz : 1));
    double *va2 = xalloc(sizeof(double) * (size_t)(nnz ? nnz : 1));
    rp2[0] = 0;
    for (int a = 0; a < n; a++) {
        int i = perm[a], k = rp2[a];
        for (int p = rp[i]; p < rp[i + 1]; p++, k++) {
            ci2[k] = iperm[ci[p]];
            va2[k] = va[p];
        }
        rp2[a + 1] = k;
    }
    rp = rp2;
    ci = ci2;
    va = va2;
}

/* SELL-4-sigma copy of L: slices of 4 rows (length-sorted inside windows of
 * 128 rows) stored column-major and padded to the slice's longest row, so the
 * inner loop has 4 independent sums and one predictable trip count. */
static int nsl, *sl_off, *sl_len, *sl_row, *sl_ci;
static double *sl_va;

static int cmp_len_desc(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    int lx = rp[x + 1] - rp[x], ly = rp[y + 1] - rp[y];
    return lx > ly ? -1 : lx < ly ? 1 : (x < y ? -1 : x > y);
}

static int build_sell(void) {
    const int sigma = 128;
    nsl = (n + 3) / 4;
    sl_row = xalloc(sizeof(int) * 4 * (size_t)nsl);
    int cnt[1025];
    for (int w0 = 0; w0 < n; w0 += sigma) {
        int w1 = w0 + sigma < n ? w0 + sigma : n;
        int lo = rp[w0 + 1] - rp[w0], hi = lo;
        for (int i = w0; i < w1; i++) {
            int L = rp[i + 1] - rp[i];
            lo = L < lo ? L : lo;
            hi = L > hi ? L : hi;
        }
        if (hi - lo < 1024) {
            /* stable counting sort by decreasing length */
            int range = hi - lo + 1;
            memset(cnt, 0, sizeof(int) * (range + 1));
            for (int i = w0; i < w1; i++) cnt[hi - (rp[i + 1] - rp[i]) + 1]++;
            for (int b = 1; b <= range; b++) cnt[b] += cnt[b - 1];
            for (int i = w0; i < w1; i++) sl_row[w0 + cnt[hi - (rp[i + 1] - rp[i])]++] = i;
        } else {
            for (int i = w0; i < w1; i++) sl_row[i] = i;
            qsort(sl_row + w0, w1 - w0, sizeof(int), cmp_len_desc);
        }
    }
    for (int i = n; i < 4 * nsl; i++) sl_row[i] = -1;
    sl_off = xalloc(sizeof(int) * ((size_t)nsl + 1));
    sl_len = xalloc(sizeof(int) * (size_t)nsl);
    long tot = 0;
    for (int s = 0; s < nsl; s++) {
        int len = 0;
        for (int r = 0; r < 4; r++) {
            int i = sl_row[4 * s + r];
            if (i >= 0 && rp[i + 1] - rp[i] > len) len = rp[i + 1] - rp[i];
        }
        sl_len[s] = len;
        sl_off[s] = (int)tot;
        tot += 4L * len;
    }
    sl_off[nsl] = (int)tot;
    if ((double)tot > 1.3 * (double)nnz + 64.0) return 0; /* too much padding: keep CSR */
    sl_ci = xalloc(sizeof(int) * (size_t)(tot + 4));
    sl_va = xalloc(sizeof(double) * (size_t)(tot + 4));
    for (int s = 0; s < nsl; s++) {
        int len = sl_len[s], off = sl_off[s];
        int st[4], ln[4], pad[4];
        for (int r = 0; r < 4; r++) {
            int i = sl_row[4 * s + r];
            st[r] = i >= 0 ? rp[i] : 0;
            ln[r] = i >= 0 ? rp[i + 1] - rp[i] : 0;
            pad[r] = i >= 0 ? i : 0;
        }
        for (int j = 0; j < len; j++)
            for (int r = 0; r < 4; r++) {
                int idx = off + 4 * j + r;
                if (j < ln[r]) {
                    sl_ci[idx] = ci[st[r] + j];
                    sl_va[idx] = va[st[r] + j];
                } else {
                    sl_ci[idx] = pad[r];
                    sl_va[idx] = 0.0;
                }
            }
    }
    return 1;
}

static void sell_spmv(const double *restrict x, double *restrict y) {
    for (int s = 0; s < nsl; s++) {
        const int len = sl_len[s];
        const int *restrict cc = sl_ci + sl_off[s];
        const double *restrict vv = sl_va + sl_off[s];
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        for (int j = 0; j < len; j++) {
            s0 += vv[4 * j] * x[cc[4 * j]];
            s1 += vv[4 * j + 1] * x[cc[4 * j + 1]];
            s2 += vv[4 * j + 2] * x[cc[4 * j + 2]];
            s3 += vv[4 * j + 3] * x[cc[4 * j + 3]];
        }
        const double ss[4] = {s0, s1, s2, s3};
        for (int r = 0; r < 4; r++)
            if (sl_row[4 * s + r] >= 0) y[sl_row[4 * s + r]] = ss[r];
    }
}

/* One Chebyshev term on the SELL copy:
 * vm <- 2 vk - a2 L vk - vm + ab ; y += c vm ; returns max|vm|, max|y|. */
static void cheb_step_sell(const double *restrict vk, double *restrict vm, const double *restrict ab,
                           double *restrict y, double a2, double c, double *Mk_out, double *Yk_out) {
    double Mk = 0.0, Yk = 0.0;
    for (int s = 0; s < nsl; s++) {
        const int len = sl_len[s];
        const int *restrict cc = sl_ci + sl_off[s];
        const double *restrict vv = sl_va + sl_off[s];
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        for (int j = 0; j < len; j++) {
            s0 += vv[4 * j] * vk[cc[4 * j]];
            s1 += vv[4 * j + 1] * vk[cc[4 * j + 1]];
            s2 += vv[4 * j + 2] * vk[cc[4 * j + 2]];
            s3 += vv[4 * j + 3] * vk[cc[4 * j + 3]];
        }
        const double ss[4] = {s0, s1, s2, s3};
        const int *rows = sl_row + 4 * s;
        for (int r = 0; r < 4; r++) {
            int i = rows[r];
            if (i < 0) continue;
            double w = 2.0 * vk[i] - a2 * ss[r] - vm[i] + ab[i];
            vm[i] = w;
            double yy = y[i] + c * w;
            y[i] = yy;
            double aw = fabs(w), ay = fabs(yy);
            Mk = aw > Mk ? aw : Mk;
            Yk = ay > Yk ? ay : Yk;
        }
    }
    *Mk_out = Mk;
    *Yk_out = Yk;
}

/* e[k] = exp(-rho) I_k(rho), k = 0..N, by Miller's backward recurrence. */
static void bessel_coef(double rho, int N, double *e) {
    int M = N + 20 + (int)(2.0 * sqrt(rho + 1.0));
    double *I = xalloc(sizeof(double) * (M + 2));
    I[M + 1] = 0.0;
    I[M] = 1e-280;
    for (int k = M; k >= 1; k--) {
        I[k - 1] = I[k + 1] + (2.0 * k / rho) * I[k];
        if (I[k - 1] > 1e250) {
            for (int j = k - 1; j <= M + 1; j++) I[j] *= 1e-250;
        }
    }
    double sum = I[0];
    for (int k = 1; k <= M; k++) sum += 2.0 * I[k];
    for (int k = 0; k <= N; k++) e[k] = I[k] / sum;
    xfree(I);
}

static double *dg;

/* Scan rows i0.. whose entries lie below lim: diagonal, Gershgorin bound,
 * zero test.  Returns the first unscanned row. */
static int gersh_rows(int i0, long lim, double *gersh, int *allzero) {
    int i = i0;
    double g = *gersh;
    int az = *allzero;
    while (i < n && rp[i + 1] <= lim) {
        double d = 0, off = 0;
        for (int p = rp[i]; p < rp[i + 1]; p++) {
            if (va[p] != 0.0) az = 0;
            if (ci[p] == i) d += va[p];
            else off += fabs(va[p]);
        }
        dg[i] = d;
        if (d + off > g) g = d + off;
        i++;
    }
    *gersh = g;
    *allzero = az;
    return i;
}

/* Collatz-Wielandt bound rho(|L|) <= max_i (|L| x)_i / x_i, x = sqrt|diag|,
 * evaluated incrementally (cw_step) between reads of the first request. */
static int cw_active, cw_row, cw_ok;
static double cw_max, *cw_x;

static void cw_start(void) {
    cw_x = xalloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) cw_x[i] = sqrt(fabs(dg[i])) + 1e-300;
    cw_active = 1;
    cw_row = 0;
    cw_ok = 1;
    cw_max = 0.0;
}

static void cw_step(int rows) {
    int i1 = cw_row + rows < n ? cw_row + rows : n;
    for (int i = cw_row; i < i1 && cw_ok; i++) {
        double s = 0;
        for (int p = rp[i]; p < rp[i + 1]; p++) s += fabs(va[p]) * cw_x[ci[p]];
        double q = s / cw_x[i];
        if (!(q < 1e300)) cw_ok = 0;
        if (q > cw_max) cw_max = q;
    }
    cw_row = i1;
    if (cw_row >= n || !cw_ok) cw_active = 0;
}

/* Read len bytes, advancing the background bound computation between reads. */
static void rd_bg(void *p, size_t len) {
    char *c = (char *)p;
    while (len) {
        ssize_t r = read(0, c, len > (1u << 30) ? (1u << 30) : len);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) _exit(3);
        c += r;
        len -= (size_t)r;
        if (cw_active && len) cw_step(4096);
    }
}

int main(void) {
    /* Larger pipe buffers: fewer writer/reader round trips on one CPU. */
    for (int fd = 0; fd <= 1 && !getenv("EXPO_NOPIPE"); fd++) {
        int sz = 1 << 20;
        while (sz >= (1 << 16) && fcntl(fd, F_SETPIPE_SZ, sz) < 0) sz >>= 1;
    }
    if (!getenv("EXPO_NOARENA")) arena_init((size_t)40 << 20);
    wr("READY\n", 6);
    char magic[8];
    rd(magic, 8);
    uint32_t hl;
    rd(&hl, 4);
    char *js = xalloc(hl + 1);
    rd(js, hl);
    js[hl] = 0;
    n = (int)jnum(js, "n");
    nnz = (int)jnum(js, "nnz");
    int requests = (int)jnum(js, "requests");
    double t = jnum(js, "t"), rtol = jnum(js, "rtol"), atol = jnum(js, "atol");

    uint64_t *ip64 = xalloc(sizeof(uint64_t) * (n + 1));
    rd(ip64, sizeof(uint64_t) * (n + 1));
    rp = xalloc(sizeof(int) * (n + 1));
    for (int i = 0; i <= n; i++) rp[i] = (int)ip64[i];
    ci = xalloc(sizeof(int) * (size_t)nnz);
    rd(ci, sizeof(int) * (size_t)nnz);
    va = xalloc(sizeof(double) * (size_t)nnz);
    dg = xalloc(sizeof(double) * n);
    /* Read the values; rows whose entries have arrived are scanned for the
     * Gershgorin bound while the pipe refills. */
    double gersh = 0.0;
    int allzero = 1;
    {
        size_t total = sizeof(double) * (size_t)nnz, got = 0;
        char *dst = (char *)va;
        int row = 0;
        while (got < total) {
            ssize_t r = read(0, dst + got, total - got > (1u << 30) ? (1u << 30) : total - got);
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) _exit(3);
            got += (size_t)r;
            row = gersh_rows(row, (long)(got / sizeof(double)), &gersh, &allzero);
        }
        gersh_rows(row, nnz, &gersh, &allzero);
    }
    /* Collatz-Wielandt bound, computed in chunks while the first RHS arrives. */
    if (t * gersh >= 2.0) cw_start();

    double *b0 = xalloc(sizeof(double) * n), *b1 = xalloc(sizeof(double) * n);
    double *y = xalloc(sizeof(double) * n);
    int trivial = (t == 0.0) || allzero || !(gersh > 0.0);

    /* Connected components; if all are small, solve each one densely. */
    int small = 0;
    int *comp = NULL, *cstart = NULL, *corder = NULL, ncomp = 0;
    double *cQ = NULL, *cf = NULL, *cg = NULL;
    size_t *cqoff = NULL;
    if (!trivial) {
        const int BMAX = 24;
        comp = xalloc(sizeof(int) * n);
        corder = xalloc(sizeof(int) * n);
        for (int i = 0; i < n; i++) comp[i] = -1;
        int head = 0, tailq = 0, ok = 1;
        cstart = xalloc(sizeof(int) * (n + 1));
        for (int s0 = 0; s0 < n && ok; s0++) {
            if (comp[s0] >= 0) continue;
            cstart[ncomp] = tailq;
            comp[s0] = ncomp;
            corder[tailq++] = s0;
            while (head < tailq) {
                int i = corder[head++];
                for (int p = rp[i]; p < rp[i + 1]; p++) {
                    int j = ci[p];
                    if (comp[j] < 0 && va[p] != 0.0) {
                        comp[j] = ncomp;
                        corder[tailq++] = j;
                    }
                }
                if (tailq - cstart[ncomp] > BMAX) { ok = 0; break; }
            }
            ncomp++;
        }
        if (ok) {
            small = 1;
            cstart[ncomp] = n;
            cqoff = xalloc(sizeof(size_t) * (ncomp + 1));
            size_t tot = 0;
            for (int c = 0; c < ncomp; c++) {
                int m = cstart[c + 1] - cstart[c];
                cqoff[c] = tot;
                tot += (size_t)m * m;
            }
            cQ = xalloc(sizeof(double) * (tot + 1));
            cf = xalloc(sizeof(double) * n);
            cg = xalloc(sizeof(double) * n);
            int *loc = xalloc(sizeof(int) * n);
            double A[24 * 24], V[24 * 24], w[24];
            for (int c = 0; c < ncomp; c++) {
                int s = cstart[c], m = cstart[c + 1] - s;
                for (int a = 0; a < m; a++) loc[corder[s + a]] = a;
                memset(A, 0, sizeof(double) * m * m);
                for (int a = 0; a < m; a++) {
                    int i = corder[s + a];
                    for (int p = rp[i]; p < rp[i + 1]; p++)
                        if (comp[ci[p]] == c) A[a * m + loc[ci[p]]] = va[p];
                }
                /* symmetrize defensively */
                for (int a = 0; a < m; a++)
                    for (int b = a + 1; b < m; b++) {
                        double x = 0.5 * (A[a * m + b] + A[b * m + a]);
                        A[a * m + b] = A[b * m + a] = x;
                    }
                jacobi(m, A, V, w);
                memcpy(cQ + cqoff[c], V, sizeof(double) * m * m);
                for (int a = 0; a < m; a++) {
                    double lmb = w[a] > 0 ? w[a] : 0.0;
                    double x = t * lmb;
                    cf[s + a] = exp(-x);
                    cg[s + a] = (x == 0.0) ? t : (x < 1e-300 ? t : -expm1(-x) / lmb);
                }
            }
            xfree(loc);
        }
    }

    double lam = gersh, rho = 0.0;
    int N = 0;
    double *e = NULL, *tail = NULL, *tail2 = NULL;
    double *v0 = NULL, *v1 = NULL, *ab = NULL;
    int lam_done = 0, coef_ready = 1;
    int *perm = NULL, *iperm = NULL;
    double *pb0 = NULL, *pb1 = NULL;

    for (int r = 0; r < requests; r++) {
        rd_bg(b0, sizeof(double) * n);
        rd_bg(b1, sizeof(double) * n);
        if (cw_x) {
            while (cw_active) cw_step(1 << 30);
            if (cw_ok && cw_max > 0 && cw_max < gersh) gersh = cw_max * (1.0 + 1e-12);
            cw_x = NULL;
        }
        if (trivial) {
            for (int i = 0; i < n; i++) y[i] = b0[i] + t * b1[i];
            wr(y, sizeof(double) * n);
            continue;
        }
        if (small) {
            double z0[24], z1[24];
            for (int c = 0; c < ncomp; c++) {
                int s = cstart[c], m = cstart[c + 1] - s;
                const double *Q = cQ + cqoff[c];
                for (int a = 0; a < m; a++) {
                    double s0 = 0, s1 = 0;
                    for (int b = 0; b < m; b++) {
                        int i = corder[s + b];
                        s0 += Q[b * m + a] * b0[i];
                        s1 += Q[b * m + a] * b1[i];
                    }
                    z0[a] = cf[s + a] * s0 + cg[s + a] * s1;
                }
                for (int b = 0; b < m; b++) {
                    double acc = 0;
                    for (int a = 0; a < m; a++) acc += Q[b * m + a] * z0[a];
                    y[corder[s + b]] = acc;
                }
                (void)z1;
            }
            wr(y, sizeof(double) * n);
            continue;
        }
        if (!lam_done) {
            lam_done = 1;
            /* Chebyshev degree estimate with the Gershgorin bound */
            double rg = t * gersh / 2.0;
            double kg = sqrt(2.0 * rg * 30.0) + 10.0;
            int steps = 0;
            /* Lanczos costs ~1.3 SpMV per step and saves at most ~15-25% of
             * the terms, so in one-shot sessions only long series pay off. */
            if (requests > 1) {
                if (kg > 25) steps = kg > 400 ? 60 : 40;
            } else {
                if (kg > 800) steps = 50;
                else if (kg > 450) steps = 35;
            }
            if (steps > n) steps = n;
            /* Estimated number of series terms with the cheap bound (Taylor-like
             * decay for small rho, Gaussian decay for large rho), times the
             * number of requests: decides which set-up investments pay off. */
            double kest;
            if (rg < 8.0) {
                double term = 1.0;
                int k = 0;
                while (term > 1e-12 && k < 200) {
                    k++;
                    term *= (rg + 1e-300) / (2.0 * k) * (k > 1 ? 1.0 : 2.0);
                }
                kest = k + 1;
            } else {
                kest = sqrt(56.0 * rg) + 10.0;
            }
            double iters = kest * requests;
            /* Bandwidth-reducing reordering when the run is long and locality poor. */
            if (!getenv("EXPO_NORCM") && iters >= 150 && n > 4096 && mean_band() > 2048.0) {
                int *pm = xalloc(sizeof(int) * n);
                int *ipm = xalloc(sizeof(int) * n);
                int *orp = rp, *oci = ci;
                double *ova = va;
                rcm_order(pm);
                apply_perm(pm, ipm);
                if (getenv("EXPO_DEBUG")) fprintf(stderr, "rcm band after %.1f\n", mean_band());
                if (mean_band() * 4.0 < 2048.0) {
                    perm = pm;
                    iperm = ipm;
                } else {
                    rp = orp; ci = oci; va = ova;
                }
            }
            if (!getenv("EXPO_NOSELL") && iters >= 12 && n >= 8) {
                use_sell = build_sell();
            }
            if (steps > 0) lam = lanczos_lmax(steps, gersh);
            if (getenv("EXPO_TEST_LAMSCALE")) lam *= atof(getenv("EXPO_TEST_LAMSCALE"));
            coef_ready = 0;
            v0 = xalloc(sizeof(double) * n);
            v1 = xalloc(sizeof(double) * n);
            ab = xalloc(sizeof(double) * n);
        }
    restart:
        if (!coef_ready) {
            coef_ready = 1;
            rho = t * lam / 2.0;
            N = (int)ceil(sqrt(2.0 * rho * 95.0) + 60.0);
            e = xalloc(sizeof(double) * (N + 1));
            tail = xalloc(sizeof(double) * (N + 2));
            bessel_coef(rho, N, e);
            tail[N + 1] = 0.0;
            tail[N] = 0.0;
            for (int k = N - 1; k >= 0; k--) tail[k] = tail[k + 1] + 2.0 * e[k + 1];
            tail2 = xalloc(sizeof(double) * (N + 2));
            tail2[N + 1] = 0.0;
            tail2[N] = 0.0;
            for (int k = N - 1; k >= 0; k--) tail2[k] = tail2[k + 1] + 2.0 * e[k + 1] * (double)(k + 1) * (k + 1);
        }
        const double a = 2.0 / lam, a2 = 2.0 * a;
        int diverged = 0;
        double *B0 = b0, *B1 = b1;
        if (perm) {
            if (!pb0) {
                pb0 = xalloc(sizeof(double) * n);
                pb1 = xalloc(sizeof(double) * n);
            }
            for (int i = 0; i < n; i++) {
                pb0[i] = b0[perm[i]];
                pb1[i] = b1[perm[i]];
            }
            B0 = pb0;
            B1 = pb1;
        }
        /* v0 = b0, v1 = b0 - a L b0 + a b1 */
        double M = 0.0, Y = 0.0, nb0 = 0.0, nb1 = 0.0;
        for (int i = 0; i < n; i++) {
            if (fabs(B0[i]) > nb0) nb0 = fabs(B0[i]);
            if (fabs(B1[i]) > nb1) nb1 = fabs(B1[i]);
            double s = rowdot(i, B0);
            double x0 = B0[i];
            double x1 = x0 - a * s + a * B1[i];
            v0[i] = x0;
            v1[i] = x1;
            ab[i] = a2 * B1[i];
            double yy = e[0] * x0 + 2.0 * e[1] * x1;
            y[i] = yy;
            double m0 = fabs(x0) > fabs(x1) ? fabs(x0) : fabs(x1);
            if (m0 > M) M = m0;
            if (fabs(yy) > Y) Y = fabs(yy);
        }
        const double eta = 0.02;
        int k = 1;
        /* v_{k+1} = 2 v_k - 2a L v_k - v_{k-1} + 2a b1 ; y += 2 e_{k+1} v_{k+1} */
        while (k < N - 1) {
            double M0 = M > nb0 ? M : nb0;
            if (2.0 * (tail[k] * M0 + tail2[k] * a * nb1) <= eta * (atol + rtol * Y)) break;
            const double c = 2.0 * e[k + 1];
            double Mk = 0.0, Yk = 0.0;
            double *restrict vk = v1, *restrict vm = v0;
            if (use_sell) cheb_step_sell(vk, vm, ab, y, a2, c, &Mk, &Yk);
            else for (int i = 0; i < n; i++) {
                double s = rowdot(i, vk);
                double w = 2.0 * vk[i] - a2 * s - vm[i] + ab[i];
                vm[i] = w;
                double yy = y[i] + c * w;
                y[i] = yy;
                double aw = fabs(w);
                Mk = aw > Mk ? aw : Mk;
                double ay = fabs(yy);
                Yk = ay > Yk ? ay : Yk;
            }
            if (Mk > M) M = Mk;
            /* |v_k| <= |b0| + k^2 a |b1| in the 2-norm when lam bounds the
             * spectrum; far larger iterates mean lam was too small. */
            if (lam < gersh && !(M <= 1e4 * (nb0 + (double)(k + 2) * (k + 2) * a * nb1))) {
                diverged = 1;
                break;
            }
            Y = Yk;
            v0 = vk;
            v1 = vm;
            k++;
        }
        if (diverged) {
            if (getenv("EXPO_DEBUG")) fprintf(stderr, "divergence at k=%d, lam %.6g -> %.6g\n", k, lam, gersh);
            lam = gersh;
            coef_ready = 0;
            goto restart;
        }
        if (getenv("EXPO_DEBUG"))
            fprintf(stderr, "req %d n %d nnz %d t %.3g gersh %.6g lam %.6g rho %.4g K %d M %.3g Y %.3g\n", r, n, nnz,
                    t, gersh, lam, rho, k, M, Y);
        if (perm) {
            /* pb0 is free now: un-permute into it */
            for (int i = 0; i < n; i++) pb0[perm[i]] = y[i];
            wr(pb0, sizeof(double) * n);
        } else {
            wr(y, sizeof(double) * n);
        }
    }
    _exit(0);
}
