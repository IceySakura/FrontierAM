/*
 * FAMEXP2 exponential propagation solver.
 *
 *   y = exp(-tL) b0 + t*phi1(-tL) b1,   L symmetric positive semidefinite (CSR)
 *
 * Method: Chebyshev expansion of f0(x)=exp(-t x) and f1(x)=(1-exp(-t x))/x on a
 * guaranteed spectral interval [lo,hi] (Gershgorin, tightened at the top by a
 * safeguarded Lanczos estimate).  Both right-hand sides run through one fused
 * three-term recurrence that reads the matrix once per step (2-column SpMM);
 * the partial sum, the running infinity norms and the next Chebyshev vector are
 * produced in the same sweep.  The degree is chosen adaptively per request from
 * the coefficient tails and the observed infinity norms of the Chebyshev vectors.
 * See README.md.
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
#include <xmmintrin.h>
#include <time.h>
static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + 1e-9 * ts.tv_nsec; }
static double T_start;
#define TLOG(msg) fprintf(stderr, "[%.4f] %s\n", now_s() - T_start, msg)

typedef double v2d __attribute__((vector_size(16)));

static void die(const char *msg) {
    fprintf(stderr, "expprop fatal: %s (errno %d)\n", msg, errno);
    _exit(3);
}

static void read_full(void *buf, size_t len) {
    char *p = (char *)buf;
    while (len) {
        size_t chunk = len > ((size_t)1 << 30) ? ((size_t)1 << 30) : len;
        ssize_t r = read(0, p, chunk);
        if (r < 0) {
            if (errno == EINTR) continue;
            die("read");
        }
        if (r == 0) die("unexpected eof");
        p += r;
        len -= (size_t)r;
    }
}

static void write_full(const void *buf, size_t len) {
    const char *p = (const char *)buf;
    while (len) {
        ssize_t r = write(1, p, len);
        if (r < 0) {
            if (errno == EINTR) continue;
            die("write");
        }
        p += r;
        len -= (size_t)r;
    }
}

static double json_num(const char *js, const char *key) {
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(js, pat);
    if (!p) die("json key missing");
    p += strlen(pat);
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p != ':') die("json syntax");
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    char *end;
    double v = strtod(p, &end);
    if (end == p) die("json number");
    return v;
}

/* Bump arena, pre-faulted before READY so the timed session takes no page faults
   for its working set; larger requests fall back to the system allocator. */
static char *arena;
static size_t arena_cap, arena_used;

static void *xmalloc(size_t sz) {
    void *p = NULL;
    if (sz == 0) sz = 64;
    sz = (sz + 63) & ~(size_t)63;
    if (arena && arena_used + sz <= arena_cap) {
        p = arena + arena_used;
        arena_used += sz;
        return p;
    }
    if (posix_memalign(&p, 64, sz)) die("malloc");
    return p;
}

static void xfree(void *p) {
    if (!p) return;
    if (arena && (char *)p >= arena && (char *)p < arena + arena_cap) return;
    free(p);
}

/* ------------------------------------------------------------------------- */
/* Matrix (off-diagonal CSR + diagonal).                                      */

static int n;
static uint32_t *rp;      /* n+1 row pointers into off-diagonal arrays */
static uint32_t *ci;      /* off-diagonal column indices */
static double *cv;        /* off-diagonal values (unscaled) */
static double *dg;        /* diagonal */
static int uniform_off;   /* all off-diagonal values identical */
static double off0;       /* that value */

/* Scaled operator for the recurrence: 2X = s_off*offdiag + diag2. */
static double scale2;     /* 2/h: the off-diagonal sums are scaled once per row */
static double sval;       /* scaled uniform value 2*off0/h */
static double *dd;        /* scaled diagonal term 2*alpha*d_i + 2*beta */

/* Plain y = L x (used by the Lanczos spectral estimate). */
static void spmv_plain(const double *x, double *y) {
    for (int i = 0; i < n; i++) {
        double s = dg[i] * x[i];
        uint32_t k0 = rp[i], k1 = rp[i + 1];
        if (uniform_off) {
            double a = 0.0;
            for (uint32_t k = k0; k < k1; k++) a += x[ci[k]];
            s += off0 * a;
        } else {
            for (uint32_t k = k0; k < k1; k++) s += cv[k] * x[ci[k]];
        }
        y[i] = s;
    }
}

/*
 * Fused Chebyshev step on interleaved pairs (b0-part, b1-part):
 *   w = sigma*(2X u) - tau*v   (stored into v)
 *   y += c0*w.x + c1*w.y
 * Returns infinity norms of the two halves of w and of y.
 */
static void cheb_step(const v2d *restrict u, v2d *restrict v, double *restrict y,
                      double sigma, double tau, double c0, double c1,
                      double *m0, double *m1, double *ym) {
    v2d mx = {0.0, 0.0};
    double ymax = 0.0;
    const v2d sg = {sigma, sigma}, ta = {tau, tau};
    if (uniform_off) {
        const v2d s2 = {sval, sval};
        for (int i = 0; i < n; i++) {
            v2d acc = {0.0, 0.0};
            uint32_t k0 = rp[i], k1 = rp[i + 1];
            for (uint32_t k = k0; k < k1; k++) acc += u[ci[k]];
            v2d di = {dd[i], dd[i]};
            v2d w = sg * (s2 * acc + di * u[i]) - ta * v[i];
            v[i] = w;
            double yi = y[i] + c0 * w[0] + c1 * w[1];
            y[i] = yi;
            v2d aw = {fabs(w[0]), fabs(w[1])};
            mx = (v2d){aw[0] > mx[0] ? aw[0] : mx[0], aw[1] > mx[1] ? aw[1] : mx[1]};
            double ay = fabs(yi);
            if (ay > ymax) ymax = ay;
        }
    } else {
        const v2d s2 = {scale2, scale2};
        for (int i = 0; i < n; i++) {
            v2d acc = {0.0, 0.0};
            uint32_t k0 = rp[i], k1 = rp[i + 1];
            for (uint32_t k = k0; k < k1; k++) {
                v2d a = {cv[k], cv[k]};
                acc += a * u[ci[k]];
            }
            v2d di = {dd[i], dd[i]};
            v2d w = sg * (s2 * acc + di * u[i]) - ta * v[i];
            v[i] = w;
            double yi = y[i] + c0 * w[0] + c1 * w[1];
            y[i] = yi;
            v2d aw = {fabs(w[0]), fabs(w[1])};
            mx = (v2d){aw[0] > mx[0] ? aw[0] : mx[0], aw[1] > mx[1] ? aw[1] : mx[1]};
            double ay = fabs(yi);
            if (ay > ymax) ymax = ay;
        }
    }
    *m0 = mx[0];
    *m1 = mx[1];
    *ym = ymax;
}




static const int *g_deg;
static int cmp_deg(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    int dx = g_deg[x], dy = g_deg[y];
    return dx < dy ? -1 : dx > dy ? 1 : (x < y ? -1 : x > y);
}
typedef struct { uint32_t c; double a; } colval;
static int cmp_col(const void *a, const void *b) {
    uint32_t x = ((const colval *)a)->c, y = ((const colval *)b)->c;
    return x < y ? -1 : x > y;
}

/* Reverse Cuthill-McKee reordering of the (symmetric) off-diagonal graph.
   Returns 1 and fills perm (new->old) if the BFS level structure is deep
   enough for the reordering to improve locality, else 0. */
static int rcm_order(int *perm) {
    int *deg = xmalloc(sizeof(int) * n);
    int maxd = 0;
    for (int i = 0; i < n; i++) { deg[i] = (int)(rp[i + 1] - rp[i]); if (deg[i] > maxd) maxd = deg[i]; }
    /* nodes sorted by degree (counting sort) to pick start nodes */
    int *cnt = calloc((size_t)maxd + 2, sizeof(int));
    int *byd = xmalloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) cnt[deg[i] + 1]++;
    for (int d = 0; d <= maxd; d++) cnt[d + 1] += cnt[d];
    for (int i = 0; i < n; i++) byd[cnt[deg[i]]++] = i;
    free(cnt);
    char *vis = calloc((size_t)n, 1);
    int head = 0, tail = 0, sp = 0, maxlevels = 0;
    while (tail < n) {
        while (vis[byd[sp]]) sp++;
        int s0 = byd[sp];
        vis[s0] = 1;
        perm[tail++] = s0;
        int levels = 0, level_end = tail;
        while (head < tail) {
            int v = perm[head++];
            int first = tail;
            for (uint32_t k = rp[v]; k < rp[v + 1]; k++) {
                int w = (int)ci[k];
                if (!vis[w]) { vis[w] = 1; perm[tail++] = w; }
            }
            /* sort the new nodes by degree */
            if (tail - first > 16) {
                g_deg = deg;
                qsort(perm + first, (size_t)(tail - first), sizeof(int), cmp_deg);
            } else {
                for (int a = first + 1; a < tail; a++) {
                    int x = perm[a], dx = deg[x], b = a - 1;
                    while (b >= first && deg[perm[b]] > dx) { perm[b + 1] = perm[b]; b--; }
                    perm[b + 1] = x;
                }
            }
            if (head == level_end) { levels++; level_end = tail; }
        }
        if (levels > maxlevels) maxlevels = levels;
    }
    free(vis); xfree(byd); xfree(deg);
    fprintf(stderr, "rcm: max BFS levels %d\n", maxlevels);
    if (maxlevels < 40) return 0;
    for (int a = 0, b = n - 1; a < b; a++, b--) { int t = perm[a]; perm[a] = perm[b]; perm[b] = t; }
    return 1;
}

/* Apply perm (new->old) to the off-diagonal CSR and diagonal. */
static void apply_perm(const int *perm, int *iperm) {
    for (int i = 0; i < n; i++) iperm[perm[i]] = i;
    size_t nnz = rp[n];
    uint32_t *nrp = xmalloc(sizeof(uint32_t) * (n + 1));
    uint32_t *nci = xmalloc(sizeof(uint32_t) * (nnz + 1));
    double *ncv = xmalloc(sizeof(double) * (nnz + 1));
    double *ndg = xmalloc(sizeof(double) * n);
    size_t w = 0;
    for (int i = 0; i < n; i++) {
        int o = perm[i];
        nrp[i] = (uint32_t)w;
        ndg[i] = dg[o];
        size_t first = w;
        uint32_t len = rp[o + 1] - rp[o];
        if (len > 16) {
            colval *tmp = malloc(sizeof(colval) * len);
            if (!tmp) die("malloc");
            for (uint32_t q = 0; q < len; q++) { tmp[q].c = (uint32_t)iperm[ci[rp[o] + q]]; tmp[q].a = cv[rp[o] + q]; }
            qsort(tmp, len, sizeof(colval), cmp_col);
            for (uint32_t q = 0; q < len; q++) { nci[w] = tmp[q].c; ncv[w] = tmp[q].a; w++; }
            free(tmp);
            continue;
        }
        for (uint32_t k = rp[o]; k < rp[o + 1]; k++) {
            uint32_t c = (uint32_t)iperm[ci[k]];
            double a = cv[k];
            size_t b = w;
            while (b > first && nci[b - 1] > c) { nci[b] = nci[b - 1]; ncv[b] = ncv[b - 1]; b--; }
            nci[b] = c;
            ncv[b] = a;
            w++;
        }
    }
    nrp[n] = (uint32_t)w;
    xfree(rp); xfree(ci); xfree(cv); xfree(dg);
    rp = nrp; ci = nci; cv = ncv; dg = ndg;
}

/* Upper-triangle (symmetric) storage: each off-diagonal pair stored once. */
static uint32_t *urp, *uci;   /* row pointers / columns (j > i) */
static double *usv;           /* upper values a_ij (unscaled) */
static v2d *zacc;             /* scatter accumulator, zero between sweeps */
static int use_sym;

static void cheb_step_sym(const v2d *restrict u, v2d *restrict v, double *restrict y,
                          double sigma, double tau, double c0, double c1,
                          double *m0, double *m1, double *ym) {
    v2d mx = {0.0, 0.0};
    double ymax = 0.0;
    v2d *restrict z = zacc;
    const v2d sg = {sigma, sigma}, ta = {tau, tau};
    if (uniform_off) {
        const v2d s2 = {sval, sval};
        for (int i = 0; i < n; i++) {
            v2d ui = u[i];
            v2d g = z[i];
            uint32_t k0 = urp[i], k1 = urp[i + 1];
            for (uint32_t k = k0; k < k1; k++) {
                uint32_t j = uci[k];
                g += u[j];
                z[j] += ui;
            }
            z[i] = (v2d){0.0, 0.0};
            v2d di = {dd[i], dd[i]};
            v2d w = sg * (s2 * g + di * ui) - ta * v[i];
            v[i] = w;
            double yi = y[i] + c0 * w[0] + c1 * w[1];
            y[i] = yi;
            v2d aw = {fabs(w[0]), fabs(w[1])};
            mx = (v2d){aw[0] > mx[0] ? aw[0] : mx[0], aw[1] > mx[1] ? aw[1] : mx[1]};
            double ay = fabs(yi);
            if (ay > ymax) ymax = ay;
        }
    } else {
        const v2d s2 = {scale2, scale2};
        for (int i = 0; i < n; i++) {
            v2d ui = u[i];
            v2d acc = z[i];
            uint32_t k0 = urp[i], k1 = urp[i + 1];
            for (uint32_t k = k0; k < k1; k++) {
                uint32_t j = uci[k];
                v2d a = {usv[k], usv[k]};
                acc += a * u[j];
                z[j] += a * ui;
            }
            z[i] = (v2d){0.0, 0.0};
            v2d di = {dd[i], dd[i]};
            v2d w = sg * (s2 * acc + di * ui) - ta * v[i];
            v[i] = w;
            double yi = y[i] + c0 * w[0] + c1 * w[1];
            y[i] = yi;
            v2d aw = {fabs(w[0]), fabs(w[1])};
            mx = (v2d){aw[0] > mx[0] ? aw[0] : mx[0], aw[1] > mx[1] ? aw[1] : mx[1]};
            double ay = fabs(yi);
            if (ay > ymax) ymax = ay;
        }
    }
    *m0 = mx[0];
    *m1 = mx[1];
    *ym = ymax;
}

/* Build upper storage; returns 0 if the stored matrix is not exactly symmetric. */
static int build_upper(void) {
    size_t nnz = rp[n];
    urp = xmalloc(sizeof(uint32_t) * (n + 1));
    uci = xmalloc(sizeof(uint32_t) * (nnz / 2 + 1));
    double *ucv = xmalloc(sizeof(double) * (nnz / 2 + 1));
    uint32_t *cur = xmalloc(sizeof(uint32_t) * (n + 1));
    size_t w = 0;
    /* first pass: upper entries in row order */
    for (int i = 0; i < n; i++) {
        urp[i] = (uint32_t)w;
        for (uint32_t k = rp[i]; k < rp[i + 1]; k++) {
            if ((int)ci[k] > i) {
                if (w >= nnz / 2 + 1) { xfree(cur); return 0; }
                uci[w] = ci[k];
                ucv[w] = cv[k];
                w++;
            }
        }
    }
    urp[n] = (uint32_t)w;
    if (2 * w != nnz) { xfree(cur); return 0; }
    /* second pass: every lower entry (i,j), j<i, must match upper entry (j,i) exactly,
       visited in increasing i for each j, i.e. in order of row j's upper list */
    for (int j = 0; j < n; j++) cur[j] = urp[j];
    for (int i = 0; i < n; i++) {
        for (uint32_t k = rp[i]; k < rp[i + 1]; k++) {
            int j = (int)ci[k];
            if (j >= i) break;
            uint32_t q = cur[j];
            if (q >= urp[j + 1] || (int)uci[q] != i || ucv[q] != cv[k]) { xfree(cur); return 0; }
            cur[j] = q + 1;
        }
    }
    xfree(cur);
    usv = ucv;
    zacc = xmalloc(sizeof(v2d) * (size_t)n);
    memset(zacc, 0, sizeof(v2d) * (size_t)n);
    return 1;
}

/* Same step with an extended-precision (x87 long double) recurrence. */
typedef struct { long double a, b; } ld2;
static long double ld_al2, ld_c;   /* 2/h and centre, extended precision */
static void cheb_step_ld(const ld2 *restrict u, ld2 *restrict v, long double *restrict y,
                         long double sigma, long double tau, long double c0, long double c1,
                         double *m0, double *m1, double *ym) {
    long double mx0 = 0, mx1 = 0, ymax = 0;
    for (int i = 0; i < n; i++) {
        long double s0 = 0, s1 = 0;
        uint32_t k0 = rp[i], k1 = rp[i + 1];
        if (uniform_off) {
            for (uint32_t k = k0; k < k1; k++) { s0 += u[ci[k]].a; s1 += u[ci[k]].b; }
            s0 *= (long double)off0; s1 *= (long double)off0;
        } else {
            for (uint32_t k = k0; k < k1; k++) {
                long double a = cv[k];
                s0 += a * u[ci[k]].a;
                s1 += a * u[ci[k]].b;
            }
        }
        long double di = (long double)dg[i] - ld_c;
        long double w0 = sigma * ld_al2 * (s0 + di * u[i].a) - tau * v[i].a;
        long double w1 = sigma * ld_al2 * (s1 + di * u[i].b) - tau * v[i].b;
        v[i].a = w0;
        v[i].b = w1;
        long double yi = y[i] + c0 * w0 + c1 * w1;
        y[i] = yi;
        if (fabsl(w0) > mx0) mx0 = fabsl(w0);
        if (fabsl(w1) > mx1) mx1 = fabsl(w1);
        if (fabsl(yi) > ymax) ymax = fabsl(yi);
    }
    *m0 = (double)mx0;
    *m1 = (double)mx1;
    *ym = (double)ymax;
}

/* ------------------------------------------------------------------------- */
/* Small dense symmetric eigensolver (cyclic Jacobi) for the Lanczos matrix.  */

static void jacobi_eig(int m, double *A, double *V, double *w) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) V[i * m + j] = (i == j) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0.0, tot = 0.0;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < m; j++) {
                double a = A[i * m + j] * A[i * m + j];
                tot += a;
                if (i != j) off += a;
            }
        if (off <= 1e-30 * tot || off == 0.0) break;
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

/* Safeguarded Lanczos estimate of an upper bound for lambda_max. */
static double lanczos_upper(int steps, double hiG) {
    if (steps > n) steps = n;
    double *v = xmalloc(sizeof(double) * n), *vp = xmalloc(sizeof(double) * n);
    double *w = xmalloc(sizeof(double) * n);
    double al[128], be[128];
    uint64_t s = 0x9E3779B97F4A7C15ULL;
    double nrm = 0.0;
    for (int i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        double r = ((double)(s >> 11) * (1.0 / 9007199254740992.0)) - 0.5;
        v[i] = r;
        nrm += r * r;
        vp[i] = 0.0;
    }
    nrm = 1.0 / sqrt(nrm);
    for (int i = 0; i < n; i++) v[i] *= nrm;
    double bprev = 0.0;
    int k;
    double bk = 0.0;
    for (k = 0; k < steps; k++) {
        spmv_plain(v, w);
        double a = 0.0;
        for (int i = 0; i < n; i++) { w[i] -= bprev * vp[i]; a += v[i] * w[i]; }
        double b = 0.0;
        for (int i = 0; i < n; i++) { w[i] -= a * v[i]; b += w[i] * w[i]; }
        b = sqrt(b);
        al[k] = a;
        be[k] = b;
        bk = b;
        if (!(b > 1e-12 * (fabs(a) + bprev + 1e-300))) { k++; break; }
        double ib = 1.0 / b;
        for (int i = 0; i < n; i++) { double t = v[i]; v[i] = w[i] * ib; vp[i] = t; }
        bprev = b;
    }
    int m = k;
    double *A = xmalloc(sizeof(double) * m * m), *V = xmalloc(sizeof(double) * m * m);
    double *ev = xmalloc(sizeof(double) * m);
    memset(A, 0, sizeof(double) * m * m);
    for (int i = 0; i < m; i++) {
        A[i * m + i] = al[i];
        if (i + 1 < m) { A[i * m + i + 1] = be[i]; A[(i + 1) * m + i] = be[i]; }
    }
    jacobi_eig(m, A, V, ev);
    int imax = 0;
    for (int i = 1; i < m; i++) if (ev[i] > ev[imax]) imax = i;
    double theta = ev[imax];
    double r = fabs(bk * V[(m - 1) * m + imax]);
    xfree(v); xfree(vp); xfree(w); xfree(A); xfree(V); xfree(ev);
    double up = (theta + 2.0 * r) * 1.004 + 1e-300;
#ifdef TEST_BADHI
    up *= TEST_BADHI;
#endif
    if (!(up > 0) || !isfinite(up)) return hiG;
    if (up > hiG) up = hiG;
    fprintf(stderr, "lanczos: theta=%.10g r=%.3g up=%.10g hiG=%.10g\n", theta, r, up, hiG);
    return up;
}

/* ------------------------------------------------------------------------- */
/* Chebyshev coefficients.                                                    */

static double T_;                       /* time */
static double f0(double x) { return exp(-T_ * x); }
static double f1(double x) {
    double z = T_ * x;
    if (fabs(z) < 1e-6) return T_ * (1.0 - 0.5 * z + z * z / 6.0);
    return -expm1(-z) / x;
}

/* Coefficients a_k (k<N) with f(c+h*s) ~ sum a_k T_k(s), via exact-table DCT. */
static void cheb_coeffs(int N, double c, double h, double *a0, double *a1,
                        double *fm0, double *fm1) {
    int M4 = 4 * N;
    double *tab = xmalloc(sizeof(double) * M4);
    for (int m = 0; m < M4; m++) tab[m] = cos(M_PI * (double)m / (2.0 * N));
    double *g0 = xmalloc(sizeof(double) * N), *g1 = xmalloc(sizeof(double) * N);
    double m0 = 0.0, m1 = 0.0;
    double lo = c - h;
    for (int j = 0; j < N; j++) {
        /* node c + h*cos(th_j) written as lo + 2h*sin^2((pi-th_j)/2): no cancellation */
        double sn = sin(M_PI * (double)(2 * (N - j) - 1) / (4.0 * N));
        double x = lo + 2.0 * h * sn * sn;
        if (x < 0) x = 0;
        g0[j] = f0(x);
        g1[j] = f1(x);
        if (fabs(g0[j]) > m0) m0 = fabs(g0[j]);
        if (fabs(g1[j]) > m1) m1 = fabs(g1[j]);
    }
    *fm0 = m0;
    *fm1 = m1;
    for (int k = 0; k < N; k++) {
        double s0 = 0.0, s1 = 0.0;
        int idx = k % M4, step = (2 * k) % M4;
        for (int j = 0; j < N; j++) {
            double tv = tab[idx];
            s0 += g0[j] * tv;
            s1 += g1[j] * tv;
            idx += step;
            if (idx >= M4) idx -= M4;
        }
        a0[k] = 2.0 * s0 / N;
        a1[k] = 2.0 * s1 / N;
    }
    a0[0] *= 0.5;
    a1[0] *= 0.5;
    xfree(tab); xfree(g0); xfree(g1);
}

/* ------------------------------------------------------------------------- */

static double lo_, hi_, hiG_, rtol_, atol_;
static double SAFE = 0.05;
static int keep_sets = 0;
struct coefset { double lo, hi; int K; double *A0, *A1, *S0, *S1, *W0, *W1; };
static void save_set(struct coefset *c);
static void load_set(const struct coefset *c);
static int K_;                  /* number of usable coefficients */
static double *A0, *A1, *S0, *S1; /* coefficients and suffix sums */
static double *W0, *W1;           /* prefix sums of k*|a_k|: rounding-growth indicator */

static void setup_interval(double lo, double hi) {
    lo_ = lo;
    hi_ = hi;
    double c = 0.5 * (hi + lo), h = 0.5 * (hi - lo);
    double a = T_ * h;
    int Kest = (int)(sqrt(2.0 * a * 46.0) + 0.5 * 46.0 + 16.0);
    int N = (int)(1.25 * Kest) + 16;
    int tries = 0;
    for (;;) {
        if (!keep_sets) { xfree(A0); xfree(A1); xfree(S0); xfree(S1); xfree(W0); xfree(W1); }
        W0 = xmalloc(sizeof(double) * (N + 1));
        W1 = xmalloc(sizeof(double) * (N + 1));
        A0 = xmalloc(sizeof(double) * (N + 1));
        A1 = xmalloc(sizeof(double) * (N + 1));
        S0 = xmalloc(sizeof(double) * (N + 1));
        S1 = xmalloc(sizeof(double) * (N + 1));
        double mx0, mx1, tl0 = 0, tl1 = 0;
        cheb_coeffs(N, c, h, A0, A1, &mx0, &mx1);
        int tail_start = (int)(0.8 * N);
        for (int k = tail_start; k < N; k++) {
            if (fabs(A0[k]) > tl0) tl0 = fabs(A0[k]);
            if (fabs(A1[k]) > tl1) tl1 = fabs(A1[k]);
        }
        fprintf(stderr, "coef N=%d tl0=%.3e mx0=%.3e tl1=%.3e mx1=%.3e\n", N, tl0, mx0, tl1, mx1);
        if ((tl0 <= 1e-15 * mx0 && tl1 <= 1e-15 * mx1) || ++tries >= 3) {
            K_ = tail_start;
            break;
        }
        N *= 2;
    }
    /* suffix sums: S[k] = sum_{j>k} |a_j| (over the usable range, plus the rest) */
    double s0 = 0, s1 = 0;
    S0[K_] = 0; S1[K_] = 0;
    for (int k = K_ - 1; k >= 0; k--) {
        S0[k] = s0; S1[k] = s1;   /* sum over j > k, j < K_ */
        s0 += fabs(A0[k]);
        s1 += fabs(A1[k]);
    }
    double w0 = 0, w1 = 0;
    for (int k = 0; k < K_; k++) {
        w0 += k * fabs(A0[k]);
        w1 += k * fabs(A1[k]);
        W0[k] = w0;
        W1[k] = w1;
    }
    fprintf(stderr, "interval [%.6g, %.6g] t*h=%.6g K=%d\n", lo, hi, a, K_);
}

/* Set the scaled operator 2X = 2*(L - c I)/h for the current interval. */
static void setup_operator(void) {
    double c = 0.5 * (hi_ + lo_), h = 0.5 * (hi_ - lo_);
    double al = 1.0 / h;
    scale2 = 2.0 * al;
    sval = 2.0 * al * off0;
    for (int i = 0; i < n; i++) dd[i] = 2.0 * al * (dg[i] - c);
    ld_c = (long double)c;
    ld_al2 = 2.0L / (long double)h;
}

static void save_set(struct coefset *c) {
    c->lo = lo_; c->hi = hi_; c->K = K_;
    c->A0 = A0; c->A1 = A1; c->S0 = S0; c->S1 = S1; c->W0 = W0; c->W1 = W1;
}
static void load_set(const struct coefset *c) {
    lo_ = c->lo; hi_ = c->hi; K_ = c->K;
    A0 = c->A0; A1 = c->A1; S0 = c->S0; S1 = c->S1; W0 = c->W0; W1 = c->W1;
    setup_operator();
}

int main(void) {
    /* Flush-to-zero / denormals-are-zero: subnormal operands (e.g. coefficients
       scaled by exp(-t*lo) near the underflow limit) would otherwise be very slow. */
    _mm_setcsr(_mm_getcsr() | 0x8040);
    /* READY before any input; generic init only. */
    arena_cap = (size_t)112 << 20;
    arena = mmap(NULL, arena_cap, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (arena == MAP_FAILED) arena = NULL;
    else {
        (void)madvise(arena, arena_cap, MADV_HUGEPAGE);
        memset(arena, 0, arena_cap);
    }
    write_full("READY\n", 6);

    char magic[8];
    read_full(magic, 8);
    T_start = now_s();
    if (memcmp(magic, "FAMEXP2\n", 8) != 0) die("bad magic");
    (void)fcntl(0, F_SETPIPE_SZ, 1 << 20);
    uint32_t jl;
    read_full(&jl, 4);
    if (jl == 0 || jl > 65536) die("json length");
    char *js = xmalloc(jl + 1);
    read_full(js, jl);
    js[jl] = 0;
    n = (int)json_num(js, "n");
    size_t nnz = (size_t)json_num(js, "nnz");
    int requests = (int)json_num(js, "requests");
    T_ = json_num(js, "t");
    rtol_ = json_num(js, "rtol");
    atol_ = json_num(js, "atol");
    if (n < 1 || requests < 1) die("header values");

    uint64_t *ip = xmalloc(sizeof(uint64_t) * (n + 1));
    ci = xmalloc(sizeof(uint32_t) * (nnz + 1));
    cv = xmalloc(sizeof(double) * (nnz + 1));
    read_full(ip, sizeof(uint64_t) * (size_t)(n + 1));
    read_full(ci, sizeof(uint32_t) * nnz);
    read_full(cv, sizeof(double) * nnz);
    TLOG("matrix read");

    /* Split diagonal, compact off-diagonals in place, Gershgorin bounds. */
    rp = xmalloc(sizeof(uint32_t) * (n + 1));
    dg = xmalloc(sizeof(double) * n);
    double loG = INFINITY, hiG = -INFINITY;
    size_t wpos = 0;
    double sumdist = 0.0;
    int have_off = 0;
    uniform_off = 1;
    for (int i = 0; i < n; i++) {
        uint64_t k0 = ip[i], k1 = ip[i + 1];
        double d = 0.0, R = 0.0;
        rp[i] = (uint32_t)wpos;
        for (uint64_t k = k0; k < k1; k++) {
            uint32_t j = ci[k];
            double a = cv[k];
            if ((int)j == i) { d += a; continue; }
            if (a == 0.0) continue;
            R += fabs(a);
            if (!have_off) { off0 = a; have_off = 1; }
            else if (a != off0) uniform_off = 0;
            ci[wpos] = j;
            cv[wpos] = a;
            wpos++;
            sumdist += (double)((int)j > i ? (int)j - i : i - (int)j);
        }
        dg[i] = d;
        if (d - R < loG) loG = d - R;
        if (d + R > hiG) hiG = d + R;
    }
    rp[n] = (uint32_t)wpos;
    xfree(ip);
    double mean_dist = wpos ? sumdist / (double)wpos : 0.0;
    int *perm = NULL, *iperm = NULL, reordered = 0;
    if (!have_off) uniform_off = 1, off0 = 0.0;
    if (loG < 0) loG = 0;
    if (hiG < loG) hiG = loG;
    hiG_ = hiG;
    fprintf(stderr, "n=%d nnz=%zu offnnz=%zu t=%.6g rtol=%.1e uniform=%d loG=%.6g hiG=%.6g\n",
            n, nnz, wpos, T_, rtol_, uniform_off, loG, hiG);

    TLOG("preprocessed");
    double *bin = xmalloc(sizeof(double) * 2 * (size_t)n);
    double *y = xmalloc(sizeof(double) * (size_t)n);
    double *pbin = xmalloc(sizeof(double) * 2 * (size_t)n);
    double *yout = xmalloc(sizeof(double) * (size_t)n);
    v2d *u = xmalloc(sizeof(v2d) * (size_t)n);
    v2d *v = xmalloc(sizeof(v2d) * (size_t)n);
    int force_ld = 0;
    double *yfirst = xmalloc(sizeof(double) * (size_t)n);
    double yfirst_max = 0.0;
    struct coefset prim, alt;
    int have_alt = 0, second = 0;
    ld2 *ul = xmalloc(sizeof(ld2) * (size_t)n), *vl = xmalloc(sizeof(ld2) * (size_t)n);
    long double *yl = xmalloc(sizeof(long double) * (size_t)n);

    /* Trivial operators: t == 0, or L == c*I (Gershgorin interval is a point). */
    int trivial = (T_ == 0.0) || (hiG - loG <= 0.0) || (T_ * (hiG - loG) == 0.0);
    if (!trivial) {
        dd = xmalloc(sizeof(double) * n);
        double hi = hiG;
        double aG = T_ * 0.5 * (hiG - loG);
        int KG = (int)(sqrt(2.0 * aG * 25.0) + 10.0);
        use_sym = 0;
        if ((double)KG * requests >= 150.0 && wpos > 0 && mean_dist > 1000.0) {
            perm = xmalloc(sizeof(int) * n);
            iperm = xmalloc(sizeof(int) * n);
            if (rcm_order(perm)) { apply_perm(perm, iperm); reordered = 1; }
            TLOG("rcm");
        }
        if ((double)KG * requests >= 60.0 && wpos > 0) use_sym = build_upper();
        fprintf(stderr, "KG=%d use_sym=%d\n", KG, use_sym);
        if ((double)KG * requests > 350.0 && n > 60) hi = lanczos_upper(50, hiG);
        if (hi <= loG) hi = hiG;
        setup_interval(loG, hi);
        setup_operator();
        TLOG("setup done");
    }

    for (int r = 0; r < requests; r++) {
        read_full(bin, sizeof(double) * 2 * (size_t)n);
        TLOG("rhs read");
        const double *b0 = bin, *b1 = bin + n;
        if (reordered) {
            for (int i = 0; i < n; i++) { pbin[i] = bin[perm[i]]; pbin[n + i] = bin[n + perm[i]]; }
            b0 = pbin;
            b1 = pbin + n;
        }
        if (trivial) {
            if (T_ == 0.0) {
                memcpy(y, b0, sizeof(double) * n);
            } else {
                double c = 0.5 * (hiG + loG);
                double e = f0(c), g = f1(c);
                for (int i = 0; i < n; i++) y[i] = e * b0[i] + g * b1[i];
            }
            write_full(y, sizeof(double) * n);
            continue;
        }
        int use_ld = force_ld;
        double t_req0 = now_s() - T_start;
    again:
        if (use_ld) {
            restart_ld:;
            double M0 = 0.0, M1 = 0.0, ym = 0.0;
            for (int i = 0; i < n; i++) {
                ul[i].a = b0[i]; ul[i].b = b1[i];
                vl[i].a = 0; vl[i].b = 0;
                double a = fabs(b0[i]), b = fabs(b1[i]);
                if (a > M0) M0 = a;
                if (b > M1) M1 = b;
                yl[i] = (long double)A0[0] * b0[i] + (long double)A1[0] * b1[i];
                if (fabsl(yl[i]) > ym) ym = (double)fabsl(yl[i]);
            }
            double Minit0 = M0, Minit1 = M1;
            int k, blew = 0;
            for (k = 1; k < K_; k++) {
                double m0, m1, yk;
                if (k == 1) cheb_step_ld(ul, vl, yl, 0.5L, 0.0L, A0[1], A1[1], &m0, &m1, &yk);
                else cheb_step_ld(ul, vl, yl, 1.0L, 1.0L, A0[k], A1[k], &m0, &m1, &yk);
                ld2 *tmp = ul; ul = vl; vl = tmp;
                if (m0 > M0) M0 = m0;
                if (m1 > M1) M1 = m1;
                if (k <= 3) { if (M0 > Minit0) Minit0 = M0; if (M1 > Minit1) Minit1 = M1; }
                ym = yk;
                double E = 2.0 * (S0[k] * M0 + S1[k] * M1);
                if (E <= SAFE * (atol_ + rtol_ * ym)) break;
                if ((M0 > 1e4 * Minit0 && M0 > 0) || (M1 > 1e4 * Minit1 && M1 > 0) || !isfinite(ym)) { blew = 1; break; }
            }
            if (blew && hi_ < hiG_) {
                fprintf(stderr, "divergence detected at k=%d; falling back to Gershgorin\n", k);
                setup_interval(lo_, hiG_);
                setup_operator();
                if (keep_sets) { save_set(&prim); have_alt = 0; }
                goto restart_ld;
            }
            for (int i = 0; i < n; i++) y[i] = (double)yl[i];
            fprintf(stderr, "req %d: ld steps=%d ymax=%.6g\n", r, k, ym);
            if (reordered) { for (int i = 0; i < n; i++) yout[perm[i]] = y[i]; write_full(yout, sizeof(double) * n); }
            else write_full(y, sizeof(double) * n);
            continue;
        }
    restart:;
        double M0 = 0.0, M1 = 0.0, ym = 0.0;
        for (int i = 0; i < n; i++) {
            u[i] = (v2d){b0[i], b1[i]};
            v[i] = (v2d){0.0, 0.0};
            double a = fabs(b0[i]), b = fabs(b1[i]);
            if (a > M0) M0 = a;
            if (b > M1) M1 = b;
            y[i] = A0[0] * b0[i] + A1[0] * b1[i];
            if (fabs(y[i]) > ym) ym = fabs(y[i]);
        }
        double Minit0 = M0, Minit1 = M1;
        int k;
        int blew = 0;
        for (k = 1; k < K_; k++) {
            double m0, m1, yk;
            if (use_sym) {
                if (k == 1) cheb_step_sym(u, v, y, 0.5, 0.0, A0[1], A1[1], &m0, &m1, &yk);
                else cheb_step_sym(u, v, y, 1.0, 1.0, A0[k], A1[k], &m0, &m1, &yk);
            } else {
                if (k == 1) cheb_step(u, v, y, 0.5, 0.0, A0[1], A1[1], &m0, &m1, &yk);
                else cheb_step(u, v, y, 1.0, 1.0, A0[k], A1[k], &m0, &m1, &yk);
            }
            v2d *tmp = u; u = v; v = tmp;
            if (m0 > M0) M0 = m0;
            if (m1 > M1) M1 = m1;
            if (k <= 3) { if (M0 > Minit0) Minit0 = M0; if (M1 > Minit1) Minit1 = M1; }
            ym = yk;
            double E = 2.0 * (S0[k] * M0 + S1[k] * M1);
            double tol = atol_ + rtol_ * ym;
            if (E <= SAFE * tol) break;
            if ((M0 > 1e4 * Minit0 && M0 > 0) || (M1 > 1e4 * Minit1 && M1 > 0) || !isfinite(ym)) {
                blew = 1;
                break;
            }
        }
        if (blew && hi_ < hiG_) {
            fprintf(stderr, "divergence detected at k=%d; falling back to Gershgorin\n", k);
            setup_interval(lo_, hiG_);
            setup_operator();
            if (keep_sets) { save_set(&prim); have_alt = 0; }
            if (second) {
                /* the second opinion must not repeat the first computation */
                setup_interval(lo_, hi_ + 1e-5 * (hi_ - lo_));
                save_set(&alt);
                have_alt = 1;
                setup_operator();
            }
            goto restart;
        }
        {
            /* Rounding-error indicator of the double recurrence: errors committed
               at step j are amplified at most linearly in the remaining degree. */
            int kk = k < K_ ? k : K_ - 1;
            double risk = 1.1e-16 * (W0[kk] * M0 + W1[kk] * M1) / (atol_ + rtol_ * ym);
            fprintf(stderr, "req %d: steps=%d ymax=%.6g risk=%.3g\n", r, k, ym, risk);
            if (blew) {
                use_ld = 1;
                goto again;
            }
            if (risk > 0.01 && !second) {
                /* second opinion: same series on a slightly wider interval, i.e.
                   independent rounding; accept only if both agree closely. */
                memcpy(yfirst, y, sizeof(double) * n);
                yfirst_max = ym;
                if (!have_alt) {
                    keep_sets = 1;
                    save_set(&prim);
                    setup_interval(lo_, hi_ + 1e-5 * (hi_ - lo_));
                    save_set(&alt);
                    have_alt = 1;
                } else {
                    load_set(&alt);
                }
                setup_operator();
                second = 1;
                goto restart;
            }
            if (second) {
                second = 0;
                load_set(&prim);
                double dmax = 0.0;
                for (int i = 0; i < n; i++) { double d = fabs(y[i] - yfirst[i]); if (d > dmax) dmax = d; }
                double tl = atol_ + rtol_ * (ym < yfirst_max ? ym : yfirst_max);
                fprintf(stderr, "req %d: second opinion diff=%.3e (%.3g tol)\n", r, dmax, dmax / tl);
                if (!(dmax <= 0.05 * tl)) {
                    /* time guard: extended precision costs ~5 double sweeps */
                    double tnow = now_s() - T_start;
                    double tdbl = (tnow - t_req0) / 2.0;
                    int left = requests - r - 1;
                    double limit = requests == 1 ? 60.0 : 240.0;
                    if (tnow + 6.0 * tdbl + left * 2.5 * tdbl < 0.8 * limit) {
                        use_ld = 1;
                        goto again;
                    }
                    fprintf(stderr, "req %d: time guard, averaging double results\n", r);
                    for (int i = 0; i < n; i++) y[i] = 0.5 * (y[i] + yfirst[i]);
                } else {
                    memcpy(y, yfirst, sizeof(double) * n);
                }
            }
        }
        if (reordered) { for (int i = 0; i < n; i++) yout[perm[i]] = y[i]; write_full(yout, sizeof(double) * n); }
        else write_full(y, sizeof(double) * n);
        TLOG("written");
    }
    _exit(0);
}
