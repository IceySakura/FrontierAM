/* FAMEXP2 solver: y = exp(-tL) b0 + t*phi1(-tL) b1, L sparse symmetric PSD.
 *
 * Standalone program speaking the FAMEXP2 protocol on stdin/stdout.
 * Paths: trivial (t=0 or L=0), small connected components (dense
 * eigendecompositions, includes diagonal and tiny matrices), and the general
 * Lanczos path (two chains sharing a fused SpMV, residual-based stopping).
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
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <dlfcn.h>

/* ------------------------------------------------------------------ */
/* LAPACK / BLAS                                                       */
/* ------------------------------------------------------------------ */
extern void dstevd_(const char *jobz, const int *n, double *d, double *e, double *z,
                    const int *ldz, double *work, const int *lwork, int *iwork,
                    const int *liwork, int *info);
extern void dstemr_(const char *jobz, const char *range, const int *n, double *d, double *e,
                    const double *vl, const double *vu, const int *il, const int *iu, int *m,
                    double *w, double *z, const int *ldz, const int *nzc, int *isuppz,
                    int *tryrac, double *work, const int *lwork, int *iwork, const int *liwork,
                    int *info);
extern void dsyevd_(const char *jobz, const char *uplo, const int *n, double *a,
                    const int *lda, double *w, double *work, const int *lwork,
                    int *iwork, const int *liwork, int *info);
extern void dgemv_(const char *trans, const int *m, const int *n, const double *alpha,
                   const double *a, const int *lda, const double *x, const int *incx,
                   const double *beta, double *y, const int *incy);

static int DBG = 0;
#define LOG(...) do { if (DBG) { fprintf(stderr, __VA_ARGS__); } } while (0)

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* ------------------------------------------------------------------ */
/* I/O                                                                 */
/* ------------------------------------------------------------------ */
static void read_exact(void *buf, size_t len)
{
    char *p = (char *)buf;
    while (len) {
        size_t chunk = len > (1u << 30) ? (1u << 30) : len;
        ssize_t r = read(0, p, chunk);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) _exit(3);
        p += r;
        len -= (size_t)r;
    }
}

static void write_exact(const void *buf, size_t len)
{
    const char *p = (const char *)buf;
    while (len) {
        ssize_t w = write(1, p, len);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) _exit(4);
        p += w;
        len -= (size_t)w;
    }
}

static void *xmalloc(size_t sz)
{
    void *p = malloc(sz ? sz : 1);
    if (!p) _exit(5);
    return p;
}

static void *xcalloc(size_t n, size_t sz)
{
    void *p = calloc(n ? n : 1, sz ? sz : 1);
    if (!p) _exit(5);
    return p;
}

/* Pre-faulted pool (filled before READY), then fresh mappings. */
static char *POOL = NULL;
static size_t POOL_SIZE = 0, POOL_USED = 0;

static void pool_init(size_t sz)
{
    void *p = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return;
#ifdef MADV_HUGEPAGE
    madvise(p, sz, MADV_HUGEPAGE);
#endif
    memset(p, 0, sz);
    POOL = (char *)p;
    POOL_SIZE = sz;
}

static void *pool_take(size_t sz)
{
    sz = (sz + 4095) & ~(size_t)4095;
    if (POOL && POOL_USED + sz <= POOL_SIZE) {
        void *p = POOL + POOL_USED;
        POOL_USED += sz;
        return p;
    }
    return NULL;
}

/* Large buffer with transparent huge pages. */
static void *big_alloc(size_t sz)
{
    void *q = pool_take(sz);
    if (q) return q;
    void *p = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
#ifdef MADV_HUGEPAGE
    madvise(p, sz, MADV_HUGEPAGE);
#endif
    return p;
}

static void big_free(void *p, size_t sz)
{
    if (p && !((char *)p >= POOL && (char *)p < POOL + POOL_SIZE)) munmap(p, sz);
}

static double json_num(const char *js, const char *key, int *ok)
{
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = js;
    size_t kl = strlen(pat);
    while ((p = strstr(p, pat)) != NULL) {
        const char *q = p + kl;
        while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
        if (*q == ':') {
            q++;
            while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
            char *end;
            double v = strtod(q, &end);
            if (end != q) { *ok = 1; return v; }
        }
        p += kl;
    }
    *ok = 0;
    return 0.0;
}

/* ------------------------------------------------------------------ */
/* scalar functions                                                    */
/* ------------------------------------------------------------------ */
static inline double fexp(double th, double t) { return exp(-t * th); }

/* t*phi1(-t*th) = (1 - exp(-t th)) / th */
static inline double fphi(double th, double t)
{
    double x = t * th;
    if (fabs(x) < 1e-4) return t * (1.0 - x * (0.5 - x * (1.0 / 6.0 - x / 24.0)));
    return t * (-expm1(-x) / x);
}

/* both functions from a single exponential */
static inline void fboth(double th, double t, double *fe, double *fp)
{
    double x = t * th;
    double e = exp(-x);
    *fe = e;
    if (fabs(x) < 1e-3)
        *fp = t * (1.0 - x * (0.5 - x * (1.0 / 6.0 - x * (1.0 / 24.0 - x / 120.0))));
    else
        *fp = (1.0 - e) / th;
}

/* ------------------------------------------------------------------ */
/* sparse kernels                                                      */
/* ------------------------------------------------------------------ */
typedef double v2d __attribute__((vector_size(16)));

/* rotation log of the tridiagonal QL iteration */
typedef struct {
    int32_t *idx;
    double *cs;      /* pairs (c, s) */
    int64_t cnt, cap;
} RotLog;


typedef struct {
    int64_t n;
    const uint64_t *ip;
    const uint32_t *ix;
    const double *a;
} Csr;

static void spmv1(const Csr *A, const double *x, double *y)
{
    const uint64_t *ip = A->ip;
    const uint32_t *ix = A->ix;
    const double *a = A->a;
    for (int64_t i = 0; i < A->n; i++) {
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

/* One Lanczos step for c (1 or 2) interleaved chains; see expo.c for the
 * layout.  dots=1: compute alpha/beta (beta <= brk => chain stopped, vn=0).
 * dots=0: replay with stored alpha/beta (bitwise identical vectors). */
static void lanczos_step(const Csr *A, int c, const double *vp, const double *vc,
                         double *vn, const double *bprev, double *alpha, double *beta,
                         double *vmax, int dots, double brk)
{
    const int64_t n = A->n;
    if (c == 1) {
        double bp = bprev[0], al = 0.0;
        spmv1(A, vc, vn);
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
        if (dots) {
            b = sqrt(nr);
            if (!(b > brk)) b = 0.0;
            beta[0] = b;
        } else b = beta[0];
        if (b > 0.0) {
            double inv = 1.0 / b, mx = 0.0;
            for (int64_t i = 0; i < n; i++) {
                double w = vn[i] * inv;
                vn[i] = w;
                double aw = fabs(w);
                mx = aw > mx ? aw : mx;
            }
            if (dots) vmax[0] = mx;
        } else {
            memset(vn, 0, sizeof(double) * n);
            if (dots) vmax[0] = 0.0;
        }
        return;
    }
    const uint64_t *ip = A->ip;
    const uint32_t *ix = A->ix;
    const double *a = A->a;
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
        if (k < e) {
            v2d a0 = {a[k], a[k]};
            s0 += a0 * x2[ix[k]];
        }
        v2d w = (s0 + s1) - bp * p2[i];
        y2[i] = w;
        if (dots) al += w * x2[i];
    }
    if (dots) {
        alpha[0] = al[0];
        alpha[1] = al[1];
    } else {
        al[0] = alpha[0];
        al[1] = alpha[1];
    }
    v2d nr = {0.0, 0.0};
    for (int64_t i = 0; i < n; i++) {
        v2d w = y2[i] - al * x2[i];
        y2[i] = w;
        nr += w * w;
    }
    double b0, b1;
    if (dots) {
        b0 = sqrt(nr[0]);
        b1 = sqrt(nr[1]);
        if (!(b0 > brk)) b0 = 0.0;
        if (!(b1 > brk)) b1 = 0.0;
        beta[0] = b0;
        beta[1] = b1;
    } else {
        b0 = beta[0];
        b1 = beta[1];
    }
    v2d inv = {b0 > 0.0 ? 1.0 / b0 : 0.0, b1 > 0.0 ? 1.0 / b1 : 0.0};
    double m0 = 0.0, m1 = 0.0;
    for (int64_t i = 0; i < n; i++) {
        v2d w = y2[i] * inv;
        y2[i] = w;
        double a0 = fabs(w[0]), a1 = fabs(w[1]);
        m0 = a0 > m0 ? a0 : m0;
        m1 = a1 > m1 ? a1 : m1;
    }
    if (dots) {
        vmax[0] = m0;
        vmax[1] = m1;
    }
}

/* Upper-triangular copy (strictly upper entries + separate diagonal). */
typedef struct {
    int64_t n;
    uint32_t *ip;
    uint32_t *ix;
    double *a;
    double *d;
} Sym;

static void sym_build(const Csr *A, Sym *S)
{
    int64_t n = A->n;
    S->n = n;
    S->ip = xmalloc(sizeof(uint32_t) * (n + 1));
    S->d = xcalloc(n, sizeof(double));
    uint64_t cnt = 0;
    for (int64_t i = 0; i < n; i++) {
        S->ip[i] = (uint32_t)cnt;
        for (uint64_t k = A->ip[i]; k < A->ip[i + 1]; k++)
            if ((int64_t)A->ix[k] > i) cnt++;
    }
    S->ip[n] = (uint32_t)cnt;
    S->ix = xmalloc(sizeof(uint32_t) * (cnt ? cnt : 1));
    S->a = xmalloc(sizeof(double) * (cnt ? cnt : 1));
    uint64_t q = 0;
    for (int64_t i = 0; i < n; i++)
        for (uint64_t k = A->ip[i]; k < A->ip[i + 1]; k++) {
            int64_t j = A->ix[k];
            if (j > i) {
                S->ix[q] = (uint32_t)j;
                S->a[q] = A->a[k];
                q++;
            } else if (j == i) S->d[i] += A->a[k];
        }
}

/* Lanczos step with the symmetric kernel; same contract as lanczos_step. */
static void lanczos_step_sym(const Sym *M, int c, const double *vp, const double *vc,
                             double *vn, const double *bprev, double *alpha, double *beta,
                             double *vmax, int dots, double brk)
{
    const int64_t n = M->n;
    const uint32_t *ip = M->ip;
    const uint32_t *ix = M->ix;
    const double *a = M->a, *d = M->d;
    if (c == 1) {
        double bp = bprev[0];
        for (int64_t i = 0; i < n; i++) vn[i] = -bp * vp[i];
        double al = 0.0, yy = 0.0, vv = 0.0;
        for (int64_t i = 0; i < n; i++) {
            double xi = vc[i], s0 = d[i] * xi, s1 = 0.0;
            vv += xi * xi;
            uint64_t k = ip[i], e = ip[i + 1];
            for (; k + 2 <= e; k += 2) {
                uint32_t j0 = ix[k], j1 = ix[k + 1];
                double a0 = a[k], a1 = a[k + 1];
                s0 += a0 * vc[j0];
                s1 += a1 * vc[j1];
                vn[j0] += a0 * xi;
                vn[j1] += a1 * xi;
            }
            if (k < e) {
                uint32_t j0 = ix[k];
                s0 += a[k] * vc[j0];
                vn[j0] += a[k] * xi;
            }
            double w = vn[i] + (s0 + s1);
            vn[i] = w;
            al += w * xi;
            yy += w * w;
        }
        double b;
        if (dots) {
            alpha[0] = al;
            double b2 = yy - al * al * (2.0 - vv);
            if (b2 > 1e-4 * yy) b = sqrt(b2);
            else {
                double nr = 0.0;
                for (int64_t i = 0; i < n; i++) {
                    double w = vn[i] - al * vc[i];
                    nr += w * w;
                }
                b = sqrt(nr);
            }
            if (!(b > brk)) b = 0.0;
            beta[0] = b;
        } else {
            al = alpha[0];
            b = beta[0];
        }
        if (b > 0.0) {
            double inv = 1.0 / b, mx = 0.0;
            for (int64_t i = 0; i < n; i++) {
                double w = (vn[i] - al * vc[i]) * inv;
                vn[i] = w;
                double aw = fabs(w);
                mx = aw > mx ? aw : mx;
            }
            if (dots) vmax[0] = mx;
        } else {
            memset(vn, 0, sizeof(double) * n);
            if (dots) vmax[0] = 0.0;
        }
        return;
    }
    const v2d *x2 = (const v2d *)vc;
    const v2d *p2 = (const v2d *)vp;
    v2d *y2 = (v2d *)vn;
    v2d bp = {bprev[0], bprev[1]};
    for (int64_t i = 0; i < n; i++) y2[i] = -(bp * p2[i]);
    v2d al = {0.0, 0.0}, yy = {0.0, 0.0}, vv = {0.0, 0.0};
    for (int64_t i = 0; i < n; i++) {
        v2d xi = x2[i];
        vv += xi * xi;
        v2d di = {d[i], d[i]};
        v2d s0 = di * xi, s1 = {0.0, 0.0};
        uint64_t k = ip[i], e = ip[i + 1];
        for (; k + 2 <= e; k += 2) {
            uint32_t j0 = ix[k], j1 = ix[k + 1];
            v2d a0 = {a[k], a[k]}, a1 = {a[k + 1], a[k + 1]};
            s0 += a0 * x2[j0];
            s1 += a1 * x2[j1];
            y2[j0] += a0 * xi;
            y2[j1] += a1 * xi;
        }
        if (k < e) {
            uint32_t j0 = ix[k];
            v2d a0 = {a[k], a[k]};
            s0 += a0 * x2[j0];
            y2[j0] += a0 * xi;
        }
        v2d w = y2[i] + (s0 + s1);
        y2[i] = w;
        al += w * xi;
        yy += w * w;
    }
    double bb[2];
    v2d alv;
    if (dots) {
        alpha[0] = al[0];
        alpha[1] = al[1];
        int need = 0;
        for (int j = 0; j < 2; j++) {
            double b2 = yy[j] - al[j] * al[j] * (2.0 - vv[j]);
            if (b2 > 1e-4 * yy[j]) bb[j] = sqrt(b2);
            else { bb[j] = -1.0; need = 1; }
        }
        if (need) {
            v2d nr = {0.0, 0.0};
            for (int64_t i = 0; i < n; i++) {
                v2d w = y2[i] - al * x2[i];
                nr += w * w;
            }
            for (int j = 0; j < 2; j++) if (bb[j] < 0.0) bb[j] = sqrt(nr[j]);
        }
        for (int j = 0; j < 2; j++) {
            if (!(bb[j] > brk)) bb[j] = 0.0;
            beta[j] = bb[j];
        }
        alv = al;
    } else {
        alv[0] = alpha[0];
        alv[1] = alpha[1];
        bb[0] = beta[0];
        bb[1] = beta[1];
    }
    v2d inv = {bb[0] > 0.0 ? 1.0 / bb[0] : 0.0, bb[1] > 0.0 ? 1.0 / bb[1] : 0.0};
    double m0 = 0.0, m1 = 0.0;
    for (int64_t i = 0; i < n; i++) {
        v2d w = (y2[i] - alv * x2[i]) * inv;
        y2[i] = w;
        double a0 = fabs(w[0]), a1 = fabs(w[1]);
        m0 = a0 > m0 ? a0 : m0;
        m1 = a1 > m1 ? a1 : m1;
    }
    if (dots) {
        vmax[0] = m0;
        vmax[1] = m1;
    }
}

/* Reverse Cuthill-McKee style ordering (BFS levels from a minimum-degree
 * start node per component, neighbours visited in degree order for small
 * degree lists).  perm[new] = old. */
static void rcm_order(const Csr *A, int64_t *perm)
{
    const int64_t n = A->n;
    int64_t *deg = xmalloc(sizeof(int64_t) * n);
    unsigned char *vis = xcalloc(n, 1);
    for (int64_t i = 0; i < n; i++) deg[i] = (int64_t)(A->ip[i + 1] - A->ip[i]);
    /* nodes sorted by degree (counting sort) to pick start nodes */
    int64_t maxd = 0;
    for (int64_t i = 0; i < n; i++) if (deg[i] > maxd) maxd = deg[i];
    int64_t *cnt = xcalloc(maxd + 2, sizeof(int64_t));
    for (int64_t i = 0; i < n; i++) cnt[deg[i] + 1]++;
    for (int64_t d = 0; d <= maxd; d++) cnt[d + 1] += cnt[d];
    int64_t *byd = xmalloc(sizeof(int64_t) * n);
    for (int64_t i = 0; i < n; i++) byd[cnt[deg[i]]++] = i;
    int64_t top = 0;
    int64_t tmp[64];
    for (int64_t q = 0; q < n; q++) {
        int64_t s0 = byd[q];
        if (vis[s0]) continue;
        vis[s0] = 1;
        int64_t base = top;
        perm[top++] = s0;
        for (int64_t h = base; h < top; h++) {
            int64_t i = perm[h];
            int64_t nt = 0, first = top;
            for (uint64_t k = A->ip[i]; k < A->ip[i + 1]; k++) {
                int64_t j = A->ix[k];
                if (!vis[j]) {
                    vis[j] = 1;
                    if (nt < 64) tmp[nt++] = j;
                    else perm[top++] = j;
                }
            }
            /* insertion sort of the first (up to 64) new neighbours by degree */
            for (int64_t x = 1; x < nt; x++) {
                int64_t v = tmp[x], y = x - 1;
                while (y >= 0 && deg[tmp[y]] > deg[v]) { tmp[y + 1] = tmp[y]; y--; }
                tmp[y + 1] = v;
            }
            if (nt) {
                memmove(perm + first + nt, perm + first, sizeof(int64_t) * (top - first));
                memcpy(perm + first, tmp, sizeof(int64_t) * nt);
                top += nt;
            }
        }
    }
    /* reverse */
    for (int64_t a2 = 0, b2 = n - 1; a2 < b2; a2++, b2--) {
        int64_t v = perm[a2];
        perm[a2] = perm[b2];
        perm[b2] = v;
    }
    free(deg); free(vis); free(cnt); free(byd);
}

/* B = P A P^T with perm[new] = old */
static void csr_permute(const Csr *A, const int64_t *perm, Csr *B)
{
    const int64_t n = A->n;
    int64_t *inv = xmalloc(sizeof(int64_t) * n);
    for (int64_t r = 0; r < n; r++) inv[perm[r]] = r;
    uint64_t *ip = big_alloc(sizeof(uint64_t) * (n + 1));
    uint64_t nnz = A->ip[n];
    uint32_t *ix = big_alloc(sizeof(uint32_t) * (nnz ? nnz : 1));
    double *a = big_alloc(sizeof(double) * (nnz ? nnz : 1));
    if (!ip || !ix || !a) _exit(5);
    uint64_t kk = 0;
    for (int64_t r = 0; r < n; r++) {
        int64_t i = perm[r];
        ip[r] = kk;
        for (uint64_t k = A->ip[i]; k < A->ip[i + 1]; k++) {
            ix[kk] = (uint32_t)inv[A->ix[k]];
            a[kk] = A->a[k];
            kk++;
        }
    }
    ip[n] = kk;
    B->n = n;
    B->ip = ip;
    B->ix = ix;
    B->a = a;
    free(inv);
}

/* Fast path: every row has at most one off-diagonal entry (singletons and
 * symmetric pairs).  Per node: y_i = fd b0_i + gd b1_i + fo b0_p + go b1_p. */
typedef struct {
    int64_t n;
    int64_t *part;
    double *fd, *fo, *gd, *go;
} Pairs;

static int pairs_setup(const Csr *A, double t, Pairs *P)
{
    const int64_t n = A->n;
    int64_t *part = big_alloc(sizeof(int64_t) * n);
    double *dg = big_alloc(sizeof(double) * n), *of = big_alloc(sizeof(double) * n);
    if (!part || !dg || !of) return 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t pj = -1;
        double d = 0.0, o = 0.0;
        for (uint64_t k = A->ip[i]; k < A->ip[i + 1]; k++) {
            int64_t j = A->ix[k];
            if (j == i) d += A->a[k];
            else {
                if (pj >= 0 && pj != j) return 0;
                pj = j;
                o = A->a[k];
            }
        }
        part[i] = pj;
        dg[i] = d;
        of[i] = o;
    }
    for (int64_t i = 0; i < n; i++)
        if (part[i] >= 0 && part[part[i]] != i) return 0;
    P->n = n;
    P->part = part;
    P->fd = big_alloc(sizeof(double) * n);
    P->fo = big_alloc(sizeof(double) * n);
    P->gd = big_alloc(sizeof(double) * n);
    P->go = big_alloc(sizeof(double) * n);
    for (int64_t i = 0; i < n; i++) {
        int64_t j = part[i];
        if (j < 0) {
            double fe, fp;
            fboth(dg[i], t, &fe, &fp);
            P->fd[i] = fe; P->gd[i] = fp; P->fo[i] = 0.0; P->go[i] = 0.0;
            continue;
        }
        if (j < i) continue;
        double a = dg[i], d = dg[j], b = 0.5 * (of[i] + of[j]);
        double h = 0.5 * (a - d), r = sqrt(h * h + b * b), mid = 0.5 * (a + d);
        double F0, F3, F1, G0, G3, G1;
        if (r == 0.0) {
            double fe, fp;
            fboth(a, t, &fe, &fp);
            F0 = F3 = fe; G0 = G3 = fp; F1 = G1 = 0.0;
        } else {
            double c2 = 0.5 * (1.0 + h / r), s2 = 0.5 * (1.0 - h / r), cs = 0.5 * b / r;
            double l1 = mid + r;
            double l2 = (mid >= r) ? (a * d - b * b) / l1 : mid - r;
            if (l1 == 0.0) l2 = mid - r;
            double f1, f2, g1, g2;
            fboth(l1, t, &f1, &g1);
            fboth(l2, t, &f2, &g2);
            F0 = c2 * f1 + s2 * f2; F3 = s2 * f1 + c2 * f2; F1 = cs * (f1 - f2);
            G0 = c2 * g1 + s2 * g2; G3 = s2 * g1 + c2 * g2; G1 = cs * (g1 - g2);
        }
        P->fd[i] = F0; P->gd[i] = G0; P->fo[i] = F1; P->go[i] = G1;
        P->fd[j] = F3; P->gd[j] = G3; P->fo[j] = F1; P->go[j] = G1;
    }
    return 1;
}

static void pairs_solve(const Pairs *P, const double *b0, const double *b1, double *y)
{
    for (int64_t i = 0; i < P->n; i++) {
        double v = P->fd[i] * b0[i] + P->gd[i] * b1[i];
        int64_t j = P->part[i];
        if (j >= 0) v += P->fo[i] * b0[j] + P->go[i] * b1[j];
        y[i] = v;
    }
}

/* ------------------------------------------------------------------ */
/* small-component (dense block) solver                                */
/* ------------------------------------------------------------------ */
typedef struct {
    int64_t ncomp;
    int64_t *start;  /* ncomp+1 offsets into node list */
    int64_t *node;   /* nodes ordered by component */
    int64_t *moff;   /* offsets into F/G (s*s per component) */
    double *F, *G;
    double *buf;
} Blocks;

static int64_t uf_find(int64_t *par, int64_t x)
{
    while (par[x] != x) {
        par[x] = par[par[x]];
        x = par[x];
    }
    return x;
}

/* Returns 1 and fills B if every component has at most maxs nodes and the
 * total dense work is acceptable (or force=1: one block for the whole matrix). */
/* BFS over the graph of A.  Returns 0 as soon as a component exceeds maxs
 * nodes; otherwise fills node[] (nodes grouped by component), start[] and
 * the number of components. */
static int find_components(const Csr *A, int64_t maxs, int64_t *node, int64_t *start, int64_t *ncomp)
{
    const int64_t n = A->n;
    for (int64_t i = 0; i < n; i++)
        if ((int64_t)(A->ip[i + 1] - A->ip[i]) > maxs) return 0;
    unsigned char *vis = big_alloc(n);
    memset(vis, 0, n);
    int64_t top = 0, nc = 0;
    int ok = 1;
    for (int64_t s0 = 0; s0 < n && ok; s0++) {
        if (vis[s0]) continue;
        int64_t base = top;
        vis[s0] = 1;
        node[top++] = s0;
        for (int64_t h = base; h < top && ok; h++) {
            int64_t i = node[h];
            for (uint64_t k = A->ip[i]; k < A->ip[i + 1]; k++) {
                int64_t j = A->ix[k];
                if (!vis[j]) {
                    vis[j] = 1;
                    node[top++] = j;
                    if (top - base > maxs) { ok = 0; break; }
                }
            }
        }
        start[nc++] = base;
    }
    start[nc] = top;
    *ncomp = nc;
    return ok;
}

static int blocks_build(const Csr *A, double t, int64_t *node, int64_t *start, int64_t ncomp, int force, Blocks *B);

/* complete component labelling: comp[i] = component id, returns count;
 * size[c] = number of nodes */
static int64_t label_components(const Csr *A, int64_t *comp, int64_t *queue, int64_t *size)
{
    const int64_t n = A->n;
    for (int64_t i = 0; i < n; i++) comp[i] = -1;
    int64_t nc = 0;
    for (int64_t s0 = 0; s0 < n; s0++) {
        if (comp[s0] >= 0) continue;
        int64_t top = 0;
        comp[s0] = nc;
        queue[top++] = s0;
        for (int64_t h = 0; h < top; h++) {
            int64_t i = queue[h];
            for (uint64_t k = A->ip[i]; k < A->ip[i + 1]; k++) {
                int64_t j = A->ix[k];
                if (comp[j] < 0) {
                    comp[j] = nc;
                    queue[top++] = j;
                }
            }
        }
        size[nc++] = top;
    }
    return nc;
}

static int blocks_setup(const Csr *A, double t, int64_t maxs, int force, Blocks *B)
{
    const int64_t n = A->n;
    LOG("bfs start %.4f\n", now());
    int64_t *node = big_alloc(sizeof(int64_t) * n);
    int64_t *start = big_alloc(sizeof(int64_t) * (n + 1));
    int64_t ncomp = 0;
    if (force) {
        for (int64_t i = 0; i < n; i++) node[i] = i;
        start[0] = 0;
        start[1] = n;
        ncomp = 1;
    } else if (!find_components(A, maxs, node, start, &ncomp)) {
        return 0;
    }
    LOG("bfs end %.4f\n", now());
    return blocks_build(A, t, node, start, ncomp, force, B);
}

/* Dense eigendecompositions of the given components (node lists). */
static int blocks_build(const Csr *A, double t, int64_t *node, int64_t *start, int64_t ncomp, int force, Blocks *B)
{
    const int64_t n = A->n;
    int64_t maxc = 0;
    double dwork = 0.0;
    for (int64_t c = 0; c < ncomp; c++) {
        int64_t sz = start[c + 1] - start[c];
        if (sz > maxc) maxc = sz;
        dwork += (double)sz * sz * sz;
    }
    if (!force && dwork > 4e8) return 0;
    B->ncomp = ncomp;
    B->start = start;
    B->node = node;
    B->moff = big_alloc(sizeof(int64_t) * (ncomp + 1));
    B->moff[0] = 0;
    for (int64_t c = 0; c < ncomp; c++) {
        int64_t sz = start[c + 1] - start[c];
        B->moff[c + 1] = B->moff[c] + sz * sz;
    }
    double tb0 = now();
    int64_t *loc = big_alloc(sizeof(int64_t) * n);
    for (int64_t c = 0; c < ncomp; c++)
        for (int64_t q = start[c]; q < start[c + 1]; q++) loc[node[q]] = q - start[c];
    int64_t tot = B->moff[ncomp];
    B->F = big_alloc(sizeof(double) * tot);
    B->G = big_alloc(sizeof(double) * tot);
    B->buf = xmalloc(sizeof(double) * 2 * (maxc + 1));
    double *M = xmalloc(sizeof(double) * maxc * maxc);
    double *w = xmalloc(sizeof(double) * maxc);
    int lwork = 1 + 6 * (int)maxc + 2 * (int)(maxc * maxc), liwork = 3 + 5 * (int)maxc;
    double *work = xmalloc(sizeof(double) * lwork);
    int *iwork = xmalloc(sizeof(int) * liwork);
    for (int64_t cc = 0; cc < ncomp; cc++) {
        int64_t s = B->start[cc + 1] - B->start[cc];
        const int64_t *nd = B->node + B->start[cc];
        double *F = B->F + B->moff[cc], *G = B->G + B->moff[cc];
        memset(M, 0, sizeof(double) * s * s);
        for (int64_t r = 0; r < s; r++) {
            int64_t i = nd[r];
            for (uint64_t k = A->ip[i]; k < A->ip[i + 1]; k++) {
                int64_t j = A->ix[k];
                M[r + loc[j] * s] = A->a[k];
            }
        }
        if (s == 1) {
            fboth(M[0], t, F, G);
            continue;
        }
        /* symmetrize */
        for (int64_t r = 0; r < s; r++)
            for (int64_t q = r + 1; q < s; q++) {
                double v = 0.5 * (M[r + q * s] + M[q + r * s]);
                M[r + q * s] = M[q + r * s] = v;
            }
        if (s == 2) {
            double a = M[0], b = M[2], d = M[3];
            double h = 0.5 * (a - d), r = sqrt(h * h + b * b), mid = 0.5 * (a + d);
            if (r == 0.0) {
                F[0] = F[3] = fexp(a, t);
                G[0] = G[3] = fphi(a, t);
                F[1] = F[2] = G[1] = G[2] = 0.0;
                continue;
            }
            double c2 = 0.5 * (1.0 + h / r), s2 = 0.5 * (1.0 - h / r), cs = 0.5 * b / r;
            double l1 = mid + r;
            double l2 = (mid >= r) ? (a * d - b * b) / l1 : mid - r;
            if (l1 == 0.0) l2 = mid - r;
            double f1, f2, g1, g2;
            fboth(l1, t, &f1, &g1);
            fboth(l2, t, &f2, &g2);
            F[0] = c2 * f1 + s2 * f2;
            F[3] = s2 * f1 + c2 * f2;
            F[1] = F[2] = cs * (f1 - f2);
            G[0] = c2 * g1 + s2 * g2;
            G[3] = s2 * g1 + c2 * g2;
            G[1] = G[2] = cs * (g1 - g2);
            continue;
        }
        int si = (int)s, info = 0;
        dsyevd_("V", "U", &si, M, &si, w, work, &lwork, iwork, &liwork, &info);
        for (int64_t r = 0; r < s; r++)
            for (int64_t q = 0; q < s; q++) {
                double fs = 0.0, gs = 0.0;
                for (int64_t k = 0; k < s; k++) {
                    double p = M[r + k * s] * M[q + k * s];
                    fs += p * fexp(w[k], t);
                    gs += p * fphi(w[k], t);
                }
                F[r + q * s] = fs;
                G[r + q * s] = gs;
            }
    }
    LOG("blocks: ncomp=%ld maxc=%ld dense part %.4f\n", (long)ncomp, (long)maxc, now() - tb0);
    free(M); free(w); free(work); free(iwork);
    return 1;
}

static void blocks_solve(const Blocks *B, const double *b0, const double *b1, double *y)
{
    double *x0 = B->buf;
    for (int64_t cc = 0; cc < B->ncomp; cc++) {
        int64_t s = B->start[cc + 1] - B->start[cc];
        const int64_t *nd = B->node + B->start[cc];
        const double *F = B->F + B->moff[cc], *G = B->G + B->moff[cc];
        if (s == 1) {
            int64_t i = nd[0];
            y[i] = F[0] * b0[i] + G[0] * b1[i];
            continue;
        }
        if (s == 2) {
            int64_t i = nd[0], j = nd[1];
            double u0 = b0[i], u1 = b0[j], w0 = b1[i], w1 = b1[j];
            y[i] = F[0] * u0 + F[2] * u1 + G[0] * w0 + G[2] * w1;
            y[j] = F[1] * u0 + F[3] * u1 + G[1] * w0 + G[3] * w1;
            continue;
        }
        double *x1 = x0 + s;
        for (int64_t r = 0; r < s; r++) {
            x0[r] = b0[nd[r]];
            x1[r] = b1[nd[r]];
        }
        for (int64_t r = 0; r < s; r++) {
            double acc = 0.0;
            for (int64_t q = 0; q < s; q++) acc += F[r + q * s] * x0[q] + G[r + q * s] * x1[q];
            y[nd[r]] = acc;
        }
    }
}

/* ------------------------------------------------------------------ */
/* CHOLMOD (SuiteSparse 5.x / CHOLMOD 3.x ABI, loaded with dlopen)     */
/* ------------------------------------------------------------------ */
typedef struct {
    size_t nrow, ncol, nzmax;
    void *p, *i, *nz, *x, *z;
    int stype, itype, xtype, dtype, sorted, packed;
} cm_sparse;
typedef struct {
    size_t nrow, ncol, nzmax, d;
    void *x, *z;
    int xtype, dtype;
} cm_dense;
typedef struct {
    size_t n, minor;
    void *Perm, *ColCount, *IPerm;
} cm_factor_head;

static struct {
    int ok;
    void *common;
    int (*start)(void *);
    void *(*analyze)(cm_sparse *, void *);
    int (*factorize_p)(cm_sparse *, double *, int *, size_t, void *, void *);
    int (*solve2)(int, void *, cm_dense *, void *, cm_dense **, void **, cm_dense **, cm_dense **, void *);
    int (*free_factor)(void **, void *);
    int amd_only;
} CM;

static void cm_load(void)
{
    void *h = dlopen("libcholmod.so.3", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("libcholmod.so", RTLD_NOW | RTLD_LOCAL);
    if (!h) return;
    CM.start = (int (*)(void *))dlsym(h, "cholmod_start");
    CM.analyze = (void *(*)(cm_sparse *, void *))dlsym(h, "cholmod_analyze");
    CM.factorize_p = (int (*)(cm_sparse *, double *, int *, size_t, void *, void *))dlsym(h, "cholmod_factorize_p");
    CM.solve2 = (int (*)(int, void *, cm_dense *, void *, cm_dense **, void **, cm_dense **, cm_dense **, void *))dlsym(h, "cholmod_solve2");
    CM.free_factor = (int (*)(void **, void *))dlsym(h, "cholmod_free_factor");
    if (!CM.start || !CM.analyze || !CM.factorize_p || !CM.solve2) return;
    /* silence SuiteSparse printing: printf_func is the 5th pointer */
    void **cfg = (void **)dlsym(h, "SuiteSparse_config");
    if (!cfg) {
        void *h2 = dlopen("libsuitesparseconfig.so.5", RTLD_NOW | RTLD_LOCAL);
        if (h2) cfg = (void **)dlsym(h2, "SuiteSparse_config");
    }
    if (cfg) cfg[4] = NULL;
    void (*ob)(int) = (void (*)(int))dlsym(RTLD_DEFAULT, "openblas_set_num_threads");
    if (ob) ob(1);
    CM.common = calloc(1, 1 << 16);
    if (!CM.common || !CM.start(CM.common)) return;
    /* AMD-only ordering if the cholmod_common layout matches CHOLMOD 3.x */
    char *cb = (char *)CM.common;
    double grow0 = *(double *)(cb + 8), swc = *(double *)(cb + 40);
    int prt = *(int *)(cb + 144), nmeth = *(int *)(cb + 168);
    int ord1 = *(int *)(cb + 316 + 168), ord2 = *(int *)(cb + 316 + 2 * 168);
    if (grow0 == 1.2 && swc == 40.0 && prt == 3 && nmeth == 0 && ord1 == 2 && ord2 == 3) {
        *(int *)(cb + 168) = 1;        /* nmethods */
        *(int *)(cb + 316) = 2;        /* method[0].ordering = CHOLMOD_AMD */
        *(int *)(cb + 144) = 0;        /* print */
        CM.amd_only = 1;
    }
    CM.ok = 1;
}

typedef struct {
    cm_sparse A;
    int32_t *p32;
    void *F;          /* cholmod_factor* */
    double sigma;
    int64_t lnz;
    double flops;
    cm_dense *X, *Y, *E;
} Chol;

/* symbolic analysis of the pattern of A (upper triangle used) */
static int chol_analyze(Chol *C, const Csr *M)
{
    memset(C, 0, sizeof *C);
    int64_t n = M->n;
    C->p32 = xmalloc(sizeof(int32_t) * (n + 1));
    for (int64_t i = 0; i <= n; i++) C->p32[i] = (int32_t)M->ip[i];
    C->A.nrow = C->A.ncol = (size_t)n;
    C->A.nzmax = M->ip[n];
    C->A.p = C->p32;
    C->A.i = (void *)M->ix;
    C->A.x = (void *)M->a;
    C->A.stype = 1;
    C->A.itype = 0;  /* CHOLMOD_INT */
    C->A.xtype = 1;  /* CHOLMOD_REAL */
    C->A.dtype = 0;  /* CHOLMOD_DOUBLE */
    C->A.sorted = 1;
    C->A.packed = 1;
    C->F = CM.analyze(&C->A, CM.common);
    if (!C->F) return 0;
    const cm_factor_head *h = (const cm_factor_head *)C->F;
    const int32_t *cc = (const int32_t *)h->ColCount;
    double fl = 0.0;
    int64_t lnz = 0;
    for (int64_t j = 0; j < n; j++) {
        lnz += cc[j];
        fl += (double)cc[j] * cc[j];
    }
    C->lnz = lnz;
    C->flops = fl;
    return 1;
}

static int chol_factor(Chol *C, double sigma)
{
    double beta[2] = {sigma, 0.0};
    C->sigma = sigma;
    if (!CM.factorize_p(&C->A, beta, NULL, 0, C->F, CM.common)) return 0;
    const cm_factor_head *h = (const cm_factor_head *)C->F;
    return h->minor == h->n;
}

/* X = (A + sigma I)^{-1} B for an n x c column-major block */
static void chol_solve(Chol *C, int64_t n, int c, double *B, double *X)
{
    cm_dense Bd;
    memset(&Bd, 0, sizeof Bd);
    Bd.nrow = (size_t)n;
    Bd.ncol = (size_t)c;
    Bd.nzmax = (size_t)n * c;
    Bd.d = (size_t)n;
    Bd.x = B;
    Bd.xtype = 1;
    Bd.dtype = 0;
    CM.solve2(0, C->F, &Bd, NULL, &C->X, NULL, &C->Y, &C->E, CM.common);
    memcpy(X, C->X->x, sizeof(double) * n * c);
}

/* ------------------------------------------------------------------ */
/* Lanczos                                                             */
/* ------------------------------------------------------------------ */
#define MAXIT 6000
#define CHUNK 16
#define NSAMP 256

typedef struct {
    Csr A;
    double t, rtol, atol, normest, deadline;
    double mem_budget;
    /* basis storage */
    double **chunk;
    int64_t nchunk_alloc;
    double *rot[3];
    int64_t len_alloc;   /* c*n of allocated storage */
    int64_t kstore;      /* vectors with index < kstore are stored */
    int64_t samp[NSAMP];
    int ns;
    /* quadrature */
    double gx[8], gw[8];
    /* work */
    double *d, *e, *Z, *work, *w;
    int *iwork, *isuppz;
    int64_t zcap;
    int last_m;
    /* shift-invert switching */
    int si_state;          /* 0 undecided, 1 use SI, -1 never */
    int switched;          /* lz_solve aborted in favour of SI */
    int64_t reqs_left;     /* including the current request */
    int prec_lvl;          /* 0 p4, 1 p7, 2 p10 */
    void *chol;            /* Chol* */
    Sym *sym;              /* symmetric copy (built lazily) */
    int nosym;
    int64_t prev_m[3];     /* iterations of the previous request per chain count */
    RotLog rl;
    int si_within;         /* allow switching inside a request (off by default) */
    int64_t sym_k0;        /* first step index using the symmetric kernel */
    int analyzed, an_ok, factored;
    double c_fact, c_req;  /* predicted factorization / per-request SI cost */
    double c_an;           /* measured analysis time */
    double poly_req;       /* measured time of the last full polynomial request */
    double si_req;         /* measured time of the last SI request */
} Lz;

static double *vec_ptr(Lz *L, int64_t k, int64_t len)
{
    if (k < L->kstore) {
        int64_t ci = k / CHUNK;
        if (ci >= L->nchunk_alloc || L->chunk[ci] == NULL) {
            if (ci >= L->nchunk_alloc) {
                int64_t nn = L->nchunk_alloc * 2 + 16;
                L->chunk = realloc(L->chunk, sizeof(double *) * nn);
                for (int64_t q = L->nchunk_alloc; q < nn; q++) L->chunk[q] = NULL;
                L->nchunk_alloc = nn;
            }
            L->chunk[ci] = big_alloc(sizeof(double) * len * CHUNK);
            if (!L->chunk[ci]) _exit(6);
        }
        return L->chunk[ci] + (k % CHUNK) * len;
    }
    return L->rot[k % 3];
}

static void lz_init(Lz *L, const Csr *A, double t, double rtol, double atol, double normest)
{
    memset(L, 0, sizeof *L);
    L->A = *A;
    L->t = t;
    L->rtol = rtol;
    L->atol = atol;
    L->normest = normest;
    L->mem_budget = 2.0e9;
    int64_t n = A->n;
    L->ns = n < NSAMP ? (int)n : NSAMP;
    for (int s = 0; s < L->ns; s++) L->samp[s] = (int64_t)((double)s * (n - 1) / (L->ns > 1 ? L->ns - 1 : 1));
    static const double x8[8] = {-0.9602898564975363, -0.7966664774136267, -0.5255324099163290,
                                 -0.1834346424956498, 0.1834346424956498, 0.5255324099163290,
                                 0.7966664774136267, 0.9602898564975363};
    static const double w8[8] = {0.1012285362903763, 0.2223810344533745, 0.3137066458778873,
                                 0.3626837833783620, 0.3626837833783620, 0.3137066458778873,
                                 0.2223810344533745, 0.1012285362903763};
    for (int i = 0; i < 8; i++) {
        L->gx[i] = 0.5 * (x8[i] + 1.0);
        L->gw[i] = 0.5 * w8[i];
    }
}

static void lz_ensure_work(Lz *L, int64_t m)
{
    if (m <= L->zcap) return;
    int64_t cap = m + m / 2 + 16;
    free(L->d); free(L->e); free(L->Z); free(L->work); free(L->iwork); free(L->w); free(L->isuppz);
    L->d = xmalloc(sizeof(double) * cap);
    L->e = xmalloc(sizeof(double) * cap);
    L->w = xmalloc(sizeof(double) * cap);
    L->Z = xmalloc(sizeof(double) * cap * cap);
    L->work = xmalloc(sizeof(double) * (1 + 18 * cap + cap * cap));
    L->iwork = xmalloc(sizeof(int) * (3 + 10 * cap));
    L->isuppz = xmalloc(sizeof(int) * 2 * cap);
    L->zcap = cap;
}

/* Eigen-decomposition of the leading m x m block of chain j's tridiagonal
 * matrix; returns effective size (truncated at breakdown). */
static int64_t lz_eig(Lz *L, const double *alpha, const double *beta, int c, int j, int64_t m)
{
    int64_t mj = m;
    for (int64_t k = 0; k + 1 < m; k++)
        if (beta[k * c + j] == 0.0) { mj = k + 1; break; }
    lz_ensure_work(L, mj);
    for (int64_t k = 0; k < mj; k++) L->d[k] = alpha[k * c + j];
    for (int64_t k = 0; k + 1 < mj; k++) L->e[k] = beta[k * c + j];
    int mi = (int)mj, ldz = (int)mj, info = 0;
    if (mj == 1) {
        L->Z[0] = 1.0;
        return mj;
    }
    /* MRRR first (O(m^2)); divide and conquer as a fallback */
    double vl = 0.0, vu = 0.0;
    int il = 0, iu = 0, mfound = 0, nzc = mi, tryrac = 1;
    int lwork = 18 * mi, liwork = 10 * mi;
    L->e[mj - 1] = 0.0;
    double *d0 = L->work + lwork, *e0 = d0 + mj;   /* copies for the fallback */
    memcpy(d0, L->d, sizeof(double) * mj);
    memcpy(e0, L->e, sizeof(double) * mj);
    dstemr_("V", "A", &mi, L->d, L->e, &vl, &vu, &il, &iu, &mfound, L->w, L->Z, &ldz, &nzc,
            L->isuppz, &tryrac, L->work, &lwork, L->iwork, &liwork, &info);
    if (info == 0 && mfound == mi) {
        memcpy(L->d, L->w, sizeof(double) * mj);
    } else {
        LOG("dstemr info %d (m=%d), fallback dstevd\n", info, mfound);
        memcpy(L->d, d0, sizeof(double) * mj);
        memcpy(L->e, e0, sizeof(double) * mj);
        int lw2 = (int)(1 + 4 * mj + mj * mj), liw2 = (int)(3 + 5 * mj);
        double *wk2 = xmalloc(sizeof(double) * lw2);
        dstevd_("V", &mi, L->d, L->e, L->Z, &ldz, wk2, &lw2, L->iwork, &liw2, &info);
        free(wk2);
        if (info != 0) LOG("dstevd info %d\n", info);
    }
    return mj;
}

/* Implicit QL (tqli) on a symmetric tridiagonal matrix tracking only the
 * first and last rows of the eigenvector matrix.  d: diagonal (out:
 * eigenvalues), e: off-diagonal e[i] couples i,i+1 (destroyed). */

static void rot_push(RotLog *R, int64_t i, double c, double s)
{
    if (R->cnt == R->cap) {
        R->cap = R->cap ? 2 * R->cap : (1 << 16);
        R->idx = realloc(R->idx, sizeof(int32_t) * R->cap);
        R->cs = realloc(R->cs, sizeof(double) * 2 * R->cap);
        if (!R->idx || !R->cs) _exit(5);
    }
    R->idx[R->cnt] = (int32_t)i;
    R->cs[2 * R->cnt] = c;
    R->cs[2 * R->cnt + 1] = s;
    R->cnt++;
}

/* w <- Z w where Z is the accumulated eigenvector matrix of tqli2 */
static void rot_apply(const RotLog *R, double *w)
{
    for (int64_t q = R->cnt - 1; q >= 0; q--) {
        int64_t i = R->idx[q];
        double c = R->cs[2 * q], s = R->cs[2 * q + 1];
        double a = w[i], b = w[i + 1];
        w[i] = c * a + s * b;
        w[i + 1] = -s * a + c * b;
    }
}

static int tqli2(int64_t n, double *d, double *e, double *z0, double *z1, RotLog *R)
{
    if (R) R->cnt = 0;
    for (int64_t k = 0; k < n; k++) {
        z0[k] = (k == 0);
        z1[k] = (k == n - 1);
    }
    if (n == 1) return 0;
    e[n - 1] = 0.0;
    for (int64_t l = 0; l < n; l++) {
        int iter = 0;
        int64_t m;
        do {
            for (m = l; m < n - 1; m++) {
                double dd = fabs(d[m]) + fabs(d[m + 1]);
                if (fabs(e[m]) <= 2.2e-16 * dd) break;
            }
            if (m != l) {
                if (iter++ == 80) return -1;
                double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                double r = sqrt(g * g + 1.0);
                g = d[m] - d[l] + e[l] / (g + (g >= 0.0 ? r : -r));
                double sn = 1.0, cs = 1.0, p = 0.0;
                int64_t i;
                int brk0 = 0;
                for (i = m - 1; i >= l; i--) {
                    double f = sn * e[i], b = cs * e[i];
                    r = sqrt(f * f + g * g);
                    e[i + 1] = r;
                    if (r == 0.0) {
                        d[i + 1] -= p;
                        e[m] = 0.0;
                        brk0 = 1;
                        break;
                    }
                    sn = f / r;
                    cs = g / r;
                    g = d[i + 1] - p;
                    r = (d[i] - g) * sn + 2.0 * cs * b;
                    p = sn * r;
                    d[i + 1] = g + p;
                    g = cs * r - b;
                    if (R) rot_push(R, i, cs, sn);
                    double t0 = z0[i + 1];
                    z0[i + 1] = sn * z0[i] + cs * t0;
                    z0[i] = cs * z0[i] - sn * t0;
                    double t1 = z1[i + 1];
                    z1[i + 1] = sn * z1[i] + cs * t1;
                    z1[i] = cs * z1[i] - sn * t1;
                }
                if (brk0) continue;
                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        } while (m != l);
    }
    return 0;
}

/* integral over [0,t] of |sum_k w_k h_s(theta_k)| ds;  phi=0: h=exp(-s th),
 * phi=1: h=(1-exp(-s th))/th. */
static double lz_resint(Lz *L, const double *th, const double *w, int64_t m, int phi, double *tmpw, double *tmpth)
{
    double t = L->t;
    double wmax = 0.0, thmax = 0.0;
    for (int64_t k = 0; k < m; k++) {
        if (fabs(w[k]) > wmax) wmax = fabs(w[k]);
        if (th[k] > thmax) thmax = th[k];
    }
    if (wmax == 0.0) return 0.0;
    int64_t me = 0;
    for (int64_t k = 0; k < m; k++)
        if (fabs(w[k]) > 1e-17 * wmax) {
            tmpw[me] = w[k];
            tmpth[me] = th[k];
            me++;
        }
    int K = (int)ceil(log2(fmax(t * thmax, 1.0))) + 6;
    if (K > 60) K = 60;
    double total = 0.0;
    for (int lev = 0; lev <= K; lev++) {
        double a, b;
        if (lev < K) {
            a = t * ldexp(1.0, -(lev + 1));
            b = t * ldexp(1.0, -lev);
        } else {
            a = 0.0;
            b = t * ldexp(1.0, -K);
        }
        for (int q = 0; q < 8; q++) {
            double s = a + (b - a) * L->gx[q];
            double acc = 0.0;
            if (phi) {
                for (int64_t k = 0; k < me; k++) acc += tmpw[k] * fphi(tmpth[k], s);
            } else {
                for (int64_t k = 0; k < me; k++) acc += tmpw[k] * exp(-s * tmpth[k]);
            }
            total += (b - a) * L->gw[q] * fabs(acc);
        }
    }
    return total;
}

static double SI_TRIGGER = 0.5;
static double SI_SIGMA = 10.0;

static void si_analyze(Lz *L, int c)
{
    if (L->analyzed) return;
    L->analyzed = 1;
    if (!CM.ok || L->chol == NULL || L->t * L->normest < 200.0) return;
    Chol *C = (Chol *)L->chol;
    double q0 = now();
    if (!chol_analyze(C, &L->A)) return;
    int64_t n = L->A.n;
    double ratio = C->flops / fmax((double)C->lnz, 1.0);
    double rate = fmin(2e10, 5e7 * ratio);
    L->c_fact = C->flops / rate + 5e-8 * (double)C->lnz + 1e-3;
    double t_solve = 3.5e-9 * (double)C->lnz * (c == 2 ? 1.4 : 1.0) + 3.5e-8 * (double)n * c;
    double m_si = L->prec_lvl == 0 ? 15.0 : (L->prec_lvl == 1 ? 22.0 : 30.0);
    L->c_req = m_si * t_solve + 1e-3;
    L->an_ok = 1;
    L->c_an = now() - q0;
    LOG("si_analyze: %.3fs lnz=%ld fl=%.3g c_fact=%.3f c_req=%.3f\n", L->c_an, (long)C->lnz, C->flops,
        L->c_fact, L->c_req);
}

static int si_factor(Lz *L)
{
    if (L->factored) return L->factored > 0;
    double q0 = now();
    if (!chol_factor((Chol *)L->chol, SI_SIGMA / L->t)) {
        L->factored = -1;
        L->si_state = -1;
        return 0;
    }
    L->factored = 1;
    LOG("si_factor: %.3fs\n", now() - q0);
    return 1;
}

/* Called at polynomial-Lanczos checks.  Returns 1 if the current request
 * should be restarted with shift-and-invert Lanczos. */
static int si_consider(Lz *L, double spent, double rem_pred, int64_t m, int c)
{
    if (!L->si_within) return 0;
    if (L->si_state != 0 || !CM.ok || L->chol == NULL) return 0;
    if (L->t * L->normest < 200.0) return 0;
    (void)rem_pred;
    (void)m;
    if (!L->analyzed) {
        if (spent > SI_TRIGGER) si_analyze(L, c);
        else return 0;
    }
    if (!L->an_ok) return 0;
    double cost = (L->factored > 0 ? 0.0 : L->c_fact) + L->c_req;
    if (spent > cost + L->c_an) {
        if (!si_factor(L)) return 0;
        L->si_state = 1;
        return 1;
    }
    return 0;
}

static void lz_solve(Lz *L, const double *b0, const double *b1, double *y)
{
    double tstart = now();
    L->switched = 0;
    L->sym_k0 = L->sym ? 0 : INT64_MAX;
    const int64_t n = L->A.n;
    const double t = L->t;
    const double *bs[2];
    int phi[2];
    double nb[2];
    int c = 0;
    double n0 = 0.0, n1 = 0.0;
    for (int64_t i = 0; i < n; i++) {
        n0 += b0[i] * b0[i];
        n1 += b1[i] * b1[i];
    }
    if (n0 > 0.0) { bs[c] = b0; phi[c] = 0; nb[c] = sqrt(n0); c++; }
    if (n1 > 0.0) { bs[c] = b1; phi[c] = 1; nb[c] = sqrt(n1); c++; }
    if (c == 0) {
        memset(y, 0, sizeof(double) * n);
        return;
    }
    const int64_t len = (int64_t)c * n;
    /* storage */
    if (L->len_alloc != len) {
        /* (re)allocate rotating buffers; drop old chunks */
        for (int64_t q = 0; q < L->nchunk_alloc; q++)
            if (L->chunk[q]) {
                big_free(L->chunk[q], sizeof(double) * L->len_alloc * CHUNK);
                L->chunk[q] = NULL;
            }
        for (int q = 0; q < 3; q++) {
            if (L->rot[q]) big_free(L->rot[q], sizeof(double) * L->len_alloc);
            L->rot[q] = big_alloc(sizeof(double) * len);
        }
        L->len_alloc = len;
    }
    int64_t kmax = (int64_t)(L->mem_budget / (8.0 * len));
    kmax = (kmax / CHUNK) * CHUNK;
    if (kmax < 2 * CHUNK) kmax = 2 * CHUNK;
    L->kstore = kmax;

    double *alpha = xcalloc((size_t)(MAXIT + 2) * c, sizeof(double));
    double *beta = xcalloc((size_t)(MAXIT + 2) * c, sizeof(double));
    double *vmax = xcalloc((size_t)(MAXIT + 3) * c, sizeof(double));
    int64_t scap = 256;
    double *S = xmalloc(sizeof(double) * scap * L->ns * c);
    double *u[2] = {NULL, NULL};
    int64_t ucap = 0;
    double *tw = NULL, *tth = NULL, *wk = NULL;

    double *v0 = vec_ptr(L, 0, len);
    for (int j = 0; j < c; j++) {
        double inv = 1.0 / nb[j], mx = 0.0;
        const double *b = bs[j];
        for (int64_t i = 0; i < n; i++) {
            double v = b[i] * inv;
            v0[i * c + j] = v;
            if (fabs(v) > mx) mx = fabs(v);
        }
        vmax[j] = mx;
    }
    for (int s = 0; s < L->ns; s++)
        for (int j = 0; j < c; j++) S[s * c + j] = v0[L->samp[s] * c + j];

    const double brk = 1e-14 * (L->normest > 0 ? L->normest : 1.0);
    int64_t m = 0, target = 2;
    const int64_t pm = getenv("EXPO_NOPM") ? 0 : L->prev_m[c];
    if (pm > 8) target = (int64_t)(0.9 * (double)pm);
    double t_adv = 0.0, ys_ratio = 1.0, best_est = 1e300;
    int stall = 0;
    int64_t n_adv = 0;
    int64_t hist_m[4] = {0};
    double hist_e[4] = {0};
    int nh = 0;
    double zero2[2] = {0.0, 0.0};
    int done_all = 0;
    for (;;) {
        /* advance */
        double tadv0 = now();
        int64_t madv0 = m;
        while (m < target) {
            int64_t k = m;
            double *vp = k > 0 ? vec_ptr(L, k - 1, len) : v0;
            double *vc = vec_ptr(L, k, len);
            double *vn = vec_ptr(L, k + 1, len);
            const double *bp = k > 0 ? beta + (k - 1) * c : zero2;
            if (L->sym && k >= L->sym_k0)
                lanczos_step_sym(L->sym, c, vp, vc, vn, bp, alpha + k * c, beta + k * c,
                                 vmax + (k + 1) * c, 1, brk);
            else
                lanczos_step(&L->A, c, vp, vc, vn, bp, alpha + k * c, beta + k * c,
                             vmax + (k + 1) * c, 1, brk);
            if (k + 2 > scap) {
                scap *= 2;
                S = realloc(S, sizeof(double) * scap * L->ns * c);
                if (!S) _exit(5);
            }
            double *Sk = S + (k + 1) * L->ns * c;
            for (int s = 0; s < L->ns; s++)
                for (int j = 0; j < c; j++) Sk[s * c + j] = vn[L->samp[s] * c + j];
            m++;
            int allz = 1;
            for (int j = 0; j < c; j++) if (beta[k * c + j] > 0.0) allz = 0;
            if (allz) { done_all = 1; break; }
        }
        if (m > madv0) { t_adv += now() - tadv0; n_adv += m - madv0; }
        /* check */
        double tchk = now();
        if (m + 1 > ucap) {
            ucap = 2 * m + 16;
            for (int j = 0; j < c; j++) { free(u[j]); u[j] = xmalloc(sizeof(double) * ucap); }
            free(tw); free(tth); free(wk);
            tw = xmalloc(sizeof(double) * ucap);
            tth = xmalloc(sizeof(double) * ucap);
            wk = xmalloc(sizeof(double) * ucap);
        }
        double est = 0.0, unorm2 = 0.0;
        double s_est = 0.0, s_goal = 0.0;
        double te_eig = 0, te_gemv = 0, te_res = 0;
        /* cheap check (eigenvalues + first/last eigenvector rows) for large m */
        if (0 && m >= 150 && !done_all) {
            double qa = now();
            double cest = 0.0, cun2 = 0.0;
            int okq = 1;
            for (int j = 0; j < c && okq; j++) {
                int64_t mj = m;
                for (int64_t k = 0; k + 1 < m; k++)
                    if (beta[k * c + j] == 0.0) { mj = k + 1; break; }
                lz_ensure_work(L, mj);
                double *dd = L->d, *ee = L->e, *z0 = L->w, *z1 = L->Z;
                for (int64_t k = 0; k < mj; k++) dd[k] = alpha[k * c + j];
                for (int64_t k = 0; k + 1 < mj; k++) ee[k] = beta[k * c + j];
                if (tqli2(mj, dd, ee, z0, z1, NULL) != 0) { okq = 0; break; }
                for (int64_t k = 0; k < mj; k++) {
                    double f = phi[j] ? fphi(dd[k], t) : fexp(dd[k], t);
                    cun2 += nb[j] * nb[j] * f * f * z0[k] * z0[k];
                }
                double bm = (mj == m) ? beta[(m - 1) * c + j] : 0.0;
                if (bm > 0.0) {
                    for (int64_t k = 0; k < mj; k++) wk[k] = z0[k] * z1[k];
                    double I = lz_resint(L, dd, wk, mj, phi[j], tw, tth);
                    cest += vmax[m * c + j] * bm * nb[j] * I;
                }
            }
            if (okq) {
                double ylb = sqrt(cun2 / (double)n) * fmax(1.0, ys_ratio);
                double ctol = L->atol + L->rtol * ylb;
                LOG("  m=%ld cheap est=%.3e tol=%.3e ratio=%.3e chk=%.2fms\n", (long)m, cest, ctol, cest / ctol, 1e3 * (now() - qa));
                if (cest < best_est) { if (cest < 0.5 * best_est) stall = 0; else stall++; best_est = cest; }
                else stall++;
                int stagnant = (cest < 20.0 * ctol && stall >= 3);
                if (cest > 0.5 * ctol && m < MAXIT && now() <= L->deadline && !stagnant) {
                    s_est = cest;
                    s_goal = 0.5 * ctol;
                    goto schedule;
                }
            }
        }
        for (int j = 0; j < c; j++) {
            double qa = now();
            if (m >= 150) {
                /* QL with logged rotations: u = Z (f(theta) .* q0) without forming Z */
                int64_t mj = m;
                for (int64_t k = 0; k + 1 < m; k++)
                    if (beta[k * c + j] == 0.0) { mj = k + 1; break; }
                lz_ensure_work(L, mj);
                double *dd = L->d, *ee = L->e, *z0 = L->w, *z1 = L->Z;
                for (int64_t k = 0; k < mj; k++) dd[k] = alpha[k * c + j];
                for (int64_t k = 0; k + 1 < mj; k++) ee[k] = beta[k * c + j];
                if (tqli2(mj, dd, ee, z0, z1, &L->rl) == 0) {
                    for (int64_t k = 0; k < mj; k++) {
                        double f = phi[j] ? fphi(dd[k], t) : fexp(dd[k], t);
                        u[j][k] = nb[j] * f * z0[k];
                    }
                    rot_apply(&L->rl, u[j]);
                    te_eig += now() - qa;
                    for (int64_t k = mj; k < m; k++) u[j][k] = 0.0;
                    for (int64_t k = 0; k < mj; k++) unorm2 += u[j][k] * u[j][k];
                    double bm = (mj == m) ? beta[(m - 1) * c + j] : 0.0;
                    if (bm > 0.0) {
                        for (int64_t k = 0; k < mj; k++) wk[k] = z0[k] * z1[k];
                        double I = lz_resint(L, dd, wk, mj, phi[j], tw, tth);
                        est += vmax[m * c + j] * bm * nb[j] * I;
                    }
                    continue;
                }
                LOG("tqli2 failed at m=%ld, using dstevd\n", (long)m);
            }
            int64_t mj = lz_eig(L, alpha, beta, c, j, m);
            te_eig += now() - qa;
            const double *th = L->d, *Z = L->Z;
            for (int64_t k = 0; k < mj; k++) {
                double f = phi[j] ? fphi(th[k], t) : fexp(th[k], t);
                wk[k] = f * Z[k * mj];
            }
            int mi = (int)mj, one = 1;
            double alp = nb[j], bet = 0.0;
            double qb = now();
            dgemv_("N", &mi, &mi, &alp, Z, &mi, wk, &one, &bet, u[j], &one);
            te_gemv += now() - qb;
            for (int64_t k = mj; k < m; k++) u[j][k] = 0.0;
            for (int64_t k = 0; k < mj; k++) unorm2 += u[j][k] * u[j][k];
            /* residual: rho(s) = beta_m nb sum_k Z[m-1,k] Z[0,k] h_s(th_k) */
            double bm = (mj == m) ? beta[(m - 1) * c + j] : 0.0;
            if (bm > 0.0) {
                for (int64_t k = 0; k < mj; k++) wk[k] = Z[k * mj] * Z[mj - 1 + k * mj];
                double qc = now();
                double I = lz_resint(L, th, wk, mj, phi[j], tw, tth);
                te_res += now() - qc;
                est += vmax[m * c + j] * bm * nb[j] * I;
            }
        }
        double ys = 0.0;
        for (int s = 0; s < L->ns; s++) {
            double acc = 0.0;
            for (int64_t k = 0; k < m; k++) {
                const double *Sk = S + k * L->ns * c + s * c;
                for (int j = 0; j < c; j++) acc += Sk[j] * u[j][k];
            }
            if (fabs(acc) > ys) ys = fabs(acc);
        }
        double ylb = fmax(ys, sqrt(unorm2 / (double)n));
        if (unorm2 > 0.0) ys_ratio = ys / sqrt(unorm2 / (double)n);
        double tol = L->atol + L->rtol * ylb;
        LOG("  m=%ld est=%.3e tol=%.3e ratio=%.3e chk=%.2fms eig=%.2f gemv=%.2f res=%.2f\n", (long)m, est, tol, est / tol, 1e3 * (now() - tchk), te_eig * 1e3, te_gemv * 1e3, te_res * 1e3);
        double goal = 0.5 * tol;
        if (est < best_est) { if (est < 0.5 * best_est) stall = 0; else stall++; best_est = est; }
        else stall++;
        int stagnant = (est < 20.0 * tol && stall >= 3);
        if (stagnant) LOG("  stagnation accepted at m=%ld\n", (long)m);
        if (est <= goal || done_all || m >= MAXIT || now() > L->deadline || stagnant) break;
        s_est = est;
        s_goal = goal;
schedule:
        /* switch to the symmetric kernel for long runs */
        if (!L->sym && m >= 20 && !L->nosym) {
            L->sym = xmalloc(sizeof(Sym));
            sym_build(&L->A, L->sym);
            L->sym_k0 = m;
        }
        /* schedule */
        if (nh == 4) {
            memmove(hist_m, hist_m + 1, 3 * sizeof(int64_t));
            memmove(hist_e, hist_e + 1, 3 * sizeof(double));
            nh = 3;
        }
        hist_m[nh] = m;
        hist_e[nh] = s_est;
        nh++;
        int64_t step;
        if (nh >= 2 && hist_e[nh - 1] < hist_e[nh - 2] && hist_e[nh - 1] > 0) {
            double r = log(hist_e[nh - 1] / hist_e[nh - 2]) / (double)(hist_m[nh - 1] - hist_m[nh - 2]);
            double need = log(s_goal / s_est) / r;
            if (L->si_state == 0 && m >= 16) {
                double spent = now() - tstart;
                double rem_pred = need * spent / (double)m;
                if (si_consider(L, spent, rem_pred, m, c)) { L->switched = 1; break; }
            }
            double chk_cost = now() - tchk, it_cost = n_adv > 0 ? t_adv / (double)n_adv : 1e-3;
            double extra = 0.5 * chk_cost / fmax(it_cost, 1e-7);
            if (extra > 0.2 * m) extra = 0.2 * m;
            step = (int64_t)ceil(need * 1.1 + extra) + 1;
            int64_t lo = m / 40 > 1 ? m / 40 : 1;
            int64_t hi = m / 4 > 3 ? m / 4 : 3;
            if (step < lo) step = lo;
            if (step > hi) step = hi;
            if (pm > 8 && m < pm) step = pm - m;
        } else {
            step = m / 3 > 2 ? m / 3 : 2;
            if (pm > 8 && m < pm) step = pm - m;
            if (L->si_state == 0 && m >= 16) {
                double spent = now() - tstart;
                if (si_consider(L, spent, 0.0, m, c)) { L->switched = 1; break; }
            }
        }
        target = m + step;
        if (target > MAXIT) target = MAXIT;
    }
    L->last_m = (int)m;
    if (!L->switched && m < MAXIT) L->prev_m[c] = m;
    if (L->switched) {
        free(alpha); free(beta); free(vmax); free(S);
        for (int j = 0; j < c; j++) free(u[j]);
        free(tw); free(tth); free(wk);
        return;
    }
    /* assemble y = sum_k sum_j u_j[k] v_k[j] */
    memset(y, 0, sizeof(double) * n);
    int64_t mst = m < L->kstore ? m : L->kstore;
    const int64_t BS = 2048;
    for (int64_t i0 = 0; i0 < n; i0 += BS) {
        int64_t i1 = i0 + BS < n ? i0 + BS : n;
        for (int64_t k = 0; k < mst; k++) {
            const double *v = vec_ptr(L, k, len);
            if (c == 1) {
                double c0 = u[0][k];
                if (c0 == 0.0) continue;
                for (int64_t i = i0; i < i1; i++) y[i] += c0 * v[i];
            } else {
                double c0 = u[0][k], c1 = u[1][k];
                for (int64_t i = i0; i < i1; i++) y[i] += c0 * v[2 * i] + c1 * v[2 * i + 1];
            }
        }
    }
    if (m > L->kstore) {
        /* replay the tail: regenerate v_kstore .. v_{m-1} */
        int64_t k0 = L->kstore - 1;
        double *coef = xmalloc(sizeof(double) * m * c);
        for (int64_t k = 0; k < m; k++)
            for (int j = 0; j < c; j++) coef[k * c + j] = u[j][k];
        double dummy[2];
        for (int64_t k = k0; k + 1 < m; k++) {
            double *vp = vec_ptr(L, k - 1, len), *vc = vec_ptr(L, k, len), *vn = vec_ptr(L, k + 1, len);
            if (L->sym && k >= L->sym_k0)
                lanczos_step_sym(L->sym, c, vp, vc, vn, beta + (k - 1) * c, alpha + k * c, beta + k * c,
                                 dummy, 0, brk);
            else
                lanczos_step(&L->A, c, vp, vc, vn, beta + (k - 1) * c, alpha + k * c, beta + k * c,
                             dummy, 0, brk);
            const double *cf = coef + (k + 1) * c;
            if (c == 1) {
                for (int64_t i = 0; i < n; i++) y[i] += cf[0] * vn[i];
            } else {
                for (int64_t i = 0; i < n; i++) y[i] += cf[0] * vn[2 * i] + cf[1] * vn[2 * i + 1];
            }
        }
        free(coef);
    }
    free(alpha); free(beta); free(vmax); free(S);
    for (int j = 0; j < c; j++) free(u[j]);
    free(tw); free(tth); free(wk);
}

/* ------------------------------------------------------------------ */
/* shift-and-invert Lanczos with a sparse Cholesky factor of L + sigma I */
/* ------------------------------------------------------------------ */
#define MAXSI 100

/* returns number of iterations, or -1 if it failed to converge */
static int si_solve(Lz *L, Chol *C, const double *b0, const double *b1, double *y)
{
    const int64_t n = L->A.n;
    const double t = L->t, sigma = C->sigma;
    const double *bs[2];
    int phi[2];
    double nb[2];
    int c = 0;
    double n0 = 0.0, n1 = 0.0;
    for (int64_t i = 0; i < n; i++) {
        n0 += b0[i] * b0[i];
        n1 += b1[i] * b1[i];
    }
    if (n0 > 0.0) { bs[c] = b0; phi[c] = 0; nb[c] = sqrt(n0); c++; }
    if (n1 > 0.0) { bs[c] = b1; phi[c] = 1; nb[c] = sqrt(n1); c++; }
    memset(y, 0, sizeof(double) * n);
    if (c == 0) return 0;
    const int64_t blk = (int64_t)c * n;
    double *V = big_alloc(sizeof(double) * blk * (MAXSI + 1));
    double *W = big_alloc(sizeof(double) * blk);
    if (!V || !W) _exit(5);
    double alpha[(MAXSI + 1) * 2], beta[(MAXSI + 1) * 2], vmax[(MAXSI + 2) * 2];
    double *S = xmalloc(sizeof(double) * (MAXSI + 2) * L->ns * c);
    double *u[2], *up[2];
    for (int j = 0; j < 2; j++) {
        u[j] = xcalloc(MAXSI + 2, sizeof(double));
        up[j] = xcalloc(MAXSI + 2, sizeof(double));
    }
    double *wk = xmalloc(sizeof(double) * (MAXSI + 2));
    int stopped[2] = {0, 0};
    for (int j = 0; j < c; j++) {
        double inv = 1.0 / nb[j], mx = 0.0;
        double *v = V + j * n;
        for (int64_t i = 0; i < n; i++) {
            v[i] = bs[j][i] * inv;
            if (fabs(v[i]) > mx) mx = fabs(v[i]);
        }
        vmax[j] = mx;
        for (int q = 0; q < L->ns; q++) S[q * c + j] = v[L->samp[q]];
    }
    int64_t m = 0;
    int good = 0, conv = 0;
    double brk = 1e-14 / sigma;
    for (int64_t k = 0; k < MAXSI; k++) {
        double *Vk = V + k * blk, *Vn = V + (k + 1) * blk;
        chol_solve(C, n, c, Vk, W);
        for (int j = 0; j < c; j++) {
            double *w = W + j * n, *vk = Vk + j * n, *vn = Vn + j * n;
            if (stopped[j]) {
                memset(vn, 0, sizeof(double) * n);
                alpha[k * c + j] = 0.0;
                beta[k * c + j] = 0.0;
                vmax[(k + 1) * c + j] = 0.0;
                continue;
            }
            if (k > 0) {
                double bp = beta[(k - 1) * c + j];
                const double *vp = V + (k - 1) * blk + j * n;
                for (int64_t i = 0; i < n; i++) w[i] -= bp * vp[i];
            }
            double al = 0.0;
            for (int64_t i = 0; i < n; i++) al += w[i] * vk[i];
            double nr = 0.0;
            for (int64_t i = 0; i < n; i++) {
                w[i] -= al * vk[i];
                nr += w[i] * w[i];
            }
            double b = sqrt(nr);
            alpha[k * c + j] = al;
            if (!(b > brk)) {
                b = 0.0;
                stopped[j] = 1;
            }
            beta[k * c + j] = b;
            double mx = 0.0;
            if (b > 0.0) {
                double inv = 1.0 / b;
                for (int64_t i = 0; i < n; i++) {
                    vn[i] = w[i] * inv;
                    if (fabs(vn[i]) > mx) mx = fabs(vn[i]);
                }
            } else memset(vn, 0, sizeof(double) * n);
            vmax[(k + 1) * c + j] = mx;
            for (int q = 0; q < L->ns; q++) S[(k + 1) * L->ns * c + q * c + j] = vn[L->samp[q]];
        }
        m = k + 1;
        /* coefficients */
        double est = 0.0, un2 = 0.0;
        for (int j = 0; j < c; j++) {
            int64_t mj = lz_eig(L, alpha, beta, c, j, m);
            const double *mu = L->d, *Z = L->Z;
            for (int64_t q = 0; q < mj; q++) {
                double f = 0.0;
                if (mu[q] > 0.0) {
                    double lam = 1.0 / mu[q] - sigma;
                    f = phi[j] ? fphi(lam, t) : fexp(lam, t);
                }
                wk[q] = f * Z[q * mj];
            }
            int mi = (int)mj, one = 1;
            double alp = nb[j], bet = 0.0;
            dgemv_("N", &mi, &mi, &alp, Z, &mi, wk, &one, &bet, u[j], &one);
            for (int64_t q = mj; q < m; q++) u[j][q] = 0.0;
            for (int64_t q = 0; q < m; q++) {
                est += fabs(u[j][q] - up[j][q]) * vmax[q * c + j];
                un2 += u[j][q] * u[j][q];
                up[j][q] = u[j][q];
            }
        }
        double ys = 0.0;
        for (int q = 0; q < L->ns; q++) {
            double acc = 0.0;
            for (int64_t kk = 0; kk < m; kk++)
                for (int j = 0; j < c; j++) acc += S[kk * L->ns * c + q * c + j] * u[j][kk];
            if (fabs(acc) > ys) ys = fabs(acc);
        }
        double tol = L->atol + L->rtol * fmax(ys, sqrt(un2 / (double)n));
        LOG("  si m=%ld est=%.3e tol=%.3e ratio=%.3e\n", (long)m, est, tol, est / tol);
        conv = (est <= 0.25 * tol) ? conv + 1 : 0;
        int allst = 1;
        for (int j = 0; j < c; j++) if (!stopped[j]) allst = 0;
        if ((m >= 3 && conv >= 2) || allst) { good = 1; break; }
        if (m >= 90) break;
    }
    /* y = sum_k sum_j u_j[k] V_k,j */
    for (int64_t k = 0; k < m; k++)
        for (int j = 0; j < c; j++) {
            double cf = u[j][k];
            if (cf == 0.0) continue;
            const double *v = V + k * blk + j * n;
            for (int64_t i = 0; i < n; i++) y[i] += cf * v[i];
        }
    big_free(V, sizeof(double) * blk * (MAXSI + 1));
    big_free(W, sizeof(double) * blk);
    free(S);
    for (int j = 0; j < 2; j++) { free(u[j]); free(up[j]); }
    free(wk);
    return good ? (int)m : -1;
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    DBG = getenv("EXPO_DEBUG") != NULL;
    if (argc > 1 && strcmp(argv[1], "--selftest") == 0) {
        /* exercise LAPACK linkage on a 3x3 tridiagonal problem */
        double d[3] = {2, 2, 2}, e[2] = {-1, -1}, z[9], wk[64];
        int nn = 3, ld = 3, lw = 64, iw[64], liw = 64, info = 0;
        dstevd_("V", &nn, d, e, z, &ld, wk, &lw, iw, &liw, &info);
        return (info == 0 && fabs(d[0] - (2 - sqrt(2.0))) < 1e-12) ? 0 : 1;
    }
#ifdef F_SETPIPE_SZ
    fcntl(0, F_SETPIPE_SZ, 1 << 20);
    fcntl(1, F_SETPIPE_SZ, 1 << 20);
#endif
    {
        size_t pmb = getenv("EXPO_POOLMB") ? (size_t)atol(getenv("EXPO_POOLMB")) : 64;
        if (!getenv("EXPO_NOPOOL") && pmb > 0) pool_init(pmb << 20);
        if (getenv("EXPO_SMAPS")) {
            FILE *f = fopen("/proc/self/smaps_rollup", "r");
            char line[256];
            while (f && fgets(line, sizeof line, f)) if (strstr(line, "AnonHuge") || strstr(line, "Rss")) fputs(line, stderr);
            if (f) fclose(f);
        }
    }
    {
        /* single-threaded BLAS regardless of the environment */
        void (*ob)(int) = (void (*)(int))dlsym(RTLD_DEFAULT, "openblas_set_num_threads");
        if (ob) ob(1);
    }
    if (getenv("EXPO_SI") && !getenv("EXPO_NOCM")) cm_load();
    if (getenv("EXPO_SIGMA")) SI_SIGMA = atof(getenv("EXPO_SIGMA"));
    if (getenv("EXPO_TRIGGER")) SI_TRIGGER = atof(getenv("EXPO_TRIGGER"));
    write_exact("READY\n", 6);
    char magic[8];
    read_exact(magic, 8);
    if (memcmp(magic, "FAMEXP2\n", 8) != 0) _exit(2);
    uint32_t hl;
    read_exact(&hl, 4);
    char *js = xmalloc(hl + 1);
    read_exact(js, hl);
    js[hl] = 0;
    double T0 = now();
    int ok;
    int64_t n = (int64_t)json_num(js, "n", &ok);
    int64_t nnz = (int64_t)json_num(js, "nnz", &ok);
    int64_t nreq = (int64_t)json_num(js, "requests", &ok);
    double t = json_num(js, "t", &ok);
    double rtol = json_num(js, "rtol", &ok);
    double atol = json_num(js, "atol", &ok);
    double limit = nreq > 1 ? 150.0 : 15.0;

    uint64_t *ip = big_alloc(sizeof(uint64_t) * (n + 1));
    uint32_t *ix = big_alloc(sizeof(uint32_t) * (nnz ? nnz : 1));
    double *a = big_alloc(sizeof(double) * (nnz ? nnz : 1));
    if (!ip || !ix || !a) _exit(5);
    read_exact(ip, sizeof(uint64_t) * (n + 1));
    if (nnz) {
        read_exact(ix, sizeof(uint32_t) * nnz);
        read_exact(a, sizeof(double) * nnz);
    }
    double TR = now();
    Csr A = {n, ip, ix, a};
    double normest = 0.0;
    int diag_only = 1;
    for (int64_t i = 0; i < n; i++) {
        double s = 0.0;
        for (uint64_t k = ip[i]; k < ip[i + 1]; k++) {
            s += fabs(a[k]);
            if (ix[k] != (uint32_t)i && a[k] != 0.0) diag_only = 0;
        }
        if (s > normest) normest = s;
    }
    (void)diag_only;
    int mode; /* 0 trivial, 1 blocks, 2 lanczos, 3 pairs */
    Blocks B;
    Pairs PR;
    Lz L;
    Csr Asub;
    int64_t split_n = 0, *split_idx = NULL, *dgroup = NULL, ndefl = 0;
    int have_blocks = 0;
    double normest_sub = 0.0, *defl_cnt = NULL;
    if (t == 0.0 || normest == 0.0) mode = 0;
    else if (n <= 200) {
        blocks_setup(&A, t, n, 1, &B);
        mode = 1;
    } else if (!getenv("EXPO_NOPAIRS") && pairs_setup(&A, t, &PR)) mode = 3;
    else if (blocks_setup(&A, t, 64, 0, &B)) mode = 1;
    else {
        /* small connected components are solved exactly; Lanczos only sees the
         * large components (their null vectors are delocalised) */
        if (t * normest >= 100.0 && !getenv("EXPO_NOSPLIT")) {
            int64_t *comp = big_alloc(sizeof(int64_t) * n), *queue = big_alloc(sizeof(int64_t) * n);
            int64_t *csize = big_alloc(sizeof(int64_t) * (n + 1));
            int64_t nc = label_components(&A, comp, queue, csize);
            int64_t nsmall = 0;
            for (int64_t i = 0; i < n; i++) if (csize[comp[i]] <= 64) nsmall++;
            /* null-space deflation groups: large components with zero row sums */
            int64_t *gid = big_alloc(sizeof(int64_t) * (nc + 1));
            {
                double *rsum = big_alloc(sizeof(double) * (nc + 1)), *rabs = big_alloc(sizeof(double) * (nc + 1));
                for (int64_t c = 0; c < nc; c++) { rsum[c] = 0.0; rabs[c] = 0.0; gid[c] = 1; }
                for (int64_t i = 0; i < n; i++) {
                    double rs = 0.0, ra = 0.0;
                    for (uint64_t k = ip[i]; k < ip[i + 1]; k++) { rs += a[k]; ra += fabs(a[k]); }
                    if (fabs(rs) > 1e-11 * ra) gid[comp[i]] = 0;
                    (void)rsum; (void)rabs;
                }
                for (int64_t c = 0; c < nc; c++) {
                    if (csize[c] <= 64 || !gid[c]) gid[c] = -1;
                    else gid[c] = ndefl++;
                }
            }
            if (nsmall == 0 && ndefl > 0) {
                /* whole matrix stays; only deflation */
                split_n = n;
                split_idx = NULL;
                Asub = A;
                normest_sub = normest;
            }
            if (ndefl > 0) {
                defl_cnt = big_alloc(sizeof(double) * ndefl);
                for (int64_t g = 0; g < ndefl; g++) defl_cnt[g] = 0.0;
            }
            if (nsmall > 0 && nsmall < n) {
                /* small components: node lists grouped by component */
                int64_t *pos = big_alloc(sizeof(int64_t) * (nc + 1));
                int64_t nsc = 0;
                for (int64_t c = 0; c < nc; c++) pos[c] = -1;
                for (int64_t c = 0; c < nc; c++) if (csize[c] <= 64) pos[c] = nsc++;
                int64_t *sstart = big_alloc(sizeof(int64_t) * (nsc + 1)), *snode = big_alloc(sizeof(int64_t) * nsmall);
                int64_t *fillp = big_alloc(sizeof(int64_t) * (nsc + 1));
                sstart[0] = 0;
                for (int64_t c = 0, q = 0; c < nc; c++) if (pos[c] >= 0) { sstart[q + 1] = sstart[q] + csize[c]; q++; }
                for (int64_t q = 0; q < nsc; q++) fillp[q] = sstart[q];
                for (int64_t i = 0; i < n; i++) if (pos[comp[i]] >= 0) snode[fillp[pos[comp[i]]]++] = i;
                blocks_build(&A, t, snode, sstart, nsc, 0, &B);
                have_blocks = 1;
                /* large part */
                int64_t nl = n - nsmall;
                split_idx = big_alloc(sizeof(int64_t) * nl);
                int64_t *newid = queue;
                int64_t q = 0;
                uint64_t snnz = 0;
                for (int64_t i = 0; i < n; i++) {
                    if (csize[comp[i]] > 64) {
                        newid[i] = q;
                        split_idx[q++] = i;
                        snnz += ip[i + 1] - ip[i];
                    } else newid[i] = -1;
                }
                uint64_t *sip = big_alloc(sizeof(uint64_t) * (nl + 1));
                uint32_t *six = big_alloc(sizeof(uint32_t) * (snnz ? snnz : 1));
                double *sa = big_alloc(sizeof(double) * (snnz ? snnz : 1));
                uint64_t kk = 0;
                double snorm = 0.0;
                for (int64_t r = 0; r < nl; r++) {
                    int64_t i = split_idx[r];
                    sip[r] = kk;
                    double rs = 0.0;
                    for (uint64_t k = ip[i]; k < ip[i + 1]; k++) {
                        six[kk] = (uint32_t)newid[ix[k]];
                        sa[kk] = a[k];
                        rs += fabs(a[k]);
                        kk++;
                    }
                    if (rs > snorm) snorm = rs;
                }
                sip[nl] = kk;
                Asub.n = nl;
                Asub.ip = sip;
                Asub.ix = six;
                Asub.a = sa;
                split_n = nl;
                normest_sub = snorm;
                if (ndefl > 0) {
                    dgroup = big_alloc(sizeof(int64_t) * nl);
                    for (int64_t r = 0; r < nl; r++) {
                        dgroup[r] = gid[comp[split_idx[r]]];
                        if (dgroup[r] >= 0) defl_cnt[dgroup[r]] += 1.0;
                    }
                }
                LOG("split: %ld small-component nodes (%ld comps), %ld in large components\n", (long)nsmall, (long)nsc, (long)nl);
            } else if (nsmall == 0 && ndefl > 0) {
                dgroup = big_alloc(sizeof(int64_t) * n);
                for (int64_t i = 0; i < n; i++) {
                    dgroup[i] = gid[comp[i]];
                    if (dgroup[i] >= 0) defl_cnt[dgroup[i]] += 1.0;
                }
            } else if (nsmall == n) {
                split_n = 0;
            }
            LOG("deflation groups: %ld\n", (long)ndefl);
        }
        /* bandwidth-reducing reordering for long runs */
        int want_rcm = !getenv("EXPO_NORCM") && (t * normest >= 1e4 || (nreq > 1 && t * normest >= 1e3));
        if (want_rcm) {
            /* only for scattered orderings: mean |i-j| over the stored entries */
            const Csr *M0 = split_n > 0 ? &Asub : &A;
            double sd = 0.0;
            uint64_t cntd = 0;
            for (int64_t i = 0; i < M0->n; i++)
                for (uint64_t k = M0->ip[i]; k < M0->ip[i + 1]; k++) {
                    int64_t d = (int64_t)M0->ix[k] - i;
                    if (d) { sd += (double)(d < 0 ? -d : d); cntd++; }
                }
            double md = cntd ? sd / (double)cntd : 0.0;
            LOG("mean |i-j| = %.1f (n=%ld)\n", md, (long)M0->n);
            if (md < 0.05 * (double)M0->n || getenv("EXPO_RCM_NOGATE") != NULL ? md < 0.05 * (double)M0->n && !getenv("EXPO_RCM_NOGATE") : 0) want_rcm = 0;
        }
        if (want_rcm) {
            double q0 = now();
            const Csr *M = split_n > 0 ? &Asub : &A;
            int64_t nl = M->n;
            int64_t *perm = big_alloc(sizeof(int64_t) * nl);
            rcm_order(M, perm);
            Csr P;
            csr_permute(M, perm, &P);
            int64_t *nidx = big_alloc(sizeof(int64_t) * nl);
            for (int64_t r = 0; r < nl; r++)
                nidx[r] = (split_n > 0 && split_idx) ? split_idx[perm[r]] : perm[r];
            if (dgroup) {
                int64_t *ng = big_alloc(sizeof(int64_t) * nl);
                for (int64_t r = 0; r < nl; r++) ng[r] = dgroup[perm[r]];
                dgroup = ng;
            }
            if (split_n == 0) normest_sub = normest;
            split_n = nl;
            split_idx = nidx;
            Asub = P;
            LOG("rcm: %.4fs\n", now() - q0);
        }
        if (split_n > 0) lz_init(&L, &Asub, t, rtol, atol, normest_sub > 0 ? normest_sub : normest);
        else lz_init(&L, &A, t, rtol, atol, normest);
        L.prec_lvl = rtol >= 1e-5 ? 0 : (rtol >= 1e-8 ? 1 : 2);
        L.chol = CM.ok ? xcalloc(1, sizeof(Chol)) : NULL;
        /* shift-and-invert is disabled by default (EXPO_SI=1 enables it) */
        if (!getenv("EXPO_SI")) L.si_state = -1;
        if (getenv("EXPO_NOSYM")) L.nosym = 1;
        mode = 2;
    }
    if (getenv("EXPO_SYMTEST")) {
        Sym SS;
        sym_build(&A, &SS);
        for (int c = 1; c <= 2; c++) {
            double *vp = xmalloc(sizeof(double) * c * n), *vc = xmalloc(sizeof(double) * c * n);
            double *y1 = xmalloc(sizeof(double) * c * n), *y2 = xmalloc(sizeof(double) * c * n);
            srand(1);
            double nv[2] = {0, 0};
            for (int64_t i = 0; i < c * n; i++) { vp[i] = rand() / (double)RAND_MAX - 0.5; vc[i] = rand() / (double)RAND_MAX - 0.5; nv[i % c] += vc[i] * vc[i]; }
            for (int64_t i = 0; i < c * n; i++) vc[i] /= sqrt(nv[i % c]);
            double bp[2] = {0.3, 0.7}, al1[2], be1[2], vm1[2], al2[2], be2[2], vm2[2];
            lanczos_step(&A, c, vp, vc, y1, bp, al1, be1, vm1, 1, 0.0);
            lanczos_step_sym(&SS, c, vp, vc, y2, bp, al2, be2, vm2, 1, 0.0);
            double md = 0;
            for (int64_t i = 0; i < c * n; i++) md = fmax(md, fabs(y1[i] - y2[i]));
            fprintf(stderr, "SYMTEST c=%d maxdiff=%.3e alpha %.6e %.6e beta %.6e %.6e\n", c, md, al1[0], al2[0], be1[0], be2[0]);
        }
    }
    if (getenv("EXPO_CHOLTEST") && CM.ok) {
        Chol C;
        double q0 = now();
        int okA = chol_analyze(&C, &A);
        double q1 = now();
        int okF = okA && chol_factor(&C, 10.0 / t);
        double q2 = now();
        double *Bt = xcalloc(2 * n, sizeof(double)), *Xt = xcalloc(2 * n, sizeof(double));
        for (int64_t i = 0; i < 2 * n; i++) Bt[i] = 1.0;
        chol_solve(&C, n, 1, Bt, Xt);
        double q3 = now();
        chol_solve(&C, n, 2, Bt, Xt);
        double q4 = now();
        fprintf(stderr, "AMDONLY=%d CHOL n=%ld nnz=%ld ok=%d/%d lnz=%ld (%.1f/row) flops=%.3g analyze=%.4f factor=%.4f solve1=%.4f solve2=%.4f\n",
                CM.amd_only, (long)n, (long)nnz, okA, okF, (long)C.lnz, (double)C.lnz / n, C.flops, q1 - q0, q2 - q1, q3 - q2, q4 - q3);
    }
    LOG("n=%ld nnz=%ld t=%g tnorm=%.3g rtol=%g mode=%d read=%.4f setup=%.4f\n", (long)n, (long)nnz, t,
        t * normest, rtol, mode, TR - T0, now() - T0);
    double *sb0 = NULL, *sb1 = NULL, *sy = NULL;
    double *dm0 = ndefl > 0 ? big_alloc(sizeof(double) * ndefl) : NULL;
    double *dm1 = ndefl > 0 ? big_alloc(sizeof(double) * ndefl) : NULL;
    if (split_n > 0) {
        sb0 = big_alloc(sizeof(double) * split_n);
        sb1 = big_alloc(sizeof(double) * split_n);
        sy = big_alloc(sizeof(double) * split_n);
    }
    double *rhs = big_alloc(sizeof(double) * 2 * n);
    double *y = big_alloc(sizeof(double) * n);
    if (!rhs || !y) _exit(5);
    for (int64_t r = 0; r < nreq; r++) {
        read_exact(rhs, sizeof(double) * 2 * n);
        const double *b0 = rhs, *b1 = rhs + n;
        double tr = now();
        if (mode == 0) {
            for (int64_t i = 0; i < n; i++) y[i] = b0[i] + t * b1[i];
        } else if (mode == 1) {
            blocks_solve(&B, b0, b1, y);
        } else if (mode == 3) {
            pairs_solve(&PR, b0, b1, y);
        } else {
            const double *b0x = b0, *b1x = b1;
            double *yx = y;
            int64_t nn = n;
            if (split_n > 0) {
                nn = split_n;
                for (int64_t q = 0; q < nn; q++) {
                    int64_t i = split_idx ? split_idx[q] : q;
                    sb0[q] = b0[i];
                    sb1[q] = b1[i];
                }
                if (ndefl > 0) {
                    for (int64_t g = 0; g < ndefl; g++) { dm0[g] = 0.0; dm1[g] = 0.0; }
                    for (int64_t q = 0; q < nn; q++) {
                        int64_t g = dgroup[q];
                        if (g >= 0) { dm0[g] += sb0[q]; dm1[g] += sb1[q]; }
                    }
                    for (int64_t g = 0; g < ndefl; g++) { dm0[g] /= defl_cnt[g]; dm1[g] /= defl_cnt[g]; }
                    for (int64_t q = 0; q < nn; q++) {
                        int64_t g = dgroup[q];
                        if (g >= 0) { sb0[q] -= dm0[g]; sb1[q] -= dm1[g]; }
                    }
                }
                b0x = sb0;
                b1x = sb1;
                yx = sy;
            }
            double remaining = limit - (now() - T0);
            L.deadline = now() + 0.8 * remaining / (double)(nreq - r);
            L.reqs_left = nreq - r;
            int done = 0;
            if (L.si_state == 1) {
                int it = si_solve(&L, (Chol *)L.chol, b0x, b1x, yx);
                L.si_req = now() - tr;
                LOG("req %ld si iters=%d time=%.4f\n", (long)r, it, L.si_req);
                if (it >= 0) {
                    done = 1;
                    if (L.poly_req > 0.0 && L.si_req > 1.2 * L.poly_req) L.si_state = -1;
                } else L.si_state = -1;
                if (!done) {
                    double rem2 = limit - (now() - T0);
                    L.deadline = now() + 0.8 * rem2 / (double)(nreq - r);
                }
            }
            if (!done) {
                int c_now = 1;
                for (int64_t i = 0; i < nn; i++) if (b0x[i] != 0.0) { c_now = 2; break; }
                double *ysi = NULL;
                int si_it = -2;
                if (L.si_state == 2) {
                    /* validation request: SI result is compared with polynomial Lanczos */
                    ysi = xmalloc(sizeof(double) * nn);
                    double q0 = now();
                    si_it = si_solve(&L, (Chol *)L.chol, b0x, b1x, ysi);
                    L.si_req = now() - q0;
                    double rem2 = limit - (now() - T0);
                    L.deadline = now() + 0.8 * rem2 / (double)(nreq - r);
                }
                double tp = now();
                lz_solve(&L, b0x, b1x, yx);
                L.poly_req = now() - tp;
                if (L.si_state == 2) {
                    double ym = 0.0, dm = 0.0;
                    for (int64_t i = 0; i < nn; i++) {
                        if (fabs(yx[i]) > ym) ym = fabs(yx[i]);
                        double dd = fabs(yx[i] - ysi[i]);
                        if (dd > dm) dm = dd;
                    }
                    double tl = atol + rtol * ym;
                    int okv = si_it >= 0 && dm <= 0.3 * tl && L.si_req < 0.8 * L.poly_req;
                    LOG("si validation: iters=%d diff/tol=%.3e si=%.3f poly=%.3f -> %d\n", si_it, dm / tl, L.si_req, L.poly_req, okv);
                    L.si_state = okv ? 1 : -1;
                    free(ysi);
                } else if (L.si_state == 0) {
                    /* reuse: decide for the remaining requests with the measured cost */
                    int64_t R = nreq - r - 1;
                    if (R >= 2 && (double)R * L.poly_req > 0.4) {
                        si_analyze(&L, c_now);
                        if (L.an_ok && (double)(R - 1) * L.poly_req > 1.3 * ((L.factored > 0 ? 0.0 : L.c_fact) + R * L.c_req)) {
                            if (si_factor(&L)) L.si_state = 2;
                        }
                        if (L.si_state == 0) L.si_state = -1;
                    }
                }
                LOG("req %ld m=%d time=%.4f\n", (long)r, L.last_m, now() - tr);
            }
            if (split_n > 0) {
                for (int64_t q = 0; q < nn; q++) {
                    double v = sy[q];
                    if (ndefl > 0 && dgroup[q] >= 0) v += dm0[dgroup[q]] + t * dm1[dgroup[q]];
                    y[split_idx ? split_idx[q] : q] = v;
                }
                if (have_blocks) blocks_solve(&B, b0, b1, y);
            }
        }
        write_exact(y, sizeof(double) * n);
    }
    LOG("total %.4f\n", now() - T0);
    _exit(0);
}
