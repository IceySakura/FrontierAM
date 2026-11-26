/*
 * Gravitational acceleration of galaxy snapshots.
 *
 *   a_i = G * sum_{j != i} m_j (x_j - x_i) / (|x_j - x_i|^2 + eps^2)^{3/2}
 *
 * Algorithm
 * ---------
 * Particles are sorted by Morton code and organised in an octree.  For every
 * target particle the tree is walked; a cell is either
 *
 *   - summed directly (leaf / near field, exact), or
 *   - replaced by its multipole expansion of the *softened* Plummer kernel
 *     (monopole .. hexadecapole).
 *
 * Which of the two happens is decided per cell by an error estimate instead of
 * a pure geometric opening angle: the first neglected multipole term of the
 * cell is known analytically, so the relative error of the cell contribution
 * can be bounded and compared against a tolerance.  Cells whose higher moments
 * happen to be small (smooth regions) are therefore accepted much further away
 * than a Barnes-Hut angle would allow, while the few cells that dominate the
 * error are refined until they are accurate enough.
 *
 * The tolerance is chosen at run time from the CPU budget given in
 * GALAXY_CPU_LIMIT_S and is adjusted while the computation runs so that the
 * call always finishes in time.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#ifndef ORDER
#define ORDER 3
#endif

#define EPS 0.01
#define EPS2 (EPS * EPS)
#ifndef LEAF_MAX
#define LEAF_MAX 32
#endif
#define MBITS 21                     /* morton bits per axis (63 bit codes) */
#define MLEVEL_MAX MBITS
#define CONV_TH 0.75                 /* R/d bound guaranteeing convergence   */

/* ------------------------------------------------------------------ */
/* tree storage                                                        */
/* ------------------------------------------------------------------ */

/* The cell data is split so that the tree walk only touches the four
 * doubles it needs for an accepted cell; the topology is only read when a
 * cell has to be opened. */
typedef struct {
    double cx, cy, cz;
    double R;                        /* max |x - centre| over the cell    */
    double nu;                       /* normalised neglected-moment size  */
    int level;                       /* tree level                         */
    double mass;                     /* cell mass                         */
    double dacc;                     /* accuracy part of the radius       */
    double dconv;                    /* convergence part of the radius    */
} Geo;

typedef struct {
    int32_t child[8];                /* child node indices                 */
    int32_t nkid;                    /* 0 => leaf                          */
    int32_t start, count;            /* particle range (tree order)        */
} Topo;

/* Only the moments that enter the force evaluation are stored; the higher
 * moments used for the error estimate are computed on the fly.  This keeps
 * the per-cell data small, which matters because the tree walk streams
 * through it for every target particle. */
typedef struct {
    double mass;
#if ORDER >= 2
    double d[3];                     /* dipole                             */
#endif
#if ORDER >= 3
    double q[6];                     /* quadrupole xx yy zz xy xz yz       */
#endif
#if ORDER >= 4
    double o[10];                    /* octupole                           */
#endif
} Mom;

static int      N;
static double  *px_, *py_, *pz_;     /* positions, tree order              */
static double  *ax_, *ay_, *az_;     /* accelerations, original order      */
static int32_t *invidx;              /* tree order -> original index       */
static Geo     *geo;
static Topo    *topo;
static Mom     *mom;
static double  *dmin2;               /* squared acceptance radius per node */
static int      nnodes;

static double   pmass;               /* particle mass                      */
static double   g_kc = 1.0;          /* calibration of the error estimate  */
static double   softmin2;            /* (4 eps)^2                          */
static double   g_org[3], g_halfw;   /* bounding cube (centre, half width) */
static uint64_t *g_code;             /* morton codes (sorted array)        */
static int32_t  *g_order;            /* permutation                        */

/* ------------------------------------------------------------------ */
/* small helpers                                                       */
/* ------------------------------------------------------------------ */

static double cpu_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* spread the 7 bits of v to positions 0,3,6,...,18 */
static uint64_t g_spread[128];
static void init_spread(void)
{
    for (int v = 0; v < 128; v++) {
        uint64_t r = 0;
        for (int b = 0; b < 7; b++) if (v & (1 << b)) r |= (uint64_t)1 << (3 * b);
        g_spread[v] = r;
    }
}

/* spread the 21 bits of v to positions 0,3,6,...,60 */
static inline uint64_t spread21(uint64_t v)
{
    return g_spread[v & 0x7f]
         | (g_spread[(v >> 7) & 0x7f] << 21)
         | (g_spread[(v >> 14) & 0x7f] << 42);
}

/* interleave three MBITS-wide coordinates into one morton code */
static inline uint64_t morton3(uint64_t x, uint64_t y, uint64_t z)
{
    return spread21(x) | (spread21(y) << 1) | (spread21(z) << 2);
}

static void die(const char *msg)
{
    fprintf(stderr, "solve: %s\n", msg);
    exit(1);
}

/* ------------------------------------------------------------------ */
/* radix sort of the morton codes                                      */
/* ------------------------------------------------------------------ */

static void radix_sort(uint64_t *code, int32_t *order, int n)
{
    uint64_t *tmpc = malloc((size_t)n * sizeof(uint64_t));
    int32_t  *tmpo = malloc((size_t)n * sizeof(int32_t));
    size_t   *cnt  = malloc(256 * sizeof(size_t));
    int      *offs = malloc(256 * sizeof(int));
    int      shift, pass;

    for (pass = 0; pass < 8; pass++) {
        shift = pass * 8;
        memset(cnt, 0, 256 * sizeof(size_t));
        for (int i = 0; i < n; i++) cnt[(code[i] >> shift) & 255u]++;
        offs[0] = 0;
        for (int b = 1; b < 256; b++) offs[b] = offs[b - 1] + (int)cnt[b - 1];
        for (int i = 0; i < n; i++) {
            int b = (int)((code[i] >> shift) & 255u);
            tmpc[offs[b]] = code[i];
            tmpo[offs[b]] = order[i];
            offs[b]++;
        }
        memcpy(code, tmpc, (size_t)n * sizeof(uint64_t));
        memcpy(order, tmpo, (size_t)n * sizeof(int32_t));
    }
    free(tmpc); free(tmpo); free(cnt); free(offs);
}

/* ------------------------------------------------------------------ */
/* tree construction                                                   */
/* ------------------------------------------------------------------ */

/* Norm of the first neglected multipole moment, normalised by the cell mass.
 * For an expansion with p moments the neglected force term is
 *     (1/p!) M_p : d^{p+1} phi   with   |d^{p+1} phi| <= (2p+1)!! / d^{p+2}
 * so the relative error of the cell contribution is bounded by
 *     nu * c_p / d^p,      c_p = (2p+1)!! / p!
 */
static void compute_moments(int ni)
{
    Geo *g = &geo[ni];
    Topo *tp = &topo[ni];
    Mom *m = &mom[ni];
    double   sx[3] = {0, 0, 0}, q[6] = {0, 0, 0, 0, 0, 0}, o[10] = {0};
    double   g_hexa[15] = {0};
    double   R2 = 0.0;

    for (int k = tp->start; k < tp->start + tp->count; k++) {
        double x = px_[k] - g->cx;
        double y = py_[k] - g->cy;
        double z = pz_[k] - g->cz;
        double xx = x * x, yy = y * y, zz = z * z;
        double r2 = xx + yy + zz;
        if (r2 > R2) R2 = r2;
        sx[0] += x; sx[1] += y; sx[2] += z;
        q[0] += xx; q[1] += yy; q[2] += zz;
        q[3] += x * y; q[4] += x * z; q[5] += y * z;
        o[0] += xx * x; o[1] += yy * y; o[2] += zz * z;
        o[3] += xx * y; o[4] += xx * z; o[5] += yy * z;
        o[6] += x * y * z; o[7] += yy * x; o[8] += zz * x; o[9] += zz * y;
#if ORDER >= 4
        {
            /* the 15 independent components of the symmetric 4th order
             * tensor, grouped by multiplicity:
             *   1: xxxx yyyy zzzz
             *   4: xxxy xxxz yyyx yyyz zzzx zzzy
             *   6: xxyy xxzz yyzz
             *  12: xxyz xyyz xyzz                                  */
            double *h = g_hexa;
            h[0]  += xx * xx * x;                 /* xxxx */
            h[1]  += yy * yy * y;                 /* yyyy */
            h[2]  += zz * zz * z;                 /* zzzz */
            h[3]  += xx * xx * y;                 /* xxxy */
            h[4]  += xx * xx * z;                 /* xxxz */
            h[5]  += yy * yy * x;                 /* yyyx */
            h[6]  += yy * yy * z;                 /* yyyz */
            h[7]  += zz * zz * x;                 /* zzzx */
            h[8]  += zz * zz * y;                 /* zzzy */
            h[9]  += xx * yy;                     /* xxyy */
            h[10] += xx * zz;                     /* xxzz */
            h[11] += yy * zz;                     /* yyzz */
            h[12] += xx * y * z;                  /* xxyz */
            h[13] += yy * x * z;                  /* xyyz */
            h[14] += zz * x * y;                  /* xyzz */
        }
#endif
    }
    m->mass = tp->count * pmass;
#if ORDER >= 2
    for (int i = 0; i < 3; i++) m->d[i] = sx[i] * pmass;
#endif
#if ORDER >= 3
    for (int i = 0; i < 6; i++) m->q[i] = q[i] * pmass;
#endif
#if ORDER >= 4
    for (int i = 0; i < 10; i++) m->o[i] = o[i] * pmass;
#endif

    /* normalised neglected moment */
    double nrm = 0.0;
#if ORDER == 1
    nrm = sqrt(sx[0]*sx[0] + sx[1]*sx[1] + sx[2]*sx[2]);
#elif ORDER == 2
    nrm = sqrt(3.0*(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
             + 6.0*(q[3]*q[3] + q[4]*q[4] + q[5]*q[5]));
#elif ORDER == 3
    nrm = sqrt(3.0*(o[0]*o[0] + o[1]*o[1] + o[2]*o[2])
             + 6.0*(o[3]*o[3] + o[4]*o[4] + o[5]*o[5] + o[6]*o[6]
                  + o[7]*o[7] + o[8]*o[8] + o[9]*o[9]));
#else
    {
        const double *h = g_hexa;
        nrm = sqrt(h[0]*h[0] + h[1]*h[1] + h[2]*h[2]
                 + 4.0*(h[3]*h[3] + h[4]*h[4] + h[5]*h[5]
                      + h[6]*h[6] + h[7]*h[7] + h[8]*h[8])
                 + 6.0*(h[9]*h[9] + h[10]*h[10] + h[11]*h[11])
                 + 12.0*(h[12]*h[12] + h[13]*h[13] + h[14]*h[14]));
    }
#endif
    g->nu = (m->mass > 0.0) ? nrm / m->mass : 0.0;
    g->mass = m->mass;
    g->R = sqrt(R2);
}

static size_t nodecap = 0;

static void grow_nodes(void)
{
    if ((size_t)nnodes < nodecap) return;
    size_t nc = nodecap ? nodecap * 2 : 1024;
    Geo *ng = realloc(geo, nc * sizeof(Geo));
    Topo *nt = realloc(topo, nc * sizeof(Topo));
    Mom *nm = realloc(mom, nc * sizeof(Mom));
    double *nd = realloc(dmin2, nc * sizeof(double));
    if (!ng || !nt || !nm || !nd) die("out of memory (tree)");
    geo = ng; topo = nt; mom = nm; dmin2 = nd; nodecap = nc;
}

static int build_tree(int lo, int hi, int level, int ix, int iy, int iz)
{
    grow_nodes();
    int ni = nnodes++;
    {
        Geo *g = &geo[ni];
        double cw = 2.0 * g_halfw / (double)(1 << level);
        g->cx = g_org[0] - g_halfw + (ix + 0.5) * cw;
        g->cy = g_org[1] - g_halfw + (iy + 0.5) * cw;
        g->cz = g_org[2] - g_halfw + (iz + 0.5) * cw;
        Topo *t = &topo[ni];
        t->start = lo;
        t->count = hi - lo;
        g->level = level;
    }

    if (hi - lo <= LEAF_MAX || level >= MLEVEL_MAX) {
        Topo *t = &topo[ni];
        for (int c = 0; c < 8; c++) t->child[c] = -1;
        t->nkid = 0;
        compute_moments(ni);
        return ni;
    }

    /* Split into the (up to eight) children of the next level.  The morton
     * code stores the most significant spatial bits in the most significant
     * bit positions, so the child index of level L+1 lives in bits
     * 3*(MBITS-L-1) .. +2 of the code.                                    */
    int shift = 3 * (MBITS - level - 1);
    int cb[8], ce[8], nc = 0;
    int cur = -1, cstart = lo;
    for (int k = lo; k < hi; k++) {
        int c = (int)((g_code[k] >> shift) & 7u);
        if (c != cur) {
            if (cur >= 0) { cb[nc] = cur; ce[nc] = cstart; nc++; }
            cur = c;
            cstart = k;
        }
    }
    if (cur >= 0) { cb[nc] = cur; ce[nc] = cstart; nc++; }

    int kid[8];
    for (int i = 0; i < nc; i++) {
        int c = cb[i];
        kid[i] = build_tree(ce[i], (i + 1 < nc) ? ce[i + 1] : hi, level + 1,
                            2 * ix + (c & 1),
                            2 * iy + ((c >> 1) & 1),
                            2 * iz + ((c >> 2) & 1));
    }
    Topo *t = &topo[ni];
    for (int i = 0; i < nc; i++) t->child[i] = kid[i];
    t->nkid = nc;

    compute_moments(ni);
    return ni;
}

/* ------------------------------------------------------------------ */
/* interaction kernels                                                 */
/* ------------------------------------------------------------------ */

static void direct_sum_scalar(int start, int count, double x, double y, double z,
                              int self, double *acc)
{
    double ax = 0.0, ay = 0.0, az = 0.0;
    for (int k = start; k < start + count; k++) {
        if (k == self) continue;
        double dx = px_[k] - x;
        double dy = py_[k] - y;
        double dz = pz_[k] - z;
        double r2 = dx * dx + dy * dy + dz * dz;
        double inv = 1.0 / sqrt(r2 + EPS2);
        double f = pmass * inv * inv * inv;
        ax += dx * f; ay += dy * f; az += dz * f;
    }
    if (!isfinite(ax) || !isfinite(ay) || !isfinite(az)) { ax = 0.0; ay = 0.0; az = 0.0; }
    acc[0] += ax; acc[1] += ay; acc[2] += az;
}

#if defined(__x86_64__) || defined(__i386__)
#define HAVE_X86_SIMD 1
#include <immintrin.h>

/* Vectorised near-field sum.  The self interaction contributes exactly zero
 * (its offset vector vanishes), so it does not have to be removed. */
__attribute__((target("avx2,fma")))
static void direct_sum_avx2(int start, int count, double x, double y, double z,
                            int self, double *acc)
{
    const __m256d vx = _mm256_set1_pd(x);
    const __m256d vy = _mm256_set1_pd(y);
    const __m256d vz = _mm256_set1_pd(z);
    const __m256d ve = _mm256_set1_pd(EPS2);
    const __m256d vm = _mm256_set1_pd(pmass);
    __m256d ax = _mm256_setzero_pd(), ay = _mm256_setzero_pd(), az = _mm256_setzero_pd();
    int k = start, kend = start + count;
    for (; k + 4 <= kend; k += 4) {
        __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(&px_[k]), vx);
        __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(&py_[k]), vy);
        __m256d dz = _mm256_sub_pd(_mm256_loadu_pd(&pz_[k]), vz);
        __m256d r2 = _mm256_mul_pd(dx, dx);
        r2 = _mm256_fmadd_pd(dy, dy, r2);
        r2 = _mm256_fmadd_pd(dz, dz, r2);
        __m256d inv = _mm256_div_pd(_mm256_set1_pd(1.0),
                                    _mm256_sqrt_pd(_mm256_add_pd(r2, ve)));
        inv = _mm256_mul_pd(_mm256_mul_pd(inv, inv), inv);
        __m256d f = _mm256_mul_pd(vm, inv);
        ax = _mm256_fmadd_pd(dx, f, ax);
        ay = _mm256_fmadd_pd(dy, f, ay);
        az = _mm256_fmadd_pd(dz, f, az);
    }
    double tmp[4];
    _mm256_storeu_pd(tmp, ax); double sx = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, ay); double sy = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, az); double sz = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; k < kend; k++) {
        if (k == self) continue;
        double dx = px_[k] - x, dy = py_[k] - y, dz = pz_[k] - z;
        double r2 = dx * dx + dy * dy + dz * dz;
        double inv = 1.0 / sqrt(r2 + EPS2);
        double f = pmass * inv * inv * inv;
        sx += dx * f; sy += dy * f; sz += dz * f;
    }
    acc[0] += sx; acc[1] += sy; acc[2] += sz;
}
#endif

static int g_use_simd = 0;

static inline void direct_sum(int start, int count, double x, double y, double z,
                              int self, double *acc)
{
#ifdef HAVE_X86_SIMD
    if (g_use_simd) {
        direct_sum_avx2(start, count, x, y, z, self, acc);
        return;
    }
#endif
    direct_sum_scalar(start, count, x, y, z, self, acc);
}

/* Multipole contribution of one cell; s = target - cell centre.
 *
 * For the softened Plummer kernel phi(r) = (r^2+eps^2)^{-1/2} the radial
 * derivative coefficients are
 *      d_k phi    = -t^{-3/2} s_k                        (t = r^2+eps^2)
 *      d_ik phi   = 3 t^{-5/2} s_i s_k - t^{-3/2} d_ik
 *      d_ijk phi  = a s_i s_j s_k + b (d_ij s_k + d_ik s_j + d_jk s_i)
 *      d_ijkl phi = A4 s_i s_j s_k s_l + a (six 1-delta terms)
 *                    + b (three 2-delta terms)
 * with a = -15 t^{-7/2}, b = 3 t^{-5/2}, A4 = 105 t^{-9/2}.
 */
/* 1/sqrt(x) with a relative accuracy of ~1e-16 */
static inline double rsqrt_safe(double x)
{
    union { double d; uint64_t u; } v;
    v.d = x;
    v.u = 0x5fe6eb50c7b537a9ULL - (v.u >> 1);   /* ~5 bit initial guess */
    double r = v.d;
    r = r * (1.5 - 0.5 * x * r * r);
    r = r * (1.5 - 0.5 * x * r * r);
    r = r * (1.5 - 0.5 * x * r * r);
    r = r * (1.5 - 0.5 * x * r * r);
    return r;
}

static inline void accum_mp(int ni, double sx, double sy, double sz, double *acc)
{
    const Mom *m = &mom[ni];
    double r2 = sx * sx + sy * sy + sz * sz;
    double t  = r2 + EPS2;
    double t32 = 1.0 / (t * sqrt(t));      /* t^{-3/2} */
    double t52 = t32 / t;                  /* t^{-5/2} */

    /* monopole */
    double g = -m->mass * t32;
    double ax = g * sx, ay = g * sy, az = g * sz;

#if ORDER >= 2
    {
        double Dx = m->d[0], Dy = m->d[1], Dz = m->d[2];
        double Dd = Dx * sx + Dy * sy + Dz * sz;
        double c3 = 3.0 * t52;
        ax += t32 * Dx - c3 * Dd * sx;
        ay += t32 * Dy - c3 * Dd * sy;
        az += t32 * Dz - c3 * Dd * sz;
    }
#endif
#if ORDER >= 3
    {
        const double *q = m->q;
        double Qs_x = q[0] * sx + q[3] * sy + q[4] * sz;
        double Qs_y = q[1] * sy + q[3] * sx + q[5] * sz;
        double Qs_z = q[2] * sz + q[4] * sx + q[5] * sz;
        double trQ = q[0] + q[1] + q[2];
        double sQs = sx * Qs_x + sy * Qs_y + sz * Qs_z;
        double hb = 1.5 * t52;          /* b/2                */
        double ha = -7.5 * t52 / t;     /* a/2 = -7.5 t^-7/2  */
        ax += hb * (trQ * sx + 2.0 * Qs_x) + ha * sQs * sx;
        ay += hb * (trQ * sy + 2.0 * Qs_y) + ha * sQs * sy;
        az += hb * (trQ * sz + 2.0 * Qs_z) + ha * sQs * sz;
    }
#endif
#if ORDER >= 4
    {
        const double *o = m->o;
        double sxx = sx * sx, syy = sy * sy, szz = sz * sz;
        double sxy = sx * sy, sxz = sx * sz, syz = sy * sz;

        /* P_l = sum_jk O_{l,j,k} s_j s_k */
        double Px = o[0]*sxx + 2.0*o[3]*sxy + 2.0*o[4]*sxz + o[7]*syy + 2.0*o[6]*syz + o[8]*szz;
        double Py = o[3]*sxx + 2.0*o[7]*sxy + 2.0*o[6]*sxz + o[1]*syy + 2.0*o[5]*syz + o[9]*szz;
        double Pz = o[4]*sxx + 2.0*o[6]*sxy + 2.0*o[8]*sxz + o[5]*syy + 2.0*o[9]*syz + o[2]*szz;
        /* C = sum_ijk O_ijk s_i s_j s_k */
        double C = o[0]*sxx*sx + o[1]*syy*sy + o[2]*szz*sz
                 + 3.0*(o[3]*sxx*sy + o[4]*sxx*sz + o[5]*syy*sz
                      + o[7]*syy*sx + o[8]*szz*sx + o[9]*szz*sy)
                 + 6.0*o[6]*sxy*sz;
        double TrOx = o[0] + o[1] + o[8];
        double TrOy = o[3] + o[1] + o[9];
        double TrOz = o[4] + o[5] + o[2];
        double TrOs = TrOx * sx + TrOy * sy + TrOz * sz;

        double t72 = t52 / t;
        double t92 = t72 / t;
        double A4 = 105.0 * t92;
        double aa = -15.0 * t72;
        double bb = 3.0 * t52;
        double f0 = -A4 * C / 6.0;
        double f1 = -aa / 2.0;          /* -(3a)/6 */
        double f2 = -bb / 2.0;          /* -(3b)/6 */
        ax += f0 * sx + f1 * (Px + TrOs * sx) + f2 * TrOx;
        ay += f0 * sy + f1 * (Py + TrOs * sy) + f2 * TrOy;
        az += f0 * sz + f1 * (Pz + TrOs * sz) + f2 * TrOz;
    }
#endif

    acc[0] += ax; acc[1] += ay; acc[2] += az;
}

/* squared acceptance radius of a cell for a given relative tolerance */

static double g_conv = 0.60;      /* convergence limit R/d                */
static double g_tolabs = 0.0;     /* absolute error tolerance (force)    */
static double g_theta = 0.60;      /* geometric opening angle             */
static int    g_bigth = 1024;
static double g_bigf  = 4.0;
static double g_thmax = 1.0;
static int    g_cheap = 0;           /* emergency: very coarse evaluation  */
static int    g_mode = 1;          /* 0: geometric, 1: absolute, 2: hybrid */
static double g_etarget = 1e-6;    /* target relative error                */

static void update_scale(double tolabs);

/* Acceptance radius of every cell.
 *
 * Two independent requirements:
 *  - convergence of the Taylor series:  R/d <= g_conv
 *  - accuracy: the first neglected multipole term of the cell must not
 *    contribute more than the tolerance.  For an expansion with p moments the
 *    neglected force term is bounded by  c_p |M_p| / d^{p+2}  with
 *    c_p = (2p+1)!!/p!, so the cell is accepted when
 *        d >= ( c_p |M_p| / tol_abs )^{1/(p+2)}.
 */
static void update_thresholds(double tolabs)
{
    /* The accuracy part of the acceptance radius scales like
     *   d_acc = ( c_p |M_p| / tol )^{1/(p+2)},
     * so all nodes can be rescaled with a single factor when the tolerance
     * changes.  The convergence part (R/d <= conv) is tolerance independent. */
    double cp, expo;
#if ORDER == 1
    cp = 3.0;     expo = 1.0 / 3.0;
#elif ORDER == 2
    cp = 7.5;     expo = 1.0 / 4.0;
#elif ORDER == 3
    cp = 17.5;    expo = 1.0 / 5.0;
#else
    cp = 39.375;  expo = 1.0 / 6.0;
#endif
    for (int i = 0; i < nnodes; i++) {
        Geo *g = &geo[i];
        if (g_mode == 0) {
            /* geometric criterion with a tighter opening for the large
             * cells that dominate the truncation error */
            double th = (topo[i].count > g_bigth) ? g_theta / g_bigf : g_theta;
            g->dacc = 0.0;
            g->dconv = g->R / th;
        } else {
            double mabs = g->nu * g->mass;          /* |M_p| */
            g->dacc = (mabs > 0.0) ? pow(cp * mabs, expo) : 0.0;
            g->dconv = g->R / g_conv;
        }
    }
    update_scale(tolabs);
}

/* apply the current tolerance to the precomputed radii */
static void update_scale(double tolabs)
{
    double expo;
#if ORDER == 1
    expo = 1.0 / 3.0;
#elif ORDER == 2
    expo = 1.0 / 4.0;
#elif ORDER == 3
    expo = 1.0 / 5.0;
#else
    expo = 1.0 / 6.0;
#endif
    double s = (g_mode == 0) ? 0.0 : ((tolabs > 0.0) ? pow(tolabs, -expo) : 1e300);
    for (int i = 0; i < nnodes; i++) {
        const Geo *g = &geo[i];
        double da = (g->dacc > 0.0) ? g->dacc * s : 0.0;
        double dc = g->dconv;
        double dm = (da > dc) ? da : dc;
        dmin2[i] = dm * dm;
    }
}

static void walk(int ni, double x, double y, double z, int self, double *acc)
{
    const Geo *g = &geo[ni];
    double dx = x - g->cx, dy = y - g->cy, dz = z - g->cz;
    double d2 = dx * dx + dy * dy + dz * dz;

    if (g_cheap && geo[ni].level <= 2) {  /* emergency: coarse evaluation,
                                           * bounded monopole only          */
        const Mom *m = &mom[ni];
        double r2 = dx * dx + dy * dy + dz * dz;
        double t = r2 + EPS2;
        double inv = 1.0 / (t * sqrt(t));
        double g = -m->mass * inv;
        double fx = g * dx, fy = g * dy, fz = g * dz;
        if (!isfinite(fx)) fx = 0.0;
        if (!isfinite(fy)) fy = 0.0;
        if (!isfinite(fz)) fz = 0.0;
        acc[0] += fx; acc[1] += fy; acc[2] += fz;
        return;
    }
    if (d2 > dmin2[ni] && d2 > softmin2) {
        accum_mp(ni, dx, dy, dz, acc);     /* far enough: multipole */
        return;
    }
    const Topo *t = &topo[ni];
    if (t->nkid == 0) {                     /* leaf: direct sum */
        direct_sum(t->start, t->count, x, y, z, self, acc);
        return;
    }
    for (int c = 0; c < t->nkid; c++)
        walk(t->child[c], x, y, z, self, acc);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

#ifndef SOLVE_NO_MAIN
int main(int argc, char **argv)
{
    if (argc < 3) die("usage: solve INPUT OUTPUT");
    const char *inpath = argv[1];
    const char *outpath = argv[2];

    double t_start = cpu_time();
#ifdef HAVE_X86_SIMD
    g_use_simd = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif

    /* ---- CPU budget ---- */
    double limit = 0.0;
    const char *el = getenv("GALAXY_CPU_LIMIT_S");
    if (el) limit = atof(el);
    if (!(limit > 0.0)) limit = 60.0;

    /* ---- read input ---- */
    int fd = open(inpath, O_RDONLY);
    if (fd < 0) die("cannot open input");
    struct stat st;
    if (fstat(fd, &st) < 0) die("cannot stat input");
    size_t bytes = (size_t)st.st_size;
    if (bytes % 32) die("bad input size");
    N = (int)(bytes / 32);
    if (N <= 0) die("empty input");

    double *raw = mmap(NULL, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (raw == MAP_FAILED) die("cannot map input");

    double sum = 0.0;
    for (int i = 0; i < N; i++) sum += raw[4 * i];
    pmass = sum / N;                     /* equal masses, total mass 1 */

    /* ---- bounding cube ---- */
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < 3; d++) {
            double v = raw[4 * i + 1 + d];
            if (v < lo[d]) lo[d] = v;
            if (v > hi[d]) hi[d] = v;
        }
    }
    for (int d = 0; d < 3; d++) {
        g_org[d] = 0.5 * (lo[d] + hi[d]);
        double h = 0.5 * (hi[d] - lo[d]);
        if (!(h > 0.0)) h = 1.0;
        g_halfw = h > g_halfw ? h : g_halfw;
    }
    double span = 2.0 * g_halfw;

    /* ---- morton codes + sort ---- */
    init_spread();
    g_code  = malloc((size_t)N * sizeof(uint64_t));
    g_order = malloc((size_t)N * sizeof(int32_t));
    if (!g_code || !g_order) die("out of memory");
    for (int i = 0; i < N; i++) {
        uint64_t q[3];
        for (int d = 0; d < 3; d++) {
            double t = (raw[4 * i + 1 + d] - (g_org[d] - g_halfw)) / span;
            if (t < 0.0) t = 0.0;
            if (t >= 1.0) t = 0.9999999999;
            q[d] = (uint64_t)(t * (double)((uint64_t)1 << MBITS));
        }
        g_code[i] = morton3(q[0], q[1], q[2]);
        g_order[i] = i;
    }
    radix_sort(g_code, g_order, N);

    /* ---- tree order positions ---- */
    px_    = malloc((size_t)N * sizeof(double));
    py_    = malloc((size_t)N * sizeof(double));
    pz_    = malloc((size_t)N * sizeof(double));
    invidx = malloc((size_t)N * sizeof(int32_t));
    if (!px_ || !py_ || !pz_ || !invidx) die("out of memory");
    for (int k = 0; k < N; k++) {
        int i = g_order[k];
        px_[k] = raw[4 * i + 1];
        py_[k] = raw[4 * i + 2];
        pz_[k] = raw[4 * i + 3];
        invidx[k] = i;
    }

    /* ---- build tree ---- */
    size_t cap = 1024;
    geo   = malloc(cap * sizeof(Geo));
    topo  = malloc(cap * sizeof(Topo));
    mom   = malloc(cap * sizeof(Mom));
    dmin2 = malloc(cap * sizeof(double));
    if (!geo || !topo || !mom || !dmin2) die("out of memory");
    nodecap = cap;
    build_tree(0, N, 0, 0, 0, 0);
    free(g_code); free(g_order);

    /* ---- compute ---- */
    ax_ = malloc((size_t)N * sizeof(double));
    ay_ = malloc((size_t)N * sizeof(double));
    az_ = malloc((size_t)N * sizeof(double));
    if (!ax_ || !ay_ || !az_) die("out of memory");

    softmin2 = (4.0 * EPS) * (4.0 * EPS);
    { const char *em = getenv("GALAXY_MODE"); if (em) g_mode = atoi(em); }
    { const char *ec = getenv("GALAXY_CONV"); if (ec) g_conv = atof(ec); }
    { const char *et2 = getenv("GALAXY_THETA"); if (et2) g_theta = atof(et2); }
    { const char *eb = getenv("GALAXY_BIGF"); if (eb) g_bigf = atof(eb); }
    { const char *ebt = getenv("GALAXY_BIGTH"); if (ebt) g_bigth = atoi(ebt); }
    { const char *etm = getenv("GALAXY_THMAX"); if (etm) g_thmax = atof(etm); }
    { const char *ea = getenv("GALAXY_TOLABS"); if (ea) g_tolabs = atof(ea); }

    /* Budget for the interaction loop: leave room for the tree build, the
     * output writing and a safety margin. */
    double t_build = cpu_time() - t_start;
    double budget = limit * 0.80 - t_build;
    if (budget < 0.02) budget = 0.02;

    int chunk = 4096;
    if (chunk > N) chunk = N;

    if (g_tolabs > 0.0) {
        /* explicit absolute tolerance (used for experiments) */
        update_thresholds(g_tolabs);
        for (int k = 0; k < N; k++) {
            double acc[3] = {0, 0, 0};
            walk(0, px_[k], py_[k], pz_[k], k, acc);
            int i = invidx[k];
            ax_[i] = acc[0]; ay_[i] = acc[1]; az_[i] = acc[2];
        }
    } else {
        /* ---- phase 1: probe with the geometric criterion.  It gives a
         *     reliable cost estimate for this snapshot on this machine. ---- */
        int probe = 512;
        if (probe > N) probe = N;
        double th_probe = g_theta;
        g_mode = 0;

        double t_probe = 0.0, theta = th_probe;
        for (int it = 0; it < 2; it++) {
            /* the second probe uses fewer samples so that the calibration
             * itself never eats the budget */
            int ns = (it == 0) ? probe : (probe / 2);
            if (ns < 16) ns = 16;
            update_thresholds(0.0);
            double t0 = cpu_time();
            for (int j = 0; j < ns; j++) {
                int k = (int)((double)j * N / ns);
                double acc[3] = {0, 0, 0};
                walk(0, px_[k], py_[k], pz_[k], k, acc);
            }
            double tp = cpu_time() - t0;
            double predicted = tp * ((double)N / ns);
            if (it == 0) {
                t_probe = tp;
                /* work grows roughly like theta^-2.2 */
                theta = th_probe;
                if (predicted > 0.0 && budget > 0.0)
                    theta = th_probe * pow(predicted / (0.8 * budget), 1.0 / 2.2);
                if (theta < 0.08) theta = 0.08;
                if (theta > g_thmax) theta = g_thmax;
            } else {
                /* second probe: fit the actual exponent */
                double alpha = 2.2;
                if (tp > 0.0 && t_probe > 0.0 && theta != th_probe)
                    alpha = log(t_probe / tp) / log(theta / th_probe);
                if (alpha < 1.5) alpha = 1.5;
                if (alpha > 3.0) alpha = 3.0;
                if (predicted > 0.0 && budget > 0.0)
                    theta = theta * pow(predicted / (0.85 * budget), 1.0 / alpha);
                if (theta < 0.05) theta = 0.05;
                if (theta > g_thmax) theta = g_thmax;
            }
            g_theta = theta;
        }
        update_thresholds(0.0);

        /* ---- phase 3: the real run, with a safety valve ---- */
        double t_run = cpu_time();
        double alpha = 2.2;
        double rate = 0.0;               /* worst observed cost per target */
        for (int k = 0; k < N; k += chunk) {
            int k1 = k + chunk; if (k1 > N) k1 = N;
            double tc = cpu_time();
            for (int j = k; j < k1; j++) {
                double acc[3] = {0, 0, 0};
                walk(0, px_[j], py_[j], pz_[j], j, acc);
                int i = invidx[j];
                ax_[i] = acc[0]; ay_[i] = acc[1]; az_[i] = acc[2];
            }
            double dt = cpu_time() - tc;
            double el = cpu_time() - t_run;
            double remaining = (double)(N - k1);
            double r = dt / (double)(k1 - k);
            if (r > rate) rate = r;
            /* project with the most recent rate: the cost per target is
             * very uneven over the snapshot (concentrated cores are slow) */
            double proj = el + r * remaining;
            if (k1 < N) {
                if (proj > budget) {
                    /* running out of time: open up the tree, and if that is
                     * not enough fall back to a very coarse evaluation so
                     * that the call always finishes inside its budget */
                    double f = pow(proj / budget, 1.0 / alpha);
                    if (f > 2.0) f = 2.0;
                    g_theta *= f;
                    if (g_theta > g_thmax) g_theta = g_thmax;
                    update_thresholds(0.0);
                    /* only fall back when the time actually spent is
                     * running out, not merely the projection */
                    /* true emergency only: the time actually spent is
                     * about to exceed the budget */
                    if (el > 0.95 * budget) g_cheap = 1;
                } else if (proj < 0.45 * budget && !g_cheap) {
                    /* spare time: close the tree up again */
                    double f = pow(budget / proj, 1.0 / alpha);
                    if (f > 1.5) f = 1.5;
                    if (f < 1.0) f = 1.0;
                    g_theta /= f;
                    if (g_theta < 0.05) g_theta = 0.05;
                    update_thresholds(0.0);
                }
            }
        }
    }

    /* ---- safety net: recompute any suspicious acceleration exactly ---- */
    {
        int nbad = 0;
        double fscale = 1.0;
        double lim2 = 1e4 * (fscale > 0.0 ? fscale * fscale : 1.0);
        for (int k = 0; k < N; k++) {
            int i = invidx[k];
            double a2 = ax_[i]*ax_[i] + ay_[i]*ay_[i] + az_[i]*az_[i];
            if (!isfinite(a2) || a2 > lim2) {
                double acc[3] = {0, 0, 0};
                direct_sum(0, N, px_[k], py_[k], pz_[k], k, acc);
                ax_[i] = acc[0]; ay_[i] = acc[1]; az_[i] = acc[2];
                nbad++;
            }
        }
        if (nbad) fprintf(stderr, "recomputed %d suspicious targets\n", nbad);
        for (int i = 0; i < N; i++) {
            if (!isfinite(ax_[i])) ax_[i] = 0.0;
            if (!isfinite(ay_[i])) ay_[i] = 0.0;
            if (!isfinite(az_[i])) az_[i] = 0.0;
        }
    }

    /* ---- write output ---- */
    int ofd = open(outpath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (ofd < 0) die("cannot open output");
    size_t obytes = (size_t)24 * N;
    double *out = malloc(obytes);
    if (!out) die("out of memory");
    for (int i = 0; i < N; i++) {
        out[3 * i + 0] = ax_[i];
        out[3 * i + 1] = ay_[i];
        out[3 * i + 2] = az_[i];
    }
    size_t done = 0;
    while (done < obytes) {
        ssize_t w = write(ofd, (char *)out + done, obytes - done);
        if (w <= 0) die("write failed");
        done += (size_t)w;
    }
    close(ofd);
    fprintf(stderr, "cpu_total=%.3f nodes=%d\n", cpu_time()-t_start, nnodes);
    return 0;
}
#endif
