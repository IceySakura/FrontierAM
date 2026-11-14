/* Gravitational acceleration for galaxy snapshots.
 *
 * Compressed Morton-octree BVH + Cartesian-Taylor multipole expansions of the
 * softened Plummer kernel K(u) = 1/sqrt(|u|^2 + eps^2), eps = 0.01.
 *
 * Near field: exact direct summation over tree leaves (softened self force = 0,
 *             so no self-exclusion is needed anywhere).
 * Far field : order-P multipole expansion about each accepted internal cell.
 * Targets are processed in groups of VW consecutive Morton-sorted particles so
 * the kernels vectorise; the acceptance test uses the group bounding sphere.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

static double EPS2 = 1e-4;

/* ------------------------------------------------------------ vector layer */
#ifdef USE_AVX2
#include <immintrin.h>
#define V __m256d
#define VLD(p) _mm256_loadu_pd(p)
#define VST(p, v) _mm256_storeu_pd((p), (v))
#define VBCAST(x) _mm256_set1_pd(x)
#define VZERO _mm256_setzero_pd()
#define VADD(a, b) _mm256_add_pd((a), (b))
#define VSUB(a, b) _mm256_sub_pd((a), (b))
#define VMUL(a, b) _mm256_mul_pd((a), (b))
#define VFMA(a, b, c) _mm256_fmadd_pd((a), (b), (c))
#define VNEG(a) _mm256_sub_pd(_mm256_setzero_pd(), (a))
#define VMULC(a, c) _mm256_mul_pd((a), _mm256_set1_pd(c))
#define VSQRT(a) _mm256_sqrt_pd(a)
#define VRCPR(a) _mm256_div_pd(_mm256_set1_pd(1.0), (a))
#define KSUF _avx
#define VW 4
#else
#define V double
#define VLD(p) (*(p))
#define VST(p, v) (*(p) = (v))
#define VBCAST(x) (x)
#define VZERO 0.0
#define VADD(a, b) ((a) + (b))
#define VSUB(a, b) ((a) - (b))
#define VMUL(a, b) ((a) * (b))
#define VFMA(a, b, c) ((a) * (b) + (c))
#define VNEG(a) (-(a))
#define VMULC(a, c) ((a) * (c))
#define VSQRT(a) sqrt(a)
#define VRCPR(a) (1.0 / (a))
#define KSUF _sc
#define VW 1
#endif

#define PP_CAT2(a, b) a##b
#define PP_CAT(a, b) PP_CAT2(a, b)
#define FUN(n) PP_CAT(n, KSUF)

#include "kern.inc"

/* ------------------------------------------------------------------ config */
#ifndef LEAF
#define LEAF 16
#endif
#define MBITS 21
#define MQMAX (1u << MBITS)
#define NIDXMAX 120 /* multi-indices with |a| <= PMAX+1 */
#define PMAXBUILD 7
#define NIDXFULL 120 /* (7+1)(7+2)(7+3)/6 */

static int nidx_of(int P) { return (P + 1) * (P + 2) * (P + 3) / 6; }

/* ------------------------------------------------------------------- timing */
static double cpu_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* ------------------------------------------------------------- Morton codes */
static inline uint64_t part1by2(uint32_t v) {
  uint64_t x = v & 0x1fffffu;
  x = (x | x << 32) & 0x1f00000000ffffULL;
  x = (x | x << 16) & 0x1f0000ff0000ffULL;
  x = (x | x << 8) & 0x100f00f00f00f00fULL;
  x = (x | x << 4) & 0x10c30c30c30c30c3ULL;
  x = (x | x << 2) & 0x1249249249249249ULL;
  return x;
}

/* ------------------------------------------------------------ tree storage */
typedef struct { double x, y, z, r; } Geo;

static Geo *geo;
static int32_t *nstart, *ncount, *nfirst; /* nfirst<0 => leaf */
static int32_t *chlist;
static int32_t *momidx;
static int nnodes = 0, nodecap = 0, ninternal = 0;
static const uint64_t *skeys;

static int newnode(void) {
  if (nnodes >= nodecap) {
    nodecap = nodecap ? nodecap * 2 : 8192;
    geo = realloc(geo, (size_t)nodecap * sizeof(Geo));
    nstart = realloc(nstart, (size_t)nodecap * 4);
    ncount = realloc(ncount, (size_t)nodecap * 4);
    nfirst = realloc(nfirst, (size_t)nodecap * 4);
    momidx = realloc(momidx, (size_t)nodecap * 4);
    chlist = realloc(chlist, (size_t)nodecap * 8 * 4);
    if (!geo || !nstart || !ncount || !nfirst || !momidx || !chlist) {
      fprintf(stderr, "oom tree\n");
      exit(1);
    }
  }
  return nnodes++;
}

static double *sx_, *sy_, *sz_, *sm_;
static int32_t *orig_;
static long S_visit,S_mint,S_dleaf,S_dsrc;

static int build(int32_t s, int32_t e, int depth) {
  int n = newnode();
  nstart[n] = s;
  ncount[n] = e - s;
  nfirst[n] = -1;
  momidx[n] = -1;

  double x0 = 1e300, y0 = 1e300, z0 = 1e300;
  double x1 = -1e300, y1 = -1e300, z1 = -1e300;
  for (int32_t i = s; i < e; i++) {
    double a = sx_[i];
    if (a < x0) x0 = a;
    if (a > x1) x1 = a;
    a = sy_[i];
    if (a < y0) y0 = a;
    if (a > y1) y1 = a;
    a = sz_[i];
    if (a < z0) z0 = a;
    if (a > z1) z1 = a;
  }
  double hx = 0.5 * (x1 - x0), hy = 0.5 * (y1 - y0), hz = 0.5 * (z1 - z0);
  geo[n].x = 0.5 * (x0 + x1);
  geo[n].y = 0.5 * (y0 + y1);
  geo[n].z = 0.5 * (z0 + z1);
  geo[n].r = sqrt(hx * hx + hy * hy + hz * hz);

  if (e - s <= LEAF || depth >= 22) return n;

  uint64_t d = skeys[s] ^ skeys[e - 1];
  if (d == 0) return n;
  int hb = 63 - __builtin_clzll(d);
  int L = (62 - hb) / 3;
  if (L >= MBITS - 1) return n;
  int shift = 60 - 3 * L;

  int32_t bnd[10];
  int k = 0;
  bnd[0] = s;
  int ci = (int)((skeys[s] >> shift) & 7);
  for (int32_t i = s + 1; i < e; i++) {
    int c2 = (int)((skeys[i] >> shift) & 7);
    if (c2 != ci) {
      if (k >= 8) return n;
      bnd[++k] = i;
      ci = c2;
    }
  }
  bnd[++k] = e;
  if (k < 2) return n;

  if (k > 8) return n;
  nfirst[n] = n * 8;
  for (int j = 0; j < 8; j++) chlist[n * 8 + j] = -1;
  for (int j = 0; j < k; j++) {
    int c = build(bnd[j], bnd[j + 1], depth + 1);
    chlist[n * 8 + j] = c; /* separate statement: build() may realloc chlist */
  }
  return n;
}

/* --------------------------------------------------------------- radix sort */
static uint32_t g_cnt[256];
static void radix_sort(uint64_t *k0, uint32_t *v0, uint64_t *k1, uint32_t *v1,
                       size_t n) {
  uint64_t *ks = k0, *kd = k1;
  uint32_t *vs = v0, *vd = v1;
  for (int pass = 0; pass < 8; pass++) {
    int sh = pass * 8;
    memset(g_cnt, 0, sizeof(g_cnt));
    for (size_t i = 0; i < n; i++) g_cnt[(ks[i] >> sh) & 0xff]++;
    uint32_t acc = 0;
    for (int b = 0; b < 256; b++) { uint32_t c = g_cnt[b]; g_cnt[b] = acc; acc += c; }
    for (size_t i = 0; i < n; i++) {
      uint32_t p = g_cnt[(ks[i] >> sh) & 0xff]++;
      kd[p] = ks[i];
      vd[p] = vs[i];
    }
    uint64_t *tk = ks; ks = kd; kd = tk;
    uint32_t *tv = vs; vs = vd; vd = tv;
  }
  if (ks != k0) { memcpy(k0, ks, n * 8); memcpy(v0, vs, n * 4); }
}

/* ------------------------------------------------------ moment computation */
static double *mom; /* ninternal * NIDXFULL, ordered by degree so that the
                       first nidx_of(P) entries are the order-P moments */

static void leafmom(int P, int32_t s, int32_t e, const Geo *g, double *o) {
  int n = nidx_of(P);
  for (int i = 0; i < n; i++) o[i] = 0.0;
  double gx = g->x, gy = g->y, gz = g->z;
  for (int32_t i = s; i < e; i++) {
    double dx = sx_[i] - gx, dy = sy_[i] - gy, dz = sz_[i] - gz, w = sm_[i];
    switch (P) {
      case 0: leafmom0(dx, dy, dz, w, o); break;
      case 1: leafmom1(dx, dy, dz, w, o); break;
      case 2: leafmom2(dx, dy, dz, w, o); break;
      case 3: leafmom3(dx, dy, dz, w, o); break;
      case 4: leafmom4(dx, dy, dz, w, o); break;
      case 5: leafmom5(dx, dy, dz, w, o); break;
      case 6: leafmom6(dx, dy, dz, w, o); break;
      case 7: leafmom7(dx, dy, dz, w, o); break;
    }
  }
}

static void m2mcall(int P, const double *ch, double tx, double ty, double tz,
                    double *par) {
  switch (P) {
    case 0: m2m0(ch, tx, ty, tz, par); break;
    case 1: m2m1(ch, tx, ty, tz, par); break;
    case 2: m2m2(ch, tx, ty, tz, par); break;
    case 3: m2m3(ch, tx, ty, tz, par); break;
    case 4: m2m4(ch, tx, ty, tz, par); break;
    case 5: m2m5(ch, tx, ty, tz, par); break;
    case 6: m2m6(ch, tx, ty, tz, par); break;
    case 7: m2m7(ch, tx, ty, tz, par); break;
  }
}

static void build_moments(int P) {
  double scratch[NIDXMAX];
  for (int n = nnodes - 1; n >= 0; n--) {
    if (nfirst[n] < 0) continue;
    double *pm = mom + (size_t)momidx[n] * NIDXFULL;
    for (int i = 0; i < NIDXFULL; i++) pm[i] = 0.0;
    for (int j = 0; j < 8; j++) {
      int c = chlist[n * 8 + j];
      if (c < 0) continue;
      const double *cm;
      if (nfirst[c] < 0) {
        leafmom(P, nstart[c], nstart[c] + ncount[c], &geo[c], scratch);
        cm = scratch;
      } else {
        cm = mom + (size_t)momidx[c] * NIDXFULL;
      }
      m2mcall(P, cm, geo[c].x - geo[n].x, geo[c].y - geo[n].y,
              geo[c].z - geo[n].z, pm);
    }
  }
}

/* ---------------------------------------------------------------- traversal */
static inline void direct_group(int n, V px, V py, V pz, V *ax, V *ay, V *az) {
  int32_t s = nstart[n], e = s + ncount[n];
  V veps = VBCAST(EPS2);
  for (int32_t i = s; i < e; i++) {
    V bx = VBCAST(sx_[i]), by = VBCAST(sy_[i]), bz = VBCAST(sz_[i]);
    V dx = VSUB(bx, px), dy = VSUB(by, py), dz = VSUB(bz, pz);
    V r2 = VFMA(dx, dx, VFMA(dy, dy, VFMA(dz, dz, veps)));
    V q = VRCPR(r2);
    V w = VMUL(VSQRT(q), VMUL(q, VBCAST(sm_[i])));
    *ax = VFMA(dx, w, *ax);
    *ay = VFMA(dy, w, *ay);
    *az = VFMA(dz, w, *az);
  }
}

static inline void group_bounds(int32_t g0, int cnt, double *gcx, double *gcy,
                                double *gcz, double *gr) {
  double x0 = 1e300, y0 = 1e300, z0 = 1e300, x1 = -1e300, y1 = -1e300, z1 = -1e300;
  for (int k = 0; k < cnt; k++) {
    double a = sx_[g0 + k];
    if (a < x0) x0 = a;
    if (a > x1) x1 = a;
    a = sy_[g0 + k];
    if (a < y0) y0 = a;
    if (a > y1) y1 = a;
    a = sz_[g0 + k];
    if (a < z0) z0 = a;
    if (a > z1) z1 = a;
  }
  double cx = 0.5 * (x0 + x1), cy = 0.5 * (y0 + y1), cz = 0.5 * (z0 + z1);
  double r = 0.0;
  for (int k = 0; k < cnt; k++) {
    double a = sx_[g0 + k] - cx, b = sy_[g0 + k] - cy, c = sz_[g0 + k] - cz;
    double t = a * a + b * b + c * c;
    if (t > r) r = t;
  }
  *gcx = cx; *gcy = cy; *gcz = cz; *gr = sqrt(r);
}

#define STACKSZ 512

#define DEF_TRAV(P)                                                          \
  static long trav_##P(V px, V py, V pz, double gcx, double gcy, double gcz, \
                       double gr, double theta, V *ax, V *ay, V *az,         \
                       int *abortflag) {                                     \
    int32_t stack[STACKSZ];                                                  \
    int sp = 0, ab = 0;                                                      \
    long nint = 0;                                                           \
    stack[sp++] = 0;                                                         \
    while (sp > 0) {                                                         \
      int n = stack[--sp]; S_visit++;                                        \
      double dx = geo[n].x - gcx, dy = geo[n].y - gcy, dz = geo[n].z - gcz;  \
      double lim = theta * (sqrt(dx * dx + dy * dy + dz * dz) - gr);         \
      int fc = nfirst[n];                                                    \
      if (fc < 0) {                                                          \
        if (lim > geo[n].r) nint++;                                          \
        S_dleaf++; S_dsrc += ncount[n];                                      \
        direct_group(n, px, py, pz, ax, ay, az);                             \
        continue;                                                            \
      }                                                                      \
      if (lim > geo[n].r) {                                                  \
        nint++; S_mint++;                                                    \
        const double *m = mom + (size_t)momidx[n] * NIDXFULL;                \
        V Dx = VSUB(VBCAST(geo[n].x), px);                                   \
        V Dy = VSUB(VBCAST(geo[n].y), py);                                   \
        V Dz = VSUB(VBCAST(geo[n].z), pz);                                   \
        V R2 = VFMA(Dx, Dx, VFMA(Dy, Dy, VFMA(Dz, Dz, VBCAST(EPS2))));       \
        FUN(m2p##P)(m, Dx, Dy, Dz, R2, ax, ay, az);                          \
        continue;                                                            \
      }                                                                      \
      for (int j = 0; j < 8; j++) {                                          \
        int c = chlist[fc + j];                                              \
        if (c >= 0 && sp < STACKSZ - 1) stack[sp++] = c;                     \
      }                                                                      \
    }                                                                        \
    *abortflag = ab;                                                         \
    return nint;                                                             \
  }

DEF_TRAV(0)
DEF_TRAV(1)
DEF_TRAV(2)
DEF_TRAV(3)
DEF_TRAV(4)
DEF_TRAV(5)
DEF_TRAV(6)
DEF_TRAV(7)

typedef long (*travfn)(V, V, V, double, double, double, double, double, V *,
                       V *, V *, int *);
static travfn travtab[8] = {trav_0, trav_1, trav_2, trav_3,
                            trav_4, trav_5, trav_6, trav_7};


/* Sample the per-group cost of a configuration over a few contiguous runs.
 * Bails out early if it would burn too much of the budget. */
static double sample_per_group(int P, double th, long ngroups, double cap,
                               double *used) {
  long bsize = 10, nblk = 8;
  if (ngroups < bsize * nblk) { bsize = ngroups > 16 ? ngroups / 8 : 1; nblk = 8; }
  long bstride = (ngroups - bsize) / (nblk > 1 ? nblk - 1 : 1);
  if (bstride < 1) bstride = 1;
  double t0 = cpu_now();
  long done = 0;
  int ab = 0;
  for (long b = 0; b < nblk; b++) {
    long g0i = b * bstride;
    for (long q = 0; q < bsize; q++) {
      int32_t g0 = (int32_t)((g0i + q) * VW);
      double gcx, gcy, gcz, gr;
      group_bounds(g0, VW, &gcx, &gcy, &gcz, &gr);
      V px = VLD(sx_ + g0), py = VLD(sy_ + g0), pz = VLD(sz_ + g0);
      V ax = VZERO, ay = VZERO, az = VZERO;
      travtab[P](px, py, pz, gcx, gcy, gcz, gr, th, &ax, &ay, &az, &ab);
      double tt[3][VW];
      VST(tt[0], ax); VST(tt[1], ay); VST(tt[2], az);
      if (tt[0][0] == 1e300 && tt[1][0] == 1e300) fprintf(stderr, "%g", tt[2][0]);
      done++;
      if ((done & 7) == 0 && cpu_now() - t0 > cap) { *used = cpu_now() - t0; return *used / (double)done; }
    }
  }
  *used = cpu_now() - t0;
  return *used / (double)done;
}

/* ------------------------------------------------------------------- main  */
int main(int argc, char **argv) {
  double t_start = cpu_now();
  if (argc < 3) { fprintf(stderr, "usage: solve IN OUT\n"); return 2; }

  double limit = 20.0;
  const char *ls = getenv("GALAXY_CPU_LIMIT_S");
  if (ls && *ls) { double v = atof(ls); if (v > 0) limit = v; }
  const char *ov = getenv("GALAXY_THETA");
  const char *op = getenv("GALAXY_P");
  double soft = limit * 0.84;
  double soft_run;
  double hard = t_start + limit * 0.87;
  double panic = t_start + limit * 0.95;

  int fd = open(argv[1], O_RDONLY);
  if (fd < 0) { perror("open in"); return 2; }
  off_t fsz = lseek(fd, 0, SEEK_END);
  lseek(fd, 0, SEEK_SET);
  if (fsz <= 0 || (fsz % 32) != 0) { fprintf(stderr, "bad input size\n"); return 2; }
  size_t N = (size_t)fsz / 32;
  double *raw = malloc((size_t)fsz);
  if (!raw) { fprintf(stderr, "oom raw\n"); return 1; }
  {
    size_t g = 0;
    while (g < (size_t)fsz) {
      ssize_t r = read(fd, (char *)raw + g, (size_t)fsz - g);
      if (r <= 0) break;
      g += (size_t)r;
    }
    close(fd);
    if (g != (size_t)fsz) { fprintf(stderr, "short read\n"); return 2; }
  }

  double xmin = 1e300, ymin = 1e300, zmin = 1e300;
  double xmax = -1e300, ymax = -1e300, zmax = -1e300;
  for (size_t i = 0; i < N; i++) {
    const double *p = raw + 4 * i;
    double a = p[1], b = p[2], c = p[3];
    if (a < xmin) xmin = a;
    if (a > xmax) xmax = a;
    if (b < ymin) ymin = b;
    if (b > ymax) ymax = b;
    if (c < zmin) zmin = c;
    if (c > zmax) zmax = c;
  }
  double ex = xmax - xmin, ey = ymax - ymin, ez = zmax - zmin;
  if (!(ex > 0)) ex = 1.0;
  if (!(ey > 0)) ey = 1.0;
  if (!(ez > 0)) ez = 1.0;
  double Lb = ex > ey ? ex : ey;
  if (ez > Lb) Lb = ez;
  Lb *= 1.0000002;
  double qs = (double)(MQMAX - 1) / Lb;

  uint64_t *k0 = malloc(N * 8), *k1 = malloc(N * 8);
  uint32_t *v0 = malloc(N * 4), *v1 = malloc(N * 4);
  if (!k0 || !k1 || !v0 || !v1) { fprintf(stderr, "oom sort\n"); return 1; }
  for (size_t i = 0; i < N; i++) {
    const double *p = raw + 4 * i;
    double fx = (p[1] - xmin) * qs, fy = (p[2] - ymin) * qs, fz = (p[3] - zmin) * qs;
    uint32_t ix = fx <= 0 ? 0 : (fx >= (double)(MQMAX - 1) ? MQMAX - 1 : (uint32_t)fx);
    uint32_t iy = fy <= 0 ? 0 : (fy >= (double)(MQMAX - 1) ? MQMAX - 1 : (uint32_t)fy);
    uint32_t iz = fz <= 0 ? 0 : (fz >= (double)(MQMAX - 1) ? MQMAX - 1 : (uint32_t)fz);
    k0[i] = part1by2(ix) | (part1by2(iy) << 1) | (part1by2(iz) << 2);
    v0[i] = (uint32_t)i;
  }
  radix_sort(k0, v0, k1, v1, N);
  free(k1); free(v1);

  size_t Npad = (N + VW - 1) / VW * VW;
  sx_ = malloc(Npad * 8); sy_ = malloc(Npad * 8);
  sz_ = malloc(Npad * 8); sm_ = malloc(Npad * 8);
  orig_ = malloc(Npad * 4);
  if (!sx_ || !sy_ || !sz_ || !sm_ || !orig_) { fprintf(stderr, "oom soa\n"); return 1; }
  for (size_t i = 0; i < N; i++) {
    const double *p = raw + 4 * (size_t)v0[i];
    sx_[i] = p[1]; sy_[i] = p[2]; sz_[i] = p[3]; sm_[i] = p[0];
    orig_[i] = (int32_t)v0[i];
  }
  for (size_t i = N; i < Npad; i++) {
    sx_[i] = sx_[N - 1]; sy_[i] = sy_[N - 1]; sz_[i] = sz_[N - 1];
    sm_[i] = 0.0; orig_[i] = (int32_t)(N - 1);
  }
  free(raw); free(v0);


  double *outv = calloc(N * 3, 8);
  if (!outv) { fprintf(stderr, "oom out\n"); return 1; }

  skeys = k0; /* sorted keys, same order as sx_ */
  build(0, (int32_t)N, 0);
  ninternal = 0;
  for (int n = 0; n < nnodes; n++)
    if (nfirst[n] >= 0) momidx[n] = ninternal++;
  free(k0); skeys = NULL;

  mom = malloc((size_t)(ninternal ? ninternal : 1) * NIDXFULL * 8);
  if (!mom) { fprintf(stderr, "oom mom\n"); return 1; }
  build_moments(PMAXBUILD);
  double t_setup = cpu_now() - t_start;

  long ngroups = (long)Npad / VW;
  int Puse = 2;
  double theta_use = 0.5;

  if (ov) theta_use = atof(ov);
  if (op) Puse = atoi(op);

  if (!ov && !op) {
    /* Candidates ordered best-expected-accuracy first (err ~ A_P*theta^(P+1)).
     * Two mid-list samples calibrate a cost model  t = a*C_P*th^-1.8 + b*th^-3
     * which then predicts every candidate; the winner is verified by sampling. */
    static const int cP[] = {7, 7, 7, 7, 7, 7, 6, 5, 5, 5, 6, 6, 6, 5, 5, 5, 5,
                             4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2,
                             1, 0};
    static double cT[] = {0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.34, 0.28,
                          0.30, 0.32, 0.42, 0.44, 0.46, 0.40, 0.42, 0.44,
                          0.46, 0.40, 0.44, 0.48, 0.52, 0.56, 0.60, 0.64,
                          0.54, 0.58, 0.62, 0.68, 0.74, 0.64, 0.70, 0.78,
                          0.86, 0.86, 0.70, 0.85};
    static const double cC[] = {2391, 2391, 2391, 2391, 2391, 2391, 1495, 908,
                                908, 908, 1495, 1495, 1495, 908, 908, 908, 908,
                                518, 518, 518, 518, 518, 518, 518, 267, 267,
                                267, 267, 267, 121, 121, 121, 121, 121, 48, 13};
    int nc = (int)(sizeof(cP) / sizeof(int));

    double remain = soft - (cpu_now() - t_start);
    double iow = 0.03 * limit + 0.03;
    remain -= iow;
    if (remain < limit * 0.12) remain = limit * 0.12;
    double cap = 0.10 * limit;

    static double pred[64];
    int i1 = (int)(nc * 0.45), i2 = (int)(nc * 0.72);
    double u;
    double pg1 = sample_per_group(cP[i1], cT[i1], ngroups, cap, &u);
    double pg2 = sample_per_group(cP[i2], cT[i2], ngroups, cap, &u);
    double M1 = cC[i1] * pow(cT[i1], -1.8), D1 = pow(cT[i1], -3.0);
    double M2 = cC[i2] * pow(cT[i2], -1.8), D2 = pow(cT[i2], -3.0);
    double det = M1 * D2 - M2 * D1;
    double a = 0, b = 0;
    if (fabs(det) > 1e-30) {
      a = (pg1 * D2 - pg2 * D1) / det;
      b = (M1 * pg2 - M2 * pg1) / det;
    }
    if (!(a > 1e-30)) a = 1e-30;
    if (!(b > 0)) b = 0;
    for (int i = 0; i < nc; i++)
      pred[i] = (a * cC[i] * pow(cT[i], -1.8) + b * pow(cT[i], -3.0)) *
                (double)ngroups;

    int chosen = -1;
    for (int i = 0; i < nc; i++)
      if (pred[i] * 1.10 < remain) { chosen = i; break; }
    if (chosen < 0) chosen = nc - 1;

    /* verify and, if needed, step down the list */
    double est = pred[chosen];
    for (int tries = 0; tries < 4; tries++) {
      double pg = sample_per_group(cP[chosen], cT[chosen], ngroups, cap, &u);
      est = pg * (double)ngroups * 1.15;
      if (getenv("GALAXY_VERBOSE"))
        fprintf(stderr, "  verify P=%d th=%.2f est=%.2fs remain=%.2fs\n",
                cP[chosen], cT[chosen], est, remain);
      if (est < remain) break;
      double scale = remain / est;
      int j = chosen + 1;
      while (j < nc && pred[j] > pred[chosen] * scale * 0.97) j++;
      if (j >= nc) { chosen = nc - 1; break; }
      chosen = j;
    }
    if (est > remain) {
      /* still over: open theta on the chosen candidate until it fits */
      double f = est / remain;
      if (f > 1.0) {
        double nt = cT[chosen] * pow(f, 0.40);
        if (nt > 0.88) nt = 0.88;
        cT[chosen] = nt;
      }
    }
    /* spend leftover budget: tighten theta (and try a more accurate order) */
    {
      double samps = 0;
      for (int it = 0; it < 5 && samps < 0.30 * limit; it++) {
        if (est > 0.80 * remain) break;
        int j = chosen;
        /* try the next more accurate order at a comparable theta first */
        if (chosen + 1 < nc && cP[chosen + 1] > cP[chosen]) j = chosen + 1;
        double ratio = est / (0.82 * remain);
        double nt = cT[chosen] * pow(ratio, 0.36);
        if (j != chosen) nt = cT[chosen] * pow(ratio, 0.42) * 1.16;
        if (nt < 0.22) nt = 0.22;
        if (nt > 0.86) nt = 0.86;
        if (j == chosen && nt >= cT[chosen] * 0.995) break;
        double pg = sample_per_group(cP[j], nt, ngroups, cap, &u);
        samps += u;
        double e2 = pg * (double)ngroups * 1.15;
        if (e2 < remain && e2 > est * 0.6) {
          chosen = j; cT[j] = nt; est = e2;
        } else if (e2 >= remain) {
          double f = e2 / remain;
          double nt2 = nt * pow(f, 0.40);
          if (nt2 > 0.86) nt2 = 0.86;
          double pg2 = sample_per_group(cP[j], nt2, ngroups, cap, &u);
          samps += u;
          double e3 = pg2 * (double)ngroups * 1.15;
          if (e3 < remain) { chosen = j; cT[j] = nt2; est = e3; }
          break;
        } else break;
      }
    }
    Puse = cP[chosen];
    theta_use = cT[chosen];
    if (getenv("GALAXY_VERBOSE"))
      fprintf(stderr, "  chose P=%d theta=%.3f pred=%.2fs remain=%.2fs\n",
              Puse, theta_use, pred[chosen], remain);
  }

  /* ---- main run ---- */
  int Plive_out = Puse;
  soft_run = t_start + limit * 0.90;  /* runtime governor safety net */
  {
    int ab = 0;
    double theta = theta_use;
    int Plive = Puse;
    travfn tf = travtab[Plive];
    double t_run0 = cpu_now();
    double runbudget = soft_run - t_run0;
    if (runbudget < 0.02 * limit) runbudget = 0.02 * limit;
    long g = 0;
    for (; g < ngroups; g++) {
      int32_t g0 = (int32_t)(g * VW);
      double gcx, gcy, gcz, gr;
      group_bounds(g0, VW, &gcx, &gcy, &gcz, &gr);
      V px = VLD(sx_ + g0), py = VLD(sy_ + g0), pz = VLD(sz_ + g0);
      V ax = VZERO, ay = VZERO, az = VZERO;
      tf(px, py, pz, gcx, gcy, gcz, gr, theta, &ax, &ay, &az, &ab);
      double tt[3][VW];
      VST(tt[0], ax); VST(tt[1], ay); VST(tt[2], az);
      for (int k = 0; k < VW; k++) {
        if (g0 + k >= (int32_t)N) break;
        double *o = outv + (size_t)orig_[g0 + k] * 3;
        o[0] = tt[0][k]; o[1] = tt[1][k]; o[2] = tt[2][k];
      }
      if ((g & 127) == 127) {
        double now = cpu_now();
        if (now > panic) { g++; break; }
        /* progress-aware degradation: keep elapsed/limit ahead of progress */
        double prog = (double)(g + 1) / (double)ngroups;
        double proj = (now - t_run0) / runbudget / (prog > 1e-9 ? prog : 1e-9);
        if (prog > 0.10 && proj > 1.08) {
          theta += 0.04;
          if (theta > 0.8) theta = 0.8;
        }
        if (prog > 0.10 && proj > 1.35) { Plive = 0; tf = travtab[0]; theta = 0.8; }
        Plive_out = Plive; theta_use = theta;
        (void)t_run0;
      }
    }
  }

  /* ---- write output ---- */
  {
    int ofd = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (ofd < 0) { perror("open out"); return 2; }
    size_t total = N * 3 * 8, w = 0;
    while (w < total) {
      ssize_t r = write(ofd, (char *)outv + w, total - w);
      if (r <= 0) break;
      w += (size_t)r;
    }
    close(ofd);
    if (w != total) { fprintf(stderr, "short write\n"); return 2; }
  }

  if (getenv("GALAXY_VERBOSE"))
    fprintf(stderr,
            "N=%zu nodes=%d nint=%d P=%d/%d theta=%.3f setup=%.3f total=%.3f "
            "limit=%.2f visit/grp=%.0f mint/grp=%.0f dleaf/grp=%.0f dsrc/grp=%.0f\n",
            N, nnodes, ninternal, Puse, Plive_out, theta_use, t_setup,
            cpu_now() - t_start, limit,
            (double)S_visit/ngroups,(double)S_mint/ngroups,(double)S_dleaf/ngroups,(double)S_dsrc/ngroups);
  return 0;
}
