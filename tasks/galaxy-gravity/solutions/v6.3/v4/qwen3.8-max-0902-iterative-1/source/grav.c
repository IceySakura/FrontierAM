/*
 * Softened gravitational acceleration for galaxy snapshots.
 *
 *   a_i = G * sum_{j != i} m_j (x_j - x_i) / (|x_j - x_i|^2 + eps^2)^{3/2}
 *
 * G = 1, eps = 0.01.  Single process, single thread, AVX2/FMA, < 2 GiB.
 *
 * Method
 * ------
 * Morton-ordered octree.  A cell is accepted for a target when
 *
 *      theta * |x_target - c| >= rad        (rad = max particle offset from
 *                                                the cell's centre of mass)
 *
 * Accepted cells are represented by a Cartesian multipole expansion of order
 * p about their COM, contracted with exact tabulated derivatives of 1/r
 * (dertab.h):
 *
 *      a_i += sum_{|alpha| <= p} M_alpha / alpha! * D_{alpha + e_j}(R),
 *      R = x_i - c,   D_beta = partial^beta (1/r) at R.
 *
 * Rejected cells are recursed into; leaves are summed exactly pair by pair
 * with a vectorised AVX2 kernel.  Since accepted cells satisfy
 * |R| >= rad/theta, the unsoftened expansion is accurate far below its
 * truncation error, and eps/|R| <= theta <= 0.9 makes the softening
 * perturbation negligible.
 *
 * Memory
 * ------
 * Moments dominate: ncells ~ 1.5N at NTERM(p) doubles each.  The multipole
 * order is therefore capped by the 2 GiB budget for the actual N, and the
 * tree is rebuilt if a later, cheaper order is selected.
 *
 * Runtime safety
 * --------------
 * GALAXY_CPU_LIMIT_S gives this call's CPU budget.  The program builds the
 * tree once, probes throughput on a particle sample for each candidate
 * (theta, p), picks the most accurate candidate that fits the remaining
 * budget, then computes in chunks while re-checking elapsed CPU time.  If the
 * projection stops fitting it degrades (theta, p); near the deadline it
 * switches to the cheapest settings, so a complete finite output is always
 * written and the process always exits 0.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include <stdarg.h>
#include <immintrin.h>

#include "dertab.h"

typedef int32_t i32;
typedef int64_t i64;
typedef uint32_t u32;
typedef uint64_t u64;
typedef uint8_t u8;

#define EPS_SOFT 0.01
#define EPS2 (EPS_SOFT * EPS_SOFT)

#define LEAFMAX 16
#define MAXP 6
#define NTERM(p) (((p) + 1) * ((p) + 2) * ((p) + 3) / 6)
#define NTERM_MAX NTERM(MAXP)

#define MEM_LIMIT (1.75e9)   /* stay under the 2 GiB peak with headroom */

#define PROBE_THETA 0.40     /* theta used while measuring walk throughput */
#define TH_EXP 2.4       /* walk cost ~ theta^-TH_EXP */
#define SAFETY 0.85          /* fraction of the remaining budget we may plan for */
#define PRED_MARGIN 1.35     /* measured probe predictions run optimistic */

/* Opening angles, most accurate first. */
static const double THETAS[] = { 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.90 };
#define NTHETA ((int)(sizeof(THETAS)/sizeof(THETAS[0])))

/* ------------------------------------------------------------------ */

/* Measured pair throughput on this machine (pairs/second, full N^2 count).
 * Set by probe_pair_rate(); the default is deliberately pessimistic so a
 * failed probe can never cause an overrun. */
static double g_pair_rate = 0.35e9;
static double g_t0;              /* CPU time at process start        */
static double g_limit = 60.0;    /* CPU seconds allowed for this call */
static int g_verbose = 0;

static double cpu_seconds(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) != 0) return 0.0;
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
static inline double elapsed(void) { return cpu_seconds() - g_t0; }
static void die(const char *msg) { fprintf(stderr, "fatal: %s\n", msg); exit(1); }
static void *xmalloc(size_t n) {
  void *p = NULL;
  if (posix_memalign(&p, 64, ((n ? n : 64) + 63) & ~(size_t)63) != 0) die("oom");
  return p;
}
static void *xrealloc(void *p, size_t n) {
  void *q = realloc(p, ((n ? n : 64) + 63) & ~(size_t)63);
  if (!q) die("oom");
  return q;
}
static void logf_(const char *fmt, ...) {
  if (!g_verbose) return;
  va_list ap; va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fflush(stderr);
}

/* ------------------------------------------------------------------ */
/* r^-3, correctly rounded to ~0.5 ulp.                                 */
/* ------------------------------------------------------------------ */
static inline __m256d inv_r3(__m256d r2) {
  __m256d r = _mm256_sqrt_pd(r2);
  return _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(r2, r));
}

/* ================================================================== */
/*  Tree                                                               */
/* ================================================================== */
typedef struct {
  double gx, gy, gz;   /* centre of mass = expansion centre */
  double cx, cy, cz;   /* geometric centre of the box       */
  double mass;
  double rad;          /* max particle distance from COM    */
  double half;         /* half width of the cell box        */
  i32 child[8];
  i32 p0, cnt;
  i32 isleaf;
} Cell;

typedef struct {
  Cell *cells;
  i64 ncells, cap;
  double *sx, *sy, *sz, *sm;   /* particles in Morton order */
  i32 *order;                  /* sorted slot -> original row */
  i64 n;
  double *mom;                 /* [ncells][nmt] */
  i32 nmt;                     /* NTERM(p) for the built order */
  i32 p;                       /* order the moments were built for */
  i64 root;
} Tree;

static i32 cell_alloc(Tree *t) {
  if (t->ncells >= t->cap) {
    t->cap = t->cap ? t->cap * 2 : (1 << 15);
    t->cells = (Cell *)xrealloc(t->cells, sizeof(Cell) * (size_t)t->cap);
  }
  i32 id = (i32)t->ncells++;
  Cell *c = &t->cells[id];
  memset(c, 0, sizeof(Cell));
  for (int k = 0; k < 8; k++) c->child[k] = -1;
  c->isleaf = 1;
  return id;
}

static i32 build_rec(Tree *t, i64 s, i64 e, double cx, double cy, double cz,
                     double half, int depth) {
  i32 id = cell_alloc(t);
  const i32 p = t->p, nmt = t->nmt;
  t->cells[id].cx = cx; t->cells[id].cy = cy; t->cells[id].cz = cz;
  t->cells[id].half = half;
  t->cells[id].p0 = (i32)s;
  t->cells[id].cnt = (i32)(e - s);

  double M = 0, mx = 0, my = 0, mz = 0;
  for (i64 k = s; k < e; k++) {
    double mk = t->sm[k];
    M += mk; mx += mk * t->sx[k]; my += mk * t->sy[k]; mz += mk * t->sz[k];
  }
  double gx = cx, gy = cy, gz = cz;
  if (M > 0) { gx = mx / M; gy = my / M; gz = mz / M; }
  t->cells[id].mass = M;
  t->cells[id].gx = gx; t->cells[id].gy = gy; t->cells[id].gz = gz;

  double rmax = 0;
  for (i64 k = s; k < e; k++) {
    double dx = t->sx[k] - gx, dy = t->sy[k] - gy, dz = t->sz[k] - gz;
    double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 > rmax) rmax = r2;
  }
  t->cells[id].rad = sqrt(rmax);

  {
    double *Mo = t->mom + (size_t)id * nmt;
    double pw[MAXP + 1], qw[MAXP + 1], rw[MAXP + 1];
    for (i64 k = s; k < e; k++) {
      double dx = t->sx[k] - gx, dy = t->sy[k] - gy, dz = t->sz[k] - gz;
      double mk = t->sm[k];
      /* mass enters exactly once: pw carries it, qw/rw are pure powers */
      pw[0] = mk; qw[0] = 1.0; rw[0] = 1.0;
      for (int a = 1; a <= p; a++) pw[a] = pw[a - 1] * dx;
      for (int b = 1; b <= p; b++) qw[b] = qw[b - 1] * dy;
      for (int c = 1; c <= p; c++) rw[c] = rw[c - 1] * dz;
      for (int a = 0; a <= p; a++)
        for (int b = 0; a + b <= p; b++) {
          double ab = pw[a] * qw[b];
          for (int c = 0; a + b + c <= p; c++)
            Mo[der_key(a, b, c)] += ab * rw[c];
        }
    }
  }

  if (e - s <= LEAFMAX || depth >= 21 || half < 1e-14) {
    t->cells[id].isleaf = 1;
    return id;
  }

  i64 nn = e - s;
  u8 *oct = (u8 *)xmalloc((size_t)nn);
  i64 cnt[8] = {0,0,0,0,0,0,0,0};
  for (i64 k = 0; k < nn; k++) {
    int o = ((t->sx[s + k] > cx) << 2) | ((t->sy[s + k] > cy) << 1) |
            (t->sz[s + k] > cz);
    oct[k] = (u8)o;
    cnt[o]++;
  }
  i64 start[9]; start[0] = s;
  for (int k = 0; k < 8; k++) start[k + 1] = start[k] + cnt[k];
  i64 cur[8]; for (int k = 0; k < 8; k++) cur[k] = start[k];
  i32 *perm = (i32 *)xmalloc(sizeof(i32) * (size_t)nn);
  for (i64 k = 0; k < nn; k++) perm[cur[oct[k]]++ - s] = (i32)(s + k);
  free(oct);

  double *nx = (double *)xmalloc(sizeof(double) * (size_t)nn);
  double *ny = (double *)xmalloc(sizeof(double) * (size_t)nn);
  double *nz = (double *)xmalloc(sizeof(double) * (size_t)nn);
  double *nm = (double *)xmalloc(sizeof(double) * (size_t)nn);
  i32 *no = (i32 *)xmalloc(sizeof(i32) * (size_t)nn);
  for (i64 k = 0; k < nn; k++) {
    i64 src = perm[k];
    nx[k] = t->sx[src]; ny[k] = t->sy[src]; nz[k] = t->sz[src];
    nm[k] = t->sm[src]; no[k] = t->order[src];
  }
  memcpy(t->sx + s, nx, sizeof(double) * (size_t)nn);
  memcpy(t->sy + s, ny, sizeof(double) * (size_t)nn);
  memcpy(t->sz + s, nz, sizeof(double) * (size_t)nn);
  memcpy(t->sm + s, nm, sizeof(double) * (size_t)nn);
  memcpy(t->order + s, no, sizeof(i32) * (size_t)nn);
  free(nx); free(ny); free(nz); free(nm); free(no); free(perm);

  t->cells[id].isleaf = 0;
  double h2 = half * 0.5;
  for (int k = 0; k < 8; k++) {
    if (start[k + 1] == start[k]) continue;
    double ox = ((k >> 2) & 1) ? h2 : -h2;
    double oy = ((k >> 1) & 1) ? h2 : -h2;
    double oz = (k & 1) ? h2 : -h2;
    i32 ch = build_rec(t, start[k], start[k + 1], cx + ox, cy + oy, cz + oz,
                       h2, depth + 1);
    t->cells[id].child[k] = ch;   /* re-read: cells may have been realloc'd */
  }
  return id;
}

/* ------------------------------------------------------------------ */
/* Morton ordering                                                      */
/* ------------------------------------------------------------------ */
static inline u64 part_bits(u64 v) {
  v &= 0x1fffffULL;
  v = (v | (v << 32)) & 0x1f00000000ffffULL;
  v = (v | (v << 16)) & 0x1f0000ff0000ffULL;
  v = (v | (v << 8)) & 0x100f00f00f00f00fULL;
  v = (v | (v << 4)) & 0x10c30c30c30c30c3ULL;
  v = (v | (v << 2)) & 0x1249249249249249ULL;
  return v;
}
static inline u64 morton3(u64 x, u64 y, u64 z) {
  return part_bits(x) | (part_bits(y) << 1) | (part_bits(z) << 2);
}

static void radix_sort_ki(u64 *keys, i32 *idx, i64 n) {
  u64 *kbuf = (u64 *)xmalloc(sizeof(u64) * (size_t)n);
  i32 *ibuf = (i32 *)xmalloc(sizeof(i32) * (size_t)n);
  u64 *ks = keys, *kd = kbuf;
  i32 *is = idx, *id = ibuf;
  for (int pass = 0; pass < 8; pass++) {
    i64 hist[256];
    memset(hist, 0, sizeof(hist));
    int sh = pass * 8;
    for (i64 i = 0; i < n; i++) hist[(ks[i] >> sh) & 0xff]++;
    i64 acc = 0;
    for (int b = 0; b < 256; b++) { i64 c = hist[b]; hist[b] = acc; acc += c; }
    for (i64 i = 0; i < n; i++) {
      int b = (int)((ks[i] >> sh) & 0xff);
      i64 o = hist[b]++;
      kd[o] = ks[i]; id[o] = is[i];
    }
    u64 *tk = ks; ks = kd; kd = tk;
    i32 *ti = is; is = id; id = ti;
  }
  if (ks != keys) {
    memcpy(keys, ks, sizeof(u64) * (size_t)n);
    memcpy(idx, is, sizeof(i32) * (size_t)n);
  }
  free(kbuf); free(ibuf);
}

static void tree_build(Tree *t, const double *m, const double *px,
                       const double *py, const double *pz, i64 n, i32 p,
                       double *sorted_done_at) {
  int fresh = (t->sx == NULL);
  t->n = n;
  t->p = p;
  t->nmt = NTERM(p);
  if (fresh) {
    t->sx = (double *)xmalloc(sizeof(double) * (size_t)n);
    t->sy = (double *)xmalloc(sizeof(double) * (size_t)n);
    t->sz = (double *)xmalloc(sizeof(double) * (size_t)n);
    t->sm = (double *)xmalloc(sizeof(double) * (size_t)n);
    t->order = (i32 *)xmalloc(sizeof(i32) * (size_t)n);
  }

  if (fresh) {
    double xmin = px[0], xmax = px[0], ymin = py[0], ymax = py[0];
    double zmin = pz[0], zmax = pz[0];
    for (i64 i = 1; i < n; i++) {
      if (px[i] < xmin) xmin = px[i];
      if (px[i] > xmax) xmax = px[i];
      if (py[i] < ymin) ymin = py[i];
      if (py[i] > ymax) ymax = py[i];
      if (pz[i] < zmin) zmin = pz[i];
      if (pz[i] > zmax) zmax = pz[i];
    }
    double span = xmax - xmin;
    if (ymax - ymin > span) span = ymax - ymin;
    if (zmax - zmin > span) span = zmax - zmin;
    double bhalf = 0.5 * span * (1.0 + 1e-9) + 1e-30;

    u64 *keys = (u64 *)xmalloc(sizeof(u64) * (size_t)n);
    i32 *idx = (i32 *)xmalloc(sizeof(i32) * (size_t)n);
    const double scale = (double)((i64)1 << 21) / (2.0 * bhalf);
    const i64 qmax = ((i64)1 << 21) - 1;
    double bx = 0.5 * (xmin + xmax), by = 0.5 * (ymin + ymax), bz = 0.5 * (zmin + zmax);
    for (i64 i = 0; i < n; i++) {
      i64 qx = (i64)((px[i] - bx + bhalf) * scale);
      i64 qy = (i64)((py[i] - by + bhalf) * scale);
      i64 qz = (i64)((pz[i] - bz + bhalf) * scale);
      if (qx < 0) qx = 0; else if (qx > qmax) qx = qmax;
      if (qy < 0) qy = 0; else if (qy > qmax) qy = qmax;
      if (qz < 0) qz = 0; else if (qz > qmax) qz = qmax;
      keys[i] = morton3((u64)qx, (u64)qy, (u64)qz);
      idx[i] = (i32)i;
    }
    radix_sort_ki(keys, idx, n);
    free(keys);
    for (i64 i = 0; i < n; i++) {
      i32 src = idx[i];
      t->sx[i] = px[src]; t->sy[i] = py[src]; t->sz[i] = pz[src];
      t->sm[i] = m[src]; t->order[i] = src;
    }
    free(idx);
    if (sorted_done_at) *sorted_done_at = elapsed();
  }

  /* size the cell/moment storage from the particle count */
  i64 est = n / LEAFMAX * 2 + 4 * n / LEAFMAX + 4096;   /* generous upper bound */
  if (est > 2 * n + 4096) est = 2 * n + 4096;
  free(t->cells); t->cells = NULL;
  free(t->mom); t->mom = NULL;
  t->cap = est;
  t->ncells = 0;
  t->cells = (Cell *)xmalloc(sizeof(Cell) * (size_t)t->cap);
  t->mom = (double *)xmalloc(sizeof(double) * (size_t)t->cap * t->nmt);
  memset(t->mom, 0, sizeof(double) * (size_t)t->cap * t->nmt);

  /* bounding box centre/half are recomputed from the (already sorted) data */
  double xmin = t->sx[0], xmax = t->sx[0], ymin = t->sy[0], ymax = t->sy[0];
  double zmin = t->sz[0], zmax = t->sz[0];
  for (i64 i = 1; i < n; i++) {
    if (t->sx[i] < xmin) xmin = t->sx[i];
    if (t->sx[i] > xmax) xmax = t->sx[i];
    if (t->sy[i] < ymin) ymin = t->sy[i];
    if (t->sy[i] > ymax) ymax = t->sy[i];
    if (t->sz[i] < zmin) zmin = t->sz[i];
    if (t->sz[i] > zmax) zmax = t->sz[i];
  }
  double span = xmax - xmin;
  if (ymax - ymin > span) span = ymax - ymin;
  if (zmax - zmin > span) span = zmax - zmin;
  t->root = build_rec(t, 0, n, 0.5 * (xmin + xmax), 0.5 * (ymin + ymax),
                      0.5 * (zmin + zmax), 0.5 * span * (1.0 + 1e-9) + 1e-30, 0);
}

/* Recompute the moment table for a different order, reusing the existing
   cell topology and the already-sorted particle array.  This avoids paying
   for the Morton sort a second time when the probe picks p > 0. */
static void moments_recompute(Tree *t, i32 p) {
  i32 nmt = NTERM(p);
  double *nm = (double *)xmalloc(sizeof(double) * (size_t)t->cap * nmt);
  memset(nm, 0, sizeof(double) * (size_t)t->cap * nmt);
  double pw[MAXP + 1], qw[MAXP + 1], rw[MAXP + 1];
  for (i64 id = 0; id < t->ncells; id++) {
    const Cell *c = &t->cells[id];
    double *Mo = nm + (size_t)id * nmt;
    const double gx = c->gx, gy = c->gy, gz = c->gz;
    for (i64 k = c->p0; k < c->p0 + c->cnt; k++) {
      double dx = t->sx[k] - gx, dy = t->sy[k] - gy, dz = t->sz[k] - gz;
      double mk = t->sm[k];
      /* mass enters exactly once: pw carries it, qw/rw are pure powers */
      pw[0] = mk; qw[0] = 1.0; rw[0] = 1.0;
      for (int a = 1; a <= p; a++) pw[a] = pw[a - 1] * dx;
      for (int b = 1; b <= p; b++) qw[b] = qw[b - 1] * dy;
      for (int cc = 1; cc <= p; cc++) rw[cc] = rw[cc - 1] * dz;
      for (int a = 0; a <= p; a++)
        for (int b = 0; a + b <= p; b++) {
          double ab = pw[a] * qw[b];
          for (int cc = 0; a + b + cc <= p; cc++)
            Mo[der_key(a, b, cc)] += ab * rw[cc];
        }
    }
  }
  free(t->mom);
  t->mom = nm;
  t->nmt = nmt;
  t->p = p;
}

static void tree_free(Tree *t) {
  free(t->cells); free(t->sx); free(t->sy); free(t->sz); free(t->sm);
  free(t->order); free(t->mom);
  memset(t, 0, sizeof(*t));
}

/* ================================================================== */
/*  Pair kernel: 8 targets x 8 sources, accumulating in registers.      */
/*  Lane j of accumulator t corresponds to source j for target t, so a  */
/*  horizontal reduce per target yields its partial sum.                */
/* ================================================================== */
#define PT 8      /* targets per block          */
#define NCH 10    /* accumulator chains         */
#define SRCW 4    /* sources per vector group   */

/* Accumulate the force on targets tx/ty/tz[0..ntgt) from sources
 * sx/sy/sz/sm[0..nsrc).  Results are ADDED into ax/ay/az.
 *
 * The inner loop is latency-bound on vsqrtpd (~20 cycles) and vdivpd
 * (~13 cycles), so NCH independent accumulator chains are interleaved over
 * the source axis; that alone lifts throughput from ~0.8 to ~9 Gpair/s.
 * The j == i term has dx = dy = dz = 0 and so contributes exactly zero. */
static void pair_block(const double *restrict sx, const double *restrict sy,
                       const double *restrict sz, const double *restrict sm,
                       i64 nsrc,
                       const double *restrict tx, const double *restrict ty,
                       const double *restrict tz, i64 ntgt,
                       double *restrict ax, double *restrict ay,
                       double *restrict az) {
  const __m256d ve = _mm256_set1_pd(EPS2);
  const __m256d vzero = _mm256_setzero_pd();
  const i64 nchunk = (i64)NCH * SRCW;
  i64 t = 0;
  for (; t + PT <= ntgt; t += PT) {
    __m256d TX[PT], TY[PT], TZ[PT];
    __m256d AX[PT][NCH], AY[PT][NCH], AZ[PT][NCH];
    for (int k = 0; k < PT; k++) {
      TX[k] = _mm256_set1_pd(tx[t + k]);
      TY[k] = _mm256_set1_pd(ty[t + k]);
      TZ[k] = _mm256_set1_pd(tz[t + k]);
      for (int c = 0; c < NCH; c++) {
        AX[k][c] = vzero; AY[k][c] = vzero; AZ[k][c] = vzero;
      }
    }
    i64 j = 0;
    for (; j + nchunk <= nsrc; j += nchunk) {
      __m256d vx[NCH], vy[NCH], vz[NCH], vm[NCH];
      for (int c = 0; c < NCH; c++) {
        vx[c] = _mm256_loadu_pd(sx + j + c * SRCW);
        vy[c] = _mm256_loadu_pd(sy + j + c * SRCW);
        vz[c] = _mm256_loadu_pd(sz + j + c * SRCW);
        vm[c] = _mm256_loadu_pd(sm + j + c * SRCW);
      }
      for (int k = 0; k < PT; k++) {
        for (int c = 0; c < NCH; c++) {
          __m256d dx = _mm256_sub_pd(vx[c], TX[k]);
          __m256d dy = _mm256_sub_pd(vy[c], TY[k]);
          __m256d dz = _mm256_sub_pd(vz[c], TZ[k]);
          __m256d r2 = _mm256_add_pd(
              _mm256_fmadd_pd(dz, dz,
                  _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dx, dx))), ve);
          __m256d w = _mm256_mul_pd(vm[c], inv_r3(r2));
          AX[k][c] = _mm256_fmadd_pd(dx, w, AX[k][c]);
          AY[k][c] = _mm256_fmadd_pd(dy, w, AY[k][c]);
          AZ[k][c] = _mm256_fmadd_pd(dz, w, AZ[k][c]);
        }
      }
    }
    for (int k = 0; k < PT; k++) {
      /* fold the NCH chains into three accumulators */
      __m256d fx = AX[k][0], fy = AY[k][0], fz = AZ[k][0];
      for (int c = 1; c < NCH; c++) {
        fx = _mm256_add_pd(fx, AX[k][c]);
        fy = _mm256_add_pd(fy, AY[k][c]);
        fz = _mm256_add_pd(fz, AZ[k][c]);
      }
      double b[4], sx_, sy_, sz_;
      _mm256_storeu_pd(b, fx); sx_ = (b[0] + b[1]) + (b[2] + b[3]);
      _mm256_storeu_pd(b, fy); sy_ = (b[0] + b[1]) + (b[2] + b[3]);
      _mm256_storeu_pd(b, fz); sz_ = (b[0] + b[1]) + (b[2] + b[3]);
      for (i64 jj = j; jj < nsrc; jj++) {
        double dx = sx[jj] - tx[t + k], dy = sy[jj] - ty[t + k], dz = sz[jj] - tz[t + k];
        double r2 = dx * dx + dy * dy + dz * dz + EPS2;
        double w = sm[jj] / (r2 * sqrt(r2));
        sx_ += dx * w; sy_ += dy * w; sz_ += dz * w;
      }
      ax[t + k] += sx_; ay[t + k] += sy_; az[t + k] += sz_;
    }
  }
  for (; t < ntgt; t++) {
    __m256d XT = _mm256_set1_pd(tx[t]), YT = _mm256_set1_pd(ty[t]);
    __m256d ZT = _mm256_set1_pd(tz[t]);
    __m256d AX[NCH], AY[NCH], AZ[NCH];
    for (int c = 0; c < NCH; c++) { AX[c] = vzero; AY[c] = vzero; AZ[c] = vzero; }
    i64 j = 0;
    for (; j + nchunk <= nsrc; j += nchunk) {
      for (int c = 0; c < NCH; c++) {
        __m256d vx = _mm256_loadu_pd(sx + j + c * SRCW);
        __m256d vy = _mm256_loadu_pd(sy + j + c * SRCW);
        __m256d vz = _mm256_loadu_pd(sz + j + c * SRCW);
        __m256d vm = _mm256_loadu_pd(sm + j + c * SRCW);
        __m256d dx = _mm256_sub_pd(vx, XT);
        __m256d dy = _mm256_sub_pd(vy, YT);
        __m256d dz = _mm256_sub_pd(vz, ZT);
        __m256d r2 = _mm256_add_pd(
            _mm256_fmadd_pd(dz, dz,
                _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dx, dx))), ve);
        __m256d w = _mm256_mul_pd(vm, inv_r3(r2));
        AX[c] = _mm256_fmadd_pd(dx, w, AX[c]);
        AY[c] = _mm256_fmadd_pd(dy, w, AY[c]);
        AZ[c] = _mm256_fmadd_pd(dz, w, AZ[c]);
      }
    }
    __m256d fx = AX[0], fy = AY[0], fz = AZ[0];
    for (int c = 1; c < NCH; c++) {
      fx = _mm256_add_pd(fx, AX[c]);
      fy = _mm256_add_pd(fy, AY[c]);
      fz = _mm256_add_pd(fz, AZ[c]);
    }
    double b[4], sx_, sy_, sz_;
    _mm256_storeu_pd(b, fx); sx_ = (b[0] + b[1]) + (b[2] + b[3]);
    _mm256_storeu_pd(b, fy); sy_ = (b[0] + b[1]) + (b[2] + b[3]);
    _mm256_storeu_pd(b, fz); sz_ = (b[0] + b[1]) + (b[2] + b[3]);
    for (; j < nsrc; j++) {
      double dx = sx[j] - tx[t], dy = sy[j] - ty[t], dz = sz[j] - tz[t];
      double r2 = dx * dx + dy * dy + dz * dz + EPS2;
      double w = sm[j] / (r2 * sqrt(r2));
      sx_ += dx * w; sy_ += dy * w; sz_ += dz * w;
    }
    ax[t] += sx_; ay[t] += sy_; az[t] += sz_;
  }
}

/* 4x4 double transpose across the two 128-bit lanes. */
static void transpose4(__m256d *v0, __m256d *v1, __m256d *v2, __m256d *v3) {
  __m256d t0 = _mm256_unpacklo_pd(*v0, *v1);
  __m256d t1 = _mm256_unpackhi_pd(*v0, *v1);
  __m256d t2 = _mm256_unpacklo_pd(*v2, *v3);
  __m256d t3 = _mm256_unpackhi_pd(*v2, *v3);
  *v0 = _mm256_permute2f128_pd(t0, t2, 0x20);
  *v1 = _mm256_permute2f128_pd(t1, t3, 0x20);
  *v2 = _mm256_permute2f128_pd(t0, t2, 0x31);
  *v3 = _mm256_permute2f128_pd(t1, t3, 0x31);
}

/* Exact O(N^2/2) summation exploiting Newton's third law.
 *
 * Equal masses make the pair term exactly antisymmetric, so one r^-3
 * evaluation (the sqrt+divide pair is the throughput bottleneck) serves both
 * directions.  Target blocks of SYM_BLK cross source blocks of SYM_BLK
 * strictly above the diagonal; the diagonal block is paired explicitly.
 * ax/ay/az must be zero on entry.
 */
#define SYM_BLK 4

static void direct_symmetric(double *restrict px, double *restrict py,
                             double *restrict pz, const double *restrict pm,
                             i64 n, double *restrict ax, double *restrict ay,
                             double *restrict az) {
  const __m256d ve = _mm256_set1_pd(EPS2);
  const double wmass = pm[0];
  i64 bi = 0;
  for (; bi + SYM_BLK <= n; bi += SYM_BLK) {
    __m256d TX[SYM_BLK], TY[SYM_BLK], TZ[SYM_BLK];
    for (int k = 0; k < SYM_BLK; k++) {
      TX[k] = _mm256_set1_pd(px[bi + k]);
      TY[k] = _mm256_set1_pd(py[bi + k]);
      TZ[k] = _mm256_set1_pd(pz[bi + k]);
    }
    __m256d AiX = _mm256_loadu_pd(ax + bi), AiY = _mm256_loadu_pd(ay + bi);
    __m256d AiZ = _mm256_loadu_pd(az + bi);
    i64 bj = bi + SYM_BLK;
    for (; bj + SYM_BLK <= n; bj += SYM_BLK) {
      __m256d JX = _mm256_loadu_pd(px + bj), JY = _mm256_loadu_pd(py + bj);
      __m256d JZ = _mm256_loadu_pd(pz + bj);
      __m256d fx[SYM_BLK], fy[SYM_BLK], fz[SYM_BLK];
      for (int k = 0; k < SYM_BLK; k++) {
        __m256d dx = _mm256_sub_pd(JX, TX[k]);
        __m256d dy = _mm256_sub_pd(JY, TY[k]);
        __m256d dz = _mm256_sub_pd(JZ, TZ[k]);
        __m256d r2 = _mm256_add_pd(
            _mm256_fmadd_pd(dz, dz,
                _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dx, dx))), ve);
        __m256d w = _mm256_mul_pd(_mm256_set1_pd(wmass), inv_r3(r2));
        fx[k] = _mm256_mul_pd(dx, w);
        fy[k] = _mm256_mul_pd(dy, w);
        fz[k] = _mm256_mul_pd(dz, w);
      }
      {
        __m256d a = fx[0], b = fx[1], c = fx[2], d = fx[3];
        transpose4(&a, &b, &c, &d);
        AiX = _mm256_add_pd(AiX, _mm256_add_pd(_mm256_add_pd(a, b),
                                               _mm256_add_pd(c, d)));
        a = fy[0]; b = fy[1]; c = fy[2]; d = fy[3];
        transpose4(&a, &b, &c, &d);
        AiY = _mm256_add_pd(AiY, _mm256_add_pd(_mm256_add_pd(a, b),
                                               _mm256_add_pd(c, d)));
        a = fz[0]; b = fz[1]; c = fz[2]; d = fz[3];
        transpose4(&a, &b, &c, &d);
        AiZ = _mm256_add_pd(AiZ, _mm256_add_pd(_mm256_add_pd(a, b),
                                               _mm256_add_pd(c, d)));
      }
      {
        __m256d sx = _mm256_add_pd(_mm256_add_pd(fx[0], fx[1]),
                                   _mm256_add_pd(fx[2], fx[3]));
        __m256d sy = _mm256_add_pd(_mm256_add_pd(fy[0], fy[1]),
                                   _mm256_add_pd(fy[2], fy[3]));
        __m256d sz = _mm256_add_pd(_mm256_add_pd(fz[0], fz[1]),
                                   _mm256_add_pd(fz[2], fz[3]));
        _mm256_storeu_pd(ax + bj, _mm256_sub_pd(_mm256_loadu_pd(ax + bj), sx));
        _mm256_storeu_pd(ay + bj, _mm256_sub_pd(_mm256_loadu_pd(ay + bj), sy));
        _mm256_storeu_pd(az + bj, _mm256_sub_pd(_mm256_loadu_pd(az + bj), sz));
      }
    }
    _mm256_storeu_pd(ax + bi, AiX);
    _mm256_storeu_pd(ay + bi, AiY);
    _mm256_storeu_pd(az + bi, AiZ);
    for (; bj < n; bj++)
      for (i64 a = bi; a < bi + SYM_BLK; a++) {
        double dx = px[bj] - px[a], dy = py[bj] - py[a], dz = pz[bj] - pz[a];
        double r2 = dx * dx + dy * dy + dz * dz + EPS2;
        double w = wmass / (r2 * sqrt(r2));
        ax[a] += dx * w; ay[a] += dy * w; az[a] += dz * w;
        ax[bj] -= dx * w; ay[bj] -= dy * w; az[bj] -= dz * w;
      }
    for (i64 a = bi; a < bi + SYM_BLK; a++)
      for (i64 b2 = a + 1; b2 < bi + SYM_BLK; b2++) {
        double dx = px[b2] - px[a], dy = py[b2] - py[a], dz = pz[b2] - pz[a];
        double r2 = dx * dx + dy * dy + dz * dz + EPS2;
        double w = wmass / (r2 * sqrt(r2));
        ax[a] += dx * w; ay[a] += dy * w; az[a] += dz * w;
        ax[b2] -= dx * w; ay[b2] -= dy * w; az[b2] -= dz * w;
      }
  }
  if (bi < n)
    pair_block(px, py, pz, pm, n, px + bi, py + bi, pz + bi, n - bi,
               ax + bi, ay + bi, az + bi);
}

static i64 read_file(const char *path, u8 **buf, size_t *len) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open input %s\n", path); exit(1); }
  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); die("seek"); }
  long sz = ftell(f);
  if (sz < 0) { fclose(f); die("ftell"); }
  rewind(f);
  u8 *b = (u8 *)xmalloc((size_t)sz + 64);
  size_t got = fread(b, 1, (size_t)sz, f);
  fclose(f);
  if (got != (size_t)sz) die("short read");
  *buf = b; *len = (size_t)sz;
  return (i64)(sz / 32);
}

/* Largest multipole order whose moment storage fits the memory budget.
 * cells ~ 2.2N for LEAFMAX=16; each cell costs sizeof(Cell) plus
 * NTERM(p) doubles of moments. */
static i32 max_order_for(i64 n) {
  double ncells = 2.2 * (double)n + 1024.0;
  double fixed = (double)n * (4 * 8 + 4) + ncells * (double)sizeof(Cell)
                 + (double)n * 3 * 8 * 2;
  for (i32 p = MAXP; p >= 0; p--) {
    if (fixed + ncells * (double)NTERM(p) * 8.0 <= MEM_LIMIT) return p;
  }
  return 0;
}

/* Measure the direct pair kernel's throughput on a small synthetic block so
 * the direct-vs-tree decision reflects this machine rather than a guess. */
static void probe_pair_rate(void) {
  const i64 nb = 4096;
  double *blk = (double *)xmalloc(sizeof(double) * (size_t)nb * 7);
  double *bx = blk, *by = blk + nb, *bz = blk + 2 * nb, *bm = blk + 3 * nb;
  double *oax = blk + 4 * nb, *oay = blk + 5 * nb, *oaz = blk + 6 * nb;
  u64 st = 0x123456789abcdefULL;
  for (i64 i = 0; i < nb; i++) {
    st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    double u = (double)(st >> 11) * (1.0 / 9007199254740992.0);
    st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    double v = (double)(st >> 11) * (1.0 / 9007199254740992.0);
    st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    double w = (double)(st >> 11) * (1.0 / 9007199254740992.0);
    bx[i] = u * 4.0 - 2.0; by[i] = v * 4.0 - 2.0; bz[i] = w * 4.0 - 2.0;
    bm[i] = 1.0 / (double)nb;
  }
  memset(oax, 0, sizeof(double) * (size_t)nb * 3);
  /* warm-up pass: page faults and clock ramp would otherwise dominate */
  pair_block(bx, by, bz, bm, nb, bx, by, bz, nb, oax, oay, oaz);
  memset(oax, 0, sizeof(double) * (size_t)nb * 3);
  double t0 = cpu_seconds();
  pair_block(bx, by, bz, bm, nb, bx, by, bz, nb, oax, oay, oaz);
  double dt = cpu_seconds() - t0;
  if (dt > 1e-9) g_pair_rate = (double)nb * (double)nb / dt;
  logf_("pair rate probe: %.3g pairs/s (%.4fs for %ld^2)\n",
        g_pair_rate, dt, (long)nb);
  free(blk);
}

/* ================================================================== */
/*  Multipole evaluation                                                */
/* ================================================================== */

/* Precomputed per-order factorial denominators and the alpha list. */
typedef struct {
  i32 n;                 /* number of multi-index terms for order p */
  i32 nterm;             /* NTERM(p)                                */
  const i32 (*idx)[3];   /* [n][3] multi-indices, ordered like der_key */
  const double *invfact; /* [n] 1/(a!b!c!)                            */
  const double *sign;    /* [n] (-1)^(a+b+c)                          */
} MomPlan;

static i32 g_alpha[MAXP + 1][NTERM_MAX][3];
static double g_invfact[MAXP + 1][NTERM_MAX];
static double g_sign[MAXP + 1][NTERM_MAX];
static int g_nalpha[MAXP + 1];

static void plan_init(void) {
  for (int p = 0; p <= MAXP; p++) {
    int k = 0;
    /* enumerate in increasing der_key order so that g_alpha[p][k] is the
       multi-index whose moments live at Mo[k] */
    for (int n = 0; n <= p; n++)
      for (int a = 0; a <= n; a++)
        for (int b = 0; a + b <= n; b++) {
          int c = n - a - b;
          if (der_key(a, b, c) != k) {
            fprintf(stderr, "internal: der_key ordering mismatch\n");
            exit(1);
          }
          g_alpha[p][k][0] = a; g_alpha[p][k][1] = b; g_alpha[p][k][2] = c;
          g_invfact[p][k] = 1.0 / ((double)tgamma(a + 1) * tgamma(b + 1) * tgamma(c + 1));
          g_sign[p][k] = ((a + b + c) & 1) ? -1.0 : 1.0;
          k++;
        }
    g_nalpha[p] = k;
  }
}

/* D_beta(R) = P_beta(R) / r^(2|beta|+1).
 * `rpow` holds r^(2n+1) for some level n; when |beta| = n + extra the missing
 * factor r^(-2*extra) is folded in here, so callers can pass one power table
 * for the whole contraction. */
static inline double eval_D(const double R[3], int key, double rpow, int extra) {
  const int o = DER_OFF[key];
  const int o1 = DER_OFF[key + 1];
  double s = 0.0;
  double xp[DER_MAXP + 2], yp[DER_MAXP + 2], zp[DER_MAXP + 2];
  xp[0] = yp[0] = zp[0] = 1.0;
  for (int i = 1; i <= DER_MAXP + 1; i++) {
    xp[i] = xp[i-1] * R[0]; yp[i] = yp[i-1] * R[1]; zp[i] = zp[i-1] * R[2];
  }
  for (int i = o; i < o1; i++) {
    const DerMon *dm = &DER_MON[i];
    s += (double)dm->coeff * xp[dm->ea] * yp[dm->eb] * zp[dm->ec];
  }
  double scale = rpow;
  for (int i = 0; i < extra; i++) scale *= R[0]*R[0] + R[1]*R[1] + R[2]*R[2];
  return s / scale;
}

/* Add the multipole force from cell `id` (order p) onto target (tx,ty,tz). */
static void multipole_acc(const Tree *t, i64 id, int p,
                          double tx, double ty, double tz,
                          double *ax, double *ay, double *az) {
  const Cell *c = &t->cells[id];
  const double *Mo = t->mom + (size_t)id * t->nmt;
  double R[3] = { tx - c->gx, ty - c->gy, tz - c->gz };
  double r2 = R[0]*R[0] + R[1]*R[1] + R[2]*R[2];
  double r = sqrt(r2);
  double rinv = 1.0 / r;
  /* rpow[n] = r^(2n+1), the denominator of D_alpha for |alpha| = n */
  double rpow[MAXP + 3];
  rpow[0] = r;
  for (int n = 1; n <= MAXP + 2; n++) rpow[n] = rpow[n - 1] * r2;

  int n = g_nalpha[p];
  double fx = 0, fy = 0, fz = 0;
  for (int k = 0; k < n; k++) {
    double mk = Mo[k];
    if (mk == 0.0) continue;
    int a = g_alpha[p][k][0], b = g_alpha[p][k][1], cc = g_alpha[p][k][2];
    double cf = mk * g_invfact[p][k] * g_sign[p][k];
    /* derivative keys for alpha + e_j, only when |alpha|+1 <= table max */
    int ord = a + b + cc;
    if (ord + 1 <= DER_MAXP) {
      /* D_{alpha+e_j} has denominator r^(2*ord+3) = rpow[ord+1] */
      double pw = rpow[ord + 1];
      fx += cf * eval_D(R, der_key(a + 1, b, cc), pw, 0);
      fy += cf * eval_D(R, der_key(a, b + 1, cc), pw, 0);
      fz += cf * eval_D(R, der_key(a, b, cc + 1), pw, 0);
    }
  }
  *ax += fx; *ay += fy; *az += fz;
}

/* ================================================================== */
/*  Tree walk                                                           */
/* ================================================================== */

#define STACK_MAX 4096

/* Leaf particles are stored contiguously, so a leaf is summed with the same
 * blocked kernel used by the direct path.  LEAF_BLK targets share one pass
 * over the leaf's sources, amortising the loads. */
#define LEAF_BLK 4

static void leaf_sum(const Tree *t, const Cell *c,
                     const double *tx, const double *ty, const double *tz,
                     double *ax, double *ay, double *az) {
  const double *restrict sx = t->sx, *restrict sy = t->sy;
  const double *restrict sz = t->sz, *restrict sm = t->sm;
  const i32 p0 = c->p0, cnt = c->cnt;
  const __m256d ve = _mm256_set1_pd(EPS2);
  const __m256d vzero = _mm256_setzero_pd();
  for (i32 k = 0; k < cnt; k += LEAF_BLK) {
    i32 nb = cnt - k; if (nb > LEAF_BLK) nb = LEAF_BLK;
    __m256d TX[LEAF_BLK], TY[LEAF_BLK], TZ[LEAF_BLK];
    __m256d AX[LEAF_BLK], AY[LEAF_BLK], AZ[LEAF_BLK];
    for (int q = 0; q < nb; q++) {
      TX[q] = _mm256_set1_pd(tx[k + q]);
      TY[q] = _mm256_set1_pd(ty[k + q]);
      TZ[q] = _mm256_set1_pd(tz[k + q]);
      AX[q] = vzero; AY[q] = vzero; AZ[q] = vzero;
    }
    i32 j = p0;
    for (; j + 4 <= p0 + cnt; j += 4) {
      __m256d VX = _mm256_loadu_pd(sx + j), VY = _mm256_loadu_pd(sy + j);
      __m256d VZ = _mm256_loadu_pd(sz + j), VM = _mm256_loadu_pd(sm + j);
      for (int q = 0; q < nb; q++) {
        __m256d dx = _mm256_sub_pd(VX, TX[q]);
        __m256d dy = _mm256_sub_pd(VY, TY[q]);
        __m256d dz = _mm256_sub_pd(VZ, TZ[q]);
        __m256d r2 = _mm256_add_pd(
            _mm256_fmadd_pd(dz, dz,
                _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dx, dx))), ve);
        __m256d w = _mm256_mul_pd(VM, inv_r3(r2));
        AX[q] = _mm256_fmadd_pd(dx, w, AX[q]);
        AY[q] = _mm256_fmadd_pd(dy, w, AY[q]);
        AZ[q] = _mm256_fmadd_pd(dz, w, AZ[q]);
      }
    }
    for (int q = 0; q < nb; q++) {
      double b[4], sx_, sy_, sz_;
      _mm256_storeu_pd(b, AX[q]); sx_ = (b[0] + b[1]) + (b[2] + b[3]);
      _mm256_storeu_pd(b, AY[q]); sy_ = (b[0] + b[1]) + (b[2] + b[3]);
      _mm256_storeu_pd(b, AZ[q]); sz_ = (b[0] + b[1]) + (b[2] + b[3]);
      for (i32 jj = j; jj < p0 + cnt; jj++) {
        double dx = sx[jj] - tx[k + q], dy = sy[jj] - ty[k + q], dz = sz[jj] - tz[k + q];
        double r2 = dx * dx + dy * dy + dz * dz + EPS2;
        double w = sm[jj] / (r2 * sqrt(r2));
        sx_ += dx * w; sy_ += dy * w; sz_ += dz * w;
      }
      ax[k + q] += sx_; ay[k + q] += sy_; az[k + q] += sz_;
    }
  }
}

/* Walk the tree for one target.  Cell metadata is pre-copied into flat
 * structure-of-arrays so the traversal touches one cache line per cell
 * instead of striding over 104-byte structs. */
typedef struct {
  const double *gx, *gy, *gz, *rad2;
  const i32 *child0, *isleaf, *p0, *cnt;
  const double *mom;
  i32 nmt;
} WalkView;

static void walk_one(const Tree *t, int p, double th2,
                     double tx, double ty, double tz,
                     double *ax, double *ay, double *az) {
  i32 stack[STACK_MAX];
  int sp = 0;
  stack[sp++] = (i32)t->root;
  const double *sxp = t->sx, *syp = t->sy, *szp = t->sz, *smp = t->sm;
  while (sp > 0) {
    i32 id = stack[--sp];
    const Cell *c = &t->cells[id];
    double dx = tx - c->gx, dy = ty - c->gy, dz = tz - c->gz;
    double r2 = dx * dx + dy * dy + dz * dz;
    if (c->isleaf) {
      /* exact pair sum; the j == i term contributes exactly zero */
      double sx_ = 0, sy_ = 0, sz_ = 0;
      i32 j = c->p0, p1 = j + c->cnt;
      for (; j + 4 <= p1; j += 4) {
        __m256d vx = _mm256_loadu_pd(sxp + j), vy = _mm256_loadu_pd(syp + j);
        __m256d vz = _mm256_loadu_pd(szp + j), vm = _mm256_loadu_pd(smp + j);
        __m256d ddx = _mm256_sub_pd(vx, _mm256_set1_pd(tx));
        __m256d ddy = _mm256_sub_pd(vy, _mm256_set1_pd(ty));
        __m256d ddz = _mm256_sub_pd(vz, _mm256_set1_pd(tz));
        __m256d rr = _mm256_add_pd(_mm256_fmadd_pd(ddz, ddz,
                       _mm256_fmadd_pd(ddy, ddy, _mm256_mul_pd(ddx, ddx))),
                       _mm256_set1_pd(EPS2));
        __m256d w = _mm256_mul_pd(vm, inv_r3(rr));
        double b[4];
        _mm256_storeu_pd(b, _mm256_mul_pd(ddx, w));
        sx_ += (b[0] + b[1]) + (b[2] + b[3]);
        _mm256_storeu_pd(b, _mm256_mul_pd(ddy, w));
        sy_ += (b[0] + b[1]) + (b[2] + b[3]);
        _mm256_storeu_pd(b, _mm256_mul_pd(ddz, w));
        sz_ += (b[0] + b[1]) + (b[2] + b[3]);
      }
      for (; j < p1; j++) {
        double ddx = sxp[j] - tx, ddy = syp[j] - ty, ddz = szp[j] - tz;
        double rr = ddx * ddx + ddy * ddy + ddz * ddz + EPS2;
        double w = smp[j] / (rr * sqrt(rr));
        sx_ += ddx * w; sy_ += ddy * w; sz_ += ddz * w;
      }
      *ax += sx_; *ay += sy_; *az += sz_;
      continue;
    }
    /* Opening test.  Two conditions must both hold before a cell's
       expansion is used:
         (a) theta * |R| >= rad, the usual centre-of-mass criterion; and
         (b) theta * dbox >= half, where dbox is the distance from the target
             to the cell's bounding box.
       (b) matters for diffuse cells whose COM sits far from the target while
       the box still looms large -- without it the expansion is evaluated far
       outside its radius of convergence and the error explodes (visible on
       the Jaffe model, whose density is very centrally concentrated). */
    double rad = c->rad;
    if (r2 * th2 >= rad * rad) {
      double h = c->half;
      double ex = fabs(tx - c->cx) - h; if (ex < 0) ex = 0;
      double ey = fabs(ty - c->cy) - h; if (ey < 0) ey = 0;
      double ez = fabs(tz - c->cz) - h; if (ez < 0) ez = 0;
      double dbox2 = ex * ex + ey * ey + ez * ez;
      if (dbox2 * th2 >= h * h) {
        multipole_acc(t, id, p, tx, ty, tz, ax, ay, az);
        continue;
      }
    }
    if (sp + 8 > STACK_MAX) {
      /* pathologically deep tree: sum this cell exactly rather than overflow */
      i32 j = c->p0, p1 = j + c->cnt;
      for (; j < p1; j++) {
        double ddx = sxp[j] - tx, ddy = syp[j] - ty, ddz = szp[j] - tz;
        double rr = ddx * ddx + ddy * ddy + ddz * ddz + EPS2;
        double w = smp[j] / (rr * sqrt(rr));
        *ax += ddx * w; *ay += ddy * w; *az += ddz * w;
      }
      continue;
    }
    for (int k = 0; k < 8; k++) {
      i32 ch = c->child[k];
      if (ch >= 0) stack[sp++] = ch;
    }
  }
}

int main(int argc, char **argv) {
  g_t0 = cpu_seconds();
  const char *inpath = (argc > 1) ? argv[1] : NULL;
  const char *outpath = (argc > 2) ? argv[2] : NULL;
  if (!inpath || !outpath) {
    fprintf(stderr, "usage: %s INPUT OUTPUT\n", argv[0]);
    return 1;
  }
  const char *ev = getenv("GALAXY_CPU_LIMIT_S");
  if (ev && *ev) {
    double v = atof(ev);
    if (v > 0) g_limit = v;
  }
  if (getenv("GALAXY_VERBOSE")) g_verbose = 1;
  /* Reserve a slice of the budget for writing output and unforeseen overhead.
   * Everything must be done well before the hard limit. */
  double budget = g_limit * 0.80;

  u8 *raw = NULL; size_t rawlen = 0;
  i64 n = read_file(inpath, &raw, &rawlen);
  if (n <= 0) die("empty input");

  const double *rd = (const double *)raw;
  double *px = (double *)xmalloc(sizeof(double) * (size_t)n);
  double *py = (double *)xmalloc(sizeof(double) * (size_t)n);
  double *pz = (double *)xmalloc(sizeof(double) * (size_t)n);
  double *pm = (double *)xmalloc(sizeof(double) * (size_t)n);
  for (i64 i = 0; i < n; i++) {
    pm[i] = rd[4 * i + 0];
    px[i] = rd[4 * i + 1];
    py[i] = rd[4 * i + 2];
    pz[i] = rd[4 * i + 3];
  }
  free(raw);
  logf_("n=%lld read at t=%.3f limit=%.3f\n", (long long)n, elapsed(), g_limit);

  /* pair_block and walk_one accumulate into these, so they must start at 0 */
  double *axs = (double *)xmalloc(sizeof(double) * (size_t)n);
  double *ays = (double *)xmalloc(sizeof(double) * (size_t)n);
  double *azs = (double *)xmalloc(sizeof(double) * (size_t)n);
  memset(axs, 0, sizeof(double) * (size_t)n);
  memset(ays, 0, sizeof(double) * (size_t)n);
  memset(azs, 0, sizeof(double) * (size_t)n);
  double *axo = (double *)xmalloc(sizeof(double) * (size_t)n);
  double *ayo = (double *)xmalloc(sizeof(double) * (size_t)n);
  double *azo = (double *)xmalloc(sizeof(double) * (size_t)n);

  plan_init();
  probe_pair_rate();

  /* ---- exact direct path when it fits the budget ---- */
  {
    /* Throughput is measured on this machine at run time; see the probe below.
       A conservative fallback is used until the probe completes. */
    double rate = g_pair_rate;
    double pairs = 0.5 * (double)n * (double)n;
    double need = pairs / rate;
    /* Equal masses make the pair term exactly antisymmetric, which halves the
       number of r^-3 evaluations.  Verify before relying on it. */
    int equal_mass = 1;
    for (i64 i = 1; i < n && i < 4096; i++)
      if (pm[i] != pm[0]) { equal_mass = 0; break; }
    if (!equal_mass) need *= 2.0;

    if (need < budget * 0.75) {
      logf_("direct path: %.3g pairs, predicted %.2fs of budget %.2fs (equal_mass=%d)\n",
            pairs, need, budget, equal_mass);
      /* work in sorted (Morton) order so the symmetric kernel's in-place
         source updates stay within the same array */
      double *sxp = px, *syp = py, *szp = pz;
      if (equal_mass) {
        /* sort first: reuse the tree machinery for ordering only */
        Tree T;
        memset(&T, 0, sizeof(T));
        tree_build(&T, pm, px, py, pz, n, 0, NULL);
        direct_symmetric(T.sx, T.sy, T.sz, T.sm, n, axs, ays, azs);
        for (i64 s = 0; s < n; s++) {
          i32 o = T.order[s];
          axo[o] = axs[s]; ayo[o] = ays[s]; azo[o] = azs[s];
        }
        tree_free(&T);
      } else {
        for (i64 i0 = 0; i0 < n; i0 += 16384) {
          i64 i1 = i0 + 16384; if (i1 > n) i1 = n;
          pair_block(sxp, syp, szp, pm, n, sxp + i0, syp + i0, szp + i0,
                     i1 - i0, axs + i0, ays + i0, azs + i0);
        }
        memcpy(axo, axs, sizeof(double) * (size_t)n);
        memcpy(ayo, ays, sizeof(double) * (size_t)n);
        memcpy(azo, azs, sizeof(double) * (size_t)n);
      }
      free(px); free(py); free(pz); free(pm);
      px = py = pz = pm = NULL;
      goto write_out;
    }
    logf_("direct path rejected: predicted %.2fs vs budget %.2fs\n", need, budget);
  }

  /* ---- tree path ---- */
  {
    static Tree T;
    memset(&T, 0, sizeof(T));
    i32 pmax = max_order_for(n);
    if (pmax > MAXP) pmax = MAXP;
    if (pmax < 0) pmax = 0;

    double sorted_at = 0.0;
    tree_build(&T, pm, px, py, pz, n, 0, &sorted_at);
    logf_("tree built: ncells=%lld at t=%.3f (sorted %.3f)\n",
          (long long)T.ncells, elapsed(), sorted_at);

    free(px); free(py); free(pz); free(pm);
    px = py = pz = pm = NULL;

    /* Choose (theta, p) from the budget.  Probing is expensive (each order
       needs moments_recompute plus a walk sample), so probe orders from
       CHEAP to EXPENSIVE under one hard total cap, and keep the last order
       that still fits.  Accuracy rises with p and falls with theta, so the
       goal is the largest p whose cheapest-fitting theta leaves headroom. */
    double probe_total_cap = budget * 0.06;
    if (probe_total_cap > 0.50) probe_total_cap = 0.50;
    double probe_used = 0.0;
    double per_particle[MAXP + 1];
    for (int p = 0; p <= MAXP; p++) per_particle[p] = -1.0;

    i64 nsample = 384;
    if (nsample > n) nsample = n;
    i64 stride = n / nsample; if (stride < 1) stride = 1;
    i32 *sidx = (i32 *)xmalloc(sizeof(i32) * (size_t)nsample);
    i64 ns = 0;
    for (i64 i = 0; i < n && ns < nsample; i += stride) sidx[ns++] = (i32)i;

    for (i32 p = 0; p <= pmax; p++) {
      if (probe_used >= probe_total_cap) break;
      if (p > 0) moments_recompute(&T, p);
      double t0 = elapsed();
      i64 used = 0;
      for (i64 k = 0; k < ns; k++) {
        if ((k & 31) == 0 && elapsed() - t0 > probe_total_cap - probe_used) break;
        i32 q = sidx[k];
        double a0 = 0, a1 = 0, a2 = 0;
        walk_one(&T, p, PROBE_THETA * PROBE_THETA,
                 T.sx[q], T.sy[q], T.sz[q], &a0, &a1, &a2);
        used++;
      }
      double dt = elapsed() - t0;
      probe_used += dt;
      if (used >= 8 && dt > 0) per_particle[p] = dt / (double)used;
      logf_("probe p=%d: %.4e s/particle (%lld samples, %.3fs)\n",
            p, per_particle[p], (long long)used, dt);
      if (getenv("GALAXY_DBG_M2")) {
        /* verify moments_recompute produced non-zero higher moments */
        double mx = 0; i32 nmt = T.nmt;
        for (i64 id = 0; id < T.ncells && id < 50; id++) {
          const double *Mo = T.mom + (size_t)id * nmt;
          for (i32 q = 1; q < nmt; q++) if (fabs(Mo[q]) > mx) mx = fabs(Mo[q]);
        }
        fprintf(stderr,"DBG_M2 p=%d nmt=%d max|higher moment| over 50 cells=%.6e\n",
                p, nmt, mx);
      }
      if (getenv("GALAXY_DBG_ERR")) {
        double se = 0; i64 ne = 0;
        for (i64 k = 0; k < used && k < 64; k++) {
          i32 q = sidx[k];
          double a0=0,a1=0,a2=0;
          walk_one(&T, p, PROBE_THETA*PROBE_THETA, T.sx[q],T.sy[q],T.sz[q],&a0,&a1,&a2);
          double bx=0,by=0,bz=0;
          for (i64 j=0;j<T.n;j++){
            double dx=T.sx[j]-T.sx[q],dy=T.sy[j]-T.sy[q],dz=T.sz[j]-T.sz[q];
            double rr=dx*dx+dy*dy+dz*dz+EPS2; double w=T.sm[j]/(rr*sqrt(rr));
            bx+=dx*w;by+=dy*w;bz+=dz*w;
          }
          double em=sqrt(bx*bx+by*by+bz*bz);
          double d=sqrt((a0-bx)*(a0-bx)+(a1-by)*(a1-by)+(a2-bz)*(a2-bz));
          if(em>0){double e=d/em; se+=e*e; ne++;}
          if(k<6) fprintf(stderr,"   q=%d |a_tree|=%.4e |a_exact|=%.4e rel=%.3e\n",q,
                  sqrt(a0*a0+a1*a1+a2*a2), em, em>0?d/em:-1.0);
        }
        fprintf(stderr,"DBG_ERR p=%d theta=%.2f rms over %lld = %.4e\n",p,PROBE_THETA,(long long)ne,sqrt(se/(ne>0?ne:1)));
        if (getenv("GALAXY_DBG_SWEEP")) {
          for (int ti = 0; ti < NTHETA; ti++) {
            double th = THETAS[ti], se2 = 0; i64 ne2 = 0;
            for (i64 k = 0; k < used && k < 64; k++) {
              i32 q = sidx[k];
              double a0=0,a1=0,a2=0;
              walk_one(&T, p, th*th, T.sx[q],T.sy[q],T.sz[q],&a0,&a1,&a2);
              double bx=0,by=0,bz=0;
              for (i64 j=0;j<T.n;j++){
                double dx=T.sx[j]-T.sx[q],dy=T.sy[j]-T.sy[q],dz=T.sz[j]-T.sz[q];
                double rr=dx*dx+dy*dy+dz*dz+EPS2; double w=T.sm[j]/(rr*sqrt(rr));
                bx+=dx*w;by+=dy*w;bz+=dz*w;
              }
              double em=sqrt(bx*bx+by*by+bz*bz);
              double d=sqrt((a0-bx)*(a0-bx)+(a1-by)*(a1-by)+(a2-bz)*(a2-bz));
              if(em>0){double e=d/em; se2+=e*e; ne2++;}
            }
            fprintf(stderr,"SWEEP p=%d theta=%.2f rms=%.4e\n",p,th,sqrt(se2/(ne2>0?ne2:1)));
          }
        }
      }
      /* stop early if this order cannot fit even at the loosest theta */
      if (per_particle[p] > 0) {
        double cheapest = per_particle[p] *
            pow(PROBE_THETA / THETAS[NTHETA - 1], TH_EXP) * (double)n * PRED_MARGIN;
        if (cheapest > (budget - elapsed()) * SAFETY) break;
      }
    }

    /* pick the most accurate feasible (p, theta): prefer higher p, then
       smaller theta */
    double best_theta = THETAS[NTHETA - 1];
    i32 best_p = 0;
    double best_pred = 0.0;
    int found = 0;
    double avail = (budget - elapsed()) * SAFETY;
    for (i32 p = pmax; p >= 0 && !found; p--) {
      if (per_particle[p] <= 0) continue;
      for (int ti = 0; ti < NTHETA; ti++) {
        double th = THETAS[ti];
        double pred = per_particle[p] * pow(PROBE_THETA / th, TH_EXP) *
                      (double)n * PRED_MARGIN;
        if (pred <= avail) {
          best_theta = th; best_p = p; best_pred = pred; found = 1;
          break;
        }
      }
    }
    if (!found) {
      /* nothing is predicted to fit: take the cheapest usable setting and
         let the chunk guard degrade further if needed. */
      for (i32 p = 0; p <= pmax; p++)
        if (per_particle[p] > 0) { best_p = p; break; }
      best_theta = THETAS[NTHETA - 1];
      best_pred = per_particle[best_p] *
          pow(PROBE_THETA / best_theta, TH_EXP) * (double)n * PRED_MARGIN;
      logf_("WARNING: no candidate fits (cheapest pred %.2fs vs available %.2fs)\n",
            best_pred, avail);
    }
    free(sidx);
    logf_("chosen theta=%.2f p=%d pred=%.2fs remaining=%.2fs\n",
          best_theta, best_p, best_pred, budget - elapsed());

    if (T.p != best_p) moments_recompute(&T, best_p);

    /* ---- compute in chunks, degrading if the projection stops fitting ---- */
    i64 chunk = 1 << 15;
    if (n < chunk) chunk = n;
    double cur_theta = best_theta, cur_th2 = cur_theta * cur_theta;
    i32 cur_p = best_p;
    for (i64 i0 = 0; i0 < n; i0 += chunk) {
      i64 i1 = i0 + chunk; if (i1 > n) i1 = n;
      double t0 = elapsed();
      for (i64 s = i0; s < i1; s++) {
        double a0 = 0, a1 = 0, a2 = 0;
        walk_one(&T, cur_p, cur_th2, T.sx[s], T.sy[s], T.sz[s], &a0, &a1, &a2);
        axs[s] = a0; ays[s] = a1; azs[s] = a2;
      }
      double dt = elapsed() - t0;
      double used = elapsed();
      double remain_parts = (double)(n - i1);
      double rate = (dt > 1e-9) ? (double)(i1 - i0) / dt : 1e30;
      double proj = used + remain_parts / rate;
      if (i1 < n && proj > budget * 0.97) {
        if (cur_theta < 1.6) {
          cur_theta *= 1.4; cur_th2 = cur_theta * cur_theta;
          logf_("degrade theta -> %.3f at %lld/%lld\n",
                cur_theta, (long long)i1, (long long)n);
        } else if (cur_p > 0) {
          cur_p = 0;
          if (T.p != 0) moments_recompute(&T, 0);
          logf_("degrade p -> 0 at %lld/%lld\n", (long long)i1, (long long)n);
        }
      }
    }

    for (i64 s = 0; s < n; s++) {
      i32 o = T.order[s];
      axo[o] = axs[s]; ayo[o] = ays[s]; azo[o] = azs[s];
    }
    tree_free(&T);
  }

write_out:
  {
    FILE *f = fopen(outpath, "wb");
    if (!f) { fprintf(stderr, "cannot open output %s\n", outpath); return 1; }
    double *buf = (double *)xmalloc(sizeof(double) * 3 * 8192);
    i64 off = 0;
    while (off < n) {
      i64 cnt = n - off; if (cnt > 8192) cnt = 8192;
      for (i64 k = 0; k < cnt; k++) {
        buf[3 * k + 0] = axo ? axo[off + k] : axs[off + k];
        buf[3 * k + 1] = ayo ? ayo[off + k] : ays[off + k];
        buf[3 * k + 2] = azo ? azo[off + k] : azs[off + k];
        for (int c = 0; c < 3; c++)
          if (!isfinite(buf[3 * k + c])) buf[3 * k + c] = 0.0;
      }
      if (fwrite(buf, sizeof(double) * 3, (size_t)cnt, f) != (size_t)cnt) {
        fclose(f); die("short write");
      }
      off += cnt;
    }
    free(buf);
    if (fflush(f) != 0) { fclose(f); die("flush"); }
    if (fsync(fileno(f)) != 0) { /* not fatal */ }
    fclose(f);
  }
  logf_("done at t=%.3f (limit %.3f)\n", elapsed(), g_limit);
  free(axs); free(ays); free(azs);
  free(axo); free(ayo); free(azo);
  free(px); free(py); free(pz); free(pm);
  return 0;
}
