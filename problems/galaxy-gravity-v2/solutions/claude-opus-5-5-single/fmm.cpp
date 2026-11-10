// Cartesian FMM (dual tree traversal) for Plummer-softened gravity, double precision.
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <cstdint>

typedef double v4d __attribute__((vector_size(32)));

#ifndef KERN
#define KERN "kern_p8.h"
#endif
#include KERN

static const double EPS2 = 1e-4;

struct P4 { double x, y, z, m; };

struct Cell {
  double cx, cy, cz, r;   // expansion centre (centre of mass) and max particle distance
  double mass;
  double bx, by, bz, bh;  // box centre and half size
  int begin, count;
  int child, nchild;      // nchild==0 -> leaf
};

static double cpu_now() {
  struct timespec ts;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

#ifdef FINAL
static double env_d(const char*, double d) { return d; }
#define GETENV(n) nullptr
#else
static double env_d(const char* n, double d) { const char* s = getenv(n); return s ? atof(s) : d; }
#define GETENV(n) getenv(n)
#endif

// ---------------------------------------------------------------- globals
static int N;
static P4* parts;          // sorted particles
static int* orig;          // original index of sorted particle
static std::vector<Cell> cells;
static double* Mexp;       // [ncell+1][NCOEF_PAD]
static double* Lexp;       // [ncell][NCOEF_PAD]
static double *accx, *accy, *accz;
static int POS3[P_ORDER + 2][P_ORDER + 2][P_ORDER + 2];
static double INVFACT[P_ORDER + 2];

// ---------------------------------------------------------------- tree
static void build_tree(int leafmax) {
  double mn[3] = {1e300, 1e300, 1e300}, mx[3] = {-1e300, -1e300, -1e300};
  for (int i = 0; i < N; i++) {
    const double* p = &parts[i].x;
    for (int d = 0; d < 3; d++) { mn[d] = std::min(mn[d], p[d]); mx[d] = std::max(mx[d], p[d]); }
  }
  double h = 0;
  for (int d = 0; d < 3; d++) h = std::max(h, 0.5 * (mx[d] - mn[d]));
  h = h * 1.000001 + 1e-300;
  cells.reserve(4 * N / std::max(1, leafmax) + 1024);
  Cell root;
  memset(&root, 0, sizeof root);
  root.bx = 0.5 * (mn[0] + mx[0]); root.by = 0.5 * (mn[1] + mx[1]); root.bz = 0.5 * (mn[2] + mx[2]);
  root.bh = h; root.begin = 0; root.count = N;
  cells.push_back(root);
  std::vector<P4> tmpp(N);
  std::vector<int> tmpo(N);
  std::vector<unsigned char> oct(N);
  const double hmin = h * 1e-12;
  for (size_t ci = 0; ci < cells.size(); ci++) {
    Cell c = cells[ci];
    if (c.count <= leafmax || c.bh < hmin) continue;
    int cnt[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = c.begin; i < c.begin + c.count; i++) {
      int o = (parts[i].x >= c.bx) | ((parts[i].y >= c.by) << 1) | ((parts[i].z >= c.bz) << 2);
      oct[i] = (unsigned char)o; cnt[o]++;
    }
    int off[8], s = c.begin;
    for (int o = 0; o < 8; o++) { off[o] = s; s += cnt[o]; }
    int st[8];
    memcpy(st, off, sizeof st);
    for (int i = c.begin; i < c.begin + c.count; i++) {
      int o = oct[i];
      tmpp[st[o]] = parts[i]; tmpo[st[o]] = orig[i]; st[o]++;
    }
    memcpy(parts + c.begin, &tmpp[c.begin], sizeof(P4) * c.count);
    memcpy(orig + c.begin, &tmpo[c.begin], sizeof(int) * c.count);
    int first = (int)cells.size(), nch = 0;
    double hh = 0.5 * c.bh;
    for (int o = 0; o < 8; o++) {
      if (!cnt[o]) continue;
      Cell d;
      memset(&d, 0, sizeof d);
      d.bx = c.bx + ((o & 1) ? hh : -hh);
      d.by = c.by + ((o & 2) ? hh : -hh);
      d.bz = c.bz + ((o & 4) ? hh : -hh);
      d.bh = hh; d.begin = off[o]; d.count = cnt[o];
      cells.push_back(d);
      nch++;
    }
    cells[ci].child = first; cells[ci].nchild = nch;
  }
}

// ---------------------------------------------------------------- expansions (generic, small cost)
static inline void powtab(double x, double* p) {  // p[t] = x^t / t!
  p[0] = 1.0;
  for (int t = 1; t <= P_ORDER; t++) p[t] = p[t - 1] * x * (1.0 / t);
}

static void p2m(Cell& c, double* M) {
  alignas(32) v4d acc[NCOEF];
  for (int i = 0; i < NCOEF; i++) acc[i] = (v4d){0, 0, 0, 0};
  for (int j0 = 0; j0 < c.count; j0 += 4) {
    alignas(32) double wx[4], wy[4], wz[4], wm[4];
    for (int k = 0; k < 4; k++) {
      if (j0 + k < c.count) {
        const P4& q = parts[c.begin + j0 + k];
        wx[k] = c.cx - q.x; wy[k] = c.cy - q.y; wz[k] = c.cz - q.z; wm[k] = q.m;
      } else { wx[k] = wy[k] = wz[k] = 0.0; wm[k] = 0.0; }
    }
    p2m4(*(v4d*)wx, *(v4d*)wy, *(v4d*)wz, *(v4d*)wm, acc);
  }
  for (int i = 0; i < NCOEF; i++) M[i] = acc[i][0] + acc[i][1] + acc[i][2] + acc[i][3];
  for (int i = NCOEF; i < NCOEF_PAD; i++) M[i] = 0;
}

// M_parent[k] += sum_{j<=k} Mc[j] * (-d)^(k-j)/(k-j)!, d = child centre - parent centre
static void m2m_add(const double* Mc, double dx, double dy, double dz, double* Mp) {
  double ex[P_ORDER + 1], ey[P_ORDER + 1], ez[P_ORDER + 1];
  powtab(-dx, ex); powtab(-dy, ey); powtab(-dz, ez);
  m2m_gen(Mc, ex, ey, ez, Mp);
}

// Lc[l] += sum_{k>=l,|k|<=P} Lp[k] t^(k-l)/(k-l)!, t = child centre - parent centre
static void l2l_add(const double* Lp, double dx, double dy, double dz, double* Lc) {
  double ex[P_ORDER + 1], ey[P_ORDER + 1], ez[P_ORDER + 1];
  powtab(dx, ex); powtab(dy, ey); powtab(dz, ez);
  l2l_gen(Lp, ex, ey, ez, Lc);
}

static int L2P_N;                    // number of monomials with |l| <= P-1
static int L2P_IX[NCOEF], L2P_IY[NCOEF], L2P_IZ[NCOEF];

static void l2p_cell(const Cell& c, const double* L) {
  for (int j0 = 0; j0 < c.count; j0 += 4) {
    alignas(32) double vx[4], vy[4], vz[4];
    for (int k = 0; k < 4; k++) {
      const P4& q = parts[c.begin + std::min(j0 + k, c.count - 1)];
      vx[k] = q.x - c.cx; vy[k] = q.y - c.cy; vz[k] = q.z - c.cz;
    }
    v4d ax = {0, 0, 0, 0}, ay = ax, az = ax;
    l2p4(L, *(v4d*)vx, *(v4d*)vy, *(v4d*)vz, ax, ay, az);
    int nk = std::min(4, c.count - j0);
    for (int k = 0; k < nk; k++) {
      int j = c.begin + j0 + k;
      accx[j] += ax[k]; accy[j] += ay[k]; accz[j] += az[k];
    }
  }
}

// ---------------------------------------------------------------- interaction lists
struct Pair { int t, s; };
static std::vector<Pair> m2l_pairs, p2p_pairs, m2p_pairs, mut_pairs;  // mut_pairs: unordered near pairs (Newton's third law)
// relative costs (units: one padded P2P pair interaction) and absolute per-unit CPU costs, per expansion order
#if P_ORDER == 12
#define DEF_CM2L 1640.0
#define DEF_CM2P 810.0
#define DEF_UM2L 2.0e-6
#define DEF_UM2P 0.99e-6
#define DEF_THETA 0.6
#define DEF_REFMULT 0.13
#elif P_ORDER == 11
#define DEF_CM2L 1220.0
#define DEF_CM2P 520.0
#define DEF_UM2L 1.46e-6
#define DEF_UM2P 0.64e-6
#define DEF_THETA 0.54
#define DEF_REFMULT 0.095
#else
#define DEF_CM2L 800.0
#define DEF_CM2P 400.0
#define DEF_UM2L 0.96e-6
#define DEF_UM2P 0.49e-6
#define DEF_THETA 0.55
#define DEF_REFMULT 0.11
#endif
static double C_PP = 1.0, C_M2L = DEF_CM2L, C_M2P = DEF_CM2P, P2P_BIAS = 1.0;

static inline double approx_cost(int t, bool m2p) { return m2p ? cells[t].count * C_M2P : C_M2L; }

static inline void push_dir(int s, int t, bool m2p, bool direct) {
  if (direct) p2p_pairs.push_back({t, s});
  else if (m2p) m2p_pairs.push_back({t, s});
  else m2l_pairs.push_back({t, s});
}

// both directions of an accepted pair: mutual direct sum if cheaper than the two expansions, else per direction
static inline void emit_pair(int a, int b, bool mAB, bool mBA) {
  const Cell& A = cells[a];
  const Cell& B = cells[b];
  double cab = approx_cost(b, mAB), cba = approx_cost(a, mBA);
  double mutual = (double)A.count * (double)B.count * C_PP;
  if (mutual <= (cab + cba) * P2P_BIAS) { mut_pairs.push_back({a, b}); return; }
  bool dab = (double)A.count * ((B.count + 3) & ~3) * C_PP <= cab * P2P_BIAS;
  bool dba = (double)B.count * ((A.count + 3) & ~3) * C_PP <= cba * P2P_BIAS;
  push_dir(a, b, mAB, dab);
  push_dir(b, a, mBA, dba);
}

static void traverse(double theta, int kp, double ppt_near) {
  std::vector<Pair> st;
  st.reserve(1 << 16);
  st.push_back({0, 0});
  while (!st.empty()) {
    Pair pr = st.back(); st.pop_back();
    int a = pr.t, b = pr.s;
    const Cell& A = cells[a];
    if (a == b) {
      if (A.nchild == 0) { p2p_pairs.push_back({a, a}); continue; }
      for (int i = 0; i < A.nchild; i++) {
        st.push_back({A.child + i, A.child + i});
        for (int j = i + 1; j < A.nchild; j++) st.push_back({A.child + i, A.child + j});
      }
      continue;
    }
    const Cell& B = cells[b];
    double dx = A.cx - B.cx, dy = A.cy - B.cy, dz = A.cz - B.cz;
    double d = sqrt(dx * dx + dy * dy + dz * dz);
    bool mAB = B.count <= kp, mBA = A.count <= kp;   // A->B uses M2P if B small
    bool okAB = mAB ? (A.r < theta * (d - B.r)) : (A.r + B.r < theta * d);
    bool okBA = mBA ? (B.r < theta * (d - A.r)) : (A.r + B.r < theta * d);
    if (okAB && okBA) { emit_pair(a, b, mAB, mBA); continue; }
    double nn = (double)A.count * (double)B.count;
    if (nn <= ppt_near || (A.nchild == 0 && B.nchild == 0)) {
      mut_pairs.push_back({a, b});
      continue;
    }
    if (A.nchild && (B.nchild == 0 || A.r >= B.r)) {
      for (int i = 0; i < A.nchild; i++) st.push_back({A.child + i, b});
    } else {
      for (int i = 0; i < B.nchild; i++) st.push_back({a, B.child + i});
    }
  }
}

static void csr(const std::vector<Pair>& pairs, int ncell, std::vector<int>& start, std::vector<int>& src) {
  start.assign(ncell + 1, 0);
  for (const Pair& p : pairs) start[p.t + 1]++;
  for (int i = 0; i < ncell; i++) start[i + 1] += start[i];
  src.resize(pairs.size());
  std::vector<int> fill(start.begin(), start.end() - 1);
  for (const Pair& p : pairs) src[fill[p.t]++] = p.s;
}

// ---------------------------------------------------------------- M2L
static inline void gather4(const double* m0, const double* m1, const double* m2, const double* m3, v4d* out) {
  for (int c = 0; c < NCOEF_PAD; c += 4) {
    __m256d a = _mm256_load_pd(m0 + c), b = _mm256_load_pd(m1 + c);
    __m256d cc = _mm256_load_pd(m2 + c), d = _mm256_load_pd(m3 + c);
    __m256d t0 = _mm256_unpacklo_pd(a, b), t1 = _mm256_unpackhi_pd(a, b);
    __m256d t2 = _mm256_unpacklo_pd(cc, d), t3 = _mm256_unpackhi_pd(cc, d);
    out[c + 0] = (v4d)_mm256_permute2f128_pd(t0, t2, 0x20);
    out[c + 1] = (v4d)_mm256_permute2f128_pd(t1, t3, 0x20);
    out[c + 2] = (v4d)_mm256_permute2f128_pd(t0, t2, 0x31);
    out[c + 3] = (v4d)_mm256_permute2f128_pd(t1, t3, 0x31);
  }
}

static void do_m2l(const std::vector<int>& start, const std::vector<int>& src, int ncell) {
  alignas(32) v4d Lacc[NCOEF_PAD];
  alignas(32) v4d Mv[NCOEF_PAD];
  alignas(32) v4d D[NCOEF_PAD];
  const int dummy = ncell;  // zero multipole
  for (int t = 0; t < ncell; t++) {
    int b = start[t], e = start[t + 1];
    if (b == e) continue;
    for (int i = 0; i < NCOEF_PAD; i++) Lacc[i] = (v4d){0, 0, 0, 0};
    const Cell& T = cells[t];
    for (int k = b; k < e; k += 4) {
      int s[4];
      alignas(32) double X[4], Y[4], Z[4];
      for (int q = 0; q < 4; q++) {
        if (k + q < e) {
          s[q] = src[k + q];
          const Cell& S = cells[s[q]];
          X[q] = T.cx - S.cx; Y[q] = T.cy - S.cy; Z[q] = T.cz - S.cz;
        } else {
          s[q] = dummy; X[q] = 1.0; Y[q] = 0.0; Z[q] = 0.0;
        }
      }
      gather4(Mexp + (size_t)s[0] * NCOEF_PAD, Mexp + (size_t)s[1] * NCOEF_PAD,
              Mexp + (size_t)s[2] * NCOEF_PAD, Mexp + (size_t)s[3] * NCOEF_PAD, Mv);
      derivs(*(v4d*)X, *(v4d*)Y, *(v4d*)Z, EPS2, D);
      m2l_contract(Mv, D, Lacc);
    }
    double* L = Lexp + (size_t)t * NCOEF_PAD;
    for (int i = 0; i < NCOEF; i++) L[i] += Lacc[i][0] + Lacc[i][1] + Lacc[i][2] + Lacc[i][3];
  }
}

// ---------------------------------------------------------------- P2P
static inline v4d rsqrt_nr(v4d x) {
  v4d y = (v4d)_mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps((__m256d)x)));
  v4d h = x * 0.5;
  y = y * (1.5 - h * y * y);
  y = y * (1.5 - h * y * y);
  return y;
}

static void do_p2p(const std::vector<int>& start, const std::vector<int>& src, int ncell) {
  std::vector<double> buf;
  for (int t = 0; t < ncell; t++) {
    int b = start[t], e = start[t + 1];
    if (b == e) continue;
    const Cell& T = cells[t];
    int ns = 0;
    for (int k = b; k < e; k++) ns += cells[src[k]].count;
    const int nsp = (ns + 1) & ~1;
    if ((int)buf.size() < 4 * nsp + 16) buf.resize(4 * nsp + 64);
    double* sx = buf.data(); double* sy = sx + nsp; double* sz = sy + nsp; double* sm = sz + nsp;
    int n = 0;
    for (int k = b; k < e; k++) {
      const Cell& S = cells[src[k]];
      for (int j = S.begin; j < S.begin + S.count; j++) { sx[n] = parts[j].x; sy[n] = parts[j].y; sz[n] = parts[j].z; sm[n] = parts[j].m; n++; }
    }
    if (ns & 1) { sx[ns] = sx[0]; sy[ns] = sy[0]; sz[ns] = sz[0]; sm[ns] = 0.0; }
    for (int i0 = 0; i0 < T.count; i0 += 4) {
      alignas(32) double tx[4], ty[4], tz[4];
      for (int k = 0; k < 4; k++) {
        int ii = T.begin + std::min(i0 + k, T.count - 1);
        tx[k] = parts[ii].x; ty[k] = parts[ii].y; tz[k] = parts[ii].z;
      }
      v4d xi = *(v4d*)tx, yi = *(v4d*)ty, zi = *(v4d*)tz;
      v4d axa = {0, 0, 0, 0}, aya = axa, aza = axa, axb = axa, ayb = axa, azb = axa;
      for (int j = 0; j < ns; j += 2) {
        v4d dxa = (v4d)_mm256_broadcast_sd(sx + j) - xi;
        v4d dya = (v4d)_mm256_broadcast_sd(sy + j) - yi;
        v4d dza = (v4d)_mm256_broadcast_sd(sz + j) - zi;
        v4d dxb = (v4d)_mm256_broadcast_sd(sx + j + 1) - xi;
        v4d dyb = (v4d)_mm256_broadcast_sd(sy + j + 1) - yi;
        v4d dzb = (v4d)_mm256_broadcast_sd(sz + j + 1) - zi;
        v4d r2a = dxa * dxa + dya * dya + dza * dza + EPS2;
        v4d r2b = dxb * dxb + dyb * dyb + dzb * dzb + EPS2;
        v4d wa = (v4d)_mm256_broadcast_sd(sm + j) / (r2a * (v4d)_mm256_sqrt_pd((__m256d)r2a));
        v4d wb = (v4d)_mm256_broadcast_sd(sm + j + 1) / (r2b * (v4d)_mm256_sqrt_pd((__m256d)r2b));
        axa += wa * dxa; aya += wa * dya; aza += wa * dza;
        axb += wb * dxb; ayb += wb * dyb; azb += wb * dzb;
      }
      v4d ox = axa + axb, oy = aya + ayb, oz = aza + azb;
      int nk = std::min(4, T.count - i0);
      for (int k = 0; k < nk; k++) {
        int ii = T.begin + i0 + k;
        accx[ii] += ox[k]; accy[ii] += oy[k]; accz[ii] += oz[k];
      }
    }
  }
}

// mutual direct sum: owner particles broadcast, partner particles in vector lanes with reaction accumulators
static void do_p2p_mut(const std::vector<int>& start, const std::vector<int>& src, int ncell) {
  std::vector<double> buf;
  std::vector<int> bidx;
  for (int t = 0; t < ncell; t++) {
    int b = start[t], e = start[t + 1];
    if (b == e) continue;
    const Cell& T = cells[t];
    int n = 0;
    for (int k = b; k < e; k++) n += cells[src[k]].count;
    const int np = (n + 7) & ~7;
    if ((int)buf.size() < 7 * np + 64) buf.resize(7 * np + 64);
    if ((int)bidx.size() < np) bidx.resize(np);
    double* bx = (double*)(((uintptr_t)buf.data() + 31) & ~(uintptr_t)31);
    double *by = bx + np, *bz = by + np, *bm = bz + np, *fx = bm + np, *fy = fx + np, *fz = fy + np;
    int q = 0;
    for (int k = b; k < e; k++) {
      const Cell& S = cells[src[k]];
      for (int j = S.begin; j < S.begin + S.count; j++) {
        bx[q] = parts[j].x; by[q] = parts[j].y; bz[q] = parts[j].z; bm[q] = parts[j].m; bidx[q] = j; q++;
      }
    }
    for (; q < np; q++) { bx[q] = T.cx; by[q] = T.cy; bz[q] = T.cz + 1.0; bm[q] = 0.0; }
    for (int k = 0; k < np; k++) { fx[k] = 0; fy[k] = 0; fz[k] = 0; }
    for (int i = T.begin; i < T.begin + T.count; i++) {
      v4d xi = (v4d)_mm256_broadcast_sd(&parts[i].x), yi = (v4d)_mm256_broadcast_sd(&parts[i].y);
      v4d zi = (v4d)_mm256_broadcast_sd(&parts[i].z), mi = (v4d)_mm256_broadcast_sd(&parts[i].m);
      v4d axa = {0, 0, 0, 0}, aya = axa, aza = axa, axb = axa, ayb = axa, azb = axa;
      for (int j = 0; j < np; j += 8) {
        v4d dxa = *(v4d*)(bx + j) - xi, dya = *(v4d*)(by + j) - yi, dza = *(v4d*)(bz + j) - zi;
        v4d dxb = *(v4d*)(bx + j + 4) - xi, dyb = *(v4d*)(by + j + 4) - yi, dzb = *(v4d*)(bz + j + 4) - zi;
        v4d r2a = dxa * dxa + dya * dya + dza * dza + EPS2;
        v4d r2b = dxb * dxb + dyb * dyb + dzb * dzb + EPS2;
        v4d wa = 1.0 / (r2a * (v4d)_mm256_sqrt_pd((__m256d)r2a));
        v4d wb = 1.0 / (r2b * (v4d)_mm256_sqrt_pd((__m256d)r2b));
        v4d ja = *(v4d*)(bm + j) * wa, jb = *(v4d*)(bm + j + 4) * wb;
        axa += ja * dxa; aya += ja * dya; aza += ja * dza;
        axb += jb * dxb; ayb += jb * dyb; azb += jb * dzb;
        v4d ia = mi * wa, ib = mi * wb;
        *(v4d*)(fx + j) -= ia * dxa; *(v4d*)(fy + j) -= ia * dya; *(v4d*)(fz + j) -= ia * dza;
        *(v4d*)(fx + j + 4) -= ib * dxb; *(v4d*)(fy + j + 4) -= ib * dyb; *(v4d*)(fz + j + 4) -= ib * dzb;
      }
      v4d ox = axa + axb, oy = aya + ayb, oz = aza + azb;
      accx[i] += ox[0] + ox[1] + ox[2] + ox[3];
      accy[i] += oy[0] + oy[1] + oy[2] + oy[3];
      accz[i] += oz[0] + oz[1] + oz[2] + oz[3];
    }
    for (int k = 0; k < n; k++) { int j = bidx[k]; accx[j] += fx[k]; accy[j] += fy[k]; accz[j] += fz[k]; }
  }
}

// M2P: source multipoles evaluated directly at the particles of (small) target cells
static void do_m2p(const std::vector<int>& start, const std::vector<int>& src, int ncell) {
  alignas(32) v4d Mv[NCOEF_PAD];
  alignas(32) v4d D1[NCOEF1_PAD];
  std::vector<v4d> acc;
  const int dummy = ncell;
  for (int t = 0; t < ncell; t++) {
    int b = start[t], e = start[t + 1];
    if (b == e) continue;
    const Cell& T = cells[t];
    acc.assign(3 * T.count, (v4d){0, 0, 0, 0});
    for (int k = b; k < e; k += 4) {
      int s[4];
      for (int q = 0; q < 4; q++) s[q] = (k + q < e) ? src[k + q] : dummy;
      gather4(Mexp + (size_t)s[0] * NCOEF_PAD, Mexp + (size_t)s[1] * NCOEF_PAD,
              Mexp + (size_t)s[2] * NCOEF_PAD, Mexp + (size_t)s[3] * NCOEF_PAD, Mv);
      alignas(32) double SX[4], SY[4], SZ[4];
      for (int q = 0; q < 4; q++) {
        if (s[q] == dummy) { SX[q] = 1e3; SY[q] = 0; SZ[q] = 0; }
        else { SX[q] = cells[s[q]].cx; SY[q] = cells[s[q]].cy; SZ[q] = cells[s[q]].cz; }
      }
      v4d sxv = *(v4d*)SX, syv = *(v4d*)SY, szv = *(v4d*)SZ;
      for (int i = 0; i < T.count; i++) {
        const P4& p = parts[T.begin + i];
        v4d X = p.x - sxv, Y = p.y - syv, Z = p.z - szv;
        derivs_p1(X, Y, Z, EPS2, D1);
        m2p_contract(Mv, D1, acc[3 * i], acc[3 * i + 1], acc[3 * i + 2]);
      }
    }
    for (int i = 0; i < T.count; i++) {
      int ii = T.begin + i;
      v4d ax = acc[3 * i], ay = acc[3 * i + 1], az = acc[3 * i + 2];
      accx[ii] += ax[0] + ax[1] + ax[2] + ax[3];
      accy[ii] += ay[0] + ay[1] + ay[2] + ay[3];
      accz[ii] += az[0] + az[1] + az[2] + az[3];
    }
  }
}

// ---------------------------------------------------------------- main
int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: solve INPUT OUTPUT\n"); return 2; }
  double t0 = cpu_now();
  const bool verbose = GETENV("FMM_VERBOSE") != nullptr;
  const double theta = env_d("FMM_THETA", DEF_THETA);
  const int leafmax = (int)env_d("FMM_LEAF", 16);
  const int kp = (int)env_d("FMM_KP", 16);
  const double ppt_near = env_d("FMM_PPT_NEAR", 64);
  C_M2L = env_d("FMM_CM2L", C_M2L); C_M2P = env_d("FMM_CM2P", C_M2P); P2P_BIAS = env_d("FMM_BIAS", P2P_BIAS);

  FILE* f = fopen(argv[1], "rb");
  if (!f) { perror("input"); return 1; }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  N = (int)(sz / 32);
  parts = (P4*)aligned_alloc(64, sizeof(P4) * (size_t)(N + 8));
  if (fread(parts, 32, N, f) != (size_t)N) { fprintf(stderr, "read error\n"); return 1; }
  fclose(f);
  // file rows are m,x,y,z -> reorder to x,y,z,m
  for (int i = 0; i < N; i++) { P4 p = parts[i]; parts[i] = {p.y, p.z, p.m, p.x}; }
  orig = (int*)malloc(sizeof(int) * (size_t)N);
  for (int i = 0; i < N; i++) orig[i] = i;

  for (int t = 0; t <= P_ORDER + 1; t++) for (int u = 0; u <= P_ORDER + 1; u++) for (int v = 0; v <= P_ORDER + 1; v++) POS3[t][u][v] = -1;
  for (int i = 0; i < NCOEF; i++) POS3[IDX_T[i]][IDX_U[i]][IDX_V[i]] = i;
  INVFACT[0] = 1; for (int i = 1; i <= P_ORDER + 1; i++) INVFACT[i] = INVFACT[i - 1] / i;
  L2P_N = 0;
  for (int i = 0; i < NCOEF; i++) {
    int t = IDX_T[i], u = IDX_U[i], v = IDX_V[i];
    if (t + u + v <= P_ORDER - 1) {
      L2P_IX[L2P_N] = POS3[t + 1][u][v]; L2P_IY[L2P_N] = POS3[t][u + 1][v]; L2P_IZ[L2P_N] = POS3[t][u][v + 1];
      L2P_N++;
    }
  }

  build_tree(leafmax);
  int ncell = (int)cells.size();
  // centres of mass, radii
  for (int ci = ncell - 1; ci >= 0; ci--) {
    Cell& c = cells[ci];
    double m = 0, sx = 0, sy = 0, sz2 = 0;
    for (int j = c.begin; j < c.begin + c.count; j++) {
      m += parts[j].m; sx += parts[j].m * parts[j].x; sy += parts[j].m * parts[j].y; sz2 += parts[j].m * parts[j].z;
    }
    c.mass = m;
    if (m > 0) { c.cx = sx / m; c.cy = sy / m; c.cz = sz2 / m; }
    else { c.cx = c.bx; c.cy = c.by; c.cz = c.bz; }
    double r2 = 0;
    for (int j = c.begin; j < c.begin + c.count; j++) {
      double dx = parts[j].x - c.cx, dy = parts[j].y - c.cy, dz = parts[j].z - c.cz;
      r2 = std::max(r2, dx * dx + dy * dy + dz * dz);
    }
    c.r = sqrt(r2);
  }
  double t1 = cpu_now();

  Mexp = (double*)aligned_alloc(64, sizeof(double) * (size_t)(ncell + 1) * NCOEF_PAD);
  Lexp = (double*)aligned_alloc(64, sizeof(double) * (size_t)ncell * NCOEF_PAD);
  memset(Mexp, 0, sizeof(double) * (size_t)(ncell + 1) * NCOEF_PAD);
  memset(Lexp, 0, sizeof(double) * (size_t)ncell * NCOEF_PAD);
  double t1b = cpu_now();
  for (int ci = ncell - 1; ci >= 0; ci--) {
    Cell& c = cells[ci];
    double* M = Mexp + (size_t)ci * NCOEF_PAD;
    if (c.nchild == 0) p2m(c, M);
    else {
      for (int k = 0; k < c.nchild; k++) {
        const Cell& d = cells[c.child + k];
        m2m_add(Mexp + (size_t)(c.child + k) * NCOEF_PAD, d.cx - c.cx, d.cy - c.cy, d.cz - c.cz, M);
      }
    }
  }
  double t2 = cpu_now();

  // per-unit costs (seconds, measured on the development machine)
  // Speed probe: the multipole phase is a fixed workload; if this machine is slower than the development
  // machine, scale all cost estimates up (never down) so the opening angle grows and the run stays in budget.
  const double speed = std::min(4.0, std::max(1.0, (t2 - t1b) / env_d("FMM_REFMULT", DEF_REFMULT)));
  const double T_DOWN = 0.08 * speed, TARGET = env_d("FMM_TARGET", 3.2), TARGET_HI = env_d("FMM_TARGET_HI", 3.5);
  const double U_M2L = env_d("FMM_UM2L", DEF_UM2L) * speed, U_M2P = env_d("FMM_UM2P", DEF_UM2P) * speed, U_PP = env_d("FMM_UPP", 1.22e-9) * speed;
  const double U_MUT = env_d("FMM_UMUT", 1.85e-9) * speed;
  const double THETA_MIN = env_d("FMM_THMIN", 0.35), THETA_MAX = env_d("FMM_THMAX", 0.75);
  auto estimate = [&](double& comp) {
    double npp = 0, nq = 0;
    for (const Pair& p : p2p_pairs) npp += (double)cells[p.s].count * ((cells[p.t].count + 3) & ~3);
    for (const Pair& p : m2p_pairs) nq += cells[p.t].count;
    double nmu = 0;
    for (const Pair& p : mut_pairs) nmu += (double)cells[p.t].count * cells[p.s].count;
    comp = U_M2L * m2l_pairs.size() + U_M2P * nq + U_PP * npp + U_MUT * nmu;
    return comp;
  };
  double theta_used = theta;
  traverse(theta, kp, ppt_near);
  {
    // Adapt the opening angle so that the estimated total CPU time lands near TARGET:
    // smaller angle (more accurate) when there is headroom, larger angle if the snapshot is unexpectedly expensive.
    double comp;
    estimate(comp);
    double tnow = cpu_now() - t0;
    double ttrav = cpu_now() - t2;
    double avail = TARGET - tnow - T_DOWN - ttrav * 1.3;
    double th2 = theta;
    if (avail > comp * 1.04) th2 = std::max(THETA_MIN, theta * pow(comp / avail, 1.0 / 3.0));
    else if (tnow + comp + T_DOWN > TARGET_HI) th2 = std::min(THETA_MAX, theta * pow(comp / std::max(avail, 0.3 * comp), 1.0 / 3.0));
    if (fabs(th2 - theta) > 0.005) {
      m2l_pairs.clear(); p2p_pairs.clear(); m2p_pairs.clear(); mut_pairs.clear();
      traverse(th2, kp, ppt_near);
      theta_used = th2;
      double comp2;
      estimate(comp2);
      if (verbose) fprintf(stderr, "est1 %.3f -> th2 %.3f est2 %.3f\n", comp, th2, comp2);
      double tnow2 = cpu_now() - t0;
      if (th2 < theta && tnow2 + comp2 + T_DOWN > TARGET + 0.25) {  // model miss: fall back to the default opening angle
        m2l_pairs.clear(); p2p_pairs.clear(); m2p_pairs.clear(); mut_pairs.clear();
        traverse(theta, kp, ppt_near);
        theta_used = theta;
      }
    }
  }
  double t3 = cpu_now();
  double comp_final; estimate(comp_final);
  std::vector<int> mst, msrc, pst, psrc, qst, qsrc;
  csr(m2l_pairs, ncell, mst, msrc);
  csr(p2p_pairs, ncell, pst, psrc);
  csr(m2p_pairs, ncell, qst, qsrc);
  std::vector<int> ust, usrc;
  csr(mut_pairs, ncell, ust, usrc);
  double nmut = 0;
  for (const Pair& p : mut_pairs) nmut += (double)cells[p.t].count * cells[p.s].count;
  double nm2p = 0;
  for (const Pair& p : m2p_pairs) nm2p += cells[p.t].count;
  double npp = 0;
  for (const Pair& p : p2p_pairs) npp += (double)cells[p.t].count * cells[p.s].count;
  size_t nm2l = m2l_pairs.size(), np2pc = p2p_pairs.size();
  std::vector<Pair>().swap(m2l_pairs);
  std::vector<Pair>().swap(p2p_pairs);
  std::vector<Pair>().swap(m2p_pairs);
  std::vector<Pair>().swap(mut_pairs);

  accx = (double*)calloc(N, sizeof(double));
  accy = (double*)calloc(N, sizeof(double));
  accz = (double*)calloc(N, sizeof(double));
  double t4 = cpu_now();
  do_m2l(mst, msrc, ncell);
  double t5 = cpu_now();
  do_p2p(pst, psrc, ncell);
  double t5b = cpu_now();
  do_p2p_mut(ust, usrc, ncell);
  double t6a = cpu_now();
  do_m2p(qst, qsrc, ncell);
  double t6 = cpu_now();
  for (int ci = 0; ci < ncell; ci++) {
    const Cell& c = cells[ci];
    const double* L = Lexp + (size_t)ci * NCOEF_PAD;
    if (c.nchild) {
      for (int k = 0; k < c.nchild; k++) {
        const Cell& d = cells[c.child + k];
        l2l_add(L, d.cx - c.cx, d.cy - c.cy, d.cz - c.cz, Lexp + (size_t)(c.child + k) * NCOEF_PAD);
      }
    } else {
      l2p_cell(c, L);
    }
  }
  double t7 = cpu_now();

  double* out = (double*)malloc(sizeof(double) * 3 * (size_t)N);
  for (int i = 0; i < N; i++) {
    int o = orig[i];
    out[3 * o] = accx[i]; out[3 * o + 1] = accy[i]; out[3 * o + 2] = accz[i];
  }
  FILE* g = fopen(argv[2], "wb");
  if (!g) { perror("output"); return 1; }
  if (fwrite(out, 24, N, g) != (size_t)N) { fprintf(stderr, "write error\n"); return 1; }
  if (fclose(g)) { perror("close"); return 1; }
  double t8 = cpu_now();
  if (verbose)
    fprintf(stderr, "speed %.3f (mult %.3f) est %.3f act %.3f P=%d th=%.3f N=%d cells=%d tree %.3f mult %.3f trav %.3f m2l %.3f (%zu) p2p %.3f (%zu cellpairs, %.3g pp) mut %.3f (%.3g) m2p %.3f (%.3g) down %.3f out %.3f total %.3f\n",
            speed, t2 - t1b, comp_final, t6 - t4, P_ORDER, theta_used, N, ncell, t1 - t0, t2 - t1, t3 - t2, t5 - t4, nm2l, t5b - t5, np2pc, npp, t6a - t5b, nmut, t6 - t6a, nm2p, t7 - t6, t8 - t7, t8 - t0);
  return 0;
}
