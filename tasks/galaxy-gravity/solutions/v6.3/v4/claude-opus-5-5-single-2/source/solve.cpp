// Plummer-softened gravity via Cartesian FMM (dual tree traversal, mutual interactions)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <immintrin.h>
#include <time.h>

static double cputime() { timespec ts; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts); return ts.tv_sec + 1e-9 * ts.tv_nsec; }
static const double EPS2 = 1e-4;

struct Part { double x, y, z, m; int64_t id; };
struct Node { double cx, cy, cz, rmax, mass, amin; int begin, end, child, nchild; };

static int N;
static std::vector<Part> PT, TMP;
static std::vector<Node> nodes;
static double *X, *Y, *Z, *MS, *AX, *AY, *AZ;
static int NLEAF = 64;
static double THETA = 0.5;
static double THR = 1000;   // pair-count threshold below which direct summation is used
static double TOL = 1e-9, THMAX = 0.6, SPARSE = 0.1, GCX = 0, GCY = 0, GCZ = 0, RCORE = 1;
static int NLEAFMIN = 4;
static std::vector<double> AEST;
static double cnt_p2p = 0, cnt_m2l = 0;
static bool count_only = false;

static void* amalloc(size_t bytes) { void* p = nullptr; if (posix_memalign(&p, 64, bytes + 64)) abort(); return p; }

// ---------------- tree build ----------------
static void build_node(int ni, int depth) {
  int b = nodes[ni].begin, e = nodes[ni].end;
  double mn[3] = {1e300, 1e300, 1e300}, mx[3] = {-1e300, -1e300, -1e300};
  double sm = 0, sx = 0, sy = 0, sz = 0;
  for (int k = b; k < e; k++) {
    const Part& p = PT[k];
    mn[0] = std::min(mn[0], p.x); mx[0] = std::max(mx[0], p.x);
    mn[1] = std::min(mn[1], p.y); mx[1] = std::max(mx[1], p.y);
    mn[2] = std::min(mn[2], p.z); mx[2] = std::max(mx[2], p.z);
    sm += p.m; sx += p.m * p.x; sy += p.m * p.y; sz += p.m * p.z;
  }
  double cx, cy, cz;
  if (sm > 0) { cx = sx / sm; cy = sy / sm; cz = sz / sm; }
  else { cx = 0.5 * (mn[0] + mx[0]); cy = 0.5 * (mn[1] + mx[1]); cz = 0.5 * (mn[2] + mx[2]); }
  double r2max = 0;
  for (int k = b; k < e; k++) {
    const Part& p = PT[k];
    double dx = p.x - cx, dy = p.y - cy, dz = p.z - cz;
    r2max = std::max(r2max, dx * dx + dy * dy + dz * dz);
  }
  Node& nd = nodes[ni];
  nd.cx = cx; nd.cy = cy; nd.cz = cz; nd.rmax = std::sqrt(r2max); nd.mass = sm;
  double am = 1e300; for (int k = b; k < e; k++) am = std::min(am, AEST[PT[k].id]);
  nd.amin = am;
  nd.child = -1; nd.nchild = 0;
  double dc = std::sqrt((cx - GCX) * (cx - GCX) + (cy - GCY) * (cy - GCY) + (cz - GCZ) * (cz - GCZ));
  bool sparse = dc > RCORE && nd.rmax > SPARSE * dc;
  double ext[3] = {mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]};
  double emax = std::max(ext[0], std::max(ext[1], ext[2]));
  if ((e - b <= NLEAF && (!sparse || e - b <= NLEAFMIN)) || depth >= 200 || !(emax > 0)) return;
  double ctr[3] = {0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1]), 0.5 * (mn[2] + mx[2])};
  bool sp[3]; for (int d = 0; d < 3; d++) sp[d] = ext[d] >= 0.35 * emax;
  int cnt[8] = {0};
  std::vector<unsigned char> code(e - b);
  for (int k = b; k < e; k++) {
    const Part& p = PT[k];
    int c = ((sp[0] && p.x >= ctr[0]) << 2) | ((sp[1] && p.y >= ctr[1]) << 1) | (int)(sp[2] && p.z >= ctr[2]);
    code[k - b] = (unsigned char)c; cnt[c]++;
  }
  int off[9]; off[0] = b; for (int c = 0; c < 8; c++) off[c + 1] = off[c] + cnt[c];
  int pos[8]; for (int c = 0; c < 8; c++) pos[c] = off[c];
  for (int k = b; k < e; k++) TMP[pos[code[k - b]]++] = PT[k];
  std::copy(TMP.begin() + b, TMP.begin() + e, PT.begin() + b);
  int first = (int)nodes.size(), nc = 0;
  for (int c = 0; c < 8; c++) if (cnt[c] > 0) {
    Node ch; ch.begin = off[c]; ch.end = off[c + 1]; ch.child = -1; ch.nchild = 0;
    nodes.push_back(ch); nc++;
  }
  nodes[ni].child = first; nodes[ni].nchild = nc;
  for (int c = 0; c < nc; c++) build_node(first + c, depth + 1);
}

// ---------------- P2P ----------------
static inline __m256d rinv3(__m256d r2) {
  const __m256d half = _mm256_set1_pd(0.5), c15 = _mm256_set1_pd(1.5), one = _mm256_set1_pd(1.0), c38 = _mm256_set1_pd(0.375);
  __m256d y = _mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(r2)));
  __m256d hr2 = _mm256_mul_pd(half, r2);
  __m256d t = _mm256_mul_pd(y, y);
  y = _mm256_mul_pd(y, _mm256_fnmadd_pd(hr2, t, c15));
  t = _mm256_mul_pd(r2, y);
  __m256d e = _mm256_fnmadd_pd(t, y, one);
  __m256d c = _mm256_fmadd_pd(c38, e, half);
  y = _mm256_fmadd_pd(_mm256_mul_pd(y, e), c, y);
  return _mm256_mul_pd(_mm256_mul_pd(y, y), y);
}
static inline double hsum(__m256d v) {
  __m128d lo = _mm256_castpd256_pd128(v), hi = _mm256_extractf128_pd(v, 1);
  lo = _mm_add_pd(lo, hi);
  return _mm_cvtsd_f64(_mm_add_sd(lo, _mm_unpackhi_pd(lo, lo)));
}
static const int64_t maskv[8] = {-1, -1, -1, -1, 0, 0, 0, 0};

// particle i interacts mutually with all j in [b0,b1)
static inline void p2p_row(int i, int b0, int b1) {
  const __m256d eps2 = _mm256_set1_pd(EPS2);
  __m256d xi = _mm256_set1_pd(X[i]), yi = _mm256_set1_pd(Y[i]), zi = _mm256_set1_pd(Z[i]), mi = _mm256_set1_pd(MS[i]);
  __m256d ax = _mm256_setzero_pd(), ay = _mm256_setzero_pd(), az = _mm256_setzero_pd();
  int j = b0;
  for (; j + 4 <= b1; j += 4) {
    __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(X + j), xi);
    __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(Y + j), yi);
    __m256d dz = _mm256_sub_pd(_mm256_loadu_pd(Z + j), zi);
    __m256d r2 = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_fmadd_pd(dz, dz, eps2)));
    __m256d q = _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(r2, _mm256_sqrt_pd(r2)));
    __m256d mjq = _mm256_mul_pd(_mm256_loadu_pd(MS + j), q);
    ax = _mm256_fmadd_pd(mjq, dx, ax); ay = _mm256_fmadd_pd(mjq, dy, ay); az = _mm256_fmadd_pd(mjq, dz, az);
    __m256d miq = _mm256_mul_pd(mi, q);
    _mm256_storeu_pd(AX + j, _mm256_fnmadd_pd(miq, dx, _mm256_loadu_pd(AX + j)));
    _mm256_storeu_pd(AY + j, _mm256_fnmadd_pd(miq, dy, _mm256_loadu_pd(AY + j)));
    _mm256_storeu_pd(AZ + j, _mm256_fnmadd_pd(miq, dz, _mm256_loadu_pd(AZ + j)));
  }
  int rem = b1 - j;
  if (rem > 0) {
    __m256i mk = _mm256_loadu_si256((const __m256i*)(maskv + 4 - rem));
    __m256d dx = _mm256_sub_pd(_mm256_maskload_pd(X + j, mk), xi);
    __m256d dy = _mm256_sub_pd(_mm256_maskload_pd(Y + j, mk), yi);
    __m256d dz = _mm256_sub_pd(_mm256_maskload_pd(Z + j, mk), zi);
    __m256d r2 = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_fmadd_pd(dz, dz, eps2)));
    __m256d q = _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(r2, _mm256_sqrt_pd(r2)));
    __m256d mjq = _mm256_mul_pd(_mm256_maskload_pd(MS + j, mk), q);
    ax = _mm256_fmadd_pd(mjq, dx, ax); ay = _mm256_fmadd_pd(mjq, dy, ay); az = _mm256_fmadd_pd(mjq, dz, az);
    __m256d miq = _mm256_mul_pd(mi, q);
    _mm256_maskstore_pd(AX + j, mk, _mm256_fnmadd_pd(miq, dx, _mm256_maskload_pd(AX + j, mk)));
    _mm256_maskstore_pd(AY + j, mk, _mm256_fnmadd_pd(miq, dy, _mm256_maskload_pd(AY + j, mk)));
    _mm256_maskstore_pd(AZ + j, mk, _mm256_fnmadd_pd(miq, dz, _mm256_maskload_pd(AZ + j, mk)));
  }
  AX[i] += hsum(ax); AY[i] += hsum(ay); AZ[i] += hsum(az);
}
static void p2p_mutual(int a0, int a1, int b0, int b1) {
  if (count_only) { cnt_p2p += (double)(a1 - a0) * (b1 - b0); return; }
  if (a1 - a0 > b1 - b0) { std::swap(a0, b0); std::swap(a1, b1); }
  for (int i = a0; i < a1; i++) p2p_row(i, b0, b1);
}
static void p2p_self(int a0, int a1) {
  if (count_only) { double n = a1 - a0; cnt_p2p += 0.5 * n * (n - 1); return; }
  for (int i = a0; i < a1 - 1; i++) p2p_row(i, i + 1, a1);
}

// ---------------- FMM ----------------
static inline int tet(int n) { return n * (n + 1) * (n + 2) / 6; }
static inline int tri(int n) { return n * (n + 1) / 2; }

template <int P>
struct FMM {
  static constexpr int NC = (P + 1) * (P + 2) * (P + 3) / 6;
  static constexpr int NCL = P * (P + 1) * (P + 2) / 6;
  static constexpr int CS = P + 1;
  static constexpr int NCP = (NC + 3) & ~3;
  int ea[NC], eb[NC], ec[NC], dg[NC], coff[NC];
  int sid[P + 1][P + 1][P + 1];
  double sgn[NC];
  int mono_src[NC], mono_comp[NC]; double mono_inv[NC];
  struct Rec { int dst, s1, s2, comp; double coef; };
  std::vector<Rec> rec;
  int Toff[P + 2], Tsize;
  std::vector<int> tr_a, tr_b, tr_g;
  int lxi[NCL > 0 ? NCL : 1], lyi[NCL > 0 ? NCL : 1], lzi[NCL > 0 ? NCL : 1];
  double *Mul, *Loc;
  std::vector<char> hasL;
  int nn;
  // batch buffers
  __m256d *Tb, *Dc, *MA, *MB, *LA, *LB;
  __m256d sgnv[NCP];
  int bufA[4], bufB[4], nbuf;

  FMM() {
    int k = 0;
    for (int n = 0; n <= P; n++)
      for (int a = n; a >= 0; a--)
        for (int c = 0; c <= n - a; c++) {
          int b = n - a - c;
          ea[k] = a; eb[k] = b; ec[k] = c; dg[k] = n; sid[a][b][c] = k;
          coff[k] = (a * CS + b) * CS + c; sgn[k] = (n & 1) ? -1.0 : 1.0; k++;
        }
    for (k = 1; k < NC; k++) {
      int a = ea[k], b = eb[k], c = ec[k];
      if (a > 0) { mono_src[k] = sid[a - 1][b][c]; mono_comp[k] = 0; mono_inv[k] = 1.0 / a; }
      else if (b > 0) { mono_src[k] = sid[a][b - 1][c]; mono_comp[k] = 1; mono_inv[k] = 1.0 / b; }
      else { mono_src[k] = sid[a][b][c - 1]; mono_comp[k] = 2; mono_inv[k] = 1.0 / c; }
    }
    Tsize = 0;
    for (int m = 0; m <= P; m++) { Toff[m] = Tsize; Tsize += tet(P - m + 1); }
    Toff[P + 1] = Tsize;
    for (int n = 1; n <= P; n++)
      for (int m = 0; m <= P - n; m++)
        for (k = tet(n); k < tet(n + 1); k++) {
          int g[3] = {ea[k], eb[k], ec[k]};
          int i = g[0] > 0 ? 0 : (g[1] > 0 ? 1 : 2);
          int g1[3] = {g[0], g[1], g[2]}; g1[i]--;
          Rec r; r.dst = Toff[m] + k; r.s1 = Toff[m + 1] + sid[g1[0]][g1[1]][g1[2]]; r.comp = i;
          if (g[i] >= 2) { int g2[3] = {g1[0], g1[1], g1[2]}; g2[i]--; r.s2 = Toff[m + 1] + sid[g2[0]][g2[1]][g2[2]]; r.coef = g[i] - 1; }
          else { r.s2 = r.s1; r.coef = 0; }
          rec.push_back(r);
        }
    for (int a = 0; a < NC; a++)
      for (int b = 0; b < NC; b++)
        if (ea[b] <= ea[a] && eb[b] <= eb[a] && ec[b] <= ec[a]) {
          tr_a.push_back(a); tr_b.push_back(b); tr_g.push_back(sid[ea[a] - ea[b]][eb[a] - eb[b]][ec[a] - ec[b]]);
        }
    for (k = 0; k < NCL; k++) {
      lxi[k] = sid[ea[k] + 1][eb[k]][ec[k]];
      lyi[k] = sid[ea[k]][eb[k] + 1][ec[k]];
      lzi[k] = sid[ea[k]][eb[k]][ec[k] + 1];
    }
    Tb = (__m256d*)amalloc(sizeof(__m256d) * Tsize);
    Dc = (__m256d*)amalloc(sizeof(__m256d) * CS * CS * CS);
    MA = (__m256d*)amalloc(sizeof(__m256d) * NCP); MB = (__m256d*)amalloc(sizeof(__m256d) * NCP);
    LA = (__m256d*)amalloc(sizeof(__m256d) * NCP); LB = (__m256d*)amalloc(sizeof(__m256d) * NCP);
    for (int k = 0; k < NCP; k++) sgnv[k] = _mm256_set1_pd(k < NC ? sgn[k] : 0.0);
    memset(Dc, 0, sizeof(__m256d) * CS * CS * CS);
    nbuf = 0; Mul = Loc = nullptr; nn = 0;
  }

  void alloc(int nnodes) {
    nn = nnodes;
    Mul = (double*)amalloc(sizeof(double) * (size_t)NCP * nn);
    Loc = (double*)amalloc(sizeof(double) * (size_t)NCP * nn);
    memset(Mul, 0, sizeof(double) * (size_t)NCP * nn);
    memset(Loc, 0, sizeof(double) * (size_t)NCP * nn);
    hasL.assign(nn, 0);
  }

  inline void mono_scalar(double dx, double dy, double dz, double* w, int cnt) const {
    w[0] = 1.0; double d[3] = {dx, dy, dz};
    for (int k = 1; k < cnt; k++) w[k] = w[mono_src[k]] * d[mono_comp[k]] * mono_inv[k];
  }
  inline void mono_vec(__m256d dx, __m256d dy, __m256d dz, __m256d* w, int cnt) const {
    w[0] = _mm256_set1_pd(1.0); __m256d d[3] = {dx, dy, dz};
    for (int k = 1; k < cnt; k++) w[k] = _mm256_mul_pd(_mm256_mul_pd(w[mono_src[k]], d[mono_comp[k]]), _mm256_set1_pd(mono_inv[k]));
  }

  void p2m(int ni) {
    const Node& nd = nodes[ni];
    double* M = Mul + (size_t)NCP * ni;
    alignas(32) __m256d w[NC], acc[NC];
    for (int k = 0; k < NC; k++) acc[k] = _mm256_setzero_pd();
    __m256d cx = _mm256_set1_pd(nd.cx), cy = _mm256_set1_pd(nd.cy), cz = _mm256_set1_pd(nd.cz);
    for (int j = nd.begin; j < nd.end; j += 4) {
      int rem = nd.end - j; if (rem > 4) rem = 4;
      __m256i mk = _mm256_loadu_si256((const __m256i*)(maskv + 4 - rem));
      __m256d dx = _mm256_sub_pd(_mm256_maskload_pd(X + j, mk), cx);
      __m256d dy = _mm256_sub_pd(_mm256_maskload_pd(Y + j, mk), cy);
      __m256d dz = _mm256_sub_pd(_mm256_maskload_pd(Z + j, mk), cz);
      __m256d m = _mm256_maskload_pd(MS + j, mk);
      mono_vec(dx, dy, dz, w, NC);
      for (int k = 0; k < NC; k++) acc[k] = _mm256_fmadd_pd(m, w[k], acc[k]);
    }
    for (int k = 0; k < NC; k++) M[k] = hsum(acc[k]);
  }

  void m2m(int child, int parent) {
    const Node &c = nodes[child], &p = nodes[parent];
    double w[NC];
    mono_scalar(c.cx - p.cx, c.cy - p.cy, c.cz - p.cz, w, NC);
    const double* Mc = Mul + (size_t)NCP * child; double* Mp = Mul + (size_t)NCP * parent;
    size_t nt = tr_a.size();
    for (size_t t = 0; t < nt; t++) Mp[tr_a[t]] += Mc[tr_b[t]] * w[tr_g[t]];
  }
  void l2l(int parent, int child) {
    const Node &c = nodes[child], &p = nodes[parent];
    double w[NC];
    mono_scalar(c.cx - p.cx, c.cy - p.cy, c.cz - p.cz, w, NC);
    const double* Lp = Loc + (size_t)NCP * parent; double* Lc = Loc + (size_t)NCP * child;
    size_t nt = tr_a.size();
    for (size_t t = 0; t < nt; t++) Lc[tr_b[t]] += Lp[tr_a[t]] * w[tr_g[t]];
    hasL[child] = 1;
  }
  void l2p(int ni) {
    const Node& nd = nodes[ni];
    const double* L = Loc + (size_t)NCP * ni;
    alignas(32) __m256d w[NC];
    __m256d cx = _mm256_set1_pd(nd.cx), cy = _mm256_set1_pd(nd.cy), cz = _mm256_set1_pd(nd.cz);
    for (int j = nd.begin; j < nd.end; j += 4) {
      int rem = nd.end - j; if (rem > 4) rem = 4;
      __m256i mk = _mm256_loadu_si256((const __m256i*)(maskv + 4 - rem));
      __m256d dx = _mm256_sub_pd(_mm256_maskload_pd(X + j, mk), cx);
      __m256d dy = _mm256_sub_pd(_mm256_maskload_pd(Y + j, mk), cy);
      __m256d dz = _mm256_sub_pd(_mm256_maskload_pd(Z + j, mk), cz);
      mono_vec(dx, dy, dz, w, NCL);
      __m256d ax0 = _mm256_setzero_pd(), ay0 = _mm256_setzero_pd(), az0 = _mm256_setzero_pd();
      __m256d ax1 = _mm256_setzero_pd(), ay1 = _mm256_setzero_pd(), az1 = _mm256_setzero_pd();
      int k = 0;
      for (; k + 2 <= NCL; k += 2) {
        ax0 = _mm256_fmadd_pd(_mm256_set1_pd(L[lxi[k]]), w[k], ax0);
        ay0 = _mm256_fmadd_pd(_mm256_set1_pd(L[lyi[k]]), w[k], ay0);
        az0 = _mm256_fmadd_pd(_mm256_set1_pd(L[lzi[k]]), w[k], az0);
        ax1 = _mm256_fmadd_pd(_mm256_set1_pd(L[lxi[k + 1]]), w[k + 1], ax1);
        ay1 = _mm256_fmadd_pd(_mm256_set1_pd(L[lyi[k + 1]]), w[k + 1], ay1);
        az1 = _mm256_fmadd_pd(_mm256_set1_pd(L[lzi[k + 1]]), w[k + 1], az1);
      }
      for (; k < NCL; k++) {
        ax0 = _mm256_fmadd_pd(_mm256_set1_pd(L[lxi[k]]), w[k], ax0);
        ay0 = _mm256_fmadd_pd(_mm256_set1_pd(L[lyi[k]]), w[k], ay0);
        az0 = _mm256_fmadd_pd(_mm256_set1_pd(L[lzi[k]]), w[k], az0);
      }
      ax0 = _mm256_add_pd(ax0, ax1); ay0 = _mm256_add_pd(ay0, ay1); az0 = _mm256_add_pd(az0, az1);
      _mm256_maskstore_pd(AX + j, mk, _mm256_add_pd(_mm256_maskload_pd(AX + j, mk), ax0));
      _mm256_maskstore_pd(AY + j, mk, _mm256_add_pd(_mm256_maskload_pd(AY + j, mk), ay0));
      _mm256_maskstore_pd(AZ + j, mk, _mm256_add_pd(_mm256_maskload_pd(AZ + j, mk), az0));
    }
  }

  static inline void transpose4(const __m256d* in, __m256d* out) {
    __m256d t0 = _mm256_unpacklo_pd(in[0], in[1]), t1 = _mm256_unpackhi_pd(in[0], in[1]);
    __m256d t2 = _mm256_unpacklo_pd(in[2], in[3]), t3 = _mm256_unpackhi_pd(in[2], in[3]);
    out[0] = _mm256_permute2f128_pd(t0, t2, 0x20); out[1] = _mm256_permute2f128_pd(t1, t3, 0x20);
    out[2] = _mm256_permute2f128_pd(t0, t2, 0x31); out[3] = _mm256_permute2f128_pd(t1, t3, 0x31);
  }
  static inline void transpose_in(const double* const* src, int k, __m256d* out) {
    __m256d in[4];
    for (int l = 0; l < 4; l++) in[l] = _mm256_load_pd(src[l] + k);
    transpose4(in, out);
  }
  void m2l_flush() {
    if (nbuf == 0) return;
    int cnt = nbuf;
    for (int l = cnt; l < 4; l++) { bufA[l] = bufA[0]; bufB[l] = bufB[0]; }
    const Node *A[4], *B[4];
    for (int l = 0; l < 4; l++) { A[l] = &nodes[bufA[l]]; B[l] = &nodes[bufB[l]]; }
    __m256d R[3];
    R[0] = _mm256_set_pd(B[3]->cx - A[3]->cx, B[2]->cx - A[2]->cx, B[1]->cx - A[1]->cx, B[0]->cx - A[0]->cx);
    R[1] = _mm256_set_pd(B[3]->cy - A[3]->cy, B[2]->cy - A[2]->cy, B[1]->cy - A[1]->cy, B[0]->cy - A[0]->cy);
    R[2] = _mm256_set_pd(B[3]->cz - A[3]->cz, B[2]->cz - A[2]->cz, B[1]->cz - A[1]->cz, B[0]->cz - A[0]->cz);
    __m256d r2 = _mm256_fmadd_pd(R[0], R[0], _mm256_fmadd_pd(R[1], R[1], _mm256_fmadd_pd(R[2], R[2], _mm256_set1_pd(EPS2))));
    __m256d u = _mm256_div_pd(_mm256_set1_pd(1.0), r2);
    __m256d g = _mm256_sqrt_pd(u);
    for (int m = 0; m <= P; m++) {
      Tb[Toff[m]] = g;
      g = _mm256_mul_pd(g, _mm256_mul_pd(u, _mm256_set1_pd(-(2.0 * m + 1.0))));
    }
    const Rec* rp = rec.data(); size_t nr = rec.size();
    for (size_t t = 0; t < nr; t++) {
      const Rec& r = rp[t];
      Tb[r.dst] = _mm256_fmadd_pd(R[r.comp], Tb[r.s1], _mm256_mul_pd(_mm256_set1_pd(r.coef), Tb[r.s2]));
    }
    for (int k = 0; k < NC; k++) Dc[coff[k]] = Tb[k];
    const double *MAp[4], *MBp[4];
    for (int l = 0; l < 4; l++) { MAp[l] = Mul + (size_t)NCP * bufA[l]; MBp[l] = Mul + (size_t)NCP * bufB[l]; }
    for (int k = 0; k < NCP; k += 4) {
      transpose_in(MAp, k, MA + k);
      transpose_in(MBp, k, MB + k);
      for (int t = 0; t < 4; t++) MA[k + t] = _mm256_mul_pd(sgnv[k + t], MA[k + t]);
    }
    for (int kb = 0; kb < NC; kb++) {
      int q = P - dg[kb];
      int na = tet(q + 1);
      const __m256d* Db = Dc + coff[kb];
      __m256d b0 = _mm256_setzero_pd(), b1 = b0, b2 = b0, b3 = b0, a0 = b0, a1 = b0, a2 = b0, a3 = b0;
      int k = 0;
      for (; k + 4 <= na; k += 4) {
        __m256d d0 = Db[coff[k]], d1 = Db[coff[k + 1]], d2 = Db[coff[k + 2]], d3 = Db[coff[k + 3]];
        b0 = _mm256_fmadd_pd(MA[k], d0, b0); a0 = _mm256_fmadd_pd(MB[k], d0, a0);
        b1 = _mm256_fmadd_pd(MA[k + 1], d1, b1); a1 = _mm256_fmadd_pd(MB[k + 1], d1, a1);
        b2 = _mm256_fmadd_pd(MA[k + 2], d2, b2); a2 = _mm256_fmadd_pd(MB[k + 2], d2, a2);
        b3 = _mm256_fmadd_pd(MA[k + 3], d3, b3); a3 = _mm256_fmadd_pd(MB[k + 3], d3, a3);
      }
      for (; k < na; k++) {
        __m256d d0 = Db[coff[k]];
        b0 = _mm256_fmadd_pd(MA[k], d0, b0); a0 = _mm256_fmadd_pd(MB[k], d0, a0);
      }
      LB[kb] = _mm256_add_pd(_mm256_add_pd(b0, b1), _mm256_add_pd(b2, b3));
      LA[kb] = _mm256_mul_pd(_mm256_set1_pd(sgn[kb]), _mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3)));
    }
    for (int k = NC; k < NCP; k++) { LA[k] = _mm256_setzero_pd(); LB[k] = _mm256_setzero_pd(); }
    double *LAp[4], *LBp[4];
    for (int l = 0; l < 4; l++) { LAp[l] = Loc + (size_t)NCP * bufA[l]; LBp[l] = Loc + (size_t)NCP * bufB[l]; }
    for (int k = 0; k < NCP; k += 4) {
      __m256d o[4];
      transpose4(LB + k, o);
      for (int l = 0; l < cnt; l++) _mm256_store_pd(LBp[l] + k, _mm256_add_pd(_mm256_load_pd(LBp[l] + k), o[l]));
      transpose4(LA + k, o);
      for (int l = 0; l < cnt; l++) _mm256_store_pd(LAp[l] + k, _mm256_add_pd(_mm256_load_pd(LAp[l] + k), o[l]));
    }
    for (int l = 0; l < cnt; l++) { hasL[bufB[l]] = 1; hasL[bufA[l]] = 1; }
    nbuf = 0;
  }
  inline void m2l(int a, int b) {
    if (count_only) { cnt_m2l += 1; return; }
    bufA[nbuf] = a; bufB[nbuf] = b; nbuf++;
    if (nbuf == 4) m2l_flush();
  }

  void interact(int ia, int ib) {
    const Node& a = nodes[ia]; const Node& b = nodes[ib];
    double na = a.end - a.begin, nb = b.end - b.begin;
    if (na * nb <= THR) { p2p_mutual(a.begin, a.end, b.begin, b.end); return; }
    double dx = a.cx - b.cx, dy = a.cy - b.cy, dz = a.cz - b.cz;
    double R2 = dx * dx + dy * dy + dz * dz, rs = a.rmax + b.rmax;
    if (rs * rs < THMAX * THMAX * R2) {
      double th = rs / std::sqrt(R2), thp = th;
      for (int k = 0; k < P; k++) thp *= th;
      if (a.mass * thp <= TOL * R2 * b.amin && b.mass * thp <= TOL * R2 * a.amin) { m2l(ia, ib); return; }
    }
    if (a.nchild == 0 && b.nchild == 0) { p2p_mutual(a.begin, a.end, b.begin, b.end); return; }
    bool splitA = (b.nchild == 0) || (a.nchild != 0 && a.rmax >= b.rmax);
    if (splitA) { int c0 = a.child, nc = a.nchild; for (int c = 0; c < nc; c++) interact(c0 + c, ib); }
    else { int c0 = b.child, nc = b.nchild; for (int c = 0; c < nc; c++) interact(ia, c0 + c); }
  }
  void interact_self(int ia) {
    const Node& a = nodes[ia];
    double na = a.end - a.begin;
    if (a.nchild == 0 || 0.5 * na * na <= THR) { p2p_self(a.begin, a.end); return; }
    int c0 = a.child, nc = a.nchild;
    for (int c = 0; c < nc; c++) interact_self(c0 + c);
    for (int c = 0; c < nc; c++) for (int d = c + 1; d < nc; d++) interact(c0 + c, c0 + d);
  }

  int direct_n() const { return std::max(64, (int)(6.0 * (double)tr_a.size() / NC)); }
  void upward() {
    int dn = direct_n();
    for (int ni = (int)nodes.size() - 1; ni >= 0; ni--) {
      const Node& nd = nodes[ni];
      if (nd.nchild == 0 || nd.end - nd.begin <= dn) p2m(ni);
      else for (int c = 0; c < nd.nchild; c++) m2m(nd.child + c, ni);
    }
  }
  void downward() {
    int dn = direct_n();
    for (int ni = 0; ni < (int)nodes.size(); ni++) {
      const Node& nd = nodes[ni];
      if (!hasL[ni]) continue;
      if (nd.nchild == 0 || nd.end - nd.begin <= dn) l2p(ni);
      else for (int c = 0; c < nd.nchild; c++) l2l(ni, nd.child + c);
    }
  }
};

template <int P>
static void run_fmm(double& t_up, double& t_tr, double& t_dn) {
  FMM<P>* f = new FMM<P>();
  f->alloc((int)nodes.size());
  double t0 = cputime();
  f->upward();
  double t1 = cputime();
  f->interact_self(0);
  f->m2l_flush();
  double t2 = cputime();
  f->downward();
  double t3 = cputime();
  t_up = t1 - t0; t_tr = t2 - t1; t_dn = t3 - t2;
  free(f->Mul); free(f->Loc); f->Mul = f->Loc = nullptr;
}
template <int P>
static void count_fmm() {
  FMM<P>* f = new FMM<P>();
  count_only = true; cnt_p2p = 0; cnt_m2l = 0;
  f->interact_self(0);
  count_only = false;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: solve INPUT OUTPUT\n"); return 1; }
  double tstart = cputime();
  FILE* fi = fopen(argv[1], "rb");
  if (!fi) { perror("input"); return 1; }
  fseek(fi, 0, SEEK_END); long bytes = ftell(fi); fseek(fi, 0, SEEK_SET);
  N = (int)(bytes / 32);
  std::vector<double> raw((size_t)N * 4);
  if (N > 0 && fread(raw.data(), 32, N, fi) != (size_t)N) { fprintf(stderr, "read error\n"); return 1; }
  fclose(fi);
  const char* ev;
  int P = 8;
  if ((ev = getenv("FMM_P"))) P = atoi(ev);
  if ((ev = getenv("FMM_THETA"))) THETA = atof(ev);
  if ((ev = getenv("FMM_NLEAF"))) NLEAF = atoi(ev);
  if ((ev = getenv("FMM_THR"))) THR = atof(ev);
  PT.resize(N); TMP.resize(N);
  for (int i = 0; i < N; i++) { PT[i].m = raw[4 * i]; PT[i].x = raw[4 * i + 1]; PT[i].y = raw[4 * i + 2]; PT[i].z = raw[4 * i + 3]; PT[i].id = i; }
  std::vector<double>().swap(raw);
  if ((ev = getenv("FMM_TOL"))) TOL = atof(ev);
  if ((ev = getenv("FMM_THMAX"))) THMAX = atof(ev);
  if ((ev = getenv("FMM_SPARSE"))) SPARSE = atof(ev);
  if ((ev = getenv("FMM_NLEAFMIN"))) NLEAFMIN = atoi(ev);
  {
    double sm = 0, sx = 0, sy = 0, sz = 0;
    for (int i = 0; i < N; i++) { sm += PT[i].m; sx += PT[i].m * PT[i].x; sy += PT[i].m * PT[i].y; sz += PT[i].m * PT[i].z; }
    if (sm > 0) { GCX = sx / sm; GCY = sy / sm; GCZ = sz / sm; }
    std::vector<std::pair<double, int>> rr(N);
    for (int i = 0; i < N; i++) { double dx = PT[i].x - GCX, dy = PT[i].y - GCY, dz = PT[i].z - GCZ; rr[i] = {dx * dx + dy * dy + dz * dz, i}; }
    std::sort(rr.begin(), rr.end());
    AEST.assign(N, 0);
    double cm = 0;
    for (int k = 0; k < N; k++) { cm += PT[rr[k].second].m; AEST[rr[k].second] = cm / (rr[k].first + EPS2); }
    RCORE = N > 0 ? std::sqrt(rr[N / 2].first) : 1.0;
  }
  nodes.reserve(4 * (size_t)N / std::max(1, NLEAF) + 1024);
  Node root; root.begin = 0; root.end = N; root.child = -1; root.nchild = 0; nodes.push_back(root);
  if (N > 0) build_node(0, 0);
  std::vector<Part>().swap(TMP);
  X = (double*)amalloc(8 * (size_t)N + 64); Y = (double*)amalloc(8 * (size_t)N + 64); Z = (double*)amalloc(8 * (size_t)N + 64);
  MS = (double*)amalloc(8 * (size_t)N + 64);
  AX = (double*)amalloc(8 * (size_t)N + 64); AY = (double*)amalloc(8 * (size_t)N + 64); AZ = (double*)amalloc(8 * (size_t)N + 64);
  for (int i = 0; i < N; i++) { X[i] = PT[i].x; Y[i] = PT[i].y; Z[i] = PT[i].z; MS[i] = PT[i].m; AX[i] = AY[i] = AZ[i] = 0; }
  double tbuild = cputime();
  if (getenv("FMM_DRY")) { fprintf(stderr, "N=%d nodes=%zu build=%.3f\n", N, nodes.size(), tbuild - tstart); return 0; }
  double tu = 0, tt = 0, td = 0;
  // ---- parameter selection from the CPU budget ----
  double limit = 0;
  if ((ev = getenv("GALAXY_CPU_LIMIT_S"))) limit = atof(ev);
  if (!(limit > 0)) limit = N <= 100000 ? 0.9 : N <= 400000 ? 4.7 : N <= 2000000 ? 21.8 : 100.0;
  struct Cand { int P; double tol, thr; };
  static const Cand cands[] = { {4, 1e-2, 100}, {6, 1e-3, 300}, {6, 1e-4, 400}, {8, 1e-3, 400}, {8, 1e-4, 600}, {8, 1e-5, 800}, {10, 1e-5, 2000},
                                {10, 1e-6, 2000}, {10, 1e-7, 2500}, {12, 1e-8, 6000}, {12, 1e-10, 8000}, {14, 1e-12, 12000}, {14, 1e-13, 12000} };
  const int NCAND = sizeof(cands) / sizeof(cands[0]);
  auto cm2l = [](int p) { return p <= 4 ? 0.45e-6 : p <= 6 ? 0.9e-6 : p <= 8 ? 1.8e-6 : p <= 10 ? 6e-6 : p <= 12 ? 16e-6 : 33e-6; };
  auto cud = [](int p) { return p <= 4 ? 0.3e-6 : p <= 6 ? 0.4e-6 : p <= 8 ? 0.7e-6 : p <= 10 ? 1.7e-6 : p <= 12 ? 2.7e-6 : 4.2e-6; };
  auto ncoef = [](int p) { return (double)(p + 1) * (p + 2) * (p + 3) / 6; };
  const double CPAIR = 1.75e-9, FRAC = 0.68;
  int chosen = 0;
  bool direct = false;
  double pred_direct = 0.5 * (double)N * N * CPAIR;
  if ((ev = getenv("FMM_P"))) {   // manual override for experiments
    P = atoi(ev); THR = getenv("FMM_THR") ? atof(getenv("FMM_THR")) : THR;
    chosen = -1;
  } else if (pred_direct <= FRAC * limit - (cputime() - tstart) - 0.2e-6 * N) {
    direct = true;
  } else {
    auto fits = [&](int k) {
      const Cand& c = cands[k];
      if ((double)nodes.size() * (ncoef(c.P) + 3) * 16.0 > 0.95e9) return false;
      TOL = c.tol; THR = c.thr;
      switch (c.P) { case 4: count_fmm<4>(); break; case 6: count_fmm<6>(); break; case 8: count_fmm<8>(); break;
                     case 10: count_fmm<10>(); break; case 12: count_fmm<12>(); break; default: count_fmm<14>(); break; }
      double pred = cnt_p2p * CPAIR + cnt_m2l * cm2l(c.P) + (double)N * cud(c.P);
      double avail = FRAC * limit - (cputime() - tstart) - 0.2e-6 * N;
      if (getenv("FMM_VERBOSE")) fprintf(stderr, "cand %d P=%d tol=%g p2p=%.3g m2l=%.3g pred=%.3f avail=%.3f\n", k, c.P, c.tol, cnt_p2p, cnt_m2l, pred, avail);
      return pred <= avail;
    };
    int k = (limit / std::max(N, 1) < 20e-6) ? 2 : 5;
    if (fits(k)) { chosen = k; while (k + 1 < NCAND && fits(k + 1)) { k++; chosen = k; } }
    else { chosen = 0; for (k = k - 1; k >= 1; k--) if (fits(k)) { chosen = k; break; } }
    P = cands[chosen].P; TOL = cands[chosen].tol; THR = cands[chosen].thr;
  }
  if (direct) { THR = 1e300; P = 4; }
  if (getenv("FMM_VERBOSE")) fprintf(stderr, "limit=%.2f chosen=%d P=%d TOL=%g THR=%g direct=%d t=%.3f\n", limit, chosen, P, TOL, THR, (int)direct, cputime() - tstart);
  if (N > 1) switch (P) {
    case 4: run_fmm<4>(tu, tt, td); break;
    case 6: run_fmm<6>(tu, tt, td); break;
    case 8: run_fmm<8>(tu, tt, td); break;
    case 10: run_fmm<10>(tu, tt, td); break;
    case 12: run_fmm<12>(tu, tt, td); break;
    case 14: run_fmm<14>(tu, tt, td); break;
    default: run_fmm<8>(tu, tt, td); break;
  }
  std::vector<double> out((size_t)N * 3);
  for (int i = 0; i < N; i++) { int64_t id = PT[i].id; out[3 * id] = AX[i]; out[3 * id + 1] = AY[i]; out[3 * id + 2] = AZ[i]; }
  FILE* fo = fopen(argv[2], "wb");
  if (!fo) { perror("output"); return 1; }
  fwrite(out.data(), 24, N, fo);
  fclose(fo);
  if (getenv("FMM_VERBOSE")) { FILE* st = fopen("/proc/self/status", "r"); char line[256]; while (st && fgets(line, sizeof line, st)) if (!strncmp(line, "VmHWM", 5)) fputs(line, stderr); if (st) fclose(st); }
  if (getenv("FMM_VERBOSE")) fprintf(stderr, "N=%d nodes=%zu build=%.3f up=%.3f trav=%.3f down=%.3f total=%.3f\n", N, nodes.size(), tbuild - tstart, tu, tt, td, cputime() - tstart);
  return 0;
}
