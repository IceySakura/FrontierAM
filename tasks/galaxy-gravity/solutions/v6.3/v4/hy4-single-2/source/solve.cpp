// Gravitational acceleration for galaxy snapshots.
//
// Adaptive dual-tree FMM (Morton octree) with Cartesian multipole and local
// expansions of order P for the softened Newtonian kernel
//     f(r) = 1 / sqrt(|r|^2 + eps^2),  a = grad_x  sum_j m_j f(x - x_j).
// All coordinates are scaled by a power of two (exact in binary64), so no
// precision is lost in differences of nearby particles.
//
// Single threaded, offline, no external dependencies.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <immintrin.h>
#include <time.h>

static inline double cputime() {
  struct timespec ts;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int DEBUG = 0;
#define DBG(...) do { if (DEBUG) fprintf(stderr, __VA_ARGS__); } while (0)

// ------------------------------------------------------------------ tables
template <int P>
struct Tables {
  static const int NP = (P + 1) * (P + 2) * (P + 3) / 6;
  static const int Q  = 2 * P;
  static const int NQ = (Q + 1) * (Q + 2) * (Q + 3) / 6;
  static const int SP = P + 1;
  static const int SQ = Q + 1;

  int idP[SP * SP * SP];
  int idQ[SQ * SQ * SQ];
  int pi[NP][3];
  int qi[NQ][3];
  int pstart[P + 2];
  int qstart[Q + 2];
  double invfac[Q + 1];
  double invk[Q + 1];
  int tidx[NP * NP];
  int midx[NP];   // packed moment index -> dense (P+1)^3 index
  struct Term { int src; double cA, cB; int axis; };
  Term terms[NQ][12];
  unsigned char nterms[NQ];

  inline int IP(int i, int j, int k) const { return idP[(i * SP + j) * SP + k]; }
  inline int IQ(int i, int j, int k) const { return idQ[(i * SQ + j) * SQ + k]; }

  Tables() {
    memset(idP, -1, sizeof(idP));
    memset(idQ, -1, sizeof(idQ));
    int c = 0;
    for (int q = 0; q <= P; q++) {
      pstart[q] = c;
      for (int i = 0; i <= q; i++)
        for (int j = 0; j <= q - i; j++) {
          int k = q - i - j;
          idP[(i * SP + j) * SP + k] = c;
          pi[c][0] = i; pi[c][1] = j; pi[c][2] = k;
          c++;
        }
    }
    pstart[P + 1] = c;
    c = 0;
    for (int q = 0; q <= Q; q++) {
      qstart[q] = c;
      for (int i = 0; i <= q; i++)
        for (int j = 0; j <= q - i; j++) {
          int k = q - i - j;
          idQ[(i * SQ + j) * SQ + k] = c;
          qi[c][0] = i; qi[c][1] = j; qi[c][2] = k;
          c++;
        }
    }
    qstart[Q + 1] = c;
    double fac = 1.0;
    invfac[0] = 1.0;
    invk[0] = 0.0;
    for (int i = 1; i <= Q; i++) { fac *= (double)i; invfac[i] = 1.0 / fac; invk[i] = 1.0 / (double)i; }
    for (int n = 0; n < NP; n++)
      for (int m = 0; m < NP; m++)
        tidx[n * NP + m] = ((pi[n][0] + pi[m][0]) * SQ + (pi[n][1] + pi[m][1])) * SQ
                           + (pi[n][2] + pi[m][2]);
    for (int m = 0; m < NP; m++)
      midx[m] = (pi[m][0] * SP + pi[m][1]) * SP + pi[m][2];
    // derivative recipe:  D_t = -(1/s) * sum_terms (cA + cB*R_axis) * D_src
    memset(nterms, 0, sizeof(nterms));
    for (int t = 1; t < NQ; t++) {
      int tv[3] = { qi[t][0], qi[t][1], qi[t][2] };
      int ax = tv[0] > 0 ? 0 : (tv[1] > 0 ? 1 : 2);
      int nv[3] = { tv[0], tv[1], tv[2] };
      nv[ax]--;
      int nt = 0;
      auto add = [&](int src, double cA, double cB, int axis) {
        terms[t][nt].src = src; terms[t][nt].cA = cA;
        terms[t][nt].cB = cB; terms[t][nt].axis = axis; nt++;
      };
      add(IQ(nv[0], nv[1], nv[2]), 0.0, 1.0, ax);
      if (nv[ax] > 0) {
        int w[3] = { nv[0], nv[1], nv[2] }; w[ax]--;
        add(IQ(w[0], w[1], w[2]), (double)nv[ax], 0.0, 0);
      }
      for (int k = 0; k < 3; k++) {
        if (nv[k] > 0) {
          int w[3] = { nv[0], nv[1], nv[2] }; w[k]--; w[ax]++;
          add(IQ(w[0], w[1], w[2]), 0.0, 2.0 * (double)nv[k], k);
        }
        if (nv[k] > 1) {
          int w[3] = { nv[0], nv[1], nv[2] }; w[k] -= 2; w[ax]++;
          add(IQ(w[0], w[1], w[2]), (double)(nv[k] * (nv[k] - 1)), 0.0, 0);
        }
      }
      nterms[t] = (unsigned char)nt;
    }
  }
};

// -------------------------------------------------------------------- tree
struct Tree {
  std::vector<int> beg, en, child, level, parent;
  std::vector<double> cx, cy, cz, hx, hy, hz, rad;
  int nnodes = 0;
  int nleaf = 0;
  int newnode(int b, int e, int lev, int par) {
    int id = nnodes++;
    beg.push_back(b); en.push_back(e); level.push_back(lev); parent.push_back(par);
    for (int k = 0; k < 8; k++) child.push_back(-1);
    cx.push_back(0); cy.push_back(0); cz.push_back(0);
    hx.push_back(0); hy.push_back(0); hz.push_back(0); rad.push_back(0);
    return id;
  }
};

static inline uint64_t spread3(uint32_t a) {
  uint64_t x = a & 0x1fffffULL;
  x = (x | (x << 32)) & 0x1f00000000ffffULL;
  x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
  x = (x | (x << 8))  & 0x100f00f00f00f00fULL;
  x = (x | (x << 4))  & 0x10c30c30c30c30c3ULL;
  x = (x | (x << 2))  & 0x1249249249249249ULL;
  return x;
}

struct KV { uint64_t key; uint32_t idx; };

// --------------------------------------------------------------- solver
template <int P>
struct FMM {
  typedef Tables<P> T;
  T tab;
  static const int NP = T::NP;
  static const int NQ = T::NQ;

  int n;
  double *X, *Y, *Z;          // scaled positions in Morton (sorted) order
  uint32_t *perm;             // sorted -> original index
  double *AX, *AY, *AZ;       // accumulators (sorted order)
  double mass, eps2;
  double theta, theta2;
  Tree tr;
  double *M, *L;              // multipole / local coefficients per node
  double *Tbuf, *Hbuf;        // scratch for M2L
  long long nM2L = 0, nP2P = 0, nP2Ppairs = 0;
  int countonly = 0;

  FMM(int n_, double *x, double *y, double *z, uint32_t *pm, double mass_,
      double eps2_, double theta_, int leafsize, double *ax, double *ay, double *az)
      : n(n_), X(x), Y(y), Z(z), perm(pm), AX(ax), AY(ay), AZ(az),
        mass(mass_), eps2(eps2_), theta(theta_) {
    theta2 = theta * theta;
    build_tree(leafsize);
    size_t sz = (size_t)tr.nnodes * NP;
    if (!countonly) {
      posix_memalign((void **)&M, 64, sz * sizeof(double));
      posix_memalign((void **)&L, 64, sz * sizeof(double));
      memset(M, 0, sz * sizeof(double));
      memset(L, 0, sz * sizeof(double));
      posix_memalign((void **)&Tbuf, 64, (size_t)T::SQ * T::SQ * T::SQ * sizeof(double));
      memset(Tbuf, 0, (size_t)T::SQ * T::SQ * T::SQ * sizeof(double));
      posix_memalign((void **)&Hbuf, 64, (size_t)T::SP * T::SP * T::SP * sizeof(double));
    } else { M = L = Tbuf = Hbuf = nullptr; }
  }
  ~FMM() { free(M); free(L); free(Tbuf); free(Hbuf); }

  // -- Morton octree construction -------------------------------------------
  void build_tree(int leafsize) {
    double xmin = X[0], xmax = X[0], ymin = Y[0], ymax = Y[0], zmin = Z[0], zmax = Z[0];
    for (int i = 1; i < n; i++) {
      xmin = std::min(xmin, X[i]); xmax = std::max(xmax, X[i]);
      ymin = std::min(ymin, Y[i]); ymax = std::max(ymax, Y[i]);
      zmin = std::min(zmin, Z[i]); zmax = std::max(zmax, Z[i]);
    }
    double lo[3] = { xmin, ymin, zmin };
    double side = std::max(std::max(xmax - xmin, ymax - ymin), zmax - zmin);
    if (!(side > 0)) side = 1.0;
    const double SCALE = 2097152.0 * 0.999999;  // 2^21
    double inv = SCALE / side;
    KV *kv = (KV *)malloc((size_t)n * sizeof(KV));
    for (int i = 0; i < n; i++) {
      uint32_t ix = (uint32_t)std::min(SCALE, std::max(0.0, (X[i] - lo[0]) * inv));
      uint32_t iy = (uint32_t)std::min(SCALE, std::max(0.0, (Y[i] - lo[1]) * inv));
      uint32_t iz = (uint32_t)std::min(SCALE, std::max(0.0, (Z[i] - lo[2]) * inv));
      kv[i].key = spread3(ix) | (spread3(iy) << 1) | (spread3(iz) << 2);
      kv[i].idx = (uint32_t)i;
    }
    // LSD radix sort by 64-bit key, 11 bits per pass
    KV *tmp = (KV *)malloc((size_t)n * sizeof(KV));
    KV *a = kv, *b = tmp;
    for (int pass = 0; pass < 6; pass++) {
      int shift = pass * 11;
      int cnt[2048] = {0};
      for (int i = 0; i < n; i++) cnt[(a[i].key >> shift) & 2047]++;
      int sum = 0;
      for (int d = 0; d < 2048; d++) { int c = cnt[d]; cnt[d] = sum; sum += c; }
      for (int i = 0; i < n; i++) b[cnt[(a[i].key >> shift) & 2047]++] = a[i];
      std::swap(a, b);
    }
    KV *sorted = a;
    // reorder positions, perm
    double *nx = (double *)malloc((size_t)n * sizeof(double));
    double *ny = (double *)malloc((size_t)n * sizeof(double));
    double *nz = (double *)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) {
      uint32_t o = sorted[i].idx;
      perm[i] = o;
      nx[i] = X[o]; ny[i] = Y[o]; nz[i] = Z[o];
    }
    memcpy(X, nx, (size_t)n * sizeof(double));
    memcpy(Y, ny, (size_t)n * sizeof(double));
    memcpy(Z, nz, (size_t)n * sizeof(double));
    free(nx); free(ny); free(nz);
    tr.newnode(0, n, 0, -1);
    // recursive splitting using sorted Morton keys
    // (keys are recovered from positions via the same quantization)
    make_children(0, sorted, 0, leafsize);
    free(kv); free(tmp);
  }

  void make_children(int node, const KV *sorted, int lev, int leafsize) {
    int b = tr.beg[node], e = tr.en[node];
    if (e - b <= leafsize || lev >= 20) { finish_leaf(node); return; }
    int shift = 3 * (20 - lev);
    uint64_t base = sorted[b].key >> (shift + 3) << (shift + 3);
    for (int oct = 0; oct < 8; oct++) {
      uint64_t key = base | ((uint64_t)oct << shift);
      // boundaries within [b,e)
      int lo = b, hi = e;
      while (lo < hi) { int mid = (lo + hi) >> 1; if (sorted[mid].key < key) lo = mid + 1; else hi = mid; }
      int cb = lo;
      lo = cb; hi = e;
      uint64_t keyend = key + ((uint64_t)1 << shift);
      while (lo < hi) { int mid = (lo + hi) >> 1; if (sorted[mid].key < keyend) lo = mid + 1; else hi = mid; }
      int ce = lo;
      if (ce > cb) {
        int ch = tr.newnode(cb, ce, lev + 1, node);
        tr.child[node * 8 + oct] = ch;
        make_children(ch, sorted, lev + 1, leafsize);
      }
    }
    // union of children bounding boxes
    double mn[3] = {1e300, 1e300, 1e300}, mx[3] = {-1e300, -1e300, -1e300};
    for (int oct = 0; oct < 8; oct++) {
      int ch = tr.child[node * 8 + oct];
      if (ch < 0) continue;
      double cmn[3] = { tr.cx[ch] - tr.hx[ch], tr.cy[ch] - tr.hy[ch], tr.cz[ch] - tr.hz[ch] };
      double cmx[3] = { tr.cx[ch] + tr.hx[ch], tr.cy[ch] + tr.hy[ch], tr.cz[ch] + tr.hz[ch] };
      for (int k = 0; k < 3; k++) { mn[k] = std::min(mn[k], cmn[k]); mx[k] = std::max(mx[k], cmx[k]); }
    }
    set_box(node, mn, mx);
  }

  void finish_leaf(int node) {
    int b = tr.beg[node], e = tr.en[node];
    double mn[3] = {1e300, 1e300, 1e300}, mx[3] = {-1e300, -1e300, -1e300};
    for (int i = b; i < e; i++) {
      mn[0] = std::min(mn[0], X[i]); mx[0] = std::max(mx[0], X[i]);
      mn[1] = std::min(mn[1], Y[i]); mx[1] = std::max(mx[1], Y[i]);
      mn[2] = std::min(mn[2], Z[i]); mx[2] = std::max(mx[2], Z[i]);
    }
    set_box(node, mn, mx);
    tr.nleaf++;
  }
  void set_box(int node, const double *mn, const double *mx) {
    double h[3];
    for (int k = 0; k < 3; k++) h[k] = 0.5 * (mx[k] - mn[k]);
    tr.cx[node] = 0.5 * (mn[0] + mx[0]); tr.cy[node] = 0.5 * (mn[1] + mx[1]);
    tr.cz[node] = 0.5 * (mn[2] + mx[2]);
    // a tiny nonzero half-extent keeps the MAC well defined
    tr.hx[node] = std::max(h[0], 1e-300); tr.hy[node] = std::max(h[1], 1e-300);
    tr.hz[node] = std::max(h[2], 1e-300);
    tr.rad[node] = std::sqrt(tr.hx[node] * tr.hx[node] + tr.hy[node] * tr.hy[node] +
                             tr.hz[node] * tr.hz[node]);
  }

  // -- upward pass: P2M then M2M -------------------------------------------
  void p2m(int node) {
    double *A = M + (size_t)node * NP;
    memset(A, 0, NP * sizeof(double));
    int b = tr.beg[node], e = tr.en[node];
    double cc[3] = { tr.cx[node], tr.cy[node], tr.cz[node] };
    for (int i = b; i < e; i++) {
      double d[3] = { -(X[i] - cc[0]), -(Y[i] - cc[1]), -(Z[i] - cc[2]) };
      double px[P + 1], py[P + 1], pz[P + 1];
      px[0] = py[0] = pz[0] = 1.0;
      for (int k = 1; k <= P; k++) {
        px[k] = px[k - 1] * d[0] * tab.invk[k];
        py[k] = py[k - 1] * d[1] * tab.invk[k];
        pz[k] = pz[k - 1] * d[2] * tab.invk[k];
      }
      for (int m = 0; m < NP; m++) {
        int *v = tab.pi[m];
        A[m] += px[v[0]] * py[v[1]] * pz[v[2]];
      }
    }
    for (int m = 0; m < NP; m++) A[m] *= mass;
  }

  void m2m(int par, int ch) {
    double sx[P + 1], sy[P + 1], sz[P + 1];
    double d[3] = { -(tr.cx[ch] - tr.cx[par]), -(tr.cy[ch] - tr.cy[par]),
                    -(tr.cz[ch] - tr.cz[par]) };
    sx[0] = sy[0] = sz[0] = 1.0;
    for (int k = 1; k <= P; k++) {
      sx[k] = sx[k - 1] * d[0] * tab.invk[k];
      sy[k] = sy[k - 1] * d[1] * tab.invk[k];
      sz[k] = sz[k - 1] * d[2] * tab.invk[k];
    }
    double *src = M + (size_t)ch * NP;
    double *dst = M + (size_t)par * NP;
    double tmp[NP];
    // separable shifts along x, y, z
    for (int a = 0; a < NP; a++) {
      const int *v = tab.pi[a];
      int i = v[0], j = v[1], k = v[2];
      double s = 0;
      for (int p = 0; p <= i; p++) s += src[tab.IP(p, j, k)] * sx[i - p];
      tmp[a] = s;
    }
    for (int a = 0; a < NP; a++) {
      const int *v = tab.pi[a];
      int i = v[0], j = v[1], k = v[2];
      double s = 0;
      for (int p = 0; p <= j; p++) s += tmp[tab.IP(i, p, k)] * sy[j - p];
      dst[a] = s;
    }
    // copy back via a second temp (avoid overwrite issues)
    static double tmp2[NP];
    for (int a = 0; a < NP; a++) {
      const int *v = tab.pi[a];
      int i = v[0], j = v[1], k = v[2];
      double s = 0;
      for (int p = 0; p <= k; p++) s += dst[tab.IP(i, j, p)] * sz[k - p];
      tmp2[a] = s;
    }
    memcpy(dst, tmp2, NP * sizeof(double));
  }

  void upward() {
    for (int node = tr.nnodes - 1; node >= 0; node--) {
      int c0 = tr.child[node * 8];
      if (c0 < 0 && tr.child[node * 8 + 1] < 0 && tr.child[node * 8 + 2] < 0 &&
          tr.child[node * 8 + 3] < 0 && tr.child[node * 8 + 4] < 0 &&
          tr.child[node * 8 + 5] < 0 && tr.child[node * 8 + 6] < 0 &&
          tr.child[node * 8 + 7] < 0) {
        p2m(node);
      } else {
        double *A = M + (size_t)node * NP;
        memset(A, 0, NP * sizeof(double));
        for (int oct = 0; oct < 8; oct++) {
          int ch = tr.child[node * 8 + oct];
          if (ch < 0) continue;
          double tmp[NP];
          m2m_into(node, ch, tmp);
          for (int a = 0; a < NP; a++) A[a] += tmp[a];
        }
      }
    }
  }
  void m2m_into(int par, int ch, double *out) {
    double sx[P + 1], sy[P + 1], sz[P + 1];
    double d[3] = { -(tr.cx[ch] - tr.cx[par]), -(tr.cy[ch] - tr.cy[par]),
                    -(tr.cz[ch] - tr.cz[par]) };
    sx[0] = sy[0] = sz[0] = 1.0;
    for (int k = 1; k <= P; k++) {
      sx[k] = sx[k - 1] * d[0] * tab.invk[k];
      sy[k] = sy[k - 1] * d[1] * tab.invk[k];
      sz[k] = sz[k - 1] * d[2] * tab.invk[k];
    }
    double *src = M + (size_t)ch * NP;
    double t1[NP], t2[NP];
    for (int a = 0; a < NP; a++) {
      const int *v = tab.pi[a];
      int i = v[0], j = v[1], k = v[2];
      double s = 0;
      for (int p = 0; p <= i; p++) s += src[tab.IP(p, j, k)] * sx[i - p];
      t1[a] = s;
    }
    for (int a = 0; a < NP; a++) {
      const int *v = tab.pi[a];
      int i = v[0], j = v[1], k = v[2];
      double s = 0;
      for (int p = 0; p <= j; p++) s += t1[tab.IP(i, p, k)] * sy[j - p];
      t2[a] = s;
    }
    for (int a = 0; a < NP; a++) {
      const int *v = tab.pi[a];
      int i = v[0], j = v[1], k = v[2];
      double s = 0;
      for (int p = 0; p <= k; p++) s += t2[tab.IP(i, j, p)] * sz[k - p];
      out[a] = s;
    }
  }

  // -- M2L: source multipole -> target local expansion ----------------------
  // Derivatives are held in a dense (Q+1)^3 array so that the recurrence uses
  // fixed strides (no gathers).  D[t] = -(1/s) * [R_a D_{t-e_a} + ...].
  int no_m2l = 0;
  unsigned long long cyA = 0, cyB = 0, cyC = 0;
  inline void m2l(int t, int s) {
    if (no_m2l) return;
    unsigned long long cy0 = __rdtsc();
    double R[3] ={ tr.cx[t] - tr.cx[s], tr.cy[t] - tr.cy[s], tr.cz[t] - tr.cz[s] };
    double ss = R[0] * R[0] + R[1] * R[1] + R[2] * R[2] + eps2;
    double invs = 1.0 / ss;
    double *D = Tbuf;
    const int SQ = T::SQ;
    const int st[3] = { SQ * SQ, SQ, 1 };
    D[0] = 1.0 / std::sqrt(ss);
    for (int q = 1; q <= T::Q; q++) {
      for (int i = 0; i <= q; i++) {
        for (int j = 0; j <= q - i; j++) {
          int k = q - i - j;
          int tv[3] = { i, j, k };
          int a = i > 0 ? 0 : (j > 0 ? 1 : 2);
          int nn[3] = { i, j, k }; nn[a]--;
          int ti = (i * SQ + j) * SQ + k;
          double val = R[a] * D[ti - st[a]];
          if (nn[a] > 0) val += (double)nn[a] * D[ti - 2 * st[a]];
          for (int c = 0; c < 3; c++) {
            if (nn[c] > 0) val += 2.0 * (double)nn[c] * R[c] * D[ti - st[c]];
            if (nn[c] > 1) val += (double)(nn[c] * (nn[c] - 1)) * D[ti - 2 * st[c]];
          }
          D[ti] = -val * invs;
        }
      }
    }
    unsigned long long cy1 = __rdtsc();
    double *A = M + (size_t)s * NP;
    double *Lp = L + (size_t)t * NP;
    const int SP = T::SP;
    double *Ad = Hbuf;
    memset(Ad, 0, (size_t)SP * SP * SP * sizeof(double));
    for (int m = 0; m < NP; m++) Ad[tab.midx[m]] = A[m];
    for (int n = 0; n < NP; n++) {
      const int *v = tab.pi[n];
      const double *Dp = D + (v[0] * SQ + v[1]) * SQ + v[2];
      double acc = 0;
      for (int m0 = 0; m0 <= P; m0++) {
        for (int m1 = 0; m1 <= P - m0; m1++) {
          const double *Ap = Ad + (m0 * SP + m1) * SP;
          const double *Dq = Dp + (m0 * SQ + m1) * SQ;
          int len = P - m0 - m1 + 1, m2 = 0;
          __m256d vacc = _mm256_setzero_pd();
          for (; m2 + 3 < len; m2 += 4)
            vacc = _mm256_fmadd_pd(_mm256_loadu_pd(Ap + m2), _mm256_loadu_pd(Dq + m2), vacc);
          double tmp[4]; _mm256_storeu_pd(tmp, vacc);
          acc += tmp[0] + tmp[1] + tmp[2] + tmp[3];
          for (; m2 < len; m2++) acc += Ap[m2] * Dq[m2];
        }
      }
      Lp[n] += acc;
    }
    unsigned long long cy2 = __rdtsc();
    cyA += cy1 - cy0; cyB += cy2 - cy1; cyC += cy2 - cy0;
  }

  // -- direct particle-particle interaction --------------------------------
  inline void p2p(int tb, int te, int sb, int se) {
    const __m256d veps = _mm256_set1_pd(eps2);
    const __m256d vmass = _mm256_set1_pd(mass);
    const __m256d one = _mm256_set1_pd(1.0);
    int i = tb;
    for (; i + 3 < te; i += 4) {
      __m256d xi = _mm256_loadu_pd(X + i), yi = _mm256_loadu_pd(Y + i),
              zi = _mm256_loadu_pd(Z + i);
      __m256d ax = _mm256_loadu_pd(AX + i), ay = _mm256_loadu_pd(AY + i),
              az = _mm256_loadu_pd(AZ + i);
      for (int j = sb; j < se; j++) {
        __m256d dx = _mm256_sub_pd(_mm256_set1_pd(X[j]), xi);
        __m256d dy = _mm256_sub_pd(_mm256_set1_pd(Y[j]), yi);
        __m256d dz = _mm256_sub_pd(_mm256_set1_pd(Z[j]), zi);
        __m256d r2 = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(dx, dx),
                                    _mm256_mul_pd(dy, dy)),
                                   _mm256_add_pd(_mm256_mul_pd(dz, dz), veps));
        __m256d inv = _mm256_mul_pd(vmass, _mm256_div_pd(one, _mm256_mul_pd(r2, _mm256_sqrt_pd(r2))));
        ax = _mm256_fmadd_pd(dx, inv, ax);
        ay = _mm256_fmadd_pd(dy, inv, ay);
        az = _mm256_fmadd_pd(dz, inv, az);
      }
      _mm256_storeu_pd(AX + i, ax); _mm256_storeu_pd(AY + i, ay);
      _mm256_storeu_pd(AZ + i, az);
    }
    for (; i < te; i++) {
      double xi = X[i], yi = Y[i], zi = Z[i];
      double ax = AX[i], ay = AY[i], az = AZ[i];
      for (int j = sb; j < se; j++) {
        double dx = X[j] - xi, dy = Y[j] - yi, dz = Z[j] - zi;
        double r2 = dx * dx + dy * dy + dz * dz + eps2;
        double inv = mass / (r2 * std::sqrt(r2));
        ax += dx * inv; ay += dy * inv; az += dz * inv;
      }
      AX[i] = ax; AY[i] = ay; AZ[i] = az;
    }
  }

  inline int isleaf(int node) const { return tr.child[node * 8] < 0 &&
         tr.child[node * 8 + 1] < 0 && tr.child[node * 8 + 2] < 0 &&
         tr.child[node * 8 + 3] < 0 && tr.child[node * 8 + 4] < 0 &&
         tr.child[node * 8 + 5] < 0 && tr.child[node * 8 + 6] < 0 &&
         tr.child[node * 8 + 7] < 0; }

  // -- dual tree traversal --------------------------------------------------
  void traverse() {
    std::vector<int> stk;
    stk.reserve(4096);
    stk.push_back(0); stk.push_back(0);
    while (!stk.empty()) {
      int s = stk.back(); stk.pop_back();
      int t = stk.back(); stk.pop_back();
      double dx = tr.cx[t] - tr.cx[s], dy = tr.cy[t] - tr.cy[s],
             dz = tr.cz[t] - tr.cz[s];
      double d2 = dx * dx + dy * dy + dz * dz;
      double rs = tr.rad[t] + tr.rad[s];
      if (d2 > (rs * rs) / theta2) {
        nM2L++;
        if (!countonly) m2l(t, s);
        continue;
      }
      int lt = isleaf(t), ls = isleaf(s);
      if (lt && ls) {
        nP2P++;
        nP2Ppairs += (long long)(tr.en[t] - tr.beg[t]) * (tr.en[s] - tr.beg[s]);
        if (!countonly) p2p(tr.beg[t], tr.en[t], tr.beg[s], tr.en[s]);
        continue;
      }
      double rt = tr.rad[t], rsv = tr.rad[s];
      int splitt, splits;
      if (lt) { splitt = 0; splits = 1; }
      else if (ls) { splitt = 1; splits = 0; }
      else if (rt > 1.4 * rsv) { splitt = 1; splits = 0; }
      else if (rsv > 1.4 * rt) { splitt = 0; splits = 1; }
      else { splitt = 1; splits = 1; }
      if (splitt && splits) {
        for (int a = 0; a < 8; a++) {
          int ca = tr.child[t * 8 + a]; if (ca < 0) continue;
          for (int b = 0; b < 8; b++) {
            int cb = tr.child[s * 8 + b]; if (cb < 0) continue;
            stk.push_back(ca); stk.push_back(cb);
          }
        }
      } else if (splitt) {
        for (int a = 0; a < 8; a++) {
          int ca = tr.child[t * 8 + a]; if (ca < 0) continue;
          stk.push_back(ca); stk.push_back(s);
        }
      } else {
        for (int b = 0; b < 8; b++) {
          int cb = tr.child[s * 8 + b]; if (cb < 0) continue;
          stk.push_back(t); stk.push_back(cb);
        }
      }
    }
  }

  // -- leaf-targeted traversal: each leaf collects M2L from well-separated
  //    source nodes and P2P from nearby leaves; no L2L is needed.
  void traverse_leaves() {
    std::vector<int> stk;
    stk.reserve(2048);
    for (int t = 0; t < tr.nnodes; t++) {
      if (!isleaf(t)) continue;
      stk.clear();
      stk.push_back(0);
      while (!stk.empty()) {
        int s = stk.back(); stk.pop_back();
        double dx = tr.cx[t] - tr.cx[s], dy = tr.cy[t] - tr.cy[s],
               dz = tr.cz[t] - tr.cz[s];
        double d2 = dx * dx + dy * dy + dz * dz;
        double rs = tr.rad[t] + tr.rad[s];
        if (d2 > (rs * rs) / theta2) {
          nM2L++;
          if (!countonly) m2l(t, s);
          continue;
        }
        if (isleaf(s)) {
          nP2P++;
          nP2Ppairs += (long long)(tr.en[t] - tr.beg[t]) * (tr.en[s] - tr.beg[s]);
          if (!countonly) p2p(tr.beg[t], tr.en[t], tr.beg[s], tr.en[s]);
          continue;
        }
        for (int b = 0; b < 8; b++) {
          int cb = tr.child[s * 8 + b];
          if (cb >= 0) stk.push_back(cb);
        }
      }
    }
  }

  // -- downward pass: L2L then L2P -----------------------------------------
  void l2l(int par, int ch) {
    double sx[P + 1], sy[P + 1], sz[P + 1];
    double d[3] = { tr.cx[ch] - tr.cx[par], tr.cy[ch] - tr.cy[par],
                    tr.cz[ch] - tr.cz[par] };
    sx[0] = sy[0] = sz[0] = 1.0;
    for (int k = 1; k <= P; k++) {
      sx[k] = sx[k - 1] * d[0] * tab.invk[k];
      sy[k] = sy[k - 1] * d[1] * tab.invk[k];
      sz[k] = sz[k - 1] * d[2] * tab.invk[k];
    }
    double *src = L + (size_t)par * NP;
    double *dst = L + (size_t)ch * NP;
    double t1[NP], t2[NP], t3[NP];
    for (int a = 0; a < NP; a++) {
      const int *v = tab.pi[a];
      int i = v[0], j = v[1], k = v[2];
      double s = 0;
      for (int p = i; p <= P - j - k; p++) s += src[tab.IP(p, j, k)] * sx[p - i];
      t1[a] = s;
    }
    for (int a = 0; a < NP; a++) {
      const int *v = tab.pi[a];
      int i = v[0], j = v[1], k = v[2];
      double s = 0;
      for (int p = j; p <= P - i - k; p++) s += t1[tab.IP(i, p, k)] * sy[p - j];
      t2[a] = s;
    }
    for (int a = 0; a < NP; a++) {
      const int *v = tab.pi[a];
      int i = v[0], j = v[1], k = v[2];
      double s = 0;
      for (int p = k; p <= P - i - j; p++) s += t2[tab.IP(i, j, p)] * sz[p - k];
      t3[a] = s;
    }
    for (int a = 0; a < NP; a++) dst[a] += t3[a];
  }

  void l2p() {
    for (int node = 0; node < tr.nnodes; node++) {
      if (!isleaf(node)) continue;
      double *Lp = L + (size_t)node * NP;
      double cc[3] = { tr.cx[node], tr.cy[node], tr.cz[node] };
      for (int i = tr.beg[node]; i < tr.en[node]; i++) {
        double d[3] = { X[i] - cc[0], Y[i] - cc[1], Z[i] - cc[2] };
        double px[P + 1], py[P + 1], pz[P + 1];
        px[0] = py[0] = pz[0] = 1.0;
        for (int k = 1; k <= P; k++) {
          px[k] = px[k - 1] * d[0] * tab.invk[k];
          py[k] = py[k - 1] * d[1] * tab.invk[k];
          pz[k] = pz[k - 1] * d[2] * tab.invk[k];
        }
        double ax = 0, ay = 0, az = 0;
        for (int m = 0; m < NP; m++) {
          const int *v = tab.pi[m];
          double c = Lp[m];
          if (c == 0.0) continue;
          if (v[0] > 0) ax += c * px[v[0] - 1] * py[v[1]] * pz[v[2]];
          if (v[1] > 0) ay += c * px[v[0]] * py[v[1] - 1] * pz[v[2]];
          if (v[2] > 0) az += c * px[v[0]] * py[v[1]] * pz[v[2] - 1];
        }
        AX[i] += ax; AY[i] += ay; AZ[i] += az;
      }
    }
  }

  void run() {
    double t0 = cputime();
    if (!countonly) upward();
    double t1 = cputime();
    traverse_leaves();
    double t2 = cputime();
    if (!countonly) {
      l2p();
    }
    double t3 = cputime();
    DBG("      up %.3f trav %.3f down %.3f  nodes %d leaves %d m2l %lld p2p %lld pairs %lld\n",
        t1 - t0, t2 - t1, t3 - t2, tr.nnodes, tr.nleaf, nM2L, nP2P, nP2Ppairs);
  }
};

// ------------------------------------------------------------- direct N^2
static void direct_sum(int n, const double *rx, const double *ry, const double *rz,
                       double mass, double eps2, double *ox, double *oy, double *oz) {
  const __m256d veps = _mm256_set1_pd(eps2);
  const __m256d vmass = _mm256_set1_pd(mass);
  const __m256d one = _mm256_set1_pd(1.0);
  for (int i = 0; i < n; i++) {
    __m256d xi = _mm256_set1_pd(rx[i]), yi = _mm256_set1_pd(ry[i]), zi = _mm256_set1_pd(rz[i]);
    __m256d ax = _mm256_setzero_pd(), ay = _mm256_setzero_pd(), az = _mm256_setzero_pd();
    int j = 0;
    for (; j + 3 < n; j += 4) {
      __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(rx + j), xi);
      __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(ry + j), yi);
      __m256d dz = _mm256_sub_pd(_mm256_loadu_pd(rz + j), zi);
      __m256d r2 = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
                                 _mm256_add_pd(_mm256_mul_pd(dz, dz), veps));
      __m256d inv = _mm256_mul_pd(vmass, _mm256_div_pd(one, _mm256_mul_pd(r2, _mm256_sqrt_pd(r2))));
      ax = _mm256_fmadd_pd(dx, inv, ax); ay = _mm256_fmadd_pd(dy, inv, ay);
      az = _mm256_fmadd_pd(dz, inv, az);
    }
    double t[4];
    _mm256_storeu_pd(t, ax); ox[i] = t[0] + t[1] + t[2] + t[3];
    _mm256_storeu_pd(t, ay); oy[i] = t[0] + t[1] + t[2] + t[3];
    _mm256_storeu_pd(t, az); oz[i] = t[0] + t[1] + t[2] + t[3];
    for (; j < n; j++) {
      double dx = rx[j] - rx[i], dy = ry[j] - ry[i], dz = rz[j] - rz[i];
      double r2 = dx * dx + dy * dy + dz * dz + eps2;
      double inv = mass / (r2 * std::sqrt(r2));
      ox[i] += dx * inv; oy[i] += dy * inv; oz[i] += dz * inv;
    }
  }
}

// ------------------------------------------------------------------- main
template <int P>
static void run_fmm(int n, double *x, double *y, double *z, uint32_t *perm,
                    double mass, double eps2, double theta, int leafsize,
                    double *ax, double *ay, double *az) {
  FMM<P> *f = new FMM<P>(n, x, y, z, perm, mass, eps2, theta, leafsize, ax, ay, az);
  if (getenv("GALAXY_NO_M2L")) f->no_m2l = 1;
  if (getenv("GALAXY_COUNT_ONLY")) {
    f->countonly = 1;
    free(f->M); free(f->L); free(f->Tbuf); free(f->Hbuf);
    f->M = f->L = f->Tbuf = f->Hbuf = nullptr;
  }
  f->run();
  if (DEBUG) fprintf(stderr, "      m2l cycles/M2L: deriv %.0f contract %.0f total %.0f (n=%lld)\n",
      (double)f->cyA / (f->nM2L + 1), (double)f->cyB / (f->nM2L + 1),
      (double)f->cyC / (f->nM2L + 1), f->nM2L);
  delete f;
}

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: solve INPUT OUTPUT\n"); return 1; }
  double tstart = cputime();
  const char *dbg = getenv("GALAXY_DEBUG");
  if (dbg) DEBUG = atoi(dbg);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  const char *inpath = argv[1], *outpath = argv[2];
  FILE *fin = fopen(inpath, "rb");
  if (!fin) { fprintf(stderr, "cannot open input\n"); return 1; }
  fseek(fin, 0, SEEK_END);
  long long fsz = ftell(fin);
  fseek(fin, 0, SEEK_SET);
  int n = (int)(fsz / 32);
  if (n <= 0) { FILE *f = fopen(outpath, "wb"); if (f) fclose(f); return 0; }
  unsigned char *raw = (unsigned char *)malloc((size_t)fsz);
  if (fread(raw, 1, (size_t)fsz, fin) != (size_t)fsz) { fprintf(stderr, "short read\n"); return 1; }
  fclose(fin);
  double *x = (double *)malloc((size_t)(n + 8) * sizeof(double));
  double *y = (double *)malloc((size_t)(n + 8) * sizeof(double));
  double *z = (double *)malloc((size_t)(n + 8) * sizeof(double));
  uint32_t *perm = (uint32_t *)malloc((size_t)(n + 8) * sizeof(uint32_t));
  double *ax = (double *)calloc((size_t)(n + 8), sizeof(double));
  double *ay = (double *)calloc((size_t)(n + 8), sizeof(double));
  double *az = (double *)calloc((size_t)(n + 8), sizeof(double));
  double m0 = 0, mmax = 0, maxabs = 0;
  {
    const double *v = (const double *)raw;
    m0 = v[0];
    for (int i = 0; i < n; i++) {
      x[i] = v[4 * i + 1]; y[i] = v[4 * i + 2]; z[i] = v[4 * i + 3];
      double mi = v[4 * i];
      if (fabs(mi - m0) > mmax) mmax = fabs(mi - m0);
      maxabs = std::max(maxabs, fabs(x[i]));
      maxabs = std::max(maxabs, fabs(y[i]));
      maxabs = std::max(maxabs, fabs(z[i]));
    }
  }
  free(raw);
  // power-of-two scaling is exact in binary64
  double S = 1.0;
  if (maxabs > 0) {
    int e = 0;
    frexp(maxabs, &e);
    S = ldexp(1.0, e);            // 2^e >= maxabs
  }
  double invS = 1.0 / S;
  double eps2 = (0.01 * invS) * (0.01 * invS);
  for (int i = 0; i < n; i++) { x[i] *= invS; y[i] *= invS; z[i] *= invS; }
  for (int i = n; i < n + 8; i++) { x[i] = y[i] = z[i] = 0; }
  double t_read = cputime();
  DBG("N=%d mass=%.17g S=%.6g read %.3f\n", n, m0, S, t_read - tstart);

  int Pdef = 8;
  double theta = 0.5;
  int leafsize = 48;
  // parameter selection (overridable for development)
  const char *e;
  if ((e = getenv("GALAXY_FORCE_P"))) Pdef = atoi(e);
  if ((e = getenv("GALAXY_FORCE_THETA"))) theta = atof(e);
  if ((e = getenv("GALAXY_FORCE_LEAF"))) leafsize = atoi(e);
  double budget = 1e9;
  if ((e = getenv("GALAXY_CPU_LIMIT_S"))) budget = atof(e);
  DBG("config P=%d theta=%.4g leaf=%d budget=%.3f\n", Pdef, theta, leafsize, budget);

  int force_fmm = getenv("GALAXY_FORCE_FMM") != nullptr;
  int used_fmm = 0;
  if (!force_fmm && (mmax > 0 || n <= 2048)) {
    // tiny input: exact direct summation (also handles unequal masses)
    (void)budget;
    if (mmax == 0) {
      direct_sum(n, x, y, z, m0, eps2, ax, ay, az);
    } else {
      for (int i = 0; i < n; i++) {
        double axi = 0, ayi = 0, azi = 0;
        for (int j = 0; j < n; j++) {
          double dx = x[j] - x[i], dy = y[j] - y[i], dz = z[j] - z[i];
          double r2 = dx * dx + dy * dy + dz * dz + eps2;
          double inv = m0 / (r2 * sqrt(r2));
          axi += dx * inv; ayi += dy * inv; azi += dz * inv;
        }
        ax[i] = axi; ay[i] = ayi; az[i] = azi;
      }
    }
  } else {
    used_fmm = 1;
    switch (Pdef) {
      case 4:  run_fmm<4>(n, x, y, z, perm, m0, eps2, theta, leafsize, ax, ay, az); break;
      case 5:  run_fmm<5>(n, x, y, z, perm, m0, eps2, theta, leafsize, ax, ay, az); break;
      case 6:  run_fmm<6>(n, x, y, z, perm, m0, eps2, theta, leafsize, ax, ay, az); break;
      case 7:  run_fmm<7>(n, x, y, z, perm, m0, eps2, theta, leafsize, ax, ay, az); break;
      case 8:  run_fmm<8>(n, x, y, z, perm, m0, eps2, theta, leafsize, ax, ay, az); break;
      case 9:  run_fmm<9>(n, x, y, z, perm, m0, eps2, theta, leafsize, ax, ay, az); break;
      case 10: run_fmm<10>(n, x, y, z, perm, m0, eps2, theta, leafsize, ax, ay, az); break;
      case 11: run_fmm<11>(n, x, y, z, perm, m0, eps2, theta, leafsize, ax, ay, az); break;
      case 12: run_fmm<12>(n, x, y, z, perm, m0, eps2, theta, leafsize, ax, ay, az); break;
      default: run_fmm<8>(n, x, y, z, perm, m0, eps2, theta, leafsize, ax, ay, az); break;
    }
  }

  double invS2 = invS * invS;
  double *obuf = (double *)malloc((size_t)n * 3 * sizeof(double));
  for (int i = 0; i < n; i++) {
    int o = used_fmm ? (int)perm[i] : i;
    double vx = ax[i] * invS2, vy = ay[i] * invS2, vz = az[i] * invS2;
    if (!std::isfinite(vx) || !std::isfinite(vy) || !std::isfinite(vz)) {
      vx = 0; vy = 0; vz = 0;
    }
    obuf[3 * o + 0] = vx; obuf[3 * o + 1] = vy; obuf[3 * o + 2] = vz;
  }
  FILE *fout = fopen(outpath, "wb");
  if (!fout) { fprintf(stderr, "cannot open output\n"); return 1; }
  size_t wrote = fwrite(obuf, 1, (size_t)n * 24, fout);
  fclose(fout);
  free(obuf);
  DBG("total cpu %.3f wrote %zu bytes\n", cputime() - tstart, wrote);
  return wrote == (size_t)n * 24 ? 0 : 1;
}
