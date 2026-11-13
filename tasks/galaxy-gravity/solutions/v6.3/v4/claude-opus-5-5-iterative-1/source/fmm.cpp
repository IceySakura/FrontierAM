// Cartesian-Taylor FMM for Plummer-softened gravity (G=1, eps=0.01)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <immintrin.h>
#include <sys/resource.h>

typedef __m256d V;
#include "kern.h"

static const double EPS2 = 1e-4;
static double C_PAIR_NS = 2.4;
static double P2P_PAIR_NS = 2.2, P2P_I_NS = 8.0, P2P_CALL_NS = 30.0, P2P_SELF_NS = 1.4;
static double cpu_time() {
  struct rusage ru; getrusage(RUSAGE_SELF, &ru);
  return ru.ru_utime.tv_sec + 1e-6 * ru.ru_utime.tv_usec + ru.ru_stime.tv_sec + 1e-6 * ru.ru_stime.tv_usec;
}
static int VERB = 0;
#include <x86intrin.h>
static unsigned long long PR_M2L = 0, PR_P2P = 0, PR_SELF = 0, PR_PC = 0;

// ---------------- multi-index tables ----------------
struct MI {
  int p = 0, nc = 0, ncp = 0;
  std::vector<int> base;
  std::vector<int> kx, ky, kz, ord;
  std::vector<int> mpar, mdim; std::vector<double> minv;
  std::vector<int> m2m, l2l, l2p;
  std::vector<int> m2m_st, l2l_st; // start (in triples) per output index
  // contraction groups
  std::vector<int> gq, gout, gdoff, gmoff; // per group: q, output base, start in doff/moff
  std::vector<int> doff, moff;
  std::vector<double> sgn;
  der_fn der = nullptr;
  int idx(int a, int b, int c) const { return base[a * (p + 1) + b] + c; }
  void init(int p_) {
    p = p_; base.assign((p + 1) * (p + 1), -1); int c = 0;
    kx.clear(); ky.clear(); kz.clear(); ord.clear(); m2m.clear(); l2l.clear(); l2p.clear();
    for (int a = 0; a <= p; a++) for (int b = 0; b <= p - a; b++) {
      base[a * (p + 1) + b] = c;
      for (int z = 0; z <= p - a - b; z++) { kx.push_back(a); ky.push_back(b); kz.push_back(z); ord.push_back(a + b + z); c++; }
    }
    nc = c; ncp = (nc + 3) & ~3;
    mpar.assign(nc, -1); mdim.assign(nc, -1); minv.assign(nc, 0);
    for (int i = 1; i < nc; i++) {
      int a = kx[i], b = ky[i], z = kz[i], d, md;
      if (z > 0) { d = 2; md = z; } else if (b > 0) { d = 1; md = b; } else { d = 0; md = a; }
      mpar[i] = idx(a - (d == 0), b - (d == 1), z - (d == 2)); mdim[i] = d; minv[i] = 1.0 / md;
    }
    for (int k = 0; k < nc; k++) for (int j = 0; j < nc; j++)
      if (kx[j] <= kx[k] && ky[j] <= ky[k] && kz[j] <= kz[k]) { m2m.push_back(k); m2m.push_back(j); m2m.push_back(idx(kx[k] - kx[j], ky[k] - ky[j], kz[k] - kz[j])); }
    for (int n = 0; n < nc; n++) for (int k = 0; k < nc; k++)
      if (ord[n] + ord[k] <= p) { l2l.push_back(n); l2l.push_back(k); l2l.push_back(idx(kx[n] + kx[k], ky[n] + ky[k], kz[n] + kz[k])); }
    m2m_st.assign(nc + 1, 0); l2l_st.assign(nc + 1, 0);
    for (size_t q = 0; q < m2m.size() / 3; q++) m2m_st[m2m[3 * q] + 1]++;
    for (size_t q = 0; q < l2l.size() / 3; q++) l2l_st[l2l[3 * q] + 1]++;
    for (int k = 0; k < nc; k++) { m2m_st[k + 1] += m2m_st[k]; l2l_st[k + 1] += l2l_st[k]; }
    for (int m = 0; m < nc; m++) if (ord[m] <= p - 1) {
      l2p.push_back(m); l2p.push_back(idx(kx[m] + 1, ky[m], kz[m])); l2p.push_back(idx(kx[m], ky[m] + 1, kz[m])); l2p.push_back(idx(kx[m], ky[m], kz[m] + 1));
    }
    gq.clear(); gout.clear(); gdoff.clear(); gmoff.clear(); doff.clear(); moff.clear();
    for (int nx = 0; nx <= p; nx++) for (int ny = 0; ny <= p - nx; ny++) {
      int q = p - nx - ny;
      gq.push_back(q); gout.push_back(idx(nx, ny, 0)); gdoff.push_back((int)doff.size()); gmoff.push_back((int)moff.size());
      for (int a = 0; a <= q; a++) for (int b = 0; b <= q - a; b++) { doff.push_back(idx(nx + a, ny + b, 0)); moff.push_back(idx(a, b, 0)); }
    }
    sgn.assign(ncp, 0.0); for (int k = 0; k < nc; k++) sgn[k] = (ord[k] & 1) ? -1.0 : 1.0;
    der = get_derivs(p);
    if (!der) { fprintf(stderr, "no kernel for p=%d\n", p); exit(1); }
  }
  inline void monomials(double sx, double sy, double sz, double* mo) const {
    mo[0] = 1.0; double s[3] = {sx, sy, sz};
    for (int i = 1; i < nc; i++) mo[i] = mo[mpar[i]] * s[mdim[i]] * minv[i];
  }
};
static MI* TT; // current table

// ---------------- particles & tree ----------------
static int N;
static double *px, *py, *pz, *pm;
static double *ax_, *ay_, *az_;
static int* pid;

struct Node {
  double cx, cy, cz, r, m;
  double gx, gy, gz, hs;
  int b, e, child, nch;
};
static std::vector<Node> nodes;
static std::vector<double> tolv; // absolute force tolerance per node
static std::vector<double> aminv; static double TOLR = 0;
static int NLEAF = 32;
static double KAPPA = 0.3;
static double F0 = 0.05;
static int AUTOMODE = 1;
static int MULTICEN = 1; static double CENFRAC = 0.05;
static double TOL0 = 1e-4;
static int FLOATMODE = 0; static double C_PAIR_FL = 0.6;
static double UD_B_NS = 1.6, UD_Q_NS = 5.0;
static double DRYCAL = 1.0;
static double TRCAL = 1.0, UDCAL = 1.6, MEMCAP = 1.0e9, RHO0 = 0.6, RHOMIN = 0.2;

static void build_tree() {
  double mn[3] = {1e300, 1e300, 1e300}, mx[3] = {-1e300, -1e300, -1e300};
  for (int i = 0; i < N; i++) {
    mn[0] = std::min(mn[0], px[i]); mx[0] = std::max(mx[0], px[i]);
    mn[1] = std::min(mn[1], py[i]); mx[1] = std::max(mx[1], py[i]);
    mn[2] = std::min(mn[2], pz[i]); mx[2] = std::max(mx[2], pz[i]);
  }
  double hs = 0.5 * std::max(mx[0] - mn[0], std::max(mx[1] - mn[1], mx[2] - mn[2])) * (1 + 1e-9) + 1e-12;
  Node root; root.gx = 0.5 * (mn[0] + mx[0]); root.gy = 0.5 * (mn[1] + mx[1]); root.gz = 0.5 * (mn[2] + mx[2]);
  root.hs = hs; root.b = 0; root.e = N; root.child = -1; root.nch = 0;
  nodes.clear(); nodes.reserve(N / 4 + 1000); nodes.push_back(root);
  std::vector<double> tx(N), ty(N), tz(N), tm(N); std::vector<int> tid(N); std::vector<unsigned char> oct(N);
  std::vector<int> stack; stack.push_back(0);
  double minhs = hs * 1e-12;
  double cm[3] = {0, 0, 0}, mt = 0;
  for (int i = 0; i < N; i++) { cm[0] += pm[i] * px[i]; cm[1] += pm[i] * py[i]; cm[2] += pm[i] * pz[i]; mt += pm[i]; }
  if (mt > 0) { cm[0] /= mt; cm[1] /= mt; cm[2] /= mt; }
  double k2 = KAPPA * KAPPA / 3.0;
  // mass centers: global COM + COMs of coarse-grid cells holding >= CENFRAC of the mass
  std::vector<double> cen; cen.push_back(cm[0]); cen.push_back(cm[1]); cen.push_back(cm[2]);
  if (MULTICEN) {
    const int G = 16; std::vector<double> gm(G * G * G, 0), gx(G * G * G, 0), gy(G * G * G, 0), gz(G * G * G, 0);
    double lo[3] = {root.gx - hs, root.gy - hs, root.gz - hs}, inv = G / (2 * hs);
    for (int i = 0; i < N; i++) {
      int ix = std::min(G - 1, std::max(0, (int)((px[i] - lo[0]) * inv))), iy = std::min(G - 1, std::max(0, (int)((py[i] - lo[1]) * inv))), iz = std::min(G - 1, std::max(0, (int)((pz[i] - lo[2]) * inv)));
      int c = (ix * G + iy) * G + iz; gm[c] += pm[i]; gx[c] += pm[i] * px[i]; gy[c] += pm[i] * py[i]; gz[c] += pm[i] * pz[i];
    }
    for (int c = 0; c < G * G * G; c++) if (gm[c] >= CENFRAC * mt && gm[c] > 0) {
      double qx = gx[c] / gm[c], qy = gy[c] / gm[c], qz = gz[c] / gm[c];
      cen.push_back(qx); cen.push_back(qy); cen.push_back(qz);
    }
  }
  const int ncen = (int)cen.size() / 3;
  while (!stack.empty()) {
    int ni = stack.back(); stack.pop_back();
    Node nd = nodes[ni];
    int cnt = nd.e - nd.b;
    if (nd.hs < minhs || cnt <= 1) continue;
    if (cnt <= NLEAF) {
      double d2 = 1e300;
      for (int c = 0; c < ncen; c++) { double dx = nd.gx - cen[3 * c], dy = nd.gy - cen[3 * c + 1], dz = nd.gz - cen[3 * c + 2]; d2 = std::min(d2, dx * dx + dy * dy + dz * dz); }
      if (nd.hs * nd.hs <= k2 * d2) continue;
    }
    int c8[8] = {0};
    for (int i = nd.b; i < nd.e; i++) {
      int o = (px[i] >= nd.gx) | ((py[i] >= nd.gy) << 1) | ((pz[i] >= nd.gz) << 2);
      oct[i] = (unsigned char)o; c8[o]++;
    }
    int st[8], s = nd.b;
    for (int o = 0; o < 8; o++) { st[o] = s; s += c8[o]; }
    int pos[8]; memcpy(pos, st, sizeof(pos));
    for (int i = nd.b; i < nd.e; i++) {
      int d = pos[oct[i]]++;
      tx[d] = px[i]; ty[d] = py[i]; tz[d] = pz[i]; tm[d] = pm[i]; tid[d] = pid[i];
    }
    memcpy(px + nd.b, &tx[nd.b], cnt * 8); memcpy(py + nd.b, &ty[nd.b], cnt * 8); memcpy(pz + nd.b, &tz[nd.b], cnt * 8);
    memcpy(pm + nd.b, &tm[nd.b], cnt * 8); memcpy(pid + nd.b, &tid[nd.b], cnt * 4);
    int first = (int)nodes.size(), nch = 0;
    double h2 = nd.hs * 0.5;
    for (int o = 0; o < 8; o++) if (c8[o]) {
      Node c; c.hs = h2;
      c.gx = nd.gx + ((o & 1) ? h2 : -h2); c.gy = nd.gy + ((o & 2) ? h2 : -h2); c.gz = nd.gz + ((o & 4) ? h2 : -h2);
      c.b = st[o]; c.e = st[o] + c8[o]; c.child = -1; c.nch = 0;
      nodes.push_back(c); nch++;
    }
    nodes[ni].child = first; nodes[ni].nch = nch;
    for (int c = first + nch - 1; c >= first; c--) stack.push_back(c);
  }
  for (int ni = (int)nodes.size() - 1; ni >= 0; ni--) {
    Node& nd = nodes[ni];
    double m = 0, sx = 0, sy = 0, sz = 0;
    if (nd.child < 0) {
      for (int i = nd.b; i < nd.e; i++) { m += pm[i]; sx += pm[i] * px[i]; sy += pm[i] * py[i]; sz += pm[i] * pz[i]; }
    } else {
      for (int c = nd.child; c < nd.child + nd.nch; c++) { const Node& ch = nodes[c]; m += ch.m; sx += ch.m * ch.cx; sy += ch.m * ch.cy; sz += ch.m * ch.cz; }
    }
    nd.m = m;
    if (m > 0) { nd.cx = sx / m; nd.cy = sy / m; nd.cz = sz / m; } else { nd.cx = nd.gx; nd.cy = nd.gy; nd.cz = nd.gz; }
    double r2 = 0;
    for (int i = nd.b; i < nd.e; i++) {
      double dx = px[i] - nd.cx, dy = py[i] - nd.cy, dz = pz[i] - nd.cz;
      r2 = std::max(r2, dx * dx + dy * dy + dz * dz);
    }
    nd.r = sqrt(r2) * (1 + 1e-12);
  }
}

// ---------------- expansions ----------------
static double *Mm, *Lm; // nodes * ncp
static size_t NCP;

static inline void transpose4(V& r0, V& r1, V& r2, V& r3) {
  V t0 = _mm256_unpacklo_pd(r0, r1), t1 = _mm256_unpackhi_pd(r0, r1), t2 = _mm256_unpacklo_pd(r2, r3), t3 = _mm256_unpackhi_pd(r2, r3);
  r0 = _mm256_permute2f128_pd(t0, t2, 0x20); r1 = _mm256_permute2f128_pd(t1, t3, 0x20);
  r2 = _mm256_permute2f128_pd(t0, t2, 0x31); r3 = _mm256_permute2f128_pd(t1, t3, 0x31);
}
static inline double hsum(V v) { alignas(32) double t[4]; _mm256_store_pd(t, v); return (t[0] + t[1]) + (t[2] + t[3]); }
static inline void mono4(const MI& T, V sx, V sy, V sz, V* mo) {
  V s[3] = {sx, sy, sz};
  mo[0] = _mm256_set1_pd(1.0);
  const int* mp = T.mpar.data(); const int* md = T.mdim.data(); const double* mi = T.minv.data();
  for (int i = 1; i < T.nc; i++) mo[i] = _mm256_mul_pd(_mm256_mul_pd(mo[mp[i]], s[md[i]]), _mm256_set1_pd(mi[i]));
}
static V *UVmo, *UVacc, *UVt;
static double* ZEROROW;

static void upward() {
  const MI& T = *TT; const int nc = T.nc, ncp = T.ncp;
  V* mo = UVmo; V* acc = UVacc; V* McT = UVt;
  const int* t3 = T.m2m.data(); const size_t nt = T.m2m.size() / 3;
  for (int ni = (int)nodes.size() - 1; ni >= 0; ni--) {
    Node& nd = nodes[ni];
    double* M = Mm + (size_t)ni * NCP;
    V cx = _mm256_set1_pd(nd.cx), cy = _mm256_set1_pd(nd.cy), cz = _mm256_set1_pd(nd.cz);
    if (nd.child < 0) {
      for (int k = 0; k < nc; k++) acc[k] = _mm256_setzero_pd();
      for (int i0 = nd.b; i0 < nd.e; i0 += 4) {
        int cnt = std::min(4, nd.e - i0);
        alignas(32) double mm[4] = {0, 0, 0, 0};
        for (int l = 0; l < cnt; l++) mm[l] = pm[i0 + l];
        V mv = _mm256_load_pd(mm);
        mono4(T, _mm256_sub_pd(_mm256_loadu_pd(px + i0), cx), _mm256_sub_pd(_mm256_loadu_pd(py + i0), cy), _mm256_sub_pd(_mm256_loadu_pd(pz + i0), cz), mo);
        for (int k = 0; k < nc; k++) acc[k] = _mm256_fmadd_pd(mv, mo[k], acc[k]);
      }
      for (int k = 0; k < nc; k++) M[k] = hsum(acc[k]);
      for (int k = nc; k < ncp; k++) M[k] = 0;
    } else {
      for (int k = 0; k < ncp; k++) M[k] = 0;
      for (int c0 = nd.child; c0 < nd.child + nd.nch; c0 += 4) {
        int cnt = std::min(4, nd.child + nd.nch - c0);
        alignas(32) double tx[4] = {0, 0, 0, 0}, ty[4] = {0, 0, 0, 0}, tz[4] = {0, 0, 0, 0};
        const double* rows[4];
        for (int l = 0; l < 4; l++) {
          if (l < cnt) { const Node& ch = nodes[c0 + l]; tx[l] = ch.cx - nd.cx; ty[l] = ch.cy - nd.cy; tz[l] = ch.cz - nd.cz; rows[l] = Mm + (size_t)(c0 + l) * NCP; }
          else rows[l] = ZEROROW;
        }
        mono4(T, _mm256_load_pd(tx), _mm256_load_pd(ty), _mm256_load_pd(tz), mo);
        for (int k = 0; k < ncp; k += 4) {
          V r0 = _mm256_load_pd(rows[0] + k), r1 = _mm256_load_pd(rows[1] + k), r2 = _mm256_load_pd(rows[2] + k), r3 = _mm256_load_pd(rows[3] + k);
          transpose4(r0, r1, r2, r3); McT[k] = r0; McT[k + 1] = r1; McT[k + 2] = r2; McT[k + 3] = r3;
        }
        const int* st = T.m2m_st.data();
        for (int k = 0; k < nc; k++) {
          V a0 = _mm256_setzero_pd(), a1 = a0;
          int q = st[k], qe = st[k + 1];
          for (; q + 1 < qe; q += 2) { a0 = _mm256_fmadd_pd(McT[t3[3 * q + 1]], mo[t3[3 * q + 2]], a0); a1 = _mm256_fmadd_pd(McT[t3[3 * q + 4]], mo[t3[3 * q + 5]], a1); }
          if (q < qe) a0 = _mm256_fmadd_pd(McT[t3[3 * q + 1]], mo[t3[3 * q + 2]], a0);
          M[k] += hsum(_mm256_add_pd(a0, a1));
        }
      }
    }
  }
}

static void downward() {
  const MI& T = *TT; const int nc = T.nc, ncp = T.ncp;
  V* mo = UVmo; V* acc = UVacc;
  const int* t3 = T.l2l.data(); const size_t nt = T.l2l.size() / 3;
  const int* tp = T.l2p.data(); const size_t np4 = T.l2p.size() / 4;
  for (int ni = 0; ni < (int)nodes.size(); ni++) {
    Node& nd = nodes[ni];
    const double* L = Lm + (size_t)ni * NCP;
    if (nd.child >= 0) {
      for (int c0 = nd.child; c0 < nd.child + nd.nch; c0 += 4) {
        int cnt = std::min(4, nd.child + nd.nch - c0);
        alignas(32) double tx[4] = {0, 0, 0, 0}, ty[4] = {0, 0, 0, 0}, tz[4] = {0, 0, 0, 0};
        for (int l = 0; l < cnt; l++) { const Node& ch = nodes[c0 + l]; tx[l] = ch.cx - nd.cx; ty[l] = ch.cy - nd.cy; tz[l] = ch.cz - nd.cz; }
        mono4(T, _mm256_load_pd(tx), _mm256_load_pd(ty), _mm256_load_pd(tz), mo);
        const int* st = T.l2l_st.data();
        for (int n = 0; n < nc; n++) {
          V a0 = _mm256_setzero_pd(), a1 = a0;
          int q = st[n], qe = st[n + 1];
          for (; q + 1 < qe; q += 2) { a0 = _mm256_fmadd_pd(_mm256_set1_pd(L[t3[3 * q + 2]]), mo[t3[3 * q + 1]], a0); a1 = _mm256_fmadd_pd(_mm256_set1_pd(L[t3[3 * q + 5]]), mo[t3[3 * q + 4]], a1); }
          if (q < qe) a0 = _mm256_fmadd_pd(_mm256_set1_pd(L[t3[3 * q + 2]]), mo[t3[3 * q + 1]], a0);
          acc[n] = _mm256_add_pd(a0, a1);
        }
        for (int k = nc; k < ncp; k++) acc[k] = _mm256_setzero_pd();
        for (int k = 0; k < ncp; k += 4) {
          V r0 = acc[k], r1 = acc[k + 1], r2 = acc[k + 2], r3 = acc[k + 3];
          transpose4(r0, r1, r2, r3);
          V rr[4] = {r0, r1, r2, r3};
          for (int l = 0; l < cnt; l++) { double* Lc = Lm + (size_t)(c0 + l) * NCP; _mm256_store_pd(Lc + k, _mm256_add_pd(_mm256_load_pd(Lc + k), rr[l])); }
        }
      }
    } else {
      V cx = _mm256_set1_pd(nd.cx), cy = _mm256_set1_pd(nd.cy), cz = _mm256_set1_pd(nd.cz);
      for (int i0 = nd.b; i0 < nd.e; i0 += 4) {
        int cnt = std::min(4, nd.e - i0);
        mono4(T, _mm256_sub_pd(_mm256_loadu_pd(px + i0), cx), _mm256_sub_pd(_mm256_loadu_pd(py + i0), cy), _mm256_sub_pd(_mm256_loadu_pd(pz + i0), cz), mo);
        V gx = _mm256_setzero_pd(), gy = gx, gz = gx;
        for (size_t q = 0; q < np4; q++) {
          V w = mo[tp[4 * q]];
          gx = _mm256_fmadd_pd(w, _mm256_set1_pd(L[tp[4 * q + 1]]), gx);
          gy = _mm256_fmadd_pd(w, _mm256_set1_pd(L[tp[4 * q + 2]]), gy);
          gz = _mm256_fmadd_pd(w, _mm256_set1_pd(L[tp[4 * q + 3]]), gz);
        }
        alignas(32) double sx[4], sy[4], sz[4];
        _mm256_store_pd(sx, gx); _mm256_store_pd(sy, gy); _mm256_store_pd(sz, gz);
        for (int l = 0; l < cnt; l++) { ax_[i0 + l] += sx[l]; ay_[i0 + l] += sy[l]; az_[i0 + l] += sz[l]; }
      }
    }
  }
}

// ---------------- SIMD helpers ----------------
static V *Xb, *Yb, *Dv, *MAv, *MBv, *LAv, *LBv;
static long n_m2l = 0, n_p2p = 0, n_pc = 0;
static double n_pairs = 0;

static inline void derivs4(V rx, V ry, V rz) {
  V s2 = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(rx, rx), _mm256_mul_pd(ry, ry)), _mm256_fmadd_pd(rz, rz, _mm256_set1_pd(EPS2)));
  V inv = _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(s2));
  TT->der(rx, ry, rz, inv, _mm256_mul_pd(inv, inv), Xb, Yb, Dv);
}
static void m2l_batch(int na, const int* A, const int* B) {
  const MI& T = *TT; const int ncp = T.ncp;
  alignas(32) double rx[4], ry[4], rz[4];
  int Aa[4], Bb[4];
  for (int l = 0; l < 4; l++) {
    int ll = l < na ? l : 0; Aa[l] = A[ll]; Bb[l] = B[ll];
    const Node &a = nodes[Aa[l]], &b = nodes[Bb[l]]; rx[l] = a.cx - b.cx; ry[l] = a.cy - b.cy; rz[l] = a.cz - b.cz;
  }
  derivs4(_mm256_load_pd(rx), _mm256_load_pd(ry), _mm256_load_pd(rz));
  const double *a0 = Mm + (size_t)Aa[0] * NCP, *a1 = Mm + (size_t)Aa[1] * NCP, *a2 = Mm + (size_t)Aa[2] * NCP, *a3 = Mm + (size_t)Aa[3] * NCP;
  const double *b0 = Mm + (size_t)Bb[0] * NCP, *b1 = Mm + (size_t)Bb[1] * NCP, *b2 = Mm + (size_t)Bb[2] * NCP, *b3 = Mm + (size_t)Bb[3] * NCP;
  for (int k = 0; k < ncp; k += 4) {
    V r0 = _mm256_load_pd(a0 + k), r1 = _mm256_load_pd(a1 + k), r2 = _mm256_load_pd(a2 + k), r3 = _mm256_load_pd(a3 + k);
    transpose4(r0, r1, r2, r3);
    MAv[k] = r0; MAv[k + 1] = r1; MAv[k + 2] = r2; MAv[k + 3] = r3;
    r0 = _mm256_load_pd(b0 + k); r1 = _mm256_load_pd(b1 + k); r2 = _mm256_load_pd(b2 + k); r3 = _mm256_load_pd(b3 + k);
    transpose4(r0, r1, r2, r3);
    MBv[k] = _mm256_mul_pd(r0, _mm256_set1_pd(T.sgn[k])); MBv[k + 1] = _mm256_mul_pd(r1, _mm256_set1_pd(T.sgn[k + 1]));
    MBv[k + 2] = _mm256_mul_pd(r2, _mm256_set1_pd(T.sgn[k + 2])); MBv[k + 3] = _mm256_mul_pd(r3, _mm256_set1_pd(T.sgn[k + 3]));
  }
  const int ng = (int)T.gq.size();
  for (int g = 0; g < ng; g++) {
    blk_fn f = BLK[T.gq[g]];
    const int* dof = T.doff.data() + T.gdoff[g]; const int* mof = T.moff.data() + T.gmoff[g];
    f(Dv, dof, MBv, mof, LAv + T.gout[g]);
    f(Dv, dof, MAv, mof, LBv + T.gout[g]);
  }
  for (int k = T.nc; k < ncp; k++) { LAv[k] = _mm256_setzero_pd(); LBv[k] = _mm256_setzero_pd(); }
  double* La[4]; double* Lb[4];
  for (int l = 0; l < 4; l++) { La[l] = Lm + (size_t)Aa[l] * NCP; Lb[l] = Lm + (size_t)Bb[l] * NCP; }
  // when na<4, padded lanes duplicate lane 0 -> skip their accumulation by zeroing
  for (int k = 0; k < ncp; k += 4) {
    V r0 = LAv[k], r1 = LAv[k + 1], r2 = LAv[k + 2], r3 = LAv[k + 3];
    transpose4(r0, r1, r2, r3);
    V rr[4] = {r0, r1, r2, r3};
    for (int l = 0; l < na; l++) _mm256_store_pd(La[l] + k, _mm256_add_pd(_mm256_load_pd(La[l] + k), rr[l]));
    V s = _mm256_load_pd(T.sgn.data() + k);
    r0 = _mm256_mul_pd(LBv[k], _mm256_set1_pd(T.sgn[k])); r1 = _mm256_mul_pd(LBv[k + 1], _mm256_set1_pd(T.sgn[k + 1]));
    r2 = _mm256_mul_pd(LBv[k + 2], _mm256_set1_pd(T.sgn[k + 2])); r3 = _mm256_mul_pd(LBv[k + 3], _mm256_set1_pd(T.sgn[k + 3]));
    (void)s;
    transpose4(r0, r1, r2, r3);
    V qq[4] = {r0, r1, r2, r3};
    for (int l = 0; l < na; l++) _mm256_store_pd(Lb[l] + k, _mm256_add_pd(_mm256_load_pd(Lb[l] + k), qq[l]));
  }
}

static int m2l_na = 0, m2l_A[4], m2l_B[4];
static int PREF = 0;
static inline void m2l_push(int a, int b) {
  if (PREF) {
    const char* pa = (const char*)(Mm + (size_t)a * NCP); const char* pb = (const char*)(Mm + (size_t)b * NCP);
    const char* la = (const char*)(Lm + (size_t)a * NCP); const char* lb = (const char*)(Lm + (size_t)b * NCP);
    for (int o = 0; o < 256; o += 64) { _mm_prefetch(pa + o, _MM_HINT_T0); _mm_prefetch(pb + o, _MM_HINT_T0); _mm_prefetch(la + o, _MM_HINT_T0); _mm_prefetch(lb + o, _MM_HINT_T0); }
  }
  m2l_A[m2l_na] = a; m2l_B[m2l_na] = b; m2l_na++; n_m2l++;
  if (m2l_na == 4) { unsigned long long t0 = __rdtsc(); m2l_batch(4, m2l_A, m2l_B); PR_M2L += __rdtsc() - t0; m2l_na = 0; }
}
static void m2l_flush() { if (m2l_na) m2l_batch(m2l_na, m2l_A, m2l_B); m2l_na = 0; }

// particle-cell: M2P(cell c -> particles of leaf l) and P2L(particles of l -> cell c)
static void pc_interact(int lf, int c) {
  const MI& T = *TT; const int nc = T.nc;
  const Node& Lf = nodes[lf]; const Node& C = nodes[c];
  const double* M = Mm + (size_t)c * NCP; double* L = Lm + (size_t)c * NCP;
  n_pc += Lf.e - Lf.b;
  const int* t = T.l2p.data(); size_t nt = T.l2p.size() / 4;
  V cx = _mm256_set1_pd(C.cx), cy = _mm256_set1_pd(C.cy), cz = _mm256_set1_pd(C.cz);
  for (int k = 0; k < nc; k++) LAv[k] = _mm256_setzero_pd();
  for (int i0 = Lf.b; i0 < Lf.e; i0 += 4) {
    int cnt = std::min(4, Lf.e - i0);
    V rx = _mm256_sub_pd(_mm256_loadu_pd(px + i0), cx), ry = _mm256_sub_pd(_mm256_loadu_pd(py + i0), cy), rz = _mm256_sub_pd(_mm256_loadu_pd(pz + i0), cz);
    derivs4(rx, ry, rz);
    // M2P: a_d = sum_k sgn_k M_k D_{k+e_d}
    V gx = _mm256_setzero_pd(), gy = gx, gz = gx;
    for (size_t q = 0; q < nt; q++) {
      int k = t[4 * q]; V w = _mm256_set1_pd(T.sgn[k] * M[k]);
      gx = _mm256_fmadd_pd(w, Dv[t[4 * q + 1]], gx); gy = _mm256_fmadd_pd(w, Dv[t[4 * q + 2]], gy); gz = _mm256_fmadd_pd(w, Dv[t[4 * q + 3]], gz);
    }
    alignas(32) double sx[4], sy[4], sz[4];
    _mm256_store_pd(sx, gx); _mm256_store_pd(sy, gy); _mm256_store_pd(sz, gz);
    for (int l = 0; l < cnt; l++) { ax_[i0 + l] += sx[l]; ay_[i0 + l] += sy[l]; az_[i0 + l] += sz[l]; }
    // P2L: L_n += m_j D_n(z_c - x_j) = m_j (-1)^{|n|} D_n(x_j - z_c)
    alignas(32) double mm[4] = {0, 0, 0, 0};
    for (int l = 0; l < cnt; l++) mm[l] = pm[i0 + l];
    V mv = _mm256_load_pd(mm);
    for (int k = 0; k < nc; k++) LAv[k] = _mm256_fmadd_pd(mv, Dv[k], LAv[k]);
  }
  for (int k = 0; k < nc; k++) {
    alignas(32) double s[4]; _mm256_store_pd(s, LAv[k]);
    L[k] += T.sgn[k] * ((s[0] + s[1]) + (s[2] + s[3]));
  }
}

// ---------------- P2P ----------------
static inline V rsqrt_nr(V x) { return _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(x)); }
static void p2p_mutual(int ab, int ae, int bb, int be) {
  n_pairs += (double)(ae - ab) * (be - bb); n_p2p++;
  const V e2 = _mm256_set1_pd(EPS2);
  int nb = be - bb;
  int nfull = nb & ~3;
  V msk = _mm256_castsi256_pd(_mm256_cmpgt_epi64(_mm256_set1_epi64x(nb - nfull), _mm256_set_epi64x(3, 2, 1, 0)));
  int i = ab;
  for (; i + 1 < ae; i += 2) {
    V xi = _mm256_set1_pd(px[i]), yi = _mm256_set1_pd(py[i]), zi = _mm256_set1_pd(pz[i]), mi = _mm256_set1_pd(pm[i]);
    V xk = _mm256_set1_pd(px[i + 1]), yk = _mm256_set1_pd(py[i + 1]), zk = _mm256_set1_pd(pz[i + 1]), mk = _mm256_set1_pd(pm[i + 1]);
    V gx = _mm256_setzero_pd(), gy = gx, gz = gx, hx = gx, hy = gx, hz = gx;
    for (int j0 = 0; j0 < nb; j0 += 4) {
      int j = bb + j0;
      V xj = _mm256_loadu_pd(px + j), yj = _mm256_loadu_pd(py + j), zj = _mm256_loadu_pd(pz + j), mj = _mm256_loadu_pd(pm + j);
      V dx = _mm256_sub_pd(xj, xi), dy = _mm256_sub_pd(yj, yi), dz = _mm256_sub_pd(zj, zi);
      V ex = _mm256_sub_pd(xj, xk), ey = _mm256_sub_pd(yj, yk), ez = _mm256_sub_pd(zj, zk);
      V r2 = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_fmadd_pd(dz, dz, e2)));
      V s2 = _mm256_fmadd_pd(ex, ex, _mm256_fmadd_pd(ey, ey, _mm256_fmadd_pd(ez, ez, e2)));
      V ri = rsqrt_nr(r2), si = rsqrt_nr(s2);
      V ri3 = _mm256_mul_pd(ri, _mm256_mul_pd(ri, ri)), si3 = _mm256_mul_pd(si, _mm256_mul_pd(si, si));
      if (j0 >= nfull) { ri3 = _mm256_and_pd(ri3, msk); si3 = _mm256_and_pd(si3, msk); }
      V wj = _mm256_mul_pd(mj, ri3), uj = _mm256_mul_pd(mj, si3);
      gx = _mm256_fmadd_pd(wj, dx, gx); gy = _mm256_fmadd_pd(wj, dy, gy); gz = _mm256_fmadd_pd(wj, dz, gz);
      hx = _mm256_fmadd_pd(uj, ex, hx); hy = _mm256_fmadd_pd(uj, ey, hy); hz = _mm256_fmadd_pd(uj, ez, hz);
      V wi = _mm256_mul_pd(mi, ri3), wk = _mm256_mul_pd(mk, si3);
      _mm256_storeu_pd(ax_ + j, _mm256_fnmadd_pd(wk, ex, _mm256_fnmadd_pd(wi, dx, _mm256_loadu_pd(ax_ + j))));
      _mm256_storeu_pd(ay_ + j, _mm256_fnmadd_pd(wk, ey, _mm256_fnmadd_pd(wi, dy, _mm256_loadu_pd(ay_ + j))));
      _mm256_storeu_pd(az_ + j, _mm256_fnmadd_pd(wk, ez, _mm256_fnmadd_pd(wi, dz, _mm256_loadu_pd(az_ + j))));
    }
    ax_[i] += hsum(gx); ay_[i] += hsum(gy); az_[i] += hsum(gz);
    ax_[i + 1] += hsum(hx); ay_[i + 1] += hsum(hy); az_[i + 1] += hsum(hz);
  }
  for (; i < ae; i++) {
    V xi = _mm256_set1_pd(px[i]), yi = _mm256_set1_pd(py[i]), zi = _mm256_set1_pd(pz[i]), mi = _mm256_set1_pd(pm[i]);
    V gx = _mm256_setzero_pd(), gy = gx, gz = gx;
    for (int j0 = 0; j0 < nb; j0 += 4) {
      int j = bb + j0;
      V dx = _mm256_sub_pd(_mm256_loadu_pd(px + j), xi), dy = _mm256_sub_pd(_mm256_loadu_pd(py + j), yi), dz = _mm256_sub_pd(_mm256_loadu_pd(pz + j), zi);
      V r2 = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_fmadd_pd(dz, dz, e2)));
      V ri = rsqrt_nr(r2);
      V ri3 = _mm256_mul_pd(ri, _mm256_mul_pd(ri, ri));
      if (j0 >= nfull) ri3 = _mm256_and_pd(ri3, msk);
      V wj = _mm256_mul_pd(_mm256_loadu_pd(pm + j), ri3);
      gx = _mm256_fmadd_pd(wj, dx, gx); gy = _mm256_fmadd_pd(wj, dy, gy); gz = _mm256_fmadd_pd(wj, dz, gz);
      V wi = _mm256_mul_pd(mi, ri3);
      _mm256_storeu_pd(ax_ + j, _mm256_fnmadd_pd(wi, dx, _mm256_loadu_pd(ax_ + j)));
      _mm256_storeu_pd(ay_ + j, _mm256_fnmadd_pd(wi, dy, _mm256_loadu_pd(ay_ + j)));
      _mm256_storeu_pd(az_ + j, _mm256_fnmadd_pd(wi, dz, _mm256_loadu_pd(az_ + j)));
    }
    ax_[i] += hsum(gx); ay_[i] += hsum(gy); az_[i] += hsum(gz);
  }
}
static void p2p_self(int ab, int ae) {
  n_pairs += 0.5 * (double)(ae - ab) * (ae - ab);
  const V e2 = _mm256_set1_pd(EPS2);
  int nb = ae - ab, nfull = nb & ~3;
  __m256i msk = _mm256_cmpgt_epi64(_mm256_set1_epi64x(nb - nfull), _mm256_set_epi64x(3, 2, 1, 0));
  for (int i = ab; i < ae; i++) {
    V xi = _mm256_set1_pd(px[i]), yi = _mm256_set1_pd(py[i]), zi = _mm256_set1_pd(pz[i]);
    V gx = _mm256_setzero_pd(), gy = gx, gz = gx;
    for (int j0 = 0; j0 < nb; j0 += 4) {
      int j = ab + j0;
      V dx = _mm256_sub_pd(_mm256_loadu_pd(px + j), xi), dy = _mm256_sub_pd(_mm256_loadu_pd(py + j), yi), dz = _mm256_sub_pd(_mm256_loadu_pd(pz + j), zi);
      V r2 = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_fmadd_pd(dz, dz, e2)));
      V ri = rsqrt_nr(r2);
      V ri3 = _mm256_mul_pd(ri, _mm256_mul_pd(ri, ri));
      if (j0 >= nfull) ri3 = _mm256_and_pd(ri3, _mm256_castsi256_pd(msk));
      V wj = _mm256_mul_pd(_mm256_loadu_pd(pm + j), ri3);
      gx = _mm256_fmadd_pd(wj, dx, gx); gy = _mm256_fmadd_pd(wj, dy, gy); gz = _mm256_fmadd_pd(wj, dz, gz);
    }
    alignas(32) double sx[4], sy[4], sz[4];
    _mm256_store_pd(sx, gx); _mm256_store_pd(sy, gy); _mm256_store_pd(sz, gz);
    ax_[i] += (sx[0] + sx[1]) + (sx[2] + sx[3]);
    ay_[i] += (sy[0] + sy[1]) + (sy[2] + sy[3]);
    az_[i] += (sz[0] + sz[1]) + (sz[2] + sz[3]);
  }
}

// single-precision P2P for low-accuracy regimes: coordinates relative to a per-call origin
static float *fbx, *fby, *fbz, *fbm, *fbax, *fbay, *fbaz;
static bool FLOATP2P = false;
static inline float hsum8f(__m256 v) { alignas(32) float t[8]; _mm256_store_ps(t, v); return ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7])); }
static void p2p_fl(int ab, int ae, int bb, int be, bool self) {
  const double ox = px[ab], oy = py[ab], oz = pz[ab];
  const int nb = be - bb, nb8 = (nb + 7) & ~7;
  for (int j = 0; j < nb; j++) { fbx[j] = (float)(px[bb + j] - ox); fby[j] = (float)(py[bb + j] - oy); fbz[j] = (float)(pz[bb + j] - oz); fbm[j] = (float)pm[bb + j]; fbax[j] = fbay[j] = fbaz[j] = 0.f; }
  for (int j = nb; j < nb8; j++) { fbx[j] = fby[j] = fbz[j] = 0.f; fbm[j] = 0.f; fbax[j] = fbay[j] = fbaz[j] = 0.f; }
  const __m256 e2 = _mm256_set1_ps((float)EPS2), c15 = _mm256_set1_ps(1.5f), ch = _mm256_set1_ps(0.5f);
  for (int i = ab; i < ae; i++) {
    __m256 xi = _mm256_set1_ps((float)(px[i] - ox)), yi = _mm256_set1_ps((float)(py[i] - oy)), zi = _mm256_set1_ps((float)(pz[i] - oz)), mi = _mm256_set1_ps((float)pm[i]);
    __m256 gx = _mm256_setzero_ps(), gy = gx, gz = gx;
    for (int j = 0; j < nb8; j += 8) {
      __m256 dx = _mm256_sub_ps(_mm256_load_ps(fbx + j), xi), dy = _mm256_sub_ps(_mm256_load_ps(fby + j), yi), dz = _mm256_sub_ps(_mm256_load_ps(fbz + j), zi);
      __m256 r2 = _mm256_fmadd_ps(dx, dx, _mm256_fmadd_ps(dy, dy, _mm256_fmadd_ps(dz, dz, e2)));
      __m256 ri = _mm256_rsqrt_ps(r2);
      ri = _mm256_mul_ps(ri, _mm256_fnmadd_ps(_mm256_mul_ps(ch, r2), _mm256_mul_ps(ri, ri), c15));
      __m256 ri3 = _mm256_mul_ps(ri, _mm256_mul_ps(ri, ri));
      __m256 wj = _mm256_mul_ps(_mm256_load_ps(fbm + j), ri3);
      gx = _mm256_fmadd_ps(wj, dx, gx); gy = _mm256_fmadd_ps(wj, dy, gy); gz = _mm256_fmadd_ps(wj, dz, gz);
      if (!self) {
        __m256 wi = _mm256_mul_ps(mi, ri3);
        _mm256_store_ps(fbax + j, _mm256_fnmadd_ps(wi, dx, _mm256_load_ps(fbax + j)));
        _mm256_store_ps(fbay + j, _mm256_fnmadd_ps(wi, dy, _mm256_load_ps(fbay + j)));
        _mm256_store_ps(fbaz + j, _mm256_fnmadd_ps(wi, dz, _mm256_load_ps(fbaz + j)));
      }
    }
    ax_[i] += hsum8f(gx); ay_[i] += hsum8f(gy); az_[i] += hsum8f(gz);
  }
  if (!self) for (int j = 0; j < nb; j++) { ax_[bb + j] += fbax[j]; ay_[bb + j] += fbay[j]; az_[bb + j] += fbaz[j]; }
}

// single-precision P2P (pre-pass estimate only)
static float *fpx, *fpy, *fpz, *fpm, *fax, *fay, *faz;
static bool FASTP2P = false;
static inline float hsum8(__m256 v) { alignas(32) float t[8]; _mm256_store_ps(t, v); return ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7])); }
static void p2p_mutual_f(int ab, int ae, int bb, int be, bool self) {
  const __m256 e2 = _mm256_set1_ps((float)EPS2);
  int nb = be - bb, nfull = nb & ~7;
  __m256 msk = _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(nb - nfull), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)));
  for (int i = ab; i < ae; i++) {
    __m256 xi = _mm256_set1_ps(fpx[i]), yi = _mm256_set1_ps(fpy[i]), zi = _mm256_set1_ps(fpz[i]), mi = _mm256_set1_ps(fpm[i]);
    __m256 gx = _mm256_setzero_ps(), gy = gx, gz = gx;
    for (int j0 = 0; j0 < nb; j0 += 8) {
      int j = bb + j0;
      __m256 dx = _mm256_sub_ps(_mm256_loadu_ps(fpx + j), xi), dy = _mm256_sub_ps(_mm256_loadu_ps(fpy + j), yi), dz = _mm256_sub_ps(_mm256_loadu_ps(fpz + j), zi);
      __m256 r2 = _mm256_fmadd_ps(dx, dx, _mm256_fmadd_ps(dy, dy, _mm256_fmadd_ps(dz, dz, e2)));
      __m256 ri = _mm256_rsqrt_ps(r2);
      __m256 ri3 = _mm256_mul_ps(ri, _mm256_mul_ps(ri, ri));
      if (j0 >= nfull) ri3 = _mm256_and_ps(ri3, msk);
      __m256 wj = _mm256_mul_ps(_mm256_loadu_ps(fpm + j), ri3);
      gx = _mm256_fmadd_ps(wj, dx, gx); gy = _mm256_fmadd_ps(wj, dy, gy); gz = _mm256_fmadd_ps(wj, dz, gz);
      if (!self) {
        __m256 wi = _mm256_mul_ps(mi, ri3);
        _mm256_storeu_ps(fax + j, _mm256_fnmadd_ps(wi, dx, _mm256_loadu_ps(fax + j)));
        _mm256_storeu_ps(fay + j, _mm256_fnmadd_ps(wi, dy, _mm256_loadu_ps(fay + j)));
        _mm256_storeu_ps(faz + j, _mm256_fnmadd_ps(wi, dz, _mm256_loadu_ps(faz + j)));
      }
    }
    fax[i] += hsum8(gx); fay[i] += hsum8(gy); faz[i] += hsum8(gz);
  }
}

// ---------------- traversal ----------------
static double RHOMAX = 0.5;
static double C_M2L = 3000, C_PC = 300; // costs in units of one mutual P2P pair
static int NPC = 8;
static int PORD = 8;
static bool DRY = false; // count only
static double cost_acc = 0;

static int GMODE = 2;
static bool GEOM = false;
static inline double gfun(double rho) { // force error bound ~ (p+1) rho^(p-1)/(1-rho)^2
  double r = 1.0;
  if (GMODE == 0) { for (int i = 0; i < PORD; i++) r *= rho; return r / (1.0 - rho); }
  for (int i = 0; i < PORD - 1; i++) r *= rho;
  return (PORD + 1) * r / ((1.0 - rho) * (1.0 - rho));
}
static inline void do_p2p(int a, int b) {
  const Node& A = nodes[a]; const Node& B = nodes[b];
  if (DRY) { double na = A.e - A.b, nb = B.e - B.b; cost_acc += (P2P_PAIR_NS * na * nb + P2P_I_NS * na + P2P_CALL_NS) / C_PAIR_NS; n_pairs += na * nb; n_p2p++; return; }
  if (FASTP2P) { p2p_mutual_f(A.b, A.e, B.b, B.e, false); return; }
  if (FLOATP2P) { n_pairs += (double)(A.e - A.b) * (B.e - B.b); n_p2p++; p2p_fl(A.b, A.e, B.b, B.e, false); return; }
  unsigned long long t0 = __rdtsc();
  p2p_mutual(A.b, A.e, B.b, B.e);
  PR_P2P += __rdtsc() - t0;
}
static inline void do_m2l(int a, int b) {
  if (DRY) { cost_acc += C_M2L; n_m2l++; return; }
  m2l_push(a, b);
}
static inline void do_pc(int lf, int c) {
  if (DRY) { cost_acc += C_PC * (nodes[lf].e - nodes[lf].b); n_pc += nodes[lf].e - nodes[lf].b; return; }
  unsigned long long t0 = __rdtsc();
  pc_interact(lf, c);
  PR_PC += __rdtsc() - t0;
}

static void interact(int a, int b) {
  const Node& A = nodes[a]; const Node& B = nodes[b];
  double dx = A.cx - B.cx, dy = A.cy - B.cy, dz = A.cz - B.cz;
  double R2 = dx * dx + dy * dy + dz * dz, s2 = R2 + EPS2;
  double rs = A.r + B.r;
  int nA = A.e - A.b, nB = B.e - B.b;
  double np = (double)nA * nB;
  if (rs * rs < RHOMAX * RHOMAX * s2) {
    bool ok;
    if (GEOM) ok = true;
    else { double g = gfun(rs / sqrt(s2));
    if (GMODE == 2) { double gt = g * g / (TOLR * TOLR * s2); ok = B.m * gt <= aminv[a] && A.m * gt <= aminv[b]; }
    else ok = B.m * g <= tolv[a] * s2 && A.m * g <= tolv[b] * s2; }
    if (ok) {
      if (np < C_M2L) do_p2p(a, b); else do_m2l(a, b);
      return;
    }
  }
  bool al = A.child < 0, bl = B.child < 0;
  if ((al && bl) || np < C_PC) { do_p2p(a, b); return; }
  // particle-cell for small leaves
  int lf = -1, c = -1;
  if (al && nA <= NPC && (!bl || nB > nA)) { lf = a; c = b; }
  else if (bl && nB <= NPC) { lf = b; c = a; }
  if (lf >= 0) {
    const Node& Lf = nodes[lf]; const Node& C = nodes[c];
    double R = sqrt(R2), d = std::max(R - Lf.r, 0.0);
    double sm2 = d * d + EPS2;
    if (C.r * C.r < RHOMAX * RHOMAX * sm2) {
      bool ok;
      if (GEOM) ok = true;
      else { double g = gfun(C.r / sqrt(sm2));
      if (GMODE == 2) { double gt = g * g / (TOLR * TOLR * sm2); ok = C.m * gt <= aminv[lf] && Lf.m * gt <= aminv[c]; }
      else ok = C.m * g <= tolv[lf] * sm2 && Lf.m * g <= tolv[c] * sm2; }
      if (ok) {
        int nl = Lf.e - Lf.b, ncn = C.e - C.b;
        if (C_PC * nl < (double)nl * ncn) do_pc(lf, c); else do_p2p(lf, c);
        return;
      }
    }
  }
  if (bl || (!al && A.r >= B.r)) { for (int c2 = A.child; c2 < A.child + A.nch; c2++) interact(c2, b); }
  else { for (int c2 = B.child; c2 < B.child + B.nch; c2++) interact(a, c2); }
}
static void self_interact(int a) {
  const Node& A = nodes[a];
  if (A.child < 0) {
    if (DRY) { double na = A.e - A.b; cost_acc += (P2P_SELF_NS * na * na + P2P_I_NS * na + P2P_CALL_NS) / C_PAIR_NS; return; }
    if (FASTP2P) { p2p_mutual_f(A.b, A.e, A.b, A.e, true); return; }
    if (FLOATP2P) { p2p_fl(A.b, A.e, A.b, A.e, true); return; }
    unsigned long long t0 = __rdtsc();
    p2p_self(A.b, A.e);
    PR_SELF += __rdtsc() - t0; return;
  }
  for (int c = A.child; c < A.child + A.nch; c++) self_interact(c);
  for (int c1 = A.child; c1 < A.child + A.nch; c1++)
    for (int c2 = c1 + 1; c2 < A.child + A.nch; c2++) interact(c1, c2);
}

static double binom(int n, int k) { double r = 1; for (int i = 1; i <= k; i++) r = r * (n - k + i) / i; return r; }
static double ud_cost_s(int p, size_t nn, int n) {
  double nc = binom(p + 3, 3), c6 = binom(p + 6, 6);
  double batches = 0, quads = 0;
  for (size_t i = 0; i < nodes.size(); i++) {
    const Node& nd = nodes[i];
    if (nd.child >= 0) batches += (nd.nch + 3) / 4; else quads += (nd.e - nd.b + 3) / 4;
  }
  (void)nn; (void)n;
  return batches * c6 * UD_B_NS * 1e-9 + quads * nc * UD_Q_NS * 1e-9;
}
static double node_ns(int p) { return 0; }
static double m2l_ns(int p) { static const double t[] = {20, 30, 50, 75, 100, 170, 280, 450, 730, 1150, 1740, 2650, 3900, 5500, 7500}; return t[p]; }
static double pc_ns(int p) { static const double t[] = {10, 15, 25, 32, 40, 65, 93, 140, 206, 300, 405, 550, 713, 900, 1100}; return t[p]; }
static void setup_pass(MI& T, double rhomax, double tol, const std::vector<double>* amin) {
  TT = &T; PORD = T.p; RHOMAX = rhomax;
  C_M2L = m2l_ns(T.p) / C_PAIR_NS; C_PC = pc_ns(T.p) / C_PAIR_NS;
  size_t nn = nodes.size();
  tolv.assign(nn, 1e300); aminv.assign(nn, 1e300); TOLR = amin ? tol : 1e150; GEOM = (amin == nullptr);
  if (amin) for (size_t i = 0; i < nn; i++) { tolv[i] = tol * (*amin)[i]; aminv[i] = (*amin)[i]; }
}
// predicted traversal work in ns
static double dry_run(MI& T, double rhomax, double tol, const std::vector<double>* amin) {
  setup_pass(T, rhomax, tol, amin);
  DRY = true; cost_acc = 0; n_m2l = n_p2p = n_pc = 0; n_pairs = 0;
  self_interact(0);
  DRY = false;
  return cost_acc * C_PAIR_NS;
}
static void run_pass(MI& T, double rhomax, double tol, const std::vector<double>* amin) {
  setup_pass(T, rhomax, tol, amin);
  size_t nn = nodes.size();
  NCP = T.ncp;
  memset(Lm, 0, nn * NCP * 8);
  double t0 = cpu_time();
  upward();
  double t1 = cpu_time();
  n_m2l = n_p2p = n_pc = 0; n_pairs = 0;
  self_interact(0);
  m2l_flush();
  double t2 = cpu_time();
  downward();
  double t3 = cpu_time();
  if (VERB) fprintf(stderr, "  prof(Gcyc): m2l %.3f p2p %.3f self %.3f pc %.3f\n", PR_M2L * 1e-9, PR_P2P * 1e-9, PR_SELF * 1e-9, PR_PC * 1e-9);
  if (VERB) fprintf(stderr, "  pass p=%d rhomax=%.3f tol=%.3g: m2l=%ld p2p=%ld pairs=%.3g pc=%ld | up %.3f trav %.3f down %.3f\n",
                    T.p, rhomax, tol, n_m2l, n_p2p, n_pairs, n_pc, t1 - t0, t2 - t1, t3 - t2);
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: solve INPUT OUTPUT\n"); return 1; }
  double t0 = cpu_time();
  const char* ev;
  if ((ev = getenv("FMM_VERB"))) VERB = atoi(ev);
  int P = 10, P0 = 3; double rho0 = 0.8, rhom = -1, tol = -1;
  if ((ev = getenv("FMM_P"))) P = atoi(ev);
  if ((ev = getenv("FMM_P0"))) P0 = atoi(ev);
  if ((ev = getenv("FMM_RHO0"))) rho0 = atof(ev);
  if ((ev = getenv("FMM_RHOM"))) rhom = atof(ev);
  if ((ev = getenv("FMM_TOL"))) tol = atof(ev);
  if ((ev = getenv("FMM_NLEAF"))) NLEAF = atoi(ev);
  if ((ev = getenv("FMM_NPC"))) NPC = atoi(ev);
  if ((ev = getenv("FMM_KAPPA"))) KAPPA = atof(ev);
  int skip_pre = (ev = getenv("FMM_NOPRE")) ? atoi(ev) : 0;
  if ((ev = getenv("FMM_GMODE"))) GMODE = atoi(ev);
  double limit = -1;
  if ((ev = getenv("GALAXY_CPU_LIMIT_S"))) limit = atof(ev);
  double SAFE = 0.75;
  if ((ev = getenv("FMM_SAFE"))) SAFE = atof(ev);
  if ((ev = getenv("FMM_TOL0"))) TOL0 = atof(ev);
  if ((ev = getenv("FMM_FLOAT"))) FLOATMODE = atoi(ev);
  if ((ev = getenv("FMM_TRCAL"))) TRCAL = atof(ev);
  if ((ev = getenv("FMM_PREF"))) PREF = atoi(ev);
  if ((ev = getenv("FMM_MULTICEN"))) MULTICEN = atoi(ev);
  if ((ev = getenv("FMM_F0"))) { F0 = atof(ev); AUTOMODE = 0; }
  if ((ev = getenv("FMM_AUTO"))) AUTOMODE = atoi(ev);
  if ((ev = getenv("FMM_CENFRAC"))) CENFRAC = atof(ev);
  if ((ev = getenv("FMM_CPAIR"))) C_PAIR_NS = atof(ev);
  if ((ev = getenv("FMM_PPAIR"))) P2P_PAIR_NS = atof(ev);
  if ((ev = getenv("FMM_UDB"))) UD_B_NS = atof(ev);
  if ((ev = getenv("FMM_UDQ"))) UD_Q_NS = atof(ev);
  if ((ev = getenv("FMM_CPFL"))) C_PAIR_FL = atof(ev);
  FILE* f = fopen(argv[1], "rb");
  if (!f) { perror("input"); return 1; }
  fseek(f, 0, SEEK_END); long nb = ftell(f); fseek(f, 0, SEEK_SET);
  N = (int)(nb / 32);
  if (!(limit > 0)) limit = 0.9 * pow((double)N / 65536.0, 1.1);
  std::vector<double> raw((size_t)N * 4);
  if (fread(raw.data(), 32, N, f) != (size_t)N) { fprintf(stderr, "read error\n"); return 1; }
  fclose(f);
  int NP = N + 8;
  px = (double*)aligned_alloc(64, NP * 8 + 64); py = (double*)aligned_alloc(64, NP * 8 + 64); pz = (double*)aligned_alloc(64, NP * 8 + 64); pm = (double*)aligned_alloc(64, NP * 8 + 64);
  ax_ = (double*)aligned_alloc(64, NP * 8 + 64); ay_ = (double*)aligned_alloc(64, NP * 8 + 64); az_ = (double*)aligned_alloc(64, NP * 8 + 64);
  pid = (int*)malloc(NP * 4);
  for (int i = 0; i < NP; i++) { px[i] = py[i] = pz[i] = pm[i] = 0; ax_[i] = ay_[i] = az_[i] = 0; pid[i] = 0; }
  for (int i = 0; i < N; i++) { pm[i] = raw[4 * i]; px[i] = raw[4 * i + 1]; py[i] = raw[4 * i + 2]; pz[i] = raw[4 * i + 3]; pid[i] = i; }
  std::vector<double>().swap(raw);
  int pforced = getenv("FMM_P") != nullptr;
  double bpp = SAFE * limit / N * 1e6; // us per particle
  if (!pforced) {
    if (N >= 500000) { if (bpp < 9) P = 8; else if (bpp < 14) P = 10; else P = 12; }
    else if (N >= 200000) { if (bpp < 10) P = 8; else if (bpp < 14) P = 10; else if (bpp < 24) P = 11; else P = 12; }
    else { if (bpp < 11) P = 8; else if (bpp < 18) P = 10; else if (bpp < 32) P = 11; else P = 12; }
  }
  if (!getenv("FMM_NLEAF")) NLEAF = (N >= 2000000) ? 128 : 64;
  if (AUTOMODE && N >= 1000) {
    // classify: extended halo (r90/r50 large) vs compact/multi-centred
    double cm[3] = {0, 0, 0}, mt = 0;
    for (int i = 0; i < N; i++) { cm[0] += pm[i] * px[i]; cm[1] += pm[i] * py[i]; cm[2] += pm[i] * pz[i]; mt += pm[i]; }
    if (mt > 0) { cm[0] /= mt; cm[1] /= mt; cm[2] /= mt; }
    std::vector<float> rr(N);
    for (int i = 0; i < N; i++) { double dx = px[i] - cm[0], dy = py[i] - cm[1], dz = pz[i] - cm[2]; rr[i] = (float)(dx * dx + dy * dy + dz * dz); }
    std::nth_element(rr.begin(), rr.begin() + N / 2, rr.end()); double r50 = sqrt((double)rr[N / 2]);
    std::nth_element(rr.begin(), rr.begin() + (int)(0.9 * N), rr.end()); double r90 = sqrt((double)rr[(int)(0.9 * N)]);
    double conc = r90 / std::max(r50, 1e-300);
    if (conc > 3.0) { MULTICEN = 0; F0 = 0.05; } else { MULTICEN = 1; F0 = 0; }
    if (VERB) fprintf(stderr, "  r90/r50=%.2f -> multicen=%d f0=%.3f\n", conc, MULTICEN, F0);
  }
  double t1 = cpu_time();
  build_tree();
  for (int guard = 0; guard < 6 && (double)nodes.size() * 36 * 16.0 > MEMCAP; guard++) {
    // pathological tree size: rebuild with bigger leaves and no outer splitting
    KAPPA = 1e9; NLEAF *= 2;
    build_tree();
  }
  for (int i = N; i < NP; i++) { px[i] = py[i] = pz[i] = 0; pm[i] = 0; }
  size_t nn = nodes.size();
  {
    double maxbytes = MEMCAP;
    while (P > 4) {
      int nc = (P + 1) * (P + 2) * (P + 3) / 6; nc = (nc + 3) & ~3;
      if ((double)nn * nc * 16.0 <= maxbytes) break;
      P--;
    }
  }
  MI T1; T1.init(P);
  if (FLOATMODE == 1 || (FLOATMODE == 2 && P <= 9)) {
    FLOATP2P = true; C_PAIR_NS = C_PAIR_FL;
    size_t fb = (size_t)N + 64;
    fbx = (float*)aligned_alloc(64, fb * 4); fby = (float*)aligned_alloc(64, fb * 4); fbz = (float*)aligned_alloc(64, fb * 4); fbm = (float*)aligned_alloc(64, fb * 4);
    fbax = (float*)aligned_alloc(64, fb * 4); fbay = (float*)aligned_alloc(64, fb * 4); fbaz = (float*)aligned_alloc(64, fb * 4);
  }
  double t2 = cpu_time();
  size_t ncmax = T1.ncp;
  Mm = (double*)aligned_alloc(64, nn * ncmax * 8 + 64); Lm = (double*)aligned_alloc(64, nn * ncmax * 8 + 64);
  size_t vb = (ncmax + 8) * 32;
  Xb = (V*)aligned_alloc(64, vb); Yb = (V*)aligned_alloc(64, vb); Dv = (V*)aligned_alloc(64, vb);
  MAv = (V*)aligned_alloc(64, vb); MBv = (V*)aligned_alloc(64, vb); LAv = (V*)aligned_alloc(64, vb); LBv = (V*)aligned_alloc(64, vb);
  UVmo = (V*)aligned_alloc(64, vb); UVacc = (V*)aligned_alloc(64, vb); UVt = (V*)aligned_alloc(64, vb);
  ZEROROW = (double*)aligned_alloc(64, ncmax * 8 + 64); memset(ZEROROW, 0, ncmax * 8 + 64);
  std::vector<double> amin;
  if (F0 > 0) {
    MI T0; T0.init(3);
    fpx = (float*)aligned_alloc(64, NP * 4 + 64); fpy = (float*)aligned_alloc(64, NP * 4 + 64); fpz = (float*)aligned_alloc(64, NP * 4 + 64); fpm = (float*)aligned_alloc(64, NP * 4 + 64);
    fax = (float*)aligned_alloc(64, NP * 4 + 64); fay = (float*)aligned_alloc(64, NP * 4 + 64); faz = (float*)aligned_alloc(64, NP * 4 + 64);
    for (int i = 0; i < NP; i++) { fpx[i] = (float)px[i]; fpy[i] = (float)py[i]; fpz[i] = (float)pz[i]; fpm[i] = (float)pm[i]; fax[i] = fay[i] = faz[i] = 0; }
    FASTP2P = true; NCP = T0.ncp;
    run_pass(T0, 0.8, 0, nullptr);
    FASTP2P = false;
    for (int i = 0; i < N; i++) { ax_[i] += fax[i]; ay_[i] += fay[i]; az_[i] += faz[i]; }
    free(fpx); free(fpy); free(fpz); free(fpm); free(fax); free(fay); free(faz);
    amin.assign(nn, 1e300);
    double afloor = 0;
    {
      std::vector<float> am(N);
      for (int i = 0; i < N; i++) am[i] = (float)sqrt(ax_[i] * ax_[i] + ay_[i] * ay_[i] + az_[i] * az_[i]);
      std::nth_element(am.begin(), am.begin() + N / 2, am.end());
      afloor = 1e-4 * am[N / 2];
    }
    for (int ni = (int)nn - 1; ni >= 0; ni--) {
      const Node& nd = nodes[ni]; double am = 1e300;
      if (nd.child < 0) { for (int i = nd.b; i < nd.e; i++) am = std::min(am, sqrt(ax_[i] * ax_[i] + ay_[i] * ay_[i] + az_[i] * az_[i])); }
      else for (int c = nd.child; c < nd.child + nd.nch; c++) am = std::min(am, amin[c]);
      amin[ni] = std::max(0.5 * am, afloor); // conservative, floored
    }
    for (int i = 0; i < NP; i++) ax_[i] = ay_[i] = az_[i] = 0;
    NCP = T1.ncp;
    GMODE = 1;
  }
  const std::vector<double>* aminp = (F0 > 0) ? &amin : nullptr;
  double t3 = cpu_time();
  for (int attempt = 0; attempt < 3 && rhom <= 0; attempt++) {
    if (attempt > 0) { P -= 2; T1.init(P); }
    double t3 = cpu_time();
    double t_write = 3e-8 * N + 0.002;
    double avail = SAFE * limit - (t3 - t0) - t_write - ud_cost_s(P, nn, N) * UDCAL;
    double best = -1, bestc = 0, rho = RHO0, lastc = 0, lastr = RHO0;
    double badr = -1, badc = 0; // largest infeasible rho below best
    double kexp = 2.0;
    for (int it = 0; it < 5; it++) {
      double td0 = cpu_time();
      PORD = T1.p;
      double c = dry_run(T1, rho, F0 * gfun(rho), aminp) * 1e-9 * TRCAL * std::max(0.85, 1.0 - 0.02 * std::max(0.0, log2((double)N / 65536.0)));
      double td1 = cpu_time();
      c += (td1 - td0) * DRYCAL;
      avail -= (td1 - td0);
      bool ok = c <= avail;
      if (VERB) fprintf(stderr, "  dry rho=%.4f pred=%.3f avail=%.3f ok=%d (dry %.3f s)\n", rho, c, avail, (int)ok, td1 - td0);
      if (it > 0 && fabs(log(rho / lastr)) > 1e-3 && c > 0 && lastc > 0) {
        double k = log(c / lastc) / log(lastr / rho);
        if (k > 0.5 && k < 6) kexp = k;
      }
      if (ok && (best < 0 || rho < best)) { best = rho; bestc = c; }
      if (!ok && (best < 0 || rho < best) && rho > badr) { badr = rho; badc = c; }
      lastc = c; lastr = rho;
      if (ok && c > 0.88 * avail) break;
      double target = 0.95 * avail;
      double nr = rho * pow(c / std::max(target, 1e-4), 1.0 / kexp);
      if (best > 0 && badr > 0) { // bracketed: interpolate between them
        double k = (badc > bestc && badr < best) ? log(badc / bestc) / log(best / badr) : kexp;
        if (!(k > 0.5 && k < 6)) k = kexp;
        nr = best * pow(bestc / std::max(target, 1e-4), 1.0 / k);
        nr = std::min(std::max(nr, badr + 0.25 * (best - badr)), best - 0.25 * (best - badr));
      }
      nr = std::min(std::max(nr, RHOMIN), 0.8);
      if (best > 0 && nr >= best) break;
      if (fabs(nr - rho) < 0.003) break;
      if (!ok && rho >= 0.8) break;
      rho = nr;
      if (avail < 0.02 * SAFE * limit) break;
    }
    if (best < 0) {
      double ex = lastr * pow(lastc / std::max(avail, 1e-4), 0.5) * 1.06;
      if (ex > 0.8 && P >= 8 && attempt < 2) { if (VERB) fprintf(stderr, "  infeasible at p=%d, reducing p\n", P); continue; }
      best = std::min(0.8, ex);
    }
    rhom = best;
    if (VERB) fprintf(stderr, "  chosen rho=%.4f pred=%.3f ud_pred=%.3f\n", rhom, bestc, ud_cost_s(P, nn, N));
  }
  if (!(rhom > 0) || rhom > 0.8) rhom = 0.8;
  PORD = T1.p;
  run_pass(T1, rhom, F0 * gfun(rhom), aminp);
  double t4 = cpu_time();
  std::vector<double> out((size_t)N * 3);
  for (int i = 0; i < N; i++) { int k = pid[i]; out[3 * k] = ax_[i]; out[3 * k + 1] = ay_[i]; out[3 * k + 2] = az_[i]; }
  FILE* g = fopen(argv[2], "wb");
  if (!g) { perror("output"); return 1; }
  size_t wr = fwrite(out.data(), 24, N, g);
  if (fclose(g) != 0 || wr != (size_t)N) { fprintf(stderr, "write error\n"); return 1; }
  double t5 = cpu_time();
  if (VERB) { FILE* st = fopen("/proc/self/status", "r"); char line[256]; while (st && fgets(line, sizeof line, st)) if (!strncmp(line, "VmHWM", 5)) fprintf(stderr, "%s", line); if (st) fclose(st); }
  if (VERB) fprintf(stderr, "N=%d p=%d nleaf=%d nodes=%zu | read %.3f tree %.3f pre %.3f main %.3f write %.3f total %.3f\n",
                    N, P, NLEAF, nn, t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t0);
  return 0;
}
