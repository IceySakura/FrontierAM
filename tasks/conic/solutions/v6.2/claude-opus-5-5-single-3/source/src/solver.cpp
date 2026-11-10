// Homogeneous-embedding primal-dual interior point method for
//   min 1/2 x'Px + q'x  s.t.  Ax + s = b, s in K
// K: zero, nonnegative, second-order, exponential (and PSD) cones.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include "io.hpp"
#include "linalg.hpp"

using std::vector;
using std::string;

enum { CZERO = 0, CNN = 1, CSOC = 2, CEXP = 3, CPSD = 4 };
struct Cone { int type; I dim; I off; I d; I hoff; };

static double t_start;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double TIME_CAP = 280.0;
static int VERB = 0;
static double EXP_BETA = 0.05;
static double SIGMA_MIN = 0.0;
static int REF_MAX = 8; static double REF_RATIO = 2.0, SREG = 1e-8, DYN_DELTA = 2e-7;
static double T_FACT = 0, T_SOLVE = 0, T_LDLS = 0; static long N_LDLS = 0;

static inline double nrminf(const double *v, I n) { double r = 0; for (I i = 0; i < n; i++) r = std::max(r, std::fabs(v[i])); return r; }
static inline double dotp(const double *a, const double *b, I n) { double r = 0; for (I i = 0; i < n; i++) r += a[i] * b[i]; return r; }

// ---------------- dense symmetric eigen (Jacobi) for PSD blocks ----------------
static void jacobi_eig(int d, vector<double> &A, vector<double> &w, vector<double> &V) {
  V.assign((size_t)d * d, 0.0);
  for (int i = 0; i < d; i++) V[(size_t)i * d + i] = 1.0;
  for (int sweep = 0; sweep < 60; sweep++) {
    double off = 0, tot = 0;
    for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) { double a = A[(size_t)i * d + j]; if (i != j) off += a * a; tot += a * a; }
    if (off <= 1e-30 * tot || off == 0) break;
    for (int p = 0; p < d; p++)
      for (int q = p + 1; q < d; q++) {
        double apq = A[(size_t)p * d + q];
        if (std::fabs(apq) < 1e-300) continue;
        double app = A[(size_t)p * d + p], aqq = A[(size_t)q * d + q];
        double theta = (aqq - app) / (2 * apq);
        double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1));
        double c = 1 / std::sqrt(t * t + 1), s = t * c;
        for (int k = 0; k < d; k++) {
          double akp = A[(size_t)k * d + p], akq = A[(size_t)k * d + q];
          A[(size_t)k * d + p] = c * akp - s * akq;
          A[(size_t)k * d + q] = s * akp + c * akq;
        }
        for (int k = 0; k < d; k++) {
          double apk = A[(size_t)p * d + k], aqk = A[(size_t)q * d + k];
          A[(size_t)p * d + k] = c * apk - s * aqk;
          A[(size_t)q * d + k] = s * apk + c * aqk;
        }
        for (int k = 0; k < d; k++) {
          double vkp = V[(size_t)k * d + p], vkq = V[(size_t)k * d + q];
          V[(size_t)k * d + p] = c * vkp - s * vkq;
          V[(size_t)k * d + q] = s * vkp + c * vkq;
        }
      }
  }
  w.resize(d);
  for (int i = 0; i < d; i++) w[i] = A[(size_t)i * d + i];
}
static void smat(int d, const double *v, vector<double> &M) {
  M.assign((size_t)d * d, 0.0);
  const double r2 = std::sqrt(2.0);
  I pos = 0;
  for (int c = 0; c < d; c++)
    for (int r = 0; r <= c; r++) {
      double val = v[pos++];
      if (r != c) val /= r2;
      M[(size_t)r * d + c] = val; M[(size_t)c * d + r] = val;
    }
}
static double psd_min_eig(int d, const double *v) {
  vector<double> M, w, V;
  smat(d, v, M);
  jacobi_eig(d, M, w, V);
  double mn = 1e300;
  for (double x : w) mn = std::min(mn, x);
  return mn;
}

// ---------------- checker-equivalent cone defects ----------------
static double exp_violation(double x, double y, double z) {
  double boundary = std::max(std::fabs(y), std::max(std::max(x, 0.0), std::max(-z, 0.0)));
  if (y > 0.0 && z > 0.0) {
    double interior = std::max(0.0, x - y * (std::log(z) - std::log(y)));
    return std::min(interior, boundary);
  }
  return boundary;
}
static double cone_defect(const vector<Cone> &cones, const double *v, bool dual) {
  double worst = 0;
  for (const Cone &c : cones) {
    const double *b = v + c.off;
    double nb = nrminf(b, c.dim);
    double sc = std::max(1.0, nb);
    double viol = 0;
    if (c.type == CZERO) viol = dual ? 0.0 : nb / sc;
    else if (c.type == CNN) { double mn = 0; for (I k = 0; k < c.dim; k++) mn = std::min(mn, b[k] / sc); viol = -mn; }
    else if (c.type == CSOC) { double s = 0; for (I k = 1; k < c.dim; k++) { double u = b[k] / sc; s += u * u; } viol = std::max(0.0, std::sqrt(s) - b[0] / sc); }
    else if (c.type == CEXP) {
      double u0 = b[0] / sc, u1 = b[1] / sc, u2 = b[2] / sc;
      viol = dual ? exp_violation(-u1, -u0, M_E * u2) : exp_violation(u0, u1, u2);
    } else {
      vector<double> u(b, b + c.dim);
      for (auto &x : u) x /= sc;
      viol = std::max(0.0, -psd_min_eig((int)c.d, u.data()));
    }
    worst = std::max(worst, viol);
  }
  return worst;
}

// ---------------- problem ----------------
struct Problem {
  I n, m;
  CSC P, A;  // canonical, full symmetric P
  vector<double> q, b;
  vector<Cone> cones;
  double c0 = 0;
};

struct Metrics { double rp, rd, gap_abs, gap_rel, pcone, dcone; bool ok; double score; };

static Metrics eval_solved(const Problem &pr, const vector<double> &x, const vector<double> &s, const vector<double> &z, double tol,
                           bool cone_check) {
  I n = pr.n, m = pr.m;
  vector<double> Ax(m, 0.0), Px(n, 0.0), Atz(n, 0.0);
  spmv_add(pr.A, x.data(), Ax.data());
  spmv_add(pr.P, x.data(), Px.data());
  spmtv_add(pr.A, z.data(), Atz.data());
  Metrics M;
  double rp = 0; for (I i = 0; i < m; i++) rp = std::max(rp, std::fabs(Ax[i] + s[i] - pr.b[i]));
  M.rp = rp / (1 + std::max(nrminf(Ax.data(), m), std::max(nrminf(s.data(), m), nrminf(pr.b.data(), m))));
  double rd = 0; for (I j = 0; j < n; j++) rd = std::max(rd, std::fabs(Px[j] + pr.q[j] + Atz[j]));
  M.rd = rd / (1 + std::max(nrminf(Px.data(), n), std::max(nrminf(pr.q.data(), n), nrminf(Atz.data(), n))));
  double xPx = dotp(x.data(), Px.data(), n), qx = dotp(pr.q.data(), x.data(), n), bz = dotp(pr.b.data(), z.data(), m);
  double p0 = 0.5 * xPx + qx, d0 = -0.5 * xPx - bz;
  M.gap_abs = std::fabs(p0 - d0);
  double scl = std::max(std::fabs(p0), std::fabs(d0));
  M.gap_rel = scl > 0 ? M.gap_abs / scl : 0.0;
  M.pcone = M.dcone = 0;
  if (cone_check) { M.pcone = cone_defect(pr.cones, s.data(), false); M.dcone = cone_defect(pr.cones, z.data(), true); }
  double g = std::min(M.gap_abs, M.gap_rel);
  M.score = std::max(std::max(M.rp, M.rd), std::max(g, std::max(M.pcone, M.dcone)));
  M.ok = std::isfinite(M.score) && M.score <= tol;
  return M;
}

static double mat_scale(const CSC &A) { double a = 0; for (double v : A.x) a = std::max(a, std::fabs(v)); return a > 0 ? a : 1.0; }

// returns violation ratio (<= tol means pass), or inf
static double eval_pinf(const Problem &pr, const vector<double> &zin) {
  I n = pr.n, m = pr.m;
  double nz = nrminf(zin.data(), m), nb = nrminf(pr.b.data(), m);
  if (!(nz > 0) || !(nb > 0) || !std::isfinite(nz)) return 1e300;
  vector<double> z(m);
  for (I i = 0; i < m; i++) z[i] = zin[i] / nz;
  double beta = 0; for (I i = 0; i < m; i++) beta += (pr.b[i] / nb) * z[i];
  beta = -beta;
  if (!(beta > 1e-8)) return 1e300;
  double a = mat_scale(pr.A);
  vector<double> Atz(n, 0.0);
  spmtv_add(pr.A, z.data(), Atz.data());
  double lin = nrminf(Atz.data(), n) / a / beta;
  double cone = cone_defect(pr.cones, z.data(), true) / beta;
  double r = std::max(lin, cone);
  return std::isfinite(r) ? r : 1e300;
}
static double eval_dinf(const Problem &pr, const vector<double> &xin, const vector<double> &sin_) {
  I n = pr.n, m = pr.m;
  double nx = nrminf(xin.data(), n), nq = nrminf(pr.q.data(), n);
  if (!(nx > 0) || !(nq > 0) || !std::isfinite(nx)) return 1e300;
  vector<double> x(n), s(m);
  for (I j = 0; j < n; j++) x[j] = xin[j] / nx;
  for (I i = 0; i < m; i++) s[i] = sin_[i] / nx;
  double alpha = 0; for (I j = 0; j < n; j++) alpha += (pr.q[j] / nq) * x[j];
  alpha = -alpha;
  if (!(alpha > 1e-8)) return 1e300;
  double a = mat_scale(pr.A), h = mat_scale(pr.P);
  vector<double> Px(n, 0.0), Ax(m, 0.0);
  spmv_add(pr.P, x.data(), Px.data());
  spmv_add(pr.A, x.data(), Ax.data());
  double quad = (nrminf(Px.data(), n) / h) / alpha;
  double lin = 0; for (I i = 0; i < m; i++) lin = std::max(lin, std::fabs(Ax[i] / a + s[i] / a));
  lin /= alpha;
  vector<double> ss(m);
  for (I i = 0; i < m; i++) ss[i] = s[i] / a;
  double cone = cone_defect(pr.cones, ss.data(), false) / alpha;
  double r = std::max(quad, std::max(lin, cone));
  return std::isfinite(r) ? r : 1e300;
}

// ---------------- IPM ----------------
struct Result { string status; vector<double> x, s, z; double obj_p = NAN, obj_d = NAN; };

struct IPM {
  const Problem &pr;
  I n, m, N;
  CSC P, A, AT;  // scaled
  vector<double> q, b, D, E;
  double cscale = 1.0;
  vector<Cone> cones;
  double nu = 0;
  bool has_exp = false, has_psd = false;
  // KKT
  vector<I> Kp, Ki, diagX, zbase, perm, pinv, KPp, KPi, kmap;
  vector<double> Kx, KPx, sgn, Pdiag;
  LDL ldl;
  double sreg = SREG;
  // cone scaling data
  vector<double> Hd;    // diagonal H for NN
  vector<double> Hb;    // dense blocks
  vector<double> lam, wsoc, etasoc, gexp, detlam;  // lambda, SOC w, SOC eta (per cone index), exp gradient
  double mu_scal = 1.0;
  // psd data
  vector<vector<double>> psdR, psdRinv, psdLam;  // W = R (applied as congruence), lambda eigen

  IPM(const Problem &p) : pr(p) {}

  // ---------- setup: scaling ----------
  void equilibrate() {
    n = pr.n; m = pr.m; N = n + m;
    P = pr.P; A = pr.A; q = pr.q; b = pr.b; cones = pr.cones;
    D.assign(n, 1.0); E.assign(m, 1.0);
    vector<double> dn(n), en(m);
    for (int it = 0; it < 15; it++) {
      std::fill(dn.begin(), dn.end(), 0.0);
      std::fill(en.begin(), en.end(), 0.0);
      for (I j = 0; j < n; j++) {
        for (I k = P.p[j]; k < P.p[j + 1]; k++) dn[j] = std::max(dn[j], std::fabs(P.x[k]));
        for (I k = A.p[j]; k < A.p[j + 1]; k++) { double v = std::fabs(A.x[k]); dn[j] = std::max(dn[j], v); en[A.i[k]] = std::max(en[A.i[k]], v); }
      }
      double maxdev = 0;
      for (I j = 0; j < n; j++) { maxdev = std::max(maxdev, std::fabs(1 - dn[j])); dn[j] = dn[j] > 0 ? 1.0 / std::sqrt(dn[j]) : 1.0; dn[j] = std::min(1e4, std::max(1e-4, dn[j])); }
      for (I i = 0; i < m; i++) { maxdev = std::max(maxdev, en[i] > 0 ? std::fabs(1 - en[i]) : 0.0); en[i] = en[i] > 0 ? 1.0 / std::sqrt(en[i]) : 1.0; en[i] = std::min(1e4, std::max(1e-4, en[i])); }
      for (const Cone &c : cones) {
        if (c.type == CSOC || c.type == CEXP || c.type == CPSD) {
          double s = 0; for (I k = 0; k < c.dim; k++) s += en[c.off + k];
          s /= c.dim;
          for (I k = 0; k < c.dim; k++) en[c.off + k] = s;
        }
      }
      if (maxdev < 1e-3 && it > 0) break;
      for (I j = 0; j < n; j++) {
        for (I k = P.p[j]; k < P.p[j + 1]; k++) P.x[k] *= dn[j] * dn[P.i[k]];
        for (I k = A.p[j]; k < A.p[j + 1]; k++) A.x[k] *= dn[j] * en[A.i[k]];
        D[j] *= dn[j];
      }
      for (I i = 0; i < m; i++) E[i] *= en[i];
    }
    for (I j = 0; j < n; j++) q[j] *= D[j];
    for (I i = 0; i < m; i++) b[i] *= E[i];
    // cost scaling
    double pm = 0;
    if (P.nnz() > 0) {
      vector<double> cn(n, 0.0);
      for (I j = 0; j < n; j++) for (I k = P.p[j]; k < P.p[j + 1]; k++) cn[j] = std::max(cn[j], std::fabs(P.x[k]));
      for (I j = 0; j < n; j++) pm += cn[j];
      pm /= std::max<I>(n, 1);
    }
    double sc = std::max(pm, nrminf(q.data(), n));
    if (sc < 1e-4) sc = 1.0;
    sc = std::min(1e4, sc);
    cscale = 1.0 / sc;
    for (auto &v : P.x) v *= cscale;
    for (auto &v : q) v *= cscale;
    AT = transpose(A);
  }

  // ---------- KKT structure ----------
  void build_kkt() {
    // upper triangle in original ordering
    Kp.assign(N + 1, 0);
    diagX.resize(n); zbase.resize(m);
    // count
    vector<I> cnt(N, 0);
    for (I j = 0; j < n; j++) {
      for (I k = P.p[j]; k < P.p[j + 1]; k++) if (P.i[k] < j) cnt[j]++;
      cnt[j]++;
    }
    vector<I> blockStart(m, -1);
    for (const Cone &c : cones) {
      bool dense = (c.type == CSOC || c.type == CEXP || c.type == CPSD);
      for (I k = 0; k < c.dim; k++) blockStart[c.off + k] = dense ? c.off : -1;
    }
    for (I r = 0; r < m; r++) {
      cnt[n + r] = (AT.p[r + 1] - AT.p[r]) + (blockStart[r] >= 0 ? (r - blockStart[r]) : 0) + 1;
    }
    for (I j = 0; j < N; j++) Kp[j + 1] = Kp[j] + cnt[j];
    I nnz = Kp[N];
    Ki.resize(nnz); Kx.assign(nnz, 0.0);
    Pdiag.assign(n, 0.0);
    for (I j = 0; j < n; j++) {
      I pos = Kp[j];
      for (I k = P.p[j]; k < P.p[j + 1]; k++) {
        if (P.i[k] < j) { Ki[pos] = P.i[k]; Kx[pos] = P.x[k]; pos++; }
        else if (P.i[k] == j) Pdiag[j] += P.x[k];
      }
      Ki[pos] = j; diagX[j] = pos; pos++;
    }
    for (I r = 0; r < m; r++) {
      I pos = Kp[n + r];
      for (I k = AT.p[r]; k < AT.p[r + 1]; k++) { Ki[pos] = AT.i[k]; Kx[pos] = AT.x[k]; pos++; }
      zbase[r] = pos;
      I o = blockStart[r] >= 0 ? blockStart[r] : r;
      for (I rr = o; rr <= r; rr++) { Ki[pos] = n + rr; pos++; }
    }
    // full symmetric pattern without diagonal for AMD
    vector<I> Cp(N + 1, 0);
    for (I j = 0; j < N; j++)
      for (I k = Kp[j]; k < Kp[j + 1]; k++) { I i = Ki[k]; if (i != j) { Cp[i + 1]++; Cp[j + 1]++; } }
    for (I j = 0; j < N; j++) Cp[j + 1] += Cp[j];
    vector<I> Ci(Cp[N]);
    vector<I> w(Cp.begin(), Cp.end() - 1);
    for (I j = 0; j < N; j++)
      for (I k = Kp[j]; k < Kp[j + 1]; k++) { I i = Ki[k]; if (i != j) { Ci[w[i]++] = j; Ci[w[j]++] = i; } }
    double t0 = now();
    perm = amd_order(N, Cp, Ci);
    if (VERB) fprintf(stderr, "amd %.3fs N=%lld nnzK=%lld\n", now() - t0, (long long)N, (long long)nnz);
    pinv.assign(N, 0);
    for (I k = 0; k < N; k++) pinv[perm[k]] = k;
    // permuted upper triangle
    KPp.assign(N + 1, 0);
    for (I j = 0; j < N; j++)
      for (I k = Kp[j]; k < Kp[j + 1]; k++) { I a = pinv[Ki[k]], bb = pinv[j]; KPp[std::max(a, bb) + 1]++; }
    for (I j = 0; j < N; j++) KPp[j + 1] += KPp[j];
    KPi.resize(nnz); KPx.resize(nnz); kmap.resize(nnz);
    vector<I> w2(KPp.begin(), KPp.end() - 1);
    for (I j = 0; j < N; j++)
      for (I k = Kp[j]; k < Kp[j + 1]; k++) {
        I a = pinv[Ki[k]], bb = pinv[j];
        I col = std::max(a, bb), row = std::min(a, bb);
        I q2 = w2[col]++;
        KPi[q2] = row; kmap[k] = q2;
      }
    sgn.resize(N);
    for (I k = 0; k < N; k++) sgn[k] = perm[k] < n ? 1.0 : -1.0;
    t0 = now();
    ldl.symbolic(N, KPp, KPi);
    if (VERB) fprintf(stderr, "symbolic %.3fs nnzL=%lld\n", now() - t0, (long long)ldl.Lp[N]);
  }

  // ---------- cone helpers ----------
  void alloc_cones() {
    Hd.assign(m, 0.0); lam.assign(m, 0.0); wsoc.assign(m, 0.0); gexp.assign(m, 0.0);
    etasoc.assign(cones.size(), 1.0); detlam.assign(cones.size(), 1.0);
    I ho = 0;
    nu = 0;
    for (Cone &c : cones) {
      c.hoff = ho;
      if (c.type == CSOC || c.type == CEXP || c.type == CPSD) ho += c.dim * c.dim;
      if (c.type == CNN) nu += c.dim;
      else if (c.type == CSOC) nu += 1;
      else if (c.type == CEXP) { nu += 3; has_exp = true; }
      else if (c.type == CPSD) { nu += c.d; has_psd = true; }
    }
    Hb.assign(ho, 0.0);
    psdR.resize(cones.size()); psdRinv.resize(cones.size()); psdLam.resize(cones.size());
  }

  // exp primal barrier gradient/Hessian at s=(u,v,w)
  static bool exp_primal_interior(const double *s) {
    double u = s[0], v = s[1], w = s[2];
    if (!(v > 0 && w > 0)) return false;
    double psi = v * std::log(w / v) - u;
    return psi > 0 && std::isfinite(psi);
  }
  static bool exp_dual_interior(const double *z) {
    double u = z[0], v = z[1], w = z[2];
    if (!(u < 0 && w > 0)) return false;
    double r = std::log(-u) + v / u - 1.0 - std::log(w);
    return r < 0 && std::isfinite(r);
  }
  static void exp_grad_hess(const double *s, double *g, double *Hs) {
    double u = s[0], v = s[1], w = s[2];
    double lw = std::log(w / v);
    double psi = v * lw - u;
    double gp[3] = {-1.0, lw - 1.0, v / w};
    g[0] = -gp[0] / psi; g[1] = -gp[1] / psi - 1.0 / v; g[2] = -gp[2] / psi - 1.0 / w;
    double hp[9] = {0, 0, 0, 0, -1.0 / v, 1.0 / w, 0, 1.0 / w, -v / (w * w)};
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) Hs[i * 3 + j] = gp[i] * gp[j] / (psi * psi) - hp[i * 3 + j] / psi;
    Hs[4] += 1.0 / (v * v);
    Hs[8] += 1.0 / (w * w);
  }
  static bool inv3(const double *M, double *R) {
    double a = M[0], b = M[1], c = M[2], d = M[3], e = M[4], f = M[5], g = M[6], h = M[7], i = M[8];
    double A_ = e * i - f * h, B_ = -(d * i - f * g), C_ = d * h - e * g;
    double det = a * A_ + b * B_ + c * C_;
    if (!(std::fabs(det) > 0) || !std::isfinite(det)) return false;
    R[0] = A_ / det; R[1] = -(b * i - c * h) / det; R[2] = (b * f - c * e) / det;
    R[3] = B_ / det; R[4] = (a * i - c * g) / det; R[5] = -(a * f - c * d) / det;
    R[6] = C_ / det; R[7] = -(a * h - b * g) / det; R[8] = (a * e - b * d) / det;
    return true;
  }

  // SOC: B(w) v  (inv: B(w)^{-1} v)
  static void soc_B(const double *w, const double *v, double *out, I d, bool inv) {
    double sg = inv ? -1.0 : 1.0;
    double w0 = w[0], zeta = 0;
    for (I k = 1; k < d; k++) zeta += w[k] * v[k];
    zeta *= sg;
    double o0 = w0 * v[0] + zeta;
    double c = v[0] + zeta / (1.0 + w0);
    for (I k = 1; k < d; k++) out[k] = v[k] + c * sg * w[k];
    out[0] = o0;
  }

  // ---------- PSD helpers (NT scaling via R with R'^{-1}... ) ----------
  // W(v) for PSD: smat(v) -> R' V R ; we store R (d x d) such that  R' Z R = R^{-1} S R^{-T} = Lambda (diag)
  void psd_apply(const Cone &c, const vector<double> &R, const double *v, double *out, bool transposeR) {
    int d = (int)c.d;
    vector<double> M, T((size_t)d * d), Y((size_t)d * d);
    smat(d, v, M);
    // out = R' M R  (transposeR=false)  or  R M R' (true)
    for (int i = 0; i < d; i++)
      for (int j = 0; j < d; j++) {
        double s = 0;
        for (int k = 0; k < d; k++) s += M[(size_t)i * d + k] * (transposeR ? R[(size_t)j * d + k] : R[(size_t)k * d + j]);
        T[(size_t)i * d + j] = s;
      }
    for (int i = 0; i < d; i++)
      for (int j = 0; j < d; j++) {
        double s = 0;
        for (int k = 0; k < d; k++) s += (transposeR ? R[(size_t)i * d + k] : R[(size_t)k * d + i]) * T[(size_t)k * d + j];
        Y[(size_t)i * d + j] = s;
      }
    const double r2 = std::sqrt(2.0);
    I pos = 0;
    for (int cc = 0; cc < d; cc++)
      for (int r = 0; r <= cc; r++) { double val = 0.5 * (Y[(size_t)r * d + cc] + Y[(size_t)cc * d + r]); out[pos++] = r == cc ? val : val * r2; }
  }

  // compute scaling for all cones; returns false on failure
  bool update_scaling(const vector<double> &s, const vector<double> &z, double mu) {
    mu_scal = mu;
    for (size_t ci = 0; ci < cones.size(); ci++) {
      const Cone &c = cones[ci];
      const double *sc = &s[c.off], *zc = &z[c.off];
      if (c.type == CNN) {
        for (I k = 0; k < c.dim; k++) {
          I r = c.off + k;
          Hd[r] = s[r] / z[r];
          lam[r] = std::sqrt(s[r] * z[r]);
        }
      } else if (c.type == CSOC) {
        I d = c.dim;
        double s1 = 0, z1 = 0, sz = 0;
        for (I k = 1; k < d; k++) { s1 += sc[k] * sc[k]; z1 += zc[k] * zc[k]; }
        for (I k = 0; k < d; k++) sz += sc[k] * zc[k];
        double ss = (sc[0] - std::sqrt(s1)) * (sc[0] + std::sqrt(s1));
        double zz = (zc[0] - std::sqrt(z1)) * (zc[0] + std::sqrt(z1));
        if (!(ss > 0) || !(zz > 0)) return false;
        double sqs = std::sqrt(ss), sqz = std::sqrt(zz);
        double dt = sz / (sqs * sqz);
        double gam = std::sqrt(std::max((1.0 + dt) / 2.0, 1e-300));
        double *w = &wsoc[c.off];
        w[0] = (sc[0] / sqs + zc[0] / sqz) / (2 * gam);
        for (I k = 1; k < d; k++) w[k] = (sc[k] / sqs - zc[k] / sqz) / (2 * gam);
        // renormalize w so that w'Jw = 1
        double wn = 0; for (I k = 1; k < d; k++) wn += w[k] * w[k];
        w[0] = std::sqrt(1.0 + wn);
        double eta = std::sqrt(sqs / sqz);
        etasoc[ci] = eta;
        detlam[ci] = sqs * sqz;
        soc_B(w, zc, &lam[c.off], d, false);
        for (I k = 0; k < d; k++) lam[c.off + k] *= eta;
        double *H = &Hb[c.hoff];
        double e2 = eta * eta;
        for (I i = 0; i < d; i++)
          for (I j = 0; j < d; j++) H[i * d + j] = e2 * (2 * w[i] * w[j] + (i == j ? (i == 0 ? -1.0 : 1.0) : 0.0));
      } else if (c.type == CEXP) {
        double Hs[9], Hi[9];
        if (!exp_primal_interior(sc)) return false;
        exp_grad_hess(sc, &gexp[c.off], Hs);
        for (int k = 0; k < 9; k++) Hs[k] *= mu;
        if (!inv3(Hs, Hi)) return false;
        double *H = &Hb[c.hoff];
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) H[i * 3 + j] = 0.5 * (Hi[i * 3 + j] + Hi[j * 3 + i]);
      } else if (c.type == CPSD) {
        int d = (int)c.d;
        vector<double> S, Z, Ls, Lz;
        smat(d, sc, S); smat(d, zc, Z);
        if (!chol(d, S, Ls) || !chol(d, Z, Lz)) return false;
        // M = Lz' Ls ; SVD via eig of M'M: M = U diag(sig) V'
        vector<double> Mx((size_t)d * d, 0.0);
        for (int i = 0; i < d; i++)
          for (int j = 0; j < d; j++) {
            double acc = 0;
            for (int k = 0; k < d; k++) acc += Lz[(size_t)k * d + i] * Ls[(size_t)k * d + j];
            Mx[(size_t)i * d + j] = acc;
          }
        vector<double> MtM((size_t)d * d, 0.0);
        for (int i = 0; i < d; i++)
          for (int j = 0; j < d; j++) {
            double acc = 0;
            for (int k = 0; k < d; k++) acc += Mx[(size_t)k * d + i] * Mx[(size_t)k * d + j];
            MtM[(size_t)i * d + j] = acc;
          }
        vector<double> ev, V;
        jacobi_eig(d, MtM, ev, V);
        vector<double> sig(d);
        for (int i = 0; i < d; i++) { if (!(ev[i] > 0)) return false; sig[i] = std::sqrt(ev[i]); }
        // R = Ls V diag(sig^{-1/2});  then R' Z R = diag(sig) = R^{-1} S R^{-T}
        vector<double> R((size_t)d * d), Rinv((size_t)d * d);
        for (int i = 0; i < d; i++)
          for (int j = 0; j < d; j++) {
            double acc = 0;
            for (int k = 0; k < d; k++) acc += Ls[(size_t)i * d + k] * V[(size_t)k * d + j];
            R[(size_t)i * d + j] = acc / std::sqrt(sig[j]);
          }
        // Rinv = diag(sig^{1/2}) V' Ls^{-1} ; compute via solving
        // Rinv = diag(sig^{-1/2}) U' Lz'  where U = M V diag(1/sig)
        vector<double> U((size_t)d * d);
        for (int i = 0; i < d; i++)
          for (int j = 0; j < d; j++) {
            double acc = 0;
            for (int k = 0; k < d; k++) acc += Mx[(size_t)i * d + k] * V[(size_t)k * d + j];
            U[(size_t)i * d + j] = acc / sig[j];
          }
        for (int i = 0; i < d; i++)
          for (int j = 0; j < d; j++) {
            double acc = 0;
            for (int k = 0; k < d; k++) acc += U[(size_t)k * d + i] * Lz[(size_t)j * d + k];
            Rinv[(size_t)i * d + j] = acc / std::sqrt(sig[i]);
          }
        psdR[ci] = R; psdRinv[ci] = Rinv; psdLam[ci] = sig;
        // lambda vector (svec of diag(sig))
        I pos = 0;
        for (int cc = 0; cc < d; cc++) for (int r = 0; r <= cc; r++) lam[c.off + pos++] = (r == cc) ? sig[cc] : 0.0;
        // H = W W' where W(v) = svec(R smat(v) R')  (ds = -H dz):  H(v) = svec(R R' smat(v) R R')
        I dim = c.dim;
        double *H = &Hb[c.hoff];
        vector<double> RRt((size_t)d * d);
        for (int i = 0; i < d; i++)
          for (int j = 0; j < d; j++) { double acc = 0; for (int k = 0; k < d; k++) acc += R[(size_t)i * d + k] * R[(size_t)j * d + k]; RRt[(size_t)i * d + j] = acc; }
        // H entries: for basis E_ab (svec), H(E_ab) = svec(RRt E_ab RRt)
        const double r2 = std::sqrt(2.0);
        vector<int> ia(dim), ib(dim);
        { I p2 = 0; for (int cc = 0; cc < d; cc++) for (int r = 0; r <= cc; r++) { ia[p2] = r; ib[p2] = cc; p2++; } }
        for (I x1 = 0; x1 < dim; x1++) {
          int a = ia[x1], bq = ib[x1];
          for (I x2 = 0; x2 <= x1; x2++) {
            int cq = ia[x2], dq = ib[x2];
            double val;
            // <E_x2, RRt E_x1 RRt> with svec normalization
            if (a == bq && cq == dq) val = RRt[(size_t)cq * d + a] * RRt[(size_t)a * d + cq];
            else if (a == bq) val = r2 * RRt[(size_t)cq * d + a] * RRt[(size_t)a * d + dq];
            else if (cq == dq) val = r2 * RRt[(size_t)cq * d + a] * RRt[(size_t)bq * d + cq];
            else val = RRt[(size_t)cq * d + a] * RRt[(size_t)bq * d + dq] + RRt[(size_t)cq * d + bq] * RRt[(size_t)a * d + dq];
            H[x1 * dim + x2] = val; H[x2 * dim + x1] = val;
          }
        }
      }
    }
    return true;
  }
  static bool chol(int d, const vector<double> &A, vector<double> &L) {
    L.assign((size_t)d * d, 0.0);
    for (int j = 0; j < d; j++) {
      double s = A[(size_t)j * d + j];
      for (int k = 0; k < j; k++) s -= L[(size_t)j * d + k] * L[(size_t)j * d + k];
      if (!(s > 0)) return false;
      double ljj = std::sqrt(s);
      L[(size_t)j * d + j] = ljj;
      for (int i = j + 1; i < d; i++) {
        double t = A[(size_t)i * d + j];
        for (int k = 0; k < j; k++) t -= L[(size_t)i * d + k] * L[(size_t)j * d + k];
        L[(size_t)i * d + j] = t / ljj;
      }
    }
    return true;
  }

  // out = H v
  void Hmul(const double *v, double *out) {
    for (const Cone &c : cones) {
      if (c.type == CZERO) { for (I k = 0; k < c.dim; k++) out[c.off + k] = 0.0; }
      else if (c.type == CNN) { for (I k = 0; k < c.dim; k++) out[c.off + k] = Hd[c.off + k] * v[c.off + k]; }
      else {
        I d = c.dim;
        const double *H = &Hb[c.hoff];
        for (I i = 0; i < d; i++) { double s = 0; for (I j = 0; j < d; j++) s += H[i * d + j] * v[c.off + j]; out[c.off + i] = s; }
      }
    }
  }

  // W v and W^{-1} v for symmetric cones (W z = W^{-1} s = lambda)
  void Wmul(const Cone &c, size_t ci, const double *v, double *out, bool inv) {
    if (c.type == CNN) {
      for (I k = 0; k < c.dim; k++) { double w = std::sqrt(Hd[c.off + k]); out[k] = inv ? v[k] / w : v[k] * w; }
    } else if (c.type == CSOC) {
      soc_B(&wsoc[c.off], v, out, c.dim, inv);
      double eta = etasoc[ci];
      for (I k = 0; k < c.dim; k++) out[k] = inv ? out[k] / eta : out[k] * eta;
    } else if (c.type == CPSD) {
      // W(v) = svec(R smat(v) R') ; W^{-1}(v) = svec(Rinv' ... ) : with R'ZR = Lam: W z must equal lambda => W(v)=svec(R' V R)
      if (!inv) psd_apply(c, psdR[ci], v, out, false);
      else psd_apply(c, psdRinv[ci], v, out, true);
    }
  }
  // Jordan product u o v
  void jprod(const Cone &c, const double *u, const double *v, double *out) {
    if (c.type == CNN) { for (I k = 0; k < c.dim; k++) out[k] = u[k] * v[k]; }
    else if (c.type == CSOC) {
      double s = 0; for (I k = 0; k < c.dim; k++) s += u[k] * v[k];
      for (I k = 1; k < c.dim; k++) out[k] = u[0] * v[k] + v[0] * u[k];
      out[0] = s;
    } else if (c.type == CPSD) {
      int d = (int)c.d;
      vector<double> U, V2, Y((size_t)d * d);
      smat(d, u, U); smat(d, v, V2);
      for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++) { double acc = 0; for (int k = 0; k < d; k++) acc += U[(size_t)i * d + k] * V2[(size_t)k * d + j]; Y[(size_t)i * d + j] = acc; }
      const double r2 = std::sqrt(2.0);
      I pos = 0;
      for (int cc = 0; cc < d; cc++) for (int r = 0; r <= cc; r++) { double val = 0.5 * (Y[(size_t)r * d + cc] + Y[(size_t)cc * d + r]); out[pos++] = r == cc ? val : val * r2; }
    }
  }
  // solve lambda o x = r
  void jdiv(const Cone &c, size_t ci, const double *lm, const double *r, double *out) {
    if (c.type == CNN) { for (I k = 0; k < c.dim; k++) out[k] = r[k] / lm[k]; }
    else if (c.type == CSOC) {
      double l1r1 = 0, l1l1 = 0;
      for (I k = 1; k < c.dim; k++) { l1r1 += lm[k] * r[k]; l1l1 += lm[k] * lm[k]; }
      double det = detlam[ci];
      (void)l1l1;
      double x0 = (lm[0] * r[0] - l1r1) / det;
      for (I k = 1; k < c.dim; k++) out[k] = (r[k] - x0 * lm[k]) / lm[0];
      out[0] = x0;
    } else if (c.type == CPSD) {
      // lambda = diag(sig): (Lam X + X Lam)/2 = R  => X_ij = 2 R_ij/(sig_i+sig_j)
      int d = (int)c.d;
      const vector<double> &sg = psdLam[ci];
      I pos = 0;
      for (int cc = 0; cc < d; cc++) for (int rr = 0; rr <= cc; rr++) { out[pos] = 2.0 * r[pos] / (sg[rr] + sg[cc]); pos++; }
    }
  }

  // ---------- KKT factor / solve ----------
  bool factor() {
    // values in original order
    for (I j = 0; j < n; j++) Kx[diagX[j]] = Pdiag[j] + sreg;
    for (const Cone &c : cones) {
      if (c.type == CZERO) { for (I k = 0; k < c.dim; k++) Kx[zbase[c.off + k]] = -sreg; }
      else if (c.type == CNN) { for (I k = 0; k < c.dim; k++) Kx[zbase[c.off + k]] = -Hd[c.off + k] - sreg; }
      else {
        I d = c.dim;
        const double *H = &Hb[c.hoff];
        for (I r = 0; r < d; r++) {
          I base = zbase[c.off + r];
          for (I rr = 0; rr <= r; rr++) Kx[base + rr] = -H[rr * d + r] - (rr == r ? sreg : 0.0);
        }
      }
    }
    I nnz = Kp[N];
    for (I k = 0; k < nnz; k++) KPx[kmap[k]] = Kx[k];
    double tf0 = now();
    ldl.factor(KPp, KPi, KPx, sgn, 1e-13, DYN_DELTA);
    T_FACT += now() - tf0;
    for (I k = 0; k < N; k++) if (!std::isfinite(ldl.D[k])) return false;
    return true;
  }
  // y = K_true v
  void kmul(const double *v, double *y, vector<double> &tmp) {
    const double *vx = v, *vz = v + n;
    double *yx = y, *yz = y + n;
    for (I j = 0; j < n; j++) yx[j] = 0.0;
    for (I i = 0; i < m; i++) yz[i] = 0.0;
    spmv_add(P, vx, yx);
    spmtv_add(A, vz, yx);
    spmv_add(A, vx, yz);
    tmp.resize(m);
    Hmul(vz, tmp.data());
    for (I i = 0; i < m; i++) yz[i] -= tmp[i];
  }
  vector<double> wk1, wk2, wk3, wk4;
  void solve(const vector<double> &rhs, vector<double> &sol) {
    double ts0 = now();
    solve_(rhs, sol);
    T_SOLVE += now() - ts0;
  }
  void ldls(double *v) { double t0 = now(); ldl.solve(v); T_LDLS += now() - t0; N_LDLS++; }
  void solve_(const vector<double> &rhs, vector<double> &sol) {
    sol.resize(N);
    wk1.resize(N);
    for (I k = 0; k < N; k++) wk1[k] = rhs[perm[k]];
    ldls(wk1.data());
    for (I k = 0; k < N; k++) sol[perm[k]] = wk1[k];
    // iterative refinement against unregularized K; only improving corrections are kept
    double nb = nrminf(rhs.data(), N);
    wk2.resize(N); wk3.resize(N);
    kmul(sol.data(), wk2.data(), wk4);
    for (I k = 0; k < N; k++) wk2[k] = rhs[k] - wk2[k];
    double rn = nrminf(wk2.data(), N);
    vector<double> &cand = refcand;
    cand.resize(N);
    for (int it = 0; it < REF_MAX; it++) {
      if (!(rn > 1e-13 * (1 + nb))) break;
      for (I k = 0; k < N; k++) wk1[k] = wk2[perm[k]];
      ldls(wk1.data());
      for (I k = 0; k < N; k++) cand[perm[k]] = sol[perm[k]] + wk1[k];
      kmul(cand.data(), wk3.data(), wk4);
      for (I k = 0; k < N; k++) wk3[k] = rhs[k] - wk3[k];
      double rn2 = nrminf(wk3.data(), N);
      if (!(rn2 < rn)) break;
      sol.swap(cand);
      wk2.swap(wk3);
      bool small_gain = !(rn2 < rn / REF_RATIO);
      rn = rn2;
      if (small_gain) break;
    }
  }
  vector<double> refcand;

  // ---------- step length ----------
  double soc_step(const double *x, const double *dx, I d, double amax) {
    double a = dx[0] * dx[0], bb = x[0] * dx[0], cc = x[0] * x[0];
    for (I k = 1; k < d; k++) { a -= dx[k] * dx[k]; bb -= x[k] * dx[k]; cc -= x[k] * x[k]; }
    bb *= 2;
    cc = std::max(cc, 0.0);
    // smallest positive root of a t^2 + bb t + cc
    double tmax = amax;
    if (a == 0) { if (bb < 0) tmax = std::min(tmax, -cc / bb); }
    else {
      double disc = bb * bb - 4 * a * cc;
      if (disc >= 0) {
        double sq = std::sqrt(disc);
        double qq = -0.5 * (bb + (bb >= 0 ? sq : -sq));
        double r1 = qq / a, r2 = qq != 0 ? cc / qq : -1;
        if (r1 > 0) tmax = std::min(tmax, r1);
        if (r2 > 0) tmax = std::min(tmax, r2);
      }
    }
    // also ensure x0 + t dx0 >= 0
    if (dx[0] < 0) tmax = std::min(tmax, -x[0] / dx[0]);
    return std::max(tmax, 0.0);
  }
  double psd_step(const Cone &c, const double *x, const double *dx, double amax) {
    // max t s.t. X + t dX psd: X = L L', eig of L^{-1} dX L^{-T}
    int d = (int)c.d;
    vector<double> X, DX, L;
    smat(d, x, X); smat(d, dx, DX);
    if (!chol(d, X, L)) return 0.0;
    // Y = L^{-1} DX L^{-T}
    vector<double> T((size_t)d * d), Y((size_t)d * d);
    for (int j = 0; j < d; j++)
      for (int i = 0; i < d; i++) {
        double s = DX[(size_t)i * d + j];
        for (int k = 0; k < i; k++) s -= L[(size_t)i * d + k] * T[(size_t)k * d + j];
        T[(size_t)i * d + j] = s / L[(size_t)i * d + i];
      }
    for (int i = 0; i < d; i++)
      for (int j = 0; j < d; j++) {
        double s = T[(size_t)i * d + j];
        for (int k = 0; k < j; k++) s -= L[(size_t)j * d + k] * Y[(size_t)i * d + k];
        Y[(size_t)i * d + j] = s / L[(size_t)j * d + j];
      }
    for (int i = 0; i < d; i++) for (int j = 0; j < i; j++) { double a2 = 0.5 * (Y[(size_t)i * d + j] + Y[(size_t)j * d + i]); Y[(size_t)i * d + j] = Y[(size_t)j * d + i] = a2; }
    vector<double> ev, V;
    jacobi_eig(d, Y, ev, V);
    double mn = 1e300;
    for (double e : ev) mn = std::min(mn, e);
    if (mn >= 0) return amax;
    return std::min(amax, -1.0 / mn);
  }
  double max_step(const vector<double> &s, const vector<double> &ds, const vector<double> &z, const vector<double> &dz, double amax) {
    double a = amax;
    for (const Cone &c : cones) {
      if (c.type == CNN) {
        for (I k = 0; k < c.dim; k++) {
          I r = c.off + k;
          if (ds[r] < 0) a = std::min(a, -s[r] / ds[r]);
          if (dz[r] < 0) a = std::min(a, -z[r] / dz[r]);
        }
      } else if (c.type == CSOC) {
        a = std::min(a, soc_step(&s[c.off], &ds[c.off], c.dim, a));
        a = std::min(a, soc_step(&z[c.off], &dz[c.off], c.dim, a));
      } else if (c.type == CPSD) {
        a = std::min(a, psd_step(c, &s[c.off], &ds[c.off], a));
        a = std::min(a, psd_step(c, &z[c.off], &dz[c.off], a));
      }
    }
    if (has_exp) {
      for (const Cone &c : cones) {
        if (c.type != CEXP) continue;
        double t[3], u[3];
        for (int it = 0; it < 80; it++) {
          for (int k = 0; k < 3; k++) { t[k] = s[c.off + k] + a * ds[c.off + k]; u[k] = z[c.off + k] + a * dz[c.off + k]; }
          if (exp_primal_interior(t) && exp_dual_interior(u)) break;
          a *= 0.8;
        }
      }
    }
    return a;
  }

  // centrality check for exp cones (neighborhood)
  bool exp_central(const vector<double> &s, const vector<double> &z, double mu, double beta) {
    for (const Cone &c : cones) {
      if (c.type != CEXP) continue;
      double sz = s[c.off] * z[c.off] + s[c.off + 1] * z[c.off + 1] + s[c.off + 2] * z[c.off + 2];
      if (sz < beta * 3 * mu) return false;
    }
    return true;
  }

  // ---------- main solve ----------
  Result run() {
    Result res;
    equilibrate();
    alloc_cones();
    build_kkt();
    vector<double> x(n, 0.0), s(m, 0.0), z(m, 0.0);
    double tau = 1, kap = 1;
    // ---- initial point ----
    bool unit_init = has_exp;
    if (!unit_init) {
      // H = I for non-zero cones, factor [P A'; A -I]
      for (const Cone &c : cones) {
        if (c.type == CNN) for (I k = 0; k < c.dim; k++) Hd[c.off + k] = 1.0;
        else if (c.type != CZERO) {
          double *H = &Hb[c.hoff];
          for (I i = 0; i < c.dim; i++) for (I j = 0; j < c.dim; j++) H[i * c.dim + j] = i == j ? 1.0 : 0.0;
        }
      }
      if (!factor()) unit_init = true;
      else {
        vector<double> rhs(N, 0.0), sol;
        if (P.nnz() == 0) {
          for (I i = 0; i < m; i++) rhs[n + i] = b[i];
          solve(rhs, sol);
          for (I j = 0; j < n; j++) x[j] = sol[j];
          for (I i = 0; i < m; i++) s[i] = -sol[n + i];
          std::fill(rhs.begin(), rhs.end(), 0.0);
          for (I j = 0; j < n; j++) rhs[j] = -q[j];
          solve(rhs, sol);
          for (I i = 0; i < m; i++) z[i] = sol[n + i];
        } else {
          for (I j = 0; j < n; j++) rhs[j] = -q[j];
          for (I i = 0; i < m; i++) rhs[n + i] = b[i];
          solve(rhs, sol);
          for (I j = 0; j < n; j++) x[j] = sol[j];
          for (I i = 0; i < m; i++) { z[i] = sol[n + i]; s[i] = -sol[n + i]; }
        }
        // shift into cones
        for (const Cone &c : cones) {
          if (c.type == CZERO) { for (I k = 0; k < c.dim; k++) s[c.off + k] = 0.0; continue; }
          for (int side = 0; side < 2; side++) {
            double *v = side == 0 ? &s[c.off] : &z[c.off];
            if (c.type == CNN) {
              double mn = 1e300; for (I k = 0; k < c.dim; k++) mn = std::min(mn, v[k]);
              if (mn < 1e-8) { for (I k = 0; k < c.dim; k++) v[k] += (1.0 - mn); }
            } else if (c.type == CSOC) {
              double t = 0; for (I k = 1; k < c.dim; k++) t += v[k] * v[k];
              double al = v[0] - std::sqrt(t);
              if (al < 1e-8) v[0] += (1.0 - al);
            } else if (c.type == CPSD) {
              double mn = psd_min_eig((int)c.d, v);
              if (mn < 1e-8) { I pos = 0; for (int cc = 0; cc < c.d; cc++) for (int r = 0; r <= cc; r++) { if (r == cc) v[pos] += (1.0 - mn); pos++; } }
            }
          }
        }
        for (double v : x) if (!std::isfinite(v)) unit_init = true;
        for (I i = 0; i < m; i++) if (!std::isfinite(s[i]) || !std::isfinite(z[i])) unit_init = true;
      }
    }
    if (unit_init) {
      std::fill(x.begin(), x.end(), 0.0);
      for (const Cone &c : cones) {
        double *sv = &s[c.off], *zv = &z[c.off];
        if (c.type == CZERO) for (I k = 0; k < c.dim; k++) { sv[k] = 0; zv[k] = 0; }
        else if (c.type == CNN) for (I k = 0; k < c.dim; k++) { sv[k] = 1; zv[k] = 1; }
        else if (c.type == CSOC) { for (I k = 0; k < c.dim; k++) { sv[k] = 0; zv[k] = 0; } sv[0] = zv[0] = 1; }
        else if (c.type == CEXP) {
          sv[0] = zv[0] = -1.051383945322714; sv[1] = zv[1] = 0.556409619469370; sv[2] = zv[2] = 1.258967884768947;
        } else { I pos = 0; for (int cc = 0; cc < c.d; cc++) for (int r = 0; r <= cc; r++) { sv[pos] = zv[pos] = (r == cc) ? 1.0 : 0.0; pos++; } }
      }
    }

    // ---- iterate ----
    vector<double> rx(n), rz(m), Px(n), Ax(m), Atz(n), x1, z1, rhs(N), sol, dx(n), dz(m), ds(m), dxa, dza, dsa, cs(m), tmp(m), tmp2(m), tmp3(std::max(n, m)),
        Px1(n);
    vector<double> bx(n), bs(m), bz(m), cx(n), csv(m), cz(m);  // unscaled candidates
    double best_score = 1e300;
    Result best; best.status = "unresolved";
    double best_pinf = 1e300, best_dinf = 1e300;
    vector<double> pinf_z, dinf_x, dinf_s;
    int maxit = 200;
    double nb_orig = nrminf(pr.b.data(), m), nq_orig = nrminf(pr.q.data(), n);
    (void)nb_orig; (void)nq_orig;
    int stall = 0;
    double prev_score = 1e300;
    for (int iter = 0; iter < maxit; iter++) {
      if (now() - t_start > TIME_CAP) break;
      // residuals
      std::fill(Px.begin(), Px.end(), 0.0); spmv_add(P, x.data(), Px.data());
      std::fill(Ax.begin(), Ax.end(), 0.0); spmv_add(A, x.data(), Ax.data());
      std::fill(Atz.begin(), Atz.end(), 0.0); spmtv_add(A, z.data(), Atz.data());
      for (I j = 0; j < n; j++) rx[j] = Px[j] + Atz[j] + q[j] * tau;
      for (I i = 0; i < m; i++) rz[i] = Ax[i] + s[i] - b[i] * tau;
      double xPx = dotp(x.data(), Px.data(), n), qx = dotp(q.data(), x.data(), n), bz = dotp(b.data(), z.data(), m);
      double rtau = qx + bz + xPx / tau + kap;
      double sz = dotp(s.data(), z.data(), m);
      double mu = (sz + tau * kap) / (nu + 1);
      // ---- termination checks on original problem ----
      for (I j = 0; j < n; j++) cx[j] = D[j] * x[j] / tau;
      for (I i = 0; i < m; i++) { csv[i] = s[i] / E[i] / tau; cz[i] = E[i] * z[i] / (cscale * tau); }
      for (const Cone &c : cones) if (c.type == CZERO) for (I k = 0; k < c.dim; k++) csv[c.off + k] = 0.0;
      Metrics M = eval_solved(pr, cx, csv, cz, 1e-8, false);
      if (VERB) fprintf(stderr, "it %3d mu %.2e tau %.2e kap %.2e rp %.2e rd %.2e gap %.2e/%.2e  t=%.3f\n", iter, mu, tau, kap, M.rp, M.rd, M.gap_abs,
                        M.gap_rel, now() - t_start);
      if (std::isfinite(M.score) && M.score < best_score) {
        best_score = M.score;
        best.status = "solved"; best.x = cx; best.s = csv; best.z = cz;
      }
      if (M.score <= 5e-9 || (M.score <= 9e-9 && stall > 0)) {
        Metrics M2 = eval_solved(pr, cx, csv, cz, 9.5e-9, true);
        if (M2.ok) break;
      }
      if (M.score > 0.9 * prev_score) stall++; else stall = 0;
      prev_score = std::min(prev_score, M.score);
      // infeasibility
      if (bz < 0) {
        vector<double> zz(m);
        for (I i = 0; i < m; i++) zz[i] = E[i] * z[i];
        double r = eval_pinf(pr, zz);
        if (r < best_pinf) { best_pinf = r; pinf_z = zz; }
        if (r <= 5e-9) break;
      }
      if (qx < 0) {
        vector<double> xx(n), ss(m);
        for (I j = 0; j < n; j++) xx[j] = D[j] * x[j];
        for (I i = 0; i < m; i++) ss[i] = s[i] / E[i];
        for (const Cone &c : cones) if (c.type == CZERO) for (I k = 0; k < c.dim; k++) ss[c.off + k] = 0.0;
        double r = eval_dinf(pr, xx, ss);
        if (r < best_dinf) { best_dinf = r; dinf_x = xx; dinf_s = ss; }
        if (r <= 5e-9) break;
        // try s = -Ax variant
        vector<double> s2(m, 0.0);
        spmv_add(pr.A, xx.data(), s2.data());
        for (auto &v : s2) v = -v;
        double r2 = eval_dinf(pr, xx, s2);
        if (r2 < best_dinf) { best_dinf = r2; dinf_x = xx; dinf_s = s2; }
        if (r2 <= 5e-9) break;
      }
      if (!std::isfinite(mu) || mu < 1e-300) break;
      // ---- scaling & factorization ----
      if (!update_scaling(s, z, mu)) { if (VERB) fprintf(stderr, "scaling failed\n"); break; }
      if (!factor()) { if (VERB) fprintf(stderr, "factor failed\n"); break; }
      // constant solve
      for (I j = 0; j < n; j++) rhs[j] = -q[j];
      for (I i = 0; i < m; i++) rhs[n + i] = b[i];
      solve(rhs, sol);
      x1.assign(sol.begin(), sol.begin() + n);
      z1.assign(sol.begin() + n, sol.end());
      // denominator
      for (I j = 0; j < n; j++) tmp3[j] = x1[j] - x[j] / tau;
      std::fill(Px1.begin(), Px1.end(), 0.0);
      spmv_add(P, tmp3.data(), Px1.data());
      double xPx1 = dotp(tmp3.data(), Px1.data(), n);
      Hmul(z1.data(), tmp.data());
      double zHz = dotp(z1.data(), tmp.data(), m);
      double den = -(xPx1 + zHz + kap / tau);
      vector<double> qPx(n);
      for (I j = 0; j < n; j++) qPx[j] = q[j] + 2 * Px[j] / tau;

      double sigma = 0, dtau_a = 0, dkap_a = 0;
      bool badstep = false;
      double alpha = 0;
      for (int pass = 0; pass < 2; pass++) {
        bool affine = (pass == 0);
        double eta = affine ? 1.0 : (1.0 - sigma);
        double smu = affine ? 0.0 : sigma * mu;
        // complementarity rhs -> cs
        for (size_t ci = 0; ci < cones.size(); ci++) {
          const Cone &c = cones[ci];
          double *o = &cs[c.off];
          if (c.type == CZERO) { for (I k = 0; k < c.dim; k++) o[k] = 0.0; }
          else if (c.type == CEXP) {
            // primal scaling: ds + H dz = -(z + smu*g) mapped: cs = -H (z + smu g)
            double v3[3];
            for (int k = 0; k < 3; k++) v3[k] = z[c.off + k] + smu * gexp[c.off + k];
            const double *H = &Hb[c.hoff];
            for (int i = 0; i < 3; i++) o[i] = -(H[i * 3] * v3[0] + H[i * 3 + 1] * v3[1] + H[i * 3 + 2] * v3[2]);
          } else if (affine) {
            for (I k = 0; k < c.dim; k++) o[k] = -s[c.off + k];
          } else {
            I d = c.dim;
            vector<double> r(d), a1(d), a2(d), lsq(d), tmpv(d);
            Wmul(c, ci, &dsa[c.off], a1.data(), true);
            Wmul(c, ci, &dza[c.off], a2.data(), false);
            jprod(c, a1.data(), a2.data(), r.data());
            jprod(c, &lam[c.off], &lam[c.off], lsq.data());
            for (I k = 0; k < d; k++) r[k] = -lsq[k] - r[k];
            // + smu * e
            if (c.type == CNN) for (I k = 0; k < d; k++) r[k] += smu;
            else if (c.type == CSOC) r[0] += smu;
            else { I pos = 0; for (int cc = 0; cc < c.d; cc++) for (int rr = 0; rr <= cc; rr++) { if (rr == cc) r[pos] += smu; pos++; } }
            jdiv(c, ci, &lam[c.off], r.data(), tmpv.data());
            if (c.type == CPSD) psd_apply(c, psdR[ci], tmpv.data(), o, true);
            else Wmul(c, ci, tmpv.data(), o, false);
          }
        }
        for (I j = 0; j < n; j++) rhs[j] = -eta * rx[j];
        for (I i = 0; i < m; i++) rhs[n + i] = -eta * rz[i] - cs[i];
        solve(rhs, sol);
        double rk = affine ? tau * kap : (tau * kap + dtau_a * dkap_a - smu);
        double num = -eta * rtau + rk / tau;
        for (I j = 0; j < n; j++) num -= qPx[j] * sol[j];
        for (I i = 0; i < m; i++) num -= b[i] * sol[n + i];
        double dtau = num / den;
        for (I j = 0; j < n; j++) dx[j] = sol[j] + dtau * x1[j];
        for (I i = 0; i < m; i++) dz[i] = sol[n + i] + dtau * z1[i];
        Hmul(dz.data(), tmp.data());
        for (I i = 0; i < m; i++) ds[i] = -tmp[i] + cs[i];
        for (const Cone &c : cones) if (c.type == CZERO) for (I k = 0; k < c.dim; k++) ds[c.off + k] = 0.0;
        double dkap = (-rk - kap * dtau) / tau;
        {
          bool bad = !std::isfinite(dtau) || !std::isfinite(dkap);
          for (I j = 0; j < n && !bad; j++) if (!std::isfinite(dx[j])) bad = true;
          for (I i = 0; i < m && !bad; i++) if (!std::isfinite(dz[i]) || !std::isfinite(ds[i])) bad = true;
          if (bad) { badstep = true; break; }
        }
        double amax = 1.0 / 0.99;
        if (!affine) amax = 1.0;
        double a = max_step(s, ds, z, dz, 1.0);
        if (dtau < 0) a = std::min(a, -tau / dtau);
        if (dkap < 0) a = std::min(a, -kap / dkap);
        if (has_exp) a = max_step(s, ds, z, dz, a);
        if (affine) {
          double aa = std::min(1.0, a);
          sigma = std::pow(1.0 - aa, 3);
          sigma = std::min(1.0, std::max(sigma, SIGMA_MIN));
          dxa = dx; dza = dz; dsa = ds; dtau_a = dtau; dkap_a = dkap;
          (void)amax;
        } else {
          alpha = std::min(1.0, 0.99 * a);
          if (has_exp) {
            // backtrack to keep exp cones central
            for (int bt = 0; bt < 30; bt++) {
              vector<double> sn(s), zn(z);
              for (I i = 0; i < m; i++) { sn[i] += alpha * ds[i]; zn[i] += alpha * dz[i]; }
              double taun = tau + alpha * dtau, kapn = kap + alpha * dkap;
              double mun = (dotp(sn.data(), zn.data(), m) + taun * kapn) / (nu + 1);
              if (exp_central(sn, zn, mun, EXP_BETA)) break;
              alpha *= 0.8;
            }
          }
          for (I j = 0; j < n; j++) x[j] += alpha * dx[j];
          for (I i = 0; i < m; i++) { s[i] += alpha * ds[i]; z[i] += alpha * dz[i]; }
          tau += alpha * dtau;
          kap += alpha * dkap;
        }
      }
      if (badstep) { if (VERB) fprintf(stderr, "non-finite direction\n"); break; }
      if (VERB > 1) fprintf(stderr, "   sigma %.3e alpha %.3e\n", sigma, alpha);
      if (alpha < 1e-10) { stall += 3; }
      if (stall > 40) break;
      // renormalize to avoid overflow
      if (tau + kap > 1e8 || tau + kap < 1e-8) {
        // no-op: homogeneous scaling is allowed but skip for simplicity
      }
    }
    // ---- choose result ----
    Metrics MB = best.x.empty() ? Metrics{1, 1, 1, 1, 1, 1, false, 1e300} : eval_solved(pr, best.x, best.s, best.z, 1e-8, true);
    bool solved_ok = !best.x.empty() && MB.score <= 9.9e-9;
    if (solved_ok) return best;
    if (best_pinf <= 9.9e-9 || best_dinf <= 9.9e-9) {
      Result r;
      if (best_pinf <= best_dinf) { r.status = "primal_infeasible"; r.z = pinf_z; }
      else { r.status = "dual_infeasible"; r.x = dinf_x; r.s = dinf_s; }
      return r;
    }
    // best effort
    double sbest = best.x.empty() ? 1e300 : MB.score;
    if (sbest <= best_pinf && sbest <= best_dinf && !best.x.empty()) return best;
    Result r;
    if (best_pinf <= best_dinf && !pinf_z.empty()) { r.status = "primal_infeasible"; r.z = pinf_z; return r; }
    if (!dinf_x.empty()) { r.status = "dual_infeasible"; r.x = dinf_x; r.s = dinf_s; return r; }
    if (!best.x.empty()) return best;
    r.status = "unresolved";
    return r;
  }
};

static string fmt_double(double v) {
  char buf[64];
  snprintf(buf, sizeof buf, "%.17g", v);
  return buf;
}

int main(int argc, char **argv) {
  t_start = now();
  string in, out;
  for (int i = 1; i < argc; i++) {
    string a = argv[i];
    if (a == "--input" && i + 1 < argc) in = argv[++i];
    else if (a == "--output" && i + 1 < argc) out = argv[++i];
    else if (a.rfind("--input=", 0) == 0) in = a.substr(8);
    else if (a.rfind("--output=", 0) == 0) out = a.substr(9);
  }
  if (getenv("CONIC_VERBOSE")) VERB = atoi(getenv("CONIC_VERBOSE"));
  if (getenv("CONIC_TIME_CAP")) TIME_CAP = atof(getenv("CONIC_TIME_CAP"));
  if (getenv("CONIC_SIGMA_MIN")) SIGMA_MIN = atof(getenv("CONIC_SIGMA_MIN"));
  if (getenv("CONIC_REF_MAX")) REF_MAX = atoi(getenv("CONIC_REF_MAX"));
  if (getenv("CONIC_REF_RATIO")) REF_RATIO = atof(getenv("CONIC_REF_RATIO"));
  if (getenv("CONIC_SREG")) SREG = atof(getenv("CONIC_SREG"));
  if (getenv("CONIC_DYN")) DYN_DELTA = atof(getenv("CONIC_DYN"));
  if (getenv("CONIC_EXP_BETA")) EXP_BETA = atof(getenv("CONIC_EXP_BETA"));
  Result res;
  res.status = "unresolved";
  Problem pr;
  bool loaded = false;
  try {
    std::map<string, NpyArr> arrs;
    read_npz(in.c_str(), arrs);
    string meta = npy_unicode(arrs.at("metadata_json"));
    JParser jp(meta);
    JV root = jp.parse();
    pr.q = npy_f64(arrs.at("q"));
    pr.b = npy_f64(arrs.at("b"));
    pr.n = (I)pr.q.size(); pr.m = (I)pr.b.size();
    for (const char *pre : {"P", "A"}) {
      string p(pre);
      CSC M;
      vector<I> shp = npy_int(arrs.at(p + "_shape"));
      M.m = shp[0]; M.n = shp[1];
      M.p = npy_int(arrs.at(p + "_indptr"));
      M.i = npy_int(arrs.at(p + "_indices"));
      M.x = npy_f64(arrs.at(p + "_data"));
      if ((I)M.p.size() != M.n + 1) throw std::runtime_error("bad indptr");
      M = canonical(M);
      if (p == "P") pr.P = std::move(M); else pr.A = std::move(M);
    }
    arrs.clear();
    const JV *off = root.get("objective_offset");
    if (off && off->t == 2) pr.c0 = off->num;
    const JV *cl = root.get("cones");
    I pos = 0;
    if (cl) for (const JV &c : cl->arr) {
      Cone cc;
      string t = c.get("type")->str;
      I dim = (I)c.get("dim")->num;
      cc.off = pos; cc.d = dim; cc.hoff = 0;
      if (t == "zero") { cc.type = CZERO; cc.dim = dim; }
      else if (t == "nonnegative") { cc.type = CNN; cc.dim = dim; }
      else if (t == "soc") { cc.type = CSOC; cc.dim = dim; }
      else if (t == "exponential") { cc.type = CEXP; cc.dim = 3; }
      else { cc.type = CPSD; cc.dim = dim * (dim + 1) / 2; }
      pos += cc.dim;
      pr.cones.push_back(cc);
    }
    if (pos != pr.m) throw std::runtime_error("cone dims mismatch");
    loaded = true;
    if (VERB) fprintf(stderr, "loaded n=%lld m=%lld t=%.3f\n", (long long)pr.n, (long long)pr.m, now() - t_start);
  } catch (std::exception &e) {
    fprintf(stderr, "load error: %s\n", e.what());
  }
  if (loaded) {
    try {
      if (pr.m == 0) {
        // unconstrained: only handle trivial case P x = -q via IPM anyway
      }
      IPM ipm(pr);
      res = ipm.run();
    } catch (std::exception &e) {
      fprintf(stderr, "solve error: %s\n", e.what());
      res.status = "unresolved";
    }
  }
  // sanitize
  auto finite = [](const vector<double> &v) { for (double x : v) if (!std::isfinite(x)) return false; return true; };
  if (!finite(res.x) || !finite(res.s) || !finite(res.z)) { res = Result(); res.status = "unresolved"; }
  string js = "{\"schema_version\": 1, \"status\": \"" + res.status + "\"";
  if (res.status == "solved") {
    // objective values
    vector<double> Px(pr.n, 0.0);
    spmv_add(pr.P, res.x.data(), Px.data());
    double xPx = dotp(res.x.data(), Px.data(), pr.n);
    double p0 = 0.5 * xPx + dotp(pr.q.data(), res.x.data(), pr.n);
    double d0 = -0.5 * xPx - dotp(pr.b.data(), res.z.data(), pr.m);
    if (std::isfinite(p0 + pr.c0) && std::isfinite(d0 + pr.c0))
      js += ", \"objective_primal\": " + fmt_double(p0 + pr.c0) + ", \"objective_dual\": " + fmt_double(d0 + pr.c0);
  }
  js += "}";
  vector<std::pair<string, vector<unsigned char>>> files;
  files.emplace_back("metadata_json", npy_str(js));
  if (res.status == "solved") {
    files.emplace_back("x", npy_vec(res.x)); files.emplace_back("s", npy_vec(res.s)); files.emplace_back("z", npy_vec(res.z));
  } else if (res.status == "primal_infeasible") {
    files.emplace_back("z", npy_vec(res.z));
  } else if (res.status == "dual_infeasible") {
    files.emplace_back("x", npy_vec(res.x)); files.emplace_back("s", npy_vec(res.s));
  }
  if (!write_npz(out.c_str(), files)) { fprintf(stderr, "write failed\n"); }
  if (VERB) fprintf(stderr, "status %s total %.3fs fact %.3f solve %.3f ldls %.3f (%ld)\n", res.status.c_str(), now() - t_start, T_FACT, T_SOLVE, T_LDLS, N_LDLS);
  return 0;
}
