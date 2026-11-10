// Homogeneous self-dual interior-point solver for
//   min 1/2 x'Px + q'x  s.t. Ax + s = b, s in K
// K: zero, nonnegative, second-order, exponential, PSD (svec) cones.
#include "io.hpp"
#include "linalg.hpp"
#include <chrono>
#include <cstdarg>

using namespace std;

extern "C" {
void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info, size_t);
void dgesvd_(const char* jobu, const char* jobvt, const int* m, const int* n, double* a, const int* lda,
             double* s, double* u, const int* ldu, double* vt, const int* ldvt, double* work,
             const int* lwork, int* info, size_t, size_t);
void dpotrs_(const char* uplo, const int* n, const int* nrhs, const double* a, const int* lda, double* b, const int* ldb, int* info, size_t);
void dsyev_(const char* jobz, const char* uplo, const int* n, double* a, const int* lda, double* w,
            double* work, const int* lwork, int* info, size_t, size_t);
}

static int g_verbose = 0;
static chrono::steady_clock::time_point g_t0;
static double elapsed() { return chrono::duration<double>(chrono::steady_clock::now() - g_t0).count(); }
static void logf(const char* fmt, ...) {
  if (!g_verbose) return;
  va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
}

enum { CZERO = 0, CNN = 1, CSOC = 2, CEXP = 3, CPSD = 4 };
struct Cone { int type; I dim, d, start; I hoff; };

static const double EPS = 1e-8;
static const double SQRT2 = 1.4142135623730951;

static inline double ninf(const double* v, I n) { double r = 0; for (I i = 0; i < n; i++) r = max(r, fabs(v[i])); return r; }
static inline double dotp(const double* a, const double* b, I n) { double r = 0; for (I i = 0; i < n; i++) r += a[i] * b[i]; return r; }

// ---------------- dense helpers (column-major) ----------------
static double sym_min_eig(vector<double> M, int d) {
  if (d == 1) return M[0];
  vector<double> w(d);
  int lwork = -1, info = 0; double wq;
  dsyev_("N", "U", &d, M.data(), &d, w.data(), &wq, &lwork, &info, 1, 1);
  lwork = (int)wq + 1;
  vector<double> work(lwork);
  dsyev_("N", "U", &d, M.data(), &d, w.data(), work.data(), &lwork, &info, 1, 1);
  if (info != 0) return -1e300;
  return w[0];
}
static inline I svidx(I i, I j) { return j * (j + 1) / 2 + i; }  // i<=j
static void smat(const double* v, I d, double* M) {
  for (I j = 0; j < d; j++)
    for (I i = 0; i <= j; i++) {
      double x = v[svidx(i, j)];
      if (i != j) x /= SQRT2;
      M[i + j * d] = x; M[j + i * d] = x;
    }
}
static void svec(const double* M, I d, double* v) {
  for (I j = 0; j < d; j++)
    for (I i = 0; i <= j; i++) {
      double x = 0.5 * (M[i + j * d] + M[j + i * d]);
      v[svidx(i, j)] = (i == j) ? x : x * SQRT2;
    }
}
// C = A^T B (A: k x m? we use square d)
static void mm(const double* A, bool ta, const double* B, bool tb, double* C, I d) {
  // C = op(A) op(B), all d x d column-major
  vector<double> At(d * d), Bt(d * d);
  // make row access fast: compute C_ij = sum_k opA_ik opB_kj
  // opA_ik: ta ? A[k + i d] : A[i + k d]
  for (I i = 0; i < d; i++) for (I k = 0; k < d; k++) At[k + i * d] = ta ? A[k + i * d] : A[i + k * d];  // At col i = row i of opA
  for (I j = 0; j < d; j++) for (I k = 0; k < d; k++) Bt[k + j * d] = tb ? B[j + k * d] : B[k + j * d];  // Bt col j = col j of opB
  for (I j = 0; j < d; j++)
    for (I i = 0; i < d; i++) {
      const double* a = &At[i * d]; const double* b = &Bt[j * d];
      double s = 0;
      for (I k = 0; k < d; k++) s += a[k] * b[k];
      C[i + j * d] = s;
    }
}

// exponential cone helpers
static inline bool exp_primal_in(const double* s) {
  double u = s[0], v = s[1], w = s[2];
  if (!(v > 0 && w > 0)) return false;
  return v * log(w / v) - u > 0;
}
static inline bool exp_dual_in(const double* z) {
  double a = z[0], b = z[1], c = z[2];
  if (!(a < 0 && c > 0)) return false;
  return b - a - a * log(-c / a) > 0;
}
static void exp_dual_grad_hess(const double* z, double* g, double* H) {
  double a = z[0], b = z[1], c = z[2];
  double l = log(-c / a);
  double psi = b - a - a * l;
  double ga = -l, gb = 1, gc = -a / c;  // grad of psi
  g[0] = -1.0 / a - ga / psi; g[1] = -gb / psi; g[2] = -1.0 / c - gc / psi;
  double ip = 1.0 / psi, ip2 = ip * ip;
  double hpsi[9] = {1.0 / a, 0, -1.0 / c, 0, 0, 0, -1.0 / c, 0, a / (c * c)};
  double gv[3] = {ga, gb, gc};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) H[i * 3 + j] = ip2 * gv[i] * gv[j] - ip * hpsi[i * 3 + j];
  H[0] += 1.0 / (a * a); H[8] += 1.0 / (c * c);
}
static bool chol3_solve(const double* H, const double* r, double* x) {
  double L00 = H[0]; if (L00 <= 0) return false; L00 = sqrt(L00);
  double L10 = H[3] / L00, L20 = H[6] / L00;
  double t = H[4] - L10 * L10; if (t <= 0) return false; double L11 = sqrt(t);
  double L21 = (H[7] - L20 * L10) / L11;
  t = H[8] - L20 * L20 - L21 * L21; if (t <= 0) return false; double L22 = sqrt(t);
  double y0 = r[0] / L00, y1 = (r[1] - L10 * y0) / L11, y2 = (r[2] - L20 * y0 - L21 * y1) / L22;
  x[2] = y2 / L22; x[1] = (y1 - L21 * x[2]) / L11; x[0] = (y0 - L10 * x[1] - L20 * x[2]) / L00;
  return true;
}

struct Prob {
  I n = 0, m = 0;
  CSC P, A;  // P full symmetric
  vector<double> q, b;
  vector<Cone> cones;
  double c0 = 0;
};

// cone defect as in the checker (u normalized per block)
static double exp_E(double a, double b, double c) {
  double B = max(fabs(b), max(max(a, 0.0), max(-c, 0.0)));
  if (b > 0 && c > 0) {
    double t = a - b * (log(c) - log(b));
    return min(B, max(t, 0.0));
  }
  return B;
}
static double cone_defect(const vector<Cone>& cones, const double* v, bool dual) {
  double worst = 0;
  for (auto& K : cones) {
    const double* x = v + K.start;
    I d = K.dim;
    double nrm = ninf(x, d);
    double sc = max(1.0, nrm);
    double def = 0;
    if (K.type == CZERO) { if (!dual) def = nrm / sc; }
    else if (K.type == CNN) { double mn = 0; for (I i = 0; i < d; i++) mn = min(mn, x[i]); def = max(-mn / sc, 0.0); }
    else if (K.type == CSOC) {
      double t = 0; for (I i = 1; i < d; i++) t += x[i] * x[i];
      def = max(sqrt(t) / sc - x[0] / sc, 0.0);
    } else if (K.type == CEXP) {
      double u0 = x[0] / sc, u1 = x[1] / sc, u2 = x[2] / sc;
      def = dual ? exp_E(-u1, -u0, exp(1.0) * u2) : exp_E(u0, u1, u2);
    } else if (K.type == CPSD) {
      I dd = K.d;
      vector<double> M(dd * dd);
      smat(x, dd, M.data());
      double s = dual ? 1.0 : 1.0 / sc;  // dual: "后不再归一化"? use same normalization
      s = 1.0 / sc;
      for (auto& e : M) e *= s;
      def = max(-sym_min_eig(M, (int)dd), 0.0);
    }
    if (!(def == def)) def = 1e300;
    worst = max(worst, def);
  }
  return worst;
}

struct Result { string status; vector<double> x, s, z; double pobj = 0, dobj = 0; bool has_obj = false; };

// =====================================================================
struct Solver {
  const Prob& pr;
  I n, m, N;
  CSC P, A;  // scaled
  vector<double> q, b, D, E; double cs = 1, bs = 1;
  double cmin = 1e-4, cmax = 1e4, bmax = 1e6; bool bscale_on = false;
  vector<Cone> cones;
  double nu = 0;
  bool has_exp = false;
  // KKT
  vector<I> Kp, Ki; vector<double> Kx;
  vector<I> perm, pinv;
  vector<signed char> sign;
  vector<I> posP, posA, posH;  // positions in Kx
  // W-scaled form for small SOC blocks: KKT uses B = T A (T = W^{-1} on those blocks) and -I block
  CSC B; vector<I> Bflag;  // Bflag: -1 fixed value, >=0 soc index (segment start), -2 inside segment
  vector<int> rowsoc;      // soc index of row if row in W-form SOC block else -1
  vector<I> soc_start, soc_dim; vector<char> soc_wf;
  vector<double> tmpm, hbuf, hvbuf;
  // reduced (normal equations) path for nonneg+PSD problems with few variables
  bool reduced = false;
  vector<double> Mden, tr1, tr2;
  vector<vector<double>> psd_Q, psd_R;
  struct PE { I col; I i, j; double v; };
  vector<vector<PE>> psd_ent;
  CSC At;  // transpose of scaled A (rows)
  vector<double> valP;          // P upper values (scaled)
  vector<I> diagX;              // position of diagonal of x-col j
  vector<double> pdiag_s;
  I nH = 0;
  vector<double> Hst;  // dense H storage per cone (nonneg/zero: diag)
  LDL ldl;
  double delta_s = 1e-8;
  // per-cone scaling data
  vector<vector<double>> soc_w; vector<double> soc_eta;
  vector<vector<double>> psd_r, psd_rti, psd_lam;
  vector<double> exp_g;  // 3 per exp cone (in cone order index)
  vector<double> exp_Hd;  // 9 per exp cone
  vector<double> lam;  // NT lambda (for nonneg and soc)
  double mu_scaling = 1;
  // original norms
  double a_scale = 1, h_scale = 1;

  Solver(const Prob& p) : pr(p) {}

  // ---------- equilibration ----------
  void equilibrate() {
    n = pr.n; m = pr.m;
    P = pr.P; A = pr.A; q = pr.q; b = pr.b; cones = pr.cones;
    D.assign(n, 1.0); E.assign(m, 1.0); cs = 1.0;
    vector<double> dn(n), en(m);
    for (int it = 0; it < 15; it++) {
      fill(dn.begin(), dn.end(), 0.0); fill(en.begin(), en.end(), 0.0);
      for (I j = 0; j < n; j++) {
        for (I k = P.p[j]; k < P.p[j + 1]; k++) dn[j] = max(dn[j], fabs(P.x[k]));
        for (I k = A.p[j]; k < A.p[j + 1]; k++) { double v = fabs(A.x[k]); dn[j] = max(dn[j], v); en[A.i[k]] = max(en[A.i[k]], v); }
      }
      double worst = 0;
      for (I j = 0; j < n; j++) { worst = max(worst, fabs(1 - dn[j])); dn[j] = dn[j] < 1e-12 ? 1.0 : 1.0 / sqrt(dn[j]); dn[j] = min(max(dn[j], 1e-4), 1e4); }
      for (auto& K : cones) {
        if (K.type == CSOC || K.type == CEXP || K.type == CPSD) {
          double mx = 0; for (I i = 0; i < K.dim; i++) mx = max(mx, en[K.start + i]);
          for (I i = 0; i < K.dim; i++) en[K.start + i] = mx;
        }
      }
      for (I i = 0; i < m; i++) { worst = max(worst, fabs(1 - en[i])); en[i] = en[i] < 1e-12 ? 1.0 : 1.0 / sqrt(en[i]); en[i] = min(max(en[i], 1e-4), 1e4); }
      for (I j = 0; j < n; j++) {
        for (I k = P.p[j]; k < P.p[j + 1]; k++) P.x[k] *= dn[j] * dn[P.i[k]];
        for (I k = A.p[j]; k < A.p[j + 1]; k++) A.x[k] *= dn[j] * en[A.i[k]];
      }
      for (I j = 0; j < n; j++) D[j] *= dn[j];
      for (I i = 0; i < m; i++) E[i] *= en[i];
      if (worst < 0.1) break;
    }
    for (I j = 0; j < n; j++) q[j] *= D[j];
    for (I i = 0; i < m; i++) b[i] *= E[i];
    // cost scaling
    double pm = 0;
    if (P.nnz() > 0) {
      for (I j = 0; j < n; j++) { double c = 0; for (I k = P.p[j]; k < P.p[j + 1]; k++) c = max(c, fabs(P.x[k])); pm += c; }
      pm /= max<I>(n, 1);
    }
    double ctmp = max(pm, ninf(q.data(), n));
    if (ctmp < 1e-6) ctmp = 1.0;
    ctmp = min(max(ctmp, cmin), cmax);
    cs = 1.0 / ctmp;
    for (auto& v : P.x) v *= cs;
    for (auto& v : q) v *= cs;
    // rhs scaling: b <- b/beta, P <- beta P (x scales by beta, z unchanged)
    bs = 1.0;
    if (bscale_on) {
      double bn = ninf(b.data(), m);
      if (bn > 1.0) bs = min(bn, bmax);
      for (auto& v : b) v /= bs;
      for (auto& v : P.x) v *= bs;
    }
  }

  // ---------- KKT ----------
  void build_kkt() {
    N = n + m;
    // count entries per original column (upper triangle)
    vector<I> rows, cols;
    I nP = 0;
    for (I j = 0; j < n; j++) for (I k = P.p[j]; k < P.p[j + 1]; k++) if (P.i[k] < j) nP++;
    I nA = A.nnz();
    nH = 0;
    I hst = 0;
    for (auto& K : cones) {
      if (K.type == CZERO || K.type == CNN) { nH += K.dim; K.hoff = hst; hst += K.dim; }
      else if (K.type == CSOC && K.dim <= wf_max) { nH += K.dim; K.hoff = hst; hst += K.dim * K.dim; }
      else { nH += K.dim * (K.dim + 1) / 2; K.hoff = hst; hst += K.dim * K.dim; }
    }
    Hst.assign(hst, 0.0);
    I tot = n + nP + nA + nH;
    rows.reserve(tot); cols.reserve(tot);
    valP.clear();
    // P diag entries first
    for (I j = 0; j < n; j++) { rows.push_back(j); cols.push_back(j); }
    vector<double> pdiag(n, 0.0);
    for (I j = 0; j < n; j++)
      for (I k = P.p[j]; k < P.p[j + 1]; k++) {
        if (P.i[k] < j) { rows.push_back(P.i[k]); cols.push_back(j); valP.push_back(P.x[k]); }
        else if (P.i[k] == j) pdiag[j] += P.x[k];
      }
    // A entries: (j, n+i)
    build_B();
    nA = B.nnz();
    for (I j = 0; j < n; j++) for (I k = B.p[j]; k < B.p[j + 1]; k++) { rows.push_back(j); cols.push_back(n + B.i[k]); }
    for (auto& K : cones) {
      if (K.type == CZERO || K.type == CNN || (K.type == CSOC && K.dim <= wf_max)) { for (I r = 0; r < K.dim; r++) { rows.push_back(n + K.start + r); cols.push_back(n + K.start + r); } }
      else {
        for (I c = 0; c < K.dim; c++) for (I r = 0; r <= c; r++) { rows.push_back(n + K.start + r); cols.push_back(n + K.start + c); }
      }
    }
    // symmetric pattern for AMD
    {
      vector<I> cnt(N + 1, 0);
      for (size_t e = 0; e < rows.size(); e++) if (rows[e] != cols[e]) { cnt[rows[e] + 1]++; cnt[cols[e] + 1]++; }
      for (I i = 0; i < N; i++) cnt[i + 1] += cnt[i];
      vector<I> Ci(cnt[N]); vector<I> w(cnt.begin(), cnt.end() - 1);
      for (size_t e = 0; e < rows.size(); e++) if (rows[e] != cols[e]) { Ci[w[rows[e]]++] = cols[e]; Ci[w[cols[e]]++] = rows[e]; }
      double t = elapsed();
      perm = amd_order(N, cnt, Ci);
      logf("amd %.3fs N=%lld nnzC=%lld\n", elapsed() - t, (long long)N, (long long)cnt[N]);
    }
    pinv.assign(N, 0);
    for (I k = 0; k < N; k++) pinv[perm[k]] = k;
    // permuted upper triangle
    Kp.assign(N + 1, 0);
    for (size_t e = 0; e < rows.size(); e++) { I a = pinv[rows[e]], c = pinv[cols[e]]; Kp[max(a, c) + 1]++; }
    for (I i = 0; i < N; i++) Kp[i + 1] += Kp[i];
    Ki.resize(rows.size()); Kx.assign(rows.size(), 0.0);
    vector<I> w(Kp.begin(), Kp.end() - 1);
    vector<I> pos(rows.size());
    for (size_t e = 0; e < rows.size(); e++) {
      I a = pinv[rows[e]], c = pinv[cols[e]];
      I col = max(a, c), row = min(a, c);
      I q2 = w[col]++;
      Ki[q2] = row; pos[e] = q2;
    }
    // fixed values
    I e = 0;
    diagX.resize(n);
    pdiag_s = pdiag;
    for (I j = 0; j < n; j++, e++) { diagX[j] = pos[e]; Kx[pos[e]] = pdiag[j] + delta_s; }
    posP.assign(pos.begin() + e, pos.begin() + e + nP);
    for (I t = 0; t < nP; t++) Kx[posP[t]] = valP[t];
    e += nP;
    posA.assign(pos.begin() + e, pos.begin() + e + nA);
    {
      I t = 0;
      for (I j = 0; j < n; j++) for (I k = B.p[j]; k < B.p[j + 1]; k++, t++) Kx[posA[t]] = B.x[k];
    }
    e += nA;
    posH.assign(pos.begin() + e, pos.begin() + e + nH);
    sign.assign(N, 1);
    for (I i = 0; i < m; i++) sign[pinv[n + i]] = -1;
    double t = elapsed();
    ldl.symbolic(N, Kp, Ki);
    logf("symbolic %.3fs nnzL=%lld\n", elapsed() - t, (long long)ldl.Lp[N]);
  }

  void build_B() {
    rowsoc.assign(m, -1);
    soc_start.clear(); soc_dim.clear(); soc_wf.clear();
    for (auto& K : cones) if (K.type == CSOC) {
      int is = (int)soc_start.size();
      soc_start.push_back(K.start); soc_dim.push_back(K.dim); soc_wf.push_back(K.dim <= wf_max);
      if (K.dim <= wf_max) for (I r = 0; r < K.dim; r++) rowsoc[K.start + r] = is;
    }
    B.m = m; B.n = n; B.p.assign(n + 1, 0); B.i.clear(); B.x.clear(); Bflag.clear();
    vector<I> seen(soc_start.size(), -1);
    for (I j = 0; j < n; j++) {
      for (I k = A.p[j]; k < A.p[j + 1]; k++) {
        I i = A.i[k]; int is = rowsoc[i];
        if (is < 0) { B.i.push_back(i); B.x.push_back(A.x[k]); Bflag.push_back(-1); }
        else if (seen[is] != j) {
          seen[is] = j;
          for (I r = 0; r < soc_dim[is]; r++) { B.i.push_back(soc_start[is] + r); B.x.push_back(0); Bflag.push_back(r == 0 ? is : -2); }
        }
      }
      B.p[j + 1] = (I)B.i.size();
    }
    tmpm.assign(m, 0.0);
  }
  void update_B() {
    for (I j = 0; j < n; j++) {
      bool any = false;
      for (I k = A.p[j]; k < A.p[j + 1]; k++) if (rowsoc[A.i[k]] >= 0) { tmpm[A.i[k]] = A.x[k]; any = true; }
      if (!any) continue;
      for (I k = B.p[j]; k < B.p[j + 1]; k++) {
        if (Bflag[k] >= 0) {
          int is = (int)Bflag[k]; I st = soc_start[is], d = soc_dim[is];
          soc_Wmul(is, d, &tmpm[st], &B.x[k], true);
          for (I r = 0; r < d; r++) tmpm[st + r] = 0;
        }
      }
    }
    I t = 0;
    for (I j = 0; j < n; j++) for (I k = B.p[j]; k < B.p[j + 1]; k++, t++) Kx[posA[t]] = B.x[k];
  }
  // apply T = W^{-1} on W-form SOC blocks (in place on z-vector)
  void apply_T(double* v) {
    double buf[16];
    for (size_t t = 0; t < soc_start.size(); t++) if (soc_wf[t]) {
      I st = soc_start[t], d = soc_dim[t];
      soc_Wmul((int)t, d, v + st, buf, true);
      for (I r = 0; r < d; r++) v[st + r] = buf[r];
    }
  }

  void write_H_to_kkt() {
    for (I j = 0; j < n; j++) Kx[diagX[j]] = pdiag_s[j] + delta_s;
    I t = 0;
    for (auto& K : cones) {
      const double* H = &Hst[K.hoff];
      if (K.type == CZERO || K.type == CNN) {
        for (I r = 0; r < K.dim; r++) Kx[posH[t++]] = -H[r] - delta_s;
      } else if (K.type == CSOC && K.dim <= wf_max) {
        for (I r = 0; r < K.dim; r++) Kx[posH[t++]] = -1.0 - delta_s;
      } else {
        I d = K.dim;
        for (I c = 0; c < d; c++) for (I r = 0; r <= c; r++) {
          double v = -H[r + c * d];
          if (r == c) v -= delta_s;
          Kx[posH[t++]] = v;
        }
      }
    }
  }

  // y = H v (z-space)
  // y = svec(R smat(v) R)
  void psd_mul(const vector<double>& R, I d, const double* v, double* y) {
    vector<double> X(d * d), T(d * d);
    smat(v, d, X.data());
    mm(R.data(), false, X.data(), false, T.data(), d);
    mm(T.data(), false, R.data(), false, X.data(), d);
    svec(X.data(), d, y);
  }
  void Hinv_mul(const double* v, double* y) {
    int ip = 0;
    for (auto& K : cones) {
      const double* vv = v + K.start; double* yy = y + K.start;
      if (K.type == CNN) { const double* H = &Hst[K.hoff]; for (I r = 0; r < K.dim; r++) yy[r] = vv[r] / H[r]; }
      else if (K.type == CPSD) { psd_mul(psd_Q[ip], K.d, vv, yy); ip++; }
      else { for (I r = 0; r < K.dim; r++) yy[r] = 0; }
    }
  }
  void setup_reduced() {
    N = n + m;
    I hst = 0;
    for (auto& K : cones) { if (K.type == CNN) { K.hoff = hst; hst += K.dim; } else K.hoff = 0; }
    Hst.assign(max<I>(hst, 1), 0.0);
    At = csc_transpose(A);
    vector<int> rc(m, -1);
    int ip = 0;
    for (auto& K : cones) { if (K.type == CPSD) { for (I r = 0; r < K.dim; r++) rc[K.start + r] = ip; ip++; } }
    psd_ent.assign(ip, {}); psd_Q.assign(ip, {}); psd_R.assign(ip, {});
    vector<I> pstart(ip);
    ip = 0;
    for (auto& K : cones) if (K.type == CPSD) {
      pstart[ip] = K.start;
      psd_Q[ip].assign(K.d * K.d, 0.0); psd_R[ip].assign(K.d * K.d, 0.0);
      for (I i = 0; i < K.d; i++) { psd_Q[ip][i + i * K.d] = 1; psd_R[ip][i + i * K.d] = 1; }
      ip++;
    }
    for (I j = 0; j < n; j++) for (I k = A.p[j]; k < A.p[j + 1]; k++) {
      I r = A.i[k]; int c = rc[r];
      if (c < 0) continue;
      I idx = r - pstart[c];
      I jj = (I)((sqrt(8.0 * idx + 1) - 1) / 2);
      while (jj * (jj + 1) / 2 > idx) jj--;
      while ((jj + 1) * (jj + 2) / 2 <= idx) jj++;
      I ii = idx - jj * (jj + 1) / 2;
      psd_ent[c].push_back({j, ii, jj, A.x[k]});
    }
    tr1.resize(m); tr2.resize(m);
  }
  bool factor_reduced() {
    {
      Mden.assign(n * n, 0.0);
      for (I j = 0; j < n; j++) for (I k = P.p[j]; k < P.p[j + 1]; k++) Mden[P.i[k] + j * n] += P.x[k];
      for (I j = 0; j < n; j++) Mden[j + j * n] += delta_s;
      for (auto& K : cones) if (K.type == CNN) {
        for (I r = 0; r < K.dim; r++) {
          I row = K.start + r; double h = 1.0 / Hst[K.hoff + r];
          for (I a = At.p[row]; a < At.p[row + 1]; a++) {
            double va = h * At.x[a]; I ca = At.i[a];
            for (I b2 = At.p[row]; b2 < At.p[row + 1]; b2++) Mden[ca + At.i[b2] * n] += va * At.x[b2];
          }
        }
      }
      int ip = 0;
      for (auto& K : cones) if (K.type == CPSD) {
        const vector<double>& Q = psd_Q[ip]; I d = K.d;
        for (auto& e1 : psd_ent[ip]) {
          double s1 = (e1.i == e1.j) ? 1.0 : SQRT2;
          for (auto& e2 : psd_ent[ip]) {
            double s2 = (e2.i == e2.j) ? 1.0 : SQRT2;
            double hv = s1 * s2 * 0.5 * (Q[e1.i + e2.i * d] * Q[e1.j + e2.j * d] + Q[e1.i + e2.j * d] * Q[e1.j + e2.i * d]);
            Mden[e1.col + e2.col * n] += hv * e1.v * e2.v;
          }
        }
        ip++;
      }
    }
    vector<double> Mc(Mden);
    double rel = 0;
    for (int attempt = 0; attempt < 8; attempt++) {
      if (attempt > 0) {
        Mden = Mc;
        rel = (attempt == 1) ? 1e-14 : rel * 30;
        for (I j = 0; j < n; j++) Mden[j + j * n] += rel * fabs(Mc[j + j * n]) + rel;
        logf("reduced chol retry rel %.1e\n", rel);
      }
      int nn = (int)n, info = 0;
      dpotrf_("L", &nn, Mden.data(), &nn, &info, 1);
      if (info == 0) return true;
    }
    return false;
  }
  void precond_reduced(const double* r, double* out) {
    Hinv_mul(r + n, tr1.data());
    for (I j = 0; j < n; j++) out[j] = r[j];
    spmtv_add(A, tr1.data(), out);
    int nn = (int)n, one = 1, info = 0;
    if (n > 0) dpotrs_("L", &nn, &one, Mden.data(), &nn, out, &nn, &info, 1);
    for (I i = 0; i < m; i++) tr2[i] = 0;
    spmv_add(A, out, tr2.data());
    Hinv_mul(tr2.data(), out + n);
    for (I i = 0; i < m; i++) out[n + i] -= tr1[i];
  }

  void H_mul(const double* v, double* y) {
    int is = 0, ipsd = 0;
    for (auto& K : cones) {
      const double* H = &Hst[K.hoff];
      const double* vv = v + K.start; double* yy = y + K.start;
      if (K.type == CZERO || K.type == CNN) { for (I r = 0; r < K.dim; r++) yy[r] = H[r] * vv[r]; }
      else if (K.type == CPSD && reduced) {
        psd_mul(psd_R[ipsd], K.d, vv, yy);
        ipsd++;
      }
      else if (K.type == CSOC) {
        if ((I)hbuf.size() < K.dim) hbuf.resize(K.dim);
        soc_Wmul(is, K.dim, vv, hbuf.data(), false);
        soc_Wmul(is, K.dim, hbuf.data(), yy, false);
        is++;
      }
      else {
        I d = K.dim;
        for (I r = 0; r < d; r++) yy[r] = 0;
        for (I c = 0; c < d; c++) { double vc = vv[c]; const double* col = H + c * d; for (I r = 0; r < d; r++) yy[r] += col[r] * vc; }
      }
    }
  }

  // K_true * v (original ordering)
  void kkt_mul(const double* v, double* y) {
    for (I i = 0; i < N; i++) y[i] = 0;
    spmv_add(P, v, y);
    spmtv_add(A, v + n, y);
    spmv_add(A, v, y + n);
    hvbuf.resize(m);
    double* hv = hvbuf.data();
    H_mul(v + n, hv);
    for (I i = 0; i < m; i++) y[n + i] -= hv[i];
  }

  vector<double> wk1, wk2, wk3, wk4, wk5, sol_tmp;
  double last_ir = 0;
  double exp_beta = 2.0, exp_sigmin = 0.0, exp_fallback = 0.9;
  int nsmall = 0;
  int ir_max = 20;
  double ir_abs = 1e-12, ir_rel = 1e-12;
  bool tight_ir = false;
  I wf_max = 16;
  void precond(const double* r, double* out) {
    if (reduced) { precond_reduced(r, out); return; }
    wk3.resize(N);
    for (I i = 0; i < N; i++) wk3[i] = r[i];
    apply_T(wk3.data() + n);
    for (I k = 0; k < N; k++) wk1[k] = wk3[perm[k]];
    ldl.solve(wk1.data());
    for (I k = 0; k < N; k++) out[perm[k]] = wk1[k];
    apply_T(out + n);
  }
  // solve K sol = rhs (original space) with iterative refinement
  void kkt_solve(const double* rhs, double* sol) {
    wk1.resize(N); wk2.resize(N); wk4.resize(N); wk5.resize(N);
    precond(rhs, sol);
    double bn = ninf(rhs, N);
    kkt_mul(sol, wk2.data());
    for (I i = 0; i < N; i++) wk2[i] = rhs[i] - wk2[i];
    double rn = ninf(wk2.data(), N);
    for (int it = 0; it < ir_max; it++) {
      if (rn <= ir_abs + ir_rel * bn) break;
      precond(wk2.data(), wk4.data());
      for (I i = 0; i < N; i++) wk4[i] += sol[i];
      kkt_mul(wk4.data(), wk5.data());
      for (I i = 0; i < N; i++) wk5[i] = rhs[i] - wk5[i];
      double rn2 = ninf(wk5.data(), N);
      if (!(rn2 < rn)) break;
      bool good = rn2 < 0.5 * rn;
      rn = rn2;
      std::swap(wk2, wk5);
      for (I i = 0; i < N; i++) sol[i] = wk4[i];
      if (!good) break;
    }
    last_ir = rn / (1 + bn);
  }

  // ---------- cone operations ----------
  void unit_init(vector<double>& s, vector<double>& z) {
    for (auto& K : cones) {
      double* ss = &s[K.start]; double* zz = &z[K.start];
      if (K.type == CZERO) { for (I i = 0; i < K.dim; i++) { ss[i] = 0; zz[i] = 0; } }
      else if (K.type == CNN) { for (I i = 0; i < K.dim; i++) { ss[i] = 1; zz[i] = 1; } }
      else if (K.type == CSOC) { for (I i = 0; i < K.dim; i++) { ss[i] = zz[i] = (i == 0); } }
      else if (K.type == CEXP) {
        ss[0] = zz[0] = -1.051383945322714; ss[1] = zz[1] = 0.556409619469370; ss[2] = zz[2] = 1.258967884768947;
      } else {
        for (I j = 0; j < K.d; j++) for (I i = 0; i <= j; i++) ss[svidx(i, j)] = zz[svidx(i, j)] = (i == j);
      }
    }
  }
  // min "eigenvalue" for symmetric cones
  void shift_to_cone(vector<double>& v) {
    for (auto& K : cones) {
      double* x = &v[K.start];
      if (K.type == CZERO) { continue; }
      double mn;
      if (K.type == CNN) { mn = 1e300; for (I i = 0; i < K.dim; i++) mn = min(mn, x[i]); }
      else if (K.type == CSOC) { double t = 0; for (I i = 1; i < K.dim; i++) t += x[i] * x[i]; mn = x[0] - sqrt(t); }
      else if (K.type == CPSD) { vector<double> M(K.d * K.d); smat(x, K.d, M.data()); mn = sym_min_eig(M, (int)K.d); }
      else continue;
      if (mn < 1e-8) {
        double a = 1.0 - mn;
        if (K.type == CNN) for (I i = 0; i < K.dim; i++) x[i] += a;
        else if (K.type == CSOC) x[0] += a;
        else for (I j = 0; j < K.d; j++) x[svidx(j, j)] += a;
      }
    }
  }

  bool update_scaling(const vector<double>& s, const vector<double>& z, double mu) {
    lam.assign(m, 0.0);
    int ie = 0, ip = 0, is = 0;
    for (auto& K : cones) {
      const double* ss = &s[K.start]; const double* zz = &z[K.start];
      double* H = &Hst[K.hoff];
      if (K.type == CZERO) { for (I i = 0; i < K.dim; i++) H[i] = 0; }
      else if (K.type == CNN) {
        for (I i = 0; i < K.dim; i++) { H[i] = ss[i] / zz[i]; lam[K.start + i] = sqrt(ss[i] * zz[i]); }
      } else if (K.type == CSOC) {
        I d = K.dim;
        double s1 = 0, z1 = 0;
        for (I i = 1; i < d; i++) { s1 += ss[i] * ss[i]; z1 += zz[i] * zz[i]; }
        s1 = sqrt(s1); z1 = sqrt(z1);
        double sres = (ss[0] - s1) * (ss[0] + s1), zres = (zz[0] - z1) * (zz[0] + z1);
        if (!(sres > 0 && zres > 0)) return false;
        double sscale = sqrt(sres), zscale = sqrt(zres);
        double eta = sqrt(sscale / zscale);
        vector<double>& w = soc_w[is];
        w.resize(d);
        double dt = 0;
        for (I i = 0; i < d; i++) dt += ss[i] * zz[i];
        dt /= (sscale * zscale);
        double gamma = sqrt((1 + dt) / 2);
        w[0] = (ss[0] / sscale + zz[0] / zscale) / (2 * gamma);
        for (I i = 1; i < d; i++) w[i] = (ss[i] / sscale - zz[i] / zscale) / (2 * gamma);
        // renormalize w so that w0^2 - |w1|^2 = 1
        double w1 = 0; for (I i = 1; i < d; i++) w1 += w[i] * w[i];
        w[0] = sqrt(1 + w1);
        soc_eta[is] = eta;
        double e2 = eta * eta;
        for (I c = 0; c < d; c++) for (I r = 0; r < d; r++) H[r + c * d] = e2 * 2 * w[r] * w[c];
        H[0] -= e2;
        for (I i = 1; i < d; i++) H[i + i * d] += e2;
        // lambda = W z
        soc_Wmul(is, d, zz, &lam[K.start], false);
        is++;
      } else if (K.type == CEXP) {
        double g[3], Hd[9];
        exp_dual_grad_hess(zz, g, Hd);
        for (int i = 0; i < 3; i++) exp_g[3 * ie + i] = g[i];
        for (int i = 0; i < 9; i++) { exp_Hd[9 * ie + i] = Hd[i]; H[i] = mu * Hd[i]; }
        ie++;
      } else {
        I d = K.d; int di = (int)d;
        vector<double> S(d * d), Z(d * d);
        smat(ss, d, S.data()); smat(zz, d, Z.data());
        int info = 0;
        dpotrf_("L", &di, S.data(), &di, &info, 1); if (info) return false;
        dpotrf_("L", &di, Z.data(), &di, &info, 1); if (info) return false;
        for (I j = 0; j < d; j++) for (I i = 0; i < j; i++) { S[i + j * d] = 0; Z[i + j * d] = 0; }
        vector<double> M(d * d);
        mm(Z.data(), true, S.data(), false, M.data(), d);  // Lz' Ls
        vector<double> U(d * d), VT(d * d), sv(d);
        int lwork = -1; double wq;
        dgesvd_("A", "A", &di, &di, M.data(), &di, sv.data(), U.data(), &di, VT.data(), &di, &wq, &lwork, &info, 1, 1);
        lwork = (int)wq + 1; vector<double> work(lwork);
        dgesvd_("A", "A", &di, &di, M.data(), &di, sv.data(), U.data(), &di, VT.data(), &di, work.data(), &lwork, &info, 1, 1);
        if (info) return false;
        vector<double>& r = psd_r[ip]; vector<double>& rti = psd_rti[ip]; vector<double>& lm = psd_lam[ip];
        r.assign(d * d, 0); rti.assign(d * d, 0); lm.assign(sv.begin(), sv.end());
        // r = Ls V diag(sv^-1/2), V = VT'
        mm(S.data(), false, VT.data(), true, r.data(), d);
        mm(Z.data(), false, U.data(), false, rti.data(), d);
        for (I k = 0; k < d; k++) {
          double f = 1.0 / sqrt(sv[k]);
          for (I i = 0; i < d; i++) { r[i + k * d] *= f; rti[i + k * d] *= f; }
        }
        // R = r r'
        vector<double> R(d * d);
        mm(r.data(), false, r.data(), true, R.data(), d);
        I dim = K.dim;
        if (reduced) {
          psd_R[ip] = R;
          psd_Q[ip].resize(d * d);
          mm(rti.data(), false, rti.data(), true, psd_Q[ip].data(), d);
          ip++;
          continue;
        }
        // H[(ij),(kl)] = s_ij s_kl (R_ik R_jl + R_il R_jk)/2
        for (I l = 0; l < d; l++) for (I k = 0; k <= l; k++) {
          I col = svidx(k, l);
          double skl = (k == l) ? 1.0 : SQRT2;
          for (I j = 0; j < d; j++) for (I i = 0; i <= j; i++) {
            I row = svidx(i, j);
            double sij = (i == j) ? 1.0 : SQRT2;
            H[row + col * dim] = sij * skl * 0.5 * (R[i + k * d] * R[j + l * d] + R[i + l * d] * R[j + k * d]);
          }
        }
        ip++;
      }
    }
    return true;
  }
  // SOC: y = W x (or W^{-1} x if inv)
  void soc_Wmul(int is, I d, const double* x, double* y, bool inv) {
    const vector<double>& w = soc_w[is];
    double eta = soc_eta[is];
    double w0 = w[0];
    double t = 0; for (I i = 1; i < d; i++) t += w[i] * x[i];
    double sg = inv ? -1.0 : 1.0;
    double y0 = w0 * x[0] + sg * t;
    double c = sg * x[0] + t / (1 + w0);
    double f = inv ? 1.0 / eta : eta;
    y[0] = f * y0;
    for (I i = 1; i < d; i++) y[i] = f * (x[i] + c * w[i]);
  }

  // d_s for the combined step (Mehrotra correction), symmetric: W'(lam \ (lam o lam + (W^-T ds) o (W dz) - sigma mu e))
  void combined_ds(const vector<double>& s, const vector<double>& z, const vector<double>& dsa, const vector<double>& dza,
                   double sigmu, vector<double>& out) {
    int ie = 0, ip = 0, is = 0;
    for (auto& K : cones) {
      I st = K.start, d = K.dim;
      double* o = &out[st];
      if (K.type == CZERO) { for (I i = 0; i < d; i++) o[i] = 0; }
      else if (K.type == CNN) {
        for (I i = 0; i < d; i++) o[i] = s[st + i] + (dsa[st + i] * dza[st + i] - sigmu) / z[st + i];
      } else if (K.type == CSOC) {
        double a[64], bb[64], v[64], u[64];
        if (d > 64) { /* large cones: fall back to heap */ }
        vector<double> ha, hb, hv2, hu;
        double *pa = a, *pb = bb, *pv = v, *pu = u;
        if (d > 64) { ha.resize(d); hb.resize(d); hv2.resize(d); hu.resize(d); pa = ha.data(); pb = hb.data(); pv = hv2.data(); pu = hu.data(); }
        combined_soc(is, d, &dsa[st], &dza[st], &lam[st], sigmu, pa, pb, pv, pu, o);
        is++;
        continue;
      } else if (K.type == CEXP) {
        for (int i = 0; i < 3; i++) o[i] = s[st + i] + sigmu * exp_g[3 * ie + i];
        ie++;
      } else {
        I dd = K.d;
        vector<double>& r = psd_r[ip]; vector<double>& rti = psd_rti[ip]; vector<double>& lm = psd_lam[ip];
        vector<double> Sm(dd * dd), Zm(dd * dd), T(dd * dd), A1(dd * dd), B1(dd * dd), V(dd * dd);
        smat(&dsa[st], dd, Sm.data()); smat(&dza[st], dd, Zm.data());
        mm(rti.data(), true, Sm.data(), false, T.data(), dd); mm(T.data(), false, rti.data(), false, A1.data(), dd);
        mm(r.data(), true, Zm.data(), false, T.data(), dd); mm(T.data(), false, r.data(), false, B1.data(), dd);
        mm(A1.data(), false, B1.data(), false, V.data(), dd);
        for (I j = 0; j < dd; j++) for (I i = 0; i <= j; i++) {
          double x = 0.5 * (V[i + j * dd] + V[j + i * dd]);
          V[i + j * dd] = x; V[j + i * dd] = x;
        }
        for (I i = 0; i < dd; i++) V[i + i * dd] += lm[i] * lm[i] - sigmu;
        for (I j = 0; j < dd; j++) for (I i = 0; i < dd; i++) V[i + j * dd] *= 2.0 / (lm[i] + lm[j]);
        // out = svec(r V r')
        mm(r.data(), false, V.data(), false, T.data(), dd); mm(T.data(), false, r.data(), true, V.data(), dd);
        svec(V.data(), dd, o);
        ip++;
      }
    }
  }

  void combined_soc(int is, I d, const double* dsa_, const double* dza_, const double* l, double sigmu,
                    double* a, double* bb, double* v, double* u, double* o) {
    soc_Wmul(is, d, dsa_, a, true);
    soc_Wmul(is, d, dza_, bb, false);
    v[0] = dotp(l, l, d) + dotp(a, bb, d) - sigmu;
    for (I i = 1; i < d; i++) v[i] = 2 * l[0] * l[i] + a[0] * bb[i] + bb[0] * a[i];
    double l1 = 0; for (I i = 1; i < d; i++) l1 += l[i] * l[i];
    double den = l[0] * l[0] - l1;
    double lv = 0; for (I i = 1; i < d; i++) lv += l[i] * v[i];
    u[0] = (l[0] * v[0] - lv) / den;
    for (I i = 1; i < d; i++) u[i] = (v[i] - u[0] * l[i]) / l[0];
    soc_Wmul(is, d, u, o, false);
  }
  // maximum step in cones (symmetric exact, exp by backtracking)
  double step_cones(const vector<double>& s, const vector<double>& z, const vector<double>& ds, const vector<double>& dz, double amax) {
    double a = amax;
    int ip = 0;
    for (auto& K : cones) {
      I st = K.start, d = K.dim;
      if (K.type == CZERO) continue;
      if (K.type == CNN) {
        for (I i = 0; i < d; i++) {
          if (ds[st + i] < 0) a = min(a, -s[st + i] / ds[st + i]);
          if (dz[st + i] < 0) a = min(a, -z[st + i] / dz[st + i]);
        }
      } else if (K.type == CSOC) {
        a = min(a, soc_step(&s[st], &ds[st], d));
        a = min(a, soc_step(&z[st], &dz[st], d));
      } else if (K.type == CPSD) {
        a = min(a, psd_step(&s[st], &ds[st], K.d));
        a = min(a, psd_step(&z[st], &dz[st], K.d));
        ip++;
      }
    }
    if (has_exp) {
      for (auto& K : cones) {
        if (K.type != CEXP) continue;
        I st = K.start;
        for (int t = 0; t < 200; t++) {
          double sp[3], zp[3];
          for (int i = 0; i < 3; i++) { sp[i] = s[st + i] + a * ds[st + i]; zp[i] = z[st + i] + a * dz[st + i]; }
          if (exp_primal_in(sp) && exp_dual_in(zp)) break;
          a *= 0.8;
        }
      }
    }
    return a;
  }
  static double soc_step(const double* x, const double* dx, I d) {
    double x1 = 0, dx1 = 0, xd = 0;
    for (I i = 1; i < d; i++) { x1 += x[i] * x[i]; dx1 += dx[i] * dx[i]; xd += x[i] * dx[i]; }
    double A2 = dx[0] * dx[0] - dx1;
    double B = 2 * (x[0] * dx[0] - xd);
    double C = (x[0] - sqrt(x1)) * (x[0] + sqrt(x1));
    if (C < 0) C = 0;
    double best = 1e300;
    // also x0 + a dx0 >= 0
    if (dx[0] < 0) best = min(best, -x[0] / dx[0]);
    double disc = B * B - 4 * A2 * C;
    if (A2 == 0) {
      if (B < 0) best = min(best, -C / B);
    } else if (disc >= 0) {
      double sq = sqrt(disc);
      double qq = -0.5 * (B + (B >= 0 ? sq : -sq));
      double r1 = qq / A2, r2 = (qq != 0) ? C / qq : 1e300;
      if (r1 > 0) best = min(best, r1);
      if (r2 > 0) best = min(best, r2);
    }
    return best;
  }
  static double psd_step(const double* x, const double* dx, I d) {
    vector<double> X(d * d), DX(d * d);
    smat(x, d, X.data()); smat(dx, d, DX.data());
    int di = (int)d, info = 0;
    dpotrf_("L", &di, X.data(), &di, &info, 1);
    if (info) return 0.0;
    // M = L^{-1} DX L^{-T}: solve column-wise via forward substitution
    for (I j = 0; j < d; j++) for (I i = 0; i < j; i++) X[i + j * d] = 0;
    // T = L^{-1} DX
    for (I c = 0; c < d; c++) {
      double* col = &DX[c * d];
      for (I i = 0; i < d; i++) { double sacc = col[i]; for (I k = 0; k < i; k++) sacc -= X[i + k * d] * col[k]; col[i] = sacc / X[i + i * d]; }
    }
    // M = T L^{-T} = (L^{-1} T')'
    vector<double> Tt(d * d);
    for (I i = 0; i < d; i++) for (I j = 0; j < d; j++) Tt[j + i * d] = DX[i + j * d];
    for (I c = 0; c < d; c++) {
      double* col = &Tt[c * d];
      for (I i = 0; i < d; i++) { double sacc = col[i]; for (I k = 0; k < i; k++) sacc -= X[i + k * d] * col[k]; col[i] = sacc / X[i + i * d]; }
    }
    double mn = sym_min_eig(Tt, di);
    if (mn >= 0) return 1e300;
    return -1.0 / mn;
  }

  // exp-cone centrality check at a trial point
  bool exp_central(const vector<double>& s, const vector<double>& z, const vector<double>& ds, const vector<double>& dz,
                   double a, double mu_new, double beta) {
    for (auto& K : cones) {
      if (K.type != CEXP) continue;
      I st = K.start;
      double sp[3], zp[3], g[3], H[9], psi[3], y[3];
      for (int i = 0; i < 3; i++) { sp[i] = s[st + i] + a * ds[st + i]; zp[i] = z[st + i] + a * dz[st + i]; }
      if (!exp_primal_in(sp) || !exp_dual_in(zp)) return false;
      exp_dual_grad_hess(zp, g, H);
      for (int i = 0; i < 3; i++) psi[i] = sp[i] + mu_new * g[i];
      if (!chol3_solve(H, psi, y)) return false;
      double nrm = psi[0] * y[0] + psi[1] * y[1] + psi[2] * y[2];
      if (nrm > beta * beta * mu_new * mu_new) return false;
    }
    return true;
  }

  // ---------- checks in original space ----------
  bool check_solved(const vector<double>& x, const vector<double>& s, const vector<double>& z, double tau,
                    vector<double>& xo, vector<double>& so, vector<double>& zo, double* info) {
    I n0 = n, m0 = m;
    xo.resize(n0); so.resize(m0); zo.resize(m0);
    for (I j = 0; j < n0; j++) xo[j] = bs * D[j] * x[j] / tau;
    for (I i = 0; i < m0; i++) { so[i] = bs * s[i] / (E[i] * tau); zo[i] = E[i] * z[i] / (cs * tau); }
    for (auto& K : cones) if (K.type == CZERO) for (I i = 0; i < K.dim; i++) so[K.start + i] = 0;
    vector<double> Ax(m0, 0.0), Px(n0, 0.0), Atz(n0, 0.0);
    spmv_add(pr.A, xo.data(), Ax.data());
    spmv_add(pr.P, xo.data(), Px.data());
    spmtv_add(pr.A, zo.data(), Atz.data());
    double rp = 0;
    for (I i = 0; i < m0; i++) rp = max(rp, fabs(Ax[i] + so[i] - pr.b[i]));
    rp /= 1 + max(ninf(Ax.data(), m0), max(ninf(so.data(), m0), ninf(pr.b.data(), m0)));
    double rd = 0;
    for (I j = 0; j < n0; j++) rd = max(rd, fabs(Px[j] + pr.q[j] + Atz[j]));
    rd /= 1 + max(ninf(Px.data(), n0), max(ninf(pr.q.data(), n0), ninf(Atz.data(), n0)));
    double xPx = dotp(xo.data(), Px.data(), n0);
    double p0 = 0.5 * xPx + dotp(pr.q.data(), xo.data(), n0);
    double d0 = -0.5 * xPx - dotp(pr.b.data(), zo.data(), m0);
    double gap = fabs(p0 - d0);
    double mx = max(fabs(p0), fabs(d0));
    double rgap = mx > 0 ? gap / mx : 0;
    info[0] = rp; info[1] = rd; info[2] = gap; info[3] = rgap; info[4] = p0; info[5] = d0;
    double tol = 0.9 * EPS;
    if (!(rp <= tol && rd <= tol && (gap <= tol || rgap <= tol))) return false;
    return true;
  }
  bool check_pinf(const vector<double>& z, vector<double>& zo, double* info) {
    zo.resize(m);
    for (I i = 0; i < m; i++) zo[i] = E[i] * z[i];
    double zn = ninf(zo.data(), m), bn = ninf(pr.b.data(), m);
    if (!(zn > 0 && bn > 0) || !std::isfinite(zn)) return false;
    for (auto& v : zo) v /= zn;
    double beta = -dotp(pr.b.data(), zo.data(), m) / bn;
    info[0] = beta;
    if (!(beta > 1e-6)) return false;
    vector<double> Atz(n, 0.0);
    spmtv_add(pr.A, zo.data(), Atz.data());
    double r = ninf(Atz.data(), n) / a_scale / beta;
    info[1] = r;
    if (!(r <= 0.9 * EPS)) return false;
    double cd = cone_defect(cones, zo.data(), true) / beta;
    info[2] = cd;
    return cd <= 0.9 * EPS;
  }
  bool check_dinf(const vector<double>& x, const vector<double>& s, vector<double>& xo, vector<double>& so, double* info) {
    xo.resize(n); so.resize(m);
    for (I j = 0; j < n; j++) xo[j] = D[j] * x[j];
    for (I i = 0; i < m; i++) so[i] = s[i] / E[i];
    for (auto& K : cones) if (K.type == CZERO) for (I i = 0; i < K.dim; i++) so[K.start + i] = 0;
    double xn = ninf(xo.data(), n), qn = ninf(pr.q.data(), n);
    if (!(xn > 0 && qn > 0) || !std::isfinite(xn)) return false;
    for (auto& v : xo) v /= xn;
    for (auto& v : so) v /= xn;
    double alpha = -dotp(pr.q.data(), xo.data(), n) / qn;
    info[0] = alpha;
    if (!(alpha > 1e-6)) return false;
    vector<double> Px(n, 0.0), Ax(m, 0.0);
    spmv_add(pr.P, xo.data(), Px.data());
    spmv_add(pr.A, xo.data(), Ax.data());
    double r1 = ninf(Px.data(), n) / h_scale / alpha;
    double r2 = 0;
    for (I i = 0; i < m; i++) r2 = max(r2, fabs(Ax[i] / a_scale + so[i] / a_scale));
    r2 /= alpha;
    info[1] = r1; info[2] = r2;
    if (!(r1 <= 0.9 * EPS && r2 <= 0.9 * EPS)) return false;
    vector<double> sa(so);
    for (auto& v : sa) v /= a_scale;
    double cd = cone_defect(cones, sa.data(), false) / alpha;
    info[3] = cd;
    return cd <= 0.9 * EPS;
  }

  // ---------- main loop ----------
  Result solve() {
    Result res; res.status = "unresolved";
    a_scale = 0; for (double v : pr.A.x) a_scale = max(a_scale, fabs(v)); if (a_scale == 0) a_scale = 1;
    h_scale = 0; for (double v : pr.P.x) h_scale = max(h_scale, fabs(v)); if (h_scale == 0) h_scale = 1;
    equilibrate();
    for (auto& K : cones) {
      if (K.type == CEXP) has_exp = true;
      if (K.type == CNN) nu += K.dim; else if (K.type == CSOC) nu += 1; else if (K.type == CEXP) nu += 3; else if (K.type == CPSD) nu += K.d;
    }
    int nsoc = 0, npsd = 0, nexp = 0;
    for (auto& K : cones) { if (K.type == CSOC) nsoc++; if (K.type == CPSD) npsd++; if (K.type == CEXP) nexp++; }
    soc_w.assign(nsoc, {}); soc_eta.assign(nsoc, 1.0);
    { int is = 0; for (auto& K : cones) if (K.type == CSOC) { soc_w[is].assign(K.dim, 0.0); soc_w[is][0] = 1.0; is++; } }
    psd_r.assign(npsd, {}); psd_rti.assign(npsd, {}); psd_lam.assign(npsd, {});
    exp_g.assign(3 * nexp, 0.0); exp_Hd.assign(9 * nexp, 0.0);
    {
      bool anyPSD = false, okr = true; I nnzpsd = 0;
      vector<char> ispsd(m, 0);
      for (auto& K : cones) { if (K.type == CPSD) { anyPSD = true; for (I r = 0; r < K.dim; r++) ispsd[K.start + r] = 1; } else if (K.type != CNN) okr = false; }
      for (I k = 0; k < A.nnz(); k++) if (ispsd[A.i[k]]) nnzpsd++;
      if (anyPSD && okr && n <= 6000 && (double)nnzpsd * (double)nnzpsd < 4e9) reduced = true;
    }
    if (!reduced) for (auto& K : cones) if (K.type == CPSD && K.dim > 1300) { logf("psd too large\n"); return res; }
    if (reduced) setup_reduced(); else build_kkt();
    vector<double> x(n, 0.0), s(m, 0.0), z(m, 0.0);
    double tau = 1, kap = 1;
    vector<double> rhs(N), sol(N);
    if (!has_exp) {
      // H = I initialization
      for (auto& K : cones) {
        double* H = &Hst[K.hoff];
        if (K.type == CZERO) for (I i = 0; i < K.dim; i++) H[i] = 0;
        else if (K.type == CNN) for (I i = 0; i < K.dim; i++) H[i] = 1;
        else if (!reduced) { for (I i = 0; i < K.dim * K.dim; i++) H[i] = 0; for (I i = 0; i < K.dim; i++) H[i + i * K.dim] = 1; }
      }
      if (reduced) { if (!factor_reduced()) return res; }
      else {
        write_H_to_kkt();
        update_B();
        ldl.factor(Kp, Ki, Kx, sign, 1e-13, 2e-7);
      }
      for (I j = 0; j < n; j++) rhs[j] = -q[j];
      for (I i = 0; i < m; i++) rhs[n + i] = b[i];
      kkt_solve(rhs.data(), sol.data());
      for (I j = 0; j < n; j++) x[j] = sol[j];
      for (I i = 0; i < m; i++) { s[i] = -sol[n + i]; z[i] = sol[n + i]; }
      for (auto& K : cones) if (K.type == CZERO) for (I i = 0; i < K.dim; i++) s[K.start + i] = 0;
      shift_to_cone(s); shift_to_cone(z);
    } else {
      unit_init(s, z);
    }
    vector<double> Px(n), rx(n), rz(m), x1(n), z1(m), x2(n), z2(m), dx(n), dz(m), ds(m), dsa(m), dza(m), dsv(m), Hdz(m), xo, so, zo;
    double info[8];
    int maxit = 200;
    double best_score = 1e300;
    Result best;
    for (int it = 0; it < maxit; it++) {
      if (elapsed() > 280) break;
      // residuals
      fill(Px.begin(), Px.end(), 0.0);
      spmv_add(P, x.data(), Px.data());
      for (I j = 0; j < n; j++) rx[j] = Px[j] + q[j] * tau;
      spmtv_add(A, z.data(), rx.data());
      for (I i = 0; i < m; i++) rz[i] = s[i] - b[i] * tau;
      spmv_add(A, x.data(), rz.data());
      double xPx = dotp(x.data(), Px.data(), n);
      double qx = dotp(q.data(), x.data(), n), bz = dotp(b.data(), z.data(), m);
      double rtau = qx + bz + xPx / tau + kap;
      double sz = dotp(s.data(), z.data(), m);
      double mu = (sz + tau * kap) / (nu + 1);
      // termination checks
      if (!std::isfinite(mu) || !std::isfinite(tau) || !std::isfinite(kap)) { logf("nonfinite\n"); break; }
      bool ok = check_solved(x, s, z, tau, xo, so, zo, info);
      if (ok) { for (double v : xo) if (!std::isfinite(v)) ok = false; for (double v : zo) if (!std::isfinite(v)) ok = false; for (double v : so) if (!std::isfinite(v)) ok = false; }
      logf("it %3d mu %.2e tau %.2e kap %.2e rp %.2e rd %.2e gap %.2e rgap %.2e pobj %.8e t=%.3f\n", it, mu, tau, kap, info[0], info[1], info[2], info[3], info[4], elapsed());
      if (ok) {
        res.status = "solved"; res.x = xo; res.s = so; res.z = zo;
        res.pobj = info[4] + pr.c0; res.dobj = info[5] + pr.c0; res.has_obj = true;
        return res;
      }
      {
        double prog = max(info[0], max(info[1], info[3]));
        (void)prog;
        if (tight_ir) { ir_abs = 1e-13; ir_rel = 1e-14; } else { ir_abs = 1e-12; ir_rel = 1e-12; }
      }
      {
        double sc = max(info[0], max(info[1], min(info[2], info[3])));
        if (sc < best_score && std::isfinite(sc)) { best_score = sc; }
      }
      double pinfo[4], dinfo[4];
      if (bz < 0 && check_pinf(z, zo, pinfo)) {
        res.status = "primal_infeasible"; res.z = zo; return res;
      }
      if (qx < 0 && check_dinf(x, s, xo, so, dinfo)) {
        res.status = "dual_infeasible"; res.x = xo; res.s = so; return res;
      }
      // scaling + factor
      if (!update_scaling(s, z, mu)) { logf("scaling failed\n"); break; }
      if (reduced) { if (!factor_reduced()) { logf("reduced factor failed\n"); break; } }
      else {
      write_H_to_kkt();
      update_B();
      if (g_verbose > 1) {
        double hmin = 1e300, hmax = 0, wmax = 0, smin = 1e300, zmin = 1e300;
        for (auto& K : cones) if (K.type == CNN) for (I r = 0; r < K.dim; r++) { double h = Hst[K.hoff + r]; hmin = min(hmin, h); hmax = max(hmax, h); smin = min(smin, s[K.start + r]); zmin = min(zmin, z[K.start + r]); }
        for (auto& w : soc_w) if (!w.empty()) wmax = max(wmax, w[0]);
        double bmax = 0; for (double v : B.x) bmax = max(bmax, fabs(v));
        logf("     H nn [%.2e, %.2e] smin %.2e zmin %.2e soc w0max %.2e Bmax %.2e\n", hmin, hmax, smin, zmin, wmax, bmax);
      }
      ldl.factor(Kp, Ki, Kx, sign, 1e-13, 2e-7);
      }
      // constant rhs
      for (I j = 0; j < n; j++) rhs[j] = -q[j];
      for (I i = 0; i < m; i++) rhs[n + i] = b[i];
      kkt_solve(rhs.data(), sol.data());
      for (I j = 0; j < n; j++) x2[j] = sol[j];
      for (I i = 0; i < m; i++) z2[i] = sol[n + i];
      // xi = x/tau
      vector<double> qp(n);
      for (I j = 0; j < n; j++) qp[j] = q[j] + 2 * Px[j] / tau;
      double xiPxi = xPx / (tau * tau);
      double den = dotp(qp.data(), x2.data(), n) + dotp(b.data(), z2.data(), m) - xiPxi - kap / tau;
      auto compute_dir = [&](const vector<double>& dxr, const vector<double>& dzr, double dtau_r, double dkap_r, const vector<double>& dsr,
                             double& dtau, double& dkap) {
        for (I j = 0; j < n; j++) rhs[j] = -dxr[j];
        for (I i = 0; i < m; i++) rhs[n + i] = -dzr[i] + dsr[i];
        kkt_solve(rhs.data(), sol.data());
        for (I j = 0; j < n; j++) x1[j] = sol[j];
        for (I i = 0; i < m; i++) z1[i] = sol[n + i];
        double num = -dtau_r + dkap_r / tau - dotp(qp.data(), x1.data(), n) - dotp(b.data(), z1.data(), m);
        dtau = num / den;
        for (I j = 0; j < n; j++) dx[j] = x1[j] + dtau * x2[j];
        for (I i = 0; i < m; i++) dz[i] = z1[i] + dtau * z2[i];
        H_mul(dz.data(), Hdz.data());
        for (I i = 0; i < m; i++) ds[i] = -dsr[i] - Hdz[i];
        if (!soc_start.empty() || reduced) {
          vector<double> adx(m, 0.0);
          spmv_add(A, dx.data(), adx.data());
          for (size_t t = 0; t < soc_start.size(); t++) {
            I st = soc_start[t], d = soc_dim[t];
            for (I r = 0; r < d; r++) { I i = st + r; ds[i] = -dzr[i] - adx[i] + b[i] * dtau; }
          }
          if (reduced) for (auto& K : cones) if (K.type == CPSD)
            for (I r = 0; r < K.dim; r++) { I i = K.start + r; ds[i] = -dzr[i] - adx[i] + b[i] * dtau; }
        }
        dkap = -(dkap_r + kap * dtau) / tau;
        if (g_verbose > 1) {
          vector<double> r2(m, 0.0);
          spmv_add(A, dx.data(), r2.data());
          double e[5] = {0,0,0,0,0};
          for (auto& K : cones) for (I r = 0; r < K.dim; r++) { I i = K.start + r; double v = fabs(r2[i] + ds[i] - b[i] * dtau + dzr[i]); e[K.type] = max(e[K.type], v); }
          logf("     eq2 res zero %.2e nn %.2e soc %.2e\n", e[0], e[1], e[2]);
        }
      };
      // affine
      double dtau, dkap;
      compute_dir(rx, rz, rtau, tau * kap, s, dtau, dkap);
      double amax = 1.0;
      if (dtau < 0) amax = min(amax, -tau / dtau);
      if (dkap < 0) amax = min(amax, -kap / dkap);
      double aa = step_cones(s, z, ds, dz, amax);
      double sigma = pow(1 - aa, 3);
      if (has_exp) sigma = max(sigma, exp_sigmin);
      dsa = ds; dza = dz;
      double dtau_a = dtau, dkap_a = dkap;
      // combined
      vector<double> rxs(n), rzs(m);
      for (I j = 0; j < n; j++) rxs[j] = (1 - sigma) * rx[j];
      for (I i = 0; i < m; i++) rzs[i] = (1 - sigma) * rz[i];
      double dkap_r = tau * kap + dtau_a * dkap_a - sigma * mu;
      if (has_exp) dkap_r = tau * kap - sigma * mu;
      combined_ds(s, z, has_exp ? vector<double>(m, 0.0) : dsa, has_exp ? vector<double>(m, 0.0) : dza, sigma * mu, dsv);
      compute_dir(rxs, rzs, (1 - sigma) * rtau, dkap_r, dsv, dtau, dkap);
      {
        bool fin = std::isfinite(dtau) && std::isfinite(dkap);
        if (fin) for (I j = 0; j < n && fin; j++) if (!std::isfinite(dx[j])) fin = false;
        if (fin) for (I i = 0; i < m && fin; i++) if (!std::isfinite(dz[i]) || !std::isfinite(ds[i])) fin = false;
        if (!fin) {
          if (delta_s < 1e-5) { delta_s *= 100; logf("nonfinite direction: delta_s -> %.1e\n", delta_s); it--; continue; }
          logf("nonfinite direction\n"); break;
        }
      }
      amax = 1.0;
      if (dtau < 0) amax = min(amax, -tau / dtau);
      if (dkap < 0) amax = min(amax, -kap / dkap);
      double alpha = step_cones(s, z, ds, dz, amax);
      alpha = min(1.0, 0.99 * alpha);
      if (has_exp) {
        double a0 = alpha;
        bool found = false;
        for (int t = 0; t < 30; t++) {
          double sz2 = 0;
          for (I i = 0; i < m; i++) sz2 += (s[i] + alpha * ds[i]) * (z[i] + alpha * dz[i]);
          double mun = (sz2 + (tau + alpha * dtau) * (kap + alpha * dkap)) / (nu + 1);
          if (exp_central(s, z, ds, dz, alpha, mun, exp_beta)) { found = true; break; }
          alpha *= 0.8;
        }
        if (!found) alpha = exp_fallback * a0;
      }
      logf("   sigma %.2e aa %.2e alpha %.2e ir %.2e nreg %lld dtau %.2e\n", sigma, aa, alpha, last_ir, (long long)ldl.nreg, dtau);
      if (alpha < 1e-10) { logf("step too small\n"); break; }
      nsmall = (alpha < 1e-3) ? nsmall + 1 : 0;
      if (nsmall >= 15) { logf("stalled\n"); break; }
      for (I j = 0; j < n; j++) x[j] += alpha * dx[j];
      for (I i = 0; i < m; i++) { s[i] += alpha * ds[i]; z[i] += alpha * dz[i]; }
      tau += alpha * dtau; kap += alpha * dkap;
      // normalize to avoid overflow
      double sc = max(tau, kap);
      if (sc > 1e6 || sc < 1e-6) {
        // keep homogeneous scaling bounded
        double f = 1.0 / sc;
        if (sc > 1e6) {
          for (auto& v : x) v *= f; for (auto& v : s) v *= f; for (auto& v : z) v *= f; tau *= f; kap *= f;
        }
      }
    }
    return res;
  }
};

// =====================================================================
static void write_result(const char* path, const Result& r) {
  NpzWriter w;
  string meta = "{\"schema_version\": 1, \"status\": \"" + r.status + "\"";
  if (r.status == "solved" && r.has_obj && std::isfinite(r.pobj) && std::isfinite(r.dobj)) {
    char buf[200];
    snprintf(buf, sizeof(buf), ", \"objective_primal\": %.17g, \"objective_dual\": %.17g", r.pobj, r.dobj);
    meta += buf;
  }
  meta += "}";
  w.add_unicode_scalar("metadata_json", meta);
  if (r.status == "solved") { w.add_double("x", r.x); w.add_double("s", r.s); w.add_double("z", r.z); }
  else if (r.status == "primal_infeasible") { w.add_double("z", r.z); }
  else if (r.status == "dual_infeasible") { w.add_double("x", r.x); w.add_double("s", r.s); }
  w.write(path);
}

int main(int argc, char** argv) {
  g_t0 = chrono::steady_clock::now();
  const char* in = nullptr; const char* out = nullptr;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--input") && i + 1 < argc) in = argv[++i];
    else if (!strcmp(argv[i], "--output") && i + 1 < argc) out = argv[++i];
  }
  if (getenv("CSOLVER_VERBOSE")) g_verbose = atoi(getenv("CSOLVER_VERBOSE"));
  if (!in || !out) { fprintf(stderr, "usage: --input X --output Y\n"); return 0; }
  Result res; res.status = "unresolved";
  try {
    auto arrs = read_npz(in);
    Prob pr;
    pr.q = npy_to_double(arrs.at("q"));
    pr.b = npy_to_double(arrs.at("b"));
    pr.n = (I)pr.q.size(); pr.m = (I)pr.b.size();
    auto mk = [&](const string& pre, I mm_, I nn_) {
      CSC M; M.m = mm_; M.n = nn_;
      M.x = npy_to_double(arrs.at(pre + "_data"));
      auto ii = npy_to_int(arrs.at(pre + "_indices"));
      auto pp = npy_to_int(arrs.at(pre + "_indptr"));
      M.i.assign(ii.begin(), ii.end()); M.p.assign(pp.begin(), pp.end());
      if ((I)M.p.size() != nn_ + 1) { M.p.assign(nn_ + 1, 0); M.i.clear(); M.x.clear(); }
      return csc_canon(M);
    };
    pr.P = mk("P", pr.n, pr.n);
    pr.A = mk("A", pr.m, pr.n);
    string meta = npy_to_string(arrs.at("metadata_json"));
    JParser jp(meta);
    JVal jv = jp.parse();
    const JVal* cj = jv.get("cones");
    I off = 0;
    if (cj) for (auto& c : cj->arr) {
      Cone K; string t = c.get("type")->str; I dim = (I)c.get("dim")->num;
      K.start = off; K.d = dim; K.hoff = 0;
      if (t == "zero") { K.type = CZERO; K.dim = dim; }
      else if (t == "nonnegative") { K.type = CNN; K.dim = dim; }
      else if (t == "soc") { K.type = CSOC; K.dim = dim; }
      else if (t == "exponential") { K.type = CEXP; K.dim = 3; }
      else if (t == "psd") { K.type = CPSD; K.dim = dim * (dim + 1) / 2; }
      else throw runtime_error("unknown cone");
      off += K.dim;
      pr.cones.push_back(K);
    }
    const JVal* oo = jv.get("objective_offset");
    if (oo && oo->t == 2) pr.c0 = oo->num;
    arrs.clear();
    logf("read %.3fs n=%lld m=%lld\n", elapsed(), (long long)pr.n, (long long)pr.m);
    {
      Solver S(pr);
      res = S.solve();
    }
    if (res.status == "unresolved" && elapsed() < 150) {
      logf("retrying with tight iterative refinement\n");
      Solver S2(pr);
      S2.tight_ir = true;
      Result r2 = S2.solve();
      if (r2.status != "unresolved") res = r2;
    }
    bool has_soc = false;
    for (auto& K : pr.cones) if (K.type == CSOC) has_soc = true;
    if (res.status == "unresolved" && has_soc && elapsed() < 150) {
      logf("retrying with H-form SOC blocks\n");
      Solver S3(pr);
      S3.tight_ir = true; S3.wf_max = 0;
      Result r3 = S3.solve();
      if (r3.status != "unresolved") res = r3;
    }
  } catch (std::exception& e) {
    fprintf(stderr, "error: %s\n", e.what());
    res = Result(); res.status = "unresolved";
  }
  write_result(out, res);
  logf("total %.3fs status %s\n", elapsed(), res.status.c_str());
  return 0;
}
