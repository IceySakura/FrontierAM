// Homogeneous self-dual embedding interior-point method for
//   min 1/2 x'Px + q'x  s.t.  Ax + s = b, s in K
// K = product of zero, nonnegative, second-order, exponential and PSD cones.
// Algorithm follows Clarabel (Goulart & Chen 2024): HSDE with quadratic objective,
// Nesterov-Todd scaling for symmetric cones, Dahl-Andersen primal-dual scaling for the
// exponential cone, Mehrotra predictor-corrector, quasi-definite LDL' with AMD ordering,
// static + dynamic regularization and iterative refinement, Ruiz equilibration.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include "npzio.hpp"
#include "amd.hpp"

using std::vector;
using std::string;
typedef int64_t i64;

extern "C" {
void dpotrf_(const char*, const int*, double*, const int*, int*, size_t);
void dgesvd_(const char*, const char*, const int*, const int*, double*, const int*, double*, double*, const int*,
             double*, const int*, double*, const int*, int*, size_t, size_t);
void dsyevr_(const char*, const char*, const char*, const int*, double*, const int*, const double*, const double*,
             const int*, const int*, const double*, int*, double*, double*, const int*, int*, double*, const int*,
             int*, const int*, int*, size_t, size_t, size_t);
void dgemm_(const char*, const char*, const int*, const int*, const int*, const double*, const double*, const int*,
            const double*, const int*, const double*, double*, const int*, size_t, size_t);
}

typedef void (*potrf_t)(const char*, const int*, double*, const int*, int*, size_t);
typedef void (*gesvd_t)(const char*, const char*, const int*, const int*, double*, const int*, double*, double*,
                        const int*, double*, const int*, double*, const int*, int*, size_t, size_t);
typedef void (*syevr_t)(const char*, const char*, const char*, const int*, double*, const int*, const double*,
                        const double*, const int*, const int*, const double*, int*, double*, double*, const int*, int*,
                        double*, const int*, int*, const int*, int*, size_t, size_t, size_t);
typedef void (*gemm_t)(const char*, const char*, const int*, const int*, const int*, const double*, const double*,
                       const int*, const double*, const int*, const double*, double*, const int*, size_t, size_t);
typedef void (*potrs_t)(const char*, const int*, const int*, const double*, const int*, double*, const int*, int*,
                        size_t);
extern "C" void dpotrs_(const char*, const int*, const int*, const double*, const int*, double*, const int*, int*,
                        size_t);
static potrf_t p_potrf = dpotrf_;
static gesvd_t p_gesvd = dgesvd_;
static syevr_t p_syevr = dsyevr_;
static gemm_t p_gemm = dgemm_;
static potrs_t p_potrs = dpotrs_;
#include <dlfcn.h>
#include <glob.h>
// use the optimized OpenBLAS shipped inside scipy if present (single thread); else reference LAPACK
static void load_openblas() {
    static bool done = false;
    if (done) return;
    done = true;
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    glob_t g;
    const char* pats[] = {"/usr/local/lib/python3*/site-packages/scipy.libs/libscipy_openblas-*.so",
                          "/usr/lib/python3*/site-packages/scipy.libs/libscipy_openblas-*.so"};
    for (const char* pat : pats) {
        if (glob(pat, 0, nullptr, &g) == 0 && g.gl_pathc > 0) {
            void* h = dlopen(g.gl_pathv[0], RTLD_NOW | RTLD_LOCAL);
            globfree(&g);
            if (!h) continue;
            void* a = dlsym(h, "scipy_dpotrf_");
            void* b = dlsym(h, "scipy_dgesvd_");
            void* c = dlsym(h, "scipy_dsyevr_");
            void* d = dlsym(h, "scipy_dgemm_");
            void* e = dlsym(h, "scipy_dpotrs_");
            void (*setnt)(int) = (void (*)(int))dlsym(h, "scipy_openblas_set_num_threads");
            if (a && b && c && d && e) {
                if (setnt) setnt(1);
                p_potrf = (potrf_t)a; p_gesvd = (gesvd_t)b; p_syevr = (syevr_t)c; p_gemm = (gemm_t)d; p_potrs = (potrs_t)e;
                return;
            }
        } else globfree(&g);
    }
}

static double wall() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double T0 = 0;
static int VERB = 0;

// ---------------------------------------------------------------- dense helpers (column major)
static void gemm(char ta, char tb, int m, int n, int k, double alpha, const double* A, int lda, const double* B,
                 int ldb, double beta, double* C, int ldc) {
    p_gemm(&ta, &tb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
}
// lower Cholesky in place; returns false if not PD
static bool chol_lower(int d, double* A) {
    int info = 0;
    p_potrf("L", &d, A, &d, &info, 1);
    if (info != 0) return false;
    for (int j = 0; j < d; j++)
        for (int i = 0; i < j; i++) A[i + (size_t)j * d] = 0.0;
    return true;
}
// minimum eigenvalue of symmetric matrix (A destroyed)
static double min_eig(int d, double* A) {
    if (d == 1) return A[0];
    int il = 1, iu = 1, m = 0, info = 0;
    double vl = 0, vu = 0, abstol = 0, w[2];
    vector<int> isuppz(2 * d + 2);
    int lwork = -1, liwork = -1, iwq = 0;
    double wq = 0;
    double zdummy[1];
    int ldz = 1;
    p_syevr("N", "I", "L", &d, A, &d, &vl, &vu, &il, &iu, &abstol, &m, w, zdummy, &ldz, isuppz.data(), &wq, &lwork,
            &iwq, &liwork, &info, 1, 1, 1);
    lwork = (int)wq + 1;
    liwork = iwq + 1;
    vector<double> work(lwork);
    vector<int> iwork(liwork);
    vector<double> wv(d);
    p_syevr("N", "I", "L", &d, A, &d, &vl, &vu, &il, &iu, &abstol, &m, wv.data(), zdummy, &ldz, isuppz.data(),
            work.data(), &lwork, iwork.data(), &liwork, &info, 1, 1, 1);
    if (info != 0) return -1e300;
    return wv[0];
}
// full SVD A = U diag(S) Vt  (A destroyed)
static bool svd(int d, double* A, double* U, double* S, double* Vt) {
    int info = 0, lwork = -1;
    double wq = 0;
    p_gesvd("A", "A", &d, &d, A, &d, S, U, &d, Vt, &d, &wq, &lwork, &info, 1, 1);
    lwork = (int)wq + 1;
    vector<double> work(lwork);
    p_gesvd("A", "A", &d, &d, A, &d, S, U, &d, Vt, &d, work.data(), &lwork, &info, 1, 1);
    return info == 0;
}

// ---------------------------------------------------------------- sparse matrices
struct CSC {
    int nr = 0, nc = 0;
    vector<i64> p;
    vector<int> i;
    vector<double> x;
};

// canonicalize: sort rows in each column, sum duplicates, drop zeros
static CSC canon(int nr, int nc, const vector<i64>& p, const vector<i64>& idx, const vector<double>& val) {
    CSC M;
    M.nr = nr;
    M.nc = nc;
    M.p.assign(nc + 1, 0);
    vector<std::pair<int, double>> buf;
    M.i.reserve(val.size());
    M.x.reserve(val.size());
    for (int j = 0; j < nc; j++) {
        buf.clear();
        for (i64 k = p[j]; k < p[j + 1]; k++) buf.push_back({(int)idx[k], val[k]});
        bool sorted = true;
        for (size_t k = 1; k < buf.size(); k++)
            if (buf[k].first <= buf[k - 1].first) { sorted = false; break; }
        if (!sorted) std::stable_sort(buf.begin(), buf.end(), [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t k = 0; k < buf.size();) {
            size_t l = k;
            double s = 0;
            while (l < buf.size() && buf[l].first == buf[k].first) s += buf[l++].second;
            if (s != 0.0) { M.i.push_back(buf[k].first); M.x.push_back(s); }
            k = l;
        }
        M.p[j + 1] = (i64)M.i.size();
    }
    return M;
}
static CSC transpose(const CSC& A, vector<i64>* map = nullptr) {
    CSC T;
    T.nr = A.nc;
    T.nc = A.nr;
    T.p.assign(A.nr + 1, 0);
    for (size_t k = 0; k < A.i.size(); k++) T.p[A.i[k] + 1]++;
    for (int r = 0; r < A.nr; r++) T.p[r + 1] += T.p[r];
    T.i.resize(A.i.size());
    T.x.resize(A.i.size());
    if (map) map->resize(A.i.size());
    vector<i64> next(T.p.begin(), T.p.end() - 1);
    for (int j = 0; j < A.nc; j++)
        for (i64 k = A.p[j]; k < A.p[j + 1]; k++) {
            i64 pos = next[A.i[k]]++;
            T.i[pos] = j;
            T.x[pos] = A.x[k];
            if (map) (*map)[pos] = k;
        }
    return T;
}
// y = A x (+= if acc)
static void spmv(const CSC& A, const double* x, double* y, bool acc = false) {
    if (!acc) std::fill(y, y + A.nr, 0.0);
    for (int j = 0; j < A.nc; j++) {
        double xj = x[j];
        if (xj == 0.0) continue;
        for (i64 k = A.p[j]; k < A.p[j + 1]; k++) y[A.i[k]] += A.x[k] * xj;
    }
}
// y = A' x
static void spmvT(const CSC& A, const double* x, double* y, bool acc = false) {
    for (int j = 0; j < A.nc; j++) {
        double s = 0;
        for (i64 k = A.p[j]; k < A.p[j + 1]; k++) s += A.x[k] * x[A.i[k]];
        y[j] = acc ? y[j] + s : s;
    }
}
// y = P x where Pu holds the upper triangle (incl diag) of symmetric P
static void symv(const CSC& Pu, const double* x, double* y) {
    std::fill(y, y + Pu.nc, 0.0);
    for (int j = 0; j < Pu.nc; j++) {
        double xj = x[j], s = 0;
        for (i64 k = Pu.p[j]; k < Pu.p[j + 1]; k++) {
            int i = Pu.i[k];
            if (i == j) { s += Pu.x[k] * xj; }
            else { y[i] += Pu.x[k] * xj; s += Pu.x[k] * x[i]; }
        }
        y[j] += s;
    }
}
static double ninf(const double* v, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; i++) m = std::max(m, std::fabs(v[i]));
    return m;
}
static double dot(const double* a, const double* b, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

// ---------------------------------------------------------------- minimal JSON
struct JVal {
    int t = 0;  // 0 null,1 bool,2 num,3 str,4 arr,5 obj
    double num = 0;
    bool b = false;
    string s;
    vector<JVal> arr;
    vector<std::pair<string, JVal>> obj;
    const JVal* get(const string& k) const {
        for (auto& kv : obj)
            if (kv.first == k) return &kv.second;
        return nullptr;
    }
};
struct JParser {
    const string& s;
    size_t p = 0;
    JParser(const string& str) : s(str) {}
    void ws() { while (p < s.size() && isspace((unsigned char)s[p])) p++; }
    JVal parse() {
        ws();
        JVal v;
        if (p >= s.size()) throw std::runtime_error("json eof");
        char c = s[p];
        if (c == '{') {
            v.t = 5; p++; ws();
            if (s[p] == '}') { p++; return v; }
            while (true) {
                ws();
                JVal k = parse();
                ws();
                if (s[p] != ':') throw std::runtime_error("json :");
                p++;
                JVal val = parse();
                v.obj.push_back({k.s, val});
                ws();
                if (s[p] == ',') { p++; continue; }
                if (s[p] == '}') { p++; break; }
                throw std::runtime_error("json obj");
            }
        } else if (c == '[') {
            v.t = 4; p++; ws();
            if (s[p] == ']') { p++; return v; }
            while (true) {
                v.arr.push_back(parse());
                ws();
                if (s[p] == ',') { p++; continue; }
                if (s[p] == ']') { p++; break; }
                throw std::runtime_error("json arr");
            }
        } else if (c == '"') {
            v.t = 3; p++;
            while (p < s.size() && s[p] != '"') {
                if (s[p] == '\\') {
                    p++;
                    char e = s[p];
                    if (e == 'u') { v.s += '?'; p += 4; }
                    else if (e == 'n') v.s += '\n';
                    else if (e == 't') v.s += '\t';
                    else v.s += e;
                    p++;
                } else v.s += s[p++];
            }
            p++;
        } else if (c == 't') { v.t = 1; v.b = true; p += 4; }
        else if (c == 'f') { v.t = 1; v.b = false; p += 5; }
        else if (c == 'n') { v.t = 0; p += 4; }
        else {
            v.t = 2;
            size_t q = p;
            while (q < s.size() && (isdigit((unsigned char)s[q]) || s[q] == '-' || s[q] == '+' || s[q] == '.' ||
                                    s[q] == 'e' || s[q] == 'E'))
                q++;
            v.num = strtod(s.substr(p, q - p).c_str(), nullptr);
            p = q;
        }
        return v;
    }
};

// ---------------------------------------------------------------- cones
enum { CZERO = 0, CNN = 1, CSOC = 2, CEXP = 3, CPSD = 4 };
struct Cone {
    int type, dim, off, d;  // dim = number of rows, d = psd order
    // scaling data
    vector<double> w;       // SOC: wbar (dim); exp: H (9)
    double eta = 1;
    vector<double> R, Rinv, B, lam;  // PSD
    vector<i64> hpos;       // KKT positions of dense upper triangle block (column major upper)
    bool elim = false;      // SOC block eliminated analytically into the x-block
};

static const double EXP_C[3] = {-1.051383945322714, 0.556409619469370, 1.258967884768947};

// exp cone primal interior check
static inline bool exp_primal_ok(const double* s) {
    if (!(s[1] > 0 && s[2] > 0)) return false;
    double psi = s[1] * std::log(s[2] / s[1]) - s[0];
    return psi > 0 && std::isfinite(psi);
}
static inline bool exp_dual_ok(const double* z) {
    if (!(z[0] < 0 && z[2] > 0)) return false;
    double psi = z[1] - z[0] - z[0] * std::log(-z[2] / z[0]);
    return psi > 0 && std::isfinite(psi);
}
// dual barrier gradient and hessian
static void exp_dual_gH(const double* z, double* g, double* H) {
    double z1 = z[0], z2 = z[1], z3 = z[2];
    double l = std::log(-z3 / z1);
    double psi = z2 - z1 - z1 * l;
    double d1 = -l, d3 = -z1 / z3;
    g[0] = l / psi - 1.0 / z1;
    g[1] = -1.0 / psi;
    g[2] = (z1 / z3) / psi - 1.0 / z3;
    if (H) {
        double p2 = psi * psi;
        H[0] = d1 * d1 / p2 - (1.0 / z1) / psi + 1.0 / (z1 * z1);
        H[1] = d1 / p2;
        H[2] = d1 * d3 / p2 + (1.0 / z3) / psi;
        H[4] = 1.0 / p2;
        H[5] = d3 / p2;
        H[8] = d3 * d3 / p2 - (z1 / (z3 * z3)) / psi + 1.0 / (z3 * z3);
        H[3] = H[1]; H[6] = H[2]; H[7] = H[5];
    }
}
static void exp_primal_g(const double* s, double* g) {
    double s1 = s[0], s2 = s[1], s3 = s[2];
    double lg = std::log(s3 / s2);
    double psi = s2 * lg - s1;
    g[0] = 1.0 / psi;
    g[1] = -(lg - 1.0) / psi - 1.0 / s2;
    g[2] = -(s2 / s3) / psi - 1.0 / s3;
}
static bool inv2(const double* M, double* Mi) {  // 2x2 row-major
    double det = M[0] * M[3] - M[1] * M[2];
    if (!(std::fabs(det) > 0) || !std::isfinite(det)) return false;
    Mi[0] = M[3] / det; Mi[1] = -M[1] / det; Mi[2] = -M[2] / det; Mi[3] = M[0] / det;
    return true;
}

// svec index helpers
static inline int sidx(int i, int j) { return j * (j + 1) / 2 + i; }  // i <= j
static const double SQ2 = 1.4142135623730951;
static void smat(int d, const double* v, double* M) {
    for (int j = 0; j < d; j++)
        for (int i = 0; i <= j; i++) {
            double val = v[sidx(i, j)];
            if (i != j) val /= SQ2;
            M[i + (size_t)j * d] = val;
            M[j + (size_t)i * d] = val;
        }
}
static void svec(int d, const double* M, double* v) {
    for (int j = 0; j < d; j++)
        for (int i = 0; i <= j; i++) {
            double val = 0.5 * (M[i + (size_t)j * d] + M[j + (size_t)i * d]);
            v[sidx(i, j)] = (i == j) ? val : val * SQ2;
        }
}

// ---------------------------------------------------------------- LDL' (QDLDL style, up-looking)
struct LDL {
    int N = 0;
    vector<i64> Kp;
    vector<int> Ki;
    vector<double> Kx;  // permuted upper triangle
    vector<int> perm, pinv;
    vector<int> etree;
    vector<i64> Lp;
    vector<int> Li, Lnz;
    vector<double> Lx, D, Dinv;
    vector<double> sign;  // +1/-1 in permuted order
    vector<double> sreg;  // static reg added on diag (permuted order) to subtract in refinement
    vector<i64> diagpos;  // permuted diag positions by permuted index
    // work
    vector<int> yIdx, elimBuf, yMark;
    vector<i64> nextCol;
    vector<double> yVals, wk1, wk2, wk3;
    double dyn_eps = 1e-13, dyn_delta = 2e-7;
    int ir_maxit = 10;
    double ir_reltol = 1e-13;
    double ir_ratio = 5.0;
    int ndyn = 0;

    bool symbolic() {
        etree.assign(N, -1);
        Lnz.assign(N, 0);
        vector<int> work(N, 0);
        for (int j = 0; j < N; j++) {
            work[j] = j;
            for (i64 p = Kp[j]; p < Kp[j + 1]; p++) {
                int i = Ki[p];
                if (i > j) return false;
                while (work[i] != j) {
                    if (etree[i] == -1) etree[i] = j;
                    Lnz[i]++;
                    work[i] = j;
                    i = etree[i];
                }
            }
        }
        Lp.assign(N + 1, 0);
        for (int i = 0; i < N; i++) Lp[i + 1] = Lp[i] + Lnz[i];
        Li.resize(Lp[N]);
        Lx.resize(Lp[N]);
        D.resize(N);
        Dinv.resize(N);
        yIdx.resize(N);
        elimBuf.resize(N);
        yMark.assign(N, 0);
        nextCol.resize(N);
        yVals.assign(N, 0.0);
        wk1.resize(N); wk2.resize(N); wk3.resize(N);
        return true;
    }
    bool numeric() {
        ndyn = 0;
        for (int i = 0; i < N; i++) { nextCol[i] = Lp[i]; }
        for (int k = 0; k < N; k++) {
            int nnzY = 0;
            double dk = 0;
            for (i64 p = Kp[k]; p < Kp[k + 1]; p++) {
                int i = Ki[p];
                if (i == k) { dk += Kx[p]; continue; }
                yVals[i] = Kx[p];
                int nxt = i;
                if (yMark[nxt] == 0) {
                    yMark[nxt] = 1;
                    elimBuf[0] = nxt;
                    int nE = 1;
                    nxt = etree[i];
                    while (nxt != -1 && nxt < k) {
                        if (yMark[nxt] == 1) break;
                        yMark[nxt] = 1;
                        elimBuf[nE++] = nxt;
                        nxt = etree[nxt];
                    }
                    while (nE) yIdx[nnzY++] = elimBuf[--nE];
                }
            }
            for (int ii = nnzY - 1; ii >= 0; ii--) {
                int c = yIdx[ii];
                i64 tmp = nextCol[c];
                double yc = yVals[c];
                for (i64 j = Lp[c]; j < tmp; j++) yVals[Li[j]] -= Lx[j] * yc;
                Li[tmp] = k;
                double lv = yc * Dinv[c];
                Lx[tmp] = lv;
                dk -= yc * lv;
                nextCol[c]++;
                yVals[c] = 0.0;
                yMark[c] = 0;
            }
            if (!(dk * sign[k] > dyn_eps)) {
                if (!std::isfinite(dk)) return false;
                dk = sign[k] * dyn_delta;
                ndyn++;
            }
            D[k] = dk;
            Dinv[k] = 1.0 / dk;
        }
        return true;
    }
    void solve_perm(double* x) const {  // in place, permuted space
        for (int i = 0; i < N; i++) {
            double xi = x[i];
            if (xi != 0.0)
                for (i64 j = Lp[i]; j < Lp[i + 1]; j++) x[Li[j]] -= Lx[j] * xi;
        }
        for (int i = 0; i < N; i++) x[i] *= Dinv[i];
        for (int i = N - 1; i >= 0; i--) {
            double s = x[i];
            for (i64 j = Lp[i]; j < Lp[i + 1]; j++) s -= Lx[j] * x[Li[j]];
            x[i] = s;
        }
    }
    void kmul_perm(const double* x, double* y) const {  // y = (K - static reg) x, permuted
        std::fill(y, y + N, 0.0);
        for (int j = 0; j < N; j++) {
            double xj = x[j], s = 0;
            for (i64 p = Kp[j]; p < Kp[j + 1]; p++) {
                int i = Ki[p];
                double v = Kx[p];
                if (i == j) s += v * xj;
                else { y[i] += v * xj; s += v * x[i]; }
            }
            y[j] += s - sreg[j] * xj;
        }
    }
    // solve K x = b (original ordering) with iterative refinement
    void solve(const double* b, double* x) {
        double* bp = wk1.data();
        double* xp = wk2.data();
        double* r = wk3.data();
        for (int k = 0; k < N; k++) bp[k] = b[perm[k]];
        std::copy(bp, bp + N, xp);
        solve_perm(xp);
        double nb = ninf(bp, N);
        vector<double>& dx = refwork;
        if ((int)dx.size() != N) dx.resize(N);
        kmul_perm(xp, r);
        for (int k = 0; k < N; k++) r[k] = bp[k] - r[k];
        double ne = ninf(r, N);
        for (int it = 0; it < ir_maxit; it++) {
            if (ne <= 1e-12 + ir_reltol * nb) break;
            std::copy(r, r + N, dx.data());
            solve_perm(dx.data());
            for (int k = 0; k < N; k++) dx[k] += xp[k];
            kmul_perm(dx.data(), r);
            for (int k = 0; k < N; k++) r[k] = bp[k] - r[k];
            double ne2 = ninf(r, N);
            if (!(ne2 < ne)) {  // no improvement: keep old, recompute residual not needed
                break;
            }
            std::copy(dx.begin(), dx.end(), xp);
            bool slow = ne / ne2 < ir_ratio;
            ne = ne2;
            if (slow) break;
        }
        for (int k = 0; k < N; k++) x[perm[k]] = xp[k];
    }
    vector<double> refwork;
};

// ---------------------------------------------------------------- solver
struct Solver {
    // original data
    int n = 0, m = 0;
    CSC P0u, A0;  // original (P upper)
    vector<double> q0, b0;
    double offset = 0;
    vector<Cone> cones;
    // scaled data
    CSC Pu, A, At;
    vector<double> q, b, Dsc, Esc;
    double csc = 1;
    double Amax = 1, Pmax = 1;
    // KKT
    LDL K;
    vector<i64> Pdiag_pos;  // KKT (permuted) position of diag of x-block
    vector<i64> Zdiag_pos;  // diag positions for z rows
    double delta_static = 1e-8, static_reg0 = 1e-8;
    int nu = 0;  // degree
    bool has_exp = false, has_psd = false, has_soc = false;
    string status = "unresolved";
    vector<double> xs, ss, zs;  // results
    int iters = 0;

    bool load(const string& path) {
        std::map<string, npz::Array> a;
        string err;
        if (!npz::read_npz(path, a, err)) throw std::runtime_error("read: " + err);
        q0 = npz::to_double(a.at("q"));
        b0 = npz::to_double(a.at("b"));
        n = (int)q0.size();
        m = (int)b0.size();
        {
            auto pd = npz::to_double(a.at("P_data"));
            auto pi = npz::to_int64(a.at("P_indices"));
            auto pp = npz::to_int64(a.at("P_indptr"));
            CSC Pf = canon(n, n, pp, pi, pd);
            // upper triangle
            P0u.nr = n; P0u.nc = n; P0u.p.assign(n + 1, 0);
            for (int j = 0; j < n; j++) {
                for (i64 k = Pf.p[j]; k < Pf.p[j + 1]; k++)
                    if (Pf.i[k] <= j) { P0u.i.push_back(Pf.i[k]); P0u.x.push_back(Pf.x[k]); }
                P0u.p[j + 1] = (i64)P0u.i.size();
            }
            Pmax = ninf(Pf.x.data(), Pf.x.size());
            if (Pmax == 0) Pmax = 1;
        }
        {
            auto ad = npz::to_double(a.at("A_data"));
            auto ai = npz::to_int64(a.at("A_indices"));
            auto ap = npz::to_int64(a.at("A_indptr"));
            A0 = canon(m, n, ap, ai, ad);
            Amax = ninf(A0.x.data(), A0.x.size());
            if (Amax == 0) Amax = 1;
        }
        string meta = npz::to_utf8(a.at("metadata_json"));
        JParser jp(meta);
        JVal root = jp.parse();
        const JVal* off = root.get("objective_offset");
        if (off && off->t == 2) offset = off->num;
        const JVal* cl = root.get("cones");
        int row = 0;
        if (cl)
            for (auto& c : cl->arr) {
                const JVal* t = c.get("type");
                const JVal* d = c.get("dim");
                Cone cn;
                int dim = (int)d->num;
                string ty = t->s;
                cn.d = 0;
                if (ty == "zero") cn.type = CZERO;
                else if (ty == "nonnegative") cn.type = CNN;
                else if (ty == "soc") cn.type = CSOC;
                else if (ty == "exponential") cn.type = CEXP;
                else if (ty == "psd") { cn.type = CPSD; cn.d = dim; dim = dim * (dim + 1) / 2; }
                else throw std::runtime_error("cone type");
                cn.dim = dim;
                cn.off = row;
                row += dim;
                cones.push_back(cn);
            }
        if (row != m) throw std::runtime_error("cone dims mismatch");
        // merge consecutive zero / nn cones
        vector<Cone> mc;
        for (auto& c : cones) {
            if (!mc.empty() && (c.type == CZERO || c.type == CNN) && mc.back().type == c.type) mc.back().dim += c.dim;
            else mc.push_back(c);
        }
        cones.swap(mc);
        return true;
    }

    // ------------------------------------------------ equilibration
    void equilibrate() {
        Pu = P0u; A = A0; q = q0; b = b0;
        Dsc.assign(n, 1.0);
        Esc.assign(m, 1.0);
        vector<double> dw(n), ew(m);
        auto lim = [](double v) { return v < 1e-4 ? 1.0 : (v > 1e4 ? 1e4 : v); };
        int eq_iters = getenv("EQ_ITERS") ? atoi(getenv("EQ_ITERS")) : 10;
        for (int it = 0; it < eq_iters; it++) {
            std::fill(dw.begin(), dw.end(), 0.0);
            std::fill(ew.begin(), ew.end(), 0.0);
            for (int j = 0; j < n; j++) {
                for (i64 k = Pu.p[j]; k < Pu.p[j + 1]; k++) {
                    double v = std::fabs(Pu.x[k]);
                    int i = Pu.i[k];
                    dw[j] = std::max(dw[j], v);
                    dw[i] = std::max(dw[i], v);
                }
                for (i64 k = A.p[j]; k < A.p[j + 1]; k++) {
                    double v = std::fabs(A.x[k]);
                    dw[j] = std::max(dw[j], v);
                    ew[A.i[k]] = std::max(ew[A.i[k]], v);
                }
            }
            double maxdev = 0;
            for (int j = 0; j < n; j++) { dw[j] = 1.0 / std::sqrt(lim(dw[j])); maxdev = std::max(maxdev, std::fabs(1 - dw[j])); }
            for (int i = 0; i < m; i++) { ew[i] = 1.0 / std::sqrt(lim(ew[i])); maxdev = std::max(maxdev, std::fabs(1 - ew[i])); }
            if (it > 0 && maxdev < 1e-3) break;  // converged: further passes change nothing
            // rectify non-separable cones
            for (auto& c : cones) {
                if (c.type == CSOC || c.type == CEXP || c.type == CPSD) {
                    double s = 0;
                    for (int r = 0; r < c.dim; r++) s += ew[c.off + r];
                    s /= c.dim;
                    for (int r = 0; r < c.dim; r++) ew[c.off + r] = s;
                }
            }
            for (int j = 0; j < n; j++) {
                for (i64 k = Pu.p[j]; k < Pu.p[j + 1]; k++) Pu.x[k] *= dw[j] * dw[Pu.i[k]];
                for (i64 k = A.p[j]; k < A.p[j + 1]; k++) A.x[k] *= dw[j] * ew[A.i[k]];
                q[j] *= dw[j];
                Dsc[j] *= dw[j];
            }
            for (int i = 0; i < m; i++) { b[i] *= ew[i]; Esc[i] *= ew[i]; }
            (void)maxdev;
        }
        // cost scaling
        vector<double> pc(n, 0.0);
        for (int j = 0; j < n; j++)
            for (i64 k = Pu.p[j]; k < Pu.p[j + 1]; k++) {
                double v = std::fabs(Pu.x[k]);
                pc[j] = std::max(pc[j], v);
                pc[Pu.i[k]] = std::max(pc[Pu.i[k]], v);
            }
        double mean = 0;
        for (int j = 0; j < n; j++) mean += pc[j];
        mean = n ? mean / n : 0;
        double cn = std::max(mean, ninf(q.data(), n));
        cn = lim(cn);
        csc = 1.0 / cn;
        for (auto& v : Pu.x) v *= csc;
        for (auto& v : q) v *= csc;
        At = transpose(A);
    }

    // ------------------------------------------------ KKT assembly
    vector<i64> Apos;  // KKT position of each A entry (permuted)
    // SOC reduction: SOC z-blocks are eliminated analytically into the x-block
    bool socred = true;
    vector<int> zmap;           // row i -> reduced KKT z index (or -1 if eliminated)
    int mr = 0;                 // number of kept z rows
    struct SocRed { vector<int> cols; vector<double> Ac; vector<i64> pos; };  // Ac: dim x ncols (col-major)
    vector<SocRed> sred;        // per cone (only SOC)
    vector<i64> xpos_touched;   // positions in x-block touched by SOC contributions
    vector<double> xbase, xbaseP;  // base values at those positions (with / without static reg)
    vector<char> xisdiag;
    bool build_kkt() {
        // decide reduced rows: eliminate SOC cones whose rows touch few columns (dense block |J_c|^2)
        zmap.assign(m, -1);
        mr = 0;
        {
            vector<int> mark(n, -1);
            for (size_t ci = 0; ci < cones.size(); ci++) {
                auto& c = cones[ci];
                bool elim = socred && c.type == CSOC;
                if (elim) {
                    int ncols = 0;
                    for (int r = 0; r < c.dim && ncols <= 200; r++)
                        for (i64 k = At.p[c.off + r]; k < At.p[c.off + r + 1]; k++)
                            if (mark[At.i[k]] != (int)ci) { mark[At.i[k]] = (int)ci; ncols++; }
                    if (ncols > 200) elim = false;
                }
                c.elim = elim;
                for (int r = 0; r < c.dim; r++) zmap[c.off + r] = elim ? -1 : mr++;
            }
        }
        int N = n + mr;
        sred.assign(cones.size(), SocRed());
        // collect SOC column sets
        vector<vector<std::pair<int,int>>> extra(n);  // per column j: (row i<=j, tag)
        {
            vector<int> mark(n, -1);
            for (size_t ci = 0; ci < cones.size(); ci++) {
                auto& c = cones[ci];
                if (!c.elim) continue;
                auto& sr = sred[ci];
                for (int r = 0; r < c.dim; r++) {
                    int i = c.off + r;
                    for (i64 k = At.p[i]; k < At.p[i + 1]; k++) {
                        int j = At.i[k];
                        if (mark[j] != (int)ci) { mark[j] = (int)ci; sr.cols.push_back(j); }
                    }
                }
                std::sort(sr.cols.begin(), sr.cols.end());
                int nc = (int)sr.cols.size();
                sr.Ac.assign((size_t)c.dim * nc, 0.0);
                for (int r = 0; r < c.dim; r++) {
                    int i = c.off + r;
                    for (i64 k = At.p[i]; k < At.p[i + 1]; k++) {
                        int j = At.i[k];
                        int a = (int)(std::lower_bound(sr.cols.begin(), sr.cols.end(), j) - sr.cols.begin());
                        sr.Ac[r + (size_t)a * c.dim] += At.x[k];
                    }
                }
            }
        }
        // x-block pattern: P upper + SOC pairs; build per-column sorted unique row lists
        bool anysoc = false;
        for (auto& sr : sred) if (!sr.cols.empty()) anysoc = true;
        vector<vector<int>> xcol(anysoc ? n : 0);
        if (anysoc) {
            for (int j = 0; j < n; j++)
                for (i64 k = Pu.p[j]; k < Pu.p[j + 1]; k++)
                    if (Pu.i[k] != j) xcol[j].push_back(Pu.i[k]);
            for (size_t ci = 0; ci < cones.size(); ci++) {
                auto& sr = sred[ci];
                int nc = (int)sr.cols.size();
                for (int b2 = 0; b2 < nc; b2++)
                    for (int a2 = 0; a2 < b2; a2++) xcol[sr.cols[b2]].push_back(sr.cols[a2]);
            }
            for (int j = 0; j < n; j++) {
                auto& v = xcol[j];
                std::sort(v.begin(), v.end());
                v.erase(std::unique(v.begin(), v.end()), v.end());
            }
        }
        // column counts
        vector<i64> cnt(N + 1, 0);
        for (int j = 0; j < n; j++) {
            if (anysoc) cnt[j] = 1 + (i64)xcol[j].size();
            else {
                i64 c = 1;
                for (i64 k = Pu.p[j]; k < Pu.p[j + 1]; k++) if (Pu.i[k] != j) c++;
                cnt[j] = c;
            }
        }
        vector<int> rowcone(m, -1);
        for (size_t ci = 0; ci < cones.size(); ci++) {
            auto& c = cones[ci];
            for (int r = 0; r < c.dim; r++) rowcone[c.off + r] = (int)ci;
        }
        for (int i = 0; i < m; i++) {
            if (zmap[i] < 0) continue;
            i64 c = 1 + (At.p[i + 1] - At.p[i]);
            auto& cn = cones[rowcone[i]];
            if (cn.type == CSOC || cn.type == CEXP || cn.type == CPSD) c += (i - cn.off);
            cnt[n + zmap[i]] = c;
        }
        vector<i64> Kp(N + 1, 0);
        for (int j = 0; j < N; j++) Kp[j + 1] = Kp[j] + cnt[j];
        i64 nnz = Kp[N];
        vector<int> Ki(nnz);
        vector<i64> Ppos(Pu.i.size()), Apos0(A.i.size(), -1), Dpos(N);
        vector<i64> At2A;
        CSC At_ = transpose(A, &At2A);
        // x columns: rows xcol[j] (sorted) then diagonal
        for (int j = 0; j < n; j++) {
            i64 p = Kp[j];
            if (anysoc) {
                for (int r : xcol[j]) Ki[p++] = r;
                Ki[p] = j; Dpos[j] = p;
                for (i64 k = Pu.p[j]; k < Pu.p[j + 1]; k++) {
                    int r = Pu.i[k];
                    if (r == j) Ppos[k] = p;
                    else Ppos[k] = Kp[j] + (std::lower_bound(xcol[j].begin(), xcol[j].end(), r) - xcol[j].begin());
                }
            } else {
                for (i64 k = Pu.p[j]; k < Pu.p[j + 1]; k++)
                    if (Pu.i[k] != j) { Ki[p] = Pu.i[k]; Ppos[k] = p; p++; }
                Ki[p] = j; Dpos[j] = p;
                for (i64 k = Pu.p[j]; k < Pu.p[j + 1]; k++)
                    if (Pu.i[k] == j) Ppos[k] = p;
            }
        }
        for (size_t ci = 0; ci < cones.size(); ci++) {
            auto& sr = sred[ci];
            int nc = (int)sr.cols.size();
            sr.pos.resize((size_t)nc * (nc + 1) / 2);
            for (int b2 = 0; b2 < nc; b2++) {
                int j = sr.cols[b2];
                for (int a2 = 0; a2 < b2; a2++)
                    sr.pos[(size_t)b2 * (b2 + 1) / 2 + a2] =
                        Kp[j] + (std::lower_bound(xcol[j].begin(), xcol[j].end(), sr.cols[a2]) - xcol[j].begin());
                sr.pos[(size_t)b2 * (b2 + 1) / 2 + b2] = Dpos[j];
            }
        }
        vector<vector<int>>().swap(xcol);
        // z columns
        for (size_t ci = 0; ci < cones.size(); ci++) {
            auto& c = cones[ci];
            if (zmap[c.off] < 0) continue;
            bool dense = (c.type == CSOC || c.type == CEXP || c.type == CPSD);
            if (dense) c.hpos.resize((size_t)c.dim * (c.dim + 1) / 2);
            for (int r = 0; r < c.dim; r++) {
                int i = c.off + r;
                int zi = zmap[i];
                i64 p = Kp[n + zi];
                for (i64 k = At_.p[i]; k < At_.p[i + 1]; k++) { Ki[p] = At_.i[k]; Apos0[At2A[k]] = p; p++; }
                if (dense) {
                    size_t base = (size_t)r * (r + 1) / 2;
                    for (int r2 = 0; r2 < r; r2++) { Ki[p] = n + zmap[c.off + r2]; c.hpos[base + r2] = p; p++; }
                    c.hpos[base + r] = p;
                }
                Ki[p] = n + zi; Dpos[n + zi] = p; p++;
            }
        }
        // ordering
        vector<int> perm;
        double t0 = wall();
        amd::order(N, Kp.data(), Ki.data(), perm, 10.0);
        if (VERB) fprintf(stderr, "amd %.3fs N=%d nnzK=%lld\n", wall() - t0, N, (long long)nnz);
        if ((int)perm.size() != N) return false;
        K.N = N;
        K.perm = perm;
        K.pinv.assign(N, -1);
        for (int k = 0; k < N; k++) {
            if (perm[k] < 0 || perm[k] >= N || K.pinv[perm[k]] != -1) return false;
            K.pinv[perm[k]] = k;
        }
        // permute
        K.Kp.assign(N + 1, 0);
        for (int c = 0; c < N; c++)
            for (i64 p = Kp[c]; p < Kp[c + 1]; p++) {
                int pr = K.pinv[Ki[p]], pc = K.pinv[c];
                K.Kp[std::max(pr, pc) + 1]++;
            }
        for (int c = 0; c < N; c++) K.Kp[c + 1] += K.Kp[c];
        vector<i64> nxt(K.Kp.begin(), K.Kp.end() - 1);
        K.Ki.resize(nnz);
        K.Kx.assign(nnz, 0.0);
        vector<i64> mp(nnz);
        for (int c = 0; c < N; c++)
            for (i64 p = Kp[c]; p < Kp[c + 1]; p++) {
                int pr = K.pinv[Ki[p]], pc = K.pinv[c];
                int col = std::max(pr, pc);
                i64 pos = nxt[col]++;
                K.Ki[pos] = std::min(pr, pc);
                mp[p] = pos;
            }
        Kp.clear(); Kp.shrink_to_fit();
        Ki.clear(); Ki.shrink_to_fit();
        // values
        for (size_t k = 0; k < Pu.i.size(); k++) K.Kx[mp[Ppos[k]]] += Pu.x[k];
        Apos.resize(A.i.size());
        for (size_t k = 0; k < A.i.size(); k++) {
            if (Apos0[k] < 0) { Apos[k] = -1; continue; }
            Apos[k] = mp[Apos0[k]];
            K.Kx[Apos[k]] = A.x[k];
        }
        Pdiag_pos.resize(n);
        for (int j = 0; j < n; j++) Pdiag_pos[j] = mp[Dpos[j]];
        Zdiag_pos.assign(m, -1);
        for (int i = 0; i < m; i++) if (zmap[i] >= 0) Zdiag_pos[i] = mp[Dpos[n + zmap[i]]];
        for (auto& c : cones)
            for (auto& h : c.hpos) h = mp[h];
        for (auto& sr : sred)
            for (auto& h : sr.pos) h = mp[h];
        // signs
        K.sign.resize(N);
        K.sreg.assign(N, 0.0);
        for (int k = 0; k < N; k++) K.sign[k] = perm[k] < n ? 1.0 : -1.0;
        // static reg on x diag
        Pdiag_base.resize(n);
        for (int j = 0; j < n; j++) Pdiag_base[j] = K.Kx[Pdiag_pos[j]];
        double maxd = 0;
        for (int j = 0; j < n; j++) maxd = std::max(maxd, std::fabs(Pdiag_base[j]));
        delta_static = static_reg0 + 4.9e-32 * maxd;
        for (int j = 0; j < n; j++) { K.Kx[Pdiag_pos[j]] = Pdiag_base[j] + delta_static; K.sreg[K.pinv[j]] = delta_static; }
        for (int i = 0; i < m; i++) if (zmap[i] >= 0) K.sreg[K.pinv[n + zmap[i]]] = -delta_static;
        // base values of x-block positions touched by SOC reductions
        xpos_touched.clear();
        for (auto& sr : sred) for (auto h : sr.pos) xpos_touched.push_back(h);
        std::sort(xpos_touched.begin(), xpos_touched.end());
        xpos_touched.erase(std::unique(xpos_touched.begin(), xpos_touched.end()), xpos_touched.end());
        xbase.resize(xpos_touched.size());
        xbaseP.resize(xpos_touched.size());
        xisdiag.assign(xpos_touched.size(), 0);
        {
            vector<char> isd(K.Kx.size() > 0 ? 0 : 0);
            std::vector<i64> dsorted(Pdiag_pos.begin(), Pdiag_pos.end());
            std::sort(dsorted.begin(), dsorted.end());
            for (size_t t = 0; t < xpos_touched.size(); t++) {
                xbase[t] = K.Kx[xpos_touched[t]];
                xisdiag[t] = std::binary_search(dsorted.begin(), dsorted.end(), xpos_touched[t]);
                xbaseP[t] = xbase[t] - (xisdiag[t] ? delta_static : 0.0);
            }
        }
        t0 = wall();
        if (!K.symbolic()) return false;
        if (VERB) fprintf(stderr, "symbolic %.3fs nnzL=%lld\n", wall() - t0, (long long)K.Lp[N]);
        return true;
    }
    vector<double> Pdiag_base;
    void set_static_reg(double d) {
        delta_static = d;
        if (schur) return;
        for (int j = 0; j < n; j++) { K.Kx[Pdiag_pos[j]] = Pdiag_base[j] + delta_static; K.sreg[K.pinv[j]] = delta_static; }
        for (int i = 0; i < m; i++) if (zmap[i] >= 0) K.sreg[K.pinv[n + zmap[i]]] = -delta_static;
        for (size_t t = 0; t < xpos_touched.size(); t++) xbase[t] = xbaseP[t] + (xisdiag[t] ? delta_static : 0.0);
    }

    // ------------------------------------------------ cone operations
    // update scaling; returns false on failure
    bool update_scaling(const double* s, const double* z, double mu) {
        for (auto& c : cones) {
            const double* sc = s + c.off;
            const double* zc = z + c.off;
            if (c.type == CNN) {
                if ((int)c.w.size() != c.dim) c.w.resize(c.dim);
                for (int i = 0; i < c.dim; i++) c.w[i] = sc[i] / zc[i];  // H diag
            } else if (c.type == CSOC) {
                int d = c.dim;
                if ((int)c.w.size() != d) c.w.resize(d);
                double ns = 0, nz = 0;
                for (int i = 1; i < d; i++) { ns += sc[i] * sc[i]; nz += zc[i] * zc[i]; }
                ns = std::sqrt(ns); nz = std::sqrt(nz);
                double sres = (sc[0] - ns) * (sc[0] + ns);
                double zres = (zc[0] - nz) * (zc[0] + nz);
                if (!(sres > 0 && zres > 0)) return false;
                double sr = std::sqrt(sres), zr = std::sqrt(zres);
                double dotsz = 0;
                for (int i = 0; i < d; i++) dotsz += sc[i] * zc[i];
                dotsz /= (sr * zr);
                double gam = std::sqrt((1 + dotsz) / 2);
                c.w[0] = (sc[0] / sr + zc[0] / zr) / (2 * gam);
                for (int i = 1; i < d; i++) c.w[i] = (sc[i] / sr - zc[i] / zr) / (2 * gam);
                // normalize w to have w0^2 - |w1|^2 = 1 (numerical safety, as Clarabel)
                double nw = 0;
                for (int i = 1; i < d; i++) nw += c.w[i] * c.w[i];
                double wres = (c.w[0] - std::sqrt(nw)) * (c.w[0] + std::sqrt(nw));
                if (!(wres > 0)) return false;
                double wn = std::sqrt(wres);
                nw = 0;
                for (int i = 1; i < d; i++) { c.w[i] /= wn; nw += c.w[i] * c.w[i]; }
                c.w[0] = std::sqrt(1 + nw);
                c.eta = std::sqrt(sr / zr);
            } else if (c.type == CEXP) {
                if (c.w.size() != 9) c.w.resize(9);
                double g[3], Hd[9];
                exp_dual_gH(zc, g, Hd);
                double sz = dot(sc, zc, 3);
                double muc = sz / 3.0;
                // primal-dual scaling (Dahl & Andersen)
                double zt[3], st[3];
                exp_primal_g(sc, zt);
                for (int i = 0; i < 3; i++) { zt[i] = -zt[i]; st[i] = -g[i]; }
                double mut = dot(st, zt, 3) / 3.0;
                bool ok = false;
                double Hs[9];
                if (!exp_dual_only && muc * mut - 1 > 1e-8 && std::isfinite(mut)) {
                    double Ha[9];
                    for (int i = 0; i < 9; i++) Ha[i] = muc * Hd[i];
                    // ZS = [z st]' [s st] ... Z = [z zt], S = [s st]
                    double ZS[4] = {sz, dot(zc, st, 3), dot(zt, sc, 3), dot(zt, st, 3)};
                    // symmetrize
                    double off = 0.5 * (ZS[1] + ZS[2]);
                    ZS[1] = ZS[2] = off;
                    double ZSi[4];
                    double HaZ[6];  // 3x2 col-major: Ha z, Ha zt
                    for (int i = 0; i < 3; i++) {
                        HaZ[i] = Ha[i * 3] * zc[0] + Ha[i * 3 + 1] * zc[1] + Ha[i * 3 + 2] * zc[2];
                        HaZ[3 + i] = Ha[i * 3] * zt[0] + Ha[i * 3 + 1] * zt[1] + Ha[i * 3 + 2] * zt[2];
                    }
                    double ZHZ[4] = {dot(zc, HaZ, 3), dot(zc, HaZ + 3, 3), dot(zt, HaZ, 3), dot(zt, HaZ + 3, 3)};
                    double o2 = 0.5 * (ZHZ[1] + ZHZ[2]);
                    ZHZ[1] = ZHZ[2] = o2;
                    double ZHZi[4];
                    if (inv2(ZS, ZSi) && inv2(ZHZ, ZHZi) && ZS[0] * ZS[3] - ZS[1] * ZS[2] > 0) {
                        const double* S0 = sc;
                        const double* S1 = st;
                        for (int i = 0; i < 3; i++)
                            for (int j = 0; j < 3; j++) {
                                double t1 = S0[i] * (ZSi[0] * S0[j] + ZSi[1] * S1[j]) + S1[i] * (ZSi[2] * S0[j] + ZSi[3] * S1[j]);
                                double t2 = HaZ[i] * (ZHZi[0] * HaZ[j] + ZHZi[1] * HaZ[3 + j]) +
                                            HaZ[3 + i] * (ZHZi[2] * HaZ[j] + ZHZi[3] * HaZ[3 + j]);
                                Hs[i * 3 + j] = t1 + Ha[i * 3 + j] - t2;
                            }
                        ok = true;
                        for (int i = 0; i < 9; i++)
                            if (!std::isfinite(Hs[i])) ok = false;
                        // check PD via Cholesky of 3x3
                        if (ok) {
                            double a = Hs[0];
                            if (!(a > 0)) ok = false;
                            else {
                                double l10 = Hs[3] / std::sqrt(a), l20 = Hs[6] / std::sqrt(a);
                                double b2 = Hs[4] - l10 * l10;
                                if (!(b2 > 0)) ok = false;
                                else {
                                    double l21 = (Hs[7] - l20 * l10) / std::sqrt(b2);
                                    double c2 = Hs[8] - l20 * l20 - l21 * l21;
                                    if (!(c2 > 0)) ok = false;
                                }
                            }
                        }
                    }
                }
                if (!ok) {
                    for (int i = 0; i < 9; i++) Hs[i] = mu * Hd[i];
                }
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++) c.w[i * 3 + j] = 0.5 * (Hs[i * 3 + j] + Hs[j * 3 + i]);
            } else if (c.type == CPSD) {
                int d = c.d;
                size_t dd = (size_t)d * d;
                vector<double> Ls(dd), Lz(dd), M(dd), U(dd), Vt(dd), S(d);
                smat(d, sc, Ls.data());
                smat(d, zc, Lz.data());
                if (!chol_lower(d, Ls.data()) || !chol_lower(d, Lz.data())) return false;
                // M = Lz' Ls
                gemm('T', 'N', d, d, d, 1.0, Lz.data(), d, Ls.data(), d, 0.0, M.data(), d);
                if (!svd(d, M.data(), U.data(), S.data(), Vt.data())) return false;
                c.R.assign(dd, 0.0);
                c.Rinv.assign(dd, 0.0);
                c.lam = S;
                // R = Ls V diag(S^-1/2): V = Vt'
                vector<double> V(dd);
                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++) V[i + (size_t)j * d] = Vt[j + (size_t)i * d] / std::sqrt(S[j]);
                gemm('N', 'N', d, d, d, 1.0, Ls.data(), d, V.data(), d, 0.0, c.R.data(), d);
                // Rinv = diag(S^-1/2) U' Lz'
                vector<double> Ut(dd);
                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++) Ut[i + (size_t)j * d] = U[j + (size_t)i * d] / std::sqrt(S[i]);
                gemm('N', 'T', d, d, d, 1.0, Ut.data(), d, Lz.data(), d, 0.0, c.Rinv.data(), d);
                c.B.resize(dd);
                gemm('N', 'T', d, d, d, 1.0, c.R.data(), d, c.R.data(), d, 0.0, c.B.data(), d);
            }
        }
        return true;
    }
    bool init_mode = false;
    double shift_thr = getenv("SHIFT_THR") ? atof(getenv("SHIFT_THR")) : 1e-8;
    double exp_bt = getenv("EXP_BT") ? atof(getenv("EXP_BT")) : 0.8;
    bool exp_dual_only = getenv("EXP_PD") == nullptr;
    double exp_sig_floor = getenv("SIG_FLOOR") ? atof(getenv("SIG_FLOOR")) : 1e-3;
    double soc_shift_frac = getenv("SOC_SHIFT") ? atof(getenv("SOC_SHIFT")) : 0.0;
    // write -(H + delta) into KKT
    // y = H^{-1} x for a SOC cone (exact: eta^-2 (2 Jw (Jw)' - J))
    void soc_Hinv(const Cone& c, const double* x, double* y) {
        if (init_mode) { for (int i = 0; i < c.dim; i++) y[i] = x[i]; return; }
        double e2 = 1.0 / (c.eta * c.eta);
        double t = c.w[0] * x[0];
        for (int i = 1; i < c.dim; i++) t -= c.w[i] * x[i];
        y[0] = e2 * (2 * c.w[0] * t - x[0]);
        for (int i = 1; i < c.dim; i++) y[i] = e2 * (-2 * c.w[i] * t + x[i]);
    }
    void fill_socred() {
        if (sred.empty()) return;
        for (size_t t = 0; t < xpos_touched.size(); t++) K.Kx[xpos_touched[t]] = xbase[t];
        // keep static reg on diagonals that are in xbase (diag base already includes delta)
        for (size_t ci = 0; ci < cones.size(); ci++) {
            auto& sr = sred[ci];
            if (sr.cols.empty()) continue;
            auto& c = cones[ci];
            int d = c.dim, nc = (int)sr.cols.size();
            vector<double> G((size_t)d * nc);
            for (int a = 0; a < nc; a++) soc_Hinv(c, sr.Ac.data() + (size_t)a * d, G.data() + (size_t)a * d);
            for (int b2 = 0; b2 < nc; b2++)
                for (int a2 = 0; a2 <= b2; a2++) {
                    double v = dot(sr.Ac.data() + (size_t)a2 * d, G.data() + (size_t)b2 * d, d);
                    K.Kx[sr.pos[(size_t)b2 * (b2 + 1) / 2 + a2]] += v;
                }
        }
    }
    void fill_H() {
        if (schur) return;
        fill_H_inner();
        fill_socred();
    }
    void fill_H_inner() {
        if (init_mode) {
            for (auto& c : cones) {
                if (zmap[c.off] < 0) continue;
                if (c.type == CZERO) for (int i = 0; i < c.dim; i++) K.Kx[Zdiag_pos[c.off + i]] = -delta_static;
                else if (c.type == CNN) for (int i = 0; i < c.dim; i++) K.Kx[Zdiag_pos[c.off + i]] = -(1.0 + delta_static);
                else {
                    for (int j = 0; j < c.dim; j++)
                        for (int i = 0; i <= j; i++)
                            K.Kx[c.hpos[(size_t)j * (j + 1) / 2 + i]] = (i == j) ? -(1.0 + delta_static) : 0.0;
                }
            }
            return;
        }
        for (auto& c : cones) {
            if (zmap[c.off] < 0) continue;
            if (c.type == CZERO) {
                for (int i = 0; i < c.dim; i++) K.Kx[Zdiag_pos[c.off + i]] = -delta_static;
            } else if (c.type == CNN) {
                for (int i = 0; i < c.dim; i++) K.Kx[Zdiag_pos[c.off + i]] = -(c.w[i] + delta_static);
            } else if (c.type == CSOC) {
                int d = c.dim;
                double e2 = c.eta * c.eta;
                for (int j = 0; j < d; j++)
                    for (int i = 0; i <= j; i++) {
                        double v = 2 * c.w[i] * c.w[j];
                        if (i == j) v += (i == 0 ? -1.0 : 1.0);
                        v *= e2;
                        if (i == j) v += delta_static;
                        K.Kx[c.hpos[(size_t)j * (j + 1) / 2 + i]] = -v;
                    }
            } else if (c.type == CEXP) {
                for (int j = 0; j < 3; j++)
                    for (int i = 0; i <= j; i++) {
                        double v = c.w[i * 3 + j];
                        if (i == j) v += delta_static;
                        K.Kx[c.hpos[(size_t)j * (j + 1) / 2 + i]] = -v;
                    }
            } else if (c.type == CPSD) {
                int d = c.d;
                const double* B = c.B.data();
                int t = c.dim;
                vector<int> I(t), J(t);
                for (int j = 0; j < d; j++)
                    for (int i = 0; i <= j; i++) { I[sidx(i, j)] = i; J[sidx(i, j)] = j; }
                for (int col = 0; col < t; col++) {
                    int k = I[col], l = J[col];
                    for (int row = 0; row <= col; row++) {
                        int i = I[row], j = J[row];
                        double v;
                        double Bik = B[i + (size_t)k * d], Bjl = B[j + (size_t)l * d];
                        if (i == j && k == l) v = Bik * Bik;
                        else if (i == j) v = SQ2 * Bik * B[i + (size_t)l * d];
                        else if (k == l) v = SQ2 * Bik * B[j + (size_t)k * d];
                        else v = Bik * Bjl + B[i + (size_t)l * d] * B[j + (size_t)k * d];
                        if (row == col) v += delta_static;
                        K.Kx[c.hpos[(size_t)col * (col + 1) / 2 + row]] = -v;
                    }
                }
            }
        }
    }
    // y = H x (cone-wise)
    void Hmul(const double* x, double* y) {
        for (auto& c : cones) {
            const double* xc = x + c.off;
            double* yc = y + c.off;
            if (c.type == CZERO) { for (int i = 0; i < c.dim; i++) yc[i] = 0; }
            else if (c.type == CNN) { for (int i = 0; i < c.dim; i++) yc[i] = c.w[i] * xc[i]; }
            else if (c.type == CSOC) {
                // eta^2 (2 w w' - J) x
                double wx = dot(c.w.data(), xc, c.dim);
                double e2 = c.eta * c.eta;
                yc[0] = e2 * (2 * c.w[0] * wx - xc[0]);
                for (int i = 1; i < c.dim; i++) yc[i] = e2 * (2 * c.w[i] * wx + xc[i]);
            } else if (c.type == CEXP) {
                for (int i = 0; i < 3; i++) yc[i] = c.w[i * 3] * xc[0] + c.w[i * 3 + 1] * xc[1] + c.w[i * 3 + 2] * xc[2];
            } else if (c.type == CPSD) {
                int d = c.d;
                size_t dd = (size_t)d * d;
                vector<double> X(dd), T(dd), Y(dd);
                smat(d, xc, X.data());
                gemm('N', 'N', d, d, d, 1.0, c.B.data(), d, X.data(), d, 0.0, T.data(), d);
                gemm('N', 'N', d, d, d, 1.0, T.data(), d, c.B.data(), d, 0.0, Y.data(), d);
                svec(d, Y.data(), yc);
            }
        }
    }
    // SOC helpers: W x with W = eta * Wbar
    static void soc_W(const Cone& c, const double* x, double* y, bool inv) {
        int d = c.dim;
        const double* w = c.w.data();
        double w0 = w[0];
        double w1x1 = 0;
        for (int i = 1; i < d; i++) w1x1 += w[i] * x[i];
        double sg = inv ? -1.0 : 1.0;
        double y0 = w0 * x[0] + sg * w1x1;
        double coef = sg * x[0] + w1x1 / (1 + w0);
        double sc = inv ? 1.0 / c.eta : c.eta;
        y[0] = sc * y0;
        for (int i = 1; i < d; i++) y[i] = sc * (x[i] + coef * w[i]);
    }
    // affine-step ds_tilde = W' lambda (= s for exact NT scaling); computed from the scaling for consistency
    void affine_ds(const double* s, const double* z, double* out) {
        for (auto& c : cones) {
            const double* sc = s + c.off;
            const double* zc = z + c.off;
            double* o = out + c.off;
            if (c.type == CSOC) {
                vector<double> lam(c.dim);
                soc_W(c, zc, lam.data(), false);
                soc_W(c, lam.data(), o, false);
            } else if (c.type == CPSD && !schur) {
                for (int i = 0; i < c.dim; i++) o[i] = sc[i];
            } else {
                for (int i = 0; i < c.dim; i++) o[i] = sc[i];
            }
            (void)zc;
        }
    }
    // combined-step ds_tilde for all cones
    void combined_ds(const double* s, const double* z, const double* dsa, const double* dza, double sigmu,
                     double* out) {
        for (auto& c : cones) {
            const double* sc = s + c.off;
            const double* zc = z + c.off;
            const double* a = dsa + c.off;
            const double* bz = dza + c.off;
            double* o = out + c.off;
            int d = c.dim;
            if (c.type == CZERO) { for (int i = 0; i < d; i++) o[i] = 0; }
            else if (c.type == CNN) {
                for (int i = 0; i < d; i++) o[i] = (sc[i] * zc[i] + a[i] * bz[i] - sigmu) / zc[i];
            } else if (c.type == CSOC) {
                vector<double> lam(d), u(d), v(d), ds(d), x(d);
                soc_W(c, zc, lam.data(), false);
                soc_W(c, a, u.data(), true);   // W^{-1} dsa
                soc_W(c, bz, v.data(), false); // W dza
                // ds = lam o lam + u o v - sigmu e
                double ll = dot(lam.data(), lam.data(), d), uv = dot(u.data(), v.data(), d);
                ds[0] = ll + uv - sigmu;
                for (int i = 1; i < d; i++) ds[i] = 2 * lam[0] * lam[i] + u[0] * v[i] + v[0] * u[i];
                // x = lam \ ds
                double l0 = lam[0], rho = l0 * l0, l1d = 0;
                for (int i = 1; i < d; i++) { rho -= lam[i] * lam[i]; l1d += lam[i] * ds[i]; }
                x[0] = (l0 * ds[0] - l1d) / rho;
                for (int i = 1; i < d; i++) x[i] = (ds[i] - x[0] * lam[i]) / l0;
                soc_W(c, x.data(), o, false);
            } else if (c.type == CEXP) {
                double g[3];
                exp_dual_gH(zc, g, nullptr);
                for (int i = 0; i < 3; i++) o[i] = sc[i] + sigmu * g[i];
            } else if (c.type == CPSD) {
                int dm = c.d;
                size_t dd = (size_t)dm * dm;
                vector<double> As(dd), Bz(dd), T(dd), U(dd), V(dd), Dm(dd);
                smat(dm, a, As.data());
                smat(dm, bz, Bz.data());
                // U = Rinv As Rinv'
                gemm('N', 'N', dm, dm, dm, 1.0, c.Rinv.data(), dm, As.data(), dm, 0.0, T.data(), dm);
                gemm('N', 'T', dm, dm, dm, 1.0, T.data(), dm, c.Rinv.data(), dm, 0.0, U.data(), dm);
                // V = R' Bz R
                gemm('T', 'N', dm, dm, dm, 1.0, c.R.data(), dm, Bz.data(), dm, 0.0, T.data(), dm);
                gemm('N', 'N', dm, dm, dm, 1.0, T.data(), dm, c.R.data(), dm, 0.0, V.data(), dm);
                // Dm = lam^2 + (UV + VU)/2 - sigmu I
                gemm('N', 'N', dm, dm, dm, 0.5, U.data(), dm, V.data(), dm, 0.0, Dm.data(), dm);
                for (int j = 0; j < dm; j++)
                    for (int i = 0; i <= j; i++) {
                        double v = Dm[i + (size_t)j * dm] + Dm[j + (size_t)i * dm];
                        Dm[i + (size_t)j * dm] = v;
                        Dm[j + (size_t)i * dm] = v;
                    }
                for (int i = 0; i < dm; i++) Dm[i + (size_t)i * dm] += c.lam[i] * c.lam[i] - sigmu;
                // X = lam \ Dm : X_ij = 2 D_ij / (l_i + l_j)
                for (int j = 0; j < dm; j++)
                    for (int i = 0; i < dm; i++) Dm[i + (size_t)j * dm] *= 2.0 / (c.lam[i] + c.lam[j]);
                // out = R X R'
                gemm('N', 'N', dm, dm, dm, 1.0, c.R.data(), dm, Dm.data(), dm, 0.0, T.data(), dm);
                gemm('N', 'T', dm, dm, dm, 1.0, T.data(), dm, c.R.data(), dm, 0.0, U.data(), dm);
                svec(dm, U.data(), o);
            }
        }
    }
    // max step for symmetric cones (s side if primal, z side otherwise)
    double step_sym(const double* x, const double* dx, double amax) {
        double a = amax;
        for (auto& c : cones) {
            const double* xc = x + c.off;
            const double* d = dx + c.off;
            if (c.type == CNN) {
                for (int i = 0; i < c.dim; i++)
                    if (d[i] < 0) a = std::min(a, -xc[i] / d[i]);
            } else if (c.type == CSOC) {
                int dm = c.dim;
                double aa = d[0] * d[0], bb = xc[0] * d[0], cc = xc[0] * xc[0];
                for (int i = 1; i < dm; i++) { aa -= d[i] * d[i]; bb -= xc[i] * d[i]; cc -= xc[i] * xc[i]; }
                bb *= 2;
                cc = std::max(cc, 0.0);
                double r = amax;
                if (d[0] < 0) r = std::min(r, -xc[0] / d[0]);
                if (aa == 0) {
                    if (bb < 0) r = std::min(r, -cc / bb);
                } else {
                    double disc = bb * bb - 4 * aa * cc;
                    if (disc >= 0) {
                        double sq = std::sqrt(disc);
                        double t = -0.5 * (bb + (bb >= 0 ? sq : -sq));
                        double r1 = (t != 0) ? cc / t : INFINITY, r2 = t / aa;
                        if (!(aa > 0 && bb > 0)) {
                            if (r1 >= 0) r = std::min(r, r1);
                            if (r2 >= 0) r = std::min(r, r2);
                        }
                    }
                }
                a = std::min(a, r);
            } else if (c.type == CPSD) {
                int dm = c.d;
                size_t dd = (size_t)dm * dm;
                vector<double> L(dd), Dm(dd), T(dd);
                smat(dm, xc, L.data());
                if (!chol_lower(dm, L.data())) { a = 0; continue; }
                smat(dm, d, Dm.data());
                // T = L^{-1} Dm L^{-T}: solve via triangular solves
                // simple: compute Linv
                vector<double> Linv(dd, 0.0);
                for (int j = 0; j < dm; j++) {
                    Linv[j + (size_t)j * dm] = 1.0 / L[j + (size_t)j * dm];
                    for (int i = j + 1; i < dm; i++) {
                        double s = 0;
                        for (int k = j; k < i; k++) s += L[i + (size_t)k * dm] * Linv[k + (size_t)j * dm];
                        Linv[i + (size_t)j * dm] = -s / L[i + (size_t)i * dm];
                    }
                }
                gemm('N', 'N', dm, dm, dm, 1.0, Linv.data(), dm, Dm.data(), dm, 0.0, T.data(), dm);
                gemm('N', 'T', dm, dm, dm, 1.0, T.data(), dm, Linv.data(), dm, 0.0, Dm.data(), dm);
                double lm = min_eig(dm, Dm.data());
                if (lm < 0) a = std::min(a, -1.0 / lm);
            }
        }
        return a;
    }
    bool exp_ok(const double* s, const double* ds, const double* z, const double* dz, double a) {
        for (auto& c : cones)
            if (c.type == CEXP) {
                double sv[3], zv[3];
                for (int i = 0; i < 3; i++) { sv[i] = s[c.off + i] + a * ds[c.off + i]; zv[i] = z[c.off + i] + a * dz[c.off + i]; }
                if (!exp_primal_ok(sv) || !exp_dual_ok(zv)) return false;
            }
        return true;
    }
    double step_length(const double* s, const double* ds, const double* z, const double* dz, double tau, double dtau,
                       double kap, double dkap, double amax) {
        double a = amax;
        if (dtau < 0) a = std::min(a, -tau / dtau);
        if (dkap < 0) a = std::min(a, -kap / dkap);
        a = step_sym(s, ds, a);
        a = step_sym(z, dz, a);
        if (has_exp) {
            int k = 0;
            while (!exp_ok(s, ds, z, dz, a) && k < 200) { a *= exp_bt; k++; }
            if (k >= 200) a = 0;
        }
        return a;
    }
    // shift into cone interior (initialization)
    void shift_to_cone(double* v) {
        for (auto& c : cones) {
            double* vc = v + c.off;
            if (c.type == CZERO) { for (int i = 0; i < c.dim; i++) vc[i] = 0; }
            else if (c.type == CNN) {
                double mn = INFINITY;
                for (int i = 0; i < c.dim; i++) mn = std::min(mn, vc[i]);
                if (mn < shift_thr) for (int i = 0; i < c.dim; i++) vc[i] += 1.0 - mn;
            } else if (c.type == CSOC) {
                double nr = 0;
                for (int i = 1; i < c.dim; i++) nr += vc[i] * vc[i];
                nr = std::sqrt(nr);
                double mn = vc[0] - nr;
                double target = soc_shift_frac * (1.0 + nr);
                if (soc_shift_frac > 0 && mn < target) vc[0] += target - mn;
                else if (mn < shift_thr) vc[0] += 1.0 - mn;
            } else if (c.type == CPSD) {
                int d = c.d;
                vector<double> M((size_t)d * d);
                smat(d, vc, M.data());
                double mn = min_eig(d, M.data());
                if (mn < 1e-8)
                    for (int i = 0; i < d; i++) vc[sidx(i, i)] += 1.0 - mn;
            } else if (c.type == CEXP) {
                for (int i = 0; i < 3; i++) vc[i] = EXP_C[i];
            }
        }
    }
    void unit_init(double* s, double* z) {
        for (auto& c : cones) {
            double* sc = s + c.off;
            double* zc = z + c.off;
            for (int i = 0; i < c.dim; i++) sc[i] = zc[i] = 0;
            if (c.type == CNN) for (int i = 0; i < c.dim; i++) sc[i] = zc[i] = 1;
            else if (c.type == CSOC) sc[0] = zc[0] = 1;
            else if (c.type == CPSD) for (int i = 0; i < c.d; i++) sc[sidx(i, i)] = zc[sidx(i, i)] = 1;
            else if (c.type == CEXP) for (int i = 0; i < 3; i++) sc[i] = zc[i] = EXP_C[i];
        }
    }

    // ------------------------------------------------ main IPM
    void solve() {
        nu = 0;
        for (auto& c : cones) {
            if (c.type == CNN) nu += c.dim;
            else if (c.type == CSOC) { nu += 1; has_soc = true; }
            else if (c.type == CEXP) { nu += 3; has_exp = true; }
            else if (c.type == CPSD) { nu += c.d; has_psd = true; }
        }
        if (getenv("IR_MAXIT")) K.ir_maxit = atoi(getenv("IR_MAXIT"));
        if (getenv("IR_RATIO")) K.ir_ratio = atof(getenv("IR_RATIO"));
        if (getenv("DYN_DELTA")) K.dyn_delta = atof(getenv("DYN_DELTA"));
        if (getenv("DYN_EPS")) K.dyn_eps = atof(getenv("DYN_EPS"));
        if (getenv("STATIC_REG")) static_reg0 = atof(getenv("STATIC_REG"));
        if (has_psd) load_openblas();
        double teq = wall();
        equilibrate();
        if (VERB) fprintf(stderr, "equilibrate %.3fs\n", wall() - teq);
        {
            bool ok = has_psd && !has_soc && !has_exp && n <= 6000;
            int maxt = 0;
            for (auto& c : cones) {
                if (c.type == CZERO) ok = false;
                if (c.type == CPSD) maxt = std::max(maxt, c.dim);
            }
            if (getenv("SOLVER_NOSCHUR")) ok = false;
            schur = ok && maxt >= 300;
        }
        if (!schur) {
            double dense_entries = 0;
            for (auto& c : cones)
                if (c.type == CPSD) dense_entries += 0.5 * (double)c.dim * c.dim;
            if (dense_entries > 1.5e8) throw std::runtime_error("PSD block too large for sparse KKT");
        }
        if (schur) schur_setup();
        else if (!build_kkt()) throw std::runtime_error("kkt build failed");
        int N = n + m;
        vector<double> x(n, 0.0), s(m, 0.0), z(m, 0.0);
        double tau = 1, kap = 1;
        vector<double> rhs(N), sol(N), x1(n), z1(m), x2(n), z2(m);
        // ---- initialization
        auto factor = [&]() -> bool {
            for (int attempt = 0; attempt < 4; attempt++) {
                if (kkt_factor()) return true;
                if (schur) { delta_static *= 100; continue; }
                // increase regularization
                delta_static *= 100;
                for (int j = 0; j < n; j++) { K.Kx[Pdiag_pos[j]] = Pdiag_base[j] + delta_static; K.sreg[K.pinv[j]] = delta_static; }
                for (int i = 0; i < m; i++) if (zmap[i] >= 0) K.sreg[K.pinv[n + zmap[i]]] = -delta_static;
                for (size_t t = 0; t < xpos_touched.size(); t++)
                    xbase[t] = xbaseP[t] + (xisdiag[t] ? delta_static : 0.0);
                fill_H();
            }
            return false;
        };
        bool unit = has_exp || getenv("UNIT_INIT");
        if (unit) {
            unit_init(s.data(), z.data());
        } else {
            // H = I
            init_mode = true;
            fill_H();
            if (!factor()) throw std::runtime_error("init factor failed");
            bool Pzero = Pu.x.empty();
            if (Pzero) {
                for (int j = 0; j < n; j++) rhs[j] = 0;
                for (int i = 0; i < m; i++) rhs[n + i] = b[i];
                kkt_solve(rhs.data(), sol.data());
                for (int j = 0; j < n; j++) x[j] = sol[j];
                for (int i = 0; i < m; i++) s[i] = -sol[n + i];
                for (int j = 0; j < n; j++) rhs[j] = -q[j];
                for (int i = 0; i < m; i++) rhs[n + i] = 0;
                kkt_solve(rhs.data(), sol.data());
                for (int i = 0; i < m; i++) z[i] = sol[n + i];
            } else {
                for (int j = 0; j < n; j++) rhs[j] = -q[j];
                for (int i = 0; i < m; i++) rhs[n + i] = b[i];
                kkt_solve(rhs.data(), sol.data());
                for (int j = 0; j < n; j++) x[j] = sol[j];
                for (int i = 0; i < m; i++) { z[i] = sol[n + i]; s[i] = -sol[n + i]; }
            }
            init_mode = false;
            shift_to_cone(s.data());
            shift_to_cone(z.data());
        }
        // ---- iterations
        vector<double> Px(n), Ax(m), Atz(n), rx(n), rz(m), dx(n), dz(m), ds(m), dsa(m), dza(m), dst(m), Hz(m),
            Px1(n), tmpn(n);
        double nb0 = ninf(b0.data(), m), nq0 = ninf(q0.data(), n);
        double best_score = INFINITY;
        int maxit = 200;
        double prev_mu = INFINITY;
        int stall = 0;
        vector<double> px, ps, pz;
        double ptau = 1, pkap = 1, pscore = INFINITY, damp = 1.0;
        bool have_prev = false;
        int retries = 0;
        for (int it = 0; it <= maxit; it++) {
            iters = it;
            symv(Pu, x.data(), Px.data());
            spmv(A, x.data(), Ax.data());
            spmvT(A, z.data(), Atz.data());
            for (int j = 0; j < n; j++) rx[j] = Px[j] + Atz[j] + q[j] * tau;
            for (int i = 0; i < m; i++) rz[i] = Ax[i] + s[i] - b[i] * tau;
            double xPx = dot(x.data(), Px.data(), n), qx = dot(q.data(), x.data(), n), bz = dot(b.data(), z.data(), m);
            double rtau = qx + bz + xPx / tau + kap;
            double sz = dot(s.data(), z.data(), m);
            double mu = (sz + tau * kap) / (nu + 1);
            // ---- unscaled metrics
            {
                double nAx = 0, ns = 0, npr = 0;
                for (int i = 0; i < m; i++) {
                    double e = 1.0 / Esc[i];
                    nAx = std::max(nAx, std::fabs(Ax[i] * e));
                    ns = std::max(ns, std::fabs(s[i] * e));
                    npr = std::max(npr, std::fabs(rz[i] * e));
                }
                nAx /= tau; ns /= tau; npr /= tau;
                double rp = npr / (1 + std::max(std::max(nAx, ns), nb0));
                double nPx = 0, nAtz = 0, ndr = 0;
                for (int j = 0; j < n; j++) {
                    double e = 1.0 / (Dsc[j] * csc * tau);
                    nPx = std::max(nPx, std::fabs(Px[j] * e));
                    nAtz = std::max(nAtz, std::fabs(Atz[j] * e));
                    ndr = std::max(ndr, std::fabs(rx[j] * e));
                }
                double rd = ndr / (1 + std::max(std::max(nPx, nAtz), nq0));
                double p0 = (0.5 * xPx / tau + qx) / (csc * tau);
                double d0 = (-0.5 * xPx / tau - bz) / (csc * tau);
                double gap = std::fabs(p0 - d0);
                double sc = std::max(std::fabs(p0), std::fabs(d0));
                double rg = sc > 0 ? gap / sc : 0;
                double gm = std::min(gap, rg);
                if (VERB)
                    fprintf(stderr, "%3d mu=%.2e tau=%.2e kap=%.2e rp=%.2e rd=%.2e gap=%.2e p0=%.6e t=%.3f\n", it, mu, tau,
                            kap, rp, rd, gm, p0 + offset, wall() - T0);
                const double TOL = 1e-8 * 0.9;
                if (rp <= TOL && rd <= TOL && gm <= TOL) {
                    vector<double> xo, so, zo;
                    unscale(x, s, z, tau, xo, so, zo);
                    double vr = verify(xo, so, zo);
                    if (VERB) fprintf(stderr, "verify %.3e\n", vr);
                    if (vr <= 0.9e-8) {
                        xs.swap(xo); ss.swap(so); zs.swap(zo);
                        status = "solved";
                        return;
                    }
                }
                double score = std::max(rp, std::max(rd, gm));
                if (score < best_score && score <= 1e-8 * 0.9) {
                    best_score = score;
                    save_best(x, s, z, tau);
                }
                // safeguard: a step that badly worsened a nearly converged iterate is undone and retried
                if (have_prev && pscore < 1e-4 && !(score <= 10 * pscore) && retries < 6) {
                    x = px; s = ps; z = pz; tau = ptau; kap = pkap;
                    retries++;
                    double nd = (retries % 2 == 1) ? delta_static * 10 : delta_static * 1e-2;
                    set_static_reg(nd);
                    damp = 0.6;
                    if (VERB) fprintf(stderr, "revert (score %.2e > prev %.2e), static reg %.1e\n", score, pscore, nd);
                    it--;
                    continue;
                }
                px = x; ps = s; pz = z; ptau = tau; pkap = kap; pscore = score; have_prev = true;
                // infeasibility certificates
                if (check_infeasible(x, s, z, Px, Ax, Atz, qx, bz)) return;
            }
            if (it == maxit) break;
            if (mu < 1e-300 || !std::isfinite(mu)) break;
            // adaptive refinement accuracy: loose while far from optimal, tight near the end
            if (!getenv("IR_FIXED") && !has_soc) K.ir_reltol = std::min(1e-9, std::max(1e-13, 1e-6 * mu));
            // ---- scaling & factorization
            if (!update_scaling(s.data(), z.data(), mu)) { if (VERB) fprintf(stderr, "scaling failed\n"); break; }
            fill_H();
            if (!factor()) { if (VERB) fprintf(stderr, "factor failed\n"); break; }
            // constant system
            for (int j = 0; j < n; j++) rhs[j] = -q[j];
            for (int i = 0; i < m; i++) rhs[n + i] = b[i];
            kkt_solve(rhs.data(), sol.data());
            std::copy(sol.begin(), sol.begin() + n, x1.begin());
            std::copy(sol.begin() + n, sol.end(), z1.begin());
            // denominator
            symv(Pu, x1.data(), Px1.data());
            Hmul(z1.data(), Hz.data());
            double zHz = dot(z1.data(), Hz.data(), m);
            // (x1 - xi)' P (x1 - xi) = x1Px1 - 2 x1'P xi + xi P xi
            double x1Px1 = dot(x1.data(), Px1.data(), n);
            double x1Pxi = dot(x1.data(), Px.data(), n) / tau;
            double xiPxi = xPx / (tau * tau);
            double quad = std::max(0.0, x1Px1 - 2 * x1Pxi + xiPxi);
            double denom = -(zHz + quad + kap / tau);
            // q + 2 P xi
            for (int j = 0; j < n; j++) tmpn[j] = q[j] + 2 * Px[j] / tau;
            auto solve_dir = [&](double sigma, const double* dst_, double dkap_rhs, double* odx, double* odz,
                                 double* ods, double& odtau, double& odkap) {
                double f = 1 - sigma;
                for (int j = 0; j < n; j++) rhs[j] = -f * rx[j];
                for (int i = 0; i < m; i++) rhs[n + i] = -f * rz[i] + dst_[i];
                kkt_solve(rhs.data(), sol.data());
                for (int j = 0; j < n; j++) x2[j] = sol[j];
                for (int i = 0; i < m; i++) z2[i] = sol[n + i];
                double num = -f * rtau + dkap_rhs / tau - dot(tmpn.data(), x2.data(), n) - dot(b.data(), z2.data(), m);
                odtau = num / denom;
                for (int j = 0; j < n; j++) odx[j] = x2[j] + odtau * x1[j];
                for (int i = 0; i < m; i++) odz[i] = z2[i] + odtau * z1[i];
                Hmul(odz, ods);
                for (int i = 0; i < m; i++) ods[i] = -ods[i] - dst_[i];
                for (auto& c : cones)
                    if (c.type == CZERO)
                        for (int i = 0; i < c.dim; i++) ods[c.off + i] = 0;
                odkap = -(dkap_rhs + kap * odtau) / tau;
            };
            // affine
            double dtaua, dkapa;
            affine_ds(s.data(), z.data(), dst.data());
            solve_dir(0.0, dst.data(), tau * kap, dx.data(), dza.data(), dsa.data(), dtaua, dkapa);
            double aa = step_length(s.data(), dsa.data(), z.data(), dza.data(), tau, dtaua, kap, dkapa, 1.0);
            double sigma = std::pow(1 - aa, 3);
            if (has_exp) sigma = std::max(sigma, exp_sig_floor);
            double sigmu = sigma * mu;
            // combined
            combined_ds(s.data(), z.data(), dsa.data(), dza.data(), sigmu, dst.data());
            double dkr = tau * kap + dtaua * dkapa - sigmu;
            if (has_exp) dkr = tau * kap - sigmu;  // no Mehrotra term for nonsymmetric
            double dtau, dkap;
            solve_dir(sigma, dst.data(), dkr, dx.data(), dz.data(), ds.data(), dtau, dkap);
            double a = step_length(s.data(), ds.data(), z.data(), dz.data(), tau, dtau, kap, dkap, 1.0 / 0.99);
            a = std::min(1.0, 0.99 * a) * damp;
            damp = 1.0;
            if (VERB > 1) fprintf(stderr, "    aa=%.3f sigma=%.3e a=%.3f ndyn=%d\n", aa, sigma, a, K.ndyn);
            if (!(a > 0) || !std::isfinite(a)) break;
            for (int j = 0; j < n; j++) x[j] += a * dx[j];
            for (int i = 0; i < m; i++) { s[i] += a * ds[i]; z[i] += a * dz[i]; }
            tau += a * dtau;
            kap += a * dkap;
            if (a < 1e-8) { if (++stall > 5) break; } else stall = 0;
            // normalize homogeneous variables to avoid overflow
            if (tau > 1e8 || tau < 1e-8 ) {
                double nrm = tau + kap;
                (void)nrm;
            }
            (void)prev_mu;
            prev_mu = mu;
        }
        if (!best_x.empty()) {
            xs = best_x; ss = best_s; zs = best_z;
            status = "solved";
        }
    }
    // ------------------------------------------------ dense Schur-complement backend (PSD-heavy problems)
    bool schur = false;
    vector<double> Mden;
    vector<vector<double>> Binv;  // per cone (PSD): (RR')^{-1}
    // per PSD cone: per column list of full-symmetric entries (r,c,v)
    struct Ent { int r, c; double v; };
    vector<vector<vector<Ent>>> colents;
    void schur_setup() {
        colents.assign(cones.size(), {});
        for (size_t ci = 0; ci < cones.size(); ci++) {
            auto& c = cones[ci];
            if (c.type != CPSD) continue;
            int d = c.d;
            vector<int> I(c.dim), J(c.dim);
            for (int j = 0; j < d; j++)
                for (int i = 0; i <= j; i++) { I[sidx(i, j)] = i; J[sidx(i, j)] = j; }
            colents[ci].assign(n, {});
            for (int j = 0; j < n; j++)
                for (i64 k = A.p[j]; k < A.p[j + 1]; k++) {
                    int r = A.i[k];
                    if (r < c.off || r >= c.off + c.dim) continue;
                    int idx = r - c.off;
                    double v = A.x[k];
                    if (I[idx] == J[idx]) colents[ci][j].push_back({I[idx], I[idx], v});
                    else {
                        colents[ci][j].push_back({I[idx], J[idx], v / SQ2});
                        colents[ci][j].push_back({J[idx], I[idx], v / SQ2});
                    }
                }
        }
    }
    void psd_binv_apply(const Cone& c, const double* Bi, const double* x, double* y) {
        int d = c.d;
        size_t dd = (size_t)d * d;
        vector<double> X(dd), T(dd), Y(dd);
        smat(d, x, X.data());
        gemm('N', 'N', d, d, d, 1.0, Bi, d, X.data(), d, 0.0, T.data(), d);
        gemm('N', 'N', d, d, d, 1.0, T.data(), d, Bi, d, 0.0, Y.data(), d);
        svec(d, Y.data(), y);
    }
    // y = H^{-1} x (schur mode cones only: NN / PSD)
    void Hinv(const double* x, double* y) {
        for (size_t ci = 0; ci < cones.size(); ci++) {
            auto& c = cones[ci];
            if (c.type == CNN) {
                for (int i = 0; i < c.dim; i++) y[c.off + i] = init_mode ? x[c.off + i] : x[c.off + i] / c.w[i];
            } else if (c.type == CPSD) {
                if (init_mode) for (int i = 0; i < c.dim; i++) y[c.off + i] = x[c.off + i];
                else psd_binv_apply(c, Binv[ci].data(), x + c.off, y + c.off);
            }
        }
    }
    void Hmul_mode(const double* x, double* y) {
        if (init_mode) { for (int i = 0; i < m; i++) y[i] = x[i]; return; }
        Hmul(x, y);
    }
    bool schur_factor() {
        size_t nn = (size_t)n * n;
        Mden.assign(nn, 0.0);
        for (int j = 0; j < n; j++)
            for (i64 k = Pu.p[j]; k < Pu.p[j + 1]; k++) {
                int i = Pu.i[k];
                Mden[i + (size_t)j * n] += Pu.x[k];
                if (i != j) Mden[j + (size_t)i * n] += Pu.x[k];
            }
        for (int j = 0; j < n; j++) Mden[j + (size_t)j * n] += delta_static;
        Binv.assign(cones.size(), {});
        for (size_t ci = 0; ci < cones.size(); ci++) {
            auto& c = cones[ci];
            if (c.type == CNN) {
                for (int r = 0; r < c.dim; r++) {
                    int i = c.off + r;
                    double wgt = init_mode ? 1.0 : 1.0 / c.w[r];
                    for (i64 k1 = At.p[i]; k1 < At.p[i + 1]; k1++)
                        for (i64 k2 = At.p[i]; k2 < At.p[i + 1]; k2++)
                            Mden[At.i[k1] + (size_t)At.i[k2] * n] += wgt * At.x[k1] * At.x[k2];
                }
            } else if (c.type == CPSD) {
                int d = c.d;
                size_t dd = (size_t)d * d;
                vector<double>& Bi = Binv[ci];
                Bi.assign(dd, 0.0);
                if (init_mode) { for (int i = 0; i < d; i++) Bi[i + (size_t)i * d] = 1.0; }
                else gemm('T', 'N', d, d, d, 1.0, c.Rinv.data(), d, c.Rinv.data(), d, 0.0, Bi.data(), d);
                vector<double> G(dd), S(dd), T(dd);
                auto& ce = colents[ci];
                vector<int> nzcols;
                for (int j = 0; j < n; j++) if (!ce[j].empty()) nzcols.push_back(j);
                for (int j : nzcols) {
                    auto& Sj = ce[j];
                    if ((int)Sj.size() <= d) {
                        std::fill(G.begin(), G.end(), 0.0);
                        for (auto& e : Sj) {
                            const double* br = Bi.data() + (size_t)e.r * d;  // column r (= row r, symmetric)
                            const double* bc = Bi.data() + (size_t)e.c * d;
                            for (int q = 0; q < d; q++) {
                                double f = e.v * bc[q];
                                if (f == 0.0) continue;
                                double* gq = G.data() + (size_t)q * d;
                                for (int p2 = 0; p2 < d; p2++) gq[p2] += br[p2] * f;
                            }
                        }
                    } else {
                        std::fill(S.begin(), S.end(), 0.0);
                        for (auto& e : Sj) S[e.r + (size_t)e.c * d] += e.v;
                        gemm('N', 'N', d, d, d, 1.0, Bi.data(), d, S.data(), d, 0.0, T.data(), d);
                        gemm('N', 'N', d, d, d, 1.0, T.data(), d, Bi.data(), d, 0.0, G.data(), d);
                    }
                    for (int i : nzcols) {
                        double acc = 0;
                        for (auto& e : ce[i]) acc += e.v * G[e.r + (size_t)e.c * d];
                        Mden[i + (size_t)j * n] += acc;
                    }
                }
            }
        }
        // symmetrize
        for (int j = 0; j < n; j++)
            for (int i = j + 1; i < n; i++) {
                double v = 0.5 * (Mden[i + (size_t)j * n] + Mden[j + (size_t)i * n]);
                Mden[i + (size_t)j * n] = v;
            }
        int info = 0, nn2 = n;
        p_potrf("L", &nn2, Mden.data(), &nn2, &info, 1);
        return info == 0;
    }
    void schur_solve_once(const double* r1, const double* r2, double* x, double* z) {
        vector<double> t(m), u(n), Ax(m);
        Hinv(r2, t.data());
        spmvT(A, t.data(), u.data());
        for (int j = 0; j < n; j++) u[j] += r1[j];
        int info = 0, one = 1, nn2 = n;
        p_potrs("L", &nn2, &one, Mden.data(), &nn2, u.data(), &nn2, &info, 1);
        for (int j = 0; j < n; j++) x[j] = u[j];
        spmv(A, x, Ax.data());
        Hinv(Ax.data(), z);
        for (int i = 0; i < m; i++) z[i] -= t[i];
    }
    void schur_solve(const double* rhs, double* sol) {
        const double* r1 = rhs;
        const double* r2 = rhs + n;
        double* x = sol;
        double* z = sol + n;
        schur_solve_once(r1, r2, x, z);
        vector<double> e1(n), e2(m), Px(n), Ax(m), Hz(m), cx(n), cz(m);
        double nb = std::max(ninf(r1, n), ninf(r2, m));
        for (int it = 0; it < 5; it++) {
            symv(Pu, x, Px.data());
            spmvT(A, z, e1.data());
            for (int j = 0; j < n; j++) e1[j] = r1[j] - Px[j] - e1[j];
            spmv(A, x, Ax.data());
            Hmul_mode(z, Hz.data());
            for (int i = 0; i < m; i++) e2[i] = r2[i] - Ax[i] + Hz[i];
            double ne = std::max(ninf(e1.data(), n), ninf(e2.data(), m));
            if (ne <= 1e-12 + 1e-13 * nb) break;
            schur_solve_once(e1.data(), e2.data(), cx.data(), cz.data());
            for (int j = 0; j < n; j++) x[j] += cx[j];
            for (int i = 0; i < m; i++) z[i] += cz[i];
        }
    }
    bool kkt_factor() {
        if (schur) return schur_factor();
        return K.numeric();
    }
    vector<double> red_rhs, red_sol;
    // y = H x with H = I (non-zero cones) during initialization
    void Hmul_full(const double* x, double* y) {
        if (!init_mode) { Hmul(x, y); return; }
        for (auto& c : cones)
            for (int i = 0; i < c.dim; i++) y[c.off + i] = (c.type == CZERO) ? 0.0 : x[c.off + i];
    }
    void kkt_solve(const double* rhs, double* sol) {
        if (schur) { schur_solve(rhs, sol); return; }
        if (mr == m) { K.solve(rhs, sol); return; }
        kkt_solve_red(rhs, sol);
        // refinement on the full (unreduced) system [P A'; A -H]
        int N = n + m;
        vector<double> e(N), c(N), Px(n), Ax(m), Hz(m);
        double nb = ninf(rhs, N);
        auto resid = [&](const double* v) {
            symv(Pu, v, Px.data());
            spmvT(A, v + n, e.data());
            for (int j = 0; j < n; j++) e[j] = rhs[j] - Px[j] - e[j];
            spmv(A, v, Ax.data());
            Hmul_full(v + n, Hz.data());
            for (int i = 0; i < m; i++) e[n + i] = rhs[n + i] - Ax[i] + Hz[i];
            // zero-cone rows: true matrix has 0 there (static reg excluded)
            return ninf(e.data(), N);
        };
        double ne = resid(sol);
        vector<double> trial(N);
        int fmax = getenv("FULL_IR") ? atoi(getenv("FULL_IR")) : 10;
        for (int it = 0; it < fmax; it++) {
            if (ne <= 1e-12 + 1e-13 * nb) break;
            kkt_solve_red(e.data(), c.data());
            for (int k = 0; k < N; k++) trial[k] = sol[k] + c[k];
            double ne2 = resid(trial.data());
            if (!(ne2 < ne)) { resid(sol); break; }
            std::copy(trial.begin(), trial.end(), sol);
            ne = ne2;
        }
    }
    void kkt_solve_red(const double* rhs, double* sol) {
        int N = n + mr;
        red_rhs.resize(N);
        red_sol.resize(N);
        for (int j = 0; j < n; j++) red_rhs[j] = rhs[j];
        for (int i = 0; i < m; i++) if (zmap[i] >= 0) red_rhs[n + zmap[i]] = rhs[n + i];
        // r1 += A_c' H^{-1} r2_c
        vector<double> t(16), u(16);
        for (size_t ci = 0; ci < cones.size(); ci++) {
            auto& sr = sred[ci];
            auto& c = cones[ci];
            if (zmap[c.off] >= 0) continue;
            int d = c.dim, nc = (int)sr.cols.size();
            if ((int)t.size() < d) { t.resize(d); u.resize(d); }
            soc_Hinv(c, rhs + n + c.off, t.data());
            for (int a = 0; a < nc; a++) red_rhs[sr.cols[a]] += dot(sr.Ac.data() + (size_t)a * d, t.data(), d);
        }
        K.solve(red_rhs.data(), red_sol.data());
        for (int j = 0; j < n; j++) sol[j] = red_sol[j];
        for (int i = 0; i < m; i++) if (zmap[i] >= 0) sol[n + i] = red_sol[n + zmap[i]];
        for (size_t ci = 0; ci < cones.size(); ci++) {
            auto& sr = sred[ci];
            auto& c = cones[ci];
            if (zmap[c.off] >= 0) continue;
            int d = c.dim, nc = (int)sr.cols.size();
            // u = A_c x - r2_c ; z_c = H^{-1} u
            for (int r = 0; r < d; r++) u[r] = -rhs[n + c.off + r];
            for (int a = 0; a < nc; a++) {
                double xa = sol[sr.cols[a]];
                for (int r = 0; r < d; r++) u[r] += sr.Ac[r + (size_t)a * d] * xa;
            }
            soc_Hinv(c, u.data(), sol + n + c.off);
        }
    }

    vector<double> best_x, best_s, best_z;
    // checker metrics on original data (max of rp, rd, min(gap_abs, gap_rel))
    double verify(const vector<double>& x, const vector<double>& s, const vector<double>& z) {
        vector<double> Ax(m), Px(n), Atz(n);
        spmv(A0, x.data(), Ax.data());
        symv(P0u, x.data(), Px.data());
        spmvT(A0, z.data(), Atz.data());
        double r1 = 0;
        for (int i = 0; i < m; i++) r1 = std::max(r1, std::fabs(Ax[i] + s[i] - b0[i]));
        double rp = r1 / (1 + std::max(std::max(ninf(Ax.data(), m), ninf(s.data(), m)), ninf(b0.data(), m)));
        double r2 = 0;
        for (int j = 0; j < n; j++) r2 = std::max(r2, std::fabs(Px[j] + q0[j] + Atz[j]));
        double rd = r2 / (1 + std::max(std::max(ninf(Px.data(), n), ninf(q0.data(), n)), ninf(Atz.data(), n)));
        double xPx = dot(x.data(), Px.data(), n), qx = dot(q0.data(), x.data(), n), bz = dot(b0.data(), z.data(), m);
        double p0 = 0.5 * xPx + qx, d0 = -0.5 * xPx - bz;
        double gap = std::fabs(p0 - d0), sc = std::max(std::fabs(p0), std::fabs(d0));
        double rg = sc > 0 ? gap / sc : 0;
        double r = std::max(rp, std::max(rd, std::min(gap, rg)));
        for (int j = 0; j < n; j++) if (!std::isfinite(x[j])) r = INFINITY;
        for (int i = 0; i < m; i++) if (!std::isfinite(s[i]) || !std::isfinite(z[i])) r = INFINITY;
        return std::isfinite(r) ? r : INFINITY;
    }
    void unscale(const vector<double>& x, const vector<double>& s, const vector<double>& z, double tau,
                 vector<double>& xo, vector<double>& so, vector<double>& zo) {
        xo.resize(n); so.resize(m); zo.resize(m);
        for (int j = 0; j < n; j++) xo[j] = Dsc[j] * x[j] / tau;
        for (int i = 0; i < m; i++) { so[i] = s[i] / (Esc[i] * tau); zo[i] = Esc[i] * z[i] / (csc * tau); }
        for (auto& c : cones)
            if (c.type == CZERO)
                for (int i = 0; i < c.dim; i++) so[c.off + i] = 0;
    }
    void save_best(const vector<double>& x, const vector<double>& s, const vector<double>& z, double tau) {
        unscale(x, s, z, tau, best_x, best_s, best_z);
    }
    void finish_solved(const vector<double>& x, const vector<double>& s, const vector<double>& z, double tau) {
        unscale(x, s, z, tau, xs, ss, zs);
        status = "solved";
    }
    bool check_infeasible(const vector<double>& x, const vector<double>& s, const vector<double>& z,
                          const vector<double>& Px, const vector<double>& Ax, const vector<double>& Atz, double qx,
                          double bz) {
        const double TOL = 1e-8 * 0.5;
        // primal infeasibility: z_u = E z (direction)
        if (bz < 0) {
            double nz = 0, nb = 0, bzu = 0;
            for (int i = 0; i < m; i++) { double zu = Esc[i] * z[i]; nz = std::max(nz, std::fabs(zu)); }
            nb = ninf(b0.data(), m);
            if (nz > 0 && nb > 0) {
                for (int i = 0; i < m; i++) bzu += b0[i] * Esc[i] * z[i];
                double beta = -bzu / (nb * nz);
                if (beta > 1e-8) {
                    double nat = 0;
                    for (int j = 0; j < n; j++) nat = std::max(nat, std::fabs(Atz[j] / Dsc[j]));
                    double lin = nat / nz / Amax / beta;
                    if (lin <= TOL) {
                        status = "primal_infeasible";
                        zs.resize(m);
                        for (int i = 0; i < m; i++) zs[i] = Esc[i] * z[i] / nz;
                        return true;
                    }
                }
            }
        }
        if (qx < 0) {
            double nx = 0;
            for (int j = 0; j < n; j++) nx = std::max(nx, std::fabs(Dsc[j] * x[j]));
            double nq = ninf(q0.data(), n);
            if (nx > 0 && nq > 0) {
                double qxu = 0;
                for (int j = 0; j < n; j++) qxu += q0[j] * Dsc[j] * x[j];
                double alpha = -qxu / (nq * nx);
                if (alpha > 1e-8) {
                    double npx = 0;
                    for (int j = 0; j < n; j++) npx = std::max(npx, std::fabs(Px[j] / (Dsc[j] * csc)));
                    double quad = npx / nx / Pmax / alpha;
                    double nl = 0;
                    for (int i = 0; i < m; i++) {
                        bool zero = false;
                        (void)zero;
                        nl = std::max(nl, std::fabs((Ax[i] + s[i]) / Esc[i]));
                    }
                    // zero-cone s must be 0: use Ax only there
                    for (auto& c : cones)
                        if (c.type == CZERO)
                            for (int i = 0; i < c.dim; i++)
                                nl = std::max(nl, std::fabs(Ax[c.off + i] / Esc[c.off + i]));
                    double lin = nl / nx / Amax / alpha;
                    if (quad <= TOL && lin <= TOL) {
                        status = "dual_infeasible";
                        xs.resize(n); ss.resize(m);
                        for (int j = 0; j < n; j++) xs[j] = Dsc[j] * x[j] / nx;
                        for (int i = 0; i < m; i++) ss[i] = s[i] / Esc[i] / nx;
                        for (auto& c : cones)
                            if (c.type == CZERO)
                                for (int i = 0; i < c.dim; i++) ss[c.off + i] = 0;
                        return true;
                    }
                }
            }
        }
        return false;
    }
};

static void write_result(const string& path, const string& status, const vector<double>* x, const vector<double>* s,
                         const vector<double>* z) {
    vector<std::pair<string, const vector<double>*>> arrs;
    if (x) arrs.push_back({"x", x});
    if (s) arrs.push_back({"s", s});
    if (z) arrs.push_back({"z", z});
    string meta = "{\"schema_version\": 1, \"status\": \"" + status + "\"}";
    string err;
    if (!npz::write_npz(path, arrs, meta, err)) fprintf(stderr, "write error: %s\n", err.c_str());
}

int main(int argc, char** argv) {
    T0 = wall();
    string in, out;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--input" && i + 1 < argc) in = argv[++i];
        else if (a == "--output" && i + 1 < argc) out = argv[++i];
        else if (a == "-v") VERB = 1;
        else if (a == "-vv") VERB = 2;
    }
    if (getenv("SOLVER_VERBOSE")) VERB = atoi(getenv("SOLVER_VERBOSE"));
    if (in.empty() || out.empty()) { fprintf(stderr, "usage: solver --input IN --output OUT\n"); return 0; }
    Solver S;
    try {
        S.load(in);
        if (VERB) fprintf(stderr, "loaded n=%d m=%d t=%.3f\n", S.n, S.m, wall() - T0);
        S.solve();
    } catch (std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        if (S.status != "solved" && S.status != "primal_infeasible" && S.status != "dual_infeasible") S.status = "unresolved";
    }
    if (VERB) fprintf(stderr, "status=%s iters=%d time=%.3f\n", S.status.c_str(), S.iters, wall() - T0);
    if (S.status == "solved") write_result(out, "solved", &S.xs, &S.ss, &S.zs);
    else if (S.status == "primal_infeasible") write_result(out, "primal_infeasible", nullptr, nullptr, &S.zs);
    else if (S.status == "dual_infeasible") write_result(out, "dual_infeasible", &S.xs, &S.ss, nullptr);
    else write_result(out, "unresolved", nullptr, nullptr, nullptr);
    return 0;
}
