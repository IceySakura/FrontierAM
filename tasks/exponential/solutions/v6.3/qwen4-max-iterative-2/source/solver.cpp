#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>
#include <lapacke.h>

static bool read_exact(void* dst, size_t n) {
    char* p = static_cast<char*>(dst);
    while (n) {
        ssize_t r = read(0, p, n);
        if (r <= 0) return false;
        p += r;
        n -= static_cast<size_t>(r);
    }
    return true;
}

static bool write_all(const void* src, size_t n) {
    const char* p = static_cast<const char*>(src);
    while (n) {
        ssize_t r = write(1, p, n);
        if (r <= 0) return false;
        p += r;
        n -= static_cast<size_t>(r);
    }
    return true;
}

static std::string json_string(const std::string& s, const char* key) {
    std::string pat = std::string("\"") + key + "\"";
    size_t p = s.find(pat);
    if (p == std::string::npos) return "";
    p = s.find(':', p + pat.size());
    if (p == std::string::npos) return "";
    ++p;
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) ++p;
    if (p >= s.size() || s[p] != '"') return "";
    size_t q = s.find('"', p + 1);
    return s.substr(p + 1, q - p - 1);
}

static double json_number(const std::string& s, const char* key) {
    std::string pat = std::string("\"") + key + "\"";
    size_t p = s.find(pat);
    if (p == std::string::npos) return 0.0;
    p = s.find(':', p + pat.size());
    if (p == std::string::npos) return 0.0;
    return std::strtod(s.c_str() + p + 1, nullptr);
}

static long long json_int(const std::string& s, const char* key) {
    std::string pat = std::string("\"") + key + "\"";
    size_t p = s.find(pat);
    if (p == std::string::npos) return 0;
    p = s.find(':', p + pat.size());
    if (p == std::string::npos) return 0;
    return std::strtoll(s.c_str() + p + 1, nullptr, 10);
}

struct CSR {
    int n = 0;
    std::vector<uint64_t> ptr;
    std::vector<uint32_t> idx;
    std::vector<double> val;
};

static inline void spmv1(const CSR& A, const double* __restrict__ x, double* __restrict__ y) {
    const uint64_t* __restrict__ ptr = A.ptr.data();
    const uint32_t* __restrict__ idx = A.idx.data();
    const double* __restrict__ val = A.val.data();
    for (int i = 0; i < A.n; ++i) {
        double s = 0.0;
        for (uint64_t p = ptr[i]; p < ptr[i + 1]; ++p) s += val[p] * x[idx[p]];
        y[i] = s;
    }
}

static inline void spmv2(const CSR& A, const double* __restrict__ x0, const double* __restrict__ x1, double* __restrict__ y0, double* __restrict__ y1) {
    const uint64_t* __restrict__ ptr = A.ptr.data();
    const uint32_t* __restrict__ idx = A.idx.data();
    const double* __restrict__ val = A.val.data();
    for (int i = 0; i < A.n; ++i) {
        double s0 = 0.0, s1 = 0.0;
        for (uint64_t p = ptr[i]; p < ptr[i + 1]; ++p) {
            double v = val[p];
            uint32_t j = idx[p];
            s0 += v * x0[j];
            s1 += v * x1[j];
        }
        y0[i] = s0;
        y1[i] = s1;
    }
}

static inline double maxabs(const double* x, int n) {
    double m = 0.0;
    for (int i = 0; i < n; ++i) m = std::max(m, std::fabs(x[i]));
    return m;
}

static inline double phi1_neg(double x) {
    if (x < 1e-8) return 1.0 - x * (0.5 - x * (1.0 / 6.0 - x * (1.0 / 24.0)));
    return -std::expm1(-x) / x;
}

static uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

struct Session {
    int n = 0, requests = 0;
    double t = 0.0, rtol = 0.0, atol = 0.0;
    CSR A;
    std::vector<double> diag;
    bool diagonal = true;
    double gersh_upper = 0.0;
    double gersh_lower = 0.0;
    double eta = 0.0;
    double lower = 0.0, width = 0.0, beta_shift = 0.0;
    int mode = 0; // 0 chebyshev, 1 diagonal, 2 blocks, 3 dense
    std::vector<int> singles;
    std::vector<std::pair<int,int>> pairs;
    std::vector<double> df, dg;
    std::vector<int> perm, invperm;
    std::vector<double> work_b0, work_b1, work_y;
    std::vector<double> sf, sg;
    struct BlockCoef { int i, j; double f00, f01, f11, g00, g01, g11; };
    std::vector<BlockCoef> blocks;
    std::vector<double> dense_w, dense_q;
    std::vector<double> ca, cb, tail;
    int kmax = 0;
    double alpha = 0.0;
};

static void analyze(Session& S) {
    const CSR& A = S.A;
    S.diag.assign(S.n, 0.0);
    double upper = 0.0;
    double lower = 1e300;
    bool off = false;
    for (int i = 0; i < S.n; ++i) {
        double sum = 0.0, d = 0.0;
        for (uint64_t p = A.ptr[i]; p < A.ptr[i + 1]; ++p) {
            double v = std::fabs(A.val[p]);
            sum += v;
            if (A.idx[p] == static_cast<uint32_t>(i)) d = A.val[p];
            else off = true;
        }
        S.diag[i] = d;
        upper = std::max(upper, sum);
        lower = std::min(lower, d - (sum - std::fabs(d)));
    }
    S.diagonal = !off;
    S.gersh_upper = upper;
    S.gersh_lower = std::max(0.0, lower);
    S.eta = S.t * upper;
}

static double lanczos_upper(const CSR& A) {
    int n = A.n;
    int m = std::min(std::getenv("FAMEXP_LANCZOS_M") ? std::atoi(std::getenv("FAMEXP_LANCZOS_M")) : 20, n);
    if (m <= 1) return 0.0;
    std::vector<double> V(static_cast<size_t>(m + 1) * n);
    std::vector<double> al(m), be(m), w(n);
    uint64_t seed = 0x123456789abcdefULL ^ static_cast<uint64_t>(n);
    double norm = 0.0;
    for (int i = 0; i < n; ++i) {
        double v = static_cast<double>(splitmix64(seed) >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
        V[i] = v;
        norm += v * v;
    }
    norm = std::sqrt(norm);
    if (norm == 0.0) return 0.0;
    for (int i = 0; i < n; ++i) V[i] /= norm;
    double beta_last = 0.0;
    int steps = 0;
    for (int j = 0; j < m; ++j) {
        const double* vj = V.data() + static_cast<size_t>(j) * n;
        spmv1(A, vj, w.data());
        double a = 0.0;
        for (int i = 0; i < n; ++i) a += vj[i] * w[i];
        al[j] = a;
        for (int i = 0; i < n; ++i) w[i] -= a * vj[i];
        if (j > 0) {
            const double* vp = V.data() + static_cast<size_t>(j - 1) * n;
            for (int i = 0; i < n; ++i) w[i] -= be[j - 1] * vp[i];
        }
        for (int pass = 0; pass < 2; ++pass) {
            for (int k = 0; k <= j; ++k) {
                const double* vk = V.data() + static_cast<size_t>(k) * n;
                double h = 0.0;
                for (int i = 0; i < n; ++i) h += vk[i] * w[i];
                for (int i = 0; i < n; ++i) w[i] -= h * vk[i];
            }
        }
        double b = 0.0;
        for (int i = 0; i < n; ++i) b += w[i] * w[i];
        b = std::sqrt(b);
        be[j] = b;
        beta_last = b;
        steps = j + 1;
        if (b < 1e-13 * std::max(1.0, std::fabs(a))) break;
        if (j + 1 < m) {
            double* vn = V.data() + static_cast<size_t>(j + 1) * n;
            for (int i = 0; i < n; ++i) vn[i] = w[i] / b;
        }
    }
    std::vector<double> d(al.begin(), al.begin() + steps);
    std::vector<double> e(be.begin(), be.begin() + std::max(0, steps - 1));
    if (steps > 1) LAPACKE_dsterf(steps, d.data(), e.data());
    double theta = d.empty() ? 0.0 : *std::max_element(d.begin(), d.end());
    return std::max(0.0, theta + beta_last);
}

static void prepare_cheb(Session& S) {
    double t0 = static_cast<double>(clock()) / CLOCKS_PER_SEC;
    double upper = S.gersh_upper;
    if (S.n > 1 && S.eta > 0.0) {
        double lu = lanczos_upper(S.A);
        if (lu > 0.0) upper = std::min(upper, lu);
    }
    S.eta = S.t * upper * (1.0 + 1e-6);
    if (S.eta <= 0.0) return;
    double a = S.t * S.gersh_lower;
    if (a <= 0.0 || S.eta - a < 0.05 * S.eta) a = 0.0;
    S.lower = a;
    S.width = S.eta - a;
    S.beta_shift = 2.0 * a / S.width;
    double logeps = std::log(1.0 / S.rtol);
    S.kmax = std::max(16, static_cast<int>(std::ceil(1.2 * std::sqrt(S.width * logeps))) + 40);
    if (a > 0.0) {
        double sa = std::sqrt(a), sb = std::sqrt(S.eta);
        double rho = (sb + sa) / (sb - sa);
        if (rho > 1.0) {
            int k2 = std::max(16, static_cast<int>(std::ceil(1.5 * logeps / std::log(rho))) + 30);
            S.kmax = std::min(S.kmax, k2);
        }
    }
    int K = S.kmax;
    int N = K + 64;
    S.ca.assign(K + 1, 0.0);
    S.cb.assign(K + 1, 0.0);
    std::vector<long double> la(K + 1, 0.0L), lb(K + 1, 0.0L);
    double half = S.width * 0.5;
    double center = (S.lower + S.eta) * 0.5;
    double pi = std::acos(-1.0);
    for (int j = 0; j < N; ++j) {
        double theta = pi * (static_cast<double>(j) + 0.5) / static_cast<double>(N);
        double x = center + half * std::cos(theta);
        double f = std::exp(-x);
        double g;
        if (x < 1e-8) g = 1.0 - x * (0.5 - x * (1.0 / 6.0 - x * (1.0 / 24.0)));
        else g = -std::expm1(-x) / x;
        double cd = std::cos(theta), sd = std::sin(theta);
        double c = 1.0, s = 0.0;
        for (int k = 0; k <= K; ++k) {
            la[k] += f * c;
            lb[k] += g * c;
            double nc = c * cd - s * sd;
            double ns = s * cd + c * sd;
            c = nc;
            s = ns;
        }
    }
    double fac = 2.0 / static_cast<double>(N);
    for (int k = 0; k <= K; ++k) {
        S.ca[k] = static_cast<double>(la[k]) * fac;
        S.cb[k] = static_cast<double>(lb[k]) * fac;
    }
    S.ca[0] *= 0.5;
    S.cb[0] *= 0.5;
    S.alpha = 2.0 * S.t / S.width;
    if (std::getenv("FAMEXP_DEBUG")) {
        double t1 = static_cast<double>(clock()) / CLOCKS_PER_SEC;
        std::fprintf(stderr, "cheb prep eta=%.6g kmax=%d coeff_cpu=%.4f total_cpu=%.4f\n", S.eta, S.kmax, t1 - t0, t1);
    }
}

static void maybe_permute(Session& S) {
    int n = S.n;
    if (n < 5000) return;
    int mindeg = n, maxdeg = 0;
    for (int i = 0; i < n; ++i) {
        int d = static_cast<int>(S.A.ptr[i + 1] - S.A.ptr[i]);
        mindeg = std::min(mindeg, d);
        maxdeg = std::max(maxdeg, d);
    }
    if (maxdeg <= 3 * mindeg + 5) return;
    std::vector<int> deg(n), count(static_cast<size_t>(maxdeg) + 1, 0);
    for (int i = 0; i < n; ++i) {
        deg[i] = static_cast<int>(S.A.ptr[i + 1] - S.A.ptr[i]);
        ++count[deg[i]];
    }
    for (int d = 1; d <= maxdeg; ++d) count[d] += count[d - 1];
    std::vector<int> order(n);
    for (int i = n - 1; i >= 0; --i) order[--count[deg[i]]] = i;
    std::vector<int> inv(n);
    for (int i = 0; i < n; ++i) inv[order[i]] = i;
    CSR B;
    B.n = n;
    B.ptr.resize(static_cast<size_t>(n) + 1);
    B.idx.resize(S.A.idx.size());
    B.val.resize(S.A.val.size());
    uint64_t p = 0;
    for (int i = 0; i < n; ++i) {
        int r = order[i];
        B.ptr[i] = p;
        for (uint64_t q = S.A.ptr[r]; q < S.A.ptr[r + 1]; ++q) {
            B.idx[p] = static_cast<uint32_t>(inv[S.A.idx[q]]);
            B.val[p] = S.A.val[q];
            ++p;
        }
    }
    B.ptr[n] = p;
    S.A = std::move(B);
    S.perm = std::move(order);
    S.invperm = std::move(inv);
}

static bool detect_blocks(Session& S) {
    int n = S.n;
    std::vector<char> seen(n, 0);
    std::vector<int> stack;
    stack.reserve(64);
    for (int start = 0; start < n; ++start) {
        if (seen[start]) continue;
        stack.clear();
        stack.push_back(start);
        seen[start] = 1;
        int nodes[2];
        int cnt = 0;
        bool ok = true;
        while (!stack.empty()) {
            int v = stack.back();
            stack.pop_back();
            if (cnt < 2) nodes[cnt] = v;
            ++cnt;
            if (cnt > 2) { ok = false; break; }
            for (uint64_t p = S.A.ptr[v]; p < S.A.ptr[v + 1]; ++p) {
                int u = static_cast<int>(S.A.idx[p]);
                if (!seen[u]) {
                    seen[u] = 1;
                    stack.push_back(u);
                }
            }
        }
        if (!ok) return false;
        if (cnt == 1) S.singles.push_back(nodes[0]);
        else S.pairs.emplace_back(nodes[0], nodes[1]);
    }
    return true;
}

static double get_entry(const CSR& A, int i, int j);

static void prepare_blocks(Session& S) {
    long double t = S.t;
    S.sf.resize(S.singles.size());
    S.sg.resize(S.singles.size());
    for (size_t k = 0; k < S.singles.size(); ++k) {
        int i = S.singles[k];
        long double x = t * S.diag[i];
        long double f = expl(-x);
        long double g = (x < 1e-8L) ? 1.0L - x * (0.5L - x / 6.0L) : -expm1l(-x) / x;
        S.sf[k] = static_cast<double>(f);
        S.sg[k] = static_cast<double>(t * g);
    }
    S.blocks.resize(S.pairs.size());
    for (size_t k = 0; k < S.pairs.size(); ++k) {
        auto [i, j] = S.pairs[k];
        long double a = get_entry(S.A, i, i);
        long double b = get_entry(S.A, i, j);
        long double d = get_entry(S.A, j, j);
        long double mean = (a + d) * 0.5L;
        long double disc = sqrtl(((a - d) * 0.5L) * ((a - d) * 0.5L) + b * b);
        long double l1 = mean - disc, l2 = mean + disc;
        if (l1 < 0.0L && l1 > -1e-12L * std::max(1.0L, fabsl(mean))) l1 = 0.0L;
        long double x1 = t * l1, x2 = t * l2;
        long double f1 = expl(-x1), f2 = expl(-x2);
        long double g1 = (x1 < 1e-8L) ? 1.0L - x1 * (0.5L - x1 / 6.0L) : -expm1l(-x1) / x1;
        long double g2 = (x2 < 1e-8L) ? 1.0L - x2 * (0.5L - x2 / 6.0L) : -expm1l(-x2) / x2;
        long double delta = x2 - x1;
        long double bf, bg;
        if (delta > 1e-10L * std::max(1.0L, x2)) {
            bf = (f2 - f1) / delta;
            bg = (g2 - g1) / delta;
        } else {
            long double xm = (x1 + x2) * 0.5L;
            bf = -expl(-xm);
            if (xm < 1e-4L) bg = -0.5L + xm * (1.0L / 3.0L - xm * (1.0L / 8.0L));
            else bg = ((xm + 1.0L) * expl(-xm) - 1.0L) / (xm * xm);
        }
        long double af = f1 - bf * x1;
        long double ag = g1 - bg * x1;
        long double ta = t * a, tb = t * b, td = t * d;
        Session::BlockCoef c;
        c.i = i; c.j = j;
        c.f00 = static_cast<double>(af + bf * ta);
        c.f01 = static_cast<double>(bf * tb);
        c.f11 = static_cast<double>(af + bf * td);
        c.g00 = static_cast<double>(t * ag + t * bg * ta);
        c.g01 = static_cast<double>(t * bg * tb);
        c.g11 = static_cast<double>(t * ag + t * bg * td);
        S.blocks[k] = c;
    }
}

static double get_entry(const CSR& A, int i, int j) {
    for (uint64_t p = A.ptr[i]; p < A.ptr[i + 1]; ++p)
        if (A.idx[p] == static_cast<uint32_t>(j)) return A.val[p];
    return 0.0;
}

static void solve_blocks(Session& S, const double* b0, const double* b1, double* y) {
    for (size_t k = 0; k < S.singles.size(); ++k) {
        int i = S.singles[k];
        y[i] = S.sf[k] * b0[i] + S.sg[k] * b1[i];
    }
    for (const auto& c : S.blocks) {
        int i = c.i, j = c.j;
        y[i] = c.f00 * b0[i] + c.f01 * b0[j] + c.g00 * b1[i] + c.g01 * b1[j];
        y[j] = c.f01 * b0[i] + c.f11 * b0[j] + c.g01 * b1[i] + c.g11 * b1[j];
    }
}

static void prepare_dense(Session& S) {
    int n = S.n;
    S.dense_w.resize(n);
    S.dense_q.resize(static_cast<size_t>(n) * n);
    for (int i = 0; i < n; ++i)
        for (uint64_t p = S.A.ptr[i]; p < S.A.ptr[i + 1]; ++p)
            S.dense_q[static_cast<size_t>(i) + static_cast<size_t>(S.A.idx[p]) * n] = S.t * S.A.val[p];
    LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'L', n, S.dense_q.data(), n, S.dense_w.data());
    for (double& w : S.dense_w) if (w < 0.0) w = 0.0;
}

static void solve_dense(Session& S, const double* b0, const double* b1, double* y) {
    int n = S.n;
    std::vector<double> c0(n), c1(n);
    for (int j = 0; j < n; ++j) {
        double s0 = 0.0, s1 = 0.0;
        const double* q = S.dense_q.data() + static_cast<size_t>(j) * n;
        for (int i = 0; i < n; ++i) {
            s0 += q[i] * b0[i];
            s1 += q[i] * b1[i];
        }
        double x = S.dense_w[j];
        double f = std::exp(-x);
        double g = phi1_neg(x);
        c0[j] = f * s0;
        c1[j] = S.t * g * s1;
    }
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) s += S.dense_q[static_cast<size_t>(i) + static_cast<size_t>(j) * n] * (c0[j] + c1[j]);
        y[i] = s;
    }
}

static void solve_taylor(Session& S, const double* b0, const double* b1, double* y) {
    int n = S.n;
    bool u0 = maxabs(b0, n) > 0.0, u1 = maxabs(b1, n) > 0.0 && S.t > 0.0;
    std::vector<double> t0(n), t1(n), r0(n), r1(n);
    for (int i = 0; i < n; ++i) {
        t0[i] = u0 ? b0[i] : 0.0;
        t1[i] = u1 ? S.t * b1[i] : 0.0;
        y[i] = t0[i] + t1[i];
    }
    if (!u0 && !u1) return;
    int min_k = std::max(1, static_cast<int>(std::ceil(S.eta)));
    for (int k = 1; k <= 100; ++k) {
        if (u0 && u1) spmv2(S.A, t0.data(), t1.data(), r0.data(), r1.data());
        else if (u0) spmv1(S.A, t0.data(), r0.data());
        else spmv1(S.A, t1.data(), r1.data());
        double inv0 = -S.t / k, inv1 = -S.t / (k + 1);
        double term = 0.0, ymax = 0.0;
        for (int i = 0; i < n; ++i) {
            if (u0) t0[i] = inv0 * r0[i];
            if (u1) t1[i] = inv1 * r1[i];
            y[i] += t0[i] + t1[i];
            term = std::max(term, std::fabs(t0[i]) + std::fabs(t1[i]));
            ymax = std::max(ymax, std::fabs(y[i]));
        }
        if (k >= min_k && term < 0.01 * (S.atol + S.rtol * ymax)) break;
    }
}

static void solve_cheb(Session& S, const double* b0, const double* b1, double* y) {
    int n = S.n;
    bool u0 = maxabs(b0, n) > 0.0, u1 = maxabs(b1, n) > 0.0 && S.t > 0.0;
    if (u0 && S.lower > 0.0) {
        double b0max = maxabs(b0, n);
        double fmax = std::exp(-S.lower) * b0max;
        double scale = b0max + (u1 ? S.t * maxabs(b1, n) : 0.0);
        if (fmax == 0.0 || fmax * n < 0.01 * (S.atol + S.rtol * scale)) u0 = false;
    }
    if (!u0 && !u1) { std::fill(y, y + n, 0.0); return; }
    if (S.eta < 2.0) { solve_taylor(S, b0, b1, y); return; }
    S.tail.assign(S.kmax + 2, 0.0);
    for (int k = S.kmax - 1; k >= 0; --k) {
        S.tail[k] = S.tail[k + 1] + (u0 ? std::fabs(S.ca[k + 1]) : 0.0) + (u1 ? std::fabs(S.cb[k + 1]) : 0.0);
    }
    std::vector<double> p0(n), p1(n), p2(n), q0(n), q1(n), q2(n), r0(n), r1(n);
    std::vector<double> yprev(n, 0.0);
    double alpha = S.alpha;
    if (u0) {
        spmv1(S.A, b0, r0.data());
        for (int i = 0; i < n; ++i) {
            p0[i] = b0[i];
            p1[i] = alpha * r0[i] - (S.beta_shift + 1.0) * b0[i];
            y[i] = S.ca[0] * p0[i] + S.ca[1] * p1[i];
        }
    } else std::fill(y, y + n, 0.0);
    if (u1) {
        spmv1(S.A, b1, r1.data());
        for (int i = 0; i < n; ++i) {
            double tb = S.t * b1[i];
            q0[i] = tb;
            q1[i] = alpha * S.t * r1[i] - (S.beta_shift + 1.0) * tb;
            y[i] += S.cb[0] * q0[i] + S.cb[1] * q1[i];
        }
    }
    double normest = 0.0;
    if (u0) normest = std::max(normest, std::max(maxabs(p0.data(), n), maxabs(p1.data(), n)));
    if (u1) normest = std::max(normest, std::max(maxabs(q0.data(), n), maxabs(q1.data(), n)));
    double normfac = std::getenv("FAMEXP_NORMFAC") ? std::atof(std::getenv("FAMEXP_NORMFAC")) : 2.0;
    double stopfrac = std::getenv("FAMEXP_STOPFRAC") ? std::atof(std::getenv("FAMEXP_STOPFRAC")) : (n >= 10000 ? 10.0 : 0.1);
    double difffrac = std::getenv("FAMEXP_DIFFFRAC") ? std::atof(std::getenv("FAMEXP_DIFFFRAC")) : (n >= 10000 ? (S.lower > 0.0 ? 0.05 : 0.1) : 0.0);
    normest *= normfac;
    double* pp = p0.data(); double* pc = p1.data(); double* pn = p2.data();
    double* qp = q0.data(); double* qc = q1.data(); double* qn = q2.data();
    int kused = 1;
    for (int k = 2; k <= S.kmax; ++k) {
        if (u0 && u1) spmv2(S.A, pc, qc, r0.data(), r1.data());
        else if (u0) spmv1(S.A, pc, r0.data());
        else spmv1(S.A, qc, r1.data());
        double ak = S.ca[k], bk = S.cb[k];
        double ymax = 0.0, pmax = 0.0, qmax = 0.0;
        for (int i = 0; i < n; ++i) {
            if (u0) {
                pn[i] = 2.0 * (alpha * r0[i] - (S.beta_shift + 1.0) * pc[i]) - pp[i];
                y[i] += ak * pn[i];
                pmax = std::max(pmax, std::fabs(pn[i]));
            }
            if (u1) {
                qn[i] = 2.0 * (alpha * r1[i] - (S.beta_shift + 1.0) * qc[i]) - qp[i];
                y[i] += bk * qn[i];
                qmax = std::max(qmax, std::fabs(qn[i]));
            }
            ymax = std::max(ymax, std::fabs(y[i]));
        }
        if (u0) { double* tmp = pp; pp = pc; pc = pn; pn = tmp; }
        if (u1) { double* tmp = qp; qp = qc; qc = qn; qn = tmp; }
        normest = std::max(normest, std::max(pmax, qmax));
        double target = S.atol + S.rtol * ymax;
        kused = k;
        bool tail_ok = S.tail[k] * normest < stopfrac * target;
        bool diff_ok = difffrac <= 0.0;
        if (difffrac > 0.0 && k > 20 && k % 10 == 0) {
            double diff = 0.0;
            for (int i = 0; i < n; ++i) diff = std::max(diff, std::fabs(y[i] - yprev[i]));
            diff_ok = diff < difffrac * target;
            std::copy(y, y + n, yprev.begin());
        }
        if (k > 10 && tail_ok && diff_ok) break;
    }
    if (std::getenv("FAMEXP_DEBUG")) {
        std::fprintf(stderr, "cheb solve kused=%d\n", kused);
    }
}

static void prepare(Session& S) {
    analyze(S);
    if (S.t == 0.0 || S.eta == 0.0 || S.diagonal) {
        S.df.resize(S.n); S.dg.resize(S.n);
        for (int i = 0; i < S.n; ++i) {
            double x = S.t * S.diag[i];
            S.df[i] = std::exp(-x);
            S.dg[i] = S.t * phi1_neg(x);
        }
        S.mode = 1;
        return;
    }
    if (S.n <= 256) { prepare_dense(S); S.mode = 3; return; }
    if (detect_blocks(S)) { prepare_blocks(S); S.mode = 2; return; }
    if (S.eta >= 2.0) {
        maybe_permute(S);
        prepare_cheb(S);
    }
    S.mode = 0;
}

static void solve(Session& S, const double* b0, const double* b1, double* y) {
    if (S.t == 0.0) { std::copy(b0, b0 + S.n, y); return; }
    if (S.mode == 1) {
        for (int i = 0; i < S.n; ++i) {
            y[i] = S.df[i] * b0[i] + S.dg[i] * b1[i];
        }
    } else if (S.mode == 2) {
        solve_blocks(S, b0, b1, y);
    } else if (S.mode == 3) {
        solve_dense(S, b0, b1, y);
    } else {
        if (!S.perm.empty()) {
            S.work_b0.resize(S.n); S.work_b1.resize(S.n); S.work_y.resize(S.n);
            for (int i = 0; i < S.n; ++i) {
                S.work_b0[i] = b0[S.perm[i]];
                S.work_b1[i] = b1[S.perm[i]];
            }
            solve_cheb(S, S.work_b0.data(), S.work_b1.data(), S.work_y.data());
            for (int i = 0; i < S.n; ++i) y[S.perm[i]] = S.work_y[i];
        } else {
            solve_cheb(S, b0, b1, y);
        }
    }
}

int main() {
    if (!write_all("READY\n", 6)) return 1;
    char magic[8];
    if (!read_exact(magic, 8)) return 1;
    uint32_t len = 0;
    if (!read_exact(&len, 4)) return 1;
    std::string js(len, '\0');
    if (len && !read_exact(js.data(), len)) return 1;
    Session S;
    S.n = static_cast<int>(json_int(js, "n"));
    S.requests = static_cast<int>(json_int(js, "requests"));
    S.t = json_number(js, "t");
    S.rtol = json_number(js, "rtol");
    S.atol = json_number(js, "atol");
    long long nnz = json_int(js, "nnz");
    S.A.n = S.n;
    S.A.ptr.resize(static_cast<size_t>(S.n) + 1);
    S.A.idx.resize(static_cast<size_t>(nnz));
    S.A.val.resize(static_cast<size_t>(nnz));
    if (!read_exact(S.A.ptr.data(), sizeof(uint64_t) * (static_cast<size_t>(S.n) + 1))) return 1;
    if (nnz && !read_exact(S.A.idx.data(), sizeof(uint32_t) * static_cast<size_t>(nnz))) return 1;
    if (nnz && !read_exact(S.A.val.data(), sizeof(double) * static_cast<size_t>(nnz))) return 1;
    prepare(S);
    std::vector<double> b0(S.n), b1(S.n), y(S.n);
    for (int r = 0; r < S.requests; ++r) {
        if (!read_exact(b0.data(), sizeof(double) * S.n)) return 1;
        if (!read_exact(b1.data(), sizeof(double) * S.n)) return 1;
        solve(S, b0.data(), b1.data(), y.data());
        if (!write_all(y.data(), sizeof(double) * S.n)) return 1;
    }
    return 0;
}
