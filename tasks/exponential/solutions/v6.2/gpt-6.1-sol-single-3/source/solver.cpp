#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <immintrin.h>
#include <Eigen/Eigenvalues>

using std::vector;

static inline double sparse_dot(int start, int end, const int* col,
                                const double* val, const double* x) {
    double a = 0, b = 0, c = 0, d = 0;
    int k = start;
    for (; k + 3 < end; k += 4) {
        a += val[k] * x[col[k]];
        b += val[k + 1] * x[col[k + 1]];
        c += val[k + 2] * x[col[k + 2]];
        d += val[k + 3] * x[col[k + 3]];
    }
    double s = (a + b) + (c + d);
    for (; k < end; ++k) s += val[k] * x[col[k]];
    return s;
}

static void read_exact(void* p, size_t n) {
    auto* q = static_cast<unsigned char*>(p);
    while (n) {
        size_t k = std::fread(q, 1, n, stdin);
        if (!k) throw std::runtime_error("incomplete input");
        q += k;
        n -= k;
    }
}

static void write_exact(const void* p, size_t n) {
    const auto* q = static_cast<const unsigned char*>(p);
    while (n) {
        size_t k = std::fwrite(q, 1, n, stdout);
        if (!k) throw std::runtime_error("output failure");
        q += k;
        n -= k;
    }
    std::fflush(stdout);
}

struct Header {
    int n = 0, nnz = 0, requests = 0;
    double t = 0, rtol = 0, atol = 0;
};

// The wire header is a flat JSON object with string and numeric values.
static Header parse_header(const std::string& text) {
    Header h;
    size_t pos = 0;
    auto ws = [&]() {
        while (pos < text.size() && static_cast<unsigned char>(text[pos]) <= 32) ++pos;
    };
    auto quoted = [&]() {
        if (text.at(pos++) != '"') throw std::runtime_error("invalid JSON string");
        std::string s;
        while (pos < text.size()) {
            char c = text[pos++];
            if (c == '"') return s;
            if (c == '\\') {
                c = text.at(pos++);
                if (c == 'n') c = '\n';
                else if (c == 't') c = '\t';
                else if (c == 'r') c = '\r';
            }
            s.push_back(c);
        }
        throw std::runtime_error("unterminated JSON string");
    };
    ws();
    if (text.at(pos++) != '{') throw std::runtime_error("invalid JSON object");
    while (true) {
        ws();
        if (text.at(pos) == '}') break;
        std::string key = quoted();
        ws();
        if (text.at(pos++) != ':') throw std::runtime_error("invalid JSON field");
        ws();
        if (text.at(pos) == '"') {
            quoted();
        } else {
            char* end = nullptr;
            double val = std::strtod(text.c_str() + pos, &end);
            if (end == text.c_str() + pos) throw std::runtime_error("invalid JSON number");
            pos = static_cast<size_t>(end - text.c_str());
            if (key == "n") h.n = static_cast<int>(val);
            else if (key == "nnz") h.nnz = static_cast<int>(val);
            else if (key == "requests") h.requests = static_cast<int>(val);
            else if (key == "t") h.t = val;
            else if (key == "rtol") h.rtol = val;
            else if (key == "atol") h.atol = val;
        }
        ws();
        if (text.at(pos) == '}') break;
        if (text.at(pos++) != ',') throw std::runtime_error("invalid JSON separator");
    }
    return h;
}

static std::pair<long double, long double> scalar_functions(long double lambda, long double t) {
    long double z = t * lambda;
    long double e = std::exp(-z);
    long double g = z == 0 ? t : -std::expm1(-z) / lambda;
    return {e, g};
}

struct DSU {
    vector<int> p, sz;
    explicit DSU(int n) : p(n), sz(n, 1) { std::iota(p.begin(), p.end(), 0); }
    int find(int x) {
        while (p[x] != x) {
            p[x] = p[p[x]];
            x = p[x];
        }
        return x;
    }
    void join(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (sz[a] < sz[b]) std::swap(a, b);
        p[b] = a;
        sz[a] += sz[b];
    }
};

struct Solver {
    Header h;
    vector<int> ptr, col;
    vector<double> val, diag;
    double upper = 0, lower = 0, shift = 0, width = 0;
    bool stationary = false;
    bool diagonal = true, direct = false, pair_structure = true, direct_pairs = false;
    bool high_direct = false;
    vector<int> mate;
    vector<double> pair_e, pair_g;
    vector<long double> efl, gfl, pair_el, pair_gl;
    vector<int> block_ptr, block_col;
    vector<double> ef, gf;
    vector<double> ec, gc;
    vector<long double> ecl, gcl, hcl, diag_ext;
    vector<long double> etail, gtail;
    vector<double> v, w, delta;
    bool stable_edges = false;
    bool sell = false;
    vector<int> permutation, sell_row, sell_ptr, sell_len, sell_col;
    vector<double> sell_val, input0, input1, output;
    vector<double> local_e, local_g;

    explicit Solver(Header header) : h(header) {
        vector<uint64_t> p64(h.n + 1);
        vector<uint32_t> c32(h.nnz);
        read_exact(p64.data(), p64.size() * sizeof(uint64_t));
        read_exact(c32.data(), c32.size() * sizeof(uint32_t));
        val.resize(h.nnz);
        read_exact(val.data(), val.size() * sizeof(double));
        ptr.assign(p64.begin(), p64.end());
        col.assign(c32.begin(), c32.end());
        diag.assign(h.n, 0);
        mate.assign(h.n, -1);
        long double minimum = std::numeric_limits<long double>::infinity();
        for (int i = 0; i < h.n; ++i) {
            long double sum = 0;
            for (int k = ptr[i]; k < ptr[i + 1]; ++k) {
                sum += std::abs(val[k]);
                if (col[k] == i) diag[i] = val[k];
                else if (val[k] != 0) {
                    diagonal = false;
                    if (mate[i] >= 0) pair_structure = false;
                    mate[i] = col[k];
                }
            }
            upper = std::max(upper, static_cast<double>(sum));
            minimum = std::min(minimum, 2 * static_cast<long double>(diag[i]) - sum);
        }
        upper *= 1 + 2e-12;
        lower = std::max(0.0, static_cast<double>(minimum) - 2e-12 * upper);
    }

    void prepare_pairs() {
        if (high_direct) {
            efl.resize(h.n);
            gfl.resize(h.n);
            pair_el.assign(h.n, 0);
            pair_gl.assign(h.n, 0);
        } else {
            ef.resize(h.n);
            gf.resize(h.n);
            pair_e.assign(h.n, 0);
            pair_g.assign(h.n, 0);
        }
        auto fun = [&](long double lambda) {
            if (high_direct) return scalar_functions(lambda, h.t);
            double z = static_cast<double>(lambda * h.t);
            long double e = std::exp(-z);
            long double g = z == 0 ? h.t : -std::expm1(-z) / lambda;
            return std::make_pair(e, g);
        };
        for (int i = 0; i < h.n; ++i) {
            int j = mate[i];
            if (j < 0) {
                auto f = fun(diag[i]);
                if (high_direct) { efl[i] = f.first; gfl[i] = f.second; }
                else { ef[i] = static_cast<double>(f.first); gf[i] = static_cast<double>(f.second); }
                mate[i] = i;
                continue;
            }
            if (j < i) continue;
            long double aa = diag[i], cc = diag[j], bb = 0;
            for (int k = ptr[i]; k < ptr[i + 1]; ++k)
                if (col[k] == j) bb = val[k];
            long double mid = (aa + cc) / 2;
            long double gap = std::hypot((aa - cc) / 2, bb);
            long double hi = mid + gap;
            long double lo = hi == 0 ? 0 : (aa * cc - bb * bb) / hi;
            auto fl = fun(lo), fh = fun(hi);
            long double de = gap == 0 || hi == lo ? 0 : (fh.first - fl.first) / (hi - lo);
            long double dg = gap == 0 || hi == lo ? 0 : (fh.second - fl.second) / (hi - lo);
            long double ei = fl.first + (aa - lo) * de;
            long double ej = fl.first + (cc - lo) * de;
            long double gi = fl.second + (aa - lo) * dg;
            long double gj = fl.second + (cc - lo) * dg;
            if (high_direct) {
                efl[i] = ei; efl[j] = ej;
                gfl[i] = gi; gfl[j] = gj;
                pair_el[i] = pair_el[j] = bb * de;
                pair_gl[i] = pair_gl[j] = bb * dg;
            } else {
                ef[i] = static_cast<double>(ei); ef[j] = static_cast<double>(ej);
                gf[i] = static_cast<double>(gi); gf[j] = static_cast<double>(gj);
                pair_e[i] = pair_e[j] = static_cast<double>(bb * de);
                pair_g[i] = pair_g[j] = static_cast<double>(bb * dg);
            }
        }
    }

    template<typename Real>
    bool prepare_blocks() {
        using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
        using Vec = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
        if (h.n > 64 && h.nnz > 3 * h.n) return false;
        DSU sets(h.n);
        for (int i = 0; i < h.n; ++i)
            for (int k = ptr[i]; k < ptr[i + 1]; ++k)
                if (val[k] != 0) sets.join(i, col[k]);
        vector<vector<int>> groups(h.n);
        for (int i = 0; i < h.n; ++i) groups[sets.find(i)].push_back(i);
        int largest = 0;
        for (const auto& g : groups) largest = std::max(largest, static_cast<int>(g.size()));
        if (largest > 16 && h.n > 64) return false;
        block_ptr.assign(h.n + 1, 0);
        for (const auto& g : groups)
            for (int i : g) block_ptr[i + 1] = static_cast<int>(g.size());
        for (int i = 0; i < h.n; ++i) block_ptr[i + 1] += block_ptr[i];
        block_col.resize(block_ptr.back());
        if constexpr (std::is_same_v<Real, long double>) {
            efl.resize(block_ptr.back());
            gfl.resize(block_ptr.back());
        } else {
            ef.resize(block_ptr.back());
            gf.resize(block_ptr.back());
        }
        vector<int> local(h.n, -1);
        for (const auto& g : groups) {
            int s = static_cast<int>(g.size());
            if (!s) continue;
            for (int j = 0; j < s; ++j) local[g[j]] = j;
            Matrix a = Matrix::Zero(s, s);
            for (int j = 0; j < s; ++j)
                for (int k = ptr[g[j]]; k < ptr[g[j] + 1]; ++k)
                    if (local[col[k]] >= 0) a(j, local[col[k]]) = val[k];
            Matrix e(s, s), f(s, s);
            if (s == 1) {
                auto fun = scalar_functions(a(0, 0), h.t);
                e(0, 0) = static_cast<Real>(fun.first);
                f(0, 0) = static_cast<Real>(fun.second);
            } else if (s == 2) {
                long double aa = a(0, 0), bb = a(0, 1), cc = a(1, 1);
                long double mid = (aa + cc) / 2;
                long double gap = std::hypot((aa - cc) / 2, bb);
                long double hi = mid + gap;
                long double lo = hi == 0 ? 0 : (aa * cc - bb * bb) / hi;
                auto fl = scalar_functions(lo, h.t);
                auto fh = scalar_functions(hi, h.t);
                if (gap == 0 || hi == lo) {
                    e = Matrix::Identity(2, 2) * static_cast<Real>(fl.first);
                    f = Matrix::Identity(2, 2) * static_cast<Real>(fl.second);
                } else {
                    long double de = (fh.first - fl.first) / (hi - lo);
                    long double df = (fh.second - fl.second) / (hi - lo);
                    e(0, 0) = static_cast<Real>(fl.first + (aa - lo) * de);
                    e(1, 1) = static_cast<Real>(fl.first + (cc - lo) * de);
                    e(0, 1) = e(1, 0) = static_cast<Real>(bb * de);
                    f(0, 0) = static_cast<Real>(fl.second + (aa - lo) * df);
                    f(1, 1) = static_cast<Real>(fl.second + (cc - lo) * df);
                    f(0, 1) = f(1, 0) = static_cast<Real>(bb * df);
                }
            } else {
                Eigen::SelfAdjointEigenSolver<Matrix> eig(a);
                if (eig.info() != Eigen::Success) return false;
                Vec ev(s), fv(s);
                for (int j = 0; j < s; ++j) {
                    auto fun = scalar_functions(eig.eigenvalues()[j], h.t);
                    ev[j] = static_cast<Real>(fun.first);
                    fv[j] = static_cast<Real>(fun.second);
                }
                e.noalias() = eig.eigenvectors() * ev.asDiagonal() * eig.eigenvectors().transpose();
                f.noalias() = eig.eigenvectors() * fv.asDiagonal() * eig.eigenvectors().transpose();
            }
            for (int j = 0; j < s; ++j) {
                int off = block_ptr[g[j]];
                for (int k = 0; k < s; ++k) {
                    block_col[off + k] = g[k];
                    if constexpr (std::is_same_v<Real, long double>) {
                        efl[off + k] = e(j, k);
                        gfl[off + k] = f(j, k);
                    } else {
                        ef[off + k] = e(j, k);
                        gf[off + k] = f(j, k);
                    }
                }
            }
            for (int i : g) local[i] = -1;
        }
        return true;
    }

    void tighten_bound() {
        if (h.t * upper < 100 || h.n < 100) return;
        vector<double> x(h.n, 1), z(h.n);
        // Collatz-Wielandt bounds for |L| remain upper bounds for lambda_max(L).
        for (int step = 0; step < 5; ++step) {
            double bound = 0, scale = 0;
            for (int i = 0; i < h.n; ++i) {
                long double s = 0;
                for (int k = ptr[i]; k < ptr[i + 1]; ++k)
                    s += static_cast<long double>(std::abs(val[k])) * x[col[k]];
                z[i] = static_cast<double>(s);
                if (x[i] > 0) bound = std::max(bound, static_cast<double>(s / x[i]));
                scale = std::max(scale, z[i]);
            }
            upper = std::min(upper, bound * (1 + 2e-12));
            if (scale == 0) break;
            for (int i = 0; i < h.n; ++i) x[i] = std::max(z[i] / scale, 1e-100);
        }
        upper *= 1 + 2e-14;
    }

    void prepare_coefficients() {
        width = upper;
        // With the public bounds, this discarded transient is below 6e-22 in norm.
        if (h.t * lower >= 60) {
            stationary = true;
            shift = lower;
            width = upper - lower;
            long double su = std::sqrt(static_cast<long double>(upper));
            long double sl = std::sqrt(static_cast<long double>(lower));
            long double q = (su - sl) / (su + sl);
            long double c0 = 1 / (su * sl);
            int m = q == 0 ? 0 : static_cast<int>(std::ceil(
                (std::log(1e-28L) - std::log(2 * c0 / (1 - q))) / std::log(q))) + 2;
            ec.assign(m + 1, 0);
            gc.resize(m + 1);
            ecl.assign(m + 1, 0);
            gcl.resize(m + 1);
            etail.assign(m + 1, 0);
            gtail.resize(m + 1);
            long double power = 1;
            for (int k = 0; k <= m; ++k) {
                gcl[k] = (k == 0 ? 1 : 2) * c0 * power;
                gc[k] = static_cast<double>(gcl[k]);
                power *= q;
                gtail[k] = 2 * c0 * power / (1 - q);
            }
            v.resize(h.n);
            w.resize(h.n);
            delta.resize(h.n);
            return;
        }
        long double a = static_cast<long double>(h.t) * upper / 2;
        int m = static_cast<int>(std::ceil(std::sqrt(120 * a))) + 60;
        vector<long double> f(m + 2, 0), g(m + 2, 0);
        if (a < 0.25L) {
            long double base = std::exp(-a), z = a * a / 4;
            for (int k = 0; k <= m; ++k) {
                long double term = 1, sum = 1;
                for (int j = 1; j < 100; ++j) {
                    term *= z / (static_cast<long double>(j) * (j + k));
                    sum += term;
                    if (term < 1e-25L * sum) break;
                }
                f[k] = base * sum * (k == 0 ? 1 : 2);
                base *= a / (2 * (k + 1));
            }
        } else {
            f[m] = 1;
            for (int k = m; k >= 1; --k) f[k - 1] = f[k + 1] + (2 * k / a) * f[k];
            long double sum = f[0];
            for (int k = 1; k <= m; ++k) sum += 2 * f[k];
            f[0] /= sum;
            for (int k = 1; k <= m; ++k) f[k] *= 2 / sum;
        }
        long double tail = 0, weighted = 0;
        // Integrating the Bessel series reduces to positive weighted tail sums.
        for (int k = m; k >= 1; --k) {
            g[k] = 2 * weighted / a;
            tail += f[k];
            weighted += tail;
        }
        g[0] = weighted / a;
        ec.resize(m + 1);
        gc.resize(m + 1);
        ecl.resize(m + 1);
        gcl.resize(m + 1);
        etail.assign(m + 1, 0);
        gtail.assign(m + 1, 0);
        long double se = 0, sg = 0;
        for (int k = m; k >= 0; --k) {
            ec[k] = static_cast<double>(f[k]);
            gc[k] = static_cast<double>(g[k] * h.t);
            ecl[k] = f[k];
            gcl[k] = g[k] * h.t;
            etail[k] = se;
            gtail[k] = sg;
            se += f[k];
            sg += g[k] * h.t;
        }
        hcl.assign(m + 1, 0);
        tail = 0;
        weighted = 0;
        for (int k = m; k >= 1; --k) {
            hcl[k] = 4 * weighted / upper;
            tail += gcl[k];
            weighted += tail;
        }
        hcl[0] = 2 * weighted / upper;
        v.resize(h.n);
        w.resize(h.n);
        delta.resize(h.n);
    }

    void prepare_kernel() {
        if (h.t * upper > 500) {
            local_e.resize(h.n);
            local_g.resize(h.n);
            for (int i = 0; i < h.n; ++i) {
                double z = h.t * diag[i];
                local_e[i] = std::exp(-z);
                local_g[i] = z == 0 ? h.t : -h.t * std::expm1(-z) / z;
            }
        }
        stable_edges = true;
        vector<long double> potential(h.n);
        for (int i = 0; i < h.n; ++i) {
            long double sum = 0, correction = 0;
            for (int k = ptr[i]; k < ptr[i + 1]; ++k) {
                if (col[k] == i) continue;
                long double term = static_cast<long double>(std::abs(val[k])) - correction;
                long double next = sum + term;
                correction = (next - sum) - term;
                sum = next;
            }
            potential[i] = static_cast<long double>(diag[i]) - sum;
            if (potential[i] < -1e-13 * std::max(diag[i], static_cast<double>(sum)))
                stable_edges = false;
        }
        if (std::getenv("FAMEXP_NO_EDGES")) stable_edges = false;
        vector<int> p(h.n + 1), c;
        vector<double> a;
        c.reserve(h.nnz);
        a.reserve(h.nnz);
        for (int i = 0; i < h.n; ++i) {
            p[i] = static_cast<int>(a.size());
            for (int k = ptr[i]; k < ptr[i + 1]; ++k) {
                if (col[k] == i || val[k] == 0) continue;
                c.push_back(col[k]);
                a.push_back(val[k]);
            }
        }
        p[h.n] = static_cast<int>(a.size());
        ptr.swap(p);
        col.swap(c);
        val.swap(a);
        diag_ext.resize(h.n);
        // Retain the received diagonal exactly to extended-precision summation;
        // do not replace a nearly Laplacian matrix by an ideal row-sum-zero one.
        for (int i = 0; i < h.n; ++i) {
            diag_ext[i] = (stable_edges ? potential[i] : static_cast<long double>(diag[i])) - shift;
            diag[i] = static_cast<double>(diag_ext[i]);
        }
        bool simd = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
        if (!simd || h.n < 128 || std::getenv("FAMEXP_CSR") ||
            (h.t * upper < 20 && !(h.requests == 16 && h.t * upper > 0.5))) return;
        permutation.resize(h.n);
        std::iota(permutation.begin(), permutation.end(), 0);
        if (h.t * upper > 500 && !std::getenv("FAMEXP_NO_RCM")) {
            int64_t old_span = 0;
            for (int i = 0; i < h.n; ++i)
                for (int j = ptr[i]; j < ptr[i + 1]; ++j)
                    old_span += std::abs(i - col[j]);
            if (old_span > static_cast<int64_t>(col.size()) * h.n / 32) {
                vector<int> seeds(h.n);
                std::iota(seeds.begin(), seeds.end(), 0);
                auto degree_less = [&](int i, int j) {
                    return ptr[i + 1] - ptr[i] < ptr[j + 1] - ptr[j];
                };
                std::stable_sort(seeds.begin(), seeds.end(), degree_less);
                vector<unsigned char> seen(h.n, 0);
                vector<int> candidate, neighbors;
                candidate.reserve(h.n);
                for (int root : seeds) {
                    if (seen[root]) continue;
                    size_t cursor = candidate.size();
                    candidate.push_back(root);
                    seen[root] = 1;
                    while (cursor < candidate.size()) {
                        int i = candidate[cursor++];
                        neighbors.clear();
                        for (int j = ptr[i]; j < ptr[i + 1]; ++j) {
                            int c = col[j];
                            if (seen[c]) continue;
                            seen[c] = 1;
                            neighbors.push_back(c);
                        }
                        std::sort(neighbors.begin(), neighbors.end(), degree_less);
                        candidate.insert(candidate.end(), neighbors.begin(), neighbors.end());
                    }
                }
                std::reverse(candidate.begin(), candidate.end());
                vector<int> inverse(h.n);
                for (int i = 0; i < h.n; ++i) inverse[candidate[i]] = i;
                int64_t new_span = 0;
                for (int i = 0; i < h.n; ++i)
                    for (int j = ptr[i]; j < ptr[i + 1]; ++j)
                        new_span += std::abs(inverse[i] - inverse[col[j]]);
                if (new_span < old_span * 0.8) permutation.swap(candidate);
                if (std::getenv("FAMEXP_DEBUG"))
                    std::fprintf(stderr, "index_span original %.6g reordered %.6g accepted %d\n",
                                 static_cast<double>(old_span) / std::max<size_t>(1, col.size()),
                                 static_cast<double>(new_span) / std::max<size_t>(1, col.size()),
                                 new_span < old_span * 0.8);
            }
        }
        for (int first = 0; first < h.n; first += 256) {
            int end = std::min(first + 256, h.n);
            std::stable_sort(permutation.begin() + first, permutation.begin() + end,
                             [&](int i, int j) {
                                 return ptr[i + 1] - ptr[i] < ptr[j + 1] - ptr[j];
                             });
        }
        vector<int> inverse(h.n);
        for (int i = 0; i < h.n; ++i) inverse[permutation[i]] = i;
        vector<int> new_ptr(h.n + 1), new_col;
        vector<double> new_val, new_diag(h.n);
        vector<long double> new_diag_ext(h.n);
        new_col.reserve(col.size());
        new_val.reserve(val.size());
        for (int i = 0; i < h.n; ++i) {
            int row = permutation[i];
            new_ptr[i] = static_cast<int>(new_col.size());
            new_diag[i] = diag[row];
            new_diag_ext[i] = diag_ext[row];
            for (int j = ptr[row]; j < ptr[row + 1]; ++j) {
                new_col.push_back(inverse[col[j]]);
                new_val.push_back(val[j]);
            }
        }
        new_ptr[h.n] = static_cast<int>(new_col.size());
        ptr.swap(new_ptr);
        col.swap(new_col);
        val.swap(new_val);
        diag.swap(new_diag);
        diag_ext.swap(new_diag_ext);
        if (!local_e.empty()) {
            vector<double> e(h.n), g(h.n);
            for (int i = 0; i < h.n; ++i) {
                e[i] = local_e[permutation[i]];
                g[i] = local_g[permutation[i]];
            }
            local_e.swap(e);
            local_g.swap(g);
        }
        for (int first = 0; first < h.n; first += 4) {
            int end = std::min(first + 4, h.n), len = 0;
            for (int i = first; i < end; ++i) len = std::max(len, ptr[i + 1] - ptr[i]);
            if (end - first < 4 || len > 128) {
                sell_row.push_back(first);
                sell_ptr.push_back(0);
                sell_len.push_back(-1);
                continue;
            }
            sell_row.push_back(first);
            sell_ptr.push_back(static_cast<int>(sell_col.size()));
            sell_len.push_back(len);
            for (int k = 0; k < len; ++k) {
                for (int i = first; i < first + 4; ++i) {
                    int j = ptr[i] + k;
                    bool present = j < ptr[i + 1];
                    sell_col.push_back(present ? col[j] : i);
                    sell_val.push_back(present ? val[j] : 0);
                }
            }
        }
        sell = true;
        input0.resize(h.n);
        input1.resize(h.n);
        output.resize(h.n);
    }

    void prepare() {
        if (h.t == 0 || upper == 0) return;
        if (diagonal) {
            ef.resize(h.n);
            gf.resize(h.n);
            for (int i = 0; i < h.n; ++i) {
                auto f = scalar_functions(diag[i], h.t);
                ef[i] = static_cast<double>(f.first);
                gf[i] = static_cast<double>(f.second);
            }
            return;
        }
        high_direct = h.rtol <= 1e-7 && h.t * upper > 100;
        if (pair_structure) {
            direct_pairs = true;
            prepare_pairs();
            return;
        }
        direct = high_direct ? prepare_blocks<long double>() : prepare_blocks<double>();
        if (direct) return;
        tighten_bound();
        prepare_coefficients();
        prepare_kernel();
    }

    int degree(long double norm0, long double norm1, long double tolerance) const {
        for (int k = 0; k < static_cast<int>(ec.size()); ++k)
            if (etail[k] * norm0 + gtail[k] * norm1 <= tolerance) return k;
        return static_cast<int>(ec.size()) - 1;
    }

    template<bool Final, bool Edges>
    __attribute__((target("avx2,fma")))
    void sell_step(double c, double d, double factor, const double* b0, const double* b1,
                   const double* x, const double* din, double* dout, double* out) {
        __m256d vc = _mm256_set1_pd(c), vd = _mm256_set1_pd(d);
        __m256d vf = _mm256_set1_pd(factor);
        __m256d sign_mask = _mm256_set1_pd(-0.0);
        for (int g = 0; g < static_cast<int>(sell_row.size()); ++g) {
            int row = sell_row[g], start = sell_ptr[g], end = start + 4 * sell_len[g];
            if (sell_len[g] < 0) {
                for (int i = row; i < std::min(row + 4, h.n); ++i) {
                    double s = diag[i] * x[i];
                    if constexpr (Edges) {
                        for (int j = ptr[i]; j < ptr[i + 1]; ++j)
                            s += val[j] * (x[col[j]] + std::copysign(1.0, val[j]) * x[i]);
                    } else {
                        s += sparse_dot(ptr[i], ptr[i + 1], col.data(), val.data(), x);
                    }
                    double diff = din[i] - factor * s + c * b0[i] + d * b1[i];
                    if constexpr (Final) out[i] = diff;
                    else {
                        dout[i] = diff;
                        out[i] = x[i] + diff;
                    }
                }
                continue;
            }
            __m256d xi = _mm256_loadu_pd(x + row);
            __m256d sum = _mm256_mul_pd(_mm256_loadu_pd(diag.data() + row), xi);
            __m256d sum1 = _mm256_setzero_pd();
            int j = start;
            for (; j + 7 < end; j += 8) {
                __m256d a = _mm256_loadu_pd(sell_val.data() + j);
                __m128i ind = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sell_col.data() + j));
                __m256d xj = _mm256_i32gather_pd(x, ind, 8);
                __m256d a1 = _mm256_loadu_pd(sell_val.data() + j + 4);
                __m128i ind1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sell_col.data() + j + 4));
                __m256d xj1 = _mm256_i32gather_pd(x, ind1, 8);
                if constexpr (Edges) {
                    xj = _mm256_add_pd(xj, _mm256_xor_pd(xi, _mm256_and_pd(a, sign_mask)));
                    xj1 = _mm256_add_pd(xj1, _mm256_xor_pd(xi, _mm256_and_pd(a1, sign_mask)));
                }
                sum = _mm256_fmadd_pd(a, xj, sum);
                sum1 = _mm256_fmadd_pd(a1, xj1, sum1);
            }
            sum = _mm256_add_pd(sum, sum1);
            for (; j < end; j += 4) {
                __m256d a = _mm256_loadu_pd(sell_val.data() + j);
                __m128i ind = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sell_col.data() + j));
                __m256d xj = _mm256_i32gather_pd(x, ind, 8);
                if constexpr (Edges)
                    xj = _mm256_add_pd(xj, _mm256_xor_pd(xi, _mm256_and_pd(a, sign_mask)));
                sum = _mm256_fmadd_pd(a, xj, sum);
            }
            __m256d rhs = _mm256_fmadd_pd(vc, _mm256_loadu_pd(b0 + row),
                                        _mm256_mul_pd(vd, _mm256_loadu_pd(b1 + row)));
            __m256d diff = _mm256_add_pd(rhs, _mm256_fnmadd_pd(
                                               vf, sum, _mm256_loadu_pd(din + row)));
            if constexpr (Final) {
                _mm256_storeu_pd(out + row, diff);
            } else {
                _mm256_storeu_pd(dout + row, diff);
                _mm256_storeu_pd(out + row, _mm256_add_pd(xi, diff));
            }
        }
    }

    void evaluate(int m, const vector<double>& b0, const vector<double>& b1, vector<double>& y) {
        // Reinsch's difference form avoids cancellation at the zero eigenvalue.
        if (m == 0) {
            for (int i = 0; i < h.n; ++i) y[i] = ec[0] * b0[i] + gc[0] * b1[i];
            return;
        }
        std::fill(v.begin(), v.end(), 0);
        std::fill(delta.begin(), delta.end(), 0);
        double factor = 4 / width;
        if (sell) {
            for (int k = m; k >= 1; --k) {
                if (stable_edges)
                    sell_step<false, true>(ec[k], gc[k], factor, b0.data(), b1.data(),
                                          v.data(), delta.data(), delta.data(), w.data());
                else
                    sell_step<false, false>(ec[k], gc[k], factor, b0.data(), b1.data(),
                                           v.data(), delta.data(), delta.data(), w.data());
                v.swap(w);
            }
            factor = 2 / width;
            if (stable_edges)
                sell_step<true, true>(ec[0], gc[0], factor, b0.data(), b1.data(),
                                     v.data(), delta.data(), delta.data(), y.data());
            else
                sell_step<true, false>(ec[0], gc[0], factor, b0.data(), b1.data(),
                                      v.data(), delta.data(), delta.data(), y.data());
            return;
        }
        for (int k = m; k >= 1; --k) {
            double c = ec[k], d = gc[k];
            for (int i = 0; i < h.n; ++i) {
                double s = diag[i] * v[i];
                if (stable_edges) {
                    for (int j = ptr[i]; j < ptr[i + 1]; ++j)
                        s += val[j] * (v[col[j]] + std::copysign(1.0, val[j]) * v[i]);
                } else {
                    s += sparse_dot(ptr[i], ptr[i + 1], col.data(), val.data(), v.data());
                }
                double diff = (delta[i] - factor * s) + (c * b0[i] + d * b1[i]);
                w[i] = v[i] + diff;
                delta[i] = diff;
            }
            v.swap(w);
        }
        factor = 2 / width;
        for (int i = 0; i < h.n; ++i) {
            double s = diag[i] * v[i];
            if (stable_edges) {
                for (int j = ptr[i]; j < ptr[i + 1]; ++j)
                    s += val[j] * (v[col[j]] + std::copysign(1.0, val[j]) * v[i]);
            } else {
                s += sparse_dot(ptr[i], ptr[i + 1], col.data(), val.data(), v.data());
            }
            y[i] = (delta[i] - factor * s) + (ec[0] * b0[i] + gc[0] * b1[i]);
        }
    }

    void evaluate_long(int m, const vector<double>& b0, const vector<double>& b1, vector<double>& y) {
        if (m == 0) {
            for (int i = 0; i < h.n; ++i)
                y[i] = static_cast<double>(ecl[0] * b0[i] + gcl[0] * b1[i]);
            return;
        }
        vector<long double> x(h.n, 0), z(h.n), d(h.n, 0);
        long double factor = 4 / static_cast<long double>(width);
        for (int k = m; k >= 1; --k) {
            for (int i = 0; i < h.n; ++i) {
                long double s = diag_ext[i] * x[i];
                if (ptr[i + 1] - ptr[i] > 32) {
                    long double correction = 0;
                    for (int j = ptr[i]; j < ptr[i + 1]; ++j) {
                        long double xx = x[col[j]];
                        if (stable_edges) xx += std::copysign(1.0, val[j]) * x[i];
                        long double term = static_cast<long double>(val[j]) * xx - correction;
                        long double next = s + term;
                        correction = (next - s) - term;
                        s = next;
                    }
                } else if (stable_edges) {
                    for (int j = ptr[i]; j < ptr[i + 1]; ++j)
                        s += static_cast<long double>(val[j]) *
                             (x[col[j]] + std::copysign(1.0, val[j]) * x[i]);
                } else {
                    for (int j = ptr[i]; j < ptr[i + 1]; ++j)
                        s += static_cast<long double>(val[j]) * x[col[j]];
                }
                d[i] = d[i] - factor * s + ecl[k] * b0[i] + gcl[k] * b1[i];
                z[i] = x[i] + d[i];
            }
            x.swap(z);
        }
        factor = 2 / static_cast<long double>(width);
        for (int i = 0; i < h.n; ++i) {
            long double s = diag_ext[i] * x[i];
            if (ptr[i + 1] - ptr[i] > 32) {
                long double correction = 0;
                for (int j = ptr[i]; j < ptr[i + 1]; ++j) {
                    long double xx = x[col[j]];
                    if (stable_edges) xx += std::copysign(1.0, val[j]) * x[i];
                    long double term = static_cast<long double>(val[j]) * xx - correction;
                    long double next = s + term;
                    correction = (next - s) - term;
                    s = next;
                }
            } else if (stable_edges) {
                for (int j = ptr[i]; j < ptr[i + 1]; ++j)
                    s += static_cast<long double>(val[j]) *
                         (x[col[j]] + std::copysign(1.0, val[j]) * x[i]);
            } else {
                for (int j = ptr[i]; j < ptr[i + 1]; ++j)
                    s += static_cast<long double>(val[j]) * x[col[j]];
            }
            y[i] = static_cast<double>(d[i] - factor * s + ecl[0] * b0[i] + gcl[0] * b1[i]);
        }
    }

    long double krylov_log_bound(const vector<double>& alpha, const vector<double>& beta,
                                long double norm, bool phi) const {
        int m = static_cast<int>(alpha.size());
        if (beta.back() == 0) return -std::numeric_limits<long double>::infinity();
        long double leading = std::log(static_cast<long double>(beta.back()) * h.t * norm);
        if (phi) leading += std::log(static_cast<long double>(h.t)) - 1;
        long double best = std::numeric_limits<long double>::infinity();
        long double best_q = 1;
        // A positive resolvent bounds the integrated Jacobi-matrix residual.
        auto bound = [&](long double q) {
            long double pivot = alpha[0] * static_cast<long double>(h.t) + q;
            if (pivot <= 0) return std::numeric_limits<long double>::infinity();
            long double log_r = -std::log(pivot);
            for (int j = 1; j < m; ++j) {
                long double b = beta[j - 1] * static_cast<long double>(h.t);
                pivot = alpha[j] * static_cast<long double>(h.t) + q - b * b / pivot;
                if (pivot <= 0) return std::numeric_limits<long double>::infinity();
                log_r += std::log(b) - std::log(pivot);
            }
            return leading + q + log_r - (phi ? std::log(q) : 0);
        };
        for (long double q = 0.125L; q <= 4096; q *= 2) {
            long double e = bound(q);
            if (e < best) {
                best = e;
                best_q = q;
            }
        }
        long double left = best_q / 2, right = best_q * 2;
        for (int iter = 0; iter < 14; ++iter) {
            long double p = (2 * left + right) / 3;
            long double q = (left + 2 * right) / 3;
            long double ep = bound(p), eq = bound(q);
            best = std::min(best, std::min(ep, eq));
            if (ep < eq) right = q;
            else left = p;
        }
        return best;
    }

    void projected_function(const vector<double>& alpha, const vector<double>& beta,
                            int which, vector<long double>& result) const {
        int m = static_cast<int>(alpha.size());
        vector<long double> x(m, 0), z(m), d(m, 0);
        const auto& coeff = which == 0 ? ecl : (which == 1 ? gcl : hcl);
        long double factor = 4 / static_cast<long double>(width);
        for (int k = static_cast<int>(coeff.size()) - 1; k >= 1; --k) {
            for (int i = 0; i < m; ++i) {
                long double s = alpha[i] * x[i];
                if (i) s += beta[i - 1] * x[i - 1];
                if (i + 1 < m) s += beta[i] * x[i + 1];
                d[i] -= factor * s;
                if (i == 0) d[i] += coeff[k];
                z[i] = x[i] + d[i];
            }
            x.swap(z);
        }
        result.resize(m);
        factor = 2 / static_cast<long double>(width);
        for (int i = 0; i < m; ++i) {
            long double s = alpha[i] * x[i];
            if (i) s += beta[i - 1] * x[i - 1];
            if (i + 1 < m) s += beta[i] * x[i + 1];
            result[i] = d[i] - factor * s + (i == 0 ? coeff[0] : 0);
        }
    }

    long double krylov_roundoff(const vector<double>& alpha, const vector<double>& beta,
                               const vector<std::unique_ptr<double[]>>& basis,
                               const vector<double>& b, long double norm, bool phi) const {
        int m = static_cast<int>(alpha.size());
        vector<long double> integrated;
        projected_function(alpha, beta, phi ? 2 : 1, integrated);
        long double error = 0;
        const long double eps = std::numeric_limits<long double>::epsilon();
        for (int j = 0; j < m; ++j) {
            long double sq = 0;
            const double* x = basis[j].get();
            for (int i = 0; i < h.n; ++i) {
                long double sum = diag_ext[i] * x[i], correction = 0;
                long double absolute = std::abs(sum);
                for (int k = ptr[i]; k < ptr[i + 1]; ++k) {
                    long double xx = x[col[k]];
                    if (stable_edges) xx += std::copysign(1.0, val[k]) * x[i];
                    long double product = static_cast<long double>(val[k]) * xx;
                    absolute += std::abs(product);
                    long double term = product - correction;
                    long double next = sum + term;
                    correction = (next - sum) - term;
                    sum = next;
                }
                long double a = static_cast<long double>(alpha[j]) * x[i];
                long double p = j ? static_cast<long double>(beta[j - 1]) * basis[j - 1][i] : 0;
                long double q = beta[j] == 0 ? 0 :
                    static_cast<long double>(beta[j]) * basis[j + 1][i];
                long double residual = std::abs(sum - a - p - q) +
                    16 * eps * (absolute + upper * std::abs(x[i]) +
                                std::abs(a) + std::abs(p) + std::abs(q));
                sq += residual * residual;
            }
            long double scale = phi ? static_cast<long double>(h.t) * h.t : h.t;
            long double coord_error = 128 * eps * (1 + h.t * upper) * scale;
            error += std::sqrt(sq) * (std::abs(integrated[j]) + coord_error);
        }
        error *= norm;
        long double initial = 0;
        for (int i = 0; i < h.n; ++i) {
            long double d = b[i] - norm * basis[0][i];
            initial += d * d;
        }
        long double scale = phi ? h.t : 1;
        return error + scale * std::sqrt(initial) +
               16 * eps * (1 + h.t * upper) * norm * scale;
    }

    bool try_krylov_rhs(const vector<double>& b, bool phi, int cap, long double tolerance,
                        vector<double>& out, long double& error) {
        long double norm = 0;
        for (double x : b) norm += static_cast<long double>(x) * x;
        norm = std::sqrt(norm);
        out.assign(h.n, 0);
        error = 0;
        if (norm == 0) return true;
        vector<std::unique_ptr<double[]>> basis;
        basis.reserve(cap + 1);
        basis.emplace_back(new double[h.n]);
        for (int i = 0; i < h.n; ++i) basis[0][i] = static_cast<double>(b[i] / norm);
        vector<double> alpha, beta, work(h.n);
        alpha.reserve(cap);
        beta.reserve(cap);
        for (int step = 0; step < cap; ++step) {
            const double* x = basis[step].get();
            long double dot = 0;
            for (int i = 0; i < h.n; ++i) {
                double sum = diag[i] * x[i];
                if (stable_edges) {
                    for (int j = ptr[i]; j < ptr[i + 1]; ++j)
                        sum += val[j] * (x[col[j]] + std::copysign(1.0, val[j]) * x[i]);
                } else {
                    sum += sparse_dot(ptr[i], ptr[i + 1], col.data(), val.data(), x);
                }
                if (step) sum -= beta[step - 1] * basis[step - 1][i];
                work[i] = sum;
                dot += static_cast<long double>(sum) * x[i];
            }
            alpha.push_back(static_cast<double>(dot));
            long double sq = 0;
            for (int i = 0; i < h.n; ++i) {
                work[i] -= alpha.back() * x[i];
                sq += static_cast<long double>(work[i]) * work[i];
            }
            beta.push_back(static_cast<double>(std::sqrt(sq)));
            int m = step + 1;
            if (beta.back() != 0) {
                basis.emplace_back(new double[h.n]);
                for (int i = 0; i < h.n; ++i) basis.back()[i] = work[i] / beta.back();
            }
            bool checkpoint = m == cap || beta.back() == 0 ||
                              (m <= 64 ? m % 16 == 0 : m % 32 == 0);
            if (checkpoint) {
                long double lb = krylov_log_bound(alpha, beta, norm, phi);
                if (std::getenv("FAMEXP_DEBUG"))
                    std::fprintf(stderr, "krylov phi %d m %d log_bound %.5Lg log_tol %.5Lg\n",
                                 phi, m, lb, std::log(tolerance));
                if (lb <= std::log(tolerance)) {
                    vector<long double> coeff;
                    projected_function(alpha, beta, phi ? 1 : 0, coeff);
                    vector<long double> sum(h.n, 0);
                    for (int j = 0; j < m; ++j) {
                        long double c = norm * coeff[j];
                        const double* q = basis[j].get();
                        for (int i = 0; i < h.n; ++i) sum[i] += c * q[i];
                    }
                    for (int i = 0; i < h.n; ++i) {
                        out[i] = static_cast<double>(sum[i]);
                        if (!std::isfinite(out[i])) return false;
                    }
                    error = std::exp(lb) + krylov_roundoff(alpha, beta, basis, b, norm, phi);
                    if (std::getenv("FAMEXP_DEBUG"))
                        std::fprintf(stderr, "krylov total_error %.8Lg\n", error);
                    return true;
                }
            }
        }
        return false;
    }

    bool integer_eigenvectors(const vector<double>& b0, const vector<double>& b1,
                              double max0, double max1, vector<double>& y) const {
        // Uniform-magnitude vectors permit an exact integer eigenvector check.
        auto sign = [](double x, double magnitude) {
            return magnitude == 0 ? 0 : (std::signbit(x) ? -1 : 1);
        };
        long long lambda0 = 0, lambda1 = 0;
        for (int i = 0; i < h.n; ++i) {
            double d = diag[i];
            if (std::abs(d) > 1e12 || std::trunc(d) != d ||
                diag_ext[i] != static_cast<long double>(d)) return false;
            int s0 = sign(b0[i], max0), s1 = sign(b1[i], max1);
            long long r0 = static_cast<long long>(d) * s0;
            long long r1 = static_cast<long long>(d) * s1;
            for (int k = ptr[i]; k < ptr[i + 1]; ++k) {
                double a = val[k];
                if (std::abs(a) > 1e12 || std::trunc(a) != a) return false;
                int q0 = sign(b0[col[k]], max0), q1 = sign(b1[col[k]], max1);
                if (stable_edges) {
                    q0 += (a < 0 ? -s0 : s0);
                    q1 += (a < 0 ? -s1 : s1);
                }
                r0 += static_cast<long long>(a) * q0;
                r1 += static_cast<long long>(a) * q1;
            }
            if (i == 0) {
                lambda0 = r0 * s0;
                lambda1 = r1 * s1;
                if (lambda0 < 0 || lambda1 < 0) return false;
            }
            if (r0 != lambda0 * s0 || r1 != lambda1 * s1) return false;
        }
        long double e = scalar_functions(lambda0, h.t).first;
        long double g = scalar_functions(lambda1, h.t).second;
        for (int i = 0; i < h.n; ++i)
            y[i] = static_cast<double>(e * b0[i] + g * b1[i]);
        return true;
    }

    void solve_inner(const vector<double>& b0, const vector<double>& b1, vector<double>& y) {
        if (h.t == 0) {
            y = b0;
            return;
        }
        if (upper == 0) {
            for (int i = 0; i < h.n; ++i)
                y[i] = static_cast<double>(static_cast<long double>(b0[i]) +
                                            static_cast<long double>(h.t) * b1[i]);
            return;
        }
        if (diagonal) {
            for (int i = 0; i < h.n; ++i) y[i] = ef[i] * b0[i] + gf[i] * b1[i];
            return;
        }
        if (direct_pairs) {
            for (int i = 0; i < h.n; ++i) {
                int j = mate[i];
                if (high_direct)
                    y[i] = static_cast<double>(efl[i] * b0[i] + pair_el[i] * b0[j] +
                                               gfl[i] * b1[i] + pair_gl[i] * b1[j]);
                else
                    y[i] = ef[i] * b0[i] + pair_e[i] * b0[j] +
                           gf[i] * b1[i] + pair_g[i] * b1[j];
            }
            return;
        }
        if (direct) {
            for (int i = 0; i < h.n; ++i) {
                if (high_direct) {
                    long double sum = 0;
                    for (int k = block_ptr[i]; k < block_ptr[i + 1]; ++k)
                        sum += efl[k] * b0[block_col[k]] + gfl[k] * b1[block_col[k]];
                    y[i] = static_cast<double>(sum);
                } else {
                    double sum = 0;
                    for (int k = block_ptr[i]; k < block_ptr[i + 1]; ++k)
                        sum += ef[k] * b0[block_col[k]] + gf[k] * b1[block_col[k]];
                    y[i] = sum;
                }
            }
            return;
        }
        long double norm0 = 0, norm1 = 0;
        double max0 = 0, max1 = 0;
        bool uniform0 = true, uniform1 = true;
        for (int i = 0; i < h.n; ++i) {
            norm0 += static_cast<long double>(b0[i]) * b0[i];
            norm1 += static_cast<long double>(b1[i]) * b1[i];
            max0 = std::max(max0, std::abs(b0[i]));
            max1 = std::max(max1, std::abs(b1[i]));
            uniform0 = uniform0 && std::abs(b0[i]) == std::abs(b0[0]);
            uniform1 = uniform1 && std::abs(b1[i]) == std::abs(b1[0]);
        }
        if ((max0 == 0 && max1 == 0) || (stationary && max1 == 0)) {
            std::fill(y.begin(), y.end(), 0);
            return;
        }
        if (!stationary && h.t * upper > 100 && uniform0 && uniform1 &&
            integer_eigenvectors(b0, b1, max0, max1, y)) return;
        norm0 = std::sqrt(norm0);
        norm1 = std::sqrt(norm1);
        long double guess = stationary ? max1 / lower : max0 + h.t * max1;
        long double initial_scale = guess;
        long double initial_factor = max1 == 0 ? 0.00001L : 0.0001L;
        if (!local_e.empty()) {
            initial_scale = 0;
            for (int i = 0; i < h.n; ++i)
                initial_scale = std::max(initial_scale,
                    static_cast<long double>(local_e[i] * std::abs(b0[i]) +
                                             local_g[i] * std::abs(b1[i])));
            initial_factor = 0.01L;
        }
        int m = degree(norm0, norm1, initial_factor * (h.atol + h.rtol * initial_scale));
        const char* krylov_setting = std::getenv("FAMEXP_KRYLOV");
        int krylov_cap = krylov_setting ? std::atoi(krylov_setting) :
                        ((h.rtol >= 1e-7 && m > 120) ? 16 : 0);
        if (krylov_cap > 0 && m > 32 && !stationary) {
            int cap = std::min(krylov_cap, m);
            vector<double> out0, out1;
            long double e0, e1;
            long double tolerance = 0.000002L * (h.atol + h.rtol * guess);
            bool ok = cap > 0 && try_krylov_rhs(b0, false, cap, tolerance, out0, e0) &&
                      try_krylov_rhs(b1, true, cap, tolerance, out1, e1);
            if (ok) {
                double ymax = 0;
                for (int i = 0; i < h.n; ++i) {
                    y[i] = out0[i] + out1[i];
                    ymax = std::max(ymax, std::abs(y[i]));
                }
                if (e0 + e1 < 0.1L * (h.atol + h.rtol * ymax) &&
                    !(h.rtol <= 1e-7 && ymax < 0.01 * guess)) return;
            }
        }
        bool extended = h.rtol <= 1e-10 && !stable_edges && h.t * upper > 1000;
        if (extended) evaluate_long(m, b0, b1, y);
        else evaluate(m, b0, b1, y);
        double ymax = 0;
        for (double a : y) ymax = std::max(ymax, std::abs(a));
        int required = degree(norm0, norm1, 0.04L * (h.atol + h.rtol * ymax));
        if (std::getenv("FAMEXP_DEBUG"))
            std::fprintf(stderr, "upper %.15g initial_degree %d required_degree %d ymax %.15g edges %d\n",
                         upper, m, required, ymax, stable_edges);
        bool cancellation = h.rtol <= 1e-7 && max0 > 0 && max1 > 0;
        bool small_signal = h.rtol <= 1e-10 || cancellation ||
                            (h.rtol <= 1e-7 && !stable_edges);
        if (!extended && small_signal && h.t * upper > 100 && ymax < 0.002 * guess) {
            evaluate_long(std::max(m, required), b0, b1, y);
        } else if (required > m) {
            if (extended) evaluate_long(required, b0, b1, y);
            else evaluate(required, b0, b1, y);
        }
    }

    void solve(const vector<double>& b0, const vector<double>& b1, vector<double>& y) {
        if (permutation.empty()) {
            solve_inner(b0, b1, y);
            return;
        }
        for (int i = 0; i < h.n; ++i) {
            input0[i] = b0[permutation[i]];
            input1[i] = b1[permutation[i]];
        }
        solve_inner(input0, input1, output);
        for (int i = 0; i < h.n; ++i) y[permutation[i]] = output[i];
    }
};

int main() {
    try {
        static char input_buffer[1 << 16], output_buffer[1 << 16];
        std::setvbuf(stdin, input_buffer, _IOFBF, sizeof(input_buffer));
        std::setvbuf(stdout, output_buffer, _IOFBF, sizeof(output_buffer));
        write_exact("READY\n", 6);
        char magic[8];
        read_exact(magic, 8);
        if (std::memcmp(magic, "FAMEXP2\n", 8)) throw std::runtime_error("bad magic");
        uint32_t length;
        read_exact(&length, sizeof(length));
        std::string json(length, '\0');
        read_exact(json.data(), length);
        Header h = parse_header(json);
        Solver solver(h);
        solver.prepare();
        vector<double> b0(h.n), b1(h.n), y(h.n);
        for (int request = 0; request < h.requests; ++request) {
            read_exact(b0.data(), b0.size() * sizeof(double));
            read_exact(b1.data(), b1.size() * sizeof(double));
            solver.solve(b0, b1, y);
            write_exact(y.data(), y.size() * sizeof(double));
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
