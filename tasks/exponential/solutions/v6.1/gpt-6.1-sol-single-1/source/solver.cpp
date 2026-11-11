#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <immintrin.h>
#include <Eigen/Eigenvalues>

using Real = long double;

static void read_exact(void* dst, size_t bytes) {
    unsigned char* p = static_cast<unsigned char*>(dst);
    while (bytes) {
        size_t k = std::fread(p, 1, bytes, stdin);
        if (!k) throw std::runtime_error("incomplete input");
        p += k;
        bytes -= k;
    }
}

static void write_exact(const void* src, size_t bytes) {
    const unsigned char* p = static_cast<const unsigned char*>(src);
    while (bytes) {
        size_t k = std::fwrite(p, 1, bytes, stdout);
        if (!k) throw std::runtime_error("output failure");
        p += k;
        bytes -= k;
    }
    std::fflush(stdout);
}

// The wire header is a flat JSON object of scalar values.
class Header {
    std::map<std::string, std::string> fields;
public:
    explicit Header(const std::string& s) {
        size_t i = 0;
        auto space = [&]() {
            while (i < s.size() && static_cast<unsigned char>(s[i]) <= ' ') ++i;
        };
        auto quoted = [&]() {
            if (s.at(i++) != '"') throw std::runtime_error("JSON string expected");
            std::string r;
            while (i < s.size() && s[i] != '"') {
                if (s[i] == '\\') {
                    ++i;
                    if (i == s.size()) throw std::runtime_error("JSON escape");
                }
                r += s[i++];
            }
            if (i == s.size()) throw std::runtime_error("JSON string");
            ++i;
            return r;
        };
        space();
        if (s.at(i++) != '{') throw std::runtime_error("JSON object expected");
        for (;;) {
            space();
            if (s.at(i) == '}') break;
            std::string key = quoted();
            space();
            if (s.at(i++) != ':') throw std::runtime_error("JSON colon expected");
            space();
            std::string value;
            if (s.at(i) == '"') value = quoted();
            else {
                size_t start = i;
                while (i < s.size() && s[i] != ',' && s[i] != '}'
                       && static_cast<unsigned char>(s[i]) > ' ') ++i;
                value = s.substr(start, i - start);
            }
            fields.emplace(std::move(key), std::move(value));
            space();
            if (s.at(i) == '}') break;
            if (s.at(i++) != ',') throw std::runtime_error("JSON comma expected");
        }
    }
    double number(const std::string& key) const {
        return std::strtod(fields.at(key).c_str(), nullptr);
    }
    int integer(const std::string& key) const {
        return std::stoi(fields.at(key));
    }
};

struct Block {
    std::vector<int> vertices;
    std::vector<Real> expmat, phimat;
};

struct Solver {
    int n, requests = 1;
    double t, rtol, atol, bound = 0;
    std::vector<uint64_t> ptr;
    std::vector<uint32_t> col;
    std::vector<double> data;
    std::vector<double> diag, residual;
    std::vector<Block> blocks;
    bool all_small = false, nonpositive = true, taylor = false;
    std::vector<int> witness;
    Real witness_residual = 0;
    std::vector<int> permutation;
    std::vector<double> input0, input1, output;
    int kernel = 0;
    std::vector<uint64_t> slice_ptr;
    std::vector<uint32_t> packed_col;
    std::vector<double> packed_data;
    std::vector<uint64_t> accurate_ptr;
    std::vector<uint32_t> accurate_col;
    std::vector<double> accurate_data;
    std::vector<Real> accurate_residual, accurate_u, accurate_next, accurate_delta;
    int last_degree = 0;
    std::vector<Real> ce, cp, tail_e, tail_p;
    std::vector<double> u, next, delta;

    void pack_rows() {
        slice_ptr.push_back(0);
        for (int start = 0; start + 4 <= n; start += 4) {
            uint64_t width = 0;
            for (int r = 0; r < 4; ++r)
                width = std::max(width, ptr[start + r + 1] - ptr[start + r]);
            for (uint64_t k = 0; k < width; ++k) {
                for (int r = 0; r < 4; ++r) {
                    uint64_t q = ptr[start + r] + k;
                    bool present = q < ptr[start + r + 1];
                    packed_col.push_back(present ? col[q] : 0);
                    packed_data.push_back(present ? data[q] : 0);
                }
            }
            slice_ptr.push_back(packed_col.size());
        }
    }

    void reorder(int mode) {
        if (!mode) return;
        std::vector<int> order(n), inverse(n);
        std::iota(order.begin(), order.end(), 0);
        auto degree = [&](int i) { return ptr[i + 1] - ptr[i]; };
        if (mode == 1 || mode == 3 || mode == 4) {
            std::stable_sort(order.begin(), order.end(),
                [&](int i, int j) { return degree(i) < degree(j); });
            std::vector<unsigned char> seen(n, 0);
            std::vector<int> queue, neighbors;
            queue.reserve(n);
            for (int root : order) {
                if (seen[root]) continue;
                size_t start = queue.size();
                queue.push_back(root);
                seen[root] = 1;
                for (size_t head = start; head < queue.size(); ++head) {
                    int i = queue[head];
                    neighbors.clear();
                    for (uint64_t k = ptr[i]; k < ptr[i + 1]; ++k) {
                        int j = col[k];
                        if (!seen[j]) {
                            seen[j] = 1;
                            neighbors.push_back(j);
                        }
                    }
                    if (mode != 4)
                        std::stable_sort(neighbors.begin(), neighbors.end(),
                            [&](int a, int b) { return degree(a) < degree(b); });
                    queue.insert(queue.end(), neighbors.begin(), neighbors.end());
                }
                std::reverse(queue.begin() + start, queue.end());
            }
            order.swap(queue);
        }
        if (mode == 2 || mode == 3) {
            for (int start = 0; start < n; start += 64) {
                int end = std::min(start + 64, n);
                std::stable_sort(order.begin() + start, order.begin() + end,
                    [&](int i, int j) { return degree(i) < degree(j); });
            }
        }
        for (int i = 0; i < n; ++i) inverse[order[i]] = i;
        std::vector<uint64_t> p(n + 1);
        std::vector<uint32_t> j;
        std::vector<double> a, r(n);
        std::vector<int> w(n);
        j.reserve(col.size());
        a.reserve(data.size());
        std::vector<std::pair<uint32_t, double>> row;
        for (int i = 0; i < n; ++i) {
            int old = order[i];
            row.clear();
            for (uint64_t k = ptr[old]; k < ptr[old + 1]; ++k)
                row.emplace_back(inverse[col[k]], data[k]);
            std::sort(row.begin(), row.end());
            for (auto [q, v] : row) {
                j.push_back(q);
                a.push_back(v);
            }
            p[i + 1] = j.size();
            r[i] = residual[old];
            w[i] = witness[old];
        }
        ptr.swap(p);
        col.swap(j);
        data.swap(a);
        residual.swap(r);
        witness.swap(w);
        permutation.swap(order);
        input0.resize(n);
        input1.resize(n);
        output.resize(n);
    }

    static Real phi(Real z) {
        return z == 0 ? 1 : std::expm1(-z) / (-z);
    }

    void prepare_blocks(const std::vector<int>& labels,
                        const std::vector<std::vector<int>>& components) {
        (void)labels;
        std::vector<int> pos(n, -1);
        for (const auto& vertices : components) {
            int m = static_cast<int>(vertices.size());
            if (m > 24) continue;
            Block b;
            b.vertices = vertices;
            b.expmat.resize(m * m);
            b.phimat.resize(m * m);
            for (int i = 0; i < m; ++i) pos[vertices[i]] = i;
            if (m == 1) {
                Real z = Real(t) * diag[vertices[0]];
                b.expmat[0] = std::exp(-z);
                b.phimat[0] = Real(t) * phi(z);
            } else if (m == 2) {
                Real aa = diag[vertices[0]], dd = diag[vertices[1]], bb = 0;
                for (uint64_t k = ptr[vertices[0]]; k < ptr[vertices[0] + 1]; ++k)
                    if (col[k] == static_cast<uint32_t>(vertices[1])) bb = data[k];
                Real center = (aa + dd) / 2;
                Real radius = std::hypot((aa - dd) / 2, bb);
                Real hi = center + radius;
                // Recover the small eigenvalue without subtracting nearly equal
                // numbers. Product residuals retain the binary64 input object.
                double ad = double(aa) * double(dd), b2 = double(bb) * double(bb);
                Real det = (Real(ad) - Real(b2))
                         + Real(std::fma(double(aa), double(dd), -ad))
                         - Real(std::fma(double(bb), double(bb), -b2));
                Real lo = hi == 0 ? center - radius : det / hi;
                Real el = std::exp(-Real(t) * lo), eh = std::exp(-Real(t) * hi);
                Real pl = Real(t) * phi(Real(t) * lo);
                Real ph = Real(t) * phi(Real(t) * hi);
                Real gap = hi - lo;
                if (gap == 0) {
                    b.expmat = {el, 0, 0, el};
                    b.phimat = {pl, 0, 0, pl};
                } else {
                    Real a = (aa - lo) / gap, d = (dd - lo) / gap, c = bb / gap;
                    b.expmat = {el + (eh - el) * a, (eh - el) * c,
                                (eh - el) * c, el + (eh - el) * d};
                    b.phimat = {pl + (ph - pl) * a, (ph - pl) * c,
                                (ph - pl) * c, pl + (ph - pl) * d};
                }
            } else {
                using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
                Matrix a = Matrix::Zero(m, m);
                for (int i = 0; i < m; ++i) {
                    int v = vertices[i];
                    for (uint64_t k = ptr[v]; k < ptr[v + 1]; ++k)
                        if (data[k] != 0) a(i, pos[col[k]]) = data[k];
                }
                Eigen::SelfAdjointEigenSolver<Matrix> es(a);
                if (es.info() != Eigen::Success) throw std::runtime_error("eigensolver");
                const Matrix& q = es.eigenvectors();
                std::vector<Real> ev(m), pv(m);
                for (int k = 0; k < m; ++k) {
                    Real z = Real(t) * es.eigenvalues()[k];
                    ev[k] = std::exp(-z);
                    pv[k] = Real(t) * phi(z);
                }
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < m; ++j) {
                        Real e = 0, p = 0;
                        for (int k = 0; k < m; ++k) {
                            Real v = q(i, k) * q(j, k);
                            e += ev[k] * v;
                            p += pv[k] * v;
                        }
                        b.expmat[i * m + j] = e;
                        b.phimat[i * m + j] = p;
                    }
            }
            blocks.push_back(std::move(b));
        }
    }

    void coefficients(Real z) {
        auto decay = [z](int m) -> Real {
            Real x = m;
            return x * std::asinh(x / z) - x * x / (std::hypot(z, x) + z);
        };
        int hi = std::max(4, int(std::sqrt(160 * z)) + 80);
        while (decay(hi) < 80) hi *= 2;
        int lo = 2;
        while (hi - lo > 1) {
            int m = (hi + lo) / 2;
            if (decay(m) >= 80) hi = m;
            else lo = m;
        }
        int m = hi + 8;
        std::vector<Real> ratio(m + 2, 0);
        for (int k = m; k >= 1; --k) ratio[k] = 1 / (2 * Real(k) / z + ratio[k + 1]);
        ce.assign(m + 1, 0);
        ce[0] = 1;
        Real sum = 1;
        Real v = 1;
        for (int k = 1; k <= m; ++k) {
            v *= ratio[k];
            ce[k] = 2 * v;
            sum += ce[k];
        }
        for (auto& c : ce) c /= sum;
        cp.assign(m + 1, 0);
        Real s = 0, w = 0;
        for (int k = m - 1; k >= 0; --k) {
            s += ce[k + 1];
            w += s;
            cp[k] = (k == 0 ? 1 : 2) * w / z;
        }
        tail_e.assign(m + 1, 0);
        tail_p.assign(m + 1, 0);
        for (int k = m - 1; k >= 0; --k) {
            tail_e[k] = tail_e[k + 1] + ce[k + 1];
            tail_p[k] = tail_p[k + 1] + cp[k + 1];
        }
    }

    void prepare() {
        if (t == 0 || data.empty()) return;
        diag.assign(n, 0);
        residual.assign(n, 0);
        for (int i = 0; i < n; ++i) {
            Real off = 0, correction = 0;
            for (uint64_t k = ptr[i]; k < ptr[i + 1]; ++k) {
                if (col[k] == static_cast<uint32_t>(i)) diag[i] = data[k];
                else {
                    Real add = std::abs(Real(data[k])) - correction;
                    Real total = off + add;
                    correction = (total - off) - add;
                    off = total;
                    if (data[k] > 0) nonpositive = false;
                }
            }
            residual[i] = static_cast<double>(Real(diag[i]) - off);
            bound = std::max(bound, static_cast<double>(Real(diag[i]) + off));
        }
        bound *= 1 + 4e-15;
        if (bound == 0) return;
        if (t * bound <= (requests == 1 ? 2 : 0.25)) {
            taylor = true;
            for (auto& v : data) v *= -t;
            u.resize(n);
            next.resize(n);
            return;
        }
        std::vector<int> labels(n, -1), stack;
        std::vector<std::vector<int>> components;
        witness.assign(n, 1);
        int largest = 0;
        for (int start = 0; start < n; ++start) {
            if (labels[start] >= 0) continue;
            int label = static_cast<int>(components.size());
            components.emplace_back();
            auto& component = components.back();
            stack.push_back(start);
            labels[start] = label;
            while (!stack.empty()) {
                int i = stack.back();
                stack.pop_back();
                component.push_back(i);
                for (uint64_t k = ptr[i]; k < ptr[i + 1]; ++k) {
                    int j = col[k];
                    if (j != i && data[k] != 0 && labels[j] < 0) {
                        labels[j] = label;
                        witness[j] = witness[i] * (data[k] > 0 ? -1 : 1);
                        stack.push_back(j);
                    }
                }
            }
            largest = std::max(largest, static_cast<int>(component.size()));
        }
        all_small = largest <= 24;
        if (all_small) {
            prepare_blocks(labels, components);
            return;
        }
        for (int i = 0; i < n; ++i) {
            Real s = 0;
            for (uint64_t k = ptr[i]; k < ptr[i + 1]; ++k)
                s += Real(data[k]) * witness[col[k]];
            witness_residual += s * s;
        }
        witness_residual = std::sqrt(witness_residual);
        // Positive diagonal similarities give rigorous upper spectral bounds.
        if (t * bound > 200) {
            std::vector<double> x(n, 1), v(n);
            int iterations = requests == 1 && t * bound <= 2000 ? 6 : 12;
            for (int iter = 0; iter < iterations; ++iter) {
                double upper = 0, maximum = 0;
                for (int i = 0; i < n; ++i) {
                    double s = 0;
                    for (uint64_t k = ptr[i]; k < ptr[i + 1]; ++k)
                        s += std::abs(data[k]) * x[col[k]];
                    v[i] = s;
                    upper = std::max(upper, s / x[i]);
                    maximum = std::max(maximum, s);
                }
                bound = std::min(bound, upper * (1 + 1e-10));
                for (int i = 0; i < n; ++i) x[i] = std::max(v[i] / maximum, 1e-100);
            }
        }
        bound *= 1 + 4e-15;
        coefficients(Real(t) * Real(bound) / 2);

        // Remove the diagonal; retain its exact row compensation separately.
        std::vector<uint64_t> p(n + 1, 0);
        std::vector<uint32_t> j;
        std::vector<double> a;
        std::vector<double> raw;
        j.reserve(data.size());
        a.reserve(data.size());
        bool retain = rtol <= 1e-10;
        if (retain) {
            raw.reserve(data.size());
            accurate_residual.resize(n);
        }
        Real scale = -4 / Real(bound);
        for (int i = 0; i < n; ++i) {
            Real off = 0, correction = 0;
            for (uint64_t k = ptr[i]; k < ptr[i + 1]; ++k) {
                if (col[k] == static_cast<uint32_t>(i)) continue;
                Real add = std::abs(Real(data[k])) - correction;
                Real total = off + add;
                correction = (total - off) - add;
                off = total;
                j.push_back(col[k]);
                a.push_back(static_cast<double>(scale * Real(data[k])));
                if (retain) raw.push_back(data[k]);
            }
            residual[i] = static_cast<double>(scale * (Real(diag[i]) - off));
            if (retain) accurate_residual[i] = Real(diag[i]) - off;
            p[i + 1] = j.size();
        }
        if (retain) {
            accurate_ptr = p;
            accurate_col = j;
            accurate_data.swap(raw);
        }
        ptr.swap(p);
        col.swap(j);
        data.swap(a);
        double work = requests * std::sqrt(35 * t * bound);
        double span = 0;
        uint64_t padded = 0;
        for (int i = 0; i < n; ++i)
            for (uint64_t k = ptr[i]; k < ptr[i + 1]; ++k)
                span += std::abs(double(i) - col[k]);
        for (int i = 0; i + 4 <= n; i += 4) {
            uint64_t width = 0;
            for (int r = 0; r < 4; ++r)
                width = std::max(width, ptr[i + r + 1] - ptr[i + r]);
            padded += 4 * width;
        }
        span /= std::max(size_t(1), col.size());
        int order_mode = 0;
        if (work >= 150 && n >= 512 && span > n * 0.04) order_mode = 3;
        else if (work >= 150 && padded > 1.3 * col.size()) order_mode = 2;
        if (const char* v = std::getenv("FAMEXP_ORDER")) order_mode = std::atoi(v);
        reorder(order_mode);
        if (order_mode) {
            padded = 0;
            for (int i = 0; i + 4 <= n; i += 4) {
                uint64_t width = 0;
                for (int r = 0; r < 4; ++r)
                    width = std::max(width, ptr[i + r + 1] - ptr[i + r]);
                padded += 4 * width;
            }
        }
        kernel = n >= 128 && work >= 40 ? 2 : 0;
        if (kernel && padded > 1.7 * col.size()) kernel = 0;
        if (const char* v = std::getenv("FAMEXP_KERNEL")) kernel = std::atoi(v);
        if (kernel) pack_rows();
        if (std::getenv("FAMEXP_DEBUG"))
            std::fprintf(stderr, "order=%d kernel=%d\n", order_mode, kernel);
        u.resize(n);
        next.resize(n);
        delta.resize(n);
    }

    template<bool Negative, bool Extended>
    void evaluate(const double* b0, const double* b1, double* y, int degree) {
        std::fill(u.begin(), u.end(), 0);
        std::fill(delta.begin(), delta.end(), 0);
        for (int k = degree; k >= 1; --k) {
            double e = static_cast<double>(ce[k]);
            double p = static_cast<double>(Real(t) * cp[k]);
            for (int i = 0; i < n; ++i) {
                using Acc = typename std::conditional<Extended, Real, double>::type;
                Acc s = Acc(residual[i]) * u[i];
                double ui = u[i];
                for (uint64_t q = ptr[i]; q < ptr[i + 1]; ++q) {
                    double v = data[q];
                    double diff;
                    if constexpr (Negative) diff = u[col[q]] - ui;
                    else diff = u[col[q]] - (v >= 0 ? ui : -ui);
                    s += Acc(v) * diff;
                }
                double d = static_cast<double>(Acc(delta[i]) + s
                           + Acc(e) * b0[i] + Acc(p) * b1[i]);
                delta[i] = d;
                next[i] = ui + d;
            }
            u.swap(next);
        }
        double e = static_cast<double>(ce[0]);
        double p = static_cast<double>(Real(t) * cp[0]);
        for (int i = 0; i < n; ++i) {
            using Acc = typename std::conditional<Extended, Real, double>::type;
            Acc s = Acc(residual[i]) * u[i];
            double ui = u[i];
            for (uint64_t q = ptr[i]; q < ptr[i + 1]; ++q) {
                double v = data[q];
                double diff;
                if constexpr (Negative) diff = u[col[q]] - ui;
                else diff = u[col[q]] - (v >= 0 ? ui : -ui);
                s += Acc(v) * diff;
            }
            y[i] = static_cast<double>(Acc(delta[i]) + s / 2
                     + Acc(e) * b0[i] + Acc(p) * b1[i]);
        }
    }

    template<bool Negative, bool Gather>
    void evaluate_packed(const double* b0, const double* b1, double* y, int degree) {
        std::fill(u.begin(), u.end(), 0);
        std::fill(delta.begin(), delta.end(), 0);
        const __m256d sign = _mm256_set1_pd(-0.0);
        int n4 = (n / 4) * 4;
        auto product = [&](int i, int slice) -> __m256d {
            __m256d ui = _mm256_loadu_pd(u.data() + i);
            __m256d s = _mm256_mul_pd(_mm256_loadu_pd(residual.data() + i), ui);
            for (uint64_t q = slice_ptr[slice]; q < slice_ptr[slice + 1]; q += 4) {
                __m256d v = _mm256_loadu_pd(packed_data.data() + q);
                __m256d uj;
                if constexpr (Gather)
                    uj = _mm256_i32gather_pd(u.data(),
                        _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed_col.data() + q)), 8);
                else
                    uj = _mm256_set_pd(u[packed_col[q + 3]], u[packed_col[q + 2]],
                                       u[packed_col[q + 1]], u[packed_col[q]]);
                __m256d base;
                if constexpr (Negative) base = ui;
                else base = _mm256_xor_pd(ui, _mm256_and_pd(v, sign));
                s = _mm256_fmadd_pd(v, _mm256_sub_pd(uj, base), s);
            }
            return s;
        };
        auto tail_product = [&](int i) -> double {
            double s = residual[i] * u[i];
            for (uint64_t q = ptr[i]; q < ptr[i + 1]; ++q) {
                double v = data[q];
                double base = Negative ? u[i] : (v >= 0 ? u[i] : -u[i]);
                s += v * (u[col[q]] - base);
            }
            return s;
        };
        for (int k = degree; k >= 1; --k) {
            double e = static_cast<double>(ce[k]), p = static_cast<double>(Real(t) * cp[k]);
            __m256d ev = _mm256_set1_pd(e), pv = _mm256_set1_pd(p);
            for (int i = 0; i < n4; i += 4) {
                __m256d s = product(i, i / 4);
                __m256d d = _mm256_add_pd(_mm256_loadu_pd(delta.data() + i), s);
                d = _mm256_fmadd_pd(ev, _mm256_loadu_pd(b0 + i), d);
                d = _mm256_fmadd_pd(pv, _mm256_loadu_pd(b1 + i), d);
                _mm256_storeu_pd(delta.data() + i, d);
                _mm256_storeu_pd(next.data() + i, _mm256_add_pd(_mm256_loadu_pd(u.data() + i), d));
            }
            for (int i = n4; i < n; ++i) {
                delta[i] += tail_product(i) + e * b0[i] + p * b1[i];
                next[i] = u[i] + delta[i];
            }
            u.swap(next);
        }
        __m256d ev = _mm256_set1_pd(static_cast<double>(ce[0]));
        __m256d pv = _mm256_set1_pd(static_cast<double>(Real(t) * cp[0]));
        __m256d half = _mm256_set1_pd(0.5);
        for (int i = 0; i < n4; i += 4) {
            __m256d s = product(i, i / 4);
            __m256d d = _mm256_fmadd_pd(half, s, _mm256_loadu_pd(delta.data() + i));
            d = _mm256_fmadd_pd(ev, _mm256_loadu_pd(b0 + i), d);
            d = _mm256_fmadd_pd(pv, _mm256_loadu_pd(b1 + i), d);
            _mm256_storeu_pd(y + i, d);
        }
        for (int i = n4; i < n; ++i)
            y[i] = delta[i] + tail_product(i) / 2
                 + static_cast<double>(ce[0]) * b0[i]
                 + static_cast<double>(Real(t) * cp[0]) * b1[i];
    }

    void evaluate_accurate(const double* b0, const double* b1, double* y, int degree) {
        accurate_u.assign(n, 0);
        accurate_next.resize(n);
        accurate_delta.assign(n, 0);
        Real scale = -4 / Real(bound);
        auto product = [&](int i) -> Real {
            Real ui = accurate_u[i];
            Real s = accurate_residual[i] * ui;
            Real correction = 0;
            bool compensated = accurate_ptr[i + 1] - accurate_ptr[i] > 32;
            for (uint64_t q = accurate_ptr[i]; q < accurate_ptr[i + 1]; ++q) {
                double v = accurate_data[q];
                Real base = v <= 0 ? ui : -ui;
                Real add = Real(v) * (accurate_u[accurate_col[q]] - base);
                if (compensated) {
                    add -= correction;
                    Real total = s + add;
                    correction = (total - s) - add;
                    s = total;
                } else s += add;
            }
            return scale * s;
        };
        for (int k = degree; k >= 1; --k) {
            Real e = ce[k], p = Real(t) * cp[k];
            for (int i = 0; i < n; ++i) {
                Real d = accurate_delta[i] + product(i) + e * Real(b0[i]) + p * Real(b1[i]);
                accurate_delta[i] = d;
                accurate_next[i] = accurate_u[i] + d;
            }
            accurate_u.swap(accurate_next);
        }
        for (int i = 0; i < n; ++i)
            y[i] = static_cast<double>(accurate_delta[i] + product(i) / 2
                     + ce[0] * Real(b0[i]) + Real(t) * cp[0] * Real(b1[i]));
    }

    bool near_null(const double* b0, const double* b1, double* y) {
        Real r0 = 0, r1 = 0, e0 = 0, e1 = 0, ymax = 0;
        constexpr Real eps = std::numeric_limits<double>::epsilon();
        auto product = [&](int i, const double* b, Real& error) {
            double hi = diag[i] * b[i];
            double lo = std::fma(diag[i], b[i], -hi);
            Real magnitude = std::abs(Real(diag[i]) * b[i]);
            for (uint64_t q = accurate_ptr[i]; q < accurate_ptr[i + 1]; ++q) {
                double v = accurate_data[q], x = b[accurate_col[q]];
                double p = v * x;
                double pe = std::fma(v, x, -p);
                double total = hi + p;
                double z = total - hi;
                lo += ((hi - (total - z)) + (p - z)) + pe;
                hi = total;
                magnitude += std::abs(Real(v) * Real(x));
            }
            Real value = Real(hi) + Real(lo);
            Real terms = accurate_ptr[i + 1] - accurate_ptr[i] + 2;
            error = 32 * eps * eps * terms * terms * magnitude
                  + 4 * std::numeric_limits<Real>::epsilon() * std::abs(value) + 1e-295L;
            return value;
        };
        for (int i = 0; i < n; ++i) {
            Real error0, error1;
            Real a = product(i, b0, error0), b = product(i, b1, error1);
            r0 += a * a;
            r1 += b * b;
            e0 += error0 * error0;
            e1 += error1 * error1;
            ymax = std::max(ymax, std::abs(Real(b0[i]) + Real(t) * b1[i]));
        }
        Real error = Real(t) * (std::sqrt(r0) + std::sqrt(e0))
                   + Real(t) * Real(t) / 2 * (std::sqrt(r1) + std::sqrt(e1));
        if (error > 0.03L * (Real(atol) + Real(rtol) * ymax)) return false;
        for (int i = 0; i < n; ++i)
            y[i] = static_cast<double>(Real(b0[i]) + Real(t) * b1[i]);
        return true;
    }

    double taylor_product(const double* x, int i, double initial) const {
        if (rtol <= 1e-10 && ptr[i + 1] - ptr[i] > 64) {
            Real s = initial, correction = 0;
            for (uint64_t q = ptr[i]; q < ptr[i + 1]; ++q) {
                Real add = Real(data[q]) * Real(x[col[q]]) - correction;
                Real total = s + add;
                correction = (total - s) - add;
                s = total;
            }
            return static_cast<double>(s);
        }
        double s = initial;
        for (uint64_t q = ptr[i]; q < ptr[i + 1]; ++q) s += data[q] * x[col[q]];
        return s;
    }

    void solve_core(const double* b0, const double* b1, double* y) {
        if (t == 0 || bound == 0) {
            for (int i = 0; i < n; ++i) y[i] = static_cast<double>(Real(b0[i]) + Real(t) * b1[i]);
            return;
        }
        if (taylor) {
            double norm = 0, ynorm = 0;
            for (int i = 0; i < n; ++i) {
                double s = taylor_product(b0, i, t * b1[i]);
                u[i] = s;
                y[i] = b0[i] + s;
                norm = std::max(norm, std::abs(s));
                ynorm = std::max(ynorm, std::abs(y[i]));
            }
            int k = 1;
            double eta = t * bound;
            while (norm * eta / (k + 1) / (1 - eta / (k + 2))
                   > 0.02 * (atol + rtol * ynorm)) {
                norm = 0;
                ynorm = 0;
                for (int i = 0; i < n; ++i) {
                    double s = taylor_product(u.data(), i, 0) / (k + 1);
                    next[i] = s;
                    y[i] += s;
                    norm = std::max(norm, std::abs(s));
                    ynorm = std::max(ynorm, std::abs(y[i]));
                }
                ++k;
                u.swap(next);
            }
            return;
        }
        if (all_small) {
            for (const auto& b : blocks) {
                int m = b.vertices.size();
                for (int i = 0; i < m; ++i) {
                    Real s = 0;
                    for (int j = 0; j < m; ++j)
                        s += b.expmat[i * m + j] * b0[b.vertices[j]]
                           + b.phimat[i * m + j] * b1[b.vertices[j]];
                    y[b.vertices[i]] = static_cast<double>(s);
                }
            }
            return;
        }
        Real norm0 = 0, norm1 = 0, projection0 = 0, projection1 = 0;
        for (int i = 0; i < n; ++i) {
            norm0 += Real(b0[i]) * b0[i];
            norm1 += Real(b1[i]) * b1[i];
            projection0 += Real(witness[i]) * b0[i];
            projection1 += Real(witness[i]) * b1[i];
        }
        norm0 = std::sqrt(norm0);
        norm1 = Real(t) * std::sqrt(norm1);
        if (norm0 + norm1 == 0) {
            std::fill(y, y + n, 0);
            return;
        }
        Real lower = (std::abs(projection0 + Real(t) * projection1)
                      - witness_residual * Real(t) * (norm0 + norm1 / 2)) / n;
        lower = std::max(Real(0), lower);
        int degree = 0;
        Real tolerance = 0.02L * (Real(atol) + Real(rtol) * lower);
        while (degree + 1 < static_cast<int>(ce.size()) &&
               tail_e[degree] * norm0 + tail_p[degree] * norm1 > tolerance) ++degree;
        last_degree = degree;
        if (kernel == 1) {
            if (nonpositive) evaluate_packed<true, true>(b0, b1, y, degree);
            else evaluate_packed<false, true>(b0, b1, y, degree);
        } else if (kernel == 2) {
            if (nonpositive) evaluate_packed<true, false>(b0, b1, y, degree);
            else evaluate_packed<false, false>(b0, b1, y, degree);
        } else {
            if (nonpositive) evaluate<true, false>(b0, b1, y, degree);
            else evaluate<false, false>(b0, b1, y, degree);
        }
        if (std::getenv("FAMEXP_DEBUG"))
            std::fprintf(stderr, "n=%d eta=%.8g degree=%d\n", n, t * bound, degree);
    }

    void solve(const double* b0, const double* b1, double* y) {
        if (permutation.empty()) {
            solve_core(b0, b1, y);
        } else {
            for (int i = 0; i < n; ++i) {
                input0[i] = b0[permutation[i]];
                input1[i] = b1[permutation[i]];
            }
            solve_core(input0.data(), input1.data(), output.data());
            for (int i = 0; i < n; ++i) y[permutation[i]] = output[i];
        }
        if (!accurate_data.empty()) {
            double ymax = 0, b0max = 0, b1max = 0;
            for (int i = 0; i < n; ++i) {
                ymax = std::max(ymax, std::abs(y[i]));
                b0max = std::max(b0max, std::abs(b0[i]));
                b1max = std::max(b1max, std::abs(b1[i]));
            }
            double risk = 2e-15 * (t * bound * b0max + t * b1max);
            if (risk > atol + rtol * ymax) {
                if (!near_null(b0, b1, y)) {
                    evaluate_accurate(b0, b1, y, last_degree);
                    if (std::getenv("FAMEXP_DEBUG"))
                        std::fprintf(stderr, "extended retry\n");
                }
            }
        }
    }
};

int main() {
    try {
        std::setvbuf(stdin, nullptr, _IOFBF, 1 << 20);
        std::setvbuf(stdout, nullptr, _IOFBF, 1 << 20);
        write_exact("READY\n", 6);
        char magic[8];
        read_exact(magic, 8);
        if (std::memcmp(magic, "FAMEXP2\n", 8)) throw std::runtime_error("bad magic");
        uint32_t length;
        read_exact(&length, 4);
        if (length > 4096) throw std::runtime_error("header too long");
        std::string json(length, '\0');
        read_exact(json.data(), length);
        Header h(json);
        Solver s;
        s.n = h.integer("n");
        int nnz = h.integer("nnz"), requests = h.integer("requests");
        s.requests = requests;
        s.t = h.number("t");
        s.rtol = h.number("rtol");
        s.atol = h.number("atol");
        s.ptr.resize(s.n + 1);
        s.col.resize(nnz);
        s.data.resize(nnz);
        read_exact(s.ptr.data(), 8 * (s.n + 1));
        read_exact(s.col.data(), 4 * nnz);
        read_exact(s.data.data(), 8 * nnz);
        s.prepare();
        std::vector<double> b0(s.n), b1(s.n), y(s.n);
        for (int k = 0; k < requests; ++k) {
            read_exact(b0.data(), 8 * s.n);
            read_exact(b1.data(), 8 * s.n);
            s.solve(b0.data(), b1.data(), y.data());
            write_exact(y.data(), 8 * s.n);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "solver: %s\n", e.what());
        return 1;
    }
}
