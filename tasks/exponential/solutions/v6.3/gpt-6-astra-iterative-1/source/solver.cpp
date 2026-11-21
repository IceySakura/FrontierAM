#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

using Vec = std::vector<double>;

struct Buffers {
    std::vector<uint64_t> wireptr;
    std::vector<uint32_t> ptr, col;
    Vec val, diagonal, rowsum, b0, b1, y, d, delta, work;
    Vec e0, e1, f0, f1;

    static size_t initial_n(int mode) {
        return mode == 2 ? 120000 : mode == 1 ? 32768 : 0;
    }

    explicit Buffers(int mode)
        : wireptr(mode ? initial_n(mode) + 1 : 0),
          ptr(mode ? initial_n(mode) + 1 : 0),
          col(mode == 2 ? 1000000 : mode == 1 ? 200000 : 0),
          val(mode == 2 ? 1000000 : mode == 1 ? 200000 : 0),
          diagonal(initial_n(mode)), rowsum(initial_n(mode)),
          b0(initial_n(mode)), b1(initial_n(mode)), y(initial_n(mode)),
          d(initial_n(mode)), delta(initial_n(mode)), work(initial_n(mode)),
          e0(initial_n(mode)), e1(initial_n(mode)),
          f0(initial_n(mode)), f1(initial_n(mode)) {
        if (mode == 1) {
            wireptr.reserve(120001); ptr.reserve(120001);
            col.reserve(1000000); val.reserve(1000000);
            for (Vec* v : {&diagonal, &rowsum, &b0, &b1, &y, &d, &delta, &work,
                           &e0, &e1, &f0, &f1}) v->reserve(120000);
        }
    }
};

static void read_all(void* p, size_t n) {
    char* s = static_cast<char*>(p);
    while (n) {
        ssize_t r = read(STDIN_FILENO, s, n);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) throw std::runtime_error("short input");
        s += r;
        n -= static_cast<size_t>(r);
    }
}

static void write_all(const void* p, size_t n) {
    const char* s = static_cast<const char*>(p);
    while (n) {
        ssize_t r = write(STDOUT_FILENO, s, n);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) throw std::runtime_error("output failure");
        s += r;
        n -= static_cast<size_t>(r);
    }
}

struct Metadata {
    int n = 0, nnz = 0, requests = 0;
    double t = 0, rtol = 0, atol = 0;
};

// The protocol uses a flat JSON object containing strings and numbers.
static Metadata parse_metadata(const std::string& json) {
    const char* p = json.c_str();
    auto space = [&]() {
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') ++p;
    };
    auto hex4 = [&]() {
        uint32_t value = 0;
        for (int j = 0; j < 4; ++j) {
            unsigned char c = *p++;
            unsigned digit = c >= '0' && c <= '9' ? c-'0' :
                             c >= 'a' && c <= 'f' ? c-'a'+10 :
                             c >= 'A' && c <= 'F' ? c-'A'+10 : 16;
            if (digit >= 16) throw std::runtime_error("invalid JSON escape");
            value = 16 * value + digit;
        }
        return value;
    };
    auto string = [&]() {
        if (*p++ != '"') throw std::runtime_error("expected JSON string");
        std::string s;
        while (*p && *p != '"') {
            if (*p == '\\') {
                ++p;
                char escape = *p++;
                if (escape == 'u') {
                    uint32_t value = hex4();
                    if (value >= 0xd800 && value <= 0xdbff && p[0] == '\\' && p[1] == 'u') {
                        p += 2;
                        uint32_t low = hex4();
                        if (low < 0xdc00 || low > 0xdfff)
                            throw std::runtime_error("invalid JSON surrogate");
                        value = 0x10000 + ((value-0xd800) << 10) + (low-0xdc00);
                    }
                    if (value < 0x80) s += static_cast<char>(value);
                    else if (value < 0x800) {
                        s += static_cast<char>(0xc0 | (value >> 6));
                        s += static_cast<char>(0x80 | (value & 63));
                    } else if (value < 0x10000) {
                        s += static_cast<char>(0xe0 | (value >> 12));
                        s += static_cast<char>(0x80 | ((value >> 6) & 63));
                        s += static_cast<char>(0x80 | (value & 63));
                    } else {
                        s += static_cast<char>(0xf0 | (value >> 18));
                        s += static_cast<char>(0x80 | ((value >> 12) & 63));
                        s += static_cast<char>(0x80 | ((value >> 6) & 63));
                        s += static_cast<char>(0x80 | (value & 63));
                    }
                } else if (escape == 'n') s += '\n';
                else if (escape == 'r') s += '\r';
                else if (escape == 't') s += '\t';
                else if (escape == 'b') s += '\b';
                else if (escape == 'f') s += '\f';
                else s += escape;
            } else s += *p++;
        }
        if (*p++ != '"') throw std::runtime_error("unterminated JSON string");
        return s;
    };
    Metadata h;
    space();
    if (*p++ != '{') throw std::runtime_error("expected JSON object");
    for (;;) {
        space();
        if (*p == '}') break;
        std::string key = string();
        space();
        if (*p++ != ':') throw std::runtime_error("expected JSON colon");
        space();
        if (*p == '"') {
            string();
        } else {
            char* end;
            double value = std::strtod(p, &end);
            if (end == p) throw std::runtime_error("expected JSON number");
            p = end;
            if (key == "n") h.n = static_cast<int>(value);
            else if (key == "nnz") h.nnz = static_cast<int>(value);
            else if (key == "requests") h.requests = static_cast<int>(value);
            else if (key == "t") h.t = value;
            else if (key == "rtol") h.rtol = value;
            else if (key == "atol") h.atol = value;
        }
        space();
        if (*p == '}') break;
        if (*p++ != ',') throw std::runtime_error("expected JSON comma");
    }
    if (h.n < 1 || h.requests < 1) throw std::runtime_error("invalid header");
    return h;
}

static double maxabs(const Vec& v) {
    double x = 0;
    for (double a : v) x = std::max(x, std::abs(a));
    return x;
}

static double norm2(const Vec& v) {
    double s = 0;
    for (double a : v) s += a * a;
    return std::sqrt(s);
}

static double accurate_dot(const double* x, const double* y, int n) {
    long double total = 0;
    for (int begin = 0; begin < n; begin += 256) {
        int end = std::min(begin + 256, n);
        double part = 0;
        for (int i = begin; i < end; ++i) part += x[i] * y[i];
        total += static_cast<long double>(part);
    }
    return static_cast<double>(total);
}

static void basis_product(const double* basis, const Vec& coefficients, Vec& result, int n) {
    std::fill(result.begin(), result.end(), 0);
    double* __restrict out = result.data();
    int m = static_cast<int>(coefficients.size()), j = 0;
    for (; j + 3 < m; j += 4) {
        const double* __restrict q0 = basis + static_cast<size_t>(j) * n;
        const double* __restrict q1 = q0 + n;
        const double* __restrict q2 = q1 + n;
        const double* __restrict q3 = q2 + n;
        double c0 = coefficients[j], c1 = coefficients[j + 1];
        double c2 = coefficients[j + 2], c3 = coefficients[j + 3];
        for (int i = 0; i < n; ++i)
            out[i] += c0 * q0[i] + c1 * q1[i] + c2 * q2[i] + c3 * q3[i];
    }
    for (; j < m; ++j) {
        const double* __restrict q = basis + static_cast<size_t>(j) * n;
        double c = coefficients[j];
        for (int i = 0; i < n; ++i) out[i] += c * q[i];
    }
}

static double phi_double(double x) {
    return x == 0 ? 1 : -std::expm1(-x) / x;
}

__attribute__((optimize("no-associative-math")))
static long double compensated_sum(const double* values, uint32_t begin, uint32_t end) {
    long double sum = 0, correction = 0;
    for (uint32_t k = begin; k < end; ++k) {
        long double value = static_cast<long double>(values[k]) - correction;
        long double next = sum + value;
        correction = (next - sum) - value;
        sum = next;
    }
    return sum;
}

struct Matrix {
    int n;
    std::vector<uint32_t> ptr, col;
    Vec val, diagonal, rowsum;
    double upper = 0, mean_distance = 0, row_sum_error = 0, logarithmic_growth = 0;
    double logarithmic_measure = -std::numeric_limits<double>::infinity();
    bool pairs = true, unbalanced = false;
    bool uniform_off = false;
    double off_value = 0;
    std::vector<int> signs;

    explicit Matrix(const Metadata& h, Buffers& buffers)
        : n(h.n), ptr(std::move(buffers.ptr)), col(std::move(buffers.col)),
          val(std::move(buffers.val)), diagonal(std::move(buffers.diagonal)),
          rowsum(std::move(buffers.rowsum)) {
        ptr.resize(n + 1); col.resize(h.nnz); val.resize(h.nnz);
        diagonal.resize(n); rowsum.resize(n);
        auto& p = buffers.wireptr;
        p.resize(n + 1);
        read_all(p.data(), p.size() * sizeof(uint64_t));
        for (int i = 0; i <= n; ++i) ptr[i] = static_cast<uint32_t>(p[i]);
        read_all(col.data(), col.size() * sizeof(uint32_t));
        read_all(val.data(), val.size() * sizeof(double));
        size_t off_count = 0;
        int sum_error_multiplier = 2;
        double max_row_sum = 0;
        for (int i = 0; i < n; ++i) {
            long double sum = 0, abs_sum = 0;
            int off = 0;
            for (uint32_t j = ptr[i]; j < ptr[i + 1]; ++j) {
                sum += static_cast<long double>(val[j]);
                abs_sum += std::abs(static_cast<long double>(val[j]));
                if (col[j] == static_cast<uint32_t>(i)) diagonal[i] = val[j];
                else if (val[j] != 0) {
                    ++off;
                    ++off_count;
                    mean_distance += std::abs(static_cast<int>(col[j]) - i);
                }
            }
            pairs = pairs && off <= 1;
            int count = ptr[i + 1] - ptr[i];
            if (count > 64) sum = compensated_sum(val.data(), ptr[i], ptr[i + 1]);
            sum_error_multiplier = std::max(sum_error_multiplier, count > 64 ? 8 : count + 2);
            rowsum[i] = static_cast<double>(sum);
            max_row_sum = std::max(max_row_sum, std::abs(rowsum[i]));
            upper = std::max(upper, static_cast<double>(abs_sum));
            logarithmic_measure = std::max(logarithmic_measure,
                static_cast<double>(abs_sum * (1 + 1e-12L) - 2 * diagonal[i]));
        }
        if (off_count) mean_distance /= off_count;
        logarithmic_growth = std::max(0.0, logarithmic_measure);
        upper *= 1 + 1e-10;
        row_sum_error = (sum_error_multiplier * std::numeric_limits<long double>::epsilon() *
                         upper + std::numeric_limits<double>::epsilon() * max_row_sum) *
                        (1 + 1e-10);
    }

    void balance_signs() {
        bool has_positive = false;
        for (int i = 0; i < n && !has_positive; ++i)
            for (uint32_t k = ptr[i]; k < ptr[i + 1]; ++k)
                if (col[k] != static_cast<uint32_t>(i) && val[k] > 0) {
                    has_positive = true;
                    break;
                }
        if (!has_positive) return;
        struct DSU {
            std::vector<int> parent, rank, parity;
            explicit DSU(int n) : parent(n), rank(n), parity(n) {
                for (int i = 0; i < n; ++i) parent[i] = i;
            }
            int find(int x) {
                if (parent[x] == x) return x;
                int previous = parent[x];
                parent[x] = find(previous);
                parity[x] ^= parity[previous];
                return parent[x];
            }
            bool join(int i, int j, int different) {
                int ri = find(i), rj = find(j);
                int relative = parity[i] ^ parity[j] ^ different;
                if (ri == rj) return relative == 0;
                if (rank[ri] < rank[rj]) std::swap(ri, rj);
                parent[rj] = ri;
                parity[rj] = relative;
                rank[ri] += rank[ri] == rank[rj];
                return true;
            }
        } dsu(n);
        for (int i = 0; i < n; ++i) {
            for (uint32_t k = ptr[i]; k < ptr[i + 1]; ++k) {
                if (col[k] <= static_cast<uint32_t>(i) || val[k] == 0) continue;
                if (!dsu.join(i, col[k], val[k] > 0)) {
                    unbalanced = true;
                    return;
                }
            }
        }
        signs.resize(n);
        for (int i = 0; i < n; ++i) {
            dsu.find(i);
            signs[i] = dsu.parity[i] ? -1 : 1;
        }
        int sum_error_multiplier = 2;
        double max_row_sum = 0;
        for (int i = 0; i < n; ++i) {
            long double sum = 0;
            for (uint32_t k = ptr[i]; k < ptr[i + 1]; ++k) {
                val[k] *= signs[i] * signs[col[k]];
                sum += static_cast<long double>(val[k]);
            }
            int count = ptr[i + 1] - ptr[i];
            if (count > 64) sum = compensated_sum(val.data(), ptr[i], ptr[i + 1]);
            sum_error_multiplier = std::max(sum_error_multiplier, count > 64 ? 8 : count + 2);
            rowsum[i] = static_cast<double>(sum);
            max_row_sum = std::max(max_row_sum, std::abs(rowsum[i]));
        }
        row_sum_error = (sum_error_multiplier * std::numeric_limits<long double>::epsilon() *
                         upper + std::numeric_limits<double>::epsilon() * max_row_sum) *
                        (1 + 1e-10);
    }

    void tighten_bound() {
        Vec w(n, 1), z(n);
        for (int step = 0; step < 4; ++step) {
            double bound = 0, scale = 0;
            for (int i = 0; i < n; ++i) {
                double s = 0;
                for (uint32_t k = ptr[i]; k < ptr[i + 1]; ++k)
                    s += std::abs(val[k]) * w[col[k]];
                z[i] = s;
                if (w[i] > 0) bound = std::max(bound, s / w[i]);
                scale = std::max(scale, s);
            }
            upper = std::min(upper, bound * (1 + 1e-10));
            if (scale == 0) break;
            for (int i = 0; i < n; ++i) w[i] = std::max(z[i] / scale, 1e-100);
        }
    }

    void separate_diagonal() {
        uint32_t out = 0;
        for (int i = 0; i < n; ++i) {
            uint32_t begin = ptr[i], end = ptr[i + 1];
            ptr[i] = out;
            for (uint32_t k = begin; k < end; ++k) {
                if (col[k] == static_cast<uint32_t>(i) || val[k] == 0) continue;
                col[out] = col[k];
                val[out++] = val[k];
            }
        }
        ptr[n] = out;
        col.resize(out);
        val.resize(out);
        uniform_off = true;
        off_value = out ? val[0] : 0;
        for (double value : val)
            if (value != off_value) {
                uniform_off = false;
                break;
            }
        if (uniform_off) {
            row_sum_error = 0;
            for (int i = 0; i < n; ++i) {
                rowsum[i] = std::fma(static_cast<double>(ptr[i + 1] - ptr[i]),
                                    off_value, diagonal[i]);
                row_sum_error = std::max(row_sum_error, std::abs(rowsum[i]) *
                                        std::numeric_limits<double>::epsilon());
            }
            row_sum_error = row_sum_error * (1 + 1e-10) +
                            std::numeric_limits<double>::min();
        }
    }
};

struct Direct {
    int n;
    Vec e0, e1, f0, f1;
    std::vector<int> partner;
    bool active;

    Direct(const Matrix& a, const Metadata& h, Buffers& buffers)
        : n(a.n), active(a.pairs && (h.requests > 1 || a.n < 128 ||
                                    h.t * a.upper > 4)) {
        if (!active) return;
        double t = h.t;
        e0 = std::move(buffers.e0); e1 = std::move(buffers.e1);
        f0 = std::move(buffers.f0); f1 = std::move(buffers.f1);
        e0.resize(n); e1.resize(n); f0.resize(n); f1.resize(n);
        partner.assign(n, -1);
        for (int i = 0; i < n; ++i) {
            long double d = a.diagonal[i];
            int j = -1;
            long double b = 0;
            for (uint32_t k = a.ptr[i]; k < a.ptr[i + 1]; ++k)
                if (a.col[k] != static_cast<uint32_t>(i) && a.val[k] != 0) {
                    j = a.col[k]; b = a.val[k];
                }
            if (j < 0) {
                double x = static_cast<double>(t * d);
                e0[i] = std::exp(-x);
                f0[i] = t * phi_double(x);
                continue;
            }
            partner[i] = j;
            if (j < i) continue;
            long double q = a.diagonal[j];
            long double delta = (d - q) / 2;
            long double rad = std::hypot(delta, b);
            long double lp = (d + q) / 2 + rad;
            long double lm = lp == 0 ? 0 : (d * q - b * b) / lp;
            long double c = std::sqrt((rad + std::abs(delta)) / (2 * rad));
            long double s = b / (2 * rad * c);
            if (delta < 0) std::swap(c, s);
            double xp = static_cast<double>(t * lp), xm = static_cast<double>(t * lm);
            long double ep = std::exp(-xp), em = std::exp(-xm);
            long double fp = t * phi_double(xp), fm = t * phi_double(xm);
            e0[i] = static_cast<double>(ep * c * c + em * s * s);
            e0[j] = static_cast<double>(ep * s * s + em * c * c);
            e1[i] = e1[j] = static_cast<double>((ep - em) * c * s);
            f0[i] = static_cast<double>(fp * c * c + fm * s * s);
            f0[j] = static_cast<double>(fp * s * s + fm * c * c);
            f1[i] = f1[j] = static_cast<double>((fp - fm) * c * s);
        }
    }

    void solve(const Vec& b0, const Vec& b1, Vec& y) const {
        for (int i = 0; i < n; ++i) {
            y[i] = e0[i] * b0[i] + f0[i] * b1[i];
            int j = partner[i];
            if (j >= 0) y[i] += e1[i] * b0[j] + f1[i] * b1[j];
        }
    }
};

struct Chebyshev {
    const Matrix& a;
    double t, rtol, atol, factor;
    Vec ec, fc, etail, ftail;
    std::vector<long double> extended_e, extended_f;
    Vec d, delta, work;
    bool difference, uniform, metzler;
    double infinity_growth = 0;
    double min_row_sum = std::numeric_limits<double>::infinity();
    double max_row_sum = -std::numeric_limits<double>::infinity();
    int last_degree = 0;
    bool last_extended = false;

    Chebyshev(const Matrix& mat, const Metadata& h, Buffers& buffers)
        : a(mat), t(h.t), rtol(h.rtol), atol(h.atol),
          factor(-4 / mat.upper), d(std::move(buffers.d)),
          delta(std::move(buffers.delta)), work(std::move(buffers.work)),
          difference(true), uniform(mat.uniform_off && !std::getenv("FAM_GENERIC")) {
        d.resize(a.n); delta.resize(a.n); work.resize(a.n);
        // Difference form preserves near-constant nullspace components.
        int positive = 0, nonzero = 0;
        for (int i = 0; i < a.n; ++i)
            for (uint32_t j = a.ptr[i]; j < a.ptr[i + 1]; ++j)
                if (a.col[j] != static_cast<uint32_t>(i) && a.val[j] != 0) {
                    ++nonzero;
                    positive += a.val[j] > 0;
                }
        difference = positive * 20 < nonzero || nonzero == 0;
        metzler = positive == 0;
        for (double row : a.rowsum) {
            infinity_growth = std::max(infinity_growth, -row + a.row_sum_error);
            min_row_sum = std::min(min_row_sum, row);
            max_row_sum = std::max(max_row_sum, row);
        }
        long double alpha = static_cast<long double>(t) * a.upper / 2;
        int m = static_cast<int>(std::ceil(std::sqrt(140 * alpha) + 140));
        std::vector<long double> ratio(m + 2), e(m + 1), f(m + 1);
        for (int k = m; k > 0; --k)
            ratio[k] = alpha / (2 * k + alpha * ratio[k + 1]);
        e[0] = 1;
        long double sum = 1, product = 1;
        for (int k = 1; k <= m; ++k) {
            product *= ratio[k];
            e[k] = 2 * product;
            sum += e[k];
        }
        for (auto& v : e) v /= sum;
        long double tail = 0, weighted = 0;
        for (int k = m; k > 0; --k) {
            f[k] = 2 * weighted / alpha;
            tail += e[k];
            weighted += tail;
        }
        f[0] = weighted / alpha;
        ec.resize(m + 1); fc.resize(m + 1);
        extended_e.resize(m + 1); extended_f.resize(m + 1);
        etail.resize(m + 1); ftail.resize(m + 1);
        long double se = 0, sf = 0;
        for (int k = m; k >= 0; --k) {
            ec[k] = static_cast<double>(e[k]);
            fc[k] = static_cast<double>(t * f[k]);
            extended_e[k] = e[k];
            extended_f[k] = t * f[k];
            etail[k] = static_cast<double>(se);
            ftail[k] = static_cast<double>(sf);
            se += e[k];
            sf += t * f[k];
        }
    }

    template<bool Difference>
    __attribute__((optimize("no-associative-math")))
    void multiply_uniform(const double* __restrict x, double* __restrict z) const {
        for (int i = 0; i < a.n; ++i) {
            double s = 0, xi = x[i];
            for (uint32_t j = a.ptr[i]; j < a.ptr[i + 1]; ++j) {
                if (Difference) s += x[a.col[j]] - xi;
                else s += x[a.col[j]];
            }
            z[i] = (Difference ? a.rowsum[i] : a.diagonal[i]) * xi + a.off_value * s;
        }
    }

    void multiply(const double* __restrict x, double* __restrict z) const {
        if (a.col.empty()) {
            for (int i = 0; i < a.n; ++i)
                z[i] = a.diagonal[i] * x[i];
            return;
        }
        if (uniform) {
            if (difference) multiply_uniform<true>(x, z);
            else multiply_uniform<false>(x, z);
            return;
        }
        if (difference) {
            for (int i = 0; i < a.n; ++i) {
                double s = a.rowsum[i] * x[i];
                double xi = x[i];
                for (uint32_t j = a.ptr[i]; j < a.ptr[i + 1]; ++j)
                    s += a.val[j] * (x[a.col[j]] - xi);
                z[i] = s;
            }
        } else {
            for (int i = 0; i < a.n; ++i) {
                double s = a.diagonal[i] * x[i];
                for (uint32_t j = a.ptr[i]; j < a.ptr[i + 1]; ++j)
                    s += a.val[j] * x[a.col[j]];
                z[i] = s;
            }
        }
    }

    void multiply(const Vec& x, Vec& z) const {
        multiply(x.data(), z.data());
    }

    bool try_component_constants(const Vec& b0, const Vec& b1, Vec& y) const {
        if (!metzler || t * infinity_growth > .5) return false;
        for (int i = 0; i < a.n; ++i) {
            for (uint32_t k = a.ptr[i]; k < a.ptr[i + 1]; ++k) {
                int j = a.col[k];
                if (b0[i] != b0[j] || b1[i] != b1[j]) return false;
            }
        }
        // Edge equality makes both inputs component-constant. A scalar
        // surrogate is certified by its row-sum defect and a semigroup bound.
        double lambda = std::max(0.0, .5 * (min_row_sum + max_row_sum));
        double lower = std::max(0.0, min_row_sum - a.row_sum_error);
        double e = std::exp(-t * lambda), f = t * phi_double(t * lambda);
        double x = t * lower, damping = std::exp(-x);
        double moment;
        if (x < 1e-4)
            moment = .5 + x * (-1.0 / 3 + x * (1.0 / 8 + x * (-1.0 / 30 + x / 144)));
        else moment = (-std::expm1(-x) - x * damping) / (x * x);
        double r0 = 0, r1 = 0, size = 0;
        for (int i = 0; i < a.n; ++i) {
            double row = std::abs(a.rowsum[i] - lambda) + a.row_sum_error;
            r0 = std::max(r0, row * std::abs(b0[i]));
            r1 = std::max(r1, row * std::abs(b1[i]));
            y[i] = std::fma(e, b0[i], f * b1[i]);
            size = std::max(size, std::abs(y[i]));
        }
        double bound = std::exp(t * infinity_growth) *
                       (t * damping * r0 + t * t * moment * r1) * (1 + 1e-12);
        return bound <= .1 * (atol + rtol * size);
    }

    template<bool Initial, bool Forcing>
    void evaluate(const Vec& b0, const Vec& b1, Vec& y, int degree) {
        if (!degree) {
            for (int i = 0; i < a.n; ++i)
                y[i] = ec[0] * b0[i] + fc[0] * b1[i];
            return;
        }
        for (int i = 0; i < a.n; ++i) {
            double v = 0;
            if (Initial) v += ec[degree] * b0[i];
            if (Forcing) v += fc[degree] * b1[i];
            d[i] = delta[i] = v;
        }
        // Reinsch's endpoint form of Clenshaw avoids the large final
        // subtraction in the usual recurrence when t*lambda is small.
        for (int k = degree - 1; k >= 1; --k) {
            multiply(d, work);
            double c = ec[k], f = fc[k];
            for (int i = 0; i < a.n; ++i) {
                double v = factor * work[i];
                if (Initial) v += c * b0[i];
                if (Forcing) v += f * b1[i];
                delta[i] += v;
                d[i] += delta[i];
            }
        }
        multiply(d, work);
        for (int i = 0; i < a.n; ++i) {
            double v = delta[i] + (factor / 2) * work[i];
            if (Initial) v += ec[0] * b0[i];
            if (Forcing) v += fc[0] * b1[i];
            y[i] = v;
        }
    }

    template<bool Difference, bool Initial, bool Forcing>
    void evaluate_fused(const Vec& b0, const Vec& b1, Vec& y, int degree) {
        if (!degree) {
            for (int i = 0; i < a.n; ++i)
                y[i] = ec[0] * b0[i] + fc[0] * b1[i];
            return;
        }
        for (int i = 0; i < a.n; ++i) {
            double v = 0;
            if (Initial) v += ec[degree] * b0[i];
            if (Forcing) v += fc[degree] * b1[i];
            d[i] = delta[i] = v;
        }
        for (int k = degree - 1; k >= 1; --k) {
            double c = ec[k], f = fc[k];
            for (int i = 0; i < a.n; ++i) {
                double xi = d[i];
                double s = (Difference ? a.rowsum[i] : a.diagonal[i]) * xi;
                for (uint32_t j = a.ptr[i]; j < a.ptr[i + 1]; ++j) {
                    if (Difference) s += a.val[j] * (d[a.col[j]] - xi);
                    else s += a.val[j] * d[a.col[j]];
                }
                double v = factor * s;
                if (Initial) v += c * b0[i];
                if (Forcing) v += f * b1[i];
                delta[i] += v;
                work[i] = xi + delta[i];
            }
            d.swap(work);
        }
        for (int i = 0; i < a.n; ++i) {
            double xi = d[i];
            double s = (Difference ? a.rowsum[i] : a.diagonal[i]) * xi;
            for (uint32_t j = a.ptr[i]; j < a.ptr[i + 1]; ++j) {
                if (Difference) s += a.val[j] * (d[a.col[j]] - xi);
                else s += a.val[j] * d[a.col[j]];
            }
            double v = delta[i] + (factor / 2) * s;
            if (Initial) v += ec[0] * b0[i];
            if (Forcing) v += fc[0] * b1[i];
            y[i] = v;
        }
    }

    template<bool Difference>
    void evaluate_fused(const Vec& b0, const Vec& b1, Vec& y, int degree,
                        bool initial, bool forcing) {
        if (!initial) evaluate_fused<Difference, false, true>(b0, b1, y, degree);
        else if (!forcing) evaluate_fused<Difference, true, false>(b0, b1, y, degree);
        else evaluate_fused<Difference, true, true>(b0, b1, y, degree);
    }

    void evaluate_extended(const Vec& b0, const Vec& b1, Vec& y, int degree) {
        if (!degree) {
            for (int i = 0; i < a.n; ++i)
                y[i] = static_cast<double>(extended_e[0] * b0[i] +
                                           extended_f[0] * b1[i]);
            return;
        }
        std::vector<long double> v(a.n), change(a.n), next(a.n);
        long double scale = -4 / static_cast<long double>(a.upper);
        for (int i = 0; i < a.n; ++i)
            v[i] = change[i] = extended_e[degree] * b0[i] +
                               extended_f[degree] * b1[i];
        for (int k = degree - 1; k >= 1; --k) {
            for (int i = 0; i < a.n; ++i) {
                long double xi = v[i];
                long double s = (difference ? a.rowsum[i] : a.diagonal[i]) * xi;
                for (uint32_t j = a.ptr[i]; j < a.ptr[i + 1]; ++j) {
                    long double xj = v[a.col[j]];
                    s += static_cast<long double>(a.val[j]) *
                         (difference ? xj - xi : xj);
                }
                change[i] += scale * s + extended_e[k] * b0[i] +
                             extended_f[k] * b1[i];
                next[i] = xi + change[i];
            }
            v.swap(next);
        }
        for (int i = 0; i < a.n; ++i) {
            long double xi = v[i];
            long double s = (difference ? a.rowsum[i] : a.diagonal[i]) * xi;
            for (uint32_t j = a.ptr[i]; j < a.ptr[i + 1]; ++j) {
                long double xj = v[a.col[j]];
                s += static_cast<long double>(a.val[j]) * (difference ? xj - xi : xj);
            }
            y[i] = static_cast<double>(extended_e[0] * b0[i] +
                                      extended_f[0] * b1[i] +
                                      change[i] + (scale / 2) * s);
        }
    }

    void solve(const Vec& b0, const Vec& b1, Vec& y, bool force_extended = false) {
        last_extended = false;
        double nb0 = norm2(b0), nb1 = norm2(b1);
        if (nb0 == 0 && nb1 == 0) {
            std::fill(y.begin(), y.end(), 0);
            return;
        }
        double input_size = std::max(maxabs(b0), t * maxabs(b1));
        double target = 0.015 * (atol + rtol * input_size);
        int degree = 0;
        int maximum = static_cast<int>(ec.size()) - 1;
        if (force_extended && rtol < 1e-9) {
            target = .01 * atol;
            while (degree < maximum &&
                   etail[degree] * nb0 + ftail[degree] * nb1 > target) ++degree;
            evaluate_extended(b0, b1, y, degree);
            last_degree = degree;
            last_extended = true;
            return;
        }
        for (int retry = 0; retry < 3; ++retry) {
            while (degree < maximum &&
                   etail[degree] * nb0 + ftail[degree] * nb1 > target) ++degree;
            if (std::getenv("FAM_FUSED") && difference)
                evaluate_fused<true>(b0, b1, y, degree, nb0 != 0, nb1 != 0);
            else if (std::getenv("FAM_FUSED"))
                evaluate_fused<false>(b0, b1, y, degree, nb0 != 0, nb1 != 0);
            else if (nb0 == 0) evaluate<false, true>(b0, b1, y, degree);
            else if (nb1 == 0) evaluate<true, false>(b0, b1, y, degree);
            else evaluate<true, true>(b0, b1, y, degree);
            if ((rtol < 1e-9 && t * a.upper > 1000 &&
                 maxabs(y) < 1e-7 * input_size) || std::getenv("FAM_EXTENDED")) {
                target = .01 * (atol + rtol * maxabs(y));
                while (degree < maximum &&
                       etail[degree] * nb0 + ftail[degree] * nb1 > target) ++degree;
                evaluate_extended(b0, b1, y, degree);
                last_extended = true;
            }
            double bound = etail[degree] * nb0 + ftail[degree] * nb1;
            target = 0.08 * (atol + rtol * maxabs(y));
            if (bound <= target) break;
        }
        last_degree = degree;
    }
};

struct Lanczos {
    Chebyshev& op;
    const Matrix& a;
    const Metadata& h;
    std::unique_ptr<double[]> basis;
    Vec diag, off, work;
    int capacity, last_degree = 0;
    double last_bound = 0;
    bool needs_extended = false;

    Lanczos(Chebyshev& c, const Metadata& meta, int cap)
        : op(c), a(c.a), h(meta), work(a.n), capacity(cap) {
        basis.reset(new double[static_cast<size_t>(a.n) * (capacity + 1)]);
        diag.reserve(capacity);
        off.reserve(capacity);
    }

    double log_resolvent(int m, double shift, double& trace) const {
        // The end-to-end inverse entry is the off-diagonal product divided
        // by the determinant. LDL pivots evaluate it without cancellation.
        double pivot = diag[0] + shift;
        double deriv = 1;
        if (!(pivot > 0)) { trace = 1e300; return 1e300; }
        double result = -std::log(pivot);
        trace = 1 / pivot;
        for (int j = 1; j < m; ++j) {
            double ratio = off[j - 1] / pivot;
            double next = diag[j] + shift - ratio * off[j - 1];
            deriv = 1 + ratio * ratio * deriv;
            if (!(next > 0)) { trace = 1e300; return 1e300; }
            result += std::log(off[j - 1]) - std::log(next);
            trace += deriv / next;
            pivot = next;
        }
        return result;
    }

    double bound(int m, double beta, double norm, bool is_phi) const {
        if (beta == 0) return 0;
        double lo = is_phi ? 1.0 : 0.0;
        double hi = 64, tr;
        for (int step = 0; step < 22; ++step) {
            double shift = (lo + hi) * .5;
            log_resolvent(m, shift, tr);
            double derivative = (is_phi ? 1 - 1 / shift : 1) - tr;
            if (derivative < 0) lo = shift;
            else hi = shift;
        }
        double shift = (lo + hi) * .5;
        double logg = log_resolvent(m, shift, tr);
        double weight = is_phi ? shift - 1 - std::log(shift) : shift;
        return std::exp(logg + weight + std::log(beta) + std::log(norm));
    }

    bool project(int m, double norm, bool is_phi, Vec& y) {
        int degree = 0, max_degree = static_cast<int>(op.ec.size()) - 1;
        double threshold = std::max(h.atol * .001 / norm, 1e-24);
        while (degree < max_degree &&
               (is_phi ? op.ftail[degree] / h.t : op.etail[degree]) > threshold)
            ++degree;
        auto coefficient = [&](int k) {
            return is_phi ? op.extended_f[k] / h.t : op.extended_e[k];
        };
        if (degree == 0) {
            double scale = static_cast<double>(norm * coefficient(0));
            for (int i = 0; i < a.n; ++i) y[i] = scale * basis[i];
            return true;
        }
        std::vector<long double> d(m), delta(m), next(m);
        d[0] = delta[0] = coefficient(degree);
        long double factor = -4 / (static_cast<long double>(h.t) * a.upper);
        for (int k = degree - 1; k >= 1; --k) {
            for (int i = 0; i < m; ++i) {
                long double z = diag[i] * d[i];
                if (i) z += off[i - 1] * d[i - 1];
                if (i + 1 < m) z += off[i] * d[i + 1];
                delta[i] += factor * z;
                if (i == 0) delta[i] += coefficient(k);
                next[i] = d[i] + delta[i];
            }
            d.swap(next);
        }
        Vec coeff(m);
        for (int i = 0; i < m; ++i) {
            long double z = diag[i] * d[i];
            if (i) z += off[i - 1] * d[i - 1];
            if (i + 1 < m) z += off[i] * d[i + 1];
            long double v = delta[i] + (factor / 2) * z;
            if (i == 0) v += coefficient(0);
            coeff[i] = static_cast<double>(norm * v);
        }
        basis_product(basis.get(), coeff, y, a.n);
        return true;
    }

    bool solve(const Vec& b, Vec& y, bool is_phi, double absolute_target = 0,
               double seconds = 2.0) {
        auto started = std::chrono::steady_clock::now();
        double input_norm = std::sqrt(accurate_dot(b.data(), b.data(), a.n));
        if (input_norm == 0) {
            std::fill(y.begin(), y.end(), 0);
            last_bound = 0;
            last_degree = 0;
            return true;
        }
        double output_scale = is_phi ? h.t : 1;
        double norm = input_norm * output_scale;
        double target = absolute_target > 0 ? absolute_target :
            .01 * (h.atol + h.rtol * maxabs(b) * output_scale);
        diag.clear(); off.clear();
        double* q = basis.get();
        for (int i = 0; i < a.n; ++i) q[i] = b[i] / input_norm;
        double previous_beta = 0;
        for (int k = 0; k < capacity; ++k) {
            q = basis.get() + static_cast<size_t>(k) * a.n;
            const double* previous = k ? q - a.n : nullptr;
            op.multiply(q, work.data());
            for (int i = 0; i < a.n; ++i) work[i] *= h.t;
            double alpha = accurate_dot(q, work.data(), a.n);
            if (k == 0) {
                for (int i = 0; i < a.n; ++i) work[i] -= alpha * q[i];
            } else {
                for (int i = 0; i < a.n; ++i)
                    work[i] -= alpha * q[i] + previous_beta * previous[i];
            }
            double beta = std::sqrt(accurate_dot(work.data(), work.data(), a.n));
            diag.push_back(alpha);
            if (k) off.push_back(previous_beta);
            int m = k + 1;
            bool breakdown = beta <= 2e-15 * std::max(1.0, h.t * a.upper);
            if (m % 8 == 0 || breakdown || m == capacity) {
                double residual = beta;
                if (h.t * a.logarithmic_growth < 10)
                    residual = std::min(residual, maxabs(work) *
                                       std::exp(h.t * a.logarithmic_growth));
                double error = bound(m, residual, norm, is_phi);
                if (error <= target || beta == 0) {
                    if (!project(m, norm, is_phi, y)) return false;
                    target = absolute_target > 0 ? absolute_target :
                        .03 * (h.atol + h.rtol * maxabs(y));
                    if (error <= target || beta == 0) {
                        last_degree = m;
                        last_bound = error;
                        return true;
                    }
                }
                double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                if (elapsed > seconds) {
                    last_degree = m;
                    return false;
                }
            }
            double* next = q + a.n;
            for (int i = 0; i < a.n; ++i) next[i] = work[i] / beta;
            previous_beta = beta;
        }
        last_degree = capacity;
        return false;
    }

    bool solve_pair(const Vec& b0, const Vec& b1, Vec& y) {
        if (!solve(b0, y, false, 0, 1.0)) return false;
        double first_bound = last_bound, first_size = maxabs(y);
        int first_degree = last_degree;
        Vec forcing(a.n);
        if (!solve(b1, forcing, true, 0, 1.0)) return false;
        double second_size = maxabs(forcing);
        for (int i = 0; i < a.n; ++i) y[i] += forcing[i];
        double size = maxabs(y);
        if (h.rtol < 1e-9 && size < 1e-4 * (first_size + second_size)) {
            double input_size = std::max(maxabs(b0), h.t * maxabs(b1));
            needs_extended = h.t * a.upper > 1000 &&
                size + first_bound + last_bound < 1e-8 * input_size;
            return false;
        }
        double target = .06 * (h.atol + h.rtol * size);
        if (first_bound + last_bound > target) {
            if (!solve(b0, y, false, target / 4, 1.0)) return false;
            first_bound = last_bound;
            first_degree = last_degree;
            if (!solve(b1, forcing, true, target / 4, 1.0)) return false;
            for (int i = 0; i < a.n; ++i) y[i] += forcing[i];
        }
        last_bound += first_bound;
        last_degree += first_degree;
        return true;
    }
};

int main() {
    try {
        // Fixed-capacity workspace is independent of the unrevealed session.
        int allocation = std::getenv("FAM_NO_PREALLOC") ? 0 :
                         std::getenv("FAM_PREALLOC_FULL") ? 2 : 1;
        Buffers buffers(allocation);
        write_all("READY\n", 6);
        char magic[8];
        read_all(magic, 8);
        if (std::memcmp(magic, "FAMEXP2\n", 8)) throw std::runtime_error("bad magic");
        uint32_t length;
        read_all(&length, sizeof(length));
        if (length > 4096) throw std::runtime_error("bad header length");
        std::string json(length, '\0');
        read_all(json.data(), length);
        Metadata h = parse_metadata(json);
        Matrix a(h, buffers);
        Direct direct(a, h, buffers);
        if (!direct.active && h.t * a.upper > 512) a.balance_signs();
        if (!direct.active && h.t * a.upper > 200) a.tighten_bound();
        bool trivial = h.t == 0 || a.upper == 0;
        Chebyshev* cheb = nullptr;
        if (!direct.active && !trivial) {
            a.separate_diagonal();
            cheb = new Chebyshev(a, h, buffers);
        }
        std::unique_ptr<Lanczos> krylov;
        const char* method = std::getenv("FAM_METHOD");
        double mean_diagonal = 0, max_diagonal = 0;
        for (double v : a.diagonal) {
            mean_diagonal += v / h.n;
            max_diagonal = std::max(max_diagonal, v);
        }
        bool clustered = a.unbalanced || max_diagonal > 1.8 * mean_diagonal ||
                         h.t * a.logarithmic_measure < -32;
        bool probe = h.t * a.upper > 2000 && a.mean_distance > 4 * std::sqrt(h.n);
        bool use_krylov = cheb && ((h.t * a.upper > 512 && clustered) || probe);
        if (method) use_krylov = cheb && std::strcmp(method, "lanczos") == 0;
        double exponential_bound = std::exp(std::min(
            .5 * std::log(static_cast<double>(h.n)), h.t * a.logarithmic_measure));
        std::array<int, 3> krylov_limits{};
        Vec b0 = std::move(buffers.b0), b1 = std::move(buffers.b1),
            y = std::move(buffers.y);
        b0.resize(h.n); b1.resize(h.n); y.resize(h.n);
        for (int request = 0; request < h.requests; ++request) {
            read_all(b0.data(), b0.size() * sizeof(double));
            read_all(b1.data(), b1.size() * sizeof(double));
            if (krylov) krylov->needs_extended = false;
            if (!a.signs.empty()) {
                for (int i = 0; i < h.n; ++i) {
                    b0[i] *= a.signs[i];
                    b1[i] *= a.signs[i];
                }
            }
            if (trivial) {
                for (int i = 0; i < h.n; ++i) y[i] = b0[i] + h.t * b1[i];
            } else if (direct.active) direct.solve(b0, b1, y);
            else {
                bool done = false;
                double initial_size = maxabs(b0), forcing_size = maxabs(b1);
                bool zero_initial = initial_size == 0, zero_forcing = forcing_size == 0;
                if (!zero_initial && exponential_bound * initial_size <= .01 * h.atol) {
                    std::fill(b0.begin(), b0.end(), 0);
                    zero_initial = true;
                }
                if (zero_initial && zero_forcing) {
                    std::fill(y.begin(), y.end(), 0);
                    done = true;
                }
                if (!done) done = cheb->try_component_constants(b0, b1, y);
                bool pair_allowed = a.unbalanced || h.t * a.upper > 10000 || probe || method;
                if (!done && use_krylov && (zero_initial || zero_forcing || pair_allowed)) {
                    if (!krylov) {
                        int cap = std::min(1200, static_cast<int>(.9 * std::sqrt(
                            h.t * a.upper * (10 - std::log(h.rtol)))));
                        cap = std::max(32, cap);
                        if (!clustered) cap = std::max(16, std::min(cap / 4, 80));
                        if (method) cap = 2500;
                        cap = std::min(cap, static_cast<int>(1500000000ULL / (8ULL * h.n)));
                        krylov.reset(new Lanczos(*cheb, h, cap));
                        krylov_limits.fill(cap);
                    }
                    int kind = zero_initial ? 1 : zero_forcing ? 0 : 2;
                    krylov->capacity = krylov_limits[kind];
                    if (zero_initial) done = krylov->solve(b1, y, true);
                    else if (zero_forcing) done = krylov->solve(b0, y, false);
                    else done = krylov->solve_pair(b0, b1, y);
                    if (!done && h.requests > 1 && !method && !krylov->needs_extended)
                        krylov_limits[kind] = std::min(krylov_limits[kind], 64);
                }
                if (!done) cheb->solve(b0, b1, y, krylov && krylov->needs_extended);
            }
            if (!a.signs.empty())
                for (int i = 0; i < h.n; ++i) y[i] *= a.signs[i];
            write_all(y.data(), y.size() * sizeof(double));
        }
        if (std::getenv("FAM_DEBUG") && cheb)
            std::fprintf(stderr, "n=%d eta=%.9g degree=%d difference=%d extended=%d lanczos=%d bound=%.3g\n",
                         h.n, h.t * a.upper, cheb->last_degree, cheb->difference,
                         cheb->last_extended,
                         krylov ? krylov->last_degree : 0,
                         krylov ? krylov->last_bound : 0);
        if (!std::getenv("FAM_REGULAR_EXIT")) std::_Exit(0);
        delete cheb;
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
