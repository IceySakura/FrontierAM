#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using LD = long double;

void read_exact(void* destination, size_t size) {
    auto* p = static_cast<unsigned char*>(destination);
    while (size) {
        const size_t got = std::fread(p, 1, size, stdin);
        if (!got) throw std::runtime_error("unexpected end of input");
        p += got;
        size -= got;
    }
}

void write_exact(const void* source, size_t size) {
    const auto* p = static_cast<const unsigned char*>(source);
    while (size) {
        const size_t sent = std::fwrite(p, 1, size, stdout);
        if (!sent) throw std::runtime_error("output failure");
        p += sent;
        size -= sent;
    }
    std::fflush(stdout);
}

class Header {
    std::unordered_map<std::string, std::string> values_;

    static void space(const std::string& text, size_t& p) {
        while (p < text.size() && (text[p] == ' ' || text[p] == '\n' ||
                                  text[p] == '\r' || text[p] == '\t')) ++p;
    }

    static std::string string(const std::string& text, size_t& p) {
        if (p == text.size() || text[p++] != '"')
            throw std::runtime_error("invalid JSON string");
        std::string result;
        while (p < text.size()) {
            char c = text[p++];
            if (c == '"') return result;
            if (c == '\\') {
                if (p == text.size()) throw std::runtime_error("invalid JSON escape");
                c = text[p++];
                if (c == 'n') c = '\n';
                else if (c == 'r') c = '\r';
                else if (c == 't') c = '\t';
                else if (c != '"' && c != '\\' && c != '/')
                    throw std::runtime_error("unsupported JSON escape");
            }
            result.push_back(c);
        }
        throw std::runtime_error("unterminated JSON string");
    }

public:
    explicit Header(const std::string& text) {
        size_t p = 0;
        space(text, p);
        if (p == text.size() || text[p++] != '{')
            throw std::runtime_error("invalid JSON header");
        for (;;) {
            space(text, p);
            if (p < text.size() && text[p] == '}') break;
            const std::string key = string(text, p);
            space(text, p);
            if (p == text.size() || text[p++] != ':')
                throw std::runtime_error("invalid JSON member");
            space(text, p);
            std::string value;
            if (p < text.size() && text[p] == '"') {
                value = string(text, p);
            } else {
                const size_t begin = p;
                while (p < text.size() && text[p] != ',' && text[p] != '}') ++p;
                value = text.substr(begin, p - begin);
            }
            values_.emplace(key, value);
            space(text, p);
            if (p < text.size() && text[p] == ',') {
                ++p;
                continue;
            }
            if (p < text.size() && text[p] == '}') break;
            throw std::runtime_error("invalid JSON object");
        }
    }

    double number(const char* name) const {
        return std::stod(values_.at(name));
    }
};

struct Matrix {
    uint32_t n = 0;
    std::vector<uint64_t> ptr;
    std::vector<uint32_t> col;
    std::vector<double> val;
    std::vector<double> diagonal;
    std::vector<int32_t> partner;
    std::vector<double> partner_value;
    std::vector<uint32_t> off_ptr, off_col;
    std::vector<double> off_weight, potential;
    double bound = 0;
    bool pairs_only = true;
    bool dominant = true, negative_only = true;

    void analyze() {
        diagonal.assign(n, 0);
        partner.assign(n, -1);
        partner_value.assign(n, 0);
        LD maximum = 0;
        for (uint32_t i = 0; i < n; ++i) {
            LD sum = 0;
            LD off_sum = 0;
            unsigned off_count = 0;
            for (uint64_t j = ptr[i]; j < ptr[i + 1]; ++j) {
                sum += std::abs(static_cast<LD>(val[j]));
                if (col[j] == i) diagonal[i] = val[j];
                else if (val[j] != 0) {
                    off_sum += std::abs(static_cast<LD>(val[j]));
                    if (val[j] > 0) negative_only = false;
                    ++off_count;
                    partner[i] = static_cast<int32_t>(col[j]);
                    partner_value[i] = val[j];
                }
            }
            maximum = std::max(maximum, sum);
            if (off_count > 1) pairs_only = false;
            if (static_cast<LD>(diagonal[i]) < off_sum * (1 - 1e-13L))
                dominant = false;
        }
        bound = static_cast<double>(maximum * (1 + 8e-16L));
    }

    void prepare_edges() {
        if (!dominant || pairs_only || bound == 0) return;
        off_ptr.resize(n + 1);
        potential.resize(n);
        off_col.reserve(val.size());
        off_weight.reserve(val.size());
        const LD scale = 4.0L / bound;
        for (uint32_t i = 0; i < n; ++i) {
            off_ptr[i] = static_cast<uint32_t>(off_col.size());
            LD sum = 0;
            for (uint64_t j = ptr[i]; j < ptr[i + 1]; ++j) {
                if (col[j] == i || val[j] == 0) continue;
                const LD weight = std::abs(static_cast<LD>(val[j]));
                sum += weight;
                off_col.push_back(col[j] | (val[j] > 0 ? 0x80000000U : 0U));
                off_weight.push_back(static_cast<double>(scale * weight));
            }
            potential[i] = static_cast<double>(scale * (diagonal[i] - sum));
        }
        off_ptr[n] = static_cast<uint32_t>(off_col.size());
    }

    template<bool Signed>
    double edge_dot(uint32_t row, const double* vector) const {
        const double center = vector[row];
        double s0 = potential[row] * center, s1 = 0;
        uint32_t j = off_ptr[row], end = off_ptr[row + 1];
        for (; j + 1 < end; j += 2) {
            double v0, v1;
            if constexpr (Signed) {
                uint64_t bits0, bits1;
                std::memcpy(&bits0, vector + (off_col[j] & 0x7fffffffU), 8);
                std::memcpy(&bits1, vector + (off_col[j + 1] & 0x7fffffffU), 8);
                bits0 ^= static_cast<uint64_t>(off_col[j] & 0x80000000U) << 32;
                bits1 ^= static_cast<uint64_t>(off_col[j + 1] & 0x80000000U) << 32;
                std::memcpy(&v0, &bits0, 8);
                std::memcpy(&v1, &bits1, 8);
            } else {
                v0 = vector[off_col[j]];
                v1 = vector[off_col[j + 1]];
            }
            s0 += off_weight[j] * (center - v0);
            s1 += off_weight[j + 1] * (center - v1);
        }
        if (j < end) {
            double v;
            if constexpr (Signed) {
                uint64_t bits;
                std::memcpy(&bits, vector + (off_col[j] & 0x7fffffffU), 8);
                bits ^= static_cast<uint64_t>(off_col[j] & 0x80000000U) << 32;
                std::memcpy(&v, &bits, 8);
            } else {
                v = vector[off_col[j]];
            }
            s0 += off_weight[j] * (center - v);
        }
        return s0 + s1;
    }

    void tighten_bound(unsigned iterations) {
        std::vector<double> v(n, 1), w(n);
        for (unsigned it = 0; it < iterations; ++it) {
            LD upper = 0;
            double maximum = 0;
            for (uint32_t i = 0; i < n; ++i) {
                LD s = 0;
                for (uint64_t j = ptr[i]; j < ptr[i + 1]; ++j)
                    s += std::abs(static_cast<LD>(val[j])) * v[col[j]];
                w[i] = static_cast<double>(s);
                maximum = std::max(maximum, w[i]);
                upper = std::max(upper, s / v[i]);
            }
            bound = std::min(bound, static_cast<double>(upper * (1 + 4e-15L)));
            if (maximum == 0) return;
            for (uint32_t i = 0; i < n; ++i)
                v[i] = std::max(w[i] / maximum, 1e-100);
        }
    }

    template<class Real, class Storage = Real>
    Real dot(uint32_t row, const Storage* vector) const {
        uint64_t j = ptr[row], end = ptr[row + 1];
        Real s0 = 0, s1 = 0;
        for (; j + 1 < end; j += 2) {
            s0 += static_cast<Real>(val[j]) * static_cast<Real>(vector[col[j]]);
            s1 += static_cast<Real>(val[j + 1]) * static_cast<Real>(vector[col[j + 1]]);
        }
        if (j < end) s0 += static_cast<Real>(val[j]) * static_cast<Real>(vector[col[j]]);
        return s0 + s1;
    }
};

LD exponential(LD x) {
    return std::exp(-x);
}

LD integrated_exponential(LD x, LD t) {
    return x == 0 ? t : t * (-std::expm1(-x) / x);
}

struct LocalFunctions {
    std::vector<std::array<LD, 4>> entries;

    static double exp_value(LD x) {
        return std::exp(-static_cast<double>(x));
    }

    static double phi_value(LD x, double t) {
        return t * exp_value(x);
    }

    void prepare(const Matrix& a, double time) {
        entries.resize(a.n);
        for (uint32_t i = 0; i < a.n; ++i) {
            const int32_t other = a.partner[i];
            if (other < 0) {
                const LD x = static_cast<LD>(time) * a.diagonal[i];
                entries[i] = {exp_value(x), phi_value(x, time), 0, 0};
                continue;
            }
            if (static_cast<uint32_t>(other) < i) continue;
            const uint32_t j = static_cast<uint32_t>(other);
            const LD aa = a.diagonal[i], dd = a.diagonal[j], bb = a.partner_value[i];
            const LD center = (aa + dd) / 2;
            const LD half_difference = (aa - dd) / 2;
            const LD radius = std::sqrt(half_difference * half_difference + bb * bb);
            const LD high = center + radius;
            const double ad = a.diagonal[i] * a.diagonal[j];
            const double b2 = a.partner_value[i] * a.partner_value[i];
            const LD determinant = (static_cast<LD>(ad) - b2) +
                (static_cast<LD>(std::fma(a.diagonal[i], a.diagonal[j], -ad)) -
                 std::fma(a.partner_value[i], a.partner_value[i], -b2));
            const LD low = high == 0 ? 0 : std::max(0.0L, determinant / high);
            const LD eh = exp_value(time * high), el = exp_value(time * low);
            const LD ph = phi_value(time * high, time);
            const LD pl = phi_value(time * low, time);
            const LD gap = high - low;
            const LD p00 = gap == 0 ? 1 : (aa - low) / gap;
            const LD p11 = gap == 0 ? 1 : (dd - low) / gap;
            const LD p01 = gap == 0 ? 0 : bb / gap;
            const LD de = eh - el, dp = ph - pl;
            entries[i] = {el + de * p00, pl + dp * p00, de * p01, dp * p01};
            entries[j] = {el + de * p11, pl + dp * p11, de * p01, dp * p01};
        }
    }

    void apply(const Matrix& a, const std::vector<double>& b0,
               const std::vector<double>& b1, std::vector<double>& y) const {
        for (uint32_t i = 0; i < a.n; ++i) {
            const auto& f = entries[i];
            LD s = f[0] * b0[i] + f[1] * b1[i];
            if (a.partner[i] >= 0) {
                const uint32_t j = static_cast<uint32_t>(a.partner[i]);
                s += f[2] * b0[j] + f[3] * b1[j];
            }
            y[i] = static_cast<double>(s);
        }
    }
};

struct Coefficients {
    std::vector<LD> exp, phi, exp_tail, phi_tail;

    explicit Coefficients(LD c) {
        const unsigned m = static_cast<unsigned>(
            std::ceil(std::sqrt(2 * c * 64) + 2 * 64 + 32));
        std::vector<LD> ratio(m + 2, 0);
        for (unsigned k = m; k >= 1; --k)
            ratio[k] = 1 / (2 * static_cast<LD>(k) / c + ratio[k + 1]);
        exp.assign(m + 1, 0);
        exp[0] = 1;
        LD normalizer = 1, product = 1;
        for (unsigned k = 1; k <= m; ++k) {
            product *= ratio[k];
            exp[k] = 2 * product;
            normalizer += exp[k];
        }
        for (auto& v : exp) v /= normalizer;
        phi.assign(m + 1, 0);
        LD slope = 0;
        for (unsigned k = m; k >= 2; --k) {
            slope += 2 * exp[k] / c;
            phi[k - 1] = phi[k] + slope;
        }
        phi[0] = (phi[1] + slope + 2 * exp[1] / c) / 2;
        normalizer = 0;
        for (const auto v : phi) normalizer += v;
        for (auto& v : phi) v /= normalizer;
        phi = exp;
        exp_tail.assign(m + 1, 0);
        phi_tail.assign(m + 1, 0);
        for (unsigned k = m; k >= 1; --k) {
            exp_tail[k - 1] = exp_tail[k] + exp[k];
            phi_tail[k - 1] = phi_tail[k] + phi[k];
        }
    }

    unsigned degree(LD n0, LD n1, LD target) const {
        for (unsigned k = 0; k < exp.size(); ++k)
            if (n0 * exp_tail[k] + n1 * phi_tail[k] <= target) return k;
        return static_cast<unsigned>(exp.size() - 1);
    }
};

template<class Real, bool Edges = false, bool Signed = false, class Storage = Real>
void clenshaw(const Matrix& a, const Coefficients& coef, unsigned degree, double t,
              const std::vector<double>& b0, const std::vector<double>& b1,
              std::vector<double>& output) {
    if (degree == 0) {
        for (uint32_t i = 0; i < a.n; ++i)
            output[i] = static_cast<double>(coef.exp[0] * b0[i] + t * coef.phi[0] * b1[i]);
        return;
    }
    std::vector<Storage> w(a.n), next(a.n);
    std::vector<Real> difference(a.n);
    const Real scale = static_cast<Real>(4.0L / a.bound);
    Real e = static_cast<Real>(coef.exp[degree]);
    Real p = static_cast<Real>(t * coef.phi[degree]);
    for (uint32_t i = 0; i < a.n; ++i) {
        difference[i] = e * static_cast<Real>(b0[i]) + p * static_cast<Real>(b1[i]);
        w[i] = static_cast<Storage>(difference[i]);
    }

    // Reinsch's form avoids subtracting two large Clenshaw vectors at x = 1.
    for (unsigned k = degree - 1; k >= 1; --k) {
        e = static_cast<Real>(coef.exp[k]);
        p = static_cast<Real>(t * coef.phi[k]);
        for (uint32_t i = 0; i < a.n; ++i) {
            Real product;
            if constexpr (Edges) product = a.edge_dot<Signed>(i, w.data());
            else product = scale * a.dot<Real>(i, w.data());
            const Real forcing = e * static_cast<Real>(b0[i]) + p * static_cast<Real>(b1[i]);
            difference[i] += forcing - product;
            next[i] = static_cast<Storage>(static_cast<Real>(w[i]) + difference[i]);
        }
        w.swap(next);
    }
    e = static_cast<Real>(coef.exp[0]);
    p = static_cast<Real>(t * coef.phi[0]);
    for (uint32_t i = 0; i < a.n; ++i) {
        Real product;
        if constexpr (Edges) product = a.edge_dot<Signed>(i, w.data());
        else product = scale * a.dot<Real>(i, w.data());
        output[i] = static_cast<double>(difference[i] - product / 2 +
            e * static_cast<Real>(b0[i]) + p * static_cast<Real>(b1[i]));
    }
}

void evaluate(const Matrix& a, const Coefficients& coefficients, unsigned degree, double time,
              const std::vector<double>& b0, const std::vector<double>& b1,
              std::vector<double>& y, int extended, bool edges) {
    if (extended == 2)
        clenshaw<LD, false, false, double>(a, coefficients, degree, time, b0, b1, y);
    else if (extended) clenshaw<LD>(a, coefficients, degree, time, b0, b1, y);
    else if (edges && a.negative_only)
        clenshaw<double, true, false>(a, coefficients, degree, time, b0, b1, y);
    else if (edges)
        clenshaw<double, true, true>(a, coefficients, degree, time, b0, b1, y);
    else clenshaw<double>(a, coefficients, degree, time, b0, b1, y);
}

LD rayleigh(const Matrix& a, const std::vector<double>& b, LD square_norm) {
    if (square_norm == 0) return 0;
    LD result = 0;
    for (uint32_t i = 0; i < a.n; ++i)
        result += static_cast<LD>(b[i]) * a.dot<double>(i, b.data());
    return std::max(0.0L, result / square_norm);
}

double maximum_abs(const std::vector<double>& v) {
    double result = 0;
    for (const double x : v) result = std::max(result, std::abs(x));
    return result;
}

LD norm2(const std::vector<double>& v) {
    LD sum = 0;
    for (const double x : v) sum += static_cast<LD>(x) * x;
    return std::sqrt(sum);
}

int run() {
    const bool debug = std::getenv("FAMEXP_DEBUG") != nullptr;
    std::setvbuf(stdin, nullptr, _IOFBF, 65536);
    std::setvbuf(stdout, nullptr, _IOFBF, 65536);
    write_exact("READY\n", 6);
    char magic[8];
    read_exact(magic, sizeof(magic));
    if (std::memcmp(magic, "FAMEXP2\n", 8) != 0)
        throw std::runtime_error("invalid protocol magic");
    uint32_t header_size;
    read_exact(&header_size, sizeof(header_size));
    if (header_size > 4096) throw std::runtime_error("oversized header");
    std::string text(header_size, '\0');
    read_exact(text.data(), text.size());
    const Header header(text);
    Matrix a;
    a.n = static_cast<uint32_t>(header.number("n"));
    const size_t nnz = static_cast<size_t>(header.number("nnz"));
    const unsigned requests = static_cast<unsigned>(header.number("requests"));
    const double time = header.number("t");
    const double rtol = header.number("rtol"), atol = header.number("atol");
    a.ptr.resize(a.n + 1);
    a.col.resize(nnz);
    a.val.resize(nnz);
    read_exact(a.ptr.data(), a.ptr.size() * sizeof(uint64_t));
    read_exact(a.col.data(), nnz * sizeof(uint32_t));
    read_exact(a.val.data(), nnz * sizeof(double));
    a.analyze();
    LocalFunctions local;
    if (a.pairs_only && time != 0 && a.bound != 0) local.prepare(a, time);
    if (!a.pairs_only && time * a.bound > 200) a.tighten_bound(6);
    const LD eta = static_cast<LD>(time) * a.bound;
    if (rtol < 1e-9 && eta > 100) a.prepare_edges();
    Coefficients coefficients(std::max(eta / 2, 1e-300L));
    std::vector<double> b0(a.n), b1(a.n), y(a.n);
    for (unsigned request = 0; request < requests; ++request) {
        read_exact(b0.data(), a.n * sizeof(double));
        read_exact(b1.data(), a.n * sizeof(double));
        unsigned degree = 0;
        if (time == 0) y = b0;
        else if (a.bound == 0) {
            for (uint32_t i = 0; i < a.n; ++i) y[i] = b0[i] + time * b1[i];
        } else if (a.pairs_only) {
            local.apply(a, b0, b1, y);
        } else {
            const LD n0 = norm2(b0), b1norm = norm2(b1), n1 = time * b1norm;
            const double b0max = maximum_abs(b0), b1max = maximum_abs(b1);
            LD scale = b0max + time * b1max;
            bool estimate = true;
            if (const char* setting = std::getenv("FAMEXP_ESTIMATE"))
                estimate = std::atoi(setting) != 0;
            if (estimate && eta > 100) {
                const LD r0 = time * rayleigh(a, b0, n0 * n0);
                const LD r1 = time * rayleigh(a, b1, b1norm * b1norm);
                scale = b0max * exponential(r0) +
                        b1max * integrated_exponential(r1, time);
            }
            LD goal = 0.015L * (atol + rtol * scale);
            degree = coefficients.degree(n0, n1, goal);
            bool edges = !a.off_ptr.empty();
            if (const char* setting = std::getenv("FAMEXP_EDGES"))
                edges = !a.off_ptr.empty() && std::atoi(setting) != 0;
            int extended = (rtol < 1e-9 && eta > 100 && !edges) ? 1 : 0;
            if (const char* mode = std::getenv("FAMEXP_PRECISION"))
                extended = std::atoi(mode);
            evaluate(a, coefficients, degree, time, b0, b1, y, extended, edges);
            const LD tail = n0 * coefficients.exp_tail[degree] +
                            n1 * coefficients.phi_tail[degree];
            goal = 0.1L * (atol + rtol * maximum_abs(y));
            if (tail > goal) {
                degree = coefficients.degree(n0, n1, goal / 4);
                evaluate(a, coefficients, degree, time, b0, b1, y, extended, edges);
            }
            if (!extended && rtol < 1e-9 && eta > 1000 && maximum_abs(y) < 0.01 * b0max) {
                evaluate(a, coefficients, degree, time, b0, b1, y, true, false);
            }
        }
        if (debug) std::fprintf(stderr, "request=%u n=%u eta=%.9Lg degree=%u local=%d\n",
                                request, a.n, eta, degree, a.pairs_only);
        write_exact(y.data(), a.n * sizeof(double));
    }
    return 0;
}

} // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "solver: %s\n", error.what());
        return 1;
    }
}
