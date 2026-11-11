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
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>
#include <sys/mman.h>
#include <cblas.h>
#include <Eigen/Eigenvalues>

using std::size_t;
using Vec = std::vector<double>;

static void read_exact(void* buffer, size_t bytes) {
    char* p = static_cast<char*>(buffer);
    while (bytes) {
        ssize_t count = ::read(STDIN_FILENO, p, bytes);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error("short input");
        p += count;
        bytes -= static_cast<size_t>(count);
    }
}

static void write_exact(const void* buffer, size_t bytes) {
    const char* p = static_cast<const char*>(buffer);
    while (bytes) {
        ssize_t count = ::write(STDOUT_FILENO, p, bytes);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error("output failure");
        p += count;
        bytes -= static_cast<size_t>(count);
    }
}

static void prefer_huge_pages(void* pointer, size_t bytes, size_t minimum = 16 * 1024 * 1024) {
    if (!pointer || bytes < minimum || std::getenv("FAM_NO_HUGEPAGES")) return;
    long page = ::sysconf(_SC_PAGESIZE);
    if (page <= 0) return;
    std::uintptr_t address = reinterpret_cast<std::uintptr_t>(pointer);
    std::uintptr_t begin = (address + page - 1) / page * page;
    std::uintptr_t end = (address + bytes) / page * page;
    if (end > begin) ::madvise(reinterpret_cast<void*>(begin), end - begin, MADV_HUGEPAGE);
}

// The wire header is a flat JSON object, with numeric and string values.
struct Header {
    int n = 0, nnz = 0, requests = 0;
    double t = 0, rtol = 0, atol = 0;

    explicit Header(const std::string& text) {
        size_t pos = 0;
        auto space = [&]() {
            while (pos < text.size() && static_cast<unsigned char>(text[pos]) <= 32) ++pos;
        };
        auto quoted = [&]() {
            if (text.at(pos++) != '"') throw std::runtime_error("JSON string expected");
            std::string out;
            while (pos < text.size()) {
                char c = text[pos++];
                if (c == '"') return out;
                if (c == '\\') {
                    c = text.at(pos++);
                    if (c != '\\' && c != '"' && c != '/') {
                        throw std::runtime_error("unsupported JSON escape");
                    }
                }
                out += c;
            }
            throw std::runtime_error("unterminated JSON string");
        };
        space();
        if (text.at(pos++) != '{') throw std::runtime_error("JSON object expected");
        while (true) {
            space();
            if (text.at(pos) == '}') break;
            std::string key = quoted();
            space();
            if (text.at(pos++) != ':') throw std::runtime_error("JSON colon expected");
            space();
            if (text.at(pos) == '"') {
                quoted();
            } else {
                char* end = nullptr;
                double value = std::strtod(text.c_str() + pos, &end);
                if (end == text.c_str() + pos) throw std::runtime_error("JSON number expected");
                pos = static_cast<size_t>(end - text.c_str());
                if (key == "n") n = static_cast<int>(value);
                else if (key == "nnz") nnz = static_cast<int>(value);
                else if (key == "requests") requests = static_cast<int>(value);
                else if (key == "t") t = value;
                else if (key == "rtol") rtol = value;
                else if (key == "atol") atol = value;
            }
            space();
            if (text.at(pos) == '}') break;
            if (text.at(pos++) != ',') throw std::runtime_error("JSON comma expected");
        }
        if (n < 1 || nnz < 0 || (requests != 1 && requests != 16)) {
            throw std::runtime_error("invalid dimensions");
        }
    }
};

static double phi1(double x) {
    if (std::abs(x) < 1e-6) {
        return 1.0 + x * (0.5 + x * (1.0 / 6 + x * (1.0 / 24 + x / 120)));
    }
    return std::expm1(x) / x;
}

static long double phi1_long(long double x) {
    if (std::abs(x) < 1e-7L) {
        return 1 + x * (0.5L + x * (1.0L / 6 + x * (1.0L / 24 + x / 120)));
    }
    return std::expm1(x) / x;
}

static double norm_inf(const Vec& x) {
    double result = 0;
    for (double value : x) result = std::max(result, std::abs(value));
    return result;
}

static double norm_two(const Vec& x) {
    double result = 0;
    for (double value : x) result += value * value;
    return std::sqrt(result);
}

struct Coefficients {
    Vec ex, ph, ex_tail, ph_tail;
    std::vector<long double> ex_long, ph_long, ph2_long;

    explicit Coefficients(double r) {
        int length = std::max(40, static_cast<int>(std::ceil(std::sqrt(110 * r) + 150)));
        std::vector<long double> ratios(length + 2), bessel(length + 1);
        long double next = 0;
        for (int k = length; k >= 1; --k) {
            next = 1 / (2 * static_cast<long double>(k) / r + next);
            ratios[k] = next;
        }
        bessel[0] = 1;
        long double total = 1;
        for (int k = 1; k <= length; ++k) {
            bessel[k] = bessel[k - 1] * ratios[k];
            total += 2 * bessel[k];
        }
        ex.resize(length + 1);
        ph.resize(length + 1);
        ex_long.resize(length + 1);
        ph_long.resize(length + 1);
        ph2_long.resize(length + 1);
        ex[0] = static_cast<double>(1 / total);
        ex_long[0] = 1 / total;
        for (int k = 1; k <= length; ++k) {
            ex_long[k] = 2 * bessel[k] / total;
            ex[k] = static_cast<double>(ex_long[k]);
        }
        long double tail = 0, weighted = 0;
        for (int k = length; k >= 1; --k) {
            ph_long[k] = 2 * weighted / r;
            ph[k] = static_cast<double>(ph_long[k]);
            tail += 2 * bessel[k] / total;
            weighted += tail;
        }
        ph[0] = static_cast<double>(weighted / r);
        ph_long[0] = weighted / r;
        tail = weighted = 0;
        for (int k = length; k >= 1; --k) {
            ph2_long[k] = 2 * weighted / r;
            tail += ph_long[k];
            weighted += tail;
        }
        ph2_long[0] = weighted / r;
        ex_tail.resize(length + 1);
        ph_tail.resize(length + 1);
        long double te = 0, tp = 0;
        for (int k = length; k >= 0; --k) {
            ex_tail[k] = static_cast<double>(te);
            ph_tail[k] = static_cast<double>(tp);
            te += ex[k];
            tp += ph[k];
        }
    }

    int degree(double nb0, double nb1, double t, double target) const {
        for (size_t k = 0; k < ex.size(); ++k) {
            if (nb0 * ex_tail[k] + (t * nb1) * ph_tail[k] <= target) {
                return static_cast<int>(k);
            }
        }
        return static_cast<int>(ex.size()) - 1;
    }
};

struct Matrix {
    int n;
    std::unique_ptr<uint64_t[]> row;
    std::unique_ptr<uint32_t[]> col;
    std::unique_ptr<double[]> val;
    Vec diag, row_sum;
    std::vector<long double> row_sum_long;
    double bound = 0;
    double lower = 0;
    double conservation_residual = 0;
    Vec lower_weights, inverse_barrier;
    double barrier_minimum = 0;
    bool pairs = true;
    bool extended = false;
    bool extended_products = false;
    bool diffusion = true;
    bool difference_products = true;
    bool diagonally_dominant = true;
    std::vector<int8_t> gauge;

    explicit Matrix(const Header& h) : n(h.n), row(new uint64_t[n + 1]), col(new uint32_t[h.nnz]),
        val(new double[h.nnz]), diag(n), row_sum(n), row_sum_long(n), extended(h.rtol <= 1e-10),
        extended_products(h.rtol <= 1e-10 && !std::getenv("FAM_MAT_DOUBLE")),
        difference_products(h.rtol <= 1e-10 || !std::getenv("FAM_DIRECT")) {
        size_t nonzeros = static_cast<size_t>(h.nnz);
        prefer_huge_pages(col.get(), nonzeros * sizeof(uint32_t), 4 * 1024 * 1024);
        prefer_huge_pages(val.get(), nonzeros * sizeof(double), 4 * 1024 * 1024);
        read_exact(row.get(), static_cast<size_t>(n + 1) * sizeof(uint64_t));
        read_exact(col.get(), nonzeros * sizeof(uint32_t));
        read_exact(val.get(), nonzeros * sizeof(double));
        Vec dominance(n);
        uint64_t position = 0;
        long double smallest_gap = std::numeric_limits<long double>::infinity();
        for (int i = 0; i < n; ++i) {
            long double sum = 0, absolute = 0;
            long double correction = 0, absolute_correction = 0;
            int off = 0;
            uint64_t begin = row[i], end = row[i + 1];
            row[i] = position;
            for (uint64_t j = begin; j < end; ++j) {
                long double increment = static_cast<long double>(val[j]) - correction;
                long double next_sum = sum + increment;
                correction = (next_sum - sum) - increment;
                sum = next_sum;
                increment = static_cast<long double>(std::abs(val[j])) - absolute_correction;
                next_sum = absolute + increment;
                absolute_correction = (next_sum - absolute) - increment;
                absolute = next_sum;
                if (col[j] == static_cast<uint32_t>(i)) diag[i] = val[j];
                else if (val[j] != 0) {
                    if (val[j] > 0) diffusion = false;
                    ++off;
                    col[position] = col[j];
                    val[position] = val[j];
                    ++position;
                }
            }
            row_sum[i] = static_cast<double>(sum);
            row_sum_long[i] = sum;
            if (sum < -1e-13L * absolute) diffusion = false;
            if (2 * static_cast<long double>(diag[i]) - absolute < -1e-13L * absolute) {
                diagonally_dominant = false;
            }
            long double gap = 2 * static_cast<long double>(diag[i]) - absolute;
            dominance[i] = static_cast<double>(gap);
            smallest_gap = std::min(smallest_gap, gap);
            bound = std::max(bound, static_cast<double>(absolute));
            pairs = pairs && off <= 1;
        }
        row[n] = position;
        if (h.rtol <= 1e-10 && h.t * bound >= 100 && diagonally_dominant &&
                !diffusion && !pairs && n > 48 && !std::getenv("FAM_NO_GAUGE")) {
            signed_diffusion_transform();
        }
        if (diffusion && !std::getenv("FAM_FORCE_LONG")) extended_products = false;
        if (diffusion && h.rtol <= 1e-7) difference_products = true;
        if (bound > 0) bound = std::nextafter(bound, std::numeric_limits<double>::infinity());
        lower = std::max(0.0, static_cast<double>(smallest_gap) - 1e-13 * bound);
        if (lower > 0) {
            inverse_barrier = std::move(dominance);
            for (double& value : inverse_barrier) value -= 1e-13 * bound;
            barrier_minimum = lower;
        }
        long double residual_square = 0;
        long double row_guard = 32 * std::numeric_limits<long double>::epsilon() * bound;
        for (long double value : row_sum_long) {
            long double magnitude = std::abs(value) + row_guard;
            residual_square += magnitude * magnitude;
        }
        conservation_residual = static_cast<double>(std::sqrt(residual_square)) * (1 + 1e-12);
    }

    bool signed_diffusion_transform() {
        std::vector<int8_t> signs(n);
        std::vector<int> queue;
        queue.reserve(n);
        for (int start = 0; start < n; ++start) {
            if (signs[start]) continue;
            signs[start] = 1;
            queue.clear();
            queue.push_back(start);
            for (size_t at = 0; at < queue.size(); ++at) {
                int i = queue[at];
                for (uint64_t j = row[i]; j < row[i + 1]; ++j) {
                    int other = static_cast<int>(col[j]);
                    int8_t expected = val[j] < 0 ? signs[i] : -signs[i];
                    if (!signs[other]) {
                        signs[other] = expected;
                        queue.push_back(other);
                    } else if (signs[other] != expected) {
                        return false;
                    }
                }
            }
        }
        gauge = std::move(signs);
        for (int i = 0; i < n; ++i) {
            long double sum = diag[i], correction = 0;
            for (uint64_t j = row[i]; j < row[i + 1]; ++j) {
                if (gauge[i] != gauge[col[j]]) val[j] = -val[j];
                long double term = static_cast<long double>(val[j]) - correction;
                long double next = sum + term;
                correction = (next - sum) - term;
                sum = next;
            }
            row_sum_long[i] = sum;
            row_sum[i] = static_cast<double>(sum);
        }
        diffusion = true;
        if (std::getenv("FAM_DEBUG")) std::fprintf(stderr, "balanced sign transform\n");
        return true;
    }

    double robust_row_product(int i, const Vec& x, double scale,
            double shift, bool differences) const {
        long double xi = x[i];
        long double sum = ((differences ? row_sum_long[i] : diag[i]) - shift) * xi;
        long double correction = 0;
        for (uint64_t j = row[i]; j < row[i + 1]; ++j) {
            long double z = x[col[j]];
            if (differences) z -= xi;
            long double term = static_cast<long double>(val[j]) * z - correction;
            long double next = sum + term;
            correction = (next - sum) - term;
            sum = next;
        }
        return static_cast<double>(sum * scale);
    }

    void refine_bound(int requests, double estimated_degree) {
        Vec x(n, 1), y(n);
        double original = bound;
        bool full = std::getenv("FAM_FULL_BOUND");
        for (int iteration = 0; iteration < 10; ++iteration) {
            double before = bound;
            double upper = 0, largest = 0;
            for (int i = 0; i < n; ++i) {
                long double sum = static_cast<long double>(diag[i]) * x[i];
                for (uint64_t j = row[i]; j < row[i + 1]; ++j) {
                    sum += static_cast<long double>(std::abs(val[j])) * x[col[j]];
                }
                y[i] = static_cast<double>(sum / original) + x[i];
                upper = std::max(upper, y[i] / x[i]);
                largest = std::max(largest, y[i]);
            }
            double candidate = (upper - 1) * original * (1 + 1e-12);
            bound = std::min(bound, candidate);
            if (iteration == 2 && bound >= original * 0.998) break;
            if (iteration >= 2 && !full &&
                    (before - bound) * estimated_degree * requests < 2 * bound) break;
            for (int i = 0; i < n; ++i) x[i] = y[i] / largest;
        }
    }

    void refine_lower(double t) {
        if (!diffusion || !bound) return;
        double largest_row_sum = 0;
        for (double value : row_sum) largest_row_sum = std::max(largest_row_sum, value);
        if (t * largest_row_sum <= 24 || t * lower >= 60) return;
        Vec x(n, 1), next(n), barrier(n);
        for (int iteration = 0; iteration < 12; ++iteration) {
            double candidate = std::numeric_limits<double>::infinity();
            double smallest_barrier = std::numeric_limits<double>::infinity();
            double largest = 0;
            for (int i = 0; i < n; ++i) {
                double z = robust_row_product(i, x, 1, 0, true);
                barrier[i] = z - 1e-13 * bound;
                smallest_barrier = std::min(smallest_barrier, barrier[i]);
                candidate = std::min(candidate, z / x[i]);
                next[i] = 2 * x[i] - z / bound;
                largest = std::max(largest, next[i]);
            }
            candidate -= 1e-13 * bound;
            if (candidate > lower) {
                lower = candidate;
                lower_weights = x;
            }
            if (smallest_barrier > barrier_minimum) {
                inverse_barrier = barrier;
                barrier_minimum = smallest_barrier;
            }
            for (int i = 0; i < n; ++i) x[i] = next[i] / largest;
        }
    }

    void accurate_residual(const Vec& x, const Vec& b, Vec& residual) const {
        for (int i = 0; i < n; ++i) {
            double high = 0, low = 0;
            auto product = [&](double coefficient, double value) {
                double p = coefficient * value;
                double error = std::fma(coefficient, value, -p);
                double sum = high + p;
                double z = sum - high;
                low += ((high - (sum - z)) + (p - z)) + error;
                high = sum;
            };
            product(diag[i], x[i]);
            for (uint64_t j = row[i]; j < row[i + 1]; ++j) product(val[j], x[col[j]]);
            residual[i] = static_cast<double>(static_cast<long double>(b[i]) -
                (static_cast<long double>(high) + low));
        }
    }

    void multiply_scaled(const Vec& x, Vec& y, double scale, double shift = 0) const {
        if (!extended_products && !difference_products) {
            for (int i = 0; i < n; ++i) {
                double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                uint64_t j = row[i], end = row[i + 1];
                if (end - j > 1024) {
                    y[i] = robust_row_product(i, x, scale, shift, false);
                    continue;
                }
                for (; j + 3 < end; j += 4) {
                    s0 = std::fma(val[j], x[col[j]], s0);
                    s1 = std::fma(val[j + 1], x[col[j + 1]], s1);
                    s2 = std::fma(val[j + 2], x[col[j + 2]], s2);
                    s3 = std::fma(val[j + 3], x[col[j + 3]], s3);
                }
                double sum = (s0 + s1) + (s2 + s3);
                for (; j < end; ++j) sum = std::fma(val[j], x[col[j]], sum);
                y[i] = std::fma(diag[i] - shift, x[i], sum) * scale;
            }
            return;
        }
        if (extended_products) {
            for (int i = 0; i < n; ++i) {
                long double sum = 0;
                long double second = 0;
                long double xi = x[i];
                uint64_t j = row[i], end = row[i + 1];
                if (end - j > 1024) {
                    y[i] = robust_row_product(i, x, scale, shift, true);
                    continue;
                }
                for (; j + 1 < end; j += 2) {
                    sum += static_cast<long double>(val[j]) *
                        (static_cast<long double>(x[col[j]]) - xi);
                    second += static_cast<long double>(val[j + 1]) *
                        (static_cast<long double>(x[col[j + 1]]) - xi);
                }
                sum += second;
                for (; j < end; ++j) {
                    sum += static_cast<long double>(val[j]) *
                        (static_cast<long double>(x[col[j]]) - xi);
                }
                sum += (row_sum_long[i] - shift) * xi;
                y[i] = static_cast<double>(sum * scale);
            }
        } else {
            for (int i = 0; i < n; ++i) {
                double sum = 0, s1 = 0, s2 = 0, s3 = 0;
                double xi = x[i];
                uint64_t j = row[i], end = row[i + 1];
                if (end - j > 1024) {
                    y[i] = robust_row_product(i, x, scale, shift, true);
                    continue;
                }
                for (; j + 3 < end; j += 4) {
                    sum = std::fma(val[j], x[col[j]] - xi, sum);
                    s1 = std::fma(val[j + 1], x[col[j + 1]] - xi, s1);
                    s2 = std::fma(val[j + 2], x[col[j + 2]] - xi, s2);
                    s3 = std::fma(val[j + 3], x[col[j + 3]] - xi, s3);
                }
                sum = (sum + s1) + (s2 + s3);
                for (; j < end; ++j) {
                    sum = std::fma(val[j], x[col[j]] - xi, sum);
                }
                y[i] = (sum + (row_sum[i] - shift) * xi) * scale;
            }
        }
    }
};

class Solver {
    Header h;
    Matrix a;
    std::vector<long double> dense_vectors, dense_exp, dense_phi;
    bool dense = false;
    std::vector<std::array<long double, 6>> pair_functions;
    std::vector<std::array<double, 6>> pair_functions_fast;
    std::vector<int> mate;
    std::vector<Coefficients> coefficients;
    std::unique_ptr<double[]> lanczos_basis;
    size_t lanczos_capacity = 0;
    std::vector<int> block_ptr, block_order;
    std::vector<size_t> block_storage;
    std::vector<long double> block_exp, block_phi;
    Vec inverse_diagonal;
    double output_lower = 0;
    double rhs_norm0 = 0, rhs_norm1 = 0;
    bool steady_disabled = false;

    double known_output_lower(const Vec& b0, const Vec& b1, double n0, double n1) const {
        double root_n = std::sqrt(h.n);
        double argument_lower = h.t * n1 * (1 - 1e-9) - n0 * (1 + 1e-9);
        double result = std::max(0.0, argument_lower) * phi1(-h.t * a.bound) / root_n;
        result *= 1 - 1e-12;
        long double sum0 = 0, sum1 = 0;
        double maximum0 = 0, maximum1 = 0;
        for (int i = 0; i < h.n; ++i) {
            sum0 += b0[i];
            sum1 += b1[i];
            maximum0 = std::max(maximum0, std::abs(b0[i]));
            maximum1 = std::max(maximum1, std::abs(b1[i]));
            if (a.row[i] == a.row[i + 1]) {
                long double z = -static_cast<long double>(h.t) * a.diag[i];
                long double value = std::exp(z) * b0[i] +
                    static_cast<long double>(h.t) * phi1_long(z) * b1[i];
                long double guard = 1e-14L * (std::abs(b0[i]) + h.t * std::abs(b1[i]));
                result = std::max(result, static_cast<double>(std::abs(value) - guard));
            }
        }
        long double mean = (sum0 + static_cast<long double>(h.t) * sum1) / h.n;
        double defect = a.conservation_residual / h.n *
            (h.t * n0 + (0.5 * h.t * h.t) * n1) * (1 + 1e-9);
        double guard = 1e-12 * (maximum0 + h.t * maximum1);
        result = std::max(result, static_cast<double>(std::abs(mean)) - defect - guard);
        return result;
    }

    bool try_steady(const Vec& b0, const Vec& b1, Vec& y) {
        if (a.lower <= 0 || h.t * a.lower < 24 || std::getenv("FAM_NO_STEADY")) return false;
        auto weighted_norm = [&](const Vec& x) {
            if (a.lower_weights.empty()) return norm_inf(x);
            double result = 0;
            for (int i = 0; i < h.n; ++i) {
                result = std::max(result, std::abs(x[i]) / a.lower_weights[i]);
            }
            return result;
        };
        auto residual_error = [&](const Vec& x) {
            if (a.inverse_barrier.empty()) return weighted_norm(x) / a.lower;
            double result = 0;
            for (int i = 0; i < h.n; ++i) {
                result = std::max(result, std::abs(x[i]) / a.inverse_barrier[i]);
            }
            return result;
        };
        double nb0 = weighted_norm(b0), nb1 = weighted_norm(b1);
        double transient = std::exp(-h.t * a.lower) * (nb0 + nb1 / a.lower);
        if (transient > 0.01 * h.atol) return false;
        if (nb1 / a.lower <= h.atol * 0.02) {
            y.assign(h.n, 0);
            return true;
        }
        if (steady_disabled) return false;
        y.assign(h.n, 0);
        if (inverse_diagonal.empty()) {
            inverse_diagonal.resize(h.n);
            for (int i = 0; i < h.n; ++i) inverse_diagonal[i] = 1 / a.diag[i];
        }
        Vec r = b1, z(h.n), p(h.n), ap(h.n);
        for (int i = 0; i < h.n; ++i) p[i] = z[i] = r[i] * inverse_diagonal[i];
        double rho = cblas_ddot(h.n, r.data(), 1, z.data(), 1);
        const auto& c = coefficients[0];
        int polynomial_degree = c.degree(rhs_norm0, rhs_norm1, h.t,
            0.04 * (h.atol + h.rtol * output_lower));
        int maximum = std::min(256, std::max(32, polynomial_degree / 5));
        double previous_error = std::numeric_limits<double>::infinity();
        int stagnant_checks = 0;
        for (int k = 0; k < maximum; ++k) {
            a.multiply_scaled(p, ap, 1);
            double denominator = cblas_ddot(h.n, p.data(), 1, ap.data(), 1);
            if (!(denominator > 0) || !(rho > 0)) {
                steady_disabled = true;
                return false;
            }
            double alpha = rho / denominator;
            for (int i = 0; i < h.n; ++i) {
                y[i] = std::fma(alpha, p[i], y[i]);
                r[i] = std::fma(-alpha, ap[i], r[i]);
            }
            double target = 0.04 * (h.atol + h.rtol * norm_inf(y));
            bool checked = false;
            if ((k + 1) % 64 == 0 || residual_error(r) < target) {
                a.accurate_residual(y, b1, r);
                double error_bound = residual_error(r) + transient;
                if (error_bound < target) {
                    if (std::getenv("FAM_DEBUG")) {
                        std::fprintf(stderr, "steady CG iterations=%d lower=%.5g bound=%.3g\n",
                            k + 1, a.lower, error_bound);
                    }
                    return true;
                }
                if (error_bound >= previous_error * 0.9) ++stagnant_checks;
                else stagnant_checks = 0;
                previous_error = error_bound;
                if (stagnant_checks >= 2 && k >= 32) {
                    steady_disabled = true;
                    if (std::getenv("FAM_DEBUG")) {
                        std::fprintf(stderr, "steady CG stagnation iterations=%d\n", k + 1);
                    }
                    return false;
                }
                checked = true;
            }
            for (int i = 0; i < h.n; ++i) z[i] = r[i] * inverse_diagonal[i];
            double next_rho = cblas_ddot(h.n, r.data(), 1, z.data(), 1);
            double beta = checked ? 0 : next_rho / rho;
            for (int i = 0; i < h.n; ++i) p[i] = std::fma(beta, p[i], z[i]);
            rho = next_rho;
        }
        steady_disabled = true;
        if (std::getenv("FAM_DEBUG")) std::fprintf(stderr, "steady CG fallback iterations=%d\n", maximum);
        return false;
    }

    Vec apply_expansion(const Vec& b0, const Vec& b1, const Vec& ex, const Vec& ph,
            int degree, double phi_scale, double right, double left = 0) const {
        Vec y(h.n), u(h.n), d(h.n), work(h.n), uc, dc;
        if (a.extended) {
            uc.resize(h.n);
            dc.resize(h.n);
        }
        double scale = 2 / (right - left);
        double ce = ex[degree], cp = phi_scale * ph[degree];
        for (int i = 0; i < h.n; ++i) u[i] = d[i] = ce * b0[i] + cp * b1[i];
        for (int k = degree - 1; k >= 1; --k) {
            a.multiply_scaled(u, work, scale, left);
            ce = ex[k];
            cp = phi_scale * ph[k];
            if (a.extended) {
                for (int i = 0; i < h.n; ++i) {
                    double q = std::fma(ce, b0[i], cp * b1[i]);
                    double step = std::fma(-2.0, work[i], q) - dc[i];
                    double next = d[i] + step;
                    dc[i] = (next - d[i]) - step;
                    d[i] = next;
                    step = next - uc[i];
                    next = u[i] + step;
                    uc[i] = (next - u[i]) - step;
                    u[i] = next;
                }
            } else {
                for (int i = 0; i < h.n; ++i) {
                    d[i] += std::fma(-2.0, work[i], std::fma(ce, b0[i], cp * b1[i]));
                    u[i] += d[i];
                }
            }
        }
        if (degree > 0) {
            a.multiply_scaled(u, work, scale, left);
            for (int i = 0; i < h.n; ++i) {
                y[i] = (ex[0] * b0[i] + phi_scale * ph[0] * b1[i]) + (d[i] - work[i]);
            }
        } else {
            for (int i = 0; i < h.n; ++i) y[i] = ex[0] * b0[i] + phi_scale * ph[0] * b1[i];
        }
        return y;
    }

    bool try_stable_cancellation(const Vec& b0, const Vec& b1, Vec& y) const {
        Vec residual(h.n), v(h.n);
        for (int i = 0; i < h.n; ++i) {
            residual[i] = std::fma(h.t, b1[i], b0[i]);
            double high = 0, low = 0;
            auto product = [&](double coefficient, double x) {
                double p = coefficient * x;
                double error = std::fma(coefficient, x, -p);
                double sum = high + p;
                double z = sum - high;
                low += ((high - (sum - z)) + (p - z)) + error;
                high = sum;
            };
            product(a.diag[i], b0[i]);
            for (uint64_t j = a.row[i]; j < a.row[i + 1]; ++j) product(a.val[j], b0[a.col[j]]);
            v[i] = static_cast<double>(-static_cast<long double>(h.t) *
                (static_cast<long double>(high) + low));
        }
        double size = norm_two(residual) + norm_two(v);
        if (size > 1e-4) return false;
        const auto& c = coefficients[0];
        Vec difference(c.ph.size());
        for (size_t k = 0; k < difference.size(); ++k) {
            difference[k] = static_cast<double>(c.ph_long[k] - c.ph2_long[k]);
        }
        int degree = c.degree(0, size, 1, h.atol * 0.02);
        y = apply_expansion(residual, v, c.ph, difference, degree, 1, a.bound);
        if (std::getenv("FAM_DEBUG")) {
            std::fprintf(stderr, "stable cancellation size=%.3g degree=%d\n", size, degree);
        }
        return true;
    }

    Vec sensitive_clenshaw(const Vec& b0, const Vec& b1) const {
        const auto& c = coefficients[0];
        int degree = c.degree(rhs_norm0, rhs_norm1, h.t, h.atol * 0.02);
        std::vector<long double> u(h.n), d(h.n), work(h.n);
        long double scale = 2 / static_cast<long double>(a.bound);
        long double time = h.t;
        long double ce = c.ex_long[degree], cp = time * c.ph_long[degree];
        for (int i = 0; i < h.n; ++i) u[i] = d[i] = ce * b0[i] + cp * b1[i];
        Vec y(h.n);
        auto update = [&](long double ec, long double pc, bool finish) {
            for (int i = 0; i < h.n; ++i) {
                long double sum = 0, second = 0;
                uint64_t j = a.row[i], end = a.row[i + 1];
                for (; j + 1 < end; j += 2) {
                    sum += static_cast<long double>(a.val[j]) * (u[a.col[j]] - u[i]);
                    second += static_cast<long double>(a.val[j + 1]) * (u[a.col[j + 1]] - u[i]);
                }
                for (; j < end; ++j) sum += static_cast<long double>(a.val[j]) * (u[a.col[j]] - u[i]);
                long double applied = (sum + second + a.row_sum_long[i] * u[i]) * scale;
                if (finish) {
                    y[i] = static_cast<double>(ec * b0[i] + pc * b1[i] + d[i] - applied);
                } else {
                    d[i] += ec * b0[i] + pc * b1[i] - 2 * applied;
                    work[i] = u[i] + d[i];
                }
            }
        };
        for (int k = degree - 1; k >= 1; --k) {
            ce = c.ex_long[k];
            cp = time * c.ph_long[k];
            update(ce, cp, false);
            u.swap(work);
        }
        if (degree > 0) {
            update(c.ex_long[0], time * c.ph_long[0], true);
        } else {
            for (int i = 0; i < h.n; ++i) {
                y[i] = static_cast<double>(c.ex_long[0] * b0[i] + time * c.ph_long[0] * b1[i]);
            }
        }
        if (std::getenv("FAM_DEBUG")) std::fprintf(stderr, "extended recurrence degree=%d\n", degree);
        return y;
    }

    bool prepare_small_blocks() {
        std::vector<int> parent(h.n), count(h.n, 1);
        std::iota(parent.begin(), parent.end(), 0);
        auto root = [&](int i) {
            while (parent[i] != i) {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            return i;
        };
        for (int i = 0; i < h.n; ++i) {
            for (uint64_t j = a.row[i]; j < a.row[i + 1]; ++j) {
                int v = static_cast<int>(a.col[j]);
                if (v <= i) continue;
                int p = root(i), q = root(v);
                if (p == q) continue;
                if (count[p] + count[q] > 24) return false;
                if (count[p] < count[q]) std::swap(p, q);
                parent[q] = p;
                count[p] += count[q];
            }
        }
        std::vector<int> group(h.n, -1);
        block_ptr.push_back(0);
        for (int i = 0; i < h.n; ++i) {
            if (parent[i] == i) {
                group[i] = static_cast<int>(block_ptr.size()) - 1;
                block_ptr.push_back(block_ptr.back() + count[i]);
            }
        }
        block_order.resize(h.n);
        std::vector<int> position = block_ptr;
        std::vector<int> local(h.n);
        for (int i = 0; i < h.n; ++i) {
            int g = group[root(i)];
            int p = position[g]++;
            block_order[p] = i;
            local[i] = p - block_ptr[g];
        }
        using LongMatrix = Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic>;
        block_storage.push_back(0);
        for (size_t g = 0; g + 1 < block_ptr.size(); ++g) {
            int begin = block_ptr[g], m = block_ptr[g + 1] - begin;
            LongMatrix matrix = LongMatrix::Zero(m, m);
            for (int j = 0; j < m; ++j) {
                int i = block_order[begin + j];
                matrix(j, j) = a.diag[i];
                for (uint64_t k = a.row[i]; k < a.row[i + 1]; ++k) {
                    matrix(j, local[a.col[k]]) = a.val[k];
                }
            }
            Eigen::SelfAdjointEigenSolver<LongMatrix> eigen(matrix);
            if (eigen.info() != Eigen::Success) throw std::runtime_error("block eigensolver failure");
            Eigen::Matrix<long double, Eigen::Dynamic, 1> ex(m), ph(m);
            for (int j = 0; j < m; ++j) {
                long double z = -h.t * eigen.eigenvalues()[j];
                ex[j] = std::exp(z);
                ph[j] = h.t * phi1_long(z);
            }
            LongMatrix f = eigen.eigenvectors() * ex.asDiagonal() * eigen.eigenvectors().transpose();
            LongMatrix p = eigen.eigenvectors() * ph.asDiagonal() * eigen.eigenvectors().transpose();
            block_exp.insert(block_exp.end(), f.data(), f.data() + static_cast<size_t>(m) * m);
            block_phi.insert(block_phi.end(), p.data(), p.data() + static_cast<size_t>(m) * m);
            block_storage.push_back(block_exp.size());
        }
        return true;
    }

    std::vector<long double> projected_function(
            const Vec& alpha, const Vec& beta, int m,
            const std::vector<long double>& cf) const {
        std::vector<long double> u(m), d(m), work(m);
        const int degree = static_cast<int>(cf.size()) - 1;
        u[0] = d[0] = cf[degree];
        for (int k = degree - 1; k >= 1; --k) {
            for (int i = 0; i < m; ++i) {
                long double z = static_cast<long double>(alpha[i]) * u[i];
                if (i) z += static_cast<long double>(beta[i - 1]) * u[i - 1];
                if (i + 1 < m) z += static_cast<long double>(beta[i]) * u[i + 1];
                work[i] = z;
            }
            d[0] += cf[k];
            for (int i = 0; i < m; ++i) {
                d[i] -= 4 * work[i];
                u[i] += d[i];
            }
        }
        for (int i = 0; i < m; ++i) {
            long double z = static_cast<long double>(alpha[i]) * u[i];
            if (i) z += static_cast<long double>(beta[i - 1]) * u[i - 1];
            if (i + 1 < m) z += static_cast<long double>(beta[i]) * u[i + 1];
            work[i] = d[i] - 2 * z;
        }
        work[0] += cf[0];
        return work;
    }

    bool try_lanczos(const Vec& b, bool source, Vec& y,
            double* achieved_error = nullptr, double absolute_budget = 0, int iteration_cap = 0,
            bool complete_output = false) {
        double magnitude = cblas_dnrm2(h.n, b.data(), 1);
        if (magnitude == 0) {
            std::fill(y.begin(), y.end(), 0);
            if (achieved_error) *achieved_error = 0;
            return true;
        }
        double upper = magnitude * (source ? h.t : 1);
        if (upper <= h.atol * 0.01) {
            std::fill(y.begin(), y.end(), 0);
            if (achieved_error) *achieved_error = upper;
            return true;
        }
        const auto& c = coefficients[0];
        double eta = h.t * a.bound;
        int policy_maximum = h.rtol >= 1e-4 ? 256 : h.rtol >= 1e-7 ? 352 : 512;
        if (const char* requested = std::getenv("FAM_KRYLOV_MAX")) {
            policy_maximum = std::clamp(std::atoi(requested), 48, 1536);
        }
        int full_degree = source ? c.degree(0, magnitude, h.t, h.atol * 0.04) :
            c.degree(magnitude, 0, 1, std::max(h.atol * 0.04, absolute_budget * 0.01));
        int maximum = std::min(h.n, std::min(policy_maximum, std::max(48, full_degree * 3 / 4)));
        if (iteration_cap > 0) maximum = std::min(maximum, std::max(48, iteration_cap));
        size_t required = static_cast<size_t>(h.n) * (maximum + 1);
        if (required > lanczos_capacity) {
            size_t block = 1U << 20;
            size_t allocation = (required + block - 1) / block * block;
            lanczos_basis.reset(new double[allocation]);
            lanczos_capacity = allocation;
            prefer_huge_pages(lanczos_basis.get(), allocation * sizeof(double));
        }
        double* first = lanczos_basis.get();
        for (int i = 0; i < h.n; ++i) first[i] = b[i] / magnitude;
        Vec alpha(maximum), beta(maximum), q(h.n), w(h.n);
        std::vector<long double> previous;
        int successful_checks = 0;
        int expensive_predictions = 0;
        double previous_residual = std::numeric_limits<double>::infinity();
        std::array<int, 96> sampled_rows;
        for (int s = 0; s < 96; ++s) {
            sampled_rows[s] = static_cast<int>((static_cast<int64_t>(s) * (h.n - 1)) / 95);
        }
        double known_size = 0;
        for (int i = 0; i < h.n; ++i) {
            if (a.row[i] == a.row[i + 1]) {
                double z = -h.t * a.diag[i];
                double factor = source ? h.t * phi1(z) : std::exp(z);
                known_size = std::max(known_size, std::abs(factor * b[i]));
            }
        }
        for (int k = 0; k < maximum; ++k) {
            const double* current = lanczos_basis.get() + static_cast<size_t>(k) * h.n;
            std::copy(current, current + h.n, q.begin());
            a.multiply_scaled(q, w, 1 / a.bound);
            if (k) {
                const double* old = current - h.n;
                for (int i = 0; i < h.n; ++i) w[i] -= beta[k - 1] * old[i];
            }
            alpha[k] = cblas_ddot(h.n, current, 1, w.data(), 1);
            for (int i = 0; i < h.n; ++i) w[i] -= alpha[k] * current[i];
            double correction = cblas_ddot(h.n, current, 1, w.data(), 1);
            alpha[k] += correction;
            for (int i = 0; i < h.n; ++i) w[i] -= correction * current[i];
            beta[k] = cblas_dnrm2(h.n, w.data(), 1);
            const int m = k + 1;
            bool breakdown = beta[k] < 1e-15 || m == h.n;
            if ((m >= 24 && m % 16 == 0) || breakdown || m == maximum) {
                auto g = projected_function(alpha, beta, m, source ? c.ph_long : c.ex_long);
                auto integral = projected_function(alpha, beta, m, source ? c.ph2_long : c.ph_long);
                long double scale = magnitude * (source ? h.t : 1);
                long double gnorm = 0, difference = 0;
                for (int j = 0; j < m; ++j) {
                    gnorm += g[j] * g[j];
                    long double delta = g[j] - (j < static_cast<int>(previous.size()) ? previous[j] : 0);
                    difference += delta * delta;
                }
                std::array<double, 96> sampled{};
                for (int j = 0; j < m; ++j) {
                    const double* column = lanczos_basis.get() + static_cast<size_t>(j) * h.n;
                    double coefficient = static_cast<double>(scale * g[j]);
                    for (int s = 0; s < 96; ++s) {
                        sampled[s] = std::fma(coefficient, column[sampled_rows[s]], sampled[s]);
                    }
                }
                double size_estimate = std::max(known_size,
                    static_cast<double>(scale * std::sqrt(gnorm) / std::sqrt(h.n)));
                double sample_maximum = 0;
                for (double value : sampled) sample_maximum = std::max(sample_maximum, std::abs(value));
                size_estimate = std::max(size_estimate, sample_maximum);
                double target = std::max(absolute_budget, 0.025 * (h.atol + h.rtol * size_estimate));
                double residual_bound = static_cast<double>(scale * eta * beta[k] * std::abs(integral.back()));
                double change_bound = static_cast<double>(scale * std::sqrt(m * difference));
                if (complete_output) {
                    double guard = static_cast<double>(128 * std::numeric_limits<double>::epsilon() *
                        eta * scale * std::sqrt(m));
                    output_lower = std::max(output_lower, sample_maximum - 2 * residual_bound - guard);
                }
                bool acceptable = residual_bound < target &&
                    (change_bound < 2 * target || breakdown);
                if (acceptable) ++successful_checks;
                else successful_checks = 0;
                if ((successful_checks >= 2 || breakdown) && acceptable) {
                    Vec weights(m);
                    for (int j = 0; j < m; ++j) weights[j] = static_cast<double>(scale * g[j]);
                    cblas_dgemv(CblasColMajor, CblasNoTrans, h.n, m, 1,
                        lanczos_basis.get(), h.n, weights.data(), 1, 0, y.data(), 1);
                    double actual_target = std::max(absolute_budget, 0.025 * (h.atol + h.rtol * norm_inf(y)));
                    if (residual_bound < actual_target && (change_bound < 2 * actual_target || breakdown)) {
                        if (achieved_error) {
                            *achieved_error = breakdown ? residual_bound : std::max(residual_bound, change_bound);
                        }
                        if (std::getenv("FAM_DEBUG")) {
                            std::fprintf(stderr, "Lanczos m=%d residual=%.3g change=%.3g target=%.3g\n",
                                m, residual_bound, change_bound, actual_target);
                        }
                        return true;
                    }
                }
                if (m >= 96 && residual_bound > 8 * target) {
                    double ratio = residual_bound / previous_residual;
                    double predicted = std::numeric_limits<double>::infinity();
                    if (ratio > 0 && ratio < 0.99) {
                        predicted = m + 16 * std::log(target / residual_bound) / std::log(ratio);
                    }
                    if (predicted > maximum * 1.4) ++expensive_predictions;
                    else expensive_predictions = 0;
                    if (expensive_predictions >= 2) {
                        if (std::getenv("FAM_DEBUG")) {
                            std::fprintf(stderr, "Lanczos early fallback m=%d estimate=%.0f\n", m, predicted);
                        }
                        return false;
                    }
                }
                previous_residual = residual_bound;
                previous = std::move(g);
            }
            if (breakdown) return false;
            double* next = lanczos_basis.get() + static_cast<size_t>(m) * h.n;
            for (int i = 0; i < h.n; ++i) next[i] = w[i] / beta[k];
        }
        if (std::getenv("FAM_DEBUG")) std::fprintf(stderr, "Lanczos fallback m=%d\n", maximum);
        return false;
    }

    void prepare_pairs() {
        mate.resize(h.n, -1);
        const bool fast = h.rtol >= 1e-7;
        if (fast) pair_functions_fast.resize(h.n);
        else pair_functions.resize(h.n);
        auto store = [&](int i, const std::array<long double, 6>& values) {
            if (fast) {
                for (int j = 0; j < 6; ++j) pair_functions_fast[i][j] = static_cast<double>(values[j]);
            } else pair_functions[i] = values;
        };
        auto exp_value = [&](long double z) {
            return fast ? static_cast<long double>(std::exp(static_cast<double>(z))) : std::exp(z);
        };
        auto phi_value = [&](long double z) {
            return fast ? static_cast<long double>(phi1(static_cast<double>(z))) : phi1_long(z);
        };
        for (int i = 0; i < h.n; ++i) {
            if (mate[i] != -1) continue;
            int other = -1;
            double off = 0;
            for (uint64_t k = a.row[i]; k < a.row[i + 1]; ++k) {
                if (a.col[k] != static_cast<uint32_t>(i) && a.val[k] != 0) {
                    other = static_cast<int>(a.col[k]);
                    off = a.val[k];
                }
            }
            if (other < 0) {
                mate[i] = i;
                long double z = -static_cast<long double>(h.t) * a.diag[i];
                store(i, {exp_value(z), 0, 0, h.t * phi_value(z), 0, 0});
                continue;
            }
            mate[i] = other;
            mate[other] = i;
            long double x = a.diag[i], z = a.diag[other], b = off;
            long double gap = std::hypotl(x - z, 2 * b);
            long double hi = (x + z + gap) / 2;
            long double lo = hi == 0 ? 0 : (x * z - b * b) / hi;
            long double e0 = exp_value(-h.t * lo), e1 = exp_value(-h.t * hi);
            long double p0 = h.t * phi_value(-h.t * lo), p1 = h.t * phi_value(-h.t * hi);
            long double p00 = gap == 0 ? 1 : (hi - x) / gap;
            long double p01 = gap == 0 ? 0 : -b / gap;
            long double p11 = gap == 0 ? 1 : (hi - z) / gap;
            store(i, {e1 + (e0 - e1) * p00, (e0 - e1) * p01,
                e1 + (e0 - e1) * p11, p1 + (p0 - p1) * p00, (p0 - p1) * p01,
                p1 + (p0 - p1) * p11});
        }
    }

    void prepare_dense() {
        dense = true;
        using LongMatrix = Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic>;
        LongMatrix matrix = LongMatrix::Zero(h.n, h.n);
        for (int i = 0; i < h.n; ++i) {
            matrix(i, i) = a.diag[i];
            for (uint64_t j = a.row[i]; j < a.row[i + 1]; ++j) {
                matrix(i, a.col[j]) = a.val[j];
            }
        }
        Eigen::SelfAdjointEigenSolver<LongMatrix> eigen(matrix);
        if (eigen.info() != Eigen::Success) throw std::runtime_error("eigensolver failure");
        dense_vectors.assign(eigen.eigenvectors().data(),
            eigen.eigenvectors().data() + static_cast<size_t>(h.n) * h.n);
        dense_exp.resize(h.n);
        dense_phi.resize(h.n);
        for (int i = 0; i < h.n; ++i) {
            long double z = -h.t * eigen.eigenvalues()[i];
            dense_exp[i] = std::exp(z);
            dense_phi[i] = h.t * phi1_long(z);
        }
    }

public:
    explicit Solver(const Header& header) : h(header), a(header) {
        if (h.t == 0 || a.bound == 0) return;
        if (h.t * a.bound < 1e-10) return;
        if (a.pairs) prepare_pairs();
        else if (h.n <= 48) prepare_dense();
        else if (h.t * a.bound >= 100 && prepare_small_blocks()) {}
        else {
            if (h.t * a.bound >= 100) {
                double estimate = std::sqrt(h.t * a.bound *
                    (std::log(1 / h.atol) + std::log(h.n) + std::max(0.0, std::log(h.t))));
                a.refine_bound(h.requests, estimate);
            }
            a.refine_lower(h.t);
            coefficients.emplace_back(h.t * a.bound / 2);
        }
    }

    Vec solve_fast(const Vec& b0, const Vec& b1) {
        Vec y(h.n);
        if (h.t == 0) return b0;
        if (a.bound == 0) {
            for (int i = 0; i < h.n; ++i) y[i] = std::fma(h.t, b1[i], b0[i]);
            return y;
        }
        if (h.t * a.bound < 1e-10) {
            Vec midpoint(h.n), work(h.n);
            for (int i = 0; i < h.n; ++i) midpoint[i] = std::fma(h.t * 0.5, b1[i], b0[i]);
            a.multiply_scaled(midpoint, work, h.t);
            for (int i = 0; i < h.n; ++i) {
                y[i] = static_cast<double>(static_cast<long double>(b0[i]) +
                    static_cast<long double>(h.t) * b1[i] - work[i]);
            }
            return y;
        }
        if (a.pairs) {
            if (!pair_functions_fast.empty()) {
                for (int i = 0; i < h.n; ++i) {
                    int j = mate[i];
                    if (j < i) continue;
                    const auto& f = pair_functions_fast[i];
                    if (j == i) y[i] = f[0] * b0[i] + f[3] * b1[i];
                    else {
                        y[i] = f[0] * b0[i] + f[1] * b0[j] + f[3] * b1[i] + f[4] * b1[j];
                        y[j] = f[1] * b0[i] + f[2] * b0[j] + f[4] * b1[i] + f[5] * b1[j];
                    }
                }
                return y;
            }
            for (int i = 0; i < h.n; ++i) {
                int j = mate[i];
                if (j < i) continue;
                const auto& f = pair_functions[i];
                if (j == i) y[i] = f[0] * b0[i] + f[3] * b1[i];
                else {
                    y[i] = f[0] * b0[i] + f[1] * b0[j] + f[3] * b1[i] + f[4] * b1[j];
                    y[j] = f[1] * b0[i] + f[2] * b0[j] + f[4] * b1[i] + f[5] * b1[j];
                }
            }
            return y;
        }
        if (dense) {
            std::vector<long double> result(h.n);
            for (int k = 0; k < h.n; ++k) {
                const long double* v = dense_vectors.data() + static_cast<size_t>(k) * h.n;
                long double c0 = 0, c1 = 0;
                for (int i = 0; i < h.n; ++i) {
                    c0 += static_cast<long double>(v[i]) * b0[i];
                    c1 += static_cast<long double>(v[i]) * b1[i];
                }
                long double ck = dense_exp[k] * c0 + dense_phi[k] * c1;
                for (int i = 0; i < h.n; ++i) result[i] += ck * v[i];
            }
            for (int i = 0; i < h.n; ++i) y[i] = static_cast<double>(result[i]);
            return y;
        }
        if (!block_ptr.empty()) {
            for (size_t g = 0; g + 1 < block_ptr.size(); ++g) {
                int begin = block_ptr[g], m = block_ptr[g + 1] - begin;
                size_t start = block_storage[g];
                for (int i = 0; i < m; ++i) {
                    long double value = 0;
                    for (int j = 0; j < m; ++j) {
                        int index = block_order[begin + j];
                        size_t at = start + i + static_cast<size_t>(j) * m;
                        value += block_exp[at] * b0[index] + block_phi[at] * b1[index];
                    }
                    y[block_order[begin + i]] = static_cast<double>(value);
                }
            }
            return y;
        }

        const auto& c = coefficients[0];
        if (h.t * a.bound >= 1200 && h.n >= 5000 &&
                (h.rtol >= 1e-7 || (a.diffusion && !std::getenv("FAM_NO_KRYLOV10")) ||
                    std::getenv("FAM_KRYLOV10")) &&
                !std::getenv("FAM_NO_KRYLOV")) {
            if (norm_inf(b0) == 0 && try_lanczos(b1, true, y, nullptr, 0, 0, true)) return y;
            if (norm_inf(b1) == 0 && try_lanczos(b0, false, y, nullptr, 0, 0, true)) return y;
            if (norm_inf(b0) != 0 && norm_inf(b1) != 0 && !std::getenv("FAM_NO_COMBINED")) {
                Vec ys(h.n);
                double es = 0, ei = 0;
                int cap = c.degree(rhs_norm0, rhs_norm1, h.t,
                    0.04 * (h.atol + h.rtol * output_lower)) / 4;
                if (try_lanczos(b1, true, ys, &es, 0, cap)) {
                    double budget = 0.02 * (h.atol + h.rtol * norm_inf(ys));
                    if (try_lanczos(b0, false, y, &ei, budget, cap)) {
                        for (int i = 0; i < h.n; ++i) y[i] += ys[i];
                        double threshold = h.atol + h.rtol * norm_inf(y);
                        if (es + ei <= 0.15 * threshold) return y;
                    }
                }
            }
        }
        int degree = c.degree(rhs_norm0, rhs_norm1, h.t,
            0.04 * (h.atol + h.rtol * output_lower));
        if (std::getenv("FAM_DEBUG")) {
            std::fprintf(stderr, "n=%d eta=%.7g degree=%d lower-output=%.5g\n",
                h.n, h.t * a.bound, degree, output_lower);
        }
        return apply_expansion(b0, b1, c.ex, c.ph, degree, h.t, a.bound);
    }

    Vec solve_internal(const Vec& b0, const Vec& b1) {
        if (!coefficients.empty()) {
            rhs_norm0 = norm_two(b0);
            rhs_norm1 = norm_two(b1);
            if (rhs_norm0 + h.t * rhs_norm1 <= h.atol * 0.01) return Vec(h.n, 0);
            output_lower = known_output_lower(b0, b1, rhs_norm0, rhs_norm1);
            Vec y;
            if (try_steady(b0, b1, y)) return y;
        }
        if (h.rtol <= 1e-10 && !coefficients.empty()) {
            double initial = norm_inf(b0), source = norm_inf(b1);
            if (std::getenv("FAM_ALL_LONG")) return sensitive_clenshaw(b0, b1);
            if (initial > 0 && source > 0) {
                double residual = 0;
                for (int i = 0; i < h.n; ++i) residual = std::max(residual, std::abs(std::fma(h.t, b1[i], b0[i])));
                if (residual < 1e-5 * initial) {
                    Vec y;
                    if (try_stable_cancellation(b0, b1, y)) return y;
                    return sensitive_clenshaw(b0, b1);
                }
            }
            Vec y = solve_fast(b0, b1);
            if (initial > 0 && source > 0 && norm_inf(y) < 1e-6 * initial) {
                if (try_stable_cancellation(b0, b1, y)) return y;
                return sensitive_clenshaw(b0, b1);
            }
            return y;
        }
        return solve_fast(b0, b1);
    }

    Vec solve(const Vec& b0, const Vec& b1) {
        if (a.gauge.empty()) return solve_internal(b0, b1);
        Vec c0(h.n), c1(h.n);
        for (int i = 0; i < h.n; ++i) {
            c0[i] = a.gauge[i] < 0 ? -b0[i] : b0[i];
            c1[i] = a.gauge[i] < 0 ? -b1[i] : b1[i];
        }
        Vec y = solve_internal(c0, c1);
        for (int i = 0; i < h.n; ++i) if (a.gauge[i] < 0) y[i] = -y[i];
        return y;
    }
};

int main() {
    try {
        write_exact("READY\n", 6);
        char magic[8];
        read_exact(magic, sizeof(magic));
        if (std::memcmp(magic, "FAMEXP2\n", 8)) throw std::runtime_error("invalid magic");
        uint32_t length;
        read_exact(&length, sizeof(length));
        if (length > 4096) throw std::runtime_error("header too long");
        std::string json(length, '\0');
        read_exact(json.data(), json.size());
        Header h(json);
        Solver solver(h);
        Vec b0(h.n), b1(h.n);
        for (int request = 0; request < h.requests; ++request) {
            read_exact(b0.data(), b0.size() * sizeof(double));
            read_exact(b1.data(), b1.size() * sizeof(double));
            Vec y = solver.solve(b0, b1);
            write_exact(y.data(), y.size() * sizeof(double));
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "solver: %s\n", e.what());
        return 1;
    }
}
