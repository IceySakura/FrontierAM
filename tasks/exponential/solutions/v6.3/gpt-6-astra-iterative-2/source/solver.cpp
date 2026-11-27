#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <immintrin.h>
#include <fcntl.h>
#include <lapacke.h>
#include <cblas.h>
#include <unistd.h>

using Vec = std::vector<double>;

extern "C" {
void umfpack_zi_defaults(double *);
int umfpack_zi_symbolic(int, int, const int *, const int *, const double *, const double *,
                        void **, const double *, double *);
int umfpack_zi_numeric(const int *, const int *, const double *, const double *,
                       void *, void **, const double *, double *);
int umfpack_zi_solve(int, const int *, const int *, const double *, const double *,
                     double *, double *, const double *, const double *, void *,
                     const double *, double *);
void umfpack_zi_free_symbolic(void **);
void umfpack_zi_free_numeric(void **);
struct fftw_plan_s;
using fftw_plan = fftw_plan_s *;
using fftw_complex = double[2];
void *fftw_malloc(size_t);
void fftw_free(void *);
fftw_plan fftw_plan_r2r(int, const int *, double *, double *, const int *, unsigned);
fftw_plan fftw_plan_dft(int, const int *, fftw_complex *, fftw_complex *, int, unsigned);
void fftw_execute_r2r(fftw_plan, double *, double *);
void fftw_execute_dft(fftw_plan, fftw_complex *, fftw_complex *);
void fftw_destroy_plan(fftw_plan);
}

static void read_all(void *buffer, size_t bytes) {
    auto *p = static_cast<unsigned char *>(buffer);
    while (bytes) {
        ssize_t got = ::read(STDIN_FILENO, p, bytes);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) throw std::runtime_error("incomplete input");
        p += got;
        bytes -= static_cast<size_t>(got);
    }
}

static void write_all(const void *buffer, size_t bytes) {
    const auto *p = static_cast<const unsigned char *>(buffer);
    while (bytes) {
        ssize_t put = ::write(STDOUT_FILENO, p, bytes);
        if (put < 0 && errno == EINTR) continue;
        if (put <= 0) throw std::runtime_error("output error");
        p += put;
        bytes -= static_cast<size_t>(put);
    }
}

// The protocol's flat JSON object contains only strings and numbers.
struct Header {
    int n = 0, nnz = 0, requests = 0;
    double t = 0, rtol = 0, atol = 0;

    explicit Header(const std::string &json) {
        size_t p = 0;
        auto space = [&]() {
            while (p < json.size() && std::strchr(" \t\r\n", json[p])) ++p;
        };
        auto text = [&]() {
            if (json.at(p++) != '"') throw std::runtime_error("JSON string");
            std::string s;
            while (json.at(p) != '"') {
                char c = json.at(p++);
                if (c == '\\') c = json.at(p++);
                s.push_back(c);
            }
            ++p;
            return s;
        };
        space();
        if (json.at(p++) != '{') throw std::runtime_error("JSON object");
        for (;;) {
            space();
            if (json.at(p) == '}') break;
            std::string key = text();
            space();
            if (json.at(p++) != ':') throw std::runtime_error("JSON colon");
            space();
            if (json.at(p) == '"') {
                text();
            } else {
                char *end = nullptr;
                double x = std::strtod(json.c_str() + p, &end);
                if (end == json.c_str() + p) throw std::runtime_error("JSON number");
                p = static_cast<size_t>(end - json.c_str());
                if (key == "n") n = static_cast<int>(x);
                else if (key == "nnz") nnz = static_cast<int>(x);
                else if (key == "requests") requests = static_cast<int>(x);
                else if (key == "t") t = x;
                else if (key == "rtol") rtol = x;
                else if (key == "atol") atol = x;
            }
            space();
            if (json.at(p) == '}') break;
            if (json.at(p++) != ',') throw std::runtime_error("JSON comma");
        }
        if (n < 1 || nnz < 0 || (requests != 1 && requests != 16))
            throw std::runtime_error("invalid header");
    }
};

template<bool Absolute = false>
static inline double csr_dot(const uint32_t *__restrict columns,
                             const double *__restrict values, uint32_t begin, uint32_t end,
                             const double *__restrict x) {
    double a = 0, b = 0, c = 0, d = 0;
    auto coefficient = [&](uint32_t k) {
        if constexpr (Absolute) return std::abs(values[k]);
        else return values[k];
    };
    uint32_t k = begin;
    for (; k + 4 <= end; k += 4) {
        a += coefficient(k) * x[columns[k]];
        b += coefficient(k + 1) * x[columns[k + 1]];
        c += coefficient(k + 2) * x[columns[k + 2]];
        d += coefficient(k + 3) * x[columns[k + 3]];
    }
    if (k < end) {
        a += coefficient(k) * x[columns[k]];
        ++k;
    }
    if (k < end) {
        b += coefficient(k) * x[columns[k]];
        ++k;
    }
    if (k < end) c += coefficient(k) * x[columns[k]];
    return (a + b) + (c + d);
}

struct Matrix {
    int n;
    std::vector<uint32_t> ptr, col;
    Vec a;
    Vec original;
    Vec diag;
    double bound = 0;
    double time_factor = 1, lognorm = 0;
    bool scaled = false;
    bool pairs = true;
    std::vector<int> partner;

    explicit Matrix(const Header &h) : n(h.n), ptr(n + 1), col(h.nnz),
                                      a(h.nnz), diag(n), partner(n, -1) {
        std::vector<uint64_t> wide(n + 1);
        read_all(wide.data(), wide.size() * sizeof(uint64_t));
        for (int i = 0; i <= n; ++i) ptr[i] = static_cast<uint32_t>(wide[i]);
        read_all(col.data(), col.size() * sizeof(uint32_t));
        read_all(a.data(), a.size() * sizeof(double));
        for (int i = 0; i < n; ++i) {
            long double rowsum = 0;
            for (uint32_t k = ptr[i]; k < ptr[i + 1]; ++k) {
                rowsum += std::abs(static_cast<long double>(a[k]));
                if (col[k] == static_cast<uint32_t>(i)) diag[i] = a[k];
                else if (a[k] != 0) {
                    if (partner[i] >= 0) pairs = false;
                    partner[i] = static_cast<int>(col[k]);
                }
            }
            bound = std::max(bound, static_cast<double>(rowsum * h.t));
            lognorm = std::max(lognorm, static_cast<double>((rowsum * (1 + 4e-14L) - 2 * diag[i]) * h.t));
        }
        bound *= 1 + 1e-12;
    }

    void scale(double t, bool preserve) {
        if (scaled) return;
        scaled = true;
        if (preserve) original = a;
        time_factor = t;
        for (double &x : a) x *= t;
        for (double &x : diag) x *= t;
    }

    void mul(const double *__restrict x, double *__restrict y) const {
        for (int i = 0; i < n; ++i) {
            y[i] = csr_dot(col.data(), a.data(), ptr[i], ptr[i + 1], x);
        }
    }

    double tighten_bound(int iterations) const {
        Vec x(n, 1.0), y(n);
        double upper = bound;
        for (int it = 0; it < iterations; ++it) {
            double next = 0, maxy = 0;
            for (int i = 0; i < n; ++i) {
                double s = csr_dot<true>(col.data(), a.data(), ptr[i], ptr[i + 1], x.data());
                y[i] = s;
                if (x[i] > 0) next = std::max(next, s / x[i]);
                maxy = std::max(maxy, s);
            }
            double previous = upper;
            upper = std::min(upper, next * (1 + 1e-10));
            if (maxy == 0) break;
            if (it >= 3 && upper > 0.995 * previous) break;
            for (int i = 0; i < n; ++i) x[i] = y[i] / maxy + 1e-8;
        }
        return upper;
    }
};

static double maxabs(const Vec &v) {
    double r = 0;
    for (double x : v) r = std::max(r, std::abs(x));
    return r;
}

static double norm2(const Vec &v) {
    double r = 0;
    for (double x : v) r += x * x;
    return std::sqrt(r);
}

static double dot(const double *x, const double *y, int n) {
    return cblas_ddot(n, x, 1, y, 1);
}

static long double phi(long double x) {
    return x == 0 ? 1 : -std::expm1(-x) / x;
}

static std::pair<double, double> scalar_functions(double x, double t) {
    if (x == 0) return {1, t};
    if (x > 40) return {0, t / x};
    double em1 = std::expm1(-x);
    return {1 + em1, -t * em1 / x};
}

struct PairFunctions {
    Vec e0, e1, p0, p1;
    explicit PairFunctions(const Matrix &A, double t)
        : e0(A.n), e1(A.n), p0(A.n), p1(A.n) {
        for (int i = 0; i < A.n; ++i) {
            int j = A.partner[i];
            if (j < 0) {
                auto f = scalar_functions(t * A.diag[i], t);
                e0[i] = f.first;
                p0[i] = f.second;
                continue;
            }
            if (j < i) continue;
            double a = A.diag[i], d = A.diag[j], b = 0;
            for (uint32_t k = A.ptr[i]; k < A.ptr[i + 1]; ++k)
                if (A.col[k] == static_cast<uint32_t>(j)) b = A.a[k];
            double bb = b * b;
            double delta = std::sqrt(std::fma(a - d, a - d, 4 * bb));
            double high = (a + d + delta) / 2;
            double determinant = std::fma(a, d, -bb) + std::fma(-b, b, bb);
            double low = high == 0 ? 0 : determinant / high;
            if (low < 0 && low > -1e-14 * high) low = 0;
            double small_weight = delta == 0 ? 0.5 : 2 * bb / (delta * (delta + std::abs(a - d)));
            double wi = a >= d ? 1 - small_weight : small_weight;
            double wj = a >= d ? small_weight : 1 - small_weight;
            double offweight = delta == 0 ? 0 : b / delta;
            auto fl = scalar_functions(t * low, t);
            auto fh = scalar_functions(t * high, t);
            double el = fl.first, eh = fh.first, pl = fl.second, ph = fh.second;
            e0[i] = std::fma(wj, el, wi * eh);
            e0[j] = std::fma(wi, el, wj * eh);
            e1[i] = e1[j] = (eh - el) * offweight;
            p0[i] = std::fma(wj, pl, wi * ph);
            p0[j] = std::fma(wi, pl, wj * ph);
            p1[i] = p1[j] = (ph - pl) * offweight;
        }
    }
    void solve(const Matrix &A, const Vec &b0, const Vec &b1, Vec &y) const {
        for (int i = 0; i < A.n; ++i) {
            int j = A.partner[i];
            y[i] = e0[i] * b0[i] + p0[i] * b1[i];
            if (j >= 0) y[i] += e1[i] * b0[j] + p1[i] * b1[j];
        }
    }
};

struct TensorSolver {
    int n = 0;
    double t = 0, rtol = 0, normalization = 1;
    bool periodic = false;
    std::vector<int> lengths, strides, modes, opposite;
    std::vector<int8_t> signs;
    Vec exponential, forcing;
    double *work0 = nullptr, *work1 = nullptr;
    fftw_plan forward = nullptr, backward = nullptr;

    ~TensorSolver() {
        if (forward) fftw_destroy_plan(forward);
        if (backward) fftw_destroy_plan(backward);
        if (work0) fftw_free(work0);
        if (work1) fftw_free(work1);
    }

    bool prepare(const Matrix &A, const Header &h) {
        if (A.n < 32 || A.bound < (h.requests > 1 ? 200 : 500)) return false;
        std::vector<int> neighbors;
        for (uint32_t k = A.ptr[0]; k < A.ptr[1]; ++k) {
            if (A.col[k] != 0 && A.a[k] != 0) {
                if (neighbors.size() == 6) return false;
                neighbors.push_back(A.col[k]);
            }
        }
        if (neighbors.empty() || neighbors[0] != 1 || neighbors.size() > 6) return false;
        bool found = false;
        if (neighbors.size() <= 3) {
            strides = neighbors;
            found = A.n % strides.back() == 0;
            for (size_t k = 1; k < strides.size(); ++k)
                found = found && strides[k] % strides[k - 1] == 0;
        }
        if (!found) {
            periodic = true;
            if (neighbors.size() == 2 && neighbors[1] == A.n - 1) {
                strides = {1};
            } else if (neighbors.size() == 4) {
                int width = neighbors[2];
                if (neighbors[1] != width - 1 || neighbors[3] != A.n - width ||
                    A.n % width != 0) return false;
                strides = {1, width};
            } else if (neighbors.size() == 6) {
                int width = neighbors[2], plane = neighbors[4];
                if (neighbors[1] != width - 1 || neighbors[3] != plane - width ||
                    neighbors[5] != A.n - plane || plane % width != 0 || A.n % plane != 0)
                    return false;
                strides = {1, width, plane};
            } else return false;
        }
        n = A.n;
        t = h.t;
        rtol = h.rtol;
        int rank = static_cast<int>(strides.size());
        lengths.resize(rank);
        modes.resize(rank);
        Vec weights(rank);
        for (int d = 0; d < rank; ++d) {
            lengths[d] = (d + 1 < rank ? strides[d + 1] : n) / strides[d];
            if (lengths[d] < (periodic ? 3 : 2)) return false;
            for (uint32_t k = A.ptr[0]; k < A.ptr[1]; ++k)
                if (A.col[k] == static_cast<uint32_t>(strides[d])) weights[d] = std::abs(A.a[k]);
            if (weights[d] == 0) return false;
            modes[d] = periodic ? 2 : 0;
            if (!periodic && lengths[d] > 2) {
                long double difference = static_cast<long double>(A.diag[strides[d]]) - A.diag[0];
                if (difference == weights[d]) modes[d] = 1;
                else if (difference != 0) return false;
            }
        }
        signs.assign(n, 0);
        signs[0] = 1;
        for (int i = 0; i < n; ++i) {
            std::array<int, 6> wanted{}, axis{};
            int count = 0;
            long double diagonal_change = 0;
            for (int d = 0; d < rank; ++d) {
                int coordinate = i / strides[d] % lengths[d];
                if (periodic || coordinate > 0) {
                    wanted[count] = coordinate > 0 ? i - strides[d] : i + (lengths[d] - 1) * strides[d];
                    axis[count++] = d;
                }
                if (periodic || coordinate + 1 < lengths[d]) {
                    wanted[count] = coordinate + 1 < lengths[d] ? i + strides[d] : i - (lengths[d] - 1) * strides[d];
                    axis[count++] = d;
                }
                if (modes[d] == 1 && coordinate > 0 && coordinate + 1 < lengths[d])
                    diagonal_change += weights[d];
            }
            if (static_cast<long double>(A.diag[i]) - A.diag[0] != diagonal_change) return false;
            if (signs[i] == 0) signs[i] = 1;
            int seen = 0;
            for (uint32_t k = A.ptr[i]; k < A.ptr[i + 1]; ++k) {
                int j = A.col[k];
                if (j == i || A.a[k] == 0) continue;
                int match = 0;
                while (match < count && wanted[match] != j) ++match;
                if (match == count || std::abs(A.a[k]) != weights[axis[match]]) return false;
                int sign = A.a[k] < 0 ? signs[i] : -signs[i];
                if (signs[j] == 0) signs[j] = static_cast<int8_t>(sign);
                else if (signs[j] != sign) return false;
                ++seen;
            }
            if (seen != count) return false;
        }
        std::vector<std::vector<long double>> axis_values(rank);
        long double pi = std::acos(-1.0L);
        for (int d = 0; d < rank; ++d) {
            axis_values[d].resize(lengths[d]);
            for (int k = 0; k < lengths[d]; ++k) {
                long double angle;
                if (modes[d] == 0) angle = pi * (k + 1) / (lengths[d] + 1);
                else if (modes[d] == 1) angle = pi * k / lengths[d];
                else angle = 2 * pi * std::min(k, lengths[d] - k) / lengths[d];
                axis_values[d][k] = -2 * static_cast<long double>(weights[d]) * std::cos(angle)
                                   + (modes[d] == 1 ? weights[d] : 0);
            }
        }
        exponential.resize(n);
        forcing.resize(n);
        if (periodic) opposite.resize(n);
        for (int i = 0; i < n; ++i) {
            long double lambda = A.diag[0];
            int reverse = 0;
            for (int d = 0; d < rank; ++d) {
                int coordinate = i / strides[d] % lengths[d];
                lambda += axis_values[d][coordinate];
                if (periodic) reverse += (coordinate == 0 ? 0 : lengths[d] - coordinate) * strides[d];
            }
            lambda *= t;
            if (lambda < -1e-12L * A.bound) return false;
            lambda = std::max(0.0L, lambda);
            exponential[i] = std::exp(-static_cast<double>(lambda));
            forcing[i] = static_cast<double>(t * (lambda > 50 ? 1 / lambda : phi(lambda)));
            if (periodic) opposite[i] = reverse;
        }
        std::vector<int> dimensions(lengths.rbegin(), lengths.rend());
        work0 = static_cast<double *>(fftw_malloc(sizeof(double) * n * (periodic ? 2 : 1)));
        work1 = static_cast<double *>(fftw_malloc(sizeof(double) * n * (periodic ? 2 : 1)));
        if (!work0 || !work1) return false;
        if (periodic) {
            forward = fftw_plan_dft(rank, dimensions.data(), reinterpret_cast<fftw_complex *>(work0),
                                    reinterpret_cast<fftw_complex *>(work0), -1, 64);
            backward = fftw_plan_dft(rank, dimensions.data(), reinterpret_cast<fftw_complex *>(work1),
                                     reinterpret_cast<fftw_complex *>(work1), 1, 64);
            normalization = 1.0 / n;
        } else {
            std::vector<int> forward_kinds(rank), backward_kinds(rank);
            for (int d = 0; d < rank; ++d) {
                int axis = rank - 1 - d;
                forward_kinds[d] = modes[axis] == 0 ? 7 : 5;
                backward_kinds[d] = modes[axis] == 0 ? 7 : 4;
                normalization /= 2.0 * (lengths[axis] + (modes[axis] == 0 ? 1 : 0));
            }
            forward = fftw_plan_r2r(rank, dimensions.data(), work0, work0, forward_kinds.data(), 64);
            backward = fftw_plan_r2r(rank, dimensions.data(), work0, work0, backward_kinds.data(), 64);
        }
        return forward && backward;
    }

    bool solve(const Vec &b0, const Vec &b1, Vec &y) {
        double norm0 = maxabs(b0), norm1 = maxabs(b1);
        if (norm0 == 0 && norm1 == 0) {
            std::fill(y.begin(), y.end(), 0);
            return true;
        }
        if (periodic) {
            for (int i = 0; i < n; ++i) {
                work0[2 * i] = signs[i] * b0[i];
                work0[2 * i + 1] = signs[i] * b1[i];
            }
            fftw_execute_dft(forward, reinterpret_cast<fftw_complex *>(work0),
                             reinterpret_cast<fftw_complex *>(work0));
            for (int i = 0; i < n; ++i) {
                int j = opposite[i];
                double r0 = (work0[2*i] + work0[2*j]) * 0.5;
                double i0 = (work0[2*i+1] - work0[2*j+1]) * 0.5;
                double r1 = (work0[2*i+1] + work0[2*j+1]) * 0.5;
                double i1 = (work0[2*j] - work0[2*i]) * 0.5;
                work1[2*i] = exponential[i] * r0 + forcing[i] * r1;
                work1[2*i+1] = exponential[i] * i0 + forcing[i] * i1;
            }
            fftw_execute_dft(backward, reinterpret_cast<fftw_complex *>(work1),
                             reinterpret_cast<fftw_complex *>(work1));
            for (int i = 0; i < n; ++i) y[i] = signs[i] * normalization * work1[2*i];
        } else {
            for (int i = 0; i < n; ++i) {
                work0[i] = signs[i] * b0[i];
                work1[i] = signs[i] * b1[i];
            }
            if (norm0 != 0) fftw_execute_r2r(forward, work0, work0);
            if (norm1 != 0) fftw_execute_r2r(forward, work1, work1);
            for (int i = 0; i < n; ++i) work0[i] = exponential[i] * work0[i] + forcing[i] * work1[i];
            fftw_execute_r2r(backward, work0, work0);
            for (int i = 0; i < n; ++i) y[i] = signs[i] * normalization * work0[i];
        }
        return rtol >= 1e-9 || norm0 == 0 || norm1 == 0 ||
               maxabs(y) >= 1e-7 * (norm0 + t * norm1);
    }
};

struct SmallBlocks {
    std::vector<int> nodes, starts;
    std::vector<size_t> offsets;
    std::vector<long double> vectors;
    Vec exponential, forcing;

    bool prepare(const Matrix &A, const Header &h) {
        if (A.bound < (h.requests > 1 ? 50 : 1000)) return false;
        int max_block = A.n < 5000 ? 32 : (h.requests > 1 ? 16 : 8);
        std::vector<int> parent(A.n, -1);
        auto root = [&](int i) {
            int j = i;
            while (parent[j] >= 0) j = parent[j];
            while (i != j) {
                int next = parent[i];
                parent[i] = j;
                i = next;
            }
            return j;
        };
        for (int i = 0; i < A.n; ++i) {
            for (uint32_t k = A.ptr[i]; k < A.ptr[i + 1]; ++k) {
                int j = A.col[k];
                if (j >= i || A.a[k] == 0) continue;
                int r = root(i), s = root(j);
                if (r == s) continue;
                if (parent[r] > parent[s]) std::swap(r, s);
                parent[r] += parent[s];
                if (parent[r] < -max_block) return false;
                parent[s] = r;
            }
        }
        std::vector<int> block(A.n), position(A.n), local(A.n);
        starts.push_back(0);
        for (int i = 0; i < A.n; ++i) {
            if (parent[i] < 0) {
                block[i] = static_cast<int>(starts.size()) - 1;
                starts.push_back(starts.back() - parent[i]);
            }
        }
        nodes.resize(A.n);
        std::copy(starts.begin(), starts.end() - 1, position.begin());
        for (int i = 0; i < A.n; ++i) {
            int b = block[root(i)];
            local[i] = position[b] - starts[b];
            nodes[position[b]++] = i;
        }
        offsets.push_back(0);
        for (size_t b = 0; b + 1 < starts.size(); ++b) {
            int m = starts[b + 1] - starts[b];
            offsets.push_back(offsets.back() + static_cast<size_t>(m) * m);
        }
        vectors.resize(offsets.back());
        exponential.resize(A.n);
        forcing.resize(A.n);
        for (size_t b = 0; b + 1 < starts.size(); ++b) {
            int begin = starts[b], m = starts[b + 1] - begin;
            std::array<long double, 1024> dense{};
            long double *q = vectors.data() + offsets[b];
            long double norm = 0;
            for (int i = 0; i < m; ++i) {
                int row = nodes[begin + i];
                q[i * m + i] = 1;
                for (uint32_t k = A.ptr[row]; k < A.ptr[row + 1]; ++k) {
                    int col = A.col[k];
                    if (A.a[k] == 0) continue;
                    dense[i * m + local[col]] = static_cast<long double>(h.t) * A.a[k];
                }
                norm = std::max(norm, std::abs(dense[i * m + i]));
            }
            // Cyclic Jacobi keeps tiny eigenvalues accurate in the local blocks.
            for (int sweep = 0; sweep < 40; ++sweep) {
                long double largest = 0;
                for (int i = 0; i < m; ++i) {
                    for (int j = i + 1; j < m; ++j) {
                        long double aij = dense[i * m + j];
                        largest = std::max(largest, std::abs(aij));
                        if (std::abs(aij) <= norm * 2e-21L) continue;
                        long double tau = (dense[j * m + j] - dense[i * m + i]) / (2 * aij);
                        long double tangent = std::copysign(1.0L, tau)
                            / (std::abs(tau) + std::sqrt(1 + tau * tau));
                        long double c = 1 / std::sqrt(1 + tangent * tangent), s = tangent * c;
                        dense[i * m + i] -= tangent * aij;
                        dense[j * m + j] += tangent * aij;
                        dense[i * m + j] = dense[j * m + i] = 0;
                        for (int k = 0; k < m; ++k) {
                            if (k != i && k != j) {
                                long double aki = dense[k * m + i], akj = dense[k * m + j];
                                dense[k * m + i] = dense[i * m + k] = c * aki - s * akj;
                                dense[k * m + j] = dense[j * m + k] = s * aki + c * akj;
                            }
                            long double qi = q[k * m + i], qj = q[k * m + j];
                            q[k * m + i] = c * qi - s * qj;
                            q[k * m + j] = s * qi + c * qj;
                        }
                    }
                }
                if (largest <= norm * 2e-21L) break;
            }
            for (int i = 0; i < m; ++i) {
                double lambda = static_cast<double>(std::max(0.0L, dense[i * m + i]));
                auto f = scalar_functions(lambda, h.t);
                exponential[begin + i] = f.first;
                forcing[begin + i] = f.second;
            }
        }
        return true;
    }

    void solve(const Vec &b0, const Vec &b1, Vec &y) const {
        for (size_t b = 0; b + 1 < starts.size(); ++b) {
            int begin = starts[b], m = starts[b + 1] - begin;
            const long double *q = vectors.data() + offsets[b];
            std::array<long double, 32> projection{};
            for (int j = 0; j < m; ++j) {
                long double x0 = 0, x1 = 0;
                for (int i = 0; i < m; ++i) {
                    int node = nodes[begin + i];
                    x0 += q[i * m + j] * b0[node];
                    x1 += q[i * m + j] * b1[node];
                }
                projection[j] = exponential[begin + j] * x0 + forcing[begin + j] * x1;
            }
            for (int i = 0; i < m; ++i) {
                long double sum = 0;
                for (int j = 0; j < m; ++j) sum += q[i * m + j] * projection[j];
                y[nodes[begin + i]] = static_cast<double>(sum);
            }
        }
    }
};

static bool null_action(const Matrix &A, const Header &h, const Vec &b0,
                        const Vec &b1, Vec &y) {
    double magnitude0 = std::abs(b0[0]), magnitude1 = std::abs(b1[0]);
    for (int i = 1; i < A.n; ++i)
        if (std::abs(b0[i]) != magnitude0 || std::abs(b1[i]) != magnitude1) return false;
    const Vec &values = A.original.empty() ? A.a : A.original;
    long double factor = A.original.empty() ? 1 : A.time_factor;
    long double maximum0 = 0, maximum1 = 0, squared0 = 0, squared1 = 0;
    double ynorm = 0;
    for (int i = 0; i < A.n; ++i) {
        long double v0 = 0, v1 = 0;
        for (uint32_t k = A.ptr[i]; k < A.ptr[i + 1]; ++k) {
            v0 += static_cast<long double>(values[k]) * b0[A.col[k]];
            v1 += static_cast<long double>(values[k]) * b1[A.col[k]];
        }
        v0 *= factor;
        v1 *= factor;
        maximum0 = std::max(maximum0, std::abs(v0));
        maximum1 = std::max(maximum1, std::abs(v1));
        squared0 += v0 * v0;
        squared1 += v1 * v1;
        y[i] = static_cast<double>(static_cast<long double>(b0[i])
                                  + static_cast<long double>(h.t) * b1[i]);
        ynorm = std::max(ynorm, std::abs(y[i]));
    }
    long double error = A.lognorm < 1e-6
        ? std::exp(A.lognorm) * (maximum0 + h.t * maximum1 / 2)
        : std::sqrt(squared0) + h.t * std::sqrt(squared1) / 2;
    return error < 0.02 * (h.atol + h.rtol * ynorm);
}

static void taylor(Matrix &A, const Header &h, const Vec &b0,
                   const Vec &b1, Vec &y) {
    double norm0 = 0, norm1 = 0, ynorm = 0;
    double margin = h.rtol < 1e-9 ? 0.5 : 0.8;
    for (int i = 0; i < A.n; ++i) {
        norm0 = std::max(norm0, std::abs(b0[i]));
        norm1 = std::max(norm1, std::abs(b1[i]));
        y[i] = std::fma(h.t, b1[i], b0[i]);
        ynorm = std::max(ynorm, std::abs(y[i]));
    }
    double eta = A.bound;
    if (eta < 2) {
        double error = eta / (1 - eta / 2) * norm0
            + h.t * eta / (2 * (1 - eta / 3)) * norm1;
        if (error < margin * (h.atol + h.rtol * ynorm)) return;
    }
    A.scale(h.t, false);
    Vec term(A.n), next(A.n);
    if (norm0 != 0) A.mul(b0.data(), term.data());
    double termnorm = 0;
    ynorm = 0;
    for (int i = 0; i < A.n; ++i) {
        term[i] = std::fma(h.t, b1[i], -term[i]);
        y[i] = b0[i] + term[i];
        termnorm = std::max(termnorm, std::abs(term[i]));
        ynorm = std::max(ynorm, std::abs(y[i]));
    }
    if (eta < 3 && termnorm * eta / (2 * (1 - eta / 3))
                         < margin * (h.atol + h.rtol * ynorm)) return;
    for (int k = 2; k < 100; ++k) {
        double term_max = 0, y_max = 0;
        for (int i = 0; i < A.n; ++i) {
            double sum = csr_dot(A.col.data(), A.a.data(), A.ptr[i], A.ptr[i + 1], term.data());
            next[i] = -sum / k;
            y[i] += next[i];
            term_max = std::max(term_max, std::abs(next[i]));
            y_max = std::max(y_max, std::abs(y[i]));
        }
        term.swap(next);
        double ratio = A.bound / (k + 1);
        if (ratio < 1 && term_max * ratio / (1 - ratio) <
                             margin * (h.atol + h.rtol * y_max)) break;
    }
}

struct Chebyshev {
    const Matrix &A;
    const Header &h;
    double upper;
    Vec ce, cp, te, tp, scaled, cd;
    std::vector<long double> ce_high, cp_high;
    std::vector<uint32_t> cptr, ccol;
    Vec sell;
    std::vector<uint32_t> sellcol, sellptr;
    int kernel = 0;

    Chebyshev(const Matrix &matrix, const Header &header)
        : A(matrix), h(header), upper(A.bound) {
        if (upper > 200 || (h.requests > 1 && upper > 50))
            upper = A.tighten_bound(upper > 5000 || h.requests > 1 ? 10 : 5);
        kernel = (A.n >= 512 && (upper > 200 || (h.requests > 1 && upper > 5))) ? 2 : 0;
        long double s = static_cast<long double>(upper) / 2;
        int m = static_cast<int>(std::ceil(std::sqrt(100 * s) + 80));
        std::vector<long double> ratios(m + 2), bess(m + 2);
        for (int k = m; k >= 1; --k)
            ratios[k] = 1 / (2 * k / s + ratios[k + 1]);
        bess[0] = 1;
        long double total = 1;
        for (int k = 1; k <= m; ++k) {
            bess[k] = bess[k - 1] * ratios[k];
            total += 2 * bess[k];
        }
        for (auto &v : bess) v /= total;
        ce.resize(m + 1);
        cp.resize(m + 1);
        te.resize(m + 1);
        tp.resize(m + 1);
        if (h.rtol < 1e-9) {
            ce_high.resize(m + 1);
            cp_high.resize(m + 1);
        }
        for (int k = 0; k <= m; ++k) {
            ce[k] = static_cast<double>((k == 0 ? 1 : 2) * bess[k]);
            if (!ce_high.empty()) ce_high[k] = (k == 0 ? 1 : 2) * bess[k];
        }
        long double tail = 0, weighted = 0;
        for (int k = m; k >= 0; --k) {
            weighted += tail;
            cp[k] = static_cast<double>((k == 0 ? 2 : 4) * weighted / s);
            if (!cp_high.empty()) cp_high[k] = (k == 0 ? 2 : 4) * weighted / s;
            tail += bess[k];
        }
        long double se = 0, sp = 0;
        for (int k = m; k >= 0; --k) {
            te[k] = static_cast<double>(se);
            tp[k] = static_cast<double>(sp);
            se += ce[k];
            sp += cp[k];
        }
        ce[0] += static_cast<double>(1 - se);
        cp[0] += static_cast<double>(1 - sp);
        if (!ce_high.empty()) {
            long double total_e = 0, total_p = 0;
            for (int k = m; k >= 0; --k) {
                total_e += ce_high[k];
                total_p += cp_high[k];
            }
            ce_high[0] += 1 - total_e;
            cp_high[0] += 1 - total_p;
        }
        if (kernel == 0) {
            scaled.resize(A.a.size());
            for (size_t k = 0; k < A.a.size(); ++k) scaled[k] = -4 * A.a[k] / upper;
            return;
        }
        cd.resize(A.n);
        cptr.resize(A.n + 1);
        scaled.reserve(A.a.size());
        ccol.reserve(A.a.size());
        for (int i = 0; i < A.n; ++i) {
            cptr[i] = static_cast<uint32_t>(scaled.size());
            cd[i] = 2 - 4 * A.diag[i] / upper;
            for (uint32_t k = A.ptr[i]; k < A.ptr[i + 1]; ++k) {
                if (A.col[k] == static_cast<uint32_t>(i) && kernel != 0) continue;
                scaled.push_back(-4 * A.a[k] / upper);
                ccol.push_back(A.col[k]);
            }
        }
        cptr[A.n] = static_cast<uint32_t>(scaled.size());
        if (kernel == 2) {
            int groups = A.n / 4;
            sellptr.resize(groups + 1);
            for (int g = 0; g < groups; ++g) {
                uint32_t width = 0;
                for (int lane = 0; lane < 4; ++lane) {
                    int i = 4 * g + lane;
                    width = std::max(width, cptr[i + 1] - cptr[i]);
                }
                sellptr[g + 1] = sellptr[g] + 4 * width;
            }
            if (sellptr.back() > scaled.size() * 1.12) {
                kernel = 1;
            } else {
                sell.resize(sellptr.back());
                sellcol.resize(sellptr.back());
                for (int g = 0; g < groups; ++g) {
                    for (int lane = 0; lane < 4; ++lane) {
                        int i = 4 * g + lane;
                        for (uint32_t j = cptr[i]; j < cptr[i + 1]; ++j) {
                            uint32_t pos = sellptr[g] + 4 * (j - cptr[i]) + lane;
                            sell[pos] = scaled[j];
                            sellcol[pos] = ccol[j];
                        }
                    }
                }
            }
        }
    }

    int degree(double n0, double n1, double target) const {
        int m = 0;
        while (m + 1 < static_cast<int>(te.size()) && n0 * te[m] + n1 * tp[m] > target)
            ++m;
        return m;
    }

    template<bool B0, bool B1>
    void evaluate(int m, const Vec &b0, const Vec &b1, Vec &y) const {
        Vec u(A.n), v(A.n), w(A.n);
        const auto &row = kernel == 0 ? A.ptr : cptr;
        const auto &col = kernel == 0 ? A.col : ccol;
        for (int i = 0; i < A.n; ++i) {
            if constexpr (B0) u[i] = ce[m] * b0[i];
            if constexpr (B1) u[i] += h.t * cp[m] * b1[i];
        }
        for (int k = m - 1; k >= 0; --k) {
            double ec = ce[k], pc = h.t * cp[k];
            double factor = k == 0 ? 0.5 : 1.0;
            int begin = 0;
            if (kernel == 2) {
                const __m256d vf = _mm256_set1_pd(factor);
                const __m256d ve = _mm256_set1_pd(ec);
                const __m256d vp = _mm256_set1_pd(pc);
                for (int g = 0; g < A.n / 4; ++g) {
                    int i = 4 * g;
                    __m256d sum = _mm256_mul_pd(_mm256_loadu_pd(cd.data() + i),
                                                _mm256_loadu_pd(u.data() + i));
                    for (uint32_t j = sellptr[g]; j < sellptr[g + 1]; j += 4) {
                        __m128i ix = _mm_loadu_si128(reinterpret_cast<const __m128i *>(sellcol.data() + j));
                        __m256d x = _mm256_i32gather_pd(u.data(), ix, 8);
                        sum = _mm256_fmadd_pd(_mm256_loadu_pd(sell.data() + j), x, sum);
                    }
                    __m256d value = _mm256_fmsub_pd(vf, sum, _mm256_loadu_pd(v.data() + i));
                    if constexpr (B0)
                        value = _mm256_fmadd_pd(ve, _mm256_loadu_pd(b0.data() + i), value);
                    if constexpr (B1)
                        value = _mm256_fmadd_pd(vp, _mm256_loadu_pd(b1.data() + i), value);
                    _mm256_storeu_pd(w.data() + i, value);
                }
                begin = A.n / 4 * 4;
            }
            for (int i = begin; i < A.n; ++i) {
                double sum = csr_dot(col.data(), scaled.data(), row[i], row[i + 1], u.data());
                double value = factor * (sum + (kernel == 0 ? 2 : cd[i]) * u[i]) - v[i];
                if constexpr (B0) value += ec * b0[i];
                if constexpr (B1) value += pc * b1[i];
                w[i] = value;
            }
            v.swap(u);
            u.swap(w);
        }
        y.swap(u);
    }

    void solve(const Vec &b0, const Vec &b1, Vec &y) const {
        double n0 = norm2(b0), n1 = h.t * norm2(b1);
        double estimate = 0;
        for (int i = 0; i < A.n; ++i)
            estimate = std::max(estimate, (std::abs(b0[i]) + h.t * std::abs(b1[i])) / (1 + A.diag[i]));
        double target = 0.02 * (h.atol + h.rtol * estimate);
        int m = degree(n0, n1, target);
        for (int attempt = 0; attempt < 3; ++attempt) {
            if (std::getenv("PROP_DIAGNOSTIC"))
                std::fprintf(stderr, "chebyshev upper=%.8g degree=%d attempt=%d kernel=%d\n",
                             upper, m, attempt, kernel);
            if (n0 == 0) evaluate<false, true>(m, b0, b1, y);
            else if (n1 == 0) evaluate<true, false>(m, b0, b1, y);
            else evaluate<true, true>(m, b0, b1, y);
            double tolerance = h.atol + h.rtol * maxabs(y);
            if (n0 * te[m] + n1 * tp[m] < 0.15 * tolerance) break;
            m = degree(n0, n1, 0.08 * tolerance);
        }
    }

    bool cancellation_suspect(const Vec &b0, const Vec &b1, const Vec &y) const {
        if (h.rtol >= 1e-9 || maxabs(b0) == 0 || maxabs(b1) == 0) return false;
        double estimate = 0;
        for (int i = 0; i < A.n; ++i)
            estimate = std::max(estimate, (std::abs(b0[i]) + h.t * std::abs(b1[i])) / (1 + A.diag[i]));
        return maxabs(y) < 1e-3 * estimate;
    }

    void solve_precise(const Vec &b0, const Vec &b1, Vec &y) const {
        double n0 = norm2(b0), n1 = h.t * norm2(b1);
        int m = degree(n0, n1, 0.005 * (h.atol + h.rtol * maxabs(y)));
        const Vec &data = A.original.empty() ? A.a : A.original;
        long double factor = -4.0L * (A.original.empty() ? 1 : h.t) / upper;
        std::vector<long double> values(data.size()), u(A.n), v(A.n), w(A.n);
        for (size_t k = 0; k < data.size(); ++k) values[k] = factor * data[k];
        for (int i = 0; i < A.n; ++i)
            u[i] = ce_high[m] * b0[i] + h.t * cp_high[m] * b1[i];
        for (int k = m - 1; k >= 0; --k) {
            long double ec = ce_high[k], pc = h.t * cp_high[k];
            long double f = k == 0 ? 0.5L : 1;
            for (int i = 0; i < A.n; ++i) {
                long double sum = 0;
                for (uint32_t j = A.ptr[i]; j < A.ptr[i + 1]; ++j)
                    sum += values[j] * u[A.col[j]];
                w[i] = f * (sum + 2 * u[i]) - v[i] + ec * b0[i] + pc * b1[i];
            }
            v.swap(u);
            u.swap(w);
        }
        for (int i = 0; i < A.n; ++i) y[i] = static_cast<double>(u[i]);
        if (std::getenv("PROP_DIAGNOSTIC"))
            std::fprintf(stderr, "extended precision degree=%d\n", m);
    }
};

struct Lanczos {
    const Matrix &A;
    const Header &h;
    explicit Lanczos(const Matrix &matrix, const Header &header) : A(matrix), h(header) {}

    double error_log(const Vec &diagonal, const Vec &off, int m,
                     double norm, bool forcing, double boundary_norm) const {
        double best = std::numeric_limits<double>::infinity();
        double numerator = std::log(norm) + std::log(boundary_norm);
        for (int i = 0; i + 1 < m; ++i) numerator += std::log(off[i]);
        for (double shift : {0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0,
                             8.0, 16.0, 32.0, 64.0, 128.0}) {
            long double d = diagonal[0] + shift;
            if (d <= 0) continue;
            long double denominator = std::log(d);
            bool valid = true;
            for (int i = 1; i < m; ++i) {
                d = static_cast<long double>(diagonal[i]) + shift
                    - static_cast<long double>(off[i - 1]) * off[i - 1] / d;
                if (d <= 0) {
                    valid = false;
                    break;
                }
                denominator += std::log(d);
            }
            if (!valid) continue;
            double factor = forcing ? (shift <= 1 ? 0 : shift - 1 - std::log(shift)) : shift;
            best = std::min(best, numerator - static_cast<double>(denominator) + factor);
        }
        return best;
    }

    bool solve(const Vec &b, bool forcing, int maxm, Vec &y, bool adaptive = false,
               double requested_scale = -1, const Vec *addend = nullptr,
               double target_scale = -1) const {
        double bn = norm2(b);
        if (bn == 0) {
            if (addend) y = *addend;
            else std::fill(y.begin(), y.end(), 0.0);
            return true;
        }
        double scale = requested_scale >= 0 ? requested_scale : (forcing ? h.t : 1);
        double target = 0.02 * (h.atol + h.rtol *
                                      (target_scale >= 0 ? target_scale : scale * maxabs(b)));
        maxm = std::max(2, std::min(maxm, A.n + 8));
        std::unique_ptr<double[]> basis(new double[static_cast<size_t>(A.n) * (maxm + 1)]);
        double *Q = basis.get();
        for (int i = 0; i < A.n; ++i) Q[i] = b[i] / bn;
        Vec alpha(maxm), beta(maxm), work(A.n);
        double first_error = 0;
        for (int k = 0; k < maxm; ++k) {
            double *q = Q + static_cast<size_t>(k) * A.n;
            double *prev = k ? q - A.n : q;
            A.mul(q, work.data());
            double a = dot(q, work.data(), A.n);
            double bp = k ? beta[k - 1] : 0;
            for (int i = 0; i < A.n; ++i)
                work[i] -= a * q[i] + bp * prev[i];
            double correction = dot(q, work.data(), A.n);
            a += correction;
            for (int i = 0; i < A.n; ++i) work[i] -= correction * q[i];
            alpha[k] = a;
            beta[k] = std::sqrt(dot(work.data(), work.data(), A.n));
            int m = k + 1;
            bool breakdown = beta[k] < 1e-28;
            if (m % 8 == 0 || breakdown || m == maxm) {
                double boundary = beta[k];
                if (A.lognorm < 8) boundary = std::min(boundary, std::exp(A.lognorm) * maxabs(work));
                double error = breakdown ? -1e300 : error_log(alpha, beta, m, bn * scale, forcing, boundary);
                if (m == 32) first_error = error;
                if (adaptive && m == 128 && first_error - error < 4 && maxm > 256)
                    return false;
                if (std::getenv("PROP_DIAGNOSTIC") && (m % 32 == 0 || m == maxm))
                    std::fprintf(stderr, "lanczos m=%d logerror=%.5g target=%.5g\n",
                                 m, error, std::log(target));
                if (error < std::log(target) || breakdown) {
                    Vec d(alpha.begin(), alpha.begin() + m);
                    Vec e(beta.begin(), beta.begin() + m);
                    Vec values(m), vectors(static_cast<size_t>(m) * m);
                    std::vector<int> support(2 * m);
                    lapack_int count = 0;
                    int info = LAPACKE_dstevr(LAPACK_COL_MAJOR, 'V', 'A', m, d.data(), e.data(),
                                             0, 0, 0, 0, 0, &count, values.data(),
                                             vectors.data(), m, support.data());
                    if (info || count != m) return false;
                    Vec weights(m), coefficients(m);
                    for (int j = 0; j < m; ++j) {
                        double val = values[j];
                        double f = forcing ? static_cast<double>(phi(val)) : std::exp(-val);
                        weights[j] = vectors[static_cast<size_t>(j) * m] * f * bn * scale;
                    }
                    cblas_dgemv(CblasColMajor, CblasNoTrans, m, m, 1.0, vectors.data(), m,
                                weights.data(), 1, 0.0, coefficients.data(), 1);
                    cblas_dgemv(CblasColMajor, CblasNoTrans, A.n, m, 1.0, Q, A.n,
                                coefficients.data(), 1, 0.0, y.data(), 1);
                    if (addend) {
                        for (int i = 0; i < A.n; ++i)
                            y[i] = static_cast<double>(static_cast<long double>(y[i]) + (*addend)[i]);
                    }
                    double tolerance = h.atol + h.rtol * maxabs(y);
                    if (error < std::log(0.15 * tolerance) || breakdown) {
                        if (std::getenv("PROP_DIAGNOSTIC"))
                            std::fprintf(stderr, "lanczos converged m=%d error=%.5g tolerance=%.5g\n",
                                         m, std::exp(error), tolerance);
                        return true;
                    }
                    target = 0.08 * tolerance;
                }
            }
            if (k + 1 < maxm) {
                double *next = q + A.n;
                for (int i = 0; i < A.n; ++i) next[i] = work[i] / beta[k];
            }
        }
        return false;
    }
};

static bool lanczos_candidate(const Matrix &A) {
    if (A.n < 5000 || A.bound < 15000) return false;
    Vec diagonal = A.diag;
    double sum = 0, maximum = 0;
    for (double x : diagonal) {
        sum += x;
        maximum = std::max(maximum, x);
    }
    double mean = sum / A.n;
    if (maximum < 2 * mean) return false;
    size_t index = diagonal.size() / 20;
    std::nth_element(diagonal.begin(), diagonal.begin() + index, diagonal.end());
    return diagonal[index] > 0.08 * mean;
}

using Complex = std::complex<double>;
using CVec = std::vector<Complex>;

struct CramCoefficients {
    static const std::array<Complex, 8> &poles() {
        static const std::array<Complex, 8> theta = {{
            {-10.843917078696988026, 19.277446167181652284},
            {-5.2649713434426468895, 16.220221473167927305},
            {5.9481522689511774808, 3.5874573620183222829},
            {3.5091036084149180974, 8.4361989858843750826},
            {6.4161776990994341923, 1.1941223933701386874},
            {1.4193758971856659786, 10.925363484496722585},
            {4.9931747377179963991, 5.9968817136039422260},
            {-1.4139284624888862114, 13.497725698892745389}
        }};
        return theta;
    }

    static const std::array<Complex, 8> &residues() {
        static const std::array<Complex, 8> alpha = {{
            {-5.0901521865224915650e-7, -2.4220017652852287970e-5},
            {2.1151742182466030907e-4, 4.3892969647380673918e-3},
            {1.1339775178483930527e2, 1.0194721704215856450e2},
            {1.5059585270023467528e1, -5.7514052776421819979},
            {-6.4500878025539646595e1, -2.2459440762652096056e2},
            {-1.4793007113557999718, 1.7686588323782937906},
            {-6.2518392463207918892e1, -1.1190391094283228480e1},
            {4.1023136835410021273e-2, -1.5743466173455468191e-1}
        }};
        return alpha;
    }

};

struct DirectCram {
    const Matrix &A;
    const Header &h;
    std::vector<int> ptr, col, diagpos;
    CVec data;
    std::array<void *, 8> numeric{};
    std::array<double, 20> control{};
    void *symbolic = nullptr;

    DirectCram(const Matrix &matrix, const Header &header) : A(matrix), h(header) {}
    ~DirectCram() {
        for (void *&factor : numeric) if (factor) umfpack_zi_free_numeric(&factor);
        if (symbolic) umfpack_zi_free_symbolic(&symbolic);
    }

    void set_shift(Complex theta) {
        for (int i = 0; i < A.n; ++i) data[diagpos[i]] = A.diag[i] + theta;
    }

    bool prepare(double baseline, int remaining) {
        ptr.resize(A.n + 1);
        diagpos.resize(A.n);
        col.reserve(A.a.size() + A.n);
        data.reserve(A.a.size() + A.n);
        for (int i = 0; i < A.n; ++i) {
            ptr[i] = static_cast<int>(col.size());
            bool diagonal = false;
            for (uint32_t k = A.ptr[i]; k < A.ptr[i + 1]; ++k) {
                int j = A.col[k];
                if (!diagonal && j >= i) {
                    diagpos[i] = static_cast<int>(col.size());
                    col.push_back(i);
                    data.emplace_back(A.diag[i], 0);
                    diagonal = true;
                }
                if (j != i) {
                    col.push_back(j);
                    data.emplace_back(A.a[k], 0);
                }
            }
            if (!diagonal) {
                diagpos[i] = static_cast<int>(col.size());
                col.push_back(i);
                data.emplace_back(A.diag[i], 0);
            }
        }
        ptr[A.n] = static_cast<int>(col.size());
        umfpack_zi_defaults(control.data());
        control[5] = 3;  // Symmetric ordering, with numerical pivoting retained.
        control[7] = 0;
        std::array<double, 90> info{};
        set_shift(CramCoefficients::poles()[0]);
        int status = umfpack_zi_symbolic(A.n, A.n, ptr.data(), col.data(),
                                        reinterpret_cast<double *>(data.data()), nullptr,
                                        &symbolic, control.data(), info.data());
        if (status != 0) return false;
        auto estimate = [&](const std::array<double, 90> &statistics) {
            double entries = statistics[36], flops = statistics[37];
            return 8 * (flops / 2.5e10 + entries * 3e-8 + A.n * 8e-7)
                + remaining * (8 * entries * 4e-9 + 0.005);
        };
        if (remaining * baseline > 4 && info[37] > 2e9 &&
            info[36] * 8 * 24 < 8e9) {
            void *alternative = nullptr;
            std::array<double, 90> alternative_info{};
            control[10] = 3;  // METIS nested dissection for larger sparse fronts.
            status = umfpack_zi_symbolic(A.n, A.n, ptr.data(), col.data(),
                                        reinterpret_cast<double *>(data.data()), nullptr,
                                        &alternative, control.data(), alternative_info.data());
            if (status == 0 && estimate(alternative_info) < estimate(info)) {
                umfpack_zi_free_symbolic(&symbolic);
                symbolic = alternative;
                info = alternative_info;
            } else {
                if (alternative) umfpack_zi_free_symbolic(&alternative);
                control[10] = 1;
            }
        }
        double entries = info[36], flops = info[37];
        double memory = 8 * (24 * entries + 128 * A.n) + 64 * info[39] * info[39];
        double predicted = estimate(info);
        if (std::getenv("PROP_DIAGNOSTIC"))
            std::fprintf(stderr, "direct estimate entries=%.6g flops=%.6g memory=%.6g seconds=%.4g baseline=%.4g\n",
                         entries, flops, memory, predicted, baseline * remaining);
        if (memory > 2.8e9 || entries <= 0 || !std::isfinite(predicted) ||
            predicted > 0.70 * remaining * baseline) return false;
        for (int pole = 0; pole < 8; ++pole) {
            set_shift(CramCoefficients::poles()[pole]);
            status = umfpack_zi_numeric(ptr.data(), col.data(),
                                       reinterpret_cast<double *>(data.data()), nullptr,
                                       symbolic, &numeric[pole], control.data(), info.data());
            if (status != 0 || 8 * info[40] * info[3] > 2.3e9) return false;
        }
        umfpack_zi_free_symbolic(&symbolic);
        return true;
    }

    bool solve(const Vec &b0, const Vec &b1, Vec &y) {
        const auto &theta = CramCoefficients::poles();
        const auto &alpha = CramCoefficients::residues();
        CVec rhs(A.n), x(A.n);
        std::vector<long double> sum(A.n);
        std::array<double, 90> info{};
        for (int i = 0; i < A.n; ++i) sum[i] = 2.1248537104952237488e-16L * b0[i];
        for (int pole = 0; pole < 8; ++pole) {
            Complex factor = h.t / theta[pole];
            for (int i = 0; i < A.n; ++i) rhs[i] = b0[i] + factor * b1[i];
            set_shift(theta[pole]);
            bool refine = h.rtol < 1e-9 && std::abs(alpha[pole]) > 10;
            control[7] = refine ? 1 : 0;
            int status = umfpack_zi_solve(0, ptr.data(), col.data(),
                                         reinterpret_cast<double *>(data.data()), nullptr,
                                         reinterpret_cast<double *>(x.data()), nullptr,
                                         reinterpret_cast<double *>(rhs.data()), nullptr,
                                         numeric[pole], control.data(), info.data());
            if (status != 0) return false;
            for (int i = 0; i < A.n; ++i)
                sum[i] -= 2 * (static_cast<long double>(alpha[pole].real()) * x[i].real()
                              - static_cast<long double>(alpha[pole].imag()) * x[i].imag());
        }
        for (int i = 0; i < A.n; ++i) y[i] = static_cast<double>(sum[i]);
        return true;
    }
};

static void initialize_libraries() {
    std::array<double, 16> diagonal{}, off{}, values{}, x{};
    std::array<double, 256> eigenvectors{};
    std::array<int, 32> support{};
    for (int i = 0; i < 16; ++i) {
        diagonal[i] = 2 + 0.01 * i;
        off[i] = 0.1;
        x[i] = 1;
    }
    lapack_int count = 0;
    LAPACKE_dstevr(LAPACK_COL_MAJOR, 'V', 'A', 16, diagonal.data(), off.data(), 0, 0, 0, 0,
                   0, &count, values.data(), eigenvectors.data(), 16, support.data());
    cblas_dgemv(CblasColMajor, CblasNoTrans, 16, 16, 1, eigenvectors.data(), 16,
                x.data(), 1, 0, values.data(), 1);
}

int main() {
    try {
        initialize_libraries();
        ::fcntl(STDIN_FILENO, F_SETPIPE_SZ, 1 << 20);
        ::fcntl(STDOUT_FILENO, F_SETPIPE_SZ, 1 << 20);
        write_all("READY\n", 6);
        char magic[8];
        read_all(magic, 8);
        if (std::memcmp(magic, "FAMEXP2\n", 8)) throw std::runtime_error("bad magic");
        uint32_t length;
        read_all(&length, 4);
        if (length > 4096) throw std::runtime_error("header too long");
        std::string json(length, '\0');
        read_all(json.data(), length);
        Header h(json);
        Matrix A(h);
        Vec b0(h.n), b1(h.n), y(h.n);
        PairFunctions *pair = nullptr;
        TensorSolver tensor;
        bool tensor_ready = false;
        SmallBlocks blocks;
        bool small_blocks = false;
        Chebyshev *cheb = nullptr;
        if (A.pairs) pair = new PairFunctions(A, h.t);
        else {
            tensor_ready = tensor.prepare(A, h);
            if (!tensor_ready) small_blocks = blocks.prepare(A, h);
            if (!small_blocks && !tensor_ready && A.bound > 2) {
                A.scale(h.t, h.rtol < 1e-9);
                cheb = new Chebyshev(A, h);
            }
        }
        bool use_lanczos = cheb && lanczos_candidate(A);
        std::unique_ptr<DirectCram> direct;
        bool direct_tried = false;
        double baseline_seconds = 0;
        for (int request = 0; request < h.requests; ++request) {
            read_all(b0.data(), b0.size() * sizeof(double));
            read_all(b1.data(), b1.size() * sizeof(double));
            if (cheb && !tensor_ready && !direct_tried &&
                h.requests > 1 && request > 0 && A.n >= 5000 && A.bound > 5000 &&
                baseline_seconds > 0.05) {
                direct.reset(new DirectCram(A, h));
                if (!direct->prepare(baseline_seconds, h.requests - request)) direct.reset();
                direct_tried = true;
            }
            auto begin = std::chrono::steady_clock::now();
            if (h.t == 0) y = b0;
            else if (pair) pair->solve(A, b0, b1, y);
            else if (small_blocks) blocks.solve(b0, b1, y);
            else if (tensor_ready) {
                if (!tensor.solve(b0, b1, y)) {
                    if (!cheb) {
                        A.scale(h.t, true);
                        cheb = new Chebyshev(A, h);
                    }
                    Vec candidate(A.n);
                    if (null_action(A, h, b0, b1, candidate)) y.swap(candidate);
                    else cheb->solve_precise(b0, b1, y);
                }
            }
            else if (cheb && null_action(A, h, b0, b1, y)) {}
            else if (cheb) {
                bool done = direct && direct->solve(b0, b1, y);
                if (!done && use_lanczos) {
                    double maximum0 = maxabs(b0), maximum1 = maxabs(b1);
                    bool forcing = maximum0 == 0;
                    bool combined = maximum0 != 0 && maximum1 != 0;
                    double estimate = 0;
                    for (int i = 0; i < A.n; ++i)
                        estimate = std::max(estimate, (std::abs(b0[i]) + h.t * std::abs(b1[i])) /
                                                       (1 + A.diag[i]));
                    double roundoff_budget = 64 * std::numeric_limits<double>::epsilon() *
                                             ((1 + A.bound) * maximum0 + h.t * maximum1);
                    bool eligible = !combined || h.atol + h.rtol * estimate > roundoff_budget;
                    if (eligible) {
                        double n0 = norm2(b0), n1 = h.t * norm2(b1);
                        double target = 0.001 * (h.atol + h.rtol * (maximum0 + h.t * maximum1));
                        int maxm = std::max(128, static_cast<int>(cheb->degree(n0, n1, target) * 0.7));
                        Lanczos lanczos(A, h);
                        if (!combined) {
                            done = lanczos.solve(forcing ? b1 : b0, forcing, maxm, y, true);
                            if (!done) use_lanczos = false;
                        } else {
                            // The phi identity needs a separate cancellation budget.
                            Vec residual(A.n);
                            const Vec &values = A.original.empty() ? A.a : A.original;
                            long double factor = A.original.empty() ? 1 : h.t;
                            for (int i = 0; i < A.n; ++i) {
                                long double value = 0;
                                for (uint32_t k = A.ptr[i]; k < A.ptr[i + 1]; ++k)
                                    value += static_cast<long double>(values[k]) * b0[A.col[k]];
                                residual[i] = static_cast<double>(static_cast<long double>(h.t) * b1[i] - factor * value);
                            }
                            done = lanczos.solve(residual, true, maxm, y, true, 1, &b0, estimate);
                            if (!done) use_lanczos = false;
                            if (h.atol + h.rtol * maxabs(y) < roundoff_budget) done = false;
                        }
                    }
                }
                if (!done) cheb->solve(b0, b1, y);
                if (cheb->cancellation_suspect(b0, b1, y)) cheb->solve_precise(b0, b1, y);
            }
            else taylor(A, h, b0, b1, y);
            if (!direct) baseline_seconds = std::max(baseline_seconds,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
            write_all(y.data(), y.size() * sizeof(double));
        }
        // All protocol writes are synchronous; the kernel releases session memory.
        ::_exit(0);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "solver: %s\n", e.what());
        return 1;
    }
}
