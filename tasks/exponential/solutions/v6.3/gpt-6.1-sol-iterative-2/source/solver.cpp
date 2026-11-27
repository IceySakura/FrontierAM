#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>
#include <lapacke.h>
#include <immintrin.h>
#include <Eigen/Eigenvalues>

using Vec = std::vector<double>;

// These buffers are always filled completely before their first read.
template<class T>
struct UninitializedAllocator : std::allocator<T> {
    using value_type = T;
    UninitializedAllocator() noexcept = default;
    template<class U> UninitializedAllocator(const UninitializedAllocator<U>&) noexcept {}
    template<class U> struct rebind { using other = UninitializedAllocator<U>; };
    template<class U> void construct(U* p) { ::new (static_cast<void*>(p)) U; }
    template<class U, class Arg, class... Args>
    void construct(U* p, Arg&& arg, Args&&... args) {
        ::new (static_cast<void*>(p)) U(std::forward<Arg>(arg), std::forward<Args>(args)...);
    }
};
template<class T, class U>
bool operator==(const UninitializedAllocator<T>&, const UninitializedAllocator<U>&) { return true; }
template<class T, class U>
bool operator!=(const UninitializedAllocator<T>&, const UninitializedAllocator<U>&) { return false; }
template<class T> using InputBuffer = std::vector<T, UninitializedAllocator<T>>;
using DataBuffer = InputBuffer<double>;

#ifdef FAMEXP_FORCE_INLINE
#define ROW_INLINE __attribute__((always_inline)) inline
#else
#define ROW_INLINE inline
#endif

static void read_all(void* buffer, size_t size) {
    auto* p = static_cast<unsigned char*>(buffer);
    while (size) {
        ssize_t count = ::read(STDIN_FILENO, p, size);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error("incomplete input");
        p += count;
        size -= static_cast<size_t>(count);
    }
}

static void write_all(const void* buffer, size_t size) {
    const auto* p = static_cast<const unsigned char*>(buffer);
    while (size) {
        ssize_t count = ::write(STDOUT_FILENO, p, size);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error("output failure");
        p += count;
        size -= static_cast<size_t>(count);
    }
}

struct Header {
    int n = 0, nnz = 0, requests = 0;
    double t = 0, rtol = 0, atol = 0;
};

// The protocol supplies a small, flat JSON object with trusted numeric fields.
class HeaderParser {
    const std::string& s;
    size_t pos = 0;
    void whitespace() {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' ||
                                  s[pos] == '\r' || s[pos] == '\n')) ++pos;
    }
    void expect(char c) {
        whitespace();
        if (pos == s.size() || s[pos++] != c)
            throw std::runtime_error("invalid header");
    }
    std::string string() {
        expect('"');
        std::string result;
        while (pos < s.size()) {
            char c = s[pos++];
            if (c == '"') return result;
            if (c == '\\') {
                if (pos == s.size()) break;
                c = s[pos++];
            }
            result.push_back(c);
        }
        throw std::runtime_error("unterminated JSON string");
    }
public:
    explicit HeaderParser(const std::string& input) : s(input) {}
    Header parse() {
        Header h;
        expect('{');
        for (;;) {
            whitespace();
            if (pos < s.size() && s[pos] == '}') { ++pos; break; }
            std::string key = string();
            expect(':');
            whitespace();
            if (s[pos] == '"') {
                string();
            } else {
                char* end = nullptr;
                double value = std::strtod(s.c_str() + pos, &end);
                if (end == s.c_str() + pos) throw std::runtime_error("invalid number");
                pos = static_cast<size_t>(end - s.c_str());
                if (key == "n") h.n = static_cast<int>(value);
                else if (key == "nnz") h.nnz = static_cast<int>(value);
                else if (key == "requests") h.requests = static_cast<int>(value);
                else if (key == "t") h.t = value;
                else if (key == "rtol") h.rtol = value;
                else if (key == "atol") h.atol = value;
            }
            whitespace();
            if (pos < s.size() && s[pos] == ',') { ++pos; continue; }
            expect('}');
            break;
        }
        return h;
    }
};

static double norm_inf(const Vec& v) {
    double norm = 0;
    for (double x : v) norm = std::max(norm, std::abs(x));
    return norm;
}

static double norm_two(const Vec& v) {
    double sum = 0;
    for (double x : v) sum += x * x;
    return std::sqrt(sum);
}

static double phi_scalar(double lambda, double t) {
    double x = t * lambda;
    if (x == 0) return t;
    if (std::abs(x) < 1e-3)
        return t * (1 + x * (-0.5 + x * (1.0 / 6 + x * (-1.0 / 24 + x / 120))));
    return -std::expm1(-x) / lambda;
}

struct Matrix {
    int n;
    std::vector<uint32_t> row;
    InputBuffer<uint32_t> col;
    DataBuffer val, absrow;
    Vec diagonal, potential;
    double upper = 0, lower = 0;
    bool pairs = true, diagonally_dominant = true, stable = false;
    bool separated = false, precise_edges = false;
    bool conservative = false;
    std::vector<int> neighbor;
    Vec edge;
    std::vector<int> permutation;
    std::vector<uint32_t> packed_row, packed_col;
    Vec packed_val;
    bool packed = false;
    std::vector<int> scalar_index;
    Vec scalar_e, scalar_f;

    explicit Matrix(const Header& h) : n(h.n), row(n + 1), col(h.nnz),
        val(h.nnz), absrow(n), precise_edges(h.rtol <= 1e-10) {
        InputBuffer<uint64_t> wire_row(n + 1);
        read_all(wire_row.data(), wire_row.size() * sizeof(uint64_t));
        for (int i = 0; i <= n; ++i) row[i] = static_cast<uint32_t>(wire_row[i]);
        read_all(col.data(), col.size() * sizeof(uint32_t));
        read_all(val.data(), val.size() * sizeof(double));
        for (int i = 0; i < n; ++i) {
            uint32_t j = row[i], end = row[i + 1];
            if (end - j > 128) {
                long double sum = 0;
                for (; j < end; ++j) sum += std::abs(static_cast<long double>(val[j]));
                absrow[i] = static_cast<double>(sum);
            } else {
                double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                for (; j + 3 < end; j += 4) {
                    s0 += std::abs(val[j]);
                    s1 += std::abs(val[j + 1]);
                    s2 += std::abs(val[j + 2]);
                    s3 += std::abs(val[j + 3]);
                }
                for (; j < end; ++j) s0 += std::abs(val[j]);
                absrow[i] = (s0 + s1) + (s2 + s3);
            }
            upper = std::max(upper, absrow[i]);
        }
        upper *= 1 + 5e-13;
        if (h.t == 0 || upper == 0 || (n > 64 && h.nnz > 2 * n && h.t * upper <= 2)) {
            pairs = false;
            return;
        }
        diagonal.resize(n);
        potential.resize(n);
        neighbor.assign(n, -1);
        edge.resize(n);
        lower = std::numeric_limits<double>::infinity();
        bool nonpositive_off = true, zero_potential = true;
        const bool quad_potential = precise_edges && h.t * upper > 1000;
        for (int i = 0; i < n; ++i) {
            long double off = 0;
            for (uint32_t j = row[i]; j < row[i + 1]; ++j) {
                if (col[j] == static_cast<uint32_t>(i)) diagonal[i] = val[j];
                else {
                    if (val[j] > 0) nonpositive_off = false;
                    off += std::abs(static_cast<long double>(val[j]));
                    if (val[j] != 0) {
                        if (neighbor[i] >= 0) pairs = false;
                        neighbor[i] = static_cast<int>(col[j]);
                        edge[i] = val[j];
                    }
                }
            }
            if (quad_potential) {
                __float128 exact_off = 0;
                for (uint32_t j = row[i]; j < row[i + 1]; ++j)
                    if (col[j] != static_cast<uint32_t>(i))
                        exact_off += static_cast<__float128>(std::abs(val[j]));
                absrow[i] = static_cast<double>(exact_off + std::abs(diagonal[i]));
                potential[i] = static_cast<double>(static_cast<__float128>(diagonal[i]) - exact_off);
            } else {
                absrow[i] = static_cast<double>(off + std::abs(diagonal[i]));
                potential[i] = static_cast<double>(static_cast<long double>(diagonal[i]) - off);
            }
            if (potential[i] < -2e-14 * absrow[i]) diagonally_dominant = false;
            if (potential[i] != 0) zero_potential = false;
            upper = std::max(upper, absrow[i]);
            lower = std::min(lower, potential[i]);
        }
        lower = std::max(0.0, lower);
        upper *= 1 + 4e-15;
        if (nonpositive_off && zero_potential) {
            conservative = quad_potential;
            if (!conservative) {
                conservative = true;
                for (double value : val) {
                    if (value != static_cast<double>(static_cast<int64_t>(value))) {
                        conservative = false;
                        break;
                    }
                }
            }
        }
        std::vector<uint32_t> offrow(n + 1);
        uint32_t count = 0;
        for (int i = 0; i < n; ++i) {
            offrow[i] = count;
            for (uint32_t j = row[i]; j < row[i + 1]; ++j) {
                if (col[j] != static_cast<uint32_t>(i) && val[j] != 0) {
                    col[count] = col[j];
                    val[count++] = val[j];
                }
            }
        }
        offrow[n] = count;
        row.swap(offrow);
        col.resize(count);
        val.resize(count);
        separated = true;
    }

    ROW_INLINE double dot_row(int i, const double* __restrict x) const {
        uint32_t j = row[i], end = row[i + 1];
        double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        double xi = x[i];
        if (precise_edges && end - j > 256) {
            long double sum = separated ?
                static_cast<long double>(stable ? potential[i] : diagonal[i]) * xi : 0;
            for (; j < end; ++j) {
                long double xj = x[col[j]];
                if (stable) xj += (val[j] < 0 ? -static_cast<long double>(xi) :
                                               static_cast<long double>(xi));
                sum += static_cast<long double>(val[j]) * xj;
            }
            return static_cast<double>(sum);
        }
        if (stable) {
            for (; j + 3 < end; j += 4) {
                s0 += val[j] * (x[col[j]] + (val[j] < 0 ? -xi : xi));
                s1 += val[j + 1] * (x[col[j + 1]] + (val[j + 1] < 0 ? -xi : xi));
                s2 += val[j + 2] * (x[col[j + 2]] + (val[j + 2] < 0 ? -xi : xi));
                s3 += val[j + 3] * (x[col[j + 3]] + (val[j + 3] < 0 ? -xi : xi));
            }
            for (; j < end; ++j) s0 += val[j] * (x[col[j]] + (val[j] < 0 ? -xi : xi));
            return ((s0 + s1) + (s2 + s3)) + potential[i] * xi;
        }
        for (; j + 3 < end; j += 4) {
            s0 += val[j] * x[col[j]];
            s1 += val[j + 1] * x[col[j + 1]];
            s2 += val[j + 2] * x[col[j + 2]];
            s3 += val[j + 3] * x[col[j + 3]];
        }
        for (; j < end; ++j) s0 += val[j] * x[col[j]];
        return ((s0 + s1) + (s2 + s3)) + (separated ? diagonal[i] * xi : 0);
    }

    inline __m256d dot_four(int block, const double* x) const {
        int i = block * 4;
        __m256d xi = _mm256_loadu_pd(x + i);
        __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd();
        uint32_t j = packed_row[block], end = packed_row[block + 1];
        if (stable) {
            __m256d mask = _mm256_set1_pd(-0.0);
            for (; j + 7 < end; j += 8) {
                __m256d v0 = _mm256_loadu_pd(packed_val.data() + j);
                __m256d v1 = _mm256_loadu_pd(packed_val.data() + j + 4);
                __m256d x0 = _mm256_i32gather_pd(x, _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(packed_col.data() + j)), 8);
                __m256d x1 = _mm256_i32gather_pd(x, _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(packed_col.data() + j + 4)), 8);
                x0 = _mm256_add_pd(x0, _mm256_xor_pd(xi, _mm256_and_pd(v0, mask)));
                x1 = _mm256_add_pd(x1, _mm256_xor_pd(xi, _mm256_and_pd(v1, mask)));
                s0 = _mm256_fmadd_pd(v0, x0, s0);
                s1 = _mm256_fmadd_pd(v1, x1, s1);
            }
            if (j < end) {
                __m256d v = _mm256_loadu_pd(packed_val.data() + j);
                __m256d xj = _mm256_i32gather_pd(x, _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(packed_col.data() + j)), 8);
                xj = _mm256_add_pd(xj, _mm256_xor_pd(xi, _mm256_and_pd(v, mask)));
                s0 = _mm256_fmadd_pd(v, xj, s0);
            }
            return _mm256_fmadd_pd(_mm256_loadu_pd(potential.data() + i), xi,
                                   _mm256_add_pd(s0, s1));
        }
        for (; j + 7 < end; j += 8) {
            __m256d v0 = _mm256_loadu_pd(packed_val.data() + j);
            __m256d v1 = _mm256_loadu_pd(packed_val.data() + j + 4);
            __m256d x0 = _mm256_i32gather_pd(x, _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(packed_col.data() + j)), 8);
            __m256d x1 = _mm256_i32gather_pd(x, _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(packed_col.data() + j + 4)), 8);
            s0 = _mm256_fmadd_pd(v0, x0, s0);
            s1 = _mm256_fmadd_pd(v1, x1, s1);
        }
        if (j < end) {
            __m256d v = _mm256_loadu_pd(packed_val.data() + j);
            __m256d xj = _mm256_i32gather_pd(x, _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(packed_col.data() + j)), 8);
            s0 = _mm256_fmadd_pd(v, xj, s0);
        }
        return _mm256_fmadd_pd(_mm256_loadu_pd(diagonal.data() + i), xi,
                               _mm256_add_pd(s0, s1));
    }

    void multiply_raw(const double* x, double* y) const {
        int i = 0;
        if (packed) {
            for (int block = 0; block < n / 4; ++block)
                _mm256_storeu_pd(y + block * 4, dot_four(block, x));
            i = n / 4 * 4;
        }
        for (; i < n; ++i) y[i] = dot_row(i, x);
    }

    void multiply(const Vec& x, Vec& y) const {
        multiply_raw(x.data(), y.data());
    }

    void prepare_scalars(double t) {
        double active_upper = 0;
        double active_lower = std::numeric_limits<double>::infinity();
        for (int i = 0; i < n; ++i) {
            if (row[i] == row[i + 1]) {
                scalar_index.push_back(i);
                scalar_e.push_back(std::exp(-t * diagonal[i]));
                scalar_f.push_back(phi_scalar(diagonal[i], t));
            } else {
                active_upper = std::max(active_upper, absrow[i]);
                active_lower = std::min(active_lower, potential[i]);
            }
        }
        if (!scalar_index.empty() && active_upper > 0) {
            upper = active_upper * (1 + 5e-13);
            lower = std::max(0.0, active_lower);
        }
    }

    double deflate_scalars(Vec& b0, Vec& b1, Vec& saved) const {
        double known = 0;
        for (size_t k = 0; k < scalar_index.size(); ++k) {
            int i = scalar_index[k];
            saved[i] = scalar_e[k] * b0[i] + scalar_f[k] * b1[i];
            known = std::max(known, std::abs(saved[i]));
            b0[i] = b1[i] = 0;
        }
        return known;
    }

    void restore_scalars(Vec& y, const Vec& saved) const {
        for (int i : scalar_index) y[i] = saved[i];
    }

    void prepare_packed(int window = 1024) {
        permutation.resize(n);
        for (int i = 0; i < n; ++i) permutation[i] = i;
        window = std::max(4, window / 4 * 4);
        for (int begin = 0; begin < n; begin += window) {
            int end = std::min(n, begin + window);
            std::stable_sort(permutation.begin() + begin, permutation.begin() + end,
                [&](int a, int b) { return row[a + 1] - row[a] < row[b + 1] - row[b]; });
        }
        size_t padded_count = 0;
        for (int i = 0; i + 3 < n; i += 4) {
            uint32_t width = 0;
            for (int k = 0; k < 4; ++k)
                width = std::max(width, row[permutation[i + k] + 1] - row[permutation[i + k]]);
            padded_count += 4ULL * width;
        }
        if (padded_count > 1.3 * col.size())
            std::stable_sort(permutation.begin(), permutation.end(),
                [&](int a, int b) { return row[a + 1] - row[a] < row[b + 1] - row[b]; });
        std::vector<int> inverse(n);
        for (int i = 0; i < n; ++i) inverse[permutation[i]] = i;
        for (int& i : scalar_index) i = inverse[i];
        std::vector<uint32_t> new_row(n + 1);
        InputBuffer<uint32_t> new_col(col.size());
        DataBuffer new_val(val.size()), new_absrow(n);
        Vec new_diagonal(n), new_potential(n);
        for (int i = 0; i < n; ++i) {
            int original = permutation[i];
            new_row[i + 1] = new_row[i] + row[original + 1] - row[original];
            new_diagonal[i] = diagonal[original];
            new_potential[i] = potential[original];
            new_absrow[i] = absrow[original];
            uint32_t out = new_row[i];
            for (uint32_t j = row[original]; j < row[original + 1]; ++j, ++out) {
                new_col[out] = inverse[col[j]];
                new_val[out] = val[j];
            }
        }
        row.swap(new_row); col.swap(new_col); val.swap(new_val);
        diagonal.swap(new_diagonal); potential.swap(new_potential); absrow.swap(new_absrow);
        int blocks = n / 4;
        packed_row.resize(blocks + 1);
        for (int block = 0; block < blocks; ++block) {
            int width = 0;
            for (int k = 0; k < 4; ++k)
                width = std::max(width, static_cast<int>(row[block * 4 + k + 1] - row[block * 4 + k]));
            packed_row[block + 1] = packed_row[block] + 4 * width;
        }
        packed_col.resize(packed_row.back());
        packed_val.resize(packed_row.back());
        for (int block = 0; block < blocks; ++block) {
            for (int k = 0; k < 4; ++k) {
                int i = block * 4 + k;
                uint32_t dst = packed_row[block] + k;
                for (uint32_t j = row[i]; j < row[i + 1]; ++j, dst += 4) {
                    packed_col[dst] = col[j];
                    packed_val[dst] = val[j];
                }
            }
        }
        packed = true;
    }

    void permute_input(Vec& v, Vec& temp) const {
        for (int i = 0; i < n; ++i) temp[i] = v[permutation[i]];
        v.swap(temp);
    }

    void unpermute_output(const Vec& y, Vec& out) const {
        for (int i = 0; i < n; ++i) out[permutation[i]] = y[i];
    }

    void tighten_bound(double eta) {
        if (eta < 100 || n < 128) return;
        Vec x(n), z(n);
        for (int i = 0; i < n; ++i)
            x[i] = row[i] == row[i + 1] ? 1e-10 : std::max(absrow[i] / upper, 1e-6);
        for (int k = 0; k < 3; ++k) {
            double bound = 0, maxz = 0;
            for (int i = 0; i < n; ++i) {
                if (row[i] == row[i + 1]) { z[i] = 0; continue; }
                double sum = std::abs(diagonal[i]) * x[i];
                for (uint32_t j = row[i]; j < row[i + 1]; ++j)
                    sum += std::abs(val[j]) * x[col[j]];
                z[i] = sum;
                bound = std::max(bound, sum / x[i]);
                maxz = std::max(maxz, sum);
            }
            upper = std::min(upper, bound * (1 + 2e-13));
            if (maxz == 0) break;
            for (int i = 0; i < n; ++i) x[i] = std::max(z[i] / maxz, 1e-10);
        }
    }
};

struct ExactPairs {
    Vec e0, e1, f0, f1;
    template<class Real>
    void prepare_pair(const Matrix& A, int i, int j, double t) {
        Real a = A.diagonal[i], b = A.diagonal[j], c = A.edge[i];
        Real mid = (a + b) / 2;
        Real r = std::hypot((a - b) / 2, c);
        Real lp = mid + r;
        Real lm = lp == 0 ? 0 : (a * b - c * c) / lp;
        lm = std::max(Real(0), lm);
        double ep = std::exp(-t * static_cast<double>(lp));
        double em = std::exp(-t * static_cast<double>(lm));
        double fp = phi_scalar(static_cast<double>(lp), t);
        double fm = phi_scalar(static_cast<double>(lm), t);
        Real weight = r == 0 ? 0 : (a - b) / (2 * r);
        double wi = static_cast<double>((1 + weight) / 2);
        double wj = 1 - wi;
        e0[i] = wi * ep + wj * em;
        e0[j] = wj * ep + wi * em;
        f0[i] = wi * fp + wj * fm;
        f0[j] = wj * fp + wi * fm;
        double factor = r == 0 ? 0 : static_cast<double>(c / (2 * r));
        e1[i] = e1[j] = factor * (ep - em);
        f1[i] = f1[j] = factor * (fp - fm);
    }
    explicit ExactPairs(const Matrix& A, double t, bool extended = true)
        : e0(A.n), e1(A.n), f0(A.n), f1(A.n) {
        for (int i = 0; i < A.n; ++i) {
            int j = A.neighbor[i];
            if (j < 0) {
                e0[i] = std::exp(-t * A.diagonal[i]);
                f0[i] = phi_scalar(A.diagonal[i], t);
            } else if (j > i) {
                if (extended) prepare_pair<long double>(A, i, j, t);
                else prepare_pair<double>(A, i, j, t);
            }
        }
    }
    void solve(const Matrix& A, const Vec& b0, const Vec& b1, Vec& y) const {
        for (int i = 0; i < A.n; ++i) {
            y[i] = e0[i] * b0[i] + f0[i] * b1[i];
            int j = A.neighbor[i];
            if (j >= 0) y[i] += e1[i] * b0[j] + f1[i] * b1[j];
        }
    }
};

struct DenseExact {
    using Real = long double;
    using Dense = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    int n;
    std::vector<Real> eigenvectors, e, f;
    DenseExact(const Matrix& A, double t)
        : n(A.n), eigenvectors(n * n), e(n), f(n) {
        Dense matrix = Dense::Zero(n, n);
        for (int i = 0; i < n; ++i) {
            matrix(i, i) = A.diagonal[i];
            for (uint32_t k = A.row[i]; k < A.row[i + 1]; ++k)
                matrix(i, A.col[k]) = A.val[k];
        }
        Eigen::SelfAdjointEigenSolver<Dense> eig(matrix);
        if (eig.info() != Eigen::Success) throw std::runtime_error("eigensolver failed");
        const Dense& V = eig.eigenvectors();
        for (int j = 0; j < n; ++j) {
            Real lambda = eig.eigenvalues()[j];
            if (t * A.upper > 100 && std::abs(lambda) < 1e-8 * A.upper) {
                __float128 rayleigh = 0;
                for (int r = 0; r < n; ++r) {
                    __float128 vr = V(r, j);
                    rayleigh += static_cast<__float128>(A.diagonal[r]) * vr * vr;
                    for (uint32_t k = A.row[r]; k < A.row[r + 1]; ++k)
                        rayleigh += static_cast<__float128>(A.val[k]) * vr *
                                    static_cast<__float128>(V(A.col[k], j));
                }
                lambda = static_cast<Real>(rayleigh);
            }
            lambda = std::max(Real(0), lambda);
            Real x = static_cast<Real>(t) * lambda;
            e[j] = std::exp(-x);
            f[j] = x == 0 ? static_cast<Real>(t) : -std::expm1(-x) / lambda;
            for (int i = 0; i < n; ++i) eigenvectors[i + n * j] = V(i, j);
        }
    }
    void solve(const Vec& b0, const Vec& b1, Vec& y) const {
        Real output[64] = {};
        for (int j = 0; j < n; ++j) {
            Real c0 = 0, c1 = 0;
            for (int i = 0; i < n; ++i) {
                c0 += eigenvectors[i + n * j] * b0[i];
                c1 += eigenvectors[i + n * j] * b1[i];
            }
            Real c = e[j] * c0 + f[j] * c1;
            for (int i = 0; i < n; ++i) output[i] += eigenvectors[i + n * j] * c;
        }
        for (int i = 0; i < n; ++i) y[i] = static_cast<double>(output[i]);
    }
};

struct Chebyshev {
    const Matrix& A;
    double t, rtol, atol;
    Vec ce, cf, tail_e, tail_f, work0, work1, work2;
    int degree_max;
    bool debug;

    Chebyshev(const Matrix& matrix, const Header& h)
        : A(matrix), t(h.t), rtol(h.rtol), atol(h.atol),
          work0(A.n), work1(A.n), work2(A.n),
          debug(std::getenv("FAMEXP_DEBUG") != nullptr) {
        long double a = static_cast<long double>(t) * A.upper / 2;
        degree_max = static_cast<int>(std::sqrt(128 * a)) + 64;
        degree_max = std::max(degree_max, 32);
        std::vector<long double> ratio(degree_max + 2), q(degree_max + 1);
        for (int k = degree_max; k >= 1; --k)
            ratio[k] = a / (2 * k + a * ratio[k + 1]);
        q[0] = 1;
        long double sum = 1;
        for (int k = 1; k <= degree_max; ++k) {
            q[k] = q[k - 1] * ratio[k];
            sum += 2 * q[k];
        }
        for (int k = 0; k <= degree_max; ++k)
            q[k] *= (k == 0 ? 1 : 2) / sum;
        ce.resize(degree_max + 1);
        cf.resize(degree_max + 1);
        long double s0 = 0, s1 = 0;
        for (int k = degree_max; k >= 1; --k) {
            ce[k] = static_cast<double>(q[k]);
            cf[k] = static_cast<double>(2 * (s1 - k * s0) / a);
            s0 += q[k];
            s1 += k * q[k];
        }
        ce[0] = static_cast<double>(q[0]);
        cf[0] = static_cast<double>(s1 / a);
        tail_e.resize(degree_max + 1);
        tail_f.resize(degree_max + 1);
        double se = 0, sf = 0;
        for (int k = degree_max; k >= 0; --k) {
            tail_e[k] = se;
            tail_f[k] = sf;
            se += ce[k];
            sf += cf[k];
        }
    }

    int degree(double n0, double n1, double threshold) const {
        int k = 0;
        while (k < degree_max &&
               tail_e[k] * n0 + tail_f[k] * t * n1 > threshold) ++k;
        return k;
    }

    void evaluate(int m, const Vec& b0, const Vec& b1, Vec& y) {
        if (m == 0) {
            for (int i = 0; i < A.n; ++i) y[i] = ce[0] * b0[i] + t * cf[0] * b1[i];
            return;
        }
        for (int i = 0; i < A.n; ++i) {
            work0[i] = ce[m] * b0[i] + t * cf[m] * b1[i];
            work1[i] = work0[i];
        }
        double scale = -4 / A.upper;
        // Reinsch's endpoint form retains differences instead of subtracting
        // two large Clenshaw states at the zero-eigenvalue endpoint.
        for (int k = m - 1; k >= 1; --k) {
            double e = ce[k], f = t * cf[k];
            const double* __restrict d1 = work1.data();
            double* __restrict delta = work0.data();
            double* __restrict out = work2.data();
            const double* __restrict v0 = b0.data();
            const double* __restrict v1 = b1.data();
            int i = 0;
            if (A.packed) {
                __m256d se = _mm256_set1_pd(e), sf = _mm256_set1_pd(f);
                __m256d ss = _mm256_set1_pd(scale);
                for (int block = 0; block < A.n / 4; ++block) {
                    int r = block * 4;
                    __m256d sum = A.dot_four(block, d1);
                    __m256d c = _mm256_mul_pd(se, _mm256_loadu_pd(v0 + r));
                    c = _mm256_fmadd_pd(sf, _mm256_loadu_pd(v1 + r), c);
                    c = _mm256_fmadd_pd(ss, sum, c);
                    __m256d d = _mm256_add_pd(_mm256_loadu_pd(delta + r), c);
                    _mm256_storeu_pd(delta + r, d);
                    _mm256_storeu_pd(out + r, _mm256_add_pd(_mm256_loadu_pd(d1 + r), d));
                }
                i = A.n / 4 * 4;
            }
            for (; i < A.n; ++i) {
                double sum = A.dot_row(i, d1);
                delta[i] += scale * sum + e * v0[i] + f * v1[i];
                out[i] = d1[i] + delta[i];
            }
            work1.swap(work2);
        }
        double e = ce[0], f = t * cf[0];
        int i = 0;
        if (A.packed) {
            __m256d se = _mm256_set1_pd(e), sf = _mm256_set1_pd(f);
            __m256d ss = _mm256_set1_pd(0.5 * scale);
            for (int block = 0; block < A.n / 4; ++block) {
                int r = block * 4;
                __m256d c = _mm256_mul_pd(se, _mm256_loadu_pd(b0.data() + r));
                c = _mm256_fmadd_pd(sf, _mm256_loadu_pd(b1.data() + r), c);
                c = _mm256_fmadd_pd(ss, A.dot_four(block, work1.data()), c);
                c = _mm256_add_pd(c, _mm256_loadu_pd(work0.data() + r));
                _mm256_storeu_pd(y.data() + r, c);
            }
            i = A.n / 4 * 4;
        }
        for (; i < A.n; ++i)
            y[i] = work0[i] + 0.5 * scale * A.dot_row(i, work1.data()) +
                   e * b0[i] + f * b1[i];
    }

    void solve(const Vec& b0, const Vec& b1, Vec& y, double known = 0) {
        double n0 = norm_two(b0), n1 = norm_two(b1);
        if (n0 == 0 && n1 == 0) {
            std::fill(y.begin(), y.end(), 0);
            return;
        }
        double input_scale = norm_inf(b0) + t * norm_inf(b1);
        if (A.conservative) {
            long double sum0 = 0, sum1 = 0, correction0 = 0, correction1 = 0;
            for (int i = 0; i < A.n; ++i) {
                long double x0 = static_cast<long double>(b0[i]) - correction0;
                long double x1 = static_cast<long double>(b1[i]) - correction1;
                long double next0 = sum0 + x0, next1 = sum1 + x1;
                correction0 = (next0 - sum0) - x0;
                correction1 = (next1 - sum1) - x1;
                sum0 = next0; sum1 = next1;
            }
            double mean = std::abs(static_cast<double>((sum0 + t * sum1) / A.n));
            input_scale = std::max(input_scale, mean * (1 + std::sqrt(t * A.upper)));
        }
        double threshold = 0.2 * (atol + rtol * input_scale /
                                   (1 + std::sqrt(t * A.upper)));
        threshold = std::max(threshold, 0.2 * (atol + rtol * known));
        int m = degree(n0, n1, threshold);
        for (;;) {
            evaluate(m, b0, b1, y);
            double tolerance = 0.4 * (atol + rtol * std::max(norm_inf(y), known));
            double bound = tail_e[m] * n0 + tail_f[m] * t * n1;
            if (bound <= tolerance) break;
            m = degree(n0, n1, tolerance);
        }
        if (debug) std::fprintf(stderr, "chebyshev n=%d eta=%.6g degree=%d norm=%.6g\n",
                                A.n, t * A.upper, m, norm_inf(y));
    }
};

static void taylor(const Matrix& A, const Header& h, const Vec& b0,
                   const Vec& b1, Vec& y, Vec& term, Vec& temp) {
    A.multiply(b0, temp);
    for (int i = 0; i < A.n; ++i) {
        term[i] = h.t * (b1[i] - temp[i]);
        y[i] = b0[i] + term[i];
    }
    double eta = h.t * A.upper;
    double initial_ratio = eta / 2;
    if (initial_ratio < 1 && norm_inf(term) * initial_ratio / (1 - initial_ratio) <
                             0.4 * (h.atol + h.rtol * norm_inf(y))) return;
    for (int k = 2; k <= 64; ++k) {
        A.multiply(term, temp);
        double factor = -h.t / k, term_max = 0, y_max = 0;
        for (int i = 0; i < A.n; ++i) {
            term[i] = factor * temp[i];
            y[i] += term[i];
            term_max = std::max(term_max, std::abs(term[i]));
            y_max = std::max(y_max, std::abs(y[i]));
        }
        double ratio = eta / (k + 1);
        if (ratio < 1 && term_max * ratio / (1 - ratio) <
                         0.4 * (h.atol + h.rtol * y_max)) break;
    }
}

#include "krylov.hpp"
#include "separable.hpp"
#include "blocks.hpp"
#include "steady.hpp"

int main() {
    try {
        write_all("READY\n", 6);
        char magic[8];
        read_all(magic, sizeof(magic));
        if (std::memcmp(magic, "FAMEXP2\n", 8)) throw std::runtime_error("invalid magic");
        uint32_t length;
        read_all(&length, sizeof(length));
        if (length > 4096) throw std::runtime_error("header too large");
        std::string json(length, '\0');
        read_all(json.data(), length);
        Header h = HeaderParser(json).parse();
        Matrix A(h);
        Vec b0(h.n), b1(h.n), y(h.n), term(h.n), temp(h.n);
        if (h.t == 0 || A.upper == 0) {
            for (int request = 0; request < h.requests; ++request) {
                read_all(b0.data(), 8 * h.n);
                read_all(b1.data(), 8 * h.n);
                for (int i = 0; i < h.n; ++i) y[i] = b0[i] + h.t * b1[i];
                write_all(y.data(), 8 * h.n);
            }
        } else if (A.pairs) {
            ExactPairs solver(A, h.t, h.rtol <= 1e-10);
            for (int request = 0; request < h.requests; ++request) {
                read_all(b0.data(), 8 * h.n);
                read_all(b1.data(), 8 * h.n);
                solver.solve(A, b0, b1, y);
                write_all(y.data(), 8 * h.n);
            }
        } else if (h.n <= 64) {
            DenseExact solver(A, h.t);
            for (int request = 0; request < h.requests; ++request) {
                read_all(b0.data(), 8 * h.n);
                read_all(b1.data(), 8 * h.n);
                solver.solve(b0, b1, y);
                write_all(y.data(), 8 * h.n);
            }
        } else if (h.t * A.upper <= 2) {
            for (int request = 0; request < h.requests; ++request) {
                read_all(b0.data(), 8 * h.n);
                read_all(b1.data(), 8 * h.n);
                taylor(A, h, b0, b1, y, term, temp);
                write_all(y.data(), 8 * h.n);
            }
        } else {
            const char* method = std::getenv("FAMEXP_METHOD");
            const double original_eta = h.t * A.upper;
            std::unique_ptr<SmallBlocks> blocks;
            if ((!method || std::strcmp(method, "blocks") == 0) &&
                (h.t * A.upper >= 100 || (h.requests == 16 && h.t * A.upper >= 4)))
                blocks = SmallBlocks::detect(A, h);
            if (blocks) {
                for (int request = 0; request < h.requests; ++request) {
                    read_all(b0.data(), 8 * h.n);
                    read_all(b1.data(), 8 * h.n);
                    blocks->solve(b0, b1, y);
                    write_all(y.data(), 8 * h.n);
                }
                return 0;
            }
            std::unique_ptr<Separable> separable;
            if ((!method || std::strcmp(method, "fft") == 0) && h.t * A.upper >= 300)
                separable = Separable::detect(A, h);
            if (!separable) {
                A.prepare_scalars(h.t);
                A.tighten_bound(h.t * A.upper);
            }
            A.stable = A.diagonally_dominant && h.rtol <= 1e-10 && original_eta > 1000;
            const char* layout = std::getenv("FAMEXP_LAYOUT");
            bool explicit_packing = layout && std::strcmp(layout, "packed") == 0;
            if (!separable && A.n >= 256 &&
                (explicit_packing || (A.stable && h.t * A.upper >= 1000)) &&
                (!layout || std::strcmp(layout, "csr") != 0)) {
                const char* sorting = std::getenv("FAMEXP_SORT");
                A.prepare_packed(sorting ? std::atoi(sorting) : 1024);
            }
            Chebyshev solver(A, h);
            std::unique_ptr<AdaptiveKrylov> krylov;
            std::unique_ptr<SteadyState> steady;
            if (!method && !separable && SteadyState::eligible(A, h))
                steady.reset(new SteadyState(A, h));
            double max_diag = 0, mean_diag = 0;
            int active_rows = 0;
            for (int i = 0; i < A.n; ++i) {
                if (A.row[i] == A.row[i + 1]) continue;
                max_diag = std::max(max_diag, A.diagonal[i]);
                mean_diag += A.diagonal[i];
                ++active_rows;
            }
            mean_diag /= std::max(1, active_rows);
            bool forced = method && std::strcmp(method, "krylov") == 0;
            bool automatic = !method && !separable && !steady && h.t * A.upper >= 1000 &&
                             max_diag > 3 * mean_diag;
            if (forced || automatic)
                krylov.reset(new AdaptiveKrylov(A, h));
            for (int request = 0; request < h.requests; ++request) {
                read_all(b0.data(), 8 * h.n);
                read_all(b1.data(), 8 * h.n);
                if (A.packed) {
                    A.permute_input(b0, temp);
                    A.permute_input(b1, temp);
                }
                if (h.rtol == 1e-7 && A.diagonally_dominant && original_eta > 1000) {
                    double combined = 0;
                    for (int i = 0; i < h.n; ++i)
                        combined = std::max(combined, std::abs(b0[i] + h.t * b1[i]));
                    double scale = norm_inf(b0) + h.t * norm_inf(b1);
                    A.stable = combined < 1e-3 * scale;
                }
                bool done = false;
                if (A.conservative) {
                    bool constant = true;
                    for (int i = 1; i < h.n; ++i)
                        if (b0[i] != b0[0] || b1[i] != b1[0]) { constant = false; break; }
                    if (constant) {
                        std::fill(y.begin(), y.end(), b0[0] + h.t * b1[0]);
                        done = true;
                    }
                }
                if (!done && separable) done = separable->solve(b0, b1, y);
                bool deflated = !done && !A.scalar_index.empty();
                double known = deflated ? A.deflate_scalars(b0, b1, term) : 0;
                if (!done && steady) {
                    done = steady->solve(b0, b1, y, known);
                    if (!done) steady.reset();
                }
                if (!done && krylov) {
                    int cap = 0;
                    if (automatic) {
                        double target = 0.2 * (h.atol + h.rtol *
                            (norm_inf(b0) + h.t * norm_inf(b1)) /
                            (1 + std::sqrt(h.t * A.upper)));
                        cap = std::max(32, static_cast<int>(0.7 *
                            solver.degree(norm_two(b0), norm_two(b1), target)));
                    }
                    done = krylov->solve(b0, b1, y, cap, known);
                    if (!done && automatic) krylov.reset();
                }
                if (!done) solver.solve(b0, b1, y, known);
                if (deflated) A.restore_scalars(y, term);
                if (A.packed) {
                    A.unpermute_output(y, temp);
                    write_all(temp.data(), 8 * h.n);
                } else write_all(y.data(), 8 * h.n);
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "solver: %s\n", e.what());
        return 1;
    }
}
