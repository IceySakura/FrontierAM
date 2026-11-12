#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>
#include <sys/uio.h>
#include <cblas.h>
#include <eigen3/Eigen/Eigenvalues>

using Vec = std::vector<double>;
using Complex = std::complex<long double>;

static const Complex cram_theta[8] = {
    {-10.843917078696988026L, 19.277446167181652284L},
    {-5.2649713434426468895L, 16.220221473167927305L},
    {5.9481522689511774808L, 3.5874573620183222829L},
    {3.5091036084149180974L, 8.4361989858843750826L},
    {6.4161776990994341923L, 1.1941223933701386874L},
    {1.4193758971856659786L, 10.925363484496723585L},
    {4.9931747377179963991L, 5.9968817136039422260L},
    {-1.4139284624888862114L, 13.497725698892745389L}
};
static const Complex cram_alpha[8] = {
    {-5.0901521865224915650e-7L, -2.4220017652852287970e-5L},
    {2.1151742182466030907e-4L, 4.3892969647380673918e-3L},
    {1.1339775178483930527e2L, 1.0194721704215856450e2L},
    {1.5059585270023467528e1L, -5.7514052776421819979L},
    {-6.4500878025539646595e1L, -2.2459440762652096056e2L},
    {-1.4793007113557999718L, 1.7686588323782937906L},
    {-6.2518392463207918892e1L, -1.1190391094283228480e1L},
    {4.1023136835410021273e-2L, -1.5743466173455468191e-1L}
};
static constexpr long double cram_constant = 2.1248537104952237488e-16L;
// Fixed-pole relative minimax approximation to phi1 on the negative real axis.
static const Complex cram_phi[8] = {
    {-0.000000943138965014723220831127554738380053822237382L,
      0.000000556892896167077762098098149920167813860726799L},
    {0.000240984021734263136868225084511722990765120359L,
     -0.0000912622479158473536129000292693692938014311988L},
    {21.5591812519046214481645711112823065850673011L,
      4.13650689390269193429363759315798300396163367L},
    {0.051814557348988608498647127084319537446543309L,
     -1.76356236935603769978973878950963776650559855L},
    {-16.012914459296421895707418734052741842556935L,
     -32.0242110819312588052006495337262280772480842L},
    {0.141899907972796316274616466698353250547004182L,
     0.153835614588617785394234803005579840576811793L},
    {-6.22836919143921772625528370316983325866371936L,
      5.23923242356449077944867617317707962286124273L},
    {-0.0118521073735330801138466425629997724405316888L,
     -0.00179771374832688432925715180241717388437112753L}
};

static long double norm2(const Vec& x) {
    long double s = 0;
    for (double v : x) s += static_cast<long double>(v) * v;
    return std::sqrt(s);
}

#include "shifted.hpp"
#include "diagonal.hpp"
#include "sliced.hpp"

static void read_all(void* ptr, size_t count) {
    auto* p = static_cast<unsigned char*>(ptr);
    while (count) {
        ssize_t got = ::read(0, p, count);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) throw std::runtime_error("short input");
        p += got;
        count -= static_cast<size_t>(got);
    }
}

static void read_blocks(std::vector<iovec> blocks) {
    size_t first = 0;
    while (first < blocks.size()) {
        while (first < blocks.size() && blocks[first].iov_len == 0) ++first;
        if (first == blocks.size()) break;
        ssize_t count = ::readv(0, blocks.data() + first,
                               static_cast<int>(blocks.size() - first));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error("short input");
        size_t got = static_cast<size_t>(count);
        while (first < blocks.size() && got >= blocks[first].iov_len) {
            got -= blocks[first].iov_len;
            ++first;
        }
        if (first < blocks.size()) {
            blocks[first].iov_base = static_cast<unsigned char*>(blocks[first].iov_base) + got;
            blocks[first].iov_len -= got;
        }
    }
}

static void configure_pipe(int fd) {
    int size = 1 << 20;
    if (const char* setting = std::getenv("FAMEXP_PIPE_SIZE")) size = std::atoi(setting);
    for (; size >= 65536; size /= 2) {
        if (::fcntl(fd, F_SETPIPE_SZ, size) >= 0) break;
    }
}

static bool zero_rhs(const Vec& b0, const Vec& b1) {
    for (size_t i = 0; i < b0.size(); ++i)
        if (b0[i] != 0 || b1[i] != 0) return false;
    return true;
}

static void write_all(const void* ptr, size_t count) {
    const auto* p = static_cast<const unsigned char*>(ptr);
    while (count) {
        ssize_t sent = ::write(1, p, count);
        if (sent < 0 && errno == EINTR) continue;
        if (sent <= 0) throw std::runtime_error("output failure");
        p += sent;
        count -= static_cast<size_t>(sent);
    }
}

// The protocol object has only string and numeric scalar values.
static std::map<std::string, std::string> parse_header(const std::string& s) {
    std::map<std::string, std::string> out;
    size_t i = 0;
    auto space = [&]() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\n' ||
                                s[i] == '\t' || s[i] == '\r')) ++i;
    };
    auto quoted = [&]() {
        if (s.at(i++) != '"') throw std::runtime_error("invalid JSON string");
        std::string value;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\') {
                ++i;
                if (i == s.size()) throw std::runtime_error("invalid JSON escape");
            }
            value += s[i++];
        }
        if (i == s.size()) throw std::runtime_error("unterminated JSON");
        ++i;
        return value;
    };
    space();
    if (s.at(i++) != '{') throw std::runtime_error("invalid JSON object");
    while (true) {
        space();
        if (s.at(i) == '}') break;
        std::string key = quoted();
        space();
        if (s.at(i++) != ':') throw std::runtime_error("missing JSON colon");
        space();
        std::string value;
        if (s.at(i) == '"') value = quoted();
        else {
            size_t first = i;
            while (i < s.size() && s[i] != ',' && s[i] != '}') ++i;
            value = s.substr(first, i - first);
        }
        out.emplace(std::move(key), std::move(value));
        space();
        if (s.at(i) == '}') break;
        if (s.at(i++) != ',') throw std::runtime_error("missing JSON comma");
    }
    return out;
}

struct DisjointSet {
    std::vector<int> p, sz;
    explicit DisjointSet(int n) : p(n), sz(n, 1) {
        std::iota(p.begin(), p.end(), 0);
    }
    int find(int i) {
        while (i != p[i]) {
            p[i] = p[p[i]];
            i = p[i];
        }
        return i;
    }
    void join(int i, int j) {
        i = find(i);
        j = find(j);
        if (i == j) return;
        if (sz[i] < sz[j]) std::swap(i, j);
        p[j] = i;
        sz[i] += sz[j];
    }
};

struct Block {
    std::vector<int> ids;
    std::vector<long double> q, ex, ph;
};

template <typename Real>
struct PairBlock {
    int i, j;
    Real q0, q1, e0, e1, p0, p1;
};

static double phi(double x) {
    if (std::abs(x) < 1e-8) {
        return 1 + x * (0.5 + x * (1.0 / 6 + x / 24));
    }
    return std::expm1(x) / x;
}

static long double phi_extended(long double x) {
    if (std::abs(x) < 1e-10L)
        return 1 + x * (0.5L + x * (1.0L / 6 + x * (1.0L / 24 + x / 120)));
    return std::expm1(x) / x;
}

struct Solver {
    int n, requests;
    double t, rtol, atol, upper = 0, input_norm = 0, mean_diagonal = 0, mean_distance = 0;
    std::vector<int> row;
    std::vector<uint32_t> col;
    Vec val, diagonal, rowsum;
    std::vector<int8_t> signs;
    Vec ce, cp;
    std::vector<long double> tail_e, tail_p;
    std::vector<Block> blocks;
    std::vector<PairBlock<long double>> pairs;
    std::vector<PairBlock<double>> fast_pairs;
    Vec diagonal_exp, diagonal_phi;
    bool direct = false, diagonal_only = false, identity_step = false, raw_taylor = false;
    bool small_operator = false;
    mutable bool block_disabled = false, krylov_disabled = false;
    mutable std::unique_ptr<DiagonalKernel> diagonal_kernel;
    mutable bool diagonal_checked = false;
    mutable std::unique_ptr<SlicedKernel> sliced_kernel;
    mutable bool sliced_checked = false;
    mutable int constant_null_status = 0;

    void prepare(const std::vector<uint64_t>& ptr,
                 std::vector<uint32_t> idx, Vec data) {
        for (int i = 0; i < n; ++i) {
            double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            size_t k = ptr[i], end = ptr[i + 1];
            for (; k + 3 < end; k += 4) {
                s0 += std::abs(data[k]);
                s1 += std::abs(data[k + 1]);
                s2 += std::abs(data[k + 2]);
                s3 += std::abs(data[k + 3]);
            }
            double sum = (s0 + s1) + (s2 + s3);
            for (; k < end; ++k) sum += std::abs(data[k]);
            upper = std::max(upper, sum);
        }
        upper *= 1 + 4e-11;
        input_norm = upper;
        if (t == 0 || upper == 0) return;
        if (t * upper <= 1e-16) {
            identity_step = true;
            return;
        }
        if (t * upper <= 0.25) {
            raw_taylor = true;
            row.resize(n + 1);
            for (int i = 0; i <= n; ++i) row[i] = static_cast<int>(ptr[i]);
            col = std::move(idx);
            val = std::move(data);
            return;
        }
        diagonal.assign(n, 0);
        rowsum.assign(n, 0);
        row.resize(n + 1);
        col = std::move(idx);
        val = std::move(data);
        size_t out = 0;
        uint64_t distance = 0;
        bool positive_off_diagonal = false;
        int max_off = 0;
        for (int i = 0; i < n; ++i) {
            row[i] = static_cast<int>(out);
            long double sum = 0, correction = 0;
            bool compensate = ptr[i + 1] - ptr[i] > 64;
            for (size_t k = ptr[i]; k < ptr[i + 1]; ++k) {
                int j = static_cast<int>(col[k]);
                double v = val[k];
                if (compensate) {
                    long double add = static_cast<long double>(v) - correction;
                    long double next = sum + add;
                    correction = (next - sum) - add;
                    sum = next;
                } else sum += v;
                if (i == j) diagonal[i] = v;
                else if (v != 0) {
                    col[out] = static_cast<uint32_t>(j);
                    val[out++] = v;
                    distance += std::abs(j - i);
                    positive_off_diagonal = positive_off_diagonal || v > 0;
                }
            }
            max_off = std::max(max_off, static_cast<int>(out) - row[i]);
            rowsum[i] = static_cast<double>(sum);
        }
        row[n] = static_cast<int>(out);
        mean_distance = out ? static_cast<double>(distance) / out : 0;
        val.resize(out);
        col.resize(out);
        long double trace = 0;
        for (double v : diagonal) trace += v;
        mean_diagonal = static_cast<double>(trace / n);
        if (max_off == 0) {
            diagonal_only = true;
            diagonal_exp.resize(n);
            diagonal_phi.resize(n);
            for (int i = 0; i < n; ++i) {
                double z = -t * diagonal[i];
                diagonal_exp[i] = std::exp(z);
                diagonal_phi[i] = t * phi(z);
            }
            return;
        }
        if (max_off <= 1) {
            if (rtol >= 1e-7) prepare_pairs(fast_pairs);
            else prepare_pairs(pairs);
            direct = true;
            return;
        }
        DisjointSet ds(n);
        bool small = true;
        for (int i = 0; i < n && small; ++i) {
            for (int k = row[i]; k < row[i + 1]; ++k) {
                if (col[k] > i) {
                    ds.join(i, col[k]);
                    if (ds.sz[ds.find(i)] > 64) {
                        small = false;
                        break;
                    }
                }
            }
        }
        if (small && n > 64) {
            uint64_t work = 0;
            for (int i = 0; i < n; ++i) {
                if (ds.find(i) == i) {
                    uint64_t size = ds.sz[i];
                    work += size * size * size;
                }
            }
            small = work <= 64000000;
        }
        if (small && n > 64) {
            if (prepare_blocks(ds, true)) {
                direct = true;
                return;
            }
            blocks.clear();
        }
        small_operator = n <= 64;
        if (positive_off_diagonal) normalize_signs();
        if (t * upper > 100) tighten_upper();
        upper *= 1 + 2e-14;
        if (t * upper > 0.25) prepare_coefficients();
    }

    void normalize_signs() {
        signs.assign(n, 0);
        std::vector<int> queue;
        queue.reserve(n);
        for (int root = 0; root < n; ++root) {
            if (signs[root]) continue;
            signs[root] = 1;
            queue.push_back(root);
            for (size_t head = 0; head < queue.size(); ++head) {
                int i = queue[head];
                for (int k = row[i]; k < row[i + 1]; ++k) {
                    int j = col[k];
                    int8_t desired = val[k] < 0 ? signs[i] : -signs[i];
                    if (!signs[j]) {
                        signs[j] = desired;
                        queue.push_back(j);
                    } else if (signs[j] != desired) {
                        signs.clear();
                        return;
                    }
                }
            }
            queue.clear();
        }
        for (int i = 0; i < n; ++i) {
            long double sum = diagonal[i], correction = 0;
            for (int k = row[i]; k < row[i + 1]; ++k) {
                val[k] *= signs[i] * signs[col[k]];
                long double add = static_cast<long double>(val[k]) - correction;
                long double next = sum + add;
                correction = (next - sum) - add;
                sum = next;
            }
            rowsum[i] = static_cast<double>(sum);
        }
    }

    bool exact_constant_null() const {
        if (constant_null_status) return constant_null_status > 0;
        constant_null_status = -1;
        for (double v : rowsum) if (v != 0) return false;
        int minimum_bit = 2048;
        auto inspect = [&](double value) {
            if (value == 0) return;
            uint64_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            int exponent = static_cast<int>((bits >> 52) & 2047);
            uint64_t mantissa = bits & ((uint64_t(1) << 52) - 1);
            int base = -1074;
            if (exponent) {
                mantissa |= uint64_t(1) << 52;
                base = exponent - 1023 - 52;
            }
            minimum_bit = std::min(minimum_bit, base + __builtin_ctzll(mantissa));
        };
        for (double v : diagonal) inspect(v);
        for (double v : val) inspect(v);
        int largest_bit = 0;
        std::frexp(input_norm, &largest_bit);
        // Every intermediate sum then fits exactly in the long-double mantissa.
        if (largest_bit - minimum_bit <= std::numeric_limits<long double>::digits - 1) {
            constant_null_status = 1;
            return true;
        }
        return false;
    }

    template <typename Real>
    void prepare_pairs(std::vector<PairBlock<Real>>& destination) {
        destination.reserve(n / 2 + 1);
        for (int i = 0; i < n; ++i) {
            int j = row[i] < row[i + 1] ? col[row[i]] : -1;
            if (j < i && j >= 0) continue;
            PairBlock<Real> b;
            b.i = i;
            b.j = j;
            if (j < 0) {
                double z = -t * diagonal[i];
                b.q0 = 1;
                b.q1 = 0;
                b.e0 = std::exp(z);
                b.p0 = t * phi(z);
                b.e1 = b.p1 = 0;
            } else {
                Real a = diagonal[i], d = diagonal[j], c = val[row[i]];
                if (static_cast<long double>(t) *
                    std::max(std::abs(a) + std::abs(c), std::abs(d) + std::abs(c)) <= 1e-16L) {
                    b.q0 = 1;
                    b.q1 = 0;
                    b.e0 = b.e1 = 1;
                    b.p0 = b.p1 = t;
                    destination.push_back(b);
                    continue;
                }
                Real delta = (a - d) / 2;
                Real radius = std::hypot(delta, c);
                Real high = (a + d) / 2 + radius;
                Real low = high != 0 ? (a * d - c * c) / high : 0;
                if (delta >= 0) {
                    b.q0 = std::sqrt((1 + delta / radius) / 2);
                    b.q1 = (c / radius) / (2 * b.q0);
                } else {
                    b.q1 = std::copysign(std::sqrt((1 - delta / radius) / 2), c);
                    b.q0 = (c / radius) / (2 * b.q1);
                }
                Real z0 = -static_cast<Real>(t) * high;
                Real z1 = -static_cast<Real>(t) * low;
                b.e0 = z0 < -50 ? 0 : std::exp(z0);
                b.e1 = z1 < -50 ? 0 : std::exp(z1);
                b.p0 = z0 == 0 ? t : (z0 < -50 ? -1 : std::expm1(z0)) / (-high);
                b.p1 = z1 == 0 ? t : (z1 < -50 ? -1 : std::expm1(z1)) / (-low);
            }
            destination.push_back(b);
        }
    }

    template <typename Real>
    void apply_pairs(const std::vector<PairBlock<Real>>& source,
                     const Vec& b0, const Vec& b1, Vec& y) const {
        for (const auto& b : source) {
            int i = b.i, j = b.j;
            if (j < 0) {
                y[i] = static_cast<double>(b.e0 * b0[i] + b.p0 * b1[i]);
                continue;
            }
            Real x0 = b.q0 * b0[i] + b.q1 * b0[j];
            Real x1 = -b.q1 * b0[i] + b.q0 * b0[j];
            Real z0 = b.q0 * b1[i] + b.q1 * b1[j];
            Real z1 = -b.q1 * b1[i] + b.q0 * b1[j];
            x0 = b.e0 * x0 + b.p0 * z0;
            x1 = b.e1 * x1 + b.p1 * z1;
            y[i] = static_cast<double>(b.q0 * x0 - b.q1 * x1);
            y[j] = static_cast<double>(b.q1 * x0 + b.q0 * x1);
        }
    }

    bool prepare_blocks(DisjointSet& ds, bool split) {
        std::vector<int> which(n, -1), local(n);
        for (int i = 0; i < n; ++i) {
            int root = split ? ds.find(i) : 0;
            if (which[root] < 0) {
                which[root] = static_cast<int>(blocks.size());
                blocks.emplace_back();
            }
            auto& b = blocks[which[root]];
            local[i] = static_cast<int>(b.ids.size());
            b.ids.push_back(i);
        }
        for (auto& b : blocks) {
            int m = static_cast<int>(b.ids.size());
            b.q.assign(m * m, 0);
            using Dense = Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic>;
            Dense matrix = Dense::Zero(m, m);
            for (int ii = 0; ii < m; ++ii) {
                int i = b.ids[ii];
                matrix(ii, ii) = diagonal[i];
                for (int k = row[i]; k < row[i + 1]; ++k) {
                    matrix(ii, local[col[k]]) = val[k];
                }
            }
            Eigen::SelfAdjointEigenSolver<Dense> eig(matrix);
            if (eig.info() != Eigen::Success) return false;
            for (int k = 0; k < m; ++k)
                for (int i = 0; i < m; ++i) b.q[i + m * k] = eig.eigenvectors()(i, k);
            b.ex.resize(m);
            b.ph.resize(m);
            for (int k = 0; k < m; ++k) {
                long double lambda = eig.eigenvalues()[k];
                if (rtol < 1e-9) {
                    __float128 rayleigh = 0, norm = 0;
                    for (int ii = 0; ii < m; ++ii) {
                        int i = b.ids[ii];
                        __float128 qi = static_cast<__float128>(b.q[ii + m * k]);
                        norm += qi * qi;
                        rayleigh += static_cast<__float128>(diagonal[i]) * qi * qi;
                        for (int j = row[i]; j < row[i + 1] &&
                             col[j] < static_cast<uint32_t>(i); ++j)
                            rayleigh += static_cast<__float128>(2 * val[j]) * qi *
                                static_cast<__float128>(b.q[local[col[j]] + m * k]);
                    }
                    lambda = static_cast<long double>(rayleigh / norm);
                }
                long double z = -static_cast<long double>(t) * lambda;
                b.ex[k] = std::exp(z);
                b.ph[k] = static_cast<long double>(t) * phi_extended(z);
            }
        }
        return true;
    }

    // Collatz bounds for |L| remain upper bounds for the spectrum of L.
    void tighten_upper() {
        Vec w(n, 1), z(n);
        for (int it = 0; it < 10; ++it) {
            double bound = 0, largest = 0;
            for (int i = 0; i < n; ++i) {
                double v = std::abs(diagonal[i]) * w[i];
                for (int k = row[i]; k < row[i + 1]; ++k)
                    v += std::abs(val[k]) * w[col[k]];
                z[i] = v;
                bound = std::max(bound, v / w[i]);
                largest = std::max(largest, v);
            }
            upper = std::min(upper, bound * (1 + 4e-11));
            for (int i = 0; i < n; ++i) w[i] = z[i] / largest + 1e-12;
        }
    }

    void prepare_coefficients() {
        long double a = static_cast<long double>(t) * upper / 2;
        int last = static_cast<int>(std::ceil(
            std::sqrt(2 * a * 72.0L) + 80.0L));
        last = std::max(last, 80);
        std::vector<long double> e(last + 2, 0), p(last + 2, 0);
        e[last] = 1;
        for (int k = last; k > 0; --k) {
            e[k - 1] = e[k + 1] + 2 * k / a * e[k];
            if (e[k - 1] > 1e200L) {
                for (int j = k - 1; j <= last + 1; ++j) e[j] *= 1e-200L;
            }
        }
        long double norm = e[0];
        for (int k = 1; k <= last; ++k) norm += 2 * e[k];
        e[0] /= norm;
        for (int k = 1; k <= last; ++k) e[k] *= 2 / norm;
        long double sum = 0, moment = 0;
        for (int k = last; k >= 1; --k) {
            p[k] = 2 * moment / a;
            sum += e[k];
            moment += sum;
        }
        p[0] = moment / a;
        ce.resize(last + 1);
        cp.resize(last + 1);
        tail_e.assign(last + 1, 0);
        tail_p.assign(last + 1, 0);
        long double te = 0, tp = 0;
        for (int k = last; k >= 0; --k) {
            ce[k] = static_cast<double>(e[k]);
            cp[k] = static_cast<double>(p[k] * t);
            tail_e[k] = te;
            tail_p[k] = tp * t;
            te += e[k];
            tp += p[k];
        }
    }

    inline double product_row(int i, const double* x) const {
        double xi = x[i], s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int k = row[i], end = row[i + 1];
        for (; k + 3 < end; k += 4) {
            s0 += val[k] * (x[col[k]] - xi);
            s1 += val[k + 1] * (x[col[k + 1]] - xi);
            s2 += val[k + 2] * (x[col[k + 2]] - xi);
            s3 += val[k + 3] * (x[col[k + 3]] - xi);
        }
        double v = (s0 + s1) + (s2 + s3) + rowsum[i] * xi;
        for (; k < end; ++k) v += val[k] * (x[col[k]] - xi);
        return v;
    }

    inline double raw_product_row(int i, const double* x) const {
        int k = row[i], end = row[i + 1];
        if (end - k > 64 && rtol < 1e-9) {
            long double sum = 0, correction = 0;
            for (; k < end; ++k) {
                long double add = static_cast<long double>(val[k]) * x[col[k]] - correction;
                long double next = sum + add;
                correction = (next - sum) - add;
                sum = next;
            }
            return static_cast<double>(sum);
        }
        double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (; k + 3 < end; k += 4) {
            s0 += val[k] * x[col[k]];
            s1 += val[k + 1] * x[col[k + 1]];
            s2 += val[k + 2] * x[col[k + 2]];
            s3 += val[k + 3] * x[col[k + 3]];
        }
        double sum = (s0 + s1) + (s2 + s3);
        for (; k < end; ++k) sum += val[k] * x[col[k]];
        return sum;
    }

    Vec taylor(const Vec& b0, const Vec& b1) const {
        Vec term(n), next(n), y(b0);
        double eta = t * upper;
        bool constant = true;
        for (int i = 1; i < n; ++i) {
            if (b0[i] != b0[0]) {
                constant = false;
                break;
            }
        }
        for (int i = 0; i < n; ++i) {
            double lb0;
            if (constant && b0[0] == 0) lb0 = 0;
            else if (raw_taylor) lb0 = raw_product_row(i, b0.data());
            else lb0 = constant ? rowsum[i] * b0[0] : product_row(i, b0.data());
            term[i] = t * (b1[i] - lb0);
        }
        int k = 1;
        for (; k < 80; ++k) {
            double size = 0, normy = 0;
            for (int i = 0; i < n; ++i) {
                y[i] += term[i];
                size = std::max(size, std::abs(term[i]));
                normy = std::max(normy, std::abs(y[i]));
            }
            double bound = size * eta / (k + 1 - eta);
            if (bound <= 0.1 * (atol + rtol * normy)) break;
            double factor = -t / (k + 1);
            for (int i = 0; i < n; ++i) {
                double product = raw_taylor ? raw_product_row(i, term.data()) :
                                              product_row(i, term.data());
                next[i] = factor * product;
            }
            term.swap(next);
        }
        if (std::getenv("FAMEXP_DEBUG"))
            std::fprintf(stderr, "taylor degree=%d eta=%.8g\n", k, eta);
        return y;
    }

    long double rational_projection(const Vec& alpha, const Vec& beta,
                                    bool forcing, double scale, Vec& weights) const {
        int m = static_cast<int>(alpha.size());
        std::vector<Complex> d(m), h(m);
        std::vector<long double> total(m, 0);
        long double bound = 0;
        for (int pole = 0; pole < 8; ++pole) {
            Complex shift = cram_theta[pole];
            Complex weight = cram_alpha[pole];
            if (forcing) weight /= shift;
            weight *= scale;
            d[0] = static_cast<long double>(alpha[0]) + shift;
            h[0] = 1;
            for (int i = 1; i < m; ++i) {
                Complex mult = static_cast<long double>(beta[i - 1]) / d[i - 1];
                d[i] = static_cast<long double>(alpha[i]) + shift -
                       static_cast<long double>(beta[i - 1]) * mult;
                h[i] = -mult * h[i - 1];
            }
            h[m - 1] /= d[m - 1];
            long double distance = shift.real() >= 0 ? std::abs(shift) :
                                                       std::abs(shift.imag());
            bound += 2 * std::abs(weight) * std::abs(h[m - 1]) / distance;
            for (int i = m - 2; i >= 0; --i)
                h[i] = (h[i] - static_cast<long double>(beta[i]) * h[i + 1]) / d[i];
            for (int i = 0; i < m; ++i) total[i] -= 2 * (weight * h[i]).real();
        }
        weights.resize(m);
        for (int i = 0; i < m; ++i) weights[i] = static_cast<double>(total[i]);
        return bound;
    }

    // After an alternating sign change, -T is Metzler. Positive scaling and
    // squaring computes its phi corners without cancellation in tiny defects.
    long double defect_corner(const Vec& alpha, const Vec& beta,
                              bool forcing) const {
        int m = static_cast<int>(alpha.size()), d = m + 2;
        double shift = 0;
        for (double a : alpha) shift = std::max(shift, a);
        double norm = shift;
        for (int i = 0; i < m; ++i) {
            double r = shift - alpha[i] + (i ? beta[i - 1] : 0) +
                       (i + 1 < m ? beta[i] : 0) + (i == 0 ? 1 : 0);
            norm = std::max(norm, r);
        }
        norm = std::max(norm, shift + 1);
        int squares = norm > 1 ? static_cast<int>(std::ceil(std::log2(norm))) : 0;
        double h = std::ldexp(1.0, -squares), sh = h * shift;
        Vec term(d * d, 0), next(d * d), p(d * d, 0), scratch(d * d);
        for (int i = 0; i < d; ++i) term[i + d * i] = p[i + d * i] = 1;
        for (int k = 1; k <= 32; ++k) {
            double fac = h / k;
            for (int j = 0; j < d; ++j) {
                int base = d * j;
                for (int i = 0; i < m; ++i) {
                    double v = (shift - alpha[i]) * term[base + i];
                    if (i) v += beta[i - 1] * term[base + i - 1];
                    if (i + 1 < m) v += beta[i] * term[base + i + 1];
                    if (i == 0) v += term[base + m];
                    next[base + i] = fac * v;
                }
                next[base + m] = fac * (shift * term[base + m] + term[base + m + 1]);
                next[base + m + 1] = sh / k * term[base + m + 1];
            }
            for (size_t i = 0; i < p.size(); ++i) p[i] += next[i];
            term.swap(next);
        }
        double factor = std::exp(-sh);
        for (double& x : p) x *= factor;
        p[m + d * m] = p[m + 1 + d * (m + 1)] = 1;
        p[m + d * (m + 1)] = h;
        for (int k = 0; k < squares; ++k) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d, d, d,
                        1, p.data(), d, p.data(), d, 0, scratch.data(), d);
            p.swap(scratch);
            p[m + d * m] = p[m + 1 + d * (m + 1)] = 1;
            p[m + d * (m + 1)] = std::ldexp(h, k + 1);
        }
        int column = forcing ? m + 1 : m;
        return static_cast<long double>(p[m - 1 + d * column]) * 1.00001L +
               std::ldexp(1e-32L, squares);
    }

    bool rational_action(const Vec& b, bool forcing, double scale,
                         long double target, int cap, Vec& y,
                         long double& residual, long double& approximation) const {
        long double bn = norm2(b);
        y.assign(n, 0);
        residual = 0;
        approximation = 0;
        if (bn == 0) return true;
        std::unique_ptr<double[]> basis(new double[static_cast<size_t>(cap) * n]);
        Vec z(n), alpha, beta, weights;
        alpha.reserve(cap);
        beta.reserve(cap);
        double inv = 1 / static_cast<double>(bn);
        for (int i = 0; i < n; ++i) basis[i] = b[i] * inv;
        bool converged = false;
        bool resolvent_bound = false;
        for (int j = 0; j < cap; ++j) {
            double* q = basis.get() + static_cast<size_t>(j) * n;
            const double* previous = j ? q - n : nullptr;
            double old_beta = j ? beta[j - 1] : 0;
            long double a = 0;
            for (int i = 0; i < n; ++i) {
                z[i] = t * product_row(i, q) -
                       (j ? old_beta * previous[i] : 0);
                a += static_cast<long double>(q[i]) * z[i];
            }
            double aj = static_cast<double>(a);
            long double norm = 0;
            for (int i = 0; i < n; ++i) {
                z[i] -= aj * q[i];
                norm += static_cast<long double>(z[i]) * z[i];
            }
            double bj = static_cast<double>(std::sqrt(norm));
            alpha.push_back(aj);
            beta.push_back(bj);
            if ((j >= 15 && (j + 1) % 8 == 0) || j + 1 == cap ||
                bj < 1e-14 * t * upper) {
                long double estimate = rational_projection(alpha, beta, forcing,
                                                           scale, weights) * bn * bj;
                resolvent_bound = estimate <= target;
                if (estimate > target &&
                    ((j + 1) % 32 == 0 || j + 1 == cap)) {
                    estimate = std::min(estimate,
                        defect_corner(alpha, beta, forcing) * bn * bj * scale);
                }
                if (estimate <= target) {
                    residual = estimate;
                    converged = true;
                    break;
                }
            }
            if (j + 1 < cap) {
                double* next = q + n;
                double reciprocal = 1 / bj;
                for (int i = 0; i < n; ++i) next[i] = z[i] * reciprocal;
            }
        }
        if (!converged) {
            if (std::getenv("FAMEXP_DEBUG"))
                std::fprintf(stderr, "cram fallback after %zu steps\n", alpha.size());
            return false;
        }
        cblas_dgemv(CblasColMajor, CblasNoTrans, n, static_cast<int>(weights.size()),
                    static_cast<double>(bn), basis.get(), n, weights.data(), 1,
                    0, y.data(), 1);
        if (!forcing) {
            for (int i = 0; i < n; ++i)
                y[i] += static_cast<double>(cram_constant * scale) * b[i];
        }
        if (resolvent_bound) {
            approximation = forcing ? 1e-13L * norm2(y) : 2.5e-16L * bn;
        } else {
            long double basis_bound = std::sqrt(static_cast<long double>(weights.size()));
            approximation = forcing ? 3.5e-14L * bn * basis_bound * norm2(weights) :
                                      3e-16L * bn * basis_bound;
        }
        if (std::getenv("FAMEXP_DEBUG"))
            std::fprintf(stderr, "cram %s steps=%zu residual=%.4Lg\n",
                         forcing ? "phi" : "exp", weights.size(), residual);
        return true;
    }

    bool krylov(const Vec& b0, const Vec& b1, long double target,
                int degree, Vec& y, double offset) const {
        int cap = std::min(32, degree / 5);
        const char* forced = std::getenv("FAMEXP_KRYLOV");
        if (forced && std::atoi(forced) > 0) cap = std::atoi(forced);
        if (cap < 32) return false;
        long double residual0 = 0, residual1 = 0, floor0 = 0, floor1 = 0;
        Vec y1;
        if (!rational_action(b0, false, 1, target / 2, cap, y, residual0, floor0))
            return false;
        if (!rational_action(b1, true, t, target / 2, cap, y1, residual1, floor1))
            return false;
        for (int i = 0; i < n; ++i) y[i] += y1[i];
        double normy = 0;
        for (double v : y) normy = std::max(normy, std::abs(v + offset));
        long double tolerance = atol + rtol * normy;
        long double floor = floor0 + floor1;
        bool passed = residual0 + residual1 + floor <= 0.4L * tolerance;
        if (std::getenv("FAMEXP_DEBUG"))
            std::fprintf(stderr, "cram check=%d floor=%.4Lg norm=%.8g\n",
                         passed, floor, normy);
        return passed;
    }

    void clenshaw(const Vec& b0, const Vec& b1, int degree, Vec& y) const {
        if (small_operator) {
            using Q = __float128;
            std::vector<Q> u(n, 0), d(n, 0), next(n);
            Q factor = static_cast<Q>(-4) / static_cast<Q>(upper);
            for (int k = degree; k >= 1; --k) {
                for (int i = 0; i < n; ++i) {
                    Q dot = static_cast<Q>(diagonal[i]) * u[i];
                    for (int j = row[i]; j < row[i + 1]; ++j)
                        dot += static_cast<Q>(val[j]) * u[col[j]];
                    d[i] += factor * dot + static_cast<Q>(ce[k]) * b0[i] +
                            static_cast<Q>(cp[k]) * b1[i];
                    next[i] = u[i] + d[i];
                }
                u.swap(next);
            }
            for (int i = 0; i < n; ++i) {
                Q dot = static_cast<Q>(diagonal[i]) * u[i];
                for (int j = row[i]; j < row[i + 1]; ++j)
                    dot += static_cast<Q>(val[j]) * u[col[j]];
                y[i] = static_cast<double>(d[i] + factor / 2 * dot +
                    static_cast<Q>(ce[0]) * b0[i] + static_cast<Q>(cp[0]) * b1[i]);
            }
            return;
        }
        const char* forced = std::getenv("FAMEXP_DIAGONAL");
        if (!diagonal_checked && (!forced || std::atoi(forced) != 0) && degree > 20) {
            diagonal_checked = true;
            diagonal_kernel.reset(new DiagonalKernel(n, row, col, val));
        }
        if (diagonal_kernel && diagonal_kernel->usable()) {
            diagonal_kernel->apply(b0, b1, ce, cp, rowsum, upper, degree, y);
            return;
        }
        const char* sliced_force = std::getenv("FAMEXP_SLICED");
        if (!sliced_checked && (!sliced_force || std::atoi(sliced_force) != 0) && degree > 48) {
            sliced_checked = true;
            sliced_kernel.reset(new SlicedKernel(n, row, col, val, rowsum));
        }
        if (sliced_kernel && sliced_kernel->usable()) {
            sliced_kernel->apply(b0, b1, ce, cp, upper, degree, y);
            return;
        }
        Vec u(n, 0), d(n, 0), next(n);
        double factor = -4 / upper;
        // Reinsch's endpoint form avoids subtracting large Clenshaw sums.
        for (int k = degree; k >= 1; --k) {
            double ek = ce[k], pk = cp[k];
            for (int i = 0; i < n; ++i) {
                double ui = u[i];
                double dot = product_row(i, u.data());
                double di = d[i] + factor * dot + ek * b0[i] + pk * b1[i];
                d[i] = di;
                next[i] = ui + di;
            }
            u.swap(next);
        }
        for (int i = 0; i < n; ++i) {
            double dot = product_row(i, u.data());
            y[i] = d[i] + 0.5 * factor * dot + ce[0] * b0[i] + cp[0] * b1[i];
        }
    }

    bool null_linear_action(const Vec& b0, const Vec& b1, Vec& y, double offset) const {
        if (rtol > 1e-9) return false;
        for (int i = 0; i < std::min(n, 16); ++i) {
            double scale = std::abs(b0[i]) + t * std::abs(b1[i]);
            if (std::abs(b0[i] + t * b1[i]) > 1e-8 * scale) return false;
        }
        long double bn0 = 0, bn1 = 0;
        double maximum = 0, base_norm = 0, scale = 0;
        for (int i = 0; i < n; ++i) {
            y[i] = static_cast<double>(static_cast<long double>(b0[i]) +
                                       static_cast<long double>(t) * b1[i]);
            base_norm = std::max(base_norm, std::abs(y[i]));
            maximum = std::max(maximum, std::abs(y[i] + offset));
            scale = std::max(scale, std::abs(b0[i]) + t * std::abs(b1[i]));
            bn0 += static_cast<long double>(b0[i]) * b0[i];
            bn1 += static_cast<long double>(b1[i]) * b1[i];
        }
        if (base_norm > 1e-8 * scale) return false;
        long double target = 0.2L * (atol + rtol * maximum);
        using Q = __float128;
        Q sum0 = 0, sum1 = 0;
        Q limit0 = static_cast<Q>(target) / t;
        Q limit1 = static_cast<Q>(2 * target) / (static_cast<Q>(t) * t);
        limit0 *= limit0;
        limit1 *= limit1;
        for (int i = 0; i < n; ++i) {
            Q r0 = static_cast<Q>(diagonal[i]) * b0[i];
            Q r1 = static_cast<Q>(diagonal[i]) * b1[i];
            for (int j = row[i]; j < row[i + 1]; ++j) {
                Q v = static_cast<Q>(val[j]);
                r0 += v * b0[col[j]];
                r1 += v * b1[col[j]];
            }
            sum0 += r0 * r0;
            sum1 += r1 * r1;
            if (sum0 > limit0 || sum1 > limit1) return false;
        }
        bn0 = std::sqrt(bn0);
        bn1 = std::sqrt(bn1);
        long double error = t * std::sqrt(static_cast<long double>(sum0)) +
                            0.5L * t * t * std::sqrt(static_cast<long double>(sum1));
        long double guard = 1e-26L * t * input_norm * (bn0 + 0.5L * t * bn1) +
                            2e-18L * (bn0 + t * bn1) +
                            4 * std::numeric_limits<double>::epsilon() *
                                (base_norm + std::abs(offset));
        return error * 1.00001L + guard <= target;
    }

    Vec solve_core(const Vec& b0, const Vec& b1, double offset) const {
        Vec y(n);
        if (t == 0 || upper == 0 || identity_step) {
            for (int i = 0; i < n; ++i) y[i] = b0[i] + t * b1[i];
            return y;
        }
        if (raw_taylor) return taylor(b0, b1);
        if (diagonal_only) {
            for (int i = 0; i < n; ++i)
                y[i] = diagonal_exp[i] * b0[i] + diagonal_phi[i] * b1[i];
            return y;
        }
        if (null_linear_action(b0, b1, y, offset)) return y;
        if (direct) {
            if (!pairs.empty()) {
                apply_pairs(pairs, b0, b1, y);
                return y;
            }
            if (!fast_pairs.empty()) {
                apply_pairs(fast_pairs, b0, b1, y);
                return y;
            }
            for (const auto& b : blocks) {
                int m = static_cast<int>(b.ids.size());
                long double result[64] = {};
                for (int k = 0; k < m; ++k) {
                    long double x0 = 0, x1 = 0, c0 = 0, c1 = 0;
                    for (int i = 0; i < m; ++i) {
                        long double a0 = b.q[i + m * k] * b0[b.ids[i]] - c0;
                        long double a1 = b.q[i + m * k] * b1[b.ids[i]] - c1;
                        long double s0 = x0 + a0, s1 = x1 + a1;
                        c0 = (s0 - x0) - a0;
                        c1 = (s1 - x1) - a1;
                        x0 = s0;
                        x1 = s1;
                    }
                    long double z = b.ex[k] * x0 + b.ph[k] * x1;
                    for (int i = 0; i < m; ++i) result[i] += b.q[i + m * k] * z;
                }
                for (int i = 0; i < m; ++i) y[b.ids[i]] = static_cast<double>(result[i]);
            }
            return y;
        }
        if (t * upper <= 0.25) return taylor(b0, b1);
        bool constant_vectors = true;
        for (int i = 1; i < n; ++i) {
            if (b0[i] != b0[0] || b1[i] != b1[0]) {
                constant_vectors = false;
                break;
            }
        }
        if (constant_vectors && exact_constant_null()) {
            double value = static_cast<double>(static_cast<long double>(b0[0]) +
                                               static_cast<long double>(t) * b1[0]);
            std::fill(y.begin(), y.end(), value);
            return y;
        }
        long double n0 = 0, n1 = 0;
        double proxy = 0;
        for (int i = 0; i < n; ++i) {
            n0 += b0[i] * b0[i];
            n1 += b1[i] * b1[i];
            double z = t * diagonal[i], estimate;
            if (t * upper < 1) estimate = b0[i] + t * b1[i];
            else if (z > 40) estimate = b1[i] / diagonal[i];
            else estimate = std::exp(-z) * b0[i] + t * phi(-z) * b1[i];
            proxy = std::max(proxy, std::abs(estimate + offset));
        }
        n0 = std::sqrt(n0);
        n1 = std::sqrt(n1);
        int degree = 0;
        long double target = 0.05L * (atol + rtol * proxy);
        while (degree + 1 < static_cast<int>(ce.size()) &&
               n0 * tail_e[degree] + n1 * tail_p[degree] > target) ++degree;
        const char* forced = std::getenv("FAMEXP_KRYLOV");
        bool use_krylov = t * upper > 300 && upper > 6 * mean_diagonal &&
                          mean_distance > 0.025 * n;
        if (forced) use_krylov = std::atoi(forced) > 0;
        const char* block_force = std::getenv("FAMEXP_BLOCK");
        bool use_block = degree > 200 && mean_distance > 0.025 * n;
        int block_cap = degree > 1200 ? 64 : 48;
        if (block_force) {
            use_block = std::atoi(block_force) > 0;
            block_cap = std::atoi(block_force);
        }
        if (use_krylov && upper > 1000 * mean_diagonal && !krylov_disabled) {
            if (krylov(b0, b1, target, degree, y, offset)) return y;
            krylov_disabled = true;
        }
        if (use_block && !block_disabled) {
            shifted::Matrix view{n, row.data(), col.data(), val.data(),
                                 diagonal.data(), rowsum.data(), t};
            if (shifted::solve(view, b0, b1, cram_theta, cram_alpha, cram_phi, cram_constant,
                               atol, rtol, target, block_cap, y, offset)) return y;
            block_disabled = true;
            krylov_disabled = true;
        }
        if (use_krylov && !krylov_disabled) {
            if (krylov(b0, b1, target, degree, y, offset)) return y;
            krylov_disabled = true;
        }
        clenshaw(b0, b1, degree, y);
        double normy = 0;
        while (true) {
            normy = 0;
            for (double v : y) normy = std::max(normy, std::abs(v + offset));
            target = 0.15L * (atol + rtol * normy);
            if (n0 * tail_e[degree] + n1 * tail_p[degree] <= target) break;
            while (degree + 1 < static_cast<int>(ce.size()) &&
                   n0 * tail_e[degree] + n1 * tail_p[degree] > target * 0.3L) ++degree;
            clenshaw(b0, b1, degree, y);
        }
        if (std::getenv("FAMEXP_DEBUG")) {
            std::fprintf(stderr, "cheb degree=%d upper=%.8g eta=%.8g norm=%.8g\n",
                         degree, upper, t * upper, normy);
        }
        return y;
    }

    static double safe_center(const Vec& b) {
        int sign = 0;
        double minimum = std::numeric_limits<double>::infinity(), maximum = 0;
        for (size_t i = 0; i < std::min<size_t>(b.size(), 16); ++i) {
            if (b[i] == 0) continue;
            int s = b[i] > 0 ? 1 : -1;
            if (sign && sign != s) return 0;
            sign = s;
            minimum = std::min(minimum, std::abs(b[i]));
            maximum = std::max(maximum, std::abs(b[i]));
        }
        if (maximum > 2 * minimum) return 0;
        long double sum = 0, correction = 0;
        for (double v : b) {
            long double add = static_cast<long double>(v) - correction;
            long double next = sum + add;
            correction = (next - sum) - add;
            sum = next;
        }
        double center = static_cast<double>(sum / b.size());
        if (center == 0) return 0;
        // An error-free TwoSum check makes this an exact input decomposition.
        for (double v : b) {
            double difference = v - center;
            double part = difference - v;
            double error = (v - (difference - part)) + (-center - part);
            if (error != 0) return 0;
        }
        return center;
    }

    Vec solve(const Vec& b0, const Vec& b1) const {
        if (raw_taylor || direct || diagonal_only || identity_step || upper == 0 ||
            t * upper < 20 || constant_null_status < 0)
            return solve_core(b0, b1, 0);
        double c0 = safe_center(b0), c1 = safe_center(b1);
        if ((c0 == 0 && c1 == 0) || !exact_constant_null())
            return solve_core(b0, b1, 0);
        Vec r0(n), r1(n);
        for (int i = 0; i < n; ++i) {
            r0[i] = b0[i] - c0;
            r1[i] = b1[i] - c1;
        }
        double offset = static_cast<double>(static_cast<long double>(c0) +
                                            static_cast<long double>(t) * c1);
        if (zero_rhs(r0, r1)) return Vec(n, offset);
        Vec y = solve_core(r0, r1, offset);
        for (double& value : y) value += offset;
        return y;
    }
};

int main() {
    try {
        double warm_matrix = 1, warm_result = 0;
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 1, 1, 1, 1,
                    &warm_matrix, 1, &warm_matrix, 1, 0, &warm_result, 1);
        configure_pipe(0);
        configure_pipe(1);
        write_all("READY\n", 6);
        char prefix[12];
        read_all(prefix, 12);
        if (std::memcmp(prefix, "FAMEXP2\n", 8) != 0) return 2;
        uint32_t len;
        std::memcpy(&len, prefix + 8, 4);
        if (len > 4096) return 2;
        std::string text(len, '\0');
        read_all(text.data(), len);
        auto h = parse_header(text);
        Solver s;
        s.n = std::stoi(h.at("n"));
        s.requests = std::stoi(h.at("requests"));
        s.t = std::strtod(h.at("t").c_str(), nullptr);
        s.rtol = std::strtod(h.at("rtol").c_str(), nullptr);
        s.atol = std::strtod(h.at("atol").c_str(), nullptr);
        int nnz = std::stoi(h.at("nnz"));
        Vec b0(s.n), b1(s.n);
        {
            std::vector<uint64_t> ptr(s.n + 1);
            std::vector<uint32_t> idx(nnz);
            Vec data(nnz);
            read_blocks({{ptr.data(), ptr.size() * sizeof(uint64_t)},
                         {idx.data(), idx.size() * sizeof(uint32_t)},
                         {data.data(), data.size() * sizeof(double)},
                         {b0.data(), b0.size() * sizeof(double)},
                         {b1.data(), b1.size() * sizeof(double)}});
            if (s.requests == 1 && zero_rhs(b0, b1)) {
                write_all(b0.data(), b0.size() * sizeof(double));
                ::_exit(0);
            }
            if (s.t != 0) s.prepare(ptr, std::move(idx), std::move(data));
        }
        for (int j = 0; j < s.requests; ++j) {
            if (j) {
                read_blocks({{b0.data(), b0.size() * sizeof(double)},
                             {b1.data(), b1.size() * sizeof(double)}});
            }
            if (zero_rhs(b0, b1)) {
                write_all(b0.data(), b0.size() * sizeof(double));
                continue;
            }
            if (!s.signs.empty()) {
                for (int i = 0; i < s.n; ++i) {
                    b0[i] *= s.signs[i];
                    b1[i] *= s.signs[i];
                }
            }
            Vec y = s.solve(b0, b1);
            if (!s.signs.empty())
                for (int i = 0; i < s.n; ++i) y[i] *= s.signs[i];
            write_all(y.data(), y.size() * sizeof(double));
        }
        ::_exit(0);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "solver: %s\n", e.what());
        return 1;
    }
}
