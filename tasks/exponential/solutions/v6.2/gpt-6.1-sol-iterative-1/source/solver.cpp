#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Vec = std::vector<double>;
using LongVec = std::vector<long double>;

static void read_exact(void* dst, size_t bytes) {
    auto* p = static_cast<unsigned char*>(dst);
    while (bytes) {
        size_t k = std::fread(p, 1, bytes, stdin);
        if (!k) throw std::runtime_error("incomplete input");
        p += k;
        bytes -= k;
    }
}

static void write_exact(const void* src, size_t bytes) {
    const auto* p = static_cast<const unsigned char*>(src);
    while (bytes) {
        size_t k = std::fwrite(p, 1, bytes, stdout);
        if (!k) throw std::runtime_error("output failure");
        p += k;
        bytes -= k;
    }
    std::fflush(stdout);
}

static double field(const std::string& s, const char* key) {
    std::string name = std::string("\"") + key + "\"";
    size_t p = s.find(name);
    if (p == std::string::npos) throw std::runtime_error("missing header field");
    p = s.find(':', p + name.size());
    return std::strtod(s.c_str() + p + 1, nullptr);
}

static long double phi(long double x) {
    return x == 0 ? 1 : -std::expm1(-x) / x;
}

struct Solver {
    int n = 0, nnz = 0;
    double t = 0, rtol = 0, atol = 0, upper = 0;
    bool direct = false, difference = false, taylor = false, prepared = false, matrix_scaled = false;
    std::vector<uint64_t> ptr;
    std::vector<int> col, mate;
    Vec a, original, rowsum, diagonal, e_diag, e_off, p_diag, p_off;
    Vec ecoef, pcoef, etail, ptail;
    Vec q, delta, work, rhs1, y;
    LongVec ecoef_long, pcoef_long, q_long, delta_long, work_long, rowsum_long;
    mutable int multiplies = 0;
    int last_degree = 0, last_attempts = 0;
    bool last_trial = false, last_small = false, used_extended = false;
    double matrix_norm = 0, infinity_factor = INFINITY;

    Solver() : ptr(120001), col(1000000), a(1000000), original(1000000),
               rowsum(120000), diagonal(120000),
               e_diag(120000), e_off(120000), p_diag(120000), p_off(120000),
               q(120000), delta(120000), work(120000), rhs1(120000), y(120000) {}

    void read() {
        char magic[8];
        read_exact(magic, 8);
        if (std::memcmp(magic, "FAMEXP2\n", 8)) throw std::runtime_error("bad magic");
        uint32_t length = 0;
        read_exact(&length, 4);
        std::string header(length, '\0');
        read_exact(header.data(), length);
        n = static_cast<int>(field(header, "n"));
        nnz = static_cast<int>(field(header, "nnz"));
        t = field(header, "t");
        rtol = field(header, "rtol");
        atol = field(header, "atol");
        requests = static_cast<int>(field(header, "requests"));
        ptr.resize(n + 1);
        col.resize(nnz);
        read_exact(ptr.data(), 8 * ptr.size());
        read_exact(col.data(), 4 * col.size());
        a.resize(nnz);
        original.resize(nnz);
        read_exact(original.data(), 8 * original.size());
        for (int i = 0; i < n; ++i) {
            double sum = 0;
            for (int k = ptr[i]; k < static_cast<int>(ptr[i + 1]); ++k) sum += std::abs(original[k]);
            matrix_norm = std::max(matrix_norm, sum);
        }
        matrix_norm *= t;
        upper = matrix_norm * (1 + 1e-10);
        y.resize(n);
        q.resize(n);
        delta.resize(n);
        work.resize(n);
        rhs1.resize(n);
    }

    int requests = 1;

    bool prepare_direct() {
        mate.assign(n, -1);
        diagonal.assign(n, 0);
        Vec off(n, 0);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            int count = 0;
            for (int k = ptr[i]; k < static_cast<int>(ptr[i + 1]); ++k) {
                int j = col[k];
                if (j == i) diagonal[i] = a[k];
                else if (a[k] != 0) {
                    ++count;
                    mate[i] = j;
                    off[i] = a[k];
                }
            }
            if (count > 1) ok = false;
        }
        if (!ok) {
            mate.clear();
            return false;
        }
        e_diag.resize(n);
        e_off.assign(n, 0);
        p_diag.resize(n);
        p_off.assign(n, 0);
        for (int i = 0; i < n; ++i) {
            int j = mate[i];
            if (j < 0) {
                long double x = diagonal[i];
                e_diag[i] = static_cast<double>(std::exp(-x));
                p_diag[i] = static_cast<double>(phi(x));
            } else if (i < j) {
                long double aa = diagonal[i], cc = diagonal[j], b = off[i];
                if (std::max(aa + std::abs(b), cc + std::abs(b)) < 4) {
                    double ad = static_cast<double>(aa), cd = static_cast<double>(cc);
                    double bd = static_cast<double>(b);
                    double h = std::hypot(ad - cd, 2 * bd);
                    double hi = (ad + cd + h) / 2;
                    double lo = hi == 0 ? 0 : static_cast<double>((aa * cc - b * b) / hi);
                    double eh = std::exp(-hi), el = std::exp(-lo);
                    double ph = hi == 0 ? 1 : -std::expm1(-hi) / hi;
                    double pl = lo == 0 ? 1 : -std::expm1(-lo) / lo;
                    double v = static_cast<double>((hi - aa) / h);
                    double w = static_cast<double>((hi - cc) / h);
                    double z = -bd / h;
                    e_diag[i] = eh + (el - eh) * v;
                    e_diag[j] = eh + (el - eh) * w;
                    e_off[i] = e_off[j] = (el - eh) * z;
                    p_diag[i] = ph + (pl - ph) * v;
                    p_diag[j] = ph + (pl - ph) * w;
                    p_off[i] = p_off[j] = (pl - ph) * z;
                    continue;
                }
                long double h = std::hypot(aa - cc, 2 * b);
                long double hi = (aa + cc + h) / 2;
                long double lo = hi == 0 ? 0 : (aa * cc - b * b) / hi;
                long double eh = std::exp(-hi), el = std::exp(-lo);
                long double ph = phi(hi), pl = phi(lo);
                long double v = (hi - aa) / h, w = (hi - cc) / h;
                long double z = -b / h;
                e_diag[i] = static_cast<double>(eh + (el - eh) * v);
                e_diag[j] = static_cast<double>(eh + (el - eh) * w);
                e_off[i] = e_off[j] = static_cast<double>((el - eh) * z);
                p_diag[i] = static_cast<double>(ph + (pl - ph) * v);
                p_diag[j] = static_cast<double>(ph + (pl - ph) * w);
                p_off[i] = p_off[j] = static_cast<double>((pl - ph) * z);
            }
        }
        return true;
    }

    void bound_spectrum() {
        upper = 0;
        difference = true;
        rowsum.resize(n);
        for (int i = 0; i < n; ++i) {
            long double sum = 0, abssum = 0;
            for (int k = ptr[i]; k < static_cast<int>(ptr[i + 1]); ++k) {
                sum += static_cast<long double>(a[k]);
                abssum += std::abs(static_cast<long double>(a[k]));
                if (col[k] != i && a[k] > 0) difference = false;
            }
            rowsum[i] = static_cast<double>(sum);
            upper = std::max(upper, static_cast<double>(abssum));
        }
        if (difference) {
            double growth = 0;
            for (auto x : rowsum) growth = std::max(growth, -x);
            if (growth < 1)
                infinity_factor = std::exp(growth + 4e-12 * std::max(1.0, upper));
        }
        if (upper > 60) {
            // Collatz upper bounds for |A| are valid upper bounds for A.
            Vec v(n, 1), w(n);
            int steps = upper > 400 ? 10 : 4;
            for (int it = 0; it < steps; ++it) {
                double u = 0, mx = 0;
                for (int i = 0; i < n; ++i) {
                    double s = 0;
                    for (int k = ptr[i]; k < static_cast<int>(ptr[i + 1]); ++k)
                        s += std::abs(a[k]) * v[col[k]];
                    w[i] = s;
                    u = std::max(u, s / v[i]);
                    mx = std::max(mx, s);
                }
                upper = std::min(upper, u * (1 + 1e-10));
                if (mx == 0) break;
                for (int i = 0; i < n; ++i) v[i] = std::max(w[i] / mx, 1e-30);
            }
        }
        upper *= 1 + 1e-10;
    }

    void coefficients() {
        long double r = static_cast<long double>(upper) / 2;
        int m = static_cast<int>(std::ceil(std::sqrt(160 * r) + 90));
        std::vector<long double> ratio(m + 2, 0), e(m + 1), p(m + 1);
        for (int k = m; k >= 1; --k)
            ratio[k] = 1 / (2 * static_cast<long double>(k) / r + ratio[k + 1]);
        e[0] = 1;
        long double norm = 1, z = 1;
        for (int k = 1; k <= m; ++k) {
            z *= ratio[k];
            e[k] = 2 * z;
            norm += e[k];
        }
        for (auto& x : e) x /= norm;
        long double sum = 0, weighted = 0;
        for (int k = m; k >= 1; --k) {
            p[k] = 2 * weighted / r;
            sum += e[k];
            weighted += sum;
        }
        p[0] = weighted / r;
        long double psum = 0;
        for (auto x : p) psum += x;
        for (auto& x : p) x /= psum;
        ecoef.resize(m + 1);
        pcoef.resize(m + 1);
        etail.resize(m + 1);
        ptail.resize(m + 1);
        long double se = 0, sp = 0;
        for (int k = m; k >= 0; --k) {
            ecoef[k] = static_cast<double>(e[k]);
            pcoef[k] = static_cast<double>(p[k]);
            etail[k] = static_cast<double>(se);
            ptail[k] = static_cast<double>(sp);
            se += e[k];
            sp += p[k];
        }
        ecoef_long = std::move(e);
        pcoef_long = std::move(p);
    }

    void prepare() {
        prepared = true;
        if (t == 0) return;
        upper = matrix_norm * (1 + 1e-10);
        taylor = n > 8 && (upper <= 0.05 || (upper <= 1.5 && requests == 1));
        if (taylor) return;
        for (int k = 0; k < nnz; ++k) a[k] = original[k] * t;
        matrix_scaled = true;
        if (!taylor) direct = prepare_direct();
        if (direct) return;
        q.resize(n);
        delta.resize(n);
        work.resize(n);
        rhs1.resize(n);
        bound_spectrum();
        if (upper == 0) return;
        coefficients();
        double scale = 4 / upper;
        for (auto& x : a) x *= scale;
        for (auto& x : rowsum) x *= scale;
    }

    __attribute__((noinline)) void multiply(const double* __restrict__ x,
                                            double* __restrict__ out) const {
        ++multiplies;
        if (difference) {
            for (int i = 0; i < n; ++i) {
                double xi = x[i], s = rowsum[i] * xi;
                for (int k = ptr[i]; k < static_cast<int>(ptr[i + 1]); ++k) {
                    int j = col[k];
                    s += a[k] * (x[j] - xi);
                }
                out[i] = s;
            }
        } else {
            const double* values = matrix_scaled ? a.data() : original.data();
            double factor = matrix_scaled ? 1 : t;
            for (int i = 0; i < n; ++i) {
                double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                int k = ptr[i], end = ptr[i + 1];
                for (; k + 3 < end; k += 4) {
                    s0 += values[k] * x[col[k]];
                    s1 += values[k + 1] * x[col[k + 1]];
                    s2 += values[k + 2] * x[col[k + 2]];
                    s3 += values[k + 3] * x[col[k + 3]];
                }
                double s = (s0 + s1) + (s2 + s3);
                for (; k < end; ++k) s += values[k] * x[col[k]];
                out[i] = s * factor;
            }
        }
    }

    int degree(double bnorm, double cnorm, double target) const {
        int lo = 0, hi = static_cast<int>(ecoef.size()) - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (etail[mid] * bnorm + ptail[mid] * cnorm <= target) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    }

    __attribute__((noinline)) void clenshaw(const Vec& b0, int d) {
        std::fill(q.begin(), q.end(), 0);
        std::fill(delta.begin(), delta.end(), 0);
        for (int k = d; k >= 1; --k) {
            if (k != d) multiply(q.data(), work.data());
            else std::fill(work.begin(), work.end(), 0);
            double e = ecoef[k], p = pcoef[k];
            #pragma GCC ivdep
            for (int i = 0; i < n; ++i)
            {
                // Reinsch's form avoids cancellation at the zero eigenvalue.
                double dd = (delta[i] - work[i]) + (e * b0[i] + p * rhs1[i]);
                delta[i] = dd;
                q[i] += dd;
            }
        }
        if (d) multiply(q.data(), work.data());
        else std::fill(work.begin(), work.end(), 0);
        for (int i = 0; i < n; ++i)
            y[i] = (delta[i] - 0.5 * work[i])
                 + (ecoef[0] * b0[i] + pcoef[0] * rhs1[i]);
    }

    void taylor_solve(const Vec& b0, const Vec& b1) {
        bool zero = true;
        for (auto x : b0) if (x != 0) { zero = false; break; }
        difference = false;
        if (!zero) multiply(b0.data(), work.data());
        else std::fill(work.begin(), work.end(), 0);
        for (int i = 0; i < n; ++i) {
            q[i] = t * b1[i] - work[i];
            y[i] = b0[i] + q[i];
        }
        for (int k = 1; k < 80; ++k) {
            double term = 0, ym = 0;
            for (int i = 0; i < n; ++i) {
                term = std::max(term, std::abs(q[i]));
                ym = std::max(ym, std::abs(y[i]));
            }
            double ratio = upper / (k + 1);
            double remainder = term * ratio / (1 - upper / (k + 2));
            if (ratio < 1 && remainder <= 0.02 * (atol + rtol * ym)) break;
            multiply(q.data(), work.data());
            double scale = -1.0 / (k + 1);
            for (int i = 0; i < n; ++i) {
                q[i] = scale * work[i];
                y[i] += q[i];
            }
        }
    }

    bool taylor_trial(const Vec& b0, const Vec& b1) {
        if (n <= 8 || t == 0 || std::getenv("FAMEXP_NO_TRIAL")) return false;
        if (upper <= 0.05 || (upper <= 1.5 && requests == 1)) {
            taylor_solve(b0, b1);
            return true;
        }
        if (upper <= 1.5) return false;
        int support = 0;
        bool b1_zero = true, b0_zero = true;
        for (int i = 0; i < n; ++i) {
            if (b0[i] != 0) { ++support; b0_zero = false; }
            if (b1[i] != 0) b1_zero = false;
        }
        if (upper > 1.5 && (!b1_zero || support > 16)) return false;
        if (rtol < 1e-8 && support > 1) return false;
        if (requests > 1 && nnz <= 2 * n && upper > 0.05) return false;
        difference = false;
        if (!b0_zero) multiply(b0.data(), work.data());
        else std::fill(work.begin(), work.end(), 0);
        for (int i = 0; i < n; ++i) {
            q[i] = t * b1[i] - work[i];
            y[i] = b0[i];
        }
        int limit = upper <= 1.5 ? 32 : 12;
        long double previous = INFINITY;
        for (int k = 0; k <= limit; ++k) {
            long double norm2 = 0;
            double ym = 0;
            for (int i = 0; i < n; ++i) {
                norm2 += static_cast<long double>(q[i]) * q[i];
                ym = std::max(ym, std::abs(y[i]));
            }
            long double norm = std::sqrt(norm2);
            // For PSD A, the phi1 Taylor remainder is bounded by the
            // norm of this first omitted term, without an exp(||A||) factor.
            if (norm <= 0.015 * (atol + rtol * ym)) {
                last_degree = k;
                last_attempts = 1;
                return true;
            }
            if (k == limit || (k > 1 && upper > 1.5 && norm > previous * 0.8)) return false;
            previous = norm;
            for (int i = 0; i < n; ++i) y[i] += q[i];
            multiply(q.data(), work.data());
            double scale = -1.0 / (k + 2);
            for (int i = 0; i < n; ++i) q[i] = scale * work[i];
        }
        return false;
    }

    void precise_product(const LongVec& x, LongVec& out, long double scale);
    void precise_direct(const Vec& b0, const Vec& b1);
    void precise_taylor(const Vec& b0, const Vec& b1);
    void precise_chebyshev(const Vec& b0, const Vec& b1, int degree);
    void refine_if_needed(const Vec& b0, const Vec& b1);

    bool small_input(const Vec& b0, const Vec& b1) const {
        long double bsum = 0, csum = 0, threshold = 0.5L * atol;
        long double threshold2 = threshold * threshold;
        for (int i = 0; i < n; ++i) {
            bsum += static_cast<long double>(b0[i]) * b0[i];
            long double c = static_cast<long double>(t) * b1[i];
            csum += c * c;
            if (bsum + csum > threshold2) return false;
        }
        return std::sqrt(bsum) + std::sqrt(csum) <= threshold;
    }

    const Vec& solve(const Vec& b0, const Vec& b1) {
        last_trial = false;
        last_small = false;
        used_extended = false;
        // The omitted action here is below 1e-93 under the protocol's bounds.
        if (matrix_norm < 1e-100) {
            for (int i = 0; i < n; ++i) y[i] = b0[i] + t * b1[i];
            return y;
        }
        if (small_input(b0, b1)) {
            std::fill(y.begin(), y.end(), 0);
            last_small = true;
            return y;
        }
        if (!prepared) {
            if (taylor_trial(b0, b1)) {
                last_trial = true;
                refine_if_needed(b0, b1);
                return y;
            }
            prepare();
        }
        if (t == 0) {
            y = b0;
        } else if (taylor) {
            taylor_solve(b0, b1);
        } else if (direct) {
            for (int i = 0; i < n; ++i) {
                int j = mate[i];
                double yy = e_diag[i] * b0[i] + t * p_diag[i] * b1[i];
                if (j >= 0) yy += e_off[i] * b0[j] + t * p_off[i] * b1[j];
                y[i] = yy;
            }
        } else if (upper == 0) {
            for (int i = 0; i < n; ++i) y[i] = b0[i] + t * b1[i];
        } else {
            long double bn2 = 0, cn2 = 0;
            double bmax = 0, cmax = 0;
            for (int i = 0; i < n; ++i) {
                rhs1[i] = t * b1[i];
                bn2 += static_cast<long double>(b0[i]) * b0[i];
                cn2 += static_cast<long double>(rhs1[i]) * rhs1[i];
                bmax = std::max(bmax, std::abs(b0[i]));
                cmax = std::max(cmax, std::abs(rhs1[i]));
            }
            double bn = static_cast<double>(std::sqrt(bn2));
            double cn = static_cast<double>(std::sqrt(cn2));
            if (infinity_factor * (bmax + cmax) <= 0.5 * atol) {
                std::fill(y.begin(), y.end(), 0);
                last_small = true;
                return y;
            }
            double estimate = bmax / (1 + upper) + cmax / (1 + std::sqrt(upper));
            double target = 0.015 * (atol + rtol * estimate);
            int d = degree(bn, cn, target);
            for (int attempt = 0; attempt < 3; ++attempt) {
                last_degree = d;
                last_attempts = attempt + 1;
                clenshaw(b0, d);
                double ymax = 0;
                for (auto x : y) ymax = std::max(ymax, std::abs(x));
                target = 0.025 * (atol + rtol * ymax);
                if (etail[d] * bn + ptail[d] * cn <= target) break;
                d = degree(bn, cn, target * 0.5);
            }
        }
        refine_if_needed(b0, b1);
        return y;
    }
};

#include "extended.hpp"

int main() {
    try {
        std::setvbuf(stdin, nullptr, _IOFBF, 1 << 20);
        std::setvbuf(stdout, nullptr, _IOFBF, 1 << 20);
        Solver solver;
        Vec b0(120000), b1(120000);
        bool stats = std::getenv("FAMEXP_STATS") != nullptr;
        write_exact("READY\n", 6);
        auto started = std::chrono::steady_clock::now();
        solver.read();
        auto loaded = std::chrono::steady_clock::now();
        if (stats) {
            std::fprintf(stderr, "read %.6f upper %.12g\n",
                         std::chrono::duration<double>(loaded - started).count(),
                         solver.upper);
        }
        b0.resize(solver.n);
        b1.resize(solver.n);
        for (int j = 0; j < solver.requests; ++j) {
            read_exact(b0.data(), 8 * b0.size());
            read_exact(b1.data(), 8 * b1.size());
            auto before = std::chrono::steady_clock::now();
            solver.multiplies = 0;
            const Vec& y = solver.solve(b0, b1);
            if (stats) {
                auto after = std::chrono::steady_clock::now();
                std::fprintf(stderr, "solve %.6f multiplies %d degree %d attempts %d method %s extended %d\n",
                             std::chrono::duration<double>(after - before).count(),
                             solver.multiplies, solver.last_degree, solver.last_attempts,
                             solver.last_small ? "small_input" : solver.last_trial ? "trial" :
                             solver.direct ? "direct" : solver.taylor ? "taylor" : "chebyshev",
                             solver.used_extended);
            }
            write_exact(y.data(), 8 * y.size());
        }
        std::_Exit(0);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "solver: %s\n", e.what());
        return 1;
    }
    return 0;
}
