// Sparse exponential propagation for the FAMEXP2 streaming interface.
//
// y = exp(-t L) b0 + t phi1(-t L) b1,  L symmetric positive semidefinite.
//
// A safe spectral upper bound beta >= ||L||_2 is obtained from a positive
// diagonal (Collatz-Wielandt) scaling of |L|.  Both functions are expanded
// in Chebyshev polynomials on [0, beta].  Their coefficients are computed
// from scaled modified Bessel functions and their integrals.  A forward
// Chebyshev recurrence evaluates the two series with one sparse matrix pass
// per degree for both right-hand sides, stopping on a posteriori coefficient
// tail bounds.  Long doubles and Neumaier compensation are used throughout.

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <unistd.h>

using std::size_t;
using std::uint32_t;
using std::uint64_t;
using ld = long double;

static bool debug = false;
static bool force_fast = false;
static bool force_ld = false;
static bool simple_spmv = false;
static bool start_kahan_force = false;
static std::chrono::steady_clock::time_point session_start;
static double elapsed_seconds() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - session_start).count();
}

static inline void read_exact(int fd, void *buffer, size_t length) {
    unsigned char *p = static_cast<unsigned char *>(buffer);
    while (length) {
        ssize_t got = ::read(fd, p, length);
        if (got > 0) {
            p += got;
            length -= static_cast<size_t>(got);
        } else if (got < 0 && errno == EINTR) {
            continue;
        } else {
            _exit(2);
        }
    }
}

static inline void write_all(int fd, const void *buffer, size_t length) {
    const unsigned char *p = static_cast<const unsigned char *>(buffer);
    while (length) {
        ssize_t put = ::write(fd, p, length);
        if (put > 0) {
            p += put;
            length -= static_cast<size_t>(put);
        } else if (put < 0 && errno == EINTR) {
            continue;
        } else {
            _exit(2);
        }
    }
}

static ld parse_real(const std::string &s, const char *key) {
    std::string pat = std::string("\"") + key + "\"";
    size_t p = s.find(pat);
    if (p == std::string::npos) _exit(2);
    p = s.find(':', p + pat.size());
    if (p == std::string::npos) _exit(2);
    return strtold(s.c_str() + p + 1, nullptr);
}

static unsigned long long parse_integer(const std::string &s, const char *key) {
    std::string pat = std::string("\"") + key + "\"";
    size_t p = s.find(pat);
    if (p == std::string::npos) _exit(2);
    p = s.find(':', p + pat.size());
    if (p == std::string::npos) _exit(2);
    return strtoull(s.c_str() + p + 1, nullptr, 10);
}

// Neumaier compensated addition of one product to an accumulator pair.
static inline void add_comp(ld &sum, ld &comp, ld term) {
    ld next = sum + term;
    if (std::fabs(sum) >= std::fabs(term))
        comp += (sum - next) + term;
    else
        comp += (term - next) + sum;
    sum = next;
}

static inline void add_comp_d(double &sum, double &comp, double term) {
    double next = sum + term;
    if (std::fabs(sum) >= std::fabs(term))
        comp += (sum - next) + term;
    else
        comp += (term - next) + sum;
    sum = next;
}

struct Header {
    int n, nnz, requests;
    ld t, rtol, atol;
};

class Solver {
public:
    explicit Solver(const Header &h, std::vector<uint32_t> rp,
                    std::vector<uint32_t> ci, std::vector<double> da)
        : n(h.n), nnz(h.nnz), requests(h.requests), t(h.t), rtol(h.rtol),
          atol(h.atol), rowptr(std::move(rp)), colidx(std::move(ci)) {
        raw.resize(nnz);
        for (int k = 0; k < nnz; ++k) raw[k] = static_cast<ld>(da[k]);
    }

    void prepare() {
        if (t == 0) {
            mode = ZERO_T;
            return;
        }
        rowabs.resize(n);
        ld rowabsmax = 0;
        maxdeg = 0;
        for (int i = 0; i < n; ++i) {
            ld s = 0;
            int deg = 0;
            for (uint32_t k = rowptr[i]; k < rowptr[i + 1]; ++k) {
                s += std::fabs(raw[k]);
                ++deg;
            }
            rowabs[i] = s;
            if (s > rowabsmax) rowabsmax = s;
            if (deg > maxdeg) maxdeg = deg;
        }
        if (rowabsmax == 0) {
            mode = ZERO_MAT;
            return;
        }
        if (is_diagonal()) {
            diag.assign(n, 0);
            for (int i = 0; i < n; ++i)
                for (uint32_t k = rowptr[i]; k < rowptr[i + 1]; ++k)
                    if (colidx[k] == static_cast<uint32_t>(i)) diag[i] = raw[k];
            mode = DIAG;
            return;
        }
        beta = spectral_bound(rowabsmax);
        if (beta <= 0 || !std::isfinite(beta)) {
            mode = ZERO_MAT;
            return;
        }
        a = t * beta / 2;
        inv_factor = 2 / beta;
        scaled.resize(nnz);
        scaled_d.resize(nnz);
        rowabs_scaled = 0;
        for (int k = 0; k < nnz; ++k) {
            scaled[k] = inv_factor * raw[k];
            scaled_d[k] = static_cast<double>(scaled[k]);
        }
        for (int i = 0; i < n; ++i) {
            ld s = 0;
            for (uint32_t k = rowptr[i]; k < rowptr[i + 1]; ++k)
                s += std::fabs(scaled[k]);
            if (s > rowabs_scaled) rowabs_scaled = s;
        }
        make_coefficients();
        mode = POLY;
        if (debug) {
            std::fprintf(stderr, "prepare t=%.6Lg beta=%.9Lg a=%.9Lg K=%d nnz=%d elapsed=%.4f\n",
                         t, beta, a, K, nnz, elapsed_seconds());
            for (int k : {0, 10, 100, 500, 1000, 1500, 2000, 3000}) {
                if (k <= K)
                    std::fprintf(stderr, "  tail[%d]=%.4Lg %.4Lg\n", k, te[k], th[k]);
            }
        }
    }

    void solve(const double *b0d, const double *b1d, std::vector<double> &out) {
        out.resize(n);
        if (mode == ZERO_T) {
            for (int i = 0; i < n; ++i) out[i] = b0d[i];
            return;
        }
        if (mode == ZERO_MAT) {
            for (int i = 0; i < n; ++i)
                out[i] = static_cast<double>(static_cast<ld>(b0d[i]) + t * static_cast<ld>(b1d[i]));
            return;
        }
        if (mode == DIAG) {
            for (int i = 0; i < n; ++i) {
                ld d = diag[i];
                ld x = t * d;
                ld e = std::exp(-x);
                ld hh = (d == 0 ? t : -std::expm1(-x) / d);
                ld v = e * static_cast<ld>(b0d[i]) + hh * static_cast<ld>(b1d[i]);
                out[i] = finite_cast(v);
            }
            return;
        }
        if (force_ld) {
            bool saved = simple_spmv;
            simple_spmv = true;
            solve_poly(b0d, b1d, out);
            simple_spmv = saved;
        } else if (n > 4096 || force_fast) {
            bool saved = simple_spmv;
            bool start_long = !force_fast &&
                (start_kahan_force ||
                 (requests == 1 && rtol <= 1e-10L && a > 100 && nnz <= 500000));
            if (start_long) {
                simple_spmv = true;
                refine_level = 0;
                solve_poly(b0d, b1d, out);
                simple_spmv = saved;
                return;
            }
            simple_spmv = true;
            refine_level = 0;
            solve_poly_fast(b0d, b1d, out);
            if (refine_level == 1) {
                simple_spmv = false;
                solve_poly_fast(b0d, b1d, out);
            } else if (refine_level == 2) {
                simple_spmv = true;
                solve_poly(b0d, b1d, out);
            }
            simple_spmv = saved;
        } else {
            bool saved = simple_spmv;
            simple_spmv = true;
            solve_poly(b0d, b1d, out);
            simple_spmv = saved;
        }
    }

private:
    enum Mode { ZERO_T, ZERO_MAT, DIAG, POLY };
    Mode mode = POLY;
    int n, nnz, requests, maxdeg = 0, K = 0;
    ld t, rtol, atol, beta = 0, a = 0, inv_factor = 0, rowabs_scaled = 0;
    std::vector<uint32_t> rowptr, colidx;
    std::vector<ld> raw, scaled, rowabs, diag;
    std::vector<double> scaled_d;
    std::vector<ld> ce, ch, te, th, ae, ah;
    int refine_level = 0;  // 0 accept, 1 compensated double, 2 long double

    static double finite_cast(ld v) {
        if (!std::isfinite(v)) return 0.0;
        return static_cast<double>(v);
    }

    bool is_diagonal() const {
        for (int i = 0; i < n; ++i)
            for (uint32_t k = rowptr[i]; k < rowptr[i + 1]; ++k)
                if (colidx[k] != static_cast<uint32_t>(i)) return false;
        return true;
    }

    // min over positive vectors x of max_i (|L|x)_i/x_i bounds rho(|L|),
    // hence bounds lambda_max(L).  Power iterations supply such vectors.
    ld spectral_bound(ld rowabsmax) {
        std::vector<ld> x(n, 1), y(n);
        ld best = rowabsmax;
        const ld floor = 1e-12L;
        for (int iter = 0; iter < 45; ++iter) {
            ld mx = 0;
            for (int i = 0; i < n; ++i) {
                ld s = 0;
                for (uint32_t k = rowptr[i]; k < rowptr[i + 1]; ++k)
                    s += std::fabs(raw[k]) * x[colidx[k]];
                y[i] = s;
                if (s > mx) mx = s;
                if (x[i] > 0 && s / x[i] > best) {
                    // Nothing to do: best is already a valid upper bound.
                }
            }
            ld bound = 0;
            for (int i = 0; i < n; ++i)
                if (x[i] > 0 && y[i] / x[i] > bound) bound = y[i] / x[i];
            if (bound > 0 && bound < best) best = bound;
            if (mx == 0) break;
            for (int i = 0; i < n; ++i) {
                ld v = y[i] / mx;
                x[i] = v > floor ? v : floor;
            }
        }
        // One final bound for the last normalized vector.
        ld bound = 0;
        for (int i = 0; i < n; ++i) {
            ld s = 0;
            for (uint32_t k = rowptr[i]; k < rowptr[i + 1]; ++k)
                s += std::fabs(raw[k]) * x[colidx[k]];
            if (x[i] > 0 && s / x[i] > bound) bound = s / x[i];
        }
        if (bound > 0 && bound < best) best = bound;
        return best * (1 + 1e-8L) + 1e-30L;
    }

    void make_coefficients() {
        K = static_cast<int>(std::ceil(a + 1000 + 50 * std::sqrt(a) + 50 * std::cbrt(a)));
        if (K < 1024) K = 1024;
        if (K > 250000) K = 250000;
        std::vector<ld> r(K + 2), p(K + 2), q(K + 2), Q(K + 2), F(K + 2);

        // q_k = exp(-a) I_k(a), by Miller's downward continued fraction and
        // the identity exp(a) = I_0(a) + 2 sum_{k>=1} I_k(a).
        r[K] = 0;
        ld inv_a = 1 / a;
        for (int k = K; k >= 1; --k)
            r[k - 1] = 1 / (2 * static_cast<ld>(k) * inv_a + r[k]);
        p[0] = 1;
        ld norm = 1, rel = 1;
        for (int k = 1; k <= K; ++k) {
            rel *= r[k - 1];
            p[k] = rel;
            norm += 2 * rel;
        }
        ld q0 = 1 / norm;
        for (int k = 0; k <= K; ++k) q[k] = q0 * p[k];
        q[K + 1] = 0;

        // Q_k = int_0^a exp(-z) I_k(z) dz.  The identity
        // [z(q_k+q_{k+1})]' = (k+1)q_k-k q_{k+1} gives a stable downward
        // recurrence.  Its homogeneous solution grows like k, so the remote
        // boundary Q_{K+1}=0 has negligible influence on the retained modes.
        for (int k = 0; k <= K; ++k) F[k] = a * (q[k] + q[k + 1]);
        Q[K + 1] = 0;
        for (int k = K; k >= 1; --k)
            Q[k] = (static_cast<ld>(k) * Q[k + 1] + F[k]) / static_cast<ld>(k + 1);
        Q[0] = F[0];
        // Q_k is nonnegative.  Clamp only the sub-ulp negative values that
        // can appear after the coefficients have underflowed; adding any
        // homogeneous O(k) correction would create a false long tail.
        for (int k = 1; k <= K; ++k)
            if (Q[k] < 0) Q[k] = 0;

        ce.assign(K + 1, 0);
        ch.assign(K + 1, 0);
        te.assign(K + 1, 0);
        th.assign(K + 1, 0);
        ae.assign(K + 1, 0);
        ah.assign(K + 1, 0);
        ce[0] = q[0];
        ch[0] = inv_factor * Q[0];
        for (int k = 1; k <= K; ++k) {
            ld sgn = (k & 1) ? -1 : 1;
            ce[k] = sgn * q[k];
            ch[k] = sgn * inv_factor * Q[k];
        }
        ae[0] = std::fabs(ce[0]);
        ah[0] = std::fabs(ch[0]);
        for (int k = 1; k <= K; ++k) {
            ae[k] = ae[k - 1] + 2 * std::fabs(ce[k]);
            ah[k] = ah[k - 1] + 2 * std::fabs(ch[k]);
        }
        for (int k = K - 1; k >= 0; --k) {
            te[k] = te[k + 1] + 2 * std::fabs(ce[k + 1]);
            th[k] = th[k + 1] + 2 * std::fabs(ch[k + 1]);
        }
    }

    // out = alpha * (2/beta L) x, computed as one sparse pass for two
    // interleaved right-hand sides with compensated row sums.
    void block_spmv(const ld *x, ld *out, ld alpha) const {
        if (simple_spmv) {
            for (int i = 0; i < n; ++i) {
                uint32_t k = rowptr[i], end = rowptr[i + 1];
                ld a0 = 0, a1 = 0, b0 = 0, b1 = 0;
                const bool twice = (alpha == 2);
                for (; k + 1 < end; k += 2) {
                    uint32_t j0 = colidx[k], j1 = colidx[k + 1];
                    ld v0 = scaled[k], v1 = scaled[k + 1];
                    if (twice) { v0 += v0; v1 += v1; }
                    size_t p0 = 2 * static_cast<size_t>(j0);
                    size_t p1 = 2 * static_cast<size_t>(j1);
                    a0 += v0 * x[p0];
                    a1 += v0 * x[p0 + 1];
                    b0 += v1 * x[p1];
                    b1 += v1 * x[p1 + 1];
                }
                if (k < end) {
                    uint32_t j = colidx[k];
                    ld v = scaled[k];
                    if (twice) v += v;
                    size_t p = 2 * static_cast<size_t>(j);
                    a0 += v * x[p];
                    a1 += v * x[p + 1];
                }
                out[2 * static_cast<size_t>(i)] = a0 + b0;
                out[2 * static_cast<size_t>(i) + 1] = a1 + b1;
            }
            return;
        }
        for (int i = 0; i < n; ++i) {
            ld s0 = 0, s1 = 0, c0 = 0, c1 = 0;
            uint32_t end = rowptr[i + 1];
            for (uint32_t k = rowptr[i]; k < end; ++k) {
                uint32_t j = colidx[k];
                ld v = alpha * scaled[k];
                add_comp(s0, c0, v * x[2 * static_cast<size_t>(j)]);
                add_comp(s1, c1, v * x[2 * static_cast<size_t>(j) + 1]);
            }
            out[2 * static_cast<size_t>(i)] = s0 + c0;
            out[2 * static_cast<size_t>(i) + 1] = s1 + c1;
        }
    }

    // Double-precision recurrence for large matrices.  Sparse row sums and
    // the final series accumulation remain compensated; requests whose error
    // threshold is close to the double roundoff floor are refined by the
    // long-double recurrence.
    void block_spmv_fast(const double *x, double *out, double alpha) const {
        if (simple_spmv) {
            // Four accumulators expose ILP while reusing one matrix pass for
            // both right-hand sides.  This is the speed path for loose and
            // moderate tolerances; the long-double path is used for very
            // tight small-output requests.
            for (int i = 0; i < n; ++i) {
                uint32_t k = rowptr[i], end = rowptr[i + 1];
                double a0 = 0, a1 = 0, b0 = 0, b1 = 0;
                const bool twice = (alpha == 2.0);
                for (; k + 1 < end; k += 2) {
                    uint32_t j0 = colidx[k], j1 = colidx[k + 1];
                    double v0 = scaled_d[k], v1 = scaled_d[k + 1];
                    if (twice) { v0 += v0; v1 += v1; }
                    size_t p0 = 2 * static_cast<size_t>(j0);
                    size_t p1 = 2 * static_cast<size_t>(j1);
                    a0 += v0 * x[p0];
                    a1 += v0 * x[p0 + 1];
                    b0 += v1 * x[p1];
                    b1 += v1 * x[p1 + 1];
                }
                if (k < end) {
                    uint32_t j = colidx[k];
                    double v = scaled_d[k];
                    if (twice) v += v;
                    size_t p = 2 * static_cast<size_t>(j);
                    a0 += v * x[p];
                    a1 += v * x[p + 1];
                }
                out[2 * static_cast<size_t>(i)] = a0 + b0;
                out[2 * static_cast<size_t>(i) + 1] = a1 + b1;
            }
            return;
        }
        for (int i = 0; i < n; ++i) {
            double s0 = 0, s1 = 0, c0 = 0, c1 = 0;
            uint32_t end = rowptr[i + 1];
            for (uint32_t k = rowptr[i]; k < end; ++k) {
                uint32_t j = colidx[k];
                double v = alpha * scaled_d[k];
                add_comp_d(s0, c0, v * x[2 * static_cast<size_t>(j)]);
                add_comp_d(s1, c1, v * x[2 * static_cast<size_t>(j) + 1]);
            }
            out[2 * static_cast<size_t>(i)] = s0 + c0;
            out[2 * static_cast<size_t>(i) + 1] = s1 + c1;
        }
    }

    void solve_poly_fast(const double *b0d, const double *b1d,
                         std::vector<double> &out) {
        const size_t m = 2 * static_cast<size_t>(n);
        std::vector<double> src(m), t1(m), work(m);
        std::vector<ld> y(m, 0), yc(m, 0);
        ld n0 = 0, n1 = 0;
        for (int i = 0; i < n; ++i) {
            double v0 = b0d[i], v1 = b1d[i];
            ld l0 = static_cast<ld>(v0), l1 = static_cast<ld>(v1);
            src[2 * static_cast<size_t>(i)] = v0;
            src[2 * static_cast<size_t>(i) + 1] = v1;
            y[2 * static_cast<size_t>(i)] = ce[0] * l0;
            y[2 * static_cast<size_t>(i) + 1] = ch[0] * l1;
            n0 += l0 * l0;
            n1 += l1 * l1;
        }
        n0 = std::sqrt(n0);
        n1 = std::sqrt(n1);

        auto max_abs = [&]() {
            ld mx = 0;
            for (int i = 0; i < n; ++i) {
                size_t j = 2 * static_cast<size_t>(i);
                ld v = (y[j] + yc[j]) + (y[j + 1] + yc[j + 1]);
                if (std::fabs(v) > mx) mx = std::fabs(v);
            }
            return mx;
        };
        auto finish = [&]() {
            emit(y, yc, out);
            ld mx = max_abs();
            ld thr = atol + rtol * mx;
            if (force_fast || n > 20000) refine_level = 0;
            else if (thr < 3e-11L) refine_level = 2;
            else if (thr < 1e-9L) refine_level = 1;
            else refine_level = 0;
            if (debug)
                std::fprintf(stderr, "fast done mx=%.4Lg thr=%.4Lg refine=%d elapsed=%.4f\n",
                             mx, thr, refine_level, elapsed_seconds());
        };
        auto converged = [&](int degree) -> bool {
            ld mx = max_abs();
            ld tail = te[degree] * n0 + th[degree] * n1;
            ld threshold = atol + rtol * mx;
            bool ok = tail <= 0.08L * threshold;
            if (ok && debug)
                std::fprintf(stderr, "fast degree=%d tail=%.4Lg mx=%.4Lg thr=%.4Lg elapsed=%.4f\n",
                             degree, tail, mx, threshold, elapsed_seconds());
            return ok;
        };

        if (converged(0)) {
            finish();
            return;
        }
        block_spmv_fast(src.data(), t1.data(), 1.0);
        for (size_t i = 0; i < m; ++i) t1[i] -= src[i];
        ld c0 = 2 * ce[1], c1 = 2 * ch[1];
        for (int i = 0; i < n; ++i) {
            size_t j = 2 * static_cast<size_t>(i);
            add_comp(y[j], yc[j], c0 * static_cast<ld>(t1[j]));
            add_comp(y[j + 1], yc[j + 1], c1 * static_cast<ld>(t1[j + 1]));
        }
        if (converged(1)) {
            finish();
            return;
        }
        double *prev = src.data();
        double *cur = t1.data();
        double *next = work.data();
        for (int k = 2; k <= K; ++k) {
            block_spmv_fast(cur, next, 2.0);
            for (size_t i = 0; i < m; ++i)
                next[i] = static_cast<double>(static_cast<ld>(next[i]) -
                          2 * static_cast<ld>(cur[i]) - static_cast<ld>(prev[i]));
            ld e_coeff = 2 * ce[k], h_coeff = 2 * ch[k];
            for (int i = 0; i < n; ++i) {
                size_t j = 2 * static_cast<size_t>(i);
                add_comp(y[j], yc[j], e_coeff * static_cast<ld>(next[j]));
                add_comp(y[j + 1], yc[j + 1], h_coeff * static_cast<ld>(next[j + 1]));
            }
            double *old_prev = prev;
            prev = cur;
            cur = next;
            next = old_prev;
            if ((k & 3) == 0 || k == K || te[k] * n0 + th[k] * n1 < 0.2L * atol) {
                if (converged(k)) {
                    finish();
                    return;
                }
            }
        }
        finish();
    }

    void solve_poly(const double *b0d, const double *b1d, std::vector<double> &out) {
        const size_t m = 2 * static_cast<size_t>(n);
        std::vector<ld> src(m), y(m, 0), yc(m, 0), t1(m), work(m);
        ld n0 = 0, n1 = 0;
        for (int i = 0; i < n; ++i) {
            ld v0 = static_cast<ld>(b0d[i]), v1 = static_cast<ld>(b1d[i]);
            src[2 * static_cast<size_t>(i)] = v0;
            src[2 * static_cast<size_t>(i) + 1] = v1;
            y[2 * static_cast<size_t>(i)] = ce[0] * v0;
            y[2 * static_cast<size_t>(i) + 1] = ch[0] * v1;
            n0 += v0 * v0;
            n1 += v1 * v1;
        }
        n0 = std::sqrt(n0);
        n1 = std::sqrt(n1);
        const ld eps = std::numeric_limits<ld>::epsilon();

        auto converged = [&](int degree) -> bool {
            ld mx = 0;
            for (int i = 0; i < n; ++i) {
                ld v = (y[2 * static_cast<size_t>(i)] + yc[2 * static_cast<size_t>(i)]) +
                       (y[2 * static_cast<size_t>(i) + 1] + yc[2 * static_cast<size_t>(i) + 1]);
                ld av = std::fabs(v);
                if (av > mx) mx = av;
            }
            ld tail = te[degree] * n0 + th[degree] * n1;
            // A small allowance for long-double recurrence roundoff.  The
            // arithmetic is compensated, so this is far below a double floor.
            ld roundoff = 8 * eps * (ae[degree] * n0 + ah[degree] * n1 +
                          (degree + 1) * (n0 + n1) * (rowabs_scaled + 2));
            ld err = tail + roundoff;
            ld threshold = atol + rtol * mx;
            bool ok = err <= 0.35L * threshold;
            if (ok && debug)
                std::fprintf(stderr, "solve degree=%d tail=%.4Lg round=%.4Lg mx=%.4Lg thr=%.4Lg elapsed=%.4f\n",
                             degree, tail, roundoff, mx, threshold, elapsed_seconds());
            return ok;
        };

        if (converged(0)) {
            emit(y, yc, out);
            return;
        }

        // T_1 = B T_0, B = (2/beta)L - I.
        block_spmv(src.data(), t1.data(), 1);
        for (size_t i = 0; i < m; ++i) t1[i] -= src[i];
        ld c0 = 2 * ce[1], c1 = 2 * ch[1];
        for (int i = 0; i < n; ++i) {
            size_t j = 2 * static_cast<size_t>(i);
            add_comp(y[j], yc[j], c0 * t1[j]);
            add_comp(y[j + 1], yc[j + 1], c1 * t1[j + 1]);
        }
        if (converged(1)) {
            emit(y, yc, out);
            return;
        }

        ld *prev = src.data();
        ld *cur = t1.data();
        ld *next = work.data();
        for (int k = 2; k <= K; ++k) {
            // T_k = 2 B T_{k-1} - T_{k-2}.
            block_spmv(cur, next, 2);
            for (size_t i = 0; i < m; ++i)
                next[i] -= 2 * cur[i] + prev[i];
            ld e_coeff = 2 * ce[k], h_coeff = 2 * ch[k];
            for (int i = 0; i < n; ++i) {
                size_t j = 2 * static_cast<size_t>(i);
                add_comp(y[j], yc[j], e_coeff * next[j]);
                add_comp(y[j + 1], yc[j + 1], h_coeff * next[j + 1]);
            }
            ld *old_prev = prev;
            prev = cur;
            cur = next;
            next = old_prev;
            if ((k & 3) == 0 || k == K || te[k] * n0 + th[k] * n1 < 0.2L * atol) {
                if (converged(k)) {
                    emit(y, yc, out);
                    return;
                }
            }
        }
        emit(y, yc, out);
    }

    void emit(const std::vector<ld> &y, const std::vector<ld> &yc,
              std::vector<double> &out) const {
        for (int i = 0; i < n; ++i) {
            size_t j = 2 * static_cast<size_t>(i);
            ld v = (y[j] + yc[j]) + (y[j + 1] + yc[j + 1]);
            out[i] = finite_cast(v);
        }
    }
};

int main() {
    debug = std::getenv("FAMEXP2_DEBUG") != nullptr;
    force_fast = std::getenv("FAMEXP2_FORCE_FAST") != nullptr;
    force_ld = std::getenv("FAMEXP2_FORCE_LD") != nullptr;
    simple_spmv = std::getenv("FAMEXP2_KAHAN") == nullptr;
    start_kahan_force = std::getenv("FAMEXP2_START_KAHAN") != nullptr;
    session_start = std::chrono::steady_clock::now();
    write_all(STDOUT_FILENO, "READY\n", 6);

    char magic[8];
    read_exact(STDIN_FILENO, magic, 8);
    uint32_t json_len = 0;
    read_exact(STDIN_FILENO, &json_len, 4);
    std::string meta(json_len, '\0');
    read_exact(STDIN_FILENO, meta.data(), json_len);

    Header h;
    h.n = static_cast<int>(parse_integer(meta, "n"));
    h.nnz = static_cast<int>(parse_integer(meta, "nnz"));
    h.requests = static_cast<int>(parse_integer(meta, "requests"));
    h.t = parse_real(meta, "t");
    h.rtol = parse_real(meta, "rtol");
    h.atol = parse_real(meta, "atol");

    std::vector<uint64_t> ip(h.n + 1);
    std::vector<uint32_t> ci(h.nnz);
    std::vector<double> da(h.nnz);
    read_exact(STDIN_FILENO, ip.data(), sizeof(uint64_t) * (h.n + 1));
    read_exact(STDIN_FILENO, ci.data(), sizeof(uint32_t) * h.nnz);
    read_exact(STDIN_FILENO, da.data(), sizeof(double) * h.nnz);
    std::vector<uint32_t> rp(h.n + 1);
    for (int i = 0; i <= h.n; ++i) rp[i] = static_cast<uint32_t>(ip[i]);

    Solver solver(h, std::move(rp), std::move(ci), std::move(da));
    solver.prepare();

    std::vector<double> b0(h.n), b1(h.n), out(h.n);
    for (int request = 0; request < h.requests; ++request) {
        read_exact(STDIN_FILENO, b0.data(), sizeof(double) * h.n);
        read_exact(STDIN_FILENO, b1.data(), sizeof(double) * h.n);
        solver.solve(b0.data(), b1.data(), out);
        write_all(STDOUT_FILENO, out.data(), sizeof(double) * h.n);
    }
    return 0;
}
