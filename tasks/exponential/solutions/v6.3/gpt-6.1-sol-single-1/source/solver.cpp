#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>
#include <unistd.h>
#include <Eigen/Eigenvalues>

using std::vector;
using Real = long double;

static inline double signed_shift(double x, double coefficient) {
    uint64_t a, b;
    std::memcpy(&a, &x, 8);
    std::memcpy(&b, &coefficient, 8);
    a ^= b & (uint64_t(1) << 63);
    std::memcpy(&x, &a, 8);
    return x;
}

static Real exp_factor(Real z) {
    return std::exp(-z);
}

static Real phi_factor(Real z, Real t) {
    return z == 0 ? t : t * (-std::expm1(-z) / z);
}

static Real phi2_factor(Real z, Real t) {
    if (std::abs(z) < 0.01L) {
        Real sum = 0.5L, term = 0.5L;
        for (int k = 1; k < 10; ++k) {
            term *= -z / Real(k + 2);
            sum += term;
        }
        return t * t * sum;
    }
    return (t / z) * (t / z) * (z + std::expm1(-z));
}

static void read_all(void *p, size_t bytes) {
    auto *s = static_cast<unsigned char *>(p);
    while (bytes) {
        ssize_t k = ::read(STDIN_FILENO, s, bytes);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) std::exit(2);
        s += k;
        bytes -= size_t(k);
    }
}

static void write_all(const void *p, size_t bytes) {
    const auto *s = static_cast<const unsigned char *>(p);
    while (bytes) {
        ssize_t k = ::write(STDOUT_FILENO, s, bytes);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) std::exit(3);
        s += k;
        bytes -= size_t(k);
    }
}

static double field(const std::string &s, const std::string &key) {
    size_t p = s.find("\"" + key + "\"");
    if (p == std::string::npos) std::exit(4);
    p = s.find(':', p);
    return std::strtod(s.c_str() + p + 1, nullptr);
}

struct Matrix {
    int n;
    vector<uint32_t> ptr, col;
    vector<double> a;
    vector<uint32_t> offptr, offcol;
    vector<double> offa, residual, diagonal;
    vector<Real> exact_residual, residual_error;
    bool dominant = false, nonpositive = true;
    double upper = 0, lower = 0;
    Real lognorm = 0;

    template<class T> void apply(const T *x, T *y) const {
        // Difference form avoids cancelling the diagonal against a near-null mode.
        if (dominant) {
            if (nonpositive) {
                for (int i = 0; i < n; ++i) {
                    T xi = x[i];
                    T s;
                    if constexpr (std::is_same_v<T, Real>) s = exact_residual[i] * xi;
                    else s = T(residual[i]) * xi;
                    for (uint32_t j = offptr[i]; j < offptr[i + 1]; ++j)
                        s += T(offa[j]) * (x[offcol[j]] - xi);
                    y[i] = s;
                }
                return;
            }
            for (int i = 0; i < n; ++i) {
                T xi = x[i];
                T s;
                if constexpr (std::is_same_v<T, Real>) s = exact_residual[i] * xi;
                else s = T(residual[i]) * xi;
                for (uint32_t j = offptr[i]; j < offptr[i + 1]; ++j) {
                    T ax = T(offa[j]);
                    T dx;
                    if constexpr (std::is_same_v<T, double>) {
                        dx = x[offcol[j]] + signed_shift(xi, offa[j]);
                    } else {
                        dx = x[offcol[j]] + T(std::copysign(1.0, offa[j])) * xi;
                    }
                    s += ax * dx;
                }
                y[i] = s;
            }
            return;
        }
        for (int i = 0; i < n; ++i) {
            T s = 0;
            for (uint32_t j = ptr[i]; j < ptr[i + 1]; ++j)
                s += T(a[j]) * x[col[j]];
            y[i] = s;
        }
    }

    void prepare_stable() {
        residual.resize(n);
        exact_residual.resize(n);
        residual_error.resize(n);
        diagonal.resize(n);
        dominant = true;
        Real minres = std::numeric_limits<Real>::infinity();
        for (int i = 0; i < n; ++i) {
            Real diag = 0, offsum = 0, comp = 0;
            for (uint32_t j = ptr[i]; j < ptr[i + 1]; ++j) {
                if (col[j] == uint32_t(i)) diag += Real(a[j]);
                else {
                    Real term = std::abs(Real(a[j])) - comp;
                    Real next = offsum + term;
                    comp = (next - offsum) - term;
                    offsum = next;
                }
            }
            residual[i] = double(diag - offsum);
            exact_residual[i] = diag - offsum;
            residual_error[i] = 4 * std::numeric_limits<Real>::epsilon() * (std::abs(diag) + offsum);
            diagonal[i] = double(diag);
            minres = std::min(minres, diag - offsum);
            lognorm = std::max(lognorm, offsum - diag + residual_error[i]);
            if (diag < offsum * (1 - 2e-15L)) dominant = false;
        }
        lower = std::max(0.0, double(minres * (1 - 1e-8L)));
        if (!dominant) {
            residual.clear();
            exact_residual.clear();
            residual_error.clear();
            return;
        }
        offptr.resize(n + 1);
        offcol.reserve(col.size());
        offa.reserve(a.size());
        for (int i = 0; i < n; ++i) {
            offptr[i] = uint32_t(offcol.size());
            for (uint32_t j = ptr[i]; j < ptr[i + 1]; ++j) {
                if (col[j] != uint32_t(i) && a[j] != 0) {
                    offcol.push_back(col[j]);
                    offa.push_back(a[j]);
                    if (a[j] > 0) nonpositive = false;
                }
            }
        }
        offptr[n] = uint32_t(offcol.size());
    }

    void bound(bool improve) {
        for (int i = 0; i < n; ++i) {
            Real s = 0;
            for (uint32_t j = ptr[i]; j < ptr[i + 1]; ++j)
                s += std::abs(Real(a[j]));
            upper = std::max(upper, double(s));
        }
        if (improve && upper > 0) {
            vector<double> x(n, 1), y(n);
            const double scale = upper;
            for (int it = 0; it < 12; ++it) {
                double u = 0, ym = 0;
                for (int i = 0; i < n; ++i) {
                    Real s = 0;
                    for (uint32_t j = ptr[i]; j < ptr[i + 1]; ++j)
                        s += std::abs(Real(a[j])) * Real(x[col[j]]);
                    y[i] = double(s / scale + 0.05L * x[i] + 1e-12L);
                    u = std::max(u, double(s / x[i]));
                    ym = std::max(ym, y[i]);
                }
                upper = std::min(upper, u * (1 + 2e-14));
                for (int i = 0; i < n; ++i) x[i] = y[i] / ym;
            }
        }
        upper *= 1 + 4e-14;
    }
};

static bool eigen_solve(const Matrix &a, double t, double rtol, double atol,
                        const double *b0, const double *b1, double *out) {
    if (!a.dominant || t * a.upper < 200) return false;
    int i0 = 0, i1 = 0;
    Real max0 = 0, max1 = 0;
    for (int i = 0; i < a.n; ++i) {
        if (std::abs(b0[i]) > max0) {
            max0 = std::abs(b0[i]);
            i0 = i;
        }
        if (std::abs(b1[i]) > max1) {
            max1 = std::abs(b1[i]);
            i1 = i;
        }
    }
    auto scalar_value = [&](const double *b, int i, Real maxb) {
        if (maxb == 0) return Real(0);
        Real sum = a.exact_residual[i] * Real(b[i]), comp = 0;
        for (uint32_t j = a.offptr[i]; j < a.offptr[i + 1]; ++j) {
            Real sign = std::copysign(1.0, a.offa[j]);
            Real term = Real(a.offa[j]) * (Real(b[a.offcol[j]]) + sign * Real(b[i])) - comp;
            Real next = sum + term;
            comp = (next - sum) - term;
            sum = next;
        }
        return sum / Real(b[i]);
    };
    Real lambda0 = scalar_value(b0, i0, max0), lambda1 = scalar_value(b1, i1, max1);
    if (t * std::min(lambda0, lambda1) < -0.01L) return false;
    Real e0 = exp_factor(Real(t) * lambda0);
    Real p1 = phi_factor(Real(t) * lambda1, t);
    Real w0 = phi_factor(Real(t) * lambda0, t);
    Real w1 = phi2_factor(Real(t) * lambda1, t);
    Real ymax = 0;
    for (int i = 0; i < a.n; ++i) {
        Real y = e0 * b0[i] + p1 * b1[i];
        out[i] = double(y);
        ymax = std::max(ymax, std::abs(y));
    }
    Real factor = std::exp(Real(t) * a.lognorm);
    Real budget = 0.5L * (atol + rtol * ymax);
    Real r0max = 0, r1max = 0;
    const Real eps = std::numeric_limits<Real>::epsilon();
    for (int i = 0; i < a.n; ++i) {
        Real x0 = b0[i], x1 = b1[i];
        Real s0 = a.exact_residual[i] * x0, s1 = a.exact_residual[i] * x1;
        Real a0 = std::abs(s0), a1 = std::abs(s1), c0 = 0, c1 = 0;
        for (uint32_t j = a.offptr[i]; j < a.offptr[i + 1]; ++j) {
            Real sign = std::copysign(1.0, a.offa[j]);
            Real v0 = Real(a.offa[j]) * (Real(b0[a.offcol[j]]) + sign * x0);
            Real v1 = Real(a.offa[j]) * (Real(b1[a.offcol[j]]) + sign * x1);
            a0 += std::abs(v0);
            a1 += std::abs(v1);
            Real term = v0 - c0, next = s0 + term;
            c0 = (next - s0) - term;
            s0 = next;
            term = v1 - c1;
            next = s1 + term;
            c1 = (next - s1) - term;
            s1 = next;
        }
        r0max = std::max(r0max, std::abs(s0 - lambda0 * x0) + 8 * eps * a0
                        + a.residual_error[i] * std::abs(x0) + 4 * eps * std::abs(lambda0 * x0));
        r1max = std::max(r1max, std::abs(s1 - lambda1 * x1) + 8 * eps * a1
                        + a.residual_error[i] * std::abs(x1) + 4 * eps * std::abs(lambda1 * x1));
        // Integrate the ODE defect using the infinity-norm semigroup bound.
        Real error = factor * (w0 * r0max + w1 * r1max)
                     + std::numeric_limits<double>::epsilon() * ymax
                     + 16 * eps * (std::abs(e0) * max0 + std::abs(p1) * max1);
        if (!(error <= budget)) return false;
    }
    if (std::getenv("FAMEXP_DEBUG")) std::fprintf(stderr, "certified invariant vectors\n");
    return true;
}

static bool steady_solve(const Matrix &a, double t, double rtol, double atol,
                         const double *b0, const double *b1, double *out) {
    if (a.lower <= 0 || t * a.lower < 50) return false;
    const int n = a.n;
    vector<double> r(n), z(n), p(n), ap(n);
    std::fill(out, out + n, 0);
    Real n0 = 0, n1 = 0;
    for (int i = 0; i < n; ++i) {
        n0 += Real(b0[i]) * b0[i];
        n1 += Real(b1[i]) * b1[i];
        r[i] = b1[i];
        p[i] = z[i] = r[i] / a.diagonal[i];
    }
    n0 = std::sqrt(n0);
    n1 = std::sqrt(n1);
    Real discarded = std::exp(-Real(t) * a.lower) * (n0 + n1 / a.lower);
    auto certify = [&]() {
        Real rn = 0;
        double xmax = 0;
        for (int i = 0; i < n; ++i) {
            Real sum = 0, comp = 0, magnitude = 0;
            for (uint32_t j = a.ptr[i]; j < a.ptr[i + 1]; ++j) {
                Real product = Real(a.a[j]) * Real(out[a.col[j]]);
                magnitude += std::abs(product);
                Real term = product - comp;
                Real next = sum + term;
                comp = (next - sum) - term;
                sum = next;
            }
            Real ri = Real(b1[i]) - sum;
            r[i] = double(ri);
            Real rounding = 8 * std::numeric_limits<Real>::epsilon()
                            * (magnitude + std::abs(Real(b1[i])));
            rn = std::max(rn, std::abs(ri) + rounding);
            xmax = std::max(xmax, std::abs(out[i]));
        }
        // Strict row dominance gives ||inverse(L)||_inf <= 1/lower.
        Real err = rn / a.lower + discarded;
        return err <= 0.025L * (atol + rtol * xmax);
    };
    if (n1 == 0) return discarded <= 0.01L * atol;
    Real rz = 0;
    for (int i = 0; i < n; ++i) rz += Real(r[i]) * z[i];
    int check_at = 16;
    for (int k = 0; k < 1400; ++k) {
        a.apply(p.data(), ap.data());
        Real pap = 0;
        for (int i = 0; i < n; ++i) pap += Real(p[i]) * ap[i];
        if (!(pap > 0) || !std::isfinite(pap)) return certify();
        double alpha = double(rz / pap);
        for (int i = 0; i < n; ++i) {
            out[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        if (k + 1 == check_at) {
            if (certify()) {
                if (std::getenv("FAMEXP_DEBUG"))
                    std::fprintf(stderr, "steady CG iterations=%d lower=%.8g\n", k + 1, a.lower);
                return true;
            }
            for (int i = 0; i < n; ++i) p[i] = z[i] = r[i] / a.diagonal[i];
            rz = 0;
            for (int i = 0; i < n; ++i) rz += Real(r[i]) * z[i];
            check_at += 64;
            continue;
        }
        Real next = 0;
        for (int i = 0; i < n; ++i) {
            z[i] = r[i] / a.diagonal[i];
            next += Real(r[i]) * z[i];
        }
        double beta = double(next / rz);
        for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
        rz = next;
    }
    return certify();
}

struct UnionFind {
    vector<int> p, size;
    explicit UnionFind(int n): p(n), size(n, 1) {
        std::iota(p.begin(), p.end(), 0);
    }
    int root(int i) {
        while (p[i] != i) {
            p[i] = p[p[i]];
            i = p[i];
        }
        return i;
    }
    void join(int i, int j) {
        i = root(i);
        j = root(j);
        if (i == j) return;
        if (size[i] < size[j]) std::swap(i, j);
        p[j] = i;
        size[i] += size[j];
    }
};

struct SmallBlocks {
    vector<int> starts, ids;
    vector<size_t> offsets;
    vector<Real> ef, pf;
    bool prepare(const Matrix &a, double t) {
        UnionFind uf(a.n);
        for (int i = 0; i < a.n; ++i)
            for (uint32_t j = a.ptr[i]; j < a.ptr[i + 1]; ++j)
                if (a.a[j] != 0) uf.join(i, a.col[j]);
        int maxsize = 0;
        for (int i = 0; i < a.n; ++i)
            if (uf.root(i) == i) maxsize = std::max(maxsize, uf.size[i]);
        if (maxsize > 32) return false;
        vector<int> rootid(a.n, -1), local(a.n), counts;
        int nc = 0;
        for (int i = 0; i < a.n; ++i) {
            int r = uf.root(i);
            if (rootid[r] < 0) {
                rootid[r] = nc++;
                counts.push_back(0);
            }
            ++counts[rootid[r]];
        }
        starts.resize(nc + 1, 0);
        for (int i = 0; i < nc; ++i) starts[i + 1] = starts[i] + counts[i];
        vector<int> next = starts;
        ids.resize(a.n);
        for (int i = 0; i < a.n; ++i) ids[next[rootid[uf.root(i)]]++] = i;
        offsets.resize(nc + 1, 0);
        for (int c = 0; c < nc; ++c)
            offsets[c + 1] = offsets[c] + size_t(counts[c]) * counts[c];
        ef.resize(offsets.back());
        pf.resize(offsets.back());
        using Dense = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
        for (int c = 0; c < nc; ++c) {
            int m = counts[c], start = starts[c];
            for (int j = 0; j < m; ++j) local[ids[start + j]] = j;
            Dense d = Dense::Zero(m, m);
            for (int j = 0; j < m; ++j) {
                int r = ids[start + j];
                for (uint32_t k = a.ptr[r]; k < a.ptr[r + 1]; ++k)
                    if (a.a[k] != 0) d(j, local[a.col[k]]) = a.a[k];
            }
            if (m == 1) {
                Real z = Real(t) * d(0, 0);
                ef[offsets[c]] = exp_factor(z);
                pf[offsets[c]] = phi_factor(z, t);
                continue;
            }
            Eigen::SelfAdjointEigenSolver<Dense> eig(d);
            const Dense &v = eig.eigenvectors();
            vector<Real> e(m), p(m);
            for (int j = 0; j < m; ++j) {
                Real z = Real(t) * eig.eigenvalues()[j];
                e[j] = exp_factor(z);
                p[j] = phi_factor(z, t);
            }
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < m; ++j) {
                    Real se = 0, sp = 0;
                    for (int k = 0; k < m; ++k) {
                        Real q = v(i, k) * v(j, k);
                        se += q * e[k];
                        sp += q * p[k];
                    }
                    ef[offsets[c] + i * m + j] = se;
                    pf[offsets[c] + i * m + j] = sp;
                }
            }
        }
        return true;
    }

    void solve(const double *b0, const double *b1, double *y) const {
        for (size_t c = 0; c + 1 < starts.size(); ++c) {
            int start = starts[c], m = starts[c + 1] - start;
            for (int i = 0; i < m; ++i) {
                Real s = 0;
                size_t o = offsets[c] + i * m;
                for (int j = 0; j < m; ++j) {
                    int k = ids[start + j];
                    s += ef[o + j] * Real(b0[k]) + pf[o + j] * Real(b1[k]);
                }
                y[ids[start + i]] = double(s);
            }
        }
    }
};

struct Chebyshev {
    const Matrix &a;
    double t, rtol, atol;
    Real h;
    vector<Real> e, p, etail, ptail;
    explicit Chebyshev(const Matrix &a_, double t_, double rt, double at):
        a(a_), t(t_), rtol(rt), atol(at), h(Real(t) * a.upper / 2) {
        int kmax = int(std::ceil(std::sqrt(180 * h))) + 100;
        vector<Real> bessel(kmax + 2, 0);
        bessel[kmax] = 1;
        // Miller recurrence, normalized by I_0(h) + 2 sum I_k(h) = exp(h).
        for (int k = kmax; k > 0; --k) {
            bessel[k - 1] = bessel[k + 1] + (2 * Real(k) / h) * bessel[k];
            if (std::abs(bessel[k - 1]) > 1e100L) {
                Real scale = 1 / bessel[k - 1];
                for (int j = k - 1; j <= kmax + 1; ++j) bessel[j] *= scale;
            }
        }
        Real sum = bessel[0];
        for (int k = 1; k <= kmax; ++k) sum += 2 * bessel[k];
        e.resize(kmax + 1);
        p.resize(kmax + 1);
        for (int k = 0; k <= kmax; ++k)
            e[k] = bessel[k] * (k == 0 ? 1 : 2) / sum;
        Real tail = 0, weighted = 0;
        // Division by 1-x converts the exponential coefficients to positive tails.
        for (int k = kmax; k >= 1; --k) {
            p[k] = 2 * weighted / h;
            tail += e[k];
            weighted += tail;
        }
        p[0] = weighted / h;
        Real ps = std::accumulate(p.begin(), p.end(), Real(0));
        for (Real &v: p) v /= ps;
        etail.resize(kmax + 1);
        ptail.resize(kmax + 1);
        tail = weighted = 0;
        for (int k = kmax; k >= 0; --k) {
            etail[k] = tail;
            ptail[k] = weighted;
            tail += e[k];
            weighted += p[k];
        }
    }

    int degree(Real norm0, Real norm1, Real target) const {
        for (int k = 0; k < int(e.size()); ++k)
            if (norm0 * etail[k] + Real(t) * norm1 * ptail[k] <= target)
                return k;
        return int(e.size()) - 1;
    }

    template<class T>
    void evaluate(const double *b0, const double *b1, double *out, int degree) const {
        int n = a.n;
        vector<T> v(n, 0), d(n, 0), lv(n);
        if (degree == 0) {
            for (int i = 0; i < n; ++i)
                out[i] = double(e[0] * Real(b0[i]) + Real(t) * p[0] * Real(b1[i]));
            return;
        }
        const T fac = T(-4 / Real(a.upper));
        const T half = T(-2 / Real(a.upper));
        // Reinsch's difference form of Clenshaw is stable near the zero eigenvalue.
        if (degree > 0) {
            T c0 = T(e[degree]), c1 = T(Real(t) * p[degree]);
            for (int i = 0; i < n; ++i) v[i] = d[i] = c0 * T(b0[i]) + c1 * T(b1[i]);
            for (int k = degree - 1; k >= 1; --k) {
                c0 = T(e[k]);
                c1 = T(Real(t) * p[k]);
                a.apply(v.data(), lv.data());
                for (int i = 0; i < n; ++i) {
                    d[i] += fac * lv[i] + c0 * T(b0[i]) + c1 * T(b1[i]);
                    v[i] += d[i];
                }
            }
            a.apply(v.data(), lv.data());
        }
        T c0 = T(e[0]), c1 = T(Real(t) * p[0]);
        for (int i = 0; i < n; ++i)
            out[i] = double(c0 * T(b0[i]) + c1 * T(b1[i]) + d[i] + half * lv[i]);
    }

    void solve(const double *b0, const double *b1, double *out) const {
        if (eigen_solve(a, t, rtol, atol, b0, b1, out)) return;
        if (steady_solve(a, t, rtol, atol, b0, b1, out)) return;
        Real norm0 = 0, norm1 = 0;
        double bmax = 0;
        for (int i = 0; i < a.n; ++i) {
            norm0 += Real(b0[i]) * b0[i];
            norm1 += Real(b1[i]) * b1[i];
            bmax = std::max(bmax, std::abs(b0[i]) + t * std::abs(b1[i]));
        }
        norm0 = std::sqrt(norm0);
        norm1 = std::sqrt(norm1);
        Real target = 0.015L * (atol + rtol * bmax / (1 + std::sqrt(h)));
        bool extended = rtol < 1e-9 && h > 300 && !a.dominant;
        if (extended) target = std::min(target, 0.01L * atol);
        int m = degree(norm0, norm1, target);
        if (extended) evaluate<Real>(b0, b1, out, m);
        else evaluate<double>(b0, b1, out, m);
        double ym = 0;
        for (int i = 0; i < a.n; ++i) ym = std::max(ym, std::abs(out[i]));
        if (!extended && rtol < 1e-9 && ym < bmax * 1e-6) {
            extended = true;
            m = degree(norm0, norm1, 0.01L * atol);
            evaluate<Real>(b0, b1, out, m);
            ym = 0;
            for (int i = 0; i < a.n; ++i) ym = std::max(ym, std::abs(out[i]));
        }
        Real actual = 0.025L * (atol + rtol * ym);
        if (norm0 * etail[m] + Real(t) * norm1 * ptail[m] > actual) {
            m = degree(norm0, norm1, actual * 0.5L);
            if (extended) evaluate<Real>(b0, b1, out, m);
            else evaluate<double>(b0, b1, out, m);
        }
        if (std::getenv("FAMEXP_DEBUG"))
            std::fprintf(stderr, "n=%d upper=%.8g h=%.8Lg degree=%d extended=%d stable=%d\n",
                         a.n, a.upper, h, m, int(extended), int(a.dominant));
    }
};

int main() {
    write_all("READY\n", 6);
    char magic[8];
    read_all(magic, 8);
    if (std::memcmp(magic, "FAMEXP2\n", 8)) return 4;
    uint32_t length;
    read_all(&length, 4);
    std::string header(length, '\0');
    read_all(header.data(), length);
    int n = int(field(header, "n"));
    int nnz = int(field(header, "nnz"));
    int requests = int(field(header, "requests"));
    double t = field(header, "t"), rtol = field(header, "rtol"), atol = field(header, "atol");
    Matrix a;
    a.n = n;
    vector<uint64_t> ptr(n + 1);
    read_all(ptr.data(), ptr.size() * 8);
    a.ptr.resize(n + 1);
    for (int i = 0; i <= n; ++i) a.ptr[i] = uint32_t(ptr[i]);
    a.col.resize(nnz);
    a.a.resize(nnz);
    read_all(a.col.data(), size_t(nnz) * 4);
    read_all(a.a.data(), size_t(nnz) * 8);
    vector<double> b0(n), b1(n), y(n);
    if (t == 0 || nnz == 0) {
        for (int j = 0; j < requests; ++j) {
            read_all(b0.data(), n * 8);
            read_all(b1.data(), n * 8);
            for (int i = 0; i < n; ++i) y[i] = b0[i] + t * b1[i];
            write_all(y.data(), n * 8);
        }
        return 0;
    }
    SmallBlocks blocks;
    bool small = blocks.prepare(a, t);
    if (small) {
        for (int j = 0; j < requests; ++j) {
            read_all(b0.data(), n * 8);
            read_all(b1.data(), n * 8);
            blocks.solve(b0.data(), b1.data(), y.data());
            write_all(y.data(), n * 8);
        }
        return 0;
    }
    a.bound(false);
    if (t * a.upper > 2000) a.bound(true);
    if (a.upper == 0) {
        for (int j = 0; j < requests; ++j) {
            read_all(b0.data(), n * 8);
            read_all(b1.data(), n * 8);
            for (int i = 0; i < n; ++i) y[i] = b0[i] + t * b1[i];
            write_all(y.data(), n * 8);
        }
        return 0;
    }
    if (t * a.upper >= 10) a.prepare_stable();
    Chebyshev solver(a, t, rtol, atol);
    for (int j = 0; j < requests; ++j) {
        read_all(b0.data(), n * 8);
        read_all(b1.data(), n * 8);
        solver.solve(b0.data(), b1.data(), y.data());
        write_all(y.data(), n * 8);
    }
    return 0;
}
