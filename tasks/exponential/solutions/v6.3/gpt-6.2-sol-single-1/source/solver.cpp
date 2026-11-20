#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <dlfcn.h>
#include <unistd.h>

#include "sparse_kernels.hpp"

using Real = long double;
using std::vector;

namespace {

void read_exact(void* dst, size_t n) {
    auto* p = static_cast<char*>(dst);
    while (n) {
        const ssize_t r = read(STDIN_FILENO, p, n);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) throw std::runtime_error("short input");
        p += r;
        n -= size_t(r);
    }
}

void write_exact(const void* src, size_t n) {
    auto* p = static_cast<const char*>(src);
    while (n) {
        const ssize_t r = write(STDOUT_FILENO, p, n);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) throw std::runtime_error("short write");
        p += r;
        n -= size_t(r);
    }
}

struct Header {
    int n = 0, nnz = 0, requests = 0;
    double t = 0, rtol = 0, atol = 0;
    static Header get() {
        char magic[8];
        uint32_t len;
        read_exact(magic, sizeof(magic));
        read_exact(&len, sizeof(len));
        if (std::memcmp(magic, "FAMEXP2\n", 8) != 0 || len == 0 || len > 4096)
            throw std::runtime_error("unsupported packet");
        std::string text(len, '\0');
        read_exact(text.data(), len);
        Header h;
        const char* p = text.c_str();
        auto ws = [&] { while (std::isspace(static_cast<unsigned char>(*p))) ++p; };
        auto expect = [&](char ch) {
            ws();
            if (*p != ch) throw std::runtime_error("invalid metadata");
            ++p;
        };
        auto str = [&] {
            expect('"');
            std::string key;
            while (*p && *p != '"') {
                char ch = *p++;
                if (ch == '\\') {
                    if (!*p) throw std::runtime_error("invalid metadata string");
                    ch = *p++;
                    if (ch == 'u') {
                        unsigned v = 0;
                        for (int k = 0; k < 4; ++k) {
                            if (!std::isxdigit(static_cast<unsigned char>(*p)))
                                throw std::runtime_error("invalid metadata escape");
                            const char c = *p++;
                            v = (v<<4) | unsigned(c <= '9' ? c-'0' : (c|32)-'a'+10);
                        }
                        // All field names and scenario names are ASCII.
                        ch = v <= 127 ? char(v) : '?';
                    }
                }
                key.push_back(ch);
            }
            expect('"');
            return key;
        };
        int version = 0;
        expect('{');
        for (;;) {
            const auto key = str();
            expect(':');
            ws();
            if (*p == '"') {
                str();
            } else {
                char* end = nullptr;
                const double v = std::strtod(p,&end);
                if (end == p) throw std::runtime_error("invalid numeric metadata");
                p = end;
                if (key == "version") version = int(v);
                else if (key == "n") h.n = int(v);
                else if (key == "nnz") h.nnz = int(v);
                else if (key == "requests") h.requests = int(v);
                else if (key == "t") h.t = v;
                else if (key == "rtol") h.rtol = v;
                else if (key == "atol") h.atol = v;
            }
            ws();
            if (*p == '}') break;
            expect(',');
        }
        if (version != 2 || h.n < 1 || h.n > 120000 || h.nnz < 0 || h.nnz > 1000000 ||
            (h.requests != 1 && h.requests != 16))
            throw std::runtime_error("invalid dimensions");
        return h;
    }
};

inline Real phi(Real x) {
    return x == 0 ? 1 : -std::expm1(-x) / x;
}

template<class T>
inline T dot_short(int from, int to, const int* __restrict col,
                   const double* __restrict vals, const T* __restrict x) {
    T a = 0, b = 0, c = 0, d = 0;
    int j = from;
    for (; j + 3 < to; j += 4) {
        a += T(vals[j]) * x[col[j]];
        b += T(vals[j + 1]) * x[col[j + 1]];
        c += T(vals[j + 2]) * x[col[j + 2]];
        d += T(vals[j + 3]) * x[col[j + 3]];
    }
    for (; j < to; ++j) a += T(vals[j]) * x[col[j]];
    return (a + b) + (c + d);
}

template<class T>
[[gnu::noinline]] T dot_wide(int from, int to, const int* col, const double* vals, const T* x) {
    // A tree over short chunks keeps the error of a hub row from growing
    // linearly with its degree.
    T stack[16];
    unsigned used = 0;
    for (int j = from; j < to; j += 64) {
        T s = dot_short(j,std::min(j+64,to),col,vals,x);
        unsigned k = 0;
        while (used & (1u<<k)) {
            s += stack[k];
            used ^= 1u<<k++;
        }
        stack[k] = s;
        used |= 1u<<k;
    }
    T s = 0;
    for (unsigned k = 0; used; ++k) {
        if (used & (1u<<k)) {
            s += stack[k];
            used ^= 1u<<k;
        }
    }
    return s;
}

template<class T>
inline T spdot(int from, int to, const int* __restrict col,
               const double* __restrict vals, const T* __restrict x) {
    if (to-from > 192) return dot_wide(from,to,col,vals,x);
    return dot_short(from,to,col,vals,x);
}

struct Matrix {
    const int n;
    vector<int> row, col, partner;
    vector<double> val, diag, abs_off, shift_error;
    double norm = 0, lower = 0, upper = 0, unscaled_lower = 0, mean_diag = 0;
    int widest = 0;
    bool small_blocks = true, has_pairs = false;

    explicit Matrix(const Header& h) : n(h.n), row(n+1), col(h.nnz), val(h.nnz) {
        {
            vector<uint64_t> wide(n+1);
            read_exact(wide.data(), sizeof(uint64_t) * wide.size());
            std::copy(wide.begin(), wide.end(), row.begin());
        }
        read_exact(col.data(), sizeof(uint32_t) * col.size());
        read_exact(val.data(), sizeof(double) * val.size());
        if (h.t == 0) return;
        diag.resize(n);
        abs_off.resize(n);
        partner.assign(n, -1);
        double minval = std::numeric_limits<double>::infinity();
        for (int i = 0; i < n; ++i) {
            const int s = row[i], e = row[i+1];
            double d = 0, r = 0;
            int nb = -1;
            for (int j = s; j < e; ++j) {
                const int k = col[j];
                const double a = val[j];
                if (k == i) d = a;
                else if (a != 0) {
                    r += std::abs(a);
                    if (nb >= 0) small_blocks = false;
                    nb = j;
                }
            }
            widest = std::max(widest, e-s);
            diag[i] = d;
            mean_diag += d;
            abs_off[i] = r;
            partner[i] = nb;
            has_pairs |= nb >= 0;
            const double guard = 1.0 + 2.3e-16 * (e-s+4);
            norm = std::max(norm, (std::abs(d) + r) * guard);
            minval = std::min(minval, d - guard * r);
        }
        mean_diag /= n;
        upper = norm;
        unscaled_lower = std::nextafter(minval,-std::numeric_limits<double>::infinity());
        lower = std::max(0.0, unscaled_lower);
    }

    void tighten(double t) {
        if (t * upper < 80 || n < 3) return;
        vector<double> w(n);
        const double inv = 1 / upper;
        // Positive test vector for a Gershgorin bound after diagonal similarity.
        const double floor = 0.07;
        for (int i = 0; i < n; ++i) w[i] = floor + (diag[i] + abs_off[i]) * inv;
        double u = 0, l = std::numeric_limits<double>::infinity();
        for (int i = 0; i < n; ++i) {
            double sum = 0;
            for (int j = row[i]; j < row[i+1]; ++j)
                if (col[j] != i) sum += std::abs(val[j]) * w[col[j]];
            const double r = sum / w[i] * (1.0 + 2.3e-16 * (row[i+1]-row[i]+8));
            u = std::max(u, diag[i] + r);
            l = std::min(l, diag[i] - r);
        }
        upper = std::min(upper, u * (1 + 1e-15));
        lower = std::max(lower, std::nextafter(l,-std::numeric_limits<double>::infinity()));
    }

    void shift_diagonal(double s) {
        if (s == 0) return;
        shift_error.resize(n);
        for (int i = 0; i < n; ++i) {
            for (int j = row[i]; j < row[i+1]; ++j) {
                if (col[j] == i) {
                    const double d = val[j] - s;
                    shift_error[i] = (val[j]-d)-s;
                    val[j] = d;
                    break;
                }
            }
        }
    }

    template<class T>
    inline T dot(int i, const T* x) const {
        return spdot(row[i], row[i+1], col.data(), val.data(), x);
    }
};

struct Blocks {
    struct Pair {
        int i, j;
        Real v, eh, el, gh, gl;
    };
    vector<Pair> pairs;
    vector<double> e, g;
    Blocks(const Matrix& a, double t, double rtol) {
        const int n = a.n;
        e.resize(n);
        g.resize(n);
        for (int i = 0; i < n; ++i) {
            int p = a.partner[i];
            if (p < 0) {
                const double x = a.diag[i] * t;
                e[i] = std::exp(-x);
                g[i] = x == 0 ? t : t * (-std::expm1(-x) / x);
                continue;
            }
            const int j = a.col[p];
            if (j < i) continue;
            const Real u = a.diag[i], w = a.diag[j], b = a.val[p];
            const Real d = (u - w) / 2, r = std::sqrt(d*d+b*b);
            const Real hi = (u + w) / 2 + r;
            const double bb = double(b)*double(b);
            const Real det = Real(std::fma(double(u),double(w),-bb)) -
                             std::fma(double(b),double(b),-bb);
            const Real lo = det / hi;
            const Real v = b / (std::abs(d) + r);
            const Real x = hi * t, y = lo * t, weight = 1 / (1 + v*v);
            const auto damp = [rtol](Real s) -> Real {
                if (s > 80) return 0;
                return rtol < 1e-9 ? std::exp(-s) : Real(std::exp(-double(s)));
            };
            const auto integ = [rtol](Real s) -> Real {
                if (s > 80) return 1/s;
                if (rtol < 1e-9) return phi(s);
                const double u = double(s);
                return u == 0 ? 1 : -std::expm1(-u) / u;
            };
            const Real eh = weight*damp(x), el = weight*damp(y);
            const Real gh = (weight*t)*integ(x), gl = (weight*t)*integ(y);
            pairs.push_back({d >= 0 ? i : j, d >= 0 ? j : i, v, eh, el, gh, gl});
        }
    }
    void solve(const double* b0, const double* b1, double* out) const {
        for (size_t i = 0; i < e.size(); ++i) out[i] = e[i] * b0[i] + g[i] * b1[i];
        for (const auto& p : pairs) {
            const int i = p.i, j = p.j;
            const Real u = p.eh * (b0[i] + p.v*b0[j]) + p.gh * (b1[i] + p.v*b1[j]);
            const Real w = p.el * (b0[j] - p.v*b0[i]) + p.gl * (b1[j] - p.v*b1[i]);
            out[i] = double(u - p.v*w);
            out[j] = double(w + p.v*u);
        }
    }
};

struct Magnitudes {
    double inf0 = 0, inf1 = 0, two0 = 0, two1 = 0;
    Magnitudes(int n, const double* __restrict b0, const double* __restrict b1) {
        double sq0 = 0, sq1 = 0;
        for (int i = 0; i < n; ++i) {
            inf0 = std::max(inf0, std::abs(b0[i]));
            inf1 = std::max(inf1, std::abs(b1[i]));
            sq0 += b0[i]*b0[i];
            sq1 += b1[i]*b1[i];
        }
        two0 = std::sqrt(sq0) * (1 + 1e-10);
        two1 = std::sqrt(sq1) * (1 + 1e-10);
    }
};

struct Taylor {
    const Matrix& a;
    const double t, rtol, atol, center, radius;
    vector<double> x, w;
    vector<Real> xl, wl;
    std::array<Real,32> e{}, g{};
    std::array<Real,32> exp_tail{}, phi_tail{};
    Taylor(const Matrix& mat, const Header& h) :
        a(mat), t(h.t), rtol(h.rtol), atol(h.atol),
        center((h.t/2)*(mat.norm+mat.unscaled_lower)),
        radius((1+6e-16)*(h.t/2)*(mat.norm-mat.unscaled_lower)),
        x(h.n), w(h.n) {
        const Real damp = std::exp(-Real(center));
        const Real bound = std::exp(std::max(0.,radius-center));
        Real ifact = 1, term = 1;
        for (int m = 0; m < 32; ++m) {
            e[m] = damp*ifact;
            Real s = ifact/(m+1), p = s;
            for (int j = 1; j < 28; ++j) {
                p *= Real(center)/(m+j+1);
                s += p;
                if (p < 0x1p-75L*s) break;
            }
            g[m] = t*damp*s;
            ifact /= m+1;
            term *= Real(radius)/(m+1);
            exp_tail[m] = bound*term;
            phi_tail[m] = t*exp_tail[m]/(m+2);
        }
    }

    template<class T>
    double evaluate(int m, const double* __restrict b0, const double* __restrict b1,
                    double* __restrict out, T* __restrict r, T* __restrict s) {
        const int n = a.n;
        const T em = T(e[m]), gm = T(g[m]);
        const T h = T(-t), c = T(center);
        if (m) {
            for (int i = 0; i < n; ++i) r[i] = em*b0[i]+gm*b1[i];
            for (int j = m-1; j > 0; --j) {
                const T ej = T(e[j]), gj = T(g[j]);
                for (int i = 0; i < n; ++i)
                    s[i] = h*a.dot(i,r) + (c*r[i] + (ej*b0[i]+gj*b1[i]));
                std::swap(r,s);
            }
        }
        double yn = 0;
        const T e0 = T(e[0]), g0 = T(g[0]);
        if (m) {
            for (int i = 0; i < n; ++i) {
                const double v = double(h*a.dot(i,r) + (c*r[i] + (e0*b0[i]+g0*b1[i])));
                out[i] = v;
                yn = std::max(yn,std::abs(v));
            }
        } else {
            for (int i = 0; i < n; ++i) {
                const double v = double(e0*b0[i]+g0*b1[i]);
                out[i] = v;
                yn = std::max(yn,std::abs(v));
            }
        }
        return yn;
    }

    void solve(const double* b0, const double* b1, double* out) {
        const int n = a.n;
        double bmax0 = 0, bmax1 = 0;
        for (int i = 0; i < n; ++i) {
            bmax0 = std::max(bmax0,std::abs(b0[i]));
            bmax1 = std::max(bmax1,std::abs(b1[i]));
        }
        auto bound = [&](int m) { return exp_tail[m]*bmax0+phi_tail[m]*bmax1; };
        const double amplitude = double(std::max(e[0]*bmax0,g[0]*bmax1));
        double cutoff = 0.23*(atol+rtol*amplitude);
        int m = 0;
        for (;;) {
            while (m < 31 && bound(m) > cutoff) ++m;
            const double yn = evaluate(m,b0,b1,out,x.data(),w.data());
            const double goal = atol+rtol*yn;
            if (m == 31 || bound(m) <= 0.37*goal) {
                if (rtol < 1e-9 && m && yn < 0.002*amplitude) {
                    xl.resize(n); wl.resize(n);
                    evaluate(m,b0,b1,out,xl.data(),wl.data());
                }
                return;
            }
            cutoff = 0.18*goal;
        }
    }
};

#include "equilibrium.hpp"
#include "krylov.hpp"
#include "precision.hpp"
#include "tensor.hpp"
#include "components.hpp"

struct Chebyshev {
    const Matrix& a;
    const double t, rtol, atol, lo, hi;
    const Real z, left;
    vector<Real> e, g, tail_e, tail_g;
    vector<double> x, w, d;
    vector<Real> xl, wl, dl;
    Diagonals dia;
    SlicedRows sell;
    vector<double> sb0, sb1, sy;
    bool use_dia, use_sell;
    Equilibrium eq;
    KrylovTrial krylov;
    Compensated precise;

    Chebyshev(Matrix& mat, const Header& h)
        : a(mat), t(h.t), rtol(h.rtol), atol(h.atol), lo(mat.lower), hi(mat.upper),
          z((Real(hi) - lo)*t/2), left(Real(lo)*t), d(h.n) {
        compute_coeffs();
        mat.shift_diagonal(lo);
        use_dia = z >= 2 && dia.setup(a);
#ifdef FAM_NO_SELL
        use_sell = false;
#else
        use_sell = !use_dia && z >= 12 && sell.setup(a,z >= 190);
#endif
        const int padded = use_dia ? 2*dia.pad : 0;
        const int size = use_sell ? sell.n4 : a.n + padded;
        x.resize(size);
        w.resize(size);
        if (use_sell) {
            sb0.resize(size);
            sb1.resize(size);
            sy.resize(size);
            d.resize(size);
        }
    }

    void compute_coeffs() {
        int m = 0;
        // Bound the mass of the omitted Bessel coefficients by a Chernoff tail.
        // Miller's backward ratio recurrence then needs no large exponentials.
        while (true) {
            m += 8;
            const Real r = (m + 1) / z;
            const Real s = std::sqrt(1 + r*r);
            if ((m + 1) * (std::asinh(r) - r/(1+s)) > 95) break;
        }
        vector<Real> p(m+2, 0), q(m+2, 0);
        Real ratio = 0;
        for (int j = m; j >= 1; --j) {
            ratio = 1 / (2*j/z + ratio);
            p[j] = ratio;
        }
        p[0] = 1;
        Real mass = 1;
        for (int j = 1; j <= m; ++j) {
            p[j] *= p[j-1];
            mass += 2*p[j];
        }
        for (Real& x : p) x /= mass;
        const Real ep = std::exp(-left);
        if (left == 0) {
            Real s = 0, u = 0;
            for (int j = m; j > 0; --j) {
                s += p[j];
                u += s;
                q[j-1] = (2/z)*u;
            }
        } else {
            // Fourier coefficients of (1-exp(-left-z+z*cos(theta))) /
            // (left+z-z*cos(theta)). Factor the three-term equation using rho,
            // the smaller root, and compute q[0] without cancellation.
            const Real delta = std::sqrt(left*(left+2*z));
            const Real omr = (2*left) / (delta + left);
            const Real rho = z / (left + z + delta);
            Real u = 0, s = 0;
            for (int j = 1; j <= m; ++j) {
                u = omr + rho*u;
                s += u*p[j];
            }
            q[0] = (-std::expm1(-left) + 2*ep*s) / delta;
            u = 0;
            vector<Real> h(m+2, 0);
            for (int j = m; j >= 1; --j) {
                u = p[j] + rho*u;
                h[j] = (2*rho/z)*ep*u;
            }
            for (int j = 1; j <= m; ++j)
                q[j] = std::max(Real(0), std::fma(rho, q[j-1], -h[j]));
        }
        e.resize(m+1);
        g.resize(m+1);
        tail_e.resize(m+1);
        tail_g.resize(m+1);
        Real te = 0, tg = 0, sum_g = q[0];
        for (int j = 1; j <= m; ++j) sum_g += 2*q[j];
        const Real gscale = t*phi(left)/sum_g;
        // The phi coefficients, like the exponential coefficients, are
        // nonnegative and bounded by their counterparts for left=0.
        for (int j = m; j >= 0; --j) {
            const Real f = j == 0 ? 1 : 2;
            e[j] = f*ep*p[j];
            g[j] = f*gscale*q[j];
            tail_e[j] = te;
            tail_g[j] = tg;
            te += e[j];
            tg += g[j];
        }
    }

    bool mode_action(const double* rhs, double norm, bool forcing,
                     double budget, Real& gain, Real& bound) const {
        if (norm == 0) {
            gain = bound = 0;
            return true;
        }
        const int n = a.n;
        const Real amp = forcing ? t : 1;
        const Real rho = 1.e-8L/t;
        const bool inf_contracts = a.unscaled_lower >= -rho;
        auto mv = [&](int i) {
            Real p = Real(lo)*rhs[i];
            if (lo) p += Real(a.shift_error[i])*rhs[i];
            Real u = 0, v = 0;
            // Roundoff of an accurate short dot and, for high-degree rows,
            // a compensated sum of disjoint contributions.
            for (int j = a.row[i]; j < a.row[i+1]; ++j) {
                const Real val = Real(a.val[j])*rhs[a.col[j]];
                const Real r = val-v, s = u+r;
                v = (s-u)-r;
                u = s;
            }
            return Real(t)*(p+u);
        };
        int k = 0;
        for (int j = 1; j <= 16; ++j) {
            const int i = (int64_t(j)*n)/17;
            if (std::abs(rhs[i]) > std::abs(rhs[k])) k = i;
        }
        // The trial is mainly intended for constants and other smooth or
        // highly structured rows. If the samples miss the support, decline.
        if (std::abs(rhs[k]) < 0.01*norm) return false;
        const Real theta = std::max(0.0L,mv(k)/rhs[k]);
        const Real p = phi(theta);
        if (forcing) {
            gain = t*p;
            if (theta < 0.001L)
                bound = t*0.5L;
            else
                bound = t*((1-p)/theta);
        } else {
            gain = theta > 80 ? 0 : std::exp(-theta);
            bound = p;
        }
        const Real target = budget / (bound + std::numeric_limits<Real>::min());
        for (int q = 0; q < 11; ++q) {
            const int i = (int64_t(2*q+1)*n)/22;
            if (std::abs(mv(i)-theta*rhs[i]) > target) return false;
        }
        Real r = 0;
        const Real err = 1.5e-18L * ((Real(t)*a.norm + theta)*norm);
        for (int i = 0; i < n; ++i) {
            Real u = std::abs(mv(i)-theta*rhs[i]) + err;
            if (inf_contracts) r = std::max(r,u);
            else r += u*u;
        }
        bound *= 1.000001L*(inf_contracts ? r : std::sqrt(r));
        bound += 4e-19L*amp*norm;
        return bound <= budget;
    }

    bool mode_trial(const Magnitudes& b, const double* b0, const double* b1,
                    double guess, double* out) const {
        const double goal = 0.15*(atol+rtol*guess);
        Real ef, gf, err0, err1;
        if (!mode_action(b0,b.inf0,false,goal,ef,err0) ||
            !mode_action(b1,b.inf1,true,goal,gf,err1))
            return false;
        double yn = 0;
        for (int i = 0; i < a.n; ++i) {
            out[i] = double(ef*b0[i]+gf*b1[i]);
            yn = std::max(yn,std::abs(out[i]));
        }
        return err0+err1 < 0.37*(atol+rtol*yn);
    }

    int degree(const Magnitudes& b, double threshold) const {
        int l = 0, r = int(e.size())-1;
        while (l < r) {
            int m = (l+r)/2;
            if (tail_e[m]*b.two0 + tail_g[m]*b.two1 <= threshold) r = m;
            else l = m+1;
        }
        return l;
    }

    template<class T>
    double evaluate_csr(int m, const double* __restrict b0, const double* __restrict b1,
                        double* __restrict out, T* __restrict x0, T* __restrict x1,
                        T* __restrict d0) const {
        const int n = a.n;
        const T h = T(-4 / (Real(hi) - lo));
        const T em = T(e[m]), gm = T(g[m]);
        if (m == 0) {
            double norm = 0;
            for (int i = 0; i < n; ++i) {
                const double v = double(em * b0[i] + gm * b1[i]);
                out[i] = v;
                norm = std::max(norm, std::abs(v));
            }
            return norm;
        }
        for (int i = 0; i < n; ++i) d0[i] = x0[i] = em * b0[i] + gm * b1[i];
        for (int j = m-1; j >= 1; --j) {
            const T ej = T(e[j]), gj = T(g[j]);
            for (int i = 0; i < n; ++i) {
                T s = a.dot(i, x0);
                if constexpr (std::is_same_v<T, Real>)
                    if (lo != 0) s += a.shift_error[i]*x0[i];
                const T b = d0[i] + (ej * b0[i] + gj * b1[i]);
                const T dp = h*s + b;
                d0[i] = dp;
                x1[i] = x0[i] + dp;
            }
            std::swap(x0, x1);
        }
        const T e0 = T(e[0]), g0 = T(g[0]), f = h/2;
        double norm = 0;
        for (int i = 0; i < n; ++i) {
            T s = a.dot(i, x0);
            if constexpr (std::is_same_v<T, Real>)
                if (lo != 0) s += a.shift_error[i]*x0[i];
            const double v = double(f*s + (d0[i] + (e0*b0[i] + g0*b1[i])));
            out[i] = v;
            norm = std::max(norm, std::abs(v));
        }
        return norm;
    }

    template<int K>
    double evaluate_dia(int m, const double* __restrict b0, const double* __restrict b1,
                        double* __restrict out) {
        int offsets[K];
        for (int j = 0; j < K; ++j) offsets[j] = dia.offsets[j];
        double* x0 = x.data() + dia.pad;
        double* x1 = w.data() + dia.pad;
        double* __restrict d0 = d.data();
        const double* __restrict coeff = dia.vals.data();
        const unsigned char* __restrict good = dia.good.data();
        const int n = a.n, n4 = n & ~3;
        const double h = double(-4 / (Real(hi) - lo));
        const double em = double(e[m]), gm = double(g[m]);
        for (int i = 0; i < n; ++i) d0[i] = x0[i] = em * b0[i] + gm * b1[i];
        double norm = 0;
        __m256d vmax = _mm256_setzero_pd();
        const auto signmask = _mm256_set1_pd(-0.);
        for (int j = m-1; j >= 0; --j) {
            const double ej = double(e[j]), gj = double(g[j]);
            const double f = j ? h : h/2;
            const auto vv_e = _mm256_set1_pd(ej);
            const auto vv_g = _mm256_set1_pd(gj);
            const auto vv_f = _mm256_set1_pd(f);
            auto scalar = [&](int i) {
                const double dp = f*a.dot(i, x0) + (d0[i] + (ej*b0[i] + gj*b1[i]));
                if (j) {
                    d0[i] = dp;
                    x1[i] = x0[i] + dp;
                } else {
                    out[i] = dp;
                    norm = std::max(norm, std::abs(dp));
                }
            };
            for (int i = 0; i < n4; i += 4) {
                if (!good[i/4]) {
                    for (int q = i; q < i+4; ++q) scalar(q);
                    continue;
                }
                const auto acc = dia.dot<K>(x0+i, coeff+size_t(i)*K, offsets,
                                            std::make_index_sequence<K>());
                const auto b = _mm256_fmadd_pd(vv_e, _mm256_loadu_pd(b0+i),
                                              _mm256_mul_pd(vv_g, _mm256_loadu_pd(b1+i)));
                const auto dp = _mm256_fmadd_pd(vv_f, acc, _mm256_add_pd(_mm256_loadu_pd(d0+i), b));
                if (j) {
                    _mm256_storeu_pd(d0+i, dp);
                    _mm256_storeu_pd(x1+i, _mm256_add_pd(_mm256_loadu_pd(x0+i), dp));
                } else {
                    _mm256_storeu_pd(out+i, dp);
                    vmax = _mm256_max_pd(vmax, _mm256_andnot_pd(signmask, dp));
                }
            }
            for (int i = n4; i < n; ++i) scalar(i);
            std::swap(x0, x1);
        }
        return std::max(norm, max_lanes(vmax));
    }

    template<bool Index16, bool Val32>
    double evaluate_sell(int m, const double* __restrict b0, const double* __restrict b1,
                         double* __restrict out) {
        const int n = sell.n4;
        auto* __restrict p0 = sb0.data();
        auto* __restrict p1 = sb1.data();
        auto* __restrict result = sy.data();
        double* x0 = x.data();
        double* x1 = w.data();
        double* __restrict d0 = d.data();
        const double h = double(-4 / (Real(hi) - lo));
        const double em = double(e[m]), gm = double(g[m]);
        for (int i = 0; i < n; ++i) {
            const int q = sell.order[i];
            p0[i] = q < a.n ? b0[q] : 0;
            p1[i] = q < a.n ? b1[q] : 0;
            d0[i] = x0[i] = em * p0[i] + gm * p1[i];
        }
        __m256d vmax = _mm256_setzero_pd();
        const auto mask = _mm256_set1_pd(-0.);
        for (int j = m-1; j >= 0; --j) {
            const double ej = double(e[j]), gj = double(g[j]);
            const double f = j ? h : h/2;
            const auto vv_e = _mm256_set1_pd(ej);
            const auto vv_g = _mm256_set1_pd(gj);
            const auto vv_f = _mm256_set1_pd(f);
            for (int k = 0; k < n; k += 4) {
                const auto ws = sell.groups[k/4];
                __m256d s;
                if (ws.width < 0) {
                    const auto& pb = sell.scalar_rows[ws.start];
                    const int* cc = sell.scalar_cols.data();
                    const double* vv = sell.scalar_vals.data();
                    const double r0 = spdot(pb[0], pb[1], cc, vv, x0);
                    const double r1 = spdot(pb[1], pb[2], cc, vv, x0);
                    const double r2 = spdot(pb[2], pb[3], cc, vv, x0);
                    const double r3 = spdot(pb[3], pb[4], cc, vv, x0);
                    s = _mm256_set_pd(r3,r2,r1,r0);
                } else {
                    s = sell.dot<Index16,Val32>(x0, ws);
                }
                const auto b = _mm256_fmadd_pd(vv_e, _mm256_loadu_pd(p0+k),
                                              _mm256_mul_pd(vv_g, _mm256_loadu_pd(p1+k)));
                const auto dp = _mm256_fmadd_pd(vv_f, s, _mm256_add_pd(_mm256_loadu_pd(d0+k), b));
                if (j) {
                    _mm256_storeu_pd(d0+k, dp);
                    _mm256_storeu_pd(x1+k, _mm256_add_pd(_mm256_loadu_pd(x0+k), dp));
                } else {
                    _mm256_storeu_pd(result+k, dp);
                    vmax = _mm256_max_pd(vmax, _mm256_andnot_pd(mask, dp));
                }
            }
            std::swap(x0, x1);
        }
        for (int i = 0; i < n; ++i) {
            const int q = sell.order[i];
            if (q < a.n) out[q] = result[i];
        }
        return max_lanes(vmax);
    }

    double evaluate(int m, const double* b0, const double* b1, double* out) {
        if (use_dia && m) {
            #define FAM_STENCIL(i) case i: return evaluate_dia<i>(m,b0,b1,out);
            switch (dia.width) {
                FAM_STENCIL(2) FAM_STENCIL(3) FAM_STENCIL(4) FAM_STENCIL(5) FAM_STENCIL(6)
                FAM_STENCIL(7) FAM_STENCIL(8) FAM_STENCIL(9) FAM_STENCIL(10)
                FAM_STENCIL(11) FAM_STENCIL(12)
            }
            #undef FAM_STENCIL
        }
        if (use_sell && m) {
            if (sell.small_indices) {
                if (sell.exact_float) return evaluate_sell<true,true>(m,b0,b1,out);
                return evaluate_sell<true,false>(m,b0,b1,out);
            }
            if (sell.exact_float) return evaluate_sell<false,true>(m,b0,b1,out);
            return evaluate_sell<false,false>(m,b0,b1,out);
        }
        return evaluate_csr(m,b0,b1,out,x.data(),w.data(),d.data());
    }

    void solve(const double* b0, const double* b1, double* out) {
        const Magnitudes b(a.n, b0, b1);
        const double damp = double(std::exp(-left));
        const double pe = double(phi(left));
        const double bsize = std::max(damp*b.inf0, t*pe*b.inf1);
        double conjectured = bsize / std::sqrt(double(1+z)*0.25 + 1);
        const double typical = conjectured;
        double lower_output = 0;
        if (z > 150 && t*b.two1 > 8*b.two0 && b.two1 > 1e-40) {
            // y = phi1(-tL) (t*b1 + h(tL)b0), where h(x)=x/(exp(x)-1).
            // Test against b1, using h's norm bound of one and the action
            // of tL on b1, to bound the output magnitude from below.
            double v = 0;
            for (int i = 0; i < a.n; ++i) {
                const double s = t*(a.dot(i,b1) + lo*b1[i]);
                v += s*s;
            }
            const double lb = (t*b.two1-b.two0) /
                              (std::sqrt(double(a.n))*(1.00000001*std::sqrt(v)/b.two1+1.));
            lower_output = 0.9999*lb;
            conjectured = std::min(conjectured,1.75*lb);
        } else if (z > 150 && b.inf1 == 0) {
            conjectured = std::min(conjectured,0.002*b.inf0);
        }
        int m = degree(b, 0.18*(atol + rtol*conjectured));
        if (m >= 28 && mode_trial(b,b0,b1,conjectured,out)) return;
#ifndef FAM_NO_EQUILIBRIUM
        if (m >= 42 && left > -std::log(rtol)+2 &&
            eq.solve(a, b, t, rtol, atol, std::min(200,m/2), b0, b1, out))
            return;
#endif
#ifndef FAM_NO_KRYLOV
        // A bounded trial is useful when the diagonal has a few isolated
        // scales. Do not spend a large fraction of the session in a failed
        // projection on the broad-spectrum default path.
        if (m >= 52 && !use_dia && (b.inf0 == 0 || b.inf1 == 0) &&
            krylov.solve(*this, b, conjectured, std::min(48, 2+m/14), b0, b1, out))
            return;
        if (m >= 70 && rtol >= 1e-7 && !use_dia && hi > 18*a.mean_diag &&
            left < 2 && b.inf0 > 0 && b.inf1 > 0 &&
            krylov.increment(*this,b,std::max(conjectured,0.35*typical),
                             std::min(120,4+m/8),b0,b1,out))
            return;
#endif
        const double expected = 3e-16 * (damp*b.inf0 + t*pe*b.inf1)
                              * double(5 + z + left);
        const double strict_goal = atol + rtol*lower_output;
        const bool start_precise =
#ifdef FAM_DOUBLE_ONLY
            false &&
#endif
            rtol < 1e-9 && expected > 6*(atol+rtol*typical) && m > 4;
        if (start_precise) {
            m = degree(b,0.18*strict_goal);
            evaluate_precise(m,b,b0,b1,out,strict_goal);
#ifdef FAM_PROFILE
            std::fprintf(stderr,"precise first %d ORDER %d\n",m,sell.reordered);
#endif
            return;
        }
        double yn = evaluate(m, b0, b1, out);
        const double goal = atol + rtol*yn;
        const bool retruncate = tail_e[m]*b.two0 + tail_g[m]*b.two1 > 0.36*goal;
        if (retruncate) m = degree(b, 0.16*goal);
        const bool rate_sensitive =
            (rtol < 1e-9 && (expected > 0.45*goal ||
                            (z > 3 && yn < 0.002*bsize))) ||
            (rtol <= 1e-7 && expected > 1.8*goal && yn < 0.002*bsize);
        if (
#ifdef FAM_DOUBLE_ONLY
            false &&
#endif
            rate_sensitive) {
            evaluate_precise(m,b,b0,b1,out,goal);
        } else if (retruncate) {
            evaluate(m, b0, b1, out);
        }
#ifdef FAM_PROFILE
        std::fprintf(stderr, "eta %.4g [%.4g, %.4g] degree %d y %.4g round %.3g adjust %d DIA %d SELL %d order %d\n",
                     t*a.norm, double(left), t*hi, m, yn, expected/goal, retruncate,
                     use_dia ? dia.width : 0, use_sell ? 1+sell.exact_float : 0,
                     sell.reordered);
#endif
    }

    void evaluate_precise(int m, const Magnitudes& b, const double* b0, const double* b1,
                          double* out, double goal) {
#ifndef FAM_NO_COMPENSATED
        int split = m;
        const Real gamma = 7.e-16L *
                (20 + (4*Real(a.norm)/(Real(hi)-lo))*(9+std::min(53,(a.widest+3)/4)));
        Real mass = 0, moment = 0, sum = 0;
        for (int k = m; k >= 1; --k) {
            mass += e[k]*b.two0+g[k]*b.two1;
            moment += mass;
            sum += moment;
            if (gamma*sum > 0.06*goal) break;
            split = k-1;
        }
        if (precise.solve(*this,m,split,b0,b1,out)) {
#ifdef FAM_PROFILE
            std::fprintf(stderr,"compensated %d/%d\n",split,m);
#endif
            return;
        }
#endif
        xl.resize(a.n);
        wl.resize(a.n);
        dl.resize(a.n);
        evaluate_csr(m,b0,b1,out,xl.data(),wl.data(),dl.data());
    }
};

} // namespace

int main() {
    try {
        const TensorFFT fft;
        write_exact("READY\n", 6);
        const Header h = Header::get();
        Matrix a(h);
        vector<double> b0(h.n), b1(h.n), y(h.n);
        auto requests = [&](auto&& compute) {
            for (int i = 0; i < h.requests; ++i) {
                read_exact(b0.data(), h.n * sizeof(double));
                read_exact(b1.data(), h.n * sizeof(double));
                const double* result = compute();
                write_exact(result, h.n * sizeof(double));
            }
        };
        if (h.t == 0) {
            requests([&] { return b0.data(); });
        } else if (a.norm == 0.) {
            requests([&] {
                for (int i = 0; i < h.n; ++i) y[i] = b0[i]+h.t*b1[i];
                return y.data();
            });
        } else if (h.t * a.norm <= 1.35 &&
                   (!a.small_blocks || (a.has_pairs && h.n >= 256))) {
            Taylor s(a, h);
            requests([&] { s.solve(b0.data(), b1.data(), y.data()); return y.data(); });
        } else if (a.small_blocks) {
            const Blocks s(a, h.t, h.rtol);
            requests([&] { s.solve(b0.data(), b1.data(), y.data()); return y.data(); });
        } else {
            TensorGrid grid(fft);
            TinyComponents blocks;
            const bool tiny = blocks.setup(a,h);
#ifndef FAM_NO_TENSOR
            const bool tensor = !tiny && grid.setup(a,h);
#else
            const bool tensor = false;
#endif
            std::unique_ptr<Chebyshev> s;
            requests([&] {
                if (!(tiny && blocks.solve(b0.data(),b1.data(),y.data())) &&
                    !(tensor && grid.solve(b0.data(),b1.data(),y.data()))) {
                    if (!s) {
                        a.tighten(h.t);
                        if (h.t*a.lower < 1e-6) a.lower = 0;
                        s.reset(new Chebyshev(a,h));
                    }
                    s->solve(b0.data(),b1.data(),y.data());
                }
                return y.data();
            });
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "famexp: %s\n", e.what());
        return 1;
    }
}
