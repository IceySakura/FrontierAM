#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

using std::vector;

static void read_all(void *p, size_t n) {
    char *s = static_cast<char *>(p);
    while (n) {
        ssize_t k = ::read(0, s, n);
        if (k <= 0) throw std::runtime_error("incomplete input");
        s += k;
        n -= size_t(k);
    }
}

static void write_all(const void *p, size_t n) {
    const char *s = static_cast<const char *>(p);
    while (n) {
        ssize_t k = ::write(1, s, n);
        if (k <= 0) throw std::runtime_error("output failure");
        s += k;
        n -= size_t(k);
    }
}

class Header {
    std::map<std::string, std::string> fields;
    std::string text;
    size_t pos = 0;
    void space() {
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
    }
    void token(char c) {
        space();
        if (pos >= text.size() || text[pos++] != c) throw std::runtime_error("invalid JSON");
    }
    std::string quoted() {
        token('"');
        std::string s;
        while (pos < text.size() && text[pos] != '"') {
            char c = text[pos++];
            if (c == '\\') {
                if (pos == text.size()) throw std::runtime_error("invalid escape");
                c = text[pos++];
            }
            s += c;
        }
        token('"');
        return s;
    }
public:
    explicit Header(std::string input) : text(std::move(input)) {
        token('{');
        space();
        while (pos < text.size() && text[pos] != '}') {
            std::string key = quoted();
            token(':');
            space();
            std::string value;
            if (text[pos] == '"') value = quoted();
            else {
                size_t start = pos;
                while (pos < text.size() && text[pos] != ',' && text[pos] != '}') ++pos;
                value = text.substr(start, pos - start);
            }
            fields.emplace(std::move(key), std::move(value));
            space();
            if (text[pos] == '}') break;
            token(',');
        }
        token('}');
    }
    double number(const char *key) const { return std::stod(fields.at(key)); }
    int integer(const char *key) const { return std::stoi(fields.at(key)); }
};

struct Matrix {
    int n;
    vector<uint32_t> ptr, col;
    vector<double> val, diag;
    double upper = 0, diagonal_cv = 0, lower = 0;
    bool positive_offdiagonal = false;
    bool pair_blocks = true;
    vector<int> partner;

    explicit Matrix(int size, int nnz) : n(size), ptr(n + 1), col(nnz), val(nnz),
                                        diag(n, 0), partner(n, -1) {
        vector<uint64_t> wide(n + 1);
        read_all(wide.data(), wide.size() * 8);
        for (int i = 0; i <= n; ++i) ptr[i] = uint32_t(wide[i]);
        read_all(col.data(), col.size() * 4);
        read_all(val.data(), val.size() * 8);
        long double sum_diag = 0, square_diag = 0, min_lower = 1e300L;
        for (int i = 0; i < n; ++i) {
            long double sum = 0;
            for (uint32_t j = ptr[i]; j < ptr[i+1]; ++j) {
                sum += std::abs((long double)val[j]);
                if (col[j] == uint32_t(i)) diag[i] = val[j];
                else if (val[j] != 0) {
                    if (partner[i] >= 0) pair_blocks = false;
                    partner[i] = int(col[j]);
                    if (val[j] > 0) positive_offdiagonal = true;
                }
            }
            upper = std::max(upper, double(sum));
            sum_diag += diag[i];
            square_diag += (long double)diag[i]*diag[i];
            min_lower = std::min(min_lower, 2*(long double)diag[i]-sum);
        }
        lower = std::max(0., double(min_lower));
        if (sum_diag > 0)
            diagonal_cv = std::sqrt(std::max(0.L, square_diag*n/(sum_diag*sum_diag)-1));
        upper *= 1 + 8 * std::numeric_limits<double>::epsilon();
    }

    void multiply(const double *__restrict x, double *__restrict y) const {
        for (int i = 0; i < n; ++i) {
            if (ptr[i+1]-ptr[i] > 32) {
                long double sum = 0;
                for (uint32_t j = ptr[i]; j < ptr[i+1]; ++j)
                    sum += (long double)val[j]*x[col[j]];
                y[i] = double(sum);
                continue;
            }
            double sum = 0;
            for (uint32_t j = ptr[i]; j < ptr[i+1]; ++j)
                sum += val[j] * x[col[j]];
            y[i] = sum;
        }
    }

    template<int Mode>
    inline double row_dot(int i, const double *x) const {
        uint32_t j = ptr[i], end = ptr[i+1];
        if constexpr (Mode != 0) {
            double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (; j+3 < end; j += 4) {
                s0 += val[j]*x[col[j]];
                s1 += val[j+1]*x[col[j+1]];
                s2 += val[j+2]*x[col[j+2]];
                s3 += val[j+3]*x[col[j+3]];
            }
            double sum = (s0+s1)+(s2+s3);
            for (; j < end; ++j) sum += val[j]*x[col[j]];
            return sum;
        } else {
            double sum = 0;
            for (; j < end; ++j) sum += val[j]*x[col[j]];
            return sum;
        }
    }

    void tighten_bound() {
        vector<double> x(n, 1), y(n);
        for (int it = 0; it < 8; ++it) {
            double old_upper = upper;
            double bound = 0, scale = 0;
            for (int i = 0; i < n; ++i) {
                long double sum = 0;
                for (uint32_t j = ptr[i]; j < ptr[i+1]; ++j)
                    sum += std::abs((long double)val[j]) * x[col[j]];
                y[i] = double(sum);
                if (x[i] > 0) bound = std::max(bound, double(sum / x[i]));
                scale = std::max(scale, double(sum));
            }
            upper = std::min(upper, bound * (1 + 1e-13));
            if (it >= 1 && upper > 0.99999*old_upper) return;
            if (scale == 0) return;
            for (int i = 0; i < n; ++i) x[i] = std::max(y[i] / scale, 1e-150);
        }
    }
};

static long double phi1(long double x) {
    if (x == 0) return 1;
    return -std::expm1(-x) / x;
}

struct BlockFunctions {
    vector<long double> e0, e1, h0, h1;
    explicit BlockFunctions(const Matrix &a, double t) :
        e0(a.n), e1(a.n, 0), h0(a.n), h1(a.n, 0) {
        for (int i = 0; i < a.n; ++i) {
            int j = a.partner[i];
            if (j < 0) {
                long double z = (long double)t * a.diag[i];
                e0[i] = std::exp(-z);
                h0[i] = (long double)t * phi1(z);
            } else if (j > i) {
                long double aa = a.diag[i], dd = a.diag[j], b = 0;
                for (uint32_t k = a.ptr[i]; k < a.ptr[i+1]; ++k)
                    if (a.col[k] == uint32_t(j)) b = a.val[k];
                long double mid = (aa + dd) / 2;
                long double delta = (aa - dd) / 2;
                long double rad = std::hypot(delta, b);
                long double hi = mid + rad;
                long double lo = hi != 0 ? (aa * dd - b * b) / hi : 0;
                long double eh = std::exp(-t * hi), el = std::exp(-t * lo);
                long double hh = t * phi1(t * hi), hl = t * phi1(t * lo);
                long double c = rad != 0 ? delta / rad : 0;
                long double s = rad != 0 ? b / rad : 0;
                e0[i] = ((1+c)*eh + (1-c)*el) / 2;
                e0[j] = ((1-c)*eh + (1+c)*el) / 2;
                e1[i] = e1[j] = s * (eh - el) / 2;
                h0[i] = ((1+c)*hh + (1-c)*hl) / 2;
                h0[j] = ((1-c)*hh + (1+c)*hl) / 2;
                h1[i] = h1[j] = s * (hh - hl) / 2;
            }
        }
    }
    void apply(const Matrix &a, const vector<double> &b0, const vector<double> &b1,
               vector<double> &y) const {
        for (int i = 0; i < a.n; ++i) {
            long double value = e0[i]*b0[i] + h0[i]*b1[i];
            int j = a.partner[i];
            if (j >= 0) value += e1[i]*b0[j] + h1[i]*b1[j];
            y[i] = double(value);
        }
    }
};

static long double phi2(long double x) {
    if (std::abs(x) < 0.01L) {
        long double sum = 0.5L, term = sum;
        for (int j = 1; j <= 12; ++j) {
            term *= -x/(j+2);
            sum += term;
        }
        return sum;
    }
    return (x+std::expm1(-x))/(x*x);
}

static bool eigenmode_action(const Matrix &a, double t, double rtol, double atol,
                             const vector<double> &b0, const vector<double> &b1,
                             vector<double> &y) {
    vector<long double> work(a.n);
    long double coefficients[2], error = 0;
    for (int k = 0; k < 2; ++k) {
        const vector<double> &b = k ? b1 : b0;
        long double norm = 0, rayleigh = 0;
        for (double x : b) norm += (long double)x*x;
        if (norm == 0) {
            coefficients[k] = 0;
            continue;
        }
        for (int i = 0; i < a.n; ++i) {
            long double sum = 0;
            for (uint32_t j = a.ptr[i]; j < a.ptr[i+1]; ++j)
                sum += (long double)a.val[j]*b[a.col[j]];
            work[i] = sum;
            rayleigh += sum*b[i];
        }
        long double lambda = std::max(0.L, rayleigh/norm), residual2 = 0;
        for (int i = 0; i < a.n; ++i) {
            long double r = work[i]-lambda*b[i];
            residual2 += r*r;
        }
        long double x = t*lambda;
        coefficients[k] = k ? t*phi1(x) : std::exp(-x);
        error += std::sqrt(residual2)*(k ? (long double)t*t*phi2(x) : t*phi1(x));
    }
    double maxy = 0;
    for (int i = 0; i < a.n; ++i) {
        y[i] = coefficients[0]*b0[i]+coefficients[1]*b1[i];
        maxy = std::max(maxy, std::abs(y[i]));
    }
    return error <= 0.02L*(atol+rtol*maxy);
}

struct Chebyshev {
    const Matrix &a;
    double t, rtol, atol, factor;
    vector<double> ce, cp;
    vector<long double> cle, clp;
    vector<long double> tail_e, tail_p;
    vector<double> u, v, w;
    int mode;
    bool diagnostics;

    Chebyshev(const Matrix &mat, double time, double relative, double absolute) :
        a(mat), t(time), rtol(relative), atol(absolute), factor(2 / mat.upper),
        u(a.n), v(a.n), w(a.n),
        mode(a.val.size() >= size_t(16)*a.n ? 3 : 2), diagnostics(false) {
        diagnostics = std::getenv("PROP_DIAGNOSTICS") != nullptr;
        long double z = (long double)t * a.upper / 2;
        int m = std::max(64, int(std::ceil(std::sqrt(128*z) + 64)));
        vector<long double> ratios(m+2, 0), coeff(m+2, 0), ph(m+2, 0);
        for (int k = m; k >= 1; --k) ratios[k-1] = z / (2*k + z*ratios[k]);
        coeff[0] = 1;
        long double total = 1;
        for (int k = 1; k <= m; ++k) {
            coeff[k] = coeff[k-1] * ratios[k-1];
            total += 2*coeff[k];
        }
        coeff[0] /= total;
        for (int k = 1; k <= m; ++k) coeff[k] *= 2/total;
        // (1-x) phi1(-z(1-x)) = (1-exp(-z(1-x)))/z.
        // Backward double summation gives its nonnegative Chebyshev coefficients.
        long double s1 = 0, s2 = 0;
        for (int k = m; k >= 0; --k) {
            ph[k] = (k ? 2 : 1) * s2 / z;
            s1 += coeff[k];
            s2 += s1;
        }
        ce.resize(m+1);
        cp.resize(m+1);
        cle.resize(m+1);
        clp.resize(m+1);
        tail_e.assign(m+1, 0);
        tail_p.assign(m+1, 0);
        long double te = 0, tp = 0;
        for (int k = m; k >= 0; --k) {
            ce[k] = double(coeff[k]);
            cp[k] = double(t * ph[k]);
            cle[k] = coeff[k];
            clp[k] = t*ph[k];
            tail_e[k] = te;
            tail_p[k] = tp;
            te += coeff[k];
            tp += t*ph[k];
        }
    }

    void evaluate_extended(int m, const vector<double> &b0, const vector<double> &b1,
                           vector<double> &y) {
        vector<long double> x(a.n, 0), difference(a.n, 0), next(a.n);
        long double f = 4.L/a.upper;
        for (int k = m; k >= 1; --k) {
            long double e = cle[k], p = clp[k];
            for (int i = 0; i < a.n; ++i) {
                long double sum = 0;
                for (uint32_t j = a.ptr[i]; j < a.ptr[i+1]; ++j)
                    sum += (long double)a.val[j]*x[a.col[j]];
                difference[i] += -f*sum+e*b0[i]+p*b1[i];
                next[i] = x[i]+difference[i];
            }
            x.swap(next);
        }
        for (int i = 0; i < a.n; ++i) {
            long double sum = 0;
            for (uint32_t j = a.ptr[i]; j < a.ptr[i+1]; ++j)
                sum += (long double)a.val[j]*x[a.col[j]];
            y[i] = difference[i]-(f/2)*sum+cle[0]*b0[i]+clp[0]*b1[i];
        }
    }

    template<int DotMode>
    void fused_evaluate(int m, const vector<double> &b0, const vector<double> &b1,
                        vector<double> &y) {
        // Reinsch differences avoid subtracting two large Clenshaw vectors
        // when evaluating modes close to the zero eigenvalue of L.
        for (int k = m; k >= 1; --k) {
            const double *x = u.data();
            double e = ce[k], p = cp[k], f = 2*factor;
            for (int i = 0; i < a.n; ++i) {
                double sum = a.row_dot<DotMode>(i, x);
                v[i] += -f*sum + e*b0[i] + p*b1[i];
                w[i] = x[i] + v[i];
            }
            u.swap(w);
        }
        for (int i = 0; i < a.n; ++i) {
            double sum = a.row_dot<DotMode>(i, u.data());
            y[i] = v[i] - factor*sum + ce[0]*b0[i] + cp[0]*b1[i];
        }
    }

    void evaluate(int m, const vector<double> &b0, const vector<double> &b1,
                  vector<double> &y) {
        std::fill(u.begin(), u.end(), 0);
        std::fill(v.begin(), v.end(), 0);
        if (mode == 3) fused_evaluate<1>(m, b0, b1, y);
        else fused_evaluate<0>(m, b0, b1, y);
    }

    void apply(const vector<double> &b0, const vector<double> &b1, vector<double> &y,
               bool force_extended = false) {
        double norm0 = 0, norm1 = 0, max0 = 0, max1 = 0, maxsum = 0;
        for (int i = 0; i < a.n; ++i) {
            norm0 += b0[i]*b0[i];
            norm1 += b1[i]*b1[i];
            max0 = std::max(max0, std::abs(b0[i]));
            max1 = std::max(max1, std::abs(b1[i]));
            maxsum = std::max(maxsum, std::abs(std::fma(t, b1[i], b0[i])));
        }
        norm0 = std::sqrt(norm0);
        norm1 = std::sqrt(norm1);
        if (norm0 == 0 && norm1 == 0) {
            std::fill(y.begin(), y.end(), 0);
            return;
        }
        double estimate = (max0 + t*max1) / std::sqrt(1+t*a.upper);
        bool extended = rtol < 1e-6 &&
            (force_extended || maxsum < 0.1*(max0+t*max1));
        long double budget = 0.02L * (atol + (extended ? 0 : rtol*estimate));
        int m = 0;
        while (m+1 < int(ce.size()) && tail_e[m]*norm0+tail_p[m]*norm1 > budget) ++m;
        for (int attempt = 0; attempt < 5; ++attempt) {
            if (extended) evaluate_extended(m, b0, b1, y);
            else evaluate(m, b0, b1, y);
            double maxy = 0;
            for (double q : y) maxy = std::max(maxy, std::abs(q));
            budget = 0.10L * (atol + rtol*maxy);
            if (!extended && rtol < 1e-6 && maxy < 1e-5*max0 && t*max1 > 0.05*max0) {
                extended = true;
                budget = 0.02L*atol;
                while (m+1 < int(ce.size()) && tail_e[m]*norm0+tail_p[m]*norm1 > budget) ++m;
                continue;
            }
            long double error = tail_e[m]*norm0 + tail_p[m]*norm1;
            if (diagnostics)
                std::fprintf(stderr, "cheb degree=%d bound=%.5g tail=%.4Lg budget=%.4Lg maxy=%.5g\n",
                             m, t*a.upper, error, budget, maxy);
            if (error <= budget) return;
            while (m+1 < int(ce.size()) && tail_e[m]*norm0+tail_p[m]*norm1 > budget/2) ++m;
        }
    }
};

struct Lanczos {
    const Matrix &a;
    const Chebyshev &series;
    int capacity;
    std::unique_ptr<double[]> basis;
    vector<double> w, alpha, beta, result0, result1;
    bool diagnostics;
    bool cancellation = false;

    explicit Lanczos(const Matrix &mat, int m, const Chebyshev &coefficients) :
        a(mat), series(coefficients), capacity(m),
        basis(new double[size_t(mat.n)*m]), w(mat.n),
        alpha(m), beta(m), result0(mat.n), result1(mat.n),
        diagnostics(std::getenv("PROP_DIAGNOSTICS") != nullptr) {}

    long double residual_bound(int m, double t, double norm, bool forcing) const {
        long double numerator = std::log((long double)norm*beta[m-1]);
        for (int j = 0; j+1 < m; ++j) numerator += std::log((long double)beta[j]);
        auto evaluate = [&](long double tau, long double &derivative) {
            long double pivot = 0, dpivot = 0, logdet = 0, trace = 0;
            for (int j = 0; j < m; ++j) {
                long double p = (long double)alpha[j] + tau/t;
                long double dp = 1;
                if (j) {
                    long double ratio = (long double)beta[j-1]/pivot;
                    p -= ratio*beta[j-1];
                    dp += ratio*ratio*dpivot;
                }
                if (p <= 0) {
                    derivative = -1e100L;
                    return 1e100L;
                }
                pivot = p;
                dpivot = dp;
                logdet += std::log(p);
                trace += dp/p;
            }
            long double factor;
            if (forcing) {
                factor = std::log((long double)t) + (tau > 1 ? tau-1-std::log(tau) : 0);
                derivative = (tau > 1 ? 1-1/tau : 0) - trace/t;
            } else {
                factor = tau;
                derivative = 1-trace/t;
            }
            return numerator-logdet+factor;
        };
        long double derivative, best = evaluate(0, derivative);
        if (derivative >= 0) return std::exp(best);
        long double lo = 0, hi = 64;
        best = std::min(best, evaluate(hi, derivative));
        while (derivative < 0 && hi < 1024) {
            hi *= 2;
            best = std::min(best, evaluate(hi, derivative));
        }
        for (int k = 0; k < 14; ++k) {
            long double mid = (lo+hi)/2;
            best = std::min(best, evaluate(mid, derivative));
            if (derivative < 0) lo = mid;
            else hi = mid;
        }
        return std::exp(best);
    }

    void project(int m, bool forcing, vector<double> &out, double norm) {
        const vector<long double> &c = forcing ? series.clp : series.cle;
        vector<long double> x(m, 0), difference(m, 0), next(m);
        long double f = 4.L/a.upper;
        for (int k = int(c.size())-1; k >= 1; --k) {
            for (int j = 0; j < m; ++j) {
                long double sum = (long double)alpha[j]*x[j];
                if (j) sum += (long double)beta[j-1]*x[j-1];
                if (j+1 < m) sum += (long double)beta[j]*x[j+1];
                difference[j] -= f*sum;
                if (j == 0) difference[j] += c[k];
                next[j] = x[j]+difference[j];
            }
            x.swap(next);
        }
        std::fill(out.begin(), out.end(), 0);
        for (int j = 0; j < m; ++j) {
            long double sum = (long double)alpha[j]*x[j];
            if (j) sum += (long double)beta[j-1]*x[j-1];
            if (j+1 < m) sum += (long double)beta[j]*x[j+1];
            long double value = difference[j]-(f/2)*sum+(j == 0 ? c[0] : 0);
            double weight = double(value*norm);
            const double *q = basis.get()+size_t(j)*a.n;
            for (int i = 0; i < a.n; ++i) out[i] += weight*q[i];
        }
    }

    bool action(const vector<double> &b, double t, bool forcing, double rtol,
                double atol, vector<double> &out, long double &final_error) {
        long double norm2 = 0;
        double maxb = 0;
        for (double x : b) {
            norm2 += (long double)x*x;
            maxb = std::max(maxb, std::abs(x));
        }
        if (norm2 == 0) {
            std::fill(out.begin(), out.end(), 0);
            final_error = 0;
            return true;
        }
        double norm = std::sqrt(norm2);
        double tolerance = 0.015*(atol+rtol*maxb*(forcing ? t : 1)/std::sqrt(1+t*a.upper));
        for (int i = 0; i < a.n; ++i) basis[i] = b[i]/norm;
        long double pivot = 0, corner = 0, error = 1e300L;
        long double previous_log = 1e300L;
        for (int j = 0; j < capacity; ++j) {
            double *q = basis.get()+size_t(j)*a.n;
            a.multiply(q, w.data());
            if (j) {
                const double *previous = q-a.n;
                double bb = beta[j-1];
                for (int i = 0; i < a.n; ++i) w[i] -= bb*previous[i];
            }
            long double aa = 0;
            for (int i = 0; i < a.n; ++i) aa += (long double)q[i]*w[i];
            alpha[j] = double(aa);
            for (int i = 0; i < a.n; ++i) w[i] -= alpha[j]*q[i];
            // A second local projection limits loss of the three-term relation.
            long double correction = 0;
            for (int i = 0; i < a.n; ++i) correction += (long double)q[i]*w[i];
            alpha[j] += double(correction);
            for (int i = 0; i < a.n; ++i) w[i] -= double(correction)*q[i];
            long double bb2 = 0;
            for (double x : w) bb2 += (long double)x*x;
            beta[j] = double(std::sqrt(bb2));
            error = (long double)norm*beta[j]*t*(forcing ? t/2 : 1);
            long double new_pivot = alpha[j];
            if (j && pivot > 0) new_pivot -= (long double)beta[j-1]*beta[j-1]/pivot;
            if (new_pivot > 0 && (!j || pivot > 0)) {
                corner = j ? corner*beta[j-1]/new_pivot : 1/new_pivot;
                pivot = new_pivot;
                // For the sign-similar M-matrix T, integral exp(-s T) ds <= T^-1
                // entrywise. Contractivity of exp(-s L) turns this into a
                // Euclidean error bound for the full Lanczos approximation.
                error = std::min(error, (long double)norm*beta[j]*corner*(forcing ? t : 1));
            } else {
                pivot = -1;
            }
            if ((j+1)%16 == 0) {
                error = std::min(error, residual_bound(j+1, t, norm, forcing));
                long double log_error = std::log(error);
                if (j+1 >= 64) {
                    long double improvement = previous_log-log_error;
                    long double predicted = log_error-improvement*(capacity-j-1)/16;
                    if (predicted > std::log(tolerance)+12) {
                        if (diagnostics)
                            std::fprintf(stderr, "lanczos early fallback m=%d bound=%.4Lg forecast=%.3Lg goal=%.3g\n",
                                         j+1, error, predicted, std::log(tolerance));
                        return false;
                    }
                }
                previous_log = log_error;
            }
            if (error < tolerance) {
                project(j+1, forcing, out, norm);
                double maxout = 0;
                for (double x : out) maxout = std::max(maxout, std::abs(x));
                tolerance = 0.015*(atol+rtol*maxout);
                if (error < tolerance) {
                    if (diagnostics) std::fprintf(stderr, "lanczos m=%d forcing=%d bound=%.4Lg\n",
                                                  j+1, int(forcing), error);
                    final_error = error;
                    return true;
                }
            }
            if (j+1 == capacity || beta[j] == 0) break;
            double *next = q+a.n;
            for (int i = 0; i < a.n; ++i) next[i] = w[i]/beta[j];
        }
        if (diagnostics) std::fprintf(stderr, "lanczos fallback m=%d forcing=%d bound=%.4Lg\n",
                                      capacity, int(forcing), error);
        return false;
    }

    bool apply(const vector<double> &b0, const vector<double> &b1, double t,
               double rtol, double atol, vector<double> &out) {
        cancellation = false;
        long double error0, error1;
        if (!action(b0, t, false, rtol, atol, result0, error0)) return false;
        if (!action(b1, t, true, rtol, atol, result1, error1)) return false;
        double maxy = 0, max0 = 0, max1 = 0;
        for (int i = 0; i < a.n; ++i) {
            out[i] = result0[i]+result1[i];
            maxy = std::max(maxy, std::abs(out[i]));
            max0 = std::max(max0, std::abs(result0[i]));
            max1 = std::max(max1, std::abs(result1[i]));
        }
        if (maxy+atol/rtol < 0.25*(max0+max1)) {
            cancellation = true;
            return false;
        }
        return error0+error1 < 0.04L*(atol+rtol*maxy);
    }
};

int main() {
    try {
        write_all("READY\n", 6);
        char magic[8];
        read_all(magic, 8);
        if (std::memcmp(magic, "FAMEXP2\n", 8)) throw std::runtime_error("invalid magic");
        uint32_t length;
        read_all(&length, 4);
        if (length > 4096) throw std::runtime_error("invalid header length");
        std::string json(length, ' ');
        read_all(json.data(), length);
        Header h(std::move(json));
        int n = h.integer("n"), nnz = h.integer("nnz"), requests = h.integer("requests");
        double t = h.number("t"), rtol = h.number("rtol"), atol = h.number("atol");
        Matrix a(n, nnz);
        vector<double> b0(n), b1(n), y(n);
        long double tiny_error = (long double)t*a.upper*std::sqrt((long double)n)*(1+t/2);
        if (t == 0 || a.upper == 0 || tiny_error < 0.01L*atol) {
            for (int k = 0; k < requests; ++k) {
                read_all(b0.data(), n*8);
                read_all(b1.data(), n*8);
                for (int i = 0; i < n; ++i) y[i] = b0[i] + t*b1[i];
                write_all(y.data(), n*8);
            }
        } else if (a.pair_blocks) {
            BlockFunctions blocks(a, t);
            for (int k = 0; k < requests; ++k) {
                read_all(b0.data(), n*8);
                read_all(b1.data(), n*8);
                blocks.apply(a, b0, b1, y);
                write_all(y.data(), n*8);
            }
        } else {
            if (t*a.upper > 64) a.tighten_bound();
            Chebyshev solver(a, t, rtol, atol);
            std::unique_ptr<Lanczos> krylov;
            bool use_krylov = t*a.upper > 2000 &&
                (a.diagonal_cv > 0.08 || a.positive_offdiagonal || a.lower > 0.02*a.upper);
            if (use_krylov) {
                int m = std::min(450, std::max(24, int(1.5*std::sqrt(t*a.upper))));
                krylov.reset(new Lanczos(a, m, solver));
            }
            for (int k = 0; k < requests; ++k) {
                read_all(b0.data(), n*8);
                read_all(b1.data(), n*8);
                bool done = t*a.upper > 64 && eigenmode_action(a, t, rtol, atol, b0, b1, y);
                bool cancellation = false;
                if (!done && krylov) {
                    done = krylov->apply(b0, b1, t, rtol, atol, y);
                    cancellation = krylov->cancellation;
                    if (!done) krylov.reset();
                }
                if (!done) {
                    solver.apply(b0, b1, y, cancellation);
                }
                write_all(y.data(), n*8);
            }
        }
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
