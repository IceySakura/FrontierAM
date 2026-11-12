#pragma once

#include <array>
#include <complex>

namespace shifted {

using C = std::complex<double>;
using CL = std::complex<long double>;
using Eight = std::array<C, 8>;

struct Matrix {
    int n;
    const int* row;
    const uint32_t* col;
    const double* val;
    const double* diagonal;
    const double* rowsum;
    double t;
};

static Eight dot(const Vec& a, const Vec& b, int n) {
    double sum[16] = {};
    for (int i = 0; i < n; ++i) {
        size_t base = static_cast<size_t>(i) * 16;
        for (int p = 0; p < 8; ++p) {
            int r = 2 * p, im = r + 1;
            sum[r] += a[base + r] * b[base + r] - a[base + im] * b[base + im];
            sum[im] += a[base + r] * b[base + im] + a[base + im] * b[base + r];
        }
    }
    Eight result;
    for (int p = 0; p < 8; ++p) result[p] = C(sum[2 * p], sum[2 * p + 1]);
    return result;
}

static void product(const Matrix& a, const Vec& x, Vec& out, const Eight& theta) {
    for (int i = 0; i < a.n; ++i) {
        const double* xi = x.data() + static_cast<size_t>(i) * 16;
        alignas(32) double sum[16];
        double rs = a.t * a.rowsum[i];
        for (int p = 0; p < 16; ++p) sum[p] = rs * xi[p];
        for (int k = a.row[i]; k < a.row[i + 1]; ++k) {
            const double* xj = x.data() + static_cast<size_t>(a.col[k]) * 16;
            double v = a.t * a.val[k];
            for (int p = 0; p < 16; ++p) sum[p] += v * (xj[p] - xi[p]);
        }
        double* yi = out.data() + static_cast<size_t>(i) * 16;
        for (int p = 0; p < 8; ++p) {
            int r = p * 2, im = r + 1;
            double tr = theta[p].real(), ti = theta[p].imag();
            yi[r] = sum[r] + tr * xi[r] - ti * xi[im];
            yi[im] = sum[im] + ti * xi[r] + tr * xi[im];
        }
    }
}

static void precondition(const Matrix& a, const Vec& r, Vec& z,
                         const Vec& inverse, const Eight& theta,
                         bool coarse, double coarse_diagonal,
                         const std::array<bool, 8>& active) {
    double mean[16] = {}, mean_z[16] = {};
    if (coarse) {
        for (int i = 0; i < a.n; ++i)
            for (int p = 0; p < 16; ++p) mean[p] += r[static_cast<size_t>(i) * 16 + p];
        for (double& v : mean) v /= a.n;
    }
    for (int i = 0; i < a.n; ++i) {
        size_t base = static_cast<size_t>(i) * 16;
        for (int p = 0; p < 8; ++p) {
            int re = 2 * p, im = re + 1;
            double rr = r[base + re] - mean[re], ri = r[base + im] - mean[im];
            double dr = inverse[base + re], di = inverse[base + im];
            double zr = rr * dr - ri * di, zi = rr * di + ri * dr;
            z[base + re] = zr;
            z[base + im] = zi;
            mean_z[re] += zr;
            mean_z[im] += zi;
        }
    }
    double correction[16] = {};
    if (coarse) {
        for (int p = 0; p < 8; ++p) {
            C value = C(mean[2 * p], mean[2 * p + 1]) /
                      (theta[p] + coarse_diagonal);
            correction[2 * p] = value.real() - mean_z[2 * p] / a.n;
            correction[2 * p + 1] = value.imag() - mean_z[2 * p + 1] / a.n;
        }
    }
    for (int i = 0; i < a.n; ++i) {
        size_t base = static_cast<size_t>(i) * 16;
        for (int p = 0; p < 8; ++p) {
            z[base + 2 * p] = active[p] ? z[base + 2 * p] + correction[2 * p] : 0;
            z[base + 2 * p + 1] = active[p] ?
                z[base + 2 * p + 1] + correction[2 * p + 1] : 0;
        }
    }
}

static Eight norms(const Vec& r, int n) {
    double sum[8] = {};
    for (int i = 0; i < n; ++i) {
        size_t base = static_cast<size_t>(i) * 16;
        for (int p = 0; p < 8; ++p) {
            double re = r[base + 2 * p], im = r[base + 2 * p + 1];
            sum[p] += re * re + im * im;
        }
    }
    Eight result;
    for (int p = 0; p < 8; ++p) result[p] = std::sqrt(sum[p]);
    return result;
}

static long double error_bound(const Eight& sizes, const double* weights) {
    long double total = 0;
    for (int p = 0; p < 8; ++p) total += weights[p] * sizes[p].real();
    return total;
}

static double output(const Vec& x, const Vec& b0, const CL* alpha,
                     long double constant, Vec& y, int n, double offset) {
    y.resize(n);
    double maximum = 0;
    for (int i = 0; i < n; ++i) {
        size_t base = static_cast<size_t>(i) * 16;
        long double v = constant * b0[i];
        for (int p = 0; p < 8; ++p)
            v -= 2 * (alpha[p].real() * x[base + 2 * p] -
                      alpha[p].imag() * x[base + 2 * p + 1]);
        y[i] = static_cast<double>(v);
        maximum = std::max(maximum, std::abs(y[i] + offset));
    }
    return maximum;
}

// This audit uses the original diagonal and long-double residual accumulation,
// not the recurred residual or the difference-form product used by the solver.
static long double audit(const Matrix& a, const Vec& x, const Vec& b0,
                         const Vec& b1, const CL* theta, const CL* alpha,
                         const CL* phi, const double* weights,
                         Vec& residual) {
    std::array<CL, 8> rhs_scale;
    for (int p = 0; p < 8; ++p)
        rhs_scale[p] = static_cast<long double>(a.t) * phi[p] / alpha[p];
    long double norm[8] = {}, round[8] = {};
    constexpr long double eps = std::numeric_limits<long double>::epsilon();
    for (int i = 0; i < a.n; ++i) {
        size_t base = static_cast<size_t>(i) * 16;
        int length = a.row[i + 1] - a.row[i];
        for (int p = 0; p < 8; ++p) {
            long double xr = x[base + 2 * p], xi = x[base + 2 * p + 1];
            long double diag = static_cast<long double>(a.t) * a.diagonal[i];
            long double rr = (diag + theta[p].real()) * xr - theta[p].imag() * xi -
                             b0[i] - rhs_scale[p].real() * b1[i];
            long double ri = (diag + theta[p].real()) * xi + theta[p].imag() * xr -
                             rhs_scale[p].imag() * b1[i];
            long double ab = std::abs(diag * xr) + std::abs(diag * xi) +
                std::abs(theta[p]) * (std::abs(xr) + std::abs(xi)) +
                std::abs(b0[i]) + std::abs(rhs_scale[p]) * std::abs(b1[i]);
            long double cr = 0, ci = 0;
            for (int k = a.row[i]; k < a.row[i + 1]; ++k) {
                size_t j = static_cast<size_t>(a.col[k]) * 16;
                long double v = static_cast<long double>(a.t) * a.val[k];
                long double vr = v * x[j + 2 * p], vi = v * x[j + 2 * p + 1];
                ab += std::abs(vr) + std::abs(vi);
                if (length > 32) {
                    long double sr = vr - cr, si = vi - ci;
                    long double nr = rr + sr, ni = ri + si;
                    cr = (nr - rr) - sr;
                    ci = (ni - ri) - si;
                    rr = nr;
                    ri = ni;
                } else {
                    rr += vr;
                    ri += vi;
                }
            }
            residual[base + 2 * p] = static_cast<double>(-rr);
            residual[base + 2 * p + 1] = static_cast<double>(-ri);
            norm[p] += rr * rr + ri * ri;
            long double guard = (length > 32 ? 16 : 2 * length + 16) * eps * ab;
            round[p] += guard * guard;
        }
    }
    long double bound = 0;
    for (int p = 0; p < 8; ++p)
        bound += weights[p] * (std::sqrt(norm[p]) + std::sqrt(round[p]));
    return bound;
}

static bool solve(const Matrix& a, const Vec& b0, const Vec& b1,
                  const CL* theta_long, const CL* alpha, const CL* phi, long double constant,
                  double atol, double rtol, long double target, int cap, Vec& y,
                  double offset) {
    size_t size = static_cast<size_t>(a.n) * 16;
    Vec x(size, 0), r(size), z(size), p(size), q(size), inverse(size);
    Eight theta, forcing;
    double weights[8];
    for (int j = 0; j < 8; ++j) {
        theta[j] = static_cast<C>(theta_long[j]);
        forcing[j] = static_cast<C>(static_cast<long double>(a.t) * phi[j] / alpha[j]);
        long double distance = theta_long[j].real() >= 0 ? std::abs(theta_long[j]) :
                                                            theta_long[j].imag();
        weights[j] = static_cast<double>(2 * std::abs(alpha[j]) / distance);
    }
    long double sum_diag = 0, sum_row = 0, norm_diag = 0, norm_row = 0;
    for (int i = 0; i < a.n; ++i) {
        sum_diag += a.diagonal[i];
        sum_row += a.rowsum[i];
        norm_diag += static_cast<long double>(a.diagonal[i]) * a.diagonal[i];
        norm_row += static_cast<long double>(a.rowsum[i]) * a.rowsum[i];
        size_t base = static_cast<size_t>(i) * 16;
        double diagonal = a.t * a.diagonal[i];
        for (int j = 0; j < 8; ++j) {
            double re = diagonal + theta[j].real(), im = theta[j].imag();
            double reciprocal = 1 / (re * re + im * im);
            inverse[base + 2 * j] = re * reciprocal;
            inverse[base + 2 * j + 1] = -im * reciprocal;
            r[base + 2 * j] = b0[i] + forcing[j].real() * b1[i];
            r[base + 2 * j + 1] = forcing[j].imag() * b1[i];
        }
    }
    bool coarse = norm_row < 1e-10L * norm_diag;
    double coarse_diagonal = static_cast<double>(a.t * sum_row / a.n);
    std::array<bool, 8> active;
    active.fill(true);
    precondition(a, r, z, inverse, theta, coarse, coarse_diagonal, active);
    p = z;
    Eight rho = dot(r, z, a.n);
    long double initial = error_bound(norms(r, a.n), weights);
    long double best = initial;
    int restarts = 0;
    long double bn0 = norm2(b0);
    for (int it = 0; it < cap; ++it) {
        product(a, p, q, theta);
        Eight denominator = dot(p, q, a.n), step;
        for (int j = 0; j < 8; ++j) {
            if (active[j]) {
                step[j] = rho[j] / denominator[j];
                if (!std::isfinite(std::abs(step[j]))) return false;
            } else step[j] = 0;
        }
        for (int i = 0; i < a.n; ++i) {
            size_t base = static_cast<size_t>(i) * 16;
            for (int j = 0; j < 8; ++j) {
                int re = 2 * j, im = re + 1;
                double ar = step[j].real(), ai = step[j].imag();
                x[base + re] += ar * p[base + re] - ai * p[base + im];
                x[base + im] += ar * p[base + im] + ai * p[base + re];
                r[base + re] -= ar * q[base + re] - ai * q[base + im];
                r[base + im] -= ar * q[base + im] + ai * q[base + re];
            }
        }
        Eight sizes = norms(r, a.n);
        long double estimate = error_bound(sizes, weights);
        best = std::min(best, estimate);
        if (std::getenv("FAMEXP_DEBUG") && (it + 1) % 8 == 0)
            std::fprintf(stderr, "block progress it=%d relative=%.4Lg estimate=%.4Lg\n",
                         it + 1, best / initial, estimate);
        bool restarted = false;
        if (estimate <= target || (it + 1) % 8 == 0 || it + 1 == cap) {
            double maximum = output(x, b0, alpha, constant, y, a.n, offset);
            long double tolerance = atol + rtol * maximum;
            target = 0.2L * tolerance;
            if (estimate <= target || it + 1 == cap) {
                long double actual = audit(a, x, b0, b1, theta_long, alpha, phi, weights, r);
                long double floor = 1e-14L * (norm2(y) + bn0);
                if (std::getenv("FAMEXP_DEBUG"))
                    std::fprintf(stderr, "block it=%d residual=%.4Lg floor=%.4Lg norm=%.8g\n",
                                 it + 1, actual, floor, maximum);
                if (actual + floor <= 0.5L * tolerance) return true;
                if (++restarts > 2) return false;
                active.fill(true);
                restarted = true;
            }
        }
        if ((it + 1) == 8 && best > initial * 0.05L && best > target * 1e5L &&
            !std::getenv("FAMEXP_BLOCK")) return false;
        if ((it + 1) == 16 && best > initial * 1e-3L && best > target * 1e4L &&
            !std::getenv("FAMEXP_BLOCK"))
            return false;
        if (!restarted) {
            for (int j = 0; j < 8; ++j)
                if (weights[j] * sizes[j].real() <= target / 16) active[j] = false;
        }
        precondition(a, r, z, inverse, theta, coarse, coarse_diagonal, active);
        Eight new_rho = dot(r, z, a.n), update;
        for (int j = 0; j < 8; ++j) {
            update[j] = (!restarted && active[j]) ? new_rho[j] / rho[j] : C(0);
            if (!std::isfinite(std::abs(update[j]))) return false;
        }
        for (int i = 0; i < a.n; ++i) {
            size_t base = static_cast<size_t>(i) * 16;
            for (int j = 0; j < 8; ++j) {
                int re = 2 * j, im = re + 1;
                double br = update[j].real(), bi = update[j].imag();
                double pr = p[base + re], pi = p[base + im];
                p[base + re] = z[base + re] + br * pr - bi * pi;
                p[base + im] = z[base + im] + br * pi + bi * pr;
            }
        }
        rho = new_rho;
    }
    return false;
}

}  // namespace shifted
