#include <memory>
#include <array>
#include <cblas.h>

class AdaptiveKrylov {
    const Matrix& A;
    const Header& h;
    int limit;
    std::unique_ptr<double[]> basis;
    Vec residual, z, diagonal, off;
    bool debug, exponential = false;
    bool reuse_hint = false, previous_exponential = false;
    double previous_input = 0, previous_output = 0, previous_mean = 0;
    std::array<double, 5> previous_signature{};

    Vec project(double start_norm) const {
        int m = static_cast<int>(diagonal.size());
        Vec d = diagonal, e = off, values(m), vectors(m * m), weights(m), result(m);
        e.resize(m);
        std::vector<lapack_int> support(2 * m);
        lapack_int found = 0;
        int status = LAPACKE_dstevr(LAPACK_COL_MAJOR, 'V', 'A', m,
            d.data(), e.data(), 0, 0, 0, 0, 0, &found, values.data(),
            vectors.data(), m, support.data());
        if (status || found != m) throw std::runtime_error("tridiagonal eigensolver failed");
        for (int j = 0; j < m; ++j) {
            double lambda = std::max(0.0, values[j]) * A.upper;
            double function = exponential ? std::exp(-h.t * lambda) : phi_scalar(lambda, h.t);
            weights[j] = vectors[j * m] * start_norm * function;
        }
        cblas_dgemv(CblasColMajor, CblasNoTrans, m, m, 1, vectors.data(), m,
                    weights.data(), 1, 0, result.data(), 1);
        return result;
    }

    void reconstruct(const Vec& c, const Vec& b0, Vec& y) const {
        if (exponential) std::fill(y.begin(), y.end(), 0);
        else y = b0;
        cblas_dgemv(CblasColMajor, CblasNoTrans, A.n, static_cast<int>(c.size()),
                    1, basis.get(), A.n, c.data(), 1, 1, y.data(), 1);
    }

    double error_bound(double beta, double start_norm, double log_off) const {
        if (beta == 0) return 0;
        int m = static_cast<int>(diagonal.size());
        double eta = h.t * A.upper;
        // After alternating signs, exp(-s T) has nonnegative entries.
        // (eta-s) <= eta*exp(q-1)/q*exp(-q*s/eta), q>=1.
        // Its integral is therefore bounded by a shifted tridiagonal inverse.
        auto objective = [&](double q) {
            double shift = q / eta;
            double pivot = diagonal[0] + shift;
            if (pivot <= 0) return std::numeric_limits<double>::infinity();
            double logdet = std::log(pivot);
            for (int i = 1; i < m; ++i) {
                pivot = diagonal[i] + shift - off[i - 1] * off[i - 1] / pivot;
                if (pivot <= 0) return std::numeric_limits<double>::infinity();
                logdet += std::log(pivot);
            }
            return (exponential ? q : q - 1 - std::log(q)) + log_off - logdet;
        };
        double minimum_q = exponential ? 1e-5 : 1;
        double left = minimum_q, right = m + 1;
        double x1 = right - (right - left) * 0.6180339887498949;
        double x2 = left + (right - left) * 0.6180339887498949;
        double f1 = objective(x1), f2 = objective(x2);
        for (int k = 0; k < 20; ++k) {
            if (f1 < f2) {
                right = x2;
                x2 = x1; f2 = f1;
                x1 = right - (right - left) * 0.6180339887498949;
                f1 = objective(x1);
            } else {
                left = x1;
                x1 = x2; f1 = f2;
                x2 = left + (right - left) * 0.6180339887498949;
                f2 = objective(x2);
            }
        }
        double best = std::min({f1, f2, objective(minimum_q)});
        double log_bound = std::log(beta) + std::log(start_norm) +
                           (exponential ? 0 : std::log(h.t)) + best + 1e-5;
        return std::exp(log_bound);
    }

public:
    AdaptiveKrylov(const Matrix& matrix, const Header& header)
        : A(matrix), h(header), limit(std::min(A.n, 700)), residual(A.n), z(A.n),
          debug(std::getenv("FAMEXP_DEBUG") != nullptr) {
        limit = std::min(limit, static_cast<int>((768ULL << 20) / (8ULL * A.n)) - 1);
        basis.reset(new double[static_cast<size_t>(A.n) * (limit + 1)]);
        const char* hint = std::getenv("FAMEXP_REUSE_HINT");
        reuse_hint = h.requests > 1 && (!hint || std::strcmp(hint, "0") != 0);
    }

    bool solve(const Vec& b0, const Vec& b1, Vec& y, int cap = 0, double known = 0) {
        int max_steps = cap > 0 ? std::min(limit, cap) : limit;
        double b1max = norm_inf(b1), b0max = norm_inf(b0);
        exponential = b1max == 0;
        if (!exponential && h.rtol <= 1e-10 && h.t * b1max < 1e-6 * b0max) return false;
        if (exponential) residual = b0;
        else {
            A.multiply(b0, residual);
            for (int i = 0; i < A.n; ++i) residual[i] = b1[i] - residual[i];
        }
        double start_norm = cblas_dnrm2(A.n, residual.data(), 1);
        if (start_norm == 0) {
            if (exponential) std::fill(y.begin(), y.end(), 0);
            else y = b0;
            return true;
        }
        double* q = basis.get();
        for (int i = 0; i < A.n; ++i) q[i] = residual[i] / start_norm;
        diagonal.clear();
        off.clear();
        double eta = h.t * A.upper;
        double input_scale = b0max + h.t * b1max;
        double guess = input_scale / (1 + std::sqrt(eta));
        std::array<double, 5> signature{};
        bool hinted = false;
        double current_mean = 0;
        if (reuse_hint) {
            double square0 = 0, square1 = 0, sum0 = 0, sum1 = 0;
            for (int i = 0; i < A.n; ++i) {
                square0 += b0[i] * b0[i]; square1 += b1[i] * b1[i];
                sum0 += b0[i]; sum1 += b1[i];
            }
            signature = {b0max ? std::sqrt(square0 / A.n) / b0max : 0,
                         b1max ? std::sqrt(square1 / A.n) / b1max : 0,
                         b0max ? std::abs(sum0 / A.n) / b0max : 0,
                         b1max ? std::abs(sum1 / A.n) / b1max : 0,
                         h.t * b1max / input_scale};
            if (A.conservative)
                current_mean = std::abs((sum0 + h.t * sum1) /
                    std::max(1, A.n - static_cast<int>(A.scalar_index.size())));
            bool compatible = previous_input > 0 && previous_exponential == exponential;
            for (int i = 0; i < 5; ++i)
                if (std::abs(signature[i] - previous_signature[i]) > 0.05) compatible = false;
            if (compatible) {
                double estimate = previous_output * input_scale / previous_input;
                if (A.conservative)
                    estimate = current_mean + std::max(0.0, previous_output - previous_mean) *
                                              input_scale / previous_input;
                guess = std::max(guess * 0.5, estimate);
                hinted = true;
            }
        }
        if (A.conservative && !reuse_hint) {
            double sum0 = 0, sum1 = 0;
            for (int i = 0; i < A.n; ++i) { sum0 += b0[i]; sum1 += b1[i]; }
            current_mean = std::abs((sum0 + h.t * sum1) /
                std::max(1, A.n - static_cast<int>(A.scalar_index.size())));
        }
        guess = std::max(guess, current_mean);
        guess = std::max(guess, known);
        double predicted = 0.05 * (h.atol + h.rtol * guess);
        double log_off = 0;
        for (int j = 0; j < max_steps; ++j) {
            const double* cur = basis.get() + static_cast<size_t>(j) * A.n;
            const double* old = j ? cur - A.n : nullptr;
            double beta_old = j ? off.back() : 0;
            A.multiply_raw(cur, z.data());
            for (int i = 0; i < A.n; ++i) z[i] /= A.upper;
            double alpha = cblas_ddot(A.n, cur, 1, z.data(), 1);
            for (int i = 0; i < A.n; ++i)
                z[i] -= alpha * cur[i] + (j ? beta_old * old[i] : 0);
            double beta = cblas_dnrm2(A.n, z.data(), 1);
            diagonal.push_back(alpha);
            int m = j + 1;
            bool inspect = m == 8 || (m >= 16 && m % (m < 128 ? 8 : 16) == 0) ||
                           m == max_steps || beta < 1e-14;
            if (inspect) {
                double residual_norm = A.diagonally_dominant ? norm_inf(z) : beta;
                double bound = error_bound(residual_norm, start_norm, log_off);
                if (bound <= predicted || beta == 0) {
                    Vec c = project(start_norm);
                    reconstruct(c, b0, y);
                    double y_norm = norm_inf(y);
                    double tolerance = 0.08 * (h.atol + h.rtol * std::max(y_norm, known));
                    if (bound <= tolerance) {
                        previous_input = input_scale; previous_output = y_norm;
                        previous_mean = current_mean;
                        previous_signature = signature; previous_exponential = exponential;
                        if (debug) std::fprintf(stderr,
                            "krylov n=%d eta=%.6g degree=%d bound=%.3g norm=%.6g hint=%d\n",
                            A.n, eta, m, bound, y_norm, static_cast<int>(hinted));
                        return true;
                    }
                    predicted = std::min(predicted, tolerance);
                }
            }
            if (beta == 0) break;
            off.push_back(beta);
            log_off += std::log(beta);
            double* next = basis.get() + static_cast<size_t>(j + 1) * A.n;
            for (int i = 0; i < A.n; ++i) next[i] = z[i] / beta;
        }
        if (debug) std::fprintf(stderr, "krylov fallback n=%d eta=%.6g degree=%zu\n",
                                A.n, eta, diagonal.size());
        return false;
    }
};
