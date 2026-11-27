class SteadyState {
    const Matrix& A;
    const Header& h;
    double lower;
    Vec residual, preconditioned, direction, product;
    bool debug;
public:
    static bool eligible(const Matrix& A, const Header& h) {
        return A.lower > 0 && h.t * A.lower >= -std::log(h.atol) + 4;
    }
    SteadyState(const Matrix& matrix, const Header& header)
        : A(matrix), h(header), lower(A.lower * (1 - 1e-10)), residual(A.n),
          preconditioned(A.n), direction(A.n), product(A.n),
          debug(std::getenv("FAMEXP_DEBUG") != nullptr) {}

    bool solve(const Vec& b0, const Vec& b1, Vec& y, double known = 0) {
        std::fill(y.begin(), y.end(), 0);
        residual = b1;
        for (int i = 0; i < A.n; ++i)
            preconditioned[i] = direction[i] =
                A.diagonal[i] > 0 ? residual[i] / A.diagonal[i] : 0;
        double rz = cblas_ddot(A.n, residual.data(), 1, preconditioned.data(), 1);
        int iterations = 0;
        for (; iterations <= 800; ++iterations) {
            double tolerance = h.atol + h.rtol * std::max(norm_inf(y), known);
            double transient = std::exp(-h.t * lower) * (norm_inf(b0) + norm_inf(y));
            if (norm_inf(residual) / lower + transient <= 0.25 * tolerance) {
                A.multiply(y, product);
                for (int i = 0; i < A.n; ++i) residual[i] = b1[i] - product[i];
                if (norm_inf(residual) / lower + transient <= 0.35 * tolerance) {
                    if (debug) std::fprintf(stderr,
                        "steady n=%d iterations=%d gap_time=%.6g norm=%.6g\n",
                        A.n, iterations, h.t * lower, norm_inf(y));
                    return true;
                }
                for (int i = 0; i < A.n; ++i)
                    preconditioned[i] = direction[i] =
                        A.diagonal[i] > 0 ? residual[i] / A.diagonal[i] : 0;
                rz = cblas_ddot(A.n, residual.data(), 1, preconditioned.data(), 1);
            }
            if (iterations == 800 || rz <= 0) break;
            A.multiply(direction, product);
            double denominator = cblas_ddot(A.n, direction.data(), 1, product.data(), 1);
            if (denominator <= 0) return false;
            double alpha = rz / denominator;
            for (int i = 0; i < A.n; ++i) {
                y[i] += alpha * direction[i];
                residual[i] -= alpha * product[i];
                preconditioned[i] = A.diagonal[i] > 0 ? residual[i] / A.diagonal[i] : 0;
            }
            double next_rz = cblas_ddot(A.n, residual.data(), 1, preconditioned.data(), 1);
            double beta = next_rz / rz;
            for (int i = 0; i < A.n; ++i) direction[i] = preconditioned[i] + beta * direction[i];
            rz = next_rz;
        }
        if (debug) std::fprintf(stderr, "steady fallback n=%d iterations=%d\n", A.n, iterations);
        return false;
    }
};
