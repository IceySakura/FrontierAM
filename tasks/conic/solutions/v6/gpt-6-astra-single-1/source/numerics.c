#include <math.h>
#include <stdint.h>

/* Cyclic coordinate minimization, with the residual maintained in place. */
double lasso_sweeps(int32_t count, const int32_t *active,
                   const int32_t *ptr, const int32_t *row, const double *data,
                   const double *diagonal, const double *penalty,
                   double *x, double *residual, int32_t sweeps, double tol) {
    double change = 0.0;
    for (int32_t it = 0; it < sweeps; ++it) {
        change = 0.0;
        for (int32_t k = 0; k < count; ++k) {
            int32_t j = active[k];
            double h = diagonal[j], g = 0.0;
            if (h <= 0.0) continue;
            for (int32_t i = ptr[j]; i < ptr[j + 1]; ++i)
                g += data[i] * residual[row[i]];
            double v = x[j] + g / h, t = penalty[j] / h;
            double next = v > t ? v - t : v < -t ? v + t : 0.0;
            double delta = next - x[j];
            if (delta != 0.0) {
                x[j] = next;
                for (int32_t i = ptr[j]; i < ptr[j + 1]; ++i)
                    residual[row[i]] -= delta * data[i];
            }
            double relative = fabs(delta) / (1.0 + fabs(next));
            if (relative > change) change = relative;
        }
        if (change < tol) break;
    }
    return change;
}
