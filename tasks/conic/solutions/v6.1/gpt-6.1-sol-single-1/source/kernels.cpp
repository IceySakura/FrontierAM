#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

extern "C" {

int64_t affine_product(
    int n, const int64_t* ap, const int32_t* ai, const double* av,
    const int64_t* tp, const int32_t* ti, const double* tv,
    const uint8_t* keep, int32_t* ri, int32_t* ci, double* val) {
    int64_t used = 0;
    for (int j = 0; j < n; ++j) {
        for (int64_t k = ap[j]; k < ap[j + 1]; ++k) {
            int row = ai[k];
            if (!keep[row]) continue;
            for (int64_t l = tp[j]; l < tp[j + 1]; ++l) {
                double v = av[k] * tv[l];
                if (v != 0.0) {
                    ri[used] = row;
                    ci[used] = ti[l];
                    val[used++] = v;
                }
            }
        }
    }
    return used;
}

void gp_derivatives(
    int terms, int n, int m,
    const int64_t* mp, const int32_t* mi, const double* mv,
    const int64_t* cp, const int32_t* ci, const double* cv,
    const double* ev, const double* weight,
    const double* base, double* J, double* H) {
    std::memcpy(J, base, sizeof(double) * m * n);
    std::memset(H, 0, sizeof(double) * n * n);
    for (int i = 0; i < terms; ++i) {
        double w = ev[i] * weight[i];
        for (int64_t a = mp[i]; a < mp[i + 1]; ++a) {
            int row = mi[a];
            double v = w * mv[a];
            for (int64_t b = mp[i]; b < mp[i + 1]; ++b)
                H[static_cast<int64_t>(row) * n + mi[b]] += v * mv[b];
        }
    }
    for (int row = 0; row < m; ++row) {
        for (int64_t a = cp[row]; a < cp[row + 1]; ++a) {
            int term = ci[a];
            double v = cv[a] * ev[term];
            for (int64_t b = mp[term]; b < mp[term + 1]; ++b)
                J[static_cast<int64_t>(row) * n + mi[b]] += v * mv[b];
        }
    }
}

int gp_evaluate(
    int terms, int n, int m,
    const int64_t* mp, const int32_t* mi, const double* mv,
    const int64_t* cp, const int32_t* ci, const double* cv,
    const double* B, const double* x, const double* s, const double* z,
    const double* qb, const double* qo, const double* offset,
    const double* yc, const double* rhs, double target,
    double* theta, double* ev, double* weight,
    double* rd, double* rp, double* rc) {
    for (int i = 0; i < terms; ++i) {
        double value = offset[i];
        for (int64_t k = mp[i]; k < mp[i + 1]; ++k)
            value += mv[k] * x[mi[k]];
        if (value > 700.0 || !std::isfinite(value)) return 1;
        theta[i] = value;
        ev[i] = yc[i] * std::exp(std::max(value, -745.0));
        weight[i] = qo[i];
    }
    std::memcpy(rd, qb, sizeof(double) * n);
    for (int row = 0; row < m; ++row) {
        double v = s[row] - rhs[row];
        const double* br = B + static_cast<int64_t>(row) * n;
        for (int col = 0; col < n; ++col) {
            v += br[col] * x[col];
            rd[col] += br[col] * z[row];
        }
        for (int64_t k = cp[row]; k < cp[row + 1]; ++k) {
            v += cv[k] * ev[ci[k]];
            weight[ci[k]] += cv[k] * z[row];
        }
        rp[row] = v;
        rc[row] = s[row] * z[row] - target;
    }
    for (int i = 0; i < terms; ++i) {
        double v = ev[i] * weight[i];
        for (int64_t k = mp[i]; k < mp[i + 1]; ++k)
            rd[mi[k]] += mv[k] * v;
    }
    return 0;
}

void lasso_sweeps(
    int n, const int64_t* ptr, const int32_t* row, const double* data,
    const double* weighted, const double* norm,
    const double* positive, const double* negative,
    double* x, double* residual, int sweeps) {
    for (int pass = 0; pass < sweeps; ++pass) {
        for (int j = 0; j < n; ++j) {
            if (norm[j] <= 1e-300) continue;
            double corr = norm[j] * x[j];
            for (int64_t k = ptr[j]; k < ptr[j + 1]; ++k)
                corr += weighted[k] * residual[row[k]];
            double value = 0.0;
            if (corr > positive[j]) value = (corr - positive[j]) / norm[j];
            else if (corr < -negative[j]) value = (corr + negative[j]) / norm[j];
            double delta = value - x[j];
            if (delta != 0.0) {
                for (int64_t k = ptr[j]; k < ptr[j + 1]; ++k)
                    residual[row[k]] -= data[k] * delta;
                x[j] = value;
            }
        }
    }
}

void lasso_metrics(
    int n, int m, const int64_t* ptr, const int32_t* row,
    const double* weighted, const double* diagonal,
    const double* positive, const double* negative,
    const double* x, const double* residual, double* out) {
    double worst = 0.0, gap = 0.0, penalty = 0.0, loss = 0.0;
    for (int i = 0; i < m; ++i)
        loss += diagonal[i] * residual[i] * residual[i];
    for (int j = 0; j < n; ++j) {
        double corr = 0.0;
        for (int64_t k = ptr[j]; k < ptr[j + 1]; ++k)
            corr += weighted[k] * residual[row[k]];
        double cost = positive[j] * std::max(x[j], 0.0) +
                      negative[j] * std::max(-x[j], 0.0);
        double violation;
        if (x[j] > 0.0) violation = std::abs(corr - positive[j]);
        else if (x[j] < 0.0) violation = std::abs(corr + negative[j]);
        else violation = std::max(0.0, std::max(corr - positive[j], -corr - negative[j]));
        worst = std::max(worst, violation);
        penalty += cost;
        gap += cost - x[j] * corr;
    }
    out[0] = worst;
    out[1] = gap;
    out[2] = 0.5 * loss + penalty;
}

}
