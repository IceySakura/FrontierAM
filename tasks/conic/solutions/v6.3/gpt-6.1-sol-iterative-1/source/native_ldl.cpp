// Sparse LDL^T numeric factorization for regularized quasi-definite systems.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

struct SparseLDL {
    int n;
    std::vector<int> ap, ai, lp, parent, count, flag, pattern, li;
    std::vector<double> lx, d, y;

    SparseLDL(int size, const int* p, const int* i) : n(size),
        ap(p, p + size + 1), ai(i, i + p[size]),
        lp(size + 1, 0), parent(size, -1), count(size, 0),
        flag(size, -1), pattern(size), d(size), y(size, 0.0) {
        for (int k = 0; k < n; ++k) {
            flag[k] = k;
            for (int pos = ap[k]; pos < ap[k + 1]; ++pos) {
                int row = ai[pos];
                if (row >= k) continue;
                while (flag[row] != k) {
                    if (parent[row] < 0) parent[row] = k;
                    ++count[row];
                    flag[row] = k;
                    row = parent[row];
                }
            }
        }
        std::int64_t total = 0;
        for (int k = 0; k < n; ++k) {
            total += count[k];
            if (total > 12000000) throw 1;
            lp[k + 1] = static_cast<int>(total);
        }
        li.resize(total);
        lx.resize(total);
    }

    int factor(const double* a, const double* signs) {
        std::fill(count.begin(), count.end(), 0);
        std::fill(flag.begin(), flag.end(), -1);
        std::fill(y.begin(), y.end(), 0.0);
        for (int k = 0; k < n; ++k) {
            int top = n;
            flag[k] = k;
            for (int pos = ap[k]; pos < ap[k + 1]; ++pos) {
                int row = ai[pos];
                if (row > k) continue;
                y[row] += a[pos];
                int len = 0;
                while (flag[row] != k) {
                    pattern[len++] = row;
                    flag[row] = k;
                    row = parent[row];
                }
                while (len) pattern[--top] = pattern[--len];
            }
            double diagonal = y[k];
            y[k] = 0.0;
            while (top < n) {
                int row = pattern[top++];
                double value = y[row];
                y[row] = 0.0;
                int end = lp[row] + count[row];
                for (int pos = lp[row]; pos < end; ++pos)
                    y[li[pos]] -= lx[pos] * value;
                double entry = value / d[row];
                diagonal -= entry * value;
                li[end] = k;
                lx[end] = entry;
                ++count[row];
            }
            if (!std::isfinite(diagonal) || std::abs(diagonal) < 1e-30 ||
                diagonal * signs[k] <= 0.0) return k + 1;
            d[k] = diagonal;
        }
        return 0;
    }

    void solve(double* x, int nrhs) const {
        for (int rhs = 0; rhs < nrhs; ++rhs) {
            double* v = x + static_cast<std::size_t>(rhs) * n;
            for (int col = 0; col < n; ++col) {
                double a = v[col];
                for (int pos = lp[col]; pos < lp[col + 1]; ++pos)
                    v[li[pos]] -= lx[pos] * a;
            }
            for (int col = 0; col < n; ++col) v[col] /= d[col];
            for (int col = n - 1; col >= 0; --col) {
                double a = v[col];
                for (int pos = lp[col]; pos < lp[col + 1]; ++pos)
                    a -= lx[pos] * v[li[pos]];
                v[col] = a;
            }
        }
    }
};

extern "C" {
int csc_positions(int nq, const int* ap, const int* ai, const int* rows,
                  const int* cols, int* positions) {
    for (int k = 0; k < nq; ++k) {
        const int* begin = ai + ap[cols[k]];
        const int* end = ai + ap[cols[k] + 1];
        const int* found = std::lower_bound(begin, end, rows[k]);
        if (found == end || *found != rows[k]) return 1;
        positions[k] = static_cast<int>(found - ai);
    }
    return 0;
}
void* sparse_ldl_create(int n, const int* ap, const int* ai) {
    try { return new SparseLDL(n, ap, ai); } catch (...) { return nullptr; }
}
void sparse_ldl_destroy(void* handle) { delete static_cast<SparseLDL*>(handle); }
std::size_t sparse_ldl_nnz(void* handle) {
    return static_cast<SparseLDL*>(handle)->lx.size();
}
int sparse_ldl_factor(void* handle, const double* a, const double* signs) {
    return static_cast<SparseLDL*>(handle)->factor(a, signs);
}
void sparse_ldl_solve(void* handle, double* x, int nrhs) {
    static_cast<SparseLDL*>(handle)->solve(x, nrhs);
}
}
