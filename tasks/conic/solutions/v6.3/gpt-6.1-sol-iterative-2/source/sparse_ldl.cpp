#include <algorithm>
#include <cmath>
#include <cstdint>
#include <new>
#include <vector>

struct Factor {
    int n;
    std::vector<int32_t> ap, ai, parent, lp, li, count, flag, pattern;
    std::vector<double> lx, d, inverse, work;
    explicit Factor(int size, const int32_t *p, const int32_t *i): n(size) {
        ap.assign(p, p + n + 1);
        ai.assign(i, i + ap.back());
        parent.assign(n, -1);
        count.assign(n, 0);
        flag.resize(n);
        pattern.resize(n);
        work.assign(n, 0);
        d.resize(n);
        inverse.resize(n);
        for (int k = 0; k < n; ++k) {
            flag[k] = k;
            for (int at = ap[k]; at < ap[k + 1]; ++at) {
                int j = ai[at];
                if (j >= k) continue;
                while (flag[j] != k) {
                    if (parent[j] == -1) parent[j] = k;
                    ++count[j];
                    flag[j] = k;
                    j = parent[j];
                }
            }
        }
        lp.resize(n + 1);
        lp[0] = 0;
        int64_t total = 0;
        for (int k = 0; k < n; ++k) {
            total += count[k];
            if (total > 100000000) throw std::bad_alloc();
            lp[k + 1] = int32_t(total);
        }
        li.resize(size_t(total));
        lx.resize(size_t(total));
    }
    bool numeric(const double *a, const double *sign) {
        for (int k = 0; k < n; ++k) {
            int top = n;
            work[k] = 0;
            flag[k] = k;
            count[k] = 0;
            for (int at = ap[k]; at < ap[k + 1]; ++at) {
                int j = ai[at];
                if (j > k) continue;
                work[j] += a[at];
                int length = 0;
                while (flag[j] != k) {
                    pattern[length++] = j;
                    flag[j] = k;
                    j = parent[j];
                }
                while (length > 0) pattern[--top] = pattern[--length];
            }
            double diagonal = work[k];
            work[k] = 0;
            for (; top < n; ++top) {
                int j = pattern[top];
                double value = work[j];
                work[j] = 0;
                int end = lp[j] + count[j];
                if (end >= lp[j + 1]) return false;
                for (int at = lp[j]; at < end; ++at)
                    work[li[at]] -= lx[at] * value;
                double l = value * inverse[j];
                diagonal -= l * value;
                li[end] = k;
                lx[end] = l;
                ++count[j];
            }
            if (!std::isfinite(diagonal) || sign[k] * diagonal <= 0) return false;
            if (std::abs(diagonal) < 1e-13) diagonal = sign[k] * 1e-13;
            d[k] = diagonal;
            inverse[k] = 1.0 / diagonal;
        }
        return true;
    }
    void solve(const double *rhs, double *x) const {
        std::copy(rhs, rhs + n, x);
        for (int j = 0; j < n; ++j) {
            double v = x[j];
            for (int at = lp[j]; at < lp[j + 1]; ++at)
                x[li[at]] -= lx[at] * v;
        }
        for (int j = 0; j < n; ++j) x[j] *= inverse[j];
        for (int j = n - 1; j >= 0; --j)
            for (int at = lp[j]; at < lp[j + 1]; ++at)
                x[j] -= lx[at] * x[li[at]];
    }
};

extern "C" void *ldl_create(int n, const int32_t *ap, const int32_t *ai) {
    try {
        return new Factor(n, ap, ai);
    } catch (...) {
        return nullptr;
    }
}
extern "C" void ldl_destroy(void *pointer) {
    delete static_cast<Factor *>(pointer);
}
extern "C" int ldl_numeric(void *pointer, const double *a, const double *sign) {
    return static_cast<Factor *>(pointer)->numeric(a, sign) ? 1 : 0;
}
extern "C" void ldl_solve(void *pointer, const double *rhs, double *x) {
    static_cast<Factor *>(pointer)->solve(rhs, x);
}
extern "C" int64_t ldl_nonzeros(void *pointer) {
    return static_cast<Factor *>(pointer)->lp.back();
}
