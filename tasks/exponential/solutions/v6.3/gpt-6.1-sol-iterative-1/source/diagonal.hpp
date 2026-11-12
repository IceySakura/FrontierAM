#pragma once

#include <immintrin.h>

class DiagonalKernel {
    int n;
    std::vector<int> offsets;
    Vec values;
    int padding = 0;
    bool valid = false;

    __m256d product4(const double* x, const Vec& rowsum, int i) const {
        __m256d xi = _mm256_loadu_pd(x + i);
        __m256d s0 = _mm256_mul_pd(_mm256_loadu_pd(rowsum.data() + i), xi);
        __m256d s1 = _mm256_setzero_pd(), s2 = s1, s3 = s1;
        int j = 0, count = static_cast<int>(offsets.size());
        for (; j + 3 < count; j += 4) {
            __m256d v0 = _mm256_loadu_pd(values.data() + static_cast<size_t>(j) * n + i);
            __m256d v1 = _mm256_loadu_pd(values.data() + static_cast<size_t>(j + 1) * n + i);
            __m256d v2 = _mm256_loadu_pd(values.data() + static_cast<size_t>(j + 2) * n + i);
            __m256d v3 = _mm256_loadu_pd(values.data() + static_cast<size_t>(j + 3) * n + i);
            s0 = _mm256_fmadd_pd(v0, _mm256_sub_pd(_mm256_loadu_pd(x + i + offsets[j]), xi), s0);
            s1 = _mm256_fmadd_pd(v1, _mm256_sub_pd(_mm256_loadu_pd(x + i + offsets[j + 1]), xi), s1);
            s2 = _mm256_fmadd_pd(v2, _mm256_sub_pd(_mm256_loadu_pd(x + i + offsets[j + 2]), xi), s2);
            s3 = _mm256_fmadd_pd(v3, _mm256_sub_pd(_mm256_loadu_pd(x + i + offsets[j + 3]), xi), s3);
        }
        __m256d sum = _mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3));
        for (; j < count; ++j) {
            __m256d v = _mm256_loadu_pd(values.data() + static_cast<size_t>(j) * n + i);
            sum = _mm256_fmadd_pd(v, _mm256_sub_pd(_mm256_loadu_pd(x + i + offsets[j]), xi), sum);
        }
        return sum;
    }

    double product1(const double* x, const Vec& rowsum, int i) const {
        double sum = rowsum[i] * x[i];
        for (size_t j = 0; j < offsets.size(); ++j)
            sum += values[j * n + i] * (x[i + offsets[j]] - x[i]);
        return sum;
    }

public:
    DiagonalKernel(int size, const std::vector<int>& row,
                   const std::vector<uint32_t>& col, const Vec& val) : n(size) {
        for (int i = 0; i < n; ++i) {
            for (int k = row[i]; k < row[i + 1]; ++k) {
                int off = static_cast<int>(col[k]) - i;
                auto at = std::lower_bound(offsets.begin(), offsets.end(), off);
                if (at == offsets.end() || *at != off) {
                    offsets.insert(at, off);
                    if (offsets.size() > 32) return;
                }
            }
        }
        if (offsets.empty() || val.size() < 0.55 * offsets.size() * n) return;
        for (int off : offsets) padding = std::max(padding, std::abs(off) + 4);
        values.assign(offsets.size() * n, 0);
        for (int i = 0; i < n; ++i) {
            for (int k = row[i]; k < row[i + 1]; ++k) {
                int off = static_cast<int>(col[k]) - i;
                size_t d = std::lower_bound(offsets.begin(), offsets.end(), off) - offsets.begin();
                values[d * n + i] = val[k];
            }
        }
        valid = true;
        if (std::getenv("FAMEXP_DEBUG"))
            std::fprintf(stderr, "diagonal kernel offsets=%zu\n", offsets.size());
    }

    bool usable() const { return valid; }

    void apply(const Vec& b0, const Vec& b1, const Vec& ce, const Vec& cp,
               const Vec& rowsum, double upper, int degree, Vec& y) const {
        Vec u_store(n + 2 * padding, 0), next_store(n + 2 * padding, 0), d(n, 0);
        double factor = -4 / upper;
        __m256d scale = _mm256_set1_pd(factor);
        for (int k = degree; k >= 1; --k) {
            const double* u = u_store.data() + padding;
            double* next = next_store.data() + padding;
            __m256d ek = _mm256_set1_pd(ce[k]), pk = _mm256_set1_pd(cp[k]);
            int i = 0;
            for (; i + 3 < n; i += 4) {
                __m256d dot = product4(u, rowsum, i);
                __m256d di = _mm256_fmadd_pd(scale, dot, _mm256_loadu_pd(d.data() + i));
                di = _mm256_fmadd_pd(ek, _mm256_loadu_pd(b0.data() + i), di);
                di = _mm256_fmadd_pd(pk, _mm256_loadu_pd(b1.data() + i), di);
                _mm256_storeu_pd(d.data() + i, di);
                _mm256_storeu_pd(next + i, _mm256_add_pd(_mm256_loadu_pd(u + i), di));
            }
            for (; i < n; ++i) {
                double di = d[i] + factor * product1(u, rowsum, i) +
                            ce[k] * b0[i] + cp[k] * b1[i];
                d[i] = di;
                next[i] = u[i] + di;
            }
            u_store.swap(next_store);
        }
        const double* u = u_store.data() + padding;
        scale = _mm256_set1_pd(0.5 * factor);
        __m256d e0 = _mm256_set1_pd(ce[0]), p0 = _mm256_set1_pd(cp[0]);
        int i = 0;
        for (; i + 3 < n; i += 4) {
            __m256d yi = _mm256_fmadd_pd(scale, product4(u, rowsum, i),
                                       _mm256_loadu_pd(d.data() + i));
            yi = _mm256_fmadd_pd(e0, _mm256_loadu_pd(b0.data() + i), yi);
            yi = _mm256_fmadd_pd(p0, _mm256_loadu_pd(b1.data() + i), yi);
            _mm256_storeu_pd(y.data() + i, yi);
        }
        for (; i < n; ++i)
            y[i] = d[i] + 0.5 * factor * product1(u, rowsum, i) +
                   ce[0] * b0[i] + cp[0] * b1[i];
    }
};
