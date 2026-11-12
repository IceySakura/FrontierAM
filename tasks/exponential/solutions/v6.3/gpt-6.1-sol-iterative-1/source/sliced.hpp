#pragma once

#include <immintrin.h>

class SlicedKernel {
    int n, padded;
    std::vector<int> order, pointers;
    std::vector<uint32_t> columns;
    Vec values, row_sum;
    bool valid = false;

    __m256d product4(const double* x, int slice) const {
        __m256d xi = _mm256_loadu_pd(x + 4 * slice);
        __m256d s0 = _mm256_mul_pd(_mm256_loadu_pd(row_sum.data() + 4 * slice), xi);
        __m256d s1 = _mm256_setzero_pd(), s2 = s1, s3 = s1;
        int k = pointers[slice], end = pointers[slice + 1];
        for (; k + 15 < end; k += 16) {
            __m128i c0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(columns.data() + k));
            __m128i c1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(columns.data() + k + 4));
            __m128i c2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(columns.data() + k + 8));
            __m128i c3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(columns.data() + k + 12));
            __m256d x0 = _mm256_i32gather_pd(x, c0, 8);
            __m256d x1 = _mm256_i32gather_pd(x, c1, 8);
            __m256d x2 = _mm256_i32gather_pd(x, c2, 8);
            __m256d x3 = _mm256_i32gather_pd(x, c3, 8);
            s0 = _mm256_fmadd_pd(_mm256_loadu_pd(values.data() + k),
                                _mm256_sub_pd(x0, xi), s0);
            s1 = _mm256_fmadd_pd(_mm256_loadu_pd(values.data() + k + 4),
                                _mm256_sub_pd(x1, xi), s1);
            s2 = _mm256_fmadd_pd(_mm256_loadu_pd(values.data() + k + 8),
                                _mm256_sub_pd(x2, xi), s2);
            s3 = _mm256_fmadd_pd(_mm256_loadu_pd(values.data() + k + 12),
                                _mm256_sub_pd(x3, xi), s3);
        }
        __m256d sum = _mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3));
        for (; k < end; k += 4) {
            __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i*>(columns.data() + k));
            __m256d xj = _mm256_i32gather_pd(x, c, 8);
            sum = _mm256_fmadd_pd(_mm256_loadu_pd(values.data() + k),
                                 _mm256_sub_pd(xj, xi), sum);
        }
        return sum;
    }

public:
    SlicedKernel(int size, const std::vector<int>& row,
                 const std::vector<uint32_t>& col, const Vec& val,
                 const Vec& rowsum) : n(size), padded((size + 3) & ~3) {
        order.resize(n);
        std::iota(order.begin(), order.end(), 0);
        for (int start = 0; start < n; start += 128) {
            int end = std::min(n, start + 128);
            std::sort(order.begin() + start, order.begin() + end, [&](int a, int b) {
                int na = row[a + 1] - row[a], nb = row[b + 1] - row[b];
                return na != nb ? na < nb : a < b;
            });
        }
        std::vector<int> inverse(n);
        for (int i = 0; i < n; ++i) inverse[order[i]] = i;
        int slices = padded / 4;
        pointers.resize(slices + 1, 0);
        for (int s = 0; s < slices; ++s) {
            int maximum = 0;
            for (int lane = 0; lane < 4; ++lane) {
                int i = 4 * s + lane;
                if (i < n) maximum = std::max(maximum, row[order[i] + 1] - row[order[i]]);
            }
            pointers[s + 1] = pointers[s] + 4 * maximum;
        }
        if (pointers.back() > 1.25 * val.size()) return;
        values.resize(pointers.back());
        columns.resize(pointers.back());
        row_sum.assign(padded, 0);
        for (int s = 0; s < slices; ++s) {
            int length = (pointers[s + 1] - pointers[s]) / 4;
            for (int lane = 0; lane < 4; ++lane) {
                int i = 4 * s + lane;
                int old = i < n ? order[i] : -1;
                int count = old >= 0 ? row[old + 1] - row[old] : 0;
                if (old >= 0) row_sum[i] = rowsum[old];
                for (int k = 0; k < length; ++k) {
                    int dest = pointers[s] + 4 * k + lane;
                    if (k < count) {
                        int source = row[old] + k;
                        values[dest] = val[source];
                        columns[dest] = inverse[col[source]];
                    } else {
                        values[dest] = 0;
                        columns[dest] = i < n ? i : 0;
                    }
                }
            }
        }
        valid = true;
        if (std::getenv("FAMEXP_DEBUG"))
            std::fprintf(stderr, "sliced kernel padding=%.5g\n",
                         static_cast<double>(values.size()) / val.size());
    }

    bool usable() const { return valid; }

    void apply(const Vec& b0, const Vec& b1, const Vec& ce, const Vec& cp,
               double upper, int degree, Vec& y) const {
        Vec p0(padded, 0), p1(padded, 0), u(padded, 0), d(padded, 0), next(padded);
        for (int i = 0; i < n; ++i) {
            p0[i] = b0[order[i]];
            p1[i] = b1[order[i]];
        }
        __m256d factor = _mm256_set1_pd(-4 / upper);
        for (int k = degree; k >= 1; --k) {
            __m256d ek = _mm256_set1_pd(ce[k]), pk = _mm256_set1_pd(cp[k]);
            for (int s = 0; s < padded / 4; ++s) {
                int i = 4 * s;
                __m256d di = _mm256_fmadd_pd(factor, product4(u.data(), s),
                                            _mm256_loadu_pd(d.data() + i));
                di = _mm256_fmadd_pd(ek, _mm256_loadu_pd(p0.data() + i), di);
                di = _mm256_fmadd_pd(pk, _mm256_loadu_pd(p1.data() + i), di);
                _mm256_storeu_pd(d.data() + i, di);
                _mm256_storeu_pd(next.data() + i, _mm256_add_pd(_mm256_loadu_pd(u.data() + i), di));
            }
            u.swap(next);
        }
        factor = _mm256_set1_pd(-2 / upper);
        __m256d e0 = _mm256_set1_pd(ce[0]), q0 = _mm256_set1_pd(cp[0]);
        for (int s = 0; s < padded / 4; ++s) {
            int i = 4 * s;
            __m256d yi = _mm256_fmadd_pd(factor, product4(u.data(), s),
                                        _mm256_loadu_pd(d.data() + i));
            yi = _mm256_fmadd_pd(e0, _mm256_loadu_pd(p0.data() + i), yi);
            yi = _mm256_fmadd_pd(q0, _mm256_loadu_pd(p1.data() + i), yi);
            _mm256_storeu_pd(next.data() + i, yi);
        }
        for (int i = 0; i < n; ++i) y[order[i]] = next[i];
    }
};
