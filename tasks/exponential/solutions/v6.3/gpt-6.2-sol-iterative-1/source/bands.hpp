#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <memory>
#include <utility>
#include <vector>
#include "memory.hpp"

struct Bands {
    static constexpr int cap = 13;
    int n = 0, count = 0, left = 0, right = 0, radius = 0;
    std::array<int, cap> offset{};
    memory::Array<double> weight;
    struct TailRow { int row, begin, end; };
    std::vector<TailRow> tail;
    std::vector<uint32_t> extra_col;
    std::vector<double> extra_a;

    bool prepare(int dim, const int* row, const uint32_t* col, const double* a) {
        if (dim < 2048 || row[dim] < 2*dim) return false;
        constexpr int sample = 128;
        int hits = 0;
        const int guard = std::max(256, dim / 24);
        std::vector<std::pair<int,int>> freq;
        for (int r = 0; r < sample; ++r) {
            const int i = (r * dim + ((73 * r) % 113) * (dim / 113)) / sample;
            for (int p = row[i]; p < row[i+1]; ++p) {
                ++hits;
                const int d = static_cast<int>(col[p]) - i;
                if (std::abs(d) <= guard) freq.emplace_back(d, 1);
            }
        }
        if (freq.empty() || hits > sample * (cap+5)) return false;
        std::sort(freq.begin(), freq.end());
        size_t out = 0;
        for (size_t p = 0; p < freq.size();) {
            const size_t first = p++;
            while (p < freq.size() && freq[p].first == freq[first].first) ++p;
            const auto appearances = static_cast<int>(p-first);
            if (appearances >= sample / 3) freq[out++] = {freq[first].first, appearances};
        }
        if (out < 2 || out > cap) return false;
        int cover = 0;
        for (size_t j = 0; j < out; ++j) cover += freq[j].second;
        if (cover < .96 * hits || cover < .65 * sample * out) return false;
        const int nb = static_cast<int>(out);
        n = dim & ~3;
        auto lookup = std::make_unique<int[]>(2 * guard + 1);
        std::fill_n(lookup.get(), 2 * guard + 1, -1);
        for (int j = 0; j < nb; ++j) {
            const int d = freq[j].first;
            offset[j] = d;
            left = std::max(left, -d);
            right = std::max(right, d);
            lookup[guard + d] = j;
        }
        weight = memory::zeros<double>(static_cast<size_t>(n) * nb);
        for (int i = 0; i < n; ++i) {
            const int begin = static_cast<int>(extra_col.size());
            for (int p = row[i]; p < row[i+1]; ++p) {
                const int d = static_cast<int>(col[p]) - i;
                const int j = std::abs(d) <= guard ? lookup[guard + d] : -1;
                if (j >= 0) {
                    weight[static_cast<size_t>(i & ~3)*nb + 4*j + (i & 3)] = a[p];
                } else {
                    extra_col.push_back(col[p]);
                    extra_a.push_back(a[p]);
                }
            }
            const int end = static_cast<int>(extra_col.size());
            if (end > begin) tail.push_back({i, begin, end});
            if (end > .06 * row[n]) {
                weight.reset();
                tail.clear();
                extra_col.clear();
                extra_a.clear();
                return false;
            }
        }
        count = nb;
        radius = std::max(left,right);
        for (const auto& p : tail)
            for (int j = p.begin; j < p.end; ++j)
                radius = std::max(radius, std::abs(static_cast<int>(extra_col[j])-p.row));
        for (int i = n; i < dim; ++i)
            for (int j = row[i]; j < row[i+1]; ++j)
                radius = std::max(radius,std::abs(static_cast<int>(col[j])-i));
        radius = (radius+3)&~3;
        return true;
    }

    template<int K, bool Diff, bool Two, bool Last>
    void step(const double* __restrict w, double* __restrict next, double* __restrict z,
              const double* __restrict rowsum, const double* __restrict x,
              const double* __restrict q, double alpha, double c0, double c1,
              double* __restrict y, int begin, int end) const {
        const int nb = K ? K : count;
        const double* wt = weight.get()+static_cast<size_t>(begin)*nb;
        std::array<const double*, cap> shift{};
        for (int j = 0; j < nb; ++j) shift[j] = w + offset[j];
        const __m256d r = _mm256_set1_pd(alpha);
        const __m256d u = _mm256_set1_pd(c0), v = _mm256_set1_pd(c1);
        for (int i = begin; i < end; i += 4, wt += 4*nb) {
            __m256d wi, p = _mm256_setzero_pd();
            __m256d s = p;
            if constexpr (Diff || !Last) wi = _mm256_loadu_pd(w+i);
            if constexpr (Diff) p = _mm256_mul_pd(_mm256_loadu_pd(rowsum+i), wi);
            if constexpr (K != 0) {
                #pragma GCC unroll 13
                for (int j = 0; j < K; ++j) {
                    __m256d xj = _mm256_loadu_pd(shift[j]+i);
                    if constexpr (Diff) xj = _mm256_sub_pd(xj, wi);
                    if (j & 1) s = _mm256_fmadd_pd(_mm256_loadu_pd(wt+4*j), xj, s);
                    else p = _mm256_fmadd_pd(_mm256_loadu_pd(wt+4*j), xj, p);
                }
            } else {
                for (int j = 0; j < nb; ++j) {
                    __m256d xj = _mm256_loadu_pd(shift[j]+i);
                    if constexpr (Diff) xj = _mm256_sub_pd(xj, wi);
                    p = _mm256_fmadd_pd(_mm256_loadu_pd(wt+4*j), xj, p);
                }
            }
            __m256d f = _mm256_fmadd_pd(u, _mm256_loadu_pd(x+i), _mm256_loadu_pd(z+i));
            if constexpr (Two) f = _mm256_fmadd_pd(v, _mm256_loadu_pd(q+i), f);
            const __m256d zi = _mm256_fmadd_pd(r, _mm256_add_pd(p,s), f);
            if constexpr (Last) _mm256_storeu_pd(y+i, zi);
            else {
                _mm256_storeu_pd(z+i, zi);
                _mm256_storeu_pd(next+i, _mm256_add_pd(wi, zi));
            }
        }
        auto it = tail.begin();
        if (begin && it != tail.end())
            it = std::lower_bound(it,tail.end(),begin,
                 [](const TailRow& p, int row) { return p.row < row; });
        for (; it != tail.end() && it->row < end; ++it) {
            const auto& ir = *it;
            double p = 0;
            const int i = ir.row;
            const double wi = Diff ? w[i] : 0;
            for (int j = ir.begin; j < ir.end; ++j)
                p = std::fma(extra_a[j], w[extra_col[j]] - wi, p);
            if constexpr (Last) y[i] = std::fma(alpha, p, y[i]);
            else {
                z[i] = std::fma(alpha, p, z[i]);
                next[i] = w[i] + z[i];
            }
        }
    }

    template<bool Diff, bool Two, bool Last>
    void apply(const double* w, double* next, double* z, const double* sums,
               const double* x, const double* q, double alpha, double c0, double c1,
               double* y, int begin=0, int end=-1) const {
        if (end < 0) end = n;
        #define BAND_STEP(k) case k: step<k, Diff, Two, Last>(w,next,z,sums,x,q,alpha,c0,c1,y,begin,end); break
        switch (count) {
            BAND_STEP(2);
            BAND_STEP(3);
            BAND_STEP(4);
            BAND_STEP(5);
            BAND_STEP(6);
            BAND_STEP(7);
            BAND_STEP(8);
            BAND_STEP(9);
            BAND_STEP(10);
            BAND_STEP(11);
            BAND_STEP(12);
            BAND_STEP(13);
            default: step<0,Diff,Two,Last>(w,next,z,sums,x,q,alpha,c0,c1,y,begin,end); break;
        }
        #undef BAND_STEP
    }
};
