#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <memory>
#include <numeric>
#include <vector>
#include "memory.hpp"

// Four rows per SIMD group; degree sorting is local to each window.
struct Slices {
    int n = 0, npad = 0;
    double locality_score = 0;
    double uniform_weight = 0;
    bool narrow_index = false, hardware_gather = false;
    std::vector<int> order, width;
    memory::Array<double> coeff, pot;
    memory::Array<int> index;
    memory::Array<uint16_t> index16;
    struct Tail { int row, begin, end; };
    std::vector<Tail> extra;
    std::vector<int> extra_col;
    std::vector<double> extra_a;

    bool prepare(int dim, const int* row, const uint32_t* col, const double* a,
                 const double* potential, bool try_locality, double unit_weight = 0) {
        if (dim < 4000 || row[dim] < 2*dim) return false;
        n = dim;
        npad = (n+3)&~3;
        order.resize(npad);
        std::vector<int> base(npad);
        std::iota(base.begin(), base.end(), 0);
#ifndef FAM_NO_REORDER
        const auto locality = [&](const int* position) {
            constexpr int k = 512;
            int score = 0, entries = 0;
            for (int j = 0; j < k; ++j) {
                const int i = (j*n + ((331*j)%997)*(n/997))/k;
                const int ii = position ? position[i] : i;
                for (int p = row[i]; p < row[i+1]; ++p) {
                    const int jj = position ? position[col[p]] : static_cast<int>(col[p]);
                    const int d = std::abs(ii-jj);
                    score += (d < 6144) + (d < 1536);
                    entries++;
                }
            }
            return entries ? static_cast<double>(score)/entries : 0.;
        };
        if (n >= 8192) {
            const double natural = locality(nullptr);
            locality_score = natural;
            if (try_locality && natural < 1.68) {
                std::vector<int> position(n, -1);
                int end = 0;
                for (int root = 0; root < n; ++root) {
                    if (position[root] >= 0) continue;
                    int p = end;
                    position[root] = end;
                    base[end++] = root;
                    for (; p < end; ++p) {
                        const int i = base[p];
                        for (int j = row[i]; j < row[i+1]; ++j) {
                            const int id = static_cast<int>(col[j]);
                            if (position[id] < 0 && a[j] != 0) {
                                position[id] = end;
                                base[end++] = id;
                            }
                        }
                    }
                }
                const double moved = locality(position.data());
                if (moved <= natural + .16)
                    std::iota(base.begin(), base.end(), 0);
                else
                    locality_score = moved;
            }
            hardware_gather = n >= 37000 && locality_score < .85;
        }
#else
        (void)try_locality;
#endif
        constexpr int block = 128, max_width = 24;
        const auto degree = [&](int i) { return i < n ? row[i+1]-row[i] : 0; };
        for (int i = 0; i < npad; i += block) {
            const int end = std::min(npad, i+block);
            int by[max_width+2] = {};
            for (int j = i; j < end; ++j) ++by[std::min(max_width+1, degree(base[j]))];
            int ptr = i;
            for (auto& count : by) {
                const int first = ptr;
                ptr += count;
                count = first;
            }
            for (int j = i; j < end; ++j)
                order[by[std::min(max_width+1, degree(base[j]))]++] = base[j];
        }
        const auto rorder = std::unique_ptr<int[]>(new int[npad]);
        for (int i = 0; i < npad; ++i) rorder[order[i]] = i;
        width.reserve(npad/4);
        int padded = 0, tails = 0;
        for (int i = 0; i < npad; i += 4) {
            const int m = std::min({max_width, degree(order[i+3]), degree(order[i+1])+3});
            width.push_back(m);
            padded += 4*m;
            for (int k = 0; k < 4; ++k) tails += std::max(0, degree(order[i+k])-m);
        }
        if (padded > 1.18*row[n] || tails > .25*row[n]) {
            width.clear();
            order.clear();
            npad = 0;
            return false;
        }
        uniform_weight = unit_weight;
        narrow_index = npad <= 65536;
        if (!uniform_weight) coeff = memory::allocate<double>(padded);
        if (narrow_index) index16 = memory::allocate<uint16_t>(padded);
        else index = memory::allocate<int>(padded);
        if (potential) pot = memory::allocate<double>(npad);
        extra_a.reserve(tails);
        extra_col.reserve(tails);
        int off = 0;
        for (int i = 0; i < npad; i += 4) {
            const int m = width[i/4];
            for (int j = 0; j < 4; ++j) {
                const int k = order[i+j];
                const int d = degree(k);
                if (pot) pot[i+j] = k < n ? potential[k] : 0;
                const int begin = d ? row[k] : 0;
                for (int p = 0; p < m; ++p) {
                    const int entry = off+4*p+j;
                    const int id = p < d ? rorder[col[begin+p]] : i+j;
                    if (coeff) coeff[entry] = p < d ? a[begin+p] : 0;
                    if (narrow_index) index16[entry] = static_cast<uint16_t>(id);
                    else index[entry] = id;
                }
                if (d > m) {
                    const int start = static_cast<int>(extra_col.size());
                    for (int p = m; p < d; ++p) {
                        extra_col.push_back(rorder[col[begin+p]]);
                        extra_a.push_back(a[begin+p]);
                    }
                    extra.push_back({i+j, start, static_cast<int>(extra_col.size())});
                }
            }
            off += 4*m;
        }
        return true;
    }

    template<bool Hardware, class Index>
    inline static __m256d gather(const double* x, const Index* ids) {
        if constexpr (Hardware) {
            __m128i v;
            if constexpr (sizeof(Index)==2)
                v = _mm_cvtepu16_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(ids)));
            else
                v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ids));
            const __m256d mask = _mm256_castsi256_pd(_mm256_set1_epi64x(-1));
            return _mm256_mask_i32gather_pd(_mm256_setzero_pd(),x,v,mask,8);
        }
        return _mm256_set_pd(x[ids[3]], x[ids[2]], x[ids[1]], x[ids[0]]);
    }

    template<bool Diff, bool Two, bool Last, bool Unit, bool Rank, bool Hardware, class Index>
    void step(const Index* ids, const double* __restrict w, double* __restrict next,
              double* __restrict z, const double* __restrict x,
              const double* __restrict q, double alpha, double c0, double c1,
              double* __restrict y, const double* __restrict rank, double rank_coef) const {
        const __m256d a4 = _mm256_set1_pd(alpha);
        const __m256d u = _mm256_set1_pd(c0), v = _mm256_set1_pd(c1);
        const __m256d weight = _mm256_set1_pd(uniform_weight);
        const __m256d rankc = _mm256_set1_pd(rank_coef);
        const double* val = coeff.get();
        for (int i = 0; i < npad; i += 4) {
            __m256d wi, s = _mm256_setzero_pd(), p = s;
            if constexpr (Diff || !Last) wi = _mm256_loadu_pd(w+i);
            if constexpr (Diff && !Unit) p = _mm256_mul_pd(_mm256_loadu_pd(pot.get()+i), wi);
            const int count = width[i/4];
            int j = 0;
            for (; j+1 < count; j += 2) {
                __m256d w0 = gather<Hardware>(w, ids+4*j);
                __m256d w1 = gather<Hardware>(w, ids+4*j+4);
                if constexpr (Diff) {
                    w0 = _mm256_sub_pd(w0, wi);
                    w1 = _mm256_sub_pd(w1, wi);
                }
                if constexpr (Unit) {
                    p = _mm256_add_pd(p,w0);
                    s = _mm256_add_pd(s,w1);
                } else {
                    p = _mm256_fmadd_pd(_mm256_loadu_pd(val+4*j), w0, p);
                    s = _mm256_fmadd_pd(_mm256_loadu_pd(val+4*j+4), w1, s);
                }
            }
            if (j < count) {
                __m256d w0 = gather<Hardware>(w, ids+4*j);
                if constexpr (Diff) w0 = _mm256_sub_pd(w0, wi);
                if constexpr (Unit) p = _mm256_add_pd(p,w0);
                else p = _mm256_fmadd_pd(_mm256_loadu_pd(val+4*j), w0, p);
            }
            if constexpr (!Unit) val += 4*count;
            ids += 4*count;
            __m256d f = _mm256_fmadd_pd(u, _mm256_loadu_pd(x+i), _mm256_loadu_pd(z+i));
            if constexpr (Two) f = _mm256_fmadd_pd(v, _mm256_loadu_pd(q+i), f);
            if constexpr (Rank) f = _mm256_fmadd_pd(rankc,_mm256_loadu_pd(rank+i),f);
            __m256d prod = _mm256_add_pd(p,s);
            if constexpr (Unit)
                prod = _mm256_fmadd_pd(weight,prod,_mm256_mul_pd(_mm256_loadu_pd(pot.get()+i),wi));
            const __m256d zi = _mm256_fmadd_pd(a4, prod, f);
            if constexpr (Last) _mm256_storeu_pd(y+i, zi);
            else {
                _mm256_storeu_pd(z+i, zi);
                _mm256_storeu_pd(next+i, _mm256_add_pd(wi, zi));
            }
        }
        const double* tail_a = extra_a.data();
        const int* tail_col = extra_col.data();
        for (const auto& ir : extra) {
            double p = 0, q = 0, r = 0, s = 0;
            const int i = ir.row;
            const double wi = Diff ? w[i] : 0;
            int j = ir.begin;
            for (; j+3 < ir.end; j += 4) {
                p = std::fma(tail_a[j], w[tail_col[j]]-wi, p);
                q = std::fma(tail_a[j+1], w[tail_col[j+1]]-wi, q);
                r = std::fma(tail_a[j+2], w[tail_col[j+2]]-wi, r);
                s = std::fma(tail_a[j+3], w[tail_col[j+3]]-wi, s);
            }
            if (j < ir.end) p = std::fma(tail_a[j], w[tail_col[j]]-wi, p);
            if (j+1 < ir.end) q = std::fma(tail_a[j+1], w[tail_col[j+1]]-wi, q);
            if (j+2 < ir.end) r = std::fma(tail_a[j+2], w[tail_col[j+2]]-wi, r);
            p = (p+q)+(r+s);
            if constexpr (Last) y[i] = std::fma(alpha, p, y[i]);
            else {
                z[i] = std::fma(alpha, p, z[i]);
                next[i] = w[i] + z[i];
            }
        }
    }

    template<bool Diff, bool Two, bool Last, bool Rank, bool Hardware>
    void apply_selected(const double* w, double* next, double* z, const double* x,
                        const double* q, double alpha, double c0, double c1, double* y,
                        const double* rank, double rank_coef) const {
        if constexpr (Diff) {
            if (uniform_weight) {
                if (narrow_index) step<Diff,Two,Last,true,Rank,Hardware>(index16.get(),w,next,z,x,q,alpha,c0,c1,y,rank,rank_coef);
                else              step<Diff,Two,Last,true,Rank,Hardware>(index.get(),w,next,z,x,q,alpha,c0,c1,y,rank,rank_coef);
                return;
            }
        }
        if (narrow_index) step<Diff,Two,Last,false,Rank,Hardware>(index16.get(),w,next,z,x,q,alpha,c0,c1,y,rank,rank_coef);
        else              step<Diff,Two,Last,false,Rank,Hardware>(index.get(),w,next,z,x,q,alpha,c0,c1,y,rank,rank_coef);
    }

    template<bool Diff, bool Two, bool Last, bool Rank = false>
    void apply(const double* w, double* next, double* z, const double* x,
               const double* q, double alpha, double c0, double c1, double* y,
               const double* rank = nullptr, double rank_coef = 0) const {
        if (hardware_gather)
            apply_selected<Diff,Two,Last,Rank,true>(w,next,z,x,q,alpha,c0,c1,y,rank,rank_coef);
        else
            apply_selected<Diff,Two,Last,Rank,false>(w,next,z,x,q,alpha,c0,c1,y,rank,rank_coef);
    }
};
