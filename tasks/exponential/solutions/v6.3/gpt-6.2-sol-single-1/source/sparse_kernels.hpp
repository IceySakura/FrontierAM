#pragma once

#include <immintrin.h>
#include <numeric>

// Four contiguous rows at a time. For stencil-like matrices, the input
// vector loads have the same relative offsets in every group of four rows.
struct Diagonals {
    int n = 0, width = 0, pad = 0;
    std::vector<int> offsets;
    std::vector<unsigned char> good;
    std::vector<double> vals;

    template<class MatrixType>
    bool setup(const MatrixType& a) {
        n = a.n;
        if (n < 128 || a.val.size() < 2u*n || a.val.size() > 17u*n) return false;
        constexpr int cap = 12;
        std::vector<int> count(2*n-1);
        int ns = 0;
        // Vary the sampling position inside each small window to avoid
        // aliasing a regularly ordered mesh with the sampling stride.
        uint32_t seed = 301773;
        for (int base = 0; base + 16 <= n; base += 16) {
            seed = 1664525u * seed + 1013904223u;
            const int i = base + (seed >> 28);
            ++ns;
            for (int j = a.row[i]; j < a.row[i+1]; ++j)
                ++count[n-1 + a.col[j]-i];
        }
        for (int i = -(n-1); i <= n-1; ++i) {
            if (i == 0 || count[n-1+i]*10 > ns) offsets.push_back(i);
            if (int(offsets.size()) > cap) return false;
        }
        width = int(offsets.size());
        if (width < 2 || size_t(width)*n > a.val.size()*1.45) return false;
        const int nn = n / 4;
        int bad = 0;
        for (int i = 0; i < width; ++i) count[n-1+offsets[i]] = -i-1;
        good.assign(nn, 1);
        for (int g = 0; g < nn; ++g) {
            for (int i = 4*g; i < 4*g+4; ++i) {
                for (int j = a.row[i]; j < a.row[i+1]; ++j) {
                    if (count[n-1+a.col[j]-i] >= 0 && a.val[j] != 0) {
                        good[g] = 0;
                        break;
                    }
                }
                if (!good[g]) break;
            }
            bad += !good[g];
        }
        if (bad > 0.15 * nn) return false;
        pad = std::max(-offsets[0], offsets.back());
        vals.assign(4u*nn*width, 0);
        for (int i = 0; i < 4*nn; ++i) {
            if (!good[i/4]) continue;
            double* v = &vals[(i & ~3)*width + (i & 3)];
            for (int j = a.row[i]; j < a.row[i+1]; ++j) {
                const int k = -count[n-1+a.col[j]-i]-1;
                if (k >= 0) v[4*k] = a.val[j];
            }
        }
        return true;
    }

    template<int K, size_t... J>
    inline __m256d dot(const double* x, const double* a, const int* off,
                       std::index_sequence<J...>) const {
        __m256d sum[3] = {_mm256_setzero_pd(), _mm256_setzero_pd(), _mm256_setzero_pd()};
        ((sum[J%3] = _mm256_fmadd_pd(_mm256_loadu_pd(a+4*J),
                                    _mm256_loadu_pd(x+off[J]), sum[J%3])), ...);
        return _mm256_add_pd(_mm256_add_pd(sum[0], sum[1]), sum[2]);
    }
};

inline double max_lanes(__m256d v) {
    const __m128d r = _mm_max_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd(v, 1));
    return _mm_cvtsd_f64(_mm_max_sd(r, _mm_unpackhi_pd(r, r)));
}

struct SlicedRows {
    struct Group { int start, width; };
    int n4 = 0;
    bool small_indices = false, exact_float = false, reordered = false;
    std::vector<int> order;
    std::vector<Group> groups;
    std::vector<double> vals, scalar_vals;
    std::vector<float> vals32;
    std::vector<int> cols, scalar_cols;
    std::vector<uint16_t> cols16;
    std::vector<std::array<int, 5>> scalar_rows;

    template<class MatrixType>
    void improve_locality(const MatrixType& a) {
        if (a.n < 4096) return;
        const int cap = std::min(8192,a.n/4);
        auto distance = [&](const int* inv) {
            uint64_t sum = 0, edges = 0;
            const int step = std::max(1,a.n/711);
            for (int i = 0; i < a.n; i += step) {
                const int p = inv ? inv[i] : i;
                for (int j = a.row[i]; j < a.row[i+1]; ++j) {
                    if (a.col[j] == i || a.val[j] == 0) continue;
                    const int q = inv ? inv[a.col[j]] : a.col[j];
                    sum += std::min(cap,std::abs(p-q));
                    ++edges;
                }
            }
            return edges ? double(sum)/edges : 0.;
        };
        const double old = distance(nullptr);
        if (old < 0.25*cap) return;
        int seed = 0, min_degree = a.n+1;
        for (int i = 0; i < a.n; ++i) {
            const int d = a.row[i+1]-a.row[i];
            if (d >= 2 && d < min_degree) {
                seed = i;
                min_degree = d;
            }
        }
        std::vector<int> queue(a.n), inverse(a.n,-1);
        int tail = 0, cur = 0, next_seed = 0;
        auto push = [&](int i) { inverse[i] = tail; queue[tail++] = i; };
        push(seed);
        while (cur < a.n) {
            if (cur == tail) {
                while (inverse[next_seed] >= 0) ++next_seed;
                push(next_seed++);
            }
            const int i = queue[cur++];
            const int begin = tail;
            for (int j = a.row[i]; j < a.row[i+1]; ++j) {
                const int c = a.col[j];
                if (inverse[c] < 0 && a.val[j] != 0) push(c);
            }
            if (tail-begin <= 96) {
                std::sort(queue.begin()+begin,queue.begin()+tail,[&](int x,int y) {
                    return a.row[x+1]-a.row[x] < a.row[y+1]-a.row[y];
                });
            }
        }
        for (int i = 0; i < a.n; ++i) inverse[queue[i]] = i;
        if (distance(inverse.data()) < 0.57*old) {
            std::copy(queue.begin(),queue.end(),order.begin());
            reordered = true;
        }
    }

    template<class MatrixType>
    bool setup(const MatrixType& a, bool reorder) {
        if (a.n < 512 || a.val.size() < size_t(a.n)*2) return false;
        n4 = (a.n + 3) & ~3;
        small_indices = a.n <= 65536;
        exact_float = std::all_of(a.val.begin(), a.val.end(),
                                  [](double x){return double(float(x)) == x;});
        order.resize(n4);
        std::iota(order.begin(), order.end(), 0);
        if (reorder) improve_locality(a);
        const auto len = [&](int i) { return i < a.n ? a.row[i+1] - a.row[i] : 0; };
        constexpr int window = 128;
        for (int base = 0; base < n4; base += window) {
            std::sort(order.begin()+base, order.begin()+std::min(base+window, n4),
                      [&](int i, int j) { return len(i) < len(j); });
        }
        std::vector<int> inverse(a.n);
        for (int k = 0; k < n4; ++k)
            if (order[k] < a.n) inverse[order[k]] = k;
        // Store outlier rows without padding their three neighbours.
        groups.resize(n4/4);
        int total = 0, scalar_count = 0;
        for (int k = 0; k < n4; k += 4) {
            const int p0 = len(order[k]), p1 = len(order[k+1]);
            const int p2 = len(order[k+2]), p3 = len(order[k+3]);
            const int sum = p0+p1+p2+p3, w = 4*p3;
            if (w-sum > 12 && w > sum * 1.22) {
                groups[k/4] = {scalar_count++, -1};
            } else {
                groups[k/4] = {total, p3};
                total += w;
            }
        }
        if (total < int(a.val.size()*0.4)) return false;
        if (small_indices) cols16.resize(total);
        else cols.resize(total);
        if (exact_float) vals32.resize(total);
        else vals.resize(total);
        scalar_rows.resize(scalar_count);
        for (int k = 0; k < n4; k += 4) {
            const Group g = groups[k/4];
            if (g.width < 0) {
                auto& bounds = scalar_rows[g.start];
                for (int r = 0; r < 4; ++r) {
                    bounds[r] = int(scalar_vals.size());
                    const int i = order[k+r];
                    if (i >= a.n) continue;
                    for (int j = a.row[i]; j < a.row[i+1]; ++j) {
                        scalar_cols.push_back(inverse[a.col[j]]);
                        scalar_vals.push_back(a.val[j]);
                    }
                }
                bounds[4] = int(scalar_vals.size());
                continue;
            }
            for (int r = 0; r < 4; ++r) {
                const int i = order[k+r];
                if (i >= a.n) continue;
                for (int j = a.row[i], p = g.start + r; j < a.row[i+1]; ++j, p += 4) {
                    const int q = inverse[a.col[j]];
                    if (small_indices) cols16[p] = uint16_t(q);
                    else cols[p] = q;
                    if (exact_float) vals32[p] = float(a.val[j]);
                    else vals[p] = a.val[j];
                }
            }
        }
        return true;
    }

    template<bool Small>
    inline __m256d gather(const double* x, int j) const {
#ifdef FAM_HW_GATHER
        __m128i c;
        if constexpr (Small) {
            c = _mm_cvtepu16_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(cols16.data()+j)));
        } else {
            c = _mm_loadu_si128(reinterpret_cast<const __m128i*>(cols.data()+j));
        }
        return _mm256_i32gather_pd(x, c, 8);
#else
        const auto* c = [&]() {
            if constexpr (Small) return cols16.data()+j;
            else return cols.data()+j;
        }();
        const __m128d l = _mm_set_pd(x[c[1]], x[c[0]]);
        const __m128d h = _mm_set_pd(x[c[3]], x[c[2]]);
        return _mm256_insertf128_pd(_mm256_castpd128_pd256(l), h, 1);
#endif
    }

    template<bool Small, bool Float>
    inline __m256d dot(const double* x, Group g) const {
        auto madd = [&](int j, __m256d acc) {
            __m256d v;
            if constexpr (Float) v = _mm256_cvtps_pd(_mm_loadu_ps(vals32.data()+j));
            else v = _mm256_loadu_pd(vals.data()+j);
            return _mm256_fmadd_pd(v, gather<Small>(x,j), acc);
        };
        __m256d a = _mm256_setzero_pd(), b = a, c = a, d = a;
        int j = g.start;
        const int end = j + 4*g.width;
        for (; j + 12 < end; j+=16) {
            a = madd(j, a);
            b = madd(j+4, b);
            c = madd(j+8, c);
            d = madd(j+12, d);
        }
        if (j < end) {
            a = madd(j, a);
            if (j+4 < end) {
                b = madd(j+4, b);
                if (j+8 < end) c = madd(j+8, c);
            }
        }
        return _mm256_add_pd(_mm256_add_pd(a,b), _mm256_add_pd(c,d));
    }
};
