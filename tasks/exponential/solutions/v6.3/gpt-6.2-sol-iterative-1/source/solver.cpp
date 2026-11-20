#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <unistd.h>
#include "bands.hpp"
#include "memory.hpp"
#include "slices.hpp"
#ifdef FAM_SMALL_BLOCKS
#include "small_blocks.hpp"
#endif
#ifdef FAM_ENABLE_STEADY
#include "steady.hpp"
#endif
#ifdef FAM_ENABLE_DEFLATE
#include "deflation.hpp"
#endif

static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
              "FAMEXP2 requires binary64 doubles");

extern "C" __m256d _ZGVdN4v_expm1(__m256d);

namespace {

bool read_bytes(void* data, size_t length) {
    auto* p = static_cast<char*>(data);
    while (length) {
        const ssize_t count = read(STDIN_FILENO, p, length);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        p += count;
        length -= static_cast<size_t>(count);
    }
    return true;
}

bool write_bytes(const void* data, size_t length) {
    auto* p = static_cast<const char*>(data);
    while (length) {
        const ssize_t count = write(STDOUT_FILENO, p, length);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        p += count;
        length -= static_cast<size_t>(count);
    }
    return true;
}

struct Header {
    int n = 0, nz = 0, requests = 0, version = 0;
    double t = 0, rtol = 0, atol = 0;

    static void whitespace(const char*& p) {
        while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') ++p;
    }

    static bool json_string(const char*& p, std::string& out) {
        if (*p != '"') return false;
        ++p;
        out.clear();
        while (*p && *p != '"') {
            if (*p != '\\') {
                out.push_back(*p++);
                continue;
            }
            ++p;
            if (*p == 0) return false;
            if (*p == 'u') {
                unsigned code = 0;
                ++p;
                for (int j = 0; j < 4; ++j) {
                    const char h = *p++;
                    unsigned v;
                    if ('0' <= h && h <= '9') v = h-'0';
                    else if ('a' <= h && h <= 'f') v = h-'a'+10;
                    else if ('A' <= h && h <= 'F') v = h-'A'+10;
                    else return false;
                    code = (code<<4) + v;
                }
                // All names and enum values in this schema are ASCII.
                out.push_back(code < 128 ? static_cast<char>(code) : '\x7f');
            } else {
                const char c = *p++;
                switch (c) {
                    case '"': case '\\': case '/': out.push_back(c); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    default: return false;
                }
            }
        }
        if (*p != '"') return false;
        ++p;
        return true;
    }

    bool read_header() {
        char first[12], json[4097];
        uint32_t length;
        if (!read_bytes(first,sizeof(first)) || std::memcmp(first,"FAMEXP2\n",8)) return false;
        std::memcpy(&length,first+8,sizeof(length));
        if (length == 0 || length > 4096 || !read_bytes(json,length)) return false;
        json[length] = 0;
        const char* p = json;
        whitespace(p);
        if (*p++ != '{') return false;
        std::string key, scenario;
        int fields = 0;
        for (;;) {
            whitespace(p);
            if (!json_string(p,key)) return false;
            whitespace(p);
            if (*p++ != ':') return false;
            whitespace(p);
            if (key == "scenario") {
                if (!json_string(p,scenario)) return false;
            } else {
                char* end = nullptr;
                if (key == "t" || key == "rtol" || key == "atol") {
                    const double d = std::strtod(p,&end);
                    if (key == "t") t = d;
                    else if (key == "rtol") rtol = d;
                    else atol = d;
                } else {
                    const long i = std::strtol(p,&end,10);
                    if (i < 0 || i > 1000000) return false;
                    if (key == "n") n = static_cast<int>(i);
                    else if (key == "nnz") nz = static_cast<int>(i);
                    else if (key == "requests") requests = static_cast<int>(i);
                    else if (key == "version") version = static_cast<int>(i);
                    else return false;
                }
                if (end == p) return false;
                p = end;
            }
            ++fields;
            whitespace(p);
            if (*p == '}') { ++p; break; }
            if (*p++ != ',') return false;
        }
        whitespace(p);
        return *p == 0 && fields == 8 && version == 2 && n > 0 && n <= 120000 &&
               ((requests == 1 && scenario == "one_shot") ||
                (requests == 16 && scenario == "reuse"));
    }
};

template<typename T> using Buffer = memory::Array<T>;
template<typename T> Buffer<T> storage(size_t n) { return memory::allocate<T>(n); }

template<typename T> inline T madd(T a, T b, T c) { return a * b + c; }
template<> inline double madd(double a, double b, double c) { return std::fma(a, b, c); }

struct RHSNorms {
    double x2, q2, xi, qi;
};

RHSNorms norms(int n, const double* x, const double* q, double t) {
    double a = 0, b = 0, c = 0, d = 0, mx = 0, mq = 0;
    int i = 0;
    for (; i + 1 < n; i += 2) {
        a = madd(x[i], x[i], a);
        b = madd(q[i], q[i], b);
        c = madd(x[i+1], x[i+1], c);
        d = madd(q[i+1], q[i+1], d);
        mx = std::max(mx, std::max(std::abs(x[i]), std::abs(x[i+1])));
        mq = std::max(mq, std::max(std::abs(q[i]), std::abs(q[i+1])));
    }
    if (i < n) {
        a = madd(x[i], x[i], a);
        b = madd(q[i], q[i], b);
        mx = std::max(mx, std::abs(x[i]));
        mq = std::max(mq, std::abs(q[i]));
    }
    return {std::sqrt(a+c), t * std::sqrt(b+d), mx, t * mq};
}

double max_abs(int n, const double* x) {
    const __m256d sign = _mm256_set1_pd(-0.0);
    const __m256d inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());
    __m256d v = _mm256_setzero_pd(), u = v, bad = v;
    int i = 0;
    for (; i + 7 < n; i += 8) {
        const auto a = _mm256_andnot_pd(sign,_mm256_loadu_pd(x+i));
        const auto b = _mm256_andnot_pd(sign,_mm256_loadu_pd(x+i+4));
        v = _mm256_max_pd(v,a);
        u = _mm256_max_pd(u,b);
        bad = _mm256_or_pd(bad,_mm256_or_pd(_mm256_cmp_pd(a,inf,_CMP_NLT_UQ),
                                           _mm256_cmp_pd(b,inf,_CMP_NLT_UQ)));
    }
    if (_mm256_movemask_pd(bad)) return std::numeric_limits<double>::infinity();
    alignas(32) double maxima[4];
    _mm256_store_pd(maxima,_mm256_max_pd(u,v));
    double a = std::max(std::max(maxima[0],maxima[1]),std::max(maxima[2],maxima[3]));
    for (; i < n; ++i) {
        if (!std::isfinite(x[i])) return std::numeric_limits<double>::infinity();
        a = std::max(a,std::abs(x[i]));
    }
    return a;
}

struct Series {
    std::vector<long double> expcoef, phicoef;
    std::vector<double> exptail, phitail;

    void prepare(long double eta) {
        // For H = I - 2*t*L/eta the exponential coefficients are the
        // probabilities of the absolute difference of two Poisson variates.
        // Compute scaled I_k by backward ratios, not by evaluating exp(eta/2).
        const long double s = eta * 0.5L;
        const int kmax = static_cast<int>(std::sqrt(160 * s)) + 141;
        expcoef.assign(kmax + 1, 0);
        phicoef.resize(kmax + 1);
        exptail.resize(kmax + 1);
        phitail.resize(kmax + 1);
        if (s == 0) {
            expcoef[0] = phicoef[0] = 1;
            return;
        }
        long double r = 0;
        for (int k = kmax; k >= 1; --k)
            expcoef[k] = r = s / (2 * k + s * r);
        long double z = 1;
        expcoef[0] = 1;
        long double c = 2;
        for (int k = 1; k <= kmax; ++k) {
            c *= expcoef[k];
            expcoef[k] = c;
            z += c;
            if (c < 1e-110L) {
                std::fill(expcoef.begin() + k + 1, expcoef.end(), 0);
                break;
            }
        }
        for (auto& ek : expcoef) ek /= z;

        // (1-T_k(H))/(1-H) = k + 2 sum_{j=1}^{k-1}(k-j)T_j(H).
        // This obtains the integral coefficients by positive summation.
        long double sum = 0, integral = 0, tail = 0;
        for (int k = kmax; k >= 0; --k) {
            expcoef[k] = std::max(0.L, expcoef[k]);
            exptail[k] = static_cast<double>(sum);
            phitail[k] = static_cast<double>(tail);
            phicoef[k] = (k ? 2 : 1) * integral / s;
            tail += phicoef[k];
            sum += expcoef[k];
            integral += sum;
        }
    }

    double error(int m, double nx, double nq) const {
        return exptail[m] * nx + phitail[m] * nq;
    }

    int degree(double goal, double nx, double nq) const {
        int l = 0, r = static_cast<int>(expcoef.size()) - 1;
        while (l < r) {
            const int m = (l + r) / 2;
            if (error(m, nx, nq) <= goal) r = m;
            else l = m + 1;
        }
        return l;
    }
};

struct Pair {
    int i, j;
    long double lo, hi, p, cross;
};

struct PairIndex {
    int i, j;
    double off;
};

struct WidePair {
    long double e00, e11, e01, f00, f11, f01;
};

template<class T>
inline void scalar_heat(T x, T t, T& e, T& f) {
    if (x > T(44)) {
        e = 0;
        f = t / x;
    } else if (x != 0) {
        const T w = -std::expm1(-x);
        e = T(1) - w;
        f = t * (w / x);
    } else {
        e = 1;
        f = t;
    }
}

template<class T>
WidePair pair_heat(const Pair& pair, double t) {
    const T p = static_cast<T>(pair.p), c = static_cast<T>(pair.cross);
    T el, eh, fl, fh;
    scalar_heat(static_cast<T>(pair.lo), static_cast<T>(t), el, fl);
    scalar_heat(static_cast<T>(pair.hi), static_cast<T>(t), eh, fh);
    return {madd(p, el-eh, eh), madd(-p, el-eh, el), c*(el-eh),
            madd(p, fl-fh, fh), madd(-p, fl-fh, fl), c*(fl-fh)};
}

struct Heat4 { __m256d e, f; };
struct PairHeat4 { Heat4 first, second, off; };

inline Heat4 heat4(__m256d x, __m256d time) {
    const __m256d zero = _mm256_setzero_pd(), one = _mm256_set1_pd(1);
    const __m256d nx = _mm256_sub_pd(zero,x);
    const __m256d w = _ZGVdN4v_expm1(nx);
    return {_mm256_add_pd(one,w), _mm256_blendv_pd(_mm256_mul_pd(time,_mm256_div_pd(w,nx)),
                                    time,_mm256_cmp_pd(x,zero,_CMP_EQ_OQ))};
}

inline PairHeat4 pair_heat4(__m256d a, __m256d d, __m256d b, __m256d time) {
    const __m256d zero = _mm256_setzero_pd(), one = _mm256_set1_pd(1);
    const __m256d half = _mm256_set1_pd(.5);
    const __m256d delta = _mm256_mul_pd(half,_mm256_sub_pd(a,d));
    const __m256d cp = _mm256_mul_pd(b,b);
    const __m256d gap = _mm256_sqrt_pd(_mm256_fmadd_pd(delta,delta,cp));
    const __m256d hi = _mm256_fmadd_pd(half,_mm256_add_pd(a,d),gap);
    const __m256d det = _mm256_sub_pd(_mm256_fmsub_pd(a,d,cp),_mm256_fmsub_pd(b,b,cp));
    const __m256d lo = _mm256_div_pd(det,_mm256_blendv_pd(hi,one,
                               _mm256_cmp_pd(hi,zero,_CMP_EQ_OQ)));
    const __m256d den = _mm256_blendv_pd(gap,one,_mm256_cmp_pd(gap,zero,_CMP_EQ_OQ));
    const __m256d proj = _mm256_mul_pd(half,_mm256_sub_pd(one,_mm256_div_pd(delta,den)));
    const __m256d cross = _mm256_div_pd(_mm256_mul_pd(_mm256_set1_pd(-.5),b),den);
    const Heat4 u = heat4(_mm256_mul_pd(time,lo),time);
    const Heat4 v = heat4(_mm256_mul_pd(time,hi),time);
    const __m256d de = _mm256_sub_pd(u.e,v.e), df = _mm256_sub_pd(u.f,v.f);
    return {{_mm256_fmadd_pd(proj,de,v.e),_mm256_fmadd_pd(proj,df,v.f)},
            {_mm256_fnmadd_pd(proj,de,u.e),_mm256_fnmadd_pd(proj,df,u.f)},
            {_mm256_mul_pd(cross,de),_mm256_mul_pd(cross,df)}};
}

struct Matrix {
    const Header h;
    Buffer<int> row;
    Buffer<uint32_t> col;
    Buffer<double> a, diag, rowsum, e, f;
    Buffer<double> ecross, fcross;
    Buffer<int> partner;
    std::vector<PairIndex> pairs;
    std::vector<WidePair> wide_pairs;
    double upper = 0, range = 0;
    double constant_loss = 0, last_rhs_scale = 0, last_y_scale = 0;
    bool small_blocks = false, differences = false, adjacent_pairs = true, packed_pairs = false;
    Series series;
    Series audit_series;
    Buffer<double> audit_y;
    bool try_interval_check = true;
    Bands bands;
    Slices slices;
#ifdef FAM_SMALL_BLOCKS
    SmallBlocks blocks;
#endif
#ifdef FAM_ENABLE_DEFLATE
    Deflation deflation;
    Series reduced_series;
    bool use_deflation = false;
#endif
#ifdef FAM_ENABLE_STEADY
    Steady steady;
    bool try_steady = true;
    double min_scaled_diag = 0, spectral_floor = 0;
#ifdef FAM_RECENTER
    Buffer<double> center, center_x, center_q;
    bool try_recenter = true;
#endif
#endif
    Buffer<double> w0, w1, dz;
    Buffer<double> ordx, ordq, ordy;
    Buffer<long double> wide0, wide1, widez;

    explicit Matrix(Header header) : h(header) {}

    bool read_matrix() {
        const auto nrows = static_cast<size_t>(h.n) + 1;
        if (h.t == 0) {
            char junk[64 * 1024];
            size_t skip = 8 * nrows + 12 * static_cast<size_t>(h.nz);
            while (skip) {
                const auto chunk = std::min(skip, sizeof(junk));
                if (!read_bytes(junk, chunk)) return false;
                skip -= chunk;
            }
            return true;
        }
        auto row64 = storage<uint64_t>(nrows);
        row = storage<int>(nrows);
        col = storage<uint32_t>(h.nz);
        a = storage<double>(h.nz);
        if (!read_bytes(row64.get(), nrows * sizeof(uint64_t)) ||
            !read_bytes(col.get(), h.nz * sizeof(uint32_t)) ||
            !read_bytes(a.get(), h.nz * sizeof(double))) return false;
        for (size_t i = 0; i < nrows; ++i) row[i] = static_cast<int>(row64[i]);
        return true;
    }

    Pair make_pair(int i, int j, double ai, double di, double off) const {
        const long double aa = ai, dd = di, cc = off;
        const long double delta = (aa - dd) * 0.5L;
        const long double gap = std::sqrt(delta*delta + cc*cc);
        const long double lhi = (aa + dd) * 0.5L + gap;
        const double cp = off*off;
        const long double det = cp >= std::numeric_limits<double>::min()
            ? static_cast<long double>(std::fma(ai, di, -cp)) - std::fma(off, off, -cp)
            : aa*dd-cc*cc;
        return {i, j, lhi == 0 ? 0 : h.t * (det/lhi), h.t * lhi,
                gap == 0 ? .5L : (1-delta/gap)*.5L, gap == 0 ? 0 : -.5L*cc/gap};
    }

    void assign_pair(const Pair& p) {
        const auto v = pair_heat<double>(p, h.t);
        const int i = p.i, j = p.j;
        e[i] = static_cast<double>(v.e00);
        e[j] = static_cast<double>(v.e11);
        f[i] = static_cast<double>(v.f00);
        f[j] = static_cast<double>(v.f11);
        ecross[i] = ecross[j] = static_cast<double>(v.e01);
        fcross[i] = fcross[j] = static_cast<double>(v.f01);
    }

    void prepare_diagonal(const double* d) {
        small_blocks = true;
        e = storage<double>(h.n);
        f = storage<double>(h.n);
        const __m256d time = _mm256_set1_pd(h.t);
        int i = 0;
        for (; i+3 < h.n; i += 4) {
            const Heat4 c = heat4(_mm256_mul_pd(time,_mm256_loadu_pd(d+i)),time);
            _mm256_storeu_pd(e.get()+i,c.e);
            _mm256_storeu_pd(f.get()+i,c.f);
        }
        for (; i < h.n; ++i) scalar_heat(h.t*d[i],h.t,e[i],f[i]);
    }

    bool prepare_full_diagonal() {
        if (h.n < 256 || h.nz != h.n) return false;
        for (int i = 0; i < h.n; ++i)
            if (row[i] != i || col[i] != static_cast<uint32_t>(i)) return false;
        upper = 1;
        prepare_diagonal(a.get());
        return true;
    }

    bool prepare_packed_pairs() {
        if (h.n < 512 || (h.n & 1) || h.nz != 2*h.n) return false;
        for (int i = 0; i < h.n; ++i) {
            const auto j = static_cast<uint32_t>(i & ~1);
            if (row[i] != 2*i || col[2*i] != j || col[2*i+1] != j+1) return false;
        }
        small_blocks = packed_pairs = true;
        upper = 1;  // The general operator bound is not needed in this branch.
        e = storage<double>(h.n);
        f = storage<double>(h.n);
        ecross = storage<double>(h.n);
        fcross = storage<double>(h.n);
        const double* const a0 = a.get();
        const __m256d time = _mm256_set1_pd(h.t);
        const auto interleave = [](__m256d a, __m256d b, double* out) {
            const auto c = _mm256_unpacklo_pd(a,b);
            const auto d = _mm256_unpackhi_pd(a,b);
            _mm256_storeu_pd(out,_mm256_permute2f128_pd(c,d,0x20));
            _mm256_storeu_pd(out+4,_mm256_permute2f128_pd(c,d,0x31));
        };
        int i = 0;
        for (; i+7 < h.n; i += 8) {
            const double* p = a0 + 2*i;
            const __m256d a = _mm256_set_pd(p[12],p[8],p[4],p[0]);
            const __m256d b = _mm256_set_pd(p[13],p[9],p[5],p[1]);
            const __m256d d = _mm256_set_pd(p[15],p[11],p[7],p[3]);
            const PairHeat4 c = pair_heat4(a,d,b,time);
            interleave(c.first.e,c.second.e,e.get()+i);
            interleave(c.first.f,c.second.f,f.get()+i);
            interleave(c.off.e,c.off.e,ecross.get()+i);
            interleave(c.off.f,c.off.f,fcross.get()+i);
        }
        for (; i < h.n; i += 2) assign_pair(make_pair(i,i+1,a[2*i],a[2*i+3],a[2*i+1]));
        return true;
    }

    void prepare_pairs(bool any_pairs) {
        if (!any_pairs) {
            prepare_diagonal(diag.get());
            return;
        }
        e = storage<double>(h.n);
        f = storage<double>(h.n);
        ecross = storage<double>(h.n);
        fcross = storage<double>(h.n);
        partner = storage<int>(h.n);
        pairs.reserve(h.n / 2);
        for (int i = 0; i < h.n; ++i) {
            int j = i;
            double off = 0;
            for (int k = row[i]; k < row[i+1]; ++k) {
                if (col[k] != static_cast<uint32_t>(i) && a[k] != 0) {
                    j = static_cast<int>(col[k]);
                    off = a[k];
                    break;
                }
            }
            partner[i] = j;
            if (i == j) {
                scalar_heat(h.t * diag[i], h.t, e[i], f[i]);
                ecross[i] = fcross[i] = 0;
            } else if (i < j) {
                pairs.push_back({i,j,off});
                adjacent_pairs = adjacent_pairs && j == (i ^ 1);
            }
        }
        size_t i = 0;
        const __m256d time = _mm256_set1_pd(h.t);
        for (; i+3 < pairs.size(); i += 4) {
            const PairIndex* p = pairs.data()+i;
            const __m256d u = _mm256_set_pd(diag[p[3].i],diag[p[2].i],diag[p[1].i],diag[p[0].i]);
            const __m256d v = _mm256_set_pd(diag[p[3].j],diag[p[2].j],diag[p[1].j],diag[p[0].j]);
            const __m256d w = _mm256_set_pd(p[3].off,p[2].off,p[1].off,p[0].off);
            const PairHeat4 c = pair_heat4(u,v,w,time);
            alignas(32) double E0[4], E1[4], F0[4], F1[4], EC[4], FC[4];
            _mm256_store_pd(E0,c.first.e); _mm256_store_pd(E1,c.second.e);
            _mm256_store_pd(F0,c.first.f); _mm256_store_pd(F1,c.second.f);
            _mm256_store_pd(EC,c.off.e); _mm256_store_pd(FC,c.off.f);
            for (int j = 0; j < 4; ++j) {
                const int a = p[j].i, b = p[j].j;
                e[a] = E0[j]; e[b] = E1[j]; f[a] = F0[j]; f[b] = F1[j];
                ecross[a] = ecross[b] = EC[j];
                fcross[a] = fcross[b] = FC[j];
            }
        }
        for (; i < pairs.size(); ++i) {
            const PairIndex& p = pairs[i];
            assign_pair(make_pair(p.i,p.j,diag[p.i],diag[p.j],p.off));
        }
    }

    void solve_pairs(const double* __restrict x, const double* __restrict q,
                     double* __restrict y) {
        if (pairs.empty() && !packed_pairs) {
            for (int i = 0; i < h.n; ++i) y[i] = madd(f[i], q[i], e[i] * x[i]);
            return;
        }
        int i = 0;
        if (adjacent_pairs) {
            for (; i + 3 < h.n; i += 4) {
                const __m256d u = _mm256_loadu_pd(x+i), v = _mm256_loadu_pd(q+i);
                __m256d w = _mm256_mul_pd(u, _mm256_loadu_pd(e.get()+i));
                w = _mm256_fmadd_pd(v, _mm256_loadu_pd(f.get()+i), w);
                w = _mm256_fmadd_pd(_mm256_permute_pd(u, 5), _mm256_loadu_pd(ecross.get()+i), w);
                w = _mm256_fmadd_pd(_mm256_permute_pd(v, 5), _mm256_loadu_pd(fcross.get()+i), w);
                _mm256_storeu_pd(y+i, w);
            }
        }
        for (; i < h.n; ++i) {
            const int j = packed_pairs ? i^1 : partner[i];
            y[i] = madd(fcross[i], q[j], madd(ecross[i], x[j], madd(f[i], q[i], e[i]*x[i])));
        }
        if (h.rtol > 1e-10 || 1.5e-15*(1+h.t) <= .22*h.atol) return;
        const double budget = .22*(h.atol+h.rtol*max_abs(h.n,y));
        if (1.5e-15*(1+h.t) <= budget ||
            1.5e-15*(max_abs(h.n,x)+h.t*max_abs(h.n,q)) <= budget) return;

        if (wide_pairs.empty()) {
            if (packed_pairs && pairs.empty()) {
                pairs.reserve(h.n/2);
                for (int i = 0; i < h.n; i += 2)
                    pairs.push_back({i,i+1,a[2*i+1]});
            }
            wide_pairs.reserve(pairs.size());
            for (const PairIndex& p : pairs) {
                const double aa = packed_pairs ? a[2*p.i] : diag[p.i];
                const double dd = packed_pairs ? a[2*p.i+3] : diag[p.j];
                wide_pairs.push_back(pair_heat<long double>(make_pair(p.i,p.j,aa,dd,p.off),h.t));
            }
        }
        for (size_t j = 0; j < pairs.size(); ++j) {
            const int r = pairs[j].i, s = pairs[j].j;
            const auto& p = wide_pairs[j];
            y[r] = static_cast<double>((p.e00*x[r] + p.e01*x[s]) + (p.f00*q[r] + p.f01*q[s]));
            y[s] = static_cast<double>((p.e01*x[r] + p.e11*x[s]) + (p.f01*q[r] + p.f11*q[s]));
        }
    }

    void prepare() {
        if (h.t == 0 || h.nz == 0) return;
        if (prepare_packed_pairs()) return;
        if (prepare_full_diagonal()) return;
        diag = storage<double>(h.n);
        bool nonpos_off = true;
        int max_degree = 0;
        [[maybe_unused]] int critical = 0;
        double maxdiag = 0;
#ifdef FAM_ENABLE_STEADY
        min_scaled_diag = spectral_floor = std::numeric_limits<double>::infinity();
#endif
        for (int i = 0; i < h.n; ++i) {
            double s = 0, d = 0;
            int degree = 0;
            for (int k = row[i]; k < row[i+1]; ++k) {
                const double v = a[k];
                s += std::abs(v);
                if (col[k] == static_cast<uint32_t>(i)) d = v;
                else if (v != 0) {
                    ++degree;
                    if (v > 0) nonpos_off = false;
                }
            }
            max_degree = std::max(max_degree, degree);
            if (s > upper) {
                upper = s;
                critical = i;
            }
            maxdiag = std::max(maxdiag, d);
            diag[i] = d;
#ifdef FAM_ENABLE_STEADY
            min_scaled_diag = std::min(min_scaled_diag,h.t*d);
            spectral_floor = std::min(spectral_floor,h.t*(2*d-s-4e-11*s));
#endif
        }
        if (upper == 0) return;
        small_blocks = max_degree <= 1;
        if (small_blocks) {
            prepare_pairs(max_degree != 0);
            return;
        }
#ifdef FAM_SMALL_BLOCKS
        if (!blocks.rejected && max_degree < SmallBlocks::max_size &&
            h.t*upper > (h.n <= SmallBlocks::max_size ? 2 : (h.requests == 1 ? 190 : 40))) {
            if (blocks.prepare(h.n,critical,row.get(),col.get(),a.get(),h.t)) return;
        }
#endif

        if (h.t * upper > 12 && maxdiag > 0) {
            auto weight = storage<double>(h.n);
            const double floor = 0.09 * maxdiag;
            for (int i = 0; i < h.n; ++i) weight[i] = diag[i] + floor;
            double bound = 0;
            for (int i = 0; i < h.n; ++i) {
                double s = 0;
                for (int k = row[i]; k < row[i+1]; ++k)
                    s = madd(std::abs(a[k]), weight[col[k]], s);
                bound = std::max(bound, s / weight[i]);
            }
            if (bound > 0) upper = std::min(upper, bound);
        }
        upper *= 1.00000000004;
        if (upper * h.t > 20) upper *= 1.002;
        range = upper * h.t;
        differences = nonpos_off && range > 16;
        double uniform = 0;
        if (differences) {
            // Sum the stored row, including its true (possibly tiny) potential.
            // Removing the diagonal from the product then preserves constants
            // without treating a rounded wire Laplacian as an exact one.
            rowsum = storage<double>(h.n);
            bool identical = true;
            double minsum = 0, maxsum = 0;
            int dest = 0, begin = row[0];
            for (int i = 0; i < h.n; ++i) {
                const int end = row[i+1];
                long double s = 0, lo = 0;
                row[i] = dest;
                for (int k = begin; k < end; ++k) {
                    const double v = a[k];
                    const long double p = s + v;
                    lo += std::abs(s) >= std::abs(v) ? (s - p) + v : (v - p) + s;
                    s = p;
                    if (col[k] != static_cast<uint32_t>(i) && v != 0) {
                        if (identical) {
                            if (!uniform) uniform = v;
                            else identical = v == uniform;
                        }
                        a[dest] = v;
                        col[dest++] = col[k];
                    }
                }
                rowsum[i] = static_cast<double>(s + lo);
                minsum = std::min(minsum, rowsum[i]);
                maxsum = std::max(maxsum, std::abs(rowsum[i]));
                begin = end;
            }
            row[h.n] = dest;
            if (!identical) uniform = 0;
            constant_loss = h.t * maxsum * std::exp(std::min(600., -h.t*minsum));
        }
#ifndef FAM_NO_BANDS
        if (range > (h.requests == 1 ? 6 : 1.5))
            bands.prepare(h.n, row.get(), col.get(), a.get());
#endif
#ifndef FAM_NO_SLICES
        if (bands.count == 0 && range > (h.requests == 1 ? 35 : 12))
            slices.prepare(h.n, row.get(), col.get(), a.get(), rowsum.get(),
                           range > (h.requests == 1 ? 300 : 50), uniform);
#endif
#ifdef FAM_ENABLE_DEFLATE
        if (!bands.count && h.rtol >= 1e-7) {
            deflation.prepare({h.n,row.get(),col.get(),a.get(),diag.get(),differences},
                              upper,h.t,h.rtol,slices.order.data(),slices.npad);
            if (deflation.vec) reduced_series.prepare(static_cast<long double>(deflation.top)*h.t);
#ifdef FAM_TRACE
            if (deflation.vec)
                std::fprintf(stderr,"deflation top=%.6g ratio=%.5g loss=%.5g\n", upper,
                             deflation.top/upper,deflation.loss);
#endif
        }
#endif
        series.prepare(static_cast<long double>(upper) * h.t);
    }

    template<class T, bool Diff>
    inline T product(int i, const T* __restrict w) const {
        const T wi = Diff ? w[i] : T(0);
        const int end = row[i+1];
        const auto* __restrict ia = col.get();
        const auto* __restrict va = a.get();
        int k = row[i];
        T p = Diff ? static_cast<T>(rowsum[i]) * wi : T(0);
        T q = 0, r = 0, s = 0;
        for (; k + 3 < end; k += 4) {
            p = madd(static_cast<T>(va[k]), w[ia[k]] - wi, p);
            q = madd(static_cast<T>(va[k+1]), w[ia[k+1]] - wi, q);
            r = madd(static_cast<T>(va[k+2]), w[ia[k+2]] - wi, r);
            s = madd(static_cast<T>(va[k+3]), w[ia[k+3]] - wi, s);
        }
        if (k < end) p = madd(static_cast<T>(va[k]), w[ia[k]] - wi, p);
        if (k+1 < end) q = madd(static_cast<T>(va[k+1]), w[ia[k+1]] - wi, q);
        if (k+2 < end) r = madd(static_cast<T>(va[k+2]), w[ia[k+2]] - wi, r);
        return (p+q) + (r+s);
    }

    template<class T, bool Diff, bool Two, int Format = 0>
    void clenshaw(int m, const double* __restrict x, const double* __restrict q,
                  bool exp_active, double* __restrict y,
                  T* __restrict w, T* __restrict next, T* __restrict z) const {
        const int n = Format == 2 ? slices.npad : h.n;
#ifdef FAM_ENABLE_DEFLATE
        const double* rank = nullptr;
        double shift = 0;
        if constexpr (std::is_same_v<T,double> && Format != 1) {
            if (use_deflation) {
                rank = Format == 2 ? deflation.packed.get() : deflation.vec.get();
                shift = -deflation.gamma;
            }
        }
#endif
        const auto c0 = [&](int k) -> T {
            return static_cast<T>(exp_active ? series.expcoef[k] : h.t * series.phicoef[k]);
        };
        const auto c1 = [&](int k) -> T {
            return static_cast<T>(h.t * series.phicoef[k]);
        };
        T u = c0(m), v = Two ? c1(m) : 0;
        if (m == 0) {
            for (int i = 0; i < n; ++i) {
                T f = u * x[i];
                if constexpr (Two) f = madd(v, static_cast<T>(q[i]), f);
                y[i] = static_cast<double>(f);
            }
            return;
        }
        // Reinsch's form of Clenshaw at the soft end of the spectrum:
        // z_k = z_{k+1} + 2(H-I) w_{k+1} + c_k, w_k = w_{k+1} + z_k.
        for (int i = 0; i < n; ++i) {
            T f = u * x[i];
            if constexpr (Two) f = madd(v, static_cast<T>(q[i]), f);
            w[i] = z[i] = f;
        }
        const T a4 = -T(4) / static_cast<T>(upper);
        int k = m-1;
#ifdef FAM_TEMPORAL_BANDS
        if constexpr (Format == 1) {
            const int radius = bands.radius;
            if (h.n >= 12000 && radius <= 704 && m >= 12) {
                const int depth = radius <= 344 ? 8 : 4;
                const int tile = radius <= 344 ? 2048 : 4096;
                // A newer step may overwrite rows in the alternating buffer
                // only after the older step's read frontier has passed them.
                while (k >= depth) {
                    int done[8] = {};
                    double cv[8], cx[8];
                    for (int j = 0; j < depth; ++j) {
                        cx[j] = c0(k-j);
                        cv[j] = Two ? c1(k-j) : 0;
                    }
                    for (int p = 0; p < bands.n;) {
                        p = std::min(p+tile,bands.n);
                        const bool last = p == bands.n;
                        for (int j = 0; j < depth; ++j) {
                            const int end = last ? p : p-j*radius;
                            if (end < done[j]) break;
                            const double* src = (j&1) ? next : w;
                            double* dst = (j&1) ? w : next;
                            if (end > done[j]) {
                                bands.apply<Diff,Two,false>(src,dst,z,rowsum.get(),x,q,a4,
                                                            cx[j],cv[j],y,done[j],end);
                                done[j] = end;
                            }
                            if (last) {
                                for (int i = bands.n; i < h.n; ++i) {
                                    const double p = product<double,Diff>(i,src);
                                    double f = madd(cx[j],x[i],z[i]);
                                    if constexpr (Two) f = madd(cv[j],q[i],f);
                                    const double zi = madd(a4,p,f);
                                    dst[i] = src[i]+zi;
                                    z[i] = zi;
                                }
                            }
                        }
                    }
                    k -= depth;
                }
            }
        }
#endif
        for (; k >= 1; --k) {
            u = c0(k);
            if constexpr (Two) v = c1(k);
            int first = 0;
#ifdef FAM_ENABLE_DEFLATE
            double ranked = 0;
            if constexpr (std::is_same_v<T,double>)
                if (rank) ranked = a4*shift*Deflation::dot(n,rank,w);
#endif
            if constexpr (Format == 1) {
                bands.apply<Diff,Two,false>(w, next, z, rowsum.get(), x, q, a4, u, v, y);
                first = bands.n;
            } else if constexpr (Format == 2) {
#ifdef FAM_ENABLE_DEFLATE
                if (rank) slices.apply<Diff,Two,false,true>(w, next, z, x, q, a4, u, v, y,rank,ranked);
                else
#endif
                    slices.apply<Diff,Two,false>(w, next, z, x, q, a4, u, v, y);
                first = n;
            }
            for (int i = first; i < n; ++i) {
                const T p = product<T, Diff>(i, w);
                T f = madd(u, static_cast<T>(x[i]), z[i]);
                if constexpr (Two) f = madd(v, static_cast<T>(q[i]), f);
#ifdef FAM_ENABLE_DEFLATE
                if constexpr (std::is_same_v<T,double>)
                    if (rank) f = madd(ranked,rank[i],f);
#endif
                const T zi = madd(a4, p, f);
                next[i] = w[i] + zi;
                z[i] = zi;
            }
            std::swap(w, next);
        }
        u = c0(0);
        if constexpr (Two) v = c1(0);
        int first = 0;
#ifdef FAM_ENABLE_DEFLATE
        double ranked = 0;
        if constexpr (std::is_same_v<T,double>)
            if (rank) ranked = .5*a4*shift*Deflation::dot(n,rank,w);
#endif
        if constexpr (Format == 1) {
            bands.apply<Diff,Two,true>(w, next, z, rowsum.get(), x, q, a4 * 0.5, u, v, y);
            first = bands.n;
        } else if constexpr (Format == 2) {
#ifdef FAM_ENABLE_DEFLATE
            if (rank) slices.apply<Diff,Two,true,true>(w, next, z, x, q, a4 * 0.5, u, v, y,rank,ranked);
            else
#endif
                slices.apply<Diff,Two,true>(w, next, z, x, q, a4 * 0.5, u, v, y);
            first = n;
        }
        for (int i = first; i < n; ++i) {
            const T p = product<T, Diff>(i, w);
            T f = madd(u, static_cast<T>(x[i]), z[i]);
            if constexpr (Two) f = madd(v, static_cast<T>(q[i]), f);
#ifdef FAM_ENABLE_DEFLATE
            if constexpr (std::is_same_v<T,double>)
                if (rank) f = madd(ranked,rank[i],f);
#endif
            y[i] = static_cast<double>(madd(a4 * T(0.5), p, f));
        }
    }

    template<class T>
    void evaluate(int m, const double* x, const double* q, double* y,
                  const RHSNorms& size, Buffer<T>& a0, Buffer<T>& a1, Buffer<T>& d) {
        constexpr bool is_double = std::is_same_v<T,double>;
        const bool use_bands = is_double && bands.count;
        const int left = use_bands ? (bands.left+7)&~7 : 0;
        const int right = use_bands ? bands.right : 0;
        const int work = is_double && slices.npad ? slices.npad : h.n;
        if (!a0 && m > 0) {
            a0 = storage<T>(work + left + right);
            a1 = storage<T>(work + left + right);
            d = storage<T>(work);
            if (use_bands) {
                std::fill_n(a0.get(), left, T(0));
                std::fill_n(a1.get(), left, T(0));
                std::fill_n(a0.get() + h.n + left, right, T(0));
                std::fill_n(a1.get() + h.n + left, right, T(0));
            }
        }
        const bool hasx = size.xi != 0, hasq = size.qi != 0;
        T* const w = a0 ? a0.get() + left : nullptr;
        T* const next = a1 ? a1.get() + left : nullptr;
        T* const z = d.get();
        if constexpr (is_double) {
            if (slices.npad && m >= 1) {
                if (!ordx) {
                    ordx = storage<double>(work);
                    ordq = storage<double>(work);
                    ordy = storage<double>(work);
                }
                for (int i = 0; i < work; ++i) {
                    const int j = slices.order[i];
                    if (hasx) ordx[i] = j < h.n ? x[j] : 0;
                    if (hasq) ordq[i] = j < h.n ? q[j] : 0;
                }
                auto* const dst = ordy.get();
                if (hasx && hasq) {
                    if (differences) clenshaw<T,true,true,2>(m, ordx.get(), ordq.get(), true, dst, w, next, z);
                    else             clenshaw<T,false,true,2>(m, ordx.get(), ordq.get(), true, dst, w, next, z);
                } else {
                    const double* src = hasx ? ordx.get() : ordq.get();
                    if (differences) clenshaw<T,true,false,2>(m, src, nullptr, hasx, dst, w, next, z);
                    else             clenshaw<T,false,false,2>(m, src, nullptr, hasx, dst, w, next, z);
                }
                for (int i = 0; i < work; ++i) {
                    const int j = slices.order[i];
                    if (j < h.n) y[j] = dst[i];
                }
                return;
            }
            if (use_bands) {
                if (hasx && hasq) {
                    if (differences) clenshaw<T,true,true,1>(m, x, q, true, y, w, next, z);
                    else             clenshaw<T,false,true,1>(m, x, q, true, y, w, next, z);
                } else {
                    const double* src = hasx ? x : q;
                    if (differences) clenshaw<T,true,false,1>(m, src, nullptr, hasx, y, w, next, z);
                    else             clenshaw<T,false,false,1>(m, src, nullptr, hasx, y, w, next, z);
                }
                return;
            }
        }
        if (hasx && hasq) {
            if (differences) clenshaw<T,true,true>(m, x, q, true, y, w, next, z);
            else             clenshaw<T,false,true>(m, x, q, true, y, w, next, z);
        } else {
            const double* src = hasx ? x : q;
            if (differences) clenshaw<T,true,false>(m, src, nullptr, hasx, y, w, next, z);
            else             clenshaw<T,false,false>(m, src, nullptr, hasx, y, w, next, z);
        }
    }

    bool check_interval(const double* x, const double* q, double* y,
                        const RHSNorms& size, double tol) {
        // An independently scaled interval changes both the coefficients and
        // the intermediate states. Agreement avoids many unnecessary wide
        // reruns without relaxing either truncation budget.
        constexpr double inflate = 1.09;
        const double audit_top = upper * inflate;
        if (!audit_y) {
            audit_y = storage<double>(h.n);
            audit_series.prepare(static_cast<long double>(audit_top)*h.t);
        }
        const int m = audit_series.degree(.13*tol,size.x2,size.q2);
        const double saved = upper;
        upper = audit_top;
        std::swap(series, audit_series);
        evaluate(m,x,q,audit_y.get(),size,w0,w1,dz);
        std::swap(series, audit_series);
        upper = saved;
        if (!std::isfinite(max_abs(h.n,audit_y.get()))) return false;
        double difference = 0;
        for (int i = 0; i < h.n; ++i)
            difference = std::max(difference, std::abs(y[i]-audit_y[i]));
#ifdef FAM_TRACE
        std::fprintf(stderr,"interval m=%d difference/tol=%.3g\n",m,difference/tol);
#endif
        if (!(difference <= .14*tol)) return false;
        for (int i = 0; i < h.n; ++i) y[i] = 0.5*(y[i]+audit_y[i]);
        return true;
    }

    bool small_taylor(int cap, double tolerance, const double* __restrict x,
                      const double* __restrict q, double* __restrict y,
                      const RHSNorms& size) {
        if (range > .28 || cap <= 0) return false;
        const int limit = std::min(cap-1,9);
        double w = range, err = 0;
        int m = 0;
        for (;; ++m) {
            const double den = 1-range/(m+2);
            err = (w/den) * (size.xi+size.qi/(m+2));
            if (err <= .33*tolerance) break;
            if (m >= limit) return false;
            w *= range/(m+2);
        }
        if (m == 0) {
            for (int i = 0; i < h.n; ++i) y[i] = madd(h.t,q[i],x[i]);
        } else {
            if (!w0) {
                w0 = storage<double>(h.n);
                w1 = storage<double>(h.n);
                dz = storage<double>(h.n);
            }
            double factorial[10] = {1};
            for (int k = 1; k <= m; ++k) factorial[k] = factorial[k-1]/k;
            double* prev = w0.get();
            double* next = w1.get();
            double d = h.t * factorial[m]/(m+1);
            for (int i = 0; i < h.n; ++i) prev[i] = madd(d,q[i],factorial[m]*x[i]);
            for (int k = m-1; k >= 0; --k) {
                const double c = factorial[k];
                d = h.t*c/(k+1);
                double* dst = k==0 ? y : next;
                for (int i = 0; i < h.n; ++i)
                    dst[i] = madd(-h.t,product<double,false>(i,prev),madd(d,q[i],c*x[i]));
                std::swap(prev,next);
            }
        }
        const double out = max_abs(h.n,y);
        return std::isfinite(out) && err <= .45*(h.atol + h.rtol*out);
    }

#if defined(FAM_ENABLE_STEADY) && defined(FAM_RECENTER)
    bool recenter(const double* x, const double* q, double* y,
                  const RHSNorms& size, int degree, double estimate, double& actual) {
        if (!try_recenter || h.rtol > 1e-10 || degree < 300 || differences || bands.count ||
            size.qi < 2*(size.xi+.001)) return false;
        const double initial_tol = h.atol+h.rtol*estimate;
        const double u = differences ? 2e-19 : 3e-18;
        const double initial_risk = u*(degree+1)*(degree+1)*(size.xi+size.qi);
        if (initial_risk < 6*initial_tol) return false;
        if (!center) {
            center = storage<double>(h.n);
            center_x = storage<double>(h.n);
            center_q = storage<double>(h.n);
        }
        const Steady::Data d{h.n, row.get(), col.get(), a.get(), diag.get(),
                             rowsum.get(), h.t, h.rtol, h.atol, 4.};
        steady.approximate(d,q,center.get(),size.q2,std::min(42,std::max(10,degree/32)));
        const auto res = Steady::accurate_residual(d,q,center.get(),center_q.get());
        for (int i = 0; i < h.n; ++i) center_x[i] = x[i]-center[i];
        const auto z = norms(h.n,center_x.get(),center_q.get(),h.t);
        const double transform = res.error + 3e-16*z.x2;
        const double trial_tol = h.atol+h.rtol*std::max(estimate,.55*max_abs(h.n,center.get()));
        const int m = series.degree(.19*trial_tol,z.x2,z.q2);
        const double risk = u*(m+1)*(m+1)*(z.xi+z.qi);
#ifdef FAM_TRACE
        std::fprintf(stderr,"center risk/tol=%.3g orig=%.3g transform/tol=%.3g from m=%d to %d\n",
                     risk/trial_tol,initial_risk/initial_tol,transform/trial_tol,degree,m);
#endif
        if (!(transform <= .085*trial_tol && risk <= 2.2*trial_tol &&
                 z.xi+z.qi < .22*(size.xi+size.qi))) {
            try_recenter = false;
            return false;
        }
        evaluate(m,center_x.get(),center_q.get(),y,z,w0,w1,dz);
        for (int i = 0; i < h.n; ++i) y[i] += center[i];
        actual = max_abs(h.n,y);
        const double tol = h.atol+h.rtol*actual;
        if (std::isfinite(actual) &&
            series.error(m,z.x2,z.q2)+transform <= .39*tol && risk <= 1.55*tol)
            return true;
        try_recenter = false;
        return false;
    }
#endif

    void solve(const double* __restrict x, const double* __restrict q,
               double* __restrict y) {
        if (h.t == 0) {
            std::memcpy(y, x, h.n * sizeof(double));
            return;
        }
        if (upper == 0) {
            for (int i = 0; i < h.n; ++i) y[i] = madd(h.t, q[i], x[i]);
            return;
        }
        if (small_blocks) {
            solve_pairs(x, q, y);
            return;
        }
#ifdef FAM_SMALL_BLOCKS
        if (blocks.ready) {
            const double err = blocks.evaluate(x,q,y,h.t);
            const double out = max_abs(h.n,y);
            if (std::isfinite(out) && err <= .37*(h.atol+h.rtol*out)) return;
            blocks.ready = false;
            blocks.rejected = true;
            upper = 0;
            prepare();
        }
#endif
        if (differences && x[0] == x[h.n-1] && q[0] == q[h.n-1] &&
            x[0] == x[h.n/2] && q[0] == q[h.n/2]) {
            const double value = madd(h.t,q[0],x[0]);
            if (constant_loss*(std::abs(x[0])+.5*h.t*std::abs(q[0]))
                    <= .20 * (h.atol+h.rtol*std::abs(value))) {
                int i = 1;
                for (; i < h.n; ++i)
                    if (x[i] != x[0] || q[i] != q[0]) break;
                if (i == h.n) {
                    std::fill_n(y,h.n,value);
                    return;
                }
            }
        }
        const RHSNorms size = norms(h.n, x, q, h.t);
        const double rhs_scale = size.xi+size.qi;
        double guess = std::max(size.xi / (1 + std::min(100., range)),
                                size.qi / (1 + 0.20 * range));
        if (range >= 10 && size.qi > 0) {
            const int stride = std::max(h.n / 384, 1);
            double pred = 0;
            for (int i = 0; i < h.n; i += stride)
                pred = std::max(pred, std::abs(q[i])/(1.+h.t*diag[i]));
            guess = std::max(guess, .26 * h.t * pred);
        }
        if (range >= 5 && last_rhs_scale > 0)
            guess = std::max(guess, .20 * last_y_scale * (rhs_scale/last_rhs_scale));
        if (range < 0.5) {
            guess = 0;
            for (int i = 0; i < h.n; ++i)
                guess = std::max(guess, std::abs(madd(h.t, q[i], x[i])));
        }
        int m = series.degree(0.30 * (h.atol + h.rtol*guess), size.x2, size.q2);
        if (small_taylor(m,h.atol+h.rtol*guess,x,q,y,size)) return;
#ifdef FAM_ENABLE_DEFLATE
        if (deflation.vec && m >= 80) {
            const double loss = deflation.loss * (size.x2+size.q2);
            const double expected = h.atol+h.rtol*guess;
            if (loss <= .12*expected) {
                const int dm = reduced_series.degree(.21*expected,size.x2,size.q2);
                const double tail = 1.001*reduced_series.error(dm,size.x2,size.q2);
                const double saved = upper;
                upper = deflation.top;
                std::swap(series,reduced_series);
                use_deflation = true;
                evaluate(dm,x,q,y,size,w0,w1,dz);
                use_deflation = false;
                std::swap(series,reduced_series);
                upper = saved;
                deflation.correct(h.n,x,q,y);
                const double yn = max_abs(h.n,y);
#ifdef FAM_TRACE
                std::fprintf(stderr,"deflated m=%d error/tol=%.4g\n",dm,(tail+loss)/(h.atol+h.rtol*yn));
#endif
                if (std::isfinite(yn) && tail+loss <= .43*(h.atol+h.rtol*yn)) {
                    last_rhs_scale = rhs_scale;
                    last_y_scale = yn;
                    return;
                }
            }
        }
#endif
#ifdef FAM_ENABLE_STEADY
        if (spectral_floor >= 10 && m >= 30) {
            const Steady::Data d{h.n, row.get(), col.get(), a.get(), diag.get(),
                                 rowsum.get(), h.t, h.rtol, h.atol, 0};
            if (steady.stationary(d,q,y,spectral_floor,size.x2,size.q2,
                                  h.atol+h.rtol*guess,std::min(72,m/3))) return;
        }
        if (try_steady && !differences && !bands.count && m >= 460 && slices.locality_score < 1.1 &&
            h.rtol >= 1e-7 && min_scaled_diag >= (h.rtol >= 1e-4 ? 140. : 280.)) {
            const Steady::Data d{h.n, row.get(), col.get(), a.get(), diag.get(),
                                 rowsum.get(), h.t, h.rtol, h.atol, Steady::sigma};
            double estimate = guess;
            for (int i = 0; i < h.n; ++i)
                estimate = std::max(estimate,std::abs(h.t*q[i])/(1.+h.t*diag[i]));
            int iters = 0, steps = 0;
            const bool ok = steady.propagate(d,x,q,y,size.x2+size.q2,estimate,m / 3,iters,steps);
#ifdef FAM_TRACE
            std::fprintf(stderr, "steady ok=%d iterations=%d steps=%d\n",ok,iters,steps);
#endif
            if (ok) return;
            try_steady = false;
        }
#endif
#if defined(FAM_ENABLE_STEADY) && defined(FAM_RECENTER)
        double actual = 0;
        if (recenter(x,q,y,size,m,guess,actual)) {
            last_rhs_scale = rhs_scale;
            last_y_scale = actual;
            return;
        }
#endif
#ifdef FAM_FORCE_WIDE
        bool used_wide = true;
#else
        bool used_wide = false;
#endif
        for (;;) {
            if (used_wide) evaluate(m, x, q, y, size, wide0, wide1, widez);
            else           evaluate(m, x, q, y, size, w0, w1, dz);
            const double out = max_abs(h.n, y);
            if (!std::isfinite(out) && !used_wide) {
                m = series.degree(.10*h.atol,size.x2,size.q2);
                used_wide = true;
                continue;
            }
            const double tol = h.atol + h.rtol*out;
            const double tail = series.error(m, size.x2, size.q2);
            const bool retry_tail = tail > 0.48 * tol;
            const double risk = (differences ? 2e-19 : 3e-18) *
                 (m+1) * (m+1) * (size.xi + size.qi);
#ifdef FAM_DOUBLE_ONLY
            const bool retry_wide = false;
            (void)risk;
#else
            const bool cancellation = h.rtol <= 1e-10 && m >= 13 && out < .003 &&
                 size.xi > 0 && size.qi > 0 && rhs_scale > 58*(out+.001);
            const bool retry_wide = !used_wide &&
                 (cancellation || (h.rtol <= 1e-10 && m >= 80 && risk > 2.0*tol));
#endif
#ifdef FAM_TRACE
            std::fprintf(stderr, "cheb m=%d eta=%.4g tail/tol=%.3g round/tol=%.3g wide=%d diff=%d bands=%d slices=%d\n",
                         m, range, tail/tol, risk/tol, used_wide, differences, bands.count, slices.npad);
#endif
            if (!retry_tail && !retry_wide) {
                last_rhs_scale = rhs_scale;
                last_y_scale = out;
                break;
            }
            if (retry_wide && !retry_tail && out >= 4e-3 && risk <= 160*tol &&
                try_interval_check) {
                if (check_interval(x,q,y,size,tol)) {
                    last_rhs_scale = rhs_scale;
                    last_y_scale = out;
                    break;
                }
                try_interval_check = false;
            }
            if (retry_tail) m = std::max(m+1, series.degree((h.atol + 0.8 * h.rtol*out)*0.20,
                                                          size.x2, size.q2));
            if (retry_wide) used_wide = true;
        }
    }
};

} // namespace

int main() {
    if (!write_bytes("READY\n", 6)) return 1;
    Header h;
    if (!h.read_header()) return 1;
    Matrix matrix(h);
    if (!matrix.read_matrix()) return 1;
    matrix.prepare();
    const auto rhs = storage<double>(2*static_cast<size_t>(h.n));
    double* const x = rhs.get();
    double* const q = x+h.n;
    const auto y = storage<double>(h.n);
    const size_t bytes = static_cast<size_t>(h.n) * sizeof(double);
    for (int j = 0; j < h.requests; ++j) {
        if (!read_bytes(x, 2*bytes)) return 1;
        matrix.solve(x, q, y.get());
        if (!write_bytes(y.get(), bytes)) return 1;
    }
    return 0;
}
