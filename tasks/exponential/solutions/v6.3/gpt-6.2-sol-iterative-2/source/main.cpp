#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cfloat>
#include <charconv>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string_view>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>
#include <immintrin.h>
#include "fftw_api.hpp"

namespace {

struct Workspace {
    static constexpr size_t size = 40*1024*1024;
    char* start = nullptr;
    size_t used = 0;
    void prepare() {
#ifndef NO_WORKSPACE
        auto* p = mmap(nullptr,size,PROT_READ|PROT_WRITE,
                       MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if (p == MAP_FAILED) return;
        start=static_cast<char*>(p);
        madvise(start,size,MADV_HUGEPAGE);
        // Fault the generic scratch pages before the session begins.
        volatile char* bytes=start;
        for (size_t i=0; i<size; i+=4096) bytes[i]=0;
#endif
    }
    bool owns(const void* p) const {
        return uintptr_t(p)-uintptr_t(start)<size && start;
    }
    void* take(size_t n) {
        if (!start || n>size-used) return nullptr;
        char* p=start+used;
        used+=n;
        return p;
    }
};

Workspace workspace;

struct Free {
    void operator()(void* p) const noexcept {
        if (!workspace.owns(p)) std::free(p);
    }
};

template<class T>
using Buffer = std::unique_ptr<T[], Free>;

template<class T>
Buffer<T> buffer(size_t size) {
    const size_t bytes=(sizeof(T)*size+63)&~size_t(63);
    auto* p = static_cast<T*>(workspace.take(bytes));
    if (!p) p=static_cast<T*>(std::aligned_alloc(64,bytes));
    if (!p && size) throw std::bad_alloc();
    return Buffer<T>(p);
}

bool transfer(int fd, void* storage, size_t bytes) {
    auto* p = static_cast<char*>(storage);
    while (bytes) {
        const ssize_t done = fd == STDIN_FILENO ? read(fd, p, bytes) : write(fd, p, bytes);
        if (done < 0 && errno == EINTR) continue;
        if (done <= 0) return false;
        p += done;
        bytes -= size_t(done);
    }
    return true;
}

template<class T>
bool input(T* p, size_t count) { return transfer(STDIN_FILENO, p, sizeof(T)*count); }

bool output(const void* data, size_t bytes) {
    return transfer(STDOUT_FILENO, const_cast<void*>(data), bytes);
}

struct Header {
    int n, nnz, requests;
    double t, rtol, atol;

    bool read() {
        char magic[8], json[4097];
        uint32_t length;
        if (!input(magic, 8) || std::memcmp(magic, "FAMEXP2\n", 8) ||
            !input(&length, 1) || length == 0 || length > 4096 || !input(json, length))
            return false;
        json[length] = 0;
        const char* pos=json;
        auto spaces = [&] {
            while (*pos==' ' || *pos=='\t' || *pos=='\r' || *pos=='\n') ++pos;
        };
        auto token = [&](char key) {
            spaces();
            if (*pos!=key) return false;
            ++pos;
            return true;
        };
        auto string = [&](char* dest) {
            if (!token('"')) return false;
            int i=0;
            while (*pos && *pos!='"') {
                char c=*pos++;
                if (c=='\\') {
                    if (*pos!='u' || json+length-pos<5) return false;
                    ++pos;
                    unsigned value=0;
                    auto r=std::from_chars(pos,pos+4,value,16);
                    if (r.ec!=std::errc{} || r.ptr!=pos+4 || !value || value>127) return false;
                    pos+=4;
                    c=char(value);
                }
                if (i==31) return false;
                dest[i++]=c;
            }
            dest[i]=0;
            return *pos && (++pos,true);
        };
        constexpr std::string_view keys[]={"version","n","nnz","requests","t","rtol","atol","scenario"};
        double values[7]{};
        unsigned seen=0;
        int scenario=0;
        if (!token('{')) return false;
        for (int field=0; field<8; ++field) {
            char name[32];
            if (field && !token(',')) return false;
            if (!string(name) || !token(':')) return false;
            auto key=std::find(std::begin(keys),std::end(keys),std::string_view(name));
            if (key==std::end(keys)) return false;
            const int i=int(key-keys);
            if (seen&(1u<<i)) return false;
            seen|=1u<<i;
            if (i==7) {
                char scenario_name[32];
                if (!string(scenario_name)) return false;
                const std::string_view value(scenario_name);
                scenario=value=="one_shot" ? 1 : value=="reuse" ? 16 : 0;
            } else {
                spaces();
                auto r=std::from_chars(pos,json+length,values[i]);
                if (r.ec!=std::errc{}) return false;
                pos=r.ptr;
            }
        }
        if (!token('}')) return false;
        spaces();
        if (*pos || values[0]!=2 || !(values[1]>=1 && values[1]<=120000) ||
            !(values[2]>=0 && values[2]<=1000000) || values[3]!=scenario) return false;
        n=int(values[1]); nnz=int(values[2]); requests=scenario;
        t=values[4]; rtol=values[5]; atol=values[6];
        return n==values[1] && nnz==values[2] &&
               nnz <= 1000000 && (requests == 1 || requests == 16) && t >= 0 &&
               t <= 10000 && rtol > 0 && std::isfinite(rtol) && atol > 0 && std::isfinite(atol);
    }
};

double phi(double x) {
    return x == 0 ? 1 : -std::expm1(-x)/x;
}

void scalar_functions(const double* x, double* e, double* f, int n, double se, double sf) {
    const __m256d zero = _mm256_setzero_pd();
    int i=0;
    for (; i+4<=n; i+=4) {
        const __m256d z = _mm256_load_pd(x+i);
        const __m256d ones = _mm256_set1_pd(1);
        const __m256d small = _mm256_cmp_pd(z,_mm256_set1_pd(0.125),_CMP_LT_OQ);
        const int blend = _mm256_movemask_pd(small);
        __m256d s = zero;
        if (blend) {
            const __m256d u = _mm256_xor_pd(z,_mm256_set1_pd(-0.0));
            s = _mm256_set1_pd(1./3628800);
            s = _mm256_fmadd_pd(u,s,_mm256_set1_pd(1./362880));
            s = _mm256_fmadd_pd(u,s,_mm256_set1_pd(1./40320));
            s = _mm256_fmadd_pd(u,s,_mm256_set1_pd(1./5040));
            s = _mm256_fmadd_pd(u,s,_mm256_set1_pd(1./720));
            s = _mm256_fmadd_pd(u,s,_mm256_set1_pd(1./120));
            s = _mm256_fmadd_pd(u,s,_mm256_set1_pd(1./24));
            s = _mm256_fmadd_pd(u,s,_mm256_set1_pd(1./6));
            s = _mm256_fmadd_pd(u,s,_mm256_set1_pd(0.5));
            s = _mm256_fmadd_pd(u,s,ones);
        }
        __m256d E, F;
        if (blend == 15) {
            E = _mm256_fnmadd_pd(z,s,ones);
            F = s;
        } else {
            const __m256d cutoff = _mm256_set1_pd(48.);
            const auto big = _mm256_cmp_pd(z,cutoff,_CMP_GT_OQ);
            if (_mm256_movemask_pd(big)==15) E = zero;
            else {
                E = _ZGVdN4v_exp(_mm256_xor_pd(_mm256_min_pd(z,cutoff),_mm256_set1_pd(-0.0)));
                E = _mm256_andnot_pd(big,E);
            }
            F = _mm256_div_pd(_mm256_sub_pd(ones,E),_mm256_blendv_pd(z,ones,small));
            F = _mm256_blendv_pd(F,s,small);
        }
        _mm256_store_pd(e+i,_mm256_mul_pd(_mm256_set1_pd(se),E));
        _mm256_store_pd(f+i,_mm256_mul_pd(_mm256_set1_pd(sf),F));
    }
    for (; i<n; ++i) {
        const double z=x[i];
        e[i] = z>48. ? 0. : se*std::exp(-z);
        f[i] = sf*phi(z);
    }
}

struct Norms {
    double i0, i1, s0, s1, initial;
};

Norms norms(const double* b0, const double* b1, int n, double t) {
    auto s0 = _mm256_setzero_pd(), s1=s0, a0=s0, a1=s0, a2=s0;
    const auto sign = _mm256_set1_pd(-0.), tt = _mm256_set1_pd(t);
    int i=0;
    for (; i+4<=n; i+=4) {
        const auto x=_mm256_load_pd(b0+i), y=_mm256_load_pd(b1+i);
        s0=_mm256_fmadd_pd(x,x,s0);
        s1=_mm256_fmadd_pd(y,y,s1);
        a0=_mm256_max_pd(a0,_mm256_andnot_pd(sign,x));
        a1=_mm256_max_pd(a1,_mm256_andnot_pd(sign,y));
        a2=_mm256_max_pd(a2,_mm256_andnot_pd(sign,_mm256_fmadd_pd(tt,y,x)));
    }
    auto horizontal_max = [](__m256d x) {
        auto y=_mm_max_pd(_mm256_castpd256_pd128(x),_mm256_extractf128_pd(x,1));
        return _mm_cvtsd_f64(_mm_max_sd(y,_mm_unpackhi_pd(y,y)));
    };
    auto horizontal_sum = [](__m256d x) {
        auto y=_mm_add_pd(_mm256_castpd256_pd128(x),_mm256_extractf128_pd(x,1));
        return _mm_cvtsd_f64(_mm_add_sd(y,_mm_unpackhi_pd(y,y)));
    };
    double i0=horizontal_max(a0), i1=horizontal_max(a1), initial=horizontal_max(a2);
    double z0=horizontal_sum(s0), z1=horizontal_sum(s1);
    for (; i<n; ++i) {
        double a = b0[i], b = b1[i];
        i0 = std::max(i0, std::abs(a));
        i1 = std::max(i1, std::abs(b));
        z0 += a*a;
        z1 += b*b;
        initial = std::max(initial, std::abs(std::fma(t, b, a)));
    }
    return {i0,t*i1,std::sqrt(z0),t*std::sqrt(z1),initial};
}

double inf_norm(const double* x, int n) {
    double r = 0;
    for (int i=0; i<n; ++i) r = std::max(r, std::abs(x[i]));
    return r;
}

struct AccurateSum {
    double hi=0, low=0;
    void product(double a, double b) {
        const double term=a*b;
        const double e=std::fma(a,b,-term);
        const double sum=hi+term, v=sum-hi;
        low+=(hi-(sum-v))+(term-v)+e;
        hi=sum;
    }
    long double value() const { return static_cast<long double>(hi)+low; }
};

// Cosine coefficients on x = a + (b-a)*(1-cos(theta))/2.
struct CosineTable {
    static constexpr int capacity = 8192;
    struct Z { long double r, i; };
    std::array<Z, capacity/2> roots;
    std::array<long double, capacity/2+1> lambda;

    void prepare() {
        constexpr long double pi = 3.141592653589793238462643383279502884L;
        for (int k=0; k<capacity/2; ++k) {
            long double angle = k*(2*pi/capacity);
            roots[k] = {std::cos(angle), -std::sin(angle)};
            long double s = std::sin(angle/2);
            lambda[k] = s*s;
        }
        lambda.back() = 1;
    }

    void fft(Z* f, int count) const {
        for (unsigned i=1, j=0; i<unsigned(count); ++i) {
            unsigned h = count/2;
            for (; j&h; h>>=1) j ^= h;
            j ^= h;
            if (i<j) std::swap(f[i], f[j]);
        }
        for (int len=2; len<=count; len*=2) {
            int stride = capacity/len, half = len/2;
            for (int l=0; l<count; l+=len) {
                for (int k=0; k<half; ++k) {
                    const Z u = f[l+k], w = roots[stride*k], a = f[l+k+half];
                    const Z v{w.r*a.r-w.i*a.i, w.i*a.r+w.r*a.i};
                    f[l+k] = {u.r+v.r, u.i+v.i};
                    f[l+k+half] = {u.r-v.r, u.i-v.i};
                }
            }
        }
    }
};

CosineTable cosine;

struct Coefficients {
    struct Pair { double e, f; };
    std::vector<Pair> c, tail;

    static double tail_bound(double halfwidth, int after) {
        const double s = std::asinh((after+1)/halfwidth);
        const double u = std::sinh(0.5*s);
        return 2*std::exp(2*halfwidth*u*u-(after+1)*s);
    }

    void build(double a, double width, double t) {
        int m = 16;
        double excess;
        do {
            excess = tail_bound(width/2, m-1);
            if (excess <= 1e-23 || 2*m == CosineTable::capacity) break;
            m *= 2;
        } while (true);
        auto v = buffer<CosineTable::Z>(2*m);
        const int stride = CosineTable::capacity/(2*m);
        for (int i=0; i<=m; ++i) {
            const long double x = a + width*cosine.lambda[i*stride];
            const long double e = std::exp(-x);
            long double f;
            if (x < 0.5L)
                f = x == 0 ? 1 : -std::expm1(-x)/x;
            else
                f = (1-e)/x;
            v[i] = {e, f};
            if (i != 0 && i != m) v[2*m-i] = v[i];
        }
        cosine.fft(v.get(), 2*m);
        c.resize(m);
        tail.resize(m);
        Pair sum{excess*std::exp(-a), excess*phi(a)};
        for (int i=m-1; i>=0; --i) {
            long double scale = i ? 1.0L/m : 0.5L/m;
            const double e = std::max(0.0, double(v[i].r*scale));
            const double f = std::max(0.0, double(v[i].i*scale));
            c[i] = {e, t*f};
            tail[i] = sum;
            sum.e += e;
            sum.f += f;
        }
    }

    int degree(const Norms& b, double tol) const {
        size_t lo=0, hi=c.size()-1;
        while (lo<hi) {
            size_t k = (lo+hi)/2;
            if (tail[k].e*b.s0+tail[k].f*b.s1 <= tol) hi = k;
            else lo = k+1;
        }
        return int(lo);
    }
};

#include "packed.hpp"
#include "krylov.hpp"
#include "tensor.hpp"
#include "rational.hpp"
#include "small_blocks.hpp"
#include "steady.hpp"
#ifndef NO_BANDED
#include "banded.hpp"
#endif

struct Solver {
    Header h;
    Buffer<int> ptr, col;
    Buffer<double> val;
    Buffer<double> b0, b1, y, w0, w1, delta;
    Buffer<double> diag, rad, shift;
    Buffer<long double> slack;
    Coefficients poly;
    Packed packed;
    Krylov krylov;
    Tensor tensor;
    Rational rational;
    Steady steady;
    SmallBlocks blocks;
#ifndef NO_BANDED
    Banded band;
#endif
    std::unique_ptr<Solver> inner;
    Buffer<int> inner_order, iso_order;
    int iso_size=0;

    Buffer<int> mate;
    Buffer<double> local_e, local_f, cross_e, cross_f;
    double lower = 0, upper = 0, bound = 0;
    double growth = 0;
    double lower_inf = 0;
    bool skewed = false;
    bool wide_pattern = false;
    bool nonpos = true;
    bool polynomial_ready = false;
    double constant_lambda=0, constant_error=-1;
    int maxlen = 0;
    int kernel = 0;
    enum { Copy, ZeroOperator, Local, Taylor, Chebyshev, Blocks, Split } method = Copy;

    explicit Solver(const Header& header): h(header) {}

    bool read_matrix() {
        auto wide_ptr = buffer<uint64_t>(h.n+1);
        ptr = buffer<int>(h.n+1);
        col = buffer<int>(h.nnz);
        val = buffer<double>(h.nnz);
        if (!input(wide_ptr.get(), h.n+1) || !input(col.get(), h.nnz) || !input(val.get(), h.nnz))
            return false;
        ptr[0] = 0;
        for (int i=1; i<=h.n; ++i) {
            ptr[i] = int(wide_ptr[i]);
            maxlen = std::max(maxlen,ptr[i]-ptr[i-1]);
        }
        b0 = buffer<double>(h.n+8);
        b1 = buffer<double>(h.n+8);
        std::fill_n(b0.get()+h.n,8,0.);
        std::fill_n(b1.get()+h.n,8,0.);
        return true;
    }

    void prepare_local() {
        const int n = h.n;
        if (!diag) {
            diag = buffer<double>(n);
            for (int i=0; i<n; ++i) {
                double d = 0;
                for (int j=ptr[i]; j<ptr[i+1]; ++j)
                    if (col[j]==i) d = val[j];
                diag[i] = d;
            }
        }
        auto lambda = buffer<double>(n);
        local_e = buffer<double>(n);
        local_f = buffer<double>(n);
        if (maxlen <= 1) {
            for (int i=0; i<n; ++i) lambda[i] = h.t*diag[i];
            scalar_functions(lambda.get(),local_e.get(),local_f.get(),n,1.,h.t);
            return;
        }
        mate = buffer<int>(n);
        cross_e = buffer<double>(n);
        cross_f = buffer<double>(n);
        std::fill_n(mate.get(), n, -1);
        for (int i=0; i<n; ++i) {
            if (mate[i] >= 0) continue;
            int j = -1;
            double c = 0;
            for (int k=ptr[i]; k<ptr[i+1]; ++k) {
                if (col[k] != i && val[k] != 0) {
                    j = col[k];
                    c = val[k];
                }
            }
            if (j < 0) {
                mate[i] = i;
                lambda[i] = h.t*diag[i];
                cross_e[i] = cross_f[i] = 0;
                continue;
            }
            mate[i] = j;
            mate[j] = i;
            const long double a = diag[i], b = diag[j], z = c;
            const long double d = (a-b)/2, rr = std::sqrt(d*d+z*z);
            const long double big = (a+b)/2 + rr;
            const long double small = std::max(0.0L, (a*b-z*z)/big);
            lambda[i] = double(small*h.t);
            lambda[j] = double(big*h.t);
            cross_e[i] = double(0.5L*(1-d/rr));
            cross_e[j] = double(0.5L*(1+d/rr));
            cross_f[i] = double(-0.5L*z/rr);
        }
        scalar_functions(lambda.get(),local_e.get(),local_f.get(),n,1.,h.t);
        for (int i=0; i<n; ++i) {
            const int j=mate[i];
            if (j <= i) continue;
            const double e = local_e[i], E = local_e[j];
            const double f = local_f[i], F = local_f[j];
            const double wi = cross_e[i], wj = cross_e[j];
            local_e[i] = wi*e+wj*E;
            local_e[j] = wj*e+wi*E;
            local_f[i] = wi*f+wj*F;
            local_f[j] = wj*f+wi*F;
            const double w = cross_f[i];
            cross_e[i] = cross_e[j] = w*(e-E);
            cross_f[i] = cross_f[j] = w*(f-F);
        }
    }

    void prepare_split(int isolated) {
        Header sub=h;
        sub.n-=isolated;
        inner=std::make_unique<Solver>(sub);
        inner_order=buffer<int>(sub.n);
        iso_order=buffer<int>(isolated);
        local_e=buffer<double>(isolated);
        local_f=buffer<double>(isolated);
        auto lambda=buffer<double>(isolated);
        auto map=buffer<int>(h.n);
        int live=0;
        for (int i=0; i<h.n; ++i) {
            bool singleton=true;
            for (int k=ptr[i]; k<ptr[i+1]; ++k)
                if (col[k]!=i && val[k]!=0) { singleton=false; break; }
            if (singleton) {
                iso_order[iso_size]=i;
                lambda[iso_size++]=h.t*diag[i];
                map[i]=-1;
            } else {
                inner_order[live]=i;
                map[i]=live++;
            }
        }
        scalar_functions(lambda.get(),local_e.get(),local_f.get(),isolated,1.,h.t);
        inner->ptr=buffer<int>(sub.n+1);
        inner->col=buffer<int>(h.nnz);
        inner->val=buffer<double>(h.nnz);
        int pos=0;
        for (int i=0; i<sub.n; ++i) {
            const int old=inner_order[i], start=pos;
            inner->ptr[i]=start;
            for (int k=ptr[old]; k<ptr[old+1]; ++k) {
                if (val[k]!=0) {
                    inner->val[pos]=val[k];
                    inner->col[pos++]=map[col[k]];
                }
            }
            inner->maxlen=std::max(inner->maxlen,pos-start);
        }
        inner->ptr[sub.n]=pos;
        inner->h.nnz=pos;
        inner->b0=buffer<double>(sub.n+8);
        inner->b1=buffer<double>(sub.n+8);
        std::fill_n(inner->b0.get()+sub.n,8,0.);
        std::fill_n(inner->b1.get()+sub.n,8,0.);
        inner->prepare();
    }

    bool try_short() {
        if (h.n<512) return false;
        double trial=0;
        for (int s=0; s<16; ++s) {
            const int i=(h.n-1)*s/15;
            double a=0;
            for (int j=ptr[i]; j<ptr[i+1]; ++j) a+=std::abs(val[j]);
            trial=std::max(trial,a);
        }
        if (h.t*trial>0.72) return false;
        double hi=0.,lo=std::numeric_limits<double>::infinity();
        for (int i=0; i<h.n; ++i) {
            double s=0., v=0., d=0.;
            int k=ptr[i], end=ptr[i+1];
            for (; k+1<end; k+=2) {
                s+=std::abs(val[k]);
                v+=std::abs(val[k+1]);
                if (col[k]==i) d=val[k];
                if (col[k+1]==i) d=val[k+1];
            }
            if (k<end) {
                s+=std::abs(val[k]);
                if (col[k]==i) d=val[k];
            }
            double r=(s+v)*(1+(end-ptr[i]+5)*DBL_EPSILON);
            hi=std::max(hi,r);
            lo=std::min(lo,2*d-r);
            if (h.t*hi>1.15) return false;
        }
        if (h.t*hi>0.6 && h.t*(hi-lo)>1.2) return false;
        method=Taylor;
        bound=hi;
        lower_inf=lo;
        w0=buffer<double>(h.n+8);
        w1=buffer<double>(h.n+8);
        return true;
    }

    void prepare() {
        if (h.t == 0) return;
        const int n = h.n;
        y = buffer<double>(n);
        if (h.nnz == 0) {
            method = ZeroOperator;
            return;
        }
        // A PSD row with a nonzero off-diagonal must also have a diagonal.
        if (maxlen <= 2) {
            method = Local;
            prepare_local();
            return;
        }
        if (try_short()) return;
        diag = buffer<double>(n);
        rad = buffer<double>(n);
        slack = buffer<long double>(n);
        int maxoff = 0;
        int far = 0;
        int isolated = 0;
        double isolate_max = 0, connected_max = 0;
        double sum_diag = 0, max_diag = 0;
        double all_lower = std::numeric_limits<double>::infinity();
        for (int i=0; i<n; ++i) {
            double d = 0;
            long double r = 0, re = 0;
            int off = 0;
            const bool compensate=ptr[i+1]-ptr[i]>128;
            for (int j=ptr[i]; j<ptr[i+1]; ++j) {
                const double a = val[j];
                if (col[j] == i) {
                    d = a;
                } else if (a != 0) {
                    const long double p=std::abs(a);
                    const long double sum=r+p;
                    if (compensate) {
                        const long double v=sum-r;
                        re+=(r-(sum-v))+(p-v);
                    }
                    r=sum;
                    nonpos &= a < 0;
                    far += std::abs(col[j]-i) > 512;
                    ++off;
                }
            }
            maxoff = std::max(maxoff, off);
            diag[i] = d;
            sum_diag += d;
            max_diag = std::max(max_diag,d);
            slack[i] = (d-r)-re;
            const long double radius=r+re;
            const long double guard = 4*DBL_EPSILON*(std::abs(d)+radius) +
                                      (off+1)*LDBL_EPSILON*radius;
            rad[i] = double(radius+guard);
            const double row=double(d+radius+guard);
            bound = std::max(bound,row);
            if (off) connected_max=std::max(connected_max,row);
            else {
                ++isolated;
                isolate_max=std::max(isolate_max,row);
            }
            all_lower = std::min(all_lower, double(slack[i]-guard));
        }
        if (maxoff <= 1) {
            method = Local;
            prepare_local();
            return;
        }
        if (maxoff<SmallBlocks::limit && (h.t*bound>30 || n<=32) &&
            blocks.prepare(n,ptr.get(),col.get(),val.get(),h.t)) {
            method=Blocks;
            return;
        }
        if (isolated && h.t*bound>8 &&
            (isolated>n/24 || isolate_max>1.25*connected_max)) {
            method=Split;
            prepare_split(isolated);
            return;
        }
        w0 = buffer<double>(n+8);
        w1 = buffer<double>(n+8);
        lower_inf = all_lower;
        if (h.t*bound <= 0.6 ||
            (h.t*bound <= 1.15 && h.t*(bound-all_lower)<=1.2)) {
            method = Taylor;
            return;
        }
        method = Chebyshev;
        lower = std::max(all_lower, 0.0);
        upper = 0;
        // Bound the largest eigenvalue using the maximal eigenvector row and
        // its maximal neighbor; take the intersection with the row-sum bound.
        for (int i=0; i<n; ++i) upper = std::max(upper, diag[i]);
        for (int i=0; i<n; ++i) {
            const double row = diag[i]+rad[i];
            if (row <= upper) continue;
            double d = 0, r = 0;
            for (int k=ptr[i]; k<ptr[i+1]; ++k) {
                int j = col[k];
                if (j != i && val[k] != 0) {
                    d = std::max(d, diag[j]);
                    r = std::max(r, rad[j]);
                }
            }
            const double diff = 0.5*(diag[i]-d);
            const double edge = 0.5*(diag[i]+d)+std::sqrt(diff*diff+rad[i]*r);
            upper = std::max(upper, std::min(row, edge));
        }
        upper = std::min(bound, upper*(1+8*DBL_EPSILON));
#ifndef NO_WEIGHTED_BOUND
        // A scaled Gershgorin bound is particularly useful for a large hub
        // whose neighbors have much smaller diagonals.
        if (h.t*upper > 40 && upper>1.12*max_diag && max_diag > 5*(sum_diag/n)) {
            auto weight=buffer<double>(n);
            const double center=1.052*max_diag;
            for (int i=0; i<n; ++i) weight[i]=1+rad[i]/(center-diag[i]);
            double hi=max_diag;
            for (int i=0; i<n; ++i) {
                double s=0;
                for (int k=ptr[i]; k<ptr[i+1]; ++k)
                    if (col[k]!=i) s += std::abs(val[k])*weight[col[k]];
                hi=std::max(hi,diag[i]+s/weight[i]*(1+(ptr[i+1]-ptr[i]+2)*DBL_EPSILON));
            }
            upper=std::min(upper,hi*(1+4*DBL_EPSILON));
        }
#endif
        if (all_lower*h.t > -0.05) growth = std::exp(std::max(0.0,-all_lower*h.t));
        skewed = max_diag > 12*(sum_diag/n) && h.t*upper > 96;
        wide_pattern = far > h.nnz/7;
        kernel = h.rtol < 1e-8 && h.t*upper > 128 &&
                 all_lower >= -32*DBL_EPSILON*upper ? (nonpos ? 1 : 2) : 0;
#ifndef NO_TENSOR
        if (n >= 256 && maxoff <= 8 &&
            h.t*(upper-lower) > (h.requests == 1 ? 24 : 6) &&
            tensor.prepare(n,ptr.get(),col.get(),val.get(),h.t,bound,all_lower))
            return;
#endif
        prepare_polynomial();
    }

    void prepare_polynomial() {
        const int n = h.n;
        shift = buffer<double>(n);
        delta = buffer<double>(n+8);
        int pos = 0;
        for (int i=0; i<n; ++i) {
            shift[i] = kernel ? double(slack[i]-lower) : diag[i]-lower;
            const int stop = ptr[i+1], start = ptr[i];
            ptr[i] = pos;
            for (int k=start; k<stop; ++k) {
                if (col[k] != i && val[k] != 0) {
                    col[pos] = col[k];
                    val[pos++] = val[k];
                }
            }
        }
        ptr[n] = pos;
        const double width = h.t*(upper-lower);
        poly.build(h.t*lower, width, h.t);
#ifndef NO_BANDED
        if (n>=512 && width>(h.requests==1 ? 16. : 2.) &&
            band.build(n,ptr.get(),col.get(),val.get(),shift.get())) {
            polynomial_ready=true;
            return;
        }
#endif
#ifndef NO_PACK
        if (n >= 384 && width > (h.requests == 1 ? 12 : 0.8)) {
            packed.build(n, ptr.get(), col.get(), val.get(), shift.get(), kernel);
            std::fill_n(w0.get()+n, 8, 0.);
            std::fill_n(w1.get()+n, 8, 0.);
        }
#endif
        polynomial_ready = true;
    }

    const double* solve_local() {
        if (!mate) {
            for (int i=0; i<h.n; ++i) y[i] = local_e[i]*b0[i]+local_f[i]*b1[i];
            return y.get();
        }
        for (int i=0; i<h.n; ++i) {
            int j = mate[i];
            const double own = local_e[i]*b0[i]+local_f[i]*b1[i];
            const double other = cross_e[i]*b0[j]+cross_f[i]*b1[j];
            y[i] = own+other;
        }
        return y.get();
    }

    const double* solve_taylor(const Norms& b) {
        const double center = h.t*std::max(0.,0.5*(bound+lower_inf));
        const double eta = std::max(h.t*bound-center,center-h.t*lower_inf)*(1+4*DBL_EPSILON);
        const double decay = std::exp(-center);
        const double lower_initial = std::max(0.0, b.initial-
            std::expm1(h.t*bound)*(b.i0+0.5*b.i1));
        const double goal = 0.8*(h.atol+h.rtol*lower_initial);
        double scalar = eta, invfact = decay;
        int d = 0;
        while (scalar*(decay*b.i0+b.i1/(d+2))/(1-eta/(d+2)) > goal) {
            ++d;
            scalar *= eta/(d+1);
        }
        for (int k=2; k<=d; ++k) invfact /= k;
        double* a = w0.get();
        double* c = w1.get();
        auto forcing = [&](int k) {
            if (!k) return h.t*phi(center);
            double u=invfact/(k+1), s=u;
            for (int j=1; j<=16; ++j) {
                u*=center/(k+j+1);
                s+=u;
                if (u<0.25*DBL_EPSILON*s) break;
            }
            return h.t*s;
        };
        double phiscale = forcing(d);
        for (int i=0; i<h.n; ++i) a[i] = invfact*b0[i]+phiscale*b1[i];
        for (int k=d-1; k>=0; --k) {
            invfact *= k+1;
            phiscale = forcing(k);
            for (int i=0; i<h.n; ++i) {
                double p=0, q=0;
                int j=ptr[i], end=ptr[i+1];
                for (; j+1<end; j+=2) {
                    p=std::fma(val[j],a[col[j]],p);
                    q=std::fma(val[j+1],a[col[j+1]],q);
                }
                if (j<end) p=std::fma(val[j],a[col[j]],p);
                c[i] = std::fma(-h.t,p+q,std::fma(center,a[i],invfact*b0[i]+phiscale*b1[i]));
            }
            std::swap(a,c);
        }
#ifdef PROFILE
        std::fprintf(stderr,"Taylor eta %.3g d %d\n", eta, d);
#endif
        return a;
    }

    template<int mode, bool last>
    void clenshaw_pass(double* __restrict out, const double* __restrict v,
                       double* __restrict diff, const double* __restrict a,
                       const double* __restrict b, double e, double f, double scale) const {
        const int* __restrict p = ptr.get();
        const int* __restrict ci = col.get();
        const double* __restrict cv = val.get();
        const double* __restrict sh = shift.get();
        for (int i=0; i<h.n; ++i) {
            const double vi = v[i];
            double z = sh[i]*vi;
            for (int j=p[i]; j<p[i+1]; ++j) {
                const double vj = v[ci[j]], w = cv[j];
                if constexpr (mode == 0) z=std::fma(w,vj,z);
                if constexpr (mode == 1) z=std::fma(w,vj-vi,z);
                if constexpr (mode == 2) {
                    const auto sign = std::bit_cast<uint64_t>(w) & (uint64_t(1)<<63);
                    const double u = std::bit_cast<double>(std::bit_cast<uint64_t>(vi)^sign);
                    z=std::fma(w,vj+u,z);
                }
            }
            double d = std::fma(scale, z, diff[i]+(e*a[i]+f*b[i]));
            if constexpr (last) out[i] = d;
            else {
                diff[i] = d;
                out[i] = vi+d;
            }
        }
    }

    template<int mode>
    void evaluate_poly(int d) {
        double* u = w0.get();
        double* v = w1.get();
        double* df = delta.get();
        const double* a = b0.get();
        const double* b = b1.get();
        auto c = poly.c[d];
        if (!d) {
            for (int i=0; i<h.n; ++i) y[i] = c.e*a[i]+c.f*b[i];
            return;
        }
        for (int i=0; i<h.n; ++i) df[i] = u[i] = c.e*a[i]+c.f*b[i];
        const double scale = -4/(upper-lower);
        for (int k=d-1; k>=1; --k) {
            c = poly.c[k];
            clenshaw_pass<mode,false>(v,u,df,a,b,c.e,c.f,scale);
            std::swap(u,v);
        }
        c = poly.c[0];
        clenshaw_pass<mode,true>(y.get(),u,df,a,b,c.e,c.f,scale*0.5);
    }

    template<int mode, bool uniform>
    const double* evaluate_packed(int d, const Norms& norm) {
        double* u = w0.get();
        double* v = w1.get();
        double* df = delta.get();
        const double* a = packed.source0;
        const double* b = packed.source1;
        const bool e=norm.i0!=0, f=norm.i1!=0;
        const auto c = poly.c[d];
        const int size = packed.size;
        for (int i=0; i<size; ++i) df[i] = u[i] = c.e*a[i]+c.f*b[i];
        const double scale = -4/(upper-lower);
        const double* result = u;
        if (d) {
            for (int k=d-1; k>=1; --k) {
                const auto cc = poly.c[k];
                packed.pass<mode,uniform,false>(v,u,df,e?cc.e:0.,f?cc.f:0.,scale,ptr.get(),col.get(),val.get());
                std::swap(u,v);
            }
            packed.pass<mode,uniform,true>(v,u,df,e?poly.c[0].e:0.,f?poly.c[0].f:0.,0.5*scale,
                                           ptr.get(),col.get(),val.get());
            result = v;
        }
        if (packed.identity) return result;
        for (int i=0; i<h.n; ++i) y[packed.order[i]] = result[i];
        return y.get();
    }

    const double* solve_poly(const Norms& b) {
        double guess = (b.i0+b.i1)/(1+std::sqrt(h.t*upper));
        const double tol = std::min(b.i1==0 ? 0.32*h.atol : std::numeric_limits<double>::infinity(),
                                    0.15*(h.atol+h.rtol*guess));
        int degree = poly.degree(b,tol);
        if (h.t*lower_inf>=6 && steady.solve(h,b,lower_inf,bound,ptr.get(),col.get(),val.get(),diag.get(),
                                             b0.get(),b1.get(),degree,maxlen,y.get()))
            return y.get();
#ifndef NO_BANDED
        if (band.degree) {
            const double* x=nullptr;
            const double s=-4/(upper-lower);
            for (int j=0; j<2; ++j) {
                if (kernel==0) x=band.solve<0>(poly,degree,s,b0.get(),b1.get());
                else if (kernel==1) x=band.solve<1>(poly,degree,s,b0.get(),b1.get());
                else x=band.solve<2>(poly,degree,s,b0.get(),b1.get());
                const double err=b.s0*poly.tail[degree].e+b.s1*poly.tail[degree].f;
                const double magnitude=inf_norm(x,h.n);
                const double threshold=h.atol+h.rtol*magnitude;
                if (err<=0.5*threshold) break;
                degree=poly.degree(b,0.28*(h.atol+h.rtol*std::max(0.,magnitude-err)));
            }
            return x;
        }
#endif
#ifndef NO_RATIONAL
        if (degree>=220 && h.n>=384 && (wide_pattern || skewed)
            && (h.rtol>=1e-8
                || (degree>=720 && (skewed || (!nonpos && b.i1>0)) &&
                    (b.i0==0 || b.i1==0 || b.initial>0.08*(b.i0+b.i1)))
                )
            && (skewed || !nonpos || degree>=480)
           ) {
            const int shifted_kernel=lower_inf>=-1e-8*bound ? (nonpos ? 1 : 2) : 0;
            const Rational::Matrix mat{h.n,shifted_kernel,ptr.get(),col.get(),val.get(),diag.get(),slack.get(),
                                     h.t,lower,lower_inf};
            if (rational.solve(mat,h,b,upper,b0.get(),b1.get(),degree,tol,y.get()))
                return y.get();
        }
#endif
        if (packed.size) packed.set_rhs(b0.get(), b1.get(), h.n);
#ifndef NO_LANCZOS
        if (packed.size && (skewed || degree >= 500)) {
            auto multiply = [&](double* dest, const double* src) {
                const int* p = ptr.get(), *c = col.get();
                const double* v = val.get();
                if (kernel == 0) {
                    if (packed.common) packed.multiply<0,true>(dest,src,lower,h.t,p,c,v);
                    else packed.multiply<0,false>(dest,src,lower,h.t,p,c,v);
                } else if (kernel == 1) {
                    if (packed.common) packed.multiply<1,true>(dest,src,lower,h.t,p,c,v);
                    else packed.multiply<1,false>(dest,src,lower,h.t,p,c,v);
                } else packed.multiply<2,false>(dest,src,lower,h.t,p,c,v);
            };
            if (krylov.apply(h,b,packed,lower,upper,growth,poly,degree,tol,multiply,w0.get(),y.get(),skewed))
                return y.get();
        }
#endif
        const double* result=y.get();
        for (int step=0; step<2; ++step) {
            if (packed.size) {
                if (kernel == 0) {
                    if (packed.common != 0) result=evaluate_packed<0,true>(degree,b);
                    else result=evaluate_packed<0,false>(degree,b);
                } else if (kernel == 1) {
                    if (packed.common != 0) result=evaluate_packed<1,true>(degree,b);
                    else result=evaluate_packed<1,false>(degree,b);
                } else result=evaluate_packed<2,false>(degree,b);
            } else {
                if (kernel == 0) evaluate_poly<0>(degree);
                else if (kernel == 1) evaluate_poly<1>(degree);
                else evaluate_poly<2>(degree);
            }
            const double err = b.s0*poly.tail[degree].e+b.s1*poly.tail[degree].f;
            const double magnitude=inf_norm(result,h.n);
            const double threshold = h.atol+h.rtol*magnitude;
#ifdef PROFILE
            std::fprintf(stderr,"Cheb eta %.3g interval %.3g %.3g d %d tail %.3g threshold %.3g kernel %d\n",
                         h.t*bound, h.t*lower, h.t*upper, degree, err, threshold, kernel);
#endif
            if (err <= 0.5*threshold) break;
            degree = poly.degree(b,0.28*(h.atol+h.rtol*std::max(0.,magnitude-err)));
        }
        return result;
    }

    static bool constant(const double* v, int n) {
        const double a=v[0];
        if (v[n/2]!=a || v[n-1]!=a) return false;
        for (int i=1; i<n; ++i)
            if (v[i]!=a) return false;
        return true;
    }

    bool try_quiet_mode(const Norms& b) {
        const int n=h.n;
        if (b.i0+b.i1==0) {
            std::fill_n(y.get(),n,0.);
            return true;
        }
        if (h.t*upper<16) return false;
        const double nscale=growth ? growth : std::sqrt(n);
        const double numeric_guard=8*DBL_EPSILON*DBL_EPSILON*(maxlen+4.)*(maxlen+4.)*h.t*bound;
        if (constant(b0.get(),n) && constant(b1.get(),n)) {
            if (constant_error<0) {
                long double a=std::numeric_limits<long double>::infinity(), c=-a;
                for (int i=0; i<n; ++i) {
                    AccurateSum sum;
                    if (polynomial_ready) sum.product(diag[i],1.);
                    for (int k=ptr[i]; k<ptr[i+1]; ++k) sum.product(val[k],1.);
                    long double r=sum.value()*h.t;
                    a=std::min(a,r); c=std::max(c,r);
                }
                constant_lambda=double(std::max(0.L,(a+c)*0.5L));
                constant_error=double(std::max(c-constant_lambda,constant_lambda-a))+numeric_guard;
            }
            const double x=constant_lambda;
            const double z=std::fma(std::exp(-x),b0[0],h.t*phi(x)*b1[0]);
            const double err=constant_error*nscale*(b.i0+0.5*b.i1)+4*DBL_EPSILON*(b.i0+b.i1);
            if (err<0.33*(h.atol+h.rtol*std::abs(z))) {
                std::fill_n(y.get(),n,z);
                return true;
            }
        }
        if (b.i0==0 || b.i1==0 || b.initial>1e-4*(b.i0+b.i1)) return false;
        // Certify y ~= b0 + t*b1 by bounding the actions of A on both
        // vectors; no assumption of an exact nullspace is made.
        const double goal=0.28*(h.atol+h.rtol*b.initial);
        const double guard=nscale*numeric_guard*(b.i0+0.5*b.i1);
        if (guard>goal) return false;
        double max0=0,max1=0,sq0=0,sq1=0;
        for (int i=0; i<n; ++i) {
            AccurateSum u,v;
            if (polynomial_ready) {
                u.product(diag[i],b0[i]); v.product(diag[i],b1[i]);
            }
            for (int k=ptr[i]; k<ptr[i+1]; ++k) {
                const int j=col[k];
                u.product(val[k],b0[j]); v.product(val[k],b1[j]);
            }
            const double r0=std::abs(double(h.t*u.value()));
            const double r1=std::abs(double((0.5L*h.t*h.t)*v.value()));
            max0=std::max(max0,r0); max1=std::max(max1,r1);
            sq0+=r0*r0; sq1+=r1*r1;
            if (max0+max1>goal) return false;
        }
        const double error=std::min((max0+max1)*nscale,std::sqrt(sq0)+std::sqrt(sq1))+guard;
        if (error>goal) return false;
        for (int i=0; i<n; ++i) y[i]=std::fma(h.t,b1[i],b0[i]);
        return true;
    }

    const double* solve_request() {
        if (method == Copy) return b0.get();
        if (method == ZeroOperator) {
            for (int j=0; j<h.n; ++j) y[j] = std::fma(h.t,b1[j],b0[j]);
            return y.get();
        }
        if (method == Local) return solve_local();
        if (method == Blocks) return blocks.solve(b0.get(),b1.get(),y.get());
        if (method == Split) {
            for (int i=0; i<inner->h.n; ++i) {
                const int j=inner_order[i];
                inner->b0[i]=b0[j];
                inner->b1[i]=b1[j];
            }
            const double* x=inner->solve_request();
            for (int i=0; i<inner->h.n; ++i) y[inner_order[i]]=x[i];
            for (int i=0; i<iso_size; ++i) {
                const int j=iso_order[i];
                y[j]=local_e[i]*b0[j]+local_f[i]*b1[j];
            }
            return y.get();
        }
        const Norms b=norms(b0.get(),b1.get(),h.n,h.t);
        if (method == Taylor) return solve_taylor(b);
        if (try_quiet_mode(b)) return y.get();
        const double* answer=tensor.fwd ? tensor.solve(h,b,b0.get(),b1.get()) : nullptr;
        if (answer) return answer;
        if (!polynomial_ready) prepare_polynomial();
        return solve_poly(b);
    }

    bool execute() {
        if (!read_matrix()) return false;
        prepare();
        for (int i=0; i<h.requests; ++i) {
            if (!input(b0.get(), h.n) || !input(b1.get(), h.n)) return false;
            const double* answer=solve_request();
            if (!output(answer, size_t(h.n)*sizeof(double))) return false;
        }
        return true;
    }
};

} // namespace

int main() {
    static_assert(std::endian::native == std::endian::little);
    static_assert(sizeof(int) == 4 && sizeof(double) == 8);
    workspace.prepare();
    cosine.prepare();
    if (!output("READY\n", 6)) return 1;
    Header header{};
    if (!header.read()) return 1;
    Solver s(header);
    return s.execute() ? 0 : 1;
}
