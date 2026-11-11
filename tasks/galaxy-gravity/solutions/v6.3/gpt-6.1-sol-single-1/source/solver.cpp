#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <immintrin.h>
#include <numeric>
#include <sys/resource.h>
#include <tuple>
#include <vector>

static constexpr double EPS2 = 0.0001;
static constexpr int PMAX = 18;
static int choose_count(int p) { return (p + 1) * (p + 2) * (p + 3) / 6; }
static double cpu_time() {
    rusage u;
    getrusage(RUSAGE_SELF, &u);
    return u.ru_utime.tv_sec + u.ru_utime.tv_usec * 1e-6
         + u.ru_stime.tv_sec + u.ru_stime.tv_usec * 1e-6;
}
static int env_int(const char* key, int def) {
    const char* s = std::getenv(key);
    return s ? std::atoi(s) : def;
}
static double env_double(const char* key, double def) {
    const char* s = std::getenv(key);
    return s ? std::atof(s) : def;
}

struct Particle { double m, x[3]; int id; };
struct DotPair { double a,b; };
static inline double vector_sum(__m256d a) {
    __m128d b=_mm_add_pd(_mm256_castpd256_pd128(a),_mm256_extractf128_pd(a,1));
    return _mm_cvtsd_f64(_mm_add_sd(b,_mm_unpackhi_pd(b,b)));
}
static inline DotPair packed_dot(const double* row,const double* ma,
                                 const double* mb,int q) {
    __m256d sa[4],sb[4];
    for(int t=0;t<4;++t) sa[t]=sb[t]=_mm256_setzero_pd();
    int k=0;
    for(;k+16<=q;k+=16) {
        #pragma GCC unroll 4
        for(int t=0;t<4;++t) {
            __m256d v=_mm256_loadu_pd(row+k+4*t);
            sa[t]=_mm256_fmadd_pd(v,_mm256_loadu_pd(mb+k+4*t),sa[t]);
            sb[t]=_mm256_fmadd_pd(v,_mm256_loadu_pd(ma+k+4*t),sb[t]);
        }
    }
    for(;k+4<=q;k+=4) {
        __m256d v=_mm256_loadu_pd(row+k);
        sa[0]=_mm256_fmadd_pd(v,_mm256_loadu_pd(mb+k),sa[0]);
        sb[0]=_mm256_fmadd_pd(v,_mm256_loadu_pd(ma+k),sb[0]);
    }
    __m256d a=_mm256_add_pd(_mm256_add_pd(sa[0],sa[1]),_mm256_add_pd(sa[2],sa[3]));
    __m256d b=_mm256_add_pd(_mm256_add_pd(sb[0],sb[1]),_mm256_add_pd(sb[2],sb[3]));
    DotPair result{vector_sum(a),vector_sum(b)};
    for(;k<q;++k) {result.a+=row[k]*mb[k];result.b+=row[k]*ma[k];}
    return result;
}
struct Node {
    double c[3], radius, mass;
    double cell = 1;
    int begin, end, left = -1, right = -1;
    bool leaf() const { return left < 0; }
};
#include "harmonic.hpp"
struct Index {
    int a[3], degree, sign;
    int prev[3], prev2[3];
    double c[3], b[3];
};
struct Translation {
    int target, source, mono;
};
struct Work {
    int a,b,h,p;
    bool full;
    double rx = 0, ry = 0, rz = 0, invscale = 1;
    int mh = 0, mp = 0;
    bool mp_full = false;
};

class Solver {
    int order, leafsize, nc, nm;
    double theta;
    double max_ratio, cutoff;
    int harmonic_order;
    Harmonic harmonic;
    std::vector<Particle> particles;
    std::vector<Node> nodes;
    std::vector<Index> ix;
    int map[PMAX+1][PMAX+1][PMAX+1];
    std::vector<std::vector<int>> kernel_index;
    std::vector<Translation> shifts;
    std::vector<double> moments, negative_moments, local;
    std::vector<double> packed_cart;
    std::vector<size_t> cart_offsets;
    std::vector<double> x, y, z, mass, ax, ay, az;
    std::vector<Work> work;
    std::vector<int> moment_order, local_order, hm_order, hl_order;
    bool adaptive;
    double tolerance, force_length;
    double force_center[3]{};
    int fast_direct;
    bool grid;
    bool use_m2p;
    double qfactor, sfactor;
    bool equal_mass = true, equal_kernel;
    double rotation[3][3]{{1,0,0},{0,1,0},{0,0,1}};
    bool budgeted;
    double interaction_deadline = 4.05;
    int order_drop = 0;
    bool down_cut = false;
    long long multipoles = 0, directpairs = 0;
    long long cachedcalls = 0, cachegroups = 0;
    double t_build = 0, t_up = 0, t_interact = 0, t_down = 0;

    int build_node(int begin, int end) {
        int ni = (int)nodes.size();
        nodes.emplace_back();
        Node n;
        n.begin = begin;
        n.end = end;
        double lo[3] = {INFINITY, INFINITY, INFINITY};
        double hi[3] = {-INFINITY, -INFINITY, -INFINITY};
        n.mass = 0;
        n.c[0] = n.c[1] = n.c[2] = 0;
        for (int i = begin; i < end; ++i) {
            const Particle& p = particles[i];
            n.mass += p.m;
            for (int d = 0; d < 3; ++d) {
                n.c[d] += p.m * p.x[d];
                lo[d] = std::min(lo[d], p.x[d]);
                hi[d] = std::max(hi[d], p.x[d]);
            }
        }
        for (int d = 0; d < 3; ++d) n.c[d] /= n.mass;
        double r2 = 0;
        for (int i = begin; i < end; ++i) {
            double a = particles[i].x[0] - n.c[0];
            double b = particles[i].x[1] - n.c[1];
            double c = particles[i].x[2] - n.c[2];
            r2 = std::max(r2, a*a + b*b + c*c);
        }
        n.radius = std::sqrt(r2);
        if (end - begin > leafsize) {
            int d = 0;
            if (hi[1] - lo[1] > hi[d] - lo[d]) d = 1;
            if (hi[2] - lo[2] > hi[d] - lo[d]) d = 2;
            double split = 0.5*(lo[d]+hi[d]);
            int mid = (int)(std::partition(particles.begin()+begin,
                                          particles.begin()+end,
                                          [d,split](const Particle& a) {
                                              return a.x[d] < split;
                                          }) - particles.begin());
            if (mid == begin || mid == end) mid = (begin+end)/2;
            n.left = build_node(begin, mid);
            n.right = build_node(mid, end);
        }
        nodes[ni] = n;
        return ni;
    }

    int build_grid(int begin,int end,std::array<double,3> center,
                   std::array<double,3> half,int depth) {
        int ni = (int)nodes.size();
        nodes.emplace_back();
        Node n;
        n.begin = begin; n.end = end;
        n.mass = 0;
        for (int i = begin; i < end; ++i) n.mass += particles[i].m;
        int mid = -1;
        if (end-begin > leafsize) {
            for (;;) {
                int axis = depth%3;
                double split = center[axis];
                mid = (int)(std::partition(particles.begin()+begin,particles.begin()+end,
                        [axis,split](const Particle& a) { return a.x[axis] < split; })
                        - particles.begin());
                if (mid != begin && mid != end) break;
                half[axis] *= 0.5;
                center[axis] += mid == begin ? half[axis] : -half[axis];
                ++depth;
                if (depth > 150) { mid = (begin+end)/2; break; }
            }
        }
        for (int d = 0; d < 3; ++d) n.c[d] = center[d];
        n.cell = 2*std::max({half[0],half[1],half[2]});
        double r2 = 0;
        for (int i = begin; i < end; ++i) {
            double a = particles[i].x[0]-n.c[0];
            double b = particles[i].x[1]-n.c[1];
            double c = particles[i].x[2]-n.c[2];
            r2 = std::max(r2,a*a+b*b+c*c);
        }
        n.radius = std::sqrt(r2);
        if (mid >= 0) {
            int axis = depth%3;
            half[axis] *= 0.5;
            auto lc = center, rc = center;
            lc[axis] -= half[axis]; rc[axis] += half[axis];
            n.left = build_grid(begin,mid,lc,half,depth+1);
            n.right = build_grid(mid,end,rc,half,depth+1);
        }
        nodes[ni] = n;
        return ni;
    }

    void make_indices() {
        std::memset(map, -1, sizeof(map));
        for (int d = 0; d <= order; ++d)
            for (int a = 0; a <= d; ++a)
                for (int b = 0; b <= d-a; ++b) {
                    int c = d-a-b;
                    Index t{};
                    t.a[0] = a; t.a[1] = b; t.a[2] = c;
                    t.degree = d;
                    t.sign = (d & 1) ? -1 : 1;
                    map[a][b][c] = (int)ix.size();
                    ix.push_back(t);
                }
        for (int i = 1; i < nc; ++i) {
            Index& t = ix[i];
            int axis = 0;
            if (t.a[1] > t.a[axis]) axis = 1;
            if (t.a[2] > t.a[axis]) axis = 2;
            for (int d = 0; d < 3; ++d) {
                auto a = std::array<int,3>{t.a[0], t.a[1], t.a[2]};
                if (a[d] > 0) {
                    --a[d];
                    t.prev[d] = map[a[0]][a[1]][a[2]];
                    t.c[d] = (d == axis ? 2*t.a[d]-1 : 2*t.a[d]);
                }
                if (t.a[d] > 1) {
                    --a[d];
                    t.prev2[d] = map[a[0]][a[1]][a[2]];
                    t.b[d] = d == axis ? (t.a[d]-1)*(t.a[d]-1)
                                      : t.a[d]*(t.a[d]-1);
                }
            }
        }
        kernel_index.resize(nc);
        for (int n = 1; n < nc; ++n) {
            auto& row = kernel_index[n];
            int count = choose_count(order-ix[n].degree);
            row.resize(count);
            for (int k = 0; k < count; ++k)
                row[k] = map[ix[n].a[0]+ix[k].a[0]]
                            [ix[n].a[1]+ix[k].a[1]]
                            [ix[n].a[2]+ix[k].a[2]];
        }
        for (int n = 0; n < nm; ++n)
            for (int k = 0; k <= n; ++k) {
                int a = ix[n].a[0]-ix[k].a[0];
                int b = ix[n].a[1]-ix[k].a[1];
                int c = ix[n].a[2]-ix[k].a[2];
                if (a >= 0 && b >= 0 && c >= 0)
                    shifts.push_back({n,k,map[a][b][c]});
            }
    }

    void monomials(double a, double b, double c, double* out, int p) const {
        double xx[PMAX+1], yy[PMAX+1], zz[PMAX+1];
        xx[0] = yy[0] = zz[0] = 1;
        for (int d = 1; d <= p; ++d) {
            xx[d] = xx[d-1]*a/d;
            yy[d] = yy[d-1]*b/d;
            zz[d] = zz[d-1]*c/d;
        }
        int count = choose_count(p);
        for (int i = 0; i < count; ++i)
            out[i] = xx[ix[i].a[0]] * yy[ix[i].a[1]] * zz[ix[i].a[2]];
    }

    void shift(const double* input,double* output,double dx,double dy,double dz,
               int degree,bool is_local) const {
        double aa[1330], bb[1330];
        int count = choose_count(degree);
        std::copy(input,input+count,aa);
        double* a = aa; double* b = bb;
        double delta[3] = {dx,dy,dz};
        for (int axis = 0; axis < 3; ++axis) if (delta[axis] != 0) {
            double power[PMAX+1]; power[0] = 1;
            for (int k = 1; k <= degree; ++k) power[k] = power[k-1]*delta[axis]/k;
            for (int n = 0; n < count; ++n) {
                const Index& t = ix[n];
                int limit = is_local ? degree-t.degree : t.a[axis];
                double sum = 0;
                for (int k = 0; k <= limit; ++k) {
                    int abc[3] = {t.a[0],t.a[1],t.a[2]};
                    abc[axis] += is_local ? k : -k;
                    sum += a[map[abc[0]][abc[1]][abc[2]]]*power[k];
                }
                b[n] = sum;
            }
            std::swap(a,b);
        }
        std::copy(a,a+count,output);
    }

    void upward(int ni) {
        const Node& n = nodes[ni];
        double* m = moments.data() + (size_t)ni*nm;
        double mono[1330];
        int degree = adaptive ? moment_order[ni] : order-1;
        int count = degree >= 0 ? choose_count(degree) : 0;
        if (n.leaf()) {
            if (degree < 0) return;
            for (int i = n.begin; i < n.end; ++i) {
                monomials(x[i]-n.c[0], y[i]-n.c[1], z[i]-n.c[2], mono, degree);
                for (int k = 0; k < count; ++k) m[k] += mass[i]*mono[k];
            }
        } else {
            for (int ch : {n.left, n.right}) {
                upward(ch);
                if (degree < 0) continue;
                const Node& child = nodes[ch];
                const double* cm = moments.data() + (size_t)ch*nm;
                shift(cm,mono,child.c[0]-n.c[0],child.c[1]-n.c[1],
                      child.c[2]-n.c[2],degree,false);
                for (int k = 0; k < count; ++k) m[k] += mono[k];
            }
        }
        double* mm = negative_moments.data() + (size_t)ni*nm;
        for (int k = 0; k < count; ++k) mm[k] = ix[k].sign*m[k];
    }

    static double hsum(__m256d a) {
        __m128d b = _mm_add_pd(_mm256_castpd256_pd128(a),
                              _mm256_extractf128_pd(a,1));
        b = _mm_add_sd(b, _mm_unpackhi_pd(b,b));
        return _mm_cvtsd_f64(b);
    }

    void m2l(int ia, int ib, int pair_order = -1, bool full = false,int direction = 0) {
        if (pair_order < 0) pair_order = order;
        const Node& a = nodes[ia];
        const Node& b = nodes[ib];
        double r[3] = {a.c[0]-b.c[0], a.c[1]-b.c[1], a.c[2]-b.c[2]};
        double s = r[0]*r[0] + r[1]*r[1] + r[2]*r[2] + EPS2;
        double invs = 1/s;
        double d[1330];
        double lap[1330];
        d[0] = std::sqrt(invs);
        double invlap = 1/(s-EPS2);
        lap[0] = std::sqrt(invlap);
        int count = choose_count(pair_order);
        for (int k = 1; k < count; ++k) {
            const Index& t = ix[k];
            d[k] = -invs*(r[0]*t.c[0]*d[t.prev[0]] + t.b[0]*d[t.prev2[0]]
                        +r[1]*t.c[1]*d[t.prev[1]] + t.b[1]*d[t.prev2[1]]
                        +r[2]*t.c[2]*d[t.prev[2]] + t.b[2]*d[t.prev2[2]]);
            if (!full)
                lap[k] = -invlap*(r[0]*t.c[0]*lap[t.prev[0]] + t.b[0]*lap[t.prev2[0]]
                                +r[1]*t.c[1]*lap[t.prev[1]] + t.b[1]*lap[t.prev2[1]]
                                +r[2]*t.c[2]*lap[t.prev[2]] + t.b[2]*lap[t.prev2[2]]);
        }
        if (harmonic_order && !full)
            for (int k = 0; k < count; ++k) d[k] -= lap[k];
        const double* ma = moments.data() + (size_t)ia*nm;
        const double* mb = negative_moments.data() + (size_t)ib*nm;
        double* la = local.data() + (size_t)ia*nc;
        double* lb = local.data() + (size_t)ib*nc;
        for (int n = 1; n < count; ++n) {
            const auto& row = kernel_index[n];
            int q = choose_count(pair_order-ix[n].degree);
            __m256d sa = _mm256_setzero_pd();
            __m256d sb = _mm256_setzero_pd();
            int k = 0;
            for (; k+4 <= q; k+=4) {
                __m128i index = _mm_loadu_si128((const __m128i*)(row.data()+k));
                __m256d v = _mm256_i32gather_pd(d, index, 8);
                sa = _mm256_fmadd_pd(v, _mm256_loadu_pd(mb+k), sa);
                sb = _mm256_fmadd_pd(v, _mm256_loadu_pd(ma+k), sb);
            }
            double va = hsum(sa), vb = hsum(sb);
            for (; k < q; ++k) {
                double v = d[row[k]];
                va += v*mb[k];
                vb += v*ma[k];
            }
            if (direction != 2) la[n] += va;
            if (direction != 1) lb[n] += ix[n].sign*vb;
        }
        ++multipoles;
    }

    void m2p(int ia,int ib,int source_order,bool full) {
        const Node& a = nodes[ia];
        const Node& b = nodes[ib];
        const double* mb = negative_moments.data()+(size_t)ib*nm;
        int count = choose_count(source_order+1);
        int mcount = choose_count(source_order);
        double d[1330], lap[1330];
        for (int i = a.begin; i < a.end; ++i) {
            double r[3] = {x[i]-b.c[0],y[i]-b.c[1],z[i]-b.c[2]};
            double r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
            double invs = 1/(r2+EPS2), invlap = 1/r2;
            d[0] = std::sqrt(invs); lap[0] = std::sqrt(invlap);
            for (int k = 1; k < count; ++k) {
                const Index& t = ix[k];
                d[k] = -invs*(r[0]*t.c[0]*d[t.prev[0]]+t.b[0]*d[t.prev2[0]]
                            +r[1]*t.c[1]*d[t.prev[1]]+t.b[1]*d[t.prev2[1]]
                            +r[2]*t.c[2]*d[t.prev[2]]+t.b[2]*d[t.prev2[2]]);
                if (!full)
                    lap[k] = -invlap*(r[0]*t.c[0]*lap[t.prev[0]]+t.b[0]*lap[t.prev2[0]]
                                     +r[1]*t.c[1]*lap[t.prev[1]]+t.b[1]*lap[t.prev2[1]]
                                     +r[2]*t.c[2]*lap[t.prev[2]]+t.b[2]*lap[t.prev2[2]]);
            }
            if (!full) for (int k = 0; k < count; ++k) d[k] -= lap[k];
            double acc[3]{};
            for (int k = 0; k < mcount; ++k) {
                const Index& t = ix[k];
                acc[0] += mb[k]*d[map[t.a[0]+1][t.a[1]][t.a[2]]];
                acc[1] += mb[k]*d[map[t.a[0]][t.a[1]+1][t.a[2]]];
                acc[2] += mb[k]*d[map[t.a[0]][t.a[1]][t.a[2]+1]];
            }
            ax[i] += acc[0]; ay[i] += acc[1]; az[i] += acc[2];
        }
    }

    void prepare_cart(double rx,double ry,double rz,int degree,bool full) {
        double r[3] = {rx,ry,rz};
        double r2 = rx*rx+ry*ry+rz*rz;
        double invs = 1/(r2+EPS2), invlap = 1/r2;
        double d[1330],lap[1330];
        d[0] = std::sqrt(invs); lap[0] = std::sqrt(invlap);
        int count = choose_count(degree);
        for(int k=1;k<count;++k) {
            const Index& t = ix[k];
            d[k] = -invs*(r[0]*t.c[0]*d[t.prev[0]]+t.b[0]*d[t.prev2[0]]
                        +r[1]*t.c[1]*d[t.prev[1]]+t.b[1]*d[t.prev2[1]]
                        +r[2]*t.c[2]*d[t.prev[2]]+t.b[2]*d[t.prev2[2]]);
            if(!full)
                lap[k] = -invlap*(r[0]*t.c[0]*lap[t.prev[0]]+t.b[0]*lap[t.prev2[0]]
                                +r[1]*t.c[1]*lap[t.prev[1]]+t.b[1]*lap[t.prev2[1]]
                                +r[2]*t.c[2]*lap[t.prev[2]]+t.b[2]*lap[t.prev2[2]]);
        }
        if(!full) for(int k=0;k<count;++k) d[k]-=lap[k];
        cart_offsets.resize(count);
        packed_cart.clear();
        for(int n=1;n<count;++n) {
            cart_offsets[n] = packed_cart.size();
            int q=choose_count(degree-ix[n].degree);
            for(int k=0;k<q;++k) packed_cart.push_back(d[kernel_index[n][k]]);
        }
    }

    void cached_cart(int ia,int ib,int degree,int direction) {
        const double* ma=moments.data()+(size_t)ia*nm;
        const double* mb=negative_moments.data()+(size_t)ib*nm;
        double* la=local.data()+(size_t)ia*nc;
        double* lb=local.data()+(size_t)ib*nc;
        int count=choose_count(degree);
        for(int n=1;n<count;++n) {
            const double* row=packed_cart.data()+cart_offsets[n];
            int q=choose_count(degree-ix[n].degree);
            if(q==1) {
                if(direction!=2) la[n]+=row[0]*mb[0];
                if(direction!=1) lb[n]+=ix[n].sign*row[0]*ma[0];
                continue;
            }
            DotPair sums=packed_dot(row,ma,mb,q);
            double va=sums.a,vb=sums.b;
            if(direction!=2) la[n]+=va;
            if(direction!=1) lb[n]+=ix[n].sign*vb;
        }
        ++multipoles;
    }

    void direct(int ia, int ib) {
        if (equal_mass && equal_kernel) { direct_equal(ia,ib); return; }
        const Node& a = nodes[ia];
        const Node& b = nodes[ib];
        const bool self = ia == ib;
        const __m256d eps = _mm256_set1_pd(EPS2);
        const __m256d one = _mm256_set1_pd(1.0);
        for (int i = a.begin; i < a.end; ++i) {
            __m256d xi = _mm256_set1_pd(x[i]);
            __m256d yi = _mm256_set1_pd(y[i]);
            __m256d zi = _mm256_set1_pd(z[i]);
            __m256d mi = _mm256_set1_pd(mass[i]);
            __m256d sx = _mm256_setzero_pd(), sy = sx, sz = sx;
            int j = self ? i+1 : b.begin;
            directpairs += b.end-j;
            for (; j+4 <= b.end; j+=4) {
                __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(x.data()+j),xi);
                __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(y.data()+j),yi);
                __m256d dz = _mm256_sub_pd(_mm256_loadu_pd(z.data()+j),zi);
                __m256d r2 = _mm256_fmadd_pd(dx,dx,eps);
                r2 = _mm256_fmadd_pd(dy,dy,r2);
                r2 = _mm256_fmadd_pd(dz,dz,r2);
                __m256d inv;
                if (fast_direct) {
                    inv = _mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(r2)));
                    for (int step = 0; step < fast_direct; ++step) {
                        __m256d residual = _mm256_fnmadd_pd(r2,_mm256_mul_pd(inv,inv),one);
                        inv = _mm256_fmadd_pd(_mm256_mul_pd(_mm256_set1_pd(0.5),inv),
                                              residual,inv);
                    }
                } else inv = _mm256_div_pd(one,_mm256_sqrt_pd(r2));
                __m256d f = _mm256_mul_pd(_mm256_mul_pd(inv,inv),inv);
                __m256d fj = _mm256_mul_pd(f,_mm256_loadu_pd(mass.data()+j));
                sx = _mm256_fmadd_pd(dx,fj,sx);
                sy = _mm256_fmadd_pd(dy,fj,sy);
                sz = _mm256_fmadd_pd(dz,fj,sz);
                __m256d fi = _mm256_mul_pd(f,mi);
                _mm256_storeu_pd(ax.data()+j,
                    _mm256_fnmadd_pd(dx,fi,_mm256_loadu_pd(ax.data()+j)));
                _mm256_storeu_pd(ay.data()+j,
                    _mm256_fnmadd_pd(dy,fi,_mm256_loadu_pd(ay.data()+j)));
                _mm256_storeu_pd(az.data()+j,
                    _mm256_fnmadd_pd(dz,fi,_mm256_loadu_pd(az.data()+j)));
            }
            double sx0 = hsum(sx), sy0 = hsum(sy), sz0 = hsum(sz);
            for (; j < b.end; ++j) {
                double dx = x[j]-x[i], dy = y[j]-y[i], dz = z[j]-z[i];
                double r2 = dx*dx+dy*dy+dz*dz+EPS2;
                double f = 1/(r2*std::sqrt(r2));
                sx0 += dx*f*mass[j];
                sy0 += dy*f*mass[j];
                sz0 += dz*f*mass[j];
                ax[j] -= dx*f*mass[i];
                ay[j] -= dy*f*mass[i];
                az[j] -= dz*f*mass[i];
            }
            ax[i] += sx0; ay[i] += sy0; az[i] += sz0;
        }
    }

    void direct_equal(int ia,int ib) {
        const Node& a = nodes[ia];
        const Node& b = nodes[ib];
        bool self = ia == ib;
        const __m256d eps = _mm256_set1_pd(EPS2);
        const __m256d m0 = _mm256_set1_pd(mass[0]);
        for (int i = a.begin; i < a.end; ++i) {
            __m256d xi = _mm256_set1_pd(x[i]), yi = _mm256_set1_pd(y[i]);
            __m256d zi = _mm256_set1_pd(z[i]);
            __m256d sx = _mm256_setzero_pd(), sy = sx, sz = sx;
            int j = self ? i+1 : b.begin;
            directpairs += b.end-j;
            for (; j+4 <= b.end; j+=4) {
                __m256d dx = _mm256_sub_pd(_mm256_loadu_pd(x.data()+j),xi);
                __m256d dy = _mm256_sub_pd(_mm256_loadu_pd(y.data()+j),yi);
                __m256d dz = _mm256_sub_pd(_mm256_loadu_pd(z.data()+j),zi);
                __m256d r2 = _mm256_fmadd_pd(dx,dx,eps);
                r2 = _mm256_fmadd_pd(dy,dy,r2);
                r2 = _mm256_fmadd_pd(dz,dz,r2);
                __m256d f = _mm256_div_pd(m0,_mm256_mul_pd(r2,_mm256_sqrt_pd(r2)));
                sx = _mm256_fmadd_pd(dx,f,sx);
                sy = _mm256_fmadd_pd(dy,f,sy);
                sz = _mm256_fmadd_pd(dz,f,sz);
                _mm256_storeu_pd(ax.data()+j,
                    _mm256_fnmadd_pd(dx,f,_mm256_loadu_pd(ax.data()+j)));
                _mm256_storeu_pd(ay.data()+j,
                    _mm256_fnmadd_pd(dy,f,_mm256_loadu_pd(ay.data()+j)));
                _mm256_storeu_pd(az.data()+j,
                    _mm256_fnmadd_pd(dz,f,_mm256_loadu_pd(az.data()+j)));
            }
            double sx0 = hsum(sx), sy0 = hsum(sy), sz0 = hsum(sz);
            for (; j < b.end; ++j) {
                double dx=x[j]-x[i],dy=y[j]-y[i],dz=z[j]-z[i];
                double r2=dx*dx+dy*dy+dz*dz+EPS2;
                double f=mass[0]/(r2*std::sqrt(r2));
                sx0+=dx*f;sy0+=dy*f;sz0+=dz*f;
                ax[j]-=dx*f;ay[j]-=dy*f;az[j]-=dz*f;
            }
            ax[i]+=sx0;ay[i]+=sy0;az[i]+=sz0;
        }
    }

    void closure(int ia,int ib) {
        const Node& a=nodes[ia];
        const Node& b=nodes[ib];
        auto add=[&](const Node& target,const Node& source) {
            for(int i=target.begin;i<target.end;++i) {
                double dx=source.c[0]-x[i],dy=source.c[1]-y[i],dz=source.c[2]-z[i];
                double r2=dx*dx+dy*dy+dz*dz+EPS2;
                double f=source.mass/(r2*std::sqrt(r2));
                ax[i]+=dx*f;ay[i]+=dy*f;az[i]+=dz*f;
            }
        };
        add(a,b);
        if(ia!=ib) add(b,a);
    }

    double work_cost(const Work& w) const {
        if(!w.p) return 0;
        double hc=0;
        for(int d=1;d<=w.h;++d) hc+=(2*d+1)*(w.h-d+1)*(w.h-d+1);
        double cc=1;
        for(int k=1;k<=6;++k) cc*=double(w.p+k)/k;
        cc-=choose_count(w.p);
        double result=hc*0.45e-9+cc*0.8e-9+choose_count(w.p)*8e-9+0.4e-6;
        if(w.mp) {
            double c=choose_count(w.mp+1)*10e-9+choose_count(w.mp)*3e-9;
            if(w.mh) c+=(1+1.5*(w.mh+1)*(w.mh+2))*5e-9
                         +(w.mh+1)*(w.mh+1)*3e-9;
            result+=(nodes[w.a].end-nodes[w.a].begin)*c;
        }
        return result;
    }

    Work effective_work(const Work& w) const {
        if(!budgeted || !order_drop) return w;
        Work r=w;
        if(r.h) r.h=std::max(3,r.h-order_drop);
        if(r.p) r.p=std::max(3,r.p-(order_drop+1)/2);
        if(r.mh) r.mh=std::max(2,r.mh-(order_drop+1)/2);
        if(r.mp) r.mp=std::max(2,r.mp-(order_drop+2)/3);
        return r;
    }

    void orient() {
        double cov[3][3]{};
        for (Particle& p : particles) {
            for (int d=0;d<3;++d) p.x[d]-=force_center[d];
            for (int d=0;d<3;++d)
                for (int e=0;e<3;++e) cov[d][e]+=p.m*p.x[d]*p.x[e];
        }
        for (int iter=0;iter<24;++iter) {
            int a=0,b=1;
            if (std::abs(cov[0][2])>std::abs(cov[a][b])) {a=0;b=2;}
            if (std::abs(cov[1][2])>std::abs(cov[a][b])) {a=1;b=2;}
            if (std::abs(cov[a][b])<1e-15*(cov[0][0]+cov[1][1]+cov[2][2])) break;
            double angle=0.5*std::atan2(2*cov[a][b],cov[b][b]-cov[a][a]);
            double c=std::cos(angle),s=std::sin(angle);
            for (int k=0;k<3;++k) {
                double va=cov[k][a],vb=cov[k][b];
                cov[k][a]=c*va-s*vb;cov[k][b]=s*va+c*vb;
            }
            for (int k=0;k<3;++k) {
                double va=cov[a][k],vb=cov[b][k];
                cov[a][k]=c*va-s*vb;cov[b][k]=s*va+c*vb;
            }
            for (int k=0;k<3;++k) {
                double va=rotation[k][a],vb=rotation[k][b];
                rotation[k][a]=c*va-s*vb;rotation[k][b]=s*va+c*vb;
            }
        }
        int perm[3]={0,1,2};
        std::sort(perm,perm+3,[&](int a,int b){return cov[a][a]>cov[b][b];});
        double r[3][3];
        for (int d=0;d<3;++d) for(int e=0;e<3;++e) r[d][e]=rotation[d][perm[e]];
        std::memcpy(rotation,r,sizeof(r));
        for (Particle& p : particles) {
            double v[3]{};
            for(int d=0;d<3;++d) for(int e=0;e<3;++e) v[d]+=rotation[e][d]*p.x[e];
            for(int d=0;d<3;++d) p.x[d]=v[d];
        }
        for(double& v:force_center) v=0;
    }

    void interact(int ia, int ib) {
        const Node& a = nodes[ia];
        const Node& b = nodes[ib];
        if (ia == ib) {
            if (a.leaf()) direct(ia,ib);
            else {
                interact(a.left,a.left);
                interact(a.left,a.right);
                interact(a.right,a.right);
            }
            return;
        }
        double dx = a.c[0]-b.c[0], dy = a.c[1]-b.c[1], dz = a.c[2]-b.c[2];
        double radius = a.radius+b.radius;
        double d2 = dx*dx+dy*dy+dz*dz;
        double maxrad = std::max(a.radius,b.radius);
        if (d2 > cutoff*cutoff
            && maxrad*maxrad < max_ratio*max_ratio*d2
            && radius*radius < theta*theta*(d2+EPS2)) {
            m2l(ia,ib);
            if (harmonic_order) harmonic.m2l(ia,ib);
        } else if (a.leaf() && b.leaf()) {
            direct(ia,ib);
        } else if (b.leaf() || (!a.leaf() && a.radius > b.radius)) {
            interact(a.left,ib);
            interact(a.right,ib);
        } else {
            interact(ia,b.left);
            interact(ia,b.right);
        }
    }

    int required_order(double q,double weight,double tol) const {
        if (weight <= tol) return 3;
        if (q >= 0.98) return 100;
        if (q < 1e-12) return 3;
        return std::max(3,(int)std::ceil(std::log(tol/weight)/std::log(q)));
    }

    void plan(int ia,int ib) {
        const Node& a = nodes[ia];
        const Node& b = nodes[ib];
        if (ia == ib) {
            if (a.leaf()) work.push_back({ia,ib,0,0,false});
            else {
                plan(a.left,a.left);
                plan(a.left,a.right);
                plan(a.right,a.right);
            }
            return;
        }
        double dx = a.c[0]-b.c[0], dy = a.c[1]-b.c[1], dz = a.c[2]-b.c[2];
        double d2 = dx*dx+dy*dy+dz*dz;
        double radius = a.radius+b.radius;
        auto field_scale = [&](const Node& n) {
            double xx = n.c[0]-force_center[0];
            double yy = n.c[1]-force_center[1];
            double zz = n.c[2]-force_center[2];
            double r = std::sqrt(xx*xx+yy*yy+zz*zz)+n.radius+force_length;
            return r*r/nodes[0].mass;
        };
        double distance = std::sqrt(d2);
        double rmax = std::max(a.radius,b.radius), rmin = std::min(a.radius,b.radius);
        double weight = std::max(b.mass*field_scale(a),a.mass*field_scale(b))/d2;
        double qh = (rmax+qfactor*rmin)/distance;
        double qp = (rmax+sfactor*rmin)/distance;
        double asym = (rmax-rmin)/(rmax+rmin+1e-100);
        int normal_hp = required_order(qh,weight*(1+30*asym),tolerance);
        int normal_cp = required_order(qp,weight*EPS2/d2*10,tolerance*0.5);
        bool normal_valid = radius*radius < theta*theta*d2 && d2 > cutoff*cutoff
                         && normal_hp <= harmonic_order && normal_cp <= order;
        if (!normal_valid && use_m2p && ((a.leaf() && a.radius > 2*b.radius)
                                    || (b.leaf() && b.radius > 2*a.radius))) {
            int ta = ia, sb = ib;
            if (!a.leaf() || b.radius > a.radius) { ta = ib; sb = ia; }
            const Node& target = nodes[ta];
            const Node& source = nodes[sb];
            double dmin = distance-target.radius;
            double cx = target.c[0]-force_center[0], cy = target.c[1]-force_center[1];
            double cz = target.c[2]-force_center[2];
            if (source.mass > 4*target.mass && cx*cx+cy*cy+cz*cz > 64*force_length*force_length
                && dmin > source.radius*2.5 && dmin > 0.05) {
                double q = source.radius/dmin;
                double weight = source.mass*field_scale(target)/(dmin*dmin);
                int mh = required_order(q,weight*3,tolerance*0.25);
                int mp = required_order(q,weight*EPS2/(dmin*dmin)*3,tolerance*0.25);
                double reverse_weight = target.mass*field_scale(source)/d2;
                double qr = radius/distance;
                int hp = required_order(qr,reverse_weight*20,tolerance);
                int cp = required_order(qr,reverse_weight*EPS2/d2*10,tolerance*0.5);
                if (mh < harmonic_order && mp < order && hp <= harmonic_order && cp <= order) {
                    bool full = cp >= hp;
                    if (full) hp = 0;
                    bool mp_full = mp >= mh;
                    if (mp_full) mh = 0;
                    Work w{ta,sb,hp,cp,full};
                    w.mh = mh; w.mp = mp; w.mp_full = mp_full;
                    work.push_back(w);
                    moment_order[ta] = std::max(moment_order[ta],cp-1);
                    moment_order[sb] = std::max(moment_order[sb],mp);
                    local_order[sb] = std::max(local_order[sb],cp);
                    if (hp) {
                        hm_order[ta] = std::max(hm_order[ta],hp-1);
                        hl_order[sb] = std::max(hl_order[sb],hp);
                    }
                    if (mh) hm_order[sb] = std::max(hm_order[sb],mh);
                    return;
                }
            }
        }
        if (normal_valid) {
            int hp = normal_hp;
            int cp = normal_cp;
            if (hp <= harmonic_order && cp <= order) {
                bool full = cp >= hp;
                if (full) hp = 0;
                work.push_back({ia,ib,hp,cp,full});
                moment_order[ia] = std::max(moment_order[ia],cp-1);
                moment_order[ib] = std::max(moment_order[ib],cp-1);
                local_order[ia] = std::max(local_order[ia],cp);
                local_order[ib] = std::max(local_order[ib],cp);
                if (hp) {
                    hm_order[ia] = std::max(hm_order[ia],hp-1);
                    hm_order[ib] = std::max(hm_order[ib],hp-1);
                    hl_order[ia] = std::max(hl_order[ia],hp);
                    hl_order[ib] = std::max(hl_order[ib],hp);
                }
                return;
            }
        }
        if (a.leaf() && b.leaf()) work.push_back({ia,ib,0,0,false});
        else if (b.leaf() || (!a.leaf() && a.radius > b.radius)) {
            plan(a.left,ib); plan(a.right,ib);
        } else {
            plan(ia,b.left); plan(ia,b.right);
        }
    }

    void propagate_orders(int ni) {
        const Node& n = nodes[ni];
        if (!n.leaf()) for (int ch : {n.left,n.right}) {
            moment_order[ch] = std::max(moment_order[ch],moment_order[ni]);
            local_order[ch] = std::max(local_order[ch],local_order[ni]);
            hm_order[ch] = std::max(hm_order[ch],hm_order[ni]);
            hl_order[ch] = std::max(hl_order[ch],hl_order[ni]);
            propagate_orders(ch);
        }
    }

    void downward(int ni) {
        const Node& n = nodes[ni];
        const double* l = local.data() + (size_t)ni*nc;
        double mono[1330];
        int degree = adaptive ? local_order[ni] : order;
        if(budgeted && (ni&63)==0 && cpu_time()>4.55) down_cut=true;
        if(down_cut) degree=std::min(degree,6);
        if (degree < 1) {
            if (!n.leaf()) { downward(n.left); downward(n.right); }
            return;
        }
        int count = choose_count(degree);
        int mcount = choose_count(degree-1);
        if (n.leaf()) {
            for (int i = n.begin; i < n.end; ++i) {
                monomials(x[i]-n.c[0],y[i]-n.c[1],z[i]-n.c[2],mono,degree-1);
                double a[3] = {};
                for (int k = 0; k < mcount; ++k) {
                    const Index& t = ix[k];
                    a[0] += mono[k]*l[map[t.a[0]+1][t.a[1]][t.a[2]]];
                    a[1] += mono[k]*l[map[t.a[0]][t.a[1]+1][t.a[2]]];
                    a[2] += mono[k]*l[map[t.a[0]][t.a[1]][t.a[2]+1]];
                }
                ax[i] += a[0]; ay[i] += a[1]; az[i] += a[2];
            }
        } else {
            for (int ch : {n.left,n.right}) {
                const Node& c = nodes[ch];
                double* cl = local.data() + (size_t)ch*nc;
                shift(l,mono,c.c[0]-n.c[0],c.c[1]-n.c[1],c.c[2]-n.c[2],degree,true);
                for (int k = 1; k < count; ++k) cl[k] += mono[k];
                downward(ch);
            }
        }
    }

public:
    Solver() : harmonic_order(env_int("DEV_HARMONIC",20)),
               harmonic(harmonic_order ? harmonic_order : 2) {
        order = std::max(2,std::min(PMAX,env_int("DEV_ORDER",12)));
        leafsize = std::max(4,env_int("DEV_LEAF",96));
        theta = env_double("DEV_THETA",0.85);
        max_ratio = env_double("DEV_MAXR",1e20);
        cutoff = env_double("DEV_CUTOFF",0);
        adaptive = env_int("DEV_ADAPT",1) != 0;
        tolerance = env_double("DEV_TOL",3e-8);
        fast_direct = env_int("DEV_FAST",0);
        grid = env_int("DEV_GRID",1) != 0;
        use_m2p = env_int("DEV_M2P",1) != 0;
        qfactor = env_double("DEV_QFAC",1);
        sfactor = env_double("DEV_SFAC",1);
        equal_kernel = env_int("DEV_DIRECT",1) != 0;
        budgeted = env_int("DEV_BUDGET",1) != 0;
        nc = choose_count(order);
        nm = choose_count(order-1);
        make_indices();
    }
    bool run(const char* input, const char* output) {
        FILE* f = std::fopen(input,"rb");
        if (!f) return false;
        std::fseek(f,0,SEEK_END);
        long bytes = std::ftell(f);
        std::rewind(f);
        if (bytes < 0 || bytes % 32) { std::fclose(f); return false; }
        size_t n = (size_t)bytes/32;
        std::vector<double> rows(4*n);
        if (std::fread(rows.data(),32,n,f) != n) { std::fclose(f); return false; }
        std::fclose(f);
        if (n == 0) {
            f = std::fopen(output,"wb");
            if (!f) return false;
            return std::fclose(f) == 0;
        }
        particles.resize(n);
        for (size_t i = 0; i < n; ++i) {
            particles[i].m = rows[4*i];
            for (int d = 0; d < 3; ++d) particles[i].x[d] = rows[4*i+d+1];
            particles[i].id = (int)i;
            for (int d = 0; d < 3; ++d)
                force_center[d] += particles[i].m*particles[i].x[d];
        }
        double total_mass = 0;
        for (const Particle& p : particles) total_mass += p.m;
        for (double& v : force_center) v /= total_mass;
        for (const Particle& p : particles)
            if (p.m != particles[0].m) equal_mass = false;
        if (env_int("DEV_ROTATE",1) && n > 128) orient();
        rows.clear(); rows.shrink_to_fit();
        nodes.reserve(2*n/std::max(1,leafsize/2)+64);
        double t = cpu_time();
        if (grid) {
            double extent = 0;
            for (const Particle& p : particles)
                for (int d = 0; d < 3; ++d) extent = std::max(extent,std::abs(p.x[d]));
            int exp;
            std::frexp(extent,&exp);
            double half = std::ldexp(1.0,exp);
            build_grid(0,(int)n,{0,0,0},{half,half,half},0);
        } else build_node(0,(int)n);
        x.resize(n); y.resize(n); z.resize(n); mass.resize(n);
        ax.resize(n); ay.resize(n); az.resize(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = particles[i].x[0]; y[i] = particles[i].x[1];
            z[i] = particles[i].x[2]; mass[i] = particles[i].m;
        }
        moments.resize((size_t)nodes.size()*nm);
        negative_moments.resize(moments.size());
        local.resize((size_t)nodes.size()*nc);
        if (adaptive) {
            moment_order.assign(nodes.size(),-1);
            local_order.assign(nodes.size(),0);
            hm_order.assign(nodes.size(),-1);
            hl_order.assign(nodes.size(),0);
            std::vector<double> r(n);
            for (size_t i = 0; i < n; ++i) {
                double dx = x[i]-force_center[0], dy = y[i]-force_center[1];
                double dz = z[i]-force_center[2];
                r[i] = std::sqrt(dx*dx+dy*dy+dz*dz);
            }
            std::nth_element(r.begin(),r.begin()+n/2,r.end());
            force_length = std::max(0.01,r[n/2]*0.4);
            plan(0,0);
            propagate_orders(0);
            if (grid) {
                for (Work& w : work) if (w.p) {
                    const Node& a = nodes[w.a];
                    const Node& b = nodes[w.b];
                    w.invscale = 1/std::max(a.cell,b.cell);
                    w.rx = (a.c[0]-b.c[0])*w.invscale;
                    w.ry = (a.c[1]-b.c[1])*w.invscale;
                    w.rz = (a.c[2]-b.c[2])*w.invscale;
                    if (!w.mp && std::tie(w.rx,w.ry,w.rz) < std::make_tuple(0.0,0.0,0.0)) {
                        std::swap(w.a,w.b);
                        w.rx = -w.rx; w.ry = -w.ry; w.rz = -w.rz;
                    }
                }
                std::sort(work.begin(),work.end(),[](const Work& a,const Work& b) {
                    if((a.p==0)!=(b.p==0)) return a.p==0;
                    return std::tie(a.rx,a.ry,a.rz,a.full,a.invscale)
                         < std::tie(b.rx,b.ry,b.rz,b.full,b.invscale);
                });
            }
        }
        if (harmonic_order) harmonic.init(nodes,x,y,z,mass);
        if (adaptive && harmonic_order) harmonic.set_orders(hm_order,hl_order);
        if(harmonic_order) harmonic.set_budget(budgeted);
        t_build = cpu_time()-t; t = cpu_time();
        upward(0);
        if (harmonic_order) harmonic.upward(0);
        t_up = cpu_time()-t; t = cpu_time();
        if (adaptive) {
            double down_estimate=0.06;
            for(size_t ni=0;ni<nodes.size();++ni) if(nodes[ni].leaf()) {
                int size=nodes[ni].end-nodes[ni].begin;
                down_estimate+=size*(hl_order[ni]*hl_order[ni]*6.0e-9
                               +(local_order[ni] ? choose_count(local_order[ni]-1)*4.0e-9 : 0));
            }
            interaction_deadline=4.75-std::max(0.25,down_estimate);
            std::vector<double> suffix(work.size()+1);
            for(size_t i=work.size();i>0;--i) suffix[i-1]=suffix[i]+work_cost(work[i-1]);
            double fmm_start=-1,completed_cost=0;
            for (size_t i = 0; i < work.size();) {
                if(budgeted && work[i].p) {
                    double now=cpu_time();
                    if(fmm_start<0) fmm_start=now;
                    double ratio=(now-fmm_start+0.052)/(completed_cost+0.08);
                    double remaining=std::max(0.001,interaction_deadline-now);
                    while(order_drop<16
                          && suffix[i]*ratio*std::pow(std::max(0.05,1-order_drop/18.0),4.5)
                             > remaining*0.90) ++order_drop;
                    if(now>4.60) order_drop=std::max(order_drop,6);
                    if(now>4.70) order_drop=100;
                }
                size_t j = i+1;
                if (grid && work[i].p) {
                    while (j < work.size() && work[j].p
                           && std::tie(work[i].rx,work[i].ry,work[i].rz,
                                       work[i].full)
                              == std::tie(work[j].rx,work[j].ry,work[j].rz,
                                          work[j].full)) ++j;
                }
                int hp=0;
                for(size_t k=i;k<j;++k) hp=std::max(hp,effective_work(work[k]).h);
                bool hcached=grid && hp && j-i>=4;
                if(hcached) {
                    harmonic.prepare(work[i].rx,work[i].ry,work[i].rz,hp);
                    cachedcalls+=j-i;
                    ++cachegroups;
                }
                for(size_t start=i;start<j;) {
                    size_t end=start+1;
                    while(end<j && work[end].invscale==work[start].invscale) ++end;
                    bool cached=grid && work[start].p
                             && end-start>=(size_t)env_int("DEV_CCACHE",16);
                    if(cached) {
                        int cp=0;
                        for(size_t k=start;k<end;++k) cp=std::max(cp,effective_work(work[k]).p);
                        prepare_cart(work[start].rx/work[start].invscale,
                                     work[start].ry/work[start].invscale,
                                     work[start].rz/work[start].invscale,cp,work[start].full);
                    }
                for (size_t k = start; k < end; ++k) {
                    if(budgeted && (k&127)==0 && work[k].p
                       && cpu_time()>4.60) order_drop=std::max(order_drop,6);
                    Work w = effective_work(work[k]);
                    if (w.p) {
                        if(cached) cached_cart(w.a,w.b,w.p,w.mp ? 2 : 0);
                        else m2l(w.a,w.b,w.p,w.full,w.mp ? 2 : 0);
                        if (w.h) {
                            if (hcached) harmonic.cached_m2l(w.a,w.b,w.h,w.invscale,w.mp ? 2 : 0);
                            else harmonic.m2l(w.a,w.b,w.h,w.mp ? 2 : 0);
                        }
                        if (w.mp) {
                            m2p(w.a,w.b,w.mp,w.mp_full);
                            if (w.mh) harmonic.m2p(w.a,w.b,w.mh,ax,ay,az);
                        }
                        completed_cost+=work_cost(w);
                    } else {
                        if(budgeted && cpu_time()>interaction_deadline-0.35) closure(w.a,w.b);
                        else direct(w.a,w.b);
                    }
                }
                    start=end;
                }
                i = j;
            }
        } else interact(0,0);
        t_interact = cpu_time()-t; t = cpu_time();
        downward(0);
        if (harmonic_order) harmonic.downward(0,ax,ay,az);
        t_down = cpu_time()-t;
        std::vector<double> out(3*n);
        for (size_t i = 0; i < n; ++i) {
            int j = particles[i].id;
            double a[3]={ax[i],ay[i],az[i]};
            for (int d=0;d<3;++d) {
                out[3*j+d]=0;
                for(int e=0;e<3;++e) out[3*j+d]+=rotation[d][e]*a[e];
                uint64_t bits;
                std::memcpy(&bits,&out[3*j+d],sizeof(bits));
                if((bits&UINT64_C(0x7ff0000000000000))==UINT64_C(0x7ff0000000000000))
                    out[3*j+d]=0;
            }
        }
        f = std::fopen(output,"wb");
        if (!f) return false;
        bool ok = std::fwrite(out.data(),24,n,f) == n;
        ok = (std::fclose(f) == 0) && ok;
        if (std::getenv("DEV_STATS"))
            std::fprintf(stderr,"p=%d leaf=%d theta=%.4f nodes=%zu m2l=%lld pp=%lld "
                         "cached=%lld groups=%lld drop=%d build=%.4f up=%.4f inter=%.4f down=%.4f total=%.4f\n",
                         order,leafsize,theta,nodes.size(),multipoles,directpairs,
                         cachedcalls,cachegroups,order_drop,t_build,t_up,t_interact,t_down,cpu_time());
        return ok;
    }
};

int main(int argc, char** argv) {
    if (argc != 3) return 1;
    Solver solver;
    return solver.run(argv[1],argv[2]) ? 0 : 1;
}
