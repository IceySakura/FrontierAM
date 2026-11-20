#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <immintrin.h>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>
#include <sys/stat.h>

namespace {

constexpr double eps2 = 0.0001;
constexpr int PMAX = 20;
constexpr int MAXCO = 2000;
constexpr int MAXPAD=2800;
using V = __m256d;

template<class T> struct Aligned {
    using value_type=T;
    Aligned() noexcept=default;
    template<class U> Aligned(const Aligned<U>&) noexcept {}
    T* allocate(size_t n) {
        if(!n) return nullptr;
        void *mem=nullptr;
        if(posix_memalign(&mem,64,n*sizeof(T))) throw std::bad_alloc();
        return static_cast<T*>(mem);
    }
    void deallocate(T* p,size_t) noexcept {std::free(p);}
    template<class U> bool operator==(const Aligned<U>&) const noexcept {return true;}
    template<class U> bool operator!=(const Aligned<U>&) const noexcept {return false;}
};
using DVec=std::vector<double,Aligned<double>>;
using FVec=std::vector<float,Aligned<float>>;

struct Vec {
    double x, y, z;
    Vec operator+(Vec b) const {return {x+b.x,y+b.y,z+b.z};}
    Vec operator-(Vec b) const {return {x-b.x,y-b.y,z-b.z};}
    Vec operator*(double f) const {return {x*f,y*f,z*f};}
    double norm2() const {return x*x+y*y+z*z;}
    double norm() const {return std::sqrt(norm2());}
    double operator[](int j) const {return j==0?x:(j==1?y:z);}
    double &operator[](int j) {return j==0?x:(j==1?y:z);}
};

struct Input { double m; Vec x; };
struct Particle { Vec x; int id; };
static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559);
static_assert(sizeof(Input)==32 && sizeof(Vec)==24);

struct Node {
    Vec c;
    Vec half;
    double r;
    int first, last, left, right, arena, arena_end=0;
    int mbase = -1, lbase = -1;
    int hm=-1,hl=-1;
    double uncertainty=0.;
    int size() const {return last-first;}
    bool leaf() const {return left < 0;}
};

struct Row { int x,y,len,start; };
struct Layout {
    int p=0, stride=0;
    int idx[PMAX+1][PMAX+1]={};
    std::vector<Row> rows;
    void init(int order) {
        p=order;
        for (int x=0; x<=p; ++x) {
            for (int y=0; y<=p-x; ++y) {
                int len=p+1-x-y;
                idx[x][y]=stride;
                rows.push_back({x,y,len,stride});
                stride+=len;
            }
        }
    }
};

struct PointTerm { uint16_t off,full; double invfact; uint8_t degree; };
struct GradTerm {
    uint16_t src, x,y,z;
    uint8_t degree;
};

// Translation instructions for a four-component run of local coefficients.
struct Quad {
    int out, x,y,z, count, beg;
};
struct Translation {
    int didx[PMAX+1][PMAX+1]{};
    int stride=0, count=0, zcap=0, degree_cap=0;
    std::vector<Quad> quads;
    std::vector<uint32_t> products;
};

inline double cpu() {return double(std::clock())/CLOCKS_PER_SEC;}

inline V ir3(V r2) {
    V u = _mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(r2)));
    V u2 = u*u;
    V d = _mm256_fnmadd_pd(u2,r2,_mm256_set1_pd(1.0));
    V d2=d*d;
    V q0=_mm256_fmadd_pd(_mm256_set1_pd(1.5),d,_mm256_set1_pd(1.));
    V q1=_mm256_fmadd_pd(_mm256_set1_pd(2.1875),d,_mm256_set1_pd(1.875));
    q1=_mm256_fmadd_pd(_mm256_set1_pd(2.4609375),d2,q1);
    return _mm256_fmadd_pd(q1,d2,q0)*(u2*u);
}

inline double sum4(V a) {
    __m128d b=_mm_add_pd(_mm256_castpd256_pd128(a),_mm256_extractf128_pd(a,1));
    return _mm_cvtsd_f64(_mm_add_sd(b,_mm_unpackhi_pd(b,b)));
}

struct Arena {
    static constexpr int batch=512;
    int stride=0,count=0;
    std::vector<std::unique_ptr<double[]>> blocks;
    void init(int s) {stride=s;}
    int alloc() {
        if(count%batch==0) blocks.push_back(std::make_unique<double[]>(batch*stride));
        return count++;
    }
    double *operator[](int i) const {return blocks[i/batch].get()+(i%batch)*stride;}
    size_t size() const {return size_t(count)*stride;}
    void zero() {
        for(size_t j=0;j<blocks.size();++j)
            std::fill_n(blocks[j].get(),std::min(batch,count-int(j)*batch)*stride,0.);
    }
};

// A total-degree interpolant on a lower set of nested one-dimensional nodes.
// Divided differences and the change to monomials are both separable.
// Refine the hardware reciprocal-square-root seed in double precision.
inline V rinv(V r2) {
    const V u=_mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(r2)));
    const V d=_mm256_fnmadd_pd(u*u,r2,_mm256_set1_pd(1.));
    V q=_mm256_fmadd_pd(_mm256_set1_pd(.2734375),d,_mm256_set1_pd(.3125));
    q=_mm256_fmadd_pd(q,d,_mm256_set1_pd(.375));
    q=_mm256_fmadd_pd(q,d,_mm256_set1_pd(.5));
    return _mm256_fmadd_pd(q,d,_mm256_set1_pd(1.))*u;
}

#include "harmonic.hpp"

class Solver {
    int p, bucket, active_order;
    double theta, mindirect;
    bool verbose, small_snapshot, conservative_disc;
    std::vector<Particle> part;
    std::vector<Node> nodes;
    DVec x,y,z,ax,ay,az;
    FVec lx,ly,lz;
    Arena moments, locals;
    Layout lay[PMAX+1];
    Translation trans[PMAX+1];
    Translation float_term[PMAX+1];
    bool mix_far=false;
    static constexpr int far_double=7;
    uint64_t mixed_far_ops=0;
    std::unique_ptr<Harmonic> harmonic;
    bool harm_mode=false;
    bool harm_points=false;
    static constexpr double harm_slack=.006;
    double tol=0.;
    uint64_t nharm=0;
    std::vector<PointTerm> point_terms[PMAX+1];
    std::vector<GradTerm> grad_terms[PMAX+1];
    double fi[PMAX+1]={};
    double cut2[PMAX+1]{}, cost[PMAX+1]{};
    bool float_near;
    bool hybrid_near=false;
    bool mixed_points=false;
    static constexpr int point_double=6;
    double float_gap2=0.;
    double one_cost;
    int moment_split;
    bool fit_sphere;
    std::vector<int> rev;
    std::vector<Vec> extra;
    std::array<Vec,3> frame{{{1.,0.,0.},{0.,1.,0.},{0.,0.,1.}}};
    double budget, deadline=1e30;
    bool degraded=false;
    bool calibrating=false;
    bool use_more=false;
    bool relaxed_pace=false;
    bool separate_prep=false, track_prep=false;
    double moment_cpu=0.;
    int warm_moments=0;
    int refined=0;
    double pp_factor=1., ml_factor=1., mp_factor=1.;
    double work_done=0.,forecast_work=0.;
    uint64_t nvisits=0, nshortcut=0;
    uint64_t npp=0, nml=0, nmp=0, npairs=0, nmp_p=0;
    double ttree=0, twalk=0;
    int packed=0;

    int make(int first,int last) {
        Vec lo=part[first].x, hi=lo;
        int ex_lo[3]={first,first,first},ex_hi[3]={first,first,first};
        for(int j=first+1;j<last;++j) {
            Vec a=part[j].x;
            if(fit_sphere) {
                if(a.x<lo.x) ex_lo[0]=j;
                if(a.y<lo.y) ex_lo[1]=j;
                if(a.z<lo.z) ex_lo[2]=j;
                if(a.x>hi.x) ex_hi[0]=j;
                if(a.y>hi.y) ex_hi[1]=j;
                if(a.z>hi.z) ex_hi[2]=j;
            }
            lo={std::min(lo.x,a.x),std::min(lo.y,a.y),std::min(lo.z,a.z)};
            hi={std::max(hi.x,a.x),std::max(hi.y,a.y),std::max(hi.z,a.z)};
        }
        Vec w=hi-lo, midc=(hi+lo)*0.5, c=midc;
        bool moved=false;
        double ritter=0.;
        if(fit_sphere && last-first>=9 && last-first<=32768) {
            int ext[6]={ex_lo[0],ex_lo[1],ex_lo[2],ex_hi[0],ex_hi[1],ex_hi[2]};
            double best=0.;
            Vec b0=c,b1=c;
            for(int i=0;i<6;++i) for(int j=0;j<i;++j) {
                Vec u=part[ext[i]].x,v=part[ext[j]].x;
                double q=(u-v).norm2();
                if(q>best) {best=q;b0=u;b1=v;}
            }
            Vec candidate=(b0+b1)*.5;
            double rad2=best*.25, rad=std::sqrt(rad2), standard=0.;
            for(int j=first;j<last;++j) {
                Vec q=part[j].x-c, v=part[j].x-candidate;
                standard=std::max(standard,q.norm2());
                double d=v.norm2();
                if(d>rad2) {
                    double dd=std::sqrt(d);
                    double step=(dd-rad)*.5;
                    candidate=candidate+v*(step/dd);
                    rad+=step; rad2=rad*rad;
                }
            }
            if(rad2<standard) {
                c=candidate; moved=true;
                ritter=rad*(1.0+4e-15);
            }
        }
        Vec h={std::max(hi.x-c.x,c.x-lo.x),std::max(hi.y-c.y,c.y-lo.y),
               std::max(hi.z-c.z,c.z-lo.z)};
        int at=int(nodes.size());
        double rb=h.norm();
        nodes.push_back({c,h,rb,first,last,-1,-1,0});
        int n=last-first;
        bool scan=n<=32768;
        if(scan) {
            double r2=0.;
            for(int j=first;j<last;++j) {
                double vv=(part[j].x-c).norm2();
                r2=std::max(r2,vv);
            }
            auto& a=nodes[at];
            a.r=moved?std::min(ritter,std::sqrt(r2)):std::sqrt(r2);
        }
        if (n<=bucket) {
            return at;
        }
        int axis=w.x>w.y?(w.x>w.z?0:2):(w.y>w.z?1:2);
        double plane=midc[axis];
        auto b=part.begin();
        int mid=int(std::partition(b+first,b+last,[&](const Particle& a) {
            return a.x[axis]<plane;
        })-b);
        if (mid==first || mid==last) {
            mid=first+(last-first)/2;
            std::nth_element(b+first,b+mid,b+last,[axis](const Particle& a,const Particle& b) {
                return a.x[axis]<b.x[axis];
            });
        }
        int l=make(first,mid), r=make(mid,last);
        auto &parent=nodes[at];
        parent.left=l; parent.right=r;
        if(!scan) {
            parent.r=std::min(parent.r,std::max((nodes[l].c-c).norm()+nodes[l].r,
                                               (nodes[r].c-c).norm()+nodes[r].r));
        }
        return at;
    }

    void pack() {
        int sz=int(part.size())+int(nodes.size())*3/2+12;
        if(float_near) sz+=int(nodes.size())*2+8;
        x.resize(sz); y.resize(sz); z.resize(sz); rev.resize(sz,-1);
        ax.resize(sz); ay.resize(sz); az.resize(sz);
        if(float_near) {
            lx.resize(sz); ly.resize(sz); lz.resize(sz);
        }
        int off=0;
        for (auto& n:nodes) {
            if (!n.leaf()) continue;
            n.arena=off;
            for (int j=n.first; j<n.last; ++j,++off) {
                auto a=part[j];
                x[off]=a.x.x; y[off]=a.x.y; z[off]=a.x.z; rev[off]=a.id;
                if(float_near) {
                    lx[off]=float(a.x.x-n.c.x); ly[off]=float(a.x.y-n.c.y); lz[off]=float(a.x.z-n.c.z);
                }
            }
            for (;off%(float_near?8:4);++off) {
                x[off]=x[off-1]; y[off]=y[off-1]; z[off]=z[off-1];
                if(float_near) {lx[off]=lx[off-1];ly[off]=ly[off-1];lz[off]=lz[off-1];}
            }
            n.arena_end=off;
        }
        for(int i=int(nodes.size())-1;i>=0;--i) {
            auto& a=nodes[i];
            if(!a.leaf()) {
                a.arena=nodes[a.left].arena;
                a.arena_end=nodes[a.right].arena_end;
            }
        }
        packed=off;
    }

    template<bool Same>
    void direct0(const Node& a,const Node& b) {
        int i0=a.arena, i1=i0+a.size(), j0=b.arena, j1=j0+b.size();
        ++npp;
        npairs+=uint64_t(a.size())*b.size();
        for (int i=i0; i<i1; ++i) {
            V bx=_mm256_set1_pd(x[i]), by=_mm256_set1_pd(y[i]), bz=_mm256_set1_pd(z[i]);
            V sx=_mm256_setzero_pd(), sy=sx, sz=sx;
            int end=Same?i:j1;
            int j=j0;
            for (; j+4<=end; j+=4) {
                V dx=_mm256_loadu_pd(&x[j])-bx, dy=_mm256_loadu_pd(&y[j])-by,
                  dz=_mm256_loadu_pd(&z[j])-bz;
                V rr=_mm256_fmadd_pd(dz,dz,_mm256_fmadd_pd(dy,dy,
                         _mm256_fmadd_pd(dx,dx,_mm256_set1_pd(eps2))));
                V f=ir3(rr);
                V fx=dx*f, fy=dy*f, fz=dz*f;
                sx+=fx; sy+=fy; sz+=fz;
                _mm256_storeu_pd(&ax[j],_mm256_loadu_pd(&ax[j])-fx);
                _mm256_storeu_pd(&ay[j],_mm256_loadu_pd(&ay[j])-fy);
                _mm256_storeu_pd(&az[j],_mm256_loadu_pd(&az[j])-fz);
            }
            if(j<end) {
                V dx=_mm256_loadu_pd(&x[j])-bx, dy=_mm256_loadu_pd(&y[j])-by,
                  dz=_mm256_loadu_pd(&z[j])-bz;
                V rr=_mm256_fmadd_pd(dz,dz,_mm256_fmadd_pd(dy,dy,
                         _mm256_fmadd_pd(dx,dx,_mm256_set1_pd(eps2))));
                V mask=_mm256_castsi256_pd(_mm256_cmpgt_epi64(_mm256_set1_epi64x(end-j),
                                                                _mm256_setr_epi64x(0,1,2,3)));
                V f=_mm256_and_pd(mask,ir3(rr));
                V fx=dx*f, fy=dy*f, fz=dz*f;
                sx+=fx; sy+=fy; sz+=fz;
                _mm256_storeu_pd(&ax[j],_mm256_loadu_pd(&ax[j])-fx);
                _mm256_storeu_pd(&ay[j],_mm256_loadu_pd(&ay[j])-fy);
                _mm256_storeu_pd(&az[j],_mm256_loadu_pd(&az[j])-fz);
            }
            ax[i]+=sum4(sx); ay[i]+=sum4(sy); az[i]+=sum4(sz);
        }
    }
    void direct_float(const Node& a,const Node& b) {
        ++npp;
        npairs+=uint64_t(a.size())*b.size();
        const int i0=a.arena, j0=b.arena, i1=i0+a.size(), jn=b.size();
        alignas(32) float tx[1040]{},ty[1040]{},tz[1040]{};
        const __m256 ep=_mm256_set1_ps(float(eps2));
        const __m256 hp=_mm256_set1_ps(0.5f),one5=_mm256_set1_ps(1.5f);
        const __m256i lane=_mm256_setr_epi32(0,1,2,3,4,5,6,7);
        for(int i=i0;i<i1;++i) {
            __m256 xi=_mm256_set1_ps(float(x[i]-b.c.x)),
                   yi=_mm256_set1_ps(float(y[i]-b.c.y)),
                   zi=_mm256_set1_ps(float(z[i]-b.c.z));
            __m256 fx=_mm256_setzero_ps(),fy=fx,fz=fx;
            for(int k=0;k<jn;k+=8) {
                int j=j0+k;
                __m256 dx=_mm256_loadu_ps(lx.data()+j)-xi,dy=_mm256_loadu_ps(ly.data()+j)-yi,
                       dz=_mm256_loadu_ps(lz.data()+j)-zi;
                __m256 q=_mm256_fmadd_ps(dz,dz,_mm256_fmadd_ps(dy,dy,_mm256_fmadd_ps(dx,dx,ep)));
                __m256 u=_mm256_rsqrt_ps(q);
                u*= _mm256_fnmadd_ps(q*hp,u*u,one5);
                __m256 w=u*u*u;
                if(k+8>jn) {
                    __m256i mask=_mm256_cmpgt_epi32(_mm256_set1_epi32(jn-k),lane);
                    w=_mm256_and_ps(w,_mm256_castsi256_ps(mask));
                }
                __m256 xx=dx*w,yy=dy*w,zz=dz*w;
                fx+=xx; fy+=yy; fz+=zz;
                _mm256_store_ps(tx+k,_mm256_load_ps(tx+k)-xx);
                _mm256_store_ps(ty+k,_mm256_load_ps(ty+k)-yy);
                _mm256_store_ps(tz+k,_mm256_load_ps(tz+k)-zz);
            }
            auto reduce=[](__m256 a) {
                return sum4(_mm256_cvtps_pd(_mm256_castps256_ps128(a))+
                            _mm256_cvtps_pd(_mm256_extractf128_ps(a,1)));
            };
            ax[i]+=reduce(fx); ay[i]+=reduce(fy); az[i]+=reduce(fz);
        }
        for(int j=0;j<(jn+3)/4*4;j+=4) {
            _mm256_storeu_pd(ax.data()+j0+j,_mm256_loadu_pd(ax.data()+j0+j)+_mm256_cvtps_pd(_mm_load_ps(tx+j)));
            _mm256_storeu_pd(ay.data()+j0+j,_mm256_loadu_pd(ay.data()+j0+j)+_mm256_cvtps_pd(_mm_load_ps(ty+j)));
            _mm256_storeu_pd(az.data()+j0+j,_mm256_loadu_pd(az.data()+j0+j)+_mm256_cvtps_pd(_mm_load_ps(tz+j)));
        }
    }
    bool use_float_pair(const Node& a,const Node& b) const {
        if(!float_near || std::min(a.size(),b.size())<12) return false;
        if(!hybrid_near) return true;
        Vec r=b.c-a.c,h=a.half+b.half;
        double dx=std::max(0.,std::abs(r.x)-h.x),
               dy=std::max(0.,std::abs(r.y)-h.y),
               dz=std::max(0.,std::abs(r.z)-h.z);
        return (dx*dx+dy*dy+dz*dz)>float_gap2;
    }
    void direct(const Node& a,const Node& b) {
        if(a.arena==b.arena) direct0<true>(a,a);
        else if(use_float_pair(a,b)) {
            if(a.size()<b.size()) direct_float(a,b); else direct_float(b,a);
        }
        else if(a.size()<=b.size()) direct0<false>(a,b);
        else direct0<false>(b,a);
    }

    static void build_translation(Translation& tr,const Layout& l,int zcap,int degree_cap) {
        int k=l.p;
        tr.zcap=zcap;
        tr.degree_cap=degree_cap;
        for(auto r:l.rows) {
            tr.didx[r.x][r.y]=tr.stride;
            int n=std::min(r.len,zcap+1);
            tr.count+=n;
            tr.stride+=n+3;
        }
        for(auto a:l.rows) for(int z0=0;z0<std::min({a.len,zcap+1,degree_cap-a.x-a.y+1});z0+=4) {
            int rem=k-a.x-a.y-z0;
            int rem0=std::min(rem,degree_cap);
            int first=int(tr.products.size());
            for(int bx=0;bx<=rem0;++bx) for(int by=0;by<=rem0-bx;++by) {
                int src=l.idx[bx][by];
                int der=tr.didx[a.x+bx][a.y+by]+z0;
                for(int z1=0;z1<=std::min(rem0-bx-by,zcap-z0);++z1)
                    tr.products.push_back(uint32_t(src+z1)|(uint32_t(der+z1)<<16));
            }
            tr.quads.push_back({0,a.x,a.y,z0,int(tr.products.size())-first,first});
        }
    }

    void build_float_translation(Translation& tr,const Layout& l,int min_order) const {
        int k=l.p;
        tr.zcap=k;
        tr.degree_cap=k;
        for(auto r:l.rows) {
            tr.didx[r.x][r.y]=tr.stride;
            tr.count+=r.len;
            tr.stride+=r.len+7;
        }
        for(auto a:l.rows) for(int z0=0;z0<a.len;z0+=8) {
            int start=int(tr.products.size());
            int rem=k-a.x-a.y-z0;
            int zhigh=std::min(z0+7,a.len-1);
            for(int bx=0;bx<=rem;++bx) for(int by=0;by<=rem-bx;++by) {
                int src=l.idx[bx][by];
                int der=tr.didx[a.x+bx][a.y+by]+z0;
                int min_z=std::max(0,min_order+1-a.x-a.y-bx-by-zhigh);
                for(int z1=min_z;z1<=rem-bx-by;++z1)
                    tr.products.push_back(uint32_t(src+z1)|(uint32_t(der+z1)<<16));
            }
            int ct=int(tr.products.size())-start;
            if(ct) tr.quads.push_back({0,a.x,a.y,z0,ct,start});
        }
    }

    void init_tables() {
        fi[0]=1.0;
        for(int i=1;i<=PMAX;++i) fi[i]=fi[i-1]/i;
        for(int k=1;k<=p;++k) {
            auto& l=lay[k];
            l.init(k);
            auto &tr=trans[k];
            build_translation(tr,l,k,k);
            if(mix_far && k>=far_double+3) build_float_translation(float_term[k],l,far_double);
            cost[k]=mindirect*double(tr.products.size()+6*l.stride);
        }
        moments.init(lay[p].stride);
        locals.init(lay[p].stride);
        for(int k=1;k<=p;++k) for(auto &a:trans[k].quads)
            a.out=lay[p].idx[a.x][a.y]+a.z;
        if(mix_far) for(int k=far_double+3;k<=p;++k) for(auto &a:float_term[k].quads)
            a.out=lay[p].idx[a.x][a.y]+a.z;
        for(int k=1;k<=p;++k) {
            const auto& l=lay[k];
            const auto& f=lay[p];
            for(auto r:l.rows) for(int z=0;z<r.len;++z) {
                if(r.x+r.y+z!=0)
                    point_terms[k].push_back({uint16_t(r.start+z),
                        uint16_t(f.idx[r.x][r.y]+z),fi[r.x]*fi[r.y]*fi[z],
                        uint8_t(r.x+r.y+z)});
            }
            for(auto r:l.rows) for(int j=0;j+1<r.len;++j) {
                grad_terms[k].push_back({uint16_t(f.idx[r.x][r.y]+j),
                    uint16_t(l.idx[r.x+1][r.y]+j),uint16_t(l.idx[r.x][r.y+1]+j),
                    uint16_t(r.start+j+1),uint8_t(r.x+r.y+j+1)});
            }
        }
        set_theta(theta);
        if(harmonic) harmonic->init(lay,p);
    }

    void set_theta(double t) {
        theta=t;
        tol=std::pow(theta,active_order);
        for(int k=1;k<=active_order;++k) {
            cut2[k]=std::pow(theta,double(2*active_order)/k);
        }
    }

    int order(double q2,double r2) const {
        if(q2>=cut2[active_order]*r2) return 0;
        double v=q2/r2;
        return int(std::upper_bound(cut2+1,cut2+active_order,v)-cut2);
    }

    int pair_order(const Node& a,const Node& b,double r2) const {
        double q=a.r+b.r;
        return order(q*q,r2);
    }
    static bool split_a(const Node& a,const Node& b) {
        return b.leaf() || (!a.leaf() && a.r>=b.r);
    }

    int point_order(const Node& target,const Node& source,int& pc) const {
        pc=0;
        if((!target.leaf() && target.size()>2*bucket) ||
           source.size()<target.size() || source.size()<32) return 0;
        Vec d=source.c-target.c;
        double dx=std::max(0.,std::abs(d.x)-target.half.x),
               dy=std::max(0.,std::abs(d.y)-target.half.y),
               dz=std::max(0.,std::abs(d.z)-target.half.z);
        double r2=dx*dx+dy*dy+dz*dz+eps2;
        int k=order(source.r*source.r,r2);
        if(!k) return 0;
        if(harm_points && k>=6 && r2>=.012) {
            double rr=r2-eps2,u=source.r/std::sqrt(rr);
            double v=(eps2/rr)*u*u;
            for(int j=2;j<=k-2;++j) {
                if(v*(1+.28*j*j)<harm_slack*tol) {
                    if(point_time(k,target.size(),j)<.94*point_time(k,target.size()))
                        pc=j;
                    break;
                }
                v*=u;
            }
        }
        double sz=pc?1.07*harmonic->lay[k].stride+1.34*lay[pc].stride:lay[k].stride;
        if(source.size()>=one_cost*(10.+sz)) return k;
        pc=0;
        return 0;
    }

    double trans_time(int k) const {
        const auto &t=trans[k];
        if(mix_far && k>=far_double+3) {
            double base=1.02e-9*trans[far_double].products.size()+7.6e-9*lay[far_double].stride;
            return ml_factor*(base+1.03e-9*float_term[k].products.size()+4.2e-9*t.count+3.5e-7);
        }
        return ml_factor*(1.02e-9*t.products.size()+
               7.6e-9*t.count+2.4e-7);
    }
    int correction_order(const Node& a,const Node& b,int k,double r2) const {
        if(!harmonic || k<5 || r2<0.008) return 0;
        double rr=r2-eps2, q=a.r+b.r, u2=q*q/rr;
        if(u2>0.86) return 0;
        double x=std::sqrt(u2), y=(eps2/rr)*u2;
        for(int j=2;j<=k-2;++j) {
            if(y*(1.+.28*j*j)<harm_slack*tol) {
                double reduced=harmonic_time(k,j);
                return reduced<0.97*trans_time(k)?j:0;
            }
            y*=x;
        }
        return 0;
    }
    double harmonic_time(int k,int j) const {
        return ml_factor*(1.1e-9*harmonic->work(k)+3e-9*lay[k].stride)+trans_time(j);
    }
    double point_time(int k,int n,int pc=0) const {
        double cc=pc?(2.0e-9*harmonic->lay[k].stride+2.9e-9*lay[pc].stride):
            (mixed_points && k>=point_double+3?
                 1.17e-9*lay[k].stride+2.15e-9*lay[point_double].stride:
                 2.15e-9*lay[k].stride);
        return mp_factor*(cc*n+1.5e-7);
    }
    double direct_time(const Node& a,const Node& b,uint64_t pairs,bool same) const {
        double slope=(same?3.05e-9:(use_float_pair(a,b)?1.40e-9:2.65e-9));
        return pp_factor*(slope*pairs+1e-7);
    }

    int choose_point(const Node& a,const Node& b,int& dir,int& pc,int km,int kh=0) const {
        dir=0;
        pc=0;
        if(km && km<5) return 0;
        int k=point_order(a,b,pc);
        double mt=!km?1e99:(kh?harmonic_time(km,kh):trans_time(km));
        if(k && point_time(k,a.size(),pc)<0.95*mt) return k;
        dir=1;
        k=point_order(b,a,pc);
        if(k && point_time(k,b.size(),pc)<0.95*mt) return k;
        pc=0;
        return 0;
    }

    struct Job {
        int i,j,tag,k,kh;
        double cost;
    };
    struct Estimate {
        double predicted=0;
        double limit=0;
        uint64_t leaves=0, pairs=0, ml=0;
        uint64_t k[PMAX+1]{};
        std::array<std::vector<Job>,3>* jobs=nullptr;
        unsigned mask=0;
        void add(int i,int j,int tag,int k0,int k1,double amount) {
            predicted+=amount;
            if(!jobs) return;
            uint32_t h=(uint32_t(i)+1)*0x85ebca6bu^(uint32_t(j)+1)*0xc2b2ae35u;
            h=(h^(h>>17))*0x7feb352du;
            auto& group=(*jobs)[tag];
            if(((h^(h>>15))&mask)==0 && group.size()<640)
                group.push_back({i,j,tag,k0,k1,amount});
        }
    };
    void estimate(int i,int j,Estimate &e) const {
        if(e.predicted>e.limit) return;
        const auto &a=nodes[i], &b=nodes[j];
        if(i==j) {
            if(a.leaf()) {
                uint64_t pairs=uint64_t(a.size())*a.size()/2;
                e.pairs+=pairs;
                ++e.leaves;
                e.add(i,j,0,0,0,direct_time(a,b,pairs,true));
            } else {
                estimate(a.left,a.left,e);
                estimate(a.left,a.right,e);
                estimate(a.right,a.right,e);
            }
            return;
        }
        Vec r=b.c-a.c;
        uint64_t pairs=uint64_t(a.size())*b.size();
        double r2=r.norm2()+eps2;
        int k=pair_order(a,b,r2);
        int hc=k?correction_order(a,b,k,r2):0;
        if(k && double(pairs)<(hc?std::min(cost[k],4.1e8*harmonic_time(k,hc)):cost[k])) k=hc=0;
        int dir,pc;
        int kp=choose_point(a,b,dir,pc,k,hc);
        if(kp) {
            e.add(dir?j:i,dir?i:j,2,kp,pc,point_time(kp,dir?b.size():a.size(),pc));
            return;
        }
        if(k) {
            ++e.ml; ++e.k[k];
            e.add(i,j,1,k,hc,hc?harmonic_time(k,hc):trans_time(k));
            return;
        }
        if(a.leaf() && b.leaf()) {
            e.pairs+=pairs;
            ++e.leaves;
            e.add(i,j,0,0,0,direct_time(a,b,pairs,false));
        } else if(split_a(a,b)) {
            estimate(a.left,j,e); estimate(a.right,j,e);
        } else {
            estimate(i,b.left,e); estimate(i,b.right,e);
        }
    }

    void sample_job(const Job& q) {
        auto& a=nodes[q.i];
        auto& b=nodes[q.j];
        if(q.tag==0) {
            direct(a,b);
        } else if(q.tag==2) {
            if(q.kh) point_harmonic(a,b,q.k,q.kh);
            else point_translate(a,b,q.k);
        } else {
            Vec r=b.c-a.c;
            double r2=r.norm2()+eps2;
            if(q.kh) translate_harmonic(q.i,q.j,r,r2,q.k,q.kh);
            else translate(q.i,q.j,r,1./std::sqrt(r2),q.k);
        }
    }

    void calibrate() {
        // Time a small deterministic selection of the current snapshot's
        // interactions. Force accumulators are reset before the real walk.
        calibrating=true;
        std::array<std::vector<Job>,3> jobs;
        Estimate e;
        const double start=cpu();
        const int n=int(part.size());
        e.jobs=&jobs;
        e.mask=n<100000 ? 127u : (n<1000000 ? 511u : 2047u);
        e.limit=0.85*budget;
        set_theta(n<100000?.61:.62);
        estimate(0,0,e);
        double m[3]{1.,1.,1.};
        for(int tag=0;tag<3;++tag) {
            auto& group=jobs[tag];
            if(group.size()<10) continue;
            double predicted=0.;
            int active=0;
            for(auto q:group) {
                sample_job(q);
                predicted+=q.cost;
                ++active;
                if(active%32==0 && cpu()-start>budget*.075) break;
            }
            if(active<10 || predicted<1e-5) continue;
            int reps=std::clamp(int(.0011/predicted)+1,1,5);
            const double t0=cpu();
            for(int r=0;r<reps;++r)
                for(int k=0;k<active;++k) sample_job(group[k]);
            double observed=(cpu()-t0)/reps;
            double buffer=small_snapshot?(tag==0?1.13:1.12):(tag==0?1.075:1.085);
            m[tag]=std::clamp(buffer*observed/predicted,0.59,1.40);
            m[tag]=std::round(80.*m[tag])/80.;
        }
        std::fill(ax.begin(),ax.end(),0.);
        std::fill(ay.begin(),ay.end(),0.);
        std::fill(az.begin(),az.end(),0.);
        locals.zero();
        if(harmonic) harmonic->locals.zero();
        npp=npairs=nml=nmp=nmp_p=nharm=mixed_far_ops=0;
        pp_factor=m[0]; ml_factor=m[1]; mp_factor=m[2];
        calibrating=false;
        if(verbose) std::fprintf(stderr,"timing calibration %.4fs; direct %.3f, mutual %.3f, point %.3f\n",
            cpu()-start,pp_factor,ml_factor,mp_factor);
    }

    void tune() {
        const int n=int(part.size());
        double tail=0.019+double(n)*(0.15e-6+
                    (conservative_disc?1.65e-9:1.31e-9)*lay[p].stride);
        deadline=(conservative_disc?.963:.966)*budget-tail;
        if(n>=16384) calibrate();
        const double start=cpu();
        double allowance=((small_snapshot?.939:.941)*budget-start-tail);
        if(use_more) allowance*=n>=1048576?1.145:1.075;
        double hi=.92, lo=.26;
        for(int trial=0;trial<6;++trial) {
            double t=(trial==0? .64:0.5*(lo+hi));
            set_theta(t);
            Estimate e;
            e.limit=std::max(0.015,allowance-(cpu()-start));
            estimate(0,0,e);
            if(verbose) std::fprintf(stderr,"plan theta=%.5f cost=%.4f limit=%.4f "
                                           "direct=%llu ml=%llu\n",t,e.predicted,e.limit,
                                           (unsigned long long)e.pairs,(unsigned long long)e.ml);
            if(e.predicted>e.limit) lo=t;
            else {hi=t; forecast_work=e.predicted;}
            if(cpu()-start>.081*budget) break;
        }
        set_theta(std::min(.96,hi+0.0025));
    }

    static void powers(double *a,double x,int n) {
        a[0]=1.0;
        for(int i=1;i<=n;++i) a[i]=a[i-1]*x;
    }

    static double intpow(double x,int k) {
        double v=1.;
        do {
            if(k&1) v*=x;
            x*=x;
        } while((k>>=1)>0);
        return v;
    }
    void note_uncertainty(Node& a,Node& b, double inv,int k,bool point) {
        if(calibrating) return;
        // RMS proxy, used only to prioritize extra work. The force itself
        // always comes from a direct sum or an expansion of the real kernel.
        double v0=(.72*(a.r+b.r))*inv, v1=v0;
        if(point) v0=v1=.76*b.r*inv;
        double denom=inv*inv*(k+1.);
        double ua=b.size()*denom*intpow(v0,k), ub=a.size()*denom*intpow(v1,k);
        a.uncertainty+=ua*ua;
        b.uncertainty+=ub*ub;
    }

    void build_moment(int id) {
        if(nodes[id].mbase>=0) return;
        if(separate_prep && track_prep) {
            double t=cpu();
            build_moment_inner(id);
            moment_cpu+=cpu()-t;
        } else build_moment_inner(id);
    }

    void build_moment_inner(int id) {
        auto& a=nodes[id];
        if(a.mbase>=0) return;
        const auto& l=lay[p];
        a.mbase=moments.alloc();
        double *dst=moments[a.mbase];
        double scale=1.0/std::max(a.r,1e-100);
        if (a.size()<moment_split || a.leaf()) {
            for(int j=a.first;j<a.last;++j) {
                Vec u=(part[j].x-a.c)*scale;
                double xp[PMAX+1],yp[PMAX+1],zp[PMAX+1];
                powers(xp,u.x,p); powers(yp,u.y,p); powers(zp,u.z,p);
                for(auto r:l.rows) {
                    double f=xp[r.x]*yp[r.y];
                    double *d=dst+r.start;
                    for(int k=0;k<r.len;++k) d[k]+=f*zp[k];
                }
            }
            for(auto r:l.rows) {
                double fac=fi[r.x]*fi[r.y];
                for(int k=0;k<r.len;++k) dst[r.start+k]*=fac*fi[k];
            }
            return;
        }
        for (int ch:{a.left,a.right}) {
            build_moment_inner(ch);
            dst=moments[a.mbase];
            const auto& b=nodes[ch];
            double v=std::max(b.r,1e-100)*scale;
            Vec shift=(b.c-a.c)*scale;
            double vp[PMAX+1],xp[PMAX+1],yp[PMAX+1],zp[PMAX+1];
            powers(vp,v,p); powers(xp,shift.x,p); powers(yp,shift.y,p); powers(zp,shift.z,p);
            for(int k=0;k<=p;++k) {xp[k]*=fi[k]; yp[k]*=fi[k]; zp[k]*=fi[k];}
            alignas(32) double t1[MAXCO],t2[MAXCO];
            for(auto r:l.rows) {
                const double *src=moments[b.mbase]+r.start;
                double *d=t1+r.start;
                for(int k=0;k<r.len;++k) {
                    double sum=0;
                    for(int j=0;j<=k;++j) sum+=zp[k-j]*vp[r.x+r.y+j]*src[j];
                    d[k]=sum;
                }
            }
            for(auto r:l.rows) {
                double *d=t2+r.start;
                std::fill(d,d+r.len,0.0);
                for(int y0=0;y0<=r.y;++y0) {
                    double f=yp[r.y-y0];
                    const double *s=t1+l.idx[r.x][y0];
                    for(int k=0;k<r.len;++k) d[k]+=f*s[k];
                }
            }
            for(auto r:l.rows) {
                double *d=dst+r.start;
                for(int x0=0;x0<=r.x;++x0) {
                    double f=xp[r.x-x0];
                    const double *s=t2+l.idx[x0][r.y];
                    for(int k=0;k<r.len;++k) d[k]+=f*s[k];
                }
            }
        }
    }

    void derivatives(double *d, Vec r, int k) const {
        rows_deriv<double>(d,lay[k],r.x,r.y,r.z,1.,1.);
    }

    template<typename T>
    static inline T splat(double x) {
        if constexpr(sizeof(T)==sizeof(double)) return x;
        else return _mm256_set1_pd(x);
    }
    template<typename T>
    static void rows_deriv(T* d,const Layout& l,T x,T y,T z,T quad,T d0,int zcap=PMAX) {
        // Differentiate (1+2 v.u+q|u|^2)^(-1/2) in lexicographic rows.
        // Entries are derivatives, with the factorials applied at translation.
        T x1[PMAX+1],y1[PMAX+1],z1[PMAX+1],p2[PMAX+1],z2[PMAX+1];
        for(int j=1;j<=l.p;++j) {
            x1[j]=x*splat<T>(-2.*j); y1[j]=y*splat<T>(-2.*j);
            z1[j]=z*splat<T>(1.-2.*j);
            p2[j]=quad*splat<T>(-j*(j-1.));
            z2[j]=quad*splat<T>(-(j-1.)*(j-1.));
        }
        const T zero=splat<T>(0.);
        T zeros[PMAX+1];
        std::fill(zeros,zeros+l.p+1,zero);
        x1[0]=y1[0]=p2[0]=zero;
        for(auto r:l.rows) {
            T *out=d+r.start;
            const T *sx1=r.x?d+l.idx[r.x-1][r.y]:zeros,
                    *sx2=r.x>1?d+l.idx[r.x-2][r.y]:zeros,
                    *sy1=r.y?d+l.idx[r.x][r.y-1]:zeros,
                    *sy2=r.y>1?d+l.idx[r.x][r.y-2]:zeros;
            T A=x1[r.x],B=y1[r.y],C=p2[r.x],D=p2[r.y];
            T v;
            if(r.y) {
                v=(A*sx1[0]+C*sx2[0])+y*splat<T>(1.-2.*r.y)*sy1[0]+z2[r.y]*sy2[0];
            } else if(r.x) {
                v=x*splat<T>(1.-2.*r.x)*sx1[0]+z2[r.x]*sx2[0];
            } else {
                v=d0;
            }
            out[0]=v;
            T prev=zero;
            int count=std::min(r.len,zcap+1);
            for(int j=1;j<count;++j) {
                T sum=(A*sx1[j]+C*sx2[j])+(B*sy1[j]+D*sy2[j]);
                T next=z1[j]*v+z2[j]*prev+sum;
                out[j]=next;
                prev=v; v=next;
            }
        }
    }

    void convolve2(const double* der, Node& a, Node& b, double sa,double sb,double inv,int k) {
        const auto &l=lay[k], &f=lay[p];
        const auto &tr=trans[k];
        if(a.lbase<0) a.lbase=locals.alloc();
        if(b.lbase<0) b.lbase=locals.alloc();
        double *la=locals[a.lbase], *lb=locals[b.lbase];
        const double *ma=moments[a.mbase], *mb=moments[b.mbase];
        alignas(32) double d[MAXPAD],wa[MAXCO],wb[MAXCO];
        double pa[PMAX+5]{},pb[PMAX+5]{};
        double ipa[PMAX+5]{},ipb[PMAX+5]{};
        powers(pa,sa,k); powers(pb,sb,k);
        for(int j=0;j<=k;++j) {ipa[j]=inv*pa[j]; ipb[j]=inv*pb[j];}
        for(auto r:l.rows) {
            const double *t=der+r.start;
            int dr=tr.didx[r.x][r.y];
            int n=std::min(r.len,tr.zcap+1);
            std::copy(t,t+n,d+dr);
            d[dr+n]=d[dr+n+1]=d[dr+n+2]=0.;
            int ar=f.idx[r.x][r.y], q=r.x+r.y;
            int src_len=std::min(n,tr.degree_cap+1-q);
            for(int j=0;j<src_len;++j) {
                wa[r.start+j]=pa[q+j]*ma[ar+j];
                wb[r.start+j]=pb[q+j]*mb[ar+j];
            }
        }
        for(const auto &a0:tr.quads) {
            const auto *s=tr.products.data()+a0.beg;
            int j=0;
            V sa0=_mm256_setzero_pd(), sb0=sa0, sa1=sa0, sb1=sa0;
            for(;j+1<a0.count;j+=2) {
                uint32_t code=s[j], code1=s[j+1];
                V der0=_mm256_loadu_pd(d+(code>>16)), der1=_mm256_loadu_pd(d+(code1>>16));
                sa0=_mm256_fmadd_pd(der0,_mm256_set1_pd(wb[code&65535]),sa0);
                sb0=_mm256_fmadd_pd(der0,_mm256_set1_pd(wa[code&65535]),sb0);
                sa1=_mm256_fmadd_pd(der1,_mm256_set1_pd(wb[code1&65535]),sa1);
                sb1=_mm256_fmadd_pd(der1,_mm256_set1_pd(wa[code1&65535]),sb1);
            }
            if(j<a0.count) {
                uint32_t code=s[j];
                V t=_mm256_loadu_pd(d+(code>>16));
                sa0=_mm256_fmadd_pd(t,_mm256_set1_pd(wb[code&65535]),sa0);
                sb0=_mm256_fmadd_pd(t,_mm256_set1_pd(wa[code&65535]),sb0);
            }
            sa0+=sa1; sb0+=sb1;
            const int ord=a0.x+a0.y+a0.z;
            const int len=std::min({4,k+1-ord,tr.zcap+1-a0.z,tr.degree_cap+1-ord});
            double fact=fi[a0.x]*fi[a0.y];
            alignas(32) double ta[4],tb[4];
            _mm256_store_pd(ta,sa0);
            _mm256_store_pd(tb,sb0);
            for(int t=0;t<len;++t) {
                if(ord+t==0) continue;
                double fa=fact*fi[a0.z+t];
                la[a0.out+t]+=ta[t]*ipa[ord+t]*fa;
                lb[a0.out+t]+=tb[t]*ipb[ord+t]*fa;
            }
        }
    }

    void convolve_mixed(const double* der,Node& a,Node& b,double sa,double sb,double inv,int k) {
        ++mixed_far_ops;
        const auto &l=lay[k],&f=lay[p];
        const auto &tr=float_term[k];
        const auto &lo=lay[far_double];
        alignas(32) double dl[MAXCO];
        for(auto r:lo.rows) std::copy_n(der+l.idx[r.x][r.y],r.len,dl+r.start);
        convolve2(dl,a,b,sa,sb,inv,far_double);
        double *la=locals[a.lbase],*lb=locals[b.lbase];
        const double *ma=moments[a.mbase],*mb=moments[b.mbase];
        alignas(32) float pad[3800],wa[MAXCO],wb[MAXCO];
        const float* d=pad;
        double pa[PMAX+5],pb[PMAX+5];
        powers(pa,sa,k);powers(pb,sb,k);
        for(auto r:l.rows) {
            int o=f.idx[r.x][r.y],ord=r.x+r.y;
            {
                float *dest=pad+tr.didx[r.x][r.y];
                int n=std::clamp(far_double+1-ord,0,r.len);
                std::fill_n(dest,n,0.f);
                for(int j=n;j<r.len;++j) dest[j]=float(der[r.start+j]);
                std::fill_n(dest+r.len,7,0.f);
            }
            for(int j=0;j<r.len;++j) {
                wa[r.start+j]=float(pa[ord+j]*ma[o+j]);
                wb[r.start+j]=float(pb[ord+j]*mb[o+j]);
            }
        }
        for(auto q:tr.quads) {
            __m256 aa=_mm256_setzero_ps(),bb=aa,aa1=aa,bb1=aa;
            const uint32_t* u=tr.products.data()+q.beg;
            int j=0;
            for(;j+1<q.count;j+=2) {
                uint32_t s=u[j],t=u[j+1];
                __m256 z0=_mm256_loadu_ps(d+(s>>16)),z1=_mm256_loadu_ps(d+(t>>16));
                aa=_mm256_fmadd_ps(z0,_mm256_set1_ps(wb[s&65535]),aa);
                bb=_mm256_fmadd_ps(z0,_mm256_set1_ps(wa[s&65535]),bb);
                aa1=_mm256_fmadd_ps(z1,_mm256_set1_ps(wb[t&65535]),aa1);
                bb1=_mm256_fmadd_ps(z1,_mm256_set1_ps(wa[t&65535]),bb1);
            }
            if(j<q.count) {
                uint32_t s=u[j];
                __m256 z0=_mm256_loadu_ps(d+(s>>16));
                aa=_mm256_fmadd_ps(z0,_mm256_set1_ps(wb[s&65535]),aa);
                bb=_mm256_fmadd_ps(z0,_mm256_set1_ps(wa[s&65535]),bb);
            }
            alignas(32) float ac[8],bc[8];
            _mm256_store_ps(ac,aa+aa1);_mm256_store_ps(bc,bb+bb1);
            int n=q.x+q.y+q.z;
            double base=inv*fi[q.x]*fi[q.y];
            for(int t=(n==0);t<std::min(8,k+1-n);++t) {
                double v=base*fi[q.z+t];
                la[q.out+t]+=double(ac[t])*pa[n+t]*v;
                lb[q.out+t]+=double(bc[t])*pb[n+t]*v;
            }
        }
    }

    void translate(int i,int j, Vec r, double inv,int k) {
        auto &a=nodes[i], &b=nodes[j];
        ++nml;
        note_uncertainty(a,b,inv,k,false);
        double aa=std::max(a.r,1e-100)*inv, bb=std::max(b.r,1e-100)*inv;
        build_moment(i);
        build_moment(j);
        alignas(32) double der[MAXCO];
        derivatives(der,r*inv,k);
        if(mix_far && k>=far_double+3) convolve_mixed(der,a,b,-aa,bb,inv,k);
        else convolve2(der,a,b,-aa,bb,inv,k);
    }

    void translate_harmonic(int i,int j, Vec r,double r2,int k,int small) {
        // Reuse the reduced Laplace expansion only for separated cells.
        // The lower-degree correction is the difference of softened and
        // unsoftened derivatives, not an unsoftened replacement kernel.
        auto& a=nodes[i];
        auto& b=nodes[j];
        ++nml; ++nharm;
        build_moment(i);build_moment(j);
        harmonic->build(a,moments[a.mbase]);
        harmonic->build(b,moments[b.mbase]);
        double inv=1./std::sqrt(r2-eps2);
        note_uncertainty(a,b,1./std::sqrt(r2),k,false);
        double w=(r2-eps2)/r2;
        Vec v=r*inv;
        double sa=-std::max(a.r,1e-100)*inv,sb=std::max(b.r,1e-100)*inv;
        alignas(32) double der[MAXCO],rem[MAXCO];
        rows_deriv<double>(der,lay[k],v.x,v.y,v.z,1.,1.);
        harmonic->apply(lay[k],der,a,b,sa,sb,inv);
        const auto& dl=lay[small];
        rows_deriv<double>(rem,dl,w*v.x,w*v.y,w*v.z,w,std::sqrt(w));
        for(auto r:dl.rows) {
            int j=lay[k].idx[r.x][r.y];
            for(int t=0;t<r.len;++t) rem[r.start+t]-=der[j+t];
        }
        convolve2(rem,a,b,sa,sb,inv,small);
    }

    void point_exact_part(Node& t,Node& s,int k) {
        int lo=t.arena,hi=t.arena_end;
        build_moment(int(&s-nodes.data()));
        if(s.lbase<0) s.lbase=locals.alloc();
        double *local=locals[s.lbase];
        const double *moment=moments[s.mbase];
        double rs=std::max(s.r,1e-100);
        alignas(32) V d[MAXCO],acc[MAXCO];
        for(const auto& m:point_terms[k]) acc[m.off]=_mm256_setzero_pd();
        for(int j=lo;j<hi;j+=4) {
            V rx=_mm256_set1_pd(s.c.x)-_mm256_loadu_pd(x.data()+j),
              ry=_mm256_set1_pd(s.c.y)-_mm256_loadu_pd(y.data()+j),
              rz=_mm256_set1_pd(s.c.z)-_mm256_loadu_pd(z.data()+j);
            V rho2=_mm256_fmadd_pd(rz,rz,_mm256_fmadd_pd(ry,ry,
                                  _mm256_fmadd_pd(rx,rx,_mm256_set1_pd(eps2))));
            V s2=_mm256_set1_pd(rs)/rho2;
            V vx=s2*rx,vy=s2*ry,vz=s2*rz, b0=s2*_mm256_set1_pd(rs);
            V mask=_mm256_castsi256_pd(_mm256_cvtepi32_epi64(_mm_cmpgt_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(rev.data()+j)),_mm_set1_epi32(-1))));
            d[0]=_mm256_and_pd(rinv(rho2),mask);
            V fx=_mm256_setzero_pd(),fy=fx,fz=fx;
            rows_deriv<V>(d,lay[k],vx,vy,vz,b0,d[0]);
            for(int jj=1;jj<lay[k].stride;++jj) acc[jj]+=d[jj];
            for(const auto& m:grad_terms[k]) {
                V f=_mm256_set1_pd(moment[m.src]);
                fx=_mm256_fnmadd_pd(d[m.x],f,fx);
                fy=_mm256_fnmadd_pd(d[m.y],f,fy);
                fz=_mm256_fnmadd_pd(d[m.z],f,fz);
            }
            V inv=_mm256_set1_pd(1./rs);
            _mm256_storeu_pd(ax.data()+j,_mm256_fmadd_pd(fx,inv,_mm256_loadu_pd(ax.data()+j)));
            _mm256_storeu_pd(ay.data()+j,_mm256_fmadd_pd(fy,inv,_mm256_loadu_pd(ay.data()+j)));
            _mm256_storeu_pd(az.data()+j,_mm256_fmadd_pd(fz,inv,_mm256_loadu_pd(az.data()+j)));
        }
        for(const auto& m:point_terms[k]) local[m.full]+=m.invfact*sum4(acc[m.off]);
    }

    static void float_derivatives(__m256* d,const Layout& l,
                                  __m256 x,__m256 y,__m256 z,__m256 quad,__m256 d0) {
        __m256 x1[PMAX+1],y1[PMAX+1],z1[PMAX+1],p2[PMAX+1],z2[PMAX+1],zeros[PMAX+1];
        const __m256 zero=_mm256_setzero_ps();
        x1[0]=y1[0]=p2[0]=zero;
        std::fill(zeros,zeros+l.p+1,zero);
        for(int j=1;j<=l.p;++j) {
            x1[j]=x*_mm256_set1_ps(float(-2*j)); y1[j]=y*_mm256_set1_ps(float(-2*j));
            z1[j]=z*_mm256_set1_ps(float(1-2*j));
            p2[j]=quad*_mm256_set1_ps(float(-j*(j-1)));
            z2[j]=quad*_mm256_set1_ps(float(-(j-1)*(j-1)));
        }
        for(auto r:l.rows) {
            __m256 *out=d+r.start;
            const __m256 *sx1=r.x?d+l.idx[r.x-1][r.y]:zeros,
                         *sx2=r.x>1?d+l.idx[r.x-2][r.y]:zeros,
                         *sy1=r.y?d+l.idx[r.x][r.y-1]:zeros,
                         *sy2=r.y>1?d+l.idx[r.x][r.y-2]:zeros;
            __m256 a=x1[r.x],b=y1[r.y],c=p2[r.x],e=p2[r.y],v;
            if(r.y) v=(a*sx1[0]+c*sx2[0])+y*_mm256_set1_ps(float(1-2*r.y))*sy1[0]+z2[r.y]*sy2[0];
            else if(r.x) v=x*_mm256_set1_ps(float(1-2*r.x))*sx1[0]+z2[r.x]*sx2[0];
            else v=d0;
            out[0]=v;
            __m256 prev=zero;
            for(int j=1;j<r.len;++j) {
                __m256 sum=(a*sx1[j]+c*sx2[j])+(b*sy1[j]+e*sy2[j]);
                __m256 next=z1[j]*v+z2[j]*prev+sum;
                out[j]=next;prev=v;v=next;
            }
        }
    }

    void point_float_tail(const Node& t,Node& s,int k) {
        alignas(32) __m256 d[MAXCO],acc[MAXCO];
        std::fill(acc,acc+lay[k].stride,_mm256_setzero_ps());
        const double *mom=moments[s.mbase];
        const double rscale=std::max(1e-100,s.r);
        const __m256 sv=_mm256_set1_ps(float(rscale));
        const V inv=_mm256_set1_pd(1./rscale);
        const __m256 ep=_mm256_set1_ps(float(eps2));
        auto sub=[](double pos,const DVec& a,int j) {
            V p=_mm256_set1_pd(pos);
            auto lo=_mm256_cvtpd_ps(p-_mm256_loadu_pd(a.data()+j));
            auto hi=_mm256_cvtpd_ps(p-_mm256_loadu_pd(a.data()+j+4));
            return _mm256_insertf128_ps(_mm256_castps128_ps256(lo),hi,1);
        };
        auto add=[&](DVec& a,int j,int last,__m256 u) {
            V lo=_mm256_cvtps_pd(_mm256_castps256_ps128(u));
            _mm256_storeu_pd(a.data()+j,_mm256_fmadd_pd(lo,inv,_mm256_loadu_pd(a.data()+j)));
            if(j+4<last) {
                V hi=_mm256_cvtps_pd(_mm256_extractf128_ps(u,1));
                _mm256_storeu_pd(a.data()+j+4,_mm256_fmadd_pd(hi,inv,_mm256_loadu_pd(a.data()+j+4)));
            }
        };
        const __m256i lane=_mm256_setr_epi32(0,1,2,3,4,5,6,7);
        for(int j=t.arena;j<t.arena_end;j+=8) {
            __m256 rx=sub(s.c.x,x,j),ry=sub(s.c.y,y,j),rz=sub(s.c.z,z,j);
            __m256 q=_mm256_fmadd_ps(rz,rz,_mm256_fmadd_ps(ry,ry,_mm256_fmadd_ps(rx,rx,ep)));
            __m256 u=_mm256_rsqrt_ps(q);
            u*=_mm256_fnmadd_ps(q*_mm256_set1_ps(.5f),u*u,_mm256_set1_ps(1.5f));
            __m256 m=sv*(u*u);
            __m256i keep=_mm256_cmpgt_epi32(_mm256_loadu_si256(
                                       reinterpret_cast<const __m256i*>(rev.data()+j)),_mm256_set1_epi32(-1));
            keep=_mm256_and_si256(keep,_mm256_cmpgt_epi32(_mm256_set1_epi32(t.arena_end-j),lane));
            __m256 d0=_mm256_and_ps(u,_mm256_castsi256_ps(keep));
            float_derivatives(d,lay[k],m*rx,m*ry,m*rz,sv*m,d0);
            __m256 f0=_mm256_setzero_ps(),f1=f0,f2=f0;
            for(const auto& g:grad_terms[k]) if(g.degree>point_double) {
                __m256 f=_mm256_set1_ps(float(mom[g.src]));
                f0=_mm256_fnmadd_ps(f,d[g.x],f0);
                f1=_mm256_fnmadd_ps(f,d[g.y],f1);
                f2=_mm256_fnmadd_ps(f,d[g.z],f2);
            }
            for(const auto& g:point_terms[k]) if(g.degree>point_double) acc[g.off]+=d[g.off];
            add(ax,j,t.arena_end,f0);add(ay,j,t.arena_end,f1);add(az,j,t.arena_end,f2);
        }
        double *local=locals[s.lbase];
        for(const auto& g:point_terms[k]) if(g.degree>point_double) {
            __m256 a=acc[g.off];
            V q=_mm256_cvtps_pd(_mm256_castps256_ps128(a))+
                _mm256_cvtps_pd(_mm256_extractf128_ps(a,1));
            local[g.full]+=sum4(q)*g.invfact;
        }
    }

    void point_translate(Node& t,Node& s,int k) {
        ++nmp; nmp_p+=t.size();
        note_uncertainty(t,s,1./std::sqrt((s.c-t.c).norm2()+eps2),k,true);
        if(mixed_points && k>=point_double+3) {
            point_exact_part(t,s,point_double);
            point_float_tail(t,s,k);
        } else point_exact_part(t,s,k);
    }

    void point_harmonic(Node& t,Node& s,int k,int kc) {
        ++nmp; nmp_p+=t.size();
        double rz2=(s.c-t.c).norm2()+eps2;
        note_uncertainty(t,s,1./std::sqrt(rz2),k,true);
        build_moment(int(&s-nodes.data()));
        harmonic->build(s,moments[s.mbase]);
        if(s.lbase<0) s.lbase=locals.alloc();
        if(s.hl<0) s.hl=harmonic->locals.alloc();
        double *hf=harmonic->locals[s.hl],*cf=locals[s.lbase];
        const double *hm=harmonic->moments[s.hm],*cm=moments[s.mbase];
        const auto& hl=harmonic->lay[k];
        const auto& large=harmonic->lay[p];
        const auto& cl=lay[kc];
        const double sr=std::max(s.r,1e-100);
        V sv=_mm256_set1_pd(sr), is=_mm256_set1_pd(1./sr);
        alignas(32) V hd[512],ha[512],cd[MAXCO],ca[MAXCO],dn[MAXCO];
        std::fill(ha,ha+hl.stride,_mm256_setzero_pd());
        std::fill(ca,ca+cl.stride,_mm256_setzero_pd());
        for(int j=t.arena;j<t.arena_end;j+=4) {
            V rx=_mm256_set1_pd(s.c.x)-_mm256_loadu_pd(x.data()+j),
              ry=_mm256_set1_pd(s.c.y)-_mm256_loadu_pd(y.data()+j),
              rz=_mm256_set1_pd(s.c.z)-_mm256_loadu_pd(z.data()+j);
            V r2=_mm256_fmadd_pd(rz,rz,_mm256_fmadd_pd(ry,ry,rx*rx));
            V r2s=r2+_mm256_set1_pd(eps2);
            V hn=sv/r2,cn=sv/r2s;
            V mask=_mm256_castsi256_pd(_mm256_cvtepi32_epi64(_mm_cmpgt_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(rev.data()+j)),_mm_set1_epi32(-1))));
            V r0=_mm256_and_pd(rinv(r2),mask);
            V r1=_mm256_and_pd(rinv(r2s),mask);
            harmonic->point_coefficients(hd,k,hn*rx,hn*ry,hn*rz,sv*hn,r0);
            for(int q=1;q<hl.stride;++q) ha[q]+=hd[q];
            for(auto r:cl.rows) {
                dn[r.start]=hd[hl.idx[0][r.x]+r.y];
                if(r.len>1) dn[r.start+1]=hd[hl.idx[1][r.x]+r.y];
            }
            for(int zz=2;zz<=kc;++zz) for(int xx=0;xx<=kc-zz;++xx)
                for(int yy=0;yy<=kc-zz-xx;++yy)
                    dn[cl.idx[xx][yy]+zz]=-dn[cl.idx[xx+2][yy]+zz-2]-dn[cl.idx[xx][yy+2]+zz-2];
            rows_deriv<V>(cd,cl,cn*rx,cn*ry,cn*rz,sv*cn,r1);
            for(int q=1;q<cl.stride;++q) {
                cd[q]-=dn[q];
                ca[q]+=cd[q];
            }
            V ax0=_mm256_setzero_pd(),ay0=ax0,az0=ax0;
            harmonic->point_gradient(hd,hm,k,ax0,ay0,az0);
            for(const auto& m:grad_terms[kc]) {
                V u=_mm256_set1_pd(cm[m.src]);
                ax0=_mm256_fnmadd_pd(u,cd[m.x],ax0);
                ay0=_mm256_fnmadd_pd(u,cd[m.y],ay0);
                az0=_mm256_fnmadd_pd(u,cd[m.z],az0);
            }
            _mm256_storeu_pd(ax.data()+j,_mm256_fmadd_pd(ax0,is,_mm256_loadu_pd(ax.data()+j)));
            _mm256_storeu_pd(ay.data()+j,_mm256_fmadd_pd(ay0,is,_mm256_loadu_pd(ay.data()+j)));
            _mm256_storeu_pd(az.data()+j,_mm256_fmadd_pd(az0,is,_mm256_loadu_pd(az.data()+j)));
        }
        for(auto r:hl.rows) {
            int dest=large.idx[r.e][r.x];
            for(int jj=(r.e+r.x==0);jj<r.len;++jj)
                hf[dest+jj]+=sum4(ha[r.start+jj])*fi[r.x]*fi[jj];
        }
        for(const auto& m:point_terms[kc]) cf[m.full]+=m.invfact*sum4(ca[m.off]);
    }

    void quick_pair(int i,int j) {
        ++nshortcut;
        const auto &a=nodes[i], &b=nodes[j];
        if(i==j) {
            for(int n=a.first;n<a.last;++n) {
                Vec r=a.c-part[n].x;
                double d=r.norm2()+eps2;
                extra[n]=extra[n]+r*((a.size()-1)/(d*std::sqrt(d)));
            }
        } else {
            Vec r=b.c-a.c;
            double q=r.norm2()+eps2;
            Vec f=r*(1.0/(q*std::sqrt(q)));
            node_extra[i]=node_extra[i]+f*b.size();
            node_extra[j]=node_extra[j]-f*a.size();
        }
    }

    std::vector<Vec> node_extra;
    using Pending = std::vector<std::pair<int,int>>;

    void walk_pair(int i,int j,Pending& pending) {
        if(degraded) {
            quick_pair(i,j);
            return;
        }
        auto &a=nodes[i], &b=nodes[j];
        if(i==j) {
            if(a.leaf()) {
                work_done+=direct_time(a,a,uint64_t(a.size())*a.size()/2,true);
                direct(a,a);
            }
            else {
                pending.emplace_back(a.right,a.right);
                pending.emplace_back(a.left,a.right);
                pending.emplace_back(a.left,a.left);
            }
            return;
        }
        const Vec r=b.c-a.c;
        double v=r.norm2()+eps2;
        int k=pair_order(a,b,v);
        int hc=k?correction_order(a,b,k,v):0;
        if(k && double(a.size())*b.size()<(hc?std::min(cost[k],4.1e8*harmonic_time(k,hc)):cost[k])) k=hc=0;
        int dir,pc;
        int kp=choose_point(a,b,dir,pc,k,hc);
        if(kp) {
            work_done+=point_time(kp,dir?b.size():a.size(),pc);
            if(pc) {
                if(dir) point_harmonic(b,a,kp,pc); else point_harmonic(a,b,kp,pc);
            } else {
                if(dir) point_translate(b,a,kp); else point_translate(a,b,kp);
            }
            return;
        }
        if(k) {
            work_done+=hc?harmonic_time(k,hc):trans_time(k);
            if(hc) translate_harmonic(i,j,r,v,k,hc);
            else translate(i,j,r,1.0/std::sqrt(v),k);
            return;
        }
        if(a.leaf() && b.leaf()) {
            work_done+=direct_time(a,b,uint64_t(a.size())*b.size(),false);
            direct(a,b);
        } else if(split_a(a,b)) {
            pending.emplace_back(a.right,j);
            pending.emplace_back(a.left,j);
        } else {
            pending.emplace_back(i,b.right);
            pending.emplace_back(i,b.left);
        }
    }

    void pace(const Pending& pending,double rate) {
        // Moment construction is paid once per node. Forecast it separately
        // instead of charging the early construction rate to every later pair.
        int made=moments.count-warm_moments;
        double prep=(separate_prep && made>0)?
             std::clamp(moment_cpu*(nodes.size()-moments.count)/made,0.,.055*budget):0.;
        const double soft=deadline-(relaxed_pace?.016:.041)*budget-prep;
        for(int attempt=0;attempt<3;++attempt) {
            double now=cpu(), allowance=(soft-now)/rate;
            if(allowance<=.028*budget) return;
            Estimate rest;
            rest.limit=allowance*1.30;
            for(auto i=pending.rbegin();i!=pending.rend() && rest.predicted<rest.limit;++i)
                estimate(i->first,i->second,rest);
            forecast_work=work_done+(rest.predicted>=rest.limit?1.7*rest.predicted:rest.predicted);
            now=cpu();
            allowance=(soft-now)/rate;
            if(rest.predicted<=allowance || allowance<=0.) return;
            double old=theta;
            if(theta>=.83 && active_order>6) {
                active_order=std::max(6,active_order-3);
                set_theta(.77);
            } else {
                set_theta(std::min(.925,theta*std::clamp(
                    std::pow(rest.predicted/std::max(1e-8,allowance),.35),1.012,1.125)));
            }
            if(verbose) std::fprintf(stderr,"budget guard: theta %.6f -> %.6f at %.4fs (order %d)\n",
                                    old,theta,now,active_order);
            if(theta==old) return;
        }
    }
    void walk() {
        Pending pending;
        pending.reserve(128);
        pending.emplace_back(0,0);
        double next_check=ttree+(small_snapshot?.155:.105)*budget, last_cpu=ttree,last_work=0.;
        double last_prep=0.;
        warm_moments=moments.count;
        track_prep=true;
        while(!pending.empty()) {
            if((++nvisits&511)==0 && !degraded) {
                double now=cpu(),remaining=deadline-now;
                if(remaining<=0.) {
                    degraded=true;
                    extra.resize(part.size());
                    node_extra.resize(nodes.size());
                } else if(now>=next_check && work_done>last_work+1e-6) {
                    double prep=separate_prep?moment_cpu-last_prep:0.;
                    const double rate=std::clamp((relaxed_pace?1.04:1.07)*
                         std::max(0.,now-last_cpu-prep)/(work_done-last_work),
                         separate_prep?.72:.77,2.4);
                    double predicted=rate*std::max(0.,forecast_work-work_done);
                    if(remaining>.112*budget) {
                        if(predicted>(relaxed_pace?.89:(small_snapshot?.865:.79))*remaining ||
                           forecast_work<=work_done)
                            pace(pending,rate);
                        last_cpu=cpu(); last_work=work_done;
                        last_prep=moment_cpu;
                        next_check=last_cpu+.135*budget;
                    } else {
                        if(predicted>.95*remaining || forecast_work<=work_done) {
                            // Do not spend the last reserve on a geometry scan.
                            active_order=std::min(active_order,7);
                            set_theta(std::max(theta,.76));
                        }
                        last_cpu=now; last_work=work_done;
                        last_prep=moment_cpu;
                        next_check=now+.036*budget;
                    }
                }
            }
            auto job=pending.back();
            pending.pop_back();
            walk_pair(job.first,job.second,pending);
        }
        track_prep=false;
    }

    void shift_local(const Node& a,Node& b) {
        const auto &l=lay[p];
        if(b.lbase<0) {
            b.lbase=locals.alloc();
        }
        const double *src=locals[a.lbase];
        double *dst=locals[b.lbase];
        double par=std::max(a.r,1e-100);
        Vec sh=(b.c-a.c)*(1.0/par);
        double ratio=std::max(b.r,1e-100)/par;
        double xp[PMAX+1],yp[PMAX+1],zp[PMAX+1],rs[PMAX+1];
        powers(xp,sh.x,p); powers(yp,sh.y,p); powers(zp,sh.z,p); powers(rs,ratio,p);
        double bin[PMAX+1][PMAX+1];
        for(int i=0;i<=p;++i) for(int j=0;j<=i;++j) bin[i][j]=fi[j]*fi[i-j]/fi[i];
        alignas(32) double t1[MAXCO],t2[MAXCO];
        for(auto r:l.rows) {
            double *t=t1+r.start;
            for(int k=0;k<r.len;++k) {
                double sum=0;
                for(int j=k;j<r.len;++j) sum+=src[r.start+j]*bin[j][k]*zp[j-k];
                t[k]=sum;
            }
        }
        for(auto r:l.rows) {
            double *t=t2+r.start;
            std::fill(t,t+r.len,0.);
            for(int y1=r.y;y1<=p-r.x;++y1) {
                double f=bin[y1][r.y]*yp[y1-r.y];
                const double *s=t1+l.idx[r.x][y1];
                for(int k=0;k<=p-r.x-y1;++k) t[k]+=f*s[k];
            }
        }
        for(auto r:l.rows) {
            double *t=dst+r.start;
            for(int x1=r.x;x1<=p-r.y;++x1) {
                double f=bin[x1][r.x]*xp[x1-r.x];
                const double *s=t2+l.idx[x1][r.y];
                for(int k=0;k<=p-x1-r.y;++k) t[k]+=f*s[k]*rs[r.x+r.y+k];
            }
        }
    }

    void push_local(int id,Vec add) {
        auto &a=nodes[id];
        if(a.hl>=0) {
            if(a.lbase<0) a.lbase=locals.alloc();
            harmonic->inflate(lay[p],a,locals[a.lbase]);
        }
        if(degraded) add=add+node_extra[id];
        if(!a.leaf()) {
            if(a.lbase>=0) {
                shift_local(a,nodes[a.left]);
                shift_local(a,nodes[a.right]);
            }
            push_local(a.left,add); push_local(a.right,add);
            return;
        }
        if(degraded) {
            for(int j=0;j<a.size();++j) {
                Vec q=add+extra[a.first+j];
                ax[a.arena+j]+=q.x; ay[a.arena+j]+=q.y; az[a.arena+j]+=q.z;
            }
        }
        if(a.lbase<0) return;
        const auto &l=lay[p];
        const double *c=locals[a.lbase];
        double s=1.0/std::max(a.r,1e-100);
        {
            V sc=_mm256_set1_pd(s);
            const V cx=_mm256_set1_pd(a.c.x),cy=_mm256_set1_pd(a.c.y),cz=_mm256_set1_pd(a.c.z);
            for(int j=a.arena;j<a.arena+a.size();j+=4) {
                V ux=(_mm256_loadu_pd(x.data()+j)-cx)*sc,
                  uy=(_mm256_loadu_pd(y.data()+j)-cy)*sc,
                  uz=(_mm256_loadu_pd(z.data()+j)-cz)*sc;
                V val=_mm256_setzero_pd(),gx=val,gy=val,gz=val;
                for(int ix=p;ix>=0;--ix) {
                    V q=_mm256_setzero_pd(),dv=q,dw=q;
                    for(int iy=p-ix;iy>=0;--iy) {
                        int rem=p-ix-iy;
                        const double *co=c+l.idx[ix][iy];
                        V r=_mm256_set1_pd(co[rem]),dr=_mm256_setzero_pd();
                        for(int iz=rem-1;iz>=0;--iz) {
                            dr=_mm256_fmadd_pd(dr,uz,r);
                            r=_mm256_fmadd_pd(r,uz,_mm256_set1_pd(co[iz]));
                        }
                        dv=_mm256_fmadd_pd(dv,uy,q);
                        dw=_mm256_fmadd_pd(dw,uy,dr);
                        q=_mm256_fmadd_pd(q,uy,r);
                    }
                    gx=_mm256_fmadd_pd(gx,ux,val);
                    gy=_mm256_fmadd_pd(gy,ux,dv);
                    gz=_mm256_fmadd_pd(gz,ux,dw);
                    val=_mm256_fmadd_pd(val,ux,q);
                }
                _mm256_storeu_pd(ax.data()+j,_mm256_fmadd_pd(gx,sc,_mm256_loadu_pd(ax.data()+j)));
                _mm256_storeu_pd(ay.data()+j,_mm256_fmadd_pd(gy,sc,_mm256_loadu_pd(ay.data()+j)));
                _mm256_storeu_pd(az.data()+j,_mm256_fmadd_pd(gz,sc,_mm256_loadu_pd(az.data()+j)));
            }
            return;
        }
    }

    static double dot(Vec a,Vec b) {return a.x*b.x+a.y*b.y+a.z*b.z;}

    Vec direct_target(int i,int lo,int hi) const {
        V tx=_mm256_set1_pd(x[i]),ty=_mm256_set1_pd(y[i]),tz=_mm256_set1_pd(z[i]);
        V u=_mm256_setzero_pd(),v=u,w=u;
        for(int j=lo;j<hi;j+=4) {
            V dx=_mm256_loadu_pd(x.data()+j)-tx,dy=_mm256_loadu_pd(y.data()+j)-ty,
              dz=_mm256_loadu_pd(z.data()+j)-tz;
            V r2=_mm256_fmadd_pd(dz,dz,_mm256_fmadd_pd(dy,dy,
                _mm256_fmadd_pd(dx,dx,_mm256_set1_pd(eps2))));
            auto keep=_mm_cmpgt_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(rev.data()+j)),
                                     _mm_set1_epi32(-1));
            V f=_mm256_and_pd(ir3(r2),_mm256_castsi256_pd(_mm256_cvtepi32_epi64(keep)));
            u=_mm256_fmadd_pd(dx,f,u); v=_mm256_fmadd_pd(dy,f,v); w=_mm256_fmadd_pd(dz,f,w);
        }
        return {sum4(u),sum4(v),sum4(w)};
    }
    struct RefineTile {
        alignas(32) double x[1028]{},y[1028]{},z[1028]{};
        alignas(32) double local[MAXCO]{};
        int visits=0;
        double stop=0.;
        bool done=true;
    };

    void refine_local(RefineTile& t,const Node& a,Node& b,Vec r,double iv,int k) {
        build_moment(int(&b-nodes.data()));
        const auto &l=lay[k],&f=lay[p];
        const auto& t0=trans[k];
        alignas(32) double d[MAXCO],der[MAXPAD],m[MAXCO];
        derivatives(d,r*iv,k);
        double sp[PMAX+1],tp[PMAX+1];
        powers(sp,std::max(1e-100,b.r)*iv,k);
        powers(tp,-std::max(1e-100,a.r)*iv,k);
        const double *bm=moments[b.mbase];
        for(auto row:l.rows) {
            int di=t0.didx[row.x][row.y];
            std::copy_n(d+row.start,row.len,der+di);
            std::fill_n(der+di+row.len,3,0.);
            const double *s=bm+f.idx[row.x][row.y];
            int q=row.x+row.y;
            for(int j=0;j<row.len;++j) m[row.start+j]=s[j]*sp[q+j];
        }
        for(auto q:t0.quads) {
            V a0=_mm256_setzero_pd(),a1=a0;
            const uint32_t* u=t0.products.data()+q.beg;
            int j=0;
            for(;j+1<q.count;j+=2) {
                uint32_t h0=u[j],h1=u[j+1];
                a0=_mm256_fmadd_pd(_mm256_loadu_pd(der+(h0>>16)),_mm256_set1_pd(m[h0&65535]),a0);
                a1=_mm256_fmadd_pd(_mm256_loadu_pd(der+(h1>>16)),_mm256_set1_pd(m[h1&65535]),a1);
            }
            if(j<q.count) {
                uint32_t h=u[j];
                a0=_mm256_fmadd_pd(_mm256_loadu_pd(der+(h>>16)),_mm256_set1_pd(m[h&65535]),a0);
            }
            alignas(32) double s[4];
            _mm256_store_pd(s,a0+a1);
            int n=q.x+q.y+q.z;
            double fa=iv*fi[q.x]*fi[q.y];
            for(int j=(n==0);j<std::min(4,k+1-n);++j)
                t.local[q.out+j]+=s[j]*fa*fi[q.z+j]*tp[n+j];
        }
    }

    void refine_points(RefineTile& t,const Node& a,Node& b,int k) {
        build_moment(int(&b-nodes.data()));
        const double *bm=moments[b.mbase];
        double s=std::max(1e-100,b.r);
        V sv=_mm256_set1_pd(s),iv=_mm256_set1_pd(1./s);
        alignas(32) V d[MAXCO];
        for(int j=0;j<a.size();j+=4) {
            int i=a.arena+j;
            V rx=_mm256_set1_pd(b.c.x)-_mm256_loadu_pd(x.data()+i),
              ry=_mm256_set1_pd(b.c.y)-_mm256_loadu_pd(y.data()+i),
              rz=_mm256_set1_pd(b.c.z)-_mm256_loadu_pd(z.data()+i);
            V r2=_mm256_fmadd_pd(rz,rz,_mm256_fmadd_pd(ry,ry,
                          _mm256_fmadd_pd(rx,rx,_mm256_set1_pd(eps2))));
            V q=sv/r2;
            rows_deriv<V>(d,lay[k],q*rx,q*ry,q*rz,sv*q,rinv(r2));
            V x0=_mm256_setzero_pd(),y0=x0,z0=x0;
            for(const auto& g:grad_terms[k]) {
                V m=_mm256_set1_pd(bm[g.src]);
                x0=_mm256_fnmadd_pd(m,d[g.x],x0);
                y0=_mm256_fnmadd_pd(m,d[g.y],y0);
                z0=_mm256_fnmadd_pd(m,d[g.z],z0);
            }
            _mm256_store_pd(t.x+j,_mm256_fmadd_pd(x0,iv,_mm256_load_pd(t.x+j)));
            _mm256_store_pd(t.y+j,_mm256_fmadd_pd(y0,iv,_mm256_load_pd(t.y+j)));
            _mm256_store_pd(t.z+j,_mm256_fmadd_pd(z0,iv,_mm256_load_pd(t.z+j)));
        }
    }

    void refine_visit(RefineTile& t,const Node& a,int j,const double* cut) {
        if(!t.done) return;
        if((++t.visits&255)==0 && cpu()>t.stop) {
            t.done=false;
            return;
        }
        auto& b=nodes[j];
        Vec r=b.c-a.c;
        double r2=r.norm2()+eps2;
        double q=a.r+b.r;
        if(q*q<cut[p]*r2) {
            int k=int(std::upper_bound(cut+1,cut+p,q*q/r2)-cut);
            if(double(b.size())*a.size()>=.31*(trans[k].products.size()+6.*lay[k].stride)) {
                refine_local(t,a,b,r,1./std::sqrt(r2),k);
                return;
            }
        }
        if(b.size()>=42) {
            double dx=std::max(0.,std::abs(r.x)-a.half.x),
                   dy=std::max(0.,std::abs(r.y)-a.half.y),
                   dz=std::max(0.,std::abs(r.z)-a.half.z);
            double ds=dx*dx+dy*dy+dz*dz+eps2, u=b.r*b.r;
            if(u<cut[p]*ds) {
                int k=int(std::upper_bound(cut+1,cut+p,u/ds)-cut);
                if(b.size()>=.98*lay[k].stride) {
                    refine_points(t,a,b,k);
                    return;
                }
            }
        }
        if(b.leaf()) {
            for(int k=0;k<a.size();++k) {
                Vec f=direct_target(a.arena+k,b.arena,b.arena_end);
                t.x[k]+=f.x; t.y[k]+=f.y; t.z[k]+=f.z;
            }
        } else {
            refine_visit(t,a,b.left,cut);
            refine_visit(t,a,b.right,cut);
        }
    }

    void refine_commit(const Node& a,const RefineTile& t) {
        const auto& l=lay[p];
        double scale=1./std::max(1e-100,a.r);
        for(int i=0;i<a.size();++i) {
            int j=i+a.arena;
            Vec u=(Vec{x[j],y[j],z[j]}-a.c)*scale;
            double vx[PMAX+1],vy[PMAX+1];
            powers(vx,u.x,p);powers(vy,u.y,p);
            Vec a0{};
            for(auto row:l.rows) {
                const double *k=t.local+row.start;
                double q=k[row.len-1],g=0.;
                for(int h=row.len-2;h>=0;--h) {
                    g=g*u.z+q;
                    q=q*u.z+k[h];
                }
                if(row.x) a0.x+=row.x*vx[row.x-1]*vy[row.y]*q;
                if(row.y) a0.y+=row.y*vx[row.x]*vy[row.y-1]*q;
                a0.z+=vx[row.x]*vy[row.y]*g;
            }
            ax[j]=t.x[i]+a0.x*scale;
            ay[j]=t.y[i]+a0.y*scale;
            az[j]=t.z[i]+a0.z*scale;
        }
    }

    float particle_rank(int j,const std::vector<const Node*>& path) const {
        Vec v{x[j],y[j],z[j]};
        double u=0.;
        for(const auto* t:path) {
            double d=(t->c-v).norm2()/std::max(1e-200,t->r*t->r);
            double k=.18+.82*std::min(1.,d);
            k*=k;k*=k;
            u+=k*t->uncertainty;
        }
        double f=ax[j]*ax[j]+ay[j]*ay[j]+az[j]*az[j];
        return float(u/std::max(1e-300,f));
    }
    void rank_tiles(int id,std::vector<const Node*>& path,std::vector<std::pair<float,int>>& rank) const {
        const auto& a=nodes[id];
        if(a.uncertainty>0.) path.push_back(&a);
        if(a.leaf()) {
            double s=0.,m=0.;
            for(int j=a.arena;j<a.arena+a.size();++j) {
                double u=particle_rank(j,path);
                s+=u;
                m=std::max(m,u);
            }
            rank.emplace_back(float(s+1.4*m),id);
        } else {
            rank_tiles(a.left,path,rank);
            rank_tiles(a.right,path,rank);
        }
        if(a.uncertainty>0.) path.pop_back();
    }

    void refine_tiles(double stop) {
        double start=cpu();
        std::vector<const Node*> path;
        std::vector<std::pair<float,int>> rank;
        path.reserve(64);
        rank.reserve(nodes.size()/2+1);
        rank_tiles(0,path,rank);
        std::sort(rank.begin(),rank.end(),std::greater<>());
        double cut[PMAX+1];
        double tt=std::clamp(.018*tol,1e-10,2e-6);
        for(int k=1;k<=p;++k) cut[k]=std::pow(tt,2./k);
        int tiles=0;
        for(auto u:rank) {
            if(cpu()>stop || u.first<=0.f || refined>=part.size()/16) break;
            RefineTile t;
            t.stop=stop;
            const auto &a=nodes[u.second];
            refine_visit(t,a,0,cut);
            if(!t.done) break;
            refine_commit(a,t);
            refined+=a.size();
            ++tiles;
        }
        if(verbose) std::fprintf(stderr,"refined %d targets in %d leaves, %.5f CPU s\n",refined,tiles,cpu()-start);
    }
    void rank_targets(int id,std::vector<const Node*>& path,std::vector<std::pair<float,int>>& rank) const {
        const auto& a=nodes[id];
        if(a.uncertainty>0.) path.push_back(&a);
        if(a.leaf()) {
            for(int j=a.arena;j<a.arena+a.size();++j) {
                rank.emplace_back(particle_rank(j,path),j);
            }
        } else {
            rank_targets(a.left,path,rank);
            rank_targets(a.right,path,rank);
        }
        if(a.uncertainty>0.) path.pop_back();
    }
    void refine() {
        const double start=cpu();
        const int n=int(part.size());
        const double stop=.944*budget-.012-n*6e-8;
        const double window=n>=262144 ? .018*budget+.012+n*1e-7 : .055*budget;
        if(n<4096 || degraded || stop-start<window) return;
        if(n>70000) {
            refine_tiles(stop);
            return;
        }
        std::vector<std::pair<float,int>> ranked;
        ranked.reserve(n);
        std::vector<const Node*> path;
        path.reserve(64);
        rank_targets(0,path,ranked);
        int cap=std::clamp(n/25,1,50000);
        std::nth_element(ranked.begin(),ranked.begin()+cap,ranked.end(),std::greater<>());
        ranked.resize(cap);
        std::sort(ranked.begin(),ranked.end(),std::greater<>());
        for(auto q:ranked) {
            if((refined&7)==0 && cpu()>stop) break;
            if(q.first<=0.f) break;
            int j=q.second;
            Vec a=direct_target(j,0,packed);
            ax[j]=a.x;ay[j]=a.y;az[j]=a.z;
            ++refined;
        }
        if(verbose) std::fprintf(stderr,"refined %d targets in %.5f CPU s\n",refined,cpu()-start);
    }

    bool align_disc() {
        const int n=int(part.size()), step=std::max(1,n/4096);
        Vec mean{};
        int cnt=0;
        for(int i=0;i<n;i+=step) {mean=mean+part[i].x; ++cnt;}
        mean=mean*(1./cnt);
        double a[3][3]{};
        for(int i=0;i<n;i+=step) {
            Vec q=part[i].x-mean;
            for(int u=0;u<3;++u) for(int v=u;v<3;++v)
                a[u][v]+=q[u]*q[v];
        }
        double trace=a[0][0]+a[1][1]+a[2][2];
        if(trace<=0.) return false;
        for(int i=0;i<3;++i) for(int j=i+1;j<3;++j) a[j][i]=a[i][j];
        auto basis=frame;
        for(int s=0;s<6;++s) for(int i=0;i<2;++i) for(int j=i+1;j<3;++j) {
            double apq=a[i][j];
            if(std::abs(apq)<1e-15*trace) continue;
            double t=(a[j][j]-a[i][i])/(2*apq);
            t=std::copysign(1.0,t)/(std::abs(t)+std::sqrt(1.+t*t));
            double c=1./std::sqrt(1.+t*t), ss=t*c;
            for(int k=0;k<3;++k) if(k!=i && k!=j) {
                double u=a[i][k], v=a[j][k];
                a[i][k]=a[k][i]=c*u-ss*v;
                a[j][k]=a[k][j]=ss*u+c*v;
            }
            a[i][i]-=t*apq;
            a[j][j]+=t*apq;
            a[i][j]=a[j][i]=0.;
            Vec u=basis[i],v=basis[j];
            basis[i]=u*c-v*ss; basis[j]=u*ss+v*c;
        }
        int low=0,high=0;
        for(int i=1;i<3;++i) {
            if(a[i][i]<a[low][low]) low=i;
            if(a[i][i]>a[high][high]) high=i;
        }
        if(low==high || a[low][low]>0.055*a[high][high]) return false;
        frame={basis[high],basis[3-low-high],basis[low]};
        for(auto& b:part) {
            Vec d=b.x-mean;
            b.x={dot(frame[0],d),dot(frame[1],d),dot(frame[2],d)};
        }
        return true;
    }

public:
    Solver(const Input* in,int n,double budget_) : part(n),budget(budget_) {
        for(int j=0;j<n;++j) part[j]={in[j].x,j};
        const bool disc=n>=1024 && align_disc();
        small_snapshot=n<=65536;
        conservative_disc=disc && n<=65536;
        p=n<=1048576?13:12;
        bucket=n<=65536?60:(n<=262144?68:(n<=1048576?86:102));
        const double microseconds=1e6*budget/n;
        if(n>=16384) {
            if(disc) {
                p=microseconds>=19.?16:14;
                if(n>1048576) bucket=118;
            } else if(n>1048576) {
                p=14; bucket=136;
                use_more=true;
            } else if(n>262144) {
                p=microseconds>=30.?15:14;
                bucket=92;
            } else if(microseconds>=30.) {
                p=15;
                bucket+=3;
            }
        }
        theta=.65; mindirect=.42; one_cost=.9;
        float_near=n>=1024 && !disc;
        if(disc && n>=16384 && n<=262144 && microseconds<26.) {
            float_near=true;
            hybrid_near=true;
            float_gap2=.016*.016;
        }
        relaxed_pace=n>=262144;
        separate_prep=n>=262144;
        mixed_points=n>=262144 && !disc;
        fit_sphere=disc && n==262144;
        mix_far=disc && n>65536;
        harm_points=n>65536 && !disc;
        harm_mode=harm_points;
        moment_split=std::max(64,2*bucket);
        verbose=std::getenv("GALAXY_TRACE")!=nullptr;
    }

    void solve(Vec *out,double mass) {
        make(0,int(part.size()));
        const double rows=16.*(nodes.size()+Arena::batch);
        while(p>4) {
            double c=(p+1)*(p+2)*(p+3)/6;
            if(harm_mode) c+=(p+1)*(p+1);
            if(c*rows<=1.305e9) break;
            --p;
        }
        if(harm_mode) harmonic=std::make_unique<Harmonic>();
        active_order=p;
        pack(); init_tables(); tune();
        ttree=cpu();
        walk();
        twalk=cpu();
        push_local(0,{});
        locals.blocks.clear();
        if(harmonic) harmonic->locals.blocks.clear();
        refine();
        moments.blocks.clear();
        if(harmonic) harmonic->moments.blocks.clear();
        for(int i=0;i<packed;++i) if(rev[i]>=0)
            out[rev[i]]=(frame[0]*ax[i]+frame[1]*ay[i]+frame[2]*az[i])*mass;
        if(verbose) std::fprintf(stderr,"theta=%g order=%d setup=%.4f traversal=%.4f finish=%.4f "
            "pairs=%llu translations=%llu points=%llu coarse=%llu nodes=%zu coefficients=%zu/%zu\n",
            theta,p,ttree,twalk-ttree,cpu()-twalk,(unsigned long long)npairs,
            (unsigned long long)nml,(unsigned long long)nmp,(unsigned long long)nshortcut,
            nodes.size(),moments.size(),locals.size());
    }
};

} // namespace

int main(int argc,char **argv) {
    if(argc!=3) return 1;
    struct stat st;
    if(stat(argv[1],&st)!=0 || st.st_size<=0 || st.st_size%sizeof(Input)!=0 ||
       st.st_size/sizeof(Input)>std::numeric_limits<int>::max()) return 1;
    int n=int(st.st_size/sizeof(Input));
    std::unique_ptr<Input[]> in(new Input[n]);
    FILE* f=std::fopen(argv[1],"rb");
    if(!f) return 1;
    size_t got=std::fread(in.get(),sizeof(Input),n,f);
    std::fclose(f);
    if(got!=size_t(n)) return 1;
    double budget=2.0;
    if(const char *e=std::getenv("GALAXY_CPU_LIMIT_S")) {
        double v=std::strtod(e,nullptr);
        if(std::isfinite(v) && v>0.) budget=v;
    }
    auto out=std::unique_ptr<Vec[]>(new Vec[n]);
    double mass=in[0].m;
    {
        Solver solver(in.get(),n,budget);
        in.reset();
        solver.solve(out.get(),mass);
    }
    f=std::fopen(argv[2],"wb");
    if(!f) return 1;
    size_t wrote=std::fwrite(out.get(),sizeof(Vec),n,f);
    int closed=std::fclose(f);
    return wrote==size_t(n) && closed==0 ? 0 : 1;
}
