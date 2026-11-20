#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <vector>
#include <immintrin.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr double eps2 = .0001;
constexpr int maxp = 28;
constexpr int terms(int p) { return (p + 1) * (p + 2) * (p + 3) / 6; }
constexpr int maxterms = terms(maxp);
constexpr int hs(int p) { return (p+1)*(p+1); }
constexpr int maxhs = hs(maxp);
constexpr size_t max_coefficient_bytes=size_t(1510)<<20;
inline int residual_order(int c,int side,int p) { return std::clamp((c*side+p-1)/p+1,2,c); }
using V = __m256d;
inline V set(double a) { return _mm256_set1_pd(a); }
inline V zero() { return _mm256_setzero_pd(); }
inline V load(const double* a) { return _mm256_loadu_pd(a); }
inline void store(double* a, V v) { _mm256_storeu_pd(a, v); }
inline V fma(V a, V b, V c) { return _mm256_fmadd_pd(a, b, c); }
inline double sum(V a) {
    const __m128d b = _mm_add_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd(a, 1));
    return _mm_cvtsd_f64(_mm_add_sd(b, _mm_unpackhi_pd(b, b)));
}
inline void transpose4(V& a,V& b,V& c,V& d) {
    const V u0=_mm256_unpacklo_pd(a,b),u1=_mm256_unpackhi_pd(a,b);
    const V u2=_mm256_unpacklo_pd(c,d),u3=_mm256_unpackhi_pd(c,d);
    a=_mm256_permute2f128_pd(u0,u2,0x20);
    b=_mm256_permute2f128_pd(u1,u3,0x20);
    c=_mm256_permute2f128_pd(u0,u2,0x31);
    d=_mm256_permute2f128_pd(u1,u3,0x31);
}
inline V inverse_distance(V dx,V dy,V dz) {
    const V rr=fma(dx,dx,fma(dy,dy,fma(dz,dz,set(eps2))));
    return set(1.)/_mm256_sqrt_pd(rr);
}
inline double integer_power(double x,int p) {
    double r=1.;
    while (p>0) {
        if (p&1) r*=x;
        p>>=1;
        x*=x;
    }
    return r;
}
double cpu() {
    timespec t;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    return t.tv_sec + 1.e-9 * t.tv_nsec;
}
int optint(const char* name, int def) {
    const char* a = std::getenv(name);
    return a ? std::atoi(a) : def;
}
double optdouble(const char* name, double def) {
    const char* a = std::getenv(name);
    return a ? std::atof(a) : def;
}
struct Timeout {};
double compute_until=1.e30;
inline void checkclock() {
    if (cpu()>compute_until) throw Timeout{};
}

struct Particle {
    double x[3];
    uint32_t original;
};
struct Node {
    double c[3], r2, r, rsoft;
    double com[3], axial, g, gw;
    int begin, count, left, right;
    int start, parent=-1;
    uint32_t mo, lo, hmo, hlo;
    uint8_t pm, pl, hm, hl;
    float shrink=1.;
    double risk[2]{};
    bool leaf() const { return left == 0; }
};

struct Tree {
    std::vector<Particle> bodies;
    std::vector<Node> nodes;
    std::vector<uint32_t> index;
    std::vector<double> x[3], a[3];
    int block, levels=0;
    double split_bias=0., centering=0., decay=0.;
    double radial_center[3]{}, outer=0.;
    bool cusp=false;
    bool rotated=false;
    double rot[3][3]={{1,0,0},{0,1,0},{0,0,1}};

    void orient() {
        double c[3]{}, a[3][3]{};
        for (const auto& p:bodies) for (int i=0;i<3;++i) c[i]+=p.x[i];
        for (double& s:c) s/=bodies.size();
        for (const auto& p:bodies) {
            const double r[3]={p.x[0]-c[0],p.x[1]-c[1],p.x[2]-c[2]};
            for (int i=0;i<3;++i) for (int j=i;j<3;++j) a[i][j]+=r[i]*r[j];
        }
        for (int i=0;i<3;++i) for (int j=0;j<i;++j) a[i][j]=a[j][i];
        double v[3][3]={{1,0,0},{0,1,0},{0,0,1}};
        for (int k=0;k<10;++k) for (int p=0;p<3;++p) for (int q=p+1;q<3;++q) {
            if (std::abs(a[p][q])<1.e-15*(std::abs(a[p][p])+std::abs(a[q][q]))) continue;
            if (a[p][q]==0.) continue;
            const double th=.5*std::atan2(2*a[p][q],a[q][q]-a[p][p]);
            const double sn=std::sin(th),cs=std::cos(th);
            for (int j=0;j<3;++j) {
                const double x=a[p][j],y=a[q][j];
                a[p][j]=cs*x-sn*y; a[q][j]=sn*x+cs*y;
            }
            for (int j=0;j<3;++j) {
                const double x=a[j][p],y=a[j][q];
                a[j][p]=cs*x-sn*y; a[j][q]=sn*x+cs*y;
                const double u=v[j][p],w=v[j][q];
                v[j][p]=cs*u-sn*w; v[j][q]=sn*u+cs*w;
            }
        }
        std::array<int,3> ix{0,1,2};
        std::sort(ix.begin(),ix.end(),[&](int i,int j){return a[i][i]>a[j][j];});
        if (!(a[ix[2]][ix[2]]<.15*a[ix[1]][ix[1]])) return;
        rotated=true;
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) rot[i][j]=v[j][ix[i]];
        for (auto& p:bodies) {
            const double r[3]={p.x[0]-c[0],p.x[1]-c[1],p.x[2]-c[2]};
            for (int i=0;i<3;++i) p.x[i]=rot[i][0]*r[0]+rot[i][1]*r[1]+rot[i][2]*r[2];
        }
    }

    void find_outer(bool split_tail) {
        if (rotated || bodies.size()<1024) return;
        const size_t n=bodies.size(),m=std::min<size_t>(2048,n);
        std::vector<double> q(m);
        for (int d=0;d<3;++d) {
            for (size_t i=0;i<m;++i) q[i]=bodies[i*n/m].x[d];
            std::nth_element(q.begin(),q.begin()+m/2,q.end());
            radial_center[d]=q[m/2];
        }
        for (size_t i=0;i<m;++i) {
            const auto& p=bodies[i*n/m];
            const double dx=p.x[0]-radial_center[0],dy=p.x[1]-radial_center[1],dz=p.x[2]-radial_center[2];
            q[i]=dx*dx+dy*dy+dz*dz;
        }
        std::sort(q.begin(),q.end());
        if (q[size_t(.95*m)]>7.5*q[m/2]) {
            cusp=q[size_t(.10*m)]<.026*q[m/2];
            if (split_tail) outer=std::sqrt(q[size_t(.89*m)]);
        }
    }

    int build(int from, int to, int level=0) {
        levels=std::max(levels,level+1);
        Node n{};
        n.begin = from;
        n.count = to - from;
        const int id = nodes.size();
        nodes.push_back(n);
        auto* p = bodies.data() + from;
        auto* e = bodies.data() + to;
        double lo[3], hi[3];
        for (int j = 0; j < 3; ++j) lo[j] = hi[j] = p->x[j];
        for (auto* q = p + 1; q != e; ++q) {
            for (int j = 0; j < 3; ++j) {
                lo[j] = std::min(lo[j], q->x[j]);
                hi[j] = std::max(hi[j], q->x[j]);
            }
        }
        int axis = 0;
        for (int j = 0; j < 3; ++j) {
            n.c[j] = .5 * hi[j] + .5 * lo[j];
            if (hi[j] - lo[j] > hi[axis] - lo[axis]) axis = j;
        }
        const double sx=.5*(hi[0]-lo[0]),sy=.5*(hi[1]-lo[1]),sz=.5*(hi[2]-lo[2]);
        n.axial=sz;
        double pivot=n.c[axis],sm=0.;
        const double norm2=sx*sx+sy*sy+sz*sz;
        const double norm=norm2>0. ? 1./norm2 : 0.;
        double s=0.,mom[3]{},moment_w=0.;
        for (auto* q = p; q != e; ++q) {
            const double dx = q->x[0] - n.c[0], dy = q->x[1] - n.c[1], dz = q->x[2] - n.c[2];
            const double rr=dx*dx+dy*dy+dz*dz,u=rr*norm,u2=u*u,u4=u2*u2;
            n.r2=std::max(n.r2,rr);
            s+=u4*u4;
            if (centering!=0.) {
                const double w=u4*u2*u;
                moment_w+=w;
                mom[0]+=w*dx; mom[1]+=w*dy; mom[2]+=w*dz;
            }
            if (split_bias!=0.) sm+=q->x[axis]-pivot;
        }
        pivot+=split_bias*(sm/n.count);
        n.r = std::sqrt(n.r2);
        n.rsoft = n.r;
        if (n.r2 > 0 && n.count >= 8) {
            if (centering>0. && moment_w>0.) {
                const double f=centering/moment_w;
                for (int j=0;j<3;++j) n.c[j]+=f*mom[j];
                n.axial=std::max(hi[2]-n.c[2],n.c[2]-lo[2]);
                n.r2=s=0.;
                for (auto* q=p;q!=e;++q) {
                    const double dx=q->x[0]-n.c[0],dy=q->x[1]-n.c[1],dz=q->x[2]-n.c[2];
                    const double rr=dx*dx+dy*dy+dz*dz,u=rr*norm,u2=u*u,u4=u2*u2;
                    n.r2=std::max(n.r2,rr);
                    s+=u4*u4;
                }
                n.r=std::sqrt(n.r2);
            }
            n.rsoft = std::sqrt(norm2)*std::sqrt(std::sqrt(std::sqrt(std::sqrt(s/n.count))));
        }
        bool split_node=n.count>block;
        if (!split_node && outer>0. && n.count>std::max(10,block/4)) {
            const double x=n.c[0]-radial_center[0],y=n.c[1]-radial_center[1],z=n.c[2]-radial_center[2];
            const double r2=x*x+y*y+z*z, rd=outer+.30*n.r;
            split_node=r2>rd*rd && n.r2>.029*r2;
        }
        if (split_node) {
            auto* m = std::partition(p, e, [&](const Particle& q) { return q.x[axis] < pivot; });
            int split = m - bodies.data();
            if (split == from || split == to || level>=128) {
                split = from + n.count / 2;
                if (level>=128)
                    std::nth_element(p,bodies.data()+split,e,[&](const Particle& a,const Particle& b){return a.x[axis]<b.x[axis];});
            }
            n.left = build(from, split,level+1);
            n.right = build(split, to,level+1);
            nodes[n.left].parent=nodes[n.right].parent=id;
            for (int i=0;i<3;++i)
                n.com[i]=(nodes[n.left].com[i]*nodes[n.left].count+
                          nodes[n.right].com[i]*nodes[n.right].count)/n.count;
            if (decay>0. && n.r>0.) for (int k:{n.left,n.right}) {
                auto& v=nodes[k];
                const double dx=v.c[0]-n.c[0],dy=v.c[1]-n.c[1],dz=v.c[2]-n.c[2];
                const double d=std::sqrt(dx*dx+dy*dy+dz*dz);
                const double ratio=v.r/std::max(.22*n.r,n.r-.60*d);
                if (ratio<1.) v.shrink=1./(1.-decay*std::log(std::max(1.e-4,ratio)));
            }
        } else {
            for (auto* q=p;q!=e;++q) for (int i=0;i<3;++i) n.com[i]+=q->x[i]-n.c[i];
            for (int i=0;i<3;++i) n.com[i]=n.c[i]+n.com[i]/n.count;
        }
        nodes[id] = n;
        return id;
    }

    void setup(int b) {
        block = b;
        nodes.reserve(3 * bodies.size() / block + 128);
        build(0, bodies.size());
        int size = 0;
        for (Node& n : nodes) if (n.leaf()) {
            n.start = size;
            size += (n.count + 3) & -4;
        }
        for (int j = 0; j < 3; ++j) {
            x[j].resize(size);
            a[j].resize(size);
        }
        index.resize(bodies.size());
        for (const Node& n : nodes) if (n.leaf()) {
            for (int i = 0; i < n.count; ++i) {
                const auto& p = bodies[i+n.begin];
                index[p.original] = n.start + i;
                for (int j = 0; j < 3; ++j) x[j][i+n.start] = p.x[j];
            }
            for (int i = n.count; i % 4; ++i)
                for (int j = 0; j < 3; ++j) x[j][i+n.start] = x[j][n.start+n.count-1];
        }
        std::vector<Particle>().swap(bodies);
    }

    void estimate(int target, int source, double* s) const {
        const auto& a=nodes[target]; const auto& b=nodes[source];
        const double dx=b.com[0]-a.c[0],dy=b.com[1]-a.c[1],dz=b.com[2]-a.c[2];
        const double d2=dx*dx+dy*dy+dz*dz, rr=d2+eps2;
        if (b.leaf() || b.r2<.7*rr) {
            const double r=rr+.32*b.r2;
            const double f=b.count/(r*std::sqrt(r));
            s[0]+=f*dx; s[1]+=f*dy; s[2]+=f*dz;
            s[3]+=b.count/r;
        } else {
            estimate(target,b.left,s);
            estimate(target,b.right,s);
        }
    }
    void scales(double weak) {
        for (size_t i=nodes.size();i-- > 0;) {
            auto& n=nodes[i];
            if (n.leaf()) {
                double s[4]{};
                estimate(i,0,s);
                const double f=.08*s[3];
                n.g=std::sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]+f*f);
            } else {
                const auto& l=nodes[n.left]; const auto& r=nodes[n.right];
                n.g=std::sqrt(n.count/(l.count/(l.g*l.g)+r.count/(r.g*r.g)));
            }
        }
        if (weak>0.) {
            const int stride=std::max(1,nodes[0].count/1024);
            std::vector<double> sample;
            int next=0;
            for (const auto& n:nodes) if (n.leaf()) while (next<n.begin+n.count) {
                sample.push_back(n.g);
                next+=stride;
            }
            auto m=sample.begin()+sample.size()/2;
            std::nth_element(sample.begin(),m,sample.end());
            const double characteristic=*m;
            for (auto& n:nodes) {
                const double factor=std::clamp(std::pow(characteristic/n.g,weak),.50,24.);
                n.g=1./n.g;
                n.gw=n.g*factor;
            }
        } else for (auto& n:nodes) n.gw=n.g=1./n.g;
    }
};

struct XYZ { int x, y, z, deg; };
template<bool negate>
inline void pack_columns(V* dest,const double* a,const double* b,const double* c,const double* d,int count,
                         const XYZ* order) {
    int i=0;
    if (a==b && a==c && a==d) {
        for (;i<count;++i) {
            const double value=a[i];
            dest[i]=set(negate && order[i].deg%2 ? -value : value);
        }
        return;
    }
    for (;i+3<count;i+=4) {
        V r0=load(a+i),r1=load(b+i),r2=load(c+i),r3=load(d+i);
        transpose4(r0,r1,r2,r3);
        if constexpr (negate) {
            dest[i]=order[i].deg%2 ? -r0 : r0;
            dest[i+1]=order[i+1].deg%2 ? -r1 : r1;
            dest[i+2]=order[i+2].deg%2 ? -r2 : r2;
            dest[i+3]=order[i+3].deg%2 ? -r3 : r3;
        } else {
            dest[i]=r0; dest[i+1]=r1; dest[i+2]=r2; dest[i+3]=r3;
        }
    }
    for (;i<count;++i) {
        const V r=_mm256_setr_pd(a[i],b[i],c[i],d[i]);
        dest[i]=negate && order[i].deg%2 ? -r : r;
    }
}
constexpr int idx(int x, int y, int z) {
    const int d = x+y+z;
    return terms(d-1) + z*(2*d+3-z)/2 + y;
}
struct Basis {
    XYZ xyz[maxterms];
    double factinv[maxp+1], xyzfactinv[maxterms], xyzfact[maxterms];
    uint16_t lowers[maxterms][6], high[maxterms][3];
    double lin[maxp+1],sq[maxp+1];
    std::vector<uint16_t> schedule;
    std::vector<uint32_t> cschedule;
    uint32_t row[maxterms];
    uint32_t crow[maxterms];
    uint16_t clen[maxterms][maxp+1];

    explicit Basis(int p, int cp, bool pairs=true) {
        factinv[0] = 1;
        for (int k = 1; k <= maxp; ++k) {
            factinv[k] = factinv[k-1] / k;
            lin[k]=(2.*k-1)/k;
            sq[k]=(k-1.)/k;
        }
        for (int d = 0; d <= p; ++d) for (int z = 0; z <= d; ++z) for (int y = 0; y <= d-z; ++y) {
            const int x = d-y-z, i = idx(x,y,z);
            xyz[i] = {x,y,z,d};
            xyzfactinv[i] = factinv[x]*factinv[y]*factinv[z];
            xyzfact[i]=1./xyzfactinv[i];
            int v[3] = {x,y,z};
            for (int j=0; j<3; ++j) {
                if (d<p) {
                    ++v[j];
                    high[i][j]=1+idx(v[0],v[1],v[2]);
                    --v[j];
                }
                if (v[j]) {
                    --v[j];
                    lowers[i][j] = 1+idx(v[0],v[1],v[2]);
                    if (v[j]) {
                        --v[j];
                        lowers[i][j+3] = 1+idx(v[0],v[1],v[2]);
                        ++v[j];
                    } else lowers[i][j+3] = 0;
                    ++v[j];
                } else lowers[i][j] = lowers[i][j+3] = 0;
            }
        }
        if (pairs) for (int i = 0; i < terms(p); ++i) {
            row[i] = schedule.size();
            auto b = xyz[i];
            for (int j = 0; j < terms(p-b.deg); ++j) {
                auto a = xyz[j];
                schedule.push_back(1+idx(a.x+b.x,a.y+b.y,a.z+b.z));
            }
        }
        if (cp>=2) for (int i=0; i<terms(cp); ++i) {
            crow[i]=cschedule.size();
            const auto b=xyz[i];
            for (int d=0;d<=cp-b.deg;++d) {
                for (int j=terms(d-1);j<terms(d);++j) {
                    const auto a=xyz[j];
                    if (a.z+b.z>=2) {
                        const unsigned k=1+idx(a.x+b.x,a.y+b.y,a.z+b.z);
                        cschedule.push_back((k<<16)|j);
                    }
                }
                clen[i][d]=cschedule.size()-crow[i];
            }
        }
    }
    void derivatives(V* der,int p,V dx,V dy,V dz,V vv) const {
        const V is=-vv*vv,cx=dx*is,cy=dy*is,cz=dz*is;
        der[0]=zero(); der[1]=vv;
        for (int d=1;d<=p;++d) {
            const V w=set(lin[d]),x=cx*w,y=cy*w,z=cz*w,h=is*set(sq[d]);
            for (int i=terms(d-1);i<terms(d);++i) {
                const auto* j=lowers[i];
                const V a=fma(der[j[2]],z,fma(der[j[1]],y,der[j[0]]*x));
                der[i+1]=fma(der[j[3]]+der[j[4]]+der[j[5]],h,a);
            }
        }
        for (int i=1;i<terms(p);++i) der[i+1]*=set(xyzfact[i]);
    }
};

struct Pair {
    uint32_t a,b;
    Pair() = default;
    Pair(int x,int y,int ca=0,int cb=0) : a(unsigned(x)|(unsigned(ca)<<24)), b(unsigned(y)|(unsigned(cb)<<24)) {}
    int first() const { return a&0xffffff; }
    int second() const { return b&0xffffff; }
    int ca() const { return a>>24; }
    int cb() const { return b>>24; }
    unsigned shape() const { return (a>>24)*32+(b>>24); }
};
void stable_group(std::vector<Pair>& data,std::vector<Pair>& buffer) {
    if (data.size()<2) return;
    unsigned bins[1024]{};
    for (const Pair& p:data) ++bins[p.shape()];
    unsigned total=0;
    for (auto& b:bins) {
        const auto n=b;
        b=total;
        total+=n;
    }
    buffer.resize(data.size());
    for (const Pair& p:data) buffer[bins[p.shape()]++]=p;
    std::copy(buffer.begin(),buffer.end(),data.begin());
}
constexpr int wbins=144;
struct Plan {
    Tree& t;
    int pmax, pmin, rawmax, usedraw=1, usedharm=0;
    double tol, prefactor, cap, rfac, tracefac, gravfac, overlap, weight_exp;
    double q2[maxp+1];
    double wq[wbins][maxp+1], wt[wbins];
    float htrim[maxp+1][maxp+1][maxp+1], rtrim[maxp+1][maxp+1][maxp+1], ctrim[maxp+1][maxp+1][maxp+1];
    std::vector<Pair> far[maxp+1], near;
    std::vector<Pair> harm[maxp+1][maxp+1];
    double hc[maxp+1], cc[maxp+1], fc[maxp+1];
    uint64_t pp = 0, visits = 0;
    double mlwork = 0, hw = 0;
    size_t maxpairs, pairs=0;
    double finish;
    double look;
    bool abort = false;
    bool record=true;
    bool trim, risks;
    Plan(Tree& t_, int pmin_, int pmax_, int rawmax_, double tol_, double ratio, double pf, double rf,
         double trace, double grav, double ov, double wexp, double look_, bool tr,bool risk_,
         size_t maxpairs_, double until=1.e30)
        : t(t_), pmax(pmax_), pmin(pmin_), rawmax(rawmax_), tol(tol_), prefactor(pf), cap(ratio*ratio),
          rfac(rf), tracefac(trace), gravfac(grav), overlap(ov), weight_exp(wexp), maxpairs(maxpairs_),
          finish(until), look(look_), trim(tr), risks(risk_) {
        angles();
        hc[0]=cc[0]=cc[1]=0.;
        for (int p=1;p<=pmax;++p) {
            double nn=0, skip=0;
            for (int d=1;d<=p;++d) {
                nn+=(2*d+1)*hs(p-d);
                int k=p-d;
                skip+=(d+1)*(k+1)*(k+2)/2 + (d+1)*k*(k+1)/2 + d*(k+1)*(k+2)/2;
            }
            hc[p]=100+nn*.9+8.*hs(p);
            fc[p]=cost(p);
            if (p>=2) cc[p]=100+(interactions(p)-skip)*1.00+9.*terms(p);
            if (trim) for (int a=2;a<=p;++a) for (int b=2;b<=p;++b) {
                double nn=0,nm=0,raw=0,sk=0;
                const auto sum=[&](int cap1,int cap2) {
                    double count=0;
                    for (int d=1;d<=cap1;++d) {
                        const int k=std::min(p-d,cap2);
                        const int s0=(k+1)*(k+2)/2,s1=k*(k+1)/2;
                        count+=(2*d+1)*hs(k);
                        raw+=(terms(d)-terms(d-1))*terms(k);
                        sk+=(d+1)*(s0+s1)+d*s0;
                    }
                    return count;
                };
                nn=sum(a,b);
                nm=sum(b,a);
                htrim[p][a][b]=100+.45*(nn+nm)+3.*(hs(a)+hs(b))+2.*hs(p);
                rtrim[p][a][b]=100+.45*raw+3.*(terms(a)+terms(b))+3.*terms(p);
                ctrim[p][a][b]=100+.50*(raw-sk)+3.*(terms(a)+terms(b))+3.*terms(p);
            }
        }
    }
    void angles() {
        for (int p=2;p<=pmax;++p) q2[p]=std::pow(tol,2.0/p);
        if (gravfac>0) for (int i=0;i<wbins;++i) {
            const int j=i-96;
            const int e=(j+96)/8-12, mantissa=(j+96)%8;
            const double w=std::ldexp(1.+(mantissa+.5)/8.,e);
            wt[i]=tol*std::clamp(gravfac/std::pow(w,weight_exp),.055,16.);
            for (int p=2;p<=pmax;++p) wq[i][p]=std::pow(wt[i],2.0/p);
        }
    }
    static double interactions(int p) {
        return (p+6.)*(p+5.)*(p+4.)*(p+3.)*(p+2.)*(p+1.)/720. - terms(p);
    }
    double cost(int p) const {
        return 100 + .90*interactions(p) + 9.*terms(p);
    }
    void clear(double accuracy,double until) {
        tol=accuracy; finish=until;
        abort=false; pp=visits=pairs=0; mlwork=hw=0;
        usedraw=1; usedharm=0;
        near.clear();
        for (auto& v:far) v.clear();
        for (auto& row:harm) for (auto& v:row) v.clear();
        for (auto& n:t.nodes) { n.pm=n.pl=n.hm=n.hl=0; n.risk[0]=n.risk[1]=0.; }
        angles();
    }
    void release() {
        std::vector<Pair>().swap(near);
        for (auto& v:far) std::vector<Pair>().swap(v);
        for (auto& row:harm) for (auto& v:row) std::vector<Pair>().swap(v);
    }
    void addnear(int aa,int bb) {
        const auto& a = t.nodes[aa]; const auto& b = t.nodes[bb];
        if (record) near.push_back({aa,bb});
        pp += uint64_t((a.count+1)&-2) * ((b.count+3)&-4);
        if (++pairs > maxpairs) abort = true;
    }
    void addfar(int aa,int bb,int p,int ca,int cb,double work) {
        if (record) far[p].emplace_back(aa,bb,ca,cb);
        auto& a = t.nodes[aa]; auto& b = t.nodes[bb];
        a.pm = std::max<int>(a.pm,std::min(ca,p-1));
        b.pm = std::max<int>(b.pm,std::min(cb,p-1));
        a.pl = std::max<int>(a.pl,ca);
        b.pl = std::max<int>(b.pl,cb);
        mlwork += work;
        usedraw=std::max(usedraw,p);
        if (++pairs > maxpairs) abort = true;
    }
    void addharm(int aa,int bb,int p,int c,int ma,int mb,double cost) {
        if (record) harm[p][c].emplace_back(aa,bb,ma,mb);
        auto& a=t.nodes[aa]; auto& b=t.nodes[bb];
        a.hm=std::max<int>(a.hm,std::min(ma,p-1));
        b.hm=std::max<int>(b.hm,std::min(mb,p-1));
        a.hl=std::max<int>(a.hl,ma);
        b.hl=std::max<int>(b.hl,mb);
        if (c>=2) {
            const int ca=residual_order(c,ma,p),cb=residual_order(c,mb,p);
            a.pm=std::max<int>(a.pm,std::min(ca,c-1));
            b.pm=std::max<int>(b.pm,std::min(cb,c-1));
            a.pl=std::max<int>(a.pl,ca);
            b.pl=std::max<int>(b.pl,cb);
            usedraw=std::max(usedraw,c);
        }
        usedharm=std::max(usedharm,p);
        hw+=cost;
        if (++pairs > maxpairs) abort=true;
    }
    struct Choice {
        int p=0,c=-1,pa=0,pb=0;
        double work=0.;
    };
    void mark_risk(Node& a,const Node& b,const Choice& u,bool flip,double ivr) const {
        const double rs=b.rsoft*ivr;
        const int tgt=flip ? u.pb : u.pa,src=flip ? u.pa : u.pb;
        const double ax=(a.axial+b.axial)*ivr;
        const double corr=eps2*ivr*ivr*std::min(3.,16.*ax*ax);
        const double base=b.count*ivr*ivr;
        for (int i=0;i<2;++i) {
            const double rt=a.r*ivr*(i ? 1. : .64);
            const double q=rs+rt;
            double e=integer_power(q,u.p)*u.p;
            if (tgt<u.p) e+=integer_power(rt/(1.-rs),tgt)*tgt;
            if (src<u.p) e+=integer_power(rs/(1.-rt),src+1)*(src+1);
            if (u.c>=0) {
                const int c=std::max(1,u.c);
                e+=corr*integer_power(q,c)*(c+2)*(c+2);
            }
            e*=base;
            a.risk[i]+=e*e;
        }
    }
    Choice decide(const Node& a,const Node& b) const {
        const double dx=a.c[0]-b.c[0], dy=a.c[1]-b.c[1], dz=a.c[2]-b.c[2];
        const double rr=dx*dx+dy*dy+dz*dz+eps2;
        const double rs=a.r+b.r, r2max=rs*rs;
        const double ra=rfac*a.rsoft+(1-rfac)*a.r, rb=rfac*b.rsoft+(1-rfac)*b.r;
        const double rt=ra+rb-overlap*std::min(a.r,b.r), r2=rt*rt;
        const double exact = double(a.count)*b.count;
        if (r2max <= cap*rr && exact >= prefactor*std::min(fc[pmin],hc[pmin])) {
            const double inv=1./rr, qr=r2*inv;
            const double* ang=q2;
            double lt=tol;
            if (gravfac>0) {
                const double w=std::max(b.count*a.gw,a.count*b.gw)*inv;
                uint64_t bits;
                std::memcpy(&bits,&w,sizeof(bits));
                const int j=std::clamp(int(bits>>49)-1023*8+96,0,wbins-1);
                lt=wt[j]; ang=wq[j];
            }
            if (qr<=ang[pmax]) {
                int p=pmin;
                while (qr>ang[p]) ++p;
                int ca=p,cb=p;
                double hcost=hc[p],rcost=fc[p];
                const double q=std::sqrt(qr);
                if (trim && rt>0.) {
                    const double qa=q*(ra/(ra+rb)),qb=q-qa;
                    const double qa1=qa/(1.-qb),qb1=qb/(1.-qa);
                    ca=cb=2;
                    while (ca<p && qa1*qa1>ang[ca]) ++ca;
                    while (cb<p && qb1*qb1>ang[cb]) ++cb;
                    hcost=htrim[p][ca][cb];
                    rcost=rtrim[p][ca][cb];
                }
                if (tracefac>0 && exact>=prefactor*(trim ? .50*hc[p] : hc[p])) {
                    const double zratio=(a.axial+b.axial)/std::max(rs,1.e-100);
                    const double sw=tracefac*eps2*inv*std::min(1.,3.*zratio*zratio);
                    int c=1;
                    double w=sw*q;
                    while (w>lt && c<p) { ++c; w*=q; }
                    if (c<2) c=0;
                    if (c>=2) hcost+=trim ? ctrim[c][residual_order(c,ca,p)][residual_order(c,cb,p)] : cc[c];
                    if (c<=rawmax && c<p && (p>rawmax || hcost<rcost) &&
                        exact>=prefactor*hcost) {
                        return {p,c,ca,cb,hcost};
                    }
                }
                if (p<=rawmax && exact>=prefactor*rcost) {
                    return {p,-1,ca,cb,rcost};
                }
            }
        }
        return {};
    }
    void accept(int a,int b,const Choice& u) {
        if (risks && record && u.p) {
            auto& x=t.nodes[a]; auto& y=t.nodes[b];
            const double dx=x.c[0]-y.c[0],dy=x.c[1]-y.c[1],dz=x.c[2]-y.c[2];
            const double iv=1./std::sqrt(dx*dx+dy*dy+dz*dz+eps2);
            mark_risk(x,y,u,false,iv);
            mark_risk(y,x,u,true,iv);
        }
        if (u.c>=0) addharm(a,b,u.p,u.c,u.pa,u.pb,u.work);
        else if (u.p) addfar(a,b,u.p,u.pa,u.pb,u.work);
        else addnear(a,b);
    }
    void walk(int aa,int bb) {
        if (abort) return;
        if ((++visits&16383)==0 && cpu()>finish) {
            abort=true;
            return;
        }
        const auto& a=t.nodes[aa]; const auto& b=t.nodes[bb];
        if (aa==bb) {
            if (a.leaf()) addnear(aa,bb);
            else {
                walk(a.left,a.left);
                walk(a.left,a.right);
                walk(a.right,a.right);
            }
            return;
        }
        if (auto u=decide(a,b); u.p) {
            if (look>0. && u.p>=10 && (!a.leaf() || !b.leaf())) {
                const bool cuta=!a.leaf() && (b.leaf() || a.r2>=b.r2);
                const auto xs=cuta ? a.left : b.left,ys=cuta ? a.right : b.right;
                const auto& x=t.nodes[xs]; const auto& y=t.nodes[ys];
                auto x1=cuta ? decide(x,b) : decide(a,x),y1=cuta ? decide(y,b) : decide(a,y);
                const auto& other=cuta ? b : a;
                if (other.leaf() && x.leaf() && !x1.p) x1.work=double(other.count)*x.count/prefactor;
                if (other.leaf() && y.leaf() && !y1.p) y1.work=double(other.count)*y.count/prefactor;
                if (x1.work>0. && y1.work>0. && (x1.work+y1.work < look*u.work)) {
                    accept(cuta ? xs : aa,cuta ? bb : xs,x1);
                    accept(cuta ? ys : aa,cuta ? bb : ys,y1);
                    return;
                }
            }
            accept(aa,bb,u);
        } else if (a.leaf() && b.leaf()) addnear(aa,bb);
        else if (!a.leaf() && (b.leaf() || a.r2 >= b.r2)) {
            walk(a.left,bb);
            walk(a.right,bb);
        } else {
            walk(aa,b.left);
            walk(aa,b.right);
        }
    }
};

struct Work {
    size_t coeffs=0, rawcoeff=0, hcoeff=0;
    double up=0, down=0, far=0, near=0;
    double seconds() const { return up+down+far+near; }
};
inline int direct_precision(double tolerance) {
    return tolerance>8.e-9 ? 2 : tolerance>8.e-13 ? 3 : 4;
}
inline int inherited(int p,const Node& n) {
    return (p<=4 || n.shrink>=1.) ? p : std::min(p,2+int((p+1)*n.shrink));
}
Work forecast(Plan& plan,double calibration) {
    Work w;
    const size_t particles=plan.t.index.size();
    std::vector<std::array<uint8_t,4>> degrees(plan.t.nodes.size());
    size_t pos=0;
    for (const auto& n:plan.t.nodes) {
        w.rawcoeff+=(n.pm ? terms(n.pm) : 0)+(n.pl ? terms(n.pl) : 0);
        w.hcoeff+=(n.hm ? hs(n.hm) : 0)+(n.hl ? hs(n.hl) : 0);
        const auto visit=degrees[pos++];
        const uint8_t pm=std::max<int>(n.pm,inherited(visit[0],n)),pl=std::max<int>(n.pl,inherited(visit[1],n));
        const uint8_t hm=std::max<int>(n.hm,inherited(visit[2],n)),hl=std::max<int>(n.hl,inherited(visit[3],n));
        if (n.leaf()) {
            w.up+=n.count*((pm ? .98e-9*terms(pm) : 0)+(hm ? 1.40e-9*hs(hm) : 0));
            w.down+=n.count*((pl ? .96e-9*terms(pl) : 0)+(hl ? 1.55e-9*hs(hl) : 0));
        } else {
            for (int i:{n.left,n.right}) degrees[i]={pm,pl,hm,hl};
            const auto raw=[](int p) { return p ? p*double(terms(p)) : 0.; };
            const auto h=[](int p) { return p ? p*double(hs(p)) : 0.; };
            w.up+=2.40e-9*raw(pm)+1.30e-9*h(hm);
            w.down+=2.40e-9*raw(pl)+1.30e-9*h(hl);
        }
    }
    w.coeffs=std::max(w.rawcoeff,w.hcoeff);
    w.up+=(w.rawcoeff+w.hcoeff)*8*.085e-9;
    w.far=.276e-9*plan.mlwork+(particles<=262144 ? .286e-9 : .274e-9)*plan.hw;
    if (calibration<=0.) calibration=(1.94+.105*(direct_precision(plan.tol)-2))*1.e-9;
    w.near=calibration*plan.pp;
    return w;
}

template<bool harmonic,class Kernel>
void expansion_up(Kernel& f,int id,const Node* parent,int require,double* dst,
                  double* scratch,size_t stride,size_t& tick) {
    if ((tick++&127)==0) checkclock();
    const auto& n=f.t.nodes[id];
    const int own=harmonic ? n.hm : n.pm, p=std::max(own,inherited(require,n));
    const size_t off=harmonic ? n.hmo : n.mo;
    const auto size=[](int d) { return harmonic ? hs(d) : terms(d); };
    double* out=p ? (own==p ? f.moments.data()+off : scratch) : nullptr;
    if (n.leaf()) {
        if (p) f.p2m(n,p,out);
    } else {
        if (p && own!=p) std::fill(out,out+size(p),0.);
        expansion_up<harmonic>(f,n.left,&n,p,out,scratch+stride,stride,tick);
        expansion_up<harmonic>(f,n.right,&n,p,out,scratch+stride,stride,tick);
    }
    if (own && own!=p) std::copy_n(out,size(own),f.moments.data()+off);
    if (require) {
        if (require>p) {
            if (out!=scratch) std::copy_n(out,size(p),scratch);
            std::fill(scratch+size(p),scratch+size(require),0.);
            out=scratch;
        }
        const double d[3]={n.c[0]-parent->c[0],n.c[1]-parent->c[1],n.c[2]-parent->c[2]};
        f.template shift<true>(out,dst,d,require);
    }
}

template<bool harmonic,class Kernel>
void expansion_down(Kernel& f,int id,const Node* parent,int require,const double* in,
                    double* scratch,size_t stride,size_t& tick) {
    if ((tick++&127)==0) checkclock();
    const auto& n=f.t.nodes[id];
    const int own=harmonic ? n.hl : n.pl, p=std::max(own,inherited(require,n));
    const size_t off=harmonic ? n.hlo : n.lo;
    const auto size=[](int d) { return harmonic ? hs(d) : terms(d); };
    const double* co=own ? f.locals.data()+off : nullptr;
    if (require) {
        const int prefix=own ? size(own) : 0;
        if (prefix) std::copy_n(co,prefix,scratch);
        std::fill(scratch+prefix,scratch+size(std::max(p,require)),0.);
        const double d[3]={n.c[0]-parent->c[0],n.c[1]-parent->c[1],n.c[2]-parent->c[2]};
        f.template shift<false>(in,scratch,d,require);
        co=scratch;
    }
    if (n.leaf()) {
        if (p) f.eval(n,p,co);
    } else {
        expansion_down<harmonic>(f,n.left,&n,p,co,scratch+stride,stride,tick);
        expansion_down<harmonic>(f,n.right,&n,p,co,scratch+stride,stride,tick);
    }
}

template<int order=4>
inline V invcube(V s) {
    const V u = _mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(s)));
    const V u2 = u*u;
    const V e = _mm256_fnmadd_pd(s,u2,set(1.0));
    V c;
    if constexpr (order==4) {
        c = fma(set(2.4609375),e,set(2.1875));
        c = fma(c,e,set(1.875));
    } else if constexpr (order==3) c=fma(set(2.1875),e,set(1.875));
    else c=set(1.875);
    c = fma(c,e,set(1.5));
    c = fma(c,e,set(1.0));
    return (u*u2)*c;
}
template<int order=4>
inline V force(V x, V y, V z) {
    const V s = fma(x,x,fma(y,y,fma(z,z,set(eps2))));
    return invcube<order>(s);
}
inline V tailmask(int count) {
    return _mm256_castsi256_pd(_mm256_set_epi64x(
        count>=4 ? -1LL : 0, count>=3 ? -1LL : 0, count>=2 ? -1LL : 0, count>=1 ? -1LL : 0));
}

__attribute__((noinline)) void direct_self(Tree& t, const Node& a) {
    const int n = a.count, end = (n+3)&-4;
    const auto xa=t.x[0].data()+a.start, ya=t.x[1].data()+a.start, za=t.x[2].data()+a.start;
    auto ax=t.a[0].data()+a.start, ay=t.a[1].data()+a.start, az=t.a[2].data()+a.start;
    const V mask = tailmask(n-end+4);
    for (int i = 0; i < n; ++i) {
        const V xi=set(xa[i]), yi=set(ya[i]), zi=set(za[i]);
        V rx=zero(),ry=zero(),rz=zero();
        for (int j = 0; j < end; j+=4) {
            V dx=load(xa+j)-xi,dy=load(ya+j)-yi,dz=load(za+j)-zi;
            V r=force(dx,dy,dz);
            if (j+4==end) r=_mm256_and_pd(r,mask);
            rx=fma(dx,r,rx); ry=fma(dy,r,ry); rz=fma(dz,r,rz);
        }
        ax[i]+=sum(rx); ay[i]+=sum(ry); az[i]+=sum(rz);
    }
}

__attribute__((noinline)) void direct_pair(Tree& t, const Node& a, const Node& b) {
    const int n = a.count, end = (b.count+3)&-4;
    const auto xa=t.x[0].data()+a.start, ya=t.x[1].data()+a.start, za=t.x[2].data()+a.start;
    const auto xb=t.x[0].data()+b.start, yb=t.x[1].data()+b.start, zb=t.x[2].data()+b.start;
    auto ax=t.a[0].data()+a.start, ay=t.a[1].data()+a.start, az=t.a[2].data()+a.start;
    auto bx=t.a[0].data()+b.start, by=t.a[1].data()+b.start, bz=t.a[2].data()+b.start;
    const V mask = tailmask(b.count-end+4);
    for (int i = 0; i < n; i+=2) {
        const V x0=set(xa[i]), y0=set(ya[i]), z0=set(za[i]);
        const V x1=set(xa[i+1]), y1=set(ya[i+1]), z1=set(za[i+1]);
        V rx0=zero(),ry0=zero(),rz0=zero(),rx1=zero(),ry1=zero(),rz1=zero();
        for (int j = 0; j < end; j+=4) {
            const V xj=load(xb+j),yj=load(yb+j),zj=load(zb+j);
            V dx0=xj-x0,dy0=yj-y0,dz0=zj-z0;
            V r0=force(dx0,dy0,dz0);
            if (j+4==end) r0=_mm256_and_pd(r0,mask);
            V dx1=xj-x1,dy1=yj-y1,dz1=zj-z1;
            V r1=force(dx1,dy1,dz1);
            if (j+4==end) r1=_mm256_and_pd(r1,mask);
            if (i+1==n) r1=zero();
            const V fx0=dx0*r0, fy0=dy0*r0, fz0=dz0*r0;
            const V fx1=dx1*r1, fy1=dy1*r1, fz1=dz1*r1;
            rx0+=fx0; ry0+=fy0; rz0+=fz0;
            rx1+=fx1; ry1+=fy1; rz1+=fz1;
            store(bx+j,load(bx+j)-(fx0+fx1));
            store(by+j,load(by+j)-(fy0+fy1));
            store(bz+j,load(bz+j)-(fz0+fz1));
        }
        ax[i]+=sum(rx0); ay[i]+=sum(ry0); az[i]+=sum(rz0);
        ax[i+1]+=sum(rx1); ay[i+1]+=sum(ry1); az[i+1]+=sum(rz1);
    }
}

// Hold the source quartet and its accumulator in registers. Only the target
// is reduced horizontally inside this loop.
__attribute__((noinline)) void direct_pair2(Tree& t, const Node& a, const Node& b) {
    const int n=a.count, end=b.count&-4;
    const auto xa=t.x[0].data()+a.start, ya=t.x[1].data()+a.start, za=t.x[2].data()+a.start;
    const auto xb=t.x[0].data()+b.start, yb=t.x[1].data()+b.start, zb=t.x[2].data()+b.start;
    auto ax=t.a[0].data()+a.start, ay=t.a[1].data()+a.start, az=t.a[2].data()+a.start;
    auto bx=t.a[0].data()+b.start, by=t.a[1].data()+b.start, bz=t.a[2].data()+b.start;
    for (int j=0;j<end;j+=4) {
        const V xj=load(xb+j),yj=load(yb+j),zj=load(zb+j);
        V rx=zero(),ry=zero(),rz=zero();
        for (int i=0;i<n;++i) {
            V dx=xj-set(xa[i]),dy=yj-set(ya[i]),dz=zj-set(za[i]);
            const V r=force(dx,dy,dz);
            dx*=r; dy*=r; dz*=r;
            rx+=dx; ry+=dy; rz+=dz;
            ax[i]+=sum(dx); ay[i]+=sum(dy); az[i]+=sum(dz);
        }
        store(bx+j,load(bx+j)-rx);
        store(by+j,load(by+j)-ry);
        store(bz+j,load(bz+j)-rz);
    }
    if (end != b.count) {
        const V xj=load(xb+end),yj=load(yb+end),zj=load(zb+end), mask=tailmask(b.count-end);
        V rx=zero(),ry=zero(),rz=zero();
        for (int i=0;i<n;++i) {
            V dx=xj-set(xa[i]),dy=yj-set(ya[i]),dz=zj-set(za[i]);
            const V r=_mm256_and_pd(force(dx,dy,dz),mask);
            dx*=r; dy*=r; dz*=r;
            rx+=dx; ry+=dy; rz+=dz;
            ax[i]+=sum(dx); ay[i]+=sum(dy); az[i]+=sum(dz);
        }
        store(bx+end,load(bx+end)-rx); store(by+end,load(by+end)-ry); store(bz+end,load(bz+end)-rz);
    }
}

template<int order>
__attribute__((noinline)) void direct_one(Tree& t, const Node& a, const Node& b) {
    const int n=a.count, end=b.count&-4;
    const auto xa=t.x[0].data()+a.start,ya=t.x[1].data()+a.start,za=t.x[2].data()+a.start;
    const auto xb=t.x[0].data()+b.start,yb=t.x[1].data()+b.start,zb=t.x[2].data()+b.start;
    auto ax=t.a[0].data()+a.start,ay=t.a[1].data()+a.start,az=t.a[2].data()+a.start;
    auto bx=t.a[0].data()+b.start,by=t.a[1].data()+b.start,bz=t.a[2].data()+b.start;
    const V mask=tailmask(b.count-end);
    for (int i=0;i<n;++i) {
        const V xi=set(xa[i]),yi=set(ya[i]),zi=set(za[i]);
        V fx=zero(),fy=zero(),fz=zero();
        const auto apply=[&](int j,bool tail) {
            const V dx=xi-load(xb+j),dy=yi-load(yb+j),dz=zi-load(zb+j);
            V f=force<order>(dx,dy,dz);
            if (tail) f=_mm256_and_pd(f,mask);
            fx=_mm256_fnmadd_pd(dx,f,fx); fy=_mm256_fnmadd_pd(dy,f,fy); fz=_mm256_fnmadd_pd(dz,f,fz);
            store(bx+j,fma(dx,f,load(bx+j)));
            store(by+j,fma(dy,f,load(by+j)));
            store(bz+j,fma(dz,f,load(bz+j)));
        };
        for (int j=0;j<end;j+=4) apply(j,false);
        if (end<b.count) apply(end,true);
        ax[i]+=sum(fx); ay[i]+=sum(fy); az[i]+=sum(fz);
    }
}

inline V dup_pair(const double* p) {
    const __m128d u=_mm_loadu_pd(p);
    return _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_unpacklo_pd(u,u)),_mm_unpackhi_pd(u,u),1);
}
inline V copy_pair(const double* p) {
    return _mm256_broadcast_pd(reinterpret_cast<const __m128d*>(p));
}
inline __m128d sum_halves(V u) {
    return _mm_add_pd(_mm256_castpd256_pd128(u),_mm256_extractf128_pd(u,1));
}
inline void add2(double* p,V u) {
    _mm_storeu_pd(p,_mm_add_pd(_mm_loadu_pd(p),_mm256_castpd256_pd128(
                 _mm256_permute4x64_pd(_mm256_hadd_pd(u,u),0xd8))));
}
template<int order>
__attribute__((noinline)) void direct_two(Tree& t, const Node& a, const Node& b) {
    const int n=a.count, end=b.count&-2;
    const auto xa=t.x[0].data()+a.start,ya=t.x[1].data()+a.start,za=t.x[2].data()+a.start;
    const auto xb=t.x[0].data()+b.start,yb=t.x[1].data()+b.start,zb=t.x[2].data()+b.start;
    auto ax=t.a[0].data()+a.start,ay=t.a[1].data()+a.start,az=t.a[2].data()+a.start;
    auto bx=t.a[0].data()+b.start,by=t.a[1].data()+b.start,bz=t.a[2].data()+b.start;
    const V bmask=_mm256_castsi256_pd(_mm256_set_epi64x(0,-1LL,0,-1LL));
    const V amask=_mm256_castsi256_pd(_mm256_set_epi64x(0,0,-1LL,-1LL));
    const auto pair=[&](int i,bool only) {
        const V xi=dup_pair(xa+i),yi=dup_pair(ya+i),zi=dup_pair(za+i);
        V fx=zero(),fy=zero(),fz=zero();
        const auto apply=[&](int j,bool tail) {
            V dx=copy_pair(xb+j)-xi,dy=copy_pair(yb+j)-yi,dz=copy_pair(zb+j)-zi;
            V f=force<order>(dx,dy,dz);
            if (tail) f=_mm256_and_pd(f,bmask);
            if (only) f=_mm256_and_pd(f,amask);
            dx*=f; dy*=f; dz*=f;
            fx+=dx; fy+=dy; fz+=dz;
            _mm_storeu_pd(bx+j,_mm_sub_pd(_mm_loadu_pd(bx+j),sum_halves(dx)));
            _mm_storeu_pd(by+j,_mm_sub_pd(_mm_loadu_pd(by+j),sum_halves(dy)));
            _mm_storeu_pd(bz+j,_mm_sub_pd(_mm_loadu_pd(bz+j),sum_halves(dz)));
        };
        for (int j=0;j<end;j+=2) apply(j,false);
        if (end<b.count) apply(end,true);
        add2(ax+i,fx); add2(ay+i,fy); add2(az+i,fz);
    };
    for (int i=0;i+1<n;i+=2) pair(i,false);
    if (n%2) pair(n-1,true);
}

template<int order>
__attribute__((noinline)) void direct_four(Tree& t,const Node& a,const Node& b) {
    const int na=a.count&-4,nb=b.count&-4;
    const auto xa=t.x[0].data()+a.start,ya=t.x[1].data()+a.start,za=t.x[2].data()+a.start;
    const auto xb=t.x[0].data()+b.start,yb=t.x[1].data()+b.start,zb=t.x[2].data()+b.start;
    auto ax=t.a[0].data()+a.start,ay=t.a[1].data()+a.start,az=t.a[2].data()+a.start;
    auto bx=t.a[0].data()+b.start,by=t.a[1].data()+b.start,bz=t.a[2].data()+b.start;
    const auto rot=[](V v) { return _mm256_permute4x64_pd(v,0x39); };
    for (int i=0;i<na;i+=4) {
        const V xi=load(xa+i),yi=load(ya+i),zi=load(za+i);
        V fx=load(ax+i),fy=load(ay+i),fz=load(az+i);
        for (int j=0;j<nb;j+=4) {
            V xj=load(xb+j),yj=load(yb+j),zj=load(zb+j);
            V gx=load(bx+j),gy=load(by+j),gz=load(bz+j);
            const auto step=[&] {
                const V dx=xj-xi,dy=yj-yi,dz=zj-zi;
                const V h=force<order>(dx,dy,dz);
                const V x=dx*h,y=dy*h,z=dz*h;
                fx+=x; fy+=y; fz+=z;
                gx=rot(gx-x); gy=rot(gy-y); gz=rot(gz-z);
                xj=rot(xj); yj=rot(yj); zj=rot(zj);
            };
            step(); step(); step(); step();
            store(bx+j,gx); store(by+j,gy); store(bz+j,gz);
        }
        store(ax+i,fx); store(ay+i,fy); store(az+i,fz);
    }
    if (na<a.count) {
        Node r=a;
        r.start+=na; r.count-=na;
        direct_one<order>(t,r,b);
    }
    if (na && nb<b.count) {
        Node x=a,r=b;
        x.count=na; r.start+=nb; r.count-=nb;
        direct_one<order>(t,x,r);
    }
}

struct Fmm {
    Tree& t;
    Basis bs;
    int pmax;
    std::vector<double> moments, locals;
    explicit Fmm(Tree& t_, int p, int cp, bool pairs=true) : t(t_), bs(p,cp,pairs), pmax(p) {}

    void setup() {
        size_t nm = 0, nl = 0;
        for (auto& n : t.nodes) {
            n.mo = nm; n.lo = nl;
            if (n.pm) nm+=terms(n.pm);
            if (n.pl) nl+=terms(n.pl);
        }
        if ((nm+nl)*sizeof(double) > max_coefficient_bytes) throw std::bad_alloc();
        moments.resize(nm);
        locals.resize(nl);
    }

    void p2m(const Node& n,int p,double* out) {
        V accum[maxterms];
        std::fill_n(accum,terms(p),zero());
        const auto xx=t.x[0].data()+n.start, yy=t.x[1].data()+n.start, zz=t.x[2].data()+n.start;
        const V c0=set(n.c[0]),c1=set(n.c[1]),c2=set(n.c[2]);
        const int end=(n.count+3)&-4;
        const V mask=tailmask(n.count-end+4);
        for (int j = 0; j < end; j+=4) {
            const V dx=load(xx+j)-c0,dy=load(yy+j)-c1,dz=load(zz+j)-c2;
            V xp[maxp+1],yp[maxp+1],zp[maxp+1];
            xp[0]=yp[0]=set(1.);
            zp[0]=j+4==end ? _mm256_and_pd(set(1.),mask) : set(1.);
            for (int i = 1; i<=p; ++i) { xp[i]=xp[i-1]*dx; yp[i]=yp[i-1]*dy; zp[i]=zp[i-1]*dz; }
            int i=0;
            for (int z=0; z<=p; ++z) for (int y=0; y<=p-z; ++y) {
                const V yz=yp[y]*zp[z];
                for (int x=0; x<=p-z-y; ++x,++i)
                    accum[i]=fma(yz,xp[x],accum[i]);
            }
        }
        int i=0;
        for (int z=0;z<=p;++z) for (int y=0;y<=p-z;++y) for (int x=0;x<=p-z-y;++x,++i) {
            const int j=idx(x,y,z);
            out[j]=sum(accum[i])*bs.xyzfactinv[j];
        }
    }

    template<bool up>
    void shift(const double* in, double* out, const double* d, int p) {
        double r[maxterms+1];
        r[0]=0.;
        std::copy_n(in,terms(p),r+1);
        for (int k=p;k>0;--k) {
            const double w=1./k, x=w*d[0],y=w*d[1],z=w*d[2];
            if constexpr (up) {
                for (int j=p-k+1;j>=1;--j) for (int i=terms(j-1);i<terms(j);++i) {
                    const auto* m=bs.lowers[i];
                    r[i+1]=std::fma(z,r[m[2]],std::fma(y,r[m[1]],std::fma(x,r[m[0]],in[i])));
                }
            } else {
                for (int i=terms(k-2);i<terms(p-1);++i) {
                    const auto* m=bs.high[i];
                    r[i+1]=std::fma(z,r[m[2]],std::fma(y,r[m[1]],std::fma(x,r[m[0]],in[i])));
                }
            }
        }
        for (int i=0;i<terms(p);++i) out[i]+=r[i+1];
    }

    void upward() {
        if (moments.empty()) return;
        size_t tick=0,stride=terms(pmax);
        std::vector<double> tmp(stride*t.levels);
        expansion_up<false>(*this,0,nullptr,0,nullptr,tmp.data(),stride,tick);
    }

    void eval(const Node& n,int p,const double* co) {
        double coeff[maxterms];
        int i=0;
        for (int z=0;z<=p;++z) for (int y=0;y<=p-z;++y) for (int x=0;x<=p-z-y;++x,++i) {
            const int j=idx(x,y,z);
            coeff[i]=co[j]*bs.xyzfactinv[j];
        }
        const auto xx=t.x[0].data()+n.start, yy=t.x[1].data()+n.start, zz=t.x[2].data()+n.start;
        auto ax=t.a[0].data()+n.start,ay=t.a[1].data()+n.start,az=t.a[2].data()+n.start;
        const V c0=set(n.c[0]),c1=set(n.c[1]),c2=set(n.c[2]);
        for (int j=0; j < n.count; j+=4) {
            const V x=load(xx+j)-c0,y=load(yy+j)-c1,z=load(zz+j)-c2;
            V r=zero(),rx=zero(),ry=zero(),rz=zero();
            int i=terms(p);
            for (int k=p;k>=0;--k) {
                V s=zero(),sx=zero(),sy=zero();
                for (int l=p-k;l>=0;--l) {
                    const int start=p-k-l;
                    V v=set(coeff[--i]), vx=zero();
                    for (int m=start-1;m>=0;--m) {
                        vx=fma(x,vx,v);
                        v=fma(x,v,set(coeff[--i]));
                    }
                    sy=fma(y,sy,s); s=fma(y,s,v); sx=fma(y,sx,vx);
                }
                rz=fma(z,rz,r); r=fma(z,r,s);
                rx=fma(z,rx,sx); ry=fma(z,ry,sy);
            }
            store(ax+j,load(ax+j)+rx); store(ay+j,load(ay+j)+ry); store(az+j,load(az+j)+rz);
        }
    }

    void downward() {
        if (locals.empty()) return;
        size_t tick=0,stride=terms(pmax);
        std::vector<double> tmp(stride*t.levels);
        expansion_down<false>(*this,0,nullptr,0,nullptr,tmp.data(),stride,tick);
    }

    template<bool correction=false>
    void m2l(const Pair* pairs, int num, int p,int harmonic_p=0) {
        int ca=pairs[0].ca(),cb=pairs[0].cb();
        if constexpr (correction) {
            ca=residual_order(p,ca,harmonic_p);
            cb=residual_order(p,cb,harmonic_p);
        }
        const Node *a[4],*b[4];
        double dx[4],dy[4],dz[4];
        for (int i=0; i<4; ++i) {
            const auto pair=pairs[std::min(i,num-1)];
            a[i]=&t.nodes[pair.first()]; b[i]=&t.nodes[pair.second()];
            dx[i]=b[i]->c[0]-a[i]->c[0]; dy[i]=b[i]->c[1]-a[i]->c[1]; dz[i]=b[i]->c[2]-a[i]->c[2];
        }
        V der[maxterms+1], ma[maxterms],mb[maxterms];
        const V xd=load(dx),yd=load(dy),zd=load(dz);
        bs.derivatives(der,p,xd,yd,zd,inverse_distance(xd,yd,zd));
        if constexpr (correction) {
            V hfull[maxterms];
            for (int i=0; i<terms(p); ++i) {
                const auto a=bs.xyz[i];
                if (a.z<2) hfull[i]=der[i+1];
                else {
                    const V v=-hfull[idx(a.x+2,a.y,a.z-2)]-hfull[idx(a.x,a.y+2,a.z-2)];
                    hfull[i]=v;
                    der[i+1]-=v;
                }
            }
        }
        const auto *a0=moments.data()+a[0]->mo, *a1=moments.data()+a[1]->mo;
        const auto *a2=moments.data()+a[2]->mo, *a3=moments.data()+a[3]->mo;
        const auto *b0=moments.data()+b[0]->mo, *b1=moments.data()+b[1]->mo;
        const auto *b2=moments.data()+b[2]->mo, *b3=moments.data()+b[3]->mo;
        const int ma_deg=std::min(ca,p-1),mb_deg=std::min(cb,p-1);
        pack_columns<true>(ma,a0,a1,a2,a3,terms(ma_deg),bs.xyz);
        pack_columns<false>(mb,b0,b1,b2,b3,terms(mb_deg),bs.xyz);
        double x[4],y[4];
        for (int d=1; d<=std::max(ca,cb); ++d) {
            const int aa=std::min(ma_deg,p-d),bb=std::min(mb_deg,p-d);
            for (int k=terms(d-1); k<terms(d); ++k) {
                V s0=zero(),s1=zero(),t0=zero(),t1=zero();
                int na,nb;
                const uint32_t* cg=nullptr;
                const uint16_t* fg=nullptr;
                if constexpr (correction) {
                    cg=bs.cschedule.data()+bs.crow[k];
                    na=d<=cb ? bs.clen[k][aa] : 0;
                    nb=d<=ca ? bs.clen[k][bb] : 0;
                } else {
                    fg=bs.schedule.data()+bs.row[k];
                    na=d<=cb ? terms(aa) : 0;
                    nb=d<=ca ? terms(bb) : 0;
                }
                const auto at=[&](int i,int& ord) {
                    if constexpr (correction) {
                        const uint32_t c=cg[i];
                        ord=c&65535;
                        return der[c>>16];
                    } else {
                        ord=i;
                        return der[fg[i]];
                    }
                };
                const int n=std::min(na,nb);
                int i=0;
                for (;i+1<n;i+=2) {
                    int j,l;
                    const V f=at(i,j),h=at(i+1,l);
                    s0=fma(f,ma[j],s0); s1=fma(h,ma[l],s1);
                    t0=fma(f,mb[j],t0); t1=fma(h,mb[l],t1);
                }
                if (i<n) {
                    int j;
                    const V f=at(i++,j);
                    s0=fma(f,ma[j],s0); t0=fma(f,mb[j],t0);
                }
                if (na>n) {
                    for (;i+1<na;i+=2) {
                        int j,l;
                        const V f=at(i,j),h=at(i+1,l);
                        s0=fma(f,ma[j],s0); s1=fma(h,ma[l],s1);
                    }
                    if (i<na) { int j; const V f=at(i,j); s0=fma(f,ma[j],s0); }
                } else if (nb>n) {
                    for (;i+1<nb;i+=2) {
                        int j,l;
                        const V f=at(i,j),h=at(i+1,l);
                        t0=fma(f,mb[j],t0); t1=fma(h,mb[l],t1);
                    }
                    if (i<nb) { int j; const V f=at(i,j); t0=fma(f,mb[j],t0); }
                }
                const V tz=t0+t1;
                store(x,d%2 ? -tz : tz);
                store(y,s0+s1);
                for (int l=0;l<num;++l) {
                    if (nb) locals[a[l]->lo+k]+=x[l];
                    if (na) locals[b[l]->lo+k]+=y[l];
                }
            }
        }
    }
};

constexpr int hidx(int x,int y,int z) {
    const int d=x+y+z;
    return d*d+y+(z ? d+1 : 0);
}
constexpr int idx2(int x,int y) {
    const int d=x+y;
    return d*(d+1)/2+y;
}
constexpr int hnone=maxhs;
constexpr int hext=maxhs+1;
constexpr int hdercount=hext+(maxp-1)*maxp/2;

// The leading expansion is harmonic in the displacement. It is the mixture
// of the two Newton kernels with axial separations +/-sqrt(Dz^2+eps^2)
// that matches the softened kernel and its first z derivative at z=0.
// Its moments are represented modulo Z^2+X^2+Y^2. The residual is handled
// with the full softened derivatives minus their harmonic extension.
struct Harmonic {
    Tree& t;
    int pmax;
    bool pair_rows;
    struct Op {
        uint16_t low[4], high[4], der[5];
    };
    Op op[maxhs];
    XYZ xyz[maxhs];
    double normalization[maxhs], lin[maxp+1], sq[maxp+1];
    uint32_t row[maxhs];
    std::vector<uint16_t> schedule;
    std::vector<double> moments,locals;

    explicit Harmonic(Tree& t_,int p) : t(t_),pmax(p),pair_rows(optint("GAL_ROWS",2)>1) {
        if (!p) return;
        double fact[maxp+1]{1.};
        for (int i=1;i<=p;++i) {
            fact[i]=fact[i-1]*i;
            lin[i]=(2.*i-1)/i;
            sq[i]=(i-1.)/i;
        }
        const auto at=[](int x,int y,int z) { return x<0 || y<0 ? hnone : hidx(x,y,z); };
        for (int d=0;d<=p;++d) for (int z=0;z<=1;++z) for (int y=0;y<=d-z;++y) {
            const int x=d-y-z, i=hidx(x,y,z);
            xyz[i]={x,y,z,d};
            auto& o=op[i];
            o.low[0]=at(x-1,y,z);
            o.low[1]=at(x,y-1,z);
            o.low[2]=z ? at(x,y,0) : at(x-2,y,1);
            o.low[3]=z ? hnone : at(x,y-2,1);
            if (d<maxp) {
                o.high[0]=at(x+1,y,z);
                o.high[1]=at(x,y+1,z);
                o.high[2]=z ? at(x+2,y,0) : at(x,y,1);
                o.high[3]=z ? at(x,y+2,0) : hnone;
            }
            o.der[0]=o.low[0];
            o.der[1]=o.low[1];
            o.der[2]=z ? at(x,y,0) : hnone;
            o.der[3]=at(x-2,y,z);
            o.der[4]=at(x,y-2,z);
            normalization[i]=fact[x]*fact[y];
        }
        for (int i=0;i<hs(p);++i) {
            row[i]=schedule.size();
            const auto b=xyz[i];
            for (int j=0;j<hs(p-b.deg);++j) {
                auto a=xyz[j];
                const int x=a.x+b.x,y=a.y+b.y,z=a.z+b.z;
                schedule.push_back(z<2 ? hidx(x,y,z) : hext+idx2(x,y));
            }
        }
    }

    void setup(size_t rawbytes) {
        size_t nm=0,nl=0;
        for (auto& n:t.nodes) {
            n.hmo=nm; n.hlo=nl;
            if (n.hm) nm+=hs(n.hm);
            if (n.hl) nl+=hs(n.hl);
        }
        if (rawbytes+(nm+nl)*sizeof(double)>max_coefficient_bytes) throw std::bad_alloc();
        moments.resize(nm);
        locals.resize(nl);
    }

    void project(V* v,const V* dx,const V* dy,const V* dz,int d) const {
        const V x=dx[d], y=dy[d], z=dz[d], nz=-z;
        const int start=d*d, mid=start+d+1, end=hs(d);
        for (int i=start;i<mid;++i) {
            const auto* j=op[i].low;
            v[i]=fma(v[j[2]]+v[j[3]],nz,fma(v[j[1]],y,v[j[0]]*x));
        }
        for (int i=mid;i<end;++i) {
            const auto* j=op[i].low;
            v[i]=fma(v[j[2]],z,fma(v[j[1]],y,v[j[0]]*x));
        }
    }

    void p2m(const Node& n,int p,double* out) {
        V v[maxhs+1], acc[maxhs];
        std::fill_n(acc,hs(p),zero());
        v[hnone]=zero();
        const auto xx=t.x[0].data()+n.start,yy=t.x[1].data()+n.start,zz=t.x[2].data()+n.start;
        const V c0=set(n.c[0]),c1=set(n.c[1]),c2=set(n.c[2]);
        const int end=(n.count+3)&-4;
        const V mask=tailmask(n.count-end+4);
        for (int j=0;j<end;j+=4) {
            const V x=load(xx+j)-c0,y=load(yy+j)-c1,z=load(zz+j)-c2;
            V dx[maxp+1],dy[maxp+1],dz[maxp+1];
            v[0]=j+4==end ? _mm256_and_pd(set(1.),mask) : set(1.);
            for (int d=1;d<=p;++d) {
                const V r=set(1./d);
                dx[d]=x*r; dy[d]=y*r; dz[d]=z*r;
                project(v,dx,dy,dz,d);
            }
            for (int i=1;i<hs(p);++i) acc[i]+=v[i];
        }
        out[0]=n.count;
        for (int i=1;i<hs(p);++i) out[i]=sum(acc[i]);
    }

    template<bool up>
    void shift(const double* in,double* out,const double* d,int p) {
        double r[maxhs+1];
        r[hnone]=0.;
        std::copy(in,in+hs(p),r);
        for (int s=p;s>=1;--s) {
            const double fac=1./s,x=d[0]*fac,y=d[1]*fac,z=d[2]*fac;
            if constexpr (up) {
                for (int j=p-s+1;j>0;--j) {
                    const int mid=j*j+j+1;
                    for (int i=j*j;i<mid;++i) {
                        const auto* k=op[i].low;
                        r[i]=std::fma(-z,r[k[2]]+r[k[3]],std::fma(x,r[k[0]],std::fma(y,r[k[1]],in[i])));
                    }
                    for (int i=mid;i<hs(j);++i) {
                        const auto* k=op[i].low;
                        r[i]=std::fma(z,r[k[2]],std::fma(x,r[k[0]],std::fma(y,r[k[1]],in[i])));
                    }
                }
            } else {
                for (int j=s-1;j<p;++j) {
                    const int mid=j*j+j+1;
                    for (int i=j*j;i<mid;++i) {
                        const auto* k=op[i].high;
                        r[i]=std::fma(z,r[k[2]],std::fma(x,r[k[0]],std::fma(y,r[k[1]],in[i])));
                    }
                    for (int i=mid;i<hs(j);++i) {
                        const auto* k=op[i].high;
                        r[i]=std::fma(-z,r[k[2]]+r[k[3]],std::fma(x,r[k[0]],std::fma(y,r[k[1]],in[i])));
                    }
                }
            }
        }
        for (int i=0;i<hs(p);++i) out[i]+=r[i];
    }

    void upward() {
        if (moments.empty()) return;
        size_t tick=0,stride=hs(pmax);
        std::vector<double> tmp(stride*t.levels);
        expansion_up<true>(*this,0,nullptr,0,nullptr,tmp.data(),stride,tick);
    }

    void eval(const Node& n,int p,const double* co) {
        V v[maxhs+1];
        v[hnone]=zero();
        v[0]=set(1.);
        double gx[maxhs],gy[maxhs],gz[maxhs];
        for (int i=0;i<hs(p-1);++i) {
            const auto* j=op[i].high;
            gx[i]=co[j[0]]; gy[i]=co[j[1]];
            gz[i]=xyz[i].z ? -co[j[2]]-co[j[3]] : co[j[2]];
        }
        const auto xx=t.x[0].data()+n.start,yy=t.x[1].data()+n.start,zz=t.x[2].data()+n.start;
        auto ax=t.a[0].data()+n.start,ay=t.a[1].data()+n.start,az=t.a[2].data()+n.start;
        const V c0=set(n.c[0]),c1=set(n.c[1]),c2=set(n.c[2]);
        for (int j=0;j<n.count;j+=4) {
            const V x=load(xx+j)-c0,y=load(yy+j)-c1,z=load(zz+j)-c2;
            V dx[maxp+1],dy[maxp+1],dz[maxp+1];
            for (int d=1;d<p;++d) {
                const V r=set(1./d);
                dx[d]=x*r; dy[d]=y*r; dz[d]=z*r;
                project(v,dx,dy,dz,d);
            }
            V rx=load(ax+j)+set(gx[0]),ry=load(ay+j)+set(gy[0]),rz=load(az+j)+set(gz[0]);
            for (int i=1;i<hs(p-1);++i) {
                rx=fma(v[i],set(gx[i]),rx);
                ry=fma(v[i],set(gy[i]),ry);
                rz=fma(v[i],set(gz[i]),rz);
            }
            store(ax+j,rx); store(ay+j,ry); store(az+j,rz);
        }
    }

    void downward() {
        if (locals.empty()) return;
        size_t tick=0,stride=hs(pmax);
        std::vector<double> tmp(stride*t.levels);
        expansion_down<true>(*this,0,nullptr,0,nullptr,tmp.data(),stride,tick);
    }

    static inline void two_rows(const V* der,const V* ma,const V* mb,const uint16_t* g,const uint16_t* h,
                                int na,int nb,V& u0,V& u1,V& v0,V& v1) {
        V aa0=zero(),aa1=zero(),ab0=zero(),ab1=zero(),ba0=zero(),ba1=zero(),bb0=zero(),bb1=zero();
        const int n=std::min(na,nb);
        int i=0;
        for (;i+1<n;i+=2) {
            const V x=ma[i],y=mb[i];
            const V u=der[g[i]],v=der[h[i]];
            aa0=fma(u,x,aa0); ab0=fma(u,y,ab0);
            ba0=fma(v,x,ba0); bb0=fma(v,y,bb0);
            const V X=ma[i+1],Y=mb[i+1];
            const V U=der[g[i+1]],W=der[h[i+1]];
            aa1=fma(U,X,aa1); ab1=fma(U,Y,ab1);
            ba1=fma(W,X,ba1); bb1=fma(W,Y,bb1);
        }
        if (i<n) {
            const V x=ma[i],y=mb[i],u=der[g[i]],v=der[h[i]];
            aa0=fma(u,x,aa0); ab0=fma(u,y,ab0);
            ba0=fma(v,x,ba0); bb0=fma(v,y,bb0);
            ++i;
        }
        if (na>n) {
            for (;i+1<na;i+=2) {
                const V x=ma[i],y=ma[i+1];
                aa0=fma(der[g[i]],x,aa0); aa1=fma(der[g[i+1]],y,aa1);
                ba0=fma(der[h[i]],x,ba0); ba1=fma(der[h[i+1]],y,ba1);
            }
            if (i<na) {
                aa0=fma(der[g[i]],ma[i],aa0);
                ba0=fma(der[h[i]],ma[i],ba0);
            }
        } else if (nb>n) {
            for (;i+1<nb;i+=2) {
                const V x=mb[i],y=mb[i+1];
                ab0=fma(der[g[i]],x,ab0); ab1=fma(der[g[i+1]],y,ab1);
                bb0=fma(der[h[i]],x,bb0); bb1=fma(der[h[i+1]],y,bb1);
            }
            if (i<nb) {
                ab0=fma(der[g[i]],mb[i],ab0);
                bb0=fma(der[h[i]],mb[i],bb0);
            }
        }
        u0=aa0+aa1; v0=ab0+ab1; u1=ba0+ba1; v1=bb0+bb1;
    }

    void m2l(const Pair* pairs,int num,int p) {
        const int ca=pairs[0].ca(),cb=pairs[0].cb();
        const Node *a[4],*b[4];
        double dx[4],dy[4],dz[4];
        for (int i=0;i<4;++i) {
            const auto pair=pairs[std::min(i,num-1)];
            a[i]=&t.nodes[pair.first()]; b[i]=&t.nodes[pair.second()];
            dx[i]=b[i]->c[0]-a[i]->c[0];
            dy[i]=b[i]->c[1]-a[i]->c[1];
            dz[i]=b[i]->c[2]-a[i]->c[2];
        }
        const V x0=load(dx),y0=load(dy),z0=load(dz),vv=inverse_distance(x0,y0,z0);
        const V is=-vv*vv,cx=x0*is,cy=y0*is,cz=z0*is;
        V der[hdercount],ma[maxhs],mb[maxhs];
        der[0]=vv; der[hnone]=zero();
        for (int d=1;d<=p;++d) {
            const V w=set(lin[d]),x=cx*w,y=cy*w,z=cz*w,h=is*set(sq[d]);
            for (int i=d*d;i<hs(d);++i) {
                const auto* j=op[i].der;
                const V u=fma(der[j[2]],z,fma(der[j[1]],y,der[j[0]]*x));
                der[i]=fma(der[j[3]]+der[j[4]],h,u);
            }
        }
        for (int i=1;i<hs(p);++i) der[i]*=set(normalization[i]);
        for (int d=0;d<=p-2;++d) for (int y=0;y<=d;++y)
            der[hext+idx2(d-y,y)]=-der[hidx(d-y+2,y,0)]-der[hidx(d-y,y+2,0)];
        const auto *a0=moments.data()+a[0]->hmo,*a1=moments.data()+a[1]->hmo;
        const auto *a2=moments.data()+a[2]->hmo,*a3=moments.data()+a[3]->hmo;
        const auto *b0=moments.data()+b[0]->hmo,*b1=moments.data()+b[1]->hmo;
        const auto *b2=moments.data()+b[2]->hmo,*b3=moments.data()+b[3]->hmo;
        const int imax=hs(std::min(ca,p-1)),jmax=hs(std::min(cb,p-1));
        pack_columns<true>(ma,a0,a1,a2,a3,imax,xyz);
        pack_columns<false>(mb,b0,b1,b2,b3,jmax,xyz);
        double x[4],y[4];
        for (int d=1;d<=std::max(ca,cb);++d) {
            const int na=d<=cb ? std::min(imax,hs(p-d)) : 0;
            const int nb=d<=ca ? std::min(jmax,hs(p-d)) : 0;
            const int n=std::min(na,nb);
            int k=d*d;
            if (pair_rows) for (;k+1<hs(d);k+=2) {
                V u0,u1,v0,v1;
                two_rows(der,ma,mb,schedule.data()+row[k],schedule.data()+row[k+1],na,nb,u0,u1,v0,v1);
                double x1[4], y1[4];
                store(x,d%2 ? -v0 : v0); store(x1,d%2 ? -v1 : v1);
                store(y,u0); store(y1,u1);
                for (int l=0;l<num;++l) {
                    if (nb) {
                        locals[a[l]->hlo+k]+=x[l]; locals[a[l]->hlo+k+1]+=x1[l];
                    }
                    if (na) {
                        locals[b[l]->hlo+k]+=y[l]; locals[b[l]->hlo+k+1]+=y1[l];
                    }
                }
            }
            for (;k<hs(d);++k) {
                V s0=zero(),s1=zero(),t0=zero(),t1=zero();
                const uint16_t* g=schedule.data()+row[k];
                int i=0;
                for (;i+1<n;i+=2) {
                    const V f=der[g[i]],h=der[g[i+1]];
                    s0=fma(f,ma[i],s0); s1=fma(h,ma[i+1],s1);
                    t0=fma(f,mb[i],t0); t1=fma(h,mb[i+1],t1);
                }
                if (i<n) {
                    const V f=der[g[i]];
                    s0=fma(f,ma[i],s0); t0=fma(f,mb[i],t0);
                    ++i;
                }
                if (na>n) {
                    for (;i+1<na;i+=2) {
                        s0=fma(der[g[i]],ma[i],s0);
                        s1=fma(der[g[i+1]],ma[i+1],s1);
                    }
                    if (i<na) s0=fma(der[g[i]],ma[i],s0);
                } else if (nb>n) {
                    for (;i+1<nb;i+=2) {
                        t0=fma(der[g[i]],mb[i],t0);
                        t1=fma(der[g[i+1]],mb[i+1],t1);
                    }
                    if (i<nb) t0=fma(der[g[i]],mb[i],t0);
                }
                const V tz=t0+t1;
                store(x,d%2 ? -tz : tz);
                store(y,s0+s1);
                for (int l=0;l<num;++l) {
                    if (nb) locals[a[l]->hlo+k]+=x[l];
                    if (na) locals[b[l]->hlo+k]+=y[l];
                }
            }
        }
    }
};

bool read_input(const char* name, Tree& t, double& mass) {
    auto* f=std::fopen(name,"rb");
    if (!f) return false;
    struct stat st{};
    if (fstat(fileno(f),&st) || st.st_size<=0 || st.st_size%32 || st.st_size/32 > (1<<24)) {
        std::fclose(f);
        return false;
    }
    size_t n=st.st_size/32;
    t.bodies.resize(n);
    double buf[2048][4];
    for (size_t i=0;i<n;i+=2048) {
        const size_t k=std::min<size_t>(2048,n-i);
        if (std::fread(buf,sizeof(buf[0]),k,f)!=k) {
            std::fclose(f);
            return false;
        }
        if (!i) mass=buf[0][0];
        for (size_t j=0;j<k;++j) {
            t.bodies[i+j]={{buf[j][1],buf[j][2],buf[j][3]},uint32_t(i+j)};
        }
    }
    return std::fclose(f)==0;
}

bool write_output(const char* name, const Tree& t, double mass) {
    auto* f=std::fopen(name,"wb");
    if (!f) return false;
    const size_t n=t.index.size();
    double buf[2048][3];
    for (size_t i=0;i<n;i+=2048) {
        const size_t k=std::min<size_t>(2048,n-i);
        for (size_t j=0;j<k;++j) {
            const auto q=t.index[i+j];
            for (int d=0;d<3;++d) {
                double z=t.a[d][q];
                if (t.rotated) z=t.rot[0][d]*t.a[0][q]+t.rot[1][d]*t.a[1][q]+t.rot[2][d]*t.a[2][q];
                buf[j][d]=std::isfinite(z) ? mass*z : 0.;
            }
        }
        if (std::fwrite(buf,sizeof(buf[0]),k,f)!=k) {
            std::fclose(f);
            return false;
        }
    }
    return std::fclose(f)==0;
}

// Orthogonal projection in the relative-error norm, using the computed
// force magnitudes as weights. An isolated system has zero total force
// and zero total torque.
void balance_field(Tree& t) {
    const auto* origin=t.nodes[0].com;
    long double totals[16]{};
    for (const auto& b:t.nodes) if (b.leaf()) {
        double q[16]{};
        for (int j=b.start;j<b.start+b.count;++j) {
            const double x=t.x[0][j]-origin[0],y=t.x[1][j]-origin[1],z=t.x[2][j]-origin[2];
            const double u=t.a[0][j],v=t.a[1][j],w=t.a[2][j],a=u*u+v*v+w*w;
            q[0]+=a;
            q[1]+=a*x; q[2]+=a*y; q[3]+=a*z;
            q[4]+=a*(y*y+z*z); q[5]+=a*(x*x+z*z); q[6]+=a*(x*x+y*y);
            q[7]-=a*x*y; q[8]-=a*x*z; q[9]-=a*y*z;
            q[10]+=u; q[11]+=v; q[12]+=w;
            q[13]+=y*w-z*v; q[14]+=z*u-x*w; q[15]+=x*v-y*u;
        }
        for (int k=0;k<16;++k) totals[k]+=q[k];
    }
    const long double weight=totals[0];
    if (!(weight>0.) || !std::isfinite(weight)) return;
    const long double x=totals[1]/weight,y=totals[2]/weight,z=totals[3]/weight;
    const long double fx=totals[10],fy=totals[11],fz=totals[12];
    const long double ixy=totals[7]+weight*x*y,ixz=totals[8]+weight*x*z,iyz=totals[9]+weight*y*z;
    const long double a[3][3]={
        {totals[4]-weight*(y*y+z*z),ixy,ixz},
        {ixy,totals[5]-weight*(x*x+z*z),iyz},
        {ixz,iyz,totals[6]-weight*(x*x+y*y)}
    };
    long double torque[3]={totals[13]-y*fz+z*fy,totals[14]-z*fx+x*fz,totals[15]-x*fy+y*fx};
    long double l[3][3]{};
    bool ok=true;
    const long double floor=(a[0][0]+a[1][1]+a[2][2])*1.e-13L;
    for (int i=0;i<3 && ok;++i) for (int j=0;j<=i;++j) {
        long double s=a[i][j];
        for (int k=0;k<j;++k) s-=l[i][k]*l[j][k];
        if (i==j) {
            if (!(s>0. && s>floor)) { ok=false; break; }
            l[i][j]=std::sqrt(s);
        } else l[i][j]=s/l[j][j];
    }
    if (ok) {
        for (int i=0;i<3;++i) {
            for (int j=0;j<i;++j) torque[i]-=l[i][j]*torque[j];
            torque[i]/=l[i][i];
        }
        for (int i=2;i>=0;--i) {
            for (int j=i+1;j<3;++j) torque[i]-=l[j][i]*torque[j];
            torque[i]/=l[i][i];
        }
    } else for (auto& w:torque) w=0.;
    const double cx=origin[0]+double(x),cy=origin[1]+double(y),cz=origin[2]+double(z);
    const double tx=fx/weight,ty=fy/weight,tz=fz/weight;
    const double wx=torque[0],wy=torque[1],wz=torque[2];
    if (!std::isfinite(tx+ty+tz+wx+wy+wz)) return;
    for (const auto& b:t.nodes) if (b.leaf()) {
        for (int j=b.start;j<b.start+b.count;++j) {
            const double x=t.x[0][j]-cx,y=t.x[1][j]-cy,z=t.x[2][j]-cz;
            const double u=t.a[0][j],v=t.a[1][j],w=t.a[2][j],a=u*u+v*v+w*w;
            t.a[0][j]=u-a*(tx+wy*z-wz*y);
            t.a[1][j]=v-a*(ty+wz*x-wx*z);
            t.a[2][j]=w-a*(tz+wx*y-wy*x);
        }
    }
}

// A bounded walk used if the budget forecast or a compute phase is
// interrupted. It still evaluates a complete softened field.
void bounded_field(Tree& t) {
    for (auto& a:t.a) std::fill(a.begin(),a.end(),0.);
    constexpr int width=40;
    using Item=std::pair<double,int>;
    std::vector<std::array<double,6>> moment(t.nodes.size());
    for (size_t i=t.nodes.size();i-- > 0;) {
        const auto& n=t.nodes[i];
        auto& q=moment[i];
        const auto add=[&](double x,double y,double z,double c) {
            q[0]+=c*x*x; q[1]+=c*y*y; q[2]+=c*z*z;
            q[3]+=c*x*y; q[4]+=c*x*z; q[5]+=c*y*z;
        };
        if (n.leaf()) {
            for (int j=n.start;j<n.start+n.count;++j)
                add(t.x[0][j]-n.com[0],t.x[1][j]-n.com[1],t.x[2][j]-n.com[2],1.);
        } else for (int j:{n.left,n.right}) {
            const auto& s=t.nodes[j];
            for (int k=0;k<6;++k) q[k]+=moment[j][k];
            add(s.com[0]-n.com[0],s.com[1]-n.com[1],s.com[2]-n.com[2],s.count);
        }
    }
    for (const auto& n:t.nodes) if (n.leaf()) {
        std::array<Item,width+1> heap;
        int nheap=0, nsrc=0, own=-1;
        double x[3][width+4],m[width+4],q[6][width+4],extent2[width+4];
        const auto accept=[&](int i) {
            const auto& s=t.nodes[i];
            if (n.begin>=s.begin && n.begin<s.begin+s.count) {
                own=i;
                return;
            }
            for (int j=0;j<3;++j) x[j][nsrc]=s.com[j];
            for (int j=0;j<6;++j) q[j][nsrc]=moment[i][j];
            extent2[nsrc]=3.0*s.r2;
            m[nsrc++]=s.count;
        };
        const auto push=[&](int j) {
            const auto& s=t.nodes[j];
            if (s.leaf()) { accept(j); return; }
            const double dx=n.c[0]-s.c[0],dy=n.c[1]-s.c[1],dz=n.c[2]-s.c[2];
            const double rr=dx*dx+dy*dy+dz*dz+eps2;
            const double score=s.r2*s.count/(rr*(rr+.15*s.r2));
            heap[nheap++]={score,j};
            std::push_heap(heap.begin(),heap.begin()+nheap);
        };
        push(0);
        for (int k=1;k<width && nheap;++k) {
            std::pop_heap(heap.begin(),heap.begin()+nheap);
            const auto& s=t.nodes[heap[--nheap].second];
            push(s.left); push(s.right);
        }
        for (int i=0;i<nheap;++i) accept(heap[i].second);
        for (int i=nsrc;i%4;++i) {
            for (int j=0;j<3;++j) x[j][i]=0.;
            for (int j=0;j<6;++j) q[j][i]=0.;
            m[i]=0.; extent2[i]=0.;
        }
        for (int j=n.start;j<n.start+n.count;++j) {
            const V xi=set(t.x[0][j]),yi=set(t.x[1][j]),zi=set(t.x[2][j]);
            V fx=zero(),fy=zero(),fz=zero();
            for (int k=0;k<nsrc;k+=4) {
                V dx=load(x[0]+k)-xi,dy=load(x[1]+k)-yi,dz=load(x[2]+k)-zi;
                const V rr=fma(dx,dx,fma(dy,dy,fma(dz,dz,set(eps2))));
                const V h=invcube<3>(rr);
                const V u=load(q[0]+k),v=load(q[1]+k),w=load(q[2]+k);
                const V xy=load(q[3]+k),xz=load(q[4]+k),yz=load(q[5]+k);
                const V ax=fma(u,dx,fma(xy,dy,xz*dz));
                const V ay=fma(xy,dx,fma(v,dy,yz*dz));
                const V az=fma(xz,dx,fma(yz,dy,w*dz));
                const V dot=fma(ax,dx,fma(ay,dy,az*dz));
                const V inv=_mm256_cvtps_pd(_mm_rcp_ps(_mm256_cvtpd_ps(rr)));
                const V damp=_mm256_min_pd(set(1.),rr/_mm256_max_pd(rr,load(extent2+k)));
                const V h5=h*inv*damp*damp;
                const V f=load(m+k)*h+h5*fma(set(7.5)*inv,dot,set(-1.5)*(u+v+w));
                const V zc=set(-3.)*h5;
                fx=fma(dx,f,fma(ax,zc,fx)); fy=fma(dy,f,fma(ay,zc,fy)); fz=fma(dz,f,fma(az,zc,fz));
            }
            t.a[0][j]=sum(fx); t.a[1][j]=sum(fy); t.a[2][j]=sum(fz);
            if (own>=0 && !t.nodes[own].leaf()) {
                const auto& s=t.nodes[own];
                const double alpha=double(s.count)/(s.count-1);
                const double dx=alpha*(s.com[0]-t.x[0][j]),dy=alpha*(s.com[1]-t.x[1][j]),
                             dz=alpha*(s.com[2]-t.x[2][j]);
                const double rr=dx*dx+dy*dy+dz*dz+eps2;
                const double f=(s.count-1)/(rr*std::sqrt(rr));
                t.a[0][j]+=dx*f; t.a[1][j]+=dy*f; t.a[2][j]+=dz*f;
            }
        }
        if (own>=0 && t.nodes[own].leaf()) direct_self(t,n);
    }
}

struct V3 { V x,y,z; };

struct Refiner {
    static constexpr int order_max=14;
    struct Group {
        std::array<int,4> at;
        int count;
        float priority;
        size_t begin=0,end=0;
    };
    struct Task { int src,order; };
    Tree& t;
    std::vector<Group> targets;
    std::vector<Task> tasks;
    double q[wbins][order_max+1]{};
    double allowance, work=0;
    size_t cur=0,visits=0;
    size_t replaced=0;
    int maxorder=0;

    Refiner(Tree& t_,double tolerance,double until) : t(t_),allowance(.65*(until-cpu())) {
        for (int i=0;i<wbins;++i) {
            const double w=std::ldexp(1.+(i%8+.5)/8.,i/8-12);
            const double tol=tolerance*std::clamp(.90/std::sqrt(w),.50,12.);
            for (int p=4;p<=order_max;p+=2) q[i][p]=std::pow(tol,2./(p+1));
        }
    }

    struct Danger {
        const Node* n=nullptr;
        double power=0.,prior=0.;
    };
    double risk_at(const Danger& a,double r2) const {
        const auto& n=*a.n;
        const double rel=std::min(r2/n.r2,1.0);
        return n.risk[1]*integer_power(rel,int(a.power));
    }
    double measures(std::vector<float>& estimates,double fraction) const {
        const size_t n=t.index.size(),m=std::min<size_t>(8192,n);
        estimates.resize(t.x[0].size());
        for (const auto& leaf:t.nodes) if (leaf.leaf()) {
            Danger a{},b{};
            const Node* v=&leaf;
            for (;;) {
                if (v->risk[1]>0. && v->r2>0.) {
                    const double x=leaf.c[0]-v->c[0],y=leaf.c[1]-v->c[1],z=leaf.c[2]-v->c[2];
                    const double rb=std::sqrt(x*x+y*y+z*z)+leaf.r;
                    const double step=std::clamp(std::log(v->risk[1]/std::max(1.e-300,v->risk[0]))*1.12,1.,24.);
                    Danger candidate{v,step,0.};
                    candidate.prior=risk_at(candidate,rb*rb);
                    if (candidate.prior>a.prior) { b=a; a=candidate; }
                    else if (candidate.prior>b.prior) b=candidate;
                }
                if (v->parent<0) break;
                v=&t.nodes[v->parent];
            }
            const Danger candidates[2]={a,b};
            for (int j=leaf.start;j<leaf.start+leaf.count;++j) {
                double s=0;
                for (const Danger& z:candidates) if (z.n) {
                    const double x=t.x[0][j]-z.n->c[0],y=t.x[1][j]-z.n->c[1],w=t.x[2][j]-z.n->c[2];
                    s+=risk_at(z,x*x+y*y+w*w);
                }
                const double x=t.a[0][j],y=t.a[1][j],z=t.a[2][j];
                estimates[j]=std::sqrt(s/std::max(1.e-100,x*x+y*y+z*z));
            }
        }
        std::vector<float> sampled(m);
        for (size_t i=0;i<m;++i) sampled[i]=estimates[t.index[i*n/m]];
        auto index=sampled.begin()+std::min(m-1,size_t((1.-fraction)*m));
        std::nth_element(sampled.begin(),index,sampled.end());
        return std::max(double(*index),1.e-100);
    }

    void select(double fraction,double risk_fraction) {
        double c[3];
        const size_t n=t.index.size(), m=std::min<size_t>(4096,n);
        std::vector<double> r(m);
        for (int d=0;d<3;++d) {
            for (size_t i=0;i<m;++i) r[i]=t.x[d][t.index[i*n/m]];
            std::nth_element(r.begin(),r.begin()+m/2,r.end());
            c[d]=r[m/2];
        }
        const auto radius=[&](int j) {
            const double x=t.x[0][j]-c[0],y=t.x[1][j]-c[1],z=t.x[2][j]-c[2];
            return x*x+y*y+z*z;
        };
        for (size_t i=0;i<m;++i) r[i]=radius(t.index[i*n/m]);
        std::sort(r.begin(),r.end());
        double limit=1e300;
        const double ratio=r[size_t(.95*m)]/std::max(1.e-100,r[m/2]);
        if (!t.rotated && ratio>5.7) {
            const double frac=fraction*(ratio<11. ? .58 : (ratio<24. ? .82 : 1.));
            limit=r[std::min(m-1,size_t((1.-frac)*m))];
        }
        std::vector<float> estimates;
        const double risk_bound=risk_fraction>0. ? measures(estimates,risk_fraction) : 1.;
        for (const auto& n:t.nodes) if (n.leaf()) {
            Group g{};
            for (int j=n.start;j<n.start+n.count;++j) {
                const double rr=radius(j);
                const double x=t.a[0][j],y=t.a[1][j],z=t.a[2][j];
                const double weak=(x*x+y*y+z*z)*(n.g*n.g);
                const double risk=risk_fraction>0. ? estimates[j]/risk_bound : 0.;
                if (rr>=limit || risk>=1. || (n.g!=0. && weak<.016)) {
                    g.at[g.count++]=j;
                    g.priority=std::max(g.priority,float(std::max({rr/limit,1.25*risk,.016/std::max(weak,1.e-20)})));
                    if (g.count==4) {
                        targets.push_back(g);
                        g=Group{};
                    }
                }
            }
            if (g.count) {
                for (int k=g.count;k<4;++k) g.at[k]=g.at[k-1];
                targets.push_back(g);
            }
        }
        std::sort(targets.begin(),targets.end(),[](const Group& a,const Group& b) { return a.priority>b.priority; });
    }

    V3 point(const Group& g) const {
        const auto at=[&](int d,int j) { return t.x[d][g.at[j]]; };
        return {_mm256_setr_pd(at(0,0),at(0,1),at(0,2),at(0,3)),
                _mm256_setr_pd(at(1,0),at(1,1),at(1,2),at(1,3)),
                _mm256_setr_pd(at(2,0),at(2,1),at(2,2),at(2,3))};
    }

    void walk(int i,const V3& pos,double invforce) {
        if ((visits++ & 4095)==0) checkclock();
        auto& n=t.nodes[i];
        if (n.leaf()) {
            tasks.push_back({i,0});
            work+=6.8e-9*n.count;
            return;
        }
        const V x=set(n.c[0])-pos.x,y=set(n.c[1])-pos.y,z=set(n.c[2])-pos.z;
        const V vv=fma(x,x,fma(y,y,fma(z,z,set(eps2))));
        const __m128d m=_mm_min_pd(_mm256_castpd256_pd128(vv),_mm256_extractf128_pd(vv,1));
        const double r=_mm_cvtsd_f64(_mm_min_sd(m,_mm_unpackhi_pd(m,m)));
        const double s=n.rsoft*n.rsoft;
        const double w0=n.count*invforce/r;
        uint64_t bits;
        std::memcpy(&bits,&w0,sizeof(bits));
        const double* angles=q[std::clamp(int(bits>>49)-1023*8+96,0,wbins-1)];
        if (s<=angles[order_max]*r && n.r2<=.40*r) {
            int p=4;
            while (s>angles[p]*r) p+=2;
            if (n.count >= (terms(p+1)*.68 + 16.)) {
                n.pm=std::max<int>(n.pm,p);
                maxorder=std::max(maxorder,p);
                tasks.push_back({i,p});
                work+=4.0e-9*terms(p+1);
                return;
            }
        }
        walk(n.left,pos,invforce);
        walk(n.right,pos,invforce);
    }

    void make_plan() {
        for (auto& n:t.nodes) { n.pm=n.pl=0; n.shrink=1.; }
        while (cur<targets.size() && work<allowance && tasks.size()<10000000U) {
            auto& g=targets[cur++];
            g.begin=tasks.size();
            double ff=1.e300;
            for (int i=0;i<g.count;++i) {
                const int j=g.at[i];
                const double x=t.a[0][j],y=t.a[1][j],z=t.a[2][j];
                ff=std::min(ff,x*x+y*y+z*z);
            }
            walk(0,point(g),1./std::sqrt(std::max(ff,1.e-100)));
            g.end=tasks.size();
        }
    }

    inline void direct(const Node& n,const V3& pos,V3& ans) const {
        const auto x=t.x[0].data()+n.start,y=t.x[1].data()+n.start,z=t.x[2].data()+n.start;
        for (int j=0;j<n.count;++j) {
            const V dx=set(x[j])-pos.x,dy=set(y[j])-pos.y,dz=set(z[j])-pos.z;
            const V f=force(dx,dy,dz);
            ans.x=fma(dx,f,ans.x);
            ans.y=fma(dy,f,ans.y);
            ans.z=fma(dz,f,ans.z);
        }
    }

    void evaluate(const Node& n,int p,const V3& pos,V3& ans,const Fmm& f) const {
        constexpr int capacity=terms(order_max+1)+1;
        V der[capacity];
        const V x=set(n.c[0])-pos.x,y=set(n.c[1])-pos.y,z=set(n.c[2])-pos.z;
        const V inv=set(1.)/_mm256_sqrt_pd(fma(x,x,fma(y,y,fma(z,z,set(eps2)))));
        f.bs.derivatives(der,p+1,x,y,z,inv);
        const double* m=f.moments.data()+n.mo;
        for (int i=0;i<terms(p);++i) {
            const auto a=f.bs.xyz[i];
            const V k=set(-m[i]);
            ans.x=fma(der[1+idx(a.x+1,a.y,a.z)],k,ans.x);
            ans.y=fma(der[1+idx(a.x,a.y+1,a.z)],k,ans.y);
            ans.z=fma(der[1+idx(a.x,a.y,a.z+1)],k,ans.z);
        }
    }

    void run() {
        if (!cur) return;
        Fmm f(t,maxorder+1,0,false);
        f.setup();
        f.upward();
        size_t tick=0;
        for (size_t i=0;i<cur;++i) {
            const auto& g=targets[i];
            const auto p=point(g);
            V3 a={zero(),zero(),zero()};
            for (size_t j=g.begin;j<g.end;++j) {
                if ((tick++&511)==0) checkclock();
                const auto job=tasks[j];
                const auto& n=t.nodes[job.src];
                if (job.order) evaluate(n,job.order,p,a,f);
                else direct(n,p,a);
            }
            double z[3][4];
            store(z[0],a.x); store(z[1],a.y); store(z[2],a.z);
            for (int k=0;k<g.count;++k) {
                if (!std::isfinite(z[0][k]) || !std::isfinite(z[1][k]) || !std::isfinite(z[2][k])) continue;
                for (int d=0;d<3;++d) t.a[d][g.at[k]]=z[d][k];
                ++replaced;
            }
        }
    }
    void exact_targets() {
        for (const auto& g:targets) {
            const auto p=point(g);
            V3 a={zero(),zero(),zero()};
            size_t n=0;
            for (const auto& box:t.nodes) if (box.leaf()) {
                if ((n++&127)==0) checkclock();
                direct(box,p,a);
            }
            double z[3][4];
            store(z[0],a.x); store(z[1],a.y); store(z[2],a.z);
            for (int k=0;k<g.count;++k) {
                for (int d=0;d<3;++d) t.a[d][g.at[k]]=z[d][k];
                ++replaced;
            }
            ++cur;
        }
    }
};

Work choose_plan(Plan& plan,double budget,double hard_finish,double near_rate,bool manual,int debug) {
    const size_t n=plan.t.index.size();
    const size_t maxcoef=max_coefficient_bytes/sizeof(double);
    if (manual) {
        plan.finish=hard_finish;
        plan.walk(0,0);
        if (plan.abort) throw Timeout{};
        auto w=forecast(plan,near_rate);
        if (w.coeffs>maxcoef) throw Timeout{};
        return w;
    }
    double lower=.75, upper=16., best=-1.;
    double e=std::clamp(6.0 + .77*std::log2(std::max(.020,budget-cpu())/std::max(1.e-9, n*16.e-6)),2.,9.);
    const double search_end=std::min(cpu()+.125*budget,.37*budget);
    double last_e=-1.,last_w=0.,slope=.36;
    plan.record=false;
    for (int i=0;i<5;++i) {
        const double started=cpu();
        if (i && (started>search_end || (upper-lower)<.14)) break;
        plan.clear(std::pow(10.,-e),std::min(hard_finish,started+.112*budget));
        plan.walk(0,0);
        Work w;
        if (!plan.abort) w=forecast(plan,near_rate);
        const double planning=1.14*(cpu()-started)+.0015;
        const double available=budget-cpu()-planning;
        const bool ok=!plan.abort && w.coeffs<=maxcoef &&
                      w.seconds()<=available;
        if (debug>1)
            std::fprintf(stderr,"plan exponent %.2f pred %.4f t %.4f fits %d coeff %.1f MiB pairs %zu\n",
                         e,w.seconds(),cpu(),ok,w.coeffs/131072.,plan.pairs);
        if (ok) {
            best=std::max(best,e);
            lower=e;
        } else upper=e;
        double next=.5*(lower+upper);
        if (!plan.abort && w.seconds()>.0 && w.coeffs<=maxcoef && available>0) {
            if (last_w>0. && std::abs(e-last_e)>.17) {
                const double measured=std::log(w.seconds()/last_w)/(e-last_e);
                if (measured>.12 && measured<.9) slope=std::clamp(measured,.20,.70);
            }
            last_e=e;
            last_w=w.seconds();
            const double delta=std::log(.988*available/w.seconds())/slope;
            next=e+delta;
            if (ok && delta<.10) break;
            if (!ok && delta>-.06 && best<0.) next=e-.10;
            if (next<lower || next>upper) next=.5*(lower+upper);
        }
        e=std::clamp(next,lower+.025,upper-.025);
    }
    plan.record=true;
    if (best<0.) best=lower;
    double retained=-1.;
    for (int i=0;i<4;++i) {
        const double started=cpu();
        plan.clear(std::pow(10.,-best),hard_finish);
        plan.walk(0,0);
        if (!plan.abort) {
            auto w=forecast(plan,near_rate);
            const double now=cpu();
            const bool memok=w.coeffs<=maxcoef;
            if (memok && now+w.seconds()<=budget*1.018) {
                const double available=budget-now-1.25*(now-started)-.001;
                if (i==0 && now+w.seconds()<.91*budget && best<15. && w.seconds()>0. &&
                    w.coeffs<.90*maxcoef && plan.pairs<.78*plan.maxpairs) {
                    const double gain=std::clamp(.73*std::log(std::max(available/w.seconds(),1.))/slope,0.,1.10);
                    if (gain>.21) {
                        retained=best;
                        best+=gain;
                        plan.release();
                        continue;
                    }
                }
                return w;
            }
            if (debug>1)
                std::fprintf(stderr,"recorded exponent %.2f predict %.4f now %.4f fits 0 coeff %.1f MiB\n",
                             best,w.seconds(),now,w.coeffs/131072.);
            if (memok && i>=2 && now+w.seconds()<=budget*1.04) return w;
            const double available=std::max(.02,budget-now-1.17*(now-started));
            const double adjust=memok ? std::clamp(std::log(w.seconds()/available)/slope+.09,.13,1.20) : .72;
            best=retained>0. ? std::min(retained,best-adjust) : best-adjust;
        } else {
            best-=i==0 ? .65 : 1.20;
        }
        plan.release();
        retained=-1.;
        best=std::max(.1,best);
        if (cpu()>hard_finish) break;
    }
    throw Timeout{};
}

} // namespace

int main(int argc, char** argv) {
    if (argc!=3) return 2;
    try {
        Tree t;
        double mass=0.;
        if (!read_input(argv[1],t,mass)) return 2;
        const double t0=cpu();
        const size_t n=t.bodies.size();
        const double suggested=n<=65536 ? 1.4 : n<=262144 ? 6. : n<=1048576 ? 28. : 120.;
        const double seconds=optdouble("GALAXY_CPU_LIMIT_S",suggested);
        const bool manual=std::getenv("GAL_TOL")!=nullptr;
        const int debug=optint("GAL_DEBUG",0);
        const int pmax0=std::clamp(optint("GAL_PMAX",26),2,maxp);
        const int pmin0=std::clamp(optint("GAL_PMIN",4),2,pmax0);
        const double tol=optdouble("GAL_TOL",1e-7);
        t.split_bias=optdouble("GAL_SPLIT",0.);
        t.centering=optdouble("GAL_CENTER",.145);
        t.decay=optdouble("GAL_DECAY",0.);
        if (n>=1024 && optint("GAL_ROT",1)) t.orient();
        t.find_outer(optint("GAL_TAIL_SPLIT",0)!=0);
        int preferred=n<=65536 ? (t.rotated || t.cusp ? 82 : 64) :
                      n<=262144 ? (t.rotated || t.cusp ? 88 : 72) :
                      n<=1048576 ? (t.rotated ? 104 : 96) : (t.rotated ? 144 : 128);
        if (n>1048576 && seconds>145) preferred+=16;
        const int block=std::clamp(optint("GAL_BLOCK",preferred),8,256);
        t.setup(block);
        const double grav=optdouble("GAL_GRAV",.22);
        if (grav>0.) t.scales(optdouble("GAL_WEAK",0.));
        const double t1=cpu();
        const int rawmax=std::clamp(optint("GAL_RAW",17),pmin0,pmax0);
        const double pf=optdouble("GAL_PF",.136);
        const double near_rate=optdouble("GAL_NEAR_RATE",0.);
        const double risk_fraction=optdouble("GAL_RISK",t.rotated ? .014 : .032);
        double final_tol=0.;
        try {
            Plan plan(t,pmin0,pmax0,rawmax,tol,optdouble("GAL_RATIO",.74),pf,
                      optdouble("GAL_RFAC",1.),optdouble("GAL_TRACE",t.cusp ? 16. : 12.),grav,optdouble("GAL_OV",0.),
                      optdouble("GAL_WPOWER",.62),optdouble("GAL_LOOK",0.),
                      optint("GAL_TRIM",1),risk_fraction>0. && !manual,
                      std::min<size_t>(12000000,n*7+100000));
            const double reserve=std::max(.070, .072*seconds) + 20.e-9*n;
            compute_until=manual && !std::getenv("GALAXY_CPU_LIMIT_S") ? 1.e30 : seconds-reserve;
            const double budget=optdouble("GAL_USE",t.rotated ? .895 : .872)*(seconds-12.e-9*n);
            const Work estimate=choose_plan(plan,budget,compute_until-.06*seconds,near_rate,manual,debug);
            const double t2=cpu();
            double up=0.,far=0.,down=0.;
            size_t tick=0;
            if (plan.trim) {
                std::vector<Pair> pair_buffer;
                for (int p=pmin0;p<=pmax0;++p) {
                    stable_group(plan.far[p],pair_buffer);
                    for (int c=0;c<p;++c) stable_group(plan.harm[p][c],pair_buffer);
                }
            }
            const auto translate=[&](const std::vector<Pair>& v,auto&& kernel) {
                for (size_t i=0;i<v.size();) {
                    size_t j=i+1;
                    while (j<v.size() && v[i].shape()==v[j].shape()) ++j;
                    for (;i<j;i+=4) {
                        if ((tick++&127)==0) checkclock();
                        kernel(v.data()+i,int(std::min<size_t>(j-i,4)));
                    }
                    i=j;
                }
            };
            if (plan.usedraw>1) {
                double now=cpu();
                Fmm fmm(t,plan.usedraw,plan.usedharm ? plan.usedraw : 0);
                fmm.setup();
                fmm.upward();
                up+=cpu()-now;
                now=cpu();
                for (int p=pmin0;p<=pmax0;++p) {
                    translate(plan.far[p],[&](const Pair* ptr,int n) { fmm.m2l(ptr,n,p); });
                    for (int c=2;c<p;++c)
                        translate(plan.harm[p][c],[&](const Pair* ptr,int n) { fmm.m2l<true>(ptr,n,c,p); });
                }
                far+=cpu()-now;
                now=cpu();
                fmm.downward();
                down+=cpu()-now;
            }
            if (plan.usedharm) {
                double now=cpu();
                Harmonic h(t,plan.usedharm);
                h.setup(0);
                h.upward();
                up+=cpu()-now;
                now=cpu();
                for (int p=pmin0;p<=pmax0;++p) for (int c=0;c<p;++c)
                    translate(plan.harm[p][c],[&](const Pair* ptr,int n) { h.m2l(ptr,n,p); });
                far+=cpu()-now;
                now=cpu();
                h.downward();
                down+=cpu()-now;
            }
            const double t4=cpu();
            const int direct=optint("GAL_DIRECT",0);
            const int prec=direct_precision(plan.tol);
            for (auto ab:plan.near) {
                if ((tick++&63)==0) checkclock();
                if (ab.a==ab.b) direct_self(t,t.nodes[ab.first()]);
                else {
                    const auto& a=t.nodes[ab.first()]; const auto& b=t.nodes[ab.second()];
                    if (!direct) {
                        if (prec==2) direct_one<2>(t,a,b);
                        else if (prec==3) direct_one<3>(t,a,b);
                        else direct_one<4>(t,a,b);
                    } else if (direct==1) direct_pair(t,a,b);
                    else if (direct==2) direct_pair2(t,a,b);
                    else if (direct==3) direct_one<4>(t,a,b);
                    else if (direct==4) direct_two<4>(t,a,b);
                    else if (direct==5) direct_one<3>(t,a,b);
                    else if (direct==6) direct_two<3>(t,a,b);
                    else if (direct==7) direct_four<3>(t,a,b);
                    else if (direct==8) direct_one<2>(t,a,b);
                    else direct_four<2>(t,a,b);
                }
            }
            const double t6=cpu();
            final_tol=plan.tol;
            if (debug) {
                std::fprintf(stderr,"cpu %.4f [in %.4f tree %.4f plan %.4f up %.4f far %.4f near %.4f down %.4f] nodes %zu pp %llu/%zu ml %.0f h %.0f visits %llu buf %.1f MiB\n",
                    t6,t0,t1-t0,t2-t1,up,far,t6-t4,down,t.nodes.size(),
                    (unsigned long long)plan.pp,plan.near.size(),plan.mlwork,plan.hw,(unsigned long long)plan.visits,
                    estimate.coeffs/131072.);
                std::fprintf(stderr,"tol %.4g rot %d predicted up %.3f far %.3f near %.3f down %.3f\n",
                    plan.tol,t.rotated,estimate.up,estimate.far,estimate.near,estimate.down);
            }
            if (debug>2) {
                for (int p=pmin0;p<=pmax0;++p) std::fprintf(stderr," %d:%zu",p,plan.far[p].size());
                std::fputc('\n',stderr);
                for (int p=pmin0;p<=pmax0;++p) {
                    size_t ct=0;
                    for (int c=0;c<p;++c) ct+=plan.harm[p][c].size();
                    if (ct) std::fprintf(stderr," %d:%zu",p,ct);
                }
                std::fputc('\n',stderr);
            }
        } catch (const Timeout&) {
            if (debug) std::fprintf(stderr,"bounded field at %.4f CPU s\n",cpu());
            bounded_field(t);
        } catch (const std::bad_alloc&) {
            if (debug) std::fprintf(stderr,"bounded field after allocation failure\n");
            bounded_field(t);
        }
        if (!manual && final_tol>0. && n>=1024 && optint("GAL_REFINE",1)) {
            const double start=cpu();
            compute_until=seconds-std::max(.034,.022*seconds)-102.e-9*n;
            if (compute_until-start>std::max(.047,.025*seconds)) {
                Refiner refine(t,std::clamp(final_tol*optdouble("GAL_TAIL_TOL",.02),2.e-14,1.e-6),compute_until);
                try {
                    refine.select(optdouble("GAL_TAIL_FRAC",.064),risk_fraction);
                    if (refine.targets.size()<=std::max<size_t>(8,n/100000) &&
                        refine.targets.size()*n*8.e-9 < .55*(compute_until-cpu())) refine.exact_targets();
                    else {
                        refine.make_plan();
                        refine.run();
                    }
                } catch (const Timeout&) {
                } catch (const std::bad_alloc&) {
                }
                if (debug) std::fprintf(stderr,"refined %zu (%zu groups of %zu, %zu tasks) in %.4f CPU s\n",
                    refine.replaced,refine.cur,refine.targets.size(),refine.tasks.size(),cpu()-start);
            }
        }
        if (final_tol>0. && n>=1024 && cpu()<seconds-.018-102.e-9*n)
            balance_field(t);
        const bool ok=write_output(argv[2],t,mass);
        if (debug) {
            rusage u{};
            getrusage(RUSAGE_SELF,&u);
            std::fprintf(stderr,"finished %.5f CPU s, peak %.1f MiB\n",cpu(),u.ru_maxrss/1024.);
        }
        return ok ? 0 : 2;
    } catch (const std::exception&) {
        return 3;
    }
}
