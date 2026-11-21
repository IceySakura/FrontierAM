#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <immintrin.h>
#include <limits>
#include <malloc.h>
#include <numeric>
#include <sys/stat.h>
#include <vector>

static double cpu_time() {
    timespec t;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

struct V {
    __m256d v;
    V() = default;
    V(double x) : v(_mm256_set1_pd(x)) {}
    V(__m256d x) : v(x) {}
    V(double a,double b,double c,double d) : v(_mm256_setr_pd(a,b,c,d)) {}
    friend V operator+(V a,V b) { return _mm256_add_pd(a.v,b.v); }
    friend V operator-(V a,V b) { return _mm256_sub_pd(a.v,b.v); }
    friend V operator*(V a,V b) { return _mm256_mul_pd(a.v,b.v); }
    friend V operator/(V a,V b) { return _mm256_div_pd(a.v,b.v); }
    friend V operator-(V a) { return _mm256_sub_pd(_mm256_setzero_pd(),a.v); }
    V& operator+=(V a) { v=_mm256_add_pd(v,a.v); return *this; }
    void store(double* p) const { _mm256_storeu_pd(p,v); }
    static V load(const double* p) { return _mm256_loadu_pd(p); }
};
static V vsqrt(V x) { return _mm256_sqrt_pd(x.v); }
static V vinvsqrt(V x) {
    V y(_mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(x.v))));
    y=y*(V(1.5)-V(.5)*x*y*y);
    y=y*(V(1.5)-V(.5)*x*y*y);
    return y;
}
template<int P> struct Kernel;
#include "kernels.hpp"
#include "harmonic.hpp"

struct Particle { double m,x,y,z; int id; };
static double coordinate(const Particle& p,int axis) {
    return axis==0?p.x:axis==1?p.y:p.z;
}
static double frame[3][3]={{1,0,0},{0,1,0},{0,0,1}};

static void principal_frame(std::vector<Particle>& data,bool flat_only) {
    double center[3]={};
    for(const auto& p:data) { center[0]+=p.x; center[1]+=p.y; center[2]+=p.z; }
    for(int j=0;j<3;++j) center[j]/=data.size();
    double c[3][3]={},v[3][3]={{1,0,0},{0,1,0},{0,0,1}};
    for(const auto& p:data) {
        double d[3]={p.x-center[0],p.y-center[1],p.z-center[2]};
        for(int j=0;j<3;++j) for(int k=j;k<3;++k) c[j][k]+=d[j]*d[k];
    }
    for(int j=0;j<3;++j) for(int k=0;k<j;++k) c[j][k]=c[k][j];
    for(int iteration=0;iteration<24;++iteration) {
        int p=0,q=1;
        if(std::abs(c[0][2])>std::abs(c[p][q])) { p=0; q=2; }
        if(std::abs(c[1][2])>std::abs(c[p][q])) { p=1; q=2; }
        if(std::abs(c[p][q])<=1e-16*(std::abs(c[0][0])+std::abs(c[1][1])+std::abs(c[2][2]))) break;
        double tau=(c[q][q]-c[p][p])/(2*c[p][q]);
        double t=std::copysign(1.,tau)/(std::abs(tau)+std::sqrt(1+tau*tau));
        double cs=1/std::sqrt(1+t*t),sn=t*cs,cpq=c[p][q];
        c[p][p]-=t*cpq; c[q][q]+=t*cpq; c[p][q]=c[q][p]=0.;
        for(int k=0;k<3;++k) {
            if(k!=p&&k!=q) {
                double a=c[k][p],b=c[k][q];
                c[k][p]=c[p][k]=cs*a-sn*b;
                c[k][q]=c[q][k]=sn*a+cs*b;
            }
            double a=v[k][p],b=v[k][q];
            v[k][p]=cs*a-sn*b; v[k][q]=sn*a+cs*b;
        }
    }
    int order[3]={0,1,2};
    std::sort(order,order+3,[&](int a,int b) {return c[a][a]>c[b][b];});
    if(flat_only&&c[order[2]][order[2]]>.15*c[order[0]][order[0]]) return;
    for(int j=0;j<3;++j) for(int k=0;k<3;++k) frame[j][k]=v[j][order[k]];
    for(auto& p:data) {
        double a=p.x-center[0],b=p.y-center[1],c=p.z-center[2];
        p.x=frame[0][0]*a+frame[1][0]*b+frame[2][0]*c;
        p.y=frame[0][1]*a+frame[1][1]*b+frame[2][1]*c;
        p.z=frame[0][2]*a+frame[1][2]*b+frame[2][2]*c;
    }
}

struct Node {
    double x,y,z,r,re;
    int begin,end,left=-1,right=-1;
    bool leaf() const { return left<0; }
};
struct Pair { int a,b; };
struct FarKernel {
    using Derivative=void(*)(V,V,V,V,V*);
    using Translation=void(*)(const V*,const V*,const V*,V*,V*);
    int nc=0,nm=0,hc=0,hm=0,q=0,h=0,work=0;
    Derivative derivative=nullptr,hderivative=nullptr;
    Translation translate=nullptr,htranslate=nullptr;
    void(*to_cart)(const V*,V*)=nullptr;
};

template<int P,int H=0> class Solver {
    using K=Kernel<P>;
    using HK=Harmonic<H>;
    static constexpr int NC=K::NC, NM=K::NM;
    int n, leafsize;
    double mass, theta, tolerance=0.,gamma=1.,shape=0.,softfactor=1.,deltafactor=.03,softboost=1.;
    double cx=0,cy=0,cz=0,scale=0;
    double budget=0;
    bool automatic=false;
    bool fastshift=false;
    bool vectorize=false;
    bool moment_radius=false;
    bool stf=true;
    double direct_factor=.25;
    bool damp_trace=true;
    bool block_profiles=false;
    bool refine=false;
    double mac_power=0.;
    int variable=2;
    double varscale=1.5,var_delta=1.,hmacthreshold[3]={},cmacthreshold[3]={};
    bool midpoint;
    std::vector<Particle> particles;
    std::vector<Node> nodes;
    std::vector<double> moments,locals,accel,hmoments,hlocals;
    std::vector<double> error_bound;
    std::vector<double> x,y,z,ax,ay,az;
    std::vector<int> ids;
    std::array<std::vector<Pair>,10> far;
    std::array<FarKernel,10> kernels;
    std::vector<Pair> near;
    int index[P+1][P+1][P+1];
    double factorial[P+1];
    unsigned long long direct_pairs=0;
    size_t pair_capacity=0,fixed_bytes=0;
    bool list_overflow=false;
    double memory_theta=.20;
    size_t merge_limit=128;

    template<int Q,int J> static FarKernel make_kernel() {
        using C=Kernel<Q>;
        FarKernel f;
        f.nc=C::NC; f.nm=C::NM; f.q=Q; f.h=J;
        f.work=C::TERMS;
        f.derivative=&C::derivatives; f.translate=&C::translate;
        if constexpr(J>Q) {
            using L=Harmonic<J>;
            f.hc=L::NC; f.hm=L::NM;
            f.work+=L::TERMS;
            f.hderivative=&L::derivatives; f.htranslate=&L::translate;
            f.to_cart=&L::template cartesian<Q>;
        }
        return f;
    }

    template<int J> void set_family(int base) {
        kernels[base]=make_kernel<P-4,J>();
        kernels[base+1]=make_kernel<P-2,J>();
        kernels[base+2]=make_kernel<P,J>();
    }

    void update_mac() {
        mac_power=std::pow(theta,H);
        for(int j=0;j<3;++j) {
            double a=varscale*std::pow(theta/varscale,double(H)/(H-4+2*j));
            hmacthreshold[j]=a*a;
            cmacthreshold[j]=mac_power*std::pow(varscale,P-4+2*j-H)*deltafactor*var_delta;
        }
    }

    size_t far_count() const {
        size_t result=0;
        for(const auto& list:far) result+=list.size();
        return result;
    }

    double far_cost(const std::array<double,10>& rate) const {
        double result=0;
        for(int i=0;i<10;++i) result+=far[i].size()*rate[i];
        return result;
    }

    bool append_pair(std::vector<Pair>& list,Pair pair) {
        if(list.size()==list.capacity()) {
            size_t old=list.capacity()*sizeof(Pair);
            size_t capacity=std::max(size_t(256),list.capacity()*2);
            size_t total=pair_capacity-old+capacity*sizeof(Pair);
            if(fixed_bytes+total>size_t(1650)*1024*1024||
               fixed_bytes+total+old>size_t(1800)*1024*1024) {
                list_overflow=true;
                return false;
            }
            list.reserve(capacity);
            pair_capacity=total;
        }
        list.push_back(pair);
        return true;
    }

    void record_error(int ai,int bi,const FarKernel& f) {
        if(!refine) return;
        const auto& a=nodes[ai]; const auto& b=nodes[bi];
        double dx=a.x-b.x,dy=a.y-b.y,dz=a.z-b.z;
        double r2=dx*dx+dy*dy+dz*dz+.0001;
        double q2=(a.r+b.r)*(a.r+b.r)/r2;
        double powers[9]; powers[0]=1.;
        for(int i=1;i<9;++i) powers[i]=powers[i-1]*q2;
        double tail=(f.q+1.)*powers[f.q/2];
        if(f.hc) tail=(f.h+1.)*powers[f.h/2]+
            .5*(f.q+1.)*(f.q+2.)*.0001/r2*powers[f.q/2];
        double gap=std::max(.05,1.-std::sqrt(q2));
        double scale=tail/(r2*gap*gap)*mass;
        error_bound[ai]+=(b.end-b.begin)*scale;
        error_bound[bi]+=(a.end-a.begin)*scale;
    }

    void make_lists() {
        for(;;) {
            for(auto& list:far) list.clear();
            near.clear(); direct_pairs=0; list_overflow=false;
            std::fill(error_bound.begin(),error_bound.end(),0.);
            update_mac();
            walk(0,0);
            if(!list_overflow) return;
            for(auto& list:far) std::vector<Pair>().swap(list);
            std::vector<Pair>().swap(near);
            pair_capacity=0;
            if(theta>=.98) merge_limit*=4;
            else theta=std::min(.98,theta*1.12+.01);
            memory_theta=theta;
        }
    }

    int build(int begin,int end) {
        int k=nodes.size();
        nodes.emplace_back();
        Node q;
        q.begin=begin; q.end=end;
        double lo[3]={1e300,1e300,1e300}, hi[3]={-1e300,-1e300,-1e300};
        double sx=0,sy=0,sz=0;
        for(int i=begin;i<end;++i) {
            const auto& a=particles[i];
            sx+=a.x; sy+=a.y; sz+=a.z;
            lo[0]=std::min(lo[0],a.x); hi[0]=std::max(hi[0],a.x);
            lo[1]=std::min(lo[1],a.y); hi[1]=std::max(hi[1],a.y);
            lo[2]=std::min(lo[2],a.z); hi[2]=std::max(hi[2],a.z);
        }
        double inv=1./(end-begin);
        q.x=sx*inv; q.y=sy*inv; q.z=sz*inv;
        double r2=0;
        for(int i=begin;i<end;++i) {
            double dx=particles[i].x-q.x,dy=particles[i].y-q.y,dz=particles[i].z-q.z;
            r2=std::max(r2,dx*dx+dy*dy+dz*dz);
        }
        q.r=std::sqrt(r2);
        q.re=q.r;
        if(moment_radius&&r2>0) {
            double s=0;
            for(int i=begin;i<end;++i) {
                double dx=particles[i].x-q.x,dy=particles[i].y-q.y,dz=particles[i].z-q.z;
                double t=(dx*dx+dy*dy+dz*dz)/r2,u=1.;
                for(int power=H;power;power>>=1) {
                    if(power&1) u*=t;
                    t*=t;
                }
                s+=u;
            }
            q.re=q.r*std::pow(s/(end-begin),1./(2*H));
        }
        double cdist=std::sqrt((q.x-cx)*(q.x-cx)+(q.y-cy)*(q.y-cy)+(q.z-cz)*(q.z-cz));
        if(end-begin>leafsize||(shape>0&&end-begin>4&&q.r>shape*(cdist+.2*scale))) {
            int axis=0;
            if(hi[1]-lo[1]>hi[axis]-lo[axis]) axis=1;
            if(hi[2]-lo[2]>hi[axis]-lo[axis]) axis=2;
            int mid=(begin+end)/2;
            if(midpoint) {
                double split=(lo[axis]+hi[axis])*.5;
                mid=std::partition(particles.begin()+begin,particles.begin()+end,
                    [axis,split](const Particle& a) { return coordinate(a,axis)<split; })-particles.begin();
            }
            if(!midpoint||mid==begin||mid==end) {
                mid=(begin+end)/2;
                std::nth_element(particles.begin()+begin,particles.begin()+mid,particles.begin()+end,
                    [axis](const Particle& a,const Particle& b) {
                        return coordinate(a,axis)<coordinate(b,axis);
                    });
            }
            q.left=build(begin,mid);
            q.right=build(mid,end);
        }
        nodes[k]=q;
        return k;
    }

    void monomials(double dx,double dy,double dz,double* a,int order) const {
        double xp[P+1],yp[P+1],zp[P+1];
        xp[0]=yp[0]=zp[0]=1.;
        for(int i=1;i<=order;++i) {
            xp[i]=xp[i-1]*dx/i;
            yp[i]=yp[i-1]*dy/i;
            zp[i]=zp[i-1]*dz/i;
        }
        int count=(order+1)*(order+2)*(order+3)/6;
        for(int i=0;i<count;++i)
            a[i]=xp[K::powers[i][0]]*yp[K::powers[i][1]]*zp[K::powers[i][2]];
    }

    void shift_moment(const double* src,double* dst,double dx,double dy,double dz) {
        double a[NM],b[NM];
        std::copy(src,src+NM,a);
        double ds[3]={dx,dy,dz};
        for(int axis=0;axis<3;++axis) {
            double pw[P+1]; pw[0]=1;
            for(int j=1;j<P;++j) pw[j]=pw[j-1]*ds[axis]/j;
            for(int i=0;i<NM;++i) {
                int ix=K::powers[i][0],iy=K::powers[i][1],iz=K::powers[i][2];
                int lim=K::powers[i][axis];
                double s=a[i];
                for(int j=1;j<=lim;++j) {
                    int ii=index[ix-(axis==0?j:0)][iy-(axis==1?j:0)][iz-(axis==2?j:0)];
                    s+=pw[j]*a[ii];
                }
                b[i]=s;
            }
            std::copy(b,b+NM,a);
        }
        for(int i=0;i<NM;++i) dst[i]+=a[i];
    }

    void upward(int ni) {
        const Node& q=nodes[ni];
        double* m=&moments[size_t(ni)*NM];
        if(q.leaf()) {
            if(vectorize) {
                V ma[NM],ha[HK::NM];
                for(int k=0;k<NM;++k) ma[k]=V(0.);
                if constexpr(H>0) for(int k=0;k<HK::NM;++k) ha[k]=V(0.);
                for(int j=q.begin;j<q.end;j+=4) {
                    V dx=V::load(&x[j])-V(q.x),dy=V::load(&y[j])-V(q.y),dz=V::load(&z[j])-V(q.z);
                    V a[NM];
                    K::monomials(dx,dy,dz,a);
                    int count=std::min(4,q.end-j);
                    __m256d mask=_mm256_castsi256_pd(_mm256_set_epi64x(
                        count>3?-1:0,count>2?-1:0,count>1?-1:0,-1));
                    for(int k=0;k<NM;++k) ma[k]+=V(_mm256_and_pd(a[k].v,mask));
                    if constexpr(H>0) {
                        V h[HK::NM];
                        HK::monomials(dx,dy,dz,h);
                        for(int k=0;k<HK::NM;++k) ha[k]+=V(_mm256_and_pd(h[k].v,mask));
                    }
                }
                for(int k=0;k<NM;++k) m[k]=sum(ma[k])*mass;
                if constexpr(H>0)
                    for(int k=0;k<HK::NM;++k) hmoments[size_t(ni)*HK::NM+k]=sum(ha[k])*mass;
            } else for(int j=q.begin;j<q.end;++j) {
                double a[NM];
                monomials(x[j]-q.x,y[j]-q.y,z[j]-q.z,a,P-1);
                for(int k=0;k<NM;++k) m[k]+=a[k]*mass;
                if constexpr(H>0) {
                    double h[HK::NM];
                    HK::monomials(x[j]-q.x,y[j]-q.y,z[j]-q.z,h);
                    for(int k=0;k<HK::NM;++k) hmoments[size_t(ni)*HK::NM+k]+=mass*h[k];
                }
            }
        } else {
            upward(q.left); upward(q.right);
            for(int c:{q.left,q.right}) {
                const auto& child=nodes[c];
                shift_moment(&moments[size_t(c)*NM],m,child.x-q.x,child.y-q.y,child.z-q.z);
                if constexpr(H>0) {
                    if(fastshift) HK::shift_moment_fast(&hmoments[size_t(c)*HK::NM],
                        child.x-q.x,child.y-q.y,child.z-q.z,&hmoments[size_t(ni)*HK::NM]);
                    else {
                        double h[HK::NM];
                        HK::monomials(child.x-q.x,child.y-q.y,child.z-q.z,h);
                        HK::shift_moment(&hmoments[size_t(c)*HK::NM],h,&hmoments[size_t(ni)*HK::NM]);
                    }
                }
            }
        }
        m[1]=m[2]=m[3]=0.;
        if constexpr(H>0) {
            double* h=&hmoments[size_t(ni)*HK::NM];
            h[1]=h[2]=h[3]=0.;
        }
    }

    void estimate_force(int ni,double tx,double ty,double tz,int own,double& fx,double& fy,double& fz) {
        if(ni==own) return;
        const auto& a=nodes[ni];
        double dx=a.x-tx,dy=a.y-ty,dz=a.z-tz;
        double r2=dx*dx+dy*dy+dz*dz+.0001;
        if(r2*.16>a.r*a.r) {
            double s=(a.end-a.begin)*mass/(r2*std::sqrt(r2));
            fx+=dx*s; fy+=dy*s; fz+=dz*s;
        } else if(a.leaf()) {
            for(int j=a.begin;j<a.end;++j) {
                dx=x[j]-tx; dy=y[j]-ty; dz=z[j]-tz;
                r2=dx*dx+dy*dy+dz*dz+.0001;
                double s=mass/(r2*std::sqrt(r2));
                fx+=dx*s; fy+=dy*s; fz+=dz*s;
            }
        } else {
            estimate_force(a.left,tx,ty,tz,own,fx,fy,fz);
            estimate_force(a.right,tx,ty,tz,own,fx,fy,fz);
        }
    }

    double estimate(int ni) {
        const auto& a=nodes[ni];
        if(a.leaf()) {
            double fx=0,fy=0,fz=0;
            estimate_force(0,a.x,a.y,a.z,ni,fx,fy,fz);
            double self=(a.end-a.begin)*mass*a.r/std::pow(a.r*a.r+.0001,1.5);
            accel[ni]=std::max(std::sqrt(fx*fx+fy*fy+fz*fz),.3*self)+1e-30;
        } else {
            accel[ni]=std::min(estimate(a.left),estimate(a.right));
        }
        return accel[ni];
    }

    void walk(int ai,int bi) {
        if(list_overflow) return;
        const auto& a=nodes[ai]; const auto& b=nodes[bi];
        if(ai==bi) {
            if(a.leaf()||(direct_factor>0&&size_t(a.end-a.begin)*(a.end-a.begin-1)<=2*merge_limit)) {
                if(append_pair(near,{ai,bi}))
                    direct_pairs+=(size_t)(a.end-a.begin)*(a.end-a.begin-1)/2;
            } else {
                walk(a.left,a.left); walk(a.left,a.right); walk(a.right,a.right);
            }
            return;
        }
        size_t products=size_t(a.end-a.begin)*(b.end-b.begin);
        if(direct_factor>0&&products<=merge_limit) {
            if(append_pair(near,{ai,bi})) direct_pairs+=products;
            return;
        }
        double dx=a.x-b.x,dy=a.y-b.y,dz=a.z-b.z,rad=std::max(a.r+b.r,gamma*std::max(a.r,b.r));
        double base=dx*dx+dy*dy+dz*dz,r2=base+0.0001;
        double metric=stf?r2:base;
        double damping=1.;
        if(stf&&damp_trace) {
            double t=base/r2;
            t*=t; t*=t; t*=t;
            damping=std::max(softfactor,t);
        }
        double rh=std::max(a.re+b.re,gamma*std::max(a.re,b.re));
        bool accept=r2*theta*theta>rad*rad;
        if constexpr(H>0) {
            double q2=rad*rad/r2,err=1.;
            for(int i=0;i<P/2;++i) err*=q2;
            double eps=mac_power;
            if(err<eps*softfactor*(base<.0016?softboost:1.)) {
                if(products<direct_factor*kernels[9].work) {
                    if(append_pair(near,{ai,bi})) direct_pairs+=products;
                } else if(append_pair(far[9],{ai,bi})) record_error(ai,bi,kernels[9]);
                return;
            }
            accept=(stf||base>.0016)&&metric*theta*theta>rh*rh&&
                (a.r+b.r)*(a.r+b.r)<.81*metric;
            if(accept) {
                // A Laplace residual uses the unsoftened convergence radius.
                double ql=rad*rad/metric,el=1.;
                for(int i=0;i<P/2;++i) el*=ql;
                accept=el*.0001<eps*metric*deltafactor*damping;
            }
        }
        if(accept&&tolerance>0.) {
            double q2=rad*rad/r2;
            double err=1/r2;
            for(int i=0;i<P/2;++i) err*=q2;
            err*=mass*std::max((b.end-b.begin)/accel[ai],(a.end-a.begin)/accel[bi]);
            accept=err<tolerance;
        }
        if(accept) {
            int hi=2,ci=2;
            if(variable) {
                if(variable!=3) {
                    double q2=rh*rh/metric;
                    if(q2<hmacthreshold[0]) hi=0;
                    else if(q2<hmacthreshold[1]) hi=1;
                }
                if(variable!=2) {
                    double q2=rad*rad/metric;
                    double e=.0001/metric;
                    for(int k=0;k<(P-4)/2;++k) e*=q2;
                    if(e<cmacthreshold[0]*damping) ci=0;
                    else if constexpr(P>=8) {
                        if(e*q2<cmacthreshold[1]*damping) ci=1;
                    }
                }
            }
            int kind=hi*3+ci;
            if(products<direct_factor*kernels[kind].work) {
                if(append_pair(near,{ai,bi})) direct_pairs+=products;
            } else if(append_pair(far[kind],{ai,bi})) record_error(ai,bi,kernels[kind]);
        } else if(a.leaf()&&b.leaf()) {
            if(append_pair(near,{ai,bi}))
                direct_pairs+=(size_t)(a.end-a.begin)*(b.end-b.begin);
        } else if(b.leaf()||(!a.leaf()&&a.r>=b.r)) {
            walk(a.left,bi); walk(a.right,bi);
        } else {
            walk(ai,b.left); walk(ai,b.right);
        }
    }

    static double sum(V a) {
        __m128d s=_mm_add_pd(_mm256_castpd256_pd128(a.v),_mm256_extractf128_pd(a.v,1));
        return _mm_cvtsd_f64(_mm_add_sd(s,_mm_unpackhi_pd(s,s)));
    }

    template<bool Quick=false> void direct(const Pair& p) {
        alignas(32) static constexpr long long masks[4][4]={
            {0,0,0,0},{-1,0,0,0},{-1,-1,0,0},{-1,-1,-1,0}};
        const Node* ap=&nodes[p.a]; const Node* bp=&nodes[p.b];
        if constexpr(Quick) {
            if(refine) {
                error_bound[p.a]+=(bp->end-bp->begin)*mass*.01;
                error_bound[p.b]+=(ap->end-ap->begin)*mass*.01;
            }
        }
        if(p.a!=p.b&&ap->end-ap->begin>bp->end-bp->begin) std::swap(ap,bp);
        const auto& a=*ap; const auto& b=*bp;
        for(int i=a.begin;i<a.end;++i) {
            V xi(x[i]),yi(y[i]),zi(z[i]),fx(0.),fy(0.),fz(0.);
            int j=(p.a==p.b)?i+1:b.begin;
            for(;j+4<=b.end;j+=4) {
                V dx=V::load(&x[j])-xi,dy=V::load(&y[j])-yi,dz=V::load(&z[j])-zi;
                V r2=dx*dx+dy*dy+dz*dz+V(.0001);
                V s;
                if constexpr(Quick) {
                    V inv(_mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(r2.v))));
                    inv=inv*(V(1.5)-V(.5)*r2*inv*inv);
                    s=V(mass)*inv*inv*inv;
                } else s=V(mass)/(r2*vsqrt(r2));
                V ux=dx*s,uy=dy*s,uz=dz*s;
                fx+=ux; fy+=uy; fz+=uz;
                (V::load(&ax[j])-ux).store(&ax[j]);
                (V::load(&ay[j])-uy).store(&ay[j]);
                (V::load(&az[j])-uz).store(&az[j]);
            }
            if(j<b.end) {
                V dx=V::load(&x[j])-xi,dy=V::load(&y[j])-yi,dz=V::load(&z[j])-zi;
                V r2=dx*dx+dy*dy+dz*dz+V(.0001);
                V s;
                if constexpr(Quick) {
                    V inv(_mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(r2.v))));
                    inv=inv*(V(1.5)-V(.5)*r2*inv*inv);
                    s=V(mass)*inv*inv*inv;
                } else s=V(mass)/(r2*vsqrt(r2));
                __m256d mask=_mm256_castsi256_pd(_mm256_load_si256(
                    reinterpret_cast<const __m256i*>(masks[b.end-j])));
                s=V(_mm256_and_pd(s.v,mask));
                V ux=dx*s,uy=dy*s,uz=dz*s;
                fx+=ux; fy+=uy; fz+=uz;
                (V::load(&ax[j])-ux).store(&ax[j]);
                (V::load(&ay[j])-uy).store(&ay[j]);
                (V::load(&az[j])-uz).store(&az[j]);
            }
            ax[i]+=sum(fx); ay[i]+=sum(fy); az[i]+=sum(fz);
        }
    }

    void multipole(size_t start,size_t end,int kind,int alternate=-1) {
        const auto& list=far[kind];
        const auto& f=kernels[alternate<0?kind:alternate];
        if(refine&&alternate>=0)
            for(size_t i=start;i<end;++i) record_error(list[i].a,list[i].b,f);
        for(size_t pos=start;pos<end;pos+=4) {
            int count=std::min(size_t(4),end-pos);
            int aa[4],bb[4];
            double dx[4],dy[4],dz[4];
            for(int lane=0;lane<4;++lane) {
                const auto& pair=list[pos+std::min(lane,count-1)];
                aa[lane]=pair.a; bb[lane]=pair.b;
                const auto& a=nodes[pair.a]; const auto& b=nodes[pair.b];
                dx[lane]=a.x-b.x; dy[lane]=a.y-b.y; dz[lane]=a.z-b.z;
            }
            bool commonA=count==4&&aa[0]==aa[1]&&aa[0]==aa[2]&&aa[0]==aa[3];
            bool commonB=count==4&&bb[0]==bb[1]&&bb[0]==bb[2]&&bb[0]==bb[3];
            V xx=V::load(dx), yy=V::load(dy), zz=V::load(dz);
            V d[NC],ma[NM],mb[NM],la[NC],lb[NC];
            V sinv=V(1.)/(xx*xx+yy*yy+zz*zz+V(.0001));
            f.derivative(xx,yy,zz,sinv,d);
            if constexpr(H>0) {
                if(f.hc) {
                    V ld[NC],hd[HK::ND],hma[HK::NM],hmb[HK::NM],hla[HK::NC],hlb[HK::NC];
                    V inv=stf?sinv:V(1.)/(xx*xx+yy*yy+zz*zz);
                    f.hderivative(xx,yy,zz,inv,hd);
                    f.to_cart(hd,ld);
                    for(int j=0;j<f.nc;++j) d[j]=d[j]-ld[j];
                    for(int j=0;j<f.hm;++j) {
                        hma[j]=commonA?V(hmoments[size_t(aa[0])*HK::NM+j]):
                               V(hmoments[size_t(aa[0])*HK::NM+j],hmoments[size_t(aa[1])*HK::NM+j],
                                 hmoments[size_t(aa[2])*HK::NM+j],hmoments[size_t(aa[3])*HK::NM+j]);
                        hmb[j]=commonB?V(hmoments[size_t(bb[0])*HK::NM+j]):
                               V(hmoments[size_t(bb[0])*HK::NM+j],hmoments[size_t(bb[1])*HK::NM+j],
                                 hmoments[size_t(bb[2])*HK::NM+j],hmoments[size_t(bb[3])*HK::NM+j]);
                    }
                    f.htranslate(hma,hmb,hd,hla,hlb);
                    for(int j=1;j<f.hc;++j) {
                        double va[4],vb[4]; hla[j].store(va); hlb[j].store(vb);
                        if(commonA) hlocals[size_t(aa[0])*HK::NC+j]+=sum(hla[j]);
                        if(commonB) hlocals[size_t(bb[0])*HK::NC+j]+=sum(hlb[j]);
                        for(int lane=0;lane<count;++lane) {
                            if(!commonA) hlocals[size_t(aa[lane])*HK::NC+j]+=va[lane];
                            if(!commonB) hlocals[size_t(bb[lane])*HK::NC+j]+=vb[lane];
                        }
                    }
                }
            }
            for(int i=0;i<f.nm;++i) {
                ma[i]=commonA?V(moments[size_t(aa[0])*NM+i]):
                      V(moments[size_t(aa[0])*NM+i],moments[size_t(aa[1])*NM+i],
                        moments[size_t(aa[2])*NM+i],moments[size_t(aa[3])*NM+i]);
                mb[i]=commonB?V(moments[size_t(bb[0])*NM+i]):
                      V(moments[size_t(bb[0])*NM+i],moments[size_t(bb[1])*NM+i],
                        moments[size_t(bb[2])*NM+i],moments[size_t(bb[3])*NM+i]);
            }
            f.translate(ma,mb,d,la,lb);
            for(int i=1;i<f.nc;++i) {
                double va[4],vb[4];
                la[i].store(va); lb[i].store(vb);
                if(commonA) locals[size_t(aa[0])*NC+i]+=sum(la[i]);
                if(commonB) locals[size_t(bb[0])*NC+i]+=sum(lb[i]);
                for(int lane=0;lane<count;++lane) {
                    if(!commonA) locals[size_t(aa[lane])*NC+i]+=va[lane];
                    if(!commonB) locals[size_t(bb[lane])*NC+i]+=vb[lane];
                }
            }
        }
    }

    void shift_local(const double* src,double* dst,double dx,double dy,double dz) {
        double a[NC],b[NC];
        std::copy(src,src+NC,a);
        double ds[3]={dx,dy,dz};
        for(int axis=0;axis<3;++axis) {
            double pw[P+1]; pw[0]=1;
            for(int j=1;j<=P;++j) pw[j]=pw[j-1]*ds[axis]/j;
            for(int i=1;i<NC;++i) {
                int ix=K::powers[i][0],iy=K::powers[i][1],iz=K::powers[i][2];
                int lim=P-ix-iy-iz;
                double s=a[i];
                for(int j=1;j<=lim;++j) {
                    int ii=index[ix+(axis==0?j:0)][iy+(axis==1?j:0)][iz+(axis==2?j:0)];
                    s+=pw[j]*a[ii];
                }
                b[i]=s;
            }
            std::copy(b+1,b+NC,a+1);
        }
        for(int i=1;i<NC;++i) dst[i]+=a[i];
    }

    void evaluate_leaf(int ni) {
        const auto& q=nodes[ni];
        const double* l=&locals[size_t(ni)*NC];
        double hl[HK::ND];
        if constexpr(H>0) HK::extend(&hlocals[size_t(ni)*HK::NC],hl);
            if(vectorize) {
                for(int j=q.begin;j<q.end;j+=4) {
                    V dx=V::load(&x[j])-V(q.x),dy=V::load(&y[j])-V(q.y),dz=V::load(&z[j])-V(q.z);
                    V t[NM],sx(0.),sy(0.),sz(0.);
                    K::monomials(dx,dy,dz,t);
                    K::local_evaluate(l,t,sx,sy,sz);
                    if constexpr(H>0) {
                        V h[HK::NM];
                        HK::monomials(dx,dy,dz,h);
                        HK::evaluate(hl,h,sx,sy,sz);
                    }
                    if(j+4<=q.end) {
                        (V::load(&ax[j])+sx).store(&ax[j]);
                        (V::load(&ay[j])+sy).store(&ay[j]);
                        (V::load(&az[j])+sz).store(&az[j]);
                    } else {
                        double vx[4],vy[4],vz[4];
                        sx.store(vx); sy.store(vy); sz.store(vz);
                        for(int k=0;k<q.end-j;++k) {
                            ax[j+k]+=vx[k]; ay[j+k]+=vy[k]; az[j+k]+=vz[k];
                        }
                    }
                }
            } else for(int j=q.begin;j<q.end;++j) {
                double t[NM];
                monomials(x[j]-q.x,y[j]-q.y,z[j]-q.z,t,P-1);
                double sx=0,sy=0,sz=0;
                for(int i=0;i<NM;++i) {
                    int ix=K::powers[i][0],iy=K::powers[i][1],iz=K::powers[i][2];
                    sx+=l[index[ix+1][iy][iz]]*t[i];
                    sy+=l[index[ix][iy+1][iz]]*t[i];
                    sz+=l[index[ix][iy][iz+1]]*t[i];
                }
                if constexpr(H>0) {
                    double h[HK::NM];
                    HK::monomials(x[j]-q.x,y[j]-q.y,z[j]-q.z,h);
                    HK::evaluate(hl,h,sx,sy,sz);
                }
                ax[j]+=sx; ay[j]+=sy; az[j]+=sz;
            }
    }

    void propagate(int ni,int c) {
        const auto& q=nodes[ni]; const auto& child=nodes[c];
        shift_local(&locals[size_t(ni)*NC],&locals[size_t(c)*NC],child.x-q.x,child.y-q.y,child.z-q.z);
        if constexpr(H>0) {
            if(fastshift) HK::shift_local_fast(&hlocals[size_t(ni)*HK::NC],
                child.x-q.x,child.y-q.y,child.z-q.z,&hlocals[size_t(c)*HK::NC]);
            else {
                double h[HK::NM];
                HK::monomials(child.x-q.x,child.y-q.y,child.z-q.z,h);
                HK::shift_local(&hlocals[size_t(ni)*HK::NC],h,&hlocals[size_t(c)*HK::NC]);
            }
        }
    }

    void downward(int ni) {
        const auto& q=nodes[ni];
        if(q.leaf()) evaluate_leaf(ni);
        else {
            for(int c:{q.left,q.right}) {
                if(refine) error_bound[c]+=error_bound[ni];
                propagate(ni,c);
                downward(c);
            }
        }
    }

    void reset_accumulators() {
        std::fill(locals.begin(),locals.end(),0.);
        std::fill(hlocals.begin(),hlocals.end(),0.);
        std::fill(ax.begin(),ax.end(),0.);
        std::fill(ay.begin(),ay.end(),0.);
        std::fill(az.begin(),az.end(),0.);
    }

    double profile_downward() {
        std::vector<int> branches,leaves;
        for(size_t i=0;i<nodes.size();++i)
            (nodes[i].leaf()?leaves:branches).push_back(i);
        double start=cpu_time();
        int shifts=0;
        size_t stride=std::max(size_t(1),branches.size()/256);
        for(size_t i=0;i<branches.size();i+=stride) {
            int a=branches[i];
            propagate(a,nodes[a].left);
            propagate(a,nodes[a].right);
            shifts+=2;
        }
        double shift_cost=(cpu_time()-start)/std::max(shifts,1);
        start=cpu_time();
        int count=0;
        stride=std::max(size_t(1),leaves.size()/128);
        for(size_t i=0;i<leaves.size();i+=stride) {
            int a=leaves[i];
            evaluate_leaf(a);
            count+=nodes[a].end-nodes[a].begin;
        }
        return shift_cost*(nodes.size()-1)+(cpu_time()-start)*n/std::max(count,1);
    }

    double profile_multipole(int kind) {
        const auto& list=far[kind];
        if(list.empty()) return 0;
        double start=cpu_time();
        size_t done=0;
        size_t bins=std::min(size_t(8),list.size());
        for(size_t bin=0;bin<bins;++bin) {
            size_t a=bin*list.size()/bins;
            size_t b=std::min(a+(n<=65536&&!block_profiles?32:64),list.size());
            multipole(a,b,kind);
            done+=b-a;
        }
        return (cpu_time()-start)/done;
    }

    double profile_direct() {
        if(near.empty()) return 3e-9;
        double start=cpu_time();
        unsigned long long count=0;
        size_t samples=n<=65536?1024:n<=262144?2048:4096;
        size_t bins=std::min(block_profiles?samples/64:samples,near.size());
        for(size_t bin=0;bin<bins;++bin) {
            size_t begin=bin*near.size()/bins,end=(bin+1)*near.size()/bins;
            unsigned long long h=(bin+0x9e3779b9ULL)*11400714819323198485ULL;
            h^=h>>29;
            size_t length=block_profiles?std::min(size_t(64),end-begin):1;
            size_t start=begin+h%(end-begin-length+1);
            for(size_t k=start;k<start+length;++k) {
                const auto& pair=near[k];
                direct(pair);
                auto na=nodes[pair.a].end-nodes[pair.a].begin;
                auto nb=nodes[pair.b].end-nodes[pair.b].begin;
                count+=pair.a==pair.b?(size_t)na*(na-1)/2:(size_t)na*nb;
            }
        }
        return (cpu_time()-start)/std::max(count,1ULL);
    }

    template<int Order=4> void low_multipole(const std::vector<Pair>& list,size_t start,size_t end) {
        using Q=Kernel<Order>;
        if(refine) {
            FarKernel f=make_kernel<Order,0>();
            for(size_t i=start;i<end;++i) record_error(list[i].a,list[i].b,f);
        }
        for(size_t pos=start;pos<end;pos+=4) {
            int count=std::min(size_t(4),end-pos),aa[4],bb[4];
            double dx[4],dy[4],dz[4];
            for(int lane=0;lane<4;++lane) {
                const auto& pair=list[pos+std::min(lane,count-1)];
                aa[lane]=pair.a; bb[lane]=pair.b;
                dx[lane]=nodes[pair.a].x-nodes[pair.b].x;
                dy[lane]=nodes[pair.a].y-nodes[pair.b].y;
                dz[lane]=nodes[pair.a].z-nodes[pair.b].z;
            }
            V xx=V::load(dx),yy=V::load(dy),zz=V::load(dz);
            V d[Q::NC],ma[Q::NM],mb[Q::NM],la[Q::NC],lb[Q::NC];
            Q::derivatives(xx,yy,zz,V(1.)/(xx*xx+yy*yy+zz*zz+V(.0001)),d);
            for(int i=0;i<Q::NM;++i) {
                ma[i]=V(moments[size_t(aa[0])*NM+i],moments[size_t(aa[1])*NM+i],
                        moments[size_t(aa[2])*NM+i],moments[size_t(aa[3])*NM+i]);
                mb[i]=V(moments[size_t(bb[0])*NM+i],moments[size_t(bb[1])*NM+i],
                        moments[size_t(bb[2])*NM+i],moments[size_t(bb[3])*NM+i]);
            }
            Q::translate(ma,mb,d,la,lb);
            for(int i=1;i<Q::NC;++i) {
                double va[4],vb[4]; la[i].store(va); lb[i].store(vb);
                for(int lane=0;lane<count;++lane) {
                    locals[size_t(aa[lane])*NC+i]+=va[lane];
                    locals[size_t(bb[lane])*NC+i]+=vb[lane];
                }
            }
        }
    }

    void exact_target(int i) {
        V xx(x[i]),yy(y[i]),zz(z[i]),fx(0.),fy(0.),fz(0.);
        int j=0;
        for(;j+4<=n;j+=4) {
            V dx=V::load(&x[j])-xx,dy=V::load(&y[j])-yy,dz=V::load(&z[j])-zz;
            V r2=dx*dx+dy*dy+dz*dz+V(.0001);
            V f=V(mass)/(r2*vsqrt(r2));
            fx+=dx*f; fy+=dy*f; fz+=dz*f;
        }
        double sx=sum(fx),sy=sum(fy),sz=sum(fz);
        for(;j<n;++j) {
            double dx=x[j]-x[i],dy=y[j]-y[i],dz=z[j]-z[i];
            double r2=dx*dx+dy*dy+dz*dz+.0001;
            double f=mass/(r2*std::sqrt(r2));
            sx+=dx*f; sy+=dy*f; sz+=dz*f;
        }
        ax[i]=sx; ay[i]=sy; az[i]=sz;
    }

    size_t refine_targets() {
        if(!refine) return 0;
        double deadline=budget-.03-n*3e-7;
        if(deadline-cpu_time()<.02+n*5e-8) return 0;
        struct Priority { double score; int i; };
        std::vector<Priority> priorities;
        priorities.reserve(n);
        double best=0.; int best_i=-1;
        for(size_t node=0;node<nodes.size();++node) {
            const auto& q=nodes[node];
            if(!q.leaf()||error_bound[node]==0) continue;
            double e2=error_bound[node]*error_bound[node];
            for(int i=q.begin;i<q.end;++i) {
                double a2=ax[i]*ax[i]+ay[i]*ay[i]+az[i]*az[i];
                double score=e2/(a2+1e-300);
                priorities.push_back({score,i});
                if(score>best) { best=score; best_i=i; }
            }
        }
        if(best_i<0||deadline-cpu_time()<.01+n*3e-9) return 0;
        double start=cpu_time();
        exact_target(best_i);
        double cost=std::max(1e-7,cpu_time()-start);
        double available=deadline-cpu_time();
        if(available<.005+n*1e-7) return 1;
        size_t k=std::min(priorities.size(),size_t(1+std::max(0.,available)/(cost*.75)));
        auto descending=[](const Priority& a,const Priority& b) { return a.score>b.score; };
        if(k<priorities.size()) std::nth_element(priorities.begin(),priorities.begin()+k,priorities.end(),descending);
        std::sort(priorities.begin(),priorities.begin()+k,descending);
        size_t done=1;
        for(size_t j=0;j<k;++j) {
            if(priorities[j].i==best_i) continue;
            if(cpu_time()+2*cost>=deadline) break;
            start=cpu_time();
            exact_target(priorities[j].i);
            double elapsed=cpu_time()-start;
            cost=std::max(elapsed,.9*cost);
            ++done;
        }
        return done;
    }

public:
    Solver(std::vector<Particle>&& data,int leaf,double th)
        : n(data.size()),leafsize(leaf),mass(data[0].m),theta(th),particles(std::move(data)) {
        if(const char* a=std::getenv("GALAXY_CPU_LIMIT_S")) budget=std::atof(a);
        automatic=budget>0&&std::getenv("FMM_THETA")==nullptr;
        fastshift=std::getenv("FMM_DENSESHIFT")==nullptr;
        vectorize=std::getenv("FMM_SCALAR")==nullptr;
        moment_radius=std::getenv("FMM_MOMENT_RADIUS")!=nullptr;
        stf=std::getenv("FMM_LAPLACE")==nullptr;
        damp_trace=std::getenv("FMM_NODAMP")==nullptr;
        block_profiles=std::getenv("FMM_OLDBLOCKS")==nullptr;
        refine=budget>0&&std::getenv("FMM_NOREFINE")==nullptr;
        if(const char* a=std::getenv("FMM_DIRECT_FACTOR")) direct_factor=std::atof(a);
        if(const char* a=std::getenv("FMM_VARIABLE")) variable=std::atoi(a);
        if(const char* a=std::getenv("FMM_VARSCALE")) varscale=std::atof(a);
        if(const char* a=std::getenv("FMM_VARDELTA")) var_delta=std::atof(a);
        midpoint=automatic||std::getenv("FMM_MIDPOINT")!=nullptr;
        if(automatic) { gamma=2.; shape=H>12?.25:.2; softfactor=.01; }
        if(const char* a=std::getenv("FMM_TOL")) tolerance=std::atof(a);
        if(const char* a=std::getenv("FMM_GAMMA")) gamma=std::atof(a);
        if(const char* a=std::getenv("FMM_SHAPE")) shape=std::atof(a);
        if(const char* a=std::getenv("FMM_SOFTFACTOR")) softfactor=std::atof(a);
        if(const char* a=std::getenv("FMM_DELTAFACTOR")) deltafactor=std::atof(a);
        if(const char* a=std::getenv("FMM_SOFTBOOST")) softboost=std::atof(a);
        if(shape>0) {
            std::vector<double> px,py,pz,rr;
            for(int i=0;i<n;i+=std::max(1,n/4096)) {
                px.push_back(particles[i].x); py.push_back(particles[i].y); pz.push_back(particles[i].z);
            }
            auto median=[](std::vector<double>& a) {
                std::nth_element(a.begin(),a.begin()+a.size()/2,a.end());
                return a[a.size()/2];
            };
            cx=median(px); cy=median(py); cz=median(pz);
            for(int i=0;i<n;i+=std::max(1,n/4096)) {
                double dx=particles[i].x-cx,dy=particles[i].y-cy,dz=particles[i].z-cz;
                rr.push_back(std::sqrt(dx*dx+dy*dy+dz*dz));
            }
            scale=median(rr);
        }
        for(int i=0;i<NC;++i) index[K::powers[i][0]][K::powers[i][1]][K::powers[i][2]]=i;
        factorial[0]=1.;
        for(int i=1;i<=P;++i) factorial[i]=factorial[i-1]*i;
        set_family<H-4>(0); set_family<H-2>(3); set_family<H>(6);
        kernels[9]=make_kernel<P,0>();
    }
    bool run(const char* output) {
        double t0=cpu_time();
        nodes.reserve(4*n/leafsize+100);
        build(0,n);
        constexpr size_t coefficients=NM+NC+(H>0?HK::NM+HK::NC:0);
        while(size_t(nodes.size())*coefficients*sizeof(double)>size_t(1000)*1024*1024) {
            leafsize*=2; shape*=1.25;
            nodes.clear();
            build(0,n);
        }
        if(nodes.capacity()>nodes.size()*2) nodes.shrink_to_fit();
        x.resize(n+4); y.resize(n+4); z.resize(n+4);
        ax.resize(n+4); ay.resize(n+4); az.resize(n+4);
        ids.resize(n);
        for(int i=0;i<n;++i) {
            x[i]=particles[i].x; y[i]=particles[i].y; z[i]=particles[i].z; ids[i]=particles[i].id;
        }
        std::vector<Particle>().swap(particles);
        moments.resize(size_t(nodes.size())*NM);
        locals.resize(size_t(nodes.size())*NC);
        if constexpr(H>0) {
            hmoments.resize(size_t(nodes.size())*HK::NM);
            hlocals.resize(size_t(nodes.size())*HK::NC);
        }
        if(refine) error_bound.resize(nodes.size());
        fixed_bytes=(moments.capacity()+locals.capacity()+hmoments.capacity()+hlocals.capacity()+
            x.capacity()+y.capacity()+z.capacity()+ax.capacity()+ay.capacity()+az.capacity()+
            error_bound.capacity())*sizeof(double)+
            nodes.capacity()*sizeof(Node)+ids.capacity()*sizeof(int)+size_t(32)*1024*1024;
        upward(0);
        if(tolerance>0) {
            accel.resize(nodes.size());
            estimate(0);
        }
        double t1=cpu_time();
        make_lists();
        double down_estimate=0,direct_rate=0;
        std::array<double,10> far_rate={};
        if(automatic&&n>1024) {
            down_estimate=profile_downward();
            for(int i=0;i<10;++i) far_rate[i]=profile_multipole(i);
            direct_rate=profile_direct();
            reset_accumulators();
            double slope=3.7;
            for(int iteration=0;iteration<3;++iteration) {
                double predicted=far_cost(far_rate)+direct_pairs*direct_rate;
                double available=std::max(.001,budget-cpu_time()-down_estimate*1.4-.015-n*1e-7);
                double target=available*.90;
                if(predicted>=target*.94&&predicted<=target) break;
                double next=std::clamp(theta*std::pow(predicted/target,1/slope),memory_theta,
                                      std::max(.88,memory_theta));
                if(std::abs(next-theta)<.002) break;
                double previous_theta=theta;
                size_t previous_counts[10];
                for(int i=0;i<10;++i) previous_counts[i]=far[i].size();
                auto previous_direct=direct_pairs;
                theta=next;
                make_lists();
                direct_rate=std::max(direct_rate,profile_direct());
                for(int i=0;i<10;++i) far_rate[i]=std::max(far_rate[i],profile_multipole(i));
                reset_accumulators();
                if(block_profiles&&std::abs(std::log(previous_theta/theta))>.002) {
                    double before=previous_direct*direct_rate;
                    for(int i=0;i<10;++i) before+=previous_counts[i]*far_rate[i];
                    double after=far_cost(far_rate)+direct_pairs*direct_rate;
                    double estimate=std::log(after/before)/std::log(previous_theta/theta);
                    if(std::isfinite(estimate))
                        slope=.7*std::clamp(estimate,1.5,6.)+.3*slope;
                }
            }
            for(int guard=0;guard<3;++guard) {
                double predicted=far_cost(far_rate)+direct_pairs*direct_rate;
                double available=budget-cpu_time()-down_estimate*1.5-.02-n*1e-7;
                if(predicted<=available*.94||theta>=.98) break;
                theta=std::min(.98,std::max(theta+.01,
                    theta*std::pow(predicted/std::max(.001,available*.85),.30)));
                make_lists();
                for(int i=0;i<10;++i) if(!far[i].empty()&&far_rate[i]==0)
                    far_rate[i]=kernels[i].work*4.5e-10+7e-7;
            }
            std::fprintf(stderr,"budget=%.5g elapsed=%.4f down=%.4f rates=%.4g,%.4g,%.4g\n",
                         budget,cpu_time(),down_estimate,far_rate[8],far_rate[9],direct_rate);
        }
        double t2=cpu_time();
        size_t low_far_count=0,low_near_count=0;
        bool low=false;
        double deadline=budget-down_estimate*1.5-.02-n*1e-7;
        double cheap_far_reserve=far_count()*3e-7;
        for(size_t pos=0;pos<near.size();++pos) {
            if(automatic&&pos%512==0)
                low=cpu_time()>deadline-cheap_far_reserve-.05*budget;
            if(low) { direct<true>(near[pos]); ++low_near_count; } else direct(near[pos]);
        }
        double t3=cpu_time();
        {
            double remaining=far_cost(far_rate);
            size_t pairs_left=far_count();
            for(int kind=0;kind<10;++kind) {
                const auto& list=far[kind];
                for(size_t pos=0;pos<list.size();pos+=512) {
                    size_t end=std::min(pos+512,list.size());
                    double now=automatic?cpu_time():0;
                    if(automatic&&now+remaining>deadline) {
                        int alternative=-1;
                        if(kind<9) {
                            int hi=kind/3,ci=kind%3;
                            int candidates[5]={hi*3+std::max(0,ci-1),
                                std::max(0,hi-1)*3+ci,
                                std::max(0,hi-1)*3+std::max(0,ci-1),1,0};
                            double allowed=(deadline-now)/std::max(size_t(1),pairs_left)*.92;
                            for(int a:candidates) {
                                if(kernels[a].q>kernels[kind].q||kernels[a].h>kernels[kind].h) continue;
                                double rate=far_rate[kind]*(kernels[a].work+1500.)/(kernels[kind].work+1500.);
                                if(a!=kind&&rate<allowed) { alternative=a; break; }
                            }
                        }
                        if(alternative>=0) multipole(pos,end,kind,alternative);
                        else if(now+pairs_left*1.5e-6<deadline) {
                            low_multipole<(P<8?P:8)>(list,pos,end);
                        } else low_multipole<4>(list,pos,end);
                        low_far_count+=end-pos;
                    } else multipole(pos,end,kind);
                    remaining-=(end-pos)*far_rate[kind];
                    pairs_left-=end-pos;
                }
            }
        }
        double t4=cpu_time();
        downward(0);
        double t5=cpu_time();
        std::vector<double>().swap(moments);
        std::vector<double>().swap(hmoments);
        size_t refined=refine_targets();
        std::vector<std::array<double,3>> result(n);
        for(int i=0;i<n;++i)
            result[ids[i]]={frame[0][0]*ax[i]+frame[0][1]*ay[i]+frame[0][2]*az[i],
                           frame[1][0]*ax[i]+frame[1][1]*ay[i]+frame[1][2]*az[i],
                           frame[2][0]*ax[i]+frame[2][1]*ay[i]+frame[2][2]*az[i]};
        FILE* f=std::fopen(output,"wb");
        if(!f) return false;
        bool ok=std::fwrite(result.data(),sizeof(result[0]),n,f)==size_t(n);
        ok=(std::fclose(f)==0)&&ok;
        std::fprintf(stderr,"P=%d N=%d leaf=%d theta=%.4f nodes=%zu far=%zu near=%zu direct=%llu "
                     "cpu=%.4f build=%.4f walk=%.4f m2l=%.4f p2p=%.4f down=%.4f\n",
                     P,n,leafsize,theta,nodes.size(),far_count(),near.size(),direct_pairs,
                     cpu_time()-t0,t1-t0,t2-t1,t4-t3,t3-t2,t5-t4);
        if constexpr(H>0) std::fprintf(stderr,"harmonic=%d soft=%zu\n",H,far[9].size());
        if(variable) {
            std::fprintf(stderr,"orders:");
            for(int i=0;i<9;++i) if(!far[i].empty())
                std::fprintf(stderr," %d/%d=%zu",kernels[i].q,kernels[i].h,far[i].size());
            std::fprintf(stderr,"\n");
        }
        if(automatic) std::fprintf(stderr,"fallback_far=%zu fallback_near=%zu\n",low_far_count,low_near_count);
        if(refine) std::fprintf(stderr,"refined=%zu\n",refined);
        return ok;
    }
};

int main(int argc,char** argv) {
    if(argc!=3) return 1;
    mallopt(M_MMAP_THRESHOLD,128*1024);
    struct stat st;
    if(stat(argv[1],&st)||st.st_size<=0||st.st_size%32) return 2;
    size_t n=st.st_size/32;
    std::vector<std::array<double,4>> raw(n);
    FILE* f=std::fopen(argv[1],"rb");
    if(!f) return 3;
    size_t got=std::fread(raw.data(),sizeof(raw[0]),n,f);
    std::fclose(f);
    if(got!=n) return 4;
    std::vector<Particle> particles(n);
    for(size_t i=0;i<n;++i) particles[i]={raw[i][0],raw[i][1],raw[i][2],raw[i][3],int(i)};
    std::vector<std::array<double,4>>().swap(raw);
    if(const char* a=std::getenv("FMM_ROTATE")) {
        if(std::atoi(a)!=0) principal_frame(particles,std::atoi(a)==2);
    } else principal_frame(particles,true);
    int order=n>=1048576?12:10;
    double theta=.6;
    if(const char* a=std::getenv("FMM_ORDER")) order=std::atoi(a);
    int leaf=order==12?160:128;
    if(const char* a=std::getenv("FMM_LEAF")) leaf=std::atoi(a);
    if(const char* a=std::getenv("FMM_THETA")) theta=std::atof(a);
    bool ok=false;
    {
        if(order==12) ok=Solver<12,16>(std::move(particles),leaf,theta).run(argv[2]);
        else ok=Solver<10,16>(std::move(particles),leaf,theta).run(argv[2]);
        return ok?0:5;
    }
}
