#include "optimizer.hpp"

#include <chrono>
#include <cstdio>

namespace {
constexpr double isq2=0.7071067811865475244;
constexpr double sq2=1.4142135623730950488;
using Clock=std::chrono::steady_clock;
struct Random {
    uint64_t a=0xa7cf4895167df483ULL;
    double next(){
        a^=a>>12;a^=a<<25;a^=a>>27;
        auto x=(a*0x2545f4914f6cdd1dULL)>>11;
        return std::ldexp(double(x),-52)-1.;
    }
};
double vdot(const double* v,const double* w,int k){
    double a=0,b=0,c=0,d=0;
    int j=0;
    for(;j+3<k;j+=4){a+=v[j]*w[j];b+=v[j+1]*w[j+1];c+=v[j+2]*w[j+2];d+=v[j+3]*w[j+3];}
    for(;j<k;++j)a+=v[j]*w[j];
    return (a+b)+(c+d);
}
}

bool diagonal_sdp(const Model& p,Verifier& verify) {
    if(p.P.nnz() || p.cones.size()!=1 || p.cones[0].kind!=Semidefinite) return false;
    int n=p.cones[0].dim;
    if(n!=p.n() || p.A.nnz()!=n || n<36 || n>2200) return false;
    Ind vr(n,-1),rp(n+1);
    Vec c(n),sg(n),q(n);
    for(int j=0;j<n;++j) {
        int k=p.A.p[j];
        if(p.A.p[j+1]!=k+1)return false;
        int r=p.A.i[k];
        int i=int((std::sqrt(9+8.*r)-3)/2+1e-7);
        if(i<0 || i>=n || i*(i+3LL)/2!=r || vr[i]>=0)return false;
        vr[i]=j;
        sg[i]=p.A.x[k];
        c[i]=-p.q[j]/sg[i];
        if(c[i]<0){
            Vec x(n);x[j]=-std::copysign(1.,sg[i]);
            return verify.dual(x);
        }
        if(!(c[i]>0) || !std::isfinite(c[i]))return false;
        q[i]=std::sqrt(c[i]);
    }
    double b0=norm(p.b);
    for(int j=0,ix=0;j<n;++j) {
        for(int i=0;i<j;++i,++ix) if(p.b[ix]!=0) {++rp[i+1];++rp[j+1];}
        ++ix;
    }
    for(int j=0;j<n;++j)rp[j+1]+=rp[j];
    if(rp[n]>850000 && n<300)return false;
    Ind ci(rp.back()),tmp=rp;
    Vec bv(ci.size());
    for(int j=0,ix=0;j<n;++j){
        for(int i=0;i<j;++i,++ix) if(p.b[ix]!=0) {
            double a=(p.b[ix]/std::max(1e-150,b0))*isq2*q[i]*q[j];
            int s=tmp[i]++,t=tmp[j]++;
            ci[s]=j;ci[t]=i;bv[s]=bv[t]=a;
        }
        ++ix;
    }
    double bc=norm(bv);
    if(!std::isfinite(bc))return false;
    if(bc==0)bc=1;
    for(double& a:bv)a/=bc;
    auto now=Clock::now();
    Random rand;
    const int rank=std::min(32, std::max(8,4+int(std::sqrt(n)*.62)));
    const int r=(rank+3)/4*4;
    Vec U((size_t)n*r),g(r),x(n),s(p.m()),z(p.m());
    for(int i=0;i<n;++i) {
        double* a=U.data()+r*i;
        for(int j=0;j<r;++j)a[j]=rand.next();
        double h=1/std::sqrt(vdot(a,a,r));
        for(int j=0;j<r;++j)a[j]*=h;
    }
    auto recover=[&](){
        for(int i=0;i<n;++i){
            const double* u=U.data()+r*i;
            double energy=0;
            for(int j=rp[i];j<rp[i+1];++j)energy+=bv[j]*vdot(u,U.data()+r*ci[j],r);
            double h=energy/c[i]*(bc*b0);
            int j=i*(i+3LL)/2;
            x[vr[i]]=(p.b[j]+h)/sg[i];
        }
        p.A.mul(x.data(),s.data());
        for(int j=0;j<p.m();++j)s[j]=p.b[j]-s[j];
        double sn=std::max(1.,norm(s));
        if(!std::isfinite(sn))return false;
        double f=verify.cones().defect(s,false,8.6e-9);
        if(f>2.7e-8)return false;
        for(int j=0,at=0;j<n;++j){
            const double* u=U.data()+r*j;
            for(int i=0;i<j;++i)z[at++]=sq2*q[i]*q[j]*vdot(u,U.data()+r*i,r);
            z[at++]=c[j];
        }
        if(verify.solved(x,s,z,false))return true;
        double sh=std::max(0.,f-6.5e-9)*sn;
        if(sh>0){
            // Use the remaining primal residual tolerance before perturbing x.
            double lim=5.5e-9*(1+std::max({sn,b0,norm(p.A.mul(x))}));
            double shift=std::min(sh,lim);
            for(int j=0;j<n;++j) {
                x[vr[j]]-=(sh-shift)/sg[j];
                s[j*(j+3LL)/2]+=sh;
            }
            if(verify.solved(x,s,z,false)) return true;
        }
        return false;
    };
    int last=-1000,tries=0;
    double prev=INFINITY;
    double budget=n<150?.14:n<550?.43:.8;
    for(int it=0;it<2200;++it){
        double delta=0;
        double momentum=it<7?0.:it<45?.55:.72;
        for(int i=0;i<n;++i){
            double* u=U.data()+r*i;
            std::fill(g.begin(),g.end(),0);
            for(int k=rp[i];k<rp[i+1];++k){
                const double a=bv[k]; const double* t=U.data()+r*ci[k];
                for(int j=0;j<r;++j)g[j]+=a*t[j];
            }
            double ng=vdot(g.data(),g.data(),r);
            if(ng>0) {
                double h=-1/std::sqrt(ng);
                double cs=std::clamp(h*vdot(u,g.data(),r),-1.,1.);
                delta=std::max(delta,1-cs);
                double a=(1+momentum),b=momentum;
                if(cs<0.85){a=1;b=0;}
                double t=1/std::sqrt((a-b)*(a-b)+2*a*b*(1-cs));
                for(int j=0;j<r;++j)u[j]=(a*h*g[j]-b*u[j])*t;
            }
        }
        if((it>=12 && delta<1e-13 && (delta<prev*.035 || it>=last+200)) ||
           (it==0 && rp.back()==0) || (it>50 && delta<1e-16 && !tries)) {
            last=it;prev=delta;++tries;
            if(debug)std::fprintf(stderr,"diagonal SDP sweep %d %.2g\n",it,delta);
            if(recover())return true;
            if(tries>=5 && delta<1.2e-16) break;
        }
        if(it%12==0 && std::chrono::duration<double>(Clock::now()-now).count()>budget)break;
    }
    if(last<0 && n<=900 && recover())return true;
    return false;
}
