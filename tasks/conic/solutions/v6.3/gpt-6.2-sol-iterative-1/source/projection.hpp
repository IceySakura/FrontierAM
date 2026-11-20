#pragma once
#include "cones.hpp"

namespace conic {

struct ExpProjection {
    std::array<double,3> s{}, theta{};
    std::array<double,9> basis{};
    bool rotated=false;
};

inline bool exp_in(double u,double v,double w,double tolerance=0.) {
    return (v>=-tolerance && v<=tolerance && u<=tolerance && w>=-tolerance) ||
           (v>0. && w>0. && exp_margin(u,v,w)>=-tolerance);
}

inline bool project_exp(const double* input,ExpProjection& out,double warm=inf) {
    using F=long double;
    out.rotated=false;
    const double scale=std::max(1e-300,norm(input,3));
    const double u=input[0]/scale,v=input[1]/scale,w=input[2]/scale;
    if(exp_in(u,v,w,5e-16)) {
        out.s={input[0],input[1],input[2]};out.theta={1.,1.,1.};return true;
    }
    const F eu=std::exp(1.L);
    if(exp_in(v,u,-double(eu)*w,5e-16)) {
        out.s={0.,0.,0.};out.theta={0.,0.,0.};return true;
    }
    const double fx=std::min(u,0.), fz=std::max(w,0.);
    if(exp_in(v, u-fx, double(eu)*(fz-w),5e-16)) {
        out.s={fx*scale,0.,fz*scale};
        out.theta={u<0.?1.:0.,0.,w>0.?1.:0.};
        return true;
    }

    struct Eval {F t,e,y, score, f, fp, lam;};
    auto eval=[&](F t) {
        const F e=std::exp(t), n=(F)u*t+v+(F)w*e, den=1+t*t+e*e;
        const F y=n/den, dp=2*(t+e*e), np=u+w*e;
        const F yp=(np-y*dp)/den;
        const F g0=u-y*t,g2=w-y*e;
        const F f=g0+g2*e;
        const F fp=-(yp*t+y)-(yp+y)*e*e+g2*e;
        return Eval{t,e,y,std::max(F(0),n/std::sqrt(den)),f,fp,g0};
    };
    auto set=[&](const Eval& T,double tol) {
        if(!(T.y>0.) || !(T.lam>=-tol))return false;
        const F t=T.t,e=T.e,s0=T.y*t,s1=T.y,s2=T.y*e;
        const std::array<F,3> normal={1.,1-t,-1/e};
        const F sx=F(u)-s0,sy=F(v)-s1,sz=F(w)-s2;
        if(std::max({std::abs(sx-T.lam),std::abs(sy-T.lam*normal[1]),
                     std::abs(sz-T.lam*normal[2])})>tol)return false;
        if(T.lam<0.)return false;
        const F ss=std::sqrt(t*t+1+e*e), nn=std::sqrt(1+(1-t)*(1-t)+1/e/e);
        const std::array<F,3> b0={t/ss,1/ss,e/ss};
        const std::array<F,3> b2={1/nn,(1-t)/nn,-1/e/nn};
        std::array<F,3> b1;
        for(int i=0;i<3;++i)b1[i]=b2[(i+1)%3]*b0[(i+2)%3]-b2[(i+2)%3]*b0[(i+1)%3];
        const F dv=b1[1]-b1[2]/e, theta=1/(1+T.lam*dv*dv/T.y);
        out.rotated=true;
        out.s={double(T.y*ss*scale),0.,0.};
        out.theta={1.,double(theta),0.};
        for(int i=0;i<3;++i) {
            out.basis[i]=double(b0[i]);out.basis[i+3]=double(b1[i]);out.basis[i+6]=double(b2[i]);
        }
        return true;
    };
    std::array<double,5> seeds={warm,v>0&&w>0?std::log(w)-std::log(v):0.,
        v>0?u/v:1.,u!=0?1.-v/u:0.,0.};
    Eval best=eval(0.);
    auto try_start=[&](double start) {
        Eval cur=eval(std::clamp(start,-75.,75.));
        if(cur.score>best.score)best=cur;
        for(int k=0;k<20;++k) {
            if(set(cur,3e-13))return true;
            if(!(cur.fp<0. && cur.y>0.))break;
            F step=std::clamp(-cur.f/cur.fp,-F(6),F(6));
            if(std::abs(step)<1e-16)break;
            Eval next=eval(std::clamp(cur.t+step,-F(80),F(80)));
            for(int j=0;j<6 && next.score<cur.score-3e-19L;++j) {
                step*=.5L;
                next=eval(cur.t+step);
            }
            if(next.score<cur.score-3e-19L)break;
            cur=next;
            if(cur.score>best.score)best=cur;
        }
        return false;
    };
    for(double x:seeds)if(std::isfinite(x) && try_start(x))return true;
    for(double x:{-70.,-28.,-14.,-8.,-4.,-2.,-1.,1.,2.,4.,8.,14.,28.,70.}) {
        Eval e=eval(x);if(e.score>best.score)best=e;
    }
    if(try_start(double(best.t)))return true;
    Eval lo=best,hi=best;
    for(F d=.15;d<180 && (lo.f<=0 || hi.f>=0);d*=1.65) {
        if(lo.f<=0)lo=eval(std::max(F(-82),best.t-d));
        if(hi.f>=0)hi=eval(std::min(F(82),best.t+d));
    }
    if(lo.f>0 && hi.f<0 && lo.t<hi.t) {
        Eval cur=eval((lo.t+hi.t)/2);
        for(int k=0;k<55;++k) {
            if(set(cur,3e-13))return true;
            if(cur.f>0)lo=cur;else hi=cur;
            if(hi.t-lo.t<2e-14)break;
            F candidate=cur.fp<0 ? cur.t-cur.f/cur.fp : F(1e6);
            cur=eval(candidate>lo.t && candidate<hi.t?candidate:(lo.t+hi.t)/2);
        }
        if(set(cur,8e-10))return true;
    }
    return set(best,8e-10);
}

inline void soc_project(double* a,int d) {
    double nrm=0.;
    for(int j=1;j<d;++j)nrm+=a[j]*a[j];
    nrm=std::sqrt(nrm);
    if(nrm<=a[0])return;
    if(nrm<=-a[0]){std::fill(a,a+d,0.);return;}
    double w=.5*(a[0]+nrm);
    double mul=w/nrm;
    for(int j=1;j<d;++j)a[j]*=mul;
    a[0]=w;
}

struct ConeProjector {
    const Problem& p;
    Vec mat,eig,work;
    Int iw;
    Vec warm;
    explicit ConeProjector(const Problem& prob):p(prob),warm(p.cones.size(),inf) {}
    bool psd_projection(int d,double* z) {
        if(d==1){z[0]=std::max(0.,z[0]);return true;}
        if(d==2) {
            double x[3]={invroot2*(z[0]+z[2]),invroot2*(z[0]-z[2]),z[1]};
            soc_project(x,3);
            z[0]=invroot2*(x[0]+x[1]);z[2]=invroot2*(x[0]-x[1]);z[1]=x[2];
            return true;
        }
        mat.resize(2*size_t(d)*d);eig.resize(d);
        int lw=1+6*d+2*d*d, liw=3+5*d;
        work.resize(lw);iw.resize(liw);
        const double sc=std::max(1e-100,norm(z,d*(d+1)/2));
        smat(d,z,mat.data(),1./sc);
        if(chol(d,mat.data()))return true;
        smat(d,z,mat.data(),1./sc);
        int info=0;
        scipy_dsyevd_("V","L",&d,mat.data(),&d,eig.data(),work.data(),&lw,iw.data(),&liw,&info);
        if(info)return false;
        int first=0;
        while(first<d && eig[first]<=0.)++first;
        if(first==d)std::fill(z,z+d*(d+1)/2,0.);
        else {
            for(int j=first;j<d;++j) {
                double scale=std::sqrt(eig[j]);
                for(int i=0;i<d;++i)mat[i+j*d]*=scale;
            }
            int k=d-first;
            const double one=1.,zero=0.;
            double* M=mat.data()+size_t(d)*d;
            scipy_dsyrk_("L","N",&d,&k,&one,mat.data()+first*d,&d,&zero,M,&d);
            symmetrize_lower(M,d);
            svec(d,M,z,sc);
        }
        return true;
    }
    bool project(Vec& x,bool dual=true) {
        for(size_t j=0;j<p.cones.size();++j) {
            const auto c=p.cones[j];
            double* a=x.data()+c.off;
            if(c.kind==Zero) {
                if(!dual)std::fill(a,a+c.width,0.);
            } else if(c.kind==Linear) {
                for(int i=0;i<c.width;++i)a[i]=std::max(0.,a[i]);
            } else if(c.kind==Soc)soc_project(a,c.width);
            else if(c.kind==PSD) {
                if(!psd_projection(c.dim,a))return false;
            } else {
                double t[3]={a[0],a[1],a[2]};
                if(dual)for(double& v:t)v=-v;
                ExpProjection out;
                if(!project_exp(t,out,warm[j]))return false;
                if(out.rotated) {
                    warm[j]=out.basis[0]/out.basis[1];
                    for(int i=0;i<3;++i)t[i]=out.s[0]*out.basis[i];
                } else std::copy(out.s.begin(),out.s.end(),t);
                for(int i=0;i<3;++i)a[i]=dual?a[i]+t[i]:t[i];
            }
        }
        return true;
    }
};

} // namespace conic
