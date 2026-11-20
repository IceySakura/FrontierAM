#include "numeric.hpp"
#include "optimizer.hpp"

#include <array>
#include <chrono>
#include <cstdio>

namespace {
using Clock=std::chrono::steady_clock;
struct Weights {
    int pos;
    Vec factor;
};
struct Cuts {
    const Model& p;
    Sparse a;
    Sparse r;
    Vec low,high,temp;
    Ind touched,stamp;
    std::vector<Weights> comb;
    int serial=0;
    Cuts(const Model& p_) : p(p_), a(p.A.transpose()), r(p.n(),0), temp(p.n()), stamp(p.n(),-1){}
    void clear(){
        r.cols=0;r.p.assign(1,0);r.i.clear();r.x.clear();low.clear();high.clear();
    }
    void unit(int i,bool equal){
        r.i.insert(r.i.end(),a.i.begin()+a.p[i],a.i.begin()+a.p[i+1]);
        r.x.insert(r.x.end(),a.x.begin()+a.p[i],a.x.begin()+a.p[i+1]);
        r.p.push_back(r.nnz());++r.cols;
        high.push_back(p.b[i]);low.push_back(equal?p.b[i]:-INFINITY);
        comb.push_back({i,{1}});
    }
    bool add(int at,Vec h){
        touched.clear();++serial;
        double b=0;
        for(size_t j=0;j<h.size();++j) if(h[j]){
            double t=h[j];int r=at+j;b+=t*p.b[r];
            for(int k=a.p[r];k<a.p[r+1];++k){
                int i=a.i[k];
                if(stamp[i]!=serial){stamp[i]=serial;temp[i]=0;touched.push_back(i);}
                temp[i]+=t*a.x[k];
            }
        }
        if(!std::isfinite(b)) return false;
        double sc=0;
        for(int i:touched) sc=std::max(sc,std::abs(temp[i]));
        if(sc>1e-250 && sc<1e250) {
            for(double& z:h) z/=sc;
            b/=sc;
        } else sc=1;
        if(!std::isfinite(b))return false;
        if(touched.empty() && b>=0)return true;
        std::sort(touched.begin(),touched.end());
        for(int i:touched) if(temp[i]){r.i.push_back(i);r.x.push_back(temp[i]/sc);}
        r.p.push_back(r.nnz());++r.cols;
        high.push_back(b);low.push_back(-INFINITY);
        comb.push_back({at,std::move(h)});
        return true;
    }
    Vec dual(const Vec& raw) const {
        Vec z(p.m());
        for(size_t i=0;i<raw.size() && i<comb.size();++i){
            const auto& w=comb[i];double a=raw[i];
            if(a)for(size_t j=0;j<w.factor.size();++j)
                z[w.pos+j]+=a*w.factor[j];
        }
        return z;
    }
};

struct Curved {
    int at,d,kind;
    Vec slopes;
};
void exp_cut(Cuts& c,Curved& cone,double r) {
    r=std::clamp(r,-175.,175.);
    for(double t:cone.slopes) if(std::abs(r-t)<1e-11*(1+std::abs(t))) return;
    if(r<0) {
        double a=std::exp(r);
        c.add(cone.at,{-a,(r-1)*a,1});
    } else c.add(cone.at,{-1,r-1,std::exp(-r)});
    cone.slopes.push_back(r);
}
double length(const double* u,int n) {
    double a=0;
    for(int i=0;i<n;++i)a=std::hypot(a,u[i]);
    return a;
}
double exp_err(const double* s){
    double sc=std::max({1.,std::abs(s[0]),std::abs(s[1]),std::abs(s[2])});
    double b=std::max({s[0],std::abs(s[1]),-s[2]});
    if(s[1]>0 && s[2]>0) b=std::min(b,std::max(0.,s[0]-s[1]*(std::log(s[2])-std::log(s[1]))));
    return b/sc;
}
}

bool supporting_planes(const Model& p,Verifier& verify) {
    constexpr double pass=7.0e-9;
    if(p.n()==0 || p.A.nnz()>650'000 || p.m()>90'000) return false;
    std::vector<Curved> curved;
    int at=0,exp=0,soc=0;
    for(auto c:p.cones){
        if(c.kind==Semidefinite || (c.kind==Lorentz && c.dim>24))return false;
        if(c.kind>=Lorentz) {
            exp+=c.kind==Exponential;soc+=c.kind==Lorentz;
            curved.push_back({at,c.dim,c.kind,{}});
        }
        at+=c.width();
    }
    if(!exp && (!soc || p.n()+p.m()>25000 || !verify.near))return false;
    if(p.P.nnz() && p.n()+p.m()>4000) return false;
    auto start=Clock::now();
    HiGHS h;
    if(!h)return false;
    Cuts cut(p);
    const auto& near=verify.near;
    at=0;
    for(auto k:p.cones) {
        if(k.kind<=Orthant) for(int j=0;j<k.dim;++j)cut.unit(at+j,k.kind==Zero);
        at+=k.width();
    }
    for(auto& c:curved){
        if(c.kind==Exponential) {
            cut.add(c.at,{0.,1.,0.});cut.add(c.at,{0.,0.,1.});
            const double* s=near?near.s.data()+c.at:nullptr;
            if(s && s[1]>0 && s[2]>0){
                double u=std::log(s[2])-std::log(s[1]);
                for(double r:{u,u-.35,u+.35,-6.,0.,6.}) exp_cut(cut,c,r);
            } else for(double t:{-8.,-4.,-2.,0.,2.,4.,8.}) exp_cut(cut,c,t);
        } else {
            Vec u(c.d);u[0]=1;cut.add(c.at,u);
            if(near) {
                auto z=near.s.data()+c.at;
                double n=length(z+1,c.d-1);
                if(n>0){
                    for(int j=1;j<c.d;++j)u[j]=-z[j]/n;
                    cut.add(c.at,u);
                }
            }
            std::fill(u.begin()+1,u.end(),0.);
            for(int j=1;j<c.d;++j) {
                u[j]=1;cut.add(c.at,u);u[j]=-1;cut.add(c.at,u);u[j]=0;
            }
        }
    }
    Vec lc(p.n(),-INFINITY),hc(p.n(),INFINITY),x,raw,z,s(p.m());
    bool qp=p.P.nnz();
    auto st=h.solve(cut.r.transpose(),p.q,lc,hc,cut.low,cut.high,qp?&p.P:nullptr,
                    "choose",true,3.,1e-10);
    int finish=verify.best<1e-5?22:26;
    for(int it=0;it<finish;++it) {
        bool ray=st==LpUnbounded;
        if(debug)std::fprintf(stderr," supporting planes %d status %d rows %zu\n",it,st,cut.comb.size());
        if(st==LpInfeasible || st==LpAmbiguous){
            raw.resize(cut.comb.size());
            if(h.ray(raw,false)){
                auto zz=cut.dual(raw);
                if(verify.primal(zz))return true;
                for(double& v:zz)v=-v;
                if(verify.primal(zz))return true;
            }
            break;
        }
        if(ray){
            x.resize(p.n());
            if(!h.ray(x,true))break;
            if(verify.dual(x))return true;
        } else {
            if(!h.point(x,raw) || x.size()!=p.q.size() || raw.size()!=cut.comb.size()) break;
        }
        p.A.mul(x.data(),s.data());
        for(int j=0;j<p.m();++j)s[j]=(ray?0:p.b[j])-s[j];
        if(!ray){
            z=cut.dual(raw);
            if(verify.solved(x,s,z))return true;
        }
        if(it>=4 && std::chrono::duration<double>(Clock::now()-start).count()>2.0)break;
        cut.clear();
        double peak=0;
        for(auto& c:curved) {
            const double* t=s.data()+c.at;
            if(c.kind==Exponential) {
                double err=exp_err(t);peak=std::max(peak,err);
                if(err<=pass && !ray) continue;
                double sc=std::max({1.,std::abs(t[0]),std::abs(t[1]),std::abs(t[2])});
                if(t[1]>0) {
                    double u=t[2]>0?std::log(t[2])-std::log(t[1]):std::log(1e-13*sc)-std::log(t[1]);
                    exp_cut(cut,c,u);
                    if(ray || it==0 || (it>5 && err>2.0e-8)){
                        double d=ray?.75:std::clamp(std::sqrt((2*std::max(pass,err)*sc)/t[1]),.00005,.65);
                        exp_cut(cut,c,u-d);exp_cut(cut,c,u+d);
                    }
                } else if(t[0]>0) exp_cut(cut,c,std::min(110.,13.+it*6.));
            } else {
                double n=length(t+1,c.d-1),sc=std::max(1.,std::abs(t[0]));
                for(int j=1;j<c.d;++j)sc=std::max(sc,std::abs(t[j]));
                double e=(n-t[0])/sc;peak=std::max(peak,e);
                if(e>pass && n>0) {
                    Vec u(c.d);u[0]=1;
                    for(int j=1;j<c.d;++j)u[j]=-t[j]/n;
                    cut.add(c.at,std::move(u));
                }
            }
        }
        if(debug)std::fprintf(stderr,"  support violation %.3g, add %d\n",peak,cut.r.cols);
        if(cut.r.cols==0 || !h.append_rows(cut.r,cut.low,cut.high))break;
        st=h.resolve();
    }
    return false;
}
