#pragma once
#include "presolve.hpp"
#include <memory>

extern "C" {
void* sealed_cut_open(int,int,const int*,const int*,const double*,const double*,const double*,
    const double*,const double*,const double*);
void sealed_cut_close(void*);
int sealed_cut_append(void*,int,const int*,const int*,const double*,const double*,const double*);
int sealed_cut_bounds(void*,int,const double*,const double*);
int sealed_cut_step(void*,int,int,double,double*,double*);
int sealed_highs_available();
}

namespace conic {

struct CutStack {
    const Problem& p;
    Int cp,ci;
    Vec cv, work;
    Int touched, mark;
    Int mp{0},mi;
    Vec mv;
    int stamp=0;
    Int ap{0},ai;
    Vec av,lower,upper;

    explicit CutStack(const Problem& pr):p(pr),cp(p.m()+1,0),ci(p.A.nz()),cv(p.A.nz()),
        work(p.n()),mark(p.n(),0) {
        for(int a=0;a<p.A.nz();++a)++cp[p.A.i[a]+1];
        for(int k=0;k<p.m();++k)cp[k+1]+=cp[k];
        Int pos=cp;
        for(int j=0;j<p.n();++j)for(int i=p.A.p[j];i<p.A.p[j+1];++i) {
            int a=pos[p.A.i[i]]++;
            ci[a]=j;cv[a]=p.A.v[i];
        }
    }
    void insert(const Int& indices,const Vec& coeff,bool eq=false) {
        double rhs=0.;
        ++stamp;
        touched.clear();
        for(size_t j=0;j<indices.size();++j) {
            const int i=indices[j];const double c=coeff[j];
            if(c==0.)continue;
            rhs+=p.b[i]*c;
            mi.push_back(i);mv.push_back(c);
            for(int k=cp[i];k<cp[i+1];++k) {
                const int a=ci[k];
                if(mark[a]!=stamp){mark[a]=stamp;work[a]=0.;touched.push_back(a);}
                work[a]+=c*cv[k];
            }
        }
        mp.push_back(int(mi.size()));
        std::sort(touched.begin(),touched.end());
        for(int j:touched)if(work[j]!=0.) {ai.push_back(j);av.push_back(work[j]);}
        ap.push_back(int(ai.size()));
        lower.push_back(eq?rhs:-inf);upper.push_back(rhs);
    }
    void insert(int offset,const Vec& v) {
        Int i(v.size());
        std::iota(i.begin(),i.end(),offset);
        insert(i,v);
    }
    void direction(int off,double t) {
        t=std::clamp(t,-635.,635.);
        double w=std::exp(-t), d=std::max({1.,std::abs(t-1.),w});
        insert(off,{-1./d,(t-1.)/d,w/d});
    }
    void clear_batch() {
        ap.assign(1,0);ai.clear();av.clear();lower.clear();upper.clear();
    }
    void lift(const Vec& lpz,Vec& z) const {
        std::fill(z.begin(),z.end(),0.);
        for(int j=0;j<(int)lpz.size();++j)for(int k=mp[j];k<mp[j+1];++k)
            z[mi[k]]+=lpz[j]*mv[k];
    }
};

template<class Accept>
bool outer_native(const Problem& p,Result& r,double until,Accept&& accept) {
    const int n=p.n(),m=p.m();
    if(n==0 || n+m>57000 || p.A.nz()>620000 || p.P.nz()!=0 ||
       std::any_of(p.cones.begin(),p.cones.end(),[](auto c){return c.kind==PSD && c.dim>24;}) ||
       !sealed_highs_available() || clock_time()>until)return false;
    CutStack rows(p);
    Vec lower(n,-inf),upper(n,inf),lpz;
    for(auto c:p.cones) {
        if(c.kind<=Linear) {
            for(int j=0;j<c.width;++j)rows.insert({c.off+j},{1.},c.kind==Zero);
        } else if(c.kind==Soc) {
            rows.insert({c.off},{1.});
            const int count=c.dim<=22?c.width:1;
            for(int j=1;j<count;++j) {
                rows.insert({c.off,c.off+j},{1.,1.});
                rows.insert({c.off,c.off+j},{1.,-1.});
            }
        } else if(c.kind==Exp) {
            rows.insert({c.off+1},{1.});rows.insert({c.off+2},{1.});
            for(double t:{-8.,-3.,0.,3.,8.})rows.direction(c.off,t);
            if(p.b[c.off+1]>0. && p.b[c.off+2]>0.)
                rows.direction(c.off,std::log(p.b[c.off+2])-std::log(p.b[c.off+1]));
        } else {
            for(int j=0;j<c.dim;++j)rows.insert({c.off+j*(j+3)/2},{1.});
        }
        if(clock_time()>=until || rows.ai.size()>2000000)return false;
    }
    int nr=int(rows.upper.size());
    std::unique_ptr<void,decltype(&sealed_cut_close)> session(sealed_cut_open(n,nr,
        rows.ap.data(),rows.ai.data(),rows.av.data(),p.q,lower.data(),upper.data(),
        rows.lower.data(),rows.upper.data()),sealed_cut_close);
    if(!session)return false;
    rows.clear_batch();
    Vec normal,mat,eig,work;
    Int iw;
    bool boxed=false, recovered=false;
    double limit=std::max(1000.,2e4*norm(p.b,m));
    auto box=[&](double l){
        for(int j=0;j<n;++j) {
            int nz=p.A.p[j+1]-p.A.p[j];
            double sc=nz?norm(p.A.v+p.A.p[j],nz):0.;
            upper[j]=l==inf?inf:std::min(1e23,l/std::max(1e-6,sc));lower[j]=-upper[j];
        }
        return sealed_cut_bounds(session.get(),n,lower.data(),upper.data())>=0;
    };
    for(int it=0;it<100 && clock_time()+.012<until;++it) {
        lpz.resize(nr);
        int code=sealed_cut_step(session.get(),n,nr,until-clock_time(),r.x.data(),lpz.data());
        if(code==2) {
            rows.lift(lpz,r.z);
            if(accept(PrimalInfeasible))return true;
            if(boxed && !recovered) {
                recovered=true;boxed=false;
                if(!box(inf))return false;
                continue;
            }
            for(double& z:r.z)z=-z;
            return accept(PrimalInfeasible);
        }
        if(code==4 && boxed && !recovered) {
            recovered=true;boxed=false;
            if(!box(inf))return false;
            continue;
        }
        if(code==5 && !boxed && !recovered) {
            boxed=box(limit);
            if(!boxed)return false;
            continue;
        }
        if(code!=1 && code!=3)return false;
        if(code==3 && accept(DualInfeasible))return true;
        if(code==1)std::copy(p.b,p.b+m,r.s.begin());
        else std::fill(r.s.begin(),r.s.end(),0.);
        p.A.add(r.x.data(),r.s.data(),-1.);
        rows.clear_batch();
        double worst=0.;
        for(auto c:p.cones) {
            double* a=r.s.data()+c.off;
            if(c.kind==Zero) {
                if(code==1)std::fill(a,a+c.width,0.);
                continue;
            }
            if(c.kind==Linear) {
                if(code==1)for(int j=0;j<c.width;++j)a[j]=std::max(0.,a[j]);
                continue;
            }
            const double scale=std::max(1.,norm(a,c.width));
            double err=0.;
            if(c.kind==Soc) {
                for(int j=1;j<c.width;++j) {double t=a[j]/scale;err+=t*t;}
                err=std::sqrt(err)-a[0]/scale;
            } else if(c.kind==Exp)err=exp_defect(a[0]/scale,a[1]/scale,a[2]/scale);
            else {
                mat.resize(size_t(c.dim)*c.dim);
                smat(c.dim,a,mat.data(),1./scale);
                err=-smallest_eigen(c.dim,mat.data(),work,iw,eig);
            }
            worst=std::max(err,worst);
            if(err>3e-9) {
                normal.assign(c.width,0.);
                if(constant_separator(c,a,normal.data()))rows.insert(c.off,normal);
            }
        }
        if(code==1 && worst<9.8e-9) {
            rows.lift(lpz,r.z);
            if(accept(Solved))return true;
            if(boxed) {
                double used=0.;
                for(int j=0;j<n;++j)used=std::max(used,std::abs(r.x[j])/upper[j]);
                if(used>.98) {
                    if(accept(DualInfeasible))return true;
                    limit*=30.;
                    if(limit>1e22)return false;
                    if(!box(limit))return false;
                    rows.mp.resize(nr+1);rows.mi.resize(rows.mp.back());rows.mv.resize(rows.mp.back());
                    continue;
                }
            }
        }
        const int count=int(rows.upper.size());
        if(count==0 || size_t(nr)+count>170000 || rows.mi.size()>4500000 ||
           sealed_cut_append(session.get(),count,rows.ap.data(),rows.ai.data(),rows.av.data(),
               rows.lower.data(),rows.upper.data())<0)return false;
        nr+=count;
    }
    return false;
}

} // namespace conic
