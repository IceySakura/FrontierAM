#include "kkt.hpp"
#include "optimizer.hpp"

#include <cstdio>

namespace {
enum Activity {Linear, SocSurface, ExpGraph, ExpLog};
struct Active {
    Activity kind;
    int start,dim;
    bool positive;
    double lambda;
};

struct LocalMatrix {
    const Sparse& a;
    Sparse rows;
    Vec values;
    Ind marks,used;
    int mark=0;
    LocalMatrix(const Sparse& csr):a(csr),rows(a.rows,0),values(a.rows),marks(a.rows,-1){}
    void start(){++mark;used.clear();}
    void term(int i,double u){
        if(!u)return;
        for(int k=a.p[i];k<a.p[i+1];++k){
            int j=a.i[k];
            if(marks[j]!=mark){marks[j]=mark;values[j]=0.;used.push_back(j);}
            values[j]+=u*a.x[k];
        }
    }
    double finish(){
        std::sort(used.begin(),used.end());
        double a=0.;
        for(int j:used) a=std::max(a,std::abs(values[j]));
        // Keep each transformed equation balanced.
        a=std::clamp(a,1e-6,1e6);
        for(int j:used) if(values[j]){
            rows.i.push_back(j);
            rows.x.push_back(values[j]/a);
        }
        rows.p.push_back(rows.nnz());
        ++rows.cols;
        return a;
    }
};

double nrm2(const double* x,int n){
    double d=0;
    for(int i=0;i<n;++i)d+=x[i]*x[i];
    return std::sqrt(d);
}
double eval(const Active& a,const Vec& s,Vec& grad){
    const double* x=s.data()+a.start;
    grad.resize(a.dim);
    if(a.kind==Linear){
        grad[0]=1;return -x[0];
    }
    if(a.kind==SocSurface){
        double d=nrm2(x+1,a.dim-1);
        if(!(d>1e-22)) return INFINITY;
        grad[0]=1;
        for(int j=1;j<a.dim;++j)grad[j]=-x[j]/d;
        return d-x[0];
    }
    if(!(x[1]>1e-25)) return INFINITY;
    if(a.kind==ExpGraph){
        double r=x[0]/x[1],e=std::exp(r);
        if(!std::isfinite(e))return INFINITY;
        grad[0]=-e;grad[1]=e*(r-1);grad[2]=1;
        return x[1]*e-x[2];
    }
    if(!(x[2]>1e-25))return INFINITY;
    double r=std::log(x[2])-std::log(x[1]);
    grad[0]=-1;grad[1]=r-1;grad[2]=x[1]/x[2];
    return x[0]-r*x[1];
}

std::vector<Active> active_set(const Model& p,const Vec& s,const Vec& z,double etol){
    std::vector<Active> list;
    auto lin=[&](int j,bool bound){list.push_back({Linear,j,1,bound,z[j]});};
    int off=0;
    for(auto c:p.cones) {
        int d=c.width();
        const double* x=s.data()+off,*y=z.data()+off;
        if(c.kind==Zero){
            for(int j=0;j<d;++j)lin(off+j,false);
        } else if(c.kind==Orthant){
            for(int j=0;j<d;++j) {
                double u=1+std::abs(p.b[off+j]);
                if((x[j]<etol*u && (y[j]>etol*.005 || x[j]<etol*.006*u)) || x[j]<-1e-9*u)
                    lin(off+j,true);
            }
        } else if(c.kind==Lorentz) {
            double u=nrm2(x+1,d-1), ys=nrm2(y,d);
            if(std::max(std::abs(x[0]),u)<etol*.25 && (ys>etol*.012 || x[0]<0)){
                for(int j=0;j<d;++j)lin(off+j,false);
            } else if(u>1e-12 && u-x[0]>=-etol*(1+u) && (y[0]>etol*.002 || u>x[0]+1e-10)){
                double v=y[0];
                for(int j=1;j<d;++j)v-=y[j]*(x[j]/u);
                list.push_back({SocSurface,off,d,true,std::max(0.,v*.5)});
            }
        } else if(c.kind==Exponential) {
            double xs=std::max({std::abs(x[0]),std::abs(x[1]),std::abs(x[2])}),ys=nrm2(y,3);
            if(xs<etol*.06 && ys>etol*.06){
                for(int j=0;j<3;++j)lin(off+j,false);
            } else if(x[1]<=etol*.025 && x[0]<-etol){
                if(y[1]>etol*.002 || x[1]<0)lin(off+1,true);
                if(x[2]<etol && (y[2]>etol*.002 || x[2]<0))lin(off+2,true);
            } else if(x[1]>1e-20){
                double r=x[0]/x[1];
                if(r<1.5){
                    double h=std::exp(r);
                    if(x[1]*h-x[2]>=-etol*(1+std::abs(x[2])) &&
                       (y[2]>etol*.0005 || x[1]*h>x[2]+1e-10)){
                        double b=h*(r-1);
                        double l=(y[2]-h*y[0]+b*y[1])/(1+h*h+b*b);
                        list.push_back({ExpGraph,off,3,true,std::max(0.,l)});
                    }
                } else if(x[2]>0){
                    double h=std::log(x[2])-std::log(x[1]);
                    if(x[0]-h*x[1]>=-etol*(1+xs) && (ys>etol*.001 || x[0]-h*x[1]>1e-10))
                        list.push_back({ExpLog,off,3,true,std::max(0.,-y[0])});
                } else {
                    list.push_back({ExpGraph,off,3,true,std::max(0.,y[2])});
                }
            }
        }
        off+=d;
    }
    return list;
}

struct Values {
    Vec s,z,grad,curv;
    Values(int n,int m,int d):s(m),z(m),grad(n),curv(d){}
    double merit=INFINITY;
    bool get(const Model& p,const Vec& x,const std::vector<Active>& list,Vec& tmp) {
        p.A.mul(x.data(),s.data());
        for(int j=0;j<p.m();++j)s[j]=p.b[j]-s[j];
        std::fill(z.begin(),z.end(),0.);
        double m=0;
        for(size_t i=0;i<list.size();++i){
            const auto& a=list[i];double v=eval(a,s,tmp);
            if(!std::isfinite(v)||!std::isfinite(a.lambda))return false;
            curv[i]=v;
            m=std::max(m,std::abs(v)/(1+std::abs(p.b[a.start])));
            for(int j=0;j<a.dim;++j)z[a.start+j]+=a.lambda*tmp[j];
        }
        p.A.tmul(z.data(),grad.data());
        Vec px=p.P.mul(x);
        for(int j=0;j<p.n();++j)grad[j]+=px[j]+p.q[j];
        merit=std::max(m,norm(grad)/(1+fastnorm(p.q)+fastnorm(px)));
        return std::isfinite(merit);
    }
};
}

bool polish_manifold(const Model& p,const Vec& D,const Vec& E,double gamma,Verifier& v) {
    if(!v.near || p.n()>15000 || p.m()>72000 || p.A.nnz()+p.P.nnz()>380000 || p.n()==0) return false;
    for(auto c:p.cones) if(c.kind==Semidefinite || (c.kind==Lorentz && c.dim>40))return false;
    auto input=v.near;
    Sparse csr=p.A.transpose();
    Vec xstart=input.x,zstart=input.z;
    for(int j=0;j<p.n();++j)xstart[j]/=D[j];
    for(int j=0;j<p.m();++j)zstart[j]*=gamma/E[j];
    Vec sstart=p.A.mul(xstart);
    for(int j=0;j<p.m();++j)sstart[j]=p.b[j]-sstart[j];
    auto check=[&](const Vec& x,const Vec& s,const Vec& z){
        Vec xx(p.n()),ss(p.m()),zz(p.m());
        for(int j=0;j<p.n();++j)xx[j]=D[j]*x[j];
        for(int j=0;j<p.m();++j){ss[j]=s[j]/E[j];zz[j]=z[j]*E[j]/gamma;}
        return v.solved(xx,ss,zz);
    };
    for(double threshold:{4e-5, 5e-7, 1e-3}) {
        Vec x=xstart;
        auto l=active_set(p,sstart,zstart,threshold);
        if(l.empty())continue;
        Values here(p.n(),p.m(),l.size()),test(p.n(),p.m(),l.size());
        Vec gp,sc(l.size()),rhs;
        auto next=l;
        for(int it=0;it<5;++it){
            if(!here.get(p,x,l,gp))break;
            if(check(x,here.s,here.z))return true;
            LocalMatrix f(csr);
            rhs.clear();
            for(size_t i=0;i<l.size();++i){
                const auto& a=l[i];
                eval(a,here.s,gp);
                f.start();
                for(int j=0;j<a.dim;++j)f.term(a.start+j,gp[j]);
                sc[i]=f.finish();
                rhs.push_back(-here.curv[i]/sc[i]);
            }
            Vec aug;
            for(const auto& a:l) if(a.kind!=Linear && a.lambda>1e-18){
                int i=a.start;
                const double* s=here.s.data()+i;
                double ua=a.lambda;
                if(a.kind==SocSurface){
                    double r=nrm2(s+1,a.dim-1);
                    if(r<=1e-18)continue;
                    for(int j=1;j<a.dim;++j){
                        f.start();
                        double u=s[j]/r;
                        for(int k=1;k<a.dim;++k)
                            f.term(i+k,(j==k?1.:0.)-u*(s[k]/r));
                        double t=f.finish();
                        aug.push_back(std::clamp(r/(ua*t*t),1e-15,1e15));
                        rhs.push_back(0);
                    }
                } else if(a.kind==ExpGraph) {
                    double r=s[0]/s[1], h=std::exp(r),sc=std::max(1.,std::abs(r));
                    f.start(); f.term(i,1/sc);f.term(i+1,-r/sc);
                    double t=f.finish();
                    aug.push_back(std::clamp((s[1]/(h*ua))/(sc*sc*t*t),1e-15,1e25));
                    rhs.push_back(0);
                } else {
                    double r=s[1]/s[2],sc=std::max(1.,r);
                    f.start(); f.term(i+1,1/sc);f.term(i+2,-r/sc);
                    double t=f.finish();
                    aug.push_back(std::clamp(s[1]/(ua*sc*sc*t*t),1e-15,1e25));
                    rhs.push_back(0);
                }
            }
            if(f.rows.nnz()>550000)break;
            Model u;
            u.A=f.rows.transpose();u.P=p.P;
            u.q.resize(p.n());u.b.resize(rhs.size());
            u.cones={{Zero,(int)l.size()}};
            if(!aug.empty())u.cones.push_back({Orthant,(int)aug.size()});
            ProductCone cc(u,Ind(u.cones.size(),0));
            for(size_t j=0;j<aug.size();++j)cc.diagonal[j+l.size()]=aug[j];
            Newton k(u,cc,0);
            const double reg=2e-10;
            if(!k.factor(reg))break;
            Vec fx(p.n()),dx(p.n()),dl(rhs.size()),tx=x;
            for(int j=0;j<p.n();++j)fx[j]=-here.grad[j];
            if(!k.solve(fx,rhs,dx,dl))break;
            if(norm(dx)>100*(1+norm(x)))break;
            bool accepted=false;
            for(double alpha:{1.,.5,.25,.125}){
                for(int j=0;j<p.n();++j)tx[j]=x[j]+alpha*dx[j];
                for(size_t j=0;j<l.size();++j){
                    next[j]=l[j];
                    next[j].lambda+=alpha*dl[j]/sc[j];
                }
                if(test.get(p,tx,next,gp) && test.merit<=here.merit*.99+2e-12){
                    if(debug)std::fprintf(stderr," polish active %zu residual %.2g -> %.2g\n",l.size(),here.merit,test.merit);
                    if(check(tx,test.s,test.z))return true;
                    x.swap(tx);l.swap(next);accepted=true;break;
                }
            }
            if(!accepted || test.merit<9e-13)break;
        }
    }
    return false;
}
