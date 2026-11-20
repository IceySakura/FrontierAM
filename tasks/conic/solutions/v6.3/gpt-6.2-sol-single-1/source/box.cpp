#include "optimizer.hpp"

#include <chrono>
#include <cstdio>

namespace {
using Clock=std::chrono::steady_clock;
bool linear(const Model& p){
    for(auto c:p.cones)if(c.kind>Orthant)return false;
    return true;
}
}

bool projection_form(const Model& p,Verifier& v) {
    int n=p.n();
    if(!n || n!=p.m() || p.A.nnz()!=n || p.P.nnz()>n || linear(p))return false;
    Vec h(n),q(n),x(n),z(n),value(n),coef(n);
    Ind var(n,-1);
    const bool quad=p.P.nnz();
    for(int j=0;j<n;++j){
        int c=p.A.p[j];
        if(p.A.p[j+1]!=c+1)return false;
        int i=p.A.i[c];
        if(var[i]>=0)return false;
        var[i]=j;coef[i]=p.A.x[c];
        double a=coef[i];
        q[i]=-p.q[j]/a;
        if(quad){
            int k=p.P.p[j];
            if(p.P.p[j+1]!=k+1 || p.P.i[k]!=j || p.P.x[k]<=0)return false;
            h[i]=(p.P.x[k]/a)/a;
            if(!(h[i]>0) || !std::isfinite(h[i]))return false;
        }
        if(!std::isfinite(q[i]))return false;
        value[i]=quad?p.b[i]-q[i]/h[i]:0;
    }
    if(!quad){
        for(int i=0;i<n;++i)x[var[i]]=p.b[i]/coef[i];
        if(v.solved(x,value,q,false))return true;
        for(int i=0;i<n;++i)value[i]=-q[i];
        v.cones().project(value,false);
        for(int i=0;i<n;++i)x[var[i]]=-value[i]/coef[i];
        return v.dual(x);
    }
    int off=0;
    for(auto c:p.cones){
        int next=off+c.width();
        if(c.kind>=Lorentz){
            double b=h[off];
            for(int j=off+1;j<next;++j)if(std::abs(h[j]/b-1)>2e-13)return false;
        }
        off=next;
    }
    if(!std::isfinite(norm(value)))return false;
    Vec s=value;
    v.cones().project_pair(s,z);
    for(int i=0;i<n;++i){
        x[var[i]]=(p.b[i]-s[i])/coef[i];
        z[i]*=h[i];
    }
    return v.solved(x,s,z,false);
}

bool box_quadratic(const Model& p,Verifier& v) {
    int n=p.n(),m=p.m();
    if(n<300 || p.P.nnz()<=n || p.A.nnz()>m || !linear(p))return false;
    Vec coef(m),lo(n,-INFINITY),hi(n,INFINITY);
    Ind row(m,-1),il(n,-1),ih(n,-1);
    for(int j=0;j<n;++j)for(int k=p.A.p[j];k<p.A.p[j+1];++k){
        int i=p.A.i[k];
        if(row[i]>=0)return false;
        row[i]=j;coef[i]=p.A.x[k];
    }
    int at=0;
    for(auto c:p.cones)for(int e=at+c.dim;at<e;++at){
        int j=row[at];
        if(j<0) {
            if(c.kind==Zero?p.b[at]!=0:p.b[at]<0)return false;
            continue;
        }
        double a=coef[at],r=p.b[at]/a;
        if(!std::isfinite(r))return false;
        if((a<0 || c.kind==Zero) && r>lo[j]){lo[j]=r;il[j]=at;}
        if((a>0 || c.kind==Zero) && r<hi[j]){hi[j]=r;ih[j]=at;}
    }
    for(int j=0;j<n;++j)if(lo[j]>hi[j]){
        Vec z(m);
        z[il[j]]=coef[ih[j]];z[ih[j]]=-coef[il[j]];
        if(v.primal(z))return true;
        z[il[j]]=-z[il[j]];z[ih[j]]=-z[ih[j]];
        return v.primal(z);
    }
    auto start=Clock::now();
    Sparse P=p.P;
    Vec D(n),Q(p.q),diag(n),u(n),L=lo,H=hi,cold(n);
    for(int j=0;j<n;++j){
        double t=0,dg=0;
        for(int k=P.p[j];k<P.p[j+1];++k){
            t=std::max(t,std::abs(P.x[k]));
            if(P.i[k]==j)dg=P.x[k];
        }
        if(dg<0)return false;
        double s=D[j]=1/std::sqrt(std::clamp(dg>0?dg:t,1e-20,1e20));
        diag[j]=std::max(dg*s*s,1e-14);
        Q[j]*=s;L[j]/=s;H[j]/=s;
        cold[j]=std::clamp(-Q[j]/diag[j],L[j],H[j]);
        if(!std::isfinite(cold[j]))cold[j]=std::clamp(0.,L[j],H[j]);
    }
    P.scale(D,&D);
    double lipschitz=0;
    bool tridiagonal=true;
    Vec lower(n);
    for(int j=0;j<n;++j){
        double s=0;
        for(int k=P.p[j];k<P.p[j+1];++k){
            s+=std::abs(P.x[k]);
            int i=P.i[k];
            if(std::abs(i-j)>1)tridiagonal=false;
            if(i==j+1)lower[i]=P.x[k];
        }
        lipschitz=std::max(lipschitz,s);
    }
    if(!(lipschitz>0) || !std::isfinite(lipschitz))return false;
    auto recover=[&](const Vec& y){
        Vec x(n),z(m);
        for(int j=0;j<n;++j)x[j]=y[j]<=L[j]?lo[j]:y[j]>=H[j]?hi[j]:D[j]*y[j];
        auto g=p.P.mul(x);
        for(int j=0;j<n;++j){
            g[j]+=p.q[j];
            if(g[j]>0 && il[j]>=0 && y[j]<=L[j])z[il[j]]=-g[j]/coef[il[j]];
            if(g[j]<0 && ih[j]>=0 && y[j]>=H[j])z[ih[j]]=-g[j]/coef[ih[j]];
        }
        auto s=p.A.mul(x);
        for(int j=0;j<m;++j)s[j]=p.b[j]-s[j];
        return v.solved(x,s,z);
    };
    if(n<25000 && recover(cold))return true;
    Vec x=cold,y=x,g(n),d(n),res(n),cg(n),hv(n),tx(n),gv(n);
    std::vector<uint8_t> use(n,1);
    const double qn=fastnorm(Q),step=1/lipschitz;
    double momentum=1.;
    int iters=n<160000?360:100;
    for(int it=0;it<iters;++it) {
        P.tmul(y.data(),g.data());
        double ns=0,ds=0;
        for(int j=0;j<n;++j){
            double u=std::clamp(y[j]-step*(g[j]+Q[j]),L[j],H[j]);
            ns=std::max(ns,std::abs(u-y[j]));
            ds+=(u-x[j])*(y[j]-u);
            tx[j]=u;
        }
        if(ns<(2e-7*step)*(1+qn) && recover(tx))return true;
        if((tridiagonal && it<5) || it==3 || it%18==17 || (ns<.003 && it%4==2)) {
            P.tmul(tx.data(),gv.data());
            double rz=0,pt=0;
            for(int j=0;j<n;++j){
                double z=gv[j]+Q[j];gv[j]=z;
                use[j]=!(L[j]==H[j] || (tx[j]<=L[j] && z>0) || (tx[j]>=H[j] && z<0));
                res[j]=use[j]?-z:0.;
                cg[j]=res[j]/diag[j];rz+=res[j]*cg[j];
                d[j]=0;pt=std::max(pt,std::abs(res[j]));
            }
            double target=std::max(4e-12*(1+qn),pt*std::clamp(pt*.1,1e-5,.02));
            int maxcg=n>=350000?28:n>=50000?42:100;
            if(tridiagonal) {
                for(int j=0;j<n;++j) {
                    if(!use[j]){hv[j]=1.;gv[j]=0;continue;}
                    double a=j && use[j-1]?lower[j]/hv[j-1]:0;
                    hv[j]=std::max(1e-12,diag[j]-a*lower[j]);
                    cg[j]=res[j]-(j?a*cg[j-1]:0);
                }
                for(int j=n-1;j>=0;--j)if(use[j])
                    d[j]=(cg[j]-(j+1<n?lower[j+1]*d[j+1]:0))/hv[j];
                // Nonworking gradients still enter the projected line search.
                P.tmul(tx.data(),gv.data());
                for(int j=0;j<n;++j)gv[j]+=Q[j];
            } else for(int k=0;k<maxcg && rz>0;++k){
                P.tmul(cg.data(),hv.data());
                double sq=dot(cg,hv);
                if(!(sq>1e-22*rz))break;
                double a=rz/sq,t=0,er=0;
                for(int j=0;j<n;++j)if(use[j]){
                    d[j]+=a*cg[j];
                    res[j]-=a*hv[j];t+=res[j]*res[j]/diag[j];er=std::max(er,std::abs(res[j]));
                }
                if(er<=target)break;
                double beta=t/rz;rz=t;
                for(int j=0;j<n;++j)if(use[j])cg[j]=res[j]/diag[j]+beta*cg[j];
            }
            bool move=false;
            for(double a:{1.,.5,.25,.125}){
                for(int j=0;j<n;++j) {u[j]=std::clamp(tx[j]+a*d[j],L[j],H[j]);res[j]=u[j]-tx[j];}
                P.tmul(res.data(),hv.data());
                double delta=.5*dot(res,hv)+dot(res,gv);
                if(delta<0){
                    tx.swap(u);move=true;break;
                }
            }
            if(move) {
                if(recover(tx))return true;
                x=tx;y=tx;momentum=1.;
                if(std::chrono::duration<double>(Clock::now()-start).count()>.5)break;
                continue;
            }
        }
        double next=(1+std::sqrt(1+4*momentum*momentum))*.5;
        if(ds>0) next=1.;
        double weight=next==1.?0.:(momentum-1)/next;
        for(int j=0;j<n;++j) y[j]=tx[j]+weight*(tx[j]-x[j]);
        x.swap(tx);momentum=next;
        if(it%4==3 && std::chrono::duration<double>(Clock::now()-start).count()>.52)break;
    }
    return recover(x);
}
