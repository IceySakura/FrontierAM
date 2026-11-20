#include "optimizer.hpp"
#include "kkt.hpp"
#include "numeric.hpp"
#include "reduce.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>

namespace {
constexpr double tolerance=9.15e-9;

Vec negative(const Vec& v) {
    Vec w(v.size());
    for(size_t j=0;j<v.size();++j) w[j]=-v[j];
    return w;
}
double dot_scaled(const Vec& a,double scale,const Vec& b) {
    double u=0;
    if(scale>1e-150 && scale<1e150){
        u=dot(a,b);
        if(std::isfinite(u)) return u/scale;
    }
    for(size_t j=0;j<a.size();++j) u+=(a[j]/scale)*b[j];
    return u;
}
Vec mul_scaled(const Sparse& a,const Vec& x,double scale,bool transposed=false) {
    if(scale>1e-150 && scale<1e150){
        auto r=transposed?a.tmul(x):a.mul(x);
        for(double& v:r) v/=scale;
        return r;
    }
    Vec r(transposed?a.cols:a.rows);
    if(transposed){
        for(int j=0;j<a.cols;++j)
            for(int k=a.p[j];k<a.p[j+1];++k) r[j]+=(a.x[k]/scale)*x[a.i[k]];
    } else {
        for(int j=0;j<a.cols;++j)
            for(int k=a.p[j];k<a.p[j+1];++k) r[a.i[k]]+=(a.x[k]/scale)*x[j];
    }
    return r;
}
Vec slack(const Model& p,const Vec& x) {
    auto s=p.A.mul(x);
    for(int j=0;j<p.m();++j) s[j]=p.b[j]-s[j];
    return s;
}
bool is_linear(const Model& p) {
    return std::all_of(p.cones.begin(),p.cones.end(),[](ConeSpec c){return c.kind<=Orthant;});
}

// Test diagonal, singleton, and constant rows before constructing a KKT pattern.
bool elementary(const Model& p,Verifier& v,bool linear){
    if(v.nQ==0 && v.cones().defect(p.b,false,tolerance)<=tolerance){
        if(v.solved(Vec(p.n()),p.b,Vec(p.m()),false)) return true;
    }
    int freecol=-1;
    if(v.nQ>0){
        double maxq=0;
        for(int j=0;j<p.n();++j)
            if(p.P.p[j]==p.P.p[j+1] && p.A.p[j]==p.A.p[j+1] && std::abs(p.q[j])>maxq){
                maxq=std::abs(p.q[j]); freecol=j;
            }
        if(freecol>=0){
            Vec x(p.n());x[freecol]=-std::copysign(1.0,p.q[freecol]);
            if(v.dual(x)) return true;
        }
    }
    if(p.A.nnz()>p.m()) return false;
    Ind var(p.m(),-1);
    Vec coeff(p.m());
    bool singular=true;
    for(int j=0;j<p.n();++j) for(int k=p.A.p[j];k<p.A.p[j+1];++k) {
        int i=p.A.i[k];
        if(var[i]!=-1) {var[i]=-2;singular=false;}
        else {var[i]=j;coeff[i]=p.A.x[k];}
    }
    int i=0;
    for(auto b:p.cones){
        int e=i+b.width();
        if(b.kind<=Orthant){
            for(;i<e;++i) if(var[i]==-1 && (b.kind==Zero?std::abs(p.b[i]):-p.b[i])>tolerance*v.nB) {
                Vec z(p.m());z[i]=-std::copysign(1.,p.b[i]);
                if(v.primal(z)) return true;
            }
        } else i=e;
    }
    if(!linear || !singular || p.P.nnz()>p.n()) return false;
    Vec px(p.n());
    for(int j=0;j<p.n();++j) for(int k=p.P.p[j];k<p.P.p[j+1];++k){
        if(p.P.i[k]!=j || p.P.x[k]<0) return false;
        px[j]=p.P.x[k];
    }
    Ind lr(p.n(),-1),hr(lr);
    Vec low(p.n(),-INFINITY),high(p.n(),INFINITY);
    i=0;
    for(auto b:p.cones) for(int e=i+b.dim;i<e;++i) if(var[i]>=0) {
        int j=var[i];
        double t=p.b[i]/coeff[i];
        if((b.kind==Zero || coeff[i]<0) && t>low[j]) {low[j]=t;lr[j]=i;}
        if((b.kind==Zero || coeff[i]>0) && t<high[j]) {high[j]=t;hr[j]=i;}
    }
    Vec x(p.n()),z(p.m());
    for(int j=0;j<p.n();++j) {
        if(low[j]>high[j]){
            if(lr[j]>=0 && hr[j]>=0) {
                z[lr[j]]=-coeff[hr[j]]; z[hr[j]]=coeff[lr[j]];
                if(v.primal(z)) return true;
                z[lr[j]]=-z[lr[j]];z[hr[j]]=-z[hr[j]];
                if(v.primal(z)) return true;
            }
            return false;
        }
        double u=px[j]>0 ? -p.q[j]/px[j] : p.q[j]>0 ? -INFINITY : p.q[j]<0 ? INFINITY : 0;
        x[j]=std::clamp(u,low[j],high[j]);
        if(!std::isfinite(x[j])){
            if(px[j]==0 && p.q[j]){
                x.assign(p.n(),0);x[j]=-std::copysign(1.,p.q[j]);
                return v.dual(x);
            }
            return false;
        }
        const double g=px[j]*x[j]+p.q[j];
        if(g>0 && lr[j]>=0) z[lr[j]]=-g/coeff[lr[j]];
        else if(g<0 && hr[j]>=0) z[hr[j]]=-g/coeff[hr[j]];
    }
    return v.solved(x,slack(p,x),z);
}

struct Equilibrated {
    Model p;
    Vec d,e;
    double gamma=1;
    Equilibrated(const Model& orig,ProductCone& c,int attempt) : p(orig),d(p.n(),1),e(p.m(),1) {
        if(p.P.nnz()) p.P=symmetric_part(orig.P);
        Vec cn(p.n()),rn(p.m()),pn(p.n());
        for(int it=0;it<(attempt?12:9);++it){
            std::fill(cn.begin(),cn.end(),0);
            std::fill(rn.begin(),rn.end(),0);
            p.A.norms(cn,&rn); p.P.norms(cn,nullptr);
            cc_rowscale(c.handle,rn.data(),attempt==1);
            double g=0;
            for(int j=0;j<p.n();++j) {
                double x=cn[j]>0 ? 1/std::sqrt(std::clamp(cn[j],1e-12,1e12)) : 1;
                cn[j]=std::clamp(x,1e-8/d[j],1e8/d[j]);d[j]*=cn[j];p.q[j]*=cn[j];
                g=std::max(g,std::abs(cn[j]-1));
            }
            for(int j=0;j<p.m();++j) {
                double x=rn[j]>0 ? 1/std::sqrt(std::clamp(rn[j],1e-12,1e12)) : 1;
                rn[j]=std::clamp(x,1e-8/e[j],1e8/e[j]);e[j]*=rn[j];p.b[j]*=rn[j];
                g=std::max(g,std::abs(rn[j]-1));
            }
            p.A.scale(cn,&rn);p.P.scale(cn,&cn);
            std::fill(pn.begin(),pn.end(),0);
            p.P.norms(pn,nullptr);
            double w=std::max(fastnorm(p.q),p.n()?std::accumulate(pn.begin(),pn.end(),0.)/p.n():0.);
            if(w>1e-18) {
                double a=std::clamp(1/w,1e-8/gamma,1e8/gamma);
                gamma*=a;
                for(double& x:p.P.x)x*=a;
                for(double& x:p.q)x*=a;
            }
            if(it>=2 && g<0.045) break;
        }
        if(attempt && fastnorm(p.b)>1e3 && !p.b.empty()) {
            Vec stat;
            stat.reserve(p.b.size());
            for(double x:p.b) if(x)stat.push_back(std::abs(x));
            auto pos=stat.begin()+stat.size()*9/10;
            std::nth_element(stat.begin(),pos,stat.end());
            double h=std::clamp(*pos*0.25,1.,1e9);
            if(h>1){
                for(double& x:p.b)x/=h;
                for(double& x:d)x*=h;
                for(double& x:e)x/=h;
                for(double& x:p.P.x)x*=h;
                gamma/=h;
                double w=std::max(fastnorm(p.q),fastnorm(p.P.x));
                if(w>16.){
                    for(double& x:p.P.x)x/=w;
                    for(double& x:p.q)x/=w;
                    gamma/=w;
                }
            }
        }
    }
};
struct Iterate {
    Vec x,s,z;
    double t=1,k=1;
    Iterate(int n,int m):x(n),s(m),z(m){}
    void update(double a,const Iterate& b){
        for(size_t j=0;j<x.size();++j) x[j]+=a*b.x[j];
        for(size_t j=0;j<s.size();++j) {s[j]+=a*b.s[j];z[j]+=a*b.z[j];}
        t+=a*b.t;k+=a*b.k;
        double v=std::max(t,k);
        if(v>100 || (v<0.01 && v>0)) {
            for(double& y:x)y/=v;
            for(double& y:s)y/=v;
            for(double& y:z)y/=v;
            t/=v;k/=v;
        }
    }
};
}

Verifier::Verifier(const Model& m) : p(m),nB(norm(m.b)),nQ(norm(m.q)),aScale(norm(m.A.x)),pScale(norm(m.P.x)) {
    if(!aScale) aScale=1;
    if(!pScale) pScale=1;
}
ProductCone& Verifier::cones() {
    if(!cone) cone=std::make_unique<ProductCone>(p);
    return *cone;
}
bool Verifier::solved(const Vec& x,const Vec& s0,const Vec& z0,bool repair) {
    if(result) return true;
    if(x.size()!=p.q.size() || s0.size()!=p.b.size() || z0.size()!=p.b.size()) return false;
    double nx=norm(x),ns=norm(s0),nz=norm(z0);
    if(!std::isfinite(nx)||!std::isfinite(ns)||!std::isfinite(nz)) return false;
    const auto ax=p.A.mul(x), px=p.P.mul(x), az=p.A.tmul(z0);
    double a=norm(ax), b=norm(px), c=norm(az);
    if(!std::isfinite(a)||!std::isfinite(b)||!std::isfinite(c)) return false;
    double xpx=dot(x,px), qx=dot(p.q,x), bz=dot(p.b,z0);
    double pobj=0.5*xpx+qx, dobj=-0.5*xpx-bz;
    if(!std::isfinite(pobj+p.offset) || !std::isfinite(dobj+p.offset)) return false;
    double g=std::abs(pobj-dobj)/std::max({1.0,std::abs(pobj),std::abs(dobj)});
    auto check=[&](const Vec& atz,double scale){
        double v=0;for(int j=0;j<p.n();++j)v=std::max(v,std::abs(px[j]+p.q[j]+atz[j]));
        return v/(1+std::max({b,nQ,scale}));
    };
    double rd=check(az,c),rp=0;
    for(int j=0;j<p.m();++j)rp=std::max(rp,std::abs(ax[j]+s0[j]-p.b[j]));
    rp/=1+std::max({a,ns,nB});
    const Vec* sn=&s0,*zn=&z0;
    Vec s,z;
    if(repair && rp>tolerance && rp<2e-3){
        s.resize(p.m());
        for(int j=0;j<p.m();++j)s[j]=p.b[j]-ax[j];
        sn=&s;
        rp=0;
        if(cones().defect(s,false,tolerance)>std::max(1e-7,rp)) {
            // Keep the actual interior slack as the better candidate.
            sn=&s0;
            for(int j=0;j<p.m();++j)rp=std::max(rp,std::abs(ax[j]+s0[j]-p.b[j]));
            rp/=1+std::max({a,ns,nB});
        }
    }
    if(repair && g>tolerance && g<1e-5 && bz){
        double sc=-(xpx+qx)/bz;
        if(sc>0.99 && sc<1.01){
            z.resize(z0.size());
            for(size_t j=0;j<z.size();++j) z[j]=z0[j]*sc;
            auto atz=p.A.tmul(z);
            double rz=check(atz,norm(atz)), d2=-0.5*xpx-dot(p.b,z);
            double g2=std::abs(pobj-d2)/std::max({1.,std::abs(pobj),std::abs(d2)});
            if(std::max(rz,g2)<std::max(rd,g)){ rd=rz;g=g2;zn=&z; }
        }
    }
    double err=std::max({rp,rd,g});
    if(err>=std::min(best,1e-4) && !debug) return false;
    double sd=cones().defect(*sn,false,tolerance),zd=cones().defect(*zn,true,tolerance);
    if(debug) std::fprintf(stderr,"  check %.2e %.2e %.2e %.2e %.2e\n",rp,rd,g,sd,zd);
    err=std::max({err,sd,zd});
    if(err<best && err<1e-3) {
        best=err;
        near={Solved,x,*sn,*zn};
    }
    if(err<=tolerance && std::isfinite(dot(*sn,*zn))){
        result={Solved,x,*sn,*zn};return true;
    }
    return false;
}
bool Verifier::primal(const Vec& z) {
    if(result) return true;
    double sz=norm(z);
    if(z.size()!=p.b.size() || !std::isfinite(sz) || sz==0 || nB==0) return false;
    Vec zz(z.size());
    for(size_t j=0;j<z.size();++j) zz[j]=z[j]/sz;
    double beta=-dot_scaled(p.b,nB,zz);
    if(beta<=1.000001e-8 || !std::isfinite(beta)) return false;
    auto atz=mul_scaled(p.A,zz,aScale,true);
    if(norm(atz)>tolerance*beta || cones().defect(zz,true,tolerance*beta)>tolerance*beta) return false;
    result.status=PrimalInfeasible;result.z=std::move(zz);
    return true;
}
bool Verifier::dual(const Vec& x,const Vec* s) {
    if(result) return true;
    double nx=norm(x);
    if(x.size()!=p.q.size() || !std::isfinite(nx) || nx==0 || nQ==0) return false;
    Vec xx(x.size());
    for(size_t j=0;j<x.size();++j) xx[j]=x[j]/nx;
    double alpha=-dot_scaled(p.q,nQ,xx);
    if(alpha<=1.000001e-8 || !std::isfinite(alpha))return false;
    double tol=tolerance*alpha;
    if(p.P.nnz() && norm(mul_scaled(p.P,xx,pScale))>tol)return false;
    Vec ad=mul_scaled(p.A,xx,aScale);
    if(!std::isfinite(norm(ad)))return false;
    if(s) {
        Vec ss(s->size());
        double err=0;
        for(size_t j=0;j<ss.size();++j){
            ss[j]=((*s)[j]/nx)/aScale;
            err=std::max(err,std::abs(ad[j]+ss[j]));
        }
        if(err<=tol && std::isfinite(norm(ss)) && cones().defect(ss,false,tol)<=tol){
            for(double& z:ss) z*=aScale;
            if(std::isfinite(norm(ss))){
                result={DualInfeasible,std::move(xx),std::move(ss),{}};return true;
            }
        }
    }
    for(double& z:ad)z=-z;
    if(cones().defect(ad,false,tol)>tol) return false;
    for(double& z:ad)z*=aScale;
    if(!std::isfinite(norm(ad))) return false;
    result={DualInfeasible,std::move(xx),std::move(ad),{}};return true;
}

bool ipm(const Model& p,Verifier& verify,int attempt){
    const int n=p.n(),m=p.m();
    Ind modes;
    modes.reserve(p.cones.size());
    bool has_soc=false,has_psd=false,alt=false;
    for(auto c:p.cones) {
        has_soc|=c.kind==Lorentz;
        has_psd|=c.kind==Semidefinite;
        int mode=0;
        if(c.kind==Lorentz) {
            if(attempt==1) mode=c.dim>28?3:0;
            else if(attempt==2 && n+m<150) {mode=c.dim>2?1:0;alt|=mode>0;}
            else mode=c.dim>2?3:0;
        } else if(c.kind==Semidefinite && c.dim>(attempt==1?6:3)) mode=2;
        else if(c.kind==Exponential && attempt!=1) mode=4;
        modes.push_back(mode);
    }
    if(alt && n+m>95) for(int& v:modes) if(v==1)v=3;
    ProductCone c(p,std::move(modes));
    Equilibrated eq(p,c,attempt);
    auto& u=eq.p;
    const auto& A=u.A;const auto& P=u.P;const auto& b=u.b;const auto& q=u.q;
    Newton K(u,c,attempt);
    Iterate v(n,m),aff(n,m),cc(n,m),extra(n,m);
    c.initial(v.s,v.z);
    double reg=attempt==1?2e-10:attempt==2?1e-7:2e-9;
    if(has_psd && !attempt) reg=8e-10;
    bool pde=attempt!=2;
    Vec az(n),px(n),ax(m),rp(m),rd(n),fx(n),fz(m),target(m),dxv(n),dzv(m),tempn(n),
        fd(n),minusq=negative(q);
    double mu=(dot(v.s,v.z)+1)/(c.degree+1);
    if(!attempt && n+m<250'000 && K.fill()<1'300'000 && !c.nexp && c.prepare(v.s,v.z,mu,pde) && K.factor(reg)){
        if(K.solve(fx,b,aff.x,aff.z) && K.solve(minusq,fz,cc.x,cc.z)){
            A.mul(aff.x.data(),ax.data());
            for(int j=0;j<m;++j) ax[j]=b[j]-ax[j];
            if(norm(aff.x)<2e5 && norm(ax)<2e5 && norm(cc.z)<2e5){
                v.x=aff.x;v.s=ax;v.z=cc.z;
                c.shift(v.s,false); c.shift(v.z,true);
            }
        }
    }
    auto original=[&](bool opt,bool primal,bool dual){
        Vec x,s,z;
        if(opt || dual){
            x.resize(n);s.resize(m);
            for(int j=0;j<n;++j)x[j]=eq.d[j]*v.x[j];
            for(int j=0;j<m;++j)s[j]=v.s[j]/eq.e[j];
        }
        if(opt || primal){
            z.resize(m);
            for(int j=0;j<m;++j)z[j]=eq.e[j]*v.z[j];
        }
        if(primal && verify.primal(z)) return true;
        if(dual && verify.dual(x,&s)) return true;
        if(opt && v.t>0){
            for(double& a:x)a/=v.t;
            for(double& a:s)a/=v.t;
            double zz=v.t*eq.gamma;
            for(double& a:z)a/=zz;
            return verify.solved(x,s,z);
        }
        return false;
    };
    auto step=[&](const Iterate& d){
        double bound=1.;
        if(d.t<0) bound=std::min(bound,-v.t/d.t);
        if(d.k<0) bound=std::min(bound,-v.k/d.k);
        bound=c.step(v.s,d.s,false,bound);
        return c.step(v.z,d.z,true,bound);
    };
    double fstep=has_soc?(attempt==1?0.95:0.975):(c.nexp?0.97:attempt==1?0.98:0.99);
    if(const char* f=std::getenv("CONIC_FRACTION")) fstep=std::clamp(std::atof(f),0.5,0.9995);
    const double nb=fastnorm(b), nq=fastnorm(q);
    int shortstep=0,stale=0,limit=90;
    if(n+m>=150000)limit=attempt?48:65;
    double best=INFINITY;
    for(int it=0;it<=limit;++it) {
        A.mul(v.x.data(),ax.data());P.mul(v.x.data(),px.data());A.tmul(v.z.data(),az.data());
        for(int j=0;j<m;++j)rp[j]=ax[j]+v.s[j]-b[j]*v.t;
        for(int j=0;j<n;++j)rd[j]=px[j]+az[j]+q[j]*v.t;
        const double xpx=dot(v.x,px),qx=dot(q,v.x),bz=dot(b,v.z), t=v.t,k=v.k;
        const double rg=qx+bz+xpx/t+k;
        mu=(dot(v.s,v.z)+t*k)/(c.degree+1);
        double pp=0.5*xpx/t+qx,dd=-0.5*xpx/t-bz;
        double rpn=norm(rp)/(t+std::max({fastnorm(ax),fastnorm(v.s),nb*t}));
        double rdn=norm(rd)/(t+std::max({fastnorm(px),fastnorm(az),nq*t}));
        double gap=std::abs(pp-dd)/std::max({t,std::abs(pp),std::abs(dd)});
        double err=std::max({rpn,rdn,gap});
        if(debug) std::fprintf(stderr,"%2d mu %.1e t %.1e k %.1e p %.1e d %.1e g %.1e\n",it,mu,t,k,rpn,rdn,gap);
        bool cert=(it>3 && k>1e-7*t && (t<3e-4*k || std::max(rpn,rdn)>0.25))||it==limit;
        bool opt=(err<5e-5 || (it && it%18==0));
        if((opt || cert) && original(opt,cert&&bz<0,cert&&qx<0)) return true;
        if(!(std::isfinite(err) && std::isfinite(mu) && mu>0 && t>0 && k>0)) break;
        if(err<best*.87) {best=err;stale=0;} else ++stale;
        if((stale>17 && t>1e-3*k && !(c.nexp && it<45)) || it==limit) break;
        if(!c.prepare(v.s,v.z,mu,pde) || !K.factor(reg) || !K.solve(minusq,b,dxv,dzv)){
            if(debug) std::fprintf(stderr,"  scaling/factorization failed\n");
            break;
        }
        for(int j=0;j<n;++j){ fd[j]=q[j]+2*px[j]/t; fx[j]=-rd[j];tempn[j]=dxv[j]-v.x[j]/t; }
        P.mul(tempn.data(),az.data());
        A.mul(dxv.data(),ax.data());
        for(int j=0;j<m;++j)ax[j]-=b[j];
        double den=-dot(tempn,az)-dot(dzv,ax)-k/t;
        if(!(den<0 && std::isfinite(den)))
            den=dot(fd,dxv)+dot(b,dzv)-xpx/(t*t)-k/t;
        if(!(den<0 && std::isfinite(den))) break;
        auto direction=[&](const Vec& tg,double sk,Iterate& d){
            for(int j=0;j<m;++j)fz[j]=-rp[j]-tg[j];
            if(!K.solve(fx,fz,d.x,d.z)) return false;
            double dt=(-rg-sk-dot(fd,d.x)-dot(b,d.z))/den;
            for(int j=0;j<n;++j) d.x[j]+=dt*dxv[j];
            A.mul(d.x.data(),ax.data());
            for(int j=0;j<m;++j){d.z[j]+=dt*dzv[j];d.s[j]=-rp[j]-ax[j]+b[j]*dt;}
            d.t=dt;d.k=sk-k/t*dt;
            return std::isfinite(dt) && std::isfinite(d.k);
        };
        for(int j=0;j<m;++j)target[j]=-v.s[j];
        if(!direction(target,-k,aff)) break;
        double aa=step(aff);
        double sg=1-aa;sg=sg*sg*sg;
        double wgt=it==0?aa:1;
        c.correct(v.s,v.z,aff.s,aff.z,sg*mu,wgt,target);
        bool ok=direction(target,-k+(sg*mu-wgt*aff.t*aff.k)/t,cc);
        double alpha=ok?step(cc):0;
        Iterate* next=&cc;
        if(!(alpha>=0.025)){
            double sig=std::clamp(sg,0.15,0.93);
            c.correct(v.s,v.z,aff.s,aff.z,sig*mu,0,target);
            if(direction(target,-k+sig*mu/t,extra)){
                double a2=step(extra);
                if(a2>1.25*alpha){alpha=a2;next=&extra;}
            }
            if(alpha<aa*0.45){alpha=aa;next=&aff;}
        }
        const double fs=c.nexp&&!pde?std::min(fstep,.94):fstep;
        alpha=std::min(1.0,fs*alpha);
        if(debug) std::fprintf(stderr,"  step %.4f aff %.4f sigma %.3f\n",alpha,aa,sg);
        if(!(alpha>1e-10 && std::isfinite(norm(next->x)) && std::isfinite(norm(next->z)))) break;
        if(alpha<0.012){
            ++shortstep;
            if(shortstep>=5 && pde && c.nexp) {pde=false;shortstep=0;}
            else if(shortstep>=8) break;
        } else shortstep=0;
        v.update(alpha,*next);
        if(!(v.t>0 && v.k>0)) break;
    }
    return original(true,true,true);
}

namespace {
struct BoundLP {
    const Model& p;
    Vec lc,uc,cl,cr,coefficient;
    Ind lower,upper,kept;
    Sparse reduced;
    bool active=false;
    BoundLP(const Model& p_,const Vec& rl) :p(p_),lc(p.n(),-INFINITY),uc(p.n(),INFINITY),
                    lower(p.n(),-1),upper(p.n(),-1) {
        const int m=p.m(),n=p.n();
        if(!m || m+n<220) return;
        Ind count(m),one(m);
        coefficient.resize(m);
        for(int j=0;j<n;++j)for(int k=p.A.p[j];k<p.A.p[j+1];++k){
            int r=p.A.i[k];
            if(count[r]++==0){one[r]=j;coefficient[r]=p.A.x[k];}
        }
        Ind to(m,-1);
        for(int j=0;j<m;++j) {
            if(count[j]==1 && !std::isfinite(rl[j])){
                double a=coefficient[j],b=p.b[j]/a; int v=one[j];
                if(std::isfinite(b) && std::abs(b)<1e29 && std::abs(a)>1e-160){
                    if(a>0 && b<uc[v]){uc[v]=b;upper[v]=j;}
                    if(a<0 && b>lc[v]){lc[v]=b;lower[v]=j;}
                    continue;
                }
            }
            to[j]=kept.size();kept.push_back(j);
        }
        if((int)kept.size()>m-std::max(12,m/16)){
            lc.assign(n,-INFINITY);uc.assign(n,INFINITY);return;
        }
        for(int j=0;j<n;++j) if(lc[j]>uc[j]){
            lc.assign(n,-INFINITY);uc.assign(n,INFINITY);return;
        }
        active=true;cl.reserve(kept.size());cr.reserve(kept.size());
        for(int j:kept){cl.push_back(rl[j]);cr.push_back(p.b[j]);}
        reduced=Sparse(kept.size(),n);
        for(int j=0;j<n;++j){
            for(int k=p.A.p[j];k<p.A.p[j+1];++k) {
                int r=to[p.A.i[k]];
                if(r>=0){reduced.i.push_back(r);reduced.x.push_back(p.A.x[k]);}
            }
            reduced.p[j+1]=reduced.nnz();
        }
    }
    Vec recover(const Vec& raw,const Vec* x=nullptr) const {
        if(!active)return raw;
        if(raw.size()!=kept.size())return {};
        Vec z(p.m());
        for(size_t j=0;j<raw.size();++j) z[kept[j]]=raw[j];
        auto station=p.A.tmul(z);
        if(x){
            auto px=p.P.mul(*x);
            for(int j=0;j<p.n();++j)station[j]+=p.q[j]+px[j];
        }
        for(int j=0;j<p.n();++j){
            int i=station[j]>0?lower[j]:upper[j];
            if(i>=0)z[i]=-station[j]/coefficient[i];
        }
        return z;
    }
};

bool direct_equality(const Model& p,Verifier& v){
    if(p.n()==0 || !std::all_of(p.cones.begin(),p.cones.end(),[](ConeSpec c){return c.kind==Zero;}))
        return false;
    try {
        ProductCone c(p,Ind(p.cones.size(),0));
        Equilibrated eq(p,c,0);
        Newton k(eq.p,c,0);
        Vec x(p.n()),z(p.m()),s(p.m()),q=negative(eq.p.q);
        for(double d:{4e-11,2e-9}) if(k.factor(d) && k.solve(q,eq.p.b,x,z)){
            for(int j=0;j<p.n();++j)x[j]*=eq.d[j];
            for(int j=0;j<p.m();++j)z[j]*=eq.e[j]/eq.gamma;
            if(v.solved(x,s,z,false) || v.primal(z) || v.primal(negative(z)) || v.dual(x))return true;
        }
    } catch(const std::exception& e){
        if(debug)std::fprintf(stderr,"equality solve skipped: %s\n",e.what());
    }
    return false;
}
}

bool linear_solver(const Model& p,Verifier& verify,bool alternate) {
    const int n=p.n(),m=p.m();
    if(!n) return false;
    const bool qp=p.P.nnz();
    Vec lc(n,-INFINITY),uc(n,INFINITY),lr(m,-INFINITY),x,z;
    int off=0;
    for(auto c:p.cones) {
        if(c.kind==Zero) std::copy(p.b.begin()+off,p.b.begin()+off+c.dim,lr.begin()+off);
        off+=c.dim;
    }
    LinearStatus st=LpUnknown;
    {
        HiGHS high;
        if(!high) return false;
        BoundLP bound(p,lr);
        const char* alg=!qp&&alternate?"ipm":"choose";
        double time=n+m<40000?(alternate?6.:4.5):n+m<160000?22.:70.;
        st=high.solve(bound.active?bound.reduced:p.A,p.q,bound.lc,bound.uc,bound.active?bound.cl:lr,
                      bound.active?bound.cr:p.b,qp?&p.P:nullptr,alg,!(qp&&alternate),time,
                      alternate?1e-10:1e-9);
        if(debug) std::fprintf(stderr,"HiGHS %d, rows %d\n",st,bound.active?bound.reduced.rows:m);
        if(high.point(x,z) && x.size()==p.q.size()) {
            auto zz=bound.recover(z,&x);
            if(zz.size()==p.b.size() && verify.solved(x,slack(p,x),zz))return true;
        }
        if((st==LpInfeasible || st==LpAmbiguous) && verify.nB){
            z.resize(bound.active?bound.reduced.rows:m);
            if(high.ray(z,false)){
                if(verify.primal(bound.recover(z)) || verify.primal(bound.recover(negative(z)))) return true;
            }
        }
        if((st==LpUnbounded || st==LpAmbiguous) && verify.nQ) {
            x.resize(n);
            if(high.ray(x,true) && (verify.dual(x) || verify.dual(negative(x)))) return true;
        }
    }
    if((st==LpInfeasible || st==LpAmbiguous) && verify.nB){
        auto B=p.A.transpose();
        for(double& t:B.x)t/=verify.aScale;
        Vec f(n), c=p.b, lo(m), hi(m,1);
        for(int j=0;j<m;++j){c[j]/=verify.nB;lo[j]=std::isfinite(lr[j])?-1.:0.;}
        HiGHS h;
        auto s=h.solve(B,c,lo,hi,f,f,nullptr,"choose",true,6.,1e-10);
        if(debug) std::fprintf(stderr,"HiGHS primal ray %d\n",s);
        if(h.point(x,z) && verify.primal(x)) return true;
    }
    if((st==LpUnbounded || st==LpAmbiguous || (qp && st==LpUnknown)) && verify.nQ){
        int mp=m+(qp?n:0);
        Sparse B(mp,n);
        B.i.reserve(p.A.nnz()+p.P.nnz());B.x.reserve(B.i.capacity());
        for(int j=0;j<n;++j){
            for(int k=p.A.p[j];k<p.A.p[j+1];++k) {
                B.i.push_back(p.A.i[k]);B.x.push_back(p.A.x[k]/verify.aScale);
            }
            if(qp) for(int k=p.P.p[j];k<p.P.p[j+1];++k){
                B.i.push_back(m+p.P.i[k]);B.x.push_back(p.P.x[k]/verify.pScale);
            }
            B.p[j+1]=B.nnz();
        }
        Vec l(mp),h(mp),q=p.q;
        for(int j=0;j<m;++j) l[j]=std::isfinite(lr[j])?0:-INFINITY;
        for(int j=0;j<n;++j) q[j]/=verify.nQ;
        lc.assign(n,-1);uc.assign(n,1);
        HiGHS high;
        auto s=high.solve(B,q,lc,uc,l,h,nullptr,"choose",true,6.,1e-10);
        if(debug) std::fprintf(stderr,"HiGHS dual ray %d\n",s);
        if(high.point(x,z) && verify.dual(x)) return true;
    }
    return false;
}

Result optimize_core(const Model& p,bool check_diagonal) {
    Verifier v(p);
    bool linear=is_linear(p),qp=p.P.nnz();
    if(elementary(p,v,linear)) return std::move(v.result);
    if(!linear && check_diagonal && diagonal_sdp(p,v))return std::move(v.result);
    if(!linear && projection_form(p,v))return std::move(v.result);
    if(!linear && logsum_form(p,v))return std::move(v.result);
    if(linear && qp && box_quadratic(p,v))return std::move(v.result);
    if(linear && qp && direct_equality(p,v))return std::move(v.result);
    if(linear && !qp && linear_solver(p,v,false)) return std::move(v.result);
    auto split=[&](int variant){
        if(p.n()+p.m()<440 && !v.near) return false;
        try {
            ProductCone c(p);
            Equilibrated eq(p,c,variant);
            return operator_split(eq.p,c,eq.d,eq.e,eq.gamma,v,variant);
        } catch(const std::exception& e) {
            if(debug)std::fprintf(stderr,"splitting skipped: %s\n",e.what());
            return false;
        }
    };
    double polished=INFINITY;
    auto polish=[&](){
        if(!v.near || v.best>2e-4 || v.best>=polished*.6 || p.n()+p.m()>55000)return false;
        polished=v.best;
        try{
            ProductCone c(p);
            Equilibrated eq(p,c,0);
            return polish_manifold(eq.p,eq.d,eq.e,eq.gamma,v);
        } catch(const std::exception& e) {
            if(debug)std::fprintf(stderr,"polish skipped: %s\n",e.what());
            return false;
        }
    };
    if(linear && !qp && v.near && polish())return std::move(v.result);
    if(linear && !qp && p.n()+p.m()<380000) {
        try {
            ProductCone c(p);
            Equilibrated eq(p,c,0);
            Verifier w(eq.p);
            if(linear_solver(eq.p,w,true)){
                auto& q=w.result;
                for(int j=0;j<(int)q.x.size();++j)q.x[j]*=eq.d[j];
                for(int j=0;j<(int)q.s.size();++j)q.s[j]/=eq.e[j];
                for(int j=0;j<(int)q.z.size();++j)q.z[j]*=eq.e[j]/eq.gamma;
                bool ok=q.status==Solved?v.solved(q.x,q.s,q.z):
                        q.status==DualInfeasible?v.dual(q.x,&q.s):v.primal(q.z);
                if(ok)return std::move(v.result);
            }
        } catch(const std::exception& e) {
            if(debug)std::fprintf(stderr,"scaled LP skipped: %s\n",e.what());
        }
    }
    for(int k=0;k<3;++k){
        try {
            if(ipm(p,v,k)) return std::move(v.result);
        } catch(const std::exception& e){
            if(debug) std::fprintf(stderr,"IPM fallback: %s\n",e.what());
        }
        if(polish())return std::move(v.result);
        if(k==0 && !v.near && p.n()+p.m()<2000 && certificate(p,v,false))
            return std::move(v.result);
        if(k==0 && !linear && (p.n()+p.m()<4500 || v.near) && supporting_planes(p,v))
            return std::move(v.result);
        if(k==0 && (p.n()+p.m()>=700 || v.near) && split(0)) return std::move(v.result);
        if(k==0 && polish())return std::move(v.result);
        if(k==0 && !linear && p.n()+p.m()>=4500 && !v.near && supporting_planes(p,v))
            return std::move(v.result);
        if(linear && (k==0 || (k==1&&qp)) && linear_solver(p,v,k!=0||!qp))
            return std::move(v.result);
    }
    if(p.n()+p.m()>600 && split(1))return std::move(v.result);
    if(certificate(p,v,true))return std::move(v.result);
    return {};
}

Result optimize(const Model& p) {
    Verifier v(p);
    bool linear=is_linear(p);
    if(elementary(p,v,linear))return std::move(v.result);
    bool diagonal=p.cones.size()==1 && p.cones[0].kind==Semidefinite &&
        p.cones[0].dim>=64 && p.n()==p.cones[0].dim && p.A.nnz()==p.n() && !p.P.nnz();
    if(diagonal && diagonal_sdp(p,v))return std::move(v.result);
    try {
        if(reduce_identity(p,v) || substitute_equalities(p,v) || reduce_rows(p,v))
            return std::move(v.result);
    } catch(const std::exception& e) {
        if(debug)std::fprintf(stderr,"presolve skipped: %s\n",e.what());
    }
    return optimize_core(p,!diagonal);
}
