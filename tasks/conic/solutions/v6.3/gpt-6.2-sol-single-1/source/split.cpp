#include "optimizer.hpp"

#include <chrono>
#include <cstdio>
#include <deque>

namespace {
using Clock=std::chrono::steady_clock;

// Normal equations for the x-update, applied without forming A'RA.
class Krylov {
    const Model& p;
    const Sparse& t;
    double sigma;
    const Vec* weight=nullptr;
    Vec pi,ay,r,v,dir,hd;
public:
    Krylov(const Model& p_,const Sparse& t_,double sig)
        :p(p_),t(t_),sigma(sig),pi(p.n()),ay(p.m()),r(p.n()),v(p.n()),dir(p.n()),hd(p.n()){}
    void setup(const Vec& w) {
        weight=&w;
        for(int j=0;j<p.n();++j) {
            double u=sigma;
            for(int k=p.P.p[j];k<p.P.p[j+1];++k) if(p.P.i[k]==j) u+=p.P.x[k];
            for(int k=p.A.p[j];k<p.A.p[j+1];++k) {
                double a=p.A.x[k];
                u+=w[p.A.i[k]]*(a*a);
            }
            pi[j]=1/std::max(u,0.1*sigma);
        }
    }
    void apply(const Vec& x,Vec& y) {
        t.tmul(x.data(),ay.data());
        for(int j=0;j<p.m();++j)ay[j]*=(*weight)[j];
        p.A.tmul(ay.data(),y.data());
        for(int j=0;j<p.n();++j) {
            double u=y[j]+sigma*x[j];
            for(int k=p.P.p[j];k<p.P.p[j+1];++k)u+=p.P.x[k]*x[p.P.i[k]];
            y[j]=u;
        }
    }
    bool solve(const Vec& f,Vec& x,double atol,int limit){
        apply(x,hd);
        double e=0,rz=0;
        for(int j=0;j<p.n();++j) {
            r[j]=f[j]-hd[j];e=std::max(e,std::abs(r[j]));
            dir[j]=pi[j]*r[j];rz+=dir[j]*r[j];
        }
        if(e<=atol)return true;
        for(int it=0;it<limit;++it){
            apply(dir,hd);
            double d=dot(dir,hd);
            if(!(d>0) || !std::isfinite(rz))break;
            double alpha=rz/d, err=0, rr=0;
            for(int j=0;j<p.n();++j){
                x[j]+=alpha*dir[j];
                double u=r[j]-alpha*hd[j];r[j]=u;
                v[j]=pi[j]*u;rr+=v[j]*u;
                err=std::max(err,std::abs(u));
            }
            if(err<=atol) return true;
            if(it && it%75==0){
                apply(x,hd);
                rr=0;
                for(int j=0;j<p.n();++j){
                    r[j]=f[j]-hd[j];v[j]=pi[j]*r[j];rr+=v[j]*r[j];
                }
                dir=v;rz=rr;continue;
            }
            double beta=rr/rz;
            for(int j=0;j<p.n();++j)dir[j]=v[j]+beta*dir[j];
            rz=rr;
        }
        return false;
    }
};

class Acceleration {
    static constexpr int depth=5;
    struct Change{Vec s,y;};
    std::deque<Change> hist;
    Vec old,older,backup,oldx;
    bool have=false,trial=false;
    double base_err=0;
public:
    bool reject(double err,Vec& w,Vec& x) {
        if(!trial)return false;
        trial=false;
        if(err>1.45*base_err+3e-12 || !std::isfinite(err)) {
            w.swap(backup);x.swap(oldx);
            reset();
            return true;
        }
        return false;
    }
    void reset(){hist.clear();have=false;trial=false;}
    void move(Vec& w,const Vec& f,const Vec& x,double err,int it){
        int m=w.size();
        if(have) {
            Change u;
            if((int)hist.size()==depth){
                u=std::move(hist.front());hist.pop_front();
            } else {u.s.resize(m);u.y.resize(m);}
            for(int j=0;j<m;++j){u.s[j]=w[j]-old[j];u.y[j]=f[j]-older[j];}
            hist.push_back(std::move(u));
        }
        old=w;older=f;have=true;
        for(int j=0;j<m;++j)w[j]+=f[j];
        int q=hist.size();
        if(it%4 || q<3 || err<5e-12) return;
        double a[depth][depth]{},b[depth]{};
        double largest=0;
        for(int j=0;j<q;++j){
            b[j]=dot(hist[j].y,f);
            for(int i=0;i<=j;++i)a[j][i]=dot(hist[i].y,hist[j].y);
            largest=std::max(largest,a[j][j]);
        }
        if(!(largest>0))return;
        double ridge=largest*3e-8;
        for(int j=0;j<q;++j)a[j][j]+=ridge;
        for(int j=0;j<q;++j) for(int k=0;k<=j;++k){
            double u=a[j][k];
            for(int i=0;i<k;++i)u-=a[j][i]*a[k][i];
            if(j==k) {
                if(!(u>0))return;
                a[j][j]=std::sqrt(u);
            } else a[j][k]=u/a[k][k];
        }
        for(int j=0;j<q;++j) {
            for(int k=0;k<j;++k)b[j]-=a[j][k]*b[k];
            b[j]/=a[j][j];
        }
        double nz=0;
        for(int j=q-1;j>=0;--j) {
            for(int k=j+1;k<q;++k)b[j]-=a[k][j]*b[k];
            b[j]/=a[j][j];nz+=b[j]*b[j];
        }
        if(!(nz<2500))return;
        backup=w;oldx=x;
        for(int j=0;j<q;++j){
            double u=b[j];const auto& h=hist[j];
            for(int k=0;k<m;++k)w[k]-=u*(h.s[k]+h.y[k]);
        }
        base_err=err;trial=true;
    }
};

struct Recover {
    const Vec& d;
    const Vec& e;
    double gamma;
    Verifier& checker;
    bool optimal(const Vec& x,const Vec& s,const Vec& z){
        Vec xx(x.size()),ss(s.size()),zz(z.size());
        for(size_t j=0;j<x.size();++j)xx[j]=x[j]*d[j];
        for(size_t j=0;j<s.size();++j){ss[j]=s[j]/e[j];zz[j]=z[j]*e[j]/gamma;}
        return checker.solved(xx,ss,zz);
    }
    bool direction(const Vec& x,const Vec& z,const Vec* xx,const Vec* zz) {
        if(checker.nB>0) {
            Vec y(z.size());
            for(size_t j=0;j<z.size();++j)y[j]=(z[j]-(zz?(*zz)[j]:0))*e[j];
            if(checker.primal(y))return true;
        }
        if(checker.nQ>0) {
            Vec y(x.size());
            for(size_t j=0;j<x.size();++j)y[j]=(x[j]-(xx?(*xx)[j]:0))*d[j];
            if(checker.dual(y))return true;
        }
        return false;
    }
};

bool polish(const Model& p,const Sparse& t,Krylov& cg,Recover& out,const Vec& x0,
            const Vec& s0,const Vec& z0,const Vec& base,double rp) {
    Vec w(p.m()),x=x0,z=z0,y(p.m()),s(p.m()),f(p.n()),r(p.m());
    const double eps=std::clamp(5*rp,5e-8,8e-4);
    int off=0;
    double pm=fastnorm(p.b);
    for(auto cone:p.cones){
        for(int j=off;j<off+cone.dim;++j){
            bool selected=cone.kind==Zero || (s0[j]<eps*(1+std::abs(p.b[j])) &&
                            (z0[j]>eps*.1 || s0[j]<eps*.03));
            w[j]=selected?std::max(8.,16*base[j]):0;
            if(!selected)z[j]=0;
        }
        off+=cone.dim;
    }
    cg.setup(w);
    for(int it=0;it<15;++it){
        for(int j=0;j<p.m();++j) y[j]=w[j]*p.b[j]-z[j];
        p.A.tmul(y.data(),f.data());
        for(int j=0;j<p.n();++j) f[j]+=1e-7*x[j]-p.q[j];
        cg.solve(f,x,6e-12*(1+fastnorm(f)),std::min(240,p.n()+2));
        if(!std::isfinite(norm(x)))break;
        t.tmul(x.data(),r.data());
        for(int j=0;j<p.m();++j) {
            s[j]=p.b[j]-r[j];
            z[j]-=w[j]*s[j];
        }
        if(out.optimal(x,s,z))return true;
        if(it==2 || it==5 || it==9) {
            bool change=false;off=0;
            for(auto cone:p.cones){
                if(cone.kind==Orthant)for(int j=off;j<off+cone.dim;++j){
                    if(w[j] && z[j]<-std::max(5e-9,eps*.06)){w[j]=0;z[j]=0;change=true;}
                    else if(!w[j] && s[j]<-std::max(4e-9,eps*.018)*(1+pm)){
                        w[j]=std::max(8.,16*base[j]);change=true;
                    }
                }
                off+=cone.dim;
            }
            if(change)cg.setup(w);
        }
    }
    cg.setup(base);
    return false;
}
}

bool operator_split(const Model& p,ProductCone& cone,const Vec& d,const Vec& e,double gamma,
                    Verifier& verify,int variant) {
    int n=p.n(),m=p.m(),maxpsd=0;
    if(n==0 || m==0) return false;
    bool linear=true;
    for(auto c:p.cones) {
        linear&=c.kind<=Orthant;
        if(c.kind==Semidefinite)maxpsd=std::max(maxpsd,c.dim);
    }
    if(maxpsd>(variant?800:500))return false;
    auto start=Clock::now();
    const double budget=variant?6.5:p.A.nnz()+p.P.nnz()>400000?1.10:.66;
    const bool accelerated=m<350000;
    Acceleration mix;
    Vec w=p.b,s(m),z(m),y(m),rp(m),f(n),x(n),px(n),az(n),lastx,lastz,R(m),update(m);
    Vec rtype(m,1.);
    int off=0;
    for(auto k:p.cones){
        if(k.kind==Zero) for(int j=off;j<off+k.dim;++j)rtype[j]=3.;
        off+=k.width();
    }
    double rho=variant?.25:.8;
    if(p.P.nnz() && linear && !variant){
        double trace=0, rows=0;
        for(int j=0;j<n;++j) {
            for(int k=p.P.p[j];k<p.P.p[j+1];++k)if(p.P.i[k]==j)trace+=p.P.x[k];
            for(int k=p.A.p[j];k<p.A.p[j+1];++k)rows+=p.A.x[k]*p.A.x[k]*rtype[p.A.i[k]];
        }
        if(trace>0 && rows>0)rho=std::clamp(trace/rows,.035,.85);
    }
    for(int j=0;j<m;++j)R[j]=rho*rtype[j];
    if(verify.near){
        for(int j=0;j<n;++j)x[j]=verify.near.x[j]/d[j];
        for(int j=0;j<m;++j)w[j]=e[j]*verify.near.s[j]-gamma*verify.near.z[j]/(e[j]*R[j]);
    }
    Sparse at=p.A.transpose();
    Krylov cg(p,at,1e-7);
    cg.setup(R);
    double last_err=1.,last_polish=INFINITY,best=INFINITY;
    int lp_iters=variant?2500:1000;
    int maxcg=std::min(n+1,variant?210:105);
    int interval=maxpsd>=100?2:m+n>200000?12:8;
    Recover out{d,e,gamma,verify};
    for(int it=0;it<lp_iters;++it){
        s=w;cone.project_pair(s,z);
        for(int j=0;j<m;++j){
            z[j]*=R[j];
            y[j]=R[j]*(p.b[j]-s[j])-z[j];
        }
        p.A.tmul(y.data(),f.data());
        for(int j=0;j<n;++j)f[j]+=1e-7*x[j]-p.q[j];
        double tol=std::clamp(0.015*last_err,1e-12,0.001)*(1+fastnorm(f));
        cg.solve(f,x,tol,maxcg);
        at.tmul(x.data(),rp.data());
        const double relax=1.65;
        double rpn=0,bn=1;
        for(int j=0;j<m;++j){
            bn=std::max(bn,std::abs(rp[j])+std::abs(s[j])+std::abs(p.b[j]));
            double r=p.b[j]-s[j]-rp[j];rp[j]=r;update[j]=relax*r;
            rpn=std::max(rpn,std::abs(r));
        }
        double fp=rpn/bn;
        if(accelerated && mix.reject(rpn,w,x))continue;
        last_err=fp;
        if(it%interval==0 || fp<2e-7){
            p.P.mul(x.data(),px.data());p.A.tmul(z.data(),az.data());
            double rdn=0, dn=1;
            for(int j=0;j<n;++j){
                rdn=std::max(rdn,std::abs(px[j]+az[j]+p.q[j]));
                dn=std::max(dn,std::abs(px[j])+std::abs(az[j])+std::abs(p.q[j]));
            }
            double fd=rdn/dn,r=std::max(fp,fd);
            best=std::min(best,r);
            if(debug && it%32==0)std::fprintf(stderr," split %d %.2g %.2g, rho %.2g\n",it,fp,fd,rho);
            if(r<5e-5 && out.optimal(x,s,z))return true;
            if(linear && r<.00055 && it>=20 && r<last_polish*.33 && n<220000) {
                if(polish(p,at,cg,out,x,s,z,R,r))return true;
                last_polish=r;
            }
            if(it>16 && it%48==0){
                if((r>1e-6 && (out.direction(x,z,lastx.empty()?nullptr:&lastx,lastz.empty()?nullptr:&lastz)
                                   || out.direction(x,z,nullptr,nullptr))))return true;
                lastx=x;lastz=z;
                double adj=std::clamp(std::sqrt((1e-15+fp)/(1e-15+fd)),.2,5.);
                if((adj>2.0 || adj<.5) && r>2e-8){
                    double nr=std::clamp(rho*adj,.005,200.);
                    if(nr!=rho){
                        rho=nr;
                        for(int j=0;j<m;++j){R[j]=rho*rtype[j];w[j]=s[j]-z[j]/R[j];}
                        cg.setup(R);mix.reset();continue;
                    }
                }
            }
            double elapsed=std::chrono::duration<double>(Clock::now()-start).count();
            if(elapsed>budget || (it>190 && best>.16 && !variant))break;
        }
        if(!std::isfinite(rpn))break;
        if(accelerated)mix.move(w,update,x,rpn,it);
        else for(int j=0;j<m;++j)w[j]+=update[j];
    }
    return out.optimal(x,s,z)||out.direction(x,z,lastx.empty()?nullptr:&lastx,lastz.empty()?nullptr:&lastz);
}
