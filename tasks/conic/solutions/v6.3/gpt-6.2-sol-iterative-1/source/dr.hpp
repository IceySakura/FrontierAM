#pragma once
#include "projection.hpp"
#include "manifold.hpp"
#include "poly.hpp"

namespace conic {

class FixedSystem {
    const Problem& p;
    int n,m;
    double until;
    Vec d,bd,cg_b,cg_r,cg_d,cg_z,cg_w,ax;
    std::unique_ptr<LinearSystem> sys;
public:
    Vec metric;
    bool ok=false;
    FixedSystem(const Problem& problem,Vec r,double end,bool indirect):
        p(problem),n(p.n()),m(p.m()),until(end),metric(std::move(r)) {
        if(!indirect && n+m<480000) {
            std::vector<Entry> entries;
            entries.reserve(size_t(n)+m+p.A.nz()+p.P.nz()/2);
            Int signs(n+m,-1);
            std::fill(signs.begin(),signs.begin()+n,1);
            for(int j=0;j<n+m;++j)entries.push_back({j,j,j<n?metric[j]:-metric[j]});
            for(int j=0;j<n;++j) {
                for(int k=p.P.p[j];k<p.P.p[j+1];++k) {
                    const int r=p.P.i[k];
                    if(r==j)entries[j].v+=p.P.v[k];
                    else if(r<j)entries.push_back({r,j,p.P.v[k]});
                }
                for(int k=p.A.p[j];k<p.A.p[j+1];++k)
                    entries.push_back({j,n+p.A.i[k],p.A.v[k]});
            }
            sys=std::make_unique<LinearSystem>(n+m,signs,entries,m==0);
            if(sys->ok && sys->work_est<2.8e8) {
                Vec values;values.reserve(entries.size());for(auto e:entries)values.push_back(e.v);
                if(sys->factor(values,1e-13)||sys->factor(values,1e-9,true)) {
                    ok=true;return;
                }
            }
            sys.reset();
        }
        if(clock_time()>until)return;
        d.resize(n);bd.resize(m);cg_b.resize(n);cg_r.resize(n);cg_d.resize(n);cg_z.resize(n);
        cg_w.resize(n);ax.resize(m);
        for(int j=0;j<n;++j) {
            double sum=metric[j];
            for(int k=p.P.p[j];k<p.P.p[j+1];++k)if(p.P.i[k]==j)sum+=p.P.v[k];
            for(int k=p.A.p[j];k<p.A.p[j+1];++k)
                sum+=p.A.v[k]*p.A.v[k]/metric[n+p.A.i[k]];
            d[j]=1./std::max(1e-12,sum);
        }
        ok=true;
    }
    void multiply(const double* x,double* f) {
        for(int j=0;j<n;++j)f[j]=metric[j]*x[j];
        p.P.add(x,f);
        std::fill(ax.begin(),ax.end(),0.);
        p.A.add(x,ax.data());
        for(int i=0;i<m;++i)bd[i]=ax[i]/metric[n+i];
        p.A.add_transpose(bd.data(),f);
    }
    bool solve(const Vec& f,Vec& ans,double tol,int refine=0) {
        if(sys)return sys->solve(f,ans,refine);
        for(int j=0;j<n;++j)cg_b[j]=f[j];
        for(int j=0;j<m;++j)bd[j]=f[n+j]/metric[n+j];
        p.A.add_transpose(bd.data(),cg_b.data());
        multiply(ans.data(),cg_w.data());
        for(int j=0;j<n;++j) {
            cg_r[j]=cg_b[j]-cg_w[j];cg_d[j]=d[j]*cg_r[j];
        }
        double rr=dot(cg_r,cg_d);
        tol*=std::max(1e-8,norm(cg_b));
        const int iterations=refine?125:48;
        for(int k=0;k<iterations && rr>0. && norm(cg_r)>tol;++k) {
            if((k&7)==0 && clock_time()>until)return false;
            multiply(cg_d.data(),cg_w.data());
            const double x=dot(cg_d,cg_w);
            if(!(x>0.))break;
            const double a=rr/x;
            for(int j=0;j<n;++j) {
                ans[j]+=a*cg_d[j];cg_r[j]-=a*cg_w[j];cg_z[j]=d[j]*cg_r[j];
            }
            const double t=dot(cg_z,cg_r), beta=t/rr;
            rr=t;
            for(int j=0;j<n;++j)cg_d[j]=cg_z[j]+beta*cg_d[j];
        }
        if(norm(cg_r)>std::max(1e-5,tol*1e4) && refine)return false;
        std::fill(ans.begin()+n,ans.end(),0.);
        p.A.add(ans.data(),ans.data()+n);
        for(int j=0;j<m;++j)ans[n+j]=(ans[n+j]-f[n+j])/metric[n+j];
        return finite(ans);
    }
};

struct Anderson {
    int size,history, pos=0, count=0;
    bool have=false, pending=false;
    double last=0.;
    const Vec& weights;
    std::vector<Vec> delta_f,delta_t;
    Vec previous,prev_f,backup;
    explicit Anderson(const Vec& d,int slots):size(int(d.size())),history(slots),weights(d),
        delta_f(slots),delta_t(slots),previous(size),prev_f(size),backup(size) {
        for(auto& x:delta_f)x.resize(size);
        for(auto& x:delta_t)x.resize(size);
    }
    double wdot(const Vec& x,const Vec& y) const {
        double t=0.;
        for(int k=0;k<size;++k)t+=weights[k]*x[k]*y[k];
        return t;
    }
    void clear() {count=pos=0;have=pending=false;}
    void next(Vec& x,const Vec& g,int it) {
        const double nn=wdot(g,g);
        if(pending && !(nn<6.*last)) {
            x=backup;clear();return;
        }
        pending=false;
        if(history && have) {
            Vec& d=delta_f[pos],&t=delta_t[pos];
            for(int i=0;i<size;++i) {
                d[i]=g[i]-prev_f[i];t[i]=x[i]-previous[i]+d[i];
            }
            ++pos;if(pos==history)pos=0;
            count=std::min(count+1,history);
        }
        if(history) {previous=x;prev_f=g;have=true;}
        for(int i=0;i<size;++i)x[i]+=g[i];
        if(count<2 || it%5!=4 || !(nn>1e-28))return;
        double a[64]{},v[8]{},md=0.;
        for(int j=0;j<count;++j) {
            v[j]=wdot(delta_f[j],g);
            for(int i=j;i<count;++i)a[i+count*j]=wdot(delta_f[i],delta_f[j]);
            md=std::max(md,a[j+count*j]);
        }
        for(int i=0;i<count;++i)a[i+count*i]+=std::max(1e-19,1e-10*md);
        if(!chol(count,a))return;
        int info=0,one=1;
        scipy_dpotrs_("L",&count,&one,a,&count,v,&count,&info);
        if(info)return;
        double s=0.;
        for(int i=0;i<count;++i)s+=std::abs(v[i]);
        if(!(s<=1e3))return;
        backup=x;last=nn;pending=true;
        for(int j=0;j<count;++j)
            for(int i=0;i<size;++i)x[i]-=v[j]*delta_t[j][i];
    }
};

template<class Optimal,class Ray>
bool solve_dr(const Problem& p,double until,Optimal&& accept,Ray&& ray,int strategy) {
    const int n=p.n(),m=p.m(),size=n+m,D=size+1;
    if(n+m>2100000 || clock_time()>until || p.A.nz()+p.P.nz()>12000000 ||
        std::any_of(p.cones.begin(),p.cones.end(),[](auto c){return c.kind==PSD && c.dim>138;}))return false;
    Vec metric(D,1.);
    for(int j=0;j<n;++j)metric[j]=strategy==2?1e-2:3e-4;
    for(auto c:p.cones) {
        const double sc=c.kind==Zero?.025:strategy==2?.5:1.;
        std::fill(metric.begin()+n+c.off,metric.begin()+n+c.off+c.width,sc);
    }
    FixedSystem system(p,metric,until,strategy==1);
    if(!system.ok || clock_time()>until)return false;
    Vec f(size),u0(size),u1(size),pp(n),xp(n),s(m),z(m),y(m),ax(m),at(n);
    Vec w(D,0.),mirror(D),diff(D);
    for(int j=0;j<n;++j)f[j]=-p.q[j];
    for(int j=0;j<m;++j)f[n+j]=p.b[j];
    if(!system.solve(f,u1,1e-11,3))return false;
    double aa=1.;
    for(int i=0;i<size;++i)aa+=metric[i]*u1[i]*u1[i];
    Vec pu1(n);
    p.P.add(u1.data(),pu1.data());
    w[size]=1.;
    Anderson accelerate(metric,size<200000?5:size<480000?2:0);
    ConeProjector projector(p);
    double merit=1.,last_polish=1.;
    const double nb=norm(p.b,m),nq=norm(p.q,n);
    for(int it=0;it<3200;++it) {
        if((it&3)==0 && clock_time()>until)break;
        for(int j=0;j<n;++j)f[j]=metric[j]*w[j];
        for(int j=0;j<m;++j)f[n+j]=-metric[n+j]*w[n+j];
        if(!system.solve(f,u0,std::clamp(.012*std::pow(merit,1.15),2e-12,.014)))return false;
        double bb=-w[size]-dot(p.q,u0.data(),n)-dot(p.b,u0.data()+n,m)
            -2.*dot(pu1.data(),u0.data(),n);
        double cc=0.;
        if(p.P.nz()) {
            std::fill(pp.begin(),pp.end(),0.);p.P.add(u0.data(),pp.data());
            cc=std::max(0.,dot(pp.data(),u0.data(),n));
        }
        const double disc=std::hypot(bb,2.*std::sqrt(aa)*std::sqrt(cc));
        const double tau=cc==0. ? std::max(0.,-bb/aa) : bb>=0. ? 2.*cc/(bb+disc) :
            (disc-bb)/(2.*aa);
        for(int i=0;i<size;++i) {
            double u=u0[i]+u1[i]*tau;
            mirror[i]=2*u-w[i];diff[i]=-u;
        }
        mirror[size]=2*tau-w[size];diff[size]=-tau;
        std::copy(mirror.begin()+n,mirror.begin()+size,z.begin());
        if(!projector.project(z))return false;
        for(int i=0;i<n;++i)diff[i]+=mirror[i];
        for(int i=0;i<m;++i)diff[n+i]+=z[i];
        diff[size]+=std::max(0.,mirror[size]);
        if(it==7 || (it+1)%24==0) {
            const double tc=std::max(0.,mirror[size]);
            std::copy(mirror.begin(),mirror.begin()+n,xp.begin());
            for(int j=0;j<m;++j)s[j]=metric[n+j]*(z[j]-mirror[n+j]);
            if(tc>1e-10) {
                for(double& a:xp)a/=tc;
                for(int j=0;j<m;++j){s[j]/=tc;y[j]=z[j]/tc;}
                std::fill(ax.begin(),ax.end(),0.);std::fill(pp.begin(),pp.end(),0.);
                std::fill(at.begin(),at.end(),0.);
                p.A.add(xp.data(),ax.data());p.P.add(xp.data(),pp.data());
                p.A.add_transpose(y.data(),at.data());
                double rp=0.,rd=0.;
                for(int j=0;j<m;++j)rp=std::max(rp,std::abs(ax[j]+s[j]-p.b[j]));
                for(int j=0;j<n;++j)rd=std::max(rd,std::abs(pp[j]+p.q[j]+at[j]));
                const double pr=rp/(1.+std::max({nb,norm(ax),norm(s)}));
                const double dr=rd/(1.+std::max({nq,norm(pp),norm(at)}));
                const double qp=dot(p.q,xp.data(),n), xpx=.5*dot(xp,pp), bz=dot(p.b,y.data(),m);
                const double gp=std::abs(2*xpx+qp+bz)/std::max({1.,std::abs(xpx+qp),std::abs(xpx+bz)});
                merit=std::max({pr,dr,gp});
                if(std::getenv("CONIC_TRACE") && ((it+1)%240==0 || it<24))
                    std::fprintf(stderr,"splitting %d: %.3g %.3g %.3g\n",it+1,pr,dr,gp);
                if(merit<1e-6 && accept(xp,s,y))return true;
                if(merit<std::min(.008,last_polish*.35) && clock_time()+.03<until) {
                    last_polish=merit;
                    const double end=std::min(until,clock_time()+.18);
                    if(is_polyhedral(p)) {
                        if(poly_polish(p,xp,y,end,accept,4))return true;
                    } else if(semismooth(p,xp,y,end,accept,strategy==2?1:0))return true;
                }
                if(!std::isfinite(merit) || merit>100.)merit=1.;
            } else {
                if(ray(xp,z))return true;
                merit=std::min(merit,.01);
            }
        }
        for(double& v:diff)v*=1.55;
        accelerate.next(w,diff,it);
        if((it&15)==15) {
            double wn=norm(w);
            if(!(wn>0.) || !std::isfinite(wn))return false;
            if(wn<.02 || wn>100.) {
                for(double& v:w)v/=wn;
                accelerate.clear();
            }
        }
    }
    return false;
}

} // namespace conic
