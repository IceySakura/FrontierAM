#pragma once
#include "kkt.hpp"

namespace conic {

inline bool is_polyhedral(const Problem& p) {
    return std::all_of(p.cones.begin(),p.cones.end(),[](auto c){return c.kind<=Linear;});
}

template<class Accept>
bool poly_polish(const Problem& p,Vec x,Vec z,double until,Accept&& accept,int attempts=5) {
    const int n=p.n(),m=p.m();
    if(!is_polyhedral(p) || clock_time()>until ||
        n+m>260000 || p.P.nz()+p.A.nz()>3000000)return false;
    Int active(m,-1);
    std::vector<unsigned char> is_eq(m,0), use(m,0);
    Vec ax(m),grad(n),s(m),rd(n);
    p.A.add(x.data(),ax.data());
    double zscale=std::max(1.,norm(z)), bscale=std::max(1.,norm(p.b,m));
    for(auto c:p.cones)for(int j=c.off;j<c.off+c.width;++j) {
        const double slack=p.b[j]-ax[j];
        is_eq[j]=c.kind==Zero;
        use[j]=is_eq[j] || slack < -2e-8*bscale ||
            (z[j]>2e-12*zscale && slack<std::max(2e-8*bscale,.7*z[j]));
    }
    for(int it=0;it<attempts && clock_time()<until;++it) {
        int k=0;
        for(int j=0;j<m;++j)active[j]=use[j]?k++:-1;
        Vec rhs(n+k,0.),result(n+k),curz(m,0.);
        Int signs(n+k,-1);
        std::fill(signs.begin(),signs.begin()+n,1);
        std::vector<Entry> es;
        es.reserve(size_t(n+k)+p.P.nz()/2+p.A.nz());
        for(int j=0;j<n+k;++j)es.push_back({j,j,0.});
        for(int j=0;j<n;++j) {
            for(int t=p.P.p[j];t<p.P.p[j+1];++t) {
                int i=p.P.i[t];
                if(i==j)es[j].v+=p.P.v[t];
                else if(i<j)es.push_back({i,j,p.P.v[t]});
            }
            for(int t=p.A.p[j];t<p.A.p[j+1];++t) {
                int i=active[p.A.i[t]];
                if(i>=0)es.push_back({j,n+i,p.A.v[t]});
            }
        }
        for(int j=0;j<m;++j)if(active[j]>=0) {
            curz[j]=z[j];rhs[n+active[j]]=p.b[j]-ax[j];
        }
        std::fill(grad.begin(),grad.end(),0.);
        p.P.add(x.data(),grad.data());
        p.A.add_transpose(curz.data(),grad.data());
        for(int j=0;j<n;++j)rhs[j]=-grad[j]-p.q[j];
        LinearSystem system(n+k,signs,es,k==0);
        if(!system.ok || system.work_est>5.5e8 || clock_time()>until)return false;
        Vec v;v.reserve(es.size());
        for(auto a:es)v.push_back(a.v);
        if(!system.factor(v,1e-9) && !system.factor(v,3e-8,true))return false;
        if(!system.solve(rhs,result,8))return false;
        Vec trialx=x, trialz=curz, trialax(m,0.);
        for(int j=0;j<n;++j)trialx[j]+=result[j];
        for(int j=0;j<m;++j)if(active[j]>=0)trialz[j]+=result[n+active[j]];
        p.A.add(trialx.data(),trialax.data());
        for(int j=0;j<m;++j)s[j]=is_eq[j]?0.:std::max(0.,p.b[j]-trialax[j]);
        if(accept(trialx,s,trialz))return true;
        if(!finite(trialax)||!finite(trialx)||!finite(trialz))return false;
        bool changed=false;
        bscale=std::max({1.,norm(trialax),norm(p.b,m)});
        zscale=std::max(1.,norm(trialz));
        for(int j=0;j<m;++j) if(!is_eq[j]) {
            if(use[j] && trialz[j]<-1e-10*zscale) {
                use[j]=0;changed=true;
            } else if(!use[j] && trialax[j]-p.b[j]>1e-10*bscale) {
                use[j]=1;changed=true;
            }
        }
        if(!changed) {
            if(it>0)return false;
            // Recompute after a better-informed active set on ambiguous rows.
            for(int j=0;j<m;++j) if(!is_eq[j] && use[j] && trialz[j]<1e-12*zscale)
                use[j]=0;
        }
        if(norm(trialx)>1000*(1+norm(x)) || norm(trialz)>1000*(1+norm(z)))return false;
        x.swap(trialx); z.swap(trialz);ax.swap(trialax);
    }
    return false;
}

// Fixed-system ADMM for polyhedral quadratic programs:
// y represents A*x, with interval projection for the linear rows.
template<class Optimal,class Ray>
bool poly_admm(const Problem& p,double until,Optimal&& accept,Ray&& ray,bool indirect=false) {
    const int n=p.n(),m=p.m(),dim=n+m;
    if(!is_polyhedral(p)||dim>2000000 || clock_time()>until)return false;
    Vec xb(n), z(m), ax(m), px(n), at(n), y(m),s(m),dx(n),dz(m);
    Vec rhs(dim), ans(dim), rinv(m), previousx(n), previousz(m);
    std::vector<unsigned char> eq(m,0);
    bool has_lin=false;
    for(auto c:p.cones) {
        if(c.kind==Zero)std::fill(eq.begin()+c.off,eq.begin()+c.off+c.width,1);
        else has_lin=true;
    }
    for(int j=0;j<m;++j)y[j]=eq[j]?p.b[j]:std::min(0.,p.b[j]);
    if(!has_lin&&poly_polish(p,xb,z,until,accept,1))return true;
    const double sigma=2e-7;
    std::unique_ptr<LinearSystem> sys;
    Vec vals;
    if(!indirect && n+m<500000) {
        std::vector<Entry> entries;
        entries.reserve(size_t(dim)+p.P.nz()/2+p.A.nz());
        Int signs(dim,-1);
        std::fill(signs.begin(),signs.begin()+n,1);
        for(int j=0;j<dim;++j)entries.push_back({j,j,j<n?sigma:-1.});
        for(int j=0;j<n;++j) {
            for(int k=p.P.p[j];k<p.P.p[j+1];++k) {
                int i=p.P.i[k];double a=p.P.v[k];
                if(i==j)entries[j].v+=a;
                else if(i<j)entries.push_back({i,j,a});
            }
            for(int k=p.A.p[j];k<p.A.p[j+1];++k)
                entries.push_back({j,n+p.A.i[k],p.A.v[k]});
        }
        sys=std::make_unique<LinearSystem>(dim,signs,entries,n==dim);
        if(!sys->ok||sys->work_est>2.5e8)sys.reset();
        else {
            vals.reserve(entries.size());for(auto a:entries)vals.push_back(a.v);
        }
    }
    if(clock_time()>until)return false;
    Vec inv_diag(n),rvec(n),pot(n),dvec(n),g(n),mult(n),bmv(m);
    auto factor = [&](double rho) {
        for(int j=0;j<m;++j) {
            rinv[j]=1./(rho*(eq[j]?120.:1.));
            if(sys)vals[n+j]=-rinv[j];
        }
        if(sys && (sys->factor(vals,1e-12) || sys->factor(vals,1e-9,true)))return true;
        sys.reset();
        for(int j=0;j<n;++j) {
            double v=sigma;
            for(int k=p.P.p[j];k<p.P.p[j+1];++k)if(p.P.i[k]==j)v+=p.P.v[k];
            for(int k=p.A.p[j];k<p.A.p[j+1];++k)v+=p.A.v[k]*p.A.v[k]/rinv[p.A.i[k]];
            inv_diag[j]=1./std::max(sigma,v);
        }
        return true;
    };
    auto apply = [&](const double* v,double* output) {
        for(int j=0;j<n;++j)output[j]=sigma*v[j];
        std::fill(bmv.begin(),bmv.end(),0.);
        p.P.add(v,output);
        p.A.add(v,bmv.data());
        for(int j=0;j<m;++j)bmv[j]/=rinv[j];
        p.A.add_transpose(bmv.data(),output);
    };
    auto cg = [&](double relative) {
        for(int i=0;i<n;++i)g[i]=rhs[i];
        for(int i=0;i<m;++i)bmv[i]=rhs[n+i]/rinv[i];
        p.A.add_transpose(bmv.data(),g.data());
        apply(ans.data(),mult.data());
        for(int j=0;j<n;++j) {
            rvec[j]=g[j]-mult[j];dvec[j]=inv_diag[j]*rvec[j];
        }
        const double tol=std::clamp(.026*std::pow(relative,1.22),2e-13,.015)*
            std::max(1e-6,norm(g));
        double rr=dot(rvec,dvec);
        int maxiters=relative>.02?40:70;
        for(int iter=0;iter<maxiters && norm(rvec)>tol && rr>0.;++iter) {
            if(iter%8==0 && clock_time()>until)return false;
            apply(dvec.data(),mult.data());
            double den=dot(dvec,mult);
            if(!(den>0.))break;
            double alpha=rr/den;
            for(int j=0;j<n;++j) {
                ans[j]+=alpha*dvec[j];rvec[j]-=alpha*mult[j];
                pot[j]=inv_diag[j]*rvec[j];
            }
            double next=dot(rvec,pot),beta=next/rr;
            rr=next;
            for(int j=0;j<n;++j)dvec[j]=pot[j]+beta*dvec[j];
        }
        std::fill(ans.begin()+n,ans.end(),0.);
        p.A.add(ans.data(),ans.data()+n);
        for(int j=0;j<m;++j)ans[n+j]=(ans[n+j]-rhs[n+j])/rinv[j];
        return finite(ans);
    };
    double rho=.15;
    if(!factor(rho))return false;
    const double qmax=norm(p.q,n),bmax=norm(p.b,m);
    double best=inf,lastp=1.;
    int lastpol=-50;
    for(int it=0;it<4000;++it) {
        if((it&7)==0 && clock_time()>until)break;
        for(int j=0;j<n;++j)rhs[j]=sigma*xb[j]-p.q[j];
        for(int j=0;j<m;++j)rhs[n+j]=y[j]-rinv[j]*z[j];
        if(!(sys?sys->solve(rhs,ans,0):cg(lastp)))return false;
        constexpr double over=1.62;
        for(int j=0;j<n;++j)xb[j]+=over*(ans[j]-xb[j]);
        for(int j=0;j<m;++j) {
            double t=y[j]+rinv[j]*(over*ans[n+j]+(1.-over)*z[j]);
            y[j]=eq[j]?p.b[j]:std::min(t,p.b[j]);
            z[j]=(t-y[j])/rinv[j];
        }
        if(it==11 || (it+1)%24==0) {
            std::fill(ax.begin(),ax.end(),0.);
            std::fill(px.begin(),px.end(),0.);
            std::fill(at.begin(),at.end(),0.);
            p.A.add(xb.data(),ax.data());
            p.P.add(xb.data(),px.data());
            p.A.add_transpose(z.data(),at.data());
            double rp=0,rd=0;
            for(int j=0;j<m;++j)rp=std::max(rp,std::abs(ax[j]-y[j]));
            for(int j=0;j<n;++j)rd=std::max(rd,std::abs(px[j]+p.q[j]+at[j]));
            rp/=1+std::max({norm(ax),norm(y),bmax});
            rd/=1+std::max({norm(px),norm(at),qmax});
            double merit=std::max(rp,rd);
            for(int j=0;j<m;++j)s[j]=p.b[j]-y[j];
            if(merit<2e-7 && accept(xb,s,z))return true;
            if(merit<std::min(.035,best*.28) && it-lastpol>=35 && clock_time()<until) {
                best=merit;lastpol=it;
                if(poly_polish(p,xb,z,until,accept,4))return true;
            }
            if((it+1)%96==0) {
                for(int j=0;j<n;++j)dx[j]=xb[j]-previousx[j];
                for(int j=0;j<m;++j)dz[j]=z[j]-previousz[j];
                if(merit>1e-6 && ray(dx,dz))return true;
                previousx=xb;previousz=z;
                if(merit>1e-8 && rp>0 && rd>0) {
                    double newrho=std::clamp(rho*std::sqrt(rp/rd),1e-6,1e6);
                    if(newrho<.15*rho || newrho>6*rho) {
                        if(!factor(newrho))return false;
                        rho=newrho;
                    }
                }
            }
            lastp=merit;
        }
    }
    return lastp<.02 && poly_polish(p,xb,z,until,accept,3);
}

} // namespace conic
