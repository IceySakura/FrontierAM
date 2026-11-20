#pragma once
#include "blas.hpp"

namespace conic {

inline long double exp_margin(long double x, long double y, long double w) {
    return y > 0 && w > 0 ? y * (std::log(w) - std::log(y)) - x : -inf;
}

inline double exp_defect(double x, double y, double w) {
    double boundary = std::max({std::abs(y), x, -w});
    if (y > 0 && w > 0)
        return std::max(0.,std::min(boundary, x-y*(std::log(w)-std::log(y))));
    return boundary;
}

inline double cone_defect(const Problem& p, const Vec& x, bool dual,
                          double stop = inf) {
    double worst = 0;
    Vec mat, work, eig;
    Int iw;
    for (const auto& c:p.cones) {
        const double* v = x.data()+c.off;
        double a, scale;
        switch(c.kind) {
        case Zero:
            a = dual ? 0 : std::min(1.,norm(v,c.width));
            break;
        case Linear:
            a = -*std::min_element(v,v+c.width)/std::max(1.,norm(v,c.width));
            break;
        case Soc:
            scale = std::max(1.,norm(v,c.width));
            a = 0;
            for (int k=1;k<c.width;++k) {
                double u = v[k]/scale;
                a += u*u;
            }
            a = std::sqrt(a)-v[0]/scale;
            break;
        case Exp:
            scale = std::max(1.,norm(v,3));
            a = dual ? exp_defect(-v[1]/scale,-v[0]/scale,(v[2]/scale)*std::exp(1.))
                     : exp_defect(v[0]/scale,v[1]/scale,v[2]/scale);
            break;
        default:
            scale = std::max(1.,norm(v,c.width));
            mat.resize(size_t(c.dim)*c.dim);
            smat(c.dim,v,mat.data(),1./scale);
            if(std::isfinite(stop) && c.dim>=5) {
                for(int j=0;j<c.dim;++j)mat[j*(c.dim+1)]+=.985*stop;
                if(chol(c.dim,mat.data())) {a=0.;break;}
                smat(c.dim,v,mat.data(),1./scale);
            }
            a = -smallest_eigen(c.dim,mat.data(),work,iw,eig);
            break;
        }
        if (!std::isfinite(a)) return inf;
        worst = std::max(worst,a);
        if (worst>stop) break;
    }
    return worst;
}

inline long double soc_inner(const double* x, const double* y, int n) {
    long double t = (long double)x[0]*y[0];
    for (int i=1;i<n;++i) t -= (long double)x[i]*y[i];
    return t;
}

inline double soc_step(const double* x, const double* d, int n) {
    long double c = soc_inner(x,x,n);
    if (!(c>0) || !(x[0]>0)) return 0;
    long double b = soc_inner(x,d,n), a = soc_inner(d,d,n);
    if (x[0]+d[0]>0 && a+2*b+c>0) return 1;
    const long double disc = std::max((long double)0, b*b-a*c);
    long double root = std::sqrt(disc);
    if (b<0 && root-b>0) return std::clamp(double(c/(root-b)),0.,1.);
    if (a<0) return std::clamp(double((root+b)/(-a)),0.,1.);
    return (d[0]<0) ? std::clamp(-x[0]/d[0],0.,1.) : 1.;
}

struct PSDScale {
    int d;
    Vec g, gi, w, wi, lam;
    Vec a, b, u, v, tmp, tmp2, work, eig;
    Int iw;
    explicit PSDScale(int dim):d(dim) {
        size_t z=size_t(d)*d;
        g.resize(z); gi.resize(z); w.resize(z); wi.resize(z); lam.resize(d);
        a.resize(z); b.resize(z); u.resize(z); v.resize(z); tmp.resize(z); tmp2.resize(z);
        work.resize(size_t(std::max(48,8+3*d))*d);
    }
    bool update(const double* s, const double* z) {
        smat(d,s,a.data()); smat(d,z,b.data());
        if (!chol(d,a.data()) || !chol(d,b.data())) return false;
        for(int j=0;j<d;++j)
            std::fill(a.begin()+j*d,a.begin()+j*d+j,0.);
        tmp=a;
        const double one=1;
        scipy_dtrmm_("L","L","T","N",&d,&d,&one,b.data(),&d,tmp.data(),&d);
        int lwork=int(work.size()), info=0;
        scipy_dgesvd_("A","A",&d,&d,tmp.data(),&d,lam.data(),u.data(),&d,v.data(),&d,
                      work.data(),&lwork,&info);
        if(info!=0 || !(lam.back()>0)) return false;
        gemm("N","T",d,d,d,a.data(),d,v.data(),d,g.data(),d);
        scipy_dtrmm_("L","L","N","N",&d,&d,&one,b.data(),&d,u.data(),&d);
        for(int j=0;j<d;++j) {
            double scale=1./std::sqrt(lam[j]);
            for(int i=0;i<d;++i) {
                g[i+j*d]*=scale;
                gi[j+i*d]=u[i+j*d]*scale;
            }
        }
        gram(d,g.data(),w.data());
        gram(d,gi.data(),wi.data(),true);
        return finite(w) && finite(wi);
    }
    void apply(const double* in, double* out, bool inverse=false) {
        smat(d,in,a.data());
        congruence(d,(inverse?wi:w).data(),a.data(),b.data(),tmp.data());
        svec(d,b.data(),out);
    }
    void correct(double* out, const double* ds, const double* dz, double center, double damp) {
        smat(d,ds,a.data()); smat(d,dz,b.data());
        congruence(d,gi.data(),a.data(),u.data(),tmp.data());
        congruence(d,g.data(),b.data(),v.data(),tmp.data(),true);
        gemm("N","N",d,d,d,u.data(),d,v.data(),d,tmp.data(),d);
        for (int j=0;j<d;++j)
            for (int i=0;i<d;++i)
                a[i+j*d] = ((i==j ? 2*center : 0)-damp*(tmp[i+j*d]+tmp[j+i*d]))/(lam[i]+lam[j]);
        congruence(d,g.data(),a.data(),b.data(),tmp.data());
        svec(d,b.data(),tmp2.data()); // tmp2 has at least d*(d+1)/2 entries.
        for(int k=0;k<d*(d+1)/2;++k) out[k]+=tmp2[k];
    }
    double step(const double* v0, const double* dv, double upper, bool dual) {
        // Test the proposed step first. It avoids an eigensolve on interior steps.
        smat(d,v0,a.data());
        smat(d,dv,b.data());
        for (int k=0;k<d*d;++k) u[k] = a[k]+upper*b[k];
        if (chol(d,u.data())) return upper;
        const Vec& basis=dual ? g : gi;
        congruence(d,basis.data(),b.data(),u.data(),tmp.data(),dual);
        for (int j=0;j<d;++j)
            for (int i=0;i<d;++i) u[i+j*d] /= std::sqrt(lam[i])*std::sqrt(lam[j]);
        double e=smallest_eigen(d,u.data(),work,iw,eig);
        return e < -1./upper ? -1./e : upper;
    }
};

struct ExpScale {
    using L = long double;
    std::array<L,3> coord{}, gradpsi{}, sh{};
    std::array<L,9> hpsi{}, hess{};
    std::array<double,9> h{};
    L psi=0;
    double mu=1.;
    bool second_secant(const double* s,const double* z,L sz) {
        if(!(s[1]>0 && s[2]>0))return false;
        const L y=s[1],w=s[2];
        const L margin=std::log(w)-std::log(y)-(L)s[0]/y;
        if(!(margin>0))return false;
        L alpha=margin>L(1) ? margin-std::log1p(margin) : margin/2;
        for(int i=0;i<5;++i)
            alpha-=(alpha+std::log1p(alpha)-margin)/(1+1/(1+alpha));
        if(!(alpha>0))return false;
        const L r=1/(y*alpha),c=(1+1/alpha)/w, t=3/sz;
        const L zbar[3]={-r,1/y-r*(1+std::log(c)-std::log(r)),c};
        L u[3],v[3],zs[3],us[3], ds[3], K[9], cross[3];
        L uv=0.;
        for(int i=0;i<3;++i) {
            ds[i]=std::sqrt(hess[4*i]);
            u[i]=zbar[i]-t*z[i];v[i]=sh[i]-t*s[i];
            zs[i]=ds[i]*z[i];us[i]=ds[i]*u[i];
            uv+=u[i]*v[i];
        }
        if(!(uv*sz>3e-11L) || !std::isfinite(uv))return false;
        for(int i=0;i<3;++i)cross[i]=zs[(i+1)%3]*us[(i+2)%3]-zs[(i+2)%3]*us[(i+1)%3];
        const L n=std::sqrt(cross[0]*cross[0]+cross[1]*cross[1]+cross[2]*cross[2]);
        if(!(n>0) || !std::isfinite(n))return false;
        for(int i=0;i<3;++i)cross[i]/=n;
        for(int j=0;j<3;++j)for(int i=j;i<3;++i)K[i+j*3]=hess[i+j*3]/ds[i]/ds[j];
        L wbar[3], q=0.;
        for(int i=0;i<3;++i) {
            for(int j=0;j<i;++j)K[i+3*i]-=K[i+3*j]*K[i+3*j];
            if(!(K[i+3*i]>2e-18L))return false;
            K[i+3*i]=std::sqrt(K[i+3*i]);
            for(int j=i+1;j<3;++j) {
                for(int k=0;k<i;++k)K[j+3*i]-=K[j+3*k]*K[i+3*k];
                K[j+3*i]/=K[i+3*i];
            }
            wbar[i]=cross[i];
            for(int j=0;j<i;++j)wbar[i]-=K[i+3*j]*wbar[j];
            wbar[i]/=K[i+3*i];q+=wbar[i]*wbar[i];
        }
        L weight=(sz/3)/q;
        std::array<double,9> h2;
        for(int j=0;j<3;++j)for(int i=0;i<3;++i) {
            h2[i+3*j]=double((L)s[i]*s[j]/sz+v[i]*v[j]/uv+weight*(cross[i]*ds[i])*(cross[j]*ds[j]));
            if(!std::isfinite(h2[i+3*j]))return false;
        }
        h=h2;
        return true;
    }

    bool update(const double* s, const double* z, double beta, bool secant,bool two_secant) {
        const L r=-z[0], y=z[1], w=z[2];
        if (!(r>0 && w>0)) return false;
        const L logwr=std::log(w)-std::log(r);
        psi=y+r*(1+logwr);
        if (!(psi>0)) return false;
        coord={r,y,w};
        gradpsi={-logwr,1,r/w};
        hpsi={-1/r,0,-1/w, 0,0,0, -1/w,0,-r/(w*w)};
        sh={-1/r+gradpsi[0]/psi,gradpsi[1]/psi,1/w+gradpsi[2]/psi};
        const L sz = (L)s[0]*z[0]+(L)s[1]*z[1]+(L)s[2]*z[2];
        if (!(sz>0)) return false;
        mu=beta;
        for(int j=0;j<3;++j) for(int i=0;i<3;++i) {
            const int k=i+j*3;
            L a=gradpsi[i]*gradpsi[j]/(psi*psi)-hpsi[k]/psi;
            if(i==j && i!=1) a+=1/(coord[i]*coord[i]);
            hess[k]=a;
            h[k]=double(secant ? sz/3*(a-sh[i]*sh[j]/3)+(L)s[i]*s[j]/sz : beta*a);
            if (!std::isfinite(h[k])) return false;
        }
        if(secant && two_secant)second_secant(s,z,sz);
        return true;
    }
    void correct(double* out, const double* ds, const double* dz, double center, double damp) const {
        for(int i=0;i<3;++i) out[i]+=center*double(sh[i]);
        if(damp==0) return;
        const L r=coord[0], w=coord[2];
        L R[9], d[3], v[3], ww[3], a[3], scale[3];
        for(int i=0;i<3;++i) scale[i]=1/std::sqrt(hess[i+3*i]);
        for(int j=0;j<3;++j) for(int i=j;i<3;++i) R[i+3*j]=hess[i+3*j]*scale[i]*scale[j];
        for(int i=0;i<3;++i) {
            for(int j=0;j<i;++j) R[i+3*i]-=R[i+3*j]*R[i+3*j];
            if(!(R[i+3*i]>1e-19L)) return;
            R[i+3*i]=std::sqrt(R[i+3*i]);
            for(int j=i+1;j<3;++j) {
                for(int k=0;k<i;++k) R[j+3*i]-=R[j+3*k]*R[i+3*k];
                R[j+3*i]/=R[i+3*i];
            }
            d[i]=scale[i]*ds[i];
            for(int j=0;j<i;++j) d[i]-=R[i+3*j]*d[j];
            d[i]/=R[i+3*i];
        }
        for(int i=2;i>=0;--i) {
            for(int j=i+1;j<3;++j) d[i]-=R[j+3*i]*d[j];
            d[i]/=R[i+3*i];
            v[i]=d[i]*scale[i];
        }
        const L ud=gradpsi[0]*dz[0]+gradpsi[1]*dz[1]+gradpsi[2]*dz[2];
        const L uv=gradpsi[0]*v[0]+gradpsi[1]*v[1]+gradpsi[2]*v[2];
        L bil=0;
        for(int i=0;i<3;++i) {
            a[i]=ww[i]=0;
            for(int j=0;j<3;++j) {
                a[i]+=hpsi[i+3*j]*dz[j];
                ww[i]+=hpsi[i+3*j]*v[j];
            }
            bil+=a[i]*v[i];
        }
        // The only nonzero third derivatives are 000, 022 and 222.
        const L trz[3]={-dz[0]*v[0]/(r*r)+dz[2]*v[2]/(w*w), 0,
            (dz[0]*v[2]+dz[2]*v[0])/(w*w)+2*r/w/w/w*dz[2]*v[2]};
        for(int i=0;i<3;++i) {
            L t = -trz[i]/psi+(a[i]*uv+ww[i]*ud+gradpsi[i]*bil)/(psi*psi)
                - 2*gradpsi[i]*ud*uv/(psi*psi*psi);
            if(i==0) t+=2*dz[0]*v[0]/(r*r*r);
            if(i==2) t-=2*dz[2]*v[2]/(w*w*w);
            double dt=double(.5*damp*t);
            if(std::isfinite(dt)) out[i]+=dt;
        }
    }
};

inline double exp_step(const double* x,const double* dx,double upper,bool dual) {
    auto inside = [&](double a) {
        const double u=x[0]+a*dx[0],v=x[1]+a*dx[1],w=x[2]+a*dx[2];
        if(w>0 && (dual?u<0:v>0)) {
            const double r=dual?-u:v;
            const double l=std::log(w)-std::log(r);
            const double a=dual?v+r*(1.+l):r*l-u;
            const double tol=4e-14*(std::abs(dual?v:u)+r*(1+std::abs(l)));
            if(std::abs(a)>tol)return a>0;
        } else if(w<0 || (!dual && v<0) || (dual && u>0))return false;
        auto t = [&](int j){return (long double)x[j]+(long double)a*dx[j];};
        if(dual) {
            long double r=-t(0), w=t(2);
            return r>0 && w>0 && t(1)+r*(1+std::log(w)-std::log(r))>0;
        }
        return exp_margin(t(0),t(1),t(2))>0;
    };
    if (inside(upper)) return upper;
    double lo=0, hi=upper;
    for(int k=0;k<25;++k) {
        double mid=.5*(lo+hi);
        if(inside(mid)) lo=mid; else hi=mid;
    }
    return lo;
}

}  // namespace conic
