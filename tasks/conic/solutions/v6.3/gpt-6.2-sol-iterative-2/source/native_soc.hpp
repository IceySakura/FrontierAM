#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace conic {
class Soc {
public:
    using Svd=void(*)(const char*,const char*,const int*,const int*,double*,const int*,
                       double*,double*,const int*,double*,const int*,double*,const int*,int*);
    int a,d,width=0,offset=0;
    std::vector<double> V,R,w,u,lam;
    double eta=1.,w0=1.,dl=1.;
    bool exp=false;
    Svd svd=nullptr;
    Soc(int start,int dim,bool exponential=false,void* f=nullptr)
        : a(start),d(dim),R(d*d),w(d-1),u(d-1),lam(d),
          exp(exponential),svd(reinterpret_cast<Svd>(f)) {}
    bool setup(const double* s,const double* z,double* scale,double mu) {
        if (exp) return exp_scale(s,z,scale,mu);
        double nr=0.,nz=0.,ip=s[0]*z[0];
        for (int j=1;j<d;++j) {
            nr+=s[j]*s[j]; nz+=z[j]*z[j]; ip+=s[j]*z[j];
        }
        nr=std::sqrt(nr); nz=std::sqrt(nz);
        const double rs=std::sqrt((s[0]-nr)*(s[0]+nr));
        const double rz=std::sqrt((z[0]-nz)*(z[0]+nz));
        if (!(rs>0.) || !(rz>0.) || s[0]<0. || z[0]<0.) return false;
        const double eta2=rs/rz;
        eta=std::sqrt(eta2);
        dl=rs*rz;
        const double d0=std::sqrt(2.*(1.+std::max(0.,ip)/dl));
        double wn=0.;
        for (int j=1;j<d;++j) {
            w[j-1]=(s[j]/rs-z[j]/rz)/d0;
            wn+=w[j-1]*w[j-1];
        }
        w0=std::sqrt(1.+wn);
        wn=std::sqrt(wn);
        std::fill(u.begin(),u.end(),0.);
        u[0]=1.;
        if (wn>0.)
            for (int j=0;j<d-1;++j) u[j]=w[j]/wn;
        constexpr double e=0.7071067811865475244;
        std::fill(R.begin(),R.end(),0.);
        R[0]=R[d]=e;
        for (int i=1;i<d;++i) {
            R[i]=e*u[i-1];
            R[d+i]=-R[i];
        }
        u[0]+=u[0]>=0.?1.:-1.;
        double us=0.;
        for (double x:u) us+=x*x;
        for (int i=2;i<d;++i)
            for (int j=1;j<d;++j)
                R[i*d+j]=(i==j?1.:0.)-2.*u[i-1]*u[j-1]/us;
        const double h=(w0+wn)*(w0+wn);
        scale[0]=1./std::max(5.e-17,eta2*h);
        scale[1]=1./std::max(5.e-17,eta2/h);
        for (int i=2;i<d;++i) scale[i]=1./std::max(5.e-17,eta2);
        half(z,lam.data(),false);
        return std::isfinite(h+eta2) && lam[0]>0.;
    }
    bool exp_scale(const double* s,const double* z,double* scale,double mu) {
        const double a=-z[0],c=z[2],v=s[1],w0=s[2];
        if (!svd || !(a>0.) || !(c>0.) || !(v>0.) || !(w0>0.)) return false;
        const double L=std::log(c)-std::log(a);
        const double diff=std::log(w0)-std::log(v);
        const double ph=z[1]+a*(1.+L), dp=diff-s[0]/v;
        if (!(ph>0.) || !(dp>0.)) return false;
        double J[3]={-L/ph,1./ph,a/(c*ph)};
        const double g[3]={J[0]-1./a,J[1],J[2]+1./c};
        std::copy(g,g+3,lam.data());
        double h=dp<2.?dp*.5:dp-std::log1p(dp);
        for (int k=0;k<4;++k) h-=(h+std::log1p(h)-dp)/(1.+1./(1.+h));
        const double g1=1./(v*h);
        const double gd[3]={-g1,g1*(2.*h-1.+s[0]/v),(1.+h)/(w0*h)};
        const double e=L+diff;
        const double pp=std::abs(e)<1.e-3?e*e*(.5+e*(1./6.+e/24.)):std::expm1(e)-e;
        const double sz=a*v*(dp+pp)+v*ph;
        if (!(sz>0.) || !std::isfinite(sz)) return false;
        double H[12]{};
        bool got=false;
        double ds[3],dz[3], dd=0., dpd=0.;
        for (int i=0;i<3;++i) {
            ds[i]=g[i]-3.*s[i]/sz; dz[i]=gd[i]-3.*z[i]/sz;
            dd+=ds[i]*dz[i];
            dpd+=g[i]*gd[i];
        }
        double n[3]={z[1]*dz[2]-z[2]*dz[1],z[2]*dz[0]-z[0]*dz[2],
                    z[0]*dz[1]-z[1]*dz[0]};
        const double nn=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
        if (nn>1.e-12 && dd>1.e-11/sz && dpd>0.) {
            for (double& x:n) x/=nn;
            const double r=ph*n[1];
            const double ua=a*(n[0]-J[0]*r),uc=c*(n[2]-J[2]*r);
            const double us=(a/(ph+2.*a))*(ua+uc);
            const double aa=a*(ua-us),cc=c*(uc-us),bb=ph*(r-J[0]*aa-J[2]*cc);
            const double t=n[0]*aa+n[1]*bb+n[2]*cc;
            if (t>0.) {
                const double fl=std::sqrt(std::sqrt(sz/dpd)/t);
                const double f1=1./std::sqrt(sz),f2=1./std::sqrt(dd);
                if (std::isfinite(fl+f1+f2)) {
                    got=true;
                    for (int j=0;j<3;++j) {
                        H[j]=s[j]*f1;
                        H[3+j]=ds[j]*f2;
                        H[6+j]=fl*n[j];
                    }
                }
            }
        }
        if (!got) {
            const double f=std::sqrt(mu*a/ph);
            H[0]=f/a; H[2]=f/c;
            for (int i=0;i<3;++i) {
                const double ea=i==0?1./a:0., ec=i==2?1./c:0.;
                H[3+i]=std::sqrt(mu*.5)*(J[i]+ea);
                H[6+i]=std::sqrt(mu/6.)*(-J[i]+ea+2.*ec);
                H[9+i]=s[i]/std::sqrt(sz);
            }
        }
        for (double a:H) if (!std::isfinite(a)) return false;
        const char u='A', no='N';
        const int iii=3,jjj=4,one=1,lw=64;
        double eig[3]{},work[64]{},nothing=0.;
        int flag=0;
        svd(&u,&no,&iii,&jjj,H,&iii,eig,R.data(),&iii,&nothing,&one,work,&lw,&flag);
        if (flag!=0) return false;
        for (int i=0;i<3;++i) scale[i]=1./std::max(6.e-18,eig[i]*eig[i]);
        return true;
    }
    void rotate_matrix(std::vector<double>& arr) const {
        if (!width) return;
        for (int i=0;i<d;++i) {
            double* data=arr.data()+offset+static_cast<std::size_t>(i)*width;
            std::fill(data,data+width,0.);
            for (int j=0;j<d;++j) {
                const double v=R[i*d+j];
                const double* coef=V.data()+static_cast<std::size_t>(j)*width;
                for (int k=0;k<width;++k) data[k]+=v*coef[k];
            }
        }
    }
    void rotate(double* v,bool transpose) const {
        // The native path only accepts small SOC blocks.
        double res[24]{};
        for (int i=0;i<d;++i)
            for (int j=0;j<d;++j)
                res[i]+=(transpose?R[j*d+i]:R[i*d+j])*v[j];
        std::copy(res,res+d,v);
    }
    void half(const double* x,double* y,bool inv) const {
        const double sign=inv?-1.:1.;
        const double f=inv?1./eta:eta;
        double c=0.;
        for (int j=1;j<d;++j) c+=x[j]*w[j-1];
        y[0]=(w0*x[0]+sign*c)*f;
        const double t=sign*x[0]+c/(1.+w0);
        for (int j=1;j<d;++j) y[j]=(x[j]+t*w[j-1])*f;
    }
    void correct(double target,double wt,const double* ds,const double* dz,double* bz) const {
        if (exp) {
            for (int i=0;i<3;++i) bz[i]-=target*lam[i];
            return;
        }
        double ds1[24]{},dz1[24]{},c[24]{},g[24]{};
        half(ds,ds1,true);
        half(dz,dz1,false);
        double ip=0.,vc=0.;
        for (int j=0;j<d;++j) ip+=ds1[j]*dz1[j];
        c[0]=target-wt*ip;
        for (int j=1;j<d;++j) {
            c[j]=-wt*(ds1[0]*dz1[j]+dz1[0]*ds1[j]);
            vc+=lam[j]*c[j];
        }
        c[0]=(lam[0]*c[0]-vc)/dl;
        for (int j=1;j<d;++j) c[j]=(c[j]-lam[j]*c[0])/lam[0];
        half(c,g,false);
        for (int j=0;j<d;++j) bz[j]-=g[j];
    }
    double step(const double* x,const double* dx,double top,bool dual=false) const {
        if (exp) {
            auto valid=[&](double f) {
                double a=x[0]+f*dx[0],b=x[1]+f*dx[1],c=x[2]+f*dx[2];
                if (dual) {std::swap(a,b); a=-a; b=-b;}
                return b>0. && c>0. && b*(std::log(c)-std::log(b)+(dual?1.:0.))>a;
            };
            if (valid(top)) return top;
            double lo=0.,hi=top;
            for (int it=0;it<24;++it) {
                const double mid=.5*(lo+hi);
                if (valid(mid)) lo=mid;
                else hi=mid;
            }
            return lo;
        }
        double sum=0.;
        for (int j=1;j<d;++j) {
            const double y=x[j]+top*dx[j]; sum+=y*y;
        }
        if (x[0]+top*dx[0]>std::sqrt(sum)) return top;
        double nv=0.,nd=0.,cp=0.;
        for (int j=1;j<d;++j) {
            nv+=x[j]*x[j]; nd+=dx[j]*dx[j]; cp+=x[j]*dx[j];
        }
        nv=std::sqrt(nv);
        const double c=(x[0]-nv)*(x[0]+nv);
        const double b=x[0]*dx[0]-cp;
        const double a=dx[0]*dx[0]-nd;
        const double root=std::sqrt(std::max(0.,b*b-a*c));
        const double tp=b<=0.? c/(root-b):(-b-root)/a;
        if (!(tp>=0.) || !std::isfinite(tp)) return 0.;
        return std::min(top,tp);
    }
};
}
