#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

extern "C" {
void scipy_dgemm_(const char*, const char*, const int*, const int*, const int*, const double*,
                  const double*, const int*, const double*, const int*, const double*, double*, const int*);
void scipy_dpotrf_(const char*, const int*, double*, const int*, int*);
void scipy_dgesvd_(const char*, const char*, const int*, const int*, double*, const int*, double*,
                   double*, const int*, double*, const int*, double*, const int*, int*);
void scipy_dgesdd_(const char*,const int*,const int*,double*,const int*,double*,double*,const int*,
                   double*,const int*,double*,const int*,int*,int*);
void scipy_dtrsm_(const char*,const char*,const char*,const char*,const int*,const int*,
                  const double*,const double*,const int*,double*,const int*);
void scipy_dsyev_(const char*, const char*, const int*, double*, const int*, double*, double*, const int*, int*);
void scipy_dsyevr_(const char*,const char*,const char*,const int*,double*,const int*,const double*,
                   const double*,const int*,const int*,const double*,int*,double*,double*,const int*,
                   int*,double*,const int*,int*,const int*,int*);
void scipy_dsyrk_(const char*,const char*,const int*,const int*,const double*,const double*,
                  const int*,const double*,double*,const int*);
}

namespace {
using Vec = std::vector<double>;
constexpr double SQ2 = 1.4142135623730950488;
constexpr double ISQ2 = 0.7071067811865475244;
constexpr double E = 2.7182818284590452354;
constexpr double INF = std::numeric_limits<double>::infinity();
enum { ZERO, NONNEG, SOC, PSD, EXP };

double infnorm(const double* p, int n) {
    double v = 0;
    for (int i=0; i<n; ++i) {
        if (!std::isfinite(p[i])) return INF;
        v = std::max(v, std::abs(p[i]));
    }
    return v;
}
double dot(const double* a, const double* b, int n) {
    double d = 0;
    for (int i=0; i<n; ++i) d += a[i]*b[i];
    return d;
}
double norm2(const double* a, int n) {
    double d=dot(a,a,n);
    if (d>0 && std::isfinite(d)) return std::sqrt(d);
    double nm=infnorm(a,n);
    if (nm==0 || !std::isfinite(nm)) return nm;
    d=0;
    for (int i=0; i<n; ++i) { double w=a[i]/nm; d+=w*w; }
    return nm*std::sqrt(d);
}
void gemm(int n, const double* a, const double* b, double* c,
          char ta='N', char tb='N', double alpha=1, double beta=0) {
    scipy_dgemm_(&ta,&tb,&n,&n,&n,&alpha,a,&n,b,&n,&beta,c,&n);
}
bool chol(double* a, int n) {
    int info;
    scipy_dpotrf_("L",&n,a,&n,&info);
    return info==0;
}
void lower(double* a, int n) {
    for(int j=0;j<n;++j) std::fill(a+j*n,a+j*n+j,0);
}
double exp_defect(double a, double b, double c) {
    double face=std::max({std::abs(b), a, -c});
    if(b>0 && c>0) return std::min(face,std::max(0.0,a-b*(std::log(c)-std::log(b))));
    return face;
}
void project_exp(double* x,double& hint,double* dual=nullptr) {
    double sc=infnorm(x,3);
    if(dual)std::fill(dual,dual+3,0.);
    if(sc==0 || !std::isfinite(sc)) return;
    double a=x[0]/sc,b=x[1]/sc,c=x[2]/sc;
    if(exp_defect(a,b,c)<=4e-16) return;
    double f0=std::min(a,0.),f2=std::max(c,0.);
    auto floor_dual=[&](){
        if(dual && dual[0]<0) {
            double floor=std::numeric_limits<double>::min()*std::max({1.,-dual[0],std::abs(dual[1])});
            dual[2]=std::max(dual[2],floor);
        }
    };
    auto face=[&](){
        double normal_w=f2-c, normal_u=a-f0;
        if(normal_u>0) {
            double need=std::exp(std::log(normal_u)+b/normal_u-1);
            // The curved projection can be closer to this face than one ulp.
            if(need<=normal_w+2e-13)normal_w=std::max(normal_w,need);
        }
        if(dual) {
            dual[0]=-normal_u*sc;dual[1]=-b*sc;dual[2]=normal_w*sc;
            floor_dual();
        }
        x[0]=f0*sc;x[1]=0;x[2]=std::max(0.,c+normal_w)*sc;
    };
    if(exp_defect(b,a-f0,E*(f2-c))<=4e-16){
        face();return;
    }
    if(b>0 && a < -700*b && c<=1e-295) {
        x[2]=std::max(x[2],std::max(std::numeric_limits<double>::min(),1e-303*sc));
        if(dual)dual[2]=std::max(0.,-c*sc);
        return;
    }
    double lo=-745,hi=745;
    if(b>0) hi=std::min(hi,a/b);
    if(b<0) lo=std::max(lo,a/b);
    if(a>0) lo=std::max(lo,1-b/a);
    if(a<0) hi=std::min(hi,1-b/a);
    if(!(lo<hi)){
        face();return;
    }
    double t=std::clamp(hint,lo,hi),u=0,v=0,w=0,la=0,root=t;
    for(int it=0;it<44;++it){
        root=t;
        double d=t*(t-1)+1,h=2*t-1;
        v=std::fma(a,t-1,b)/d;
        la=std::fma(-b,t,a)/d;
        double vp=(a-h*v)/d,lp=(-b-h*la)/d;
        double ex=std::exp(-std::abs(t)), f,df;
        if(t>=0) {
            f=v-ex*(c+ex*la);
            df=vp+ex*(c+ex*(2*la-lp));
            w=std::max(0.,c+la*ex);
        } else {
            f=ex*(ex*v-c)-la;
            df=ex*((vp+2*v)*ex-c)-lp;
            w=std::max(0.,v*ex);
        }
        u=t*v;
        if(f<0) lo=t; else hi=t;
        double t1=t-f/df;
        if(std::abs(t1-t)<=4e-14*(1+std::abs(t)) || hi-lo<=4e-14*(1+std::abs(t))) break;
        if(!(t1>lo && t1<hi)) t1=0.5*(lo+hi);
        t=t1;
    }
    t=root;
    if(t>=0 && la>=-1e-14) {
        // Reconstruct from w on the steep side, where v may be obtained
        // as the difference of nearly equal terms in the root equation.
        v=w*std::exp(-t);
        u=t*v;
    }
    double dist=(u-a)*(u-a)+(v-b)*(v-b)+(w-c)*(w-c);
    double df=(f0-a)*(f0-a)+b*b+(f2-c)*(f2-c);
    if(v>=0 && la>=-1e-14 && dist<df+4e-15){
        x[0]=u*sc;x[1]=v*sc;x[2]=w*sc;hint=t;
        if(x[1]>0 && t<0)
            x[2]=std::max(x[2],std::numeric_limits<double>::min()*std::max({1.,std::abs(x[0]),x[1]}));
        if(dual){
            if(t>=0) {
                double h=std::max(0.,la)*sc;
                dual[0]=-h;dual[1]=(t-1)*h;dual[2]=std::exp(-t)*h;
            } else {
                double e=std::exp(t),z=std::max(0.,w-c)*sc;
                dual[0]=-e*z;dual[1]=(t-1)*e*z;dual[2]=z;
            }
            floor_dual();
        }
    } else {
        face();
    }
}

struct ExpDual {
    double v, w, r, l, g[3], H[9], shadow[3];
    bool set(const double* z) {
        v=-z[0]; w=z[2];
        if(!(v>0 && w>0)) return false;
        r=std::log(w)-std::log(v);
        l=std::fma(v,1+r,z[1]);
        if(!(l>0)) return false;
        g[0]=-r; g[1]=1; g[2]=v/w;
        for(int i=0;i<3;++i){
            shadow[i]=g[i]/l;
            for(int j=0;j<3;++j) H[i+3*j]=(g[i]/l)*(g[j]/l);
        }
        shadow[0]-=1/v; shadow[2]+=1/w;
        H[0]+=(1+v/l)/(v*v);
        H[2]+=1/(w*l); H[6]=H[2];
        H[8]+=(1+v/l)/(w*w);
        return std::isfinite(H[0]) && std::isfinite(H[8]) && H[0]>0 && H[8]>0;
    }
    void inverse(const double* b, double* a) const {
        // Eliminate the rank-one term using the middle coordinate.
        long double h=(long double)v/l;
        long double u0=((long double)b[0]-g[0]*b[1])*v;
        long double u2=((long double)b[2]-g[2]*b[1])*w;
        long double t=((1+h)*u0-h*u2)/(1+2*h);
        long double f=((1+h)*u2-h*u0)/(1+2*h);
        a[0]=t*v; a[2]=f*w; a[1]=(long double)l*l*b[1]-g[0]*t*v-g[2]*f*w;
    }
    void third(const double* as, const double* bz, double* out) const {
        double ax[3];
        inverse(as,ax);
        long double a0=(long double)ax[0]/v, a2=(long double)ax[2]/w;
        long double b0=(long double)bz[0]/v, b2=(long double)bz[2]/w;
        long double t=dot(g,ax,3)/(long double)l, f=dot(g,bz,3)/(long double)l;
        long double h=(long double)v/l, ai=a0+a2, bi=b0+b2;
        long double ab=-h*ai*bi;
        out[0]=(g[0]/(long double)l)*(-t*f+0.5L*ab)
          + (a0*b0+0.5L*h*(a0*b0-a2*b2-ai*f-bi*t))/v;
        out[1]=(-t*f+0.5L*ab)/l;
        out[2]=(g[2]/(long double)l)*(-t*f+0.5L*ab)
          + (-a2*b2-0.5L*h*(a0*b2+a2*b0+2*a2*b2+ai*f+bi*t))/w;
    }
};

struct Block {
    int type, d, k, start, mode, offset, aux;
    double eta=1, etainv=1, ldet=1, project_hint=0;
    bool dual_scaling=false;
    bool negative_spectrum=true;
    Vec w, lambda;  // SOC boost or spectrum of the NT scaling.
    Vec T, Ti, R, Ri, Ls,Lz;
    Vec a,b,c,f,g,v1,v2,work,eval,svwork;
    std::vector<int> row,col,svint,ieigen,support;
    ExpDual exp;

    Block(int type_, int d_, int start_, int mode_, int offset_, int aux_)
        : type(type_), d(d_), k(type_==PSD?d_*(d_+1)/2:d_), start(start_),
          mode(mode_), offset(offset_), aux(aux_) {
        if(type==SOC) {
            w.resize(d); lambda.resize(d); v1.resize(d); v2.resize(d);
        } else if(type==PSD) {
            if(d>4500) throw std::bad_alloc();
            int dd=d*d;
            if(mode!=5) {
                T.resize(dd); Ti.resize(dd);
                if(mode==2) Ri.resize(dd); else R.resize(dd);
                f.resize(dd);g.resize(dd);
                if(d>=40){Ls.resize(dd);Lz.resize(dd);}
            }
            a.resize(dd); b.resize(dd); c.resize(dd);
            lambda.resize(d); eval.resize(d); work.resize(std::max(16,100*d));
            if(d>=40){ieigen.resize(10LL*d);support.resize(2LL*d);}
            if(mode!=5 && d>=56){
                int w=-1,info; double v{};
                svint.resize(8LL*d);
                scipy_dgesdd_("A",&d,&d,a.data(),&d,lambda.data(),f.data(),&d,g.data(),&d,&v,&w,svint.data(),&info);
                if(info==0 && v>0 && v<=60000000)svwork.resize((size_t)v);
            }
            row.resize(k); col.resize(k);
            int ix=0;
            for(int j=0;j<d;++j) for(int i=0;i<=j;++i,++ix) {row[ix]=i; col[ix]=j;}
        }
    }
    void smat(const double* x,double* y,double sc=1) const {
        for(int l=0;l<k;++l) {
            int i=row[l], j=col[l];
            y[i+j*d]=y[j+i*d]=(i==j ? sc : sc*ISQ2)*x[l];
        }
    }
    void svec(const double* x,double* y) const {
        for(int l=0;l<k;++l) {
            int i=row[l], j=col[l];
            y[l]=i==j ? x[i+j*d] : ISQ2*(x[i+j*d]+x[j+i*d]);
        }
    }
    double mineig(double* y) {
        int nwork=(int)work.size(),info;
        scipy_dsyev_("N","L",&d,y,&d,eval.data(),work.data(),&nwork,&info);
        return info==0?eval[0]:-INF;
    }
    void unit(double* s,double* z) const {
        std::fill(s,s+k,0); std::fill(z,z+k,0);
        if(type==NONNEG) { std::fill(s,s+k,1); std::fill(z,z+k,1); }
        if(type==SOC) s[0]=z[0]=1;
        if(type==PSD) for(int j=0;j<d;++j) s[j*(j+3)/2]=z[j*(j+3)/2]=1;
        if(type==EXP) {
            // A primal-dual pair on the barrier's central path.
            z[0]=-1; z[1]=1; z[2]=1;
            s[0]=-1; s[1]=0.5; s[2]=1.5;
        }
    }
    void shift(double* x,bool dual) {
        if(type==ZERO) {
            if(!dual) std::fill(x,x+k,0);
        } else if(type==NONNEG) {
            double mn=*std::min_element(x,x+k);
            double sh=std::max(0.0,1-mn);
            for(int i=0;i<k;++i) x[i]+=sh;
        } else if(type==SOC) {
            x[0]=std::max(x[0],1+norm2(x+1,d-1));
        } else if(type==PSD) {
            smat(x,a.data());
            double sh=std::max(0.0,1-mineig(a.data()));
            for(int j=0;j<d;++j) x[j*(j+3)/2]+=sh;
        } else {
            // Keep exponentials near the center initially.
            if(dual) { x[0]=-1; x[1]=1; x[2]=1; }
            else { x[0]=-1; x[1]=0.5; x[2]=1.5; }
        }
    }
    void socW(double* out,const double* in,bool inv) const {
        double r=dot(w.data()+1,in+1,d-1);
        double sign=inv?-1:1, sc=inv?etainv:eta;
        out[0]=sc*(w[0]*in[0]+sign*r);
        double u=sign*in[0]+r/(w[0]+1);
        for(int i=1;i<d;++i) out[i]=sc*(in[i]+w[i]*u);
    }
    bool prepare(const double* s, const double* z, double* diag, double* off, double* auxdiag,
                 double mu, bool pd_exp) {
        if(type==ZERO) { std::fill(diag,diag+k,0); return true; }
        if(type==NONNEG) {
            for(int i=0;i<k;++i) {
                if(!(s[i]>0 && z[i]>0)) return false;
                diag[i]=s[i]/z[i];
            }
            return true;
        }
        if(type==SOC) {
            auto det=[](const double* u,int n){
                long double v=(long double)u[0]*u[0];
                for(int i=1;i<n;++i) v-=(long double)u[i]*u[i];
                return v;
            };
            long double ss=det(s,d), zz=det(z,d);
            if(!(ss>0 && zz>0 && s[0]>0 && z[0]>0)) return false;
            long double sr=std::sqrt(ss), zr=std::sqrt(zz);
            long double prod=(long double)s[0]*z[0];
            for(int i=1;i<d;++i) prod+=(long double)s[i]*z[i];
            long double wn=std::sqrt(2*(1+std::max(1.0L,prod/(sr*zr))));
            for(int i=1;i<d;++i) w[i]=(s[i]/sr-z[i]/zr)/wn;
            w[0]=std::hypot(1,norm2(w.data()+1,d-1));
            double es=sr/zr;
            eta=std::sqrt(es); etainv=1/eta;
            if(!std::isfinite(eta) || !std::isfinite(w[0])) return false;
            ldet=sr*zr;
            lambda[0]=std::sqrt(sr*zr)*wn/2;
            long double lu=((long double)z[0]+lambda[0]/eta)/(w[0]+1);
            for(int i=1;i<d;++i) lambda[i]=eta*(z[i]+w[i]*lu);
            if(mode==3) {
                // A signed rank-two update with a quasi-definite sparse extension.
                double f=2*w[0]*w[0]-1, d0=0.5/f;
                double u=std::sqrt(f-d0);
                double h=eta*(2*w[0]/u), v=eta*std::sqrt(2*(1+d0)/(f-d0));
                std::fill(diag,diag+d,es);
                diag[0]=es*d0;
                off[0]=eta*u;
                for(int i=1;i<d;++i) {
                    off[i]=h*w[i];
                    off[d+i-1]=v*w[i];
                }
                auxdiag[0]=1; auxdiag[1]=-1;
            } else if(mode==1){
                // -H = eta^2 J - 2 eta^2 w w'.  Expose a single auxiliary pivot.
                std::fill(diag,diag+d,es);
                diag[0]=-es;
                for(int i=0;i<d;++i) off[i]=SQ2*eta*w[i];
                *auxdiag=1;
            } else {
                diag[0]=es*(2*w[0]*w[0]-1);
                for(int j=1,l=0;j<d;++j) {
                    diag[j]=es*(1+2*w[j]*w[j]);
                    for(int i=0;i<j;++i,++l) off[l]=-2*es*w[i]*w[j];
                }
            }
            return true;
        }
        if(type==EXP) {
            if(!exp.set(z)) return false;
            double sv=s[1], sw=s[2];
            if(!(sv>0 && sw>0)) return false;
            double sr=std::log(sw)-std::log(sv);
            double sl=std::fma(sv,sr,-s[0]);
            if(!(sl>0)) return false;
            double H[9], U[9]{}, sz=dot(s,z,3), local=sz/3, muscale=mu;
            if(!(local>0 && mu>0)) return false;
            dual_scaling=true;
            for(int i=0;i<9;++i) H[i]=mu*exp.H[i];
            if(pd_exp) {
                double zp[3]={-1/sl,1/sv+(sr-1)/sl,1/sw+(sv/sw)/sl};
                double u[3],v[3],cross[3],tmp[3];
                for(int i=0;i<3;++i) { u[i]=zp[i]-z[i]/local; v[i]=exp.shadow[i]-s[i]/local; }
                cross[0]=z[1]*u[2]-z[2]*u[1];
                cross[1]=z[2]*u[0]-z[0]*u[2];
                cross[2]=z[0]*u[1]-z[1]*u[0];
                double cn=norm2(cross,3), vu=dot(v,u,3);
                if(cn>1e-10*norm2(z,3)*norm2(zp,3) && vu>1e-12/local){
                    for(double& i:cross) i/=cn;
                    exp.inverse(cross,tmp);
                    double den=dot(tmp,cross,3)/local;
                    if(den>0 && std::isfinite(den)){
                        double a=1/std::sqrt(sz),b=1/std::sqrt(vu),c=1/std::sqrt(den);
                        for(int i=0;i<3;++i){ U[i]=a*s[i];U[i+3]=b*v[i];U[i+6]=c*cross[i];}
                        for(int i=0;i<3;++i) for(int j=0;j<3;++j)
                            H[i+3*j]=U[i]*U[j]+U[i+3]*U[j+3]+U[i+6]*U[j+6];
                        dual_scaling=false;
                    }
                } else {
                    muscale=local;
                    for(int i=0;i<9;++i) H[i]=local*exp.H[i];
                }
            }
            for(int i=0;i<3;++i) if(!std::isfinite(H[i+3*i]) || H[i+3*i]<=0) return false;
            if(mode==4){
                std::fill(diag,diag+3,0);
                if(dual_scaling) {
                    double rm=std::sqrt(muscale), a=rm/exp.l, b=rm*std::sqrt(exp.v/exp.l);
                    for(int i=0;i<3;++i) U[i]=a*exp.g[i];
                    U[3]=b/exp.v; U[4]=0; U[5]=b/exp.w;
                    diag[0]=(rm/exp.v)*(rm/exp.v);diag[2]=(rm/exp.w)*(rm/exp.w);
                }
                for(int j=0;j<3;++j) {
                    auto u=U+3*j;
                    // The congruence scaling of an auxiliary variable is free.
                    double sc=1/std::sqrt(std::max(1.,infnorm(u,3)));
                    auxdiag[j]=sc*sc;
                    for(int i=0;i<3;++i) off[i+3*j]=sc*u[i];
                }
                return true;
            }
            diag[0]=H[0]; diag[1]=H[4]; diag[2]=H[8];
            off[0]=-H[1]; off[1]=-H[2]; off[2]=-H[5];
            return true;
        }
        smat(s,a.data()); smat(z,b.data());
        if(!chol(a.data(),d) || !chol(b.data(),d)) return false;
        lower(a.data(),d); lower(b.data(),d);
        if(d>=40){Ls=a;Lz=b;}
        gemm(d,b.data(),a.data(),c.data(),'T','N');
        int info,nwork=(int)work.size();
        if(!svwork.empty()){
            nwork=(int)svwork.size();
            scipy_dgesdd_("A",&d,&d,c.data(),&d,lambda.data(),f.data(),&d,g.data(),&d,svwork.data(),
                          &nwork,svint.data(),&info);
        } else scipy_dgesvd_("A","A",&d,&d,c.data(),&d,lambda.data(),f.data(),&d,g.data(),&d,
                              work.data(),&nwork,&info);
        if(info!=0 || !(lambda.back()>0)) return false;
        gemm(d,a.data(),g.data(),T.data(),'N','T');
        gemm(d,f.data(),b.data(),Ti.data(),'T','T');
        for(int j=0;j<d;++j) for(int i=0;i<d;++i){
            T[i+d*j]/=std::sqrt(lambda[j]);
            Ti[i+d*j]/=std::sqrt(lambda[i]);
        }
        if(mode==2) {
            gemm(d,Ti.data(),Ti.data(),Ri.data(),'T','N');
            std::fill(diag,diag+k,0);
        } else {
            gemm(d,T.data(),T.data(),R.data(),'N','T');
            for(int j=0,l=0;j<k;++j) {
                diag[j]=hentry(R.data(),j,j);
                for(int i=0;i<j;++i,++l) off[l]=-hentry(R.data(),i,j);
            }
        }
        return true;
    }
    double hentry(const double* r,int u,int v) const {
        int i=row[u],j=col[u],a_=row[v],b_=col[v];
        double t=r[i+a_*d]*r[j+b_*d];
        if(a_!=b_) {
            t+=r[i+b_*d]*r[j+a_*d];
            if(i==j) t*=ISQ2;
        } else if(i!=j) {
            t*=SQ2;
        }
        return t;
    }
    void correct(const double* s,const double* z,const double* ds,const double* dz,
                 double smu,double weight,double* out) {
        if(type==ZERO) { std::fill(out,out+k,0); return; }
        if(type==NONNEG) {
            for(int i=0;i<k;++i) out[i]=-s[i]+(smu-(weight*ds[i])*dz[i])/z[i];
        } else if(type==SOC) {
            socW(v1.data(),ds,true);
            socW(v2.data(),dz,false);
            // Solve the Jordan multiplication by lambda via its scalar Schur complement.
            double x0=smu-weight*dot(v1.data(),v2.data(),d);
            double lt=lambda[0];
            for(int i=1;i<d;++i) wtmp(i,v1[0],v2[0],weight);
            if(ldet>0){
                long double num=(long double)lt*x0;
                for(int i=1;i<d;++i) num-=(long double)lambda[i]*v1[i];
                double u0=num/ldet;
                v1[0]=u0;
                for(int i=1;i<d;++i) v1[i]=(v1[i]-lambda[i]*u0)/lt;
                socW(out,v1.data(),false);
                for(int i=0;i<d;++i) out[i]-=s[i];
            } else {
                for(int i=0;i<d;++i) out[i]=-s[i]+((i==0?smu:0)-weight*ds[i]*dz[i])/std::max(z[0],1e-300);
            }
        } else if(type==EXP) {
            double eta_[3]={0,0,0};
            if(weight) exp.third(ds,dz,eta_);
            for(int i=0;i<3;++i) out[i]=-s[i]+smu*exp.shadow[i]+weight*eta_[i];
        } else {
            smat(ds,a.data()); gemm(d,Ti.data(),a.data(),b.data()); gemm(d,b.data(),Ti.data(),c.data(),'N','T');
            smat(dz,a.data()); gemm(d,T.data(),a.data(),b.data(),'T','N'); gemm(d,b.data(),T.data(),f.data());
            gemm(d,c.data(),f.data(),g.data(),'N','N',-weight);
            for(int j=0;j<d;++j) for(int i=0;i<=j;++i) {
                double t=(i==j?2*smu:0)+(g[i+j*d]+g[j+i*d]);
                a[i+j*d]=a[j+i*d]=t/(lambda[i]+lambda[j]);
            }
            gemm(d,T.data(),a.data(),b.data()); gemm(d,b.data(),T.data(),g.data(),'N','T');
            svec(g.data(),out);
            for(int i=0;i<k;++i) out[i]-=s[i];
        }
    }
    void wtmp(int i,double x,double y,double weight) { v1[i]=-weight*(x*v2[i]+y*v1[i]); }
    void mul(const double* z,double* out,bool inverse) {
        if(type==SOC){
            if(inverse) return;
            double q=2*eta*eta*dot(w.data(),z,d);
            out[0]=q*w[0]-eta*eta*z[0];
            for(int i=1;i<d;++i) out[i]=q*w[i]+eta*eta*z[i];
        } else if(type==PSD){
            smat(z,a.data());
            const double* r=inverse?Ri.data():R.data();
            if (inverse && d<32) {
                gemm(d,Ti.data(),a.data(),b.data()); gemm(d,b.data(),Ti.data(),c.data(),'N','T');
                gemm(d,Ti.data(),c.data(),b.data(),'T','N'); gemm(d,b.data(),Ti.data(),a.data());
            } else {
                gemm(d,r,a.data(),b.data()); gemm(d,b.data(),r,a.data());
            }
            svec(a.data(),out);
        }
    }
    void schur(int n,const int* p,const int* r,const double* v,double* out) {
        const double* ir=Ri.data();
        int ix=0;
        // Singletons and sparse basis matrices are common in SDPs.
        if((long long)p[n]<std::min((long long)n*std::max(5,d),(long long)2*d*d+2*n)){
            for(int i=0;i<n;++i) for(int j=i;j<n;++j){
                double ss=0;
                for(int q=p[i];q<p[i+1];++q) {
                    double t=0;
                    for(int l=p[j];l<p[j+1];++l) t += v[l]*hentry(ir,r[q],r[l]);
                    ss += v[q]*t;
                }
                out[ix++]=ss;
            }
        } else {
            for(int i=0;i<n;++i){
                std::fill(a.begin(),a.end(),0);
                for(int l=p[i];l<p[i+1];++l){
                    int u=row[r[l]],j=col[r[l]];
                    a[u+j*d]=a[j+u*d]=(u==j?1:ISQ2)*v[l];
                }
                gemm(d,ir,a.data(),b.data()); gemm(d,b.data(),ir,c.data());
                for(int j=i;j<n;++j) {
                    double ss=0;
                    for(int l=p[j];l<p[j+1];++l){
                        int u=row[r[l]],h=col[r[l]];
                        ss+=v[l]*(u==h?1:SQ2)*c[u+h*d];
                    }
                    out[ix++]=ss;
                }
            }
        }
    }
    double step(const double* x,const double* h,bool dual,double bound) {
        if(type==ZERO) return bound;
        if(type==NONNEG) {
            for(int i=0;i<d;++i) if(h[i]<0) bound=std::min(bound,-x[i]/h[i]);
            return bound;
        }
        if(type==SOC) {
            auto ok=[&](double t){
                double y=x[0]+t*h[0];
                if(!(y>0)) return false;
                double sum=0;
                for(int i=1;i<d;++i) {double v=(x[i]+t*h[i])/y; sum+=v*v;}
                return sum<1;
            };
            if(ok(bound)) return bound;
            long double b_=(long double)x[0]*h[0], c_=0, a_=(long double)h[0]*h[0];
            for(int i=1;i<d;++i) {a_-=(long double)h[i]*h[i]; b_-=(long double)x[i]*h[i];}
            c_=(long double)x[0]*x[0];
            for(int i=1;i<d;++i) c_-=(long double)x[i]*x[i];
            long double discr=b_*b_-a_*c_;
            if(discr>=0) {
                long double den=-b_+std::sqrt(discr);
                if(den>0) return std::min(bound,std::max(0.0,(double)(c_/den)));
            }
            double lo=0,hi=bound;
            for(int i=0;i<36;++i) {double t=(lo+hi)*0.5; if(ok(t)) lo=t; else hi=t;}
            return lo;
        }
        if(type==EXP){
            auto ok=[&](double t){
                double u=x[0]+t*h[0], v=x[1]+t*h[1], w_=x[2]+t*h[2];
                if(dual){
                    if(!(u<0 && w_>0)) return false;
                    return std::fma(-u,1+std::log(w_)-std::log(-u),v)>0;
                }
                return v>0 && w_>0 && std::fma(v,std::log(w_)-std::log(v),-u)>0;
            };
            if(ok(bound)) return bound;
            double hi=bound,lo=0;
            for(int i=0;i<34;++i) {double t=(lo+hi)*0.5; if(ok(t)) lo=t; else hi=t;}
            return lo;
        }
        if(d>=40) {
            for(int l=0;l<k;++l) {
                int i=row[l],j=col[l];
                a[j+i*d]=(i==j?1:ISQ2)*(x[l]+bound*h[l]);
            }
            if(chol(a.data(),d))return bound;
            smat(h,c.data());
            const double* f=dual?Lz.data():Ls.data();
            const double one=1;
            scipy_dtrsm_("L","L","N","N",&d,&d,&one,f,&d,c.data(),&d);
            scipy_dtrsm_("R","L","T","N",&d,&d,&one,f,&d,c.data(),&d);
            double mn=mineig(c.data());
            return mn<0?std::min(bound,-1/mn):bound;
        }
        smat(h,a.data());
        if(dual) {
            gemm(d,T.data(),a.data(),b.data(),'T','N'); gemm(d,b.data(),T.data(),c.data());
        } else {
            gemm(d,Ti.data(),a.data(),b.data()); gemm(d,b.data(),Ti.data(),c.data(),'N','T');
        }
        for(int j=0;j<d;++j) for(int i=j;i<d;++i)
            c[i+j*d]/=(std::sqrt(lambda[i])*std::sqrt(lambda[j]));
        double mn=mineig(c.data());
        return (mn<0)?std::min(bound,-1/mn):bound;
    }
    double defect(const double* x,bool dual,double limit) {
        if(dual && type==ZERO) return 0;
        double norm=infnorm(x,k);
        if(!std::isfinite(norm)) return INF;
        double sc=std::max(1.0,norm);
        if(type==ZERO) return norm/sc;
        if(type==NONNEG) return std::max(0.0,-*std::min_element(x,x+k)/sc);
        if(type==SOC) return std::max(0.0,(norm2(x+1,d-1)-x[0])/sc);
        if(type==EXP) return dual?exp_defect(-x[1]/sc,-x[0]/sc,E*(x[2]/sc)):
                                      exp_defect(x[0]/sc,x[1]/sc,x[2]/sc);
        if(d>=24) {
            bool diagonal=true;
            double eig=INF;
            for(int j=0,at=0;j<d && diagonal;++j){
                for(int i=0;i<j;++i)if(x[at+i]!=0){diagonal=false;break;}
                eig=std::min(eig,x[at+j]);
                at+=j+1;
            }
            if(diagonal)return std::max(0.,-eig/sc);
        }
        smat(x,a.data(),1/sc);
        if(limit>0) {
            for(int j=0;j<d;++j) a[j*(d+1)]+=limit;
            if(chol(a.data(),d)) return 0;
            smat(x,a.data(),1/sc);
        }
        return std::max(0.0,-mineig(a.data()));
    }
    void project(double* x,bool dual) {
        if(type==ZERO){if(!dual)std::fill(x,x+k,0);return;}
        if(type==NONNEG){for(int i=0;i<k;++i)x[i]=std::max(0.,x[i]);return;}
        if(type==SOC) {
            double h=norm2(x+1,d-1);
            if(h<=x[0]) return;
            if(h<=-x[0]) {std::fill(x,x+k,0);return;}
            double t=0.5*(1+x[0]/h);
            x[0]=t*h;
            for(int j=1;j<d;++j)x[j]*=t;
            return;
        }
        if(type==EXP){
            if(dual) {
                if(exp_defect(-x[1],-x[0],E*x[2])<=0) return;
                double temp[3]={-x[0],-x[1],-x[2]};
                project_exp(temp,project_hint,x);
            } else project_exp(x,project_hint);
            return;
        }
        smat(x,a.data());
        int info,lwork=work.size();
        if(d>=40){
            const double vl=negative_spectrum?-std::numeric_limits<double>::max():0.,
                         vu=negative_spectrum?0.:std::numeric_limits<double>::max(), tol=0.;
            const int ilo=0,ihi=0,liwork=ieigen.size();
            int rank=0;
            scipy_dsyevr_("V","V","L",&d,a.data(),&d,&vl,&vu,&ilo,&ihi,&tol,&rank,eval.data(),
                          b.data(),&d,support.data(),work.data(),&lwork,ieigen.data(),&liwork,&info);
            if(info==0) {
                if(rank==0){
                    if(!negative_spectrum)std::fill(x,x+k,0.);
                    return;
                }
                if(rank==d && negative_spectrum){
                    std::fill(x,x+k,0.);negative_spectrum=false;return;
                }
                if(rank==d && !negative_spectrum){negative_spectrum=true;return;}
                for(int j=0;j<rank;++j) {
                    double e=std::sqrt(std::abs(eval[j]));
                    for(int i=0;i<d;++i)b[i+d*j]*=e;
                }
                const double one=1,zero=0;
                scipy_dsyrk_("L","N",&d,&rank,&one,b.data(),&d,&zero,c.data(),&d);
                for(int l=0;l<k;++l) {
                    int i=row[l],j=col[l];
                    double e=c[j+i*d]*(i==j?1:SQ2);
                    x[l]=negative_spectrum?x[l]+e:e;
                }
                if(rank>d*0.60)negative_spectrum=!negative_spectrum;
                return;
            }
            smat(x,a.data());
        }
        scipy_dsyev_("V","L",&d,a.data(),&d,eval.data(),work.data(),&lwork,&info);
        if(info || eval[0]>=0) return;
        if(eval.back()<=0){std::fill(x,x+k,0);return;}
        for(int j=0;j<d;++j){
            double t=std::sqrt(std::max(0.,eval[j]));
            for(int i=0;i<d;++i)b[i+d*j]=a[i+d*j]*t;
        }
        gemm(d,b.data(),b.data(),c.data(),'N','T');
        svec(c.data(),x);
    }
};

struct Cones {
    std::vector<Block> blocks;
    Cones(int n,const int* kinds,const int* dims,const int* modes) {
        blocks.reserve(n);
        int pos=0,off=0,aux=0;
        for(int i=0;i<n;++i) {
            blocks.emplace_back(kinds[i],dims[i],pos,modes[i],off,aux);
            auto& b=blocks.back();
            pos+=b.k;
            if(b.mode==4) {off+=9; aux+=3;}
            else if(b.mode==3) {off+=2*b.k-1; aux+=2;}
            else if(b.mode==1) {off+=b.k; ++aux;}
            else if(b.mode==0 && b.type>=SOC) off+=b.k*(b.k-1LL)/2;
        }
    }
};
}

extern "C" {
void* cc_create(int n,const int* kinds,const int* dims,const int* modes) {
    try { return new Cones(n,kinds,dims,modes); } catch(...) { return nullptr; }
}
void cc_free(Cones* c) {delete c;}
void cc_init(Cones* c,double* s,double* z) {
    for(auto& b:c->blocks) b.unit(s+b.start,z+b.start);
}
void cc_shift(Cones* c,double* s,int dual) {
    for(auto& b:c->blocks) b.shift(s+b.start,dual);
}
int cc_prepare(Cones* c,const double* s,const double* z,double* diag,double* off,double* aux,
               double mu,int pd_exp) {
    for(auto& b:c->blocks)
        if(!b.prepare(s+b.start,z+b.start,diag+b.start,off?off+b.offset:nullptr,
                      aux?aux+b.aux:nullptr,mu,pd_exp)) return 0;
    return 1;
}
void cc_correct(Cones* c,const double* s,const double* z,const double* ds,const double* dz,
                double smu,double weight,double* out) {
    for(auto& b:c->blocks) b.correct(s+b.start,z+b.start,ds+b.start,dz+b.start,smu,weight,out+b.start);
}
double cc_step(Cones* c,const double* s,const double* ds,int dual,double lim) {
    static const bool dbg=std::getenv("CONIC_STEP_DEBUG");
    int arg=-1, kind=-1;
    for(auto& b:c->blocks) {
        double v=b.step(s+b.start,ds+b.start,dual,lim);
        if(dbg && v<lim) {arg=b.start;kind=b.type;}
        lim=v;
    }
    if(dbg && lim<0.06 && arg>=0)
        std::fprintf(stderr,"limit %g dual %d row %d type %d x0 %g dir0 %g\n",
                    lim,dual,arg,kind,s[arg],ds[arg]);
    return lim;
}
void cc_psd_mul(Cones* c,int block,const double* in,double* out,int inv) {
    c->blocks[block].mul(in,out,inv);
}
void cc_psd_schur(Cones* c,int block,int n,const int* p,const int* r,const double* v,double* out) {
    c->blocks[block].schur(n,p,r,v,out);
}
double cc_defect(Cones* c,const double* s,int dual,double lim) {
    double err=0;
    for(auto& b:c->blocks) {
        err=std::max(err,b.defect(s+b.start,dual,lim));
        if(err>lim && lim>0) break;
    }
    return err;
}
void cc_project(Cones* c,double* x,int dual) {
    for(auto& b:c->blocks) b.project(x+b.start,dual);
}
void cc_project_pair(Cones* c,double* x,double* z) {
    for(auto& b:c->blocks) {
        auto* s=x+b.start;auto* y=z+b.start;
        if(b.type==EXP)project_exp(s,b.project_hint,y);
        else {
            for(int j=0;j<b.k;++j)y[j]=-s[j];
            b.project(s,false);
            for(int j=0;j<b.k;++j)y[j]+=s[j];
        }
    }
}
int cc_constant(int kind,int d,const double* b,double* z,double lim) {
    try {
        Block c(kind,d,0,5,0,0);
        if(c.defect(b,false,lim)<=lim) return 0;
        for(int j=0;j<c.k;++j)z[j]=-b[j];
        c.project(z,true);
        return 1;
    } catch(...) {return -1;}
}
void cc_norms(int n,const int* p,const int* r,const double* v,double* rows,double* cols) {
    for(int j=0;j<n;++j) {
        double m=cols[j];
        for(int k=p[j];k<p[j+1];++k){
            double t=std::abs(v[k]);
            m=std::max(m,t);
            if(rows) rows[r[k]]=std::max(rows[r[k]],t);
        }
        cols[j]=m;
    }
}
void cc_scale(int n,const int* p,const int* r,double* v,const double* rows,const double* cols) {
    for(int j=0;j<n;++j){
        double s=cols[j];
        if(rows) for(int k=p[j];k<p[j+1];++k) v[k]*=s*rows[r[k]];
        else for(int k=p[j];k<p[j+1];++k) v[k]*=s;
    }
}
void cc_rowscale(Cones* c,double* e,int averaging) {
    for(auto& b:c->blocks) if(b.type>=SOC) {
        double* v=e+b.start;
        double t=0;
        for(int j=0;j<b.k;++j) t=averaging?t+v[j]:std::max(t,v[j]);
        if(averaging) t/=b.k;
        std::fill(v,v+b.k,t);
    }
}
}
