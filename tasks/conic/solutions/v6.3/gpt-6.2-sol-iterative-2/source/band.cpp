#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "native_kkt.hpp"
#include "native_soc.hpp"

namespace {
using V=std::vector<double>;
using I=std::vector<int>;
using Clock=std::chrono::steady_clock;
double infnorm(const V& v) {
    double r=0.;
    for (double x:v) r=std::max(r,std::abs(x));
    return r;
}
double dot(const V& a,const V& b) {
    long double value=0.;
    std::size_t j=0;
    for (;j+128<=a.size();j+=128) {
        double s0=0.,s1=0.,s2=0.,s3=0.;
        for (std::size_t k=j;k<j+128;k+=4) {
            s0+=a[k]*b[k]; s1+=a[k+1]*b[k+1];
            s2+=a[k+2]*b[k+2]; s3+=a[k+3]*b[k+3];
        }
        value+=static_cast<long double>(s0)+s1+s2+s3;
    }
    for (;j<a.size();++j) value+=a[j]*b[j];
    return value;
}
struct Step { V x,s,z; double t,k; };

class Qp {
    int n,m,d,degree=0,ld;
    I row,col,ptr,ir,order;
    I ro,ci;
    V A,P,b,q,E,D,mat,B0,scale,W;
    V cx,cz,zero,ar,br;
    std::vector<unsigned char> eq,issoc;
    std::vector<conic::Soc> soc;
    double c=1.;
    double apmax=1., pmax=1., bmax=0., qmax=0.;
    Clock::time_point stop;
    std::unique_ptr<conic::NativeKkt> gen;
    bool plain=false,has_exp=false;

    void mulA(const V& x,V& y) {
        for (int i=0;i<m;++i) {
            double v=0.;
            for (int k=row[i];k<row[i+1];++k) v+=A[k]*x[col[k]];
            y[i]=v;
        }
    }
    void mulAt(const V& z,V& y) {
        std::fill(y.begin(),y.end(),0.);
        for (int i=0;i<m;++i) {
            const double v=z[i];
            for (int k=row[i];k<row[i+1];++k) y[col[k]]+=A[k]*v;
        }
    }
    void mulP(const V& x,V& y) {
        std::fill(y.begin(),y.end(),0.);
        for (int j=0;j<n;++j)
            for (int k=ptr[j];k<ptr[j+1];++k) y[ir[k]]+=P[k]*x[order[j]];
    }
    bool factor(const V& s,const V& z,double mu) {
        if (gen) {
            for (int i=0;i<m;++i) if (!issoc[i])
                W[i]=eq[i]?3.e11:1./std::max(6.e-14,s[i]/z[i]);
            for (auto& c:soc) {
                if (!c.setup(s.data()+c.a,z.data()+c.a,W.data()+c.a,mu)) return false;
                c.rotate_matrix(ar);
            }
            return gen->factor(W);
        }
        for (int i=0;i<m;++i) W[i]=eq[i]?3.e11:1./std::max(6.e-14,s[i]/z[i]);
        return band_factor();
    }
    bool band_factor() {
        mat=B0;
        for (int i=0;i<m;++i) {
            const double w=W[i];
            for (int k=row[i];k<row[i+1];++k) {
                const int j=col[k];
                const double a=A[k]*w;
                double* r=mat.data()+std::size_t(j)*ld;
                for (int t=row[i];t<=k;++t) r[j-col[t]]+=a*A[t];
            }
        }
        for (int i=0;i<n;++i)
            scale[i]=1./std::sqrt(std::max(1.e-9,mat[std::size_t(i)*ld]));
        for (int i=0;i<n;++i) {
            double* a=mat.data()+std::size_t(i)*ld;
            const int lo=std::max(0,i-d);
            for (int j=lo;j<i;++j) {
                const double* r=mat.data()+std::size_t(j)*ld;
                double v=a[i-j]*scale[i]*scale[j];
                for (int k=lo;k<j;++k) v-=a[i-k]*r[j-k];
                a[i-j]=v*r[0];
            }
            double v=a[0]*scale[i]*scale[i]+1.e-11;
            for (int j=lo;j<i;++j) v-=a[i-j]*a[i-j];
            if (!(v>0.) || !std::isfinite(v)) return false;
            a[0]=1./std::sqrt(v);
        }
        return true;
    }
    void solve_chol(V& u) {
        for (int i=0;i<n;++i) {
            const double* a=mat.data()+std::size_t(i)*ld;
            double v=u[i]*scale[i];
            for (int j=std::max(0,i-d);j<i;++j) v-=a[i-j]*u[j];
            u[i]=v*a[0];
        }
        for (int i=n-1;i>=0;--i) {
            const double* a=mat.data()+std::size_t(i)*ld;
            const double v=u[i]*a[0];
            u[i]=v;
            for (int j=std::max(0,i-d);j<i;++j) u[j]-=a[i-j]*v;
        }
        for (int i=0;i<n;++i) u[i]*=scale[i];
    }
    void solve_linear(const V& bx,const V& bz,V& x,V& z) {
        if (gen) {
            if (soc.empty()) gen->solve(W,bx,bz,x,z);
            else {
                br=bz;
                for (const auto& block:soc) block.rotate(br.data()+block.a,false);
                gen->solve(W,bx,br,x,z);
                for (const auto& block:soc) block.rotate(z.data()+block.a,true);
            }
            return;
        }
        x=bx;
        for (int i=0;i<m;++i) {
            const double b=bz[i]*W[i];
            for (int k=row[i];k<row[i+1];++k) x[col[k]]+=A[k]*b;
        }
        solve_chol(x);
        for (int i=0;i<m;++i) {
            long double a=-bz[i];
            for (int k=row[i];k<row[i+1];++k) a+=static_cast<long double>(A[k])*x[col[k]];
            z[i]=a*W[i];
        }
    }
    void correction(const V& bx,V& x,V& z,V& r,V& px,V& e,V& de) {
        mulAt(z,r);
        mulP(x,px);
        for (int j=0;j<n;++j) r[j]=bx[j]-r[j]-px[j];
        if (infnorm(r)<3.e-12*(1.+infnorm(bx))) return;
        if (gen) {
            solve_linear(r,zero,cx,cz);
            for (int j=0;j<n;++j) x[j]+=cx[j];
            for (int i=0;i<m;++i) z[i]+=cz[i];
            return;
        }
        solve_chol(r);
        mulA(r,e);
        for (int j=0;j<n;++j) x[j]+=r[j];
        for (int i=0;i<m;++i) z[i]+=e[i]*W[i];
    }
    double step(const V& s,const V& z,double tau,double kap,const Step& x) {
        double a=1.;
        if (x.t<0.) a=std::min(a,-tau/x.t);
        if (x.k<0.) a=std::min(a,-kap/x.k);
        for (int i=0;i<m;++i) if (!eq[i] && !issoc[i]) {
            if (x.s[i]<0. && s[i]+a*x.s[i]<0.) a=-s[i]/x.s[i];
            if (x.z[i]<0. && z[i]+a*x.z[i]<0.) a=-z[i]/x.z[i];
        }
        for (const auto& block:soc) {
            a=block.step(s.data()+block.a,x.s.data()+block.a,a);
            a=block.step(z.data()+block.a,x.z.data()+block.a,a,true);
        }
        return a;
    }

public:
    double quality=std::numeric_limits<double>::infinity();
    void switch_mode() { plain=!plain; }

    Qp(int nn,int mm,int bw,const I& perm,const int* ap,const int* ai,const double* va,
       const int* pp,const int* pi,const double* vp,const double* qq,const double* bb,
       const unsigned char* equality,Clock::time_point deadline,
       conic::Lapack lapack=conic::Lapack(),bool use_plain=false,
       int groups=0,const int* starts=nullptr,const int* dims=nullptr,
       const unsigned char* expmark=nullptr,void* svd=nullptr)
        : n(nn),m(mm),d(bw),ld(bw+1),
          row(ap,ap+m+1),col(ai,ai+ap[m]),ptr(pp,pp+n+1),ir(pi,pi+pp[n]),order(perm),
          A(va,va+ap[m]),P(vp,vp+pp[n]),b(bb,bb+m),q(n),E(m,1.),D(n,1.),
          B0(std::size_t(n)*ld,0.),scale(n),W(m),eq(equality,equality+m),issoc(m,0),stop(deadline),
          plain(use_plain) {
        apmax=infnorm(A); if (!apmax) apmax=1.;
        pmax=infnorm(P); if (!pmax) pmax=1.;
        bmax=infnorm(b);
        for (int k=0;k<groups;++k) {
            const bool isexp=expmark && expmark[k];
            soc.emplace_back(starts[k],dims[k],isexp,svd);
            has_exp|=isexp;
            std::fill(issoc.begin()+starts[k],issoc.begin()+starts[k]+dims[k],1);
        }
        for (int j=0;j<n;++j) q[order[j]]=qq[j];
        qmax=infnorm(q);
        for (int& j:ir) j=order[j];
        for (int& j:col) j=order[j];
        bool reordered=false;
        for (int j=0;j<n;++j) reordered |= order[j]!=j;
        if (reordered) {
            std::vector<std::pair<int,double>> tmp;
            for (int i=0;i<m;++i) {
                tmp.clear();
                for (int k=row[i];k<row[i+1];++k) tmp.emplace_back(col[k],A[k]);
                std::sort(tmp.begin(),tmp.end());
                for (int k=row[i];k<row[i+1];++k) {
                    col[k]=tmp[k-row[i]].first;
                    A[k]=tmp[k-row[i]].second;
                }
            }
        }
        V dc(n),dr(m);
        for (int it=0;it<5 && Clock::now()<stop;++it) {
            std::fill(dc.begin(),dc.end(),0.);
            for (int i=0;i<m;++i) {
                double s=0.;
                for (int k=row[i];k<row[i+1];++k) {
                    const double v=std::abs(A[k]);
                    dc[col[k]]=std::max(dc[col[k]],v);
                    s=std::max(s,v);
                }
                dr[i]=s;
            }
            for (const auto& g:soc) {
                auto first=dr.begin()+g.a;
                std::fill(first,first+g.d,*std::max_element(first,first+g.d));
            }
            for (int j=0;j<n;++j) {
                auto& v=dc[order[j]];
                for (int k=ptr[j];k<ptr[j+1];++k) v=std::max(v,std::abs(P[k]));
            }
            for (double& s:dc) s=s>1.e-13?1./std::sqrt(s):1.;
            for (double& s:dr) s=s>1.e-13?1./std::sqrt(s):1.;
            for (int j=0;j<n;++j) { q[j]*=dc[j]; D[j]*=dc[j]; }
            for (int i=0;i<m;++i) {
                for (int k=row[i];k<row[i+1];++k) A[k]*=dr[i]*dc[col[k]];
                b[i]*=dr[i]; E[i]*=dr[i];
            }
            for (int j=0;j<n;++j)
                for (int k=ptr[j];k<ptr[j+1];++k) P[k]*=dc[order[j]]*dc[ir[k]];
            const double mag=std::max(infnorm(q),.1*infnorm(P));
            if (mag>1.e-13) {
                double v=std::clamp(1./mag,1.e-12/c,1.e12/c);
                for (double& s:q) s*=v;
                for (double& s:P) s*=v;
                c*=v;
            }
        }
        double nrm=infnorm(b);
        if (nrm>30. || (nrm>0. && nrm<.03)) {
            nrm=std::clamp(nrm,1.e-7,1.e7);
            for (double& a:b) a/=nrm;
            for (double& e:E) e/=nrm;
            for (double& x:D) x*=nrm;
            for (double& p:P) p*=nrm;
            c/=nrm;
        }
        if (d<0) {
            if (soc.empty()) {
                gen=std::make_unique<conic::NativeKkt>(n,m,row,col,A,ptr,ir,P,eq,lapack,deadline);
            } else {
                ro.resize(m+1);
                std::size_t k=0;
                for (int i=0;i<m;) {
                    if (k<soc.size() && i==soc[k].a) {
                        auto& g=soc[k++];
                        I cc(col.begin()+row[i],col.begin()+row[i+g.d]);
                        std::sort(cc.begin(),cc.end());
                        cc.erase(std::unique(cc.begin(),cc.end()),cc.end());
                        g.width=cc.size();
                        if (ci.size()+cc.size()*g.d>700000)
                            throw std::runtime_error("SOC rotation fill");
                        g.offset=ci.size();
                        g.V.assign(cc.size()*g.d,0.);
                        for (int j=0;j<g.d;++j) {
                            ro[i+j]=ci.size();
                            ci.insert(ci.end(),cc.begin(),cc.end());
                            ar.insert(ar.end(),cc.size(),0.);
                            for (int r=row[i+j];r<row[i+j+1];++r) {
                                auto idx=std::lower_bound(cc.begin(),cc.end(),col[r])-cc.begin();
                                g.V[j*cc.size()+idx]=A[r];
                            }
                        }
                        i+=g.d;
                    } else {
                        ro[i]=ci.size();
                        ci.insert(ci.end(),col.begin()+row[i],col.begin()+row[i+1]);
                        ar.insert(ar.end(),A.begin()+row[i],A.begin()+row[i+1]);
                        ++i;
                    }
                }
                ro[m]=ci.size();
                br.resize(m);
                gen=std::make_unique<conic::NativeKkt>(n,m,ro,ci,ar,ptr,ir,P,eq,
                            lapack,deadline,&issoc);
            }
            cx.resize(n);
            cz.resize(m);
            zero.resize(m);
        } else {
            for (int j=0;j<n;++j) for (int k=ptr[j];k<ptr[j+1];++k) {
                const int a=ir[k],b=order[j];
                if (a>=b) B0[std::size_t(a)*ld+a-b]+=P[k];
            }
        }
        degree=m-std::count(eq.begin(),eq.end(),1);
        for (const auto& g:soc) if (!g.exp) degree-=g.d-1;
    }

    int solve(double* xo,double* zo,double* so=nullptr,
              const double* wx=nullptr,const double* wz=nullptr) {
        quality=std::numeric_limits<double>::infinity();
        V x(n),s(m,1.),z(m,1.),ra(m),rb(n),px(n),az(n),gx(n), a(n),y(m), bz(m),bx(n),
          sx(n),sz(m),work(n),wp(n),we(m);
        for (int i=0;i<m;++i) if (eq[i]) s[i]=z[i]=0.;
        for (const auto& b:soc) {
            if (b.exp) {
                s[b.a]=-1.; s[b.a+1]=1.; s[b.a+2]=2.;
                z[b.a]=-1.; z[b.a+1]=0.; z[b.a+2]=1.;
            } else {
                for (int i=b.a+1;i<b.a+b.d;++i) s[i]=z[i]=0.;
            }
        }
        double tau=1.,kap=1.,best=.04;
        if (gen && soc.empty() && wx && wz) {
            for (int j=0;j<n;++j) x[order[j]]=wx[j]/D[order[j]];
            for (int i=0;i<m;++i) z[i]=wz[i]*c/E[i];
            mulA(x,ra);
            mulP(x,px);
            mulAt(z,az);
            double ss=1.,zs=std::max(1.,infnorm(z)),r=0.,rp=0.;
            for (int i=0;i<m;++i) {
                s[i]=eq[i]?0.:b[i]-ra[i];
                ss=std::max(ss,std::abs(s[i]));
                rp=std::max(rp,eq[i]?std::abs(b[i]-ra[i]):-s[i]);
            }
            for (int j=0;j<n;++j)
                r=std::max(r,std::abs(px[j]+q[j]+az[j]));
            r=std::max(rp/ss,r/(1.+std::max({infnorm(px),infnorm(az),infnorm(q)})));
            if (r<.4 && std::isfinite(r+zs+ss)) {
                const double f=std::clamp(.06*std::sqrt(r),2.e-6,.11);
                for (int i=0;i<m;++i) if (!eq[i]) {
                    const double zp=std::max(0.,z[i]);
                    s[i]=std::max(s[i],f*ss/(1.+zp/zs));
                    z[i]=std::max(zp,f*zs/(1.+s[i]/ss));
                }
                kap=std::clamp(dot(s,z)/(degree+1.),1.e-7,1.);
            } else {
                std::fill(x.begin(),x.end(),0.);
                for (int i=0;i<m;++i) s[i]=z[i]=eq[i]?0.:1.;
            }
        } else if (soc.empty() && !P.empty() && degree>30 &&
                   ((gen && n+m<38000) || (!gen && n+m<620000 && d<=9))) {
            std::fill(W.begin(),W.end(),1.);
            if (gen?gen->factor(W,true):band_factor()) {
                for (int j=0;j<n;++j) bx[j]=-q[j];
                solve_linear(bx,b,x,z);
                double smin=0.,zmin=0.;
                for (int i=0;i<m;++i) if (!eq[i]) {
                    s[i]=-z[i];
                    smin=std::min(smin,s[i]);
                    zmin=std::min(zmin,z[i]);
                }
                for (int i=0;i<m;++i) if (!eq[i]) {
                    s[i]+=(1.-smin);
                    z[i]+=(1.-zmin);
                }
            }
        }
        bool has=false,refit=false;
        double prev=std::numeric_limits<double>::infinity();
        int stuck=0;
        Step df{V(n),V(m),V(m),0.,0.},dn=df;
        auto save=[&](double t) {
            for (int j=0;j<n;++j) xo[j]=D[order[j]]*x[order[j]]/t;
            for (int i=0;i<m;++i) zo[i]=E[i]*z[i]/(c*t);
            if (so)
                for (int i=0;i<m;++i) so[i]=s[i]/(E[i]*t);
        };
        for (int it=0;it<46 && Clock::now()<stop;++it) {
            mulP(x,px); mulA(x,ra); mulAt(z,az);
            const double xx=dot(x,px),qx=dot(q,x),bb=dot(b,z),ss=dot(s,z);
            const double kt=kap/tau;
            const double gap=std::abs(xx/tau+qx+bb)/std::max({c*tau,std::abs(.5*xx/tau+qx),
                                                                  std::abs(.5*xx/tau+bb)});
            double rp=0.,rd=0.,an=0.,yn=0.;
            for (int i=0;i<m;++i) {
                an=std::max(an,(std::abs(b[i]*tau)+std::abs(ra[i])+std::abs(s[i]))/E[i]);
                ra[i]=b[i]*tau-ra[i]-s[i];
                rp=std::max(rp,std::abs(ra[i]/E[i]));
            }
            for (int j=0;j<n;++j) {
                yn=std::max(yn,(std::abs(px[j])+std::abs(az[j])+std::abs(q[j]*tau))/D[j]/c);
                rb[j]=-px[j]-az[j]-q[j]*tau;
                rd=std::max(rd,std::abs(rb[j]/D[j]/c));
            }
            rp/=tau+an; rd/=tau+yn;
            const double metric=std::max({gap,rp,rd});
            if (metric<best) { best=quality=metric; save(tau); has=true; }
            if (metric<2.4e-9) return 1;
            if (metric<.78*prev || kt>1.e3) { prev=metric; stuck=0; }
            else if (++stuck>12) break;
            if (kt>200. && (it%2==0 || kt>1.e7)) {
                const double pm=bmax>0.?-bb/bmax:0.;
                const double dm=qmax>0.?-qx/c/qmax:0.;
                double zn=0.,xn=0.,certp=0.,certd=0.,certlin=0.;
                for (int i=0;i<m;++i) {
                    zn=std::max(zn,std::abs(E[i]*z[i]));
                    certlin=std::max(certlin,std::abs((b[i]*tau-ra[i])/E[i]));
                }
                for (int j=0;j<n;++j) {
                    xn=std::max(xn,std::abs(D[j]*x[j]));
                    certp=std::max(certp,std::abs(az[j]/D[j]));
                    certd=std::max(certd,std::abs(px[j]/D[j]/c));
                }
                if ((zn>0. && pm>1.02e-8*zn && certp<7.e-9*apmax*pm) ||
                    (xn>0. && dm>1.02e-8*xn && certd<7.e-9*pmax*dm
                     && certlin<7.e-9*apmax*dm && so!=nullptr)) {
                    save(1.);
                    return 2;
                }
            }
            if (soc.empty() && !refit && it>=6 && m<(gen?31000:550000) && !P.empty() && gap<4.e-4
                && std::max(rp,rd)<3.e-5 && Clock::now()<stop) {
                refit=true;
                V xp=x,zp=z,H(m);
                const double sm=infnorm(s),zm=infnorm(z);
                for (int i=0;i<m;++i) {
                    const bool active=eq[i] || (s[i]<4.e-5*(1.+sm) &&
                                                 z[i]>2.e-7*std::max(1.e-10,zm));
                    zp[i]=active?z[i]:0.;
                    H[i]=active?2.e8:1.e-13;
                    bz[i]=active?ra[i]+s[i]:0.;
                }
                mulAt(zp,az);
                for (int j=0;j<n;++j) bx[j]=-px[j]-q[j]*tau-az[j];
                W=H;
                if (gen?gen->factor(W):band_factor()) {
                    solve_linear(bx,bz,sx,sz);
                    double linear=0., cone=0.;
                    for (int j=0;j<n;++j) xp[j]+=sx[j];
                    for (int i=0;i<m;++i)
                        zp[i]=eq[i]?zp[i]+sz[i]:std::max(0.,zp[i]+sz[i]);
                    mulP(xp,wp); mulAt(zp,az); mulA(xp,we);
                    const double bd=dot(b,zp), net=dot(xp,wp), qt=dot(q,xp);
                    const double g=std::abs(net/tau+qt+bd)/
                        std::max({c*tau,std::abs(.5*net/tau+qt),std::abs(.5*net/tau+bd)});
                    double am=0.,cm=0.;
                    for (int i=0;i<m;++i) {
                        double h=b[i]*tau-we[i];
                        cone=std::max(cone,(eq[i]?std::abs(h):std::max(0.,-h))/E[i]);
                        am=std::max(am,std::max(std::abs(b[i]*tau),std::abs(we[i]))/E[i]);
                    }
                    for (int j=0;j<n;++j) {
                        linear=std::max(linear,std::abs(wp[j]+az[j]+q[j]*tau)/D[j]/c);
                        cm=std::max(cm,std::max({std::abs(wp[j]),std::abs(az[j]),std::abs(q[j]*tau)})/D[j]/c);
                    }
                    if (std::isfinite(g) && g<3.e-9 && linear<3.e-9*(tau+cm)
                        && cone<2.4e-9*(tau+am)) {
                        x.swap(xp); z.swap(zp); save(tau);
                        quality=2.5e-9;
                        return 1;
                    }
                }
            }
            bool frozen=(plain && degree>0) || (it>=5 && kt<.0006 && std::max(rp,rd)<.0008 && gap<.025);
            double mu=frozen && degree>0?ss/degree:(ss+tau*kap)/(degree+1.);
            if (!(mu>0.) || !std::isfinite(mu) || !factor(s,z,mu)) break;
            const double rg=-(xx/tau + qx+bb+kap);
            double den=1.;
            if (!frozen) {
                for (int j=0;j<n;++j) { gx[j]=q[j]+2.*px[j]/tau; bx[j]=-q[j]; }
                solve_linear(bx,b,a,y);
                correction(bx,a,y,work,wp,we,sz);
                den=dot(gx,a)+dot(b,y)-(xx/tau+kap)/tau;
                if (gen && den>=-1.e-10*(1.+std::abs(qx)+std::abs(bb))) {
                    for (int i=0;i<n;++i) work[i]=a[i]-x[i]/tau;
                    mulP(work,wp);
                    mulA(a,we);
                    for (int i=0;i<m;++i) we[i]-=b[i];
                    den=-dot(work,wp)-std::max(0.,dot(y,we))-kap/tau;
                }
                if (!(den<0.) || !std::isfinite(den)) break;
            }
            auto direction=[&](Step& dp,double sg,double wt) {
                const double target=sg*mu, rct=-tau*kap+target-wt*df.t*df.k;
                const double mul=1.-sg;
                for (int j=0;j<n;++j) bx[j]=mul*rb[j];
                for (int i=0;i<m;++i) bz[i]=mul*ra[i]+s[i]-
                    ((eq[i]||issoc[i])?0.:(target-wt*df.s[i]*df.z[i])/z[i]);
                if (target || wt)
                    for (const auto& block:soc)
                        block.correct(target,wt,df.s.data()+block.a,df.z.data()+block.a,
                                      bz.data()+block.a);
                solve_linear(bx,bz,dp.x,dp.z);
                correction(bx,dp.x,dp.z,work,wp,we,sz);
                double corr=0.;
                if (gen && !frozen && wt && !P.empty()) {
                    for (int j=0;j<n;++j) work[j]=df.x[j]-x[j]*df.t/tau;
                    mulP(work,wp);
                    corr=wt*dot(work,wp)/tau;
                    if (!std::isfinite(corr) || std::abs(corr)>5.*(std::abs(rg)+mu)) corr=0.;
                }
                dp.t=frozen?0.:(mul*rg-corr-rct/tau-dot(gx,dp.x)-dot(b,dp.z))/den;
                dp.k=frozen?-kap:(rct-kap*dp.t)/tau;
                for (int j=0;j<n;++j) dp.x[j]+=dp.t*a[j];
                mulA(dp.x,dp.s);
                for (int i=0;i<m;++i) {
                    dp.z[i]+=dp.t*y[i];
                    dp.s[i]=eq[i]?0.:mul*ra[i]+b[i]*dp.t-dp.s[i];
                }
            };
            direction(df,0.,0.);
            double t=step(s,z,tau,kap,df);
            double sg=std::clamp(std::pow(1.-t,3.),1.e-6,.98);
            direction(dn,sg,has_exp?std::clamp(t,.05,1.):std::clamp(t/.55,.08,1.));
            double ln=step(s,z,tau,kap,dn);
            if (ln<.055) {
                direction(dn,std::max(.22,sg),0.);
                ln=step(s,z,tau,kap,dn);
            }
            const double alpha=ln*(has_exp?.99:.995);
            if (!(alpha>1.e-9) || !std::isfinite(alpha)) break;
            for (int j=0;j<n;++j) x[j]+=alpha*dn.x[j];
            for (int i=0;i<m;++i) {s[i]+=alpha*dn.s[i]; z[i]+=alpha*dn.z[i];}
            tau+=alpha*dn.t; kap+=alpha*dn.k;
            if (!(tau>0.) || !(kap>0.)) break;
        }
        if (!has && kap/tau>1.e5) {
            save(1.);
            return 2;
        }
        return has?1:0;
    }

    int split(double* xo,double* zo,const double* wx,const double* wz) {
        if (!gen || !soc.empty() || n<1 || m<1) return 0;
        V x(n),v(m),s(m),z(m),u(m),a(n),bz(m),bx(n),ax(m),az(n),px(n),
          gx(n),gv(m),xp(n),zp(m),H(m),sc(m),saved_x(n),saved_v(m);
        V prev_x,prev_v,prev_gx,prev_gv;
        std::vector<V> ys,xs,vs;
        double rho=.45;
        const double sig=2.e-6;
        const double cut=4.e-5*std::max(1.,infnorm(q));
        auto prepare=[&]() {
            for (int j=0;j<m;++j) W[j]=rho*(eq[j]?75.:1.);
            return gen->factor(W,true,sig);
        };
        if (!prepare()) return 0;
        if (wx && wz) {
            for (int j=0;j<n;++j) x[order[j]]=wx[j]/D[order[j]];
            mulA(x,v);
            for (int j=0;j<m;++j) v[j]+=wz[j]*c/E[j]/W[j];
        }
        double best=1.e20,prevsize=0.;
        int polishes=0,lastadapt=-60;
        bool out=false,accel=false;
        auto save=[&](const V& x,const V& z) {
            for (int j=0;j<n;++j) xo[j]=D[order[j]]*x[order[j]];
            for (int j=0;j<m;++j) zo[j]=E[j]*z[j]/c;
            out=true;
        };
        auto measure=[&](const V& x,const V& z) {
            mulA(x,ax); mulP(x,px); mulAt(z,az);
            double pi=0.,du=0.,a0=0.,d0=0.;
            for (int j=0;j<m;++j) {
                const double h=(ax[j]-b[j])/E[j];
                pi=std::max(pi,eq[j]?std::abs(h):h);
                a0=std::max(a0,std::abs(ax[j]/E[j]));
            }
            for (int j=0;j<n;++j) {
                du=std::max(du,std::abs(px[j]+q[j]+az[j])/D[j]/c);
                d0=std::max(d0,std::max({std::abs(px[j]),std::abs(q[j]),std::abs(az[j])})/D[j]/c);
            }
            const double net=dot(px,x),vx=dot(q,x),bz=dot(b,z);
            const double gap=std::abs(net+vx+bz)/std::max({c,std::abs(.5*net+vx),std::abs(-.5*net-bz)});
            return std::max({pi/(1.+std::max(a0,bmax)),du/(1.+d0),gap});
        };
        for (int it=0;it<460 && Clock::now()<stop;++it) {
            for (int j=0;j<m;++j) {
                u[j]=eq[j]?b[j]:std::min(v[j],b[j]);
                bz[j]=2.*u[j]-v[j];
                z[j]=W[j]*(v[j]-u[j]);
            }
            for (int j=0;j<n;++j) bx[j]=sig*x[j]-q[j];
            gen->solve(W,bx,bz,a,sc);
            mulA(a,ax);
            for (int j=0;j<n;++j) gx[j]=1.57*(a[j]-x[j]);
            for (int j=0;j<m;++j) gv[j]=1.57*(ax[j]-u[j]);
            const double size=std::max(infnorm(gv),.15*infnorm(gx));
            if (!std::isfinite(size) || size>1.e12) break;
            if (accel) {
                if (size>2.2*prevsize+1.e-11) {
                    x=saved_x;
                    v=saved_v;
                    prev_x.clear();
                    xs.clear(); vs.clear(); ys.clear();
                    accel=false;
                    continue;
                }
                accel=false;
            }
            if (it%8==0) {
                double r=measure(x,z);
                if (r<best && r<.05) {best=r;save(x,z);}
                if (r<3.e-9) return 1;
                if (it>12 && polishes<3 && size<1.e-3 && it>=polishes*44) {
                    ++polishes;
                    xp=x; zp=z;
                    for (int j=0;j<m;++j) {
                        const bool active=eq[j] || z[j]>cut;
                        zp[j]=active?z[j]:0.;
                        H[j]=active?3.e10:1.e-13;
                        bz[j]=active?b[j]-ax[j]:0.;
                    }
                    // measure() above left ax and px at the current iterate.
                    mulAt(zp,az);
                    for (int j=0;j<n;++j) bx[j]=-px[j]-q[j]-az[j];
                    if (gen->factor(H)) {
                        gen->solve(H,bx,bz,a,sc);
                        for (int j=0;j<n;++j) xp[j]+=a[j];
                        for (int j=0;j<m;++j)
                            zp[j]=eq[j]?zp[j]+sc[j]:std::max(0.,zp[j]+sc[j]);
                        double re=measure(xp,zp);
                        if (re<best && re<.05) {best=re;save(xp,zp);}
                        if (re<4.e-9) return 1;
                    }
                    if (!prepare()) break;
                } else if (it>24 && it-lastadapt>42) {
                    const double a=infnorm(gv),b0=infnorm(gx);
                    if (a>8.*b0 && rho<400.) {
                        for (int j=0;j<m;++j) v[j]=u[j]+.25*(v[j]-u[j]);
                        rho*=4.;
                    } else if (b0>16.*a && rho>.002) {
                        for (int j=0;j<m;++j) v[j]=u[j]+4.*(v[j]-u[j]);
                        rho*=.25;
                    } else goto mix;
                    lastadapt=it;
                    if (!prepare()) break;
                    prev_x.clear(); xs.clear(); vs.clear(); ys.clear();
                    continue;
                }
            }
mix:
            if (n+m<41000 && it>=9) {
                if (!prev_x.empty()) {
                    V delta(n+m),sx(n),sv(m);
                    for (int j=0;j<n;++j) {
                        delta[j]=.2*(gx[j]-prev_gx[j]);
                        sx[j]=x[j]-prev_x[j]+gx[j]-prev_gx[j];
                    }
                    for (int j=0;j<m;++j) {
                        delta[n+j]=gv[j]-prev_gv[j];
                        sv[j]=v[j]-prev_v[j]+gv[j]-prev_gv[j];
                    }
                    if (ys.size()==2) {ys.erase(ys.begin()); xs.erase(xs.begin()); vs.erase(vs.begin());}
                    ys.push_back(std::move(delta)); xs.push_back(std::move(sx)); vs.push_back(std::move(sv));
                }
                prev_x=x; prev_v=v; prev_gx=gx; prev_gv=gv;
            }
            for (int j=0;j<n;++j) x[j]+=gx[j];
            for (int j=0;j<m;++j) v[j]+=gv[j];
            if (ys.size()==2 && it%3==0) {
                V r(n+m);
                for (int j=0;j<n;++j) r[j]=.2*gx[j];
                std::copy(gv.begin(),gv.end(),r.begin()+n);
                double a=dot(ys[0],ys[0]),d=dot(ys[1],ys[1]),b=dot(ys[0],ys[1]);
                double reg=2.e-8*(a+d)+1.e-32;
                a+=reg; d+=reg;
                const double det=a*d-b*b;
                double v0=dot(ys[0],r),v1=dot(ys[1],r);
                if (det>0.) {
                    double g0=(d*v0-b*v1)/det,g1=(a*v1-b*v0)/det;
                    if (std::max(std::abs(g0),std::abs(g1))<8.) {
                        prevsize=size; saved_x=x; saved_v=v; accel=true;
                        for (int j=0;j<n;++j) x[j]-=g0*xs[0][j]+g1*xs[1][j];
                        for (int j=0;j<m;++j) v[j]-=g0*vs[0][j]+g1*vs[1][j];
                    }
                }
            }
        }
        return out?1:0;
    }
};

bool rcm(int n,int m,const int* ap,const int* ai,const int* pp,const int* pi,
         I& perm,Clock::time_point stop) {
    if (n>210000 || ap[m]>2200000) return false;
    std::vector<I> adj(n);
    for (int j=0;j<n;++j)
        for (int k=pp[j];k<pp[j+1];++k)
            if (pi[k]!=j) adj[j].push_back(pi[k]);
    long long pairs=0;
    for (int i=0;i<m;++i) {
        const int d=ap[i+1]-ap[i];
        if (d>15) return false;
        pairs+=d*(d-1);
        if (pairs>8000000) return false;
        for (int k=ap[i];k<ap[i+1];++k)
            for (int l=ap[i];l<ap[i+1];++l)
                if (l!=k) adj[ai[k]].push_back(ai[l]);
    }
    if (Clock::now()>stop) return false;
    I deg(n), who(n), path;
    path.reserve(n);
    for (int i=0;i<n;++i) {
        auto& a=adj[i];
        std::sort(a.begin(),a.end());
        a.erase(std::unique(a.begin(),a.end()),a.end());
        deg[i]=a.size();
        who[i]=i;
    }
    auto key=[&](int a,int b){return deg[a]<deg[b] || (deg[a]==deg[b] && a<b);};
    std::sort(who.begin(),who.end(),key);
    std::fill(perm.begin(),perm.end(),-1);
    for (int first:who) if (perm[first]<0) {
        auto begin=path.size();
        perm[first]=n-1-static_cast<int>(path.size());
        path.push_back(first);
        for (auto k=begin;k<path.size();++k) {
            const int u=path[k];
            auto& a=adj[u];
            std::sort(a.begin(),a.end(),key);
            for (int v:a) if (perm[v]<0) {
                perm[v]=n-1-static_cast<int>(path.size());
                path.push_back(v);
            }
        }
    }
    return Clock::now()<stop;
}

extern "C" int conic_core_qp(int n,int m,const int* ap,const int* ai,const double* a,
               const int* pp,const int* pi,const double* p,const double* q,const double* b,
               const unsigned char* eq,double* x,double* z,double secs,
               void* potrf,void* potrs,void* sytrf,void* sytrs,void* syrk,int plain,
               const double* wx,const double* wz) {
    try {
        auto stop=Clock::now()+std::chrono::microseconds(static_cast<long long>(secs*1.e6));
        I order(n);
        std::iota(order.begin(),order.end(),0);
        Qp qp(n,m,-1,order,ap,ai,a,pp,pi,p,q,b,eq,stop,
              conic::Lapack(potrf,potrs,sytrf,sytrs,syrk),plain!=0);
        if (plain==2) return qp.split(x,z,wx,wz);
        int ans=qp.solve(x,z,nullptr,wx,wz);
        if (ans==2 || qp.quality<3.e-9 || Clock::now()+std::chrono::milliseconds(50)>stop)
            return ans;
        V xp(n),zp(m);
        double old=qp.quality;
        qp.switch_mode();
        int alt=qp.solve(xp.data(),zp.data(),nullptr,ans==1?x:wx,ans==1?z:wz);
        if (alt==2 || (alt==1 && (ans!=1 || qp.quality<old))) {
            std::copy(xp.begin(),xp.end(),x);
            std::copy(zp.begin(),zp.end(),z);
            return alt;
        }
        return ans;
    } catch (...) {
        return 0;
    }
}

extern "C" int conic_core_soc(int n,int m,const int* ap,const int* ai,const double* a,
               const int* pp,const int* pi,const double* p,const double* q,const double* b,
               const unsigned char* eq,double* x,double* z,double* s,double secs,
               void* potrf,void* potrs,void* sytrf,void* sytrs,void* syrk,
               int blocks,const int* starts,const int* dims,const unsigned char* expmark,void* svd) {
    try {
        auto stop=Clock::now()+std::chrono::microseconds(static_cast<long long>(secs*1.e6));
        I order(n);
        std::iota(order.begin(),order.end(),0);
        Qp qp(n,m,-1,order,ap,ai,a,pp,pi,p,q,b,eq,stop,
              conic::Lapack(potrf,potrs,sytrf,sytrs,syrk),false,blocks,starts,dims,expmark,svd);
        return qp.solve(x,z,s);
    } catch (...) {
        return 0;
    }
}
}

extern "C" int conic_band_qp(int n,int m,const int* ap,const int* ai,const double* a,
               const int* pp,const int* pi,const double* p,const double* q,const double* b,
               const unsigned char* eq,double* x,double* z,double secs) {
    try {
        auto stop=Clock::now()+std::chrono::microseconds(static_cast<long long>(secs*1.e6));
        int bw=0;
        I perm(n);
        std::iota(perm.begin(),perm.end(),0);
        for (int attempt=0;attempt<3;++attempt) {
            if (attempt==1) {
                for (int j=0;j<n;++j) perm[j]=2*j<n?2*j:2*(n-j)-1;
                for (int j=0;j<n;++j) if (perm[j]>=n) perm[j]=n-1;
            }
            if (attempt==2 && !rcm(n,m,ap,ai,pp,pi,perm,stop)) return 0;
            bw=0;
            auto width=[&](int a,int b){ bw=std::max(bw,std::abs(perm[a]-perm[b])); };
            for (int j=0;j<n && bw<=14;++j)
                for (int k=pp[j];k<pp[j+1];++k) width(j,pi[k]);
            for (int i=0;i<m && bw<=14;++i) {
                int lo=n,hi=-1;
                for (int k=ap[i];k<ap[i+1];++k) {
                    lo=std::min(lo,perm[ai[k]]); hi=std::max(hi,perm[ai[k]]);
                }
                bw=std::max(bw,hi-lo);
            }
            if (bw<=14) {
                if (Clock::now()>stop || (bw+1.)*n>6.5e6) return 0;
                Qp qp(n,m,bw,perm,ap,ai,a,pp,pi,p,q,b,eq,stop);
                return qp.solve(x,z);
            }
        }
        return 0;
    } catch (...) {
        return 0;
    }
}
