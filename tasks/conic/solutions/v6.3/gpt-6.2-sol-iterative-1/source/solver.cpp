#include "kkt.hpp"
#include "presolve.hpp"
#include "chordal.hpp"
#include "poly.hpp"
#include "manifold.hpp"
#include "box.hpp"
#include "substitute.hpp"
#include "dr.hpp"
#include "epigraph.hpp"
#include "outer_native.hpp"

namespace conic {

struct Check {
    const Problem& p;
    const double qmax, bmax, amax, pmax;
    Vec ax,px,atz,sc;
    explicit Check(const Problem& prob):p(prob),qmax(norm(p.q,p.n())),bmax(norm(p.b,p.m())),
        amax(norm(p.A.v,p.A.nz())), pmax(norm(p.P.v,p.P.nz())),
        ax(p.m()),px(p.n()),atz(p.n()),sc(p.m()) {}

    bool optimal(Result& r, double tol=9.75e-9, bool polish=true, bool verbose=false) {
        if(!finite(r.x)||!finite(r.s)||!finite(r.z)) return false;
        std::fill(ax.begin(),ax.end(),0.);
        std::fill(px.begin(),px.end(),0.);
        std::fill(atz.begin(),atz.end(),0.);
        p.A.add(r.x.data(),ax.data());
        p.P.add(r.x.data(),px.data());
        p.A.add_transpose(r.z.data(),atz.data());
        if(!finite(ax)||!finite(px)||!finite(atz)) return false;
        double rp=0,rd=0;
        for(int i=0;i<p.m();++i) rp=std::max(rp,std::abs(ax[i]+r.s[i]-p.b[i]));
        const double scale=1+std::max({norm(px),norm(atz),qmax});
        rp/=1+std::max({norm(ax),norm(r.s),bmax});
        for(int i=0;i<p.n();++i) rd=std::max(rd,std::abs((px[i]+p.q[i])+atz[i]));
        rd/=scale;
        const double xpx=.5*dot(r.x,px), qx=dot(r.x.data(),p.q,p.n());
        double bz=dot(r.z.data(),p.b,p.m());
        double po=xpx+qx, du=-xpx-bz;
        double gap=std::abs(po-du)/std::max({1.,std::abs(po),std::abs(du)});
        if (verbose) std::fprintf(stderr,"validate %.3g %.3g %.3g; p=%g d=%g\n",rp,rd,gap,po,du);
        if(!std::isfinite(gap)||!std::isfinite(dot(r.s,r.z))) return false;
        if (rp>tol || rd>tol) return false;
        double multiplier=1.;
        if(gap>tol && polish && bz!=0) {
            // A positive rescale preserves dual cone membership and can remove
            // a small objective mismatch without harming the stationarity test.
            double change=-(po-du)/bz;
            if(std::abs(change)<1e-5) {
                multiplier=1+change;
                double err=0, zterm=0;
                for(int i=0;i<p.n();++i) {
                    err=std::max(err,std::abs((px[i]+p.q[i])+multiplier*atz[i]));
                    zterm=std::max(zterm,std::abs(multiplier*atz[i]));
                }
                if(err>tol*(1+std::max({norm(px),zterm,qmax}))) multiplier=1.;
            }
        }
        if (gap>tol && multiplier==1.) return false;
        if (cone_defect(p,r.s,false,tol)>tol) return false;
        if(multiplier!=1.) {
            for(double& x:r.z) x*=multiplier;
            double bz_new=dot(r.z.data(),p.b,p.m());
            if(std::abs(po+xpx+bz_new)>tol*std::max({1.,std::abs(po),std::abs(-xpx-bz_new)}))
                return false;
        }
        if(cone_defect(p,r.z,true,tol)>tol) return false;
        r.status=Solved;return true;
    }

    bool primal(Result& r,double tol=9.75e-9) {
        double zn=norm(r.z);
        if(!(zn>0 && bmax>0) || !finite(r.z)) return false;
        double beta=0;
        for(int k=0;k<p.m();++k) {
            r.z[k]/=zn;
            beta-=(p.b[k]/bmax)*r.z[k];
        }
        if(!(beta>eps)) return false;
        double ascale=amax>0?amax:1.;
        for(int j=0;j<p.n();++j) {
            double v=0;
            for(int k=p.A.p[j];k<p.A.p[j+1];++k) v+=(p.A.v[k]/ascale)*r.z[p.A.i[k]];
            if(!(std::abs(v)<=tol*beta)) return false;
        }
        if(cone_defect(p,r.z,true,tol*beta)>tol*beta) return false;
        r.status=PrimalInfeasible;return true;
    }

    bool dual(Result& r,double tol=9.75e-9) {
        double xn=norm(r.x);
        if(!(xn>0 && qmax>0) || !finite(r.x)) return false;
        double alpha=0;
        for(int j=0;j<p.n();++j) {
            r.x[j]/=xn;
            alpha-=(p.q[j]/qmax)*r.x[j];
        }
        if(!(alpha>eps)) return false;
        const double h=pmax>0?pmax:1., a=amax>0?amax:1.;
        std::fill(px.begin(),px.end(),0.);
        std::fill(sc.begin(),sc.end(),0.);
        for(int j=0;j<p.n();++j) {
            double u=r.x[j];
            if(u==0) continue;
            for(int k=p.P.p[j];k<p.P.p[j+1];++k) px[p.P.i[k]]+=(p.P.v[k]/h)*u;
            for(int k=p.A.p[j];k<p.A.p[j+1];++k) sc[p.A.i[k]]-=(p.A.v[k]/a)*u;
        }
        if(norm(px)>tol*alpha || !finite(px) || !finite(sc)) return false;
        if(cone_defect(p,sc,false,tol*alpha)>tol*alpha) return false;
        for(int k=0;k<p.m();++k) r.s[k]=a*sc[k];
        if(!finite(r.s)) return false;
        r.status=DualInfeasible;return true;
    }
};

struct Scaled {
    Vec av,pv,q,b,d,e;
    double c=1.;
    Problem p;
    Scaled(const Problem& o, int rounds,int variant=0):av(o.A.nz()),pv(o.P.nz()),
        q(o.n()),b(o.m()),d(o.n(),1.),e(o.m(),1.),p(o) {
        if(o.A.nz()) std::copy(o.A.v,o.A.v+o.A.nz(),av.begin());
        if(o.P.nz()) std::copy(o.P.v,o.P.v+o.P.nz(),pv.begin());
        if(o.n()) std::copy(o.q,o.q+o.n(),q.begin());
        if(o.m()) std::copy(o.b,o.b+o.m(),b.begin());
        p.A.v=av.data();p.P.v=pv.data();p.q=q.data();p.b=b.data();
        p.cones.clear();
        for(auto B:o.cones) {
            if(!p.cones.empty() && B.kind<=Linear && p.cones.back().kind==B.kind) {
                p.cones.back().width+=B.width;p.cones.back().dim+=B.width;
            } else p.cones.push_back(B);
        }
        Vec dr(d.size()),er(e.size()),pr(d.size()),psdscale;
        const double dcap=variant==28?1e13:1e7;
        const double ccap=variant==28?1e16:1e9;
        for(int iter=0;iter<rounds;++iter) {
            std::fill(dr.begin(),dr.end(),0.);
            std::fill(er.begin(),er.end(),0.);
            for(int j=0;j<p.n();++j) {
                for(int k=p.P.p[j];k<p.P.p[j+1];++k) dr[j]=std::max(dr[j],std::abs(pv[k]));
                for(int k=p.A.p[j];k<p.A.p[j+1];++k) {
                    int i=p.A.i[k];double a=std::abs(av[k]);
                    dr[j]=std::max(dr[j],a);
                    er[i]=std::max(er[i],a);
                }
            }
            for(auto B:p.cones) if(B.kind>=Soc) {
                if(variant==28 && B.kind==PSD) {
                    const int dd=B.dim;
                    psdscale.assign(dd,0.);
                    int k=B.off;
                    for(int j=0;j<dd;++j)for(int i=0;i<=j;++i,++k) {
                        double a=std::max(er[k],std::abs(b[k]));
                        psdscale[i]=std::max(psdscale[i],a);psdscale[j]=std::max(psdscale[j],a);
                    }
                    for(double& a:psdscale)a=std::sqrt(a>0?a:1.);
                    k=B.off;
                    for(int j=0;j<dd;++j)for(int i=0;i<=j;++i,++k)
                        er[k]=psdscale[i]*psdscale[j];
                    continue;
                }
                double a=norm(er.data()+B.off,B.width);
                if((variant==18 || variant==28) && a>0) {
                    double t=0.;int cnt=0;
                    for(int j=B.off;j<B.off+B.width;++j)if(er[j]>0){t+=er[j];++cnt;}
                    if(cnt>0)a=std::sqrt(a*t/cnt);
                }
                std::fill(er.begin()+B.off,er.begin()+B.off+B.width,a);
            }
            for(int j=0;j<p.n();++j) {
                dr[j]=std::clamp(dr[j]>0 ? 1./std::sqrt(dr[j]) : 1.,1./(dcap*d[j]),dcap/d[j]);
                d[j]*=dr[j];q[j]*=dr[j];
            }
            for(int i=0;i<p.m();++i) {
                er[i]=std::clamp(er[i]>0 ? 1./std::sqrt(er[i]) : 1.,1./(dcap*e[i]),dcap/e[i]);
                e[i]*=er[i];b[i]*=er[i];
            }
            double sum=0;int cnt=0;
            for(int j=0;j<p.n();++j) {
                double v=0.;
                for(int k=p.P.p[j];k<p.P.p[j+1];++k) {
                    pv[k]*=dr[p.P.i[k]]*dr[j];v=std::max(v,std::abs(pv[k]));
                }
                for(int k=p.A.p[j];k<p.A.p[j+1];++k) av[k]*=er[p.A.i[k]]*dr[j];
                sum+=v;cnt+=v>0;
            }
            double maxcost=std::max(norm(q),cnt?sum/cnt:0.);
            if(maxcost>0) {
                double cs=std::clamp(1./maxcost,1./(ccap*c),ccap/c);
                c*=cs;
                for(double& x:q)x*=cs;
                for(double& x:pv)x*=cs;
            }
        }
        if(variant==17 || variant==18 || variant==28) {
            const double nb=norm(b);
            if(nb>0.) {
                const double f=std::clamp(variant==17?std::sqrt(nb):nb,1e-6,1e7);
                for(double& v:b)v/=f;
                for(double& v:e)v/=f;
                for(double& v:d)v*=f;
                const double t=1./std::max({1.,f*norm(pv),norm(q)});
                c*=t/f;
                for(double& v:pv)v*=f*t;
                for(double& v:q)v*=t;
            }
        }
    }
};

struct Direction {
    Vec x,s,z;
    double t=0,k=0;
    Direction(int n,int m):x(n),s(m),z(m) {}
};

using Gate=std::function<bool(const Result&)>;

// A secondary affine projection removes the last few digits of stationarity
// and objective error without asking the barrier system to become more singular.
bool repair_dual(const Problem& p,const Vec& x,Vec& z,double until,
                 const std::function<bool(const Vec&)>& accept,
                 double certificate_objective=inf) {
    const int n=p.n(),m=p.m();
    if(n+m>90000 || p.A.nz()>900000 || m==0 || clock_time()>until) return false;
    const double bscale=norm(p.b,m);
    const int k=n+(bscale>0);
    Int signs(k+m,-1);
    std::fill(signs.begin(),signs.begin()+k,1);
    std::vector<Entry> entries;
    entries.reserve(size_t(k)+2*size_t(m)+p.A.nz());
    for(int j=0;j<k+m;++j)entries.push_back({j,j,j<k?0.:-1.});
    for(int j=0;j<n;++j)
        for(int i=p.A.p[j];i<p.A.p[j+1];++i)entries.push_back({j,k+p.A.i[i],p.A.v[i]});
    if(k>n) for(int i=0;i<m;++i)if(p.b[i]!=0)entries.push_back({n,k+i,p.b[i]/bscale});
    const double zmax=std::max(1e-20,norm(z));
    for(auto c:p.cones) if(c.kind==Linear) {
        for(int i=c.off;i<c.off+c.width;++i) {
            double w=.002+std::min(1.,std::abs(z[i])/zmax);
            entries[k+i].v=-1./w;
        }
    }
    LinearSystem sys(k+m,signs,entries,false);
    if(!sys.ok || sys.work_est>1.2e8 || clock_time()>until)return false;
    Vec values, rhs(k+m,0.), ans(k+m), target(n), at(n);
    values.reserve(entries.size());
    for(auto e:entries)values.push_back(e.v);
    if(!sys.factor(values,1e-10) && !sys.factor(values,1e-8,true))return false;
    double obj=certificate_objective;
    if(!std::isfinite(certificate_objective)) {
        p.P.add(x.data(),target.data());
        obj=-dot(x,target)-dot(x.data(),p.q,n);
        for(int j=0;j<n;++j)target[j]=-target[j]-p.q[j];
    }
    Vec trial(z),mat,work,eig;
    Int iw;
    for(int it=0;it<8 && clock_time()<until;++it) {
        std::fill(at.begin(),at.end(),0.);
        p.A.add_transpose(trial.data(),at.data());
        for(int j=0;j<n;++j)rhs[j]=target[j]-at[j];
        if(k>n)rhs[n]=(obj-dot_precise(trial.data(),p.b,m))/bscale;
        if(!sys.solve(rhs,ans,4))return false;
        for(int i=0;i<m;++i)trial[i]+=ans[k+i];
        if(accept(trial))return true;
        // Re-enter the dual cones, then remove the resulting affine defect.
        for(auto c:p.cones) {
            double* a=trial.data()+c.off;
            if(c.kind==Linear) {
                for(int i=0;i<c.width;++i)a[i]=std::max(0.,a[i]);
            } else if(c.kind==Soc) {
                double l=std::sqrt(dot(a+1,a+1,c.width-1));
                if(l>a[0]) {
                    const double s0=.5*(l+a[0]);
                    if(s0>0 && std::isfinite(l)) {
                        double t=s0/l;
                        for(int i=1;i<c.width;++i)a[i]*=t;
                        a[0]=s0;
                    } else std::fill(a,a+c.width,0.);
                }
            } else if(c.kind==PSD) {
                const int d=c.dim;
                mat.resize(size_t(d)*d);
                smat(d,a,mat.data());
                double l=smallest_eigen(d,mat.data(),work,iw,eig);
                if(l<0 && std::isfinite(l))for(int i=0;i<d;++i)a[i*(i+3)/2]-=l;
            } else if(c.kind==Exp) {
                if(a[0]>=0) {
                    a[0]=0.;a[1]=std::max(0.,a[1]);a[2]=std::max(0.,a[2]);
                } else {
                    double r=-a[0];
                    a[2]=std::max(a[2],1e-13*std::max(r,std::abs(a[1])));
                    a[1]=std::max(a[1],-r*(1+std::log(a[2])-std::log(r)));
                }
            }
        }
        if(accept(trial))return true;
    }
    return false;
}

class Interior {
    const Problem& orig;
    const int n,m, mode;
    const double stop;
    const bool verbose;
    const Gate& lifted_check;
    Scaled scaled;
    const Problem& p;
    KKT sys;
    Check verify;
    Vec x,s,z,ax,px,atz,rd,rp,gs,kxd,kzp,kxq,kzb,r1,r2,v, pv, qneg;
    Direction aff,dir;
    double tau=1., kappa=1., xpx=0., qx=0., bz=0., gap=0., mu=1., den=0., oldstep=1;
    double pr=1., dr=1., gp=1.;
    double best=3e-3, repair_best=1e-5;
    Vec bx,bs,bz0;
    int iter=0;
    bool fixed_tau=false;

    bool checked(Result& r,bool good) const {
        if(good && (!lifted_check || lifted_check(r)))return true;
        r.status=Unresolved;
        return false;
    }

    void set_initial() {
        for(const auto& c:p.cones) {
            switch(c.kind) {
            case Zero:break;
            case Linear:
                std::fill(s.begin()+c.off,s.begin()+c.off+c.width,1.);
                std::fill(z.begin()+c.off,z.begin()+c.off+c.width,1.);
                break;
            case Soc:s[c.off]=z[c.off]=1.;break;
            case Exp:
                s[c.off]=-1.;s[c.off+1]=1.;s[c.off+2]=2.;
                z[c.off]=-1.;z[c.off+2]=1.;
                break;
            default:
                for(int j=0;j<c.dim;++j) s[c.off+j*(j+3)/2]=z[c.off+j*(j+3)/2]=1.;
                break;
            }
        }
    }

    void improve_initial() {
        if(!sys.update(s,z,1.,true,1e-9) ||
           !sys.solve(qneg,scaled.b,kxd,kzp,1)) return;
        if(p.P.nz()==0) {
            if(!sys.solve(r1,scaled.b,kxd,kzb,1)) return;
            if(!sys.solve(qneg,r2,kxq,kzp,1)) return;
            for(int j=0;j<m;++j) ax[j]=-kzb[j];
        } else {
            for(int j=0;j<m;++j) ax[j]=-kzp[j];
        }
        if(norm(kxd)>1e12||norm(kzp)>1e12) return;
        x=kxd;
        for(auto& B:sys.blocks) {
            auto c=B.c;
            double* u=ax.data()+c.off, *v=kzp.data()+c.off;
            double* ss=s.data()+c.off, *zz=z.data()+c.off;
            if(c.kind==Exp) continue;
            if(c.kind==Zero) {
                std::copy(v,v+c.width,zz);
                continue;
            }
            double sl=0.,zl=0.;
            if(c.kind==Linear) {
                sl=*std::min_element(u,u+c.width);zl=*std::min_element(v,v+c.width);
            } else if(c.kind==Soc) {
                sl=u[0]-std::sqrt(dot(u+1,u+1,c.width-1));
                zl=v[0]-std::sqrt(dot(v+1,v+1,c.width-1));
            } else {
                auto& S=*B.psd;
                smat(c.dim,u,S.a.data());smat(c.dim,v,S.b.data());
                sl=smallest_eigen(c.dim,S.a.data(),S.work,S.iw,S.eig);
                zl=smallest_eigen(c.dim,S.b.data(),S.work,S.iw,S.eig);
            }
            const double as=std::max(0.,.8-sl), az=std::max(0.,.8-zl);
            if(!std::isfinite(as+az)) continue;
            for(int j=0;j<c.width;++j) {
                ss[j]=u[j]+as*ss[j];zz[j]=v[j]+az*zz[j];
            }
        }
    }

    void residuals() {
        std::fill(ax.begin(),ax.end(),0.);
        std::fill(px.begin(),px.end(),0.);
        std::fill(atz.begin(),atz.end(),0.);
        p.A.add(x.data(),ax.data());
        p.P.add(x.data(),px.data());
        p.A.add_transpose(z.data(),atz.data());
        double rpmax=0., axmax=0., smax=0., rdmax=0., pxmax=0., atmax=0.;
        for(int i=0;i<m;++i) {
            rp[i]=ax[i]+s[i]-tau*p.b[i];
            double e=scaled.e[i];
            rpmax=std::max(rpmax,std::abs(rp[i]/e));
            axmax=std::max(axmax,std::abs(ax[i]/e));
            smax=std::max(smax,std::abs(s[i]/e));
        }
        for(int i=0;i<n;++i) {
            rd[i]=px[i]+tau*p.q[i]+atz[i];
            double d=scaled.d[i]*scaled.c;
            rdmax=std::max(rdmax,std::abs(rd[i]/d));
            pxmax=std::max(pxmax,std::abs(px[i]/d));
            atmax=std::max(atmax,std::abs(atz[i]/d));
        }
        pr=rpmax/(tau+std::max({axmax,smax,tau*verify.bmax}));
        dr=rdmax/(tau+std::max({pxmax,atmax,tau*verify.qmax}));
        xpx=dot(x,px);qx=dot(x,scaled.q);bz=dot(z,scaled.b);
        gap=xpx/tau+qx+bz+kappa;
        const double po=.5*xpx+tau*qx, du=-.5*xpx-tau*bz;
        gp=std::abs(po-du)/std::max({tau*tau*scaled.c,std::abs(po),std::abs(du)});
        mu=fixed_tau?dot(s,z)/std::max(1,sys.degree-1):(dot(s,z)+tau*kappa)/sys.degree;
    }

    void unscale(Result& r,bool opt) const {
        double a=opt ? 1./tau : 1.;
        for(int j=0;j<n;++j) r.x[j]=x[j]*(a*scaled.d[j]);
        for(int j=0;j<m;++j) {
            r.s[j]=s[j]*(a/scaled.e[j]);
            r.z[j]=z[j]*((a/scaled.c)*scaled.e[j]);
        }
    }

    bool polish(Result& r,const Vec& xs,const Vec& ss,const Vec& zz,double div) {
        for(int i=0;i<n;++i)r.x[i]=xs[i]*scaled.d[i]/div;
        for(int i=0;i<m;++i){
            r.s[i]=ss[i]/(div*scaled.e[i]);r.z[i]=zz[i]*(scaled.e[i]/(div*scaled.c));
        }
        if(checked(r,verify.optimal(r)))return true;
        Vec old_slack=r.s;
        Vec plain(orig.b,orig.b+m);
        orig.A.add(r.x.data(),plain.data(),-1.);
        r.s=plain;
        for(auto c:p.cones) {
            if(c.kind==Zero)std::fill(r.s.begin()+c.off,r.s.begin()+c.off+c.width,0.);
            else if(c.kind==Linear)for(int i=c.off;i<c.off+c.width;++i)r.s[i]=std::max(r.s[i],0.);
        }
        if(checked(r,verify.optimal(r)))return true;
        Vec sx=xs,sz=zz;
        for(double& a:sx)a/=div;
        for(double& a:sz)a/=div;
        auto accept = [&](const Vec& xx,const Vec& ss,const Vec& zz) {
            for(int j=0;j<n;++j)r.x[j]=xx[j]*scaled.d[j];
            for(int j=0;j<m;++j){r.z[j]=zz[j]*scaled.e[j]/scaled.c;r.s[j]=ss[j]/scaled.e[j];}
            return checked(r,verify.optimal(r));
        };
        if(!is_polyhedral(p)) {
            Vec ts=r.s;
            if(semismooth(p,sx,sz,std::min(stop,clock_time()+.17),accept,mode==16?1:0)) {
                if(verbose)std::fprintf(stderr,"accepted cone-face repair\n");
                return true;
            }
            r.s.swap(ts);
            for(int j=0;j<n;++j)r.x[j]=xs[j]*scaled.d[j]/div;
        }
        if(cone_defect(orig,r.s,false,9.7e-9)>9.7e-9 || norm(plain)>2*norm(old_slack)+1.)
            r.s.swap(old_slack);
        if(cone_defect(orig,r.s,false,9.7e-9)>9.7e-9)return false;
        double err=0.;
        for(int i=0;i<m;++i)err=std::max(err,std::abs(r.s[i]-plain[i]));
        if(err>9.75e-9*(1+std::max(norm(verify.ax),std::max(verify.bmax,norm(r.s))))) {
            err=0;
            for(int i=0;i<m;++i)err=std::max(err,std::abs(old_slack[i]-plain[i]));
            if(err>9.75e-9*(1+std::max(norm(verify.ax),std::max(verify.bmax,norm(old_slack)))))return false;
            r.s=std::move(old_slack);
        }
        if(is_polyhedral(p)) {
            Vec original_slack=r.s;
            if(poly_polish(p,sx,sz,std::min(stop,clock_time()+.12),accept,3))return true;
            r.s.swap(original_slack);
        }
        for(int j=0;j<n;++j)r.x[j]=xs[j]*scaled.d[j]/div;
        return repair_dual(p,sx,sz,std::min(stop,clock_time()+.22),[&](const Vec& zz) {
            for(int j=0;j<m;++j)r.z[j]=zz[j]*scaled.e[j]/scaled.c;
            if(checked(r,verify.optimal(r))) {
                if(verbose)std::fprintf(stderr,"accepted affine repair\n");
                return true;
            }
            return false;
        });
    }

    bool save_if_valid(Result& r,bool last=false) {
        if(pr<1.05e-8 && dr<1.05e-8 && (gp<2e-7 || last)) {
            unscale(r,true);
            if(checked(r,verify.optimal(r,9.75e-9,true,verbose)))return true;
        }
        const double merit=std::max({pr,dr,gp});
        if(tau>0 && merit<best) {
            best=merit;bx=x;bs=s;bz0=z;
            for(double& v:bx)v/=tau;
            for(double& v:bs)v/=tau;
            for(double& v:bz0)v/=tau;
        }
        if(mode!=10 && m && ((last && !bx.empty()) ||
                (pr<3e-8 && merit<std::min(4e-8,repair_best*.25)))) {
            repair_best=merit;
            if(last) {if(polish(r,bx,bs,bz0,1.))return true;}
            else if(polish(r,x,s,z,tau))return true;
        }
        if(last || (iter>=4 && (kappa>250*tau || tau<1e-4))) {
            unscale(r,false);
            if(checked(r,verify.primal(r)) || checked(r,verify.dual(r)))return true;
        }
        if(last && kappa>25*tau && n+m<50000 && clock_time()<stop) {
            const double zn=norm(z);
            if(zn>0. && bz < -1e-8*zn*norm(scaled.b)) {
                Vec normalized=z,zero(n,0.), at(n,0.);
                for(double& a:normalized)a/=zn;
                p.A.add_transpose(normalized.data(),at.data());
                if(norm(at)<2e-3*(1.+std::abs(bz/zn)) &&
                    repair_dual(p,zero,normalized,std::min(stop,clock_time()+.18),
                        [&](const Vec& v) {
                            for(int j=0;j<m;++j)r.z[j]=v[j]*scaled.e[j]/scaled.c;
                            return checked(r,verify.primal(r));
                        },bz/zn))return true;
            }
        }
        return false;
    }

    bool rhs_constant() {
        if(fixed_tau)return true;
        if(!sys.solve(qneg,scaled.b,kxq,kzb, n+m>150000 ? 2 : 4)) return false;
        for(int j=0;j<n;++j) v[j]=x[j]/tau-kxq[j];
        std::fill(pv.begin(),pv.end(),0.);
        p.P.add(v.data(),pv.data());
        den=-std::max(0.,dot(v,pv))-std::max(0.,sys.energy(kzb))-kappa/tau;
        return std::isfinite(den) && den<0;
    }

    bool direction(Direction& d,double sigma,double damp) {
        const double b=1.-sigma, center=sigma*mu;
        for(int j=0;j<m;++j) gs[j]=-s[j];
        if(sigma!=0 || damp!=0) {
            for(auto& B:sys.blocks) {
                auto c=B.c;
                double* g=gs.data()+c.off;
                const double* da=aff.s.data()+c.off,*db=aff.z.data()+c.off;
                const double* zz=z.data()+c.off;
                switch(c.kind) {
                case Zero:break;
                case Linear:
                    for(int j=0;j<c.width;++j) g[j]+=(center-damp*da[j]*db[j])/zz[j];
                    break;
                case Soc:B.soc->correct(g,da,db,center,damp);break;
                case Exp:B.exp.correct(g,da,db,center,damp);break;
                default:B.psd->correct(g,da,db,center,damp);break;
                }
            }
        }
        const double gk=-kappa+(center-damp*aff.t*aff.k)/tau;
        for(int j=0;j<n;++j) r1[j]=-b*rd[j];
        for(int j=0;j<m;++j) r2[j]=-b*rp[j]-gs[j];
        if(!sys.solve(r1,r2,kxd,kzp,n+m>150000?2:4)) return false;
        if(fixed_tau) {
            d.t=d.k=0.;
            d.x=kxd;d.z=kzp;
        } else {
            long double rhs=-b*(long double)gap-gk;
            for(int j=0;j<n;++j) rhs-=((long double)p.q[j]+2.L*px[j]/tau)*kxd[j];
            for(int j=0;j<m;++j) rhs-=(long double)p.b[j]*kzp[j];
            d.t=double(rhs/den);
            if(!std::isfinite(d.t)) return false;
            d.k=gk-(kappa/tau)*d.t;
            for(int j=0;j<n;++j) d.x[j]=kxd[j]+d.t*kxq[j];
            for(int j=0;j<m;++j) d.z[j]=kzp[j]+d.t*kzb[j];
        }
        if(sys.eliminates || mode==5 || mode==6) {
            for(int j=0;j<m;++j) d.s[j]=d.t*p.b[j]-b*rp[j];
            p.A.add(d.x.data(),d.s.data(),-1.);
        }
        sys.direction_slack(d.z,gs,d.s);
        if(mode==5 || mode==6) {
            for(int j=0;j<m;++j) d.s[j]=d.t*p.b[j]-b*rp[j];
            p.A.add(d.x.data(),d.s.data(),-1.);
            for(auto c:p.cones) if(c.kind==Zero)
                std::fill(d.s.begin()+c.off,d.s.begin()+c.off+c.width,0.);
        }
        return std::isfinite(d.k) && finite(d.x) && finite(d.s) && finite(d.z);
    }

    double step(Direction& d, bool combine,double upper=1.) {
        double a=upper;
        if(d.t<0) a=std::min(a,-tau/d.t);
        if(d.k<0) a=std::min(a,-kappa/d.k);
        for(auto& B:sys.blocks) {
            auto c=B.c;
            const double* ss=s.data()+c.off,*zz=z.data()+c.off;
            const double* ds=d.s.data()+c.off,*dz=d.z.data()+c.off;
            if(c.kind==Linear) {
                for(int j=0;j<c.width;++j) {
                    if(ds[j]<0) a=std::min(a,-ss[j]/ds[j]);
                    if(dz[j]<0) a=std::min(a,-zz[j]/dz[j]);
                }
            } else if(c.kind==Soc) {
                a=std::min({a,soc_step(ss,ds,c.width),soc_step(zz,dz,c.width)});
            }
        }
        if(a<=0) return 0.;
        for(auto& B:sys.blocks) {
            auto c=B.c;
            const double* ss=s.data()+c.off,*zz=z.data()+c.off;
            const double* ds=d.s.data()+c.off,*dz=d.z.data()+c.off;
            if(c.kind==Exp) {
                a=exp_step(ss,ds,a,false);
                if(a>0) a=exp_step(zz,dz,a,true);
            } else if(c.kind==PSD) {
                a=B.psd->step(ss,ds,a,false);
                if(a>0) a=B.psd->step(zz,dz,a,true);
            }
            if(a<=0) return 0.;
        }
        return combine?a*(mode==2?.96:.994):a;
    }

    void take(const Direction& d,double a) {
        tau+=a*d.t;kappa+=a*d.k;
        for(int j=0;j<n;++j) x[j]+=a*d.x[j];
        for(int j=0;j<m;++j) {s[j]+=a*d.s[j];z[j]+=a*d.z[j];}
        double scale=std::max(tau,kappa);
        if (scale>1e6 || (scale>0 && scale<1e-6)) {
            tau/=scale;kappa/=scale;
            for(double& v:x)v/=scale;
            for(double& v:s)v/=scale;
            for(double& v:z)v/=scale;
        }
    }

    bool nudge() {
        bool changed=false;
        for(auto& B:sys.blocks) {
            const auto c=B.c;
            for(int dual=0;dual<2;++dual) {
                double* v=(dual?z:s).data()+c.off;
                const double vr=norm(v,c.width);
                if(c.kind==Soc && v[0]>0 && soc_inner(v,v,c.width)<=0) {
                    long double square=0.;
                    for(int j=1;j<c.width;++j)square+=(long double)v[j]*v[j];
                    double required=double(std::sqrt(square));
                    if(required-v[0]>1e-12*vr)return false;
                    v[0]=std::nextafter(required,inf);changed=true;
                } else if(c.kind==PSD) {
                    auto& S=*B.psd;
                    smat(c.dim,v,S.a.data());
                    if(chol(c.dim,S.a.data()))continue;
                    smat(c.dim,v,S.a.data());
                    const double ee=smallest_eigen(c.dim,S.a.data(),S.work,S.iw,S.eig);
                    if(ee<-1e-12*std::max(1e-50,vr))return false;
                    double delta=std::max(0.,-ee)+8e-16*std::max(1e-50,vr);
                    for(int j=0;j<c.dim;++j)v[j*(j+3)/2]+=delta;
                    changed=true;
                } else if(c.kind==Exp) {
                    if(dual) {
                        const long double r=-v[0], w=v[2];
                        if(!(r>0&&w>0))continue;
                        const long double lo=-r*(1+std::log(w)-std::log(r));
                        if(v[1]<=lo && lo-v[1]<1e-13*vr) {
                            v[1]=std::nextafter(double(lo),inf);changed=true;
                        }
                    } else if(v[1]>0 && v[2]>0) {
                        const long double hi=(long double)v[1]*(std::log((long double)v[2])-std::log((long double)v[1]));
                        if(v[0]>=hi && v[0]-hi<1e-13*vr) {
                            v[0]=std::nextafter(double(hi),-inf);changed=true;
                        }
                    }
                }
            }
        }
        return changed;
    }

public:
    Interior(const Problem& o,int variant,double until,const Gate& gate):orig(o),n(o.n()),m(o.m()),mode(variant),
        stop(until),verbose(std::getenv("CONIC_TRACE")!=nullptr),lifted_check(gate),
        scaled(o,mode==3?0:mode==28?18:8,mode),p(scaled.p),
        sys(p,mode==4,mode==6,mode==7,mode!=8,mode==29),verify(orig),x(n),s(m),z(m),ax(m),px(n),atz(n),rd(n),rp(m),
        gs(m),kxd(n),kzp(m),kxq(n),kzb(m),r1(n),r2(m),v(n),pv(n),qneg(n),aff(n,m),dir(n,m) {
        for(int i=0;i<n;++i) qneg[i]=-p.q[i];
    }

    bool ready() const { return sys.ok; }

    bool run(Result& r) {
        if(!sys.ok) return false;
        if(verbose) std::fprintf(stderr,"ipm mode=%d dim=%d blocks=%zu work=%.0f psd=%d\n",
            mode,sys.dim,sys.blocks.size(),sys.ls->work_est,sys.maxpsd);
        set_initial();
        const bool has_exp=std::any_of(p.cones.begin(),p.cones.end(),[](auto c){return c.kind==Exp;});
        if(mode>=5 && mode<=7) improve_initial();
        if(mode==28 && n+m<110000)improve_initial();
        if(mode==29)improve_initial();
        if(mode==16) {improve_initial();fixed_tau=true;}
        bool secant=mode!=1;
        const bool two_secant=(mode==17 || mode==18 || mode==19);
        int slow=0;
        int recoveries=0;
        auto recover=[&]() {
            if(recoveries>=2 || best>=.002 || bx.empty() || clock_time()+.02>stop)return false;
            x=bx;s=bs;z=bz0;tau=1.;
            kappa=std::max(1e-15,dot(s,z)/std::max(1,sys.degree-1));
            fixed_tau=true;secant=has_exp?false:secant;
            ++recoveries;slow=0;oldstep=.9;
            if(verbose)std::fprintf(stderr,"retrying from best iterate\n");
            return true;
        };
        const int max_iter=mode==0?105:140;
        for(iter=0;iter<max_iter;++iter) {
            residuals();
            if((mode==15 || mode==28) && !fixed_tau && tau>1e-6 && kappa<.25*tau &&
                std::max({pr,dr,gp})<.04) {
                fixed_tau=true;mu=dot(s,z)/std::max(1,sys.degree-1);
                if(verbose)std::fprintf(stderr,"fixed-scale continuation\n");
            }
            if(verbose) std::fprintf(stderr,"%3d %.3g %.3g %.3g tau=%.4g k=%.4g mu=%.3g step=%.3g\n",
                iter,pr,dr,gp,tau,kappa,mu,oldstep);
            if(save_if_valid(r)) return true;
            if(!(mu>0 && tau>0 && kappa>0) || !std::isfinite(gap) || clock_time()>stop) break;
            if(recoveries==0 && iter>12 && best<2e-5 &&
               std::max({pr,dr,gp})>100*best && recover())continue;
            bool extended=(recoveries || mode==5 || mode==6 || (iter>7 && (pr<1e-5 || dr<1e-5)));
            double reg=recoveries==2 ? 2e-8 : (recoveries || mode==2 || mode==15 || mode==16) ? 3e-10 :
                (extended && !sys.ls->dense ? 1e-15 : 1e-12);
            if(!sys.update(s,z,mu,secant,reg,extended,mu<1e-4 || mode==9,two_secant)) {
                if(nudge() && iter+1<max_iter)continue;
                if(mode!=11) {
                    if(save_if_valid(r,true))return true;
                    if(recover())continue;
                }
                if(verbose) std::fprintf(stderr,"stalled in scaling or factor\n");
                break;
            }
            if(!rhs_constant() || !direction(aff,0.,0.)) {
                if(mode!=11 && recover())continue;
                if(verbose) std::fprintf(stderr,"nonfinite Newton system\n");
                break;
            }
            double aa=step(aff,false);
            double sigma=std::pow(1-aa,3), damp=iter==0?aa:1.;
            bool okay=direction(dir,sigma,damp);
            double a=okay?step(dir,true):0.;
            if(a<.3*aa || a<.005) {
                okay=direction(dir,std::max(.01,sigma),aa*.5);
                double b=okay?step(dir,true):0.;
                if(b>=a || a<.02) a=b;
                if(b<a) {
                    okay=direction(dir,sigma,damp);
                    a=okay?step(dir,true):0.;
                }
            }
            if(a<.1*aa && aa>1e-5) {
                a=aa*.9;
                take(aff,a);
            } else if(a>1e-9) {
                take(dir,a);
            } else {
                if(secant && has_exp) {secant=false;continue;}
                if(mode!=11 && recover())continue;
                break;
            }
            if(a<.003) {
                ++slow;
                if(has_exp && secant && slow>=2) {secant=false;slow=0;}
                if(slow>=8) {
                    if(mode!=11 && recover())continue;
                    break;
                }
            } else slow=0;
            oldstep=a;
        }
        residuals();
        return save_if_valid(r,true);
    }
};

inline bool trivial(const Problem& p,Result& r) {
    const int n=p.n(),m=p.m();
    std::fill(r.x.begin(),r.x.end(),0.);
    std::fill(r.z.begin(),r.z.end(),0.);
    if(m) std::copy(p.b,p.b+m,r.s.begin());
    if(norm(p.q,n)<=9.e-9 && cone_defect(p,r.s,false,9.75e-9)<=9.75e-9) {
        r.status=Solved;return true;
    }
    const double qmax=norm(p.q,n);
    if(qmax>0) {
        int col=-1;double best=qmax*eps;
        for(int j=0;j<n;++j)
            if(p.A.p[j]==p.A.p[j+1] && p.P.p[j]==p.P.p[j+1] && std::abs(p.q[j])>best) {
                best=std::abs(p.q[j]);col=j;
            }
        if(col>=0) {
            r.x[col]=p.q[col]>0?-1.:1.;
            std::fill(r.s.begin(),r.s.end(),0.);
            r.status=DualInfeasible;return true;
        }
    }
    return false;
}

inline bool separable(const Problem& p,Result& r) {
    const int n=p.n(),m=p.m();
    if (p.A.nz()>m || p.P.nz()>n ||
        std::any_of(p.cones.begin(),p.cones.end(),[](auto c){return c.kind>Linear;})) return false;
    for(int j=0;j<n;++j)
        if(p.P.p[j+1]>p.P.p[j] && (p.P.p[j+1]-p.P.p[j]!=1 || p.P.i[p.P.p[j]]!=j))
            return false;
    Int col(m,-1);
    Vec coef(m,0.);
    for(int j=0;j<n;++j) for(int k=p.A.p[j];k<p.A.p[j+1];++k) {
        int i=p.A.i[k];
        if(col[i]>=0) return false;
        col[i]=j;coef[i]=p.A.v[k];
    }
    Int low(n,-1), high(n,-1);
    Vec lo(n,-inf), hi(n,inf);
    int inconsistent=-1;
    double const_bad=0;
    for(auto c:p.cones) {
        for(int i=c.off;i<c.off+c.width;++i) {
            const int j=col[i];
            const double a=coef[i], b=p.b[i];
            if(j<0) {
                double bad=c.kind==Zero?std::abs(b):-b;
                if(bad>const_bad){ const_bad=bad;inconsistent=i; }
                continue;
            }
            double v=b/a;
            if(!std::isfinite(v))return false;
            if(c.kind==Zero || a<0) {
                if(v>lo[j] || (v==lo[j] && (low[j]<0 || std::abs(a)>std::abs(coef[low[j]])))) {
                    lo[j]=v;low[j]=i;
                }
            }
            if(c.kind==Zero || a>0) {
                if(v<hi[j] || (v==hi[j] && (high[j]<0 || std::abs(a)>std::abs(coef[high[j]])))) {
                    hi[j]=v;high[j]=i;
                }
            }
        }
    }
    Check ck(p);
    std::fill(r.x.begin(),r.x.end(),0.);
    std::fill(r.z.begin(),r.z.end(),0.);
    if(inconsistent>=0 && const_bad>ck.bmax*eps) {
        r.z[inconsistent]=p.b[inconsistent]<0?1.:-1.;
        if(ck.primal(r))return true;
        r.z[inconsistent]=0.;
    }
    int conflict=-1;
    double margin=0.;
    for(int j=0;j<n;++j) if(lo[j]>hi[j]) {
        double a=coef[low[j]],b=coef[high[j]];
        double d=(lo[j]-hi[j])*std::min(std::abs(a),std::abs(b));
        if(d>margin) {margin=d;conflict=j;}
    }
    if(conflict>=0) {
        int i=low[conflict],j=high[conflict];
        double a=coef[i],b=coef[j],d=std::min(std::abs(a),std::abs(b));
        r.z[i]=-d/a;r.z[j]=d/b;
        if(ck.primal(r)) return true;
        r.z[i]=r.z[j]=0.;
    }
    bool ray=false;
    for(int j=0;j<n;++j) {
        if(p.P.p[j]!=p.P.p[j+1]) continue;
        if((p.q[j]>0 && low[j]<0) || (p.q[j]<0 && high[j]<0)) {
            r.x[j]=p.q[j]>0?-1.:1.; ray=true;
        }
    }
    if(ray && ck.dual(r)) return true;
    if(ray || conflict>=0) return false;
    for(int j=0;j<n;++j) {
        double q=p.q[j], h=p.P.p[j]==p.P.p[j+1]?0.:p.P.v[p.P.p[j]];
        if(h<0)return false;
        double x=h>0 ? -q/h : q>0 ? lo[j] : q<0 ? hi[j] : 0.;
        r.x[j]=x=std::clamp(x,lo[j],hi[j]);
        if(!std::isfinite(x)) return false;
        double grad=std::fma(h,x,q);
        int row=grad>0 && x==lo[j]?low[j]:grad<0 && x==hi[j]?high[j]:-1;
        if(row>=0) r.z[row]=-grad/coef[row];
    }
    for(int i=0;i<m;++i) r.s[i]=p.b[i]-(col[i]>=0?coef[i]*r.x[col[i]]:0.);
    for(auto c:p.cones) for(int i=c.off;i<c.off+c.width;++i) {
        if(c.kind==Zero) r.s[i]=0.;
        else r.s[i]=std::max(0.,r.s[i]);
    }
    return ck.optimal(r);
}

bool solve_model(const Problem& p,Result& r,int mode,double stop,bool presolve=true,bool chordal=true,
                 bool substitute=true,bool epigraph=true,const Gate& gate={}) {
    auto lifted=[&](const Result& x) {return !gate || gate(x);};
    auto check_lifted=[&](Check& ck,Result& x) {
        bool ok=x.status==Solved?ck.optimal(x):x.status==PrimalInfeasible?ck.primal(x):ck.dual(x);
        return ok && lifted(x);
    };
    if((trivial(p,r) || separable(p,r)) && lifted(r))return true;
    r.status=Unresolved;
    if(mode<0)return false;
    if(mode==27) {
        Check ck(p);
        return outer_native(p,r,stop,[&](int status) {
            r.status=status;return check_lifted(ck,r);
        });
    }
    if(mode==22 || mode==23) {
        Check ck(p);
        if(mode==22) {
            Box b(p,true);
            return b.diagonal_balance(stop,[&](const Vec& x,const Vec& s,const Vec& z) {
                r.x=x;r.s=s;r.z=z;return ck.optimal(r) && lifted(r);
            },[&](const Vec& z) {r.z=z;return ck.primal(r) && lifted(r);});
        }
        Scaled sc(p,6);
        Box b(sc.p,false);
        return projected_qp(sc.p,b,stop,[&](const Vec& x,const Vec& s,const Vec& z) {
            for(int j=0;j<p.n();++j)r.x[j]=x[j]*sc.d[j];
            for(int j=0;j<p.m();++j) {
                r.s[j]=s[j]/sc.e[j];r.z[j]=z[j]*sc.e[j]/sc.c;
            }
            return ck.optimal(r) && lifted(r);
        });
    }
    if(presolve && mode!=11) {
        Reduction reduced(p);
        bool changed=reduced.build([&](Result& ray) {
            Check ck(p);return ck.primal(ray) && lifted(ray);
        });
        if(reduced.has_cert) {
            r.z=std::move(reduced.cert.z);r.status=PrimalInfeasible;return true;
        }
        if(changed && clock_time()<stop) {
            if(std::getenv("CONIC_TRACE"))
                std::fprintf(stderr,"presolve %d,%d -> %d,%d\n",p.n(),p.m(),reduced.p.n(),reduced.p.m());
            Result inner{Vec(reduced.p.n()),Vec(reduced.p.m()),Vec(reduced.p.m()),Unresolved};
            Check ck(p);
            auto accept=[&](const Result& u){reduced.recover(u,r);return check_lifted(ck,r);};
            if(solve_model(reduced.p,inner,mode,stop,false,chordal,substitute,epigraph,accept))return true;
        }
    }
    if(substitute && mode!=11 && mode!=4 && clock_time()<stop) {
        Substitution s(p);
        if(s.build(stop)) {
            if(std::getenv("CONIC_TRACE"))
                std::fprintf(stderr,"pair presolve %d,%d -> %d,%d\n",p.n(),p.m(),s.p.n(),s.p.m());
            Result inner{Vec(s.p.n()),Vec(s.p.m()),Vec(s.p.m()),Unresolved};
            Check ck(p);
            auto accept=[&](const Result& u){s.recover(u,r);return check_lifted(ck,r);};
            if(solve_model(s.p,inner,mode,stop,true,chordal,false,epigraph,accept))return true;
        }
    }
    if(epigraph && mode!=11 && mode!=4 && clock_time()<stop) {
        Epigraph qp(p);
        if(qp.build(stop)) {
            if(std::getenv("CONIC_TRACE"))
                std::fprintf(stderr,"epigraph %d,%d -> %d,%d\n",p.n(),p.m(),qp.p.n(),qp.p.m());
            Result inner{Vec(qp.p.n()),Vec(qp.p.m()),Vec(qp.p.m()),Unresolved};
            int submode=qp.p.m()==0?14:mode;
            Check ck(p);
            auto accept=[&](const Result& u){return qp.recover(u,r) && check_lifted(ck,r);};
            if(solve_model(qp.p,inner,submode,stop,true,chordal,substitute,false,accept))return true;
        }
    }
    if(chordal && mode!=4 && mode!=11 && clock_time()<stop) {
        Chordal conv(p);
        if(conv.build(mode==12)) {
            if(std::getenv("CONIC_TRACE"))
                std::fprintf(stderr,"chordal %d,%d -> %d,%d\n",p.n(),p.m(),conv.p.n(),conv.p.m());
            Result r2{Vec(conv.p.n()),Vec(conv.p.m()),Vec(conv.p.m()),Unresolved};
            Check ck(p);
            auto accept=[&](const Result& u) {
                for(double diag:{0.,2e-12,1e-10}) {
                    if(!conv.recover(u,r,diag))break;
                    if(check_lifted(ck,r))return true;
                    if(r.status==DualInfeasible)break;
                }
                return false;
            };
            if(solve_model(conv.p,r2,mode,stop,false,false,false,false,accept))return true;
        }
    }
    if(clock_time()>=stop)return false;
    if(mode==13 || mode==14 || mode==21 || (mode>=24&&mode<=26)) {
        if(mode<24 && !is_polyhedral(p))return false;
        Scaled scaled(p,6);
        Check ck(p);
        auto opt=[&](const Vec& xx,const Vec& ss,const Vec& zz) {
            for(int j=0;j<p.n();++j)r.x[j]=xx[j]*scaled.d[j];
            for(int j=0;j<p.m();++j) {
                r.s[j]=ss[j]/scaled.e[j];r.z[j]=zz[j]*scaled.e[j]/scaled.c;
            }
            return ck.optimal(r) && lifted(r);
        };
        auto ray=[&](const Vec& xx,const Vec& zz) {
            for(int j=0;j<p.n();++j)r.x[j]=xx[j]*scaled.d[j];
            for(int j=0;j<p.m();++j)r.z[j]=zz[j]*scaled.e[j]/scaled.c;
            return (ck.primal(r) && lifted(r)) || (ck.dual(r) && lifted(r));
        };
        if(mode>=24)return solve_dr(scaled.p,stop,opt,ray,mode-24);
        if(mode==14)return poly_polish(scaled.p,Vec(p.n()),Vec(p.m()),stop,opt,5);
        return poly_admm(scaled.p,stop,opt,ray,mode==21);
    }
    {
        Interior solver(p,mode,stop,gate);
        if(solver.ready())return solver.run(r);
    }
    if(clock_time()+.035>=stop || mode==11)return false;
    return solve_model(p,r,is_polyhedral(p)?21:25,stop,false,false,false,false,gate);
}

} // namespace conic

extern "C" int conic_solve(int n,int m,const int* pp,const int* pi,const double* pv,
        const int* ap,const int* ai,const double* av,const double* q,const double* b,
        int nc,const int* kinds,const int* dims,double time_budget,int mode,
        double* ox,double* os,double* oz) {
    using namespace conic;
    try {
        const double stop=clock_time()+time_budget;
        Problem p{{n,n,pp,pi,pv},{m,n,ap,ai,av},q,b,{}};
        p.cones.reserve(nc);
        int off=0;
        for(int k=0;k<nc;++k) {
            int d=dims[k],w=kinds[k]==PSD?int(int64_t(d)*(d+1)/2):d;
            p.cones.push_back({kinds[k],d,off,w});off+=w;
        }
        if(off!=m)return Unresolved;
        Result r{Vec(n),Vec(m),Vec(m),Unresolved};
        bool success=solve_model(p,r,mode,stop);
        if(success) {
            if(r.status!=PrimalInfeasible && n) std::copy(r.x.begin(),r.x.end(),ox);
            if(r.status!=PrimalInfeasible && m) std::copy(r.s.begin(),r.s.end(),os);
            if(r.status!=DualInfeasible && m) std::copy(r.z.begin(),r.z.end(),oz);
        }
        return success?r.status:Unresolved;
    } catch (const std::exception& e) {
        if (std::getenv("CONIC_TRACE")) std::fprintf(stderr,"conic: %s\n",e.what());
        return Unresolved;
    }
}

extern "C" void conic_set_lu(conic::LUCallbacks::Make make,conic::LUCallbacks::Solve solve,
                             conic::LUCallbacks::Free release) {
    conic::external_lu={make,solve,release};
}

extern "C" int conic_check(int n,int m,const int* pp,const int* pi,const double* pv,
        const int* ap,const int* ai,const double* av,const double* q,const double* b,
        int nc,const int* kinds,const int* dims,int status,double* ox,double* os,double* oz) {
    using namespace conic;
    try {
        const bool reconstruct=status&16;
        status&=15;
        Problem p{{n,n,pp,pi,pv},{m,n,ap,ai,av},q,b,{}};
        int off=0;
        for(int k=0;k<nc;++k) {
            int d=dims[k],w=kinds[k]==PSD?int(int64_t(d)*(d+1)/2):d;
            p.cones.push_back({kinds[k],d,off,w});off+=w;
        }
        if(off!=m)return Unresolved;
        Result r{Vec(n),Vec(m),Vec(m),Unresolved};
        if(status!=PrimalInfeasible && n)std::copy(ox,ox+n,r.x.begin());
        if(status==Solved && reconstruct && m) {
            std::copy(b,b+m,r.s.begin());
            p.A.add(r.x.data(),r.s.data(),-1.);
            for(auto c:p.cones) {
                if(c.kind==Zero)std::fill(r.s.begin()+c.off,r.s.begin()+c.off+c.width,0.);
                else if(c.kind==Linear)for(int i=c.off;i<c.off+c.width;++i)
                    r.s[i]=std::max(0.,r.s[i]);
            }
        } else if(status!=PrimalInfeasible && m)std::copy(os,os+m,r.s.begin());
        if(status!=DualInfeasible && m)std::copy(oz,oz+m,r.z.begin());
        Check ck(p);
        bool ok=status==Solved?ck.optimal(r):status==PrimalInfeasible?ck.primal(r):ck.dual(r);
        if(ok) {
            if(status!=PrimalInfeasible && n)std::copy(r.x.begin(),r.x.end(),ox);
            if(status!=PrimalInfeasible && m)std::copy(r.s.begin(),r.s.end(),os);
            if(status!=DualInfeasible && m)std::copy(r.z.begin(),r.z.end(),oz);
        }
        return ok?status:Unresolved;
    } catch (...) {return Unresolved;}
}

extern "C" int conic_polish(int n,int m,const int* pp,const int* pi,const double* pv,
        const int* ap,const int* ai,const double* av,const double* q,const double* b,
        int nc,const int* kinds,const int* dims,double seconds,double* ox,double* os,double* oz) {
    using namespace conic;
    try {
        const double stop=clock_time()+seconds;
        Problem p{{n,n,pp,pi,pv},{m,n,ap,ai,av},q,b,{}};
        int off=0;
        for(int k=0;k<nc;++k) {
            int w=kinds[k]==PSD?int(int64_t(dims[k])*(dims[k]+1)/2):dims[k];
            p.cones.push_back({kinds[k],dims[k],off,w});off+=w;
        }
        if(off!=m || n+m>260000 || !is_polyhedral(p))return Unresolved;
        Result r{Vec(n),Vec(m),Vec(m),Unresolved};
        Scaled sc(p,6);
        for(int j=0;j<n;++j)r.x[j]=ox[j]/sc.d[j];
        for(int j=0;j<m;++j)r.z[j]=oz[j]*sc.c/sc.e[j];
        if(!finite(r.x)||!finite(r.z)||clock_time()>stop)return Unresolved;
        Check ck(p);
        bool ok=poly_polish(sc.p,r.x,r.z,stop,[&](const Vec& x,const Vec& s,const Vec& z) {
            for(int j=0;j<n;++j)r.x[j]=x[j]*sc.d[j];
            for(int j=0;j<m;++j) {
                r.s[j]=s[j]/sc.e[j];r.z[j]=z[j]*(sc.e[j]/sc.c);
            }
            return ck.optimal(r);
        },4);
        if(!ok)return Unresolved;
        if(n)std::copy(r.x.begin(),r.x.end(),ox);
        if(m) {
            std::copy(r.s.begin(),r.s.end(),os);
            std::copy(r.z.begin(),r.z.end(),oz);
        }
        return Solved;
    } catch (...) {return Unresolved;}
}

extern "C" int sealed_highs_solve(int,int,const int*,const int*,const double*,const int*,const int*,
    const double*,const double*,const double*,const double*,const double*,const double*,int,double,
    double*,double*,double*,int*);

extern "C" int conic_highs(int n,int m,const int* pp,const int* pi,const double* pv,
        const int* ap,const int* ai,const double* av,const double* q,const double* b,
        int nc,const int* kinds,const int* dims,double time_budget,int engine,
        double* ox,double* os,double* oz) {
    using namespace conic;
    try {
        const double start=clock_time();
        Problem p{{n,n,pp,pi,pv},{m,n,ap,ai,av},q,b,{}};
        int m2=0;
        for(int k=0;k<nc;++k) {
            if(kinds[k]>Linear)return Unresolved;
            p.cones.push_back({kinds[k],dims[k],m2,dims[k]});m2+=dims[k];
        }
        if(m2!=m || n==0)return Unresolved;
        Vec lower(n,-inf),upper(n,inf), rl(m,-inf);
        for(auto c:p.cones)if(c.kind==Zero)
            std::copy(b+c.off,b+c.off+c.width,rl.begin()+c.off);
        // Compact a substantial group of singleton rows into column bounds.
        // Store the source of each strongest bound to recover row multipliers.
        Int ap2,ai2,map,lrow,urow;
        Vec av2,rl2,b2,coeff;
        bool shrunk=false;
        if(m>=160 && !(engine&4) && p.A.nz()>=110) {
            Int count(m,0);
            for(int k=0;k<p.A.nz();++k)++count[ai[k]];
            int single=0;
            for(int i=0;i<m;++i)single+=count[i]==1;
            if(single>=std::max(50,m/12)) {
                lrow.assign(n,-1);urow.assign(n,-1);map.assign(m,-1);
                coeff.resize(m);
                for(int j=0;j<n;++j)for(int k=ap[j];k<ap[j+1];++k) {
                    int i=ai[k];double a=av[k];
                    if(count[i]!=1 || std::abs(a)<1e-12)continue;
                    double d=b[i]/a;
                    if(!(std::abs(d)<1e24))continue;
                    map[i]=-2;coeff[i]=a;
                    if((a<0. || rl[i]!=-inf) && (d>lower[j] || (d==lower[j] && std::abs(a)>std::abs(coeff[lrow[j]])))) {
                        lower[j]=d;lrow[j]=i;
                    }
                    if((a>0. || rl[i]!=-inf) && (d<upper[j] || (d==upper[j] && std::abs(a)>std::abs(coeff[urow[j]])))) {
                        upper[j]=d;urow[j]=i;
                    }
                }
                bool good=true;
                for(int j=0;j<n;++j)if(lower[j]>upper[j]){good=false;break;}
                if(good) {
                    for(int i=0;i<m;++i)if(map[i]!=-2) {
                        map[i]=int(b2.size());b2.push_back(b[i]);rl2.push_back(rl[i]);
                    }
                    ap2.resize(n+1);ai2.reserve(p.A.nz());av2.reserve(p.A.nz());
                    for(int j=0;j<n;++j) {
                        for(int k=ap[j];k<ap[j+1];++k)if(map[ai[k]]>=0) {
                            ai2.push_back(map[ai[k]]);av2.push_back(av[k]);
                        }
                        ap2[j+1]=int(ai2.size());
                    }
                    shrunk=int(b2.size())<m;
                } else {
                    std::fill(lower.begin(),lower.end(),-inf);
                    std::fill(upper.begin(),upper.end(),inf);
                }
            }
        }
        Int hp,hi;
        Vec hv;
        if(pp[n]>0) {
            hp.resize(n+1);
            hi.reserve(pp[n]/2+n);hv.reserve(pp[n]/2+n);
            for(int j=0;j<n;++j) {
                for(int a=pp[j];a<pp[j+1];++a) if(pi[a]>=j) {
                    hi.push_back(pi[a]);hv.push_back(pv[a]);
                }
                hp[j+1]=int(hi.size());
            }
        }
        Result r{Vec(n),Vec(m),Vec(m),Unresolved};
        int flags=0;
        Vec zsmall(shrunk?b2.size():0),cd(shrunk?n:0);
        const int code=sealed_highs_solve(n,shrunk?int(b2.size()):m,
            hp.empty()?pp:hp.data(),hp.empty()?pi:hi.data(), hp.empty()?pv:hv.data(),
            shrunk?ap2.data():ap,shrunk?ai2.data():ai,shrunk?av2.data():av,
            q,lower.data(),upper.data(),shrunk?rl2.data():rl.data(),shrunk?b2.data():b,engine&1,
            time_budget,r.x.data(),shrunk?zsmall.data():r.z.data(),shrunk?cd.data():nullptr,&flags);
        if(std::getenv("CONIC_TRACE"))
            std::fprintf(stderr,"native HiGHS %d/%d time %.3f\n",code,flags,clock_time()-start);
        if(!flags)return Unresolved;
        if(shrunk && (flags&10)) {
            for(int j=0;j<m;++j)if(map[j]>=0)r.z[j]=zsmall[map[j]];
            if((flags&8)||!(flags&16)) {
                std::fill(cd.begin(),cd.end(),0.);
                p.A.add_transpose(r.z.data(),cd.data());
                if(!(flags&8)) {
                    p.P.add(r.x.data(),cd.data());
                    for(int j=0;j<n;++j)cd[j]+=q[j];
                }
            }
            for(int j=0;j<n;++j) {
                const int row=cd[j]>0?lrow[j]:urow[j];
                if(row>=0)r.z[row]=-cd[j]/coeff[row];
            }
        }
        Check ck(p);
        if(flags&8) {
            if(!ck.primal(r)) {
                for(double& v:r.z)v=-v;
                ck.primal(r);
            }
        } else if(flags&4) {
            if(!ck.dual(r)) {
                for(double& v:r.x)v=-v;
                ck.dual(r);
            }
        } else if((flags&3)==3) {
            if(m)std::copy(b,b+m,r.s.begin());
            p.A.add(r.x.data(),r.s.data(),-1.);
            for(auto c:p.cones)for(int j=c.off;j<c.off+c.width;++j)
                r.s[j]=c.kind==Zero?0.:std::max(0.,r.s[j]);
            ck.optimal(r);
            if(r.status==Unresolved && code==7 && clock_time()-start<.65 &&
               n+m<150000 && time_budget>.05) {
                if(n)std::copy(r.x.begin(),r.x.end(),ox);
                if(m)std::copy(r.z.begin(),r.z.end(),oz);
                return conic_polish(n,m,pp,pi,pv,ap,ai,av,q,b,nc,kinds,dims,.12,ox,os,oz);
            }
        }
        if(r.status==Unresolved)return Unresolved;
        if(r.status!=PrimalInfeasible) {
            if(n)std::copy(r.x.begin(),r.x.end(),ox);
            if(m)std::copy(r.s.begin(),r.s.end(),os);
        }
        if(r.status!=DualInfeasible && m)std::copy(r.z.begin(),r.z.end(),oz);
        return r.status;
    } catch (...) {return Unresolved;}
}
