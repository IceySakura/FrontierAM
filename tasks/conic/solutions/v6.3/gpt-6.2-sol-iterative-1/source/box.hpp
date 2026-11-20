#pragma once
#include "poly.hpp"

namespace conic {

struct Box {
    const Problem& p;
    Vec lo,hi,coeff, coupling;
    Int lowrow,highrow, col,eq;
    int aggregate=-1;
    bool good=false;
    explicit Box(const Problem& problem,bool allow_aggregate):p(problem) {
        if(!is_polyhedral(p) || p.A.nz()>p.m()+(allow_aggregate?p.n():0))return;
        const int n=p.n(),m=p.m();
        Int count(m,0);
        for(int k=0;k<p.A.nz();++k)++count[p.A.i[k]];
        for(int i=0;i<m;++i)if(count[i]>1) {
            if(!allow_aggregate||aggregate>=0)return;
            aggregate=i;
        }
        lo.assign(n,-inf);hi.assign(n,inf);
        lowrow.assign(n,-1);highrow.assign(n,-1);
        col.assign(m,-1);coeff.resize(m);eq.resize(m);
        if(aggregate>=0)coupling.resize(n);
        for(int j=0;j<n;++j)for(int k=p.A.p[j];k<p.A.p[j+1];++k) {
            const int i=p.A.i[k];
            if(i==aggregate)coupling[j]+=p.A.v[k];
            else {col[i]=j;coeff[i]=p.A.v[k];}
        }
        for(auto c:p.cones)for(int i=c.off;i<c.off+c.width;++i) {
            eq[i]=c.kind==Zero;
            if(i==aggregate)continue;
            const int j=col[i];
            if(j<0) {
                if(eq[i]?p.b[i]!=0.:p.b[i]<0.)return;
                continue;
            }
            const double a=coeff[i], v=p.b[i]/a;
            if(!std::isfinite(v))return;
            if(eq[i] || a<0.) if(v>lo[j] ||
                (v==lo[j] && (lowrow[j]<0 || std::abs(a)>std::abs(coeff[lowrow[j]])))) {
                lo[j]=v;lowrow[j]=i;
            }
            if(eq[i] || a>0.) if(v<hi[j] ||
                (v==hi[j] && (highrow[j]<0 || std::abs(a)>std::abs(coeff[highrow[j]])))) {
                hi[j]=v;highrow[j]=i;
            }
            if(lo[j]>hi[j])return;
        }
        good=true;
    }

    template<class Accept>
    bool lift(Vec& x,const Vec& g,double lambda,Accept&& accept) const {
        Vec s(p.m()),z(p.m(),0.);
        for(int j=0;j<p.n();++j) {
            int row=x[j]==lo[j] && g[j]>0.?lowrow[j]:x[j]==hi[j] && g[j]<0.?highrow[j]:-1;
            if(row>=0)z[row]=-g[j]/coeff[row];
        }
        for(int i=0;i<p.m();++i) {
            if(i==aggregate) {
                double slack=p.b[i]-dot(coupling,x);
                s[i]=eq[i]?0.:std::max(0.,slack);
                z[i]=lambda;
            } else s[i]=eq[i]?0.:std::max(0.,p.b[i]-(col[i]>=0?coeff[i]*x[col[i]]:0.));
        }
        return accept(x,s,z);
    }

    template<class Optimal,class PrimalRay>
    bool diagonal_balance(double until,Optimal&& opt,PrimalRay&& ray) {
        const int n=p.n(), row=aggregate;
        if(!good || row<0 || p.P.nz()>n)return false;
        Vec h(n,0.), x(n), g(n);
        for(int j=0;j<n;++j) {
            if(p.P.p[j]+1==p.P.p[j+1] && p.P.i[p.P.p[j]]==j)h[j]=p.P.v[p.P.p[j]];
            else if(p.P.p[j]!=p.P.p[j+1] || coupling[j]!=0.)return false;
            if(coupling[j]!=0. && !(h[j]>0))return false;
            if(h[j]==0. && ((p.q[j]>0. && lowrow[j]<0) || (p.q[j]<0. && highrow[j]<0)))return false;
        }
        auto extreme=[&](int orientation) {
            long double v=0.;
            for(int j=0;j<n;++j) if(coupling[j]!=0.) {
                double b=orientation*coupling[j]>0.?lo[j]:hi[j];
                if(!std::isfinite(b))return false;
                v+=(long double)b*coupling[j];
            }
            if(!(orientation*(v-p.b[row])>1e-12))return false;
            Vec z(p.m(),0.);
            z[row]=orientation;
            for(int j=0;j<n;++j) if(coupling[j]!=0.) {
                int at=orientation*coupling[j]>0.?lowrow[j]:highrow[j];
                z[at]=-orientation*coupling[j]/coeff[at];
            }
            return ray(z);
        };
        if(extreme(1) || (eq[row] && extreme(-1)))return true;
        auto residual=[&](double l) {
            long double f=-p.b[row],d=0.,sum=0.;
            for(int j=0;j<n;++j) {
                double aj=coupling[j];
                double g0=std::fma(l,aj,p.q[j]);
                double u=h[j]>0.?-g0/h[j]:g0>0.?lo[j]:g0<0.?hi[j]:0.;
                x[j]=std::clamp(u,lo[j],hi[j]);
                f+=(long double)x[j]*aj;sum+=std::abs(x[j]*aj);
                if(u==x[j]&&u>lo[j]&&u<hi[j]&&aj!=0.)d-=(long double)aj*aj/h[j];
                g[j]=std::fma(h[j],x[j],g0);
            }
            return std::array<double,3>{double(f),double(d),double(sum)};
        };
        auto goal=[&](std::array<double,3> e) {
            return std::abs(e[0]) <= 7e-14*(1.+std::abs(p.b[row]))+
                3e-17*e[2];
        };
        auto f=residual(0.);
        if(goal(f) || (!eq[row] && f[0]<0.))return lift(x,g,0.,opt);
        double l=0.,u=0.;
        double a=f[1]<0.?std::max(1e-18,std::abs(f[0]/f[1])):1.;
        if(f[0]<0.)a=-a;
        for(int t=0;t<64;++t) {
            if(clock_time()>until)return false;
            auto g0=residual(a);
            if(goal(g0))return lift(x,g,a,opt);
            if(a>0. && g0[0]<0.){u=a;break;}
            if(a<0. && g0[0]>0.){l=a;break;}
            a*=2.;
        }
        if(l==u)return false;
        double dual=.5*l+.5*u;
        for(int t=0;t<65 && clock_time()<until;++t) {
            auto g0=residual(dual);
            if(goal(g0) && lift(x,g,dual,opt))return true;
            if(g0[0]>0.)l=dual;
            else u=dual;
            double nw=g0[1]<0.?dual-g0[0]/g0[1]:inf;
            double next=(nw>l && nw<u)?nw:.5*l+.5*u;
            if(next==dual) return lift(x,g,dual,opt);
            dual=next;
        }
        return false;
    }
};

template<class Optimal>
bool projected_qp(const Problem& p,Box& bound,double until,Optimal&& accept) {
    const int n=p.n();
    if(!bound.good || bound.aggregate>=0 || !p.P.nz())return false;
    Vec x(n),g(n),di(n),step(n),proj(n),trial(n),d(n),pi(n),r(n),y(n),t(n),dx(n);
    std::vector<unsigned char> free(n,0);
    for(int j=0;j<n;++j) {
        double v=0.,mx=0.;
        for(int k=p.P.p[j];k<p.P.p[j+1];++k) {
            mx=std::max(mx,std::abs(p.P.v[k]));if(p.P.i[k]==j)v=p.P.v[k];
        }
        di[j]=1./std::max({v,.1*mx,1e-8});
        x[j]=std::clamp(-p.q[j]*di[j],bound.lo[j],bound.hi[j]);
    }
    const double qmax=norm(p.q,n);
    double last=inf;
    int stagnate=0;
    auto gain=[&](Vec& target) {
        for(int j=0;j<n;++j)d[j]=target[j]-x[j];
        std::fill(y.begin(),y.end(),0.);
        p.P.add(d.data(),y.data());
        const double gd=dot(g,d), curvature=dot(d,y);
        if(!(gd<0.))return std::pair<double,double>{0.,0.};
        double a=curvature>0. ? std::clamp(-gd/curvature,0.,1.):1.;
        return std::pair<double,double>{-a*(gd+.5*a*curvature),a};
    };
    for(int it=0;it<100 && clock_time()<until;++it) {
        if(n)std::copy(p.q,p.q+n,g.begin());
        p.P.add(x.data(),g.data());
        double merit=0.;
        for(int j=0;j<n;++j) {
            double b=std::clamp(x[j]-di[j]*g[j],bound.lo[j],bound.hi[j]);
            proj[j]=b;
            const bool active=(b==bound.lo[j] && g[j]>0.) || (b==bound.hi[j] && g[j]<0.) ||
                bound.lo[j]==bound.hi[j];
            free[j]=!active;
            const bool optimal_bound=(x[j]==bound.lo[j] && g[j]>0.) ||
                (x[j]==bound.hi[j] && g[j]<0.) || bound.lo[j]==bound.hi[j];
            if(!optimal_bound)merit=std::max(merit,std::abs(g[j]));
            step[j]=active?b-x[j]:0.;
        }
        if((merit<2e-7*(1.+qmax) || it==0) && bound.lift(x,g,0.,accept))return true;
        std::fill(y.begin(),y.end(),0.);
        p.P.add(step.data(),y.data());
        for(int j=0;j<n;++j) {
            r[j]=free[j]?-g[j]-y[j]:0.;
            pi[j]=r[j]*di[j];dx[j]=0.;
        }
        double rr=dot(r,pi);
        const double tol=std::max(3e-13*(1+qmax),std::min(.03,merit*.01)*norm(r));
        for(int k=0;k<65 && rr>0. && norm(r)>tol;++k) {
            if((k&7)==0 && clock_time()>until)break;
            std::fill(y.begin(),y.end(),0.);
            p.P.add(pi.data(),y.data());
            double py=dot(pi,y);
            if(!(py>1e-26*rr))break;
            const double a=rr/py;
            for(int j=0;j<n;++j) {
                dx[j]+=a*pi[j];
                if(free[j])r[j]-=a*y[j];
                t[j]=di[j]*r[j];
            }
            double rs=dot(r,t),beta=rs/rr;
            rr=rs;
            for(int j=0;j<n;++j)pi[j]=t[j]+beta*pi[j];
        }
        for(int j=0;j<n;++j)trial[j]=std::clamp(x[j]+step[j]+dx[j],bound.lo[j],bound.hi[j]);
        auto [g0,a0]=gain(proj);
        auto [g1,a1]=gain(trial);
        if(!(std::max(g0,g1)>0.))return bound.lift(x,g,0.,accept);
        const Vec& v=g1>=g0?trial:proj;
        double alpha=g1>=g0?a1:a0;
        if(alpha==1.)x=v;
        else for(int j=0;j<n;++j)x[j]+=alpha*(v[j]-x[j]);
        stagnate=merit>last*.995?stagnate+1:0;last=merit;
        if(stagnate>=12)break;
    }
    if(n)std::copy(p.q,p.q+n,g.begin());
    p.P.add(x.data(),g.data());
    return bound.lift(x,g,0.,accept);
}

} // namespace conic
