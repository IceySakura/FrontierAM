#pragma once
#include "blas.hpp"

namespace conic {

struct Epigraph {
    const Problem& o;
    Problem p;
    CSC A;
    int pivot=-1, side=-1, sign=1, which=-1;
    ConeInfo cone{};
    double a0=0.,q0=0.,gamma=0.,ell=0.;
    Int rowmap,ap,ai,pp,pi;
    Vec av,pv,q,b,first;
    explicit Epigraph(const Problem& prob):o(prob),p(o),A(o.A) {}

    bool find() {
        if(o.n()<2 || o.n()>200000 || o.A.nz()>3000000)return false;
        Int block(o.m(),-1);
        int nsoc=0;
        for(size_t k=0;k<o.cones.size();++k) if(o.cones[k].kind==Soc) {
            auto c=o.cones[k];block[c.off]=int(k);++nsoc;
        }
        if(!nsoc)return false;
        const int n=o.n();
        for(int j=0;j<n;++j) {
            if(o.q[j]==0. || o.P.p[j]!=o.P.p[j+1])continue;
            const int i=A.p[j],nz=A.p[j+1]-i;
            if(nz<1 || nz>2 || block[A.i[i]]<0)continue;
            auto c=o.cones[block[A.i[i]]];
            double g=-o.q[j]/A.v[i];
            if(!(g>1e-15 && g<1e15))continue;
            int s=-1,sgn=1;
            double el=0.;
            if(nz==1) {
                if(o.P.nz() || nsoc>1)continue;
                int ct=0;
                for(int k=0;k<n;++k)ct+=o.q[k]!=0.;
                if(ct!=1)continue;
            } else {
                s=A.i[i+1];
                if(s<=c.off || s>=c.off+c.width || std::abs(A.v[i+1])!=std::abs(A.v[i]))continue;
                sgn=A.v[i]*A.v[i+1]>0.?1:-1;
                el=o.b[c.off]-sgn*o.b[s];
                if(!(el>1e-12 && el<1e15) || g/el>1e15)continue;
            }
            first.assign(n,0.);
            bool ok=true;
            for(int k=0;k<n;++k)if(k!=j) {
                double t=0.;
                for(int a=A.p[k];a<A.p[k+1];++a) {
                    if(A.i[a]==c.off)first[k]=A.v[a];
                    if(A.i[a]==s)t=A.v[a];
                }
                if(first[k]!=sgn*t) {ok=false;break;}
            }
            if(ok) {
                pivot=j;cone=c;side=s;sign=sgn;ell=el;a0=A.v[i];q0=o.q[j];gamma=g;which=block[c.off];
                return true;
            }
        }
        return false;
    }

    bool build(double stop) {
        if(std::none_of(o.cones.begin(),o.cones.end(),[](auto c){return c.kind==Soc;}) || !find())return false;
        const int n=o.n(),r=n-1,m=o.m(),w=cone.width,p0=cone.off;
        Int count(w,0);
        for(int j=0;j<n;++j)if(j!=pivot)for(int k=A.p[j];k<A.p[j+1];++k) {
            int i=A.i[k];
            if(i>p0 && i<p0+w && i!=side)++count[i-p0];
        }
        double flop=0;
        for(int a:count)flop+=double(a)*a;
        if(flop>6500000 || clock_time()>stop)return false;
        Int cp(w+1,0);
        for(int i=0;i<w;++i)cp[i+1]=cp[i]+count[i];
        Int ci(cp.back()),pos=cp;
        Vec cv(cp.back());
        const double g=side<0?1.:gamma/ell;
        q.reserve(r);
        for(int j=0;j<n;++j)if(j!=pivot) {
            const int v=j-(j>pivot);
            double c=o.q[j]+gamma*first[j];
            for(int k=A.p[j];k<A.p[j+1];++k) {
                int i=A.i[k];
                if(i>p0 && i<p0+w && i!=side) {
                    int l=pos[i-p0]++;
                    ci[l]=v;cv[l]=A.v[k];
                    c-=g*A.v[k]*o.b[i];
                }
            }
            q.push_back(c);
        }
        std::vector<std::vector<std::pair<int,double>>> columns(r);
        for(int j=0;j<n;++j)if(j!=pivot)for(int k=o.P.p[j];k<o.P.p[j+1];++k) {
            int i=o.P.i[k];if(i==pivot)return false;
            columns[j-(j>pivot)].emplace_back(i-(i>pivot),o.P.v[k]);
        }
        for(int i=0;i<w;++i)for(int a=cp[i];a<cp[i+1];++a)
            for(int b=cp[i];b<cp[i+1];++b)
                columns[ci[a]].emplace_back(ci[b],g*cv[a]*cv[b]);
        pp.reserve(n);pp.push_back(0);
        for(auto& col:columns) {
            std::sort(col.begin(),col.end(),[](auto a,auto b){return a.first<b.first;});
            for(size_t i=0;i<col.size();) {
                int j=col[i].first;
                double v=col[i++].second;
                while(i<col.size() && col[i].first==j)v+=col[i++].second;
                if(v!=0.) {pi.push_back(j);pv.push_back(v);}
            }
            pp.push_back(int(pi.size()));
        }
        p.cones.clear();rowmap.assign(m,-1);
        for(size_t j=0;j<o.cones.size();++j)if(int(j)!=which) {
            auto c=o.cones[j];
            const int off=int(b.size());
            p.cones.push_back({c.kind,c.dim,off,c.width});
            for(int i=c.off;i<c.off+c.width;++i) {rowmap[i]=int(b.size());b.push_back(o.b[i]);}
        }
        ap.push_back(0);
        for(int j=0;j<n;++j)if(j!=pivot) {
            for(int k=A.p[j];k<A.p[j+1];++k) {
                int i=rowmap[A.i[k]];
                if(i>=0) {ai.push_back(i);av.push_back(A.v[k]);}
            }
            ap.push_back(int(av.size()));
        }
        p.A={int(b.size()),r,ap.data(),ai.data(),av.data()};p.P={r,r,pp.data(),pi.data(),pv.data()};
        p.q=q.data();p.b=b.data();
        return true;
    }

    bool recover(const Result& sub,Result& out) const {
        out.status=sub.status;
        const int n=o.n(),m=o.m();
        if(sub.status==PrimalInfeasible) {
            for(int i=0;i<m;++i)out.z[i]=rowmap[i]>=0?sub.z[rowmap[i]]:0.;
            return true;
        }
        for(int j=0;j<n;++j)out.x[j]=j==pivot?0.:sub.x[j-(j>pivot)];
        if(sub.status==DualInfeasible) {
            if(side<0)return false;
            out.x[pivot]=-dot(out.x,first)/a0;
            return true;
        }
        Vec slack(o.b,o.b+m);
        o.A.add(out.x.data(),slack.data(),-1.);
        double radius=0.;
        for(int i=cone.off+1;i<cone.off+cone.width;++i)
            if(i!=side)radius=std::hypot(radius,slack[i]);
        double t=0.,multiplier=1., zt=gamma,zr=0.,ztail=0.;
        if(side<0) {
            t=(slack[cone.off]-radius)/a0;
            if(radius<1e-13) {
                multiplier=0.;ztail=0.;
            } else {
                multiplier=gamma/radius;ztail=-multiplier;
            }
        } else {
            const double rho=radius/ell, p=rho*radius;
            t=(slack[cone.off]+sign*slack[side]-p)/(2.*a0);
            zt=.5*gamma*(1+rho*rho);
            zr=sign*.5*gamma*(1-rho*rho);
            ztail=-gamma/ell;
        }
        out.x[pivot]=t;
        for(int i=0;i<m;++i) {
            if(rowmap[i]>=0) {
                out.z[i]=sub.z[rowmap[i]]*multiplier;out.s[i]=sub.s[rowmap[i]];
            } else {
                out.z[i]=i==cone.off?zt:i==side?zr:ztail*slack[i];
                out.s[i]=slack[i];
                if(i==cone.off)out.s[i]-=a0*t;
                if(i==side)out.s[i]-=sign*a0*t;
            }
        }
        return true;
    }
};

} // namespace conic
