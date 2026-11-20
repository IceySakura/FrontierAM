#pragma once
#include "kkt.hpp"
#include "projection.hpp"

namespace conic {

struct ProjBlock {
    ConeInfo c;
    double rho=1.;
    Vec theta, pr, Q;
    explicit ProjBlock(ConeInfo cone):c(cone),theta(c.width,0.),pr(c.width,0.) {}
    void fwd(const double* a,double* b) const {
        if(Q.empty())std::copy(a,a+c.width,b);
        else for(int i=0;i<c.width;++i)b[i]=dot(Q.data()+size_t(i)*c.width,a,c.width);
    }
    void bwd(const double* a,double* b) const {
        if(Q.empty())std::copy(a,a+c.width,b);
        else {
            std::fill(b,b+c.width,0.);
            for(int i=0;i<c.width;++i)for(int j=0;j<c.width;++j)b[j]+=Q[size_t(i)*c.width+j]*a[i];
        }
    }
    bool evaluate(const double* w,const double* z,int which) {
        const int n=c.width;
        rho=std::clamp((1.+norm(w,n))/(1.+norm(z,n)),1e-5,1e5);
        rho*=which==1?.2:which==2?5.:1.;
        for(int i=0;i<n;++i)pr[i]=w[i]-rho*z[i];
        Q.clear();
        Vec u=pr;
        if(c.kind==Zero) {
            std::fill(pr.begin(),pr.end(),0.);std::fill(theta.begin(),theta.end(),0.);
        } else if(c.kind==Linear) {
            for(int j=0;j<n;++j) {
                pr[j]=std::max(0.,u[j]);
                theta[j]=u[j]>0?1.:0.;
            }
        } else if(c.kind==Soc) {
            const double tail=std::sqrt(dot(u.data()+1,u.data()+1,n-1));
            if(u[0]>=tail) {
                std::fill(theta.begin(),theta.end(),1.);
            } else if(u[0]<=-tail) {
                std::fill(pr.begin(),pr.end(),0.);std::fill(theta.begin(),theta.end(),0.);
            } else {
                if(n>48)return false;
                Q.assign(size_t(n)*n,0.);
                const double sg=u[1]>=0?1.:-1.;
                Vec h(n);
                for(int j=1;j<n;++j)h[j]=u[j]/tail;
                h[1]+=sg;
                const double beta=1./(1+std::abs(u[1]/tail));
                Q[0]=Q[n]=invroot2;
                for(int j=1;j<n;++j) {
                    Q[j]=u[j]*invroot2/tail;Q[n+j]=-Q[j];
                }
                for(int i=2;i<n;++i)for(int j=1;j<n;++j)
                    Q[size_t(i)*n+j]=(i==j)-beta*h[i]*h[j];
                theta[0]=1.;theta[1]=0.;
                for(int i=2;i<n;++i)theta[i]=.5+.5*u[0]/tail;
                std::fill(pr.begin(),pr.end(),0.);pr[0]=(u[0]+tail)*invroot2;
            }
        } else if(c.kind==PSD) {
            const int d=c.dim;
            if(d>11)return false;
            Vec a(size_t(d)*d),e(d);
            smat(d,u.data(),a.data());
            const int lw=1+6*d+2*d*d, liw=3+5*d;
            Vec work(lw);Int iw(liw);
            int info=0;
            scipy_dsyevd_("V","L",&d,a.data(),&d,e.data(),work.data(),&lw,iw.data(),&liw,&info);
            if(info)return false;
            Int ix,jx;
            for(int j=0;j<d;++j)for(int i=0;i<=j;++i){ix.push_back(i);jx.push_back(j);}
            Q.resize(size_t(n)*n);
            const double etol=1e-14*std::max(1e-15,norm(e));
            for(int k=0;k<n;++k) {
                const int i=ix[k],j=jx[k];
                for(int l=0;l<n;++l) {
                    const int r=ix[l],s=jx[l];
                    double v=a[r+i*d]*a[s+j*d];
                    if(i==j && r==s){}
                    else if(i==j || r==s)v*=root2;
                    else v+=a[s+i*d]*a[r+j*d];
                    Q[size_t(k)*n+l]=v;
                }
                const double ei=e[i],ej=e[j];
                if(std::abs(ej-ei)<=etol)theta[k]=ei>0.?1.:0.;
                else theta[k]=(std::max(0.,ej)-std::max(0.,ei))/(ej-ei);
                pr[k]=i==j?std::max(0.,ei):0.;
            }
        } else if(c.kind==Exp) {
            double initial=w[1]>0 && w[2]>0?std::log(w[2])-std::log(w[1]):inf;
            if(!std::isfinite(initial) && z[0]<0.)initial=1.-z[1]/z[0];
            ExpProjection prj;
            if(!project_exp(u.data(),prj,initial))return false;
            std::copy(prj.s.begin(),prj.s.end(),pr.begin());
            std::copy(prj.theta.begin(),prj.theta.end(),theta.begin());
            if(prj.rotated)Q.assign(prj.basis.begin(),prj.basis.end());
        } else return false;
        return finite(pr)&&finite(theta);
    }
};

template<class Accept>
bool semismooth(const Problem& p,Vec x,Vec z,double until,Accept&& accept,int strategy=0) {
    const int n=p.n(),m=p.m();
    if(n+m>80000 || p.A.nz()>1200000 || clock_time()>until ||
        std::any_of(p.cones.begin(),p.cones.end(),[](auto c) {
            return c.kind==PSD && c.dim>11;
        }))return false;
    std::vector<ProjBlock> blocks;
    Int block_id(m);
    for(auto c:p.cones) {
        std::fill(block_id.begin()+c.off,block_id.begin()+c.off+c.width,int(blocks.size()));
        blocks.emplace_back(c);
    }
    Int index(m,-1);
    Vec w(m),s(m),zproj(m),dz(m),feas(m),bd(m),bx(n),grad(n),r(m),loc;
    double best=inf;
    for(int it=0;it<9 && clock_time()<until;++it) {
        if(m) std::copy(p.b,p.b+m,w.begin());
        p.A.add(x.data(),w.data(),-1.);
        int count=0;
        for(auto& B:blocks) {
            auto c=B.c;
            const int o=c.off,k=c.width;
            if(!B.evaluate(w.data()+o,z.data()+o,strategy))return false;
            B.fwd(w.data()+o,r.data()+o);
            for(int j=0;j<k;++j) {
                double res=r[o+j]-B.pr[j], th=std::clamp(B.theta[j],0.,1.);
                if(th>=1.-1e-12) {
                    index[o+j]=-1;bd[o+j]=-res/B.rho;
                } else {
                    index[o+j]=count++;
                    r[o+j]=res/(1.-th);feas[o+j]=-B.rho*th/(1.-th);
                    bd[o+j]=0.;
                }
            }
            B.bwd(B.pr.data(),s.data()+o);
            B.bwd(bd.data()+o,dz.data()+o);
            for(int j=0;j<k;++j)zproj[o+j]=z[o+j]+(s[o+j]-w[o+j])/B.rho;
        }
        if(accept(x,s,z) || accept(x,s,zproj))return true;
        Vec rhs(n+count),u(n+count);
        if(n)std::copy(p.q,p.q+n,grad.begin());
        p.P.add(x.data(),grad.data());
        p.A.add_transpose(z.data(),grad.data());
        double pf=0.;
        for(int i=0;i<m;++i)pf=std::max(pf,std::abs(s[i]-w[i]));
        double merit=std::max(norm(grad)/(1+norm(p.q,n)),pf/(1+norm(p.b,m)));
        if(it>1 && merit>best*4.)return false;
        best=std::min(best,merit);
        p.A.add_transpose(dz.data(),grad.data());
        Int signs(n+count,-1);
        std::fill(signs.begin(),signs.begin()+n,1);
        std::vector<Entry> es;
        es.reserve(size_t(n+count)+p.P.nz()/2+p.A.nz()*2);
        for(int j=0;j<n+count;++j)es.push_back({j,j,0.});
        for(int j=0;j<m;++j)if(index[j]>=0) {
            int i=n+index[j];es[i].v=feas[j];rhs[i]=r[j];
        }
        for(int j=0;j<n;++j) {
            rhs[j]=-grad[j];
            for(int k=p.P.p[j];k<p.P.p[j+1];++k) {
                int i=p.P.i[k];double a=p.P.v[k];
                if(i==j)es[j].v+=a;
                else if(i<j)es.push_back({i,j,a});
            }
            for(int k=p.A.p[j];k<p.A.p[j+1];++k) {
                int i=p.A.i[k];double a=p.A.v[k];
                auto& B=blocks[block_id[i]];
                int o=B.c.off,w=B.c.width;
                if(B.Q.empty()) {
                    if(index[i]>=0)es.push_back({j,n+index[i],a});
                } else {
                    int pos=i-o;
                    for(int t=0;t<w;++t)if(index[o+t]>=0) {
                        double v=B.Q[size_t(t)*w+pos]*a;
                        if(v!=0.)es.push_back({j,n+index[o+t],v});
                    }
                }
            }
            if(es.size()>4800000)return false;
        }
        LinearSystem sys(n+count,signs,es,count==0);
        if(!sys.ok || sys.work_est>2e8 || clock_time()>until)return false;
        Vec values;values.reserve(es.size());
        for(auto e:es)values.push_back(e.v);
        if(!sys.factor(values,1e-9) && !sys.factor(values,6e-8,true))return false;
        if(!sys.solve(rhs,u,6))return false;
        for(auto& B:blocks) {
            const int o=B.c.off,k=B.c.width;
            for(int i=0;i<k;++i)if(index[o+i]>=0)bd[o+i]=u[n+index[o+i]];
            B.bwd(bd.data()+o,dz.data()+o);
        }
        double sx=norm(u.data(),n)/(1+norm(x)), sz=norm(dz)/(1+norm(z));
        double step=std::min(1.,3./std::max(sx,sz));
        if(step<1e-6 || !std::isfinite(sx+sz))return false;
        if(it>1 && merit>2*best)step*=.5;
        for(int j=0;j<n;++j)x[j]+=step*u[j];
        for(int j=0;j<m;++j)z[j]+=step*dz[j];
    }
    return false;
}

} // namespace conic
