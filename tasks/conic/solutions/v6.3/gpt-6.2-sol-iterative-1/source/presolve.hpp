#pragma once
#include "cones.hpp"

namespace conic {

inline bool constant_separator(const ConeInfo& C,const double* b,double* z) {
    const int d=C.dim,w=C.width;
    switch(C.kind) {
    case Zero: {
        int j=0;
        for(int i=1;i<w;++i) if(std::abs(b[i])>std::abs(b[j]))j=i;
        if(b[j]==0)return false;
        z[j]=b[j]>0?-1.:1.;return true;
    }
    case Linear: {
        int j=int(std::min_element(b,b+w)-b);
        if(b[j]>=0)return false;
        z[j]=1.;return true;
    }
    case Soc: {
        double r=0.;
        for(int i=1;i<d;++i)r=std::hypot(r,b[i]);
        if(b[0]>=r) return false;
        z[0]=1.;
        if(r>0.)for(int i=1;i<d;++i)z[i]=-b[i]/r;
        return true;
    }
    case Exp: {
        if(b[1]<0) {z[1]=1.;return true;}
        if(b[2]<0) {z[2]=1.;return true;}
        if(b[1]==0 && b[0]<=0) return false;
        double t=0.;
        if(b[1]>0 && b[2]>0) {
            t=std::log(b[2])-std::log(b[1]);
            if(b[0]<=b[1]*t)return false;
        } else if(b[1]>0) {
            t=std::clamp(b[0]/b[1]-1.,-500.,500.);
        } else if(b[2]>0) {
            t=std::clamp(std::log(b[2])-std::log(b[0])+1.,-500.,500.);
        }
        z[0]=-1.;z[1]=t-1.;z[2]=std::exp(-t);return true;
    }
    default: {
        if(d==1) {
            if(b[0]>=0)return false;
            z[0]=1.;return true;
        }
        Vec a(size_t(d)*d), work(size_t(36)*d), eig(d), v(d);
        Int iw(12*d);
        smat(d,b,a.data(),1./std::max(1.,norm(b,w)));
        int lo=1,hi=1,num=0,support[2],info=0,lw=int(work.size()),liw=int(iw.size());
        const double zero=0;
        scipy_dsyevr_("V","I","L",&d,a.data(),&d,&zero,&zero,&lo,&hi,&zero,&num,eig.data(),
            v.data(),&d,support,work.data(),&lw,iw.data(),&liw,&info);
        if(info || num!=1 || !(eig[0]<0))return false;
        int k=0;
        for(int j=0;j<d;++j)for(int i=0;i<=j;++i)
            z[k++]=v[i]*v[j]*(i==j?1.:root2);
        return true;
    }
    }
}

struct Reduction {
    struct Fix { int i,j;double a; };
    enum LiftKind { SocZero, ExpZero, SocConstant, ExpUpper, ExpLower, SocLinear };
    struct Lift { int kind,offset,width;double shift=0.,param=0.;int partner=-1; };
    const Problem& orig;
    Problem p;
    Vec bwork, q, b, av, pv, fixedvals, rsign;
    Int ap,ai,pp,pi,colmap,rowmap,reversecol;
    std::vector<Fix> fixed;
    std::vector<Lift> lifts;
    bool altered=false;
    bool has_cert=false;
    Result cert;

    explicit Reduction(const Problem& o):orig(o),p(o) {}

    void recover(const Result& reduced,Result& out) const {
        const int m=orig.m();
        out.status=reduced.status;
        if(reduced.status==Solved) {
            out.x=fixedvals;
            for(size_t j=0;j<reversecol.size();++j)out.x[reversecol[j]]=reduced.x[j];
            if(m)std::copy(orig.b,orig.b+m,out.s.begin());
            orig.A.add(out.x.data(),out.s.data(),-1.);
            for(auto c:orig.cones)if(c.kind==Zero)std::fill(out.s.begin()+c.off,out.s.begin()+c.off+c.width,0.);
            else if(c.kind==Linear)for(int i=c.off;i<c.off+c.width;++i)out.s[i]=std::max(out.s[i],0.);
            // Keep the interior iterate's slack on the unreduced cone blocks.
            for(int i=0;i<m;++i)
                if(rowmap[i]>=0)out.s[i]=rsign[i]*reduced.s[rowmap[i]];
            for(auto L:lifts) {
                if(L.kind==SocConstant || L.kind==ExpUpper)out.s[L.offset]+=L.shift;
                else if(L.kind==ExpLower)out.s[L.offset+2]+=L.shift;
                else if(L.kind==SocLinear) {
                    const double x=out.s[L.offset],y=out.s[L.partner];
                    out.s[L.offset]=(x+y)*invroot2;out.s[L.partner]=(x-y)*invroot2;
                }
            }
        } else if(reduced.status==DualInfeasible) {
            std::fill(out.x.begin(),out.x.end(),0.);
            for(size_t j=0;j<reversecol.size();++j)out.x[reversecol[j]]=reduced.x[j];
            std::fill(out.s.begin(),out.s.end(),0.);
            orig.A.add(out.x.data(),out.s.data(),-1.);
            return;
        }
        std::fill(out.z.begin(),out.z.end(),0.);
        for(int i=0;i<m;++i) if(rowmap[i]>=0)out.z[i]=rsign[i]*reduced.z[rowmap[i]];
        for(auto L:lifts) {
            double* z=out.z.data()+L.offset;
            if(L.kind==SocZero) {
                z[0]=0.;
                for(int j=1;j<L.width;++j)z[0]=std::hypot(z[0],z[j]);
            } else if(L.kind==SocLinear) {
                const double x=z[0], y=out.z[L.partner];
                z[0]=(x+y)*invroot2;out.z[L.partner]=(x-y)*invroot2;
            } else if(L.kind==SocConstant) {
                if(L.shift>0.) for(int j=1;j<L.width;++j)
                    z[j]=-z[0]*(bwork[L.offset+j]/L.shift);
            } else if(L.kind==ExpUpper) {
                const double lam=-z[0];
                z[1]=lam*(L.param-1.);
                z[2]=lam*std::exp(-L.param);
            } else if(L.kind==ExpLower) {
                const double lam=z[2]*std::exp(L.param);
                z[0]=-lam;
                z[1]=lam*(L.param-1.);
            } else {
                z[0]=std::min(0.,z[0]);z[2]=std::max(0.,z[2]);
                const double r=-z[0];
                if(r>0.) {
                    z[2]=std::max(z[2],std::max(r,1e-12)*2e-15);
                    z[1]=std::max(0.,-r*(1.+std::log(z[2])-std::log(r)));
                } else z[1]=0.;
            }
        }
        recover_equalities(out);
    }

    void recover_equalities(Result& r) const {
        const auto& A=orig.A;
        for(auto it=fixed.rbegin();it!=fixed.rend();++it) {
            const int j=it->j,i=it->i;
            long double v=0;
            if(r.status==Solved) {
                v=orig.q[j];
                const auto& P=orig.P;
                for(int k=P.p[j];k<P.p[j+1];++k)
                    v+=(long double)P.v[k]*r.x[P.i[k]];
            }
            for(int k=A.p[j];k<A.p[j+1];++k)
                if(A.i[k]!=i) v+=(long double)A.v[k]*r.z[A.i[k]];
            r.z[i]=double(-v/it->a);
        }
    }

    template<class Verify>
    bool build(Verify&& verify,bool facial=true) {
        const int n=orig.n(),m=orig.m();
        if(m==0 || n+m>4000000)return false;
        Int start(m+1,0),count(m,0);
        for(int k=0;k<orig.A.nz();++k)++count[orig.A.i[k]];
        for(int i=0;i<m;++i)start[i+1]=start[i]+count[i];
        std::vector<unsigned char> eq(m,0),locked(n,0),is_fix(m,0);
        Int queue;
        int singletons=0, constants=0;
        for(auto c:orig.cones) {
            if(c.kind==Zero)for(int i=c.off;i<c.off+c.width;++i) {
                eq[i]=1;
                if(count[i]==1) {++singletons;queue.push_back(i);}
            }
            if(c.kind<=Linear)for(int i=c.off;i<c.off+c.width;++i)constants+=count[i]==0;
            else if(c.kind==PSD)constants+=2;
            else if(c.kind==Soc) {
                constants+=c.dim==2 || count[c.off]==0 ||
                    std::none_of(count.begin()+c.off+1,count.begin()+c.off+c.width,[](int k){return k>0;});
            }
            else if(c.kind==Exp)constants+=count[c.off+1]==0 || count[c.off+2]==0;
        }
        if(singletons+constants==0)return false;
        bwork.assign(orig.b,orig.b+m);
        fixedvals.assign(n,0.);
        if(singletons) {
            Int owner(orig.A.nz()), pos=start;
            Vec coeff(orig.A.nz());
            for(int j=0;j<n;++j)
                for(int k=orig.A.p[j];k<orig.A.p[j+1];++k) {
                    int t=pos[orig.A.i[k]]++;owner[t]=j;coeff[t]=orig.A.v[k];
                }
            for(size_t index=0;index<queue.size();++index) {
                int i=queue[index];
                if(count[i]!=1 || is_fix[i])continue;
                int k=start[i];
                while(k<start[i+1] && locked[owner[k]])++k;
                if(k==start[i+1])continue;
                int j=owner[k];
                double a=coeff[k], t=bwork[i]/a;
                if(!std::isfinite(t) || std::abs(t)>1e30) continue;
                fixed.push_back({i,j,a});fixedvals[j]=t;locked[j]=1;is_fix[i]=1;
                for(int k=orig.A.p[j];k<orig.A.p[j+1];++k) {
                    int r=orig.A.i[k];
                    --count[r];
                    bwork[r]=std::fma(-orig.A.v[k],t,bwork[r]);
                    if(eq[r] && count[r]==1)queue.push_back(r);
                }
                bwork[i]=0.;
            }
        }
        p.cones.clear();
        rowmap.assign(m,-1);rsign.assign(m,1.);
        colmap.assign(n,-1);
        for(int j=0;j<n;++j) if(!locked[j]) {
            colmap[j]=int(reversecol.size());reversecol.push_back(j);
        }
        const double bscale=norm(orig.b,m);
        Vec local_ray;
        auto try_cert=[&](ConeInfo c) {
            if(bscale==0 || has_cert)return;
            local_ray.assign(c.width,0.);
            if(!constant_separator(c,bwork.data()+c.off,local_ray.data()))return;
            const double ns=norm(local_ray);
            const double margin=-dot_precise(bwork.data()+c.off,local_ray.data(),c.width);
            if (!(margin>eps*ns*bscale))return;
            cert.z.assign(m,0.);
            std::copy(local_ray.begin(),local_ray.end(),cert.z.begin()+c.off);
            cert.status=PrimalInfeasible;
            recover_equalities(cert);
            has_cert=verify(cert);
        };
        auto append=[&](int r,double sgn=1.) {
            rowmap[r]=int(b.size());rsign[r]=sgn;b.push_back(sgn*bwork[r]);
        };
        auto add_cone=[&](int kind,int dim,int off) {
            if(dim>0)p.cones.push_back({kind,dim,off,int(b.size())-off});
        };
        Int rotation;

        for(auto C:orig.cones) {
            int off=int(b.size());
            if(C.kind<=Linear || (C.kind==PSD && C.dim==1)) {
                const int kind=C.kind==Zero?Zero:Linear;
                for(int i=C.off;i<C.off+C.width;++i) {
                    if(is_fix[i]) continue;
                    if(count[i]==0) {
                        const double bad=kind==Zero?std::abs(bwork[i]):-bwork[i];
                        if(bad>0)try_cert({kind,1,i,1});
                        if(has_cert)return true;
                        if(bad<=1e-13*(1.+bscale))continue;
                    }
                    append(i);
                }
                add_cone(kind,int(b.size())-off,off);
                if(C.kind==PSD)altered=true;
            } else {
                if(C.kind==Soc && count[C.off]==0 && bwork[C.off]<0.)try_cert({Linear,1,C.off,1});
                if(C.kind==Exp)for(int j=1;j<3;++j)
                    if(count[C.off+j]==0 && bwork[C.off+j]<0.)try_cert({Linear,1,C.off+j,1});
                if(C.kind==PSD)for(int j=0;j<C.dim;++j) {
                    int pos=C.off+j*(j+3)/2;
                    if(count[pos]==0 && bwork[pos]<0.)try_cert({Linear,1,pos,1});
                }
                if(has_cert)return true;
                bool empty=true;
                for(int i=C.off;i<C.off+C.width;++i)if(count[i]>0){empty=false;break;}
                if(empty) {
                    try_cert(C);
                    if(has_cert)return true;
                    // Do not drop a constant constraint unless its cone check passes.
                    Problem temp{orig.P,orig.A,orig.q,orig.b,{{C.kind,C.dim,0,C.width}}};
                    Vec slack(bwork.begin()+C.off,bwork.begin()+C.off+C.width);
                    if(cone_defect(temp,slack,false,5e-12)<=5e-12)continue;
                }
                int one_tail=-1;
                if(facial && C.kind==Soc)for(int i=C.off+1;i<C.off+C.width;++i)
                    if(count[i]>0 || bwork[i]!=0.) {
                        if(one_tail>=0){one_tail=-2;break;}
                        one_tail=i;
                    }
                if(facial && C.kind==Soc && one_tail>=0 &&
                    std::max(std::abs(bwork[C.off]),std::abs(bwork[one_tail]))<1e290) {
                    if(rotation.empty())rotation.assign(m,-1);
                    rotation[C.off]=one_tail;rotation[one_tail]=C.off;
                    append(C.off);append(one_tail);
                    b[off]=(bwork[C.off]+bwork[one_tail])*invroot2;
                    b[off+1]=(bwork[C.off]-bwork[one_tail])*invroot2;
                    add_cone(Linear,2,off);
                    lifts.push_back({SocLinear,C.off,C.width,0.,0.,one_tail});
                } else if(facial && C.kind==Soc && count[C.off]==0 && bwork[C.off]==0.) {
                    for(int i=1;i<C.width;++i)append(C.off+i);
                    add_cone(Zero,C.width-1,off);
                    lifts.push_back({SocZero,C.off,C.width});
                } else if(facial && C.kind==Soc &&
                          std::none_of(count.begin()+C.off+1,count.begin()+C.off+C.width,[](int k){return k>0;})) {
                    double tail=0.;
                    for(int j=1;j<C.width;++j)tail=std::hypot(tail,bwork[C.off+j]);
                    append(C.off);b.back()-=tail;
                    add_cone(Linear,1,off);
                    lifts.push_back({SocConstant,C.off,C.width,tail});
                } else if(facial && C.kind==Exp && count[C.off+1]==0 && bwork[C.off+1]==0.) {
                    append(C.off,-1.);append(C.off+2);
                    add_cone(Linear,2,off);
                    lifts.push_back({ExpZero,C.off,C.width});
                } else if(facial && C.kind==Exp && count[C.off+1]==0 && bwork[C.off+1]>0) {
                    const double v=bwork[C.off+1], w=bwork[C.off+2];
                    const double t=(count[C.off+2]==0 && w>0.) ? std::log(w)-std::log(v) : inf;
                    const double u=(count[C.off]==0) ? bwork[C.off]/v : inf;
                    if(std::abs(t)<650 && std::isfinite(v*t)) {
                        double shift=v*t;
                        append(C.off,-1.);b.back()+=shift;
                        add_cone(Linear,1,off);
                        lifts.push_back({ExpUpper,C.off,3,shift,t});
                    } else if(std::abs(u)<650 && std::isfinite(v*std::exp(u))) {
                        double shift=v*std::exp(u);
                        append(C.off+2);b.back()-=shift;
                        add_cone(Linear,1,off);
                        lifts.push_back({ExpLower,C.off,3,shift,u});
                    } else {
                        for(int i=0;i<C.width;++i)append(C.off+i);
                        add_cone(C.kind,C.dim,off);
                    }
                } else if(C.kind==PSD && C.dim>1) {
                    // PSD constraints with disconnected structural support split
                    // into independent principal submatrices.
                    const int d=C.dim;
                    Int parent(d),sizes(d,1);
                    std::iota(parent.begin(),parent.end(),0);
                    auto root=[&](int i) {
                        while(i!=parent[i]){parent[i]=parent[parent[i]];i=parent[i];}
                        return i;
                    };
                    int components=d, r=C.off;
                    for(int j=0;j<d;++j) {
                        for(int i=0;i<j;++i,++r)if(count[r] || bwork[r]!=0.) {
                            int a=root(i),b=root(j);
                            if(a!=b) {
                                if(sizes[a]<sizes[b])std::swap(a,b);
                                parent[b]=a;sizes[a]+=sizes[b];--components;
                            }
                        }
                        ++r;
                    }
                    if(components==1) {
                        for(int i=0;i<C.width;++i)append(C.off+i);
                        add_cone(PSD,d,off);
                    } else {
                        std::vector<Int> sets(d);
                        for(int i=0;i<d;++i)sets[root(i)].push_back(i);
                        for(auto& set:sets)if(!set.empty()) {
                            int r=int(b.size());
                            for(int j:set) {
                                int start=C.off+int(int64_t(j)*(j+1)/2);
                                for(int i:set) {if(i>j)break;append(start+i);}
                            }
                            add_cone(set.size()==1?Linear:PSD,int(set.size()),r);
                        }
                    }
                } else {
                    for(int i=0;i<C.width;++i)append(C.off+i);
                    add_cone(C.kind,C.dim,off);
                }
            }
        }
        altered=altered || !fixed.empty() || int(b.size())<m || !lifts.empty();
        if(!altered)return false;
        int nr=int(reversecol.size()),mr=int(b.size());
        Vec cost(n);
        if(n)std::copy(orig.q,orig.q+n,cost.begin());
        if(!fixed.empty())orig.P.add(fixedvals.data(),cost.data());
        q.reserve(nr);
        ap.reserve(nr+1);pp.reserve(nr+1);
        ap.push_back(0);pp.push_back(0);
        ai.reserve(orig.A.nz());av.reserve(orig.A.nz());
        pi.reserve(orig.P.nz());pv.reserve(orig.P.nz());
        std::vector<std::pair<int,double>> col;
        for(int j:reversecol) {
            q.push_back(cost[j]);
            if(rotation.empty()) {
                for(int k=orig.A.p[j];k<orig.A.p[j+1];++k) {
                    int r=orig.A.i[k];
                    if(rowmap[r]>=0){ai.push_back(rowmap[r]);av.push_back(rsign[r]*orig.A.v[k]);}
                }
            } else {
                col.clear();
                for(int k=orig.A.p[j];k<orig.A.p[j+1];++k) {
                    int r=orig.A.i[k],v=rotation[r];
                    double a=orig.A.v[k];
                    if(v<0) {if(rowmap[r]>=0)col.emplace_back(rowmap[r],rsign[r]*a);}
                    else {
                        col.emplace_back(rowmap[std::min(r,v)],invroot2*a);
                        col.emplace_back(rowmap[std::max(r,v)],r<v?invroot2*a:-invroot2*a);
                    }
                }
                std::sort(col.begin(),col.end(),[](auto a,auto b){return a.first<b.first;});
                for(size_t k=0;k<col.size();) {
                    int i=col[k].first;double v=col[k++].second;
                    while(k<col.size() && col[k].first==i)v+=col[k++].second;
                    if(v!=0.){ai.push_back(i);av.push_back(v);}
                }
            }
            for(int k=orig.P.p[j];k<orig.P.p[j+1];++k) {
                int i=colmap[orig.P.i[k]];
                if(i>=0){pi.push_back(i);pv.push_back(orig.P.v[k]);}
            }
            ap.push_back(int(ai.size()));pp.push_back(int(pi.size()));
        }
        p.A={mr,nr,ap.data(),ai.data(),av.data()};p.P={nr,nr,pp.data(),pi.data(),pv.data()};
        p.q=q.data();p.b=b.data();
        return true;
    }
};

} // namespace conic
