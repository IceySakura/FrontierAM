#pragma once
#include "common.hpp"

namespace conic {

// Affine substitution for a forest of two-variable equalities.
struct Substitution {
    struct Link {int u,v,row;double a,b;};
    const Problem& o;
    Problem p;
    Int ap,ai,pp,pi,map,rowmap,parent;
    Vec av,pv,q,b,shift,weight;
    std::vector<Link> tree;
    std::vector<unsigned char> remove;

    explicit Substitution(const Problem& orig):o(orig),p(orig) {}

    int root(int j) {
        if(parent[j]==j)return j;
        long double s=0.,w=1.;
        int i=j;
        while(parent[i]!=i) {
            s+=w*shift[i];w*=weight[i];i=parent[i];
        }
        shift[j]=double(s);weight[j]=double(w);parent[j]=i;
        return i;
    }

    bool build(double until) {
        const int m=o.m(),n=o.n();
        if(n<20 || n+m>1300000 || o.A.nz()>4500000 ||
            std::none_of(o.cones.begin(),o.cones.end(),[](auto c){return c.kind==Zero;}))return false;
        Int counts(m,0);
        for(int k=0;k<o.A.nz();++k)++counts[o.A.i[k]];
        Int idx(m,-1);
        std::vector<Link> links;
        for(auto c:o.cones)if(c.kind==Zero)for(int i=c.off;i<c.off+c.width;++i)
            if(counts[i]==2 && std::abs(o.b[i])<1e15) {
                idx[i]=int(links.size());
                links.push_back({-1,-1,i,0.,0.});
            }
        if(links.size()<size_t(std::max(6,n/35)))return false;
        for(int j=0;j<n;++j)for(int k=o.A.p[j];k<o.A.p[j+1];++k) {
            int i=idx[o.A.i[k]];
            if(i>=0) {
                auto& L=links[i];
                if(L.u<0){L.u=j;L.a=o.A.v[k];}
                else {L.v=j;L.b=o.A.v[k];}
            }
        }
        shift.assign(n,0.);weight.assign(n,1.);
        parent.resize(n);std::iota(parent.begin(),parent.end(),0);
        remove.assign(m,0);
        Int size(n,1);
        std::vector<Link> chosen;
        for(auto L:links) {
            int a=root(L.u),b=root(L.v);
            if(a==b)continue;
            double u=L.a*weight[L.u], v=L.b*weight[L.v];
            const double rhs=std::fma(-L.a,shift[L.u],std::fma(-L.b,shift[L.v],o.b[L.row]));
            if(!(std::abs(u)>0 && std::abs(v)>0))continue;
            if((std::abs(u)<.1*std::abs(v)) ||
               (std::abs(v)<=10.*std::abs(u) && size[a]>size[b])) {
                std::swap(a,b);std::swap(u,v);
            }
            const double w=-v/u,d=rhs/u;
            if(!(std::abs(w)>=1e-6 && std::abs(w)<=1e6 && std::abs(d)<1e14))continue;
            parent[a]=b;weight[a]=w;shift[a]=d;
            size[b]+=size[a];remove[L.row]=1;
            chosen.push_back(L);
        }
        if(chosen.size()<size_t(std::max(6,n/45)) || clock_time()>until)return false;
        map.assign(n,-1);
        int nr=0;
        for(int j=0;j<n;++j)if(root(j)==j)map[j]=nr++;
        for(int j=0;j<n;++j) {
            if(!std::isfinite(shift[j]) || !(std::abs(weight[j])>=1e-11 && std::abs(weight[j])<=1e11))
                return false;
            map[j]=map[parent[j]];
        }
        Vec rhs(o.b,o.b+m),cost(o.q,o.q+n);
        o.A.add(shift.data(),rhs.data(),-1.);
        o.P.add(shift.data(),cost.data());
        q.assign(nr,0.);
        for(int j=0;j<n;++j) q[map[j]]+=weight[j]*cost[j];
        p.cones.clear();rowmap.assign(m,-1);
        for(auto c:o.cones) {
            const int off=int(b.size());
            for(int i=c.off;i<c.off+c.width;++i)
                if(!remove[i]){rowmap[i]=int(b.size());b.push_back(rhs[i]);}
            const int w=int(b.size())-off;
            if(w>0)p.cones.push_back({c.kind,c.kind==Zero?w:c.dim,off,w});
        }
        auto build_csc=[&](CSC c,bool hessian,Int& ptr,Int& index,Vec& value) {
            ptr.assign(nr+1,0);
            std::vector<std::pair<int,double>> temp(c.nz());
            for(int j=0;j<n;++j)
                for(int r=c.p[j];r<c.p[j+1];++r)
                    if(hessian || !remove[c.i[r]])++ptr[map[j]+1];
            for(int j=0;j<nr;++j)ptr[j+1]+=ptr[j];
            Int pos=ptr;
            for(int j=0;j<n;++j)
                for(int r=c.p[j];r<c.p[j+1];++r) {
                    int i=c.i[r], row=hessian?map[i]:rowmap[i];
                    if(row>=0)temp[pos[map[j]]++]={row,c.v[r]*weight[j]*(hessian?weight[i]:1.)};
                }
            index.reserve(ptr.back());value.reserve(ptr.back());
            for(int j=0;j<nr;++j) {
                int a=ptr[j],end=ptr[j+1];
                ptr[j]=int(index.size());
                std::stable_sort(temp.begin()+a,temp.begin()+end,[](auto a,auto b){return a.first<b.first;});
                for(int k=a;k<end;) {
                    int i=temp[k].first;
                    long double v=temp[k++].second;
                    while(k<end && temp[k].first==i)v+=temp[k++].second;
                    if(v!=0) {index.push_back(i);value.push_back(double(v));}
                }
            }
            ptr[nr]=int(index.size());
        };
        build_csc(o.A,false,ap,ai,av);build_csc(o.P,true,pp,pi,pv);
        p.A={int(b.size()),nr,ap.data(),ai.data(),av.data()};p.P={nr,nr,pp.data(),pi.data(),pv.data()};
        p.q=q.data();p.b=b.data();

        Int gp(n+1,0), gi(2*chosen.size());
        for(auto L:chosen){++gp[L.u+1];++gp[L.v+1];}
        for(int j=0;j<n;++j)gp[j+1]+=gp[j];
        Int pos=gp;
        for(int k=0;k<(int)chosen.size();++k) {
            const auto L=chosen[k];
            gi[pos[L.u]++]=k;gi[pos[L.v]++]=k;
        }
        Int stack;
        std::vector<unsigned char> visited(n,0);
        for(int j=0;j<n;++j)if(parent[j]==j && gp[j]<gp[j+1]) {
            visited[j]=1;stack.push_back(j);
            while(!stack.empty()) {
                int u=stack.back();stack.pop_back();
                for(int k=gp[u];k<gp[u+1];++k) {
                    auto e=chosen[gi[k]];
                    if(e.v!=u){std::swap(e.u,e.v);std::swap(e.a,e.b);}
                    if(visited[e.u])continue;
                    visited[e.u]=1;tree.push_back(e);stack.push_back(e.u);
                }
            }
        }
        return tree.size()==chosen.size();
    }

    void recover(const Result& s,Result& r) const {
        const int n=o.n(),m=o.m();
        r.status=s.status;
        if(s.status!=PrimalInfeasible) {
            for(int j=0;j<n;++j)r.x[j]=std::fma(weight[j],s.x[map[j]],s.status==Solved?shift[j]:0.);
            for(int i=0;i<m;++i)r.s[i]=rowmap[i]<0?0.:s.s[rowmap[i]];
        }
        if(s.status==DualInfeasible)return;
        for(int i=0;i<m;++i)r.z[i]=rowmap[i]<0?0.:s.z[rowmap[i]];
        Vec g(n,0.);
        if(s.status==Solved) {
            std::copy(o.q,o.q+n,g.begin());o.P.add(r.x.data(),g.data());
        }
        o.A.add_transpose(r.z.data(),g.data());
        Vec residual(p.n(),0.),den(p.n(),0.);
        for(int j=0;j<n;++j) {
            residual[map[j]]+=weight[j]*g[j];den[map[j]]+=std::abs(weight[j]);
        }
        for(int j=0;j<n;++j)
            g[j]-=std::copysign(1.,weight[j])*(residual[map[j]]/den[map[j]]);
        for(auto it=tree.rbegin();it!=tree.rend();++it) {
            auto e=*it;
            double v=-g[e.u]/e.a;
            r.z[e.row]=v;g[e.v]=std::fma(e.b,v,g[e.v]);
        }
    }
};

} // namespace conic
