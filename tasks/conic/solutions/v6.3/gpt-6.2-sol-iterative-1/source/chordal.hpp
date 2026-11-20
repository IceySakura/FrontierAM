#pragma once
#include "kkt.hpp"

namespace conic {

struct Chordal {
    struct Clq {
        Int vertices, prev, fresh;
        int off=0,parent=-1;
    };
    struct Split {
        ConeInfo block;
        Int order, inv;
        std::vector<Clq> cliques;
    };
    const Problem& o;
    Problem p;
    Int ap,ai,pp,pi,rowmap;
    Vec av,pv,q,b;
    std::vector<std::pair<int,int>> links;
    std::vector<Split> split;
    bool valid=false;

    explicit Chordal(const Problem& orig):o(orig),p(orig) {}
    static int packed(int i,int j) {
        if(i>j)std::swap(i,j);
        return i+int(int64_t(j)*(j+1)/2);
    }

    bool graph(ConeInfo c,const Int& support,Split& out) {
        const int d=c.dim;
        Int zp(d+1,0),zi;
        int edges=0;
        for(int j=0;j<d;++j) {
            int offset=c.off+packed(0,j);
            for(int i=0;i<j;++i) if(support[offset+i] || o.b[offset+i]!=0.) {
                zi.push_back(i);++edges;
            }
            zi.push_back(j);zp[j+1]=int(zi.size());
        }
        if(edges>.16*double(d)*d) return false;
        Int order=minimum_degree(d,zp,zi);
        if(order.empty())return false;
        Int inv(d);
        for(int i=0;i<d;++i)inv[order[i]]=i;
        const int w=(d+63)/64;
        std::vector<uint64_t> adj(size_t(d)*w,0), work(w);
        for(int j=0;j<d;++j) for(int t=zp[j];t<zp[j+1]-1;++t) {
            int a=inv[zi[t]], b=inv[j];
            adj[size_t(a)*w+(b>>6)]|=uint64_t(1)<<(b&63);
            adj[size_t(b)*w+(a>>6)]|=uint64_t(1)<<(a&63);
        }
        std::vector<Int> occur(d);
        std::vector<Clq> cliques;
        int sum=0, largest=0;
        for(int k=0;k<d;++k) {
            const uint64_t* row=adj.data()+size_t(k)*w;
            Int C;
            C.push_back(k);
            for(int j=k>>6;j<w;++j) {
                uint64_t bits=row[j];
                if(j==(k>>6)) bits &= (k&63)==63 ? uint64_t(0) : (~uint64_t(0) << ((k&63)+1));
                while(bits) {
                    const int bit=__builtin_ctzll(bits);bits&=bits-1;
                    C.push_back(j*64+bit);
                }
            }
            for(size_t j=1;j<C.size();++j) {
                uint64_t* a=adj.data()+size_t(C[j])*w;
                for(int t=k>>6;t<w;++t) {
                    uint64_t b=row[t];
                    if(t==(k>>6))b&=(k&63)==63 ? uint64_t(0) : (~uint64_t(0) << ((k&63)+1));
                    a[t]|=b;
                }
            }
            bool maximal=true;
            for(int i:occur[k]) {
                auto& old=cliques[i].vertices;
                if(old.size()>=C.size() && std::includes(old.begin(),old.end(),C.begin(),C.end())) {
                    maximal=false;break;
                }
            }
            if(maximal) {
                const int idx=int(cliques.size()), s=int(C.size());
                sum+=s*(s+1)/2;largest=std::max(largest,s);
                if(sum>.82*c.width || largest>.75*d)return false;
                for(int i:C)occur[i].push_back(idx);
                Clq a;a.vertices=std::move(C);cliques.push_back(std::move(a));
            }
        }
        if(cliques.size()<2 || sum>.68*c.width)return false;

        // A maximum-overlap spanning forest is a clique tree for each component.
        const int nc=int(cliques.size());
        std::vector<uint16_t> sup(size_t(nc)*nc,0);
        for(auto& members:occur)for(size_t j=0;j<members.size();++j)for(size_t k=0;k<j;++k) {
            ++sup[size_t(members[j])*nc+members[k]];++sup[size_t(members[k])*nc+members[j]];
        }
        Int weights(nc,0), parent(nc,-1), index(nc,-1);
        std::vector<unsigned char> seen(d,0);
        out.order=std::move(order);out.inv=std::move(inv);out.block=c;
        for(int step=0;step<nc;++step) {
            int k=-1;
            for(int j=0;j<nc;++j)
                if(weights[j]>=0 && (k<0 || weights[j]>weights[k]))k=j;
            if(k<0)return false;
            weights[k]=-1;index[k]=step;
            Clq item=std::move(cliques[k]);
            item.parent=parent[k]<0?-1:index[parent[k]];
            for(int v:item.vertices) {
                if(seen[v])item.prev.push_back(v);
                else {item.fresh.push_back(v);seen[v]=1;}
            }
            if(item.parent<0 && !item.prev.empty())return false;
            if(item.parent>=0) {
                const auto& ps=out.cliques[item.parent].vertices;
                if(!std::includes(ps.begin(),ps.end(),item.prev.begin(),item.prev.end()))return false;
            }
            out.cliques.push_back(std::move(item));
            for(int i=0;i<nc;++i)if(weights[i]>=0 && sup[size_t(k)*nc+i]>weights[i]) {
                weights[i]=sup[size_t(k)*nc+i];parent[i]=k;
            }
        }
        return true;
    }

    bool build(int limit_mode) {
        if(std::none_of(o.cones.begin(),o.cones.end(),[&](ConeInfo c){
            return c.kind==PSD && c.dim>=(limit_mode?12:26) && c.dim<=1700;
        })) return false;
        Int support(o.m(),0);
        for(int k=0;k<o.A.nz();++k)++support[o.A.i[k]];
        rowmap.assign(o.m(),-1);
        p.cones.clear();
        for(auto c:o.cones) {
            Split s;
            if(c.kind==PSD && c.dim>=(limit_mode?12:26) && c.dim<=1700 && graph(c,support,s)) {
                for(auto& item:s.cliques) {
                    const auto& vs=item.vertices;
                    const int d=int(vs.size()),width=d*(d+1)/2;
                    item.off=int(b.size());
                    p.cones.push_back({d==1?Linear:PSD,d,item.off,width});
                    const Clq* par=item.parent<0?nullptr:&s.cliques[item.parent];
                    const int at=item.off;
                    b.resize(size_t(at)+width);
                    for(int j=0;j<d;++j) for(int i=0;i<=j;++i) {
                        int rr=c.off+packed(s.order[vs[i]],s.order[vs[j]]);
                        int nr=at+packed(i,j);
                        if(rowmap[rr]<0){rowmap[rr]=nr;b[nr]=o.b[rr];}
                        else {
                            if(!par)return false;
                            const auto& t=par->vertices;
                            int u=int(std::lower_bound(t.begin(),t.end(),vs[i])-t.begin());
                            int v=int(std::lower_bound(t.begin(),t.end(),vs[j])-t.begin());
                            if(u==int(t.size()) || v==int(t.size()) || t[u]!=vs[i] || t[v]!=vs[j])return false;
                            links.emplace_back(par->off+packed(u,v),nr);
                        }
                    }
                }
                split.push_back(std::move(s));
            } else {
                const int off=int(b.size());
                p.cones.push_back({c.kind,c.dim,off,c.width});
                for(int j=0;j<c.width;++j)rowmap[c.off+j]=off+j;
                b.insert(b.end(),o.b+c.off,o.b+c.off+c.width);
            }
        }
        if(split.empty())return false;
        const int n=o.n(),nr=n+int(links.size()), m=int(b.size());
        if(nr+m>2000000)return false;
        for(int k=0;k<o.A.nz();++k)if(rowmap[o.A.i[k]]<0)return false;
        ap.reserve(nr+1);ai.reserve(o.A.nz()+2*links.size());av.reserve(ai.capacity());
        ap.push_back(0);
        for(int j=0;j<n;++j) {
            for(int k=o.A.p[j];k<o.A.p[j+1];++k) {
                ai.push_back(rowmap[o.A.i[k]]);
                av.push_back(o.A.v[k]);
            }
            ap.push_back(int(ai.size()));
        }
        for(auto [a,b]:links) {
            ai.push_back(a);ai.push_back(b);av.push_back(1.);av.push_back(-1.);
            ap.push_back(int(ai.size()));
        }
        q.assign(nr,0.);
        if(n)std::copy(o.q,o.q+n,q.begin());
        pp.resize(nr+1,o.P.nz());std::copy(o.P.p,o.P.p+n+1,pp.begin());
        p.A={m,nr,ap.data(),ai.data(),av.data()};p.P={nr,nr,pp.data(),o.P.i,o.P.v};
        p.q=q.data();p.b=b.data();
        valid=true;return true;
    }

    bool complete(const Split& s,const Vec& z,double* out,double regularize) const {
        const int d=s.block.dim;
        Vec Z(size_t(d)*d,0.),S,T,rhs,work,eig;
        Int iw,seen;
        double zn=0.;
        for(int j=0;j<d;++j)for(int i=0;i<=j;++i) {
            int rr=rowmap[s.block.off+packed(s.order[i],s.order[j])];
            if(rr>=0) {
                double val=z[rr]*(i==j?1.:invroot2);
                Z[i+j*d]=Z[j+i*d]=val;
                zn=std::max(zn,std::abs(val));
            }
        }
        if(zn==0.) {
            std::fill(out,out+s.block.width,0.);return true;
        }
        for(int i=0;i<d;++i)Z[i+i*d]+=regularize*zn;
        for(auto& c:s.cliques) {
            const auto& sep=c.prev;
            const auto& fresh=c.fresh;
            const int k=int(sep.size()),h=int(fresh.size());
            if(k>0 && h>0) {
                S.resize(size_t(k)*k);T.resize(size_t(k)*k);rhs.resize(size_t(k)*h);
                for(int j=0;j<k;++j)for(int i=0;i<k;++i)S[i+j*k]=Z[sep[i]+sep[j]*d];
                for(int j=0;j<h;++j)for(int i=0;i<k;++i)rhs[i+j*k]=Z[sep[i]+fresh[j]*d];
                T=S;
                int info=0;
                bool good=chol(k,T.data());
                if(good) {
                    for(int i=0;i<k;++i)if(T[i+i*k]*T[i+i*k]<2e-12*zn){good=false;break;}
                }
                if(good) {
                    scipy_dpotrs_("L",&k,&h,T.data(),&k,rhs.data(),&k,&info);
                    if(info)return false;
                } else {
                    int lw=1+6*k+2*k*k,liw=3+5*k;
                    work.resize(lw);iw.resize(liw);eig.resize(k);
                    scipy_dsyevd_("V","L",&k,S.data(),&k,eig.data(),work.data(),&lw,iw.data(),&liw,&info);
                    if(info)return false;
                    T.resize(size_t(k)*h);
                    gemm("T","N",k,h,k,S.data(),k,rhs.data(),k,T.data(),k);
                    const double zero_tol=std::max(zn*2e-12,std::abs(eig.back())*3e-13);
                    for(int j=0;j<h;++j)for(int i=0;i<k;++i)
                        T[i+j*k]=eig[i]>zero_tol?T[i+j*k]/eig[i]:0.;
                    gemm("N","N",k,h,k,S.data(),k,T.data(),k,rhs.data(),k);
                }
                for(int f=0;f<h;++f)for(int v:seen) {
                    if(std::binary_search(sep.begin(),sep.end(),v))continue;
                    double a=0.;
                    for(int j=0;j<k;++j)a+=Z[v+sep[j]*d]*rhs[j+f*k];
                    Z[v+fresh[f]*d]=Z[fresh[f]+v*d]=a;
                }
            }
            seen.insert(seen.end(),fresh.begin(),fresh.end());
        }
        for(int j=0;j<d;++j) for(int i=0;i<=j;++i)
            out[packed(i,j)]=Z[s.inv[i]+s.inv[j]*d]*(i==j?1.:root2);
        return true;
    }

    bool recover(const Result& r,Result& original,double regularize=0.) const {
        original.status=r.status;
        if(r.status==Solved || r.status==DualInfeasible) {
            std::copy(r.x.begin(),r.x.begin()+o.n(),original.x.begin());
            if(r.status==DualInfeasible) {
                std::fill(original.s.begin(),original.s.end(),0.);
                o.A.add(original.x.data(),original.s.data(),-1.);
                return true;
            }
            for(int i=0;i<o.m();++i) original.s[i]=rowmap[i]<0?0.:r.s[rowmap[i]];
        }
        for(int i=0;i<o.m();++i)original.z[i]=rowmap[i]<0?0.:r.z[rowmap[i]];
        for(const auto& s:split) {
            if(r.status==Solved) {
                double* slack=original.s.data()+s.block.off;
                std::fill(slack,slack+s.block.width,0.);
                for(const auto& c:s.cliques) {
                    const int d=int(c.vertices.size());
                    for(int j=0;j<d;++j)for(int i=0;i<=j;++i)
                        slack[packed(s.order[c.vertices[i]],s.order[c.vertices[j]])]+=r.s[c.off+packed(i,j)];
                }
            }
            if(!complete(s,r.z,original.z.data()+s.block.off,regularize))return false;
        }
        return true;
    }
};

} // namespace conic
