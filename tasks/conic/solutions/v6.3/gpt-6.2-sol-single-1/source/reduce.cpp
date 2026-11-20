#include "reduce.hpp"

#include <cstdio>
#include <numeric>

namespace {
constexpr double isqrt2=0.7071067811865475244;
constexpr double euler=2.7182818284590452354;
void append_cone(Model& m,int k,int d){
    if(d<=0)return;
    if(k<=Orthant && !m.cones.empty() && m.cones.back().kind==k) m.cones.back().dim+=d;
    else m.cones.push_back({k,d});
}
Sparse from_row(const Sparse& a) { return a.transpose(); }

bool receive(Verifier& v,const Result& r) {
    if(r.status==Solved) return v.solved(r.x,r.s,r.z);
    if(r.status==PrimalInfeasible) return v.primal(r.z);
    if(r.status==DualInfeasible) return v.dual(r.x,&r.s);
    return false;
}
}

bool reduce_identity(const Model& p,Verifier& verify) {
    if(p.P.nnz() || p.n()<12) return false;
    int np=0,mp=0;
    for(auto c:p.cones) if(c.kind==Semidefinite){mp+=c.width();++np;}
    const int n=p.n(),m=p.m(),other=m-mp;
    if(!np || mp!=n || n<other*1.4 || n<15) return false;
    Ind kp(m,-1),ko(m,-1),of;
    int a=0,b=0,jj=0;
    Model d;
    for(auto c:p.cones) {
        int w=c.width();
        if(c.kind==Semidefinite) {
            for(int i=0;i<w;++i) kp[jj+i]=a++;
            d.cones.push_back(c);
        } else {
            for(int i=0;i<w;++i){ko[jj+i]=b++;of.push_back(jj+i);}
        }
        jj+=w;
    }
    Ind col(n,-1),back(n,-1),org(n);
    Vec diag(n);
    for(int j=0;j<n;++j){
        for(int k=p.A.p[j];k<p.A.p[j+1];++k){
            int i=p.A.i[k], r=kp[i];
            if(r>=0){
                if(col[j]>=0 || back[r]>=0) return false;
                col[j]=r;back[r]=j;diag[j]=p.A.x[k];org[j]=i;
            }
        }
        if(col[j]<0) return false;
    }
    d.q.resize(other); d.b.resize(n);
    for(int i=0;i<other;++i) d.q[i]=p.b[of[i]];
    Sparse rows(other,n);
    for(int r=0;r<n;++r){
        int j=back[r];
        double inv=1/diag[j], w=p.b[org[j]]*inv;
        d.b[r]=-p.q[j]*inv;
        if(!std::isfinite(inv) || !std::isfinite(d.b[r]) || !std::isfinite(w)) return false;
        for(int k=p.A.p[j];k<p.A.p[j+1];++k){
            int i=ko[p.A.i[k]];
            if(i>=0){
                d.q[i]-=p.A.x[k]*w;
                rows.i.push_back(i);rows.x.push_back(p.A.x[k]*inv);
            }
        }
        rows.p[r+1]=rows.nnz();
    }
    struct Tail{int row;double coef;};
    std::vector<Tail> tail;
    int o=0;
    for(auto c:p.cones) {
        if(c.kind==Semidefinite) continue;
        int w=c.width();
        if(c.kind!=Zero) {
            append_cone(d,c.kind,c.dim);
            if(c.kind==Exponential) {
                tail.push_back({o+1,-1});tail.push_back({o,-1});tail.push_back({o+2,euler});
            } else for(int i=0;i<w;++i)tail.push_back({o+i,1});
        }
        o+=w;
    }
    rows.cols+=tail.size();
    for(auto r:tail) {
        rows.i.push_back(r.row); rows.x.push_back(-r.coef);
        rows.p.push_back(rows.nnz());
        d.b.push_back(0);
    }
    d.A=from_row(rows); d.P=Sparse(other,other);
    if(debug) std::fprintf(stderr,"transpose PSD identity formulation (%d,%d) -> (%d,%d)\n",n,m,d.n(),d.m());
    auto t=optimize_core(d);
    if(!t) return false;
    Result z;
    if(t.status==Solved||t.status==PrimalInfeasible){
        z.status=t.status==Solved?Solved:DualInfeasible;
        z.x.resize(n);z.s.resize(m);
        for(int j=0;j<n;++j) z.x[j]=((t.status==Solved?p.b[org[j]]:0)-t.z[col[j]])/diag[j];
        p.A.mul(z.x.data(),z.s.data());
        for(int i=0;i<m;++i) z.s[i]=(t.status==Solved?p.b[i]:0)-z.s[i];
        for(int j=0;j<n;++j) z.s[org[j]]=t.z[col[j]];
        if(t.status==Solved){
            z.z.resize(m);
            for(int j=0;j<n;++j) z.z[org[j]]=t.s[col[j]];
            for(int i=0;i<other;++i) z.z[of[i]]=t.x[i];
        }
    } else {
        z.status=PrimalInfeasible; z.z.resize(m);
        for(int j=0;j<n;++j) z.z[org[j]]=t.s[col[j]];
        for(int i=0;i<other;++i) z.z[of[i]]=t.x[i];
    }
    return receive(verify,z);
}

bool substitute_equalities(const Model& p,Verifier& verify) {
    const int n=p.n(),m=p.m();
    if(!m || !n || p.A.nnz()>4'000'000)return false;
    if(std::none_of(p.cones.begin(),p.cones.end(),[](ConeSpec c){return c.kind==Zero;}))return false;
    std::vector<uint8_t> zero(m);
    int off=0;
    for(auto b:p.cones){
        if(b.kind==Zero)std::fill(zero.begin()+off,zero.begin()+off+b.dim,1);
        off+=b.width();
    }
    struct Pivot {int col=-1,link=-1;double a=0.,ratio=0.,priority=0.;};
    std::vector<Pivot> pivot(m);
    int count=0;
    for(int j=0;j<n;++j){
        if(p.q[j]!=0 || p.P.p[j]!=p.P.p[j+1])continue;
        int k=p.A.p[j],nr=p.A.p[j+1]-k;
        if(!nr || nr>2) continue;
        int index=p.A.i[k],link=nr==1?-1:p.A.i[k+1];
        if(link>=0 && zero[link]){
            if(zero[index])continue;
            std::swap(index,link);++k;
        }
        if(!zero[index])continue;
        double a=p.A.x[k];
        if(std::abs(a)<std::max(1e-250,verify.aScale*1e-14))continue;
        double s=link<0?0.:p.A.x[2*p.A.p[j]+1-k]/a;
        if(std::abs(s)>2500 || !std::isfinite(p.b[index]/a) || std::abs(p.b[index]/a)>1e18)continue;
        double pr=std::abs(a)/(1+std::abs(s));
        if(link<0)pr*=200.;
        auto& w=pivot[index];
        if(pr>w.priority) {
            if(w.col<0)++count;
            w={j,link,a,s,pr};
        }
    }
    if(count<std::max(2,(n+m)/1250))return false;
    Ind map(n), row(m);
    std::vector<uint8_t> drop(n);
    Vec b=p.b;
    for(int i=0;i<m;++i)if(pivot[i].col>=0){
        const auto& w=pivot[i];drop[w.col]=true;
        if(w.link>=0)b[w.link]-=w.ratio*b[i];
    }
    int j0=0;
    for(int j=0;j<n;++j)map[j]=drop[j]?-1:j0++;
    Model d;
    d.q.resize(j0);d.offset=p.offset;
    off=0;
    for(auto c:p.cones){
        int sz=0;
        for(int i=off;i<off+c.width();++i){
            row[i]=-1;
            if(pivot[i].col<0){row[i]=d.b.size();d.b.push_back(b[i]);++sz;}
        }
        if(c.kind==Zero)append_cone(d,Zero,sz);else d.cones.push_back(c);
        off+=c.width();
    }
    d.P=Sparse(j0,j0);d.A=Sparse(d.m(),j0);
    d.P.i.reserve(p.P.nnz());d.P.x.reserve(p.P.nnz());
    d.A.i.reserve(p.A.nnz());d.A.x.reserve(p.A.nnz());
    for(int j=0;j<n;++j)if(map[j]>=0){
        int v=map[j];d.q[v]=p.q[j];
        for(int l=p.P.p[j];l<p.P.p[j+1];++l){
            if(map[p.P.i[l]]<0)return false;
            d.P.i.push_back(map[p.P.i[l]]);d.P.x.push_back(p.P.x[l]);
        }
        for(int l=p.A.p[j];l<p.A.p[j+1];++l){
            int i=p.A.i[l],u=row[i];
            double a=p.A.x[l];
            if(u<0 && pivot[i].link>=0){u=row[pivot[i].link];a*=-pivot[i].ratio;}
            if(u>=0&&a){d.A.i.push_back(u);d.A.x.push_back(a);}
        }
        d.A.p[v+1]=d.A.nnz();d.P.p[v+1]=d.P.nnz();
    }
    if(!std::isfinite(norm(d.A.x)) || !std::isfinite(norm(d.b)))return false;
    d.A.compact();
    if(debug)std::fprintf(stderr,"substituted %d single-link equality variables\n",count);
    auto r=optimize_core(d);
    if(!r)return false;
    Result use;use.status=r.status;
    if(r.status!=DualInfeasible) {
        use.z.resize(m);
        for(int i=0;i<m;++i)if(row[i]>=0)use.z[i]=r.z[row[i]];
        for(int i=0;i<m;++i)if(row[i]<0&&pivot[i].link>=0)use.z[i]=-pivot[i].ratio*use.z[pivot[i].link];
    }
    if(r.status!=PrimalInfeasible) {
        use.x.resize(n);use.s.resize(m);
        for(int j=0;j<n;++j)if(map[j]>=0)use.x[j]=r.x[map[j]];
        const auto rem=p.A.mul(use.x);
        for(int i=0;i<m;++i) {
            if(row[i]>=0)use.s[i]=r.s[row[i]];
            else use.x[pivot[i].col]=((r.status==Solved?p.b[i]:0)-rem[i])/pivot[i].a;
        }
    }
    return receive(verify,use);
}

bool reduce_rows(const Model& p,Verifier& verify) {
    const int m=p.m(),n=p.n();
    if(m==0) return false;
    Ind count(m),one(m,-1);
    Vec coeff(m);
    for(int j=0;j<n;++j) for(int k=p.A.p[j];k<p.A.p[j+1];++k){
        int i=p.A.i[k];
        if(count[i]++==0){one[i]=j;coeff[i]=p.A.x[k];}
    }
    int i0=0, removed=0;
    Ind fix(n,-1),vc(n,-1);
    Vec xv(n),rhs=p.b;
    for(auto c:p.cones){
        if(c.kind==Zero){
            for(int i=i0;i<i0+c.dim;++i) if(count[i]==1 && fix[one[i]]<0){
                double v=p.b[i]/coeff[i];
                if(std::isfinite(v) && std::abs(v)<1e16 && std::abs(coeff[i])>verify.aScale*1e-15) {
                    int j=one[i];fix[j]=i;xv[j]=v; ++removed;
                }
            }
        }
        i0+=c.width();
    }
    Ind nzrows=count;
    if(removed){
        Vec shift=p.A.mul(xv);
        for(int i=0;i<m;++i) rhs[i]-=shift[i];
        for(int j=0;j<n;++j) if(fix[j]>=0) {
            rhs[fix[j]]=0;
            for(int k=p.A.p[j];k<p.A.p[j+1];++k)--nzrows[p.A.i[k]];
        }
    }
    int nd=0;
    for(int j=0;j<n;++j) if(fix[j]<0) {
        if(p.P.p[j]==p.P.p[j+1] && p.A.p[j]==p.A.p[j+1] && p.q[j]==0){
            fix[j]=-2;++removed;
        } else vc[j]=nd++;
    }
    struct Recipe{int a,b;double u,v;};
    struct Diagonal{int at; Ind rows;};
    struct Face{int at;bool vzero;};
    std::vector<Recipe> transform;
    transform.reserve(m);
    std::vector<Diagonal> diagonal;
    std::vector<Face> face;
    Model d;
    d.b.reserve(m);
    bool changed=removed>0;
    auto row=[&](int i,double a=1.,int j=-1,double b=0.){
        transform.push_back({i,j,a,b});
        d.b.push_back(a*rhs[i]+(j<0?0.:b*rhs[j]));
    };
    i0=0;
    for(auto c:p.cones){
        int len=c.width(), last=i0+len;
        if(c.kind<=Orthant) {
            int k=0;
            for(int i=i0;i<last;++i) {
                bool okay=c.kind==Zero?rhs[i]==0:rhs[i]>=0;
                if(nzrows[i]==0 && okay) changed=true;
                else {row(i);++k;}
            }
            append_cone(d,c.kind,k);
        } else if(c.kind==Lorentz && c.dim==2){
            row(i0,isqrt2,i0+1,isqrt2);row(i0,isqrt2,i0+1,-isqrt2);
            append_cone(d,Orthant,2);changed=true;
        } else if(c.kind==Semidefinite){
            int s=c.dim;
            Ind par(s), fn;
            std::vector<bool> gone(s);
            std::iota(par.begin(),par.end(),0);
            auto find=[&](int i){
                while(par[i]!=i){par[i]=par[par[i]];i=par[i];}
                return i;
            };
            for(int i=0;i<s;++i) {
                int r=i0+i*(i+3LL)/2;
                if(p.b[r]<0 && count[r]==0) {
                    Vec v(m);v[r]=1;
                    if(verify.primal(v))return true;
                }
                if(count[r]==0 && p.b[r]==0) {gone[i]=true;fn.push_back(r);}
            }
            int rowi=i0, nf=0;
            for(int j=0;j<s;++j) for(int i=0;i<=j;++i,++rowi) {
                if(i==j || (nzrows[rowi]==0 && rhs[rowi]==0)) continue;
                if(gone[i] || gone[j]){row(rowi);++nf;}
                else {int x=find(i), y=find(j); if(x!=y) par[y]=x;}
            }
            if(!fn.empty()){
                changed=true;
                diagonal.push_back({i0,std::move(fn)});
            }
            append_cone(d,Zero,nf);
            std::vector<Ind> comps(s);
            for(int i=0;i<s;++i) if(!gone[i]) comps[find(i)].push_back(i);
            for(auto& comp:comps) if(!comp.empty()) {
                int size=comp.size();
                if(size!=s || s==1)changed=true;
                for(int j=0;j<size;++j) for(int i=0;i<=j;++i)
                    row(i0+comp[j]*(comp[j]+1LL)/2+comp[i]);
                append_cone(d,size==1?Orthant:Semidefinite,size);
            }
        } else if(c.kind==Exponential && count[i0+1]==0 && p.b[i0+1]==0) {
            row(i0,-1);row(i0+2);
            append_cone(d,Orthant,2);face.push_back({i0,true});changed=true;
        } else if(c.kind==Exponential && count[i0+2]==0 && p.b[i0+2]==0) {
            row(i0,-1);append_cone(d,Orthant,1);
            row(i0+1);append_cone(d,Zero,1);
            face.push_back({i0,false});changed=true;
        } else {
            bool constant=true;
            for(int i=i0;constant && i<last;++i)constant=nzrows[i]==0;
            if(constant && (count[i0]==0 || len>=3)) {
                Vec cand(len);
                int test=cc_constant(c.kind,c.dim,rhs.data()+i0,cand.data(),3e-12);
                if(test==0) {changed=true;i0=last;continue;}
                if(test==1) {
                    Vec z(m);std::copy(cand.begin(),cand.end(),z.begin()+i0);
                    if(removed) {
                        auto az=p.A.tmul(z);
                        for(int j=0;j<n;++j) if(fix[j]>=0) z[fix[j]]-=az[j]/coeff[fix[j]];
                    }
                    if(verify.primal(z))return true;
                }
            }
            for(int i=i0;i<last;++i)row(i);
            append_cone(d,c.kind,c.dim);
        }
        i0=last;
    }
    if(!changed) return false;
    if(debug) std::fprintf(stderr,"presolve (%d,%d) -> (%d,%d)\n",n,m,nd,d.m());
    struct Target{int one=-1,two=-1;double a=0,b=0;};
    std::vector<Target> target(m);
    for(int i=0;i<d.m();++i){
        auto r=transform[i];
        auto add=[&](int j,double v){
            auto& t=target[j];
            if(t.one<0){t.one=i;t.a=v;}else{t.two=i;t.b=v;}
        };
        add(r.a,r.u);if(r.b>=0)add(r.b,r.v);
    }
    const auto sh=p.P.mul(xv);
    d.P=Sparse(nd,nd);
    d.A=Sparse(d.m(),nd);
    d.P.i.reserve(p.P.nnz()); d.P.x.reserve(p.P.nnz());
    d.A.i.reserve(p.A.nnz()); d.A.x.reserve(p.A.nnz());
    d.q.resize(nd);
    for(int j=0;j<n;++j) if(vc[j]>=0) {
        int r=vc[j];
        d.q[r]=p.q[j]+sh[j];
        for(int k=p.P.p[j];k<p.P.p[j+1];++k) if(vc[p.P.i[k]]>=0){
            d.P.i.push_back(vc[p.P.i[k]]);d.P.x.push_back(p.P.x[k]);
        }
        for(int k=p.A.p[j];k<p.A.p[j+1];++k) {
            const auto t=target[p.A.i[k]];
            if(t.one>=0){d.A.i.push_back(t.one);d.A.x.push_back(t.a*p.A.x[k]);}
            if(t.two>=0){d.A.i.push_back(t.two);d.A.x.push_back(t.b*p.A.x[k]);}
        }
        d.P.p[r+1]=d.P.nnz();d.A.p[r+1]=d.A.nnz();
    }
    d.A.compact();
    auto t=optimize_core(d);
    if(!t) return false;
    Result answer;
    answer.status=t.status;
    auto pull=[&](const Vec& v,Vec& w){
        for(int i=0;i<d.m();++i) {
            auto r=transform[i];
            w[r.a]+=r.u*v[i];
            if(r.b>=0)w[r.b]+=r.v*v[i];
        }
    };
    if(t.status!=PrimalInfeasible) {
        answer.x=t.status==Solved?xv:Vec(n);
        for(int j=0;j<n;++j)if(vc[j]>=0) answer.x[j]=t.x[vc[j]];
        answer.s.resize(m);
        pull(t.s,answer.s);
        Vec as=p.A.mul(answer.x);
        for(int i=0;i<m;++i)if(target[i].one<0)answer.s[i]=(t.status==Solved?p.b[i]:0)-as[i];
    }
    if(t.status!=DualInfeasible) {
        answer.z.resize(m);
        pull(t.z,answer.z);
        for(auto f:face) {
            double* v=answer.z.data()+f.at;
            double u=std::max(0.,-v[0]);
            if(f.vzero) {
                if(u>0){
                    v[2]=std::max(v[2],u*1e-28);
                    v[1]=u*(std::max(0.,std::log(u)-std::log(v[2])-1)+.01);
                }
            } else {
                double rg=u>0?-v[1]/u-1:INFINITY;
                if(u>0 && rg<42.) v[2]=u*std::exp(rg+0.01);
                else if(v[1]<0 || u>0) v[2]=2e8*std::max(u,-v[1]);
            }
        }
        auto restore=[&](){
            if(!removed)return;
            Vec station=p.A.tmul(answer.z);
            if(answer.status==Solved) {
                auto px=p.P.mul(answer.x);
                for(int j=0;j<n;++j) station[j]+=px[j]+p.q[j];
            }
            for(int j=0;j<n;++j) if(fix[j]>=0) answer.z[fix[j]]-=station[j]/coeff[fix[j]];
        };
        restore();
        if(receive(verify,answer))return true;
        if(!diagonal.empty()){
            double sc=(norm(answer.z)+1);
            for(double a:{10.,1e4,1e8,1e11}){
                for(auto& s:diagonal) for(int i:s.rows) answer.z[i]=a*sc;
                restore();
                if(receive(verify,answer))return true;
            }
        }
        return false;
    }
    return receive(verify,answer);
}
