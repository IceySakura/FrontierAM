#include "kkt.hpp"
#include "numeric.hpp"

#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>

extern "C" {
void scipy_dsytrf_(const char*,const int*,double*,const int*,int*,double*,const int*,int*);
void scipy_dsytrs_(const char*,const int*,const int*,const double*,const int*,const int*,double*,const int*,int*);
void scipy_dpotrf_(const char*,const int*,double*,const int*,int*);
void scipy_dpotrs_(const char*,const int*,const int*,const double*,const int*,double*,const int*,int*);
}

namespace {
using U64=std::uint64_t;
inline U64 key(int i,int j){return (U64(j)<<32)|unsigned(i);}
struct Edge {
    U64 k;
    int id;
    bool operator<(const Edge& e) const { return k<e.k; }
};
constexpr int fill_cap=26000000;

// Symbolic and numeric simplicial LDL using the elimination tree. The input
// is the upper triangle of a statically regularized quasi-definite system.
class LDL {
    int n;
    Ind p,r, parent,lp,li, next,flag,seq, path, at, orig, var;
    Vec b,diag,y,lx;
    std::vector<int8_t> sign;
    long long flops=0;
    bool cache=false;
    Ind reach;
    int reaches=0;
public:
    LDL(const Sparse& a,const Ind& perm,const std::vector<int8_t>& sig,long long budget) : n(a.cols),p(n+1),r(a.nnz()),
            parent(n,-1),lp(n+1),next(n),flag(n,-1),seq(n),path(n),at(n),
            orig(a.nnz()),var(perm),b(a.nnz()),diag(n),y(n),sign(n) {
        for(int j=0;j<n;++j) {
            sign[perm[j]]=sig[j];
            for(int k=a.p[j];k<a.p[j+1];++k) ++p[1+std::max(perm[j],perm[a.i[k]])];
        }
        for(int j=0;j<n;++j) p[j+1]+=p[j];
        {
            Ind q=p;
            for(int j=0;j<n;++j) for(int k=a.p[j];k<a.p[j+1];++k){
                int v=perm[j],u=perm[a.i[k]];
                if(u>v) std::swap(u,v);
                int s=q[v]++;
                r[s]=u; orig[k]=s;
                if(u==v) at[v]=s;
            }
        }
        long long total=0;
        for(int j=0;j<n;++j) {
            flag[j]=j;
            for(int k=p[j];k<p[j+1];++k) for(int i=r[k];flag[i]!=j;i=parent[i]){
                if(parent[i]<0) parent[i]=j;
                flag[i]=j;
                ++lp[i+1];
                if(++total>fill_cap) throw std::runtime_error("sparse fill limit");
            }
        }
        for(int j=0;j<n;++j){
            flops+=lp[j+1]*(long long)(lp[j+1]+1);
            if(flops>budget) throw std::runtime_error("sparse operation budget");
            lp[j+1]+=lp[j];
        }
        li.resize(total);
        lx.resize(total);
        cache=(total>0 && total<8'500'000);
        if(cache) reach.reserve(total);
        if(debug) std::fprintf(stderr,"  LDL %lld entries, %lld work\n",total,flops);
    }
    int elements() const {return lp.back();}
    bool calculate(const Vec& v,double reg) {
        for(size_t k=0;k<orig.size();++k) b[orig[k]]=v[k];
        std::copy(lp.begin(),lp.end()-1,next.begin());
        std::fill(y.begin(),y.end(),0.0);
        std::fill(flag.begin(),flag.end(),-1);
        bool first=cache && reach.empty() && elements()!=0;
        if(first) reaches=0;
        for(int k=0;k<n;++k){
            double d=b[at[k]]+sign[k]*reg;
            int pos=0,count=0;
            if(!cache || first){
                int top=n;
                flag[k]=k;
                for(int j=p[k];j<p[k+1];++j) {
                    int v=r[j];
                    if(v==k) continue;
                    y[v]=b[j];
                    int len=0;
                    for(;flag[v]!=k;v=parent[v]) {
                        path[len++]=v;flag[v]=k;
                    }
                    while(len) seq[--top]=path[--len];
                }
                count=n-top;
                if(first){
                    pos=reach.size();
                    reach.insert(reach.end(),seq.begin()+top,seq.end());
                    atcache(k,count);
                } else pos=top;
            } else {
                int countstart=reachptr[k];
                pos=countstart;count=reachptr[k+1]-countstart;
                for(int j=p[k];j<p[k+1];++j) if(r[j]!=k) y[r[j]]=b[j];
            }
            const int* pattern=(cache?reach.data():seq.data());
            for(int j=pos;j<pos+count;++j){
                int i=pattern[j],end=next[i];
                double w=y[i]; y[i]=0;
                for(int l=lp[i];l<end;++l) y[li[l]]-=lx[l]*w;
                double l=w*diag[i];
                d-=l*w;
                li[end]=k; lx[end]=l; next[i]=end+1;
            }
            if(!std::isfinite(d)) {if(first){reach.clear();reachptr.clear();} return false;}
            if(d*sign[k]<1e-14) d=sign[k]*std::max(1e-12,reg*0.02);
            diag[k]=1/d;
        }
        return true;
    }
    void solve(const Vec& f,Vec& v) {
        for(int i=0;i<n;++i) y[var[i]]=f[i];
        for(int j=0;j<n;++j){
            double z=y[j];
            for(int k=lp[j];k<lp[j+1];++k) y[li[k]]-=lx[k]*z;
        }
        for(int j=0;j<n;++j) y[j]*=diag[j];
        for(int j=n-1;j>=0;--j){
            double z=y[j];
            for(int k=lp[j];k<lp[j+1];++k) z-=lx[k]*y[li[k]];
            y[j]=z;
        }
        for(int i=0;i<n;++i) v[i]=y[var[i]];
    }
private:
    Ind reachptr;
    void atcache(int k,int count) {
        if(k==0) reachptr.resize(n+1);
        reaches+=count;
        reachptr[k+1]=reaches;
    }
};

struct PsdSchur {
    int block, at, width;
    Ind active,scatter;
    Sparse a;
    Vec out,tmp,tmp1;
};
}

struct Newton::System {
    const Model& p;
    ProductCone& c;
    int n,m,nz,N,naux;
    bool dense,positive=false;
    Sparse K;
    Vec base,scaled,row,frhs,v,r,cor,temp,bmat,densework;
    Ind hdi, hoff, use, ipiv;
    std::vector<int8_t> sign;
    std::vector<PsdSchur> psd;
    std::unique_ptr<LDL> ldl;
    int variant;
    System(const Model& p_,ProductCone& c_,int variant_) : p(p_),c(c_),n(p.n()),m(p.m()),variant(variant_) {
        use.assign(m,-1);
        std::vector<int> elims;
        nz=0;
        for(size_t j=0;j<p.cones.size();++j){
            int b=c.start[j],e=c.start[j+1];
            if(c.modes[j]==2) elims.push_back(j);
            else for(int i=b;i<e;++i) use[i]=nz++;
        }
        naux=c.aux.size(); N=n+nz+naux;
        K=Sparse(N,N);
        if(N<=0) throw std::runtime_error("empty Newton system");
        psd.reserve(elims.size());
        if(!elims.empty()){
            Ind psdrow(m,-1);
            for(int v:elims){
                PsdSchur b;
                b.block=v; b.at=c.start[v]; b.width=c.start[v+1]-b.at;
                for(int k=b.at;k<b.at+b.width;++k) psdrow[k]=(int)psd.size();
                b.a.p.push_back(0); b.a.rows=b.width;
                b.tmp.resize(b.width); b.tmp1.resize(b.width);
                psd.push_back(std::move(b));
            }
            for(int j=0;j<n;++j) for(int k=p.A.p[j];k<p.A.p[j+1];++k){
                int i=p.A.i[k], v=psdrow[i];
                if(v<0) continue;
                auto& b=psd[v];
                if(b.active.empty() || b.active.back()!=j) {
                    b.active.push_back(j);b.a.p.push_back(b.a.nnz());
                }
                b.a.i.push_back(i-b.at);b.a.x.push_back(p.A.x[k]); ++b.a.p.back();
            }
            for(auto& b:psd) {
                b.a.cols=b.active.size();
                size_t ds=b.active.size()*(b.active.size()+1)/2;
                if(ds>20'000'000) throw std::runtime_error("semidefinite Schur limit");
                b.scatter.resize(ds);
                b.out.resize(ds);
            }
        }
        size_t nnz=N+(p.P.nnz()+n)/2+p.A.nnz()+c.off.size();
        for(auto& b:psd) nnz+=b.scatter.size();
        if(nnz>45'000'000) throw std::runtime_error("Newton matrix size");
        std::vector<Edge> entries;
        entries.reserve(nnz);
        auto entry=[&](int i,int j){int id=entries.size(); entries.push_back({key(i,j),id});};
        for(int j=0;j<N;++j) entry(j,j);
        Vec vals;
        vals.reserve(p.P.nnz()/2+p.A.nnz());
        for(int j=0;j<n;++j){
            for(int l=p.P.p[j];l<p.P.p[j+1] && p.P.i[l]<=j;++l){
                entry(p.P.i[l],j); vals.push_back(p.P.x[l]);
            }
            for(int l=p.A.p[j];l<p.A.p[j+1];++l) if(use[p.A.i[l]]>=0) {
                entry(j,n+use[p.A.i[l]]); vals.push_back(p.A.x[l]);
            }
        }
        int off0=entries.size(),aa=n+nz;
        sign.assign(N,1);
        std::fill(sign.begin()+n,sign.begin()+n+nz,-1);
        for(size_t j=0;j<p.cones.size();++j){
            int lo=c.start[j],w=c.start[j+1]-lo;
            if(c.modes[j]==4){
                for(int k=0;k<3;++k) for(int i=0;i<3;++i) entry(n+use[lo+i],aa+k);
                aa+=3;
            } else if(c.modes[j]==3){
                for(int i=0;i<w;++i) entry(n+use[lo+i],aa);
                for(int i=1;i<w;++i) entry(n+use[lo+i],aa+1);
                sign[aa+1]=-1;aa+=2;
            } else if(c.modes[j]==1) {
                for(int i=0;i<w;++i) entry(n+use[lo+i],aa);
                sign[n+use[lo]]=1; ++aa;
            } else if(c.modes[j]==0 && p.cones[j].kind>=Lorentz) {
                for(int k=1;k<w;++k) for(int i=0;i<k;++i) entry(n+use[lo+i],n+use[lo+k]);
            }
        }
        for(auto& b:psd){
            auto& q=b.active;
            for(size_t i=0;i<q.size();++i) for(size_t j=i;j<q.size();++j) entry(q[i],q[j]);
        }
        {
            std::sort(entries.begin(),entries.end());
            Ind mapping(entries.size());
            K.i.reserve(entries.size());
            U64 prev=UINT64_MAX;
            for(auto e:entries) {
                if(e.k!=prev){
                    ++K.p[(e.k>>32)+1];K.i.push_back(int(e.k&0xffffffffu));
                    prev=e.k;
                }
                mapping[e.id]=int(K.i.size())-1;
            }
            for(int j=0;j<N;++j) K.p[j+1]+=K.p[j];
            base.resize(K.i.size());
            hdi.assign(mapping.begin(),mapping.begin()+N);
            for(size_t j=0;j<vals.size();++j) base[mapping[N+j]]=vals[j];
            auto s=mapping.begin()+off0;
            hoff.assign(s,s+c.off.size());s+=c.off.size();
            for(auto& b:psd) {std::copy(s,s+b.scatter.size(),b.scatter.begin());s+=b.scatter.size();}
        }
        std::vector<Edge>().swap(entries);
        const int dense_limit=!psd.empty() && nz<=n/5?4000:1950;
        dense=N<=145 || (N<=dense_limit && K.i.size()>0.117*N*N);
        if(debug) std::fprintf(stderr,"KKT %d %zu %s, Schur blocks %zu\n",N,K.i.size(),dense?"dense":"sparse",psd.size());
        K.x.resize(K.i.size());scaled.resize(K.i.size());row.resize(N);temp.resize(N);
        frhs.resize(N);v.resize(N);r.resize(N);cor.resize(N);
        if(dense){
            bmat.resize(N*(size_t)N); ipiv.resize(N); densework.resize(96LL*N);
        } else {
            const long long work=variant==0?110'000'000LL:variant==1?630'000'000LL:410'000'000LL;
            auto order=order_mmd(K,variant==2?1:0);
            try { ldl=std::make_unique<LDL>(K,order,sign,work); }
            catch(const std::runtime_error&) {
                order=order_mmd(K,variant==2?0:1);
                ldl=std::make_unique<LDL>(K,order,sign,work);
            }
        }
    }
    bool factor(double reg) {
        K.x=base;
        for(int i=0;i<m;++i) if(use[i]>=0) K.x[hdi[n+use[i]]]-=c.diagonal[i];
        for(int i=0;i<naux;++i) K.x[hdi[n+nz+i]]+=c.aux[i];
        for(size_t i=0;i<hoff.size();++i) K.x[hoff[i]]+=c.off[i];
        for(auto& b:psd) {
            if(b.active.empty()) continue;
            cc_psd_schur(c.handle,b.block,b.a.cols,b.a.p.data(),b.a.i.data(),b.a.x.data(),b.out.data());
            for(size_t i=0;i<b.out.size();++i) K.x[b.scatter[i]]+=b.out[i];
        }
        std::fill(row.begin(),row.end(),0.0);
        for(int j=0;j<N;++j) {
            double d=row[j];
            for(int k=K.p[j];k<K.p[j+1];++k){
                double t=std::abs(K.x[k]);
                d=std::max(d,t);
                row[K.i[k]]=std::max(row[K.i[k]],t);
            }
            row[j]=d;
        }
        for(double& x:row) x=1/std::sqrt(std::clamp(x,1e-8,1e180));
        for(int j=0;j<N;++j) {
            const double dj=row[j];
            for(int k=K.p[j];k<K.p[j+1];++k) scaled[k]=K.x[k]*(row[K.i[k]]*dj);
        }
        if(ldl) return ldl->calculate(scaled,reg);
        auto pack=[&](){
            for(int j=0;j<N;++j){
                auto a=bmat.data()+j*(size_t)N;
                std::fill(a,a+j+1,0.0);
                for(int k=K.p[j];k<K.p[j+1];++k) a[K.i[k]]=scaled[k];
                a[j]+=sign[j]*reg;
            }
        };
        pack();
        int info,nwork=densework.size();
        if(N==n){
            scipy_dpotrf_("U",&N,bmat.data(),&N,&info);
            if(info==0){positive=true;return true;}
            pack();
        }
        positive=false;
        scipy_dsytrf_("U",&N,bmat.data(),&N,ipiv.data(),densework.data(),&nwork,&info);
        return info==0;
    }
    void apply(const Vec& x,Vec& y) {
        std::fill(y.begin(),y.end(),0.);
        for(int j=0;j<N;++j) {
            double vj=x[j], z=0;
            int lo=K.p[j],hi=K.p[j+1]-1;
            for(int k=lo;k<hi;++k) {
                int i=K.i[k]; double a=scaled[k];
                y[i]+=a*vj;z+=a*x[i];
            }
            y[j]=z+scaled[hi]*vj;
        }
    }
    bool inner(const Vec& f,Vec& x) {
        if(ldl){ldl->solve(f,x);return true;}
        x=f;
        int one=1,info;
        if(positive)scipy_dpotrs_("U",&N,&one,bmat.data(),&N,x.data(),&N,&info);
        else scipy_dsytrs_("U",&N,&one,bmat.data(),&N,ipiv.data(),x.data(),&N,&info);
        return info==0;
    }
    bool solve_once(const Vec& f,const Vec& g,Vec& x,Vec& z) {
        std::copy(f.begin(),f.end(),frhs.begin());
        for(int i=0;i<m;++i) if(use[i]>=0) frhs[n+use[i]]=g[i];
        std::fill(frhs.begin()+n+nz,frhs.end(),0.0);
        for(auto& b:psd){
            if(!b.a.cols) continue;
            cc_psd_mul(c.handle,b.block,g.data()+b.at,b.tmp.data(),1);
            for(int j=0;j<b.a.cols;++j){
                double sum=0;
                for(int k=b.a.p[j];k<b.a.p[j+1];++k) sum+=b.a.x[k]*b.tmp[b.a.i[k]];
                frhs[b.active[j]]+=sum;
            }
        }
        for(int j=0;j<N;++j) frhs[j]*=row[j];
        if(!inner(frhs,v)) return false;
        double target=2e-13*(1+fastnorm(frhs));
        double last=INFINITY;
        for(int it=0;it<(N<25000?6:3);++it) {
            apply(v,r);
            for(int j=0;j<N;++j) r[j]=frhs[j]-r[j];
            double err=norm(r);
            if(err<=target || !std::isfinite(err)) break;
            if(err>last*0.85){
                if(err>last && it) v.swap(temp);
                break;
            }
            if(!inner(r,cor) || norm(cor)>5*(1+fastnorm(v))) break;
            last=err;
            temp=v;
            for(int j=0;j<N;++j) v[j]+=cor[j];
        }
        for(int j=0;j<n;++j) x[j]=v[j]*row[j];
        for(int i=0;i<m;++i) if(use[i]>=0) z[i]=v[n+use[i]]*row[n+use[i]];
        for(auto& b:psd){
            for(int i=0;i<b.width;++i) b.tmp[i]=-g[b.at+i];
            for(int j=0;j<b.a.cols;++j){
                double v=x[b.active[j]];
                for(int k=b.a.p[j];k<b.a.p[j+1];++k) b.tmp[b.a.i[k]]+=b.a.x[k]*v;
            }
            cc_psd_mul(c.handle,b.block,b.tmp.data(),z.data()+b.at,1);
        }
        return true;
    }
    Vec pwork,awork,dx,dz,zero;
    bool solve(const Vec& f,const Vec& g,Vec& x,Vec& z) {
        if(!solve_once(f,g,x,z)) return false;
        if(psd.empty() || n==0) return true;
        pwork.resize(n);awork.resize(n);
        const double scale=1+fastnorm(f);
        const double tig=2e-12*scale;
        double last=INFINITY;
        for(int it=0;it<2;++it) {
            p.P.mul(x.data(),pwork.data());p.A.tmul(z.data(),awork.data());
            for(int j=0;j<n;++j) pwork[j]=f[j]-pwork[j]-awork[j];
            double err=norm(pwork);
            if(err<=tig || err>=last*.8 || !std::isfinite(err)) break;
            dx.resize(n);dz.resize(m);zero.resize(m);
            if(!solve_once(pwork,zero,dx,dz) || norm(dx)>2*(1+norm(x)) || norm(dz)>2*(1+norm(z))) break;
            for(int j=0;j<n;++j) x[j]+=dx[j];
            for(int j=0;j<m;++j) z[j]+=dz[j];
            last=err;
        }
        return true;
    }
};

Newton::Newton(const Model& p,ProductCone& c,int variant) : sys(std::make_unique<System>(p,c,variant)) {}
Newton::~Newton()=default;
bool Newton::factor(double reg) {return sys->factor(reg);}
bool Newton::solve(const Vec& xrhs,const Vec& zrhs,Vec& x,Vec& z) {return sys->solve(xrhs,zrhs,x,z);}
long long Newton::fill() const {return sys->ldl?sys->ldl->elements():sys->N*(long long)sys->N/2;}
