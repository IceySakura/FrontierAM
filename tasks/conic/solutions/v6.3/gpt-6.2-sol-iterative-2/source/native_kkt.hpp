#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <vector>

extern "C" {
void* conic_ldl_create(int,const int*,const int*,const int*,const double*,int,double);
int conic_ldl_factor(void*,const double*);
void conic_ldl_solve(void*,const double*,double*);
void conic_ldl_free(void*);
}

namespace conic {
using I=std::vector<int>;
using V=std::vector<double>;
using Clock=std::chrono::steady_clock;

struct Lapack {
    using Potrf=void(*)(const char*,const int*,double*,const int*,int*);
    using Potrs=void(*)(const char*,const int*,const int*,const double*,const int*,double*,const int*,int*);
    using Sytrf=void(*)(const char*,const int*,double*,const int*,int*,double*,const int*,int*);
    using Sytrs=void(*)(const char*,const int*,const int*,const double*,const int*,const int*,double*,const int*,int*);
    using Syrk=void(*)(const char*,const char*,const int*,const int*,const double*,
                       const double*,const int*,const double*,double*,const int*);
    Potrf potrf=nullptr;
    Potrs potrs=nullptr;
    Sytrf sytrf=nullptr;
    Sytrs sytrs=nullptr;
    Syrk syrk=nullptr;
    explicit Lapack(void* p=nullptr,void* s=nullptr,void* f=nullptr,void* t=nullptr,void* rk=nullptr)
      : potrf(reinterpret_cast<Potrf>(p)),potrs(reinterpret_cast<Potrs>(s)),
        sytrf(reinterpret_cast<Sytrf>(f)),sytrs(reinterpret_cast<Sytrs>(t)),
        syrk(reinterpret_cast<Syrk>(rk)) {}
    explicit operator bool() const {return potrf && potrs && sytrf && sytrs;}
};

inline bool min_degree(int n,const I& ptr,const I& row,I& perm,
                       Clock::time_point deadline) {
    std::vector<std::unordered_set<int>> adjacency(n);
    for (int j=0;j<n;++j) {
        auto& v=adjacency[j];
        v.reserve(ptr[j+1]-ptr[j]);
        for (int p=ptr[j];p<ptr[j+1];++p)
            if (row[p]!=j) v.insert(row[p]);
    }
    using Key=std::pair<int,int>;
    std::priority_queue<Key,std::vector<Key>,std::greater<Key>> q;
    for (int j=0;j<n;++j) q.emplace(adjacency[j].size(),j);
    I neighbors;
    long long cost=0;
    perm.assign(n,-1);
    for (int k=0;k<n;++k) {
        if ((k&127)==0 && Clock::now()>deadline) return false;
        while (!q.empty() && (perm[q.top().second]>=0 ||
            static_cast<int>(adjacency[q.top().second].size())!=q.top().first)) q.pop();
        if (q.empty()) return false;
        const auto [deg,v]=q.top(); q.pop();
        perm[v]=k;
        cost+=static_cast<long long>(deg)*deg;
        if (cost>11500000LL || deg>680) return false;
        neighbors.assign(adjacency[v].begin(),adjacency[v].end());
        std::sort(neighbors.begin(),neighbors.end());
        for (int a:neighbors) {
            auto& dest=adjacency[a];
            const auto before=dest.size();
            dest.erase(v);
            for (int b:neighbors)
                if (a!=b) dest.insert(b);
            if (dest.size()!=before) q.emplace(dest.size(),a);
        }
        adjacency[v].clear();
    }
    return true;
}

class NativeKkt {
    struct Trip {
        std::uint64_t pos;
        double val;
        int source; // fixed -1, condensed row >= 0
    };
    int n,m,N,retained=0;
    const I& ap;
    const I& ai;
    const V& av;
    const std::vector<unsigned char>& eq;
    I ret, ptr, row, diagonal, slot, src, piv, dslot, dsrc;
    I orow;
    V base,val,sv,scale,sign,values,rhs,work,rect,staticrect,X,C;
    bool dense=false,chol=false,gram=false,sparse_gram=false;
    Lapack lapack;
    std::unique_ptr<void,decltype(&conic_ldl_free)> ldl{nullptr,&conic_ldl_free};

    void set_dense(int N) {
        dense=true;
        chol=(retained==0);
        rect.resize(static_cast<std::size_t>(N)*N);
        piv.resize(N);
        work.resize(64*static_cast<std::size_t>(N));
    }

    bool dense_factor() {
        static constexpr char low='L';
        int info=0;
        if (chol) lapack.potrf(&low,&N,rect.data(),&N,&info);
        else {
            int lw=work.size();
            lapack.sytrf(&low,&N,rect.data(),&N,piv.data(),work.data(),&lw,&info);
        }
        return info==0;
    }

public:
    NativeKkt(int n,int m,const I& ap,const I& ai,const V& av,
              const I& pp,const I& pi,const V& pv,
              const std::vector<unsigned char>& eq,Lapack lapack,
              Clock::time_point deadline,const std::vector<unsigned char>* rotational=nullptr)
        : n(n),m(m),ap(ap),ai(ai),av(av),eq(eq),ret(m,-1),lapack(lapack) {
        const int ne=std::count(eq.begin(),eq.end(),1);
        const bool blasgram=(!rotational && lapack && lapack.syrk && n>=75 && n<=790 && n+ne<=1250
            && double(n)*(m-ne)<2750000 && ap[m]>.085*double(n)*m);
        if (!blasgram && lapack && 175<=n && n<=1310 && n+ne<=1420 && n+m>650
            && ap[m]>5*m && ne<.55*n) {
            int wide=0;
            long long pairs=0;
            for (int i=0;i<m;++i) if (!eq[i]) {
                const int sz=ap[i+1]-ap[i];
                pairs+=static_cast<long long>(sz)*sz;
                if (sz>=4 && ai[ap[i+1]-1]-ai[ap[i]]>.31*n) ++wide;
            }
            sparse_gram=(wide>n/6 && pairs<6400000);
        }
        if (blasgram || sparse_gram) {
            gram=true;
            for (int i=0;i<m;++i) {
                if (eq[i]) ret[i]=n+retained++;
                else if (ap[i+1]>ap[i]) orow.push_back(i);
            }
            N=n+retained;
            set_dense(N);
            staticrect.resize(static_cast<std::size_t>(N)*N);
            if (!sparse_gram) {
                X.resize(static_cast<std::size_t>(n)*orow.size());
                C.resize(X.size());
            }
            for (int j=0;j<n;++j)
                for (int k=pp[j];k<pp[j+1];++k) if (pi[k]>=j)
                    staticrect[static_cast<std::size_t>(j)*N+pi[k]]=pv[k];
            for (int i=0;i<m;++i) if (ret[i]>=0) {
                for (int j=ap[i];j<ap[i+1];++j)
                    staticrect[static_cast<std::size_t>(ai[j])*N+ret[i]]=av[j];
            }
            if (!sparse_gram)
                for (std::size_t j=0;j<orow.size();++j) {
                    const int i=orow[j];
                    for (int k=ap[i];k<ap[i+1];++k) X[j*n+ai[k]]=av[k];
                }
            scale.resize(N); rhs.resize(N); sv.resize(N);
            return;
        }
        const int maxrow=(n<420)?24:9;
        long long total=0;
        for (int r=0;r<m;++r) {
            const int d=ap[r+1]-ap[r];
            if (eq[r] || (rotational && (*rotational)[r]) || d>maxrow || total>680000)
                ret[r]=n+retained++;
            else total+=d*d;
        }
        N=n+retained;
        if (N>71000) throw std::runtime_error("oversized native Newton system");
        std::vector<Trip> t;
        t.reserve(total+pp[n]+2*ap[m]+N);
        auto put=[&](int i,int j,double v,int s) {
            t.push_back({std::uint64_t(j)*N+i,v,s});
        };
        for (int j=0;j<n;++j)
            for (int k=pp[j];k<pp[j+1];++k) put(pi[k],j,pv[k],-1);
        for (int j=0;j<N;++j) put(j,j,0.,-1);
        for (int r=0;r<m;++r) {
            if (ret[r]>=0) {
                const int id=ret[r];
                const bool rc=rotational && (*rotational)[r];
                for (int k=ap[r];k<ap[r+1];++k) {
                    put(ai[k],id,av[k],rc?-2-k:-1); put(id,ai[k],av[k],rc?-2-k:-1);
                }
            } else {
                for (int k=ap[r];k<ap[r+1];++k)
                    for (int h=ap[r];h<ap[r+1];++h) put(ai[k],ai[h],av[k]*av[h],r);
            }
        }
        if (Clock::now()>deadline || t.size()>3500000)
            throw std::runtime_error("native Newton construction limit");
        std::sort(t.begin(),t.end(),[](const Trip& a,const Trip& b){return a.pos<b.pos;});
        diagonal.resize(N);
        ptr.assign(N+1,0);
        std::uint64_t prior=UINT64_MAX;
        for (const Trip& a:t) {
            if (prior!=a.pos) {
                prior=a.pos;
                const int wc=a.pos/N, wr=a.pos%N;
                if (wr==wc) diagonal[wc]=row.size();
                ++ptr[wc+1]; row.push_back(wr); base.push_back(0.);
            }
            if (a.source==-1) base.back()+=a.val;
            else if (a.source<-1) {
                dslot.push_back(row.size()-1); dsrc.push_back(-2-a.source);
            }
            else {
                slot.push_back(row.size()-1); src.push_back(a.source); val.push_back(a.val);
            }
        }
        for (int j=0;j<N;++j) ptr[j+1]+=ptr[j];
        sign.resize(N);
        for (int j=0;j<N;++j) sign[j]=j<n?1.:-1.;
        I order;
        const double density=static_cast<double>(row.size())/(double(N)*N);
        if (lapack && N<=1160 && (N<=95 || density>.125)) set_dense(N);
        if (!dense && min_degree(N,ptr,row,order,deadline)) {
            ldl.reset(conic_ldl_create(N,ptr.data(),row.data(),order.data(),
                                      sign.data(),1650000,48000000.));
        }
        if (!dense && !ldl && lapack && N<1100) set_dense(N);
        if ((!dense && !ldl) || Clock::now()>deadline)
            throw std::runtime_error("native Newton fill limit");
        scale.resize(N); rhs.resize(N); sv.resize(N);
    }

    bool factor(const V& w,bool initial=false,double shift=0.) {
        if (gram) {
            rect=staticrect;
            if (initial)
                for (int i=0;i<m;++i) if (ret[i]>=0)
                    rect[static_cast<std::size_t>(ret[i])*N+ret[i]]=-1./w[i];
            if (sparse_gram) {
                for (int i:orow) {
                    const double f=w[i];
                    for (int k=ap[i];k<ap[i+1];++k) {
                        double* col=rect.data()+std::size_t(ai[k])*N;
                        const double a=f*av[k];
                        for (int t=k;t<ap[i+1];++t) col[ai[t]]+=a*av[t];
                    }
                }
            } else {
                for (std::size_t k=0;k<orow.size();++k) {
                    const double a=std::sqrt(w[orow[k]]);
                    for (int j=0;j<n;++j) C[k*n+j]=X[k*n+j]*a;
                }
                const char lower='L', no='N';
                const int r=orow.size();
                const double one=1.;
                lapack.syrk(&lower,&no,&n,&r,&one,C.data(),&n,&one,rect.data(),&N);
            }
            if (shift)
                for (int j=0;j<n;++j) rect[static_cast<std::size_t>(j)*N+j]+=shift;
            std::fill(scale.begin(),scale.end(),1.e-10);
            for (int j=0;j<N;++j)
                for (int i=j;i<N;++i) {
                    const double a=std::abs(rect[static_cast<std::size_t>(j)*N+i]);
                    scale[j]=std::max(scale[j],a); scale[i]=std::max(scale[i],a);
                }
            for (double& x:scale) x=1./std::sqrt(x);
            for (int j=0;j<N;++j) {
                double* b=rect.data()+static_cast<std::size_t>(j)*N;
                for (int i=j;i<N;++i) b[i]*=scale[j]*scale[i];
                b[j]+= (j<n?8.e-12:-8.e-12);
            }
            return dense_factor();
        }
        values=base;
        if (shift)
            for (int j=0;j<n;++j) values[diagonal[j]]+=shift;
        for (std::size_t k=0;k<dslot.size();++k) values[dslot[k]]+=av[dsrc[k]];
        for (std::size_t k=0;k<val.size();++k)
            values[slot[k]]+=val[k]*w[src[k]];
        for (int i=0;i<m;++i) if (ret[i]>=0 && (!eq[i] || initial))
            values[diagonal[ret[i]]]-=1./w[i];
        for (int j=0;j<N;++j) {
            double s=1.e-10;
            for (int k=ptr[j];k<ptr[j+1];++k) s=std::max(s,std::abs(values[k]));
            scale[j]=1./std::sqrt(s);
        }
        for (int j=0;j<N;++j) {
            const double c=scale[j];
            for (int k=ptr[j];k<ptr[j+1];++k) values[k]*=scale[row[k]]*c;
            values[diagonal[j]]+=8.e-12*sign[j];
        }
        if (!dense) {
            if (conic_ldl_factor(ldl.get(),values.data())) return true;
            for (int j=0;j<N;++j) values[diagonal[j]]+=7.e-10*sign[j];
            return conic_ldl_factor(ldl.get(),values.data());
        }
        std::fill(rect.begin(),rect.end(),0.);
        for (int j=0;j<N;++j)
            for (int k=ptr[j];k<ptr[j+1];++k)
                if (j<=row[k]) rect[static_cast<std::size_t>(j)*N+row[k]]=values[k];
        return dense_factor();
    }

    void solve(const V& w,const V& bx,const V& bz,V& x,V& z) {
        std::copy(bx.begin(),bx.end(),rhs.begin());
        std::fill(rhs.begin()+n,rhs.end(),0.);
        for (int i=0;i<m;++i) {
            if (ret[i]>=0) rhs[ret[i]]=bz[i];
            else {
                const double a=w[i]*bz[i];
                for (int k=ap[i];k<ap[i+1];++k) rhs[ai[k]]+=av[k]*a;
            }
        }
        for (int j=0;j<N;++j) rhs[j]*=scale[j];
        if (!dense) {
            conic_ldl_solve(ldl.get(),rhs.data(),sv.data());
        } else {
            sv=rhs;
            int info=0, one=1;
            static constexpr char low='L';
            if (chol) lapack.potrs(&low,&N,&one,rect.data(),&N,sv.data(),&N,&info);
            else lapack.sytrs(&low,&N,&one,rect.data(),&N,piv.data(),sv.data(),&N,&info);
        }
        for (int j=0;j<n;++j) x[j]=sv[j]*scale[j];
        for (int i=0;i<m;++i) {
            if (ret[i]>=0) z[i]=sv[ret[i]]*scale[ret[i]];
            else {
                long double r=-bz[i];
                for (int k=ap[i];k<ap[i+1];++k) r+=static_cast<long double>(av[k])*x[ai[k]];
                z[i]=r*w[i];
            }
        }
    }
};
}
