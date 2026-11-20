#include "numeric.hpp"
#include "deps_autogen.hpp"

#include <cstring>
#include <dlfcn.h>
#include <link.h>
#include <numeric>
#include <type_traits>

namespace {
class DynamicModule {
    void* handle=nullptr;
    uintptr_t base=0;
public:
    explicit DynamicModule(const char* path){
        if(!*path) return;
        handle=dlopen(path,RTLD_LAZY|RTLD_LOCAL);
        link_map* map=nullptr;
        if(handle && dlinfo(handle,RTLD_DI_LINKMAP,&map)==0 && map) base=map->l_addr;
    }
    template<class T> T function(unsigned pos) const {
        return base&&pos?reinterpret_cast<T>(base+pos):nullptr;
    }
};
const DynamicModule& slu(){
    static DynamicModule m(installed::superlu);
    return m;
}
const DynamicModule& hs(){
    static DynamicModule m(installed::highs);
    return m;
}
template<class T> void hi_option(void* p,const char* name,T value,unsigned off) {
    // This translation unit uses the libstdc++ string ABI of the SciPy wheel.
    if(auto f=hs().function<int(*)(void*,const std::string&,T)>(off))
        f(p,std::string(name),value);
}
}

Ind order_mmd(const Sparse& a,int variant) {
    int n=a.cols;
    Ind perm(n);
    std::iota(perm.begin(),perm.end(),0);
    if(n<2 || a.nnz()==n) return perm;
    using Mmd=void(*)(int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,int*);
    auto mmd=slu().function<Mmd>(installed::sl_genmmd_);
    Ind p(n+1), r(2LL*(a.nnz()-n));
    for(int j=0;j<n;++j) for(int l=a.p[j];l<a.p[j+1];++l){
        int i=a.i[l];
        if(i!=j) {++p[j+1]; ++p[i+1];}
    }
    for(int j=0;j<n;++j) p[j+1]+=p[j];
    {
        Ind w=p;
        for(int j=0;j<n;++j) for(int l=a.p[j];l<a.p[j+1];++l) {
            int i=a.i[l];
            if(i!=j) {r[w[j]++]=i;r[w[i]++]=j;}
        }
    }
    if(variant==1){
        using Rec=size_t(*)(int,int,int);
        using Amd=int(*)(int,int,int,int*,int*,double*,int*);
        auto rec=slu().function<Rec>(installed::sl_colamd_recommended);
        auto amd=slu().function<Amd>(installed::sl_colamd);
        if(rec && amd) {
            size_t s=rec((int)r.size(),n,n);
            if(s>r.size() && s<INT32_MAX && s<65000000){
                r.resize(s);
                int stats[20]{};
                if(amd(n,n,s,r.data(),p.data(),nullptr,stats)){
                    Ind seen(n,-1);
                    for(int j=0;j<n;++j) {
                        if(p[j]<0||p[j]>=n||seen[p[j]]>=0)return perm;
                        seen[p[j]]=j;
                    }
                    perm.swap(seen);
                    return perm;
                }
            }
            return perm;
        }
    }
    if(mmd) {
        for(int& x:p) ++x;
        for(int& x:r) ++x;
        Ind w(5LL*n);
        int delta=variant==2?1:0, maxint=INT32_MAX, cnt=0;
        mmd(&n,p.data(),r.data(),perm.data(),w.data(),&delta,w.data()+n,w.data()+2*n,
            w.data()+3*n,w.data()+4*n,&maxint,&cnt);
        bool valid=true;
        std::fill(w.begin(),w.begin()+n,0);
        for(int& x:perm) {
            --x;
            if(x<0 || x>=n || w[x]++) {valid=false;break;}
        }
        if(!valid)std::iota(perm.begin(),perm.end(),0);
        return perm;
    }
    // Reverse Cuthill-McKee for installations without a linked orderer.
    Ind deg(n),que; que.reserve(n);
    for(int j=0;j<n;++j) deg[j]=p[j+1]-p[j];
    Ind rank=perm; std::fill(perm.begin(),perm.end(),-1);
    std::sort(rank.begin(),rank.end(),[&](int u,int v){return deg[u]<deg[v];});
    for(int root:rank) if(perm[root]<0) {
        size_t z=que.size();
        que.push_back(root);perm[root]=0;
        for(;z<que.size();++z){
            int j=que[z];size_t tail=que.size();
            for(int k=p[j];k<p[j+1];++k) if(perm[r[k]]<0) {perm[r[k]]=0;que.push_back(r[k]);}
            std::sort(que.begin()+tail,que.end(),[&](int u,int v){return deg[u]<deg[v];});
        }
    }
    for(int j=0;j<n;++j) perm[que[n-1-j]]=j;
    return perm;
}

HiGHS::HiGHS() {
    auto f=hs().function<void(*)(void*)>(installed::hi_constructor);
    active=f && installed::hi_destructor && installed::hi_run && installed::hi_pass_lp &&
        installed::hi_status_member && installed::hi_get_solution && installed::hi_set_b;
    if(active){
        std::memset(instance,0,sizeof(instance));
        f(instance);
        hi_option(instance,"output_flag",false,installed::hi_set_b);
        hi_option(instance,"threads",1,installed::hi_set_i);
    }
}
HiGHS::~HiGHS() {
    if(active) hs().function<void(*)(void*)>(installed::hi_destructor)(instance);
}
LinearStatus HiGHS::status() const {
    int value=0;
    if(active) std::memcpy(&value,instance+installed::hi_status_member,sizeof(value));
    return (LinearStatus)value;
}
LinearStatus HiGHS::solve(const Sparse& a,const Vec& q,const Vec& cl,const Vec& cu,const Vec& rl,const Vec& ru,
                          const Sparse* p,const char* alg,bool presolve,double limit,double feasibility) {
    if(!active) return LpUnknown;
    hi_option(instance,"solver",alg,installed::hi_set_s);
    hi_option(instance,"presolve",presolve?"on":"off",installed::hi_set_s);
    hi_option(instance,"parallel","off",installed::hi_set_s);
    hi_option(instance,"primal_feasibility_tolerance",feasibility,installed::hi_set_d);
    hi_option(instance,"dual_feasibility_tolerance",feasibility,installed::hi_set_d);
    hi_option(instance,"ipm_optimality_tolerance",1e-11,installed::hi_set_d);
    hi_option(instance,"small_matrix_value",1e-12,installed::hi_set_d);
    hi_option(instance,"large_matrix_value",1e20,installed::hi_set_d);
    hi_option(instance,"infinite_bound",1e30,installed::hi_set_d);
    if(limit>0) hi_option(instance,"time_limit",limit,installed::hi_set_d);
    int st=-1;
    if(!p || !p->nnz()){
        using Fn=int(*)(void*,int,int,int,int,int,double,const double*,const double*,const double*,
                       const double*,const double*,const int*,const int*,const double*,const int*);
        st=hs().function<Fn>(installed::hi_pass_lp)(instance,a.cols,a.rows,a.nnz(),1,1,0,q.data(),cl.data(),
                cu.data(),rl.data(),ru.data(),a.p.data(),a.i.data(),a.x.data(),nullptr);
    } else if(auto fn=hs().function<int(*)(void*,int,int,int,int,int,int,int,double,const double*,const double*,
                    const double*,const double*,const double*,const int*,const int*,const double*,
                    const int*,const int*,const double*,const int*)>(installed::hi_pass_qp)){
        Sparse tri(p->rows,p->cols);
        tri.i.reserve((p->nnz()+p->cols)/2);tri.x.reserve(tri.i.capacity());
        for(int j=0;j<p->cols;++j) {
            for(int k=p->p[j];k<p->p[j+1];++k) if(p->i[k]>=j) {
                tri.i.push_back(p->i[k]);tri.x.push_back(p->x[k]);
            }
            tri.p[j+1]=tri.nnz();
        }
        st=fn(instance,a.cols,a.rows,a.nnz(),tri.nnz(),1,1,1,0,q.data(),cl.data(),cu.data(),rl.data(),ru.data(),
              a.p.data(),a.i.data(),a.x.data(),tri.p.data(),tri.i.data(),tri.x.data(),nullptr);
    }
    if(st<0) return LpUnknown;
    hs().function<int(*)(void*)>(installed::hi_run)(instance);
    return status();
}
bool HiGHS::point(Vec& x,Vec& z,Vec* zd) const {
    if(!active) return false;
    struct Solution {
        bool value_valid,dual_valid;
        Vec col_value,col_dual,row_value,row_dual;
    };
    const auto* s=hs().function<const Solution*(*)(const void*)>(installed::hi_get_solution)(instance);
    if(!s || !s->value_valid) return false;
    x=s->col_value;
    if(s->dual_valid) {
        z.resize(s->row_dual.size());
        for(size_t j=0;j<z.size();++j) z[j]=-s->row_dual[j];
        if(zd) *zd=s->col_dual;
    } else {
        z.clear();
        if(zd) zd->clear();
    }
    return true;
}
bool HiGHS::ray(Vec& z,bool primal) {
    if(!active || z.empty()) return false;
    auto f=hs().function<int(*)(void*,bool&,double*)>(primal?installed::hi_get_primal_ray:installed::hi_get_dual_ray);
    if(!f) return false;
    bool exists=false;
    int st=f(instance,exists,z.data());
    return st>=0 && exists;
}
bool HiGHS::append_rows(const Sparse& csr,const Vec& lo,const Vec& hi) {
    auto f=hs().function<int(*)(void*,int,const double*,const double*,int,const int*,
                             const int*,const double*)>(installed::hi_add_rows);
    return active && f && f(instance,csr.cols,lo.data(),hi.data(),csr.nnz(),csr.p.data(),csr.i.data(),csr.x.data())>=0;
}
LinearStatus HiGHS::resolve(){
    if(!active)return LpUnknown;
    hs().function<int(*)(void*)>(installed::hi_run)(instance);
    return status();
}
