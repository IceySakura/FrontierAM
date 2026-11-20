#include "common.hpp"
#include <dlfcn.h>
#include <link.h>
#include <new>
#include <string>

// Compile this translation unit with the old libstdc++ string ABI used by the
// pinned SciPy wheel. Activate only after confirming its exact fingerprint.
#include "highs_symbols.hpp"

namespace {
#if SEALED_HIGHS

struct HighsSolution {
    bool value_valid,dual_valid;
    conic::Vec col_value,col_dual,row_value,row_dual;
};

struct Library {
    uintptr_t base=0;
    Library() {
        if(!dlopen(libpython,RTLD_NOW|RTLD_GLOBAL))return;
        void* dl=dlopen(wheel,RTLD_NOW|RTLD_LOCAL);
        if(!dl)return;
        link_map* map=nullptr;
        if(dlinfo(dl,RTLD_DI_LINKMAP,&map)==0 && map)base=map->l_addr;
    }
    template<class F> F f(size_t offset) const {
        return reinterpret_cast<F>(base+offset);
    }
};

Library& library(){ static Library lib; return lib;}

class Hi {
    const Library& lib;
public:
    void* self=nullptr;
    explicit Hi(const Library& l):lib(l) {
        if(lib.base) {
            self=::operator new(65536,std::align_val_t(64));
            try {lib.f<void(*)(void*)>(ofs_constructor)(self);}
            catch (...) {::operator delete(self,std::align_val_t(64));self=nullptr;}
        }
    }
    ~Hi(){
        if(self) {
            lib.f<void(*)(void*)>(ofs_destructor)(self);
            ::operator delete(self,std::align_val_t(64));
        }
    }
    void option(const std::string& name,bool value){
        lib.f<int(*)(void*,const std::string&,bool)>(ofs_set_b)(self,name,value);
    }
    void option(const std::string& name,int value){
        lib.f<int(*)(void*,const std::string&,int)>(ofs_set_i)(self,name,value);
    }
    void option(const std::string& name,double value){
        lib.f<int(*)(void*,const std::string&,double)>(ofs_set_d)(self,name,value);
    }
    void option(const std::string& name,const char* value){
        lib.f<int(*)(void*,const std::string&,const char*)>(ofs_set_s)(self,name,value);
    }
    int pass(int n,int m,const int* ap,const int* ai,const double* av,
                 const double* q,const double* lb,const double* ub,const double* l,const double* u,
                 bool rowwise=false) {
        using F=int(*)(void*,int,int,int,int,int,double,const double*,const double*,
            const double*,const double*,const double*,const int*,const int*,const double*,const int*);
        return lib.f<F>(ofs_pass_lp)(self,n,m,ap[rowwise?m:n],rowwise?2:1,1,0.,q,lb,ub,l,u,ap,ai,av,nullptr);
    }
    int hessian(int n,const int* pp,const int* pi,const double* pv){
        return lib.f<int(*)(void*,int,int,int,const int*,const int*,const double*)>(ofs_pass_p)
            (self,n,pp[n],1,pp,pi,pv);
    }
    int run(){
        lib.f<int(*)(void*)>(ofs_run)(self);
        return lib.f<int(*)(void*)>(ofs_status)(self);
    }
    const HighsSolution& sol(){
        return lib.f<const HighsSolution&(*)(const void*)>(ofs_solution)(self);
    }
    bool ray(bool dual,double* out) {
        bool exists=false;
        auto h=lib.f<int(*)(void*,bool&,double*)>(dual?ofs_dual_ray:ofs_primal_ray)(self,exists,out);
        return h>=0 && exists;
    }
    int add(int m,const int* p,const int* i,const double* a,const double* l,const double* u) {
        return lib.f<int(*)(void*,int,const double*,const double*,int,const int*,const int*,const double*)>(ofs_add_rows)
            (self,m,l,u,p[m],p,i,a);
    }
    int bound(int n,const double* lo,const double* hi) {
        return lib.f<int(*)(void*,int,int,const double*,const double*)>(ofs_change_bounds)
            (self,0,n-1,lo,hi);
    }
};
#endif
}

extern "C" void* sealed_cut_open(int n,int m,const int* p,const int* i,const double* a,
        const double* q,const double* lb,const double* ub,const double* l,const double* u) {
#if SEALED_HIGHS
    try {
        auto* h=new Hi(library());
        if(!h->self) {delete h;return nullptr;}
        h->option("output_flag",false);h->option("threads",1);
        h->option("parallel","off");h->option("solver","simplex");
        h->option("primal_feasibility_tolerance",1e-10);
        h->option("dual_feasibility_tolerance",1e-10);
        h->option("small_matrix_value",1e-12);
        h->option("large_matrix_value",1e20);
        h->option("infinite_bound",1e25);
        if(h->pass(n,m,p,i,a,q,lb,ub,l,u,true)<0) {delete h;return nullptr;}
        return h;
    } catch (...) {return nullptr;}
#else
    return nullptr;
#endif
}

extern "C" void sealed_cut_close(void* h) {
#if SEALED_HIGHS
    delete static_cast<Hi*>(h);
#endif
}

extern "C" int sealed_cut_append(void* h,int m,const int* p,const int* i,const double* a,
        const double* lo,const double* hi) {
#if SEALED_HIGHS
    try {return static_cast<Hi*>(h)->add(m,p,i,a,lo,hi);}catch(...){return -1;}
#else
    return -1;
#endif
}

extern "C" int sealed_cut_bounds(void* h,int n,const double* lo,const double* hi) {
#if SEALED_HIGHS
    try {return static_cast<Hi*>(h)->bound(n,lo,hi);}catch(...){return -1;}
#else
    return -1;
#endif
}

extern "C" int sealed_cut_step(void* handle,int n,int m,double budget,double* x,double* z) {
#if SEALED_HIGHS
    try {
        auto& h=*static_cast<Hi*>(handle);
        h.option("time_limit",std::max(.015,budget));
        const double start=conic::clock_time();
        int st=h.run();
        if(st==7 || st==13 || st==14 || st==6) {
            auto& s=h.sol();
            if(s.value_valid && s.col_value.size()==size_t(n) && s.dual_valid &&
                 s.row_dual.size()==size_t(m)) {
                std::copy(s.col_value.begin(),s.col_value.end(),x);
                for(int j=0;j<m;++j)z[j]=-s.row_dual[j];
                return 1;
            }
        } else if(conic::clock_time()-start<budget-.01) {
            h.option("time_limit",std::max(.015,budget-(conic::clock_time()-start)));
            if((st==8 || st==9) && h.ray(true,z)) {
                for(int j=0;j<m;++j)z[j]=-z[j];
                return 2;
            }
            if(st==10 && h.ray(false,x))return 3;
            return st==8?4:st==10?5:0;
        }
    } catch (...) {}
#endif
    return 0;
}

extern "C" int sealed_highs_available() {
#if SEALED_HIGHS
    return library().base!=0;
#else
    return 0;
#endif
}

extern "C" int sealed_highs_solve(int n,int m,const int* pp,const int* pi,const double* pv,
    const int* ap,const int* ai,const double* av,const double* q,const double* lb,
    const double* ub,const double* l,const double* u,int engine,double budget,
    double* x,double* z,double* cd,int* flags) {
    *flags=0;
#if SEALED_HIGHS
    try {
        if(n<=0)return -999;
        const double start=conic::clock_time();
        Hi h(library());
        if(!h.self)return -999;
        h.option("output_flag",false);
        h.option("threads",1);
        h.option("parallel","off");
        h.option("time_limit",std::max(.025,budget));
        h.option("primal_feasibility_tolerance",1e-10);
        h.option("dual_feasibility_tolerance",1e-10);
        h.option("ipm_optimality_tolerance",1e-11);
        h.option("small_matrix_value",1e-12);
        h.option("large_matrix_value",1e20);
        h.option("infinite_bound",1e25);
        h.option("infinite_cost",1e25);
        const bool qp=pp[n]!=0;
        if(!qp) {
            h.option("solver",engine==1?"ipm":"simplex");
            if(engine==1)h.option("run_crossover","off");
        }
        if(h.pass(n,m,ap,ai,av,q,lb,ub,l,u)<0)return 2;
        if(qp && h.hessian(n,pp,pi,pv)<0)return 2;
        int status=h.run();
        if(status==6 || status==7 || status==13 || status==14) {
            const HighsSolution& sol=h.sol();
            if(sol.value_valid && sol.col_value.size()==size_t(n)) {
                std::copy(sol.col_value.begin(),sol.col_value.end(),x);*flags|=1;
            }
            if(sol.dual_valid && sol.row_dual.size()==size_t(m)) {
                for(int i=0;i<m;++i)z[i]=-sol.row_dual[i];
                *flags|=2;
            }
            if(cd && sol.dual_valid && sol.col_dual.size()==size_t(n)) {
                std::copy(sol.col_dual.begin(),sol.col_dual.end(),cd);*flags|=16;
            }
        } else if(!qp && (status==8 || status==9 || status==10) &&
                  conic::clock_time()-start<budget-.015) {
            h.option("time_limit",std::max(.025,budget-(conic::clock_time()-start)));
            if(m && (status==8 || status==9) && h.ray(true,z)) {
                for(int j=0;j<m;++j)z[j]=-z[j];
                *flags|=8;
            }
            if(status==10 && h.ray(false,x))*flags|=4;
        }
        return status;
    } catch (...) {return -999;}
#else
    return -999;
#endif
}
