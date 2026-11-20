#include "npz.hpp"
#include <memory>
#include <sys/resource.h>

extern "C" {
struct json_t;
json_t* json_loadb(const char*,size_t,size_t,void*);
void json_delete(json_t*);
json_t* json_object_get(const json_t*,const char*);
size_t json_array_size(const json_t*);
json_t* json_array_get(const json_t*,size_t);
const char* json_string_value(const json_t*);
long long json_integer_value(const json_t*);

int conic_solve(int,int,const int*,const int*,const double*,const int*,const int*,
    const double*,const double*,const double*,int,const int*,const int*,double,int,double*,double*,double*);
int conic_highs(int,int,const int*,const int*,const double*,const int*,const int*,
    const double*,const double*,const double*,int,const int*,const int*,double,int,double*,double*,double*);
int sealed_highs_available();
}

namespace conic {
void read_cones(const Zip& zip,Int& kinds,Int& dims) {
    std::string m=Npy(zip.extract("metadata_json")).text();
    std::unique_ptr<json_t,decltype(&json_delete)> meta(json_loadb(m.data(),m.size(),0,nullptr),json_delete);
    require(bool(meta),"invalid input JSON");
    json_t* cones=json_object_get(meta.get(),"cones");
    require(cones!=nullptr,"missing cone list");
    size_t n=json_array_size(cones);
    kinds.reserve(n);dims.reserve(n);
    for(size_t j=0;j<n;++j) {
        json_t* c=json_array_get(cones,j);
        const char* label=json_string_value(json_object_get(c,"type"));
        require(label!=nullptr,"missing cone kind");
        const std::string_view kind(label);
        const long long d=json_integer_value(json_object_get(c,"dim"));
        require(d>0 && d<=std::numeric_limits<int>::max(),"invalid cone dimension");
        kinds.push_back(kind=="zero"?Zero:kind=="nonnegative"?Linear:
                        kind=="soc"?Soc:kind=="psd"?PSD:kind=="exponential"?Exp:-1);
        require(kinds.back()>=0,"unknown cone kind");
        dims.push_back(int(d));
    }
}

void python_fallback(const char* mainpath,const char* input,const char* output,
                     const char* mode="",double budget=16.) {
    std::string prog(mainpath), script;
    size_t slash=prog.rfind('/');
    script=(slash!=prog.npos?prog.substr(0,slash+1):"")+ "run.py";
    std::string time=std::to_string(std::max(.05,budget));
    execlp("python","python",script.c_str(),"--input",input,"--output",output,
           "--backend",mode,"--budget",time.c_str(),static_cast<char*>(nullptr));
}
}

int main(int argc,char** argv) {
    using namespace conic;
    const double start=clock_time();
    const char *in=nullptr,*out=nullptr;
    for(int j=1;j+1<argc;j+=2) {
        if(!std::strcmp(argv[j],"--input"))in=argv[j+1];
        else if(!std::strcmp(argv[j],"--output"))out=argv[j+1];
    }
    if(!in||!out)return 2;
    // Keep the candidate below the container's shared memory ceiling even
    // when a fill-heavy symbolic elimination requests an oversized factor.
    struct rlimit memory{};
    if(getrlimit(RLIMIT_AS,&memory)==0) {
        const rlim_t cap=rlim_t(3460)*1024*1024;
        if(memory.rlim_cur>cap) {
            memory.rlim_cur=cap;
            setrlimit(RLIMIT_AS,&memory);
        }
    }
    int status=Unresolved;
    Result r;
    try {
        Zip file(in);
        Int kinds,dims;
        read_cones(file,kinds,dims);
        const bool use_lp=std::all_of(kinds.begin(),kinds.end(),[](int k){return k<=Linear;});
        const bool use_exp=std::find(kinds.begin(),kinds.end(),Exp)!=kinds.end();
        const bool use_soc=std::find(kinds.begin(),kinds.end(),Soc)!=kinds.end();
        const bool use_psd=std::find(kinds.begin(),kinds.end(),PSD)!=kinds.end();
        const char* mode_env=std::getenv("CONIC_MODE");
        const bool python=std::getenv("CONIC_PY")!=nullptr;
        if(python)python_fallback(argv[0],in,out);
        if(std::getenv("CONIC_OUTER"))python_fallback(argv[0],in,out,"outer");
        // Looking at the Hessian payload alone avoids decoding the constraint
        // matrix twice on the linear route.
        const bool lp=use_lp && file.member_size("P_data") <= 512 &&
                       Npy(file.extract("P_data")).count==0;
        const bool high_ok=use_lp && !mode_env && !std::getenv("CONIC_HIGHS_PY") && sealed_highs_available();
        if(lp && !mode_env && !high_ok) {
            const Int shape=Npy(file.extract("A_shape")).ints();
            if(shape.size()==2 && file.member_size("A_data")>8*size_t(shape[0])+512)
                python_fallback(argv[0],in,out);
        }
        Vec q=Npy(file.extract("q")).floats(),b=Npy(file.extract("b")).floats();
        const int n=int(q.size()),m=int(b.size()),size=n+m;
        OwnedCSC P,A;
        P.read(file,"P",n,n);A.read(file,"A",m,n);
        r.x.resize(n);r.s.resize(m);r.z.resize(m);
        auto invoke=[&](bool high,int mode,double t) {
            double a=clock_time();
            auto method=high?conic_highs:conic_solve;
            int flag=method(n,m,P.p.data(),P.i.data(),P.a.data(),A.p.data(),A.i.data(),
                A.a.data(),q.data(),b.data(),int(kinds.size()),kinds.data(),dims.data(),t,mode,
                r.x.data(),r.s.data(),r.z.data());
            if(std::getenv("CONIC_TRACE"))
                std::fprintf(stderr,"variant %d status %d time %.4f\n",mode,flag,clock_time()-a);
            return flag;
        };
        auto solve=[&](int mode,double t) {return invoke(false,mode,t);};
        auto highs=[&](int mode,double t) {return invoke(true,mode,t);};
        if(mode_env)status=solve(std::atoi(mode_env),120.);
        else if(lp) {
            status=solve(-1,.5);
            const int engine=A.a.size()>20000 && size>5000 ? 1:0;
            if(status==Unresolved && high_ok) {
                status=highs(engine,std::max(.28,.82-(clock_time()-start)));
                if(status==Unresolved && clock_time()-start<.45)
                    status=solve(0,.22);
                if(status==Unresolved && clock_time()-start<.8)
                    status=highs(4+(1-engine),std::max(.12,.86-(clock_time()-start)));
            }
            if(status==Unresolved)python_fallback(argv[0],in,out,"",15.-(clock_time()-start));
        }
        else {
            const int first=use_soc&&!use_exp&&!use_psd?2:0;
            const double fast=std::max(.13,.74-(clock_time()-start));
            const bool equality_only=use_lp&&std::all_of(kinds.begin(),kinds.end(),[](int k){return k==Zero;});
            if(use_lp && !P.a.empty() && A.a.size()<=size_t(n)+m && P.a.size()<=size_t(n))
                status=solve(22,.28);
            if(status==Unresolved && use_lp && !P.a.empty() && P.a.size()>size_t(n) && A.a.size()<=size_t(m))
                status=solve(23,.27);
            if(status==Unresolved && equality_only)status=solve(14,fast);
            if(status==Unresolved)status=solve(first, size>24000?10.:fast);
            if(status==Unresolved && clock_time()-start<2.) {
                if(use_exp)status=solve(17,.50);
                else if(first!=0)status=solve(15,.55);
                else status=solve(2,.55);
            }
            if(status==Unresolved && use_lp && P.a.size() && clock_time()-start<5.)
                status=solve(13,std::max(.18,.88-(clock_time()-start)));
            if(status==Unresolved && high_ok && clock_time()-start<9.)
                status=highs(0,std::max(.2,.87-(clock_time()-start)));
            if(status==Unresolved && use_lp)
                python_fallback(argv[0],in,out,"quadratic",13.8-(clock_time()-start));
            if(status==Unresolved && (use_soc||use_exp) && P.a.empty() &&
               size<57000 && clock_time()-start<6.)
                status=solve(27,std::max(.17,.87-(clock_time()-start)));
            if(status==Unresolved && (use_soc || use_exp) && size<70000 && clock_time()-start<4.5)
                status=solve(24,std::max(.18,.87-(clock_time()-start)));
            if(status==Unresolved && use_exp && clock_time()-start<7.)
                status=solve(1,2.5);
            if(status==Unresolved && use_soc && clock_time()-start<7.)
                status=solve(18,2.5);
            if(status==Unresolved && clock_time()-start<12.)
                status=solve(3,6.);
            if(status==Unresolved && use_psd && m<15000 && clock_time()-start<12.)
                status=solve(4,5.);
            if(status==Unresolved && use_soc && size<6000 && clock_time()-start<12.)
                status=solve(16,2.);
            if(status==Unresolved && (use_soc||use_exp||use_psd) && size<12000 && clock_time()-start<7.)
                python_fallback(argv[0],in,out,"pivoted",13.9-(clock_time()-start));
            if(status==Unresolved && (use_soc||use_exp||use_psd) && size<58000 && clock_time()-start<13.)
                python_fallback(argv[0],in,out,"outer",14.4-(clock_time()-start));
            if(status==Unresolved && size<=24000 && clock_time()-start>=.65 && clock_time()-start<12.)
                status=solve(first,std::max(.1,13.8-(clock_time()-start)));
        }
    } catch(const std::exception& e) {
        if(std::getenv("CONIC_TRACE"))std::fprintf(stderr,"input/solver: %s\n",e.what());
        status=Unresolved;
    }
    try {
        NpzOutput result(out);
        const char* labels[]={"unresolved","solved","primal_infeasible","dual_infeasible"};
        result.metadata(labels[status]);
        if(status==Solved || status==DualInfeasible) {
            result.vec("x",r.x);result.vec("s",r.s);
        }
        if(status==Solved || status==PrimalInfeasible)result.vec("z",r.z);
        result.finish();
    } catch(const std::exception& e) {
        std::fprintf(stderr,"output: %s\n",e.what());return 2;
    }
    return 0;
}
