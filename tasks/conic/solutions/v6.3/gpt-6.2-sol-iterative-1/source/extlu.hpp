#pragma once
#include <cstdint>

namespace conic {

struct LUCallbacks {
    using Make=std::int64_t(*)(std::int64_t,int,const int*,const int*,const double*);
    using Solve=int(*)(std::int64_t,int,double*);
    using Free=void(*)(std::int64_t);
    Make make=nullptr;
    Solve solve=nullptr;
    Free release=nullptr;
    explicit operator bool()const{return make && solve && release;}
};
inline thread_local LUCallbacks external_lu;

}
