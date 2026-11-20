#pragma once
#include "model.hpp"

Ind order_mmd(const Sparse& a,int variant);

enum LinearStatus {
    LpUnknown=0, LpEmpty=6, LpOptimal=7, LpInfeasible=8, LpAmbiguous=9,
    LpUnbounded=10, LpTimeLimit=13
};

class HiGHS {
    alignas(64) unsigned char instance[65536];
    bool active=false;
public:
    HiGHS();
    ~HiGHS();
    HiGHS(const HiGHS&)=delete;
    HiGHS& operator=(const HiGHS&)=delete;
    explicit operator bool() const {return active;}
    LinearStatus solve(const Sparse& a,const Vec& q,const Vec& cl,const Vec& cu,const Vec& rl,const Vec& ru,
                       const Sparse* p=nullptr,const char* alg="choose",bool presolve=true,
                       double limit=0,double feasibility=1e-9);
    bool point(Vec& x,Vec& z,Vec* zd=nullptr) const;
    bool ray(Vec& z,bool primal);
    bool append_rows(const Sparse& csr,const Vec& low,const Vec& high);
    LinearStatus resolve();
    LinearStatus status() const;
};
