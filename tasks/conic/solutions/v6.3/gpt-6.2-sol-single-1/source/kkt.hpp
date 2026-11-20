#pragma once
#include "cone.hpp"
#include <memory>

class Newton {
    struct System;
    std::unique_ptr<System> sys;
public:
    Newton(const Model&,ProductCone&,int);
    ~Newton();
    bool factor(double reg);
    bool solve(const Vec& xrhs,const Vec& zrhs,Vec& x,Vec& z);
    long long fill() const;
};
