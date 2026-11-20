#pragma once
#include "cone.hpp"
#include <memory>

struct Verifier {
    const Model& p;
    double nB,nQ,aScale,pScale;
    std::unique_ptr<ProductCone> cone;
    Result result,near;
    double best=INFINITY;
    explicit Verifier(const Model&);
    ProductCone& cones();
    bool solved(const Vec& x,const Vec& s,const Vec& z,bool repair=true);
    bool primal(const Vec& z);
    bool dual(const Vec& x,const Vec* s=nullptr);
};

bool ipm(const Model&,Verifier&,int);
bool linear_solver(const Model&,Verifier&,bool);
bool supporting_planes(const Model&,Verifier&);
bool operator_split(const Model&,ProductCone&,const Vec&,const Vec&,double,Verifier&,int);
bool polish_manifold(const Model&,const Vec&,const Vec&,double,Verifier&);
bool diagonal_sdp(const Model&,Verifier&);
bool projection_form(const Model&,Verifier&);
bool box_quadratic(const Model&,Verifier&);
bool certificate(const Model&,Verifier&,bool);
bool logsum_form(const Model&,Verifier&);
Result optimize(const Model&);
