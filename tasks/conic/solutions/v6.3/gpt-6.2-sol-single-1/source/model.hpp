#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Vec = std::vector<double>;
using Ind = std::vector<int>;
enum ConeKind { Zero, Orthant, Lorentz, Semidefinite, Exponential };
struct ConeSpec {
    int kind, dim;
    int width() const { return kind == Semidefinite ? dim*(dim+1LL)/2 : dim; }
};

struct Sparse {
    int rows=0, cols=0;
    Ind p, i;
    Vec x;
    Sparse() = default;
    Sparse(int m,int n) : rows(m),cols(n),p(n+1) {}
    int nnz() const {return (int)x.size();}
    void compact();
    Sparse transpose() const;
    void mul(const double* a,double* y) const;
    void tmul(const double* a,double* y) const;
    Vec mul(const Vec& a) const { Vec y(rows); mul(a.data(),y.data()); return y; }
    Vec tmul(const Vec& a) const { Vec y(cols); tmul(a.data(),y.data()); return y; }
    void scale(const Vec& c,const Vec* r);
    void norms(Vec& c,Vec* r) const;
};
Sparse symmetric_part(const Sparse& p);

struct Model {
    Sparse P,A;
    Vec q,b;
    std::vector<ConeSpec> cones;
    double offset=0;
    int n() const {return (int)q.size();}
    int m() const {return (int)b.size();}
};
enum ResultStatus { Unresolved, Solved, PrimalInfeasible, DualInfeasible };
struct Result {
    ResultStatus status=Unresolved;
    Vec x,s,z;
    explicit operator bool() const { return status!=Unresolved; }
};
Model read_npz(const char* path);
void write_npz(const char* path,const Result& r);

extern "C" double scipy_ddot_(const int*,const double*,const int*,const double*,const int*);
inline double dot(const Vec& a,const Vec& b) {
    int n=(int)a.size(),one=1;
    return n==0?0:scipy_ddot_(&n,a.data(),&one,b.data(),&one);
}
inline double norm(const Vec& a) {
    double mx=0;
    for(double v:a) {
        if(!std::isfinite(v)) return INFINITY;
        mx=std::max(mx,std::abs(v));
    }
    return mx;
}
inline double fastnorm(const Vec& a) {
    double mx=0;
    for(double v:a) mx=std::max(mx,std::abs(v));
    return mx;
}
extern bool debug;
