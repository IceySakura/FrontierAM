#pragma once
#include "model.hpp"

extern "C" {
void* cc_create(int,const int*,const int*,const int*);
void cc_free(void*);
void cc_init(void*,double*,double*);
void cc_shift(void*,double*,int);
int cc_prepare(void*,const double*,const double*,double*,double*,double*,double,int);
void cc_correct(void*,const double*,const double*,const double*,const double*,double,double,double*);
double cc_step(void*,const double*,const double*,int,double);
void cc_psd_mul(void*,int,const double*,double*,int);
void cc_psd_schur(void*,int,int,const int*,const int*,const double*,double*);
double cc_defect(void*,const double*,int,double);
void cc_rowscale(void*,double*,int);
void cc_project(void*,double*,int);
void cc_project_pair(void*,double*,double*);
int cc_constant(int,int,const double*,double*,double);
}
struct ProductCone {
    void* handle=nullptr;
    Ind start,modes;
    Vec diagonal,off,aux;
    int degree=0, nexp=0;
    ProductCone(const Model& model,Ind variant,bool verifier=false) : modes(std::move(variant)) {
        start.reserve(model.cones.size()+1);start.push_back(0);
        Ind kind,dim;
        kind.reserve(model.cones.size());dim.reserve(model.cones.size());
        size_t na=0,no=0;
        for(size_t j=0;j<model.cones.size();++j){
            auto c=model.cones[j];
            int w=c.width();kind.push_back(c.kind);dim.push_back(c.dim);start.push_back(start.back()+w);
            degree+=c.kind==Orthant || c.kind==Semidefinite?c.dim:c.kind==Lorentz?1:c.kind==Exponential?3:0;
            nexp+=c.kind==Exponential;
            if(modes[j]==4){na+=3;no+=9;}
            else if(modes[j]==3){na+=2;no+=2*w-1;}
            else if(modes[j]==1){++na;no+=w;}
            else if(modes[j]==0 && c.kind>=Lorentz) no+=w*(w-1LL)/2;
        }
        if(!verifier){
            if(no>40000000) throw std::bad_alloc();
            diagonal.resize(model.m());
            off.resize(no);aux.resize(na);
        }
        handle=cc_create(kind.size(),kind.data(),dim.data(),modes.data());
        if(!handle) throw std::bad_alloc();
    }
    explicit ProductCone(const Model& p) : ProductCone(p,Ind(p.cones.size(),5),true) {}
    ProductCone(const ProductCone&)=delete;
    ProductCone& operator=(const ProductCone&)=delete;
    ~ProductCone(){cc_free(handle);}
    bool prepare(const Vec& s,const Vec& z,double mu,bool pde) {
        return cc_prepare(handle,s.data(),z.data(),diagonal.data(),off.data(),aux.data(),mu,pde);
    }
    void initial(Vec& s,Vec& z) {cc_init(handle,s.data(),z.data());}
    void shift(Vec& x,bool dual) {cc_shift(handle,x.data(),dual);}
    void correct(const Vec& s,const Vec& z,const Vec& ds,const Vec& dz,double mu,double w,Vec& out) {
        cc_correct(handle,s.data(),z.data(),ds.data(),dz.data(),mu,w,out.data());
    }
    double step(const Vec& s,const Vec& ds,bool dual,double lim) {
        return lim>0 ? cc_step(handle,s.data(),ds.data(),dual,lim) : 0;
    }
    double defect(const Vec& s,bool dual,double lim){
        return cc_defect(handle,s.data(),dual,lim);
    }
    void project(Vec& s,bool dual){cc_project(handle,s.data(),dual);}
    void project_pair(Vec& s,Vec& z){cc_project_pair(handle,s.data(),z.data());}
};
