#include "optimizer.hpp"

#include <cstdio>

extern "C" void scipy_dgelsy_(const int*,const int*,const int*,double*,const int*,double*,const int*,
                              int*,const double*,int*,double*,const int*,int*);

namespace {
Vec residual(const Sparse& A,const Sparse* B,const Vec& rhs,const Vec* metric,double cutoff) {
    int m=A.rows,n=A.cols+(B?B->cols:0);
    if(!m || !n || (long long)m*n>2'400'000 || std::min(m,n)>3300)return {};
    Vec w((size_t)m*n);
    auto copy=[&](const Sparse& a,int at){
        for(int j=0;j<a.cols;++j){
            auto v=w.data()+(size_t)(j+at)*m;
            double c=0;
            for(int k=a.p[j];k<a.p[j+1];++k){
                double h=a.x[k];int i=a.i[k];
                if(metric)h*=(*metric)[i];
                v[i]=h;c=std::max(c,std::abs(h));
            }
            if(c>0)for(int k=a.p[j];k<a.p[j+1];++k)v[a.i[k]]/=c;
        }
    };
    copy(A,0);if(B)copy(*B,A.cols);
    Vec b(std::max(m,n));
    for(int j=0;j<m;++j)b[j]=rhs[j]*(metric?(*metric)[j]:1);
    Ind perm(n);
    int one=1,ldb=b.size(),rank=0,info=0,lwork=-1;
    double k=0;
    scipy_dgelsy_(&m,&n,&one,w.data(),&m,b.data(),&ldb,perm.data(),&cutoff,&rank,&k,&lwork,&info);
    if(info || k<1 || k>10000000)return {};
    lwork=int(k);
    Vec work(lwork);
    scipy_dgelsy_(&m,&n,&one,w.data(),&m,b.data(),&ldb,perm.data(),&cutoff,&rank,work.data(),&lwork,&info);
    if(info || rank>=m)return {};
    Vec r(m);
    for(int j=0;j<m;++j)r[j]=rhs[j]*(metric?(*metric)[j]:1);
    auto rem=[&](const Sparse& a,int at){
        for(int j=0;j<a.cols;++j){
            double c=0;
            for(int k=a.p[j];k<a.p[j+1];++k){
                int i=a.i[k];
                c=std::max(c,std::abs(a.x[k])*(metric?(*metric)[i]:1));
            }
            if(c>0)for(int k=a.p[j];k<a.p[j+1];++k){
                int i=a.i[k];
                r[i]-=(a.x[k]*(metric?(*metric)[i]:1)/c)*b[j+at];
            }
        }
    };
    rem(A,0);if(B)rem(*B,A.cols);
    if(metric)for(int j=0;j<m;++j)r[j]*=(*metric)[j];
    return r;
}

Vec scale_rows(const Sparse& A,const Sparse* B=nullptr) {
    Vec r(A.rows),c(A.cols);
    A.norms(c,&r);
    if(B){ c.resize(B->cols); B->norms(c,&r); }
    for(double& x:r)x=1/std::sqrt(std::clamp(x,1e-14,1e14));
    return r;
}

Model primal_problem(const Model& p,const Verifier& v) {
    const int n=p.n(),m=p.m();
    Model t;
    Sparse rows=p.A;
    for(double& a:rows.x)a/=v.aScale;
    for(int i=0;i<m;++i)if(p.b[i]){rows.i.push_back(i);rows.x.push_back(p.b[i]/v.nB);}
    rows.p.push_back(rows.nnz());++rows.cols;
    t.cones.push_back({Zero,n+1});t.b.resize(n+1);t.b.back()=-1;
    int j=0;
    auto term=[&](int i,double a){
        rows.i.push_back(i);rows.x.push_back(a);
        rows.p.push_back(rows.nnz());++rows.cols;t.b.push_back(0);
    };
    for(auto c:p.cones){
        int d=c.width();
        if(c.kind!=Zero){
            t.cones.push_back(c);
            if(c.kind==Exponential){
                term(j+1,1);term(j,1);term(j+2,-2.7182818284590452354);
            } else for(int i=0;i<d;++i)term(j+i,-1);
        }
        j+=d;
    }
    t.A=rows.transpose();t.P=Sparse(m,m);t.q.resize(m);
    return t;
}
Model dual_problem(const Model& p,const Verifier& v) {
    const int n=p.n(),m=p.m(),nn=p.P.nnz()?n:0;
    Model t;
    t.A=Sparse(m+nn+1,n);t.P=Sparse(n,n);t.q.resize(n);t.b.resize(m+nn+1);
    t.b.back()=-1;
    t.cones=p.cones;t.cones.push_back({Zero,nn+1});
    t.A.i.reserve(p.A.nnz()+p.P.nnz()+n);t.A.x.reserve(t.A.i.capacity());
    for(int j=0;j<n;++j){
        for(int k=p.A.p[j];k<p.A.p[j+1];++k) {
            t.A.i.push_back(p.A.i[k]);t.A.x.push_back(p.A.x[k]/v.aScale);
        }
        for(int k=p.P.p[j];k<p.P.p[j+1];++k) {
            t.A.i.push_back(m+p.P.i[k]);t.A.x.push_back(p.P.x[k]/v.pScale);
        }
        if(p.q[j]){t.A.i.push_back(m+nn);t.A.x.push_back(p.q[j]/v.nQ);}
        t.A.p[j+1]=t.A.nnz();
    }
    return t;
}
}

bool certificate(const Model& p,Verifier& v,bool full) {
    if(p.n()+p.m()>13000 || p.A.nnz()+p.P.nnz()>450000)return false;
    if(v.nQ>0) {
        Vec q=p.q;
        for(double& x:q)x/=-v.nQ;
        if(v.dual(q))return true;
        auto at=p.A.transpose();
        if(v.dual(residual(at,p.P.nnz()?&p.P:nullptr,q,nullptr,5e-13)))return true;
        if(p.P.nnz() && v.dual(residual(p.P,nullptr,q,nullptr,5e-13)))return true;
        if(full) {
            auto s=scale_rows(at,p.P.nnz()?&p.P:nullptr);
            if(v.dual(residual(at,p.P.nnz()?&p.P:nullptr,q,&s,2e-10)))return true;
            if(p.P.nnz() && v.dual(residual(p.P,nullptr,q,&s,2e-11)))return true;
        }
    }
    if(v.nB>0) {
        Vec b=p.b;
        for(double& x:b)x/=-v.nB;
        if(v.primal(b) || v.primal(residual(p.A,nullptr,b,nullptr,5e-13)))return true;
        if(full) {
            auto s=scale_rows(p.A);
            if(v.primal(residual(p.A,nullptr,b,&s,3e-11)))return true;
        }
    }
    if(!full || p.n()+p.m()>4500 || p.A.nnz()+p.P.nnz()>155000 || v.best<1e-6)return false;
    for(int pass=0;pass<2;++pass){
        if(pass==0 && !(v.nQ>0))continue;
        if(pass==1 && !(v.nB>0))continue;
        try {
            Model t=pass?primal_problem(p,v):dual_problem(p,v);
            Verifier w(t);
            if(debug)std::fprintf(stderr,"normalized %s certificate model\n",pass?"primal":"dual");
            for(int i=0;i<2;++i){
                if(ipm(t,w,i) && w.result)break;
            }
            if(w.result.status!=Solved)continue;
            if(pass?v.primal(w.result.x):v.dual(w.result.x))return true;
        } catch(const std::exception& e) {
            if(debug)std::fprintf(stderr,"certificate solve skipped: %s\n",e.what());
        }
    }
    return false;
}
