#pragma once
#include "common.hpp"

extern "C" {
void scipy_dgemm_(const char*,const char*,const int*,const int*,const int*,const double*,
                 const double*,const int*,const double*,const int*,const double*,double*,const int*);
void scipy_dsyrk_(const char*,const char*,const int*,const int*,const double*,const double*,
                 const int*,const double*,double*,const int*);
void scipy_dpotrf_(const char*,const int*,double*,const int*,int*);
void scipy_dpotrs_(const char*,const int*,const int*,const double*,const int*,double*,const int*,int*);
void scipy_dsytrf_(const char*,const int*,double*,const int*,int*,double*,const int*,int*);
void scipy_dsytrs_(const char*,const int*,const int*,const double*,const int*,const int*,double*,const int*,int*);
void scipy_dgesvd_(const char*,const char*,const int*,const int*,double*,const int*,double*,
                   double*,const int*,double*,const int*,double*,const int*,int*);
void scipy_dsyevr_(const char*,const char*,const char*,const int*,double*,const int*,const double*,
                   const double*,const int*,const int*,const double*,int*,double*,double*,const int*,
                   int*,double*,const int*,int*,const int*,int*);
void scipy_dsyevd_(const char*,const char*,const int*,double*,const int*,double*,
                   double*,const int*,int*,const int*,int*);
void scipy_dtrmm_(const char*,const char*,const char*,const char*,const int*,const int*,
                 const double*,const double*,const int*,double*,const int*);
void scipy_dtrsm_(const char*,const char*,const char*,const char*,const int*,const int*,
                 const double*,const double*,const int*,double*,const int*);
void scipy_dgeqp3_(const int*,const int*,double*,const int*,int*,double*,double*,const int*,int*);
void scipy_dormqr_(const char*,const char*,const int*,const int*,const int*,const double*,const int*,
                  const double*,double*,const int*,double*,const int*,int*);
}

namespace conic {

inline bool chol(int n, double* a) {
    int info;
    scipy_dpotrf_("L", &n, a, &n, &info);
    return info == 0;
}

inline void gemm(const char* ta, const char* tb, int m, int n, int k, const double* a,
                 int lda, const double* b, int ldb, double* c, int ldc,
                 double alpha = 1., double beta = 0.) {
    scipy_dgemm_(ta,tb,&m,&n,&k,&alpha,a,&lda,b,&ldb,&beta,c,&ldc);
}

inline void symmetrize_lower(double* a, int d) {
    for (int j = 1; j < d; ++j)
        for (int i = 0; i < j; ++i) a[i + j*d] = a[j + i*d];
}

inline void gram(int d, const double* a, double* b, bool trans = false) {
    double one = 1, zero = 0;
    scipy_dsyrk_("L", trans ? "T" : "N", &d, &d, &one, a, &d, &zero, b, &d);
    symmetrize_lower(b, d);
}

inline void congruence(int d, const double* a, const double* s, double* out, double* tmp,
                       bool trans = false) {
    gemm(trans ? "T" : "N","N",d,d,d,a,d,s,d,tmp,d);
    gemm("N",trans ? "N" : "T",d,d,d,tmp,d,a,d,out,d);
}

inline void smat(int d, const double* u, double* a, double scale=1.) {
    const double off = scale * invroot2;
    int k = 0;
    for (int j = 0; j < d; ++j) {
        for (int i = 0; i < j; ++i) a[i + d*j] = a[j + d*i] = u[k++] * off;
        a[j + d*j] = u[k++] * scale;
    }
}

inline void svec(int d, const double* a, double* u, double scale=1.) {
    const double off = scale * invroot2;
    int k = 0;
    for (int j = 0; j < d; ++j) {
        for (int i = 0; i < j; ++i) u[k++] = (a[i + d*j] + a[j + d*i]) * off;
        u[k++] = a[j + d*j] * scale;
    }
}

inline double smallest_eigen(int d, double* a, Vec& work, Int& iw, Vec& eig) {
    if (d == 1) return a[0];
    if (d == 2) {
        const double mid = .5 * (a[0] + a[3]);
        return mid - std::hypot(.5 * (a[0] - a[3]), a[1]);
    }
    if (work.size() < size_t(40)*d) work.resize(size_t(40)*d);
    if (iw.size() < size_t(12)*d) iw.resize(size_t(12)*d);
    if (eig.size() < size_t(d)) eig.resize(d);
    const int one=1, lw=int(work.size()), liw=int(iw.size());
    int got=0, info=0, support[2];
    const double zero=0;
    double dummy=0;
    scipy_dsyevr_("N","I","L",&d,a,&d,&zero,&zero,&one,&one,&zero,
                 &got,eig.data(),&dummy,&one,support,work.data(),&lw,iw.data(),&liw,&info);
    return (info == 0 && got == 1 && std::isfinite(eig[0])) ? eig[0] : -inf;
}

}  // namespace conic
