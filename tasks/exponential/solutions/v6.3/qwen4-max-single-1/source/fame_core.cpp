#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>

extern "C" {

void cheb_apply(int64_t n, const int32_t* __restrict__ rowptr,
                const uint32_t* __restrict__ col, const double* __restrict__ val,
                double b, int64_t m, const double* __restrict__ ce,
                const double* __restrict__ ch, const double* __restrict__ b0,
                const double* __restrict__ b1, double* __restrict__ y) {
    if (m < 0) m = 0;
    if (b == 0.0) {
        for (int64_t i=0;i<n;++i) y[i]=b0[i];
        return;
    }
    double* v0 = new double[n]; double* v1 = new double[n];
    double* w0 = new double[n]; double* w1 = new double[n];
    double* u0 = new double[n]; double* u1 = new double[n];
    double* y0 = new double[n]; double* y1 = new double[n];
    double alpha = 2.0/b, gamma = 4.0/b;
    for (int64_t i=0;i<n;++i) {
        v0[i]=b0[i]; v1[i]=b1[i];
        y0[i]=ce[0]*b0[i]; y1[i]=ch[0]*b1[i];
    }
    if (m >= 1) {
        for (int64_t i=0;i<n;++i) {
            double s0=0.0,s1=0.0;
            for (int32_t p=rowptr[i]; p<rowptr[i+1]; ++p) {
                uint32_t j=col[p]; double a=val[p];
                s0 += a*v0[j]; s1 += a*v1[j];
            }
            w0[i]=alpha*s0-v0[i]; w1[i]=alpha*s1-v1[i];
            y0[i]+=ce[1]*w0[i]; y1[i]+=ch[1]*w1[i];
        }
        for (int64_t k=2;k<=m;++k) {
            double ck_e=ce[k], ck_h=ch[k];
            for (int64_t i=0;i<n;++i) {
                double s0=0.0,s1=0.0;
                for (int32_t p=rowptr[i]; p<rowptr[i+1]; ++p) {
                    uint32_t j=col[p]; double a=val[p];
                    s0 += a*w0[j]; s1 += a*w1[j];
                }
                double t0=gamma*s0-2.0*w0[i]-v0[i];
                double t1=gamma*s1-2.0*w1[i]-v1[i];
                u0[i]=t0; u1[i]=t1;
                y0[i]+=ck_e*t0; y1[i]+=ck_h*t1;
            }
            double* tmp;
            tmp=v0; v0=w0; w0=u0; u0=tmp;
            tmp=v1; v1=w1; w1=u1; u1=tmp;
        }
    }
    for (int64_t i=0;i<n;++i) y[i]=y0[i]+y1[i];
    delete[] v0; delete[] v1; delete[] w0; delete[] w1; delete[] u0; delete[] u1; delete[] y0; delete[] y1;
}

} // extern C

extern "C" {
void clenshaw_apply(int64_t n, const int32_t* __restrict__ rowptr,
                const int32_t* __restrict__ col, const double* __restrict__ val,
                double b, int64_t m, const double* __restrict__ ce,
                const double* __restrict__ ch, const double* __restrict__ b0,
                const double* __restrict__ b1, double* __restrict__ y) {
    if (m < 0) m = 0;
    if (b == 0.0) { for (int64_t i=0;i<n;++i) y[i]=b0[i]; return; }
    double* u10 = new double[n]; double* u11 = new double[n];
    double* u20 = new double[n]; double* u21 = new double[n];
    double* c0 = new double[n]; double* c1 = new double[n];
    double alpha=2.0/b, gamma=4.0/b;
    for (int64_t i=0;i<n;++i) { u10[i]=0;u11[i]=0;u20[i]=0;u21[i]=0; }
    for (int64_t k=m;k>=1;--k) {
        double ke=ce[k], kh=ch[k];
        if (k==m) {
            for (int64_t i=0;i<n;++i) { c0[i]=ke*b0[i]; c1[i]=kh*b1[i]; }
        } else {
            for (int64_t i=0;i<n;++i) {
                double s0=0,s1=0;
                for (int32_t p=rowptr[i]; p<rowptr[i+1]; ++p) {
                    int32_t j=col[p]; double a=val[p]; s0+=a*u10[j]; s1+=a*u11[j];
                }
                c0[i]=gamma*s0-2.0*u10[i]-u20[i]+ke*b0[i];
                c1[i]=gamma*s1-2.0*u11[i]-u21[i]+kh*b1[i];
            }
        }
        double* tmp;
        tmp=u20; u20=u10; u10=c0; c0=tmp;
        tmp=u21; u21=u11; u11=c1; c1=tmp;
    }
    for (int64_t i=0;i<n;++i) {
        double s0=0,s1=0;
        for (int32_t p=rowptr[i]; p<rowptr[i+1]; ++p) {
            int32_t j=col[p]; double a=val[p]; s0+=a*u10[j]; s1+=a*u11[j];
        }
        y[i]=alpha*s0-u10[i]-u20[i]+ce[0]*b0[i] + alpha*s1-u11[i]-u21[i]+ch[0]*b1[i];
    }
    delete[] u10;delete[] u11;delete[] u20;delete[] u21;delete[] c0;delete[] c1;
}
}

extern "C" {
void cheb_apply_ld(int64_t n, const int32_t* __restrict__ rowptr,
                const int32_t* __restrict__ col, const double* __restrict__ val,
                double b, int64_t m, const long double* __restrict__ ce,
                const long double* __restrict__ ch, const double* __restrict__ b0,
                const double* __restrict__ b1, double* __restrict__ y) {
    if (m < 0) m = 0;
    if (b == 0.0) { for (int64_t i=0;i<n;++i) y[i]=b0[i]; return; }
    long double* v0 = new long double[n]; long double* v1 = new long double[n];
    long double* w0 = new long double[n]; long double* w1 = new long double[n];
    long double* u0 = new long double[n]; long double* u1 = new long double[n];
    long double* y0 = new long double[n]; long double* y1 = new long double[n];
    long double alpha = 2.0L/b, gamma = 4.0L/b;
    for (int64_t i=0;i<n;++i) {
        v0[i]=b0[i]; v1[i]=b1[i]; y0[i]=ce[0]*b0[i]; y1[i]=ch[0]*b1[i];
    }
    if (m >= 1) {
        for (int64_t i=0;i<n;++i) {
            long double s0=0,s1=0;
            for (int32_t p=rowptr[i]; p<rowptr[i+1]; ++p) {
                int32_t j=col[p]; long double a=val[p]; s0+=a*v0[j]; s1+=a*v1[j];
            }
            w0[i]=alpha*s0-v0[i]; w1[i]=alpha*s1-v1[i];
            y0[i]+=ce[1]*w0[i]; y1[i]+=ch[1]*w1[i];
        }
        for (int64_t k=2;k<=m;++k) {
            long double ke=ce[k], kh=ch[k];
            for (int64_t i=0;i<n;++i) {
                long double s0=0,s1=0;
                for (int32_t p=rowptr[i]; p<rowptr[i+1]; ++p) {
                    int32_t j=col[p]; long double a=val[p]; s0+=a*w0[j]; s1+=a*w1[j];
                }
                long double t0=gamma*s0-2.0L*w0[i]-v0[i];
                long double t1=gamma*s1-2.0L*w1[i]-v1[i];
                u0[i]=t0; u1[i]=t1; y0[i]+=ke*t0; y1[i]+=kh*t1;
            }
            long double* tmp;
            tmp=v0;v0=w0;w0=u0;u0=tmp; tmp=v1;v1=w1;w1=u1;u1=tmp;
        }
    }
    for (int64_t i=0;i<n;++i) y[i]=(double)(y0[i]+y1[i]);
    delete[] v0;delete[] v1;delete[] w0;delete[] w1;delete[] u0;delete[] u1;delete[] y0;delete[] y1;
}
}

extern "C" {
void cheb_apply_mix(int64_t n, const int32_t* __restrict__ rowptr,
                const int32_t* __restrict__ col, const double* __restrict__ val,
                double b, int64_t m, const double* __restrict__ ce,
                const long double* __restrict__ ch, const double* __restrict__ b0,
                const double* __restrict__ b1, double* __restrict__ y) {
    if (m < 0) m = 0;
    if (b == 0.0) { for (int64_t i=0;i<n;++i) y[i]=b0[i]; return; }
    double* v0 = new double[n]; double* w0 = new double[n]; double* u0 = new double[n]; double* y0 = new double[n];
    long double* v1 = new long double[n]; long double* w1 = new long double[n]; long double* u1 = new long double[n]; long double* y1 = new long double[n];
    long double alpha = 2.0L/b, gamma = 4.0L/b;
    double alphad=2.0/b, gammad=4.0/b;
    for (int64_t i=0;i<n;++i) { v0[i]=b0[i]; v1[i]=b1[i]; y0[i]=ce[0]*b0[i]; y1[i]=ch[0]*b1[i]; }
    if (m>=1) {
        for (int64_t i=0;i<n;++i) {
            double s0=0; long double s1=0;
            for (int32_t p=rowptr[i]; p<rowptr[i+1]; ++p) { int32_t j=col[p]; double a=val[p]; s0+=a*v0[j]; s1+=(long double)a*v1[j]; }
            w0[i]=alphad*s0-v0[i]; w1[i]=alpha*s1-v1[i]; y0[i]+=ce[1]*w0[i]; y1[i]+=ch[1]*w1[i];
        }
        for (int64_t k=2;k<=m;++k) {
            double ke=ce[k]; long double kh=ch[k];
            for (int64_t i=0;i<n;++i) {
                double s0=0; long double s1=0;
                for (int32_t p=rowptr[i]; p<rowptr[i+1]; ++p) { int32_t j=col[p]; double a=val[p]; s0+=a*w0[j]; s1+=(long double)a*w1[j]; }
                double t0=gammad*s0-2.0*w0[i]-v0[i]; long double t1=gamma*s1-2.0L*w1[i]-v1[i];
                u0[i]=t0; u1[i]=t1; y0[i]+=ke*t0; y1[i]+=kh*t1;
            }
            double* td; long double* tl;
            td=v0;v0=w0;w0=u0;u0=td; tl=v1;v1=w1;w1=u1;u1=tl;
        }
    }
    for (int64_t i=0;i<n;++i) y[i]=y0[i]+(double)y1[i];
    delete[] v0;delete[] w0;delete[] u0;delete[] y0;delete[] v1;delete[] w1;delete[] u1;delete[] y1;
}
}

extern "C" {
double row_bound(int64_t n, const int32_t* __restrict__ rowptr,
                 const int32_t* __restrict__ col, const double* __restrict__ val,
                 int32_t* diag) {
    double mx=0.0; int dg=1;
    for (int64_t i=0;i<n;++i) {
        double s=0.0; int32_t st=rowptr[i], en=rowptr[i+1];
        if (en-st!=1 || col[st]!=(int32_t)i) dg=0;
        for (int32_t p=st;p<en;++p) s += std::fabs(val[p]);
        if (s>mx) mx=s;
    }
    *diag=dg;
    return mx;
}
}

extern "C" {
void cheb_apply_mix2(int64_t n, const int32_t* __restrict__ rowptr,
                const int32_t* __restrict__ col, const double* __restrict__ val,
                double b, int64_t m, const double* __restrict__ ce,
                const long double* __restrict__ ch, const double* __restrict__ b0,
                const double* __restrict__ b1, double* __restrict__ y) {
    if (m < 0) m = 0;
    if (b == 0.0) { for (int64_t i=0;i<n;++i) y[i]=b0[i]; return; }
    double* v0 = new double[n]; double* w0 = new double[n]; double* u0 = new double[n]; double* y0 = new double[n];
    long double* v1 = new long double[n]; long double* w1 = new long double[n]; long double* u1 = new long double[n]; long double* y1 = new long double[n];
    long double alpha = 2.0L/b, gamma = 4.0L/b;
    double alphad=2.0/b, gammad=4.0/b;
    for (int64_t i=0;i<n;++i) { v0[i]=b0[i]; v1[i]=b1[i]; y0[i]=ce[0]*b0[i]; y1[i]=ch[0]*b1[i]; }
    if (m>=1) {
        for (int64_t i=0;i<n;++i) {
            double s0=0,s1=0;
            for (int32_t p=rowptr[i]; p<rowptr[i+1]; ++p) { int32_t j=col[p]; double a=val[p]; s0+=a*v0[j]; s1+=a*(double)v1[j]; }
            w0[i]=alphad*s0-v0[i]; w1[i]=alpha*(long double)s1-v1[i]; y0[i]+=ce[1]*w0[i]; y1[i]+=ch[1]*w1[i];
        }
        for (int64_t k=2;k<=m;++k) {
            double ke=ce[k]; long double kh=ch[k];
            for (int64_t i=0;i<n;++i) {
                double s0=0,s1=0;
                for (int32_t p=rowptr[i]; p<rowptr[i+1]; ++p) { int32_t j=col[p]; double a=val[p]; s0+=a*w0[j]; s1+=a*(double)w1[j]; }
                double t0=gammad*s0-2.0*w0[i]-v0[i]; long double t1=gamma*(long double)s1-2.0L*w1[i]-v1[i];
                u0[i]=t0; u1[i]=t1; y0[i]+=ke*t0; y1[i]+=kh*t1;
            }
            double* td; long double* tl;
            td=v0;v0=w0;w0=u0;u0=td; tl=v1;v1=w1;w1=u1;u1=tl;
        }
    }
    for (int64_t i=0;i<n;++i) y[i]=y0[i]+(double)y1[i];
    delete[] v0;delete[] w0;delete[] u0;delete[] y0;delete[] v1;delete[] w1;delete[] u1;delete[] y1;
}
}
