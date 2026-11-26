#ifndef KKT_H
#define KKT_H
#include "common.h"
typedef struct {
    idx_t N; long nnz,fullnnz;
    const idx_t *ti,*tj; const double *tv;
    idx_t *perm,*pinv;
    idx_t *Ap,*Ai; double *Ax; long *slot;
    idx_t *etree,*Lp,*Li; double *Lx,*D,*work;
} KKT;
void kkt_setup(KKT *K,idx_t N,long nnz,const idx_t *ti,const idx_t *tj,const double *tv);
void kkt_refill(KKT *K,const double *tv);
int kkt_factor(KKT *K);
void kkt_solve(KKT *K,const double *rhs,double *sol);
void kkt_matvec(const KKT *K,const double *x,double *y);
#endif
