/* Sparse matrix helpers + sparse LDL^T (no pivoting, for quasi-definite KKT). */
#ifndef SPARSE_H
#define SPARSE_H

typedef struct { int m, n, nnz; int *p, *i; double *x; } Sp;   /* CSC */
typedef struct { int m, n, nnz; int *p, *j; double *x; } SpR;   /* CSR */

Sp  *sp_alloc(int m, int n, int nnz);
void sp_free(Sp *A);
Sp  *sp_rows(const Sp *A, int r0, int r1);            /* extract rows [r0,r1) */
Sp  *sp_transpose(const Sp *A);
void spr_from_sp(SpR *R, const Sp *A);
void spr_free(SpR *R);

/* ordering */
void ord_natural(int *perm, int n);
void ord_rcm(const Sp *Ksym, int n, int *perm);
void ord_deg(const Sp *Ksym, int n, int *perm);

/* LDL^T with dynamic columns */
typedef struct {
    int N, inited;
    int *perm, *pinv;
    double *d;
    int **Li; double **Lx; int *Lc, *Lcap;
    int *parent, *mark, *stack, *path, *tmark, *touched;
    double *x, *tmp;
    long nnzL;
} LDL;

void ldl_init(LDL *F, int N);
void ldl_free(LDL *F);
void ldl_setperm(LDL *F, const int *perm);
int  ldl_factor(LDL *F, const Sp *K);       /* 0 ok, else pivot index+1 failed */
void ldl_solve(LDL *F, const double *b, double *out);

#endif
