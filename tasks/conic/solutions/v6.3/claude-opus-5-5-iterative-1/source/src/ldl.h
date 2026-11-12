#ifndef LDL_H
#define LDL_H

/* Sparse LDL' factorization for quasi-definite matrices (QDLDL-style, up-looking,
   no pivoting) with a fill-reducing symmetric permutation and Clarabel-style
   dynamic regularization. */

typedef struct ldl_s ldl_t;

/* Symbolic analysis of the n x n symmetric matrix K given by its UPPER triangle in CSC
   (Ap, Ai). Row indices within a column need not be sorted; duplicates are summed.
   (An entry given below the diagonal is treated as its mirrored upper entry.)
   A missing diagonal entry is treated as a structural zero.
   perm: fill-reducing ordering (perm[k] = original index of k-th pivot), or NULL to
   compute amd_order internally. The pattern is copied; Ap/Ai/perm need not outlive the call.
   Returns NULL on failure (allocation, invalid input, or nnz(L) > INT_MAX). */
ldl_t *ldl_symbolic(int n, const int *Ap, const int *Ai, const int *perm);

/* number of strictly-lower nonzeros in L */
long ldl_nnz_L(const ldl_t *f);

/* Numeric factorization P K P' = L D L' using values Ax aligned with the (Ap,Ai)
   pattern passed to ldl_symbolic.
   sign[i] (i = ORIGINAL index) is +1 or -1: the expected sign of that pivot.
   Dynamic regularization: if sign[i]*d_k <= eps then d_k = sign[i]*delta (counted).
   sign == NULL disables regularization (then a zero pivot is an error).
   Returns the number of regularized pivots (>= 0), or -1 if a non-finite value
   (or, with sign == NULL, a zero pivot) appears. */
int ldl_numeric(ldl_t *f, const double *Ax, const signed char *sign, double eps, double delta);

/* Solve (P' L D L' P) x = b in place; b and the result are in ORIGINAL ordering. */
void ldl_solve(ldl_t *f, double *b);

void ldl_free(ldl_t *f);

/* --- optional accessors --- */
int ldl_n(const ldl_t *f);
/* fill-reducing permutation used (perm[k] = original index of k-th pivot) */
const int *ldl_perm(const ldl_t *f);
/* D of the last successful numeric factorization, in PERMUTED order (D[k] for pivot k) */
const double *ldl_D(const ldl_t *f);

#endif
