#ifndef LDL_H
#define LDL_H

/*
 * Sparse symmetric LDL^T factorization (QDLDL-style up-looking algorithm)
 * with an approximate minimum degree (AMD) fill-reducing ordering.
 *
 * Intended for quasi-definite KKT systems arising in interior point methods.
 * All indices are int64_t.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ldl_s ldl_t;

/* Symbolic analysis. Input: the UPPER triangle (row <= col), including ALL
   diagonal entries, of a symmetric n x n matrix in CSC format: Kp (n+1),
   Ki (Kp[n] row indices; within a column rows may be unsorted; no duplicates).
   If perm != NULL use it as the elimination ordering (perm[k] = original index
   eliminated k-th), otherwise compute AMD.
   Returns NULL on failure (e.g. out of memory or invalid input). */
ldl_t *ldl_create(int64_t n, const int64_t *Kp, const int64_t *Ki, const int64_t *perm);

/* Numeric factorization of P*K*P' = L*D*L' using values Kx laid out exactly
   like Ki. signs[i] in {+1,-1} is the expected pivot sign for ORIGINAL index i
   (signs may be NULL: no dynamic regularization). Dynamic regularization: if
   signs[i]*d <= eps then d = signs[i]*delta (counted).
   Returns the number of regularized pivots (>= 0), or -1 if a pivot is still
   zero or nonfinite. Does not allocate memory. */
int64_t ldl_factor(ldl_t *f, const double *Kx, const signed char *signs, double eps, double delta);

/* Solve K x = b in place (b and result in ORIGINAL ordering). */
void ldl_solve(ldl_t *f, double *b);

/* Number of strictly-lower nonzeros in L. */
int64_t ldl_nnz_L(const ldl_t *f);

/* The ordering used (perm[k] = original index eliminated k-th). */
const int64_t *ldl_perm(const ldl_t *f);

void ldl_free(ldl_t *f);

/* Optional BLAS (Fortran ABI, LP64 integers, hidden string lengths) used by
   the supernodal factorization for large dense updates. Global setting;
   NULL (default) = use the built-in loops. Results agree up to rounding. */
#define LDL_HAVE_BLAS_HOOKS 1
typedef void (*ldl_dgemm_fn)(const char *ta, const char *tb, const int *m, const int *n, const int *k,
                             const double *alpha, const double *A, const int *lda, const double *B,
                             const int *ldb, const double *beta, double *C, const int *ldc, size_t,
                             size_t);
typedef void (*ldl_dtrsm_fn)(const char *side, const char *uplo, const char *transa, const char *diag,
                             const int *m, const int *n, const double *alpha, const double *A,
                             const int *lda, double *B, const int *ldb, size_t, size_t, size_t,
                             size_t);
void ldl_set_blas(ldl_dgemm_fn gemm, ldl_dtrsm_fn trsm);

/* Approximate minimum degree ordering of the symmetric matrix whose upper
   triangle (CSC, diagonal ignored) is given. perm[k] = k-th pivot.
   Returns 0 on success, negative on failure. */
int64_t amd_order(int64_t n, const int64_t *Kp, const int64_t *Ki, int64_t *perm);

#ifdef __cplusplus
}
#endif

#endif /* LDL_H */
