/* npzio.h - minimal, self-contained NumPy .npz reader/writer (C99, no external deps). */
#ifndef NPZIO_H
#define NPZIO_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

enum { NPZ_CONE_ZERO = 0, NPZ_CONE_NONNEG = 1, NPZ_CONE_SOC = 2, NPZ_CONE_EXP = 3, NPZ_CONE_PSD = 4 };

typedef struct {
  int64_t n, m;                 /* n = len(q) (columns), m = len(b) (rows) */
  int64_t *P_colptr, *P_rowind; double *P_val; int64_t P_nnz;   /* CSC, n x n, as stored (may have duplicates, unsorted) */
  int64_t *A_colptr, *A_rowind; double *A_val; int64_t A_nnz;   /* CSC, m x n */
  double *q, *b;
  int ncones; int *cone_type; int64_t *cone_dim; /* type: 0 zero, 1 nonnegative, 2 soc, 3 exponential, 4 psd; dim exactly as in JSON */
  double objective_offset;
} npz_problem;

/* returns 0 on success, nonzero on error (writes a message into err, size errlen).
   On error *p is left zeroed (nothing to free). All array pointers are non-NULL on success. */
int npz_read_problem(const char *path, npz_problem *p, char *err, int errlen);
void npz_free_problem(npz_problem *p);

/* Write result NPZ. status is one of "solved","primal_infeasible","dual_infeasible","unresolved".
   Any of x,s,z may be NULL (then that array is omitted). has_obj!=0 -> include "objective_primal" and "objective_dual"
   (finite doubles) in the JSON; otherwise omit them. Returns 0 on success. */
int npz_write_result(const char *path, const char *status,
                     const double *x, int64_t nx, const double *s, int64_t ns, const double *z, int64_t nz,
                     int has_obj, double obj_primal, double obj_dual);

#ifdef __cplusplus
}
#endif
#endif
