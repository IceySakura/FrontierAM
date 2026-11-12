/* npzio.h - read conic problem NPZ files and write result NPZ files (no Python).
 *
 * Input : NPZ (zip, DEFLATE or STORED entries, ZIP64 aware) containing
 *         q, b, P_data, P_indices, P_indptr, P_shape, A_data, A_indices,
 *         A_indptr, A_shape, metadata_json (.npy).
 * Output: uncompressed NPZ with metadata_json and optional x, s, z.
 *
 * Link with zlib: -l:libz.so.1 (no zlib.h needed).
 */
#ifndef NPZIO_H
#define NPZIO_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { CONE_ZERO = 0, CONE_NONNEG = 1, CONE_SOC = 2, CONE_EXP = 3, CONE_PSD = 4 };

typedef struct {
    int n, m;
    int *Pp, *Pi; double *Px;   /* P: n x n full symmetric, CSC, 0-based */
    int *Ap, *Ai; double *Ax;   /* A: m x n, CSC, 0-based */
    double *q, *b;              /* lengths n and m */
    int ncones;
    int *cone_type;             /* 0 zero, 1 nonnegative, 2 soc, 3 exponential, 4 psd */
    int *cone_dim;              /* JSON "dim" (psd: matrix order) */
    double objective_offset;
} problem_t;

/* Reads the problem. Matrices are returned in canonical CSC form (rows
 * strictly increasing within each column, duplicates summed, explicit zeros
 * removed). All array pointers are non-NULL on success (even for empty
 * arrays). Returns 0 on success; on failure writes a message into err (if
 * err != NULL) and leaves *prob zeroed. */
int npz_read_problem(const char *path, problem_t *prob, char *err, size_t errlen);

/* Frees all arrays and zeroes the struct (safe on a zeroed struct). */
void problem_free(problem_t *prob);

/* status: "solved", "primal_infeasible", "dual_infeasible", "unresolved".
 * x (length n), s, z (length m) may be NULL to omit that vector.
 * If has_obj != 0, both objectives are finite and status is "solved",
 * "objective_primal" and "objective_dual" are written into the JSON;
 * otherwise they are omitted.  Vectors are written as given (<f8).
 * Returns 0 on success. */
int npz_write_result(const char *path, const char *status, const double *x, int n,
                     const double *s, const double *z, int m, int has_obj,
                     double obj_p, double obj_d);

#ifdef __cplusplus
}
#endif

#endif /* NPZIO_H */
