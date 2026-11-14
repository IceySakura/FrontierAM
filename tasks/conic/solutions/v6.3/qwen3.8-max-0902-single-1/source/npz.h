/* npz.h - minimal reader/writer for NumPy .npz files (ZIP + .npy).
 *
 * Plain C99, no global state.  Implementation in npz.c, which needs to be
 * linked against zlib (libz) for the raw-DEFLATE inflater and crc32().
 *
 * Reading supports zip entries that are STORED (method 0) or DEFLATE
 * (method 8), including the ZIP64 extra fields that numpy >= 1.25 emits,
 * data descriptors, and .npy format versions 1.0 / 2.0 / 3.0.
 */
#ifndef NPZ_H
#define NPZ_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  double *q;  long n;            /* q array, length n */
  double *b;  long m;            /* b array, length m */
  double *P_data;  long *P_indices; long *P_indptr; long Pn; /* CSC, Pn = P_shape[0] */
  double *A_data; long *A_indices; long *A_indptr; long An, Am; /* CSC of m x n matrix */
  char *metadata_json;           /* NUL-terminated malloc'd copy of the metadata_json string */
  /* internals: raw buffers - opaque, freed by npz_free */
  void *_own[16];
} NpzProblem;

typedef struct { const char *name; const double *data; long len; } NpzArrayOut;

/* Returns 0 on success, nonzero on failure (prints a short message to stderr).
 * On failure *out is left zeroed, so npz_free(out) is always safe to call. */
int npz_read(const char *path, NpzProblem *out);

/* Releases every buffer owned by p and zeroes the struct.  Safe on NULL and
 * safe on a struct that was never successfully filled in. */
void npz_free(NpzProblem *p);

/* Creates an uncompressed (STORED) .npz at path.  `arrays` is a list of named
 * float64 1-D arrays; metadata_json is written as a 0-d (scalar) unicode array
 * exactly like numpy does (dtype '<U<len>', fortran_order False, shape ()).
 * The x/s/z arguments may be NULL when the corresponding length is 0; they are
 * written as float64 arrays named "x", "s", "z" (a zero-length array is still
 * written, as shape (0,)).  Returns 0 on success. */
int npz_write(const char *path, const NpzArrayOut *arrays, int narrays,
              const char *metadata_json, const double *x, long nx,
              const double *s, long ns, const double *z, long nz);

#ifdef __cplusplus
}
#endif
#endif /* NPZ_H */
