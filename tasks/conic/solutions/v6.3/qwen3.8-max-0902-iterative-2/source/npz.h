/* NPZ (zip) reading/writing without numpy. */
#ifndef NPZ_H
#define NPZ_H
#include <stddef.h>

typedef struct {
    char *name;      /* malloc'd, ".npy" suffix stripped */
    char descr[24];  /* e.g. "<f8", "<i4", "<U3894" */
    int fortran;
    int ndim;
    long shape[8];
    long count;      /* number of elements */
    int itemsize;
    void *data;      /* malloc'd raw bytes */
} NpyArr;

typedef struct { NpyArr *arr; int n; } Npz;

Npz *npz_read(const char *path);
void npz_free(Npz *z);
NpyArr *npz_get(Npz *z, const char *name);

/* ---- writer (uncompressed zip) ---- */
typedef struct {
    char *name; char *descr; int ndim; long shape[2]; char *data; long nbytes;
} WEnt;
typedef struct { WEnt *e; int n, cap; } NpzW;

void npzw_init(NpzW *w);
void npzw_add_f8(NpzW *w, const char *name, const double *data, long n);
void npzw_add_empty_f8(NpzW *w, const char *name);
void npzw_add_ustr(NpzW *w, const char *name, const char *utf8);
int  npzw_write(NpzW *w, const char *path);
void npzw_free(NpzW *w);

#endif
