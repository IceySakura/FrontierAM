/* Hard-capped ctypes adapter for the cubature libraries.
 *
 * Python owns the point budget: the C layer consumes one count per integrand
 * invocation and, once exhausted, refuses every further request (it never
 * asks Python for a value it cannot account for).  Every value returned by the
 * library therefore counts against the single 2048-point oracle budget,
 * including batched/vector evaluations, cached re-evaluations and the
 * library's own rule blocks.
 *
 * Build:
 *   cc -shared -fPIC -O2 -I. adapter.c hcubature.c -lm -o libhcubature.so
 *   cc -shared -fPIC -O2 -I. adapter.c pcubature.c -lm -o libpcubature.so
 */
#include <stddef.h>
#include "cubature.h"

/* Python callback signatures, exactly matching the ctypes CFUNCTYPE objects
 * that Python passes through (they arrive as genuine C function pointers). */
typedef int (*py_scalar_t)(unsigned ndim, const double *x, double *fval,
                           void *pydata);
typedef int (*py_vector_t)(unsigned ndim, size_t npt, const double *x,
                           void *pydata, unsigned fdim, double *fval);

typedef struct {
    void          *pydata;   /* passed straight through to Python     */
    py_scalar_t    scalar;
    py_vector_t    vector;
    size_t         max_pts; /* hard point cap for this run           */
    size_t         used;    /* points already consumed               */
} cbdata;

/* hcubature calls this once per single point. */
static int scalar_cb(unsigned ndim, const double *x, void *fd,
                     unsigned fdim, double *fval) {
    cbdata *d = (cbdata *)fd;
    (void)fdim;
    if (d->used >= d->max_pts) return 1;
    d->used++;
    if (d->scalar(ndim, x, fval, d->pydata)) return 1;
    return 0;
}

/* pcubature calls this with a batch of npt points.  A batch that would
 * overflow the cap is rejected before any value is produced, so the oracle is
 * never asked for more points than we are allowed to spend. */
static int vector_cb(unsigned ndim, size_t npt, const double *x, void *fd,
                     unsigned fdim, double *fval) {
    cbdata *d = (cbdata *)fd;
    (void)fdim;
    if (d->used + npt > d->max_pts) return 1;
    d->used += npt;
    if (d->vector(ndim, npt, x, d->pydata, fdim, fval)) return 1;
    return 0;
}

int call_hcubature(unsigned fdim, py_scalar_t f, void *pydata, unsigned dim,
                   const double *xmin, const double *xmax, size_t maxEval,
                   double reqAbsError, double reqRelError, int norm,
                   size_t max_pts, double *val, double *err) {
    cbdata d = {pydata, f, NULL, max_pts, 0};
    return (int)hcubature(fdim, scalar_cb, &d, dim, xmin, xmax, maxEval,
                          reqAbsError, reqRelError, (error_norm)norm, val, err);
}

int call_pcubature(py_vector_t f, void *pydata, unsigned dim,
                   const double *xmin, const double *xmax, size_t maxEval,
                   double reqAbsError, double reqRelError, int norm,
                   size_t max_pts, double *val, double *err) {
    cbdata d = {pydata, NULL, f, max_pts, 0};
    return (int)pcubature_v(1, vector_cb, &d, dim, xmin, xmax, maxEval,
                            reqAbsError, reqRelError, (error_norm)norm, val, err);
}
