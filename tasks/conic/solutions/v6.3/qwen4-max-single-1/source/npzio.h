#ifndef NPZIO_H
#define NPZIO_H
#include "common.h"

typedef struct { int type; idx_t dim,start,width; } Block; /*0 zero,1 nonneg,2 soc,3 exp,4 psd*/
typedef struct {
    idx_t n,m;
    CSC P,A;
    double *q,*b;
    double offset;
    int nblk; Block *blk;
} Problem;

unsigned char *read_file(const char *path,size_t *len);
unsigned char *inflate_raw(const unsigned char *in,size_t inlen,size_t *outlen);
int load_problem(const char *path,Problem *P);
int write_result(const char *path,const char *status,const double *x,const double *s,const double *z,idx_t n,idx_t m);

#endif
