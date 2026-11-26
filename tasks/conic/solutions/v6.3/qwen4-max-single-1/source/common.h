#ifndef COMMON_H
#define COMMON_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

typedef int32_t idx_t;

static inline void *xmalloc(size_t n){ void *p=malloc(n?n:1); if(!p){fprintf(stderr,"oom\n");exit(3);} return p; }
static inline void *xcalloc(size_t n,size_t s){ void *p=calloc(n?n:1,s?s:1); if(!p){fprintf(stderr,"oom\n");exit(3);} return p; }
static inline void *xrealloc(void *q,size_t n){ void *p=realloc(q,n?n:1); if(!p){fprintf(stderr,"oom\n");exit(3);} return p; }

static inline double now_sec(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + 1e-9*ts.tv_nsec;
}

/* CSC sparse matrix, 0-based indices */
typedef struct { idx_t nrow,ncol; idx_t *p,*i; double *x; } CSC;

#endif
