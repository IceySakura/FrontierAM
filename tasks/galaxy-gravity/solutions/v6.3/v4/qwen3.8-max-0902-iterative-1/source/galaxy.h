/* galaxy.h -- shared definitions for the softened-gravity solver. */
#ifndef GALAXY_H
#define GALAXY_H

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef int32_t i32;
typedef int64_t i64;
typedef uint32_t u32;
typedef uint64_t u64;

#define GAL_EPS 0.01
#define GAL_EPS2 (GAL_EPS * GAL_EPS)

static const i64 MORTON_MASK = 0x1FFFFFll;   /* 21 bits per axis */
#define MBITS 21

/* ------------------------------------------------------------------ */
/* Particle data (structure of arrays).                                */
/* ------------------------------------------------------------------ */
typedef struct {
  i64 n;
  double *m;     /* [n]      */
  double *px;    /* [n]      */
  double *py;
  double *pz;
} Particles;

/* ------------------------------------------------------------------ */
/* Tree node.  A node is either an internal cell (children[i] >= 0)    */
/* or a leaf holding particles [p0,p1).                                */
/* ------------------------------------------------------------------ */
typedef struct {
  double cx, cy, cz;   /* expansion centre (centre of mass)         */
  double m;            /* total mass                                */
  float rad;           /* max distance of any particle from centre  */
  i32 children[8];     /* -1 when a leaf                            */
  i32 p0, p1;          /* particle range when a leaf                */
} Node;

typedef struct {
  Node *nodes;
  i64 nnodes, cap;
  /* sorted particle arrays (Morton order) */
  double *sx, *sy, *sz, *sm;
  i64 n;
  i32 *midx;           /* [nnodes][p] moment offset table           */
  double *mom;         /* flat moment storage                       */
  i64 nmom_per_node;
  i32 p;               /* expansion order                           */
  double bx0, by0, bz0, bsize;
} Tree;

/* moment index helpers */
static inline i32 mom_idx(i32 p, i32 a, i32 b, i32 c)
{
  i32 o = 0;
  for (i32 l = 0; l < p; l++) o += (l + 1) * (l + 2) / 2;
  return o + (a * (a + 1) * (a + 2)) / 6 + (b * (b + 1)) / 2 + c;
}
static inline i32 mom_count(i32 p)
{
  return (p + 1) * (p + 2) * (p + 3) / 6;
}

/* ------------------------------------------------------------------ */
/* direct.c : exact all-pairs summation (AVX2)                         */
/* ------------------------------------------------------------------ */
void direct_f64(const double *m, const double *px, const double *py,
                const double *pz, i64 n, double *ax, double *ay, double *az,
                double eps2, i64 i0, i64 i1);

/* ------------------------------------------------------------------ */
/* tree.c                                                              */
/* ------------------------------------------------------------------ */
void tree_build(Tree *t, const Particles *p, i32 leafmax, i32 porder,
                double time_budget, i64 mem_budget);
void tree_free(Tree *t);
void tree_compute_moments(Tree *t);

/* walk for target block of NT targets starting at i0 (sorted order) */
void tree_walk(Tree *t, double theta, i64 i0, i64 i1,
               double *ax, double *ay, double *az);

#endif
