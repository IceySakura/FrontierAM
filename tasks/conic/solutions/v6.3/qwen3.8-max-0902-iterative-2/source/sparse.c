#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sparse.h"

Sp *sp_alloc(int m, int n, int nnz) {
    Sp *A = (Sp *)calloc(1, sizeof(Sp));
    A->m = m; A->n = n; A->nnz = nnz;
    A->p = (int *)calloc(n + 1, sizeof(int));
    A->i = (int *)malloc((nnz > 0 ? nnz : 1) * sizeof(int));
    A->x = (double *)malloc((nnz > 0 ? nnz : 1) * sizeof(double));
    return A;
}
void sp_free(Sp *A) { if (!A) return; free(A->p); free(A->i); free(A->x); free(A); }

Sp *sp_rows(const Sp *A, int r0, int r1) {
    int nr = r1 - r0, nnz = 0;
    int *cnt = (int *)calloc(A->n + 1, sizeof(int));
    for (int j = 0; j < A->n; j++)
        for (int p = A->p[j]; p < A->p[j + 1]; p++) { int r = A->i[p]; if (r >= r0 && r < r1) { cnt[j]++; nnz++; } }
    Sp *B = sp_alloc(nr, A->n, nnz);
    for (int j = 0; j < A->n; j++) B->p[j + 1] = B->p[j] + cnt[j];
    int *cur = (int *)memcpy(malloc((A->n + 1) * sizeof(int)), B->p, (A->n + 1) * sizeof(int));
    for (int j = 0; j < A->n; j++)
        for (int p = A->p[j]; p < A->p[j + 1]; p++) {
            int r = A->i[p]; if (r < r0 || r >= r1) continue;
            int q = cur[j]++; B->i[q] = r - r0; B->x[q] = A->x[p];
        }
    free(cur); free(cnt);
    for (int j = 0; j < A->n; j++) { /* sort rows within column */
        int a = B->p[j], b = B->p[j + 1];
        for (int u = a; u < b; u++) for (int v = u + 1; v < b; v++) if (B->i[v] < B->i[u]) {
            int ti = B->i[u]; B->i[u] = B->i[v]; B->i[v] = ti;
            double tx = B->x[u]; B->x[u] = B->x[v]; B->x[v] = tx;
        }
    }
    return B;
}

Sp *sp_transpose(const Sp *A) {
    Sp *T = sp_alloc(A->n, A->m, A->nnz);
    int *cnt = (int *)calloc(A->m + 1, sizeof(int));
    for (int p = 0; p < A->nnz; p++) cnt[A->i[p]]++;
    for (int i = 0; i < A->m; i++) T->p[i + 1] = T->p[i] + cnt[i];
    memcpy(cnt, T->p, (A->m + 1) * sizeof(int));
    for (int j = 0; j < A->n; j++)
        for (int p = A->p[j]; p < A->p[j + 1]; p++) { int q = cnt[A->i[p]]++; T->i[q] = j; T->x[q] = A->x[p]; }
    free(cnt);
    return T;
}

void spr_from_sp(SpR *R, const Sp *A) {
    R->m = A->m; R->n = A->n; R->nnz = A->nnz;
    R->p = (int *)calloc(A->m + 1, sizeof(int));
    R->j = (int *)malloc((A->nnz > 0 ? A->nnz : 1) * sizeof(int));
    R->x = (double *)malloc((A->nnz > 0 ? A->nnz : 1) * sizeof(double));
    for (int p = 0; p < A->nnz; p++) R->p[A->i[p] + 1]++;
    for (int i = 0; i < A->m; i++) R->p[i + 1] += R->p[i];
    int *cur = (int *)malloc((A->m + 1) * sizeof(int));
    memcpy(cur, R->p, (A->m + 1) * sizeof(int));
    for (int j = 0; j < A->n; j++)
        for (int p = A->p[j]; p < A->p[j + 1]; p++) { int q = cur[A->i[p]]++; R->j[q] = j; R->x[q] = A->x[p]; }
    free(cur);
}
void spr_free(SpR *R) { free(R->p); free(R->j); free(R->x); R->p = R->j = NULL; R->x = NULL; }

/* ---------------- orderings ---------------- */
void ord_natural(int *perm, int n) { for (int i = 0; i < n; i++) perm[i] = i; }

static void bfs_order(const Sp *K, int n, int start, int *order, int *deg) {
    int *q = (int *)malloc(n * sizeof(int));
    int *lvl = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) lvl[i] = -1;
    int qh = 0, qt = 0;
    q[qt++] = start; lvl[start] = 0;
    while (qh < qt) {
        int v = q[qh++];
        for (int p = K->p[v]; p < K->p[v + 1]; p++) {
            int w = K->i[p]; if (w == v || lvl[w] >= 0) continue;
            lvl[w] = lvl[v] + 1; q[qt++] = w;
        }
    }
    /* sort by (level, degree) using counting sort on level then stable by degree */
    int maxl = 0;
    for (int i = 0; i < qt; i++) if (lvl[q[i]] > maxl) maxl = lvl[q[i]];
    int *bc = (int *)calloc(maxl + 2, sizeof(int));
    for (int i = 0; i < qt; i++) bc[lvl[q[i]] + 1]++;
    for (int i = 0; i <= maxl; i++) bc[i + 1] += bc[i];
    int *tmp = (int *)malloc(qt * sizeof(int));
    { int *cur = (int *)malloc((maxl + 2) * sizeof(int)); memcpy(cur, bc, (maxl + 2) * sizeof(int));
      for (int i = 0; i < qt; i++) tmp[cur[lvl[q[i]]]++] = q[i]; free(cur); }
    /* within each level, sort by degree (insertion, levels are usually small) */
    for (int l = 0; l <= maxl; l++) {
        int a = bc[l], b = bc[l + 1];
        for (int i = a; i < b; i++) { int v = tmp[i], dv = deg[v]; int j = i - 1;
            while (j >= a && deg[tmp[j]] > dv) { tmp[j + 1] = tmp[j]; j--; } tmp[j + 1] = v; }
    }
    int no = 0;
    for (int i = 0; i < qt; i++) order[no++] = tmp[i];
    /* append unvisited */
    for (int i = 0; i < n; i++) if (lvl[i] < 0) order[no++] = i;
    free(q); free(lvl); free(bc); free(tmp);
}

void ord_rcm(const Sp *K, int n, int *perm) {
    int *deg = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) deg[i] = K->p[i + 1] - K->p[i];
    int start = 0;
    for (int i = 1; i < n; i++) if (deg[i] < deg[start]) start = i;
    int *ord1 = (int *)malloc(n * sizeof(int));
    /* pseudo-peripheral refinement (2 rounds) */
    for (int it = 0; it < 2; it++) {
        bfs_order(K, n, start, ord1, deg);
        start = ord1[n - 1];
    }
    bfs_order(K, n, start, ord1, deg);
    for (int i = 0; i < n; i++) perm[i] = ord1[n - 1 - i];
    free(deg); free(ord1);
}

void ord_deg(const Sp *K, int n, int *perm) {
    int *deg = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) deg[i] = K->p[i + 1] - K->p[i];
    for (int i = 0; i < n; i++) perm[i] = i;
    /* counting sort by degree (capped) */
    int maxd = 1; for (int i = 0; i < n; i++) if (deg[i] > maxd) maxd = deg[i];
    if ((long)maxd * 4 < 200000000L) {
        int *bc = (int *)calloc(maxd + 2, sizeof(int));
        for (int i = 0; i < n; i++) bc[deg[i] + 1]++;
        for (int i = 0; i <= maxd; i++) bc[i + 1] += bc[i];
        int *cur = (int *)malloc((maxd + 2) * sizeof(int)); memcpy(cur, bc, (maxd + 2) * sizeof(int));
        for (int i = 0; i < n; i++) perm[cur[deg[i]]++] = i;
        free(bc); free(cur);
    }
    free(deg);
}

/* ---------------- LDL^T ---------------- */
void ldl_init(LDL *F, int N) {
    memset(F, 0, sizeof(*F));
    F->N = N;
    F->perm = (int *)malloc(N * sizeof(int));
    F->pinv = (int *)malloc(N * sizeof(int));
    F->d = (double *)calloc(N, sizeof(double));
    F->Li = (int **)calloc(N, sizeof(int *));
    F->Lx = (double **)calloc(N, sizeof(double *));
    F->Lc = (int *)calloc(N, sizeof(int));
    F->Lcap = (int *)calloc(N, sizeof(int));
    F->parent = (int *)malloc(N * sizeof(int));
    F->mark = (int *)malloc(N * sizeof(int));
    F->stack = (int *)malloc((N + 1) * sizeof(int));
    F->path = (int *)malloc((N + 1) * sizeof(int));
    F->tmark = (int *)malloc(N * sizeof(int));
    F->touched = (int *)malloc(N * sizeof(int));
    F->x = (double *)calloc(N, sizeof(double));
    F->tmp = (double *)calloc(N, sizeof(double));
    for (int i = 0; i < N; i++) { F->parent[i] = -1; F->mark[i] = -1; F->tmark[i] = -1; F->perm[i] = i; F->pinv[i] = i; }
    F->inited = 1;
}
void ldl_free(LDL *F) {
    if (!F->inited) return;
    for (int i = 0; i < F->N; i++) { free(F->Li[i]); free(F->Lx[i]); }
    free(F->perm); free(F->pinv); free(F->d); free(F->Li); free(F->Lx); free(F->Lc); free(F->Lcap);
    free(F->parent); free(F->mark); free(F->stack); free(F->path); free(F->tmark); free(F->touched);
    free(F->x); free(F->tmp);
    memset(F, 0, sizeof(*F));
}
void ldl_setperm(LDL *F, const int *perm) {
    for (int i = 0; i < F->N; i++) { F->perm[i] = perm[i]; F->pinv[perm[i]] = i; }
}
void ldl_reset(LDL *F) {
    for (int i = 0; i < F->N; i++) { F->Lc[i] = 0; F->parent[i] = -1; F->mark[i] = -1; F->tmark[i] = -1; F->d[i] = 0.0; }
    F->nnzL = 0;
}

int ldl_factor(LDL *F, const Sp *K) {
    int N = F->N;
    const int *perm = F->perm, *pinv = F->pinv;
    double *x = F->x;
    int *touched = F->touched;
    ldl_reset(F);
    for (int k = 0; k < N; k++) {
        int nt = 0;
        int c = perm[k];
        for (int p = K->p[c]; p < K->p[c + 1]; p++) {
            int r = K->i[p], i = pinv[r];
            if (i > k) continue;
            if (F->tmark[i] != k) { F->tmark[i] = k; touched[nt++] = i; }
            x[i] += K->x[p];
        }
        int top = N;
        for (int t = 0; t < nt; t++) {
            int i = touched[t];
            if (i == k) continue;
            int len = 0;
            while (i != -1 && F->mark[i] != k) { F->mark[i] = k; F->path[len++] = i; i = F->parent[i]; }
            while (len > 0) F->stack[--top] = F->path[--len];
        }
        for (int px = top; px < N; px++) {
            int i = F->stack[px];
            double di = F->d[i];
            if (di == 0.0) { for (int t = 0; t < nt; t++) x[touched[t]] = 0.0; x[k] = 0.0; return k + 1; }
            double lki = x[i] / di;
            x[i] = 0.0;
            double sc = lki * di;
            x[k] -= lki * sc;
            int cn = F->Lc[i];
            if (sc != 0.0) {
                int *Li = F->Li[i]; double *Lx = F->Lx[i];
                for (int q = 0; q < cn; q++) {
                    int r = Li[q]; if (r > k) break;
                    if (F->tmark[r] != k) { F->tmark[r] = k; touched[nt++] = r; }
                    x[r] -= Lx[q] * sc;
                }
            }
            if (cn == F->Lcap[i]) {
                F->Lcap[i] = F->Lcap[i] ? F->Lcap[i] * 2 : 4;
                F->Li[i] = (int *)realloc(F->Li[i], F->Lcap[i] * sizeof(int));
                F->Lx[i] = (double *)realloc(F->Lx[i], F->Lcap[i] * sizeof(double));
            }
            F->Li[i][cn] = k; F->Lx[i][cn] = lki; F->Lc[i] = cn + 1; F->nnzL++;
            if (F->parent[i] == -1) F->parent[i] = k;
        }
        F->d[k] = x[k];
        for (int t = 0; t < nt; t++) x[touched[t]] = 0.0;
        x[k] = 0.0;
        if (!(F->d[k] > 0.0) && !(F->d[k] < 0.0)) return k + 1;
    }
    return 0;
}

void ldl_solve(LDL *F, const double *b, double *out) {
    int N = F->N;
    double *y = F->tmp;
    for (int k = 0; k < N; k++) y[k] = b[F->perm[k]];
    for (int j = 0; j < N; j++) {
        double yj = y[j];
        if (yj == 0.0) continue;
        int cn = F->Lc[j], *Li = F->Li[j]; double *Lx = F->Lx[j];
        for (int q = 0; q < cn; q++) y[Li[q]] -= Lx[q] * yj;
    }
    for (int k = 0; k < N; k++) y[k] /= F->d[k];
    for (int j = N - 1; j >= 0; j--) {
        double acc = y[j];
        int cn = F->Lc[j], *Li = F->Li[j]; double *Lx = F->Lx[j];
        for (int q = 0; q < cn; q++) acc -= Lx[q] * y[Li[q]];
        y[j] = acc;
    }
    for (int k = 0; k < N; k++) out[F->perm[k]] = y[k];
}
