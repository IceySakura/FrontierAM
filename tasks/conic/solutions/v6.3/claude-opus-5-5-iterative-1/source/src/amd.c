/* Approximate minimum degree ordering: a faithful port of Tim Davis's CSparse cs_amd
   (order 1: AMD of A+A' without the diagonal), including the quotient graph with
   element absorption, approximate external degrees (w-array trick), mass elimination,
   supernode detection by hashing, aggressive absorption, dense-row handling and the
   final postorder of the assembly tree (cs_tdfs).
   Indices are int; all quantities that could overflow are computed in int64. */
#include "amd.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define AMD_FLIP(i) (-(i) - 2)
#define AMD_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define AMD_MIN(a, b) (((a) < (b)) ? (a) : (b))

/* clear w if mark is too small or if further marks could overflow int.
   Invariant on return: w[0..n-1] < mark, and mark + lemax + n + 2 < INT_MAX. */
static int amd_wclear(int64_t mark, int lemax, int *w, int n)
{
    if (mark < 2 || mark + (int64_t)lemax + (int64_t)n + 2 >= (int64_t)INT_MAX) {
        for (int k = 0; k < n; k++)
            if (w[k] != 0) w[k] = 1;
        mark = 2;
    }
    return (int)mark;
}

/* depth-first search and postorder of a tree rooted at node j (cs_tdfs) */
static int amd_tdfs(int j, int k, int *head, const int *next, int *post, int *stack)
{
    int i, p, top = 0;
    stack[0] = j;
    while (top >= 0) {
        p = stack[top];
        i = head[p];
        if (i == -1) {
            top--;
            post[k++] = p;
        } else {
            head[p] = next[i];
            stack[++top] = i;
        }
    }
    return k;
}

int amd_order(int n, const int *Ap, const int *Ai, int *perm)
{
    int *ATp = NULL, *ATi = NULL, *Cp = NULL, *Ci = NULL, *W = NULL, *wk = NULL;
    int *last, *len, *nv, *next, *P, *head, *elen, *degree, *w, *hhead;
    int d, dk, dext, lemax = 0, e, elenk, eln, i, j, k, k1, k2, k3, jlast, ln, dense, nzmax,
        mindeg = 0, nvi, nvj, nvk, mark, wnvi, ok, cnz, nel = 0, p, p1, p2, p3, p4, pj, pk,
        pk1, pk2, pn, q, nzoff;
    int64_t h, cnt64, t64;

    if (n < 0 || !Ap) return -1;
    if (n == 0) return 0;
    if (!perm) return -1;
    if (n == 1) {
        perm[0] = 0;
        return 0;
    }
    if (n > INT_MAX / 16) return -1; /* keep 9*(n+1) and 2n-sized arithmetic safe */

    /* ---- Construct C = pattern of A+A' without diagonal (cs_add(A, A') + cs_fkeep) ---- */
    /* transpose of the off-diagonal pattern: AT(:,i) lists columns j with A(i,j) != 0 */
    ATp = (int *)calloc((size_t)n + 1, sizeof(int));
    wk = (int *)malloc((size_t)n * sizeof(int));
    if (!ATp || !wk) goto fail;
    nzoff = 0;
    for (j = 0; j < n; j++) {
        for (p = Ap[j]; p < Ap[j + 1]; p++) {
            i = Ai[p];
            if (i < 0 || i >= n) goto fail;
            if (i != j) {
                ATp[i + 1]++;
                nzoff++;
            }
        }
    }
    for (i = 0; i < n; i++) ATp[i + 1] += ATp[i];
    ATi = (int *)malloc((size_t)AMD_MAX(nzoff, 1) * sizeof(int));
    if (!ATi) goto fail;
    for (i = 0; i < n; i++) wk[i] = ATp[i];
    for (j = 0; j < n; j++) {
        for (p = Ap[j]; p < Ap[j + 1]; p++) {
            i = Ai[p];
            if (i != j) ATi[wk[i]++] = j;
        }
    }
    /* count entries of C (duplicates removed) */
    Cp = (int *)malloc(((size_t)n + 1) * sizeof(int));
    if (!Cp) goto fail;
    for (i = 0; i < n; i++) wk[i] = -1;
    cnt64 = 0;
    for (j = 0; j < n; j++) {
        Cp[j] = (int)AMD_MIN(cnt64, (int64_t)INT_MAX);
        for (p = Ap[j]; p < Ap[j + 1]; p++) {
            i = Ai[p];
            if (i != j && wk[i] != j) {
                wk[i] = j;
                cnt64++;
            }
        }
        for (p = ATp[j]; p < ATp[j + 1]; p++) {
            i = ATi[p];
            if (wk[i] != j) {
                wk[i] = j;
                cnt64++;
            }
        }
    }
    t64 = cnt64 + cnt64 / 5 + 2 * (int64_t)n; /* add elbow room to C */
    if (t64 + 2 * (int64_t)n + 8 >= (int64_t)INT_MAX) goto fail;
    cnz = (int)cnt64;
    Cp[n] = cnz;
    nzmax = (int)t64;
    Ci = (int *)malloc((size_t)AMD_MAX(nzmax, 1) * sizeof(int));
    if (!Ci) goto fail;
    for (i = 0; i < n; i++) wk[i] = -1;
    q = 0;
    for (j = 0; j < n; j++) {
        for (p = Ap[j]; p < Ap[j + 1]; p++) {
            i = Ai[p];
            if (i != j && wk[i] != j) {
                wk[i] = j;
                Ci[q++] = i;
            }
        }
        for (p = ATp[j]; p < ATp[j + 1]; p++) {
            i = ATi[p];
            if (wk[i] != j) {
                wk[i] = j;
                Ci[q++] = i;
            }
        }
    }
    free(ATi);
    ATi = NULL;
    free(ATp);
    ATp = NULL;
    free(wk);
    wk = NULL;

    dense = (int)AMD_MAX(16.0, 10.0 * sqrt((double)n)); /* find dense threshold */
    dense = AMD_MIN(n - 2, dense);

    W = (int *)malloc(9 * ((size_t)n + 1) * sizeof(int));
    if (!W) goto fail;
    len = W;
    nv = W + (n + 1);
    next = W + 2 * (n + 1);
    head = W + 3 * (n + 1);
    elen = W + 4 * (n + 1);
    degree = W + 5 * (n + 1);
    w = W + 6 * (n + 1);
    hhead = W + 7 * (n + 1);
    P = W + 8 * (n + 1);
    last = P; /* use P as workspace for last */

    /* --- Initialize quotient graph ---------------------------------------- */
    for (k = 0; k < n; k++) len[k] = Cp[k + 1] - Cp[k];
    len[n] = 0;
    for (i = 0; i <= n; i++) {
        head[i] = -1; /* degree list i is empty */
        last[i] = -1;
        next[i] = -1;
        hhead[i] = -1; /* hash list i is empty */
        nv[i] = 1;     /* node i is just one node */
        w[i] = 1;      /* node i is alive */
        elen[i] = 0;   /* Ek of node i is empty */
        degree[i] = len[i];
    }
    mark = amd_wclear(0, 0, w, n); /* clear w */
    elen[n] = -2;                  /* n is a dead element */
    Cp[n] = -1;                    /* n is a root of assembly tree */
    w[n] = 0;                      /* n is a dead element */

    /* --- Initialize degree lists ------------------------------------------ */
    for (i = 0; i < n; i++) {
        d = degree[i];
        if (d == 0) { /* node i is empty */
            elen[i] = -2;
            nel++;
            Cp[i] = -1;
            w[i] = 0;
        } else if (d > dense) { /* node i is dense */
            nv[i] = 0;          /* absorb i into element n */
            elen[i] = -1;       /* node i is dead */
            nel++;
            Cp[i] = AMD_FLIP(n);
            nv[n]++;
        } else {
            if (head[d] != -1) last[head[d]] = i;
            next[i] = head[d]; /* put node i in degree list d */
            head[d] = i;
        }
    }

    while (nel < n) { /* while (selecting pivots) do */
        /* --- Select node of minimum approximate degree -------------------- */
        for (k = -1; mindeg < n && (k = head[mindeg]) == -1; mindeg++)
            ;
        if (next[k] != -1) last[next[k]] = -1;
        head[mindeg] = next[k]; /* remove k from degree list */
        elenk = elen[k];        /* elenk = |Ek| */
        nvk = nv[k];            /* # of nodes k represents */
        nel += nvk;             /* nv[k] nodes of A eliminated */

        /* --- Garbage collection ------------------------------------------- */
        if (elenk > 0 && (int64_t)cnz + mindeg >= nzmax) {
            for (j = 0; j < n; j++) {
                if ((p = Cp[j]) >= 0) { /* j is a live node or element */
                    Cp[j] = Ci[p];      /* save first entry of object */
                    Ci[p] = AMD_FLIP(j); /* first entry is now AMD_FLIP(j) */
                }
            }
            for (q = 0, p = 0; p < cnz;) { /* scan all of memory */
                if ((j = AMD_FLIP(Ci[p++])) >= 0) { /* found object j */
                    Ci[q] = Cp[j];                   /* restore first entry of object */
                    Cp[j] = q++;                     /* new pointer to object j */
                    for (k3 = 0; k3 < len[j] - 1; k3++) Ci[q++] = Ci[p++];
                }
            }
            cnz = q; /* Ci [cnz...nzmax-1] now free */
        }

        /* --- Construct new element ---------------------------------------- */
        dk = 0;
        nv[k] = -nvk; /* flag k as in Lk */
        p = Cp[k];
        pk1 = (elenk == 0) ? p : cnz; /* do in place if elen[k] == 0 */
        pk2 = pk1;
        for (k1 = 1; k1 <= elenk + 1; k1++) {
            if (k1 > elenk) {
                e = k;               /* search the nodes in k */
                pj = p;              /* list of nodes starts at Ci[pj]*/
                ln = len[k] - elenk; /* length of list of nodes in k */
            } else {
                e = Ci[p++]; /* search the nodes in e */
                pj = Cp[e];
                ln = len[e]; /* length of list of nodes in e */
            }
            for (k2 = 1; k2 <= ln; k2++) {
                i = Ci[pj++];
                if ((nvi = nv[i]) <= 0) continue; /* node i dead, or seen */
                dk += nvi;                        /* degree[Lk] += size of node i */
                nv[i] = -nvi;                     /* negate nv[i] to denote i in Lk*/
                Ci[pk2++] = i;                    /* place i in Lk */
                if (next[i] != -1) last[next[i]] = last[i];
                if (last[i] != -1) { /* remove i from degree list */
                    next[last[i]] = next[i];
                } else {
                    head[degree[i]] = next[i];
                }
            }
            if (e != k) {
                Cp[e] = AMD_FLIP(k); /* absorb e into k */
                w[e] = 0;            /* e is now a dead element */
            }
        }
        if (elenk != 0) cnz = pk2; /* Ci [cnz...nzmax] is free */
        degree[k] = dk;            /* external degree of k - |Lk\i| */
        Cp[k] = pk1;               /* element k is in Ci[pk1..pk2-1] */
        len[k] = pk2 - pk1;
        elen[k] = -2; /* k is now an element */

        /* --- Find set differences ----------------------------------------- */
        mark = amd_wclear(mark, lemax, w, n); /* clear w if necessary */
        for (pk = pk1; pk < pk2; pk++) {      /* scan 1: find |Le\Lk| */
            i = Ci[pk];
            if ((eln = elen[i]) <= 0) continue; /* skip if elen[i] empty */
            nvi = -nv[i];                       /* nv [i] was negated */
            wnvi = mark - nvi;
            for (p = Cp[i]; p <= Cp[i] + eln - 1; p++) { /* scan Ei */
                e = Ci[p];
                if (w[e] >= mark) {
                    w[e] -= nvi; /* decrement |Le\Lk| */
                } else if (w[e] != 0) {        /* ensure e is a live element */
                    w[e] = degree[e] + wnvi; /* 1st time e seen in scan 1 */
                }
            }
        }

        /* --- Degree update ------------------------------------------------ */
        for (pk = pk1; pk < pk2; pk++) { /* scan2: degree update */
            i = Ci[pk];                  /* consider node i in Lk */
            p1 = Cp[i];
            p2 = p1 + elen[i] - 1;
            pn = p1;
            for (h = 0, d = 0, p = p1; p <= p2; p++) { /* scan Ei */
                e = Ci[p];
                if (w[e] != 0) {         /* e is an unabsorbed element */
                    dext = w[e] - mark;  /* dext = |Le\Lk| */
                    if (dext > 0) {
                        d += dext;     /* sum up the set differences */
                        Ci[pn++] = e;  /* keep e in Ei */
                        h += e;        /* compute the hash of node i */
                    } else {
                        Cp[e] = AMD_FLIP(k); /* aggressive absorb. e->k */
                        w[e] = 0;            /* e is a dead element */
                    }
                }
            }
            elen[i] = pn - p1 + 1; /* elen[i] = |Ei| */
            p3 = pn;
            p4 = p1 + len[i];
            for (p = p2 + 1; p < p4; p++) { /* prune edges in Ai */
                j = Ci[p];
                if ((nvj = nv[j]) <= 0) continue; /* node j dead or in Lk */
                d += nvj;                         /* degree(i) += |j| */
                Ci[pn++] = j;                     /* place j in node list of i */
                h += j;                           /* compute hash for node i */
            }
            if (d == 0) { /* check for mass elimination */
                Cp[i] = AMD_FLIP(k); /* absorb i into k */
                nvi = -nv[i];
                dk -= nvi;  /* |Lk| -= |i| */
                nvk += nvi; /* |k| += nv[i] */
                nel += nvi;
                nv[i] = 0;
                elen[i] = -1; /* node i is dead */
            } else {
                degree[i] = AMD_MIN(degree[i], d); /* update degree(i) */
                Ci[pn] = Ci[p3];                   /* move first node to end */
                Ci[p3] = Ci[p1];                   /* move 1st el. to end of Ei */
                Ci[p1] = k;                        /* add k as 1st element in of Ei */
                len[i] = pn - p1 + 1;              /* new len of adj. list of node i */
                h = ((h < 0) ? (-h) : h) % n;      /* finalize hash of i */
                next[i] = hhead[h];                /* place i in hash bucket */
                hhead[h] = i;
                last[i] = (int)h; /* save hash of i in last[i] */
            }
        } /* scan2 is done */
        degree[k] = dk; /* finalize |Lk| */
        lemax = AMD_MAX(lemax, dk);
        mark = amd_wclear((int64_t)mark + lemax, lemax, w, n); /* clear w */

        /* --- Supernode detection ------------------------------------------ */
        for (pk = pk1; pk < pk2; pk++) {
            i = Ci[pk];
            if (nv[i] >= 0) continue; /* skip if i is dead */
            h = last[i];              /* scan hash bucket of node i */
            i = hhead[h];
            hhead[h] = -1; /* hash bucket will be empty */
            for (; i != -1 && next[i] != -1; i = next[i], mark++) {
                ln = len[i];
                eln = elen[i];
                for (p = Cp[i] + 1; p <= Cp[i] + ln - 1; p++) w[Ci[p]] = mark;
                jlast = i;
                for (j = next[i]; j != -1;) { /* compare i with all j */
                    ok = (len[j] == ln) && (elen[j] == eln);
                    for (p = Cp[j] + 1; ok && p <= Cp[j] + ln - 1; p++) {
                        if (w[Ci[p]] != mark) ok = 0; /* compare i and j*/
                    }
                    if (ok) {                /* i and j are identical */
                        Cp[j] = AMD_FLIP(i); /* absorb j into i */
                        nv[i] += nv[j];
                        nv[j] = 0;
                        elen[j] = -1; /* node j is dead */
                        j = next[j];  /* delete j from hash bucket */
                        next[jlast] = j;
                    } else {
                        jlast = j; /* j and i are different */
                        j = next[j];
                    }
                }
            }
        }

        /* --- Finalize new element------------------------------------------ */
        for (p = pk1, pk = pk1; pk < pk2; pk++) { /* finalize Lk */
            i = Ci[pk];
            if ((nvi = -nv[i]) <= 0) continue; /* skip if i is dead */
            nv[i] = nvi;                       /* restore nv[i] */
            d = degree[i] + dk - nvi;          /* compute external degree(i) */
            d = AMD_MIN(d, n - nel - nvi);
            if (head[d] != -1) last[head[d]] = i;
            next[i] = head[d]; /* put i back in degree list */
            last[i] = -1;
            head[d] = i;
            mindeg = AMD_MIN(mindeg, d); /* find new minimum degree */
            degree[i] = d;
            Ci[p++] = i; /* place i in Lk */
        }
        nv[k] = nvk;                    /* # nodes absorbed into k */
        if ((len[k] = p - pk1) == 0) {  /* length of adj list of element k*/
            Cp[k] = -1;                 /* k is a root of the tree */
            w[k] = 0;                   /* k is now a dead element */
        }
        if (elenk != 0) cnz = p; /* free unused space in Lk */
    }

    /* --- Postordering ----------------------------------------------------- */
    for (i = 0; i < n; i++) Cp[i] = AMD_FLIP(Cp[i]); /* fix assembly tree */
    for (j = 0; j <= n; j++) head[j] = -1;
    for (j = n; j >= 0; j--) { /* place unordered nodes in lists */
        if (nv[j] > 0) continue; /* skip if j is an element */
        next[j] = head[Cp[j]];   /* place j in list of its parent */
        head[Cp[j]] = j;
    }
    for (e = n; e >= 0; e--) { /* place elements in lists */
        if (nv[e] <= 0) continue; /* skip unless e is an element */
        if (Cp[e] != -1) {
            next[e] = head[Cp[e]]; /* place e in list of its parent */
            head[Cp[e]] = e;
        }
    }
    for (k = 0, i = 0; i <= n; i++) { /* postorder the assembly tree */
        if (Cp[i] == -1) k = amd_tdfs(i, k, head, next, P, w);
    }
    /* P[0..n] is a permutation of 0..n with P[n] == n (element n is the last root) */
    {
        int kk = 0;
        for (k = 0; k <= n; k++)
            if (P[k] != n) perm[kk++] = P[k];
        if (kk != n) goto fail;
    }
    free(Cp);
    free(Ci);
    free(W);
    return 0;

fail:
    free(ATp);
    free(ATi);
    free(wk);
    free(Cp);
    free(Ci);
    free(W);
    return -1;
}
