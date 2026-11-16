/*
 * ldl.c -- Sparse symmetric LDL^T factorization with AMD ordering.
 *
 *  - amd_order: approximate minimum degree ordering (Amestoy, Davis, Duff),
 *    quotient graph, approximate external degrees, element absorption
 *    (incl. aggressive absorption), mass elimination, supervariable detection
 *    by hashing, dense row/column postponing, assembly tree postordering.
 *  - ldl_create: symbolic analysis (permuted upper pattern, entry map,
 *    elimination tree, column counts, allocation of L).
 *  - ldl_factor: QDLDL-style up-looking numeric LDL^T with dynamic
 *    regularization; no allocation.
 *  - ldl_solve: permuted triangular solves.
 *
 * All indices are int64_t.
 */
#include "ldl.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int64_t Int;

#define EMPTY (-1)
#define FLIP(i) (-(i) - 2)
#define LMAX(a, b) (((a) > (b)) ? (a) : (b))
#define LMIN(a, b) (((a) < (b)) ? (a) : (b))

static void *xmalloc(size_t count, size_t size)
{
    if (count == 0) count = 1;
    if (count > SIZE_MAX / size) return NULL;
    return malloc(count * size);
}

/* ========================================================================= */
/* AMD                                                                       */
/* ========================================================================= */

static Int amd_clear_flag(Int wflg, Int wbig, Int *W, Int n)
{
    Int x;
    if (wflg < 2 || wflg >= wbig) {
        for (x = 0; x < n; x++) {
            if (W[x] != 0) W[x] = 1;
        }
        wflg = 2;
    }
    /* at this point, W[0..n-1] < wflg holds */
    return wflg;
}

/* non-recursive depth-first postorder of the tree rooted at root */
static Int amd_post_tree(Int root, Int k, Int *Child, const Int *Sibling, Int *Order, Int *Stack)
{
    Int f, head, h, i;
    head = 0;
    Stack[0] = root;
    while (head >= 0) {
        i = Stack[head];
        if (Child[i] != EMPTY) {
            /* push children in reverse order so the first child is popped first */
            for (f = Child[i]; f != EMPTY; f = Sibling[f]) head++;
            h = head;
            for (f = Child[i]; f != EMPTY; f = Sibling[f]) Stack[h--] = f;
            Child[i] = EMPTY;
        } else {
            head--;
            Order[i] = k++;
        }
    }
    return k;
}

static void amd_postorder(Int nn, const Int *Parent, const Int *Nv, const Int *Fsize, Int *Order,
                          Int *Child, Int *Sibling, Int *Stack)
{
    Int i, j, k, parent, frsize, f, fprev, maxfrsize, bigf, bigfprev, fnext;

    for (j = 0; j < nn; j++) {
        Child[j] = EMPTY;
        Sibling[j] = EMPTY;
    }
    /* place the children in link lists - bigger elements tend to be last */
    for (j = nn - 1; j >= 0; j--) {
        if (Nv[j] > 0) {
            parent = Parent[j];
            if (parent != EMPTY) {
                Sibling[j] = Child[parent];
                Child[parent] = j;
            }
        }
    }
    /* place the largest child last in the list of children for each node */
    for (i = 0; i < nn; i++) {
        if (Nv[i] > 0 && Child[i] != EMPTY) {
            fprev = EMPTY;
            maxfrsize = EMPTY;
            bigfprev = EMPTY;
            bigf = EMPTY;
            for (f = Child[i]; f != EMPTY; f = Sibling[f]) {
                frsize = Fsize[f];
                if (frsize >= maxfrsize) {
                    maxfrsize = frsize;
                    bigfprev = fprev;
                    bigf = f;
                }
                fprev = f;
            }
            fnext = Sibling[bigf];
            if (fnext != EMPTY) {
                if (bigfprev == EMPTY) {
                    Child[i] = fnext;
                } else {
                    Sibling[bigfprev] = fnext;
                }
                Sibling[bigf] = EMPTY;
                Sibling[fprev] = bigf;
            }
        }
    }
    for (i = 0; i < nn; i++) Order[i] = EMPTY;
    k = 0;
    for (i = 0; i < nn; i++) {
        if (Parent[i] == EMPTY && Nv[i] > 0) {
            k = amd_post_tree(i, k, Child, Sibling, Order, Stack);
        }
    }
}

/*
 * Core AMD routine (follows the structure of SuiteSparse AMD's amd_2).
 * On input, Iw[Pe[i] .. Pe[i]+Len[i]-1] holds the adjacency of i (no
 * diagonal, no duplicates), Iw[pfree..iwlen-1] is free, iwlen >= pfree + n.
 * On output, Last[k] = k-th pivot (permutation), Next[i] = inverse.
 */
static int amd_2(Int n, Int *Pe, Int *Iw, Int *Len, Int iwlen, Int pfree, Int *Nv, Int *Next,
                 Int *Last, Int *Head, Int *Elen, Int *Degree, Int *W, Int dense_in, double budget,
                 const Int *deg0, Int *thr_out, Int *ndn)
{
    Int deg, degme, dext, lemax, e, elenme, eln, i, ilast, inext, j, jlast, jnext, k, knt1,
        knt2, knt3, lenj, ln, me, mindeg, nel, nleft, nvi, nvj, nvpiv, slenme, wbig, we, wflg,
        wnvi, ok, ndense, dense;
    uint64_t hash;
    Int p, p1, p2, p3, p4, pdst, pend, pj, pme, pme1, pme2, pn, psrc;
    double alpha = 10.0, dd, work = 0.0, budget_hard = 100.0 / 30.0 * budget;

    /* dense threshold: max(16, 10*sqrt(n)), at most n (SuiteSparse default),
       unless an explicit threshold is given */
    if (dense_in >= 0) {
        dense = LMIN(n, dense_in);
    } else {
        dd = alpha * sqrt((double)n);
        if (dd > (double)n) dd = (double)n;
        dense = (Int)dd;
        dense = LMAX(16, dense);
        dense = LMIN(n, dense);
    }
    if (thr_out) *thr_out = EMPTY;

    me = EMPTY;
    mindeg = 0;
    nel = 0;
    lemax = 0;

    for (i = 0; i < n; i++) {
        Last[i] = EMPTY;
        Head[i] = EMPTY;
        Next[i] = EMPTY;
        Nv[i] = 1;
        W[i] = 1;
        Elen[i] = 0;
        Degree[i] = Len[i];
    }

    wbig = INT64_MAX - n;
    wflg = amd_clear_flag(0, wbig, W, n);

    /* number of dense neighbours of each node (used to order the members of
       each eliminated group: nodes adjacent to dense rows go last) */
    ndense = 0;
    for (i = 0; i < n; i++) ndense += (Len[i] > dense);
    if (ndense > 0 && ndn) {
        for (i = 0; i < n; i++) {
            Int cnt = 0;
            for (p = Pe[i]; p < Pe[i] + Len[i]; p++) cnt += (Len[Iw[p]] > dense);
            ndn[i] = cnt;
        }
    }

    /* initialize degree lists and eliminate dense and empty rows */
    ndense = 0;
    for (i = 0; i < n; i++) {
        deg = Degree[i];
        if (deg == 0) {
            /* empty variable: eliminate at once (treated as an element) */
            Elen[i] = FLIP(1);
            nel++;
            Pe[i] = EMPTY;
            W[i] = 0;
        } else if (deg > dense) {
            /* dense variable: unordered, ordered last */
            ndense++;
            Nv[i] = 0;
            Elen[i] = EMPTY;
            nel++;
            Pe[i] = EMPTY;
        } else {
            inext = Head[deg];
            if (inext != EMPTY) Last[inext] = i;
            Next[i] = inext;
            Head[deg] = i;
        }
    }
    (void)ndense;

    while (nel < n) {
        /* ------------------------------------------------------------- */
        /* get pivot of minimum degree */
        /* ------------------------------------------------------------- */
        for (deg = mindeg; deg < n; deg++) {
            me = Head[deg];
            if (me != EMPTY) break;
        }
        mindeg = deg;

        inext = Next[me];
        if (inext != EMPTY) Last[inext] = EMPTY;
        Head[deg] = inext;

        elenme = Elen[me];
        nvpiv = Nv[me];
        nel += nvpiv;

        /* ------------------------------------------------------------- */
        /* construct new element */
        /* ------------------------------------------------------------- */
        Nv[me] = -nvpiv;
        degme = 0;

        if (elenme == 0) {
            /* construct the new element in place */
            pme1 = Pe[me];
            pme2 = pme1 - 1;
            work += (double)Len[me];
            for (p = pme1; p <= pme1 + Len[me] - 1; p++) {
                i = Iw[p];
                nvi = Nv[i];
                if (nvi > 0) {
                    degme += nvi;
                    Nv[i] = -nvi;
                    Iw[++pme2] = i;
                    /* remove i from degree list */
                    ilast = Last[i];
                    inext = Next[i];
                    if (inext != EMPTY) Last[inext] = ilast;
                    if (ilast != EMPTY) {
                        Next[ilast] = inext;
                    } else {
                        Head[Degree[i]] = inext;
                    }
                }
            }
        } else {
            /* construct the new element in empty space, Iw[pfree ...] */
            p = Pe[me];
            pme1 = pfree;
            slenme = Len[me] - elenme;

            for (knt1 = 1; knt1 <= elenme + 1; knt1++) {
                if (knt1 > elenme) {
                    /* search the supervariables in me */
                    e = me;
                    pj = p;
                    ln = slenme;
                } else {
                    /* search the elements in me */
                    e = Iw[p++];
                    pj = Pe[e];
                    ln = Len[e];
                }
                work += (double)ln;

                for (knt2 = 1; knt2 <= ln; knt2++) {
                    i = Iw[pj++];
                    nvi = Nv[i];
                    if (nvi > 0) {
                        /* compress Iw, if necessary */
                        if (pfree >= iwlen) {
                            Pe[me] = p;
                            Len[me] -= knt1;
                            if (Len[me] == 0) Pe[me] = EMPTY;
                            Pe[e] = pj;
                            Len[e] = ln - knt2;
                            if (Len[e] == 0) Pe[e] = EMPTY;

                            /* store first entry of each object in Pe,
                               FLIP the first entry in each object */
                            for (j = 0; j < n; j++) {
                                pn = Pe[j];
                                if (pn >= 0) {
                                    Pe[j] = Iw[pn];
                                    Iw[pn] = FLIP(j);
                                }
                            }

                            psrc = 0;
                            pdst = 0;
                            pend = pme1 - 1;
                            while (psrc <= pend) {
                                j = FLIP(Iw[psrc++]);
                                if (j >= 0) {
                                    Iw[pdst] = Pe[j];
                                    Pe[j] = pdst++;
                                    lenj = Len[j];
                                    for (knt3 = 0; knt3 <= lenj - 2; knt3++) {
                                        Iw[pdst++] = Iw[psrc++];
                                    }
                                }
                            }

                            /* move the new partially-constructed element */
                            p1 = pdst;
                            for (psrc = pme1; psrc <= pfree - 1; psrc++) {
                                Iw[pdst++] = Iw[psrc];
                            }
                            pme1 = p1;
                            pfree = pdst;
                            pj = Pe[e];
                            p = Pe[me];
                        }

                        degme += nvi;
                        Nv[i] = -nvi;
                        Iw[pfree++] = i;

                        /* remove i from degree list */
                        ilast = Last[i];
                        inext = Next[i];
                        if (inext != EMPTY) Last[inext] = ilast;
                        if (ilast != EMPTY) {
                            Next[ilast] = inext;
                        } else {
                            Head[Degree[i]] = inext;
                        }
                    }
                }

                if (e != me) {
                    /* element e is absorbed into new element me */
                    Pe[e] = FLIP(me);
                    W[e] = 0;
                }
            }
            pme2 = pfree - 1;
        }

        /* me has now been converted into an element in Iw[pme1..pme2] */
        Degree[me] = degme;
        Pe[me] = pme1;
        Len[me] = pme2 - pme1 + 1;
        Elen[me] = FLIP(nvpiv + degme);

        wflg = amd_clear_flag(wflg, wbig, W, n);

        /* ------------------------------------------------------------- */
        /* Scan 1: compute (W[e] - wflg) = |Le \ Lme| for all elements */
        /* ------------------------------------------------------------- */
        for (pme = pme1; pme <= pme2; pme++) {
            i = Iw[pme];
            eln = Elen[i];
            if (eln > 0) {
                nvi = -Nv[i];
                wnvi = wflg - nvi;
                work += (double)eln;
                for (p = Pe[i]; p <= Pe[i] + eln - 1; p++) {
                    e = Iw[p];
                    we = W[e];
                    if (we >= wflg) {
                        we -= nvi;
                    } else if (we != 0) {
                        we = Degree[e] + wnvi;
                    }
                    W[e] = we;
                }
            }
        }

        /* ------------------------------------------------------------- */
        /* Scan 2: degree update and element absorption */
        /* ------------------------------------------------------------- */
        for (pme = pme1; pme <= pme2; pme++) {
            i = Iw[pme];
            p1 = Pe[i];
            p2 = p1 + Elen[i] - 1;
            pn = p1;
            hash = 0;
            deg = 0;
            work += (double)Len[i];

            /* scan the element list of i (aggressive absorption) */
            for (p = p1; p <= p2; p++) {
                e = Iw[p];
                we = W[e];
                if (we != 0) {
                    dext = we - wflg;
                    if (dext > 0) {
                        deg += dext;
                        Iw[pn++] = e;
                        hash += (uint64_t)e;
                    } else {
                        /* external degree of e is zero: absorb e into me */
                        Pe[e] = FLIP(me);
                        W[e] = 0;
                    }
                }
            }

            /* count the number of elements in i (including me) */
            Elen[i] = pn - p1 + 1;

            /* scan the supervariables in the list of i */
            p3 = pn;
            p4 = p1 + Len[i];
            for (p = p2 + 1; p < p4; p++) {
                j = Iw[p];
                nvj = Nv[j];
                if (nvj > 0) {
                    deg += nvj;
                    Iw[pn++] = j;
                    hash += (uint64_t)j;
                }
            }

            if (Elen[i] == 1 && p3 == pn) {
                /* mass elimination */
                Pe[i] = FLIP(me);
                nvi = -Nv[i];
                degme -= nvi;
                nvpiv += nvi;
                nel += nvi;
                Nv[i] = 0;
                Elen[i] = EMPTY;
            } else {
                /* update the upper-bound degree of i */
                Degree[i] = LMIN(Degree[i], deg);

                /* add me to the list for i */
                Iw[pn] = Iw[p3];
                Iw[p3] = Iw[p1];
                Iw[p1] = me;
                Len[i] = pn - p1 + 1;

                /* place in hash bucket; save hash key in Last[i] */
                hash = hash % (uint64_t)n;
                j = Head[hash];
                if (j <= EMPTY) {
                    Next[i] = FLIP(j);
                    Head[hash] = FLIP(i);
                } else {
                    Next[i] = Last[j];
                    Last[j] = i;
                }
                Last[i] = (Int)hash;
            }
        }

        Degree[me] = degme;

        /* clear the counter array W by incrementing wflg */
        lemax = LMAX(lemax, degme);
        wflg += lemax;
        wflg = amd_clear_flag(wflg, wbig, W, n);

        /* ------------------------------------------------------------- */
        /* supervariable detection */
        /* ------------------------------------------------------------- */
        for (pme = pme1; pme <= pme2; pme++) {
            i = Iw[pme];
            if (Nv[i] < 0) {
                /* i is a principal variable in Lme; empty its hash bucket */
                hash = (uint64_t)Last[i];
                j = Head[hash];
                if (j == EMPTY) {
                    i = EMPTY;
                } else if (j < EMPTY) {
                    i = FLIP(j);
                    Head[hash] = EMPTY;
                } else {
                    i = Last[j];
                    Last[j] = EMPTY;
                }

                while (i != EMPTY && Next[i] != EMPTY) {
                    ln = Len[i];
                    eln = Elen[i];
                    /* do not flag the first element in the list (me) */
                    for (p = Pe[i] + 1; p <= Pe[i] + ln - 1; p++) W[Iw[p]] = wflg;
                    work += (double)ln;

                    jlast = i;
                    j = Next[i];
                    while (j != EMPTY) {
                        ok = (Len[j] == ln) && (Elen[j] == eln);
                        work += 1.0 + (ok ? (double)ln : 0.0);
                        for (p = Pe[j] + 1; ok && p <= Pe[j] + ln - 1; p++) {
                            if (W[Iw[p]] != wflg) ok = 0;
                        }
                        if (ok) {
                            /* j can be absorbed into i */
                            Pe[j] = FLIP(i);
                            Nv[i] += Nv[j];
                            Nv[j] = 0;
                            Elen[j] = EMPTY;
                            j = Next[j];
                            Next[jlast] = j;
                        } else {
                            jlast = j;
                            j = Next[j];
                        }
                    }
                    wflg++;
                    i = Next[i];
                }
            }
        }

        /* ------------------------------------------------------------- */
        /* restore degree lists and remove nonprincipal supervariables */
        /* ------------------------------------------------------------- */
        p = pme1;
        nleft = n - nel;
        for (pme = pme1; pme <= pme2; pme++) {
            i = Iw[pme];
            nvi = -Nv[i];
            if (nvi > 0) {
                Nv[i] = nvi;
                deg = Degree[i] + degme - nvi;
                deg = LMIN(deg, nleft - nvi);
                if (deg < 0) deg = 0;
                inext = Head[deg];
                if (inext != EMPTY) Last[inext] = i;
                Next[i] = inext;
                Last[i] = EMPTY;
                Head[deg] = i;
                mindeg = LMIN(mindeg, deg);
                Degree[i] = deg;
                Iw[p++] = i;
            }
        }

        /* ------------------------------------------------------------- */
        /* finalize the new element */
        /* ------------------------------------------------------------- */
        Nv[me] = nvpiv;
        Len[me] = p - pme1;
        if (Len[me] == 0) {
            Pe[me] = EMPTY;
            W[me] = 0;
        }
        if (elenme != 0) {
            pfree = p;
        }

        if (budget > 0.0 && work > budget && nel < n) {
            /* too much work: find the original degrees of the variables with
               the longest element lists (the "moderately dense" rows). Abort at
               the soft budget only if they are clearly dense-like (degree at
               least 2*sqrt(n)); always abort at the hard budget. */
            Int emax = 0, mind = n;
            for (i = 0; i < n; i++)
                if (Nv[i] > 0 && Elen[i] > emax) emax = Elen[i];
            if (emax >= 16) {
                for (i = 0; i < n; i++)
                    if (Nv[i] > 0 && Elen[i] >= emax / 4 && deg0[i] < mind) mind = deg0[i];
            }
            if (work > budget_hard || (emax >= 16 && (double)mind >= 2.0 * sqrt((double)n))) {
                if (thr_out && emax >= 16) *thr_out = LMAX(16, mind / 2);
                return 1;
            }
            budget = budget_hard;
        }
    }

    /* ----------------------------------------------------------------- */
    /* postordering */
    /* ----------------------------------------------------------------- */
    for (i = 0; i < n; i++) Pe[i] = FLIP(Pe[i]);
    for (i = 0; i < n; i++) Elen[i] = FLIP(Elen[i]);

    /* compress the paths of the variables */
    for (i = 0; i < n; i++) {
        if (Nv[i] == 0) {
            j = Pe[i];
            if (j == EMPTY) continue; /* dense variable */
            while (Nv[j] == 0) j = Pe[j];
            e = j;
            j = i;
            while (Nv[j] == 0) {
                jnext = Pe[j];
                Pe[j] = e;
                j = jnext;
            }
        }
    }

    amd_postorder(n, Pe, Nv, Elen, W, Head, Next, Last);

    for (k = 0; k < n; k++) {
        Head[k] = EMPTY;
        Next[k] = EMPTY;
    }
    for (e = 0; e < n; e++) {
        k = W[e];
        if (k != EMPTY) Head[k] = e;
    }
    nel = 0;
    for (k = 0; k < n; k++) {
        e = Head[k];
        if (e == EMPTY) break;
        Next[e] = nel;
        nel += Nv[e];
    }
    /* order non-principal variables (dense, and those merged into supervariables) */
    if (ndense > 0 && ndn) {
        /* within each group: first the members without dense neighbours, then
           the others (each class: absorbed variables in index order, then the
           principal variable, as in SuiteSparse) */
        int pass;
        for (e = 0; e < n; e++) Degree[e] = (Nv[e] > 0) ? Next[e] : EMPTY;
        for (pass = 0; pass < 2; pass++) {
            for (i = 0; i < n; i++) {
                if (Nv[i] == 0 && Pe[i] != EMPTY && ((ndn[i] > 0) == pass)) Next[i] = Degree[Pe[i]]++;
            }
            for (e = 0; e < n; e++) {
                if (Nv[e] > 0 && ((ndn[e] > 0) == pass)) Next[e] = Degree[e]++;
            }
        }
        for (i = 0; i < n; i++) {
            if (Nv[i] == 0 && Pe[i] == EMPTY) Next[i] = nel++;
        }
    } else {
        for (i = 0; i < n; i++) {
            if (Nv[i] == 0) {
                e = Pe[i];
                if (e != EMPTY) {
                    Next[i] = Next[e];
                    Next[e]++;
                } else {
                    Next[i] = nel++;
                }
            }
        }
    }
    for (i = 0; i < n; i++) {
        k = Next[i];
        Last[k] = i;
    }
    return 0;
}

/* pattern of A+A' without diagonal and duplicates: Iw[Pe[i] .. Pe[i]+Len[i]-1];
   returns pfree */
static Int amd_pattern(Int n, const Int *Kp, const Int *Ki, Int *Len, Int *Pe, Int *Iw, Int *W,
                       Int *Head)
{
    Int i, j, p, p1, p2, pfree;
    memset(Len, 0, (size_t)n * sizeof(Int));
    for (j = 0; j < n; j++) {
        for (p = Kp[j]; p < Kp[j + 1]; p++) {
            i = Ki[p];
            if (i != j) {
                Len[i]++;
                Len[j]++;
            }
        }
    }
    Pe[0] = 0;
    for (i = 0; i < n; i++) Pe[i + 1] = Pe[i] + Len[i];
    for (i = 0; i < n; i++) Head[i] = Pe[i]; /* insertion pointers */
    for (j = 0; j < n; j++) {
        for (p = Kp[j]; p < Kp[j + 1]; p++) {
            i = Ki[p];
            if (i != j) {
                Iw[Head[i]++] = j;
                Iw[Head[j]++] = i;
            }
        }
    }
    /* remove duplicates and compact */
    for (i = 0; i < n; i++) W[i] = EMPTY;
    pfree = 0;
    for (i = 0; i < n; i++) {
        p1 = Pe[i];
        p2 = p1 + Len[i];
        Pe[i] = pfree;
        for (p = p1; p < p2; p++) {
            j = Iw[p];
            if (W[j] != i) {
                W[j] = i;
                Iw[pfree++] = j;
            }
        }
        Len[i] = pfree - Pe[i];
    }
    return pfree;
}

int64_t amd_order(int64_t n, const int64_t *Kp, const int64_t *Ki, int64_t *perm)
{
    Int i, j, p, nz, iwlen, pfree, status = -1, thr = EMPTY;
    Int *Len = NULL, *Pe = NULL, *Iw = NULL, *Nv = NULL, *Next = NULL, *Last = NULL,
        *Head = NULL, *Elen = NULL, *Degree = NULL, *W = NULL, *deg0 = NULL, *ndn = NULL;
    double budget;

    if (n < 0 || perm == NULL) return -1;
    if (n == 0) return 0;
    if (Kp == NULL) return -1;
    for (j = 0; j < n; j++) {
        if (Kp[j] < 0 || Kp[j + 1] < Kp[j]) return -1;
    }
    if (Ki == NULL && Kp[n] > Kp[0]) return -1;
    nz = 0;
    for (j = 0; j < n; j++) {
        for (p = Kp[j]; p < Kp[j + 1]; p++) {
            i = Ki[p];
            if (i < 0 || i >= n) return -1;
            if (i != j) nz += 2;
        }
    }

    Len = (Int *)xmalloc((size_t)n, sizeof(Int));
    Pe = (Int *)xmalloc((size_t)n + 1, sizeof(Int));
    Nv = (Int *)xmalloc((size_t)n, sizeof(Int));
    Next = (Int *)xmalloc((size_t)n, sizeof(Int));
    Last = (Int *)xmalloc((size_t)n, sizeof(Int));
    Head = (Int *)xmalloc((size_t)n, sizeof(Int));
    Elen = (Int *)xmalloc((size_t)n, sizeof(Int));
    Degree = (Int *)xmalloc((size_t)n, sizeof(Int));
    W = (Int *)xmalloc((size_t)n, sizeof(Int));
    deg0 = (Int *)xmalloc((size_t)n, sizeof(Int));
    ndn = (Int *)xmalloc((size_t)n, sizeof(Int));
    iwlen = nz + nz / 5 + 2 * n + 16;
    Iw = (Int *)xmalloc((size_t)iwlen, sizeof(Int));
    if (!Len || !Pe || !Nv || !Next || !Last || !Head || !Elen || !Degree || !W || !deg0 || !ndn || !Iw)
        goto done;

    /* attempt 1: SuiteSparse semantics with a work budget. Matrices with
       "moderately dense" rows (degree below the dense threshold but far above
       the typical degree, e.g. Lasso-type KKT systems) make the quotient-graph
       scans quadratic; for those, retry once with the dense threshold lowered
       below the degrees of the offending rows (they are then ordered last). */
    pfree = amd_pattern(n, Kp, Ki, Len, Pe, Iw, W, Head);
    memcpy(deg0, Len, (size_t)n * sizeof(Int));
    budget = 30.0 * (double)(pfree + n) + 3.0e5; /* soft budget; hard budget = 100/30 of it */
    if (amd_2(n, Pe, Iw, Len, iwlen, pfree, Nv, Next, Last, Head, Elen, Degree, W, EMPTY, budget, deg0,
              &thr, ndn) != 0) {
        Int ndense = 0, dense_in = EMPTY;
        if (thr >= 16) {
            for (i = 0; i < n; i++) ndense += (deg0[i] > thr);
            if (ndense <= n / 2) dense_in = thr;
        }
        pfree = amd_pattern(n, Kp, Ki, Len, Pe, Iw, W, Head);
        amd_2(n, Pe, Iw, Len, iwlen, pfree, Nv, Next, Last, Head, Elen, Degree, W, dense_in, 0.0, deg0,
              NULL, ndn);
    }

    for (i = 0; i < n; i++) perm[i] = Last[i];
    status = 0;

done:
    free(Len);
    free(Pe);
    free(Iw);
    free(Nv);
    free(Next);
    free(Last);
    free(Head);
    free(Elen);
    free(Degree);
    free(W);
    free(deg0);
    free(ndn);
    return status;
}

/* ========================================================================= */
/* LDL^T                                                                     */
/* ========================================================================= */

/*
 * Two numeric back-ends share the symbolic analysis:
 *  - mode 0: QDLDL-style up-looking factorization (sparse L in CSC),
 *            used for very sparse factors where dense kernels do not pay off;
 *  - mode 1: left-looking supernodal factorization with relaxed
 *            amalgamation, dense column-major panels and register-blocked
 *            update kernels (no external BLAS).
 * The mode is chosen automatically in ldl_create.
 */

#ifdef LDL_TEST_HOOKS
int ldl_force_mode = 0; /* 0: automatic, 1: up-looking, 2: supernodal */
#endif

#define LDL_NB 32     /* block size of the in-panel blocked factorization */
#define LDL_CB 128    /* column chunk of descendant updates */
#define LDL_KB 128    /* k-blocking of the update kernel */
#define LDL_MB 256    /* row blocking of the update kernel */
#define LDL_NBB 128   /* block size of the panel factorization when BLAS is available */
#define LDL_GEMM_MIN 20000.0 /* minimum m*n*k for calling BLAS */

#ifndef LDL_HAVE_BLAS_HOOKS
typedef void (*ldl_dgemm_fn)(const char *ta, const char *tb, const int *m, const int *n, const int *k,
                             const double *alpha, const double *A, const int *lda, const double *B,
                             const int *ldb, const double *beta, double *C, const int *ldc, size_t,
                             size_t);
typedef void (*ldl_dtrsm_fn)(const char *side, const char *uplo, const char *transa, const char *diag,
                             const int *m, const int *n, const double *alpha, const double *A,
                             const int *lda, double *B, const int *ldb, size_t, size_t, size_t,
                             size_t);
void ldl_set_blas(ldl_dgemm_fn gemm, ldl_dtrsm_fn trsm);
#endif

static ldl_dgemm_fn g_dgemm = NULL;
static ldl_dtrsm_fn g_dtrsm = NULL;

void ldl_set_blas(ldl_dgemm_fn gemm, ldl_dtrsm_fn trsm)
{
    g_dgemm = gemm;
    g_dtrsm = trsm;
}

/* C(m x n, ldc) = beta*C - A(m x k, lda) * B(n x k, ldb)^T via BLAS */
static void blas_gemm_nt(Int m, Int n, Int k, const double *A, Int lda, const double *B, Int ldb, double beta,
                         double *C, Int ldc)
{
    int im = (int)m, in = (int)n, ik = (int)k, ilda = (int)lda, ildb = (int)ldb, ildc = (int)ldc;
    double mone = -1.0;
    g_dgemm("N", "T", &im, &in, &ik, &mone, A, &ilda, B, &ildb, &beta, C, &ildc, 1, 1);
}

struct ldl_s {
    Int n;
    Int p0, nnzK; /* input entries occupy positions p0 .. nnzK-1 */
    Int nnzC;
    Int nnzL;     /* structural strictly-lower nonzeros of L */
    int mode;     /* 0 up-looking, 1 supernodal */
    Int *perm, *iperm;
    double *D, *Dinv;
    double *xwork, *xwork2; /* n each, for solves */

    /* ---- up-looking ---- */
    Int *Cp, *Ci; /* permuted upper triangular pattern */
    Int *map;     /* map[p] = position in Ci of input entry p */
    double *Cx;
    Int *etree, *Lnz, *Lp, *Li;
    double *Lx;
    Int *iwork; /* 3n */
    double *y;  /* n, kept zero between rows */
    unsigned char *mark;

    /* ---- supernodal ---- */
    Int nsuper;
    Int *Super;    /* nsuper+1: first column of each supernode */
    Int *colsup;   /* n: supernode containing each column */
    Int *Lpi;      /* nsuper+1: start of row indices of each supernode in Ls */
    Int *Ls;       /* row indices (diagonal block rows first, then sorted below rows) */
    Int *Lpx;      /* nsuper+1: start of each dense panel in Sx */
    double *Sx;    /* panels, column-major nsrow x nscol */
    Int *aptr;     /* nsuper+1: assembly list pointers */
    Int *asrc, *adst; /* input position -> panel position */
    Int *sHead, *sNext, *sLpos; /* nsuper each: descendant link lists */
    Int *Map, *Rel; /* n, maxnsrow */
    double *Wbuf, *Cbuf;
    Int maxnsrow, maxnscol;
};

void ldl_free(ldl_t *f)
{
    if (!f) return;
    free(f->perm);
    free(f->iperm);
    free(f->D);
    free(f->Dinv);
    free(f->xwork);
    free(f->xwork2);
    free(f->Cp);
    free(f->Ci);
    free(f->map);
    free(f->Cx);
    free(f->etree);
    free(f->Lnz);
    free(f->Lp);
    free(f->Li);
    free(f->Lx);
    free(f->iwork);
    free(f->y);
    free(f->mark);
    free(f->Super);
    free(f->colsup);
    free(f->Lpi);
    free(f->Ls);
    free(f->Lpx);
    free(f->Sx);
    free(f->aptr);
    free(f->asrc);
    free(f->adst);
    free(f->sHead);
    free(f->sNext);
    free(f->sLpos);
    free(f->Map);
    free(f->Rel);
    free(f->Wbuf);
    free(f->Cbuf);
    free(f);
}

/* permuted upper triangular pattern C = P K P' (Cp, Ci) and entry map */
static void build_C(ldl_t *f, const Int *Kp, const Int *Ki, Int *work)
{
    Int n = f->n, j, k, p, q, i2, j2;
    for (k = 0; k <= n; k++) f->Cp[k] = 0;
    for (j = 0; j < n; j++) {
        j2 = f->iperm[j];
        for (p = Kp[j]; p < Kp[j + 1]; p++) {
            i2 = f->iperm[Ki[p]];
            f->Cp[LMAX(i2, j2) + 1]++;
        }
    }
    for (k = 0; k < n; k++) f->Cp[k + 1] += f->Cp[k];
    for (k = 0; k < n; k++) work[k] = f->Cp[k];
    for (p = 0; p < f->p0; p++) f->map[p] = EMPTY;
    for (j = 0; j < n; j++) {
        j2 = f->iperm[j];
        for (p = Kp[j]; p < Kp[j + 1]; p++) {
            i2 = f->iperm[Ki[p]];
            q = work[LMAX(i2, j2)]++;
            f->Ci[q] = LMIN(i2, j2);
            f->map[p] = q;
        }
    }
}

/* elimination tree only (Liu's algorithm with path compression) */
static void etree_only(Int n, const Int *Cp, const Int *Ci, Int *parent, Int *ancestor)
{
    Int k, p, i, inext;
    for (k = 0; k < n; k++) {
        parent[k] = EMPTY;
        ancestor[k] = EMPTY;
        for (p = Cp[k]; p < Cp[k + 1]; p++) {
            i = Ci[p];
            while (i != EMPTY && i < k) {
                inext = ancestor[i];
                ancestor[i] = k;
                if (inext == EMPTY) parent[i] = k;
                i = inext;
            }
        }
    }
}

/* postorder of a forest: post[k] = k-th node */
static void tree_postorder(Int n, const Int *parent, Int *post, Int *head, Int *next, Int *stack)
{
    Int j, k = 0;
    for (j = 0; j < n; j++) head[j] = EMPTY;
    for (j = n - 1; j >= 0; j--) {
        if (parent[j] != EMPTY) {
            next[j] = head[parent[j]];
            head[parent[j]] = j;
        }
    }
    for (j = 0; j < n; j++) {
        if (parent[j] == EMPTY) {
            Int top = 0;
            stack[0] = j;
            while (top >= 0) {
                Int pp = stack[top], i = head[pp];
                if (i == EMPTY) {
                    top--;
                    post[k++] = pp;
                } else {
                    head[pp] = next[i];
                    stack[++top] = i;
                }
            }
        }
    }
}

/* ------------------------------------------------------------------------- */
/* supernodal symbolic analysis                                              */
/* ------------------------------------------------------------------------- */

/* returns 1 if built, 0 if not worthwhile, -1 on out of memory */
static int build_super(ldl_t *f)
{
    Int n = f->n, j, s, x, i, p, q, nfs, nsuper, c;
    const Int *parent = f->etree, *Lnz = f->Lnz;
    Int *ffirst = NULL, *fcolsup = NULL, *fpar = NULL, *grp = NULL, *gcols = NULL, *mark = NULL,
        *fillpos = NULL, *sparent = NULL;
    double *gact = NULL, *fact = NULL;
    double flops = 0.0, npanel = 0.0;
    int status = -1;

    ffirst = (Int *)xmalloc((size_t)n + 1, sizeof(Int));
    fcolsup = (Int *)xmalloc((size_t)n, sizeof(Int));
    fpar = (Int *)xmalloc((size_t)n, sizeof(Int));
    grp = (Int *)xmalloc((size_t)n, sizeof(Int));
    gcols = (Int *)xmalloc((size_t)n, sizeof(Int));
    gact = (double *)xmalloc((size_t)n, sizeof(double));
    fact = (double *)xmalloc((size_t)n, sizeof(double));
    if (!ffirst || !fcolsup || !fpar || !grp || !gcols || !gact || !fact) goto done;

    for (j = 0; j < n; j++) flops += (double)Lnz[j] * (double)Lnz[j];

    /* fundamental supernodes (chains with nested structure) */
    nfs = 0;
    for (j = 0; j < n; j++) {
        if (j == 0 || !(parent[j - 1] == j && Lnz[j - 1] == Lnz[j] + 1)) ffirst[nfs++] = j;
        fcolsup[j] = nfs - 1;
    }
    ffirst[nfs] = n;
    for (s = 0; s < nfs; s++) {
        Int last = ffirst[s + 1] - 1;
        fpar[s] = (parent[last] == EMPTY) ? EMPTY : fcolsup[parent[last]];
        fact[s] = 0.0;
        for (c = ffirst[s]; c <= last; c++) fact[s] += (double)(Lnz[c] + 1);
    }

    /* relaxed amalgamation (top-down: merge child s into the group of s+1) */
    grp[nfs - 1] = nfs - 1;
    gcols[nfs - 1] = ffirst[nfs] - ffirst[nfs - 1];
    gact[nfs - 1] = fact[nfs - 1];
    for (s = nfs - 2; s >= 0; s--) {
        Int g = grp[s + 1], sp = fpar[s], merge = 0, ncs = ffirst[s + 1] - ffirst[s];
        Int nc = 0;
        double act = 0.0;
        if (sp != EMPTY && sp <= g) {
            double nr, stored, z;
            nc = ncs + gcols[g];
            nr = (double)nc + (double)Lnz[ffirst[g + 1] - 1];
            stored = (double)nc * nr - (double)nc * (double)(nc - 1) / 2.0;
            act = fact[s] + gact[g];
            z = (stored - act) / stored;
            merge = (stored <= act) || (nc <= 4 && z < 0.5) || (nc <= 16 && z < 0.3) ||
                    (nc <= 48 && z < 0.1) || (z < 0.05);
        }
        if (merge) {
            grp[s] = g;
            gcols[g] = nc;
            gact[g] = act;
        } else {
            grp[s] = s;
            gcols[s] = ncs;
            gact[s] = fact[s];
        }
    }
    nsuper = 0;
    for (s = 0; s < nfs; s++) {
        if (s == 0 || grp[s] != grp[s - 1]) nsuper++;
    }

    /* decide whether the supernodal factorization is worthwhile */
    {
        /* dense kernels pay off unless the factor is extremely sparse
           (average number of updates per entry of L below ~8) */
        int use = (flops >= 1.0e4) && (flops > 8.0 * (double)(f->nnzL + 1));
#ifdef LDL_TEST_HOOKS
        if (ldl_force_mode == 1) use = 0;
        if (ldl_force_mode == 2) use = 1;
#endif
        if (!use) {
            status = 0;
            goto done;
        }
    }

    f->nsuper = nsuper;
    f->Super = (Int *)xmalloc((size_t)nsuper + 1, sizeof(Int));
    f->colsup = (Int *)xmalloc((size_t)n, sizeof(Int));
    f->Lpi = (Int *)xmalloc((size_t)nsuper + 1, sizeof(Int));
    f->Lpx = (Int *)xmalloc((size_t)nsuper + 1, sizeof(Int));
    sparent = (Int *)xmalloc((size_t)nsuper, sizeof(Int));
    mark = (Int *)xmalloc((size_t)nsuper, sizeof(Int));
    fillpos = (Int *)xmalloc((size_t)nsuper, sizeof(Int));
    if (!f->Super || !f->colsup || !f->Lpi || !f->Lpx || !sparent || !mark || !fillpos) goto done;

    nsuper = 0;
    for (s = 0; s < nfs; s++) {
        if (s == 0 || grp[s] != grp[s - 1]) f->Super[nsuper++] = ffirst[s];
    }
    f->Super[nsuper] = n;
    f->maxnsrow = 0;
    f->maxnscol = 0;
    f->Lpi[0] = 0;
    f->Lpx[0] = 0;
    for (s = 0; s < nsuper; s++) {
        Int k1 = f->Super[s], k2 = f->Super[s + 1], nscol = k2 - k1;
        Int nsrow = nscol + Lnz[k2 - 1];
        for (c = k1; c < k2; c++) f->colsup[c] = s;
        f->Lpi[s + 1] = f->Lpi[s] + nsrow;
        f->Lpx[s + 1] = f->Lpx[s] + nsrow * nscol;
        npanel += (double)nsrow * (double)nscol;
        f->maxnsrow = LMAX(f->maxnsrow, nsrow);
        f->maxnscol = LMAX(f->maxnscol, nscol);
    }
    for (s = 0; s < nsuper; s++) {
        Int last = f->Super[s + 1] - 1;
        sparent[s] = (parent[last] == EMPTY) ? EMPTY : f->colsup[parent[last]];
    }
#ifdef LDL_TEST_HOOKS
    if (ldl_force_mode != 2)
#endif
    if (npanel > 3.0 * (double)(f->nnzL + n) + 1e6) {
        /* too much padding: not worthwhile */
        status = 0;
        goto done;
    }

    f->Ls = (Int *)xmalloc((size_t)f->Lpi[nsuper], sizeof(Int));
    if (!f->Ls) goto done;
    for (s = 0; s < nsuper; s++) {
        Int k1 = f->Super[s], k2 = f->Super[s + 1];
        q = f->Lpi[s];
        for (c = k1; c < k2; c++) f->Ls[q++] = c;
        fillpos[s] = q;
        mark[s] = EMPTY;
    }
    /* row structure: supernodal row subtrees (rows in increasing order) */
    for (i = 0; i < n; i++) {
        for (p = f->Cp[i]; p < f->Cp[i + 1]; p++) {
            Int r = f->Ci[p];
            if (r >= i) continue;
            x = f->colsup[r];
            while (x != EMPTY && f->Super[x + 1] <= i && mark[x] != i) {
                mark[x] = i;
                if (fillpos[x] >= f->Lpi[x + 1]) {
                    status = 0; /* inconsistent structure: fall back */
                    goto done;
                }
                f->Ls[fillpos[x]++] = i;
                x = sparent[x];
            }
        }
    }
    for (s = 0; s < nsuper; s++) {
        if (fillpos[s] != f->Lpi[s + 1]) {
            status = 0;
            goto done;
        }
    }

    /* assembly lists: input entry -> panel position */
    f->aptr = (Int *)xmalloc((size_t)nsuper + 1, sizeof(Int));
    f->asrc = (Int *)xmalloc((size_t)f->nnzC, sizeof(Int));
    f->adst = (Int *)xmalloc((size_t)f->nnzC, sizeof(Int));
    f->Map = (Int *)xmalloc((size_t)n, sizeof(Int));
    if (!f->aptr || !f->asrc || !f->adst || !f->Map) goto done;
    for (s = 0; s <= nsuper; s++) f->aptr[s] = 0;
    for (j = 0; j < n; j++) {
        for (p = f->Cp[j]; p < f->Cp[j + 1]; p++) {
            /* entry (Ci[p], j) of the upper part: column min = Ci[p] */
            f->aptr[f->colsup[f->Ci[p]] + 1]++;
        }
    }
    for (s = 0; s < nsuper; s++) f->aptr[s + 1] += f->aptr[s];
    for (s = 0; s < nsuper; s++) fillpos[s] = f->aptr[s];
    {
        Int *tmp = (Int *)xmalloc((size_t)f->nnzC, sizeof(Int));  /* C position -> input position */
        Int *acol = (Int *)xmalloc((size_t)f->nnzC, sizeof(Int)); /* column (smaller index) */
        if (!tmp || !acol) {
            free(tmp);
            free(acol);
            goto done;
        }
        for (p = f->p0; p < f->nnzK; p++) tmp[f->map[p]] = p;
        for (j = 0; j < n; j++) {
            for (p = f->Cp[j]; p < f->Cp[j + 1]; p++) {
                Int r = f->Ci[p];
                s = f->colsup[r];
                q = fillpos[s]++;
                f->asrc[q] = tmp[p];
                f->adst[q] = j; /* row (larger index), replaced below */
                acol[q] = r;
            }
        }
        for (s = 0; s < nsuper; s++) {
            Int psi = f->Lpi[s], nsrow = f->Lpi[s + 1] - psi, k1 = f->Super[s];
            for (q = 0; q < nsrow; q++) f->Map[f->Ls[psi + q]] = q;
            for (q = f->aptr[s]; q < f->aptr[s + 1]; q++) {
                Int row = f->adst[q], col = acol[q], lr = f->Map[row];
                if (lr < 0 || lr >= nsrow || f->Ls[psi + lr] != row || col < k1 || col >= f->Super[s + 1]) {
                    free(tmp);
                    free(acol);
                    status = 0;
                    goto done;
                }
                f->adst[q] = f->Lpx[s] + lr + (col - k1) * nsrow;
            }
        }
        free(tmp);
        free(acol);
    }

    f->Sx = (double *)xmalloc((size_t)f->Lpx[nsuper], sizeof(double));
    f->sHead = (Int *)xmalloc((size_t)nsuper, sizeof(Int));
    f->sNext = (Int *)xmalloc((size_t)nsuper, sizeof(Int));
    f->sLpos = (Int *)xmalloc((size_t)nsuper, sizeof(Int));
    f->Rel = (Int *)xmalloc((size_t)f->maxnsrow, sizeof(Int));
    f->Wbuf = (double *)xmalloc(2 * (size_t)LMAX(LDL_CB, LDL_NB) * (size_t)f->maxnscol, sizeof(double));
    f->Cbuf = (double *)xmalloc((size_t)f->maxnsrow * (size_t)LMIN(LDL_CB, f->maxnsrow), sizeof(double));
    if (!f->Sx || !f->sHead || !f->sNext || !f->sLpos || !f->Rel || !f->Wbuf || !f->Cbuf) goto done;
    status = 1;

done:
    free(ffirst);
    free(fcolsup);
    free(fpar);
    free(grp);
    free(gcols);
    free(gact);
    free(fact);
    free(mark);
    free(fillpos);
    free(sparent);
    if (status != 1) {
        /* release partial supernodal data */
        free(f->Super);
        free(f->colsup);
        free(f->Lpi);
        free(f->Lpx);
        free(f->Ls);
        free(f->aptr);
        free(f->asrc);
        free(f->adst);
        free(f->Map);
        free(f->Sx);
        free(f->sHead);
        free(f->sNext);
        free(f->sLpos);
        free(f->Rel);
        free(f->Wbuf);
        free(f->Cbuf);
        f->Super = f->colsup = f->Lpi = f->Lpx = f->Ls = f->aptr = f->asrc = f->adst = f->Map = NULL;
        f->sHead = f->sNext = f->sLpos = f->Rel = NULL;
        f->Sx = f->Wbuf = f->Cbuf = NULL;
        f->nsuper = 0;
    }
    return status;
}

ldl_t *ldl_create(int64_t n, const int64_t *Kp, const int64_t *Ki, const int64_t *perm)
{
    ldl_t *f;
    Int i, j, k, p, nnzK, p0, *work;
    size_t nn;
    int sup;

    if (n < 0) return NULL;
    if (n > 0 && (Kp == NULL)) return NULL;
    p0 = (n > 0) ? Kp[0] : 0;
    nnzK = (n > 0) ? Kp[n] : 0;
    if (p0 < 0) return NULL;
    for (j = 0; j < n; j++) {
        if (Kp[j + 1] < Kp[j]) return NULL;
    }
    if (nnzK > p0 && Ki == NULL) return NULL;
    for (p = p0; p < nnzK; p++) {
        if (Ki[p] < 0 || Ki[p] >= n) return NULL;
    }

    f = (ldl_t *)calloc(1, sizeof(ldl_t));
    if (!f) return NULL;
    f->n = n;
    f->p0 = p0;
    f->nnzK = nnzK;
    f->nnzC = nnzK - p0;
    nn = (size_t)n;

    f->perm = (Int *)xmalloc(nn, sizeof(Int));
    f->iperm = (Int *)xmalloc(nn, sizeof(Int));
    f->Cp = (Int *)xmalloc(nn + 1, sizeof(Int));
    f->Ci = (Int *)xmalloc((size_t)f->nnzC, sizeof(Int));
    f->map = (Int *)xmalloc((size_t)nnzK, sizeof(Int));
    f->etree = (Int *)xmalloc(nn, sizeof(Int));
    f->Lnz = (Int *)xmalloc(nn, sizeof(Int));
    f->D = (double *)xmalloc(nn, sizeof(double));
    f->Dinv = (double *)xmalloc(nn, sizeof(double));
    f->iwork = (Int *)xmalloc(3 * nn, sizeof(Int));
    f->xwork = (double *)xmalloc(nn, sizeof(double));
    f->xwork2 = (double *)xmalloc(nn, sizeof(double));
    if (!f->perm || !f->iperm || !f->Cp || !f->Ci || !f->map || !f->etree || !f->Lnz || !f->D ||
        !f->Dinv || !f->iwork || !f->xwork || !f->xwork2) {
        ldl_free(f);
        return NULL;
    }
    work = f->iwork;

    /* ordering */
    if (perm) {
        for (i = 0; i < n; i++) f->iperm[i] = EMPTY;
        for (k = 0; k < n; k++) {
            i = perm[k];
            if (i < 0 || i >= n || f->iperm[i] != EMPTY) {
                ldl_free(f);
                return NULL;
            }
            f->iperm[i] = k;
            f->perm[k] = i;
        }
        build_C(f, Kp, Ki, work);
    } else {
        Int *tmp;
        if (amd_order(n, Kp, Ki, f->perm) != 0) {
            ldl_free(f);
            return NULL;
        }
        for (i = 0; i < n; i++) f->iperm[i] = EMPTY;
        for (k = 0; k < n; k++) {
            i = f->perm[k];
            if (i < 0 || i >= n || f->iperm[i] != EMPTY) {
                ldl_free(f);
                return NULL;
            }
            f->iperm[i] = k;
        }
        build_C(f, Kp, Ki, work);
        /* postorder the elimination tree (equivalent ordering, same fill) */
        tmp = (Int *)xmalloc(2 * nn, sizeof(Int));
        if (!tmp) {
            ldl_free(f);
            return NULL;
        }
        etree_only(n, f->Cp, f->Ci, f->etree, work);
        tree_postorder(n, f->etree, tmp, work, work + n, work + 2 * n);
        for (k = 0; k < n; k++) tmp[n + k] = f->perm[tmp[k]];
        for (k = 0; k < n; k++) {
            f->perm[k] = tmp[n + k];
            f->iperm[f->perm[k]] = k;
        }
        free(tmp);
        build_C(f, Kp, Ki, work);
    }

    /* elimination tree and column counts of L */
    for (i = 0; i < n; i++) {
        work[i] = EMPTY;
        f->Lnz[i] = 0;
        f->etree[i] = EMPTY;
    }
    for (j = 0; j < n; j++) {
        work[j] = j;
        for (p = f->Cp[j]; p < f->Cp[j + 1]; p++) {
            i = f->Ci[p];
            while (work[i] != j) {
                if (f->etree[i] == EMPTY) f->etree[i] = j;
                f->Lnz[i]++;
                work[i] = j;
                i = f->etree[i];
            }
        }
    }
    f->nnzL = 0;
    for (i = 0; i < n; i++) f->nnzL += f->Lnz[i];
    for (i = 0; i < n; i++) {
        f->D[i] = 1.0;
        f->Dinv[i] = 1.0;
    }

    sup = (n > 0) ? build_super(f) : 0;
    if (sup < 0) {
        ldl_free(f);
        return NULL;
    }
    if (sup == 1) {
        f->mode = 1;
        /* up-looking data no longer needed */
        free(f->Cp);
        free(f->Ci);
        free(f->map);
        free(f->etree);
        free(f->Lnz);
        free(f->iwork);
        f->Cp = f->Ci = f->map = f->etree = f->Lnz = f->iwork = NULL;
        return f;
    }

    /* up-looking mode */
    f->mode = 0;
    f->Cx = (double *)xmalloc((size_t)f->nnzC, sizeof(double));
    f->Lp = (Int *)xmalloc(nn + 1, sizeof(Int));
    f->y = (double *)calloc(nn ? nn : 1, sizeof(double));
    f->mark = (unsigned char *)calloc(nn ? nn : 1, 1);
    if (!f->Cx || !f->Lp || !f->y || !f->mark) {
        ldl_free(f);
        return NULL;
    }
    f->Lp[0] = 0;
    for (i = 0; i < n; i++) f->Lp[i + 1] = f->Lp[i] + f->Lnz[i];
    f->Li = (Int *)xmalloc((size_t)f->nnzL, sizeof(Int));
    f->Lx = (double *)xmalloc((size_t)f->nnzL, sizeof(double));
    if (!f->Li || !f->Lx) {
        ldl_free(f);
        return NULL;
    }
    return f;
}

/* ------------------------------------------------------------------------- */
/* dynamic regularization of a pivot                                         */
/* ------------------------------------------------------------------------- */
static inline double ldl_regularize(double d, signed char s, double eps, double delta, Int *nreg)
{
    if (s > 0) {
        if (d <= eps) {
            d = delta;
            (*nreg)++;
        }
    } else if (s < 0) {
        if (-d <= eps) {
            d = -delta;
            (*nreg)++;
        }
    }
    return d;
}

/* ------------------------------------------------------------------------- */
/* up-looking numeric factorization                                          */
/* ------------------------------------------------------------------------- */
static int64_t factor_up(ldl_t *f, const double *Kx, const signed char *signs, double eps, double delta)
{
    Int n, k, p, i, nreg = 0;
    const Int *Cp, *Ci, *etree, *Lp, *perm, *map;
    Int *Li, *yIdx, *elim, *Lnext;
    double *Lx, *D, *Dinv, *Cx, *y;
    unsigned char *mark;

    n = f->n;
    Cp = f->Cp;
    Ci = f->Ci;
    etree = f->etree;
    Lp = f->Lp;
    perm = f->perm;
    map = f->map;
    Li = f->Li;
    Lx = f->Lx;
    D = f->D;
    Dinv = f->Dinv;
    Cx = f->Cx;
    y = f->y;
    mark = f->mark;
    yIdx = f->iwork;
    elim = f->iwork + n;
    Lnext = f->iwork + 2 * n;

    for (p = f->p0; p < f->nnzK; p++) Cx[map[p]] = Kx[p];

    memset(y, 0, (size_t)n * sizeof(double));
    memset(mark, 0, (size_t)n);
    for (i = 0; i < n; i++) Lnext[i] = Lp[i];

    for (k = 0; k < n; k++) {
        Int nnzY = 0, pend = Cp[k + 1];
        double dk = 0.0;

        for (p = Cp[k]; p < pend; p++) {
            Int b = Ci[p];
            if (b == k) {
                dk += Cx[p];
                continue;
            }
            y[b] += Cx[p];
            if (!mark[b]) {
                Int nE = 0, nx;
                mark[b] = 1;
                elim[nE++] = b;
                nx = etree[b];
                while (nx != EMPTY && nx < k) {
                    if (mark[nx]) break;
                    mark[nx] = 1;
                    elim[nE++] = nx;
                    nx = etree[nx];
                }
                while (nE) yIdx[nnzY++] = elim[--nE];
            }
        }

        for (i = nnzY - 1; i >= 0; i--) {
            Int c = yIdx[i];
            Int q, qend = Lnext[c];
            double yc = y[c], lkc;
            for (q = Lp[c]; q < qend; q++) y[Li[q]] -= Lx[q] * yc;
            lkc = yc * Dinv[c];
            Li[qend] = k;
            Lx[qend] = lkc;
            dk -= yc * lkc;
            Lnext[c] = qend + 1;
            y[c] = 0.0;
            mark[c] = 0;
        }

        if (signs) dk = ldl_regularize(dk, signs[perm[k]], eps, delta, &nreg);
        if (dk == 0.0 || !isfinite(dk)) {
            D[k] = dk;
            return -1;
        }
        D[k] = dk;
        Dinv[k] = 1.0 / dk;
    }
    return nreg;
}

/* ------------------------------------------------------------------------- */
/* dense kernels                                                             */
/* ------------------------------------------------------------------------- */
typedef double ldl_v2d __attribute__((vector_size(16)));

static inline ldl_v2d v2_ld(const double *p)
{
    ldl_v2d v;
    memcpy(&v, p, sizeof v);
    return v;
}
static inline void v2_st(double *p, ldl_v2d v) { memcpy(p, &v, sizeof v); }

/* C[0:4, 0:4] -= A[0:4, 0:kk] * W[0:kk, 0:4]; W holds duplicated pairs, stride ws per k */
static inline void mk4x4(Int kk, const double *A, Int lda, const double *W, Int ws, double *C, Int ldc)
{
    ldl_v2d c00 = {0.0, 0.0};
    ldl_v2d c10 = c00, c01 = c00, c11 = c00, c02 = c00, c12 = c00, c03 = c00, c13 = c00;
    Int k;
    for (k = 0; k < kk; k++) {
        ldl_v2d a0 = v2_ld(A), a1 = v2_ld(A + 2);
        ldl_v2d w0 = v2_ld(W), w1 = v2_ld(W + 2), w2 = v2_ld(W + 4), w3 = v2_ld(W + 6);
        c00 += a0 * w0;
        c10 += a1 * w0;
        c01 += a0 * w1;
        c11 += a1 * w1;
        c02 += a0 * w2;
        c12 += a1 * w2;
        c03 += a0 * w3;
        c13 += a1 * w3;
        A += lda;
        W += ws;
    }
    v2_st(C, v2_ld(C) - c00);
    v2_st(C + 2, v2_ld(C + 2) - c10);
    C += ldc;
    v2_st(C, v2_ld(C) - c01);
    v2_st(C + 2, v2_ld(C + 2) - c11);
    C += ldc;
    v2_st(C, v2_ld(C) - c02);
    v2_st(C + 2, v2_ld(C + 2) - c12);
    C += ldc;
    v2_st(C, v2_ld(C) - c03);
    v2_st(C + 2, v2_ld(C + 2) - c13);
}

/*
 * C[i, j] -= sum_k A[i, k] * W(k, j)   for 0 <= j < n and (4*floor(j/4)) <= i < m.
 * (All entries with i >= j are updated; a few entries above the diagonal of
 * each 4x4 diagonal block may be touched as well.)
 * A column-major (lda); W(k, j) = Wd[2*(k*n + j)] (= Wd[2*(k*n+j)+1]).
 */
static void update_lower(Int m, Int n, Int kk, const double *A, Int lda, const double *Wd, double *C,
                         Int ldc)
{
    Int k0, i0, j, i, k, c;
    for (k0 = 0; k0 < kk; k0 += LDL_KB) {
        Int kb = LMIN(LDL_KB, kk - k0);
        const double *Ak = A + k0 * lda;
        const double *Wk = Wd + 2 * k0 * n;
        for (i0 = 0; i0 < m; i0 += LDL_MB) {
            Int i1 = LMIN(m, i0 + LDL_MB);
            for (j = 0; j < n && j < i1; j += 4) {
                Int ist = LMAX(i0, j);
                if (j + 4 <= n) {
                    for (i = ist; i + 4 <= i1; i += 4)
                        mk4x4(kb, Ak + i, lda, Wk + 2 * j, 2 * n, C + i + j * ldc, ldc);
                    for (; i < i1; i++) {
                        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                        for (k = 0; k < kb; k++) {
                            double av = Ak[i + k * lda];
                            const double *w = Wk + 2 * (k * n + j);
                            s0 += av * w[0];
                            s1 += av * w[2];
                            s2 += av * w[4];
                            s3 += av * w[6];
                        }
                        C[i + j * ldc] -= s0;
                        C[i + (j + 1) * ldc] -= s1;
                        C[i + (j + 2) * ldc] -= s2;
                        C[i + (j + 3) * ldc] -= s3;
                    }
                } else {
                    for (c = j; c < n; c++) {
                        double *cc = C + c * ldc;
                        for (k = 0; k < kb; k++) {
                            double w = Wk[2 * (k * n + c)];
                            const double *a = Ak + k * lda;
                            for (i = ist; i < i1; i++) cc[i] -= a[i] * w;
                        }
                    }
                }
            }
        }
    }
}

/* dense LDL^T of a supernodal panel (nsrow x nscol, column-major), with
   dynamic regularization; returns 0 or -1 */
static int panel_factor(double *P, Int nsrow, Int nscol, Int k1, double *D, double *Dinv,
                        const signed char *signs, const Int *perm, double eps, double delta,
                        Int *nreg, double *Wd)
{
    Int j0, j1, j, c, i, k;
    for (j0 = 0; j0 < nscol; j0 += LDL_NB) {
        j1 = LMIN(j0 + LDL_NB, nscol);
        for (j = j0; j < j1; j++) {
            double *pj = P + j * nsrow;
            double d = pj[j], dinv;
            if (signs) d = ldl_regularize(d, signs[perm[k1 + j]], eps, delta, nreg);
            if (d == 0.0 || !isfinite(d)) {
                D[k1 + j] = d;
                return -1;
            }
            dinv = 1.0 / d;
            D[k1 + j] = d;
            Dinv[k1 + j] = dinv;
            for (c = j + 1; c < j1; c++) {
                double fc = pj[c] * dinv;
                double *pc = P + c * nsrow;
                for (i = c; i < nsrow; i++) pc[i] -= pj[i] * fc;
            }
            for (i = j + 1; i < nsrow; i++) pj[i] *= dinv;
        }
        if (j1 < nscol) {
            Int nt = nscol - j1, kb = j1 - j0;
            for (k = 0; k < kb; k++) {
                double dk = D[k1 + j0 + k];
                const double *lk = P + (j0 + k) * nsrow + j1;
                double *w = Wd + 2 * k * nt;
                for (c = 0; c < nt; c++) {
                    double v = dk * lk[c];
                    w[2 * c] = v;
                    w[2 * c + 1] = v;
                }
            }
            update_lower(nsrow - j1, nt, kb, P + j0 * nsrow + j1, nsrow, Wd, P + j1 * nsrow + j1, nsrow);
        }
    }
    return 0;
}

/* panel factorization using BLAS: unblocked LDL^T of each diagonal sub-block
   (with per-pivot dynamic regularization), dtrsm + scaling for the rows below,
   and dgemm for the trailing update */
static int panel_factor_blas(double *P, Int nsrow, Int nscol, Int k1, double *D, double *Dinv,
                             const signed char *signs, const Int *perm, double eps, double delta,
                             Int *nreg, double *Wb)
{
    Int j0, j1, j, c, i, k;
    for (j0 = 0; j0 < nscol; j0 += LDL_NBB) {
        Int nb, m, nt;
        j1 = LMIN(j0 + LDL_NBB, nscol);
        nb = j1 - j0;
        m = nsrow - j1;
        for (j = j0; j < j1; j++) {
            double *pj = P + j * nsrow;
            double d = pj[j], dinv;
            if (signs) d = ldl_regularize(d, signs[perm[k1 + j]], eps, delta, nreg);
            if (d == 0.0 || !isfinite(d)) {
                D[k1 + j] = d;
                return -1;
            }
            dinv = 1.0 / d;
            D[k1 + j] = d;
            Dinv[k1 + j] = dinv;
            for (c = j + 1; c < j1; c++) {
                double fc = pj[c] * dinv;
                double *pc = P + c * nsrow;
                for (i = c; i < j1; i++) pc[i] -= pj[i] * fc;
            }
            for (i = j + 1; i < j1; i++) pj[i] *= dinv;
        }
        if (m > 0) {
            int im = (int)m, inb = (int)nb, ld = (int)nsrow;
            double one = 1.0;
            g_dtrsm("R", "L", "T", "U", &im, &inb, &one, P + j0 * nsrow + j0, &ld, P + j0 * nsrow + j1, &ld,
                    1, 1, 1, 1);
            for (k = 0; k < nb; k++) {
                double di = Dinv[k1 + j0 + k];
                double *b = P + (j0 + k) * nsrow + j1;
                for (i = 0; i < m; i++) b[i] *= di;
            }
        }
        nt = nscol - j1;
        if (nt > 0) {
            Int c0;
            for (k = 0; k < nb; k++) {
                double dk = D[k1 + j0 + k];
                const double *l = P + (j0 + k) * nsrow + j1;
                double *w = Wb + k * nt;
                for (c = 0; c < nt; c++) w[c] = dk * l[c];
            }
            for (c0 = 0; c0 < nt; c0 += LDL_CB) {
                Int c1 = LMIN(c0 + LDL_CB, nt);
                blas_gemm_nt(m - c0, c1 - c0, nb, P + j0 * nsrow + j1 + c0, nsrow, Wb + c0, nt, 1.0,
                             P + (j1 + c0) * nsrow + j1 + c0, nsrow);
            }
        }
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* supernodal numeric factorization (left-looking)                           */
/* ------------------------------------------------------------------------- */
static int64_t factor_super(ldl_t *f, const double *Kx, const signed char *signs, double eps, double delta)
{
    Int nsuper = f->nsuper, s, d, q, nreg = 0;
    const Int *Super = f->Super, *Lpi = f->Lpi, *Ls = f->Ls, *Lpx = f->Lpx, *colsup = f->colsup;
    const Int *aptr = f->aptr, *asrc = f->asrc, *adst = f->adst, *perm = f->perm;
    Int *Head = f->sHead, *Next = f->sNext, *Lpos = f->sLpos, *Map = f->Map, *Rel = f->Rel;
    double *Sx = f->Sx, *D = f->D, *Dinv = f->Dinv, *Wd = f->Wbuf, *Cb = f->Cbuf;
    int use_blas = (g_dgemm != NULL) && (f->maxnsrow < 2147483647L);
    int use_blas_panel = use_blas && (g_dtrsm != NULL);

    for (s = 0; s < nsuper; s++) Head[s] = EMPTY;

    for (s = 0; s < nsuper; s++) {
        Int k1 = Super[s], k2 = Super[s + 1], nscol = k2 - k1;
        Int psi = Lpi[s], nsrow = Lpi[s + 1] - psi;
        double *Ps = Sx + Lpx[s];

        for (q = 0; q < nsrow; q++) Map[Ls[psi + q]] = q;

        /* assemble the original entries */
        memset(Ps, 0, (size_t)nsrow * (size_t)nscol * sizeof(double));
        for (q = aptr[s]; q < aptr[s + 1]; q++) Sx[adst[q]] += Kx[asrc[q]];

        /* updates from descendants */
        d = Head[s];
        while (d != EMPTY) {
            Int dnext = Next[d];
            Int pdi = Lpi[d], pdend = Lpi[d + 1], ldd = pdend - pdi;
            Int ndcol = Super[d + 1] - Super[d];
            Int p1 = Lpos[d], p2 = p1, ndrow1, ndrow2, ii, c, k;
            const double *A = Sx + Lpx[d] + (p1 - pdi);
            const double *Dd = D + Super[d];

            while (p2 < pdend && Ls[p2] < k2) p2++;
            ndrow1 = p2 - p1;
            ndrow2 = pdend - p1;
            for (ii = 0; ii < ndrow2; ii++) Rel[ii] = Map[Ls[p1 + ii]];

            if (use_blas && (double)ndrow2 * (double)ndrow1 * (double)ndcol > LDL_GEMM_MIN) {
                Int jc;
                for (jc = 0; jc < ndrow1; jc += LDL_CB) {
                    Int nc = LMIN(LDL_CB, ndrow1 - jc), mc = ndrow2 - jc;
                    const double *Ac = A + jc;
                    int contiguous = (Rel[ndrow2 - 1] - Rel[jc] == mc - 1);
                    for (k = 0; k < ndcol; k++) {
                        double dk = Dd[k];
                        const double *a = Ac + k * ldd;
                        double *w = Wd + k * nc;
                        for (c = 0; c < nc; c++) w[c] = dk * a[c];
                    }
                    if (contiguous && (Ls[p1 + jc + nc - 1] - Ls[p1 + jc] == nc - 1)) {
                        double *T = Ps + Rel[jc] + (Ls[p1 + jc] - k1) * nsrow;
                        blas_gemm_nt(mc, nc, ndcol, Ac, ldd, Wd, nc, 1.0, T, nsrow);
                    } else {
                        blas_gemm_nt(mc, nc, ndcol, Ac, ldd, Wd, nc, 0.0, Cb, mc);
                        for (c = 0; c < nc; c++) {
                            double *tcol = Ps + (Ls[p1 + jc + c] - k1) * nsrow;
                            const double *cc = Cb + c * mc;
                            const Int *rel = Rel + jc;
                            for (ii = c; ii < mc; ii++) tcol[rel[ii]] += cc[ii];
                        }
                    }
                }
            } else if (ndcol < 4) {
                /* direct scatter update */
                for (c = 0; c < ndrow1; c++) {
                    double *tcol = Ps + (Ls[p1 + c] - k1) * nsrow;
                    for (k = 0; k < ndcol; k++) {
                        const double *a = A + k * ldd;
                        double w = Dd[k] * a[c];
                        for (ii = c; ii < ndrow2; ii++) tcol[Rel[ii]] -= a[ii] * w;
                    }
                }
            } else {
                Int jc;
                for (jc = 0; jc < ndrow1; jc += LDL_CB) {
                    Int nc = LMIN(LDL_CB, ndrow1 - jc), mc = ndrow2 - jc;
                    const double *Ac = A + jc;
                    int contiguous = (Rel[ndrow2 - 1] - Rel[jc] == mc - 1);
                    for (k = 0; k < ndcol; k++) {
                        double dk = Dd[k];
                        const double *a = Ac + k * ldd;
                        double *w = Wd + 2 * k * nc;
                        for (c = 0; c < nc; c++) {
                            double v = dk * a[c];
                            w[2 * c] = v;
                            w[2 * c + 1] = v;
                        }
                    }
                    if (contiguous && (Ls[p1 + jc + nc - 1] - Ls[p1 + jc] == nc - 1)) {
                        /* target rows and columns are contiguous: update in place */
                        double *T = Ps + Rel[jc] + (Ls[p1 + jc] - k1) * nsrow;
                        update_lower(mc, nc, ndcol, Ac, ldd, Wd, T, nsrow);
                    } else {
                        memset(Cb, 0, (size_t)mc * (size_t)nc * sizeof(double));
                        update_lower(mc, nc, ndcol, Ac, ldd, Wd, Cb, mc);
                        for (c = 0; c < nc; c++) {
                            double *tcol = Ps + (Ls[p1 + jc + c] - k1) * nsrow;
                            const double *cc = Cb + c * mc;
                            const Int *rel = Rel + jc;
                            for (ii = c; ii < mc; ii++) tcol[rel[ii]] += cc[ii];
                        }
                    }
                }
            }

            Lpos[d] = p2;
            if (p2 < pdend) {
                Int t = colsup[Ls[p2]];
                Next[d] = Head[t];
                Head[t] = d;
            }
            d = dnext;
        }
        Head[s] = EMPTY;

        if (use_blas_panel && nscol >= 16 && (double)nscol * (double)nscol * (double)nsrow > 8.0 * LDL_GEMM_MIN) {
            if (panel_factor_blas(Ps, nsrow, nscol, k1, D, Dinv, signs, perm, eps, delta, &nreg, Wd) < 0)
                return -1;
        } else {
            if (panel_factor(Ps, nsrow, nscol, k1, D, Dinv, signs, perm, eps, delta, &nreg, Wd) < 0)
                return -1;
        }

        if (nsrow > nscol) {
            Int t = colsup[Ls[psi + nscol]];
            Lpos[s] = psi + nscol;
            Next[s] = Head[t];
            Head[t] = s;
        }
    }
    return nreg;
}

int64_t ldl_factor(ldl_t *f, const double *Kx, const signed char *signs, double eps, double delta)
{
    if (!f) return -1;
    if (f->n == 0) return 0;
    if (!Kx && f->nnzC > 0) return -1;
    if (f->mode == 1) return factor_super(f, Kx, signs, eps, delta);
    return factor_up(f, Kx, signs, eps, delta);
}

/* ------------------------------------------------------------------------- */
/* solves                                                                    */
/* ------------------------------------------------------------------------- */
static void solve_up(ldl_t *f, double *x)
{
    Int n = f->n, k, p;
    const Int *Lp = f->Lp, *Li = f->Li;
    const double *Lx = f->Lx, *Dinv = f->Dinv;
    for (k = 0; k < n; k++) {
        double xk = x[k];
        if (xk != 0.0) {
            Int pend = Lp[k + 1];
            for (p = Lp[k]; p < pend; p++) x[Li[p]] -= Lx[p] * xk;
        }
    }
    for (k = 0; k < n; k++) x[k] *= Dinv[k];
    for (k = n - 1; k >= 0; k--) {
        double s = x[k];
        Int pend = Lp[k + 1];
        for (p = Lp[k]; p < pend; p++) s -= Lx[p] * x[Li[p]];
        x[k] = s;
    }
}

static void solve_super(ldl_t *f, double *x)
{
    Int nsuper = f->nsuper, n = f->n, s, j, i, r, k;
    const Int *Super = f->Super, *Lpi = f->Lpi, *Ls = f->Ls, *Lpx = f->Lpx;
    const double *Sx = f->Sx, *Dinv = f->Dinv;
    double *t = f->xwork2;

    /* forward: L y = b */
    for (s = 0; s < nsuper; s++) {
        Int k1 = Super[s], nscol = Super[s + 1] - k1, psi = Lpi[s], nsrow = Lpi[s + 1] - psi;
        Int nb = nsrow - nscol;
        const double *P = Sx + Lpx[s];
        double *xs = x + k1;
        const Int *rows = Ls + psi + nscol;
        for (j = 0; j < nscol; j++) {
            const double *pj = P + j * nsrow;
            double xj = xs[j];
            for (i = j + 1; i < nscol; i++) xs[i] -= pj[i] * xj;
        }
        if (nb > 0) {
            if (nscol <= 4) {
                for (j = 0; j < nscol; j++) {
                    const double *pj = P + j * nsrow + nscol;
                    double xj = xs[j];
                    if (xj != 0.0)
                        for (r = 0; r < nb; r++) x[rows[r]] -= pj[r] * xj;
                }
            } else {
                for (r = 0; r < nb; r++) t[r] = 0.0;
                for (j = 0; j < nscol; j++) {
                    const double *pj = P + j * nsrow + nscol;
                    double xj = xs[j];
                    for (r = 0; r < nb; r++) t[r] += pj[r] * xj;
                }
                for (r = 0; r < nb; r++) x[rows[r]] -= t[r];
            }
        }
    }
    for (k = 0; k < n; k++) x[k] *= Dinv[k];
    /* backward: L' x = y */
    for (s = nsuper - 1; s >= 0; s--) {
        Int k1 = Super[s], nscol = Super[s + 1] - k1, psi = Lpi[s], nsrow = Lpi[s + 1] - psi;
        Int nb = nsrow - nscol;
        const double *P = Sx + Lpx[s];
        double *xs = x + k1;
        const Int *rows = Ls + psi + nscol;
        if (nb > 0) {
            if (nscol <= 4) {
                for (j = 0; j < nscol; j++) {
                    const double *pj = P + j * nsrow + nscol;
                    double sum = 0.0;
                    for (r = 0; r < nb; r++) sum += pj[r] * x[rows[r]];
                    xs[j] -= sum;
                }
            } else {
                for (r = 0; r < nb; r++) t[r] = x[rows[r]];
                for (j = 0; j < nscol; j++) {
                    const double *pj = P + j * nsrow + nscol;
                    double sum = 0.0;
                    for (r = 0; r < nb; r++) sum += pj[r] * t[r];
                    xs[j] -= sum;
                }
            }
        }
        for (j = nscol - 1; j >= 0; j--) {
            const double *pj = P + j * nsrow;
            double sum = xs[j];
            for (i = j + 1; i < nscol; i++) sum -= pj[i] * xs[i];
            xs[j] = sum;
        }
    }
}

void ldl_solve(ldl_t *f, double *b)
{
    Int n, k;
    const Int *perm;
    double *x;

    if (!f || !b) return;
    n = f->n;
    perm = f->perm;
    x = f->xwork;
    for (k = 0; k < n; k++) x[k] = b[perm[k]];
    if (f->mode == 1) {
        solve_super(f, x);
    } else {
        solve_up(f, x);
    }
    for (k = 0; k < n; k++) b[perm[k]] = x[k];
}

int64_t ldl_nnz_L(const ldl_t *f)
{
    if (!f) return 0;
    return f->nnzL;
}

const int64_t *ldl_perm(const ldl_t *f)
{
    if (!f) return NULL;
    return f->perm;
}
