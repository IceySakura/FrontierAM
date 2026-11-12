#ifndef AMD_H
#define AMD_H

/* Approximate minimum degree ordering (port of CSparse cs_amd, order 1 semantics:
   AMD of the symmetric pattern A+A' with the diagonal removed).

   n x n pattern in CSC (Ap length n+1, Ai row indices). The input may be the upper
   triangle only, the lower triangle only, or the full matrix; the diagonal may or may
   not be present; row indices may be unsorted; duplicates are allowed.
   Output: perm[k] = original index of the k-th pivot (perm has length n).
   Returns 0 on success, -1 on allocation failure / int overflow / invalid input. */
int amd_order(int n, const int *Ap, const int *Ai, int *perm);

#endif
