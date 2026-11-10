// Sparse matrices, approximate minimum degree ordering and LDL^T factorization.
#pragma once
#include <cmath>
#include <vector>
#include <algorithm>
#include "io.hpp"

struct CSC {
  I m = 0, n = 0;
  std::vector<I> p, i;
  std::vector<double> x;
  I nnz() const { return p.empty() ? 0 : p[n]; }
};

static CSC transpose(const CSC &A) {
  CSC T;
  T.m = A.n; T.n = A.m;
  T.p.assign(A.m + 1, 0);
  I nz = A.nnz();
  T.i.resize(nz); T.x.resize(nz);
  for (I k = 0; k < nz; k++) T.p[A.i[k] + 1]++;
  for (I r = 0; r < A.m; r++) T.p[r + 1] += T.p[r];
  std::vector<I> w(T.p.begin(), T.p.end() - 1);
  for (I j = 0; j < A.n; j++)
    for (I k = A.p[j]; k < A.p[j + 1]; k++) {
      I q = w[A.i[k]]++;
      T.i[q] = j; T.x[q] = A.x[k];
    }
  return T;
}

// sort row indices and sum duplicates (two transposes), drop exact zeros
static CSC canonical(const CSC &A) {
  CSC T = transpose(transpose(A));
  CSC R; R.m = T.m; R.n = T.n; R.p.assign(T.n + 1, 0);
  R.i.reserve(T.nnz()); R.x.reserve(T.nnz());
  for (I j = 0; j < T.n; j++) {
    I start = (I)R.i.size();
    for (I k = T.p[j]; k < T.p[j + 1]; k++) {
      if ((I)R.i.size() > start && R.i.back() == T.i[k]) R.x.back() += T.x[k];
      else { R.i.push_back(T.i[k]); R.x.push_back(T.x[k]); }
    }
    // drop zeros
    I w = start;
    for (I k = start; k < (I)R.i.size(); k++)
      if (R.x[k] != 0.0) { R.i[w] = R.i[k]; R.x[w] = R.x[k]; w++; }
    R.i.resize(w); R.x.resize(w);
    R.p[j + 1] = w;
  }
  return R;
}

// y += A x
static inline void spmv_add(const CSC &A, const double *x, double *y) {
  for (I j = 0; j < A.n; j++) {
    double xj = x[j];
    if (xj == 0.0) continue;
    for (I k = A.p[j]; k < A.p[j + 1]; k++) y[A.i[k]] += A.x[k] * xj;
  }
}
// y += A' x
static inline void spmtv_add(const CSC &A, const double *x, double *y) {
  for (I j = 0; j < A.n; j++) {
    double s = 0;
    for (I k = A.p[j]; k < A.p[j + 1]; k++) s += A.x[k] * x[A.i[k]];
    y[j] += s;
  }
}

// ---------------- AMD ordering (approximate minimum degree, quotient graph) ----------------
#define AMD_FLIP(i) (-(i)-2)
static I amd_wclear(I mark, I lemax, I *w, I n) {
  if (mark < 2 || (mark + lemax < 0)) {
    for (I k = 0; k < n; k++) if (w[k] != 0) w[k] = 1;
    mark = 2;
  }
  return mark;
}
static I amd_tdfs(I j, I k, I *head, const I *next, I *post, I *stack) {
  I top = 0;
  stack[0] = j;
  while (top >= 0) {
    I p = stack[top];
    I i = head[p];
    if (i == -1) { top--; post[k++] = p; }
    else { head[p] = next[i]; stack[++top] = i; }
  }
  return k;
}

// Cp, Ci: symmetric pattern (both triangles), no diagonal. returns perm (new -> old)
static std::vector<I> amd_order(I n, std::vector<I> Cp, std::vector<I> Ci) {
  std::vector<I> Pv(n + 1);
  if (n == 0) return std::vector<I>();
  I dense = (I)std::max(16.0, 10.0 * std::sqrt((double)n));
  dense = std::min(n - 2, dense);
  I cnz = Cp[n];
  I t = cnz + cnz / 5 + 2 * n;
  Ci.resize(std::max<I>(t, 1));
  I nzmax = (I)Ci.size();
  std::vector<I> W(8 * (n + 1));
  I *len = &W[0], *nv = &W[n + 1], *next = &W[2 * (n + 1)], *head = &W[3 * (n + 1)], *elen = &W[4 * (n + 1)],
    *degree = &W[5 * (n + 1)], *w = &W[6 * (n + 1)], *hhead = &W[7 * (n + 1)], *last = Pv.data();
  I *cp = Cp.data(), *ci;
  I d, dk, dext, lemax = 0, e, elenk, eln, i, j, k, k1, k2, k3, jlast, ln, mindeg = 0, nvi, nvj, nvk, mark, wnvi, ok, nel = 0,
    p, p1, p2, p3, p4, pj, pk, pk1, pk2, pn, q, h;
  for (k = 0; k < n; k++) len[k] = cp[k + 1] - cp[k];
  len[n] = 0;
  ci = Ci.data();
  for (i = 0; i <= n; i++) {
    head[i] = -1; last[i] = -1; next[i] = -1; hhead[i] = -1; nv[i] = 1; w[i] = 1; elen[i] = 0; degree[i] = len[i];
  }
  mark = amd_wclear(0, 0, w, n);
  elen[n] = -2; cp[n] = -1; w[n] = 0;
  for (i = 0; i < n; i++) {
    d = degree[i];
    if (d == 0) { elen[i] = -2; nel++; cp[i] = -1; w[i] = 0; }
    else if (d > dense) { nv[i] = 0; elen[i] = -1; nel++; cp[i] = AMD_FLIP(n); nv[n]++; }
    else {
      if (head[d] != -1) last[head[d]] = i;
      next[i] = head[d]; head[d] = i;
    }
  }
  while (nel < n) {
    for (k = -1; mindeg < n && (k = head[mindeg]) == -1; mindeg++);
    if (next[k] != -1) last[next[k]] = -1;
    head[mindeg] = next[k];
    elenk = elen[k]; nvk = nv[k]; nel += nvk;
    if (elenk > 0 && cnz + mindeg >= nzmax) {
      for (j = 0; j < n; j++) {
        if ((p = cp[j]) >= 0) { cp[j] = ci[p]; ci[p] = AMD_FLIP(j); }
      }
      for (q = 0, p = 0; p < cnz;) {
        if ((j = AMD_FLIP(ci[p++])) >= 0) {
          ci[q] = cp[j]; cp[j] = q++;
          for (k3 = 0; k3 < len[j] - 1; k3++) ci[q++] = ci[p++];
        }
      }
      cnz = q;
    }
    dk = 0;
    nv[k] = -nvk;
    p = cp[k];
    pk1 = (elenk == 0) ? p : cnz;
    pk2 = pk1;
    for (k1 = 1; k1 <= elenk + 1; k1++) {
      if (k1 > elenk) { e = k; pj = p; ln = len[k] - elenk; }
      else { e = ci[p++]; pj = cp[e]; ln = len[e]; }
      for (k2 = 1; k2 <= ln; k2++) {
        i = ci[pj++];
        if ((nvi = nv[i]) <= 0) continue;
        dk += nvi; nv[i] = -nvi; ci[pk2++] = i;
        if (next[i] != -1) last[next[i]] = last[i];
        if (last[i] != -1) next[last[i]] = next[i];
        else head[degree[i]] = next[i];
      }
      if (e != k) { cp[e] = AMD_FLIP(k); w[e] = 0; }
    }
    if (elenk != 0) cnz = pk2;
    degree[k] = dk; cp[k] = pk1; len[k] = pk2 - pk1; elen[k] = -2;
    mark = amd_wclear(mark, lemax, w, n);
    for (pk = pk1; pk < pk2; pk++) {
      i = ci[pk];
      if ((eln = elen[i]) <= 0) continue;
      nvi = -nv[i]; wnvi = mark - nvi;
      for (p = cp[i]; p <= cp[i] + eln - 1; p++) {
        e = ci[p];
        if (w[e] >= mark) w[e] -= nvi;
        else if (w[e] != 0) w[e] = degree[e] + wnvi;
      }
    }
    for (pk = pk1; pk < pk2; pk++) {
      i = ci[pk];
      p1 = cp[i]; p2 = p1 + elen[i] - 1; pn = p1;
      for (h = 0, d = 0, p = p1; p <= p2; p++) {
        e = ci[p];
        if (w[e] != 0) {
          dext = w[e] - mark;
          if (dext > 0) { d += dext; ci[pn++] = e; h += e; }
          else { cp[e] = AMD_FLIP(k); w[e] = 0; }
        }
      }
      elen[i] = pn - p1 + 1;
      p3 = pn; p4 = p1 + len[i];
      for (p = p2 + 1; p < p4; p++) {
        j = ci[p];
        if ((nvj = nv[j]) <= 0) continue;
        d += nvj; ci[pn++] = j; h += j;
      }
      if (d == 0) {
        cp[i] = AMD_FLIP(k); nvi = -nv[i]; dk -= nvi; nvk += nvi; nel += nvi; nv[i] = 0; elen[i] = -1;
      } else {
        degree[i] = std::min(degree[i], d);
        ci[pn] = ci[p3]; ci[p3] = ci[p1]; ci[p1] = k;
        len[i] = pn - p1 + 1;
        h = ((h < 0) ? (-h) : h) % n;
        next[i] = hhead[h]; hhead[h] = i; last[i] = h;
      }
    }
    degree[k] = dk;
    lemax = std::max(lemax, dk);
    mark = amd_wclear(mark + lemax, lemax, w, n);
    for (pk = pk1; pk < pk2; pk++) {
      i = ci[pk];
      if (nv[i] >= 0) continue;
      h = last[i]; i = hhead[h]; hhead[h] = -1;
      for (; i != -1 && next[i] != -1; i = next[i], mark++) {
        ln = len[i]; eln = elen[i];
        for (p = cp[i] + 1; p <= cp[i] + ln - 1; p++) w[ci[p]] = mark;
        jlast = i;
        for (j = next[i]; j != -1;) {
          ok = (len[j] == ln) && (elen[j] == eln);
          for (p = cp[j] + 1; ok && p <= cp[j] + ln - 1; p++) if (w[ci[p]] != mark) ok = 0;
          if (ok) {
            cp[j] = AMD_FLIP(i); nv[i] += nv[j]; nv[j] = 0; elen[j] = -1;
            j = next[j]; next[jlast] = j;
          } else { jlast = j; j = next[j]; }
        }
      }
    }
    for (p = pk1, pk = pk1; pk < pk2; pk++) {
      i = ci[pk];
      if ((nvi = -nv[i]) <= 0) continue;
      nv[i] = nvi;
      d = degree[i] + dk - nvi;
      d = std::min(d, n - nel - nvi);
      if (head[d] != -1) last[head[d]] = i;
      next[i] = head[d]; last[i] = -1; head[d] = i;
      mindeg = std::min(mindeg, d);
      degree[i] = d;
      ci[p++] = i;
    }
    nv[k] = nvk;
    if ((len[k] = p - pk1) == 0) { cp[k] = -1; w[k] = 0; }
    if (elenk != 0) cnz = p;
  }
  for (i = 0; i < n; i++) cp[i] = AMD_FLIP(cp[i]);
  for (j = 0; j <= n; j++) head[j] = -1;
  for (j = n; j >= 0; j--) {
    if (nv[j] > 0) continue;
    next[j] = head[cp[j]]; head[cp[j]] = j;
  }
  for (e = n; e >= 0; e--) {
    if (nv[e] <= 0) continue;
    if (cp[e] != -1) { next[e] = head[cp[e]]; head[cp[e]] = e; }
  }
  for (k = 0, i = 0; i <= n; i++) {
    if (cp[i] == -1) k = amd_tdfs(i, k, head, next, Pv.data(), w);
  }
  Pv.resize(n);
  return Pv;
}

// ---------------- LDL^T (up-looking, elimination tree) ----------------
struct LDL {
  I n = 0;
  std::vector<I> etree, Lnz, Lp, Li;
  std::vector<double> Lx, D, Dinv;
  std::vector<I> iwork; std::vector<char> bwork; std::vector<double> fwork;
  // Ap/Ai upper triangular CSC (with diagonal in every column)
  bool symbolic(I n_, const std::vector<I> &Ap, const std::vector<I> &Ai) {
    n = n_;
    etree.assign(n, -1); Lnz.assign(n, 0);
    std::vector<I> work(n, 0);
    for (I j = 0; j < n; j++) {
      work[j] = j;
      for (I p = Ap[j]; p < Ap[j + 1]; p++) {
        I i = Ai[p];
        if (i > j) return false;
        while (work[i] != j) {
          if (etree[i] == -1) etree[i] = j;
          Lnz[i]++;
          work[i] = j;
          i = etree[i];
        }
      }
    }
    Lp.assign(n + 1, 0);
    for (I i = 0; i < n; i++) Lp[i + 1] = Lp[i] + Lnz[i];
    Li.resize(std::max<I>(Lp[n], 1)); Lx.resize(std::max<I>(Lp[n], 1));
    D.resize(n); Dinv.resize(n);
    iwork.resize(3 * n); bwork.resize(n); fwork.resize(n);
    return true;
  }
  // sign: +1/-1 expected pivot sign; returns number of dynamic regularizations
  I factor(const std::vector<I> &Ap, const std::vector<I> &Ai, const std::vector<double> &Ax, const std::vector<double> &sign,
           double deps, double ddelta) {
    I *yIdx = iwork.data(), *elimBuffer = iwork.data() + n, *LNext = iwork.data() + 2 * n;
    char *yMarkers = bwork.data();
    double *yVals = fwork.data();
    I nreg = 0;
    for (I i = 0; i < n; i++) { yMarkers[i] = 0; yVals[i] = 0.0; D[i] = 0.0; LNext[i] = Lp[i]; }
    for (I k = 0; k < n; k++) {
      I nnzY = 0;
      for (I i = Ap[k]; i < Ap[k + 1]; i++) {
        I bidx = Ai[i];
        if (bidx == k) { D[k] += Ax[i]; continue; }
        yVals[bidx] += Ax[i];
        I nextIdx = bidx;
        if (!yMarkers[nextIdx]) {
          yMarkers[nextIdx] = 1;
          elimBuffer[0] = nextIdx;
          I nnzE = 1;
          nextIdx = etree[bidx];
          while (nextIdx != -1 && nextIdx < k) {
            if (yMarkers[nextIdx]) break;
            yMarkers[nextIdx] = 1;
            elimBuffer[nnzE++] = nextIdx;
            nextIdx = etree[nextIdx];
          }
          while (nnzE) yIdx[nnzY++] = elimBuffer[--nnzE];
        }
      }
      for (I i = nnzY - 1; i >= 0; i--) {
        I cidx = yIdx[i];
        I tmpIdx = LNext[cidx];
        double yc = yVals[cidx];
        for (I j = Lp[cidx]; j < tmpIdx; j++) yVals[Li[j]] -= Lx[j] * yc;
        Li[tmpIdx] = k;
        Lx[tmpIdx] = yc * Dinv[cidx];
        D[k] -= yc * Lx[tmpIdx];
        LNext[cidx]++;
        yVals[cidx] = 0.0;
        yMarkers[cidx] = 0;
      }
      if (!(D[k] * sign[k] > deps)) { D[k] = sign[k] * ddelta; nreg++; }
      Dinv[k] = 1.0 / D[k];
    }
    return nreg;
  }
  void solve(double *x) const {
    for (I j = 0; j < n; j++) {
      double xj = x[j];
      if (xj != 0.0) for (I p = Lp[j]; p < Lp[j + 1]; p++) x[Li[p]] -= Lx[p] * xj;
    }
    for (I j = 0; j < n; j++) x[j] *= Dinv[j];
    for (I j = n - 1; j >= 0; j--) {
      double s = x[j];
      for (I p = Lp[j]; p < Lp[j + 1]; p++) s -= Lx[p] * x[Li[p]];
      x[j] = s;
    }
  }
};
