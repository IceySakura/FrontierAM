// Sparse utilities, approximate minimum degree ordering (after CSparse cs_amd), and
// an up-looking LDL^T factorization (after QDLDL) with dynamic regularization.
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

typedef int64_t I;

struct CSC {
  I m = 0, n = 0;
  std::vector<I> p, i;
  std::vector<double> x;
  I nnz() const { return p.empty() ? 0 : p[n]; }
};

static CSC csc_transpose(const CSC& A) {
  CSC T; T.m = A.n; T.n = A.m;
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

// sorted indices, duplicates summed, explicit zeros dropped
static CSC csc_canon(const CSC& A) {
  CSC B = csc_transpose(csc_transpose(A));
  CSC C; C.m = B.m; C.n = B.n; C.p.assign(B.n + 1, 0);
  C.i.reserve(B.nnz()); C.x.reserve(B.nnz());
  for (I j = 0; j < B.n; j++) {
    I start = (I)C.i.size();
    for (I k = B.p[j]; k < B.p[j + 1]; k++) {
      if ((I)C.i.size() > start && C.i.back() == B.i[k]) C.x.back() += B.x[k];
      else { C.i.push_back(B.i[k]); C.x.push_back(B.x[k]); }
    }
    // drop zeros
    I w = start;
    for (I k = start; k < (I)C.i.size(); k++) if (C.x[k] != 0.0) { C.i[w] = C.i[k]; C.x[w] = C.x[k]; w++; }
    C.i.resize(w); C.x.resize(w);
    C.p[j + 1] = w;
  }
  return C;
}

// y += A*x
static void spmv_add(const CSC& A, const double* x, double* y) {
  for (I j = 0; j < A.n; j++) {
    double xj = x[j];
    if (xj == 0.0) continue;
    for (I k = A.p[j]; k < A.p[j + 1]; k++) y[A.i[k]] += A.x[k] * xj;
  }
}
// y += A'*x
static void spmtv_add(const CSC& A, const double* x, double* y) {
  for (I j = 0; j < A.n; j++) {
    double s = 0;
    for (I k = A.p[j]; k < A.p[j + 1]; k++) s += A.x[k] * x[A.i[k]];
    y[j] += s;
  }
}

// ---------------- AMD ----------------
#define AMD_FLIP(i) (-(i)-2)
static I amd_wclear(I mark, I lemax, I* w, I n) {
  if (mark < 2 || (mark + lemax < 0)) {
    for (I k = 0; k < n; k++) if (w[k] != 0) w[k] = 1;
    mark = 2;
  }
  return mark;
}
static I amd_tdfs(I j, I k, I* head, const I* next, I* post, I* stack) {
  I i, p, top = 0;
  stack[0] = j;
  while (top >= 0) {
    p = stack[top];
    i = head[p];
    if (i == -1) { top--; post[k++] = p; }
    else { head[p] = next[i]; stack[++top] = i; }
  }
  return k;
}
// Cp (n+1), Ci: symmetric pattern, both triangles, no diagonal. Returns permutation (new->old).
static std::vector<I> amd_order(I n, std::vector<I> Cpv, std::vector<I> Civ) {
  std::vector<I> perm(n);
  if (n <= 3) { for (I k = 0; k < n; k++) perm[k] = k; return perm; }
  I cnz = Cpv[n];
  I nzmax = cnz + cnz / 5 + 2 * n + 1;
  Civ.resize(nzmax);
  I* Cp = Cpv.data(); I* Ci = Civ.data();
  std::vector<I> Pv(n + 1), Wv(8 * (n + 1));
  I* P = Pv.data(); I* W = Wv.data();
  I *len = W, *nv = W + (n + 1), *next = W + 2 * (n + 1), *head = W + 3 * (n + 1),
    *elen = W + 4 * (n + 1), *degree = W + 5 * (n + 1), *w = W + 6 * (n + 1), *hhead = W + 7 * (n + 1);
  I* last = P;
  I d, dk, dext, lemax = 0, e, elenk, eln, i, j, k, k1, k2, k3, jlast, ln, dense, mindeg = 0, nvi, nvj, nvk, mark, wnvi, ok, nel = 0, p, p1, p2, p3, p4, pj, pk, pk1, pk2, pn, q, h;
  dense = std::max((I)16, (I)(10 * std::sqrt((double)n)));
  dense = std::min(n - 2, dense);
  for (k = 0; k < n; k++) len[k] = Cp[k + 1] - Cp[k];
  len[n] = 0;
  for (i = 0; i <= n; i++) {
    head[i] = -1; last[i] = -1; next[i] = -1; hhead[i] = -1;
    nv[i] = 1; w[i] = 1; elen[i] = 0; degree[i] = len[i];
  }
  mark = amd_wclear(0, 0, w, n);
  elen[n] = -2; Cp[n] = -1; w[n] = 0;
  for (i = 0; i < n; i++) {
    d = degree[i];
    if (d == 0) { elen[i] = -2; nel++; Cp[i] = -1; w[i] = 0; }
    else if (d > dense) { nv[i] = 0; elen[i] = -1; nel++; Cp[i] = AMD_FLIP(n); nv[n]++; }
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
        if ((p = Cp[j]) >= 0) { Cp[j] = Ci[p]; Ci[p] = AMD_FLIP(j); }
      }
      for (q = 0, p = 0; p < cnz;) {
        if ((j = AMD_FLIP(Ci[p++])) >= 0) {
          Ci[q] = Cp[j]; Cp[j] = q++;
          for (k3 = 0; k3 < len[j] - 1; k3++) Ci[q++] = Ci[p++];
        }
      }
      cnz = q;
    }
    dk = 0;
    nv[k] = -nvk;
    p = Cp[k];
    pk1 = (elenk == 0) ? p : cnz;
    pk2 = pk1;
    for (k1 = 1; k1 <= elenk + 1; k1++) {
      if (k1 > elenk) { e = k; pj = p; ln = len[k] - elenk; }
      else { e = Ci[p++]; pj = Cp[e]; ln = len[e]; }
      for (k2 = 1; k2 <= ln; k2++) {
        i = Ci[pj++];
        if ((nvi = nv[i]) <= 0) continue;
        dk += nvi; nv[i] = -nvi; Ci[pk2++] = i;
        if (next[i] != -1) last[next[i]] = last[i];
        if (last[i] != -1) next[last[i]] = next[i];
        else head[degree[i]] = next[i];
      }
      if (e != k) { Cp[e] = AMD_FLIP(k); w[e] = 0; }
    }
    if (elenk != 0) cnz = pk2;
    degree[k] = dk; Cp[k] = pk1; len[k] = pk2 - pk1; elen[k] = -2;
    mark = amd_wclear(mark, lemax, w, n);
    for (pk = pk1; pk < pk2; pk++) {
      i = Ci[pk];
      if ((eln = elen[i]) <= 0) continue;
      nvi = -nv[i]; wnvi = mark - nvi;
      for (p = Cp[i]; p <= Cp[i] + eln - 1; p++) {
        e = Ci[p];
        if (w[e] >= mark) w[e] -= nvi;
        else if (w[e] != 0) w[e] = degree[e] + wnvi;
      }
    }
    for (pk = pk1; pk < pk2; pk++) {
      i = Ci[pk];
      p1 = Cp[i]; p2 = p1 + elen[i] - 1; pn = p1;
      for (h = 0, d = 0, p = p1; p <= p2; p++) {
        e = Ci[p];
        if (w[e] != 0) {
          dext = w[e] - mark;
          if (dext > 0) { d += dext; Ci[pn++] = e; h += e; }
          else { Cp[e] = AMD_FLIP(k); w[e] = 0; }
        }
      }
      elen[i] = pn - p1 + 1;
      p3 = pn; p4 = p1 + len[i];
      for (p = p2 + 1; p < p4; p++) {
        j = Ci[p];
        if ((nvj = nv[j]) <= 0) continue;
        d += nvj; Ci[pn++] = j; h += j;
      }
      if (d == 0) {
        Cp[i] = AMD_FLIP(k); nvi = -nv[i]; dk -= nvi; nvk += nvi; nel += nvi; nv[i] = 0; elen[i] = -1;
      } else {
        degree[i] = std::min(degree[i], d);
        Ci[pn] = Ci[p3]; Ci[p3] = Ci[p1]; Ci[p1] = k;
        len[i] = pn - p1 + 1;
        h = ((h < 0) ? (-h) : h) % n;
        next[i] = hhead[h]; hhead[h] = i; last[i] = h;
      }
    }
    degree[k] = dk;
    lemax = std::max(lemax, dk);
    mark = amd_wclear(mark + lemax, lemax, w, n);
    for (pk = pk1; pk < pk2; pk++) {
      i = Ci[pk];
      if (nv[i] >= 0) continue;
      h = last[i]; i = hhead[h]; hhead[h] = -1;
      for (; i != -1 && next[i] != -1; i = next[i], mark++) {
        ln = len[i]; eln = elen[i];
        for (p = Cp[i] + 1; p <= Cp[i] + ln - 1; p++) w[Ci[p]] = mark;
        jlast = i;
        for (j = next[i]; j != -1;) {
          ok = (len[j] == ln) && (elen[j] == eln);
          for (p = Cp[j] + 1; ok && p <= Cp[j] + ln - 1; p++) if (w[Ci[p]] != mark) ok = 0;
          if (ok) {
            Cp[j] = AMD_FLIP(i); nv[i] += nv[j]; nv[j] = 0; elen[j] = -1;
            j = next[j]; next[jlast] = j;
          } else { jlast = j; j = next[j]; }
        }
      }
    }
    for (p = pk1, pk = pk1; pk < pk2; pk++) {
      i = Ci[pk];
      if ((nvi = -nv[i]) <= 0) continue;
      nv[i] = nvi;
      d = degree[i] + dk - nvi;
      d = std::min(d, n - nel - nvi);
      if (head[d] != -1) last[head[d]] = i;
      next[i] = head[d]; last[i] = -1; head[d] = i;
      mindeg = std::min(mindeg, d);
      degree[i] = d;
      Ci[p++] = i;
    }
    nv[k] = nvk;
    if ((len[k] = p - pk1) == 0) { Cp[k] = -1; w[k] = 0; }
    if (elenk != 0) cnz = p;
  }
  for (i = 0; i < n; i++) Cp[i] = AMD_FLIP(Cp[i]);
  for (j = 0; j <= n; j++) head[j] = -1;
  for (j = n; j >= 0; j--) {
    if (nv[j] > 0) continue;
    next[j] = head[Cp[j]]; head[Cp[j]] = j;
  }
  for (e = n; e >= 0; e--) {
    if (nv[e] <= 0) continue;
    if (Cp[e] != -1) { next[e] = head[Cp[e]]; head[Cp[e]] = e; }
  }
  for (k = 0, i = 0; i <= n; i++) {
    if (Cp[i] == -1) k = amd_tdfs(i, k, head, next, P, w);
  }
  // extract permutation of 0..n-1 (drop the dummy node n)
  std::vector<char> seen(n, 0);
  I c = 0;
  bool okp = true;
  for (I t = 0; t <= n && t < (I)Pv.size(); t++) {
    I v = P[t];
    if (v == n) continue;
    if (v < 0 || v >= n || seen[v]) { okp = false; break; }
    seen[v] = 1;
    if (c < n) perm[c++] = v;
  }
  if (!okp || c != n) { for (I t = 0; t < n; t++) perm[t] = t; }
  return perm;
}

// ---------------- LDL ----------------
struct LDL {
  I N = 0;
  std::vector<I> Lp, Li, etree, Lnz, LNext, yIdx, elimBuf;
  std::vector<double> Lx, D, Dinv, yVals;
  std::vector<char> yMark;
  I nreg = 0;
  bool symbolic(I n, const std::vector<I>& Kp, const std::vector<I>& Ki) {
    N = n;
    etree.assign(N, -1); Lnz.assign(N, 0);
    std::vector<I> work(N, 0);
    for (I j = 0; j < N; j++) {
      work[j] = j;
      for (I p = Kp[j]; p < Kp[j + 1]; p++) {
        I i = Ki[p];
        if (i > j) return false;
        while (work[i] != j) {
          if (etree[i] == -1) etree[i] = j;
          Lnz[i]++;
          work[i] = j;
          i = etree[i];
        }
      }
    }
    Lp.assign(N + 1, 0);
    for (I i = 0; i < N; i++) Lp[i + 1] = Lp[i] + Lnz[i];
    Li.resize(std::max<I>(Lp[N], 1)); Lx.resize(std::max<I>(Lp[N], 1));
    D.assign(N, 0); Dinv.assign(N, 0); yVals.assign(N, 0); yMark.assign(N, 0);
    LNext.resize(N); yIdx.resize(N); elimBuf.resize(N);
    return true;
  }
  // sign[k] = +1/-1 expected pivot sign; regularize when sign*D < eps
  void factor(const std::vector<I>& Kp, const std::vector<I>& Ki, const std::vector<double>& Kx,
              const std::vector<signed char>& sign, double eps, double delta) {
    nreg = 0;
    for (I i = 0; i < N; i++) { LNext[i] = Lp[i]; yMark[i] = 0; yVals[i] = 0; }
    for (I k = 0; k < N; k++) {
      I nnzY = 0;
      D[k] = 0;
      for (I p = Kp[k]; p < Kp[k + 1]; p++) {
        I bidx = Ki[p];
        if (bidx == k) { D[k] = Kx[p]; continue; }
        yVals[bidx] = Kx[p];
        I nextIdx = bidx;
        if (!yMark[nextIdx]) {
          yMark[nextIdx] = 1; elimBuf[0] = nextIdx; I nnzE = 1;
          nextIdx = etree[bidx];
          while (nextIdx != -1 && nextIdx < k) {
            if (yMark[nextIdx]) break;
            yMark[nextIdx] = 1; elimBuf[nnzE++] = nextIdx;
            nextIdx = etree[nextIdx];
          }
          while (nnzE) yIdx[nnzY++] = elimBuf[--nnzE];
        }
      }
      double Dk = D[k];
      for (I t = nnzY - 1; t >= 0; t--) {
        I cidx = yIdx[t];
        I tmpIdx = LNext[cidx];
        double yc = yVals[cidx];
        const I* li = Li.data(); const double* lx = Lx.data();
        double* yv = yVals.data();
        for (I j = Lp[cidx]; j < tmpIdx; j++) yv[li[j]] -= lx[j] * yc;
        Li[tmpIdx] = k;
        double l = yc * Dinv[cidx];
        Lx[tmpIdx] = l;
        Dk -= yc * l;
        LNext[cidx]++;
        yVals[cidx] = 0; yMark[cidx] = 0;
      }
      if (!(sign[k] * Dk >= eps)) { Dk = sign[k] * delta; nreg++; }
      D[k] = Dk; Dinv[k] = 1.0 / Dk;
    }
  }
  void solve(double* x) const {
    for (I i = 0; i < N; i++) {
      double xi = x[i];
      if (xi != 0.0) for (I j = Lp[i]; j < Lp[i + 1]; j++) x[Li[j]] -= Lx[j] * xi;
    }
    for (I i = 0; i < N; i++) x[i] *= Dinv[i];
    for (I i = N - 1; i >= 0; i--) {
      double s = x[i];
      for (I j = Lp[i]; j < Lp[i + 1]; j++) s -= Lx[j] * x[Li[j]];
      x[i] = s;
    }
  }
};
