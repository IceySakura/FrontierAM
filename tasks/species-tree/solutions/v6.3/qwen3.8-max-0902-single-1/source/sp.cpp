// Species-tree inference from gene trees under the multispecies coalescent.
//
// Method:
//   1. Parse the gene trees (Newick, unrooted binary, 250 leaves).
//   2. Build a starting tree: average internode distances across gene trees
//      (ASTRID/NJst-style) + neighbor joining.
//   3. Optimize the exact quartet support score (the ASTRAL objective):
//      score(T) = sum over internal splits B|C of T of w(B|C), where w(B|C)
//      is the number of (gene tree, quartet) pairs in which the quartet is
//      resolved as B-side pair | C-side pair by the gene tree.
//      w(B|C) per gene tree is computed in O(#nodes): for each degree-3 node
//      with incident components (P,Q,R), every resolved quartet has its two
//      bifurcation nodes; counting "pair-in-one-component, singles in the
//      other two" at each node counts each quartet exactly twice, so
//      w = (1/2) * sum over nodes v, over choice of pair-component k:
//          b_i*b_j*C(c_k,2) + c_i*c_j*C(b_k,2)   {i,j} = other two comps.
//   4. Search: NNI hill climbing (an NNI changes exactly one internal split,
//      so delta = w(new) - w(old), with memoized w), plus randomized SPR
//      perturbation and re-climbing, until a wall-clock deadline.
//
// Single-threaded, deterministic given SEED. Logs go to stderr.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <vector>
#include <array>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <chrono>

using namespace std;
using ll = long long;
using u64 = uint64_t;

static const int MAXN = 256;       // max leaves supported (input has 250)
static const int W = MAXN / 64;    // words per bitmask
static const int MAXV = 2 * MAXN;  // max species-tree nodes (leaves + internal)

int n = 0;              // number of leaves
int G = 0;              // number of gene trees
double DEADLINE = 115.0; // seconds of wall clock for the search phase

static auto T0 = chrono::steady_clock::now();
inline double elapsed() {
  return chrono::duration<double>(chrono::steady_clock::now() - T0).count();
}

// ---------------- bitmask helpers ----------------
struct Mask { u64 w[W]; };
inline bool operator==(const Mask& a, const Mask& b) {
  for (int i = 0; i < W; i++) if (a.w[i] != b.w[i]) return false;
  return true;
}
struct MaskHash {
  size_t operator()(const Mask& m) const {
    u64 h = 1469598103934665603ULL;
    for (int i = 0; i < W; i++) {
      u64 x = m.w[i] * 0x9E3779B97F4A7C15ULL;
      h = (h ^ x) * 1099511628211ULL;
    }
    return (size_t)h;
  }
};
inline void clearMask(Mask& m) { memset(&m, 0, sizeof(Mask)); }
inline void setBit(Mask& m, int i) { m.w[i >> 6] |= 1ULL << (i & 63); }
inline int testBit(const Mask& m, int i) { return (m.w[i >> 6] >> (i & 63)) & 1; }
inline void orMask(Mask& d, const Mask& a, const Mask& b) {
  for (int i = 0; i < W; i++) d.w[i] = a.w[i] | b.w[i];
}
inline Mask orMaskF(const Mask& a, const Mask& b) { Mask d; orMask(d, a, b); return d; }
inline void notMaskN(Mask& d, const Mask& a, int nbits) {
  for (int i = 0; i < W; i++) d.w[i] = ~a.w[i];
  if (nbits & 63) d.w[nbits >> 6] &= (1ULL << (nbits & 63)) - 1;
}
inline int popMask(const Mask& a) {
  int s = 0;
  for (int i = 0; i < W; i++) s += __builtin_popcountll(a.w[i]);
  return s;
}

static Mask ALLMASK; // bits 0..n-1 set

// ---------------- gene trees ----------------
struct Gene {
  int m = 0;                       // nodes allocated
  int root = 0;
  vector<array<int, 3>> ch;        // children; ch[v] = {c0,c1,c2}, root has 3, internal 2, leaves have ch[0]==-1
  vector<int> par, nl, leafid;     // leafid[v] >= 0 for leaves
  vector<int> post;                // parents after children (reverse preorder)
  vector<int> internalNodes;       // all degree-3 nodes incl. root
  vector<array<int, 3>> adj;       // unrooted adjacency (for internode distances)
  vector<int> deg;
  vector<int> b;                   // scratch for scoring
};
static vector<Gene> genes;
static vector<string> labels;
static unordered_map<string, int> lab2id;

// Parse one Newick line into a Gene, rooted at an internal degree-3 node.
static int parseNode(Gene& g, const string& s, int& pos) {
  if (s[pos] == '(') {
    pos++;
    vector<int> kids;
    kids.push_back(parseNode(g, s, pos));
    while (pos < (int)s.size() && s[pos] == ',') { pos++; kids.push_back(parseNode(g, s, pos)); }
    if (pos >= (int)s.size() || s[pos] != ')') throw runtime_error("bad newick: ) expected");
    pos++;
    while (pos < (int)s.size() && s[pos] != ':' && s[pos] != ',' && s[pos] != ')' && s[pos] != ';') pos++;
    if (pos < (int)s.size() && s[pos] == ':') {
      pos++;
      while (pos < (int)s.size() && (isdigit((unsigned char)s[pos]) || s[pos] == '.' || s[pos] == 'e' || s[pos] == 'E' || s[pos] == '-' || s[pos] == '+')) pos++;
    }
    int v = g.m++;
    g.ch[v][0] = kids[0];
    g.ch[v][1] = kids.size() > 1 ? kids[1] : -1;
    g.ch[v][2] = kids.size() > 2 ? kids[2] : -1;
    if (kids.size() > 3) throw runtime_error("unexpected multifurcation >3");
    for (int k : kids) g.par[k] = v;
    return v;
  } else {
    int st = pos;
    while (pos < (int)s.size() && s[pos] != ':' && s[pos] != ',' && s[pos] != ')' && s[pos] != ';') pos++;
    string lab = s.substr(st, pos - st);
    // trim
    size_t a = lab.find_first_not_of(" \t\r\n");
    size_t b = lab.find_last_not_of(" \t\r\n");
    if (a == string::npos) throw runtime_error("empty leaf label");
    lab = lab.substr(a, b - a + 1);
    if (pos < (int)s.size() && s[pos] == ':') {
      pos++;
      while (pos < (int)s.size() && (isdigit((unsigned char)s[pos]) || s[pos] == '.' || s[pos] == 'e' || s[pos] == 'E' || s[pos] == '-' || s[pos] == '+')) pos++;
    }
    auto it = lab2id.find(lab);
    if (it == lab2id.end()) {
      if ((int)lab2id.size() >= MAXN) throw runtime_error("too many labels");
      it = lab2id.emplace(lab, (int)labels.size()).first;
      labels.push_back(lab);
    }
    int v = g.m++;
    g.leafid[v] = it->second;
    return v;
  }
}

static void finishGene(Gene& g) {
  // g was parsed with an outer node that may have only 2 children; re-root so
  // that the root has exactly 3 children.
  // First find the outermost node: it was created last? No -- parseNode returns
  // it; we stored it as g.root before calling finishGene.
  int t = g.root;
  if (g.ch[t][1] >= 0 && g.ch[t][2] < 0) {
    // degree-2 serialization root: absorb one internal child
    int A = g.ch[t][0], B = g.ch[t][1];
    int victim = (g.ch[A][0] >= 0) ? A : B;      // prefer absorbing an internal node
    int other  = (victim == A) ? B : A;
    if (g.ch[victim][0] < 0) throw runtime_error("cannot re-root: both children are leaves");
    g.ch[t][0] = other;
    g.ch[t][1] = g.ch[victim][0];
    g.ch[t][2] = g.ch[victim][1] >= 0 ? g.ch[victim][1] : g.ch[victim][2];
    if (g.ch[t][2] < 0) throw runtime_error("re-root failed");
    g.par[g.ch[t][0]] = t; g.par[g.ch[t][1]] = t; g.par[g.ch[t][2]] = t;
    // mark victim orphaned
    g.ch[victim][0] = g.ch[victim][1] = g.ch[victim][2] = -2; // sentinel: dead
    g.leafid[victim] = -9;
  }
  if (g.ch[t][2] < 0) throw runtime_error("root does not have 3 children");
  g.root = t;
  // traversal (preorder via stack), compute nl, par (par already set), post order
  g.post.clear();
  g.post.reserve(g.m);
  vector<int> stack;
  stack.push_back(t);
  g.par[t] = -1;
  while (!stack.empty()) {
    int v = stack.back(); stack.pop_back();
    g.post.push_back(v);
    for (int k = 0; k < 3; k++) {
      int c = g.ch[v][k];
      if (c >= 0) { g.par[c] = v; stack.push_back(c); }
    }
  }
  reverse(g.post.begin(), g.post.end()); // children before parents
  for (int v : g.post) {
    if (g.ch[v][0] < 0) g.nl[v] = 1;
    else {
      g.nl[v] = 0;
      for (int k = 0; k < 3; k++) if (g.ch[v][k] >= 0) g.nl[v] += g.nl[g.ch[v][k]];
    }
  }
  // internal node list
  g.internalNodes.clear();
  for (int v : g.post) if (g.ch[v][0] >= 0) g.internalNodes.push_back(v);
  // unrooted adjacency
  g.adj.assign(g.m, {-1, -1, -1});
  g.deg.assign(g.m, 0);
  for (int v : g.post) {
    if (v == t) continue;
    int p = g.par[v];
    g.adj[v][g.deg[v]++] = p;
    // find free slot in p
    for (int k = 0; k < 3; k++) if (g.adj[p][k] == -1) { g.adj[p][k] = v; g.deg[p]++; break; }
  }
  g.b.assign(g.m, 0);
}

// w(B|C) for a single gene tree: number of quartets {x,y} in B, {z,w} in C
// resolved as xy|zw by this gene tree.
static ll wScoreGene(const Gene& g, const Mask& B, int bTot) {
  int* b = const_cast<int*>(g.b.data());
  for (int v : g.post) {
    if (g.ch[v][0] < 0) {
      b[v] = testBit(B, g.leafid[v]);
    } else {
      int s = b[g.ch[v][0]];
      if (g.ch[v][1] >= 0) s += b[g.ch[v][1]];
      if (g.ch[v][2] >= 0) s += b[g.ch[v][2]];
      b[v] = s;
    }
  }
  ll s = 0;
  for (int v : g.internalNodes) {
    int n1, n2, n3, b1, b2, b3;
    n1 = g.nl[g.ch[v][0]]; b1 = b[g.ch[v][0]];
    n2 = g.nl[g.ch[v][1]]; b2 = b[g.ch[v][1]];
    if (g.ch[v][2] >= 0) { n3 = g.nl[g.ch[v][2]]; b3 = b[g.ch[v][2]]; }
    else { n3 = n - g.nl[v]; b3 = bTot - b[v]; }
    ll c1 = n1 - b1, c2 = n2 - b2, c3 = n3 - b3;
    s += (ll)b1 * b2 * (c3 * (c3 - 1) / 2) + (ll)c1 * c2 * (b3 * (ll)(b3 - 1) / 2);
    s += (ll)b1 * b3 * (c2 * (c2 - 1) / 2) + (ll)c1 * c3 * (b2 * (ll)(b2 - 1) / 2);
    s += (ll)b2 * b3 * (c1 * (c1 - 1) / 2) + (ll)c2 * c3 * (b1 * (ll)(b1 - 1) / 2);
  }
  return s / 2;
}

// memoized w over all gene trees; key canonicalized so that leaf 0 is on the C side
static unordered_map<Mask, ll, MaskHash> memo;
static ll wScore(Mask B) {
  if (testBit(B, 0)) notMaskN(B, B, n); // canonical: bit0 clear
  auto it = memo.find(B);
  if (it != memo.end()) return it->second;
  if (memo.size() > 26000000u) memo.clear();
  int bTot = popMask(B);
  ll s = 0;
  for (int gi = 0; gi < G; gi++) s += wScoreGene(genes[gi], B, bTot);
  memo.emplace(B, s);
  return s;
}

// ---------------- species tree (unrooted, adjacency) ----------------
struct ST {
  array<int, 3> adj[MAXV];
  int deg[MAXV];
  // derived (valid after buildRooted):
  int par[MAXV];
  Mask mask[MAXV];
  Mask comp[MAXV][3]; // comp[v][k] = leaf set on adj[v][k]'s side of edge (v, adj[v][k])
  int order[MAXV];    // preorder from root
  int norder;
};

static int ROOTID = 250; // any internal node; NJ guarantees it exists when n==250

static void stClear(ST& t) {
  for (int v = 0; v < MAXV; v++) { t.deg[v] = 0; t.adj[v] = {-1, -1, -1}; }
}
static void stLink(ST& t, int a, int b) {
  t.adj[a][t.deg[a]++] = b;
  t.adj[b][t.deg[b]++] = a;
}
static bool stValidate(const ST& t) {
  // all leaves deg1, internal deg3, connected, correct counts
  int leaves = 0, internal = 0;
  for (int v = 0; v < n; v++) { if (t.deg[v] != 1) return false; leaves++; }
  for (int v = n; v < MAXV; v++) {
    if (t.deg[v] == 0) continue;
    if (t.deg[v] != 3) return false;
    internal++;
  }
  if (leaves != n || internal != n - 2) return false;
  // connectivity + each leaf label exactly once is implied by construction
  vector<int> seen(MAXV, 0), stack;
  stack.push_back(ROOTID);
  if (t.deg[ROOTID] == 0) {
    for (int v = n; v < MAXV; v++) if (t.deg[v] == 3) { stack[0] = v; break; }
  }
  int cnt = 0;
  seen[stack[0]] = 1;
  while (!stack.empty()) {
    int v = stack.back(); stack.pop_back(); cnt++;
    for (int k = 0; k < t.deg[v]; k++) {
      int u = t.adj[v][k];
      if (!seen[u]) { seen[u] = 1; stack.push_back(u); }
    }
  }
  return cnt == 2 * n - 2;
}

static void buildRooted(ST& t, int root) {
  t.norder = 0;
  vector<int> stack;
  stack.push_back(root);
  t.par[root] = -1;
  while (!stack.empty()) {
    int v = stack.back(); stack.pop_back();
    t.order[t.norder++] = v;
    for (int k = 0; k < t.deg[v]; k++) {
      int u = t.adj[v][k];
      if (u != t.par[v]) { t.par[u] = v; stack.push_back(u); }
    }
  }
  for (int i = t.norder - 1; i >= 0; i--) {
    int v = t.order[i];
    clearMask(t.mask[v]);
    if (v < n) setBit(t.mask[v], v);
    else {
      for (int k = 0; k < t.deg[v]; k++) {
        int u = t.adj[v][k];
        if (u != t.par[v]) { Mask tmp; orMask(tmp, t.mask[v], t.mask[u]); t.mask[v] = tmp; }
      }
    }
  }
  for (int i = 0; i < t.norder; i++) {
    int v = t.order[i];
    for (int k = 0; k < t.deg[v]; k++) {
      int u = t.adj[v][k];
      if (t.par[u] == v) t.comp[v][k] = t.mask[u];
      else notMaskN(t.comp[v][k], t.mask[v], n); // parent side
    }
  }
}

static ll evalFull(ST& t, int root) {
  buildRooted(t, root);
  ll s = 0;
  for (int i = 0; i < t.norder; i++) {
    int v = t.order[i];
    if (v < n || v == root) continue;
    s += wScore(t.mask[v]);
  }
  return s;
}

// Find an internal node id with deg 3.
static int anyInternal(const ST& t) {
  for (int v = n; v < MAXV; v++) if (t.deg[v] == 3) return v;
  return -1;
}

// ---------------- NNI hill climbing ----------------
struct NniMove { int u, v, ui, vi; ll delta; };

// One NNI round: try all 2 NNIs per internal edge, apply the best improving.
// Returns score delta applied (0 if none).
static ll nniRound(ST& t, int root, ll& curScore) {
  buildRooted(t, root);
  ll bestD = 0;
  NniMove best{};
  best.u = -1;
  for (int i = 0; i < t.norder; i++) {
    int u = t.order[i];
    if (u < n) continue;
    for (int ku = 0; ku < 3; ku++) {
      int v = t.adj[u][ku];
      if (v < n || v < u) continue; // each internal edge once (u<v both internal)
      // side indices
      int i1 = (ku + 1) % 3, i2 = (ku + 2) % 3;
      int kv = -1;
      for (int k = 0; k < 3; k++) if (t.adj[v][k] == u) kv = k;
      int j1 = (kv + 1) % 3, j2 = (kv + 2) % 3;
      ll cur = wScore(t.comp[v][kv]);
      // mode 1: swap subtree at u-side i2 with v-side j1 -> new split = comp[u][i1] | comp[v][j1]
      Mask cand = orMaskF(t.comp[u][i1], t.comp[v][j1]);
      ll d1 = wScore(cand) - cur;
      if (d1 > bestD) { bestD = d1; best = {u, v, i1, j1, d1}; best.ui = i2; best.vi = j1; }
      // mode 2: swap i1 with j1 -> new split = comp[u][i2] | comp[v][j1]
      cand = orMaskF(t.comp[u][i2], t.comp[v][j1]);
      ll d2 = wScore(cand) - cur;
      if (d2 > bestD) { bestD = d2; best = {u, v, i2, j1, d2}; best.ui = i1; best.vi = j1; }
    }
  }
  if (best.u < 0) return 0;
  // apply: move subtree best.ui (at u) to v, and best.vi (at v) to u
  int a = t.adj[best.u][best.ui];
  int b = t.adj[best.v][best.vi];
  t.adj[best.u][best.ui] = b;
  t.adj[best.v][best.vi] = a;
  for (int k = 0; k < 3; k++) { if (t.adj[a][k] == best.u) t.adj[a][k] = best.v; }
  for (int k = 0; k < 3; k++) { if (t.adj[b][k] == best.v) t.adj[b][k] = best.u; }
  curScore += bestD;
  return bestD;
}

static ll climb(ST& t, int root, ll startScore) {
  ll cur = startScore;
  while (true) {
    if (elapsed() > DEADLINE - 2.0) break;
    ll d = nniRound(t, root, cur);
    if (d == 0) break;
  }
  return cur;
}

// ---------------- random SPR ----------------
static void sprRandom(ST& t, mt19937& rng) {
  // pick a random edge (s,u) with s internal
  int s = -1;
  for (int tries = 0; tries < 100; tries++) {
    int c = n + (int)(rng() % (n - 2));
    if (t.deg[c] == 3) { s = c; break; }
  }
  if (s < 0) return;
  int u = t.adj[s][rng() % 3];
  // cut edge (s,u)
  int us = -1, uu = -1;
  for (int k = 0; k < 3; k++) if (t.adj[s][k] == u) us = k;
  for (int k = 0; k < t.deg[u]; k++) if (t.adj[u][k] == s) uu = k;
  if (us < 0 || uu < 0) return;
  // suppress s (deg 3 -> 2): connect s's other two neighbors
  int p = t.adj[s][(us + 1) % 3], q = t.adj[s][(us + 2) % 3];
  for (int k = 0; k < 3; k++) if (t.adj[p][k] == s) t.adj[p][k] = q;
  for (int k = 0; k < 3; k++) if (t.adj[q][k] == s) t.adj[q][k] = p;
  // remove s entirely; u keeps its slot (may become deg 2 if internal, deg 0 if leaf)
  t.deg[s] = 0; t.adj[s] = {-1, -1, -1};
  // u: remove s
  for (int k = uu; k < t.deg[u] - 1; k++) t.adj[u][k] = t.adj[u][k + 1];
  t.adj[u][--t.deg[u]] = -1;
  // moving component = component containing u. Mark it (BFS from u, s is gone).
  static vector<int> inU(MAXV, 0);
  static int stamp = 1;
  stamp++;
  vector<int> stack;
  stack.push_back(u); inU[u] = stamp;
  while (!stack.empty()) {
    int v = stack.back(); stack.pop_back();
    for (int k = 0; k < t.deg[v]; k++) {
      int w2 = t.adj[v][k];
      if (inU[w2] != stamp) { inU[w2] = stamp; stack.push_back(w2); }
    }
  }
  // pick random edge (x,y) in the complement component
  int x = -1, y = -1;
  for (int tries = 0; tries < 2000; tries++) {
    int cx = rng() % (2 * n - 2);
    if (cx == s || inU[cx] == stamp || t.deg[cx] == 0) continue;
    int kk = rng() % t.deg[cx];
    int cy = t.adj[cx][kk];
    if (cy == s || inU[cy] == stamp || t.deg[cy] == 0) continue;
    x = cx; y = cy; break;
  }
  if (x < 0) { // no valid target (shouldn't happen); abort is unsafe -> re-attach anywhere valid via p-q
    x = p; y = q;
    if (inU[x] == stamp || inU[y] == stamp) { // p,q inside U? impossible; but be safe
      return; // leaves tree broken -- caller must not use; guarded below
    }
  }
  // subdivide edge (x,y) with node s, attach u
  for (int k = 0; k < 3; k++) if (t.adj[x][k] == y) t.adj[x][k] = s;
  for (int k = 0; k < 3; k++) if (t.deg[y] > k && t.adj[y][k] == x) t.adj[y][k] = s;
  t.deg[s] = 3; t.adj[s] = {x, y, u};
  t.adj[u][t.deg[u]++] = s;
}

// ---------------- NJ on average internode distances ----------------
static void njTree(const vector<vector<double>>& D, ST& out) {
  stClear(out);
  int nextID = n; // internal node ids from n..
  vector<int> active;
  for (int i = 0; i < n; i++) active.push_back(i);
  vector<double> r(2 * MAXN, 0.0);
  vector<vector<double>> Dm(2 * MAXN, vector<double>(2 * MAXN, 0.0));
  for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) Dm[i][j] = D[i][j];
  while ((int)active.size() > 3) {
    int m = active.size();
    for (int i : active) { r[i] = 0; for (int j : active) r[i] += Dm[i][j]; }
    double bestQ = 1e300; int bi = -1, bj = -1;
    for (int ai = 0; ai < m; ai++) {
      int i = active[ai];
      for (int aj = ai + 1; aj < m; aj++) {
        int j = active[aj];
        double q = (m - 2) * Dm[i][j] - r[i] - r[j];
        if (q < bestQ) { bestQ = q; bi = i; bj = j; }
      }
    }
    int u = nextID++;
    for (int k : active) {
      if (k == bi || k == bj) continue;
      Dm[u][k] = Dm[k][u] = (Dm[bi][k] + Dm[bj][k] - Dm[bi][bj]) / 2.0;
    }
    stLink(out, u, bi);
    stLink(out, u, bj);
    vector<int> na;
    for (int k : active) if (k != bi && k != bj) na.push_back(k);
    na.push_back(u);
    active.swap(na);
  }
  // connect last three through a center node
  int z = nextID++;
  for (int k : active) stLink(out, z, k);
  (void)z;
}

// Convert gene tree topology to an ST (fresh internal ids).
static void geneToST(const Gene& g, ST& out) {
  stClear(out);
  vector<int> remap(g.m, -1);
  int nextID = n;
  for (int v : g.post) {
    if (g.ch[v][0] < 0) { remap[v] = g.leafid[v]; continue; }
    int id;
    if (v == g.root) id = nextID++;
    else id = nextID++;
    remap[v] = id;
  }
  // link parent-child
  for (int v : g.post) {
    if (v == g.root) continue;
    if (g.ch[v][0] == -2) continue; // dead node
    if (g.par[v] >= 0 && g.ch[g.par[v]][0] != -2) stLink(out, remap[g.par[v]], remap[v]);
  }
}

// ---------------- distances ----------------
static void internodeDist(vector<vector<double>>& Dsum) {
  // per gene tree, per leaf: DFS counting internal nodes on path
  vector<int> carry(MAXV, 0), from(MAXV, -1), seen(MAXV, 0);
  int stamp = 0;
  for (int gi = 0; gi < G; gi++) {
    const Gene& g = genes[gi];
    for (int src = 0; src < n; src++) {
      // find the node of leaf src
      stamp++;
      vector<pair<int, int>> stack; // (node, carry)
      // find leaf node id: scan? precompute leafnode map per gene
      // (leafnode[gi] built below in main)
      extern vector<vector<int>> leafNodeOfGene;
      int start = leafNodeOfGene[gi][src];
      seen[start] = stamp; from[start] = -1; carry[start] = 0;
      stack.push_back({start, 0});
      while (!stack.empty()) {
        auto [v, c] = stack.back(); stack.pop_back();
        if (g.leafid[v] >= 0 && g.ch[v][0] < 0) {
          Dsum[src][g.leafid[v]] += c;
        }
        for (int k = 0; k < g.deg[v]; k++) {
          int u = g.adj[v][k];
          if (seen[u] == stamp) continue;
          seen[u] = stamp;
          int nc = c + ((g.ch[u][0] >= 0 && g.leafid[u] < 0) ? 1 : 0);
          carry[u] = nc;
          stack.push_back({u, nc});
        }
      }
    }
  }
}
vector<vector<int>> leafNodeOfGene;

// ---------------- brute-force self test ----------------
#ifdef SELFTEST_OK
#endif
static void selfTest() {
  // validate wScoreGene against brute force on gene 0 with a random split
  mt19937 rng(12345);
  const Gene& g = genes[0];
  Mask B; clearMask(B);
  for (int i = 0; i < n; i++) if (rng() % 2) setBit(B, i);
  int bTot = popMask(B);
  ll fast = wScoreGene(g, B, bTot);
  // all-pairs topological distances (edges) from each leaf
  vector<vector<int>> dist(n, vector<int>(n, 0));
  vector<int> seen(MAXV, 0), dcur(MAXV, 0);
  int stamp = 0;
  extern vector<vector<int>> leafNodeOfGene;
  for (int src = 0; src < n; src++) {
    stamp++;
    vector<int> stack;
    int start = leafNodeOfGene[0][src];
    seen[start] = stamp; dcur[start] = 0; stack.push_back(start);
    while (!stack.empty()) {
      int v = stack.back(); stack.pop_back();
      if (g.leafid[v] >= 0 && g.ch[v][0] < 0) dist[src][g.leafid[v]] = dcur[v];
      for (int k = 0; k < g.deg[v]; k++) {
        int u = g.adj[v][k];
        if (seen[u] == stamp) continue;
        seen[u] = stamp; dcur[u] = dcur[v] + 1; stack.push_back(u);
      }
    }
  }
  ll brute = 0;
  for (int a = 0; a < n; a++) {
    bool aB = testBit(B, a);
    if (!aB) continue; // count each quartet once: {a,b} = the B-side pair
    for (int b2 = a + 1; b2 < n; b2++) {
      if (testBit(B, b2) != aB) continue;
      for (int c = 0; c < n; c++) {
        if (testBit(B, c) == aB) continue;
        for (int d = c + 1; d < n; d++) {
          if (testBit(B, d) == aB) continue;
          // quartet {a,b} in B, {c,d} in C: resolved as ab|cd iff
          // d(a,b)+d(c,d) < min(d(a,c)+d(b,d), d(a,d)+d(b,c))
          int s1 = dist[a][b2] + dist[c][d];
          int s2 = dist[a][c] + dist[b2][d];
          int s3 = dist[a][d] + dist[b2][c];
          if (s1 < s2 && s1 < s3) brute++;
        }
      }
    }
  }
  fprintf(stderr, "SELFTEST gene0: fast=%lld brute=%lld %s\n", fast, brute, fast == brute ? "OK" : "MISMATCH");
  if (fast != brute) exit(9);
}

// ---------------- main ----------------
static string firstLine; // emergency fallback

static void writeNewick(const ST& t, const string& path) {
  int root = anyInternal(t);
  if (root < 0) throw runtime_error("no internal node to write");
  string out;
  out.reserve(4096);
  vector<int> par(MAXV, -1);
  // iterative write with explicit stack
  // We produce ((sub1,sub2,sub3)); using recursion via stack of frames.
  struct Frame { int v; int nextChild; int par; };
  vector<Frame> st;
  out += "(";
  st.push_back({root, 0, -1});
  bool firstAtRoot = true;
  // simpler: recursive lambda
  // (depth <= 500, stack fine)
  out.clear();
  struct W {
    const ST* t; string* o;
    void rec(int v, int p) {
      if (v < n) { (*o) += labels[v]; return; }
      (*o) += '(';
      bool first = true;
      for (int k = 0; k < 3; k++) {
        int u = t->adj[v][k];
        if (u == p) continue;
        if (!first) (*o) += ',';
        first = false;
        rec(u, v);
      }
      (*o) += ')';
    }
  } wr{&t, &out};
  out += "(";
  {
    int cnt = 0;
    for (int k = 0; k < 3; k++) {
      int u = t.adj[root][k];
      if (cnt++) out += ",";
      if (u < n) out += labels[u];
      else wr.rec(u, root);
    }
  }
  out += ");\n";
  FILE* f = fopen(path.c_str(), "w");
  if (!f) throw runtime_error("cannot open output");
  fwrite(out.data(), 1, out.size(), f);
  fclose(f);
  (void)st; (void)par; (void)firstAtRoot;
}

int main(int argc, char** argv) {
  try {
    if (argc >= 2 && string(argv[1]) == "--selftest") {
      // usage: sp --selftest genes.tre
      const char* inp = argc >= 3 ? argv[2] : nullptr;
      if (!inp) { fprintf(stderr, "need input\n"); return 2; }
      FILE* f = fopen(inp, "r");
      char buf[1 << 20];
      string line;
      if (fgets(buf, sizeof(buf), f)) line = buf;
      fclose(f);
      // parse one gene tree
      genes.resize(1);
      Gene& g = genes[0];
      g.ch.assign(2 * MAXN, {-1, -1, -1});
      g.par.assign(2 * MAXN, -1);
      g.nl.assign(2 * MAXN, 0);
      g.leafid.assign(2 * MAXN, -1);
      int pos = 0;
      g.root = parseNode(g, line, pos);
      finishGene(g);
      G = 1;
      n = labels.size();
      clearMask(ALLMASK);
      for (int i = 0; i < n; i++) setBit(ALLMASK, i);
      leafNodeOfGene.assign(1, vector<int>(n, -1));
      for (int v : g.post) if (g.ch[v][0] < 0) leafNodeOfGene[0][g.leafid[v]] = v;
      selfTest();
      return 0;
    }
    if (argc < 3) { fprintf(stderr, "usage: sp INPUT OUTPUT\n"); return 2; }
    const char* inPath = argv[1];
    const char* outPath = argv[2];
    const char* seedEnv = getenv("SEED");
    unsigned seed = seedEnv ? (unsigned)atoi(seedEnv) : 1u;
    if (seed == 0) seed = 1;
    const char* dl = getenv("SP_DEADLINE");
    if (dl) DEADLINE = atof(dl);
    mt19937 rng(seed);

    // ---- read gene trees ----
    FILE* f = fopen(inPath, "r");
    if (!f) throw runtime_error("cannot open input");
    vector<string> lines;
    {
      string cur;
      char buf[1 << 16];
      while (fgets(buf, sizeof(buf), f)) {
        cur += buf;
        if (!cur.empty() && cur.back() == '\n') {
          // trim
          while (!cur.empty() && (cur.back() == '\n' || cur.back() == '\r' || cur.back() == ' ' || cur.back() == '\t')) cur.pop_back();
          if (!cur.empty()) lines.push_back(cur);
          cur.clear();
        }
      }
      fclose(f);
    }
    if (lines.empty()) throw runtime_error("no trees in input");
    firstLine = lines[0];
    G = lines.size();
    genes.resize(G);
    for (int gi = 0; gi < G; gi++) {
      Gene& g = genes[gi];
      g.ch.assign(2 * MAXN, {-1, -1, -1});
      g.par.assign(2 * MAXN, -1);
      g.nl.assign(2 * MAXN, 0);
      g.leafid.assign(2 * MAXN, -1);
      int pos = 0;
      g.root = parseNode(g, lines[gi], pos);
      finishGene(g);
    }
    if ((int)labels.size() < 4) throw runtime_error("too few labels");
    n = labels.size();
    clearMask(ALLMASK);
    for (int i = 0; i < n; i++) setBit(ALLMASK, i);
    // each tree must cover all leaves exactly once -- verify via nl[root]==n
    for (int gi = 0; gi < G; gi++)
      if (genes[gi].nl[genes[gi].root] != n) throw runtime_error("leaf count mismatch");
    leafNodeOfGene.assign(G, vector<int>(n, -1));
    for (int gi = 0; gi < G; gi++)
      for (int v : genes[gi].post)
        if (genes[gi].ch[v][0] < 0) leafNodeOfGene[gi][genes[gi].leafid[v]] = v;
    fprintf(stderr, "parsed %d gene trees, n=%d leaves, t=%.2fs\n", G, n, elapsed());

    // ---- average internode distances + NJ ----
    vector<vector<double>> Dsum(n, vector<double>(n, 0.0));
    internodeDist(Dsum);
    vector<vector<double>> D(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        D[i][j] = Dsum[i][j] / G;
    ST njT;
    njTree(D, njT);
    if (!stValidate(njT)) throw runtime_error("NJ tree invalid");
    ROOTID = anyInternal(njT);
    fprintf(stderr, "NJ built, t=%.2fs\n", elapsed());

    memo.reserve(1 << 20);
    ll njScore = evalFull(njT, ROOTID);
    fprintf(stderr, "NJ quartet score = %lld, t=%.2fs\n", njScore, elapsed());

    ST best = njT;
    ll bestScore = climb(best, ROOTID, njScore);
    fprintf(stderr, "climb(NJ) -> %lld, t=%.2fs\n", bestScore, elapsed());

    // extra starting points: a couple of gene tree topologies
    for (int gi = 0; gi < G && gi < 2; gi++) {
      if (elapsed() > DEADLINE * 0.3) break;
      ST g0;
      geneToST(genes[gi], g0);
      if (!stValidate(g0)) continue;
      int rt = anyInternal(g0);
      ll s0 = evalFull(g0, rt);
      ll s = climb(g0, rt, s0);
      fprintf(stderr, "climb(gene %d) -> %lld, t=%.2fs\n", gi, s, elapsed());
      if (s > bestScore) { best = g0; bestScore = s; ROOTID = rt; }
    }

    // ---- perturbation search ----
    long long climbs = 0, accepted = 0;
    int stag = 0;
    while (elapsed() < DEADLINE - 3.0) {
      ST cur = best;
      int cap = min(4 + stag, 30);
      int k = 1 + (int)(rng() % cap);
      if (stag > 20) k = 20 + (int)(rng() % 30);
      bool ok = true;
      for (int i = 0; i < k; i++) {
        int rt = anyInternal(cur);
        sprRandom(cur, rng);
        if (!stValidate(cur)) { ok = false; break; }
        (void)rt;
      }
      climbs++;
      if (!ok) continue;
      int root = ROOTID;
      if (cur.deg[root] != 3) root = anyInternal(cur);
      ll s0 = evalFull(cur, root);
      ll s = climb(cur, root, s0);
      if (s > bestScore) {
        best = cur; bestScore = s; stag = 0; accepted++;
        fprintf(stderr, "improved to %lld after %lld climbs (k=%d), t=%.2fs\n", bestScore, climbs, k, elapsed());
      } else stag++;
      if (climbs % 100 == 0)
        fprintf(stderr, "climbs=%lld accepted=%lld best=%lld memo=%zu t=%.2fs\n",
                climbs, accepted, bestScore, memo.size(), elapsed());
    }

    // ---- final validation + output ----
    if (!stValidate(best)) {
      fprintf(stderr, "WARNING: best tree invalid, falling back to NJ\n");
      best = njT;
      if (!stValidate(best)) throw runtime_error("fallback invalid");
    }
    int root = anyInternal(best);
    // check leaf labels appear exactly once
    vector<int> cntLeaves(n, 0);
    for (int v = 0; v < n; v++) cntLeaves[v] = 0;
    {
      vector<int> stack;
      vector<int> seen(MAXV, 0);
      stack.push_back(root); seen[root] = 1;
      while (!stack.empty()) {
        int v = stack.back(); stack.pop_back();
        if (v < n) cntLeaves[v]++;
        for (int k = 0; k < best.deg[v]; k++) {
          int u = best.adj[v][k];
          if (!seen[u]) { seen[u] = 1; stack.push_back(u); }
        }
      }
    }
    for (int v = 0; v < n; v++) if (cntLeaves[v] != 1) throw runtime_error("leaf count wrong");

    writeNewick(best, outPath);
    fprintf(stderr, "DONE best=%lld climbs=%lld accepted=%lld t=%.2fs\n",
            bestScore, climbs, accepted, elapsed());
    return 0;
  } catch (const exception& e) {
    fprintf(stderr, "FATAL: %s -- writing fallback\n", e.what());
    // fallback: write the first input line (a valid gene tree over all taxa)
    const char* outPath = (argc >= 3) ? argv[2] : nullptr;
    if (outPath && !firstLine.empty()) {
      FILE* f = fopen(outPath, "w");
      if (f) {
        string s = firstLine;
        if (s.back() != ';') s += ";";
        s += "\n";
        fwrite(s.data(), 1, s.size(), f);
        fclose(f);
        return 0;
      }
    }
    return 1;
  }
}
