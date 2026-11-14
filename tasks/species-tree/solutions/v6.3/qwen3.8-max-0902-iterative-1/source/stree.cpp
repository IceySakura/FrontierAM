// Species tree inference from estimated gene trees.
//
// Method: exact dynamic programming over the bipartitions present in the input
// gene trees, maximising a branch-weighted quartet-support score (the ASTRAL
// objective generalised with per-edge reliability weights).
//
// Weighting matters because the loci are short: a gene-tree edge with (near)
// zero length or no reported support carries almost no phylogenetic signal,
// and letting it vote like a well supported edge only injects noise into the
// quartet counts.
//
// Build:  g++ -O2 -o stree stree.cpp
// Usage:  ./stree genes.tre pred.tre [ref.tre]     (ref.tre = dev diagnostics)
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <unordered_map>

typedef uint64_t u64;

static int NW = 4;                  // 64-bit words per taxon bitset
static double TSTART = 0;
static double DEADLINE = 1e18;

static double now() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
static int VERBOSE = 1;
static void dbg(const char *fmt, ...) {
    if (!VERBOSE) return;
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "[stree %6.2fs] ", now() - TSTART);
    vfprintf(stderr, fmt, ap); fprintf(stderr, "\n");
    va_end(ap);
}

// ---------------------------------------------------------------- bitsets ---
static inline void bs_zero(u64 *a) { for (int i = 0; i < NW; i++) a[i] = 0; }
static inline void bs_set(u64 *a, int i) { a[i >> 6] |= (u64)1 << (i & 63); }
static inline bool bs_test(const u64 *a, int i) { return (a[i >> 6] >> (i & 63)) & 1; }
static inline void bs_or(u64 *d, const u64 *a, const u64 *b) { for (int i = 0; i < NW; i++) d[i] = a[i] | b[i]; }
static inline void bs_andnot(u64 *d, const u64 *a, const u64 *b) { for (int i = 0; i < NW; i++) d[i] = a[i] & ~b[i]; }
static inline bool bs_eq(const u64 *a, const u64 *b) { for (int i = 0; i < NW; i++) if (a[i] != b[i]) return false; return true; }
static inline bool bs_subset(const u64 *a, const u64 *b) { for (int i = 0; i < NW; i++) if (a[i] & ~b[i]) return false; return true; }
static inline int bs_pop(const u64 *a) { int c = 0; for (int i = 0; i < NW; i++) c += __builtin_popcountll(a[i]); return c; }
static inline u64 bs_hash(const u64 *a) {
    u64 h = 1469598103934665603ULL;
    for (int i = 0; i < NW; i++) { h ^= a[i]; h *= 1099511628211ULL; h ^= h >> 29; }
    return h;
}

struct BitMap {                     // open addressing: bitset -> int
    std::vector<u64> keys;
    std::vector<int> vals;
    size_t mask = 0, cnt = 0;
    void init(size_t cap) {
        size_t s = 1; while (s < cap * 2) s <<= 1;
        keys.assign(s * NW, 0); vals.assign(s, -1); mask = s - 1; cnt = 0;
    }
    int find(const u64 *k) const {
        size_t i = bs_hash(k) & mask;
        while (vals[i] >= 0) { if (bs_eq(&keys[i * NW], k)) return vals[i]; i = (i + 1) & mask; }
        return -1;
    }
    int add(const u64 *k, int nid) {   // existing id, or -1 after inserting nid
        size_t i = bs_hash(k) & mask;
        while (vals[i] >= 0) { if (bs_eq(&keys[i * NW], k)) return vals[i]; i = (i + 1) & mask; }
        memcpy(&keys[i * NW], k, NW * sizeof(u64));
        vals[i] = nid; cnt++;
        return -1;
    }
};

// --------------------------------------------------------------- structures --
struct GTree {
    int nn = 0, root = -1;
    std::vector<int> par, nkid;
    std::vector<int> kids, kstart;    // children of u: kids[kstart[u] .. +nkid[u])
    int kid(int u, int i) const { return kids[kstart[u] + i]; }
    std::vector<int> taxon;          // leaf -> taxon id; internal -> -1
    std::vector<double> len, sup;    // of the edge to the parent
    std::vector<double> om;          // reliability weight of the edge to the parent
    std::vector<int> post;           // postorder
    std::vector<u64> ls;             // leafset per node (nn*NW)
};

struct Input {
    std::vector<std::string> labels;
    std::vector<GTree> trees;
    int n = 0;
};

struct Parser {
    const char *p;
    Input &in;
    std::unordered_map<std::string,int> &tmap;
    Parser(const char *p_, Input &i, std::unordered_map<std::string,int> &m) : p(p_), in(i), tmap(m) {}
    int taxon_id(const std::string &s) {
        auto it = tmap.find(s);
        if (it != tmap.end()) return it->second;
        int id = (int)in.labels.size();
        tmap[s] = id; in.labels.push_back(s);
        return id;
    }
    int subtree(GTree &t) {
        int u = t.nn++;
        t.par.push_back(-1); t.nkid.push_back(0);
        t.kstart.push_back(0);
        t.taxon.push_back(-1); t.len.push_back(0); t.sup.push_back(-1);
        if (*p == '(') {
            p++;
            std::vector<int> ch;
            for (;;) {
                int c = subtree(t);
                t.par[c] = u;
                ch.push_back(c);
                if (*p == ',') { p++; continue; }
                break;
            }
            t.kstart[u] = (int)t.kids.size();
            for (int c : ch) t.kids.push_back(c);
            t.nkid[u] = (int)ch.size();
            if (*p == ')') p++;
            std::string lab;
            while (*p && *p != ':' && *p != ',' && *p != ')' && *p != ';' && *p != '(') lab += *p++;
            if (!lab.empty()) {
                char *end; double v = strtod(lab.c_str(), &end);
                if (end != lab.c_str() && *end == 0) t.sup[u] = v;
            }
        } else {
            std::string lab;
            while (*p && *p != ':' && *p != ',' && *p != ')' && *p != ';' && *p != '(') lab += *p++;
            t.taxon[u] = taxon_id(lab);
        }
        if (*p == ':') { p++; char *end; double v = strtod(p, &end); if (end != p) { t.len[u] = v; p = end; } }
        return u;
    }
};

static bool read_file(const char *path, std::string &out) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    out.resize((size_t)sz);
    size_t rd = fread(&out[0], 1, (size_t)sz, f); out.resize(rd);
    fclose(f);
    return true;
}

struct Opts {
    int    use_sup = 1, use_len = 1;
    double sup_lo = 1.0/3.0;      // support <= this is "no signal"
    double sup_pow = 0.5;         // weight *= ((sup-sup_lo)/(1-sup_lo))^sup_pow
    double len_ref = 0.0;         // weight *= min(1,len/len_ref)^len_pow
    double len_pow = 0.0;
    double len_exp = 0.0;         // weight *= 1-exp(-len/len_exp)
    int    kill_zerolen = 1;      // zero-length edge -> weight 0
    int    kill_nosup = 0;        // missing support -> weight 0
    double def_sup = -1;          // value assumed when support is missing (-1 => no signal)
    double tree_pow = 0.0;        // per-gene-tree weight (mean branch length based)
    double tree_ref = 0.0;
    int    sym = 1;               // symmetrise weighted quartet score
};

int main(int argc, char **argv) {
    TSTART = now();
    if (argc < 3) { fprintf(stderr, "usage: stree in.tre out.tre [ref.tre]\n"); return 2; }
    Opts o;
    auto getd = [&](const char *k, double d) { const char *e = getenv(k); return e ? atof(e) : d; };
    auto geti = [&](const char *k, int d)    { const char *e = getenv(k); return e ? atoi(e) : d; };
    o.use_sup = geti("ST_USE_SUP", o.use_sup);
    o.use_len = geti("ST_USE_LEN", o.use_len);
    o.sup_lo = getd("ST_SUP_LO", o.sup_lo);
    o.sup_pow = getd("ST_SUP_POW", o.sup_pow);
    o.len_ref = getd("ST_LEN_REF", o.len_ref);
    o.len_pow = getd("ST_LEN_POW", o.len_pow);
    o.len_exp = getd("ST_LEN_EXP", o.len_exp);
    o.kill_zerolen = geti("ST_KILL_ZEROLEN", o.kill_zerolen);
    o.kill_nosup = geti("ST_KILL_NOSUP", o.kill_nosup);
    o.def_sup = getd("ST_DEF_SUP", o.def_sup);
    o.tree_pow = getd("ST_TREE_POW", o.tree_pow);
    o.tree_ref = getd("ST_TREE_REF", o.tree_ref);
    o.sym = geti("ST_SYM", o.sym);
    VERBOSE = geti("ST_VERBOSE", 1);
    double budget = getd("ST_BUDGET", 100.0);
    DEADLINE = TSTART + budget;

    std::string data;
    if (!read_file(argv[1], data)) { fprintf(stderr, "cannot read %s\n", argv[1]); return 2; }

    Input in;
    std::unordered_map<std::string,int> tmap;
    tmap.reserve(2048);
    {
        const char *s = data.c_str(), *e = s + data.size();
        while (s < e) {
            const char *nl = (const char*)memchr(s, ';', e - s);
            if (!nl) break;
            const char *b = s;
            while (b < nl && (unsigned char)*b <= ' ') b++;
            if (b < nl) {
                GTree t;
                Parser pr(b, in, tmap);
                t.root = pr.subtree(t);
                t.nn = (int)t.par.size();
                in.trees.push_back(std::move(t));
            }
            s = nl + 1;
        }
    }
    int n = (int)in.labels.size();
    in.n = n; NW = (n + 63) / 64;
    int k = (int)in.trees.size();
    dbg("n=%d k=%d", n, k);
    if (n < 4 || k < 1) { fprintf(stderr, "bad input\n"); return 2; }

    int t0ref = 0;
    for (int i = 0; i < n; i++) if (in.labels[i] == "0") { t0ref = i; break; }

    std::vector<u64> full(NW); bs_zero(full.data());
    for (int i = 0; i < n; i++) bs_set(full.data(), i);

    // ------------- postorder + leafsets -------------
    for (auto &t : in.trees) {
        t.post.clear(); t.post.reserve(t.nn);
        std::vector<int> st; st.push_back(t.root);
        std::vector<char> done(t.nn, 0);
        while (!st.empty()) {
            int u = st.back();
            if (done[u]) { st.pop_back(); t.post.push_back(u); continue; }
            done[u] = 1;
            for (int i = 0; i < t.nkid[u]; i++) st.push_back(t.kid(u,i));
        }
        t.ls.assign((size_t)t.nn * NW, 0);
        for (int u : t.post) {
            if (t.taxon[u] >= 0) bs_set(&t.ls[(size_t)u * NW], t.taxon[u]);
            else { u64 *d = &t.ls[(size_t)u*NW]; bs_zero(d);
                   for (int i = 0; i < t.nkid[u]; i++) bs_or(d, d, &t.ls[(size_t)t.kid(u,i)*NW]); }
        }
    }

    // ------------- edge weights -------------
    int nint = 0, nsup = 0, nlen = 0, nzero = 0;
    double sumlen = 0;
    std::vector<double> tmean(k, 0.0);
    for (int gi = 0; gi < k; gi++) {
        auto &t = in.trees[gi];
        double sl = 0; int nl = 0;
        for (int u : t.post) {
            if (u == t.root) continue;
            if (t.nkid[t.root] == 2 && u == t.kid(t.root,1)) continue;
            int pl = bs_pop(&t.ls[(size_t)u * NW]);
            if (pl < 2 || pl > n - 2) continue;             // pendant edge: resolves no quartet
            nint++;
            if (t.sup[u] >= 0) nsup++;
            if (t.len[u] > 1e-9) { nlen++; sl += t.len[u]; nl++; } else nzero++;
        }
        tmean[gi] = nl ? sl / nl : 0.0;
        sumlen += sl;
    }
    double gmean = nlen ? sumlen / nlen : 0.0;
    bool have_sup = nsup > 0.5 * nint;
    bool have_len = nlen > 0.2 * nint;
    dbg("internalEdges=%d withSup=%d len>0=%d zeroLen=%d meanLen=%.5f have_sup=%d have_len=%d",
        nint, nsup, nlen, nzero, gmean, (int)have_sup, (int)have_len);

    for (int gi = 0; gi < k; gi++) {
        auto &t = in.trees[gi];
        t.om.assign(t.nn, 0.0);
        double tw = 1.0;
        if (o.tree_pow > 0 && o.tree_ref > 0) {
            double r = tmean[gi] / o.tree_ref; if (r > 1) r = 1;
            tw = pow(r, o.tree_pow);
        }
        for (int u : t.post) {
            if (u == t.root) continue;
            int pl = bs_pop(&t.ls[(size_t)u * NW]);
            if (pl < 2 || pl > n - 2) continue;
            double w = tw;
            if (o.use_len && have_len) {
                double L = t.len[u];
                if (L <= 1e-9) { if (o.kill_zerolen) w = 0.0; }
                else {
                    if (o.len_ref > 0) { double r = L / o.len_ref; if (r > 1) r = 1; w *= (o.len_pow > 0 ? pow(r, o.len_pow) : r); }
                    if (o.len_exp > 0) w *= 1.0 - exp(-L / o.len_exp);
                }
            }
            if (w > 0 && o.use_sup && have_sup) {
                double s = t.sup[u];
                if (s < 0) {
                    if (o.kill_nosup) w = 0.0;
                    else if (o.def_sup >= 0) s = o.def_sup;
                    else w = (o.sup_pow > 0 ? 0.0 : w);
                }
                if (w > 0 && s >= 0 && o.sup_pow > 0) {
                    double r = (s - o.sup_lo) / (1.0 - o.sup_lo);
                    if (r < 0) r = 0; if (r > 1) r = 1;
                    w *= pow(r, o.sup_pow);
                }
            }
            t.om[u] = w;
        }
        if (t.nkid[t.root] == 2) {   // degree-2 serialization root: both halves are one edge
            int a = t.kid(t.root,0), b = t.kid(t.root,1);
            double m = std::max(t.om[a], t.om[b]);
            t.om[a] = t.om[b] = m;
        }
    }
    double omsum = 0; for (auto &t : in.trees) for (int u : t.post) omsum += t.om[u];
    dbg("total edge weight %.1f / %d internal edges", omsum, nint);

    // ------------- bipartition set S -------------
    BitMap smap; smap.init(1u << 17);
    std::vector<u64> S;
    std::vector<int> Spc;
    std::vector<float> Scnt;
    auto addS = [&](const u64 *b) -> int {
        int id = smap.add(b, (int)Spc.size());
        if (id >= 0) return id;
        S.insert(S.end(), b, b + NW);
        Spc.push_back(bs_pop(b));
        Scnt.push_back(0.f);
        return (int)Spc.size() - 1;
    };
    std::vector<u64> tmp(NW), diff(NW);
    for (int i = 0; i < n; i++) {
        bs_zero(tmp.data()); bs_set(tmp.data(), i); addS(tmp.data());
        bs_andnot(tmp.data(), full.data(), tmp.data()); addS(tmp.data());
    }
    int nsplit_tot = 0;
    int CNTW = geti("ST_CNT_WEIGHT", 1);
    for (int gi = 0; gi < k; gi++) {
        auto &t = in.trees[gi];
        for (int u : t.post) {
            if (u == t.root) continue;
            if (t.nkid[t.root] == 2 && u == t.kid(t.root,1)) continue;
            const u64 *L = &t.ls[(size_t)u * NW];
            int pl = bs_pop(L);
            if (pl < 2 || pl > n - 2) continue;
            nsplit_tot++;
            bs_andnot(tmp.data(), full.data(), L);
            const u64 *side = bs_test(L, t0ref) ? tmp.data() : L;
            int i1 = addS(L), i2 = addS(tmp.data());
            if (CNTW) { float ow = (float)t.om[u]; Scnt[i1] += ow; Scnt[i2] += ow; }
            else if (t.om[u] > 0) { Scnt[i1] += 1.f; Scnt[i2] += 1.f; }   // only informative edges vote
            (void)side;
        }
    }
    int NS = (int)Spc.size();
    dbg("splits=%d distinct=%d |S|=%d", nsplit_tot, nsplit_tot ? (int)smap.cnt : 0, NS);
    if (NS < 2) { fprintf(stderr, "no splits\n"); return 3; }

    // ------------- quartet-score machinery -------------
    std::vector<double> ct(n + 1);
    for (int x = 0; x <= n; x++) ct[x] = 0.5 * (double)x * (double)(x - 1);

    struct Scratch {
        std::vector<int> a, c1;      // per node: |A below|, |A1 below|
        std::vector<int> Rp;         // nodes with a>0 (postorder)
        std::vector<int> R;          // branching nodes of the induced subtree
        std::vector<int> Roff, Rdeg; // component list per relevant node
        std::vector<int> Rcomp;      // child id, or -1 for the "up" component
    };
    std::vector<Scratch> scr(k);
    for (int gi = 0; gi < k; gi++) {
        int nn = in.trees[gi].nn;
        scr[gi].a.assign(nn, 0); scr[gi].c1.assign(nn, 0);
        scr[gi].Rp.reserve(nn); scr[gi].R.reserve(nn);
        scr[gi].Roff.reserve(nn); scr[gi].Rdeg.reserve(nn); scr[gi].Rcomp.reserve(4 * nn);
    }
    std::vector<char> mark(n, 0);

    auto prep = [&](int gi, const u64 *A, int pa) {
        GTree &t = in.trees[gi];
        Scratch &s = scr[gi];
        s.Rp.clear(); s.R.clear(); s.Roff.clear(); s.Rdeg.clear(); s.Rcomp.clear();
        for (int u : t.post) {
            int av;
            if (t.taxon[u] >= 0) av = bs_test(A, t.taxon[u]) ? 1 : 0;
            else { av = 0; for (int i = 0; i < t.nkid[u]; i++) av += s.a[t.kid(u,i)]; }
            s.a[u] = av;
            if (av == 0) { s.c1[u] = 0; continue; }
            s.Rp.push_back(u);
        }
        for (int u : s.Rp) {
            if (t.nkid[u] == 0) continue;
            int start = (int)s.Rcomp.size(), deg = 0;
            for (int i = 0; i < t.nkid[u]; i++) {
                int v = t.kid(u,i);
                if (s.a[v] > 0) { s.Rcomp.push_back(v); deg++; }
            }
            if (u != t.root && s.a[u] < pa) { s.Rcomp.push_back(-1); deg++; }
            if (deg < 3) { s.Rcomp.resize(start); continue; }   // contributes nothing
            s.R.push_back(u); s.Roff.push_back(start); s.Rdeg.push_back(deg);
        }
    };

    double p2c = 0, q2c = 0;         // p*p, q*q of the current query
    int sym = o.sym;
    auto scoreW = [&](int pb, int qc) {
        double sc = 0;
        for (int gi = 0; gi < k; gi++) {
            GTree &t = in.trees[gi];
            Scratch &s = scr[gi];
            for (int u : s.Rp) {
                if (t.taxon[u] >= 0) s.c1[u] = mark[t.taxon[u]];
                else { int v = 0; for (int i = 0; i < t.nkid[u]; i++) v += s.c1[t.kid(u,i)]; s.c1[u] = v; }
            }
            int m = (int)s.R.size();
            for (int j = 0; j < m; j++) {
                int w = s.R[j], off = s.Roff[j], deg = s.Rdeg[j];
                double d = 0, r = 0;
                if (deg == 3) {
                    int c0 = s.Rcomp[off], c1i = s.Rcomp[off+1], c2 = s.Rcomp[off+2];
                    double a0,b0,w0, a1,b1,w1, a2,b2,w2;
                    if (c0 >= 0) { a0 = s.c1[c0]; b0 = s.a[c0] - a0; w0 = t.om[c0]; }
                    else { a0 = pb - s.c1[w]; b0 = (qc + pb - s.a[w]) - a0; w0 = t.om[w]; }
                    if (c1i >= 0) { a1 = s.c1[c1i]; b1 = s.a[c1i] - a1; w1 = t.om[c1i]; }
                    else { a1 = pb - s.c1[w]; b1 = (qc + pb - s.a[w]) - a1; w1 = t.om[w]; }
                    if (c2 >= 0) { a2 = s.c1[c2]; b2 = s.a[c2] - a2; w2 = t.om[c2]; }
                    else { a2 = pb - s.c1[w]; b2 = (qc + pb - s.a[w]) - a2; w2 = t.om[w]; }
                    d = a0*a1*(w2*ct[(int)b2]) + a0*a2*(w1*ct[(int)b1]) + a1*a2*(w0*ct[(int)b0]);
                    if (sym) r = b0*b1*(w2*ct[(int)a2]) + b0*b2*(w1*ct[(int)a1]) + b1*b2*(w0*ct[(int)a0]);
                } else {
                    double Sa2 = 0, Sb2 = 0, Wa = 0, Wb = 0, corrA = 0, corrB = 0;
                    for (int i = 0; i < deg; i++) {
                        int cid = s.Rcomp[off+i];
                        double ai, bi, wi;
                        if (cid >= 0) { ai = s.c1[cid]; bi = s.a[cid] - ai; wi = t.om[cid]; }
                        else { ai = pb - s.c1[w]; bi = (qc + pb - s.a[w]) - ai; wi = t.om[w]; }
                        Sa2 += ai*ai; Sb2 += bi*bi;
                        double Cai = ct[(int)ai], Cbi = ct[(int)bi];
                        Wa += wi*Cai; Wb += wi*Cbi;
                        corrA += wi*Cbi*ai*(pb-ai);
                        corrB += wi*Cai*bi*(qc-bi);
                    }
                    d = 0.5*(p2c - Sa2)*Wb - corrA;
                    if (sym) r = 0.5*(q2c - Sb2)*Wa - corrB;
                }
                sc += sym ? 0.5*(d + r) : d;
            }
        }
        return sc;
    };

    // ------------- per-split weighted quartet support  wv[A] = w(A | X\\A) -------------
    auto setMark = [&](const u64 *A, const u64 *B) {
        for (int w = 0; w < NW; w++) {
            u64 x = A[w];
            while (x) { int b = __builtin_ctzll(x); mark[w*64+b] = (char)((B[w] >> b) & 1); x &= x - 1; }
        }
    };
    std::vector<double> wv(NS, 0.0);
    double NORM = 1.0;             // quartet part: divide by (C(p,2)C(q,2))^NORM
    double QSCALE = 0.0, CNTSCALE = 1.0, CNTPOW = 0.5, QBASE = 1.0/3.0;
    int QMODE = 0;                 // 0: full-complement quartets per split; 1: restricted (ASTRAL-style) per decomposition
    {
        const char *e;
        if ((e = getenv("ST_NORM"))) NORM = atof(e);
        if ((e = getenv("ST_QSCALE"))) QSCALE = atof(e);
        if ((e = getenv("ST_QBASE"))) QBASE = atof(e);
        if ((e = getenv("ST_CNTSCALE"))) CNTSCALE = atof(e);
        if ((e = getenv("ST_CNTPOW"))) CNTPOW = atof(e);
        if ((e = getenv("ST_QMODE"))) QMODE = atoi(e);
    }
    // ORACLE (dev only): score splits by whether they occur in the reference tree
    BitMap oracle;
    bool use_oracle = getenv("ST_ORACLE") && argc > 3;
    double orc_pos = getd("ST_ORACLE_W", 1.0), orc_neg = getd("ST_ORACLE_NEG", 0.0);
    if (use_oracle) {
        std::string td2;
        if (read_file(argv[3], td2)) {
            GTree T; Parser pr(td2.c_str(), in, tmap);
            T.root = pr.subtree(T); T.nn = (int)T.par.size();
            {
                std::vector<int> st2; st2.push_back(T.root);
                std::vector<char> dn(T.nn, 0);
                while (!st2.empty()) { int u = st2.back();
                    if (dn[u]) { st2.pop_back(); T.post.push_back(u); continue; }
                    dn[u] = 1; for (int i = 0; i < T.nkid[u]; i++) st2.push_back(T.kid(u,i)); }
            }
            T.ls.assign((size_t)T.nn * NW, 0);
            for (int u : T.post) {
                if (T.taxon[u] >= 0) bs_set(&T.ls[(size_t)u*NW], T.taxon[u]);
                else { u64 *d = &T.ls[(size_t)u*NW]; bs_zero(d);
                       for (int i = 0; i < T.nkid[u]; i++) bs_or(d, d, &T.ls[(size_t)T.kid(u,i)*NW]); }
            }
            oracle.init(1u << 12);
            int nid = 0;
            for (int u : T.post) {
                const u64 *L = &T.ls[(size_t)u*NW];
                int pl = bs_pop(L);
                if (pl < 2 || pl > n - 2) continue;
                bs_andnot(tmp.data(), full.data(), L);
                const u64 *side = bs_test(L, t0ref) ? tmp.data() : L;
                oracle.add(side, nid++);
            }
            dbg("oracle: %d reference splits", nid);
        }
    }
    double OMTOT = (nint > 0 ? omsum / nint * k : (double)k);   // max attainable quartet fraction scale
    auto qnorm = [&](double raw, int pb, int qc) {
        double den = ct[pb] * ct[qc];
        double qf = (den > 0) ? raw / (den * OMTOT) : 0.0;
        if (QBASE != 0.0) { qf -= QBASE; if (qf < 0) qf = 0; }
        if (NORM != 1.0 && NORM > 0 && qf > 0) qf = pow(qf, NORM);
        return QSCALE * qf;
    };
    double tw = now();
    long long nsc = 0;
    FILE *dumpq = getenv("ST_DUMPQ") ? fopen(getenv("ST_DUMPQ"), "w") : nullptr;
    int wv_timeout = 0;
    {
        std::vector<int> todo;
        todo.reserve(NS);
        for (int ai = 0; ai < NS; ai++) if (Spc[ai] >= 2 && (long long)Spc[ai] * 2 <= n) todo.push_back(ai);
        std::sort(todo.begin(), todo.end(), [&](int a, int b){ return Spc[a] < Spc[b]; });
        for (int gi = 0; gi < k; gi++) prep(gi, full.data(), n);   // A1 u A2 = X for every query
        for (size_t ii = 0; ii < todo.size(); ii++) {
            int ai = todo[ii];
            int pa = Spc[ai];
            const u64 *A = &S[(size_t)ai * NW];
            double w = 0.0;
            if (use_oracle) {
                bs_andnot(tmp.data(), full.data(), A);
                const u64 *side = bs_test(A, t0ref) ? tmp.data() : A;
                w = (oracle.find(side) >= 0) ? orc_pos : orc_neg;
                wv[ai] = w; nsc++;
                bs_andnot(diff.data(), full.data(), A);
                int ci = smap.find(diff.data());
                if (ci >= 0) wv[ci] = w;
                continue;
            }
            if (QSCALE != 0.0) {
                setMark(full.data(), A);
                p2c = (double)pa * pa; q2c = (double)(n - pa) * (double)(n - pa);
                w = scoreW(pa, n - pa);
                double den = ct[pa] * ct[n - pa];
                if (NORM > 0 && den > 0) w /= pow(den, NORM);
                if (QBASE != 0.0 && den > 0) {          // remove the "random agreement" baseline
                    double base = QBASE * (NORM > 0 ? (double)k : den);
                    w -= base; if (w < 0) w = 0;
                }
                w = QSCALE * w;
            }
            if (CNTSCALE != 0.0) {
                double c = Scnt[ai];
                w += CNTSCALE * (CNTPOW == 1.0 ? c : (c > 0 ? pow(c, CNTPOW) : 0.0));
            }
            if (CNTW && QSCALE == 0.0) { /* quartet part unused */ }
            wv[ai] = w; nsc++;
            if (dumpq) {
                double qf = 0;
                setMark(full.data(), A);
                p2c = (double)pa * pa; q2c = (double)(n - pa) * (double)(n - pa);
                double den = ct[pa] * ct[n - pa];
                if (den > 0) qf = scoreW(pa, n - pa) / ((double)k * den);
                fprintf(dumpq, "%d %.4f %.6f %.6f\n", pa, (double)Scnt[ai], wv[ai], qf);
            }
            if (getenv("ST_DEBUG")) dbg("  wv |A|=%d cnt=%.0f score=%.6g", pa, Scnt[ai], wv[ai]);
            bs_andnot(diff.data(), full.data(), A);
            int ci = smap.find(diff.data());
            if (ci >= 0) wv[ci] = wv[ai];
            if (now() > DEADLINE) { wv_timeout = 1; dbg("budget hit during wv at %zu/%zu", ii, todo.size()); break; }
        }
    }
    if (dumpq) fclose(dumpq);
    dbg("wv for %lld subsets in %.2fs (timeout=%d)", nsc, now() - tw, wv_timeout);

    // ------------- DP:  F[A] = wv[A] + max_{B1+B2=A} (F[B1]+F[B2]) -------------
    const double NEG = -1e100;
    std::vector<double> F(NS, NEG);
    std::vector<int> argB(NS, -1), argC(NS, -1);
    std::vector<int> order(NS);
    for (int i = 0; i < NS; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b){ return Spc[a] < Spc[b]; });
    std::vector<std::vector<int>> byPc(n + 1);
    for (int i = 0; i < NS; i++) byPc[Spc[i]].push_back(i);

    double td = now();
    std::vector<int> candB, candC;
    long long nPairs = 0;
    for (int ai : order) {
        int pa = Spc[ai];
        if (pa < 2) { F[ai] = 0.0; continue; }
        const u64 *A = &S[(size_t)ai * NW];
        candB.clear(); candC.clear();
        for (int p = 1; p <= pa / 2; p++) {
            for (int bi : byPc[p]) {
                const u64 *B = &S[(size_t)bi * NW];
                if (!bs_subset(B, A)) continue;
                bs_andnot(diff.data(), A, B);
                int ci = smap.find(diff.data());
                if (ci < 0) continue;
                candB.push_back(bi); candC.push_back(ci);
            }
        }
        nPairs += (long long)candB.size();
        int useQ = (QMODE == 1 && QSCALE != 0.0 && !candB.empty());
        if (useQ) for (int gi = 0; gi < k; gi++) prep(gi, A, pa);
        double best = NEG; int bb = -1, cc = -1;
        for (size_t idx = 0; idx < candB.size(); idx++) {
            int bi = candB[idx], ci = candC[idx];
            if (F[bi] <= NEG/2 || F[ci] <= NEG/2) continue;
            double sc = F[bi] + F[ci];
            if (useQ) {
                int pb = Spc[bi], qc = pa - pb;
                setMark(A, &S[(size_t)bi * NW]);
                p2c = (double)pb * pb; q2c = (double)qc * qc;
                sc += qnorm(scoreW(pb, qc), pb, qc);
            }
            if (sc > best) { best = sc; bb = bi; cc = ci; }
        }
        F[ai] = (best > NEG/2 ? best : 0.0) + wv[ai];
        argB[ai] = bb; argC[ai] = cc;
    }
    double bestRoot = NEG; int rootB = -1, rootC = -1;
    for (int ai = 0; ai < NS; ai++) {
        int pb = Spc[ai];
        if ((long long)pb * 2 > n) continue;
        if (F[ai] <= NEG/2) continue;
        bs_andnot(diff.data(), full.data(), &S[(size_t)ai * NW]);
        int ci = smap.find(diff.data());
        if (ci < 0 || F[ci] <= NEG/2) continue;
        double sc = F[ai] + F[ci] - wv[ai];     // the root edge is shared by both sides
        if (QMODE == 1 && QSCALE != 0.0) {
            setMark(full.data(), &S[(size_t)ai * NW]);
            p2c = (double)pb * pb; q2c = (double)(n - pb) * (double)(n - pb);
            sc += qnorm(scoreW(pb, n - pb), pb, n - pb);
        }
        if (sc > bestRoot) { bestRoot = sc; rootB = ai; rootC = ci; }
    }
    dbg("DP pairs=%lld time=%.2fs score=%.6g", nPairs, now() - td, bestRoot);
    if (rootB < 0) dbg("WARNING: no feasible root split, falling back to a caterpillar");

    // ------------- output -------------
    auto labelOf = [&](int ai) -> const std::string& {
        const u64 *b = &S[(size_t)ai * NW];
        for (int w = 0; w < NW; w++) if (b[w]) return in.labels[w*64 + __builtin_ctzll(b[w])];
        return in.labels[0];
    };
    // arbitrary binary (caterpillar) resolution of a taxon set - safety net for
    // subsets the DP could not decompose (e.g. if a gene tree is a polytomy)
    auto caterpillar = [&](const u64 *bits) {
        std::vector<int> tx;
        for (int w = 0; w < NW; w++) { u64 x = bits[w];
            while (x) { int b = __builtin_ctzll(x); tx.push_back(w*64+b); x &= x-1; } }
        if (tx.empty()) return std::string();
        std::string out = in.labels[tx.back()];
        for (int i = (int)tx.size() - 2; i >= 1; i--) out = "(" + in.labels[tx[i]] + "," + out + ")";
        if (tx.size() >= 2) out = "(" + in.labels[tx[0]] + "," + out + ")";
        else out = in.labels[tx[0]];
        return out;
    };
    struct Frame { int ai; int stage; };
    auto buildIter = [&](int startAi) {
        std::string out;
        std::vector<Frame> st;
        st.push_back({startAi, 0});
        while (!st.empty()) {
            Frame fr = st.back(); st.pop_back();
            int ai = fr.ai;
            if (ai < 0 || ai >= NS) continue;
            if (Spc[ai] == 1) { out += labelOf(ai); continue; }
            if (argB[ai] < 0) { out += caterpillar(&S[(size_t)ai * NW]); continue; }
            if (fr.stage == 0)      { out += "("; st.push_back({ai,1}); st.push_back({argB[ai],0}); }
            else if (fr.stage == 1) { out += ","; st.push_back({ai,2}); st.push_back({argC[ai],0}); }
            else                    out += ")";
        }
        return out;
    };
    std::string res;
    res.reserve(32 * (size_t)n);
    if (rootB >= 0) {
        int b = argB[rootB], c = argC[rootB];
        if (b >= 0) res = "(" + buildIter(b) + "," + buildIter(c) + "," + buildIter(rootC) + ");";
        else        res = "(" + buildIter(rootB) + "," + buildIter(rootC) + ");";
    } else {
        res = "(" + caterpillar(full.data()) + ");";     // never happens for binary input
    }
    FILE *fo = fopen(argv[2], "wb");
    if (!fo) { fprintf(stderr, "cannot write %s\n", argv[2]); return 2; }
    fwrite(res.data(), 1, res.size(), fo);
    fclose(fo);
    dbg("wrote %zu bytes total=%.2fs", res.size(), now() - TSTART);

    // ------------- dev diagnostics against a reference tree -------------
    if (argc > 3) {
        std::string td2;
        if (!read_file(argv[3], td2)) return 0;
        GTree T;
        Parser pr(td2.c_str(), in, tmap);
        T.root = pr.subtree(T);
        T.nn = (int)T.par.size();
        {
            std::vector<int> st2; st2.push_back(T.root);
            std::vector<char> dn(T.nn, 0);
            while (!st2.empty()) {
                int u = st2.back();
                if (dn[u]) { st2.pop_back(); T.post.push_back(u); continue; }
                dn[u] = 1;
                for (int i = 0; i < T.nkid[u]; i++) st2.push_back(T.kid(u,i));
            }
        }
        T.ls.assign((size_t)T.nn * NW, 0);
        for (int u : T.post) {
            if (T.taxon[u] >= 0) bs_set(&T.ls[(size_t)u*NW], T.taxon[u]);
            else { u64 *d = &T.ls[(size_t)u*NW]; bs_zero(d);
                   for (int i = 0; i < T.nkid[u]; i++) bs_or(d, d, &T.ls[(size_t)T.kid(u,i)*NW]); }
        }
        std::vector<u64> ps;
        {
            GTree Pv;
            Parser pr2(res.c_str(), in, tmap);
            Pv.root = pr2.subtree(Pv);
            Pv.nn = (int)Pv.par.size();
            std::vector<int> st2; st2.push_back(Pv.root);
            std::vector<char> dn(Pv.nn, 0);
            while (!st2.empty()) {
                int u = st2.back();
                if (dn[u]) { st2.pop_back(); Pv.post.push_back(u); continue; }
                dn[u] = 1;
                for (int i = 0; i < Pv.nkid[u]; i++) st2.push_back(Pv.kid(u,i));
            }
            Pv.ls.assign((size_t)Pv.nn * NW, 0);
            for (int u : Pv.post) {
                if (Pv.taxon[u] >= 0) bs_set(&Pv.ls[(size_t)u*NW], Pv.taxon[u]);
                else { u64 *d = &Pv.ls[(size_t)u*NW]; bs_zero(d);
                       for (int i = 0; i < Pv.nkid[u]; i++) bs_or(d, d, &Pv.ls[(size_t)Pv.kid(u,i)*NW]); }
            }
            for (int u : Pv.post) {
                if (u == Pv.root && Pv.nkid[u] == 3) continue;
                const u64 *L = &Pv.ls[(size_t)u * NW];
                int pl = bs_pop(L);
                if (pl < 2 || pl > n - 2) continue;
                bs_andnot(tmp.data(), full.data(), L);
                const u64 *can = bs_test(L, t0ref) ? tmp.data() : L;
                ps.insert(ps.end(), can, can + NW);
            }
        }
        double qsc = 0; int tn = 0, shared = 0;
        for (int u : T.post) {
            if (u == T.root && T.nkid[u] == 3) continue;
            const u64 *L = &T.ls[(size_t)u * NW];
            int pl = bs_pop(L);
            if (pl < 2 || pl > n - 2) continue;
            tn++;
            bs_andnot(tmp.data(), full.data(), L);
            const u64 *Bs = bs_test(L, t0ref) ? tmp.data() : L;
            for (size_t j = 0; j < ps.size(); j += NW) if (bs_eq(Bs, &ps[j])) { shared++; break; }
            setMark(full.data(), L);
            for (int gi = 0; gi < k; gi++) prep(gi, full.data(), n);
            p2c = (double)pl*pl; q2c = (double)(n-pl)*(n-pl);
            double sv = scoreW(pl, n - pl);
            qsc += sv;
            dbg("  refsplit |L|=%d score=%.4g", pl, sv);
        }
        dbg("REF splits=%d pred=%zu shared=%d RF=%.4f refScore=%.6g ourScore=%.6g",
            tn, ps.size()/NW, shared, (double)(tn-shared)/(double)(n-3), qsc, bestRoot);
    }
    return 0;
}
