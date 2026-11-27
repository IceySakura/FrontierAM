#pragma once

#include "tree.hpp"
#include <cstdint>

struct Mask {
    std::array<uint64_t, 4> words{};
    bool operator==(const Mask &a) const { return words == a.words; }
    bool operator!=(const Mask &a) const { return words != a.words; }
    bool operator<(const Mask &a) const { return words < a.words; }
    void put(int i) { words[i >> 6] |= uint64_t(1) << (i & 63); }
    bool has(int i) const { return (words[i >> 6] >> (i & 63)) & 1u; }
    int size() const {
        int s = 0;
        for (auto w : words) s += __builtin_popcountll(w);
        return s;
    }
    template<class F> void each(F action) const {
        for (int j = 0; j < 4; ++j) {
            for (uint64_t w = words[j]; w; w &= w - 1)
                action(j*64 + __builtin_ctzll(w));
        }
    }
    friend Mask operator|(const Mask &a, const Mask &b) {
        return {{a.words[0]|b.words[0], a.words[1]|b.words[1],
                 a.words[2]|b.words[2], a.words[3]|b.words[3]}};
    }
    friend Mask operator&(const Mask &a, const Mask &b) {
        return {{a.words[0]&b.words[0], a.words[1]&b.words[1],
                 a.words[2]&b.words[2], a.words[3]&b.words[3]}};
    }
    friend Mask operator^(const Mask &a, const Mask &b) {
        return {{a.words[0]^b.words[0], a.words[1]^b.words[1],
                 a.words[2]^b.words[2], a.words[3]^b.words[3]}};
    }
    bool subset_of(const Mask &a) const { return (*this & a) == *this; }
    bool empty() const { return !(words[0] | words[1] | words[2] | words[3]); }
};

inline uint64_t hash_mix(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

struct MaskHash {
    size_t operator()(const Mask &a) const {
        return hash_mix(a.words[0] + 0x9e3779b97f4a7c15ull * a.words[1]) ^
               hash_mix(a.words[2] + 0x7b4c371e92752a61ull * a.words[3]);
    }
};

using Quad = std::array<Mask, 4>;
using Votes = std::array<float, 3>;

struct QuadHash {
    size_t operator()(const Quad &q) const {
        MaskHash hash;
        return hash(q[0]) + 1248859*hash(q[1]) + 17474797*hash(q[2]) + 109626029*hash(q[3]);
    }
};

inline Mask walk_mask(const Tree &t, int u, int p) {
    Mask a;
    if (u < t.n) a.put(u);
    else for (const auto &e : t.v[u].edges)
        if (e.to != p) a = a | walk_mask(t, e.to, u);
    return a;
}

struct MaskTree {
    Mask all;
    std::vector<Mask> down;
    std::vector<int> parent;
    std::vector<std::pair<int,int>> internal;

    explicit MaskTree(const Tree &t) : down(t.used), parent(t.used, -1) {
        for (int i = 0; i < t.n; ++i) all.put(i);
        down[0] = all;
        dfs(t, t.v[0].edges[0].to, 0);
    }
    Mask side(int v, int u) const {
        return parent[u] == v ? down[u] : (all ^ down[v]);
    }
private:
    Mask dfs(const Tree &t, int u, int p) {
        parent[u] = p;
        Mask a;
        if (u < t.n) a.put(u);
        else {
            if (p >= t.n) internal.emplace_back(p, u);
            for (const auto &e : t.v[u].edges)
                if (e.to != p) a = a | dfs(t, e.to, u);
        }
        return down[u] = a;
    }
};

inline std::array<int,2> other_neighbors(const Tree &t, int u, int v) {
    std::array<int,2> x;
    int k = 0;
    for (const auto &e : t.v[u].edges) if (e.to != v) x[k++] = e.to;
    return x;
}

inline Quad edge_groups(const Tree &t, int u, int v) {
    auto a = other_neighbors(t, u, v), b = other_neighbors(t, v, u);
    return {walk_mask(t,a[0],u), walk_mask(t,a[1],u),
            walk_mask(t,b[0],v), walk_mask(t,b[1],v)};
}

// All keys use the side that excludes the outgroup.
inline std::vector<Mask> internal_splits(const Tree &t) {
    MaskTree r(t);
    std::vector<Mask> ans;
    ans.reserve(r.internal.size());
    for (auto [p,u] : r.internal) ans.push_back(r.down[u]);
    return ans;
}
