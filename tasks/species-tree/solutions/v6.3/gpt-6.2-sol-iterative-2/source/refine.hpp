#pragma once

#include "quartet.hpp"
#include <functional>
#include <tuple>

struct BlockChoice {
    int degree;
    struct Weight {
        int a, b, c, d;
        Votes value;
    };
    std::vector<Weight> weights;
    std::vector<double> best;
    std::vector<int> choice;
    explicit BlockChoice(int k) : degree(k), best(1 << (k-1)), choice(1 << (k-1)) {}

    void optimize() {
        int m = 1 << (degree-1);
        for (int s = 3; s < m; ++s) {
            int low = s & -s;
            if (low == s) continue;
            int c = ((m << 1)-1) ^ s;
            double score = -std::numeric_limits<double>::infinity();
            int pick = 0;
            for (int a = (s-1)&s; a != 0; a = (a-1)&s) {
                if (!(a & low)) continue;
                int b = s ^ a;
                double v = best[a]+best[b];
                for (auto q : weights) {
                    // A quartet has a resolved pairing here iff exactly two of its
                    // leaves enter one arc and one enters each of the other arcs.
                    int x0 = q.a & a, x1 = q.b & a, x2 = q.c & a, x3 = q.d & a;
                    int y0 = q.a & b, y1 = q.b & b, y2 = q.c & b, y3 = q.d & b;
                    int z0 = q.a & c, z1 = q.b & c, z2 = q.c & c, z3 = q.d & c;
                    int ax = !!x0 + !!x1 + !!x2 + !!x3;
                    int bx = !!y0 + !!y1 + !!y2 + !!y3;
                    int cx = 4 - ax - bx;
                    if (ax == 0 || bx == 0 || cx == 0) continue;
                    if (ax != 2) {
                        if (bx == 2) { x0 = y0; x1 = y1; x2 = y2; x3 = y3; }
                        else { x0 = z0; x1 = z1; x2 = z2; x3 = z3; }
                    }
                    int split = (bool(x0) == bool(x1)) ? 0 : ((bool(x0) == bool(x2)) ? 1 : 2);
                    v += q.value[split];
                }
                if (v > score) { score = v; pick = a; }
            }
            best[s] = score;
            choice[s] = pick;
        }
    }
};

template<class Scorer>
inline void resolve_block(Tree &t, const std::vector<int> &inner, Scorer &oracle,
                          float size_normalization = 0.75f, float score_normalization = 0.25f) {
    if (inner.size() < 2) return;
    int k = inner.size()+2;
    std::vector<bool> is_inner(t.used,false);
    for (int u : inner) is_inner[u] = true;
    std::vector<int> leaf;
    std::vector<Mask> groups;
    for (int u : inner)
        for (Arc e : t.v[u].edges)
            if (!is_inner[e.to]) {
                leaf.push_back(e.to);
                groups.push_back(walk_mask(t,e.to,u));
            }
    if (int(leaf.size()) != k) throw std::logic_error("disconnected block");
    BlockChoice dp(k);
    std::vector<float> sz(k);
    for (int a = 0; a < k; ++a) sz[a] = std::pow(groups[a].size(),size_normalization);
    for (int a = 0; a < k; ++a) for (int b = a+1; b < k; ++b)
        for (int c = b+1; c < k; ++c) for (int d = c+1; d < k; ++d) {
            Votes v = oracle({groups[a],groups[b],groups[c],groups[d]});
            float mean = (v[0]+v[1]+v[2])/3;
            float den = sz[a]*sz[b]*sz[c]*sz[d];
            den *= std::pow(0.1f+(3*mean)/(groups[a].size()*groups[b].size()*
                          groups[c].size()*groups[d].size()),score_normalization);
            for (float &x : v) x = (x-mean)/den;
            dp.weights.push_back({1<<a,1<<b,1<<c,1<<d,v});
        }
    dp.optimize();
    int s = (1 << (k-1))-1;
    if (dp.best[s] <= 0) return;
    for (int u : inner)
        while (t.v[u].degree) t.unlink(u,t.v[u].edges[0].to);
    size_t next = 0;
    std::function<int(int)> attach = [&](int a) {
        if (!(a & (a-1))) return leaf[__builtin_ctz(unsigned(a))];
        int b = dp.choice[a], u = inner[next++];
        int l = attach(b), r = attach(a^b);
        t.link(u,l);
        t.link(u,r);
        return u;
    };
    t.link(attach(s),leaf.back());
}

struct Components {
    std::vector<int> parent, sz;
    explicit Components(int n) : parent(n), sz(n,1) { std::iota(parent.begin(),parent.end(),0); }
    int root(int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }
    void join(int x, int y, int limit) {
        x = root(x); y = root(y);
        if (x == y || sz[x]+sz[y] > limit) return;
        if (sz[x] < sz[y]) std::swap(x,y);
        parent[y] = x;
        sz[x] += sz[y];
    }
};

template<class Scorer, class Credibility>
inline void refine_blocks(Tree &t, Scorer &oracle, Credibility confidence, float threshold,
                          int max_ports, float size_normalization, float score_normalization) {
    MaskTree mt(t);
    std::vector<std::tuple<float,int,int>> ranked;
    for (auto [u,v] : mt.internal) {
        Quad q = edge_groups(t,u,v);
        float z = confidence(q,mt.down[v]);
        if (z < threshold) ranked.emplace_back(z,u,v);
    }
    std::sort(ranked.begin(),ranked.end());
    Components cc(t.used);
    for (auto [w,u,v] : ranked) cc.join(u,v,max_ports-2);
    std::vector<std::vector<int>> by_root(t.used);
    for (int u = t.n; u < t.used; ++u) by_root[cc.root(u)].push_back(u);
    for (auto &c : by_root)
        resolve_block(t,c,oracle,size_normalization,score_normalization);
}

template<class Scorer>
inline bool relocate(Tree &t, int point, int sub, int radius, Scorer &oracle,
                     float size_normalization = 0.0f) {
    const Mask moved = walk_mask(t,sub,point);
    auto children = other_neighbors(t,point,sub);
    int a=children[0], b=children[1], x=a, y=b;
    t.unlink(point,a);
    t.unlink(point,b);
    t.link(a,b);
    Mask rest;
    for (int i=0;i<t.n;++i) if (!moved.has(i)) rest.put(i);
    double best = 0;
    std::function<void(int,int,int,double,Mask)> explore =
        [&](int u, int p, int dist, double gain, Mask back) {
        if (u < t.n || dist == radius) return;
        auto nn=other_neighbors(t,u,p);
        Mask one=walk_mask(t,nn[0],u), two=rest ^ back ^ one;
        Quad groups{moved,back,one,two};
        auto v=oracle(groups);
        double den=1;
        if (size_normalization>0)
            den=std::pow(double(back.size()*one.size()*two.size()),size_normalization);
        for (int j=0;j<2;++j) {
            double s=gain+(v[j+1]-v[0])/den;
            if (s>best) { x=u; y=nn[j]; best=s; }
            explore(nn[j],u,dist+1,s,rest^(j==0?one:two));
        }
    };
    Mask am=walk_mask(t,a,b);
    explore(a,b,0,0,rest^am);
    explore(b,a,0,0,am);
    t.unlink(x,y);
    t.link(point,x);
    t.link(point,y);
    return best>0;
}

template<class Scorer, class Credibility>
inline int limited_spr(Tree &t, Scorer &oracle, Credibility confidence,
                       int picks = 48, int radius = 3, float threshold = 2.4f,
                       float size_normalization = 0.0f) {
    MaskTree mt(t);
    std::vector<std::tuple<float,int,int>> ranked;
    for (auto [u,v] : mt.internal) {
        Quad q=edge_groups(t,u,v);
        float z=confidence(q,mt.down[v]);
        if (z>=threshold) continue;
        for (int a:other_neighbors(t,u,v)) ranked.emplace_back(z,u,a);
        for (int a:other_neighbors(t,v,u)) ranked.emplace_back(z,v,a);
    }
    std::sort(ranked.begin(),ranked.end());
    std::vector<bool> processed(t.used*t.used,false);
    int count=0, steps=0;
    for (auto [z,p,s] : ranked) {
        int key=p*t.used+s;
        if (processed[key]) continue;
        processed[key]=true;
        bool adj=false;
        for (auto x:t.v[p].edges) adj|=x.to==s;
        if (!adj) continue;
        count+=relocate(t,p,s,radius,oracle,size_normalization);
        if (++steps>=picks) break;
    }
    return count;
}

struct SplitPool {
    struct Clade {
        Mask mask;
        int size;
        int taxon = -1;
        float weight = 0;
        std::vector<std::pair<int,int>> options;
    };
    std::vector<Clade> clades;
    std::unordered_map<Mask,int,MaskHash> index;
    int root = 0;

    int insert(Mask a, int taxon = -1) {
        auto [it, added] = index.try_emplace(a,clades.size());
        if (added) clades.push_back({a,a.size(),taxon,0,{}});
        return it->second;
    }

    void record(const Tree &t, float mass) {
        root = visit(t,t.v[0].edges[0].to,0,mass);
    }

    float weight(Mask s) const {
        auto i = index.find(s);
        return i == index.end() ? 0 : clades[i->second].weight;
    }

    Tree build(int n) const {
        std::vector<int> order(clades.size());
        std::iota(order.begin(),order.end(),0);
        std::stable_sort(order.begin(),order.end(),
                         [&](int a, int b) { return clades[a].size < clades[b].size; });
        std::vector<double> best(order.size());
        std::vector<std::pair<int,int>> choice(order.size(),{-1,-1});
        for (int k : order) {
            const auto &c = clades[k];
            if (c.size == 1) continue;
            double s = -std::numeric_limits<double>::infinity();
            for (auto [a,b] : c.options) {
                double x = best[a]+best[b];
                if (x > s) { choice[k] = {a,b}; s = x; }
            }
            best[k] = s + c.weight;
        }
        Tree t(n);
        std::function<int(int)> make = [&](int i) {
            if (clades[i].size == 1) return clades[i].taxon;
            auto [a,b] = choice[i];
            if (a < 0) throw std::logic_error("clade has no split");
            int u = t.node();
            int l = make(a), r = make(b);
            t.link(u,l);
            t.link(u,r);
            return u;
        };
        t.link(0,make(root));
        return t;
    }
private:
    int visit(const Tree &t, int v, int p, float mass) {
        if (v < t.n) {
            Mask a;
            a.put(v);
            return insert(a,v);
        }
        auto child = other_neighbors(t,v,p);
        int a = visit(t,child[0],v,mass), b = visit(t,child[1],v,mass);
        if (a > b) std::swap(a,b);
        int k = insert(clades[a].mask | clades[b].mask);
        auto &c = clades[k];
        c.weight += mass;
        if (std::find(c.options.begin(),c.options.end(),std::make_pair(a,b)) == c.options.end())
            c.options.emplace_back(a,b);
        return k;
    }
};

template<class RNG>
inline std::vector<float> bootstrap_weights(int n, RNG &r, float sd) {
    float alpha = 1/(sd*sd);
    std::gamma_distribution<float> draw(alpha,1/alpha);
    std::vector<float> x(n);
    for (float &w : x) w = draw(r);
    return x;
}
