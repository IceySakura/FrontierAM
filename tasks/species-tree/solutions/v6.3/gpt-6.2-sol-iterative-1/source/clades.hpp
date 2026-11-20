#pragma once
#include "quartet.hpp"
#include <limits>
#include <unordered_set>

namespace st {

using CladeSet = std::unordered_set<Mask, MaskHash>;

inline void add_clades(CladeSet& dest, const Tree& t, int out) {
    t.clades(t.v[out].edges[0].to, out, [&](const Mask& m, Edge) { dest.insert(m); });
}

inline void add_alternatives(CladeSet& dest, const Tree& t, int out, Mask all) {
    std::vector<Mask> rooted(t.v.size());
    t.edge_masks(t.v[out].edges[0].to, out, rooted);
    int root = t.v[out].edges[0].to;
    auto walk = [&](auto& self, int u, int p) -> void {
        if (u < t.n) return;
        std::array<int,2> child{};
        int k = 0;
        for (int j = 0; j < 3; ++j) {
            int c = t.v[u].edges[j].to;
            if (c != p) child[k++] = c;
        }
        for (int side = 0; side < 2; ++side) {
            int c = child[side];
            if (c < t.n) continue;
            Mask other = rooted[child[1-side]];
            for (int j = 0; j < 3; ++j) {
                int g = t.v[c].edges[j].to;
                if (g != u) dest.insert(rooted[g] | other);
            }
            self(self,c,u);
        }
    };
    walk(walk,root,out);
    (void)all;
}

class CladeProblem {
public:
    struct Part {
        int a, b;
        double trip = std::numeric_limits<double>::quiet_NaN();
    };
    struct Clade {
        Mask mask;
        int size;
        std::vector<Part> parts;
    };
    std::vector<Clade> clades;
    const Input& data;
    size_t part_count{};

    CladeProblem(const Input& data, const CladeSet& source, size_t part_limit = 400000)
        : data(data) {
        Mask ingroup = data.all ^ Mask::leaf(data.out);
        clades.reserve(source.size()+data.n());
        for (int i = 0; i < data.n(); ++i)
            if (i != data.out) clades.push_back({Mask::leaf(i),1,{}});
        for (const Mask& m : source) {
            int n = m.size();
            if (n > 1 && n < data.n()-1 && ingroup.contains(m))
                clades.push_back({m,n,{}});
        }
        clades.push_back({ingroup,data.n()-1,{}});
        std::sort(clades.begin(),clades.end(),[](const Clade& a,const Clade& b) {
            return a.size != b.size ? a.size < b.size : a.mask < b.mask;
        });
        std::unordered_map<Mask,int,MaskHash> ids;
        ids.reserve(clades.size());
        for (size_t i = 0; i < clades.size(); ++i) ids[clades[i].mask] = int(i);
        std::vector<bool> reachable(clades.size());
        for (size_t i = 0; i < clades.size(); ++i) {
            auto& c = clades[i];
            if (c.size == 1) {
                reachable[i] = true;
                continue;
            }
            for (size_t j = 0; j < i; ++j) {
                if (clades[j].size > c.size/2) break;
                if (!reachable[j]) continue;
                const Mask& a = clades[j].mask;
                if (!c.mask.contains(a)) continue;
                auto it = ids.find(c.mask ^ a);
                if (it != ids.end() && int(j) < it->second && reachable[it->second]) {
                    c.parts.push_back({int(j),it->second});
                }
            }
            reachable[i] = !c.parts.empty();
            part_count += c.parts.size();
            if (part_count > part_limit) break;
        }
    }

    bool complete() const { return !clades.back().parts.empty(); }

    template<class Prior>
    Tree solve(QuartetScorer* scorer, const Prior& prior, double trip_factor = 1) {
        if (!complete()) throw std::runtime_error("candidate clades do not cover a complete tree");
        std::vector<double> best(clades.size(),-1e300);
        std::vector<std::pair<int,int>> trace(clades.size(),{-1,-1});
        for (size_t i = 0; i < clades.size(); ++i) {
            Clade& c = clades[i];
            if (c.size == 1) {
                best[i] = 0;
                continue;
            }
            double top = -1e300;
            Mask out = data.all ^ c.mask;
            for (Part& p : c.parts) {
                double score = best[p.a] + best[p.b];
                if (scorer && trip_factor != 0) {
                    if (!std::isfinite(p.trip))
                        p.trip = scorer->trip(clades[p.a].mask,clades[p.b].mask,out);
                    score += trip_factor*p.trip;
                }
                if (score > top) {
                    top = score;
                    trace[i] = {p.a,p.b};
                }
            }
            best[i] = top + prior(c.mask);
        }
        Tree t(data.n());
        auto visit = [&](auto& self, int i) -> int {
            if (clades[i].size == 1) return clades[i].mask.first();
            int u = t.add();
            t.link(u,self(self,trace[i].first));
            t.link(u,self(self,trace[i].second));
            return u;
        };
        int r = visit(visit,int(clades.size())-1);
        t.link(r,data.out);
        return t;
    }
};

}  // namespace st
