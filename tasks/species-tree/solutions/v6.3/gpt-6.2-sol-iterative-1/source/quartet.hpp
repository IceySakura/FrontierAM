#pragma once
#include "evidence.hpp"

namespace st {

struct QuartetConfig {
    double support_power = 1.0;
    double length_abs = .018;
    double length_rel = .05;
    double length_power = 1.0;
    double locus_power = .65;
    double flank_abs = 0.;
    double flank_rel = 0.;
    double flank_topo = 0.;
    double tip_only = 0.;
    double intersection = 0.;
};

using Vec4f [[gnu::vector_size(16)]] = float;

struct QuartetKey {
    std::array<Mask, 4> group;
    bool operator==(const QuartetKey& other) const { return group == other.group; }
};
struct QuartetKeyHash {
    size_t operator()(const QuartetKey& x) const {
        MaskHash h;
        size_t a = h(x.group[0]);
        size_t b = h(x.group[1]);
        size_t c = h(x.group[2]);
        size_t d = h(x.group[3]);
        return a ^ (b * 0x9e3779b97f4a7c15ULL) ^ (c * 0xd6e8feb86659fd93ULL) ^
               (d * 0xbf58476d1ce4e5b9ULL);
    }
};

class QuartetScorer {
    struct Entry {
        uint16_t l, r;
        float w, flank;
    };
    const int n, out;
    const bool use_flanks;
    std::vector<std::vector<Entry>> forest;
    std::unordered_map<QuartetKey, std::array<double, 3>, QuartetKeyHash> cache;
    std::unordered_map<QuartetKey, std::array<double, 3>, QuartetKeyHash> coherent_cache;
    std::unordered_map<QuartetKey, std::vector<std::array<float,3>>, QuartetKeyHash> locus_cache;
    std::vector<float> locus_weight;
    std::vector<double> resample_weight;
    double weight_sum = 0;
    double resample_sum = 0;

    static inline Vec4f splat(float a) { return Vec4f{a,a,a,a}; }

    static inline QuartetKey canonical(const std::array<Mask,4>& groups,std::array<int,4>& inverse) {
        std::array<int,4> p{0,1,2,3}, sizes;
        for (int i = 0; i < 4; ++i) sizes[i] = groups[i].size();
        std::sort(p.begin(),p.end(),[&](int a,int b) {
            return sizes[a] != sizes[b] ? sizes[a] < sizes[b] : groups[a] < groups[b];
        });
        QuartetKey key;
        for (int i = 0; i < 4; ++i) {
            key.group[i] = groups[p[i]];
            inverse[p[i]] = i;
        }
        return key;
    }
    static inline std::array<double,3> permute(const std::array<double,3>& x, const std::array<int,4>& inv) {
        return {x[(inv[0]^inv[1])-1],x[(inv[0]^inv[2])-1],x[(inv[0]^inv[3])-1]};
    }

    std::array<double,3> evaluate_coherent(const std::array<Mask,4>& group) const {
        uint32_t count[2*capacity]{};
        std::array<float,4> inv;
        for (int i = 0; i < 4; ++i) {
            uint32_t color = 1U << (8*i);
            group[i].each([&](int j) { count[j] = color; });
            inv[i] = 1.f/group[i].size();
        }
        std::array<double,3> result{};
        for (size_t g = 0; g < forest.size(); ++g) {
            const auto& t = forest[g];
            float m0 = 0., m1 = 0., m2 = 0.;
            for (int i = n; i < 2*n-2; ++i) {
                auto [l,r,w,f] = t[i];
                uint32_t x = count[i] = count[l]+count[r];
                float a = (x&255)*inv[0], b = ((x>>8)&255)*inv[1],
                      c = ((x>>16)&255)*inv[2], d = (x>>24)*inv[3];
                float A=1-a,B=1-b,C=1-c,D=1-d;
                float x0 = a*b*C*D + A*B*c*d;
                float x1 = a*c*B*D + A*C*b*d;
                float x2 = a*d*B*C + A*D*b*c;
                m0 = std::max(m0,w*x0*x0);
                m1 = std::max(m1,w*x1*x1);
                m2 = std::max(m2,w*x2*x2);
            }
            result[0] += double(m0)*locus_weight[g];
            result[1] += double(m1)*locus_weight[g];
            result[2] += double(m2)*locus_weight[g];
        }
        for (double& a : result) a /= weight_sum;
        return result;
    }

    template<bool Trip> static inline Vec4f pairs(uint32_t x) {
        int a = x & 255, b = (x >> 8) & 255, c = (x >> 16) & 255;
        if constexpr (Trip) {
            return Vec4f{float(a*(a-1)/2), float(b*(b-1)/2), float(c*(c-1)/2), 0};
        } else {
            int d = x >> 24;
            return Vec4f{float(a*b), float(a*c), float(a*d), 0};
        }
    }

    template<bool Trip> static inline Vec4f cross(uint32_t x, uint32_t y) {
        int xb = (x >> 8) & 255, xc = (x >> 16) & 255;
        int yb = (y >> 8) & 255, yc = (y >> 16) & 255;
        if constexpr (Trip) {
            int xa = x & 255, ya = y & 255;
            return Vec4f{float(xb*yc+xc*yb), float(xa*yc+xc*ya), float(xa*yb+xb*ya), 0};
        } else {
            int xd = x >> 24, yd = y >> 24;
            return Vec4f{float(xc*yd+xd*yc), float(xb*yd+xd*yb), float(xb*yc+xc*yb), 0};
        }
    }

    template<bool Trip> static inline Vec4f paired(Vec4f x, Vec4f y) {
        if constexpr (Trip) {
            return x*y;
        } else {
            return splat(x[0])*Vec4f{y[1],y[2],y[3],0} + splat(y[0])*Vec4f{x[1],x[2],x[3],0};
        }
    }

    template<bool Trip> static inline Vec4f crossf(Vec4f x, Vec4f y) {
        if constexpr (Trip) {
            return Vec4f{x[1],x[0],x[0],0}*Vec4f{y[2],y[2],y[1],0} +
                   Vec4f{y[1],y[0],y[0],0}*Vec4f{x[2],x[2],x[1],0};
        } else {
            return Vec4f{x[2],x[1],x[1],0}*Vec4f{y[3],y[3],y[2],0} +
                   Vec4f{y[2],y[1],y[1],0}*Vec4f{x[3],x[3],x[2],0};
        }
    }

    // The single-leaf messages accumulate the attenuation on pendant paths.
    // Pair messages separate this from the weight on the quartet's internal path.
    template<bool Trip> std::array<double,3> evaluate_flank(const Mask* group,
                                                const double* leaf_weight = nullptr,
                                                std::array<float,3>* per_gene = nullptr) const {
        constexpr int nc = Trip ? 3 : 4;
        int end = 2*n-2;
        Vec4f color[capacity]{}, sup[2*capacity], sdown[2*capacity], pup[2*capacity],
              pdown[2*capacity], qup[2*capacity], qdown[2*capacity];
        for (int i = 0; i < nc; ++i) {
            Vec4f e{};
            e[i] = 1;
            group[i].each([&](int j) { color[j] = leaf_weight ? e*splat(float(leaf_weight[j])) : e; });
        }
        for (int i = 0; i < n; ++i) pup[i] = qup[i] = Vec4f{};
        pdown[end-1] = qdown[end-1] = Vec4f{};
        std::array<double,3> result{};
        for (size_t g = 0; g < forest.size(); ++g) {
            const auto& t = forest[g];
            for (int i = 0; i < n; ++i) sup[i] = color[i]*splat(t[i].flank);
            for (int i = n; i < end; ++i) {
                auto [l,r,w,e] = t[i];
                Vec4f L = sup[l], R = sup[r];
                sup[i] = splat(e)*(L+R);
                pup[i] = pup[l]+pup[r]+paired<Trip>(L,R);
                qup[i] = splat(w)*pup[i]+splat(1-w)*(qup[l]+qup[r]);
            }
            sdown[end-1] = sup[out];
            Vec4f a{}, b{}, c{};
            for (int i = end-1; i >= n; --i) {
                auto [l,r,w,e] = t[i];
                Vec4f L = sup[l], R = sup[r], O = sdown[i];
                Vec4f U = qup[l], V = qup[r], W = qdown[i];
                a += crossf<Trip>(L,R)*W;
                b += crossf<Trip>(L,O)*V;
                c += crossf<Trip>(R,O)*U;
                if (l >= n) {
                    float q = t[l].w;
                    sdown[l] = splat(t[l].flank)*(R+O);
                    pdown[l] = pdown[i]+pup[r]+paired<Trip>(R,O);
                    qdown[l] = splat(q)*pdown[l]+splat(1-q)*(W+V);
                }
                if (r >= n) {
                    float q = t[r].w;
                    sdown[r] = splat(t[r].flank)*(L+O);
                    pdown[r] = pdown[i]+pup[l]+paired<Trip>(L,O);
                    qdown[r] = splat(q)*pdown[r]+splat(1-q)*(W+U);
                }
            }
            Vec4f sum = a+b+c;
            for (int j = 0; j < 3; ++j) {
                if (per_gene) per_gene[g][j] = sum[j];
                result[j] += double(sum[j])*locus_weight[g];
            }
        }
        return result;
    }

    template<bool Trip> std::array<double, 3> evaluate(const Mask* group,
                                                     std::array<float,3>* per_gene = nullptr) const {
        if (use_flanks) return evaluate_flank<Trip>(group,nullptr,per_gene);
        constexpr int nc = Trip ? 3 : 4;
        int end = 2*n-2;
        uint32_t count[2*capacity]{};
        Vec4f upward[2*capacity], downward[2*capacity];
        uint32_t all = 0;
        for (int i = 0; i < nc; ++i) {
            uint32_t color = 1U << (8*i);
            all |= (uint32_t(group[i].size()) << (8*i));
            group[i].each([&](int j) { count[j] = color; });
        }
        for (int i = 0; i < n; ++i) upward[i] = Vec4f{};
        downward[end-1] = Vec4f{};
        std::array<double, 3> result{};
        for (size_t g = 0; g < forest.size(); ++g) {
            const auto& tree = forest[g];
            for (int i = n; i < end; ++i) {
                auto [l,r,w,e] = tree[i];
                count[i] = count[l] + count[r];
                upward[i] = splat(w)*pairs<Trip>(count[i]) + splat(1-w)*(upward[l] + upward[r]);
            }
            Vec4f a{}, b{}, c{};
            for (int i = end-1; i >= n; --i) {
                auto [l,r,w,e] = tree[i];
                uint32_t L = count[l], R = count[r], P = all-count[i];
                const Vec4f U = upward[l], V = upward[r], W = downward[i];
                a += cross<Trip>(L,R)*W;
                b += cross<Trip>(L,P)*V;
                c += cross<Trip>(R,P)*U;
                if (l >= n) {
                    float v = tree[l].w;
                    downward[l] = splat(v)*pairs<Trip>(all-L) + splat(1-v)*(V+W);
                }
                if (r >= n) {
                    float v = tree[r].w;
                    downward[r] = splat(v)*pairs<Trip>(all-R) + splat(1-v)*(U+W);
                }
            }
            Vec4f sum = a+b+c;
            for (int j = 0; j < 3; ++j) {
                if (per_gene) per_gene[g][j] = sum[j];
                result[j] += double(sum[j])*locus_weight[g];
            }
        }
        return result;
    }

public:
    QuartetScorer(const Input& data, const LocusEvidence& loc, const QuartetConfig& cfg,
                  const std::vector<double>* bootstrap = nullptr,
                  const std::vector<std::vector<float>>* tip_weight = nullptr)
        : n(data.n()), out(data.out),
          use_flanks(cfg.flank_abs != 0 || cfg.flank_rel != 0 || cfg.flank_topo != 0 ||
                     cfg.tip_only != 0 || tip_weight) {
        size_t g = 0;
        for (const Tree& t : data.genes) {
            double w = std::pow(loc.weight[g], cfg.locus_power);
            if (bootstrap) w *= (*bootstrap)[g];
            if (w == 0) {
                ++g;
                continue;
            }
            locus_weight.push_back(float(w));
            weight_sum += w;
            std::vector<Entry> tree(n, {0,0,0,1});
            double tau = cfg.length_abs + cfg.length_rel*loc.scale[g];
            double flank_rate = cfg.flank_abs + cfg.flank_rel/(.02+loc.scale[g]);
            for (int i = 0; i < n; ++i) {
                Edge e = t.v[i].edges[0];
                tree[i].flank = float(std::exp(-(flank_rate+cfg.tip_only)*e.length));
                if (tip_weight) tree[i].flank *= (*tip_weight)[g][i];
            }
            std::function<int(int,int)> walk = [&](int i, int parent) {
                if (i < n) return i;
                int child[2], k = 0;
                Edge up;
                for (int j = 0; j < t.v[i].degree; ++j) {
                    Edge e = t.v[i].edges[j];
                    if (e.to != parent) child[k++] = walk(e.to, i);
                    else up = e;
                }
                double ql = 1, qs = 1;
                if (tau > 0) {
                    double x = up.length/tau;
                    if (cfg.length_power != 1) x = std::pow(x, cfg.length_power);
                    ql = std::exp(-x);
                }
                if (cfg.support_power > 0) {
                    double r = edge_reliability(up.length, up.support);
                    qs = std::pow(1-r, cfg.support_power);
                }
                double w = !tau && cfg.support_power == 0 ? 1 : 1-ql*qs;
                if (cfg.intersection > 0 && tau && cfg.support_power > 0)
                    w = (1-cfg.intersection)*w + cfg.intersection*(1-ql)*(.25+.75*(1-qs));
                int id = int(tree.size());
                double flank = std::exp(-flank_rate*up.length - cfg.flank_topo*edge_reliability(up.length,up.support));
                tree.push_back({uint16_t(child[0]),uint16_t(child[1]),float(w),float(flank)});
                return id;
            };
            walk(t.v[data.out].edges[0].to, data.out);
            forest.push_back(std::move(tree));
            ++g;
        }
        cache.reserve(2600);
    }

    std::array<double, 3> quad(const std::array<Mask,4>& groups) {
        std::array<int,4> inverse;
        QuartetKey key = canonical(groups,inverse);
        if (!resample_weight.empty()) {
            auto it = locus_cache.find(key);
            std::vector<std::array<float,3>> scratch;
            const std::vector<std::array<float,3>>* values;
            if (it == locus_cache.end()) {
                scratch.resize(forest.size());
                evaluate<false>(key.group.data(),scratch.data());
                if (locus_cache.size() < 15000)
                    values = &locus_cache.emplace(key,std::move(scratch)).first->second;
                else values = &scratch;
            } else values = &it->second;
            std::array<double,3> result{};
            double div = resample_sum;
            for (const Mask& m : key.group) div *= m.size();
            for (size_t g = 0; g < forest.size(); ++g)
                for (int i = 0; i < 3; ++i) result[i] += resample_weight[g]*(*values)[g][i];
            for (double& d : result) d /= div;
            return permute(result,inverse);
        }
        auto it = cache.find(key);
        if (it == cache.end()) {
            auto x = evaluate<false>(key.group.data());
            double denom = weight_sum;
            for (const Mask& m : key.group) denom *= m.size();
            for (double& y : x) y /= denom;
            it = cache.emplace(key, x).first;
        }
        return permute(it->second,inverse);
    }

    void resample(const std::vector<double>& w) {
        if (w.empty()) {
            resample_weight.clear();
            return;
        }
        if (w.size() != forest.size()) throw std::runtime_error("invalid resample size");
        resample_weight.resize(w.size());
        resample_sum = 0;
        for (size_t g = 0; g < w.size(); ++g) {
            resample_weight[g] = w[g]*locus_weight[g];
            resample_sum += resample_weight[g];
        }
    }

    std::array<double,3> coherent(const std::array<Mask,4>& groups) {
        std::array<int,4> inverse;
        QuartetKey key = canonical(groups,inverse);
        auto it = coherent_cache.find(key);
        if (it == coherent_cache.end())
            it = coherent_cache.emplace(key,evaluate_coherent(key.group)).first;
        return permute(it->second,inverse);
    }

    double trip(const Mask& a, const Mask& b, const Mask& c) const {
        std::array<Mask,3> group{a,b,c};
        auto x = evaluate<true>(group.data());
        return x[0]+x[1]+x[2];
    }

    std::array<double,3> weighted_quad(const std::array<Mask,4>& g, const double* leaf_weight) const {
        auto x = evaluate_flank<false>(g.data(),leaf_weight);
        double denom = weight_sum;
        for (const Mask& m : g) {
            double sum = 0;
            m.each([&](int i) { sum += leaf_weight[i]; });
            denom *= sum;
        }
        for (double& v : x) v /= denom;
        return x;
    }
};

inline Mask beneath(const Tree& t, int u, int p) {
    if (u < t.n) return Mask::leaf(u);
    Mask m;
    for (int i = 0; i < t.v[u].degree; ++i) {
        int c = t.v[u].edges[i].to;
        if (c != p) m = m | beneath(t,c,u);
    }
    return m;
}

struct NniConfig {
    int rounds = 6;
    double split_prior = .08;
    double distance = .0;
    double representation = 0;
    double locality = 0;
    double coherence = 0;
};

struct NniProtection {
    std::function<double(const Mask&)> posterior;
    double lock = .9;
    double bias = 0;
};

inline double between(const Matrix& d, const Mask& a, const Mask& b) {
    double x = 0;
    a.each([&](int i) {
        double s = 0;
        b.each([&](int j) { s += d(i,j); });
        x += s;
    });
    return x / (a.size()*b.size());
}

inline std::array<double,3> local_scores(QuartetScorer& q, const std::array<Mask,4>& g,
                                        const SplitEvidence& splits, const Matrix* d, const NniConfig& cfg,
                                        const double* tip_weights = nullptr) {
    auto s = tip_weights ? q.weighted_quad(g,tip_weights) : q.quad(g);
    if (cfg.coherence) {
        auto z = q.coherent(g);
        for (int i = 0; i < 3; ++i) s[i] += cfg.coherence*z[i];
    }
    for (int i = 0; i < 3; ++i) {
        if (cfg.split_prior) s[i] += cfg.split_prior*splits.signal(g[0] | g[i+1]);
        if (d && cfg.distance) {
            int j = (i+1)%3, k = (i+2)%3;
            s[i] -= cfg.distance*(between(*d,g[0],g[i+1])+between(*d,g[j+1],g[k+1]));
        }
    }
    return s;
}

inline void leaf_representatives(const Tree& t, const std::array<Mask,4>& groups,
                                  const std::array<int,4>& nodes, int u, int v,
                                  double alpha, double locality, const Matrix* dist, double* w) {
    double fraction = std::exp2(-alpha);
    auto visit = [&](auto& self, int x, int p, double f) -> void {
        if (x < t.n) w[x] = f;
        else
            for (int j = 0; j < 3; ++j) {
                int c = t.v[x].edges[j].to;
                if (c != p) self(self,c,x,f*fraction);
            }
    };
    for (int k = 0; k < 4; ++k) {
        const Mask& m = groups[k];
        visit(visit,nodes[k],k < 2 ? u : v,1);
        double sum = 0;
        m.each([&](int i) { sum += w[i]; });
        const double norm = m.size()/sum;
        m.each([&](int i) { w[i] = .2 + .8*w[i]*norm; });
        if (locality > 0 && dist && m.size() >= 4) {
            double loc[capacity];
            double closest = 1e300;
            m.each([&](int i) {
                double av = 0;
                for (int j = 0; j < 4; ++j) {
                    if (k==j) continue;
                    double z = 0;
                    groups[j].each([&](int x) { z += (*dist)(i,x); });
                    av += z/groups[j].size();
                }
                loc[i] = av/3;
                closest = std::min(closest,loc[i]);
            });
            m.each([&](int i) { w[i] *= .16 + .84*std::exp(-locality*(loc[i]-closest)); });
        }
    }
}

inline int refine_nni(Tree& t, QuartetScorer& q, const SplitEvidence& splits,
                      const Matrix* dist, const NniConfig& cfg,
                      const NniProtection* protect = nullptr) {
    int changes = 0;
    for (int pass = 0; pass < cfg.rounds; ++pass) {
        int moved = 0;
        for (int u = t.n; u < int(t.v.size()); ++u) {
            for (int j = 0; j < 3; ++j) {
                int v = t.v[u].edges[j].to;
                if (v < t.n || v >= u) continue;
                std::array<int,4> sub;
                int a = 0, b = 2;
                for (int k = 0; k < 3; ++k) {
                    if (t.v[u].edges[k].to != v) sub[a++] = t.v[u].edges[k].to;
                    if (t.v[v].edges[k].to != u) sub[b++] = t.v[v].edges[k].to;
                }
                std::array<Mask,4> g;
                for (int i = 0; i < 4; ++i) g[i] = beneath(t,sub[i],i < 2 ? u : v);
                std::array<double,3> prior{};
                if (protect) {
                    auto posterior = [&](Mask m) {
                        return protect->posterior(m.has(splits.out) ? splits.all^m : m);
                    };
                    prior[0] = posterior(g[0]|g[1]);
                    if (prior[0] >= protect->lock) continue;
                    if (protect->bias != 0)
                        for (int i = 1; i < 3; ++i) prior[i] = posterior(g[0]|g[i+1]);
                }
                double w[capacity];
                if (cfg.representation > 0 || cfg.locality > 0)
                    leaf_representatives(t,g,sub,u,v,cfg.representation,cfg.locality,dist,w);
                auto s = local_scores(q,g,splits,dist,cfg,
                                      cfg.representation > 0 || cfg.locality > 0 ? w : nullptr);
                if (protect && protect->bias)
                    for (int i = 0; i < 3; ++i) s[i] += protect->bias*prior[i];
                int best = int(std::max_element(s.begin(),s.end())-s.begin());
                if (best && s[best] > s[0] + 1e-7) {
                    t.swap_subtrees(u,v,sub[1],sub[best+1]);
                    ++moved;
                }
            }
        }
        changes += moved;
        if (!moved) break;
    }
    return changes;
}

}  // namespace st
