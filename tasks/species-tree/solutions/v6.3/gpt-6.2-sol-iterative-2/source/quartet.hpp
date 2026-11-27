#pragma once

#include "distance.hpp"
#include "splits.hpp"
#include <functional>
#include <immintrin.h>
#include <random>

struct QuartetOptions {
    float decay = 2.0f;
    float length_mix = 0.45f;
    float internal_scale = 0.0075f;
    float support_power = 1.0f;
    float gene_power = 0.0f;
    float per_gene_norm = 0.0f;
    float norm_floor = 0.5f;
    float scale_floor = 0.006f;
    float adaptive_scale = 0;
    float tip_factor = 1.0f;
    float upper_scale = 0;
};

struct QuartetGene {
    struct Joint {
        int l, r, lidx, ridx;
        float wl, wr, fl, fr;
    };
    std::vector<Joint> joints;
    std::vector<float> leaf_reliability;
    float w0 = 1, multiplier = 1;
};

class QuartetKernel {
public:
    static constexpr int MAXN = 256;
private:
    std::vector<QuartetGene> gt;
    int n;
    float per_gene_norm;
    float norm_floor;
    struct alignas(32) Pairs { __m256 all, fail; };
    struct alignas(16) Singles { __m128 value; };
    struct alignas(16) Children { __m128 l, r; };

    static __m256 pairs(__m128 x, __m128 y) {
        __m256 a = _mm256_castps128_ps256(x);
        __m256 b = _mm256_castps128_ps256(y);
        a = _mm256_insertf128_ps(a, x, 1);
        b = _mm256_insertf128_ps(b, y, 1);
        const __m256i fi = _mm256_setr_epi32(0,0,0,1,1,2,0,0);
        const __m256i se = _mm256_setr_epi32(1,2,3,2,3,3,0,0);
        __m256 ax = _mm256_permutevar8x32_ps(a, fi);
        __m256 ay = _mm256_permutevar8x32_ps(a, se);
        __m256 bx = _mm256_permutevar8x32_ps(b, fi);
        __m256 by = _mm256_permutevar8x32_ps(b, se);
        return _mm256_fmadd_ps(ax, by, _mm256_mul_ps(ay,bx));
    }
    static __m256 complement(__m256 x) {
        return _mm256_permutevar8x32_ps(x, _mm256_setr_epi32(5,4,3,2,1,0,6,7));
    }

    static int arrange(QuartetGene &g, const Tree &t, int u, int p, float scale,
                       const QuartetOptions &o) {
        if (u < t.n) return u;
        Arc a, b;
        bool first = true;
        for (Arc e : t.v[u].edges) if (e.to != p) {
            if (first) a = e;
            else b = e;
            first = false;
        }
        int l = arrange(g,t,a.to,u,scale,o), r = arrange(g,t,b.to,u,scale,o);
        auto failure = [&](Arc x) -> float {
            float v = std::min(10.0f,x.length/(o.internal_scale + o.adaptive_scale*scale));
            float r = (1-o.length_mix)*std::pow(reliability(x), o.support_power);
            return 1-(r-o.length_mix*std::expm1(-v));
        };
        int here = g.joints.size();
        auto spatial_weight = [&](Arc x) {
            return std::exp(-o.decay*x.length/scale*(x.to < t.n ? o.tip_factor : 1.0f));
        };
        g.joints.push_back({l,r,l-t.n,r-t.n,spatial_weight(a),spatial_weight(b),failure(a),failure(b)});
        return t.n+here;
    }
public:
    QuartetKernel(const GeneSet &set, const std::vector<GeneDistances> &d, const QuartetOptions &o,
                  const std::vector<float> &gene_weights = {},
                  const std::vector<std::vector<float>> &leaf_reliability = {})
        : n(set.n), per_gene_norm(o.per_gene_norm), norm_floor(o.norm_floor) {
        if (n > MAXN) throw std::runtime_error("too many taxa");
        for (size_t i = 0; i < set.genes.size(); ++i) {
            const auto &t = set.genes[i];
            QuartetGene g;
            Arc a = t.v[0].edges[0];
            float scale = d[i].scale + o.scale_floor;
            g.joints.reserve(t.n-2);
            g.w0 = std::exp(-o.decay*a.length/scale*o.tip_factor);
            g.multiplier = std::pow(d[i].scale / (d[i].scale+0.045f), o.gene_power);
            if (o.upper_scale > 0)
                g.multiplier /= 1+std::pow(d[i].scale/o.upper_scale,2.0f);
            if (!gene_weights.empty()) g.multiplier *= gene_weights[i];
            if (!leaf_reliability.empty()) g.leaf_reliability = leaf_reliability[i];
            arrange(g,t,a.to,0,scale,o);
            gt.push_back(std::move(g));
        }
    }

    // A quartet has two branching endpoints. Pair messages retain both the
    // total pendant weight and the portion whose internal path stays unresolved.
    Votes evaluate(const Quad &groups, std::vector<Votes> &details,
                   const std::vector<float> &leaf_weights) const {
        Singles s[2*MAXN], outside[MAXN], leaves[MAXN];
        Pairs messages[MAXN];
        Children below[MAXN];
        for (int i = 0; i < n; ++i) s[i].value = _mm_setzero_ps();
        const __m128 units[4] = {_mm_setr_ps(1,0,0,0), _mm_setr_ps(0,1,0,0),
                                 _mm_setr_ps(0,0,1,0), _mm_setr_ps(0,0,0,1)};
        float total_possible = 1;
        for (int i = 0; i < 4; ++i) {
            float tot = 0;
            groups[i].each([&,u=units[i]](int k) {
                float w = leaf_weights.empty() ? 1.0f : leaf_weights[k];
                s[k].value = _mm_mul_ps(u, _mm_set1_ps(w));
                tot += w;
            });
            total_possible *= tot;
        }
        details.assign(gt.size(), Votes{});
        std::copy(s,s+n,leaves);

        const int last = n-3;
        const auto zero = _mm256_setzero_ps();
        const Pairs empty{zero,zero};
        auto result = zero;
        for (size_t k = 0; k < gt.size(); ++k) {
            const auto &g = gt[k];
            float w = g.multiplier;
            if (w <= 0) continue;
            if (!g.leaf_reliability.empty()) {
                for (int i=0;i<n;++i)
                    s[i].value=_mm_mul_ps(leaves[i].value,_mm_set1_ps(g.leaf_reliability[i]));
            }
            for (int i = 0; i <= last; ++i) {
                const auto &a = g.joints[i];
                __m128 sl = _mm_mul_ps(s[a.l].value,_mm_set1_ps(a.wl));
                __m128 sr = _mm_mul_ps(s[a.r].value,_mm_set1_ps(a.wr));
                below[i] = {sl,sr};
                s[n+i].value = _mm_add_ps(sl,sr);
            }
            outside[last].value = _mm_mul_ps(s[0].value,_mm_set1_ps(g.w0));
            for (int i = last; i >= 0; --i) {
                const auto &a = g.joints[i];
                __m128 os = outside[i].value;
                if (a.lidx >= 0)
                    outside[a.lidx].value = _mm_mul_ps(_mm_add_ps(os,below[i].r),_mm_set1_ps(a.wl));
                if (a.ridx >= 0)
                    outside[a.ridx].value = _mm_mul_ps(_mm_add_ps(os,below[i].l),_mm_set1_ps(a.wr));
            }
            auto total = zero;
            for (int i = 0; i <= last; ++i) {
                const auto &a = g.joints[i];
                bool il = a.lidx >= 0, ir = a.ridx >= 0;
                Pairs pl = il ? messages[a.lidx] : empty;
                Pairs pr = ir ? messages[a.ridx] : empty;
                pl.fail = _mm256_mul_ps(pl.fail, _mm256_set1_ps(a.fl));
                pr.fail = _mm256_mul_ps(pr.fail, _mm256_set1_ps(a.fr));
                __m128 sl = below[i].l, sr = below[i].r, os = outside[i].value;
                if (il && ir) {
                    // Quartet path goes through both child edges.
                    total = _mm256_add_ps(total,
                               _mm256_fmsub_ps(pl.all,complement(pr.all),
                                               _mm256_mul_ps(pl.fail,complement(pr.fail))));
                }
                if (il) {
                    // The upper endpoint sends one leaf into each of the other two arcs.
                    auto ev = _mm256_sub_ps(pl.all,pl.fail);
                    total = _mm256_fmadd_ps(ev,
                                           complement(pairs(sr,os)),total);
                }
                if (ir) {
                    auto ev = _mm256_sub_ps(pr.all,pr.fail);
                    total = _mm256_fmadd_ps(ev,
                                           complement(pairs(sl,os)),total);
                }
                __m256 split = pairs(sl,sr);
                messages[i] = {_mm256_add_ps(split,_mm256_add_ps(pl.all,pr.all)),
                               _mm256_add_ps(split,_mm256_add_ps(pl.fail,pr.fail))};
            }
            alignas(32) float v[8];
            _mm256_store_ps(v,total);
            if (per_gene_norm > 0) {
                float total_good = (v[0]+v[1])+(v[2]+v[3])+(v[4]+v[5]);
                float divisor = norm_floor + total_good / std::max(1.0f,total_possible);
                w /= std::pow(divisor,per_gene_norm);
            }
            details[k] = {w*(v[0]+v[5]), w*(v[1]+v[4]), w*(v[2]+v[3])};
            result = _mm256_fmadd_ps(_mm256_set1_ps(w),total,result);
        }
        alignas(32) float f[8];
        _mm256_store_ps(f,result);
        return {f[0]+f[5], f[1]+f[4], f[2]+f[3]};
    }
};

struct OrderQuad {
    Quad canonical;
    std::array<int,3> mapping;
    explicit OrderQuad(const Quad &q) {
        std::array<int,4> p{0,1,2,3};
        std::sort(p.begin(),p.end(),[&](int a, int b) { return q[a] < q[b]; });
        for (int i = 0; i < 4; ++i) canonical[i] = q[p[i]];
        for (int i = 0; i < 3; ++i) mapping[i] = (p[0]^p[i+1])-1;
    }
    Votes unorder(Votes v) const {
        Votes out;
        for (int i = 0; i < 3; ++i) out[mapping[i]] = v[i];
        return out;
    }
};

class LocusQuartets {
public:
    struct Entry {
        Votes sum{};
        std::vector<Votes> part;
    };
    const QuartetKernel &kernel;
    std::unordered_map<Quad, Entry, QuadHash> data;
    explicit LocusQuartets(const QuartetKernel &k) : kernel(k) { data.reserve(4096); }
    using LeafWeightFunction = std::function<std::vector<float>(const Quad &)>;
    LeafWeightFunction leaf_weights;
    const Entry &entry(const Quad &canon) {
        auto [it, add] = data.try_emplace(canon);
        if (add) it->second.sum = kernel.evaluate(canon,it->second.part,
                                                 leaf_weights ? leaf_weights(canon) : std::vector<float>{});
        return it->second;
    }
    Votes operator()(const Quad &q) {
        OrderQuad p(q);
        return p.unorder(entry(p.canonical).sum);
    }

    float zsupport(const Quad &q) {
        OrderQuad p(q);
        const auto &e = entry(p.canonical);
        Votes v = p.unorder(e.sum);
        std::array<double,2> squares{};
        for (Votes a : e.part) {
            a = p.unorder(a);
            for (int j = 0; j < 2; ++j) {
                double s = a[0]-a[j+1];
                squares[j] += s*s;
            }
        }
        float z = 1000;
        for (int j = 0; j < 2; ++j) {
            double dif = v[0]-v[j+1];
            double var = squares[j]-dif*dif/e.part.size();
            double r = dif / std::sqrt(std::max(1e-25,var));
            z = std::min(z, float(r));
        }
        return z;
    }
};

template<class Oracle>
inline int nni_sweep(Tree &t, Oracle &score, std::mt19937 &rng, int sweeps = 12) {
    int total = 0;
    for (int round = 0; round < sweeps; ++round) {
        MaskTree q(t);
        auto &edges = q.internal;
        std::shuffle(edges.begin(),edges.end(),rng);
        int changes = 0;
        for (auto [u,v] : edges) {
            bool adjacent = false;
            for (auto e : t.v[u].edges) adjacent |= e.to == v;
            if (!adjacent) continue;
            Votes votes = score(edge_groups(t,u,v));
            int win = std::max_element(votes.begin(),votes.end()) - votes.begin();
            float tolerance = 2e-7f*(votes[0]+votes[1]+votes[2]);
            if (win != 0 && votes[win] > votes[0]+tolerance) {
                auto a = other_neighbors(t,u,v), b = other_neighbors(t,v,u);
                t.nni(u,v,a[1],b[win-1]);
                ++changes;
            }
        }
        total += changes;
        if (changes == 0) break;
    }
    return total;
}
