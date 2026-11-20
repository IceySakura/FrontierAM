#pragma once
#include "distance.hpp"

namespace st {

struct LocusEvidence {
    std::vector<double> scale;
    std::vector<double> agreement;
    std::vector<double> weight;
    std::vector<double> fraction_informative;

    explicit LocusEvidence(const Input& in) {
        int n = in.n();
        int k = int(in.genes.size());
        std::vector<Matrix> shape;
        for (const Tree& gene : in.genes) {
            double useful = 0;
            gene.clades(gene.v[in.out].edges[0].to, in.out, [&](const Mask&, Edge e) {
                useful += edge_reliability(e.length, e.support);
            });
            fraction_informative.push_back(useful / (n-3));
            Matrix raw = gene_distances(gene, EdgeMode::Length);
            scale.push_back(pair_scale(raw, in.out));
            Matrix soft = gene_distances(gene, EdgeMode::Support);
            double med = pair_scale(soft, in.out);
            std::vector<double> row(n);
            double mean = 0;
            for (int i = 0; i < n; ++i) {
                if (i == in.out) continue;
                for (int j = 0; j < i; ++j) {
                    if (j == in.out) continue;
                    double x = std::min(1.75, .55*soft(i,j)/med + .45*raw(i,j)/scale.back());
                    soft(i,j) = x;
                    row[i] += x;
                    row[j] += x;
                }
            }
            for (double& x : row) {
                mean += x;
                x /= n-2;
            }
            mean /= double(n-1)*(n-2);
            for (int i = 0; i < n; ++i)
                if (i != in.out)
                    for (int j = 0; j < i; ++j)
                        if (j != in.out) soft(i,j) += mean - row[i] - row[j];
            shape.push_back(std::move(soft));
        }
        Matrix mean(n);
        std::vector<double> rough_w(k);
        for (int g = 0; g < k; ++g) {
            rough_w[g] = std::clamp(fraction_informative[g]*2, .03, 1.);
            for (int i = 0; i < n; ++i)
                if (i != in.out)
                    for (int j = 0; j < i; ++j)
                        if (j != in.out) mean(i,j) += rough_w[g]*shape[g](i,j);
        }
        for (int g = 0; g < k; ++g) {
            double cross = 0, x2 = 0, y2 = 0;
            for (int i = 0; i < n; ++i) {
                if (i == in.out) continue;
                for (int j = 0; j < i; ++j) {
                    if (j == in.out) continue;
                    double x = shape[g](i,j);
                    double y = mean(i,j) - rough_w[g]*x;
                    cross += x*y;
                    x2 += x*x;
                    y2 += y*y;
                }
            }
            double r = std::clamp(cross / std::sqrt(std::max(1e-20, x2*y2)), 0., 1.);
            agreement.push_back(r);
            weight.push_back(.04 + (r*r) / (1.05 - .8*r*r));
        }
        double base = median(weight);
        for (double& x : weight) x = std::clamp(x / base, .04, 2.8);
    }

    std::vector<double> weights(double power) const {
        std::vector<double> r;
        for (double w : weight) r.push_back(std::pow(w, power));
        return r;
    }
};

struct SplitEvidence {
    struct Vote {
        double count = 0;
        double supported = 0;
        double length = 0;
    };
    std::unordered_map<Mask, Vote, MaskHash> votes;
    int out;
    Mask all;
    double total = 0;

    SplitEvidence(const Input& in, const LocusEvidence& loc, const std::vector<double>& weights)
        : out(in.out), all(in.all) {
        for (int g = 0; g < int(in.genes.size()); ++g) {
            const Tree& t = in.genes[g];
            double w = weights[g];
            total += w;
            t.clades(t.v[out].edges[0].to, out, [&](const Mask& m, Edge e) {
                Vote& v = votes[m];
                v.count += w;
                v.supported += w*edge_reliability(e.length,e.support);
                v.length += w*(-std::expm1(-e.length / (.01 + .055*loc.scale[g])));
            });
        }
    }
    Vote lookup(Mask m) const {
        if (m.has(out)) m = all ^ m;
        auto it = votes.find(m);
        return it == votes.end() ? Vote{} : it->second;
    }
    double signal(const Mask& m) const {
        Vote v = lookup(m);
        return (.1*v.count + .55*v.supported + .35*v.length)/total;
    }
};

}  // namespace st
