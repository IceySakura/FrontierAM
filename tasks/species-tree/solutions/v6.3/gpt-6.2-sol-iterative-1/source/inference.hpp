#pragma once
#include "quartet.hpp"

namespace st {

class Starter {
    const Input& data;
    const LocusEvidence& loc;
    struct Paths {
        std::vector<Matrix> mat;
        std::vector<double> scale;
    };
    std::map<std::pair<int,double>, Paths> by_mode;

public:
    Starter(const Input& data, const LocusEvidence& loc) : data(data), loc(loc) {}
    Matrix make(DistConfig c, double quality_power, const std::vector<double>* bs = nullptr) {
        auto key = std::make_pair(int(c.mode), c.mode == EdgeMode::ShortLength ? c.edge_scale : 0.);
        auto& p = by_mode[key];
        if (p.mat.empty()) {
            for (const Tree& g : data.genes) {
                p.mat.push_back(gene_distances(g,c.mode,c.edge_scale));
                p.scale.push_back(pair_scale(p.mat.back(),data.out));
            }
        }
        auto w = loc.weights(quality_power);
        if (bs)
            for (size_t i = 0; i < w.size(); ++i) w[i] *= (*bs)[i];
        return aggregate_distances(p.mat,p.scale,c,&w);
    }
};

inline Matrix initial_distance(Starter& s) {
    DistConfig topo, len;
    topo.mode = EdgeMode::Soft;
    topo.first = .02;
    topo.last = .85;
    topo.norm = .50;
    len.mode = EdgeMode::ShortLength;
    len.first = .05;
    len.last = .85;
    len.norm = .9;
    Matrix d = s.make(topo,.6);
    Matrix m = s.make(len,.8);
    for (size_t i = 0; i < d.data.size(); ++i) d.data[i] = .6*d.data[i] + .4*m.data[i];
    return d;
}

inline Tree infer_tree(const Input& data) {
    LocusEvidence loc(data);
    Starter start(data,loc);
    Matrix d = initial_distance(start);
    Tree t = join(d,1);
    QuartetScorer q(data,loc,QuartetConfig{});
    SplitEvidence split(data,loc,loc.weights(.65));
    refine_nni(t,q,split,&d,NniConfig{6,.08,.10});
    return t;
}

}  // namespace st
