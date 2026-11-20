#pragma once
#include "inference.hpp"
#include "clades.hpp"
#include <random>

namespace st {

struct EnsembleOptions {
    int replicates = 30;
    double boot_fraction = .42;
    double boot_amplitude = .82;
    double seed_fraction = .12;
};

inline std::vector<DistConfig> distance_configs() {
    std::vector<DistConfig> c(8);
    c[0] = {EdgeMode::Soft,1.,0.,.73,.12,.38};
    c[1] = {EdgeMode::Soft,.67,0.,.91,.14,.6};
    c[2] = {EdgeMode::ShortLength,1.,.03,.79,.12,.92,.018};
    c[3] = {EdgeMode::Support,1.,0.,.80,.12,.25};
    c[4] = {EdgeMode::Length,1.,.12,.84,.12,.98};
    c[5] = {EdgeMode::Topology,1.,0.,.57,.12,.20};
    c[6] = {EdgeMode::InternalLength,1.,.09,.66,.12,.93};
    c[7] = {EdgeMode::ShortLength,.75,.06,.88,.12,.94,.065};
    return c;
}

struct EnsembleModel {
    QuartetConfig quartet;
    NniConfig refine;
    double weight;
};

inline std::vector<EnsembleModel> ensemble_models() {
    return {
        {{.55,.034,.135,1.0,.30,0,.10,0,0}, {6,.19,.17}, 1.15},
        {{.95,.027,.105,1.0,.15,0,.0,0,0}, {6,.18,.19}, 1.0},
        {{.38,.036,.135,1.5,.36,0,.28,.023,0}, {6,.23,.19}, 1.05},
        {{.30,.032,.14,1.0,.52,0,0,0,0}, {6,.14,.16}, .85},
        {{.67,.038,.15,1.0,.25,0,.58,.045,0}, {6,.22,.18}, 1.0},
        {{.8,0,0,1.0,.34,0,.07,0,.23}, {6,.26,.20}, .9}
    };
}

class SplitBag {
    std::unordered_map<Mask,double,MaskHash> counts;
    double mass = 0;
    int out;
public:
    explicit SplitBag(int out) : out(out) {}
    void add(const Tree& tree,double weight) {
        tree.clades(tree.v[out].edges[0].to,out,[&](const Mask& m,Edge) { counts[m] += weight; });
        mass += weight;
    }
    double frequency(const Mask& m) const {
        auto it = counts.find(m);
        return it == counts.end() ? 0.0 : it->second / mass;
    }
    void include(CladeSet& set) const {
        for (const auto& p : counts) set.insert(p.first);
    }
};

inline Matrix distance_mix(const std::vector<Matrix>& ds, const std::vector<double>& weights) {
    Matrix d(ds[0].n);
    double w = std::accumulate(weights.begin(),weights.end(),0.);
    for (size_t k = 0; k < ds.size(); ++k)
        for (size_t i = 0; i < d.data.size(); ++i) d.data[i] += (weights[k]/w)*ds[k].data[i];
    return d;
}

inline Tree rf_consensus(const Input& data,const SplitEvidence& sp,const Tree& fallback,
                         const SplitBag& point,const SplitBag* bootstrap = nullptr,double boot_weight = 0,
                         const SplitBag* seeds = nullptr,double seed_weight = 0) {
    CladeSet set;
    point.include(set);
    if (bootstrap) bootstrap->include(set);
    if (seeds) seeds->include(set);
    add_clades(set,fallback,data.out);
    CladeProblem p(data,set,150000);
    if (!p.complete()) return fallback;
    return p.solve(nullptr,[&](const Mask& m) {
        return (1.-boot_weight-seed_weight)*point.frequency(m) +
               (bootstrap ? boot_weight*bootstrap->frequency(m) : 0.) +
               (seeds ? seed_weight*seeds->frequency(m) : 0.) + .012*sp.signal(m);
    });
}

inline uint64_t inference_seed() {
    const char* s = std::getenv("SEED");
    if (!s || !*s) return 1;
    char* end;
    auto v = std::strtoull(s,&end,10);
    return *end ? 1 : v;
}

inline Tree ensemble_tree(const Input& data, const EnsembleOptions& opt = {},
                          const std::function<void(const Tree&,const char*)>& diagnostic = {}) {
    LocusEvidence loc(data);
    Starter starter(data,loc);
    SplitEvidence sp(data,loc,loc.weights(.32));
    const auto dc = distance_configs();
    const auto qm = ensemble_models();
    const std::vector<double> dweight{1.3,1.35,1.15,1.1,.55,.65,.75,.7};
    std::vector<Matrix> ds;
    for (size_t i = 0; i < dc.size(); ++i) ds.push_back(starter.make(dc[i],i == 4 ? .65 : .38));
    Matrix mix = distance_mix(ds,dweight);

    std::vector<Tree> start;
    SplitBag seeds(data.out), point(data.out), boot(data.out);
    start.push_back(join(mix,1));
    start.push_back(join(mix,0));
    for (size_t i = 0; i < ds.size(); ++i) {
        start.push_back(join(ds[i],i%2));
        seeds.add(start.back(),dweight[i]);
    }
    const Tree& fallback = start[0];
    seeds.add(fallback,1.5);
    if (diagnostic) diagnostic(fallback,"distance");

    for (size_t k = 0; k < qm.size(); ++k) {
        const auto& model = qm[k];
        QuartetScorer score(data,loc,model.quartet);
        for (int j = 0; j < 4; ++j) {
            size_t i = j < 2 ? j : (2*k+(j-2)*3) % ds.size()+2;
            Tree t = start[i];
            refine_nni(t,score,sp,&mix,model.refine);
            point.add(t,model.weight * (j < 2 ? 1.0 : .8));
            if (diagnostic) diagnostic(t,"refine");
        }
    }
    Tree middle = rf_consensus(data,sp,fallback,point,nullptr,0.,&seeds,opt.seed_fraction);
    if (diagnostic) diagnostic(middle,"point");
    if (opt.replicates <= 0) return middle;

    std::mt19937_64 rng(inference_seed());
    std::uniform_real_distribution<double> u(0.,1.);
    for (int r = 0; r < opt.replicates; ++r) {
        std::vector<double> w(data.genes.size());
        for (double& x : w) x = (1.-opt.boot_amplitude) - opt.boot_amplitude*std::log1p(-u(rng));
        size_t k = r % qm.size();
        const auto& model = qm[k];
        QuartetScorer score(data,loc,model.quartet,&w);
        std::vector<double> wsplit = loc.weights(.32);
        for (size_t j = 0; j < w.size(); ++j) wsplit[j] *= w[j];
        SplitEvidence resampled(data,loc,wsplit);
        Matrix dr = mix;
        Tree t = middle;
        if (r%3 != 0) {
            size_t i = (r/3+r%3) % dc.size();
            Matrix m = starter.make(dc[i],.38,&w);
            t = join(m,r%2);
            for (size_t j = 0; j < m.data.size(); ++j) dr.data[j] = .70*m.data[j] + .30*mix.data[j];
        }
        NniConfig conf = model.refine;
        conf.rounds = 5;
        refine_nni(t,score,resampled,&dr,conf);
        boot.add(t,model.weight);
        if (diagnostic) diagnostic(t,"bootstrap");
    }
    return rf_consensus(data,sp,middle,point,&boot,opt.boot_fraction,&seeds,opt.seed_fraction);
}

} // namespace st
