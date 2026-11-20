#pragma once
#include "ensemble.hpp"
#include "join_repeated.hpp"

namespace st {

inline Tree mosaic_tree(const Input& data) {
    SplitBag qp(data.out), qb(data.out), dist(data.out), db(data.out), base(data.out);
    Tree baseline = ensemble_tree(data,{24,.42,.82,.12},[&](const Tree& t,const char* label) {
        if (label[0]=='r') qp.add(t,1.);
        if (label[0]=='b') qb.add(t,1.);
    });
    base.add(baseline,1.);
    LocusEvidence le(data);
    Starter starter(data,le);
    SplitEvidence sp(data,le,le.weights(.29));
    auto cfg=distance_configs();
    cfg.push_back({EdgeMode::Topology,1.,.04,.97,.12,.37,.02,2.6});
    cfg.push_back({EdgeMode::Topology,1.,.0,.73,.12,.27});
    cfg.push_back({EdgeMode::ShortLength,1.,.045,.96,.12,.92,.017,2.1});
    cfg.push_back({EdgeMode::Soft,1.,.02,.97,.12,.61,.02,2.2});
    cfg.push_back({EdgeMode::Topology,1.,.0,.48,.12,.29});
    std::vector<double> weight{1.2,1.15,1.05,1.05,.45,.90,.55,.65,1.0,.8,.72,.72,.75};
    std::vector<Matrix> ds;
    for (size_t i=0;i<cfg.size();++i) ds.push_back(starter.make(cfg[i],i==4 ? .61 : .32));
    Matrix mix = distance_mix(ds,weight);
    for (size_t i=0;i<cfg.size();++i) {
        for (int j=0;j<2;++j) {
            Matrix d=ds[i];
            double frac = j==0 ? .86 : .65;
            for (size_t k=0;k<d.data.size();++k) d.data[k]=frac*d.data[k]+(1-frac)*mix.data[k];
            dist.add(join(std::move(d),j),weight[i]);
        }
    }
    RepeatedPaths sdata(data,le,{EdgeMode::Soft,1.,0.,1.,.12,.51},.32);
    for (const auto& setting:std::vector<RepeatedJoinConfig>{{0,.30,0},{.52,.24,0},{.62,.48,.13}}) {
        dist.add(repeated_join(mix,sdata,setting),1.6);
    }
    for (int k : {0,1}) dist.add(join(mix,k),1.65);

    std::mt19937_64 rng(inference_seed() ^ 0x364a9021f16bc50bULL);
    std::uniform_real_distribution<double> u(0.,1.);
    for (int r=0;r<20;++r) {
        size_t k=(5*r) % cfg.size();
        double amp = r % 4 == 0 ? .84 : .64;
        std::vector<double> w(data.genes.size());
        for(double& x:w) x=1-amp-amp*std::log1p(-u(rng));
        Matrix d=starter.make(cfg[k],.32,&w);
        for (size_t i=0;i<d.data.size();++i) d.data[i]=.76*d.data[i]+.24*mix.data[i];
        db.add(join(std::move(d),r%2),weight[k]);
    }
    CladeSet set;
    qp.include(set);
    qb.include(set);
    dist.include(set);
    db.include(set);
    base.include(set);
    CladeProblem prob(data,set);
    if (!prob.complete()) return baseline;

    // Divergent length scales make the distance model less reliable.
    auto scales = le.scale;
    std::sort(scales.begin(),scales.end());
    const double scale_spread = std::log(std::max(1e-8,quantile_sorted(scales,.90)) /
                                         std::max(1e-8,quantile_sorted(scales,.10)));
    const double w = .18 + .43*std::clamp((3.46-scale_spread)/2.15,0.,1.);
    return prob.solve(nullptr,[&](const Mask& m) {
        double q=.54*qp.frequency(m)+.38*qb.frequency(m)+.08*base.frequency(m);
        double d=.73*dist.frequency(m)+.27*db.frequency(m);
        double k=std::min(m.size(),data.n()-m.size());
        double recurrent=sp.signal(m)*std::pow(std::max(1.,k-1),.4);
        return (1-w)*q + w*d + .09*recurrent;
    });
}

} // namespace st
