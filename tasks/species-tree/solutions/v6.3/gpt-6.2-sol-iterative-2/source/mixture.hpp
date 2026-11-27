#pragma once

#include "weights.hpp"
#include "resample.hpp"
#include "dispersion.hpp"
#include "triplets.hpp"
#include "calibrate.hpp"
#include <optional>

class QuartetMixture {
    struct Entry {
        Votes sum{};
        std::vector<Votes> contributions;
    };
    struct Part { LocusQuartets *quartets; float weight; };
    std::vector<Part> models;
    std::unordered_map<Quad, Entry, QuadHash> entries;
    const Entry &load(const Quad &c) {
        auto [it,added]=entries.try_emplace(c);
        auto &entry=it->second;
        if (!added) return entry;
        auto &ref=models[0].quartets->entry(c);
        float scale=(ref.sum[0]+ref.sum[1]+ref.sum[2]);
        entry.contributions.assign(ref.part.size(),Votes{});
        if (scale<=1e-22f) scale=1e-22f;
        int cat=calibration?CalibrationMix::category(c):0;
        int idx=0;
        for (auto m:models) {
            auto &p=m.quartets->entry(c);
            float sum=p.sum[0]+p.sum[1]+p.sum[2];
            float w=(calibration?calibration->table[cat][idx]:m.weight)*scale/std::max(1e-22f,sum);
            ++idx;
            for (int j=0;j<3;++j) entry.sum[j]+=p.sum[j]*w;
            for (size_t k=0;k<p.part.size();++k)
                for (int j=0;j<3;++j) entry.contributions[k][j]+=p.part[k][j]*w;
        }
        if (extra) {
            auto v=extra(c);
            if (!v.empty()) {
                double tot=0;
                for (auto x:v) tot+=x[0]+x[1]+x[2];
                if (tot>1e-16) {
                    float w=extra_weight*scale/tot;
                    for (size_t k=0;k<v.size();++k)
                        for (int j=0;j<3;++j)
                            entry.contributions[k][j]=(1-extra_weight)*entry.contributions[k][j]+w*v[k][j];
                }
            }
        }
        if (adjustment>0 || clipping>0 || extra) {
            temper_dispersion(entry.contributions,adjustment,clipping);
            entry.sum={};
            for (const auto &v:entry.contributions)
                for (int j=0;j<3;++j) entry.sum[j]+=v[j];
        }
        if (shape_prior>0) {
            int n0=0, anchor=-1, sizes[4];
            for (int i=0;i<4;++i) {
                sizes[i]=c[i].size();
                n0+=sizes[i];
                if (c[i].has(0)) anchor=i;
            }
            if (n0==taxa && anchor>=0) {
                double var=0;
                double mu=(entry.sum[0]+entry.sum[1]+entry.sum[2])/3;
                for (Votes v:entry.contributions) {
                    double m=(v[0]+v[1]+v[2])/3;
                    for (int i=0;i<3;++i) {
                        double x=v[i]-m-(entry.sum[i]-mu)/entry.contributions.size();
                        var+=x*x;
                    }
                }
                float strength=shape_prior*std::sqrt(var/3);
                float dist[3], mean=0;
                for (int i=0;i<3;++i) {
                    int s=sizes[0]+sizes[i+1];
                    if (anchor==0||anchor==i+1) s=n0-s;
                    dist[i]=-std::log(float(s-1));
                    mean+=dist[i]/3;
                }
                for (int i=0;i<3;++i) {
                    float p=(dist[i]-mean)*strength;
                    entry.sum[i]+=p;
                    for (auto &v:entry.contributions) v[i]+=p/entry.contributions.size();
                }
            }
        }
        return entry;
    }
public:
    float adjustment=0;
    float clipping=0;
    float extra_weight=0;
    float shape_prior=0;
    int taxa=0;
    std::function<std::vector<Votes>(const Quad &)> extra;
    std::optional<CalibrationMix> calibration;
    void add(LocusQuartets &q,float w) { models.push_back({&q,w}); }
    std::vector<Votes> observations(const Quad &q) {
        OrderQuad p(q);
        auto data=load(p.canonical).contributions;
        for (auto &v:data) v=p.unorder(v);
        return data;
    }
    Votes operator()(const Quad &q) {
        OrderQuad p(q);
        return p.unorder(load(p.canonical).sum);
    }
    float confidence(const Quad &q) {
        OrderQuad p(q);
        const auto &e=load(p.canonical);
        auto v=p.unorder(e.sum);
        double xy=0, xz=0;
        for (Votes u:e.contributions) {
            u=p.unorder(u);
            double y=u[0]-u[1], z=u[0]-u[2];
            xy+=y*y; xz+=z*z;
        }
        double y=v[0]-v[1], z=v[0]-v[2];
        double n=e.contributions.size();
        return std::min(y/std::sqrt(std::max(1e-24,xy-y*y/n)),
                        z/std::sqrt(std::max(1e-24,xz-z*z/n)));
    }
};

struct MixtureOptions {
    bool tighter_kernel=false;
    int max_ports=8;
    float threshold=2.3f;
    float group_size=0.6f;
    int relocate=0;
    bool resample=false;
    float resample_sd=0.74f;
    float central_mass=5.0f;
    float projected=0;
    float balance=0.52f;
    float localization=0;
    bool balance_primary=false;
    float dispersion=0;
    float clipping=0;
    float local_quality=0;
    float rooted_mix=0;
    float rooted_decay=1.65f;
    float shape_prior=0;
    float cross_validation=0;
};

inline Tree mixture_inference(const GeneSet &set, const std::vector<GeneDistances> &d,
                              const Tree &start, std::mt19937 &rng, MixtureOptions opt) {
    QuartetOptions q;
    q.decay=1.5f; q.length_mix=0.65f; q.gene_power=1;
    QuartetKernel baseline(set,d,q);
    LocusQuartets primary(baseline);
    Tree guide=start;
    nni_sweep(guide,primary,rng);
    auto w=calibration_weights(guide,primary,d,0.75f,0.75f);
    QuartetOptions a=q;
    a.decay=2.0f;
    a.tip_factor=0.52f;
    a.adaptive_scale=0.025f;
    a.internal_scale=0.0135f;
    a.length_mix=0.74f;
    a.upper_scale=1.2f;
    QuartetOptions c=q;
    c.decay=1.1f;
    c.length_mix=0.5f;
    c.per_gene_norm=0.55f;
    c.norm_floor=0.07f;
    if (opt.tighter_kernel) {
        a.decay=1.35f; a.tip_factor=0.71f;
        a.internal_scale=0.0035f; a.adaptive_scale=0.018f;
        c.length_mix=0.38f; c.support_power=1.25f; c.decay=1.35f;
    }
    std::vector<std::vector<float>> leaf_quality;
    if (opt.local_quality>0) leaf_quality=ProfileQuality(d).leaves(1-opt.local_quality,0.84f);
    QuartetKernel ka(set,d,a,w,leaf_quality), kc(set,d,c);
    LocusQuartets sa(ka), sc(kc), sw(baseline);
    Matrix loc;
    if (opt.localization>0) loc=aggregate(d,Metric::Length,1,1,0,0.12f);
    GuideWeights bw(guide,opt.balance,loc.empty()?nullptr:&loc,opt.localization);
    if (opt.balance_primary) sw.leaf_weights=std::ref(bw);
    sa.leaf_weights=std::ref(bw);
    sc.leaf_weights=std::ref(bw);
    QuartetMixture score;
    score.shape_prior=opt.shape_prior;
    score.taxa=set.n;
    std::optional<RootedTriplets> rooted;
    if (opt.rooted_mix>0) {
        rooted.emplace(set,d,opt.rooted_decay);
        score.extra=std::ref(*rooted);
        score.extra_weight=opt.rooted_mix;
    }
    score.adjustment=opt.dispersion;
    score.clipping=opt.clipping;
    auto &first=opt.balance_primary?sw:primary;
    MixWeights prior{0.48f,0.28f,0.24f};
    if (opt.cross_validation>0 && d.size()>=10) {
        std::mt19937 calibration_rng=rng;
        score.calibration=tune_mixture(guide,primary,{&first,&sa,&sc},prior,d.size(),
                                       calibration_rng,opt.cross_validation);
    }
    score.add(first,prior[0]);
    score.add(sa,prior[1]);
    score.add(sc,prior[2]);
    auto conf=[&](const Quad &u,Mask) {return score.confidence(u);};
    Tree t=guide;
    nni_sweep(t,score,rng,8);
    if (opt.relocate)
        limited_spr(t,score,conf,opt.relocate,3,opt.threshold+0.25f,0.28f);
    if (opt.resample) {
        std::optional<ProjectedSplits> sp;
        if (opt.projected>0) sp.emplace(set,d);
        local_posterior(t,score,rng,d.size(),opt.threshold,opt.max_ports,
                        opt.group_size,opt.resample_sd,opt.central_mass,
                        sp?&*sp:nullptr,opt.projected);
    } else {
        refine_blocks(t,score,conf,opt.threshold,opt.max_ports,opt.group_size,0.25f);
    }
    return t;
}
