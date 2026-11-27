#pragma once

#include "mixture.hpp"

inline MixtureOptions base_options() {
    MixtureOptions opt;
    opt.max_ports=9;
    opt.relocate=28;
    opt.local_quality=0.44f;
    opt.shape_prior=0.065f;
    opt.localization=1.0f;
    opt.clipping=0.20f;
    opt.balance=0.47f;
    return opt;
}

inline Tree combined_inference(const GeneSet &set,const std::vector<GeneDistances> &d,
                               const Tree &start,std::mt19937 &rng) {
    auto a=base_options();
    auto seed=rng;
    Tree t=mixture_inference(set,d,start,rng,a);
    SplitPool pool;
    pool.record(t,0.39f);
    auto b=a;
    b.relocate=48;
    b.threshold=2.65f;
    b.cross_validation=14;
    b.group_size=0.62f;
    auto r1=seed;
    pool.record(mixture_inference(set,d,start,r1,b),0.21f);

    auto c=a;
    c.balance_primary=true;
    c.tighter_kernel=true;
    c.localization=1.3f;
    c.dispersion=0.22f;
    c.clipping=0.35f;
    c.threshold=2.55f;
    c.relocate=36;
    auto r2=seed;
    pool.record(mixture_inference(set,d,t,r2,c),0.19f);

    auto e=a;
    e.resample=true;
    e.projected=0.48f;
    e.resample_sd=0.65f;
    e.central_mass=7.5f;
    e.threshold=2.6f;
    e.rooted_mix=0.16f;
    e.rooted_decay=1.9f;
    auto r3=seed;
    pool.record(mixture_inference(set,d,t,r3,e),0.21f);
    return pool.build(set.n);
}
