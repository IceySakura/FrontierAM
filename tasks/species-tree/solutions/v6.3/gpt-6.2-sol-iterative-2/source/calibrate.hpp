#pragma once

#include "quartet.hpp"

using MixWeights = std::array<float,3>;

struct CalibrationMix {
    // Use a small model for quartets near tips and one for broader sampled clades.
    std::array<MixWeights,3> table;

    static int category(const Quad &q) {
        std::array<int,4> sz;
        for (int i=0;i<4;++i) sz[i]=q[i].size();
        std::sort(sz.begin(),sz.end());
        return sz[1]<=2?0:(sz[1]<=7?1:2);
    }
};

inline CalibrationMix tune_mixture(const Tree &t,LocusQuartets &reference,
                                   const std::array<LocusQuartets *,3> &samples,
                                   MixWeights prior,int ngenes,std::mt19937 &rng,float strength) {
    constexpr int trials=16;
    int k=std::clamp(ngenes/5,2,ngenes-1);
    std::array<std::vector<int>,trials> held;
    std::vector<int> gene(ngenes);
    std::iota(gene.begin(),gene.end(),0);
    for (auto &f:held) {
        std::shuffle(gene.begin(),gene.end(),rng);
        f.assign(gene.begin(),gene.begin()+k);
    }
    std::array<std::array<double,3>,3> losses{};
    std::array<float,3> rate{};
    MaskTree mt(t);
    for (auto [u,v]:mt.internal) {
        Quad edge=edge_groups(t,u,v);
        float z=reference.zsupport(edge);
        if (z<3.0f) continue;
        OrderQuad o(edge);
        const auto &full=reference.entry(o.canonical);
        int cat=CalibrationMix::category(edge);
        int correct=std::find(o.mapping.begin(),o.mapping.end(),0)-o.mapping.begin();
        auto d=full.sum;
        double weight=1/(1+0.02*(z-3)*(z-3));
        std::array<double,3> b{};
        for (int i=0;i<3;++i) {
            const auto &s=samples[i]->entry(o.canonical);
            double sum=0;
            for (auto x:s.part) {
                double ave=(x[0]+x[1]+x[2])/3.0;
                for (int c=0;c<3;++c) sum+=(x[c]-ave)*(x[c]-ave);
            }
            b[i]=std::sqrt(sum*k/ngenes);
        }
        for (const auto &f:held) {
            auto remaining=d;
            for (int i:f) for (int j=0;j<3;++j) remaining[j]-=full.part[i][j];
            bool valid=true;
            for (int j=0;j<3;++j) {
                if (j!=correct && remaining[correct]<1.12f*remaining[j]) valid=false;
            }
            if (!valid) continue;
            for (int i=0;i<3;++i) {
                const auto &e=samples[i]->entry(o.canonical);
                Votes hold{};
                if (b[i]>1e-18) {
                    for (int j:f) for (int c=0;c<3;++c) hold[c]+=e.part[j][c];
                }
                float other=-1e20f;
                for (int j=0;j<3;++j) if (j!=correct) other=std::max(other,hold[j]);
                float zscore=(hold[correct]-other)/std::max(1e-18,b[i]);
                losses[cat][i]+=weight/(1+std::exp(2.2f*zscore));
            }
            rate[cat]+=weight;
        }
    }
    CalibrationMix cal;
    const double total=rate[0]+rate[1]+rate[2];
    for (int c=0;c<3;++c) {
        float sum=0;
        for (int i=0;i<3;++i) {
            double all_loss=losses[0][i]+losses[1][i]+losses[2][i];
            double adjust=0.38*(losses[c][0]-losses[c][i])/(rate[c]+80) +
                          0.62*((losses[0][0]+losses[1][0]+losses[2][0])-all_loss)/(total+60);
            cal.table[c][i]=prior[i]*std::exp(strength*std::clamp(adjust,-0.13,0.13));
            sum+=cal.table[c][i];
        }
        for (float &w:cal.table[c]) w/=sum;
    }
    return cal;
}
