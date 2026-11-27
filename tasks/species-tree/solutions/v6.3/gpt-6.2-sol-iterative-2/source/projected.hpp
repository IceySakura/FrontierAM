#pragma once

#include "quartet.hpp"

class ProjectedSplits {
    struct Joint { int l, r; float fail; };
    std::vector<std::vector<Joint>> genes;
    std::vector<float> gene_mass;
    int n;

    static int pack(const Tree &t,int u,int p,std::vector<Joint> &dst,float fail,float scale) {
        if (u<t.n) return u;
        Arc l,r;
        bool first=true;
        for (Arc e:t.v[u].edges) if (e.to!=p) {
            if (first) l=e; else r=e;
            first=false;
        }
        auto f=[&](Arc e) {
            return 1-0.6f*reliability(e)+0.4f*std::expm1(-e.length/(0.0075f+0.025f*scale));
        };
        int a=pack(t,l.to,u,dst,f(l),scale), b=pack(t,r.to,u,dst,f(r),scale);
        int idx=t.n+dst.size();
        dst.push_back({a,b,fail});
        return idx;
    }
    static double null_split(int n,int b) {
        double choose=1;
        for (int i=1;i<=b;++i) choose=choose*(n+1-i)/i;
        return (2.0*n/(b*(b+1)) + 2.0*n/((n-b)*(n-b+1)) - 2.0/(n-1))/choose;
    }
public:
    ProjectedSplits(const GeneSet &set,const std::vector<GeneDistances> &d)
        : n(set.n) {
        for (size_t g=0;g<set.genes.size();++g) {
            const auto &t=set.genes[g];
            auto &dst=genes.emplace_back();
            dst.reserve(t.n-2);
            pack(t,t.v[0].edges[0].to,0,dst,1,d[g].scale);
            gene_mass.push_back(d[g].scale/(d[g].scale+0.055f));
        }
    }

    std::vector<std::array<float,24>> score(const std::vector<Mask> &groups,
                                           const std::array<std::vector<float>,24> &w,
                                           std::mt19937 &rng,int reps=16) const {
        int k=groups.size(), half=1<<(k-1), all=2*half-1;
        std::vector<std::vector<int>> tips(k);
        for (int j=0;j<k;++j) groups[j].each([&](int i){ tips[j].push_back(i); });
        std::vector<std::array<float,24>> votes(half);
        if (k<=4) return votes;
        std::vector<float> mean(half), failure(half), star(half);
        std::vector<int> flags(2*n), chosen(k);
        std::vector<uint8_t> valid(2*half);
        for (int a=0;a<=all;++a) {
            int b=__builtin_popcount(unsigned(a));
            valid[a]=b>=2&&b<=k-2;
            if (valid[a]&&a<half) star[a]=null_split(k,b);
        }
        std::array<float,24> total{};
        for (size_t g=0;g<genes.size();++g) {
            std::fill(mean.begin(),mean.end(),0.0f);
            int active_reps=1;
            for (auto &a:tips) if (a.size()>1) { active_reps=reps; break; }
            double weight=gene_mass[g]/active_reps;
            for (int r=0;r<active_reps;++r) {
                for (int j=0;j<k;++j) {
                    chosen[j]=tips[j][rng()%tips[j].size()];
                    flags[chosen[j]]=1<<j;
                }
                std::fill(failure.begin(),failure.end(),1.0f);
                int u=n;
                for (const auto &a:genes[g]) {
                    int s=flags[a.l]|flags[a.r];
                    flags[u++]=s;
                    if (valid[s]) failure[s<half?s:(all^s)]*=a.fail;
                }
                for (int a=1;a<half;++a) mean[a]+=weight*(1-failure[a]);
                for (int v:chosen) flags[v]=0;
            }
            for (int r=0;r<24;++r) total[r]+=gene_mass[g]*w[r][g];
            for (int a=1;a<half;++a) if (mean[a]>0) {
                for (int r=0;r<24;++r) votes[a][r]+=mean[a]*w[r][g];
            }
        }
        for (int a=1;a<half;++a) if (valid[a]) {
            for (int r=0;r<24;++r)
                votes[a][r]=votes[a][r]/std::max(1e-8f,total[r])-0.45f*star[a];
        }
        return votes;
    }
};
