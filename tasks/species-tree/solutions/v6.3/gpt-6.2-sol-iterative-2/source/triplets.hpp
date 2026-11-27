#pragma once

#include "clock.hpp"

class RootedTriplets {
    const int n;
    struct Joint { int l, r; float fl, fr, early; };
    struct Gene { std::vector<Joint> nodes; float weight; };
    std::vector<Gene> genes;
    float decay;

    float fail(Arc e,float scale) const {
        return 1-0.5f*reliability(e)+0.5f*std::expm1(-e.length/(0.005f+0.032f*scale));
    }
    int pack(const Tree &t, Gene &g, const ClockProjection &p,int u,int parent,float scale) {
        if (u<n) return u;
        Arc a,b;
        bool first=true;
        for (Arc e:t.v[u].edges) if (e.to!=parent) {
            if (first) a=e; else b=e;
            first=false;
        }
        int l=pack(t,g,p,a.to,u,scale);
        int r=pack(t,g,p,b.to,u,scale);
        int x=n+g.nodes.size();
        g.nodes.push_back({l,r,fail(a,scale),fail(b,scale),std::exp(-decay*2*p.age(u)/(scale+0.018f))});
        return x;
    }
public:
    RootedTriplets(const GeneSet &set,const std::vector<GeneDistances> &d,
                   float early_decay=1.7f,float outgroup_weight=11)
        : n(set.n), decay(early_decay) {
        for (size_t z=0;z<set.genes.size();++z) {
            const auto &t=set.genes[z];
            ClockProjection p(t,outgroup_weight,0.016f,0.4f);
            p.fit(0.17f,0.30f);
            auto [x,y]=p.root_edge();
            auto &g=genes.emplace_back();
            g.nodes.reserve(n-1);
            float scale=d[z].scale;
            g.weight=scale/(scale+0.035f);
            int a=pack(t,g,p,x,y,scale), b=pack(t,g,p,y,x,scale);
            auto e=t.arc(x,y);
            float f=fail(e,scale);
            float frac=e.length>0?p.root_piece()/e.length:0.5f;
            g.nodes.push_back({a,b,std::pow(f,frac),std::pow(f,1-frac),
                               std::exp(-decay*2*p.age_root()/(scale+0.018f))});
        }
    }

    std::vector<Votes> operator()(const Quad &q) const {
        int idx[3], c=0;
        bool rooted=false;
        for (int j=0;j<4;++j) {
            if (q[j].has(0)) rooted=true;
            else if (c<3) idx[c++]=j;
        }
        if (!rooted) return {};
        constexpr int order[3][3]={{0,1,2},{0,2,1},{1,2,0}};
        Votes counts[2*QuartetKernel::MAXN]{}, pairs[2*QuartetKernel::MAXN]{},
              weak[2*QuartetKernel::MAXN]{};
        for (int i=0;i<3;++i)
            q[idx[i]].each([&](int j) {counts[j][i]=1;});
        std::vector<Votes> out;
        out.reserve(genes.size());
        for (const auto &g:genes) {
            Votes sum{};
            int v=n;
            for (Joint u:g.nodes) {
                const auto &a=counts[u.l], &b=counts[u.r];
                for (int i=0;i<3;++i) counts[v][i]=a[i]+b[i];
                for (int i=0;i<3;++i) {
                    int x=order[i][0], y=order[i][1], z=order[i][2];
                    float pl=pairs[u.l][i], pr=pairs[u.r][i];
                    float fl=weak[u.l][i]*u.fl, fr=weak[u.r][i]*u.fr;
                    float first=(a[x]*b[y]+a[y]*b[x])*u.early;
                    sum[i]+=(pl-fl)*b[z]+(pr-fr)*a[z];
                    pairs[v][i]=first+pl+pr;
                    weak[v][i]=first+fl+fr;
                }
                ++v;
            }
            Votes t{};
            for (int i=0;i<3;++i) t[(idx[order[i][0]]^idx[order[i][1]])-1]=g.weight*sum[i];
            out.push_back(t);
        }
        return out;
    }
};
