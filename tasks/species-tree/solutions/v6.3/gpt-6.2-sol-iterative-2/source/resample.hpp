#pragma once

#include "refine.hpp"
#include "projected.hpp"

class RegionPosterior {
    static constexpr int draws=24;
    using Weights = std::array<float,draws>;
    struct Resolution {
        std::array<int,4> port;
        int mask;
        std::array<Weights,3> weight;
    };
    int k, m;
    std::vector<Resolution> votes;
    std::vector<Weights> best;
    std::vector<std::array<uint16_t,draws>> choice;
    std::vector<float> counts;
    std::vector<Weights> extra;

    void count(int s, int i, float w) {
        if (!(s & (s-1))) return;
        counts[s]+=w;
        int a=choice[s][i];
        count(a,i,w);
        count(s^a,i,w);
    }
public:
    explicit RegionPosterior(int ports)
        : k(ports),m(1<<(ports-1)),best(m),choice(m),counts(m) {}

    void add(int a,int b,int c,int d,std::array<Weights,3> x) {
        votes.push_back({{1<<a,1<<b,1<<c,1<<d},(1<<a)|(1<<b)|(1<<c)|(1<<d),x});
    }

    void split_prior(std::vector<Weights> projected, float mix) {
        extra=std::move(projected);
        Weights mean{};
        for (const auto &q:votes) {
            for (int r=0;r<draws;++r) {
                mean[r]+=std::max({q.weight[0][r],q.weight[1][r],q.weight[2][r]})-
                         std::min({q.weight[0][r],q.weight[1][r],q.weight[2][r]});
            }
        }
        for (auto &v:extra)
            for (int r=0;r<draws;++r) v[r]*=mix*mean[r]/(k-3);
    }

    std::vector<int> optimize(float central_mass, float best_mixture) {
        const int all=(m<<1)-1;
        for (int s=3;s<m;++s) {
            int low=s&-s;
            if (low==s) continue;
            int c=all^s;
            best[s].fill(-std::numeric_limits<float>::infinity());
            for (int a=(s-1)&s;a;a=(a-1)&s) {
                if (!(a&low)) continue;
                int b=s^a;
                Weights x{};
                for (int i=0;i<draws;++i) x[i]=best[a][i]+best[b][i];
                for (const auto &q:votes) {
                    int u=a&q.mask,v=b&q.mask,w=c&q.mask;
                    if (!u||!v||!w) continue;
                    int z=u&(u-1)?u:(v&(v-1)?v:w);
                    if (!(z&q.port[0])) z^=q.mask;
                    int pick=(z&q.port[1])?0:((z&q.port[2])?1:2);
                    for (int i=0;i<draws;++i) x[i]+=q.weight[pick][i];
                }
                for (int i=0;i<draws;++i) if (x[i]>best[s][i]) {
                    best[s][i]=x[i];
                    choice[s][i]=a;
                }
            }
            if (!extra.empty() && s!=m-1)
                for (int i=0;i<draws;++i) best[s][i]+=extra[s][i];
        }
        for (int i=0;i<draws;++i) count(m-1,i,i==0?central_mass:1);

        std::vector<float> cred(counts.size());
        std::vector<int> resolve(counts.size());
        for (int s=3;s<m;++s) {
            int low=s&-s;
            if (low==s) continue;
            float best_x=-1e15f;
            for (int a=(s-1)&s;a;a=(a-1)&s) if (a&low) {
                float x=cred[a]+cred[s^a];
                if (a==choice[s][0]) x+=best_mixture;
                if (x>best_x) { best_x=x; resolve[s]=a; }
            }
            cred[s]=best_x+counts[s];
        }
        return resolve;
    }
};

template<class Score>
inline void resample_region(Tree &t, const std::vector<int> &inner, Score &score,
                            const std::array<std::vector<float>,24> &weights,
                            float group_power, float central_mass,
                            ProjectedSplits *projected, float prior_mix, std::mt19937 &rng) {
    if (inner.size()<2) return;
    std::vector<bool> use(t.used,false);
    for (int u:inner) use[u]=true;
    std::vector<int> ports;
    std::vector<Mask> group;
    std::vector<float> sizes;
    for (int u:inner) for (Arc e:t.v[u].edges) if (!use[e.to]) {
        ports.push_back(e.to);
        group.push_back(walk_mask(t,e.to,u));
        sizes.push_back(group.back().size());
    }
    int k=ports.size();
    RegionPosterior dp(k);
    for (int a=0;a<k;++a) for (int b=a+1;b<k;++b)
        for (int c=b+1;c<k;++c) for (int d=c+1;d<k;++d) {
            Quad q{group[a],group[b],group[c],group[d]};
            auto obs=score.observations(q);
            float size=sizes[a]*sizes[b]*sizes[c]*sizes[d];
            float ms=0;
            for (auto x:obs) ms+=x[0]+x[1]+x[2];
            float factor=1/(std::pow(size,group_power)*std::pow(0.1f+ms/size,0.25f));
            std::array<std::array<float,24>,3> v{};
            for (size_t i=0;i<obs.size();++i) {
                auto x=obs[i];
                float mean=(x[0]+x[1]+x[2])/3;
                for (int j=0;j<3;++j) {
                    float s=(x[j]-mean)*factor;
                    for (int r=0;r<24;++r) v[j][r]+=s*weights[r][i];
                }
            }
            dp.add(a,b,c,d,v);
        }
    if (projected && k>4) dp.split_prior(projected->score(group,weights,rng),prior_mix);
    auto z=dp.optimize(central_mass,0.5f);
    for (int u:inner) while (t.v[u].degree) t.unlink(u,t.v[u].edges[0].to);
    size_t next=0;
    std::function<int(int)> add=[&](int s) {
        if (!(s&(s-1))) return ports[__builtin_ctz(unsigned(s))];
        int u=inner[next++], a=z[s];
        int l=add(a), r=add(s^a);
        t.link(u,l); t.link(u,r);
        return u;
    };
    t.link(add((1<<(k-1))-1),ports.back());
}

template<class Score>
inline void local_posterior(Tree &t, Score &score, std::mt19937 &rng, int genes,
                            float threshold=2.35f,int ports=8,float group_power=0.6f,
                            float sd=0.78f,float central_mass=4,
                            ProjectedSplits *projected=nullptr,float prior_mix=0) {
    MaskTree mt(t);
    std::vector<std::tuple<float,int,int>> ranked;
    for (auto [u,v]:mt.internal) {
        float z=score.confidence(edge_groups(t,u,v));
        if (z<threshold) ranked.emplace_back(z,u,v);
    }
    std::sort(ranked.begin(),ranked.end());
    Components cc(t.used);
    for (auto [z,u,v]:ranked) cc.join(u,v,ports-2);
    std::vector<std::vector<int>> by_root(t.used);
    for (int u=t.n;u<t.used;++u) by_root[cc.root(u)].push_back(u);
    std::array<std::vector<float>,24> w;
    w[0].assign(genes,1);
    for (int i=1;i<24;++i) w[i]=bootstrap_weights(genes,rng,sd);
    for (auto &c:by_root)
        resample_region(t,c,score,w,group_power,central_mass,projected,prior_mix,rng);
}
