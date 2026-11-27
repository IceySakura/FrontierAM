#pragma once

#include "refine.hpp"

class GuideWeights {
    Tree reference;
    MaskTree groups;
    float power;
    const Matrix *distance;
    float localization;
    std::unordered_map<Mask,std::pair<int,int>,MaskHash> sides;
    using List = std::vector<std::pair<int,float>>;
    std::unordered_map<Mask,List,MaskHash> lists;

    void distribute(List &out, int u, int p, float count, float mass) const {
        if (u < reference.n) { out.emplace_back(u,mass); return; }
        auto nn = other_neighbors(reference,u,p);
        float a = groups.side(u,nn[0]).size();
        float b = count - a;
        float w = std::pow(a,power)/(std::pow(a,power)+std::pow(b,power));
        distribute(out,nn[0],u,a,mass*w);
        distribute(out,nn[1],u,b,mass*(1-w));
    }

    void sample(List &out, Mask selected) const {
        auto hit = sides.find(selected);
        if (hit != sides.end()) {
            auto [u,p] = hit->second;
            int s = selected.size();
            distribute(out,u,p,s,s);
        } else {
            selected.each([&](int i) { out.emplace_back(i,1.0f); });
        }
        if (!distance || out.size()<=1) return;
        Mask others=groups.all^selected;
        int no=others.size();
        if (!no) return;
        std::vector<float> depth(out.size());
        for (size_t j=0;j<out.size();++j) {
            int i=out[j].first;
            float d=0;
            others.each([&](int k) { d+=(*distance)[i*reference.n+k]; });
            depth[j]=d/no;
        }
        float best=*std::min_element(depth.begin(),depth.end()), sum=0;
        for (size_t j=0;j<depth.size();++j) {
            auto &w=out[j].second;
            w*=std::exp(-localization*(depth[j]-best));
            sum+=w;
        }
        for (auto &v:out) v.second*=out.size()/sum;
    }
public:
    GuideWeights(const Tree &t, float p, const Matrix *d=nullptr, float loc=0)
        : reference(t), groups(reference), power(p), distance(d), localization(loc) {
        for (int u = 0; u < t.used; ++u)
            for (int j = 0; j < t.v[u].degree; ++j) {
                int v = t.v[u].edges[j].to;
                sides.try_emplace(groups.side(u,v),v,u);
            }
        lists.reserve(1024);
    }
    std::vector<float> operator()(const Quad &q) {
        std::vector<float> w(reference.n);
        for (const Mask &g : q) {
            auto [it, add] = lists.try_emplace(g);
            if (add) sample(it->second,g);
            for (auto [i,x] : it->second) w[i] = x;
        }
        return w;
    }
};

inline std::vector<float> calibration_weights(const Tree &t, LocusQuartets &pilot,
                                              const std::vector<GeneDistances> &d,
                                              float reg = 0.5f, float exponent = 0.7f) {
    std::vector<double> means(d.size()), variances(d.size());
    std::vector<float> w(d.size(),1);
    double mass = 0;
    MaskTree mt(t);
    for (auto [u,v] : mt.internal) {
        Quad q = edge_groups(t,u,v);
        float z = pilot.zsupport(q);
        if (z < 2.25f) continue;
        OrderQuad p(q);
        const auto &e = pilot.entry(p.canonical);
        double length = e.sum[0]+e.sum[1]+e.sum[2];
        if (length < 1e-6) continue;
        double imp = 1.0/(1.0+0.02*(z-2.25)*(z-2.25));
        double ref = 1.0f + length/d.size();
        for (size_t i = 0; i < d.size(); ++i) {
            auto a = p.unorder(e.part[i]);
            double x = (a[0]-0.5*(a[1]+a[2]))/ref;
            double y = (a[1]-a[2])/ref;
            means[i] += x*imp;
            variances[i] += y*y*imp;
        }
        mass += imp;
    }
    if (mass < 20) return w;
    double mean = std::accumulate(means.begin(),means.end(),0.0)/d.size();
    double var = std::accumulate(variances.begin(),variances.end(),0.0)/d.size();
    if (mean < 1e-10 || var < 1e-10) return w;
    for (size_t i = 0; i < w.size(); ++i) {
        double s = std::max(0.1,mean*reg+means[i]) / (mean * (1+reg));
        double v = (reg*var+variances[i]) / (var*(1+reg));
        w[i] = std::clamp(float(std::pow(s/v,exponent)),0.2f,3.0f);
    }
    return w;
}

struct ProfileQuality {
    std::vector<std::vector<float>> agreement;
    std::vector<float> typical;
    float median=0;
    explicit ProfileQuality(const std::vector<GeneDistances> &d, float topology_fraction=0.35f) {
        int n=std::lround(std::sqrt(d[0].lengths.size()));
        Matrix mean(n*n);
        std::vector<Matrix> centered;
        for (const auto &g:d) {
            auto &out=centered.emplace_back(n*n);
            std::vector<float> row(n);
            float glen=std::max(g.scale,0.012f);
            float gtop=g.mean_confident+0.5f;
            for (int i=0;i<n;++i) for (int j=0;j<i;++j) {
                int k=i*n+j;
                float x=(1-topology_fraction)*std::min(3.3f,g.lengths[k]/glen) +
                         topology_fraction*std::min(3.3f,g.confident[k]/gtop);
                out[k]=out[j*n+i]=x;
                row[i]+=x; row[j]+=x;
            }
            for (float &x:row) x/=n-1;
            float mid=std::accumulate(row.begin(),row.end(),0.0f)/n;
            for (int i=0;i<n;++i) for (int j=0;j<i;++j) {
                int k=i*n+j;
                float x=out[k]-row[i]-row[j]+mid;
                out[k]=out[j*n+i]=x;
                mean[k]+=x/d.size();
                mean[j*n+i]+=x/d.size();
            }
        }
        std::vector<float> all;
        all.reserve(n*d.size());
        float fak=1.0f/(d.size()-1);
        for (const Matrix &cur:centered) {
            auto &rho=agreement.emplace_back(n,0);
            for (int i=0;i<n;++i) {
                double mc=0, cg=0, cross=0;
                for (int j=0;j<n;++j) if (i!=j&&j!=0) {
                    int k=i*n+j;
                    float a=cur[k], b=(mean[k]*d.size()-a)*fak;
                    cross+=a*b; cg+=a*a; mc+=b*b;
                }
                double corr=cross/std::sqrt((0.07+mc)*(0.07+cg));
                rho[i]=std::clamp(float(corr),-1.0f,1.0f);
                if (i>0) all.push_back(rho[i]);
            }
            typical.push_back(std::accumulate(rho.begin()+1,rho.end(),0.0f)/(n-1));
        }
        auto m=all.begin()+all.size()/2;
        std::nth_element(all.begin(),m,all.end());
        median=std::clamp(*m,0.15f,1.0f);
    }

    std::vector<std::vector<float>> leaves(float floor, float level, float exp=1) const {
        auto w=agreement;
        for (auto &q:w) for (float &x:q)
            x=floor+(1-floor)*std::pow(std::clamp(x/(level*median),0.0f,1.0f),exp);
        return w;
    }

    std::vector<float> genes(float floor, float level, float exp=1) const {
        auto w=typical;
        for (float &x:w)
            x=floor+(1-floor)*std::pow(std::clamp(x/(level*median),0.0f,1.0f),exp);
        return w;
    }
};
