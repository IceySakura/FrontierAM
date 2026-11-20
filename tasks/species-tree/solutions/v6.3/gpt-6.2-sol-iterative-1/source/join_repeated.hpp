#pragma once
#include "inference.hpp"

namespace st {

struct RepeatedPaths {
    std::vector<std::vector<float>> value;
    std::vector<double> weight;
    int n;

    RepeatedPaths(const Input& in,const LocusEvidence& le,DistConfig c,double qual=.31)
        : weight(le.weights(qual)),n(in.n()) {
        std::vector<Matrix> d;
        std::vector<double> scale;
        for (const Tree& g : in.genes) {
            d.push_back(gene_distances(g,c.mode,c.edge_scale));
            scale.push_back(pair_scale(d.back(),in.out));
        }
        double med = median(scale);
        for (size_t g=0;g<d.size();++g) {
            double f=1/(med*std::pow(scale[g]/med,c.norm));
            std::vector<float> a(d[g].data.size());
            for (size_t i=0;i<a.size();++i) {
                double x=std::min(4.3,f*d[g].data[i]);
                if (c.power==0) x=std::log(x+c.shift);
                else if(c.power!=1) x=(std::pow(x+c.shift,c.power)-1)/c.power;
                a[i]=float(x);
            }
            value.push_back(std::move(a));
        }
    }
};

struct RepeatedJoinConfig {
    double variance=.65;
    double cherry=.22;
    double shrink=.15;
};

inline Tree repeated_join(Matrix d,const RepeatedPaths& data,const RepeatedJoinConfig& c) {
    const int n=d.n, k=int(data.value.size());
    Tree t(n);
    std::vector<int> alive(n), cluster(n);
    std::iota(alive.begin(),alive.end(),0);
    std::iota(cluster.begin(),cluster.end(),0);
    auto by_gene=data.value;
    std::vector<double> rows(n), residual(k*n), correction(n);
    Matrix cross(n);
    if (c.cherry>0) {
        for (int i=0;i<n;++i)
            for (int j=0;j<=i;++j) {
                double x=0;
                for (int a=0;a<n;++a) x+=d(i,a)*d(j,a);
                cross(i,j)=cross(j,i)=x;
            }
    }
    while (alive.size()>3) {
        int count=int(alive.size());
        for (int i:alive) {
            double x=0;
            for (int j:alive) x+=d(i,j);
            rows[i]=x/(count-2);
        }
        int a=-1,b=-1;
        double best=1e300;
        for (int i=0;i<count;++i) {
            int x=alive[i];
            for (int j=0;j<i;++j) {
                int y=alive[j];
                double sep=d(x,y), delta=rows[x]-rows[y];
                double z=sep-rows[x]-rows[y];
                if (c.cherry>0) {
                    double ss=cross(x,x)+cross(y,y)-2*cross(x,y)-2*sep*sep;
                    double variance=std::max(0.0,ss/(count-2)-delta*delta);
                    z+=c.cherry*std::sqrt(variance);
                }
                if (z<best) best=z,a=x,b=y;
            }
        }
        double alpha=.5;
        if (c.variance>0 || c.shrink>0) {
            double cross_cov=0, sum_sq=0;
            for (int g=0;g<k;++g) {
                const auto& m=by_gene[g];
                double av=0,bv=0;
                for (int j:alive) if(j!=a && j!=b) {
                    av+=m[size_t(a)*n+j]-d(a,j);
                    bv+=m[size_t(b)*n+j]-d(b,j);
                }
                av/=(count-2);
                bv/=(count-2);
                for (int j:alive) if(j!=a && j!=b) {
                    double x=std::clamp(m[size_t(a)*n+j]-d(a,j)-av,-.72,.72);
                    double y=std::clamp(m[size_t(b)*n+j]-d(b,j)-bv,-.72,.72);
                    double v=x-y;
                    cross_cov-=data.weight[g]*y*v;
                    sum_sq+=data.weight[g]*v*v;
                }
            }
            double est=std::clamp(cross_cov/std::max(1e-9,sum_sq),.06,.94);
            alpha=(1-c.variance)*.5+c.variance*est;
        }
        int u=t.add();
        t.link(u,cluster[a]);
        t.link(u,cluster[b]);
        cluster[a]=u;
        double constant=d(a,b)*.5+(alpha-.5)*(rows[a]-rows[b]);
        std::vector<double> line(n);
        for (int j:alive) if(j!=a && j!=b) line[j]=alpha*d(a,j)+(1-alpha)*d(b,j)-constant;
        if (c.variance>0 || c.shrink>0) {
            for (int g=0;g<k;++g) {
                auto& m=by_gene[g];
                double off=0;
                for (int i:alive) if(i!=a && i!=b) {
                    double x=alpha*m[size_t(a)*n+i]+(1-alpha)*m[size_t(b)*n+i]-constant;
                    m[size_t(a)*n+i]=m[size_t(i)*n+a]=float(x);
                    off+=x-line[i];
                }
                off/=(count-2);
                for (int i:alive) if(i!=a && i!=b)
                    residual[size_t(i)*k+g]=m[size_t(a)*n+i]-line[i]-off;
            }
        }
        if(c.shrink>0) {
            double mean_correction=0;
            const double total=std::accumulate(data.weight.begin(),data.weight.end(),0.);
            for(int i:alive) if(i!=a && i!=b) {
                double x=0,y=0;
                for (int g=0;g<k;++g) {
                    double v=residual[size_t(i)*k+g];
                    x+=data.weight[g]*v;
                    y+=data.weight[g]*std::clamp(v,-.28,.20);
                }
                correction[i]=std::clamp((y-.5*x)/total,-.12,.12);
                mean_correction+=correction[i];
            }
            mean_correction/=(count-2);
            for(int i:alive) if(i!=a && i!=b)
                line[i]+=c.shrink*(correction[i]-mean_correction);
        }
        if (c.cherry>0) {
            for (int i:alive) if(i!=a && i!=b)
                for (int j:alive) if(j<=i && j!=a && j!=b)
                    cross(i,j)=cross(j,i)=cross(i,j)-d(i,a)*d(j,a)-d(i,b)*d(j,b)+line[i]*line[j];
            double ss=0;
            for (int i:alive) if(i!=a && i!=b) {
                double val=0;
                for(int j:alive) if(j!=a && j!=b) val+=line[j]*d(i,j);
                cross(i,a)=cross(a,i)=val;
                ss+=line[i]*line[i];
            }
            cross(a,a)=ss;
        }
        for (int i:alive) if(i!=a && i!=b) d(a,i)=d(i,a)=line[i];
        alive.erase(std::find(alive.begin(),alive.end(),b));
    }
    int u=t.add();
    for(int i:alive) t.link(u,cluster[i]);
    return t;
}

}  // namespace st
