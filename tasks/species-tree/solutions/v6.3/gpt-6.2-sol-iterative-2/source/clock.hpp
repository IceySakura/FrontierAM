#pragma once

#include "quartet.hpp"

class ClockProjection {
    struct VertexSummary {
        double w = 0, sum = 0, square = 0;
        double prec = 0, height = 0;
    };
    const Tree &tree;
    std::vector<VertexSummary> stats;
    std::vector<int> order, parent;
    std::vector<double> length;
    std::vector<float> fitted;
    float tip_weight;
    float noise_floor;
    float power;
    double root_age = 0;
    float fitted_root = 0;
    int best_u = -1, best_v = -1;
    double best_pos = 0;

    void visit(int u, int p, float len) {
        parent[u] = p;
        length[u] = len;
        order.push_back(u);
        auto &a = stats[u];
        if (u < tree.n) a.w = u == 0 ? tip_weight : 1;
        else for (auto e : tree.v[u].edges) if (e.to != p) {
            visit(e.to,u,e.length);
            const auto &b = stats[e.to];
            double l = e.length;
            a.w += b.w;
            a.sum += b.sum + l*b.w;
            a.square += b.square + 2*l*b.sum + l*l*b.w;
        }
    }

    // Variance of root-to-tip distances is quadratic along each unrooted edge.
    void find_root(float anchor_bonus) {
        double best = std::numeric_limits<double>::infinity();
        std::vector<double> from(tree.used), second(tree.used);
        int r = order.front();
        double w = stats[r].w;
        from[r] = stats[r].sum;
        second[r] = stats[r].square;
        for (int u : order) if (u != r) {
            int p = parent[u];
            double wp = w-2*stats[u].w;
            double cross = from[p] - 2*(stats[u].sum+length[u]*stats[u].w);
            double curv = w-wp*wp/w;
            double slope = cross-from[p]*wp/w;
            double x = std::clamp(-slope/curv,0.0,length[u]);
            double s = from[p]+wp*x, q = second[p]+2*cross*x+w*x*x;
            double value = q-s*s/w;
            if (u == 0 || p == 0) value *= 1-anchor_bonus;
            if (value < best) {
                best_u = p; best_v = u; best_pos = x; best = value;
                root_age = s/w;
            }
            from[u] = from[p]+wp*length[u];
            second[u] = second[p]+2*cross*length[u]+w*length[u]*length[u];
        }
    }

    double prec(double len) const { return std::pow(noise_floor+len,-power); }

    std::pair<double,double> fit_subtree(int u, int p) {
        VertexSummary &a = stats[u];
        if (u < tree.n)
            return {a.prec = 1e10, a.height = 0};
        a.height = a.prec = 0;
        for (auto e : tree.v[u].edges) if (e.to != p) {
            auto [p0,h0] = fit_subtree(e.to,u);
            double w = prec(e.length), m = p0*w/(p0+w);
            a.prec += m;
            a.height += m*(h0+e.length);
        }
        a.height /= a.prec;
        return {a.prec,a.height};
    }

    void assign_times(int u, int p, double hp, double len) {
        if (u < tree.n) return;
        auto &a = stats[u];
        double w = prec(len);
        // Condition the local estimate on the height of its parent.
        double h = std::clamp((a.prec*a.height+w*(hp-len))/(a.prec+w),0.0,hp);
        fitted[u] = h;
        for (auto e : tree.v[u].edges) if (e.to != p)
            assign_times(e.to,u,h,e.length);
    }
public:
    ClockProjection(const Tree &t, float outgroup_weight, float floor, float weight_power)
        : tree(t), stats(t.used), parent(t.used), length(t.used), fitted(t.used),
          tip_weight(outgroup_weight), noise_floor(floor), power(weight_power) {}

    void fit(float anchor_bonus, float root_prior) {
        visit(tree.v[0].edges[0].to,-1,0);
        find_root(anchor_bonus);
        double l = best_pos, r = length[best_v]-best_pos;
        auto [pl,hl] = fit_subtree(best_u,best_v);
        auto [pr,hr] = fit_subtree(best_v,best_u);
        double a = prec(l)*pl/(prec(l)+pl), b = prec(r)*pr/(prec(r)+pr);
        double h = (1-root_prior)*(a*(hl+l)+b*(hr+r))/(a+b) + root_prior*root_age;
        fitted_root=h;
        assign_times(best_u,best_v,h,l);
        assign_times(best_v,best_u,h,r);
    }
    std::pair<int,int> root_edge() const { return {best_u,best_v}; }
    float age(int u) const { return fitted[u]; }
    float age_root() const { return fitted_root; }
    float root_piece() const { return best_pos; }
};
