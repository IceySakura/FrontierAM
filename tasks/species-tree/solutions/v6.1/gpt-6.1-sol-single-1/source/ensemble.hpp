#ifndef SPECIES_ENSEMBLE_HPP
#define SPECIES_ENSEMBLE_HPP

#include <memory>

struct InferenceClock {
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    double limit = 100;
    double elapsed() const {
        return chrono::duration<double>(chrono::steady_clock::now()-start).count();
    }
    bool expired(double reserve = 0) const { return elapsed() >= limit-reserve; }
};

struct CandidateTrees {
    vector<Tree> trees;
    unique_ptr<Quartets> evidence;
    Tree fallback;
    bool complete = false;
};

int bounded_nni(Tree& t, const Quartets& q, mt19937_64& rng, const InferenceClock& clock) {
    int rounds = 0;
    while (rounds < 30 && !clock.expired() && nni_pass(t,q,rng)) ++rounds;
    return rounds;
}

void bounded_local_search(Tree& t, const Quartets& q, mt19937_64& rng,
                          const InferenceClock& clock, int rounds = 8) {
    bounded_nni(t,q,rng,clock);
    for (int r = 0; r < rounds && !clock.expired(); ++r) {
        int moves = spr_pass(t,q,rng,t.n/2);
        int nn = bounded_nni(t,q,rng,clock);
        if (!moves && !nn) break;
    }
}

CandidateTrees infer_candidates(const Input& input, uint64_t seed, const InferenceClock& clock) {
    CandidateTrees result;
    map<pair<int,int>,Matrix> distance_cache;
    auto distances = [&](int metric, int norm) -> const Matrix& {
        pair<int,int> key = {metric,norm};
        auto it = distance_cache.find(key);
        if (it == distance_cache.end())
            it = distance_cache.emplace(key,average_distances(input.genes,metric,norm)).first;
        return it->second;
    };
    result.fallback = neighbor_joining(distances(9,0),input.names);
    for (auto [metric,norm] : vector<pair<int,int>>{{0,0},{2,0},{2,1},{3,1},{4,1},{5,1},
                                                  {6,1},{7,1},{8,1},{9,0},{9,1},{9,2},{10,1}}) {
        const Matrix& d = distances(metric,norm);
        Tree t = neighbor_joining(d,input.names);
        mt19937_64 rng(seed);
        for (int r = 0; r < 10 && bme_spr_pass(t,d,rng); ++r) {}
        result.trees.push_back(move(t));
    }
    for (int k = 0; k < 2; ++k) {
        Matrix d = robust_average_distances(input.genes,k == 0 ? 7 : 9,k == 0);
        Tree t = neighbor_joining(d,input.names);
        mt19937_64 rng(seed);
        for (int r = 0; r < 10 && bme_spr_pass(t,d,rng); ++r) {}
        result.trees.push_back(move(t));
    }
    mt19937_64 rng(seed);
    CladeSearch search(input);
    for (const Tree& t : input.genes) search.add_tree(t);
    vector<Tree> seeds;
    for (auto [metric,norm] : vector<pair<int,int>>{{9,0},{9,1},{2,0},{7,1},{0,0},{10,1}}) {
        const Matrix& d = distances(metric,norm);
        Tree t = neighbor_joining(d,input.names);
        search.add_tree(t);
        if (metric == 9 && norm == 0) seeds.push_back(t);
        for (int r = 0; r < 4 && bme_spr_pass(t,d,rng); ++r) {}
        search.add_tree(t);
        seeds.push_back(move(t));
    }
    result.evidence = make_unique<Quartets>(input,1,2);
    const Quartets& q = *result.evidence;
    Tree base = seeds[0];
    bounded_local_search(base,q,rng,clock);
    result.fallback = base;
    if (clock.expired(15)) return result;
    search.add_tree(base);
    result.trees.push_back(base);
    double best_score = q.score(base);
    Tree best = base;
    for (int k : {1,3,4}) {
        Tree t = seeds[k];
        bounded_local_search(t,q,rng,clock);
        search.add_tree(t);
        double value = q.score(t);
        if (value > best_score+1e-4) { best_score = value; best = move(t); }
        result.fallback = best;
        if (clock.expired(15)) return result;
    }
    result.trees.push_back(best);
    TripartitionScorer scorer(input,1,2);
    Tree dp = search.solve(scorer,[&] { return clock.expired(15); });
    if (!dp.n) return result;
    result.trees.push_back(dp);
    result.trees.push_back(search.bootstrap_consensus(rng,250,0.5));
    result.trees.push_back(search.bootstrap_consensus(rng,250,1));
    bounded_local_search(dp,q,rng,clock);
    double dp_score = q.score(dp);
    if (dp_score > best_score+1e-4) { best_score = dp_score; best = move(dp); }
    result.fallback = best;
    result.trees.push_back(best);
    if (clock.expired(15)) return result;
    {
        Quartets softened = q;
        softened.transform(3,2);
        Tree soft = base;
        bounded_local_search(soft,softened,rng,clock);
        result.trees.push_back(soft);
        double best_soft = softened.score(soft);
        for (int k : {1,3}) {
            Tree t = seeds[k];
            bounded_local_search(t,softened,rng,clock);
            double score = softened.score(t);
            if (score > best_soft+1e-4) { best_soft = score; soft = move(t); }
            if (clock.expired(12)) return result;
        }
        result.trees.push_back(move(soft));
    }
    if (clock.expired(12)) return result;
    {
        Quartets alternate(input,1,1);
        Tree t = neighbor_joining(distances(9,0),input.names);
        mt19937_64 alternate_rng(seed);
        bounded_local_search(t,alternate,alternate_rng,clock,10);
        result.trees.push_back(move(t));
    }
    if (clock.expired(4)) return result;
    for (Tree& t : result.trees) {
        string text = t.newick();
        auto labels = input.labels;
        t = NewickParser(text,labels,false).parse();
    }
    result.complete = result.trees.size() == 24;
    return result;
}

struct LocusQuartets {
    TripartitionScorer base;
    int n, nn;
    vector<array<int,4>> counts;
    vector<array<double,6>> down, up;
    static constexpr int px[6] = {0,0,0,1,1,2};
    static constexpr int py[6] = {1,2,3,2,3,3};
    explicit LocusQuartets(const Input& input) : base(input,1,2), n(input.names.size()), nn(2*n+4),
        counts(nn), down(nn), up(nn) {}
    vector<array<double,3>> score(const array<vector<int>,4>& groups) {
        array<int,4> totals = {};
        for (int c = 0; c < 4; ++c) for (int a : groups[c]) {
            counts[a] = {0,0,0,0}; counts[a][c] = 1; ++totals[c];
        }
        vector<array<double,3>> result;
        double nq = double(totals[0])*totals[1]*totals[2]*totals[3];
        for (const auto& g : base.genes) {
            for (int u : g.post) {
                int l = g.left[u], r = g.right[u];
                for (int c = 0; c < 4; ++c) counts[u][c] = counts[l][c]+counts[r][c];
                for (int p = 0; p < 6; ++p) {
                    int a = px[p], b = py[p];
                    down[u][p] = g.residual[l]*down[l][p]+g.residual[r]*down[r][p]
                        +counts[l][a]*counts[r][b]+counts[l][b]*counts[r][a];
                }
            }
            array<double,3> weights = {};
            for (int u : g.pre) {
                int par = g.parent[u], l = g.left[u], r = g.right[u];
                array<int,4> above;
                for (int c = 0; c < 4; ++c) above[c] = totals[c]-counts[u][c];
                if (par < n) up[u] = {};
                else {
                    int sibling = g.left[par] == u ? g.right[par] : g.left[par];
                    for (int p = 0; p < 6; ++p) {
                        int a = px[p], b = py[p];
                        up[u][p] = g.residual[par]*up[par][p]+g.residual[sibling]*down[sibling][p]
                            +(totals[a]-counts[par][a])*counts[sibling][b]
                            +(totals[b]-counts[par][b])*counts[sibling][a];
                    }
                }
                double pair_l[6], pair_r[6], pair_p[6];
                for (int p = 0; p < 6; ++p) {
                    int a = px[p], b = py[p];
                    pair_l[p] = counts[l][a]*counts[l][b]-g.residual[l]*down[l][p];
                    pair_r[p] = counts[r][a]*counts[r][b]-g.residual[r]*down[r][p];
                    pair_p[p] = above[a]*above[b]-g.residual[u]*up[u][p];
                }
                const int first[3] = {0,1,2}, second[3] = {5,4,3};
                for (int k = 0; k < 3; ++k) {
                    int p = first[k], q = second[k];
                    int a = px[p], b = py[p], c = px[q], d = py[q];
                    weights[k] +=
                        pair_l[p]*(counts[r][c]*above[d]+counts[r][d]*above[c])
                       +pair_l[q]*(counts[r][a]*above[b]+counts[r][b]*above[a])
                       +pair_r[p]*(counts[l][c]*above[d]+counts[l][d]*above[c])
                       +pair_r[q]*(counts[l][a]*above[b]+counts[l][b]*above[a])
                       +pair_p[p]*(counts[l][c]*counts[r][d]+counts[l][d]*counts[r][c])
                       +pair_p[q]*(counts[l][a]*counts[r][b]+counts[l][b]*counts[r][a]);
                }
            }
            for (double& w : weights) w /= 2*nq;
            result.push_back(weights);
        }
        return result;
    }
};

using BranchContextKey = array<uint64_t,8>;

BranchContextKey canonical_context(array<vector<int>,4>& groups) {
    array<TaxonSet,4> sets;
    for (int c = 0; c < 4; ++c) for (int a : groups[c]) sets[c].add(a);
    if (sets[1] < sets[0]) { swap(sets[0],sets[1]); swap(groups[0],groups[1]); }
    if (sets[3] < sets[2]) { swap(sets[2],sets[3]); swap(groups[2],groups[3]); }
    BranchContextKey key;
    for (int c = 0; c < 4; ++c) { key[2*c] = sets[c].lo; key[2*c+1] = sets[c].hi; }
    return key;
}

array<double,6> locus_statistics(const vector<array<double,3>>& scores, const vector<double>& rates) {
    vector<double> ps, margins;
    double wins = 0, info = 0, mean = 0, rate_mean = 0;
    for (int i = 0; i < int(scores.size()); ++i) {
        auto w = scores[i];
        double total = w[0]+w[1]+w[2];
        double p0 = w[0]+(1-total)/3;
        double p1 = w[1]+(1-total)/3, p2 = w[2]+(1-total)/3;
        double maximum = max({p0,p1,p2});
        int ties = (p0 >= maximum-1e-8)+(p1 >= maximum-1e-8)+(p2 >= maximum-1e-8);
        wins += p0 >= maximum-1e-8 ? 1.0/ties : 0;
        ps.push_back(p0); margins.push_back(p0-max(p1,p2));
        info += total; mean += p0; rate_mean += rates[i];
    }
    int m = scores.size();
    mean /= m; rate_mean /= m;
    double variance = 0, cov = 0, rate_var = 0;
    double mean_margin = accumulate(margins.begin(),margins.end(),0.0)/m;
    for (int i = 0; i < m; ++i) {
        variance += (ps[i]-mean)*(ps[i]-mean);
        cov += (rates[i]-rate_mean)*(margins[i]-mean_margin);
        rate_var += (rates[i]-rate_mean)*(rates[i]-rate_mean);
    }
    double margin_var = 0;
    for (double x : margins) margin_var += (x-mean_margin)*(x-mean_margin);
    sort(margins.begin(),margins.end());
    int trim = m/5;
    double trimmed = accumulate(margins.begin()+trim,margins.end()-trim,0.0)/(m-2*trim);
    return {sqrt(variance/m),0.5*(margins[(m-1)/2]+margins[m/2]),trimmed,
            wins/m,cov/sqrt(max(1e-16,rate_var*margin_var)),info/m};
}

struct BranchRecord {
    vector<double> votes;
    vector<array<double,17>> contexts;
    vector<array<vector<int>,4>> groups;
};

using BranchFeatureMap = map<TaxonSet,vector<double>>;

BranchFeatureMap calculate_branch_features(const Input& input, const vector<Tree>& trees,
                                          const Quartets& q, bool include_loci = true) {
    map<TaxonSet,BranchRecord> records;
    vector<double> rates(q.n), locus_rates;
    for (const Tree& g : input.genes) {
        double sum = 0, length = 0;
        for (int leaf = 0; leaf < q.n; ++leaf) sum += g.adj[leaf][0].length;
        double factor = q.n/max(1e-12,sum);
        for (int leaf = 0; leaf < q.n; ++leaf)
            rates[leaf] += g.adj[leaf][0].length*factor/input.genes.size();
        for (int u = 0; u < int(g.adj.size()); ++u)
            for (auto e : g.adj[u]) if (e.to > u) length += e.length;
        locus_rates.push_back(log(max(1e-8,length)));
    }
    for (int k = 0; k < int(trees.size()); ++k) {
        const Tree& t = trees[k];
        for (int u = t.n; u < int(t.adj.size()); ++u)
            for (auto e : t.adj[u]) if (e.to > u) {
                int v = e.to;
                vector<int> us, vs;
                for (auto z : t.adj[u]) if (z.to != v) us.push_back(z.to);
                for (auto z : t.adj[v]) if (z.to != u) vs.push_back(z.to);
                auto A = t.leaves(us[0],u), B = t.leaves(us[1],u);
                auto C = t.leaves(vs[0],v), D = t.leaves(vs[1],v);
                if (find(A.begin(),A.end(),0) != A.end() || find(B.begin(),B.end(),0) != B.end()) {
                    swap(A,C); swap(B,D);
                }
                TaxonSet clade;
                for (int a : A) clade.add(a);
                for (int a : B) clade.add(a);
                auto& r = records[clade];
                if (r.votes.empty()) r.votes.resize(trees.size());
                r.votes[k] = 1;
                double s[3] = {};
                for (int a : A) for (int b : B) for (int c : C) for (int d : D) {
                    s[0] += q.get(a,b,c,d); s[1] += q.get(a,c,b,d); s[2] += q.get(a,d,b,c);
                }
                double count = double(A.size())*B.size()*C.size()*D.size();
                double p[3], likelihood[3], suml = 0;
                for (int j = 0; j < 3; ++j) {
                    s[j] /= count;
                    p[j] = 1.0/3+s[j]/q.ngenes;
                    likelihood[j] = q.log_likelihood(s[j]);
                }
                double maxl = max({likelihood[0],likelihood[1],likelihood[2]});
                for (int j = 0; j < 3; ++j) {
                    likelihood[j] = exp(likelihood[j]-maxl); suml += likelihood[j];
                }
                double entropy = 0;
                for (double z : p) if (z > 0) entropy -= z*log(z);
                double small = min({A.size(),B.size(),C.size(),D.size()});
                double large = max({A.size(),B.size(),C.size(),D.size()});
                double child_balance = double(min(A.size(),B.size()))/max(A.size(),B.size());
                auto mean_rate = [&](const vector<int>& side) {
                    double sum = 0;
                    for (int leaf : side) sum += rates[leaf];
                    return sum/side.size();
                };
                double ra = mean_rate(A), rb = mean_rate(B), rc = mean_rate(C), rd = mean_rate(D);
                double mean_in = (ra*A.size()+rb*B.size())/(A.size()+B.size());
                double mean_out = (rc*C.size()+rd*D.size())/(C.size()+D.size());
                double zone = (2*(ra*rb+rc*rd)-(ra+rb)*(rc+rd))/
                              max(1e-10,ra*ra+rb*rb+rc*rc+rd*rd);
                r.contexts.push_back({p[0],p[0]-max(p[1],p[2]),fabs(p[1]-p[2]),
                    likelihood[0]/suml,entropy,log(count),log(small),log(large),child_balance,
                    double(A.size()+B.size())/t.n,1.0/double(min(A.size()+B.size(),C.size()+D.size())),
                    double(k),log(max(1e-6,mean_in)/max(1e-6,mean_out)),zone,
                    log(max(1e-6,max({ra,rb,rc,rd}))/max(1e-6,min({ra,rb,rc,rd}))),
                    log(max(1e-6,ra+rb)/max(1e-6,rc+rd)),(ra+rb+rc+rd)*0.25});
                r.groups.push_back({move(A),move(B),move(C),move(D)});
            }
    }
    unique_ptr<LocusQuartets> locus_scorer;
    if (include_loci) locus_scorer = make_unique<LocusQuartets>(input);
    map<BranchContextKey,array<double,6>> cache;
    BranchFeatureMap result;
    for (auto& [clade,r] : records) {
        vector<double> x = r.votes;
        double vote_sum = accumulate(r.votes.begin(),r.votes.end(),0.0);
        double vote_mean = vote_sum/r.votes.size();
        double ls = log(min(clade.size(),q.n-clade.size()));
        x.push_back(ls); x.push_back(ls*vote_mean); x.push_back(vote_mean*vote_mean);
        for (int field = 0; field < 11; ++field) {
            double sum = 0, lo = 1e100, hi = -1e100;
            for (const auto& c : r.contexts) {
                sum += c[field]; lo = min(lo,c[field]); hi = max(hi,c[field]);
            }
            x.push_back(sum/r.contexts.size());
            if (field < 5) { x.push_back(lo); x.push_back(hi); }
        }
        double p0 = 0, margin = 0, post = 0;
        for (const auto& c : r.contexts) { p0 += c[0]; margin += c[1]; post += c[3]; }
        p0 /= r.contexts.size(); margin /= r.contexts.size(); post /= r.contexts.size();
        x.push_back(p0*ls); x.push_back(margin*ls); x.push_back(post*ls); x.push_back(margin*vote_mean);
        for (int field = 12; field < 17; ++field) {
            double sum = 0, lo = 1e100, hi = -1e100;
            for (const auto& c : r.contexts) {
                sum += c[field]; lo = min(lo,c[field]); hi = max(hi,c[field]);
            }
            x.push_back(sum/r.contexts.size()); x.push_back(lo); x.push_back(hi);
        }
        if (!include_loci || vote_sum == double(trees.size())) {
            x.insert(x.end(),18,0);
        } else {
            array<double,6> sum = {}, lo, hi;
            lo.fill(1e100); hi.fill(-1e100);
            for (auto& groups : r.groups) {
                auto key = canonical_context(groups);
                auto it = cache.find(key);
                if (it == cache.end())
                    it = cache.emplace(key,locus_statistics(locus_scorer->score(groups),locus_rates)).first;
                for (int k = 0; k < 6; ++k) {
                    sum[k] += it->second[k];
                    lo[k] = min(lo[k],it->second[k]); hi[k] = max(hi[k],it->second[k]);
                }
            }
            for (int k = 0; k < 6; ++k) {
                x.push_back(sum[k]/r.groups.size()); x.push_back(lo[k]); x.push_back(hi[k]);
            }
        }
        result.emplace(clade,move(x));
    }
    return result;
}

enum class CalibrationMode { Probability, Linear, Greedy };

struct BranchCalibration {
    double intercept = 0;
    vector<double> coefficients;
    CalibrationMode mode = CalibrationMode::Probability;
};

map<TaxonSet,double> calibrated_branch_weights(const BranchFeatureMap& features,
                                              const BranchCalibration& model) {
    map<TaxonSet,double> result;
    for (const auto& [clade,x] : features) {
        double score = model.intercept;
        for (int j = 0; j < int(model.coefficients.size()); ++j) {
            if (j >= int(x.size()) || !isfinite(x[j])) throw runtime_error("Invalid branch feature");
            score += model.coefficients[j]*x[j];
        }
        if (model.mode == CalibrationMode::Probability) {
            if (score >= 0) score = 1/(1+exp(-score));
            else { double p = exp(score); score = p/(1+p); }
        }
        result[clade] = score;
    }
    return result;
}

bool high_word_order(const TaxonSet& a, const TaxonSet& b) {
    if (a.size() != b.size()) return a.size() < b.size();
    return tie(a.hi,a.lo) < tie(b.hi,b.lo);
}

Tree weighted_clade_tree(const Input& input, const map<TaxonSet,double>& weights) {
    int n = input.names.size();
    vector<TaxonSet> clades;
    TaxonSet full;
    for (int a = 1; a < n; ++a) {
        TaxonSet b; b.add(a); clades.push_back(b); full.add(a);
    }
    clades.push_back(full);
    for (const auto& [b,w] : weights) clades.push_back(b);
    sort(clades.begin(),clades.end(),high_word_order);
    clades.erase(unique(clades.begin(),clades.end()),clades.end());
    unordered_map<TaxonSet,int,TaxonSetHash> index;
    for (int i = 0; i < int(clades.size()); ++i) index[clades[i]] = i;
    vector<double> values(clades.size(),-1e100);
    vector<pair<int,int>> choice(clades.size(),{-1,-1});
    for (int i = 0; i < int(clades.size()); ++i) {
        int sz = clades[i].size();
        if (sz == 1) { values[i] = 0; continue; }
        for (int j = 0; j < i && clades[j].size() <= sz/2; ++j) {
            if (!clades[j].subset(clades[i])) continue;
            auto it = index.find(clades[i]^clades[j]);
            if (it == index.end()) continue;
            int k = it->second;
            if (clades[j].size() == clades[k].size() && j > k) continue;
            double value = values[j]+values[k];
            if (value > values[i]) { values[i] = value; choice[i] = {j,k}; }
        }
        auto it = weights.find(clades[i]);
        if (it != weights.end()) values[i] += it->second;
    }
    Tree t(n);
    t.names = input.names;
    function<int(int)> dfs = [&](int i) {
        if (clades[i].size() == 1) {
            for (int a = 1; a < n; ++a) if (clades[i].contains(a)) return a;
        }
        auto [j,k] = choice[i];
        if (j < 0) throw runtime_error("Incomplete weighted clade tree");
        int u = t.node();
        t.link(u,dfs(j)); t.link(u,dfs(k));
        return u;
    };
    t.link(0,dfs(index.at(full)));
    return t;
}

bool compatible_clades(const TaxonSet& a, const TaxonSet& b) {
    return !((a.lo&b.lo)|(a.hi&b.hi)) || a.subset(b) || b.subset(a);
}

Tree greedy_clade_tree(const Input& input, const map<TaxonSet,double>& weights,
                       const Tree& reference) {
    int n = input.names.size();
    vector<pair<TaxonSet,double>> candidates(weights.begin(),weights.end());
    stable_sort(candidates.begin(),candidates.end(),[](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    vector<TaxonSet> clades;
    for (const auto& [a,w] : candidates) {
        bool good = true;
        for (const auto& b : clades) if (!compatible_clades(a,b)) { good = false; break; }
        if (good) clades.push_back(a);
    }
    TaxonSet full;
    for (int a = 1; a < n; ++a) {
        TaxonSet b; b.add(a); clades.push_back(b); full.add(a);
    }
    clades.push_back(full);
    sort(clades.begin(),clades.end(),high_word_order);
    clades.erase(unique(clades.begin(),clades.end()),clades.end());
    vector<vector<int>> children(clades.size());
    int root = int(find(clades.begin(),clades.end(),full)-clades.begin());
    for (int i = 0; i < int(clades.size()); ++i) if (i != root) {
        int parent = -1;
        for (int j = i+1; j < int(clades.size()); ++j)
            if (clades[i].subset(clades[j])) { parent = j; break; }
        if (parent < 0) throw runtime_error("Disconnected clade hierarchy");
        children[parent].push_back(i);
    }
    Matrix rooted = tree_distances(reference,0);
    auto outgroup_distance = rooted[0];
    // Shared-depth distances refine polytomies without changing any selected clade.
    for (int a = 0; a < n; ++a) for (int b = 0; b < n; ++b)
        rooted[a][b] -= outgroup_distance[a]+outgroup_distance[b];
    Tree t(n);
    t.names = input.names;
    struct Component { int node; vector<int> leaves; };
    function<Component(int)> dfs = [&](int i) -> Component {
        if (clades[i].size() == 1) {
            for (int a = 1; a < n; ++a) if (clades[i].contains(a)) return {a,{a}};
        }
        int u = t.node();
        vector<Component> parts;
        for (int child : children[i]) parts.push_back(dfs(child));
        if (parts.size() < 2) throw runtime_error("Incomplete clade hierarchy");
        while (parts.size() > 2) {
            int bi = -1, bj = -1;
            double best = numeric_limits<double>::infinity();
            for (int a = 1; a < int(parts.size()); ++a) for (int b = 0; b < a; ++b) {
                double distance = 0;
                for (int x : parts[a].leaves) for (int y : parts[b].leaves) distance += rooted[x][y];
                distance /= double(parts[a].leaves.size())*parts[b].leaves.size();
                if (distance < best) { best = distance; bi = a; bj = b; }
            }
            int v = t.node();
            t.link(v,parts[bi].node); t.link(v,parts[bj].node);
            vector<int> leaves = parts[bi].leaves;
            leaves.insert(leaves.end(),parts[bj].leaves.begin(),parts[bj].leaves.end());
            parts.erase(parts.begin()+bi); parts.erase(parts.begin()+bj);
            parts.push_back({v,move(leaves)});
        }
        t.link(u,parts[0].node); t.link(u,parts[1].node);
        vector<int> leaves = move(parts[0].leaves);
        leaves.insert(leaves.end(),parts[1].leaves.begin(),parts[1].leaves.end());
        return {u,move(leaves)};
    };
    t.link(0,dfs(root).node);
    return t;
}

Tree assemble_calibrated_tree(const Input& input, const BranchFeatureMap& features,
                             const BranchCalibration& model, const Tree& reference) {
    auto weights = calibrated_branch_weights(features,model);
    if (model.mode == CalibrationMode::Greedy) return greedy_clade_tree(input,weights,reference);
    return weighted_clade_tree(input,weights);
}

bool valid_prediction(const Tree& t, int n) {
    if (t.n != n || int(t.names.size()) != n) return false;
    int internal = 0;
    for (int u = 0; u < int(t.adj.size()); ++u) {
        if (u < n && t.adj[u].size() != 1) return false;
        if (u >= n && !t.adj[u].empty()) {
            if (t.adj[u].size() != 3) return false;
            ++internal;
        }
    }
    if (internal != n-2) return false;
    vector<bool> visited(t.adj.size());
    vector<int> stack = {0};
    visited[0] = true;
    int leaves = 0;
    while (!stack.empty()) {
        int u = stack.back(); stack.pop_back();
        if (u < n) ++leaves;
        for (auto e : t.adj[u]) if (!visited[e.to]) {
            visited[e.to] = true; stack.push_back(e.to);
        }
    }
    return leaves == n;
}

#endif
