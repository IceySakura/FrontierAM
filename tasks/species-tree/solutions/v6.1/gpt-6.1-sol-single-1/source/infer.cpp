#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

struct Edge {
    int to;
    double length = 0;
    double support = -1;
};

struct Tree {
    int n = 0;
    vector<vector<Edge>> adj;
    vector<string> names;

    Tree() = default;
    explicit Tree(int n_) : n(n_), adj(n_), names(n_) {
        for (int i = 0; i < n; ++i) names[i] = to_string(i);
    }
    int node() {
        adj.emplace_back();
        return int(adj.size()) - 1;
    }
    void link(int a, int b, double length = 0, double support = -1) {
        adj[a].push_back({b, length, support});
        adj[b].push_back({a, length, support});
    }
    void unlink(int a, int b) {
        auto erase_one = [](vector<Edge>& es, int v) {
            for (size_t i = 0; i < es.size(); ++i)
                if (es[i].to == v) {
                    es.erase(es.begin() + i);
                    return;
                }
            throw runtime_error("Missing edge");
        };
        erase_one(adj[a], b);
        erase_one(adj[b], a);
    }
    void suppress_degree_two() {
        for (int u = n; u < int(adj.size()); ++u) {
            if (adj[u].size() != 2) continue;
            Edge a = adj[u][0], b = adj[u][1];
            unlink(u, a.to);
            unlink(u, b.to);
            double s = a.support < 0 ? b.support :
                       b.support < 0 ? a.support : min(a.support, b.support);
            link(a.to, b.to, a.length + b.length, s);
        }
    }
    vector<int> leaves(int root, int parent) const {
        vector<int> result;
        function<void(int,int)> dfs = [&](int u, int p) {
            if (u < n) result.push_back(u);
            for (const auto& e : adj[u])
                if (e.to != p) dfs(e.to, u);
        };
        dfs(root, parent);
        return result;
    }
    string newick() const {
        int root = -1;
        for (int u = n; u < int(adj.size()); ++u)
            if (adj[u].size() == 3) { root = u; break; }
        if (root < 0) throw runtime_error("No internal root");
        auto quoted = [](const string& x) {
            if (x.find_first_of(" \t\r\n()[],:;'") == string::npos) return x;
            string r = "'";
            for (char c : x) { r += c; if (c == '\'') r += c; }
            return r + "'";
        };
        function<string(int,int)> dfs = [&](int u, int p) -> string {
            if (u < n) return quoted(names[u]);
            string s = "(";
            bool first = true;
            for (const auto& e : adj[u]) if (e.to != p) {
                if (!first) s += ",";
                first = false;
                s += dfs(e.to, u);
            }
            return s + ")";
        };
        return dfs(root, -1) + ";\n";
    }
};

struct NewickParser {
    const string& text;
    size_t pos = 0;
    Tree t;
    unordered_map<string,int>& labels;
    bool discover;
    struct Subtree { int u; double len; double sup; };
    NewickParser(const string& s, unordered_map<string,int>& l, bool d)
        : text(s), t(int(l.size())), labels(l), discover(d) {}
    void skip() {
        for (;;) {
            while (pos < text.size() && isspace(static_cast<unsigned char>(text[pos]))) ++pos;
            if (pos == text.size() || text[pos] != '[') break;
            int depth = 0;
            do {
                if (text[pos] == '[') ++depth;
                if (text[pos] == ']') --depth;
                ++pos;
            } while (pos < text.size() && depth);
        }
    }
    string token() {
        skip();
        string s;
        if (pos < text.size() && text[pos] == '\'') {
            ++pos;
            while (pos < text.size()) {
                char c = text[pos++];
                if (c == '\'') {
                    if (pos < text.size() && text[pos] == '\'') ++pos;
                    else break;
                }
                s += c;
            }
            return s;
        }
        while (pos < text.size()) {
            char c = text[pos];
            if (isspace(static_cast<unsigned char>(c)) || strchr("(),:;[]", c)) break;
            s += c;
            ++pos;
        }
        return s;
    }
    Subtree subtree() {
        skip();
        int u;
        double support = -1, length = 0;
        if (pos < text.size() && text[pos] == '(') {
            ++pos;
            u = t.node();
            for (;;) {
                auto child = subtree();
                t.link(u, child.u, child.len, child.sup);
                skip();
                if (pos >= text.size()) throw runtime_error("Incomplete Newick");
                char c = text[pos++];
                if (c == ')') break;
                if (c != ',') throw runtime_error("Invalid Newick child list");
            }
            skip();
            if (pos < text.size() && !strchr("(),:;[", text[pos])) {
                string s = token();
                char* end;
                double p = strtod(s.c_str(), &end);
                if (end != s.c_str() && *end == '\0') support = p;
            }
        } else {
            string name = token();
            auto it = labels.find(name);
            if (it == labels.end()) throw runtime_error("Unknown taxon: " + name);
            u = it->second;
        }
        skip();
        if (pos < text.size() && text[pos] == ':') {
            ++pos;
            string s = token();
            if (s.empty()) throw runtime_error("Missing length");
            length = stod(s);
        }
        return {u, length, support};
    }
    Tree parse() {
        subtree();
        skip();
        if (pos >= text.size() || text[pos] != ';') throw runtime_error("Missing semicolon");
        for (const auto& kv : labels) t.names[kv.second] = kv.first;
        t.suppress_degree_two();
        return move(t);
    }
};

vector<string> newick_tokens(const string& text) {
    vector<string> leaves;
    bool want_leaf = true;
    size_t pos = 0;
    while (pos < text.size()) {
        char c = text[pos];
        if (isspace(static_cast<unsigned char>(c))) { ++pos; continue; }
        if (c == '[') {
            while (pos < text.size() && text[pos] != ']') ++pos;
            if (pos < text.size()) ++pos;
            continue;
        }
        if (strchr("(),:;", c)) {
            want_leaf = (c == '(' || c == ',');
            ++pos;
            continue;
        }
        string s;
        if (c == '\'') {
            ++pos;
            while (pos < text.size()) {
                char q = text[pos++];
                if (q == '\'') {
                    if (pos < text.size() && text[pos] == '\'') ++pos;
                    else break;
                }
                s += q;
            }
        } else {
            while (pos < text.size() && !isspace(static_cast<unsigned char>(text[pos])) &&
                   !strchr("(),:;[]", text[pos])) s += text[pos++];
        }
        if (want_leaf) leaves.push_back(s);
        want_leaf = false;
    }
    return leaves;
}

struct Input {
    vector<Tree> genes;
    unordered_map<string,int> labels;
    vector<string> names;
};

Input read_input(const string& path) {
    ifstream in(path);
    if (!in) throw runtime_error("Cannot open input");
    string all((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
    if (all.compare(0,3,"\xEF\xBB\xBF") == 0) all.erase(0,3);
    vector<string> texts;
    bool quote = false;
    int comment = 0;
    size_t begin = 0;
    for (size_t p = 0; p < all.size(); ++p) {
        char c = all[p];
        if (c == '\'' && !comment) {
            if (quote && p + 1 < all.size() && all[p + 1] == '\'') ++p;
            else quote = !quote;
        }
        if (!quote) {
            if (c == '[') ++comment;
            if (c == ']') --comment;
            if (c == ';' && !comment) {
                texts.push_back(all.substr(begin, p - begin + 1));
                begin = p + 1;
            }
        }
    }
    if (texts.empty()) throw runtime_error("No gene trees");
    Input result;
    result.names = newick_tokens(texts[0]);
    sort(result.names.begin(), result.names.end());
    if (adjacent_find(result.names.begin(), result.names.end()) != result.names.end())
        throw runtime_error("Duplicate taxa");
    for (int i = 0; i < int(result.names.size()); ++i) result.labels[result.names[i]] = i;
    for (const auto& text : texts)
        result.genes.push_back(NewickParser(text, result.labels, false).parse());
    return result;
}

using Matrix = vector<vector<double>>;

double edge_metric(const Tree& t, int u, const Edge& e, int method) {
    if (method == 0) return 1;
    if (method == 6) return e.length;
    if (method == 7) return sqrt(max(0.0, e.length));
    if (method == 8) return log1p(max(0.0, e.length) * 200);
    if (u < t.n || e.to < t.n) return 1;
    double p = e.support < 0 ? (e.length > 1e-5 ? 1.0 : 1.0/3) : e.support;
    p = clamp(p, 0.0, 1.0);
    if (method == 1) return p;
    if (method == 2) return max(0.0, (3*p-1)*0.5);
    if (method == 3) return p >= 0.9;
    if (method == 4) return p >= 0.5;
    if (method == 5) return e.length > 1e-5;
    if (method == 9) return pow(p, 4);
    if (method == 10) return -log(max(1e-4, min(1.0, 1.5*(1-p))));
    return 1;
}

Matrix tree_distances(const Tree& t, int method) {
    Matrix d(t.n, vector<double>(t.n));
    for (int a = 0; a < t.n; ++a) {
        function<void(int,int,double)> dfs = [&](int u, int p, double dist) {
            if (u < t.n) d[a][u] = dist;
            for (const auto& e : t.adj[u])
                if (e.to != p) dfs(e.to, u, dist + edge_metric(t, u, e, method));
        };
        dfs(a, -1, 0);
    }
    return d;
}

Matrix average_distances(const vector<Tree>& genes, int method, int normalize) {
    int n = genes[0].n;
    Matrix d(n, vector<double>(n));
    for (const auto& t : genes) {
        Matrix g = tree_distances(t, method);
        double sum = 0, sumsq = 0;
        for (int a = 0; a < n; ++a)
            for (int b = 0; b < a; ++b) { sum += g[a][b]; sumsq += g[a][b]*g[a][b]; }
        double scale = 1;
        if (normalize == 1) scale = n*(n-1)*0.5/max(1e-20,sum);
        if (normalize == 2) scale = n*(n-1)*0.5/max(1e-20,sumsq);
        for (int a = 0; a < n; ++a)
            for (int b = 0; b < n; ++b) d[a][b] += g[a][b]*scale/genes.size();
    }
    return d;
}

Matrix robust_average_distances(const vector<Tree>& genes, int method, bool quantile) {
    int n = genes[0].n;
    vector<Matrix> matrices;
    for (const auto& t : genes) {
        Matrix d = tree_distances(t,method);
        double sum = 0;
        for (const auto& row : d) sum += accumulate(row.begin(),row.end(),0.0);
        double scale = n*(n-1)/max(1e-20,sum);
        for (auto& row : d) for (double& x : row) x *= scale;
        matrices.push_back(move(d));
    }
    Matrix result(n,vector<double>(n));
    for (int a = 0; a < n; ++a) for (int b = 0; b < a; ++b) {
        vector<double> values;
        for (const auto& d : matrices) values.push_back(d[a][b]);
        sort(values.begin(),values.end());
        double value;
        if (quantile) value = values[values.size()/4];
        else {
            int trim = int(values.size()/5);
            value = accumulate(values.begin()+trim,values.end()-trim,0.0)/(values.size()-2*trim);
        }
        result[a][b] = result[b][a] = value;
    }
    return result;
}

Tree neighbor_joining(const Matrix& input, const vector<string>& names, bool upgma = false,
                      double noise = 0, mt19937_64* rng = nullptr) {
    int n = int(input.size());
    Tree t(n);
    t.names = names;
    int cap = 2*n;
    Matrix d(cap, vector<double>(cap));
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < a; ++b) {
            double x = input[a][b];
            if (noise && rng) x *= exp(noise * normal_distribution<double>()(*rng));
            d[a][b] = d[b][a] = x;
        }
    vector<int> active(n), sizes(cap, 1);
    iota(active.begin(), active.end(), 0);
    while (active.size() > 3) {
        int m = int(active.size());
        vector<double> sums(cap);
        for (int a : active) for (int b : active) sums[a] += d[a][b];
        double best = numeric_limits<double>::infinity();
        int ba = -1, bb = -1;
        for (int i = 0; i < m; ++i) for (int j = 0; j < i; ++j) {
            int a = active[i], b = active[j];
            double q = upgma ? d[a][b] : (m-2)*d[a][b] - sums[a] - sums[b];
            if (q < best) { best = q; ba = a; bb = b; }
        }
        int u = t.node();
        t.link(u, ba);
        t.link(u, bb);
        for (int a : active) if (a != ba && a != bb) {
            double x = upgma ? (sizes[ba]*d[ba][a] + sizes[bb]*d[bb][a])/(sizes[ba]+sizes[bb])
                             : (d[ba][a] + d[bb][a] - d[ba][bb])*0.5;
            d[u][a] = d[a][u] = x;
        }
        sizes[u] = sizes[ba] + sizes[bb];
        active.erase(remove(active.begin(), active.end(), ba), active.end());
        active.erase(remove(active.begin(), active.end(), bb), active.end());
        active.push_back(u);
    }
    int root = t.node();
    for (int a : active) t.link(root, a);
    return t;
}

struct Quartets {
    int n, npairs, ngenes;
    vector<int> pair_id;
    vector<float> q;
    Quartets(const Input& input, int method = 1, double parameter = 1) {
        n = int(input.names.size());
        ngenes = int(input.genes.size());
        npairs = n*(n-1)/2;
        pair_id.assign(n*n, -1);
        int p = 0;
        for (int b = 1; b < n; ++b) for (int a = 0; a < b; ++a)
            pair_id[a*n+b] = pair_id[b*n+a] = p++;
        size_t nq = size_t(n)*(n-1)*(n-2)*(n-3)/24;
        vector<array<float,3>> compact(nq);
        for (const auto& g : input.genes) {
            Matrix topo = tree_distances(g, 0);
            Matrix evidence;
            Matrix extra;
            if (method == 0) {
                evidence = topo;
            } else {
                Tree weighted = g;
                for (int u = 0; u < int(weighted.adj.size()); ++u)
                    for (auto& e : weighted.adj[u]) {
                        double p = e.support < 0 ? (e.length > 1e-5 ? 1.0 : 1.0/3) : e.support;
                        double s = max(0.0, min(1.0, (3*p-1)*0.5));
                        // Additive residual lengths combine uncertain edges along a quartet's central path.
                        if (method == 1 || method >= 9)
                            e.length = -log(max(1e-8, 1-pow(s, method == 1 ? parameter : 1)));
                        if (method == 2) e.length = e.length > 1e-5;
                        if (method == 3) e.length = p >= parameter;
                        if (method == 4) e.length = s;
                        if (method >= 5 && method <= 8) {
                            // Original substitution lengths are used below.
                        }
                    }
                evidence = tree_distances(weighted, 6);
            }
            if (method == 9 || method == 10 || method == 13)
                extra = tree_distances(g,6);
            if (method == 12) {
                Tree weighted = g;
                for (int u = 0; u < int(weighted.adj.size()); ++u)
                    for (auto& e : weighted.adj[u]) {
                        double p = e.support < 0 ? (e.length > 1e-5 ? 1.0 : 1.0/3) : e.support;
                        e.length = (u < n || e.to < n) ? 0 : -log(0.5+0.5*p);
                    }
                extra = tree_distances(weighted,6);
            }
            size_t index = 0;
            for (int d = 3; d < n; ++d) for (int c = 2; c < d; ++c)
                for (int b = 1; b < c; ++b) for (int a = 0; a < b; ++a, ++index) {
                    double z[3] = {topo[a][b]+topo[c][d], topo[a][c]+topo[b][d],
                                   topo[a][d]+topo[b][c]};
                    int best = z[1] < z[0] ? 1 : 0;
                    if (z[2] < z[best]) best = 2;
                    double w = 1;
                    if (method) {
                        double e[3] = {evidence[a][b]+evidence[c][d],
                                       evidence[a][c]+evidence[b][d],
                                       evidence[a][d]+evidence[b][c]};
                        double central = max(0.0, (e[(best+1)%3]-e[best])*0.5);
                        if (method == 1 || method == 4) w = -expm1(-central);
                        if (method == 2 || method == 3) w = central > 1e-8;
                        if (method == 5) w = central;
                        if (method == 6) w = -expm1(-central*parameter);
                        if (method == 7) w = -expm1(-central*parameter) * exp(-0.5*e[(best+1)%3]);
                        if (method == 8) w = central / max(1e-8, e[(best+1)%3]);
                        if (method >= 9) {
                            w = -expm1(-central);
                            double extra_min = 0;
                            if (!extra.empty()) {
                                double zextra[3] = {extra[a][b]+extra[c][d],
                                    extra[a][c]+extra[b][d], extra[a][d]+extra[b][c]};
                                extra_min = zextra[best];
                            }
                            if (method == 9 || method == 12) w *= exp(-parameter*extra_min);
                            if (method == 10) w *= pow(max(0.02,extra_min),-parameter);
                            if (method == 11) w *= exp(-parameter*(z[best]-4));
                            if (method == 13) w *= exp(-parameter*(extra_min+central));
                        }
                    }
                    compact[index][best] += float(w);
                }
        }
        q.assign(size_t(npairs)*npairs, 0);
        size_t index = 0;
        for (int d = 3; d < n; ++d) for (int c = 2; c < d; ++c)
            for (int b = 1; b < c; ++b) for (int a = 0; a < b; ++a, ++index) {
                const auto& x = compact[index];
                double center = (x[0]+x[1]+x[2])/3.0;
                auto put = [&](int a, int b, int c, int d, float score) {
                    int p = pair_id[a*n+b], r = pair_id[c*n+d];
                    q[size_t(p)*npairs+r] = q[size_t(r)*npairs+p] = score-float(center);
                };
                put(a,b,c,d,x[0]);
                put(a,c,b,d,x[1]);
                put(a,d,b,c,x[2]);
            }
    }
    float get(int a, int b, int c, int d) const {
        return q[size_t(pair_id[a*n+b])*npairs+pair_id[c*n+d]];
    }
    void set(int a, int b, int c, int d, float value) {
        int p = pair_id[a*n+b], r = pair_id[c*n+d];
        q[size_t(p)*npairs+r] = q[size_t(r)*npairs+p] = value;
    }
    double log_likelihood(double centered) const {
        double p = clamp(1.0/3+centered/ngenes,1.0/3,1.0);
        if (p <= 1.0/3) return 0;
        return ngenes*(p*log(3*p)+(p >= 1 ? 0 : (1-p)*log(1.5*(1-p))));
    }
    void transform(int mode, double parameter = 1) {
        for (int d = 3; d < n; ++d) for (int c = 2; c < d; ++c)
            for (int b = 1; b < c; ++b) for (int a = 0; a < b; ++a) {
                double x[3] = {get(a,b,c,d),get(a,c,b,d),get(a,d,b,c)};
                double y[3];
                if (mode == 1 || mode == 2)
                    for (int k = 0; k < 3; ++k) x[k] = log_likelihood(x[k]);
                if (mode == 2 || mode == 3) {
                    double maxx = max({x[0],x[1],x[2]}), sum = 0;
                    for (int k = 0; k < 3; ++k) { y[k] = exp((x[k]-maxx)/parameter); sum += y[k]; }
                    for (double& v : y) v /= sum;
                } else if (mode == 4) {
                    double maxx = max({x[0],x[1],x[2]}), sum = 0;
                    for (int k = 0; k < 3; ++k) { y[k] = x[k] >= maxx-1e-5; sum += y[k]; }
                    for (double& v : y) v /= sum;
                } else if (mode == 5) {
                    for (int k = 0; k < 3; ++k) y[k] = pow(max(0.0,x[k]),parameter);
                } else if (mode == 6) {
                    for (int k = 0; k < 3; ++k) y[k] = tanh(x[k]/parameter);
                } else for (int k = 0; k < 3; ++k) y[k] = x[k];
                double mean = (y[0]+y[1]+y[2])/3;
                set(a,b,c,d,float(y[0]-mean));
                set(a,c,b,d,float(y[1]-mean));
                set(a,d,b,c,float(y[2]-mean));
            }
    }
    Matrix consensus_distances(int mode = 4, double parameter = 1) const {
        Matrix distances(n,vector<double>(n));
        for (int d = 3; d < n; ++d) for (int c = 2; c < d; ++c)
            for (int b = 1; b < c; ++b) for (int a = 0; a < b; ++a) {
                double x[3] = {get(a,b,c,d),get(a,c,b,d),get(a,d,b,c)};
                double y[3];
                if (mode == 2) for (double& v : x) v = log_likelihood(v);
                if (mode == 2 || mode == 3) {
                    double maxx = max({x[0],x[1],x[2]}), sum = 0;
                    for (int k = 0; k < 3; ++k) { y[k] = exp((x[k]-maxx)/parameter); sum += y[k]; }
                    for (double& v : y) v /= sum;
                } else if (mode == 4) {
                    double maxx = max({x[0],x[1],x[2]}), sum = 0;
                    for (int k = 0; k < 3; ++k) { y[k] = x[k] >= maxx-1e-5; sum += y[k]; }
                    for (double& v : y) v /= sum;
                } else for (int k = 0; k < 3; ++k) y[k] = x[k]/ngenes+1.0/3;
                distances[a][b] += y[0]; distances[c][d] += y[0];
                distances[a][c] += y[1]; distances[b][d] += y[1];
                distances[a][d] += y[2]; distances[b][c] += y[2];
            }
        double count = (n-2)*(n-3)*0.5;
        for (int a = 0; a < n; ++a) for (int b = a+1; b < n; ++b)
            distances[a][b] = distances[b][a] = 1-distances[a][b]/count;
        return distances;
    }
    double score(const Tree& t) const {
        Matrix dist = tree_distances(t, 0);
        double result = 0;
        for (int d = 3; d < n; ++d) for (int c = 2; c < d; ++c)
            for (int b = 1; b < c; ++b) for (int a = 0; a < b; ++a) {
                double s[3] = {dist[a][b]+dist[c][d], dist[a][c]+dist[b][d],
                               dist[a][d]+dist[b][c]};
                int best = s[1] < s[0] ? 1 : 0;
                if (s[2] < s[best]) best = 2;
                if (best == 0) result += get(a,b,c,d);
                if (best == 1) result += get(a,c,b,d);
                if (best == 2) result += get(a,d,b,c);
            }
        return result;
    }
};

bool nni_pass(Tree& t, const Quartets& q, mt19937_64& rng) {
    vector<pair<int,int>> edges;
    for (int u = t.n; u < int(t.adj.size()); ++u)
        for (const auto& e : t.adj[u])
            if (e.to > u) edges.emplace_back(u,e.to);
    shuffle(edges.begin(), edges.end(), rng);
    bool changed = false;
    for (auto [u,v] : edges) {
        bool present = false;
        for (auto e : t.adj[u]) if (e.to == v) present = true;
        if (!present) continue;
        vector<int> us, vs;
        for (auto e : t.adj[u]) if (e.to != v) us.push_back(e.to);
        for (auto e : t.adj[v]) if (e.to != u) vs.push_back(e.to);
        auto A = t.leaves(us[0],u), B = t.leaves(us[1],u);
        auto C = t.leaves(vs[0],v), D = t.leaves(vs[1],v);
        double s0 = 0, s1 = 0, s2 = 0;
        for (int a : A) for (int b : B) for (int c : C) for (int d : D) {
            s0 += q.get(a,b,c,d);
            s1 += q.get(a,c,b,d);
            s2 += q.get(a,d,b,c);
        }
        int choice = s1 > s0 + 1e-5 ? 1 : 0;
        if (s2 > (choice == 0 ? s0 : s1) + 1e-5) choice = 2;
        if (choice) {
            int a = us[1], b = choice == 1 ? vs[0] : vs[1];
            t.unlink(u,a); t.unlink(v,b);
            t.link(u,b); t.link(v,a);
            changed = true;
        }
    }
    return changed;
}

int optimize_nni(Tree& t, const Quartets& q, mt19937_64& rng, int rounds = 30) {
    int done = 0;
    while (done < rounds && nni_pass(t,q,rng)) ++done;
    return done;
}

struct RootedView {
    int root, n, nn;
    vector<int> parent, depth, order, direction, lca;
    RootedView(const Tree& t, int root_) : root(root_), n(t.n), nn(int(t.adj.size())),
        parent(nn, -2), depth(nn), direction(nn*n,-1), lca(n*n,-1) {
        parent[root] = -1;
        order.push_back(root);
        for (size_t i = 0; i < order.size(); ++i) {
            int u = order[i];
            for (auto e : t.adj[u]) if (e.to != parent[u]) {
                parent[e.to] = u;
                depth[e.to] = depth[u]+1;
                order.push_back(e.to);
            }
        }
        for (int leaf = 0; leaf < n; ++leaf) {
            if (parent[leaf] == -2) continue;
            int prev = leaf, u = parent[leaf];
            while (u != -1) {
                direction[u*n+leaf] = prev;
                prev = u; u = parent[u];
            }
        }
        for (int u : order) for (int leaf = 0; leaf < n; ++leaf)
            if (direction[u*n+leaf] == -1) direction[u*n+leaf] = parent[u];
        for (int a = 0; a < n; ++a) if (parent[a] != -2)
            for (int b = 0; b < a; ++b) if (parent[b] != -2) {
                int u = a, v = b;
                while (depth[u] > depth[v]) u = parent[u];
                while (depth[v] > depth[u]) v = parent[v];
                while (u != v) { u = parent[u]; v = parent[v]; }
                lca[a*n+b] = lca[b*n+a] = u;
            }
    }
    int median(int a, int b, int c) const {
        int u = lca[a*n+b], v = lca[a*n+c], w = lca[b*n+c];
        if (depth[v] > depth[u]) u = v;
        if (depth[w] > depth[u]) u = w;
        return u;
    }
};

double spr_move(Tree& t, const Quartets& q, int a, int parent, int max_size = 101) {
    vector<int> pruned = t.leaves(a,parent);
    int m = int(pruned.size()), n = t.n;
    if (m > max_size || n-m < 3) return 0;
    vector<int> other_neighbors;
    for (auto e : t.adj[parent]) if (e.to != a) other_neighbors.push_back(e.to);
    if (other_neighbors.size() != 2) return 0;
    int x = other_neighbors[0], y = other_neighbors[1];
    t.unlink(a,parent); t.unlink(parent,x); t.unlink(parent,y); t.link(x,y);
    vector<bool> is_pruned(n);
    for (int u : pruned) is_pruned[u] = true;
    vector<int> rest;
    for (int u = 0; u < n; ++u) if (!is_pruned[u]) rest.push_back(u);
    RootedView view(t,rest[0]);
    vector<double> tags(t.adj.size());
    vector<vector<const float*>> rows;
    rows.resize(m, vector<const float*>(n,nullptr));
    for (int i = 0; i < m; ++i) for (int b : rest)
        rows[i][b] = q.q.data()+size_t(q.pair_id[pruned[i]*n+b])*q.npairs;
    for (int bi = 0; bi < int(rest.size()); ++bi) {
        int b = rest[bi];
        for (int ci = 0; ci < bi; ++ci) {
            int c = rest[ci];
            int bc = q.pair_id[b*n+c];
            for (int di = 0; di < ci; ++di) {
                int d = rest[di];
                int bd = q.pair_id[b*n+d], cd = q.pair_id[c*n+d];
                double wb = 0, wc = 0, wd = 0;
                for (int i = 0; i < m; ++i) {
                    wb += rows[i][b][cd];
                    wc += rows[i][c][bd];
                    wd += rows[i][d][bc];
                }
                int u = view.median(b,c,d);
                int db = view.direction[u*n+b], dc = view.direction[u*n+c],
                    dd = view.direction[u*n+d], p = view.parent[u];
                // Component updates become insertion-edge scores after one prefix traversal.
                double base = db == p ? wb : dc == p ? wc : wd;
                if (db != p) tags[db] += wb-base;
                if (dc != p) tags[dc] += wc-base;
                if (dd != p) tags[dd] += wd-base;
            }
        }
    }
    int original = view.parent[x] == y ? x : y;
    double best = -numeric_limits<double>::infinity();
    int best_node = original;
    for (int u : view.order) {
        if (u == view.root) continue;
        tags[u] += tags[view.parent[u]];
        if (tags[u] > best) { best = tags[u]; best_node = u; }
    }
    double delta = best-tags[original];
    if (delta > 1e-4) {
        int v = view.parent[best_node];
        t.unlink(best_node,v);
        t.link(parent,best_node); t.link(parent,v); t.link(parent,a);
    } else {
        t.unlink(x,y); t.link(parent,x); t.link(parent,y); t.link(parent,a);
        delta = 0;
    }
    return delta;
}

int spr_pass(Tree& t, const Quartets& q, mt19937_64& rng, int max_size = 1) {
    vector<pair<int,int>> edges;
    for (int u = 0; u < int(t.adj.size()); ++u)
        for (const auto& e : t.adj[u]) if (e.to >= t.n) {
            int sz = int(t.leaves(u,e.to).size());
            if (sz <= max_size && sz <= t.n/2) edges.emplace_back(u,e.to);
        }
    shuffle(edges.begin(), edges.end(), rng);
    int moves = 0;
    for (auto [a,parent] : edges) {
        bool present = false;
        for (auto e : t.adj[a]) if (e.to == parent) present = true;
        if (!present) continue;
        if (spr_move(t,q,a,parent,max_size) > 0) ++moves;
    }
    return moves;
}

using Profile = vector<pair<int,double>>;

Profile balanced_profile(const Tree& t, int u, int parent) {
    Profile result;
    function<void(int,int,double)> dfs = [&](int v, int p, double w) {
        if (v < t.n) { result.emplace_back(v,w); return; }
        int count = int(t.adj[v].size())-1;
        for (auto e : t.adj[v]) if (e.to != p) dfs(e.to,v,w/count);
    };
    dfs(u,parent,1);
    return result;
}

double bme_spr_move(Tree& t, const Matrix& distances, int a, int parent, int max_size) {
    Profile pruned = balanced_profile(t,a,parent);
    if (int(pruned.size()) > max_size || int(pruned.size()) > t.n/2) return 0;
    vector<int> nb;
    for (auto e : t.adj[parent]) if (e.to != a) nb.push_back(e.to);
    if (nb.size() != 2) return 0;
    int x = nb[0], y = nb[1], n = t.n;
    t.unlink(a,parent); t.unlink(parent,x); t.unlink(parent,y); t.link(x,y);
    vector<double> virtual_dist(n);
    vector<bool> absent(n);
    for (auto [u,w] : pruned) {
        absent[u] = true;
        for (int v = 0; v < n; ++v) virtual_dist[v] += w*distances[u][v];
    }
    int root = 0;
    while (absent[root]) ++root;
    vector<int> order = {root}, par(t.adj.size(),-2);
    par[root] = -1;
    for (size_t i = 0; i < order.size(); ++i) {
        int u = order[i];
        for (auto e : t.adj[u]) if (e.to != par[u]) {
            par[e.to] = u;
            order.push_back(e.to);
        }
    }
    auto merge_profile = [](const Profile& a, const Profile& b) {
        Profile p;
        p.reserve(a.size()+b.size());
        for (auto [u,w] : a) p.emplace_back(u,0.5*w);
        for (auto [u,w] : b) p.emplace_back(u,0.5*w);
        return p;
    };
    vector<Profile> down(t.adj.size()), up(t.adj.size());
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        int u = *it;
        if (u < n) down[u] = {{u,1}};
        else {
            vector<int> ch;
            for (auto e : t.adj[u]) if (e.to != par[u]) ch.push_back(e.to);
            down[u] = merge_profile(down[ch[0]],down[ch[1]]);
        }
    }
    for (int u : order) {
        if (u == root) continue;
        int p = par[u];
        if (p < n) up[u] = {{p,1}};
        else {
            int sibling = -1;
            for (auto e : t.adj[p]) if (e.to != u && e.to != par[p]) sibling = e.to;
            up[u] = merge_profile(up[p],down[sibling]);
        }
    }
    int original = par[x] == y ? x : y, best_node = original;
    double best = numeric_limits<double>::infinity(), original_score = 0;
    for (int u : order) if (u != root) {
        double das = 0, dat = 0, dst = 0;
        for (auto [v,w] : down[u]) {
            das += w*virtual_dist[v];
            double r = 0;
            for (auto [z,wz] : up[u]) r += wz*distances[v][z];
            dst += w*r;
        }
        for (auto [v,w] : up[u]) dat += w*virtual_dist[v];
        double score = das+dat-dst;
        if (u == original) original_score = score;
        if (score < best) { best = score; best_node = u; }
    }
    double delta = original_score-best;
    if (delta > 1e-10) {
        int v = par[best_node];
        t.unlink(best_node,v);
        t.link(parent,best_node); t.link(parent,v); t.link(parent,a);
    } else {
        t.unlink(x,y); t.link(parent,x); t.link(parent,y); t.link(parent,a);
        delta = 0;
    }
    return delta;
}

int bme_spr_pass(Tree& t, const Matrix& distances, mt19937_64& rng, int max_size = 50) {
    vector<pair<int,int>> edges;
    for (int u = 0; u < int(t.adj.size()); ++u)
        for (auto e : t.adj[u]) if (e.to >= t.n) {
            int sz = int(t.leaves(u,e.to).size());
            if (sz <= max_size && sz <= t.n/2) edges.emplace_back(u,e.to);
        }
    shuffle(edges.begin(),edges.end(),rng);
    int moves = 0;
    for (auto [a,p] : edges) {
        bool present = false;
        for (auto e : t.adj[a]) if (e.to == p) present = true;
        if (present && bme_spr_move(t,distances,a,p,max_size) > 0) ++moves;
    }
    return moves;
}

struct TaxonSet {
    uint64_t lo = 0, hi = 0;
    bool operator==(const TaxonSet& b) const { return lo == b.lo && hi == b.hi; }
    bool operator<(const TaxonSet& b) const { return tie(lo,hi) < tie(b.lo,b.hi); }
    TaxonSet operator|(const TaxonSet& b) const { return {lo|b.lo,hi|b.hi}; }
    TaxonSet operator^(const TaxonSet& b) const { return {lo^b.lo,hi^b.hi}; }
    bool subset(const TaxonSet& b) const { return !(lo&~b.lo) && !(hi&~b.hi); }
    bool contains(int a) const { return a < 64 ? (lo>>a)&1 : (hi>>(a-64))&1; }
    void add(int a) { if (a < 64) lo |= uint64_t(1)<<a; else hi |= uint64_t(1)<<(a-64); }
    int size() const { return __builtin_popcountll(lo)+__builtin_popcountll(hi); }
};

struct TaxonSetHash {
    size_t operator()(const TaxonSet& b) const {
        uint64_t x = b.lo ^ (b.hi+0x9e3779b97f4a7c15ULL+(b.lo<<6)+(b.lo>>2));
        x ^= x>>30; x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x>>27; x *= 0x94d049bb133111ebULL;
        return x^(x>>31);
    }
};

double branch_residual(const Edge& e, int method, double parameter) {
    double p = e.support < 0 ? (e.length > 1e-5 ? 1.0 : 1.0/3) : e.support;
    double s = max(0.0, min(1.0, (3*p-1)*0.5));
    if (method == 0) return 0;
    if (method == 1) return max(1e-8,1-pow(s,parameter));
    if (method == 2) return e.length > 1e-5 ? 0 : 1;
    if (method == 3) return p >= parameter ? 0 : 1;
    if (method == 4) return exp(-s);
    return 0;
}

struct TripartitionScorer {
    struct Gene {
        vector<int> parent, left, right, pre, post;
        vector<double> residual;
    };
    int n, nn;
    vector<Gene> genes;
    vector<float> last_scores;
    vector<array<int,3>> counts;
    vector<array<double,3>> pairs_down, pairs_up;
    TripartitionScorer(const Input& input, int method, double parameter) {
        n = int(input.names.size());
        nn = 2*n+4;
        counts.resize(nn);
        pairs_down.resize(nn);
        pairs_up.resize(nn);
        for (const auto& t : input.genes) {
            Gene g;
            int m = int(t.adj.size());
            g.parent.assign(m,-2); g.left.assign(m,-1); g.right.assign(m,-1);
            g.residual.assign(m,0);
            vector<int> order = {0};
            g.parent[0] = -1;
            for (size_t i = 0; i < order.size(); ++i) {
                int u = order[i];
                for (auto e : t.adj[u]) if (e.to != g.parent[u]) {
                    g.parent[e.to] = u;
                    g.residual[e.to] = branch_residual(e,method,parameter);
                    order.push_back(e.to);
                    if (g.left[u] == -1) g.left[u] = e.to; else g.right[u] = e.to;
                }
                if (u >= n) g.pre.push_back(u);
            }
            g.post = g.pre;
            reverse(g.post.begin(),g.post.end());
            genes.push_back(move(g));
        }
        last_scores.resize(genes.size());
    }
    double score(const TaxonSet& a, const TaxonSet& b) {
        array<int,3> totals = {a.size(),b.size(),n-a.size()-b.size()};
        for (int leaf = 0; leaf < n; ++leaf) {
            counts[leaf] = {0,0,0};
            int c = a.contains(leaf) ? 0 : b.contains(leaf) ? 1 : 2;
            counts[leaf][c] = 1;
        }
        double result = 0;
        int gene_index = 0;
        for (const Gene& g : genes) {
            double gene_start = result;
            for (int u : g.post) {
                int l = g.left[u], r = g.right[u];
                double rl = g.residual[l], rr = g.residual[r];
                for (int c = 0; c < 3; ++c) {
                    counts[u][c] = counts[l][c]+counts[r][c];
                    // Residual pair mass carries the unresolved probability to the pair's branching point.
                    pairs_down[u][c] = rl*pairs_down[l][c]+rr*pairs_down[r][c]
                                            +counts[l][c]*counts[r][c];
                }
            }
            for (int u : g.pre) {
                int p = g.parent[u], l = g.left[u], r = g.right[u];
                array<int,3> above;
                for (int c = 0; c < 3; ++c) above[c] = totals[c]-counts[u][c];
                if (p < n) pairs_up[u] = {0,0,0};
                else {
                    int sibling = g.left[p] == u ? g.right[p] : g.left[p];
                    for (int c = 0; c < 3; ++c)
                        pairs_up[u][c] = g.residual[p]*pairs_up[p][c]
                            +g.residual[sibling]*pairs_down[sibling][c]
                            +(totals[c]-counts[p][c])*counts[sibling][c];
                }
                for (int c = 0; c < 3; ++c) {
                    int b = (c+1)%3, d = (c+2)%3;
                    double nl = counts[l][c], nr = counts[r][c], np = above[c];
                    double wl = nl*(nl-1)*0.5-g.residual[l]*pairs_down[l][c];
                    double wr = nr*(nr-1)*0.5-g.residual[r]*pairs_down[r][c];
                    double wp = np*(np-1)*0.5-g.residual[u]*pairs_up[u][c];
                    result += wl*(counts[r][b]*above[d]+counts[r][d]*above[b])
                             +wr*(counts[l][b]*above[d]+counts[l][d]*above[b])
                             +wp*(counts[l][b]*counts[r][d]+counts[l][d]*counts[r][b]);
                }
            }
            last_scores[gene_index++] = float(result-gene_start);
        }
        return result;
    }
};

struct CladeSearch {
    int n;
    vector<string> names;
    unordered_map<TaxonSet,int,TaxonSetHash> index;
    vector<TaxonSet> clades;
    vector<int> sizes;
    vector<pair<int,int>> choice;
    vector<double> values;
    struct Alternative {
        int parent, left, right;
        double weight;
        vector<float> gene_weights;
    };
    vector<Alternative> alternatives;
    size_t evaluations = 0;
    explicit CladeSearch(const Input& input) : n(int(input.names.size())), names(input.names) {
        if (n > 128) throw runtime_error("Too many taxa");
        for (int a = 1; a < n; ++a) { TaxonSet b; b.add(a); add(b); }
    }
    void add(const TaxonSet& b) {
        if (b.size() && !index.count(b)) {
            index[b] = int(clades.size());
            clades.push_back(b);
        }
    }
    void add_tree(const Tree& t) {
        function<TaxonSet(int,int)> dfs = [&](int u, int p) {
            TaxonSet b;
            if (u < n) b.add(u);
            for (auto e : t.adj[u]) if (e.to != p) b = b|dfs(e.to,u);
            add(b);
            return b;
        };
        dfs(t.adj[0][0].to,0);
    }
    double tree_score(const Tree& t, TripartitionScorer& scorer) {
        double total = 0;
        function<TaxonSet(int,int)> dfs = [&](int u, int p) {
            TaxonSet b;
            if (u < n) b.add(u);
            vector<TaxonSet> child;
            for (auto e : t.adj[u]) if (e.to != p) {
                child.push_back(dfs(e.to,u)); b = b|child.back();
            }
            if (child.size() == 2) total += scorer.score(child[0],child[1]);
            return b;
        };
        dfs(t.adj[0][0].to,0);
        return total;
    }
    Tree solve(TripartitionScorer& scorer, const function<bool()>& stop = {}) {
        sort(clades.begin(),clades.end(),[](const TaxonSet& a, const TaxonSet& b) {
            if (a.size() != b.size()) return a.size() < b.size();
            return a < b;
        });
        index.clear();
        sizes.clear();
        for (int i = 0; i < int(clades.size()); ++i) {
            index[clades[i]] = i;
            sizes.push_back(clades[i].size());
        }
        values.assign(clades.size(),-1e100);
        choice.assign(clades.size(),{-1,-1});
        alternatives.clear();
        for (int i = 0; i < int(clades.size()); ++i) {
            if (stop && stop()) return Tree();
            if (sizes[i] == 1) { values[i] = 0; continue; }
            for (int j = 0; j < i && sizes[j] <= sizes[i]/2; ++j) {
                if (!clades[j].subset(clades[i])) continue;
                TaxonSet other = clades[i]^clades[j];
                auto it = index.find(other);
                if (it == index.end()) continue;
                int k = it->second;
                if (sizes[j] == sizes[k] && j > k) continue;
                double lower = values[j]+values[k];
                if (lower < -1e90) continue;
                double weight = scorer.score(clades[j],clades[k]);
                alternatives.push_back({i,j,k,weight,scorer.last_scores});
                double v = lower+weight;
                ++evaluations;
                if (v > values[i]) { values[i] = v; choice[i] = {j,k}; }
            }
        }
        return current_tree();
    }
    int root_index() const {
        TaxonSet full;
        for (int a = 1; a < n; ++a) full.add(a);
        return index.at(full);
    }
    Tree current_tree() const {
        int root = root_index();
        Tree t(n);
        t.names = names;
        function<int(int)> dfs = [&](int i) {
            if (sizes[i] == 1) {
                for (int a = 1; a < n; ++a) if (clades[i].contains(a)) return a;
            }
            auto [j,k] = choice[i];
            if (j < 0) throw runtime_error("Unresolvable candidate clade");
            int u = t.node();
            t.link(u,dfs(j)); t.link(u,dfs(k));
            return u;
        };
        t.link(0,dfs(root));
        return t;
    }
    void rescore(const vector<double>& gene_weights) {
        fill(values.begin(),values.end(),-1e100);
        for (int i = 0; i < int(clades.size()); ++i) if (sizes[i] == 1) values[i] = 0;
        for (const auto& a : alternatives) {
            double weight = 0;
            for (int g = 0; g < int(gene_weights.size()); ++g)
                weight += gene_weights[g]*a.gene_weights[g];
            double value = values[a.left]+values[a.right]+weight;
            if (value > values[a.parent]) {
                values[a.parent] = value; choice[a.parent] = {a.left,a.right};
            }
        }
    }
    void record_clades(vector<double>& counts) const {
        function<void(int)> dfs = [&](int i) {
            if (sizes[i] == 1) return;
            counts[i] += 1;
            auto [j,k] = choice[i];
            dfs(j); dfs(k);
        };
        dfs(root_index());
    }
    Tree bootstrap_consensus(mt19937_64& rng, int replicates, double temperature = 1,
                             bool bayesian = false) {
        int ng = int(alternatives.front().gene_weights.size());
        vector<double> frequency(clades.size());
        uniform_int_distribution<int> draw(0,ng-1);
        exponential_distribution<double> exponential(1);
        for (int boot = 0; boot < replicates; ++boot) {
            vector<double> weights(ng,1-temperature);
            if (bayesian) {
                for (double& w : weights) w += temperature*exponential(rng);
            } else {
                for (int g = 0; g < ng; ++g) weights[draw(rng)] += temperature;
            }
            rescore(weights);
            record_clades(frequency);
        }
        fill(values.begin(),values.end(),-1e100);
        for (int i = 0; i < int(clades.size()); ++i) if (sizes[i] == 1) values[i] = 0;
        for (const auto& a : alternatives) {
            double v = values[a.left]+values[a.right]+frequency[a.parent]+1e-10*a.weight;
            if (v > values[a.parent]) { values[a.parent] = v; choice[a.parent] = {a.left,a.right}; }
        }
        return current_tree();
    }
};

#ifndef INFER_LIBRARY
#include "ensemble.hpp"
#include "calibration.hpp"

int main(int argc, char** argv) {
    InferenceClock clock;
    try {
        if (argc != 3) throw runtime_error("Usage: run.sh INPUT OUTPUT");
        Input input = read_input(argv[1]);
        Tree result = neighbor_joining(average_distances(input.genes,9,0),input.names);
        const char* seed_text = getenv("SEED");
        uint64_t seed = seed_text ? strtoull(seed_text,nullptr,10) : 1;
        try {
            CandidateTrees candidates = infer_candidates(input,seed,clock);
            if (valid_prediction(candidates.fallback,input.names.size())) result = candidates.fallback;
            if (candidates.complete && !clock.expired(3)) {
                auto features = calculate_branch_features(input,candidates.trees,*candidates.evidence);
                Tree calibrated = assemble_calibrated_tree(input,features,fixed_calibration(),result);
                if (valid_prediction(calibrated,input.names.size())) result = move(calibrated);
            }
        } catch (const exception& e) {
            cerr << "Refinement fallback: " << e.what() << '\n';
        }
        if (!valid_prediction(result,input.names.size())) throw runtime_error("Invalid inferred tree");
        ofstream out(argv[2]);
        if (!out) throw runtime_error("Cannot open output");
        out << result.newick();
        out.close();
        if (!out) throw runtime_error("Cannot write output");
        cerr << "inference_seconds " << clock.elapsed() << '\n';
        return 0;
    } catch (const exception& e) {
        cerr << "Inference failed: " << e.what() << '\n';
        return 1;
    }
}
#endif
