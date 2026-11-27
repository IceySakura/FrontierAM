#define main inference_main
#include "infer.cpp"
#undef main
#include <cassert>

static array<double,3> brute(const Tree &tree, const vector<int> &color, const Options &opt) {
    int n = color.size(), m = tree.adj.size();
    Matrix dist(m,vector<double>(m)), length(m,vector<double>(m));
    Matrix failure(m,vector<double>(m,1)), support_failure(m,vector<double>(m,1));
    vector<int> leaves(n);
    double total = 0;
    for (int u = 0; u < m; ++u) {
        if (tree.taxon[u] >= 0) leaves[tree.taxon[u]] = u;
        for (Edge e : tree.adj[u]) total += e.length;
    }
    double scale = pow(max(1e-12,total/n),opt.qnormalize);
    double rate = support_rate(tree);
    for (int root = 0; root < m; ++root) {
        function<void(int,int)> walk = [&](int u, int p) {
            for (Edge e : tree.adj[u]) if (e.to != p) {
                dist[root][e.to] = dist[root][u]+1;
                length[root][e.to] = length[root][u]+e.length;
                double r = edge_reliability(e.support,e.length,rate,opt.legacy_missing);
                if (opt.qthreshold > 0)
                    r = (e.support < 0 ? (1+2*r)/3 : e.support) > opt.qthreshold ? 1 : 0;
                double s = opt.qblend+(1-opt.qblend)*opt.qsupport*pow(r,opt.qpower);
                double f = (opt.qproduct ? s : 1-s)*exp(-opt.qlength*e.length/scale);
                failure[root][e.to] = failure[root][u]*f;
                support_failure[root][e.to] = support_failure[root][u]*(1-s);
                walk(e.to,u);
            }
        };
        walk(root,-1);
    }
    array<double,3> result{};
    vector<int> groups[4];
    for (int i = 0; i < n; ++i) if (color[i] >= 0) groups[color[i]].push_back(leaves[i]);
    auto median = [&](int a, int b, int c) {
        int best = -1;
        double value = 1e100;
        for (int u = 0; u < m; ++u) if (!tree.adj[u].empty()) {
            double d = dist[u][a]+dist[u][b]+dist[u][c];
            if (d < value) { best = u; value = d; }
        }
        return best;
    };
    for (int a : groups[0]) for (int b : groups[1])
        for (int c : groups[2]) for (int d : groups[3]) {
            array<double,3> sums = {dist[a][b]+dist[c][d],
                                    dist[a][c]+dist[b][d],dist[a][d]+dist[b][c]};
            int topology = min_element(sums.begin(),sums.end())-sums.begin();
            int w = a, x = b, y = c, z = d;
            if (topology == 1) { x = c; y = b; }
            if (topology == 2) { x = d; z = b; }
            int u = median(w,x,y), v = median(w,y,z);
            double pendant = length[w][u]+length[x][u]+length[y][v]+length[z][v];
            double confidence = opt.qproduct ? failure[u][v] : 1-failure[u][v];
            if (opt.qand)
                confidence = (1-support_failure[u][v])*(-expm1(-opt.qlength*length[u][v]/scale));
            result[topology] += exp(-opt.qexternal*pendant/scale)*confidence;
        }
    return result;
}

static void valid(const Tree &tree, int n) {
    vector<int> count(n);
    int vertices = 0, edges = 0;
    int root = -1;
    for (int u = 0; u < (int)tree.adj.size(); ++u) {
        if (tree.adj[u].empty()) continue;
        root = u;
        ++vertices;
        edges += tree.adj[u].size();
        if (tree.taxon[u] >= 0) {
            assert(tree.adj[u].size() == 1);
            ++count.at(tree.taxon[u]);
        } else assert(tree.adj[u].size() == 3);
    }
    assert(vertices == 2*n-2 && edges == 2*(vertices-1));
    for (int c : count) assert(c == 1);
    vector<bool> seen(tree.adj.size());
    vector<int> stack = {root};
    int reached = 0;
    while (!stack.empty()) {
        int u = stack.back();
        stack.pop_back();
        if (seen[u]) continue;
        seen[u] = true;
        ++reached;
        for (Edge e : tree.adj[u]) if (!seen[e.to]) stack.push_back(e.to);
    }
    assert(reached == vertices);
}

int main() {
    assert(edge_reliability(-1,0,100,false) == 0);
    assert(edge_reliability(-1,2e-6,100,false) == 0);
    assert(abs(edge_reliability(-1,0.01,100,false)-(1-exp(-1))) < 1e-12);
    {
        Labels names;
        string text = "[outer[inner]] (('0':0.1,'a b':0.2):0,('x''y':0.3,z:0.4)0.9:0.1);";
        Parser parser{text,0,names,{}};
        Tree tree = parser.parse();
        valid(tree,4);
        assert(names.names[1] == "a b" && names.names[2] == "x'y");
        QuartetScorer scorer({tree},Options{});
        assert(abs(tree_score(tree,scorer,4)-0.85) < 1e-12);
        string output = newick(tree,names);
        Parser again{output,0,names,{}};
        assert(abs(tree_score(again.parse(),scorer,4)-0.85) < 1e-12);
    }
    mt19937 rng(123);
    uniform_real_distribution<double> uniform(0,1);
    for (int test = 0; test < 2000; ++test) {
        int n = 4+rng()%9;
        Tree tree;
        Labels names;
        for (int i = 0; i < n; ++i) { tree.node(i); names.get(to_string(i)); }
        int root = tree.node();
        for (int i = 0; i < 3; ++i) tree.link(root,i);
        for (int i = 3; i < n; ++i) {
            vector<pair<int,int>> edges;
            for (int u = 0; u < (int)tree.adj.size(); ++u)
                for (Edge e : tree.adj[u]) if (u < e.to) edges.emplace_back(u,e.to);
            auto [u,v] = edges[rng()%edges.size()];
            tree.unlink(u,v);
            int x = tree.node();
            tree.link(x,u); tree.link(x,v); tree.link(x,i);
        }
        for (int u = 0; u < (int)tree.adj.size(); ++u)
            for (Edge &e : tree.adj[u]) if (u < e.to) {
                e.length = uniform(rng);
                e.support = uniform(rng);
                if (test%4 == 0 && uniform(rng) < 0.3) {
                    e.support = -1;
                    if (uniform(rng) < 0.5) e.length = 0;
                }
                for (Edge &f : tree.adj[e.to]) if (f.to == u) {
                    f.length = e.length; f.support = e.support;
                }
            }
        Options opt;
        opt.qexternal = test%2 ? 2 : 0;
        opt.qlength = test%3 ? 1 : 0;
        opt.qsupport = test%4 ? 1 : 0;
        opt.qblend = test%5 ? 0 : 1;
        opt.qnormalize = test%11 ? 0 : 1;
        opt.qproduct = test%3 == 0;
        opt.qand = test%7 == 0 && !opt.qproduct;
        opt.qthreshold = test%13 ? 0 : 0.8;
        opt.qpower = test%17 ? 1 : 2;
        QuartetScorer scorer({tree},opt);
        vector<int> colors(n);
        for (int &c : colors) c = rng()%4;
        auto fast = scorer.score(colors), slow = brute(tree,colors,opt);
        for (int q = 0; q < 3; ++q) assert(abs(fast[q]-slow[q]) < 1e-8);
        for (int &c : colors) c = rng()%3;
        double expected_trip = 0;
        for (int c = 0; c < 3; ++c) expected_trip += scorer.score(colors,c)[0]/2;
        assert(abs(scorer.trip_score(colors)-expected_trip) < 1e-8);
        if (test < 200) {
            Options maximum = opt;
            maximum.qand = maximum.qproduct = false;
            maximum.qsupport = 1;
            constexpr int levels[] = {1,2,8,32};
            maximum.qmax = levels[test%4];
            maximum.qmixture = test%3 == 0 ? 0 : test%3 == 1 ? 0.5 : 1;
            Tree permuted = tree;
            for (int &taxon : permuted.taxon) if (taxon >= 0) taxon = (taxon+1)%n;
            QuartetScorer combined({tree,permuted,tree},maximum);
            QuartetScorer repeated({tree,permuted,tree},maximum,false);
            for (int &c : colors) c = rng()%4;
            auto a = combined.score(colors), b = repeated.score(colors);
            for (int q = 0; q < 3; ++q) assert(abs(a[q]-b[q]) < 1e-8);
            for (int &c : colors) c = rng()%3;
            assert(abs(combined.trip_score(colors)-repeated.trip_score(colors)) < 1e-8);
        }
        Tree alternative = tree;
        if (test < 30) {
            Matrix d(n,vector<double>(n));
            for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j)
                d[i][j] = d[j][i] = uniform(rng);
            Tree candidate = join(d,"nj");
            double before = tree_score(candidate,scorer,n);
            Options search = opt;
            search.nni = 15;
            search.spr = test < 10 ? 2 : 0;
            search.limit = 1e9;
            mt19937 search_rng(891+test);
            ostringstream quiet;
            streambuf *saved = cerr.rdbuf(quiet.rdbuf());
            double gain = improve_nni(candidate,scorer,n,search,search_rng);
            if (search.spr) gain += improve_spr(candidate,scorer,n,search,search_rng);
            cerr.rdbuf(saved);
            valid(candidate,n);
            assert(abs(tree_score(candidate,scorer,n)-before-gain) < 1e-6);
            alternative = candidate;
        }
        opt.qexternal = opt.qlength = opt.qnormalize = 0;
        opt.qand = false;
        opt.qblend = 1;
        QuartetScorer unweighted({tree},opt);
        double expected = double(n)*(n-1)*(n-2)*(n-3)/24;
        assert(abs(tree_score(tree,unweighted,n)-expected) < 1e-8);
        string output = newick(tree,names);
        Parser roundtrip{output,0,names,{}};
        Tree restored = roundtrip.parse();
        valid(restored,n);
        assert(abs(tree_score(restored,unweighted,n)-expected) < 1e-8);
        if (test < 30) {
            Tree consensus = split_consensus({tree,tree,alternative},alternative,n,0,0);
            valid(consensus,n);
            assert(abs(tree_score(consensus,unweighted,n)-expected) < 1e-8);
        }
    }
    cerr << "Passed 2000 randomized quartet, tripartition, topology, and parser tests.\n";
}
