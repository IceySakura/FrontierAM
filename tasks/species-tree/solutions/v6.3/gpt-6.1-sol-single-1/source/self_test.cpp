#define main inference_main
#include "main.cpp"
#undef main

static void require(bool condition, const string &message) {
    if (!condition) throw runtime_error(message);
}

static Tree random_tree(int n, mt19937 &rng) {
    Tree tree(n);
    int root = tree.addnode();
    tree.add(root,0); tree.add(root,1); tree.add(root,2);
    for (int leaf = 3; leaf < n; ++leaf) {
        vector<pair<int,int>> edges;
        for (int v = 0; v < int(tree.adj.size()); ++v)
            for (auto e : tree.adj[v]) if (v < e.v) edges.push_back({v,e.v});
        auto [a,b] = edges[rng()%edges.size()];
        tree.del(a,b);
        int v = tree.addnode();
        tree.add(v,a); tree.add(v,b); tree.add(v,leaf);
    }
    for (int v = 0; v < int(tree.adj.size()); ++v)
        for (auto &e : tree.adj[v]) if (v < e.v) {
            e.support = (rng()%1001)/1000.0;
            e.length = (rng()%1001)/1000.0;
            for (auto &reverse : tree.adj[e.v]) if (reverse.v == v) {
                reverse.support = e.support;
                reverse.length = e.length;
            }
        }
    return tree;
}

static array<double,3> brute(const Tree &tree, const vector<int> &color, int mode) {
    vector<int> groups[4];
    for (int i = 0; i < tree.n; ++i) groups[color[i]].push_back(i);
    vector<pair<Bits,double>> splits;
    for (int v = tree.n; v < int(tree.adj.size()); ++v)
        for (auto e : tree.adj[v]) if (v < e.v) {
            Bits side;
            for (int x : tree.leaves(v,e.v)) side.set(x);
            splits.push_back({side,1-reliability(e.support,e.length,mode)});
        }
    array<double,3> result{};
    for (int a : groups[0]) for (int b : groups[1])
        for (int c : groups[2]) for (int d : groups[3]) {
            int leaf[4] = {a,b,c,d}, resolution = -1;
            double q = 1;
            for (auto [side,p] : splits) {
                int mask = 0;
                for (int i = 0; i < 4; ++i)
                    if (side.a[leaf[i]/64] >> (leaf[i]%64) & 1) mask |= 1 << i;
                if (__builtin_popcount(unsigned(mask)) != 2) continue;
                q *= p;
                if (mask == 3 || mask == 12) resolution = 0;
                if (mask == 5 || mask == 10) resolution = 1;
                if (mask == 9 || mask == 6) resolution = 2;
            }
            require(resolution >= 0,"Unresolved generated quartet");
            result[resolution] += 1-q;
        }
    return result;
}

static vector<Tree> enumerate(int n) {
    Tree tree(n);
    int root = tree.addnode();
    tree.add(root,0); tree.add(root,1); tree.add(root,2);
    vector<Tree> trees{tree};
    for (int leaf = 3; leaf < n; ++leaf) {
        vector<Tree> next;
        for (const auto &t : trees)
            for (int v = 0; v < int(t.adj.size()); ++v)
                for (auto e : t.adj[v]) if (v < e.v) {
                    Tree u = t;
                    u.del(v,e.v);
                    int x = u.addnode();
                    u.add(x,v); u.add(x,e.v); u.add(x,leaf);
                    next.push_back(move(u));
                }
        trees = move(next);
    }
    return trees;
}

int main() {
    try {
        for (const string text : {
            "(0:0,(1:0,(2:0,3:0)0.95:0.1):0.2);",
            "[note](0:0,1:0,(2:0,3:0)[edge]0.9:0);",
            "('0':1,('a b':0,'c''d':0)0.7:0,('e,f':0,'x;z':1):0);"
        }) {
            Parser parser(text);
            int root = parser.parse();
            unordered_map<string,int> ids;
            for (const auto &v : parser.nodes) if (v.child.empty()) {
                int id = ids.size();
                ids.emplace(v.label,id);
            }
            Tree parsed = convert(parser,root,ids);
            require(parsed.n == int(ids.size()),"Parser taxon count mismatch");
        }
        mt19937 rng(7123);
        for (int rep = 0; rep < 100; ++rep) {
            int n = 4+rng()%9;
            vector<Tree> genes;
            for (int i = 0; i < 3; ++i) genes.push_back(random_tree(n,rng));
            vector<int> color(n);
            iota(color.begin(),color.begin()+4,0);
            for (int i = 4; i < n; ++i) color[i] = rng()%4;
            shuffle(color.begin(),color.end(),rng);
            for (int mode : {0,1,4,6}) for (double info : {0.0,0.5}) {
                QuartetScorer scorer(genes,info);
                scorer.setmode(mode);
                scorer.geneweight = {0,2,0.5};
                array<double,3> expected{};
                double potential = 1;
                for (int c = 0; c < 4; ++c)
                    potential *= count(color.begin(),color.end(),c);
                for (int g = 0; g < 3; ++g) {
                    auto s = brute(genes[g],color,mode);
                    double density = (s[0]+s[1]+s[2])/potential;
                    double weight = scorer.geneweight[g]*pow(max(0.2,density),-info);
                    for (int i = 0; i < 3; ++i) expected[i] += weight*s[i];
                }
                auto observed = scorer.score(color);
                for (int i = 0; i < 3; ++i)
                    require(abs(expected[i]-observed[i]) < 1e-7,"Quartet score mismatch");
            }
            vector<string> labels;
            unordered_map<string,int> ids;
            for (int i = 0; i < n; ++i) {
                labels.push_back(i == 0 ? "0" : "leaf '"+to_string(i)+", name");
                ids[labels.back()] = i;
            }
            string serialized = newick(genes[0],labels);
            Parser parser(serialized);
            int root = parser.parse();
            Tree parsed = convert(parser,root,ids);
            auto a = clusters(genes[0]), b = clusters(parsed);
            sort(a.begin(),a.end()); sort(b.begin(),b.end());
            require(a == b,"Quoted-label Newick round trip mismatch");
        }
        auto all = enumerate(6);
        require(all.size() == 105,"Incorrect tree enumeration");
        for (int rep = 0; rep < 30; ++rep) {
            vector<Tree> ensemble;
            for (int i = 0; i < 10; ++i) ensemble.push_back(all[rng()%all.size()]);
            unordered_map<Bits,int,BitsHash> frequencies;
            for (const auto &t : ensemble) for (auto b : clusters(t)) ++frequencies[b];
            auto value = [&](const Tree &t, bool restricted) {
                int result = 0;
                for (auto b : clusters(t)) {
                    auto it = frequencies.find(b);
                    if (restricted && it == frequencies.end()) return -100000;
                    if (it != frequencies.end()) result += it->second;
                }
                return result;
            };
            int best = -1;
            for (const auto &t : all) best = max(best,value(t,true));
            Tree inferred = consensus(ensemble);
            require(value(inferred,true) == best,"Consensus dynamic program mismatch");
        }
        cout << "Passed: exhaustive quartet scores, support calibrations, locus weights,\n"
             << "local information adjustment, Newick round trips, and consensus DP.\n";
        return 0;
    } catch (const exception &error) {
        cerr << error.what() << "\n";
        return 1;
    }
}
