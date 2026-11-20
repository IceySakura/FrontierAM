#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace st {

constexpr int capacity = 256;

struct Mask {
    std::array<uint64_t, 4> w{};

    static Mask leaf(int i) {
        Mask m;
        m.w[i >> 6] = uint64_t(1) << (i & 63);
        return m;
    }
    bool has(int i) const { return (w[i >> 6] >> (i & 63)) & 1; }
    int size() const {
        return __builtin_popcountll(w[0]) + __builtin_popcountll(w[1]) +
               __builtin_popcountll(w[2]) + __builtin_popcountll(w[3]);
    }
    bool empty() const { return !(w[0] | w[1] | w[2] | w[3]); }
    int first() const {
        for (int i = 0; i < 4; ++i)
            if (w[i]) return (i << 6) + __builtin_ctzll(w[i]);
        return -1;
    }
    template<class F> void each(F f) const {
        for (int i = 0; i < 4; ++i) {
            uint64_t x = w[i];
            while (x) {
                f((i << 6) + __builtin_ctzll(x));
                x &= x - 1;
            }
        }
    }
    friend Mask operator|(const Mask& a, const Mask& b) {
        return {{a.w[0] | b.w[0], a.w[1] | b.w[1], a.w[2] | b.w[2], a.w[3] | b.w[3]}};
    }
    friend Mask operator&(const Mask& a, const Mask& b) {
        return {{a.w[0] & b.w[0], a.w[1] & b.w[1], a.w[2] & b.w[2], a.w[3] & b.w[3]}};
    }
    friend Mask operator^(const Mask& a, const Mask& b) {
        return {{a.w[0] ^ b.w[0], a.w[1] ^ b.w[1], a.w[2] ^ b.w[2], a.w[3] ^ b.w[3]}};
    }
    bool operator==(const Mask& b) const { return w == b.w; }
    bool operator!=(const Mask& b) const { return !(*this == b); }
    bool operator<(const Mask& b) const { return w < b.w; }
    bool contains(const Mask& b) const {
        return ((w[0] & b.w[0]) == b.w[0]) && ((w[1] & b.w[1]) == b.w[1]) &&
               ((w[2] & b.w[2]) == b.w[2]) && ((w[3] & b.w[3]) == b.w[3]);
    }
};

struct MaskHash {
    size_t operator()(const Mask& m) const {
        uint64_t x = m.w[0] * 0xbf58476d1ce4e5b9ULL ^ m.w[1] * 0x94d049bb133111ebULL ^
                     m.w[2] * 0xd6e8feb86659fd93ULL ^ m.w[3] * 0x9e3779b97f4a7c15ULL;
        x ^= x >> 32;
        return x;
    }
};

struct Edge {
    int to = -1;
    float length = 0;
    float support = -1;
};
struct Node {
    std::array<Edge, 3> edges{};
    int degree = 0;
};

struct Tree {
    int n = 0;
    std::vector<Node> v;

    explicit Tree(int n = 0) : n(n), v(n) {}
    int add() {
        v.emplace_back();
        return int(v.size()) - 1;
    }
    void link(int a, int b, float l = 0, float s = -1) {
        if (v[a].degree == 3 || v[b].degree == 3) throw std::runtime_error("node degree exceeds three");
        v[a].edges[v[a].degree++] = {b, l, s};
        v[b].edges[v[b].degree++] = {a, l, s};
    }
    Edge remove(int a, int b) {
        for (int i = 0; i < v[a].degree; ++i) {
            if (v[a].edges[i].to == b) {
                Edge e = v[a].edges[i];
                v[a].edges[i] = v[a].edges[--v[a].degree];
                return e;
            }
        }
        throw std::runtime_error("missing edge");
    }
    void cut(int a, int b) {
        remove(a, b);
        remove(b, a);
    }
    Edge edge(int a, int b) const {
        for (int i = 0; i < v[a].degree; ++i)
            if (v[a].edges[i].to == b) return v[a].edges[i];
        throw std::runtime_error("missing edge");
    }
    void swap_subtrees(int u, int v2, int a, int c) {
        cut(u, a);
        cut(v2, c);
        link(u, c);
        link(v2, a);
    }
    void edge_masks(int u, int p, std::vector<Mask>& masks) const {
        if (u < n) masks[u] = Mask::leaf(u);
        else masks[u] = {};
        for (int j = 0; j < v[u].degree; ++j) {
            int c = v[u].edges[j].to;
            if (c != p) {
                edge_masks(c, u, masks);
                masks[u] = masks[u] | masks[c];
            }
        }
    }
    template<class F> void clades(int u, int p, const F& callback) const {
        auto visit = [&](auto& self, int a, int b) -> Mask {
            if (a < n) return Mask::leaf(a);
            Mask out;
            Edge up;
            for (int j = 0; j < v[a].degree; ++j) {
                Edge e = v[a].edges[j];
                if (e.to == b) up = e;
                else out = out | self(self, e.to, a);
            }
            if (b >= n) callback(out, up);
            return out;
        };
        visit(visit, u, p);
    }
};

struct Matrix {
    int n;
    std::vector<double> data;
    explicit Matrix(int n) : n(n), data(size_t(n) * n) {}
    double& operator()(int i, int j) { return data[size_t(i) * n + j]; }
    double operator()(int i, int j) const { return data[size_t(i) * n + j]; }
};

class NewickReader {
    struct Piece {
        std::string name;
        std::array<int, 3> kids{};
        int degree = 0;
        float length = 0;
        float support = -1;
    };

    const char* start;
    const char* p;
    std::vector<Piece> pieces;

    [[noreturn]] void fail(const char* why) const {
        throw std::runtime_error(std::string(why) + " at byte " + std::to_string(p - start));
    }
    void skip() {
        for (;;) {
            while (std::isspace(static_cast<unsigned char>(*p))) ++p;
            if (*p != '[') return;
            int depth = 0;
            do {
                if (*p == '[') ++depth;
                if (*p == ']') --depth;
                if (!*p) fail("unterminated Newick comment");
                ++p;
            } while (depth);
        }
    }
    std::string label() {
        skip();
        std::string r;
        if (*p == '\'') {
            ++p;
            for (;;) {
                if (!*p) fail("unterminated quoted label");
                if (*p == '\'') {
                    ++p;
                    if (*p != '\'') break;
                }
                r += *p++;
            }
        } else {
            const char* b = p;
            while (*p && !std::isspace(static_cast<unsigned char>(*p)) &&
                   *p != '(' && *p != ')' && *p != ',' && *p != ';' &&
                   *p != ':' && *p != '[' && *p != ']') ++p;
            r.assign(b, p);
        }
        return r;
    }
    int element(bool root) {
        skip();
        Piece x;
        if (*p == '(') {
            ++p;
            do {
                if (x.degree == (root ? 3 : 2)) fail("too many children");
                x.kids[x.degree++] = element(false);
                skip();
                if (*p != ',') break;
                ++p;
            } while (true);
            if (*p != ')' || x.degree < 2) fail("invalid internal node");
            ++p;
            x.name = label();
            if (!x.name.empty()) {
                char* end;
                double s = std::strtod(x.name.c_str(), &end);
                if (!*end && s >= 0 && s <= 1) x.support = float(s);
            }
        } else {
            x.name = label();
            if (x.name.empty()) fail("missing leaf label");
        }
        skip();
        if (*p == ':') {
            ++p;
            skip();
            char* end;
            double l = std::strtod(p, &end);
            if (end == p || !std::isfinite(l) || l < 0) fail("invalid edge length");
            x.length = static_cast<float>(std::min(l, 1e12));
            p = end;
        }
        int id = int(pieces.size());
        pieces.push_back(std::move(x));
        return id;
    }

public:
    explicit NewickReader(const std::string& text) : start(text.c_str()), p(start) {
        if (text.size() >= 3 && static_cast<unsigned char>(p[0]) == 0xef &&
            static_cast<unsigned char>(p[1]) == 0xbb && static_cast<unsigned char>(p[2]) == 0xbf) p += 3;
    }

    bool done() {
        skip();
        return !*p;
    }
    Tree tree(std::vector<std::string>& names, std::map<std::string, int>& ids) {
        pieces.clear();
        int root = element(true);
        skip();
        if (*p != ';') fail("missing semicolon");
        ++p;

        if (names.empty()) {
            for (const Piece& a : pieces)
                if (!a.degree) names.push_back(a.name);
            std::sort(names.begin(), names.end());
            if (names.size() < 4 || names.size() > capacity) fail("unexpected number of taxa");
            for (int i = 0; i < int(names.size()); ++i) {
                if (!ids.emplace(names[i], i).second) fail("duplicate taxon");
            }
        }
        int n = int(names.size());
        Tree t(n);
        std::vector<bool> seen(n);
        std::vector<int> nodes(pieces.size(), -1);
        for (int i = 0; i <= root; ++i) {
            const Piece& x = pieces[i];
            if (!x.degree) {
                auto f = ids.find(x.name);
                if (f == ids.end()) fail("unexpected taxon");
                if (seen[f->second]) fail("duplicate taxon");
                seen[f->second] = true;
                nodes[i] = f->second;
            } else if (i == root && x.degree == 2) {
                int a = x.kids[0], b = x.kids[1];
                t.link(nodes[a], nodes[b], pieces[a].length + pieces[b].length,
                       std::max(pieces[a].support, pieces[b].support));
            } else {
                int a = nodes[i] = t.add();
                for (int j = 0; j < x.degree; ++j) {
                    int b = x.kids[j];
                    t.link(a, nodes[b], pieces[b].length, pieces[b].support);
                }
            }
        }
        if (std::count(seen.begin(), seen.end(), true) != n) fail("missing taxon");
        return t;
    }
};

struct Input {
    std::vector<std::string> names;
    std::map<std::string, int> ids;
    std::vector<Tree> genes;
    int out = 0;
    Mask all;

    explicit Input(const char* path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("could not open input");
        std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        NewickReader reader(text);
        while (!reader.done()) genes.push_back(reader.tree(names, ids));
        if (genes.empty()) throw std::runtime_error("no input trees");
        if (ids.count("0")) out = ids.at("0");
        for (int i = 0; i < n(); ++i) all = all | Mask::leaf(i);
    }
    int n() const { return int(names.size()); }
};

inline std::string escaped(const std::string& s) {
    bool plain = !s.empty();
    for (unsigned char c : s)
        if (!std::isalnum(c) && c != '_' && c != '-' && c != '.') plain = false;
    if (plain) return s;
    std::string r = "'";
    for (char c : s) {
        r += c;
        if (c == '\'') r += c;
    }
    return r + "'";
}

inline void print_tree(const Tree& t, const Input& data, std::ostream& os) {
    auto visit = [&](auto& self, int u, int p) -> void {
        if (u < t.n) {
            os << escaped(data.names[u]);
            return;
        }
        os << '(';
        bool first = true;
        for (int j = 0; j < t.v[u].degree; ++j) {
            int c = t.v[u].edges[j].to;
            if (c == p) continue;
            if (!first) os << ',';
            first = false;
            self(self, c, u);
        }
        os << ')';
    };
    visit(visit, t.v[data.out].edges[0].to, -1);
    os << ";\n";
}

}  // namespace st
