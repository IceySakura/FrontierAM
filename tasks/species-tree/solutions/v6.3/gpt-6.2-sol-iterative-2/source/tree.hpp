#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

struct Arc {
    int to = -1;
    float length = 0;
    float support = -1;
};

struct Vertex {
    std::array<Arc, 3> edges{};
    int degree = 0;
};

struct Tree {
    int n = 0;
    int used = 0;
    std::vector<Vertex> v;
    explicit Tree(int taxa = 0) : n(taxa), used(taxa), v(2 * taxa) {}

    int node() { return used++; }
    void link(int a, int b, float len = 0, float sup = -1) {
        if (v[a].degree == 3 || v[b].degree == 3)
            throw std::runtime_error("Newick tree is not binary");
        v[a].edges[v[a].degree++] = {b, len, sup};
        v[b].edges[v[b].degree++] = {a, len, sup};
    }
    Arc arc(int a, int b) const {
        for (int i = 0; i < v[a].degree; ++i)
            if (v[a].edges[i].to == b) return v[a].edges[i];
        throw std::logic_error("vertices are not adjacent");
    }
    void unlink_one(int a, int b) {
        for (int i = 0; i < v[a].degree; ++i) if (v[a].edges[i].to == b) {
            v[a].edges[i] = v[a].edges[--v[a].degree];
            return;
        }
        throw std::logic_error("vertices are not adjacent");
    }
    void unlink(int a, int b) { unlink_one(a, b); unlink_one(b, a); }
    void nni(int u, int vtx, int a, int b) {
        unlink(u, a);
        unlink(vtx, b);
        link(u, b);
        link(vtx, a);
    }
};

struct ParsedVertex {
    std::string name;
    std::vector<int> children;
    float length = 0;
    float support = -1;
};

class Newick {
    const std::string &text;
    size_t pos = 0;

    void skip() {
        while (pos < text.size()) {
            if (std::isspace(static_cast<unsigned char>(text[pos]))) {
                ++pos;
            } else if (text[pos] == '[') {
                int depth = 1;
                while (++pos < text.size() && depth) {
                    depth += (text[pos] == '[') - (text[pos] == ']');
                }
            } else break;
        }
    }
    char peek() { skip(); return pos < text.size() ? text[pos] : '\0'; }
    void expect(char c) {
        if (peek() != c) throw std::runtime_error("malformed Newick tree");
        ++pos;
    }
    std::string label() {
        if (peek() == '\'') {
            std::string s;
            ++pos;
            while (pos < text.size()) {
                char c = text[pos++];
                if (c == '\'') {
                    if (pos == text.size() || text[pos] != '\'') return s;
                    ++pos;
                }
                s.push_back(c);
            }
            throw std::runtime_error("unterminated quoted label");
        }
        size_t start = pos;
        while (pos < text.size() && !std::isspace(static_cast<unsigned char>(text[pos])) &&
               std::string_view("[](),:;").find(text[pos]) == std::string_view::npos)
            ++pos;
        return text.substr(start, pos - start);
    }
    static float number(const std::string &s, float fallback) {
        if (s.empty()) return fallback;
        char *end = nullptr;
        float f = std::strtof(s.c_str(), &end);
        if (*end || !std::isfinite(f)) return fallback;
        return f;
    }
    int parse_node() {
        ParsedVertex a;
        if (peek() == '(') {
            ++pos;
            a.children.push_back(parse_node());
            while (peek() == ',') {
                ++pos;
                a.children.push_back(parse_node());
            }
            expect(')');
            a.support = number(label(), -1.0f);
        } else {
            a.name = label();
            if (a.name.empty()) throw std::runtime_error("missing leaf name");
        }
        if (peek() == ':') {
            ++pos;
            a.length = std::max(0.0f, number(label(), 0.0f));
        }
        int i = static_cast<int>(nodes.size());
        nodes.push_back(std::move(a));
        return i;
    }
public:
    std::vector<ParsedVertex> nodes;
    int root;
    explicit Newick(const std::string &s) : text(s) {
        root = parse_node();
        expect(';');
        if (peek()) throw std::runtime_error("extra text after Newick tree");
    }
};

struct GeneSet {
    int n = 0;
    std::vector<std::string> names;
    std::vector<Tree> genes;

    explicit GeneSet(const char *path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("could not open input");
        std::unordered_map<std::string, int> ids;
        for (std::string line; std::getline(in, line); ) {
            if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
            Newick t(line);
            if (genes.empty()) {
                for (const auto &p : t.nodes)
                    if (p.children.empty()) names.push_back(p.name);
                auto out = std::find(names.begin(), names.end(), "0");
                if (out != names.end()) std::iter_swap(out, names.begin());
                n = static_cast<int>(names.size());
                if (n < 4 || n > 3000) throw std::runtime_error("invalid number of taxa");
                for (int i = 0; i < n; ++i) {
                    if (!ids.emplace(names[i], i).second)
                        throw std::runtime_error("duplicate leaf");
                }
            }
            Tree g(n);
            std::vector<int> ip(t.nodes.size(), -1);
            std::vector<bool> seen(n, false);
            bool suppress = t.nodes[t.root].children.size() == 2;
            for (size_t i = 0; i < t.nodes.size(); ++i) {
                const auto &p = t.nodes[i];
                if (p.children.empty()) {
                    auto it = ids.find(p.name);
                    if (it == ids.end() || seen[it->second])
                        throw std::runtime_error("gene trees must have the same leaf set");
                    ip[i] = it->second;
                    seen[it->second] = true;
                } else if (static_cast<int>(i) != t.root || !suppress) {
                    if (g.used >= 2 * n - 2) throw std::runtime_error("too many internal vertices");
                    ip[i] = g.node();
                    for (int j : p.children)
                        g.link(ip[i], ip[j], t.nodes[j].length, t.nodes[j].support);
                } else {
                    const auto &a = t.nodes[p.children[0]];
                    const auto &b = t.nodes[p.children[1]];
                    g.link(ip[p.children[0]], ip[p.children[1]], a.length + b.length,
                           std::max(a.support, b.support));
                }
            }
            if (g.used != 2 * n - 2)
                throw std::runtime_error("invalid number of internal vertices");
            for (int i = 0; i < g.used; ++i)
                if (g.v[i].degree != (i < n ? 1 : 3))
                    throw std::runtime_error("invalid vertex degree");
            genes.push_back(std::move(g));
        }
        if (genes.empty()) throw std::runtime_error("empty input");
    }
};

inline void write_name(std::ostream &out, const std::string &name) {
    if (name.find_first_of(" \t\r\n'[](),:;") == std::string::npos) out << name;
    else {
        out << '\'';
        for (char c : name) {
            if (c == '\'') out << '\'';
            out << c;
        }
        out << '\'';
    }
}

inline void write_subtree(std::ostream &out, const Tree &t, int u, int p,
                          const std::vector<std::string> &names) {
    if (u < t.n) { write_name(out, names[u]); return; }
    out << '(';
    bool first = true;
    for (int i = 0; i < t.v[u].degree; ++i) {
        int x = t.v[u].edges[i].to;
        if (x == p) continue;
        if (!first) out << ',';
        first = false;
        write_subtree(out, t, x, u, names);
    }
    out << ')';
}

inline void write_tree(const char *path, const Tree &t,
                       const std::vector<std::string> &names) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("could not open output");
    write_subtree(out, t, t.v[0].edges[0].to, -1, names);
    out << ";\n";
    out.close();
    if (!out) throw std::runtime_error("could not write output");
}
