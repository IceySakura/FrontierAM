#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Branch {
    int other;
    double length;
    double support;
};

struct Node {
    std::string label;
    std::vector<Branch> edges;
};

struct Tree {
    std::vector<Node> nodes;
    int root = -1;

    int add_node(std::string label = {}) {
        nodes.push_back({std::move(label), {}});
        return static_cast<int>(nodes.size()) - 1;
    }

    void connect(int a, int b, double length, double support) {
        nodes[a].edges.push_back({b, length, support});
        nodes[b].edges.push_back({a, length, support});
    }

    void remove_edge(int a, int b) {
        auto& edges = nodes[a].edges;
        edges.erase(std::remove_if(edges.begin(), edges.end(),
                                  [b](const Branch& e) { return e.other == b; }),
                    edges.end());
    }

    void suppress_root() {
        if (root < 0 || nodes[root].edges.size() != 2) {
            return;
        }
        const Branch a = nodes[root].edges[0];
        const Branch b = nodes[root].edges[1];
        remove_edge(a.other, root);
        remove_edge(b.other, root);
        nodes[root].edges.clear();
        double support = -1.0;
        if (a.support >= 0.0 && b.support >= 0.0) {
            support = 0.5 * (a.support + b.support);
        } else {
            support = std::max(a.support, b.support);
        }
        connect(a.other, b.other, a.length + b.length, support);
        root = a.other;
    }
};

class Parser {
public:
    explicit Parser(std::string input) : input_(std::move(input)) {}

    bool has_next() {
        skip();
        return position_ < input_.size();
    }

    Tree parse() {
        Tree tree;
        const Parsed top = subtree(tree);
        tree.root = top.node;
        expect(';');
        tree.suppress_root();
        return tree;
    }

private:
    struct Parsed {
        int node;
        double length;
        double support;
    };

    std::string input_;
    std::size_t position_ = 0;

    [[noreturn]] void fail(const std::string& message) const {
        throw std::runtime_error(message + " at byte " +
                                 std::to_string(position_));
    }

    void skip() {
        while (position_ < input_.size()) {
            if (std::isspace(static_cast<unsigned char>(input_[position_]))) {
                ++position_;
            } else if (input_[position_] == '[') {
                int depth = 1;
                ++position_;
                while (position_ < input_.size() && depth > 0) {
                    const char ch = input_[position_++];
                    if (ch == '[') {
                        ++depth;
                    } else if (ch == ']') {
                        --depth;
                    }
                }
                if (depth != 0) {
                    fail("Unterminated Newick comment");
                }
            } else {
                break;
            }
        }
    }

    bool take(char ch) {
        skip();
        if (position_ < input_.size() && input_[position_] == ch) {
            ++position_;
            return true;
        }
        return false;
    }

    void expect(char ch) {
        if (!take(ch)) {
            fail(std::string("Expected '") + ch + "'");
        }
    }

    static bool delimiter(char ch) {
        return ch == '(' || ch == ')' || ch == ',' || ch == ':' ||
               ch == ';' || ch == '[' || ch == ']' ||
               std::isspace(static_cast<unsigned char>(ch));
    }

    bool has_label() {
        skip();
        return position_ < input_.size() && !delimiter(input_[position_]);
    }

    std::string label() {
        skip();
        if (position_ >= input_.size()) {
            fail("Missing label");
        }
        std::string result;
        if (input_[position_] == '\'') {
            ++position_;
            while (position_ < input_.size()) {
                const char ch = input_[position_++];
                if (ch != '\'') {
                    result.push_back(ch);
                } else if (position_ < input_.size() &&
                           input_[position_] == '\'') {
                    result.push_back('\'');
                    ++position_;
                } else {
                    return result;
                }
            }
            fail("Unterminated quoted label");
        }
        while (position_ < input_.size() && !delimiter(input_[position_])) {
            result.push_back(input_[position_++]);
        }
        if (result.empty()) {
            fail("Missing label");
        }
        return result;
    }

    static bool number(const std::string& text, double& result) {
        if (text.empty()) {
            return false;
        }
        char* end = nullptr;
        result = std::strtod(text.c_str(), &end);
        return end == text.c_str() + text.size() && std::isfinite(result);
    }

    Parsed subtree(Tree& tree) {
        int node;
        double support = -1.0;
        if (take('(')) {
            node = tree.add_node();
            int children = 0;
            do {
                const Parsed child = subtree(tree);
                tree.connect(node, child.node, child.length, child.support);
                ++children;
            } while (take(','));
            expect(')');
            if (children < 2) {
                fail("Internal node has fewer than two children");
            }
            if (has_label()) {
                const std::string value = label();
                double parsed = 0.0;
                if (number(value, parsed) && parsed >= 0.0 && parsed <= 1.0) {
                    support = parsed;
                }
            }
        } else {
            node = tree.add_node(label());
            if (tree.nodes[node].label.empty()) {
                fail("Empty leaf label");
            }
        }
        double length = 0.0;
        if (take(':')) {
            const std::string value = label();
            if (!number(value, length) || length < 0.0) {
                fail("Invalid branch length");
            }
        }
        return {node, length, support};
    }
};

std::map<std::string, int> taxa_of(const Tree& tree) {
    std::map<std::string, int> taxa;
    for (const Node& node : tree.nodes) {
        if (node.edges.empty()) {
            continue;
        }
        if (node.label.empty()) {
            if (node.edges.size() != 3) {
                throw std::runtime_error("Input is not an unrooted binary tree");
            }
        } else {
            if (node.edges.size() != 1 ||
                !taxa.emplace(node.label, 0).second) {
                throw std::runtime_error("Invalid or duplicate leaf label");
            }
        }
    }
    int index = 0;
    for (auto& entry : taxa) {
        entry.second = index++;
    }
    return taxa;
}

double edge_weight(const Tree& tree, int a, const Branch& branch) {
    if (!tree.nodes[a].label.empty() ||
        !tree.nodes[branch.other].label.empty()) {
        return 1.0;
    }
    if (branch.support < 0.0) {
        return 1.0;
    }
    const double confidence =
        std::max(0.0, std::min(1.0, 1.5 * branch.support - 0.5));
    return 0.05 + 0.95 * confidence;
}

void accumulate_distances(const Tree& tree,
                          const std::map<std::string, int>& taxa,
                          std::vector<double>& total) {
    const int n = static_cast<int>(taxa.size());
    const int m = static_cast<int>(tree.nodes.size());
    std::vector<int> leaf_index(m, -1);
    std::vector<int> leaf_node(n, -1);
    for (int node = 0; node < m; ++node) {
        if (!tree.nodes[node].label.empty()) {
            const auto found = taxa.find(tree.nodes[node].label);
            if (found == taxa.end()) {
                throw std::runtime_error("Gene trees have different taxa");
            }
            leaf_index[node] = found->second;
            leaf_node[found->second] = node;
        }
    }

    struct Visit {
        int node;
        int parent;
        double distance;
    };
    std::vector<Visit> stack;
    stack.reserve(m);
    std::vector<double> locus(static_cast<std::size_t>(n) * n, 0.0);
    double sum = 0.0;
    for (int source = 0; source < n; ++source) {
        stack.clear();
        stack.push_back({leaf_node[source], -1, 0.0});
        while (!stack.empty()) {
            const Visit current = stack.back();
            stack.pop_back();
            const int destination = leaf_index[current.node];
            if (destination > source) {
                locus[source * n + destination] = current.distance;
                locus[destination * n + source] = current.distance;
                sum += current.distance;
            }
            for (const Branch& branch : tree.nodes[current.node].edges) {
                if (branch.other != current.parent) {
                    stack.push_back(
                        {branch.other, current.node,
                         current.distance +
                             edge_weight(tree, current.node, branch)});
                }
            }
        }
    }

    const double mean = sum / (0.5 * n * (n - 1));
    if (!(mean > 0.0) || !std::isfinite(mean)) {
        throw std::runtime_error("Degenerate gene-tree distances");
    }
    for (std::size_t k = 0; k < total.size(); ++k) {
        total[k] += locus[k] / mean;
    }
}

struct OutputNode {
    int leaf = -1;
    int left = -1;
    int right = -1;
};

std::string quote_label(const std::string& value) {
    bool quote = value.empty();
    for (const unsigned char ch : value) {
        if (std::isspace(ch) || ch == '\'' || ch == '(' || ch == ')' ||
            ch == '[' || ch == ']' || ch == ',' || ch == ':' || ch == ';') {
            quote = true;
        }
    }
    if (!quote) {
        return value;
    }
    std::string result = "'";
    for (char ch : value) {
        result.push_back(ch);
        if (ch == '\'') {
            result.push_back('\'');
        }
    }
    result.push_back('\'');
    return result;
}

void write_subtree(std::ostream& out, int node,
                   const std::vector<OutputNode>& topology,
                   const std::vector<std::string>& labels) {
    const OutputNode& current = topology[node];
    if (current.leaf >= 0) {
        out << quote_label(labels[current.leaf]);
        return;
    }
    out << '(';
    write_subtree(out, current.left, topology, labels);
    out << ',';
    write_subtree(out, current.right, topology, labels);
    out << ')';
}

void infer_and_write(const std::vector<double>& average,
                     const std::vector<std::string>& labels,
                     const std::string& output) {
    const int n = static_cast<int>(labels.size());
    const int capacity = 2 * n;
    std::vector<double> distance(
        static_cast<std::size_t>(capacity) * capacity, 0.0);
    auto at = [&](int a, int b) -> double& {
        return distance[static_cast<std::size_t>(a) * capacity + b];
    };
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            at(i, j) = average[i * n + j];
        }
    }

    std::vector<int> active(n);
    std::iota(active.begin(), active.end(), 0);
    std::vector<OutputNode> topology;
    topology.reserve(capacity);
    for (int i = 0; i < n; ++i) {
        topology.push_back({i, -1, -1});
    }
    std::vector<double> row_sum(capacity);

    while (active.size() > 3) {
        const int count = static_cast<int>(active.size());
        for (int a : active) {
            double sum = 0.0;
            for (int b : active) {
                sum += at(a, b);
            }
            row_sum[a] = sum;
        }

        double best = -std::numeric_limits<double>::infinity();
        int first = -1;
        int second = -1;
        for (int i = 0; i < count; ++i) {
            const int a = active[i];
            for (int j = i + 1; j < count; ++j) {
                const int b = active[j];
                const double corrected =
                    (count - 2.0) * at(a, b) - row_sum[a] - row_sum[b];
                // The largest corrected separation selects the shared split.
                if (corrected > best + 1e-12) {
                    best = corrected;
                    first = a;
                    second = b;
                }
            }
        }
        if (first < 0 || second < 0) {
            throw std::runtime_error("Could not choose a tree join");
        }

        const int parent = static_cast<int>(topology.size());
        topology.push_back({-1, first, second});
        for (int other : active) {
            if (other != first && other != second) {
                const double updated =
                    0.5 * (at(first, other) + at(second, other) -
                           at(first, second));
                at(parent, other) = at(other, parent) = updated;
            }
        }
        active.erase(std::remove_if(active.begin(), active.end(),
                                    [&](int node) {
                                        return node == first || node == second;
                                    }),
                     active.end());
        active.push_back(parent);
    }

    std::ofstream out(output, std::ios::out | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Cannot open output: " + output);
    }
    out << '(';
    for (std::size_t i = 0; i < active.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        write_subtree(out, active[i], topology, labels);
    }
    out << ");\n";
    out.close();
    if (!out) {
        throw std::runtime_error("Failed writing output: " + output);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error("Usage: infer INPUT OUTPUT");
        }
        const auto started = std::chrono::steady_clock::now();
        std::ifstream input(argv[1], std::ios::in | std::ios::binary);
        if (!input) {
            throw std::runtime_error(std::string("Cannot open input: ") +
                                     argv[1]);
        }
        std::string text((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
        if (input.bad()) {
            throw std::runtime_error("Failed reading input");
        }
        Parser parser(std::move(text));
        std::map<std::string, int> taxa;
        std::vector<double> average;
        std::vector<std::string> labels;
        int genes = 0;
        while (parser.has_next()) {
            Tree tree = parser.parse();
            const auto observed = taxa_of(tree);
            if (genes == 0) {
                taxa = observed;
                if (taxa.size() < 4) {
                    throw std::runtime_error("At least four taxa are required");
                }
                labels.resize(taxa.size());
                for (const auto& entry : taxa) {
                    labels[entry.second] = entry.first;
                }
                average.assign(taxa.size() * taxa.size(), 0.0);
            } else if (observed != taxa) {
                throw std::runtime_error("Gene trees have different leaf sets");
            }
            accumulate_distances(tree, taxa, average);
            ++genes;
        }
        if (genes == 0) {
            throw std::runtime_error("Input contains no gene trees");
        }
        for (double& value : average) {
            value /= genes;
        }
        infer_and_write(average, labels, argv[2]);
        const double elapsed =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started)
                .count();
        std::cerr << "genes=" << genes << " taxa=" << taxa.size()
                  << " seconds=" << std::fixed << std::setprecision(3)
                  << elapsed << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "infer: " << error.what() << '\n';
        return 1;
    }
}
