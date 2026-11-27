#include "consensus.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: infer INPUT OUTPUT");
        GeneSet set(argv[1]);
        if (set.n > QuartetKernel::MAXN) throw std::runtime_error("too many taxa");

        unsigned seed = 1;
        if (const char *s = std::getenv("SEED")) {
            char *end = nullptr;
            unsigned long value = std::strtoul(s, &end, 10);
            if (end != s && *end == '\0') seed = static_cast<unsigned>(value);
        }
        std::mt19937 rng(seed);

        std::vector<GeneDistances> distances;
        distances.reserve(set.genes.size());
        for (const auto &gene : set.genes) distances.emplace_back(gene);
        Tree guide = join_tree(aggregate(distances, Metric::Hybrid), set.n, Join::BioNJ);
        Tree result = combined_inference(set, distances, guide, rng);
        write_tree(argv[2], result, set.names);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "infer: " << e.what() << '\n';
        return 1;
    }
}
