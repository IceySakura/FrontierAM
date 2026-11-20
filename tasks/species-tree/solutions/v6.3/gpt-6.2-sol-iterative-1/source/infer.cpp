#include "mosaic.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: run.sh INPUT OUTPUT\n";
        return 2;
    }
    try {
        st::Input data(argv[1]);
        st::Tree inferred = st::mosaic_tree(data);
        std::ofstream out(argv[2]);
        if (!out) throw std::runtime_error("could not open output");
        st::print_tree(inferred, data, out);
        out.close();
        if (!out) throw std::runtime_error("could not write output");
    } catch (const std::exception& e) {
        std::cerr << "inference: " << e.what() << '\n';
        return 1;
    }
}
