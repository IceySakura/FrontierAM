#include "optimizer.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" void scipy_openblas_set_num_threads_(const int*);

bool debug=false;
int main(int argc,char** argv){
    const char* input=nullptr,*output=nullptr;
    for(int j=1;j+1<argc;++j) {
        if(std::strcmp(argv[j],"--input")==0) input=argv[++j];
        else if(std::strcmp(argv[j],"--output")==0) output=argv[++j];
    }
    if(!input || !output){
        std::fputs("usage: run.sh --input INPUT.npz --output OUTPUT.npz\n",stderr);
        return 2;
    }
    debug=std::getenv("CONIC_DEBUG")!=nullptr;
    const int one=1;
    scipy_openblas_set_num_threads_(&one);
    Result r;
    try {
        auto m=read_npz(input);
        r=optimize(m);
    } catch(const std::exception& ex){
        if(debug) std::fprintf(stderr,"unresolved: %s\n",ex.what());
    }
    try {
        write_npz(output,r);
    } catch(const std::exception& ex) {
        std::fprintf(stderr,"%s\n",ex.what());
        return 1;
    }
    return 0;
}
