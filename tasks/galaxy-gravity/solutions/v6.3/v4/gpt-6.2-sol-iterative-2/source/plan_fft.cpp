#include <fftw3.h>
#include <cstdio>
#include "fft_support.hpp"

int main() {
    const int sizes[] = {12,14,15,16,18,20,21,24,25,27,28,30,32,36,40};
    fftw_set_timelimit(.44);
    for(int n:sizes){
        double *r=fftw_alloc_real(n*n*n);
        fftw_complex *q=fftw_alloc_complex(n*n*(n/2+1));
        if(!r||!q)return 1;
        auto f=fftw_plan_dft_r2c_3d(n,n,n,r,q,FFTW_PATIENT);
        auto b=fftw_plan_dft_c2r_3d(n,n,n,q,r,FFTW_PATIENT);
        if(!f||!b)return 1;
        fftw_destroy_plan(f);
        fftw_destroy_plan(b);
        fftw_free(r);
        fftw_free(q);
    }
    fftw_set_timelimit(.16);
    for(int p=7;p<=20;++p){
        const int n=galaxy_fft::dimension(p);
        double *r=fftw_alloc_real(n*n*n);
        fftw_complex *q=fftw_alloc_complex(n*n*(n/2+1));
        if(!r||!q)return 1;
        {
            galaxy_fft::Pruned tr(p,n,r,q,FFTW_PATIENT);
            if(!tr.okay())return 1;
        }
        fftw_free(r);
        fftw_free(q);
    }
    char *w=fftw_export_wisdom_to_string();
    if(!w)return 1;
    int ok=std::printf("static const char fft_wisdom[] = R\"WISDOM(%s)WISDOM\";\n",w);
    fftw_free(w);
    return ok<0;
}
