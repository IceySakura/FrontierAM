#pragma once
#include <algorithm>
#include <iterator>
#include <fftw3.h>

namespace galaxy_fft {

inline int dimension(int p) {
    static const int sizes[]={12,14,16,18,20,21,24,25,27,30,32,36,40};
    return *std::lower_bound(std::begin(sizes),std::end(sizes),2*p-1);
}

struct Pruned {
    fftw_plan xf=nullptr, yf=nullptr, zf=nullptr;
    fftw_plan xb=nullptr, yb=nullptr, zb=nullptr;
    Pruned(int p,int n,double *real,fftw_complex *field,unsigned flags){
        const int c=n/2+1;
        fftw_iodim x={n,1,1};
        fftw_iodim groups_x[2]={{p,n*p,n*c},{p,n,c}};
        fftw_iodim groups_xb[2]={{p,n*c,n*p},{p,c,n}};
        xf=fftw_plan_guru_dft_r2c(1,&x,2,groups_x,real,field,flags);
        xb=fftw_plan_guru_dft_c2r(1,&x,2,groups_xb,field,real,flags);
        fftw_iodim y={n,c,c};
        fftw_iodim groups_y[2]={{p,n*c,n*c},{c,1,1}};
        yf=fftw_plan_guru_dft(1,&y,2,groups_y,field,field,FFTW_FORWARD,flags);
        yb=fftw_plan_guru_dft(1,&y,2,groups_y,field,field,FFTW_BACKWARD,flags);
        fftw_iodim z={n,n*c,n*c};
        fftw_iodim groups_z={n*c,1,1};
        zf=fftw_plan_guru_dft(1,&z,1,&groups_z,field,field,FFTW_FORWARD,flags);
        zb=fftw_plan_guru_dft(1,&z,1,&groups_z,field,field,FFTW_BACKWARD,flags);
    }
    bool okay() const {return xf&&yf&&zf&&xb&&yb&&zb;}
    void forward() const {fftw_execute(xf);fftw_execute(yf);fftw_execute(zf);}
    void backward() const {fftw_execute(zb);fftw_execute(yb);fftw_execute(xb);}
    ~Pruned(){
        for(const auto p:{xf,yf,zf,xb,yb,zb})if(p)fftw_destroy_plan(p);
    }
    Pruned(const Pruned&)=delete;
    Pruned& operator=(const Pruned&)=delete;
};

} // namespace galaxy_fft
