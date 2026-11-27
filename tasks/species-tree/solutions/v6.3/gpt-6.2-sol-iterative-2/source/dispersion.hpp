#pragma once

#include "splits.hpp"

// Give moderate weight to contrasts with smaller across-locus variance.
inline void temper_dispersion(std::vector<Votes> &data, float adjustment, float clipping) {
    if (data.empty()) return;
    if (clipping>0) {
        double total=0;
        for (const auto &v:data) total+=v[0]+v[1]+v[2];
        double ave=total/data.size();
        if (ave<1e-19) return;
        double total2=0;
        for (auto &v:data) {
            double s=v[0]+v[1]+v[2];
            float w=std::pow(std::max(1.0,s/(ave*(1.0+0.40))),-clipping);
            for (float &x:v) x*=w;
            total2+=w*s;
        }
        float rescale=total/std::max(1e-25,total2);
        for (auto &v:data) for (float &x:v) x*=rescale;
    }
    if (adjustment<=0) return;
    constexpr double root3=0.86602540378443864676;
    double a=0,b=0,c=0,xm=0,ym=0;
    for (const auto &v:data) {
        double x=v[0]-0.5*(v[1]+v[2]), y=root3*(v[1]-v[2]);
        xm+=x; ym+=y;
        a+=x*x; b+=x*y; c+=y*y;
    }
    a-=xm*xm/data.size(); c-=ym*ym/data.size(); b-=xm*ym/data.size();
    double var=0.5*(a+c);
    if (var<1e-25) return;
    // Shrink the two-dimensional covariance toward isotropy before inversion.
    a=0.55*a+0.45*var; b*=0.55; c=0.55*c+0.45*var;
    double det=a*c-b*b;
    if (det<=0) return;
    constexpr double dir[3][2]={{1,0},{-0.5,root3},{-0.5,-root3}};
    double divisor[3];
    for (int j=0;j<3;++j) {
        double p=dir[j][0], q=dir[j][1];
        divisor[j]=std::sqrt(var/(det*(p*p*c-2*p*q*b+q*q*a))) * (2.0/3.0);
    }
    for (auto &v:data) {
        double s=(v[0]+v[1]+v[2])/3, x=v[0]-0.5*(v[1]+v[2]), y=root3*(v[1]-v[2]);
        double f=c*x-b*y,g=a*y-b*x;
        double unbias[3], m=0;
        for (int j=0;j<3;++j) {
            unbias[j]=divisor[j]*(dir[j][0]*f+dir[j][1]*g);
            m+=unbias[j]/3;
        }
        for (int j=0;j<3;++j)
            v[j]=(1-adjustment)*v[j]+adjustment*(s+unbias[j]-m);
    }
}
