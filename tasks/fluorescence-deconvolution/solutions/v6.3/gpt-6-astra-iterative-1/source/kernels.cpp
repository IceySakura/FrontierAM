#include <algorithm>
#include <cmath>
#include <vector>

extern "C" {

void separable_conv(const float *input, const float *kernels, int radius,
                    int nz, int ny, int nx, float *tmp1, float *tmp2, float *out) {
    const int n = nz*ny*nx, ksize=2*radius+1;
    std::fill(tmp1, tmp1+n, 0.0f);
    std::fill(tmp2, tmp2+n, 0.0f);
    std::fill(out, out+n, 0.0f);
    for (int z=0; z<nz; ++z) {
        float *dst=tmp1+z*ny*nx;
        for (int j=std::max(-radius,-z); j<=std::min(radius,nz-1-z); ++j) {
            const float h=kernels[radius-j];
            const float *src=input+(z+j)*ny*nx;
            for (int i=0; i<ny*nx; ++i) dst[i]+=h*src[i];
        }
    }
    for (int z=0; z<nz; ++z) {
        for (int y=0; y<ny; ++y) {
            float *dst=tmp2+(z*ny+y)*nx;
            for (int j=std::max(-radius,-y); j<=std::min(radius,ny-1-y); ++j) {
                const float h=kernels[ksize+radius-j];
                const float *src=tmp1+(z*ny+y+j)*nx;
                for (int i=0; i<nx; ++i) dst[i]+=h*src[i];
            }
        }
    }
    for (int row=0; row<nz*ny; ++row) {
        float *dst=out+row*nx;
        const float *src=tmp2+row*nx;
        for (int j=-radius; j<=radius; ++j) {
            const float h=kernels[2*ksize+radius-j];
            const int start=std::max(0,-j), end=std::min(nx,nx-j);
            for (int i=start; i<end; ++i) dst[i]+=h*src[i+j];
        }
    }
}

void score_lines(const double *coords, const double *values, const double *directions,
                 const double *confidence, int count, const double *anchors,
                 const double *vectors, int candidates, double *scores, double *bounds) {
    for (int k=0;k<candidates;++k) {
        const double *a=anchors+3*k, *v=vectors+3*k;
        double hist[128]={0}, smooth[128], maximum=0;
        for (int i=0;i<count;++i) {
            const double *c=coords+3*i, *d=directions+3*i;
            const double z=c[0]-a[0], y=c[1]-a[1], x=c[2]-a[2];
            const double t=z*v[0]+y*v[1]+x*v[2];
            const double d2=std::max(0.0,z*z+y*y+x*x-t*t);
            const double dp=d[0]*v[0]+d[1]*v[1]+d[2]*v[2];
            const double weight=values[i]*std::exp(-0.5*d2)*(confidence[i]*dp*dp*dp*dp+0.1);
            const int bin=std::max(0,std::min(127,int(std::floor(t+64))));
            hist[bin]+=weight;
        }
        for (int i=0;i<128;++i) {
            smooth[i]=0.5*hist[i]+(i?0.25*hist[i-1]:0)+(i<127?0.25*hist[i+1]:0);
            maximum=std::max(maximum,smooth[i]);
        }
        double best=-1;
        int bestlo=0,besthi=0;
        for (int i=0;i<128;) {
            if (smooth[i]<=0.25*maximum) {++i;continue;}
            const int lo=i;
            double score=0;
            while(i<128 && smooth[i]>0.25*maximum) score+=smooth[i++];
            const int hi=i-1;
            if (hi-lo>=6 && score>best) {best=score;bestlo=lo;besthi=hi;}
        }
        scores[k]=best;
        bounds[2*k]=bestlo-64;
        bounds[2*k+1]=besthi-64;
    }
}

void render_model(const double *params, int count, int nz, int ny, int nx,
                  float *out, float *components) {
    const int n = nz * ny * nx;
    std::fill(out, out + n, 0.0f);
    for (int k = 0; k < count; ++k) {
        const double *p = params + 8*k;
        const float cz=p[0], cy=p[1], cx=p[2], dz=p[3], dy=p[4], dx=p[5];
        const float invlen=1.0f/std::max(dz*dz+dy*dy+dx*dx, 1e-16f);
        const float invs2=std::exp(-2.0*p[6]), amp=std::exp(p[7]);
        float *component = components + k*n;
        for (int z=0; z<nz; ++z) {
            const float rz=z-cz;
            for (int y=0; y<ny; ++y) {
                const float ry=y-cy;
                const int start=(z*ny+y)*nx;
                for (int x=0; x<nx; ++x) {
                    const float rx=x-cx;
                    const float t=std::min(1.0f, std::max(-1.0f, (rz*dz+ry*dy+rx*dx)*invlen));
                    const float ez=rz-t*dz, ey=ry-t*dy, ex=rx-t*dx;
                    const float v=amp*std::exp(-0.5f*invs2*(ez*ez+ey*ey+ex*ex));
                    component[start+x]=v;
                    out[start+x]+=v;
                }
            }
        }
    }
}

void model_jacobian(const double *params, int count, int nz, int ny, int nx,
                    const float *components, float *jacobian) {
    const int n=nz*ny*nx;
    for (int k=0;k<count;++k) {
        const double *p=params+8*k;
        const float cz=p[0],cy=p[1],cx=p[2],dz=p[3],dy=p[4],dx=p[5];
        const float il=1/std::max(dz*dz+dy*dy+dx*dx,1e-16f), is=std::exp(-2*p[6]);
        const float *component=components+k*n;
        float *j0=jacobian+8*k*n,*j1=j0+n,*j2=j1+n,*j3=j2+n;
        float *j4=j3+n,*j5=j4+n,*j6=j5+n,*j7=j6+n;
        for (int z=0;z<nz;++z) {
            const float rz=z-cz;
            for (int y=0;y<ny;++y) {
                const float ry=y-cy;
                for (int x=0;x<nx;++x) {
                    const int i=(z*ny+y)*nx+x;
                    const float rx=x-cx;
                    const float t=std::min(1.0f,std::max(-1.0f,(rz*dz+ry*dy+rx*dx)*il));
                    const float ez=rz-t*dz,ey=ry-t*dy,ex=rx-t*dx;
                    const float v=component[i],w=v*is;
                    j0[i]=w*ez;j1[i]=w*ey;j2[i]=w*ex;
                    j3[i]=w*t*ez;j4[i]=w*t*ey;j5[i]=w*t*ex;
                    j6[i]=w*(ez*ez+ey*ey+ex*ex);j7[i]=v;
                }
            }
        }
    }
}

void batch_convolve(const float *input, const float *kernels, int radius,
                    int count, int nz, int ny, int nx, float *tmp1, float *tmp2, float *out) {
    const int size=nz*ny*nx;
    for (int k=0;k<count;++k)
        separable_conv(input+k*size,kernels,radius,nz,ny,nx,tmp1,tmp2,out+k*size);
}

void render_points(const double *params, int count, int nz, int ny, int nx,
                   const float *kernels, int radius, int forward, int derivatives,
                   float *out, float *jac) {
    const int shape[3]={nz,ny,nx}, n=nz*ny*nx, ksize=2*radius+1;
    std::fill(out,out+n,0.0f);
    for (int k=0;k<count;++k) {
        const double *p=params+5*k;
        const double width=std::exp(p[3]), amp=std::exp(p[4]);
        std::vector<float> g[3][3];
        for (int a=0;a<3;++a) {
            std::vector<float> raw[3];
            for (int d=0;d<3;++d) {
                raw[d].resize(shape[a]);
                g[a][d].resize(shape[a]);
            }
            for (int i=0;i<shape[a];++i) {
                const double r=(i-p[a])/width, v=std::exp(-0.5*r*r);
                raw[0][i]=v; raw[1][i]=v*r/width; raw[2][i]=v*r*r;
            }
            for (int d=0;d<3;++d) {
                if (!forward) {
                    g[a][d]=raw[d];
                    continue;
                }
                for (int i=0;i<shape[a];++i) {
                    float v=0;
                    for (int j=std::max(-radius,-i);j<=std::min(radius,shape[a]-1-i);++j)
                        v+=kernels[a*ksize+radius-j]*raw[d][i+j];
                    g[a][d][i]=v;
                }
            }
        }
        float *j0=jac+(5*k)*n, *j1=j0+n, *j2=j1+n, *j3=j2+n, *j4=j3+n;
        for (int z=0;z<nz;++z) {
            for (int y=0;y<ny;++y) {
                const int start=(z*ny+y)*nx;
                const float a=amp*g[0][0][z]*g[1][0][y];
                const float az=amp*g[0][1][z]*g[1][0][y];
                const float ay=amp*g[0][0][z]*g[1][1][y];
                const float as=amp*(g[0][2][z]*g[1][0][y]+g[0][0][z]*g[1][2][y]);
                for (int x=0;x<nx;++x) {
                    const int i=start+x;
                    const float v=a*g[2][0][x];
                    out[i]+=v;
                    if (derivatives) {
                        j0[i]=az*g[2][0][x];
                        j1[i]=ay*g[2][0][x];
                        j2[i]=a*g[2][1][x];
                        j3[i]=as*g[2][0][x]+a*g[2][2][x];
                        j4[i]=v;
                    }
                }
            }
        }
    }
}

}
