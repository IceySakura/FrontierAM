#include <string.h>
#include <math.h>

static void conv_axis(const float *in, float *out, int nz, int ny, int nx,
                      const float *k, int kk, int axis, int adj) {
    int c=kk/2, strides[3]={ny*nx,nx,1}, dims[3]={nz,ny,nx};
    int s=strides[axis], n=dims[axis], total=nz*ny*nx, outer=total/n;
    for (int o=0;o<outer;++o) {
        int rem=o, base=0;
        for (int a=0;a<3;++a) {
            if (a==axis) continue;
            int coord=rem%dims[a]; rem/=dims[a]; base+=coord*strides[a];
        }
        for (int i=0;i<n;++i) {
            float acc=0.f;
            for (int t=0;t<kk;++t) {
                int j=adj ? i-(t-c) : i+(t-c);
                if (j>=0 && j<n) acc+=k[t]*in[base+j*s];
            }
            out[base+i*s]=acc;
        }
    }
}

static void conv_sep(const float *in, float *out, float *t1, float *t2,
                     int nz, int ny, int nx, const float *kz, const float *ky,
                     const float *kx, int kk, int adj) {
    conv_axis(in,t1,nz,ny,nx,kz,kk,0,adj);
    conv_axis(t1,t2,nz,ny,nx,ky,kk,1,adj);
    conv_axis(t2,out,nz,ny,nx,kx,kk,2,adj);
}

void rl_multi(const float *y, float *x10, float *x20, float *x30,
              float *p, float *r, float *norm, float *t1, float *t2,
              int nz, int ny, int nx, const float *kz, const float *ky,
              const float *kx, int kk, float lam, float init) {
    int N=nz*ny*nx;
    float *x=x30;
    for (int i=0;i<N;++i) x[i]=init;
    for (int i=0;i<N;++i) r[i]=1.f;
    conv_sep(r,norm,t1,t2,nz,ny,nx,kz,ky,kx,kk,1);
    for (int it=1;it<=30;++it) {
        conv_sep(x,p,t1,t2,nz,ny,nx,kz,ky,kx,kk,0);
        for (int i=0;i<N;++i) r[i]=y[i]/(p[i]+1e-9f);
        conv_sep(r,p,t1,t2,nz,ny,nx,kz,ky,kx,kk,1);
        for (int i=0;i<N;++i) x[i]*=p[i]/(norm[i]+lam);
        if (it==10) memcpy(x10,x,N*sizeof(float));
        if (it==20) memcpy(x20,x,N*sizeof(float));
    }
}

void threshold(float *x, int N, float rel) {
    float m=0.f;
    for (int i=0;i<N;++i) if (x[i]>m) m=x[i];
    float t=rel*m;
    for (int i=0;i<N;++i) if (x[i]<t) x[i]=0.f;
}

void mlp_apply(const float *y, const float *z10, const float *z20, const float *z30,
               const float *mean, const float *scale, const float *W1, const float *b1,
               const float *W2, const float *b2, float *out,
               int nz, int ny, int nx, int hidden, float weight) {
    int N=nz*ny*nx;
    for (int iz=0; iz<nz; ++iz) {
        for (int iy=0; iy<ny; ++iy) {
            for (int ix=0; ix<nx; ++ix) {
                int i=iz*ny*nx+iy*nx+ix;
                float feat[57];
                int k=0;
                for (int dz=-1; dz<=1; ++dz) {
                    int zz=iz+dz;
                    for (int dy=-1; dy<=1; ++dy) {
                        int yy=iy+dy;
                        for (int dx=-1; dx<=1; ++dx) {
                            int xx=ix+dx;
                            float v=0.f;
                            if (zz>=0&&zz<nz&&yy>=0&&yy<ny&&xx>=0&&xx<nx)
                                v=z20[zz*ny*nx+yy*nx+xx];
                            feat[k]=(v-mean[k])/scale[k]; ++k;
                        }
                    }
                }
                for (int dz=-1; dz<=1; ++dz) {
                    int zz=iz+dz;
                    for (int dy=-1; dy<=1; ++dy) {
                        int yy=iy+dy;
                        for (int dx=-1; dx<=1; ++dx) {
                            int xx=ix+dx;
                            float v=0.f;
                            if (zz>=0&&zz<nz&&yy>=0&&yy<ny&&xx>=0&&xx<nx)
                                v=y[zz*ny*nx+yy*nx+xx];
                            feat[k]=(v-mean[k])/scale[k]; ++k;
                        }
                    }
                }
                feat[54]=(z10[i]-mean[54])/scale[54];
                feat[55]=(z30[i]-mean[55])/scale[55];
                feat[56]=(z20[i]-mean[56])/scale[56];
                float hid[256];
                for (int j=0; j<hidden; ++j) {
                    float acc=b1[j];
                    const float *wrow=W1+j*57;
                    for (int q=0; q<57; ++q) acc+=feat[q]*wrow[q];
                    hid[j]=acc>0.f?acc:0.f;
                }
                float res=b2[0];
                for (int j=0; j<hidden; ++j) res+=hid[j]*W2[j];
                out[i]+=weight*res;
            }
        }
    }
}
