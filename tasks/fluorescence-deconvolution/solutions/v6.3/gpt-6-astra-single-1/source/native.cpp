#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

// Fixed-size, single-threaded forward/adjoint and analytic shape derivatives.
constexpr int N = 32;
constexpr int NN = N*N*N;
static std::vector<float> coeff,invscale,zero_loss,zero_grad;
static std::vector<int> lower;
static bool exact_likelihood=false;
static double point_amp_prior=std::log(.85);
extern "C" void configure(double amp) {point_amp_prior=std::log(amp);}

extern "C" void prepare_likelihood(const float* y,double sigma,double gain,int mode) {
    exact_likelihood=mode!=0;
    if(!exact_likelihood) return;
    coeff.resize(NN*14); invscale.resize(NN); lower.resize(NN);
    zero_loss.resize(NN);zero_grad.resize(NN);
    const double sg=sigma*gain;
    for(int i=0;i<NN;++i) {
        const double yy=y[i]*gain;
        const int lo=std::max(0,int(std::floor(yy-6)));
        const double scale=std::max(1.,yy);
        lower[i]=lo;
        invscale[i]=1/scale;
        double lc[14],mx=-1e100;
        for(int j=0;j<14;++j) {
            const int n=lo+j;
            double lp;
            if(y[i]==0) lp=std::log(std::max(1e-300,.5*std::erfc(n/(sg*std::sqrt(2.)))));
            else lp=-.5*(n-yy)*(n-yy)/(sg*sg);
            lc[j]=lp+n*std::log(scale)-std::lgamma(n+1.);
            mx=std::max(mx,lc[j]);
        }
        for(int j=0;j<14;++j) coeff[i*14+j]=std::exp(lc[j]-mx);
        zero_loss[i]=-std::log(std::max(1e-30f,coeff[i*14]));
        zero_grad[i]=gain*(1-coeff[i*14+1]*invscale[i]/std::max(1e-30f,coeff[i*14]));
    }
}

static void blur_axis(const float* __restrict__ x, float* __restrict__ y,
                      const float* k, int axis, bool reflect=false) {
    if (axis == 0) {
        for (int z=0; z<N; ++z) {
            float* dst = y + z*N*N;
            std::fill(dst, dst+N*N, 0.f);
            for (int d=-4; d<=4; ++d) {
                int zz=z+d;
                if (zz<0 || zz>=N) {
                    if(!reflect) continue;
                    zz=zz<0 ? -zz-1 : 2*N-zz-1;
                }
                const float* src=x+zz*N*N;
                const float a=k[d+4];
                for (int i=0; i<N*N; ++i) dst[i] += a*src[i];
            }
        }
    } else if (axis == 1) {
        for (int z=0; z<N; ++z) for(int j=0;j<N;++j) {
            float* dst=y+(z*N+j)*N;
            std::fill(dst,dst+N,0.f);
            for(int d=-4;d<=4;++d) {
                int jj=j+d;
                if(jj<0||jj>=N) {
                    if(!reflect) continue;
                    jj=jj<0 ? -jj-1 : 2*N-jj-1;
                }
                const float* src=x+(z*N+jj)*N;
                const float a=k[d+4];
                for(int i=0;i<N;++i) dst[i]+=a*src[i];
            }
        }
    } else {
        for(int row=0;row<N*N;++row) {
            float buf[40]={};
            std::memcpy(buf+4,x+row*N,N*sizeof(float));
            if(reflect) for(int d=0;d<4;++d) {
                buf[3-d]=x[row*N+d];
                buf[36+d]=x[row*N+N-1-d];
            }
            float* dst=y+row*N;
            for(int i=0;i<N;++i) {
                dst[i]=k[0]*buf[i]+k[1]*buf[i+1]+k[2]*buf[i+2]+
                    k[3]*buf[i+3]+k[4]*buf[i+4]+k[5]*buf[i+5]+
                    k[6]*buf[i+6]+k[7]*buf[i+7]+k[8]*buf[i+8];
            }
        }
    }
}

static void blur(const float* x, float* out, const float* k, float* tmp, float* tmp2) {
    blur_axis(x,tmp,k,0);
    blur_axis(tmp,tmp2,k+9,1);
    blur_axis(tmp2,out,k+18,2);
}

extern "C" void convolve(const float* x, float* out, const float* k) {
    std::vector<float> t(NN),u(NN);
    blur(x,out,k,t.data(),u.data());
}

extern "C" void smooth(const float* x,float* out,const float* k) {
    std::vector<float> t(NN),u(NN);
    blur_axis(x,t.data(),k,0,true);
    blur_axis(t.data(),u.data(),k,1,true);
    blur_axis(u.data(),out,k,2,true);
}

extern "C" void maximum5(const float* x,float* out) {
    std::vector<float> t(NN),u(NN);
    for(int axis=0;axis<3;++axis) {
        const float* src=axis==0 ? x : (axis==1 ? t.data() : u.data());
        float* dst=axis==0 ? t.data() : (axis==1 ? u.data() : out);
        const int stride=axis==0 ? N*N : (axis==1 ? N : 1);
        for(int z=0;z<N;++z) for(int y=0;y<N;++y) for(int xx=0;xx<N;++xx) {
            const int coord=axis==0 ? z : (axis==1 ? y : xx);
            const int i=(z*N+y)*N+xx;
            float v=-1e30f;
            for(int d=std::max(-2,-coord);d<=std::min(2,N-1-coord);++d)
                v=std::max(v,src[i+d*stride]);
            dst[i]=v;
        }
    }
}

static void object(const double* p,int type,float* image,const float* residual,double* grad) {
    const float amp=std::exp(p[0]), s=std::exp(p[7]), is2=1/(s*s);
    const bool deriv=residual!=nullptr;
    if(type>=2) {
        const int axis=type-2;
        const int a1=(axis+1)%3,a2=(axis+2)%3;
        float shapes[N];
        for(int i=0;i<N;++i) {
            const float d=i-p[1];
            shapes[i]=amp*std::exp(-.5f*d*d*is2);
        }
        for(int z=0;z<N;++z) for(int y=0;y<N;++y) for(int x=0;x<N;++x) {
            int cc[3]={z,y,x};
            const int i=(z*N+y)*N+x;
            const float d=cc[axis]-p[1], q1=(cc[a1]-15.5f)/15.5f,q2=(cc[a2]-15.5f)/15.5f;
            const float shape=shapes[cc[axis]];
            const float raw_ramp=1+p[4]*q1+p[5]*q2;
            const float ramp=std::max(.01f,raw_ramp);
            const float val=shape*ramp;
            if(!deriv) image[i]+=val;
            else {
                const float r=residual[i]*val;
                grad[0]+=r;
                grad[1]+=r*d*is2;
                if(raw_ramp>.01f) {
                    grad[4]+=residual[i]*shape*q1;
                    grad[5]+=residual[i]*shape*q2;
                }
                grad[7]+=r*d*d*is2;
            }
        }
        return;
    }
    const float v[3]={(float)p[4],(float)p[5],(float)p[6]};
    const float vv=v[0]*v[0]+v[1]*v[1]+v[2]*v[2]+1e-12f;
    int lo[3],hi[3];
    for(int a=0;a<3;++a) {
        const float extent=type==1 ? std::abs(v[a]) : 0.f;
        lo[a]=std::max(0,(int)std::floor(p[a+1]-extent-4.5*s));
        hi[a]=std::min(N-1,(int)std::ceil(p[a+1]+extent+4.5*s));
    }
    for(int z=lo[0];z<=hi[0];++z) for(int y=lo[1];y<=hi[1];++y) for(int x=lo[2];x<=hi[2];++x) {
        float r[3]={float(z-p[1]),float(y-p[2]),float(x-p[3])};
        const float t=type==1 ? std::max(-1.f,std::min(1.f,(r[0]*v[0]+r[1]*v[1]+r[2]*v[2])/vv)):0.f;
        for(int a=0;a<3;++a) r[a]-=t*v[a];
        const float d2=r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
        if(d2*is2>25.f) continue;
        const float val=amp*std::exp(-.5f*d2*is2);
        const int i=(z*N+y)*N+x;
        if(!deriv) image[i]+=val;
        else {
            const float rr=residual[i]*val;
            grad[0]+=rr;
            grad[7]+=rr*d2*is2;
            for(int a=0;a<3;++a) {
                grad[a+1]+=rr*r[a]*is2;
                if(type==1) grad[a+4]+=rr*t*r[a]*is2;
            }
        }
    }
}

extern "C" void render(const double* p,const int* types,int count,float* image) {
    std::fill(image,image+NN,0.f);
    for(int j=0;j<count;++j) object(p+8*j,types[j],image,nullptr,nullptr);
}

extern "C" double evaluate(const double* p,const int* types,int count,
                          const float* y,const float* weights,const float* k,
                          double sigma,double gain,double prior,
                          double* grad,float* image) {
    std::vector<float> m(NN),t(NN),u(NN),r(NN),adj(NN);
    render(p,types,count,image);
    blur(image,m.data(),k,t.data(),u.data());
    double loss=0.;
    const float b0=sigma*.3989422804014327;
    const float a=1./gain;
    const float b1=sigma>0 ? sigma*std::exp(-.5*a*a/(sigma*sigma))*.3989422804014327
                           -.5*a*std::erfc(a/(sigma*1.4142135623730951)):0;
    for(int i=0;i<NN;++i) {
        if(exact_likelihood) {
            const float lam=std::max(1e-12f,float(gain*m[i]));
            if(lam<1e-7f && lower[i]==0) {
                loss+=zero_loss[i]+m[i]*zero_grad[i];
                r[i]=zero_grad[i];
                continue;
            }
            const float t=lam*invscale[i];
            const float* cc=coeff.data()+14*i;
            float pp=cc[13],dd=0.f;
            for(int j=12;j>=0;--j) {dd=dd*t+pp;pp=pp*t+cc[j];}
            pp=std::max(1e-35f,pp);
            loss+=lam-lower[i]*std::log(t)-std::log(pp);
            r[i]=gain*(1-lower[i]/lam-dd*invscale[i]/pp);
        } else {
            const float gm=gain*m[i],e=std::exp(-gm);
            const float bias=e*(b0+b1*gm);
            const float deriv=1+gain*e*(b1-b0-b1*gm);
            const float err=m[i]+bias-y[i];
            loss+=.5*err*err*weights[i];
            r[i]=err*weights[i]*deriv;
        }
    }
    blur(r.data(),adj.data(),k,t.data(),u.data());
    std::fill(grad,grad+count*8,0.);
    for(int j=0;j<count;++j) {
        object(p+8*j,types[j],nullptr,adj.data(),grad+8*j);
        if(prior>0 && types[j]<2) {
            const double la=p[8*j] - (types[j]==0 ? point_amp_prior : std::log(.85));
            const double ls=p[8*j+7] - std::log(.85);
            const double aw=types[j]==0 ? prior : .1*prior;
            loss+=aw*(la*la+.5*ls*ls);
            grad[j*8]+=2*aw*la;
            grad[j*8+7]+=aw*ls;
            if(types[j]==0) for(int a=1;a<=3;++a) {
                const double c=p[8*j+a];
                const double delta=c<2 ? c-2 : (c>30 ? c-30 : 0);
                loss+=.5*prior*delta*delta;
                grad[8*j+a]+=prior*delta;
            }
        } else if(prior>0 && types[j]>=2) {
            const double ls=p[8*j+7]-std::log(.7);
            loss+=30*prior*ls*ls;
            grad[j*8+7]+=60*prior*ls;
        }
    }
    return loss;
}
