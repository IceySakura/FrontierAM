#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <ctime>
#include <vector>

static double monotonic_seconds() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

struct Convolution {
    int nz, ny, nx, hz, hy, hx, n;
    std::vector<float> az, ay, ax, kernel, scratch1, scratch2;
    bool separable;

    Convolution(const float* h, int z, int y, int x, int kz, int ky, int kx)
        : nz(z), ny(y), nx(x), hz(kz), hy(ky), hx(kx), n(z*y*x),
          az(kz, 0), ay(ky, 0), ax(kx, 0), kernel(h, h+kz*ky*kx),
          scratch1(n), scratch2(n) {
        double total = 0;
        for (int a = 0; a < hz; ++a)
            for (int b = 0; b < hy; ++b)
                for (int c = 0; c < hx; ++c) {
                    const float v = h[(a*hy+b)*hx+c];
                    az[a] += v; ay[b] += v; ax[c] += v; total += v;
                }
        for (float& v: az) v /= total;
        for (float& v: ay) v /= total;
        for (float& v: ax) v /= total;
        double error = 0, energy = 0;
        for (int a = 0; a < hz; ++a)
            for (int b = 0; b < hy; ++b)
                for (int c = 0; c < hx; ++c) {
                    int i = (a*hy+b)*hx+c;
                    kernel[i] /= total;
                    double d = kernel[i] - az[a]*ay[b]*ax[c];
                    error += d*d;
                    energy += double(kernel[i])*kernel[i];
                }
        separable = error < 1e-9 * energy;
    }

    void axis(const float* src, float* dst, const std::vector<float>& w,
              int a, bool adjoint) {
        std::fill(dst, dst+n, 0.f);
        const int stride = a == 0 ? ny*nx : (a == 1 ? nx : 1);
        const int length = a == 0 ? nz : (a == 1 ? ny : nx);
        const int block = stride*length;
        for (int k = 0; k < int(w.size()); ++k) {
            const int shift = k-int(w.size())/2;
            const float weight = w[adjoint ? k : int(w.size())-1-k];
            const int begin = std::max(0, -shift);
            const int end = std::min(length, length-shift);
            for (int base = 0; base < n; base += block) {
                const int l = base+begin*stride, r = base+end*stride;
                const float* in = src+shift*stride;
                for (int i = l; i < r; ++i) dst[i] += weight*in[i];
            }
        }
    }

    void apply(const float* src, float* dst, bool adjoint = false) {
        if (separable) {
            axis(src, scratch1.data(), az, 0, adjoint);
            axis(scratch1.data(), scratch2.data(), ay, 1, adjoint);
            axis(scratch2.data(), dst, ax, 2, adjoint);
            return;
        }
        std::fill(dst, dst+n, 0.f);
        for (int a = 0; a < hz; ++a)
            for (int b = 0; b < hy; ++b)
                for (int c = 0; c < hx; ++c) {
                    const int dz=a-hz/2, dy=b-hy/2, dx=c-hx/2;
                    const float w = adjoint ? kernel[(a*hy+b)*hx+c]
                        : kernel[((hz-1-a)*hy+hy-1-b)*hx+hx-1-c];
                    if (w == 0) continue;
                    const int offset=(dz*ny+dy)*nx+dx;
                    for (int z=std::max(0,-dz); z<std::min(nz,nz-dz); ++z)
                        for (int y=std::max(0,-dy); y<std::min(ny,ny-dy); ++y) {
                            int l=(z*ny+y)*nx+std::max(0,-dx);
                            int r=(z*ny+y)*nx+std::min(nx,nx-dx);
                            for (int i=l; i<r; ++i) dst[i] += w*src[i+offset];
                        }
                }
    }

    void laplace(const float* src, float* dst) {
        std::fill(dst, dst+n, 0.f);
        const int strides[3] = {ny*nx, nx, 1};
        const int lengths[3] = {nz, ny, nx};
        for (int a=0; a<3; ++a) {
            const int stride=strides[a], block=stride*lengths[a];
            for (int base=0; base<n; base+=block)
                for (int i=base; i<base+block-stride; ++i) {
                    float d=src[i+stride]-src[i];
                    dst[i] += d; dst[i+stride] -= d;
                }
        }
    }
};

static float estimate_noise(const float* y, Convolution& H) {
    std::vector<float> smooth(H.n), temp(H.n), sorted(H.n);
    std::vector<float> w(5, .2f);
    H.axis(y, temp.data(), w, 0, false);
    H.axis(temp.data(), smooth.data(), w, 1, false);
    H.axis(smooth.data(), temp.data(), w, 2, false);
    sorted = temp;
    int index = std::max(0, int(H.n*.35));
    std::nth_element(sorted.begin(), sorted.begin()+index, sorted.end());
    const float cutoff = sorted[index];
    double sum = 0;
    int count = 0;
    for (int i=0; i<H.n; ++i) if (temp[i] <= cutoff) {
        sum += double(y[i])*y[i]; ++count;
    }
    return std::max(1e-6f, float(std::sqrt(2*sum/std::max(count,1))*1.06));
}

#include "geometry.h"

extern "C" int deconvolve(const float* observed, const float* psf, float* output,
                         int nz, int ny, int nx, int hz, int hy, int hx,
                         int iterations, double deadline, float regularizer,
                         float debias) {
    Convolution H(psf, nz, ny, nx, hz, hy, hx);
    float sigma = estimate_noise(observed, H);
    if (regularizer < 0) regularizer = std::max(.0005f, sigma*.25f);
    const float bias = debias*sigma*.3989422804f;
    const float step = .95f / (1+144*regularizer);
    std::vector<float> x(H.n), z(H.n), next(H.n), pred(H.n), grad(H.n),
                       lap(H.n), lap2(H.n), target(H.n);
    for (int i=0; i<H.n; ++i) {
        target[i] = observed[i]-bias;
        x[i] = z[i] = std::max(target[i], 0.f);
    }
    double t=1;
    int completed=0;
    for (int k=0; k<iterations; ++k) {
        if (monotonic_seconds() > deadline-.004) break;
        H.apply(z.data(), pred.data());
        for (int i=0; i<H.n; ++i) pred[i] -= target[i];
        H.apply(pred.data(), grad.data(), true);
        H.laplace(z.data(), lap.data());
        H.laplace(lap.data(), lap2.data());
        double tnew=(1+std::sqrt(1+4*t*t))*.5;
        float momentum=float((t-1)/tnew);
        for (int i=0; i<H.n; ++i) {
            next[i]=std::max(z[i]-step*(grad[i]+regularizer*lap2[i]), 0.f);
            z[i]=next[i]+momentum*(next[i]-x[i]);
        }
        x.swap(next); t=tnew; ++completed;
    }
    for (int i=0; i<H.n; ++i)
        output[i] = std::isfinite(x[i]) ? std::max(x[i], 0.f) : observed[i];
    return completed;
}

extern "C" int reconstruct(const float* observed, const float* psf, float* output,
                          int nz, int ny, int nx, int hz, int hy, int hx,
                          double deadline, int flags) {
    int iterations=deconvolve(observed,psf,output,nz,ny,nx,hz,hy,hx,80,deadline,-1.,1.);
    Convolution H(psf,nz,ny,nx,hz,hy,hx);
    return iterations+geometry_refine(H,observed,output,deadline,flags);
}

extern "C" int fast_cpu_available() {
#if defined(__x86_64__) && defined(__GNUC__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return 0;
#endif
}

extern "C" void reorder_float32(const unsigned char* source, float* destination,
                               int nz, int ny, int nx, int fortran, int swap) {
    for (int z=0; z<nz; ++z) for (int y=0; y<ny; ++y) for (int x=0; x<nx; ++x) {
        int out=(z*ny+y)*nx+x;
        int in=fortran ? z+nz*(y+ny*x) : out;
        unsigned int bits;
        std::memcpy(&bits,source+4*in,4);
        if (swap) bits=__builtin_bswap32(bits);
        std::memcpy(destination+out,&bits,4);
    }
}
