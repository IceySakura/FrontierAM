#include <math.h>
#include <stdlib.h>
#include <string.h>

static void conv_axis(const float *src, float *dst, int nz, int ny, int nx,
                      int axis, const float *k, int r) {
    int st = axis == 2 ? 1 : (axis == 1 ? nx : ny * nx);
    int len = axis == 2 ? nx : (axis == 1 ? ny : nz);
    int lines = axis == 2 ? nz * ny : (axis == 1 ? nz * nx : ny * nx);
    for (int l = 0; l < lines; ++l) {
        int base;
        if (axis == 2) base = l * nx;
        else if (axis == 1) base = (l / nx) * ny * nx + (l % nx);
        else base = l;
        const float *s = src + base;
        float *d = dst + base;
        if (r == 4) {
            for (int i = 0; i < len; ++i) {
                float acc = 0.f;
                if (i >= 4 && i + 4 < len) {
                    acc = s[(i-4)*st]*k[0] + s[(i-3)*st]*k[1] + s[(i-2)*st]*k[2]
                        + s[(i-1)*st]*k[3] + s[i*st]*k[4] + s[(i+1)*st]*k[5]
                        + s[(i+2)*st]*k[6] + s[(i+3)*st]*k[7] + s[(i+4)*st]*k[8];
                } else {
                    for (int t = -4; t <= 4; ++t) {
                        int j = i + t;
                        if ((unsigned)j < (unsigned)len) acc += s[j*st] * k[t+4];
                    }
                }
                d[i*st] = acc;
            }
        } else {
            for (int i = 0; i < len; ++i) {
                float acc = 0.f;
                for (int t = -r; t <= r; ++t) {
                    int j = i + t;
                    if ((unsigned)j < (unsigned)len) acc += s[j*st] * k[t+r];
                }
                d[i*st] = acc;
            }
        }
    }
}

static void tv_huber(const float *x, float *out, int nz, int ny, int nx,
                     float eps, float *gx, float *gy, float *gz, float *den) {
    int n = nz * ny * nx;
    int plane = ny * nx;
    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                int i = iz * plane + iy * nx + ix;
                float a = 0.f, b = 0.f, c = 0.f;
                if (ix + 1 < nx) a = x[i + 1] - x[i];
                if (iy + 1 < ny) b = x[i + nx] - x[i];
                if (iz + 1 < nz) c = x[i + plane] - x[i];
                gx[i] = a; gy[i] = b; gz[i] = c;
                den[i] = sqrtf(a*a + b*b + c*c + eps*eps);
            }
        }
    }
    for (int i = 0; i < n; ++i) out[i] = 0.f;
    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                int i = iz * plane + iy * nx + ix;
                float q = gx[i] / den[i];
                if (ix + 1 < nx) { out[i] -= q; out[i + 1] += q; }
                q = gy[i] / den[i];
                if (iy + 1 < ny) { out[i] -= q; out[i + nx] += q; }
                q = gz[i] / den[i];
                if (iz + 1 < nz) { out[i] -= q; out[i + plane] += q; }
            }
        }
    }
}

int solve_separable(const float *y, float *out, int nz, int ny, int nx,
                    const float *kz, const float *ky, const float *kx,
                    int rz, int ry, int rx, int iters, float lam, float eps,
                    float accel) {
    int n = nz * ny * nx;
    float *x = malloc(n * sizeof(float));
    float *t1 = malloc(n * sizeof(float));
    float *t2 = malloc(n * sizeof(float));
    float *t3 = malloc(n * sizeof(float));
    float *ratio = malloc(n * sizeof(float));
    float *adj = malloc(n * sizeof(float));
    float *norm = malloc(n * sizeof(float));
    float *tv = malloc(n * sizeof(float));
    float *gx = malloc(n * sizeof(float));
    float *gy = malloc(n * sizeof(float));
    float *gz = malloc(n * sizeof(float));
    float *den = malloc(n * sizeof(float));
    float *kzr = malloc((2*rz+1)*sizeof(float));
    float *kyr = malloc((2*ry+1)*sizeof(float));
    float *kxr = malloc((2*rx+1)*sizeof(float));
    if (!x || !t1 || !t2 || !t3 || !ratio || !adj || !norm || !tv ||
        !gx || !gy || !gz || !den || !kzr || !kyr || !kxr) return -1;
    (void)accel;
    for (int i = 0; i <= 2*rz; ++i) kzr[i] = kz[2*rz-i];
    for (int i = 0; i <= 2*ry; ++i) kyr[i] = ky[2*ry-i];
    for (int i = 0; i <= 2*rx; ++i) kxr[i] = kx[2*rx-i];
    float mean = 0.f;
    for (int i = 0; i < n; ++i) mean += y[i];
    mean /= n;
    float init = mean * 0.1f;
    if (init < 1e-6f) init = 1e-6f;
    for (int i = 0; i < n; ++i) x[i] = y[i] > init ? y[i] : init;
    /* norm = A^T 1 */
    for (int i = 0; i < n; ++i) t1[i] = 1.f;
    conv_axis(t1, t2, nz, ny, nx, 0, kzr, rz);
    conv_axis(t2, t3, nz, ny, nx, 1, kyr, ry);
    conv_axis(t3, norm, nz, ny, nx, 2, kxr, rx);
    for (int it = 0; it < iters; ++it) {
        conv_axis(x, t1, nz, ny, nx, 2, kx, rx);
        conv_axis(t1, t2, nz, ny, nx, 1, ky, ry);
        conv_axis(t2, t3, nz, ny, nx, 0, kz, rz);
        for (int i = 0; i < n; ++i) {
            float f = t3[i];
            if (f < 1e-6f) f = 1e-6f;
            ratio[i] = y[i] / f;
        }
        conv_axis(ratio, t1, nz, ny, nx, 0, kzr, rz);
        conv_axis(t1, t2, nz, ny, nx, 1, kyr, ry);
        conv_axis(t2, adj, nz, ny, nx, 2, kxr, rx);
        if (lam > 0.f) tv_huber(x, tv, nz, ny, nx, eps, gx, gy, gz, den);
        for (int i = 0; i < n; ++i) {
            float d = norm[i] + (lam > 0.f ? lam * tv[i] : 0.f);
            if (d < 0.05f) d = 0.05f;
            float c = adj[i] / d;
            if (c < 1e-8f) c = 1e-8f;
            float v = x[i] * c;
            if (v < 1e-8f) v = 1e-8f;
            out[i] = v;
        }
        memcpy(x, out, n * sizeof(float));
    }
    memcpy(out, x, n * sizeof(float));
    free(x); free(t1); free(t2); free(t3); free(ratio); free(adj); free(norm);
    free(tv); free(gx); free(gy); free(gz); free(den);
    free(kzr); free(kyr); free(kxr);
    return 0;
}
