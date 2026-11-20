#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <time.h>
#include <vector>

#include <fftw3.h>

namespace {

constexpr int N = 32, S = N * N * N, R = 4, K = 2 * R + 1;
constexpr int F = N + 2 * R, FS = F * F * F, FC = F * F * (F / 2 + 1);
constexpr float pi = 3.14159265358979323846f;
using Volume = std::array<float, S>;

double now() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1.e-9 * ts.tv_nsec;
}

inline int index(int z, int y, int x) { return (z * N + y) * N + x; }

struct Box {
    int lo[3], hi[3], nx, ny, nz, plane, size;
    Box() : Box({0, 0, 0}, {N, N, N}) {}
    Box(std::array<int, 3> u, std::array<int, 3> v) {
        for (int i = 0; i < 3; ++i) {
            lo[i] = std::clamp(u[i], 0, N - 1);
            hi[i] = std::clamp(v[i], lo[i] + 1, N);
        }
        lo[2] &= ~7;
        hi[2] = (hi[2] + 7) & ~7;
        nx = hi[2] - lo[2];
        ny = hi[1] - lo[1];
        nz = hi[0] - lo[0];
        plane = nx * ny;
        size = plane * nz;
    }
};

enum Family { Spot, Line, Plane };

struct Atom {
    Family family = Spot;
    int axis = 0;
    std::array<float, 8> p{};
    int count() const { return family == Line ? 8 : 5; }
    int sw() const { return family == Line ? 6 : family == Spot ? 3 : 1; }
    int amp() const { return family == Line ? 7 : family == Spot ? 4 : 2; }
    float length() const {
        float ss = 0;
        for (int d = 0; d < 3; d++)
            ss += (p[d] - p[d + 3]) * (p[d] - p[d + 3]);
        return std::sqrt(ss);
    }
};

struct LineSeed {
    Atom atom;
    float value = 0;
};

struct Table {
    float a[3][N][N]{};
    int lo[3][N]{}, hi[3][N]{};

    void set_ranges(float rel) {
        for (int d = 0; d < 3; d++)
            for (int i = 0; i < N; i++) {
                const float m = *std::max_element(a[d][i], a[d][i] + N);
                int j = 0, k = N;
                while (j + 1 < k && a[d][i][j] < m * rel) j++;
                while (j + 1 < k && a[d][i][k - 1] < m * rel) k--;
                lo[d][i] = j;
                hi[d][i] = k;
            }
    }
};

struct Wiener {
    std::array<float, FS> pad{};
    std::array<std::array<float, 2>, FC> spec{}, inv{};
    fftwf_plan fw = nullptr, bw = nullptr;

    Wiener(const float h[3][K], float alpha) {
        auto* c = reinterpret_cast<fftwf_complex*>(spec.data());
        fw = fftwf_plan_dft_r2c_3d(F, F, F, pad.data(), c, FFTW_ESTIMATE);
        bw = fftwf_plan_dft_c2r_3d(F, F, F, c, pad.data(), FFTW_ESTIMATE);
        if (!fw || !bw) {
            if (fw) fftwf_destroy_plan(fw);
            if (bw) fftwf_destroy_plan(bw);
            throw std::bad_alloc();
        }
        pad.fill(0);
        for (int z = -R; z <= R; z++)
            for (int y = -R; y <= R; y++)
                for (int x = -R; x <= R; x++)
                    pad[(((z + F) % F) * F + (y + F) % F) * F + (x + F) % F]
                        = h[0][z + R] * h[1][y + R] * h[2][x + R];
        fftwf_execute(fw);
        float omega[F];
        for (int i = 0; i < F; i++) {
            const float s = std::sin(pi * i / F);
            omega[i] = 4 * s * s;
        }
        int i = 0;
        for (int z = 0; z < F; z++)
            for (int y = 0; y < F; y++)
                for (int x = 0; x <= F / 2; x++, i++) {
                    const auto a = spec[i];
                    const float lap = omega[z] + omega[y] + omega[x];
                    const float t = 1.f / (FS * (a[0] * a[0] + a[1] * a[1]
                                               + 0.002f + alpha * lap * lap));
                    inv[i] = {t * a[0], -t * a[1]};
                }
    }
    ~Wiener() {
        if (fw) fftwf_destroy_plan(fw);
        if (bw) fftwf_destroy_plan(bw);
    }
    void apply(const float* u, float* v) {
        pad.fill(0);
        // Smoothly extend the observation by one voxel before inverse
        // filtering. Component fitting itself always uses zero exterior.
        for (int z = -1; z <= N; z++)
            for (int y = -1; y <= N; y++)
                for (int x = -1; x <= N; x++) {
                    const int ez = std::clamp(z, 0, N - 1);
                    const int ey = std::clamp(y, 0, N - 1);
                    const int ex = std::clamp(x, 0, N - 1);
                    const float w = (z != ez || y != ey || x != ex) ? .55f : 1.f;
                    pad[(((z + F) % F) * F + (y + F) % F) * F + (x + F) % F]
                        = w * u[index(ez, ey, ex)];
                }
        fftwf_execute(fw);
        for (int i = 0; i < FC; i++) {
            const auto a = spec[i], h = inv[i];
            spec[i] = {a[0] * h[0] - a[1] * h[1],
                       a[1] * h[0] + a[0] * h[1]};
        }
        fftwf_execute(bw);
        for (int z = 0; z < N; z++)
            for (int y = 0; y < N; y++)
                std::memcpy(v + index(z, y, 0),
                            pad.data() + (z * F + y) * F, N * sizeof(float));
    }
};

template <int P>
bool solve_system(const double (&a)[P][P], const double* g, int n, double damp, float* step) {
    double l[P][P]{}, b[P];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double t = a[i][j];
            if (i == j) t += (a[i][i] + .01) * damp + 1.e-8;
            for (int k = 0; k < j; k++) t -= l[i][k] * l[j][k];
            if (j == i) {
                if (!(t > 0)) return false;
                l[i][j] = std::sqrt(t);
            } else {
                l[i][j] = t / l[j][j];
            }
        }
        double t = g[i];
        for (int j = 0; j < i; j++) t -= l[i][j] * b[j];
        b[i] = t / l[i][i];
    }
    for (int i = n - 1; i >= 0; i--) {
        double t = b[i];
        for (int j = i + 1; j < n; j++) t -= l[j][i] * step[j];
        step[i] = static_cast<float>(t / l[i][i]);
    }
    return true;
}

float dot(const float* a, const float* b, int n) {
    float m = 0;
    for (int i = 0; i < n; i++) m += a[i] * b[i];
    return m;
}

class Engine {
    const float* y;
    const float* rough;
    float h[3][K], edge[3][N]{}, ramp[3][N]{}, hramp[3][N]{};
    float sigma, shot, gvar, peak;
    float spot_amp = 0.f, amp_pc = 0.f;
    double deadline;
    int debug;
    bool recovery_search = false;
    Table spots_t, spots_sq;
    Volume pred{}, estimate{}, resid{}, ray{}, w_global{}, wres{}, diag{}, cov{}, saved{};
    Volume pg_y, pg_w;
    bool score_refinement = false;
    Volume tmp1, tmp2, shape, a0, a1, target, invsd, linres, linw;
    std::array<Volume, 8> jac, jconv;
    std::vector<Atom> atoms;
    std::unique_ptr<Wiener> inverse;

public:
    Engine(const float* y_, const float* rough_, const float* psf, float sigma_,
           float shot_, double dl, int dbg)
        : y(y_), rough(rough_), sigma(sigma_), shot(shot_), deadline(dl), debug(dbg) {
        std::memcpy(h, psf, sizeof h);
        gvar = std::max(0.84f * sigma * sigma, 1.e-14f);
        peak = std::max(1.e-7f, *std::max_element(rough, rough + S));
        atoms.reserve(32);
        for (int d = 0; d < 3; d++) {
            for (int i = 0; i < N; i++) ramp[d][i] = (2.f * i - N + 1) / (N - 1);
            float ones[N];
            std::fill_n(ones, N, 1.f);
            one_d(d, ones, edge[d]);
            one_d(d, ramp[d], hramp[d]);
            for (int i = 0; i < N; i++) {
                float g[N];
                for (int k = 0; k < N; k++) g[k] = std::exp(-.5f * ((k-i)/.87f) * ((k-i)/.87f));
                one_d(d, g, spots_t.a[d][i]);
                for (int k = 0; k < N; k++)
                    spots_sq.a[d][i][k] = spots_t.a[d][i][k] * spots_t.a[d][i][k];
            }
        }
        spots_t.set_ranges(1.e-5f);
        spots_sq.set_ranges(1.e-5f);
    }

    void one_d(int d, const float* a, float* b) const {
        for (int i = 0; i < N; i++) {
            float s = 0;
            for (int j = std::max(-R, i - (N - 1)); j <= std::min(R, i); j++)
                s += h[d][R + j] * a[i - j];
            b[i] = s;
        }
    }

    void convolve(const float* restrict_in, float* restrict_out, const Box& b, bool adj = false) {
        const float* __restrict in = restrict_in;
        float* __restrict out = restrict_out;
        float* __restrict t1 = tmp1.data();
        float* __restrict t2 = tmp2.data();
        const int nx = b.nx, np = b.plane;
        for (int i = 0; i < b.size; i += nx) {
            for (int j = 0; j < nx; j++) t1[i + j] = h[2][R] * in[i + j];
            for (int k = 1; k <= R; k++) {
                const float m = h[2][R + (adj ? -k : k)], p = h[2][R + (adj ? k : -k)];
                for (int j = k; j < nx; j++) t1[i + j] += m * in[i + j - k];
                for (int j = 0; j < nx - k; j++) t1[i + j] += p * in[i + j + k];
            }
        }
        for (int z = 0; z < b.nz; z++) {
            for (int j = 0; j < np; j++) t2[z * np + j] = h[1][R] * t1[z * np + j];
            for (int k = 1; k <= R && k < b.ny; k++) {
                const float m = h[1][R + (adj ? -k : k)], p = h[1][R + (adj ? k : -k)];
                const int kd = k * nx, od = z * np;
                for (int j = kd; j < np; j++) t2[od + j] += m * t1[od + j - kd];
                for (int j = 0; j < np - kd; j++) t2[od + j] += p * t1[od + j + kd];
            }
        }
        for (int j = 0; j < b.size; j++) out[j] = h[0][R] * t2[j];
        for (int k = 1; k <= R && k < b.nz; k++) {
            const float m = h[0][R + (adj ? -k : k)], p = h[0][R + (adj ? k : -k)];
            const int kd = k * np;
            for (int j = kd; j < b.size; j++) out[j] += m * t2[j - kd];
            for (int j = 0; j < b.size - kd; j++) out[j] += p * t2[j + kd];
        }
    }

    void distribute(const Table& t, const float* in, float* out) {
        for (int i = 0; i < S; i += N) {
            for (int x = 0; x < N; x++) {
                float s = 0;
                const float* ma = t.a[2][x];
                for (int k = t.lo[2][x]; k < t.hi[2][x]; k++) s += ma[k] * in[i + k];
                tmp1[i + x] = s;
            }
        }
        tmp2.fill(0);
        for (int z = 0; z < N; z++)
            for (int y_ = 0; y_ < N; y_++) {
                const int off = index(z, y_, 0);
                for (int k = t.lo[1][y_]; k < t.hi[1][y_]; k++) {
                    const float v = t.a[1][y_][k];
                    const int b = index(z, k, 0);
                    for (int x = 0; x < N; x++) tmp2[off + x] += v * tmp1[b + x];
                }
            }
        std::fill_n(out, S, 0.f);
        for (int z = 0; z < N; z++)
            for (int k = t.lo[0][z]; k < t.hi[0][z]; k++) {
                const float v = t.a[0][z][k];
                for (int j = 0; j < N*N; j++) out[z*N*N + j] += v * tmp2[k*N*N + j];
            }
    }

    static Box fit_box(const Atom& p) {
        std::array<int, 3> lo{0, 0, 0}, hi{N, N, N};
        const float pad = R + 5.5f + 2.6f * std::exp(p.p[p.sw()]);
        if (p.family == Plane) {
            lo[p.axis] = static_cast<int>(std::floor(p.p[0] - pad));
            hi[p.axis] = static_cast<int>(std::ceil(p.p[0] + pad)) + 1;
        } else {
            for (int i = 0; i < 3; i++) {
                const float a = p.p[i], b = p.family == Line ? p.p[i + 3] : a;
                lo[i] = static_cast<int>(std::floor(std::min(a,b) - pad));
                hi[i] = static_cast<int>(std::ceil(std::max(a,b) + pad)) + 1;
            }
        }
        return Box(lo, hi);
    }

    void paint_spot(const Atom& a, const Box& b, float* out, bool deriv, bool blur) {
        const auto& p = a.p;
        const float iv = std::exp(-2 * p[3]), A = p[4];
        float f[3][N], u[3][N], v[3][N];
        for (int d = 0; d < 3; d++) {
            float f0[N], u0[N], v0[N];
            for (int j = 0; j < N; j++) {
                const float t = j - p[d];
                f0[j] = std::exp(-.5f * iv * t * t);
                u0[j] = f0[j] * iv * t;
                v0[j] = u0[j] * t;
            }
            if (blur) {
                one_d(d, f0, f[d]);
                if (deriv) {
                    one_d(d, u0, u[d]);
                    one_d(d, v0, v[d]);
                }
            } else {
                std::memcpy(f[d], f0, sizeof f0);
            }
        }
        int k = 0;
        for (int z = b.lo[0]; z < b.hi[0]; z++)
            for (int y_ = b.lo[1]; y_ < b.hi[1]; y_++) {
                const float fzy = f[0][z] * f[1][y_];
                const int ox = b.lo[2], ex = b.hi[2], o = k - b.lo[2];
                if (deriv) {
                    const float vz = A * u[0][z] * f[1][y_], vy = A * f[0][z] * u[1][y_];
                    const float vs = A * (v[0][z] * f[1][y_] + f[0][z] * v[1][y_]);
                    for (int x = ox; x < ex; x++) {
                        jconv[0][o+x] = vz * f[2][x];
                        jconv[1][o+x] = vy * f[2][x];
                        jconv[2][o+x] = A * fzy * u[2][x];
                        jconv[3][o+x] = A * fzy * v[2][x] + vs * f[2][x];
                        jconv[4][o+x] = fzy * f[2][x];
                        out[o+x] = A * fzy * f[2][x];
                    }
                } else {
                    for (int x = ox; x < ex; x++) out[o+x] = A * fzy * f[2][x];
                }
                k += b.nx;
            }
    }

    void paint_plane(const Atom& a, const Box& b, float* out, bool deriv, bool blur) {
        const auto& p = a.p;
        const int d = a.axis, i = d == 0 ? 1 : 0, j = d == 2 ? 1 : 2;
        const float iv = std::exp(-2 * p[1]);
        float g[N], dg[N], sg[N], f[N], u[N], v[N];
        for (int k = 0; k < N; k++) {
            const float t = k - p[0];
            g[k] = std::exp(-.5f * iv * t * t);
            dg[k] = g[k] * iv * t;
            sg[k] = dg[k] * t;
        }
        if (blur) {
            one_d(d, g, f);
            if (deriv) {
                one_d(d, dg, u);
                one_d(d, sg, v);
            }
        } else {
            std::memcpy(f, g, sizeof(g));
        }
        const int e[3] = {b.hi[0], b.hi[1], b.hi[2]};
        const int s[3] = {b.plane, b.nx, 1};
        for (int II = b.lo[i]; II < e[i]; II++)
            for (int JJ = b.lo[j]; JJ < e[j]; JJ++) {
                const float E = blur ? edge[i][II] * edge[j][JJ] : 1.f;
                const float I = blur ? hramp[i][II] * edge[j][JJ] : ramp[i][II];
                const float J = blur ? edge[i][II] * hramp[j][JJ] : ramp[j][JJ];
                const float A = p[2] * E + p[3] * I + p[4] * J;
                int k = (II-b.lo[i])*s[i] + (JJ-b.lo[j])*s[j];
                for (int DD = b.lo[d]; DD < e[d]; DD++, k += s[d]) {
                    out[k] = f[DD] * A;
                    if (deriv) {
                        jconv[0][k] = u[DD] * A;
                        jconv[1][k] = v[DD] * A;
                        jconv[2][k] = f[DD] * E;
                        jconv[3][k] = f[DD] * I;
                        jconv[4][k] = f[DD] * J;
                    }
                }
            }
    }

    void paint_line(const Atom& a, const Box& b, float* out, bool deriv, bool blur) {
        const auto& p = a.p;
        const float iv = std::exp(-2 * p[6]), A = p[7];
        const float vz = p[3]-p[0], vy = p[4]-p[1], vx = p[5]-p[2];
        const float il = 1.f / std::max(.1f, vz*vz + vy*vy + vx*vx);
        int k = 0;
        for (int z = b.lo[0]; z < b.hi[0]; z++)
            for (int y_ = b.lo[1]; y_ < b.hi[1]; y_++) {
                const float z0 = z-p[0], y0 = y_-p[1];
                const float c = il * (z0*vz + y0*vy), d = il * vx;
                const int ox = b.lo[2], ex = b.hi[2], o = k - b.lo[2];
                if (deriv) {
                    for (int x = ox; x < ex; x++) {
                        const float x0 = x-p[2];
                        const float t = std::clamp(c + x0*d, 0.f, 1.f);
                        const float rz = z0 - vz*t, ry = y0 - vy*t, rx = x0 - vx*t;
                        const float q = iv * (rz*rz + ry*ry + rx*rx);
                        const float gs = std::exp(-.5f*q);
                        const float f = A*gs, g = f*iv;
                        const float aa = (1-t)*g, bb = t*g;
                        shape[o+x] = f;
                        jac[0][o+x] = rz * aa;
                        jac[1][o+x] = ry * aa;
                        jac[2][o+x] = rx * aa;
                        jac[3][o+x] = rz * bb;
                        jac[4][o+x] = ry * bb;
                        jac[5][o+x] = rx * bb;
                        jac[6][o+x] = f * q;
                        jac[7][o+x] = gs;
                    }
                } else {
                    float* v = blur ? shape.data() : out;
                    for (int x = ox; x < ex; x++) {
                        const float x0 = x-p[2];
                        const float t = std::clamp(c + x0*d, 0.f, 1.f);
                        const float rz = z0 - vz*t, ry = y0 - vy*t, rx = x0 - vx*t;
                        const float q = iv * (rz*rz + ry*ry + rx*rx);
                        v[o+x] = A * std::exp(-.5f*q);
                    }
                }
                k += b.nx;
            }
        if (blur) {
            convolve(shape.data(), out, b);
            if (deriv)
                for (int i = 0; i < 8; i++) convolve(jac[i].data(), jconv[i].data(), b);
        }
    }

    void paint(const Atom& a, const Box& b, float* out, bool deriv, bool blur = true) {
        if (a.family == Spot) paint_spot(a, b, out, deriv, blur);
        else if (a.family == Plane) paint_plane(a, b, out, deriv, blur);
        else paint_line(a, b, out, deriv, blur);
    }

    void add(const Atom& a, float s) {
        const Box b = fit_box(a);
        paint(a, b, a0.data(), false);
        paint(a, b, a1.data(), false, false);
        int v = 0;
        for (int z = b.lo[0]; z < b.hi[0]; z++)
            for (int i = b.lo[1]; i < b.hi[1]; i++) {
                const int o = index(z, i, b.lo[2]);
                for (int x = 0; x < b.nx; x++) {
                    pred[o+x] += s * a0[v+x];
                    estimate[o+x] += s * a1[v+x];
                }
                v += b.nx;
            }
    }

    void choose_weights(const Atom& initial, const Box& b, bool old) {
        if (old || initial.family == Plane) paint(initial, b, a1.data(), false);
        int v = 0;
        for (int z = b.lo[0]; z < b.hi[0]; z++)
            for (int i = b.lo[1]; i < b.hi[1]; i++) {
                const int o = index(z, i, b.lo[2]);
                for (int x = 0; x < b.nx; x++) {
                    const float base = std::max(0.f, pred[o+x]);
                    const float init = (old || initial.family == Plane) ? a1[v+x] : 0.f;
                    const float veil = (old ? .17f : .69f) * rough[o+x];
                    const float r = y[o+x] - base;
                    if (score_refinement && initial.family != Plane) {
                        target[v+x] = pg_y[o+x] - base;
                        invsd[v+x] = pg_w[o+x];
                    } else {
                        target[v+x] = r;
                        invsd[v+x] = 1.f / std::sqrt(gvar + shot * std::max(base + init, veil));
                    }
                }
                v += b.nx;
            }
    }

    static float data_cost(const float* v, const float* t, const float* w, int n, bool robust) {
        float f = 0;
        if (robust) {
            for (int i = 0; i < n; i++) {
                const float r = (t[i] - (v ? v[i] : 0.f)) * w[i];
                f += 12.5f * (std::sqrt(1.f + .16f * r * r) - 1.f);
            }
        } else {
            for (int i = 0; i < n; i++) {
                const float r = (t[i] - (v ? v[i] : 0.f)) * w[i];
                f += r * r;
            }
        }
        return f;
    }

    bool poisson_gaussian_score() {
        // At the fitted mean, use the censored Poisson/Gaussian score as
        // the right-hand side of a Fisher-scoring subproblem.
        const float sn = sigma/shot;
        if (!(sn > .35f && sn < 1.42f) || peak/shot > 180.f) return false;
        constexpr int terms = 11;
        const float pn = 1.f/shot;
        const float inv = .5f/(sn*sn);
        float zero_terms[terms];
        float scale = 1.f;
        for (int k = 0; k < terms; k++) {
            zero_terms[k] = scale * .5f * std::erfc(k/(1.41421356f*sn));
            scale /= (k+1);
        }
        for (int i = 0; i < S; i++) {
            const float u = std::max(0.f, pred[i]);
            const float var = gvar + shot * u;
            pg_w[i] = 1.f/std::sqrt(var);
            const float v = std::max(0.f,y[i]) * pn;
            const float rate = std::max(.00015f, u*pn);
            const int k0 = std::max(0, int(v) - 5);
            float s = 0, d = 0;
            if (v <= 0) {
                for (int k = terms-1; k >= 0; k--) {
                    d = d * rate + s;
                    s = s * rate + zero_terms[k];
                }
            } else {
                float q[terms], c = 1.f;
                for (int k = 0; k < terms; k++) {
                    const float delta = v - (k0+k);
                    q[k] = c * std::exp(-inv*delta*delta);
                    c /= (k0+k+1);
                }
                for (int k = terms-1; k >= 0; k--) {
                    d = d * rate + s;
                    s = s * rate + q[k];
                }
            }
            float move = 0;
            if (s > 1.e-29f) move = ((k0/rate + d/s) - 1.f) * (var*pn);
            const float r = y[i] - u, stddev = 1.f/pg_w[i];
            if (s < 1.e-29f || move > 3.2f * std::max(r,0.f) + 3.8f*stddev ||
                move < 3.2f * std::min(r,0.f) - 3.8f*stddev) move = r;
            pg_y[i] = u + (.76f*move + .24f*r);
        }
        return true;
    }

    float cost(const Atom& a, const Box& b, const float* v, float pm, float pc) {
        const float d = (a.p[a.sw()] - pm);
        float loss = data_cost(v, target.data(), invsd.data(), b.size, a.family == Plane) + pc*d*d;
        if (a.family == Spot && amp_pc > 0.f) {
            const float q = std::log(std::max(a.p[4], .08f*spot_amp) / spot_amp);
            loss += amp_pc*q*q;
        }
        return loss;
    }

    void bound(Atom& a, const Atom& p0) const {
        auto& p = a.p;
        if (a.family == Spot) {
            for (int i = 0; i < 3; i++)
                p[i] = std::clamp(p[i], std::max(-.85f,p0.p[i]-3.f), std::min(N-.15f,p0.p[i]+3.f));
            p[3] = std::clamp(p[3], -.755f, .53f);
        } else if (a.family == Plane) {
            p[0] = std::clamp(p[0], -.82f, N-.18f);
            p[1] = std::clamp(p[1], -.79f, 1.05f);
            float sum = std::abs(p[3]) + std::abs(p[4]);
            if (sum > p[2] && sum > 0) {
                const float c = std::max(0.f, p[2]) / sum;
                p[3] *= c;
                p[4] *= c;
            }
        } else {
            for (int i = 0; i < 6; i++)
                p[i] = std::clamp(p[i], std::max(-8.f,p0.p[i]-9.f), std::min(N+7.f,p0.p[i]+9.f));
            p[6] = std::clamp(p[6], -.69f, .94f);
        }
        p[a.amp()] = std::clamp(p[a.amp()], 0.f, std::max(12.f * peak, .25f * sigma));
    }

    float fit(Atom& p, bool old, int max_iter) {
        const Atom initial = p;
        const Box b = fit_box(p);
        choose_weights(p, b, old);
        const int n = p.count(), swi = p.sw(), ai = p.amp();
        const bool robust = p.family == Plane;
        const bool ed = robust && std::min(p.p[0], N-1-p.p[0]) <= 3.5f;
        const float plog = std::log(robust ? (ed ? .68f : .75f) : p.family == Line ? 1.02f : .88f);
        const float pc = robust ? (ed ? 18.f : 8.f) : p.family == Line ? 17.f : 25.f;
        float loss_after = data_cost(nullptr, target.data(), invsd.data(), b.size, robust);
        const float loss_null = loss_after;
        float damp = .025f;
        for (int it = 0; it < max_iter && (it == 0 || now() < deadline); it++) {
            paint(p, b, a0.data(), true);
            const float loss = cost(p, b, a0.data(), plog, pc);
            loss_after = loss;
            if (robust) {
                for (int k = 0; k < b.size; k++) {
                    const float r = (target[k]-a0[k])*invsd[k];
                    const float w = invsd[k] * std::sqrt(1.f / std::sqrt(1.f + .16f*r*r));
                    linw[k] = w;
                    linres[k] = w * (target[k] - a0[k]);
                }
            } else {
                for (int k = 0; k < b.size; k++) {
                    const float w = invsd[k];
                    linres[k] = w * (target[k] - a0[k]);
                    linw[k] = w;
                }
            }
            double g[8]{}, A[8][8]{};
            for (int j = 0; j < n; j++) {
                float* v = jconv[j].data();
                for (int k = 0; k < b.size; k++) v[k] *= linw[k];
                g[j] = dot(v, linres.data(), b.size);
                for (int i = 0; i <= j; i++) A[j][i] = dot(v, jconv[i].data(), b.size);
            }
            g[swi] -= pc * (p.p[swi] - plog);
            A[swi][swi] += pc;
            if (p.family == Spot && amp_pc > 0.f) {
                const float a = std::max(p.p[4], .08f*spot_amp);
                const float inv = 1.f / a;
                g[ai] -= amp_pc * inv * std::log(a/spot_amp);
                A[ai][ai] += amp_pc * inv * inv;
            }
            bool accept = false;
            float scaled_step = 0;
            for (int k = 0; k < 5; k++) {
                float step[8]{};
                if (!solve_system(A, g, n, damp, step)) break;
                float factor = 1.f;
                for (int j = 0; j < n; j++) {
                    const float limit = j == swi ? .28f : j >= ai ? std::max(.35f * p.p[ai], .44f * peak) : 1.7f;
                    factor = std::min(factor, limit / (1.e-9f + std::abs(step[j])));
                }
                Atom q = p;
                for (int j = 0; j < n; j++) q.p[j] += step[j] * factor;
                bound(q, initial);
                paint(q, b, a1.data(), false);
                const float trial = cost(q, b, a1.data(), plog, pc);
                if (trial < loss) {
                    for (int j = 0; j < n; j++)
                        scaled_step = std::max(scaled_step, std::abs(q.p[j] - p.p[j]));
                    p = q;
                    loss_after = trial;
                    damp = std::max(damp * .34f, .00014f);
                    accept = true;
                    break;
                }
                damp *= 5.f;
            }
            if (!accept || scaled_step < .0025f || now() >= deadline) break;
            if (it > 0 && loss - loss_after < .002f) break;
        }
        return loss_null - loss_after;
    }

    void print(const Atom& a, float score) const {
        if (!debug) return;
        std::fprintf(stderr, "%s(%d)", a.family == Spot ? "spot" : a.family == Plane ? "plane" : "line", a.axis);
        for (int j = 0; j < a.count(); j++)
            std::fprintf(stderr, " %6.3f", j == a.sw() ? std::exp(a.p[j]) : a.p[j]);
        std::fprintf(stderr, "  chi2 %7.2f\n", score);
    }

    float propose_plane(int d, Atom& out) {
        const int i = d == 0 ? 1 : 0, j = d == 2 ? 1 : 2;
        const int s[3] = {N*N, N, 1};
        float tbl[N][N], norm[N]{};
        for (int k = 0; k < N; k++) {
            float g[N];
            for (int x = 0; x < N; x++) g[x] = std::exp(-.5f * ((x-k)/.75f) * ((x-k)/.75f));
            one_d(d, g, tbl[k]);
            norm[k] = dot(tbl[k], tbl[k], N);
        }
        float v[(N-4)*(N-4)];
        float best = -1.f;
        for (int k = 0; k < N; k++) {
            bool valid = true;
            for (const auto& a : atoms)
                if (a.family == Plane && a.axis == d && std::abs(k-a.p[0]) < 5.5f) valid = false;
            if (!valid) continue;
            int a = 0;
            const float inv = 1.f / std::max(norm[k], 1.e-10f);
            for (int I = 2; I < N-2; I++)
                for (int J = 2; J < N-2; J++) {
                    const int o = s[i]*I + s[j]*J;
                    float t = 0;
                    for (int M = 0; M < N; M++) t += tbl[k][M] * resid[o+s[d]*M];
                    v[a++] = t * inv / std::max(edge[i][I]*edge[j][J], .01f);
                }
            const int q = a * 3 / 10;
            std::nth_element(v, v + q, v + a);
            const float r = v[q];
            if (r > best && r > std::max(2.05f * sigma, .071f * peak)) {
                std::nth_element(v, v + a*8/10, v + a);
                if (r < .33f * v[a*8/10]) continue;
                std::nth_element(v, v + a/2, v + a);
                out.family = Plane;
                out.axis = d;
                out.p[0] = k;
                out.p[1] = std::log(std::min(k,N-1-k) <= 3 ? .68f : .75f);
                out.p[2] = v[a/2];
                best = r;
            }
        }
        return best;
    }

    void find_planes() {
        int count = 0;
        for (; count < 4 && now() + .009 < deadline; count++) {
            for (int i = 0; i < S; i++) resid[i] = y[i] - pred[i];
            Atom best, a;
            float vmax = -1;
            for (int i = 0; i < 3; i++) {
                const float v = propose_plane(i, a);
                if (v > vmax) { vmax = v; best = a; }
            }
            if (vmax < 0) break;
            const float score = fit(best, false, 16);
            if (score < 450.f || best.p[2] < 2.f * sigma) break;
            print(best, score);
            atoms.push_back(best);
            add(best, 1);
        }
        for (int it = 0; it < 3 && count; it++) {
            for (auto& a : atoms) {
                if (now() + .008 >= deadline) return;
                add(a, -1);
                fit(a, true, 11);
                add(a, 1);
            }
        }
    }

    float propose_spot(Atom& a, const Atom* near = nullptr) {
        for (int i = 0; i < S; i++) {
            const float q = 1.f / (gvar + shot * std::max(.69f * rough[i], pred[i]));
            w_global[i] = q;
            wres[i] = q * (y[i] - pred[i]);
        }
        distribute(spots_t, wres.data(), cov.data());
        distribute(spots_sq, w_global.data(), diag.data());
        float maxscore = 0;
        int j = 0;
        float vz = 0, vy = 0, vx = 0, invl = 0, rr = 0;
        if (near) {
            const auto& p = near->p;
            vz = p[3] - p[0]; vy = p[4] - p[1]; vx = p[5] - p[2];
            invl = 1.f / std::max(.01f, vz*vz + vy*vy + vx*vx);
            rr = std::max(2.25f, 1.55f*std::exp(p[6]));
            rr *= rr;
        }
        for (int i = 0; i < S; i++)
            if (cov[i] > 0) {
                const float sc = cov[i] * cov[i] / std::max(diag[i], 1.e-8f);
                if (sc <= maxscore) continue;
                if (near) {
                    const float z = i/(N*N)-near->p[0], y_ = (i/N)%N-near->p[1], x = i%N-near->p[2];
                    const float t = std::clamp((z*vz+y_*vy+x*vx)*invl, 0.f, 1.f);
                    const float p = z-t*vz, q = y_-t*vy, s = x-t*vx;
                    if (p*p + q*q + s*s > rr) continue;
                }
                maxscore = sc;
                j = i;
            }
        a.family = Spot;
        a.p = {float(j/(N*N)), float((j/N)%N), float(j%N), std::log(.87f),
               std::clamp(cov[j] / std::max(diag[j],1.e-8f), .08f*peak, 9.f*peak)};
        return maxscore;
    }

    float comparison_gain() const {
        float v = 0;
        for (int i = 0; i < S; i++) {
            const float d = pred[i] - saved[i];
            v += d * (2.f*(y[i]-saved[i])-d)
                 / (gvar + shot * std::max(.69f*rough[i], saved[i]));
        }
        return v;
    }

    int compete(const Atom& line, Atom* alternative, float& gain) {
        saved = pred;
        add(line, 1);
        const float line_gain = comparison_gain();
        add(line, -1);
        int n = 0;
        const int limit = line.length() > 12.f ? 3 : 2;
        for (; n < limit && now() + .013 < deadline; n++) {
            Atom& q = alternative[n];
            if (propose_spot(q, &line) < 16.f || fit(q, false, 12) < 23.f) break;
            add(q, 1);
        }
        if (n > 1)
            for (int i = 0; i < n && now() + .012 < deadline; i++) {
                Atom& q = alternative[i];
                add(q, -1);
                fit(q, true, 7);
                add(q, 1);
            }
        gain = comparison_gain();
        float width_cost = 0;
        if (recovery_search) {
            for (int i = 0; i < n; i++) {
                const float extra = std::max(0.f, alternative[i].p[3] - .266f);
                width_cost += (1900.f + .27f*line_gain) * extra*extra;
            }
        }
        for (int i = 0; i < n; i++) add(alternative[i], -1);
        if (debug > 1)
            std::fprintf(stderr, "comparison line %.1f vs %d spots %.1f\n", line_gain, n, gain);
        const float complexity = 4.5f * (5*n - 8);
        return gain > line_gain + complexity + width_cost ? n : 0;
    }

    static float interp(const float* u, float z, float y_, float x) {
        if (z < -.25f || z > N-.75f || y_ < -.25f || y_ > N-.75f || x < -.25f || x > N-.75f)
            return 0.f;
        z = std::clamp(z, 0.f, N-1.f);
        y_ = std::clamp(y_, 0.f, N-1.f);
        x = std::clamp(x, 0.f, N-1.f);
        int i = std::min(N-2, int(z)), j = std::min(N-2, int(y_)), k = std::min(N-2, int(x));
        const float a = z-i, b = y_-j, c = x-k;
        const float* w = u+index(i,j,k);
        const float u0 = w[0] + c*(w[1]-w[0]);
        const float u1 = w[N] + c*(w[N+1]-w[N]);
        const float u2 = w[N*N] + c*(w[N*N+1]-w[N*N]);
        const float u3 = w[N*N+N] + c*(w[N*N+N+1]-w[N*N+N]);
        const float p = u0 + b*(u1-u0), q = u2 + b*(u3-u2);
        return p + a*(q-p);
    }

    static bool same_ray(const Atom& a, const Atom& b) {
        float xx = 0, yy = 0, xy = 0, dd = 0, dx = 0, dy = 0;
        for (int i = 0; i < 3; i++) {
            const float x = a.p[i+3] - a.p[i];
            const float y_ = b.p[i+3] - b.p[i];
            const float d = .5f*(a.p[i+3] + a.p[i] - b.p[i+3] - b.p[i]);
            xx += x*x;
            yy += y_*y_;
            xy += x*y_;
            dd += d*d;
            dx += d*x;
            dy += d*y_;
        }
        return xy*xy > .905f*xx*yy && dd - dx*dx/xx < 5.1f &&
               dd - dy*dy/yy < 5.1f && dd < .17f*(xx+yy)+5.f;
    }

    static void admit_seed(std::array<LineSeed, 4>& list, const Atom& a, float value) {
        if (value <= list.back().value) return;
        unsigned i = 0;
        for (; i < list.size() && list[i].value > 0; i++) {
            if (same_ray(list[i].atom, a)) {
                if (value <= list[i].value) return;
                break;
            }
        }
        if (i == list.size()) i--;
        while (i && value > list[i-1].value) {
            list[i] = list[i-1];
            i--;
        }
        list[i] = {a, value};
    }

    std::array<LineSeed, 4> propose_lines() {
        if (!inverse) {
            const float alpha = std::clamp(.013f * std::pow(sigma / (.073f*peak), .64f),
                                           .0055f, .048f);
            inverse = std::make_unique<Wiener>(h, alpha);
        }
        for (int i = 0; i < S; i++) resid[i] = y[i] - pred[i];
        inverse->apply(resid.data(), ray.data());
        wres = ray;

        constexpr int np = 18, nd = 150;
        constexpr float inc = 1.45f;
        float inv[2*np+2];
        for (int i = 1; i <= 2*np+1; i++) inv[i] = 1.f/std::sqrt(float(i));
        std::array<std::array<float, 3>, nd + 3> dirs{};
        dirs[0] = {inc, 0, 0}; dirs[1] = {0, inc, 0}; dirs[2] = {0, 0, inc};
        for (int i = 0; i < nd; i++) {
            const float z = (i + .5f)/nd;
            const float r = std::sqrt(1-z*z);
            const float t = i * (pi * (3.f-2.2360679775f));
            dirs[i+3] = {inc*z, inc*r*std::cos(t), inc*r*std::sin(t)};
        }

        std::array<LineSeed, 4> list{};
        float tops = 0;
        for (int m = 0; m < 62; m++) {
            if (m % 8 == 0 && now() + .006 >= deadline) break;
            const auto im = std::max_element(wres.begin(), wres.end());
            const int is = int(im - wres.begin());
            const float val = *im;
            if (m == 0) tops = val;
            if (val < std::max({1.05f*sigma, .15f*tops, .098f*peak})) break;
            const int z0 = is/(N*N), y0 = (is/N)%N, x0 = is%N;
            for (int z = std::max(0,z0-2); z <= std::min(N-1,z0+2); z++)
                for (int y_ = std::max(0,y0-2); y_ <= std::min(N-1,y0+2); y_++)
                    for (int x = std::max(0,x0-2); x <= std::min(N-1,x0+2); x++)
                        wres[index(z,y_,x)] = -1.f;

            const float trim = std::max({.13f * val, .52f * sigma, .035f * peak});
            const float clip = recovery_search ? .65f*tops : 1.e10f;
            const float start = std::min(val, clip);
            for (const auto& d : dirs) {
                float sum[2][np+1]{};
                int len[2]{};
                for (int side = 0; side < 2; side++) {
                    const float sign = side ? 1.f : -1.f;
                    int miss = 0;
                    for (int i = 1; i <= np; i++) {
                        const float t = i*sign;
                        const float f = interp(ray.data(), z0 + t*d[0], y0 + t*d[1], x0 + t*d[2]);
                        if (f < trim) {
                            if (++miss >= 2) break;
                        } else {
                            miss = 0;
                            len[side] = i;
                        }
                        sum[side][i] = sum[side][i-1] + std::clamp(f, -trim, clip);
                    }
                }
                if (len[0]+len[1] < 5) continue;
                float short_sc = start, long_sc = 0;
                int lo = 0, hi = 0;
                for (int i = 0; i <= len[0]; i++)
                    for (int j = 0; j <= len[1]; j++) {
                        const int length = i+j+1;
                        const float r = (start + sum[0][i] + sum[1][j]) * inv[length];
                        if (length < 6) { short_sc = std::max(short_sc,r); continue; }
                        if (r > long_sc) { long_sc = r; lo = i; hi = j; }
                    }
                if (long_sc < short_sc * 1.024f) continue;
                const float mean = (start + sum[0][lo] + sum[1][hi]) / (lo+hi+1);
                // Reward sustained signal over a bright landmark with weak tails.
                const float sc = long_sc * std::min(1.f, (.5f*mean/std::max(start,.001f)+.62f));
                if (sc <= list[recovery_search ? 3 : 0].value) continue;
                Atom a;
                a.family = Line;
                a.p = {z0 + (.53f-lo)*d[0], y0 + (.53f-lo)*d[1], x0 + (.53f-lo)*d[2],
                         z0 + (hi-.53f)*d[0], y0 + (hi-.53f)*d[1], x0 + (hi-.53f)*d[2],
                         std::log(1.05f), std::max(1.58f*mean, .12f*peak)};
                if (recovery_search) admit_seed(list, a, sc);
                else list[0] = {a, sc};
            }
        }
        return list;
    }

    int find_features(int limit, bool use_lines, int line_trials = 2) {
        int added = 0, failures = 0;
        for (int i = 0; i < limit && atoms.size() < 30 && now() + .012 < deadline; i++) {
            Atom a;
            float score = -1.f;
            if (use_lines && failures < line_trials && now() + .025 < deadline) {
                auto proposals = propose_lines();
                const float threshold = std::max({3.72f*sigma, .29f*peak, .001f});
                if (proposals[0].value >= threshold) {
                    Atom short_line;
                    float short_score = 0;
                    const int count = recovery_search ? 2 : 1;
                    for (int n = 0; n < count; n++) {
                        if (proposals[n].value < std::max(threshold, .65f*proposals[0].value) ||
                            (n && now() + .036 >= deadline)) break;
                        Atom q = proposals[n].atom;
                        float f = fit(q, false, n == 0 ? 16 : 9);
                        const float l = q.length();
                        if (f < 68.f || l < 4.3f || l < 2.65f * std::exp(q.p[6])) {
                            if (l > 1.9f && f > short_score) { short_score = f; short_line = q; }
                        } else if (f > score) {
                            a = q;
                            score = f;
                        }
                    }
                    if (score < 0 && debug > 1 && short_score > 0) print(short_line, -short_score);
                    if (score >= 0 && now() + .027 < deadline) {
                        Atom pts[3];
                        float gain;
                        const int n = compete(a, pts, gain);
                        if (n) {
                            for (int j = 0; j < n; j++) {
                                print(pts[j], gain);
                                atoms.push_back(pts[j]);
                                add(pts[j], 1);
                            }
                            added += n;
                            failures++;
                            continue;
                        }
                    }
                }
                if (score < 0) failures++;
                else failures = 0;
            }
            if (score < 0) {
                if (propose_spot(a) < 14.f) break;
                score = fit(a, false, 16);
                if (score < 23.5f) break;
            }
            print(a, score);
            atoms.push_back(a);
            add(a, 1);
            added++;
            if (recovery_search && a.family == Line && now() + .032 < deadline)
                for (std::size_t j = 0; j + 1 < atoms.size(); j++) {
                    auto& u = atoms[j];
                    if (u.family != Line || now() + .020 >= deadline) continue;
                    add(u,-1); fit(u,true,5); add(u,1);
                }
        }
        return added;
    }

    void refine(int rounds, int n = 9) {
        for (int k = 0; k < rounds; k++) {
            for (auto& a : atoms) {
                if (now() + .006 >= deadline) return;
                if (a.family == Plane) continue;
                add(a, -1);
                const float score = fit(a, true, n);
                add(a, 1);
                if (debug > 1) print(a, score);
            }
        }
    }

    bool calibrate_spots() {
        if (atoms.size() < 3) return false;
        float amps[32];
        int count = 0;
        float spot_mass = 0.f, line_mass = 0.f;
        for (const auto& p : atoms) {
            if (p.family == Plane) return false;
            if (p.family == Line) {
                line_mass += p.p[7]*std::exp(2*p.p[6])*(p.length()+2.5f*std::exp(p.p[6]));
            } else {
                spot_mass += 2.5066f*p.p[4]*std::exp(3*p.p[3]);
                if (p.p[4] > .32f*peak && p.p[3] > -.64f && p.p[3] < .39f)
                    amps[count++] = std::log(p.p[4]);
            }
        }
        if (count < 2 || line_mass > .25f*spot_mass) return false;
        std::sort(amps, amps+count);
        const float loc = .5f*(amps[count/2]+amps[(count-1)/2]);
        const float d = loc - std::log(.875f);
        const float trust = .77f*std::exp(-3.2f*d*d);
        spot_amp = std::exp(loc - trust*d);
        amp_pc = 13.f;
        if (debug > 1) std::fprintf(stderr, "spot brightness prior %.3f, strength %.1f\n", spot_amp, amp_pc);
        return true;
    }

    float pair_penalty(const Atom& a) const {
        const float w = a.p[3]-std::log(.88f);
        if (amp_pc <= 0.f) return 25.f*w*w;
        const float e = std::log(std::max(a.p[4], .08f*spot_amp) / spot_amp);
        return 25.f*w*w + amp_pc*e*e;
    }

    void overlap_weights(const Box& b) {
        int v = 0;
        for (int z = b.lo[0]; z < b.hi[0]; z++)
            for (int i = b.lo[1]; i < b.hi[1]; i++) {
                const int o = index(z, i, b.lo[2]);
                for (int x = 0; x < b.nx; x++) {
                    target[v+x] = y[o+x] - pred[o+x];
                    invsd[v+x] = 1.f/std::sqrt(gvar+shot*std::max(saved[o+x],.22f*rough[o+x]));
                }
                v += b.nx;
            }
    }

    static float separation(const Atom& u, const Atom& v) {
        float rr = 0;
        for (int d = 0; d < 3; d++) {
            const float p = u.p[d] - v.p[d];
            rr += p*p;
        }
        return rr;
    }

    float pair_loss(const Atom& u, const Atom& v, const Box& b, bool deriv = false) {
        paint(u, b, a0.data(), deriv);
        if (deriv)
            for (int i = 0; i < 5; i++)
                std::copy_n(jconv[i].data(), b.size, jac[i].data());
        paint(v, b, a1.data(), deriv);
        for (int i = 0; i < b.size; i++) a0[i] += a1[i];
        const float d = std::max(0.f, 3.24f - separation(u,v));
        return data_cost(a0.data(), target.data(), invsd.data(), b.size, false)
               + pair_penalty(u) + pair_penalty(v) + 1.5f*d*d;
    }

    float fit_pair(Atom& p, Atom& q, const Box& b, int max_iter) {
        const Atom p0 = p, q0 = q;
        float damp = .018f;
        float loss = 0;
        for (int it = 0; it < max_iter && (it == 0 || now() < deadline); it++) {
            loss = pair_loss(p,q,b,true);
            for (int i = 0; i < b.size; i++) linres[i] = invsd[i] * (target[i]-a0[i]);
            double g[10]{}, A[10][10]{};
            const float* J[10];
            for (int j = 0; j < 10; j++) {
                float* v = (j < 5 ? jac[j].data() : jconv[j-5].data());
                J[j] = v;
                for (int k = 0; k < b.size; k++) v[k] *= invsd[k];
                g[j] = dot(v, linres.data(), b.size);
                for (int i = 0; i <= j; i++) A[j][i] = dot(v, J[i], b.size);
            }
            for (int j = 0; j <= 5; j += 5) {
                const auto& a = (j ? q : p).p;
                A[j+3][j+3] += 25.f;
                g[j+3] -= 25.f*(a[3]-std::log(.88f));
                if (amp_pc > 0.f) {
                    const float v = std::max(a[4], .08f*spot_amp);
                    A[j+4][j+4] += amp_pc / (v*v);
                    g[j+4] -= amp_pc * std::log(v/spot_amp)/v;
                }
            }
            bool moved = false;
            float reduction = 0;
            for (int j = 0; j < 5; j++) {
                float step[10];
                if (!solve_system(A,g,10,damp,step)) break;
                Atom a=p, c=q;
                for (int off = 0; off <= 5; off += 5) {
                    auto& aa = off ? c : a;
                    float rate = 1.f;
                    for (int i = 0; i < 5; i++) {
                        float limit = i == 3 ? .20f : i == 4 ? std::max(.30f*aa.p[4], .25f*peak) : .75f;
                        rate = std::min(rate, limit/(std::abs(step[off+i])+1.e-8f));
                    }
                    for (int i = 0; i < 5; i++) aa.p[i] += rate*step[off+i];
                    bound(aa,off ? q0 : p0);
                }
                const float trial = pair_loss(a,c,b);
                if (trial < loss) {
                    p=a; q=c;
                    moved=true;
                    reduction=loss-trial;
                    loss=trial;
                    damp=std::max(damp*.40f,.0005f);
                    break;
                }
                damp *= 5;
            }
            if (!moved || (it > 1 && reduction < .02f)) break;
        }
        return loss;
    }

    void split_candidates() {
        // Check whether an unusually wide landmark is better explained by
        // two close emitters. Test in observation space before replacing it.
        int tested = 0;
        const unsigned initial = atoms.size();
        for (unsigned k = 0; k < initial && now() + .032 < deadline && tested < 7; k++) {
            const Atom a = atoms[k];
            if ((a.family != Spot || a.p[3] < .080f) &&
                (a.family != Line || a.length() > 7.65f || a.length() < 3.4f)) continue;
            if (a.family == Spot && a.p[4] < .32f*peak) continue;
            if (atoms.size() >= 30) break;
            tested++;
            saved = pred;
            add(a,-1);
            const Box b = fit_box(a);
            overlap_weights(b);
            paint(a,b,a0.data(),false);
            const float null = data_cost(a0.data(),target.data(),invsd.data(),b.size,false)
                + (a.family == Spot ? pair_penalty(a) : 17.f*std::pow(a.p[6]-std::log(1.02f),2));
            Atom start[2]{};
            float best = 1.e30f;
            if (a.family == Line) {
                for (int i = 0; i < 2; i++) {
                    auto& p = start[i].p;
                    for (int d = 0; d < 3; d++)
                        p[d] = std::clamp(.89f*a.p[3*i+d]+.11f*a.p[3*(1-i)+d], -.8f, N-.2f);
                    const float s = std::clamp(std::exp(a.p[6]),.68f,1.21f);
                    p[3]=std::log(s);
                    p[4]=a.p[7]*std::exp(2*a.p[6])*(a.length()+2.5f*s)/(5.013f*s*s*s);
                }
            } else {
                // A small angular bank only seeds the model; the positions
                // of both emitters are free in the subsequent fit.
                for (int z = -1; z <= 1; z++)
                    for (int y_ = -1; y_ <= 1; y_++)
                        for (int x = -1; x <= 1; x++) {
                            if (9*z+3*y_+x <= 0) continue;
                            Atom p = a, q = a;
                            const float s = std::exp(a.p[3]);
                            const float d = std::clamp(.97f*s,.95f,1.72f)/std::sqrt(float(z*z+y_*y_+x*x));
                            const int u[] = {z,y_,x};
                            for (int i = 0; i < 3; i++) {
                                p.p[i]=std::clamp(a.p[i]+d*u[i],-.82f,N-.18f);
                                q.p[i]=std::clamp(a.p[i]-d*u[i],-.82f,N-.18f);
                            }
                            p.p[3]=q.p[3]=std::clamp(a.p[3]-.18f,-.43f,.16f);
                            p.p[4]=q.p[4]=.50f*a.p[4]*std::exp(3*(a.p[3]-p.p[3]));
                            const float qloss = pair_loss(p,q,b);
                            if (qloss < best) {
                                best=qloss;
                                start[0]=p; start[1]=q;
                            }
                        }
            }
            const float loss = fit_pair(start[0],start[1],b,15);
            const float gain = null-loss;
            const bool good = separation(start[0],start[1]) > 2.05f &&
                std::min(start[0].p[4],start[1].p[4]) > .29f*peak &&
                std::max(start[0].p[3],start[1].p[3]) < .355f;
            if (debug > 1)
                std::fprintf(stderr,"split gain %.1f of %s, centers apart %.2f\n", gain,
                             a.family==Spot ? "spot" : "line", std::sqrt(separation(start[0],start[1])));
            if (good && gain > (a.family == Spot ? 23.5f : 9.5f)) {
                atoms[k] = start[0];
                atoms.push_back(start[1]);
                add(start[0],1); add(start[1],1);
            } else {
                add(a,1);
            }
        }
    }

    void smooth_spot(const Atom& a, const Box& b, const float* variance, float weight) {
        const auto& p = a.p;
        const float iv = std::exp(-2*p[3]);
        float f[3][N];
        for (int d = 0; d < 3; d++) {
            const float vv = 1.f / (1.f + iv*variance[d]);
            for (int i = b.lo[d]; i < b.hi[d]; i++)
                f[d][i] = std::sqrt(vv) * std::exp(-.5f*iv*vv*(i-p[d])*(i-p[d]));
        }
        for (int z = b.lo[0]; z < b.hi[0]; z++)
            for (int i = b.lo[1]; i < b.hi[1]; i++) {
                const float s = weight * p[4] * f[0][z] * f[1][i];
                const int o = index(z,i,0);
                for (int x = b.lo[2]; x < b.hi[2]; x++) estimate[o+x] += s * f[2][x];
            }
    }

    void average_spots() {
        for (const auto& a : atoms) {
            if (a.family != Spot || !(a.p[4] > 0) || now() + .004 >= deadline) continue;
            const Box b = fit_box(a);
            paint(a,b,a0.data(),true);
            int v = 0;
            for (int z = b.lo[0]; z < b.hi[0]; z++)
                for (int i = b.lo[1]; i < b.hi[1]; i++) {
                    const int o = index(z,i,b.lo[2]);
                    for (int x = 0; x < b.nx; x++) {
                        const float mu = std::max(pred[o+x], .16f*rough[o+x]);
                        linw[v+x] = 1.f/std::sqrt(gvar+shot*mu);
                    }
                    v+=b.nx;
                }
            double A[5][5]{};
            for (int j = 0; j < 5; j++) {
                float* vv = jconv[j].data();
                const float s = j == 4 ? a.p[4] : 1.f;
                for (int k = 0; k < b.size; k++) vv[k] *= s * linw[k];
                for (int i = 0; i <= j; i++) A[j][i] = dot(vv,jconv[i].data(),b.size);
            }
            A[3][3] += 25.f;
            A[4][4] += amp_pc + 2.f;
            float d[5]{}, ss[5]{};
            for (int j = 0; j < 4; j++) {
                double u[5]{};
                u[j] = 1.;
                float vec[5];
                if (!solve_system(A,u,5,.0002,vec)) continue;
                d[j] = std::max(0.f,vec[j]);
                if (j == 3) std::copy_n(vec,5,ss);
            }
            const float s2 = std::exp(2*a.p[3]);
            const float strength = .93f;
            const float factor = std::min(1.f,.021f/std::max(d[3],1.e-9f));
            float variance[3], zero[3]{};
            for (int i = 0; i < 3; i++) {
                const float reduced = d[3] > 1.e-9f ? factor*ss[i]*ss[i]/d[3] : 0;
                variance[i] = std::clamp(strength*(d[i]-reduced),0.f,.34f*s2);
            }
            smooth_spot(a,b,zero,-1.f);
            if (factor == 0.f || d[3] < .00025f) {
                smooth_spot(a,b,variance,1.f);
                continue;
            }
            Atom sides[2];
            float sum_mass = 2.f/3.f;
            for (int k = 0; k < 2; k++) {
                auto p = a;
                const float t = (k ? 1.f : -1.f)*std::sqrt(3.f*strength*factor/d[3]);
                for (int j = 0; j < 3; j++) p.p[j] += t*ss[j];
                p.p[3] = std::clamp(p.p[3]+t*ss[3],-.79f,.58f);
                p.p[4] *= std::exp(std::clamp(t*ss[4],-.82f,.82f));
                sum_mass += std::exp(3*(p.p[3]-a.p[3])) * (p.p[4]/a.p[4]) / 6.f;
                sides[k]=p;
            }
            smooth_spot(a,b,variance,2.f/(3.f*sum_mass));
            for (const auto& p : sides) smooth_spot(p,b,variance,1.f/(6.f*sum_mass));
        }
    }

    float model_evidence() const {
        float result = 0;
        for (int i = 0; i < S; i++)
            result += pred[i]*(2.f*y[i]-pred[i]) / (gvar+shot*std::max(0.f,.81f*rough[i]));
        for (const auto& a : atoms) {
            result -= 5.8f*a.count();
            if (a.family == Spot) {
                const float wide = std::max(0.f,a.p[3]-.245f);
                result -= 1450.f*wide*wide;
            }
        }
        return result;
    }

    void try_again() {
        if (now() + .148 >= deadline || atoms.empty()) return;
        int n_spots = 0, n_wide = 0, n_lines = 0;
        for (const auto& a : atoms) {
            if (a.family == Plane) return;
            if (a.family == Line) n_lines++;
            if (a.family == Spot) {
                n_spots++;
                n_wide += (a.p[3] > .253f);
            }
        }
        const bool unlikely = (n_spots >= 7 && n_wide >= 2) ||
                              (n_lines && n_spots >= 5 && n_wide >= 1);
        if (!unlikely) return;
        const float first = model_evidence();
        const auto original = atoms;
        atoms.clear();
        pred.fill(0);
        estimate.fill(0);
        const double whole = deadline;
        deadline -= .013;
        recovery_search = true;
        find_features(18,true,2);
        refine(2,9);
        recovery_search = false;
        deadline = whole;
        const float second = model_evidence();
        if (debug > 1) std::fprintf(stderr,"model evidence first %.1f second %.1f\n",first,second);
        if (second < first + 13.f || !std::isfinite(second)) {
            atoms = original;
            pred.fill(0);
            estimate.fill(0);
            for (const auto& a : atoms) add(a,1);
        }
    }

    bool run(float* out);
};

bool Engine::run(float* out) {
    find_planes();
    const bool planes = !atoms.empty();
    find_features(19, !planes);
    refine(2, 10);
    const int m = find_features(5, !planes, 1);
    if (m) refine(2, 9);
    if (!planes) try_again();
    if (!planes) split_candidates();
    if (now() + .018 < deadline && calibrate_spots()) refine(2, 11);
    if (now() + .023 < deadline) {
        for (int k = 0; k < 2 && now() + .019 < deadline; k++) {
            score_refinement = poisson_gaussian_score();
            if (!score_refinement) break;
            refine(1, 5);
        }
        score_refinement = false;
    }
    if (atoms.empty()) return false;
    average_spots();
    for (int i = 0; i < S; i++)
        out[i] = std::max(0.f, estimate[i]);
    return true;
}

} // namespace

extern "C" int recover(const float* y, const float* rough, const float* psf_marginals,
                       float sigma, float shot, double deadline, float* out, int debug) {
    try {
        auto engine = std::make_unique<Engine>(y, rough, psf_marginals, sigma, shot, deadline, debug);
        return engine->run(out);
    } catch (...) {
        return 0;
    }
}
