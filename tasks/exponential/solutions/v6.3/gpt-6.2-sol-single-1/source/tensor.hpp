#pragma once

// Only these stable C FFTW entry points are needed. The library is optional;
// a missing library or unavailable plan leaves the general solver unchanged.
class TensorFFT {
    void* handle = nullptr;
public:
    struct Plan;
    Plan* (*plan1)(int, Real*, Real*, int, unsigned) = nullptr;
    Plan* (*plan2)(int, int, Real*, Real*, int, int, unsigned) = nullptr;
    void (*exec)(Plan*, Real*, Real*) = nullptr;
    void (*destroy)(Plan*) = nullptr;
    void* (*allocate)(size_t) = nullptr;
    void (*release)(void*) = nullptr;

    TensorFFT() {
        handle = dlopen("libfftw3l.so.3", RTLD_NOW | RTLD_LOCAL);
        if (!handle) return;
        plan1 = reinterpret_cast<decltype(plan1)>(dlsym(handle,"fftwl_plan_r2r_1d"));
        plan2 = reinterpret_cast<decltype(plan2)>(dlsym(handle,"fftwl_plan_r2r_2d"));
        exec = reinterpret_cast<decltype(exec)>(dlsym(handle,"fftwl_execute_r2r"));
        destroy = reinterpret_cast<decltype(destroy)>(dlsym(handle,"fftwl_destroy_plan"));
        allocate = reinterpret_cast<decltype(allocate)>(dlsym(handle,"fftwl_malloc"));
        release = reinterpret_cast<decltype(release)>(dlsym(handle,"fftwl_free"));
    }
    bool available() const { return plan1 && plan2 && exec && destroy && allocate && release; }
};

class TensorGrid {
    enum Bound {Neumann, Dirichlet, Periodic};
    struct Dimension {
        int n;
        Bound bound;
        int diagonal(int j) const {
            if (n == 1) return 0;
            if (bound == Neumann && (j == 0 || j == n-1)) return 1;
            return 2;
        }
        int neighbors(int j, int (&ids)[3]) const {
            int num = 0;
            if (j > 0) ids[num++] = j-1;
            else if (bound == Periodic && n > 1) {
                ids[0] = 0;
                ids[1] = 1;
                ids[2] = n-1;
                return 3;
            }
            if (j == n-1 && bound == Periodic && n > 1) {
                ids[0] = 0;
                ids[1] = j-1;
                ids[2] = j;
                return 3;
            }
            ids[num++] = j;
            if (j+1 < n) ids[num++] = j+1;
            return num;
        }
        int norm() const {
            if (n == 1) return 1;
            if (bound == Dirichlet) return 2*(n+1);
            if (bound == Neumann) return 2*n;
            return n;
        }
        int kind(bool inv) const {
            if (bound == Dirichlet) return 7; // RODFT00
            if (bound == Neumann) return inv ? 4 : 5; // REDFT01, REDFT10
            return 2; // DHT
        }
        Real lambda(int j) const {
            if (n == 1) return 0;
            const Real pi = 3.141592653589793238462643383279502884L;
            const Real arg = bound == Neumann ? (pi*j)/(2*n) :
                             bound == Dirichlet ? (pi*(j+1))/(2*(n+1)) :
                             (pi*std::min(j,n-j))/n;
            const Real s = std::sin(arg);
            return 4*s*s;
        }
    };
    struct Model {
        Dimension x, y;
        Real c0, cx, cy, cxy;
        Real dd[3][3], ox[3], oy[3];
        void coefficients() {
            for (int j = 0; j < 3; ++j) {
                ox[j] = -(cx+cxy*j);
                oy[j] = -(cy+cxy*j);
                for (int i = 0; i < 3; ++i)
                    dd[j][i] = c0+cx*i+cy*j+cxy*j*i;
            }
        }
        bool row_matches(const Matrix& a, int i, int j) const {
            const int nx = x.n;
            const int r = j*nx+i;
            int xs[3], ys[3];
            const int ux = x.neighbors(i,xs), uy = y.neighbors(j,ys);
            const int dx = x.diagonal(i), dy = y.diagonal(j);
            int p = a.row[r], end = a.row[r+1];
            for (int l = 0; l < uy; ++l) for (int k = 0; k < ux; ++k) {
                const Real v = ys[l] == j ? (xs[k] == i ? dd[dy][dx] : ox[dy]) :
                                          (xs[k] == i ? oy[dx] : cxy);
                if (!v) continue;
                while (p < end && a.val[p] == 0) ++p;
                if (p == end || a.col[p] != ys[l]*nx+xs[k] || a.val[p] != v) return false;
                ++p;
            }
            while (p < end && a.val[p] == 0) ++p;
            return p == end;
        }
        bool matches(const Matrix& a) {
            coefficients();
            const int nx = x.n, ny = y.n;
            if (!row_matches(a,0,0) || !row_matches(a,nx-1,0) ||
                !row_matches(a,0,ny-1) || !row_matches(a,nx-1,ny-1) ||
                !row_matches(a,nx/3,ny/3) || !row_matches(a,nx/2,ny/2) ||
                !row_matches(a,nx-1,ny/2) || !row_matches(a,nx/2,ny-1))
                return false;
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    if (!row_matches(a,i,j)) return false;
            return true;
        }
    };
    const TensorFFT& fft;
    TensorFFT::Plan *forward = nullptr, *inverse = nullptr;
    Real *p = nullptr, *q = nullptr;
    vector<Real> e, g;
    int n = 0;
    double rtol = 0, atol = 0, t = 0, round_e = 0, round_g = 0;

    static double entry(const Matrix& a, int i, int j) {
        const auto first = a.col.begin()+a.row[i], last = a.col.begin()+a.row[i+1];
        const auto it = std::lower_bound(first,last,j);
        return it == last || *it != j ? 0. : a.val[it-a.col.begin()];
    }
    bool init(const Model& m, const Header& h) {
        n = h.n; t = h.t; rtol = h.rtol; atol = h.atol;
        p = static_cast<Real*>(fft.allocate(n*sizeof(Real)));
        q = static_cast<Real*>(fft.allocate(n*sizeof(Real)));
        if (!p || !q) return false;
        constexpr unsigned estimate = 1u << 6;
        if (m.y.n == 1) {
            forward = fft.plan1(m.x.n,p,p,m.x.kind(false),estimate);
            inverse = fft.plan1(m.x.n,p,p,m.x.kind(true),estimate);
        } else {
            forward = fft.plan2(m.y.n,m.x.n,p,p,m.y.kind(false),m.x.kind(false),estimate);
            inverse = fft.plan2(m.y.n,m.x.n,p,p,m.y.kind(true),m.x.kind(true),estimate);
        }
        if (!forward || !inverse) return false;
        const Real weight = 1 / (Real(m.x.norm())*m.y.norm());
        vector<Real> lx(m.x.n);
        for (int i = 0; i < m.x.n; ++i) lx[i] = m.x.lambda(i);
        e.resize(n); g.resize(n);
        Real de = 0, dg = 0;
        for (int j = 0; j < m.y.n; ++j) {
            const Real ly = m.y.lambda(j);
            for (int i = 0; i < m.x.n; ++i) {
                const Real x = t*((m.cxy*ly+m.cx)*lx[i] + m.cy*ly + m.c0);
                if (x < -1e-11L) return false;
                const Real dx = 2e-18L*t*(std::abs(m.c0) + std::abs(m.cx*lx[i]) +
                                    std::abs(m.cy*ly) + std::abs(m.cxy*ly*lx[i]));
                const int r = j*m.x.n+i;
                e[r] = x > 80 ? 0 : weight*std::exp(-x);
                g[r] = (weight*t)*(x > 80 ? 1/x : phi(x));
                if (x < 80) de = std::max(de,1.000001L*dx*e[r]/weight);
                const Real invx = 1/std::max(1.0L,x-dx);
                dg = std::max(dg,dx*std::min(0.500001L,invx*invx));
            }
        }
        // Includes the formation of the scalar multipliers near zero.
        const double fft_err = 8e-19*std::log2(2.*n);
        round_e = fft_err + double(de);
        round_g = fft_err + double(dg);
#ifdef FAM_PROFILE
        std::fprintf(stderr, "tensor %d %d %d %d\n", m.y.n,m.x.n,m.y.bound,m.x.bound);
#endif
        return true;
    }

public:
    explicit TensorGrid(const TensorFFT& f) : fft(f) {}
    ~TensorGrid() {
        if (forward) fft.destroy(forward);
        if (inverse) fft.destroy(inverse);
        if (p) fft.release(p);
        if (q) fft.release(q);
    }
    bool setup(const Matrix& a, const Header& h) {
        const double crossover = h.rtol > 1e-5 ? 12000 : h.rtol > 1e-8 ? 3800 : 1000;
        if (!fft.available() || h.t*a.norm < crossover || h.n < 50 || a.val.size() > 9.2*h.n)
            return false;
        vector<int> candidates;
        const auto push = [&](int nx) {
            if (nx >= 4 && nx <= a.n/4 && a.n % nx == 0 &&
                std::find(candidates.begin(),candidates.end(),nx) == candidates.end())
                candidates.push_back(nx);
        };
        for (int i : {a.n/2, a.n/3, 2*a.n/3, 3*a.n/5}) {
            for (int j = a.row[i]; j < a.row[i+1]; ++j) {
                if (a.val[j] == 0) continue;
                const int k = std::abs(a.col[j]-i);
                if (k > 2) { push(k); push(k-1); push(k+1); }
            }
        }
        if (a.val.size() < 3.1*h.n) candidates.push_back(a.n);
        for (const int nx : candidates) {
            const int ny = a.n/nx, i = (ny/2)*nx + nx/2;
            const double xx = entry(a,i,i+1), dd = entry(a,i,i);
            const double yy = ny == 1 ? 0. : entry(a,i,i+nx);
            const double xy = ny == 1 ? 0. : entry(a,i,i+nx+1);
            if (xx == 0 && xy == 0) continue;
            if (ny > 1 && yy == 0 && xy == 0) continue;
            // These four measurements have a short common dyadic expansion.
            // All the small integer combinations used by matches() are exact.
            const int unit = std::ilogb(a.norm)-55;
            bool exact = true;
            for (double v : {dd,xx,yy,xy}) {
                const Real g = std::ldexp(Real(v),-unit);
                exact &= g == std::trunc(g);
            }
            if (!exact) continue;
            const Real cx = -Real(xx)-2*Real(xy), cy = -Real(yy)-2*Real(xy);
            const Real c0 = Real(dd)-2*cx-2*cy-4*Real(xy);
            for (Bound y : {Neumann,Dirichlet,Periodic}) {
                for (Bound x : {Neumann,Dirichlet,Periodic}) {
                    Model m{{nx,x},{ny,y},c0,cx,cy,xy,{},{},{}};
                    if (m.matches(a)) return init(m,h);
                }
                if (ny == 1) break;
            }
        }
        return false;
    }
    bool solve(const double* b0, const double* b1, double* out) {
        const Magnitudes b(n,b0,b1);
        if (b.two0) {
            std::copy_n(b0,n,p);
            fft.exec(forward,p,p);
        }
        if (b.two1) {
            std::copy_n(b1,n,q);
            fft.exec(forward,q,q);
        }
        for (int i = 0; i < n; ++i)
            p[i] = (b.two0 ? e[i]*p[i] : 0) + (b.two1 ? g[i]*q[i] : 0);
        fft.exec(inverse,p,p);
        double yn = 0;
        for (int i = 0; i < n; ++i) {
            out[i] = double(p[i]);
            yn = std::max(yn,std::abs(out[i]));
        }
        return round_e*b.two0+round_g*t*b.two1 < 0.3*(atol+rtol*yn);
    }
};
