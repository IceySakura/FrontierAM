struct Tensor {
    enum class Boundary { Free, Fixed, Periodic };
    struct Line {
        int col[3], count, diag;
    };
    struct Form {
        int nx, ny;
        long double diag, wx, wy, xy;
        Boundary bx, by;
    };

    fftw_plan fwd = nullptr, inv = nullptr;
    Buffer<double> e, f, u, v;
    double perturbation = 0, contraction_inf = 0;
    int size = 0;

    ~Tensor() { clear(); }
    void clear() {
        if (fwd) fftw_destroy_plan(fwd);
        if (inv) fftw_destroy_plan(inv);
        fwd = inv = nullptr;
    }

    static Line line(int i, int n, Boundary b) {
        if (n==1) return {{0,0,0},1,0};
        if (i==0) {
            if (b==Boundary::Periodic) return {{0,1,n-1},3,2};
            return {{0,1,0},2,b==Boundary::Free ? 1 : 2};
        }
        if (i==n-1) {
            if (b==Boundary::Periodic) return {{0,n-2,n-1},3,2};
            return {{n-2,n-1,0},2,b==Boundary::Free ? 1 : 2};
        }
        return {{i-1,i,i+1},3,2};
    }

    static double entry(int row, int j, const int* ptr, const int* col, const double* val) {
        const int* b = col+ptr[row], *e = col+ptr[row+1];
        const int* q = std::lower_bound(b,e,j);
        return q==e || *q!=j ? 0 : val[q-col];
    }

    static Form guess(int nx, int ny, const int* ptr, const int* col, const double* val) {
        const int ix = nx/2, iy = ny/2, i = ix*ny+iy;
        const long double wx = nx > 1 ? entry(i,i+ny,ptr,col,val) : 0;
        const long double wy = ny > 1 ? entry(i,i+1,ptr,col,val) : 0;
        const long double xy = nx > 1 && ny > 1 ? entry(i,i+ny+1,ptr,col,val) : 0;
        return {nx,ny,entry(i,i,ptr,col,val),wx,wy,xy,Boundary::Free,Boundary::Free};
    }

    static long double row_check(const Form& m, int ix, int iy, const int* ptr,
                                 const int* col, const double* val, long double& low) {
        const int row = ix*m.ny+iy;
        int k = ptr[row], end = ptr[row+1];
        const auto x = line(ix,m.nx,m.bx), y = line(iy,m.ny,m.by);
        const long double base_x = m.nx>1 ? 2 : 0;
        const long double base_y = m.ny>1 ? 2 : 0;
        const long double dx = base_x-x.diag, dy = base_y-y.diag;
        const long double dd = m.diag+m.wx*dx+m.wy*dy+m.xy*dx*dy;
        const long double ax = m.wx+m.xy*dy, ay = m.wy+m.xy*dx;
        long double err = 0, radius = 0;
        for (int a=0; a<x.count; ++a) {
            for (int b=0; b<y.count; ++b) {
                const int j = x.col[a]*m.ny+y.col[b];
                while (k<end && col[k]<j) err += std::abs(val[k++]);
                const long double v = x.col[a]==ix ?
                    (y.col[b]==iy ? dd : ay) : (y.col[b]==iy ? ax : m.xy);
                if (j!=row) radius += std::abs(v);
                const long double exact = k<end && col[k]==j ? val[k++] : 0;
                err += std::abs(v-exact);
            }
        }
        while (k<end) err += std::abs(val[k++]);
        low = std::min(low,dd-radius);
        return err;
    }

    static bool check(Form& m, const int* ptr, const int* col, const double* val,
                      double bound, double& err, double& lower_inf) {
        const int nx=m.nx, ny=m.ny;
        const long double max_error = 7e-15L*bound;
        const int x[] = {0, nx-1, nx/2, nx/3, 1 < nx ? 1 : 0};
        const int y[] = {0, ny-1, ny/2, ny/3, 1 < ny ? 1 : 0};
        for (int i : x) for (int j : y) {
            long double low = std::numeric_limits<long double>::infinity();
            if (row_check(m,i,j,ptr,col,val,low)>max_error) return false;
        }
        long double row_error=0, lo=std::numeric_limits<long double>::infinity();
        for (int i=0; i<nx; ++i) for (int j=0; j<ny; ++j) {
            row_error = std::max(row_error,row_check(m,i,j,ptr,col,val,lo));
            if (row_error>max_error) return false;
        }
        err = double(row_error)*(1+8*DBL_EPSILON);
        lower_inf = double(lo)-8*DBL_EPSILON*bound;
        return true;
    }

    static double fft_scale(Boundary b, int n) {
        if (b==Boundary::Periodic) return n;
        return n==1 ? 2 : b==Boundary::Fixed ? 2*(n+1) : 2*n;
    }

    static int fft_kind(Boundary b, bool inverse, int n) {
        if (n>1 && b==Boundary::Fixed) return 7; // RODFT00
        if (n>1 && b==Boundary::Periodic) return inverse ? 1 : 0; // HC2R / R2HC
        return inverse ? 4 : 5; // REDFT01 / REDFT10
    }

    static std::vector<long double> axis_eigen(int n, Boundary b) {
        std::vector<long double> l(n);
        if (n==1) return l;
        constexpr long double pi = 3.141592653589793238462643383279502884L;
        for (int i=0; i<n; ++i) {
            long double x = b==Boundary::Free ? i*pi/(2*n) :
                            b==Boundary::Fixed ? (i+1)*pi/(2*(n+1)) : std::min(i,n-i)*pi/n;
            const long double s = std::sin(x);
            l[i] = 4*s*s;
        }
        return l;
    }

    bool make(const Form& m, double t, double row_error, double min_inf) {
        const int nx=m.nx, ny=m.ny;
        size=nx*ny;
        u=buffer<double>(size); v=buffer<double>(size);
        e=buffer<double>(size); f=buffer<double>(size);
        constexpr unsigned estimate = 64;
        fwd=fftw_plan_r2r_2d(nx,ny,u.get(),u.get(),fft_kind(m.bx,false,nx),fft_kind(m.by,false,ny),estimate);
        inv=fftw_plan_r2r_2d(nx,ny,u.get(),u.get(),fft_kind(m.bx,true,nx),fft_kind(m.by,true,ny),estimate);
        if (!fwd || !inv) {
            clear();
            return false;
        }
        auto a=axis_eigen(nx,m.bx), b=axis_eigen(ny,m.by);
        const long double dx = nx>1 ? 2 : 0, dy = ny>1 ? 2 : 0;
        const double factor = 1/(fft_scale(m.bx,nx)*fft_scale(m.by,ny));
        for (int i=0; i<nx; ++i)
            for (int j=0; j<ny; ++j) {
                const long double p = dx-a[i], q = dy-b[j];
                const long double x = t*(m.diag+m.wx*p+m.wy*q+m.xy*p*q);
                u[i*ny+j] = double(x);
            }
        scalar_functions(u.get(),e.get(),f.get(),size,factor,factor*t);
        perturbation = t*row_error;
        contraction_inf = min_inf*t;
#ifdef PROFILE
        std::fprintf(stderr,"Tensor %d x %d, %d %d, error %.3g\n",nx,ny,int(m.bx),int(m.by),perturbation);
#endif
        return true;
    }

    bool prepare(int n, const int* ptr, const int* col, const double* val,
                 double t, double bound, double lower) {
        int strides[24], count=0;
        strides[count++]=n;
        const int row = n/2;
        for (int k=ptr[row]; k<ptr[row+1]; ++k) {
            int d = std::abs(col[k]-row);
            for (int s=d-1; s<=d+1 && count<24; ++s) {
                if (s<3 || s>n/3 || n%s) continue;
                if (std::find(strides,strides+count,s)==strides+count) strides[count++]=s;
            }
        }
        const auto free = Boundary::Free, fixed = Boundary::Fixed, periodic = Boundary::Periodic;
        for (int i=0; i<count; ++i) {
            auto m=guess(n/strides[i],strides[i],ptr,col,val);
            if (i==0 && m.wy==0) continue;
            if (i && m.wx==0 && m.xy==0) continue;
            for (auto bx : {free,fixed,periodic}) {
                if (m.nx==1 && bx!=free) continue;
                m.bx=bx;
                for (auto by : {free,fixed,periodic}) {
                    m.by=by;
                    double err, low;
                    if (check(m,ptr,col,val,bound,err,low)) return make(m,t,err,std::min(low,lower));
                }
            }
        }
        return false;
    }

    const double* solve(const Header& h, const Norms& b, const double* a, const double* c) {
        if (b.i0) {
            std::memcpy(u.get(),a,size*sizeof(double));
            fftw_execute_r2r(fwd,u.get(),u.get());
        }
        if (b.i1) {
            std::memcpy(v.get(),c,size*sizeof(double));
            fftw_execute_r2r(fwd,v.get(),v.get());
        }
        double* x = b.i0 ? u.get() : v.get();
        if (b.i0 && b.i1)
            for (int i=0; i<size; ++i) x[i] = x[i]*e[i]+v[i]*f[i];
        else if (b.i0)
            for (int i=0; i<size; ++i) x[i] *= e[i];
        else if (b.i1)
            for (int i=0; i<size; ++i) x[i] *= f[i];
        else {
            std::fill_n(x,size,0.);
            return x;
        }
        fftw_execute_r2r(inv,x,x);
        const double two_norm = (b.s0+0.5*b.s1)*(1+2*perturbation);
        const double max_norm = (b.i0+0.5*b.i1)*std::exp(std::min(600.,std::max(0.,-contraction_inf)));
        const double threshold = h.atol+h.rtol*inf_norm(x,size);
        const double estimate = perturbation*std::min(two_norm,max_norm) + 5e-15*(b.i0+b.i1);
        return estimate <= 0.23*threshold ? x : nullptr;
    }
};
