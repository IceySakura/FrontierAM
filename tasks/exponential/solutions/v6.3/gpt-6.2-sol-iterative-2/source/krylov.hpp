// Short, unrestarted Lanczos with a residual-integral stopping test.
struct Krylov {
    Buffer<double> basis;
    size_t space = 0;
    int rejected[3]{};
    int completed[3]{};
    enum { Initial, Source, Combined };

    static double dot(const double* x, const double* y, int n) {
        __m256d s = _mm256_setzero_pd(), t = s;
        int i=0;
        for (; i+8<=n; i+=8) {
            s = _mm256_fmadd_pd(_mm256_load_pd(x+i),_mm256_load_pd(y+i),s);
            t = _mm256_fmadd_pd(_mm256_load_pd(x+i+4),_mm256_load_pd(y+i+4),t);
        }
        if (i<n) s = _mm256_fmadd_pd(_mm256_load_pd(x+i),_mm256_load_pd(y+i),s);
        s = _mm256_add_pd(s,t);
        const __m128d p = _mm_add_pd(_mm256_castpd256_pd128(s),_mm256_extractf128_pd(s,1));
        return _mm_cvtsd_f64(_mm_add_sd(p,_mm_unpackhi_pd(p,p)));
    }

    // For a positive-subdiagonal tridiagonal T, the corner of exp(-sT)
    // has a constant sign. Its absolute Laplace transform equals the
    // absolute corner of (T+shift*I)^(-1), obtained from scalar LDL.
    static double integral_bound(const double* diag, const double* off, int m, bool exp) {
        double best = std::numeric_limits<double>::infinity();
        constexpr double z[] = {0.5, 1.5, 3., 6., 12., 20., 32., 48., 72., 108., 160.};
        for (double shift : z) {
            if (shift > m+1) break;
            double d = diag[0]+shift;
            double p = 1/d;
            for (int i=1; i<m; ++i) {
                d = std::fma(-off[i-1]/d,off[i-1],diag[i]+shift);
                p *= off[i-1]/d;
            }
            if (!(d>0) || !(p>=0)) continue;
            // max_{0<=s<=1} [(1-s)*exp(shift*s)] for the phi_1 case.
            double w = exp ? std::exp(shift) : shift <= 1 ? 1 : std::exp(shift-1)/shift;
            best = std::min(best,p*w);
        }
        return best*1.01;
    }

    template<class R>
    static void reduced(const Coefficients& poly, const double* a, const double* b,
                        int m, int degree, double begin, double width, double t,
                        double factor, int kind, double* coef) {
        auto coeff = [&](int k) {
            if (kind == Initial) return poly.c[k].e;
            if (kind == Source) return poly.c[k].f;
            return poly.c[k].f/t;
        };
        if (m == 1) {
            coef[0] = factor*(kind == Initial ? std::exp(-a[0]) : (kind == Source ? t : 1)*phi(a[0]));
            return;
        }
        if (!degree) {
            std::fill_n(coef,m,0.);
            coef[0] = factor*coeff(0);
            return;
        }
        std::vector<R> u(m+2), v(m+2), diff(m+2), da(m+2), db(m+2);
        for (int i=0; i<m; ++i) da[i+1] = R(a[i])-R(begin);
        for (int i=0; i<m-1; ++i) db[i+1] = b[i];
        u[1] = diff[1] = coeff(degree);
        const R scale = -R(4)/R(width);
        for (int k=degree-1; k>=1; --k) {
            diff[1] += coeff(k);
            for (int j=1; j<=m; ++j) {
                const R z = da[j]*u[j]+(db[j]*u[j+1]+db[j-1]*u[j-1]);
                diff[j] += scale*z;
                v[j] = u[j]+diff[j];
            }
            u.swap(v);
        }
        diff[1] += coeff(0);
        for (int j=1; j<=m; ++j) {
            const R z = da[j]*u[j]+(db[j]*u[j+1]+db[j-1]*u[j-1]);
            coef[j-1] = double((diff[j]+(scale/2)*z)*factor);
        }
    }

    static void lift(double* __restrict out, const double* __restrict init,
                     const double* __restrict coef, const double* __restrict basis,
                     int n, int stride, int count) {
        const double* first = basis;
        if (init) {
            for (int i=0; i<n; ++i) out[i] = std::fma(coef[0],first[i],init[i]);
        } else {
            for (int i=0; i<n; ++i) out[i] = coef[0]*first[i];
        }
        int j=1;
        for (; j+3<count; j+=4) {
            const double* u = basis+size_t(j)*stride;
            const double* v = u+stride; const double* w = v+stride; const double* x = w+stride;
            const double la=coef[j], lb=coef[j+1], lc=coef[j+2], ld=coef[j+3];
            for (int i=0; i<n; ++i) {
                const double z = std::fma(la,u[i],lb*v[i]);
                const double q = std::fma(lc,w[i],ld*x[i]);
                out[i] += z+q;
            }
        }
        for (; j<count; ++j) {
            const double* u = basis+size_t(j)*stride;
            const double la = coef[j];
            for (int i=0; i<n; ++i) out[i] = std::fma(la,u[i],out[i]);
        }
    }

    template<class Op>
    bool apply(const Header& h, const Norms& b, const Packed& p, double a,
               double hi, double infty_growth, const Coefficients& poly,
               int degree, double tol, Op matvec, double* scratch, double* y,
               bool skewed) {
        const int kind = b.i1 == 0 ? Initial : b.i0 == 0 ? Source : Combined;
        if (rejected[kind] > completed[kind] ||
            (kind == Combined && h.rtol < 1e-8 && b.i1 < 4*b.i0))
            return false;
        const int size = p.size, stride = size+4;
        constexpr int cap = 384;
        const int limit = skewed ? std::min(cap,degree/10) : 4;
        if (limit < 4) return false;
        const size_t wanted = size_t(stride)*(limit+1);
        if (space < wanted) {
            basis = buffer<double>(wanted);
            space = wanted;
        }
        const double* b0 = p.source0;
        const double* b1 = p.source1;
        const double* u = kind == Initial ? b0 : b1;
        double len = kind == Initial ? b.s0 : b.s1/h.t;
        if (kind == Combined) {
            matvec(scratch, b0);
            for (int i=0; i<size; ++i) scratch[i] = std::fma(h.t,b1[i],-scratch[i]);
            u = scratch;
            len = std::sqrt(dot(u,u,size));
        }
        double* q = basis.get();
        for (int i=0; i<size; ++i) q[i] = len == 0 ? 0 : u[i]/len;
        std::fill_n(q+size,4,0.);
        double d[cap], e[cap], coef[cap];
        const double begin = h.t*a, width = h.t*(hi-a);
        const double amp = len*(kind == Source ? h.t : 1);
        for (int j=0; j<limit; ++j) {
            const int m = j+1;
            double* z = q+stride;
            matvec(z,q);
            const double alpha = dot(z,q,size);
            d[j] = alpha;
            if (j) {
                const double* prev = q-stride;
                const double beta = e[j-1];
                for (int i=0; i<size; ++i)
                    z[i] = std::fma(-alpha,q[i],std::fma(-beta,prev[i],z[i]));
            } else {
                for (int i=0; i<size; ++i) z[i] = std::fma(-alpha,q[i],z[i]);
            }
            const double beta = std::sqrt(dot(z,z,size));
            e[j] = beta;
            const bool check = m<=4 || m==limit || (m<32 ? m%4==0 : m%8==0);
            if (check) {
                const double norm = infty_growth ? std::min(beta,infty_growth*inf_norm(z,size)) : beta;
                const double error = norm*amp*integral_bound(d,e,m,kind==Initial);
                if (error <= tol || beta == 0) {
                    Norms v{};
                    if (kind == Initial) v.s0 = len;
                    else v.s1 = amp;
                    const int k = poly.degree(v,0.18*tol);
                    if (h.rtol < 1e-8)
                        reduced<long double>(poly,d,e,m,k,begin,width,h.t,len,kind,coef);
                    else
                        reduced<double>(poly,d,e,m,k,begin,width,h.t,len,kind,coef);
                    lift(scratch, kind==Combined ? b0 : nullptr, coef,basis.get(),size,stride,m);
                    const double magnitude = inf_norm(scratch,size);
                    const double threshold = h.atol+h.rtol*magnitude;
                    const bool unsafe = kind==Combined &&
                       (h.rtol < 1e-8 &&
                         8*DBL_EPSILON*h.t*hi*b.i0 > 0.12*threshold);
                    if (error <= 0.35*threshold && !unsafe &&
                       v.s0*poly.tail[k].e+v.s1*poly.tail[k].f <= 0.20*threshold) {
#ifdef PROFILE
                        std::fprintf(stderr,"Lanczos %d/%d error %.3g threshold %.3g type %d\n",
                                     m,degree,error,threshold,kind);
#endif
                        for (int i=0; i<h.n; ++i) y[p.order[i]] = scratch[i];
                        ++completed[kind];
                        return true;
                    }
                    if (unsafe) break;
                    tol = std::min(tol,0.22*threshold);
                }
            }
            if (beta == 0 || !std::isfinite(beta)) break;
            const double inv = 1/beta;
            for (int i=0; i<size; ++i) z[i] *= inv;
            q = z;
        }
#ifdef PROFILE
        std::fprintf(stderr,"Lanczos fallback (limit %d, d %d, type %d)\n",limit,degree,kind);
#endif
        ++rejected[kind];
        return false;
    }
};
