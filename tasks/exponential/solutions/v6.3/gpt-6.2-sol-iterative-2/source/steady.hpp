// Steady-state shortcut. Both the discarded transient and the residual
// of the linear solve are bounded using the positive Gershgorin lower bound.
struct Steady {
    Buffer<double> p,r,v,inv;
    bool disabled=false;

    template<bool certify>
    static double row(int i, const double* x, const int* ptr, const int* col,
                      const double* a, const double* diag, double rhs) {
        if constexpr (certify) {
            AccurateSum sum;
            sum.product(diag[i],x[i]);
            for (int j=ptr[i]; j<ptr[i+1]; ++j) sum.product(a[j],x[col[j]]);
            return double(static_cast<long double>(rhs)-sum.value());
        } else {
            double s0=diag[i]*x[i],s1=0;
            int j=ptr[i],end=ptr[i+1];
            for (;j+1<end;j+=2) {
                s0=std::fma(a[j],x[col[j]],s0);
                s1=std::fma(a[j+1],x[col[j+1]],s1);
            }
            if (j<end) s0=std::fma(a[j],x[col[j]],s0);
            return s0+s1;
        }
    }

    bool solve(const Header& h, const Norms& b, double lower, double upper_bound,
               const int* ptr, const int* col, const double* a, const double* diag,
               const double* b0, const double* b1, int degree, int maxrow, double* out) {
        const int n=h.n;
        const double eta=lower*h.t;
        if (eta<6) return false;
        const double decay=std::exp(-eta);
        if (!b.i1) {
            if (decay*b.i0 >= 0.68*h.atol) return false;
            std::fill_n(out,n,0.);
            return true;
        }
        if (disabled || degree<30) return false;
        const double integral=-std::expm1(-eta)/lower;
        double magnitude=0;
        for (int i=0; i<n; ++i) {
            out[i]=b1[i]/diag[i];
            magnitude=std::max(magnitude,std::abs(out[i]));
        }
        if (decay*(b.i0+magnitude)>0.6*(h.atol+h.rtol*magnitude)) return false;
        if (!v) {
            p=buffer<double>(n);
            r=buffer<double>(n);
            v=buffer<double>(n);
            inv=buffer<double>(n);
            for (int i=0; i<n; ++i) inv[i]=1/diag[i];
        }
        auto reload=[&](bool precise) {
            double rho=0, error=0, displacement=0;
            for (int i=0; i<n; ++i) {
                double q = precise ? row<true>(i,out,ptr,col,a,diag,b1[i]) :
                                     b1[i]-row<false>(i,out,ptr,col,a,diag,0.);
                r[i]=q;
                p[i]=q*inv[i];
                rho+=q*p[i];
                error=std::max(error,std::abs(q));
                displacement=std::max(displacement,std::abs(out[i]-b0[i]));
            }
            return std::array<double,3>{rho,error,displacement};
        };
        const double eps=DBL_EPSILON*(maxrow+4.);
        const int budget=std::min(64,std::max(8,degree/5));
        auto state=reload(false);
        double rho=state[0], residual=state[1];
        for (int k=0, refreshed=0;k<budget;++k) {
            magnitude=inf_norm(out,n);
            const double tol=h.atol+h.rtol*magnitude;
            if (residual*integral<=0.32*tol) {
                state=reload(true);
                const double round=8*eps*eps*upper_bound*magnitude*integral;
                if (state[1]*integral+decay*state[2]+round<=0.48*tol) return true;
                if (++refreshed>2 || decay*state[2]>0.48*tol) break;
                rho=state[0];
            }
            double den=0;
            for (int i=0; i<n; ++i) {
                v[i]=row<false>(i,p.get(),ptr,col,a,diag,0.);
                den+=v[i]*p[i];
            }
            if (!(den>0) || !(rho>0)) break;
            double alpha=rho/den, nr=0;
            residual=0;
            for (int i=0;i<n;++i) {
                out[i]=std::fma(alpha,p[i],out[i]);
                const double x=std::fma(-alpha,v[i],r[i]);
                r[i]=x;
                v[i]=inv[i]*x;
                nr+=x*v[i];
                residual=std::max(residual,std::abs(x));
            }
            const double beta=nr/rho;
            for (int i=0;i<n;++i) p[i]=std::fma(beta,p[i],v[i]);
            rho=nr;
        }
        disabled=true;
        return false;
    }
};
