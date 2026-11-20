#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>

// An implicit-Euler sequence is useful on the rapidly decaying subspace.
// The norm of its newest change bounds the modelling error without an
// assumed spectral gap. Every inaccurate or slow attempt falls back.
struct Steady {
    static constexpr double sigma = 16;
    static constexpr int steps = 16;

    struct Data {
        int n;
        const int* row;
        const uint32_t* col;
        const double* a;
        const double* diag;
        const double* row_sum;
        double t, rtol, atol, shift;
    };

    std::unique_ptr<double[]> work;

    void prepare(const Data& d) {
        work.reset(new double[static_cast<size_t>(d.n)*7]);
    }

    template<bool Diff, bool Dot>
    static double times_impl(const Data& d, const double* __restrict x,
                             double* __restrict y) {
        const auto* __restrict col = d.col;
        const auto* __restrict a = d.a;
        double rho[4] = {};
        for (int i = 0; i < d.n; ++i) {
            const double xi = x[i], wi = Diff ? xi : 0;
            double b0 = Diff ? d.row_sum[i]*wi : 0;
            double b1=0, b2=0, b3=0;
            const int end = d.row[i+1];
            int j = d.row[i];
            for (; j+3 < end; j += 4) {
                b0 = std::fma(a[j], x[col[j]]-wi,b0);
                b1 = std::fma(a[j+1], x[col[j+1]]-wi,b1);
                b2 = std::fma(a[j+2], x[col[j+2]]-wi,b2);
                b3 = std::fma(a[j+3], x[col[j+3]]-wi,b3);
            }
            if (j < end) b0 = std::fma(a[j], x[col[j]]-wi,b0);
            if (j+1 < end) b1 = std::fma(a[j+1], x[col[j+1]]-wi,b1);
            if (j+2 < end) b2 = std::fma(a[j+2], x[col[j+2]]-wi,b2);
            y[i] = std::fma(d.t, (b0+b1)+(b2+b3), d.shift*xi);
            if constexpr (Dot) rho[i&3] = std::fma(xi,y[i],rho[i&3]);
        }
        return (rho[0]+rho[1])+(rho[2]+rho[3]);
    }

    template<bool Dot>
    double times(const Data& d, const double* x, double* y) const {
        return d.row_sum ? times_impl<true,Dot>(d,x,y)
                         : times_impl<false,Dot>(d,x,y);
    }

    bool cg(const Data& d, const double* __restrict inv, const double* __restrict rhs,
            double* __restrict sol, double* __restrict r, double* __restrict p,
            double* __restrict z, double goal, int limit, int& iters,
            double& final_residual) const {
        const int n = d.n;
        const double rt = goal*goal;
        double rho = 0, r2 = 0;
        #pragma omp simd reduction(+:rho,r2)
        for (int i = 0; i < n; ++i) {
            const double x = rhs[i];
            sol[i] = 0;
            r[i] = x;
            p[i] = inv[i] * x;
            rho += x*p[i];
            r2 += x*x;
        }
        if (r2 <= rt) {
            final_residual = std::sqrt(r2);
            return true;
        }
        for (int step = 0; step < 96 && iters < limit; ++step) {
            const double den = times<true>(d,p,z);
            ++iters;
            if (!(den > 0)) return false;
            const double alpha = rho / den;
            double newrho = 0, r2 = 0;
            #pragma omp simd reduction(+:newrho,r2)
            for (int i = 0; i < n; ++i) {
                sol[i] = std::fma(alpha, p[i], sol[i]);
                const double ri = std::fma(-alpha, z[i], r[i]);
                r[i] = ri;
                z[i] = ri*inv[i];
                r2 += ri*ri;
                newrho += ri*z[i];
            }
            if (r2 <= .45 * rt) {
                times<false>(d,sol,z);
                ++iters;
                double sq = 0;
                #pragma omp simd reduction(+:sq)
                for (int i = 0; i < n; ++i) {
                    const double r = rhs[i]-z[i];
                    sq += r*r;
                }
                final_residual = std::sqrt(sq);
                return sq <= rt;
            }
            if (!(newrho > 0)) return false;
            const double beta = newrho/rho;
            #pragma omp simd
            for (int i = 0; i < n; ++i) p[i] = std::fma(beta,p[i],z[i]);
            rho = newrho;
        }
        return false;
    }

    bool propagate(const Data& a, const double* x, const double* q, double* __restrict y,
                   double unorm, double guess, int limit, int& iters, int& k_done) {
        if (!work) prepare(a);
        const int n = a.n;
        double* __restrict inv = work.get();
        double* __restrict rhs = inv+n;
        double* __restrict sol = rhs+n;
        double* __restrict r = sol+n;
        double* __restrict p = r+n;
        double* __restrict z = p+n;
        double* __restrict base = z+n;
        const double expected_tol = a.atol + a.rtol*guess;
        const double other_roundoff = 3e-14 * unorm;
        const double target = .008 * expected_tol;
        #pragma omp simd
        for (int i = 0; i < n; ++i) {
            rhs[i] = std::fma(a.t,q[i],sigma*x[i]);
            inv[i] = 1./std::fma(a.t,a.diag[i],sigma);
        }
        double res_total = 0, acc = 0, prior_norm = 0;
        for (int k = 1; k <= steps; ++k) {
            double residual = 0;
            if (!cg(a, inv, rhs, sol, r,p,z, target, limit,iters,residual)) return false;
            const auto exact = accurate_residual(a,rhs,sol,nullptr,true);
            residual = exact.norm+exact.error;
            if (residual > target) return false;
            k_done = k;
            res_total += residual;
            acc += res_total/sigma;
            double dnorm = 0, ynorm = 0;
            if (k == 1) {
                #pragma omp simd reduction(+:dnorm) reduction(max:ynorm)
                for (int i = 0; i < n; ++i) {
                    base[i] = sol[i];
                    const double delta = sigma*(sol[i]-x[i]);
                    rhs[i] = delta;
                    dnorm += delta*delta;
                    ynorm = std::max(ynorm,std::abs(sol[i]));
                }
            } else {
                #pragma omp simd reduction(+:dnorm) reduction(max:ynorm)
                for (int i = 0; i < n; ++i) {
                    base[i] += sol[i];
                    const double delta = sigma*sol[i];
                    rhs[i] = delta;
                    dnorm += delta*delta;
                    ynorm = std::max(ynorm,std::abs(base[i]));
                }
            }
            dnorm = std::sqrt(dnorm);
            const double slope = 1 - k/sigma;
            const double weight = std::min(1., .5*(slope+std::sqrt(slope*slope+2*k/(sigma*sigma))));
            const double err = weight*(dnorm+res_total) + acc + other_roundoff;
            const double tol = a.atol+a.rtol*ynorm;
            if (err <= .43 * tol) {
                std::memcpy(y,base,n*sizeof(double));
                return true;
            }
            if (k >= 2 && prior_norm > 0 && dnorm > res_total) {
                const double rate = std::min(1., (dnorm-res_total)/(prior_norm+res_total));
                if (dnorm * std::pow(rate, steps-k) > 4*tol) return false;
            }
            prior_norm = dnorm;
            if (k == 1 && iters > limit/3) return false;
        }
        return false;
    }

    struct Residual { double norm, error; };

    static Residual accurate_residual(const Data& a, const double* __restrict q,
                                      const double* __restrict v, double* __restrict out,
                                      bool shifted = false) {
        constexpr long double u = std::numeric_limits<long double>::epsilon()*.5L;
        constexpr long double ud = std::numeric_limits<double>::epsilon()*.5L;
        long double n2 = 0, er2 = 0;
        for (int i = 0; i < a.n; ++i) {
            long double p = a.row_sum ? static_cast<long double>(a.diag[i])*v[i] : 0;
            long double s = 0, h = std::abs(p), g = 0;
            const int end = a.row[i+1];
            int j = a.row[i];
            for (; j+1 < end; j += 2) {
                const long double x = static_cast<long double>(a.a[j])*v[a.col[j]];
                const long double y = static_cast<long double>(a.a[j+1])*v[a.col[j+1]];
                p += x; s += y;
                h += std::abs(x); g += std::abs(y);
            }
            if (j < end) {
                const long double x = static_cast<long double>(a.a[j])*v[a.col[j]];
                p += x; h += std::abs(x);
            }
            const long double shift = shifted ? static_cast<long double>(a.shift)*v[i] : 0;
            const long double scale = shifted ? a.t : 1;
            const long double r = q[i]-(scale*(p+s)+shift);
            if (out) out[i] = static_cast<double>(r);
            const long double round = ((end-a.row[i])*.5L+9)*u*scale*(h+g) +
                    4*u*std::abs(shift) +
                    (3*u+(out ? 2*ud : 0))*std::abs(r);
            n2 += r*r;
            er2 += round*round;
        }
        const double scale = shifted ? 1 : a.t;
        return {static_cast<double>(scale*std::sqrt(n2)),
                static_cast<double>(scale*std::sqrt(er2))};
    }

    void approximate(const Data& a, const double* q, double* v, double qnorm, int cap) {
        if (!work) prepare(a);
        const int n = a.n;
        double* __restrict inv = work.get();
        double* __restrict rhs = inv+n;
        double* __restrict r = rhs+n;
        double* __restrict p = r+n;
        double* __restrict z = p+n;
        #pragma omp simd
        for (int i = 0; i < n; ++i) {
            inv[i] = 1./std::fma(a.t,a.diag[i],a.shift);
            rhs[i] = a.t*q[i];
        }
        int iters = 0;
        double ignored = 0;
        cg(a,inv,rhs,v,r,p,z,.0003*qnorm,cap,iters,ignored);
    }

    bool stationary(const Data& a, const double* q, double* __restrict y,
                    double lower, double xnorm, double qnorm, double expect, int limit) {
        const double decay = std::exp(-lower);
        if (decay*(xnorm+qnorm/lower) > .16*expect) return false;
        if (qnorm == 0) {
            if (decay*xnorm > .35*a.atol) return false;
            std::fill_n(y,a.n,0.);
            return true;
        }
        if (!work) prepare(a);
        const int n = a.n;
        double* __restrict inv = work.get();
        double* __restrict rhs = inv+n;
        double* __restrict r = rhs+n;
        double* __restrict p = r+n;
        double* __restrict z = p+n;
        #pragma omp simd
        for (int i = 0; i < n; ++i) {
            inv[i] = 1/(a.t*a.diag[i]);
            rhs[i] = a.t*q[i];
        }
        double rnorm = 0;
        int iters = 0;
        if (!cg(a,inv,rhs,y,r,p,z,.055*expect*lower,limit,iters,rnorm)) return false;
        double ymax = 0, ynorm2 = 0;
        #pragma omp simd reduction(max:ymax) reduction(+:ynorm2)
        for (int i = 0; i < n; ++i) {
            ymax = std::max(ymax,std::abs(y[i]));
            ynorm2 += y[i]*y[i];
        }
        const double ynorm = std::sqrt(ynorm2);
        const auto rtrue = accurate_residual(a,q,y,nullptr);
        const double err = (rtrue.norm+rtrue.error)/lower+decay*(xnorm+ynorm);
        return err <= .38*(a.atol+a.rtol*ymax);
    }
};
