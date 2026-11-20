#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <memory>

struct Deflation {
    struct Data {
        int n;
        const int* row;
        const uint32_t* col;
        const double* a;
        const double* diag;
        bool omit_diagonal;
    };
    std::unique_ptr<double[]> vec, packed;
    double lambda = 0, replacement = 0, gamma = 0, top = 0, loss = 0;
    double exp_correction = 0, phi_correction = 0;

    template<class T>
    static T row_dot(const Data& d, int i, const double* w) {
        T p = d.omit_diagonal ? static_cast<T>(d.diag[i])*w[i] : 0;
        T q = 0, r = 0, s = 0;
        const auto add = [](T a, T b, T c) {
            if constexpr (sizeof(T) == 8) return std::fma(a,b,c);
            else return a*b+c;
        };
        const int end = d.row[i+1];
        int j = d.row[i];
        for (; j+3 < end; j += 4) {
            p = add(d.a[j],w[d.col[j]],p);
            q = add(d.a[j+1],w[d.col[j+1]],q);
            r = add(d.a[j+2],w[d.col[j+2]],r);
            s = add(d.a[j+3],w[d.col[j+3]],s);
        }
        if (j < end) p = add(d.a[j],w[d.col[j]],p);
        if (j+1 < end) q = add(d.a[j+1],w[d.col[j+1]],q);
        if (j+2 < end) r = add(d.a[j+2],w[d.col[j+2]],r);
        return (p+q)+(r+s);
    }

    // A dimension n-1 principal submatrix bounds the second eigenvalue.
    static double complement(const Data& d, int root, double next_biggest) {
        auto w = std::unique_ptr<double[]>(new double[d.n]);
        auto v = std::unique_ptr<double[]>(new double[d.n]);
        for (int i = 0; i < d.n; ++i) w[i] = d.diag[i] + .045*next_biggest;
        double best = 0;
        for (int pass = 0; pass < 3; ++pass) {
            double b = 0, unweighted = 0;
            for (int i = 0; i < d.n; ++i) {
                if (i == root) continue;
                double s = d.omit_diagonal ? d.diag[i]*w[i] : 0;
                double q = d.omit_diagonal ? d.diag[i] : 0;
                for (int j = d.row[i]; j < d.row[i+1]; ++j) {
                    const int k = static_cast<int>(d.col[j]);
                    if (k == root) continue;
                    const double a = std::abs(d.a[j]);
                    s = std::fma(a,w[k],s);
                    if (pass == 0) q += a;
                }
                v[i] = s;
                if (pass == 0) unweighted = std::max(unweighted,q);
                b = std::max(b,s/w[i]);
            }
            if (pass == 0) best = unweighted;
            const double old = best;
            best = std::min(best,b);
            if (!(best > 0) || (pass > 0 && best >= .995*old)) break;
            const double inv = 1/b;
            for (int i = 0; i < d.n; ++i)
                if (i != root) v[i] = std::fma(inv,v[i],.012*next_biggest);
            std::swap(w,v);
        }
        return best*1.00000000005;
    }

    void prepare(const Data& d, double full_top, double time, double rtol,
                 const int* order, int n_pad) {
        if (d.n < 2000 || time*full_top < 2400) return;
        double largest = 0, second = 0;
        int root = 0;
        for (int i = 0; i < d.n; ++i) {
            const double x = d.diag[i];
            if (x > largest) {
                second = largest;
                largest = x;
                root = i;
            } else {
                second = std::max(second,x);
            }
        }
        if (!(second > 0 && second < .57*largest)) return;
        const double low = complement(d,root,second);
        if (!(low > 0 && low < .58*full_top && low < .61*largest)) return;

        auto w = std::make_unique<double[]>(d.n);
        auto v = std::unique_ptr<double[]>(new double[d.n]);
        w[root] = 1;
        const double target = (rtol <= 1e-10 ? 1e-11 : (rtol <= 1e-7 ? 2e-9 : 5e-7)) / time;
        const int limit = low < .39*largest ? 38 : 62;
        // Solve the eigenvector equation with the root component fixed.
        for (int k = 0; k < limit; ++k) {
            const double shift = row_dot<double>(d,root,w.get());
            if (!(shift >= .95*largest)) return;
            const double inv = 1/shift;
            double dsq = 0;
            for (int i = 0; i < d.n; ++i) {
                if (i == root) {
                    v[i] = 1;
                    continue;
                }
                v[i] = inv*row_dot<double>(d,i,w.get());
                const double diff = v[i]-w[i];
                dsq = std::fma(diff,diff,dsq);
            }
            std::swap(w,v);
            if (std::sqrt(dsq)*shift <= .25*target) break;
        }

        auto prod = std::unique_ptr<long double[]>(new long double[d.n]);
        long double norm = 0, norm_lo = 0, rq = 0, rq_lo = 0, arithmetic2 = 0;
        constexpr long double unit = std::numeric_limits<long double>::epsilon()*.5L;
        const auto accumulate = [](long double x, long double& s, long double& lo) {
            const long double t = s+x;
            lo += std::abs(s) >= std::abs(x) ? (s-t)+x : (x-t)+s;
            s = t;
        };
        for (int i = 0; i < d.n; ++i) {
            const long double u = w[i];
            const long double di = d.diag[i]*u;
            long double off = 0, off_lo = 0, magnitude = 0;
            for (int j = d.row[i]; j < d.row[i+1]; ++j) {
                const int k = static_cast<int>(d.col[j]);
                if (k == i) continue;
                const long double p = static_cast<long double>(d.a[j])*w[k];
                magnitude += std::abs(p);
                accumulate(p,off,off_lo);
            }
            prod[i] = di+(off+off_lo);
            accumulate(u*u,norm,norm_lo);
            accumulate(u*prod[i],rq,rq_lo);
            const long double e = 7*unit*(magnitude+std::abs(di));
            arithmetic2 += e*e;
        }
        norm += norm_lo;
        const long double rayleigh = (rq+rq_lo)/norm;
        lambda = static_cast<double>(rayleigh);
        long double r2 = 0;
        for (int i = 0; i < d.n; ++i) {
            const long double r = prod[i]-rayleigh*w[i];
            r2 += r*r;
        }
        const double residual = static_cast<double>(
            (std::sqrt(r2)+std::sqrt(arithmetic2))/std::sqrt(norm) +
            8*unit*std::abs(rayleigh));
        const double delta = residual*time;
        if (!(lambda > low + 4*residual) || !(delta < 2e-5)) return;
        const long double anchor = std::max(.97L*low,30.L/time);
        gamma = static_cast<double>((rayleigh-anchor)/norm);
        const long double adjusted = rayleigh - gamma*norm;
        replacement = static_cast<double>(adjusted);
        const double s = replacement*time, l = lambda*time;
        // The shifted operator's negative endpoint is bounded in magnitude
        // by residual^2/replacement.
        top = std::max(low + 2*residual,replacement+residual)*1.002;
        // Compare with the two block-diagonal operators. The first-order
        // heat terms are bounded by delta/l and delta/s; the two remainders
        // together are at most exp(delta)*delta^2.
        loss = delta*(1/l+1/s)+std::exp(delta)*delta*delta +
               1e-14/std::min(s,l);
        exp_correction = static_cast<double>((std::exp(-l)-std::exp(-s))/norm);
        phi_correction = static_cast<double>(time*(-std::expm1(-l)/l
                                                  +std::expm1(-s)/s)/norm);
        vec = std::move(w);
        if (n_pad) {
            packed.reset(new double[n_pad]);
            for (int i = 0; i < n_pad; ++i) packed[i] = order[i] < d.n ? vec[order[i]] : 0;
        }
    }

    static double dot(int n, const double* __restrict x, const double* __restrict y) {
        __m256d a = _mm256_setzero_pd(), b = a, c = a, d = a;
        int i = 0;
        for (; i+15 < n; i += 16) {
            a = _mm256_fmadd_pd(_mm256_loadu_pd(x+i),_mm256_loadu_pd(y+i),a);
            b = _mm256_fmadd_pd(_mm256_loadu_pd(x+i+4),_mm256_loadu_pd(y+i+4),b);
            c = _mm256_fmadd_pd(_mm256_loadu_pd(x+i+8),_mm256_loadu_pd(y+i+8),c);
            d = _mm256_fmadd_pd(_mm256_loadu_pd(x+i+12),_mm256_loadu_pd(y+i+12),d);
        }
        alignas(32) double partial[4];
        _mm256_store_pd(partial,_mm256_add_pd(_mm256_add_pd(a,b),_mm256_add_pd(c,d)));
        double sum = (partial[0]+partial[1])+(partial[2]+partial[3]);
        for (; i < n; ++i) sum = std::fma(x[i],y[i],sum);
        return sum;
    }

    void correct(int n, const double* __restrict x, const double* __restrict q,
                 double* __restrict y) const {
        const double r = exp_correction*dot(n,x,vec.get())+
                         phi_correction*dot(n,q,vec.get());
        for (int i = 0; i < n; ++i) y[i] = std::fma(r,vec[i],y[i]);
    }
};
