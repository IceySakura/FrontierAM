#pragma once

#include <memory>

class KrylovTrial {
    static constexpr int maxdim = 192;
    static constexpr std::array<double, 10> shifts = {1, 2, 4, 7, 11, 16, 23, 32, 44, 60};
    std::array<double,maxdim> alpha{}, beta{}, defect{};
    std::unique_ptr<double[]> v;
    int capacity = 0;
    std::vector<double> tmp, second, delta;
    std::vector<Real> lp;
    bool increment_stalled = false;

    static bool positive(int m, const double* d, const double* s, double end, int sign) {
        Real piv = sign*(Real(d[0])-end);
        if (!(piv > 0)) return false;
        for (int i = 1; i < m; ++i) {
            piv = sign*(Real(d[i])-end) - Real(s[i-1])*s[i-1]/piv;
            if (!(piv > 0)) return false;
        }
        return true;
    }

    template<class Poly>
    void project(const Poly& pol, int m, int pdeg, bool forcing, Real start,
                 double* __restrict out) const {
        const int n = pol.a.n;
        const auto& coef = forcing ? pol.g : pol.e;
        std::array<Real,maxdim> r{}, s{}, d{};
        Real* x0 = r.data();
        Real* x1 = s.data();
        const Real c = -2/pol.z;
        x0[0] = d[0] = coef[pdeg];
        for (int j = pdeg-1; j >= 0; --j) {
            const Real f = j ? c : c/2;
            for (int i = 0; i < m; ++i) {
                Real p = alpha[i]*x0[i];
                if (i > 0) p += beta[i-1]*x0[i-1];
                if (i+1 < m) p += beta[i]*x0[i+1];
                d[i] += f*p;
            }
            d[0] += coef[j];
            if (j) {
                for (int i = 0; i < m; ++i) x1[i] = x0[i]+d[i];
                std::swap(x0,x1);
            }
        }
        std::fill(out, out+n, 0.);
        const auto* __restrict p = v.get();
        for (int j = 0; j < m; ++j, p += n) {
            const double w = double(start*d[j]);
            for (int i = 0; i < n; ++i) out[i] += w*p[i];
        }
    }

    // The endpoint defect is an integral of a fixed-sign tridiagonal
    // exponential entry.  Replace its time weight by an exponential majorant
    // and extend the interval to infinity. The integral becomes an entry of
    // a positive shifted tridiagonal inverse.
    template<class Poly>
    double action(const Poly& pol, const double* rhs, double bnorm, bool forcing,
                  double budget, int& remaining, double* out, int& steps) {
        const auto& a = pol.a;
        const int n = a.n;
        const double t = pol.t;
        const double aa = double(pol.left);
        const double damp = forcing ? 1 : std::exp(-aa);
        const double amp = forcing ? t : 1;
        const double start_bound = forcing ? bnorm*t*double(phi(pol.left)) : bnorm*damp;
        if (start_bound <= budget*0.2) {
            std::fill(out, out+n, 0.);
            return start_bound;
        }
        if (remaining == 0 || !bnorm) return -1;
        {
            auto* p0 = v.get();
            const double scale = 1 / bnorm;
            for (int i = 0; i < n; ++i) p0[i] = rhs[i]*scale;
        }
        std::array<double,shifts.size()> pivot{}, product{}, multipliers{};
        product.fill(1.);
        for (size_t q = 0; q < shifts.size(); ++q) {
            const double sh = shifts[q];
            multipliers[q] = forcing ? std::exp(sh-1)/sh : std::exp(sh)*damp;
        }
        const double eta = t*a.norm, width = double(2*pol.z);
        const bool inf_contract = a.unscaled_lower >= (forcing ? 0. : a.lower) - 1e-8/t;
        double last_inf = 0, cur_inf = inf_contract ? max_abs(v.get(),n) : 1.;
        constexpr double u = 1.113e-16;
        int pdeg = int(pol.e.size())-1;
        const auto& tail = forcing ? pol.tail_g : pol.tail_e;
        while (pdeg > 0 && tail[pdeg-1]*bnorm < budget/64) --pdeg;
        const double round_init = start_bound*(8*u);
        for (int k = 0; remaining && k < maxdim; ++k) {
            const double bm = k ? beta[k-1] : 0.;
            const auto* __restrict vk = v.get() + size_t(n)*k;
            const auto* __restrict pm = k ? vk-n : vk;
            auto* __restrict z = tmp.data();
            double ak = 0.;
            for (int i = 0; i < n; ++i) {
                const double w = t*a.dot(i,vk) - bm*pm[i];
                z[i] = w;
                ak += w*vk[i];
            }
            alpha[k] = ak;
            auto r0 = _mm256_setzero_pd(), r1 = r0, mx = r0;
            const auto av = _mm256_set1_pd(ak);
            const auto sign = _mm256_set1_pd(-0.);
            int i = 0;
            for (; i + 7 < n; i += 8) {
                auto x = _mm256_fnmadd_pd(av, _mm256_loadu_pd(vk+i), _mm256_loadu_pd(z+i));
                auto y = _mm256_fnmadd_pd(av, _mm256_loadu_pd(vk+i+4), _mm256_loadu_pd(z+i+4));
                _mm256_storeu_pd(z+i, x);
                _mm256_storeu_pd(z+i+4, y);
                r0 = _mm256_fmadd_pd(x,x,r0);
                r1 = _mm256_fmadd_pd(y,y,r1);
                mx = _mm256_max_pd(mx, _mm256_andnot_pd(sign, x));
                mx = _mm256_max_pd(mx, _mm256_andnot_pd(sign, y));
            }
            double rn = sum_lanes(_mm256_add_pd(r0,r1)), inf = max_lanes(mx);
            for (; i < n; ++i) {
                z[i] -= ak*vk[i];
                rn += z[i]*z[i];
                inf = std::max(inf, std::abs(z[i]));
            }
            const double bb = std::sqrt(rn);
            beta[k] = bb;
            --remaining;
            ++steps;
            const double vn = inf_contract ? cur_inf : 1;
            const double pmn = inf_contract ? last_inf : (k != 0);
            const double nz = inf_contract ? inf : bb;
            defect[k] = u * ((7 + std::min(53,(a.widest+3)/4))*eta*vn +
                         5*(bm*pmn + std::abs(ak)*vn + nz));
            double bound = std::numeric_limits<double>::infinity();
            for (size_t j = 0; j < shifts.size(); ++j) {
                if (!std::isfinite(product[j])) continue;
                const double sh = shifts[j] + (forcing ? aa : 0);
                const double piv = ak + sh - (k ? bm*(bm/pivot[j]) : 0);
                if (!(piv > 1e-9)) {
                    product[j] = std::numeric_limits<double>::infinity();
                    continue;
                }
                pivot[j] = piv;
                product[j] *= bb/piv;
                bound = std::min(bound, product[j] * multipliers[j]);
            }
            if (inf_contract && bb > 0) bound *= (1.000001 * inf/bb);
            bound *= 1.001 * amp * bnorm;
            if (bound <= budget*0.8) {
                // Bound the numerically perturbed Lanczos relation too.
                // A shift of one works for every coordinate of the integral.
                std::array<Real,maxdim> p{}, l{}, iv{};
                const Real sh = 1 + (forcing ? aa : 0);
                p[0] = alpha[0]+sh;
                iv[0] = 1;
                if (!(p[0] > 0)) return -1;
                for (int q = 1; q <= k; ++q) {
                    l[q] = beta[q-1]/p[q-1];
                    p[q] = alpha[q]+sh-beta[q-1]*l[q];
                    if (!(p[q] > 0)) return -1;
                    iv[q] = iv[q-1]*l[q];
                }
                Real err = 0, prev = 0;
                for (int q = k; q >= 0; --q) {
                    prev = iv[q]/p[q] + (q == k ? 0 : l[q+1]*prev);
                    err += prev*defect[q];
                }
                bound += double(1.002L * bnorm*amp * err *
                                (forcing ? 1 : 2.71828182845904524L*damp));
                bound += round_init + double(tail[pdeg]*bnorm);
                bound += (steps+8)*u*start_bound * std::sqrt(double(k+1));
                if (bound <= budget) {
                    const double margin = std::max(2e-13, width*2e-13);
                    if (!positive(k+1,alpha.data(),beta.data(),-margin,1) ||
                        !positive(k+1,alpha.data(),beta.data(),width+margin,-1)) return -1;
                    project(pol,k+1,pdeg,forcing,bnorm,out);
                    return bound;
                }
                if (bound > 3*budget && bb < std::max(1.,eta)*1e-12) return -1;
            }
            if (!remaining || !(bb > std::max(1.,eta)*1e-15)) break;
            auto* __restrict next = v.get() + size_t(n)*(k+1);
            const double inv = 1/bb;
            for (int q = 0; q < n; ++q) next[q] = z[q]*inv;
            last_inf = cur_inf;
            cur_inf = 1.000000001*inf*inv;
        }
        return -1;
    }

    static double max_abs(const double* x, int n) {
        auto s = _mm256_setzero_pd();
        const auto sign = _mm256_set1_pd(-0.);
        int i = 0;
        for (; i+3 < n; i += 4)
            s = _mm256_max_pd(s, _mm256_andnot_pd(sign, _mm256_loadu_pd(x+i)));
        double r = max_lanes(s);
        for (; i < n; ++i) r = std::max(r, std::abs(x[i]));
        return r;
    }
public:
    template<class Poly, class Norms>
    bool increment(const Poly& pol, const Norms& b, double predicted, int cap,
                   const double* b0, const double* b1, double* out) {
        if (increment_stalled) return false;
        const auto& a = pol.a;
        const int n = a.n;
        cap = std::min({n,cap,maxdim});
        if (cap < 1) return false;
        const bool inf = a.unscaled_lower*pol.t >= -1e-8;
        delta.resize(n); lp.resize(n);
        std::copy_n(b0,n,lp.data());
        Real dnorm = 0, round = 0;
        for (int i = 0; i < n; ++i) {
            Real v = a.dot(i,lp.data())+Real(pol.lo)*lp[i];
            if (pol.lo) v += Real(a.shift_error[i])*lp[i];
            const Real d = Real(b1[i])-v;
            delta[i] = double(d);
            dnorm += Real(delta[i])*delta[i];
            const Real err = std::abs(d-Real(delta[i])) +
                1.12e-19L*(8+std::min(53,(a.row[i+1]-a.row[i]+3)/4)) *
                ((a.diag[i]+a.abs_off[i])*b.inf0 + std::abs(Real(b1[i])));
            if (inf) round = std::max(round,err);
            else round += err*err;
        }
        round = 1.000001L*pol.t*(inf ? round : std::sqrt(round));
        const double guess = pol.atol+pol.rtol*predicted;
        if (round >= 0.1*guess) return false;
        if (capacity < cap) {
            v.reset(new double[size_t(n)*cap]);
            capacity = cap;
        }
        tmp.resize(n);
        int rem = cap, steps = 0;
        const double err = action(pol,delta.data(),double(1.0000000001L*std::sqrt(dnorm)),
                                  true,0.11*guess,rem,out,steps);
        if (rem == 0) increment_stalled = true;
        if (err < 0.) return false;
        double yn = 0.;
        for (int i = 0; i < n; ++i) {
            out[i] += b0[i];
            yn = std::max(yn,std::abs(out[i]));
        }
        if (round+err+5e-16*b.inf0 > 0.29*(pol.atol+pol.rtol*yn)) return false;
#ifdef FAM_PROFILE
        std::fprintf(stderr,"krylov increment %d %.4g\n",steps,
                     double((round+err)/(pol.atol+pol.rtol*yn)));
#endif
        return true;
    }

    template<class Poly, class Norms>
    bool solve(const Poly& pol, const Norms& b, double predicted, int cap,
               const double* b0, const double* b1, double* out) {
        const int n = pol.a.n;
        cap = std::min({cap, n, maxdim});
        if (cap < 1) return false;
        if (capacity < cap) {
            v.reset(new double[size_t(n)*cap]);
            capacity = cap;
        }
        tmp.resize(n);
        second.resize(n);
        int rem = cap, steps = 0;
        const double budget = 0.16*(pol.atol+pol.rtol*predicted);
        const double err1 = action(pol, b1, b.two1, true, budget, rem, out, steps);
        if (err1 < 0) return false;
        const double err0 = action(pol, b0, b.two0, false, budget, rem, second.data(), steps);
        if (err0 < 0) return false;
        double yn = 0.;
        for (int i = 0; i < n; ++i) {
            out[i] += second[i];
            yn = std::max(yn,std::abs(out[i]));
        }
        if (err0+err1 > 0.36*(pol.atol+pol.rtol*yn)) return false;
#ifdef FAM_PROFILE
        std::fprintf(stderr, "krylov %d %.3g\n", steps, (err0+err1)/(pol.atol+pol.rtol*yn));
#endif
        return true;
    }
};
