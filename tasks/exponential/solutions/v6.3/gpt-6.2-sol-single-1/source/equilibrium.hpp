#pragma once

inline double sum_lanes(__m256d v) {
    const __m128d s = _mm_add_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd(v, 1));
    return _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)));
}

// The trial solution is accepted only after evaluating the residual against
// the original matrix and bounding both terms of the variation-of-constants
// error: exp(-t L)(b0-x) and t phi1(-t L)(b1-L x).
class Equilibrium {
    std::vector<double> invdiag, r, p, ap;
    std::vector<Real> xx;
public:
    template<class MatrixType, class Norms>
    bool solve(const MatrixType& a, const Norms& b, double t, double rtol, double atol,
               int cap, const double* b0, const double* b1, double* x) {
        const int n = a.n;
        const double lo = a.lower, left = t*lo;
        const double decay = std::exp(-left);
        const double integral = -std::expm1(-left) / lo;
        if (r.empty()) {
            invdiag.resize(n);
            r.resize(n);
            p.resize(n);
            ap.resize(n);
            for (int i = 0; i < n; ++i) invdiag[i] = 1 / a.diag[i];
        }
        std::fill(x, x+n, 0.);
        double rho = 0;
        for (int i = 0; i < n; ++i) {
            r[i] = b1[i];
            p[i] = invdiag[i] * r[i];
            rho += p[i] * r[i];
        }
        if (rho == 0) return decay*b.two0 <= 0.08*atol;
        const double yn_lower = b.two1 / (a.upper * std::sqrt(double(n)));
        double replace_at = 0.004*(atol+rtol*yn_lower) / integral;
        const double nu = 2.23e-16, ul = 1.085e-19;
        for (int k = 0; k < cap; ++k) {
            double pq = 0;
            for (int i = 0; i < n; ++i) {
                const double u = a.dot(i, p.data()) + lo*p[i];
                ap[i] = u;
                pq += p[i] * u;
            }
            const double alpha = rho / pq;
            if (!(alpha > 0 && std::isfinite(alpha))) return false;
            auto d = _mm256_setzero_pd(), e = d;
            const auto u = _mm256_set1_pd(alpha);
            const int n4 = n & ~3;
            for (int i = 0; i < n4; i += 4) {
                const auto ri = _mm256_fnmadd_pd(u, _mm256_loadu_pd(ap.data()+i),
                                                _mm256_loadu_pd(r.data()+i));
                _mm256_storeu_pd(r.data()+i, ri);
                _mm256_storeu_pd(x+i, _mm256_fmadd_pd(u, _mm256_loadu_pd(p.data()+i),
                                                      _mm256_loadu_pd(x+i)));
                const auto zz = _mm256_mul_pd(_mm256_loadu_pd(invdiag.data()+i), ri);
                _mm256_storeu_pd(ap.data()+i, zz);
                d = _mm256_fmadd_pd(ri, zz, d);
                e = _mm256_fmadd_pd(ri, ri, e);
            }
            double nr = sum_lanes(d), rr = sum_lanes(e);
            for (int i = n4; i < n; ++i) {
                r[i] -= alpha * ap[i];
                x[i] += alpha * p[i];
                ap[i] = invdiag[i] * r[i];
                nr += r[i] * ap[i];
                rr += r[i] * r[i];
            }
            bool restarted = false;
            if (rr < replace_at*replace_at || (k == cap-1 && rr < 1e4*replace_at*replace_at)) {
                xx.resize(n);
                Real xn = 0, dn = 0, err = 0, rbound2 = 0;
                for (int i = 0; i < n; ++i) {
                    xx[i] = x[i];
                    xn = std::max(xn, std::abs(xx[i]));
                    const Real v = b0[i]-xx[i];
                    dn += v*v;
                }
                nr = 0;
                for (int i = 0; i < n; ++i) {
                    const Real u = a.dot(i, xx.data()) + (lo+Real(a.shift_error[i]))*xx[i];
                    const Real v = Real(b1[i]) - u;
                    r[i] = double(v);
                    ap[i] = invdiag[i]*r[i];
                    nr += r[i]*ap[i];
                    const Real rb = std::abs(v) + ul*(12+(a.row[i+1]-a.row[i])/4)
                                  * ((a.diag[i]+a.abs_off[i])*xn + std::abs(Real(b1[i])));
                    err = std::max(err, rb);
                    rbound2 += rb*rb;
                }
                // The lower Gershgorin endpoint yields an infinity-norm
                // contraction only if it also holds without similarity.
                const bool inf_contracts = a.unscaled_lower >= lo;
                Real diff = inf_contracts ? std::min(std::sqrt(dn), xn+b.inf0) : std::sqrt(dn);
                Real rb = inf_contracts ? err : std::sqrt(rbound2);
                const double goal = atol + rtol*double(xn);
                if (1.001*(decay*diff + integral*rb) <= 0.35*goal) {
#ifdef FAM_PROFILE
                    std::fprintf(stderr, "equilibrium %d %.3g\n", k+1,
                                 double(decay*diff + integral*rb)/goal);
#endif
                    return true;
                }
                if (decay*diff >= 0.30*goal) return false;
                replace_at = 0.04*goal / integral;
                if (inf_contracts) replace_at *= 2;
                restarted = true;
            }
            const double beta = restarted ? 0 : nr/rho;
            if (!(nr > 0 && std::isfinite(beta))) return false;
            for (int i = 0; i < n; ++i) p[i] = ap[i] + beta*p[i];
            rho = nr;
            // Prevent a roundoff-level residual from repeatedly restarting
            // an unsuccessful trial.
            if (!restarted && k > 10 && rho < nu*nu*1e-5) return false;
        }
        return false;
    }
};
