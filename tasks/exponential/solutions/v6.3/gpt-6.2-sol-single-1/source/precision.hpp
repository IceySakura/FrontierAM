#pragma once

// Error-free transforms below deliberately keep the named rounded products
// separate. Explicit FMA operations recover their rounding residuals.
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")

struct DD4 {
    __m256d hi, lo;
    static DD4 zero() {
        const auto z = _mm256_setzero_pd();
        return {z,z};
    }
    static DD4 constant(Real r) {
        const double h = double(r);
        return {_mm256_set1_pd(h), _mm256_set1_pd(double(r-h))};
    }
    static DD4 load(const double* h, const double* l) {
        return {_mm256_loadu_pd(h), _mm256_loadu_pd(l)};
    }
    void store(double* h, double* l) const {
        _mm256_storeu_pd(h,hi);
        _mm256_storeu_pd(l,lo);
    }
    static __m256d add_error(__m256d a, __m256d b, __m256d s) {
        const auto z = _mm256_sub_pd(s,a);
        return _mm256_add_pd(_mm256_sub_pd(a,_mm256_sub_pd(s,z)),_mm256_sub_pd(b,z));
    }
    static DD4 normalized(__m256d a, __m256d b) {
        const auto h = _mm256_add_pd(a,b);
        return {h,add_error(a,b,h)};
    }
    DD4 add(DD4 b) const {
        const auto z = _mm256_add_pd(hi,b.hi);
        const auto v = _mm256_add_pd(_mm256_add_pd(lo,b.lo),add_error(hi,b.hi,z));
        return normalized(z,v);
    }
    DD4 times(__m256d b) const {
        const auto p = _mm256_mul_pd(hi,b);
        return {p,_mm256_fmadd_pd(lo,b,_mm256_fmsub_pd(hi,b,p))};
    }
    DD4 times(DD4 b) const {
        const auto p = _mm256_mul_pd(hi,b.hi);
        return {p,_mm256_fmadd_pd(hi,b.lo,_mm256_fmadd_pd(lo,b.hi,_mm256_fmsub_pd(hi,b.hi,p)))};
    }
    void dot_add(__m256d a, __m256d x, __m256d l) {
        const auto p = _mm256_mul_pd(a,x);
        const auto s = _mm256_add_pd(hi,p);
        const auto e = _mm256_add_pd(add_error(hi,p,s),_mm256_fmsub_pd(a,x,p));
        lo = _mm256_add_pd(lo,_mm256_fmadd_pd(a,l,e));
        hi = s;
    }
};

class Compensated {
    std::vector<double> u0, u1, result, sh;
    std::vector<double> xh, xl, yh, yl, dh, dl;

    template<class MatrixType>
    static DD4 dot_csr4(const MatrixType& a, int row, const int* order,
                        const double* h, const double* l) {
        double rh[4] = {0,0,0,0}, rl[4] = {0,0,0,0};
        for (int j = 0; j < 4; ++j) {
            const int r = order ? order[row+j] : row+j;
            if (r >= a.n) continue;
            Real v = 0, s = 0;
            for (int k = a.row[r]; k < a.row[r+1]; ++k) {
                const int c = a.col[k];
                const Real p = Real(a.val[k])*(Real(h[c])+l[c]);
                const Real y = p-s, q = v+y;
                s = (q-v)-y;
                v = q;
            }
            rh[j] = double(v);
            rl[j] = double(v-rh[j]);
        }
        return DD4::load(rh,rl);
    }
    template<int K, size_t... J>
    static DD4 dot_dia(const double* h, const double* l, const double* ac, const int* off,
                       std::index_sequence<J...>) {
        DD4 sum[2] = {DD4::zero(), DD4::zero()};
        ((sum[J%2].dot_add(_mm256_loadu_pd(ac+4*J), _mm256_loadu_pd(h+off[J]),
                           _mm256_loadu_pd(l+off[J]))), ...);
        return sum[0].add(sum[1]);
    }
    template<bool Small, bool Val32>
    static DD4 dot_sell(const SlicedRows& s, int i, const double* h, const double* l) {
        const auto g = s.groups[i/4];
        if (g.width < 0) {
            const auto& pb = s.scalar_rows[g.start];
            double rh[4], rl[4];
            for (int k = 0; k < 4; ++k) {
                Real v = 0, e = 0;
                for (int j = pb[k]; j < pb[k+1]; ++j) {
                    const int c = s.scalar_cols[j];
                    const Real prod = Real(s.scalar_vals[j])*(Real(h[c])+l[c]);
                    const Real y = prod-e, q = v+y;
                    e = (q-v)-y;
                    v = q;
                }
                rh[k] = double(v);
                rl[k] = double(v-rh[k]);
            }
            return DD4::load(rh,rl);
        }
        auto update = [&](DD4& v, int j) {
            __m256d c;
            if constexpr (Val32) c = _mm256_cvtps_pd(_mm_loadu_ps(s.vals32.data()+j));
            else c = _mm256_loadu_pd(s.vals.data()+j);
            v.dot_add(c,s.gather<Small>(h,j),s.gather<Small>(l,j));
        };
        DD4 v0 = DD4::zero(), v1 = DD4::zero();
        const int end = g.start+4*g.width;
        int j = g.start;
        for (; j+4 < end; j += 8) {
            update(v0,j);
            update(v1,j+4);
        }
        if (j < end) update(v0,j);
        return v0.add(v1);
    }

    template<class Poly, class Dot, class FastDot>
    double evaluate(const Poly& p, int m, int split, const int* order, int n4, int pad,
                    Dot&& dot, FastDot&& fastdot, const double* b0, const double* b1,
                    double* out) {
        const int n = p.a.n, size = n4+2*pad;
        u0.resize(n4); u1.resize(n4); sh.resize(n4); result.resize(n4);
        xh.resize(size); xl.resize(size); yh.resize(size); yl.resize(size);
        dh.resize(n4); dl.resize(n4);
        const bool shifted = p.lo != 0;
        for (int k = 0; k < n4; ++k) {
            const int i = order ? order[k] : k;
            u0[k] = i < n ? b0[i] : 0;
            u1[k] = i < n ? b1[i] : 0;
            sh[k] = shifted && i < n ? p.a.shift_error[i] : 0;
        }
        auto *q0 = xh.data()+pad, *q0l = xl.data()+pad;
        auto *q1 = yh.data()+pad, *q1l = yl.data()+pad;
        auto* __restrict s0 = dh.data();
        auto* __restrict s0l = dl.data();
        const auto em = DD4::constant(p.e[m]), gm = DD4::constant(p.g[m]);
        for (int i = 0; i < n4; i += 4) {
            const DD4 w = em.times(_mm256_loadu_pd(u0.data()+i)).
                         add(gm.times(_mm256_loadu_pd(u1.data()+i)));
            w.store(q0+i,q0l+i);
            w.store(s0+i,s0l+i);
        }
        if (split < m) {
            std::fill(s0l,s0l+n4,0.);
            std::fill(q0l,q0l+n4,0.);
            std::fill(q1l,q1l+n4,0.);
        }
        const Real scale = -4/(Real(p.hi)-p.lo);
        for (int k = m-1; k >= 0; --k) {
            if (k > split) {
                const auto ev = _mm256_set1_pd(double(p.e[k]));
                const auto gv = _mm256_set1_pd(double(p.g[k]));
                const auto hv = _mm256_set1_pd(double(scale));
                for (int i = 0; i < n4; i += 4) {
                    const auto s = fastdot(i,q0);
                    const auto f = _mm256_fmadd_pd(ev,_mm256_loadu_pd(u0.data()+i),
                                                  _mm256_mul_pd(gv,_mm256_loadu_pd(u1.data()+i)));
                    const auto d = _mm256_fmadd_pd(hv,s,_mm256_add_pd(f,_mm256_loadu_pd(s0+i)));
                    _mm256_storeu_pd(s0+i,d);
                    _mm256_storeu_pd(q1+i,_mm256_add_pd(_mm256_loadu_pd(q0+i),d));
                }
                std::swap(q0,q1);
                std::swap(q0l,q1l);
                continue;
            }
            const auto em = DD4::constant(p.e[k]), gm = DD4::constant(p.g[k]);
            const auto h = DD4::constant(k ? scale : scale/2);
            for (int i = 0; i < n4; i += 4) {
                auto s = dot(i,q0,q0l);
                if (shifted) s.lo = _mm256_fmadd_pd(_mm256_loadu_pd(sh.data()+i),
                                                    _mm256_loadu_pd(q0+i),s.lo);
                const DD4 src = em.times(_mm256_loadu_pd(u0.data()+i)).
                                add(gm.times(_mm256_loadu_pd(u1.data()+i)));
                const DD4 d = DD4::load(s0+i,s0l+i).add(src).add(h.times(s));
                if (k) {
                    d.store(s0+i,s0l+i);
                    DD4::load(q0+i,q0l+i).add(d).store(q1+i,q1l+i);
                } else {
                    _mm256_storeu_pd(result.data()+i,_mm256_add_pd(d.hi,d.lo));
                }
            }
            std::swap(q0,q1);
            std::swap(q0l,q1l);
        }
        double norm = 0;
        for (int j = 0; j < n4; ++j) {
            const int i = order ? order[j] : j;
            if (i >= n) continue;
            const double u = result[j];
            out[i] = u;
            norm = std::max(norm,std::abs(u));
        }
        return norm;
    }

    template<int K, class Poly>
    double via_dia(const Poly& p, int m, int split,
                   const double* b0, const double* b1, double* out) {
        const auto& a = p.a;
        const auto& d = p.dia;
        const double* val = d.vals.data();
        const int* off = d.offsets.data();
        auto dot = [&](int i, const double* h, const double* l) {
            if (i+3 < a.n && d.good[i/4]) return dot_dia<K>(h+i,l+i,val+size_t(i)*K,
                                                           off,std::make_index_sequence<K>());
            return dot_csr4(a,i,nullptr,h,l);
        };
        auto fast = [&](int i, const double* h) {
            if (i+3 < a.n && d.good[i/4]) return d.template dot<K>(h+i,val+size_t(i)*K,
                    off,std::make_index_sequence<K>());
            double q[4] = {0,0,0,0};
            for (int j=0; j<4 && i+j<a.n; ++j) q[j] = a.dot(i+j,h);
            return _mm256_loadu_pd(q);
        };
        return evaluate(p,m,split,nullptr,(a.n+3)&~3,d.pad,dot,fast,b0,b1,out);
    }
public:
    template<class Poly>
    bool solve(const Poly& p, int m, int split, const double* b0, const double* b1,
               double* out) {
        if (m == 0) return false;
        if (p.use_dia) {
            #define FAM_CMP_DIA(i) case i: via_dia<i>(p,m,split,b0,b1,out); return true;
            switch (p.dia.width) {
                FAM_CMP_DIA(2) FAM_CMP_DIA(3) FAM_CMP_DIA(4) FAM_CMP_DIA(5)
                FAM_CMP_DIA(6) FAM_CMP_DIA(7) FAM_CMP_DIA(8) FAM_CMP_DIA(9)
                FAM_CMP_DIA(10) FAM_CMP_DIA(11) FAM_CMP_DIA(12)
            }
            #undef FAM_CMP_DIA
        }
        if (!p.use_sell) return false;
        const auto& s = p.sell;
        auto run = [&](auto small, auto val32) {
            auto dot = [&](int i, const double* h, const double* l) {
                return dot_sell<decltype(small)::value,decltype(val32)::value>(s,i,h,l);
            };
            auto fast = [&](int i, const double* h) {
                const auto w = s.groups[i/4];
                if (w.width >= 0)
                    return s.template dot<decltype(small)::value,decltype(val32)::value>(h,w);
                const auto& pb = s.scalar_rows[w.start];
                double q[4];
                for (int r=0; r<4; ++r)
                    q[r] = spdot(pb[r],pb[r+1],s.scalar_cols.data(),s.scalar_vals.data(),h);
                return _mm256_loadu_pd(q);
            };
            evaluate(p,m,split,s.order.data(),s.n4,0,dot,fast,b0,b1,out);
        };
        if (s.small_indices) {
            if (s.exact_float) run(std::true_type{},std::true_type{});
            else run(std::true_type{},std::false_type{});
        } else {
            if (s.exact_float) run(std::false_type{},std::true_type{});
            else run(std::false_type{},std::false_type{});
        }
        return true;
    }
};

#pragma GCC pop_options
