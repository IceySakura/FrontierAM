// Rational actions evaluated with diagonally preconditioned complex symmetric CG.
struct Rational {
    using C = std::complex<double>;
    using CL = std::complex<long double>;
    struct alignas(32) Z { __m256d a; };
    struct Pole { long double a, b, c, d; };
    struct Rule { int count; double exp_abs, phi_rel; const Pole* poles; };

    // The shorter rules use AAA in x/(x+scale), refined to partial fractions.
    static constexpr Pole small[] = {
        {3.58084965815242169901319L, 1.22833647488114154620098L, 2.33079900325594832741455L, 14.0899953217331375186632L},
        {2.66993054665281777298364L, 3.70341929307009495681927L,-3.14139889989730079904397L,-5.78771108054633180242374L},
        {0.682619609466723843295632L,6.24911888307941031160163L,0.850441012320995044011981L,0.805897544081788792027744L},
        {-2.93933846846358236593656L,9.00042885105963445615449L,-0.0398525285676215597312008L,-0.0274385296876827879053739L}
    };
    static constexpr Pole medium[] = {
        {4.24366650378230668027348L,1.21703392915710834374547L,5.55343342704451608613068L,26.7443967437474011660781L},
        {3.49997257029520793192396L,3.66541017932374739913252L,-8.2643240154788630666697L,-11.6775603309692234845152L},
        {1.93249513049877887589694L,6.16399629714447694168592L,3.03904435472688365983238L,1.88699939060619474030684L},
        {-0.670530116469504653224826L,8.77630431797157374461755L,-0.335597349708498382750304L,-0.0782153292557633984387895L},
        {-4.90482208148739901088903L,11.6592289830671178109856L,0.00744327041352264894072222L,0.00110741618370653189379822L}
    };
    static constexpr Pole high[] = {
        {5.17687232863412815715081L,1.21163343795805776272657L,16.1016713263853087328987L,66.9600584623775019897178L},
        {4.56951634484247136497741L,3.64218607496079137963872L,-26.0722492598333176076416L,-30.4016527270570774876982L},
        {3.30923201894278667659529L,6.09706032689030522891563L,11.8935976076085615487785L,4.92336150566738790025954L},
        {1.28506953806287772409529L,8.60300699547991422762649L,-2.03765098500091691311116L,0.10519680439784025462615L},
        {-1.74477984977154160676961L,11.2136915930635292499734L,0.116135139350964945429368L,-0.0510352072331748237630187L},
        {-6.42693545419699568211947L,14.0762680836973698751025L,-0.00150383194821731207587056L,0.000895175757711621030585393L}
    };
    // Pusa's CRAM16 partial fractions, written for exp(-x).
    static constexpr Pole strict[] = {
        {6.4161776990994341923e0L,1.1941223933701386874e0L,6.4500878025539646595e1L,2.2459440762652096056e2L},
        {5.9481522689511774808e0L,3.5874573620183222829e0L,-1.1339775178483930527e2L,-1.0194721704215856450e2L},
        {4.9931747377179963991e0L,5.9968817136039422260e0L,6.2518392463207918892e1L,1.1190391094283228480e1L},
        {3.5091036084149180974e0L,8.4361989858843750826e0L,-1.5059585270023467528e1L,5.7514052776421819979e0L},
        {1.4193758971856659786e0L,1.0925363484496722585e1L,1.4793007113557999718e0L,-1.7686588323782937906e0L},
        {-1.4139284624888862114e0L,1.3497725698892745389e1L,-4.1023136835410021273e-2L,1.5743466173455468191e-1L},
        {-5.2649713434426468895e0L,1.6220221473167927305e1L,-2.1151742182466030907e-4L,-4.3892969647380673918e-3L},
        {-1.0843917078696988026e1L,1.9277446167181652284e1L,5.0901521865224915650e-7L,2.4220017652852287970e-5L}
    };
    static constexpr Rule rules[] = {
        {4,9.1e-8,9.1e-8,small},
        {5,1.10e-9,1.95e-9,medium},
        {6,1.4e-11,1.45e-11,high},
        {8,8e-16,4e-14,strict}
    };

    Buffer<Z> x, p, r, q, invd, extra;
    Buffer<long double> wide_result;
    bool disabled = false;

    static inline __m256d pack(C x, C y) {
        return _mm256_setr_pd(x.real(),x.imag(),y.real(),y.imag());
    }
    static inline void unpack(__m256d x, C& a, C& b) {
        alignas(32) double d[4];
        _mm256_store_pd(d,x);
        a = C(d[0],d[1]); b = C(d[2],d[3]);
    }
    static inline __m256d mul(__m256d a, __m256d b) {
        auto br = _mm256_movedup_pd(b);
        auto bi = _mm256_permute_pd(b,15);
        return _mm256_fmaddsub_pd(a,br,_mm256_mul_pd(_mm256_permute_pd(a,5),bi));
    }
    static inline __m256d reciprocal(__m256d a) {
        const auto v = _mm256_mul_pd(a,a);
        auto d = _mm256_add_pd(v,_mm256_permute_pd(v,5));
        return _mm256_div_pd(_mm256_xor_pd(a,_mm256_setr_pd(0.,-0.,0.,-0.)),d);
    }
    static inline __m256d abs_lane(__m256d x) {
        return _mm256_andnot_pd(_mm256_set1_pd(-0.),x);
    }

    struct Matrix {
        int n, mode;
        const int *ptr, *col;
        const double *val, *diag;
        const long double *slack;
        double t, lower, lower_inf;
    };

    // A compensated product sum is used only when verifying a shifted solve.
    // The residual is formed against the received matrix, not its preconditioner.
    struct ProductSum {
        __m256d hi = _mm256_setzero_pd(), low=hi;
        inline void add(__m256d a, __m256d b) {
            const auto p=_mm256_mul_pd(a,b);
            const auto pe=_mm256_fmsub_pd(a,b,p);
            const auto s=_mm256_add_pd(hi,p);
            const auto v=_mm256_sub_pd(s,hi);
            const auto u=_mm256_add_pd(_mm256_sub_pd(hi,_mm256_sub_pd(s,v)),_mm256_sub_pd(p,v));
            low=_mm256_add_pd(low,_mm256_add_pd(u,pe));
            hi=s;
        }
    };

    struct WideVec {
        __m256d hi, low;
        static WideVec from(CL a, CL b) {
            C c(a), d(b);
            return {pack(c,d),pack(C(a-CL(c)),C(b-CL(d)))};
        }
    };

    template<bool wide>
    static __m256d residual_row(const Matrix& a, int i, const Z* x, const Z* xl,
                               WideVec z, WideVec we, WideVec wf, double b0, double b1) {
        const auto v=x[i].a;
        ProductSum ax, res;
        ax.add(v,_mm256_set1_pd(a.diag[i]));
        if constexpr (wide) ax.add(xl[i].a,_mm256_set1_pd(a.diag[i]));
        for (int k=a.ptr[i]; k<a.ptr[i+1]; ++k) {
            ax.add(x[a.col[k]].a,_mm256_set1_pd(a.val[k]));
            if constexpr (wide) ax.add(xl[a.col[k]].a,_mm256_set1_pd(a.val[k]));
        }
        const auto neg_t=_mm256_set1_pd(-a.t), sign=_mm256_set1_pd(-0.);
        res.add(we.hi,_mm256_set1_pd(b0));
        res.add(wf.hi,_mm256_set1_pd(b1));
        if constexpr (wide) {
            res.add(we.low,_mm256_set1_pd(b0));
            res.add(wf.low,_mm256_set1_pd(b1));
        }
        res.add(ax.hi,neg_t);
        res.add(ax.low,neg_t);
        const auto m=_mm256_setr_pd(0.,-0.,0.,-0.);
        res.add(v,_mm256_xor_pd(_mm256_movedup_pd(z.hi),sign));
        res.add(_mm256_permute_pd(v,5),_mm256_xor_pd(_mm256_permute_pd(z.hi,15),m));
        if constexpr (wide) {
            const auto vl=xl[i].a;
            res.add(v,_mm256_xor_pd(_mm256_movedup_pd(z.low),sign));
            res.add(_mm256_permute_pd(v,5),_mm256_xor_pd(_mm256_permute_pd(z.low,15),m));
            res.add(vl,_mm256_xor_pd(_mm256_movedup_pd(z.hi),sign));
            res.add(_mm256_permute_pd(vl,5),_mm256_xor_pd(_mm256_permute_pd(z.hi,15),m));
        }
        return _mm256_add_pd(res.hi,res.low);
    }

    template<int mode>
    static inline __m256d dotrow(const Matrix& a, int i, const Z* vec, __m256d center) {
        const double diag = mode ? double(a.slack[i]) : a.diag[i];
        auto s0 = _mm256_mul_pd(center,_mm256_set1_pd(diag));
        auto s1 = _mm256_setzero_pd();
        auto edge = [&](int k, __m256d sum) {
            auto v = vec[a.col[k]].a;
            auto w = _mm256_set1_pd(a.val[k]);
            if constexpr (mode==1) v = _mm256_sub_pd(v,center);
            if constexpr (mode==2)
                v = _mm256_add_pd(v,_mm256_xor_pd(center,_mm256_and_pd(w,_mm256_set1_pd(-0.))));
            return _mm256_fmadd_pd(w,v,sum);
        };
        const int end=a.ptr[i+1];
        int j=a.ptr[i];
        for (; j+1<end; j+=2) {
            s0 = edge(j,s0);
            s1 = edge(j+1,s1);
        }
        if (j<end) s0=edge(j,s0);
        return _mm256_add_pd(s0,s1);
    }

    template<int mode>
    __m256d matvec(const Matrix& a, const Z* __restrict state, Z* __restrict out,
                   __m256d shift, bool dot) const {
        const auto t = _mm256_set1_pd(a.t);
        auto s0=_mm256_setzero_pd(), s1=s0;
        for (int i=0; i<a.n; ++i) {
            auto own = state[i].a;
            auto y = _mm256_fmadd_pd(t,dotrow<mode>(a,i,state,own),mul(own,shift));
            out[i].a = y;
            if (dot) {
                if (i&1) s1 = _mm256_add_pd(s1,mul(y,own));
                else s0 = _mm256_add_pd(s0,mul(y,own));
            }
        }
        return _mm256_add_pd(s0,s1);
    }

    __m256d apply(const Matrix& a, const Z* state, Z* out, __m256d shift, bool dot) const {
        if (a.mode==1) return matvec<1>(a,state,out,shift,dot);
        if (a.mode==2) return matvec<2>(a,state,out,shift,dot);
        return matvec<0>(a,state,out,shift,dot);
    }

    struct Residual {
        __m256d euclidean = _mm256_setzero_pd(), maximum = euclidean;
        void add(__m256d r) {
            euclidean = _mm256_fmadd_pd(r,r,euclidean);
            maximum = _mm256_max_pd(maximum,abs_lane(r));
        }
        double test(int lane, C z, const Matrix& a) const {
            alignas(32) double l2[4], li[4];
            _mm256_store_pd(l2,euclidean);
            _mm256_store_pd(li,maximum);
            const double sigma = std::hypot(std::max(0.,a.t*a.lower+z.real()),z.imag());
            double norm = std::sqrt(l2[lane]+l2[lane+1])/sigma;
            const double gi = a.t*a.lower_inf+z.real();
            if (gi > 0) norm = std::min(norm,std::hypot(li[lane],li[lane+1])/gi);
            return norm;
        }
    };

    template<bool wide=false>
    bool pair(const Matrix& a, const double* b0, const double* b1, double* result,
              CL z_, CL v_, CL w0, CL w1, CL f0, CL f1, double target,
              int& remaining, double& verified, double& accum_mag) {
        const auto zero = _mm256_setzero_pd();
        const C z(z_),v(v_);
        const auto shift = WideVec::from(z_,v_);
        const auto we = WideVec::from(w0,w1), wf = WideVec::from(f0,f1);
        const auto z2 = shift.hi, w = we.hi, f = wf.hi;
        Z* __restrict X = x.get();
        Z* __restrict XL = extra.get();
        Z* __restrict P = p.get();
        Z* __restrict R = r.get();
        Z* __restrict Q = q.get();
        Z* __restrict M = invd.get();
        __m256d rho = zero;
        for (int i=0; i<a.n; ++i) {
            X[i].a = zero;
            if constexpr (wide) XL[i].a = zero;
            M[i].a = reciprocal(_mm256_add_pd(_mm256_set1_pd(a.t*a.diag[i]),z2));
            R[i].a = _mm256_fmadd_pd(w,_mm256_set1_pd(b0[i]),_mm256_mul_pd(f,_mm256_set1_pd(b1[i])));
            P[i].a = mul(M[i].a,R[i].a);
            rho = _mm256_add_pd(rho,mul(P[i].a,R[i].a));
        }
        bool active0=(w0!=CL{} || f0!=CL{}), active1=(w1!=CL{} || f1!=CL{});
        for (int j=0, restarts=0; j<140 && remaining>0; ++j) {
            --remaining;
            auto product = apply(a,P,Q,z2,true);
            C ro0,ro1,p0,p1;
            unpack(rho,ro0,ro1);
            unpack(product,p0,p1);
            const C al0 = active0 && p0!=C{} ? ro0/p0 : C{};
            const C al1 = active1 && p1!=C{} ? ro1/p1 : C{};
            if (!std::isfinite(al0.real()+al0.imag()+al1.real()+al1.imag())) return false;
            const auto alpha = pack(al0,al1);
            auto next = zero;
            Residual estimate;
            for (int i=0; i<a.n; ++i) {
                auto change=mul(alpha,P[i].a);
                auto old=X[i].a;
                auto sum=_mm256_add_pd(old,change);
                if constexpr (wide) {
                    auto v=_mm256_sub_pd(sum,old);
                    auto err=_mm256_add_pd(_mm256_sub_pd(old,_mm256_sub_pd(sum,v)),_mm256_sub_pd(change,v));
                    XL[i].a=_mm256_add_pd(XL[i].a,err);
                }
                X[i].a=sum;
                auto y = _mm256_sub_pd(R[i].a,mul(alpha,Q[i].a));
                estimate.add(y);
                auto u = mul(M[i].a,y);
                next = _mm256_add_pd(next,mul(u,y));
                R[i].a = y;
                Q[i].a = u;
            }
            active0 &= estimate.test(0,z,a)>target;
            active1 &= estimate.test(2,v,a)>target;
            if (!active0 && !active1) {
                Residual exact;
                rho = zero;
                for (int i=0; i<a.n; ++i) {
                    auto e = residual_row<wide>(a,i,X,XL,shift,we,wf,b0[i],b1[i]);
                    exact.add(e);
                    auto u = mul(M[i].a,e);
                    rho = _mm256_add_pd(rho,mul(u,e));
                    R[i].a = e;
                    P[i].a = u;
                }
                const double e0=exact.test(0,z,a), e1=exact.test(2,v,a);
                if (e0<=1.15*target && e1<=1.15*target) {
                    verified += e0+e1;
                    if constexpr (wide) {
                        __m256d magnitude=zero;
                        for (int i=0; i<a.n; ++i) {
                            alignas(32) double c[4], d[4];
                            _mm256_store_pd(c,X[i].a);
                            _mm256_store_pd(d,XL[i].a);
                            wide_result[i]+=(static_cast<long double>(c[0])+d[0])+
                                            (static_cast<long double>(c[2])+d[2]);
                            magnitude=_mm256_max_pd(magnitude,abs_lane(X[i].a));
                        }
                        alignas(32) double v[4];
                        _mm256_store_pd(v,magnitude);
                        accum_mag+=v[0]+v[2];
                    } else for (int i=0; i<a.n; ++i) {
                            const auto y = X[i].a;
                            result[i] += _mm_cvtsd_f64(_mm_add_sd(_mm256_castpd256_pd128(y),
                                                                  _mm256_extractf128_pd(y,1)));
                    }
                    return true;
                }
                if (++restarts>(wide ? 4 : 2)) return false;
                active0 = e0>target; active1 = e1>target;
                continue;
            }
            C ne0, ne1;
            unpack(next,ne0,ne1);
            const auto beta = pack(active0 && ro0!=C{} ? ne0/ro0 : C{},
                                   active1 && ro1!=C{} ? ne1/ro1 : C{});
            rho = next;
            for (int i=0; i<a.n; ++i) P[i].a = _mm256_add_pd(Q[i].a,mul(beta,P[i].a));
        }
        return false;
    }

    // Normalize the exponential to r(0)=1, so (1-r(x))/x has
    // the same poles and needs no division by the operator.
    bool solve(const Matrix& a, const Header& h, const Norms& b, double upper,
              const double* b0, const double* b1, int d, double tol, double* out) {
        if (disabled || !std::isfinite(tol) || b.i0+b.i1==0) return false;
        if (!x) {
            x=buffer<Z>(a.n); p=buffer<Z>(a.n); r=buffer<Z>(a.n);
            q=buffer<Z>(a.n); invd=buffer<Z>(a.n);
        }
        const bool wide=h.rtol<1e-8;
        double g=0;
        if (b.i1>0)
            for (int i=0; i<a.n; ++i)
                g=std::max(g,std::abs(b1[i])/(1+a.t*a.diag[i]));
        const double max_error=std::max(tol,0.15*(h.atol+h.rtol*g*h.t));
        const double root_n=std::sqrt(a.n);
        const Rule* rule=&rules[3];
#ifndef LARGE_RULE_ONLY
        for (int j=0; j<3; ++j) {
            const Rule& rr=rules[j];
            if ((!b.i1 || rr.phi_rel*root_n <= 0.34*h.rtol) &&
                (rr.exp_abs+(b.i1 ? rr.phi_rel : 0.))*b.s0 <= 0.8*max_error) {
                rule=&rr;
                break;
            }
        }
#endif
        CL s[9], w[9], f[9];
        long double constant=1.0L;
        for (int i=0; i<rule->count; ++i) {
            const Pole& q=rule->poles[i];
            s[i]=CL(q.a,q.b);
            w[i]=CL(2*q.c,2*q.d);
            const auto rho=w[i]/s[i];
            constant-=std::real(rho);
            f[i]=static_cast<long double>(h.t)*rho;
        }
        s[rule->count]=CL(1,1); w[rule->count]=f[rule->count]=CL();
        if (wide) {
            if (!extra) {
                extra=buffer<Z>(a.n);
                wide_result=buffer<long double>(a.n);
            }
            for (int i=0; i<a.n; ++i) wide_result[i]=constant*b0[i];
        } else for (int i=0; i<a.n; ++i) out[i]=double(constant)*b0[i];
        const double target=(wide ? 0.95 : 0.58)*max_error/rule->count;
        int budget=std::min(wide ? 390 : 340,std::max(28,d/3)), count=0;
        const int start=budget;
        double bound=0, accumulation=0;
        for (int i=0; i<rule->count; i+=2) {
            const bool ok=wide ?
                pair<true>(a,b0,b1,out,s[i],s[i+1],w[i],w[i+1],f[i],f[i+1],target,budget,bound,accumulation) :
                pair<false>(a,b0,b1,out,s[i],s[i+1],w[i],w[i+1],f[i],f[i+1],target,budget,bound,accumulation);
            if (!ok) break;
            count+=2;
        }
        if (wide)
            for (int i=0; i<a.n; ++i) out[i]=double(wide_result[i]);
        const double y=inf_norm(out,a.n), threshold=h.atol+h.rtol*y;
        const double round=wide ? 64*LDBL_EPSILON*(accumulation+b.i0+b.i1)+DBL_EPSILON*y :
                   DBL_EPSILON*1024*(b.i0+0.16*b.i1)*(1+std::sqrt(h.t*upper));
        const double relative=b.i1 ? rule->phi_rel : 0.;
        const double error=(bound+round+rule->exp_abs*b.s0+relative*(b.s0+root_n*y))/
                           (1-relative*root_n);
        const bool success=count>=rule->count && error<=0.51*threshold;
#ifdef PROFILE
        std::fprintf(stderr,"Rational %d/%d solved, %d/%d steps, error %.3g / %.3g, success %d\n",
                     std::min(count,rule->count),rule->count,start-budget,start,error,threshold,success);
#else
        (void)start;
#endif
        if (!success) disabled=true;
        return success;
    }
};
