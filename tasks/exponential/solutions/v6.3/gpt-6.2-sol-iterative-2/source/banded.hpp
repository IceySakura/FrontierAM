// A fixed-offset kernel for sparse stencils, with optional two-step tiling.
struct Banded {
    static constexpr int max_degree=12, tile=4096;
    struct alignas(32) D4 { double v[4]; };
    struct alignas(16) F4 { float v[4]; };
    int n=0, size=0, pad=0, degree=0;
    int offsets[max_degree]{};
    bool flt=false;
    Buffer<D4> weights;
    Buffer<F4> weights_f;
    Buffer<double> diag, state[2], difference[2], local, local_d;

    bool build(int nn, const int* ptr, const int* col, const double* val, const double* shift) {
        int d=0, reach=0;
        for (int i=0; i<nn; ++i) {
            const int count=ptr[i+1]-ptr[i];
            if (count>max_degree) return false;
            for (int k=ptr[i]; k<ptr[i+1]; ++k) {
                const int off=col[k]-i;
                if (std::abs(off)>1024) return false;
                if (std::find(offsets,offsets+d,off)==offsets+d) {
                    if (d==max_degree) return false;
                    offsets[d++]=off;
                    reach=std::max(reach,std::abs(off));
                }
            }
        }
        if (d<2 || (d&1) || ptr[nn] < 0.8*nn*d) return false;
        n=nn;
        size=(n+3)&~3;
        degree=d;
        pad=(reach+7)&~3;
        std::sort(offsets,offsets+d);
        int slot[2049]{};
        for (int j=0; j<d; ++j) slot[1024+offsets[j]]=j;
        const int total=size/4*d;
        flt=true;
        for (int k=0; k<ptr[n]; ++k)
            if (double(float(val[k]))!=val[k]) { flt=false; break; }
        if (flt) {
            weights_f=buffer<F4>(total);
            std::memset(weights_f.get(),0,total*sizeof(F4));
        } else {
            weights=buffer<D4>(total);
            std::memset(weights.get(),0,total*sizeof(D4));
        }
        for (int i=0; i<n; ++i) {
            for (int k=ptr[i]; k<ptr[i+1]; ++k) {
                const int pos=(i/4)*d+slot[1024+col[k]-i];
                if (flt) weights_f[pos].v[i%4]=float(val[k]);
                else weights[pos].v[i%4]=val[k];
            }
        }
        diag=buffer<double>(size);
        std::copy_n(shift,n,diag.get());
        std::fill_n(diag.get()+n,size-n,0.);
        for (int k=0; k<2; ++k) {
            state[k]=buffer<double>(size+2*pad);
            difference[k]=buffer<double>(size);
            std::fill_n(state[k].get(),pad,0.);
            std::fill_n(state[k].get()+pad+size,pad,0.);
        }
        if (size>=35000) {
            local=buffer<double>(tile+4*pad);
            std::fill_n(local.get(),tile+4*pad,0.);
            local_d=buffer<double>(tile+2*pad);
        }
#ifdef PROFILE
        std::fprintf(stderr,"Banded degree %d, pad %d, n %d, exact float %d\n",d,pad,n,flt);
#endif
        return true;
    }

    template<int mode, bool use_float, size_t... js>
    inline __m256d dot(const double* v, __m256d center, __m256d z, int pos,
                       std::index_sequence<js...>) const {
        __m256d a[2]{z,_mm256_setzero_pd()};
        auto edge = [&](auto j) {
            constexpr int k=decltype(j)::value;
            auto x=_mm256_loadu_pd(v+offsets[k]);
            auto w=use_float ? _mm256_cvtps_pd(_mm_load_ps(weights_f[pos+k].v)) :
                               _mm256_load_pd(weights[pos+k].v);
            if constexpr (mode==1) x=_mm256_sub_pd(x,center);
            if constexpr (mode==2) x=_mm256_add_pd(x,_mm256_xor_pd(center,_mm256_and_pd(w,_mm256_set1_pd(-0.))));
            a[k&1]=_mm256_fmadd_pd(x,w,a[k&1]);
        };
        (edge(std::integral_constant<int,js>{}),...);
        return _mm256_add_pd(a[0],a[1]);
    }

    template<int d, int mode, bool use_float, bool last>
    [[gnu::noinline]] void pass(double* __restrict v, double* __restrict diffo, int lo_out,
              const double* __restrict u, const double* __restrict diffi, int lo_in,
              const double* __restrict b0, const double* __restrict b1,
              int start, int end, Coefficients::Pair c, double multiplier) const {
        const auto e=_mm256_set1_pd(c.e), f=_mm256_set1_pd(c.f), w=_mm256_set1_pd(multiplier);
        for (int i=start; i<end; i+=4) {
            const double* x=u+(i-lo_in);
            const auto vi=_mm256_load_pd(x);
            const auto a=_mm256_mul_pd(_mm256_load_pd(diag.get()+i),vi);
            const auto s=dot<mode,use_float>(x,vi,a,(i/4)*d,std::make_index_sequence<d>{});
            const auto q=_mm256_fmadd_pd(_mm256_load_pd(b0+i),e,
                                     _mm256_mul_pd(_mm256_load_pd(b1+i),f));
            const auto z=_mm256_fmadd_pd(w,s,_mm256_add_pd(q,_mm256_load_pd(diffi+(i-lo_in))));
            if constexpr (last) _mm256_store_pd(v+(i-lo_out),z);
            else {
                _mm256_store_pd(diffo+(i-lo_out),z);
                _mm256_store_pd(v+(i-lo_out),_mm256_add_pd(z,vi));
            }
        }
    }

    template<int d, int mode, bool use_float>
    const double* evaluate(const Coefficients& p, int count, double scale,
                           const double* b0, const double* b1) {
        double* u=state[0].get()+pad, *v=state[1].get()+pad;
        double* df0=difference[0].get(), *df1=difference[1].get();
        const auto c=p.c[count];
        for (int i=0; i<size; ++i) u[i]=df0[i]=c.e*b0[i]+c.f*b1[i];
        if (!count) return u;
        int k=count-1;
#ifndef NO_TILE
        if (local && count>64 && pad<680) {
            for (; k>=2; k-=2) {
                for (int j=0; j<size; j+=tile) {
                    int end=std::min(size,j+tile);
                    const int lo=std::max(0,j-pad), hi=std::min(size,end+pad);
                    pass<d,mode,use_float,false>(local.get()+pad,local_d.get(),lo,u,df0,0,b0,b1,
                                                  lo,hi,p.c[k],scale);
                    pass<d,mode,use_float,false>(v,df1,0,local.get()+pad,local_d.get(),lo,b0,b1,
                                                  j,end,p.c[k-1],scale);
                }
                std::swap(u,v);
                std::swap(df0,df1);
            }
        }
#endif
        for (;k>=1;--k) {
            pass<d,mode,use_float,false>(v,df1,0,u,df0,0,b0,b1,0,size,p.c[k],scale);
            std::swap(u,v);
            std::swap(df0,df1);
        }
        pass<d,mode,use_float,true>(v,nullptr,0,u,df0,0,b0,b1,0,size,p.c[0],0.5*scale);
        return v;
    }

    template<int mode, bool use_float>
    const double* solve(const Coefficients& p, int d, double scale, const double* b0, const double* b1) {
        switch (degree) {
            case 2: return evaluate<2,mode,use_float>(p,d,scale,b0,b1);
            case 4: return evaluate<4,mode,use_float>(p,d,scale,b0,b1);
            case 6: return evaluate<6,mode,use_float>(p,d,scale,b0,b1);
            case 8: return evaluate<8,mode,use_float>(p,d,scale,b0,b1);
            case 10: return evaluate<10,mode,use_float>(p,d,scale,b0,b1);
            default: return evaluate<12,mode,use_float>(p,d,scale,b0,b1);
        }
    }

    template<int mode>
    const double* solve(const Coefficients& p, int d, double scale, const double* b0, const double* b1) {
        if (flt) return solve<mode,true>(p,d,scale,b0,b1);
        return solve<mode,false>(p,d,scale,b0,b1);
    }
};
