// Included in the solver's implementation namespace.
struct Packed {
    static constexpr int longrow = -65;
    struct Ind { int j[4]; };
    struct Ind16 { uint16_t j[4]; };
    struct alignas(32) Val { double x[4]; };
    struct alignas(16) Val32 { float x[4]; };
    struct Group { int offset, degree; };
    int size = 0;
    double common = 0;
    bool short_index = false;
    bool short_value = false;
    bool identity=true;
    const double *source0=nullptr,*source1=nullptr;
    Buffer<Group> groups;
    Buffer<Ind> indices;
    Buffer<Ind16> indices16;
    Buffer<Val> values;
    Buffer<Val32> values32;
    Buffer<int> order, inverse, tail_col;
    Buffer<double> diagonal, rhs0, rhs1;

    void build(int n, const int* ptr, const int* col, const double* val,
               const double* shift, int mode) {
        size = (n+3)&~3;
        order = buffer<int>(size);
        inverse = buffer<int>(n);
        diagonal = buffer<double>(size);
        groups = buffer<Group>(size/4);
        common = val[0];
        if (mode == 2) common = 0;
        else {
            for (int k=1; k<ptr[n]; ++k)
                if (val[k] != common) {
                    common = 0;
                    break;
                }
        }
#ifndef NO_COMPACT
        short_index=size+4 <= 65536;
        if (!common) {
            short_value=true;
            for (int j=0; j<ptr[n]; ++j) {
                if (double(float(val[j])) != val[j]) {
                    short_value=false;
                    break;
                }
            }
        }
#endif
        constexpr int block = 256;
        for (int start=0; start<n; start+=block) {
            int end = std::min(n, start+block);
            int degrees[block], padding = 0, groupmax = 0;
            for (int i=start; i<end; ++i) {
                int d = std::min(65,ptr[i+1]-ptr[i]);
                degrees[i-start] = d;
                groupmax = std::max(groupmax, d);
                padding -= d;
                if ((i&3) == 3 || i+1 == end) {
                    padding += 4*groupmax;
                    groupmax = 0;
                }
            }
            if (padding > (end-start)/2 && padding*5 > ptr[end]-ptr[start]) {
                int count[66]{};
                for (int i=start; i<end; ++i) ++count[degrees[i-start]];
                int total = start;
                for (int& c : count) {
                    int next = total+c;
                    c = total;
                    total = next;
                }
                for (int i=start; i<end; ++i) order[count[degrees[i-start]]++] = i;
            } else {
                for (int i=start; i<end; ++i) order[i] = i;
            }
        }
        for (int i=0; i<n; ++i) {
            inverse[order[i]] = i;
            diagonal[i] = shift[order[i]];
            identity &= order[i]==i;
        }
        if (!identity) {
            rhs0=buffer<double>(size+4);
            rhs1=buffer<double>(size+4);
        }
        for (int i=n; i<size; ++i) {
            order[i] = -1;
            diagonal[i] = 0;
        }
        int at = 0, tail = 0;
        for (int i=0; i<size; i+=4) {
            int d = 0;
            for (int k=i; k<std::min(n,i+4); ++k) {
                int r = order[k];
                d = std::max(d,ptr[r+1]-ptr[r]);
            }
            groups[i/4] = {d<=64 ? at : tail, d <= 64 ? d : longrow};
            if (d <= 64) at += d;
            else if (!identity) {
                for (int k=i; k<std::min(n,i+4); ++k) {
                    const int r=order[k];
                    tail+=ptr[r+1]-ptr[r];
                }
            }
        }
        if (tail) tail_col=buffer<int>(tail);
        if (short_index) indices16 = buffer<Ind16>(at);
        else indices = buffer<Ind>(at);
        if (common == 0) {
            if (short_value) values32 = buffer<Val32>(at);
            else values = buffer<Val>(at);
        }
        for (int i=0; i<size; i+=4) {
            auto& g = groups[i/4];
            if (g.degree==longrow && tail_col) {
                int pos=g.offset;
                for (int k=i; k<std::min(n,i+4); ++k) {
                    const int r=order[k];
                    for (int j=ptr[r]; j<ptr[r+1]; ++j) tail_col[pos++]=inverse[col[j]];
                }
            }
            if (g.degree <= 0) continue;
            int start[4], lens[4];
            bool linear = true;
            for (int k=0; k<4; ++k) {
                int r = order[i+k];
                start[k] = r < 0 ? 0 : ptr[r];
                lens[k] = r < 0 ? 0 : ptr[r+1]-ptr[r];
            }
            for (int s=0; s<g.degree; ++s) {
                int dst[4];
                const int off = g.offset+s;
                for (int k=0; k<4; ++k) {
                    int j = s < lens[k] ? inverse[col[start[k]+s]] :
                            common && mode == 0 ? size+k : i+k;
                    dst[k] = j;
                    if (short_index) indices16[off].j[k]=uint16_t(j);
                    else indices[off].j[k]=j;
                    if (common == 0) {
                        const double x = s < lens[k] ? val[start[k]+s] : 0.;
                        if (short_value) values32[off].x[k] = float(x);
                        else values[off].x[k] = x;
                    }
                }
                linear &= dst[1] == dst[0]+1 && dst[2] == dst[0]+2 && dst[3] == dst[0]+3;
            }
            if (linear) g.degree = -g.degree;
        }
#ifdef PROFILE
        int nlinear=0, nlong=0;
        for (int i=0; i<size/4; ++i) {
            const int d = groups[i].degree;
            if (d == longrow) ++nlong;
            else if (d<0) ++nlinear;
        }
        std::fprintf(stderr,"Packed %d rows, %d slots (%d edges), %d linear groups, %d long groups, uniform %d, indices %d bits, values %d bits\n",
                     n,4*at,ptr[n],nlinear,nlong,common!=0,short_index?16:32,short_value?32:64);
#endif
    }

    void set_rhs(const double* a, const double* b, int n) {
        if (identity) {
            source0=a;
            source1=b;
            return;
        }
        for (int i=0; i<n; ++i) {
            int j = order[i];
            rhs0[i] = a[j];
            rhs1[i] = b[j];
        }
        for (int i=n; i<size+4; ++i) rhs0[i] = rhs1[i] = 0;
        source0=rhs0.get();
        source1=rhs1.get();
    }

    template<bool linear, bool small_index>
    inline __m256d gather(const double* x, int at) const {
        if constexpr(linear) {
            if constexpr(small_index) return _mm256_loadu_pd(x+indices16[at].j[0]);
            else return _mm256_loadu_pd(x+indices[at].j[0]);
        }
        else {
#ifdef PACK_MANUAL_GATHER
            int j0,j1,j2,j3;
            if constexpr(small_index) {
                const auto& j=indices16[at].j;
                j0=j[0];j1=j[1];j2=j[2];j3=j[3];
            } else {
                const auto& j=indices[at].j;
                j0=j[0];j1=j[1];j2=j[2];j3=j[3];
            }
            const __m128d lo = _mm_loadh_pd(_mm_load_sd(x+j0), x+j1);
            const __m128d hi = _mm_loadh_pd(_mm_load_sd(x+j2), x+j3);
            return _mm256_insertf128_pd(_mm256_castpd128_pd256(lo),hi,1);
#else
            const auto all = _mm256_castsi256_pd(_mm256_set1_epi64x(-1));
            __m128i index;
            if constexpr(small_index) index=_mm_cvtepu16_epi32(
                    _mm_loadl_epi64(reinterpret_cast<const __m128i*>(indices16[at].j)));
            else index=_mm_load_si128(reinterpret_cast<const __m128i*>(indices[at].j));
            return _mm256_mask_i32gather_pd(_mm256_setzero_pd(),x,index,all,8);
#endif
        }
    }

    template<int mode, bool uniform, bool linear, bool small_index, bool small_value>
    inline __m256d add_edge(__m256d acc, const double* v, __m256d vi, int pos) const {
        __m256d x = gather<linear,small_index>(v,pos);
        const __m256d w = uniform ? _mm256_setzero_pd() :
                small_value ? _mm256_cvtps_pd(_mm_load_ps(values32[pos].x)) : _mm256_load_pd(values[pos].x);
        if constexpr (mode == 1) x = _mm256_sub_pd(x,vi);
        if constexpr (mode == 2) {
            const __m256d sign = _mm256_and_pd(w, _mm256_set1_pd(-0.0));
            x = _mm256_add_pd(x, _mm256_xor_pd(vi,sign));
        }
        if constexpr (uniform) return _mm256_add_pd(acc,x);
        else return _mm256_fmadd_pd(w,x,acc);
    }

    template<int mode, bool uniform, bool linear, bool small_index, bool small_value>
    inline __m256d group_dot(const double* v, __m256d vi, int offset, int d, __m256d z) const {
        __m256d r0 = uniform ? _mm256_setzero_pd() : z, r1 = _mm256_setzero_pd();
        int k=0;
        for (; k+1<d; k+=2) {
            r0 = add_edge<mode,uniform,linear,small_index,small_value>(r0,v,vi,offset+k);
            r1 = add_edge<mode,uniform,linear,small_index,small_value>(r1,v,vi,offset+k+1);
        }
        if (k<d) r0 = add_edge<mode,uniform,linear,small_index,small_value>(r0,v,vi,offset+k);
        const __m256d r = _mm256_add_pd(r0,r1);
        if constexpr (uniform) return _mm256_fmadd_pd(_mm256_set1_pd(common),r,z);
        else return r;
    }

    template<int mode, bool uniform>
    inline double scalar_edge(const double* v, double x, int j, double w, double z) const {
        double q = v[j];
        if constexpr (mode == 1) q -= x;
        if constexpr (mode == 2) {
            const auto sign = std::bit_cast<uint64_t>(w) & (uint64_t(1)<<63);
            q += std::bit_cast<double>(std::bit_cast<uint64_t>(x)^sign);
        }
        if constexpr (uniform) return q+z;
        else return std::fma(w,q,z);
    }

    template<int mode, bool uniform>
    [[gnu::noinline]] __m256d long_dot(int start, int offset, const double* v, const int* ptr,
                                      const int* col, const double* val) const {
        alignas(32) double sums[4];
        const int* mapped=tail_col ? tail_col.get()+offset : nullptr;
        for (int k=0; k<4; ++k) {
            int i = start+k;
            int r = order[i];
            double vi = v[i];
            double s0=diagonal[i]*vi, s1=0, s2=0, s3=0;
            if constexpr (uniform) s0=0;
            if (r>=0) {
                int j = ptr[r], stop = ptr[r+1];
                const int* ci=mapped ? mapped : col+j;
                if (mapped) mapped+=stop-j;
                for (; j+3<stop; j+=4) {
                    s0=scalar_edge<mode,uniform>(v,vi,ci[0],uniform ? common : val[j],s0);
                    s1=scalar_edge<mode,uniform>(v,vi,ci[1],uniform ? common : val[j+1],s1);
                    s2=scalar_edge<mode,uniform>(v,vi,ci[2],uniform ? common : val[j+2],s2);
                    s3=scalar_edge<mode,uniform>(v,vi,ci[3],uniform ? common : val[j+3],s3);
                    ci+=4;
                }
                for (; j<stop; ++j) s0=scalar_edge<mode,uniform>(v,vi,*ci++,uniform ? common : val[j],s0);
            }
            sums[k] = (s0+s1)+(s2+s3);
            if constexpr (uniform) sums[k]=std::fma(common,sums[k],diagonal[i]*vi);
        }
        return _mm256_load_pd(sums);
    }

    template<int mode, bool uniform, bool last, bool small_index, bool small_value>
    [[gnu::noinline]] void pass_impl(double* __restrict out, const double* __restrict v, double* __restrict diff,
              double c, double d, double scale, const int* ptr, const int* col,
              const double* val) const {
        const __m256d cc = _mm256_set1_pd(c), dd = _mm256_set1_pd(d), ss = _mm256_set1_pd(scale);
        const bool dual=c!=0 && d!=0;
        const double* src=c!=0 ? source0 : source1;
        const __m256d coeff=c!=0 ? cc : dd;
        for (int i=0; i<size; i+=4) {
            const auto g = groups[i/4];
            const __m256d vi = _mm256_load_pd(v+i);
            __m256d z = _mm256_mul_pd(_mm256_load_pd(diagonal.get()+i),vi);
            if (g.degree == longrow) z = long_dot<mode,uniform>(i,g.offset,v,ptr,col,val);
            else if (g.degree<0) z = group_dot<mode,uniform,true,small_index,small_value>(v,vi,g.offset,-g.degree,z);
            else z = group_dot<mode,uniform,false,small_index,small_value>(v,vi,g.offset,g.degree,z);
            const __m256d input = _mm256_load_pd(src+i);
            const __m256d cd = dual ? _mm256_fmadd_pd(cc,input,_mm256_mul_pd(dd,_mm256_load_pd(source1+i))) :
                                           _mm256_mul_pd(coeff,input);
            const __m256d w = _mm256_fmadd_pd(ss,z,_mm256_add_pd(_mm256_load_pd(diff+i),cd));
            if constexpr (last) _mm256_store_pd(out+i,w);
            else {
                _mm256_store_pd(diff+i,w);
                _mm256_store_pd(out+i,_mm256_add_pd(vi,w));
            }
        }
    }

    template<int mode, bool uniform, bool last>
    inline void pass(double* out, const double* v, double* diff,
                     double c, double d, double scale, const int* ptr, const int* col,
                     const double* val) const {
        if (short_index) {
            if (!uniform && short_value) pass_impl<mode,uniform,last,true,true>(out,v,diff,c,d,scale,ptr,col,val);
            else pass_impl<mode,uniform,last,true,false>(out,v,diff,c,d,scale,ptr,col,val);
        } else {
            if (!uniform && short_value) pass_impl<mode,uniform,last,false,true>(out,v,diff,c,d,scale,ptr,col,val);
            else pass_impl<mode,uniform,last,false,false>(out,v,diff,c,d,scale,ptr,col,val);
        }
    }

    template<int mode, bool uniform, bool small_index, bool small_value>
    [[gnu::noinline]] void multiply_impl(double* __restrict out, const double* __restrict v,
                  double lower, double t, const int* ptr, const int* col,
                  const double* val) const {
        const __m256d sc = _mm256_set1_pd(t), lo = _mm256_set1_pd(lower);
        for (int i=0; i<size; i+=4) {
            const Group g = groups[i/4];
            const __m256d vi = _mm256_load_pd(v+i);
            __m256d z = _mm256_mul_pd(_mm256_load_pd(diagonal.get()+i),vi);
            if (g.degree == longrow) z = long_dot<mode,uniform>(i,g.offset,v,ptr,col,val);
            else if (g.degree<0) z = group_dot<mode,uniform,true,small_index,small_value>(v,vi,g.offset,-g.degree,z);
            else z = group_dot<mode,uniform,false,small_index,small_value>(v,vi,g.offset,g.degree,z);
            _mm256_store_pd(out+i, _mm256_mul_pd(sc,_mm256_fmadd_pd(lo,vi,z)));
        }
        std::fill_n(out+size,4,0.);
    }

    template<int mode, bool uniform>
    inline void multiply(double* out, const double* v, double lower, double t,
                         const int* ptr, const int* col, const double* val) const {
        if (short_index) {
            if (!uniform && short_value) multiply_impl<mode,uniform,true,true>(out,v,lower,t,ptr,col,val);
            else multiply_impl<mode,uniform,true,false>(out,v,lower,t,ptr,col,val);
        } else {
            if (!uniform && short_value) multiply_impl<mode,uniform,false,true>(out,v,lower,t,ptr,col,val);
            else multiply_impl<mode,uniform,false,false>(out,v,lower,t,ptr,col,val);
        }
    }
};
