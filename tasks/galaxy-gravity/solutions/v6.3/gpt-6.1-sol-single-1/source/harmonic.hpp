#pragma once

class Harmonic {
    struct HI { int x,y,z,degree,sign; };
    struct DI {
        int x,y,z,degree;
        int prev[3]{}, prev2[3]{};
        double c[3]{}, b[3]{};
        int trace1 = 0, trace2 = 0;
    };
    struct MT { int src,shift; double sign; };
    struct MR { int x,y,z,w; };
    int p, nc, nm, nd;
    int hm[34][34][3], dm[34][34][3];
    std::vector<HI> hi;
    std::vector<MR> mr;
    std::vector<DI> di;
    std::vector<std::vector<int>> kernel;
    std::vector<std::vector<MT>> translate;
    std::vector<int> h_to_d;
    std::vector<std::array<int,3>> gradient;
    std::vector<double> moments, minus, locals;
    const std::vector<Node>* nodes;
    const std::vector<double> *x,*y,*z,*mass;
    const std::vector<int> *moment_orders = nullptr, *local_orders = nullptr;
    std::vector<double> packed;
    std::vector<size_t> offsets;
    bool budgeted=false,down_cut=false;

    static double sum4(__m256d a) {
        __m128d b = _mm_add_pd(_mm256_castpd256_pd128(a),
                              _mm256_extractf128_pd(a,1));
        return _mm_cvtsd_f64(_mm_add_sd(b,_mm_unpackhi_pd(b,b)));
    }
    int hindex(int a,int b,int c) const {
        if (a < 0 || b < 0 || c < 0 || c > 1) return -1;
        return hm[a][b][c];
    }
    double value(const double* v,int a,int b,int c) const {
        int k = hindex(a,b,c);
        return k < 0 ? 0 : v[k];
    }
    void monomials(double a,double b,double c,double* out,int limit) const {
        out[0] = 1;
        out[1089] = 0;
        for (int degree = 1; degree <= limit; ++degree) {
            double inv = 1.0/degree;
            int begin = degree*degree;
            int mid = begin+degree+1;
            int end = (degree+1)*(degree+1);
            for (int k = begin; k < mid; ++k) {
                const MR& t = mr[k];
                out[k] = (a*out[t.x]+b*out[t.y]-c*(out[t.z]+out[t.w]))*inv;
            }
            for (int k = mid; k < end; ++k) {
                const MR& t = mr[k];
                out[k] = (a*out[t.x]+b*out[t.y]+c*out[t.z])*inv;
            }
        }
    }
    void shift(const double* input,double* output,double dx,double dy,double dz,
               int degree,bool local) const {
        double aa[1100]{}, bb[1100]{};
        int count = (degree+1)*(degree+1);
        std::copy(input,input+count,aa);
        double* a = aa; double* b = bb;
        double delta[2] = {dx,dy};
        for (int axis = 0; axis < 2; ++axis) if (delta[axis] != 0) {
            double power[34]; power[0] = 1;
            for (int k = 1; k <= degree; ++k) power[k] = power[k-1]*delta[axis]/k;
            for (int n = 0; n < count; ++n) {
                const HI& t = hi[n];
                int limit = local ? degree-t.degree : (axis ? t.y : t.x);
                double sum = 0;
                for (int k = 0; k <= limit; ++k) {
                    int off = local ? k : -k;
                    int index = hm[t.x+(axis == 0 ? off : 0)]
                                  [t.y+(axis == 1 ? off : 0)][t.z];
                    sum += a[index]*power[k];
                }
                b[n] = sum;
            }
            std::swap(a,b);
        }
        std::copy(a,a+count,output);
        if (dz == 0) return;
        for (int step = 1; step <= degree; ++step) {
            double factor = dz/step;
            int begin = local ? 0 : step*step;
            int end = local ? (degree-step+1)*(degree-step+1) : count;
            for (int n = begin; n < end; ++n) {
                const HI& t = hi[n];
                if (local) {
                    if (t.z == 0) {
                        int k = hm[t.x][t.y][1];
                        b[n] = factor*(k < count ? a[k] : 0);
                    } else {
                        int k = hm[t.x+2][t.y][0];
                        int l = hm[t.x][t.y+2][0];
                        b[n] = -factor*((k < count ? a[k] : 0)+(l < count ? a[l] : 0));
                    }
                } else {
                    if (t.z == 1) b[n] = factor*a[hm[t.x][t.y][0]];
                    else b[n] = -factor*(value(a,t.x-2,t.y,1)+value(a,t.x,t.y-2,1));
                }
                output[n] += b[n];
            }
            std::swap(a,b);
        }
    }
    void expand_local(const double* l,double* full,int degree) const {
        for (int k = 0; k < (degree+1)*(degree+1); ++k) full[h_to_d[k]] = l[k];
        for (int k = 0; k < 1+3*degree*(degree+1)/2; ++k)
            if (di[k].z == 2)
                full[k] = -full[di[k].trace1]-full[di[k].trace2];
    }
public:
    explicit Harmonic(int order) : p(order) {
        p = std::max(2,std::min(32,p));
        nc = (p+1)*(p+1);
        nm = p*p;
        std::memset(hm,-1,sizeof(hm));
        std::memset(dm,-1,sizeof(dm));
        for (int d = 0; d <= p; ++d) {
            for (int zz = 0; zz <= 1 && zz <= d; ++zz)
                for (int xx = 0; xx <= d-zz; ++xx) {
                    int yy = d-zz-xx;
                    hm[xx][yy][zz] = (int)hi.size();
                    hi.push_back({xx,yy,zz,d,(d&1) ? -1 : 1});
                }
            for (int zz = 0; zz <= 2 && zz <= d; ++zz)
                for (int xx = 0; xx <= d-zz; ++xx) {
                    int yy = d-zz-xx;
                    dm[xx][yy][zz] = (int)di.size();
                    DI t{};
                    t.x = xx; t.y = yy; t.z = zz; t.degree = d;
                    di.push_back(t);
                }
        }
        nd = (int)di.size();
        mr.resize(nc);
        auto safe_h = [&](int x,int y,int z) {
            int k = hindex(x,y,z);
            return k < 0 ? 1089 : k;
        };
        for (int k = 1; k < nc; ++k) {
            const HI& t = hi[k];
            mr[k] = {safe_h(t.x-1,t.y,t.z),safe_h(t.x,t.y-1,t.z),
                     t.z == 0 ? safe_h(t.x-2,t.y,1) : safe_h(t.x,t.y,0),
                     t.z == 0 ? safe_h(t.x,t.y-2,1) : 1089};
        }
        h_to_d.resize(nc);
        for (int k = 0; k < nc; ++k)
            h_to_d[k] = dm[hi[k].x][hi[k].y][hi[k].z];
        for (int k = 1; k < nd; ++k) {
            DI& t = di[k];
            if (t.z == 2) {
                t.trace1 = dm[t.x+2][t.y][0];
                t.trace2 = dm[t.x][t.y+2][0];
                continue;
            }
            int a[3] = {t.x,t.y,t.z};
            int axis = 0;
            if (a[1] > a[axis]) axis = 1;
            if (a[2] > a[axis]) axis = 2;
            for (int d = 0; d < 3; ++d) {
                int b[3] = {a[0],a[1],a[2]};
                if (a[d] > 0) {
                    --b[d];
                    t.prev[d] = dm[b[0]][b[1]][b[2]];
                    t.c[d] = d == axis ? 2*a[d]-1 : 2*a[d];
                }
                if (a[d] > 1) {
                    --b[d];
                    t.prev2[d] = dm[b[0]][b[1]][b[2]];
                    t.b[d] = d == axis ? (a[d]-1)*(a[d]-1) : a[d]*(a[d]-1);
                }
            }
        }
        kernel.resize(nc);
        for (int n = 1; n < nc; ++n) {
            int q = (p-hi[n].degree+1)*(p-hi[n].degree+1);
            auto& row = kernel[n];
            row.resize(q);
            for (int k = 0; k < q; ++k)
                row[k] = dm[hi[n].x+hi[k].x][hi[n].y+hi[k].y]
                           [hi[n].z+hi[k].z];
        }
        translate.resize(nm);
        for (int n = 0; n < nm; ++n) {
            int q = (p-hi[n].degree)*(p-hi[n].degree);
            for (int k = 0; k < q; ++k) {
                int xx = hi[n].x+hi[k].x;
                int yy = hi[n].y+hi[k].y;
                int zz = hi[n].z+hi[k].z;
                if (zz < 2) translate[hm[xx][yy][zz]].push_back({n,k,1});
                else {
                    translate[hm[xx+2][yy][0]].push_back({n,k,-1});
                    translate[hm[xx][yy+2][0]].push_back({n,k,-1});
                }
            }
        }
        gradient.resize(nm);
        for (int k = 0; k < nm; ++k)
            gradient[k] = {dm[hi[k].x+1][hi[k].y][hi[k].z],
                           dm[hi[k].x][hi[k].y+1][hi[k].z],
                           dm[hi[k].x][hi[k].y][hi[k].z+1]};
    }
    void init(const std::vector<Node>& nn,const std::vector<double>& xx,
              const std::vector<double>& yy,const std::vector<double>& zz,
              const std::vector<double>& mm) {
        nodes = &nn; x = &xx; y = &yy; z = &zz; mass = &mm;
        moments.resize(nn.size()*nm);
        minus.resize(moments.size());
        locals.resize(nn.size()*nc);
    }
    void set_orders(const std::vector<int>& mo,const std::vector<int>& lo) {
        moment_orders = &mo;
        local_orders = &lo;
    }
    void set_budget(bool b) {budgeted=b;}
    void upward(int ni) {
        const Node& n = (*nodes)[ni];
        double* m = moments.data()+(size_t)ni*nm;
        double mono[1100];
        int degree = moment_orders ? (*moment_orders)[ni] : p-1;
        int count = degree >= 0 ? (degree+1)*(degree+1) : 0;
        if (n.leaf()) {
            if (degree < 0) return;
            for (int i = n.begin; i < n.end; ++i) {
                monomials((*x)[i]-n.c[0],(*y)[i]-n.c[1],(*z)[i]-n.c[2],
                          mono,degree);
                double mi = (*mass)[i];
                for (int k = 0; k < count; ++k) m[k] += mi*mono[k];
            }
        } else for (int ch : {n.left,n.right}) {
            upward(ch);
            if (degree < 0) continue;
            const Node& c = (*nodes)[ch];
            const double* cm = moments.data()+(size_t)ch*nm;
            shift(cm,mono,c.c[0]-n.c[0],c.c[1]-n.c[1],c.c[2]-n.c[2],degree,false);
            for (int k = 0; k < count; ++k) m[k] += mono[k];
        }
        double* mm = minus.data()+(size_t)ni*nm;
        for (int k = 0; k < count; ++k) mm[k] = hi[k].sign*m[k];
    }
    void m2l(int ia,int ib,int pair_order = -1,int direction = 0) {
        if (pair_order < 0) pair_order = p;
        const Node& a = (*nodes)[ia];
        const Node& b = (*nodes)[ib];
        double r[3] = {a.c[0]-b.c[0],a.c[1]-b.c[1],a.c[2]-b.c[2]};
        double invs = 1/(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
        double d[1634];
        d[0] = std::sqrt(invs);
        int dcount = 1+3*pair_order*(pair_order+1)/2;
        int count = (pair_order+1)*(pair_order+1);
        for (int k = 1; k < dcount; ++k) {
            const DI& t = di[k];
            if (t.z == 2) d[k] = -d[t.trace1]-d[t.trace2];
            else
                d[k] = -invs*(r[0]*t.c[0]*d[t.prev[0]]+t.b[0]*d[t.prev2[0]]
                            +r[1]*t.c[1]*d[t.prev[1]]+t.b[1]*d[t.prev2[1]]
                            +r[2]*t.c[2]*d[t.prev[2]]+t.b[2]*d[t.prev2[2]]);
        }
        const double* ma = moments.data()+(size_t)ia*nm;
        const double* mb = minus.data()+(size_t)ib*nm;
        double* la = locals.data()+(size_t)ia*nc;
        double* lb = locals.data()+(size_t)ib*nc;
        for (int n = 1; n < count; ++n) {
            const auto& row = kernel[n];
            int q = (pair_order-hi[n].degree+1)*(pair_order-hi[n].degree+1);
            __m256d sa = _mm256_setzero_pd(), sb = sa;
            int k = 0;
            for (; k+4 <= q; k+=4) {
                __m128i index = _mm_loadu_si128((const __m128i*)(row.data()+k));
                __m256d v = _mm256_i32gather_pd(d,index,8);
                sa = _mm256_fmadd_pd(v,_mm256_loadu_pd(mb+k),sa);
                sb = _mm256_fmadd_pd(v,_mm256_loadu_pd(ma+k),sb);
            }
            double va = sum4(sa), vb = sum4(sb);
            for (; k < q; ++k) {
                va += d[row[k]]*mb[k];
                vb += d[row[k]]*ma[k];
            }
            if (direction != 2) la[n] += va;
            if (direction != 1) lb[n] += hi[n].sign*vb;
        }
    }
    void prepare(double rx,double ry,double rz,int order) {
        double r[3] = {rx,ry,rz};
        double invs = 1/(rx*rx+ry*ry+rz*rz);
        double d[1634];
        d[0] = std::sqrt(invs);
        int dcount = 1+3*order*(order+1)/2;
        int count = (order+1)*(order+1);
        for (int k = 1; k < dcount; ++k) {
            const DI& t = di[k];
            if (t.z == 2) d[k] = -d[t.trace1]-d[t.trace2];
            else d[k] = -invs*(r[0]*t.c[0]*d[t.prev[0]]+t.b[0]*d[t.prev2[0]]
                             +r[1]*t.c[1]*d[t.prev[1]]+t.b[1]*d[t.prev2[1]]
                             +r[2]*t.c[2]*d[t.prev[2]]+t.b[2]*d[t.prev2[2]]);
        }
        offsets.resize(count);
        packed.clear();
        for (int n = 1; n < count; ++n) {
            offsets[n] = packed.size();
            int q = (order-hi[n].degree+1)*(order-hi[n].degree+1);
            const auto& row = kernel[n];
            for (int k = 0; k < q; ++k) packed.push_back(d[row[k]]);
        }
    }
    void cached_m2l(int ia,int ib,int order,double invscale,int direction = 0) {
        double ma[1089], mb[1089], powers[34];
        powers[0] = 1;
        for (int k = 1; k <= order+1; ++k) powers[k] = powers[k-1]*invscale;
        int mcount = order*order;
        const double* sma = moments.data()+(size_t)ia*nm;
        const double* smb = minus.data()+(size_t)ib*nm;
        for (int k = 0; k < mcount; ++k) {
            ma[k] = sma[k]*powers[hi[k].degree];
            mb[k] = smb[k]*powers[hi[k].degree];
        }
        double* la = locals.data()+(size_t)ia*nc;
        double* lb = locals.data()+(size_t)ib*nc;
        int count = (order+1)*(order+1);
        for (int n = 1; n < count; ++n) {
            const double* row = packed.data()+offsets[n];
            int q = (order-hi[n].degree+1)*(order-hi[n].degree+1);
            if(q==1) {
                if(direction!=2) la[n]+=row[0]*mb[0]*powers[hi[n].degree+1];
                if(direction!=1) lb[n]+=hi[n].sign*row[0]*ma[0]*powers[hi[n].degree+1];
                continue;
            }
            DotPair sums=packed_dot(row,ma,mb,q);
            double va=sums.a,vb=sums.b;
            if (direction != 2) la[n] += va*powers[hi[n].degree+1];
            if (direction != 1) lb[n] += hi[n].sign*vb*powers[hi[n].degree+1];
        }
    }
    void m2p(int ia,int ib,int source_order,std::vector<double>& ax,
             std::vector<double>& ay,std::vector<double>& az) {
        const Node& a = (*nodes)[ia];
        const Node& b = (*nodes)[ib];
        const double* m = minus.data()+(size_t)ib*nm;
        int degree = source_order+1;
        int dcount = 1+3*degree*(degree+1)/2;
        int mcount = (source_order+1)*(source_order+1);
        double d[1634];
        for (int i = a.begin; i < a.end; ++i) {
            double r[3] = {(*x)[i]-b.c[0],(*y)[i]-b.c[1],(*z)[i]-b.c[2]};
            double invs = 1/(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
            d[0] = std::sqrt(invs);
            for (int k = 1; k < dcount; ++k) {
                const DI& t = di[k];
                if (t.z == 2) d[k] = -d[t.trace1]-d[t.trace2];
                else d[k] = -invs*(r[0]*t.c[0]*d[t.prev[0]]+t.b[0]*d[t.prev2[0]]
                                 +r[1]*t.c[1]*d[t.prev[1]]+t.b[1]*d[t.prev2[1]]
                                 +r[2]*t.c[2]*d[t.prev[2]]+t.b[2]*d[t.prev2[2]]);
            }
            double acc[3]{};
            for (int k = 0; k < mcount; ++k) {
                acc[0] += m[k]*d[gradient[k][0]];
                acc[1] += m[k]*d[gradient[k][1]];
                acc[2] += m[k]*d[gradient[k][2]];
            }
            ax[i] += acc[0]; ay[i] += acc[1]; az[i] += acc[2];
        }
    }
    void downward(int ni,std::vector<double>& ax,std::vector<double>& ay,
                  std::vector<double>& az) {
        const Node& n = (*nodes)[ni];
        const double* l = locals.data()+(size_t)ni*nc;
        double mono[1100], full[1634];
        int degree = local_orders ? (*local_orders)[ni] : p;
        if(budgeted && (ni&63)==0 && cpu_time()>4.65) down_cut=true;
        if(down_cut) degree=std::min(degree,6);
        if (degree < 1) {
            if (!n.leaf()) {
                downward(n.left,ax,ay,az); downward(n.right,ax,ay,az);
            }
            return;
        }
        int count = (degree+1)*(degree+1);
        int mcount = degree*degree;
        if (n.leaf()) {
            expand_local(l,full,degree);
            for (int i = n.begin; i < n.end; ++i) {
                monomials((*x)[i]-n.c[0],(*y)[i]-n.c[1],(*z)[i]-n.c[2],mono,degree-1);
                double a[3]{};
                for (int k = 0; k < mcount; ++k) {
                    a[0] += mono[k]*full[gradient[k][0]];
                    a[1] += mono[k]*full[gradient[k][1]];
                    a[2] += mono[k]*full[gradient[k][2]];
                }
                ax[i] += a[0]; ay[i] += a[1]; az[i] += a[2];
            }
        } else for (int ch : {n.left,n.right}) {
            const Node& c = (*nodes)[ch];
            double* cl = locals.data()+(size_t)ch*nc;
            shift(l,mono,c.c[0]-n.c[0],c.c[1]-n.c[1],c.c[2]-n.c[2],degree,true);
            for (int k = 1; k < count; ++k) cl[k] += mono[k];
            downward(ch,ax,ay,az);
        }
    }
};
