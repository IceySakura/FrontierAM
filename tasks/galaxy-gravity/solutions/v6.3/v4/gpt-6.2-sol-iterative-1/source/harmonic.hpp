// Harmonic Cartesian coefficients are represented using only z powers 0
// and 1. For derivatives of the Laplace kernel, z^2 = -(x^2+y^2).
struct HRow {
    int x,e,len,start;
};
struct HLayout {
    int idx[2][PMAX+1]{};
    int di[3][PMAX+1]{};
    int stride=0, dstride=0;
    std::vector<HRow> rows;
    std::vector<Quad> blocks;
    std::vector<uint32_t> terms;
};
struct HProject {
    uint16_t src, dst;
    int16_t val;
};
struct HGradient {
    uint16_t src,x,y,z,zz;
};
struct Harmonic {
    HLayout lay[PMAX+1];
    std::vector<HProject> project;
    std::vector<HGradient> grad[PMAX+1][2];
    Arena moments,locals;
    int p=0;

    void init(const Layout* full,int max_order) {
        p=max_order;
        for(int k=1;k<=p;++k) {
            auto& d=lay[k];
            for(int e=0;e<=std::min(2,k);++e) for(int x=0;x<=k-e;++x) {
                d.di[e][x]=d.dstride;
                d.dstride+=k-e-x+4;
            }
            for(int e=0;e<=1;++e) for(int x=0;x<=k-e;++x) {
                d.idx[e][x]=d.stride;
                d.rows.push_back({x,e,k-e-x+1,d.stride});
                d.stride+=k-e-x+1;
            }
            for(auto a:d.rows) for(int y=0;y<a.len;y+=4) {
                int left=k-a.x-a.e-y;
                int first=d.terms.size();
                for(int e=0;e<=std::min(left,1);++e) for(int x=0;x<=left-e;++x) {
                    int dr=d.di[a.e+e][a.x+x]+y;
                    int mo=d.idx[e][x];
                    for(int yy=0;yy<=left-x-e;++yy)
                        d.terms.push_back(uint32_t(mo+yy)|(uint32_t(dr+yy)<<16));
                }
                d.blocks.push_back({0,a.x,y,a.e,int(d.terms.size())-first,first});
            }
        }
        const auto &f=lay[p];
        moments.init(f.stride); locals.init(f.stride);
        for(int k=1;k<=p;++k) for(auto& q:lay[k].blocks)
            q.out=f.idx[q.z][q.x]+q.y;
        for(int k=1;k<=p;++k) for(auto r:lay[k].rows) for(int j=0;j+1<r.len;++j) {
            const auto& l=lay[k];
            grad[k][r.e].push_back({uint16_t(f.idx[r.e][r.x]+j),
                uint16_t(l.idx[r.e][r.x+1]+j),uint16_t(r.start+j+1),
                uint16_t(r.e?l.idx[0][r.x+2]+j:l.idx[1][r.x]+j),
                uint16_t(r.e?l.idx[0][r.x]+j+2:0)});
        }
        for(auto r:full[p].rows) for(int z=0;z<r.len;++z) {
            int z2=z/2,e=z%2, bin=1;
            for(int x=0;x<=z2;++x) {
                int dst=f.idx[e][r.x+2*x]+r.y+2*(z2-x);
                project.push_back({uint16_t(r.start+z),uint16_t(dst),int16_t((z2&1)?-bin:bin)});
                bin=bin*(z2-x)/(x+1);
            }
        }
    }

    void build(Node& a,const double* src) {
        if(a.hm>=0) return;
        a.hm=moments.alloc();
        double *d=moments[a.hm];
        for(auto q:project) d[q.dst]+=q.val*src[q.src];
    }

    void point_coefficients(V* d,int k,V x,V y,V z,V quad,V d0) const {
        auto& l=lay[k];
        const V zero=_mm256_setzero_pd();
        V a[PMAX+1],b[PMAX+1],c[PMAX+1],p0[PMAX+1],p1[PMAX+1],zero_row[PMAX+1];
        a[0]=b[0]=p0[0]=zero;
        for(int j=1;j<=k;++j) {
            a[j]=x*_mm256_set1_pd(-2.*j); b[j]=y*_mm256_set1_pd(-2.*j);
            c[j]=y*_mm256_set1_pd(1.-2*j);
            p0[j]=quad*_mm256_set1_pd(-j*(j-1.));
            p1[j]=quad*_mm256_set1_pd(-(j-1.)*(j-1.));
        }
        std::fill(zero_row,zero_row+k+1,zero);
        for(auto r:l.rows) {
            V *out=d+r.start;
            const V *q1=r.x?d+l.idx[r.e][r.x-1]:zero_row,
                    *q2=r.x>1?d+l.idx[r.e][r.x-2]:zero_row;
            V mx=a[r.x], cx=p0[r.x];
            if(!r.e) {
                V last=r.x?_mm256_fmadd_pd(p1[r.x],q2[0],
                           (x*_mm256_set1_pd(1-2*r.x))*q1[0]):d0;
                out[0]=last;
                V prev=zero;
                for(int j=1;j<r.len;++j) {
                    V u=_mm256_fmadd_pd(mx,q1[j],cx*q2[j]);
                    V next=_mm256_fmadd_pd(c[j],last,_mm256_fmadd_pd(p1[j],prev,u));
                    out[j]=next; prev=last;last=next;
                }
            } else {
                const V *h=d+l.idx[0][r.x];
                V last=zero,prev=zero;
                for(int j=0;j<r.len;++j) {
                    V u=_mm256_fmadd_pd(mx,q1[j],_mm256_fnmadd_pd(z,h[j],cx*q2[j]));
                    V next=_mm256_fmadd_pd(b[j],last,_mm256_fmadd_pd(p0[j],prev,u));
                    out[j]=next;prev=last;last=next;
                }
            }
        }
    }

    void point_gradient(const V* d,const double* m,int k,V& x,V& y,V& z) const {
        for(const auto& q:grad[k][0]) {
            V f=_mm256_set1_pd(m[q.src]);
            x=_mm256_fnmadd_pd(f,d[q.x],x);
            y=_mm256_fnmadd_pd(f,d[q.y],y);
            z=_mm256_fnmadd_pd(f,d[q.z],z);
        }
        for(const auto& q:grad[k][1]) {
            V f=_mm256_set1_pd(m[q.src]);
            x=_mm256_fnmadd_pd(f,d[q.x],x);
            y=_mm256_fnmadd_pd(f,d[q.y],y);
            z=_mm256_fmadd_pd(f,d[q.z]+d[q.zz],z);
        }
    }

    void apply(const Layout& full,const double *der,Node& a,Node& b,double sa,double sb,double inv) {
        using D = __m256d;
        const int k=full.p;
        const auto &l=lay[k], &big=lay[p];
        if(a.hl<0) a.hl=locals.alloc();
        if(b.hl<0) b.hl=locals.alloc();
        double *la=locals[a.hl], *lb=locals[b.hl];
        const double *ma=moments[a.hm], *mb=moments[b.hm];
        alignas(32) double w0[512],w1[512],dt[1600],pa[PMAX+5]{1.},pb[PMAX+5]{1.},fi[PMAX+1]{1.};
        for(int j=1;j<=k;++j) {
            pa[j]=pa[j-1]*sa; pb[j]=pb[j-1]*sb; fi[j]=fi[j-1]/j;
        }
        for(int e=0;e<=std::min(2,k);++e) for(int x=0;x<=k-e;++x) {
            double *u=dt+l.di[e][x];
            int n=k-e-x+1;
            for(int y=0;y<n;++y) u[y]=der[full.idx[x][y]+e];
            u[n]=u[n+1]=u[n+2]=0.;
        }
        for(auto r:l.rows) {
            int so=big.idx[r.e][r.x],deg=r.x+r.e;
            for(int j=0;j<r.len;++j) {
                w0[r.start+j]=pa[deg+j]*ma[so+j];
                w1[r.start+j]=pb[deg+j]*mb[so+j];
            }
        }
        for(auto q:l.blocks) {
            D a0=_mm256_setzero_pd(),b0=a0,a1=a0,b1=a0;
            const uint32_t* t=l.terms.data()+q.beg;
            int j=0;
            for(;j+1<q.count;j+=2) {
                uint32_t u=t[j],v=t[j+1];
                D f0=_mm256_loadu_pd(dt+(u>>16)),f1=_mm256_loadu_pd(dt+(v>>16));
                a0=_mm256_fmadd_pd(f0,_mm256_set1_pd(w1[u&65535]),a0);
                b0=_mm256_fmadd_pd(f0,_mm256_set1_pd(w0[u&65535]),b0);
                a1=_mm256_fmadd_pd(f1,_mm256_set1_pd(w1[v&65535]),a1);
                b1=_mm256_fmadd_pd(f1,_mm256_set1_pd(w0[v&65535]),b1);
            }
            if(j<q.count) {
                uint32_t u=t[j];
                D f=_mm256_loadu_pd(dt+(u>>16));
                a0=_mm256_fmadd_pd(f,_mm256_set1_pd(w1[u&65535]),a0);
                b0=_mm256_fmadd_pd(f,_mm256_set1_pd(w0[u&65535]),b0);
            }
            alignas(32) double xa[4],xb[4];
            _mm256_store_pd(xa,a0+a1); _mm256_store_pd(xb,b0+b1);
            int deg=q.x+q.y+q.z;
            double s0=inv*fi[q.x];
            for(int v=(deg==0);v<std::min(4,k+1-deg);++v) {
                double s=s0*fi[q.y+v];
                la[q.out+v]+=xa[v]*s*pa[deg+v];
                lb[q.out+v]+=xb[v]*s*pb[deg+v];
            }
        }
    }

    void inflate(const Layout& full,const Node& a,double* dest) const {
        alignas(32) double w[MAXCO];
        const auto &h=lay[p];
        const double *src=locals[a.hl];
        for(auto row:h.rows)
            for(int j=0;j<row.len;++j) w[full.idx[row.x][j]+row.e]=src[row.start+j];
        for(int z=2;z<=p;++z) {
            double sz=-1./(z*(z-1.));
            for(int x=0;x<=p-z;++x) for(int y=0;y<=p-z-x;++y)
                w[full.idx[x][y]+z]=sz*((x+2.)*(x+1.)*w[full.idx[x+2][y]+z-2]+
                                       (y+2.)*(y+1.)*w[full.idx[x][y+2]+z-2]);
        }
        for(int i=0;i<full.stride;++i) dest[i]+=w[i];
    }
    double work(int k) const {
        return double(lay[k].terms.size())+6.0*lay[k].stride;
    }
};
