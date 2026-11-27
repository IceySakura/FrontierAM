struct SmallBlocks {
    static constexpr int limit = 8;
    std::vector<int> order, ends;
    Buffer<long double> e, f;

    bool find(int n, const int* ptr, const int* col, const double* val) {
        auto assigned=buffer<unsigned char>(n);
        std::fill_n(assigned.get(),n,0);
        order.reserve(n);
        ends.push_back(0);
        for (int i=0; i<n; ++i) {
            if (assigned[i]) continue;
            const int start=int(order.size());
            order.push_back(i);
            assigned[i]=1;
            for (int u=start; u<int(order.size()); ++u) {
                const int j=order[u];
                for (int k=ptr[j]; k<ptr[j+1]; ++k) {
                    const int x=col[k];
                    if (assigned[x] || val[k]==0) continue;
                    if (int(order.size())-start>=limit) return false;
                    order.push_back(x);
                    assigned[x]=1;
                }
            }
            ends.push_back(int(order.size()));
        }
        return true;
    }

    static void jacobi(long double (&a)[limit][limit], long double (&q)[limit][limit], int n) {
        long double scale = 0;
        for (int i=0; i<n; ++i) scale=std::max(scale,std::abs(a[i][i]));
        const long double eps=0.20L*LDBL_EPSILON*scale;
        for (int pass=0; pass<24; ++pass) {
            bool change=false;
            for (int i=0; i<n; ++i) {
                for (int j=i+1; j<n; ++j) {
                    if (std::abs(a[i][j])<=eps) continue;
                    change=true;
                    const long double diff=0.5L*(a[j][j]-a[i][i]);
                    const long double b=a[i][j];
                    const long double r=std::sqrt(diff*diff+b*b);
                    const long double t=b/(diff+std::copysign(r,diff));
                    const long double c=1/std::sqrt(1+t*t), s=c*t;
                    a[i][i]-=t*b;
                    a[j][j]+=t*b;
                    a[i][j]=a[j][i]=0;
                    for (int k=0; k<n; ++k) {
                        if (k!=i && k!=j) {
                            const long double x=a[k][i], y=a[k][j];
                            a[k][i]=a[i][k]=c*x-s*y;
                            a[k][j]=a[j][k]=s*x+c*y;
                        }
                        const long double x=q[k][i], y=q[k][j];
                        q[k][i]=c*x-s*y;
                        q[k][j]=s*x+c*y;
                    }
                }
            }
            if (!change) break;
        }
    }

    bool prepare(int n, const int* ptr, const int* col, const double* val, double t) {
        if (!find(n,ptr,col,val)) return false;
        int nz=0;
        for (size_t i=1; i<ends.size(); ++i) {
            int k=ends[i]-ends[i-1];
            nz+=k*k;
        }
        e=buffer<long double>(nz);
        f=buffer<long double>(nz);
        int off=0;
        for (size_t b=1; b<ends.size(); ++b) {
            int first=ends[b-1], m=ends[b]-first;
            long double a[limit][limit]{}, q[limit][limit]{};
            for (int i=0; i<m; ++i) {
                int row=order[first+i];
                q[i][i]=1;
                for (int k=ptr[row]; k<ptr[row+1]; ++k) {
                    if (!val[k]) continue;
                    for (int j=0; j<m; ++j) {
                        if (col[k]==order[first+j]) {
                            a[i][j]=static_cast<long double>(t)*val[k];
                            break;
                        }
                    }
                }
            }
            jacobi(a,q,m);
            long double values_e[limit], values_f[limit];
            for (int i=0; i<m; ++i) {
                const long double x=std::max(0.L,a[i][i]);
                values_e[i]=x>48 ? 0.L : std::exp(-x);
                values_f[i]=x==0 ? t : (t*(-std::expm1(-x))/x);
            }
            for (int i=0; i<m; ++i) for (int j=0; j<m; ++j) {
                long double u=0,w=0;
                for (int k=0; k<m; ++k) {
                    const long double s=q[i][k]*q[j][k];
                    u+=s*values_e[k];
                    w+=s*values_f[k];
                }
                e[off]=u;
                f[off++]=w;
            }
        }
        return true;
    }

    const double* solve(const double* b0, const double* b1, double* y) const {
        int off=0;
        for (size_t b=1; b<ends.size(); ++b) {
            const int first=ends[b-1], m=ends[b]-first;
            double a[limit], c[limit];
            for (int i=0; i<m; ++i) {
                int j=order[first+i];
                a[i]=b0[j];
                c[i]=b1[j];
            }
            for (int i=0; i<m; ++i) {
                long double r=0,s=0;
                for (int j=0; j<m; ++j,++off) {
                    r+=e[off]*a[j];
                    s+=f[off]*c[j];
                }
                y[order[first+i]]=double(r+s);
            }
        }
        return y;
    }
};
