// Locus scales are calibrated from many different levels of a provisional
// tree rather than from just its deepest clades.
struct CalibratedMatrices {
    vector<Matrix> d;
    CalibratedMatrices(const vector<Gene> &genes,const Tree &guide,const Matrix &reference)
        :d(8,Matrix(N*N)) {
        struct Pair { int a,b; double time; };
        vector<Pair> pairs;
        auto many=[&](auto &&self,int u,int p)->vector<int> {
            if(u<N) return {u};
            auto [a,b]=guide.children(u,p);
            auto x=self(self,a,u),y=self(self,b,u);
            int na=min(4,int(x.size())),nb=min(4,int(y.size()));
            for(int i=0;i<na;++i) for(int j=0;j<nb;++j) {
                int q=x[(2*i+1)*x.size()/(2*na)],r=y[(2*j+1)*y.size()/(2*nb)];
                pairs.push_back({q,r,reference[q*N+r]});
            }
            x.insert(x.end(),y.begin(),y.end());
            return x;
        };
        many(many,guide.adj[0][0],0);
        sort(pairs.begin(),pairs.end(),[](const Pair &a,const Pair &b) { return a.time<b.time; });
        // Reuse the same distant anchor set for nearby tips whenever possible.
        array<vector<int>,N> far;
        vector<int> perm;
        for(int j=1;j<N;++j) perm.push_back(j);
        for(int i=0;i<N;++i) {
            sort(perm.begin(),perm.end(),[&](int a,int b) { return reference[i*N+a]>reference[i*N+b]; });
            int n=64;
            for(int j=4;j<n+4;++j) if(perm[j]!=i) far[i].push_back(perm[j]);
        }
        struct Fit {
            double scale, wt;
            array<double,9> old{}, current{};
            array<float,N> offset{};
        };
        vector<Fit> fit;
        for(const auto &g:genes) {
            Fit f;
            double r=max(.009,double(g.rate));
            f.wt=(.21+.79*sqrt(g.resolution))/(1+.4*pow(r/1.2,2));
            vector<float> src;
            for(size_t j=pairs.size()/6;j<pairs.size()*5/6;++j) {
                const auto &p=pairs[j];
                double y=g.dist[p.a*N+p.b];
                src.push_back(float(y/max(.12,p.time)));
            }
            double s=median(src);
            f.scale=clamp(.23*r+.77*s,.42*r,1.85*r);
            f.current[0]=f.old[0]=0;
            for(int i=1;i<8;++i) {
                size_t lo=pairs.size()*(i-1)/7, hi=pairs.size()*i/7;
                src.clear();
                for(size_t j=lo;j<hi;++j) src.push_back(g.dist[pairs[j].a*N+pairs[j].b]/f.scale);
                f.old[i]=max(f.old[i-1]+.038,double(median(src)));
                f.current[i]=max(f.current[i-1]+.038,pairs[(lo+hi)/2].time);
            }
            f.old[8]=f.old[7]+1.; f.current[8]=f.current[7]+.95;
            auto map=[&](double x) {
                int i=1;
                while(i<8 && f.old[i]<x) ++i;
                double slope=clamp((f.current[i]-f.current[i-1])/(f.old[i]-f.old[i-1]),.45,2.);
                return f.current[i-1]+(x-f.old[i-1])*slope;
            };
            vector<float> offsets;
            for(int i=0;i<N;++i) {
                src.clear();
                for(int j:far[i]) {
                    double x=g.dist[i*N+j]/f.scale;
                    src.push_back(.56*x+.44*map(x)-reference[i*N+j]);
                }
                offsets.push_back(median(src));
            }
            float center=median(offsets);
            for(int i=0;i<N;++i) f.offset[i]=clamp(offsets[i]-center,-.42f,.42f);
            fit.push_back(f);
        }
        vector<pair<double,double>> v;
        v.reserve(genes.size());
        for(int i=0;i<N;++i) for(int j=i+1;j<N;++j) {
            int ij=i*N+j;
            v.clear();
            double sums[8]{}, wt=0;
            for(size_t g=0;g<genes.size();++g) {
                const auto &f=fit[g];
                double x=genes[g].dist[ij]/f.scale;
                int a=1;
                while(a<8 && f.old[a]<x) ++a;
                double slope=clamp((f.current[a]-f.current[a-1])/(f.old[a]-f.old[a-1]),.45,2.);
                double y=f.current[a-1]+(x-f.old[a-1])*slope;
                double z=max(0.,.60*x+.4*y);
                double u=max(0.,z-.53*(f.offset[i]+f.offset[j]));
                sums[0]+=f.wt*exp(-z/.35); sums[1]+=f.wt*exp(-u/.35);
                sums[2]+=f.wt*exp(-u/.60); sums[3]+=f.wt*exp(-u/.19);
                sums[4]+=f.wt*min(2.65,u);
                sums[5]+=f.wt*exp(-x/.29);
                v.emplace_back(u,f.wt); wt+=f.wt;
            }
            sort(v.begin(),v.end());
            double taken=0;
            double goal=wt*.50;
            for(auto [a,b]:v) if(taken<goal) {
                double w=min(b,goal-taken);
                sums[6]+=w*a;
                taken+=w;
            }
            taken=0;
            for(auto [a,b]:v) {
                taken+=b;
                if(taken>=wt*.21) { sums[7]=a; break; }
            }
            for(int k=0;k<8;++k) {
                double x=(k==4)?sums[k]/wt:(k==6)?sums[k]/goal:(k==7)?sums[k]:
                    -log(max(1e-40,sums[k]/wt));
                d[k][ij]=d[k][j*N+i]=x;
            }
        }
        for(auto &m:d) {
            double sum=0;
            for(int i=1;i<N;++i) for(int j=i+1;j<N;++j) sum+=m[i*N+j];
            double f=max(1e-8,sum/((N-1)*(N-2)/2));
            for(double &x:m) x/=f;
        }
    }
};
