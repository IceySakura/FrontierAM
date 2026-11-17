struct Vec3 {
    double x[3];
    double& operator[](int i) { return x[i]; }
    double operator[](int i) const { return x[i]; }
};

static double dot3(const Vec3& a, const Vec3& b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

static Vec3 subtract3(const Vec3& a, const Vec3& b) {
    return {{a[0]-b[0], a[1]-b[1], a[2]-b[2]}};
}

static void eigen3(double matrix[3][3], double values[3], double vectors[3][3]) {
    for (int i=0; i<3; ++i)
        for (int j=0; j<3; ++j) vectors[i][j] = i==j ? 1. : 0.;
    for (int iteration=0; iteration<20; ++iteration) {
        int p=0, q=1;
        for (int a=0; a<3; ++a) for (int b=a+1; b<3; ++b)
            if (std::abs(matrix[a][b])>std::abs(matrix[p][q])) { p=a; q=b; }
        if (std::abs(matrix[p][q])<1e-12) break;
        double theta=.5*std::atan2(2*matrix[p][q], matrix[q][q]-matrix[p][p]);
        double c=std::cos(theta), s=std::sin(theta);
        double app=matrix[p][p], aqq=matrix[q][q], apq=matrix[p][q];
        matrix[p][p]=c*c*app-2*s*c*apq+s*s*aqq;
        matrix[q][q]=s*s*app+2*s*c*apq+c*c*aqq;
        matrix[p][q]=matrix[q][p]=0;
        for (int r=0; r<3; ++r) if (r!=p && r!=q) {
            double rp=matrix[r][p], rq=matrix[r][q];
            matrix[r][p]=matrix[p][r]=c*rp-s*rq;
            matrix[r][q]=matrix[q][r]=s*rp+c*rq;
        }
        for (int r=0; r<3; ++r) {
            double rp=vectors[r][p], rq=vectors[r][q];
            vectors[r][p]=c*rp-s*rq;
            vectors[r][q]=s*rp+c*rq;
        }
    }
    for (int i=0; i<3; ++i) values[i]=matrix[i][i];
    for (int i=0; i<3; ++i) for (int j=i+1; j<3; ++j)
        if (values[j]<values[i]) {
            std::swap(values[i],values[j]);
            for (int k=0; k<3; ++k) std::swap(vectors[k][i],vectors[k][j]);
        }
}

struct Box {
    int lo[3], hi[3];
};

struct Component {
    // 0: Gaussian spot; 1: capped tube; 2: sheet; 3: anisotropic spot.
    int type=0, axis=0, begin=0;
    bool elongated=false;
    double frame[3][3]={{1.,0.,0.},{0.,1.,0.},{0.,0.,1.}};
    std::vector<double> p;
    Box box;
    std::vector<float> image, blurred;
    std::vector<std::vector<float>> jac;

    Component(int t, const std::vector<double>& parameters, int direction=0)
        : type(t), axis(direction), p(parameters) {}
};

struct Sample {
    Vec3 position;
    double value;
};

static void smooth_volume(Convolution& H, const float* x, float* out, double sigma) {
    std::vector<float> w(5), temp(H.n), temp2(H.n);
    double total=0;
    for (int i=0; i<5; ++i) { w[i]=std::exp(-.5*(i-2)*(i-2)/(sigma*sigma)); total+=w[i]; }
    for (float& v:w) v/=total;
    H.axis(x,temp.data(),w,0,false);
    H.axis(temp.data(),temp2.data(),w,1,false);
    H.axis(temp2.data(),out,w,2,false);
}

static Vec3 voxel_position(int i, const Convolution& H) {
    return {{double(i/(H.ny*H.nx)), double((i/H.nx)%H.ny), double(i%H.nx)}};
}

static int voxel_index(const Vec3& p, const Convolution& H) {
    int z=std::clamp(int(std::rint(p[0])),0,H.nz-1);
    int y=std::clamp(int(std::rint(p[1])),0,H.ny-1);
    int x=std::clamp(int(std::rint(p[2])),0,H.nx-1);
    return (z*H.ny+y)*H.nx+x;
}

static double weighted_moments(const std::vector<Sample>& samples,
                               const std::vector<double>& weights,
                               Vec3& centre, double eigen[3], double vectors[3][3]) {
    centre={{0,0,0}};
    double total=0;
    for (size_t i=0; i<samples.size(); ++i) {
        total+=weights[i];
        for (int d=0; d<3; ++d) centre[d]+=weights[i]*samples[i].position[d];
    }
    if (total<1e-12) return 0;
    for (int d=0; d<3; ++d) centre[d]/=total;
    double matrix[3][3]={};
    for (size_t i=0; i<samples.size(); ++i) {
        Vec3 r=subtract3(samples[i].position,centre);
        for (int a=0; a<3; ++a) for (int b=0; b<3; ++b)
            matrix[a][b]+=weights[i]*r[a]*r[b]/total;
    }
    eigen3(matrix,eigen,vectors);
    return total;
}

struct Random {
    unsigned long long state=0x9e3779b97f4a7c15ULL;
    double uniform() {
        state^=state>>12; state^=state<<25; state^=state>>27;
        return ((state*2685821657736338717ULL)>>11)*(1.0/9007199254740992.0);
    }
};

static std::vector<Component> tube_seeds(Convolution& H, const float* x,
                                        std::vector<float>& remaining,
                                        double deadline) {
    std::vector<Component> components;
    smooth_volume(H,x,remaining.data(),.5);
    const std::vector<float> original=remaining;
    float maximum=*std::max_element(remaining.begin(),remaining.end());
    std::vector<unsigned char> eligible(H.n,0), visited(H.n,0);
    std::vector<int> queue;
    const int strides[3]={H.ny*H.nx,H.nx,1}, dimensions[3]={H.nz,H.ny,H.nx};
    float component_threshold=std::max(.025f,.1f*maximum);
    for (int seed=0; seed<H.n; ++seed) if (!visited[seed] && original[seed]>component_threshold) {
        queue.clear(); queue.push_back(seed); visited[seed]=1;
        for (size_t j=0; j<queue.size(); ++j) {
            int i=queue[j]; Vec3 p=voxel_position(i,H);
            for (int d=0; d<3; ++d) for (int sign=-1; sign<=1; sign+=2) {
                if (p[d]+sign<0 || p[d]+sign>=dimensions[d]) continue;
                int k=i+sign*strides[d];
                if (!visited[k] && original[k]>component_threshold) { visited[k]=1; queue.push_back(k); }
            }
        }
        if (queue.size()<60) continue;
        std::vector<Sample> samples;
        std::vector<double> weights;
        for (int i:queue) {
            samples.push_back({voxel_position(i,H),original[i]});
            weights.push_back(original[i]);
        }
        Vec3 centre; double eigen[3],vectors[3][3];
        weighted_moments(samples,weights,centre,eigen,vectors);
        if (eigen[2]>5) for (int i:queue) eligible[i]=1;
    }
    Random random;
    for (int iteration=0; iteration<6; ++iteration) {
        if (monotonic_seconds()>deadline-.040) break;
        std::vector<Sample> samples;
        std::vector<double> cumulative;
        double total=0;
        for (int i=0; i<H.n; ++i) if (eligible[i] && remaining[i]>component_threshold) {
            samples.push_back({voxel_position(i,H),remaining[i]});
            total+=double(remaining[i])*remaining[i];
            cumulative.push_back(total);
        }
        if (samples.size()<40) break;
        double best=-1;
        Vec3 centre={{0,0,0}}, direction={{1,0,0}};
        for (int attempt=0; attempt<180; ++attempt) {
            if (attempt%8==0 && monotonic_seconds()>deadline-.045) break;
            size_t ia=std::lower_bound(cumulative.begin(),cumulative.end(),random.uniform()*total)-cumulative.begin();
            size_t ib=std::lower_bound(cumulative.begin(),cumulative.end(),random.uniform()*total)-cumulative.begin();
            ia=std::min(ia,samples.size()-1); ib=std::min(ib,samples.size()-1);
            Vec3 a=samples[ia].position;
            Vec3 line=subtract3(samples[ib].position,a);
            double length=std::sqrt(dot3(line,line));
            if (length<7) continue;
            for (int d=0; d<3; ++d) line[d]/=length;
            double score=0;
            for (const Sample& sample:samples) {
                Vec3 r=subtract3(sample.position,a);
                double t=dot3(r,line), rho=std::max(0.,dot3(r,r)-t*t);
                score+=sample.value*sample.value*std::exp(-.5*rho);
            }
            if (score>best) { best=score; centre=a; direction=line; }
        }
        if (best<0 || monotonic_seconds()>deadline-.035) break;
        double eigen[3], vectors[3][3];
        std::vector<double> weights(samples.size());
        for (int iteration=0; iteration<12; ++iteration) {
            double radius=iteration<6 ? 1.4 : 1.;
            for (size_t i=0; i<samples.size(); ++i) {
                Vec3 r=subtract3(samples[i].position,centre);
                double t=dot3(r,direction), rho=std::max(0.,dot3(r,r)-t*t);
                weights[i]=samples[i].value*std::exp(-rho/(2*radius*radius));
            }
            if (!weighted_moments(samples,weights,centre,eigen,vectors)) break;
            direction={{vectors[0][2],vectors[1][2],vectors[2][2]}};
        }
        std::vector<std::pair<double,double>> projection;
        double mass=0;
        for (const Sample& sample:samples) {
            Vec3 r=subtract3(sample.position,centre);
            double t=dot3(r,direction), rho=std::max(0.,dot3(r,r)-t*t);
            if (rho<6.25) {
                double weight=sample.value*std::exp(-.5*rho);
                projection.emplace_back(t,weight); mass+=weight;
            }
        }
        if (mass<10 || eigen[2]<5 || eigen[2]<3*std::max(eigen[1],.1)) break;
        std::sort(projection.begin(),projection.end());
        double left=projection.front().first, right=projection.back().first, sum=0;
        bool found=false;
        for (auto pair:projection) {
            sum+=pair.second;
            if (!found && sum>=.02*mass) { left=pair.first; found=true; }
            if (sum>=.98*mass) { right=pair.first; break; }
        }
        int bins=std::max(1,int(std::ceil(right-left))+1);
        std::vector<double> histogram(bins,0);
        for (int i=0; i<H.n; ++i) if (original[i]>std::max(.025f,.1f*maximum)) {
            Vec3 r=subtract3(voxel_position(i,H),centre);
            double t=dot3(r,direction), rho=std::max(0.,dot3(r,r)-t*t);
            if (rho<6.25 && t>=left && t<=right) {
                int b=std::clamp(int(t-left),0,bins-1);
                histogram[b]+=original[i]*std::exp(-.5*rho);
            }
        }
        std::vector<double> averaged=histogram;
        for (int i=0; i<bins; ++i) {
            averaged[i]=histogram[i];
            if (i>0) averaged[i]+=histogram[i-1];
            if (i+1<bins) averaged[i]+=histogram[i+1];
            averaged[i]/=1+(i>0)+(i+1<bins);
        }
        histogram.swap(averaged);
        double peak=*std::max_element(histogram.begin(),histogram.end());
        int occupied=0;
        for (double value:histogram) if (value>.2*peak) ++occupied;
        if (occupied<.65*bins) break;
        double width=std::clamp(std::sqrt(std::max((eigen[0]+eigen[1])*.5-.15,.2)/.6),.65,1.5);
        left+=1.2*width; right-=1.2*width;
        double length=right-left;
        if (length<5) break;
        std::vector<double> amplitudes;
        for (const Sample& sample:samples) {
            Vec3 r=subtract3(sample.position,centre);
            double t=dot3(r,direction), rho=std::max(0.,dot3(r,r)-t*t);
            if (rho<1) amplitudes.push_back(sample.value);
        }
        if (amplitudes.empty()) break;
        size_t q=std::min(amplitudes.size()-1,size_t(.85*amplitudes.size()));
        std::nth_element(amplitudes.begin(),amplitudes.begin()+q,amplitudes.end());
        std::vector<double> p(8);
        for (int d=0; d<3; ++d) {
            p[d]=centre[d]+left*direction[d];
            p[d+3]=centre[d]+right*direction[d];
        }
        p[6]=width; p[7]=amplitudes[q];
        components.emplace_back(1,p);
        Vec3 start={{p[0],p[1],p[2]}};
        for (int i=0; i<H.n; ++i) {
            Vec3 r=subtract3(voxel_position(i,H),start);
            double t=dot3(r,direction), rho=std::max(0.,dot3(r,r)-t*t);
            if (rho<12.96*width*width && t>-2*width && t<length+2*width) remaining[i]=0;
        }
    }
    return components;
}

static void add_spot_seed(std::vector<Component>& components, const Vec3& centre,
                          const float* x, Convolution& H, double separation) {
    for (const Component& c:components) if (c.type==0 || c.type==3 || c.type==4) {
        Vec3 prior={{c.p[0],c.p[1],c.p[2]}};
        Vec3 r=subtract3(prior,centre);
        if (dot3(r,r)<separation*separation) return;
    }
    components.emplace_back(0,std::vector<double>{centre[0],centre[1],centre[2],.95,
                            std::clamp(double(x[voxel_index(centre,H)]),.02,1.5)});
}

static std::vector<Component> spot_seeds(Convolution& H, const float* x, int limit=16, int flags=0) {
    std::vector<Component> components;
    std::vector<float> smooth(H.n);
    smooth_volume(H,x,smooth.data(),.6);
    float maximum=*std::max_element(smooth.begin(),smooth.end());
    float threshold=std::max(.035f,.15f*maximum);
    std::vector<unsigned char> visited(H.n,0);
    std::vector<int> queue;
    const int strides[3]={H.ny*H.nx,H.nx,1};
    const int dimensions[3]={H.nz,H.ny,H.nx};
    for (int seed=0; seed<H.n; ++seed) if (!visited[seed] && smooth[seed]>threshold) {
        queue.clear(); queue.push_back(seed); visited[seed]=1;
        for (size_t j=0; j<queue.size(); ++j) {
            int i=queue[j]; Vec3 p=voxel_position(i,H);
            for (int d=0; d<3; ++d) for (int s=-1; s<=1; s+=2) {
                if (p[d]+s<0 || p[d]+s>=dimensions[d]) continue;
                int k=i+s*strides[d];
                if (!visited[k] && smooth[k]>threshold) { visited[k]=1; queue.push_back(k); }
            }
        }
        if (queue.size()<12) continue;
        std::vector<Sample> samples;
        std::vector<double> weights;
        for (int i:queue) { samples.push_back({voxel_position(i,H),smooth[i]}); weights.push_back(smooth[i]); }
        Vec3 centre;
        double eigen[3],vectors[3][3];
        weighted_moments(samples,weights,centre,eigen,vectors);
        if (flags&1)
            std::fprintf(stderr,"spot region %zu centre %.3g %.3g %.3g eigen %.3g %.3g %.3g\n",
                         queue.size(),centre[0],centre[1],centre[2],eigen[0],eigen[1],eigen[2]);
        if ((flags&512) && ((eigen[2]>2 && eigen[2]>2*std::max(eigen[1],.1))
                            || (queue.size()>130 && eigen[2]>3))) {
            int count=eigen[2]>5.5 || (queue.size()>180 && eigen[1]>1.) ? 3 : 2;
            double distance=count==3 ? 1.1*std::sqrt(eigen[2])
                : (eigen[2]<2 ? 1.4 : 1.)*std::sqrt(eigen[2]);
            for (int k=0; k<count; ++k) {
                double position=count==3 ? k-1. : (k==0 ? -1. : 1.);
                Vec3 seed=centre;
                for (int d=0; d<3; ++d) seed[d]+=position*distance*vectors[d][2];
                add_spot_seed(components,seed,x,H,1.);
            }
        } else if (eigen[2]>2 && eigen[2]<6 && eigen[2]>2.5*std::max(eigen[1],.1)) {
            Vec3 a=centre,b=centre;
            for (int d=0; d<3; ++d) {
                double step=.8*std::sqrt(eigen[2])*vectors[d][2];
                a[d]-=step; b[d]+=step;
            }
            add_spot_seed(components,a,x,H,1.);
            add_spot_seed(components,b,x,H,1.);
        } else if (eigen[2]<2) {
            size_t before=components.size();
            add_spot_seed(components,centre,x,H,2.6);
            if (components.size()>before && eigen[2]>1.2 && eigen[2]>2*eigen[1]) {
                components.back().elongated=true;
                for (int a=0; a<3; ++a) for (int b=0; b<3; ++b)
                    components.back().frame[a][b]=vectors[a][b];
            }
        }
        if (int(components.size())>=limit) break;
    }
    std::vector<std::pair<float,int>> peaks;
    for (int i=0; i<H.n; ++i) if (smooth[i]>.035f) {
        Vec3 p=voxel_position(i,H);
        bool maximum=true;
        for (int z=std::max(0,int(p[0])-2); z<std::min(H.nz,int(p[0])+3) && maximum; ++z)
            for (int y=std::max(0,int(p[1])-2); y<std::min(H.ny,int(p[1])+3) && maximum; ++y)
                for (int x=std::max(0,int(p[2])-2); x<std::min(H.nx,int(p[2])+3); ++x)
                    if (smooth[(z*H.ny+y)*H.nx+x]>smooth[i]) { maximum=false; break; }
        if (maximum) peaks.emplace_back(smooth[i],i);
    }
    std::sort(peaks.rbegin(),peaks.rend());
    for (auto peak:peaks) {
        if (int(components.size())>=limit) break;
        add_spot_seed(components,voxel_position(peak.second,H),x,H,flags&16 ? 3.5 : 2.6);
    }
    if (flags&1) {
        for (const Component& c:components)
            std::fprintf(stderr,"spot seed %.3g %.3g %.3g amp %.3g\n",c.p[0],c.p[1],c.p[2],c.p[4]);
    }
    return components;
}

static std::vector<Component> sheet_seeds(Convolution& H, const float* x) {
    std::vector<Component> components;
    double total=0, square=0;
    for (int i=0; i<H.n; ++i) { total+=x[i]; square+=double(x[i])*x[i]; }
    double mean=total/H.n, energy=square/H.n-mean*mean;
    if (energy<1e-8) return components;
    int dimensions[3]={H.nz,H.ny,H.nx};
    std::vector<std::vector<double>> profiles(3);
    double explained=0;
    for (int d=0; d<3; ++d) {
        profiles[d].assign(dimensions[d],0);
        for (int i=0; i<H.n; ++i) profiles[d][int(voxel_position(i,H)[d])]+=x[i]*double(dimensions[d])/H.n;
        for (double value:profiles[d]) explained+=(value-mean)*(value-mean)/dimensions[d];
    }
    if (explained<.55*energy) return components;
    for (int d=0; d<3; ++d) {
        std::vector<double> sorted=profiles[d];
        std::sort(sorted.begin(),sorted.end());
        double baseline=sorted[size_t(sorted.size()*.3)];
        for (int i=0; i<dimensions[d]; ++i) {
            double value=profiles[d][i];
            if (value<baseline+.04) continue;
            bool peak=true;
            for (int j=std::max(0,i-2); j<std::min(dimensions[d],i+3); ++j)
                if (profiles[d][j]>value) peak=false;
            if (peak) components.emplace_back(2,std::vector<double>{double(i),.8,value-baseline,0.,0.},d);
        }
    }
    if (components.size()>5) components.clear();
    return components;
}

struct PhotonPixel {
    int start=3, count=1;
    double reference=1.;
    double coefficient[12]={};
};

static double integer_power(double x, unsigned int exponent) {
    double result=1;
    while (exponent) {
        if (exponent&1) result*=x;
        exponent>>=1;
        if (exponent) x*=x;
    }
    return result;
}

struct NoiseModel {
    double sigma, alpha, c0, c1;
    double spot_amplitude_mean=0., spot_amplitude_sd=.8;
    bool exact=false, posterior=false;
    std::vector<PhotonPixel> photons;
    explicit NoiseModel(double s, int flags=0): sigma(s), alpha(1.7*s) {
        if (flags&2) { spot_amplitude_mean=.85; spot_amplitude_sd=.35; }
        exact=flags&4;
        posterior=flags&8;
        c0=s*.3989422804014327;
        double ratio=alpha/s;
        c1=s*.3989422804014327*std::exp(-.5*ratio*ratio)
            -.5*alpha*std::erfc(ratio/std::sqrt(2.));
    }
    void initialize_likelihood(const float* y, int n) {
        if (!exact && !posterior) return;
        photons.resize(n);
        for (int i=0; i<n; ++i) {
            PhotonPixel& cell=photons[i];
            double observed=y[i]/alpha;
            int middle=std::max(0,int(std::min(observed,1e7)));
            cell.start=std::max(3,middle-4);
            cell.count=std::clamp(middle+5-cell.start,1,9);
            cell.reference=std::max(observed,.3);
            if (y[i]==0) { cell.start=3; cell.count=5; cell.reference=1.; }
            double logreference=std::log(cell.reference);
            double logweights[12];
            double maximum=-1e300;
            for (int k=0; k<3+cell.count; ++k) {
                int count=k<3 ? k : cell.start+k-3;
                double value;
                if (y[i]==0) {
                    value=std::log(std::max(.5*std::erfc(alpha*count/(sigma*std::sqrt(2.))),1e-300));
                } else {
                    double r=(y[i]-alpha*count)/sigma;
                    value=-.5*r*r;
                }
                value+=count*logreference-std::lgamma(count+1.);
                logweights[k]=value;
                maximum=std::max(maximum,value);
            }
            for (int k=0; k<3+cell.count; ++k)
                cell.coefficient[k]=std::exp(logweights[k]-maximum);
        }
    }
    double likelihood(int index, double f, double* score=nullptr, double* information=nullptr) const {
        const PhotonPixel& cell=photons[index];
        f=std::max(f,1e-12);
        double lambda=f/alpha, t=lambda/cell.reference;
        double power=1, p=0, first=0, second=0;
        for (int k=0; k<3; ++k) {
            double value=cell.coefficient[k]*power;
            p+=value; first+=k*value; second+=k*k*value;
            power*=t;
        }
        power=integer_power(t,cell.start);
        for (int k=0; k<cell.count; ++k) {
            int count=cell.start+k;
            double value=cell.coefficient[k+3]*power;
            p+=value; first+=count*value; second+=double(count)*count*value;
            power*=t;
        }
        if (score || information) {
            double mean=p>1e-290 ? first/p : cell.start;
            double variance=p>1e-290 ? std::max(0.,second/p-mean*mean) : 0.;
            if (score) *score=(lambda-mean)/f;
            if (information) {
                double fisher=1/std::max(alpha*f+sigma*sigma*.5,1e-16);
                *information=std::clamp((mean-variance)/(f*f),.05*fisher,100*fisher);
            }
            return 0.;
        }
        return lambda-std::log(std::max(p,1e-300));
    }
    void moments(double f, double& mean, double& derivative, double& variance) const {
        f=std::max(f,0.);
        double q=f/alpha, e=q<40 ? std::exp(-q) : 0.;
        double b=(c0+c1*q)*e;
        mean=f+b;
        derivative=1+(c1-c0-c1*q)*e/alpha;
        variance=std::max(.3408450569*sigma*sigma,
                          alpha*f+sigma*sigma*(1-.5*e)-2*f*b-b*b);
    }
};

static void convolve_profile(const std::vector<double>& source,
                             const std::vector<float>& kernel,
                             std::vector<double>& result) {
    int n=source.size(), m=kernel.size();
    result.assign(n,0.);
    for (int i=0; i<n; ++i) for (int j=0; j<m; ++j) {
        int k=i+j-m/2;
        if (k>=0 && k<n) result[i]+=source[k]*kernel[m-1-j];
    }
}

static void component_box(Component& c, Convolution& H) {
    int dimensions[3]={H.nz,H.ny,H.nx}, support[3]={H.hz/2,H.hy/2,H.hx/2};
    if (c.type==2) {
        for (int d=0; d<3; ++d) { c.box.lo[d]=0; c.box.hi[d]=dimensions[d]; }
        return;
    }
    double radius=c.type==1 ? c.p[6] : c.p[3];
    for (int d=0; d<3; ++d) {
        if (c.type==3) radius=c.p[d+3];
        if (c.type==4) {
            double variance=0;
            for (int j=0; j<3; ++j) variance+=std::pow(c.frame[d][j]*c.p[j+3],2);
            radius=std::sqrt(variance);
        }
        double left=c.p[d], right=c.p[d];
        if (c.type==1) { left=std::min(c.p[d],c.p[d+3]); right=std::max(c.p[d],c.p[d+3]); }
        c.box.lo[d]=std::clamp(int(std::floor(left-4*radius-support[d])),0,dimensions[d]);
        c.box.hi[d]=std::clamp(int(std::ceil(right+4*radius+support[d]))+1,0,dimensions[d]);
    }
}

static void evaluate_component(Component& c, Convolution& H, bool derivatives) {
    component_box(c,H);
    c.image.resize(H.n); c.blurred.resize(H.n);
    int parameters=c.p.size();
    if (derivatives) {
        c.jac.resize(parameters);
        for (auto& col:c.jac) col.resize(H.n);
    }
    if (c.type==4) {
        double amplitude=c.p[6];
        for (int i=0; i<H.n; ++i) {
            Vec3 r=subtract3(voxel_position(i,H),{{c.p[0],c.p[1],c.p[2]}});
            double rotated[3]={}, q=0;
            for (int j=0; j<3; ++j) {
                for (int d=0; d<3; ++d) rotated[j]+=r[d]*c.frame[d][j];
                q+=rotated[j]*rotated[j]/(c.p[j+3]*c.p[j+3]);
            }
            double basis=std::exp(-.5*q), image=amplitude*basis;
            c.image[i]=image;
            if (derivatives) {
                for (int d=0; d<3; ++d) {
                    double value=0;
                    for (int j=0; j<3; ++j)
                        value+=rotated[j]*c.frame[d][j]/(c.p[j+3]*c.p[j+3]);
                    c.jac[d][i]=image*value;
                }
                for (int j=0; j<3; ++j)
                    c.jac[j+3][i]=image*rotated[j]*rotated[j]/std::pow(c.p[j+3],3);
                c.jac[6][i]=basis;
            }
        }
        H.apply(c.image.data(),c.blurred.data());
        if (derivatives) {
            std::vector<float> temp(H.n);
            for (auto& col:c.jac) { H.apply(col.data(),temp.data()); col.swap(temp); }
        }
        return;
    }
    if (c.type==1) {
        Vec3 a={{c.p[0],c.p[1],c.p[2]}}, b={{c.p[3],c.p[4],c.p[5]}};
        Vec3 v=subtract3(b,a);
        double length2=std::max(dot3(v,v),1e-9), s=c.p[6], amplitude=c.p[7];
        double inverse=1/(s*s);
        for (int i=0; i<H.n; ++i) {
            Vec3 r=subtract3(voxel_position(i,H),a);
            double t=std::clamp(dot3(r,v)/length2,0.,1.);
            for (int d=0; d<3; ++d) r[d]-=t*v[d];
            double d2=dot3(r,r), basis=std::exp(-.5*d2*inverse), image=amplitude*basis;
            c.image[i]=image;
            if (derivatives) {
                for (int d=0; d<3; ++d) {
                    c.jac[d][i]=image*(1-t)*r[d]*inverse;
                    c.jac[d+3][i]=image*t*r[d]*inverse;
                }
                c.jac[6][i]=image*d2*inverse/s;
                c.jac[7][i]=basis;
            }
        }
        H.apply(c.image.data(),c.blurred.data());
        if (derivatives) {
            std::vector<float> temp(H.n);
            for (auto& col:c.jac) {
                H.apply(col.data(),temp.data());
                col.swap(temp);
            }
        }
        return;
    }
    const int dimensions[3]={H.nz,H.ny,H.nx};
    const std::vector<float>* kernels[3]={&H.az,&H.ay,&H.ax};
    std::vector<double> clean[3], g[3], dc[3], ds[3], unit[3], linear[3];
    for (int d=0; d<3; ++d) {
        std::vector<double> derivc(dimensions[d]), derivs(dimensions[d]);
        clean[d].resize(dimensions[d]);
        if (c.type==0 || c.type==3 || d==c.axis) {
            double centre=c.type==0 || c.type==3 ? c.p[d] : c.p[0];
            double s=c.type==3 ? c.p[d+3] : (c.type==0 ? c.p[3] : c.p[1]);
            for (int i=0; i<dimensions[d]; ++i) {
                double r=i-centre;
                clean[d][i]=std::exp(-.5*r*r/(s*s));
                derivc[i]=clean[d][i]*r/(s*s);
                derivs[i]=clean[d][i]*r*r/(s*s*s);
            }
            convolve_profile(clean[d],*kernels[d],g[d]);
            if (derivatives) {
                convolve_profile(derivc,*kernels[d],dc[d]);
                convolve_profile(derivs,*kernels[d],ds[d]);
            }
        }
        if (c.type==2 && d!=c.axis) {
            std::vector<double> ones(dimensions[d],1.), coords(dimensions[d]);
            for (int i=0; i<dimensions[d]; ++i)
                coords[i]=(i-(dimensions[d]-1)*.5)/(dimensions[d]*.5);
            convolve_profile(ones,*kernels[d],unit[d]);
            convolve_profile(coords,*kernels[d],linear[d]);
        }
    }
    if (c.type==0 || c.type==3) {
        double amplitude=c.type==3 ? c.p[6] : c.p[4];
        for (int z=0; z<H.nz; ++z) for (int y=0; y<H.ny; ++y) for (int x=0; x<H.nx; ++x) {
            int i=(z*H.ny+y)*H.nx+x;
            double basis=g[0][z]*g[1][y]*g[2][x];
            c.image[i]=amplitude*clean[0][z]*clean[1][y]*clean[2][x];
            c.blurred[i]=amplitude*basis;
            if (derivatives) {
                c.jac[0][i]=amplitude*dc[0][z]*g[1][y]*g[2][x];
                c.jac[1][i]=amplitude*g[0][z]*dc[1][y]*g[2][x];
                c.jac[2][i]=amplitude*g[0][z]*g[1][y]*dc[2][x];
                if (c.type==3) {
                    c.jac[3][i]=amplitude*ds[0][z]*g[1][y]*g[2][x];
                    c.jac[4][i]=amplitude*g[0][z]*ds[1][y]*g[2][x];
                    c.jac[5][i]=amplitude*g[0][z]*g[1][y]*ds[2][x];
                    c.jac[6][i]=basis;
                } else {
                    c.jac[3][i]=amplitude*(ds[0][z]*g[1][y]*g[2][x]+
                                          g[0][z]*ds[1][y]*g[2][x]+g[0][z]*g[1][y]*ds[2][x]);
                    c.jac[4][i]=basis;
                }
            }
        }
        return;
    }
    int d=c.axis, e=(d+1)%3, f=(d+2)%3;
    // Slopes are ordered by the remaining coordinate axes.
    if (e>f) std::swap(e,f);
    for (int z=0; z<H.nz; ++z) for (int y=0; y<H.ny; ++y) for (int x=0; x<H.nx; ++x) {
        int position[3]={z,y,x}, i=(z*H.ny+y)*H.nx+x;
        int rd=position[d], re=position[e], rf=position[f];
        double u=unit[e][re]*unit[f][rf];
        double l1=linear[e][re]*unit[f][rf], l2=unit[e][re]*linear[f][rf];
        double amplitude=c.p[2]*u+c.p[3]*l1+c.p[4]*l2;
        double te=(re-(dimensions[e]-1)*.5)/(dimensions[e]*.5);
        double tf=(rf-(dimensions[f]-1)*.5)/(dimensions[f]*.5);
        c.image[i]=clean[d][rd]*std::max(c.p[2]+c.p[3]*te+c.p[4]*tf,0.);
        c.blurred[i]=g[d][rd]*amplitude;
        if (derivatives) {
            c.jac[0][i]=dc[d][rd]*amplitude;
            c.jac[1][i]=ds[d][rd]*amplitude;
            c.jac[2][i]=g[d][rd]*u;
            c.jac[3][i]=g[d][rd]*l1;
            c.jac[4][i]=g[d][rd]*l2;
        }
    }
}

static void evaluate_model(std::vector<Component>& components, Convolution& H,
                           std::vector<float>& blurred, bool derivatives) {
    std::fill(blurred.begin(),blurred.end(),0.f);
    for (Component& c:components) {
        evaluate_component(c,H,derivatives);
        for (int i=0; i<H.n; ++i) blurred[i]+=c.blurred[i];
    }
}

static double prior_cost(const std::vector<Component>& components, const NoiseModel& noise) {
    double cost=0;
    for (const Component& c:components) {
        if (c.type==0) {
            cost+=std::pow((c.p[3]-.95)/.25,2)
                +std::pow((c.p[4]-noise.spot_amplitude_mean)/noise.spot_amplitude_sd,2);
        } else if (c.type==1) {
            cost+=std::pow((c.p[6]-1.)/.5,2)+std::pow(c.p[7]/1.2,2);
        } else if (c.type==3 || c.type==4) {
            cost+=std::pow(((c.p[3]+c.p[4]+c.p[5])/3-.95)/.25,2);
            for (int d=0; d<3; ++d)
                cost+=std::pow((c.p[d+3]-c.p[(d+1)%3+3])/.7,2);
            cost+=std::pow((c.p[6]-noise.spot_amplitude_mean)/noise.spot_amplitude_sd,2);
        } else {
            cost+=std::pow((c.p[1]-.8)/.4,2)+std::pow(c.p[3]/.3,2)+std::pow(c.p[4]/.3,2);
        }
    }
    return cost;
}

static double model_cost(const std::vector<Component>& components,
                         const std::vector<float>& f, const float* y,
                         const std::vector<float>& weights, const NoiseModel& noise) {
    double cost=prior_cost(components,noise);
    for (size_t i=0; i<f.size(); ++i) {
        if (noise.exact) { cost+=2*noise.likelihood(i,f[i]); continue; }
        double mean,derivative,variance;
        noise.moments(f[i],mean,derivative,variance);
        double r=mean-y[i];
        cost+=r*r*weights[i];
    }
    return cost;
}

static Box intersection(const Box& a, const Box& b) {
    Box c;
    for (int d=0; d<3; ++d) { c.lo[d]=std::max(a.lo[d],b.lo[d]); c.hi[d]=std::min(a.hi[d],b.hi[d]); }
    return c;
}

static double dot_box(const std::vector<float>& a, const std::vector<float>& b,
                      const Box& box, const Convolution& H) {
    double total=0;
    for (int z=box.lo[0]; z<box.hi[0]; ++z)
        for (int y=box.lo[1]; y<box.hi[1]; ++y) {
            int left=(z*H.ny+y)*H.nx+box.lo[2], right=(z*H.ny+y)*H.nx+box.hi[2];
            double sum=0;
            #pragma omp simd reduction(+:sum)
            for (int i=left; i<right; ++i) sum+=double(a[i])*b[i];
            total+=sum;
        }
    return total;
}

static bool cholesky_solve(std::vector<double> matrix, const std::vector<double>& gradient,
                           std::vector<double>& delta, double damping) {
    int n=gradient.size();
    std::vector<double> L(n*n,0), temp(n);
    for (int i=0; i<n; ++i) matrix[i*n+i]+=damping*(matrix[i*n+i]+1e-5);
    for (int i=0; i<n; ++i) for (int j=0; j<=i; ++j) {
        double value=matrix[i*n+j];
        for (int k=0; k<j; ++k) value-=L[i*n+k]*L[j*n+k];
        if (i==j) {
            if (value<=0 || !std::isfinite(value)) return false;
            L[i*n+j]=std::sqrt(value);
        } else L[i*n+j]=value/L[j*n+j];
    }
    for (int i=0; i<n; ++i) {
        double value=-gradient[i];
        for (int j=0; j<i; ++j) value-=L[i*n+j]*temp[j];
        temp[i]=value/L[i*n+i];
    }
    delta.assign(n,0);
    for (int i=n-1; i>=0; --i) {
        double value=temp[i];
        for (int j=i+1; j<n; ++j) value-=L[j*n+i]*delta[j];
        delta[i]=value/L[i*n+i];
    }
    return true;
}

static void constrain(Component& c, const Convolution& H) {
    int dimensions[3]={H.nz,H.ny,H.nx};
    if (c.type==0 || c.type==3 || c.type==4) {
        for (int d=0; d<3; ++d) c.p[d]=std::clamp(c.p[d],-.5,dimensions[d]-.5);
        if (c.type==0) {
            c.p[3]=std::clamp(c.p[3],.5,1.8); c.p[4]=std::clamp(c.p[4],.00001,2.);
        } else {
            for (int d=3; d<6; ++d) c.p[d]=std::clamp(c.p[d],.4,2.2);
            c.p[6]=std::clamp(c.p[6],.00001,2.);
        }
    } else if (c.type==1) {
        for (int d=0; d<6; ++d) c.p[d]=std::clamp(c.p[d],-8.,dimensions[d%3]+7.);
        c.p[6]=std::clamp(c.p[6],.4,2.2); c.p[7]=std::clamp(c.p[7],.00001,2.);
    } else {
        c.p[0]=std::clamp(c.p[0],-.5,dimensions[c.axis]-.5);
        c.p[1]=std::clamp(c.p[1],.4,2.5); c.p[2]=std::clamp(c.p[2],.00001,2.);
        double sum=std::abs(c.p[3])+std::abs(c.p[4]);
        if (sum>.95*c.p[2]) {
            c.p[3]*=.95*c.p[2]/sum; c.p[4]*=.95*c.p[2]/sum;
        }
    }
}

static int fit_components(std::vector<Component>& components, Convolution& H,
                          const float* y, const float* initial, const NoiseModel& noise,
                          double deadline, int iterations) {
    if (components.empty()) return 0;
    int parameters=0;
    for (Component& c:components) {
        constrain(c,H); component_box(c,H);
        c.begin=parameters; parameters+=c.p.size();
    }
    if (parameters>128) return 0;
    std::vector<float> f(H.n), fnew(H.n), weights(H.n), residual(H.n), multiplier(H.n);
    H.apply(initial,f.data());
    for (int i=0; i<H.n; ++i) weights[i]=1/std::max(noise.alpha*f[i]+noise.sigma*noise.sigma,1e-16);
    double work=0, evaluation=.003;
    for (size_t a=0; a<components.size(); ++a) {
        evaluation+=components[a].type==1 ? .004 : .0006;
        for (size_t b=a; b<components.size(); ++b) {
            Box box=intersection(components[a].box,components[b].box);
            double voxels=1;
            for (int d=0; d<3; ++d) voxels*=std::max(0,box.hi[d]-box.lo[d]);
            work+=voxels*components[a].p.size()*components[b].p.size()*(a==b ? .5 : 1.);
        }
    }
    double damping=.003, last_duration=std::max(.008,evaluation+work*8e-10);
    int completed=0;
    for (int iteration=0; iteration<iterations; ++iteration) {
        double began=monotonic_seconds();
        if (began+std::max(.008,last_duration*1.5)>deadline) break;
        evaluate_model(components,H,f,true);
        if (iteration && iteration%5==0) {
            for (int i=0; i<H.n; ++i) {
                double mean,derivative,variance;
                noise.moments(f[i],mean,derivative,variance);
                weights[i]=1/std::max(variance,1e-16);
            }
            damping=.003;
        }
        for (int i=0; i<H.n; ++i) {
            if (noise.exact) {
                double score,information;
                noise.likelihood(i,f[i],&score,&information);
                multiplier[i]=std::sqrt(information);
                residual[i]=score/multiplier[i];
                continue;
            }
            double mean,derivative,variance;
            noise.moments(f[i],mean,derivative,variance);
            double root=std::sqrt(weights[i]);
            residual[i]=(mean-y[i])*root;
            multiplier[i]=derivative*root;
        }
        for (Component& c:components)
            for (auto& col:c.jac)
                for (int i=0; i<H.n; ++i) col[i]*=multiplier[i];
        std::vector<double> matrix(parameters*parameters,0), gradient(parameters,0);
        for (size_t ca=0; ca<components.size(); ++ca) {
            Component& a=components[ca];
            for (size_t p=0; p<a.p.size(); ++p)
                gradient[a.begin+p]=dot_box(a.jac[p],residual,a.box,H);
            for (size_t cb=ca; cb<components.size(); ++cb) {
                Component& b=components[cb];
                Box box=intersection(a.box,b.box);
                for (size_t p=0; p<a.p.size(); ++p) for (size_t q=0; q<b.p.size(); ++q) {
                    int i=a.begin+p, j=b.begin+q;
                    if (j<i) continue;
                    double value=dot_box(a.jac[p],b.jac[q],box,H);
                    matrix[i*parameters+j]=matrix[j*parameters+i]=value;
                }
            }
        }
        for (const Component& c:components) {
            auto prior=[&](int p, double mean, double deviation) {
                int i=c.begin+p;
                double precision=1/(deviation*deviation);
                gradient[i]+=(c.p[p]-mean)*precision;
                matrix[i*parameters+i]+=precision;
            };
            if (c.type==0) { prior(3,.95,.25); prior(4,noise.spot_amplitude_mean,noise.spot_amplitude_sd); }
            if (c.type==1) { prior(6,1.,.5); prior(7,0,1.2); }
            if (c.type==2) { prior(1,.8,.4); prior(3,0,.3); prior(4,0,.3); }
            if (c.type==3 || c.type==4) {
                double mean=(c.p[3]+c.p[4]+c.p[5])/3;
                for (int a=3; a<6; ++a) {
                    gradient[c.begin+a]+=(mean-.95)/(3*.25*.25);
                    for (int b=3; b<6; ++b)
                        matrix[(c.begin+a)*parameters+c.begin+b]+=1/(9*.25*.25);
                }
                for (int d=0; d<3; ++d) {
                    int a=c.begin+d+3, b=c.begin+(d+1)%3+3;
                    double precision=1/(.7*.7), difference=c.p[d+3]-c.p[(d+1)%3+3];
                    gradient[a]+=difference*precision; gradient[b]-=difference*precision;
                    matrix[a*parameters+a]+=precision; matrix[b*parameters+b]+=precision;
                    matrix[a*parameters+b]-=precision; matrix[b*parameters+a]-=precision;
                }
                prior(6,noise.spot_amplitude_mean,noise.spot_amplitude_sd);
            }
        }
        double oldcost=model_cost(components,f,y,weights,noise);
        std::vector<std::vector<double>> oldp;
        for (const Component& c:components) oldp.push_back(c.p);
        bool accepted=false;
        double relative=1;
        for (int retry=0; retry<7; ++retry) {
            if (monotonic_seconds()>deadline-.005) break;
            std::vector<double> delta;
            if (!cholesky_solve(matrix,gradient,delta,damping)) { damping*=10; continue; }
            double scale=1;
            for (const Component& c:components) for (size_t p=0; p<c.p.size(); ++p) {
                double maximum=2.;
                if ((c.type==0 && p==3) || (c.type==1 && p==6) || (c.type==2 && p==1)) maximum=.3;
                if ((c.type==0 && p==4) || (c.type==1 && p==7) || (c.type==2 && p>=2)) maximum=.4;
                if ((c.type==3 || c.type==4) && p>=3 && p<=5) maximum=.3;
                if ((c.type==3 || c.type==4) && p==6) maximum=.4;
                double v=std::abs(delta[c.begin+p]);
                if (v>maximum) scale=std::min(scale,maximum/v);
            }
            for (size_t k=0; k<components.size(); ++k) {
                Component& c=components[k];
                for (size_t p=0; p<c.p.size(); ++p) c.p[p]=oldp[k][p]+scale*delta[c.begin+p];
                constrain(c,H);
            }
            evaluate_model(components,H,fnew,false);
            double cost=model_cost(components,fnew,y,weights,noise);
            if (std::isfinite(cost) && cost<oldcost) {
                accepted=true; damping=std::max(1e-7,damping*.3);
                relative=(oldcost-cost)/std::max(oldcost,1.);
                break;
            }
            for (size_t k=0; k<components.size(); ++k) components[k].p=oldp[k];
            damping*=10;
        }
        if (!accepted) for (size_t k=0; k<components.size(); ++k) components[k].p=oldp[k];
        ++completed;
        last_duration=monotonic_seconds()-began;
        if (!accepted && iteration>=10) break;
        if (relative<1e-8 && iteration>=12 && iteration%5>=2) break;
    }
    evaluate_model(components,H,f,false);
    return completed;
}

static void prune_components(std::vector<Component>& components, Convolution& H,
                             const float* y, const NoiseModel& noise) {
    if (components.empty()) return;
    std::vector<float> f(H.n);
    evaluate_model(components,H,f,false);
    std::vector<unsigned char> keep(components.size(),1);
    for (size_t k=0; k<components.size(); ++k) {
        double gain=0;
        const Component& c=components[k];
        for (int z=c.box.lo[0]; z<c.box.hi[0]; ++z)
            for (int yy=c.box.lo[1]; yy<c.box.hi[1]; ++yy)
                for (int x=c.box.lo[2]; x<c.box.hi[2]; ++x) {
                    int i=(z*H.ny+yy)*H.nx+x;
                    double mean,d,v,mean0,d0,v0;
                    if (noise.exact) {
                        gain+=2*(noise.likelihood(i,std::max(0.f,f[i]-c.blurred[i]))-noise.likelihood(i,f[i]));
                        continue;
                    }
                    noise.moments(f[i],mean,d,v);
                    noise.moments(std::max(0.f,f[i]-c.blurred[i]),mean0,d0,v0);
                    gain+=((mean0-y[i])*(mean0-y[i])-(mean-y[i])*(mean-y[i]))/v;
                }
        if (gain<(c.type==0 || c.type==3 || c.type==4 ? 20. : 40.)) keep[k]=0;
    }
    for (size_t k=components.size(); k>0; --k)
        if (!keep[k-1]) components.erase(components.begin()+k-1);
}

static double matched_residual(std::vector<Component>& components, Convolution& H,
                               const float* y, const NoiseModel& noise,
                               std::vector<float>& correlation,
                               std::vector<float>& uncertainty) {
    std::vector<float> f(H.n), residual(H.n), variance(H.n), temp(H.n), temp2(H.n);
    evaluate_model(components,H,f,false);
    for (int i=0; i<H.n; ++i) {
        double mean,d,v;
        noise.moments(f[i],mean,d,v);
        residual[i]=y[i]-mean; variance[i]=v;
    }
    H.apply(residual.data(),correlation.data(),true);
    auto squared=[](const std::vector<float>& a) {
        std::vector<float> b=a; for (float& x:b) x*=x; return b;
    };
    H.axis(variance.data(),temp.data(),squared(H.az),0,true);
    H.axis(temp.data(),temp2.data(),squared(H.ay),1,true);
    H.axis(temp2.data(),uncertainty.data(),squared(H.ax),2,true);
    double error=0, expected=0;
    for (int i=0; i<H.n; ++i) {
        error+=double(correlation[i])*correlation[i];
        expected+=uncertainty[i];
    }
    return error/std::max(expected,1e-18);
}

static int add_residual_spots(std::vector<Component>& components, Convolution& H,
                             const float* y, const float* initial,
                             const NoiseModel& noise, int limit) {
    std::vector<float> correlation(H.n), variance(H.n), score(H.n);
    matched_residual(components,H,y,noise,correlation,variance);
    for (int i=0; i<H.n; ++i) score[i]=correlation[i]/std::sqrt(std::max(variance[i],1e-18f));
    std::vector<std::pair<float,int>> peaks;
    for (int i=0; i<H.n; ++i) if (score[i]>4.5) {
        Vec3 p=voxel_position(i,H); bool peak=true;
        for (int z=std::max(0,int(p[0])-2); z<std::min(H.nz,int(p[0])+3) && peak; ++z)
            for (int yy=std::max(0,int(p[1])-2); yy<std::min(H.ny,int(p[1])+3) && peak; ++yy)
                for (int x=std::max(0,int(p[2])-2); x<std::min(H.nx,int(p[2])+3); ++x)
                    if (score[(z*H.ny+yy)*H.nx+x]>score[i]) { peak=false; break; }
        if (peak) peaks.emplace_back(score[i],i);
    }
    std::sort(peaks.rbegin(),peaks.rend());
    int before=components.size();
    for (auto peak:peaks) {
        if (int(components.size())-before>=limit) break;
        add_spot_seed(components,voxel_position(peak.second,H),initial,H,2.6);
    }
    return components.size()-before;
}

static bool invert3(const double matrix[3][3], double inverse[3][3], double* determinant=nullptr) {
    double det=matrix[0][0]*(matrix[1][1]*matrix[2][2]-matrix[1][2]*matrix[2][1])
              -matrix[0][1]*(matrix[1][0]*matrix[2][2]-matrix[1][2]*matrix[2][0])
              +matrix[0][2]*(matrix[1][0]*matrix[2][1]-matrix[1][1]*matrix[2][0]);
    if (!(det>1e-20) || !std::isfinite(det)) return false;
    inverse[0][0]=(matrix[1][1]*matrix[2][2]-matrix[1][2]*matrix[2][1])/det;
    inverse[0][1]=(matrix[0][2]*matrix[2][1]-matrix[0][1]*matrix[2][2])/det;
    inverse[0][2]=(matrix[0][1]*matrix[1][2]-matrix[0][2]*matrix[1][1])/det;
    inverse[1][0]=(matrix[1][2]*matrix[2][0]-matrix[1][0]*matrix[2][2])/det;
    inverse[1][1]=(matrix[0][0]*matrix[2][2]-matrix[0][2]*matrix[2][0])/det;
    inverse[1][2]=(matrix[0][2]*matrix[1][0]-matrix[0][0]*matrix[1][2])/det;
    inverse[2][0]=(matrix[1][0]*matrix[2][1]-matrix[1][1]*matrix[2][0])/det;
    inverse[2][1]=(matrix[0][1]*matrix[2][0]-matrix[0][0]*matrix[2][1])/det;
    inverse[2][2]=(matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0])/det;
    if (determinant) *determinant=det;
    return true;
}

static int average_spots(std::vector<Component>& components, Convolution& H,
                         const NoiseModel& noise, std::vector<float>& average,
                         double deadline) {
    std::vector<float> f(H.n), others(H.n), replacement(H.n), information(H.n);
    evaluate_model(components,H,f,true);
    std::fill(average.begin(),average.end(),0.f);
    for (const Component& c:components)
        for (int i=0; i<H.n; ++i) average[i]+=c.image[i];
    for (int i=0; i<H.n; ++i) {
        double mean,d,v;
        noise.moments(f[i],mean,d,v);
        information[i]=d*d/v;
    }
    int completed=0;
    double last_duration=.025;
    for (const Component& source:components) {
        double began=monotonic_seconds();
        if (source.type!=0 || began+last_duration*1.3>deadline) continue;
        double precision[5][5]={};
        for (int z=source.box.lo[0]; z<source.box.hi[0]; ++z)
            for (int y=source.box.lo[1]; y<source.box.hi[1]; ++y)
                for (int x=source.box.lo[2]; x<source.box.hi[2]; ++x) {
                    int i=(z*H.ny+y)*H.nx+x;
                    for (int a=0; a<5; ++a) for (int b=0; b<5; ++b)
                        precision[a][b]+=double(source.jac[a][i])*source.jac[b][i]*information[i];
                }
        precision[3][3]+=16;
        precision[4][4]+=1/(noise.spot_amplitude_sd*noise.spot_amplitude_sd);
        double cc[3][3], covariance[3][3];
        for (int a=0; a<3; ++a) for (int b=0; b<3; ++b) cc[a][b]=precision[a][b];
        for (int a=0; a<3; ++a) cc[a][a]+=1e-8;
        if (!invert3(cc,covariance)) continue;
        double bs[3]={}, ba[3]={};
        for (int a=0; a<3; ++a) for (int b=0; b<3; ++b) {
            bs[a]+=covariance[a][b]*precision[b][3];
            ba[a]+=covariance[a][b]*precision[b][4];
        }
        double ss=precision[3][3], sa=precision[3][4], aa=precision[4][4];
        for (int a=0; a<3; ++a) {
            ss-=precision[3][a]*bs[a];
            sa-=precision[3][a]*ba[a];
            aa-=precision[4][a]*ba[a];
        }
        double deviation=std::clamp(std::sqrt(1/std::max(ss-sa*sa/std::max(aa,1e-8),1e-8)),.025,.35);
        double maxvariance=std::max({covariance[0][0],covariance[1][1],covariance[2][2]});
        double scale=std::min(1.,.25/std::max(maxvariance,1e-12));
        for (int a=0; a<3; ++a) for (int b=0; b<3; ++b) covariance[a][b]*=scale;
        for (int i=0; i<H.n; ++i) others[i]=std::max(0.f,f[i]-source.blurred[i]);
        const int nodes=9;
        double low=std::max(.5,source.p[3]-2.5*deviation);
        double high=std::min(1.8,source.p[3]+2.5*deviation);
        double widths[nodes], amplitudes[nodes], logweights[nodes], centres[nodes][3];
        Component trial(0,source.p);
        trial.p[3]=high;
        component_box(trial,H);
        Box box=trial.box;
        double prior_precision=1/(noise.spot_amplitude_sd*noise.spot_amplitude_sd);
        bool complete=true;
        for (int node=0; node<nodes; ++node) {
            if (monotonic_seconds()>deadline-.004) { complete=false; break; }
            double width=low+(high-low)*node/(nodes-1);
            double amplitude=std::clamp(source.p[4]*std::pow(source.p[3]/width,3),.001,2.);
            double final_information=1.;
            for (int round=0; round<2; ++round) {
                trial.p[3]=width; trial.p[4]=1.;
                for (int d=0; d<3; ++d)
                    trial.p[d]=source.p[d]-bs[d]*(width-source.p[3])-ba[d]*(amplitude-source.p[4]);
                evaluate_component(trial,H,false);
                for (int iteration=0; iteration<3; ++iteration) {
                    double gradient=(amplitude-noise.spot_amplitude_mean)*prior_precision;
                    double hessian=prior_precision;
                    for (int z=box.lo[0]; z<box.hi[0]; ++z)
                        for (int y=box.lo[1]; y<box.hi[1]; ++y)
                            for (int x=box.lo[2]; x<box.hi[2]; ++x) {
                                int i=(z*H.ny+y)*H.nx+x;
                                double basis=trial.blurred[i], score,info;
                                noise.likelihood(i,others[i]+amplitude*basis,&score,&info);
                                gradient+=basis*score; hessian+=basis*basis*info;
                            }
                    double delta=std::clamp(-gradient/std::max(hessian,1e-8),-.5,.5);
                    amplitude=std::clamp(amplitude+delta,.001,2.);
                    final_information=hessian;
                }
            }
            double cost=.5*std::pow((amplitude-noise.spot_amplitude_mean)/noise.spot_amplitude_sd,2)
                       +.5*std::pow((width-.95)/.25,2);
            for (int z=box.lo[0]; z<box.hi[0]; ++z)
                for (int y=box.lo[1]; y<box.hi[1]; ++y)
                    for (int x=box.lo[2]; x<box.hi[2]; ++x) {
                        int i=(z*H.ny+y)*H.nx+x;
                        cost+=noise.likelihood(i,others[i]+amplitude*trial.blurred[i]);
                    }
            widths[node]=width; amplitudes[node]=amplitude;
            logweights[node]=-cost-.5*std::log(std::max(final_information,1e-10))
                            +(node==0 || node==nodes-1 ? std::log(.5) : 0.);
            for (int d=0; d<3; ++d)
                centres[node][d]=source.p[d]-bs[d]*(width-source.p[3])-ba[d]*(amplitude-source.p[4]);
        }
        if (!complete) break;
        double maximum=*std::max_element(logweights,logweights+nodes), total=0;
        for (int node=0; node<nodes; ++node) total+=std::exp(logweights[node]-maximum);
        if (!(total>0) || !std::isfinite(total)) continue;
        std::fill(replacement.begin(),replacement.end(),0.f);
        for (int node=0; node<nodes; ++node) {
            double weight=std::exp(logweights[node]-maximum)/total, width=widths[node];
            double matrix[3][3], inverse[3][3], determinant;
            for (int a=0; a<3; ++a) for (int b=0; b<3; ++b)
                matrix[a][b]=covariance[a][b]+(a==b ? width*width : 0.);
            if (!invert3(matrix,inverse,&determinant)) continue;
            double amplitude=weight*amplitudes[node]*width*width*width/std::sqrt(determinant);
            for (int z=0; z<H.nz; ++z) for (int y=0; y<H.ny; ++y) {
                double rz=z-centres[node][0], ry=y-centres[node][1];
                double constant=rz*rz*inverse[0][0]+ry*ry*inverse[1][1]+2*rz*ry*inverse[0][1];
                double linear=2*(rz*inverse[0][2]+ry*inverse[1][2]);
                for (int x=0; x<H.nx; ++x) {
                    double rx=x-centres[node][2];
                    double q=constant+linear*rx+rx*rx*inverse[2][2];
                    replacement[(z*H.ny+y)*H.nx+x]+=amplitude*std::exp(-.5*q);
                }
            }
        }
        for (int i=0; i<H.n; ++i) average[i]+=replacement[i]-source.image[i];
        ++completed;
        last_duration=monotonic_seconds()-began;
    }
    return completed;
}

static int geometry_refine(Convolution& H, const float* y, float* output,
                           double deadline, int flags) {
    if (!H.separable || monotonic_seconds()>deadline-.075) return 0;
    std::vector<float> initial(output,output+H.n);
    double sigma=estimate_noise(y,H);
    if (!std::isfinite(sigma) || sigma<=0) return 0;
    NoiseModel noise(sigma,flags);
    std::vector<Component> components=sheet_seeds(H,initial.data());
    bool sheet=!components.empty();
    if (!sheet) {
        std::vector<float> remaining(H.n);
        components=tube_seeds(H,initial.data(),remaining,deadline);
    }
    bool spots=components.empty();
    if (spots && (flags&1024)) {
        noise.spot_amplitude_mean=.85;
        noise.spot_amplitude_sd=.35;
    }
    if (spots) components=spot_seeds(H,initial.data(),16,flags);
    if (components.empty()) return 0;
    if (spots && (flags&(32|64|128))) {
        for (Component& c:components) if (c.type==0 && c.elongated) {
            c.type=flags&(64|128) ? 4 : 3;
            double width=c.p[3], amplitude=c.p[4];
            c.p={c.p[0],c.p[1],c.p[2],width,width,width,amplitude};
        }
    }
    if (noise.exact || spots) noise.initialize_likelihood(y,H.n);
    double now=monotonic_seconds();
    double first_deadline=spots ? deadline-.015 : std::min(deadline-.015,now+.65*(deadline-now-.02));
    int iterations=fit_components(components,H,y,initial.data(),noise,first_deadline,
                                  spots ? 40 : 15);
    prune_components(components,H,y,noise);
    if (spots && (flags&128) && monotonic_seconds()<deadline-.05) {
        std::vector<Component> expanded;
        bool changed=false;
        for (const Component& c:components) {
            int longest=0;
            if (c.type==4) for (int j=1; j<3; ++j)
                if (c.p[j+3]>c.p[longest+3]) longest=j;
            int a=(longest+1)%3, b=(longest+2)%3;
            double radius=c.type==4 ? std::sqrt(c.p[a+3]*c.p[b+3]) : 1.;
            if (c.type==4 && c.p[longest+3]>1.7*radius
                && c.p[longest+3]>.3+std::max(c.p[a+3],c.p[b+3]) && expanded.size()<17) {
                double distance=std::sqrt(c.p[longest+3]*c.p[longest+3]-radius*radius);
                double amplitude=c.p[6]*c.p[3]*c.p[4]*c.p[5]/(2*radius*radius*radius);
                for (int sign=-1; sign<=1; sign+=2) {
                    std::vector<double> p={c.p[0],c.p[1],c.p[2],radius,amplitude};
                    for (int d=0; d<3; ++d) p[d]+=sign*distance*c.frame[d][longest];
                    expanded.emplace_back(0,p);
                }
                changed=true;
            } else expanded.push_back(c);
        }
        if (changed) {
            components.swap(expanded);
            iterations+=fit_components(components,H,y,initial.data(),noise,deadline-.020,20);
            prune_components(components,H,y,noise);
        }
    }
    if (monotonic_seconds()<deadline-.035) {
        int added=add_residual_spots(components,H,y,initial.data(),noise,spots ? 3 : 8);
        if (added) {
            iterations+=fit_components(components,H,y,initial.data(),noise,deadline-.015,25);
            prune_components(components,H,y,noise);
        }
    }
    if (components.empty()) return iterations;
    std::vector<float> averaged(H.n);
    bool use_average=noise.posterior && spots && monotonic_seconds()<deadline-.04
        && average_spots(components,H,noise,averaged,deadline-.015)>0;
    std::vector<float> correlation(H.n), uncertainty(H.n);
    double mismatch=matched_residual(components,H,y,noise,correlation,uncertainty);
    double fallback=std::clamp((mismatch-1.3)/2.,0.,.95);
    std::fill(output,output+H.n,0.f);
    if (use_average) {
        std::copy(averaged.begin(),averaged.end(),output);
    } else {
        for (const Component& c:components)
            for (int i=0; i<H.n; ++i) output[i]+=c.image[i];
    }
    for (int i=0; i<H.n; ++i) {
        output[i]=std::isfinite(output[i])
            ? std::max(0.f,float((1-fallback)*output[i]+fallback*initial[i])) : initial[i];
    }
    if (flags&1) {
        std::fprintf(stderr,"sigma %.6g alpha %.6g components %zu iterations %d mismatch %.4g fallback %.3g\n",
                     noise.sigma,noise.alpha,components.size(),iterations,mismatch,fallback);
        for (const Component& c:components) {
            std::fprintf(stderr,"type %d axis %d:",c.type,c.axis);
            for (double value:c.p) std::fprintf(stderr," %.5g",value);
            std::fprintf(stderr,"\n");
        }
    }
    return iterations;
}
