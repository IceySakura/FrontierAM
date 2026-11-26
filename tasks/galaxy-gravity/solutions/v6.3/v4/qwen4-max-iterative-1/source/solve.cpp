#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <chrono>
#include <immintrin.h>

extern "C" {
void openblas_set_num_threads(int);
void cblas_sgemm(int, int, int, int, int, int, float, const float*, int, const float*, int, float, float*, int);
}

static const double EPS = 0.01;
static const double EPS2 = 0.0001;
static const int MAXLEVEL = 20;

static inline double dot_r(const double* a, const double* b, int r) {
    __m256d s0=_mm256_setzero_pd(), s1=_mm256_setzero_pd();
    int i=0;
    for (; i+8<=r; i+=8) {
        s0=_mm256_fmadd_pd(_mm256_loadu_pd(a+i), _mm256_loadu_pd(b+i), s0);
        s1=_mm256_fmadd_pd(_mm256_loadu_pd(a+i+4), _mm256_loadu_pd(b+i+4), s1);
    }
    for (; i+4<=r; i+=4) s0=_mm256_fmadd_pd(_mm256_loadu_pd(a+i), _mm256_loadu_pd(b+i), s0);
    __m256d s=_mm256_add_pd(s0,s1);
    double tmp[4]; _mm256_storeu_pd(tmp,s);
    double res=tmp[0]+tmp[1]+tmp[2]+tmp[3];
    for (; i<r; i++) res+=a[i]*b[i];
    return res;
}

static inline void addMatVec(const double* D, const double* q, double* l, int r) {
    int a=0;
    for (; a+4<=r; a+=4) {
        __m256d s0=_mm256_setzero_pd(), s1=_mm256_setzero_pd(), s2=_mm256_setzero_pd(), s3=_mm256_setzero_pd();
        const double* d0=D+(size_t)a*r;
        const double* d1=d0+r;
        const double* d2=d1+r;
        const double* d3=d2+r;
        int b=0;
        for (; b+4<=r; b+=4) {
            __m256d qv=_mm256_loadu_pd(q+b);
            s0=_mm256_fmadd_pd(_mm256_loadu_pd(d0+b),qv,s0);
            s1=_mm256_fmadd_pd(_mm256_loadu_pd(d1+b),qv,s1);
            s2=_mm256_fmadd_pd(_mm256_loadu_pd(d2+b),qv,s2);
            s3=_mm256_fmadd_pd(_mm256_loadu_pd(d3+b),qv,s3);
        }
        double t0[4],t1[4],t2[4],t3[4];
        _mm256_storeu_pd(t0,s0); _mm256_storeu_pd(t1,s1); _mm256_storeu_pd(t2,s2); _mm256_storeu_pd(t3,s3);
        double r0=t0[0]+t0[1]+t0[2]+t0[3], r1=t1[0]+t1[1]+t1[2]+t1[3], r2=t2[0]+t2[1]+t2[2]+t2[3], r3=t3[0]+t3[1]+t3[2]+t3[3];
        for (; b<r; b++) { double qb=q[b]; r0+=d0[b]*qb; r1+=d1[b]*qb; r2+=d2[b]*qb; r3+=d3[b]*qb; }
        l[a]+=r0; l[a+1]+=r1; l[a+2]+=r2; l[a+3]+=r3;
    }
    for (; a<r; a++) l[a]+=dot_r(D+(size_t)a*r,q,r);
}

static void gemmBatch(const double* D, const double* QT, double* LT, int r, int B) {
    int a=0;
    for (; a+4<=r; a+=4) {
        const double* d0=D+(size_t)a*r;
        const double* d1=d0+r;
        const double* d2=d1+r;
        const double* d3=d2+r;
        int j=0;
        for (; j+4<=B; j+=4) {
            __m256d acc0=_mm256_setzero_pd(), acc1=_mm256_setzero_pd(), acc2=_mm256_setzero_pd(), acc3=_mm256_setzero_pd();
            int b=0;
            for (; b+4<=r; b+=4) {
                __m256d q0=_mm256_loadu_pd(QT+(size_t)b*B+j);
                __m256d q1=_mm256_loadu_pd(QT+(size_t)(b+1)*B+j);
                __m256d q2=_mm256_loadu_pd(QT+(size_t)(b+2)*B+j);
                __m256d q3=_mm256_loadu_pd(QT+(size_t)(b+3)*B+j);
                acc0=_mm256_fmadd_pd(_mm256_set1_pd(d0[b]),q0,acc0);
                acc1=_mm256_fmadd_pd(_mm256_set1_pd(d1[b]),q0,acc1);
                acc2=_mm256_fmadd_pd(_mm256_set1_pd(d2[b]),q0,acc2);
                acc3=_mm256_fmadd_pd(_mm256_set1_pd(d3[b]),q0,acc3);
                acc0=_mm256_fmadd_pd(_mm256_set1_pd(d0[b+1]),q1,acc0);
                acc1=_mm256_fmadd_pd(_mm256_set1_pd(d1[b+1]),q1,acc1);
                acc2=_mm256_fmadd_pd(_mm256_set1_pd(d2[b+1]),q1,acc2);
                acc3=_mm256_fmadd_pd(_mm256_set1_pd(d3[b+1]),q1,acc3);
                acc0=_mm256_fmadd_pd(_mm256_set1_pd(d0[b+2]),q2,acc0);
                acc1=_mm256_fmadd_pd(_mm256_set1_pd(d1[b+2]),q2,acc1);
                acc2=_mm256_fmadd_pd(_mm256_set1_pd(d2[b+2]),q2,acc2);
                acc3=_mm256_fmadd_pd(_mm256_set1_pd(d3[b+2]),q2,acc3);
                acc0=_mm256_fmadd_pd(_mm256_set1_pd(d0[b+3]),q3,acc0);
                acc1=_mm256_fmadd_pd(_mm256_set1_pd(d1[b+3]),q3,acc1);
                acc2=_mm256_fmadd_pd(_mm256_set1_pd(d2[b+3]),q3,acc2);
                acc3=_mm256_fmadd_pd(_mm256_set1_pd(d3[b+3]),q3,acc3);
            }
            for (; b<r; b++) {
                __m256d qv=_mm256_loadu_pd(QT+(size_t)b*B+j);
                acc0=_mm256_fmadd_pd(_mm256_set1_pd(d0[b]),qv,acc0);
                acc1=_mm256_fmadd_pd(_mm256_set1_pd(d1[b]),qv,acc1);
                acc2=_mm256_fmadd_pd(_mm256_set1_pd(d2[b]),qv,acc2);
                acc3=_mm256_fmadd_pd(_mm256_set1_pd(d3[b]),qv,acc3);
            }
            _mm256_storeu_pd(LT+(size_t)a*B+j, _mm256_add_pd(_mm256_loadu_pd(LT+(size_t)a*B+j),acc0));
            _mm256_storeu_pd(LT+(size_t)(a+1)*B+j, _mm256_add_pd(_mm256_loadu_pd(LT+(size_t)(a+1)*B+j),acc1));
            _mm256_storeu_pd(LT+(size_t)(a+2)*B+j, _mm256_add_pd(_mm256_loadu_pd(LT+(size_t)(a+2)*B+j),acc2));
            _mm256_storeu_pd(LT+(size_t)(a+3)*B+j, _mm256_add_pd(_mm256_loadu_pd(LT+(size_t)(a+3)*B+j),acc3));
        }
        for (; j<B; j++) {
            double s0=0,s1=0,s2=0,s3=0;
            for (int b=0;b<r;b++) {
                double q=QT[(size_t)b*B+j];
                s0+=d0[b]*q; s1+=d1[b]*q; s2+=d2[b]*q; s3+=d3[b]*q;
            }
            LT[(size_t)a*B+j]+=s0; LT[(size_t)(a+1)*B+j]+=s1; LT[(size_t)(a+2)*B+j]+=s2; LT[(size_t)(a+3)*B+j]+=s3;
        }
    }
    for (; a<r; a++) {
        const double* d0=D+(size_t)a*r;
        for (int j=0;j<B;j++) {
            double s=0;
            for (int b=0;b<r;b++) s+=d0[b]*QT[(size_t)b*B+j];
            LT[(size_t)a*B+j]+=s;
        }
    }
}

static void gemmBatchF(const float* D, const float* QT, float* LT, int r, int B) {
    int a=0;
    for (; a+8<=r; a+=8) {
        const float* d0=D+(size_t)a*r;
        const float* d1=d0+r;
        const float* d2=d1+r;
        const float* d3=d2+r;
        const float* d4=d3+r;
        const float* d5=d4+r;
        const float* d6=d5+r;
        const float* d7=d6+r;
        int j=0;
        for (; j+8<=B; j+=8) {
            __m256 acc0=_mm256_setzero_ps(), acc1=_mm256_setzero_ps(), acc2=_mm256_setzero_ps(), acc3=_mm256_setzero_ps();
            __m256 acc4=_mm256_setzero_ps(), acc5=_mm256_setzero_ps(), acc6=_mm256_setzero_ps(), acc7=_mm256_setzero_ps();
            int b=0;
            for (; b+4<=r; b+=4) {
                __m256 q0=_mm256_loadu_ps(QT+(size_t)b*B+j);
                __m256 q1=_mm256_loadu_ps(QT+(size_t)(b+1)*B+j);
                __m256 q2=_mm256_loadu_ps(QT+(size_t)(b+2)*B+j);
                __m256 q3=_mm256_loadu_ps(QT+(size_t)(b+3)*B+j);
                acc0=_mm256_fmadd_ps(_mm256_set1_ps(d0[b]),q0,acc0);
                acc1=_mm256_fmadd_ps(_mm256_set1_ps(d1[b]),q0,acc1);
                acc2=_mm256_fmadd_ps(_mm256_set1_ps(d2[b]),q0,acc2);
                acc3=_mm256_fmadd_ps(_mm256_set1_ps(d3[b]),q0,acc3);
                acc4=_mm256_fmadd_ps(_mm256_set1_ps(d4[b]),q0,acc4);
                acc5=_mm256_fmadd_ps(_mm256_set1_ps(d5[b]),q0,acc5);
                acc6=_mm256_fmadd_ps(_mm256_set1_ps(d6[b]),q0,acc6);
                acc7=_mm256_fmadd_ps(_mm256_set1_ps(d7[b]),q0,acc7);
                acc0=_mm256_fmadd_ps(_mm256_set1_ps(d0[b+1]),q1,acc0);
                acc1=_mm256_fmadd_ps(_mm256_set1_ps(d1[b+1]),q1,acc1);
                acc2=_mm256_fmadd_ps(_mm256_set1_ps(d2[b+1]),q1,acc2);
                acc3=_mm256_fmadd_ps(_mm256_set1_ps(d3[b+1]),q1,acc3);
                acc4=_mm256_fmadd_ps(_mm256_set1_ps(d4[b+1]),q1,acc4);
                acc5=_mm256_fmadd_ps(_mm256_set1_ps(d5[b+1]),q1,acc5);
                acc6=_mm256_fmadd_ps(_mm256_set1_ps(d6[b+1]),q1,acc6);
                acc7=_mm256_fmadd_ps(_mm256_set1_ps(d7[b+1]),q1,acc7);
                acc0=_mm256_fmadd_ps(_mm256_set1_ps(d0[b+2]),q2,acc0);
                acc1=_mm256_fmadd_ps(_mm256_set1_ps(d1[b+2]),q2,acc1);
                acc2=_mm256_fmadd_ps(_mm256_set1_ps(d2[b+2]),q2,acc2);
                acc3=_mm256_fmadd_ps(_mm256_set1_ps(d3[b+2]),q2,acc3);
                acc4=_mm256_fmadd_ps(_mm256_set1_ps(d4[b+2]),q2,acc4);
                acc5=_mm256_fmadd_ps(_mm256_set1_ps(d5[b+2]),q2,acc5);
                acc6=_mm256_fmadd_ps(_mm256_set1_ps(d6[b+2]),q2,acc6);
                acc7=_mm256_fmadd_ps(_mm256_set1_ps(d7[b+2]),q2,acc7);
                acc0=_mm256_fmadd_ps(_mm256_set1_ps(d0[b+3]),q3,acc0);
                acc1=_mm256_fmadd_ps(_mm256_set1_ps(d1[b+3]),q3,acc1);
                acc2=_mm256_fmadd_ps(_mm256_set1_ps(d2[b+3]),q3,acc2);
                acc3=_mm256_fmadd_ps(_mm256_set1_ps(d3[b+3]),q3,acc3);
                acc4=_mm256_fmadd_ps(_mm256_set1_ps(d4[b+3]),q3,acc4);
                acc5=_mm256_fmadd_ps(_mm256_set1_ps(d5[b+3]),q3,acc5);
                acc6=_mm256_fmadd_ps(_mm256_set1_ps(d6[b+3]),q3,acc6);
                acc7=_mm256_fmadd_ps(_mm256_set1_ps(d7[b+3]),q3,acc7);
            }
            for (; b<r; b++) {
                __m256 qv=_mm256_loadu_ps(QT+(size_t)b*B+j);
                acc0=_mm256_fmadd_ps(_mm256_set1_ps(d0[b]),qv,acc0);
                acc1=_mm256_fmadd_ps(_mm256_set1_ps(d1[b]),qv,acc1);
                acc2=_mm256_fmadd_ps(_mm256_set1_ps(d2[b]),qv,acc2);
                acc3=_mm256_fmadd_ps(_mm256_set1_ps(d3[b]),qv,acc3);
                acc4=_mm256_fmadd_ps(_mm256_set1_ps(d4[b]),qv,acc4);
                acc5=_mm256_fmadd_ps(_mm256_set1_ps(d5[b]),qv,acc5);
                acc6=_mm256_fmadd_ps(_mm256_set1_ps(d6[b]),qv,acc6);
                acc7=_mm256_fmadd_ps(_mm256_set1_ps(d7[b]),qv,acc7);
            }
            _mm256_storeu_ps(LT+(size_t)a*B+j, _mm256_add_ps(_mm256_loadu_ps(LT+(size_t)a*B+j),acc0));
            _mm256_storeu_ps(LT+(size_t)(a+1)*B+j, _mm256_add_ps(_mm256_loadu_ps(LT+(size_t)(a+1)*B+j),acc1));
            _mm256_storeu_ps(LT+(size_t)(a+2)*B+j, _mm256_add_ps(_mm256_loadu_ps(LT+(size_t)(a+2)*B+j),acc2));
            _mm256_storeu_ps(LT+(size_t)(a+3)*B+j, _mm256_add_ps(_mm256_loadu_ps(LT+(size_t)(a+3)*B+j),acc3));
            _mm256_storeu_ps(LT+(size_t)(a+4)*B+j, _mm256_add_ps(_mm256_loadu_ps(LT+(size_t)(a+4)*B+j),acc4));
            _mm256_storeu_ps(LT+(size_t)(a+5)*B+j, _mm256_add_ps(_mm256_loadu_ps(LT+(size_t)(a+5)*B+j),acc5));
            _mm256_storeu_ps(LT+(size_t)(a+6)*B+j, _mm256_add_ps(_mm256_loadu_ps(LT+(size_t)(a+6)*B+j),acc6));
            _mm256_storeu_ps(LT+(size_t)(a+7)*B+j, _mm256_add_ps(_mm256_loadu_ps(LT+(size_t)(a+7)*B+j),acc7));
        }
        for (; j<B; j++) {
            float s0=0,s1=0,s2=0,s3=0;
            float s4=0,s5=0,s6=0,s7=0;
            for (int b=0;b<r;b++) {
                float q=QT[(size_t)b*B+j];
                s0+=d0[b]*q; s1+=d1[b]*q; s2+=d2[b]*q; s3+=d3[b]*q;
                s4+=d4[b]*q; s5+=d5[b]*q; s6+=d6[b]*q; s7+=d7[b]*q;
            }
            LT[(size_t)a*B+j]+=s0; LT[(size_t)(a+1)*B+j]+=s1; LT[(size_t)(a+2)*B+j]+=s2; LT[(size_t)(a+3)*B+j]+=s3;
            LT[(size_t)(a+4)*B+j]+=s4; LT[(size_t)(a+5)*B+j]+=s5; LT[(size_t)(a+6)*B+j]+=s6; LT[(size_t)(a+7)*B+j]+=s7;
        }
    }
    for (; a<r; a++) {
        const float* d0=D+(size_t)a*r;
        for (int j=0;j<B;j++) {
            float s=0;
            for (int b=0;b<r;b++) s+=d0[b]*QT[(size_t)b*B+j];
            LT[(size_t)a*B+j]+=s;
        }
    }
}

struct Data {
    int p, r, n3, noff, npow;
    double epsMax;
    const double *nodes, *bary, *U, *V, *D, *W, *Lmat;
    const int8_t (*offs)[3];
    int offMap[343];
    void* map;
    size_t mapLen;
    bool load(const char* path) {
        int fd = open(path, O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        fstat(fd, &st);
        mapLen = st.st_size;
        map = mmap(nullptr, mapLen, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (map == MAP_FAILED) return false;
        const char* ptr = (const char*)map;
        if (memcmp(ptr, "FMM1", 4)) return false;
        ptr += 4;
        memcpy(&p, ptr, 4); ptr += 4;
        memcpy(&r, ptr, 4); ptr += 4;
        memcpy(&n3, ptr, 4); ptr += 4;
        memcpy(&noff, ptr, 4); ptr += 4;
        memcpy(&npow, ptr, 4); ptr += 4;
        memcpy(&epsMax, ptr, 8); ptr += 8;
        nodes = (const double*)ptr; ptr += p*8;
        bary = (const double*)ptr; ptr += p*8;
        U = (const double*)ptr; ptr += (size_t)n3*r*8;
        V = (const double*)ptr; ptr += (size_t)n3*r*8;
        D = (const double*)ptr; ptr += (size_t)npow*noff*r*r*8;
        W = (const double*)ptr; ptr += 2*p*p*8;
        Lmat = (const double*)ptr; ptr += 8*r*r*8;
        offs = (const int8_t(*)[3])ptr;
        std::fill(offMap, offMap+343, -1);
        for (int i=0;i<noff;i++) {
            int idx = (offs[i][0]+3)*49 + (offs[i][1]+3)*7 + (offs[i][2]+3);
            offMap[idx] = i;
        }
        return true;
    }
};

struct Node {
    double cx, cy, cz, h;
    int start, count, parent;
    int child[8];
    uint8_t level;
    bool leaf;
    double m0, comx, comy, comz;
    double qxx, qyy, qzz, qxy, qxz, qyz;
    double tr;
};

static inline uint64_t part1by2(uint32_t x) {
    uint64_t v = x & 0x1fffff;
    v = (v | (v << 32)) & 0x1f00000000ffffULL;
    v = (v | (v << 16)) & 0x1f0000ff0000ffULL;
    v = (v | (v << 8))  & 0x100f00f00f00f00fULL;
    v = (v | (v << 4))  & 0x10c30c30c30c30c3ULL;
    v = (v | (v << 2))  & 0x1249249249249249ULL;
    return v;
}

struct Solver {
    Data data;
    int N;
    double mass;
    int leafCap = 64;
    std::vector<double> x, y, z;
    std::vector<float> xf, yf, zf, axf, ayf, azf;
    std::vector<int> orig;
    std::vector<Node> nodes;
    std::vector<double> out, loc;
    std::vector<double> ax, ay, az;
    std::vector<std::pair<int,int>> nearPairs;
    struct M2LPair { int s, t, off; };
    std::vector<std::vector<M2LPair>> m2l;
    std::vector<double> levelDEff;
    std::vector<float> levelDEffF;
    std::vector<int> levelUsed;
    bool noM2L = false;
    bool noM2P = false;
    bool noNear = false;
    bool onlyNear = false;
    bool noL2L = false;
    double epsMaxRun;
    double sepFactor = 2.0;
    bool useP2L = false;
    bool noAxial = false;
    bool fastRsqrt = false;
    double bhTheta = 0.7;
    bool highMul = false;
    bool isDisc = false;

    bool init(const char* input, const char* dataPath) {
        if (!data.load(dataPath)) return false;
        FILE* f = fopen(input, "rb");
        if (!f) return false;
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        N = sz / 32;
        if (N <= 0 || sz % 32) return false;
        mass = 1.0 / N;
        if (const char* e=getenv("FMM_LEAF")) leafCap = atoi(e);
        std::vector<double> raw((size_t)N*4);
        if (fread(raw.data(), 8, raw.size(), f) != raw.size()) return false;
        fclose(f);
        {
            double mx=0,my=0,mz=0;
            for (int i=0;i<N;i++) { mx+=raw[4*i+1]; my+=raw[4*i+2]; mz+=raw[4*i+3]; }
            mx/=N; my/=N; mz/=N;
            double c[3][3]={{0,0,0},{0,0,0},{0,0,0}};
            for (int i=0;i<N;i++) {
                double dx=raw[4*i+1]-mx, dy=raw[4*i+2]-my, dz=raw[4*i+3]-mz;
                c[0][0]+=dx*dx; c[0][1]+=dx*dy; c[0][2]+=dx*dz;
                c[1][1]+=dy*dy; c[1][2]+=dy*dz; c[2][2]+=dz*dz;
            }
            c[1][0]=c[0][1]; c[2][0]=c[0][2]; c[2][1]=c[1][2];
            for (int it=0;it<12;it++) {
                int p=0,q=1; double maxv=std::abs(c[0][1]);
                if (std::abs(c[0][2])>maxv) {p=0;q=2;maxv=std::abs(c[0][2]);}
                if (std::abs(c[1][2])>maxv) {p=1;q=2;maxv=std::abs(c[1][2]);}
                if (maxv<1e-12) break;
                double theta=(c[q][q]-c[p][p])/(2*c[p][q]);
                double t=std::copysign(1.0,theta)/(std::abs(theta)+std::sqrt(theta*theta+1));
                double cs=1/std::sqrt(t*t+1), sn=t*cs;
                double cpp=c[p][p], cqq=c[q][q], cpq=c[p][q];
                c[p][p]=cs*cs*cpp-2*sn*cs*cpq+sn*sn*cqq;
                c[q][q]=sn*sn*cpp+2*sn*cs*cpq+cs*cs*cqq;
                c[p][q]=c[q][p]=0;
                for (int k=0;k<3;k++) if (k!=p&&k!=q) {
                    double ckp=c[k][p], ckq=c[k][q];
                    c[k][p]=c[p][k]=cs*ckp-sn*ckq;
                    c[k][q]=c[q][k]=sn*ckp+cs*ckq;
                }
            }
            double e0=c[0][0],e1=c[1][1],e2=c[2][2];
            double emin=std::min({e0,e1,e2}), emax=std::max({e0,e1,e2});
            isDisc = emin < 0.1*emax;
        }
        double minx=raw[1], maxx=raw[1], miny=raw[2], maxy=raw[2], minz=raw[3], maxz=raw[3];
        for (int i=0;i<N;i++) {
            double xx=raw[4*i+1], yy=raw[4*i+2], zz=raw[4*i+3];
            minx=std::min(minx,xx); maxx=std::max(maxx,xx);
            miny=std::min(miny,yy); maxy=std::max(maxy,yy);
            minz=std::min(minz,zz); maxz=std::max(maxz,zz);
        }
        double cx=(minx+maxx)*0.5, cy=(miny+maxy)*0.5, cz=(minz+maxz)*0.5;
        double h = std::max({maxx-minx, maxy-miny, maxz-minz}) * 0.5 * (1.0+1e-12);
        if (h <= 0) h = 1.0;
        std::vector<std::pair<uint64_t,int>> codes(N);
        double scale = 2097151.0 / (2.0*h);
        for (int i=0;i<N;i++) {
            uint32_t ix = (uint32_t)std::min(2097151.0, std::max(0.0, (raw[4*i+1]-cx+h)*scale));
            uint32_t iy = (uint32_t)std::min(2097151.0, std::max(0.0, (raw[4*i+2]-cy+h)*scale));
            uint32_t iz = (uint32_t)std::min(2097151.0, std::max(0.0, (raw[4*i+3]-cz+h)*scale));
            uint64_t code = (part1by2(ix)<<2) | (part1by2(iy)<<1) | part1by2(iz);
            codes[i] = {code, i};
        }
        std::sort(codes.begin(), codes.end());
        x.resize(N); y.resize(N); z.resize(N); orig.resize(N);
        for (int i=0;i<N;i++) {
            int j = codes[i].second;
            x[i]=raw[4*j+1]; y[i]=raw[4*j+2]; z[i]=raw[4*j+3]; orig[i]=j;
        }
        raw.clear(); raw.shrink_to_fit();
        xf.resize(N); yf.resize(N); zf.resize(N);
        axf.assign(N,0.0f); ayf.assign(N,0.0f); azf.assign(N,0.0f);
        for (int i=0;i<N;i++) { xf[i]=(float)x[i]; yf[i]=(float)y[i]; zf[i]=(float)z[i]; }
        std::vector<uint64_t> sortedCodes(N);
        for (int i=0;i<N;i++) sortedCodes[i]=codes[i].first;
        codes.clear(); codes.shrink_to_fit();
        nodes.reserve((size_t)N/leafCap*2 + 1024);
        build(sortedCodes, 0, N, 0, cx, cy, cz, h, -1);
        sortedCodes.clear(); sortedCodes.shrink_to_fit();
        size_t nn = nodes.size();
        out.assign(nn*data.r, 0.0);
        loc.assign(nn*data.r, 0.0);
        ax.assign(N, 0.0); ay.assign(N, 0.0); az.assign(N, 0.0);
        m2l.resize(MAXLEVEL+1);
        noM2L = getenv("FMM_NOM2L") != nullptr;
        noM2P = getenv("FMM_M2P") == nullptr;
        noNear = getenv("FMM_NONEAR") != nullptr;
        onlyNear = getenv("FMM_ONLYNEAR") != nullptr;
        noL2L = getenv("FMM_NOL2L") != nullptr;
        epsMaxRun = data.epsMax;
        if (const char* e=getenv("FMM_EPSMAX")) epsMaxRun = atof(e);
        if (const char* e=getenv("FMM_SEP")) sepFactor = atof(e);
        useP2L = getenv("FMM_P2L") != nullptr;
        noAxial = getenv("FMM_NOAXIAL") != nullptr;
        fastRsqrt = getenv("FMM_FASTRSQRT") != nullptr;
        if (const char* e=getenv("FMM_THETA")) bhTheta = atof(e);
        else {
            double lim = 2.0;
            if (const char* e=getenv("GALAXY_CPU_LIMIT_S")) lim = atof(e);
            double perM = lim / ((double)N / 1048576.0);
            bhTheta = perM > 30.0 ? 0.6 : 1.2;
        }
        return true;
    }

    int build(const std::vector<uint64_t>& codes, int start, int count, int level,
              double cx, double cy, double cz, double h, int parent) {
        int idx = nodes.size();
        Node nd;
        nd.cx=cx; nd.cy=cy; nd.cz=cz; nd.h=h;
        nd.start=start; nd.count=count; nd.parent=parent; nd.level=level;
        std::fill(nd.child, nd.child+8, -1);
        nd.leaf = (count <= leafCap || level >= MAXLEVEL);
        nodes.push_back(nd);
        if (!nd.leaf) {
            int shift = 3*(MAXLEVEL-level);
            int pos = start;
            for (int oct=0; oct<8; oct++) {
                int begin = pos;
                while (pos < start+count && ((codes[pos] >> shift) & 7) == (uint64_t)oct) pos++;
                if (pos > begin) {
                    double nh = h*0.5;
                    double ncx = cx + ((oct&4)?nh:-nh);
                    double ncy = cy + ((oct&2)?nh:-nh);
                    double ncz = cz + ((oct&1)?nh:-nh);
                    int ci = build(codes, begin, pos-begin, level+1, ncx, ncy, ncz, nh, idx);
                    nodes[idx].child[oct] = ci;
                }
            }
        }
        return idx;
    }

    inline void weights(double u, double* L) const {
        int p = data.p;
        double s = 0.0;
        for (int j=0;j<p;j++) {
            double d = u - data.nodes[j];
            if (std::abs(d) < 1e-14) {
                for (int k=0;k<p;k++) L[k]=0.0;
                L[j]=1.0;
                return;
            }
            L[j] = data.bary[j] / d;
            s += L[j];
        }
        double inv = 1.0/s;
        for (int j=0;j<p;j++) L[j]*=inv;
    }

    void p2m(int ni, double* ch) {
        const Node& nd = nodes[ni];
        int p = data.p;
        double wx[16], wy[16], wz[16];
        for (int i=nd.start;i<nd.start+nd.count;i++) {
            weights((x[i]-nd.cx)/nd.h, wx);
            weights((y[i]-nd.cy)/nd.h, wy);
            weights((z[i]-nd.cz)/nd.h, wz);
            for (int a=0;a<p;a++) {
                double mx = mass*wx[a];
                for (int b=0;b<p;b++) {
                    double mxy = mx*wy[b];
                    int base = (a*p+b)*p;
                    for (int c=0;c<p;c++) ch[base+c] += mxy*wz[c];
                }
            }
        }
    }

    void m2mAdd(double* parent, const double* child, int oct) {
        int p = data.p;
        double tmp1[16*16*16], tmp2[16*16*16];
        const double* Wx = data.W + ((oct>>2)&1)*p*p;
        const double* Wy = data.W + ((oct>>1)&1)*p*p;
        const double* Wz = data.W + (oct&1)*p*p;
            for (int a=0;a<p;a++) for (int b=0;b<p;b++) for (int k=0;k<p;k++) {
                double s=0;
                for (int c=0;c<p;c++) s += Wz[c*p+k]*child[(a*p+b)*p+c];
                tmp1[(a*p+b)*p+k]=s;
            }
            for (int a=0;a<p;a++) for (int j=0;j<p;j++) for (int k=0;k<p;k++) {
                double s=0;
                for (int b=0;b<p;b++) s += Wy[b*p+j]*tmp1[(a*p+b)*p+k];
                tmp2[(a*p+j)*p+k]=s;
            }
        for (int i=0;i<p;i++) for (int j=0;j<p;j++) for (int k=0;k<p;k++) {
            double s=0;
            for (int a=0;a<p;a++) s += Wx[a*p+i]*tmp2[(a*p+j)*p+k];
            parent[(i*p+j)*p+k]+=s;
        }
    }

    void projectOut(int ni, const double* ch) {
        double* o = &out[(size_t)ni*data.r];
        int r = data.r;
        for (int i=0;i<data.n3;i++) {
            double c = ch[i];
            const double* v = data.V + (size_t)i*r;
            __m256d cv=_mm256_set1_pd(c);
            int a=0;
            for (; a+4<=r; a+=4) {
                __m256d ov=_mm256_loadu_pd(o+a);
                ov=_mm256_fmadd_pd(_mm256_loadu_pd(v+a), cv, ov);
                _mm256_storeu_pd(o+a,ov);
            }
            for (; a<r; a++) o[a]+=v[a]*c;
        }
    }

    void upwardRec(int ni, double* ch) {
        const Node& nd = nodes[ni];
        if (nd.leaf) {
            std::fill(ch, ch+data.n3, 0.0);
            p2m(ni, ch);
        } else {
            std::fill(ch, ch+data.n3, 0.0);
            for (int oct=0;oct<8;oct++) {
                int ci = nd.child[oct];
                if (ci < 0) continue;
                double cch[1000];
                upwardRec(ci, cch);
                m2mAdd(ch, cch, oct);
            }
        }
        projectOut(ni, ch);
    }

    void upward() {
        double ch[1000];
        upwardRec(0, ch);
    }

    bool wellSep(const Node& a, const Node& b) const {
        double dx = std::abs(a.cx-b.cx), dy = std::abs(a.cy-b.cy), dz = std::abs(a.cz-b.cz);
        double gap = std::max({dx-a.h-b.h, dy-a.h-b.h, dz-a.h-b.h});
        return gap >= sepFactor*std::max(a.h,b.h) - 1e-12*std::max(a.h,b.h);
    }

    int offsetIndex(const Node& s, const Node& t) const {
        double inv = 0.5/s.h;
        int dx = (int)std::lround((t.cx-s.cx)*inv);
        int dy = (int)std::lround((t.cy-s.cy)*inv);
        int dz = (int)std::lround((t.cz-s.cz)*inv);
        if (dx<-3||dx>3||dy<-3||dy>3||dz<-3||dz>3) return -1;
        return data.offMap[(dx+3)*49+(dy+3)*7+(dz+3)];
    }

    void m2p(int s, int t) {
        const Node& ns = nodes[s];
        const Node& nt = nodes[t];
        int r = data.r, n3 = data.n3, p = data.p;
        std::vector<double> ch(n3);
        const double* o = &out[(size_t)s*r];
        for (int i=0;i<n3;i++) {
            const double* v = data.V + (size_t)i*r;
            ch[i]=dot_r(v,o,r);
        }
        if (getenv("FMM_CHECK")) {
            double sum=0; for (int i=0;i<n3;i++) sum+=ch[i];
            static double maxerr=0; double want=mass*ns.count;
            if (std::abs(sum-want)>maxerr) {maxerr=std::abs(sum-want); fprintf(stderr,"m2p charge sum err=%g want=%g got=%g node=%d\n",maxerr,want,sum,s);}
        }
        std::vector<double> nx(n3), ny(n3), nz(n3);
        for (int a=0;a<p;a++) for (int b=0;b<p;b++) for (int c=0;c<p;c++) {
            int idx=(a*p+b)*p+c;
            nx[idx]=ns.cx+ns.h*data.nodes[a];
            ny[idx]=ns.cy+ns.h*data.nodes[b];
            nz[idx]=ns.cz+ns.h*data.nodes[c];
        }
        for (int i=nt.start;i<nt.start+nt.count;i++) {
            double fx=0,fy=0,fz=0;
            for (int j=0;j<n3;j++) {
                double dx=nx[j]-x[i], dy=ny[j]-y[i], dz=nz[j]-z[i];
                double r2=dx*dx+dy*dy+dz*dz+EPS2;
                double inv=1.0/std::sqrt(r2);
                double f=ch[j]*inv/r2;
                fx+=f*dx; fy+=f*dy; fz+=f*dz;
            }
            ax[i]+=fx; ay[i]+=fy; az[i]+=fz;
        }
        if (getenv("FMM_CHECK")) {
            {
                double maxrel=0;
                for (int i=nt.start;i<nt.start+nt.count;i++) {
                    double ex=0,ey=0,ez=0;
                    for (int j=ns.start;j<ns.start+ns.count;j++) {
                        double dx=x[j]-x[i], dy=y[j]-y[i], dz=z[j]-z[i];
                        double r2=dx*dx+dy*dy+dz*dz+EPS2; double f=mass/std::sqrt(r2)/r2;
                        ex+=f*dx; ey+=f*dy; ez+=f*dz;
                    }
                    double axx=0,ayy=0,azz=0;
                    for (int j=0;j<n3;j++) {
                        double dx=nx[j]-x[i], dy=ny[j]-y[i], dz=nz[j]-z[i];
                        double r2=dx*dx+dy*dy+dz*dz+EPS2; double f=ch[j]/std::sqrt(r2)/r2;
                        axx+=f*dx; ayy+=f*dy; azz+=f*dz;
                    }
                    double rel=std::sqrt((axx-ex)*(axx-ex)+(ayy-ey)*(ayy-ey)+(azz-ez)*(azz-ez))/std::sqrt(ex*ex+ey*ey+ez*ez);
                    if (rel>maxrel) maxrel=rel;
                }
                if (maxrel>1e-3) fprintf(stderr,"m2p BAD s=%d t=%d count=%d/%d maxrel=%g\n",s,t,ns.count,nt.count,maxrel);
            }
        }
    }

    void p2l(int s, int t) {
        const Node& ns = nodes[s];
        const Node& nt = nodes[t];
        int n3 = data.n3, p = data.p, r = data.r;
        std::vector<double> val(n3,0.0);
        std::vector<double> nx(n3), ny(n3), nz(n3);
        for (int a=0;a<p;a++) for (int b=0;b<p;b++) for (int c=0;c<p;c++) {
            int idx=(a*p+b)*p+c;
            nx[idx]=nt.cx+nt.h*data.nodes[a];
            ny[idx]=nt.cy+nt.h*data.nodes[b];
            nz[idx]=nt.cz+nt.h*data.nodes[c];
        }
        for (int i=ns.start;i<ns.start+ns.count;i++) {
            for (int j=0;j<n3;j++) {
                double dx=nx[j]-x[i], dy=ny[j]-y[i], dz=nz[j]-z[i];
                val[j] += mass/std::sqrt(dx*dx+dy*dy+dz*dz+EPS2);
            }
        }
        double* l = &loc[(size_t)t*r];
        for (int i=0;i<n3;i++) {
            double v=val[i];
            const double* u=data.U+(size_t)i*r;
            __m256d vv=_mm256_set1_pd(v);
            int a=0;
            for (; a+4<=r; a+=4) {
                __m256d lv=_mm256_loadu_pd(l+a);
                lv=_mm256_fmadd_pd(_mm256_loadu_pd(u+a), vv, lv);
                _mm256_storeu_pd(l+a,lv);
            }
            for (; a<r; a++) l[a]+=u[a]*v;
        }
        if (getenv("FMM_CHECK")) {
            std::vector<double> coeff(r,0.0), val2(n3,0.0);
            for (int i=0;i<n3;i++) { const double* u=data.U+(size_t)i*r; for(int a=0;a<r;a++) coeff[a]+=u[a]*val[i]; }
            for (int i=0;i<n3;i++) { const double* u=data.U+(size_t)i*r; for(int a=0;a<r;a++) val2[i]+=u[a]*coeff[a]; }
            double num=0,den=0; for(int i=0;i<n3;i++){num+=(val2[i]-val[i])*(val2[i]-val[i]); den+=val[i]*val[i];}
            double rel=std::sqrt(num/den);
            if (rel>1e-3) fprintf(stderr,"p2l BAD s=%d t=%d rel=%g\n",s,t,rel);
        }
    }

    void interact(int a, int b) {
        if (a == b) {
            if (nodes[a].leaf) {
                nearPairs.push_back({a,a});
            } else {
                for (int i=0;i<8;i++) if (nodes[a].child[i]>=0)
                    for (int j=i;j<8;j++) if (nodes[a].child[j]>=0)
                        interact(nodes[a].child[i], nodes[a].child[j]);
            }
            return;
        }
        const Node& na = nodes[a];
        const Node& nb = nodes[b];
        if (wellSep(na, nb)) {
            if (onlyNear) return;
            double epsScaled = EPS / std::min(na.h, nb.h);
            if (!noM2L && na.level == nb.level && epsScaled <= epsMaxRun) {
                int off = offsetIndex(na, nb);
                if (off >= 0) {
                    int nz = (data.offs[off][0]!=0) + (data.offs[off][1]!=0) + (data.offs[off][2]!=0);
                    if (!noAxial || nz != 1) {
                        m2l[na.level].push_back({a,b,off});
                        return;
                    }
                }
            }
            if (na.leaf && nb.leaf) {
                nearPairs.push_back({a,b});
            } else if (na.leaf) {
                if (noM2P) splitA(b,a); else if (useP2L) p2l(a,b); else m2p(b,a);
            } else if (nb.leaf) {
                if (noM2P) splitA(a,b); else m2p(a,b);
            } else {
                if (na.level < nb.level) splitA(a,b);
                else splitA(b,a);
            }
            return;
        }
        if (na.leaf && nb.leaf) {
            nearPairs.push_back({a,b});
        } else if (na.leaf) {
            splitA(b,a);
        } else if (nb.leaf) {
            splitA(a,b);
        } else if (na.level < nb.level) {
            splitA(a,b);
        } else if (nb.level < na.level) {
            splitA(b,a);
        } else {
            splitA(a,b);
        }
    }

    void splitA(int a, int b) {
        for (int i=0;i<8;i++) if (nodes[a].child[i]>=0)
            interact(nodes[a].child[i], b);
    }

    void traverse() {
        interact(0,0);
    }

    void p2p(int a, int b) {
        const Node& na = nodes[a];
        const Node& nb = nodes[b];
        if ((highMul || getenv("FMM_DEEPMUL")) && a!=b && na.leaf && nb.leaf && (highMul || std::max(na.h,nb.h) < EPS/8.0)) {
            for (int i=na.start;i<na.start+na.count;i++) {
                double rx=x[i]-nb.comx, ry=y[i]-nb.comy, rz=z[i]-nb.comz;
                double s=rx*rx+ry*ry+rz*rz+EPS2;
                double inv=1.0/std::sqrt(s), s3=s/inv, s5=s3*s, s7=s5*s;
                double qx=nb.qxx*rx+nb.qxy*ry+nb.qxz*rz;
                double qy=nb.qxy*rx+nb.qyy*ry+nb.qyz*rz;
                double qz=nb.qxz*rx+nb.qyz*ry+nb.qzz*rz;
                double qrr=rx*qx+ry*qy+rz*qz;
                ax[i] += -nb.m0*rx/s3 + 3.0*qx/s5 - 7.5*rx*qrr/s7;
                ay[i] += -nb.m0*ry/s3 + 3.0*qy/s5 - 7.5*ry*qrr/s7;
                az[i] += -nb.m0*rz/s3 + 3.0*qz/s5 - 7.5*rz*qrr/s7;
            }
            for (int i=nb.start;i<nb.start+nb.count;i++) {
                double rx=x[i]-na.comx, ry=y[i]-na.comy, rz=z[i]-na.comz;
                double s=rx*rx+ry*ry+rz*rz+EPS2;
                double inv=1.0/std::sqrt(s), s3=s/inv, s5=s3*s, s7=s5*s;
                double qx=na.qxx*rx+na.qxy*ry+na.qxz*rz;
                double qy=na.qxy*rx+na.qyy*ry+na.qyz*rz;
                double qz=na.qxz*rx+na.qyz*ry+na.qzz*rz;
                double qrr=rx*qx+ry*qy+rz*qz;
                ax[i] += -na.m0*rx/s3 + 3.0*qx/s5 - 7.5*rx*qrr/s7;
                ay[i] += -na.m0*ry/s3 + 3.0*qy/s5 - 7.5*ry*qrr/s7;
                az[i] += -na.m0*rz/s3 + 3.0*qz/s5 - 7.5*rz*qrr/s7;
            }
            return;
        }
        if (a == b) {
            for (int i=na.start;i<na.start+na.count;i++) {
                double fx=0,fy=0,fz=0;
                for (int j=i+1;j<na.start+na.count;j++) {
                    double dx=x[j]-x[i], dy=y[j]-y[i], dz=z[j]-z[i];
                    double r2=dx*dx+dy*dy+dz*dz+EPS2;
                    double inv=1.0/std::sqrt(r2);
                    double f=mass*inv/r2;
                    double fxx=f*dx, fyy=f*dy, fzz=f*dz;
                    fx+=fxx; fy+=fyy; fz+=fzz;
                    ax[j]-=fxx; ay[j]-=fyy; az[j]-=fzz;
                }
                ax[i]+=fx; ay[i]+=fy; az[i]+=fz;
            }
        } else {
            for (int i=na.start;i<na.start+na.count;i++) {
                __m256 xi=_mm256_set1_ps(xf[i]), yi=_mm256_set1_ps(yf[i]), zi=_mm256_set1_ps(zf[i]);
                __m256 fxv=_mm256_setzero_ps(), fyv=_mm256_setzero_ps(), fzv=_mm256_setzero_ps();
                __m256 epsv=_mm256_set1_ps((float)EPS2), mv=_mm256_set1_ps((float)mass), one=_mm256_set1_ps(1.0f);
                int j=nb.start;
                for (; j+8<=nb.start+nb.count; j+=8) {
                    __m256 dx=_mm256_sub_ps(_mm256_loadu_ps(&xf[j]),xi);
                    __m256 dy=_mm256_sub_ps(_mm256_loadu_ps(&yf[j]),yi);
                    __m256 dz=_mm256_sub_ps(_mm256_loadu_ps(&zf[j]),zi);
                    __m256 r2=_mm256_add_ps(_mm256_mul_ps(dx,dx),_mm256_mul_ps(dy,dy));
                    r2=_mm256_add_ps(r2,_mm256_mul_ps(dz,dz));
                    r2=_mm256_add_ps(r2,epsv);
                    __m256 inv=_mm256_rsqrt_ps(r2);
                    if (!fastRsqrt) {
                        __m256 half=_mm256_set1_ps(0.5f), threehalfs=_mm256_set1_ps(1.5f);
                        inv=_mm256_mul_ps(inv,_mm256_sub_ps(threehalfs,_mm256_mul_ps(_mm256_mul_ps(half,r2),_mm256_mul_ps(inv,inv))));
                    }
                    __m256 inv2=_mm256_mul_ps(inv,inv);
                    __m256 f=_mm256_mul_ps(mv,_mm256_mul_ps(inv,inv2));
                    __m256 fxx=_mm256_mul_ps(f,dx), fyy=_mm256_mul_ps(f,dy), fzz=_mm256_mul_ps(f,dz);
                    fxv=_mm256_add_ps(fxv,fxx); fyv=_mm256_add_ps(fyv,fyy); fzv=_mm256_add_ps(fzv,fzz);
                    _mm256_storeu_ps(&axf[j],_mm256_sub_ps(_mm256_loadu_ps(&axf[j]),fxx));
                    _mm256_storeu_ps(&ayf[j],_mm256_sub_ps(_mm256_loadu_ps(&ayf[j]),fyy));
                    _mm256_storeu_ps(&azf[j],_mm256_sub_ps(_mm256_loadu_ps(&azf[j]),fzz));
                }
                float tmp[8];
                _mm256_storeu_ps(tmp,fxv); double fx=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
                _mm256_storeu_ps(tmp,fyv); double fy=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
                _mm256_storeu_ps(tmp,fzv); double fz=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
                for (; j<nb.start+nb.count; j++) {
                    double dx=x[j]-x[i], dy=y[j]-y[i], dz=z[j]-z[i];
                    double r2=dx*dx+dy*dy+dz*dz+EPS2;
                    double inv=1.0/std::sqrt(r2);
                    double f=mass*inv/r2;
                    double fxx=f*dx, fyy=f*dy, fzz=f*dz;
                    fx+=fxx; fy+=fyy; fz+=fzz;
                    ax[j]-=fxx; ay[j]-=fyy; az[j]-=fzz;
                }
                ax[i]+=fx; ay[i]+=fy; az[i]+=fz;
            }
        }
    }

    void processNear() {
        for (auto pr : nearPairs) p2p(pr.first, pr.second);
        for (int i=0;i<N;i++) { ax[i]+=axf[i]; ay[i]+=ayf[i]; az[i]+=azf[i]; }
    }

    void computeLevelD(int level) {
        double hlevel = nodes[0].h / std::pow(2.0, level);
        double e = EPS / hlevel;
        double scale = 1.0 / hlevel;
        double coeff[5];
        coeff[0]=scale;
        double e2=e*e;
        coeff[1]=-0.5*e2*scale;
        coeff[2]=0.375*e2*e2*scale;
        coeff[3]=-0.3125*e2*e2*e2*scale;
        coeff[4]=0.2734375*e2*e2*e2*e2*scale;
        size_t mat=(size_t)data.r*data.r;
        levelDEff.resize((size_t)data.noff*mat);
        for (int off=0;off<data.noff;off++) {
            double* dst=&levelDEff[(size_t)off*mat];
            const double* d0=data.D + ((size_t)0*data.noff+off)*mat;
            const double* d1=data.D + ((size_t)1*data.noff+off)*mat;
            const double* d2=data.D + ((size_t)2*data.noff+off)*mat;
            const double* d3=data.D + ((size_t)3*data.noff+off)*mat;
            const double* d4=data.D + ((size_t)4*data.noff+off)*mat;
            __m256d c1=_mm256_set1_pd(coeff[1]), c2=_mm256_set1_pd(coeff[2]), c3=_mm256_set1_pd(coeff[3]), c4=_mm256_set1_pd(coeff[4]);
            __m256d c0=_mm256_set1_pd(coeff[0]);
            size_t i=0;
            for (; i+4<=mat; i+=4) {
                __m256d v=_mm256_mul_pd(c0,_mm256_loadu_pd(d0+i));
                v=_mm256_fmadd_pd(c1,_mm256_loadu_pd(d1+i),v);
                v=_mm256_fmadd_pd(c2,_mm256_loadu_pd(d2+i),v);
                v=_mm256_fmadd_pd(c3,_mm256_loadu_pd(d3+i),v);
                v=_mm256_fmadd_pd(c4,_mm256_loadu_pd(d4+i),v);
                _mm256_storeu_pd(dst+i,v);
            }
            for (; i<mat; i++) dst[i]=coeff[0]*d0[i]+coeff[1]*d1[i]+coeff[2]*d2[i]+coeff[3]*d3[i]+coeff[4]*d4[i];
        }
    }

    void computeLevelDF(int level) {
        double hlevel = nodes[0].h / std::pow(2.0, level);
        double e = EPS / hlevel;
        double scale = 1.0 / hlevel;
        double coeff[5];
        coeff[0]=scale;
        double e2=e*e;
        coeff[1]=-0.5*e2*scale;
        coeff[2]=0.375*e2*e2*scale;
        coeff[3]=-0.3125*e2*e2*e2*scale;
        coeff[4]=0.2734375*e2*e2*e2*e2*scale;
        size_t mat=(size_t)data.r*data.r;
        levelDEffF.resize((size_t)data.noff*mat);
        __m256d c0=_mm256_set1_pd(coeff[0]), c1=_mm256_set1_pd(coeff[1]), c2=_mm256_set1_pd(coeff[2]), c3=_mm256_set1_pd(coeff[3]), c4=_mm256_set1_pd(coeff[4]);
        for (int off=0;off<data.noff;off++) {
            float* dst=&levelDEffF[(size_t)off*mat];
            const double* d0=data.D + ((size_t)0*data.noff+off)*mat;
            const double* d1=data.D + ((size_t)1*data.noff+off)*mat;
            const double* d2=data.D + ((size_t)2*data.noff+off)*mat;
            const double* d3=data.D + ((size_t)3*data.noff+off)*mat;
            const double* d4=data.D + ((size_t)4*data.noff+off)*mat;
            size_t i=0;
            for (; i+4<=mat; i+=4) {
                __m256d v=_mm256_mul_pd(c0,_mm256_loadu_pd(d0+i));
                v=_mm256_fmadd_pd(c1,_mm256_loadu_pd(d1+i),v);
                v=_mm256_fmadd_pd(c2,_mm256_loadu_pd(d2+i),v);
                v=_mm256_fmadd_pd(c3,_mm256_loadu_pd(d3+i),v);
                v=_mm256_fmadd_pd(c4,_mm256_loadu_pd(d4+i),v);
                _mm_storeu_ps(dst+i,_mm256_cvtpd_ps(v));
            }
            for (; i<mat; i++) dst[i]=(float)(coeff[0]*d0[i]+coeff[1]*d1[i]+coeff[2]*d2[i]+coeff[3]*d3[i]+coeff[4]*d4[i]);
        }
    }

    void m2lPair(const M2LPair& pr, const double* Deff) {
        int r=data.r;
        size_t mat=(size_t)r*r;
        const double* qs=&out[(size_t)pr.s*r];
        const double* qt=&out[(size_t)pr.t*r];
        double* lt=&loc[(size_t)pr.t*r];
        double* ls=&loc[(size_t)pr.s*r];
        const double* D=Deff+(size_t)pr.off*mat;
        addMatVec(D,qs,lt,r);
        int dx=-data.offs[pr.off][0], dy=-data.offs[pr.off][1], dz=-data.offs[pr.off][2];
        int rev=data.offMap[(dx+3)*49+(dy+3)*7+(dz+3)];
        const double* Dr=Deff+(size_t)rev*mat;
        addMatVec(Dr,qt,ls,r);
    }

    void processM2L() {
        for (int level=0;level<=MAXLEVEL;level++) {
            if (m2l[level].empty()) continue;
            computeLevelDF(level);
            std::vector<std::pair<int,int>> buckets[316];
            for (const auto& pr : m2l[level]) {
                buckets[pr.off].push_back({pr.s,pr.t});
                int dx=-data.offs[pr.off][0], dy=-data.offs[pr.off][1], dz=-data.offs[pr.off][2];
                int rev=data.offMap[(dx+3)*49+(dy+3)*7+(dz+3)];
                buckets[rev].push_back({pr.t,pr.s});
            }
            int r=data.r;
            for (int off=0;off<data.noff;off++) {
                auto& buck=buckets[off];
                if (buck.empty()) continue;
                int B=buck.size();
                std::vector<float> QT((size_t)r*B), LT((size_t)r*B,0.0f);
                for (int j=0;j<B;j++) {
                    const double* o=&out[(size_t)buck[j].first*r];
                    for (int a=0;a<r;a++) QT[(size_t)a*B+j]=(float)o[a];
                }
                cblas_sgemm(101,111,111,r,B,r,1.0f,levelDEffF.data()+(size_t)off*r*r,r,QT.data(),B,0.0f,LT.data(),B);
                for (int j=0;j<B;j++) {
                    double* l=&loc[(size_t)buck[j].second*r];
                    for (int a=0;a<r;a++) l[a]+=(double)LT[(size_t)a*B+j];
                }
            }
            m2l[level].clear(); m2l[level].shrink_to_fit();
        }
    }

    void downward() {
        int r=data.r;
        for (size_t i=0;i<nodes.size();i++) {
            const Node& nd=nodes[i];
            if (nd.leaf) continue;
            const double* lp=&loc[(size_t)i*r];
            for (int oct=0;oct<8;oct++) {
                int ci=nd.child[oct];
                if (ci<0) continue;
                const double* L=data.Lmat+(size_t)oct*r*r;
                double* lc=&loc[(size_t)ci*r];
                addMatVec(L,lp,lc,r);
            }
        }
    }

    void l2p(int ni) {
        const Node& nd=nodes[ni];
        int p=data.p, r=data.r, n3=data.n3;
        std::vector<double> val(n3);
        const double* l=&loc[(size_t)ni*r];
        for (int i=0;i<n3;i++) {
            const double* u=data.U+(size_t)i*r;
            val[i]=dot_r(u,l,r);
        }
        double wx[16],wy[16],wz[16],dx[16],dy[16],dz[16];
        const double h=1e-5;
        for (int i=nd.start;i<nd.start+nd.count;i++) {
            double ux=(x[i]-nd.cx)/nd.h, uy=(y[i]-nd.cy)/nd.h, uz=(z[i]-nd.cz)/nd.h;
            weights(ux,wx); weights(uy,wy); weights(uz,wz);
            double tp[16], tm[16];
            weights(ux+h,tp); weights(ux-h,tm); for(int j=0;j<p;j++) dx[j]=(tp[j]-tm[j])/(2*h);
            weights(uy+h,tp); weights(uy-h,tm); for(int j=0;j<p;j++) dy[j]=(tp[j]-tm[j])/(2*h);
            weights(uz+h,tp); weights(uz-h,tm); for(int j=0;j<p;j++) dz[j]=(tp[j]-tm[j])/(2*h);
            double fx=0,fy=0,fz=0;
            for (int a=0;a<p;a++) for (int b=0;b<p;b++) {
                int base=(a*p+b)*p;
                double wxy=wx[a]*wy[b], dxy=dx[a]*wy[b], xdy=wx[a]*dy[b];
                for (int c=0;c<p;c++) {
                    double v=val[base+c];
                    fx += dxy*wz[c]*v;
                    fy += xdy*wz[c]*v;
                    fz += wxy*dz[c]*v;
                }
            }
            ax[i]+=fx/nd.h; ay[i]+=fy/nd.h; az[i]+=fz/nd.h;
        }
    }

    void processL2P() {
        for (size_t i=0;i<nodes.size();i++) if (nodes[i].leaf) l2p(i);
    }

    void computeMoments() {
        for (int i=(int)nodes.size()-1;i>=0;i--) {
            Node& nd=nodes[i];
            double m0=0,mx=0,my=0,mz=0,xx=0,yy=0,zz=0,xy=0,xz=0,yz=0;
            if (nd.leaf) {
                for (int j=nd.start;j<nd.start+nd.count;j++) {
                    m0+=mass; mx+=mass*x[j]; my+=mass*y[j]; mz+=mass*z[j];
                    xx+=mass*x[j]*x[j]; yy+=mass*y[j]*y[j]; zz+=mass*z[j]*z[j];
                    xy+=mass*x[j]*y[j]; xz+=mass*x[j]*z[j]; yz+=mass*y[j]*z[j];
                }
            } else {
                for (int oct=0;oct<8;oct++) {
                    int ci=nd.child[oct]; if (ci<0) continue;
                    const Node& c=nodes[ci];
                    m0+=c.m0; mx+=c.m0*c.comx; my+=c.m0*c.comy; mz+=c.m0*c.comz;
                    double cx=c.comx, cy=c.comy, cz=c.comz, m=c.m0;
                    xx+=c.qxx + m*cx*cx + c.tr/3.0;
                    yy+=c.qyy + m*cy*cy + c.tr/3.0;
                    zz+=c.qzz + m*cz*cz + c.tr/3.0;
                    xy+=c.qxy + m*cx*cy;
                    xz+=c.qxz + m*cx*cz;
                    yz+=c.qyz + m*cy*cz;
                }
            }
            nd.m0=m0;
            nd.comx=mx/m0; nd.comy=my/m0; nd.comz=mz/m0;
            double cx=nd.comx, cy=nd.comy, cz=nd.comz;
            double cxx=xx-m0*cx*cx, cyy=yy-m0*cy*cy, czz=zz-m0*cz*cz;
            double cxy=xy-m0*cx*cy, cxz=xz-m0*cx*cz, cyz=yz-m0*cy*cz;
            double tr=(cxx+cyy+czz)/3.0;
            nd.qxx=cxx-tr; nd.qyy=cyy-tr; nd.qzz=czz-tr;
            nd.qxy=cxy; nd.qxz=cxz; nd.qyz=cyz;
            nd.tr=cxx+cyy+czz;
        }
    }

    void bhParticle(int i) {
        const double theta=bhTheta, theta2=theta*theta;
        int stack[4096], sp=0; stack[sp++]=0;
        double fx=0,fy=0,fz=0;
        while (sp) {
            int ni=stack[--sp];
            const Node& nd=nodes[ni];
            double rx=x[i]-nd.comx, ry=y[i]-nd.comy, rz=z[i]-nd.comz;
            double s=rx*rx+ry*ry+rz*rz+EPS2;
            bool contains = (i>=nd.start && i<nd.start+nd.count);
            if (!nd.leaf && !contains && nd.h*nd.h < theta2*s) {
                if (getenv("FMM_BHDEBUG") && orig[i]==43652) fprintf(stderr,"accept ni=%d m0=%g com=%g %g %g R=%g %g %g s=%g h=%g\n",ni,nd.m0,nd.comx,nd.comy,nd.comz,rx,ry,rz,s,nd.h);
                double inv=1.0/std::sqrt(s);
                double s3=s/inv;
                double m3=-nd.m0/s3;
                if (getenv("FMM_BHDEBUG") && orig[i]==43652 && std::abs(m3*rx)>1e-4) fprintf(stderr,"BIG ni=%d m0=%g R=%g %g %g s=%g\n",ni,nd.m0,rx,ry,rz,s);
                double qx=nd.qxx*rx+nd.qxy*ry+nd.qxz*rz;
                double qy=nd.qxy*rx+nd.qyy*ry+nd.qyz*rz;
                double qz=nd.qxz*rx+nd.qyz*ry+nd.qzz*rz;
                double qrr=rx*qx+ry*qy+rz*qz;
                double s5=s3*s, s7=s5*s;
                if (getenv("FMM_BHMONO")) {
                    fx += m3*rx; fy += m3*ry; fz += m3*rz;
                } else {
                    fx += m3*rx + 3.0*qx/s5 - 7.5*rx*qrr/s7;
                    fy += m3*ry + 3.0*qy/s5 - 7.5*ry*qrr/s7;
                    fz += m3*rz + 3.0*qz/s5 - 7.5*rz*qrr/s7;
                }
            } else if (nd.leaf) {
                if (getenv("FMM_BHDEBUG") && orig[i]==43652) fprintf(stderr,"leaf ni=%d start=%d count=%d i=%d h=%g com=%g %g %g pos=%g %g %g\n",ni,nd.start,nd.count,i,nd.h,nd.comx,nd.comy,nd.comz,x[i],y[i],z[i]);
                __m256 xi=_mm256_set1_ps(xf[i]), yi=_mm256_set1_ps(yf[i]), zi=_mm256_set1_ps(zf[i]);
                __m256 fxv=_mm256_setzero_ps(), fyv=_mm256_setzero_ps(), fzv=_mm256_setzero_ps();
                __m256 epsv=_mm256_set1_ps((float)EPS2), mv=_mm256_set1_ps((float)mass);
                int j=nd.start;
                for (; j+8<=nd.start+nd.count; j+=8) {
                    __m256 dx=_mm256_sub_ps(_mm256_loadu_ps(&xf[j]),xi);
                    __m256 dy=_mm256_sub_ps(_mm256_loadu_ps(&yf[j]),yi);
                    __m256 dz=_mm256_sub_ps(_mm256_loadu_ps(&zf[j]),zi);
                    __m256 r2=_mm256_add_ps(_mm256_mul_ps(dx,dx),_mm256_mul_ps(dy,dy));
                    r2=_mm256_add_ps(r2,_mm256_mul_ps(dz,dz));
                    r2=_mm256_add_ps(r2,epsv);
                    __m256 inv=_mm256_rsqrt_ps(r2);
                    __m256 f=_mm256_mul_ps(mv,_mm256_mul_ps(inv,_mm256_mul_ps(inv,inv)));
                    fxv=_mm256_add_ps(fxv,_mm256_mul_ps(f,dx));
                    fyv=_mm256_add_ps(fyv,_mm256_mul_ps(f,dy));
                    fzv=_mm256_add_ps(fzv,_mm256_mul_ps(f,dz));
                }
                float tmp[8];
                _mm256_storeu_ps(tmp,fxv); fx+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
                _mm256_storeu_ps(tmp,fyv); fy+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
                _mm256_storeu_ps(tmp,fzv); fz+=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
                for (; j<nd.start+nd.count; j++) {
                    if (j==i) continue;
                    double dx=x[j]-x[i], dy=y[j]-y[i], dz=z[j]-z[i];
                    double r2=dx*dx+dy*dy+dz*dz+EPS2;
                    double inv=1.0/std::sqrt(r2);
                    double f=mass*inv/r2;
                    fx+=f*dx; fy+=f*dy; fz+=f*dz;
                }
            } else {
                for (int oct=0;oct<8;oct++) if (nd.child[oct]>=0) stack[sp++]=nd.child[oct];
            }
        }
        ax[i]+=fx; ay[i]+=fy; az[i]+=fz;
        if (getenv("FMM_BHDEBUG") && orig[i]==43652) fprintf(stderr,"total fx=%g fy=%g fz=%g\n",fx,fy,fz);
    }

    void bhAll() {
        for (int i=0;i<N;i++) bhParticle(i);
    }

    bool write(const char* output) {
        std::vector<double> buf((size_t)N*3);
        for (int i=0;i<N;i++) {
            int j=orig[i];
            buf[3*j]=ax[i]; buf[3*j+1]=ay[i]; buf[3*j+2]=az[i];
        }
        FILE* f=fopen(output,"wb");
        if (!f) return false;
        size_t wr=fwrite(buf.data(),8,buf.size(),f);
        fclose(f);
        return wr==buf.size();
    }
};

int main(int argc, char** argv) {
    if (argc < 3) return 1;
    openblas_set_num_threads(1);
    std::string dataPath = "fmm_data.bin";
    char exe[4096];
    ssize_t len = readlink("/proc/self/exe", exe, sizeof(exe)-1);
    if (len > 0) {
        exe[len] = 0;
        std::string dir(exe);
        size_t slash = dir.find_last_of('/');
        if (slash != std::string::npos) dataPath = dir.substr(0, slash+1) + "fmm_data.bin";
    }
    if (const char* env = getenv("FMM_DATA")) dataPath = env;
    else {
        struct stat st;
        long long n = 0;
        if (stat(argv[1], &st) == 0) n = st.st_size / 32;
        double limit = 2.0;
        if (const char* e = getenv("GALAXY_CPU_LIMIT_S")) limit = atof(e);
        struct Variant { const char* file; double b65, b262, b1m, b4m; };
        Variant variants[] = {
            {"fmm_p10_r220.bin", 2.1, 8.5, 45.0, 1e9},
            {"fmm_p8_r140.bin", 1.0, 3.8, 41.0, 85.0},
            {"fmm_p8_r100.bin", 0.8, 3.2, 12.0, 70.0},
        };
        std::string dir;
        size_t slash = dataPath.find_last_of('/');
        if (slash != std::string::npos) dir = dataPath.substr(0, slash+1);
        dataPath = dir + "fmm_p8_r100.bin";
        for (const auto& v : variants) {
            double est = n <= 65536 ? v.b65 : n <= 262144 ? v.b262 : n <= 1048576 ? v.b1m : v.b4m;
            if (est <= 0.90 * limit) {
                dataPath = dir + v.file;
                break;
            }
        }
        if (getenv("FMM_DEBUG")) fprintf(stderr,"selected %s N=%lld limit=%g\n",dataPath.c_str(),n,limit);
    }
    Solver s;
    if (!s.init(argv[1], dataPath.c_str())) return 2;
    auto t0=std::chrono::steady_clock::now();
    s.upward();
    auto t1=std::chrono::steady_clock::now();
    s.traverse();
    auto t2=std::chrono::steady_clock::now();
    long double directPairs=0;
    for (auto pr:s.nearPairs) {
        long long ca=s.nodes[pr.first].count, cb=s.nodes[pr.second].count;
        directPairs += pr.first==pr.second ? ca*(ca-1)/2 : ca*cb;
    }
    if (getenv("FMM_HIGHMUL")) s.highMul = true;
    double lim = 2.0;
    if (const char* e=getenv("GALAXY_CPU_LIMIT_S")) lim = atof(e);
    double perM = lim / ((double)s.N / 1048576.0);
    bool discFallback = s.isDisc && s.N >= 1048576 && perM < 30.0;
    bool needsFallback = (directPairs > 8000.0L*s.N) || discFallback;
    if (needsFallback) s.fastRsqrt = true;
    if (getenv("FMM_BH") || (needsFallback && perM < 30.0)) {
        s.computeMoments();
        s.bhAll();
        if (!s.write(argv[2])) return 3;
        return 0;
    }
    if (getenv("FMM_DEBUG")) {
        size_t nm2l=0; for (auto& v:s.m2l) nm2l+=v.size();
        fprintf(stderr,"nodes=%zu near=%zu m2l=%zu\n",s.nodes.size(),s.nearPairs.size(),nm2l);
        long double pairs=0;
        for (auto pr:s.nearPairs) {
            long long ca=s.nodes[pr.first].count, cb=s.nodes[pr.second].count;
            pairs += pr.first==pr.second ? ca*(ca-1)/2 : ca*cb;
        }
        fprintf(stderr,"direct pairs=%.0Lf expected=%.0Lf\n",pairs,(long double)s.N*(s.N-1)/2);
        long long leafsum=0; int leaves=0; for (auto& nd:s.nodes) if (nd.leaf) {leafsum+=nd.count; leaves++;}
        fprintf(stderr,"leaves=%d leafsum=%lld N=%d\n",leaves,leafsum,s.N);
    }
    if (s.highMul || getenv("FMM_DEEPMUL")) s.computeMoments();
    if (!s.noNear) s.processNear();
    auto t3=std::chrono::steady_clock::now();
    if (const char* dump=getenv("FMM_DUMP")) {
        FILE* f=fopen(dump,"wb");
        int nn=s.nodes.size(); fwrite(&nn,4,1,f);
        for (auto& nd:s.nodes) { fwrite(&nd.cx,8,1,f); fwrite(&nd.cy,8,1,f); fwrite(&nd.cz,8,1,f); fwrite(&nd.h,8,1,f); fwrite(&nd.level,1,1,f); fwrite(&nd.leaf,1,1,f); fwrite(&nd.start,4,1,f); fwrite(&nd.count,4,1,f); }
        fwrite(s.out.data(),8,s.out.size(),f);
        for (int level=0;level<=MAXLEVEL;level++) { int sz=s.m2l[level].size(); fwrite(&sz,4,1,f); if(sz) fwrite(s.m2l[level].data(),sizeof(Solver::M2LPair),sz,f); }
        fclose(f);
    }
    s.processM2L();
    auto t4=std::chrono::steady_clock::now();
    if (!s.noL2L) s.downward();
    s.processL2P();
    auto t5=std::chrono::steady_clock::now();
    if (!s.write(argv[2])) return 3;
    if (getenv("FMM_TIME")) {
        auto ms=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};
        fprintf(stderr,"up %.0f trav %.0f near %.0f m2l %.0f l2p %.0f\n",ms(t0,t1),ms(t1,t2),ms(t2,t3),ms(t3,t4),ms(t4,t5));
    }
    return 0;
}
