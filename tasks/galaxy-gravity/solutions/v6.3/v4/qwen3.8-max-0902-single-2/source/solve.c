/* Gravitational acceleration for galaxy snapshots (softened, G=1, eps=0.01).
 *
 * Flat Morton-ordered octree Barnes-Hut code with high-order Cartesian
 * multipole moments (order P) and AVX2-vectorised direct summation for the
 * near field.  The opening angle is chosen at run time from the CPU budget
 * advertised in GALAXY_CPU_LIMIT_S.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#define MAXP      5        /* highest multipole order supported        */
#define LEAF_MAX  16
#define NBITS     16       /* Morton bits per axis                     */
#define MAXDEPTH  NBITS
#define EPS2      1.0e-4
#define NMON      96       /* >= (P+2)(P+3)(P+4)/6 for P<=5 (=84)      */

static int    P;
static int    NT, ND;

static double cpu_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* ---------------- multi-index tables ---------------- */
static int EX[NMON][3], IDXI[MAXP+2][MAXP+2][MAXP+2], ORD[NMON], LASTK[NMON];
static double INVFACT[NMON], MOMW[NMON];
static int DIDX[NMON][3];
typedef struct { unsigned char b; unsigned char n; double c; } Term;
static Term DTERM[NMON][64];
static int  DNT[NMON];

static void build_tables(int p) {
    int lim = p + 1;
    memset(IDXI, -1, sizeof(IDXI));
    int cnt = 0;
    for (int deg = 0; deg <= lim; deg++)
        for (int a = 0; a <= deg; a++)
            for (int b = 0; b <= deg - a; b++) {
                int c = deg - a - b;
                EX[cnt][0]=a; EX[cnt][1]=b; EX[cnt][2]=c; ORD[cnt]=deg;
                double f = 1.0;
                for (int k=2;k<=a;k++) f*=k;
                for (int k=2;k<=b;k++) f*=k;
                for (int k=2;k<=c;k++) f*=k;
                INVFACT[cnt] = 1.0/f;
                MOMW[cnt] = (deg&1 ? -1.0 : 1.0) * INVFACT[cnt];
                IDXI[a][b][c] = cnt;
                LASTK[cnt] = (c>0?2:(b>0?1:(a>0?0:-1)));
                cnt++;
            }
    ND = cnt;
    NT = 0; for (int i=0;i<ND;i++) if (ORD[i] <= p) NT++;
    for (int i=0;i<ND;i++)
        for (int k=0;k<3;k++) {
            int a=EX[i][0], b=EX[i][1], c=EX[i][2];
            if (k==0) a++; else if (k==1) b++; else c++;
            DIDX[i][k] = IDXI[a][b][c];
        }
}

static void build_deriv_terms(int p) {
    static double cf[NMON][NMON][MAXP+3];
    int lim = p+1;
    memset(cf,0,sizeof(cf));
    cf[0][0][0] = 1.0;
    for (int j=1;j<ND;j++) {
        int k = LASTK[j];
        int a=EX[j][0], b=EX[j][1], c=EX[j][2];
        int pj = (k==0)?IDXI[a-1][b][c]:(k==1)?IDXI[a][b-1][c]:IDXI[a][b][c-1];
        for (int bb=0;bb<ND;bb++) if (ORD[bb] <= ORD[j]-1)
            for (int n=0;n<=lim;n++) {
                double v = cf[pj][bb][n];
                if (v == 0.0) continue;
                int e = EX[bb][k];
                if (e > 0) {
                    int b2 = (k==0)?IDXI[EX[bb][0]-1][EX[bb][1]][EX[bb][2]]
                            :(k==1)?IDXI[EX[bb][0]][EX[bb][1]-1][EX[bb][2]]
                            :       IDXI[EX[bb][0]][EX[bb][1]][EX[bb][2]-1];
                    cf[j][b2][n] += v*e;
                }
                int nb[3]={EX[bb][0],EX[bb][1],EX[bb][2]}; nb[k]++;
                if (nb[0]<=lim && nb[1]<=lim && nb[2]<=lim && n+1<=lim)
                    cf[j][IDXI[nb[0]][nb[1]][nb[2]]][n+1] += -v*(2*n+1);
            }
    }
    for (int j=0;j<ND;j++) {
        int t=0;
        for (int bb=0;bb<ND;bb++) for (int n=0;n<=lim;n++)
            if (cf[j][bb][n]!=0.0 && t<64) { DTERM[j][t].b=(unsigned char)bb; DTERM[j][t].n=(unsigned char)n; DTERM[j][t].c=cf[j][bb][n]; t++; }
        DNT[j]=t;
    }
}

/* moment shift pair list: Dst[A] += C * Src[B] * d^(A-B) */
static int SHA, SHB, SHD, SH_N;
static int SH_A[4096], SH_B[4096], SH_D[4096]; static double SH_C[4096];
static void build_shift(int p) {
    double bct[MAXP+2][MAXP+2];
    for (int n=0;n<=p+1;n++){ bct[n][0]=1.0; for(int k=1;k<=n;k++) bct[n][k]=bct[n-1][k-1]*n/k; }
    SH_N=0;
    for (int a=0;a<NT;a++) for (int b=0;b<NT;b++) {
        int da=EX[a][0]-EX[b][0], db=EX[a][1]-EX[b][1], dc=EX[a][2]-EX[b][2];
        if (da<0||db<0||dc<0) continue;
        SH_A[SH_N]=a; SH_B[SH_N]=b; SH_D[SH_N]=IDXI[da][db][dc];
        SH_C[SH_N]=bct[EX[a][0]][da]*bct[EX[a][1]][db]*bct[EX[a][2]][dc];
        SH_N++;
    }
}

/* ---------------- data ---------------- */
static int N;
static double *PX,*PY,*PZ,*PM;
static unsigned long long *KEY;
static unsigned *PERM;
static double *OUT;
static int NN,NNMAX;
static double *Nmom,*Ncom;
static float  *Nrmax;
static int    *Nchild,*Nstart,*Nend;
static int    *STACK;
static double THETA;
static long long CNT_LEAF,CNT_NODE,CNT_PUSH;
static double *montmp;

static inline __m256d vrsqrt(__m256d x) {
    __m256d t = _mm256_cvtps_pd(_mm256_castps256_ps128(
                    _mm256_rsqrt_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(x)))));
    const __m256d h15=_mm256_set1_pd(1.5), h05=_mm256_set1_pd(0.5);
    for (int k=0;k<3;k++) {
        __m256d xt=_mm256_mul_pd(x,t);
        t=_mm256_mul_pd(t,_mm256_sub_pd(h15,_mm256_mul_pd(h05,_mm256_mul_pd(xt,t))));
    }
    return t;
}

static inline void leaf_force(int i,int s,int e,double *acc) {
    __m256d xi=_mm256_set1_pd(PX[i]), yi=_mm256_set1_pd(PY[i]), zi=_mm256_set1_pd(PZ[i]);
    __m256d e2=_mm256_set1_pd(EPS2);
    __m256d ax=_mm256_setzero_pd(), ay=_mm256_setzero_pd(), az=_mm256_setzero_pd();
    int e4 = s + ((e - s) & ~3);
    for (int j=s;j<e4;j+=4) {
        __m256d dx=_mm256_sub_pd(_mm256_loadu_pd(PX+j),xi);
        __m256d dy=_mm256_sub_pd(_mm256_loadu_pd(PY+j),yi);
        __m256d dz=_mm256_sub_pd(_mm256_loadu_pd(PZ+j),zi);
        __m256d r2=_mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(dx,dx),_mm256_mul_pd(dy,dy)),
                                 _mm256_add_pd(_mm256_mul_pd(dz,dz),e2));
        __m256d ir=vrsqrt(r2);
        __m256d f=_mm256_mul_pd(_mm256_loadu_pd(PM+j),_mm256_mul_pd(_mm256_mul_pd(ir,ir),ir));
        ax=_mm256_add_pd(ax,_mm256_mul_pd(f,dx));
        ay=_mm256_add_pd(ay,_mm256_mul_pd(f,dy));
        az=_mm256_add_pd(az,_mm256_mul_pd(f,dz));
    }
    double t[12];
    _mm256_storeu_pd(t,ax); _mm256_storeu_pd(t+4,ay); _mm256_storeu_pd(t+8,az);
    acc[0]+=t[0]+t[1]+t[2]+t[3];
    acc[1]+=t[4]+t[5]+t[6]+t[7];
    acc[2]+=t[8]+t[9]+t[10]+t[11];
    for (int j=e4;j<e;j++) {
        double dx=PX[j]-PX[i], dy=PY[j]-PY[i], dz=PZ[j]-PZ[i];
        double r2=dx*dx+dy*dy+dz*dz+EPS2;
        double ir=1.0/sqrt(r2), f=PM[j]*ir*ir*ir;
        acc[0]+=f*dx; acc[1]+=f*dy; acc[2]+=f*dz;
    }
}

static inline void node_force(int i,int n,double *acc) {
    const double *cx = Ncom + 3*(size_t)n;
    double ux=PX[i]-cx[0], uy=PY[i]-cx[1], uz=PZ[i]-cx[2];
    double t = ux*ux+uy*uy+uz*uz+EPS2;
    double g0=1.0/sqrt(t), q=g0*g0, g[MAXP+3];
    g[0]=g0; for (int k=1;k<=P+1;k++) g[k]=g[k-1]*q;
    double mon[NMON]; mon[0]=1.0;
    for (int b=1;b<ND;b++) {
        int k=LASTK[b], a=EX[b][0], bb=EX[b][1], c=EX[b][2];
        int pj=(k==0)?IDXI[a-1][bb][c]:(k==1)?IDXI[a][bb-1][c]:IDXI[a][bb][c-1];
        mon[b]=mon[pj]*(k==0?ux:(k==1?uy:uz));
    }
    double D[NMON];
    for (int j=0;j<ND;j++) {
        const Term *tm=DTERM[j]; int nt=DNT[j]; double s=0.0;
        for (int k=0;k<nt;k++) s += tm[k].c*mon[tm[k].b]*g[tm[k].n];
        D[j]=s;
    }
    const double *M=Nmom+(size_t)n*NT;
    double s0=0,s1=0,s2=0;
    for (int a=0;a<NT;a++) {
        double c=MOMW[a]*M[a];
        s0+=c*D[DIDX[a][0]]; s1+=c*D[DIDX[a][1]]; s2+=c*D[DIDX[a][2]];
    }
    acc[0]+=s0; acc[1]+=s1; acc[2]+=s2;
}

static void particle_force(int i,double *acc) {
    acc[0]=acc[1]=acc[2]=0.0;
    int sp=0; STACK[sp++]=0;
    double th2=THETA*THETA;
    while (sp>0) {
        int n=STACK[--sp];
        int s=Nstart[n], e=Nend[n];
        if (s>=e) continue;
        int ch=Nchild[n];
        if (ch<0) { CNT_LEAF++; leaf_force(i,s,e,acc); continue; }
        if (!(s<=i && i<e)) {
            const double *cx=Ncom+3*(size_t)n;
            double ux=PX[i]-cx[0], uy=PY[i]-cx[1], uz=PZ[i]-cx[2];
            double r2=ux*ux+uy*uy+uz*uz;
            double rm=Nrmax[n];
            if (r2*th2 > rm*rm) { CNT_NODE++; node_force(i,n,acc); continue; }
        }
        STACK[sp]=ch; STACK[sp+1]=ch+1; STACK[sp+2]=ch+2; STACK[sp+3]=ch+3;
        STACK[sp+4]=ch+4; STACK[sp+5]=ch+5; STACK[sp+6]=ch+6; STACK[sp+7]=ch+7;
        sp+=8; CNT_PUSH++;
    }
}

/* ---------------- tree build ---------------- */
static void leaf_moments(int n,int s,int e) {
    double m=0,cx=0,cy=0,cz=0;
    for (int j=s;j<e;j++){ m+=PM[j]; cx+=PM[j]*PX[j]; cy+=PM[j]*PY[j]; cz+=PM[j]*PZ[j]; }
    if (m<=0){ cx=(e>s)?PX[s]:0; cy=(e>s)?PY[s]:0; cz=(e>s)?PZ[s]:0; m=0; }
    else { cx/=m; cy/=m; cz/=m; }
    double *C=Ncom+3*(size_t)n; C[0]=cx; C[1]=cy; C[2]=cz;
    double *M=Nmom+(size_t)n*NT;
    for (int a=0;a<NT;a++) M[a]=0.0;
    M[0]=m;
    double rmax=0;
    for (int j=s;j<e;j++){
        double dx=PX[j]-cx, dy=PY[j]-cy, dz=PZ[j]-cz;
        double rr=dx*dx+dy*dy+dz*dz; if(rr>rmax) rmax=rr;
        montmp[0]=1.0;
        for (int b=1;b<NT;b++){
            int k=LASTK[b], a2=EX[b][0],b2=EX[b][1],c2=EX[b][2];
            int pj=(k==0)?IDXI[a2-1][b2][c2]:(k==1)?IDXI[a2][b2-1][c2]:IDXI[a2][b2][c2-1];
            montmp[b]=montmp[pj]*(k==0?dx:(k==1?dy:dz));
        }
        double mj=PM[j];
        for (int b=1;b<NT;b++) M[b]+=mj*montmp[b];
    }
    Nrmax[n]=(float)sqrt(rmax);
    Nstart[n]=s; Nend[n]=e; Nchild[n]=-1;
}

static double dpow[NMON];
static inline void fill_dpow(double dx,double dy,double dz){
    double pw[3][MAXP+2];
    pw[0][0]=pw[1][0]=pw[2][0]=1.0;
    for(int k=1;k<=P+1;k++){pw[0][k]=pw[0][k-1]*dx;pw[1][k]=pw[1][k-1]*dy;pw[2][k]=pw[2][k-1]*dz;}
    for(int a=0;a<NT;a++){const int*e=EX[a]; dpow[a]=pw[0][e[0]]*pw[1][e[1]]*pw[2][e[2]];}
}
static inline void shift_accum(double *Dst,const double *Src){
    for(int k=0;k<SH_N;k++) Dst[SH_A[k]] += SH_C[k]*Src[SH_B[k]]*dpow[SH_D[k]];
}

static void build(int slot,int s,int e,int depth,double cx,double cy,double cz,double h) {
    Nchild[slot]=-1; Nstart[slot]=s; Nend[slot]=e; Nrmax[slot]=0.f;
    if (e<=s){ Nstart[slot]=Nend[slot]=0; return; }
    if ((e-s)<=LEAF_MAX || depth>=MAXDEPTH+40 || NN+8>NNMAX) { leaf_moments(slot,s,e); return; }
    int base=NN; NN+=8;
    for(int k=0;k<8;k++){ Nchild[base+k]=-1; Nstart[base+k]=Nend[base+k]=0; Nrmax[base+k]=0.f; }
    Nchild[slot]=base;
    if (depth<MAXDEPTH) {
        int shift=3*(NBITS-1-depth);
        int bs[9]; int c=0; bs[0]=s;
        for(int pos=s;pos<e;pos++){ int cc=(int)((KEY[pos]>>shift)&7ULL); while(c<cc){c++;bs[c]=pos;} }
        while(c<8){c++;bs[c]=e;}
        double hh=0.5*h;
        for(int k=0;k<8;k++){
            int ss=bs[k], ee=bs[k+1];
            if(ee>ss) build(base+k,ss,ee,depth+1,
                            cx+((k&1)?hh:-hh), cy+((k&2)?hh:-hh), cz+((k&4)?hh:-hh), hh);
            else { Nstart[base+k]=Nend[base+k]=0; }
        }
    } else {
        int mid=(s+e)/2;
        build(base+0,s,mid,depth+1,cx,cy,cz,h*0.5);
        build(base+1,mid,e,depth+1,cx,cy,cz,h*0.5);
        for(int k=2;k<8;k++) Nstart[base+k]=Nend[base+k]=0;
    }
    double m=0,cx2=0,cy2=0,cz2=0;
    for(int k=0;k<8;k++){ int cn=base+k; if(Nend[cn]<=Nstart[cn])continue;
        double mc=Nmom[(size_t)cn*NT]; const double *C=Ncom+3*(size_t)cn;
        m+=mc; cx2+=mc*C[0]; cy2+=mc*C[1]; cz2+=mc*C[2]; }
    if(m<=0){ m=1e-300; cx2=cx; cy2=cy; cz2=cz; } else { cx2/=m; cy2/=m; cz2/=m; }
    double *C=Ncom+3*(size_t)slot; C[0]=cx2; C[1]=cy2; C[2]=cz2;
    double *M=Nmom+(size_t)slot*NT;
    for(int a=0;a<NT;a++) M[a]=0.0;
    double rmax=0;
    for(int k=0;k<8;k++){ int cn=base+k; if(Nend[cn]<=Nstart[cn])continue;
        const double *Cc=Ncom+3*(size_t)cn;
        double dx=Cc[0]-cx2, dy=Cc[1]-cy2, dz=Cc[2]-cz2;
        double rr=sqrt(dx*dx+dy*dy+dz*dz)+Nrmax[cn]; if(rr>rmax)rmax=rr;
        fill_dpow(dx,dy,dz);
        shift_accum(M,Nmom+(size_t)cn*NT);
    }
    Nrmax[slot]=(float)rmax;
}

/* ---------------- Morton ---------------- */
static unsigned long long SPTAB[16];
static void init_spread(void){
    for(int n=0;n<16;n++){ unsigned long long r=0; for(int k=0;k<4;k++) if((n>>k)&1) r|=1ULL<<(3*k); SPTAB[n]=r; }
}
static inline unsigned long long spread(unsigned v){
    return SPTAB[v&15] | (SPTAB[(v>>4)&15]<<12) | (SPTAB[(v>>8)&15]<<24) | (SPTAB[(v>>12)&15]<<36);
}
static void *xmalloc(size_t n){void*p=malloc(n);if(!p){fprintf(stderr,"oom\n");exit(1);}return p;}

/* ---------------- main ---------------- */
int main(int argc,char**argv){
    double t0=cpu_now();
    if(argc<3){fprintf(stderr,"usage: solve IN OUT\n");return 1;}
    double budget=1.0;
    {const char*s=getenv("GALAXY_CPU_LIMIT_S"); if(s)budget=atof(s); if(!(budget>0))budget=1.0;}
    P=2; {const char*s=getenv("GALAXY_P"); if(s){int v=atoi(s); if(v>=2&&v<=MAXP)P=v;}}
    {const char*s=getenv("GALAXY_PMAX"); if(s){int v=atoi(s); if(v>=2&&v<=MAXP&&v<P)P=v;}}
    double th_lo=0.25, th_hi=0.60, target_frac=0.75;
    {const char*s=getenv("GALAXY_THETA"); if(s){th_lo=th_hi=atof(s); target_frac=1e9;}}
    {const char*s=getenv("GALAXY_TLO"); if(s)th_lo=atof(s);}
    {const char*s=getenv("GALAXY_THI"); if(s)th_hi=atof(s);}
    {const char*s=getenv("GALAXY_FRAC"); if(s)target_frac=atof(s);}

    init_spread();
    build_tables(P); build_deriv_terms(P); build_shift(P);
    montmp=(double*)xmalloc(NT*sizeof(double));

    /* read */
    FILE*f=fopen(argv[1],"rb");
    if(!f){fprintf(stderr,"no input\n");return 1;}
    fseek(f,0,SEEK_END); long fsz=ftell(f); fseek(f,0,SEEK_SET);
    N=(int)(fsz/32); if(N<1){fclose(f);return 1;}
    double *raw=(double*)xmalloc((size_t)N*4*sizeof(double));
    N=(int)fread(raw,32,(size_t)N,f); fclose(f);

    double mn[3]={1e300,1e300,1e300},mx[3]={-1e300,-1e300,-1e300};
    for(int i=0;i<N;i++){const double*r=raw+4*(size_t)i;
        for(int k=0;k<3;k++){double v=r[1+k]; if(!isfinite(v))v=0; if(v<mn[k])mn[k]=v; if(v>mx[k])mx[k]=v;}}
    double L=0,o[3];
    for(int k=0;k<3;k++){double w=mx[k]-mn[k]; if(w>L)L=w; o[k]=0.5*(mn[k]+mx[k]);}
    if(!(L>0))L=1.0;
    L*=1.0000002;
    double inv=(double)(1u<<NBITS)/L;

    KEY=(unsigned long long*)xmalloc((size_t)N*sizeof(unsigned long long));
    PERM=(unsigned*)xmalloc((size_t)N*sizeof(unsigned));
    for(int i=0;i<N;i++){const double*r=raw+4*(size_t)i;
        unsigned ix=(unsigned)((r[1]-o[0]+0.5*L)*inv); if(ix>=(1u<<NBITS))ix=(1u<<NBITS)-1;
        unsigned iy=(unsigned)((r[2]-o[1]+0.5*L)*inv); if(iy>=(1u<<NBITS))iy=(1u<<NBITS)-1;
        unsigned iz=(unsigned)((r[3]-o[2]+0.5*L)*inv); if(iz>=(1u<<NBITS))iz=(1u<<NBITS)-1;
        KEY[i]=spread(ix)|(spread(iy)<<1)|(spread(iz)<<2);
        PERM[i]=(unsigned)i;}

    /* radix sort */
    {
        unsigned long long *k2=(unsigned long long*)xmalloc((size_t)N*sizeof(unsigned long long));
        unsigned *p2=(unsigned*)xmalloc((size_t)N*sizeof(unsigned));
        const int nb=8, passes=(NBITS*3+nb-1)/nb;
        int *cnt=(int*)xmalloc((1<<nb)*sizeof(int));
        unsigned long long *ks=KEY,*kd=k2; unsigned *ps=PERM,*pd=p2;
        for(int pass=0;pass<passes;pass++){
            int sh=pass*nb;
            memset(cnt,0,(1<<nb)*sizeof(int));
            for(int i=0;i<N;i++) cnt[(ks[i]>>sh)&255]++;
            int sum=0; for(int b=0;b<256;b++){int c=cnt[b];cnt[b]=sum;sum+=c;}
            for(int i=0;i<N;i++){int b=(ks[i]>>sh)&255;int d=cnt[b]++;kd[d]=ks[i];pd[d]=ps[i];}
        {unsigned long long*t2; t2=ks;ks=kd;kd=t2;} {unsigned*t; t=ps;ps=pd;pd=t;}
        }
        if(ks!=KEY){memcpy(KEY,ks,(size_t)N*sizeof(unsigned long long));memcpy(PERM,ps,(size_t)N*sizeof(unsigned));}
        free(k2);free(p2);free(cnt);
    }

    PX=(double*)xmalloc(((size_t)N+8)*sizeof(double));
    PY=(double*)xmalloc(((size_t)N+8)*sizeof(double));
    PZ=(double*)xmalloc(((size_t)N+8)*sizeof(double));
    PM=(double*)xmalloc(((size_t)N+8)*sizeof(double));
    for(int i=0;i<N;i++){unsigned src=PERM[i];const double*r=raw+4*(size_t)src;
        PX[i]=r[1];PY[i]=r[2];PZ[i]=r[3];PM[i]=r[0];}
    for(int i=N;i<N+8;i++){PX[i]=PY[i]=PZ[i]=0.0;PM[i]=0.0;}
    free(raw);
    OUT=(double*)xmalloc((size_t)N*3*sizeof(double));
    for(int i=0;i<3*N;i++) OUT[i]=0.0;

    NNMAX = 6*(N/LEAF_MAX)+4096; if(NNMAX<64)NNMAX=64;
    { size_t cap=(size_t)NNMAX;
      Nmom  =(double*)xmalloc(cap*NT*sizeof(double));
      Ncom  =(double*)xmalloc(cap*3*sizeof(double));
      Nrmax =(float*) xmalloc(cap*sizeof(float));
      Nchild=(int*)   xmalloc(cap*sizeof(int));
      Nstart=(int*)   xmalloc(cap*sizeof(int));
      Nend  =(int*)   xmalloc(cap*sizeof(int));
    }
    STACK=(int*)xmalloc((8*(MAXDEPTH+40)+64)*sizeof(int));

    NN=1; Nchild[0]=-1; Nstart[0]=Nend[0]=0; Nrmax[0]=0.f;
    build(0,0,N,0,o[0],o[1],o[2],0.5*L);
    double t_build=cpu_now();
    if (getenv("GALAXY_DUMP")) {
        for (int n=0;n<40;n++) fprintf(stderr,"n=%3d s=%6d e=%6d ch=%6d rmax=%10.5g com=(%9.4g,%9.4g,%9.4g) m=%g\n",
            n,Nstart[n],Nend[n],Nchild[n],(double)Nrmax[n],Ncom[3*n],Ncom[3*n+1],Ncom[3*n+2],Nmom[n*NT]);
        fprintf(stderr,"px0=(%g,%g,%g)\n",PX[0],PY[0],PZ[0]);
    }
    if (getenv("GALAXY_DEBUG")) fprintf(stderr,"[dbg] N=%d L=%.4g NN=%d/%d t_read=%.3f t_build=%.3f budget=%.3f\n",N,L,NN,NNMAX,t_build-t0,t_build-t0,budget);

    /* ---- choose (P,theta) from the budget using a strided sample ---- */
    int stride = N>384 ? N/384 : 1;
    int M = (N+stride-1)/stride;
    double acc[3];
    double avail = budget - (t_build - t0);
    if (avail < 0.15*budget) avail = 0.15*budget;
    double target = avail*target_frac;
    int    Plist[3] = {0,0,0};
    double Tlist[5] = {0.40,0.52,0.62,0.72,0.82};
    int bestP = 2; double bestT = 0.85; int found = 0;
    if (target_frac > 1e8) { bestP = P; bestT = th_hi; found = 1; }
    for (int pi=0; pi<3 && !found; pi++) {
        if (Plist[pi] != P) continue;
        for (int ti=0; ti<5; ti++) {
            THETA = Tlist[ti];
            double tt = cpu_now();
            for (int i=0;i<N;i+=stride) particle_force(i,acc);
            double bt = cpu_now()-tt;
            bestP = P; bestT = THETA;
            if (bt*(double)N/(double)M <= target) { found = 1; break; }
            if (cpu_now()-t0 > 0.13*budget) { found = 1; break; }
        }
        if (!found && cpu_now()-t0 > 0.13*budget) { bestP=2; bestT=0.82; found=1; }
    }
    P = bestP; build_tables(P); build_deriv_terms(P); build_shift(P);
    THETA = bestT;
    double t_after_trial=cpu_now();
    /* ---- main pass with online safety valve ---- */
    int next_check = 1024;
    for (int i=0;i<N;i++) {
        if (i>=next_check) {
            next_check = i + (i>>4) + 256;
            double el=cpu_now();
            double frac=(double)(i+1)/(double)N;
            double proj=(el-t_after_trial)/frac + (t_after_trial-t0);
            if (proj > budget*0.60 && THETA < 0.95) THETA = (THETA*1.4 < 0.95) ? THETA*1.4 : 0.95;
        }
        particle_force(i,acc);
        unsigned o=PERM[i];
        OUT[3*(size_t)o]=acc[0]; OUT[3*(size_t)o+1]=acc[1]; OUT[3*(size_t)o+2]=acc[2];
    }

    if (getenv("GALAXY_DEBUG")) fprintf(stderr,"[dbg] theta=%.4f t_force=%.3f total=%.3f leaf=%lld node=%lld push=%lld perPart=%.1f\n",THETA,cpu_now()-t_after_trial,cpu_now()-t0,CNT_LEAF,CNT_NODE,CNT_PUSH,(double)(CNT_LEAF+CNT_NODE+CNT_PUSH)/N);
    FILE*g=fopen(argv[2],"wb");
    if(!g){fprintf(stderr,"no output\n");return 1;}
    fwrite(OUT,24,(size_t)N,g);
    fclose(g);
    return 0;
}
