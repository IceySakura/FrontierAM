/*
 * Gravitational acceleration for galaxy snapshots.
 * Softened gravity: a_i = sum_{j!=i} m_j (x_j - x_i) / (|x_j-x_i|^2 + eps^2)^{3/2}
 * G = 1, eps = 0.01.
 *
 * Method: Barnes-Hut octree over Morton-sorted particles with Cartesian
 * multipole moments (mass, center of mass, quadrupole tensor).  Contiguous
 * arrays and space-filling-curve order give good cache locality at large N.
 *
 * Single process, single thread.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <sys/resource.h>

typedef int32_t i32;
typedef int64_t i64;
typedef uint64_t u64;

static const double EPS = 0.01;
static const double EPS2 = 0.0001;

/* ---------------- configuration ---------------- */
static double THETA = 0.5;     /* opening angle (overridden adaptively at runtime) */
static double LRADC = 0.7;   /* MAC leaf-radius factor (x size_L subtracted from dist) */
static int    BUCKET = 32;     /* leaf capacity */
static int    USE_QUAD = 1;    /* use quadrupole moments */
static int    MAXDEPTH = 21;   /* Morton levels */
static const int64_t MAXCELLS = 4000000; /* hard cap on cells (OOM insurance) */
static int    TIMING = 0;

/* ---------------- timing helper ---------------- */
static double cpu_seconds(void) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + 1e-6*ru.ru_utime.tv_usec
         + ru.ru_stime.tv_sec + 1e-6*ru.ru_stime.tv_usec;
}

/* ---------------- particle data (Morton sorted) ---------------- */
static double *PX, *PY, *PZ, *PM;   /* sorted positions and masses */
static u64    *PCODE;               /* sorted Morton codes */
static i32    *PERM;                /* PERM[sorted] = original index */
static int64_t N;

/* ---------------- octree cells (structure of arrays) ---------------- */
typedef struct {
    /* traversal-hot: MAC test + recursion (first ~76 bytes) */
    double comx, comy, comz, size;  /* center of mass + full cell width */
    i32 start;      /* first particle index (sorted array) */
    i32 count;      /* number of particles */
    i32 leaf;       /* 1 if leaf */
    i32 child[8];   /* child ids, -1 if empty/leaf */
    /* multipole data */
    double mass;
    double quad[6]; /* Qxx Qxy Qxz Qyy Qyz Qzz about com */
} Cell;

static Cell *cells;
static int64_t ncells, cellcap;
static i32 ROOT;

static void cell_reserve(int64_t need) {
    if (need <= cellcap) return;
    int64_t nc = cellcap ? cellcap : 1024;
    while (nc < need) nc <<= 1;
    cells = (Cell*)realloc(cells, (size_t)nc * sizeof(Cell));
    if (!cells) { fprintf(stderr, "oom cells\n"); exit(1); }
    cellcap = nc;
}

/* ---------------- Morton encoding ---------------- */
static inline u64 expand_bits(u64 v) {
    v = (v | (v << 32)) & 0x1f00000000ffffULL;
    v = (v | (v << 16)) & 0x1f0000ff0000ffULL;
    v = (v | (v <<  8)) & 0x100f00f00f00f00fULL;
    v = (v | (v <<  4)) & 0x10c30c30c30c30c3ULL;
    v = (v | (v <<  2)) & 0x1249249249249249ULL;
    return v;
}

/* ---------------- radix sort of (key=u64, payload=i32 orig index) ---------- */
/* Sorts keys ascending, carrying payload. Also permutes PX..PM via gather. */
static void radix_sort_keys(u64 *keys, i32 *pay, int64_t n,
                            u64 *tkeys, i32 *tpay) {
    const int PASSES = 8; /* 8 bits * 8 = 64 */
    u64 *ck = keys, *ok = tkeys;
    i32 *cp = pay,  *op = tpay;
    for (int p = 0; p < PASSES; p++) {
        int shift = p * 8;
        size_t cnt[256] = {0};
        for (int64_t i = 0; i < n; i++)
            cnt[(ck[i] >> shift) & 0xff]++;
        size_t sum = 0;
        for (int b = 0; b < 256; b++) { size_t c = cnt[b]; cnt[b] = sum; sum += c; }
        for (int64_t i = 0; i < n; i++) {
            int b = (ck[i] >> shift) & 0xff;
            size_t pos = cnt[b]++;
            ok[pos] = ck[i];
            op[pos] = cp[i];
        }
        u64 *tk = ck; ck = ok; ok = tk;
        i32 *tp2 = cp; cp = op; op = tp2;
    }
    if (ck != keys) { memcpy(keys, ck, (size_t)n*sizeof(u64)); memcpy(pay, cp, (size_t)n*sizeof(i32)); }
}

/* ---------------- tree build ---------------- */
/* Compute leaf moments directly from particles [start,end). */
static void leaf_moments(Cell *C, i32 start, i32 count) {
    double m = 0, cx = 0, cy = 0, cz = 0;
    for (i32 i = start; i < start+count; i++) {
        double w = PM[i];
        m += w; cx += w*PX[i]; cy += w*PY[i]; cz += w*PZ[i];
    }
    double inv = (m > 0) ? 1.0/m : 0.0;
    cx *= inv; cy *= inv; cz *= inv;
    C->mass = m; C->comx = cx; C->comy = cy; C->comz = cz;
    double q0=0,q1=0,q2=0,q3=0,q4=0,q5=0;
    if (USE_QUAD) {
        for (i32 i = start; i < start+count; i++) {
            double w = PM[i];
            double dx = PX[i]-cx, dy = PY[i]-cy, dz = PZ[i]-cz;
            q0 += w*dx*dx; q1 += w*dx*dy; q2 += w*dx*dz;
            q3 += w*dy*dy; q4 += w*dy*dz; q5 += w*dz*dz;
        }
    }
    C->quad[0]=q0; C->quad[1]=q1; C->quad[2]=q2;
    C->quad[3]=q3; C->quad[4]=q4; C->quad[5]=q5;
}

static int64_t build(i32 start, i32 end, int depth,
                     double cx, double cy, double cz, double hw) {
    cell_reserve(ncells + 9);
    int64_t id = ncells++;
    cells[id].start = start;
    cells[id].count = end - start;
    cells[id].size = 2.0*hw;
    for (int k = 0; k < 8; k++) cells[id].child[k] = -1;

    i32 n = end - start;
    if (n <= BUCKET || depth >= MAXDEPTH || ncells >= MAXCELLS) {
        cells[id].leaf = 1;
        leaf_moments(&cells[id], start, n);
        return id;
    }
    cells[id].leaf = 0;

    /* Octant boundaries from Morton bits: guaranteed contiguous & monotonic
       because the particles are Morton-sorted. shift selects this depth's bits. */
    i32 bnd[9];
    bnd[0] = start;
    int bi = 0;
    int shift = 60 - 3*depth;
    for (i32 i = start; i < end; i++) {
        int o = (int)((PCODE[i] >> shift) & 7);
        while (bi < o) { bnd[bi+1] = i; bi++; }
    }
    while (bi < 8) { bnd[bi+1] = end; bi++; }

    /* child centers */
    double qhw = 0.5*hw;
    for (int k = 0; k < 8; k++) {
        if (bnd[k+1] <= bnd[k]) { cells[id].child[k] = -1; continue; }
        int kx = (k >> 2) & 1, ky = (k >> 1) & 1, kz = k & 1;
        double ccx = cx + (kx ? qhw : -qhw);
        double ccy = cy + (ky ? qhw : -qhw);
        double ccz = cz + (kz ? qhw : -qhw);
        /* build() may realloc cells; assign child id only after it returns */
        i32 cid = (i32)build(bnd[k], bnd[k+1], depth+1, ccx, ccy, ccz, qhw);
        cells[id].child[k] = cid;
    }

    /* combine moments from children (cells pointer stable now) */
    double M = 0, gx = 0, gy = 0, gz = 0;
    for (int k = 0; k < 8; k++) {
        i32 ch = cells[id].child[k];
        if (ch < 0) continue;
        Cell *K = &cells[ch];
        M += K->mass; gx += K->mass*K->comx; gy += K->mass*K->comy; gz += K->mass*K->comz;
    }
    double inv = (M > 0) ? 1.0/M : 0.0;
    gx *= inv; gy *= inv; gz *= inv;
    cells[id].mass = M; cells[id].comx = gx; cells[id].comy = gy; cells[id].comz = gz;
    double q0=0,q1=0,q2=0,q3=0,q4=0,q5=0;
    if (USE_QUAD) {
        for (int k = 0; k < 8; k++) {
            i32 ch = cells[id].child[k];
            if (ch < 0) continue;
            Cell *K = &cells[ch];
            double dx = K->comx-gx, dy = K->comy-gy, dz = K->comz-gz;
            double m = K->mass;
            const double *Q = K->quad;
            q0 += Q[0] + m*dx*dx;
            q1 += Q[1] + m*dx*dy;
            q2 += Q[2] + m*dx*dz;
            q3 += Q[3] + m*dy*dy;
            q4 += Q[4] + m*dy*dz;
            q5 += Q[5] + m*dz*dz;
        }
    }
    cells[id].quad[0]=q0; cells[id].quad[1]=q1; cells[id].quad[2]=q2;
    cells[id].quad[3]=q3; cells[id].quad[4]=q4; cells[id].quad[5]=q5;
    return id;
}

/* ---------------- force evaluation: leaf-batched, SIMD-friendly ---------- */
static double *AX, *AY, *AZ; /* accelerations in sorted order */
static long long g_mult, g_direct_pairs, g_opened; static int g_guard=-1, g_guard_fired=0;

/* current leaf state during traversal */
static i32   curL;            /* current leaf cell id */
static i32   curLs, curLe;    /* particle range of current leaf */
static double Lcx, Lcy, Lcz;  /* current leaf center of mass */
static double Lrad;           /* conservative radius of current leaf */
static double THETA2;

/* Apply cell S's multipole to all particles of the current leaf.
   Element-wise over particles -> auto-vectorizes (AVX2, 4 doubles). */
static void apply_multipole(const Cell *S) {
    g_mult++;
    const double Mx=S->comx, My=S->comy, Mz=S->comz, M=S->mass;
    const double q0=S->quad[0],q1=S->quad[1],q2=S->quad[2];
    const double q3=S->quad[3],q4=S->quad[4],q5=S->quad[5];
    const double Tr=q0+q3+q5;
    const i32 s=curLs, e=curLe;
    if (USE_QUAD) {
        for (i32 i=s;i<e;i++){
            double Rx=PX[i]-Mx, Ry=PY[i]-My, Rz=PZ[i]-Mz;
            double r2=Rx*Rx+Ry*Ry+Rz*Rz+EPS2;
            double inv=1.0/r2;
            double inv32=inv*sqrt(inv);
            double inv52=inv32*inv, inv72=inv52*inv;
            double f0=-M*inv32;
            double QRx=q0*Rx+q1*Ry+q2*Rz;
            double QRy=q1*Rx+q3*Ry+q4*Rz;
            double QRz=q2*Rx+q4*Ry+q5*Rz;
            double RQR=Rx*QRx+Ry*QRy+Rz*QRz;
            double c1=1.5*Tr*inv52-7.5*RQR*inv72;
            double c2=3.0*inv52;
            AX[i]+=f0*Rx+c1*Rx+c2*QRx;
            AY[i]+=f0*Ry+c1*Ry+c2*QRy;
            AZ[i]+=f0*Rz+c1*Rz+c2*QRz;
        }
    } else {
        for (i32 i=s;i<e;i++){
            double Rx=PX[i]-Mx, Ry=PY[i]-My, Rz=PZ[i]-Mz;
            double r2=Rx*Rx+Ry*Ry+Rz*Rz+EPS2;
            double inv=1.0/r2;
            double f0=-M*inv*sqrt(inv);
            AX[i]+=f0*Rx; AY[i]+=f0*Ry; AZ[i]+=f0*Rz;
        }
    }
}

/* Direct summation: current leaf particles x source particles [ss,se). */
static void direct_leaf(i32 ss, i32 se) {
    g_direct_pairs += (i64)(curLe-curLs)*(se-ss);
    const i32 s=curLs, e=curLe;
    for (i32 j=ss;j<se;j++){
        const double xj=PX[j], yj=PY[j], zj=PZ[j], mj=PM[j];
        for (i32 i=s;i<e;i++){
            double dx=xj-PX[i], dy=yj-PY[i], dz=zj-PZ[i];
            double r2=dx*dx+dy*dy+dz*dz+EPS2;
            double inv=1.0/r2;
            double w=mj*inv*sqrt(inv);
            AX[i]+=w*dx; AY[i]+=w*dy; AZ[i]+=w*dz;
        }
    }
}

/* Recursively traverse source tree for the current leaf, applying multipoles
   for well-separated cells and direct summation for near leaves. */
static void traverse_apply(i32 S) {
    const Cell *Sc=&cells[S];
    double dx=Lcx-Sc->comx, dy=Lcy-Sc->comy, dz=Lcz-Sc->comz;
    double d2=dx*dx+dy*dy+dz*dz;
    double sz=Sc->size;
    double d=sqrt(d2);
    double deff=d-Lrad;
    if (S!=curL && deff>0.0 && sz*sz < THETA2*deff*deff) {
        apply_multipole(Sc);
        return;
    }
    if (Sc->leaf) { direct_leaf(Sc->start, Sc->start+Sc->count); return; }
    for (int k=0;k<8;k++){ i32 ch=Sc->child[k]; if(ch>=0) traverse_apply(ch); }
}

static i32 *LEAFIDS; static i32 NLEAVES;

static void collect_leaves(void){
    LEAFIDS=(i32*)malloc((size_t)(ncells+1)*sizeof(i32));
    NLEAVES=0;
    for(i32 c=0;c<(i32)ncells;c++) if(cells[c].leaf) LEAFIDS[NLEAVES++]=c;
}

/* Process leaves (every `stride`-th). zero_acc: clear accelerations first.
   deadline>0: hard CPU-time guard -> switch to very coarse theta if exceeded. */
static void run_leaves(int stride, int zero_acc, double deadline){
    if(zero_acc){ for(i64 i=0;i<N;i++){ AX[i]=0; AY[i]=0; AZ[i]=0; } }
    int coarse=0;
    for(i32 k=0;k<NLEAVES;k+=stride){
        if(deadline>0.0 && !coarse && (k&127)==0){
            if(cpu_seconds()>deadline){ THETA2=1.69; coarse=1; g_guard=k; } /* theta=1.3 */
        }
        i32 c=LEAFIDS[k];
        curL=c; curLs=cells[c].start; curLe=curLs+cells[c].count;
        Lcx=cells[c].comx; Lcy=cells[c].comy; Lcz=cells[c].comz;
        Lrad=LRADC*cells[c].size;
        traverse_apply(ROOT);
    }
    if(coarse) g_guard_fired=1;
}

/* ---------------- main ---------------- */
int main(int argc, char **argv) {
    double t0 = cpu_seconds();
    if (argc < 3) { fprintf(stderr, "usage: %s INPUT OUTPUT\n", argv[0]); return 2; }
    const char *inpath = argv[1], *outpath = argv[2];

    { const char *e;
      if ((e=getenv("GALAXY_THETA"))) THETA = atof(e);
      if ((e=getenv("GALAXY_LRADC"))) LRADC = atof(e);
      if ((e=getenv("GALAXY_BUCKET"))) BUCKET = atoi(e);
      if ((e=getenv("GALAXY_QUAD"))) USE_QUAD = atoi(e);
      if ((e=getenv("GALAXY_TIMING"))) TIMING = atoi(e);
    }

    /* read input */
    FILE *f = fopen(inpath, "rb");
    if (!f) { fprintf(stderr, "cannot open input\n"); return 2; }
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    N = fsz / 32;
    if (N <= 0) { fclose(f); fprintf(stderr,"empty input\n"); return 2; }

    double *buf = (double*)malloc((size_t)N*4*sizeof(double));
    if (!buf) { fprintf(stderr,"oom buf\n"); return 1; }
    if (fread(buf, 32, (size_t)N, f) != (size_t)N) { fprintf(stderr,"read error\n"); return 2; }
    fclose(f);

    /* de-interleave: m,x,y,z */
    double *ux=(double*)malloc((size_t)N*sizeof(double));
    double *uy=(double*)malloc((size_t)N*sizeof(double));
    double *uz=(double*)malloc((size_t)N*sizeof(double));
    double *um=(double*)malloc((size_t)N*sizeof(double));
    if(!ux||!uy||!uz||!um){ fprintf(stderr,"oom deint\n"); return 1; }
    for (int64_t i=0;i<N;i++){ um[i]=buf[4*i]; ux[i]=buf[4*i+1]; uy[i]=buf[4*i+2]; uz[i]=buf[4*i+3]; }
    free(buf);

    /* handle trivial N */
    if (N == 1) {
        FILE *o=fopen(outpath,"wb"); double z[3]={0,0,0}; fwrite(z,24,1,o); fclose(o);
        return 0;
    }

    /* bounding cube */
    double minx=ux[0],maxx=ux[0],miny=uy[0],maxy=uy[0],minz=uz[0],maxz=uz[0];
    for (int64_t i=1;i<N;i++){
        if(ux[i]<minx)minx=ux[i]; if(ux[i]>maxx)maxx=ux[i];
        if(uy[i]<miny)miny=uy[i]; if(uy[i]>maxy)maxy=uy[i];
        if(uz[i]<minz)minz=uz[i]; if(uz[i]>maxz)maxz=uz[i];
    }
    double c0x=0.5*(minx+maxx), c0y=0.5*(miny+maxy), c0z=0.5*(minz+maxz);
    double L=0.5*fmax(maxx-minx, fmax(maxy-miny, maxz-minz));
    if (!(L > 0)) L = 1.0;
    L *= 1.0000001;  /* margin so max maps inside */
    double inv2L = 1.0/(2.0*L);

    /* morton codes */
    u64 *keys=(u64*)malloc((size_t)N*sizeof(u64));
    i32 *pay=(i32*)malloc((size_t)N*sizeof(i32));
    const u64 MASK = (1ULL<<21)-1;
    for (int64_t i=0;i<N;i++){
        double fx=(ux[i]-(c0x-L))*inv2L;
        double fy=(uy[i]-(c0y-L))*inv2L;
        double fz=(uz[i]-(c0z-L))*inv2L;
        u64 ix=(u64)(fx*(double)(1ULL<<21)); if(ix>MASK)ix=MASK;
        u64 iy=(u64)(fy*(double)(1ULL<<21)); if(iy>MASK)iy=MASK;
        u64 iz=(u64)(fz*(double)(1ULL<<21)); if(iz>MASK)iz=MASK;
        keys[i]=(expand_bits(ix)<<2)|(expand_bits(iy)<<1)|expand_bits(iz);
        pay[i]=(i32)i;
    }
    u64 *tkeys=(u64*)malloc((size_t)N*sizeof(u64));
    i32 *tpay=(i32*)malloc((size_t)N*sizeof(i32));
    double tr=cpu_seconds();
    radix_sort_keys(keys,pay,N,tkeys,tpay);
    free(tkeys); free(tpay);
    if(TIMING) fprintf(stderr,"sort %.3f\n", cpu_seconds()-tr);

    /* gather into sorted arrays */
    PX=(double*)malloc((size_t)N*sizeof(double));
    PY=(double*)malloc((size_t)N*sizeof(double));
    PZ=(double*)malloc((size_t)N*sizeof(double));
    PM=(double*)malloc((size_t)N*sizeof(double));
    PERM=(i32*)malloc((size_t)N*sizeof(i32));
    if(!PX||!PY||!PZ||!PM||!PERM){ fprintf(stderr,"oom sorted\n"); return 1; }
    for (int64_t i=0;i<N;i++){
        i32 o=pay[i];
        PX[i]=ux[o]; PY[i]=uy[o]; PZ[i]=uz[o]; PM[i]=um[o]; PERM[i]=o;
    }
    free(ux);free(uy);free(uz);free(um);free(pay);
    PCODE = keys;  /* keep sorted Morton codes for the tree build */

    /* build tree */
    double tb=cpu_seconds();
    cell_reserve(1024);
    ncells=0;
    ROOT=(i32)build(0,(i32)N,0,c0x,c0y,c0z,L);
    if(TIMING) fprintf(stderr,"build %.3f ncells=%lld\n", cpu_seconds()-tb,(long long)ncells);

    /* forces */
    AX=(double*)malloc((size_t)N*sizeof(double));
    AY=(double*)malloc((size_t)N*sizeof(double));
    AZ=(double*)malloc((size_t)N*sizeof(double));
    if(!AX||!AY||!AZ){fprintf(stderr,"oom acc\n");return 1;}
    collect_leaves();
    if(!LEAFIDS){fprintf(stderr,"oom leaves\n");return 1;}

    const char *lim_e = getenv("GALAXY_CPU_LIMIT_S");
    double Tbudget = lim_e ? atof(lim_e) : 0.0;
    int adaptive = (Tbudget > 0.0 && getenv("GALAXY_THETA") == NULL);
    double deadline = 0.0;

    if (adaptive) {
        /* Two-probe adaptive theta. Probe a strided leaf sample at theta=0.5,
           guess theta, re-probe AT that theta, fit the local time~theta^-alpha
           exponent from the two probes, then solve for the theta that fills the
           budget. Accurate prediction keeps the hard guard from firing. */
        int nsample = NLEAVES/24; if(nsample<80)nsample=80; if(nsample>600)nsample=600;
        if(nsample>NLEAVES) nsample=NLEAVES;
        int stride = NLEAVES/nsample; if(stride<1)stride=1;
        int actual = (NLEAVES + stride - 1)/stride;
        double scale = (double)NLEAVES/(double)actual;
        double write_allow = 0.015*Tbudget + 0.02;

        /* probe 1 @ theta=0.5 */
        THETA=0.5; THETA2=0.25;
        double tp0=cpu_seconds(); run_leaves(stride,0,0.0); double t1=cpu_seconds()-tp0;
        double T1 = t1*scale;
        double usable = 0.90*Tbudget - (cpu_seconds()-t0) - write_allow;
        if(usable < 0.05*Tbudget) usable = 0.05*Tbudget;
        double theta = (T1>0.0) ? 0.5*pow(T1/usable,0.5) : 0.5;   /* alpha~2 guess */
        if(theta<0.25)theta=0.25; if(theta>1.2)theta=1.2;

        /* probe 2 @ theta (direct measurement near the operating point) */
        THETA=theta; THETA2=theta*theta;
        tp0=cpu_seconds(); run_leaves(stride,0,0.0); double t2=cpu_seconds()-tp0;
        double T2 = t2*scale;
        /* fit local exponent: T ~ theta^-alpha  =>  alpha = ln(T1/T2)/ln(theta/0.5) */
        double alpha=2.0;
        if(T1>0 && T2>0 && theta>0.52){ double lg=log(theta/0.5); if(lg>1e-3){ alpha=log(T1/T2)/lg; } }
        else if(T1>0 && T2>0 && theta<0.48){ double lg=log(0.5/theta); if(lg>1e-3){ alpha=log(T1/T2)/lg; } }
        if(alpha<1.2)alpha=1.2; if(alpha>3.5)alpha=3.5;
        usable = 0.90*Tbudget - (cpu_seconds()-t0) - write_allow;
        if(usable < 0.05*Tbudget) usable = 0.05*Tbudget;
        double theta2 = (T2>0.0) ? theta*pow(T2/usable,1.0/alpha) : theta;
        if(theta2<0.25)theta2=0.25; if(theta2>1.2)theta2=1.2;
        theta2 *= 1.005; if(theta2>1.2)theta2=1.2;   /* tiny safety bump */
        THETA=theta2; THETA2=theta2*theta2;
        deadline = t0 + 0.96*Tbudget;
        if(TIMING) fprintf(stderr,"adaptive: T1(0.5)=%.3f T2(%.3f)=%.3f alpha=%.2f usable=%.3f theta=%.3f nleaves=%d\n",
                           T1,theta,T2,alpha,usable,theta2,NLEAVES);
    } else {
        THETA2=THETA*THETA;
        if(Tbudget>0.0) deadline = t0 + 0.90*Tbudget;
    }

    g_mult=0; g_direct_pairs=0;
    double tf=cpu_seconds();
    run_leaves(1, 1, deadline);
    if(TIMING) fprintf(stderr,"forces %.3f theta=%.3f mult=%lld direct=%lld guard=%d(at %d)\n",
        cpu_seconds()-tf, THETA, g_mult, g_direct_pairs, g_guard_fired, g_guard);

    /* unsort and write */
    double *out=(double*)malloc((size_t)N*3*sizeof(double));
    for (int64_t i=0;i<N;i++){
        i32 o=PERM[i];
        out[3*o+0]=AX[i]; out[3*o+1]=AY[i]; out[3*o+2]=AZ[i];
    }
    FILE *o=fopen(outpath,"wb");
    if(!o){fprintf(stderr,"cannot open output\n");return 2;}
    fwrite(out,24,(size_t)N,o);
    fclose(o);
    free(out);

    if(TIMING){ struct rusage ru; getrusage(RUSAGE_SELF,&ru);
        fprintf(stderr,"total %.3f N=%lld maxrss=%.0fMB ncells=%lld\n", cpu_seconds()-t0,(long long)N, ru.ru_maxrss/1024.0,(long long)ncells); }
    return 0;
}
