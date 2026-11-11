/*
 * chebyshev2: second Chebyshev panel member for
 *   y = exp(-t L) b0 + t phi_1(-t L) b1,  L symmetric PSD.
 * Techniques (CHEB2_DISABLE=lanczos,fused,adaptive,sell,rcm,reuse):
 *  1 short Lanczos bound + Gershgorin fallback (guard in recurrence)
 *  2 fused forward three-term recurrence (interleaved pair)
 *  3 certified adaptive a-posteriori truncation
 *  4 SELL-4-64 (Kreutzer et al. 2014) AVX2/FMA + optional RCM
 *  5 session reuse of bound/coefficients/format
 * Coefficients: scaled Bessel exp(-a)I_k(a), q for g(x)=t phi_1(-t x)
 * via (1+z)q = t(1-f)/a in long double (as panel/chebyshev).
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

#define SAFETY_FACTOR 8.0L
#define MAX_HEADER 4096
#define MAX_DEGREE 4096
#define SELL_C 4
#define SELL_SIGMA 64

static int n_rows;
static uint64_t *row_ptr;      /* original CSR */
static uint32_t *col_idx;
static double *a_val;
static uint64_t nnz_g;
static int n_requests_g;
static double t_g;

/* disable flags */
static int dis_lanczos, dis_fused, dis_adaptive, dis_sell, dis_rcm, dis_reuse;

/* session state (reused unless dis_reuse) */
static double lambda_gersh, lambda_upper, lambda_lower;
static double *q_coeff, *f_coeff, *e_ratio; /* e_ratio[k]=I_{k+1}/I_k */
static long double *q_tail, *f_tail;
static int coeff_top;
static int *perm_new2old; /* size n, NULL = identity */
static int *perm_old2new;
/* permuted CSR (when !sell) */
static uint64_t *p_row_ptr;
static uint32_t *p_col;
static double *p_val;
/* SELL */
static int sell_nslices;
static uint32_t *sell_ptr; /* nslices+1 offsets in units of entries (padded) */
static uint32_t *sell_col;
static double *sell_val;
static int use_sell;

static void die(const char *s){ fprintf(stderr,"stream_chebyshev2: %s\n",s); exit(2); }
static void *xmalloc(size_t b){ void *p=malloc(b?b:1); if(!p) die("oom"); return p; }
static void *xcalloc(size_t c,size_t b){ void *p=calloc(c?c:1,b?b:1); if(!p) die("oom"); return p; }
static void read_exact(void *p,size_t bytes){ unsigned char *q=p; while(bytes){ ssize_t g=read(STDIN_FILENO,q,bytes); if(g<=0) die("short input"); q+=(size_t)g; bytes-=(size_t)g; } }
static void write_all(const void *p,size_t bytes){ const unsigned char *q=p; while(bytes){ ssize_t g=write(STDOUT_FILENO,q,bytes); if(g<=0) die("short output"); q+=(size_t)g; bytes-=(size_t)g; } }
static double json_number(const char *json,const char *key){
  char needle[32]; snprintf(needle,sizeof needle,"\"%s\"",key);
  const char *p=strstr(json,needle); if(!p||!(p=strchr(p+strlen(needle),':'))) die("missing JSON field");
  errno=0; char *end; double v=strtod(p+1,&end); if(end==p+1||errno) die("invalid JSON number"); return v;
}
static uint64_t splitmix64(uint64_t x){ x+=0x9e3779b97f4a7c15ULL; x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL; x=(x^(x>>27))*0x94d049bb133111ebULL; return x^(x>>31); }

static void parse_disable(void){
  const char *s=getenv("CHEB2_DISABLE"); if(!s) return;
  char buf[256]; snprintf(buf,sizeof buf,"%s",s);
  for(char *tok=strtok(buf,", "); tok; tok=strtok(NULL,", ")){
    if(!strcmp(tok,"lanczos")) dis_lanczos=1;
    else if(!strcmp(tok,"fused")) dis_fused=1;
    else if(!strcmp(tok,"adaptive")) dis_adaptive=1;
    else if(!strcmp(tok,"sell")) dis_sell=1;
    else if(!strcmp(tok,"rcm")) dis_rcm=1;
    else if(!strcmp(tok,"reuse")) dis_reuse=1;
  }
}

/* ---------- SpMV on active format: y = L x (permuted space) ---------- */
static void spmv_active_lo(const double *restrict x, double *restrict y, double *restrict ylo){
  if(use_sell){
    for(int s=0;s<sell_nslices;s++){
      uint32_t base=sell_ptr[s]; int slen=(int)(sell_ptr[s+1]-sell_ptr[s])/SELL_C;
      int rows = n_rows - s*SELL_C; if(rows>SELL_C) rows=SELL_C;
#ifdef __AVX2__
      __m256d acc=_mm256_setzero_pd(), comp=_mm256_setzero_pd();
      const __m256d signm=_mm256_set1_pd(-0.0);
      for(int j=0;j<slen;j++){
        __m256d v=_mm256_loadu_pd(&sell_val[base+(size_t)j*SELL_C]);
        __m128i idx=_mm_loadu_si128((const __m128i*)&sell_col[base+(size_t)j*SELL_C]);
        __m256d xv=_mm256_i32gather_pd(x, idx, 8);
        __m256d pr=_mm256_mul_pd(v,xv);
        __m256d pe=_mm256_fmsub_pd(v,xv,pr);
        __m256d tt=_mm256_add_pd(acc,pr);
        __m256d e1=_mm256_add_pd(_mm256_sub_pd(acc,tt),pr);
        __m256d e2=_mm256_add_pd(_mm256_sub_pd(pr,tt),acc);
        __m256d mask=_mm256_cmp_pd(_mm256_andnot_pd(signm,acc),_mm256_andnot_pd(signm,pr),_CMP_GE_OQ);
        comp=_mm256_add_pd(comp,_mm256_add_pd(_mm256_blendv_pd(e2,e1,mask),pe));
        acc=tt;
      }
      double tmp[4]; _mm256_storeu_pd(tmp,acc);
      for(int l=0;l<rows;l++) y[s*SELL_C+l]=tmp[l];
      if(ylo){ double tc[4]; _mm256_storeu_pd(tc,comp); for(int l=0;l<rows;l++) ylo[s*SELL_C+l]=tc[l]; }
#else
      double acc[SELL_C]={0,0,0,0}, cmp[SELL_C]={0,0,0,0};
      for(int j=0;j<slen;j++) for(int l=0;l<SELL_C;l++){ uint32_t e=base+(uint32_t)j*SELL_C+(uint32_t)l; double xv=x[sell_col[e]]; double pr=sell_val[e]*xv; double pe=fma(sell_val[e],xv,-pr); double tt=acc[l]+pr; cmp[l]+=((fabs(acc[l])>=fabs(pr))?((acc[l]-tt)+pr):((pr-tt)+acc[l]))+pe; acc[l]=tt; }
      for(int l=0;l<rows;l++){ y[s*SELL_C+l]=acc[l]; if(ylo) ylo[s*SELL_C+l]=cmp[l]; }
#endif
    }
  } else if(p_row_ptr){
    for(int i=0;i<n_rows;i++){ double sm=0,cp=0; for(uint64_t k=p_row_ptr[i];k<p_row_ptr[i+1];k++){ double xv=x[p_col[k]]; double pr=p_val[k]*xv; double pe=fma(p_val[k],xv,-pr); double tt=sm+pr; cp+=((fabs(sm)>=fabs(pr))?((sm-tt)+pr):((pr-tt)+sm))+pe; sm=tt; } y[i]=sm; if(ylo) ylo[i]=cp; }
  } else {
    for(int i=0;i<n_rows;i++){ double sm=0,cp=0; for(uint64_t k=row_ptr[i];k<row_ptr[i+1];k++){ double xv=x[col_idx[k]]; double pr=a_val[k]*xv; double pe=fma(a_val[k],xv,-pr); double tt=sm+pr; cp+=((fabs(sm)>=fabs(pr))?((sm-tt)+pr):((pr-tt)+sm))+pe; sm=tt; } y[i]=sm; if(ylo) ylo[i]=cp; }
  }
}
static void spmv_active(const double *restrict x, double *restrict y){ spmv_active_lo(x,y,NULL); }
static void spmv_pair_active_lo(const double *restrict xp, double *restrict yp, double *restrict yplo);
/* original CSR SpMV (for Lanczos, before format built / in original numbering) */
static void spmv_pair_active(const double *restrict xp, double *restrict yp){ spmv_pair_active_lo(xp,yp,NULL); }
static void spmv_orig(const double *restrict x, double *restrict y){
  for(int i=0;i<n_rows;i++){ double sm=0; for(uint64_t k=row_ptr[i];k<row_ptr[i+1];k++) sm+=a_val[k]*x[col_idx[k]]; y[i]=sm; }
}
/* fused pair SpMV: xp interleaved [2i]=comp0,[2i+1]=comp1 ; one matrix pass */
static void spmv_pair_active_lo(const double *restrict xp, double *restrict yp, double *restrict yplo){
  if(use_sell){
    for(int s=0;s<sell_nslices;s++){
      uint32_t base=sell_ptr[s]; int slen=(int)(sell_ptr[s+1]-sell_ptr[s])/SELL_C;
      int rows=n_rows-s*SELL_C; if(rows>SELL_C) rows=SELL_C;
#ifdef __AVX2__
      __m256d acc0=_mm256_setzero_pd(), acc1=_mm256_setzero_pd(), cp0=_mm256_setzero_pd(), cp1=_mm256_setzero_pd();
      const __m256d signm=_mm256_set1_pd(-0.0);
      for(int j=0;j<slen;j++){
        __m256d v=_mm256_loadu_pd(&sell_val[base+(size_t)j*SELL_C]);
        __m128i idx=_mm_loadu_si128((const __m128i*)&sell_col[base+(size_t)j*SELL_C]);
        __m128i idx0=_mm_add_epi32(idx,idx);
        __m128i idx1=_mm_add_epi32(idx0,_mm_set1_epi32(1));
        __m256d xv0=_mm256_i32gather_pd(xp,idx0,8);
        __m256d xv1=_mm256_i32gather_pd(xp,idx1,8);
        __m256d pr0=_mm256_mul_pd(v,xv0); __m256d pe0=_mm256_fmsub_pd(v,xv0,pr0);
        __m256d tt0=_mm256_add_pd(acc0,pr0);
        __m256d e10=_mm256_add_pd(_mm256_sub_pd(acc0,tt0),pr0);
        __m256d e20=_mm256_add_pd(_mm256_sub_pd(pr0,tt0),acc0);
        __m256d m0=_mm256_cmp_pd(_mm256_andnot_pd(signm,acc0),_mm256_andnot_pd(signm,pr0),_CMP_GE_OQ);
        cp0=_mm256_add_pd(cp0,_mm256_add_pd(_mm256_blendv_pd(e20,e10,m0),pe0)); acc0=tt0;
        __m256d pr1=_mm256_mul_pd(v,xv1); __m256d pe1=_mm256_fmsub_pd(v,xv1,pr1);
        __m256d tt1=_mm256_add_pd(acc1,pr1);
        __m256d e11=_mm256_add_pd(_mm256_sub_pd(acc1,tt1),pr1);
        __m256d e21=_mm256_add_pd(_mm256_sub_pd(pr1,tt1),acc1);
        __m256d m1=_mm256_cmp_pd(_mm256_andnot_pd(signm,acc1),_mm256_andnot_pd(signm,pr1),_CMP_GE_OQ);
        cp1=_mm256_add_pd(cp1,_mm256_add_pd(_mm256_blendv_pd(e21,e11,m1),pe1)); acc1=tt1;
      }
      double t0[4],t1[4],c0[4],c1[4]; _mm256_storeu_pd(t0,acc0); _mm256_storeu_pd(t1,acc1); _mm256_storeu_pd(c0,cp0); _mm256_storeu_pd(c1,cp1);
      for(int l=0;l<rows;l++){ yp[2*(size_t)(s*SELL_C+l)]=t0[l]; yp[2*(size_t)(s*SELL_C+l)+1]=t1[l]; if(yplo){ yplo[2*(size_t)(s*SELL_C+l)]=c0[l]; yplo[2*(size_t)(s*SELL_C+l)+1]=c1[l]; } }
#else
      double acc0[SELL_C]={0,0,0,0}, acc1[SELL_C]={0,0,0,0}, cp0[SELL_C]={0,0,0,0}, cp1[SELL_C]={0,0,0,0};
      for(int j=0;j<slen;j++) for(int l=0;l<SELL_C;l++){ uint32_t e=base+(uint32_t)j*SELL_C+(uint32_t)l; double v=sell_val[e]; const double *xv=&xp[2*(size_t)sell_col[e]]; double pr0=v*xv[0]; double pe0=fma(v,xv[0],-pr0); double t0=acc0[l]+pr0; cp0[l]+=((fabs(acc0[l])>=fabs(pr0))?((acc0[l]-t0)+pr0):((pr0-t0)+acc0[l]))+pe0; acc0[l]=t0; double pr1=v*xv[1]; double pe1=fma(v,xv[1],-pr1); double t1=acc1[l]+pr1; cp1[l]+=((fabs(acc1[l])>=fabs(pr1))?((acc1[l]-t1)+pr1):((pr1-t1)+acc1[l]))+pe1; acc1[l]=t1; }
      for(int l=0;l<rows;l++){ yp[2*(size_t)(s*SELL_C+l)]=acc0[l]; yp[2*(size_t)(s*SELL_C+l)+1]=acc1[l]; if(yplo){ yplo[2*(size_t)(s*SELL_C+l)]=cp0[l]; yplo[2*(size_t)(s*SELL_C+l)+1]=cp1[l]; } }
#endif
    }
  } else {
    const uint64_t *rp = p_row_ptr?p_row_ptr:row_ptr;
    const uint32_t *ci = p_row_ptr?p_col:col_idx;
    const double *av = p_row_ptr?p_val:a_val;
    for(int i=0;i<n_rows;i++){ double s0=0,s1=0,c0=0,c1=0; for(uint64_t k=rp[i];k<rp[i+1];k++){ double v=av[k]; const double *xv=&xp[2*(size_t)ci[k]]; double pr0=v*xv[0]; double pe0=fma(v,xv[0],-pr0); double t0=s0+pr0; c0+=((fabs(s0)>=fabs(pr0))?((s0-t0)+pr0):((pr0-t0)+s0))+pe0; s0=t0; double pr1=v*xv[1]; double pe1=fma(v,xv[1],-pr1); double t1=s1+pr1; c1+=((fabs(s1)>=fabs(pr1))?((s1-t1)+pr1):((pr1-t1)+s1))+pe1; s1=t1; } yp[2*(size_t)i]=s0; yp[2*(size_t)i+1]=s1; if(yplo){ yplo[2*(size_t)i]=c0; yplo[2*(size_t)i+1]=c1; } }
  }
}


/* plain (uncompensated) SpMV kernels: used for the lo part of the hi/lo
   split, whose rounding is second-order. */
static void spmv_plain(const double *restrict x, double *restrict y){
  if(use_sell){
    for(int s=0;s<sell_nslices;s++){
      uint32_t base=sell_ptr[s]; int slen=(int)(sell_ptr[s+1]-sell_ptr[s])/SELL_C;
      int rows=n_rows-s*SELL_C; if(rows>SELL_C) rows=SELL_C;
#ifdef __AVX2__
      __m256d acc=_mm256_setzero_pd();
      for(int j=0;j<slen;j++){
        __m256d v=_mm256_loadu_pd(&sell_val[base+(size_t)j*SELL_C]);
        __m128i idx=_mm_loadu_si128((const __m128i*)&sell_col[base+(size_t)j*SELL_C]);
        acc=_mm256_fmadd_pd(v,_mm256_i32gather_pd(x,idx,8),acc);
      }
      double tmp[4]; _mm256_storeu_pd(tmp,acc);
      for(int l=0;l<rows;l++) y[s*SELL_C+l]=tmp[l];
#else
      double acc[SELL_C]={0,0,0,0};
      for(int j=0;j<slen;j++) for(int l=0;l<SELL_C;l++){ uint32_t e=base+(uint32_t)j*SELL_C+(uint32_t)l; acc[l]+=sell_val[e]*x[sell_col[e]]; }
      for(int l=0;l<rows;l++) y[s*SELL_C+l]=acc[l];
#endif
    }
  } else if(p_row_ptr){
    for(int i=0;i<n_rows;i++){ double sm=0; for(uint64_t k=p_row_ptr[i];k<p_row_ptr[i+1];k++) sm+=p_val[k]*x[p_col[k]]; y[i]=sm; }
  } else {
    for(int i=0;i<n_rows;i++){ double sm=0; for(uint64_t k=row_ptr[i];k<row_ptr[i+1];k++) sm+=a_val[k]*x[col_idx[k]]; y[i]=sm; }
  }
}
static void spmv_pair_plain(const double *restrict xp, double *restrict yp){
  if(use_sell){
    for(int s=0;s<sell_nslices;s++){
      uint32_t base=sell_ptr[s]; int slen=(int)(sell_ptr[s+1]-sell_ptr[s])/SELL_C;
      int rows=n_rows-s*SELL_C; if(rows>SELL_C) rows=SELL_C;
#ifdef __AVX2__
      __m256d acc0=_mm256_setzero_pd(), acc1=_mm256_setzero_pd();
      for(int j=0;j<slen;j++){
        __m256d v=_mm256_loadu_pd(&sell_val[base+(size_t)j*SELL_C]);
        __m128i idx=_mm_loadu_si128((const __m128i*)&sell_col[base+(size_t)j*SELL_C]);
        __m128i idx0=_mm_add_epi32(idx,idx);
        __m128i idx1=_mm_add_epi32(idx0,_mm_set1_epi32(1));
        acc0=_mm256_fmadd_pd(v,_mm256_i32gather_pd(xp,idx0,8),acc0);
        acc1=_mm256_fmadd_pd(v,_mm256_i32gather_pd(xp,idx1,8),acc1);
      }
      double t0[4],t1[4]; _mm256_storeu_pd(t0,acc0); _mm256_storeu_pd(t1,acc1);
      for(int l=0;l<rows;l++){ yp[2*(size_t)(s*SELL_C+l)]=t0[l]; yp[2*(size_t)(s*SELL_C+l)+1]=t1[l]; }
#else
      double acc0[SELL_C]={0,0,0,0}, acc1[SELL_C]={0,0,0,0};
      for(int j=0;j<slen;j++) for(int l=0;l<SELL_C;l++){ uint32_t e=base+(uint32_t)j*SELL_C+(uint32_t)l; double v=sell_val[e]; const double *xv=&xp[2*(size_t)sell_col[e]]; acc0[l]+=v*xv[0]; acc1[l]+=v*xv[1]; }
      for(int l=0;l<rows;l++){ yp[2*(size_t)(s*SELL_C+l)]=acc0[l]; yp[2*(size_t)(s*SELL_C+l)+1]=acc1[l]; }
#endif
    }
  } else {
    const uint64_t *rp=p_row_ptr?p_row_ptr:row_ptr; const uint32_t *ci=p_row_ptr?p_col:col_idx; const double *av=p_row_ptr?p_val:a_val;
    for(int i=0;i<n_rows;i++){ double s0=0,s1=0; for(uint64_t k=rp[i];k<rp[i+1];k++){ double v=av[k]; const double *xv=&xp[2*(size_t)ci[k]]; s0+=v*xv[0]; s1+=v*xv[1]; } yp[2*(size_t)i]=s0; yp[2*(size_t)i+1]=s1; }
  }
}


/* Combined hi/lo SpMV: y = L*(xhi+xlo) with KBS compensation, one matrix pass. */
static void spmv_dd(const double *restrict xhi,const double *restrict xlo,double *restrict y,double *restrict ylo){
  if(use_sell){
    for(int s=0;s<sell_nslices;s++){
      uint32_t base=sell_ptr[s]; int slen=(int)(sell_ptr[s+1]-sell_ptr[s])/SELL_C;
      int rows=n_rows-s*SELL_C; if(rows>SELL_C) rows=SELL_C;
#ifdef __AVX2__
      __m256d acc=_mm256_setzero_pd(), comp=_mm256_setzero_pd(), acclo=_mm256_setzero_pd();
      const __m256d signm=_mm256_set1_pd(-0.0);
      for(int j=0;j<slen;j++){
        __m256d v=_mm256_loadu_pd(&sell_val[base+(size_t)j*SELL_C]);
        __m128i idx=_mm_loadu_si128((const __m128i*)&sell_col[base+(size_t)j*SELL_C]);
        __m256d xv=_mm256_i32gather_pd(xhi,idx,8);
        __m256d pr=_mm256_mul_pd(v,xv);
        __m256d pe=_mm256_fmsub_pd(v,xv,pr);
        __m256d tt=_mm256_add_pd(acc,pr);
        __m256d e1=_mm256_add_pd(_mm256_sub_pd(acc,tt),pr);
        __m256d e2=_mm256_add_pd(_mm256_sub_pd(pr,tt),acc);
        __m256d mask=_mm256_cmp_pd(_mm256_andnot_pd(signm,acc),_mm256_andnot_pd(signm,pr),_CMP_GE_OQ);
        comp=_mm256_add_pd(comp,_mm256_add_pd(_mm256_blendv_pd(e2,e1,mask),pe));
        acc=tt;
        acclo=_mm256_fmadd_pd(v,_mm256_i32gather_pd(xlo,idx,8),acclo);
      }
      double tmp[4],tc[4],tl[4]; _mm256_storeu_pd(tmp,acc); _mm256_storeu_pd(tc,comp); _mm256_storeu_pd(tl,acclo);
      for(int l=0;l<rows;l++){ y[s*SELL_C+l]=tmp[l]; if(ylo) ylo[s*SELL_C+l]=tc[l]+tl[l]; }
#else
      double acc[SELL_C]={0,0,0,0}, cmp[SELL_C]={0,0,0,0}, acclo[SELL_C]={0,0,0,0};
      for(int j=0;j<slen;j++) for(int l=0;l<SELL_C;l++){ uint32_t e=base+(uint32_t)j*SELL_C+(uint32_t)l; double v=sell_val[e]; uint32_t c=sell_col[e];
        double xv=xhi[c]; double pr=v*xv; double pe=fma(v,xv,-pr); double tt=acc[l]+pr; cmp[l]+=((fabs(acc[l])>=fabs(pr))?((acc[l]-tt)+pr):((pr-tt)+acc[l]))+pe; acc[l]=tt; acclo[l]+=v*xlo[c]; }
      for(int l=0;l<rows;l++){ y[s*SELL_C+l]=acc[l]; if(ylo) ylo[s*SELL_C+l]=cmp[l]+acclo[l]; }
#endif
    }
  } else {
    const uint64_t *rp=p_row_ptr?p_row_ptr:row_ptr; const uint32_t *ci=p_row_ptr?p_col:col_idx; const double *av=p_row_ptr?p_val:a_val;
    for(int i=0;i<n_rows;i++){ double sm=0,cp=0,slo=0;
      for(uint64_t k=rp[i];k<rp[i+1];k++){ double v=av[k]; uint32_t c=ci[k];
        double pr=v*xhi[c]; double tt=sm+pr; cp+=((fabs(sm)>=fabs(pr))?((sm-tt)+pr):((pr-tt)+sm)); sm=tt; slo+=v*xlo[c]; }
      y[i]=sm; if(ylo) ylo[i]=cp+slo; }
  }
}
static void spmv_pair_dd(const double *restrict xhi,const double *restrict xlo,double *restrict yp,double *restrict yplo){
  if(use_sell){
    for(int s=0;s<sell_nslices;s++){
      uint32_t base=sell_ptr[s]; int slen=(int)(sell_ptr[s+1]-sell_ptr[s])/SELL_C;
      int rows=n_rows-s*SELL_C; if(rows>SELL_C) rows=SELL_C;
#ifdef __AVX2__
      __m256d acc0=_mm256_setzero_pd(), acc1=_mm256_setzero_pd(), cp0=_mm256_setzero_pd(), cp1=_mm256_setzero_pd(), al0=_mm256_setzero_pd(), al1=_mm256_setzero_pd();
      const __m256d signm=_mm256_set1_pd(-0.0);
      for(int j=0;j<slen;j++){
        __m256d v=_mm256_loadu_pd(&sell_val[base+(size_t)j*SELL_C]);
        __m128i idx=_mm_loadu_si128((const __m128i*)&sell_col[base+(size_t)j*SELL_C]);
        __m128i idx0=_mm_add_epi32(idx,idx);
        __m128i idx1=_mm_add_epi32(idx0,_mm_set1_epi32(1));
        __m256d xv0=_mm256_i32gather_pd(xhi,idx0,8);
        __m256d xv1=_mm256_i32gather_pd(xhi,idx1,8);
        __m256d pr0=_mm256_mul_pd(v,xv0); __m256d pe0=_mm256_fmsub_pd(v,xv0,pr0);
        __m256d tt0=_mm256_add_pd(acc0,pr0);
        __m256d e10=_mm256_add_pd(_mm256_sub_pd(acc0,tt0),pr0);
        __m256d e20=_mm256_add_pd(_mm256_sub_pd(pr0,tt0),acc0);
        __m256d m0=_mm256_cmp_pd(_mm256_andnot_pd(signm,acc0),_mm256_andnot_pd(signm,pr0),_CMP_GE_OQ);
        cp0=_mm256_add_pd(cp0,_mm256_add_pd(_mm256_blendv_pd(e20,e10,m0),pe0)); acc0=tt0;
        __m256d pr1=_mm256_mul_pd(v,xv1); __m256d pe1=_mm256_fmsub_pd(v,xv1,pr1);
        __m256d tt1=_mm256_add_pd(acc1,pr1);
        __m256d e11=_mm256_add_pd(_mm256_sub_pd(acc1,tt1),pr1);
        __m256d e21=_mm256_add_pd(_mm256_sub_pd(pr1,tt1),acc1);
        __m256d m1=_mm256_cmp_pd(_mm256_andnot_pd(signm,acc1),_mm256_andnot_pd(signm,pr1),_CMP_GE_OQ);
        cp1=_mm256_add_pd(cp1,_mm256_add_pd(_mm256_blendv_pd(e21,e11,m1),pe1)); acc1=tt1;
        al0=_mm256_fmadd_pd(v,_mm256_i32gather_pd(xlo,idx0,8),al0);
        al1=_mm256_fmadd_pd(v,_mm256_i32gather_pd(xlo,idx1,8),al1);
      }
      double t0[4],t1[4],c0[4],c1[4],a0[4],a1[4]; _mm256_storeu_pd(t0,acc0); _mm256_storeu_pd(t1,acc1); _mm256_storeu_pd(c0,cp0); _mm256_storeu_pd(c1,cp1); _mm256_storeu_pd(a0,al0); _mm256_storeu_pd(a1,al1);
      for(int l=0;l<rows;l++){ yp[2*(size_t)(s*SELL_C+l)]=t0[l]; yp[2*(size_t)(s*SELL_C+l)+1]=t1[l]; if(yplo){ yplo[2*(size_t)(s*SELL_C+l)]=c0[l]+a0[l]; yplo[2*(size_t)(s*SELL_C+l)+1]=c1[l]+a1[l]; } }
#else
      double acc0[SELL_C]={0,0,0,0}, acc1[SELL_C]={0,0,0,0}, cp0[SELL_C]={0,0,0,0}, cp1[SELL_C]={0,0,0,0};
      double al0[SELL_C]={0,0,0,0}, al1[SELL_C]={0,0,0,0};
      for(int j=0;j<slen;j++) for(int l=0;l<SELL_C;l++){ uint32_t e=base+(uint32_t)j*SELL_C+(uint32_t)l; double v=sell_val[e]; const double *xb=&xhi[2*(size_t)sell_col[e]]; const double *xl=&xlo[2*(size_t)sell_col[e]];
        double pr0=v*xb[0]; double pe0=fma(v,xb[0],-pr0); double t0=acc0[l]+pr0; cp0[l]+=((fabs(acc0[l])>=fabs(pr0))?((acc0[l]-t0)+pr0):((pr0-t0)+acc0[l]))+pe0; acc0[l]=t0; double pr1=v*xb[1]; double pe1=fma(v,xb[1],-pr1); double t1=acc1[l]+pr1; cp1[l]+=((fabs(acc1[l])>=fabs(pr1))?((acc1[l]-t1)+pr1):((pr1-t1)+acc1[l]))+pe1; acc1[l]=t1; al0[l]+=v*xl[0]; al1[l]+=v*xl[1]; }
      for(int l=0;l<rows;l++){ yp[2*(size_t)(s*SELL_C+l)]=acc0[l]; yp[2*(size_t)(s*SELL_C+l)+1]=acc1[l]; if(yplo){ yplo[2*(size_t)(s*SELL_C+l)]=cp0[l]+al0[l]; yplo[2*(size_t)(s*SELL_C+l)+1]=cp1[l]+al1[l]; } }
#endif
    }
  } else {
    const uint64_t *rp=p_row_ptr?p_row_ptr:row_ptr; const uint32_t *ci=p_row_ptr?p_col:col_idx; const double *av=p_row_ptr?p_val:a_val;
    for(int i=0;i<n_rows;i++){ double s0=0,s1=0,c0=0,c1=0,a0=0,a1=0;
      for(uint64_t k=rp[i];k<rp[i+1];k++){ double v=av[k]; const double *xb=&xhi[2*(size_t)ci[k]]; const double *xl=&xlo[2*(size_t)ci[k]];
        double pr0=v*xb[0]; double t0=s0+pr0; c0+=((fabs(s0)>=fabs(pr0))?((s0-t0)+pr0):((pr0-t0)+s0)); s0=t0; double pr1=v*xb[1]; double t1=s1+pr1; c1+=((fabs(s1)>=fabs(pr1))?((s1-t1)+pr1):((pr1-t1)+s1)); s1=t1; a0+=v*xl[0]; a1+=v*xl[1]; }
      yp[2*(size_t)i]=s0; yp[2*(size_t)i+1]=s1; if(yplo){ yplo[2*(size_t)i]=c0+a0; yplo[2*(size_t)i+1]=c1+a1; } }
  }
}

/* ---------- coefficients ---------- */
static void free_coeffs(void){ free(q_coeff); free(f_coeff); free(e_ratio); free(q_tail); free(f_tail); q_coeff=f_coeff=e_ratio=NULL; q_tail=f_tail=NULL; }
static void make_coefficients(double t){
  free_coeffs();
  long double aa=(long double)t*(long double)lambda_upper*0.5L;
  int top=(int)ceill(12.0L*sqrtl(aa+1.0L)+128.0L);
  if(top<160) top=160;
  if(top>MAX_DEGREE-2) die("spectral interval exceeds coefficient capacity");
  coeff_top=top;
  long double *ratio=xmalloc((size_t)(top+1)*sizeof *ratio);
  long double *e=xmalloc((size_t)(top+1)*sizeof *e);
  long double *f=xmalloc((size_t)(top+1)*sizeof *f);
  long double *q=xmalloc((size_t)(top+1)*sizeof *q);
  long double *qt=xmalloc((size_t)(top+2)*sizeof *qt);
  long double *ft=xmalloc((size_t)(top+2)*sizeof *ft);
  ratio[top]=0.0L;
  for(int k=top;k>=1;--k) ratio[k-1]=1.0L/(2.0L*(long double)k/aa+ratio[k]);
  e[0]=1.0L; long double norm=1.0L;
  for(int k=1;k<=top;k++){ e[k]=e[k-1]*ratio[k-1]; norm+=2.0L*e[k]; }
  for(int k=0;k<=top;k++) e[k]/=norm;
  f[0]=e[0]; for(int k=1;k<=top;k++) f[k]=(k&1?-2.0L:2.0L)*e[k];
  memset(q,0,(size_t)(top+1)*sizeof *q);
  long double qn=0,qh=0;
  for(int j=top;j>=2;--j){ long double rj=-f[j]/aa; long double qp=2.0L*(rj-qh)-qn; q[j-1]=qp; qn=qh; qh=qp; }
  q[0]=-f[1]/aa-q[1]-0.5L*q[2];
  q_coeff=xmalloc((size_t)(top+1)*sizeof *q_coeff);
  f_coeff=xmalloc((size_t)(top+1)*sizeof *f_coeff);
  e_ratio=xmalloc((size_t)(top+1)*sizeof *e_ratio);
  for(int k=0;k<=top;k++){ q_coeff[k]=(double)((long double)t*q[k]); f_coeff[k]=(double)f[k]; e_ratio[k]=(k<top&&e[k]!=0.0L)?(double)(e[k+1]/e[k]):0.0; }
  qt[top+1]=0; ft[top+1]=0;
  for(int k=top;k>=0;--k){ qt[k]=qt[k+1]+(long double)t*fabsl(q[k]); ft[k]=ft[k+1]+fabsl(f[k]); }
  q_tail=qt; f_tail=ft;
  free(ratio); free(e); free(f); free(q);
}
static int choose_degree_s(long double scale,long double b0n,long double b1n,double atol,double rtol,long double S){
  long double budget=((long double)atol+(long double)rtol*scale)/S;
  for(int d=0;d<coeff_top;d++) if(f_tail[d+1]*b0n+q_tail[d+1]*b1n<=budget) return d;
  return coeff_top;
}
static int choose_degree(long double scale,long double b0n,long double b1n,double atol,double rtol){
  return choose_degree_s(scale,b0n,b1n,atol,rtol,SAFETY_FACTOR);
}

/* ---------- Lanczos ---------- */
static double lanczos_bound(double *row_sum_out_maxrow, int *maxrow_out){
  /* caller computed lambda_gersh/lambda_lower; returns candidate or gersh */
  int n=n_rows; int mmax=24;
  double *V=xcalloc((size_t)mmax+1,(size_t)n*sizeof(double)/sizeof(double)); /* actually bytes below */
  /* xcalloc above mis-sized if sizeof; redo properly: */
  free(V); V=xmalloc((size_t)(mmax+1)*(size_t)n*sizeof(double));
  double *w=xmalloc((size_t)n*sizeof(double));
  double *vcur=V, *vprev=NULL;
  /* deterministic permutation-invariant start: sign/weight from row Gershgorin data */
  double *rowsum=xmalloc((size_t)n*sizeof(double));
  int imax=0; long double best=-1;
  for(int i=0;i<n;i++){
    long double rs=0, diag=0, rad=0;
    for(uint64_t k=row_ptr[i];k<row_ptr[i+1];k++){ if(col_idx[k]==(uint32_t)i) diag=(long double)a_val[k]; else rad+=fabsl((long double)a_val[k]); rs+=(long double)a_val[k]; }
    rowsum[i]=(double)rs;
    long double edge=diag+rad; if(edge>best){best=edge;imax=i;}
    (void)row_sum_out_maxrow;
  }
  *maxrow_out=imax;
  long double nrm=0;
  for(int i=0;i<n;i++){
    uint64_t h=splitmix64((uint64_t)llrint(rowsum[i]*1e6) ^ ((uint64_t)(row_ptr[i+1]-row_ptr[i])<<32) ^ splitmix64((uint64_t)llrint(fabs(rowsum[i])*1e3+1.0)));
    double sgn=(h&1)?1.0:-1.0;
    double wt=sgn*(0.5+fabs(rowsum[i]));
    V[i]=wt; nrm+=(long double)wt*wt;
  }
  V[imax]+= (V[imax]>=0?2.0:-2.0)*(1.0+fabs(rowsum[imax]));
  nrm=0; for(int i=0;i<n;i++) nrm+=(long double)V[i]*V[i];
  nrm=sqrtl(nrm); for(int i=0;i<n;i++) V[i]/=(double)nrm;
  double alpha[24]={0}, beta[24]={0};
  double theta_hist[24]={0};
  double result=lambda_gersh;
  int done_m=0; double final_rho=0, final_beta=0, final_theta=0;
  /* Jacobi eigenvalue helper via tridiagonal stored each iteration: recompute with simple QL-like Jacobi on dense T */
  for(int m=1;m<=mmax;m++){
    double *vm = V+(size_t)(m-1)*n;
    spmv_orig(vm,w);
    double al=0; for(int i=0;i<n;i++) al+=vm[i]*w[i];
    alpha[m-1]=al;
    for(int i=0;i<n;i++){ double val=w[i]-al*vm[i]; if(m>=2) val-=beta[m-2]*V[(size_t)(m-2)*n+i]; w[i]=val; }
    /* full reorthogonalization (twice) */
    for(int pass=0;pass<2;pass++) for(int j=0;j<m;j++){ double d=0; double *vj=V+(size_t)j*n; for(int i=0;i<n;i++) d+=vj[i]*w[i]; for(int i=0;i<n;i++) w[i]-=d*vj[i]; }
    double bn=0; for(int i=0;i<n;i++) bn+=(long double)w[i]*w[i];
    bn=(double)sqrtl((long double)bn); if(m<24) beta[m-1]=bn;
    /* build T_m dense and Jacobi largest eigenpair */
    int mm=m; double *T=xcalloc((size_t)mm*mm,sizeof(double)); double *evec=xmalloc((size_t)mm*mm*sizeof(double));
    for(int i=0;i<mm;i++){ for(int j=0;j<mm;j++) evec[(size_t)i*mm+j]=(i==j)?1.0:0.0; T[(size_t)i*mm+i]=alpha[i]; if(i+1<mm){ T[(size_t)i*mm+i+1]=beta[i]; T[(size_t)(i+1)*mm+i]=beta[i]; } }
    for(int sweep=0;sweep<100;sweep++){ double off=0; for(int i=0;i<mm;i++) for(int j=i+1;j<mm;j++) off+=T[(size_t)i*mm+j]*T[(size_t)i*mm+j]; if(off<1e-30) break;
      for(int p=0;p<mm;p++) for(int qq=p+1;qq<mm;qq++){ double apq=T[(size_t)p*mm+qq]; if(apq==0) continue; double app=T[(size_t)p*mm+p], aqq=T[(size_t)qq*mm+qq]; double tau=(aqq-app)/(2*apq); double tt=(tau>=0?1.0:-1.0)/(fabs(tau)+sqrt(1+tau*tau)); double c=1/sqrt(1+tt*tt), s=tt*c;
        for(int k=0;k<mm;k++){ double akp=T[(size_t)k*mm+p], akq=T[(size_t)k*mm+qq]; T[(size_t)k*mm+p]=c*akp-s*akq; T[(size_t)k*mm+qq]=s*akp+c*akq; }
        for(int k=0;k<mm;k++){ double apk=T[(size_t)p*mm+k], aqk=T[(size_t)qq*mm+k]; T[(size_t)p*mm+k]=c*apk-s*aqk; T[(size_t)qq*mm+k]=s*apk+c*aqk; }
        for(int k=0;k<mm;k++){ double vkp=evec[(size_t)k*mm+p], vkq=evec[(size_t)k*mm+qq]; evec[(size_t)k*mm+p]=c*vkp-s*vkq; evec[(size_t)k*mm+qq]=s*vkp+c*vkq; } } }
    int bi=0; for(int i=1;i<mm;i++) if(T[(size_t)i*mm+i]>T[(size_t)bi*mm+bi]) bi=i;
    double theta=T[(size_t)bi*mm+bi];
    double lastcomp=fabs(evec[(size_t)(mm-1)*mm+bi]);
    free(T); free(evec);
    theta_hist[m-1]=theta; done_m=m; final_theta=theta; final_beta=bn; final_rho=bn*lastcomp;
    if(m>=10 && final_rho<=1e-3*theta && (theta-theta_hist[m-4])<=1e-3*theta) break;
    if(bn==0.0) break;
    if(m<mmax){ double *vn=V+(size_t)m*n; for(int i=0;i<n;i++) vn[i]=w[i]/bn; vcur=vn; }
    (void)vprev; (void)vcur;
  }
  if(done_m>=4){ /* need theta_{m-3}: index done_m-3 (1-based step done_m-2?) use hist[done_m-4] guard */
    double tm3 = (done_m-3>=1)? theta_hist[done_m-4] : theta_hist[0];
    if(final_theta>=lambda_lower && final_rho<=0.05*final_theta && (final_theta-tm3)<=0.02*final_theta){
      double cand=1.05*(final_theta+final_rho+0.25*final_beta);
      if(cand<result) result=cand;
      if(result<lambda_lower) result=lambda_lower; /* safety: cannot be below lower bound */
    }
  }
  free(V); free(w); free(rowsum);
  return result;
}

/* ---------- RCM ---------- */
static int cmp_deg_idx(const void *a,const void *b,void *ctx){ (void)ctx; return 0; }
typedef struct { int idx; uint64_t deg; } DegIdx;
static int cmp_degidx(const void *a,const void *b){ const DegIdx *x=a,*y=b; if(x->deg!=y->deg) return x->deg<y->deg?-1:1; return x->idx<y->idx?-1:(x->idx>y->idx); }

static int *compute_rcm(void){
  int n=n_rows;
  int *order=xmalloc((size_t)n*sizeof(int));
  char *visited=xcalloc((size_t)n,1);
  int *queue=xmalloc((size_t)n*sizeof(int));
  int olen=0;
  /* neighbors sorted by degree: build sorted adjacency copy */
  uint32_t *sadj=xmalloc((size_t)(nnz_g?nnz_g:1)*sizeof(uint32_t));
  memcpy(sadj,col_idx,(size_t)nnz_g*sizeof(uint32_t));
  DegIdx *tmp=xmalloc((size_t)(1024)*sizeof(DegIdx));
  size_t tmpcap=1024;
  for(int i=0;i<n;i++){
    uint64_t b=row_ptr[i], e=row_ptr[i+1]; size_t cnt=(size_t)(e-b);
    if(cnt>tmpcap){ tmpcap=cnt*2; tmp=realloc(tmp,tmpcap*sizeof(DegIdx)); if(!tmp) die("oom"); }
    size_t m=0; for(uint64_t k=b;k<e;k++){ uint32_t c=col_idx[k]; if(c==(uint32_t)i) continue; tmp[m].idx=(int)c; tmp[m].deg=row_ptr[c+1]-row_ptr[c]; m++; }
    qsort(tmp,m,sizeof(DegIdx),cmp_degidx);
    for(size_t j=0;j<m;j++) sadj[b+j]= (uint32_t)tmp[j].idx;
    /* keep any diagonal entries after: fill remainder with original tail if cnt>m (diag removed) - append diag at end */
    if(cnt>m){ sadj[b+m]=(uint32_t)i; }
  }
  free(tmp);
  /* per component */
  while(olen<n){
    int start=-1; uint64_t bestdeg=UINT64_MAX;
    for(int i=0;i<n;i++) if(!visited[i]){ uint64_t d=row_ptr[i+1]-row_ptr[i]; if(d<bestdeg){bestdeg=d;start=i;} }
    /* 2-sweep pseudo-peripheral */
    int root=start;
    for(int sweep=0;sweep<2;sweep++){
      /* BFS levels from root within unvisited */
      int qh=0,qt=0; queue[qt++]=root; char *seen=xcalloc((size_t)n,1); seen[root]=1; int *level=xcalloc((size_t)n,sizeof(int));
      int far=root;
      while(qh<qt){ int u=queue[qh++]; far=u; for(uint64_t k=row_ptr[u];k<row_ptr[u+1];k++){ uint32_t c=sadj[k]; if(c==(uint32_t)u||visited[c]||seen[c]) continue; seen[c]=1; level[c]=level[u]+1; queue[qt++]=(int)c; } }
      /* farthest with min degree */
      int maxlev=0; for(int i=0;i<qt;i++) if(level[queue[i]]>maxlev) maxlev=level[queue[i]];
      uint64_t bd=UINT64_MAX; int cand=root;
      for(int i=0;i<qt;i++){ int u=queue[i]; if(level[u]==maxlev){ uint64_t d=row_ptr[u+1]-row_ptr[u]; if(d<bd){bd=d;cand=u;} } }
      root=cand; free(seen); free(level);
    }
    /* CM BFS from root */
    int qh=0,qt=0, comp_start=olen;
    queue[qt++]=root; visited[root]=1;
    while(qh<qt){ int u=queue[qh++]; order[olen++]=u; for(uint64_t k=row_ptr[u];k<row_ptr[u+1];k++){ uint32_t c=sadj[k]; if(c==(uint32_t)u||visited[c]) continue; visited[c]=1; queue[qt++]=(int)c; } }
    /* reverse component */
    for(int i=comp_start,j=olen-1;i<j;i++,j--){ int t=order[i]; order[i]=order[j]; order[j]=t; }
  }
  free(visited); free(queue); free(sadj);
  (void)cmp_deg_idx;
  return order;
}

/* ---------- session build ---------- */
static void free_format(void){
  free(perm_new2old); free(perm_old2new); perm_new2old=perm_old2new=NULL;
  free(p_row_ptr); free(p_col); free(p_val); p_row_ptr=NULL; p_col=NULL; p_val=NULL;
  free(sell_ptr); free(sell_col); free(sell_val); sell_ptr=NULL; sell_col=NULL; sell_val=NULL; use_sell=0; sell_nslices=0;
}
static double avg_jump_for_perm(const int *new2old, const int *old2new){
  if(nnz_g==0) return 0;
  long double s=0; (void)new2old;
  for(int i=0;i<n_rows;i++){ int pi=old2new?old2new[i]:i; for(uint64_t k=row_ptr[i];k<row_ptr[i+1];k++){ int pj=old2new?old2new[(int)col_idx[k]]:(int)col_idx[k]; s+=fabsl((long double)pi-pj); } }
  return (double)(s/(long double)nnz_g);
}

static void build_session(void){
  free_format(); free_coeffs();
  /* Gershgorin + lower + avg_jump */
  long double bound=0; uint64_t maxrow=0; long double lower=0;
  for(int i=0;i<n_rows;i++){
    long double rad=0,diag=0,ss=0; uint64_t rn=row_ptr[i+1]-row_ptr[i]; if(rn>maxrow) maxrow=rn;
    for(uint64_t k=row_ptr[i];k<row_ptr[i+1];k++){ long double v=(long double)a_val[k]; if(col_idx[k]==(uint32_t)i) diag=v; else rad+=fabsl(v); ss+=v*v; }
    /* include diag in ss: recompute ss included it above (loop includes diag) */
    long double edge=diag+rad; if(edge>bound) bound=edge;
    long double rownorm=sqrtl(ss); long double lb=diag>rownorm?diag:rownorm; if(lb>lower) lower=lb;
  }
  if(bound<0) bound=0;
  lambda_gersh = (bound==0)?0.0: nextafter((double)bound,INFINITY)*(1.0+(double)(8+maxrow)*DBL_EPSILON);
  lambda_lower=(double)lower;
  lambda_upper=lambda_gersh;
  double a_gersh = (t_g*lambda_gersh)/2.0;
  int use_lanczos = !dis_lanczos && t_g>0 && n_rows>=64 && lambda_gersh>0 && lambda_lower < 0.92*lambda_gersh && (double)n_requests_g*sqrt(a_gersh>=0?a_gersh:0) >= 8.0;
  if(use_lanczos){ int mr=0; double cand=lanczos_bound(NULL,&mr); if(cand<lambda_gersh && cand>=lambda_lower) lambda_upper=cand; }
  if(t_g!=0 && lambda_upper!=0) make_coefficients(t_g);
  /* avg_jump */
  double avg_jump = avg_jump_for_perm(NULL,NULL);
  /* RCM decision */
  int *rcm_order=NULL; /* new2old */
  if(!dis_rcm && n_rows>=4096 && avg_jump>0.04*(double)n_rows && (double)n_requests_g*sqrt(a_gersh>=0?a_gersh:0)>=25.0){
    rcm_order=compute_rcm();
    int *o2n=xmalloc((size_t)n_rows*sizeof(int));
    for(int i=0;i<n_rows;i++) o2n[rcm_order[i]]=i;
    double aj2=avg_jump_for_perm(rcm_order,o2n);
    if(!(aj2 < 0.70*avg_jump)){ free(rcm_order); rcm_order=NULL; }
    free(o2n);
  }
  /* base perm = rcm_order or identity */
  int *base_new2old = rcm_order;
  /* Figure final perm: if sell, apply sigma window stable sort by row length */
  use_sell = !dis_sell && t_g!=0; /* sell irrelevant for t=0 but build anyway? keep; if nnz=0 slices still fine */
  if(!use_sell && !base_new2old){
    /* original CSR direct: p_* stays NULL */
    return;
  }
  /* build final new2old */
  int *final_new2old=xmalloc((size_t)n_rows*sizeof(int));
  if(base_new2old) memcpy(final_new2old,base_new2old,(size_t)n_rows*sizeof(int));
  else for(int i=0;i<n_rows;i++) final_new2old[i]=i;
  free(base_new2old);
  if(use_sell){
    /* sigma stable sort within windows by row length descending (of original rows) */
    for(int ws=0; ws<n_rows; ws+=SELL_SIGMA){
      int we=ws+SELL_SIGMA; if(we>n_rows) we=n_rows;
      /* insertion stable sort */
      for(int i=ws+1;i<we;i++){ int v=final_new2old[i]; uint64_t lv=row_ptr[v+1]-row_ptr[v]; int j=i-1; while(j>=ws){ uint64_t lj=row_ptr[final_new2old[j]+1]-row_ptr[final_new2old[j]]; if(lj>=lv) break; final_new2old[j+1]=final_new2old[j]; j--; } final_new2old[j+1]=v; }
    }
  }
  perm_new2old=final_new2old;
  perm_old2new=xmalloc((size_t)n_rows*sizeof(int));
  for(int i=0;i<n_rows;i++) perm_old2new[final_new2old[i]]=i;
  if(use_sell){
    sell_nslices=(n_rows+SELL_C-1)/SELL_C;
    sell_ptr=xmalloc((size_t)(sell_nslices+1)*sizeof(uint32_t));
    sell_ptr[0]=0;
    for(int s=0;s<sell_nslices;s++){ int mx=0; for(int l=0;l<SELL_C;l++){ int p=s*SELL_C+l; if(p>=n_rows) break; int o=final_new2old[p]; int len=(int)(row_ptr[o+1]-row_ptr[o]); if(len>mx) mx=len; } sell_ptr[s+1]=sell_ptr[s]+(uint32_t)(mx*SELL_C); }
    uint32_t total=sell_ptr[sell_nslices];
    sell_col=xcalloc(total?total:1,sizeof(uint32_t));
    sell_val=xcalloc(total?total:1,sizeof(double));
    for(int s=0;s<sell_nslices;s++){ uint32_t base=sell_ptr[s]; int mx=(int)(sell_ptr[s+1]-sell_ptr[s])/SELL_C;
      for(int l=0;l<SELL_C;l++){ int p=s*SELL_C+l; if(p>=n_rows){ for(int j=0;j<mx;j++){ sell_col[base+(uint32_t)j*SELL_C+(uint32_t)l]=(uint32_t)(n_rows-1); sell_val[base+(uint32_t)j*SELL_C+(uint32_t)l]=0; } continue; }
        int o=final_new2old[p]; int j=0; for(uint64_t k=row_ptr[o];k<row_ptr[o+1];k++,j++){ sell_col[base+(uint32_t)j*SELL_C+(uint32_t)l]=(uint32_t)perm_old2new[(int)col_idx[k]]; sell_val[base+(uint32_t)j*SELL_C+(uint32_t)l]=a_val[k]; } } }
  } else {
    /* permuted CSR */
    p_row_ptr=xmalloc((size_t)(n_rows+1)*sizeof(uint64_t));
    p_col=xmalloc((size_t)(nnz_g?nnz_g:1)*sizeof(uint32_t));
    p_val=xmalloc((size_t)(nnz_g?nnz_g:1)*sizeof(double));
    p_row_ptr[0]=0;
    for(int p=0;p<n_rows;p++){ int o=final_new2old[p]; p_row_ptr[p+1]=p_row_ptr[p]+(row_ptr[o+1]-row_ptr[o]); }
    for(int p=0;p<n_rows;p++){ int o=final_new2old[p]; uint64_t dst=p_row_ptr[p]; for(uint64_t k=row_ptr[o];k<row_ptr[o+1];k++,dst++){ p_col[dst]=(uint32_t)perm_old2new[(int)col_idx[k]]; p_val[dst]=a_val[k]; } }
  }
}

/* ---------- solver (permuted space) ---------- */
static double *wk_vprev,*wk_vcur,*wk_vnext,*wk_tmp,*wk_y,*wk_comp,*wk_pair_prev,*wk_pair_cur,*wk_pair_next;
static void alloc_work(void){
  size_t n=(size_t)n_rows;
  wk_vprev=xmalloc(n*sizeof(double)); wk_vcur=xmalloc(n*sizeof(double)); wk_vnext=xmalloc(n*sizeof(double)); wk_tmp=xmalloc(n*sizeof(double)); wk_y=xmalloc(n*sizeof(double)); wk_comp=xmalloc(n*sizeof(double));
  wk_pair_prev=xmalloc(2*n*sizeof(double)); wk_pair_cur=xmalloc(2*n*sizeof(double)); wk_pair_next=xmalloc(2*n*sizeof(double));
}
static double vec_inf(const double *x){ double m=0; for(int i=0;i<n_rows;i++){ double v=fabs(x[i]); if(v>m)m=v; } return m; }
static long double vec_2(const double *x){ long double s=0; for(int i=0;i<n_rows;i++) s+=(long double)x[i]*x[i]; return sqrtl(s); }

/* returns 1 if guard triggered (iterate too large) */
static int solve_plain(double atol,double rtol,const double *b0,const double *b1,double *yout,int *guard){
  *guard=0;
  int n=n_rows;
  if(t_g==0.0 || lambda_upper==0.0){ for(int i=0;i<n;i++) yout[i]=b0[i]+t_g*b1[i]; return 0; }
  int has0=0,has1=0; for(int i=0;i<n;i++){ if(b0[i]!=0) has0=1; if(b1[i]!=0) has1=1; }
  long double b0n=vec_2(b0), b1n=vec_2(b1);
  double alpha=2.0/lambda_upper;
  double *y=wk_y; for(int i=0;i<n;i++){ y[i]=0; wk_comp[i]=0; }
  /* target degree for non-adaptive */
  long double scale0=0; for(int i=0;i<n;i++){ long double s0=fabsl((long double)b0[i]), s1=fabsl((long double)t_g*(long double)b1[i]); if(s0>scale0)scale0=s0; if(s1>scale0)scale0=s1; }
  int target = dis_adaptive ? choose_degree_s(scale0,b0n,b1n,atol,rtol,64.0L) : coeff_top;
  int both = has0&&has1;
  int fused = !dis_fused && both;
  /* histories for adaptive */
  double hist_inf0[4]={0,0,0,0}, hist_inf1[4]={0,0,0,0};
  double prev_delta=0; int consec=0;
  /* init v_cur = b (k=0) */
  if(fused){ for(int i=0;i<n;i++){ wk_pair_cur[2*(size_t)i]=b0[i]; wk_pair_cur[2*(size_t)i+1]=b1[i]; } }
  else { memcpy(wk_vcur,b0,(size_t)n*sizeof(double)); /* for single/separate we maintain two singles in separate passes below if needed */ }
  /* We implement loop generically:
     fused: pair arrays; separate-both: two single arrays reusing vprev/vcur plus extra? Need second set.
     Allocate static second set on fly using pair arrays as storage for separate singles (first halves). Simpler:
     separate mode uses wk_pair_prev/cur/next as two independent single vectors stored in [0..n) and [n..2n). */
  double *s0_prev=NULL,*s0_cur=NULL,*s0_next=NULL,*s1_prev=NULL,*s1_cur=NULL,*s1_next=NULL;
  if(!fused){
    s0_cur=wk_pair_cur; s0_prev=wk_pair_prev; s0_next=wk_pair_next;
    s1_cur=wk_pair_cur+n; s1_prev=wk_pair_prev+n; s1_next=wk_pair_next+n;
    if(has0) memcpy(s0_cur,b0,(size_t)n*sizeof(double));
    if(has1) memcpy(s1_cur,b1,(size_t)n*sizeof(double));
  }
  for(int k=0;;k++){
    double cur_inf0=0,cur_inf1=0, delta_inf=0;
    if(k==0){
      if(fused){ for(int i=0;i<n;i++){ double v0=wk_pair_cur[2*(size_t)i], v1=wk_pair_cur[2*(size_t)i+1]; y[i]=f_coeff[0]*v0+q_coeff[0]*v1; double d=fabs(f_coeff[0]*v0+q_coeff[0]*v1); if(d>delta_inf)delta_inf=d; } cur_inf0=vec_inf(b0); cur_inf1=vec_inf(b1); /* approx via loops above fine */ 
        cur_inf0=0;cur_inf1=0; for(int i=0;i<n;i++){ double a=fabs(wk_pair_cur[2*(size_t)i]),b=fabs(wk_pair_cur[2*(size_t)i+1]); if(a>cur_inf0)cur_inf0=a; if(b>cur_inf1)cur_inf1=b; } }
      else { for(int i=0;i<n;i++){ double v0=has0?s0_cur[i]:0, v1=has1?s1_cur[i]:0; y[i]=(has0?f_coeff[0]*v0:0)+(has1?q_coeff[0]*v1:0); } if(has0)cur_inf0=vec_inf(s0_cur); if(has1)cur_inf1=vec_inf(s1_cur); delta_inf=vec_inf(y); }
    } else {
      /* compute v_k from v_{k-1} */
      if(fused){
        if(k==1){ spmv_pair_plain(wk_pair_cur,wk_pair_next); for(int i=0;i<2*n;i++) wk_pair_next[i]=alpha*wk_pair_next[i]-wk_pair_cur[i]; /* that gives Z b -? Wait: need v1=Z v0, alpha*L v0 - v0 : computed as alpha*(Lv)-v correct */ }
        else { spmv_pair_plain(wk_pair_cur,wk_pair_next); for(int i=0;i<2*n;i++) wk_pair_next[i]=2.0*(alpha*wk_pair_next[i]-wk_pair_cur[i])-wk_pair_prev[i]; /* 2*Z v - vprev =2(alpha Lv - v)-vprev */ }
        /* rotate: prev=cur, cur=next (swap pointers via memcpy? swap global pointers) */
        double *tmp=wk_pair_prev; wk_pair_prev=wk_pair_cur; wk_pair_cur=wk_pair_next; wk_pair_next=tmp;
        cur_inf0=0;cur_inf1=0;
        for(int i=0;i<n;i++){ double v0=wk_pair_cur[2*(size_t)i], v1=wk_pair_cur[2*(size_t)i+1]; double d=f_coeff[k]*v0+q_coeff[k]*v1; double yy=d-wk_comp[i]; double tt2=y[i]+yy; wk_comp[i]=(tt2-y[i])-yy; y[i]=tt2; double ad=fabs(d); if(ad>delta_inf)delta_inf=ad; double a=fabs(v0),b=fabs(v1); if(a>cur_inf0)cur_inf0=a; if(b>cur_inf1)cur_inf1=b; }
      } else {
        /* two separate SpMVs (or one if single) */
        if(has0){ if(k==1){ spmv_plain(s0_cur,wk_tmp); for(int i=0;i<n;i++) s0_next[i]=alpha*wk_tmp[i]-s0_cur[i]; } else { spmv_plain(s0_cur,wk_tmp); for(int i=0;i<n;i++) s0_next[i]=2.0*(alpha*wk_tmp[i]-s0_cur[i])-s0_prev[i]; } double *tt=s0_prev; s0_prev=s0_cur; s0_cur=s0_next; s0_next=tt; /* note wk_pair pointers changed? s pointers are offsets into pair buffers; swapping local pointers fine but they alias pair arrays distinct buffers */ }
        if(has1){ if(k==1){ spmv_plain(s1_cur,wk_tmp); for(int i=0;i<n;i++) s1_next[i]=alpha*wk_tmp[i]-s1_cur[i]; } else { spmv_plain(s1_cur,wk_tmp); for(int i=0;i<n;i++) s1_next[i]=2.0*(alpha*wk_tmp[i]-s1_cur[i])-s1_prev[i]; } double *tt=s1_prev; s1_prev=s1_cur; s1_cur=s1_next; s1_next=tt; }
        /* careful: s0_* and s1_* point into same 3 pair buffers at offsets 0 and n; swapping s0 pointers among buffers is ok, s1 similarly */
        for(int i=0;i<n;i++){ double d=(has0?f_coeff[k]*s0_cur[i]:0)+(has1?q_coeff[k]*s1_cur[i]:0); double yy=d-wk_comp[i]; double tt2=y[i]+yy; wk_comp[i]=(tt2-y[i])-yy; y[i]=tt2; double ad=fabs(d); if(ad>delta_inf)delta_inf=ad; }
        if(has0)cur_inf0=vec_inf(s0_cur); if(has1)cur_inf1=vec_inf(s1_cur);
      }
    }
    /* guard */
    double lim0=2.0*(double)b0n, lim1=2.0*(double)b1n;
    if((has0&&cur_inf0>lim0&&lim0>0)||(has1&&cur_inf1>lim1&&lim1>0)){ *guard=1; }
    /* histories */
    hist_inf0[0]=hist_inf0[1]; hist_inf0[1]=hist_inf0[2]; hist_inf0[2]=hist_inf0[3]; hist_inf0[3]=cur_inf0;
    hist_inf1[0]=hist_inf1[1]; hist_inf1[1]=hist_inf1[2]; hist_inf1[2]=hist_inf1[3]; hist_inf1[3]=cur_inf1;
    if(*guard) { /* still need copy y */ memcpy(yout,y,(size_t)n*sizeof(double)); return 1; }
    if(!dis_adaptive){
      if(k>=2 && k<coeff_top){
        double rk=e_ratio[k];
        if(rk<=0.9){
          double m0=hist_inf0[0],m1=hist_inf1[0]; for(int j=1;j<4;j++){ if(hist_inf0[j]>m0)m0=hist_inf0[j]; if(hist_inf1[j]>m1)m1=hist_inf1[j]; }
          double W0=2.0*m0; if(W0>(double)b0n) W0=(double)b0n; if(!has0)W0=0;
          double W1=2.0*m1; if(W1>(double)b1n) W1=(double)b1n; if(!has1)W1=0;
          long double tailE = f_tail[k+1]*(long double)W0 + q_tail[k+1]*(long double)W1;
          double geo = (rk/(1.0-rk)) * (delta_inf>prev_delta?delta_inf:prev_delta);
          long double Ek = tailE>(long double)geo?tailE:(long double)geo;
          double yinf=vec_inf(y);
          long double outscale = (long double)yinf - Ek; if(outscale<0) outscale=0;
          long double budget=((long double)atol+(long double)rtol*outscale)/SAFETY_FACTOR;
          /* certified gate: also full tail bound must satisfy budget with certified scale */
          if(Ek<=budget) consec++; else consec=0;
          if(consec>=2){ memcpy(yout,y,(size_t)n*sizeof(double)); return 0; }
        } else consec=0;
      }
      if(k>=coeff_top){ memcpy(yout,y,(size_t)n*sizeof(double)); return 0; }
    } else {
      if(k>=target){
        /* output-relative check, extend if needed */
        long double tail_bound=f_tail[k+1]* (has0?b0n:0) + q_tail[k+1]*(has1?b1n:0);
        double yinf=vec_inf(y);
        long double outscale=(long double)yinf-tail_bound; if(outscale<0)outscale=0;
        int corr=choose_degree_s(outscale,b0n,b1n,atol,rtol,64.0L);
        if(corr<=k || k>=coeff_top){ memcpy(yout,y,(size_t)n*sizeof(double)); return 0; }
        target=corr;
      }
      if(k>=coeff_top){ memcpy(yout,y,(size_t)n*sizeof(double)); return 0; }
    }
    prev_delta=delta_inf;
  }
}


static int solve_comp(double atol,double rtol,const double *b0,const double *b1,double *yout,int *guard);
static int solve_permuted(double atol,double rtol,const double *b0,const double *b1,double *yout,int *guard){
  if(t_g==0.0 || lambda_upper==0.0){ for(int i=0;i<n_rows;i++) yout[i]=b0[i]+t_g*b1[i]; *guard=0; return 0; }
  long double b0n=vec_2(b0), b1n=vec_2(b1);
  int est = coeff_top ? choose_degree(0.0L,b0n,b1n,atol,rtol) : 0;
  if(est<=512) return solve_plain(atol,rtol,b0,b1,yout,guard);
  return solve_comp(atol,rtol,b0,b1,yout,guard);
}

/* returns 1 if guard triggered (iterate too large) */
/* State is kept in long double; each SpMV is applied to the double hi/lo
   split of the state, so the forward recurrence rounding is compensated
   (Correction 1, see CALIBRATION.md). */
static long double *st_prev,*st_cur,*st_next;
static double *pair_lo,*pair_outB,*pair_outAlo;
static int solve_comp(double atol,double rtol,const double *b0,const double *b1,double *yout,int *guard){
  *guard=0;
  int n=n_rows;
  if(t_g==0.0 || lambda_upper==0.0){ for(int i=0;i<n;i++) yout[i]=b0[i]+t_g*b1[i]; return 0; }
  if(!st_prev){ st_prev=xmalloc(2*(size_t)n*sizeof(long double)); st_cur=xmalloc(2*(size_t)n*sizeof(long double)); st_next=xmalloc(2*(size_t)n*sizeof(long double)); pair_lo=xmalloc(2*(size_t)n*sizeof(double)); pair_outB=xmalloc(2*(size_t)n*sizeof(double)); pair_outAlo=xmalloc(2*(size_t)n*sizeof(double)); }
  int has0=0,has1=0; for(int i=0;i<n;i++){ if(b0[i]!=0) has0=1; if(b1[i]!=0) has1=1; }
  long double b0n=vec_2(b0), b1n=vec_2(b1);
  double alpha=2.0/lambda_upper;
  double *y=wk_y; for(int i=0;i<n;i++){ y[i]=0; wk_comp[i]=0; }
  long double scale0=0; for(int i=0;i<n;i++){ long double s0=fabsl((long double)b0[i]), s1=fabsl((long double)t_g*(long double)b1[i]); if(s0>scale0)scale0=s0; if(s1>scale0)scale0=s1; }
  int target = dis_adaptive ? choose_degree_s(scale0,b0n,b1n,atol,rtol,64.0L) : coeff_top;
  int both = has0&&has1;
  int fused = !dis_fused && both;
  double hist_inf0[4]={0,0,0,0}, hist_inf1[4]={0,0,0,0};
  double prev_delta=0; int consec=0;
  /* layout: fused -> st[2i]=c0, st[2i+1]=c1 ; separate -> st[i]=c0, st[n+i]=c1 */
  for(int i=0;i<2*n;i++){ st_cur[i]=0; st_prev[i]=0; }
  if(fused){ for(int i=0;i<n;i++){ st_cur[2*(size_t)i]=(long double)b0[i]; st_cur[2*(size_t)i+1]=(long double)b1[i]; } }
  else { if(has0) for(int i=0;i<n;i++) st_cur[i]=(long double)b0[i]; if(has1) for(int i=0;i<n;i++) st_cur[n+i]=(long double)b1[i]; }
  double *hi=wk_pair_prev, *outA=wk_pair_next; /* 2n scratch; wk_pair_cur used as ??? no: hi=wk_pair_cur */
  hi=wk_pair_cur;
  for(int k=0;;k++){
    double cur_inf0=0,cur_inf1=0, delta_inf=0;
    if(k==0){
      for(int i=0;i<n;i++){
        long double v0,v1;
        if(fused){ v0=st_cur[2*(size_t)i]; v1=st_cur[2*(size_t)i+1]; }
        else { v0=has0?st_cur[i]:0.0L; v1=has1?st_cur[n+i]:0.0L; }
        y[i]=(double)((long double)f_coeff[0]*v0+(long double)q_coeff[0]*v1);
        double a=(double)fabsl(v0), b=(double)fabsl(v1); if(a>cur_inf0)cur_inf0=a; if(b>cur_inf1)cur_inf1=b;
      }
      delta_inf=vec_inf(y);
    } else {
      /* split state into hi/lo doubles */
      for(int i=0;i<2*n;i++){ double h=(double)st_cur[i]; hi[i]=h; pair_lo[i]=(double)(st_cur[i]-(long double)h); }
      if(fused){ spmv_pair_dd(hi,pair_lo,outA,pair_outAlo); }
      else {
        if(has0){ spmv_dd(hi,pair_lo,outA,pair_outAlo); }
        if(has1){ spmv_dd(hi+n,pair_lo+n,outA+n,pair_outAlo+n); }
      }
      for(int i=0;i<2*n;i++){
        int active = fused || (i<n? has0: has1);
        if(!active){ st_next[i]=0; continue; }
        long double Lv=(long double)outA[i]+(long double)pair_outAlo[i];
        long double zv=(long double)alpha*Lv - st_cur[i];
        st_next[i] = (k==1)? zv : 2.0L*zv - st_prev[i];
      }
      long double *tmp=st_prev; st_prev=st_cur; st_cur=st_next; st_next=tmp;
      for(int i=0;i<n;i++){
        long double v0,v1;
        if(fused){ v0=st_cur[2*(size_t)i]; v1=st_cur[2*(size_t)i+1]; }
        else { v0=has0?st_cur[i]:0.0L; v1=has1?st_cur[n+i]:0.0L; }
        double d=(double)((long double)f_coeff[k]*v0+(long double)q_coeff[k]*v1);
        double yy=d-wk_comp[i]; double tt=y[i]+yy; wk_comp[i]=(tt-y[i])-yy; y[i]=tt;
        double ad=fabs(d); if(ad>delta_inf)delta_inf=ad;
        double a=(double)fabsl(v0), b=(double)fabsl(v1); if(a>cur_inf0)cur_inf0=a; if(b>cur_inf1)cur_inf1=b;
      }
    }
    double lim0=2.0*(double)b0n, lim1=2.0*(double)b1n;
    if((has0&&cur_inf0>lim0&&lim0>0)||(has1&&cur_inf1>lim1&&lim1>0)){ *guard=1; }
    hist_inf0[0]=hist_inf0[1]; hist_inf0[1]=hist_inf0[2]; hist_inf0[2]=hist_inf0[3]; hist_inf0[3]=cur_inf0;
    hist_inf1[0]=hist_inf1[1]; hist_inf1[1]=hist_inf1[2]; hist_inf1[2]=hist_inf1[3]; hist_inf1[3]=cur_inf1;
    if(*guard){ memcpy(yout,y,(size_t)n*sizeof(double)); return 1; }
    if(!dis_adaptive){
      if(k>=2 && k<coeff_top){
        double rk=e_ratio[k];
        if(rk<=0.9){
          double m0=hist_inf0[0],m1=hist_inf1[0]; for(int j=1;j<4;j++){ if(hist_inf0[j]>m0)m0=hist_inf0[j]; if(hist_inf1[j]>m1)m1=hist_inf1[j]; }
          double W0=2.0*m0; if(W0>(double)b0n) W0=(double)b0n; if(!has0)W0=0;
          double W1=2.0*m1; if(W1>(double)b1n) W1=(double)b1n; if(!has1)W1=0;
          long double tailE = f_tail[k+1]*(long double)W0 + q_tail[k+1]*(long double)W1;
          double geo = (rk/(1.0-rk)) * (delta_inf>prev_delta?delta_inf:prev_delta);
          long double Ek = tailE>(long double)geo?tailE:(long double)geo;
          double yinf=vec_inf(y);
          long double outscale = (long double)yinf - Ek; if(outscale<0) outscale=0;
          long double budget=((long double)atol+(long double)rtol*outscale)/SAFETY_FACTOR;
          if(Ek<=budget) consec++; else consec=0;
          if(consec>=2){ memcpy(yout,y,(size_t)n*sizeof(double)); return 0; }
        } else consec=0;
      }
      if(k>=coeff_top){ memcpy(yout,y,(size_t)n*sizeof(double)); return 0; }
    } else {
      if(k>=target){
        long double tail_bound=f_tail[k+1]* (has0?b0n:0) + q_tail[k+1]*(has1?b1n:0);
        double yinf=vec_inf(y);
        long double outscale=(long double)yinf-tail_bound; if(outscale<0)outscale=0;
        int corr=choose_degree_s(outscale,b0n,b1n,atol,rtol,64.0L);
        if(corr<=k || k>=coeff_top){ memcpy(yout,y,(size_t)n*sizeof(double)); return 0; }
        target=corr;
      }
      if(k>=coeff_top){ memcpy(yout,y,(size_t)n*sizeof(double)); return 0; }
    }
    prev_delta=delta_inf;
  }
}

int main(void){
  parse_disable();
  write_all("READY\n",6);
  char magic[8]; read_exact(magic,sizeof magic);
  if(memcmp(magic,"FAMEXP2\n",8)!=0) die("bad magic");
  uint32_t json_len; read_exact(&json_len,sizeof json_len);
  if(json_len==0||json_len>MAX_HEADER) die("bad header length");
  char json[MAX_HEADER+1]; read_exact(json,json_len); json[json_len]='\0';
  n_rows=(int)json_number(json,"n");
  nnz_g=(uint64_t)json_number(json,"nnz");
  n_requests_g=(int)json_number(json,"requests");
  t_g=json_number(json,"t");
  double rtol=json_number(json,"rtol"); double atol=json_number(json,"atol");
  if(n_rows<1||nnz_g>1000000||(n_requests_g!=1&&n_requests_g!=16)) die("invalid dimensions");
  row_ptr=xmalloc((size_t)(n_rows+1)*sizeof *row_ptr);
  col_idx=xmalloc((size_t)(nnz_g?nnz_g:1)*sizeof *col_idx);
  a_val=xmalloc((size_t)(nnz_g?nnz_g:1)*sizeof *a_val);
  read_exact(row_ptr,(size_t)(n_rows+1)*sizeof *row_ptr);
  read_exact(col_idx,(size_t)nnz_g*sizeof *col_idx);
  read_exact(a_val,(size_t)nnz_g*sizeof *a_val);
  build_session();
  alloc_work();
  size_t bytes=(size_t)n_rows*sizeof(double);
  double *b0=xmalloc(bytes),*b1=xmalloc(bytes),*y=xmalloc(bytes);
  double *b0p=xmalloc(bytes),*b1p=xmalloc(bytes),*yp=xmalloc(bytes);
  for(int r=0;r<n_requests_g;r++){
    read_exact(b0,bytes); read_exact(b1,bytes);
    if(dis_reuse && r>0){ build_session(); }
    /* degenerate handled inside permuted solver directly on permuted vectors */
    const double *ub0=b0,*ub1=b1;
    if(perm_new2old){ for(int p=0;p<n_rows;p++){ b0p[p]=b0[perm_new2old[p]]; b1p[p]=b1[perm_new2old[p]]; } ub0=b0p; ub1=b1p; }
    int guard=0;
    solve_permuted(atol,rtol,ub0,ub1,yp,&guard);
    if(guard && lambda_upper<lambda_gersh){
      /* fallback to Gershgorin: rebuild coefficients (and lambda), recompute.
         Format permutation unchanged (independent of lambda). */
      lambda_upper=lambda_gersh;
      if(t_g!=0&&lambda_upper!=0) make_coefficients(t_g);
      solve_permuted(atol,rtol,ub0,ub1,yp,&guard);
    }
    if(perm_new2old){ for(int p=0;p<n_rows;p++) y[perm_new2old[p]]=yp[p]; }
    else memcpy(y,yp,bytes);
    write_all(y,bytes);
  }
  return 0;
}
