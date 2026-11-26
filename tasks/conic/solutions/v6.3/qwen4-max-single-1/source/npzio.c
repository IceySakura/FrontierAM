#include "common.h"
#include "npzio.h"

unsigned char *read_file(const char *path,size_t *len){
    FILE *f=fopen(path,"rb"); if(!f) return NULL;
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    unsigned char *buf=xmalloc((size_t)sz+1);
    size_t rd=fread(buf,1,(size_t)sz,f); fclose(f);
    if(rd!=(size_t)sz){free(buf);return NULL;}
    *len=(size_t)sz; return buf;
}

/* ---- ZIP ---- */
static uint32_t rd32(const unsigned char *p){return p[0]|(p[1]<<8)|(p[2]<<16)|((uint32_t)p[3]<<24);}
static uint16_t rd16(const unsigned char *p){return p[0]|(p[1]<<8);}

typedef struct { char name[256]; int method; size_t csize,usize; size_t lho; } ZipEnt;

static int zip_entries(const unsigned char *buf,size_t len,ZipEnt **ents,int *nent){
    /* find EOCD */
    long eocd=-1;
    long start=(long)len-22; if(start<0)start=0;
    for(long i=(long)len-22;i>=start;i--){
        if(rd32(buf+i)==0x06054b50){eocd=i;break;}
    }
    if(eocd<0) return -1;
    int total=rd16(buf+eocd+10);
    size_t cd=rd32(buf+eocd+16);
    ZipEnt *es=xcalloc(total,sizeof(ZipEnt));
    size_t p=cd; int k=0;
    for(int e=0;e<total;e++){
        if(p+46>len||rd32(buf+p)!=0x02014b50){free(es);return -1;}
        int method=rd16(buf+p+10);
        size_t csize=rd32(buf+p+20),usize=rd32(buf+p+24);
        int nlen=rd16(buf+p+28),xlen=rd16(buf+p+30),clen=rd16(buf+p+32);
        size_t lho=rd32(buf+p+42);
        if(nlen>255)nlen=255;
        memcpy(es[k].name,buf+p+46,nlen); es[k].name[nlen]=0;
        es[k].method=method;es[k].csize=csize;es[k].usize=usize;es[k].lho=lho;
        k++;
        p+=46+nlen+xlen+clen;
    }
    *ents=es;*nent=k;return 0;
}

static unsigned char *zip_read(const unsigned char *buf,size_t len,ZipEnt *e,size_t *outlen){
    size_t p=e->lho;
    if(p+30>len||rd32(buf+p)!=0x04034b50) return NULL;
    int nlen=rd16(buf+p+26),xlen=rd16(buf+p+28);
    size_t data=p+30+nlen+xlen;
    if(data+e->csize>len) return NULL;
    if(e->method==0){
        unsigned char *out=xmalloc(e->usize?e->usize:1);
        memcpy(out,buf+data,e->usize); *outlen=e->usize; return out;
    } else if(e->method==8){
        return inflate_raw(buf+data,e->csize,outlen);
    }
    return NULL;
}

/* ---- NPY ---- */
typedef struct { char kind; int itemsize; int ndim; long shape[8]; unsigned char *data; size_t nbytes; } Npy;

static int parse_npy(unsigned char *buf,size_t len,Npy *a){
    if(len<10||buf[0]!=0x93||memcmp(buf+1,"NUMPY",5)) return -1;
    int major=buf[6];
    size_t hlen, hstart;
    if(major==1){ hlen=rd16(buf+8); hstart=10; }
    else { hlen=rd32(buf+8); hstart=12; }
    if(hstart+hlen>len) return -1;
    char *hdr=xmalloc(hlen+1); memcpy(hdr,buf+hstart,hlen); hdr[hlen]=0;
    /* descr */
    char *d=strstr(hdr,"'descr'"); if(!d)d=strstr(hdr,"\"descr\"");
    if(!d){free(hdr);return -1;}
    char *q1=strchr(d+7,'\''); if(!q1)q1=strchr(d+7,'"');
    char qc=q1[0]; char *q2=strchr(q1+1,qc);
    char descr[32]; size_t dl=q2-q1-1; if(dl>31)dl=31; memcpy(descr,q1+1,dl);descr[dl]=0;
    /* kind/itemsize */
    a->kind=0;a->itemsize=0;
    char kc=0; int isz=0;
    /* descr like '<f8','<i4','|u1','<U5' */
    for(char *c=descr;*c;c++){
        if(*c=='f'||*c=='i'||*c=='u'||*c=='U'||*c=='b'){kc=*c;c++;isz=atoi(c);break;}
    }
    if(kc=='f'&&isz==8){a->kind='f';a->itemsize=8;}
    else if(kc=='i'&&(isz==4||isz==8)){a->kind='i';a->itemsize=isz;}
    else if(kc=='u'&&(isz==4||isz==8)){a->kind='u';a->itemsize=isz;}
    else if(kc=='U'){a->kind='U';a->itemsize=4*isz;}
    else if(kc=='b'&&isz==1){a->kind='b';a->itemsize=1;}
    else {free(hdr);return -1;}
    /* fortran_order */
    a->ndim=0;
    char *sh=strstr(hdr,"'shape'"); if(!sh)sh=strstr(hdr,"\"shape\"");
    if(sh){
        char *lp=strchr(sh,'('); char *rp=strchr(lp,')');
        char tmp[256]; size_t tl=rp-lp-1; if(tl>255)tl=255; memcpy(tmp,lp+1,tl);tmp[tl]=0;
        char *tok=strtok(tmp,","); int k=0;
        while(tok&&k<8){ while(*tok==' ')tok++; if(*tok) a->shape[k++]=atol(tok); tok=strtok(NULL,","); }
        a->ndim=k;
    }
    free(hdr);
    size_t n=1; for(int i=0;i<a->ndim;i++) n*=a->shape[i];
    a->nbytes=n*a->itemsize;
    a->data=buf+hstart+hlen;
    if(a->data+a->nbytes>buf+len) return -1;
    return 0;
}

/* ---- NPZ container ---- */
typedef struct { unsigned char *file; size_t filelen; ZipEnt *ents; int nent; } Npz;

static int npz_open(Npz *z,const char *path){
    z->file=read_file(path,&z->filelen);
    if(!z->file) return -1;
    if(zip_entries(z->file,z->filelen,&z->ents,&z->nent)) return -1;
    return 0;
}
static void npz_close(Npz *z){ free(z->file); free(z->ents); }

static int npz_get(Npz *z,const char *name,Npy *a,unsigned char **owned){
    char target[280]; snprintf(target,sizeof target,"%s.npy",name);
    for(int e=0;e<z->nent;e++){
        if(strcmp(z->ents[e].name,target)==0){
            size_t ol; unsigned char *buf=zip_read(z->file,z->filelen,&z->ents[e],&ol);
            if(!buf) return -1;
            if(parse_npy(buf,ol,a)){free(buf);return -1;}
            *owned=buf; return 0;
        }
    }
    return -1;
}

/* convert integer array to idx_t */
static idx_t *to_idx(Npy *a,long *count){
    long n=1; for(int i=0;i<a->ndim;i++)n*=a->shape[i];
    *count=n;
    idx_t *out=xmalloc(n*sizeof(idx_t));
    if(a->kind=='i'&&a->itemsize==4){int32_t*s=(int32_t*)a->data;for(long i=0;i<n;i++)out[i]=s[i];}
    else if(a->kind=='i'&&a->itemsize==8){int64_t*s=(int64_t*)a->data;for(long i=0;i<n;i++)out[i]=(idx_t)s[i];}
    else if(a->kind=='u'&&a->itemsize==4){uint32_t*s=(uint32_t*)a->data;for(long i=0;i<n;i++)out[i]=(idx_t)s[i];}
    else if(a->kind=='u'&&a->itemsize==8){uint64_t*s=(uint64_t*)a->data;for(long i=0;i<n;i++)out[i]=(idx_t)s[i];}
    else {free(out);return NULL;}
    return out;
}
static double *to_f64(Npy *a,long *count){
    long n=1; for(int i=0;i<a->ndim;i++)n*=a->shape[i];
    *count=n;
    if(a->kind!='f'||a->itemsize!=8) return NULL;
    double *out=xmalloc(n*sizeof(double)); memcpy(out,a->data,n*sizeof(double)); return out;
}

/* ---- JSON minimal parser ---- */
typedef struct JV JV;
struct JV { int type; /*0 obj,1 arr,2 str,3 num,4 bool,5 null*/ char *s; double num; JV *child,*next; char *key; };
static void jskip(const char **p){ while(**p==' '||**p=='\t'||**p=='\n'||**p=='\r')(*p)++; }
static JV *jparse(const char **p);
static char *jstring(const char **p){
    (*p)++; const char *s=*p; size_t cap=32,len=0; char *out=xmalloc(cap);
    while(**p&&**p!='"'){
        char c=**p;
        if(c=='\\'){ (*p)++; char e=**p;
            if(e=='n')c='\n';else if(e=='t')c='\t';else if(e=='r')c='\r';else if(e=='b')c='\b';else if(e=='f')c='\f';
            else if(e=='u'){ c='?'; *p+=4; }
            else c=e;
        }
        if(len+1>=cap){cap*=2;out=xrealloc(out,cap);}
        out[len++]=c; (*p)++;
    }
    (*p)++; out[len]=0; (void)s; return out;
}
static JV *jparse(const char **p){
    jskip(p);
    JV *v=xcalloc(1,sizeof(JV));
    char c=**p;
    if(c=='{'){
        v->type=0;(*p)++;jskip(p);
        JV **tail=&v->child;
        while(**p&&**p!='}'){
            jskip(p); char *key=jstring(p); jskip(p); (*p)++; /*colon*/
            JV *ch=jparse(p); ch->key=key; *tail=ch; tail=&ch->next;
            jskip(p); if(**p==',')(*p)++;
        }
        if(**p=='}')(*p)++;
    } else if(c=='['){
        v->type=1;(*p)++;jskip(p);
        JV **tail=&v->child;
        while(**p&&**p!=']'){
            JV *ch=jparse(p); *tail=ch; tail=&ch->next;
            jskip(p); if(**p==',')(*p)++;
        }
        if(**p==']')(*p)++;
    } else if(c=='"'){ v->type=2; v->s=jstring(p); }
    else if(c=='t'){v->type=4;v->num=1;*p+=4;}
    else if(c=='f'){v->type=4;v->num=0;*p+=5;}
    else if(c=='n'){v->type=5;*p+=4;}
    else { v->type=3; v->num=strtod(*p,(char**)p); }
    return v;
}
static JV *jget(JV *o,const char *key){ for(JV*c=o->child;c;c=c->next) if(c->key&&strcmp(c->key,key)==0)return c; return NULL; }
static void jfree(JV *v){ if(!v)return; jfree(v->child); jfree(v->next); free(v->s); free(v->key); free(v); }

/* ---- Problem ---- */
static void csc_from_raw(idx_t nrow,idx_t ncol,long nnz,double *data,idx_t *ri,idx_t *cp,CSC *M){
    M->nrow=nrow;M->ncol=ncol;
    M->p=xmalloc((ncol+1)*sizeof(idx_t));
    idx_t total=cp[ncol];
    idx_t *tmpi=xmalloc((total?total:1)*sizeof(idx_t)); double *tmpx=xmalloc((total?total:1)*sizeof(double));
    idx_t w=0;
    for(idx_t c=0;c<ncol;c++){
        idx_t st=cp[c],en=cp[c+1];
        for(idx_t a=st+1;a<en;a++){
            idx_t rr=ri[a]; double vv=data[a]; idx_t bb=a-1;
            while(bb>=st&&ri[bb]>rr){ri[bb+1]=ri[bb];data[bb+1]=data[bb];bb--;}
            ri[bb+1]=rr;data[bb+1]=vv;
        }
        M->p[c]=w;
        for(idx_t a=st;a<en;){
            idx_t rr=ri[a]; double vv=data[a]; a++;
            while(a<en&&ri[a]==rr){vv+=data[a];a++;}
            if(vv!=0.0){ tmpi[w]=rr;tmpx[w]=vv;w++; }
        }
    }
    M->p[ncol]=w;
    M->i=tmpi;M->x=tmpx;
}

int load_problem(const char *path,Problem *P){
    memset(P,0,sizeof(*P));
    Npz z;
    if(npz_open(&z,path)) return -1;
    Npy a; unsigned char *own;
    long nq,nb;
    if(npz_get(&z,"q",&a,&own))return -1; P->q=to_f64(&a,&nq); free(own);
    if(npz_get(&z,"b",&a,&own))return -1; P->b=to_f64(&a,&nb); free(own);
    if(!P->q||!P->b)return -1;
    P->n=(idx_t)nq; P->m=(idx_t)nb;
    /* matrices */
    for(int pass=0;pass<2;pass++){
        const char *pre=pass?"A":"P";
        char nm[64];
        Npy ad,ai,ap,ash; unsigned char *o1,*o2,*o3,*o4;
        snprintf(nm,64,"%s_data",pre); if(npz_get(&z,nm,&ad,&o1))return -1;
        snprintf(nm,64,"%s_indices",pre); if(npz_get(&z,nm,&ai,&o2))return -1;
        snprintf(nm,64,"%s_indptr",pre); if(npz_get(&z,nm,&ap,&o3))return -1;
        snprintf(nm,64,"%s_shape",pre); if(npz_get(&z,nm,&ash,&o4))return -1;
        long nnz,nri,ncp;
        double *data=to_f64(&ad,&nnz);
        idx_t *ri=to_idx(&ai,&nri);
        idx_t *cp=to_idx(&ap,&ncp);
        long sh[2]; {
            if(ash.kind=='i'&&ash.itemsize==8){int64_t*s=(int64_t*)ash.data;sh[0]=s[0];sh[1]=s[1];}
            else {int32_t*s=(int32_t*)ash.data;sh[0]=s[0];sh[1]=s[1];}
        }
        CSC M; csc_from_raw((idx_t)sh[0],(idx_t)sh[1],nnz,data,ri,cp,&M);
        if(pass==0)P->P=M; else P->A=M;
        free(data);free(ri);free(cp);free(o1);free(o2);free(o3);free(o4);
    }
    /* metadata */
    if(npz_get(&z,"metadata_json",&a,&own))return -1;
    /* unicode UCS4 -> ascii */
    long nch=a.nbytes/4;
    char *js=xmalloc(nch+1);
    uint32_t *u=(uint32_t*)a.data;
    for(long i=0;i<nch;i++) js[i]=(char)(u[i]<128?u[i]:'?');
    js[nch]=0; free(own);
    const char *p=js; JV *root=jparse(&p);
    JV *off=jget(root,"objective_offset"); P->offset=off?off->num:0.0;
    JV *cones=jget(root,"cones");
    int nbl=0; for(JV*c=cones->child;c;c=c->next)nbl++;
    P->nblk=nbl; P->blk=xcalloc(nbl,sizeof(Block));
    int bi=0; idx_t pos=0;
    for(JV*c=cones->child;c;c=c->next,bi++){
        JV *t=jget(c,"type"),*d=jget(c,"dim");
        Block *B=&P->blk[bi];
        B->dim=(idx_t)d->num; B->start=pos;
        if(strcmp(t->s,"zero")==0){B->type=0;B->width=B->dim;}
        else if(strcmp(t->s,"nonnegative")==0){B->type=1;B->width=B->dim;}
        else if(strcmp(t->s,"soc")==0){B->type=2;B->width=B->dim;}
        else if(strcmp(t->s,"exponential")==0){B->type=3;B->width=3;}
        else {B->type=4;B->width=B->dim*(B->dim+1)/2;}
        pos+=B->width;
    }
    jfree(root); free(js);
    npz_close(&z);
    return 0;
}

/* ---- NPY/ZIP writer ---- */
static void put32(unsigned char *p,uint32_t v){p[0]=v&255;p[1]=(v>>8)&255;p[2]=(v>>16)&255;p[3]=(v>>24)&255;}
static void put16(unsigned char *p,uint16_t v){p[0]=v&255;p[1]=(v>>8)&255;}

typedef struct { char name[64]; unsigned char *data; size_t len; } OutEnt;

static unsigned char *make_npy_f64(const double *v,size_t n,size_t *len){
    char hdr[128];
    int hl=snprintf(hdr,128,"{'descr': '<f8', 'fortran_order': False, 'shape': (%zu,), }",n);
    while((10+hl+1)%64) hdr[hl++]=' ';
    hdr[hl]='\n'; hl++;
    size_t total=10+hl+n*8;
    unsigned char *b=xmalloc(total);
    b[0]=0x93;memcpy(b+1,"NUMPY",5);b[6]=1;b[7]=0;put16(b+8,(uint16_t)hl);
    memcpy(b+10,hdr,hl);
    memcpy(b+10+hl,v,n*8);
    *len=total;return b;
}
static unsigned char *make_npy_u(const char *s,size_t *len){
    size_t n=strlen(s);
    char hdr[128];
    int hl=snprintf(hdr,128,"{'descr': '<U%zu', 'fortran_order': False, 'shape': (), }",n);
    while((10+hl+1)%64) hdr[hl++]=' ';
    hdr[hl]='\n'; hl++;
    size_t total=10+hl+n*4;
    unsigned char *b=xmalloc(total);
    b[0]=0x93;memcpy(b+1,"NUMPY",5);b[6]=1;b[7]=0;put16(b+8,(uint16_t)hl);
    memcpy(b+10,hdr,hl);
    uint32_t *u=(uint32_t*)(b+10+hl);
    for(size_t i=0;i<n;i++)u[i]=(unsigned char)s[i];
    *len=total;return b;
}

static uint32_t crc32_tab[256]; static int crc_init=0;
static uint32_t crc32(const unsigned char *d,size_t n){
    if(!crc_init){for(uint32_t i=0;i<256;i++){uint32_t c=i;for(int k=0;k<8;k++)c=c&1?0xedb88320^(c>>1):c>>1;crc32_tab[i]=c;}crc_init=1;}
    uint32_t c=0xffffffff;
    for(size_t i=0;i<n;i++)c=crc32_tab[(c^d[i])&255]^(c>>8);
    return c^0xffffffff;
}

int write_npz(const char *path,OutEnt *ents,int nent){
    FILE *f=fopen(path,"wb"); if(!f)return -1;
    size_t *lho=xmalloc(nent*sizeof(size_t));
    uint32_t *crcs=xmalloc(nent*sizeof(uint32_t));
    size_t off=0;
    for(int e=0;e<nent;e++){
        lho[e]=off;
        unsigned char lh[30]; memset(lh,0,30);
        put32(lh,0x04034b50);put16(lh+4,20);put16(lh+8,0);
        crcs[e]=crc32(ents[e].data,ents[e].len);
        put32(lh+14,crcs[e]);put32(lh+18,(uint32_t)ents[e].len);put32(lh+22,(uint32_t)ents[e].len);
        put16(lh+26,(uint16_t)strlen(ents[e].name));
        fwrite(lh,1,30,f); fwrite(ents[e].name,1,strlen(ents[e].name),f);
        fwrite(ents[e].data,1,ents[e].len,f);
        off+=30+strlen(ents[e].name)+ents[e].len;
    }
    size_t cd=off;
    for(int e=0;e<nent;e++){
        unsigned char ch[46]; memset(ch,0,46);
        put32(ch,0x02014b50);put16(ch+4,20);put16(ch+6,20);put16(ch+10,0);
        put32(ch+16,crcs[e]);put32(ch+20,(uint32_t)ents[e].len);put32(ch+24,(uint32_t)ents[e].len);
        put16(ch+28,(uint16_t)strlen(ents[e].name));
        put32(ch+42,(uint32_t)lho[e]);
        fwrite(ch,1,46,f);fwrite(ents[e].name,1,strlen(ents[e].name),f);
        off+=46+strlen(ents[e].name);
    }
    unsigned char eocd[22]; memset(eocd,0,22);
    put32(eocd,0x06054b50);put16(eocd+8,(uint16_t)nent);put16(eocd+10,(uint16_t)nent);
    put32(eocd+12,(uint32_t)(off-cd));put32(eocd+16,(uint32_t)cd);
    fwrite(eocd,1,22,f);
    fclose(f);free(lho);free(crcs);
    return 0;
}

int write_result(const char *path,const char *status,const double *x,const double *s,const double *z,idx_t n,idx_t m){
    OutEnt ents[4]; int ne=0;
    char meta[256];
    snprintf(meta,256,"{\"schema_version\": 1, \"status\": \"%s\"}",status);
    size_t ml; ents[ne].data=make_npy_u(meta,&ml); ents[ne].len=ml; strcpy(ents[ne].name,"metadata_json.npy"); ne++;
    size_t l;
    if(strcmp(status,"solved")==0){
        ents[ne].data=make_npy_f64(x,n,&l);ents[ne].len=l;strcpy(ents[ne].name,"x.npy");ne++;
        ents[ne].data=make_npy_f64(s,m,&l);ents[ne].len=l;strcpy(ents[ne].name,"s.npy");ne++;
        ents[ne].data=make_npy_f64(z,m,&l);ents[ne].len=l;strcpy(ents[ne].name,"z.npy");ne++;
    } else if(strcmp(status,"primal_infeasible")==0){
        ents[ne].data=make_npy_f64(z,m,&l);ents[ne].len=l;strcpy(ents[ne].name,"z.npy");ne++;
    } else if(strcmp(status,"dual_infeasible")==0){
        ents[ne].data=make_npy_f64(x,n,&l);ents[ne].len=l;strcpy(ents[ne].name,"x.npy");ne++;
        ents[ne].data=make_npy_f64(s,m,&l);ents[ne].len=l;strcpy(ents[ne].name,"s.npy");ne++;
    }
    int r=write_npz(path,ents,ne);
    for(int e=0;e<ne;e++)free(ents[e].data);
    return r;
}
