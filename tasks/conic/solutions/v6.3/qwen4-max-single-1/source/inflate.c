#include "common.h"

/* Minimal raw-DEFLATE decoder (RFC1951). Returns malloc'd buffer, sets *outlen. */
typedef struct { const unsigned char *in; size_t inlen,pos; unsigned bitbuf; int bitcnt; } BitR;
static int bits(BitR *b,int need){
    while(b->bitcnt<need){
        if(b->pos>=b->inlen) return -1;
        b->bitbuf |= (unsigned)b->in[b->pos++] << b->bitcnt;
        b->bitcnt += 8;
    }
    int v = b->bitbuf & ((1u<<need)-1);
    b->bitbuf >>= need; b->bitcnt -= need;
    return v;
}
typedef struct { short count[16]; short symbol[288+32]; } Huff;
static int build_huff(Huff *h,const short *length,int n){
    int i,offs[16];
    for(i=0;i<16;i++) h->count[i]=0;
    for(i=0;i<n;i++) h->count[length[i]]++;
    if(h->count[0]==n) return 0; /* no codes */
    int left=1;
    for(i=1;i<16;i++){ left<<=1; left-=h->count[i]; if(left<0) return -1; }
    offs[1]=0;
    for(i=1;i<15;i++) offs[i+1]=offs[i]+h->count[i];
    for(i=0;i<n;i++) if(length[i]) h->symbol[offs[length[i]]++]=(short)i;
    return left;
}
static int decode(BitR *b,Huff *h){
    int code=0,first=0,index=0,len;
    for(len=1;len<16;len++){
        int bit=bits(b,1); if(bit<0) return -1;
        code|=bit;
        int count=h->count[len];
        if(code-first<count) return h->symbol[index+(code-first)];
        index+=count; first=(first+count)<<1; code<<=1;
    }
    return -1;
}
static const short LBASE[29]={3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258};
static const short LEXT[29]={0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
static const short DBASE[30]={1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
static const short DEXT[30]={0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};

unsigned char *inflate_raw(const unsigned char *in,size_t inlen,size_t *outlen){
    BitR b={in,inlen,0,0,0};
    size_t cap=1<<16,len=0; unsigned char *out=xmalloc(cap);
    int last=0;
    do{
        last=bits(&b,1); int type=bits(&b,2);
        if(last<0||type<0){free(out);return NULL;}
        if(type==0){
            b.bitbuf=0;b.bitcnt=0;
            if(b.pos+4>b.inlen){free(out);return NULL;}
            unsigned L=b.in[b.pos]|(b.in[b.pos+1]<<8); b.pos+=4;
            if(b.pos+L>b.inlen){free(out);return NULL;}
            if(len+L>cap){while(len+L>cap)cap*=2;out=xrealloc(out,cap);}
            memcpy(out+len,b.in+b.pos,L); len+=L; b.pos+=L;
        } else if(type==1||type==2){
            Huff lh,dh;
            if(type==1){
                short lengths[288+30]; int i;
                for(i=0;i<144;i++)lengths[i]=8;
                for(;i<256;i++)lengths[i]=9;
                for(;i<280;i++)lengths[i]=7;
                for(;i<288;i++)lengths[i]=8;
                build_huff(&lh,lengths,288);
                for(i=0;i<30;i++)lengths[i]=5;
                build_huff(&dh,lengths,30);
            } else {
                int nlit=bits(&b,5)+257,ndist=bits(&b,5)+1,ncode=bits(&b,4)+4;
                static const short order[19]={16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};
                short cl[19]; int i;
                for(i=0;i<19;i++)cl[i]=0;
                for(i=0;i<ncode;i++){int v=bits(&b,3);if(v<0){free(out);return NULL;}cl[order[i]]=(short)v;}
                Huff ch; build_huff(&ch,cl,19);
                short lengths[288+32]; i=0;
                while(i<nlit+ndist){
                    int sym=decode(&b,&ch);
                    if(sym<0){free(out);return NULL;}
                    if(sym<16) lengths[i++]=(short)sym;
                    else if(sym==16){ if(i==0){free(out);return NULL;} int prev=lengths[i-1];int r=3+bits(&b,2);while(r--&&i<nlit+ndist)lengths[i++]=prev; }
                    else if(sym==17){int r=3+bits(&b,3);while(r--&&i<nlit+ndist)lengths[i++]=0;}
                    else {int r=11+bits(&b,7);while(r--&&i<nlit+ndist)lengths[i++]=0;}
                }
                if(build_huff(&lh,lengths,nlit)<0||build_huff(&dh,lengths+nlit,ndist)<0){free(out);return NULL;}
            }
            for(;;){
                int sym=decode(&b,&lh);
                if(sym<0){free(out);return NULL;}
                if(sym<256){ if(len+1>cap){cap*=2;out=xrealloc(out,cap);} out[len++]=(unsigned char)sym; }
                else if(sym==256) break;
                else {
                    sym-=257;
                    if(sym>=29){free(out);return NULL;}
                    int L=LBASE[sym]+bits(&b,LEXT[sym]);
                    int ds=decode(&b,&dh);
                    if(ds<0||ds>=30){free(out);return NULL;}
                    int D=DBASE[ds]+bits(&b,DEXT[ds]);
                    if((size_t)D>len){free(out);return NULL;}
                    if(len+L>cap){while(len+L>cap)cap*=2;out=xrealloc(out,cap);}
                    size_t from=len-D;
                    for(int k=0;k<L;k++) out[len+k]=out[from+k];
                    len+=L;
                }
            }
        } else {free(out);return NULL;}
    }while(!last);
    *outlen=len;
    return out;
}
