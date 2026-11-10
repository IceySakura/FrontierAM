#!/usr/bin/env python3
"""Cartesian harmonic expansions, using z degree zero or one."""
import sys

p = int(sys.argv[1]) if len(sys.argv) > 1 else 16
path = sys.argv[2] if len(sys.argv) > 2 else "harmonic.hpp"
idx = [(i,n-i-k,k) for n in range(p+1) for k in range(min(n,1)+1)
       for i in range(n-k+1)]
lookup = {a:i for i,a in enumerate(idx)}
degree = lambda a: sum(a)
source = [a for a in idx if degree(a) < p and degree(a) != 1]
local = idx[1:]
lines = []
emit = lines.append

def reduced(a, name):
    if a[2] < 2:
        return f"{name}[{lookup[a]}]"
    assert a[2] == 2
    return f"(-{name}[{lookup[(a[0]+2,a[1],0)]}]-{name}[{lookup[(a[0],a[1]+2,0)]}])"

def monomial_ref(a):
    return f"v[{lookup[a]}]"

emit("namespace H {")
emit(f"constexpr int ORDER={p},NC={len(idx)},NM={p*p};")
emit(f"constexpr int GX={lookup[(1,0,0)]},GY={lookup[(0,1,0)]},GZ={lookup[(0,0,1)]};")
emit("using V = double __attribute__((vector_size(32)));")
emit("static inline void monomials(double x,double y,double z,double *v) {")
emit("v[0]=1;")
for a in idx[1:]:
    i,j,k=a
    n=degree(a)
    terms=[]
    if i:
        terms.append(f"x*{monomial_ref((i-1,j,k))}")
    if j:
        terms.append(f"y*{monomial_ref((i,j-1,k))}")
    if k:
        terms.append(f"z*{monomial_ref((i,j,0))}")
    else:
        if i>=2:
            terms.append(f"-z*{monomial_ref((i-2,j,1))}")
        if j>=2:
            terms.append(f"-z*{monomial_ref((i,j-2,1))}")
    emit(f"v[{lookup[a]}]=("+"+".join(terms).replace("+-","-")+f")/{n}.0;")
emit("}")

emit("static __attribute__((noinline)) void derivatives4(V x,V y,V z,V *d) {")
emit("const V qi=1.0/(x*x+y*y+z*z); d[0]=_mm256_sqrt_pd(qi);")
xyz=("x","y","z")
for a in idx[1:]:
    axis=next(k for k in range(3) if a[k])
    terms=[]
    for k in range(3):
        if a[k]:
            b=list(a); b[k]-=1
            coef=2*a[k]-(1 if k==axis else 0)
            terms.append(f"{coef}.0*{xyz[k]}*d[{lookup[tuple(b)]}]")
        coef=(a[k]-1)**2 if k==axis else a[k]*(a[k]-1)
        if coef and a[k]>=2:
            b=list(a); b[k]-=2
            terms.append(f"{coef}.0*d[{lookup[tuple(b)]}]")
    emit(f"d[{lookup[a]}]=-qi*("+"+".join(terms)+");")
emit("}")

emit("static __attribute__((noinline)) void derivatives(double x,double y,double z,double *d) {")
emit("const double qi=1.0/(x*x+y*y+z*z); d[0]=std::sqrt(qi);")
for a in idx[1:]:
    axis=next(k for k in range(3) if a[k])
    terms=[]
    for k in range(3):
        if a[k]:
            b=list(a); b[k]-=1
            coef=2*a[k]-(1 if k==axis else 0)
            terms.append(f"{coef}.0*{xyz[k]}*d[{lookup[tuple(b)]}]")
        coef=(a[k]-1)**2 if k==axis else a[k]*(a[k]-1)
        if coef and a[k]>=2:
            b=list(a); b[k]-=2
            terms.append(f"{coef}.0*d[{lookup[tuple(b)]}]")
    emit(f"d[{lookup[a]}]=-qi*("+"+".join(terms)+");")
emit("}")

extra=[(i,n-2-i,2) for n in range(2,p+1) for i in range(n-1)]
extended=dict(lookup)
extended.update({a:len(idx)+i for i,a in enumerate(extra)})
emit("static __attribute__((noinline)) void m2l_mutual(const double *__restrict ma,const double *__restrict mb,double *__restrict la,double *__restrict lb,double x,double y,double z) {")
emit(f"double d[{len(idx)+len(extra)+4}]; derivatives(x,y,z,d);")
for a in extra:
    emit(f"d[{extended[a]}]=-d[{lookup[(a[0]+2,a[1],0)]}]-d[{lookup[(a[0],a[1]+2,0)]}];")
for bn in range(p):
    if bn==1:
        continue
    for bz in range(min(bn,1)+1):
        bs=lookup[(0,bn-bz,bz)]
        emit(f"for(int bx=0;bx<={bn-bz};bx++) {{")
        emit(f"const double am=ma[{bs}+bx],bm=mb[{bs}+bx];")
        emit("const V a4={am,am,am,am},b4={bm,bm,bm,bm};")
        emit("const __m128d a2=_mm256_castpd256_pd128(a4),b2=_mm256_castpd256_pd128(b4);")
        for an in range(1,p-bn+1):
            for az in range(min(an,1)+1):
                ls=lookup[(0,an-az,az)]
                ds=extended[(0,an+bn-az-bz,az+bz)]
                length=an-az+1
                pos=0
                negative=(an+bn)%2
                while pos+4<=length:
                    emit("{")
                    emit(f"V v=_mm256_loadu_pd(d+{ds+pos}+bx);")
                    emit(f"_mm256_storeu_pd(la+{ls+pos},_mm256_fmadd_pd(b4,v,_mm256_loadu_pd(la+{ls+pos})));")
                    op="_mm256_fnmadd_pd" if negative else "_mm256_fmadd_pd"
                    emit(f"_mm256_storeu_pd(lb+{ls+pos},{op}(a4,v,_mm256_loadu_pd(lb+{ls+pos})));")
                    emit("}")
                    pos+=4
                if pos+2<=length:
                    emit("{")
                    emit(f"__m128d v=_mm_loadu_pd(d+{ds+pos}+bx);")
                    emit(f"_mm_storeu_pd(la+{ls+pos},_mm_fmadd_pd(b2,v,_mm_loadu_pd(la+{ls+pos})));")
                    op="_mm_fnmadd_pd" if negative else "_mm_fmadd_pd"
                    emit(f"_mm_storeu_pd(lb+{ls+pos},{op}(a2,v,_mm_loadu_pd(lb+{ls+pos})));")
                    emit("}")
                    pos+=2
                if pos<length:
                    emit(f"la[{ls+pos}]+=bm*d[{ds+pos}+bx];")
                    sign="-" if negative else "+"
                    emit(f"lb[{ls+pos}]{sign}=am*d[{ds+pos}+bx];")
        emit("}")
emit("}")

emit("static __attribute__((noinline)) void m2l4(const double *m0,const double *m1,const double *m2,const double *m3,V *l,V x,V y,V z) {")
emit("V d[NC],m[NM]; derivatives4(x,y,z,d);")
emit("int k=0; for(;k+4<=NM;k+=4) {")
emit("V a=_mm256_loadu_pd(m0+k),b=_mm256_loadu_pd(m1+k),c=_mm256_loadu_pd(m2+k),e=_mm256_loadu_pd(m3+k);")
emit("V u=_mm256_unpacklo_pd(a,b),v=_mm256_unpackhi_pd(a,b),w=_mm256_unpacklo_pd(c,e),t=_mm256_unpackhi_pd(c,e);")
emit("m[k]=_mm256_permute2f128_pd(u,w,0x20); m[k+1]=_mm256_permute2f128_pd(v,t,0x20);")
emit("m[k+2]=_mm256_permute2f128_pd(u,w,0x31); m[k+3]=_mm256_permute2f128_pd(v,t,0x31);")
emit("} for(;k<NM;k++) m[k]=V{m0[k],m1[k],m2[k],m3[k]};")
for a in local:
    terms=[]
    for b in source:
        if degree(a)+degree(b)<=p:
            c=tuple(x+y for x,y in zip(a,b))
            terms.append(f"m[{lookup[b]}]*{reduced(c,'d')}")
    emit(f"l[{lookup[a]}]+="+"+".join(terms)+";")
emit("}")

emit("static __attribute__((noinline)) void m2p4(const double *m,V *l,V x,V y,V z,V mass,V &ax,V &ay,V &az) {")
emit("V d[NC]; derivatives4(x,y,z,d);")
for axis,name in enumerate(("ax","ay","az")):
    terms=[]
    for b in source:
        a=list(b); a[axis]+=1
        terms.append(f"m[{lookup[b]}]*{reduced(tuple(a),'d')}")
    emit(name+"+="+"+".join(terms)+";")
for a in local:
    sign="-" if degree(a)%2 else "+"
    emit(f"l[{lookup[a]}]{sign}=mass*d[{lookup[a]}];")
emit("}")

emit("static __attribute__((noinline)) void shift_m(const double *m,double *q,double x,double y,double z) {")
emit("double v[NC]; monomials(-x,-y,-z,v);")
for a in source:
    terms=[]
    for b in source:
        c=tuple(x-y for x,y in zip(a,b))
        if c in lookup:
            terms.append(f"m[{lookup[b]}]*v[{lookup[c]}]")
        if a[2]==0 and b[2]==1:
            for cx,cy in ((a[0]-b[0]-2,a[1]-b[1]),(a[0]-b[0],a[1]-b[1]-2)):
                c=(cx,cy,1)
                if c in lookup:
                    terms.append(f"-m[{lookup[b]}]*v[{lookup[c]}]")
    emit(f"q[{lookup[a]}]+="+"+".join(terms).replace("+-","-")+";")
emit("}")

emit("static __attribute__((noinline)) void shift_l(const double *l,double *q,double x,double y,double z) {")
emit("double v[NC]; monomials(x,y,z,v);")
for a in local:
    terms=[]
    for c in idx:
        if degree(a)+degree(c)<=p:
            b=tuple(x+y for x,y in zip(a,c))
            terms.append(f"v[{lookup[c]}]*{reduced(b,'l')}")
    emit(f"q[{lookup[a]}]+="+"+".join(terms)+";")
emit("}")

emit("static __attribute__((noinline)) void evaluate_l(const double *l,double x,double y,double z,double &ax,double &ay,double &az) {")
emit("double v[NC]; monomials(x,y,z,v);")
for axis,name in enumerate(("ax","ay","az")):
    terms=[]
    for a in idx:
        if degree(a)>=p:
            continue
        b=list(a); b[axis]+=1
        terms.append(f"v[{lookup[a]}]*{reduced(tuple(b),'l')}")
    emit(name+"+="+"+".join(terms)+";")
emit("}")
low=min(6,p)
low_idx=[a for a in idx if degree(a)<=low]
low_src=[a for a in low_idx if degree(a)<low and degree(a)!=1]
emit("static __attribute__((noinline)) void m2l_coarse(const double *ma,const double *mb,double *la,double *lb,double x,double y,double z) {")
emit(f"double d[{len(low_idx)}]; const double qi=1.0/(x*x+y*y+z*z); d[0]=std::sqrt(qi);")
for a in low_idx[1:]:
    axis=next(k for k in range(3) if a[k])
    terms=[]
    for k in range(3):
        if a[k]:
            b=list(a); b[k]-=1
            coef=2*a[k]-(1 if k==axis else 0)
            terms.append(f"{coef}.0*{xyz[k]}*d[{lookup[tuple(b)]}]")
        coef=(a[k]-1)**2 if k==axis else a[k]*(a[k]-1)
        if coef and a[k]>=2:
            b=list(a); b[k]-=2
            terms.append(f"{coef}.0*d[{lookup[tuple(b)]}]")
    emit(f"d[{lookup[a]}]=-qi*("+"+".join(terms)+");")
for a in low_idx[1:]:
    aa,bb=[],[]
    for b in low_src:
        if degree(a)+degree(b)<=low:
            c=tuple(x+y for x,y in zip(a,b))
            dr=reduced(c,"d")
            aa.append(f"mb[{lookup[b]}]*{dr}")
            sign="-" if (degree(a)+degree(b))%2 else ""
            bb.append(f"{sign}ma[{lookup[b]}]*{dr}")
    emit(f"la[{lookup[a]}]+="+"+".join(aa)+";")
    emit(f"lb[{lookup[a]}]+="+"+".join(bb).replace("+-","-")+";")
emit("}")
emit("}")
with open(path,"w") as f:
    f.write("\n".join(lines)+"\n")
