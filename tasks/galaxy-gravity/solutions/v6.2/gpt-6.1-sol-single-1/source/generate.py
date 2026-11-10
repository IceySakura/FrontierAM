#!/usr/bin/env python3
"""Generate fixed-order Cartesian expansion kernels."""
import sys

p = int(sys.argv[1]) if len(sys.argv) > 1 else 8
out = sys.argv[2] if len(sys.argv) > 2 else "kernels.hpp"
indices = [(i, j, n-i-j) for n in range(p+1)
           for i in range(n+1) for j in range(n-i+1)]
lookup = {a: k for k, a in enumerate(indices)}
degree = lambda a: sum(a)
source = [a for a in indices if degree(a) < p and degree(a) != 1]
local = indices[1:]
lines = []
emit = lines.append
emit(f"constexpr int ORDER = {p};")
emit(f"constexpr int NC = {len(indices)};")
emit(f"constexpr int NM = {sum(degree(a) < p for a in indices)};")
emit("static const int powers[NC][3] = {")
for a in indices:
    emit("{" + ",".join(map(str, a)) + "},")
emit("};")

emit("static inline void derivatives(double x, double y, double z, double *d) {")
emit("const double qi=1.0/(x*x+y*y+z*z+0.0001);")
emit("d[0]=std::sqrt(qi);")
xyz = ["x", "y", "z"]
for a in indices[1:]:
    k = lookup[a]
    axis = next(i for i in range(3) if a[i])
    terms = []
    for j in range(3):
        if a[j]:
            b = list(a)
            b[j] -= 1
            c = 2*a[j] - (1 if j == axis else 0)
            terms.append(f"{c}.0*{xyz[j]}*d[{lookup[tuple(b)]}]")
        c = (a[j]-1)**2 if j == axis else a[j]*(a[j]-1)
        if c and a[j] >= 2:
            b = list(a)
            b[j] -= 2
            terms.append(f"{c}.0*d[{lookup[tuple(b)]}]")
    emit(f"d[{k}]=-qi*(" + "+".join(terms) + ");")
emit("}")

emit("using V = double __attribute__((vector_size(32)));")
emit("static inline void derivatives4(V x, V y, V z, V *d) {")
emit("const V qi=1.0/(x*x+y*y+z*z+0.0001);")
emit("d[0]=_mm256_sqrt_pd(qi);")
for a in indices[1:]:
    k = lookup[a]
    axis = next(i for i in range(3) if a[i])
    terms = []
    for j in range(3):
        if a[j]:
            b = list(a)
            b[j] -= 1
            c = 2*a[j] - (1 if j == axis else 0)
            terms.append(f"{c}.0*{xyz[j]}*d[{lookup[tuple(b)]}]")
        c = (a[j]-1)**2 if j == axis else a[j]*(a[j]-1)
        if c and a[j] >= 2:
            b = list(a)
            b[j] -= 2
            terms.append(f"{c}.0*d[{lookup[tuple(b)]}]")
    emit(f"d[{k}]=-qi*(" + "+".join(terms) + ");")
emit("}")

emit("static __attribute__((noinline)) void m2l4(const double *m0, const double *m1, const double *m2, const double *m3, V *l, V x, V y, V z) {")
emit("V d[NC],m[NM]; derivatives4(x,y,z,d);")
emit("int k=0;")
emit("for(;k+4<=NM;k+=4) {")
emit("V a=_mm256_loadu_pd(m0+k), b=_mm256_loadu_pd(m1+k), c=_mm256_loadu_pd(m2+k), e=_mm256_loadu_pd(m3+k);")
emit("V u=_mm256_unpacklo_pd(a,b),v=_mm256_unpackhi_pd(a,b),w=_mm256_unpacklo_pd(c,e),t=_mm256_unpackhi_pd(c,e);")
emit("m[k]=_mm256_permute2f128_pd(u,w,0x20); m[k+1]=_mm256_permute2f128_pd(v,t,0x20);")
emit("m[k+2]=_mm256_permute2f128_pd(u,w,0x31); m[k+3]=_mm256_permute2f128_pd(v,t,0x31);")
emit("}")
emit("for(;k<NM;k++) m[k]=V{m0[k],m1[k],m2[k],m3[k]};")
for a in local:
    terms = []
    for b in source:
        if degree(a)+degree(b) <= p:
            kd = lookup[tuple(x+y for x, y in zip(a, b))]
            terms.append(f"m[{lookup[b]}]*d[{kd}]")
    emit(f"l[{lookup[a]}]+=" + "+".join(terms) + ";")
emit("}")

emit("static inline void m2l(const double *ma, const double *mb, double *la, double *lb, double x, double y, double z) {")
emit("double d[NC]; derivatives(x,y,z,d);")
for a in local:
    ka = lookup[a]
    ta, tb = [], []
    for b in source:
        if degree(a)+degree(b) > p:
            continue
        kd = lookup[tuple(x+y for x, y in zip(a, b))]
        kb = lookup[b]
        ta.append(f"mb[{kb}]*d[{kd}]")
        term = f"ma[{kb}]*d[{kd}]"
        if (degree(a)+degree(b)) % 2:
            term = "-" + term
        tb.append(term)
    emit(f"la[{ka}]+=" + "+".join(ta) + ";")
    emit(f"lb[{ka}]+=" + "+".join(tb).replace("+-", "-") + ";")
emit("}")

emit("static inline void m2p(const double *m, double *l, double x, double y, double z, double mass, double &ax, double &ay, double &az) {")
emit("double d[NC]; derivatives(x,y,z,d);")
for axis,name in enumerate(("ax","ay","az")):
    terms = []
    for b in source:
        a = list(b)
        a[axis] += 1
        terms.append(f"m[{lookup[b]}]*d[{lookup[tuple(a)]}]")
    emit(name+"+="+"+".join(terms)+";")
for a in local:
    sign = "-" if degree(a)%2 else "+"
    emit(f"l[{lookup[a]}]{sign}=mass*d[{lookup[a]}];")
emit("}")

emit("static __attribute__((noinline)) void m2p4(const double *m, V *l, V x, V y, V z, V mass, V &ax, V &ay, V &az) {")
emit("V d[NC]; derivatives4(x,y,z,d);")
for axis,name in enumerate(("ax","ay","az")):
    terms = []
    for b in source:
        a = list(b)
        a[axis] += 1
        terms.append(f"m[{lookup[b]}]*d[{lookup[tuple(a)]}]")
    emit(name+"+="+"+".join(terms)+";")
for a in local:
    sign = "-" if degree(a)%2 else "+"
    emit(f"l[{lookup[a]}]{sign}=mass*d[{lookup[a]}];")
emit("}")

emit("static inline void monomials(double x, double y, double z, double *v) {")
emit(f"double xx[{p+1}],yy[{p+1}],zz[{p+1}];")
emit("xx[0]=yy[0]=zz[0]=1;")
for k in range(1, p+1):
    emit(f"xx[{k}]=xx[{k-1}]*x/{k}.0;")
    emit(f"yy[{k}]=yy[{k-1}]*y/{k}.0;")
    emit(f"zz[{k}]=zz[{k-1}]*z/{k}.0;")
for a in indices:
    i,j,k = a
    emit(f"v[{lookup[a]}]=xx[{i}]*yy[{j}]*zz[{k}];")
emit("}")

emit("static inline void shift_m(const double *m, double *q, double x, double y, double z) {")
emit("double v[NC]; monomials(-x,-y,-z,v);")
for a in source:
    terms = []
    for b in source:
        if all(x >= y for x, y in zip(a,b)):
            c = tuple(x-y for x,y in zip(a,b))
            terms.append(f"m[{lookup[b]}]*v[{lookup[c]}]")
    emit(f"q[{lookup[a]}]+=" + "+".join(terms) + ";")
emit("}")

emit("static inline void shift_l(const double *l, double *q, double x, double y, double z) {")
emit("double v[NC]; monomials(x,y,z,v);")
for a in local:
    terms = []
    for b in local:
        if all(y >= x for x,y in zip(a,b)):
            c = tuple(y-x for x,y in zip(a,b))
            terms.append(f"l[{lookup[b]}]*v[{lookup[c]}]")
    emit(f"q[{lookup[a]}]+=" + "+".join(terms) + ";")
emit("}")

emit("static inline void evaluate_l(const double *l, double x, double y, double z, double &ax, double &ay, double &az) {")
emit("double v[NC]; monomials(x,y,z,v);")
for axis,name in enumerate(("ax","ay","az")):
    terms = []
    for a in indices:
        if degree(a) >= p:
            continue
        b = list(a)
        b[axis] += 1
        terms.append(f"l[{lookup[tuple(b)]}]*v[{lookup[a]}]")
    emit(name + "+=" + "+".join(terms) + ";")
emit("}")

q = min(p,int(sys.argv[3]) if len(sys.argv)>3 else 8)
rq = [a for a in indices if degree(a) <= q]
sq = [a for a in rq if degree(a) < q and degree(a) != 1]
lq = rq[1:]
emit(f"constexpr int RES_ORDER={q},RES_NC={len(rq)},RES_NM={sum(degree(a)<q for a in rq)};")
emit("static __attribute__((noinline)) void residual_derivatives4(V x,V y,V z,V *d) {")
emit("V ds[RES_NC],dn[RES_NC]; const V rr=x*x+y*y+z*z; const V qi=1.0/(rr+0.0001),qn=1.0/rr;")
emit("ds[0]=_mm256_sqrt_pd(qi); dn[0]=_mm256_sqrt_pd(qn);")
for a in rq[1:]:
    axis=next(k for k in range(3) if a[k])
    terms=[]
    for j in range(3):
        if a[j]:
            b=list(a); b[j]-=1
            coef=2*a[j]-(1 if j==axis else 0)
            terms.append(f"{coef}.0*{xyz[j]}*D[{lookup[tuple(b)]}]")
        coef=(a[j]-1)**2 if j==axis else a[j]*(a[j]-1)
        if coef and a[j]>=2:
            b=list(a); b[j]-=2
            terms.append(f"{coef}.0*D[{lookup[tuple(b)]}]")
    expr="+".join(terms)
    emit(f"ds[{lookup[a]}]=-qi*("+expr.replace("D[","ds[")+");")
    emit(f"dn[{lookup[a]}]=-qn*("+expr.replace("D[","dn[")+");")
emit("for(int k=0;k<RES_NC;k++) d[k]=ds[k]-dn[k];")
emit("}")

emit("static __attribute__((noinline)) void residual_m2l4(const double *m0,const double *m1,const double *m2,const double *m3,V *l,V x,V y,V z) {")
emit("V d[RES_NC],m[RES_NM]; residual_derivatives4(x,y,z,d);")
emit("int k=0; for(;k+4<=RES_NM;k+=4) {")
emit("V a=_mm256_loadu_pd(m0+k),b=_mm256_loadu_pd(m1+k),c=_mm256_loadu_pd(m2+k),e=_mm256_loadu_pd(m3+k);")
emit("V u=_mm256_unpacklo_pd(a,b),v=_mm256_unpackhi_pd(a,b),w=_mm256_unpacklo_pd(c,e),t=_mm256_unpackhi_pd(c,e);")
emit("m[k]=_mm256_permute2f128_pd(u,w,0x20); m[k+1]=_mm256_permute2f128_pd(v,t,0x20);")
emit("m[k+2]=_mm256_permute2f128_pd(u,w,0x31); m[k+3]=_mm256_permute2f128_pd(v,t,0x31);")
emit("} for(;k<RES_NM;k++) m[k]=V{m0[k],m1[k],m2[k],m3[k]};")
for a in lq:
    terms=[]
    for b in sq:
        if degree(a)+degree(b)<=q:
            c=tuple(x+y for x,y in zip(a,b))
            terms.append(f"m[{lookup[b]}]*d[{lookup[c]}]")
    emit(f"l[{lookup[a]}]+="+"+".join(terms)+";")
emit("}")

emit("static __attribute__((noinline)) void residual_derivatives(double x,double y,double z,double *d) {")
emit("double ds[RES_NC],dn[RES_NC]; const double rr=x*x+y*y+z*z; const double qi=1.0/(rr+0.0001),qn=1.0/rr;")
emit("ds[0]=std::sqrt(qi); dn[0]=std::sqrt(qn);")
for a in rq[1:]:
    axis=next(k for k in range(3) if a[k])
    terms=[]
    for j in range(3):
        if a[j]:
            b=list(a); b[j]-=1
            coef=2*a[j]-(1 if j==axis else 0)
            terms.append(f"{coef}.0*{xyz[j]}*D[{lookup[tuple(b)]}]")
        coef=(a[j]-1)**2 if j==axis else a[j]*(a[j]-1)
        if coef and a[j]>=2:
            b=list(a); b[j]-=2
            terms.append(f"{coef}.0*D[{lookup[tuple(b)]}]")
    expr="+".join(terms)
    emit(f"ds[{lookup[a]}]=-qi*("+expr.replace("D[","ds[")+");")
    emit(f"dn[{lookup[a]}]=-qn*("+expr.replace("D[","dn[")+");")
emit("for(int k=0;k<RES_NC;k++) d[k]=ds[k]-dn[k];")
emit("}")

emit("static __attribute__((noinline)) void residual_m2l_mutual(const double *__restrict ma,const double *__restrict mb,double *__restrict la,double *__restrict lb,double x,double y,double z) {")
emit("double d[RES_NC]; residual_derivatives(x,y,z,d);")
for bn in range(q):
    if bn==1:
        continue
    for bx in range(bn+1):
        bs=lookup[(bx,0,bn-bx)]
        emit(f"for(int by=0;by<={bn-bx};by++) {{")
        emit(f"const double am=ma[{bs}+by],bm=mb[{bs}+by];")
        emit("const V a4={am,am,am,am},b4={bm,bm,bm,bm};")
        emit("const __m128d a2=_mm256_castpd256_pd128(a4),b2=_mm256_castpd256_pd128(b4);")
        for an in range(1,q-bn+1):
            for ax in range(an+1):
                ls=lookup[(ax,0,an-ax)]
                ds=lookup[(ax+bx,0,an+bn-ax-bx)]
                length=an-ax+1
                pos=0
                negative=(an+bn)%2
                while pos+4<=length:
                    emit("{")
                    emit(f"V v=_mm256_loadu_pd(d+{ds+pos}+by);")
                    emit(f"_mm256_storeu_pd(la+{ls+pos},_mm256_fmadd_pd(b4,v,_mm256_loadu_pd(la+{ls+pos})));")
                    op="_mm256_fnmadd_pd" if negative else "_mm256_fmadd_pd"
                    emit(f"_mm256_storeu_pd(lb+{ls+pos},{op}(a4,v,_mm256_loadu_pd(lb+{ls+pos})));")
                    emit("}")
                    pos+=4
                if pos+2<=length:
                    emit("{")
                    emit(f"__m128d v=_mm_loadu_pd(d+{ds+pos}+by);")
                    emit(f"_mm_storeu_pd(la+{ls+pos},_mm_fmadd_pd(b2,v,_mm_loadu_pd(la+{ls+pos})));")
                    op="_mm_fnmadd_pd" if negative else "_mm_fmadd_pd"
                    emit(f"_mm_storeu_pd(lb+{ls+pos},{op}(a2,v,_mm_loadu_pd(lb+{ls+pos})));")
                    emit("}")
                    pos+=2
                if pos<length:
                    emit(f"la[{ls+pos}]+=bm*d[{ds+pos}+by];")
                    sign="-" if negative else "+"
                    emit(f"lb[{ls+pos}]{sign}=am*d[{ds+pos}+by];")
        emit("}")
emit("}")

emit("static __attribute__((noinline)) void residual_m2p4(const double *m,V *l,V x,V y,V z,V mass,V &ax,V &ay,V &az) {")
emit("V d[RES_NC]; residual_derivatives4(x,y,z,d);")
for axis,name in enumerate(("ax","ay","az")):
    terms=[]
    for b in sq:
        a=list(b); a[axis]+=1
        terms.append(f"m[{lookup[b]}]*d[{lookup[tuple(a)]}]")
    emit(name+"+="+"+".join(terms)+";")
for a in lq:
    sign="-" if degree(a)%2 else "+"
    emit(f"l[{lookup[a]}]{sign}=mass*d[{lookup[a]}];")
emit("}")
low=min(4,p)
low_idx=[a for a in indices if degree(a)<=low]
low_src=[a for a in low_idx if degree(a)<low and degree(a)!=1]
low_nm=sum(degree(a)<low for a in low_idx)
emit("static __attribute__((noinline)) void residual_m2l4_coarse(const double *m0,const double *m1,const double *m2,const double *m3,V *l,V x,V y,V z) {")
emit(f"V ds[{len(low_idx)}],dn[{len(low_idx)}],d[{len(low_idx)}],m[{low_nm}];")
emit("const V rr=x*x+y*y+z*z,qi=1.0/(rr+0.0001),qn=1.0/rr;")
emit("ds[0]=_mm256_sqrt_pd(qi); dn[0]=_mm256_sqrt_pd(qn);")
for a in low_idx[1:]:
    axis=next(k for k in range(3) if a[k])
    terms=[]
    for j in range(3):
        if a[j]:
            b=list(a); b[j]-=1
            coef=2*a[j]-(1 if j==axis else 0)
            terms.append(f"{coef}.0*{xyz[j]}*D[{lookup[tuple(b)]}]")
        coef=(a[j]-1)**2 if j==axis else a[j]*(a[j]-1)
        if coef and a[j]>=2:
            b=list(a); b[j]-=2
            terms.append(f"{coef}.0*D[{lookup[tuple(b)]}]")
    expr="+".join(terms)
    emit(f"ds[{lookup[a]}]=-qi*("+expr.replace("D[","ds[")+");")
    emit(f"dn[{lookup[a]}]=-qn*("+expr.replace("D[","dn[")+");")
emit(f"for(int k=0;k<{len(low_idx)};k++) d[k]=ds[k]-dn[k];")
emit(f"for(int k=0;k<{low_nm};k++) m[k]=V{{m0[k],m1[k],m2[k],m3[k]}};")
for a in low_idx[1:]:
    terms=[]
    for b in low_src:
        if degree(a)+degree(b)<=low:
            c=tuple(x+y for x,y in zip(a,b))
            terms.append(f"m[{lookup[b]}]*d[{lookup[c]}]")
    emit(f"l[{lookup[a]}]+="+"+".join(terms)+";")
emit("}")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
