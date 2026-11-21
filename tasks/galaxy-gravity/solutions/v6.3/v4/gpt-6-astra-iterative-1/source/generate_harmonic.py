#!/usr/bin/env python3
from pathlib import Path
from math import comb


def indices(p):
    return [(x,d-z-x,z) for d in range(p+1) for z in range(min(1,d)+1)
            for x in range(d-z,-1,-1)]


def dot(terms, groups=4):
    if not terms:
        return "T(0.)"
    return "+".join("("+"".join(terms[j::groups]).lstrip("+")+")"
                    for j in range(min(groups,len(terms))))


def make(p):
    idx = indices(p)
    lookup = {a:i for i,a in enumerate(idx)}
    ext = [(x,d-2-x,2) for d in range(2,p+1) for x in range(d-2,-1,-1)]
    extlookup = dict(lookup)
    extlookup.update({a:len(idx)+i for i,a in enumerate(ext)})
    nm = p*p
    out = [
        f"template<> struct Harmonic<{p}> {{",
        f"static constexpr int P={p},NC={(p+1)**2},NM={nm},ND={len(idx)+len(ext)};",
        f"static constexpr int TERMS={comb(p+4,4)+2*comb(p+3,4)+comb(p+2,4)-(p+1)**2-3*(p*p-1)};",
        "static constexpr int powers[NC][3]={",
        ",".join("{%d,%d,%d}" % a for a in idx), "};",
        "template<class T> static void monomials(T x,T y,T z,T* h) {",
        "h[0]=T(1.);"
    ]
    for i,a in enumerate(idx[1:nm],1):
        x,y,z=a
        terms=[]
        if x: terms.append(f"+x*h[{lookup[(x-1,y,z)]}]")
        if y: terms.append(f"+y*h[{lookup[(x,y-1,z)]}]")
        if z:
            terms.append(f"+z*h[{lookup[(x,y,0)]}]")
        else:
            if x>=2: terms.append(f"-z*h[{lookup[(x-2,y,1)]}]")
            if y>=2: terms.append(f"-z*h[{lookup[(x,y-2,1)]}]")
        out.append(f"h[{i}]=("+"".join(terms).lstrip("+")+f")*T({1/sum(a):.17g});")
    out += ["}", "static void derivatives(V x,V y,V z,V inv,V* d) {",
            "V tau=(x*x+y*y+z*z)*inv;", "d[0]=vsqrt(inv);"]
    xyz=["x","y","z"]
    for i,a in enumerate(idx[1:],1):
        axis=max(range(3),key=lambda j:a[j])
        terms=[]
        for j in range(3):
            if not a[j]: continue
            b=list(a); b[j]-=1
            c=2*a[j]-1 if j==axis else 2*a[j]
            terms.append(f"V({c}.0)*{xyz[j]}*d[{lookup[tuple(b)]}]")
            if a[j]>=2:
                b[j]-=1
                c=(a[j]-1)**2 if j==axis else a[j]*(a[j]-1)
                terms.append(f"V({c}.0)*tau*d[{lookup[tuple(b)]}]")
        out.append(f"d[{i}]=-inv*("+ "+".join(terms)+");")
    for a in ext:
        x,y,z=a
        out.append(f"d[{extlookup[a]}]=-d[{lookup[(x+2,y,0)]}]-d[{lookup[(x,y+2,0)]}];")
    out += ["}", "static void extend(const double* src,double* dst) {"]
    out.append("std::copy(src,src+NC,dst);")
    for a in ext:
        x,y,z=a
        out.append(f"dst[{extlookup[a]}]=-src[{lookup[(x+2,y,0)]}]-src[{lookup[(x,y+2,0)]}];")
    out += ["}"]
    cartorders=[q for q in (6,8,10,12) if q<=p]
    for q in cartorders:
        out.append(f"static void cartesian{q}(const V* h,V* d) {{")
        cart=[(x,y,n-x-y) for n in range(q+1) for x in range(n,-1,-1) for y in range(n-x,-1,-1)]
        for i,(x,y,z) in enumerate(cart):
            k=z//2
            terms=[]
            for j in range(k+1):
                coeff=comb(k,j)*((-1)**k)
                terms.append(f"{coeff:+d}.0*h[{lookup[(x+2*j,y+2*(k-j),z%2)]}]")
            out.append(f"d[{i}]="+"".join(terms).lstrip("+")+";")
        out.append("}")
    out += ["template<int Q> static void cartesian(const V* h,V* d) {"]
    for q in cartorders:
        out.append(f"if constexpr(Q=={q}) cartesian{q}(h,d);")
    out += ["}"]

    def derivative(a,prefix):
        x,y,z=a
        return f"{prefix}[{extlookup[a]}]"

    out += ["static void translate(const V* ma,const V* mb,const V* d,V* la,V* lb) {"]
    for ib,b in enumerate(idx[1:],1):
        aa=[];bb=[]
        for ia,a in enumerate(idx[:nm]):
            if sum(a)+sum(b)>p: break
            if sum(a)==1: continue
            ab=tuple(a[j]+b[j] for j in range(3))
            dd=derivative(ab,"d")
            aa.append(("-" if sum(a)%2 else "+")+f"mb[{ia}]*{dd}")
            bb.append(f"+ma[{ia}]*{dd}")
        out.append(f"la[{ib}]="+dot(aa)+";")
        out.append(f"lb[{ib}]="+("-(" if sum(b)%2 else "(")+dot(bb)+");")
    out += ["}"]

    terms=[[] for _ in range(nm)]
    for ia,a in enumerate(idx[:nm]):
        for ib,b in enumerate(idx[:nm]):
            if sum(a)+sum(b)>=p: break
            c=tuple(a[j]+b[j] for j in range(3))
            t=f"src[{ia}]*shift[{ib}]"
            if c[2]<2:
                terms[lookup[c]].append("+"+t)
            else:
                terms[lookup[(c[0]+2,c[1],0)]].append("-"+t)
                terms[lookup[(c[0],c[1]+2,0)]].append("-"+t)
    out += ["static void shift_moment(const double* src,const double* shift,double* dst) {"]
    for i,t in enumerate(terms):
        out.append(f"dst[{i}]+="+dot(t)+";")
    out += ["}","static void shift_local(const double* source,const double* shift,double* dst) {",
            "double src[ND]; extend(source,src);"]
    for ib,b in enumerate(idx[1:],1):
        terms=[]
        for ia,a in enumerate(idx[:nm]):
            if sum(a)+sum(b)>p: break
            ab=tuple(a[j]+b[j] for j in range(3))
            terms.append(f"+shift[{ia}]*"+derivative(ab,"src"))
        out.append(f"dst[{ib}]+="+dot(terms)+";")
    out += ["}","template<class T> static void evaluate(const double* l,const T* t,T& fx,T& fy,T& fz) {"]
    for j,name in enumerate(["fx","fy","fz"]):
        terms=[]
        for ia,a in enumerate(idx[:nm]):
            b=list(a); b[j]+=1
            terms.append(f"+t[{ia}]*"+derivative(tuple(b),"l"))
        out.append(f"{name}+="+dot(terms)+";")
    out += ["}"]

    for local in (False,True):
        suffix="local" if local else "moment"
        count=len(idx) if local else nm
        first=1 if local else 0
        maxorder=p if local else p-1
        out += [f"static void shift_{suffix}_fast(const double* src,double dx,double dy,double dz,double* dst) {{",
                f"double a[{count}],b[{count}],pw[P+1];",
                f"std::copy(src,src+{count},a);"]
        for axis,coord in ((0,"dx"),(1,"dy")):
            out += ["pw[0]=1.;",f"for(int k=1;k<={maxorder};++k) pw[k]=pw[k-1]*{coord}/k;"]
            for i,tup in enumerate(idx[:count]):
                if i<first: continue
                terms=[f"+a[{i}]"]
                lim=maxorder-sum(tup) if local else tup[axis]
                for k in range(1,lim+1):
                    b=list(tup)
                    b[axis]+=k if local else -k
                    terms.append(f"+pw[{k}]*a[{lookup[tuple(b)]}]")
                out.append(f"b[{i}]="+dot(terms)+";")
            out += [f"std::copy(b+{first},b+{count},a+{first});"]
        out += [f"for(int i={first};i<{count};++i) dst[i]+=a[i];"]
        one=[];two=[];sign=[]
        for x,y,z in idx[:count]:
            if local:
                if x+y+z==p:
                    i1=i2=-1
                elif z:
                    i1=lookup[(x+2,y,0)]; i2=lookup[(x,y+2,0)]
                else:
                    i1=lookup[(x,y,1)]; i2=-1
                sg=-1 if z else 1
            else:
                if z:
                    i1=lookup[(x,y,0)]; i2=-1
                else:
                    i1=lookup[(x-2,y,1)] if x>=2 else -1
                    i2=lookup[(x,y-2,1)] if y>=2 else -1
                sg=1 if z else -1
            one.append(str(i1));two.append(str(i2));sign.append(str(sg))
        out += ["static constexpr int one[]={"+",".join(one)+"};",
                "static constexpr int two[]={"+",".join(two)+"};",
                "static constexpr int sign[]={"+",".join(sign)+"};",
                f"for(int k=1;k<={maxorder};++k) {{",
                "double s=dz/k;"]
        if local:
            out += ["int begin=1,end=(P-k+1)*(P-k+1);"]
        else:
            out += ["int begin=k*k,end=NM;"]
        out += ["for(int i=begin;i<end;++i) {",
                "double v=one[i]>=0?a[one[i]]:0.;",
                "if(two[i]>=0) v+=a[two[i]];",
                "b[i]=v*(s*sign[i]); dst[i]+=b[i];",
                "}",
                "std::copy(b+begin,b+end,a+begin);",
                "}", "}"]
    out += ["};"]
    return "\n".join(out)


Path(__file__).with_name("harmonic.hpp").write_text(
    "// Generated by generate_harmonic.py.\n"
    "template<int P> struct Harmonic { static constexpr int NC=1,NM=1,ND=1; };\n" +
    "\n".join(make(p) for p in [12,14,16]))
