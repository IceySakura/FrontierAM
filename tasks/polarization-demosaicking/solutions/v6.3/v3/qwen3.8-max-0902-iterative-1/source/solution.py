"""Polarization (DoFP) mosaic demosaicking, two-pass variant.

Mosaic layout (row parity, col parity) -> measured angle:
  (even, even) -> 90, (even, odd) -> 45, (odd, even) -> 135, (odd, odd) -> 0

Pass 1 reconstructs the four planes as I_k = C + D_k where C is a
Nyquist-nulling 5-tap guide, D_k the residual known exactly on channel k's
quarter lattice, extended by separable bilinear interpolation and by a guided
local-linear model D_k ~ a*C + b fitted on the lattice samples of the 5x5
window around each pixel; the result is blended against the plain
per-channel bilinear interpolant with a leave-one-out (on the sampled
lattice) spatially varying weight.

Pass 2 repeats the same construction with the guide replaced by the channel
average of the pass-1 reconstruction, which estimates the common intensity
component with about half the error of the filtered-mosaic guide.
Sampled pixels are reproduced exactly in both passes.

Returns float64 array of shape (4, H, W), order [0, 45, 90, 135].
Single threaded, ~0.45 s per 1024x1024 scene.
"""
import numpy as np, sys, time
from scipy.ndimage import uniform_filter
LAT = {0: (1, 1), 1: (0, 1), 2: (0, 0), 3: (1, 0)}


def lat_interp(L, rr, cc, out):
    Lp = np.pad(L, 1, mode='reflect')
    out[rr::2, cc::2] = L
    out[rr::2, 1 - cc::2] = 0.5 * (Lp[1:-1, 1:-1] + Lp[1:-1, 2:])
    out[1 - rr::2, cc::2] = 0.5 * (Lp[1:-1, 1:-1] + Lp[2:, 1:-1])
    out[1 - rr::2, 1 - cc::2] = 0.25 * (Lp[1:-1, 1:-1] + Lp[1:-1, 2:] + Lp[2:, 1:-1] + Lp[2:, 2:])
    return out


def _sep5(x, b):
    """Separable [b, .25, 1-2b-.5, .25, b] low-pass (nulls Nyquist)."""
    c = 0.25
    h0 = 1.0 - 2.0 * b - 2.0 * c
    o = np.empty_like(x)
    o[:, 2:-2] = b * x[:, :-4] + c * x[:, 1:-3] + h0 * x[:, 2:-2] + c * x[:, 3:-1] + b * x[:, 4:]
    o[:, 0] = h0 * x[:, 0] + 2 * c * x[:, 1] + 2 * b * x[:, 2]
    o[:, 1] = c * x[:, 0] + h0 * x[:, 1] + c * x[:, 2] + b * x[:, 3]
    o[:, -1] = h0 * x[:, -1] + 2 * c * x[:, -2] + 2 * b * x[:, -3]
    o[:, -2] = c * x[:, -1] + h0 * x[:, -2] + c * x[:, -3] + b * x[:, -4]
    y = np.empty_like(x)
    y[2:-2, :] = b * o[:-4, :] + c * o[1:-3, :] + h0 * o[2:-2, :] + c * o[3:-1, :] + b * o[4:, :]
    y[0, :] = h0 * o[0, :] + 2 * c * o[1, :] + 2 * b * o[2, :]
    y[1, :] = c * o[0, :] + h0 * o[1, :] + c * o[2, :] + b * o[3, :]
    y[-1, :] = h0 * o[-1, :] + 2 * c * o[-2, :] + 2 * b * o[-3, :]
    y[-2, :] = c * o[-1, :] + h0 * o[-2, :] + c * o[-3, :] + b * o[-4, :]
    return y


def _shc(X, d):
    """shc(X,d)[u,t] = X[u, t+d] with edge replication."""
    if d == 0:
        return X
    if d > 0:
        return np.concatenate([X[:, d:], np.repeat(X[:, -1:], d, axis=1)], axis=1)
    return np.concatenate([np.repeat(X[:, :1], -d, axis=1), X[:, :d]], axis=1)


def _shr(X, d):
    """shr(X,d)[u,t] = X[u+d, t] with edge replication."""
    if d == 0:
        return X
    if d > 0:
        return np.concatenate([X[d:, :], np.repeat(X[-1:, :], d, axis=0)], axis=0)
    return np.concatenate([np.repeat(X[:1, :], -d, axis=0), X[:d, :]], axis=0)


def wsum(A, oh, ov):
    """Window sums over the lattice grid for the four position classes:
    own 3x3, h 3x2, v 2x3, d 2x2 (with per-channel offsets oh, ov)."""
    Pv = np.pad(A, 1, mode='reflect')
    R3 = Pv[0:-2, 1:-1] + Pv[1:-1, 1:-1] + Pv[2:, 1:-1]
    H3 = Pv[1:-1, 0:-2] + Pv[1:-1, 1:-1] + Pv[1:-1, 2:]
    S2 = A + Pv[2:, 1:-1]
    H3u = np.pad(H3, 1, mode='reflect')
    own = H3u[0:-2, 1:-1] + H3u[1:-1, 1:-1] + H3u[2:, 1:-1]
    h = _shc(R3, oh) + _shc(R3, oh + 1)
    v = _shr(H3, ov) + _shr(H3, ov + 1)
    S2a = _shc(S2, oh)
    S2b = _shc(S2, oh + 1)
    d = _shr(S2a, ov) + _shr(S2b, ov)
    return {'own': own, 'h': h, 'v': v, 'd': d}


_wsum = wsum


def one_pass(r, C, b, gb, eps, lwsz):
    H,W=r.shape
    c0=0.25; h0=1-2*b-2*c0
    P=np.pad(r,2,mode='reflect')
    outs=[]
    for k,(rr,cc) in LAT.items():
        rs,cs=1-rr,1-cc
        oh=(cs-cc-1)//2; ov=(rs-rr-1)//2
        nrr=len(range(rr,H,2)); ncs_=len(range(cs,W,2)); nrs=len(range(rs,H,2)); ncc=len(range(cc,W,2))
        y=r[rr::2,cc::2]; nr,nc=y.shape
        B=np.empty((H,W),np.float32); B[rr::2,cc::2]=y
        B[rr::2,cs::2]=0.5*(P[rr+2:rr+2+2*nrr:2,cs+1:cs+1+2*ncs_:2]+P[rr+2:rr+2+2*nrr:2,cs+3:cs+3+2*ncs_:2])
        B[rs::2,cc::2]=0.5*(P[rs+1:rs+1+2*nrs:2,cc+2:cc+2+2*ncc:2]+P[rs+3:rs+3+2*nrs:2,cc+2:cc+2+2*ncc:2])
        B[rs::2,cs::2]=0.25*(P[rs+1:rs+1+2*nrs:2,cs+1:cs+1+2*ncs_:2]+P[rs+1:rs+1+2*nrs:2,cs+3:cs+3+2*ncs_:2]+
                             P[rs+3:rs+3+2*nrs:2,cs+1:cs+1+2*ncs_:2]+P[rs+3:rs+3+2*nrs:2,cs+3:cs+3+2*ncs_:2])
        Cl=C[rr::2,cc::2]; Dl=y-Cl
        Da=_shc(Dl,oh); Db=_shc(Dl,oh+1)
        Rb=np.empty((H,W),np.float32)
        Rb[rr::2,cc::2]=Dl
        Rb[rr::2,cs::2]=0.5*(Da+Db)
        Rb[rs::2,cc::2]=0.5*(_shr(Dl,ov)+_shr(Dl,ov+1))
        Rb[rs::2,cs::2]=0.25*(_shr(Da,ov)+_shr(Db,ov)+_shr(Da,ov+1)+_shr(Db,ov+1))
        Q={}
        for name,A in (('one',np.ones((nr,nc),np.float32)),('C',Cl),('CC',Cl*Cl),('CD',Cl*Dl),('D',Dl)):
            Q[name]=_wsum(A,oh,ov)
        al=np.empty((H,W),np.float32); be=np.empty((H,W),np.float32)
        for cname,sl in (('own',(slice(rr,H,2),slice(cc,W,2))),('h',(slice(rr,H,2),slice(cs,W,2))),
                         ('v',(slice(rs,H,2),slice(cc,W,2))),('d',(slice(rs,H,2),slice(cs,W,2)))):
            cnt=np.maximum(Q['one'][cname],1e-6)
            muC=Q['C'][cname]/cnt; muD=Q['D'][cname]/cnt
            varC=Q['CC'][cname]/cnt-muC*muC; covCD=Q['CD'][cname]/cnt-muC*muD
            alv=covCD/(varC+eps); al[sl]=alv; be[sl]=muD-alv*muC
        G=C+gb*(al*C+be)+(1-gb)*Rb
        pl=np.pad(y,1,mode='reflect')
        Bloo=0.25*(pl[0:nr,0:nc]+pl[0:nr,2:nc+2]+pl[2:nr+2,0:nc]+pl[2:nr+2,2:nc+2])
        d=Bloo-y
        pd=np.pad(d,2,mode='reflect')
        s=((h0*h0)*d+h0*b*(pd[0:nr,2:nc+2]+pd[4:nr+4,2:nc+2]+pd[2:nr+2,0:nc]+pd[2:nr+2,4:nc+4])
           +b*b*(pd[0:nr,0:nc]+pd[0:nr,4:nc+4]+pd[4:nr+4,0:nc]+pd[4:nr+4,4:nc+4]))
        Cloo=Cl+s; Dloo=Bloo-Cloo
        pdl=np.pad(Dloo,1,mode='reflect')
        Gloo=Cloo+0.25*(pdl[0:nr,0:nc]+pdl[0:nr,2:nc+2]+pdl[2:nr+2,0:nc]+pdl[2:nr+2,2:nc+2])
        e1=d; e2=Gloo-y
        n11=uniform_filter(e1*e1,lwsz,mode='reflect'); n12=uniform_filter(e1*e2,lwsz,mode='reflect'); n22=uniform_filter(e2*e2,lwsz,mode='reflect')
        den=n11-2*n12+n22
        wl=np.clip(np.where(den>1e-9,(n11-n12)/np.maximum(den,1e-9),1.0),0,1).astype(np.float32)
        Wa=_shc(wl,oh); Wb=_shc(wl,oh+1)
        Wf=np.empty((H,W),np.float32)
        Wf[rr::2,cc::2]=wl
        Wf[rr::2,cs::2]=0.5*(Wa+Wb)
        Wf[rs::2,cc::2]=0.5*(_shr(wl,ov)+_shr(wl,ov+1))
        Wf[rs::2,cs::2]=0.25*(_shr(Wa,ov)+_shr(Wb,ov)+_shr(Wa,ov+1)+_shr(Wb,ov+1))
        o=(1-Wf)*B+Wf*G; o[rr::2,cc::2]=y
        wopt=np.clip((n11-n12)/np.maximum(den,1e-9),0,1)
        met=float(np.mean(e1*e1-wopt*(n11-n12)))
        outs.append((o,met))
    return outs
def m20(raw,mode='mean',b=-0.05,gb=0.4,eps=40.0,lwsz=9):  # tuned
    r0=np.asarray(raw,np.float32); H0,W0=r0.shape
    if H0<8 or W0<8:
        rf=np.asarray(raw,np.float64)
        return np.stack([rf,rf,rf,rf])
    if H0%2 or W0%2:
        rp=np.pad(r0,((0,H0%2),(0,W0%2)),mode='reflect')
        return m20(rp,mode,b,gb,eps,lwsz)[:, :H0, :W0]
    r=r0; H,W=r.shape
    C=_sep5(r.astype(np.float64),b).astype(np.float32)
    p1=one_pass(r,C,b,gb,eps,lwsz)
    if mode=='none':
        return np.stack([o for o,_ in p1]).astype(np.float64)
    Gm=np.mean([o for o,_ in p1],axis=0)
    if mode in ('mean','sel'): C2=Gm
    elif mode=='sep': C2=_sep5(Gm.astype(np.float64),b).astype(np.float32)
    elif mode=='half': C2=0.5*C+0.5*Gm
    p2=one_pass(r,C2,b,gb,eps,lwsz)
    if mode=='sel':
        return np.stack([o2 if m2<m1 else o1 for (o1,m1),(o2,m2) in zip(p1,p2)]).astype(np.float64)
    return np.stack([o for o,_ in p2]).astype(np.float64)


def reconstruct(raw):
    return m20(raw, 'mean')
