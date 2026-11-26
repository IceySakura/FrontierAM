"""Numerical illustration of R3 on an independently generated ellipsoid."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from numerics import ellipsoid_prob


def main():
    rng = np.random.default_rng(7171)
    d = 8
    rho = 2.85
    V = np.linalg.qr(rng.normal(size=(d, d)))[0]
    A = (V*np.linspace(1.08, 1.88, d)) @ V.T
    b = V@np.linspace(-.22,.19,d)
    W = V*np.linspace(.38,.92,d)
    phases = rng.uniform(0, 4, size=d)
    c = np.linspace(.004,.009,d)
    lipschitz = np.sum(abs(c)*np.sum(abs(W),axis=0))
    k = max(1,int(np.ceil(64*lipschitz*rho)))
    i,j = np.triu_indices(d,1)
    E = np.eye(d)
    v = np.vstack([E,E[i]+E[j]])
    probe = np.vstack([v,-v])
    truth = ellipsoid_prob(A/rho**2,b/rho,-1.)

    def background(z):
        return .31+np.sin(z@W+phases)@c

    def f(z):
        z=np.atleast_2d(z)
        u=z/rho
        inside = np.sum((u@A)*u,axis=1)+u@b < 1.
        return background(z)+inside

    grid=np.linspace(.5,4,k+1)
    table=f((rho/grid[None,:,None]*probe[:,None,:]).reshape(-1,d)).reshape(len(probe),-1)
    diff=np.diff(table,axis=1)>.5
    assert np.all(diff.sum(axis=1)==1)
    first=diff.argmax(axis=1)
    nodes=rng.normal(size=(1024,d))
    # Include points on the true boundary to exercise the ambiguous branch.
    alpha=np.sum((probe@A)*probe,axis=1)
    beta=probe@b
    exact_s=(beta+np.sqrt(beta*beta+4*alpha))*.5
    nodes=np.vstack([nodes,rho*probe[:12]/exact_s[:12,None]])
    u=nodes/rho
    print(f"d={d}, rho={rho}, Lipschitz constant={lipschitz:.6f}, k={k}")
    print("steps  training    relative mass error    theory bound    decoder extra calls")
    for steps in (12,16,20,24):
        delta=80*d*2.**(-steps)
        if delta>=1 or 16*lipschitz*rho*delta>1:
            continue
        lo,hi=grid[first].copy(),grid[first+1].copy()
        flo=table[np.arange(len(probe)),first].copy()
        for _ in range(steps):
            mid=.5*(lo+hi)
            fm=f(rho*probe/mid[:,None])
            inside=fm-flo>.5
            hi=np.where(inside,mid,hi)
            lo=np.where(inside,lo,mid)
            flo=np.where(inside,flo,fm)
        r,s=np.split(.5*(lo+hi),2)
        q=r*s
        bb=(r-s)[:d]
        AA=np.diag(q[:d])
        AA[i,j]=AA[j,i]=.5*(q[d:]-q[i]-q[j])
        # On this example the learned ellipsoid is already inside the clip.
        assert np.linalg.eigvalsh(AA)[0]*3-np.sqrt(3)*np.linalg.norm(bb)>1
        I=ellipsoid_prob(AA/rho**2,bb/rho,-1.)
        p=np.sum((u@AA)*u,axis=1)+u@bb
        active=np.sum(u*u,axis=1)<=3
        close=active&(p>=1-delta)&(p<1+delta)
        yes=active&(p<1-delta)
        fv=f(nodes)
        yes[close]=fv[close]-f(nodes[close]/(1+2*delta))>-.5
        h=fv-yes
        np.testing.assert_allclose(h,background(nodes),rtol=0,atol=5e-16)
        print(f"{steps:5d}  {len(probe)*(k+1+steps):8d}    "
              f"{abs(I/truth-1):19.6g}  {(1+delta)**d-1:13.6g}    {close.sum():6d}")


def rank_demo():
    rng=np.random.default_rng(194321)
    d=64
    rho=2.2
    print("\nrank   inferred   queries   relative Frobenius error")
    for r in (2,7,19):
        V=np.linalg.qr(rng.standard_normal((d,r)),mode="reduced")[0]
        A=(V*np.linspace(.41,1.84,r))@V.T
        E=np.eye(d)
        calls=0

        def radii(v):
            nonlocal calls
            z=(rho/4)*v
            u=z/rho
            calls+=len(z)
            f=np.maximum(0.,1.-np.sum((u@A)*u,axis=1))
            return 16.*(1.-f)

        D=radii(E)
        B=np.zeros((d,d))
        inferred=0
        for _ in range(d):
            slack=D-B.diagonal()
            j=int(np.argmax(slack))
            p=slack[j]
            # Only the demonstration uses this floating-point tolerance.
            if p<1e-11:
                break
            J=np.arange(d)!=j
            a=D.copy()
            a[J]=.5*(radii(E[J]+E[j])-D[J]-D[j])
            y=a-B[j]
            B+= np.outer(y,y)/p
            inferred+=1
        print(f"{r:4d}  {inferred:8d}  {calls:8d}  {np.linalg.norm(B-A)/np.linalg.norm(A):24.6g}")


def binary_cylinder_demo():
    rng=np.random.default_rng(254451)
    d=64
    rho=2.1
    U=np.linalg.qr(rng.standard_normal((d,2)),mode="reduced")[0]
    w=U[:,0]
    E=np.eye(d)
    print("\nrank   steps   queries   entry max error   relative mass error")
    for rank,lam in [(1,[1.31,0.]),(2,[1.31,.43])]:
        A=(U*np.asarray(lam))@U.T
        truth=ellipsoid_prob(A/rho**2,np.zeros(d),-1.)
        for m in (30,34):
            calls=0

            def measure(v):
                nonlocal calls
                v=np.atleast_2d(v)
                lo=np.full(len(v),.5)
                hi=np.full(len(v),4.)
                for _ in range(m):
                    s=.5*(lo+hi)
                    u=(rho/s[:,None])*v
                    assert np.max(abs(u))<=40
                    y=u/rho
                    inside=np.sum((y@A)*y,axis=1)<1
                    calls+=len(v)
                    lo=np.where(inside,lo,s)
                    hi=np.where(inside,s,hi)
                return ((lo+hi)*.5)**2

            a=measure(w)[0]
            P,N=np.split(measure(np.vstack([w+E/16,w-E/16])),2)
            D=128*(P+N-2*a)
            b=4*(P-N)
            B=np.outer(b,b)/a
            residual=D-B.diagonal()
            j=int(np.argmax(residual))
            t=600*16*2.**(-m)
            if residual[j]>2*t:
                r=128*(measure(w+(E[j]+E)/16)-P[j]-P+a)-B[j]
                B+=np.outer(r,r)/residual[j]
            I=ellipsoid_prob(B/rho**2,np.zeros(d),-1.)
            print(f"{rank:4d}  {m:6d}  {calls:8d}  "
                  f"{np.max(abs(B-A)):16.6g}   {abs(I/truth-1):19.6g}")


if __name__ == "__main__":
    main()
    rank_demo()
    binary_cylinder_demo()
