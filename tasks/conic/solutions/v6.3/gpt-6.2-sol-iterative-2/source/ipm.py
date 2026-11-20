"""Homogeneous primal-dual predictor-corrector method for conic QPs."""
from time import perf_counter

import numpy as np
from scipy.linalg import eigvalsh

from cones import Soc, Exp, Psd, scalar_step
from kkt import Kkt, QrKkt
from model import Scaled, maxabs, smat


def identity(K):
    s, z = np.zeros(K.m), np.zeros(K.m)
    s[K.lpos] = z[K.lpos] = 1.
    for (k,d), starts in K.groups.items():
        if k == 'soc':
            s[starts] = z[starts] = 1.
        else:
            i = starts[:, None] + np.arange(3)
            s[i] = (-1.,1.,2.)
            z[i] = (-1.,0.,1.)
    for k,d,a,b in K.nonlinear:
        if k == 'psd':
            i = a + ((np.arange(1,d+1) * np.arange(2,d+2))//2-1)
            s[i] = z[i] = 1.
    return s, z


def shift_into_cone(K, v):
    i = K.lpos
    if i.size:
        vl = float(v[i].min())
        if vl < .1:
            v[i] += (1.-vl)
    for (k,d), starts in K.groups.items():
        if k == 'soc':
            nv = np.linalg.norm(v[starts[:, None]+np.arange(1, d, dtype=np.int32)], axis=1)
            v[starts] += np.maximum(0., .8+nv-v[starts])
    for k, d, a, b in K.nonlinear:
        if k == 'psd':
            vl = eigvalsh(smat(v[a:b], d), subset_by_index=(0,0),
                           check_finite=False)[0]
            i = a + (np.arange(1,d+1)*np.arange(2,d+2))//2-1
            v[i] += max(0., .8-float(vl))


class ConePack:
    def __init__(self, K, kkt, s, z, mu, exp_dual):
        self.s, self.z = s, z
        self.linear = K.lpos
        self.sl, self.zl = s[self.linear], z[self.linear]
        self.groups, self.psd = {}, {}
        self.ix = {}
        if np.any(self.zl <= 0.) or np.any(self.sl <= 0.):
            raise ValueError('nonpositive orthant iterate')
        self.linear_h = self.sl/self.zl
        for (k, d), loc in K.groups.items():
            ix = loc[:, None]+np.arange(d, dtype=np.int32)
            self.ix[k, d] = ix
            self.groups[k,d] = (Soc(s[ix], z[ix], d > 32, (k,d) in kkt.rotations) if k == 'soc'
                                else Exp(s[ix], z[ix], mu, exp_dual, (k,d) in kkt.rotations))
        for k, d, a, b in K.nonlinear:
            if k == 'psd':
                self.psd[a] = Psd(s[a:b], z[a:b], d, not kkt.ret[a])

    def step(self, ds, dz, top=1.):
        l = self.linear
        top = scalar_step(self.sl, ds[l], top)
        top = scalar_step(self.zl, dz[l], top)
        for key, st in self.groups.items():
            if key[0] == 'soc':
                ix = self.ix[key]
                top = st.step_one(st.s, ds[ix], top)
                top = st.step_one(st.z, dz[ix], top)
        for a, st in self.psd.items():
            b = a+st.d*(st.d+1)//2
            top = st.step(ds[a:b], dz[a:b], top)
        for key, st in self.groups.items():
            if key[0] == 'exponential':
                ix = self.ix[key]
                top = st.step(ds[ix], dz[ix], top)
        return top

    def correct(self, target, ds, dz, weight):
        q = -self.s
        i = self.linear
        q[i] += (target - weight*ds[i]*dz[i])/self.zl
        for key, st in self.groups.items():
            ix = self.ix[key]
            q[ix] += st.correct(target, ds[ix], dz[ix], weight)
        for a, st in self.psd.items():
            b = a+st.d*(st.d+1)//2
            q[a:b] += st.correct(target, ds[a:b], dz[a:b], weight)
        return q


def solve_ipm(p, deadline, *, maxiter=80, equi=True, rhs_scale=True, exp_dual=False,
              condense=True, verbose=0, start_style=1, reg=1.e-11, metric_limit=5.e-2,
              rotate=True, qr=False, bounds=True, dense=False, row_schur=True,
              early_polish=True, fixed_tau=False, warm=None, plain=False, extras=True):
    """Return a verified result, plus the best nearby point for polishing."""
    u = Scaled(p, scale_rhs=rhs_scale) if equi else p
    A, P, q, b, K = u.A, u.P, u.q, u.b, u.K
    At = A.T
    if equi:
        D, E, c = u.D, u.E, u.c
        yc = E/c
        dc = D*c
    else:
        D, E, c, yc, dc = 1., 1., 1., 1., 1.
    n, m = p.n, p.m
    s, z = identity(K)
    x = np.zeros(n)
    tau, kap = 1., 1.
    best, point = metric_limit, None
    last = np.inf
    stuck = 0
    polishes = 0
    polished_metric = np.inf
    step_previous = 1.
    timer = perf_counter()
    try:
        if dense:
            from dense import DenseKkt
            kkt = DenseKkt(A,P,K,reg)
        else:
            kkt = (QrKkt(A,P,K,reg) if qr else
                   Kkt(A,P,K,condense=condense,reg=reg,rotate=rotate,bounds=bounds,row_schur=row_schur))
        if warm is not None and not K.nonlinear and K.degree:
            xw,zw = warm
            x, z = xw/D,zw/yc
            s = b-A@x
            s[K.eq] = 0.
            ix = K.lpos
            sr,zr=s[ix],z[ix]
            pm=max(1.,maxabs(sr))
            dm=max(1.,maxabs(zr))
            res=max(maxabs((P@x+At@z+q)/dc)/(1.+p.qn),
                    float(np.max(-sr,initial=0.))/pm)
            if not np.isfinite(res) or res>5.:
                s,z=identity(K)
                x=np.zeros(n)
            else:
                theta=np.clip(.04*np.sqrt(res),.00001,.16)
                fs,fz=theta*pm,theta*dm
                s[ix]=np.maximum(sr,fs/(1.+np.maximum(zr,0.)/dm))
                z[ix]=np.maximum(zr,fz/(1.+np.maximum(sr,0.)/pm))
                kap=max(1.e-7,min(1.,float(s@z)/(K.degree+1.)))
        elif start_style and not K.exp and (not K.psd or start_style > 1) and n+m < 80_000:
            kkt.factor(None, initial=True)
            x, zl = kkt.solve(np.zeros(n) if not P.nnz else -q, b)
            s = -zl.copy()
            if not P.nnz:
                _, z = kkt.solve(-q, np.zeros(m))
            else:
                z = zl
            s[K.eq] = 0.
            shift_into_cone(K, s)
            shift_into_cone(K, z)
    except (ValueError, RuntimeError, np.linalg.LinAlgError, OverflowError, MemoryError):
        if verbose:
            import traceback
            traceback.print_exc()
        return None, point
    for it in range(maxiter+1):
        if it and perf_counter() > deadline:
            break
        if (not np.isfinite(tau) or not np.isfinite(kap) or tau <= 0. or kap <= 0.
            or not np.isfinite(x).all() or not np.isfinite(z).all()
            or not np.isfinite(s).all()):
            break
        Ax, Px, Atz = A@x, P@x, At@z
        qx, bz, xx, sz = float(q@x), float(b@z), float(x@Px), float(s@z)
        mu = (sz+tau*kap)/(K.degree+1.)
        rx = -Px - Atz - q*tau
        rp = b*tau - Ax - s
        rg = -(xx/tau + qx + bz + kap)
        gap = abs(xx/tau + qx + bz)/max(c*tau, abs(.5*xx/tau+qx), abs(.5*xx/tau+bz))
        dual = maxabs(rx/dc)/max(tau, maxabs(Px/dc), p.qn*tau, maxabs(Atz/dc))
        prim = maxabs(rp/E)/max(tau, maxabs(Ax/E), maxabs(s/E), p.bn*tau)
        ktr = kap/tau
        metric = max(dual, prim, gap)
        if verbose:
            print(" %2d  %.2e %.2e %.2e   mu %.2e tau %.2e kt %.2e  a %.3f t %.3f" %
                  (it, prim, dual, gap, mu, tau, ktr, step_previous, perf_counter()-timer), flush=True)
        if metric <= max(best, 2.1e-8):
            x0, s0, z0 = x*(D/tau), s/E/tau, z*(yc/tau)
            if metric < best:
                best = metric
                point = x0, s0, z0
            if metric <= 2.1e-8:
                ans = p.optimal(x0, z0, s0, polish_slack=False, monitor=2 if verbose > 1 else False)
                if ans is not None:
                    return ans, point
        if (early_polish and point is not None and polishes<2 and it>=5
                and p.n+p.m<92_000 and max(prim,dual)<5.e-5 and gap<2.e-3
                and metric*50.<polished_metric and perf_counter()<deadline-.045):
            allowed = (not K.nonlinear and bool(P.nnz)) or (K.exp and not K.psd and n+m<850)
            if allowed:
                # Refit active constraints before the barrier scales become
                # extremely unbalanced.
                polishes += 1
                polished_metric = metric
                try:
                    if not K.nonlinear:
                        from poly import polish_polyhedral
                        ans = polish_polyhedral(p,point[0],point[2],min(deadline,perf_counter()+.12))
                    else:
                        from polish import polish_cone
                        ans = polish_cone(p,point,min(deadline,perf_counter()+.105),steps=2)
                    if ans is not None:
                        return ans, point
                except (ValueError,RuntimeError,np.linalg.LinAlgError,MemoryError):
                    pass
        if ktr > 100. and (it % 3 == 0 or ktr > 1.e7):
            ans = p.primal_cert(yc*z)
            if ans is None:
                ans = p.dual_cert(D*x, s/E)
            if ans is not None:
                return ans, point
        if not np.isfinite(mu) or mu <= 0. or it == maxiter:
            break
        if metric < .8*last or ktr > 1.e3:
            last, stuck = metric, 0
        else:
            stuck += 1
        if stuck > 16 and it >= 18:
            break
        try:
            frozen = K.degree>0 and (plain or (
                      fixed_tau and it>=5 and ktr<8.e-4
                      and max(dual,prim)<8.e-4 and gap<.01))
            if frozen:
                mu = sz/K.degree
                if not mu>0.:
                    break
            con = ConePack(K, kkt, s, z, mu, exp_dual)
            kkt.factor(con)
            if not frozen:
                ux, uz = kkt.solve(-q, b)
                gx = q + (2./tau)*Px if P.nnz else q
                dd = float(gx@ux+b@uz)-(xx/tau+kap)/tau
                if abs(dd) < 1.e-6 * (abs(float(gx@ux))+abs(float(b@uz)) + abs(xx/tau**2)+kap/tau):
                    # Avoid cancellation of the homogeneous Schur complement.
                    h = ux-x/tau
                    hz = A@ux-b
                    dd = -(float(h@(P@h)) + max(0., float(uz@hz)) + kap/tau)
                if not np.isfinite(dd) or dd == 0.:
                    break

            def direction(rc, rct, mult, rgcorr=0.):
                zz = mult*rp-rc
                vx, vz = kkt.solve(mult*rx, zz)
                if frozen:
                    dt,dk = 0.,-kap*mult
                    dx,dz = vx,vz
                    ds = mult*rp - A@dx
                else:
                    dt = (mult*rg - rgcorr - rct/tau - gx@vx - b@vz)/dd
                    dx, dz = vx+dt*ux, vz+dt*uz
                    dk = (rct-kap*dt)/tau
                    ds = mult*rp + b*dt - A@dx
                ds[K.eq] = 0.
                return dx, ds, dz, dt, dk

            def step(d, top=1.):
                _, ds, dz, dt, dk = d
                if dt < 0.:
                    top = min(top, -tau/dt)
                if dk < 0.:
                    top = min(top, -kap/dk)
                return con.step(ds, dz, top)

            affine = direction(-s, -tau*kap, 1.)
            aff_step = step(affine)
            sigma = max(1.e-6, min(.975, (1.-aff_step)**3))
            weight = 1. if aff_step > .6 else max(.1, aff_step/.6)
            if K.exp:
                weight = min(weight, max(.05, aff_step))
            previous = None
            for retry in range(3):
                dx, ds, dz, dt, dk = affine
                target = sigma*mu
                ct = -tau*kap+target-weight*dt*dk
                rc = con.correct(target, ds, dz, weight)
                if P.nnz and weight:
                    xc = dx-x*(dt/tau)
                    corr = weight*float(xc@(P@xc))/tau
                    if not np.isfinite(corr) or abs(corr) > 5.*(abs(rg)+mu):
                        corr = 0.
                else:
                    corr = 0.
                candidate = direction(rc, ct, 1.-sigma, corr)
                alpha = step(candidate)
                if previous is None or alpha*(1.-sigma) > previous[0]*(1.-previous[2]):
                    previous = alpha, candidate, sigma
                if alpha >= .07 or (alpha >= .008 and retry != 0):
                    break
                if retry == 0:
                    weight = 0.
                    sigma = max(sigma, .15)
                else:
                    sigma = max(sigma, .65)
            alpha, d, sig_used = previous
            if (extras and not K.nonlinear and P.nnz and K.lpos.size>=70 and
                    .07 < alpha < .8 and it>=2):
                # One centrality correction with zero residual RHS. This
                # reuses the current factor and targets the pairs blocking
                # a longer step.
                ix=K.lpos
                at=min(.98,alpha*1.25+.085)
                xs=s[ix]+at*d[1][ix]
                yz=z[ix]+at*d[2][ix]
                mu_at=mu*max(.08,1.-at*(1.-sig_used))
                prod=xs*yz
                prod=np.where((xs<0.) & (yz<0.),-np.abs(prod),prod)
                rc0=np.zeros(m)
                diff=np.clip(prod,.08*mu_at,8.*mu_at)-prod
                rc0[ix]=np.clip(diff,-mu_at,mu_at)/(at*z[ix])
                a0=direction(rc0,0.,0.)
                step_new=tuple(a+b for a,b in zip(d,a0))
                longer=step(step_new)
                if longer>alpha*1.06+.016:
                    d,alpha=step_new,longer
            fraction = .99 if K.exp else .995
            alpha = min(1., fraction*alpha)
            if not np.isfinite(alpha) or alpha < 5.e-10:
                break
            dx, ds, dz, dt, dk = d
            x += alpha*dx
            s += alpha*ds
            z += alpha*dz
            tau += alpha*dt
            kap += alpha*dk
            nrm = max(maxabs(x), maxabs(s), maxabs(z), tau, kap)
            if nrm > 1.e7 or 0. < nrm < 1.e-7:
                x /= nrm
                s /= nrm
                z /= nrm
                tau /= nrm
                kap /= nrm
            step_previous = alpha
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError, OverflowError, MemoryError):
            if verbose:
                import traceback
                traceback.print_exc()
            break
    if tau > 0. and kap/tau > 10.:
        answer = p.primal_cert(yc*z)
        if answer is None:
            answer = p.dual_cert(D*x, s/E)
        if answer is not None:
            return answer, point
    if point is not None:
        x, s, z = point
        answer = p.optimal(x,z,s,monitor=2 if verbose else False)
        if answer is not None:
            return answer, point
    return None, point
