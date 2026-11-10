"""Homogeneous primal-dual predictor/corrector conic interior-point method."""
import os
import sys
import time
import warnings

import numpy as np
from scipy import linalg, sparse
from scipy.sparse import linalg as sla

from cones import Layout


def norm(x):
    return float(np.max(np.abs(x), initial=0.0))

class NoNative:
    suspect=True


def equilibrate(P, q, A, b, layout, rounds=4):
    P, A, q, b = P.copy(), A.copy(), q.copy(), b.copy()
    d, e = np.ones(len(q)), np.ones(len(b))
    for k in range(rounds):
        ac = np.asarray(abs(A).max(axis=0).toarray()).ravel()
        pc = np.asarray(abs(P).max(axis=0).toarray()).ravel()
        dc = 1.0 / np.sqrt(np.maximum(np.maximum(ac, pc), 1e-4))
        dc = np.clip(dc, 1e-3, 1e3)
        ar = np.asarray(abs(A).max(axis=1).toarray()).ravel()
        er = 1.0 / np.sqrt(np.maximum(ar, 1e-4))
        er = layout.row_scale(np.clip(er, 1e-3, 1e3))
        d *= dc
        e *= er
        A = sparse.diags(er) @ A @ sparse.diags(dc)
        P = sparse.diags(dc) @ P @ sparse.diags(dc)
        q *= dc
        b *= er
    cost = 1.0 / max(1.0, norm(q), norm(P.data))
    return (P * cost).tocsc(), q * cost, A.tocsc(), b, d, e, cost


def accepted(P, q, A, b, layout, x, s, z, tol=8e-9):
    if not (np.isfinite(x).all() and np.isfinite(s).all() and np.isfinite(z).all()):
        return False
    ax, px, atz = A @ x, P @ x, A.T @ z
    if not (np.isfinite(ax).all() and np.isfinite(px).all() and np.isfinite(atz).all()):
        return False
    rp = norm(ax + s - b) / (1.0 + max(norm(ax), norm(s), norm(b)))
    rd = norm(px + q + atz) / (1.0 + max(norm(px), norm(q), norm(atz)))
    p = 0.5 * (x @ px) + q @ x
    dd = -0.5 * (x @ px) - b @ z
    gap = abs(p - dd) / max(1.0, abs(p), abs(dd))
    if not np.isfinite([rp,rd,gap]).all() or max(rp, rd, gap) > tol:
        return False
    return max(layout.violation(s), layout.violation(z, True)) <= tol


def certificate(P, q, A, b, layout, x, s, z, tol=4e-9):
    zn, bn = norm(z), norm(b)
    an, pn = norm(A.data) or 1.0, norm(P.data) or 1.0
    if zn and bn:
        zz = z / zn
        beta = -(b / bn) @ zz
        if beta > 1e-8:
            scaled_A=A.copy()
            scaled_A.data/=an
            if (norm(scaled_A.T@zz)<tol*beta and
                layout.violation(zz, True)<tol*beta):
                return {"status": "primal_infeasible", "z": zz}
    xn, qn = norm(x), norm(q)
    if xn and qn:
        xx = x / xn
        ss = -A @ xx
        if not np.isfinite(ss).all():
            return None
        alpha = -(q / qn) @ xx
        if alpha > 1e-8:
            scaled_P=P.copy()
            scaled_P.data/=pn
            scaled_A=A.copy()
            scaled_A.data/=an
            slack=ss/an
            if (norm(scaled_P@xx)<tol*alpha and
                norm(scaled_A@xx+slack)<tol*alpha and
                layout.violation(slack)<tol*alpha):
                return {"status": "dual_infeasible", "x": xx, "s": ss}
    return None


def polish(P, q, A, b, layout, x, s, z):
    """A small affine projection, accepted only after checking every cone."""
    px = P @ x
    atz = A.T @ z
    residual = -px - q - atz
    scale = np.asarray(abs(A).max(axis=0).toarray()).ravel()
    scale = np.maximum(scale, 1e-8)
    C = sparse.diags(1.0 / scale) @ A.T
    rhs = residual / scale
    try:
        delta = sla.lsmr(C, rhs, atol=1e-13, btol=1e-13,
                         maxiter=min(300, len(q) + 10))[0]
        z1 = z + delta
        for s1 in (s, b - A @ x):
            if accepted(P, q, A, b, layout, x, s1, z1, tol=9e-9):
                return {"status": "solved", "x": x, "s": s1, "z": z1}
        if norm(b):
            C = sparse.vstack([C, sparse.csr_matrix(b / norm(b))], format="csr")
            gap = x @ px + q @ x + b @ z
            rhs = np.r_[rhs, -gap / norm(b)]
            delta = sla.lsmr(C, rhs, atol=1e-13, btol=1e-13,
                             maxiter=min(400, len(q) + 20))[0]
            z1 = z + delta
            for s1 in (s, b - A @ x):
                if accepted(P, q, A, b, layout, x, s1, z1, tol=9e-9):
                    return {"status": "solved", "x": x, "s": s1, "z": z1}
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        pass
    return None


class Factor:
    def __init__(self, M, Ae, reg=1e-10, H=None, U=None, permutation=None,symbolic=None):
        self.n = M.shape[0]
        self.me = Ae.shape[0]
        self.extra = 0
        self.permutation = None
        self.next_permutation = None
        self.next_symbolic = None
        if isinstance(M, np.ndarray):
            if self.me:
                aa = Ae.toarray()
                self.K = np.block([[M, aa.T], [aa, np.zeros((self.me, self.me))]])
                self.kind = "lu"
                K = self.K.copy()
                ii = np.arange(len(K))
                K[ii[:self.n], ii[:self.n]] += reg
                K[ii[self.n:], ii[self.n:]] -= reg
                self.fac = linalg.lu_factor(K, check_finite=False)
            else:
                self.K = M
                self.kind = "chol"
                K = M.copy()
                K.flat[::len(K) + 1] += reg
                self.fac = linalg.cho_factor(K, check_finite=False)
        else:
            if U is not None:
                self.extra = U.shape[1]
                self.K = sparse.bmat([[M,Ae.T,None], [Ae,-H,U],
                    [None,U.T,sparse.eye(self.extra)]], format="csc")
            elif self.me:
                self.K = sparse.bmat([[M, Ae.T], [Ae, None if H is None else -H]],
                                     format="csc")
            else:
                self.K = M
            diag = np.r_[np.full(self.n, reg), np.full(self.me, -reg),
                         np.zeros(self.extra)]
            K = self.K + sparse.diags(diag, format="csc")
            self.factor_K=K
            allow_arrow=(U is None or len(diag)>=40000 or
                         os.environ.get("SMALL_ARROW_NATIVE","0")!="0")
            if (len(diag)>=3000 and allow_arrow and not os.environ.get("NO_NATIVE") and
                (symbolic is None or not symbolic.suspect)):
                try:
                    from ldl import NativeLDL
                    nk=K
                    self.native_reg=diag
                    state=(symbolic if symbolic is not None and symbolic.matches(nk)
                        else NativeLDL(nk,permutation,quad=U is not None))
                    if state.factor(nk,enforce_sign=U is None):
                        self.kind="native"
                        self.fac=state
                        self.next_permutation=state.permutation
                        self.next_symbolic=state
                        return
                except (ValueError,RuntimeError,MemoryError):
                    self.next_symbolic=NoNative()
            if symbolic is not None and symbolic.suspect:
                self.next_symbolic=symbolic
            self.permutation=permutation
            if permutation is not None:
                K=K[permutation][:,permutation].tocsc()
            self.fac = sla.splu(K, permc_spec=("NATURAL" if permutation is not None else
                os.environ.get("KKT_ORDER","COLAMD")), diag_pivot_thresh=0.01,
                options={"SymmetricMode":True})
            self.next_permutation=(np.argsort(self.fac.perm_c) if permutation is None
                                   else permutation)
            self.kind = "sparse"

    def direct(self, rhs):
        if self.kind=="native":
            return self.fac.solve(rhs)
        if self.kind == "sparse":
            if self.permutation is None:
                return self.fac.solve(rhs)
            out=np.empty_like(rhs)
            out[self.permutation]=self.fac.solve(rhs[self.permutation])
            return out
        if self.kind == "chol":
            return linalg.cho_solve(self.fac, rhs, check_finite=False)
        return linalg.lu_solve(self.fac, rhs, check_finite=False)

    def solve(self, rhs):
        if self.extra:
            rhs = np.r_[rhs, np.zeros(self.extra)]
        x = self.direct(rhs)
        last = np.inf
        for k in range(12 if self.extra and self.kind=="native" else 5):
            r = (self.fac.residual(rhs,x,self.native_reg) if self.kind=="native"
                 else rhs - self.K @ x)
            err = norm(r)
            if err < 2e-12 * (1.0 + norm(rhs)) or err > last * 0.95:
                break
            last = err
            x += self.direct(r)
        if self.kind=="native":
            error=norm(self.fac.residual(rhs,x,self.native_reg)/(1.+np.abs(rhs)))
            if not np.isfinite(error) or error>2e-8:
                state=self.fac
                state.suspect=True
                self.next_symbolic=state
                perm=state.permutation
                kk=self.factor_K[perm][:,perm].tocsc()
                self.fac=sla.splu(kk,permc_spec="NATURAL",diag_pivot_thresh=.1)
                self.kind="sparse"
                self.permutation=perm
                return self.solve(rhs[:self.n+self.me])
        return x[:self.n+self.me]


def conic(P0, q0, A0, b0, cones, max_iter=100, verifier=None,warm=None,
          robust=False,unscaled=False):
    start = time.monotonic()
    debug = os.environ.get("SOLVER_DEBUG")
    n, m = len(q0), len(b0)
    layout = Layout(cones, m)
    for block in layout.psd:
        block.robust=robust
    P, q, A, b, d, e, cost = equilibrate(P0, q0, A0, b0, layout)
    layout.prepare_A(A)
    ae = A[layout.eq].tocsc()
    x = np.zeros(n)
    use_rhs=os.environ.get("RHS_INIT", "1" if layout.soc and n>5000 else "")
    s, z = layout.initial(b if use_rhs else None)
    tau = 1.0
    initial_mu=float(os.environ.get("INIT_MU","1.0"))
    z*=initial_mu
    kappa=initial_mu
    if warm is not None:
        x=np.asarray(warm["x"]/d,dtype=np.float64).copy()
        s=e*(b0-A0@warm["x"])
        z=np.asarray(cost*warm["z"]/e,dtype=np.float64).copy()
        target=1e-6
        nn=layout.nn
        both=(s[nn]<np.sqrt(target))&(z[nn]<np.sqrt(target))
        sv=np.maximum(s[nn],target/np.maximum(z[nn],np.sqrt(target)))
        zv=np.maximum(z[nn],target/np.maximum(sv,np.sqrt(target)))
        sv[both]=np.sqrt(target)
        zv[both]=np.sqrt(target)
        s[nn],z[nn]=sv,zv
        s[layout.eq]=0.
        for ids in layout.soc.values():
            ss,zz=s[ids],z[ids]
            sn=np.linalg.norm(ss[:,1:],axis=1)
            zn=np.linalg.norm(zz[:,1:],axis=1)
            ss[:,0]=np.maximum(ss[:,0],sn+np.maximum(1e-9,target/(1+np.max(np.abs(zz),axis=1))))
            zz[:,0]=np.maximum(zz[:,0],zn+np.maximum(1e-9,target/(1+np.max(np.abs(ss),axis=1))))
            s[ids],z[ids]=ss,zz
        kappa=max(1e-12,float(s@z)/max(1,layout.degree))
    degree = layout.degree + 1
    result = {"status": "unresolved"}
    permutations={}
    symbolics={}
    def checked(candidate):
        return candidate if verifier is None else verifier(candidate)
    for iteration in range(max_iter):
        px = P @ x
        ax = A @ x
        atz = A.T @ z
        rp = ax + s - b * tau
        rd = -px - atz - q * tau
        xpx = x @ px
        rg = q @ x + b @ z + xpx / tau + kappa
        mu = (s @ z + tau * kappa) / degree
        if not np.isfinite(mu) or mu <= 0:
            break
        xx = d * x / tau
        ss = s / (e * tau)
        zz = e * z / (cost * tau)
        if debug:
            p0 = 0.5 * (xx @ (P0 @ xx)) + q0 @ xx
            d0 = -0.5 * (xx @ (P0 @ xx)) - b0 @ zz
            print(iteration, "mu %.2e tau %.2e k %.2e" % (mu, tau, kappa),
                  "rp %.2e rd %.2e gap %.2e" % (norm(rp)/tau, norm(rd)/tau,
                      abs(p0-d0)/max(1.,abs(p0),abs(d0))), file=sys.stderr, flush=True)
        if accepted(P0, q0, A0, b0, layout, xx, ss, zz):
            answer=checked({"status": "solved", "x": xx, "s": ss, "z": zz})
            if answer is not None:
                return answer
        if layout.psd and iteration > 7 and iteration % 3 == 0 and (
                max(norm(rp), norm(rd)) / tau < 1e-5):
            polished = polish(P0, q0, A0, b0, layout, xx, ss, zz)
            if polished is not None:
                answer=checked(polished)
                if answer is not None:
                    return answer
        if iteration > 2 and tau < 0.1 * max(1.0, kappa):
            cert = certificate(P0, q0, A0, b0, layout, d*x, s/e, e*z/cost)
            if cert is not None:
                answer=checked(cert)
                if answer is not None:
                    return answer
        try:
            soc_system=os.environ.get("SOC_SYSTEM","auto")
            expanded=bool(layout.soc) and (soc_system=="full" or
                (soc_system=="auto" and mu<1e-7) or max(layout.soc)>64)
            augmented = not layout.psd and (len(layout.ex) or
                (len(layout.soc) and (soc_system in ("full","direct") or
                    (soc_system=="auto" and mu<1e-3))) or os.environ.get("AUGMENTED_IPM"))
            if expanded:
                augmented=True
            layout.update(s, z, mu, inverse=not augmented,expanded=expanded)
            key=2 if expanded else int(bool(augmented))
            perm=permutations.get(key)
            if augmented:
                fac = Factor(P, A, reg=1e-10 if expanded else 1e-8,
                             H=layout.Hb if expanded else layout.H,
                             U=layout.U if expanded else None,permutation=perm,
                             symbolic=symbolics.get(key))
                v = fac.solve(np.r_[-q, b])
                vx, vz = v[:n], v[n:]
            else:
                M = layout.schur(P, A)
                fac = Factor(M, ae,reg=1e-10 if layout.psd else 1e-8,
                             permutation=perm,symbolic=symbolics.get(key))
                hb = layout.apply_hinv(b)
                vrhs = np.r_[-q + A.T @ hb, b[layout.eq]]
                v = fac.solve(vrhs)
                vx, vze = v[:n], v[n:]
                vz = layout.apply_hinv(A @ vx - b)
                vz[layout.eq] = vze
            if fac.next_permutation is not None and iteration>=2:
                permutations[key]=fac.next_permutation
            if fac.next_symbolic is not None:
                symbolics[key]=fac.next_symbolic
            dq = q + 2.0 * px / tau
            denom = dq @ vx + b @ vz - xpx / tau**2 - kappa / tau

            def direction(c, rtk, residual_scale=1.0, rg_correction=0.0):
                rpp=rp*residual_scale
                rdd=rd*residual_scale
                if augmented:
                    u = fac.solve(np.r_[rdd, -rpp-c])
                    ux, uz = u[:n], u[n:]
                else:
                    hr = layout.apply_hinv(c + rpp)
                    rhs = np.r_[rdd - A.T @ hr, -rpp[layout.eq]]
                    u = fac.solve(rhs)
                    ux = u[:n]
                    uz = layout.apply_hinv(c + rpp + A @ ux)
                    uz[layout.eq] = u[n:]
                dt = (-rg*residual_scale-rg_correction - dq @ ux - b @ uz - rtk / tau) / denom
                dx = ux + vx * dt
                dz = uz + vz * dt
                ds = -rpp - A @ dx + b * dt
                ds[layout.eq] = 0.0
                dk = (rtk - kappa * dt) / tau
                return dx, ds, dz, dt, dk

            c = layout.corrector(0.0)
            da = direction(c, -tau * kappa)

            def step(dr, fraction=1.0):
                dx, ds, dz, dt, dk = dr
                a = min(layout.step(s, ds), layout.step(z, dz, True))
                if dt < 0:
                    a = min(a, -tau / dt)
                if dk < 0:
                    a = min(a, -kappa / dk)
                return max(0.0, min(1.0, a * fraction))

            alpha_aff = step(da)
            muaff = ((s + alpha_aff * da[1]) @ (z + alpha_aff * da[2])
                    + (tau + alpha_aff * da[3]) * (kappa + alpha_aff * da[4])) / degree
            sigma = min(1.0, max(0.0, (muaff / mu) ** 3))
            if len(layout.ex):
                sigma = max(0.08, sigma)
            c = layout.corrector(sigma * mu, da[1], da[2])
            residual_scale=1.0 if unscaled or os.environ.get("UNSCALED_RESIDUALS") else 1.0-sigma
            rg_correction=0.0
            if P.nnz:
                pp=da[0]-x*(da[3]/tau)
                rg_correction=pp@(P@pp)/tau
            dr = direction(c, sigma * mu - tau * kappa - da[3] * da[4],
                           residual_scale,rg_correction)
            alpha = step(dr, 0.99)
            if (layout.soc and os.environ.get("SOC_NEIGHBORHOOD","0")!="0"
                and (warm is None or iteration>=5)):
                for backtrack in range(40):
                    sn = s + alpha*dr[1]
                    zn = z + alpha*dr[2]
                    mun = (sn@zn+(tau+alpha*dr[3])*(kappa+alpha*dr[4]))/degree
                    if layout.neighborhood(sn, zn, mun):
                        break
                    alpha *= 0.8
            if debug:
                print("  step %.3e aff %.3e sigma %.3e" %
                      (alpha, alpha_aff, sigma), file=sys.stderr)
            if alpha < 1e-12 or not np.isfinite(alpha):
                break
            x += alpha * dr[0]
            s += alpha * dr[1]
            z += alpha * dr[2]
            tau += alpha * dr[3]
            kappa += alpha * dr[4]
            if max(norm(x), norm(s), norm(z), tau, kappa) > 1e100:
                scale = max(norm(x), norm(s), norm(z), tau, kappa)
                x /= scale
                s /= scale
                z /= scale
                tau /= scale
                kappa /= scale
        except (np.linalg.LinAlgError, ValueError, RuntimeError, FloatingPointError):
            if debug:
                import traceback
                traceback.print_exc()
            break
    cert = certificate(P0, q0, A0, b0, layout, d*x, s/e, e*z/cost, tol=9e-9)
    if cert is not None:
        answer=checked(cert)
        if answer is not None:
            return answer
    xx, ss, zz = d*x/tau, s/e/tau, e*z/cost/tau
    if accepted(P0, q0, A0, b0, layout, xx, ss, zz, tol=9e-9):
        answer=checked({"status": "solved", "x": xx, "s": ss, "z": zz})
        if answer is not None:
            return answer
    polished = polish(P0, q0, A0, b0, layout, xx, ss, zz) if (
        layout.psd or (not layout.soc and n<2000)) else None
    if polished is not None:
        answer=checked(polished)
        if answer is not None:
            return answer
    if os.environ.get("RETURN_CANDIDATE"):
        return {"status": "solved", "x": xx, "s": ss, "z": zz}
    return result
