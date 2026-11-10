"""Barrier Newton method on a normalized homogeneous self-dual embedding.

The extra embedding coordinate makes each positive barrier subproblem well
defined, including for infeasible original problems.
"""
import math
import os
import sys

import numpy as np
from scipy import sparse

from cones import Layout, exp_barrier
from ipm import Factor, accepted, certificate, equilibrate, norm, polish


def barrier(layout, z):
    g = np.zeros(len(z))
    vals = []
    nn = layout.nn
    g[nn] = -1.0 / z[nn]
    vals.append(1.0 / z[nn] ** 2)
    for ids in layout.soc.values():
        zz = z[ids].astype(np.longdouble)
        jz = zz.copy()
        jz[:, 1:] *= -1
        nr = np.sqrt(np.sum(zz[:, 1:]**2, axis=1))
        det = (zz[:, 0]-nr)*(zz[:, 0]+nr)
        g[ids] = -2.0 * jz / det[:, None]
        hh = 4.0 * jz[:, :, None] * jz[:, None, :] / det[:, None, None]**2
        for i in range(ids.shape[1]):
            hh[:, i, i] += (-2.0 if i == 0 else 2.0) / det
        vals.append(hh.ravel())
    if len(layout.ex):
        gg, hh = exp_barrier(z[layout.ex])
        g[layout.ex] = gg
        vals.append(hh.ravel())
    h = sparse.csc_matrix((np.asarray(np.concatenate(vals), dtype=np.float64),
                            (layout.hrows, layout.hcols)),
                           shape=(len(z), len(z)))
    return g, h


def solve_barrier(P0, q0, A0, b0, cones):
    n, m = len(q0), len(b0)
    layout = Layout(cones, m)
    P, q, A, b, d, e, cost = equilibrate(P0, q0, A0, b0, layout)
    x = np.zeros(n)
    s, z = layout.initial()
    tau = kappa = theta = 1.0
    degree = layout.degree + 1
    rd0 = -A.T @ z - q
    rp0 = s - b
    rg0 = b @ z + 1.0
    debug = os.environ.get("SOLVER_DEBUG")
    target = 1.0
    count = 0
    wd = 1.0 / (1.0 + np.abs(q) + np.abs(rd0))
    wp = 1.0 / (1.0 + np.abs(b) + np.abs(rp0))

    def residual(x, s, z, tau, kappa, theta, target, g=None):
        if g is None:
            g, _ = barrier(layout, z)
        px = P @ x
        rd = -px - A.T @ z - q*tau - rd0*theta
        rp = A @ x + s - b*tau - rp0*theta
        rg = q @ x + b @ z + x @ px / tau + kappa - rg0*theta
        rn = rd0 @ x + rp0 @ z + rg0*tau - degree
        rc = s + target*g
        rc[layout.eq] = 0.0
        rt = tau*kappa - target
        return rd, rp, rg, rn, rc, rt, px

    for outer in range(23):
        target *= 0.2
        for inner in range(18):
            count += 1
            xx, ss, zz = d*x/tau, s/e/tau, e*z/cost/tau
            if accepted(P0, q0, A0, b0, layout, xx, ss, zz):
                return {"status": "solved", "x": xx, "s": ss, "z": zz}
            if tau < 0.1*max(1.0, kappa):
                cert = certificate(P0, q0, A0, b0, layout, d*x, s/e, e*z/cost)
                if cert:
                    return cert
            try:
                g, h = barrier(layout, z)
                rd, rp, rg, rn, rc, rt, px = residual(
                    x, s, z, tau, kappa, theta, target, g)
                wc = 1.0 / (1.0 + np.abs(s) + target*np.abs(g))
                wg = 1.0 / (1.0 + abs(b@z) + abs(q@x) + abs(rg0*theta))
                wt = 1.0 / max(target, tau*kappa)

                def merit(rr):
                    a, bb, c, dd, cc, tt, pp = rr
                    return (np.sum((a*wd)**2) + np.sum((bb*wp)**2)
                            + (c*wg)**2 + (dd/degree)**2
                            + np.sum((cc*wc)**2) + (tt*wt)**2)

                old_merit = merit((rd,rp,rg,rn,rc,rt,px))
                cent = max(norm(rc*wc), abs(rt)*wt)
                if debug:
                    print("B", outer, inner, count,
                          "target %.1e theta %.1e tau %.1e cent %.1e merit %.1e" %
                          (target,theta,tau,cent,old_merit), file=sys.stderr, flush=True)
                if inner > 0 and cent < 0.01 and old_merit < 1e-3:
                    break
                fac = Factor(P, A, reg=1e-12, H=target*h)
                u = fac.solve(np.r_[rd, -rp+rc])
                v = fac.solve(np.r_[-q, b])
                w = fac.solve(np.r_[-rd0, rp0])
                ux, uz = u[:n], u[n:]
                vx, vz = v[:n], v[n:]
                wx, wz = w[:n], w[n:]
                dq = q + 2*px/tau
                xpx = x@px
                mat = np.array([
                    [dq@vx+b@vz-xpx/tau**2-kappa/tau, dq@wx+b@wz-rg0],
                    [rd0@vx+rp0@vz+rg0, rd0@wx+rp0@wz]])
                rhs = np.array([-rg-dq@ux-b@uz+rt/tau,
                                -rn-rd0@ux-rp0@uz])
                dt, dth = np.linalg.solve(mat, rhs)
                dx = ux+vx*dt+wx*dth
                dz = uz+vz*dt+wz*dth
                ds = -rp - A@dx + b*dt + rp0*dth
                ds[layout.eq] = 0.0
                dk = (-rt-kappa*dt)/tau
                alpha = layout.step(z, dz, True)
                for a, da in ((tau,dt),(kappa,dk),(theta,dth)):
                    if da < 0:
                        alpha = min(alpha, -a/da)
                alpha = min(1.0, 0.995*alpha)
                for ls in range(45):
                    x1, s1, z1 = x+alpha*dx, s+alpha*ds, z+alpha*dz
                    t1, k1, th1 = tau+alpha*dt,kappa+alpha*dk,theta+alpha*dth
                    rr = residual(x1,s1,z1,t1,k1,th1,target)
                    new_merit = merit(rr)
                    if np.isfinite(new_merit) and new_merit < old_merit*(1.0-0.005*alpha):
                        break
                    alpha *= 0.5
                if debug:
                    print(" step",alpha, file=sys.stderr)
                if alpha < 1e-13:
                    break
                x, s, z = x1,s1,z1
                tau,kappa,theta = t1,k1,th1
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
                if debug:
                    import traceback
                    traceback.print_exc()
                break
        if theta > max(1e-7, target*100) and outer > 5:
            break
    xx, ss, zz = d*x/tau, s/e/tau, e*z/cost/tau
    result = polish(P0,q0,A0,b0,layout,xx,ss,zz)
    if result:
        return result
    cert = certificate(P0,q0,A0,b0,layout,d*x,s/e,e*z/cost,tol=9e-9)
    if cert:
        return cert
    if accepted(P0,q0,A0,b0,layout,xx,ss,zz,tol=9e-9) or os.environ.get("RETURN_CANDIDATE"):
        return {"status":"solved","x":xx,"s":ss,"z":zz}
    return {"status":"unresolved"}
