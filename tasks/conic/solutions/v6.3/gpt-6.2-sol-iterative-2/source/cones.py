"""Cone scaling and boundary steps for the primal-dual iteration."""
import numpy as np
from scipy.linalg import cholesky, eigvalsh, svd

from model import smat, svec, upper


def rowdot(a, b):
    return np.einsum('ij,ij->i', a, b)


def outer(a, b=None):
    return a[:, :, None] * (a if b is None else b)[:, None, :]


def scalar_step(v, dv, top):
    if not v.size:
        return top
    limit = v + top*dv <= 0.
    return min(top, float(np.min(-v[limit]/dv[limit], initial=np.inf))) if np.any(limit) else top


def valid_exp(x, dual):
    if dual:
        a, b, c = -x[:, 1], -x[:, 0], x[:, 2]
    else:
        a, b, c = x.T
    return ((b > 0.) & (c > 0.) &
            (b*(np.log(c)-np.log(b)+(1. if dual else 0.)) > a))


class Soc:
    def __init__(self, s, z, large, rotated=False):
        self.s, self.z = s, z
        self.large = large
        ns = np.sqrt(rowdot(s[:, 1:], s[:, 1:]))
        nz = np.sqrt(rowdot(z[:, 1:], z[:, 1:]))
        ds = (s[:, 0]-ns)*(s[:, 0]+ns)
        dz = (z[:, 0]-nz)*(z[:, 0]+nz)
        if np.any(ds <= 0.) or np.any(dz <= 0.):
            raise ValueError('SOC iterate outside the interior')
        rs, rz = np.sqrt(ds), np.sqrt(dz)
        self.eta2 = eta2 = rs / rz
        self.detlam = rs*rz
        self.eta = np.sqrt(eta2)
        # w is determinant-one. Compute w0 from wv to preserve that identity.
        qr = np.maximum(0., rowdot(s, z)) / self.detlam
        self.wv = wv = ((s[:, 1:] / rs[:, None] - z[:, 1:]/rz[:, None]) /
                        np.sqrt(2.*(1.+qr))[:, None])
        self.w0 = w0 = np.sqrt(1. + rowdot(wv, wv))
        if rotated:
            d = s.shape[1]
            r = np.sqrt(rowdot(wv, wv))
            unit = np.zeros_like(wv)
            unit[:, 0] = 1.
            np.divide(wv, r[:, None], out=unit, where=r[:, None]>0.)
            v = unit.copy()
            v[:, 0] += np.where(v[:, 0]<0., -1., 1.)
            Q = np.eye(d-1) - (2./rowdot(v, v))[:, None, None]*outer(v)
            R = np.zeros((len(r), d, d))
            R[:, :2, 0] = 2.**-.5
            R[:, 0, 1:] = (2.**-.5)*unit
            R[:, 1, 1:] = -(2.**-.5)*unit
            R[:, 2:, 1:] = Q[:, 1:, :]
            det = (w0+r)**2
            dd = np.repeat(eta2[:, None], d, axis=1)
            dd[:, 0] *= det
            dd[:, 1] /= det
            self.rotation, self.diagonal = R, dd
        elif not large:
            w = np.column_stack((w0, wv))
            H = 2.*outer(w)
            diag = np.arange(s.shape[1])
            H[:, diag, diag] += np.r_[-1., np.ones(s.shape[1]-1)]
            self.h = H*eta2[:, None, None]
        else:
            # H = diag(eta2*[1/2,1,...,1]) + u*u^T - v*v^T.
            al = np.sqrt(np.maximum(.5, 2.*w0*w0 - 1.5))
            self.u = self.eta[:, None] * np.column_stack((al, (2.*w0/al)[:, None]*wv))
            self.v = self.eta[:, None] * np.column_stack((np.zeros(w0.shape), (np.sqrt(3.)/al)[:, None]*wv))

    def half(self, x, inv):
        wv, w0 = self.wv, self.w0
        sign = -1. if inv else 1.
        cross = rowdot(x[:, 1:], wv)
        y = np.column_stack((w0*x[:, 0] + sign*cross,
             x[:, 1:] + (sign*x[:, 0] + cross/(1.+w0))[:, None]*wv))
        if inv:
            y /= self.eta[:, None]
        else:
            y *= self.eta[:, None]
        return y

    def correct(self, target, ds, dz, weight):
        lam = self.half(self.z, False)
        p = self.half(ds, True)
        q = self.half(dz, False)
        b0 = target - weight*rowdot(p, q)
        bv = -weight * (p[:, 0, None]*q[:, 1:] + q[:, 0, None]*p[:, 1:])
        t0 = (lam[:, 0]*b0 - rowdot(lam[:, 1:], bv))/self.detlam
        tv = (bv - t0[:, None]*lam[:, 1:])/lam[:, 0, None]
        return self.half(np.column_stack((t0, tv)), False)

    @staticmethod
    def step_one(x, dx, top):
        trial = x + top*dx
        bad = trial[:, 0] <= np.sqrt(rowdot(trial[:, 1:], trial[:, 1:]))
        if not np.any(bad):
            return top
        x, dx = x[bad], dx[bad]
        r = np.sqrt(rowdot(x[:, 1:], x[:, 1:]))
        c = (x[:, 0]-r)*(x[:, 0]+r)
        b = x[:, 0]*dx[:, 0] - rowdot(x[:, 1:], dx[:, 1:])
        r = np.sqrt(rowdot(dx[:, 1:], dx[:, 1:]))
        a = (dx[:, 0]-r)*(dx[:, 0]+r)
        root = np.sqrt(np.maximum(0., b*b-a*c))
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(b <= 0., c/(root-b), (-b-root)/a)
        ratio = np.where((ratio >= 0) & np.isfinite(ratio), ratio, 0.)
        return min(top, float(np.min(ratio)))


class Exp:
    def __init__(self, s, z, mu, dual_only, rotated):
        self.s, self.z = s, z
        a, b, c = z.T
        d = -a
        v = s[:, 1]
        w = s[:, 2]
        if np.any((c <= 0.) | (d <= 0.) | (v <= 0.) | (w <= 0.)):
            raise ValueError('exponential iterate outside the interior')
        L = np.log(c)-np.log(d)
        phi = b+d*(1.+L)
        dp = (np.log(w)-np.log(v)) - s[:, 0]/v
        if np.any((phi <= 0.) | (dp <= 0.)):
            raise ValueError('exponential iterate outside the interior')
        J = np.column_stack((-L, np.ones(d.size), d/c))/phi[:, None]
        g = np.column_stack((-1./d, np.zeros(d.size), 1./c)) + J
        fl = (d/phi)**.5
        G = np.column_stack((fl/d, np.zeros(d.size), fl/c))
        self.g, self.J, self.d, self.c, self.phi = g, J, d, c, phi
        self.dual_only = dual_only
        ea = np.zeros_like(J)
        ec = np.zeros_like(J)
        ea[:, 0], ec[:, 2] = 1./d, 1./c
        root = np.sqrt(mu)
        if dual_only:
            self.setup(root*np.stack([J,G,ea,ec],axis=2),rotated)
            return
        # Two secants with the dual barrier and its conjugate. Solve the
        # conjugate-gradient scalar equation h + log(1+h) = dp.
        h = np.where(dp < 2., dp*.5, dp-np.log1p(dp))
        for _ in range(4):
            h -= (h + np.log1p(h) - dp)/(1.+1./(1.+h))
        a1 = 1./(v*h)
        gd = np.column_stack((-a1, a1*(2.*h-1.+s[:, 0]/v), (1.+h)/(w*h)))
        lv = L + np.log(w)-np.log(v)
        # Positive decomposition of s'z; expm1(lv)-lv is evaluated locally
        # by its Taylor series when lv is small.
        nl = np.where(np.abs(lv)<1.e-3,lv*lv*(.5+lv*(1./6.+lv/24.)),
                      np.expm1(lv)-lv)
        sz = d*v*(dp+nl)+v*phi
        oo = np.isfinite(sz) & (sz>0.)
        if not np.all(oo):
            sz = np.where(oo,sz,rowdot(s,z))
        if np.any(sz <= 0.):
            raise ValueError('nonpositive exponential complementarity')
        # In square-root coordinates the projection removing the first
        # secant is constant, since [J,G,ea,ec]' z = [1,0,-1,1].
        B = np.stack([root*G,(root/np.sqrt(2.))*(J+ea),
                      (root/np.sqrt(6.))*(-J+ea+2.*ec),s/np.sqrt(sz)[:,None]],axis=2)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            uz = gd - (3./sz)[:, None]*z
            us = g - (3./sz)[:, None]*s
            uzus = rowdot(uz, us)
            n = np.cross(z, uz)
            norm2 = rowdot(n, n)
            keep = (uzus > 1.e-11/sz) & (norm2 > 1.e-24)
            if np.any(keep):
                n /= np.sqrt(norm2)[:, None]
                denom = rowdot(n, self.inv_hess(n))
                t = np.sqrt(sz/np.maximum(rowdot(g, gd), 3.e-99))/denom
                val = np.stack([s/np.sqrt(sz)[:,None],us/np.sqrt(uzus)[:,None],
                                np.sqrt(t)[:,None]*n,np.zeros_like(n)],axis=2)
                keep &= np.isfinite(val).all(axis=(1, 2)) & (denom > 0.)
                B[keep] = val[keep]
        self.setup(B,rotated)

    def setup(self,B,rotated):
        if rotated:
            U,D,_ = np.linalg.svd(B,full_matrices=False)
            self.rotation = U.transpose(0,2,1)
            self.diagonal = D*D
        else:
            self.h = B@B.transpose(0,2,1)

    def inv_hess(self, x):
        J, d, c, phi = self.J, self.d, self.c, self.phi
        r = phi*x[:, 1]
        wa = d*(x[:, 0]-J[:, 0]*r)
        wc = c*(x[:, 2]-J[:, 2]*r)
        u = (d/(phi+2.*d))*(wa+wc)
        a, cc = d*(wa-u), c*(wc-u)
        b = phi*(r-J[:, 0]*a-J[:, 2]*cc)
        return np.column_stack((a, b, cc))

    def correct(self, target, ds, dz, weight):
        if not self.dual_only:
            return target*self.g
        # Mixed second-order correction of the dual-barrier relation.
        y = self.inv_hess(ds)
        q = dz
        d, c, phi, J = self.d, self.c, self.phi, self.J
        a1, c1 = y[:, 0]/d, y[:, 2]/c
        a2, c2 = q[:, 0]/d, q[:, 2]/c
        j1, j2 = rowdot(J, y), rowdot(J, q)
        s1, s2 = a1+c1, a2+c2
        f = -2.*j1*j2-(d/phi)*s1*s2
        g = -(s1*j2+s2*j1)/phi
        out = f[:, None]*J
        out[:, 0] += (g + (-s1*s2 + a1*s2 + a2*s1)/phi + 2.*a1*a2/d)
        out[:, 2] += ((d/c)*(g - (c1*s2+c2*s1)/phi) - 2.*c1*c2/c)
        return target*self.g + .5*weight*out

    def step(self, ds, dz, top):
        ep, ed = self.s, self.z
        top = scalar_step(ep[:, 1:], ds[:, 1:], top)
        top = scalar_step(np.column_stack((-ed[:, 0], ed[:, 2])),
                          np.column_stack((-dz[:, 0], dz[:, 2])), top)
        kp = ~valid_exp(ep + top*ds, False)
        kd = ~valid_exp(ed + top*dz, True)
        if not np.any(kp) and not np.any(kd):
            return top
        ep,ds,ed,dz = ep[kp],ds[kp],ed[kd],dz[kd]
        lo, hi = 0., top
        for _ in range(21):
            mid = 0.5*(lo+hi)
            if np.all(valid_exp(ep + mid*ds, False)) and np.all(valid_exp(ed + mid*dz, True)):
                lo = mid
            else:
                hi = mid
        return lo


class Psd:
    def __init__(self, s, z, d, condensed):
        Ls = cholesky(smat(s, d), lower=True, check_finite=False)
        Lz = cholesky(smat(z, d), lower=True, check_finite=False)
        U, lam, Vt = svd(Lz.T @ Ls, full_matrices=False, check_finite=False)
        if not np.all(lam > 0.):
            raise ValueError('nonpositive SDP complementarity')
        self.R = (Ls @ Vt.T)/np.sqrt(lam)
        self.Ri = (U.T @ Lz.T)/np.sqrt(lam)[:, None]
        self.lam = lam
        self.invlam = 1./np.sqrt(lam)
        self.ij = np.diag_indices(d)
        self.d = d
        self.condensed = condensed
        if condensed:
            self.G = self.Ri.T @ self.Ri
        else:
            W = self.R @ self.R.T
            i, j, w = upper(d)
            v = (W[i[:, None], i] * W[j[:, None], j] + W[i[:, None], j] * W[j[:, None], i])
            self.h = (v*(w[:, None]*w)/2.).reshape(1, len(i), len(i))

    def scaled_directions(self, ds, dz):
        Ri, R = self.Ri, self.R
        return Ri @ smat(ds, self.d) @ Ri.T, R.T @ smat(dz, self.d) @ R

    def step(self, ds, dz, top):
        p, q = self.scaled_directions(ds, dz)
        dl = self.invlam
        for d in (p, q):
            alpha = eigvalsh(d*dl[:, None]*dl, subset_by_index=(0, 0),
                              check_finite=False)[0]
            if alpha < 0.:
                top = min(top, -1./float(alpha))
        return top

    def correct(self, target, ds, dz, weight):
        if weight != 0.:
            p, q = self.scaled_directions(ds, dz)
            r = p @ q
            r = (-.5*weight)*(r+r.T)
        else:
            r = np.zeros((self.d, self.d))
        r[self.ij] += target
        r *= 2./(self.lam[:, None]+self.lam)
        return svec(self.R @ r @ self.R.T)
