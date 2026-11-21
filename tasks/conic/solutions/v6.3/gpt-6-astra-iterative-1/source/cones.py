import os
import numpy as np
from scipy import sparse, linalg

SQRT2 = np.sqrt(2.0)
EXPMODE = os.environ.get("CONIC_EXPMODE", "path")


def exp_transform(z):
    u = z.copy()
    u[:, 0], u[:, 1], u[:, 2] = -z[:, 1], -z[:, 0], np.e * z[:, 2]
    return u


def exp_inside(s):
    u, v, w = s.T
    if np.any(v <= 0) or np.any(w <= 0):
        return False
    return bool(np.all(v * (np.log(w) - np.log(v)) - u > 0))


def exp_barrier(z, third_direction=None):
    """Gradient/Hessian of the dual logarithmically homogeneous barrier."""
    s = exp_transform(z)
    u, v, w = s.T
    logratio = np.log(w) - np.log(v)
    t = v * logratio - u
    gt = np.column_stack((-np.ones_like(v), logratio - 1, v / w))
    ht = np.zeros((len(z), 3, 3))
    ht[:, 1, 1] = -1 / v
    ht[:, 1, 2] = ht[:, 2, 1] = 1 / w
    ht[:, 2, 2] = -v / w**2
    g = -gt / t[:, None]
    g[:, 1] -= 1 / v
    g[:, 2] -= 1 / w
    H = gt[:, :, None] * gt[:, None, :] / t[:, None, None]**2
    H -= ht / t[:, None, None]
    H[:, 1, 1] += 1 / v**2
    H[:, 2, 2] += 1 / w**2
    T = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., np.e]])
    gd = g @ T
    Hd = np.einsum("ij,bjk,kl->bil", T.T, H, T)
    if third_direction is None:
        return gd, Hd
    d = exp_transform(third_direction)
    dv, dw = d[:, 1], d[:, 2]
    htd = np.einsum("bij,bj->bi", ht, d)
    td = np.sum(gt * d, axis=1)
    tdd = np.sum(d * htd, axis=1)
    tddd = np.zeros_like(d)
    tddd[:, 1] = dv**2 / v**2 - dw**2 / w**2
    tddd[:, 2] = -2 * dv * dw / w**2 + 2 * v * dw**2 / w**3
    third = -tddd / t[:, None]
    third += (2 * htd * td[:, None] + gt * tdd[:, None]) / t[:, None]**2
    third -= 2 * gt * td[:, None]**2 / t[:, None]**3
    third[:, 1] -= 2 * dv**2 / v**3
    third[:, 2] -= 2 * dw**2 / w**3
    return gd, Hd, third @ T


def exp_factors(z):
    u, v, w = exp_transform(z).T
    lr = np.log(w) - np.log(v)
    t = v * lr - u
    f = np.zeros((len(z), 3, 4))
    f[:, :, 0] = np.column_stack((1 - lr, np.ones_like(v), np.e*v/w)) / t[:, None]
    f[:, :, 1] = np.column_stack((-1/np.sqrt(v), np.zeros_like(v),
                                  -np.e*np.sqrt(v)/w)) / np.sqrt(t)[:, None]
    f[:, 0, 2] = 1/v
    f[:, 2, 3] = np.e/w
    return f


def exp_hsolve(z, r):
    v, w = -z[:, 0], z[:, 2]
    lr = 1 + np.log(w) - np.log(v)
    t = v*lr + z[:, 1]
    if r.ndim == 3:
        v, w, lr, t = (a[:, None] for a in (v, w, lr, t))
    a = -v*(r[:, 0] + (lr-1)*r[:, 1])
    b = w*r[:, 2] - v*r[:, 1]
    plus = .5*(a+b)
    minus = .5*(a-b)*t/(t+2*v)
    y0, y1 = plus+minus, plus-minus
    return np.stack((-v*y0, -v*((lr-1)*y0+y1)+t*t*r[:, 1], w*y1), axis=1)


def exp_third(z, d, e):
    u, v, w = exp_transform(z).T
    d, e = exp_transform(d), exp_transform(e)
    lr = np.log(w) - np.log(v)
    t = v*lr-u
    gt = np.column_stack((-np.ones_like(v), lr-1, v/w))
    hd, he = np.zeros_like(d), np.zeros_like(e)
    hd[:, 1] = -d[:, 1]/v + d[:, 2]/w
    hd[:, 2] = d[:, 1]/w - v*d[:, 2]/w**2
    he[:, 1] = -e[:, 1]/v + e[:, 2]/w
    he[:, 2] = e[:, 1]/w - v*e[:, 2]/w**2
    td, te = np.sum(gt*d, axis=1), np.sum(gt*e, axis=1)
    tde = np.sum(d*he, axis=1)
    third = np.zeros_like(d)
    third[:, 1] = d[:, 1]*e[:, 1]/v**2 - d[:, 2]*e[:, 2]/w**2
    third[:, 2] = (-(d[:, 1]*e[:, 2]+d[:, 2]*e[:, 1])/w**2
                    + 2*v*d[:, 2]*e[:, 2]/w**3)
    third /= -t[:, None]
    third += (hd*te[:, None] + he*td[:, None] + gt*tde[:, None])/t[:, None]**2
    third -= 2*gt*(td*te)[:, None]/t[:, None]**3
    third[:, 1] -= 2*d[:, 1]*e[:, 1]/v**3
    third[:, 2] -= 2*d[:, 2]*e[:, 2]/w**3
    out = np.empty_like(third)
    out[:, 0], out[:, 1], out[:, 2] = -third[:, 1], -third[:, 0], np.e*third[:, 2]
    return out


def exp_centrality(s, z, mu, aggregate=False):
    v, w = -z[:, 0], z[:, 2]
    if np.any(v <= 0) or np.any(w <= 0):
        return np.inf
    lr = 1 + np.log(w) - np.log(v)
    t = v*lr + z[:, 1]
    if np.any(t <= 0):
        return np.inf
    a = 1 - t*s[:, 1]/mu
    b = -v*(s[:, 0]+(lr-1)*s[:, 1])/mu - 1
    c = (w*s[:, 2]-v*s[:, 1])/mu - 1
    value = a*a + .5*((b+c)**2 + (b-c)**2*t/(t+2*v))
    if not np.isfinite(value).all():
        return np.inf
    return float(np.sqrt(np.sum(value) if aggregate else np.max(value, initial=0)))


def soc_det(x):
    tail = np.linalg.norm(x[:, 1:], axis=1)
    return (x[:, 0] - tail) * (x[:, 0] + tail)


def soc_jordan(a, b):
    out = a * b[:, :1] + b * a[:, :1]
    out[:, 0] = np.sum(a * b, axis=1)
    return out


def soc_lsolve(a, b):
    out = np.empty_like(b)
    det = soc_det(a)
    out[:, 0] = (a[:, 0] * b[:, 0] - np.sum(a[:, 1:] * b[:, 1:], axis=1)) / det
    out[:, 1:] = (b[:, 1:] - out[:, :1] * a[:, 1:]) / a[:, :1]
    return out


def soc_rapply(w, beta, x, inverse=False):
    sign = -1 if inverse else 1
    dot = np.sum(w[:, 1:] * x[:, 1:], axis=1)
    out = np.empty_like(x)
    out[:, 0] = w[:, 0] * x[:, 0] + sign * dot
    out[:, 1:] = x[:, 1:] + w[:, 1:] * (
        sign * x[:, 0] + dot / (1 + w[:, 0]))[:, None]
    out *= (1 / np.sqrt(beta) if inverse else np.sqrt(beta))[:, None]
    return out


class PSD:
    def __init__(self, start, dim):
        self.start, self.dim = start, dim
        self.width = dim * (dim + 1) // 2
        self.sl = slice(start, start + self.width)
        self.c, self.r = np.tril_indices(dim)
        self.off = self.r != self.c
        self.diag = np.flatnonzero(~self.off)

    def mat(self, x):
        v = np.array(x, copy=True)
        v[self.off] /= SQRT2
        out = np.zeros((self.dim, self.dim))
        out[self.r, self.c] = v
        out[self.c, self.r] = v
        return out

    def vec(self, x):
        out = x[self.r, self.c].copy()
        out[self.off] *= SQRT2
        return out

    def batch_mat(self, x):
        out = np.zeros((x.shape[1], self.dim, self.dim))
        v = x.T.copy()
        v[:, self.off] /= SQRT2
        out[:, self.r, self.c] = v
        out[:, self.c, self.r] = v
        return out

    def batch_vec(self, x):
        out = x[:, self.r, self.c].copy()
        out[:, self.off] *= SQRT2
        return out.T


class ConeProduct:
    def __init__(self, cones, m):
        self.m = m
        zero, nn, exp, psd, groups = [], [], [], [], {}
        start = 0
        for cone in cones:
            kind, dim = cone["type"], cone["dim"]
            if kind == "psd":
                obj = PSD(start, dim)
                psd.append(obj)
                width = obj.width
            else:
                width = dim
                if kind == "zero":
                    zero.append(np.arange(start, start+dim, dtype=np.int32))
                elif kind == "nonnegative":
                    nn.append(np.arange(start, start+dim, dtype=np.int32))
                elif kind == "soc":
                    groups.setdefault(dim, []).append(start)
                elif kind == "exponential":
                    exp.append(start)
            start += width
        self.zero = ((zero[0] if len(zero) == 1 else np.concatenate(zero))
                     if zero else np.empty(0, dtype=np.int32))
        self.nn = ((nn[0] if len(nn) == 1 else np.concatenate(nn))
                   if nn else np.empty(0, dtype=np.int32))
        self.exp = np.asarray(exp, dtype=np.int32)[:,None]+np.arange(3,dtype=np.int32)
        self.soc = [np.asarray(g, dtype=np.int32)[:,None]+np.arange(d,dtype=np.int32)
                    for d,g in groups.items()]
        self.psd = psd
        self.degree = len(self.nn) + len(exp) * 3 + sum(p.dim for p in psd)
        self.degree += sum(len(g) for g in self.soc)

    def initial(self):
        s, z = np.zeros(self.m), np.zeros(self.m)
        s[self.nn] = z[self.nn] = 1
        for ids in self.soc:
            s[ids[:, 0]] = z[ids[:, 0]] = 1
        if len(self.exp):
            s[self.exp] = [-1., 1., 2.]
            z[self.exp] = [-1., 0., 1.]
        for p in self.psd:
            s[p.start + p.diag] = z[p.start + p.diag] = 1
        return s, z

    def interior_shift(self, vector, dual=False, margin=1.):
        out = vector.copy()
        if not dual:
            out[self.zero] = 0.
        out[self.nn] = np.maximum(out[self.nn], margin)
        for ids in self.soc:
            tail = np.linalg.norm(out[ids[:, 1:]], axis=1)
            out[ids[:, 0]] = np.maximum(out[ids[:, 0]], tail+margin)
        for p in self.psd:
            M = p.mat(out[p.sl])
            eigmin = linalg.eigvalsh(M, subset_by_index=[0, 0], check_finite=False)[0]
            out[p.start+p.diag] += max(0., margin-eigmin)
        return out

    def step(self, s, ds, dual=False):
        alpha = 1.0
        if len(self.nn):
            d = ds[self.nn]
            neg = d < 0
            if np.any(neg):
                alpha = min(alpha, float(np.min(-s[self.nn][neg] / d[neg])))
        for ids in self.soc:
            x, d = s[ids], ds[ids]
            scale = np.maximum(1, np.max(np.abs(x), axis=1))
            x, d = x / scale[:, None], d / scale[:, None]
            a = d[:, 0]**2 - np.sum(d[:, 1:]**2, axis=1)
            b = 2 * (x[:, 0] * d[:, 0] - np.sum(x[:, 1:] * d[:, 1:], axis=1))
            c = soc_det(x)
            disc = np.maximum(0, b*b - 4*a*c)
            root = -.5 * (b + np.copysign(np.sqrt(disc), b))
            with np.errstate(divide="ignore", invalid="ignore"):
                r1, r2 = root / a, c / root
            r1[(r1 <= 0) | ~np.isfinite(r1)] = np.inf
            r2[(r2 <= 0) | ~np.isfinite(r2)] = np.inf
            bound = np.minimum(r1, r2)
            endpoint = x + alpha * d
            bad = (soc_det(endpoint) <= 0) | (endpoint[:, 0] <= 0)
            if np.any(bad):
                alpha = min(alpha, float(np.min(bound[bad])))
            neg = d[:, 0] < 0
            if np.any(neg):
                alpha = min(alpha, float(np.min(-x[neg, 0] / d[neg, 0])))
        for p in self.psd:
            S, dS = p.mat(s[p.sl]), p.mat(ds[p.sl])
            try:
                mineig = linalg.eigvalsh(dS, S, subset_by_index=[0, 0],
                                        check_finite=False)[0]
                if mineig < 0:
                    alpha = min(alpha, -.999999999 / mineig)
            except linalg.LinAlgError:
                alpha = 0
        if len(self.exp):
            x, d = s[self.exp], ds[self.exp]
            if dual:
                x, d = exp_transform(x), exp_transform(d)
            if not exp_inside(x + alpha * d):
                lo, hi = 0., alpha
                for _ in range(50):
                    mid = .5 * (lo + hi)
                    if exp_inside(x + mid * d):
                        lo = mid
                    else:
                        hi = mid
                    if hi-lo <= 1e-7*hi:
                        break
                alpha = lo
        return max(0., min(1., alpha))

    def scaling(self, s, z, mu):
        return Scaling(self, s, z, mu)

    def centrality(self, s, z, mu, tau=1., kappa=1., individual=False):
        value = (tau*kappa/mu - 1)**2
        worst = value
        if len(self.nn):
            v = s[self.nn] * z[self.nn] / mu - 1
            value += np.sum(v*v)
            if individual:
                worst = max(worst, np.max(v*v, initial=0.))
        for ids in self.soc:
            sp, zp = s[ids], z[ids]
            dot = np.sum(sp*zp, axis=1) / mu
            v = 2*dot*dot - 2*dot + 1 - soc_det(sp)*soc_det(zp)/mu**2
            value += np.sum(np.maximum(0, v))
            if individual:
                worst = max(worst, np.max(v, initial=0.))
        if len(self.exp):
            v = exp_centrality(s[self.exp], z[self.exp], mu, True)
            value += v*v
            if individual:
                w = exp_centrality(s[self.exp], z[self.exp], mu)
                worst = max(worst,w*w)
        for p in self.psd:
            Z, S = p.mat(z[p.sl]), p.mat(s[p.sl])
            try:
                L = linalg.cholesky(Z, lower=True, check_finite=False)
            except linalg.LinAlgError:
                return np.inf
            v = (L.T @ S @ L)/mu - np.eye(p.dim)
            value += np.sum(v*v)
            if individual:
                worst = max(worst,np.sum(v*v))
        if individual:
            return float(np.sqrt(value)),float(np.sqrt(worst))
        return float(np.sqrt(value))


class Scaling:
    def __init__(self, cone, s, z, mu):
        self.cone, self.s, self.z, self.mu = cone, s, z, mu
        nn = cone.nn
        self.diag = s[nn] / z[nn]
        self.soc = []
        for ids in cone.soc:
            sp, zp = s[ids], z[ids]
            ds, dz = soc_det(sp), soc_det(zp)
            if np.any(ds <= 0) or np.any(dz <= 0):
                raise ArithmeticError("SOC iterate outside the interior")
            beta = np.sqrt(ds / dz)
            den = np.sqrt(2 + 2 * np.sum(sp * zp, axis=1) / np.sqrt(ds * dz))
            jz = zp.copy()
            jz[:, 1:] *= -1
            w = (sp / np.sqrt(ds)[:, None] + jz / np.sqrt(dz)[:, None]) / den[:, None]
            lam = soc_rapply(w, beta, zp)
            self.soc.append((ids, beta, w, lam))
        if len(cone.exp):
            self.eg, self.eh = exp_barrier(z[cone.exp])
            self.ew = mu * self.eh
            self.ef = np.sqrt(mu) * exp_factors(z[cone.exp])
            if EXPMODE == "bfgs":
                sv, zv = s[cone.exp], z[cone.exp]
                wz = -mu * self.eg
                self.ew += (sv[:, :, None] * sv[:, None, :]
                            / np.sum(sv * zv, axis=1)[:, None, None])
                self.ew -= (wz[:, :, None] * wz[:, None, :]
                            / np.sum(wz * zv, axis=1)[:, None, None])
        self.psd = []
        for p in cone.psd:
            S, Z = p.mat(s[p.sl]), p.mat(z[p.sl])
            ls = linalg.cholesky(S, lower=True, check_finite=False)
            lz = linalg.cholesky(Z, lower=True, check_finite=False)
            u, lam, vt = linalg.svd(lz.T @ ls, check_finite=False)
            R = (ls @ vt.T) / np.sqrt(lam)[None, :]
            Ri = (np.sqrt(lam)[:, None] * vt) @ linalg.solve_triangular(
                ls, np.eye(p.dim), lower=True, check_finite=False)
            D = R @ R.T
            Di = Ri.T @ Ri
            self.psd.append((p, R, Ri, lam, D, Di))

    def rhs(self, target, ds=None, dz=None):
        c, s, z = self.cone, self.s, self.z
        rhs = -s.copy()
        rhs[c.zero] = 0
        rhs[c.nn] += target / z[c.nn]
        if ds is not None:
            rhs[c.nn] -= ds[c.nn] * dz[c.nn] / z[c.nn]
        for ids, beta, w, lam in self.soc:
            zp = z[ids].copy()
            zp[:, 1:] *= -1
            rhs[ids] += target * zp / soc_det(z[ids])[:, None]
            if ds is not None:
                dsbar = soc_rapply(w, beta, ds[ids], True)
                dzbar = soc_rapply(w, beta, dz[ids])
                corr = soc_lsolve(lam, soc_jordan(dsbar, dzbar))
                rhs[ids] -= soc_rapply(w, beta, corr)
        if len(c.exp):
            rhs[c.exp] -= target * self.eg
            if ds is not None and EXPMODE == "third":
                _, _, third = exp_barrier(z[c.exp], dz[c.exp])
                rhs[c.exp] -= .5 * self.mu * third
            if ds is not None and EXPMODE == "pc":
                v = exp_hsolve(z[c.exp], ds[c.exp])
                rhs[c.exp] += .5 * exp_third(z[c.exp], dz[c.exp], v)
        for p, R, Ri, lam, D, Di in self.psd:
            center = np.diag(target / lam - lam)
            if ds is not None:
                dsbar = Ri @ p.mat(ds[p.sl]) @ Ri.T
                dzbar = R.T @ p.mat(dz[p.sl]) @ R
                prod = dsbar @ dzbar
                center -= (prod + prod.T) / (lam[:, None] + lam[None, :])
            rhs[p.sl] = p.vec(R @ center @ R.T)
        return rhs

    def wmul(self, v):
        c = self.cone
        out = np.zeros_like(v)
        out[c.nn] = self.diag * v[c.nn]
        for ids, beta, w, lam in self.soc:
            x = v[ids]
            block = 2 * w * np.sum(w * x, axis=1)[:, None] + x
            block[:, 0] -= 2 * x[:, 0]
            out[ids] = beta[:, None] * block
        if len(c.exp):
            out[c.exp] = np.einsum("bij,bj->bi", self.ew, v[c.exp])
        for p, R, Ri, lam, D, Di in self.psd:
            out[p.sl] = p.vec(D @ p.mat(v[p.sl]) @ D)
        return out

    def hmul(self, v):
        c = self.cone
        out = np.zeros_like(v)
        out[c.nn] = v[c.nn] / self.diag
        for ids, beta, w, lam in self.soc:
            jw = w.copy()
            jw[:, 1:] *= -1
            x = v[ids]
            block = 2 * jw * np.sum(jw * x, axis=1)[:, None] + x
            block[:, 0] -= 2 * x[:, 0]
            out[ids] = block / beta[:, None]
        if len(c.exp):
            if EXPMODE == "bfgs":
                out[c.exp] = np.linalg.solve(self.ew, v[c.exp, None])[..., 0]
            else:
                out[c.exp] = exp_hsolve(self.z[c.exp], v[c.exp])/self.mu
        for p, R, Ri, lam, D, Di in self.psd:
            out[p.sl] = p.vec(Di @ p.mat(v[p.sl]) @ Di)
        return out

    def sparse_w(self, dense_soc=False):
        """Sparse expanded representation of -W; SOC updates use extra rows."""
        c = self.cone
        rows = [c.nn, c.zero]
        cols = [c.nn, c.zero]
        vals = [-self.diag, np.zeros(len(c.zero))]
        extra = c.m
        for ids, beta, w, lam in self.soc:
            count, dim = ids.shape
            if dense_soc:
                blocks = 2*w[:, :, None]*w[:, None, :]
                blocks[:, np.arange(dim), np.arange(dim)] += 1.
                blocks[:, 0, 0] -= 2.
                blocks *= -beta[:, None, None]
                rows.append(np.repeat(ids, dim, axis=1).ravel())
                cols.append(np.tile(ids, (1, dim)).ravel())
                vals.append(blocks.ravel())
                continue
            diag = -np.broadcast_to(beta[:, None], ids.shape).copy()
            diag[:, 0] *= -1
            rows.append(ids.ravel())
            cols.append(ids.ravel())
            vals.append(diag.ravel())
            ex = np.arange(extra, extra + count)
            extra += count
            couplings = np.sqrt(2 * beta)[:, None] * w
            r, t = ids.ravel(), np.repeat(ex, dim)
            rows += [r, t, ex]
            cols += [t, r, ex]
            vals += [couplings.ravel(), couplings.ravel(), np.ones(count)]
        if len(c.exp):
            ids = c.exp
            if EXPMODE != "bfgs":
                ex = np.arange(extra, extra + 2 * len(ids)).reshape(-1, 2)
                extra += 2 * len(ids)
                diag = -np.sum(self.ef[:, :, 2:]**2, axis=2)
                rows.append(ids.ravel())
                cols.append(ids.ravel())
                vals.append(diag.ravel())
                r = np.repeat(ids, 2, axis=1).ravel()
                t = np.tile(ex, (1, 3)).ravel()
                values = self.ef[:, :, :2].ravel()
                rows += [r, t, ex.ravel()]
                cols += [t, r, ex.ravel()]
                vals += [values, values, np.ones(ex.size)]
            else:
                rows.append(np.repeat(ids, 3, axis=1).ravel())
                cols.append(np.tile(ids, (1, 3)).ravel())
                vals.append(-self.ew.ravel())
        for p, R, Ri, lam, D, Di in self.psd:
            basis = p.batch_mat(np.eye(p.width))
            block = p.batch_vec(D @ basis @ D)
            ids = np.arange(p.start, p.start + p.width)
            rows.append(np.repeat(ids, p.width))
            cols.append(np.tile(ids, p.width))
            vals.append(-block.ravel())
        return sparse.csc_matrix((np.concatenate(vals),
                                  (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(extra, extra))
