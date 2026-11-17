import numpy as np
from scipy import linalg

from model import smat, svec


def soc_det(v):
    radius = np.linalg.norm(v[..., 1:], axis=-1)
    return (v[..., 0] - radius) * (v[..., 0] + radius)


def soc_q(w, determinant, v):
    result = 2.0 * w * np.sum(w * v, axis=-1)[..., None]
    jv = v.copy()
    jv[..., 1:] *= -1.0
    result -= determinant[..., None] * jv
    return result


def soc_jordan(a, b):
    result = np.empty_like(a)
    result[..., 0] = np.sum(a * b, axis=-1)
    result[..., 1:] = (
        a[..., 0, None] * b[..., 1:] + b[..., 0, None] * a[..., 1:])
    return result


def soc_lsolve(lam, rhs):
    result = np.empty_like(rhs)
    det = soc_det(lam)
    result[..., 0] = (
        lam[..., 0] * rhs[..., 0]
        - np.sum(lam[..., 1:] * rhs[..., 1:], axis=-1)) / det
    result[..., 1:] = (
        rhs[..., 1:] - result[..., 0, None] * lam[..., 1:]
    ) / lam[..., 0, None]
    return result


def soc_step(v, dv):
    a = soc_det(dv)
    b = 2.0 * (
        v[..., 0] * dv[..., 0]
        - np.sum(v[..., 1:] * dv[..., 1:], axis=-1))
    c = soc_det(v)
    discr = np.maximum(b * b - 4.0 * a * c, 0.0)
    root = np.sqrt(discr)
    t = np.full_like(a, np.inf)
    negative_a = a < 0.0
    # This form does not subtract nearly equal numbers when b is negative.
    t[negative_a] = 2 * c[negative_a] / (
        -b[negative_a] + root[negative_a])
    positive_a = (a > 0.0) & (b < 0.0) & (discr > 0.0)
    t[positive_a] = 2 * c[positive_a] / (
        -b[positive_a] + root[positive_a])
    linear = (a == 0.0) & (b < 0.0)
    t[linear] = -c[linear] / b[linear]
    neg_t = dv[..., 0] < 0.0
    t[neg_t] = np.minimum(t[neg_t], -v[..., 0][neg_t] / dv[..., 0][neg_t])
    return min(1.0, float(np.min(t, initial=1.0)))


def exp_barrier(s):
    u, v, w = s.T
    L = np.log(w) - np.log(v)
    r = v * L - u
    rg = np.stack((-np.ones_like(v), L - 1.0, v / w), axis=1)
    grad = -rg / r[:, None]
    grad[:, 1] -= 1.0 / v
    grad[:, 2] -= 1.0 / w
    H = rg[:, :, None] * rg[:, None, :] / r[:, None, None] ** 2
    H[:, 1, 1] += 1.0 / v ** 2 + 1.0 / (v * r)
    H[:, 1, 2] -= 1.0 / (w * r)
    H[:, 2, 1] -= 1.0 / (w * r)
    H[:, 2, 2] += 1.0 / w ** 2 + v / (w * w * r)
    return grad, H


def exp_gradient(s):
    u, v, w = s.T
    L = np.log(w) - np.log(v)
    r = v * L - u
    rg = np.stack((-np.ones_like(v), L - 1.0, v / w), axis=1)
    grad = -rg / r[:, None]
    grad[:, 1] -= 1.0 / v
    grad[:, 2] -= 1.0 / w
    return grad


def exp_inside(v, dual=False):
    if v.size == 0:
        return True
    if dual:
        a, b, c = -v[:, 1], -v[:, 0], np.e * v[:, 2]
    else:
        a, b, c = v.T
    if np.any(b <= 0.0) or np.any(c <= 0.0):
        return False
    r = b * (np.log(c) - np.log(b)) - a
    return bool(np.all(r > 0.0) and np.isfinite(r).all())


def exp_shadow(z):
    r = -1.0 / z[:, 0]
    C = np.maximum(1.0 + np.log(r * z[:, 2]) + r * z[:, 1], 1e-30)
    k = np.where(C < 1.0, C / 2.0, C - np.log1p(C))
    for unused in range(6):
        k -= (k + np.log1p(k) - C) * (1.0 + k) / (2.0 + k)
    v = r / k
    w = (1.0 + 1.0 / k) / z[:, 2]
    u = v * (np.log(w) - np.log(v)) - r
    return np.stack((u, v, w), axis=1)


def exp_inverse_barrier(s):
    u, v, w = s.T
    L = np.log(w) - np.log(v)
    r = v * L - u
    h = v / r
    diagonal = (1.0 + h) / (1.0 + 2.0 * h)
    off = h / (1.0 + 2.0 * h)
    vv, vw, ww = diagonal * v * v, off * v * w, diagonal * w * w
    a, b = L - 1.0, v / w
    G = np.zeros((s.shape[0], 3, 3), dtype=s.dtype)
    G[:, 1, 1], G[:, 1, 2], G[:, 2, 1], G[:, 2, 2] = vv, vw, vw, ww
    G[:, 0, 1] = a * vv + b * vw
    G[:, 0, 2] = a * vw + b * ww
    G[:, 1, 0], G[:, 2, 0] = G[:, 0, 1], G[:, 0, 2]
    G[:, 0, 0] = r * r + a * G[:, 0, 1] + b * G[:, 0, 2]
    return G


class ConeSystem:
    def __init__(self, problem):
        self.m = problem.m
        self.zero, self.linear, self.exp = [], [], []
        self.soc, self.psd = {}, []
        self.nu = 0
        start = 0
        for cone in problem.cones:
            kind, dim = cone["type"], cone["dim"]
            width = dim * (dim + 1) // 2 if kind == "psd" else dim
            ids = np.arange(start, start + width)
            if kind == "zero":
                self.zero.append(ids)
            elif kind == "nonnegative":
                self.linear.append(ids)
                self.nu += dim
            elif kind == "soc":
                self.soc.setdefault(dim, []).append(ids)
                self.nu += 1
            elif kind == "exponential":
                self.exp.append(ids)
                self.nu += 3
            else:
                cols, rows = np.tril_indices(dim)
                block = problem.A[ids, :].tocsc()
                active = np.unique(block.indices)
                variables = np.flatnonzero(np.diff(block.indptr))
                Aa = block[active, :][:, variables].tocsc()
                weight = np.where(rows[active] == cols[active],
                                  1.0 / np.sqrt(2.0), 1.0)
                self.psd.append(dict(
                    ids=ids, dim=dim, rows=rows, cols=cols,
                    active=active, variables=variables, Aa=Aa,
                    weight=weight))
                self.nu += dim
            start += width
        self.zero = np.concatenate(self.zero) if self.zero else np.empty(0, dtype=np.int64)
        self.linear = np.concatenate(self.linear) if self.linear else np.empty(0, dtype=np.int64)
        self.exp = np.asarray(self.exp, dtype=np.int64).reshape((-1, 3))
        self.soc = {dim: dict(ids=np.asarray(ids, dtype=np.int64))
                    for dim, ids in self.soc.items()}
        self.nonpsd = np.ones(self.m, dtype=bool)
        for block in self.psd:
            self.nonpsd[block["ids"]] = False
        self.other = np.flatnonzero(self.nonpsd)
        self.inverse_index = np.full(self.m, -1, dtype=np.int64)
        self.inverse_index[self.other] = np.arange(self.other.size)

    def initial(self):
        s, z = np.zeros(self.m), np.zeros(self.m)
        s[self.linear], z[self.linear] = 1.0, 1.0
        for block in self.soc.values():
            ids = block["ids"]
            s[ids[:, 0]], z[ids[:, 0]] = 1.0, 1.0
        if self.exp.size:
            s[self.exp] = np.array([-1.0, 1.0, 1.0])
            z[self.exp] = np.array([-1.0, 0.0, 2.0])
        for block in self.psd:
            diag = block["ids"][block["rows"] == block["cols"]]
            s[diag], z[diag] = 1.0, 1.0
        return s, z

    def scaling(self, s, z):
        self.linear_inverse = s[self.linear] / z[self.linear]
        for dim, block in self.soc.items():
            ids = block["ids"]
            sb, zb = s[ids].astype(np.longdouble), z[ids].astype(np.longdouble)
            for v in (sb, zb):
                radius = np.linalg.norm(v[:, 1:], axis=1)
                v[:, 0] = np.maximum(v[:, 0], radius + 1e-19 * np.maximum(radius, 1e-30))
            sd, zd = np.sqrt(soc_det(sb)), np.sqrt(soc_det(zb))
            ss, zz = sb / sd[:, None], zb / zd[:, None]
            w = ss + zz * np.r_[1.0, -np.ones(dim - 1)]
            w /= np.sqrt(2.0 + 2.0 * np.sum(ss * zz, axis=1))[:, None]
            detw = sd / zd
            w *= np.sqrt(detw)[:, None]
            radius = np.linalg.norm(w[:, 1:], axis=1)
            plus = np.sqrt(w[:, 0] + radius)
            minus = np.sqrt(detw / (w[:, 0] + radius))
            root = w.copy()
            root[:, 0] = 0.5 * (plus + minus)
            root[:, 1:] /= (plus + minus)[:, None]
            detroot = np.sqrt(detw)
            invroot = root * np.r_[1.0, -np.ones(dim - 1)]
            invroot /= detroot[:, None]
            invdetroot = 1.0 / detroot
            lam = 0.5 * (soc_q(root, detroot, zb)
                         + soc_q(invroot, invdetroot, sb))
            block.update(w=w, det=detw, root=root, detroot=detroot,
                         invroot=invroot, invdetroot=invdetroot, lam=lam)
            if dim > 24:
                unit = np.zeros_like(w)
                unit[:, 0] = 1.0
                np.divide(w[:, 1:], radius[:, None], out=unit[:, 1:],
                          where=radius[:, None] > 0)
                uplus = np.sqrt(radius * (w[:, 0] + radius))[:, None] * unit
                unit[:, 1:] *= -1
                uminus = np.sqrt(radius * detw / (w[:, 0] + radius))[:, None] * unit
                uminus *= 1.0 - 8e-15
                block.update(uplus=uplus, uminus=uminus)
        if self.exp.size:
            sb = np.asarray(s[self.exp], dtype=np.longdouble)
            zb = np.asarray(z[self.exp], dtype=np.longdouble)
            r = sb[:, 1] * (np.log(sb[:, 2]) - np.log(sb[:, 1])) - sb[:, 0]
            wide = (self.exp.shape[0] >= 512 and np.min(sb[:, 1:]) > 1e-30
                    and max(np.max(np.abs(sb)), np.max(np.abs(zb))) < 1e30
                    and np.min(r) > 1e-12 * max(1.0, np.max(np.abs(sb))))
            shadow = exp_shadow(zb)
            if wide:
                sb, zb, shadow = sb.astype(float), zb.astype(float), shadow.astype(float)
            grad = exp_gradient(sb)
            gap = np.sum(sb * zb, axis=1)
            mu = gap / 3.0
            G = exp_inverse_barrier(sb) / mu[:, None, None]
            B = np.eye(3)[None, :, :] - sb[:, :, None] * zb[:, None, :] / gap[:, None, None]
            G = B @ G @ B.transpose(0, 2, 1)
            G += sb[:, :, None] * sb[:, None, :] / gap[:, None, None]
            delta_s = sb - mu[:, None] * shadow
            delta_z = zb + mu[:, None] * grad
            delta_gap = np.sum(delta_s * delta_z, axis=1)
            mask = delta_gap > 1e-10 * gap
            B = (np.eye(3)[None, :, :]
                 - delta_s[mask, :, None] * delta_z[mask, None, :]
                 / delta_gap[mask, None, None])
            G[mask] = B @ G[mask] @ B.transpose(0, 2, 1)
            G[mask] += (delta_s[mask, :, None] * delta_s[mask, None, :]
                       / delta_gap[mask, None, None])
            self.exp_gradient = grad
            self.exp_inverse = np.asarray(
                0.5 * (G + G.transpose(0, 2, 1)) if wide else G, dtype=float)
        for block in self.psd:
            ids, dim = block["ids"], block["dim"]
            rows, cols = block["rows"], block["cols"]
            S, Z = smat(s[ids], dim, rows, cols), smat(z[ids], dim, rows, cols)
            L = linalg.cholesky(S, lower=True, check_finite=False)
            Lz = linalg.cholesky(Z, lower=True, check_finite=False)
            eigen, V = linalg.eigh(L.T @ Z @ L, check_finite=False)
            if np.min(eigen) <= 0.0:
                raise np.linalg.LinAlgError("Nonpositive PSD scaling")
            lam = np.sqrt(eigen)
            R = (L @ V) / np.sqrt(lam)[None, :]
            invL = linalg.solve_triangular(L, np.eye(dim), lower=True,
                                          check_finite=False)
            Ri = np.sqrt(lam)[:, None] * (V.T @ invL)
            W = Ri.T @ Ri
            active = block["active"]
            i, j = rows[active], cols[active]
            Haa = (W[i[:, None], i[None, :]] * W[j[:, None], j[None, :]]
                   + W[i[:, None], j[None, :]] * W[j[:, None], i[None, :]])
            Haa *= block["weight"][:, None] * block["weight"][None, :]
            Aa = block["Aa"]
            normal = Aa.T @ (Aa.T @ Haa.T).T
            block.update(S=S, Z=Z, L=L, Lz=Lz, R=R, Ri=Ri,
                         W=W, lam=lam, normal=np.asarray(normal))

    def inverse_apply(self, vector):
        result = np.zeros(self.m)
        result[self.linear] = self.linear_inverse * vector[self.linear]
        for block in self.soc.values():
            ids = block["ids"]
            if "uplus" in block:
                v = vector[ids]
                plus, minus = block["uplus"], block["uminus"]
                result[ids] = (block["det"][:, None] * v
                               + plus * np.sum(plus * v, axis=1)[:, None]
                               - minus * np.sum(minus * v, axis=1)[:, None])
            else:
                result[ids] = soc_q(block["w"], block["det"], vector[ids])
        if self.exp.size:
            result[self.exp] = np.einsum(
                "bij,bj->bi", self.exp_inverse, vector[self.exp])
        return result

    def psd_apply(self, vector):
        result = np.zeros(self.m)
        for block in self.psd:
            ids = block["ids"]
            X = smat(vector[ids], block["dim"], block["rows"], block["cols"])
            result[ids] = svec(block["W"] @ X @ block["W"],
                               block["rows"], block["cols"])
        return result

    def complementarity_rhs(self, s, z, mu_target=0.0, affine=None):
        result = -z.copy()
        result[self.zero] = 0.0
        correction = np.full(self.linear.size, mu_target)
        if affine is not None:
            ds, dz = affine
            correction -= ds[self.linear] * dz[self.linear]
        result[self.linear] += correction / s[self.linear]
        for block in self.soc.values():
            ids = block["ids"]
            target = np.zeros_like(block["lam"])
            target[:, 0] = mu_target
            if affine is not None:
                dss = soc_q(block["invroot"], block["invdetroot"], ds[ids])
                dzs = soc_q(block["root"], block["detroot"], dz[ids])
                target -= soc_jordan(dss, dzs)
            scaled = soc_lsolve(block["lam"], target)
            result[ids] += soc_q(block["invroot"], block["invdetroot"], scaled)
        if self.exp.size:
            result[self.exp] -= mu_target * self.exp_gradient
        for block in self.psd:
            dim, ids, Ri, R = (
                block["dim"], block["ids"], block["Ri"], block["R"])
            target = mu_target * np.eye(dim)
            if affine is not None:
                DSS = Ri @ smat(ds[ids], dim, block["rows"], block["cols"]) @ Ri.T
                DZS = R.T @ smat(dz[ids], dim, block["rows"], block["cols"]) @ R
                product = DSS @ DZS
                target -= 0.5 * (product + product.T)
            lam = block["lam"]
            scaled = target * 2.0 / (lam[:, None] + lam[None, :])
            result[ids] += svec(Ri.T @ scaled @ Ri,
                                block["rows"], block["cols"])
        return result

    def step(self, s, z, ds, dz, tau, kappa, dtau, dkappa):
        alpha = 1.0
        for v, dv in ((s[self.linear], ds[self.linear]),
                      (z[self.linear], dz[self.linear])):
            negative = dv < 0.0
            if np.any(negative):
                alpha = min(alpha, float(np.min(-v[negative] / dv[negative])))
        if dtau < 0.0:
            alpha = min(alpha, -tau / dtau)
        if dkappa < 0.0:
            alpha = min(alpha, -kappa / dkappa)
        for block in self.soc.values():
            ids = block["ids"]
            alpha = min(alpha, soc_step(s[ids], ds[ids]),
                        soc_step(z[ids], dz[ids]))
        for block in self.psd:
            ids, dim = block["ids"], block["dim"]
            for L, dv in ((block["L"], ds[ids]), (block["Lz"], dz[ids])):
                X = smat(dv, dim, block["rows"], block["cols"])
                Y = linalg.solve_triangular(L, X, lower=True, check_finite=False)
                Y = linalg.solve_triangular(L, Y.T, lower=True,
                                            check_finite=False).T
                mineig = linalg.eigh(Y, eigvals_only=True, subset_by_index=[0, 0],
                                    check_finite=False)[0]
                if mineig < 0.0:
                    alpha = min(alpha, -1.0 / mineig)
        if self.exp.size:
            while alpha > 1e-14:
                if (exp_inside(s[self.exp] + alpha * ds[self.exp])
                        and exp_inside(z[self.exp] + alpha * dz[self.exp], dual=True)):
                    break
                alpha *= 0.8
        return max(0.0, min(1.0, alpha))
