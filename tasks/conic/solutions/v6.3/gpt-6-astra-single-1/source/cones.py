import numpy as np
from scipy import sparse, linalg
from common import blocks, smat, svec


def exp_derivatives(s, direction=None):
    u, v, w = s.T
    lv = np.log(w / v)
    f = v * lv - u
    j = np.stack((-np.ones_like(v), lv - 1, v / w), axis=1)
    f2 = np.zeros((len(s), 3, 3))
    f2[:, 1, 1] = -1 / v
    f2[:, 1, 2] = f2[:, 2, 1] = 1 / w
    f2[:, 2, 2] = -v / w**2
    g = -j / f[:, None]
    g[:, 1] -= 1 / v
    g[:, 2] -= 1 / w
    H = j[:, :, None] * j[:, None, :] / f[:, None, None]**2 - f2 / f[:, None, None]
    H[:, 1, 1] += 1 / v**2
    H[:, 2, 2] += 1 / w**2
    if direction is None:
        return g, H
    du, dv, dw = direction.T
    jd = np.sum(j * direction, axis=1)
    f2d = np.einsum("bij,bj->bi", f2, direction)
    d2f = np.sum(direction * f2d, axis=1)
    f3dd = np.zeros_like(s)
    f3dd[:, 1] = dv**2 / v**2 - dw**2 / w**2
    f3dd[:, 2] = -2 * dv * dw / w**2 + 2 * v * dw**2 / w**3
    t = (-f3dd / f[:, None] +
         (2 * f2d * jd[:, None] + j * d2f[:, None]) / f[:, None]**2 -
         2 * j * jd[:, None]**2 / f[:, None]**3)
    t[:, 1] -= 2 * dv**2 / v**3
    t[:, 2] -= 2 * dw**2 / w**3
    return g, H, t


def batch_mv(M, v):
    if isinstance(M, tuple):
        w, scale = M
        jv = -v.copy()
        jv[:, 0] *= -1
        return scale[:, None] * (2 * w * np.sum(w * v, axis=1)[:, None] - jv)
    return np.einsum("bij,bj->bi", M, v)


def soc_jordan(x, y):
    a = x[:, :1] * y + y[:, :1] * x
    a[:, 0] = np.sum(x * y, axis=1)
    return a


def soc_jordan_solve(x, y):
    det = (x[:, 0] - np.linalg.norm(x[:, 1:], axis=1)) * (
        x[:, 0] + np.linalg.norm(x[:, 1:], axis=1))
    t = (x[:, 0] * y[:, 0] - np.sum(x[:, 1:] * y[:, 1:], axis=1)) / det
    r = (y - x * t[:, None]) / x[:, :1]
    r[:, 0] = t
    return r


def soc_maxstep(x, dx):
    xx = x[:, 0]**2 - np.sum(x[:, 1:]**2, axis=1)
    xd = x[:, 0] * dx[:, 0] - np.sum(x[:, 1:] * dx[:, 1:], axis=1)
    dd = dx[:, 0]**2 - np.sum(dx[:, 1:]**2, axis=1)
    disc = np.maximum(0.0, xd**2 - xx * dd)
    root = np.sqrt(disc)
    qq = -xd - np.copysign(root, xd)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = qq / dd
        b = xx / qq
        c = -x[:, 0] / dx[:, 0]
    a[(a <= 0) | ~np.isfinite(a) | ((dd >= 0) & (xd >= 0))] = np.inf
    b[(b <= 0) | ~np.isfinite(b) | ((dd >= 0) & (xd >= 0))] = np.inf
    c[(c <= 0) | ~np.isfinite(c)] = np.inf
    return min(1.0, float(np.min(a, initial=np.inf)),
               float(np.min(b, initial=np.inf)), float(np.min(c, initial=np.inf)))


def exp_interior(x, dual=False):
    if dual:
        x = np.stack((-x[:, 1], -x[:, 0], np.e * x[:, 2]), axis=1)
    u, v, w = x.T
    if np.any(v <= 0) or np.any(w <= 0):
        return False
    return bool(np.all(v * np.log(w / v) - u > 0))


class ConeSystem:
    def __init__(self, cones, A):
        self.m, self.n = A.shape
        self.cones = cones
        nn, exp = [], []
        soc = {}
        self.psd = []
        self.nu = 0
        for k, d, i, j in blocks(cones):
            if k == "nonnegative":
                nn.extend(range(i, j))
                self.nu += d
            elif k == "soc":
                soc.setdefault(d, []).append(np.arange(i, j))
                self.nu += 1
            elif k == "exponential":
                exp.append(np.arange(i, j))
                self.nu += 3
            elif k == "psd":
                Ab = A[i:j].tocsc()
                cols = np.flatnonzero(np.diff(Ab.indptr))
                av = Ab[:, cols].toarray().T
                mats = np.array([smat(v, d) for v in av])
                if len(cols) == 0:
                    mats = np.empty((0, d, d))
                rr, cc = np.meshgrid(cols, cols, indexing="ij")
                self.psd.append(dict(d=d, i=i, j=j, cols=cols,
                                     mats=mats, rr=rr.ravel(), cc=cc.ravel()))
                self.nu += d
        self.nn = np.asarray(nn, dtype=int)
        self.exp = np.asarray(exp, dtype=int).reshape(-1, 3)
        self.soc = {d: np.asarray(v) for d, v in soc.items()}
        self.small_soc = {d: ids for d, ids in self.soc.items() if d <= 32}
        self.large_soc = {d: ids for d, ids in self.soc.items() if d > 32}
        rows, cols = [self.nn], [self.nn]
        for ids in list(self.small_soc.values()) + ([self.exp] if len(self.exp) else []):
            d = ids.shape[1]
            rows.append(np.repeat(ids, d, axis=1).ravel())
            cols.append(np.tile(ids, (1, d)).ravel())
        for ids in self.large_soc.values():
            rows.append(ids.ravel())
            cols.append(ids.ravel())
        self.hrows = np.concatenate(rows)
        self.hcols = np.concatenate(cols)

    def initial(self):
        s, z = np.zeros(self.m), np.zeros(self.m)
        s[self.nn] = z[self.nn] = 1.0
        for ids in self.soc.values():
            s[ids[:, 0]] = z[ids[:, 0]] = 1.0
        if len(self.exp):
            s[self.exp] = [-1.0, 1.0, 1.0]
            z[self.exp] = [-1.0, 0.0, 2.0]
        for p in self.psd:
            v = svec(np.eye(p["d"]))
            s[p["i"]:p["j"]] = z[p["i"]:p["j"]] = v
        return s, z

    def scale(self, s, z):
        self.s, self.z = s, z
        data = [z[self.nn] / s[self.nn]]
        invdata = [s[self.nn] / z[self.nn]]
        tdata = [np.sqrt(data[0])]
        tddata = [np.sqrt(invdata[0])]
        self.soc_state = []
        self.large_state = []
        for d, ids in self.soc.items():
            a, b = s[ids], z[ids]
            an = np.linalg.norm(a[:, 1:], axis=1)
            bn = np.linalg.norm(b[:, 1:], axis=1)
            sa = np.sqrt((a[:, 0] - an) * (a[:, 0] + an))
            sb = np.sqrt((b[:, 0] - bn) * (b[:, 0] + bn))
            jb = b.copy()
            jb[:, 1:] *= -1
            w = (a / sa[:, None] + jb / sb[:, None]) / np.sqrt(
                2 * (1 + np.sum(a * b, axis=1) / (sa * sb)))[:, None]
            v = w.copy()
            v[:, 0] += 1
            v /= np.sqrt(2 * (w[:, 0] + 1))[:, None]
            beta = np.sqrt(sa / sb)
            if d <= 32:
                J = -np.eye(d)
                J[0, 0] = 1.0
                W = beta[:, None, None] * (2 * v[:, :, None] * v[:, None, :] - J)
                v[:, 1:] *= -1
                Wi = (2 * v[:, :, None] * v[:, None, :] - J) / beta[:, None, None]
                H = Wi @ Wi
                data.append(H.ravel())
                invdata.append((W @ W).ravel())
                tdata.append(Wi.ravel())
                tddata.append(W.ravel())
            else:
                vi = v.copy()
                vi[:, 1:] *= -1
                W, Wi = (v, beta), (vi, 1 / beta)
                jw = w.copy()
                jw[:, 1:] *= -1
                self.large_state.append((ids, W, Wi, jw, beta))
            lam = batch_mv(W, b)
            self.soc_state.append((ids, W, Wi, lam))
        if len(self.exp):
            a, b = s[self.exp], z[self.exp]
            g, H = exp_derivatives(a)
            sz = np.sum(a * b, axis=1)
            mul = sz / 3
            M = mul[:, None, None] * (H - g[:, :, None] * g[:, None, :] / 3)
            M += b[:, :, None] * b[:, None, :] / sz[:, None, None]
            self.exp_state = (g, H, mul)
            data.append(M.ravel())
            try:
                L = np.linalg.cholesky(M)
            except np.linalg.LinAlgError:
                floor = np.max(np.abs(M), axis=(1, 2)) * 1e-14
                L = np.linalg.cholesky(M + floor[:, None, None] * np.eye(3))
            Li = np.linalg.inv(L)
            invdata.append((Li.transpose(0, 2, 1) @ Li).ravel())
            tdata.append(L.transpose(0, 2, 1).ravel())
            tddata.append(Li.ravel())
        for ids, W, Wi, jw, beta in self.large_state:
            diag = np.ones_like(jw)
            diag[:, 0] = -1
            data.append((diag / beta[:, None]**2).ravel())
            invdata.append((diag * beta[:, None]**2).ravel())
            tdata.append((diag / beta[:, None]).ravel())
            tddata.append((diag * beta[:, None]).ravel())
        self.H = sparse.coo_matrix((np.concatenate(data), (self.hrows, self.hcols)),
                                    shape=(self.m, self.m)).tocsc()
        self.Hinv = sparse.coo_matrix((np.concatenate(invdata), (self.hrows, self.hcols)),
                                       shape=(self.m, self.m)).tocsc()
        self.T = sparse.coo_matrix((np.concatenate(tdata), (self.hrows, self.hcols)),
                                    shape=(self.m, self.m)).tocsc()
        self.Tdual = sparse.coo_matrix((np.concatenate(tddata), (self.hrows, self.hcols)),
                                        shape=(self.m, self.m)).tocsc()
        for p in self.psd:
            i, j, d = p["i"], p["j"], p["d"]
            S, Z = smat(s[i:j], d), smat(z[i:j], d)
            Ls = linalg.cholesky(S, lower=True, check_finite=False)
            Lz = linalg.cholesky(Z, lower=True, check_finite=False)
            U, lam, Vt = linalg.svd(Lz.T @ Ls, check_finite=False)
            R = (Ls @ Vt.T) / np.sqrt(lam)[None, :]
            Ri = linalg.solve_triangular(Ls, np.eye(d), lower=True, check_finite=False)
            Ri = np.sqrt(lam)[:, None] * (Vt @ Ri)
            p.update(R=R, Ri=Ri, lam=lam, Ls=Ls, Lz=Lz)

    def hmul(self, v):
        out = self.H @ v
        for ids, W, Wi, jw, beta in self.large_state:
            out[ids] = batch_mv(Wi, batch_mv(Wi, v[ids]))
        for p in self.psd:
            i, j, Ri = p["i"], p["j"], p["Ri"]
            G = Ri.T @ Ri
            out[i:j] = svec(G @ smat(v[i:j], p["d"]) @ G)
        return out

    def gram(self, A):
        K = (A.T @ self.H @ A).tocsc()
        for ids, W, Wi, jw, beta in self.large_state:
            for ii, v, be in zip(ids, jw, beta):
                h = np.sqrt(2) / be * (A[ii].T @ v)
                nz = np.flatnonzero(h)
                rr, cc = np.meshgrid(nz, nz, indexing="ij")
                K += sparse.coo_matrix(((h[nz, None] * h[None, nz]).ravel(),
                                        (rr.ravel(), cc.ravel())), shape=K.shape).tocsc()
        for p in self.psd:
            Ri, mats = p["Ri"], p["mats"]
            if not len(mats):
                continue
            T = Ri @ mats @ Ri.T
            T = T.reshape(len(mats), -1)
            values = (T @ T.T).ravel()
            K += sparse.coo_matrix((values, (p["rr"], p["cc"])),
                                  shape=(self.n, self.n)).tocsc()
        return K

    def scaled_A(self, A):
        B = (self.T @ A).toarray()
        for ids, W, Wi, jw, beta in self.large_state:
            vv = Wi[0] * np.sqrt(2 / beta)[:, None]
            for ii, v in zip(ids, vv):
                h = A[ii].T @ v
                B[ii] += v[:, None] * h[None, :]
        for p in self.psd:
            T = p["Ri"] @ p["mats"] @ p["Ri"].T
            c, r = np.tril_indices(p["d"])
            V = T[:, r, c]
            V[:, r != c] *= np.sqrt(2.0)
            B[p["i"]:p["j"], p["cols"]] = V.T
        return B

    def scale_primal(self, v):
        out = self.T @ v
        for ids, W, Wi, jw, beta in self.large_state:
            out[ids] = batch_mv(Wi, v[ids])
        for p in self.psd:
            i, j, Ri = p["i"], p["j"], p["Ri"]
            out[i:j] = svec(Ri @ smat(v[i:j], p["d"]) @ Ri.T)
        return out

    def scale_dual(self, v):
        out = self.Tdual @ v
        for ids, W, Wi, jw, beta in self.large_state:
            out[ids] = batch_mv(W, v[ids])
        for p in self.psd:
            i, j, R = p["i"], p["j"], p["R"]
            out[i:j] = svec(R.T @ smat(v[i:j], p["d"]) @ R)
        return out

    def unscale_dual(self, v):
        out = self.T.T @ v
        for ids, W, Wi, jw, beta in self.large_state:
            out[ids] = batch_mv(Wi, v[ids])
        for p in self.psd:
            i, j, Ri = p["i"], p["j"], p["Ri"]
            out[i:j] = svec(Ri.T @ smat(v[i:j], p["d"]) @ Ri)
        return out

    def lowrank_columns(self, A, ne):
        rr, cc, vv = [], [], []
        offset = 0
        for ids, W, Wi, jw, beta in self.large_state:
            vectors = Wi[0] * np.sqrt(2 / beta)[:, None]
            for ii, v in zip(ids, vectors):
                h = A[ii].T @ v
                nz = np.flatnonzero(h)
                rr.extend([nz, self.n + ne + ii])
                cc.extend([np.full(len(nz), offset), np.full(len(ii), offset + 1)])
                vv.extend([h[nz], v])
                offset += 2
        if not offset:
            return None
        return sparse.coo_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                                 shape=(self.n + ne + self.m, offset)).tocsc()

    def complement_rhs(self, target=0.0, affine=None):
        s, z = self.s, self.z
        if affine is None and target == 0:
            return -z.copy()
        ds, dz = affine if affine is not None else (np.zeros(self.m), np.zeros(self.m))
        rhs = -z.copy()
        ids = self.nn
        rhs[ids] += (target - ds[ids] * dz[ids]) / s[ids]
        for ids, W, Wi, lam in self.soc_state:
            cross = soc_jordan(batch_mv(Wi, ds[ids]), batch_mv(W, dz[ids]))
            r = -cross
            r[:, 0] += target
            rhs[ids] += batch_mv(Wi, soc_jordan_solve(lam, r))
        if len(self.exp):
            g, H, mul = self.exp_state
            da = ds[self.exp]
            rhs[self.exp] -= target * g
            if affine is not None:
                _, _, third = exp_derivatives(s[self.exp], da)
                correction = mul[:, None] * (batch_mv(H, da) - 0.5 * third)
                # Limit high-order terms far from the central path.
                size = np.max(np.abs(correction), axis=1)
                base = np.maximum(np.max(np.abs(z[self.exp]), axis=1), 1e-30)
                correction *= np.minimum(1.0, 5 * base / np.maximum(size, 1e-30))[:, None]
                rhs[self.exp] += correction
        for p in self.psd:
            i, j, R, Ri, lam = p["i"], p["j"], p["R"], p["Ri"], p["lam"]
            a = Ri @ smat(ds[i:j], p["d"]) @ Ri.T
            b = R.T @ smat(dz[i:j], p["d"]) @ R
            r = target * np.eye(p["d"]) - (a @ b + b @ a) / 2
            r *= 2 / (lam[:, None] + lam[None, :])
            rhs[i:j] += svec(Ri.T @ r @ Ri)
        return rhs

    def step(self, s, ds, z, dz):
        alpha = 1.0
        ids = self.nn
        for a, da in ((s[ids], ds[ids]), (z[ids], dz[ids])):
            neg = da < 0
            if np.any(neg):
                alpha = min(alpha, float(np.min(-a[neg] / da[neg])))
        for ids in self.soc.values():
            alpha = min(alpha, soc_maxstep(s[ids], ds[ids]),
                        soc_maxstep(z[ids], dz[ids]))
        for p in self.psd:
            i, j = p["i"], p["j"]
            for L, da in ((p["Ls"], ds[i:j]), (p["Lz"], dz[i:j])):
                X = linalg.solve_triangular(L, smat(da, p["d"]),
                                             lower=True, check_finite=False)
                X = linalg.solve_triangular(L, X.T, lower=True, check_finite=False).T
                eigen = linalg.eigh(X, subset_by_index=[0, 0],
                                   eigvals_only=True, check_finite=False)[0]
                if eigen < 0:
                    alpha = min(alpha, -1 / eigen)
        if len(self.exp):
            ids = self.exp
            for _ in range(80):
                if (exp_interior(s[ids] + alpha * ds[ids]) and
                        exp_interior(z[ids] + alpha * dz[ids], True)):
                    break
                alpha *= 0.8
        return max(0.0, min(alpha, 1.0))
