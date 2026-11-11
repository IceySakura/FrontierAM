"""KKT system  [P+dI  A'; A  -(H+dI)]  factorisation.

Sparse mode : SuperLU (MMD on A'+A, diagonal pivots, symmetric mode) on the
              quasi-definite matrix, values updated in place each iteration.
Dense mode  : normal equations M = P + A' H^{-1} A (dense Cholesky), used when
              large PSD blocks would make the sparse KKT dense; zero-cone rows
              are kept as a Schur complement.
Both modes use iterative refinement against the unregularised system."""

import os
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy import linalg as dla


REL_REG = float(os.environ.get("REL_REG", "1e-11"))


class KKTSparse:
    def __init__(self, P, A, cones):
        n, m = P.shape[0], A.shape[0]
        self.n, self.m = n, m
        self.P, self.A, self.AT = P, A, A.T.tocsr()
        self.cones = cones
        Pc = P.tocoo()
        off = Pc.row != Pc.col
        self.Pdiag = P.diagonal().copy()
        Ac = A.tocoo()
        rows = [Pc.row[off], np.arange(n), Ac.col, n + Ac.row]
        cols = [Pc.col[off], np.arange(n), n + Ac.row, Ac.col]
        vals = [Pc.data[off], self.Pdiag.copy(), Ac.data, Ac.data]
        hrows, hcols = [], []
        z = cones.zero_rows
        hrows.append(n + z); hcols.append(n + z)
        if len(cones.nn_rows):
            r = cones.nn_rows
            hrows.append(n + r); hcols.append(n + r)
        for d, R in cones.soc_groups:
            hrows.append((n + R[:, :, None] + 0 * R[:, None, :]).ravel())
            hcols.append((n + R[:, None, :] + 0 * R[:, :, None]).ravel())
        if cones.exp_rows.shape[0]:
            R = cones.exp_rows
            hrows.append((n + R[:, :, None] + 0 * R[:, None, :]).ravel())
            hcols.append((n + R[:, None, :] + 0 * R[:, :, None]).ravel())
        for d, r in cones.psd:
            N = len(r)
            hrows.append(np.repeat(n + r, N)); hcols.append(np.tile(n + r, N))
        hr = np.concatenate(hrows).astype(np.int64)
        hc = np.concatenate(hcols).astype(np.int64)
        self.nstatic = sum(len(v) for v in vals)
        self.ndiagP_start = len(rows[0])
        rows_all = np.concatenate([np.concatenate(rows).astype(np.int64), hr])
        cols_all = np.concatenate([np.concatenate(cols).astype(np.int64), hc])
        self.vals = np.concatenate([np.concatenate(vals), np.zeros(len(hr))])
        N = n + m
        key = cols_all * N + rows_all
        perm = np.argsort(key, kind="stable")
        self.perm = perm
        indices = rows_all[perm]
        counts = np.bincount(cols_all, minlength=N)
        indptr = np.concatenate([[0], np.cumsum(counts)])
        self.K = sp.csc_matrix((self.vals[perm], indices, indptr), shape=(N, N))
        self.hstart = self.nstatic
        self.nH = len(hr)
        self.lu = None
        self.dpt = float(__import__("os").environ.get("DPT", "0.0"))
        self.last_res = 0.0
        self.base_time = None
        self.no_pivot = False

    def factor(self, Hparts, dp, dd):
        v = self.vals
        a = self.ndiagP_start
        v[a:a + self.n] = self.Pdiag + dp
        h = np.concatenate(Hparts) if len(Hparts) else np.zeros(0)
        # add regularisation on the diagonal of H blocks
        hv = -h
        self.Hvals = h
        v[self.hstart:] = hv
        self._add_diag_reg(self._diag_reg_vector(Hparts, dd))
        self.K.data[:] = v[self.perm]
        if __import__("os").environ.get("DUMPK"):
            self._nf = getattr(self, "_nf", 0) + 1
            if self._nf == int(__import__("os").environ["DUMPK"]):
                sp.save_npz("/tmp/Kdump.npz", self.K)
                raise SystemExit(0)
        t0 = time.time()
        self.lu = sla.splu(self.K, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=self.dpt,
                           options=dict(SymmetricMode=True))
        dt = time.time() - t0
        self.last_ft = dt
        self.last_fill = self.lu.L.nnz + self.lu.U.nnz
        if self.dpt == 0.0:
            self.base_time = dt if self.base_time is None else min(self.base_time, dt)
        elif self.base_time is not None and dt > max(0.5, 20.0 * self.base_time):
            # pivoting is too expensive for this problem: never use it again
            self.dpt = 0.0
            self.no_pivot = True

    def _diag_reg_vector(self, Hparts, dd):
        """Static regularisation, proportional to each dense block's scale (emulates the
        dynamic regularisation of LDL solvers for nearly singular SOC/exp blocks)."""
        c = self.cones
        regs = [np.full(len(c.zero_rows), dd)]
        i = 1
        if len(c.nn_rows):
            regs.append(np.full(len(c.nn_rows), dd)); i += 1
        for d, R in c.soc_groups:
            Hb = Hparts[i].reshape(-1, d, d); i += 1
            dm = np.trace(Hb, axis1=1, axis2=2) / d
            regs.append(np.repeat(dd + REL_REG * dm, d))
        if c.exp_rows.shape[0]:
            Hb = Hparts[i].reshape(-1, 3, 3); i += 1
            dm = np.trace(Hb, axis1=1, axis2=2) / 3
            regs.append(np.repeat(dd + REL_REG * dm, 3))
        for d, r in c.psd:
            N = len(r)
            Hb = Hparts[i].reshape(N, N); i += 1
            regs.append(np.full(N, dd + REL_REG * np.trace(Hb) / N))
        return np.concatenate(regs)

    def _add_diag_reg(self, dd):
        # diagonal positions inside the H block list
        if not hasattr(self, "_hdiag"):
            c = self.cones
            pos = []
            off = 0
            nz = len(c.zero_rows)
            pos.append(np.arange(nz)); off += nz
            if len(c.nn_rows):
                k = len(c.nn_rows)
                pos.append(off + np.arange(k)); off += k
            for d, R in c.soc_groups:
                k = R.shape[0]
                base = off + np.arange(k)[:, None] * d * d
                pos.append((base + (np.arange(d) * (d + 1))[None, :]).ravel()); off += k * d * d
            if c.exp_rows.shape[0]:
                k = c.exp_rows.shape[0]
                base = off + np.arange(k)[:, None] * 9
                pos.append((base + np.array([0, 4, 8])[None, :]).ravel()); off += 9 * k
            for d, r in c.psd:
                N = len(r)
                pos.append(off + np.arange(N) * (N + 1)); off += N * N
            self._hdiag = self.hstart + np.concatenate(pos).astype(np.int64)
        self.vals[self._hdiag] -= dd

    def apply_true(self, x, z):
        """[P A'; A -H] [x; z] with the unregularised H."""
        r1 = self.P @ x + self.AT @ z
        r2 = self.A @ x - self.cones.apply_H(z)
        return r1, r2

    def solve(self, r1, r2, refine=int(__import__("os").environ.get("REFINE", "10"))):
        n = self.n
        rhs = np.concatenate([r1, r2])
        nr0 = np.max(np.abs(rhs)) + 1e-300

        def resid(sol):
            a1, a2 = self.apply_true(sol[:n], sol[n:])
            res = rhs - np.concatenate([a1, a2])
            return res, float(np.max(np.abs(res)))

        sol = self.lu.solve(rhs)
        if not np.all(np.isfinite(sol)):
            self.last_res = np.inf
            return np.zeros(n), np.zeros(self.m)
        res, nres = resid(sol)
        for _ in range(refine):
            if nres <= 1e-14 * nr0:
                break
            new = sol + self.lu.solve(res)
            if not np.all(np.isfinite(new)):
                break
            res_n, nres_n = resid(new)
            if not (nres_n < nres):
                break
            improved = nres_n < 0.5 * nres
            sol, res, nres = new, res_n, nres_n
            if not improved:
                break
        self.last_res = nres / nr0
        return sol[:n], sol[n:]


class KKTDense:
    """Normal-equation solver for problems with large PSD blocks (moderate n)."""

    def __init__(self, P, A, cones):
        self.n, self.m = P.shape[0], A.shape[0]
        self.P, self.A, self.AT = P, A, A.T.tocsr()
        self.cones = cones
        self.Pd = P.toarray()
        c = cones
        Acsr = A.tocsr()
        self.A0 = Acsr[c.zero_rows].toarray() if len(c.zero_rows) else None
        self.Ann = Acsr[c.nn_rows] if len(c.nn_rows) else None
        self.Asoc = [Acsr[R.ravel()] for d, R in c.soc_groups]
        self.Aexp = Acsr[c.exp_rows.ravel()] if c.exp_rows.shape[0] else None
        self.use_qr = False
        self.Rqr = None
        self.Apsd = []
        for d, r in c.psd:
            Ab = Acsr[r].tocsc()
            used = np.unique(Ab.indices)
            self.Apsd.append((Ab, used, Ab[used]))

    def qr_applicable(self):
        c = self.cones
        rows = sum(len(r) for d, r in c.psd) + len(c.nn_rows) + self.n
        return (self.P.nnz == 0 and not c.soc_groups and c.exp_rows.shape[0] == 0
                and rows * self.n <= 4e7 and self.n <= 4000)

    def _factor_qr(self, dp):
        c = self.cones
        st = c.st
        n = self.n
        blocks = []
        if self.Ann is not None:
            hi = 1.0 / st["nn_H"]
            blocks.append((sp.diags(np.sqrt(hi)) @ self.Ann).toarray())
        for i, (Ab, used, Au) in enumerate(self.Apsd):
            d = c.psd[i][0]
            ix = c.psd_idx[d]
            Ri = st["psd"][i]["Rinv"]
            Ad = Ab.toarray()
            cols = np.where(np.any(Ad != 0, axis=0))[0]
            B = np.zeros((len(ix.I), n))
            if len(cols):
                Smat = np.zeros((len(cols), d, d))
                vals = Ad[:, cols].T * ix.fi[None, :]
                Smat[:, ix.I, ix.J] = vals
                Smat[:, ix.J, ix.I] = vals
                T = np.matmul(np.matmul(Ri[None], Smat), Ri.T[None])
                B[:, cols] = (0.5 * (T[:, ix.I, ix.J] + T[:, ix.J, ix.I]) * ix.f[None, :]).T
            blocks.append(B)
        blocks.append(np.sqrt(dp) * np.eye(n))
        Bt = np.vstack(blocks)
        R = dla.qr(Bt, mode="r", overwrite_a=True, check_finite=False)[0][:n, :n]
        self.Rqr = R

    def _Msolve(self, rhs):
        if self.Rqr is not None:
            y = dla.solve_triangular(self.Rqr, rhs, trans="T", check_finite=False)
            return dla.solve_triangular(self.Rqr, y, check_finite=False)
        return dla.cho_solve(self.cho, rhs, check_finite=False)

    def factor(self, Hparts, dp, dd):
        self.Rqr = None
        if self.use_qr:
            try:
                self._factor_qr(dp)
                if self.A0 is not None:
                    Y = self._Msolve(self.A0.T)
                    S = self.A0 @ Y
                    S[np.diag_indices(S.shape[0])] += dd
                    self.Y0 = Y
                    self.choS = dla.cho_factor(S, lower=True, check_finite=False)
                return
            except Exception:
                self.Rqr = None
        c = self.cones
        st = c.st
        n = self.n
        M = self.Pd.copy()
        M[np.diag_indices(n)] += dp
        self.Hinv_blocks = {}
        if self.Ann is not None:
            hi = 1.0 / (st["nn_H"] + dd)
            self.nn_Hinv = hi
            M += (self.Ann.T @ sp.diags(hi) @ self.Ann).toarray()
        self.soc_Hinv = []
        for (d, R), g, As in zip(c.soc_groups, st["soc"], self.Asoc):
            Hb = g["H"] + (dd + REL_REG * np.trace(g["H"], axis1=1, axis2=2) / d)[:, None, None] * np.eye(d)[None]
            Hi = np.linalg.inv(Hb)
            self.soc_Hinv.append(Hi)
            k = R.shape[0]
            Bd = sp.block_diag(list(Hi), format="csr") if k else None
            M += (As.T @ Bd @ As).toarray()
        if self.Aexp is not None:
            Hb = st["exp_H"] + (dd + REL_REG * np.trace(st["exp_H"], axis1=1, axis2=2) / 3)[:, None, None] * np.eye(3)[None]
            Hi = np.linalg.inv(Hb)
            self.exp_Hinv = Hi
            Bd = sp.block_diag(list(Hi), format="csr")
            M += (self.Aexp.T @ Bd @ self.Aexp).toarray()
        self.psd_Hinv = []
        for i, (Ab, used, Au) in enumerate(self.Apsd):
            d = c.psd[i][0]
            if len(used) ** 2 <= 2.5e7 or n * d * d > 4e8:
                Hi = c.psd_Hinv_dense(i, used)
                M += (Au.T @ (Au.T @ Hi).T)
            else:
                # column approach: Y_j = svec(Gi smat(A_j) Gi) using batched dense products
                ix = c.psd_idx[d]
                g = st["psd"][i]
                Gi = g["Rinv"].T @ g["Rinv"]
                Ad = Ab.toarray()  # N x n
                cols = np.where(np.any(Ad != 0, axis=0))[0]
                if len(cols):
                    Smat = np.zeros((len(cols), d, d))
                    vals = Ad[:, cols].T * ix.fi[None, :]
                    Smat[:, ix.I, ix.J] = vals
                    Smat[:, ix.J, ix.I] = vals
                    T = np.matmul(np.matmul(Gi[None], Smat), Gi[None])
                    Y = 0.5 * (T[:, ix.I, ix.J] + T[:, ix.J, ix.I]) * ix.f[None, :]   # ncols x N
                    Mc = Y @ Ad[:, cols]   # ncols x ncols  (= A_c' H^-1 A_c)
                    M[np.ix_(cols, cols)] += 0.5 * (Mc + Mc.T)
        self.M = M
        self.cho = dla.cho_factor(M, lower=True, check_finite=False)
        if self.A0 is not None:
            Y = self._Msolve(self.A0.T)
            S = self.A0 @ Y
            S[np.diag_indices(S.shape[0])] += dd
            self.Y0 = Y
            self.choS = dla.cho_factor(S, lower=True, check_finite=False)

    def _Hinv_apply(self, v):
        c = self.cones
        out = np.zeros_like(v)
        if self.Ann is not None:
            out[c.nn_rows] = self.nn_Hinv * v[c.nn_rows]
        for (d, R), Hi in zip(c.soc_groups, self.soc_Hinv):
            out[R] = np.einsum('kij,kj->ki', Hi, v[R])
        if self.Aexp is not None:
            R = c.exp_rows
            out[R] = np.einsum('kij,kj->ki', self.exp_Hinv, v[R])
        for (d, r), (Ab, used, Au), Hi in zip(c.psd, self.Apsd, self.psd_Hinv):
            # full H^{-1} needed for arbitrary vectors: use the matrix form
            g = c.st["psd"][c.psd.index((d, r))] if False else None
        for i, (d, r) in enumerate(c.psd):
            g = c.st["psd"][i]
            ix = c.psd_idx[d]
            Gi = g["Rinv"].T @ g["Rinv"]
            out[r] = ix.svec(Gi @ ix.smat(v[r]) @ Gi)
        return out

    def _solve_once(self, r1, r2):
        c = self.cones
        hv = self._Hinv_apply(r2)
        hv[c.zero_rows] = 0.0
        rt = r1 + self.AT @ hv
        if self.A0 is not None:
            r20 = r2[c.zero_rows]
            u = self._Msolve(rt)
            z0 = dla.cho_solve(self.choS, self.A0 @ u - r20, check_finite=False)
            x = u - self.Y0 @ z0
        else:
            x = self._Msolve(rt)
            z0 = None
        z = self._Hinv_apply(self.A @ x - r2)
        if z0 is not None:
            z[c.zero_rows] = z0
        return x, z

    def apply_true(self, x, z):
        r1 = self.P @ x + self.AT @ z
        r2 = self.A @ x - self.cones.apply_H(z)
        return r1, r2

    def solve(self, r1, r2, refine=int(__import__("os").environ.get("REFINE", "10"))):
        x, z = self._solve_once(r1, r2)
        nr0 = max(np.max(np.abs(r1), initial=0), np.max(np.abs(r2), initial=0)) + 1e-300

        def resid(x, z):
            a1, a2 = self.apply_true(x, z)
            e1, e2 = r1 - a1, r2 - a2
            return e1, e2, max(np.max(np.abs(e1), initial=0), np.max(np.abs(e2), initial=0))

        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(z))):
            self.last_res = np.inf
            return np.zeros_like(x), np.zeros_like(z)
        e1, e2, nres = resid(x, z)
        for _ in range(refine):
            if nres <= 1e-14 * nr0:
                break
            dx, dz = self._solve_once(e1, e2)
            xn, zn = x + dx, z + dz
            if not (np.all(np.isfinite(xn)) and np.all(np.isfinite(zn))):
                break
            f1, f2, nn_ = resid(xn, zn)
            if not (nn_ < nres):
                break
            improved = nn_ < 0.5 * nres
            x, z, e1, e2, nres = xn, zn, f1, f2, nn_
            if not improved:
                break
        self.last_res = nres / nr0
        return x, z
