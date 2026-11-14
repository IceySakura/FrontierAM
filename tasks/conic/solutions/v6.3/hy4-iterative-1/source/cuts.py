"""Cutting-plane conic solver built on the bundled HiGHS LP engine.

For a cone K and any g in the dual cone K*, the inequality g's >= 0 is
valid for every s in K.  Hence its row

        (g' A_block) x <= g' b_block

is a valid linear *outer* approximation of the conic constraint
s = b - A x in K.  HiGHS solves the resulting LP exactly; its row duals
are nonnegative multiples of the cut normals, so the reconstructed z is
exactly in K*, stationarity is exact and complementarity is exact
(lam * (g's) = 0 at the LP optimum).  Cuts tangent at the most violated
point are added until s lies in K to the required 1e-8 accuracy.

SOC/PSD/exponential cones all admit such tangent cuts:
  SOC          g = (1, -v/||v||)
  PSD          g = svec(u u'),  u eigenvector of lambda_min
  exponential  g = (-1, log(w/v)-1, v/w)   (Fenchel cut for exp(u/v))
"""
import glob
import importlib.util
import math
import os
import sys
import time

import numpy as np

INF = 1e30
_CORE = None


def load_highs():
    global _CORE
    if _CORE is not None:
        return _CORE
    candidates = []
    here = os.path.dirname(os.path.abspath(__file__))
    candidates += sorted(glob.glob(os.path.join(here, "_core*.so")))
    for root in sys.path:
        candidates += sorted(glob.glob(os.path.join(
            root, "scipy", "optimize", "_highspy", "_core*.so")))
    for path in candidates:
        try:
            spec = importlib.util.spec_from_file_location("_core", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_core"] = mod
            spec.loader.exec_module(mod)
            if hasattr(mod, "_Highs"):
                _CORE = mod
                return mod
        except Exception:
            sys.modules.pop("_core", None)
    from scipy.optimize._highspy import _core
    _CORE = _core
    return _core


def _csc_matvec(data, indices, indptr, n, x):
    cols = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr).astype(np.int64))
    return np.bincount(cols, weights=data * x[indices], minlength=n)


class CutProblem:
    def __init__(self, P_data, P_indices, P_indptr, P_shape, q,
                 A_data, A_indices, A_indptr, A_shape, b, cones, deadline):
        self.n = int(q.size)
        self.m = int(b.size)
        self.q = np.asarray(q, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.deadline = deadline
        self.Ad = np.asarray(A_data, dtype=np.float64)
        self.Ai = np.asarray(A_indices, dtype=np.int64)
        self.Ap = np.asarray(A_indptr, dtype=np.int64)
        self.Acols = np.repeat(np.arange(self.n, dtype=np.int64),
                               np.diff(self.Ap).astype(np.int64))
        self.Pd = np.asarray(P_data, dtype=np.float64)
        self.Pi = np.asarray(P_indices, dtype=np.int64)
        self.Pp = np.asarray(P_indptr, dtype=np.int64)
        self.Pcols = np.repeat(np.arange(self.n, dtype=np.int64),
                               np.diff(self.Pp).astype(np.int64))
        self.has_p = bool(self.Pd.size and np.any(self.Pd != 0))
        # Checker uses sum_duplicates() before taking max|A_ij| / max|P_ij|;
        # reproduce that for certificate scaling (cancelling duplicates).
        self.a_scale = self._dup_max(self.Ad, self.Ai, self.Acols, self.n) or 1.0
        self.h_scale = self._dup_max(self.Pd, self.Pi, self.Pcols, self.n) or 1.0
        self.blocks = []
        start = 0
        for c in cones:
            typ, dim = c["type"], c["dim"]
            w = dim * (dim + 1) // 2 if typ == "psd" else dim
            self.blocks.append((typ, dim, w, start))
            start += w
        # base rows: zero (equality) and nonnegative (upper bound) rows
        self.base_rows = []
        self.base_eq = []
        self.base_map = np.full(self.m, -1, dtype=np.int64)
        self.block_of_row = np.full(self.m, -1, dtype=np.int64)
        for k, (typ, dim, w, off) in enumerate(self.blocks):
            if typ in ("zero", "nonnegative"):
                for r in range(off, off + w):
                    self.base_map[r] = len(self.base_rows)
                    self.base_rows.append(r)
                    self.base_eq.append(typ == "zero")
            else:
                self.block_of_row[off:off + w] = k
        self.base_rows = np.asarray(self.base_rows, dtype=np.int64)
        self.base_eq = np.asarray(self.base_eq, dtype=bool)
        self.nbase = len(self.base_rows)
        # per-block sparse column data, used to form cut rows g' A_block
        self.block_data = []
        for k, (typ, dim, w, off) in enumerate(self.blocks):
            if typ in ("zero", "nonnegative"):
                self.block_data.append(None)
                continue
            mask = (self.Ai >= off) & (self.Ai < off + w)
            self.block_data.append((self.Acols[mask],
                                    self.Ai[mask] - off,
                                    self.Ad[mask]))
        # Canonical (duplicate-summed) rowwise triples of the fixed base rows.
        if self.nbase:
            mask = self.base_map[self.Ai] >= 0
            r0 = self.base_map[self.Ai[mask]].astype(np.int64)
            c0 = self.Acols[mask].astype(np.int64)
            v0 = self.Ad[mask]
            if r0.size:
                key = r0 * self.n + c0
                ukey, inv = np.unique(key, return_inverse=True)
                if ukey.size != key.size:
                    v0 = np.bincount(inv, weights=v0, minlength=ukey.size)
                    r0 = ukey // self.n
                    c0 = ukey % self.n
                self.base_triples = (r0, c0, np.ascontiguousarray(v0))
            else:
                self.base_triples = (r0, c0, v0)
        else:
            self.base_triples = (np.empty(0, dtype=np.int64),
                                 np.empty(0, dtype=np.int64), np.empty(0))
        self.cuts = []          # list of (block index, normalised g)
        self.cut_rhs = []       # g' b_block
        self.cut_entries = []   # sparse (column indices, values) of g' A_block
        self.cut_keys = set()
        try:
            from cones import ConeData, smat, svec, exp_violation
            self.cd = ConeData(cones, self.m)
            self.smat, self.svec, self.exp_violation = smat, svec, exp_violation
        except Exception:
            self.cd = None
        self._base_cuts()

    @staticmethod
    def _dup_max(data, rows, cols, ncols):
        """max |entry| after summing duplicate CSC positions."""
        if data.size == 0 or ncols == 0:
            return 0.0
        key = rows.astype(np.int64) * int(ncols) + cols
        u, inv = np.unique(key, return_inverse=True)
        v = np.bincount(inv, weights=data, minlength=u.size)
        return float(np.max(np.abs(v), initial=0.0))

    # -- sparse products ----------------------------------------------------
    def matvec_A(self, x):
        if not self.Ad.size:
            return np.zeros(self.m)
        return np.bincount(self.Ai, weights=self.Ad * x[self.Acols],
                           minlength=self.m)

    def matvec_P(self, x):
        if not self.has_p:
            return np.zeros(self.n)
        return np.bincount(self.Pi, weights=self.Pd * x[self.Pcols],
                           minlength=self.n)

    def matvec_At(self, y):
        if not self.Ad.size:
            return np.zeros(self.n)
        return np.bincount(self.Acols, weights=self.Ad * y[self.Ai],
                           minlength=self.n)

    # -- cut management -----------------------------------------------------
    def _add(self, k, g):
        g = np.asarray(g, dtype=np.float64)
        scale = np.max(np.abs(g))
        if not np.isfinite(scale) or scale <= 0:
            return False
        g = g / scale
        key = (k, tuple(np.round(g, 15)))
        if key in self.cut_keys:
            return False
        self.cut_keys.add(key)
        typ, dim, w, off = self.blocks[k]
        self.cuts.append((k, g))
        self.cut_rhs.append(float(g @ self.b[off:off + w]))
        cols, rows, vals = self.block_data[k]
        if cols.size:
            u, inv = np.unique(cols, return_inverse=True)
            wv = np.bincount(inv, weights=vals * g[rows], minlength=u.size)
            nz = np.nonzero(wv)[0]
            self.cut_entries.append((u[nz].astype(np.int32),
                                     np.ascontiguousarray(wv[nz])))
        else:
            self.cut_entries.append((np.empty(0, dtype=np.int32),
                                     np.empty(0, dtype=np.float64)))
        return True

    def _base_cuts(self):
        for k, (typ, dim, w, off) in enumerate(self.blocks):
            if typ == "soc":
                js = range(1, dim) if dim <= 20 else \
                    sorted(set(list(range(1, 17)) +
                               [int(round(1 + (dim - 2) * t / 15.0))
                                for t in range(16)]))
                for j in js:
                    for sgn in (1.0, -1.0):
                        g = np.zeros(w)
                        g[0], g[j] = 1.0, sgn
                        self._add(k, g)
            elif typ == "psd":
                for i in range(dim):
                    v = np.zeros(dim)
                    v[i] = 1.0
                    self._add(k, self.svec(np.outer(v, v), dim))
                if dim <= 20:
                    for i in range(dim):
                        for j in range(i + 1, dim):
                            for sgn in (1.0, -1.0):
                                v = np.zeros(dim)
                                v[i], v[j] = 1.0, sgn
                                self._add(k, self.svec(np.outer(v, v), dim))
                else:
                    rng = np.random.default_rng(12345)
                    for _ in range(60):
                        v = rng.standard_normal(dim)
                        self._add(k, self.svec(np.outer(v, v), dim))
            elif typ == "exponential":
                self._add(k, np.array([0.0, 1.0, 0.0]))
                self._add(k, np.array([0.0, 0.0, 1.0]))
                for tau in (1e-4, 1e-2, 1.0, 1e2, 1e4):
                    self._add(k, np.array([-1.0, math.log(tau) - 1.0, 1.0 / tau]))

    def violations(self, s):
        """Return (worst normalised cone defect, list of new cuts)."""
        worst = 0.0
        new = []
        for k, (typ, dim, w, off) in enumerate(self.blocks):
            blk = s[off:off + w]
            if w == 0:
                continue
            u = blk / max(1.0, float(np.max(np.abs(blk))))
            if typ == "zero":
                v = float(np.max(np.abs(u)))
            elif typ == "nonnegative":
                v = max(0.0, -float(np.min(u)))
            elif typ == "soc":
                nv = float(np.linalg.norm(u[1:]))
                v = max(0.0, nv - float(u[0]))
                if v > 1e-12 and nv > 0:
                    g = np.zeros(w)
                    g[0] = 1.0
                    g[1:] = -u[1:] / nv
                    new.append((k, g))
            elif typ == "psd":
                M = self.smat(u, dim)
                ev, V = np.linalg.eigh(M)
                v = max(0.0, -float(ev[0]))
                if v > 1e-12:
                    nneg = min(dim, 12, int(np.sum(ev < -1e-12)))
                    for j in range(max(1, nneg)):
                        uu = V[:, j]
                        new.append((k, self.svec(np.outer(uu, uu), dim)))
            else:  # exponential
                v = self.exp_violation(u)
                if v > 1e-12:
                    uu, vv, ww = float(u[0]), float(u[1]), float(u[2])
                    if vv > 0 and ww > 0:
                        tau = max(1e-14, ww / vv)
                        for mult in (1.0, 0.45, 2.2, 0.15, 6.0):
                            t = tau * mult
                            new.append((k, np.array([-1.0, math.log(t) - 1.0,
                                                     1.0 / t])))
                    else:
                        new.append((k, np.array([0.0, 1.0, 0.0])))
                        new.append((k, np.array([0.0, 0.0, 1.0])))
                        # Fenchel cuts also separate points with u>0, w<=0
                        # that the two closure rays alone do not separate.
                        for tv in (1e-6, 1e-3, 1.0, 1e3):
                            new.append((k, np.array([-1.0, math.log(tv) - 1.0,
                                                     1.0 / tv])))
            if v > worst:
                worst = v
        return worst, new

    # -- LP assembly --------------------------------------------------------
    def _assemble(self):
        nrows = self.nbase + len(self.cuts)
        M = np.zeros((nrows, self.n))
        if self.nbase:
            mask = self.base_map[self.Ai] >= 0
            if np.any(mask):
                np.add.at(M, (self.base_map[self.Ai[mask]],
                              self.Acols[mask]), self.Ad[mask])
        for j, (cc, vv) in enumerate(self.cut_entries):
            if cc.size:
                M[self.nbase + j, cc] = vv
        lo, hi = self._bounds(nrows)
        rr, cc = np.nonzero(M)
        starts = np.searchsorted(rr, np.arange(nrows + 1)).astype(np.int32)
        return M, lo, hi, starts, cc.astype(np.int32), M[rr, cc].astype(np.float64)

    def _bounds(self, nrows):
        lo = np.empty(nrows)
        hi = np.empty(nrows)
        if self.nbase:
            bb = self.b[self.base_rows]
            lo[:self.nbase] = np.where(self.base_eq, bb, -INF)
            hi[:self.nbase] = bb
        if self.cuts:
            lo[self.nbase:] = -INF
            hi[self.nbase:] = np.asarray(self.cut_rhs)
        return lo, hi

    def _assemble_sparse(self):
        """Rowwise CSR of the LP matrix, without ever forming a dense one."""
        nrows = self.nbase + len(self.cuts)
        rl, cl, vl = [], [], []
        if self.nbase and self.base_triples[0].size:
            rl.append(self.base_triples[0])
            cl.append(self.base_triples[1])
            vl.append(self.base_triples[2])
        for j, (cc, vv) in enumerate(self.cut_entries):
            if cc.size:
                rl.append(np.full(cc.size, self.nbase + j, dtype=np.int64))
                cl.append(cc.astype(np.int64))
                vl.append(vv)
        if rl:
            rr = np.concatenate(rl)
            cc = np.concatenate(cl)
            vv = np.concatenate(vl)
            order = np.argsort(rr, kind="stable")
            rr = rr[order]
            cc = cc[order]
            vv = vv[order]
            starts = np.searchsorted(rr, np.arange(nrows + 1)).astype(np.int32)
        else:
            cc, vv = np.empty(0, dtype=np.int64), np.empty(0)
            starts = np.zeros(nrows + 1, dtype=np.int32)
        lo, hi = self._bounds(nrows)
        return lo, hi, starts, cc.astype(np.int32), np.ascontiguousarray(vv)

    def _cut_row(self, k, g):
        cols, rows, vals = self.block_data[k]
        if not cols.size:
            return np.zeros(self.n)
        return np.bincount(cols, weights=vals * g[rows], minlength=self.n)

    def _lp_model(self, core, lo, hi, starts, index, value, with_hessian=True):
        nrows = lo.size
        lp = core.HighsLp()
        lp.num_col_ = self.n
        lp.num_row_ = int(nrows)
        lp.col_cost_ = np.ascontiguousarray(self.q, dtype=np.float64)
        lp.col_lower_ = np.full(self.n, -INF, dtype=np.float64)
        lp.col_upper_ = np.full(self.n, INF, dtype=np.float64)
        lp.row_lower_ = np.ascontiguousarray(lo, dtype=np.float64)
        lp.row_upper_ = np.ascontiguousarray(hi, dtype=np.float64)
        lp.sense_ = core.ObjSense.kMinimize
        mat = lp.a_matrix_
        mat.format_ = core.MatrixFormat.kRowwise
        mat.num_col_ = self.n
        mat.num_row_ = int(nrows)
        mat.start_ = np.ascontiguousarray(starts, dtype=np.int32)
        mat.index_ = np.ascontiguousarray(index, dtype=np.int32)
        mat.value_ = np.ascontiguousarray(value, dtype=np.float64)
        if self.has_p and with_hessian and self.n <= 20000:
            model = core.HighsModel()
            model.lp_ = lp
            hess = core.HighsHessian()
            hess.dim_ = self.n
            hess.format_ = core.HessianFormat.kTriangular
            # lower triangle, column-wise (HiGHS convention)
            mask = self.Pi >= self.Pcols
            hc, hr, hv = self.Pcols[mask], self.Pi[mask], self.Pd[mask]
            counts = np.bincount(hc, minlength=self.n)
            hess.start_ = np.r_[0, np.cumsum(counts)].astype(np.int32)
            hess.index_ = hr.astype(np.int32)
            hess.value_ = hv
            model.hessian_ = hess
            return model
        return lp

    def _solve_lp(self, core, highs, lo, hi, starts, index, value):
        highs.passModel(self._lp_model(core, lo, hi, starts, index, value))
        highs.run()
        return highs.getModelStatus(), highs.getSolution()

    # -- exact active-set KKT polish for QP models --------------------------
    def _dense_P(self):
        if self.n > 3000:
            return None
        P = np.zeros((self.n, self.n))
        if self.has_p:
            np.add.at(P, (self.Pi, self.Pcols), self.Pd)
            P = 0.5 * (P + P.T)
        return P

    def _polish_qp(self, x, yd, M, hi):
        """Solve the equality-constrained KKT system on the active rows."""
        try:
            nrows = M.shape[0]
            lam_all = -np.asarray(yd, dtype=np.float64).reshape(-1)
            scale = max(1.0, float(np.max(np.abs(lam_all), initial=0.0)))
            tol_a = 1e-10 * scale
            eqmask = self._ineq_mask(nrows) == False
            act = []
            for i in range(self.nbase):
                if eqmask[i] or lam_all[i] > tol_a:
                    act.append(i)
            for j in range(len(self.cuts)):
                if lam_all[self.nbase + j] > tol_a:
                    act.append(self.nbase + j)
            if not act:
                return None
            act = np.asarray(act, dtype=np.int64)
            Aact = M[act, :]
            bact = np.asarray(hi, dtype=np.float64)[act]
            na = act.size
            P = self._dense_P()
            if P is None or self.n + na > 4500:
                return None
            K = np.zeros((self.n + na, self.n + na))
            K[:self.n, :self.n] = P
            K[:self.n, self.n:] = Aact.T
            K[self.n:, :self.n] = Aact
            K[self.n:, self.n:].flat[::na + 1] = -1e-12
            rhs = np.concatenate((-self.q, bact))
            try:
                sol = np.linalg.solve(K, rhs)
            except np.linalg.LinAlgError:
                if self.n + na > 700:
                    return None
                sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
            x_new = sol[:self.n]
            lam = sol[self.n:]
            z = np.zeros(self.m)
            cut_pos = 0
            for pos, i in enumerate(act):
                if i < self.nbase:
                    z[self.base_rows[i]] = lam[pos]
                else:
                    j = int(i - self.nbase)
                    k, g = self.cuts[j]
                    typ, dim, w, off = self.blocks[k]
                    z[off:off + w] += lam[pos] * g
            s = self.b - self.matvec_A(x_new)
            ok, _ = self.validate(x_new, s, z, tol=3e-9)
            if ok and np.isfinite(x_new).all() and np.isfinite(z).all():
                return {"status": "solved", "x": x_new, "s": s, "z": z}
        except Exception:
            return None
        return None

    # -- validation ---------------------------------------------------------
    def validate(self, x, s, z, tol=3e-9):
        Ax = self.matvec_A(x)
        Px = self.matvec_P(x)
        Atz = self.matvec_At(z)
        rp = float(np.max(np.abs(Ax + s - self.b), initial=0.0))
        denp = 1.0 + max(float(np.max(np.abs(Ax), initial=0.0)),
                         float(np.max(np.abs(s), initial=0.0)),
                         float(np.max(np.abs(self.b), initial=0.0)))
        rdn = Px + self.q + Atz
        rd = float(np.max(np.abs(rdn), initial=0.0))
        dend = 1.0 + max(float(np.max(np.abs(Px), initial=0.0)),
                         float(np.max(np.abs(self.q), initial=0.0)),
                         float(np.max(np.abs(Atz), initial=0.0)))
        rp, rd = rp / denp, rd / dend
        with np.errstate(over="ignore", invalid="ignore"):
            xPx = float(x @ Px)
            p0 = 0.5 * xPx + float(self.q @ x)
            d0 = -0.5 * xPx - float(self.b @ z)
            comp = float(s @ z)
        if not (np.isfinite(p0) and np.isfinite(d0) and np.isfinite(comp)):
            return False, (rp, rd, float("inf"), float("inf"), 0.0, 0.0)
        gap = abs(p0 - d0)
        rel = gap / max(abs(p0), abs(d0)) if max(abs(p0), abs(d0)) > 0 else 0.0
        cp = self.cd.defect_primal(s) if self.cd is not None else 0.0
        cz = self.cd.defect_dual(z) if self.cd is not None else 0.0
        ok = (np.isfinite(rp) and np.isfinite(rd) and rp <= tol and rd <= tol
              and cp <= tol and cz <= tol and (gap <= tol or rel <= tol))
        return ok, (rp, rd, gap, rel, cp, cz)

    # -- main loop ----------------------------------------------------------
    def _fresh_highs(self, core):
        highs = core._Highs()
        highs.setOptionValue("output_flag", False)
        highs.setOptionValue("threads", 1)
        highs.setOptionValue("presolve", "on")
        highs.setOptionValue("solver", "choose")
        highs.setOptionValue("primal_feasibility_tolerance", 1e-10)
        highs.setOptionValue("dual_feasibility_tolerance", 1e-10)
        highs.setOptionValue("ipm_optimality_tolerance", 1e-12)
        highs.setOptionValue("time_limit",
                             max(0.05, self.deadline - time.perf_counter()))
        return highs

    def _qp_active_loop(self, A, P, active, eq, max_steps):
        n, m = self.n, self.m
        seen = set()
        for _ in range(max_steps):
            idx = np.nonzero(active)[0]
            if idx.size == 0:
                return None
            key = idx.tobytes()
            if key in seen:
                return None
            seen.add(key)
            na = idx.size
            if n + na > 4500:
                return None
            K = np.zeros((n + na, n + na))
            K[:n, :n] = P
            K[:n, n:] = A[idx].T
            K[n:, :n] = A[idx]
            K[n:, n:].flat[::na + 1] = -1e-12
            rhs = np.concatenate((-self.q, self.b[idx]))
            try:
                sol = np.linalg.solve(K, rhs)
            except np.linalg.LinAlgError:
                if n + na > 1200:
                    return None
                sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
            if not np.isfinite(sol).all():
                return None
            x = sol[:n]
            lam = sol[n:]
            z = np.zeros(m)
            z[idx] = lam
            s = self.b - A @ x
            ok, _ = self.validate(x, s, z, tol=9.5e-9)
            if ok:
                return {"status": "solved", "x": x, "s": s, "z": z}
            lam_rows = np.zeros(m)
            lam_rows[idx] = lam
            tols = 1e-9 * max(1.0, float(np.max(np.abs(s), initial=0.0)))
            toll = 1e-9 * max(1.0, float(np.max(np.abs(lam), initial=0.0)))
            add = (~active) & (~eq) & (s < -tols)
            drop = active & (~eq) & (lam_rows < -toll)
            new_active = active.copy()
            new_active[add] = True
            new_active[drop] = False
            if np.array_equal(new_active, active):
                return None
            active = new_active
        return None

    def solve_qp_active_set(self, core, max_steps=14):
        """Iterative primal active-set method for a convex QP with only
        zero/nonnegative cones.  Tries the fast LP-relaxation basis first and
        then the unconstrained-QP basis (needed when the LP is unbounded)."""
        if self.cuts or not self.has_p or self.nbase != self.m:
            return None
        n, m = self.n, self.m
        if float(n) * m > 2.0e7 or n > 400:
            return None
        max_steps = min(max_steps, 18)
        try:
            A = np.zeros((m, n))
            np.add.at(A, (self.Ai, self.Acols), self.Ad)
            P = self._dense_P()
            if P is None:
                return None
            eq = self.base_eq.copy()
            # start 1: LP relaxation active set
            M, lo, hi, starts, index, value = self._assemble()
            h = self._fresh_highs(core)
            h.passModel(self._lp_model(core, lo, hi, starts, index, value,
                                       with_hessian=False))
            h.run()
            if h.getModelStatus() == core.HighsModelStatus.kOptimal:
                yd = np.asarray(h.getSolution().row_dual, dtype=np.float64).reshape(-1)
                lam0 = -yd
                tol0 = 1e-9 * max(1.0, float(np.max(np.abs(lam0), initial=0.0)))
                got = self._qp_active_loop(A, P, eq | (lam0 > tol0), eq,
                                           max_steps)
                if got is not None:
                    return got
            # start 2: equality-only active set (unconstrained QP optimum)
            got = self._qp_active_loop(A, P, eq.copy(), eq, max_steps)
            if got is not None:
                return got
        except Exception:
            return None
        return None

    def solve(self, max_iter=60):
        core = load_highs()
        if float((self.nbase + len(self.cuts)) * self.n) > 3.0e7:
            return self.solve_sparse(max_iter)
        if self.has_p and not self.cuts and self.nbase == self.m:
            got = self.solve_qp_active_set(core)
            if got is not None:
                return got
        # Pure convex QP (only zero/nonnegative rows): HiGHS' QP algorithm is
        # slow on larger models.  A plain LP relaxation is fast, and an
        # active-set KKT solve on its optimal basis often already gives the
        # exact QP optimum.  Try that first.
        if self.has_p and not self.cuts and self.nbase == self.m:
            try:
                M, lo, hi, starts, index, value = self._assemble()
                h = self._fresh_highs(core)
                h.passModel(self._lp_model(core, lo, hi, starts, index, value,
                                           with_hessian=False))
                h.run()
                if h.getModelStatus() == core.HighsModelStatus.kOptimal:
                    sol = h.getSolution()
                    x = np.asarray(sol.col_value, dtype=np.float64).reshape(-1)
                    yd = np.asarray(sol.row_dual, dtype=np.float64).reshape(-1)
                    polished = self._polish_qp(x, yd, M, hi)
                    if polished is not None:
                        return polished
            except Exception:
                pass
        best = None
        # A fresh HiGHS model is built on every outer iteration: reusing a
        # simplex basis across cuts loses ~1 decimal of accuracy near the
        # cone boundary, which is decisive at the 1e-8 checker tolerance.
        for it in range(max_iter):
            if time.perf_counter() > self.deadline:
                break
            nrows = self.nbase + len(self.cuts)
            if float(nrows) * self.n > 3.0e7:
                break
            M, lo, hi, starts, index, value = self._assemble()
            csr = (starts, index, value, nrows)
            csr_t = self._csr_of_transpose(M)
            highs = self._fresh_highs(core)
            status, sol = self._solve_lp(core, highs, lo, hi, starts, index, value)
            pending = []
            if status == core.HighsModelStatus.kOptimal:
                x = np.asarray(sol.col_value, dtype=np.float64).reshape(-1)
                if x.size != self.n or not np.isfinite(x).all():
                    break
                s = self.b - self.matvec_A(x)
                worst, new = self.violations(s)
                z = np.zeros(self.m)
                yd = np.asarray(sol.row_dual, dtype=np.float64).reshape(-1)
                if self.nbase:
                    z[self.base_rows] = -yd[:self.nbase]
                for j, (k, g) in enumerate(self.cuts):
                    lam = -yd[self.nbase + j]
                    if lam != 0.0:
                        typ, dim, w, off = self.blocks[k]
                        z[off:off + w] += lam * g
                ok, res = self.validate(x, s, z, tol=9.5e-9)
                if ok:
                    return {"status": "solved", "x": x, "s": s, "z": z}
                if self.has_p and worst < 2e-7:
                    polished = self._polish_qp(x, yd, M, hi)
                    if polished is not None:
                        return polished
                if best is None or worst < best[0]:
                    best = (worst, x.copy(), s.copy(), z.copy())
                    self._best_polish = (x.copy(), np.asarray(yd).copy(),
                                         M.copy(), np.asarray(hi).copy())
                pending = new
            elif status in (core.HighsModelStatus.kInfeasible,
                            core.HighsModelStatus.kUnboundedOrInfeasible):
                cert = self._primal_certificate(core, lo, hi, csr_t)
                if cert is not None:
                    return cert
                break
            elif status == core.HighsModelStatus.kUnbounded:
                cert = self._dual_certificate(core, csr)
                if cert is not None:
                    return cert
                # The relaxation is unbounded but the conic problem may not
                # be: add tangent cuts at the recession ray's slacks.
                xray = self._recession_ray(core, csr)
                if xray is None:
                    break
                _, pending = self.violations(-self.matvec_A(xray))
            else:
                break

            if not pending:
                break
            if sum(self._add(k, g) for k, g in pending) == 0:
                break
        # last chance: return the best candidate only if it validates
        if best is not None:
            _, x, s, z = best
            ok, _ = self.validate(x, s, z, tol=9.5e-9)
            if ok:
                return {"status": "solved", "x": x, "s": s, "z": z}
            if self.has_p and getattr(self, "_best_polish", None) is not None:
                polished = self._polish_qp(*self._best_polish)
                if polished is not None:
                    return polished
        return {"status": "unresolved"}

    def _transpose_csr(self, starts, index, value, nrows):
        """CSR (rowwise) of M' from CSR of M, without a dense matrix."""
        if value.size == 0:
            return (np.zeros(self.n + 1, dtype=np.int32),
                    np.empty(0, dtype=np.int32), np.empty(0))
        row = np.repeat(np.arange(nrows, dtype=np.int64),
                        np.diff(starts).astype(np.int64))
        order = np.argsort(index, kind="stable")
        sr = index[order].astype(np.int64)
        sc = row[order].astype(np.int32)
        sv = np.ascontiguousarray(value[order])
        st = np.searchsorted(sr, np.arange(self.n + 1)).astype(np.int32)
        return st, sc, sv

    def _sparse_active_qp(self, core, lo, hi, starts, index, value, nrows):
        """Large convex QP: iterative sparse active-set KKT solves."""
        try:
            from scipy import sparse
            from scipy.sparse.linalg import spsolve
            row = np.repeat(np.arange(nrows, dtype=np.int64),
                            np.diff(starts).astype(np.int64))
            A = sparse.csr_matrix((value, (row, index)),
                                  shape=(nrows, self.n)).tocsc()
            P = sparse.csc_matrix((self.Pd, self.Pi, self.Pp),
                                  shape=(self.n, self.n))
            P = 0.5 * (P + P.T).tocsc()
            pscale = max(1.0, float(np.max(np.abs(self.Pd), initial=0.0)))
            delta = 1e-10 * pscale
            eq = ~self._ineq_mask(nrows)
            starts_list = []
            h = self._fresh_highs(core)
            h.passModel(self._lp_model(core, lo, hi, starts, index, value,
                                       with_hessian=False))
            h.run()
            if h.getModelStatus() == core.HighsModelStatus.kOptimal:
                yd = np.asarray(h.getSolution().row_dual,
                                dtype=np.float64).reshape(-1)
                lam0 = -yd
                tol_a = 1e-9 * max(1.0, float(np.max(np.abs(lam0), initial=0.0)))
                starts_list.append(eq | (lam0 > tol_a))
            starts_list.append(eq.copy())
            hiarr = np.asarray(hi)
            for active in starts_list:
                seen = set()
                for _ in range(18):
                    idx = np.nonzero(active)[0]
                    if idx.size == 0 or idx.size > 200000:
                        break
                    key = idx.tobytes()
                    if key in seen:
                        break
                    seen.add(key)
                    Aact = A[idx]
                    reg = -delta * sparse.identity(idx.size, format="csc")
                    Pr = P + delta * sparse.identity(self.n, format="csc")
                    K = sparse.bmat([[Pr, Aact.T], [Aact, reg]], format="csc")
                    rhs = np.concatenate((-self.q, hiarr[idx]))
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with np.errstate(all="ignore"):
                            solk = spsolve(K, rhs)
                    if not np.isfinite(solk).all() or \
                            float(np.max(np.abs(K @ solk - rhs), initial=0.0)) > \
                            1e-6 * (1.0 + float(np.max(np.abs(rhs), initial=0.0))):
                        try:
                            from scipy.sparse.linalg import lsqr
                            fit = lsqr(K, rhs, atol=1e-13, btol=1e-13,
                                       iter_lim=min(4000, 8 * self.n))
                            solk = fit[0]
                        except Exception:
                            break
                    if not np.isfinite(solk).all():
                        break
                    x_new = solk[:self.n]
                    lam = solk[self.n:]
                    z = np.zeros(self.m)
                    z[self.base_rows[idx]] = lam
                    s = self.b - self.matvec_A(x_new)
                    ok, _ = self.validate(x_new, s, z, tol=9.5e-9)
                    if ok:
                        return {"status": "solved", "x": x_new, "s": s, "z": z}
                    lam_rows = np.zeros(nrows)
                    lam_rows[idx] = lam
                    tols = 1e-9 * max(1.0, float(np.max(np.abs(s), initial=0.0)))
                    toll = 1e-9 * max(1.0, float(np.max(np.abs(lam), initial=0.0)))
                    add = (~active) & (~eq) & (s < -tols)
                    drop = active & (~eq) & (lam_rows < -toll)
                    new_active = active.copy()
                    new_active[add] = True
                    new_active[drop] = False
                    if np.array_equal(new_active, active):
                        break
                    active = new_active
        except Exception:
            return None
        return None

    def solve_sparse(self, max_iter=40):
        """Cutting-plane loop for models too large for a dense LP matrix."""
        core = load_highs()
        if self.has_p and not self.cuts:
            lo, hi, starts, index, value = self._assemble_sparse()
            got = self._sparse_active_qp(core, lo, hi, starts, index, value,
                                         lo.size)
            if got is not None:
                return got
            return {"status": "unresolved"}
        best = None
        for it in range(max_iter):
            if time.perf_counter() > self.deadline:
                break
            nrows = self.nbase + len(self.cuts)
            if nrows > 3000000:
                break
            lo, hi, starts, index, value = self._assemble_sparse()
            if value.size > 2.0e7:
                break
            csr = (starts, index, value, nrows)
            csr_t = self._transpose_csr(starts, index, value, nrows)
            highs = self._fresh_highs(core)
            status, sol = self._solve_lp(core, highs, lo, hi, starts, index, value)
            pending = []
            if status == core.HighsModelStatus.kOptimal:
                x = np.asarray(sol.col_value, dtype=np.float64).reshape(-1)
                if x.size != self.n or not np.isfinite(x).all():
                    break
                s = self.b - self.matvec_A(x)
                worst, new = self.violations(s)
                z = np.zeros(self.m)
                yd = np.asarray(sol.row_dual, dtype=np.float64).reshape(-1)
                if self.nbase:
                    z[self.base_rows] = -yd[:self.nbase]
                for j, (k, g) in enumerate(self.cuts):
                    lam = -yd[self.nbase + j]
                    if lam != 0.0:
                        typ, dim, w, off = self.blocks[k]
                        z[off:off + w] += lam * g
                ok, _ = self.validate(x, s, z, tol=9.5e-9)
                if ok:
                    return {"status": "solved", "x": x, "s": s, "z": z}
                if best is None or worst < best[0]:
                    best = (worst, x.copy(), s.copy(), z.copy())
                pending = new
            elif status in (core.HighsModelStatus.kInfeasible,
                            core.HighsModelStatus.kUnboundedOrInfeasible):
                cert = self._primal_certificate(core, lo, hi, csr_t)
                if cert is not None:
                    return cert
                break
            elif status == core.HighsModelStatus.kUnbounded:
                cert = self._dual_certificate(core, csr)
                if cert is not None:
                    return cert
                xray = self._recession_ray(core, csr)
                if xray is None:
                    break
                _, pending = self.violations(-self.matvec_A(xray))
            else:
                break
            if not pending:
                break
            if sum(self._add(k, g) for k, g in pending) == 0:
                break
        if best is not None:
            _, x, s, z = best
            ok, _ = self.validate(x, s, z, tol=9.5e-9)
            if ok:
                return {"status": "solved", "x": x, "s": s, "z": z}
        return {"status": "unresolved"}

    def solve_fast(self, max_iter=24):
        """HiGHS basis-reusing cutting planes for the early, coarse phase.

        Once the conic violation is small we STOP reusing the basis, because
        that costs about one decimal of accuracy near the 1e-8 boundary; the
        caller then falls back to the fresh-model `solve()` loop.
        """
        if self.has_p:
            return {"status": "unresolved"}
        core = load_highs()
        nrows = self.nbase + len(self.cuts)
        if float(nrows) * self.n > 3.0e7:
            return {"status": "unresolved"}
        try:
            M, lo, hi, starts, index, value = self._assemble()
            h = self._fresh_highs(core)
            h.passModel(self._lp_model(core, lo, hi, starts, index, value))
            h.run()
            for _ in range(max_iter):
                if time.perf_counter() > self.deadline:
                    break
                status, sol = h.getModelStatus(), h.getSolution()
                if status != core.HighsModelStatus.kOptimal:
                    break
                x = np.asarray(sol.col_value, dtype=np.float64).reshape(-1)
                if x.size != self.n or not np.isfinite(x).all():
                    break
                s = self.b - self.matvec_A(x)
                worst, new = self.violations(s)
                yd = np.asarray(sol.row_dual, dtype=np.float64).reshape(-1)
                z = np.zeros(self.m)
                if self.nbase:
                    z[self.base_rows] = -yd[:self.nbase]
                for j, (k, g) in enumerate(self.cuts):
                    lam = -yd[self.nbase + j]
                    if lam:
                        typ, dim, w, off = self.blocks[k]
                        z[off:off + w] += lam * g
                ok, _ = self.validate(x, s, z, tol=9.5e-9)
                if ok:
                    return {"status": "solved", "x": x, "s": s, "z": z}
                if worst <= 1e-3 or not new:
                    break
                rows, rlo, rhi = [], [], []
                for k, g in new:
                    if self._add(k, g):
                        kk, gg = self.cuts[-1]
                        cc, vv = self.cut_entries[-1]
                        row = np.zeros(self.n)
                        if cc.size:
                            row[cc] = vv
                        rows.append(row)
                        rlo.append(-INF)
                        rhi.append(self.cut_rhs[-1])
                if not rows:
                    break
                R = np.asarray(rows)
                rr, cc = np.nonzero(R)
                st = np.searchsorted(rr, np.arange(len(rows) + 1)).astype(np.int32)
                if h.addRows(len(rows), np.asarray(rlo), np.asarray(rhi),
                             cc.size, st, cc.astype(np.int32),
                             np.ascontiguousarray(R[rr, cc])) != core.HighsStatus.kOk:
                    break
                M = np.vstack((M, R))
                lo = np.concatenate((lo, np.asarray(rlo)))
                hi = np.concatenate((hi, np.asarray(rhi)))
                h.run()
        except Exception:
            pass
        return {"status": "unresolved"}

    # -- auxiliary LP helper ------------------------------------------------
    def _aux_lp(self, core, cost, clo, cup, rlo, rup, starts, index, value,
                nvar, nrow, tlimit=1.0):
        h = core._Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.setOptionValue("presolve", "on")
        h.setOptionValue("solver", "choose")
        h.setOptionValue("primal_feasibility_tolerance", 1e-10)
        h.setOptionValue("dual_feasibility_tolerance", 1e-10)
        h.setOptionValue("time_limit", max(0.05, tlimit))
        lp = core.HighsLp()
        lp.num_col_ = int(nvar)
        lp.num_row_ = int(nrow)
        lp.col_cost_ = np.ascontiguousarray(cost, dtype=np.float64)
        lp.col_lower_ = np.ascontiguousarray(clo, dtype=np.float64)
        lp.col_upper_ = np.ascontiguousarray(cup, dtype=np.float64)
        lp.row_lower_ = np.ascontiguousarray(rlo, dtype=np.float64)
        lp.row_upper_ = np.ascontiguousarray(rup, dtype=np.float64)
        lp.sense_ = core.ObjSense.kMinimize
        mat = lp.a_matrix_
        mat.format_ = core.MatrixFormat.kRowwise
        mat.num_col_ = int(nvar)
        mat.num_row_ = int(nrow)
        mat.start_ = np.ascontiguousarray(starts, dtype=np.int32)
        mat.index_ = np.ascontiguousarray(index, dtype=np.int32)
        mat.value_ = np.ascontiguousarray(value, dtype=np.float64)
        if h.passModel(lp) != core.HighsStatus.kOk:
            return None, None
        h.run()
        return h.getModelStatus(), h.getSolution()

    def _csr_of_transpose(self, M):
        # rowwise CSR of M.T, i.e. entries sorted by column of M
        Mt = np.ascontiguousarray(M.T)
        rr, cc = np.nonzero(Mt)
        starts = np.searchsorted(rr, np.arange(self.n + 1)).astype(np.int32)
        return starts, cc.astype(np.int32), Mt[rr, cc].astype(np.float64)

    def _csr_of(self, M):
        rr, cc = np.nonzero(M)
        nrow = M.shape[0]
        starts = np.searchsorted(rr, np.arange(nrow + 1)).astype(np.int32)
        return starts, cc.astype(np.int32), M[rr, cc].astype(np.float64)

    # -- primal infeasibility (Farkas) --------------------------------------
    def _primal_certificate(self, core, lo, hi, csr_t):
        nrows = lo.size
        starts, index, value = csr_t
        clo = np.where(self._ineq_mask(nrows), 0.0, -1.0)
        cup = np.ones(nrows)
        status, sol = self._aux_lp(
            core, hi, clo, cup,
            np.zeros(self.n), np.zeros(self.n), starts, index, value,
            nrows, self.n)
        if status != core.HighsModelStatus.kOptimal or sol is None:
            return None
        y = np.asarray(sol.col_value, dtype=np.float64).reshape(-1)
        if y.size != nrows:
            return None
        z = np.zeros(self.m)
        z[self.base_rows] = y[:self.nbase]
        for j, (k, g) in enumerate(self.cuts):
            typ, dim, w, off = self.blocks[k]
            z[off:off + w] += y[self.nbase + j] * g
        z = self._clip_dual_small(z)
        if not np.isfinite(z).all() or not np.any(z):
            return None
        a = self.a_scale
        zn = z / float(np.max(np.abs(z)))
        bscale = float(np.max(np.abs(self.b), initial=0.0)) or 1.0
        beta = -float((self.b / bscale) @ zn)
        tol = 8e-9
        if beta <= tol:
            return None
        lin = self.matvec_At(zn) / a
        if float(np.max(np.abs(lin), initial=0.0)) / beta > tol:
            return None
        if self.cd.defect_dual(zn) / beta > tol:
            return None
        return {"status": "primal_infeasible", "z": z}

    def _ineq_mask(self, nrows):
        mask = np.ones(nrows, dtype=bool)
        if self.nbase:
            mask[:self.nbase] = ~self.base_eq
        return mask

    def _clip_dual_small(self, z):
        # Suppress roundoff-sized negative multipliers on inequality rows.
        ineq = self._ineq_mask(self.nbase + len(self.cuts))
        scale = max(1.0, float(np.max(np.abs(z), initial=0.0)))
        for i in range(min(self.nbase, z.size)):
            if self.base_rows[i] < z.size and ineq[i]:
                r = self.base_rows[i]
                if -scale * 1e-11 < z[r] < 0:
                    z[r] = 0.0
        return z

    # -- dual infeasibility (recession direction) ---------------------------
    def _recession_ray(self, core, csr):
        starts, index, value, nrows = csr
        eq = ~self._ineq_mask(nrows)
        rlo = np.where(eq, 0.0, -INF)
        rup = np.zeros(nrows)
        status, sol = self._aux_lp(
            core, self.q, np.full(self.n, -1.0), np.ones(self.n),
            rlo, rup, starts, index, value, self.n, nrows)
        if status != core.HighsModelStatus.kOptimal or sol is None:
            return None
        x = np.asarray(sol.col_value, dtype=np.float64).reshape(-1)
        if x.size != self.n or not np.isfinite(x).all():
            return None
        return x

    def _dual_certificate(self, core, csr):
        x = self._recession_ray(core, csr)
        if x is None:
            return None
        s = -self.matvec_A(x)
        qn = float(np.max(np.abs(self.q), initial=0.0))
        xn = float(np.max(np.abs(x), initial=0.0))
        if qn <= 0 or xn <= 0:
            return None
        xh = x / xn
        sh = s / xn
        alpha = -float((self.q / qn) @ xh)
        tol = 8e-9
        if alpha <= tol:
            return None
        a = self.a_scale
        h = self.h_scale
        if float(np.max(np.abs(self.matvec_P(xh) / h), initial=0.0)) / alpha > tol:
            return None
        lin = (self.matvec_A(xh) + sh) / a
        if float(np.max(np.abs(lin), initial=0.0)) / alpha > tol:
            return None
        if self.cd.defect_primal(sh / a) / alpha > tol:
            return None
        return {"status": "dual_infeasible", "x": x, "s": s}


def new_cuts_for_unbounded(self):
    return False


def b_norm(b):
    return float(np.max(np.abs(b), initial=0.0))


def solve_by_cuts(P_data, P_indices, P_indptr, P_shape, q,
                  A_data, A_indices, A_indptr, A_shape, b,
                  cones, deadline, max_iter=40):
    try:
        prob = CutProblem(P_data, P_indices, P_indptr, P_shape, q,
                          A_data, A_indices, A_indptr, A_shape, b,
                          cones, deadline)
        if prob.cd is None:
            return {"status": "unresolved"}
        if prob.has_p and prob.n > 20000 and prob.cuts:
            return {"status": "unresolved"}
        if (prob.has_p and not prob.cuts and prob.nbase == prob.m
                and prob.n > 400):
            lo, hi, st, idx, val = prob._assemble_sparse()
            got = prob._sparse_active_qp(load_highs(), lo, hi, st, idx, val,
                                         lo.size)
            if got is not None:
                return got
        if not prob.has_p and not prob.cuts:
            # Pure LP: never form a dense m x n matrix, even for medium sizes.
            return prob.solve_sparse(4)
        return prob.solve(max_iter=max_iter)
    except Exception:
        return {"status": "unresolved"}
