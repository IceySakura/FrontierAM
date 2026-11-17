#!/usr/bin/env python3
"""Black-box cubature on [0,1]^d under a hard function-evaluation budget.

Main engine: dimension-adaptive sparse grid quadrature built from nested
Clenshaw-Curtis rules (hierarchical difference weights), refined greedily by
the magnitude of each Smolyak term's contribution.

Safety net: if the sparse grid's own error indicator says the integrand is
not resolvable on a sparse grid (e.g. a genuinely high-dimensional ridge),
the remaining budget is handed to a scrambled Sobol net, whose error is
compared against the sparse-grid indicator before the answer is emitted.
"""
import json
import math
import sys
import time

import numpy as np

T0 = time.time()
# Hard wall-clock guard: stop querying and answer well before the 30 s limit.
T_LIMIT = 18.0
BATCH = 512
EPS = 1e-9          # inward clip so that no point sits exactly on a bound


def elapsed():
    return time.time() - T0


# --------------------------------------------------------------------------
# 1-D nested Clenshaw-Curtis hierarchy
# --------------------------------------------------------------------------
def cc_rule(n):
    """Clenshaw-Curtis rule with n intervals (n+1 nodes) on [0,1].

    Returns (x, w) with x_j = (1-cos(pi j/n))/2 ascending and sum(w) = 1.
    Weights are obtained by integrating the Chebyshev interpolant exactly.
    """
    j = np.arange(n + 1)
    ks = np.arange(0, n + 1, 2)                 # even Chebyshev modes only
    hk = np.full(len(ks), 0.5)
    hk[ks != 0] = 1.0
    if n % 2 == 0:                              # k = n term is halved too
        hk[ks == n] = 0.5
    ik = 2.0 / (1.0 - ks.astype(float) ** 2)    # integral of T_k over [-1,1]
    ck = hk * ik * (2.0 / n)
    gj = np.full(n + 1, 1.0)
    gj[0] = gj[n] = 0.5
    m = np.cos(np.pi * np.outer(ks, j) / n)
    w = gj * (m.T @ ck)                         # weights on [-1,1]
    x = 0.5 * (1.0 - np.cos(np.pi * j / n))
    return x, 0.5 * w


class CC1D:
    """Registry of 1-D nodes (deduplicated across levels) and rule data."""

    def __init__(self):
        self.key2id = {}
        self.xlist = []
        self.xarr = np.zeros(0)
        self.cache = {}

    def _reg(self, t, f):
        k = (t, f)
        i = self.key2id.get(k)
        if i is None:
            i = len(self.xlist)
            self.key2id[k] = i
            self.xlist.append(0.5 * (1.0 - math.cos(math.pi * t / (2.0 ** f))))
            self.xarr = np.array(self.xlist)
        return i

    @staticmethod
    def _key(level, j):
        if level == 1:
            return (1, 1)                       # the midpoint 0.5
        e = level - 1
        if j == 0:
            return (0, 0)                       # x = 0
        v = (j & -j).bit_length() - 1           # trailing zero bits
        return (j >> v, e - v)

    def level(self, l):
        """Return dict with m, x, w (rule weights), dw (hierarchical weights),
        ids (global 1-D node ids) for level l >= 1."""
        c = self.cache.get(l)
        if c is not None:
            return c
        if l == 1:
            ids = np.array([self._reg(1, 1)], dtype=np.int64)
            w = np.array([1.0])
            dw = np.array([1.0])
        else:
            n = 1 << (l - 1)
            x, w = cc_rule(n)
            dw = w.copy()
            if l == 2:
                dw[1] -= 1.0                    # parent = midpoint rule
            else:
                dw[0::2] -= self.level(l - 1)["w"]
            ids = np.array([self._reg(*self._key(l, j)) for j in range(n + 1)],
                           dtype=np.int64)
        c = {"m": len(w), "w": w, "dw": dw, "ids": ids}
        self.cache[l] = c
        return c


# --------------------------------------------------------------------------
# oracle channel
# --------------------------------------------------------------------------
class Oracle:
    def __init__(self, budget, lo, hi):
        self.budget = int(budget)
        self.used = 0
        self.lo = lo
        self.hi = hi
        self.span = hi - lo
        self.dead = False

    def ask(self, unit_pts):
        """Evaluate points given in unit-cube coordinates. Returns values."""
        out = []
        pts = self.lo + unit_pts * self.span
        pts = np.clip(pts, self.lo + EPS * self.span, self.hi - EPS * self.span)
        rows = pts.tolist()
        for s in range(0, len(rows), BATCH):
            chunk = rows[s:s + BATCH]
            if self.used + len(chunk) > self.budget:
                self.dead = True
                break
            sys.stdout.write(json.dumps({"type": "query", "points": chunk}) + "\n")
            sys.stdout.flush()
            line = sys.stdin.readline()
            if not line:
                self.dead = True
                break
            msg = json.loads(line)
            vals = np.asarray(msg.get("values", []), dtype=float)
            if vals.size != len(chunk):
                self.dead = True
                vals = np.resize(vals, len(chunk))
            out.append(vals)
            self.used += len(chunk)
            if not np.all(np.isfinite(vals)):
                vals = np.nan_to_num(vals, nan=0.0, posinf=1e300, neginf=-1e300)
        return np.concatenate(out) if out else np.zeros(0)

    def left(self):
        return self.budget - self.used


# --------------------------------------------------------------------------
# adaptive sparse grid
# --------------------------------------------------------------------------
class SparseGrid:
    def __init__(self, d, budget, oracle, cc, seed=0):
        self.d = d
        self.cap = int(budget)      # points this branch is allowed to spend
        self.budget = budget
        self.orc = oracle
        self.cc = cc
        self.rng = np.random.default_rng(seed)
        self.P = np.zeros((budget, d))
        self.V = np.zeros(budget)
        self.W = np.zeros(budget)
        self.n = 0
        self.gid = {}
        self.terms = {}          # multi-index -> (gids, weights)
        self.C = {}              # multi-index -> contribution
        self.refined = set()
        self.pending = []        # global ids awaiting evaluation

    # ---- geometry -------------------------------------------------------
    def _grid(self, l):
        d = self.d
        rows = np.zeros((1, d), dtype=np.int64)
        wts = np.ones(1)
        for j in range(d):
            lv = self.cc.level(l[j])
            ids_j = lv["ids"]
            dw_j = lv["dw"]
            m = len(ids_j)
            nprev = rows.shape[0]
            rows = np.repeat(rows, m, axis=0)
            rows[:, j] = np.tile(ids_j, nprev)
            wts = np.repeat(wts, m) * np.tile(dw_j, nprev)
        return rows, wts

    def add(self, l, commit=True):
        """Add Smolyak term l. Returns number of new points (or -1 if it does
        not fit in the budget)."""
        rows, wts = self._grid(l)
        uniq, inv = np.unique(rows, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        wu = np.empty(len(uniq))
        wu[inv] = wts
        gids = np.empty(len(uniq), dtype=np.int64)
        fresh = []
        for i in range(len(uniq)):
            k = tuple(int(v) for v in uniq[i])
            g = self.gid.get(k)
            if g is None:
                g = self.n + len(fresh)
                fresh.append(k)
            gids[i] = g
        if self.n + len(fresh) > self.cap or self.n + len(fresh) > self.budget:
            return -1
        if not commit:
            return len(fresh)
        xs = self.cc.xarr
        for k in fresh:
            g = self.n
            self.gid[k] = g
            self.P[g] = xs[list(k)]
            self.n += 1
            self.pending.append(g)
        self.W[gids] += wu
        self.terms[l] = (gids, wu)
        self.C[l] = float(wu @ self.V[gids]) if len(gids) else 0.0
        return len(fresh)

    def flush(self):
        """Evaluate every pending point."""
        if not self.pending or self.orc.left() <= 0:
            self.pending = []
            return
        ids = np.array(self.pending, dtype=np.int64)
        self.pending = []
        vals = self.orc.ask(self.P[ids])
        if len(vals) != len(ids):
            vals = np.resize(vals, len(ids))
        self.V[ids] = vals
        self.vals_seen = True

    def refresh(self):
        for l, (g, w) in self.terms.items():
            self.C[l] = float(w @ self.V[g])

    def estimate(self):
        return float(self.W[:self.n] @ self.V[:self.n])

    def err_proxy(self):
        """Sum of |contributions| over the outermost terms: asymptotic error
        indicator of the sparse-grid expansion."""
        tot = 0.0
        for l in self.terms:
            if any(tuple(np.array(l) + np.eye(self.d, dtype=int)[j]) in self.terms
                   for j in range(self.d)):
                continue
            tot += abs(self.C.get(l, 0.0))
        return tot

    # ---- construction ---------------------------------------------------
    def seed_grid(self, surplus):
        """Add every multi-index with sum(l) <= d + surplus (budget allowing)."""
        one = (1,) * self.d
        if self.add(one) < 0:
            return
        frontier = [one]
        while frontier:
            nxt = []
            for l in frontier:
                for j in range(self.d):
                    if sum(l) - self.d >= surplus:
                        continue
                    c = list(l)
                    c[j] += 1
                    c = tuple(c)
                    if c in self.terms:
                        continue
                    if self.add(c) >= 0:
                        nxt.append(c)
            frontier = nxt
            if self.orc.left() <= 0:
                break

    def refine_round(self, target):
        """Greedy refinement until `target` new points are queued."""
        guarded = 0
        while len(self.pending) < target and self.orc.left() - len(self.pending) > 0:
            if elapsed() > T_LIMIT:
                return
            self.refresh()
            best = None
            bestv = -1.0
            for l in self.terms:
                if l in self.refined:
                    continue
                v = abs(self.C.get(l, 0.0))
                if v > bestv:
                    bestv = v
                    best = l
            if best is None:
                return
            self.refined.add(best)
            added = False
            for j in range(self.d):
                c = list(best)
                c[j] += 1
                c = tuple(c)
                if c in self.terms:
                    continue
                if self.add(c) >= 0:
                    added = True
            if not added:
                # nothing affordable anywhere: mark and continue
                guarded += 1
                if guarded > 4 * self.d + 8:
                    return


def sobol_branch(orc, d, alloc, seed):
    """Pooled scrambled-Sobol estimate using up to `alloc` points.

    Points are drawn as a few independent power-of-two scrambled nets and
    pooled, so no allocation is wasted on rounding.  Returns
    (mean, sd/sqrt(n)); (None, None) when nothing could be sampled.
    """
    if alloc < 16:
        return None, None
    alloc = int(min(alloc, orc.left()))
    vals = []
    drawn = 0
    k = 0
    while alloc - drawn >= 16 and orc.left() > 0 and k < 8:
        n = 2 ** int(math.floor(math.log2(min(alloc - drawn, orc.left()))))
        pts = None
        try:
            from scipy.stats import qmc
            pts = qmc.Sobol(d=d, scramble=True,
                            seed=(seed + 7919 * (k + 1)) % 99991).random(n)
        except Exception:
            pts = None
        if pts is None:
            pts = np.random.default_rng(seed + 104729 + k).random((n, d))
        pts = np.asarray(pts, dtype=float)[:n]
        if pts.shape[0] == 0:
            break
        v = orc.ask(pts)
        if len(v) == 0:
            break
        v = np.nan_to_num(np.asarray(v, dtype=float))
        vals.append(v)
        drawn += len(v)
        k += 1
        if elapsed() > T_LIMIT:
            break
    if not vals:
        return None, None
    v = np.concatenate(vals)
    if v.size == 0:
        return None, None
    bar = float(np.std(v) / math.sqrt(v.size))
    if len(vals) >= 2:
        # spread between independent scrambled nets: an honest indicator that
        # also reacts to features a single net systematically misses
        ms = np.array([float(np.mean(x)) for x in vals])
        ns = np.array([float(len(x)) for x in vals])
        spread = float(np.std(ms, ddof=1)) if len(ms) > 1 else 0.0
        bar = max(bar, spread * float(np.median(ns)) / float(v.size))
    return float(np.mean(v)), bar


def main():
    line = sys.stdin.readline()
    if not line:
        return
    init = json.loads(line)
    d = int(init["dimension"])
    budget = int(init.get("budget", 2048))
    seed = int(init.get("seed", 0) or 0)
    bnds = init.get("bounds") or [[0.0, 1.0]] * d
    arr = np.asarray(bnds, dtype=float).reshape(-1, 2)
    lo, hi = arr[:, 0].copy(), arr[:, 1].copy()

    orc = Oracle(budget, lo, hi)
    cc = CC1D()

    # budget split between the two branches
    if d <= 2:
        frac = 0.80
    elif d == 3:
        frac = 0.78
    elif d == 4:
        frac = 0.70
    elif d == 5:
        frac = 0.60
    else:
        frac = 0.55
    cap = max(64, int(budget * frac))

    sg = SparseGrid(d, budget, orc, cc, seed)
    sg.cap = cap

    asg_est = None
    asg_err = float("inf")
    try:
        sg.seed_grid(2)
        sg.flush()
        if sg.n:
            asg_est = sg.estimate()
        while orc.left() > 0 and sg.n < cap and elapsed() < T_LIMIT:
            room = min(cap - sg.n, orc.left())
            tgt = min(BATCH, room)
            if tgt <= 0:
                break
            before = sg.n + len(sg.pending)
            sg.refine_round(tgt)
            if sg.n + len(sg.pending) == before:
                break
            sg.flush()
            if sg.n:
                asg_est = sg.estimate()
        sg.refresh()
        if sg.n:
            asg_est = sg.estimate()
            asg_err = sg.err_proxy()
    except Exception:
        try:
            sg.refresh()
            if sg.n:
                asg_est = sg.estimate()
                asg_err = sg.err_proxy()
        except Exception:
            pass

    # hand whatever is left (including any slack the sparse grid could not use)
    # to the QMC branch
    qmc_est, qmc_err = (None, None)
    if orc.left() > 0 and elapsed() < T_LIMIT - 1.0:
        try:
            qmc_est, qmc_err = sobol_branch(orc, d, orc.left(), seed)
        except Exception:
            qmc_est, qmc_err = (None, None)

    answer = asg_est
    if qmc_est is not None and np.isfinite(qmc_est):
        if answer is None or not np.isfinite(answer):
            answer = qmc_est
        else:
            diff = abs(asg_est - qmc_est)
            # Switch only when the sparse grid's own indicator is clearly worse
            # than the QMC error bar *and* the two answers actually disagree
            # (a conservative guard: the sparse-grid indicator is known to
            # over-estimate on non-smooth integrands).
            if qmc_err is not None and qmc_err < 0.15 * asg_err and diff > qmc_err:
                answer = qmc_est
    if answer is None or not np.isfinite(answer):
        v = sg.V[:sg.n] if sg.n else np.zeros(0)
        if v.size and np.isfinite(np.mean(v)):
            answer = float(np.mean(v))
        else:
            # last resort: a few points so the answer is never a fixed constant
            try:
                n = max(1, min(8, orc.left()))
                pts = np.full((n, d), 0.5)
                pts[1:] = np.random.default_rng(seed).random((n - 1, d))
                vv = orc.ask(pts)
                vv = np.nan_to_num(np.asarray(vv, dtype=float))
                answer = float(np.mean(vv)) if vv.size else 0.0
            except Exception:
                answer = 0.0

    sys.stdout.write(json.dumps({"type": "result", "estimate": float(answer)}) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
