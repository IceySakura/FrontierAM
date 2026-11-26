"""One-dimensional machinery: Gauss rules for the standard normal weight on an interval, and the batched
adaptive integration of piecewise smooth functions along many lines at once.

Every line is parametrised by s in [-T, T] with weight phi(s) (standard normal density); the caller maps s to a
point of R^d (parallel lines: z = y + s v; rays: z = rho(s) theta with rho the chi_d quantile of Phi(s)).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr

SQ2PI = math.sqrt(2.0 * math.pi)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(96)


def phi(s):
    return np.exp(-0.5 * s * s) / SQ2PI


def gauss_rules(a, b, n):
    """n-point Gauss rules for the weight phi on [a_i, b_i] (arrays of length P). Returns nodes, weights (P, n).

    Discretised Stieltjes procedure on a 96-point Gauss-Legendre rule (exact for these weights to rounding)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    P = len(a)
    if P == 0:
        return np.zeros((0, n)), np.zeros((0, n))
    half = 0.5 * (b - a)
    mid = 0.5 * (b + a)
    x = mid[:, None] + half[:, None] * _GL_X[None, :]
    w = half[:, None] * _GL_W[None, :] * phi(x)
    mass = w.sum(1)
    tiny = mass <= 1e-300
    if n == 1:
        node = np.where(tiny, mid, (w * x).sum(1) / np.where(tiny, 1, mass))
        return node[:, None], mass[:, None]
    # Stieltjes in the scaled variable xi = (x - mid)/half for conditioning
    xi = np.broadcast_to(_GL_X[None, :], x.shape)
    wn = w / np.where(tiny, 1.0, mass)[:, None]
    wn[tiny] = _GL_W / 2
    alpha = np.zeros((P, n))
    beta = np.zeros((P, n))
    p_prev = np.zeros_like(xi)
    p = np.ones_like(xi)
    nrm_prev = np.ones(P)
    for k in range(n):
        nrm = (wn * p * p).sum(1)
        nrm = np.maximum(nrm, 1e-300)
        alpha[:, k] = (wn * xi * p * p).sum(1) / nrm
        if k > 0:
            beta[:, k] = nrm / nrm_prev
        p_new = (xi - alpha[:, k][:, None]) * p - (beta[:, k][:, None] * p_prev if k > 0 else 0.0)
        # rescale to avoid underflow
        sc = np.sqrt(np.maximum((wn * p_new * p_new).sum(1), 1e-300))
        p_prev = p / sc[:, None]
        p = p_new / sc[:, None]
        nrm_prev = nrm / (sc * sc)
    J = np.zeros((P, n, n))
    idx = np.arange(n)
    J[:, idx, idx] = alpha
    off = np.sqrt(np.maximum(beta[:, 1:], 0))
    J[:, idx[:-1], idx[1:]] = off
    J[:, idx[1:], idx[:-1]] = off
    ev, evec = np.linalg.eigh(J)
    nodes = mid[:, None] + half[:, None] * ev
    weights = mass[:, None] * evec[:, 0, :] ** 2
    return nodes, weights


def _lin_extrap(t1, f1, t2, f2, t):
    """Value at t of the line through (t1, f1), (t2, f2)."""
    den = t2 - t1
    den = np.where(np.abs(den) < 1e-300, 1e-300, den)
    return f1 + (f2 - f1) * (t - t1) / den


def _quad_extrap(t0, f0, t1, f1, t2, f2, t):
    """Value at t of the quadratic through three points (Lagrange form)."""
    l0 = (t - t1) * (t - t2) / ((t0 - t1) * (t0 - t2))
    l1 = (t - t0) * (t - t2) / ((t1 - t0) * (t1 - t2))
    l2 = (t - t0) * (t - t1) / ((t2 - t0) * (t2 - t1))
    return f0 * l0 + f1 * l1 + f2 * l2


class LineIntegrator:
    """Integrates F_l(s) = f(point_l(s)) against phi(s) ds on [-T, T] for many lines l at once.

    evaluate(l_idx, s) -> values: callback evaluating f at the points of lines l_idx with parameters s (one query
    batch). The integrator works in rounds so that every round is a single batch."""

    def __init__(self, evaluate, T=6.5, grid=None, eps_line=1e-9, max_ref=40, flag_factor=4.0,
                 nodes_per_unit=1.6, min_nodes=3, max_nodes=22, kink_factor=8.0, log=None):
        self.kink_factor = kink_factor
        self.evaluate = evaluate
        self.T = T
        self.grid = np.asarray(grid, float) if grid is not None else np.linspace(-T, T, 33)
        self.eps_line = eps_line
        self.max_ref = max_ref
        self.flag_factor = flag_factor
        self.nodes_per_unit = nodes_per_unit
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.log = log
        self.used = 0

    # ---------------------------------------------------------------- detection
    def _flag(self, G, Fv, scale):
        """Boolean (L, n-1): grid intervals that look non-smooth. ENO-type test: an interval is smooth if at least
        one 5-point stencil containing it has a small fourth difference relative to the first differences of the
        stencil (scale invariant; catches jumps, kinks and narrow pulses)."""
        L, n = Fv.shape
        h = np.diff(G)
        hm = np.mean(h)
        dd = Fv.copy()
        for k in range(1, 5):   # divided differences of order k
            dd = np.diff(dd, axis=1) / (G[k:] - G[:-k])[None, :]
        d4 = np.abs(dd) * hm ** 4                                   # (L, n-4), stencil j..j+4
        big = np.full((L, 3), np.inf)
        ext = np.concatenate([big, d4, big], axis=1)                # stencil k -> column k+3
        cols = np.arange(n - 1)
        score = ext[:, cols]
        for q in range(1, 4):
            score = np.minimum(score, ext[:, cols + q])
        # local smooth level of gap i: the smaller of the nearest stencils on each side that do not contain it
        # (k = i-4 and k = i+1, columns i-1 and i+4); robust to zero regions, growth and a second breakpoint nearby
        zero = np.zeros((L, 1))
        ext0 = np.concatenate([np.full((L, 3), np.inf), d4, np.full((L, 3), np.inf)], axis=1)
        lv = np.concatenate([np.full((L, 1), np.inf), ext0[:, :-1]], axis=1)[:, cols]     # column i-1
        rv = ext0[:, np.clip(cols + 4, 0, ext0.shape[1] - 1)]
        local = np.minimum(lv, rv)
        local = np.where(np.isfinite(local), local, np.median(d4, axis=1, keepdims=True))
        thr = np.maximum(self.flag_factor * local, 1e-9 * scale)
        weight = np.maximum(phi(G[:-1]), phi(G[1:]))[None, :] * h[None, :]
        flag = (score > thr) & (score * weight > 0.1 * self.eps_line)
        # kinks sitting (almost) on a grid point: one-sided derivative jump at interior points j = 2..n-3
        t0, t1, t2 = G[:-4], G[1:-3], G[2:-2]
        u1, u2 = G[3:-1], G[4:]
        fa, fb, fc, fd, fe = Fv[:, :-4], Fv[:, 1:-3], Fv[:, 2:-2], Fv[:, 3:-1], Fv[:, 4:]
        # derivative at t2 of the quadratic through (t0,t1,t2) and through (t2,u1,u2)
        dL = fa * (t2 - t1) / ((t0 - t1) * (t0 - t2)) + fb * (t2 - t0) / ((t1 - t0) * (t1 - t2)) \
            + fc * (2 * t2 - t0 - t1) / ((t2 - t0) * (t2 - t1))
        dR = fc * (2 * t2 - u1 - u2) / ((t2 - u1) * (t2 - u2)) + fd * (t2 - u2) / ((u1 - t2) * (u1 - u2)) \
            + fe * (t2 - u1) / ((u2 - t2) * (u2 - u1))
        dj = np.abs(dL - dR) * hm                                   # (L, n-4), point j+2
        djl = np.concatenate([np.full((L, 3), np.inf), dj[:, :-3]], axis=1)
        djr = np.concatenate([dj[:, 3:], np.full((L, 3), np.inf)], axis=1)
        locd = np.minimum(djl, djr)
        locd = np.where(np.isfinite(locd), locd, np.median(dj, axis=1, keepdims=True))
        w2 = phi(t2)[None, :] * hm
        kink = (dj > np.maximum(self.kink_factor * locd, 1e-9 * scale)) & (dj * hm * w2 > 0.1 * self.eps_line)
        idx = np.arange(2, n - 2)
        # a kink within a small distance eps of a grid point costs only ~ jump(slope) eps^2 if the piece is cut
        # exactly at the grid point, so such points become cuts unless a bracket lies within two intervals
        near = np.zeros_like(kink)
        for q in (-2, -1, 0, 1):
            near |= flag[:, np.clip(idx + q, 0, n - 2)]
        kink &= ~near
        self._gridcuts = [G[idx[np.nonzero(kink[l])[0]]].tolist() for l in range(L)]
        return flag

    def _brackets(self, G, Fv, flag):
        """One bracket per flagged interval; branch stencils never cross another flagged interval."""
        L, n = Fv.shape
        out = []
        for l, j in np.argwhere(flag):
            left = [(G[j], Fv[l, j])]
            if j >= 1 and not flag[l, j - 1]:
                left.insert(0, (G[j - 1], Fv[l, j - 1]))
            else:
                left.insert(0, (G[j] - 1e-3, Fv[l, j]))
            right = [(G[j + 1], Fv[l, j + 1])]
            if j + 2 <= n - 1 and not flag[l, j + 1]:
                right.append((G[j + 2], Fv[l, j + 2]))
            else:
                right.append((G[j + 1] + 1e-3, Fv[l, j + 1]))
            out.append({'line': int(l), 'a': G[j], 'b': G[j + 1], 'left': left, 'right': right})
        return out

    # ---------------------------------------------------------------- refinement
    def _refine(self, brackets, scale):
        """brackets: list of dicts with line, a, b, left [(t,f)...] sorted ascending (nearest last), right [(t,f)...]
        sorted ascending (nearest first). Returns the breakpoint estimates and records all evaluations."""
        active = list(range(len(brackets)))
        step = 0
        while active and step < self.max_ref:
            props = []
            keep = []
            for i in active:
                br = brackets[i]
                a, b = br['a'], br['b']
                (ta1, fa1), (ta0, fa0) = br['left'][-2], br['left'][-1]
                (tb0, fb0), (tb1, fb1) = br['right'][0], br['right'][1]
                c = 0.5 * (a + b)
                pl = _lin_extrap(ta1, fa1, ta0, fa0, c)
                pr = _lin_extrap(tb0, fb0, tb1, fb1, c)
                D = max(abs(pl - pr), abs(fa0 - fb0))
                w = b - a
                wmax = max(phi(a), phi(b)) if not (a < 0 < b) else phi(0.0)
                if D * w * wmax < self.eps_line or w < 1e-12 * max(1.0, abs(c)):
                    br['done'] = True
                    continue
                # proposal: intersection of the two branch lines on odd steps (kinks), else bisection
                m = c
                if False:
                    sl = (fa0 - fa1) / (ta0 - ta1) if ta0 != ta1 else 0.0
                    sr = (fb1 - fb0) / (tb1 - tb0) if tb1 != tb0 else 0.0
                    if abs(sl - sr) > 1e-14:
                        x = (fb0 - sr * tb0 - fa0 + sl * ta0) / (sl - sr)
                        if a < x < b:
                            m = min(max(x, a + 0.02 * w), b - 0.02 * w)
                br['nstep'] = br.get('nstep', 0) + 1
                br['m'] = m
                br['pl'] = _lin_extrap(ta1, fa1, ta0, fa0, m)
                br['pr'] = _lin_extrap(tb0, fb0, tb1, fb1, m)
                props.append(i)
            if not props:
                break
            lines = np.array([brackets[i]['line'] for i in props])
            ss = np.array([brackets[i]['m'] for i in props])
            vals = self.evaluate(lines, ss)
            self.used += len(props)
            for i, fm in zip(props, vals):
                br = brackets[i]
                m = br['m']
                br.setdefault('evals', []).append((m, fm))
                if abs(fm - br['pl']) <= abs(fm - br['pr']):
                    br['a'] = m
                    br['left'].append((m, fm))
                else:
                    br['b'] = m
                    br['right'].insert(0, (m, fm))
                keep.append(i)
            active = [i for i in keep if not brackets[i].get('done')]
            step += 1
        out = []
        for br in brackets:
            a, b = br['a'], br['b']
            # final location: intersection of branch lines if inside (kink), else midpoint
            (ta1, fa1), (ta0, fa0) = br['left'][-2], br['left'][-1]
            (tb0, fb0), (tb1, fb1) = br['right'][0], br['right'][1]
            sl = (fa0 - fa1) / (ta0 - ta1) if ta0 != ta1 else 0.0
            sr = (fb1 - fb0) / (tb1 - tb0) if tb1 != tb0 else 0.0
            loc = 0.5 * (a + b)
            if abs(sl - sr) > 1e-14:
                x = (fb0 - sr * tb0 - fa0 + sl * ta0) / (sl - sr)
                if a <= x <= b:
                    loc = x
            out.append(loc)
        return out

    # ---------------------------------------------------------------- main
    def run(self, L, scale):
        G = self.grid
        n = len(G)
        lines = np.repeat(np.arange(L), n)
        ss = np.tile(G, L)
        Fv = self.evaluate(lines, ss).reshape(L, n)
        self.used += L * n
        flag = self._flag(G, Fv, scale)
        brackets = self._brackets(G, Fv, flag)
        bps = self._refine(brackets, scale) if brackets else []
        cuts = [[] for _ in range(L)]
        for br, loc in zip(brackets, bps):
            cuts[br['line']].append(loc)
        for l in range(L):
            cuts[l].extend(self._gridcuts[l])
        # pieces and Gauss nodes
        T = self.T
        pa, pb, pl = [], [], []
        for l in range(L):
            c = sorted(cuts[l])
            edges = [-T] + [x for x in c if -T < x < T] + [T]
            for k in range(len(edges) - 1):
                if edges[k + 1] > edges[k]:
                    pa.append(edges[k]); pb.append(edges[k + 1]); pl.append(l)
        pa, pb, pl = np.array(pa), np.array(pb), np.array(pl, int)
        # number of nodes per piece from its effective length (Gaussian-weighted)
        core = np.clip(np.minimum(pb, 4.5) - np.maximum(pa, -4.5), 0, None)
        tail = (pb - pa) - core
        npn = np.clip(np.ceil(self.min_nodes + self.nodes_per_unit * (core + 0.35 * tail)), self.min_nodes,
                      self.max_nodes).astype(int)
        mass = ndtr(pb) - ndtr(pa)
        npn = np.where(mass < 1e-7, np.minimum(npn, 2), npn)
        npn = np.where(mass < 1e-11, 1, npn)
        # pieces on which every observed value is the same constant (e.g. zero regions of payoffs) are integrated
        # exactly without new nodes when the piece contains at least 4 grid points
        const_val = np.full(len(pa), np.nan)
        for q in range(len(pa)):
            l = pl[q]
            inside = (G > pa[q]) & (G < pb[q])
            if inside.sum() >= 4:
                vals_in = Fv[l, inside]
                if np.all(vals_in == vals_in[0]):
                    const_val[q] = vals_in[0]
        isconst = np.isfinite(const_val)
        npn = np.where(isconst, 0, npn)
        result = np.zeros(L)
        moment = np.zeros(L)
        all_lines, all_s, all_w, all_piece = [], [], [], []
        if isconst.any():
            np.add.at(result, pl[isconst], const_val[isconst] * mass[isconst])
            mid1 = phi(pa[isconst]) - phi(pb[isconst])          # int_a^b s phi(s) ds
            np.add.at(moment, pl[isconst], const_val[isconst] * mid1)
        for k in np.unique(npn):
            if k == 0:
                continue
            sel = np.nonzero(npn == k)[0]
            nodes, weights = gauss_rules(pa[sel], pb[sel], int(k))
            all_lines.append(np.repeat(pl[sel], k)); all_s.append(nodes.ravel()); all_w.append(weights.ravel())
        if all_lines:
            ll = np.concatenate(all_lines); s2 = np.concatenate(all_s); w2 = np.concatenate(all_w)
            vals = self.evaluate(ll, s2)
            self.used += len(ll)
            np.add.at(result, ll, w2 * vals)
            np.add.at(moment, ll, w2 * s2 * vals)
        self.ncuts = sum(len(c) for c in cuts)
        self.cuts = cuts
        self.moment = moment
        return result


class WindowIntegrator:
    """Lines on which f vanishes outside one interval (s_lo, s_hi) and is smooth inside (payoffs along a monotone
    direction: calls, digitals, knock-outs). Coarse grid -> bisection of both ends on the predicate f != 0 ->
    Gauss rules for the normal weight on the window (or exact mass times the constant for constant payoffs).
    Lines whose grid shows several windows are returned in self.irregular for a generic treatment."""

    def __init__(self, evaluate, grid, tol=2e-5, nodes_per_unit=2.5, min_nodes=4, max_nodes=20):
        self.evaluate = evaluate
        self.grid = np.asarray(grid, float)
        self.tol = tol
        self.nodes_per_unit = nodes_per_unit
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.used = 0

    def run(self, L, scale):
        G = self.grid
        n = len(G)
        Fv = self.evaluate(np.repeat(np.arange(L), n), np.tile(G, L)).reshape(L, n)
        self.used += L * n
        nz = Fv != 0
        result = np.zeros(L)
        moment = np.zeros(L)
        self.cuts = [[] for _ in range(L)]
        self.irregular = []
        # brackets: lo end in (G[i-1], G[i]) with G[i] first nonzero; hi end in (G[j], G[j+1]) with G[j] last nonzero
        lo_a, lo_b, hi_a, hi_b, lines = [], [], [], [], []
        T = 7.5
        for l in range(L):
            idx = np.nonzero(nz[l])[0]
            if len(idx) == 0:
                continue
            i, j = idx[0], idx[-1]
            if j - i + 1 != len(idx):
                self.irregular.append(l)
                continue
            lines.append(l)
            lo_a.append(G[i - 1] if i > 0 else -T)
            lo_b.append(G[i])
            hi_a.append(G[j])
            hi_b.append(G[j + 1] if j < n - 1 else T)
        lines = np.array(lines, int)
        self.ncuts = 0
        self.moment = moment
        if len(lines) == 0:
            return result
        lo_a, lo_b, hi_a, hi_b = map(np.array, (lo_a, lo_b, hi_a, hi_b))
        lo_open = lo_a <= -T      # window reaches the left end of the domain
        hi_open = hi_b >= T
        # bisection on both ends simultaneously
        for _ in range(40):
            act_lo = (~lo_open) & (lo_b - lo_a > self.tol)
            act_hi = (~hi_open) & (hi_b - hi_a > self.tol)
            if not act_lo.any() and not act_hi.any():
                break
            ql = lines[act_lo]
            sl = 0.5 * (lo_a[act_lo] + lo_b[act_lo])
            qh = lines[act_hi]
            sh = 0.5 * (hi_a[act_hi] + hi_b[act_hi])
            vals = self.evaluate(np.concatenate([ql, qh]), np.concatenate([sl, sh]))
            self.used += len(vals)
            vl, vh = vals[:len(ql)], vals[len(ql):]
            ia = np.nonzero(act_lo)[0]
            lo_b[ia] = np.where(vl != 0, sl, lo_b[ia])
            lo_a[ia] = np.where(vl != 0, lo_a[ia], sl)
            ib = np.nonzero(act_hi)[0]
            hi_a[ib] = np.where(vh != 0, sh, hi_a[ib])
            hi_b[ib] = np.where(vh != 0, hi_b[ib], sh)
        s_lo = np.where(lo_open, -T, 0.5 * (lo_a + lo_b))
        s_hi = np.where(hi_open, T, 0.5 * (hi_a + hi_b))
        for k, l in enumerate(lines):
            self.cuts[l] = ([] if lo_open[k] else [s_lo[k]]) + ([] if hi_open[k] else [s_hi[k]])
        # constant payoff on the window?
        const = np.zeros(len(lines), bool)
        cval = np.zeros(len(lines))
        for k, l in enumerate(lines):
            v = Fv[l][nz[l]]
            if len(v) >= 2 and np.all(v == v[0]):
                const[k] = True
                cval[k] = v[0]
        mass = ndtr(s_hi) - ndtr(s_lo)
        m1 = phi(s_lo) - phi(s_hi)
        result[lines[const]] = cval[const] * mass[const]
        moment[lines[const]] = cval[const] * m1[const]
        rest = np.nonzero(~const)[0]
        if len(rest):
            a, b = s_lo[rest], s_hi[rest]
            core = np.clip(np.minimum(b, 4.5) - np.maximum(a, -4.5), 0, None)
            tail = (b - a) - core
            npn = np.clip(np.ceil(self.min_nodes + self.nodes_per_unit * (core + 0.35 * tail)), self.min_nodes,
                          self.max_nodes).astype(int)
            npn = np.where(mass[rest] < 1e-12, 1, npn)
            ll, ss, ww = [], [], []
            for k in np.unique(npn):
                sel = np.nonzero(npn == k)[0]
                nodes, weights = gauss_rules(a[sel], b[sel], int(k))
                ll.append(np.repeat(lines[rest][sel], k)); ss.append(nodes.ravel()); ww.append(weights.ravel())
            ll = np.concatenate(ll); ss = np.concatenate(ss); ww = np.concatenate(ww)
            vals = self.evaluate(ll, ss)
            self.used += len(ll)
            np.add.at(result, ll, ww * vals)
            np.add.at(moment, ll, ww * ss * vals)
        self.ncuts = sum(len(c) for c in self.cuts)
        self.moment = moment
        return result
