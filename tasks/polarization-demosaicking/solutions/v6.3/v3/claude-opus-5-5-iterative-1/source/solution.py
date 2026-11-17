"""Polarization mosaic demosaicking (0/45/90/135).

Mosaic layout (row, col parity): (0,0)=90, (0,1)=45, (1,0)=135, (1,1)=0.
Output channel order: 0, 45, 90, 135.

NumPy only.  Most work happens in the FFT domain of the four 2x-subsampled
sub-lattices ("cosets"):
 0. A variance-stabilising transform (sensor noise variance ~ 0.027 * I) is
    applied first; everything below runs in that domain and is inverted at the
    end.
 1. Polarization fields D_c (smooth deviation of channel c from the common
    intensity image) predicted from the mosaic by learned linear filters.
 2. Common image J: the polarization-corrected mosaic raw - D_{channel at x}
    is Wiener-filtered; the white-noise floor is estimated per scene from the
    smoothed periodogram.  Estimate of channel c = J + D_c.
 3. Guided refinement: locally, channel c is fitted as a*J + b on its own
    samples (guided-filter / residual-interpolation style) and the estimate is
    moved half-way towards that fit.
 4. Safety gate: the four captures may be spatially misaligned.  The shift
    between channels is estimated from the phase slope of the demodulated
    sub-lattice cross-spectra; with growing misalignment the result is blended
    towards a learned cross-channel LMMSE estimate (aligned-trained filters,
    switching to filters trained on mildly misaligned captures as the shift
    grows) and then towards a learned own-channel-only LMMSE estimate
    (weights calibrated on real and synthetically misaligned scenes).
Observed samples are passed through unchanged.
"""
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_R = 5                       # cross / own filter half-width (full-res pixels)
_RD = 7                      # polarization-field filter half-width
_PAD = 8                     # even reflect padding keeps the mosaic phase
_PCH = ((1, 1), (0, 1), (0, 0), (1, 0))   # coset of channel 0, 45, 90, 135
_CH_AT = {p: c for c, p in enumerate(_PCH)}
_PHASES = ((0, 0), (0, 1), (1, 0), (1, 1))
_PSM = 8                     # half-width of periodogram smoothing (bins)
_GATE = (6.8, -10.0, 3.35, -1.87)   # blend weights vs. estimated misalignment
_GF_R, _GF_EPS, _GF_BETA = 2, 1.0, 0.6   # guided refinement
_ALPHA = 0.027               # sensor noise variance ~ _ALPHA * intensity
_S2_RANGE = (0.6, 2.0)       # noise floor in the stabilised domain (~1)


def _vst(v):
    """Variance-stabilising transform: unit noise variance for this sensor."""
    return 2.0 * np.sqrt(np.maximum(v, 0.0) / _ALPHA)


def _ivst(z):
    return _ALPHA * (np.maximum(z, 0.0) * 0.5) ** 2


def _jobs():
    jobs = []
    for kind in ("cross", "own"):
        for c in range(4):
            for p in _PHASES:
                if _PCH[c] != p:
                    jobs.append((kind, c, p))
    return jobs


_EXP_CACHE = {}


def _exp_table(k, lo, hi):
    key = (k.shape[0], float(k[1]) if k.shape[0] > 1 else 0.0, lo, hi)
    t = _EXP_CACHE.get(key)
    if t is None:
        m = np.arange(lo, hi + 1, dtype=np.float64)
        t = np.exp(2j * np.pi * k[:, None] * m[None, :])
        _EXP_CACHE[key] = t
    return t


def _spectra(w2, R, p, ky, kx):
    """Polyphase responses {input coset: (my, mx)} of a correlation filter
    out(x) = sum_d w2[d] in(x + d), output on coset p."""
    out = {}
    lo, hi = -(R // 2) - 1, R // 2 + 1
    ty = _exp_table(ky, lo, hi)
    tx = _exp_table(kx, lo, hi)
    for q in _PHASES:
        dys = [dy for dy in range(-R, R + 1) if (p[0] + dy) % 2 == q[0]]
        dxs = [dx for dx in range(-R, R + 1) if (p[1] + dx) % 2 == q[1]]
        sub = w2[np.ix_([dy + R for dy in dys], [dx + R for dx in dxs])]
        if not np.any(sub):
            continue
        my = [(p[0] + dy - q[0]) // 2 - lo for dy in dys]
        mx = [(p[1] + dx - q[1]) // 2 - lo for dx in dxs]
        out[q] = (ty[:, my] @ sub) @ tx[:, mx].T
    return out


def _apply(w, R, p, Y, ky, kx, npix):
    """Spectrum (coset p) of a full-res filter with bias w[-1]."""
    acc = None
    for q, h in _spectra(w[:-1].reshape(2 * R + 1, 2 * R + 1), R, p, ky, kx).items():
        acc = h * Y[q] if acc is None else acc + h * Y[q]
    acc[0, 0] += w[-1] * npix
    return acc


def _box_wrap0(a, r):
    """Box mean, circular along axis 0, renormalised at the ends of axis 1."""
    n0, n1 = a.shape
    ext = np.concatenate([a[n0 - r:], a, a[:r + 1]], axis=0)
    c = np.cumsum(ext, axis=0)
    s = (c[2 * r + 1:2 * r + 1 + n0] - c[:n0]) / (2 * r + 1)
    c = np.cumsum(s, axis=1)
    c = np.concatenate([np.zeros((n0, 1)), c], axis=1)
    hi = np.minimum(np.arange(n1) + r + 1, n1)
    lo = np.maximum(np.arange(n1) - r, 0)
    return (c[:, hi] - c[:, lo]) / (hi - lo)[None, :]


def _box_last2(a, r):
    """Box mean over the last two axes (window 2r+1) with edge renormalisation."""
    out = a
    for ax in (-2, -1):
        n = out.shape[ax]
        pad = [(0, 0)] * out.ndim
        pad[ax] = (r, r)
        z = np.pad(out, pad)
        s = None
        for k in range(2 * r + 1):
            sl = [slice(None)] * out.ndim
            sl[ax] = slice(k, k + n)
            s = z[tuple(sl)].copy() if s is None else s + z[tuple(sl)]
        idx = np.arange(n)
        cnt = (np.minimum(idx + r, n - 1) - np.maximum(idx - r, 0) + 1).astype(out.dtype)
        shape = [1] * out.ndim
        shape[ax] = n
        out = s / cnt.reshape(shape)
    return out


def _half_shift(a, dy, dx):
    """Bilinear value of a coset array at index offsets (dy/2, dx/2), dy,dx in {-1,0,1}."""
    if dy == 1:
        a = 0.5 * (a + np.concatenate([a[1:], a[-1:]], 0))
    elif dy == -1:
        a = 0.5 * (a + np.concatenate([a[:1], a[:-1]], 0))
    if dx == 1:
        a = 0.5 * (a + np.concatenate([a[:, 1:], a[:, -1:]], 1))
    elif dx == -1:
        a = 0.5 * (a + np.concatenate([a[:, :1], a[:, :-1]], 1))
    return a


def _misalignment(Y, ky, kx):
    """Statistics of the demodulated sub-lattice cross-spectra (all channel
    pairs).  Returns (largest phase-slope shift in full-resolution pixels,
    smallest phase coherence in a mid band).  Aligned captures give a small
    shift and a coherence close to 1."""
    rad = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    band = (rad >= 0.04) & (rad < 0.2)
    mid = ((rad >= 0.12) & (rad < 0.2))[band]
    kyb = np.broadcast_to(ky[:, None], rad.shape)[band]
    kxb = np.broadcast_to(kx[None, :], rad.shape)[band]
    Yb = [Y[_PCH[c]][band] for c in range(4)]
    A = np.stack([np.pi * kyb, np.pi * kxb], 1)
    shift = 0.0
    coh = 1.0
    for a in range(4):
        for b in range(a + 1, 4):
            dp = np.array(_PCH[a], dtype=np.float64) - np.array(_PCH[b], dtype=np.float64)
            r = Yb[a] * np.conj(Yb[b]) * np.exp(-1j * np.pi * (kyb * dp[0] + kxb * dp[1]))
            w = np.abs(r)
            den = w[mid].sum()
            coh = min(coh, float(np.real(r[mid]).sum() / den) if den > 0 else 0.0)
            G = (A * w[:, None]).T @ A + 1e-9 * np.eye(2)
            ph = np.angle(r)
            sol = np.linalg.solve(G, (A * w[:, None]).T @ ph)
            res = np.angle(np.exp(1j * (ph - A @ sol)))
            sol = sol + np.linalg.solve(G, (A * w[:, None]).T @ res)
            shift = max(shift, float(np.hypot(sol[0], sol[1])))
    return shift, coh


def _fallback(raw):
    """Bilinear demosaicking (used only if the main path fails)."""
    x = raw.astype(np.float64)
    H, W = x.shape
    out = np.empty((4, H, W))
    for c in range(4):
        r0, c0 = _PCH[c]
        m = np.zeros((H, W))
        m[r0::2, c0::2] = 1.0
        z = np.pad(x * m, 1, mode="reflect")
        mp = np.pad(m, 1, mode="reflect")

        def conv(a):
            a = 0.5 * a[:, :-2] + a[:, 1:-1] + 0.5 * a[:, 2:]
            return 0.5 * a[:-2, :] + a[1:-1, :] + 0.5 * a[2:, :]

        out[c] = conv(z) / np.maximum(conv(mp), 1e-6)
    return out


def _sig(z):
    return 1.0 / (1.0 + np.exp(-min(max(z, -50.0), 50.0)))


def _main(raw):
    x_lin = np.asarray(raw, dtype=np.float64)
    x = _vst(x_lin)
    H, W = x.shape
    h2, w2 = H // 2, W // 2
    xp = np.pad(x, _PAD, mode="reflect")
    Hp, Wp = xp.shape
    mpy, mpx = Hp // 2, Wp // 2
    ky = np.fft.fftfreq(mpy)
    kx = np.fft.rfftfreq(mpx)
    Y = {q: np.fft.rfft2(xp[q[0]::2, q[1]::2]) for q in _PHASES}
    o0 = _PAD // 2
    crop = (slice(o0, o0 + h2), slice(o0, o0 + w2))
    npix = float(mpy * mpx)

    # ---- misalignment gate ----
    shift, coh = _misalignment(Y, ky, kx)
    if not (np.isfinite(shift) and np.isfinite(coh)):
        shift, coh = 99.0, 0.0
    w_two = _sig(_GATE[0] + _GATE[1] * shift) * _sig(40.0 * (coh - 0.85))
    w_cross = (1.0 - w_two) * _sig(_GATE[2] + _GATE[3] * shift)
    w_own = max(0.0, 1.0 - w_two - w_cross)
    ws = [w if w >= 0.03 else 0.0 for w in (w_two, w_cross, w_own)]
    w_two, w_cross, w_own = (w / sum(ws) for w in ws)

    obs = [x[p[0]::2, p[1]::2] for p in _PCH]          # observed samples per channel
    out = np.empty((4, H, W), dtype=np.float32)
    for c in range(4):
        p = _PCH[c]
        out[c, p[0]::2, p[1]::2] = obs[c]
    targets = [(c, p) for c in range(4) for p in _PHASES if _PCH[c] != p]

    # ---- learned-filter fallbacks (cross / own), blended in the spectral domain ----
    rest = None
    if w_two < 1.0:
        wts = np.load(os.path.join(_HERE, "filters_v1t.npy"))
        # the cross-channel component moves from the aligned-trained filters to
        # filters trained on mildly misaligned captures as the shift grows
        mu = _sig(6.0 * (shift - 0.8))
        mu = 0.0 if mu < 0.02 else (1.0 if mu > 0.98 else mu)
        wrob = np.load(os.path.join(_HERE, "filters_rob.npy")) if (w_cross > 0.0 and mu > 0.0) else None
        rest = {}
        for i, (kind, c, p) in enumerate(_jobs()):
            if kind == "cross":
                if w_cross <= 0.0:
                    continue
                s = None
                if mu < 1.0:
                    s = (w_cross * (1.0 - mu)) * _apply(wts[i], _R, p, Y, ky, kx, npix)
                if wrob is not None:
                    r = (w_cross * mu) * _apply(wrob[i], _R, p, Y, ky, kx, npix)
                    s = r if s is None else s + r
            else:
                if w_own <= 0.0:
                    continue
                s = w_own * _apply(wts[i], _R, p, Y, ky, kx, npix)
            rest[(c, p)] = s if (c, p) not in rest else rest[(c, p)] + s

    if w_two > 0.0:
        # ---- polarization fields ----
        wd = np.load(os.path.join(_HERE, "filters_dt.npy"))
        Dsp = {}
        i = 0
        for c in range(4):
            for p in _PHASES:
                Dsp[(c, p)] = _apply(wd[i], _RD, p, Y, ky, kx, npix)
                i += 1

        # ---- common image: full-res spectrum of the corrected mosaic held as
        #      its four alias blocks u = k + a*M, Wiener-filtered ----
        eky = np.exp(-1j * np.pi * ky)[:, None]
        ekx = np.exp(-1j * np.pi * kx)[None, :]
        Mt = {}
        for q in _PHASES:
            m = Y[q] - Dsp[(_CH_AT[q], q)]
            if q[0]:
                m = m * eky
            if q[1]:
                m = m * ekx
            Mt[q] = m
        Ja = []
        for a in _PHASES:
            acc = None
            for q in _PHASES:
                s = -1.0 if (a[0] * q[0] + a[1] * q[1]) % 2 else 1.0
                acc = s * Mt[q] if acc is None else acc + s * Mt[q]
            Ja.append(acc)
        inv_n = 1.0 / float(Hp * Wp)
        Ps = [_box_wrap0((J.real ** 2 + J.imag ** 2) * inv_n, _PSM) for J in Ja]
        s2 = float(np.clip(np.percentile(np.concatenate([P.ravel() for P in Ps]), 25), *_S2_RANGE))
        for i in range(4):
            Ja[i] = Ja[i] * (np.maximum(Ps[i] - s2, 0.0) / np.maximum(Ps[i], 1e-12))

        two = np.empty((4, H, W), dtype=np.float32)
        for c in range(4):
            p = _PCH[c]
            two[c, p[0]::2, p[1]::2] = obs[c]
        for p in _PHASES:
            jp = None
            for ai, a in enumerate(_PHASES):
                s = -1.0 if (a[0] * p[0] + a[1] * p[1]) % 2 else 1.0
                jp = s * Ja[ai] if jp is None else jp + s * Ja[ai]
            jp = 0.25 * jp
            if p[0]:
                jp = jp * np.conj(eky)
            if p[1]:
                jp = jp * np.conj(ekx)
            for c in range(4):
                if _PCH[c] != p:
                    spec = w_two * (jp + Dsp[(c, p)])
                    if rest is not None and (c, p) in rest:
                        spec = spec + rest[(c, p)]
                    two[c, p[0]::2, p[1]::2] = np.fft.irfft2(spec, s=(mpy, mpx))[crop]

        # ---- guided refinement: channel c ~ a*J + b fitted on its own samples ----
        J = two.mean(0)
        G = np.stack([J[p[0]::2, p[1]::2] for p in _PCH])
        Dd = np.stack(obs).astype(np.float32)
        st = _box_last2(np.stack([G, Dd, G * Dd, G * G]), _GF_R)
        mg, md, mgd, mgg = st[0], st[1], st[2], st[3]
        A = (mgd - mg * md) / (mgg - mg * mg + np.float32(_GF_EPS))
        B = md - A * mg
        AB = _box_last2(np.stack([A, B]), _GF_R)
        beta = np.float32(_GF_BETA * w_two)
        for c, q in targets:
            p = _PCH[c]
            dy, dx = q[0] - p[0], q[1] - p[1]
            Aq = _half_shift(AB[0, c], dy, dx)
            Bq = _half_shift(AB[1, c], dy, dx)
            cur = two[c, q[0]::2, q[1]::2]
            out[c, q[0]::2, q[1]::2] = cur + beta * (Aq * J[q[0]::2, q[1]::2] + Bq - cur)
    else:
        tot = w_cross + w_own
        for c, q in targets:
            out[c, q[0]::2, q[1]::2] = np.fft.irfft2(rest[(c, q)], s=(mpy, mpx))[crop] / tot
    res = _ivst(out).astype(np.float32)
    for c in range(4):
        p = _PCH[c]
        res[c, p[0]::2, p[1]::2] = x_lin[p[0]::2, p[1]::2]
    return res


def reconstruct(raw):
    raw = np.asarray(raw)
    try:
        if raw.ndim != 2 or raw.shape[0] % 2 or raw.shape[1] % 2 or min(raw.shape) < 32:
            raise ValueError("unsupported shape")
        out = _main(raw)
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("non-finite output")
        return out
    except Exception:
        out = _fallback(raw).astype(np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=255.0, neginf=0.0)
