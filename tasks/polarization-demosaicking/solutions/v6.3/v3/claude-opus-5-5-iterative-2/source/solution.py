"""Polarization demosaicking with a scene-adaptive (shift-aware) Wiener filter.

See README.md for the method.  Only NumPy is used at run time.
"""
import os
import numpy as np

# channel order of the output: 0, 45, 90, 135 degrees.
# lattice offset (row, col) of each channel inside the 2x2 mosaic period
OFF = ((1, 1), (0, 1), (0, 0), (1, 0))
PH2C = {(0, 0): 2, (0, 1): 1, (1, 0): 3, (1, 1): 0}
R_WIN = 4          # half size of the (2R+1)^2 support used for the model search
R_FINAL = 5        # half size of the support of the final demosaicking filters (noisy scenes)
BIG_WIN_MIN_F = 0.7   # use R_FINAL only when the selected white-noise level is at least this
LAM = 1e-4         # ridge (relative to the mean diagonal)
MAX_SHIFT = 12.0   # sanity limit for the estimated inter-channel shifts (pixels)
SEARCH_MAX_SPREAD = 3   # skip the model search when channels are misaligned by more than this (pixels)

_here = os.path.dirname(os.path.abspath(__file__))
_tab = np.load(os.path.join(_here, 'wtab.npz'))
CTR = _tab['C']        # (S, 4, 4, 2T+1, 2T+1) cross-covariances of aligned training scenes
DTR = _tab['d']        # (S, 4, 2) shifts of the training scenes (same estimator as below)
T = (CTR.shape[-1] - 1) // 2
NSUM = float(_tab['n'].sum()) if 'n' in _tab.files else 0.0
_tabb = np.load(os.path.join(_here, 'wtab_bayer.npz'))
CTR_B = _tabb['C']     # tables of the training scenes after a simulated Bayer capture + demosaicking
DTR_B = _tabb['d']


# ----------------------------------------------------------------------------- shifts
def _phase_slope(Cs, Y, X, t0, iters=4):
    t = np.array(t0, float)
    D = np.stack([Y, X], 1)
    for _ in range(iters):
        Cr = Cs * np.exp(-1j * (Y * t[0] + X * t[1]))
        ph = np.angle(Cr)
        wt = np.abs(Cr)
        Wd = D * wt[:, None]
        t = t + np.linalg.solve(D.T @ Wd, Wd.T @ ph)
    return t


def mosaic_shifts(M, fmax=0.5):
    """d[c] = d_c - d_90 (row, col) in full-res pixels, model I_c(x) = A(x - d_c)."""
    ny, nx = M.shape[0] // 2, M.shape[1] // 2
    w = np.outer(np.hanning(ny), np.hanning(nx))
    F = []
    for c in range(4):
        s = M[OFF[c][0]::2, OFF[c][1]::2][:ny, :nx]
        F.append(np.fft.rfft2((s - s.mean()) * w))
    fy = (np.fft.fftfreq(ny) * 2 * np.pi)[:, None]
    fx = (np.fft.rfftfreq(nx) * 2 * np.pi)[None, :]
    sel = (np.abs(fy) < fmax * np.pi) & (fx < fmax * np.pi) & ((fy != 0) | (fx != 0))
    Y = np.broadcast_to(fy, sel.shape)[sel]
    X = np.broadcast_to(fx, sel.shape)[sel]
    d = np.zeros((4, 2))
    for c in range(4):
        if c == 2:
            continue
        R = F[c] * np.conj(F[2])
        pc = np.fft.irfft2(R / (np.abs(R) + 1e-12), s=(ny, nx))
        k = np.unravel_index(np.argmax(pc), pc.shape)
        t0 = [-(k[0] if k[0] < ny // 2 else k[0] - ny), -(k[1] if k[1] < nx // 2 else k[1] - nx)]
        t = _phase_slope(R[sel], Y, X, t0)
        d[c] = np.array(OFF[c], float) - 2 * t
    if not np.all(np.isfinite(d)) or np.abs(d).max() > MAX_SHIFT:
        return None
    return d


# ----------------------------------------------------------------------------- filters
def _keys(t, a=-0.5):
    """Keys cubic convolution kernel."""
    t = np.minimum(np.abs(t), 2.0)
    t2 = t * t
    inner = ((a + 2) * t - (a + 3)) * t2 + 1.0
    outer = a * (((t - 5.0) * t + 8.0) * t - 4.0)
    return np.where(t <= 1.0, inner, outer)


def _interp_mat(L, off):
    lags = np.arange(-L, L + 1) + off
    return _keys(lags[:, None] - np.arange(-T, T + 1)[None, :])


LAG_SCALES = (0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.2)
STOKES_BETA = (1.0, 0.5, 0.0)   # scaling of the Stokes-inconsistency component of the model
POL_ALPHA = ()          # scalings of the polarisation components tried for the best model
NOISE_F = (0.03, 0.1, 0.3, 0.7, 1.5)
ENSEMBLE_B = True
TOP2_AVG = True
ENS_MAX_F = 0.3
ENS_EXTRA = 0.08
DEFAULT_F = 1.0     # white noise of the fallback model (relative to the summed training noise)
BS_STEP = 4   # subsampling of superpixels for the blind-spot selection


def _test_cov(dte, L, k, CT=None, DT=None):
    """Signal covariance of the test scene at integer lags |l|<=L (training covariances re-aligned
    from their own shifts to the test shifts, lag axis scaled by k), and the training white noise
    smeared by the same lag scaling (a model of spatially correlated noise)."""
    CT = CTR if CT is None else CT
    DT = DTR if DT is None else DT
    S = CT.shape[0]
    lags = np.arange(-L, L + 1).astype(float)
    grid = np.arange(-T, T + 1).astype(float)
    Dt = dte[None, :, :] - dte[:, None, :]                # (4,4,2) d_c2 - d_c1
    Ds = DT[:, None, :, :] - DT[:, :, None, :]            # (S,4,4,2)
    eye = np.eye(4, dtype=bool)
    Dt = np.where(eye[:, :, None], 0.0, Dt)
    Ds = np.where(eye[None, :, :, None], 0.0, Ds)
    uy = (lags[None, None, None, :] - Dt[None, :, :, 0, None]) * k + Ds[:, :, :, 0, None]   # (S,4,4,n)
    ux = (lags[None, None, None, :] - Dt[None, :, :, 1, None]) * k + Ds[:, :, :, 1, None]
    g = int(min(T, np.ceil(max(np.abs(uy).max(), np.abs(ux).max())) + 2))
    sub = CT[:, :, :, T - g:T + g + 1, T - g:T + g + 1]
    gr = grid[T - g:T + g + 1]
    Wy = _keys(uy[..., None] - gr)                        # (S,4,4,n,G')
    Wx = _keys(ux[..., None] - gr)
    Ct = np.einsum('sabyg,sabgh,sabxh->abyx', Wy, sub, Wx, optimize=True)
    ky = _keys(lags * k)
    sm = np.outer(ky, ky) * NSUM
    return Ct, sm


def _psd_solve(A, b, extra, lam=LAM):
    """solve (A+ + extra + ridge) w = b where A+ is A with negative eigenvalues clipped to zero."""
    A = 0.5 * (A + A.T)
    ev, V = np.linalg.eigh(A)
    ev = np.maximum(ev, 0.0)
    ridge = lam * max(float(ev.mean()), 1e-12)
    return V @ ((V.T @ b) / (ev + extra + ridge))


def _chan(qy, qx):
    return PH2C[(qy % 2, qx % 2)]


def build(dte, Ct, L, extra=0.0, r=R_WIN, lam=LAM):
    """Weights (12, F) over superpixel features (c1, my, mx); output rows = (phase, target)."""
    cent = np.round(dte[:, None, :] - dte[None, :, :]).astype(int)  # cent[c1, c] = round(d_c1 - d_c)
    rows = []
    feat_index = {}
    entries = []
    for (i, j), cc in PH2C.items():
        for c in range(4):
            if c == cc:
                continue
            fc, fy, fx = [], [], []
            for c1 in range(4):
                cy, cx = cent[c1, c]
                for qy in range(cy - r, cy + r + 1):
                    for qx in range(cx - r, cx + r + 1):
                        if _chan(i + qy, j + qx) == c1:
                            fc.append(c1); fy.append(qy); fx.append(qx)
            fc = np.array(fc); fy = np.array(fy); fx = np.array(fx)
            A = Ct[fc[:, None], fc[None, :], L + fy[None, :] - fy[:, None], L + fx[None, :] - fx[:, None]]
            b = Ct[fc, c, L - fy, L - fx]
            w = _psd_solve(A, b, extra, lam)
            n = len(fc)
            ri = len(rows)
            rows.append((i, j, c))
            for k in range(n):
                c1 = fc[k]
                my = (i + fy[k] - OFF[c1][0]) // 2
                mx = (j + fx[k] - OFF[c1][1]) // 2
                key = (c1, my, mx)
                if key not in feat_index:
                    feat_index[key] = len(feat_index)
                entries.append((ri, feat_index[key], w[k]))
    W = np.zeros((len(rows), len(feat_index)), np.float32)
    for ri, fi, wv in entries:
        W[ri, fi] += wv
    feats = [None] * len(feat_index)
    for key, k in feat_index.items():
        feats[k] = key
    return rows, feats, W


def apply(M, rows, feats, W):
    H, Wd = M.shape
    h, w = H // 2, Wd // 2
    mu = np.array([M[OFF[c][0]::2, OFF[c][1]::2].mean() for c in range(4)])
    m = max(max(abs(f[1]), abs(f[2])) for f in feats) + 1      # superpixel margin
    pad = 2 * m
    P = np.pad(M.astype(np.float32), pad, mode='reflect')        # parity preserving
    sub = []
    for c in range(4):
        s = np.ascontiguousarray(P[OFF[c][0]::2, OFF[c][1]::2])
        s -= np.float32(mu[c])
        sub.append(s)
    X = np.empty((len(feats), h, w), np.float32)
    for k, (c1, my, mx) in enumerate(feats):
        X[k] = sub[c1][m + my:m + my + h, m + mx:m + mx + w]
    Y = W @ X.reshape(len(feats), -1)
    out = np.empty((4, H, Wd), np.float32)
    for (i, j), cc in PH2C.items():
        out[cc, i::2, j::2] = M[i::2, j::2]
    for ri, (i, j, c) in enumerate(rows):
        out[c, i::2, j::2] = Y[ri].reshape(h, w) + np.float32(mu[c])
    return out


def _bs_setup(M, dte, r=R_WIN):
    """features of the blind-spot task (predict each observed sample from its neighbours)."""
    cent = np.round(dte[:, None, :] - dte[None, :, :]).astype(int)
    H, Wd = M.shape
    pad = 2 * (r + int(np.abs(cent).max()) + 2)
    P = np.pad(M.astype(np.float32), pad, mode='reflect')
    mu = np.array([M[OFF[c1][0]::2, OFF[c1][1]::2].mean() for c1 in range(4)], np.float32)
    st = 2 * BS_STEP
    tasks = []
    for (i, j), c in PH2C.items():
        fc, fy, fx = [], [], []
        for c1 in range(4):
            cy, cx = cent[c1, c]
            for qy in range(cy - r, cy + r + 1):
                for qx in range(cx - r, cx + r + 1):
                    if (qy, qx) == (0, 0):
                        continue
                    if _chan(i + qy, j + qx) == c1:
                        fc.append(c1); fy.append(qy); fx.append(qx)
        fc = np.array(fc); fy = np.array(fy); fx = np.array(fx)
        tgt = P[pad + i:pad + i + H:st, pad + j:pad + j + Wd:st] - mu[c]
        X = np.empty((len(fc),) + tgt.shape, np.float32)
        for k in range(len(fc)):
            y0 = pad + i + fy[k]; x0 = pad + j + fx[k]
            X[k] = P[y0:y0 + H:st, x0:x0 + Wd:st] - mu[fc[k]]
        tasks.append((c, fc, fy, fx, X.reshape(len(fc), -1), tgt.ravel(), np.ones(tgt.size, np.float32)))
    return tasks


def _bs_mse_multi(tasks, Ct, L, extras, lam=LAM):
    """blind-spot MSE for several white-noise levels (one eigendecomposition per task)."""
    extras = np.asarray(extras, float)
    tot = np.zeros(len(extras))
    cnt = 0.0
    for c, fc, fy, fx, X, tgt, wgt in tasks:
        A = Ct[fc[:, None], fc[None, :], L + fy[None, :] - fy[:, None], L + fx[None, :] - fx[:, None]]
        b = Ct[fc, c, L - fy, L - fx]
        A = 0.5 * (A + A.T)
        n = len(fc)
        ev, V = np.linalg.eigh(A)
        ev = np.maximum(ev, 0.0)
        ev = ev + lam * max(float(ev.mean()), 1e-12)
        Vb = V.T @ b
        Wm = V @ (Vb[:, None] / (ev[:, None] + extras[None, :]))   # (n, F)
        E = Wm.T.astype(np.float32) @ X - tgt[None, :]
        tot += (E * E) @ wgt
        cnt += float(wgt.sum())
    return tot / cnt


def _pol(Ct, alpha):
    """scale the two polarisation (Stokes S1/S2-like) components of the model by alpha."""
    if alpha == 1.0:
        return Ct
    va = np.array([-1.0, -1.0, 1.0, 1.0]) / 2.0     # a = (I90 - I45 + I135 - I0)/4 in channel order 0,45,90,135
    vb = np.array([-1.0, 1.0, 1.0, -1.0]) / 2.0     # b = (I90 + I45 - I135 - I0)/4
    Pi = np.eye(4) + (alpha - 1.0) * (np.outer(va, va) + np.outer(vb, vb))
    return np.einsum('ab,bcyx,dc->adyx', Pi, Ct, Pi)


def _stokes(Ct, beta):
    if beta == 1.0:
        return Ct
    v = np.array([1.0, -1.0, 1.0, -1.0]) / 2.0
    Pi = np.eye(4) - (1.0 - beta) * np.outer(v, v)
    return np.einsum('ab,bcyx,dc->adyx', Pi, Ct, Pi)


def _bilinear(M):
    """per-channel bilinear interpolation (fallback when the channels cannot be registered)."""
    H, Wd = M.shape
    P = np.pad(M.astype(np.float32), 1, mode='reflect')
    out = np.empty((4, H, Wd), np.float32)
    k = ((0.25, 0.5, 0.25), (0.5, 1.0, 0.5), (0.25, 0.5, 0.25))
    yy = (np.arange(H + 2) - 1) % 2
    xx = (np.arange(Wd + 2) - 1) % 2
    for c in range(4):
        mask = ((yy[:, None] == OFF[c][0]) & (xx[None, :] == OFF[c][1])).astype(np.float32)
        Q = P * mask
        acc = np.zeros((H, Wd), np.float32)
        for a in range(3):
            for bb in range(3):
                acc += np.float32(k[a][bb]) * Q[a:a + H, bb:bb + Wd]
        out[c] = acc
    return out


def _nn_ref(M):
    """cheap per-channel nearest-neighbour reconstruction (sanity reference only)."""
    H, Wd = M.shape
    out = np.empty((4, H, Wd), np.float32)
    for c in range(4):
        s = M[OFF[c][0]::2, OFF[c][1]::2].astype(np.float32)
        out[c] = np.repeat(np.repeat(s, 2, axis=0), 2, axis=1)[:H, :Wd]
    return out


def _valid(out, ref):
    if out is None or not np.all(np.isfinite(out)):
        return False
    d = out - ref
    return float(np.sqrt(np.mean(d * d))) < 40.0


def _run(M, d, L, select):
    """main pipeline; select=False uses the default (training-like) model."""
    info = {}
    if select:
        tasks = _bs_setup(M, d)
        best = None
        extras = [f * NSUM for f in NOISE_F]
        covs = {}
        perk = []
        allc = []
        cands = [(k, 1.0) for k in LAG_SCALES]
        for stage in (0, 1):
            for k, beta in cands:
                if k not in covs:
                    covs[k] = _test_cov(d, L, k)[0]
                Ct = _stokes(covs[k], beta)
                es = _bs_mse_multi(tasks, Ct, L, extras)
                kb = np.inf
                for f, ex, e in zip(NOISE_F, extras, es):
                    if not np.isfinite(e):
                        continue
                    kb = min(kb, e)
                    allc.append((float(e), k, ex, beta, f))
                    if best is None or e < best[0]:
                        best = (e, k, ex, Ct, (beta, f))
                perk.append((kb, k))
            if stage == 0:
                top = [k for _, k in sorted(perk)[:2]]
                cands = [(k, beta) for k in top for beta in STOKES_BETA if beta != 1.0]
        if best is None:
            return None, info
        base = best
        for alpha in POL_ALPHA:
            Ct = _pol(base[3], alpha)
            es = _bs_mse_multi(tasks, Ct, L, extras)
            for f, ex, e in zip(NOISE_F, extras, es):
                if np.isfinite(e) and e < best[0]:
                    best = (e, base[1], ex, Ct, base[4] + (alpha,))
        _, k, ex, Ct, tag = best
        info['k'] = (k, tag)
        r_fin = R_FINAL if tag[1] >= BIG_WIN_MIN_F else R_WIN
        if r_fin != R_WIN:
            # recompute the covariance table on the larger lag range needed by the final filters
            cent = np.round(d[:, None, :] - d[None, :, :]).astype(int)
            L = 2 * r_fin + 2 * int(np.abs(cent).max()) + 2
            Ct = _test_cov(d, L, k)[0]
            Ct = _stokes(Ct, tag[0])
            if len(tag) > 2:
                Ct = _pol(Ct, tag[2])
    else:
        cent = np.round(d[:, None, :] - d[None, :, :]).astype(int)
        L = 2 * R_FINAL + 2 * int(np.abs(cent).max()) + 2
        Ct, _ = _test_cov(d, L, 1.0)
        ex = DEFAULT_F * NSUM
        info['k'] = ('default',)
        r_fin = R_FINAL
    rows, feats, W = build(d, Ct, L, extra=ex, r=r_fin)
    if not np.all(np.isfinite(W)):
        return None, info
    if select and TOP2_AVG and len(allc) > 1:
        allc.sort(key=lambda t: t[0])
        e2, k2, ex2, beta2, f2 = allc[1]
        r2 = R_FINAL if f2 >= BIG_WIN_MIN_F else R_WIN
        if r2 == r_fin:
            C2 = _stokes(_test_cov(d, L, k2)[0], beta2)
            rr, ff, W2 = build(d, C2, L, extra=ex2, r=r_fin)
            if np.all(np.isfinite(W2)) and rr == rows and ff == feats:
                W = 0.5 * (W + W2)
                info['top2'] = (k2, beta2, f2)
    if select and ENSEMBLE_B and info.get('k', ('default',))[0] != 'default' and info['k'][1][1] <= ENS_MAX_F and info['k'][1][0] < 1.0:
        # clean, Stokes-consistent scene: hedge with the Bayer-capture model.  Both members are
        # linear, so their average is the filter with averaged weights (union of the supports).
        cent = np.round(d[:, None, :] - d[None, :, :]).astype(int)
        Lb = 2 * R_WIN + 2 * int(np.abs(cent).max()) + 2
        Cb = _stokes(_test_cov(d, Lb, 1.0, CTR_B, DTR_B)[0], 0.0)
        rb, fb, Wb = build(d, Cb, Lb, extra=ENS_EXTRA * CTR_B.shape[0], r=R_WIN)
        if np.all(np.isfinite(Wb)) and rb == rows:
            index = {f: i for i, f in enumerate(feats)}
            for f in fb:
                if f not in index:
                    index[f] = len(index)
            Wm = np.zeros((len(rows), len(index)), np.float32)
            Wm[:, :len(feats)] += 0.5 * W
            Wm[:, [index[f] for f in fb]] += 0.5 * Wb
            feats = [None] * len(index)
            for f, i in index.items():
                feats[i] = f
            W = Wm
            info['ens'] = True
    out = apply(M, rows, feats, W)
    np.clip(out, 0.0, 255.0, out=out)
    return out, info


def reconstruct(raw, return_info=False):
    M = np.asarray(raw).astype(np.float64)
    H0, W0 = M.shape
    if H0 % 2 or W0 % 2 or min(H0, W0) < 16:
        # odd or tiny inputs: pad (parity preserving) to an even size >= 16, crop at the end
        ph = max(H0 % 2, 16 - H0)
        pw = max(W0 % 2, 16 - W0)
        ph += ph % 2 if (H0 + ph) % 2 else 0
        pw += pw % 2 if (W0 + pw) % 2 else 0
        Mp = np.pad(M, ((0, ph), (0, pw)), mode='symmetric' if min(H0, W0) < 2 else 'reflect')
        res = reconstruct(Mp, return_info)
        if return_info:
            return res[0][:, :H0, :W0].copy(), res[1]
        return res[:, :H0, :W0].copy()
    try:
        d = mosaic_shifts(M)
    except Exception:
        d = np.zeros((4, 2))
    if d is None:
        out = _bilinear(M)
        np.clip(out, 0.0, 255.0, out=out)
        return (out, {'k': ('bilinear-unregistered',)}) if return_info else out
    cent = np.round(d[:, None, :] - d[None, :, :]).astype(int)
    L = 2 * R_WIN + 2 * int(np.abs(cent).max()) + 2
    ref = _nn_ref(M)
    out, info = None, {}
    spread = int(np.abs(cent).max())
    for select in ((True, False) if spread <= SEARCH_MAX_SPREAD else (False,)):
        try:
            out, info = _run(M, d, L, select)
        except Exception:
            out = None
        if _valid(out, ref):
            break
        out = None
    if out is None:
        out = _bilinear(M)
        info = {'k': ('bilinear',)}
    out = np.nan_to_num(out, nan=0.0, posinf=255.0, neginf=0.0)
    np.clip(out, 0.0, 255.0, out=out)
    out = out.astype(np.float32, copy=False)
    if return_info:
        return out, info
    return out
