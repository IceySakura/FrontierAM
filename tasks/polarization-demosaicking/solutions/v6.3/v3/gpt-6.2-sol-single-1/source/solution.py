"""Single-scene polarization demosaicking by registered spectral estimation."""

import os

# The evaluator pins the process to one CPU. Limit the numerical libraries too.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.fft import irfft2, rfft2
from scipy.ndimage import gaussian_filter, map_coordinates, uniform_filter1d


_K = 64
_HOP = 48
_PHASES = ((0, 0), (0, 1), (1, 0), (1, 1))
_OUTPUT_INDEX = (2, 1, 3, 0)
_HAD = np.array(((1, 1, 1, 1), (1, -1, 1, -1), (1, 1, -1, -1),
                 (1, -1, -1, 1)), dtype=np.float32)


def _tiles(mosaic, k, s):
    h, w = mosaic.shape
    t = k - s
    ny, nx = (h+t+s-1)//s, (w+t+s-1)//s
    padded = np.pad(mosaic, ((t, ny*s-h), (t, nx*s-w)), mode="reflect")
    i, j = padded.strides
    return np.lib.stride_tricks.as_strided(
        padded, shape=(ny, nx, k, k),
        strides=(i * s, j * s, i, j), writeable=False,
    )


def _window(k, s, translation=(0., 0.)):
    x = np.arange(k, dtype=np.float32) + .5
    a = x + translation[0]
    b = x + translation[1]
    t = k - s
    a = np.sin(np.float32(np.pi / (2*t)) * np.clip(np.minimum(a, k-a), 0, t)) ** 2
    b = np.sin(np.float32(np.pi / (2*t)) * np.clip(np.minimum(b, k-b), 0, t)) ** 2
    return b[:, None] * a[None, :]


def _overlap(blocks, window, s, shape):
    ny, nx, k, _ = blocks.shape
    t = k-s
    w0, wx = window[:s, :s], window[:s, s:]
    wy, wd = window[s:, :s], window[s:, s:]
    norm = w0*w0
    norm[:, :t] += wx*wx
    norm[:t, :] += wy*wy
    norm[:t, :t] += wd*wd
    result = blocks[:, :, :s, :s] * (w0/norm)
    result[:, 1:, :, :t] += blocks[:, :-1, :s, s:] * (wx/norm[:, :t])
    result[1:, :, :t, :] += blocks[:-1, :, s:, :s] * (wy/norm[:t, :])
    result[1:, 1:, :t, :t] += blocks[:-1, :-1, s:, s:] * (wd/norm[:t, :t])
    h, w = shape
    return result.transpose(0, 2, 1, 3).reshape(ny*s, nx*s)[t:t+h, t:t+w]


def _fold(a, j):
    # Pack one of the four aliases for each baseband frequency. The x-shift
    # aliases use the conjugate half of the real FFT.
    l = a.shape[-2] // 4
    if j == 0:
        return np.concatenate((a[:, -l:, :l+1], a[:, :l, :l+1]), axis=1)
    if j == 1:
        r = np.concatenate((a[:, l::-1, -1:l-1:-1],
                            a[:, -1:-l:-1, -1:l-1:-1]), axis=1)
    elif j == 2:
        return a[:, l:3*l, :l+1].copy()
    else:
        r = a[:, 3*l:l:-1, -1:l-1:-1].copy()
    return np.conjugate(r, out=r)


def _unfold(a, j, value):
    l = a.shape[-2] // 4
    if j == 0:
        a[:, -l:, :l+1] = value[:, :l]
        a[:, :l, :l+1] = value[:, l:]
    elif j == 1:
        a[:, l::-1, -1:l-1:-1] = value[:, :l+1].conj()
        a[:, -1:-l:-1, -1:l-1:-1] = value[:, l+1:].conj()
    elif j == 2:
        a[:, l:3*l, :l+1] = value
    else:
        a[:, 3*l:l:-1, -1:l-1:-1] = value.conj()


def _hadamard(v):
    a, b, c, d = v
    p, q = a + b, a - b
    r, s = c + d, c - d
    return [p + r, q + s, p - r, q - s]


def _abs2(a):
    return a.real * a.real + a.imag * a.imag


def _lowband(tiles, window, low, high):
    k = window.shape[0]
    blocks = rfft2(tiles[::2, ::2] * window, workers=1)
    blocks = blocks.reshape(-1, k, k//2 + 1)
    fy = np.arange(-k//4, k//4, dtype=np.float32) / k
    fx = np.arange(k//4+1, dtype=np.float32) / k
    rad = np.hypot(fy[:, None], fx[None, :])
    use = (rad >= low) & (rad < high)
    px = np.broadcast_to(fx[None, :], rad.shape)[use] * np.float32(2*np.pi)
    py = np.broadcast_to(fy[:, None], rad.shape)[use] * np.float32(2*np.pi)
    angles = _hadamard([_fold(blocks, j)[:, use] for j in range(4)])
    power = [_abs2(a) for a in angles]
    white = (sum(power) * .25 + 15.) ** (-.68)
    return angles, power, white, px, py


def _translations(tiles, window):
    angles, power, white, px, py = _lowband(tiles, window, .027, .132)
    quality = (power[0] * white).sum(axis=-1) ** .3
    quality /= max(float(quality.mean()), 1.e-9)
    white *= quality[:, None]
    px2, py2, pxy = px*px, py*py, px*py
    ww = np.exp(-.7 * (px2 + py2))
    grid = np.arange(-6, 7, dtype=np.float32)
    dxs = np.tile(grid, len(grid))
    dys = np.repeat(grid, len(grid))
    responses = np.exp(-1j * (dxs[:, None]*px + dys[:, None]*py))
    shifts = np.zeros((4, 2), np.float32)
    ref_power = (power[0] * white).sum()
    for i in range(1, 4):
        cross = np.sum(angles[i] * angles[0].conj() * white, axis=0)
        magn = abs(cross)
        c = responses @ (ww * cross)
        c = c.real - .0009 * magn.sum() * (dxs*dxs + dys*dys)
        best = int(np.argmax(c))
        dx, dy = float(dxs[best]), float(dys[best])
        for n in range(7):
            if n == 0:
                w = magn * ww
                taper = ww
            else:
                w = magn
                taper = 1.
            e = (cross * np.exp(-1j * (dx*px + dy*py))).imag * taper
            xx = w @ px2 + 1.e-4 * w.sum()
            yy = w @ py2 + 1.e-4 * w.sum()
            xy = w @ pxy
            denom = xx*yy - xy*xy + 1.e-6
            ex, ey = e @ px, e @ py
            dx += np.clip((yy*ex - xy*ey) / denom, -.8, .8)
            dy += np.clip((xx*ey - xy*ex) / denom, -.8, .8)
        aligned = (cross * np.exp(-1j * (dx*px + dy*py))).real.sum()
        coher = aligned / (np.sqrt(ref_power * (power[i]*white).sum()) + 1.e-6)
        conf = np.clip((coher - .30) / .30, 0., 1.)
        shifts[i] = (conf*dx, conf*dy)
    np.clip(shifts, -6., 6., out=shifts)
    return shifts - shifts.mean(axis=0)


def _locations(shape, blocks, k, hop, step):
    yy, xx = [((np.arange(0, n, step, dtype=np.float32)*hop + hop-k/2)
               / (sz*.5) - 1.) for sz, n in zip(shape, blocks)]
    x, y = np.broadcast_arrays(xx[None, :], yy[:, None])
    return np.stack((np.ones_like(x).ravel(), x.ravel(), y.ravel()), axis=-1)


def _registration(mosaic, target_tiles, k, hop):
    pilot = _tiles(mosaic, 64, 32)
    window = _window(64, 32)
    shift = _translations(pilot, window)
    for (i, j), d in zip(_PHASES, shift):
        window[i::2, j::2] = _window(64, 32, d)[i::2, j::2]
    # The second fit corrects the window displacement and estimates a smooth
    # affine field. A low-frequency band limits contamination by CFA aliases.
    extent = float(np.max(abs(shift)))
    cutoff = .137 + .022 / (1 + 4*extent*extent)
    angle, power, white, px, py = _lowband(pilot, window, .035, cutoff)
    mass = (power[0] * white).sum(-1)
    quality = (mass / max(float(mass.mean()), 1.e-9)) ** .25
    design = _locations(mosaic.shape, pilot.shape[:2], 64, 32, 2)
    outer = (design[:, :, None] * design[:, None, :]).reshape(-1, 9)
    reg = np.array([.00018, .018, .018, .00018, .018, .018], np.float32)
    coeff = np.zeros((4, 3, 2), np.float32)
    pilot_shape = pilot[::2, ::2].shape[:2]
    ty, tx = [(np.arange(n, dtype=np.float32)*hop + hop-k/2)/64.
              for n in target_tiles.shape[:2]]
    tx, ty = np.meshgrid(tx, ty)
    points = np.stack((ty.ravel(), tx.ravel()))
    residual = np.zeros((4, tx.size, 2), np.float32)
    pxx, pyy, pxy = px*px, py*py, px*py
    for i in range(1, 4):
        cross = angle[i] * angle[0].conj() * white
        amp = abs(cross)
        ref = np.sqrt(mass * (power[i]*white).sum(-1)) + 1.e-7
        d = shift[i] - shift[0]
        v = coeff[i]
        for _ in range(3):
            phase = design @ v + d
            cp = cross * np.exp(-1j * (phase[:, :1]*px + phase[:, 1:]*py))
            coh = np.clip((cp.real.sum(-1)/ref - .30)/.65, 0, 1)
            conf = (coh**1.5) * quality
            w = conf[:, None] * (.72 * np.maximum(cp.real, 0) + .28 * amp)
            xx, yy, xy = w @ pxx, w @ pyy, w @ pxy
            mat = np.empty((6, 6), np.float32)
            mat[:3, :3] = (xx @ outer).reshape(3, 3)
            mat[3:, 3:] = (yy @ outer).reshape(3, 3)
            mat[:3, 3:] = (xy @ outer).reshape(3, 3)
            mat[3:, :3] = mat[:3, 3:]
            mat.flat[::7] += reg * (.5*(xx.sum() + yy.sum()) + 1.e-5) + 1.e-5
            bx = (conf * (cp.imag @ px)) @ design
            by = (conf * (cp.imag @ py)) @ design
            update = np.linalg.solve(mat, np.concatenate((bx, by))).reshape(2, 3).T
            v += np.clip(update, -.6, .6)
        v *= min(1., float(np.quantile(coh, .70))/.6)
        # Accumulate local phase evidence before solving for a small
        # residual displacement. The affine estimate remains the anchor.
        phase = design @ v + d
        cp = cross * np.exp(-1j * (phase[:, :1]*px + phase[:, 1:]*py))
        coh = np.clip((cp.real.sum(-1)/ref - .38)/.58, 0, 1)
        conf = (coh**2.5) * quality
        w = conf[:, None] * (.72 * np.maximum(cp.real, 0) + .28 * amp)

        def smooth(a):
            return gaussian_filter(a.reshape(pilot_shape), 1.35, mode="nearest")

        xx, yy, xy = smooth(w @ pxx), smooth(w @ pyy), smooth(w @ pxy)
        bx, by = smooth(conf*(cp.imag @ px)), smooth(conf*(cp.imag @ py))
        local_mass = .5*(xx+yy)
        floor = .06*(local_mass + .35*local_mass.mean()) + .0001
        xx += floor
        yy += floor
        den = np.maximum(xx*yy-xy*xy, 1.e-7)
        lim = .19*(1. + .22*extent)
        extra = ((yy*bx-xy*by)/den, (xx*by-xy*bx)/den)
        for j, a in enumerate(extra):
            residual[i, :, j] = map_coordinates(
                np.clip(a, -lim, lim), points, order=1, mode="nearest", prefilter=False)
    np.clip(coeff, -1., 1., out=coeff)
    coeff -= coeff.mean(axis=0)
    coeff[:, 0] += shift
    target = _locations(mosaic.shape, target_tiles.shape[:2], k, hop, 1)
    strength = .53 + .29*extent/(.45+extent)
    positions = target @ coeff + strength*(residual-residual.mean(axis=0))
    return coeff[:, 0], np.clip(positions, -7., 7.)


def _phase(k, shift):
    x = np.fft.rfftfreq(k).astype(np.float32)
    y = np.fft.fftfreq(k).astype(np.float32)
    ax = np.exp(2j * np.pi * x * shift[0])
    ay = np.exp(2j * np.pi * y * shift[1])
    ax[-1] = ax[-1].real
    ay[k//2] = ay[k//2].real
    return (ay[:, None] * ax[None, :])[None]


def _phase_axes(k, shifts):
    x = np.fft.rfftfreq(k).astype(np.float32)
    y = np.fft.fftfreq(k).astype(np.float32)
    ax = np.exp(2j * np.pi * x * shifts[:, :1])
    ay = np.exp(2j * np.pi * y * shifts[:, 1:])
    ax[:, -1] = ax[:, -1].real
    ay[:, k//2] = ay[:, k//2].real
    return ax, ay


def _local_geometry(k, shifts):
    # Relative phases of the high aliases are constant in each frequency
    # quadrant, except on the Nyquist lines.
    x = np.exp((1j*np.pi) * shifts[:, :, 0])
    y = np.exp((1j*np.pi) * shifts[:, :, 1])
    yparts = ((slice(0, k//4), y), (slice(k//4, k//4+1), y.real),
              (slice(k//4+1, None), y.conj()))
    xparts = ((slice(0, 1), x.real), (slice(1, None), x.conj()))
    for sl_y, b in yparts:
        for sl_x, a in xparts:
            rel = [None, a, b, a*b]
            geometry = [[None] + [
                ((_HAD[i]*_HAD[j]) @ rel[j])[:, None, None] * .25
                for j in range(1, 4)] for i in range(4)]
            yield (slice(None), sl_y, sl_x), geometry


def _gains(aligned, k, shape):
    u = np.arange(k//4+1, dtype=np.float32) / k
    v = np.arange(-k//4, k//4, dtype=np.float32) / k
    rad = np.hypot(v[:, None], u[None, :])
    band = (.043 <= rad) & (rad < .135)
    values = [a[:, band] for a in aligned]
    power = sum(_abs2(a) for a in values) * .25
    wt = (power + .08*power.mean(axis=-1, keepdims=True) + 8.) ** (-.65)
    guide = sum(values) * .25
    strength = (_abs2(guide)*wt).sum(axis=1)
    coef = np.stack([(np.real(a*guide.conj())*wt).sum(axis=1)
                     for a in values]) / np.maximum(strength, 1.e-7)
    coef = np.clip(coef-1., -.40, .40)
    corr = np.clip((strength / np.maximum((power*wt).sum(axis=1), 1.e-7) - .33)/.62, 0, 1)
    mix = (corr**1.7 * np.minimum((power.mean(axis=1)+1.)**.30/16., 1.)).reshape(shape[:2])
    a = gaussian_filter(coef.reshape(4, *shape[:2])*mix, (0., .75, .75),
                        mode="nearest", truncate=2.5)
    w = gaussian_filter(mix, .75, mode="nearest", truncate=2.5)
    coef = (a/np.maximum(w, 1.e-7)).reshape(coef.shape)
    g = 1. + np.clip(corr+.15, 0, 1)*coef
    return g / g.mean(axis=0)


def _basis(k, win):
    x = np.fft.rfftfreq(k).astype(np.float32)[None, :]
    y = np.fft.fftfreq(k).astype(np.float32)[:, None]
    atoms = [np.ones((k, k//2+1), np.float32)]
    cuts = [1]
    peak = _abs2(rfft2(win, workers=1))[None]
    peak /= peak.max()
    dc = np.empty_like(peak)
    for i, (dy, dx) in enumerate(_PHASES):
        u = (x + (.5 - .5*dx)) % 1 - .5
        v = (y + (.5 - .5*dy)) % 1 - .5
        uu, vv = u*u, v*v
        r2 = uu + vv
        for j in range(4):
            _unfold(dc, j, _fold(peak, j^i))
        atoms.append(dc[0].copy())
        sigmas = [1.28/k, .065, .105, .163, .24] if not i else [1.28/k, .068, .111, .165]
        for s in sigmas:
            atoms.append(np.exp(-r2 * (.5 / (s*s))))
        for s in ([.085, .162, .25] if not i else [.080, .132]):
            fac = .3 if not i else .55
            atoms.append(np.exp(-(uu + vv/(fac*fac)) * (.5 / (s*s))))
            atoms.append(np.exp(-(uu/(fac*fac) + vv) * (.5 / (s*s))))
        cuts.append(len(atoms))
    bank = np.stack(atoms).reshape(len(atoms), -1)
    bank /= np.sqrt(np.sum(bank*bank, axis=1, keepdims=True))
    weights = np.ones((k, k//2+1), np.float32)
    weights[:, [0, -1]] = .5
    return bank, cuts, weights.reshape(1, -1)


def _shift_basis(bank, cuts, geometry, k):
    lifted = bank.reshape(-1, k, k//2+1).copy()
    central = lifted[cuts[0]:cuts[1]]
    base = [_fold(central, j) for j in range(4)]
    for i, row in enumerate(geometry):
        v = base[0].copy() if i == 0 else np.zeros_like(base[0])
        for j in range(1, 4):
            v += base[j] * _abs2(row[j])
        _unfold(central, i, v)
    return lifted.reshape(bank.shape)


def _fit(power, bank, wt, init, n, damping, anchor, cap=None):
    a = init
    scale = 1. if cap is None else 1./(1.-np.exp(-cap))
    for _ in range(n):
        estimate = np.maximum(a @ bank, 1.e-7)
        inv = wt / estimate
        relative = power / estimate
        if cap is not None:
            np.minimum(relative, cap, out=relative)
        ratio = ((inv * relative) @ bank.T + 1.e-12)
        ratio /= inv @ bank.T + 1.e-12
        ratio *= scale
        if anchor:
            ratio *= 1. - anchor
            ratio += anchor * init / np.maximum(a, 1.e-7)
        np.clip(ratio, .03, 10, out=ratio)
        a = np.maximum(a * ((1.-damping) + damping*ratio), 1.e-6)
    return a


def _solve4(diag, lower, vec):
    # LDL^H for the Hermitian covariance, vectorized over bins and tiles.
    a, b, c, d = diag
    f, g, i, h, j, k = lower
    ia = 1 / np.maximum(a, 1.e-7)
    F, G, H = f*ia, g*ia, h*ia
    ib = 1 / np.maximum(b - _abs2(f)*ia, 1.e-7)
    t21 = i - g*F.conj()
    t31 = j - h*F.conj()
    I, J = t21*ib, t31*ib
    ic = 1 / np.maximum(c - _abs2(g)*ia - _abs2(t21)*ib, 1.e-7)
    t32 = k - h*G.conj() - t31*I.conj()
    K = t32*ic
    id_ = 1 / np.maximum(d - _abs2(h)*ia - _abs2(t31)*ib - _abs2(t32)*ic, 1.e-7)
    q, w, e, r = vec
    w = w - F*q
    e = e - G*q - I*w
    r = (r - H*q - J*w - K*e)*id_
    e = e*ic - K.conj()*r
    w = w*ib - J.conj()*r - I.conj()*e
    q = q*ia - H.conj()*r - G.conj()*e - F.conj()*w
    return [q, w, e, r]


def _wiener(freq, centered, geom, shifts, win):
    k = win.shape[0]
    for j, v in enumerate(centered):
        _unfold(freq, j, v)
    power = _abs2(freq).reshape(len(freq), -1) / np.sum(win*win)
    bank, cuts, weights = _basis(k, win)
    basis = _shift_basis(bank, cuts, geom, k)
    pm = power.mean(axis=0, keepdims=True)
    init = np.full((1, len(bank)), max(pm.mean() / len(bank), 1.e-5), np.float32)
    init = _fit(pm, basis, weights, init, 75, .78, 0.)
    # Nearby periodogram bins are redundant for fitting the smooth templates.
    # Retain extra bins near the four carrier centers.
    y = np.arange(k)[:, None]
    x = np.arange(k//2+1)[None, :]
    r2 = np.minimum(y%(k//2), (-y)%(k//2))**2 + np.minimum(x%(k//2), (-x)%(k//2))**2
    sample = np.flatnonzero((((3*y + 5*x) % 7 >= 3) | (r2 < 8)).ravel())
    subpower = np.take(power, sample, axis=1)
    subpower *= .98
    subpower += .02*pm[:, sample]
    local = _fit(subpower, np.take(basis, sample, axis=1), weights[:, sample],
                 init, 9, .83, .035, cap=7.)
    del subpower
    start, end = cuts[:2]
    common = (local[:, start:end] @ bank[start:end]).reshape(freq.shape)
    commons = [_fold(common, j) for j in range(4)]
    bonus = [(.75*(local[:, l:h] @ bank[l:h])).reshape(freq.shape)
             for l, h in zip(cuts[1:], cuts[2:])]
    # The empirical power ratio sharpens the noise/signal separation in
    # locally sparse bands without relying on a single noisy periodogram bin.
    ref = np.maximum(local @ basis, 1.e-7)
    power /= ref
    np.sqrt(np.minimum(power, 25., out=power), out=power)
    ratio = uniform_filter1d(power.reshape(freq.shape), 5, axis=-2, mode="wrap")
    ratio = uniform_filter1d(ratio, 5, axis=-1, mode="mirror")
    ratio /= .90
    ratio *= ratio
    np.clip(ratio, .12, 8., out=ratio)
    noise = (local[:, :1, None] * (2.1 * bank[0, 0])) / (.30 + .70*ratio)
    del ratio, ref, power
    rem = sum(bonus) + noise
    diagonal_rem = [_fold(rem, j) for j in range(4)]
    coeff = [np.empty_like(p) for p in centered]
    restored = [np.empty_like(p) for p in centered]
    del rem
    # Shared structure has a generally non-diagonal observation covariance
    # after registration. The three other fields contribute to its diagonal.
    for sl, geom_local in _local_geometry(k, shifts):
        a = [p[sl] for p in commons]
        diagonal, lower = [], []
        for i, row in enumerate(geom_local):
            n = diagonal_rem[i][sl].copy()
            if i == 0:
                n += a[0]
            for j in range(1, 4):
                n += a[j] * _abs2(row[j])
            diagonal.append(n)
            for h in range(i):
                lower.append(sum(a[j] * (row[j] * geom_local[h][j].conj())
                                 for j in range(1, 4)))
        z = _solve4(diagonal, lower, [v[sl] for v in centered])
        for j in range(4):
            coeff[j][sl] = z[j]
        restored[0][sl] = a[0] * z[0]
        for j in range(1, 4):
            restored[j][sl] = a[j] * sum(v * r[j].conj() for v, r in zip(z, geom_local))
    common = np.empty_like(freq)
    for j in range(4):
        _unfold(freq, j, coeff[j])
        _unfold(common, j, restored[j])
    spectrum = [common]
    for j in range(1, 4):
        a = bonus[j-1] * freq
        if j & 1:
            a = a[:, (-np.arange(k)) % k, ::-1].conj()
        if j & 2:
            a = np.roll(a, k//2, axis=-2)
        spectrum.append(a)
    return spectrum


def _interp_half(src, i, j):
    """Bilinear extension of one even/odd sampling grid."""
    h, w = src.shape
    a = np.empty((h, 2*w), np.float32)
    a[:, j::2] = src
    a[:, 1:-1][:, j::2] = .5*(src[:, :-1] + src[:, 1:])
    if j:
        a[:, 0] = src[:, 0]
    else:
        a[:, -1] = src[:, -1]
    b = np.empty((2*h, 2*w), np.float32)
    b[i::2] = a
    b[1:-1][i::2] = .5*(a[:-1] + a[1:])
    if i:
        b[0] = a[0]
    else:
        b[-1] = a[-1]
    return b


def _mean3(a):
    # Separable reflected 3x3 stencil, using contiguous row operations.
    t = np.empty_like(a)
    np.add(a[:-2], a[2:], out=t[1:-1])
    t[1:-1] += a[1:-1]
    t[0] = a[0] + a[0] + a[1]
    t[-1] = a[-2] + a[-1] + a[-1]
    t *= 1./3.
    result = np.empty_like(a)
    np.add(t[:, :-2], t[:, 2:], out=result[:, 1:-1])
    result[:, 1:-1] += t[:, 1:-1]
    result[:, 0] = t[:, 0] + t[:, 0] + t[:, 1]
    result[:, -1] = t[:, -2] + t[:, -1] + t[:, -1]
    result *= 1./3.
    return result


def _spatial_refine(out, raw, shifts):
    # Small residual corrections: suppress angle-specific sample noise and
    # retain shared detail where residual energy exceeds the noise floor.
    error = np.empty(raw.shape, np.float32)
    interpolants = []
    for (y, x), c in zip(_PHASES, _OUTPUT_INDEX):
        sub = error[y::2, x::2]
        np.subtract(raw[y::2, x::2], out[c, y::2, x::2], out=sub)
        interpolants.append(_interp_half(sub, y, x))
    avg = sum(interpolants)*.25
    for c, prior in zip(_OUTPUT_INDEX, interpolants):
        out[c] -= .16*(prior-.53*avg)
    del interpolants, avg
    hp = error - gaussian_filter(error, .82, mode="reflect", truncate=2.35)
    energy = _mean3(error*error)
    g = out.mean(axis=0)
    mean = _mean3(g)
    noise = 1.10 + 1.22*np.maximum(.025*(mean-7.), .20)
    gain = np.maximum((energy-noise)/(energy+noise+2.), 0.)
    var = np.maximum(_mean3(g*g)-mean*mean, 0.)
    gain *= var/(var+2.)
    motion = float(np.max(np.abs(shifts)))
    gain *= 1.42*(.85+.15/(1.+motion))
    np.minimum(gain, .63, out=gain)
    out += gain*hp


def _reconstruct(raw, k, hop):
    mosaic = raw.astype(np.float32)
    tiles = _tiles(mosaic, k, hop)
    win = _window(k, hop)
    averages, shifts = _registration(mosaic, tiles, k, hop)
    windows = [_window(k, hop, d) for d in averages]
    mask = win.copy()
    for (i, j), w in zip(_PHASES, windows):
        mask[i::2, j::2] = w[i::2, j::2]
    freq = rfft2(tiles * mask, workers=1)
    shape = freq.shape
    freq = freq.reshape(-1, k, k//2+1)
    mean_phases = [_phase(k, d) for d in averages]
    pf = [[_fold(p, j)[0] for j in range(4)] for p in mean_phases]
    geom = [[None] + [
        sum((_HAD[h, i] * _HAD[j, i]) * (p[j] * p[0].conj())
            for i, p in enumerate(pf)) * .25
        for j in range(1, 4)
    ] for h in range(4)]
    phases = [_phase_axes(k, d) for d in shifts]
    angles = _hadamard([_fold(freq, j) for j in range(4)])
    for ab, (ax, ay) in zip(angles, phases):
        ab *= ax[:, None, :k//4+1].conj()
        ab *= np.concatenate((ay[:, -k//4:], ay[:, :k//4]), axis=1).conj()[:, :, None]
        ab *= .25
    gains = _gains(angles, k, shape)
    for ab, g in zip(angles, gains):
        ab /= g[:, None, None]
    spec = _wiener(freq, _hadamard(angles), geom, shifts, win)
    del angles, freq
    out = np.empty((4, *raw.shape), dtype=np.float32)
    for h, ((i, j), w, (ax, ay), g) in enumerate(zip(_PHASES, windows, phases, gains)):
        cs = spec[0].copy()
        for u in range(1, 4):
            if _HAD[h, u] > 0:
                cs += spec[u]
            else:
                cs -= spec[u]
        cs *= ax[:, None, :]
        cs *= ay[:, :, None] * g[:, None, None]
        spatial = irfft2(cs.reshape(shape), overwrite_x=True, workers=1)
        channel = out[_OUTPUT_INDEX[h]]
        channel[:] = _overlap(spatial, w, hop, raw.shape)
        np.clip(channel, 0, 255, out=channel)
    _spatial_refine(out, raw, shifts)
    np.clip(out, 0, 255, out=out)
    for (i, j), c in zip(_PHASES, _OUTPUT_INDEX):
        channel = out[c]
        channel[i::2, j::2] = raw[i::2, j::2]
    not_finite = ~np.isfinite(out)
    if not_finite.any():
        # Defensive fallback for an ill-conditioned or degenerate tile.
        np.copyto(out, raw, where=not_finite)
    return out


def reconstruct(raw):
    """Return full-resolution float32 planes in the order 0, 45, 90, 135."""
    h, w = raw.shape
    block = raw[:2, :2]
    flat = all(np.all(raw[i::32, j::32] == block[i, j]) for i, j in _PHASES)
    if flat and np.all(raw.reshape(h//2, 2, w//2, 2) == block[None, :, None, :]):
        levels = np.array((block[1, 1], block[0, 1], block[0, 0], block[1, 0]), np.float32)
        return np.broadcast_to(levels[:, None, None], (4, h, w)).copy()
    return _reconstruct(raw, _K, _HOP)
