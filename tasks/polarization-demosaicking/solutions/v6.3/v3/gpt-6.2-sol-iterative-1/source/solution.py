"""Single-frame polarization demosaicking with local angle agreement."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


_PHASES = ((0, 0), (0, 1), (1, 0), (1, 1))
_ORDER = (3, 1, 0, 2)


def _filter_axis(a, kernel, axis):
    r = len(kernel) // 2
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    t = np.pad(a, pad, mode="edge")
    sl = [slice(None)] * a.ndim
    sl[axis] = slice(0, a.shape[axis])
    out = t[tuple(sl)] * kernel[0]
    for i in range(1, len(kernel)):
        sl[axis] = slice(i, i + a.shape[axis])
        out += t[tuple(sl)] * kernel[i]
    return out


def _align(a, sigma):
    r = 2 if sigma < 1.75 else 3
    x = np.arange(-r, r + 1, dtype=np.float64) * 2 - .5
    k = np.exp(-.5 * (x / sigma) ** 2)
    k = (k / k.sum()).astype(np.float32)
    hx = np.stack((k, k[::-1], k, k[::-1]), axis=1)[:, :, None, None]
    hy = hx[:, [0, 0, 1, 1]]
    return _filter_axis(_filter_axis(a, hx, 2), hy, 1)


def _upsample_axis(a):
    r = np.empty((*a.shape[:-1], 2 * a.shape[-1]), dtype=np.float32)
    r[..., 0] = a[..., 0]
    r[..., -1] = a[..., -1]
    r[..., 1:-1:2] = .75 * a[..., :-1] + .25 * a[..., 1:]
    r[..., 2::2] = .25 * a[..., :-1] + .75 * a[..., 1:]
    return r


def _upsample(a):
    x = _upsample_axis(a)
    r = np.empty((*x.shape[:-2], 2*x.shape[-2], x.shape[-1]), dtype=np.float32)
    r[..., 0, :] = x[..., 0, :]
    r[..., -1, :] = x[..., -1, :]
    r[..., 1:-1:2, :] = .75 * x[..., :-1, :] + .25 * x[..., 1:, :]
    r[..., 2::2, :] = .25 * x[..., :-1, :] + .75 * x[..., 1:, :]
    return r


def _cubic_axis(a, phase):
    n = a.shape[-1]
    t = np.pad(a, [(0, 0)] * (a.ndim - 1) + [(2, 2)], mode="reflect")
    b = np.empty((*a.shape[:-1], 2*n), dtype=np.float32)
    b[..., phase::2] = a
    s = 1 - phase
    b[..., 1 - phase::2] = (
        .5625 * (t[..., s+1:s+1+n] + t[..., s+2:s+2+n])
        - .0625 * (t[..., s:s+n] + t[..., s+3:s+3+n])
    )
    return b


def _cubic(a, i, j):
    x = _cubic_axis(a, j)
    n = len(x)
    t = np.pad(x, ((2, 2), (0, 0)), mode="reflect")
    b = np.empty((2*n, x.shape[1]), dtype=np.float32)
    b[i::2] = x
    s = 1 - i
    b[1 - i::2] = (
        .5625 * (t[s+1:s+1+n] + t[s+2:s+2+n])
        - .0625 * (t[s:s+n] + t[s+3:s+3+n])
    )
    return b


def _box(a, r):
    h, w = a.shape
    p = np.pad(a, ((0, 0), (r, r)), mode="edge")
    f = p[:, :w].copy()
    for i in range(1, 2*r + 1):
        f += p[:, i:i+w]
    p = np.pad(f, ((r, r), (0, 0)), mode="edge")
    f[:] = p[:h]
    for i in range(1, 2*r + 1):
        f += p[i:i+h]
    f *= 1. / (2*r + 1)**2
    return f


def _blur(a):
    k = (.25, .5, .25)
    return _filter_axis(_filter_axis(a, k, -1), k, -2)


def _cell_average(x, radius):
    s = (x[::2, ::2] + x[::2, 1::2] + x[1::2, ::2] + x[1::2, 1::2]) * .25
    return _box(s, radius)


def _derivatives(a):
    p = np.pad(a, 1, mode="edge")
    y = .5 * (p[2:, 1:-1] - p[:-2, 1:-1])
    x = .5 * (p[1:-1, 2:] - p[1:-1, :-2])
    return y, x


def _noise_model(a, mu):
    v = .15 + .026 * np.maximum(mu - 10., 0.)
    # The mixed second difference on a single sampling grid cancels
    # locally planar intensity. Estimate its power in quiet regions.
    d = .5 * (a[:, :-1, :-1] + a[:, 1:, 1:] - a[:, 1:, :-1] - a[:, :-1, 1:])
    power = _box(np.mean(d*d, axis=0), 2)[1:-1:2, 1:-1:2]
    b = v[1:-2:2, 1:-2:2]
    lum = mu[1:-2:2, 1:-2:2]
    gy, gx = _derivatives(mu)
    flat = _box(gy*gy + gx*gx, 2)[1:-2:2, 1:-2:2] / b
    ratios, means = [], []
    for lo, hi in ((12, 29), (29, 53), (53, 85), (85, 132), (132, 221)):
        sel = (lum >= lo) & (lum < hi)
        if np.count_nonzero(sel) < b.size * .009:
            continue
        f = flat[sel]
        ix = f < min(1., np.quantile(f, .25))
        if np.count_nonzero(ix) < b.size * .002:
            continue
        rat = np.median((power[sel] / b[sel])[ix])
        ratios.append(np.clip(rat / .96, .38, 1.75))
        means.append(float(np.median(lum[sel][ix])))
    if ratios:
        v *= np.interp(mu, means, ratios).astype(np.float32) ** .45
    return v


def _denoise(guide, noise):
    p = np.pad(guide, 1, mode="edge")
    details = (
        .5 * (p[1:-1, 2:] + p[1:-1, :-2]) - guide,
        .5 * (p[2:, 1:-1] + p[:-2, 1:-1]) - guide,
        .5 * (p[2:, 2:] + p[:-2, :-2]) - guide,
        .5 * (p[2:, :-2] + p[:-2, 2:]) - guide,
        _blur(_blur(guide)) - guide,
    )
    # Locally fit a regularized Wiener combination of symmetric
    # predictors. Cholesky factors are arrays on the cell grid.
    lower, rhs = [], []
    for i, x in enumerate(details):
        row = []
        for j in range(i + 1):
            c = _cell_average(x * details[j], 2)
            other = row if i == j else lower[j]
            for k in range(j):
                c -= other[k] * row[k]
            if i == j:
                c += .23 * noise + .03
                np.sqrt(np.maximum(c, .03, out=c), out=c)
            else:
                c /= other[j]
            row.append(c)
        r = noise * (.8 if i < 4 else .664)
        for k in range(i):
            r -= row[k] * rhs[k]
        r /= row[i]
        lower.append(row)
        rhs.append(r)
    for i in range(len(rhs) - 1, -1, -1):
        for j in range(i + 1, len(rhs)):
            rhs[i] -= rhs[j] * lower[j][i]
        rhs[i] /= lower[i][i]
    gain = .98 / np.maximum(np.sum(rhs, axis=0), .98)
    out = guide.copy()
    for x, c in zip(details, rhs):
        out += x * _upsample(c * gain)
    return out


def _spectral_denoise(guide, noise, pilot):
    # Overlapping Hann windows. The directional estimate supplies the local
    # signal spectrum, while the coefficients come from the original guide.
    n, step = 32, 16
    x = np.arange(n, dtype=np.float32)
    win = np.sin((x+.5)*np.float32(np.pi/n))**2
    win = win[:, None]*win
    energy = win*win
    norm = energy + np.roll(energy, step, axis=0)
    norm += np.roll(norm, step, axis=1)
    norm = norm[:step, :step]
    wy = (np.fft.fftfreq(n)*n)**2
    wx = (np.fft.rfftfreq(n)*n)**2
    protect = np.exp(-(wy[:, None] + wx)/2.48).astype(np.float32)
    protect[0, 0] = protect[0, 1] = protect[1, 0] = protect[-1, 0] = 1.

    h, w = guide.shape
    tiles = sliding_window_view(np.pad(guide, step, mode="reflect"), (n, n))[::step, ::step]
    proxy = sliding_window_view(np.pad(pilot, step, mode="reflect"), (n, n))[::step, ::step]
    ng = sliding_window_view(np.pad(noise, step//2, mode="reflect"), (step, step))[::step//2, ::step//2]
    var = ng.mean(axis=(-2, -1))[..., None, None] * (.83*np.sum(energy))
    buf = np.zeros((h+2*step, w), dtype=np.float32)
    for lo in range(0, len(tiles), 8):
        hi = min(lo+8, len(tiles))
        f = np.fft.rfft2(tiles[lo:hi]*win, axes=(-2, -1)).astype(np.complex64)
        f0 = np.fft.rfft2(proxy[lo:hi]*win, axes=(-2, -1)).astype(np.complex64)
        power = f.real*f.real + f.imag*f.imag
        smooth = .54*power + .115*(np.roll(power, 1, axis=-2) + np.roll(power, -1, axis=-2)
                                  + np.roll(power, 1, axis=-1) + np.roll(power, -1, axis=-1))
        nv = var[lo:hi]
        prior = .18*np.maximum(smooth-nv, 0.) + .82*(f0.real*f0.real + f0.imag*f0.imag)
        f *= np.maximum(prior/(prior + 1.65*nv + .001), protect)
        t = np.fft.irfft2(f, s=(n, n), axes=(-2, -1)).astype(np.float32)
        t *= win
        r = t[:, 1:, :, :step] + t[:, :-1, :, step:]
        a, b = lo*step, hi*step
        buf[a:b] += r[:, :, :step, :].transpose(0, 2, 1, 3).reshape(b-a, w)
        buf[a+step:b+step] += r[:, :, step:, :].transpose(0, 2, 1, 3).reshape(b-a, w)
    result = buf[step:-step]
    result.reshape(h//step, step, w//step, step)[:] /= norm[:, None, :]
    return pilot + .77*(result-pilot)


def _relative_model(b, mu, noise):
    # Fit gain and first-order translation on the aligned, mid-band
    # angle planes. Each tile is shrunk toward a common scene model.
    span = 16

    def block(z):
        h, w = z.shape[-2:]
        return (
            z.reshape(-1, h//span, span, w//span, span)
            .transpose(1, 3, 2, 4, 0)
            .reshape(h//span, w//span, span*span, -1)
        )

    s = mu - _blur(mu)
    y, x = _derivatives(s)
    g = block(np.stack((.5*y, .5*x, s)))
    d = block(b - _blur(b) - s)
    gt = g.swapaxes(-1, -2)
    gg = (gt @ g) / (span*span)
    gd = (gt @ d) / (span*span)
    gl = gg.mean(axis=(0, 1))
    gl.flat[::4] += .03
    prior = np.clip(np.linalg.solve(gl, gd.mean(axis=(0, 1))), -.60, .60)
    v = .13 + .03 * block(noise).mean(axis=(-2, -1))
    for i in range(3):
        gg[..., i, i] += v
    gd += v[..., None, None] * prior
    co = prior + .7 * (np.linalg.solve(gg, gd) - prior)
    maps = co.transpose(2, 3, 0, 1).copy()
    np.clip(maps[:2], -.7, .7, out=maps[:2])
    np.clip(maps[2], -.32, .32, out=maps[2])
    return maps


def _lift_tiles(maps, shape):
    h, w = shape
    y = (np.arange(h, dtype=np.float32) - 15.5) / 32.
    x = (np.arange(w, dtype=np.float32) - 15.5) / 32.
    y = np.clip(y, 0., maps.shape[-2]-1)
    x = np.clip(x, 0., maps.shape[-1]-1)
    i, j = y.astype(np.intp), x.astype(np.intp)
    fy, fx = y - i.astype(np.float32), x - j.astype(np.float32)
    z = maps[..., j] * (1.-fx) + maps[..., np.minimum(j+1, maps.shape[-1]-1)] * fx
    return z[..., i, :] * (1.-fy[:, None]) + z[..., np.minimum(i+1, maps.shape[-2]-1), :] * fy[:, None]


def _register(b):
    # Correlate high-passed planes on a subsampled grid, using a
    # quadratic peak fit for fractional displacements.
    rad, stride, margin = 4, 5, 10
    v = b - _blur(_blur(b))
    h, w = v.shape[-2:]
    t = v[0, margin:h-margin:stride, margin:w-margin:stride]
    power = _box(v[0]*v[0], 3)[margin:h-margin:stride, margin:w-margin:stride]
    wt = 1. / np.sqrt(power + .15)
    tw = t * wt
    vv = np.sum(tw*t)

    def score(z, dy, dx):
        q = z[margin+dy:h-margin+dy:stride, margin+dx:w-margin+dx:stride]
        return np.sum(tw*q) / np.sqrt(vv*np.sum(wt*q*q) + 1.e-12)

    before = np.array([score(z, 0, 0) for z in v[1:]])
    if before.min() > .92:
        return None
    shifts, gains, strengths = [np.zeros(2)], [], []
    for c, z in enumerate(v[1:]):
        cor = np.array([[score(z, dy, dx) for dx in range(-rad, rad+1)]
                        for dy in range(-rad, rad+1)], dtype=np.float64)
        i, j = np.unravel_index(cor.argmax(), cor.shape)
        best = cor[i, j]
        d = np.array((i-rad, j-rad), dtype=np.float64)
        if 0 < i < 2*rad and 0 < j < 2*rad:
            grad = .5 * np.array((cor[i+1, j]-cor[i-1, j], cor[i, j+1]-cor[i, j-1]))
            cross = .25 * (cor[i+1, j+1]-cor[i+1, j-1]-cor[i-1, j+1]+cor[i-1, j-1])
            mat = np.array(((cor[i+1, j]+cor[i-1, j]-2*best, cross),
                            (cross, cor[i, j+1]+cor[i, j-1]-2*best)))
            if np.linalg.det(mat) > 5.e-6:
                d -= np.clip(np.linalg.solve(mat, grad), -.65, .65)
        shifts.append(d)
        strengths.append(best)
        gains.append(best - before[c])
    shifts = np.stack(shifts)
    gain = max(gains)
    if min(strengths) < .73 or gain < .065 or np.max(np.abs(shifts)) < .30:
        return None
    return shifts, v, float(np.clip((gain - .02) / .17, 0., 1.))


def _translate(a, delta):
    pad = 7
    h, w = a.shape
    p = np.pad(a, pad, mode="edge")

    def taps(x):
        i = int(np.floor(x))
        t = x - i
        t2, t3 = t*t, t*t*t
        return pad+i-1, (-.5*t+t2-.5*t3, 1-2.5*t2+1.5*t3,
                         .5*t+2*t2-1.5*t3, -.5*t2+.5*t3)

    i, ky = taps(delta[0])
    j, kx = taps(delta[1])
    z = sum(k*p[:, j+s:j+s+w] for s, k in enumerate(kx))
    return sum(k*z[i+s:i+s+h] for s, k in enumerate(ky))


def _motion_filters(shifts, target, phase, levels, chroma):
    # Kriging after registration, with an independent DC level per
    # angle and a common linear ramp in the registered coordinates.
    offsets = []
    labels = np.repeat(np.arange(4), 9)
    here = np.array(_PHASES[phase])
    for j, p in enumerate(_PHASES):
        q = np.array(p) - here
        q += 2 * np.round((shifts[j] - shifts[target] - q) * .5).astype(np.int32)
        offsets.extend((q[0]+2*y, q[1]+2*x) for y in (-1, 0, 1) for x in (-1, 0, 1))
    offsets = np.array(offsets, dtype=np.int32)
    coords = offsets - shifts[labels] + shifts[target]
    mask = labels[:, None] == labels[None, :]
    s = np.exp(-np.sum((coords[:, None, :] - coords[None, :, :])**2, axis=-1) / (2*1.47**2))
    d = (mask - .25) * np.exp(-np.sum(
        (offsets[:, None, :] - offsets[None, :, :])**2, axis=-1) / (2*2.1**2))
    s0 = np.exp(-np.sum(coords**2, axis=-1) / (2*1.47**2))
    d0 = ((labels == target) - .25) * np.exp(-np.sum(offsets**2, axis=-1) / (2*2.1**2))
    f = np.column_stack((labels[:, None] == np.arange(4)[None, :], coords))
    goal = np.zeros((6, 1))
    goal[target, 0] = 1.
    sv, dv = levels[:, None, None, None], chroma[None, :, None, None]
    cov = np.eye(len(offsets)) + sv*s + dv*d
    cross = sv*s0[:, None] + dv*d0[:, None]
    def fit(a, k, trend):
        aug = np.concatenate((k, np.broadcast_to(trend, k.shape[:-1] + (6,))), axis=-1)
        z = np.linalg.solve(a, aug)
        coeff, drift = z[..., :1], z[..., 1:]
        adjust = np.linalg.solve(trend.T @ drift, goal - trend.T @ coeff)
        return coeff + drift @ adjust

    trial = fit(cov, cross, f)[..., 0]
    importance = np.sqrt(np.mean(trial*trial, axis=(0, 1))) + .4*np.max(abs(trial), axis=(0, 1))
    order = np.argsort(-importance)
    use = np.zeros(len(offsets), dtype=bool)
    for c in range(4):
        use[order[labels[order] == c][:4]] = True
    use[order[~use[order]][:26-np.count_nonzero(use)]] = True
    offsets = offsets[use]
    filters = fit(cov[..., use, :][..., use], cross[..., use, :], f[use])
    table = np.ascontiguousarray(filters.reshape(-1, len(offsets)).T, dtype=np.float32)
    return offsets, table


def _motion_fusion(raw, out, a, state):
    sh, band, amount = state
    registered = np.stack([_translate(band[c], sh[c]) for c in range(4)])
    shared = registered.mean(axis=0)
    ss = _box(shared*shared, 3)
    dd = _box(np.mean((registered-shared)**2, axis=0), 3)
    noise = .15 + .028 * np.maximum(a.mean(axis=0) - 10., 0.)
    agree = ss / (ss + dd + .13*noise)
    strengths = np.array((.55, 2.3, 8., 25., 80., 230.))
    chroma = np.array((.32, 2.5, 12.))
    span = float(np.max(np.ptp(sh, axis=0))) * 2
    pad = max(24, int(np.ceil(span)) + 4)
    h, w = raw.shape
    buf = np.pad(raw.astype(np.float32), pad, mode="reflect")
    edge = max(6, int(np.ceil(span)) + 2)
    y = np.arange(h, dtype=np.float32)
    x = np.arange(w, dtype=np.float32)
    edge_y = np.minimum(1., np.minimum(y, h-1-y) / edge)
    edge_x = np.minimum(1., np.minimum(x, w-1-x) / edge)

    for c in range(4):
        v = np.maximum(_translate(ss, -sh[c]), 0.)
        d = np.maximum(_translate(dd, -sh[c]) - .044*noise, 0.)
        g = _translate(agree, -sh[c])
        sl = np.searchsorted(np.sqrt(strengths[:-1]*strengths[1:]), 9.*v/noise)
        dl = np.searchsorted(np.sqrt(chroma[:-1]*chroma[1:]), 5.76*d/noise)
        which = sl*len(chroma) + dl
        weight = _upsample(amount * np.clip((g-.42)/.38, 0., 1.) * (v/(v+.085*noise)))
        weight *= edge_y[:, None]
        weight *= edge_x
        for m, (i, j) in enumerate(_PHASES):
            if m == c:
                continue
            pts, table = _motion_filters(2*sh, c, m, strengths, chroma)
            dest = out[_ORDER.index(c), i::2, j::2]
            for row in range(0, h//2, 64):
                end = min(row+64, h//2)
                r0, r1 = pad+2*row+i, pad+2*end+i
                wt = weight[2*row+i:2*end:2, j::2]
                if not np.any(wt > .001):
                    continue
                q = np.stack([buf[r0+y:r1+y:2, pad+j+x:pad+w+x:2]
                              for y, x in pts]).reshape(len(pts), -1)
                k = np.take(table, which[row:end].ravel(), axis=1)
                p = np.einsum("cn,cn->n", q, k).reshape(wt.shape)
                dest[row:end] += wt * (p - dest[row:end])
    np.clip(out, 0., 255., out=out)


def _quick_denoise(guide, noise):
    base = _blur(guide)
    detail = guide - base
    t = .77*noise / (_cell_average(detail*detail, 1) + .10)
    mid = base - _blur(_blur(base))
    tm = .035*noise / (_cell_average(mid*mid, 1) + .012)
    return guide - detail*_upsample(np.minimum(1., t)) - mid*_upsample(np.minimum(.95, tm))


def reconstruct(raw):
    """Return the 0, 45, 90 and 135 degree planes as float32."""
    raw = np.asarray(raw, dtype=np.uint8, order="C")
    a = np.stack([raw[i::2, j::2] for i, j in _PHASES]).astype(np.float32)
    lo = a.min(axis=(1, 2))
    if np.array_equal(lo, a.max(axis=(1, 2))):
        out = np.empty((4, *raw.shape), dtype=np.float32)
        out[:] = lo[list(_ORDER), None, None]
        return out
    b = _align(a, 1.65)
    mu = b.mean(axis=0)
    motion = _register(b)
    noise = _noise_model(a, mu) if motion is None else .15 + .026 * np.maximum(mu-10., 0.)

    # Angle disagreement in a mid-frequency band determines whether local
    # structure is shared or each angle should instead be interpolated alone.
    wide = _align(a, 2.65)
    band = _align(a, 1.18) - wide
    mid = b - wide
    shared = band.mean(axis=0)
    shared_mid = mid.mean(axis=0)
    band -= shared
    mid -= shared_mid
    common, coherent, kblend = (.25, 1.2, .43) if motion is None else (.43, .912, .34)
    v_s = np.maximum(common*_box(shared*shared, 3) + coherent*_box(shared*shared_mid, 3), 0.)
    v_d = np.maximum(_box(np.mean(band*(common*band + coherent*mid), axis=0), 3) - .10*noise, 0.)
    w = (v_d / (v_d + kblend * (v_s + .24 * noise) + .09)) ** .8
    mix = _upsample(np.clip((w - .13) / .65, 0., 1.))

    # Broaden the offset estimates in parts of the image without
    # convincing angle-dependent detail.
    offsets = b - mu
    if motion is None:
        excess = np.maximum(_box(np.mean(mid * mid, axis=0), 2) - .025 * noise, 0.)
        taper = 1. - excess / (excess + .03 + .04 * noise + .04 * v_s)
        offsets -= .7 * (taper * mid)
    off = _upsample(offsets)
    guide = raw.astype(np.float32)
    for c, (i, j) in enumerate(_PHASES):
        guide[i::2, j::2] -= off[c, i::2, j::2]
    if motion is None:
        guide = _spectral_denoise(guide, noise, _denoise(guide, noise))
        rel = _relative_model(b, mu, noise)
        hi = guide - _upsample(mu)
        dy, dx = _derivatives(hi)
        corrections = []
        for c, (i, j) in enumerate(_PHASES):
            ratio = _upsample(np.clip(offsets[c] / (mu + 10.), -.35, .35))
            ry, rx, rs = _lift_tiles(rel[:, c], raw.shape)
            corr = ry*dy + rx*dx + (rs + .35*ratio)*hi
            corrections.append(corr)
            guide[i::2, j::2] -= .65 * corr[i::2, j::2]
    else:
        guide = _quick_denoise(guide, noise)

    out = np.empty((4, *raw.shape), dtype=np.float32)
    for k, c in enumerate(_ORDER):
        i, j = _PHASES[c]
        f = out[k]
        np.add(off[c], guide, out=f)
        if motion is None:
            f += corrections[c]
            hf = a[c] - _blur(a[c])
            gain = np.minimum(1., .375 * noise / (_box(hf * hf, 2) + .12))
            ind = _cubic(a[c] - hf*gain, i, j)
        else:
            ind = _cubic(a[c], i, j)
        f += (ind - f) * mix
        f[i::2, j::2] = raw[i::2, j::2]
    np.clip(out, 0., 255., out=out)
    if motion is not None:
        _motion_fusion(raw, out, a, motion)
    return out
