"""Noise-aware, joint Fourier demosaicking of a polarization mosaic."""

import os
from functools import cache

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import fftfreq, irfft2, rfft2, rfftfreq
from scipy.ndimage import correlate1d, uniform_filter


_PHASES = ((1, 1), (0, 1), (0, 0), (1, 0))
_TILE, _HOP, _BATCH = 96, 80, 3


def _split(raw):
    return np.stack([raw[y::2, x::2] for y in range(2) for x in range(2)])


def _blur(a, kernel):
    a = correlate1d(a, kernel, axis=-1, mode="mirror")
    return correlate1d(a, kernel, axis=-2, mode="mirror")


def _box(a, size):
    return uniform_filter(a, (1,) * (a.ndim - 2) + (size, size), mode="mirror")


def _variance_curve(energy, brightness, light, percentile, scale, floor):
    # The lower envelope rejects blocks whose detail band contains texture.
    keys = np.clip((brightness.ravel() * .125).astype(np.int32), 0, 37)
    order = np.argsort(keys, kind="stable")
    values = energy.ravel()[order]
    edges = np.r_[0, np.cumsum(np.bincount(keys, minlength=38))]
    grid = np.arange(38, dtype=np.float32) * 8
    curve = np.array([
        np.quantile(values[a:b], percentile) if b - a >= 80 else np.nan
        for a, b in zip(edges[:-1], edges[1:])
    ])
    keep = np.isfinite(curve)
    if keep.any():
        curve = np.interp(grid, grid[keep], curve[keep])
        curve = correlate1d(curve, [.2, .6, .2])
        variance = np.interp(light, grid + 4, curve).astype(np.float32)
    else:
        variance = np.full(light.shape, np.median(energy), dtype=np.float32)
    return np.maximum(variance * scale, floor)


def _noise_maps(raw, planes):
    luminance = planes.mean(0)
    light = _box(luminance, 5)
    brightness = _box(luminance, 6)[::2, ::2]
    diagonal = planes[:, ::2, ::2] - planes[:, 1::2, ::2]
    diagonal -= planes[:, ::2, 1::2] - planes[:, 1::2, 1::2]
    energy = _box((diagonal * diagonal).mean(0) * .25, 5)
    coarse = _variance_curve(energy, brightness, light, .22, 1.19, .06)

    # The finer band rejects both DC and the exact period-two carriers.
    kernel = np.array([-1., 8., -27., 48., -42., 0., 42., -48., 27., -8., 1.])
    kernel /= np.linalg.norm(kernel)
    high = _blur(raw, kernel)
    energy = _box(high * high, 12)[::4, ::4]
    fine = _variance_curve(energy, brightness, light, .21, 1.22, .035)
    return light, coarse, fine


def _joint_weight(planes, noise):
    # Shift the four half-resolution lattices to their shared cell center
    # before measuring cross-angle covariance of local detail.
    frac = [-.0703125, .8671875, .2265625, -.0234375]
    high = []
    for k, plane in enumerate(planes):
        for axis, offset in ((1, k & 1), (0, k & 2)):
            plane = correlate1d(
                plane, frac[::-1] if offset else frac, axis=axis,
                origin=0 if offset else -1, mode="mirror",
            )
        low = correlate1d(plane, [.25, .5, .25], axis=1)
        low = correlate1d(low, [.25, .5, .25], axis=0)
        high.append(plane - low)
    high = np.stack(high)
    individual = uniform_filter((high * high).mean(0), size=15)
    common = uniform_filter(high.mean(0) ** 2, size=15)
    n = .46 * noise
    cov = np.maximum((common - individual * .25) * (4. / 3.), 0)
    power = np.maximum(individual - n, 0) + n * .3 + .1
    w = .4 + .59 * np.clip(2.8 * ((cov + n * .1) / power - .18), 0, 1)
    return w.repeat(2, axis=0).repeat(2, axis=1)


def _upsample(a, y, x):
    """Separable cubic interpolation on the measured angle's lattice."""
    h, w = a.shape
    tmp = np.empty((h, 2 * w), np.float32)
    tmp[:, x::2] = a
    if x:
        tmp[:, 2::2] = (a[:, :-1] + a[:, 1:]) * .5
        tmp[:, 0] = a[:, 0]
        col = slice(4, -2, 2)
    else:
        tmp[:, 1:-1:2] = (a[:, :-1] + a[:, 1:]) * .5
        tmp[:, -1] = a[:, -1]
        col = slice(3, -3, 2)
    tmp[:, col] += (a[:, 1:-2] + a[:, 2:-1] - a[:, :-3] - a[:, 3:]) * .0625

    out = np.empty((2 * h, 2 * w), np.float32)
    out[y::2] = tmp
    if y:
        out[2::2] = (tmp[:-1] + tmp[1:]) * .5
        out[0] = tmp[0]
        row = slice(4, -2, 2)
    else:
        out[1:-1:2] = (tmp[:-1] + tmp[1:]) * .5
        out[-1] = tmp[-1]
        row = slice(3, -3, 2)
    out[row] += (tmp[1:-2] + tmp[2:-1] - tmp[:-3] - tmp[3:]) * .0625
    return out


def _spatial_estimate(planes, noise):
    low = _blur(planes, [.135, .73, .135])
    high = planes - low
    energy = _box(high * high, 5)
    clean = low + high * (np.maximum(energy - .65 * noise, 0) / (energy + 1e-6))
    return np.stack([_upsample(clean[2*y + x], y, x) for y, x in _PHASES])


@cache
def _spectral_model():
    n = _TILE
    x = rfftfreq(n).astype(np.float32)[None, :]
    y = fftfreq(n).astype(np.float32)[:, None]
    bank, groups, rings = [], [], []
    for k in range(4):
        xx = (x - (k % 2) * .5 + .5) % 1 - .5
        yy = (y - (k // 2) * .5 + .5) % 1 - .5
        ax, ay = np.abs(xx), np.abs(yy)
        lobes = []
        for width in ((.016, .045, .09, .155, .245) if k == 0 else (.016, .045, .09, .155)):
            lobes.append(np.exp(-.5 * ((ax / width) ** 2 + (ay / width) ** 2)))
            if k == 0 and width > .03:
                for ratio in (2.3 ** -1, 2.3):
                    base = (ax / width * ratio) ** 2 + (ay / width / ratio) ** 2
                    lobes.append(np.exp(-.5 * base))
        for angle in np.arange(8) * (np.pi / 8):
            u = np.cos(angle) * xx + np.sin(angle) * yy
            v = np.cos(angle) * yy - np.sin(angle) * xx
            for length in ((.13, .28) if k == 0 else (.112,)):
                width = .025 if length >= .20 else .029
                lobes.append(np.exp(-.5 * ((u / length) ** 2 + (v / width) ** 2)))
        groups.append(slice(len(bank), len(bank) + len(lobes)))
        bank.extend(lobes)
        radius = np.hypot(xx, yy)
        ring = np.exp(-.5 * ((radius - .159) / .03528) ** 2)
        ring[(radius < .066) | (radius > .2704)] = 0
        rings.append(ring.ravel() / ring.sum())

    # Remove the remaining mean leakage from the spectrum used for fitting.
    weight = 1 - np.exp(-(.5 * n) * (y*y + x*x))
    weight[:2, :2] *= .03
    weight[-1, :2] *= .03
    weight = np.minimum(1., weight * 9.)
    ix, iy = np.arange(0, n, 5), np.arange(0, n // 2 + 1, 5)
    bank = np.ascontiguousarray(bank, dtype=np.float32)
    binned = np.add.reduceat(np.add.reduceat(bank * weight, ix, axis=-2), iy, axis=-1)
    count = np.add.reduceat(np.add.reduceat(weight, ix, axis=-2), iy, axis=-1)
    binned /= np.maximum(count, 1e-6)
    binned = binned.reshape(len(bank), -1)
    basis = np.concatenate((binned, np.ones_like(binned[:1])), axis=0)
    return basis, basis.sum(1), weight, count.ravel(), ix, iy, bank.reshape(len(bank), -1), groups, np.stack(rings)


def _power(f):
    return f.real ** 2 + f.imag ** 2


def _fit_powers(f, noise):
    basis, norms, weight, count, ix, iy, bank, groups, rings = _spectral_model()
    p = _power(f) * (1. / (_HOP * _HOP))
    energy = np.add.reduceat(np.add.reduceat(p * weight, ix, axis=-2), iy, axis=-1)
    energy = energy.reshape(len(f), -1) / np.maximum(count, 1e-6)
    n = noise.reshape(-1, 1) * 1.06

    # Nonnegative fit of four carrier spectra and a white observation floor.
    a = np.maximum((energy @ basis.T) / norms - n, 1e-4)
    a[:, :-1] *= .15
    a[:, -1] = n[:, 0] * .55
    floor = n * .45 + 1e-5
    ev = np.maximum(p.reshape(len(f), -1) @ rings.T - n * .8, 0)
    wide = np.clip((ev[:, 1:] / (ev[:, :1] * .25 + n * 1.25 + .08) - .08) * 2.3, 0, 1)
    gate = np.ones_like(a)
    for k, sl in enumerate(groups[1:]):
        gate[:, sl.start + 4:sl.stop] = .15 + .85 * wide[:, k:k+1]
        gate[:, sl.start + 3] = .76 + .24 * wide[:, k]
    a *= gate
    penalty = np.sqrt(gate)
    for _ in range(14):
        inv = 1. / (a @ basis + floor)
        ratio = ((energy * inv * inv) @ basis.T) / np.maximum(inv @ basis.T, 1e-8)
        a *= np.maximum(ratio, 1e-7) ** .78
        a *= penalty

    parts = [(a[:, sl] @ bank[sl]).reshape(f.shape) for sl in groups]
    white = (a[:, -1:] * 1.08 + floor).reshape(-1, 1, 1)

    # Refine the envelope with neighboring frequency bins, favoring
    # components that already have evidence in that band.
    h = [.08, .22, .4, .22, .08]
    local = correlate1d(correlate1d(p, h, axis=-1, mode="mirror"), h, axis=-2, mode="wrap")
    signal = sum(parts)
    excess = np.maximum(local - white - signal * .5, 0)
    mul = np.clip(1. + .5 * (excess / (signal + white * .08 + 1e-6) - .25), .75, 16.)
    total = signal + white
    ratio = np.clip(local / (total + 1e-6) - 1., -.92, 10.)
    inv = 1. / total
    for k, part in enumerate(parts):
        w = np.clip(1 + .78 * ratio * (part * inv) * (1.24 if k == 0 else 1.), .22, 9.)
        part *= .7 * w + .3 * mul
    return parts, sum(parts) + white


@cache
def _alias_indices():
    n = _TILE
    x = np.arange(n // 4)
    y = (np.arange(n // 2) - n // 4) % n
    ys = (y, (-y) % n, (y + n // 2) % n, (n // 2 - y) % n)
    xs = (x, n // 2 - x, x, n // 2 - x)
    return tuple((slice(None), i[:, None], j) for i, j in zip(ys, xs))


def _coupled_wiener(f, parts, power):
    # The four frequencies in an alias group differ by Nyquist carriers.
    # Fold them together, estimating each off-diagonal covariance once.
    indices = _alias_indices()

    def take(z, i):
        return np.ascontiguousarray(z[indices[i]])

    obs = [take(f, i).conj() if i & 1 else take(f, i) for i in range(4)]
    p = [[take(parts[s ^ i], i) for i in range(4)] for s in range(4)]
    total = [take(power, i) for i in range(4)]
    frac = [[v / t for v, t in zip(row, total)] for row in p]
    estimate = [[v * w for v, w in zip(obs, row)] for row in frac]
    h = [.065, .125, .19, .24, .19, .125, .065]

    def smooth(a):
        a = correlate1d(a, h, axis=-1, mode="nearest")
        return correlate1d(a, h, axis=-2, mode="nearest")

    floor = .24 * (_HOP * _HOP)
    rms = [np.sqrt(np.maximum(smooth(_power(v)), t * floor + 1e-6))
           for v, t in zip(obs, total)]
    roots = [[np.sqrt(v + 1e-6) for v in row] for row in p]
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        f0, f1 = obs[i], obs[j]
        e0, e1 = rms[i], rms[j]
        corr = smooth(f0 * f1.conj()) / (e0 * e1)
        magnitude = np.abs(corr)
        corr.imag *= .92
        corr *= np.maximum(0, np.minimum(magnitude, .58) - .066) * .82 / (magnitude + 1e-7)
        corr[..., 0, :] *= .72
        corr[..., -1, :] *= .72
        corr[..., :, -1] *= .72

        # First-order correction to the diagonal Wiener allocation.
        cov = [row[i] * row[j] for row in roots]
        cov_sum = sum(cov)
        denom = np.maximum(cov_sum, np.sqrt(total[i] * total[j]) * .065)
        b0 = corr * f1 * (e0 / e1) / denom
        b1 = corr.conj() * f0 * (e1 / e0) / denom
        for s, c in enumerate(cov):
            estimate[s][i] += b0 * (c - frac[s][i] * cov_sum)
            estimate[s][j] += b1 * (c - frac[s][j] * cov_sum)

    out = []
    for k, part in enumerate(parts):
        v = f * (part / power)
        for i in range(4):
            z = estimate[k ^ i][i]
            v[indices[i]] = z.conj() if i & 1 else z
        out.append(v)
    return out


@cache
def _window():
    n = _TILE - _HOP
    taper = np.ones(_TILE, np.float32)
    ramp = np.sin((np.arange(n) + .5) * (np.pi / (2*n))).astype(np.float32)
    taper[:n], taper[-n:] = ramp, ramp[::-1]
    return taper[:, None] * taper


def _spectral_estimate(raw, noise):
    n, hop = _TILE, _HOP
    h, w = raw.shape
    pre = n - hop
    posty = (h + pre + hop - 1) // hop * hop - h
    postx = (w + pre + hop - 1) // hop * hop - w
    padded = np.pad(raw, ((pre, posty), (pre, postx)), mode="reflect")
    tiles = sliding_window_view(padded, (n, n))[::hop, ::hop]
    ny, nx = tiles.shape[:2]
    npad = np.pad(noise, ((pre//2, posty//2), (pre//2, postx//2)), mode="reflect")
    nv = sliding_window_view(npad, (n//2, n//2))[::hop//2, ::hop//2].mean((-2, -1))
    accum = np.zeros((4, *padded.shape), dtype=np.float32)
    blocks = sliding_window_view(accum, (n, n), axis=(1, 2), writeable=True)[:, ::hop, ::hop]
    window = _window()
    window_sum = window.sum()
    for start in range(0, ny, _BATCH):
        size = min(_BATCH, ny - start)
        batch = tiles[start:start+size].reshape(-1, n, n)
        center = np.sum(batch * window, axis=(-2, -1), keepdims=True) / window_sum
        f = rfft2((batch - center) * window, workers=1)
        parts, power = _fit_powers(f, nv[start:start+size])
        spectra = _coupled_wiener(f, parts, power)
        for k, spectrum in enumerate(spectra):
            tile = irfft2(spectrum, s=(n, n), workers=1)
            if k == 0:
                tile += center * window
            tile *= window
            tile = tile.reshape(size, nx, n, n)
            for y in range(min(2, size)):
                for x in (0, 1):
                    blocks[k, start+y:start+size:2, x::2] += tile[y::2, x::2]
    modes = accum[:, pre:pre+h, pre:pre+w]
    modes[1::2, :, 1::2] *= -1
    modes[2:, 1::2] *= -1
    l, x, y, xy = modes
    out = np.stack((l-x-y+xy, l-x+y-xy, l+x+y+xy, l+x-y-xy))
    return np.clip(out, 0., 255., out=out)


def reconstruct(raw):
    """Return finite float32 planes in the order (0, 45, 90, 135) degrees."""
    a = np.asarray(raw, dtype=np.float32)
    planes = _split(a)
    light, noise, fine = _noise_maps(a, planes)
    weight = _joint_weight(planes, noise)

    # Work in a mildly variance-stabilized intensity coordinate, including
    # the squared Jacobian when converting the input noise variance.
    offset = np.float32(12.)
    root = np.sqrt(offset)
    scale = np.float32(255.) / (np.sqrt(offset + np.float32(255.)) - root)
    stabilized = (np.sqrt(a + offset) - root) * scale
    n = (.68 * noise + .32 * np.minimum(1.4 * noise, 1.06 * fine))
    n = n * (scale * scale * .25) / (light + offset)
    out = _spectral_estimate(stabilized, n)
    out /= scale
    out += root
    np.square(out, out=out)
    out -= offset

    spatial = _spatial_estimate(planes, noise)
    out -= spatial
    out *= weight
    out += spatial
    for k, (y, x) in enumerate(_PHASES):
        out[k, y::2, x::2] = a[y::2, x::2]
    return np.clip(out, 0., 255., out=out)
