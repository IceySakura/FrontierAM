import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy import ndimage

MASKS = [(1, 1), (0, 1), (0, 0), (1, 0)]
SIGMA2 = 64.0
SPATIAL = 2.5
RADIUS = 4
LAP = -0.12


def _offsets(pr, pc, qr, qc):
    offs = []
    for dy in range(-RADIUS, RADIUS + 1):
        for dx in range(-RADIUS, RADIUS + 1):
            if dy == 0 and dx == 0:
                continue
            if (qr + dy) % 2 != pr or (qc + dx) % 2 != pc:
                continue
            d2 = dy * dy + dx * dx
            if d2 > RADIUS * RADIUS + 1:
                continue
            offs.append((dy, dx, d2))
    return offs


OFFSETS = {}
for pr, pc in MASKS:
    for qr in range(2):
        for qc in range(2):
            if (qr, qc) != (pr, pc):
                OFFSETS[(pr, pc, qr, qc)] = _offsets(pr, pc, qr, qc)


def _half(arr, axis, offset):
    n = arr.shape[axis] - 4
    if axis == 0:
        a = arr[:n, 2:-2]
        b = arr[1:n + 1, 2:-2]
        c = arr[2:n + 2, 2:-2]
        d = arr[3:n + 3, 2:-2]
        e = arr[4:n + 4, 2:-2]
        if offset < 0:
            return (-a + 9 * b + 9 * c - d) * (1.0 / 16.0)
        return (-b + 9 * c + 9 * d - e) * (1.0 / 16.0)
    a = arr[2:-2, :n]
    b = arr[2:-2, 1:n + 1]
    c = arr[2:-2, 2:n + 2]
    d = arr[2:-2, 3:n + 3]
    e = arr[2:-2, 4:n + 4]
    if offset < 0:
        return (-a + 9 * b + 9 * c - d) * (1.0 / 16.0)
    return (-b + 9 * c + 9 * d - e) * (1.0 / 16.0)


def _cubic(x):
    out = np.empty((4, *x.shape), np.float32)
    for k, (r, c) in enumerate(MASKS):
        R = x[r::2, c::2]
        Rp = np.pad(R, 2, mode="reflect")
        roff = -0.5 if r == 1 else 0.5
        coff = -0.5 if c == 1 else 0.5
        V = _half(Rp, 0, roff)
        H = _half(Rp, 1, coff)
        C = _half(np.pad(V, 2, mode="reflect"), 1, coff)
        rr, cc = r ^ 1, c ^ 1
        out[k, r::2, c::2] = x[r::2, c::2]
        out[k, rr::2, c::2] = V
        out[k, r::2, cc::2] = H
        out[k, rr::2, cc::2] = C
    return out


def _bilinear(x):
    out = np.empty((4, *x.shape), np.float32)
    for k, (r, c) in enumerate(MASKS):
        R = x[r::2, c::2]
        Rp = np.pad(R, 1, mode="symmetric")
        h2, w2 = R.shape
        if r == 1:
            v0 = Rp[:h2, 1:w2 + 1]
            v1 = Rp[1:h2 + 1, 1:w2 + 1]
        else:
            v0 = Rp[1:h2 + 1, 1:w2 + 1]
            v1 = Rp[2:h2 + 2, 1:w2 + 1]
        V = (v0 + v1) * 0.5
        if c == 1:
            h0 = Rp[1:h2 + 1, :w2]
            h1 = Rp[1:h2 + 1, 1:w2 + 1]
        else:
            h0 = Rp[1:h2 + 1, 1:w2 + 1]
            h1 = Rp[1:h2 + 1, 2:w2 + 2]
        H = (h0 + h1) * 0.5
        ri0 = 0 if r == 1 else 1
        ri1 = 1 if r == 1 else 2
        ci0 = 0 if c == 1 else 1
        ci1 = 1 if c == 1 else 2
        C = (Rp[ri0:ri0 + h2, ci0:ci0 + w2] + Rp[ri0:ri0 + h2, ci1:ci1 + w2] +
             Rp[ri1:ri1 + h2, ci0:ci0 + w2] + Rp[ri1:ri1 + h2, ci1:ci1 + w2]) * 0.25
        rr, cc = r ^ 1, c ^ 1
        out[k, r::2, c::2] = x[r::2, c::2]
        out[k, rr::2, c::2] = V
        out[k, r::2, cc::2] = H
        out[k, rr::2, cc::2] = C
    return out


def _metric(base):
    hp = base - ndimage.gaussian_filter(base, (0, 1, 1), mode="mirror")
    hp -= hp.mean(axis=(1, 2), keepdims=True)
    norms = np.sqrt(np.einsum("cij,cij->c", hp, hp))
    flat = hp.reshape(4, -1)
    dots = np.einsum("ci,ji->cj", flat, flat)
    corr = dots / np.maximum(norms[:, None] * norms[None, :], 1e-12)
    return float((corr.sum() - 4.0) / 12.0)


def reconstruct(raw):
    x = raw.astype(np.float32)
    h, w = x.shape
    base = _bilinear(x)
    cubic = _cubic(x)
    met = _metric(base)
    alpha = np.clip((met - 0.05) / 0.30, 0.0, 1.0)
    if alpha <= 0.001:
        return cubic
    guide = base.mean(axis=0)
    out = _bilateral(x, guide, spatial=2.5, max_d2=10)
    if alpha > 0.5:
        guide2 = 0.75 * guide + 0.25 * out.mean(axis=0)
        out = _bilateral(x, guide2, spatial=3.0, max_d2=17)
        out += LAP * ndimage.laplace(guide2)
    else:
        out += LAP * ndimage.laplace(guide)
    for k, (r, c) in enumerate(MASKS):
        out[k, r::2, c::2] = x[r::2, c::2]
    if alpha >= 0.999:
        return np.nan_to_num(out, copy=False)
    blended = np.asarray(alpha * out + (1.0 - alpha) * cubic, dtype=np.float32)
    for k, (r, c) in enumerate(MASKS):
        blended[k, r::2, c::2] = x[r::2, c::2]
    return np.nan_to_num(blended, copy=False)


def _bilateral(x, guide, spatial=SPATIAL, max_d2=17, sigma2=SIGMA2):
    h, w = x.shape
    xp = np.pad(x, RADIUS, mode="reflect")
    gp = np.pad(guide, RADIUS, mode="reflect")
    h2, w2 = h // 2, w // 2
    out = np.empty((4, h, w), np.float32)
    inv_sigma2 = 1.0 / sigma2
    inv_spatial2 = 1.0 / (2.0 * spatial * spatial)
    for k, (pr, pc) in enumerate(MASKS):
        plane = guide.copy()
        plane[pr::2, pc::2] = x[pr::2, pc::2]
        for qr in range(2):
            for qc in range(2):
                if (qr, qc) == (pr, pc):
                    continue
                num = np.zeros((h2, w2), np.float32)
                den = np.zeros((h2, w2), np.float32)
                d = np.empty((h2, w2), np.float32)
                tmp = np.empty((h2, w2), np.float32)
                gc = gp[RADIUS + qr:RADIUS + qr + 2 * h2:2, RADIUS + qc:RADIUS + qc + 2 * w2:2]
                for dy, dx, d2 in OFFSETS[(pr, pc, qr, qc)]:
                    if d2 > max_d2:
                        continue
                    ws = float(np.exp(-d2 * inv_spatial2))
                    gn = gp[RADIUS + qr + dy:RADIUS + qr + dy + 2 * h2:2, RADIUS + qc + dx:RADIUS + qc + dx + 2 * w2:2]
                    xn = xp[RADIUS + qr + dy:RADIUS + qr + dy + 2 * h2:2, RADIUS + qc + dx:RADIUS + qc + dx + 2 * w2:2]
                    np.subtract(gn, gc, out=d)
                    np.square(d, out=d)
                    d *= inv_sigma2
                    d += 1.0
                    np.reciprocal(d, out=d)
                    d *= ws
                    np.subtract(xn, gn, out=tmp)
                    np.multiply(d, tmp, out=tmp)
                    num += tmp
                    den += d
                plane[qr::2, qc::2] = guide[qr::2, qc::2] + num / den
        out[k] = plane
    return out
