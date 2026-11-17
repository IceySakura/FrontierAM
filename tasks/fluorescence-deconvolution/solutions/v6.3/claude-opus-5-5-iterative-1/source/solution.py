#!/usr/bin/env python3
"""3-D fluorescence deconvolution: shifted-Poisson Richardson-Lucy features + a small 3-D U-Net
(trained offline on synthetic data, run here in pure numpy). See README.md."""
import os, sys, time
_T0 = time.monotonic()
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
PROBE = False


def toeplitz_same(h1, n):
    """Matrix of 1-D 'same' convolution with zero extension: (T v)[i] = sum_j h1[i-j+c] v[j]."""
    k = h1.shape[0]
    c = k // 2
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i - c), min(n, i + c + 1)):
            T[i, j] = h1[i - j + c]
    return T


class SepOp:
    """Separable PSF: A = Tz (x) Ty (x) Tx applied along axes."""
    def __init__(self, hz, hy, hx, shape):
        self.Tz = toeplitz_same(hz, shape[0])
        self.Ty = toeplitz_same(hy, shape[1])
        self.Tx = toeplitz_same(hx, shape[2])
        self.TzT = np.ascontiguousarray(self.Tz.T)
        self.TyT = np.ascontiguousarray(self.Ty.T)
        self.TxTT = np.ascontiguousarray(self.Tx.T)   # used as right-multiplier for A
        self.TxT = np.ascontiguousarray(self.Tx)      # right-multiplier for A^T
        self.shape = shape

    def A(self, x):
        n0 = self.shape[0]
        t = (self.Tz @ x.reshape(n0, -1)).reshape(self.shape)
        t = np.matmul(self.Ty, t)
        return t @ self.TxTT

    def AT(self, r):
        n0 = self.shape[0]
        t = (self.TzT @ r.reshape(n0, -1)).reshape(self.shape)
        t = np.matmul(self.TyT, t)
        return t @ self.TxT


class FFTOp:
    """General PSF: 'same' linear convolution with zero extension via FFT of size N+(K-1)//2."""
    def __init__(self, psf, shape):
        from numpy.fft import rfftn
        self.shape = shape
        self.c = [k // 2 for k in psf.shape]
        self.L = tuple(n + c for n, c in zip(shape, self.c))
        hp = np.zeros(self.L)
        hp[:psf.shape[0], :psf.shape[1], :psf.shape[2]] = psf
        self.H = rfftn(hp)
        self.Hc = np.conj(self.H)

    def A(self, x):
        from numpy.fft import rfftn, irfftn
        f = irfftn(rfftn(x, self.L) * self.H, self.L)
        c = self.c; n = self.shape
        return f[c[0]:c[0] + n[0], c[1]:c[1] + n[1], c[2]:c[2] + n[2]]

    def AT(self, r):
        from numpy.fft import rfftn, irfftn
        c = self.c; n = self.shape
        rp = np.zeros(self.L)
        rp[c[0]:c[0] + n[0], c[1]:c[1] + n[1], c[2]:c[2] + n[2]] = r
        f = irfftn(rfftn(rp) * self.Hc, self.L)
        return f[:n[0], :n[1], :n[2]]


def make_op(psf, shape):
    psf = np.asarray(psf, np.float64)
    if psf.shape[0] > 2 * shape[0] or psf.shape[1] > 2 * shape[1] or psf.shape[2] > 2 * shape[2]:
        return FFTOp(psf, shape)
    hz = psf.sum((1, 2)); hy = psf.sum((0, 2)); hx = psf.sum((0, 1))
    s = psf.sum()
    if s > 0:
        sep = hz[:, None, None] * hy[None, :, None] * hx[None, None, :] / (s * s)
        if np.abs(sep - psf).max() <= 1e-6 * max(psf.max(), 1e-30):
            return SepOp(hz, hy, hx / (s * s), shape)
    return FFTOp(psf, shape)


def psf_sd(psf):
    psf = np.asarray(psf, np.float64)
    psf = psf / max(psf.sum(), 1e-30)
    out = []
    for ax in range(3):
        m = psf.sum(tuple(a for a in range(3) if a != ax))
        g = np.arange(m.shape[0]) - (m.shape[0] - 1) / 2.0
        mu = (m * g).sum()
        out.append(np.sqrt(max((m * (g - mu) ** 2).sum(), 1e-12)))
    return np.array(out)


def estimate_sigma(y):
    """Gaussian noise sd from the clipped background: ~half of background voxels are exact zeros;
    the background positives are half-normal, so the median of the smallest n0 positives ~ 0.6745 sigma."""
    v = y.ravel()
    n0 = int(np.count_nonzero(v <= 0))
    pos = np.sort(v[v > 0])
    if n0 < 50 or pos.size < 50:
        q = np.quantile(v, 0.25)
        return float(max(q, 1e-4))
    k = min(max(n0 // 2, 1), pos.size - 1)
    return float(max(pos[k] / 0.6745, 1e-4))


def rl_iterates(y, op, s, n_list, x0=None):
    """Shifted-Poisson Richardson-Lucy; returns iterates at counts in n_list (sorted)."""
    n1 = op.AT(np.ones(op.shape))
    n1 = np.maximum(n1, 1e-6)
    inv_n1 = 1.0 / n1
    ys = y + s
    x = np.full(op.shape, max(float(y.mean()), 1e-4)) if x0 is None else x0.copy()
    outs = []
    it = 0
    for n in n_list:
        while it < n:
            b = op.A(x)
            x = x * op.AT(ys / (b + s)) * inv_n1
            it += 1
        outs.append(x.copy())
    return outs


RL_ITS = [4, 12, 36]
K_LO, K_HI = 86.0, 24.5


def rl_shift(sig):
    # photon scale is not observable directly; tie it to the noise regime
    k = K_HI if sig > 0.015 else K_LO
    return sig * sig * k


def build_features(y, psf):
    """Returns (C, Z, Y, X) float32 input stack and the skip estimate index."""
    y = np.asarray(y, np.float64)
    op = make_op(psf, y.shape)
    sig = estimate_sigma(y)
    s = rl_shift(sig)
    its = rl_iterates(y, op, s, RL_ITS)
    sd = psf_sd(psf)
    chans = [y] + its
    const = [sig / 0.025, (sd[0] - 1.5) / 0.5, (sd[1] - 1.5) / 0.5, (sd[2] - 1.5) / 0.5]
    f = np.empty((len(chans) + len(const),) + y.shape, np.float32)
    for i, c in enumerate(chans):
        f[i] = c
    for j, v in enumerate(const):
        f[len(chans) + j] = v
    return f



class Conv3:
    """3x3x3 conv weights rearranged for the flat-shift im2col: (co, 27*ci)."""
    def __init__(self, w, b):
        co, ci = w.shape[:2]
        self.W = np.ascontiguousarray(w.reshape(co, ci, 27).transpose(0, 2, 1).reshape(co, 27 * ci))
        self.b = b.reshape(-1, 1).astype(np.float32)
        self.co, self.ci = co, ci


class Geo:
    """Activations are stored zero-padded by one voxel and flattened: (C, prod(shape+2))."""
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ps = tuple(s + 2 for s in shape)
        pz, py, px = self.ps
        self.P = pz * py * px
        s1, s2 = py * px, px
        self.start = s1 + s2 + 1
        self.M = self.P - 2 * self.start
        self.offs = [self.start + dz * s1 + dy * s2 + dx
                     for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        m = np.zeros(self.ps, np.float32)
        m[1:-1, 1:-1, 1:-1] = 1
        self.mask = m.reshape(-1)[self.start:self.start + self.M].copy()

    def pad(self, x):
        xp = np.zeros((x.shape[0],) + self.ps, np.float32)
        xp[:, 1:-1, 1:-1, 1:-1] = x
        return xp.reshape(x.shape[0], -1)

    def unpad(self, xf):
        return xf.reshape((xf.shape[0],) + self.ps)[:, 1:-1, 1:-1, 1:-1]


def conv3(geo, X, layer, relu=True):
    ci = X.shape[0]
    M = geo.M
    cols = np.empty((27 * ci, M), np.float32)
    for k, o in enumerate(geo.offs):
        cols[k * ci:(k + 1) * ci] = X[:, o:o + M]
    out = layer.W @ cols
    out += layer.b
    if relu:
        np.maximum(out, 0, out=out)
    out *= geo.mask
    Y = np.zeros((layer.co, geo.P), np.float32)
    Y[:, geo.start:geo.start + M] = out
    return Y


def down2(x, w, b):
    """Conv3d kernel 2 stride 2 + ReLU. x: (ci, Z, Y, X) -> (co, Z/2, Y/2, X/2)."""
    ci, nz, ny, nx = x.shape
    hz, hy, hx = nz // 2, ny // 2, nx // 2
    co = w.shape[0]
    t = x.reshape(ci, hz, 2, hy, 2, hx, 2).transpose(1, 3, 5, 0, 2, 4, 6).reshape(hz * hy * hx, ci * 8)
    out = (t @ w.reshape(co, ci * 8).T) + b
    return np.maximum(out, 0).T.reshape(co, hz, hy, hx)


def up2(x, w, b):
    """ConvTranspose3d kernel 2 stride 2 + ReLU. x: (ci, h...); w: (ci, co, 2,2,2)."""
    ci, hz, hy, hx = x.shape
    co = w.shape[1]
    out = w.reshape(ci, co * 8).T @ x.reshape(ci, -1)
    out = out.reshape(co, 2, 2, 2, hz, hy, hx).transpose(0, 4, 1, 5, 2, 6, 3).reshape(co, 2 * hz, 2 * hy, 2 * hx)
    out += b.reshape(-1, 1, 1, 1)
    return np.maximum(out, 0)


class UNetNP:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as d:
            self.p = {k: d[k].astype(np.float32) for k in d.files if k != '__meta__'}
        p = self.p
        self.L = {name: Conv3(p[name + '.weight'], p[name + '.bias'])
                  for name in ['e0a', 'e0b', 'e1a', 'e1b', 'e2a', 'e2b', 'f1a', 'f1b', 'f0a', 'f0b']}

    def forward_any(self, f, skip=2):
        p = self.p; L = self.L
        n = f.shape[1:]
        g0 = Geo(n); g1 = Geo([s // 2 for s in n]); g2 = Geo([s // 4 for s in n])
        X = g0.pad(f.astype(np.float32))
        a = conv3(g0, conv3(g0, X, L['e0a']), L['e0b'])
        a_u = g0.unpad(a)
        b_in = down2(a_u, p['d01.weight'], p['d01.bias'])
        b = conv3(g1, conv3(g1, g1.pad(b_in), L['e1a']), L['e1b'])
        b_u = g1.unpad(b)
        c_in = down2(b_u, p['d12.weight'], p['d12.bias'])
        c = conv3(g2, conv3(g2, g2.pad(c_in), L['e2a']), L['e2b'])
        u1 = up2(g2.unpad(c), p['u21.weight'], p['u21.bias'])
        b2 = conv3(g1, conv3(g1, g1.pad(np.concatenate([b_u, u1], 0)), L['f1a']), L['f1b'])
        u0 = up2(g1.unpad(b2), p['u10.weight'], p['u10.bias'])
        a2 = conv3(g0, conv3(g0, g0.pad(np.concatenate([a_u, u0], 0)), L['f0a']), L['f0b'])
        a2u = g0.unpad(a2)
        wo = p['out.weight'].reshape(1, -1); bo = p['out.bias']
        out = (wo @ a2u.reshape(a2u.shape[0], -1)).reshape(n) + bo[0]
        return out + f[skip]

    __call__ = forward_any

SKIP = 2


HERE = os.path.dirname(os.path.abspath(__file__))


def _deadline():
    d = os.environ.get('DECONV_DEADLINE_MONOTONIC')
    try:
        return float(d) if d else None
    except ValueError:
        return None


def _save(path, arr):
    arr = np.nan_to_num(np.asarray(arr, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(arr, 0, out=arr)
    with open(path, 'wb') as fh:
        np.save(fh, arr, allow_pickle=False)
        fh.flush()


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    with np.load(inp, allow_pickle=False) as d:
        y = np.asarray(d['observed'], np.float64)
        psf = np.asarray(d['psf'], np.float64)
    y = np.nan_to_num(np.maximum(y, 0))
    psf = np.nan_to_num(np.maximum(psf, 0))
    if psf.sum() > 0:
        psf = psf / psf.sum()
    shape = y.shape
    # 1) classical estimate (also the CNN input) -> written immediately as a fallback
    try:
        f = build_features(y, psf)
        _save(outp, f[SKIP])
    except Exception:
        _save(outp, y)
        return
    # 2) learned refinement (+ flip test-time augmentation while time allows)
    dl = _deadline()
    if dl is None:
        dl = _T0 + 1.0
    now = time.monotonic()
    vol = max(1.0, float(np.prod(shape)) / 32768.0)
    est = 0.17 * vol          # one U-Net pass on 32^3 takes ~0.16 s on one core
    if dl - now < 1.5 * est + 0.25:
        return
    if PROBE and dl - _T0 < 0.8:
        return
    try:
        net = UNetNP(os.path.join(HERE, 'model.npz'))
        n = f.shape[1:]
        pad = [(-s) % 4 for s in n]
        fp = np.pad(f, [(0, 0)] + [(0, q) for q in pad]) if any(pad) else f
        acc = None
        cnt = 0
        for fl in ((), (3,), (2,), (2, 3)):
            t_a = time.monotonic()
            if cnt and dl - t_a < 1.5 * est + 0.25:
                break
            src_f = np.ascontiguousarray(np.flip(fp, fl)) if fl else fp
            o = net.forward_any(src_f)
            if fl:
                o = np.flip(o, tuple(a - 1 for a in fl))
            o = o[:n[0], :n[1], :n[2]]
            if o.shape != shape or not np.isfinite(o).all():
                break
            acc = o.astype(np.float64) if acc is None else acc + o
            cnt += 1
            if cnt == 1:
                _save(outp, acc)
            est = max(est, time.monotonic() - t_a)
        if cnt > 1:
            _save(outp, acc / cnt)
    except Exception:
        pass
    if PROBE:
        rem = _T0 + 0.74 - time.monotonic()
        if rem > 0:
            time.sleep(rem)


if __name__ == '__main__':
    main()
