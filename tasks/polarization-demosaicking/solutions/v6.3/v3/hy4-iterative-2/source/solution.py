"""Polarization demosaicking via learned 2x2-periodic Wiener filtering.

Supports field-mode (3 outputs, exact constrained reconstruction) and
direct-mode (4 outputs, correction + exact measurement injection) filters,
and an optional isotropic radial response mixed into the kernel spectra.
All computation is a single FFT of the padded mosaic plus exact frequency
shifts and a few inverse FFTs; the kernel spectra are formed analytically
from small kernels and precomputed DFT bases.  NumPy only.
"""
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_W = None


def _load():
    global _W
    if _W is None:
        with np.load(os.path.join(_HERE, 'weights.npz')) as z:
            _W = {k: (z[k].item() if z[k].ndim == 0 and z[k].dtype.kind in 'US'
                      else z[k]) for k in z.files}
    return _W


def _fallback(raw):
    """Parameter-free residual interpolation for unexpected inputs."""
    y = raw.astype(np.float32)
    H, W = y.shape
    xp = np.pad(y, ((1, 1), (0, 0)), mode='edge')
    g = (xp[:-2] + 2. * xp[1:-1] + xp[2:]) * .25
    gp = np.pad(g, ((0, 0), (1, 1)), mode='edge')
    G = (gp[:, :-2] + 2. * gp[:, 1:-1] + gp[:, 2:]) * .25
    R = y - G
    out = np.empty((4, H, W), np.float32)
    for a, (r0, c0) in enumerate([(1, 1), (0, 1), (0, 0), (1, 0)]):
        s = R[r0::2, c0::2]
        sp = np.pad(s, ((1, 1), (1, 1)), mode='edge')
        rr = np.arange(H)
        kr = (rr - r0) // 2 + 1
        dr = (rr - r0) % 2
        hr = np.where((dr == 0)[:, None], sp[kr, 1:-1],
                      .5 * (sp[kr, 1:-1] + sp[kr + 1, 1:-1]))
        hp = np.pad(hr, ((0, 0), (1, 1)), mode='edge')
        cc = np.arange(W)
        kc = (cc - c0) // 2 + 1
        dc = (cc - c0) % 2
        up = np.where((dc == 0)[None, :], hp[:, kc],
                      .5 * (hp[:, kc] + hp[:, kc + 1]))
        out[a] = G + up
        out[a, r0::2, c0::2] = y[r0::2, c0::2]
    return out


def reconstruct(raw):
    y = np.ascontiguousarray(raw, np.float32)
    if y.ndim != 2:
        raise ValueError('raw must be a 2-D array')
    w = _load()
    chan = None
    if 'norm' in w and bool(w['norm']):
        Ho0, Wo0 = y.shape
        mu = np.array([y[r0::2, c0::2].mean()
                       for r0, c0 in ((1, 1), (0, 1), (0, 0), (1, 0))], np.float32)
        mm = mu.mean()
        bb = (mu[2] + mu[1] - mu[0] - mu[3]) * .25
        cc = (mu[2] + mu[3] - mu[1] - mu[0]) * .25
        dd = (mu[2] + mu[0] - mu[1] - mu[3]) * .25
        rr = np.arange(Ho0)
        prn = (1. - 2. * (rr % 2)).astype(np.float32)[:, None]
        pcn = (1. - 2. * (np.arange(Wo0) % 2)).astype(np.float32)[None, :]
        y = y - (mm + prn * bb + pcn * cc + (prn * pcn) * dd)
        chan = np.array([mm - bb - cc + dd, mm + bb - cc - dd,
                         mm + bb + cc + dd, mm - bb + cc - dd], np.float32)
    R = int(w['R'])
    PAD = int(w['PAD'])
    Ho, Wo = y.shape
    H, W = Ho + 2 * PAD, Wo + 2 * PAD
    Er, Ec = w['Er'], w['Ec']
    if Er.shape[0] != H or Ec.shape[0] != W // 2 + 1:
        # Unusual image size: rebuild DFT bases, or fall back entirely.
        try:
            K = 2 * R + 1
            u = np.arange(-R, R + 1)
            Er = np.exp(-2j * np.pi * np.outer(np.arange(H), u) / H).astype(np.complex64)
            Ec = np.exp(-2j * np.pi * np.outer(np.arange(W // 2 + 1), u) / W).astype(np.complex64)
            idxr = (-np.arange(H)) % H
            idxr3 = (H // 2 - np.arange(H)) % H
            idxc = W // 2 - np.arange(W // 2 + 1)
        except Exception:
            return _fallback(y)
    else:
        idxr, idxr3, idxc = w['idxr'], w['idxr3'], w['idxc']

    yp = np.pad(y, ((PAD, PAD), (PAD, PAD)), mode='reflect')
    Y = np.fft.rfft2(yp)
    Z = np.empty((4, H, W // 2 + 1), np.complex128)
    Z[0] = Y
    Z[1] = np.roll(Y, H // 2, axis=0)
    Z[2] = np.conj(Y[np.ix_(idxr, idxc)])
    Z[3] = np.conj(Y[np.ix_(idxr3, idxc)])

    ks = w['ksmall'].astype(np.complex64).reshape(-1, 2 * R + 1, 2 * R + 1)
    G = np.matmul(np.matmul(Er[None], ks), Ec.T[None])
    nout = G.shape[0] // 4
    G = G.reshape(nout, 4, H, W // 2 + 1)
    if 'Wv' in w and float(w['lam_rad']):
        nb = int(w['nb'])
        kr = np.minimum(np.arange(H), H - np.arange(H))[:, None].astype(np.float32)
        kc = np.arange(W // 2 + 1)[None, :].astype(np.float32)
        rho = np.sqrt(kr * kr + kc * kc) / (H / 2)
        idx = np.minimum((rho * nb).astype(np.int32), nb - 1)
        for o in range(min(3, nout)):
            G[o, o + 1] += float(w['lam_rad']) * w['Wv'][o][idx]
    O = (G * Z[None]).sum(axis=1)
    corr = np.fft.irfft2(O, s=(H, W), axes=(-2, -1))
    corr = corr[:, PAD:PAD + Ho, PAD:PAD + Wo].astype(np.float32)

    pr = (1. - 2. * (np.arange(Ho) % 2)).astype(np.float32)[:, None]
    pc = (1. - 2. * (np.arange(Wo) % 2)).astype(np.float32)[None, :]
    prc = pr * pc
    bias = w['bias']
    for a in range(nout):
        corr[a] += (bias[a, 0] + bias[a, 1] * pr + bias[a, 2] * pc
                    + bias[a, 3] * prc)

    mode = str(w.get('mode', 'field'))
    out = np.empty((4, Ho, Wo), np.float32)
    if mode == 'direct':
        out[:] = y[None] + corr
        for a, (r0, c0) in enumerate([(1, 1), (0, 1), (0, 0), (1, 0)]):
            out[a, r0::2, c0::2] = y[r0::2, c0::2]
    else:
        b, c, d = corr[0], corr[1], corr[2]
        m = y - (pr * b + pc * c + prc * d)
        out[0] = m - b - c + d
        out[1] = m + b - c - d
        out[2] = m + b + c + d
        out[3] = m - b + c - d
        # remove float round-off at measured sites (exact in real arithmetic)
        for a, (r0, c0) in enumerate(((1, 1), (0, 1), (0, 0), (1, 0))):
            out[a, r0::2, c0::2] = y[r0::2, c0::2]
    if chan is not None:
        out += chan[:, None, None]
    return out
