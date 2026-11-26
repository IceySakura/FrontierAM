"""Polarization (DoFP) mosaic demosaicking.

Filter pattern (row, col parities, 0-based):

    even row, even col -> 90 deg      even row, odd col -> 45 deg
    odd  row, even col -> 135 deg     odd  row, odd col -> 0 deg

Each plane is observed on one of the four 2x2 sublattices.  The estimator
splits every plane into a shared intensity field S (the mean of the four
planes) and a per-plane polarization detail D_k = I_k - S:

  1. S is initialised with the mean of four independent cubic (Keys family)
     upsamplings of the observed sublattices,
  2. the detail D_k is observed on the same sublattice as I_k
     (D_k = I_k - S there), so it is upsamplinged with the same cubic,
  3. I_k = S + cubic(D_k);  S is re-estimated from the new planes and the
     step is repeated once.

Averaging the four planes makes S much cleaner than any single plane, and the
detail fields are interpolated from their own samples, which is markedly more
accurate than interpolating each plane independently.  Observed samples are
reproduced exactly by construction and the result is clipped to [0, 255].

The cubic tension adapts to a robust per-scene noise estimate (median second
difference of the lattice samples): cleaner scenes get a slightly sharper
kernel; the detail fields use a marginally softer kernel.  Pure NumPy,
single threaded, ~0.3 s per 1024x1024 frame.
"""
import numpy as np

# lattice parity (row, col) of channel k, channels ordered 0/45/90/135 deg
_OFFS = ((1, 1), (0, 1), (0, 0), (1, 0))


def _cubic_ch(V, a, b, H, W, t, q=0.0):
    """Upsample sublattice array V (offset a,b) to (H,W) with a separable
    Keys cubic of tension t (weights -t, .5+t, .5+t, -t).  q blends the cubic
    taps towards the noise-averaging 2-tap/box taps (0 = pure cubic)."""
    n, m = V.shape
    w0, w1, w2, w3 = -t, 0.5 + t, 0.5 + t, -t
    Vp = np.pad(V, 3, mode='edge')
    ov = 3 if a == 0 else 2   # padded index of the lower lattice tap
    oh = 3 if b == 0 else 2
    # pixels whose row parity differs from the lattice: cubic along columns
    cv = (w0*Vp[ov-1:ov+n-1, 3:m+3] + w1*Vp[ov:ov+n, 3:m+3] +
          w2*Vp[ov+1:ov+n+1, 3:m+3] + w3*Vp[ov+2:ov+n+2, 3:m+3])
    # pixels whose col parity differs: cubic along rows
    ch = (w0*Vp[3:n+3, oh-1:oh+m-1] + w1*Vp[3:n+3, oh:oh+m] +
          w2*Vp[3:n+3, oh+1:oh+m+1] + w3*Vp[3:n+3, oh+2:oh+m+2])
    # diagonal-missing pixels: tensor-product cubic (average of the two
    # cubic passes over the half-filled fields)
    Cvp = np.pad(cv, ((0, 0), (3, 3)), mode='edge')
    dH = (w0*Cvp[:, oh-1:oh+m-1] + w1*Cvp[:, oh:oh+m] +
          w2*Cvp[:, oh+1:oh+m+1] + w3*Cvp[:, oh+2:oh+m+2])
    Chp = np.pad(ch, ((3, 3), (0, 0)), mode='edge')
    dV = (w0*Chp[ov-1:ov+n-1, :] + w1*Chp[ov:ov+n, :] +
          w2*Chp[ov+1:ov+n+1, :] + w3*Chp[ov+2:ov+n+2, :])
    if q > 0.0:
        c2v = 0.5*(Vp[ov:ov+n, 3:m+3] + Vp[ov+1:ov+n+1, 3:m+3])
        c2h = 0.5*(Vp[3:n+3, oh:oh+m] + Vp[3:n+3, oh+1:oh+m+1])
        bx = 0.25*(Vp[ov:ov+n, oh:oh+m] + Vp[ov+1:ov+n+1, oh:oh+m] +
                   Vp[ov:ov+n, oh+1:oh+m+1] + Vp[ov+1:ov+n+1, oh+1:oh+m+1])
        cv = (1-q)*cv + q*c2v
        ch = (1-q)*ch + q*c2h
        dH = (1-q)*dH + q*bx
        dV = (1-q)*dV + q*bx
    O = np.empty((H, W), np.float64)
    O[a::2, b::2] = V
    O[(1-a) % 2::2, b::2] = cv
    O[a::2, (1-b) % 2::2] = ch
    O[(1-a) % 2::2, (1-b) % 2::2] = 0.5*(dH + dV)
    return O


def _noise_sigma(raw):
    """Robust per-scene noise scale from lattice second differences."""
    vals = []
    for a, b in _OFFS:
        V = raw[a::2, b::2]
        L = (V[2:, 1:-1] + V[:-2, 1:-1] + V[1:-1, 2:] + V[1:-1, :-2] -
             4.0*V[1:-1, 1:-1])
        if L.size:
            vals.append(np.median(np.abs(L)))
    if not vals:
        return 1.5
    return float(np.median(vals))/np.sqrt(20.0)/0.6745


def reconstruct(raw):
    raw = np.asarray(raw, dtype=np.float64)
    H, W = raw.shape
    sg = _noise_sigma(raw)
    if not np.isfinite(sg):
        sg = 1.5
    t = 0.05 + 0.045*min(1.0, max(0.0, (1.5 - sg)/1.3))
    q = min(1.0, max(0.0, (sg - 0.8)/2.0))   # noisy scenes: average more
    lat = [raw[a::2, b::2] for a, b in _OFFS]
    S = np.mean([_cubic_ch(V, a, b, H, W, t)
                 for V, (a, b) in zip(lat, _OFFS)], axis=0)
    out = None
    for _ in range(2):
        out = np.empty((4, H, W), np.float64)
        for k, (a, b) in enumerate(_OFFS):
            out[k] = S + _cubic_ch(lat[k] - S[a::2, b::2], a, b, H, W, t - 0.01, q)
        S = np.mean(out, axis=0)
    np.clip(out, 0.0, 255.0, out=out)
    return out.astype(np.float32)
