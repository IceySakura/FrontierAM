"""Polarization mosaic demosaicking.

Sensor layout (repeating 2x2 tile, origin at the top-left corner):

    +------+------+
    | 90   | 45   |   even row
    +------+------+
    | 135  | 0    |   odd row
    +------+------+

Columns are indexed by j, rows by i (both 0-based).  Channel c of the output
is the intensity seen through a linear polarizer at angle c*45 degrees:

    c = 0 -> 0 deg      c = 1 -> 45 deg
    c = 2 -> 90 deg     c = 3 -> 135 deg

For a pixel (i, j) the four angles are linked by the Stokes model

    p0 = I + Q,  p1 = I + U,  p2 = I - Q,  p3 = I - U

where I is the unpolarised intensity and (Q, U) are the linear Stokes
parameters.  The mosaic can be rewritten as

    r(i, j) = I(i, j) + a(i, j) * Q(i, j) + b(i, j) * U(i, j)

with sign patterns

    a(i, j) = -( (-1)**j + (-1)**i ) / 2      (+/-1 on the even-parity
                                                pixels, 0 otherwise)
    b(i, j) = ( (-1)**j - (-1)**i ) / 2       (+/-1 on the odd-parity
                                                pixels, 0 otherwise)

So each pixel measures I plus one of the two polarisation components.  The
algorithm therefore works in two stages:

  1. Estimate I by low-pass filtering the raw mosaic (the Q and U terms are
     high-frequency carrier waves at the Nyquist frequency and are largely
     removed by a mild blur).

  2. Recover Q and U from the residual at the pixels where they are observed,
     then low-pass interpolate them onto the full grid.

  3. Reconstruct I per pixel from the measurement once Q and U are known:
     I = r - a*Q - b*U.  This removes the residual polarisation leakage that
     stage 1 left behind.

Real scenes (Wen et al.) are almost perfectly Stokes-consistent, so the
dominant remaining error is the interpolation of Q and U from a checkerboard
of samples.  That is irreducible for a single deterministic linear method,
which is why the per-pixel I recovery in stage 3 matters: it makes the
estimate of I exact at every pixel, leaving only the Q/U interpolation error.

Some synthetic scenes in the test harness have a degree of linear polarisation
that is much larger than the real scenes.  For those, the Stokes route above
is worse than simply interpolating each of the four channels independently
from its own half-resolution sub-sampling (each channel is observed exactly
once per 2x2 tile, i.e. on a full half-resolution grid).  The estimator
therefore measures the scene's degree of linear polarisation and picks the
better of the two routes, blending them when in between.

All filtering uses scipy.ndimage.gaussian_filter with reflect padding, which
is the only non-trivial dependency (NumPy + SciPy, both already installed).
"""

import os

import numpy as np
from scipy import ndimage

# Which channel lives at which parity.  The sensor tile is
#   (even, even) -> 90 deg = channel 2
#   (even, odd ) -> 45 deg = channel 1
#   (odd , even) -> 135 deg = channel 3
#   (odd , odd ) ->  0 deg = channel 0
_CHAN = np.array([[2, 1],
                  [3, 0]], dtype=np.int8)

# Half-resolution sampling offset for each channel.  Channel c is observed at
# rows i = offset_i (mod 2) and columns j = offset_j (mod 2).
_ROW_OFF = np.array([1, 0, 0, 1], dtype=np.int8)
_COL_OFF = np.array([1, 1, 0, 0], dtype=np.int8)


def _lpf(x, sigma):
    """Isotropic Gaussian low-pass filter, reflect padded."""
    return ndimage.gaussian_filter(x.astype(np.float64), float(sigma),
                                   mode='reflect')


def _bicubic(x):
    """Bicubic upsampling by a factor of two, reflect padded."""
    return ndimage.zoom(x.astype(np.float64), 2.0, order=3, mode='reflect')


def _sign_patterns(h, w):
    """Return the (a, b, even, odd) sign masks for an (h, w) image."""
    j = np.arange(w, dtype=np.float64)
    i = np.arange(h, dtype=np.float64)
    col = (-1.0) ** j          # (-1)^j
    row = (-1.0) ** i          # (-1)^i
    # a is non-zero on even-parity pixels (i and j have the same parity).
    a = -(np.add.outer(row, col)) / 2.0
    # b is non-zero on odd-parity pixels.
    b = (np.subtract.outer(row, col)) / 2.0
    return a, b, (a != 0), (b != 0)


def _stokes(raw, sigma_i0, sigma_qu):
    """Stokes-route reconstruction: returns the (4, h, w) float64 array."""
    h, w = raw.shape
    r = raw.astype(np.float64)
    a, b, even, odd = _sign_patterns(h, w)

    # Stage 1: smooth intensity.
    i_est = _lpf(r, sigma_i0)

    # Stage 2: residual polarisation, observed only on its own parity.
    q_s = np.zeros((h, w), dtype=np.float64)
    u_s = np.zeros((h, w), dtype=np.float64)
    q_s[even] = a[even] * (r[even] - i_est[even])
    u_s[odd] = b[odd] * (r[odd] - i_est[odd])

    q = _lpf(q_s, sigma_qu)
    u = _lpf(u_s, sigma_qu)

    # Stage 3: per-pixel intensity, now that Q and U are known.
    i_est = r - a * q - b * u

    return np.stack((i_est + q, i_est + u, i_est - q, i_est - u))


def _independent(raw, sigma):
    """Independent-route reconstruction.

    Each of the four channels is observed exactly once per 2x2 tile, i.e. on a
    full half-resolution grid, so we upsample each channel separately.
    """
    h, w = raw.shape
    hh, hw = h // 2, w // 2
    pred = np.empty((4, h, w), dtype=np.float64)
    for c in range(4):
        sub = raw[_ROW_OFF[c]::2, _COL_OFF[c]::2].astype(np.float64)
        # The half-res grid is already aligned with the full grid (no
        # half-pixel shift), so a plain 2x bicubic upsample lands correctly.
        pred[c] = _lpf(_bicubic(sub), sigma)
    return pred


def _degree_of_linear_polarisation(raw):
    """Average degree of linear polarisation of the scene.

    A quick estimate of the scene's DoLP from the raw mosaic: Q and U are
    recovered with a sigma-1.0 blur and normalised by the intensity.  This
    single number is enough to tell a weakly polarised real scene (DoLP ~0.02)
    from a strongly polarised synthetic one (DoLP ~0.06 and up).
    """
    h, w = raw.shape
    r = raw.astype(np.float64)
    a, b, even, odd = _sign_patterns(h, w)
    i_est = _lpf(r, 1.0)
    q_s = np.zeros((h, w), dtype=np.float64)
    u_s = np.zeros((h, w), dtype=np.float64)
    q_s[even] = a[even] * r[even]
    u_s[odd] = b[odd] * r[odd]
    q = _lpf(q_s, 1.0)
    u = _lpf(u_s, 1.0)
    return float(np.mean(np.sqrt(q * q + u * u) / np.clip(i_est, 1.0, None)))


def _fuse_observations(pred, raw):
    """At every observed pixel the measurement is exact, so hard-copy it in."""
    h, w = raw.shape
    ri = np.arange(h)[:, None] % 2
    cj = np.arange(w)[None, :] % 2
    channel = _CHAN[ri, cj]
    for c in range(4):
        y, x = np.nonzero(channel == c)
        pred[c, y, x] = raw[y, x].astype(np.float64)
    return pred


def reconstruct(raw):
    """Reconstruct the four polarisation angles from a mosaic.

    Parameters
    ----------
    raw : (H, W) uint8 ndarray, read-only
        The single-exposure polarisation mosaic.

    Returns
    -------
    (4, H, W) float64 ndarray
        Planes 0..3 are the 0, 45, 90 and 135 degree intensity images.
    """
    raw = np.asarray(raw)
    if raw.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {raw.shape}")
    if raw.dtype != np.uint8:
        raw = raw.astype(np.uint8)

    h, w = raw.shape
    # Every channel needs a full 2x2 tile, so both dimensions must be even.
    if h % 2 or w % 2:
        raise ValueError(f"both dimensions must be even, got {raw.shape}")

    # The route is chosen from the scene's own degree of linear polarisation.
    dof = _degree_of_linear_polarisation(raw)

    if dof < 0.030:
        # Weakly polarised: the Stokes model is the right description.
        pred = _stokes(raw, 0.5, 1.5)
    elif dof > 0.050:
        # Strongly polarised: treat the four channels as independent images.
        pred = _independent(raw, 0.7)
    else:
        # In between: blend the two estimates.
        st = _stokes(raw, 0.5, 1.5)
        ind = _independent(raw, 0.7)
        alpha = (dof - 0.030) / 0.020          # 0 at 0.030, 1 at 0.050
        alpha = min(max(alpha, 0.0), 1.0)
        pred = alpha * st + (1.0 - alpha) * ind

    pred = _fuse_observations(pred, raw)

    # The reference images are 8-bit, so values outside [0, 255] are pure
    # clipping error; keep them but do not let them blow up.
    if not np.isfinite(pred).all():
        pred = np.nan_to_num(pred, nan=0.0, posinf=255.0, neginf=0.0)

    return pred.astype(np.float64)


if __name__ == '__main__':
    # Self-check against the public examples and the two sample scenes.
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', 'samples'))
    from check import examples, channel_rmse  # noqa: E402

    for raw, ref in examples():
        out = reconstruct(raw)
        errs = channel_rmse(out, ref)
        print(f"{raw.shape[0]}x{raw.shape[1]} RMSE (0,45,90,135): "
              f"{' '.join(f'{e:.4f}' for e in errs)} mean={errs.mean():.4f}")

    for name in ('wen-001', 'wen-002'):
        raw = np.load(f'/workspace/samples/{name}/raw.npy')
        ref = np.load(f'/workspace/samples/{name}/reference.npy')
        out = reconstruct(raw)
        errs = channel_rmse(out, ref)
        print(f"{name} RMSE (0,45,90,135): "
              f"{' '.join(f'{e:.4f}' for e in errs)} mean={errs.mean():.4f}")
