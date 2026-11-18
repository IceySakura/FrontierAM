"""Polarization demosaicking via a smoothed Stokes decomposition.

The mosaic repeats every 2 rows / 2 columns (0-indexed):

    even row:  90  45
    odd row:  135   0

So each pixel measures one of {I0, I45, I90, I135} and the other three angles
must be inferred. We work in Stokes coordinates:

    I = (I0 + I90)/2 = (I45 + I135)/2      total intensity
    Q = I0 - I = I - I90                    linear horizontal/vertical
    U = I45 - I = I - I135                  linear +/-45

so that I0 = I + Q, I45 = I + U, I90 = I - Q, I135 = I - U.

Only I can be read directly from the raw mosaic: every pixel has a
180-degree-rotated neighbour exactly one pixel away on the diagonal, and that
neighbour measures the complementary angle. Averaging the two gives I exactly
at every pixel. Q and U, however, are each observed on only one coset of the
checkerboard (Q on the i+j-even coset, U on the i+j-odd coset) and must be
interpolated to full resolution.

The two design choices that matter most:

1. The exact-per-pixel I from a single diagonal pair is a noisy estimator.
   Because I is a low-pass quantity (the four angles average to I), replacing
   the exact diagonal average with a mild Gaussian blur of the raw mosaic
   trades a small bias for far less high-frequency variance in the derived Q
   and U fields. A Gaussian with sigma = 0.68 (truncate = 4) is the best
   compromise found by a sweep over the public development scenes.

2. Interpolating a checkerboard-sampled field by separable convolution must be
   normalised by the sum of the kernel over the *odd* offsets only: the
   non-zero samples sit on one coset, and a kernel with even-symmetric offsets
   (such as [0.5, 1, 0.5]) accumulates weight only from the coset that is
   occupied. Without this correction the interpolated field is scaled by 1/S.
   For the bilinear kernel S = 2.0.

3. Because the Gaussian I estimator is a blurred version of the true I, the
   derived Q and U amplitudes are attenuated. Restoring them with a gain of
   alpha = 1.20 closes part of the gap. (For the exact-diagonal I estimator the
   opposite is true and damping is needed; the optimal direction of the
   correction depends on the I estimator.)

No other interpolation kernel helped: Catmull-Rom, Mitchell-Netravali,
bicubic and Lanczos3 all score worse (Lanczos3's odd-offset sum is ~1e-16 and
produces NaNs), and edge-adaptive guided interpolation is too slow for the
1 s/scène wall-clock budget.

The whole pipeline is a handful of separable convolutions over 1024x1024 and
runs in well under 0.1 s of compute per scene.
"""

import numpy as np
from scipy.ndimage import correlate, gaussian_filter

# Mosaic phase: (row % 2, col % 2) -> measured angle in degrees.
PHASE = {(0, 0): 90, (0, 1): 45, (1, 0): 135, (1, 1): 0}

# Mild Gaussian blur used as the I estimator. Chosen by a sweep on the public
# development scenes (sigma=0.68, truncate=4). Unlike the exact-diagonal I
# estimator, for this low-pass estimator the Q/U amplitude should be *boosted*
# rather than damped: alpha=1.20 is optimal.
SIGMA_I = 0.68
ALPHA = 1.20

# Separable bilinear stencil for checkerboard interpolation. Its odd-offset
# weight sum is S = 2.0, which is what the convolution output is divided by.
BILINEAR = np.array([0.5, 1.0, 0.5])


def _checkered_interp(src, kernel, mode="reflect"):
    """Interpolate a field that is non-zero on only one checkerboard coset.

    `src` is sampled on a single parity coset of (row + col); the other coset
    is zero. A standard convolution would therefore accumulate only the kernel
    weights whose offsets have odd row+col, so we renormalise by that sum.
    """
    n = kernel.shape[0] // 2
    K = np.asarray(kernel, dtype=np.float64)[:, None] * np.asarray(
        kernel, dtype=np.float64
    )[None, :]
    odd_sum = float(
        sum(
            K[d_i + n, d_j + n]
            for d_i in range(-n, n + 1)
            for d_j in range(-n, n + 1)
            if (d_i + d_j) % 2 != 0
        )
    )
    if odd_sum == 0.0:
        raise ValueError("kernel has no weight on odd offsets; cannot interpolate")
    return correlate(src, K, mode=mode) / odd_sum


def _stokes(raw):
    """Return (I, Qf, Uf) at full resolution from the raw mosaic."""
    H, W = raw.shape
    rf = raw.astype(np.float64)

    # Low-pass estimate of the total intensity I.
    I = gaussian_filter(
        rf, sigma=SIGMA_I, mode="reflect", truncate=4.0
    )

    # Q and U observed on their respective cosets; the rest is zero.
    Q = np.zeros_like(I)
    U = np.zeros_like(I)

    # Coset (i+j) even: (0,0) measures 90, (1,1) measures 0.
    Q[0::2, 0::2] = I[0::2, 0::2] - rf[0::2, 0::2]
    Q[1::2, 1::2] = rf[1::2, 1::2] - I[1::2, 1::2]

    # Coset (i+j) odd: (0,1) measures 45, (1,0) measures 135.
    U[0::2, 1::2] = rf[0::2, 1::2] - I[0::2, 1::2]
    U[1::2, 0::2] = I[1::2, 0::2] - rf[1::2, 0::2]

    Qf = _checkered_interp(Q, BILINEAR) * ALPHA
    Uf = _checkered_interp(U, BILINEAR) * ALPHA
    return I, Qf, Uf


def reconstruct(raw):
    """Reconstruct the four full-resolution polarization angles.

    Parameters
    ----------
    raw : np.ndarray, shape (H, W), dtype uint8
        The raw single-channel mosaic, one angle per pixel.

    Returns
    -------
    np.ndarray, shape (4, H, W), dtype float64
        The four intensity images in angle order: 0, 45, 90, 135 degrees.
    """
    I, Qf, Uf = _stokes(np.asarray(raw))
    out = np.stack([I + Qf, I + Uf, I - Qf, I - Uf], axis=0)
    if not np.all(np.isfinite(out)):
        raise ValueError("reconstruction produced non-finite values")
    return out


if __name__ == "__main__":
    # Self-check: the Stokes identities should hold exactly for the exact-I
    # estimator, and approximately here because I is smoothed.
    raw = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    pred = reconstruct(raw)
    print("pred", pred.shape, pred.dtype, "finite", np.isfinite(pred).all())
    print("0+90 vs 45+135 max abs diff:",
          np.max(np.abs(pred[0] + pred[2] - pred[1] - pred[3])))
