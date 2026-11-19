#!/usr/bin/env python3
"""3D fluorescence deconvolution.

Usage: python3 -I -B solution.py INPUT.NPZ OUTPUT.NPY

INPUT.NPZ must contain exactly two float32 arrays:
    observed  - (Z, Y, X) finite nonnegative voxel values
    psf       - (K, K, K) centred odd PSF, summing to 1

Writes OUTPUT.NPY, a (Z, Y, X) float32 array of finite nonnegative values.
"""

import os
import sys
import time
import warnings

# The host pins the process to one logical CPU; do not let BLAS assume more.
for _k in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
):
    os.environ[_k] = "1"
del _k

import numpy as np  # noqa: E402
from scipy import fft  # noqa: E402

warnings.filterwarnings("ignore")

# Prior exponent.  Larger p damps high frequencies more aggressively; 2.5 sits
# in a flat optimum on the public cases.
PRIOR_P = 2.5
# Mapping from estimated background Poisson scale to lambda.  Chosen by
# leave-one-out cross-validation on the public cases; the optimum is very flat
# (a factor of 3 either way changes the mean nMSE by well under 1%) so a
# single global value is robust.
KAPPA = 0.584
# Block size for the local variance/mean noise estimator.
BLOCK = 4
# Safety margin (seconds) to leave before the absolute deadline.
DEADLINE_SLACK = 0.25
# Fallback lambda used if the noise estimator degenerates.
DEFAULT_LAMBDA = 0.01


def _now():
    return time.monotonic()


def _exhausted(deadline, slack):
    return _now() > deadline - slack


def alpha_hat(obs, block=BLOCK):
    """Estimate the background Poisson scale (variance per unit mean).

    The observed image is split into non-overlapping block x block x block
    cubes; for each cube the mean m and variance v are computed.  In a
    Poisson-like regime v is proportional to m, with the proportionality
    constant equal to the noise scale.  Restricting the regression to the
    low-intensity cubes (the bottom 40% by mean) excludes the bright signal,
    which would otherwise inflate the variance with genuine structure rather
    than noise.  Exact-zero cubes are excluded so that empty backgrounds do
    not bias the ratio.
    """
    z, y, x = obs.shape
    if block < 2 or z < 2 * block or y < 2 * block or x < 2 * block:
        return DEFAULT_LAMBDA
    zb, yb, xb = z // block * block, y // block * block, x // block * block
    core = obs[:zb, :yb, :xb].reshape(
        zb // block, block, yb // block, block, xb // block, block
    )
    means = core.mean(axis=(1, 3, 5)).ravel()
    vars_ = core.var(axis=(1, 3, 5)).ravel()
    q = float(np.percentile(means, 40.0))
    low = (means <= q) & (means > 0.0)
    if not np.any(low) or means[low].sum() <= 0.0:
        return DEFAULT_LAMBDA
    ratio = float(vars_[low].sum() / means[low].sum())
    if not np.isfinite(ratio) or ratio <= 0.0:
        return DEFAULT_LAMBDA
    return ratio


def grad_spectrum(pad, exponent=PRIOR_P):
    """Symbol of the 3-D discrete gradient prior, in rfftn layout.

    The rfftn output is ordered (z, y, x) with z and y full and x half, so the
    full frequencies are built with indexing='ij' and the x axis is the short
    rfftfreq vector.
    """
    kz = np.fft.fftfreq(pad[0], d=1.0) * (2.0 * np.pi)   # z axis: full
    ky = np.fft.fftfreq(pad[1], d=1.0) * (2.0 * np.pi)   # y axis: full
    kx = np.fft.rfftfreq(pad[2], d=1.0) * (2.0 * np.pi)  # x axis: half
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    s = 2.0 * (1.0 - np.cos(KX)) + 2.0 * (1.0 - np.cos(KY)) + 2.0 * (1.0 - np.cos(KZ))
    return s ** np.float64(exponent)


def pad_shape(shape, psf_shape):
    """FFT grid size for an exact zero-padded 'same' convolution.

    A 'same' convolution with a kernel of odd side k needs k - 1 extra samples
    along that axis (half on each side) so that the circular FFT convolution
    never wraps.  The required offset is (k - 1) // 2, which is also the
    half-length of the kernel support.  Each axis is padded independently so
    that cuboid PSFs are handled exactly.  This reproduces
    scipy.signal.convolve(mode='same') to machine precision for any odd
    kernel size and any array shape.
    """
    offs = tuple((k - 1) // 2 for k in psf_shape)
    return tuple(n + 2 * o for n, o in zip(shape, offs)), offs


def main(argv):
    t0 = _now()
    if len(argv) != 3:
        sys.stderr.write("usage: %s INPUT.NPZ OUTPUT.NPY\n" % argv[0])
        return 2

    in_path, out_path = argv[1], argv[2]

    deadline_raw = os.environ.get("DECONV_DEADLINE_MONOTONIC", "")
    deadline = float(deadline_raw) if deadline_raw.strip() else None

    # --- input ------------------------------------------------------------
    try:
        raw = np.load(in_path)
        observed = np.asarray(raw["observed"], dtype=np.float64)
        psf = np.asarray(raw["psf"], dtype=np.float64)
    except Exception as exc:  # never crash on a bad input file
        sys.stderr.write("deconv: failed to read %s: %s\n" % (in_path, exc))
        return 1

    if observed.ndim != 3 or psf.ndim != 3 or observed.size == 0 or psf.size == 0:
        sys.stderr.write("deconv: expected 3-D observed and PSF arrays\n")
        return 1

    shape = observed.shape

    # If the budget is already gone, return the clipped observation at the
    # right shape rather than an unusable placeholder.
    if deadline is not None and _exhausted(deadline, DEADLINE_SLACK):
        _write_simple(out_path, observed)
        return 0

    # --- operator ---------------------------------------------------------
    pad, offs = pad_shape(shape, psf.shape)
    dom = np.s_[: shape[0], : shape[1], : shape[2]]          # where the estimate lives
    slc = tuple(np.s_[o : o + n] for o, n in zip(offs, shape))

    def _embed_y(y):
        e = np.zeros(pad, dtype=np.float64)
        e[slc] = y
        return e

    psf_pad = np.zeros(pad, dtype=np.float64)
    psf_pad[: psf.shape[0], : psf.shape[1], : psf.shape[2]] = psf
    Hf = fft.rfftn(psf_pad)
    Hsq = np.abs(Hf) ** 2
    Htf = np.conj(Hf)

    if deadline is not None and _exhausted(deadline, DEADLINE_SLACK):
        _write_simple(out_path, observed)
        return 0

    # --- noise estimate ---------------------------------------------------
    lam = KAPPA * alpha_hat(observed)
    if not np.isfinite(lam) or lam <= 0.0:
        lam = DEFAULT_LAMBDA

    if deadline is not None and _exhausted(deadline, DEADLINE_SLACK):
        _write_simple(out_path, observed)
        return 0

    # --- frequency-domain Wiener ------------------------------------------
    S = grad_spectrum(pad)
    Y = fft.rfftn(_embed_y(observed))
    est = fft.irfftn(Y * Htf / (Hsq + lam * S), s=pad)[dom]
    # Nonnegativity: the physical quantity (fluorophore concentration) cannot
    # be negative, and clipping also damps the residual ringing.
    np.clip(est, 0.0, None, out=est)

    if not np.all(np.isfinite(est)):
        est = np.clip(observed, 0.0, None)

    if est.dtype != np.float32:
        est = est.astype(np.float32, copy=False)

    np.save(out_path, est)
    sys.stderr.write(
        "deconv: shape=%s pad=%s off=%s lambda=%.6g elapsed=%.3fs\n"
        % (str(shape), str(pad), str(offs), lam, _now() - t0)
    )
    return 0


def _write_simple(path, obs):
    """Deadline fallback: return the (clipped) observation unchanged."""
    out = np.clip(obs, 0.0, None)
    if out.dtype != np.float32:
        out = out.astype(np.float32, copy=False)
    np.save(path, out)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
