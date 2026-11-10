"""Print the one-trajectory matrix error. Requires Python 3 and NumPy."""
import argparse
import sys

import numpy as np


def relative_error(reference, prediction):
    if prediction.shape != reference.shape or prediction.dtype.str != "<f8":
        raise ValueError("prediction must have the reference shape and dtype <f8")
    if not np.isfinite(prediction).all():
        raise ValueError("prediction contains a nonfinite value")

    # Scale before taking norms to avoid overflow when checking the tolerance.
    scale = max(1.0, np.abs(prediction).max())
    h = prediction / scale
    tol = 1e-10 * max(1.0 / scale, np.linalg.norm(h, "fro"))
    if np.linalg.norm(h - h.T, "fro") > tol:
        raise ValueError("prediction exceeds the symmetry tolerance")
    h = (h + h.T) / 2
    eig, vectors = np.linalg.eigh(h)
    if eig[0] < -tol:
        raise ValueError("prediction has a negative eigenvalue beyond tolerance")
    if eig[0] < 0:
        h = (vectors * np.maximum(eig, 0.0)) @ vectors.T
        accepted = h * scale
    elif np.array_equal(prediction, prediction.T):
        accepted = prediction
    else:
        accepted = h * scale
    difference = accepted - reference
    factor = np.linalg.cholesky(reference)
    left = np.linalg.solve(factor, difference)
    whitened = np.linalg.solve(factor, left.T).T
    return np.hypot.reduce(whitened.ravel())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("prediction")
    args = parser.parse_args()
    try:
        reference = np.load(args.reference, allow_pickle=False)
        with open(args.prediction, "rb") as f:
            prediction = np.load(f, allow_pickle=False)
            if f.read(1) or not isinstance(prediction, np.ndarray) or not prediction.flags.c_contiguous:
                raise ValueError("expected one C-order NPY array without trailing data")
        with np.errstate(over="raise", invalid="raise"):
            error = relative_error(reference, prediction)
        print(f"{error:.17g}")
    except (ValueError, OSError, AttributeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        print(f"Invalid: {exc}", file=sys.stderr)
        sys.exit(1)
