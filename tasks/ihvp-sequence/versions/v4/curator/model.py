"""Host-only factored SPD operators for the v4 hard-sequence supplement.

Teacher factors and this construction are not part of candidate sample mounts.
H_t = Q_t sqrt(D_t) B_t^2 sqrt(D_t) Q_t.T, B_t symmetric positive definite.
b_t = Q_t sqrt(D_t) B_t y_t; x*_t = Q_t D_t^-1/2 B_t^-1 y_t.
Thus x*_t.T H_t x*_t = ||y_t||^2, balancing reference energy across coordinates.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import json
from pathlib import Path
import numpy as np
from scipy.fft import dct, idct

SCHEMA = "frontieram/ihvp-routed-factored-spd-4"


def _scale(values, scale):
    return values * (scale if values.ndim == 1 else scale[:, None])


def _rotate_pairs(values, permutation, cosine, sine, transpose=False):
    """Disjoint pairs are consecutive entries in permutation, in that order."""
    pair = values[permutation]
    first, second = pair[0::2], pair[1::2]
    if transpose:
        sine = -sine
    result = np.empty_like(values)
    result[permutation[0::2]] = _scale(first, cosine) - _scale(second, sine)
    result[permutation[1::2]] = _scale(first, sine) + _scale(second, cosine)
    return result




class HardCase:
    """Evaluator-side case; candidate code must receive only controlled HVPs."""
    def __init__(self, path, validate_teacher=True):
        if isinstance(path, (str, Path)):
            with np.load(path, allow_pickle=False) as source:
                self.arrays = {key: source[key].copy() for key in source.files}
        else:
            self.arrays = dict(path)
        self.data = self.arrays
        raw = self.arrays["metadata_json"]
        self.meta = json.loads(str(raw.item() if isinstance(raw, np.ndarray) else raw))
        if self.meta.get("schema") != SCHEMA:
            raise ValueError("Unsupported hard-case schema")
        self.metadata = self.meta
        self.dimension, self.steps = int(self.meta["dimension"]), int(self.meta["steps"])
        self._rhs, self._reference = self.arrays["b"], self.arrays["z_star"]
        self._phase = self.arrays["phase"]
        self.mu, self.L = float(self.meta["mu"]), float(self.meta["L"])
        self._base_cosines = np.cos(self.arrays["base_angles"])
        self._base_sines = np.sin(self.arrays["base_angles"])
        self._cached_tick = None
        self._temporal_cosines = self._temporal_sines = None
        self._sqrt_diagonal = None
        self._reference_energy = np.full(self.steps, np.nan)
        if self._rhs.shape != (self.steps, self.dimension) or self._reference.shape != self._rhs.shape:
            raise ValueError("Invalid teacher dimensions")
        if validate_teacher:
            if not np.isfinite(self._rhs).all() or not np.isfinite(self._reference).all():
                raise ArithmeticError("Nonfinite teacher body")
            for t in range(self.steps):
                self._energy_denominator(t)

    def rhs(self, t):
        return self._rhs[int(t)]

    def reference(self, t):
        return self._reference[int(t)]

    def phase(self, t):
        return int(self._phase[int(t)])

    def __getitem__(self, key):
        if key in ("metadata", "meta"):
            return self.meta
        if key == "mu":
            return self.mu
        if key == "L":
            return self.L
        return self.arrays[key]

    def _prepare(self, t):
        if not 0 <= int(t) < self.steps:
            raise ValueError("Invalid sequence step")
        if self._cached_tick != int(t):
            self._cached_tick = int(t)
            angle = self.arrays["temporal_angles"][t]
            self._temporal_cosines, self._temporal_sines = np.cos(angle), np.sin(angle)
            self._sqrt_diagonal = np.sqrt(self.arrays["diagonal"][t])

    def _values(self, values):
        values = np.asarray(values, dtype=np.float64)
        if (values.ndim not in (1, 2) or values.shape[0] != self.dimension
                or (values.ndim == 2 and not 1 <= values.shape[1] <= 64)):
            raise ValueError("Expected vector[d] or block[d,k] with 1<=k<=64")
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite vector")
        return values

    def apply_base_q(self, values, transpose=False):
        values = np.asarray(values, dtype=np.float64)
        permutations, signs = self.arrays["base_permutations"], self.arrays["base_signs"]
        routing = self.arrays["base_routing"]
        result = _scale(values, self.arrays["post_signs"]) if transpose else values
        if not transpose:
            for layer in range(len(routing)):
                result = _rotate_pairs(result, routing[layer], self._base_cosines[layer],
                                       self._base_sines[layer])
        stages = range(len(permutations)-1, -1, -1) if transpose else range(len(permutations))
        for stage in stages:
            if not transpose:
                result = dct(_scale(result[permutations[stage]], signs[stage]), type=2,
                             norm="ortho", axis=0, workers=1)
            else:
                result = _scale(idct(result, type=2, norm="ortho", axis=0, workers=1), signs[stage])
                output = np.empty_like(result)
                output[permutations[stage]] = result
                result = output
        if transpose:
            for layer in range(len(routing)-1, -1, -1):
                result = _rotate_pairs(result, routing[layer], self._base_cosines[layer],
                                       self._base_sines[layer], transpose=True)
            return result
        return _scale(result, self.arrays["post_signs"])

    def apply_q(self, t, values):
        """Q_t = Q_base R_1(t) R_0(t). Pairings are spectrally local."""
        self._prepare(t)
        result = np.asarray(values, dtype=np.float64)
        for layer in range(2):
            result = _rotate_pairs(result, self.arrays["temporal_permutations"][layer],
                                   self._temporal_cosines[layer], self._temporal_sines[layer])
        return self.apply_base_q(result)

    def apply_qt(self, t, values):
        self._prepare(t)
        result = self.apply_base_q(values, transpose=True)
        for layer in (1, 0):
            result = _rotate_pairs(result, self.arrays["temporal_permutations"][layer],
                                   self._temporal_cosines[layer], self._temporal_sines[layer], True)
        return result

    def apply_b(self, t, values, inverse=False):
        """B blocks [[a,b],[b,c]] are stored as coefficients (...,3)."""
        permutation = self.arrays["block_permutation"]
        pair = np.asarray(values)[permutation]
        first, second = pair[0::2], pair[1::2]
        a, b, c = self.arrays["block_coefficients"][t].T
        if inverse:
            determinant = a * c - b * b
            a, b, c = c / determinant, -b / determinant, a / determinant
        result = np.empty_like(values, dtype=np.float64)
        result[permutation[0::2]] = _scale(first, a) + _scale(second, b)
        result[permutation[1::2]] = _scale(first, b) + _scale(second, c)
        return result

    def factor_forward(self, t, values):
        """F_t=B_t sqrt(D_t) Q_t.T, so H_t=F_t.T F_t."""
        self._prepare(t)
        return self.apply_b(t, _scale(self.apply_qt(t, values), self._sqrt_diagonal))

    def factor_transpose(self, t, values):
        self._prepare(t)
        return self.apply_q(t, _scale(self.apply_b(t, values), self._sqrt_diagonal))

    def hvp(self, t, values):
        values = self._values(values)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = self.factor_transpose(t, self.factor_forward(t, values))
            if not np.isfinite(result).all():
                raise FloatingPointError("Nonfinite operator product")
            return result

    def _energy_denominator(self, t):
        if np.isnan(self._reference_energy[t]):
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    white = self.factor_forward(t, self._reference[t])
                    energy = float(np.dot(white, white))
            except (FloatingPointError, ValueError) as error:
                raise ArithmeticError("Invalid teacher energy") from error
            if not np.isfinite(energy) or energy <= 0:
                raise ArithmeticError("Teacher energy must be finite and strictly positive")
            self._reference_energy[t] = energy
        return float(self._reference_energy[t])

    def score(self, t, prediction):
        prediction = np.asarray(prediction)
        if (prediction.shape != (self.dimension,) or prediction.dtype.kind != "f"
                or prediction.dtype.itemsize not in (4, 8) or not np.isfinite(prediction).all()):
            raise ValueError("Invalid finite float32/float64 prediction vector")
        prediction = prediction.astype(np.float64, copy=False)
        denominator = self._energy_denominator(t)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            error = prediction - self._reference[t]
            white_error = self.factor_forward(t, error)
            energy_loss = float(np.dot(white_error, white_error) / denominator)
            residual = self.hvp(t, prediction) - self._rhs[t]
            result = {"energy_loss": energy_loss, "loss": energy_loss,
                      "relative_energy_error": float(np.sqrt(energy_loss)),
                      "relative_l2_error": float(np.linalg.norm(error) / np.linalg.norm(self._reference[t])),
                      "relative_residual": float(np.linalg.norm(residual) / np.linalg.norm(self._rhs[t]))}
            if not all(np.isfinite(value) for value in result.values()):
                raise FloatingPointError("Nonfinite prediction score")
            return result


def build_operator_data(path):
    return HardCase(path)


def case_matvec(data, t, vector):
    return data.hvp(t, vector)


def read_case(path):
    data = HardCase(path)
    return data, data.meta
