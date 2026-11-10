"""Generic mathematical helpers for coupled SPD sequences; contains no dataset seed plan."""

import numpy as np

STEPS = 50

FAMILIES = (
    "low_effective_rank_smooth",
    "wide_spectrum_smooth",
    "clustered_spectrum_smooth",
    "fullrank_drift",
    "phase_jump",
)

FAMILY_SETTINGS = {
    "low_effective_rank_smooth": {"diagonal_amplitude": 0.0, "rank_weight_max": 0.6,
                                  "description": "Unit background plus a small set of broad spikes and a changing PSD low-rank correction."},
    "wide_spectrum_smooth": {"diagonal_amplitude": 0.04, "rank_weight_max": 0.4,
                             "description": "A broad log-spaced spectrum with slow shared-control changes."},
    "clustered_spectrum_smooth": {"diagonal_amplitude": 0.025, "rank_weight_max": 0.15,
                                  "description": "Six narrow spectral bands with slow shared-control changes."},
    "fullrank_drift": {"diagonal_amplitude": 0.18, "rank_weight_max": 0.65,
                       "description": "Broad spectrum; a low-dimensional control drives full-rank diagonal drift plus noncommuting low-rank changes."},
    "phase_jump": {"diagonal_amplitude": 0.18, "rank_weight_max": 0.65,
                   "description": "The same coupled family, with announced control restarts at one-based steps 21 and 36."},
}

def _orthonormal(rng, rows, columns):
    q, r = np.linalg.qr(rng.standard_normal((rows, columns)), mode="reduced")
    return q * np.where(np.diag(r) < 0.0, -1.0, 1.0)

def _unit(rng, dimension):
    vector = rng.standard_normal(dimension)
    return vector / np.linalg.norm(vector)

def _make_spectrum(rng, family, dimension, condition_target):
    if family == "low_effective_rank_smooth":
        spike_count = min(32, dimension // 16)
        spectrum = np.ones(dimension)
        spectrum[-spike_count:] = np.geomspace(2.0, condition_target, spike_count)
    elif family == "clustered_spectrum_smooth":
        centers = np.geomspace(1.0, condition_target, 6)
        bands = np.minimum(5, np.arange(dimension) * 6 // dimension)
        spectrum = centers[bands] * np.exp(rng.uniform(-0.025, 0.025, dimension))
        spectrum.sort()
        log_spectrum = np.log(spectrum)
        log_spectrum = ((log_spectrum - log_spectrum[0]) /
                        (log_spectrum[-1] - log_spectrum[0]) * np.log(condition_target))
        spectrum = np.exp(log_spectrum)
    else:
        spectrum = np.geomspace(1.0, condition_target, dimension)
    return spectrum

def _control_path(rng, family, steps, controls):
    phase = np.zeros(steps, dtype=np.int64)
    if family == "phase_jump":
        phase[20:35] = 1
        phase[35:] = 2
    path = np.empty((steps, controls))
    for segment in np.unique(phase):
        selected = np.flatnonzero(phase == segment)
        local_time = np.arange(len(selected))[:, None]
        offset = rng.uniform(-0.9, 0.9, controls)
        amplitude = rng.uniform(0.35, 0.85, controls)
        frequency = rng.uniform(0.015, 0.05, controls)
        angle = rng.uniform(-np.pi, np.pi, controls)
        path[selected] = offset + amplitude * np.sin(local_time * frequency + angle)
    return path, phase

def _inverse_middle(diagonal, v, weights, rhs):
    """Woodbury inverse of diag(diagonal)+V diag(weights)V.T; no dense H."""
    inv_diagonal = 1.0 / diagonal
    columns = v * np.sqrt(weights)[None, :]
    scaled_columns = inv_diagonal[:, None] * columns
    small = np.eye(v.shape[1]) + columns.T @ scaled_columns
    initial = inv_diagonal * rhs
    return initial - scaled_columns @ np.linalg.solve(small, columns.T @ initial)
