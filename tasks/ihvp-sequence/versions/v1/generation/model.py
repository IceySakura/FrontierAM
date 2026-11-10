"""Generic signed-DCT coupled quadratic operator and teacher generation.

No fixed public-feedback or private-final seed plan is embedded. Use generate.py
with the two public sample recipes, or freeze new private seeds under var/.
"""

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.fft import dct, idct
from scipy.sparse.linalg import LinearOperator, cg
import common as original

SCHEMA = "coupled_quadratic_bilevel_ihvp_signed_dct_v1"

def _row_scale(values, scale):
    return values * (scale if values.ndim == 1 else scale[:, None])

def q_forward(data, values):
    """Q = D_out * IDCT_ortho * D_in * permutation; acts on columns."""
    values = np.asarray(values, dtype=np.float64)
    transformed = _row_scale(values[data["permutation"]], data["sign_in"])
    transformed = idct(transformed, type=2, norm="ortho", axis=0, workers=1)
    return _row_scale(transformed, data["sign_out"])

def q_transpose(data, values):
    """Apply the exact mathematical transpose of q_forward, without a matrix."""
    values = np.asarray(values, dtype=np.float64)
    transformed = dct(_row_scale(values, data["sign_out"]), type=2,
                      norm="ortho", axis=0, workers=1)
    transformed = _row_scale(transformed, data["sign_in"])
    result = np.empty_like(transformed)
    result[data["permutation"]] = transformed
    return result

def build_operator_data(npz):
    if isinstance(npz, (str, Path)):
        with np.load(npz, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
    elif hasattr(npz, "files"):
        data = {key: npz[key].copy() for key in npz.files}
    else:
        data = dict(npz)
    if not isinstance(data.get("metadata"), dict):
        raw = data["metadata_json"]
        data["metadata"] = json.loads(str(raw.item() if isinstance(raw, np.ndarray) else raw))
    if data["metadata"]["schema"] != SCHEMA:
        raise ValueError("Expected signed-DCT coupled IHVP schema")
    data["mu"], data["L"] = float(data["metadata"]["mu"]), float(data["metadata"]["L"])
    data["_lowrank_basis"] = q_forward(data, np.sqrt(data["spectrum"])[:, None] * data["V"])
    return data

def read_case(path):
    data = build_operator_data(path)
    return data, data["metadata"]

def case_matvec(data, t, vectors):
    """H_t times v[d] or V[d,k]; the runner charges k calls for a block."""
    vectors = np.asarray(vectors, dtype=np.float64)
    dimension = int(data["metadata"]["dimension"])
    if vectors.ndim not in (1, 2) or vectors.shape[0] != dimension:
        raise ValueError("Expected v[d] or V[d,k]")
    diagonal = data["spectrum"] * data["diagonal_weights"][t]
    u = data["_lowrank_basis"]
    reduced = q_transpose(data, vectors)
    result = q_forward(data, _row_scale(reduced, diagonal))
    lowrank = u.T @ vectors
    return result + u @ _row_scale(lowrank, data["lowrank_weights"][t])

def case_matrix(data, t, max_dimension=512):
    """Small-case audit only; refuse accidental multi-GiB dense allocation."""
    dimension = int(data["metadata"]["dimension"])
    if dimension > max_dimension:
        raise ValueError(f"Dense construction is disabled for d={dimension}; use case_matvec")
    q = q_forward(data, np.eye(dimension))
    u = data["_lowrank_basis"]
    matrix = ((q * (data["spectrum"] * data["diagonal_weights"][t])) @ q.T
              + (u * data["lowrank_weights"][t]) @ u.T)
    return (matrix + matrix.T) / 2.0

def case_forcing(data, t):
    control = data["control"][t]
    force = data["forcing_initial"] + 0.45 * (data["forcing_basis"] @ control)
    return force / np.linalg.norm(force)

def case_inner_reference(data, t):
    """Evaluator-only c_t and w*_t, reconstructed from shared control."""
    force = case_forcing(data, t)
    middle = original._inverse_middle(data["diagonal_weights"][t], data["V"],
                                       data["lowrank_weights"][t], force)
    sqrt_spectrum = np.sqrt(data["spectrum"])
    return q_forward(data, sqrt_spectrum * force), q_forward(data, middle / sqrt_spectrum)

def _independent_whitened_reference(data, t):
    """Independent, well-conditioned CG reference in the private B_t basis.

    This is reference construction using teacher factors, not a solver baseline.
    It avoids an ill-conditioned dense factorization at d=16384.
    """
    diagonal = data["diagonal_weights"][t]
    v, weights = data["V"], data["lowrank_weights"][t]
    rhs = q_transpose(data, data["b"][t]) / np.sqrt(data["spectrum"])
    operator = LinearOperator((len(rhs), len(rhs)),
        matvec=lambda value: diagonal * value + v @ (weights * (v.T @ value)), dtype=np.float64)
    solution, status = cg(operator, rhs, rtol=1e-13, atol=0.0, maxiter=100)
    if status != 0:
        raise ArithmeticError("Independent whitened reference did not converge")
    return q_forward(data, solution / np.sqrt(data["spectrum"]))

def validate_case(data, dense_crosscheck=False):
    meta = data["metadata"]
    dimension, length = int(meta["dimension"]), int(meta["steps"])
    rng = np.random.default_rng(meta["seed"] + 99871)
    probe = rng.standard_normal((dimension, 3))
    roundtrip_error = float(np.linalg.norm(q_transpose(data, q_forward(data, probe)) - probe) / np.linalg.norm(probe))
    v_error = float(np.linalg.norm(data["V"].T @ data["V"] - np.eye(data["V"].shape[1]), ord="fro"))
    assert roundtrip_error < 1e-12 and v_error < 1e-11
    assert np.min(data["diagonal_weights"]) >= meta["mu"] - 1e-14
    assert np.max(data["lowrank_weights"]) <= meta["family_settings"]["rank_weight_max"] + 1e-14
    max_outer_residual, max_inner_residual, max_b_error = 0.0, 0.0, 0.0
    for t in range(length):
        b, z = data["b"][t], data["z_star"][t]
        max_outer_residual = max(max_outer_residual,
            float(np.linalg.norm(case_matvec(data, t, z) - b) / np.linalg.norm(b)))
        c, w = case_inner_reference(data, t)
        max_inner_residual = max(max_inner_residual,
            float(np.linalg.norm(case_matvec(data, t, w) - c) / np.linalg.norm(c)))
        b_check = q_forward(data, np.sqrt(data["spectrum"]) *
                            (q_transpose(data, w) - data["validation_target_eigen"]))
        max_b_error = max(max_b_error, float(np.linalg.norm(b_check - b) / np.linalg.norm(b)))
    sampled = []
    for t in sorted({0, length // 2, length - 1}):
        independent = _independent_whitened_reference(data, t)
        error = data["z_star"][t] - independent
        energy = float(np.sqrt(max(0.0, float(error @ case_matvec(data, t, error))) /
                               float(independent @ data["b"][t])))
        record = {"step_one_based": t + 1,
                  "independent_whitened_cg_relative_energy_error": energy}
        assert energy < 1e-7
        if dense_crosscheck:
            matrix = case_matrix(data, t)
            chol = np.linalg.cholesky(matrix)
            exact = np.linalg.solve(chol.T, np.linalg.solve(chol, data["b"][t]))
            delta = data["z_star"][t] - exact
            dense_energy = float(np.sqrt(max(0.0, float(delta @ matrix @ delta)) / float(exact @ data["b"][t])))
            evals = np.linalg.eigvalsh(matrix)
            block_error = float(np.linalg.norm(case_matvec(data, t, probe) - matrix @ probe) / np.linalg.norm(matrix @ probe))
            record.update(dense_cholesky_relative_energy_error=dense_energy,
                          minimum_eigenvalue=float(evals[0]), maximum_eigenvalue=float(evals[-1]),
                          dense_vs_fast_block_relative_error=block_error)
            assert dense_energy < 1e-6 and block_error < 1e-11
            assert evals[0] >= meta["mu"] * (1.0 - 1e-7)
            assert evals[-1] <= meta["L"] * (1.0 + 1e-7)
        sampled.append(record)
    residual_tolerance = 2e-7 if meta["condition_target"] >= 1e7 else 1e-8
    assert max_outer_residual < residual_tolerance, max_outer_residual
    assert max_inner_residual < residual_tolerance, max_inner_residual
    assert max_b_error < 1e-8, max_b_error
    return {"status": "PASS", "transform_roundtrip_relative_error": roundtrip_error,
            "V_orthogonality_error_fro": v_error,
            "maximum_reference_relative_residual_all_steps": max_outer_residual,
            "maximum_inner_reference_relative_residual_all_steps": max_inner_residual,
            "maximum_validation_gradient_relative_error_all_steps": max_b_error,
            "sampled_independent_references": sampled,
            "dense_crosscheck": bool(dense_crosscheck),
            "uniform_spectral_bounds": "Analytic for all steps; dense eigenvalue checks additionally used in small-d self-test."}

def generate_case(config, dense_crosscheck=False):
    begin = time.perf_counter()
    rng = np.random.default_rng(int(config["seed"]))
    dimension, length, rank, controls = int(config["dimension"]), int(config["steps"]), 12, 6
    family = config["family"]
    settings = original.FAMILY_SETTINGS[family]
    spectrum = original._make_spectrum(rng, family, dimension, config["condition_target"])
    permutation = rng.permutation(dimension).astype(np.int64)
    sign_in = rng.choice([-1.0, 1.0], dimension)
    sign_out = rng.choice([-1.0, 1.0], dimension)
    v = original._orthonormal(rng, dimension, rank)
    control, phase = original._control_path(rng, family, length, controls)
    diagonal_features = rng.standard_normal((dimension, controls)) / np.sqrt(controls)
    rank_features = rng.standard_normal((rank, controls)) / np.sqrt(controls)
    diagonal_weights = 1 + settings["diagonal_amplitude"] * np.tanh(control @ diagonal_features.T)
    lowrank_weights = settings["rank_weight_max"] / (1 + np.exp(-np.clip(control @ rank_features.T, -30, 30)))
    forcing_basis = original._orthonormal(rng, dimension, controls)
    forcing_initial = original._unit(rng, dimension)
    validation_latent = 0.7 * original._unit(rng, dimension)
    seqid = f"large_dct__{family}__{config['profile']}__d{dimension}__seed{config['seed']}"
    mu = 1.0 - settings["diagonal_amplitude"]
    upper = float(spectrum[-1]) * (1 + settings["diagonal_amplitude"] + settings["rank_weight_max"])
    meta = {
        "schema": SCHEMA, "seqid": seqid, "family": family, "profile": config["profile"],
        "dimension": dimension, "steps": length, "seed": int(config["seed"]),
        "repeat": int(config.get("repeat", 1)), "condition_target": float(config["condition_target"]),
        "mu": mu, "L": upper, "dtype": "float64", "family_settings": settings,
        "orientation_change": "Signed/permuted orthogonal DCT replaces original dense Haar Q. Same spectral/control family, different eigenbasis distribution; not a bitwise or distribution-identical enlargement.",
        "Q_definition": "Q=D_out*IDCT_ortho(type2)*D_in*permutation; Q.T is its exact mathematical transpose.",
        "H_definition": "H(s)=Q sqrt(D)[diag(a(s))+V diag(r(s))V.T]sqrt(D)Q.T",
        "inner_objective": "g_s(w)=0.5*w.T@H(s)@w-c(s).T@w; c(s)=Q sqrt(D)u(s)",
        "forcing_definition": "u(s)=(forcing_initial+0.45*forcing_basis@s)/norm(...)",
        "validation_objective": "f(w)=0.5*(Q.T@w-validation_target_eigen).T@sqrt(D)@(Q.T@w-validation_target_eigen)",
        "validation_target_definition": "validation_target_eigen=D^(-1/2)*validation_latent; fixed per sequence.",
        "b_definition": "b(s)=gradient f(w_star(s)), w_star(s)=H(s)^(-1)c(s)",
        "target_definition": "z_star(s)=H(s)^(-1)b(s)",
        "trajectory_kind": "Exogenous shared-control sinusoidal path; not a true optimizer/gradient trajectory.",
        "coordinate_transport": "Identity at every step; no hidden gauge changes.",
        "phase_boundaries_one_based": [int(i) + 1 for i in np.flatnonzero(np.r_[True, phase[1:] != phase[:-1]])],
        "reference_method": "Woodbury in the well-conditioned whitened basis; independently checked using CG in that private basis.",
        "uniform_spectral_proof": "a_i in[1-amplitude,1+amplitude], r_j in[0,rank_weight_max], V orthonormal, Q orthogonal; hence mu I<=H_t<=L I.",
        "operator_complexity": "Two orthonormal DCT transforms O(d log d) plus O(d*12) low-rank work per vector; no O(d^2) arrays.",
        "stored_time_arrays": "b,z_star,diagonal_weights; c and w_star are reconstructed only for evaluator reference checks.",
        "information_boundary": "Private teacher factors/solutions are not solver inputs; solver gets b,phase,mu/L and counted HVP only.",
        "block_cost": "One HVP charged per input column even when FFT is evaluated as a block.",
        "selection_policy": "Fixed family/profile/repeat seed plan; no solver results used to select cases.",
        "parent_generator_sha256": hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
    }
    arrays = {"spectrum": spectrum, "V": v, "permutation": permutation,
              "sign_in": sign_in, "sign_out": sign_out, "control": control, "phase": phase,
              "diagonal_features": diagonal_features, "rank_features": rank_features,
              "diagonal_weights": diagonal_weights, "lowrank_weights": lowrank_weights,
              "forcing_basis": forcing_basis, "forcing_initial": forcing_initial,
              "validation_latent": validation_latent,
              "validation_target_eigen": validation_latent / np.sqrt(spectrum),
              "metadata_json": np.array(json.dumps(meta, ensure_ascii=False, allow_nan=False))}
    data = build_operator_data(arrays)
    b = np.empty((length, dimension))
    z = np.empty_like(b)
    sqrt_spectrum = np.sqrt(spectrum)
    for t in range(length):
        forcing = case_forcing(data, t)
        middle = original._inverse_middle(diagonal_weights[t], v, lowrank_weights[t], forcing)
        b_eigen = middle - validation_latent
        z_eigen = original._inverse_middle(diagonal_weights[t], v, lowrank_weights[t],
                                           b_eigen / sqrt_spectrum) / sqrt_spectrum
        b[t] = q_forward(data, b_eigen)
        z[t] = q_forward(data, z_eigen)
    arrays["b"], arrays["z_star"] = b, z
    data["b"], data["z_star"] = b, z
    meta["validation"] = validate_case(data, dense_crosscheck=dense_crosscheck)
    meta["stored_array_bytes_uncompressed"] = sum(a.nbytes for a in arrays.values() if isinstance(a, np.ndarray))
    meta["cached_operator_basis_bytes"] = data["_lowrank_basis"].nbytes
    meta["generation_and_audit_kernel_seconds"] = time.perf_counter() - begin
    arrays["metadata_json"] = np.array(json.dumps(meta, ensure_ascii=False, allow_nan=False))
    return arrays, meta
