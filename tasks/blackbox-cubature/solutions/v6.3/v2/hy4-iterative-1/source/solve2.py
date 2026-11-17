"""solve2: orchestrates a single QMC net, axis lines, feature discovery, and CV selection."""
from __future__ import annotations
import math, time, warnings
import numpy as np
import advanced as A


def solve(oracle, verbose=False, tlimit=20.0, qmc_sink=None):
    import warnings
    t0 = time.perf_counter()
    d, B, seed = oracle.d, oracle.budget, oracle.seed
    vol = getattr(oracle, 'volume', 1.0)

    # ---- One scrambled Sobol net of every point we can spare for QMC.
    # A single net has markedly better low-discrepancy properties than the
    # union of several independent small scrambles, especially for kinked
    # integrands.  Cross-fitting folds are random subsets of this one net.
    N = int(min(64, max(16, B // (4 * max(1, d)) - 1)))
    nlines = d * (N + 1)
    nqmc = B - nlines
    if nqmc < 64:
        N = max(4, B // max(1, d) // 2 - 1)
        nlines = d * (N + 1)
        nqmc = B - nlines
    nqmc = max(1, min(nqmc, B))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Xq = A.sobol(d, nqmc, seed + 17)
    yq = oracle.values(Xq)
    if qmc_sink is not None:
        qmc_sink.append(np.asarray(yq).copy())

    jmax = int(np.argmax(yq))
    x0 = Xq[jmax].copy()
    f0 = float(yq[jmax])
    scale = float(np.max(np.abs(yq))) + 1e-12
    X, y = Xq, yq

    # ---- random folds of the single net, used only for cross-fitted CV
    nfold = 4
    rr = np.random.default_rng(seed + 424242)
    perm = rr.permutation(len(X))
    blocks = []
    for k in range(nfold):
        idx = np.sort(perm[k::nfold])
        blocks.append((X[idx], y[idx]))

    poly = A.poly_features(d, 2)
    ep, rp, _ = A.cv_estimate(blocks, poly, lam=1e-13)
    if rp < 1e-12 * scale:
        return ep * vol

    # ---- axis lines through the sample with the largest value
    tgrid = np.linspace(0.0, 1.0, N)
    lines = []
    for i in range(d):
        t = np.unique(np.concatenate([tgrid, [float(x0[i])]]))
        P = np.tile(x0, (len(t), 1))
        P[:, i] = t
        lines.append({'t': t, 'y': oracle.values(P)})

    # ---- feature discovery
    waves, spectra = A.discover_waves(lines, X, y, d, max_waves=3)
    wf = A.wave_features(waves)
    try:
        w0s = waves[0][0] if waves else None
        hf = A.wave_features(A.harmonic_waves(spectra, w0s, X, y, d, M=10)) if waves else []
    except Exception:
        hf = []
    try:
        cf = A.wave_features(A.signcombo_waves(spectra, X, y, d)) if waves else []
    except Exception:
        cf = []
    mn = float(np.min(y))
    cands = [mn - 1e-11 * scale, mn, mn + 1e-9 * scale, 0.0,
             mn - 1e-6 * scale, float(np.mean(y))]
    prod = A.product_features(lines, x0, f0, X, y, d, cands, max_evals=6)

    groups = {
        'plain': [],
        'poly': poly,
        'wave': wf,
        'harm': hf,
        'signcombo': cf,
        'prod': prod,
        'wave+poly': wf + poly,
        'harm+poly': hf + poly,
        'signcombo+poly': cf + poly,
        'prod+poly': prod + poly,
        'wave+prod': wf + prod,
        'harm+prod': hf + prod,
        'all': wf + hf + cf + prod + poly,
    }
    results = {}
    for name, feats in groups.items():
        if not feats and name != 'plain':
            continue
        try:
            lam = 1e-9 * max(1.0, len(feats) / 40.0)
            est, rs, spread = A.cv_estimate(blocks, feats, lam=lam)
            if np.isfinite(est):
                results[name] = (est, rs, spread, len(feats))
        except Exception:
            continue
    # ---- extra structure discovery when no exact surrogate has been found
    sy0 = float(np.std(y)) + 1e-12
    need_more = (not results) or min(v[1] for v in results.values()) > 1e-6 * sy0
    if time.perf_counter() - t0 > 9.0:
        need_more = False
    need_hinge = (not results) or min(v[1] for v in results.values()) > 1e-6 * sy0
    if need_hinge:
        try:
            import kinkfit
            import time as _time
            tkill = _time.perf_counter() + 3.5
            combos = [(3, 0), (3, 1), (4, 0), (2, 0), (4, 1), (5, 0), (3, 2)]
            for K, sd in combos:
                if _time.perf_counter() > tkill or oracle.out_of_time():
                    break
                try:
                    sp = kinkfit.scan_hinges(X, y, lines, x0, d, K=K,
                                            seed=seed + 100 * K + sd, ndirc=200)
                    if not sp:
                        continue
                    # cheap exactness check with uncentered hinges
                    Mc = [np.ones(len(X))] + [f.fn(X) + f.integral for f in []]
                    Mc = np.column_stack(
                        [np.ones(len(X))] +
                        [np.abs(X @ a - b) for a, b in sp] +
                        [f.fn(X) for f in poly])
                    bt, *_ = np.linalg.lstsq(Mc, y, rcond=1e-9)
                    rms = float(np.sqrt(np.mean((y - Mc @ bt) ** 2))) / sy0
                    if rms < 0.15:
                        fs = kinkfit.kink_features(sp, X, y, d)
                        group = poly + fs
                        lam = 1e-9 * max(1.0, len(group) / 40.0)
                        est, rs, se = A.cv_estimate(blocks, group, lam=lam)
                        if np.isfinite(est):
                            results[f'kink{K}_{sd}'] = (est, rs, se, len(group))
                    if rms < 1e-8:
                        break
                except Exception:
                    continue
        except Exception:
            pass

    if not results:
        return float(np.mean(y)) * vol
    # Strict selection: an essentially exact surrogate wins outright.  A
    # partial surrogate is used only if its cross-fitted fold estimates are
    # much more stable than the plain QMC block means AND it agrees with the
    # plain estimate within the QMC noise.  Otherwise plain QMC is returned.
    bm = np.array([float(np.mean(b[1])) for b in blocks])
    se_plain = float(np.std(bm)) / math.sqrt(len(bm)) + 1e-15
    sy = float(np.std(y)) + 1e-12
    plain = float(np.mean(y))
    best_exact = None
    best_partial = None
    for name, (e, rs, se, nf) in results.items():
        if name == 'plain' or not np.isfinite(e):
            continue
        if rs < 1e-7 * sy:
            if best_exact is None or rs < best_exact[0]:
                best_exact = (rs, e, name)
        elif (rs < 0.35 * sy and se < 0.6 * se_plain and
              abs(e - plain) < 3.0 * se_plain):
            score = se + 0.01 * rs / math.sqrt(len(bm))
            if best_partial is None or score < best_partial[0]:
                best_partial = (score, e, name)
    if best_exact is not None:
        est = best_exact[1]
        best_name = best_exact[2]
    elif best_partial is not None:
        est = best_partial[1]
        best_name = best_partial[2]
    else:
        est = plain
        best_name = 'plain'
    if not np.isfinite(est):
        est = plain
    # a finite function's integral lies within the observed value range
    lo, hi = float(np.min(y)), float(np.max(y))
    margin = 2.0 * (hi - lo) + 1e-9
    if est < lo - margin or est > hi + margin:
        est = plain
    if verbose:
        print('  waves', [(np.round(w[0], 4).tolist(), round(w[3], 5)) for w in waves])
        print('  best', best_name, 'se_model', results[best_name][2] if best_name else None,
              'se_plain', se_plain, 'rs', results[best_name][1] if best_name else None)
    return est * vol
