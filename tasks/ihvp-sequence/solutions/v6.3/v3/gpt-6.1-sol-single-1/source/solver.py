import os
import time
import math
import ctypes

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np


try:
    _core = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "core.so"))
    _core.cg_step.argtypes = [ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double]
    _core.cg_step.restype = ctypes.c_double
    _core.cg_direction.argtypes = [ctypes.c_size_t, ctypes.c_void_p,
                                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double]
    _core.cg_direction.restype = None
except (OSError, AttributeError):
    _core = None


class Solver:
    def __init__(self, dimension, seed, options):
        self.d = int(dimension)
        self.options = options or {}
        self.history = []
        self.rhs_history = []
        self.solution_history = []
        self.hcost = 0.003
        self.batch_ratio = 1.0
        self.phase_lengths = []
        self.phase_start = 0
        self.exhausted = False
        self.native = _core is not None and self.options.get("native", True)
        self.verbose = self.options.get("verbose", False)
        self.predict_rhs = self.options.get("predict_rhs", True)
        self.next_prediction_step = 0
        self.prediction_failures = 0

    def _prediction(self, b, normb, tick):
        m = len(self.rhs_history)
        if not self.predict_rhs or m < 4 or tick < self.next_prediction_step:
            return None
        B = np.asfortranarray(np.column_stack(self.rhs_history))
        g = B.T @ B
        h = (B.T @ b) / normb
        w, v = np.linalg.eigh((g + g.T) * 0.5)
        good = w > max(1e-8 * w[-1], 1e-15)
        vg = v[:, good]
        c = vg @ ((vg.T @ h) / w[good])
        error = max(0.0, 1.0 - 2.0 * float(c @ h) + float(c @ (g @ c)))
        if error > 1e-4 or float(c @ c) > 2500.0:
            self.prediction_failures += 1
            self.next_prediction_step = tick + min(8, 2 + self.prediction_failures)
            return None
        self.prediction_failures = 0
        self.next_prediction_step = tick + 1
        X = np.asfortranarray(np.column_stack(self.solution_history))
        p = X @ c
        pn = float(np.linalg.norm(p))
        if pn > 1e-100 and np.isfinite(pn):
            return p / pn
        return None

    def _fallback(self, b, L):
        if not self.history:
            return np.zeros_like(b)
        x = self.history[-2] if len(self.history) >= 2 else self.history[-1]
        den = L * float(x @ x)
        if den > 0.0:
            return x * (float(x @ b) / den)
        return np.zeros_like(b)

    def solve(self, hvp, b, context):
        b = np.asarray(b, dtype=np.float64)
        start = time.monotonic()
        tick = int(context.get("step", 0))
        steps = int(context.get("steps", 50))
        remaining = max(1, steps - tick)
        mu = max(float(context["mu"]), 1e-300)
        L = max(float(context["L"]), mu)
        deadline = float(context.get("deadline", start + 0.5)) - 0.04
        changed = bool(context.get("phase_changed", tick == 0))
        if changed:
            self.next_prediction_step = tick
        if self.exhausted or start + 0.001 >= deadline:
            return self._fallback(b, L)

        if self.d <= 64:
            try:
                a = np.asarray(hvp(np.eye(self.d)), dtype=np.float64)
                x = np.linalg.solve((a + a.T) * 0.5, b)
                if np.isfinite(x).all():
                    return x
            except Exception:
                pass

        if changed and tick > self.phase_start:
            self.phase_lengths.append(tick - self.phase_start)
            self.phase_start = tick
        log_condition = math.log10(max(1.0, L / mu))
        weight = float(self.options.get("weight", min(6.0, max(1.0, 2.0 * log_condition - 9.0))))
        reserve = float(self.options.get("reserve", 2.0))
        if self.phase_lengths:
            period = float(np.mean(self.phase_lengths))
        else:
            period = steps / reserve if reserve > 0.0 else 1e100
        future_changes = (remaining - 1) / max(1.0, period)
        current_weight = weight if changed else 1.0
        units = remaining + (weight - 1.0) * future_changes + current_weight - 1.0
        allowance = max(0.0, deadline - start) * current_weight / max(units, current_weight)
        end = min(deadline, start + allowance) - 0.001
        rank = max(2, min(16, int(self.options.get("history_rank", 4))))

        x = np.zeros_like(b)
        r = b.copy()
        safe_x = x
        Q = AQ = None
        p = None
        niter = calls = 0
        bn = float(b @ b)
        if not (bn > 0.0 and np.isfinite(bn)):
            return x
        normb = math.sqrt(bn)
        prep = 0.0
        error = None
        numeric_failure = False

        def product(v):
            nonlocal calls
            before = time.monotonic()
            out = hvp(v)
            duration = time.monotonic() - before
            count = 1 if v.ndim == 1 else v.shape[1]
            if count == 1:
                self.hcost = 0.85 * self.hcost + 0.15 * duration
            else:
                ratio = duration / (count * max(self.hcost, 1e-6))
                self.batch_ratio = 0.35 * self.batch_ratio + 0.65 * ratio
            calls += count
            return out

        try:
            bank = self.history[-rank:]
            pred = self._prediction(b, normb, tick)
            if pred is not None:
                bank = [*bank, pred]
            if bank and time.monotonic() + (len(bank) + 2) * self.hcost < end:
                S = np.asfortranarray(np.column_stack(bank))
                g = S.T @ S
                w, v = np.linalg.eigh((g + g.T) * 0.5)
                keep = w > max(1e-11 * w[-1], 1e-18)
                U = np.asfortranarray(S @ (v[:, keep] / np.sqrt(w[keep])))
                k = U.shape[1]
                batch = (k > 1 and (self.batch_ratio < 1.05 or tick % 10 == 1))
                if batch:
                    AU = product(U)
                else:
                    AU = np.empty_like(U)
                    for j in range(k):
                        AU[:, j] = product(U[:, j])
                c = U.T @ AU
                w, v = np.linalg.eigh((c + c.T) * 0.5)
                keep = w > max(1e-12 * max(w[-1], 0.0), 0.1 * mu)
                if np.any(keep):
                    transform = v[:, keep] / np.sqrt(w[keep])
                    Q = np.asfortranarray(U @ transform)
                    AQ = np.asfortranarray(AU @ transform)
                    coefficients = Q.T @ b
                    x = Q @ coefficients
                    r = b - AQ @ coefficients
                    safe_x = x.copy()
            prep = time.monotonic() - start
            rr = float(r @ r)
            p = np.zeros_like(b)
            xp, rp, pp = x.ctypes.data, r.ctypes.data, p.ctypes.data
            rr_old = 1.0
            while (rr > 1e-20 * bn
                   and time.monotonic() + max(1.45 * self.hcost, 0.0002) < end):
                correction = None if Q is None else Q @ (AQ.T @ r)
                beta = rr / rr_old if niter else 0.0
                if self.native:
                    _core.cg_direction(self.d, pp, rp,
                                       0 if correction is None else correction.ctypes.data,
                                       beta)
                else:
                    p *= beta
                    p += r
                    if correction is not None:
                        p -= correction
                ap = product(p)
                if self.native:
                    rr_next = _core.cg_step(self.d, xp, rp, pp, ap.ctypes.data, rr)
                    if not (rr_next >= 0.0 and math.isfinite(rr_next)):
                        numeric_failure = not math.isfinite(rr_next)
                        break
                else:
                    den = float(p @ ap)
                    if not (den > 0.0 and math.isfinite(den)):
                        break
                    alpha = rr / den
                    x += alpha * p
                    r -= alpha * ap
                    rr_next = float(r @ r)
                rr_old, rr = rr, rr_next
                niter += 1

            # Restore the small Galerkin constraints after accumulated roundoff.
            if Q is not None:
                coefficients = Q.T @ r
                x += Q @ coefficients
                r -= AQ @ coefficients
            numerator = float(x @ b)
            denominator = numerator - float(x @ r)
            if denominator > 0.0 and numerator > 0.0:
                scale = numerator / denominator
                if math.isfinite(scale) and abs(scale - 1.0) > 1e-10:
                    x *= scale
                    r = b - scale * (b - r)
        except Exception as exc:
            error = type(exc).__name__
            if time.monotonic() >= deadline:
                self.exhausted = True

        if numeric_failure or not np.isfinite(x).all():
            x = safe_x if np.isfinite(safe_x).all() else np.zeros_like(b)
            self.history = []
            p = None
        nx = float(np.linalg.norm(x))
        if not math.isfinite(nx) or nx > 4.0 * normb / mu:
            x = np.zeros_like(b)
            nx = 0.0
            self.history = []
            p = None
        if nx > 1e-100 and math.isfinite(nx):
            self.history.append(x / nx)
            if niter and p is not None:
                pn = float(np.linalg.norm(p))
                if pn > 1e-100 and math.isfinite(pn):
                    self.history.append(p / pn)
            self.history = self.history[-rank:]
            self.rhs_history.append(b / normb)
            self.solution_history.append(x / normb)
            self.rhs_history = self.rhs_history[-12:]
            self.solution_history = self.solution_history[-12:]

        diagnostics = {"iterations": niter, "hvp_vectors": calls,
                       "relative_residual_squared": float(r @ r) / bn,
                       "rank": 0 if Q is None else Q.shape[1]}
        if self.verbose and (tick % 5 == 0 or tick == steps - 1):
            import sys
            import resource
            print("SOLVER", tick, diagnostics, "prep", prep,
                  "seconds", time.monotonic() - start, "batch", self.batch_ratio,
                  "rss_mib", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  "exception", error, file=sys.stderr, flush=True)
        return x, diagnostics


def make_solver(dimension, seed, options):
    return Solver(dimension, seed, options)
