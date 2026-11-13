import math
import time
import numpy as np

try:
    import spd_core
except ImportError:
    spd_core = None


class Solver:
    def __init__(self, dimension, seed, options):
        self.d = int(dimension)
        self.native = spd_core is not None and options.get("native", True)
        self.restart_weight = options.get("first_weight")
        self.phase_prior = float(options.get("phase_prior", 0.02))
        self.verbose = options.get("verbose", False)
        self.iteration_cap = int(options.get("iteration_cap", 50000))
        self.x = None
        self.p = None
        self.b_previous = None
        self.phase_changes = 0
        self.call_cost = 0.004
        self.exhausted = False

    def _fallback(self, b, changed):
        if changed or self.x is None or self.b_previous is None:
            return np.zeros_like(b)
        denominator = float(self.b_previous @ self.b_previous)
        scale = float(b @ self.b_previous) / max(denominator, 1e-300)
        scale = min(1.5, max(0.0, scale))
        return scale * self.x

    def solve(self, hvp, b, context):
        start = time.monotonic()
        b = np.asarray(b, dtype=np.float64)
        step = int(context.get("step", 0))
        steps = max(step + 1, int(context.get("steps", 50)))
        changed = bool(context.get("phase_changed", step == 0)) or step == 0
        mu, upper = float(context["mu"]), float(context["L"])
        deadline = float(context.get("deadline", start + 28.0))
        if "sequence_deadline" in context:
            deadline = min(deadline, float(context["sequence_deadline"]))
        if "hard_deadline" in context:
            deadline = min(deadline, float(context["hard_deadline"]) - 0.35)
        if changed and step:
            self.phase_changes += 1
        previous_phase_x = self.x if changed else None
        previous_phase_b = self.b_previous if changed else None
        if changed:
            self.x = self.p = self.b_previous = None
        if self.exhausted or start >= deadline - 0.08:
            return self._fallback(b, changed)
        weight = self.restart_weight
        if weight is None:
            weight = min(6.0, max(1.0, 2.0 * (upper / mu / 3e5) ** 0.25))
        weight = max(1.0, float(weight))
        current_weight = weight if changed or self.x is None else 1.0
        remaining = steps - step
        # Keep some time available for later phase changes.
        expected_changes = ((self.phase_changes + 50.0 * self.phase_prior) /
                            (step + 50.0) * (remaining - 1))
        denominator = remaining - 1 + current_weight + (weight - 1.0) * expected_changes
        available = max(0.0, deadline - start - 0.15)
        stop = min(deadline - 0.08, start + available * current_weight / denominator)

        def query(v):
            if self.exhausted:
                return None
            now = time.monotonic()
            if now >= deadline - 0.065:
                self.exhausted = True
                return None
            if now + max(0.001, self.call_cost * 0.6) >= stop:
                return None
            try:
                result = hvp(v)
            except Exception as error:
                if ("budget" in type(error).__name__.lower()
                        or "deadline" in str(error).lower()
                        or time.monotonic() >= deadline - 0.08):
                    self.exhausted = True
                    return None
                raise
            elapsed = time.monotonic() - now
            self.call_cost = 0.9 * self.call_cost + 0.1 * elapsed
            return np.asarray(result, dtype=np.float64)

        if self.d <= 64:
            A = query(np.eye(self.d))
            if A is None:
                return self._fallback(b, changed)
            try:
                answer = np.linalg.solve(A, b)
                if np.isfinite(answer).all():
                    self.x = answer.copy()
                    self.b_previous = b.copy()
                    return answer
            except np.linalg.LinAlgError:
                pass

        x = np.zeros_like(b)
        r = b.copy()
        U = V = None
        queries = 0
        cached_product = None
        if previous_phase_x is not None:
            norm = float(np.linalg.norm(previous_phase_x))
            if math.isfinite(norm) and norm > 1e-100:
                product = query(previous_phase_x / norm)
                if product is not None:
                    queries += 1
                    alignment = float(product @ b)
                    scale = math.sqrt(max(0.0, float(product @ product) * float(b @ b)))
                    if scale > 0.0 and abs(alignment) > 0.85 * scale:
                        self.x = previous_phase_x
                        self.b_previous = previous_phase_b
                        cached_product = product
        if self.x is not None:
            columns, products = [], []
            for vector in (self.x, self.p):
                if vector is None:
                    continue
                norm = float(np.linalg.norm(vector))
                if not math.isfinite(norm) or norm < 1e-100:
                    continue
                column = vector / norm
                if vector is self.x and cached_product is not None:
                    product = cached_product
                    cached_product = None
                else:
                    product = query(column)
                    if product is not None:
                        queries += 1
                if product is None:
                    break
                columns.append(column)
                products.append(product)
            if columns:
                U = np.asfortranarray(np.column_stack(columns))
                V = np.asfortranarray(np.column_stack(products))
                # Whiten the refreshed space in the current energy inner product.
                E = U.T @ V
                E = 0.5 * (E + E.T)
                try:
                    values, rotation = np.linalg.eigh(E)
                    keep = values > max(1e-30, values[-1] * 1e-12)
                    if np.any(keep):
                        rotation = rotation[:, keep] / np.sqrt(values[keep])
                        U = np.asfortranarray(U @ rotation)
                        V = np.asfortranarray(V @ rotation)
                        coefficients = U.T @ b
                        x = U @ coefficients
                        r = b - V @ coefficients
                    else:
                        U = V = None
                except np.linalg.LinAlgError:
                    U = V = None

        bn = float(b @ b)
        rr = float(r @ r)
        p = r.copy() if U is None else r - U @ (V.T @ r)
        iterations = 0
        while iterations < self.iteration_cap and rr > max(1e-300, bn * 1e-20):
            if not math.isfinite(rr):
                break
            q = query(p)
            if q is None:
                break
            queries += 1
            if self.native:
                new_rr, _ = spd_core.step(x, r, p, q, U, V, rr)
                if new_rr < 0.0:
                    break
                rr = new_rr
            else:
                curvature = float(p @ q)
                if not math.isfinite(curvature) or curvature <= 0.0:
                    break
                alpha = rr / curvature
                x += alpha * p
                r -= alpha * q
                new_rr = float(r @ r)
                z = r if U is None else r - U @ (V.T @ r)
                p *= new_rr / rr
                p += z
                rr = new_rr
            iterations += 1

        if not math.isfinite(rr) or not np.isfinite(x).all() or not np.isfinite(r).all():
            self.p = None
            return self._fallback(b, changed)
        if queries == 0 and self.x is not None:
            return self._fallback(b, changed)
        self.x = x.copy()
        self.p = p.copy() if np.isfinite(p).all() else None
        self.b_previous = b.copy()
        # The spectral bound makes this final, query-free gradient step safe.
        answer = x + r / upper
        if not np.isfinite(answer).all():
            answer = x
        diagnostics = {
            "iterations": iterations,
            "hvp_vectors": queries,
            "relative_residual": math.sqrt(rr / bn) if bn > 0 and math.isfinite(rr) else 1.0,
            "seconds": time.monotonic() - start,
        }
        if self.verbose:
            import sys
            print(step, diagnostics, file=sys.stderr, flush=True)
        return answer, diagnostics


def make_solver(dimension, seed, options):
    return Solver(dimension, seed, options or {})
