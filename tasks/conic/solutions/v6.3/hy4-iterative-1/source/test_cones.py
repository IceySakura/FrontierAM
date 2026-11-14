"""Validation for /workspace/submission/cones.py.

Run with:  python /workspace/submission/test_cones.py

Checks central finite differences of the barrier and of its gradient, dual
interiority of -grad(s), exact agreement with the independent checker's
``_cones`` defects, positive definiteness of the exponential Hessian, and the
fraction-to-boundary step rules.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/samples/io")
import check  # noqa: E402  (independent checker; scipy is available here)

from cones import ConeData, exp_grad_hess, smat, svec  # noqa: E402


RNG = np.random.default_rng(20261113)
TOL = 1e-5


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def cone_width(spec):
    d = spec["dim"]
    return d * (d + 1) // 2 if spec["type"] == "psd" else d


def barrier_value(cd, s):
    total = 0.0
    for b in cd.blocks:
        x = s[b["start"]:b["stop"]]
        kind = b["type"]
        if kind == "zero":
            continue
        if kind == "nonnegative":
            total -= float(np.sum(np.log(x)))
        elif kind == "soc":
            total -= math.log(float(x[0]) ** 2 - float(x[1:] @ x[1:]))
        elif kind == "psd":
            sign, logdet = np.linalg.slogdet(smat(x, b["dim"]))
            assert sign > 0.0
            total -= logdet
        else:  # exponential
            u, v, w = map(float, x)
            psi = v * (math.log(w) - math.log(v)) - u
            total -= math.log(psi) + math.log(v) + math.log(w)
    return total


def random_interior(cd, rng):
    s = np.zeros(cd.m, dtype=np.float64)
    for b in cd.blocks:
        sl = slice(b["start"], b["stop"])
        kind, dim, width = b["type"], b["dim"], b["width"]
        if kind == "zero":
            continue
        if kind == "nonnegative":
            s[sl] = np.exp(rng.uniform(-0.7, 0.7, size=width))
        elif kind == "soc":
            v = 0.3 * rng.normal(size=dim - 1)
            s[sl] = 0.0
            s[b["start"]] = float(np.linalg.norm(v)) + rng.uniform(0.4, 1.5)
            s[b["start"] + 1:b["stop"]] = v
        elif kind == "psd":
            a = rng.normal(size=(dim, dim))
            mat = a @ a.T + rng.uniform(0.5, 1.5) * np.eye(dim)
            s[sl] = svec(mat, dim)
        else:  # exponential
            v = math.exp(rng.uniform(-0.5, 0.5))
            u = float(rng.uniform(-1.0, 1.0))
            delta = float(rng.uniform(0.4, 1.2))
            w = v * math.exp(u / v + delta)
            s[sl] = (u, v, w)  # psi = v*delta > 0
    return s


def fd_grad(cd, s, step=1e-6):
    out = np.zeros(cd.m, dtype=np.float64)
    for i in range(cd.m):
        h = step * max(1.0, abs(float(s[i])))
        sp, sm = s.copy(), s.copy()
        sp[i] += h
        sm[i] -= h
        out[i] = (barrier_value(cd, sp) - barrier_value(cd, sm)) / (2.0 * h)
    return out


def fd_hess_apply(cd, s, Y, step=1e-6):
    out = np.zeros((cd.m, Y.shape[1]), dtype=np.float64)
    for j in range(cd.m):
        h = step * max(1.0, abs(float(s[j])))
        sp, sm = s.copy(), s.copy()
        sp[j] += h
        sm[j] -= h
        column = (cd.grad(sp) - cd.grad(sm)) / (2.0 * h)
        out += np.outer(column, Y[j, :])
    return out


def max_relerr(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b))))


def check_cost():
    return {"psd_order_limit": None, "psd_eigenvalue_calls": 0,
            "psd_eigenvalue_seconds": 0.0, "psd_max_order_checked": 0,
            "psd_max_matrix_bytes": 0}


# ---------------------------------------------------------------------------
# 0. bookkeeping / packing
# ---------------------------------------------------------------------------

def test_structure():
    mixed = [{"type": "zero", "dim": 2}, {"type": "nonnegative", "dim": 3},
             {"type": "soc", "dim": 4}, {"type": "psd", "dim": 3},
             {"type": "exponential", "dim": 3}]
    m = sum(cone_width(c) for c in mixed)
    cd = ConeData(mixed, m)
    assert cd.m == m and cd.nblocks == len(mixed)
    assert cd.nu == 0 + 3 + 2 + 3 + 3
    assert cd.s0().dtype == np.float64 and cd.s0().shape == (m,)
    assert cd.interior_primal(cd.s0())
    starts, stops = [], []
    pos = 0
    for b in cd.blocks:
        assert set(b) >= {"type", "dim", "width", "start", "stop"}
        assert b["start"] == pos and b["stop"] == pos + b["width"]
        starts.append(b["start"])
        stops.append(b["stop"])
        pos = b["stop"]
    assert pos == m
    assert np.array_equal(np.sort(np.concatenate((cd.zero_idx, cd.cone_idx))), np.arange(m))
    assert np.array_equal(np.sort(cd.zero_idx), np.arange(2))
    for dim in (1, 2, 3, 5):
        width = cone_width({"type": "psd", "dim": dim})
        v = RNG.normal(size=width)
        assert np.allclose(svec(smat(v, dim), dim), v)
        mat = RNG.normal(size=(dim, dim))
        mat = mat + mat.T
        assert np.allclose(smat(svec(mat, dim), dim), mat)
        vb = RNG.normal(size=(width, 4))
        assert smat(vb, dim).shape == (4, dim, dim)
        assert np.allclose(svec(smat(vb, dim), dim), vb)
    for bad in ([{"type": "soc", "dim": 1}], [{"type": "exponential", "dim": 2}],
                [{"type": "unknown", "dim": 1}], [{"type": "psd", "dim": 0}]):
        try:
            ConeData(bad, 100)
            raise AssertionError("accepted invalid cone spec %r" % bad)
        except ValueError:
            pass
    try:
        ConeData([{"type": "nonnegative", "dim": 2}], 3)
        raise AssertionError("accepted mismatched m")
    except ValueError:
        pass
    return m


# ---------------------------------------------------------------------------
# 1/2. finite differences, dual interior of -grad, Hessian PD
# ---------------------------------------------------------------------------

def test_barrier_calculus():
    specs = [
        [{"type": "zero", "dim": 3}],
        [{"type": "nonnegative", "dim": 4}],
        [{"type": "soc", "dim": 2}],
        [{"type": "soc", "dim": 5}],
        [{"type": "psd", "dim": 1}],
        [{"type": "psd", "dim": 3}],
        [{"type": "psd", "dim": 4}],
        [{"type": "exponential", "dim": 3}],
        [{"type": "zero", "dim": 2}, {"type": "nonnegative", "dim": 2},
         {"type": "soc", "dim": 3}, {"type": "psd", "dim": 2},
         {"type": "exponential", "dim": 3}],
    ]
    worst_g, worst_h, min_dense_eig, min_exp_eig = 0.0, 0.0, math.inf, math.inf
    for spec in specs:
        cd = ConeData(spec, sum(cone_width(c) for c in spec))
        assert cd.nu == sum({"zero": 0, "nonnegative": c["dim"], "soc": 2,
                             "psd": c["dim"], "exponential": 3}[c["type"]]
                            for c in spec)
        for _ in range(5):
            s = random_interior(cd, RNG)
            assert cd.interior_primal(s), "generated point not interior"
            g = cd.grad(s)
            g_fd = fd_grad(cd, s)
            e = max_relerr(g, g_fd)
            worst_g = max(worst_g, e)
            assert e < TOL, ("grad finite difference", spec, e)

            Y = RNG.normal(size=(cd.m, 3))
            hy = cd.hess_apply(s, Y)
            hy_fd = fd_hess_apply(cd, s, Y)
            e = max_relerr(hy, hy_fd)
            worst_h = max(worst_h, e)
            assert e < TOL, ("Hessian finite difference", spec, e)
            e = max_relerr(cd.hess_apply(s, Y[:, 0]), hy_fd[:, 0])
            assert e < TOL, ("Hessian 1-D shape", spec, e)

            hdense = cd.hess_apply(s, np.eye(cd.m))
            assert max_relerr(hdense, hdense.T) < 1e-10
            if cd.cone_idx.size:
                hc = hdense[np.ix_(cd.cone_idx, cd.cone_idx)]
                lam = float(np.linalg.eigvalsh(hc)[0])
                min_dense_eig = min(min_dense_eig, lam)
                assert lam > 0.0, ("Hessian not positive definite", spec, lam)

            # -grad(s) is in the strict interior of the dual cone.
            z = -g
            assert cd.interior_dual(z), ("-grad not dual interior", spec, z)

    # Explicit numerical eigenvalues of the exponential 3x3 Hessian.
    cd_exp = ConeData([{"type": "exponential", "dim": 3}], 3)
    for _ in range(200):
        s = random_interior(cd_exp, RNG)
        g, H = exp_grad_hess(s)
        assert max_relerr(g, cd_exp.grad(s)) < 1e-14
        assert max_relerr(H, H.T) < 1e-14
        lam = float(np.linalg.eigvalsh(H)[0])
        min_exp_eig = min(min_exp_eig, lam)
        assert lam > 0.0, ("exponential Hessian not positive definite", s, lam)
    return worst_g, worst_h, min_dense_eig, min_exp_eig


# ---------------------------------------------------------------------------
# 3. exact agreement with check.py::_cones
# ---------------------------------------------------------------------------

def test_defects_match_checker():
    mixed = [{"type": "zero", "dim": 2}, {"type": "nonnegative", "dim": 3},
             {"type": "soc", "dim": 4}, {"type": "psd", "dim": 3},
             {"type": "exponential", "dim": 3}]
    m = sum(cone_width(c) for c in mixed)
    cd = ConeData(mixed, m)
    check_cones = [(b["type"], b["dim"], b["width"]) for b in cd.blocks]

    vectors = []
    for _ in range(150):
        scale = 10.0 ** RNG.uniform(-6.0, 6.0)
        vectors.append(RNG.normal(size=m) * scale)
    # Crafted vectors exercising the exponential closure/boundary branches.
    crafted = np.zeros(m)
    exp_sl = slice(cd.blocks[-1]["start"], cd.blocks[-1]["stop"])
    specials = [
        (-1.0, 0.0, 2.0),   # closure point: u<=0, v=0, w>=0
        (1.0, 0.0, 2.0),    # v=0, u>0 -> violation 1
        (0.0, 1.0, 0.0),    # w=0 -> violation 1
        (0.0, 1.0, 3.0),    # interior
        (1.0, 1.0, 3.0),    # interior boundary of the log inequality
        (2.0, 1.0, 3.0),    # violates log inequality
        (0.0, -1.0, 3.0),   # v<0
        (0.0, 1.0, -3.0),   # w<0
        (1e8, 1e-8, 1e8),
        (-1e8, 1e-8, 1e8),
    ]
    for triple in specials:
        v = crafted.copy()
        v[exp_sl] = triple
        vectors.append(v)
    vectors.append(np.zeros(m))
    vectors.append(np.ones(m))
    vectors.append(-np.ones(m))

    worst_diff = 0.0
    singles = []
    for b in cd.blocks:
        spec = {"type": b["type"], "dim": b["dim"]}
        singles.append((ConeData([spec], b["width"]), b["start"], b["stop"]))

    for v in vectors:
        for dual in (False, True):
            checker_worst, blocks = check._cones(v, check_cones, dual, check_cost())
            ours = cd.defect_dual(v) if dual else cd.defect_primal(v)
            diff = abs(ours - checker_worst)
            worst_diff = max(worst_diff, diff)
            assert diff <= 1e-12, ("checker worst mismatch", dual, diff, ours, checker_worst)
            for (cd1, start, stop), ref in zip(singles, blocks):
                local = cd1.defect_dual(v[start:stop]) if dual else cd1.defect_primal(v[start:stop])
                d = abs(local - ref["residual"])
                worst_diff = max(worst_diff, d)
                assert d <= 1e-12, ("checker block mismatch", dual, ref, local)
    return worst_diff, len(vectors)


# ---------------------------------------------------------------------------
# 4. max_step: 0.99*alpha interior, 1.01*alpha not interior
# ---------------------------------------------------------------------------

def check_step(cd, x, dx, dual, label, expect=None):
    interior = cd.interior_dual if dual else cd.interior_primal
    alpha = cd.max_step_dual(x, dx) if dual else cd.max_step_primal(x, dx)
    assert 0.0 <= alpha <= 1.0, (label, alpha)
    if expect is not None:
        assert abs(alpha - expect) < 1e-4, (label, alpha, expect)
    if alpha < 1.0:
        assert interior(x + 0.99 * alpha * dx), (label, "0.99*alpha not interior", alpha)
        assert not interior(x + 1.01 * alpha * dx), (label, "1.01*alpha interior", alpha)
    else:
        assert interior(x + dx), (label, "full step not interior")
    return alpha


def test_max_step():
    results = []
    # Closed-form cap checks: boundary at alpha=0.5 -> 0.995*0.5.
    specs = {
        "nonnegative": ([{"type": "nonnegative", "dim": 3}],
                        np.ones(3), np.array([-2.0, 0.1, 0.1])),
        "soc": ([{"type": "soc", "dim": 4}],
                np.array([2.0, 0.0, 0.0, 0.0]), np.array([-4.0, 0.0, 0.0, 0.0])),
        "psd": ([{"type": "psd", "dim": 3}],
                svec(np.eye(3), 3), svec(-2.0 * np.eye(3), 3)),
    }
    for name, (spec, s, ds) in specs.items():
        cd = ConeData(spec, sum(cone_width(c) for c in spec))
        results.append(("primal_" + name, check_step(cd, s, ds, False, name, 0.4975)))
        results.append(("dual_" + name, check_step(cd, s, ds, True, name + "_dual", 0.4975)))

    # Exponential: bisection on v=0 boundary (primal) and on z -> 0 (dual).
    cd = ConeData([{"type": "exponential", "dim": 3}], 3)
    s = np.array([0.0, 1.0, 3.0])
    ds = np.array([-1.0, -2.0, -2.0])
    results.append(("primal_exp", check_step(cd, s, ds, False, "exp", 0.4975)))
    z = -cd.grad(s)
    assert cd.interior_dual(z)
    results.append(("dual_exp", check_step(cd, z, -2.0 * z, True, "exp_dual", 0.4975)))

    # Mixed product, random interior points and random directions.
    mixed = [{"type": "zero", "dim": 2}, {"type": "nonnegative", "dim": 3},
             {"type": "soc", "dim": 4}, {"type": "psd", "dim": 3},
             {"type": "exponential", "dim": 3}]
    cd = ConeData(mixed, sum(cone_width(c) for c in mixed))
    n_alpha_lt1 = 0
    for _ in range(40):
        s = random_interior(cd, RNG)
        ds = 0.5 * RNG.normal(size=cd.m)
        ds[cd.zero_idx] = 0.0
        a = check_step(cd, s, ds, False, "mixed_primal")
        n_alpha_lt1 += int(a < 1.0)
        z = -cd.grad(s)
        dz = 0.5 * RNG.normal(size=cd.m)
        check_step(cd, z, dz, True, "mixed_dual")
    assert n_alpha_lt1 > 0, "no finite primal boundary was exercised"

    # Zero-cone rows must stay zero in the primal; dual zero rows are free.
    s = cd.s0()
    z = -cd.grad(s)
    assert cd.interior_dual(z)
    ds = np.zeros(cd.m)
    ds[cd.zero_idx[0]] = 1.0
    assert cd.max_step_primal(s, ds) == 0.0
    assert cd.max_step_dual(z, ds) == 1.0

    # s0 and margin behaviour.
    assert cd.interior_primal(cd.s0())
    assert cd.interior_primal(cd.s0(), 1e-3)
    assert not cd.interior_primal(-cd.s0())
    return results


def main():
    m = test_structure()
    print("structure: mixed cone with m=%d, packing round trips and validation OK" % m)
    worst_g, worst_h, min_dense_eig, min_exp_eig = test_barrier_calculus()
    print("finite differences: max grad err = %.3e, max hess err = %.3e" %
          (worst_g, worst_h))
    print("Hessian eigenvalues: min over cone rows = %.6e, min exp 3x3 = %.6e (both > 0)" %
          (min_dense_eig, min_exp_eig))
    worst_diff, nvectors = test_defects_match_checker()
    print("checker defects: %d vectors, max |difference| = %.3e" % (nvectors, worst_diff))
    results = test_max_step()
    print("max_step: %d targeted cases, all fraction-to-boundary checks passed" % len(results))
    for name, alpha in results:
        print("  %-22s alpha = %.8f" % (name, alpha))
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
