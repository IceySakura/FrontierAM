"""Adaptive Gaussian cubature."""

import numpy as np

from cubature import normals, quadrature_control
from ellipsoid import ellipse_estimate, get_jump, search_large_jump
from kinks import Atlas, fit_kinks, debug
from options import digital
from regression import regression_option
from protocol import Deadline, Session
from smooth import smooth_control, smooth_correct


def ordinary(s, rng):
    q = min(24, max(0, (s.remaining - 8192) // (s.d + 1)))
    basis = np.eye(s.d)
    if q:
        a = rng.standard_normal((q, s.d))
        h = 2e-5
        pts = np.concatenate([a[:, None, :], a[:, None, :] + h * basis[None]], axis=1)
        yy = s.query(pts).reshape(q, s.d + 1)
        g = (yy[:, 1:] - yy[:, :1]) / h
        np.clip(g, -1e8, 1e8, out=g)
        basis = np.linalg.eigh(g.T @ g)[1][:, ::-1]
    return quadrature_control(s, rng, basis)


def main():
    s = Session()
    rng = np.random.default_rng(s.seed & ((1 << 64)-1))
    f0 = float("nan")
    try:
        zp = normals(rng, 8, s.d)
        pilot = s.query(zp)
        f0 = np.mean(pilot)
        s.arm()
        if np.any(pilot == 0):
            pp = pilot[pilot > 0]
            ans = None
            basis = None
            if len(pp) and np.all(pp == pp[0]):
                ans = digital(s, rng, zp, pilot, pp[0])
            else:
                ans,basis = regression_option(s,rng,zp,pilot)
            if ans is not None:
                f0 = ans
            elif basis is not None:
                f0 = quadrature_control(s,rng,basis)
            else:
                f0 = ordinary(s, rng)
        else:
            atlas = Atlas(s, rng)
            jump = get_jump(s, rng, atlas, zp, pilot)
            if jump is not None:
                answer = ellipse_estimate(s, rng, *jump, zp, pilot, atlas)
                if answer is not None and np.isfinite(answer):
                    s.disarm()
                    s.result(answer)
                    return
            model, patch = fit_kinks(s, atlas)
            if model is None and jump is None and s.remaining > 10500:
                result=search_large_jump(s,rng,atlas,zp,pilot)
                if result is not None:
                    ans=ellipse_estimate(s,rng,*result,zp,pilot,atlas)
                    if ans is not None and np.isfinite(ans):
                        s.disarm()
                        s.result(ans)
                        return
            if model is None:
                f0 = ordinary(s, rng)
            else:
                model=model.canonicalize(zp,pilot)
                p, U, H = patch[1:]
                st = p + np.array([[-0.00008], [0.00008]])*U[:,0]
                HH = H.copy()
                if model.kind == "sum":
                    for q, sig in zip(model.qs, model.sign):
                        HH -= 2*(U.T @ q.A @ U)[None] * (sig*q(st) > 0)[:, None, None]
                else:
                    val=np.stack([np.zeros(2),*(q(st) for q in model.qs)])
                    for i,q in enumerate(model.qs,1):
                        HH -= 2*model.sign*(U.T@q.A@U)[None]*(val.argmax(axis=0)==i)[:,None,None]
                initial = U @ HH.mean(axis=0) @ U.T if np.isfinite(HH).all() else None
                rotation, cv, integral = smooth_control(s, model, rng, initial=initial)
                if cv is None:
                    total_control = model
                else:
                    total_control = lambda z: model(z) + cv(z)
                exact = model.integral(rng, 19 if s.d < 40 else 18) + integral
                f0 = smooth_correct(s, rng, rotation, total_control, exact)
        if not np.isfinite(f0):
            f0 = np.mean(pilot)
    except Exception as exc:
        debug("falling back after",type(exc).__name__,str(exc))
        if not np.isfinite(f0):
            raise
        if not isinstance(exc,Deadline) and s.remaining>=1024 and s.elapsed<s.limit-6.:
            try:
                val=quadrature_control(s,rng,np.eye(s.d))
                if np.isfinite(val):
                    f0=val
            except Exception as exc:
                debug("fallback shortened",type(exc).__name__)
    finally:
        s.disarm()
    s.result(f0)


if __name__ == "__main__":
    main()
