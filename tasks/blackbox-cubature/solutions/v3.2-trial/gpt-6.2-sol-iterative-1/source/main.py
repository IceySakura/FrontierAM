import numpy as np

from common import Work, gradient, moment_rotation, report, rqm, unit
from cubature_protocol import Session


def main():
    session = Session()
    w = Work(session)
    try:
        x, y = w.pilot(min(768, w.left))
        u, cnt = np.unique(y, return_counts=True)
        if len(u) == 2 and cnt.min() >= 4:
            from options import digital
            est = digital(w, x, y, u)
        elif np.max(cnt) >= .08 * len(y) and len(u) != 1:
            from options import payoff
            est = payoff(w, x, y, u[np.argmax(cnt)])
        elif len(u) == 1 and w.left >= 4500:
            # Very rare nonnegative payoffs can leave the entire initial pilot flat.
            narrow = .67*w.normal(1024)
            v1 = w.ask(narrow)-u[0]
            pos = np.flatnonzero(v1>0.)
            if len(pos)>=3 and np.all(v1>=0.):
                from options import importance_payoff
                p = pos[np.argsort(v1[pos])][-min(len(pos),12):]
                _,G = gradient(w,narrow[p])
                mag = np.linalg.norm(G,axis=1)
                keep = (mag>0) & (mag<=5*np.median(mag)+1.e-12)
                direction = (
                    unit(np.mean(G[keep]/mag[keep,None],axis=0)) if np.any(keep)
                    else unit(v1@narrow)
                )
                est = importance_payoff(w,np.vstack([x,narrow]),np.r_[y-u[0],v1],
                                        direction,G[keep] if np.any(keep) else None,u[0])
            else:
                est = rqm(w, moment_rotation(narrow,v1))
        else:
            from hard import synthetic
            est = synthetic(w, x, y)
        if not np.isfinite(est):
            raise ArithmeticError('non-finite estimate')
    except Exception as error:
        report('cubature recovery:', type(error).__name__, str(error))
        est = w.safe if w.safe is not None else 0.
        if w.left >= 64 and w.elapsed() < min(18., w.limit*.60):
            try:
                est = rqm(w, moment_rotation(w.pilot_x, w.pilot_y))
            except Exception as recovery_error:
                report('fallback:', str(recovery_error))
    session.result(est)


if __name__ == '__main__':
    main()
