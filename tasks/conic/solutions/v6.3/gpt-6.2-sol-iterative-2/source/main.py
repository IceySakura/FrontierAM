from time import perf_counter
START = perf_counter()

import os
import sys
import warnings
START -= float(os.environ.get('CONIC_PRELUDE_SECONDS','0'))

import numpy as np
from fastlp import norm as maxabs, Data, output, run as fast_polyhedral


def solve(p, verbose=0, presolve=True, fastpoint=None, chord=True, dualize=True, components=True):
    x0, z0 = np.zeros(p.n), np.zeros(p.m)
    if p.qn <= 1.e-8:
        answer = p.optimal(x0, z0, polish_slack=False)
        if answer is not None:
            return answer
    if p.qn:
        free = (np.diff(p.A.indptr) == 0) & (np.diff(p.P.indptr) == 0)
        if np.any(free):
            scores = np.where(free, np.abs(p.q), 0.)
            idx = int(np.argmax(scores))
            if scores[idx] > 1.02e-8*p.qn:
                x0[idx] = -np.sign(p.q[idx])
                answer = p.dual_cert(x0)
                if answer is not None:
                    return answer
    if not p.K.nonlinear and p.A.nnz <= p.m:
        from poly import solve_boxes
        answer = solve_boxes(p, max(START+.85,perf_counter()+.03))
        if answer is not None:
            return answer
    if components and p.n>=1800 and (p.P.nnz or p.K.nonlinear):
        from components import decompose
        ans=decompose(p,lambda g:solve(g,verbose=verbose,components=False),
                      max(START+1.1,perf_counter()+.08))
        if ans is not None:
            return ans
    if dualize and p.K.nonlinear and p.P.nnz==0 and p.n==p.m-len(p.K.eq):
        from dualize import apply
        g,op=apply(p)
        if op is not None:
            if verbose:
                print('dualize',(p.n,p.m),'->',(g.n,g.m),flush=True)
            rr=solve(g,verbose=verbose,dualize=False,components=components)
            if rr is not None:
                ans=op.restore(rr)
                if ans is not None:
                    return ans
                if rr['status']=='solved' and perf_counter()<START+1.8:
                    # A residual acceptable in the dual formulation can be
                    # too large after separately scaled KKT terms cancel.
                    g.tol=min(g.tol,1.2e-9)
                    rr=solve(g,verbose=verbose,dualize=False,components=components)
                    if rr is not None:
                        ans=op.restore(rr)
                        if ans is not None:
                            return ans
    reduce_poly = bool(p.P.nnz) and p.K.eq.size>max(100,p.n//4) and fastpoint is None
    if (presolve and (p.K.nonlinear or reduce_poly) and p.A.nnz < 1_300_000
            and len(p.K.blocks) < 10_000 and p.n+p.m < 190_000):
        from presolve import reduce
        rd, undo, ans = reduce(p)
        if ans is not None:
            return ans
        if undo:
            if verbose:
                print('presolve', (p.n,p.m), '->', (rd.n,rd.m), flush=True)
            ans = solve(rd,verbose=verbose,presolve=False,chord=chord,dualize=dualize,
                        components=components)
            if ans is not None:
                for op in reversed(undo):
                    ans = op.restore(ans)
                    if ans is None:
                        break
                if ans is not None:
                    return ans
    if p.n==1 and p.K.psd and not p.K.groups:
        from pencil import solve_pencil
        ans = solve_pencil(p)
        if ans is not None:
            return ans
    if (p.P.nnz or p.K.nonlinear) and 36<=p.n<=795 and p.K.eq.size>=.48*p.n:
        from nullspace import reduce
        g,op,c=reduce(p)
        if c is not None:
            return c
        if op is not None:
            if verbose:
                print('nullspace',(p.n,p.m),'->',(g.n,g.m),flush=True)
            r=solve(g,verbose=verbose,components=components)
            if r is not None:
                ans=op.restore(r)
                if ans is not None:
                    return ans
    if chord and p.K.psd:
        from chordal import decompose
        r, op = decompose(p,perf_counter()+.18)
        if op is not None:
            if verbose:
                print('chordal', (p.n,p.m), '->', (r.n,r.m), flush=True)
            rr = solve(r,verbose=verbose,presolve=False,chord=False,dualize=dualize,
                       components=components)
            if rr is not None:
                ans = op.restore(rr,perf_counter()+.8)
                if ans is not None:
                    return ans
    if not p.K.nonlinear and p.n and p.m:
        from lp import solve_linear
        size = p.n+p.m
        risk = 7.0 if size <= 25_000 else 15.
        sec = .23 if size <= 10_000 else max(.2, .88-(perf_counter()-START))
        if maxabs(p.A.data) > 1.e10:
            scale = True
        else:
            scale = False
        if fastpoint is not None and p.P.nnz and p.n+p.m<35_000:
            from poly import polish_polyhedral
            answer = polish_polyhedral(p,*fastpoint,max(START+.83,perf_counter()+.027))
            if answer is not None:
                return answer
        if not p.K.lpos.size and p.P.nnz and size<90_000:
            from poly import polish_polyhedral
            answer=polish_polyhedral(p,x0,z0,perf_counter()+.24)
            if answer is not None:
                return answer
        if p.P.nnz and perf_counter()<START+.39 and 150<=p.n<=13_500:
            from poly import few_active
            answer=few_active(p,min(START+.58,perf_counter()+.095))
            if answer is not None:
                return answer
        if (p.P.nnz and 1<=p.m<=155 and p.n>=140 and p.m<=.16*p.n
                and perf_counter()<START+.48):
            from qpdual import solve as small_dual
            answer=small_dual(p,lambda g:solve(g,verbose=verbose,components=False),
                              START+.64)
            if answer is not None:
                return answer
        # Large curved QPs are usually better served by the interior-point routine.
        if (not p.P.nnz or p.n<70) and fastpoint is None and perf_counter()<START+.68:
            answer = solve_linear(p, limit=sec, scale_first=scale)
            if answer is not None:
                return answer
    else:
        risk = 9.0 if p.n+p.m < 250_000 else 16.0

    from ipm import solve_ipm
    from dense import suitable
    deadline = START+risk
    if os.environ.get('CONIC_FORCE_SPLIT'):
        from split import solve_split
        return solve_split(p,deadline,verbose=verbose)[0]
    dense = suitable(p.A,p.K) and not os.environ.get('CONIC_NO_DENSE')
    point = None
    if (p.K.nonlinear and not p.K.psd and 110<=p.n+p.m<=26000
            and p.n>0 and p.A.nnz<250000 and p.P.nnz<380000
            and all(d<=18 for k,d,a,b in p.K.nonlinear)
            and not os.environ.get('CONIC_NO_CORESOC')):
        import native
        left=min(.24 if dense and p.n>375 else .45,START+.82-perf_counter())
        if native.lib is not None and left>.08:
            status,x,s,z=native.core_soc(p,left)
            if status==1:
                ans=p.optimal(x,z,s)
                if ans is not None:
                    return ans
                point=x,s,z
                if p.n+p.m<3800 and p.A.nnz<55_000 and perf_counter()<START+.81:
                    from polish import polish_cone
                    ans=polish_cone(p,point,min(START+.9,perf_counter()+.085),steps=2)
                    if ans is not None:
                        return ans
            elif status==2:
                ans=p.primal_cert(z) or p.dual_cert(x,s)
                if ans is not None:
                    return ans
    big = p.n+p.m>130_000
    high_fill = p.A.nnz > 4.5*(p.m+p.n) or p.P.nnz > 5*p.n
    if big and high_fill and not p.K.psd and (p.K.nonlinear or p.P.nnz):
        from split import solve_split
        answer, point = solve_split(p,min(deadline,START+3.0),maxiter=260,verbose=verbose)
        if answer is not None:
            return answer
    initial_lim = min(deadline,START+(5.0 if dense else (9.0 if big else 6.0)))
    if not (p.n+p.m>650_000 and high_fill and point is None):
        plain = (not p.K.nonlinear and p.P.nnz and p.n>=1100
                 and p.K.lpos.size and p.K.eq.size<.22*p.n
                 and not os.environ.get('CONIC_NO_PLAIN'))
        if plain and perf_counter()<START+.57:
            ans,pt0=solve_ipm(p,min(initial_lim,START+1.25),maxiter=24,
                              verbose=verbose,dense=dense,plain=True,warm=fastpoint)
            if ans is not None:
                return ans
            if pt0 is not None:
                fastpoint=pt0[0],pt0[2]
                point=pt0
        answer, ip = solve_ipm(p,initial_lim,verbose=verbose,dense=dense,
                              fixed_tau=bool(p.P.nnz) and not p.K.nonlinear,
                              warm=fastpoint if p.P.nnz else None)
        if answer is not None:
            return answer
        if ip is not None:
            point = ip
    if point is not None and not p.K.nonlinear and perf_counter() < deadline:
        from poly import polish_polyhedral
        answer = polish_polyhedral(p,point[0],point[2],deadline)
        if answer is not None:
            return answer
    if point is not None and p.K.nonlinear and not p.K.psd and perf_counter() < deadline:
        from polish import polish_cone
        answer = polish_cone(p,point,deadline)
        if answer is not None:
            return answer
    if point is not None and p.K.psd and perf_counter() < deadline:
        from polish import polish_sdp
        answer = polish_sdp(p,point,min(deadline,perf_counter()+.16))
        if answer is not None:
            return answer
    if point is not None and p.K.nonlinear and not p.K.exp and perf_counter()<deadline-.02:
        from semismooth import polish
        ans=polish(p,point,min(deadline,perf_counter()+.24))
        if ans is not None:
            return ans
    if perf_counter() < deadline-.025 and p.K.exp:
        if not p.P.nnz and not p.K.psd:
            from cuts import solve_cuts
            answer=solve_cuts(p,min(deadline,perf_counter()+.78),point)
            if answer is not None:
                return answer
        answer, backup = solve_ipm(p,deadline,maxiter=65,exp_dual=True,
                                   rhs_scale=False,verbose=verbose,dense=dense)
        if answer is not None:
            return answer
        if backup is not None and perf_counter() < deadline:
            from polish import polish_cone
            answer = polish_cone(p,backup,deadline)
            if answer is not None:
                return answer
            if point is None:
                point = backup
    elif perf_counter() < deadline-.025 and p.K.psd and p.m < 1800:
        from kkt import can_qr
        use_qr = can_qr(p.A, p.K)
        answer, backup = solve_ipm(p, deadline, maxiter=60, condense=False, qr=use_qr,
                                   verbose=verbose)
        if answer is not None:
            return answer
        if backup is not None and perf_counter() < deadline:
            from polish import polish_sdp
            answer = polish_sdp(p,backup,min(deadline,perf_counter()+.16))
            if answer is not None:
                return answer
        if use_qr and perf_counter() < deadline-.1 and p.m<1700:
            answer, _ = solve_ipm(p, deadline, maxiter=47, condense=False,
                                  verbose=verbose)
            if answer is not None:
                return answer
    elif perf_counter()<deadline-.07 and p.K.nonlinear and not p.K.psd:
        if dense or point is None:
            answer, backup = solve_ipm(p,min(deadline,perf_counter()+2.4),maxiter=55,
                      start_style=0,bounds=not dense,rhs_scale=not dense,verbose=verbose)
            if answer is not None:
                return answer
            if point is None:
                point = backup
    if (not p.K.nonlinear and p.n and p.m and perf_counter()<deadline-.13
            and (not p.P.nnz or p.n<=14000)):
        if p.P.nnz and not os.environ.get('CONIC_NO_ALM'):
            from alm import solve_alm
            ans=solve_alm(p,min(deadline,perf_counter()+1.75),
                          point=point if point is not None else fastpoint,verbose=verbose)
            if ans is not None:
                return ans
        from lp import solve_linear
        alt_mode = 'ipm' if not p.P.nnz and (p.n+p.m<90_000) else 'choose'
        answer = solve_linear(p,limit=min(deadline-perf_counter(),4.9),
                              scale_first=p.A.nnz<500000,mode=alt_mode)
        if answer is not None:
            return answer
    if perf_counter()<deadline-.07:
        from split import solve_split
        answer, _ = solve_split(p,deadline,point,verbose=verbose)
        if answer is not None:
            return answer
    return None


def main():
    args = dict(zip(sys.argv[1::2],sys.argv[2::2]))
    infile, outfile = args.get('--input'),args.get('--output')
    if not infile or not outfile:
        print('usage: run.sh --input INPUT.npz --output RESULT.npz',file=sys.stderr)
        return 2
    verbose = int(os.environ.get('CONIC_VERBOSE', '0'))
    ans, pt = None,None
    watchdog = None
    def arm_watchdog():
        nonlocal watchdog
        import native
        if native.lib is not None:
            output(outfile,None)
            watchdog = native.lib.conic_watchdog
            watchdog(max(1000,int(1000.*(23.-(perf_counter()-START)))))
    with np.errstate(all='ignore'), warnings.catch_warnings():
        if not verbose:
            warnings.simplefilter('ignore', RuntimeWarning)
        try:
            data = Data(infile)
            if data.n+data.m>91000 or len(data.A.val)+len(data.P.val)>1_000_000:
                arm_watchdog()
            if not os.environ.get('CONIC_NO_FASTLP'):
                try:
                    ans, pt, _ = fast_polyhedral(data,START)
                except (ValueError,TypeError,RuntimeError,ImportError,AttributeError,MemoryError):
                    if verbose:
                        import traceback
                        traceback.print_exc()
            if ans is None:
                if watchdog is None:
                    arm_watchdog()
                p = data.inflate()
                del data
                ans = solve(p, verbose, fastpoint=pt)
        except Exception:
            if verbose:
                import traceback
                traceback.print_exc()
            ans = None
    if watchdog is not None:
        watchdog(0)
    output(outfile, ans)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
