"""Shrinkage of low-frequency spectral departures after prewhitening."""

import numpy as np


def sym(a):
    return (a + a.T) / 2


def compress_periodograms(e, whitener):
    n, d = e.shape
    f = np.fft.rfft(e @ whitener, axis=0)[1:(n+1)//2] / np.sqrt(n)
    bands = np.unique(np.r_[np.arange(min(20, len(f))),
                            (np.geomspace(min(20,len(f)),len(f),100)).astype(int), len(f)])
    counts = np.diff(bands)
    h = []
    for i, j in zip(bands[:-1],bands[1:]):
        a = f[i:j]
        h.append((a.real.T @ a.real + a.imag.T @ a.imag)/(j-i))
    omega = 2-2*np.cos((bands[1:]+bands[:-1]+1)*np.pi/n)
    return np.asarray(h), counts, omega


def fit_peak(curve, h, counts, start, penalty=0.0):
    hi = min(100.0, 0.985/max(1e-6, -curve.min()))
    lo = -0.975/curve.max()
    estimate = float(np.clip(start, lo, hi))

    def merit(a):
        f = 1+curve*a
        return -np.dot(counts, np.log(f) + h*(1/f - 1)) - 0.5*penalty*a*a

    value = merit(estimate)
    if value < 0:
        estimate = 0.0
        value = 0.0
    for _ in range(8):
        f = 1+curve*estimate
        w = (curve/f)*counts
        g = np.dot(w, h/f - 1) - penalty*estimate
        info = np.dot(w, curve/f) + penalty
        step = g / info
        if abs(step) < 0.0005 * max(0.5, abs(estimate)):
            break
        for j in range(5):
            a = float(np.clip(estimate + 0.55**j*step, lo, hi))
            u = merit(a)
            if u >= value:
                value = u
                estimate = a
                break
        else:
            break
    return estimate, value, info


def pole_adjust(e, cautious=True, diagnostics=None, span=8.0, guard=0.0):
    n, d = e.shape
    c = e.T @ e / n
    vals, vecs = np.linalg.eigh(c)
    vals = np.maximum(vals, max(1e-30, vals[-1]*1e-12))
    w = vecs/np.sqrt(vals)
    unw = vecs*np.sqrt(vals)
    h, counts, omega = compress_periodograms(e, w)
    modes = 1.55**np.arange(1,25)
    modes = modes[modes < n/17]
    curve = 1/(1 + (modes**2-modes)[:,None]*omega)
    avg = curve@counts/counts.sum()
    curve -= avg[:,None]
    variance = (curve*curve)@counts
    signal = (curve*counts) @ h.reshape(len(h),d*d)
    signal = (signal/variance[:,None]).reshape(-1,d,d)
    ident = np.eye(d)
    candidates = []
    for i, time in enumerate(modes):
        ke = curve[i]
        B0 = sym(signal[i])
        err = 1/np.sqrt(variance[i])
        if np.linalg.norm(B0) < (0.95 if cautious else .88)*err*np.sqrt(d*(d+1)/2):
            continue
        bs, v = np.linalg.eigh(B0)
        sh = np.einsum("ij,kil,lj->kj", v,h,v)
        floor = (2.6 if cautious else 2.0) + guard
        additional = max(0., np.log(time/max(7.0, 3.0*span)))*0.18
        if time > n/100:
            additional += 0.23 * np.log(100*time/n)
        best = []
        for j in np.argsort(-abs(bs)):
            start = bs[j]
            if abs(start) < max(0.075,(1.05+additional)*err):
                continue
            a, imp, info = fit_peak(ke, sh[:,j], counts, start, penalty=.035)
            best.append((imp, a, info, j))
        dc = np.zeros(d)
        score = 0.
        rank = 0
        for imp, a, info, j in sorted(best,reverse=True):
            # The direction is chosen from the same periodograms. Pay for
            # that search, rather than treating it as a fixed scalar fit.
            threshold = 0.83*(d-rank) + floor + additional*1.8
            extra = imp-threshold
            if extra <= 0:
                continue
            shrinkage = extra / max(1e-10,imp)
            if a > 0:
                shrinkage /= (1+0.11 / max(0.006, info*a))
                a = min(a, 1.3*np.sqrt(n/time))
            dc[j] = a*shrinkage
            score += extra
            rank += 1
        scalar = float(np.mean(bs))
        tr_score = 0.0
        if abs(scalar)*np.sqrt(d) > 1.6*err:
            a, imp, info = fit_peak(ke, np.mean(sh,axis=1), counts, scalar)
            imp *= d
            threshold = (4.0 if cautious else 3.1) + additional*1.6 + guard
            extra = imp-threshold
            if extra > 0:
                tr_score = extra
                a *= extra/imp
                diff = 0.82*a*ident
        if score > 0 or tr_score > 0:
            if score > tr_score*1.2:
                diff = (v*dc)@v.T
                if tr_score > 0:
                    diff += 0.25*(0.82*a-np.mean(dc))*ident
            elif score > 0:
                diff = 0.6*diff + 0.4*((v*dc)@v.T)
            merit = max(score,tr_score) + 0.3*min(score,tr_score)
            candidates.append((merit, time, (1-avg[i])*diff))
    if not candidates:
        return np.zeros((d,d))
    ms = max(x[0] for x in candidates)
    mix = [(a*np.exp(min(0.,0.55*(a-ms))),t,b) for a,t,b in candidates if a>=ms-9]
    total = sum(x[0] for x in mix)
    out = sum(a*b for a,_,b in mix)/total
    if diagnostics is not None:
        diagnostics.update(pole_scale=sum(a*t for a,t,_ in mix)/total, pole_evidence=ms)
    return sym(unw @ out @ unw.T)
