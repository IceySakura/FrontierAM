"""Long-run covariance from observed finite-state transitions."""

import numpy as np


def symmetric(a):
    return (a + a.T) * 0.5


def balance(a, margins, iterations=45):
    """Diagonal congruence scaling to prescribed row totals."""
    v = np.ones(len(a))
    for _ in range(iterations):
        r = margins / np.maximum(v * (a @ v), 1e-100)
        v *= np.sqrt(r)
        if np.max(np.abs(r - 1)) < 1e-7:
            break
    return (v[:, None] * a) * v[None, :]


def skew_evidence(flow):
    s = flow + flow.T
    a = flow - flow.T
    edges = np.triu(s > 0, k=1)
    k = len(flow)
    m = s[edges]
    v = a[edges]
    df = max(0, len(m) - k + 1)
    a2 = v * v
    pearson = np.sum(a2 / m)
    # Repeated transitions supply more evidence than singleton edges.
    more = m > 2
    stat2 = np.sum((a2[more] / m[more]) - 1)
    null2 = np.sqrt(max(1, 2 * more.sum()))
    stat = pearson + 0.9 * max(0.0, stat2 - 1.5 * null2)
    return max(0.0, 1.0 - (0.80 * df + 1.5) / max(stat, 1e-12))


def regularize_jumps(jump, strength):
    k = len(jump)
    if k <= 2 or strength == 0:
        return jump
    size = jump.sum(axis=1)
    w = np.sqrt(np.maximum(size, 1e-12))
    js = (jump / w[:, None]) / w[None, :]
    eig = np.linalg.eigvalsh(symmetric(js))
    gap = max(0.005, 1 - eig[-2])

    target = np.outer(size, size) / size.sum()
    np.fill_diagonal(target, 0)
    if np.max(size) < 0.49 * size.sum():
        target = balance(target, size)
    else:
        return jump
    i, j = np.triu_indices(k, 1)
    mu = target[i, j] * 2
    m = jump[i, j] * 2
    # Variance of a gamma/Poisson edge model, after removing counting
    # noise. Mild contraction is most useful when many edges are sparse.
    weight = 1 / (0.35 + mu)
    excess = np.sum(weight * ((m - mu)**2 - mu))
    excess += max(1, k-1) * 0.75
    excess = max(excess, 0.11 * np.sum(weight * mu**2))
    dispersion = excess / np.sum(weight * mu**2)
    raw_weight = strength / (1.0 + dispersion * np.sum(mu**2) / np.sum(mu))
    cut = min(raw_weight, (0.39 if gap < .16 else 0.82) * gap)
    return (1.0 - cut) * jump + cut * target


def stationary(p):
    k = len(p)
    a = np.eye(k)-p.T
    a[-1] = 1
    b = np.zeros(k)
    b[-1] = 1
    pi = np.maximum(np.linalg.solve(a,b),0.)
    return pi/pi.sum()


def stationary_pool(p, pi, Z, n, factor):
    k = len(p)
    S = symmetric(pi[:,None]*Z)*2 - np.diag(pi) - np.outer(pi,pi)
    vals, vecs = np.linalg.eigh(S)
    small = vals <= max(1e-11, vals[-1]*1e-9)
    f = pi - 1/k
    score = n*np.sum((vecs[:,~small].T @ f)**2/vals[~small])
    if np.linalg.norm(vecs[:,small].T @ f) > 0.025 or score > 2.25*(k-1)+5:
        return p,pi,Z,0.0
    w = min(.88, factor*max(1.,k-2)/max(1.,score))
    target = (1-w)*pi + w/k
    flow = pi[:,None]*p
    if np.linalg.norm(flow-flow.T) < 1e-7:
        flow = balance(symmetric(flow),target,iterations=25)
    else:
        for _ in range(25):
            flow = flow * (target/flow.sum(axis=1))[:,None]
            flow *= (target/flow.sum(axis=0))[None,:]
    P = flow / flow.sum(axis=1,keepdims=True)
    stationary_prob = stationary(P)
    Z = np.linalg.solve(np.eye(k)-P+stationary_prob[None,:],np.eye(k))
    return P,stationary_prob,Z,w


def finite_covariance(st, counts, regularization=0.80, risk=0.70,
                      diagnostics=None, pool=0.0):
    k, d = st.shape
    flow = counts.copy()
    dwell = np.diag(flow).copy()
    np.fill_diagonal(flow, 0)
    js = symmetric(flow)
    row = js.sum(axis=1)
    nt = row + dwell
    exposure = np.min(nt) / max(1,nt.mean())
    adapted = regularization*min(1.,max(.42,3.2*exposure)**0.28)
    j0 = regularize_jumps(js, adapted)
    sk = skew_evidence(flow)
    # In sparse data the nonreversible component may have a large effect
    # on the smallest eigenvalues of the generator. Preserve its support.
    jump = j0 + 0.5 * sk * (flow - flow.T)
    jump = np.maximum(0.0, jump)
    support = js > 0
    deg = support.sum(axis=1)
    jump += support * (0.15 / np.maximum(1, np.maximum.outer(deg, deg)))
    q0 = row.sum() / nt.sum()
    q = (row + 0.35 * q0) / (nt + 0.35)
    if k >= 3 and 0.0 < q0 < 1.0 and regularization > 0:
        # Empirical Bayes concentration for the geometric holding times.
        exposure = max(1.0, np.sum(row**2 / nt) / row.sum() * np.sum(nt) / np.sum(row))
        var = np.sum((q - q0)**2 * nt) / (q0*(1-q0))
        ratio = max(0.0, var / max(k - 1, 1) - 1.0)
        prior = min(nt.mean() * 4.0, regularization * nt.mean() * 0.65 / max(ratio, 0.04))
        prior *= min(1.0, 2 / exposure)
        q = (row + (0.35 + prior)*q0) / (nt + 0.35 + prior)
    p = q[:, None] * (jump / jump.sum(axis=1, keepdims=True))
    p.flat[:: k + 1] = 1 - q

    ident = np.eye(k)
    pi = stationary(p)
    z = np.linalg.solve(ident - p + pi[None, :], ident)
    if pool and k >= 4 and pi.min() > .17/k:
        p,pi,z,weight = stationary_pool(p,pi,z,nt.sum(),pool)
        if diagnostics is not None:
            diagnostics["stationary_pool"] = weight
    f = st - (pi @ st)
    g = z @ f
    prediction = p @ g
    h = np.zeros((d, d))
    for i in range(k):
        v = (g - prediction[i]) * np.sqrt(pi[i] * p[i, :, None])
        h += v.T @ v
    if risk > 0.0:
        h = risk_shrink(h, f, g, z, p, pi, nt, risk)
    if diagnostics is not None:
        diagnostics.update(model="finite", states=k, skew=sk)
    return symmetric(h)


def risk_shrink(h, f, g, z, p, pi, nt, factor):
    # Delta variance in units of H. A row-wise multinomial perturbation
    # includes the change both in the stationary measure and in the resolvent.
    vals, vecs = np.linalg.eigh(symmetric(h))
    if vals[0] <= 1e-12 * vals[-1]:
        return h
    w = vecs / np.sqrt(vals)
    f = f @ w
    g = g @ w
    k, d = f.shape
    v = np.einsum("ki,kj->kij", f, g)
    v = v + v.transpose(0, 2, 1) - np.einsum("ki,kj->kij", f, f)
    v = (z @ v.reshape(k, d*d)).reshape(k, d, d)
    left = z.T @ (pi[:, None] * f)
    noise = np.zeros((d,d))
    for i in range(k):
        t = np.flatnonzero(p[i] > 0)
        m = np.einsum("i,kj->kij", left[i], g[t])
        m = pi[i] * v[t] + m + m.transpose(0, 2, 1)
        w_i = p[i, t]
        m -= np.sum(w_i[:, None, None] * m, axis=0)
        m *= np.sqrt(w_i / max(1, nt[i]))[:, None, None]
        m = m.reshape(-1, d)
        noise += m.T @ m
    vals2, vecs2 = np.linalg.eigh(symmetric(noise))
    out = vecs * np.sqrt(vals)
    return out @ ((vecs2 / (1.0 + factor*np.clip(vals2, 0, 1.6))) @ vecs2.T) @ out.T
