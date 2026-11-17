"""3-D fluorescence deconvolution: censoring-aware least squares with a sparse,
nonnegative Gaussian-blob synthesis prior, solved with FISTA on the exact
(zero-extended, cropped) forward operator, applied as separable 1-D matrices
(FFT fallback for non-separable PSFs). numpy only: importing scipy would cost
~0.12 s of the time budget."""
import os, sys, time

T0 = time.monotonic()
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    try:
        budget = float(os.environ.get('TIME_BUDGET_SECONDS', '0.5'))
    except ValueError:
        budget = 0.5
    if not (0.05 < budget < 60.0):
        budget = 0.5
    budget = min(budget, 0.5)
    try:
        deadline = float(os.environ.get('DECONV_DEADLINE_MONOTONIC', ''))
    except ValueError:
        deadline = T0 + budget
    if not (T0 - 5.0 < deadline < T0 + 600.0):  # missing / nonsensical value
        deadline = T0 + budget
    # never plan beyond the advertised budget, keep margin for write + exit
    stop_at = min(deadline, T0 + budget - 0.03) - 0.10

    with np.load(inp, allow_pickle=False) as d:
        y = np.asarray(d['observed'], dtype=np.float64)
        psf = np.asarray(d['psf'], dtype=np.float64)
    shape = y.shape
    out = np.zeros(shape, dtype=np.float32)
    try:
        x = solve(y, psf, stop_at)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.maximum(x, 0).astype(np.float32)
    except Exception as e:  # never fail the case
        print('solver error:', repr(e))
        try:
            out = np.maximum(np.nan_to_num(y), 0).astype(np.float32)
        except Exception:
            pass
    with open(outp, 'wb') as f:
        np.save(f, out, allow_pickle=False)
        f.flush()


def _good(n):
    """smallest 2^a 3^b 5^c >= n (fast pocketfft sizes)"""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def _same_matrix(k, n, circular=False):
    """n x n matrix of 1-D centred convolution with odd kernel k: zero extension
    ('same' crop) or periodic wrap."""
    r = len(k) // 2
    M = np.zeros((n, n))
    for i in range(n):
        for m in range(-r, r + 1):
            j = i - m
            if circular:
                M[i, j % n] += k[m + r]
            elif 0 <= j < n:
                M[i, j] += k[m + r]
    return M


def _sep_apply(v, Mz, My, Mx):
    v = v @ Mx.T
    v = np.matmul(My, v)
    return (Mz @ v.reshape(v.shape[0], -1)).reshape(Mz.shape[0], v.shape[1], v.shape[2])


class Op:
    """Synthesis model x = G c (G: small Gaussian blob, c >= 0 sparse, both on
    the input grid; G wraps periodically, which ties opposite faces together and
    helps structures that cross the volume) observed exactly through
    y = crop(conv(zero-extend(x), psf)).
    G is separable, i.e. three small matrices. If the psf is separable too
    (rank-1 within float32 rounding), the whole forward model is three small
    matmuls; otherwise the psf part uses rfftn on a non-aliasing padded grid."""

    def __init__(self, psf, shape, sg, gr):
        ax = np.arange(-gr, gr + 1, dtype=np.float64)
        g = np.exp(-0.5 * (ax / sg) ** 2)
        g /= g.sum()
        self.G = [_same_matrix(g, n, circular=True) for n in shape]
        self.GT = [G.T.copy() for G in self.G]
        marg = [psf.sum(axis=(1, 2)), psf.sum(axis=(0, 2)), psf.sum(axis=(0, 1))]
        sep = marg[0][:, None, None] * marg[1][None, :, None] * marg[2][None, None, :]
        self.sep = float(np.abs(sep - psf).max()) <= 1e-4 * float(psf.max())
        if self.sep:
            self.B = [_same_matrix(m, n) @ G for m, n, G in zip(marg, shape, self.G)]
            self.BT = [B.T.copy() for B in self.B]
            self.L = float(np.prod([np.linalg.norm(B, 2) ** 2 for B in self.B]))
        else:
            ks = psf.shape
            self.N = tuple(_good(s + k // 2) for s, k in zip(shape, ks))
            Hp = np.zeros(self.N)
            Hp[:ks[0], :ks[1], :ks[2]] = psf
            Hp = np.roll(Hp, tuple(-(k // 2) for k in ks), axis=(0, 1, 2))
            self.H = np.fft.rfftn(Hp)
            self.Hc = np.conj(self.H)
            self.sl = tuple(slice(0, s) for s in shape)
            self.L = float(np.max(np.abs(self.H)) ** 2) * float(
                np.prod([np.linalg.norm(G, 2) ** 2 for G in self.G]))

    def X(self, c):
        return _sep_apply(c, *self.G)

    def A(self, c):
        if self.sep:
            return _sep_apply(c, *self.B)
        x = self.X(c)
        return np.fft.irfftn(np.fft.rfftn(x, self.N) * self.H, self.N)[self.sl]

    def AT(self, r):
        if self.sep:
            return _sep_apply(r, *self.BT)
        x = np.fft.irfftn(np.fft.rfftn(r, self.N) * self.Hc, self.N)[self.sl]
        return _sep_apply(x, *self.GT)


def _interp(s, lo, hi):
    """linear in the read-noise level between the two observed regimes."""
    f = min(max((s - 0.008) / 0.017, -0.3), 1.6)
    return lo + f * (hi - lo)


def solve(y, psf, stop_at):
    psf = np.maximum(psf, 0)
    psf = psf / max(psf.sum(), 1e-12)
    op = Op(psf, y.shape, 0.8, 3)
    # background (read) noise level from dark voxels: clipped N(0,s^2) has E[y^2]=s^2/2
    b = op.A(op.A(y))
    dark = b <= np.percentile(b, 30)
    s = float(np.sqrt(2.0 * np.mean(y[dark] ** 2)))
    # censored (clipped) zeros -> conditional mean of the noise below zero
    yi = np.where(y > 0, y, -0.798 * s)
    # L1 weight and early stop tuned on the two noise regimes
    # (s~0.008: lam 0.003, 60 its; s~0.025: lam 0.0065, 45 its)
    lam = max(_interp(s, 0.003, 0.0065), 0.002)
    iters = int(round(min(max(_interp(s, 60.0, 45.0), 35.0), 70.0)))
    L = op.L * 1.01
    c = np.zeros(y.shape); z = c.copy(); tk = 1.0
    for k in range(iters):
        if time.monotonic() > stop_at:
            break
        g = op.AT(op.A(z) - yi)
        cn = np.maximum(z - (g + lam) / L, 0)
        tn = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        z = cn + ((tk - 1) / tn) * (cn - c); c = cn; tk = tn
    # undo the global L1 shrinkage: least-squares amplitude refit to the data
    Ac = op.A(c)
    den = float((Ac * Ac).sum())
    alpha = float((Ac * yi).sum()) / den if den > 0 else 1.0
    alpha = min(max(alpha, 0.9), 1.5)
    return alpha * op.X(c)


if __name__ == '__main__':
    main()
