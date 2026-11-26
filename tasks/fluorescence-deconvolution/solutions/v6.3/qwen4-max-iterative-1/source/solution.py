import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, time, ctypes
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = None
WEIGHTS = None

def get_lib():
    global LIB
    if LIB is None:
        LIB = ctypes.CDLL(os.path.join(HERE, 'deconv.so'))
        F = ctypes.POINTER(ctypes.c_float)
        LIB.rl_multi.argtypes = [F, F, F, F, F, F, F, F, F,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 F, F, F, ctypes.c_int, ctypes.c_float, ctypes.c_float]
        LIB.mlp_apply.argtypes = [F, F, F, F, F, F, F, F, F, F, F,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_float]
    return LIB

def separable_factors(h):
    f = []
    for a in range(3):
        v = h.sum(axis=tuple(j for j in range(3) if j != a))
        s = v.sum()
        if s <= 0:
            return None
        f.append(v / s)
    rec = f[0][:, None, None] * f[1][None, :, None] * f[2][None, None, :]
    if np.max(np.abs(rec - h)) > 1e-5:
        return None
    return [v.astype(np.float32) for v in f]

def rl_c(y, f):
    lib = get_lib()
    F = ctypes.POINTER(ctypes.c_float)
    nz, ny, nx = y.shape
    N = y.size
    y32 = np.ascontiguousarray(y, dtype=np.float32)
    x10 = np.empty(N, np.float32); x20 = np.empty(N, np.float32); x30 = np.empty(N, np.float32)
    p = np.empty(N, np.float32); r = np.empty(N, np.float32); norm = np.empty(N, np.float32)
    t1 = np.empty(N, np.float32); t2 = np.empty(N, np.float32)
    ptr = lambda a: a.ctypes.data_as(F)
    lib.rl_multi(ptr(y32), ptr(x10), ptr(x20), ptr(x30), ptr(p), ptr(r), ptr(norm), ptr(t1), ptr(t2),
                 nz, ny, nx, ptr(f[0]), ptr(f[1]), ptr(f[2]), len(f[0]), 0.03, float(y.mean()))
    return x10.reshape(y.shape), x20.reshape(y.shape), x30.reshape(y.shape)

def rl_fft(y, h):
    n = 40
    hp = np.zeros((n, n, n), np.float32)
    idx = (np.arange(h.shape[0]) - h.shape[0] // 2) % n
    idy = (np.arange(h.shape[1]) - h.shape[1] // 2) % n
    idz = (np.arange(h.shape[2]) - h.shape[2] // 2) % n
    hp[np.ix_(idx, idy, idz)] = h
    H = np.fft.rfftn(hp)
    Hc = np.conj(H)
    def conv(z, adj=False):
        return np.fft.irfftn(np.fft.rfftn(z, s=(n, n, n)) * (Hc if adj else H), s=(n, n, n))[:32, :32, :32]
    x = np.full(y.shape, max(y.mean(), 1e-8), np.float32)
    norm = conv(np.ones(y.shape, np.float32), True)
    for _ in range(20):
        p = conv(x)
        x *= conv(y / (p + 1e-9), True) / (norm + 0.03)
    return x, x, x

def patches(a):
    p = np.pad(a, 1)
    return np.lib.stride_tricks.sliding_window_view(p, (3, 3, 3)).reshape(-1, 27)

def mlp_denoise(y, z10, z20, z30):
    F = np.concatenate([patches(z20), patches(y),
                        z10.ravel()[:, None], z30.ravel()[:, None], z20.ravel()[:, None]], axis=1)
    outs = []
    for name in ('mlp_weights.npz', 'mlp_torch.npz', 'mlp_torch128.npz'):
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        w = np.load(path, allow_pickle=False)
        G = (F - w['mean']) / w['scale']
        hid = G @ w['W1'] + w['b1']
        np.maximum(hid, 0, out=hid)
        outs.append(hid @ w['W2'] + w['b2'])
    if not outs:
        raise RuntimeError('no weights')
    out = sum(outs) / len(outs)
    return np.maximum(out.reshape(y.shape), 0)

def get_weights():
    global WEIGHTS
    if WEIGHTS is None:
        WEIGHTS = []
        for name in ('mlp_weights.npz', 'mlp_torch.npz', 'mlp_torch128.npz'):
            path = os.path.join(HERE, name)
            if os.path.exists(path):
                w = np.load(path, allow_pickle=False)
                d = {k: np.ascontiguousarray(w[k], dtype=np.float32) for k in w.files}
                d['W1T'] = np.ascontiguousarray(d['W1'].T, dtype=np.float32)
                WEIGHTS.append(d)
    return WEIGHTS

def mlp_denoise_c(y, z10, z20, z30):
    lib = get_lib()
    ws = get_weights()
    if not ws:
        raise RuntimeError('no weights')
    F = ctypes.POINTER(ctypes.c_float)
    ptr = lambda a: a.ctypes.data_as(F)
    y = np.ascontiguousarray(y, np.float32)
    z10 = np.ascontiguousarray(z10, np.float32)
    z20 = np.ascontiguousarray(z20, np.float32)
    z30 = np.ascontiguousarray(z30, np.float32)
    out = np.zeros(y.size, np.float32)
    nz, ny, nx = y.shape
    weight = 1.0 / len(ws)
    for w in ws:
        hidden = w['W1'].shape[1]
        lib.mlp_apply(ptr(y), ptr(z10), ptr(z20), ptr(z30),
                      ptr(w['mean']), ptr(w['scale']), ptr(w['W1T']), ptr(w['b1']),
                      ptr(w['W2']), ptr(w['b2']), ptr(out),
                      nz, ny, nx, hidden, weight)
    return np.maximum(out.reshape(y.shape), 0)

def mlp_tta(y, z10, z20, z30):
    a = mlp_denoise_c(y, z10, z20, z30)
    b = mlp_denoise_c(y[::-1], z10[::-1], z20[::-1], z30[::-1])[::-1]
    return 0.5 * (a + b)

def main():
    deadline = float(os.environ.get('DECONV_DEADLINE_MONOTONIC', '1e9'))
    budget = float(os.environ.get('TIME_BUDGET_SECONDS', '1.25'))
    deadline = min(deadline, time.monotonic() + budget, time.monotonic() + 1.25)
    inp, out = sys.argv[1], sys.argv[2]
    with np.load(inp, allow_pickle=False) as d:
        y = np.asarray(d['observed'], dtype=np.float32)
        h = np.asarray(d['psf'], dtype=np.float32)
    f = separable_factors(h)
    if f is not None:
        z10, z20, z30 = rl_c(y, f)
    else:
        z10, z20, z30 = rl_fft(y, h)
    x = z20
    if time.monotonic() < deadline - 0.10 and os.path.exists(os.path.join(HERE, 'mlp_weights.npz')):
        try:
            x = mlp_tta(y, z10, z20, z30)
        except Exception:
            try:
                x = mlp_denoise(y, z10, z20, z30)
            except Exception:
                x = z20
    m = float(x.max())
    if m > 0:
        mean = float(x.mean())
        if m / (mean + 1e-12) > 200.0:
            x /= m
        np.minimum(x, 1.0, out=x)
    m = float(x.max())
    if m > 0:
        x[x < 0.02 * m] = 0
    np.save(out, np.ascontiguousarray(x, dtype=np.float32))

if __name__ == '__main__':
    main()
