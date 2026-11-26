"""Fast, single-threaded polarization demosaicking."""
import ctypes
import importlib.util
from pathlib import Path

import numpy as np

PHASES = ((1, 1), (0, 1), (0, 0), (1, 0))
MISSING = (0, 1, 2, 4, 6, 7, 9, 10, 11, 12, 13, 15)
_WEIGHTS = None
_NATIVE = None
_BACKEND = None
_KERNELS = {}


def limit_blas_threads():
    import numpy.core._multiarray_umath as module
    library = ctypes.CDLL(module.__file__)
    for name in ("openblas_set_num_threads64_", "openblas_set_num_threads"):
        try:
            setter = getattr(library, name)
        except AttributeError:
            continue
        setter.argtypes = [ctypes.c_int]
        setter(1)
        return


def box5(a):
    tail = ((0, 0),)*(a.ndim-2)
    p = np.pad(a, ((0, 0), (2, 2))+tail, mode="edge")
    b = (p[:, :-4]+p[:, 1:-3]+p[:, 2:-2]+p[:, 3:-1]+p[:, 4:])*0.2
    p = np.pad(b, ((2, 2), (0, 0))+tail, mode="edge")
    return (p[:-4]+p[1:-3]+p[2:-2]+p[3:-1]+p[4:])*0.2


def baseline(raw):
    h, w = raw.shape
    out = np.empty((4, h, w), np.float32)
    for c, (y, x) in enumerate(PHASES):
        a = raw[y::2, x::2].astype(np.float32)
        p = np.pad(a, 1, mode="edge")
        oy = 1 if y == 0 else -1
        ox = 1 if x == 0 else -1
        ay = p[1+oy:1+oy+a.shape[0], 1:-1]
        ax = p[1:-1, 1+ox:1+ox+a.shape[1]]
        axy = p[1+oy:1+oy+a.shape[0], 1+ox:1+ox+a.shape[1]]
        out[c, y::2, x::2] = a
        out[c, 1-y::2, x::2] = (a+ay)*0.5
        out[c, y::2, 1-x::2] = (a+ax)*0.5
        out[c, 1-y::2, 1-x::2] = (a+ay+ax+axy)*0.25
    return out


def cubic(raw):
    out = np.empty((4, *raw.shape), np.float32)
    weights = (-0.0625, 0.5625, 0.5625, -0.0625)
    for c, (y, x) in enumerate(PHASES):
        a = raw[y::2, x::2].astype(np.float32)
        h, w = a.shape
        p = np.pad(a, 2, mode="edge")
        sy, sx = (0 if y else 1), (0 if x else 1)
        horizontal = sum(weights[k]*p[2:-2, sx+k:sx+k+w] for k in range(4))
        vertical = sum(weights[k]*p[sy+k:sy+k+h, 2:-2] for k in range(4))
        hp = np.pad(horizontal, ((2, 2), (0, 0)), mode="edge")
        diagonal = sum(weights[k]*hp[sy+k:sy+k+h] for k in range(4))
        out[c, y::2, x::2] = a
        out[c, y::2, 1-x::2] = horizontal
        out[c, 1-y::2, x::2] = vertical
        out[c, 1-y::2, 1-x::2] = diagonal
    np.clip(out, 0, 255, out=out)
    return out


def conv(a, weights, bias, kernel, relu=True, tile=32):
    """NumPy reference implementation for numerical kernel checks."""
    h, w, channels = a.shape
    outputs = weights.shape[1]
    if kernel == 1:
        result = (a.reshape(-1, channels) @ weights).reshape(h, w, outputs)
        result += bias
        if relu:
            np.maximum(result, 0, out=result)
        return result
    r = kernel//2
    padded = np.pad(a, ((r, r), (r, r), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel, kernel), axis=(0, 1))
    out = np.empty((h, w, outputs), np.float32)
    for start in range(0, h, tile):
        stop = min(start+tile, h)
        matrix = np.ascontiguousarray(windows[start:stop].transpose(0, 1, 3, 4, 2))
        result = matrix.reshape(-1, kernel*kernel*channels) @ weights
        result += bias
        if relu:
            np.maximum(result, 0, out=result)
        out[start:stop] = result.reshape(stop-start, w, outputs)
    return out


def native_layers(weights, width=512):
    global _BACKEND
    root = Path(__file__).resolve().parent
    if _BACKEND is None:
        spec = importlib.util.spec_from_file_location("polar_native", root/"native.py")
        _BACKEND = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_BACKEND)
    if not _BACKEND.available():
        return None
    normalized = int(weights.get("normalize_mode", 0))
    full = int(weights.get("layout_mode", 0))
    cin = (2 if normalized else 1) if full else (8 if normalized else 4)
    count = sum(key.startswith("w") and key[1:].isdigit() for key in weights)
    result = []
    try:
        for i in range(count):
            matrix, bias = weights[f"w{i}"], weights[f"b{i}"]
            kernel = int(round((matrix.shape[0]/cin)**0.5))
            cout = matrix.shape[1]
            if cout < 8 or 8 < cout < 16:
                padded = 8 if cout < 8 else 16
                matrix = np.pad(matrix, ((0, 0), (0, padded-cout)))
                bias = np.pad(bias, (0, padded-cout))
                cout = padded
            key = (cin, cout, kernel, width, i < count-1)
            if key not in _KERNELS:
                path = root/"kernels"/("kernel_"+"_".join(map(str, key))+".bin")
                code = path.read_bytes() if path.is_file() else None
                _KERNELS[key] = _BACKEND.Convolution(cin, cout, kernel, width,
                                                  relu=i < count-1, code=code)
            result.append((_KERNELS[key], matrix, bias))
            cin = cout
    except OSError:
        return None
    return result


def infer(raw, weights, native=None, clip=True):
    if native is None:
        return cubic(raw)
    h, w = raw.shape
    packed = raw.reshape(h//2, 2, w//2, 2).transpose(0, 2, 1, 3).reshape(h//2, w//2, 4).astype(np.float32)
    mean = (box5(packed) if int(weights.get("center_mode", 0))
            else box5(packed.mean(axis=2))[:, :, None])
    centered = packed-mean
    normalized = int(weights.get("normalize_mode", 0))
    full = int(weights.get("layout_mode", 0))
    gain = None
    if normalized:
        sigma = np.sqrt(box5(centered*centered)+4)
        a = np.concatenate((centered/sigma, 2/sigma), axis=2)
        sigma_raw = sigma.reshape(h//2, w//2, 2, 2).transpose(0, 2, 1, 3).reshape(h, w)
        gain = baseline(sigma_raw)
    else:
        a = centered*(1/32)
    if full:
        channels = 2 if normalized else 1
        a = a.reshape(h//2, w//2, channels, 2, 2).transpose(0, 3, 1, 4, 2).reshape(h, w, channels)
    for function, matrix, bias in native:
        a = function(a, matrix, bias)
    out = baseline(raw)
    if full:
        for c, (y, x) in enumerate(PHASES):
            for j, (yy, xx) in enumerate(((y, 1-x), (1-y, x), (1-y, 1-x))):
                out[c, yy::2, xx::2] += a[yy::2, xx::2, j]*(gain[c, yy::2, xx::2] if normalized else 32)
    else:
        for j, index in enumerate(MISSING):
            c, phase = divmod(index, 4)
            y, x = divmod(phase, 2)
            out[c, y::2, x::2] += a[:, :, j]*(gain[c, y::2, x::2] if normalized else 32)
    if clip:
        np.clip(out, 0, 255, out=out)
    return out


def reconstruct(raw):
    global _WEIGHTS, _NATIVE
    minimum = raw.min()
    if minimum == raw.max():
        return np.full((4, *raw.shape), minimum, np.float32)
    if _WEIGHTS is None:
        limit_blas_threads()
        with np.load(Path(__file__).with_name("weights.npz"), allow_pickle=False) as archive:
            _WEIGHTS = {key: archive[key] for key in archive.files}
        width = raw.shape[1] if int(_WEIGHTS.get("layout_mode", 0)) else raw.shape[1]//2
        _NATIVE = native_layers(_WEIGHTS, width)
    out = infer(raw, _WEIGHTS, native=_NATIVE)
    for c, (y, x) in enumerate(PHASES):
        out[c, y::2, x::2] = raw[y::2, x::2]
    return out
