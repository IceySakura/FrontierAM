"""Local, sample-preserving polarization demosaicking."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import ctypes
import glob
import importlib.util
import numpy as np
import scipy


# The caller may have imported NumPy before this module.
for _path in glob.glob(os.path.join(os.path.dirname(np.__file__), "..", "numpy.libs", "*openblas*")):
    _blas = ctypes.CDLL(_path)
    for _name in ("openblas_set_num_threads64_", "openblas_set_num_threads"):
        try:
            _set_threads = getattr(_blas, _name)
        except AttributeError:
            continue
        _set_threads.argtypes = [ctypes.c_int]
        _set_threads.restype = None
        _set_threads(1)
        break

with np.load(os.path.join(os.path.dirname(__file__), "weights.npz"), allow_pickle=False) as _archive:
    _weights = {key: _archive[key] for key in _archive.files}
_models = [_weights]
_auxiliary_path = os.path.join(os.path.dirname(__file__), "auxiliary.npz")
if os.path.isfile(_auxiliary_path):
    with np.load(_auxiliary_path, allow_pickle=False) as _archive:
        _models.append({key: _archive[key] for key in _archive.files})

_radius = int(_weights["radius"])
_normalized = bool(_weights.get("normalize", False))
_phase_channels = (2, 1, 3, 0)
_outputs = tuple((p, c) for p in range(4) for c in range(4) if p != c)
_output_channels = np.array([c for p, c in _outputs])
_tile_rows = 8
_use_fusion = (len(_models) == 2 and
               str(_models[0].get("kind", "mlp")) == "mlp" and
               str(_models[1].get("kind", "mlp")) == "kpn")
if _use_fusion:
    _mlp_width = _models[0]["w0"].shape[1]
    _gate_width = _models[1]["gw0"].shape[1]
    _first_w = np.concatenate(
        (_models[0]["w0"], _models[1]["gw0"], _models[1]["bw"]), axis=1
    )
    _first_b = np.concatenate(
        (_models[0]["b0"], _models[1]["gb0"], _models[1]["bb"])
    )

_nd_path = glob.glob(os.path.join(os.path.dirname(scipy.__file__), "ndimage", "_nd_image*.so"))[0]
_nd_spec = importlib.util.spec_from_file_location("_nd_image", _nd_path)
_nd = importlib.util.module_from_spec(_nd_spec)
_nd_spec.loader.exec_module(_nd)


def _box(a, size):
    temp = np.empty_like(a)
    out = np.empty_like(a)
    _nd.uniform_filter1d(a, size, 2, temp, 0, 0.0, 0)
    _nd.uniform_filter1d(temp, size, 1, out, 0, 0.0, 0)
    return out


def _predict(x, model):
    if str(model.get("kind", "mlp")) == "kpn":
        gates = x
        for i in range(3):
            gates = gates @ model[f"gw{i}"]
            gates += model[f"gb{i}"]
            if i < 2:
                np.maximum(gates, 0, out=gates)
        gates -= gates.max(axis=1, keepdims=True)
        np.exp(gates, out=gates)
        gates /= gates.sum(axis=1, keepdims=True)
        bases = x @ model["bw"]
        bases += model["bb"]
        bases = bases.reshape(-1, int(model["experts"]), 12)
        bases *= gates[:, :, None]
        return bases.sum(axis=1)
    depth = int(model.get("depth", 3))
    y = x
    for i in range(depth):
        y = y @ model[f"w{i}"]
        y += model[f"b{i}"]
        if i < depth - 1:
            np.maximum(y, 0, out=y)
    return y


def _predict_fused(x):
    projected = x @ _first_w
    projected += _first_b
    mlp, kpn = _models
    y = projected[:, :_mlp_width]
    np.maximum(y, 0, out=y)
    depth = int(mlp.get("depth", 3))
    for i in range(1, depth):
        y = y @ mlp[f"w{i}"]
        y += mlp[f"b{i}"]
        if i < depth - 1:
            np.maximum(y, 0, out=y)
    gates = projected[:, _mlp_width:_mlp_width + _gate_width]
    np.maximum(gates, 0, out=gates)
    for i in range(1, 3):
        gates = gates @ kpn[f"gw{i}"]
        gates += kpn[f"gb{i}"]
        if i < 2:
            np.maximum(gates, 0, out=gates)
    gates -= gates.max(axis=1, keepdims=True)
    np.exp(gates, out=gates)
    gates /= gates.sum(axis=1, keepdims=True)
    bases = projected[:, _mlp_width + _gate_width:].reshape(-1, int(kpn["experts"]), 12)
    bases *= gates[:, :, None]
    y += bases.sum(axis=1)
    y *= 0.5
    return y


def reconstruct(raw):
    """Return float32 images in the order 0, 45, 90, 135 degrees."""
    a = np.asarray(raw, dtype=np.float32)
    h, w = a.shape
    planes = np.stack([a[r::2, c::2] for r in range(2) for c in range(2)])
    bh, bw = planes.shape[1:]
    radius = _radius
    size = 2 * radius + 1
    padded = np.pad(planes, ((0, 0), (radius, radius), (radius, radius)), mode="edge")
    patches = np.lib.stride_tricks.sliding_window_view(padded, (size, size), axis=(1, 2))
    if _normalized:
        mean = _box(planes, size)
        variance = _box(planes * planes, size) - mean * mean
        np.maximum(variance, 0, out=variance)
        scale = np.sqrt(variance + 4).transpose(1, 2, 0)
        mean = mean.transpose(1, 2, 0)
    else:
        mean = planes.transpose(1, 2, 0)
        scale = np.full((bh, bw, 4), 16, dtype=np.float32)

    corrections = np.empty((bh, bw, 12), dtype=np.float32)
    for row in range(0, bh, _tile_rows):
        rows = min(_tile_rows, bh - row)
        x = patches[:, row:row + rows].transpose(1, 2, 0, 3, 4).reshape(rows * bw, 4, size * size)
        x -= mean[row:row + rows].reshape(rows * bw, 4, 1)
        sc = scale[row:row + rows].reshape(rows * bw, 4)
        x /= sc[:, :, None]
        x = x.reshape(rows * bw, 4 * size * size)
        if _use_fusion:
            prediction = _predict_fused(x)
        else:
            prediction = _predict(x, _models[0])
            for model in _models[1:]:
                prediction += _predict(x, model)
            prediction *= 1 / len(_models)
        prediction *= sc[:, _output_channels]
        corrections[row:row + rows] = prediction.reshape(rows, bw, 12)

    result = np.empty((4, h, w), dtype=np.float32)
    for c in range(4):
        r, col = divmod(c, 2)
        p = np.pad(planes[c], 1, mode="edge")
        center = p[1:-1, 1:-1]
        rs = slice(None, -2) if r else slice(2, None)
        cs = slice(None, -2) if col else slice(2, None)
        image = result[_phase_channels[c]]
        image[r::2, col::2] = center
        image[r::2, (1 - col)::2] = (center + p[1:-1, cs]) * 0.5
        image[(1 - r)::2, col::2] = (center + p[rs, 1:-1]) * 0.5
        image[(1 - r)::2, (1 - col)::2] = (
            center + p[1:-1, cs] + p[rs, 1:-1] + p[rs, cs]
        ) * 0.25
    for i, (p, c) in enumerate(_outputs):
        result[_phase_channels[c], p // 2::2, p % 2::2] += corrections[:, :, i]
    np.clip(result, 0, 255, out=result)
    return result
