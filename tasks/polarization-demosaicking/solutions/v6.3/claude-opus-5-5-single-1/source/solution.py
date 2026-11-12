"""Polarization mosaic demosaicking with orientation/strength-gated learned filters.

Pattern (2x2 tile, row-major): [90, 45] / [135, 0].  Output order: 0, 45, 90, 135.
Each output pixel is a gated sum of linear filters applied to the 9x9 raw
neighbourhood:  y_c = sum_g gate_g * (w_{g,c} . patch + b_{g,c}),
with gates [1, c1, c2, st, st*c1, st*c2] derived from a smoothed structure tensor.
Weights (per 2x2 phase) were fitted offline by least squares; see README.md.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_HERE = os.path.dirname(os.path.abspath(__file__))
_R = 4
_PHASE_OBS = {(0, 0): 2, (0, 1): 1, (1, 0): 3, (1, 1): 0}  # phase -> observed channel
_SIG = 0.6
_GR = int(4.0 * _SIG + 0.5)
_GK = np.exp(-0.5 * (np.arange(-_GR, _GR + 1) / _SIG) ** 2)
_GK = (_GK / _GK.sum()).astype(np.float32)
_EPS = 1.0
_SK = 2.0


def _smooth(a, axis):
    """Gaussian smoothing along axis (symmetric boundary), float32."""
    n = a.shape[axis]
    pad = [(0, 0)] * a.ndim
    pad[axis] = (_GR, _GR)
    p = np.pad(a, pad, mode="symmetric")
    sl = [slice(None)] * a.ndim
    sl[axis] = slice(_GR, _GR + n)
    out = p[tuple(sl)] * _GK[_GR]
    for k in range(1, _GR + 1):
        sl[axis] = slice(_GR - k, _GR - k + n)
        lo = p[tuple(sl)]
        sl[axis] = slice(_GR + k, _GR + k + n)
        out += (lo + p[tuple(sl)]) * _GK[_GR + k]
    return out


def gates(r):
    """r: float32 (H, W) raw.  Returns float32 (H, W, 6) gate values."""
    P = np.pad(r, 1, mode="reflect")
    # 2-pixel central differences compare pixels behind the same polarizer
    # angle, so the polarization modulation cancels without blurring.
    gx = (P[1:-1, 2:] - P[1:-1, :-2]) * np.float32(0.5)
    gy = (P[2:, 1:-1] - P[:-2, 1:-1]) * np.float32(0.5)
    J = np.stack([gx * gx, gy * gy, gx * gy])
    J = _smooth(_smooth(J, 1), 2)
    jxx, jyy, jxy = J
    tr = jxx + jyy
    inv = 1.0 / (tr + np.float32(_EPS))
    c1 = (jxx - jyy) * inv
    c2 = 2 * jxy * inv
    s = np.sqrt(tr)
    st = s / (s + np.float32(_SK))
    return np.stack([np.ones_like(c1), c1, c2, st, st * c1, st * c2], -1)


def phase_inputs(raw):
    """Yield (pr, pc, X (N, K*K) float32, gate (N, G) float32) per phase."""
    r = np.asarray(raw).astype(np.float32)
    K = 2 * _R + 1
    win = sliding_window_view(np.pad(r, _R, mode="reflect"), (K, K))
    g = gates(r)
    for (pr, pc) in _PHASE_OBS:
        X = win[pr::2, pc::2]
        h, w = X.shape[:2]
        yield pr, pc, X.reshape(h * w, K * K), g[pr::2, pc::2].reshape(h * w, -1), (h, w)


_W = None


def _load_weights():
    # Same bytes as gated_r4.npz (written by train.py), embedded so that
    # solution.py is self-contained.
    import base64
    import io
    return dict(np.load(io.BytesIO(base64.b64decode(_WEIGHTS_B64))))


def reconstruct(raw):
    global _W
    if _W is None:
        _W = _load_weights()
    raw = np.asarray(raw)
    H, W = raw.shape
    out = np.empty((4, H, W), dtype=np.float32)
    for pr, pc, X, g, (h, w) in phase_inputs(raw):
        c_obs = _PHASE_OBS[(pr, pc)]
        cs = [c for c in range(4) if c != c_obs]
        Wm = _W["w%d%d" % (pr, pc)]  # (K*K, G*3) float32
        b = _W["b%d%d" % (pr, pc)]  # (G*3,) float32
        Z = X @ Wm
        Z += b
        G = g.shape[1]
        Y = np.einsum("ngc,ng->cn", Z.reshape(-1, G, 3), g)
        for i, c in enumerate(cs):
            out[c, pr::2, pc::2] = Y[i].reshape(h, w)
        out[c_obs, pr::2, pc::2] = raw[pr::2, pc::2]
    np.clip(out, 0, 255, out=out)
    return out


_WEIGHTS_B64 = (
    "UEsDBC0AAAAAAAAAIQBwH7Yp//////////8HABQAdzAwLm5weQEAEABIFwAAAAAAAEgXAAAAAAAAk05VTVBZAQB2AHsnZGVz"
    "Y3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDgxLCAxOCksIH0gICAgICAgICAgICAgICAg"
    "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIApg2jE8fmn4O3ZNzDuVIoo7hY18uukRe7s2Hss7/NNN"
    "OoxsETs2hLi8cARJvItGirydbqs7aOOUO6sHIjxBJrW78S+Au756hjqCHwW8XI7hOk/jFrxj0GQ70F9oOnD1v7uWPqC7Z01v"
    "u0fhMTtPhlg8dtgHPAAgLTxQkH67ek5hO4wC9TsrHV08M7AgPIxolzunJXA7XUTlO2bd2jsKx1a6Kg0LPApsgzr3ba07hHDs"
    "u3345bv6u7C82qizvAJal7xvrdS7vO2fvGNVGbp5VbI6laguPN54RjyidCY7k9rDO0WMy7vpYfM79c3OO9ja9bufQpq6xiMt"
    "u/6O5DrwQVk7n1gnPImubDtfWbK70yLAO7JRqDtWIfQ7CiIEO7efkztGioq7U14KOyGRwDuLLsq8vxuYvOtEC7onwFM7o2A4"
    "OaF8Czuwvam8O+6ovBBQnbyZfCo7AQEcOy1xZ7zG/ii7P6WquyTwtLuh0c27YnkHPOzoJryYcJS7U5dpO/scdLy06hK7xbpw"
    "u5O/mLuSPac8mT8oPMrOHDyTBOM8aminPG3drDyB7a877Cu8PFEyRDs12Cq8Hnvcu5ZbfTueLI46FvR7OjJCmzlE7YG8REsl"
    "vD76/bqFmVc7XiZ1uw/EH7wOt8K8d0mmvCa52LoTZ1k8YFl1uhepdTnRAa87BBUGPG3sd7sUbYU7LO/mO9jPnzuNfJ+7xXUJ"
    "vDTJHrwluEK8rqCXufI5P7rN1ik7/jM/O42sorsxnDU8X2OtPHutRTybLC67GySYu8y61zrklxa7gIIpuy2CF7rISUO8OOru"
    "uZcyc7w2rYY7XyqrO6nwu7sR2e47ZitYuUhUJTsIyXI8VY8/OyrDhzzPChO8qQ0ivMBn0DpQ9ee71TWoOgwtfLv3PAA8gBQ1"
    "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"
    "MDAubnB5AQAQAMgAAAAAAAAAyAAAAAAAAACTTlVNUFkBAHYAeydkZXNjcic6ICc8ZjQnLCAnZm9ydHJhbl9vcmRlcic6IEZh"
    "bHNlLCAnc2hhcGUnOiAoMTgsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
    "ICAgICAgICAgCm+fUT02Lwa8AWQZPYe557trGiO8lzvbuxuXvrxDnhi8y7QBO6nW3j1ZvJ87S7N8Pb0mc7y6cpe8L22luzZP"
    "sLysizm81IciOlBLAwQtAAAAAAAAACEAScjO1///////////BwAUAHcwMS5ucHkBABAASBcAAAAAAABIFwAAAAAAAJNOVU1Q"
    "WQEAdgB7J2Rlc2NyJzogJzxmNCcsICdmb3J0cmFuX29yZGVyJzogRmFsc2UsICdzaGFwZSc6ICg4MSwgMTgpLCB9ICAgICAg"
    "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKwRMDud133zqobkA71IHvum+A77vt"
    "1qa7+arYO0nmNDpDcd068aFMu/2npLt3Sgi8TEo4PB6BxjufRv87SiCkuz78Srv5lWA6gnmgOqtTiroYBym6UdK3OmXM8btX"
    "soa7mNFKO4iIPbzSMTW6m5oevDhNRjx9Zye7yJQbuiUZLTzvHh88sBmDuwI7XTyEd+E7My3MOyc3LzvJC286JH+Hu608pLoV"
    "9pe7GGf1OpJAKDq/fCQ8jKtovJrXULwEcly8Z0KgO8n2i7vj6+27OA0KPDwyq7uQo0y8mJiXvGHvMjxKBLK6qOVjvOBP+Tv6"
    "zWs6m0/AOnKz4jqQ8jm6QXCQPNUDj7ksC0A7h+BGPOnZGjvZiLg68nYdPHVlwrvLjxE82QqfOz/6jrv4sZm7ZuzxOflAo7xp"
    "nLy8m/Vyu+LTyzpng+C6jtJJvOXQpbtIqEm84pmCu4oneDwXVoc866mVu5o5t7tgcU678R31u7s18Tv0my873csKOiglZTof"
    "A047FPNAvAn2qzvQU366aVbCudKerTvoBPg4BBWQvKa+rDwGWTM5Nx+XPDLNBzvX3Oc7JT+CO1fOK7oSIio6y/ZaOjsQk7u/"
    "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"
    "0VC8PNWQvBCvCbxEqSy9dhbkvUFBuL3AHRu+byi+PNLf67uLnq088WJIPdYspr1tWjq9laaUPkq3iT39dSc+t1FhPR9Dm7wF"
    "y/27D4NPPBmFvrsUGOW6OlI1vnPPfL37ygy+dK4JPJoqHj2YKlc9VEsXvfEgqzzSYPU8bHz4Pf+x9j0M2Vw+YuXhO4UZozx9"
    "9Yc84+/1OwGsLzp8jJY86AEEvvXL5r2ZWTa+2JjMvI2z17vYPsi7hpDKvMVLWz10bPo7veVoPZWyGL0Yt847JXZHvT5KqDyt"
    "iFY7CS4GPc23gTvrv0w82XvxvKvvKjyYCZ68WCYCPdBVpbxEJpy8ZnrHu9QccL2ntUi9b/VVvJPN+LybLHs6jta9vHD2A7tZ"
    "d/y8V/E3PMo7bjsWBIE8v/zRPKOVZD0r0kg9s6y4PD0iSLtbp4A8rm9/vBPnwzxWCZs8rwidu37Nd7wQRwa84FkSvGKRejsS"
    "kEG84T8uPD7o47sn3uE7Y+r0PDfcizyv6Nc8Zennu4lCBDqnKXc7eeY6vEhusDpJJea76bdru8uQvjx6rHc8LE+kPERk6brL"
    "68I8Hp0ZPFoEHDy/EsY6Zp9RvD5WMr2hxkO93siMvI0Vyzt2rmi8jV2gvKfokryTdWe8T0IXvftsizs99xi9zWnePKOMGr2N"
    "3j68CWL4OsLHuDunUSU6BoDePD/N8TyzYVU9Ug2tvCuQzDxoXD88802VO6H7jrv6POc7jdgmvDd+ij3kw6a741CPu7/uqrq3"
    "PNA8sYmPO30J77t8NQ88AZVbvEGkJL679N29ZmPYPJ7ncz1YUWY94QOpvFskwTrJMRe9ohmcPZZ5Xj7GaRM+288FvFm9tbyo"
    "xdK8D5mFO12GRbwfdq+7kjYRvma40bsJh5c8/fKVPXndB76qFKO9RYwRvExFqDyCDJQ8ZdCNvR575704Jnm+DdnIOy1tcr1n"
    "NjW9t7h0PH86gLsipS48qcGBP2eOWz+iiYg/6EYTvp39KT6/Cic9ofzrvKNHZzyFcpK8nVOUPbL9cj7nlSw+XnUpvMnk5Ls7"
    "8LE77t59u3B23Tp9IuU6H97gu7K8sr2/y9G9vYGPPfmAFb4Wu7S9rjkHO+f5UzvECFA812gMPZPJWT1aBQa9bi4lvBhO57qj"
    "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"
    "seG6RQGeu1OHSTw0xFM7MdpHvOD0f7xY+bG8Qi8rPBcGhrqiBGu7PUFdPNm0Y7xLbbg7HU/7uyL937rF7Ma78YGFOwquUbz2"
    "VQK6hQenO/vbQTkUy0Q7N35nO59wRzy8CgE8/2kzvFD9wjwfvDk7XZk5OwBaSbxVVKO74xZIO7omBzz8CQ88vqX3O5ljwjo2"
    "Bbq6MnbkOzUod7zR3B+7IsMAvHhblbzOYrO8OuYsvOoKr7s5jTG58Xmeu8mC9Tzh9EE8UEsDBC0AAAAAAAAAIQDncH4X////"
    "//////8HABQAYjAxLm5weQEAEADIAAAAAAAAAMgAAAAAAAAAk05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5f"
    "b3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDE4LCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
    "ICAgICAgICAgICAgICAgICAgIArodVs9b4ZcPP6DFD0f3ZY8S/GNvBtmhLycSqK7KUYyu2s3nDwszMs9DsxYO4SJVD0Py1U8"
    "VHkxvMIjSbxhm9K6OZ8JudndhjxQSwMELQAAAAAAAAAhABD9ezH//////////wcAFAB3MTAubnB5AQAQAEgXAAAAAAAASBcA"
    "AAAAAACTTlVNUFkBAHYAeydkZXNjcic6ICc8ZjQnLCAnZm9ydHJhbl9vcmRlcic6IEZhbHNlLCAnc2hhcGUnOiAoODEsIDE4"
    "KSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiTejjrv+NC7hUwG"
    "vNm0aDq6uL8666btO5WMSTwvN1c7D/O0OoEwBLs+/pM7N/K4O+wl5LoeClO7gjeju55qNLxZnJY7A01VO76GPTtNwRG6wlpy"
    "u+4JLjudzMC6a8UIvBuxFDumlsc6MnXEurl6/DtZiCY7+hqZOlRdZDtUjFg8MxdvPJQGh7wecwy8MELcO1BZCDubIqG7KNSA"
    "ucX0BrxLEWa7weY+Or91BTz4G7g7qPQ3OeG1ibza+hm8yCbbuymoLbumUQO8SToNvC0miTvLuQs8Rs2yO2FOzDvYgeu7orZO"
    "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"
    "vJQ2Ozz4Z7E7Pg9IvFMEdDmxQ4w5PF+Uu1qBFjxtSrs7wksZPEZfGTupI7m6UbVUPCU4PDx+8i48tJQpOwnJe7u3HXi7OLyP"
    "O2kRqLwRp6G8hVkWO10VSLxlBda78O1iu6XeQryyAGe8DmBIvMXA9zvQJhc76C8/vM33Yzzmw9O6PDAdO3HGx7vwyCe8h0JA"
    "vDkXKTu6nWi50+omu/JKFDyxW1Y8yYf5Oy3d2Tx5CZA8o0kXPNx0czyaDTE86rcuOx6Es7wtOa68LNzkuwEqeDvR1Qc78U/f"
    "O+ZYrLpHkMU78QptO2qp4js59zc7b8Tgu+SzabyTp4y8Bxd6vLxVgrzTLsu8yOwZvLBqv7qgnPY7EsePPEJaFTy9VqM7Gx2i"
    "ukmrT7xLq/e7v7bLujVjYbs3qYi7YLjFOyUo4rt81y68p5lburUe2zwKJqQ89VMbPJ8Jg7s+DDs7TbX1u0thcbt+pdK7gXAu"
    "u0/xTTpTAOU7C4QdPKhW3DuViCk8Nq71Ok0kqTtwmd47e9dJu4DExrtzwHa8ppSLvLPjrbpOq1C7+ZMCO1BLAwQtAAAAAAAA"
    "ACEAmNrmVf//////////BwAUAGIxMC5ucHkBABAAyAAAAAAAAADIAAAAAAAAAJNOVU1QWQEAdgB7J2Rlc2NyJzogJzxmNCcs"
    "ICdmb3J0cmFuX29yZGVyJzogRmFsc2UsICdzaGFwZSc6ICgxOCwpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
    "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAK2hZ/PBe1ZL1Dv8C8EoEAvTBxsDvRJTI8j0VovATXQLy0v0S8sx6BPZaR"
    "77wxNmq8j+oIvZxVR7sxPXs8/zdDvNh9bbwLkSm8UEsDBC0AAAAAAAAAIQCodC1r//////////8HABQAdzExLm5weQEAEABI"
    "FwAAAAAAAEgXAAAAAAAAk05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBl"
    "JzogKDgxLCAxOCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoU"
    "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"
)
