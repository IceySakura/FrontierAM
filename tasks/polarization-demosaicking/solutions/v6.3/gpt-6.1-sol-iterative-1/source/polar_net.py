"""Compact single-threaded NumPy inference for the residual demosaicker."""

import numpy as np


def mean_filter(image, size):
    h, w = image.shape
    padded = np.pad(image, size // 2, mode="reflect")
    horizontal = padded[:, :w].copy()
    for i in range(1, size):
        horizontal += padded[:, i:i + w]
    out = horizontal[:h].copy()
    for i in range(1, size):
        out += horizontal[i:i + h]
    out *= 1 / (size * size)
    return out


class ResidualModel:
    def __init__(self, path):
        self.layers = []
        with np.load(path) as archive:
            channels = archive["w0"].shape[1]
            flags = int(archive["features"]) if "features" in archive else {
                4: 0, 5: 2, 8: 1, 9: 3, 12: 5, 13: 7
            }[channels]
            self.guided = bool(flags & 1)
            self.levels = bool(flags & 2)
            self.ratios = bool(flags & 4)
            self.multiscale = bool(flags & 8)
            i = 0
            while f"w{i}" in archive:
                w, b = archive[f"w{i}"], archive[f"b{i}"]
                k = w.shape[2]
                matrix = np.ascontiguousarray(
                    w.transpose(2, 3, 1, 0).reshape(-1, w.shape[0]))
                self.layers.append((matrix, b.copy(), k))
                i += 1
        self.radius = sum(k // 2 for _, _, k in self.layers)

    @staticmethod
    def _conv(x, layer, relu):
        weight, bias, k = layer
        h, w = x.shape[0] - k + 1, x.shape[1] - k + 1
        out = np.empty((h, w, bias.size), dtype=np.float32)
        if k == 1:
            np.matmul(x.reshape(h * w, -1), weight, out=out.reshape(h * w, -1))
            out += bias
            if relu:
                np.maximum(out, 0, out=out)
            return out
        rows_per_block = 8
        while (rows_per_block > 1
               and rows_per_block * w * weight.shape[0] * 4 > 4 * 1024 * 1024):
            rows_per_block //= 2
        for y in range(0, h, rows_per_block):
            rows = min(rows_per_block, h - y)
            window = np.lib.stride_tricks.sliding_window_view(
                x[y:y + rows + k - 1], (k, k), axis=(0, 1))
            columns = np.ascontiguousarray(
                window.transpose(0, 1, 3, 4, 2)).reshape(rows * w, -1)
            block = out[y:y + rows].reshape(rows * w, -1)
            np.matmul(columns, weight, out=block)
            block += bias
            if relu:
                np.maximum(block, 0, out=block)
        return out

    def predict(self, raw, guide):
        edge = 1 + 4 * int(self.multiscale)
        pad = 2 * (self.radius + edge)
        r = np.pad(raw.astype(np.float32), pad, mode="reflect")
        x = np.stack([r[y::2, z::2] for y in range(2) for z in range(2)], -1)
        if self.guided:
            g = np.pad(guide, pad, mode="reflect")
            g = np.stack([g[y::2, z::2] for y in range(2) for z in range(2)], -1)
            mean = mean_filter(g.mean(axis=-1), 3)
            x = np.concatenate([x - g, g - mean[..., None]], axis=-1)
        else:
            mean = mean_filter(x.mean(axis=-1), 3)
            x -= mean[..., None]
        if self.levels:
            x = np.concatenate([x, mean[..., None] * 0.25], axis=-1)
        if self.ratios or self.multiscale:
            ratio = x[..., :4] * 32 / np.maximum(mean[..., None], 16)
        if self.ratios:
            x = np.concatenate([x, ratio], axis=-1)
        if self.multiscale:
            pooled = np.stack([mean_filter(ratio[..., c], 9) for c in range(4)], -1)
            x = np.concatenate([x, pooled], axis=-1)
        x *= 1 / 32
        x = np.ascontiguousarray(x[edge:-edge, edge:-edge])
        for i, layer in enumerate(self.layers):
            x = self._conv(x, layer, i + 1 < len(self.layers))
        x *= 32
        return x
