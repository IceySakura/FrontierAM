"""Offline generation of reusable convolution kernels."""
import json
from pathlib import Path
import py_compile

import numpy as np

import native


def main():
    root = Path(__file__).resolve().parent
    config = root/"config.json"
    if config.is_file():
        models = json.loads(config.read_text())["models"]
    else:
        models = [{"file": "weights.npz"}]
    folder = root/"kernels"
    folder.mkdir(exist_ok=True)
    signatures = set()
    for model in models:
        with np.load(root/model["file"], allow_pickle=False) as archive:
            weights = {key: archive[key] for key in archive.files}
        full = int(weights.get("layout_mode", 0))
        normalized = int(weights.get("normalize_mode", 0))
        cin = (2 if normalized else 1) if full else (8 if normalized else 4)
        width = 1024 if full else 512
        count = sum(key.startswith("w") and key[1:].isdigit() for key in weights)
        for index in range(count):
            matrix = weights[f"w{index}"]
            kernel = int(round((matrix.shape[0]/cin)**0.5))
            cout = matrix.shape[1]
            if cout < 8 or 8 < cout < 16:
                cout = 8 if cout < 8 else 16
            signatures.add((cin, cout, kernel, width, index < count-1))
            cin = cout
    for signature in signatures:
        path = folder/("kernel_"+"_".join(map(str, signature))+".bin")
        path.write_bytes(native.generate(*signature))
    for name in ("solution.py", "native.py"):
        py_compile.compile(str(root/name), doraise=True)
    print(f"Built {len(signatures)} single-threaded convolution kernels.")


if __name__ == "__main__":
    main()
