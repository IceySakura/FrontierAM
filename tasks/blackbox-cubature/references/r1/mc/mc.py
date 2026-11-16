"""Plain Monte Carlo through the query protocol: the mean of `budget` independent uniform points (seeded by the case)."""
import json
import sys

import numpy as np


def main():
    init = json.loads(sys.stdin.readline())
    d, budget, seed = int(init["dimension"]), int(init["budget"]), int(init["seed"])
    points = np.random.default_rng(seed).random((budget, d))
    values = []
    for start in range(0, budget, 512):
        print(json.dumps({"type": "query", "points": points[start:start + 512].tolist()}), flush=True)
        values += json.loads(sys.stdin.readline())["values"]
    print(json.dumps({"type": "result", "estimate": float(np.mean(values))}), flush=True)


if __name__ == "__main__":
    main()
