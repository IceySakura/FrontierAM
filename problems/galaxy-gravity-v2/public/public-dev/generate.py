"""Write a development snapshot as little-endian binary64 rows (m, x, y, z)."""
import argparse

import numpy as np

MODELS = ("smooth-hernquist", "disc", "group", "jaffe", "einasto")


def directions(rng, n):
    xyz = rng.normal(size=(n, 3))
    return xyz / np.sqrt(np.sum(xyz * xyz, axis=1))[:, None]


def centers(rng, n):
    return directions(rng, n) * (4.0 * np.cbrt(rng.random(n)))[:, None]


def tapered_radius(model, u):
    b = 100.0
    if model == "jaffe":
        tail_mass = 1.0 / (2.0 * b + 1.0)
        tail_scale = (1.0 + b) / 2.0
        q = ((b + 0.5) / (b + 1.0)) * u
    else:
        tail_mass = 2.0 / (2.0 * b + 1.0)
        tail_scale = b * (1.0 + b) / (2.0 * b - 1.0)
        q = np.sqrt(((b / (1.0 + b)) ** 2 / (1.0 - tail_mass)) * u)
    return np.where(u < 1.0 - tail_mass, q / (1.0 - q),
                    b - tail_scale * np.log((1.0 - u) / tail_mass))


def generate(model, n, seed):
    if n < 2 or (model == "group" and n % 4):
        raise ValueError("N must be at least 2; group requires a multiple of 4")
    rng = np.random.default_rng(np.random.SeedSequence([seed, MODELS.index(model), n]))
    if model == "disc":
        normal = directions(rng, 1)[0]
        xyz = rng.normal(size=(n, 3))
        height = np.sum(xyz * normal, axis=1)
        xyz -= 0.9 * height[:, None] * normal
    else:
        radius = (tapered_radius(model, rng.random(n))
                  if model in ("jaffe", "smooth-hernquist")
                  else (rng.gamma(10.5, size=n) / 10.168613781773965) ** 3.5)
        xyz = directions(rng, n) * radius[:, None]
    if model == "group":
        xyz += np.repeat(centers(rng, 4), n // 4, axis=0)
        rng.shuffle(xyz, axis=0)
    else:
        xyz += centers(rng, 1)
    rows = np.empty((n, 4), dtype="<f8")
    rows[:, 0] = 1.0 / n
    rows[:, 1:] = xyz
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("n", type=int)
    parser.add_argument("seed", type=int)
    parser.add_argument("output")
    args = parser.parse_args()
    generate(args.model, args.n, args.seed).tofile(args.output)
