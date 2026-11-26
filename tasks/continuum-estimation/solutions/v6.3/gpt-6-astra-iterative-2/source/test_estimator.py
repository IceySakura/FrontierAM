"""Offline regression checks. Not invoked by run.sh."""

import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from continuum import estimate


ROOT = Path(__file__).resolve().parent


class EstimatorTests(unittest.TestCase):
    def test_constant(self):
        for value in [0.0, -3.5, 1e-100, 1e100]:
            self.assertEqual(estimate([.5, .03, .2, .1], [value]*4), value)

    def test_single_power(self):
        x = np.array([.52, .04, .3, .18, .43])
        for n in [4, 5]:
            for p in [.1, .2, .5, 1.0, 1.5, 2.3, 3.0, 5.0, 8.0]:
                for sign in [-1.0, 1.0]:
                    y = 1.3 + sign*1.2*x[:n]**p
                    self.assertAlmostEqual(estimate(x[:n], y), 1.3, delta=2e-10)

    def test_small_positive_order(self):
        x = np.array([.04, .18, .3, .52])
        for p in [.001, .005, .01]:
            y = -1.3 + .9*x**p
            self.assertAlmostEqual(estimate(x, y), -1.3, delta=1e-8)

    def test_permutation(self):
        x = np.array([.08, .17, .23, .38, .51])
        y = -.6 + .8*x**.9 - .4*x**2.1 + .2*x**3.6
        expected = estimate(x, y)
        order = np.array([3, 0, 4, 1, 2])
        self.assertEqual(estimate(x[order], y[order]), expected)

    def test_sign_reversal(self):
        x = np.array([.06, .18, .33, .49])
        y = 1.8 - .7*x**1.1 + .35*x**2.3 + .2*x**3.8
        self.assertAlmostEqual(estimate(x, -y), -estimate(x, y), delta=2e-9)

    def run_case(self, data):
        with tempfile.TemporaryDirectory() as directory:
            inp = Path(directory) / "input.json"
            out = Path(directory) / "nested" / "output.json"
            inp.write_text(json.dumps(data), encoding="utf-8")
            env = dict(os.environ)
            env["PATH"] = "/workspace/runtime/bin:" + env.get("PATH", "")
            result = subprocess.run(
                ["/bin/sh", str(ROOT / "run.sh"), str(inp), str(out)],
                env=env, capture_output=True, text=True, timeout=45, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            obj = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(set(obj), {"estimate"})
            self.assertIn(type(obj["estimate"]), (float, int))
            self.assertTrue(math.isfinite(obj["estimate"]))
            self.assertLess(out.stat().st_size, 65536)
            return obj["estimate"]

    def test_interface_and_scales(self):
        cases = [
            {"x": [.2, .5, .1, .3], "y": [1., 1., 1., 1.]},
            {"x": [1e-300, 2e-300, 3e-300, 4e-300], "y": [2., 3., 4., 5.]},
            {"x": [1e100, 2e100, 3e100, 4e100], "y": [2., 3., 4., 5.]},
            {"x": [1e-308, 1e-100, 1e100, 1e308], "y": [1., 1.2, 1.3, 1.4]},
            {"x": [.1, .2, .3, .4], "y": [1e308, -1e308, 1e308, -1e308]},
            {"x": [1., 1.+1e-13, 1.+2e-13, 1.+3e-13],
             "y": [3., 3.+2e-13, 3.+4e-13, 3.+6e-13]},
        ]
        for data in cases:
            self.run_case(data)

    def test_public_interface(self):
        for path in sorted(Path("/workspace/public/dev").glob("*.input.json")):
            self.run_case(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
