#!/usr/bin/env python3
"""Offline numerical and command-line smoke tests."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from estimator import estimate_long_run_covariance


class EstimatorTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_shape_symmetry_and_psd(self):
        x = self.rng.normal(size=(3000, 6))
        x[1:] += 0.6 * x[:-1]
        result = estimate_long_run_covariance(x)
        self.assertEqual(result.shape, (6, 6))
        self.assertEqual(result.dtype.str, "<f8")
        self.assertTrue(result.flags.c_contiguous)
        np.testing.assert_array_equal(result, result.T)
        self.assertGreaterEqual(np.linalg.eigvalsh(result)[0], -1e-12)

    def test_translation_and_linear_transformation(self):
        x = self.rng.normal(size=(3000, 4))
        transform = self.rng.normal(size=(4, 4)) + 3.0 * np.eye(4)
        reference = estimate_long_run_covariance(x)
        transformed = estimate_long_run_covariance(x @ transform + 25.0)
        np.testing.assert_allclose(
            transformed, transform.T @ reference @ transform,
            rtol=1e-10, atol=1e-12,
        )

    def test_constant_and_rank_deficient_inputs(self):
        np.testing.assert_array_equal(
            estimate_long_run_covariance(np.ones((100, 3))), np.zeros((3, 3))
        )
        column = self.rng.normal(size=1000)
        x = np.column_stack((column, 2 * column, np.ones(1000)))
        result = estimate_long_run_covariance(x)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreaterEqual(np.linalg.eigvalsh(result)[0], -1e-12)

    def test_cli(self):
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.npy"
            target = Path(directory) / "output.npy"
            x = self.rng.normal(size=(1000, 4))
            np.save(source, x)
            subprocess.run(
                [str(root / "run.sh"), str(source), str(target)],
                check=True, cwd=directory, timeout=5,
            )
            np.testing.assert_array_equal(
                np.load(target, allow_pickle=False),
                estimate_long_run_covariance(x),
            )


if __name__ == "__main__":
    unittest.main()
