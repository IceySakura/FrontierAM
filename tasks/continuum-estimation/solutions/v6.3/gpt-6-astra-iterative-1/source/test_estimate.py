"""Synthetic regression checks; run with the bundled interpreter."""
import math
import unittest
from unittest.mock import patch

import numpy as np

from estimate import estimate
from quadrature import PowerModel


class EstimatorTests(unittest.TestCase):
    def test_constant(self):
        self.assertEqual(estimate(np.array([.2, .4, .1, .3]), np.full(4, -2.25)), -2.25)

    def test_single_powers(self):
        x = np.array([.37, .09, .28, .51, .18])
        for p in [.08, .6, 1., 2.35, 5.5]:
            y = -1.25 + .8*x**p
            self.assertAlmostEqual(estimate(x, y), -1.25, places=9)

    def test_three_powers_and_permutation(self):
        x = np.array([.05, .12, .24, .39, .52])
        y = 1.3 + .75*x**1.8 - .4*x**2.7 - .12*x**4.2
        first = estimate(x, y)
        self.assertLess(abs(first-1.3), .002)
        order = np.array([2, 0, 4, 1, 3])
        self.assertEqual(first, estimate(x[order], y[order]))

    def test_extreme_finite_values(self):
        examples = [
            (np.array([1e-200, 1e-100, .1, 1.]), np.array([1., 2., 3., 4.])),
            (np.array([.1, .2, .3, .4]), np.array([1e300, -1e300, 1e300, -1e300])),
            (np.array([.1, .2, .3, .4]), np.array([1e-300, 2e-300, 3e-300, 4e-300])),
        ]
        for x, y in examples:
            self.assertTrue(math.isfinite(estimate(x, y)))

    def test_clustered_levels(self):
        for width in [1e-7, 1e-5]:
            x = np.linspace(.3, .3+width, 5)
            y = 1.2 - 1.1*x**.65 - .7*x**1.3 - .1*x**2.8
            self.assertLess(abs(estimate(x, y)-1.2), .15)

    def test_budget_fallback(self):
        x = np.array([.05, .13, .26, .4, .52])
        y = 1.1 - .8*x**1.3 - .4*x**2.1 - .15*x**3.6
        with patch.object(PowerModel, "check_time", side_effect=ValueError("budget")):
            self.assertTrue(math.isfinite(estimate(x, y)))


if __name__ == "__main__":
    unittest.main()
