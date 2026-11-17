"""Check blocked enumeration against an independent tiny brute-force search."""

import itertools
import math
import unittest

import numpy as np

from analyze_split_feasibility import analyze_losses


class FeasibilityChecks(unittest.TestCase):
    def test_matches_tiny_brute_force(self):
        losses = np.array([
            [[13., 2.], [1., 4.], [3., 1.]],
            [[2., 3.], [8., 12.], [1., 2.]],
            [[1., 7.], [5., 2.], [3., 9.]],
            [[2., 1.], [4., 4.], [6., 3.]],
        ])
        thresholds = (1.25, 1.5, 1.75, 2.)
        counts = {t: 0 for t in thresholds}
        optimum = math.inf
        for choices in itertools.product(range(3), repeat=len(losses)):
            ratios = []
            for method in range(2):
                public = math.fsum(losses[g, option, method] for g, option in enumerate(choices)) / len(losses)
                private = math.fsum(losses[g, other, method] for g, option in enumerate(choices)
                                    for other in range(3) if other != option) / (2 * len(losses))
                ratios.append(max(public / private, private / public))
            worst = max(ratios)
            optimum = min(optimum, worst)
            for threshold in thresholds:
                counts[threshold] += worst <= threshold
        for block_size in (1, 2, 64):
            result = analyze_losses(losses, thresholds=thresholds, block_size=block_size)
            self.assertEqual(result["enumerated_splits"], 81)
            self.assertAlmostEqual(result["minimum_worst_mean_ratio"], optimum, places=13)
            self.assertEqual({r["max_ratio"]: r["feasible_count"] for r in result["thresholds"]}, counts)
            self.assertNotIn("cases", result)

    def test_zero_pool_and_invalid_input(self):
        result = analyze_losses([[[0., 0.], [1., 1.], [1., 1.]]])
        self.assertEqual(result["total_splits"], 3)
        self.assertTrue(math.isfinite(result["minimum_worst_mean_ratio"]))
        with self.assertRaises(ValueError):
            analyze_losses(np.zeros((2, 3, 2)))
        with self.assertRaises(ValueError):
            analyze_losses([[[1., -1.], [1., 1.], [1., 1.]]])


if __name__ == "__main__":
    unittest.main()
