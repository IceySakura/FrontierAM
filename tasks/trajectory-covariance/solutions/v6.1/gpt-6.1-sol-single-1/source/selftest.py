"""Offline mathematical and numerical checks; not used by run.sh."""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import unittest

import numpy as np
from scipy.signal import lfilter

from estimator import estimate, regression_statistics
from moving_average import exact_objective, matrix_filter, unpack


class NumericalTests(unittest.TestCase):
    def test_regression_moments(self):
        rng = np.random.default_rng(17)
        y = rng.normal(size=(211, 3))
        y -= y.mean(axis=0)
        for order in (1, 2, 7):
            g, c, v = regression_statistics(y, order)
            x = np.column_stack([
                y[order-j-1:len(y)-j-1] for j in range(order)
            ])
            response = y[order:].copy()
            x -= x.mean(axis=0)
            response -= response.mean(axis=0)
            np.testing.assert_allclose(g, x.T @ x/len(x), atol=2e-14)
            np.testing.assert_allclose(c, response.T @ x/len(x), atol=2e-14)
            np.testing.assert_allclose(v, response.T @ response/len(x), atol=2e-14)

    def test_schur_filter(self):
        rng = np.random.default_rng(29)
        b = np.array([[0.5, 0.3, 0.2], [-0.3, 0.5, 0.1], [0.0, 0.0, -0.7]])
        data = rng.normal(size=(81, 3, 2))
        expected = data.copy()
        for t in range(1, len(data)):
            expected[t] -= b @ expected[t-1]
        np.testing.assert_allclose(matrix_filter(b, data), expected, atol=2e-13)

    def test_exact_likelihood(self):
        rng = np.random.default_rng(31)
        n, d = 17, 3
        y = rng.normal(size=(n, d))
        parameters = np.r_[rng.normal(size=d*d)*0.2, np.zeros(d*(d+1)//2)]
        actual, gradient = exact_objective(parameters, y)
        b, lower, omega = unpack(parameters, d)
        covariance = np.kron(np.eye(n), omega+b @ omega @ b.T)
        for j in range(n-1):
            covariance[(j+1)*d:(j+2)*d, j*d:(j+1)*d] = b @ omega
            covariance[j*d:(j+1)*d, (j+1)*d:(j+2)*d] = omega @ b.T
        design = np.tile(np.eye(d), (n, 1))
        inverse = np.linalg.inv(covariance)
        mean = np.linalg.solve(
            design.T @ inverse @ design, design.T @ inverse @ y.ravel()
        )
        residual = y.ravel()-design @ mean
        expected = (
            np.linalg.slogdet(covariance)[1]+residual @ inverse @ residual
        )/n
        self.assertAlmostEqual(actual, expected, places=11)
        numerical = []
        for j in range(len(parameters)):
            above, below = parameters.copy(), parameters.copy()
            above[j] += 1e-5
            below[j] -= 1e-5
            numerical.append(
                (exact_objective(above, y)[0]-exact_objective(below, y)[0])/2e-5
            )
        np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-7)

    def test_affine_coordinates(self):
        rng = np.random.default_rng(47)
        data = rng.normal(size=(3500, 4))
        for j, rho in enumerate((0.85, 0.6, -0.6, 0.1)):
            data[:, j] = lfilter([1.0], [1.0, -rho], data[:, j])
        before = data.copy()
        base = estimate(data)
        transform = rng.normal(size=(4, 4))+2*np.eye(4)
        offset = np.array([100.0, -200.0, 300.0, 0.0])
        changed = estimate(data @ transform.T+offset)
        np.testing.assert_allclose(changed, transform @ base @ transform.T,
                                   rtol=2e-7, atol=2e-8)
        np.testing.assert_array_equal(data, before)

    def test_degenerate_inputs(self):
        for data in (np.ones((200, 6)), np.ones((1, 4))):
            np.testing.assert_array_equal(
                estimate(data), np.zeros((data.shape[1], data.shape[1]))
            )
        rng = np.random.default_rng(53)
        z = rng.normal(size=(500, 2))
        data = np.column_stack((z, z[:, 0], np.zeros(500)))
        h = estimate(data)
        self.assertEqual(h.shape, (4, 4))
        self.assertTrue(np.isfinite(h).all())
        self.assertLess(np.linalg.norm(h-h.T), 1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(h)[0], -1e-12)

    def test_finite_state_endpoints(self):
        rng = np.random.default_rng(61)
        states = rng.normal(size=(6, 5))
        ids = np.r_[np.tile([0, 1, 2], 150), np.tile([3, 4, 5], 150)]
        h = estimate(states[ids])
        self.assertEqual(h.shape, (5, 5))
        self.assertTrue(np.isfinite(h).all())
        self.assertLess(np.linalg.norm(h-h.T), 1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(h)[0], -1e-10)


if __name__ == "__main__":
    unittest.main()
