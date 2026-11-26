"""Plain Monte Carlo: the whole budget on independent N(0, I_d) points (the 0-point anchor of the final score).

The error bound is the CLT half-width at 95 % (1.96 standard errors), reported as a confidence-0.95 bound.
"""
import math

import numpy as np

from cubature_protocol import Session


def main():
    session = Session()
    z = np.random.default_rng([session.seed, 99]).standard_normal((session.budget, session.d))
    values = session.query(z)
    half_width = 1.96 * values.std(ddof=1) / math.sqrt(len(values))
    session.result(values.mean(), error_bound=half_width, confidence=0.95)


if __name__ == '__main__':
    main()
