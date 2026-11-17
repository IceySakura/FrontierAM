import numpy as np

class Solver:
    def solve(self, hvp, b, context):
        return np.zeros_like(b)

def make_solver(dimension, seed, options):
    return Solver()
