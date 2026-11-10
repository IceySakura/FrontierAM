"""Build reusable bytecode caches without touching any problem inputs."""
import argparse
import compileall
import json
import os
import sys
import sysconfig

import numpy
from scipy import linalg, sparse
from scipy.sparse import linalg as sparse_linalg

import solver
import conic

solver.native_highs()
from scipy.optimize import minimize
compileall.compile_dir(os.path.dirname(__file__), quiet=2, maxlevels=1)
