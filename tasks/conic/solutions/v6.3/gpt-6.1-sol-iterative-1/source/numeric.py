"""Minimal loader for the installed SuperLU extension."""

import importlib.util
import os
import sys

import numpy as np
from scipy import sparse


_superlu = None


def superlu():
    global _superlu
    if _superlu is None:
        name = "scipy.sparse.linalg._dsolve._superlu"
        if name in sys.modules:
            _superlu = sys.modules[name]
        else:
            base = os.path.dirname(importlib.util.find_spec("scipy").origin)
            directory = os.path.join(base, "sparse", "linalg", "_dsolve")
            path = next(os.path.join(directory, f) for f in os.listdir(directory)
                        if f.startswith("_superlu.") and f.endswith(".so"))
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            _superlu = module
    return _superlu


def splu(A, permc_spec="COLAMD", diag_pivot_thresh=1., relax=None, panel_size=None, options=None):
    A = A.tocsc()
    A.sum_duplicates()
    A.sort_indices()
    if A.dtype != np.float64:
        A = A.astype(np.float64)
    opts = {"DiagPivotThresh": diag_pivot_thresh, "ColPerm": permc_spec,
            "PanelSize": panel_size, "Relax": relax}
    if options:
        opts.update(options)
    if opts["ColPerm"] == "NATURAL":
        opts["SymmetricMode"] = True
    return superlu().gstrf(A.shape[0], A.nnz, A.data,
                          np.asarray(A.indices, dtype=np.intc),
                          np.asarray(A.indptr, dtype=np.intc),
                          csc_construct_func=sparse.csc_matrix, ilu=False, options=opts)


class LazyLinalg:
    def __getattr__(self, name):
        from scipy import linalg
        return getattr(linalg, name)


linalg = LazyLinalg()
