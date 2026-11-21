import glob
import importlib.util
import sys
import numpy as np

_CORE = None


def core():
    global _CORE
    if _CORE is None:
        name = "scipy.optimize._highspy._core"
        if name in sys.modules:
            _CORE = sys.modules[name]
            return _CORE
        root = importlib.util.find_spec("scipy").submodule_search_locations[0]
        path = glob.glob(root + "/optimize/_highspy/_core*.so")[0]
        spec = importlib.util.spec_from_file_location(name, path)
        _CORE = importlib.util.module_from_spec(spec)
        sys.modules[name] = _CORE
        spec.loader.exec_module(_CORE)
    return _CORE


def lp(q, indptr, indices, data, lower, upper, col_lower=None, col_upper=None,
       options=None):
    h = core()
    n, m = len(q), len(upper)
    solver = h._Highs()
    settings = {
        "output_flag": False, "threads": 1, "parallel": "off",
        "primal_feasibility_tolerance": 1e-9,
        "dual_feasibility_tolerance": 1e-9,
        "ipm_optimality_tolerance": 1e-11,
        "small_matrix_value": 1e-12, "time_limit": 280.0,
    }
    if options:
        settings.update(options)
    for key, value in settings.items():
        solver.setOptionValue(key, value)
    lb = np.full(n, -np.inf) if col_lower is None else col_lower
    ub = np.full(n, np.inf) if col_upper is None else col_upper
    solver.passModel(n, m, len(data), 1, 1, 0., q, lb, ub, lower, upper,
                     indptr, indices, data, np.zeros(n, dtype=np.int32))
    return solver
