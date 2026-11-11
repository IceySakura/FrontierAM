"""Load the pinned numerical extension modules without eager package imports."""
import glob
import importlib.util
import os
import sys

_root = None


def extension(name, relative):
    global _root
    if name in sys.modules:
        return sys.modules[name]
    if _root is None:
        _root = list(importlib.util.find_spec("scipy").submodule_search_locations)[0]
    paths = glob.glob(os.path.join(_root, relative))
    if len(paths) != 1:
        raise ImportError("Numerical extension not found: " + relative)
    spec = importlib.util.spec_from_file_location(name, paths[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def highs():
    return extension("scipy.optimize._highspy._core", "optimize/_highspy/_core*.so")
