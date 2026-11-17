"""Skip documentation and testing imports not used by the numerical runtime."""
import sys
import types
from collections import namedtuple

import numpy as np


class _Tester:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise RuntimeError("Test discovery is not part of the solver runtime")


class _FunctionDoc(dict):
    def __init__(self, function, *args, **kwargs):
        super().__init__(Notes=[])
        self.original = function.__doc__ or ""

    def __str__(self):
        return "\n" + self.original

    def __missing__(self, key):
        self[key] = []
        return self[key]


def prepare():
    tester = types.ModuleType("scipy._lib._testutils")
    tester.PytestTester = _Tester
    sys.modules["scipy._lib._testutils"] = tester
    docs = types.ModuleType("scipy._lib._docscrape")
    docs.FunctionDoc = _FunctionDoc
    docs.Parameter = namedtuple("Parameter", ["name", "type", "desc"])
    sys.modules["scipy._lib._docscrape"] = docs
    exported = np.__all__
    np.__all__ = [name for name in exported
                  if name not in ("f2py", "testing", "ma", "polynomial", "char",
                                  "ctypeslib", "rec", "core")]
    from scipy import sparse
    np.__all__ = exported
