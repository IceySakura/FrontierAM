import importlib
import types
import numpy as np


class DeferredModule(types.ModuleType):
    def __getattr__(self, name):
        module = importlib.import_module(self.__name__)
        return getattr(module, name)


def load_solver():
    # Array-API compatibility imports NumPy's public namespace eagerly.
    # Development-only modules remain available without delaying numeric startup.
    deferred = {}
    for name in ("testing", "f2py"):
        if name not in np.__dict__:
            proxy = DeferredModule("numpy."+name)
            deferred[name] = proxy
            setattr(np, name, proxy)
    try:
        from solver import main
    finally:
        for name, proxy in deferred.items():
            if np.__dict__.get(name) is proxy:
                delattr(np, name)
    return main
