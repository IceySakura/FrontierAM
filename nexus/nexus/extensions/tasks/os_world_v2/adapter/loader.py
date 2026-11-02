"""Independent V2 adapter: lazy fork imports inside the per-attempt worker."""

import ast
import importlib
import sys

from pathlib import Path
from types import ModuleType


def load(root: Path, selected: str):
    sys.path.insert(0, str(root))
    importlib.import_module('desktop_env.evaluators')
    for package in ('getters', 'metrics'):
        name = f'desktop_env.evaluators.{package}'
        directory = root / 'desktop_env/evaluators' / package
        exports = {}
        tree = ast.parse((directory / '__init__.py').read_text())
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                for alias in node.names:
                    exports[alias.asname or alias.name] = (node.module, alias.name)
        module = ModuleType(name)
        module.__path__ = [str(directory)]
        module.__file__ = str(directory / '__init__.py')

        def lazy(attribute, *, mapping=exports, prefix=name, target=module):
            if attribute not in mapping:
                raise AttributeError(attribute)
            child, original = mapping[attribute]
            value = getattr(importlib.import_module(f'{prefix}.{child}'), original)
            setattr(target, attribute, value)
            return value

        module.__dict__['__getattr__'] = lazy
        sys.modules[name] = module
        setattr(sys.modules['desktop_env.evaluators'], package, module)
    module = importlib.import_module(f'evaluation_examples.task_class.task_{selected}')
    factory = getattr(module, 'get_task', None) or getattr(module, 'TASK_CLASS', None)
    if factory is None:
        raise ValueError('Pinned task does not expose TASK_CLASS/get_task')
    task = factory()
    if task.get('id') != selected:
        raise ValueError('Loaded task identity mismatch')
    return task
