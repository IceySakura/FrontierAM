"""Automatic built-in rule discovery.

Every ``rules/<rule_id>.py`` module (except private modules and ``base.py``)
must export one ``RULE`` object whose id equals the filename. Adding a detector
therefore requires adding exactly one Python file; no central registry edit is
needed. Modules are loaded in filename order for deterministic output.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules

from .base import Rule


def load_rules() -> tuple[Rule, ...]:
    package_dir = Path(__file__).parent
    module_names = sorted(
        item.name for item in iter_modules([str(package_dir)]) if item.name != 'base' and not item.name.startswith('_')
    )
    rules: list[Rule] = []
    for module_name in module_names:
        module = import_module(f'{__name__}.{module_name}')
        rule = getattr(module, 'RULE', None)
        if not isinstance(rule, Rule):
            raise RuntimeError(f'rule module {module_name!r} must export a Rule as RULE')
        if rule.id != module_name:
            raise RuntimeError(f'rule id {rule.id!r} must match module filename {module_name!r}')
        rules.append(rule)
    ids = [rule.id for rule in rules]
    if len(ids) != len(set(ids)):
        raise RuntimeError('duplicate bad-pattern rule id')
    return tuple(rules)


RULES = load_rules()
RULE_BY_ID = {rule.id: rule for rule in RULES}
RULE_TITLES = {rule.id: rule.title for rule in RULES}

__all__ = ['RULES', 'RULE_BY_ID', 'RULE_TITLES', 'Rule', 'load_rules']
