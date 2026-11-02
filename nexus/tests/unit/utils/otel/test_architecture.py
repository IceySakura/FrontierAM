from __future__ import annotations

import ast

from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[4]
NEXUS_ROOT = PROJECT_ROOT / 'nexus'


def _production_python_files():
    for path in NEXUS_ROOT.rglob('*.py'):
        if 'tests' not in path.relative_to(NEXUS_ROOT).parts:
            yield path


def test_legacy_trace_otel_package_is_removed() -> None:
    assert not (NEXUS_ROOT / 'trace' / 'otel').exists()


def test_legacy_otel_implementations_are_removed() -> None:
    forbidden_symbols = {
        'ASGIInstrumentor',
        'FastAPIInstrumentor',
        'NEXUS_OTEL_INSTRUMENTATIONS',
        'SanitizingSpanExporter',
        'bootstrap_otel',
    }
    offenders: list[str] = []

    for path in _production_python_files():
        source = path.read_text()
        for symbol in forbidden_symbols:
            if symbol in source:
                offenders.append(f'{path.relative_to(PROJECT_ROOT)}:{symbol}')

    assert offenders == []


def test_direct_otel_imports_are_confined_to_utils_otel() -> None:
    offenders: list[str] = []

    for path in _production_python_files():
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if relative.startswith('nexus/utils/otel/'):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            imported_modules: list[str] = []
            if isinstance(node, ast.Import):
                imported_modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules = [node.module]
            if any(module == 'opentelemetry' or module.startswith('opentelemetry.') for module in imported_modules):
                offenders.append(f'{relative}:{node.lineno}')

    assert offenders == []
