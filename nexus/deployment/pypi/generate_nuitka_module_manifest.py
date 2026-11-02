from __future__ import annotations

import argparse
import pprint
import sys

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol


class ModuleMetaLike(Protocol):
    name: str
    module_path: str
    description: str
    version: str
    authors: Sequence[Mapping[str, str]]
    maintainers: Sequence[Mapping[str, str]]
    base_deps: Sequence[str]
    host_extras: Sequence[str]
    runtime_extras: Sequence[str]
    objects: Mapping[str, Any]
    kind: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _record_for_meta(meta: ModuleMetaLike) -> dict[str, Any]:
    return {
        'name': meta.name,
        'module_path': meta.module_path,
        'description': meta.description,
        'version': meta.version,
        'authors': [dict(person) for person in meta.authors],
        'maintainers': [dict(person) for person in meta.maintainers],
        'base_deps': list(meta.base_deps),
        'host_extras': list(meta.host_extras),
        'runtime_extras': list(meta.runtime_extras),
        'objects': meta.objects,
        'kind': meta.kind,
    }


def render_manifest_source(modules: list[ModuleMetaLike]) -> str:
    records = [_record_for_meta(meta) for meta in sorted(modules, key=lambda item: item.module_path)]
    return (
        '"""Generated module metadata for Nuitka binary builds."""\n\n'
        'from __future__ import annotations\n\n'
        f'COMPILED_MODULES = {pprint.pformat(records, width=120, sort_dicts=True)}\n'
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()

    root = _repo_root()
    sys.path.insert(0, str(root))

    from nexus.core.module_meta import discover_all_modules

    output = root / 'nexus' / '_compiled_module_manifest.py'
    source = render_manifest_source(discover_all_modules(project_root=root, include_compiled_manifest=False))

    if args.check:
        if not output.exists() or output.read_text() != source:
            print(f'{output} is out of date', file=sys.stderr)
            return 1
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source)
    print(f'Wrote {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
