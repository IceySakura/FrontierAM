from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from nexus.core.module_dependencies import ModuleSelector
from nexus.distributed.artifacts.models import ArtifactKind, ArtifactRef
from nexus.distributed.dependency_build.wheelhouse import (
    WheelhouseBuildSpec,
    extract_wheelhouse_archive,
)
from nexus.distributed.dependency_build.worker import build_wheelhouse
from nexus.distributed.environment.limits import SourcePackageLimitsSpec
from nexus.distributed.submission.source_package import PackageLimits, build_source_package
from nexus.distributed.submission.specs import resolve_source_package_extras


def build_local_module_wheelhouse(
    *,
    project_root: Path,
    uv_lock: Path,
    nexus_wheel: Path,
    output: Path,
    selectors: Sequence[ModuleSelector],
    representative_item_id: str | None,
    extras: Sequence[str] = (),
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    root = project_root.expanduser().resolve(strict=True)
    lock = _regular_file(uv_lock, label='uv lock')
    wheel = _regular_file(nexus_wheel, label='Nexus wheel')
    destination = Path(os.path.abspath(output.expanduser()))
    _reject_symlink_components(destination)
    if destination == root or destination in root.parents:
        raise ValueError('module wheel output must not replace the project root or one of its parents')

    limits = PackageLimits()
    with tempfile.TemporaryDirectory(prefix='nexus-module-wheel-') as temporary_name:
        temporary = Path(temporary_name)
        source = build_source_package(
            root,
            temporary / 'source.tar.zst',
            limits=limits,
            excluded_paths=(destination,),
        )
        spec = WheelhouseBuildSpec(
            owner_principal_id='local',
            environment_id='local',
            source_package=_artifact_ref(source.path, kind='source_package'),
            nexus_wheel=_artifact_ref(wheel, kind='nexus_wheel'),
            uv_lock=_artifact_ref(lock, kind='uv_lock'),
            source_package_limits=SourcePackageLimitsSpec.from_package_limits(limits),
            extras=resolve_source_package_extras(list(extras)),
            module_selectors=list(selectors),
            workload_image='local/module-wheel',
            builder_image='local/module-wheel',
        )
        archive, metadata = build_wheelhouse(
            spec=spec,
            source_package=source.path,
            nexus_wheel=wheel,
            uv_lock=lock,
            workspace=temporary / 'workspace',
            runner=runner,
        )
        extract_wheelhouse_archive(
            archive,
            destination,
            max_files=100_000,
            max_total_bytes=8 * 1024 * 1024 * 1024,
        )

    selector_values = [selector.model_dump(mode='json') for selector in selectors]
    report = {
        'schema_version': 1,
        'representative_item_id': representative_item_id,
        'selectors': selector_values,
        'wheelhouse': metadata.model_dump(mode='json'),
    }
    _write_json(destination / 'module-wheel-report.json', report)
    _write_json(destination / 'wheelhouse-metadata.json', metadata.model_dump(mode='json'))
    return report


def _regular_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f'{label} must be a regular file')
    return resolved


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError('module wheel output path must not contain symlinks')
        if not current.exists():
            break


def _artifact_ref(path: Path, *, kind: ArtifactKind) -> ArtifactRef:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ArtifactRef(
        artifact_id=f'art_local_{digest[:32]}',
        kind=kind,
        sha256=digest,
        size_bytes=path.stat().st_size,
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f'.{path.name}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


__all__ = ['build_local_module_wheelhouse']
