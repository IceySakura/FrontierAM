from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import Version


def read_project_constraints(document: dict[str, Any]) -> tuple[str, ...]:
    tool = document.get('tool') or {}
    if tool and not isinstance(tool, dict):
        raise ValueError('source project tool metadata must be a table')
    uv = tool.get('uv') or {}
    if uv and not isinstance(uv, dict):
        raise ValueError('source project tool.uv metadata must be a table')
    return _requirement_list(
        uv.get('constraint-dependencies') or [],
        label='tool.uv.constraint-dependencies',
    )


def read_target_lock(path: Path) -> dict[str, Requirement]:
    requirements: dict[str, Requirement] = {}
    for raw in _lock_requirement_lines(path):
        try:
            requirement = Requirement(raw)
        except InvalidRequirement as exc:
            raise ValueError(f'lock contains an invalid requirement: {raw}') from exc
        if not _requirement_applies(requirement):
            continue
        name = canonicalize_name(requirement.name)
        previous = requirements.get(name)
        if previous is not None and str(previous) != str(requirement):
            raise ValueError(f'lock selects multiple requirements for {name}: {previous}; {requirement}')
        requirements[name] = requirement
    return requirements


def validate_wheels_against_lock(lock_path: Path, wheels_dir: Path) -> dict[str, str]:
    locked = read_target_lock(lock_path)
    wheel_versions: dict[str, set[Version]] = {}
    for path in wheels_dir.glob('*.whl'):
        try:
            name, version, _build, _tags = parse_wheel_filename(path.name)
        except InvalidWheelFilename as exc:
            raise ValueError(f'invalid dependency wheel filename: {path.name}') from exc
        wheel_versions.setdefault(canonicalize_name(name), set()).add(version)

    unexpected = sorted(set(wheel_versions) - set(locked))
    missing = sorted(set(locked) - set(wheel_versions))
    if unexpected:
        raise ValueError(f'wheelhouse contains distributions absent from target lock: {", ".join(unexpected)}')
    if missing:
        raise ValueError(f'target lock distributions are missing from wheelhouse: {", ".join(missing)}')

    selected: dict[str, str] = {}
    for name, requirement in locked.items():
        versions = wheel_versions[name]
        if len(versions) != 1:
            rendered = ', '.join(str(version) for version in sorted(versions))
            raise ValueError(f'wheelhouse contains multiple versions for {name}: {rendered}')
        version = next(iter(versions))
        if requirement.specifier and not requirement.specifier.contains(version, prereleases=True):
            raise ValueError(f'wheelhouse version {name}=={version} does not satisfy target lock {requirement}')
        selected[name] = str(version)
    return selected


def _requirement_applies(requirement: Requirement) -> bool:
    environment = cast(dict[str, str], default_environment())
    return requirement.marker is None or requirement.marker.evaluate(environment=environment)


def _requirement_list(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f'{label} must be a list of PEP 508 strings')
    try:
        for item in value:
            Requirement(item)
    except InvalidRequirement as exc:
        raise ValueError(f'{label} contains an invalid requirement') from exc
    return tuple(value)


def _lock_requirement_lines(path: Path) -> list[str]:
    result: list[str] = []
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith(('#', '--')):
            continue
        candidate = line.removesuffix('\\').strip().split(' --hash=', 1)[0]
        if candidate:
            result.append(candidate)
    return result


__all__ = [
    'read_project_constraints',
    'read_target_lock',
    'validate_wheels_against_lock',
]
