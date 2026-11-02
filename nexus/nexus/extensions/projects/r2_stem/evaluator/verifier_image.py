"""Trusted post-agent evaluation in a separate verifier image."""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import logging
import math
import os
import posixpath
import re
import shlex

from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nexus.core import require_context
from nexus.extensions.projects.r2_stem.evaluator.common import (
    build_verifier_test_results,
    collect_verifier_outputs,
    resolve_verifier_env,
    verifier_registry_network_target,
)
from nexus.runtime.executors.models.file import FileItem
from nexus.trace import get_current_span_id, update_span


logger = logging.getLogger(__name__)

IMAGE_DIGEST_RE = re.compile(r'^.+@sha256:[0-9a-fA-F]{64}$')
MAX_VERIFIER_ARTIFACT_FILES = 2048
MAX_VERIFIER_ARTIFACT_BYTES = 5 * 1024 * 1024 * 1024
# Oversized artifacts are split on the agent so no single transfer has to be held whole, and
# transfers are flushed once a batch reaches this size. Together these bound peak memory to
# roughly one batch regardless of how large the declared artifacts are. The ceiling is the
# driver rather than the sandboxes: a batch costs ~4x its size there (base64 response body,
# parsed string, upload body) and the runner stages many samples concurrently.
ARTIFACT_CHUNK_BYTES = 1024 * 1024 * 1024
ARTIFACT_BATCH_BYTES = 1024 * 1024 * 1024
TESTS_DIR = '/tests'
TEST_SCRIPT = f'{TESTS_DIR}/test.sh'
AGENT_ARTIFACT_ROOT = '/tmp/nexus_agent_artifacts'
VERIFIER_ARTIFACT_ROOT = '/tmp/nexus_verifier_artifacts'
DOCKER_SOCKET = '/var/run/docker.sock'
PROTECTED_VERIFIER_PATHS = (
    TESTS_DIR,
    '/logs/verifier',
    VERIFIER_ARTIFACT_ROOT,
    DOCKER_SOCKET,
    '/proc',
    '/sys',
    '/dev',
    '/etc',
    '/usr',
    '/bin',
    '/sbin',
    '/lib',
    '/lib64',
    '/opt',
)


class VerifierArtifact(BaseModel):
    """One explicitly allowed artifact copied from the agent to the verifier."""

    model_config = ConfigDict(extra='forbid')

    source: str
    target: str | None = None
    exclude: list[str] = Field(default_factory=list)


def _safe_component(value: str) -> str:
    safe = ''.join(ch.lower() if ch.isalnum() else '-' for ch in value).strip('-') or 'task'
    if len(safe) <= 48:
        return safe
    suffix = hashlib.sha256(value.encode()).hexdigest()[:12]
    return safe[:48].rstrip('-') + '-' + suffix


def _finite_reward(value: Any) -> float:
    if isinstance(value, bool):
        raise RuntimeError('isolated verifier reward must be a finite number, not a boolean')
    try:
        reward = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError('isolated verifier reward must be a finite number') from exc
    if not math.isfinite(reward):
        raise RuntimeError('isolated verifier reward must be finite')
    return reward


async def _pull_docker_image(runtime: Any, image: str, timeout: int) -> None:
    attempts = max(1, int(os.getenv('STEM_AGENT_DOCKER_PULL_RETRIES', '6')))
    last_result: Any = None
    for attempt in range(1, attempts + 1):
        last_result = await runtime.run_command(command=f'docker pull {shlex.quote(image)}', timeout=timeout)
        if getattr(last_result, 'return_code', 0) == 0:
            return
        if attempt < attempts:
            await asyncio.sleep(min(10.0 * (2 ** (attempt - 1)), 60.0))
    raise RuntimeError(
        f'pull verifier image failed after {attempts} attempts; '
        f'exit_code={getattr(last_result, "return_code", None)}; '
        f'stderr={getattr(last_result, "stderr", "") or ""}'
    )


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'1', 'true', 'yes', 'on'}:
            return True
        if normalized in {'0', 'false', 'no', 'off', ''}:
            return False
    if value in {0, 1, None}:
        return bool(value)
    raise ValueError(f'invalid boolean runtime setting: {value!r}')


def _validate_verifier_runtime_topology(
    ctx: Any,
    *,
    verifier_runtime_name: str,
    verifier_image: str,
    verifier_dind: bool,
    verifier_network_access: bool,
) -> None:
    provider = getattr(ctx, 'runtime_provider', None)
    if provider is None or not hasattr(provider, 'get_params'):
        raise RuntimeError('isolated verification requires an inspectable runtime provider')
    params = provider.get_params(verifier_runtime_name)
    declared = dict((getattr(ctx, 'runtime_configs', None) or {}).get(verifier_runtime_name) or {})
    effective_image = params.get('image') or declared.get('image')
    if not IMAGE_DIGEST_RE.fullmatch(verifier_image):
        raise ValueError(f'verifier_image must be pinned by sha256 digest: {verifier_image!r}')
    if effective_image != verifier_image:
        raise ValueError(
            f'runtime_configs.{verifier_runtime_name}.image must exactly match task_config.verifier_image; '
            f'effective={effective_image!r}, verifier_image={verifier_image!r}'
        )

    effective_dind = _coerce_bool(params.get('enable_dind', declared.get('enable_dind', False)))
    if effective_dind != verifier_dind:
        raise ValueError(
            f'runtime_configs.{verifier_runtime_name}.enable_dind does not match task_config.verifier_dind; '
            f'effective={effective_dind}, verifier_dind={verifier_dind}'
        )

    network_policy = params.get('network_policy', declared.get('network_policy'))
    default_action = network_policy.get('default_action') if isinstance(network_policy, dict) else None
    if not isinstance(network_policy, dict) or default_action not in {'deny', 'allow'}:
        raise ValueError('isolated verifier runtime requires network_policy.default_action of "deny" or "allow"')
    egress = network_policy.get('egress') or []
    if not isinstance(egress, list) or any(not isinstance(rule, dict) for rule in egress):
        raise ValueError('isolated verifier runtime network_policy.egress must be a list of rules')
    if default_action == 'allow':
        if not verifier_network_access:
            raise ValueError('network_policy.default_action="allow" requires verifier_network_access=true')
        return

    # Only the nested runner needs egress at all, and only to re-pull its own image into the
    # inner daemon. Direct execution reads the tests off the rootfs the platform already built.
    registry_rule = {'action': 'allow', 'target': verifier_registry_network_target(verifier_image)}
    if verifier_dind and registry_rule not in egress:
        raise ValueError('dind verifier runtime must allow its exact image registry for the inner bootstrap pull')
    if not verifier_network_access:
        permitted = {json.dumps(registry_rule, sort_keys=True)} if verifier_dind else set()
        unexpected_allows = [
            rule
            for rule in egress
            if rule.get('action') == 'allow' and json.dumps(rule, sort_keys=True) not in permitted
        ]
        if unexpected_allows:
            raise ValueError(
                'verifier_network_access=false permits only the verifier image registry bootstrap rule'
                if verifier_dind
                else 'verifier_network_access=false forbids allow rules in the verifier runtime egress policy'
            )


def _paths_overlap(left: str, right: str) -> bool:
    return left == right or left.startswith(right + '/') or right.startswith(left + '/')


def _normalize_artifact_path(raw_path: str, *, label: str) -> str:
    if not isinstance(raw_path, str) or not raw_path or '\x00' in raw_path:
        raise ValueError(f'invalid {label}: {raw_path!r}')
    if not raw_path.startswith('/') or raw_path.startswith('//') or '..' in PurePosixPath(raw_path).parts:
        raise ValueError(f'invalid {label}: {raw_path!r}')
    normalized = posixpath.normpath(raw_path)
    if normalized.startswith('//') or normalized == '/':
        raise ValueError(f'invalid {label}: {raw_path!r}')
    return normalized


def _normalize_exclude_patterns(raw_patterns: list[str]) -> list[str]:
    normalized: list[str] = []
    for raw_pattern in raw_patterns:
        if not isinstance(raw_pattern, str) or not raw_pattern or '\x00' in raw_pattern:
            raise ValueError(f'invalid verifier artifact exclude pattern: {raw_pattern!r}')
        pattern = raw_pattern[2:] if raw_pattern.startswith('./') else raw_pattern
        if pattern.startswith('/') or '..' in PurePosixPath(pattern).parts:
            raise ValueError(f'invalid verifier artifact exclude pattern: {raw_pattern!r}')
        normalized.append(raw_pattern)
    return list(dict.fromkeys(normalized))


def _artifact_path_excluded(relative_path: PurePosixPath, patterns: list[str]) -> bool:
    relative = relative_path.as_posix()
    name = relative_path.name
    parts = relative_path.parts
    return any(
        fnmatch.fnmatch(relative, pattern)
        or fnmatch.fnmatch(name, pattern)
        or ('/' not in pattern and pattern in parts)
        for raw_pattern in patterns
        for pattern in [raw_pattern[2:] if raw_pattern.startswith('./') else raw_pattern]
    )


def _artifact_mount_spec(raw_mount: Any) -> tuple[str, str, list[str]]:
    if isinstance(raw_mount, str):
        return raw_mount, raw_mount, []
    if isinstance(raw_mount, BaseModel):
        raw_mount = raw_mount.model_dump(exclude_none=True)
    if isinstance(raw_mount, dict):
        unknown = set(raw_mount) - {'source', 'target', 'exclude'}
        if unknown:
            raise ValueError(f'unknown verifier artifact mount fields: {sorted(unknown)!r}')
        source = raw_mount.get('source')
        target = raw_mount.get('target') or source
        excludes = raw_mount.get('exclude') or []
        if not isinstance(source, str) or not source:
            raise ValueError(f'invalid verifier artifact mount: {raw_mount!r}')
        if not isinstance(target, str) or not target:
            raise ValueError(f'invalid verifier artifact target: {raw_mount!r}')
        if not isinstance(excludes, list) or any(not isinstance(item, str) for item in excludes):
            raise ValueError(f'invalid verifier artifact exclude list: {raw_mount!r}')
        return source, target, _normalize_exclude_patterns(excludes)
    raise ValueError(f'invalid verifier artifact mount: {raw_mount!r}')


def _validate_artifact_mounts(
    mounts: list[Any],
    allowed_protected_mounts: list[str] | None = None,
) -> list[tuple[str, str, list[str]]]:
    allowed: set[str] = set()
    for raw_path in allowed_protected_mounts or []:
        path = _normalize_artifact_path(raw_path, label='protected verifier artifact exception')
        if path in PROTECTED_VERIFIER_PATHS:
            raise ValueError(f'invalid protected verifier artifact exception: {raw_path!r}')
        allowed.add(path)

    normalized: list[tuple[str, str, list[str]]] = []
    for raw_mount in mounts:
        raw_source, raw_target, excludes = _artifact_mount_spec(raw_mount)
        source = _normalize_artifact_path(raw_source, label='verifier artifact mount')
        target = _normalize_artifact_path(raw_target, label='verifier artifact target')
        if any(_paths_overlap(source, protected) for protected in PROTECTED_VERIFIER_PATHS) and source not in allowed:
            raise ValueError(f'verifier artifact source overlaps a protected path: {raw_mount!r}')
        if any(_paths_overlap(target, protected) for protected in PROTECTED_VERIFIER_PATHS) and target not in allowed:
            raise ValueError(f'verifier artifact target overlaps a protected path: {raw_mount!r}')
        if any(_paths_overlap(target, existing_target) for _, existing_target, _ in normalized):
            raise ValueError(f'overlapping verifier artifact mounts are not allowed: {raw_mount!r}')
        normalized.append((source, target, excludes))
    return sorted(normalized, key=lambda item: item[1])


async def _require_image_test_script(verifier_runtime: Any) -> None:
    """Fail fast when the verifier image does not ship the trusted tests it is supposed to own."""
    checked = await verifier_runtime.run_command(
        command=f'test -f {shlex.quote(TEST_SCRIPT)} && test ! -L {shlex.quote(TEST_SCRIPT)}',
        timeout=120,
    )
    if getattr(checked, 'return_code', 0) != 0:
        raise FileNotFoundError(f'verifier image must provide {TEST_SCRIPT} as a regular file')


# Freezing and splitting happen in one pass on the agent so the bytes that get scored are a
# single point-in-time copy, taken before any transfer round trip begins.
_ARTIFACT_SNAPSHOT_PROGRAM = """
import fnmatch
import json
import os
import stat
import sys

staging_root = sys.argv[1]
chunk_bytes = int(sys.argv[2])
mounts = json.loads(sys.argv[3])
READ_BLOCK = 256 * 1024 * 1024


def excluded(root, path, patterns):
    rel = os.path.relpath(path, root).replace(os.sep, "/")
    if rel == ".":
        return False
    name = os.path.basename(path)
    parts = rel.split("/")
    for raw_pattern in patterns:
        pattern = raw_pattern[2:] if raw_pattern.startswith("./") else raw_pattern
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern) or ("/" not in pattern and pattern in parts):
            return True
    return False


def freeze(source, target):
    staged = staging_root + target
    os.makedirs(os.path.dirname(staged), exist_ok=True)
    single = os.path.getsize(source) <= chunk_bytes
    size = 0
    parts = []
    with open(source, "rb") as handle:
        while True:
            part = staged if single else "%s.part%05d" % (staged, len(parts))
            written = 0
            # Read in small blocks so this pass stays flat in memory even when a chunk is large.
            with open(part, "wb") as out:
                while written < chunk_bytes:
                    block = handle.read(min(READ_BLOCK, chunk_bytes - written))
                    if not block:
                        break
                    out.write(block)
                    written += len(block)
            if written == 0 and parts:
                os.remove(part)
                break
            size += written
            parts.append({"path": part, "size": written})
            if single or written < chunk_bytes:
                break
    return {"target": target, "size": size, "parts": parts}


payload = {"mounts": [], "files": []}
for mount in mounts:
    source = mount["source"]
    target = mount["target"]
    patterns = mount["exclude"]
    kind = "missing"
    if os.path.lexists(source):
        mode = os.lstat(source).st_mode
        if stat.S_ISLNK(mode):
            kind = "symlink"
        elif stat.S_ISREG(mode):
            kind = "file"
            payload["files"].append(freeze(source, target))
        elif stat.S_ISDIR(mode):
            kind = "dir"
            for current, directories, names in os.walk(source, followlinks=False):
                directories[:] = [
                    name for name in directories
                    if not os.path.islink(os.path.join(current, name))
                    and not excluded(source, os.path.join(current, name), patterns)
                ]
                for name in sorted(names):
                    path = os.path.join(current, name)
                    if excluded(source, path, patterns) or not stat.S_ISREG(os.lstat(path).st_mode):
                        continue
                    relative = os.path.relpath(path, source).replace(os.sep, "/")
                    payload["files"].append(freeze(path, target + "/" + relative))
        else:
            kind = "unsupported"
    payload["mounts"].append({"source": source, "target": target, "kind": kind})
print(json.dumps(payload, sort_keys=True))
"""


def _base64_decoded_size(encoded: str) -> int:
    """Decoded length of a canonical base64 payload, without materialising the bytes."""
    if len(encoded) % 4:
        raise ValueError('malformed base64 artifact payload')
    if not encoded:
        return 0
    padding = 2 if encoded.endswith('==') else 1 if encoded.endswith('=') else 0
    return len(encoded) // 4 * 3 - padding


async def _freeze_agent_artifacts(
    agent_runtime: Any,
    agent_root: str,
    normalized: list[tuple[str, str, list[str]]],
) -> dict[str, Any]:
    prepared = await agent_runtime.run_command(
        command=f'rm -rf {shlex.quote(agent_root)} && mkdir -p {shlex.quote(agent_root)}',
        timeout=120,
    )
    if getattr(prepared, 'return_code', 0) != 0:
        raise RuntimeError('could not prepare agent artifact snapshot directory')

    spec = [{'source': source, 'target': target, 'exclude': excludes} for source, target, excludes in normalized]
    snapshot = await agent_runtime.run_command(
        command=(
            f'python3 -c {shlex.quote(_ARTIFACT_SNAPSHOT_PROGRAM)} {shlex.quote(agent_root)} '
            f'{ARTIFACT_CHUNK_BYTES} {shlex.quote(json.dumps(spec))}'
        ),
        timeout=600,
    )
    if getattr(snapshot, 'return_code', 0) != 0:
        raise RuntimeError(f'could not snapshot verifier artifacts: {getattr(snapshot, "stderr", "") or ""}')
    try:
        manifest = json.loads((getattr(snapshot, 'stdout', '') or '').strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError('invalid verifier artifact manifest') from exc
    if not isinstance(manifest, dict) or set(manifest) != {'mounts', 'files'}:
        raise TypeError('invalid verifier artifact manifest schema')
    return manifest


def _validated_manifest_mounts(
    manifest: dict[str, Any],
    normalized: list[tuple[str, str, list[str]]],
) -> list[str]:
    """Confirm the runtime described exactly the mounts we asked for, and return their kinds."""
    reported = manifest.get('mounts')
    if not isinstance(reported, list) or len(reported) != len(normalized):
        raise TypeError('invalid verifier artifact mount list')
    kinds: list[str] = []
    for entry, (source_mount, target_mount, _excludes) in zip(reported, normalized, strict=True):
        if not isinstance(entry, dict) or set(entry) != {'source', 'target', 'kind'}:
            raise TypeError(f'invalid artifact mount entry for {source_mount}')
        if entry.get('source') != source_mount or entry.get('target') != target_mount:
            raise ValueError(f'artifact manifest reordered the declared mounts: {entry!r}')
        kind = entry.get('kind')
        if kind not in {'missing', 'file', 'dir'}:
            raise ValueError(f'verifier artifact mount must be a regular file or directory: {source_mount}')
        kinds.append(str(kind))
    return kinds


def _validated_manifest_files(
    manifest: dict[str, Any],
    normalized: list[tuple[str, str, list[str]]],
    agent_root: str,
) -> list[dict[str, Any]]:
    """Re-check the reported files against the declared mounts, then against the transfer budget."""

    files = manifest.get('files')
    if not isinstance(files, list):
        raise TypeError('invalid verifier artifact file list')

    seen_targets: set[str] = set()
    total_bytes = 0
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {'target', 'size', 'parts'}:
            raise TypeError(f'invalid artifact entry: {entry!r}')
        size, parts = entry['size'], entry['parts']
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise TypeError(f'invalid artifact size for {entry["target"]!r}')
        if not isinstance(parts, list) or not parts:
            raise TypeError(f'invalid artifact part list for {entry["target"]!r}')

        target = _normalize_artifact_path(entry['target'], label='artifact manifest path')
        if target in seen_targets:
            raise ValueError(f'duplicate artifact manifest path: {target}')
        seen_targets.add(target)
        owner = next(
            (
                (source_mount, target_mount, excludes)
                for source_mount, target_mount, excludes in normalized
                if target == target_mount or target.startswith(target_mount + '/')
            ),
            None,
        )
        if owner is None:
            raise ValueError(f'artifact escaped every declared mount: {target}')
        _source_mount, target_mount, excludes = owner
        if target != target_mount:
            relative = PurePosixPath(target).relative_to(target_mount)
            if _artifact_path_excluded(relative, excludes):
                raise ValueError(f'excluded artifact was returned by runtime manifest: {target}')

        part_bytes = 0
        for part in parts:
            if not isinstance(part, dict) or set(part) != {'path', 'size'}:
                raise TypeError(f'invalid artifact part for {target}')
            part_size = part['size']
            if isinstance(part_size, bool) or not isinstance(part_size, int) or part_size < 0:
                raise TypeError(f'invalid artifact part size for {target}')
            if part_size > ARTIFACT_CHUNK_BYTES:
                raise ValueError(f'artifact part exceeds the chunk limit for {target}')
            path = _normalize_artifact_path(part['path'], label='artifact part path')
            if not path.startswith(agent_root + '/'):
                raise ValueError(f'artifact part escaped the snapshot directory: {path}')
            part_bytes += part_size
        if part_bytes != size:
            raise ValueError(f'artifact parts do not add up for {target}: parts={part_bytes}, size={size}')
        total_bytes += size

    # Budget is enforced on the manifest, so an oversized artifact set costs nothing to reject.
    if len(files) > MAX_VERIFIER_ARTIFACT_FILES:
        raise ValueError(f'verifier artifacts exceed the {MAX_VERIFIER_ARTIFACT_FILES} file limit: {len(files)}')
    if total_bytes > MAX_VERIFIER_ARTIFACT_BYTES:
        raise ValueError(f'verifier artifacts exceed the {MAX_VERIFIER_ARTIFACT_BYTES} byte limit: {total_bytes}')
    return files


async def _transfer_artifact_batch(
    agent_runtime: Any,
    verifier_runtime: Any,
    batch: list[tuple[str, str, int]],
) -> None:
    """Move one bounded batch of parts across in a single download and a single upload."""
    if not batch:
        return
    downloaded = await agent_runtime.download_file(
        files=[FileItem(path=source, encoding='base64') for source, _target, _expected in batch]
    )
    received = getattr(downloaded, 'files', None) or []
    if len(received) != len(batch):
        raise RuntimeError(f'could not download declared verifier artifacts: {getattr(downloaded, "errors", None)}')

    uploads: list[FileItem] = []
    for item, (source, target, expected) in zip(received, batch, strict=True):
        content = item.content
        if not isinstance(content, str):
            raise TypeError(f'invalid downloaded artifact content for {source}')
        actual = _base64_decoded_size(content)
        if actual != expected:
            raise ValueError(f'artifact changed during transfer for {source}: expected={expected}, actual={actual}')
        uploads.append(FileItem(path=target, content=content, encoding='base64'))

    uploaded = await verifier_runtime.upload_file(files=uploads, overwrite=True)
    results = getattr(uploaded, 'results', None) or []
    if len(results) != len(uploads) or any(not result.success for result in results):
        raise RuntimeError(f'could not upload declared verifier artifacts: {results}')


async def _stage_verifier_artifacts(
    agent_runtime: Any,
    verifier_runtime: Any,
    mounts: list[Any],
    allowed_protected_mounts: list[str],
    sample_id: str,
) -> list[tuple[str, str]]:
    component = _safe_component(sample_id)
    agent_root = f'{AGENT_ARTIFACT_ROOT}/{component}'
    staging_root = f'{VERIFIER_ARTIFACT_ROOT}/{component}'
    normalized = _validate_artifact_mounts(mounts, allowed_protected_mounts)

    manifest = await _freeze_agent_artifacts(agent_runtime, agent_root, normalized)
    kinds = _validated_manifest_mounts(manifest, normalized)
    files = _validated_manifest_files(manifest, normalized, agent_root)

    mount_pairs = [(staging_root + target_mount, target_mount) for _source, target_mount, _excludes in normalized]
    # Directory and missing mounts still need their root to exist so the verifier sees an
    # empty directory rather than a bind-mount failure.
    skeleton = ' '.join(
        shlex.quote(staged)
        for (staged, _target), kind in zip(mount_pairs, kinds, strict=True)
        if kind in {'dir', 'missing'}
    )
    prepared = await verifier_runtime.run_command(
        command=(
            f'rm -rf {shlex.quote(staging_root)} && mkdir -p {shlex.quote(staging_root)}'
            + (f' && mkdir -p {skeleton}' if skeleton else '')
        ),
        timeout=120,
    )
    if getattr(prepared, 'return_code', 0) != 0:
        raise RuntimeError('could not prepare verifier artifact staging directory')

    batch: list[tuple[str, str, int]] = []
    batch_bytes = 0
    reassembly: list[str] = []
    for entry in files:
        staged_parts = []
        for part in entry['parts']:
            staged = staging_root + part['path'][len(agent_root) :]
            staged_parts.append(staged)
            batch.append((part['path'], staged, part['size']))
            batch_bytes += part['size']
            if batch_bytes >= ARTIFACT_BATCH_BYTES:
                await _transfer_artifact_batch(agent_runtime, verifier_runtime, batch)
                batch, batch_bytes = [], 0
        if len(staged_parts) > 1:
            joined = ' '.join(shlex.quote(path) for path in staged_parts)
            reassembly.append(f'cat {joined} > {shlex.quote(staging_root + entry["target"])} && rm -f {joined}')
    await _transfer_artifact_batch(agent_runtime, verifier_runtime, batch)

    if reassembly:
        rejoined = await verifier_runtime.run_command(command='set -e; ' + '; '.join(reassembly), timeout=600)
        if getattr(rejoined, 'return_code', 0) != 0:
            raise RuntimeError('could not reassemble chunked verifier artifacts')
    return mount_pairs


def _direct_command(mount_pairs: list[tuple[str, str]], *, artifacts_read_only: bool) -> str:
    """Run the image-owned tests on the verifier runtime's own rootfs."""
    steps: list[str] = ['set -euo pipefail', 'cd /']
    for staged_path, target_path in mount_pairs:
        staged = shlex.quote(staged_path)
        target = shlex.quote(target_path)
        parent = shlex.quote(str(PurePosixPath(target_path).parent))
        steps.append(
            f'rm -rf {target}; mkdir -p {parent}; '
            f'if [ -d {staged} ]; then mkdir -p {target}; cp -a {staged}/. {target}/; '
            f'else cp -a {staged} {target}; fi'
        )
        if artifacts_read_only:
            steps.append(f'chmod -R a-w {target}')
    # run_command does not inherit the image's WORKDIR, unlike `docker run` below.
    steps.append(f'cd {shlex.quote(TESTS_DIR)}')
    steps.append(f'bash {shlex.quote(TEST_SCRIPT)} 2>&1 | tee /logs/verifier/test.txt')
    return '; '.join(steps)


def _dind_command(
    mount_pairs: list[tuple[str, str]],
    *,
    verifier_image: str,
    container_name: str,
    network: str,
    network_access: bool,
    artifacts_read_only: bool,
    env_names: list[str],
) -> str:
    """Run the image-owned tests in a throwaway container on the runtime's inner daemon.

    ``/tests`` is deliberately not mounted: the container must see the tests exactly as the
    audited image built them, including anything the image generates at build time.

    The inner daemon is handed to the tests through its socket, which is what ``verifier_dind``
    buys: an image is free to spend it on nested containers of its own, typically to score the
    submission under isolation only it knows how to configure. Reaching the socket requires
    running inside the audited image, so the submission never gets there on its own.
    """
    mode = 'ro' if artifacts_read_only else 'rw'
    mount_args = [
        '-v /logs/verifier:/logs/verifier',
        f'-v {shlex.quote(f"{DOCKER_SOCKET}:{DOCKER_SOCKET}")}',
    ]
    mount_args += [f'-v {shlex.quote(f"{staged}:{target}:{mode}")}' for staged, target in mount_pairs]
    env_args = ' '.join(f'--env {shlex.quote(name)}' for name in sorted(env_names))
    docker_run = (
        f'docker run --name {shlex.quote(container_name)} --network {shlex.quote(network)} '
        + ' '.join(mount_args)
        + (f' {env_args}' if env_args else '')
        + f' {shlex.quote(verifier_image)} bash -o pipefail -c '
        + shlex.quote(f'bash {TEST_SCRIPT} 2>&1 | tee /logs/verifier/test.txt')
    )
    network_internal = '' if network_access else ' --internal'
    return f'docker network create{network_internal} {shlex.quote(network)} >/dev/null && {docker_run}'


async def _run_verifier_image_impl(
    *,
    agent_runtime_name: str,
    verifier_runtime_name: str,
    sample_id: str,
    verifier_image: str,
    verifier_dind: bool,
    verifier_network_access: bool,
    verifier_artifacts_read_only: bool,
    verifier_mounts: list[Any],
    verifier_allowed_protected_mounts: list[str],
    verifier_env: dict[str, str] | None,
    verifier_env_refs: dict[str, str] | None,
    test_timeout: int,
    docker_timeout: int,
) -> tuple[float, dict[str, Any]]:
    ctx = require_context()
    agent_runtime = await ctx.get_runtime(agent_runtime_name)
    verifier_runtime = await ctx.get_runtime(verifier_runtime_name)
    span_id = get_current_span_id()
    verifier_runtime_id = ctx.get_runtime_id(verifier_runtime_name)
    if span_id and verifier_runtime_id:
        update_span(span_id, attributes={'hytuner.eval_runtime_id': verifier_runtime_id})

    await _require_image_test_script(verifier_runtime)
    if verifier_dind:
        await _pull_docker_image(verifier_runtime, verifier_image, docker_timeout)
    prepared_logs = await verifier_runtime.run_command(
        command='rm -rf /logs/verifier && mkdir -p /logs/verifier',
        timeout=120,
    )
    if getattr(prepared_logs, 'return_code', 0) != 0:
        raise RuntimeError('could not prepare verifier logs')

    mount_pairs = await _stage_verifier_artifacts(
        agent_runtime,
        verifier_runtime,
        verifier_mounts,
        verifier_allowed_protected_mounts,
        sample_id,
    )
    resolved_env = resolve_verifier_env(verifier_env, verifier_env_refs)
    resolved_env.setdefault('PMIX_MCA_gds', 'hash')

    network = 'nexus-verifier-' + _safe_component(sample_id)
    container_name = network + '-runner'
    if verifier_dind:
        command = _dind_command(
            mount_pairs,
            verifier_image=verifier_image,
            container_name=container_name,
            network=network,
            network_access=verifier_network_access,
            artifacts_read_only=verifier_artifacts_read_only,
            env_names=list(resolved_env),
        )
    else:
        command = _direct_command(mount_pairs, artifacts_read_only=verifier_artifacts_read_only)

    started_at = asyncio.get_running_loop().time()
    try:
        result = await verifier_runtime.invoke(
            'run_command',
            {'command': command, 'timeout': test_timeout, 'extra_env': resolved_env},
            request_timeout=test_timeout + 120,
        )
    finally:
        if verifier_dind:
            # The inner daemon serves this evaluation alone, so the sweep also catches whatever
            # the image started for itself. Containers left over from a timeout still hold
            # /logs/verifier read-write and must go before the rewards are read back.
            await verifier_runtime.run_command(
                command=(
                    f'docker rm -f {shlex.quote(container_name)} >/dev/null 2>&1 || true; '
                    'docker ps -aq | xargs -r docker rm -f >/dev/null 2>&1 || true; '
                    f'docker network rm {shlex.quote(network)} >/dev/null 2>&1 || true'
                ),
                timeout=120,
            )
    logger.info(
        'Verifier-image evaluation finished: status=%s return_code=%s',
        getattr(result, 'status', None),
        getattr(result, 'return_code', None),
    )
    status = getattr(result, 'status', None)
    status_value = str(getattr(status, 'value', status or '')).lower()
    return_code = getattr(result, 'return_code', None)
    if status_value not in {'finished', 'completed', 'success', 'succeeded'}:
        raise RuntimeError(f'verifier execution failed: status={status_value!r}, return_code={return_code!r}')

    outputs = await collect_verifier_outputs(verifier_runtime)
    if outputs.reward_text:
        reward = _finite_reward(outputs.reward_text)
    elif outputs.reward_json_text:
        payload = json.loads(outputs.reward_json_text)
        if (
            not isinstance(payload, dict)
            or isinstance(payload.get('reward'), bool)
            or not isinstance(payload.get('reward'), (int, float))
        ):
            raise RuntimeError("verifier reward.json must contain numeric 'reward'")
        reward = _finite_reward(payload['reward'])
    else:
        raise RuntimeError('verifier produced neither reward.txt nor reward.json')

    if return_code != 0:
        logger.info('Verifier returned exit code %s after writing a valid reward; preserving that reward.', return_code)
    duration_seconds = asyncio.get_running_loop().time() - started_at
    return reward, build_verifier_test_results(
        json.loads(outputs.ctrf_text) if outputs.ctrf_text else {},
        reward=reward,
        return_code=return_code,
        status=status,
        duration_seconds=duration_seconds,
        log_tail=outputs.log_tail,
        log_truncated=outputs.log_truncated,
        verifier_result=outputs.result,
    )


async def verifier_image_evaluate(
    *,
    agent_runtime_name: str = 'main',
    verifier_runtime_name: str = 'verifier',
    sample_id: str = '',
    verifier_image: str,
    verifier_dind: bool = False,
    verifier_network_access: bool = False,
    verifier_artifacts_read_only: bool = True,
    verifier_mounts: list[Any] | None = None,
    verifier_allowed_protected_mounts: list[str] | None = None,
    verifier_env: dict[str, str] | None = None,
    verifier_env_refs: dict[str, str] | None = None,
    test_timeout: int = 3600,
    docker_timeout: int = 3600,
) -> tuple[float, dict[str, Any]]:
    """Evaluate declared agent artifacts against the image-owned tests in a trusted runtime.

    The verifier runtime is created from ``verifier_image`` itself, so the trusted tests are
    whatever that image ships at ``/tests``. Nothing is uploaded from the host, which keeps the
    scored tests byte-identical to the audited image and removes any dependency on ``task_dir``
    being reachable from the machine driving the run.

    ``verifier_dind`` selects how those tests are executed: directly on the verifier runtime's
    rootfs, or in a throwaway container started from the same image on the runtime's inner
    docker daemon. The nested form costs a second pull of the image but gives the tests a fresh
    container, kernel-enforced read-only artifact mounts, and that daemon's socket to spend on
    any further isolation the image wants to set up for itself.
    """
    if not verifier_image:
        raise ValueError('verifier_image is required for verifier_image evaluation')

    ctx = require_context()
    component = _safe_component(sample_id or getattr(ctx, 'run_id', '') or 'task')
    _validate_verifier_runtime_topology(
        ctx,
        verifier_runtime_name=verifier_runtime_name,
        verifier_image=verifier_image,
        verifier_dind=verifier_dind,
        verifier_network_access=verifier_network_access,
    )
    verifier_runtime = await ctx.get_runtime(verifier_runtime_name)
    try:
        return await _run_verifier_image_impl(
            agent_runtime_name=agent_runtime_name,
            verifier_runtime_name=verifier_runtime_name,
            sample_id=sample_id or component,
            verifier_image=verifier_image,
            verifier_dind=verifier_dind,
            verifier_network_access=verifier_network_access,
            verifier_artifacts_read_only=verifier_artifacts_read_only,
            verifier_mounts=verifier_mounts or [],
            verifier_allowed_protected_mounts=verifier_allowed_protected_mounts or [],
            verifier_env=verifier_env,
            verifier_env_refs=verifier_env_refs,
            test_timeout=test_timeout,
            docker_timeout=docker_timeout,
        )
    finally:
        # Both runtimes are released right after scoring, so this only reclaims disk early.
        # Raising here would replace whatever the evaluation itself was reporting.
        try:
            cleanup = await verifier_runtime.run_command(
                command=f'rm -rf {shlex.quote(f"{VERIFIER_ARTIFACT_ROOT}/{component}")}',
                timeout=120,
            )
            if getattr(cleanup, 'return_code', 0) != 0:
                logger.warning('Could not remove verifier artifact staging for %s: %s', component, cleanup)
        except Exception:
            logger.warning('Could not remove verifier artifact staging for %s', component, exc_info=True)


__all__ = ['VerifierArtifact', 'verifier_image_evaluate']
