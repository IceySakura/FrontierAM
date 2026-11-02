# /// script
# dependencies = []
#
# [tool.nexus]
# run_hooks = { output_fetch = "nexus.runner.hooks.output_fetch" }
# ///

"""Copy files a task produced *inside* a runtime back to the host.

Why this module exists
----------------------
Every ``ToolType.RUNTIME`` tool executes inside the sandbox, so anything the
model writes there -- e.g. the ``out.pptx`` an image2pptx task is asked to
save -- exists only in the sandbox filesystem.  The provider stops the
sandbox when the run ends (``/sandbox/v3/runtime/stop``), and
``jupyter_with_image``'s ``_scan_new_images`` only carries *images* back as
base64 in the tool response, so a ``.pptx`` is simply lost.

Re-running the model's code on the host is not an acceptable substitute:

* it is a *different* file (a re-render, not the artifact the model produced);
* it executes untrusted model code outside the sandbox -- which is exactly the
  constraint the sandbox exists to enforce, and which has already clobbered
  host data once (a model's ``add_picture(SOURCE)`` overwrote the input
  images in the shared dataset).

This is a runner hook (``RunHook.on_run_output``) so the copy runs inside
``run()`` at the stage that contract names: after the task and its processors
finished, before output callbacks and sandbox hold/release -- so earlier
``on_run_output`` callbacks (e.g. HyTuner's capture) and no hold window are
needed.  Results land in ``RunOutput.hook_results['output_fetch']``.

Example YAML config::

    runner:
      hooks:
        output_fetch:
          name: output_fetch
          required: false
          args:
            dest_dir: /data/out
            globs: ['/tmp/jupyter_work_*/*/out.pptx']

Byte-identity
-------------
The listing step asks the sandbox for each file's ``sha256`` and size, and
those travel in the manifest next to the fetched bytes.  The host re-hashes
what it received and records whether the two agree.  A "file exists and
opens" check would pass for a re-generated file too; a hash agreement is what
establishes that the bytes on the host are the bytes that were in the sandbox.

Only a verified copy may occupy a file's final path: bytes are written to a
sibling temp name first and moved into place (``os.replace``) after the host
re-hash agrees with the sandbox listing.  A mismatched copy is quarantined
under ``<stem>.sha_mismatch<ext>`` instead -- still on disk for inspection,
but impossible to mistake for the artifact by name.

When ``artifact_upload`` is enabled the same identity chain extends over the
artifact service: the hook itself uploads the verified disk copy through the
standard ``nexus.utils.artifact.ArtifactClient`` (the same client the
``gui_recording`` hook already drives directly).  Only a copy whose host
re-hash matched the sandbox sha is shipped; ``ArtifactClient.upload_file``
re-reads and re-hashes the file for its own content check, the upload carries
a per-run idempotency key, and the returned ``ArtifactInfo`` is asserted
against the verified size.  Upload credentials come from the running
environment exactly as for every other ``ArtifactClient`` consumer
(``$ARTIFACT_TAIHU_PAT`` or the local key file, ``$ARTIFACT_SERVICE_URL``);
the Cloud execution ticket is never borrowed.  No step trusts any other
step's word for the bytes.

Consumption
-----------
``fetch_manifest.json`` is the authoritative record of what was fetched and
where each copy landed.  Consumers must take paths from the manifest's
``files[].local_path`` rather than assume the standard basename: the name can
be disambiguated (``out_2.pptx``) when several sandbox files share a basename,
mismatched copies are quarantined (``out.sha_mismatch.pptx``), and a rerun
writing into a ``dest_dir`` that still holds a previous attempt's files never
overwrites them -- the new copy lands on a suffix name.  When
``artifact_upload`` is enabled the durable delivery channel is the artifact
service (the hook summary's ``artifacts`` refs); the on-disk tree is always
written, for inspection and for disk-only setups.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import time

from typing import TYPE_CHECKING, Any

from nexus.runner.hooks.base import RunHook, run_hook_registry
from nexus.utils.artifact import ArtifactClient, ArtifactUploadError, load_taihu_pat


if TYPE_CHECKING:
    from nexus.core.context import ExecutionContext
    from nexus.runner.single import RunOutput


logger = logging.getLogger(__name__)

# Prints one TSV row per match: <path>\t<size>\t<sha256>
#
# Written as a shell loop rather than a Python heredoc on purpose: it needs no
# interpreter to be present in the sandbox (the worker venv path carries a
# random id and cannot be pinned), and the globs are expanded by the same shell
# that `run_command` already uses.  Paths are emitted verbatim and split on the
# first two tabs, so a filename containing spaces or a tab is still handled.
#
# ⛔⛔ The patterns must NOT be quoted.  `for p in "/tmp/work/*/*.pptx"` iterates
#    once over the literal string -- the shell skips pathname expansion inside
#    quotes -- so `[ -f "$p" ]` is false and every fetch silently returns zero
#    files.  That is exactly how the first version of this module failed on
#    2026-10-01 (three runs matched 0 files while reporting no error).  The
#    patterns therefore go in bare, and are charset-validated first so "bare"
#    cannot become a shell-injection hole for a config-supplied glob.
#
# The trailing `; true` keeps the command's exit status clean; because of it a
# non-Finished response (e.g. TimeLimitExceeded) must be detected through
# `response.status`, never through an empty stdout.
_LIST_TEMPLATE = 'for p in {patterns}; do [ -f "$p" ] && printf "%s\\t%s\\t%s\\n" "$p" "$(stat -c%s "$p")" "$(sha256sum "$p" | cut -d" " -f1)"; done; true'

# Everything a glob legitimately needs, and nothing that a shell would act on.
_SAFE_GLOB = re.compile(r'^[A-Za-z0-9_./*?\[\]{}!-]+$')

# hook_results is embedded in the run_result artifact (16 MiB JSON budget), so
# the artifact ref list is capped; the full per-file record stays in the on-disk
# manifest, which has no such limit.
MAX_ARTIFACT_REFS_IN_RESULT = 32

# Default per-file size cap. A fetch briefly holds one file's base64 string
# plus its decoded bytes in memory (peak ~2.3x the file size) in a process
# shared with other runs, and with ``artifact_upload`` every fetched file is
# also pushed through the artifact service, whose transfers share bandwidth
# and server-side quotas with every other artifact consumer.  An accidentally
# wide glob (``**/*``) or a huge sandbox file must not be able to drag the
# worker down: over-limit files are recorded in ``manifest['errors']`` and
# skipped before download.  Real outputs are small (image2pptx decks measured
# 159 KB - 744 KB), so the 64 MiB default still keeps ~100x headroom. ``None``
# (also ``max_file_bytes: null`` in config) disables the cap.
DEFAULT_MAX_FILE_BYTES = 64 * 1024 * 1024
# The artifact service rejects stored names outside 1-255 characters; upload
# names are truncated against this ceiling, not against local filesystem
# limits (which are 255 *bytes* per component on ext4, a different budget).
ARTIFACT_UPLOAD_NAME_LIMIT = 255


def _sanitize(n: str) -> str:
    """Make *n* safe to use as a single path component.

    ``/`` and ``\\`` are flattened. ``.`` and ``..`` would otherwise resolve
    to the current / parent directory once the component is joined into a
    path, escaping ``dest_dir`` -- so they are mapped to underscore-prefixed
    literal names that keep the joined path inside the fetch tree.
    """
    cleaned = n.replace('/', '_').replace('\\', '_')
    if cleaned in ('.', '..'):
        return f'_{cleaned}'
    return cleaned or 'runtime'


def _decode_and_hash(content: str) -> tuple[bytes, str]:
    """b64-decode a downloaded file and hash the decoded bytes.

    Both steps are CPU-bound over the whole payload, so the caller runs them
    in a worker thread (``asyncio.to_thread``): a large fetch must not stall
    the event loop that every other run in the process shares.
    """
    payload = base64.b64decode(content)
    return payload, hashlib.sha256(payload).hexdigest()


def _quarantine_path(out_dir: str, stem: str, ext: str, used: set[str]) -> str:
    """Final path for a mismatched copy: unmistakable, and never colliding.

    The name is not a valid "artifact-looking" basename (``.sha_mismatch``
    inserted before the extension), and it participates in the same claim
    machinery as normal files so a second mismatch lands on
    ``<stem>.sha_mismatch_2<ext>`` instead of overwriting the first.
    """
    path = os.path.join(out_dir, f'{stem}.sha_mismatch{ext}')
    counter = 1
    while path in used or os.path.exists(path):
        counter += 1
        path = os.path.join(out_dir, f'{stem}.sha_mismatch_{counter}{ext}')
    used.add(path)
    return path


class OutputFetchError(RuntimeError):
    """Raised when fetching sandbox outputs failed.

    ``manifest`` carries the (partial) fetch record that was persisted to
    ``fetch_manifest.json`` before the raise, so callers can still report what
    was fetched. The runner's post-run hook channel records this failure
    without touching the task's own state, so raising here reports a lost
    artifact instead of silently swallowing it.
    """

    def __init__(self, message: str, *, manifest: dict[str, Any]) -> None:
        super().__init__(message)
        self.manifest = manifest


def _glob_base(g: str) -> str:
    """Directory prefix ahead of the first path component holding a wildcard.

    ``/tmp/work/*.pptx`` -> ``/tmp/work``; ``foo/bar?.txt`` -> ``foo``; a
    literal path with no wildcard is returned as-is (``ls`` on it still shows
    the file).  Stripping at the *component* level is what makes ``?``/``[``
    globs correct: chopping the string at the wildcard would leave half a
    component (``foo/bar``) behind.
    """
    parts = g.split('/')
    for i, part in enumerate(parts):
        if '*' in part or '?' in part or '[' in part:
            return '/'.join(parts[:i]) or '.'
    return g


async def _run_command(runtime: Any, command: str, timeout: float) -> Any:
    """``run_command`` with a Finished-status check.

    ``run_command`` reports a timeout as ``status=TimeLimitExceeded`` without
    raising, and the listing command deliberately ends in ``; true``.  Without
    this check a slow ``sha256sum`` would parse as an empty listing and be
    recorded as "no file matched" -- pointing the manifest's diagnostics at
    the wrong cause.
    """
    response = await runtime.run_command(command=command, timeout=timeout)
    status = getattr(response, 'status', None)
    if status is not None and getattr(status, 'value', status) != 'Finished':
        raise RuntimeError(
            f'sandbox command status={getattr(status, "value", status)}: {(response.stdout or "")[:200]}'
        )
    return response


async def _list_matches(runtime: Any, globs: list[str], timeout: float) -> tuple[list[tuple[str, int, str]], str, str]:
    """Return ``([(path, size, sha256), ...], command, diagnostic)``.

    *diagnostic* is the base directory listing, captured only when the patterns
    match nothing -- so a silent zero is always explainable after the sandbox is
    gone.
    """
    for g in globs:
        if not _SAFE_GLOB.match(g):
            raise ValueError(f'unsafe glob (would not survive shell expansion): {g!r}')
    patterns = ' '.join(globs)
    command = _LIST_TEMPLATE.format(patterns=patterns)
    response = await _run_command(runtime, command, timeout)
    rows: list[tuple[str, int, str]] = []
    for line in (response.stdout or '').splitlines():
        parts = line.split('\t')
        if len(parts) != 3:
            continue
        path, size, digest = parts
        try:
            rows.append((path, int(size), digest))
        except ValueError:
            continue

    diagnostic = ''
    if not rows:
        # A zero-match is the failure that silently loses the artifact (the
        # sandbox is gone right after this), so capture enough to explain it
        # from the manifest alone: what exists under each pattern's base dir.
        bases = sorted({_glob_base(g) for g in globs})
        names = sorted({os.path.basename(g.rstrip('/')) or '*' for g in globs})
        pattern_args = ' -o '.join(f'-name "{n}"' for n in names)
        if len(names) > 1:
            pattern_args = f'\\( {pattern_args} \\)'
        parts = []
        for base in bases:
            try:
                listing = await _run_command(
                    runtime,
                    f'ls -la "{base}" 2>&1; echo "--- find ---"; find "{base}" -maxdepth 3 {pattern_args} 2>&1',
                    timeout,
                )
                parts.append(f'$ ls {base}\n{(listing.stdout or "").strip()}')
            except Exception as exc:
                parts.append(f'$ ls {base}  # FAILED: {exc}')
        diagnostic = '\n'.join(parts)
        logger.warning('output_fetch: no match for %s; diagnostic:\n%s', globs, diagnostic)
    return rows, command, diagnostic


# Uploads reuse the standard ``nexus.utils.artifact.ArtifactClient``, whose
# own retry policy is the budget for transient failures: control-plane
# requests (via the Taihu/MCP gateway) retry 502/503/504, and pre-signed COS
# transfers retry with a deep exponential backoff.  The hook therefore does
# not add an outer retry loop -- one exhausted ``upload_file`` call is a
# failed upload, which lands in ``manifest['errors']`` and follows the hook's
# required/optional policy like any other fetch error.
def _artifact_credentials_available() -> bool:
    """True when the running environment provides Artifact service credentials.

    Resolution mirrors :class:`ArtifactClient`'s own, and none of the sources
    is the Cloud execution ticket: the Taihu PAT (``$ARTIFACT_TAIHU_PAT`` or
    the local key file ``$ARTIFACT_TAIHU_PAT_FILE`` /
    ``~/.nexus/auth/taihu``), or the dev-only ``$ARTIFACT_DEV_EMPLOYEE``
    identity header honoured by local backends.  Local and Cloud use this
    same hook; a run whose environment provides none of these simply stays
    disk-only (the skip is recorded in the manifest, not raised).
    """
    if load_taihu_pat() is not None:
        return True
    dev_employee = os.getenv('ARTIFACT_DEV_EMPLOYEE')
    return bool(dev_employee and dev_employee.strip())


def _artifact_idempotency_key(*, run_id: str | None, runtime: str, remote_path: str, sandbox_sha256: str) -> str:
    """Fixed-length idempotency key: the server validates 1-128 characters.

    The plaintext form ``task-output:<run_id>:<sha16>-<name>`` grows with both
    the run id and the file name, and a legal-but-long name is rejected
    outright (a 36-char UUID run id plus an 85-char name already exceeds 128).
    Hashing the identity keeps the key constant at 44 characters while staying
    stable across retries and unique per artifact.

    The identity is the same (run, runtime, remote path) triple the upload
    name digest covers, plus content -- NOT the local file name: a retried
    fetch re-disambiguates same-basename downloads locally (``out.pptx`` can
    land as ``out_2.pptx`` on one attempt and ``out_3.pptx`` on the next,
    depending on fetch order), so keying on the local name would break a
    retry's reuse of the same server-side session exactly when a retry
    happens.
    """
    digest = hashlib.sha256(f'{run_id}:{runtime}:{remote_path}:{sandbox_sha256}'.encode()).hexdigest()
    return f'task-output:{digest}'


def _artifact_upload_name(*, run_id: str | None, runtime: str, remote_path: str) -> str:
    """Collision-free upload name inside the service's 255-character limit.

    The artifact service keys stored files by owner + parent folder + name and
    validates 1-255 characters. A basename prefix alone collides in two real
    cases: runs whose ids share their first characters (``item-000000000001``
    and ``item-000000000002`` share the first 8), and one run whose runtimes
    each ship their own ``out.pptx``. The name therefore starts with a digest
    of the full (run id, runtime, remote path) identity -- complete inputs, not
    truncated ids -- so both collisions resolve, and the readable stem comes
    from the remote basename rather than the local disambiguated name, so a
    retried fetch of the same identity reproduces the same stored name even
    when the local ``out_2.pptx`` suffix shifts. The stem is truncated
    (extension kept whole) so a legal 244-character file name plus the digest
    prefix still fits under 255.
    """
    digest = hashlib.sha256(f'{run_id}:{runtime}:{remote_path}'.encode()).hexdigest()[:16]
    stem, ext = os.path.splitext(os.path.basename(remote_path))
    budget = ARTIFACT_UPLOAD_NAME_LIMIT - len(digest) - 1 - len(ext)
    if budget < 1:
        # Pathological only: the extension alone eats the budget. Keep the
        # digest identity, drop the readable stem.
        return f'{digest}{ext}' if ext else digest
    return f'{digest}-{stem[:budget]}{ext}'


async def _upload_fetched_artifact(
    *,
    run_id: str | None,
    runtime: str,
    remote_path: str,
    path: str,
    name: str,
    sandbox_sha256: str,
    size_bytes: int,
    artifact_folder: str | None,
    storage_pool: str | None,
) -> dict[str, Any]:
    """Upload one verified fetched file through the standard ArtifactClient.

    The caller has already proven the on-disk bytes equal the sandbox listing
    (``sha256_match``); ``ArtifactClient.upload_file`` re-opens the file,
    re-hashes it for its content check, and streams it to the artifact
    service.  The idempotency key pins the upload to (run, content, name), so
    a retry after an interrupted run reuses the same server-side session
    instead of creating a duplicate artifact.  The stored name derives from the
    full (run, runtime, remote path) identity, so runs sharing an id prefix and
    multiple runtimes of one run cannot overwrite each other's artifacts.
    Returns the ``ArtifactInfo`` payload (artifact_id, name, size_bytes,
    status) for the manifest; raises on any failure, which the caller records.
    """
    async with ArtifactClient() as client:
        parent_folder_id = None
        if artifact_folder and artifact_folder != '/':
            parent_folder_id = await client.resolve_resource_id(artifact_folder, expect='folder')
        upload_name = _artifact_upload_name(run_id=run_id, runtime=runtime, remote_path=remote_path)
        info = await client.upload_file(
            path,
            name=upload_name,
            parent_folder_id=parent_folder_id,
            storage_pool=storage_pool,
            idempotency_key=_artifact_idempotency_key(
                run_id=run_id, runtime=runtime, remote_path=remote_path, sandbox_sha256=sandbox_sha256
            ),
        )
        if info.size_bytes != size_bytes:
            raise ArtifactUploadError(
                f'artifact service returned size {info.size_bytes} for {name}; expected {size_bytes}'
            )
        return {
            'artifact_id': info.artifact_id,
            'name': info.name,
            'size_bytes': info.size_bytes,
            'status': info.status,
        }


async def fetch_runtime_outputs(
    runtime_provider: Any,
    *,
    dest_dir: str,
    globs: list[str],
    timeout: float = 120.0,
    per_runtime_subdir: bool = False,
    run_id_subdir: bool = False,
    run_id: str | None = None,
    artifact_upload: bool = False,
    artifact_folder: str | None = None,
    storage_pool: str | None = None,
    max_file_bytes: int | None = DEFAULT_MAX_FILE_BYTES,
    max_files: int | None = None,
) -> dict[str, Any]:
    """Copy sandbox files matching ``globs`` into ``dest_dir``.

    Failure contract: the manifest is always persisted first (a partial
    artifact list must stay inspectable once the sandbox is gone), and then
    -- if any error was recorded -- :class:`OutputFetchError` is raised with
    the manifest attached. The hook layer must not swallow fetch failures:
    they belong in the standard hook failure channel (``hook_errors`` /
    ``hook_failures``), where a required/optional policy decides what a lost
    artifact means for the run. The task's own state is unaffected by the
    raise; the runner's post-run hook handling keeps it.

    With ``artifact_upload`` the fetch gains a second, additive backend: after
    a file is verified on disk, it is also uploaded to the artifact service
    through the standard ``ArtifactClient``, driven from this hook.  Uploads
    happen only when *both* hold: ``artifact_upload`` is set and the running
    environment provides Artifact credentials (``$ARTIFACT_TAIHU_PAT`` or the
    local key file; the Cloud execution ticket is never used), plus the
    file's ``sha256_match`` is True (a mismatched copy must not be shipped).
    Otherwise the behaviour is byte-for-byte the disk-only path, and the
    manifest records why via ``artifact_upload.skipped``.  A credential
    source that raises (e.g. a PAT file misconfigured as a directory) also
    lands in disk-only mode -- but as an error recorded after the local
    fetch, not as an early abort.  Upload failures join
    ``manifest['errors']`` -- the same required/optional policy as fetch
    failures decides whether they fail the run.

    Layout written under ``dest_dir`` (plus ``<run_id>/`` when
    ``run_id_subdir`` is set)::

        <dest_dir>/<runtime>/<basename>       fetched bytes (verified copies only)
        <dest_dir>/<stem>.sha_mismatch<ext>   quarantined mismatched copies
        <dest_dir>/fetch_manifest.json         paths, sizes, sha256, verdicts

    Only a copy whose host re-hash matches the sandbox sha occupies its final
    path: bytes land on a ``.fetching`` sibling first and are moved in with
    ``os.replace`` once verified, so neither a corrupted transfer nor a
    partially written file can sit at the standard name.  A mismatch is
    quarantined under ``<stem>.sha_mismatch<ext>`` -- data, not an error, but
    impossible to mistake for the artifact by name.

    Two name collisions are handled explicitly:

    * several sandbox files sharing a basename (``jupyter_with_image`` keeps
      one work dir per ``(run_id, python_path)``): later files are saved with
      a ``_2``, ``_3``... suffix instead of overwriting the first, and each
      record carries ``disambiguated: <final name>``.  The suffix is bumped
      until the *final path* is free -- considering both the other files in
      this fetch (the claim set is shared across runtimes, so with
      ``per_runtime_subdir=False`` runtimes cannot overwrite each other, and
      a generated ``out_2.pptx`` cannot be clobbered by a later literal
      ``out_2.pptx``) and what already sits on disk in ``out_dir`` (earlier
      fetches sharing ``dest_dir``).  The sha256 verdict is computed per
      fetched copy, so the manifest cannot claim a match for bytes that were
      silently replaced on disk.
    * several runs sharing ``dest_dir``: they would also overwrite each
      other's ``fetch_manifest.json``; set ``run_id_subdir`` to give each run
      its own subdirectory (the manifest itself always records ``run_id``).
      A rerun into a ``dest_dir`` that still holds the previous attempt's
      files never deletes or overwrites them -- the new copy lands on a
      suffix name.  This is deliberate (the fetch destroys nothing on disk),
      and it is why consumers must read ``files[].local_path`` from the
      manifest instead of assuming the standard basename.
    """
    started = time.time()
    base_dir = os.path.join(dest_dir, _sanitize(run_id)) if (run_id_subdir and run_id) else dest_dir
    # Probe credentials before any manifest exists, but never let a broken
    # credential source abort the fetch itself: an unreadable PAT file raises
    # out of load_taihu_pat(), and without this guard the run would end with
    # no local files and no manifest -- the sandbox is released right after,
    # so the outputs would be lost for good.  The probe failure is recorded
    # and later joins manifest['errors'] (the same required/optional policy
    # as any other hook error) once the local fetch has completed.
    upload_active = False
    credentials_probe_error: str | None = None
    if artifact_upload:
        try:
            upload_active = _artifact_credentials_available()
        except Exception as exc:
            credentials_probe_error = (
                f'artifact credentials probe failed ({type(exc).__name__}: {exc}); staying disk-only'
            )
            logger.warning('output_fetch: %s', credentials_probe_error)
    artifact_summary: dict[str, Any] = {
        'mode': 'artifact_service' if upload_active else 'disk_only',
        'uploaded': 0,
        'failed': 0,
    }
    if artifact_upload and not upload_active:
        if credentials_probe_error is not None:
            artifact_summary['skipped'] = credentials_probe_error
        else:
            artifact_summary['skipped'] = (
                'artifact_upload requested but no Artifact service credentials in this '
                'environment (ARTIFACT_TAIHU_PAT / key file / ARTIFACT_DEV_EMPLOYEE); '
                'staying disk-only'
            )
        logger.warning('output_fetch: %s', artifact_summary['skipped'])
    manifest: dict[str, Any] = {
        'run_id': run_id,
        'dest_dir': dest_dir,
        'base_dir': base_dir,
        'globs': list(globs),
        'started_at': started,
        'runtimes': [],
        'files': [],
        'errors': [],
        'artifact_upload': artifact_summary,
    }

    def _save() -> None:
        manifest['elapsed_s'] = round(time.time() - started, 3)
        try:
            os.makedirs(base_dir, exist_ok=True)
            with open(os.path.join(base_dir, 'fetch_manifest.json'), 'w', encoding='utf-8') as fh:
                json.dump(manifest, fh, ensure_ascii=False, indent=2)
        except Exception:
            logger.exception('output_fetch: failed to write manifest to %s', base_dir)

    def _finalize() -> dict[str, Any]:
        """Persist the manifest, then raise if the fetch recorded any error.

        Every error recorded along the way (provider enumeration failure,
        rejected glob, listing failure, a file that failed to download, a
        file skipped for exceeding ``max_file_bytes`` / ``max_files``) makes
        the fetch as a whole a failure: the composed hook must get an
        exception so its required/optional policy runs, instead of the
        failure hiding inside ``hook_results`` where standard callers never
        look. A per-file ``sha256_match: False`` verdict is data, not an
        error, and stays in the manifest only -- the mismatched bytes are
        quarantined under ``<stem>.sha_mismatch<ext>``, never at the final
        path.
        """
        _save()
        if manifest['errors']:
            summary = '; '.join(manifest['errors'][:3])
            raise OutputFetchError(
                f'{len(manifest["errors"])} error(s) fetching outputs (run_id={run_id}): {summary}',
                manifest=manifest,
            )
        return manifest

    try:
        active = runtime_provider.get_active_runtimes()
    except Exception as exc:
        manifest['errors'].append(f'get_active_runtimes failed: {type(exc).__name__}: {exc}')
        logger.exception('output_fetch: cannot enumerate runtimes')
        return _finalize()

    if not active:
        manifest['errors'].append('no active runtimes at fetch time; sandbox already released?')
        logger.error('output_fetch: no active runtimes, nothing to fetch (run_id=%s)', run_id)
        return _finalize()

    # Final paths already claimed by *this* fetch.  Shared across runtimes on
    # purpose: with per_runtime_subdir=False every runtime writes the same
    # base_dir, so a per-runtime counter would reset between runtimes and let
    # runtime 2's ``out.pptx`` overwrite runtime 1's while both manifest lines
    # still report ``sha256_match: True``.
    used_paths: set[str] = set()
    # Downloads started this fetch, across runtimes (for max_files).
    fetched_count = 0

    for name, runtime in active.items():
        entry: dict[str, Any] = {'name': name, 'base_url': getattr(runtime, 'base_url', None), 'files': 0}
        manifest['runtimes'].append(entry)
        try:
            rows, command, diagnostic = await _list_matches(runtime, list(globs), timeout)
        except Exception as exc:
            entry['error'] = f'list failed: {type(exc).__name__}: {exc}'
            manifest['errors'].append(f'[{name}] list failed: {exc}')
            logger.exception('output_fetch: listing failed on runtime %s', name)
            continue
        entry['list_command'] = command
        if diagnostic:
            entry['diagnostic'] = diagnostic
        entry['matched'] = len(rows)
        if not rows:
            logger.warning(
                'output_fetch: no file matched %s on runtime %s (run_id=%s)',
                globs,
                name,
                run_id,
            )
            continue

        out_dir = os.path.join(base_dir, _sanitize(name)) if per_runtime_subdir else base_dir
        for remote_path, remote_size, remote_sha in rows:
            record: dict[str, Any] = {
                'runtime': name,
                'remote_path': remote_path,
                'sandbox_size': remote_size,
                'sandbox_sha256': remote_sha,
            }
            try:
                # Inside the per-file try on purpose: an unwritable dest_dir
                # or a full disk must not escape as a raw OSError -- it would
                # skip _finalize() and break the manifest-first-then-raise
                # contract (OutputFetchError.manifest would be lost).  Each
                # file records the failure and continues; _finalize() still
                # persists the manifest before raising.
                os.makedirs(out_dir, exist_ok=True)

                if max_files is not None and fetched_count >= max_files:
                    message = f'skipped: per-fetch max_files={max_files} already downloaded'
                    record['error'] = message
                    manifest['errors'].append(f'[{name}] {remote_path}: {message}')
                    logger.error('output_fetch: %s (run_id=%s)', message, run_id)
                    manifest['files'].append(record)
                    continue

                if max_file_bytes is not None and remote_size > max_file_bytes:
                    message = f'skipped: sandbox size {remote_size} exceeds max_file_bytes={max_file_bytes}'
                    record['error'] = message
                    manifest['errors'].append(f'[{name}] {remote_path}: {message}')
                    logger.error('output_fetch: %s for %s', message, remote_path)
                    manifest['files'].append(record)
                    continue

                from nexus.runtime.executors.models.file import FileItem

                response = await runtime.download_file(files=[FileItem(path=remote_path, encoding='base64')])
                fetched_count += 1
                if response.errors or not response.files:
                    message = response.errors[0].message if response.errors else 'download returned no file'
                    record['error'] = message
                    manifest['errors'].append(f'[{name}] {remote_path}: {message}')
                    logger.error('output_fetch: download failed for %s: %s', remote_path, message)
                    manifest['files'].append(record)
                    continue

                # Decode + re-hash in a worker thread: both are CPU-bound over
                # the whole payload, and the event loop is shared with every
                # other run in this process.
                data, host_sha = await asyncio.to_thread(_decode_and_hash, response.files[0].content or '')
                if max_file_bytes is not None and len(data) > max_file_bytes:
                    # The sandbox file grew between listing and download; the
                    # listing size passed the cap but this copy does not.
                    message = f'skipped: downloaded size {len(data)} exceeds max_file_bytes={max_file_bytes}'
                    record['error'] = message
                    manifest['errors'].append(f'[{name}] {remote_path}: {message}')
                    logger.error('output_fetch: %s for %s', message, remote_path)
                    manifest['files'].append(record)
                    continue
                # Several sandbox paths can share a basename (one work dir per
                # run/python_path); keep every copy instead of letting the
                # last one overwrite the rest.  The claim is on the *final
                # path*, not on a basename counter, for two reasons:
                # * a generated suffix can collide with a real basename in the
                #   same fetch (b's out.pptx -> out_2.pptx, then c's literal
                #   out_2.pptx would land on the same path);
                # * dest_dir may already hold files an earlier fetch wrote
                #   (runs share dest_dir without run_id_subdir).
                # So the suffix keeps bumping until the path is free both in
                # this fetch (used_paths) and on disk (os.path.exists).
                local_name = os.path.basename(remote_path)
                stem, ext = os.path.splitext(local_name)
                local_path = os.path.join(out_dir, local_name)
                counter = 1
                while local_path in used_paths or os.path.exists(local_path):
                    counter += 1
                    local_path = os.path.join(out_dir, f'{stem}_{counter}{ext}')
                if counter > 1:
                    record['disambiguated'] = os.path.basename(local_path)
                    logger.warning(
                        'output_fetch: basename %s already taken; saving %s as %s',
                        local_name,
                        remote_path,
                        os.path.basename(local_path),
                    )
                used_paths.add(local_path)
                # Write to a sibling temp name first and only let a verified
                # copy occupy the final path: bytes that do not match the
                # sandbox sha -- and equally a partially written file -- can
                # never sit at the standard name a consumer reading fetch/ by
                # path would pick up.  A mismatch is quarantined under
                # <stem>.sha_mismatch<ext> instead: still on disk for
                # inspection, impossible to mistake for the artifact.
                tmp_path = f'{local_path}.fetching'
                with open(tmp_path, 'wb') as fh:
                    fh.write(data)
                if host_sha == remote_sha:
                    final_path = local_path
                    os.replace(tmp_path, final_path)
                else:
                    final_path = _quarantine_path(out_dir, stem, ext, used_paths)
                    os.replace(tmp_path, final_path)
                    record['quarantined_as'] = os.path.basename(final_path)
                record.update({
                    'local_path': final_path,
                    'host_size': len(data),
                    'host_sha256': host_sha,
                    # The load-bearing field: False here means the copy is
                    # not the sandbox's file and must not be shipped.
                    'sha256_match': host_sha == remote_sha,
                })
                entry['files'] += 1
                if host_sha != remote_sha:
                    logger.error(
                        'output_fetch: SHA256 MISMATCH for %s -> %s (sandbox=%s host=%s); quarantined',
                        remote_path,
                        final_path,
                        remote_sha,
                        host_sha,
                    )
                else:
                    logger.info('output_fetch: %s -> %s (%d bytes, sha256 ok)', remote_path, final_path, len(data))
                if artifact_upload:
                    if not record.get('sha256_match'):
                        # Not the sandbox's bytes: the upload identity check
                        # would reject this copy anyway, and sha256_match
                        # already flags it as data, not an error.
                        record['artifact_uploaded'] = False
                        record['upload_error'] = 'not uploaded: sandbox/host sha256 mismatch'
                        artifact_summary['failed'] += 1
                    elif upload_active:
                        final_name = record.get('disambiguated') or os.path.basename(remote_path)
                        try:
                            # Upload from disk with the sandbox sha pinned in
                            # the idempotency key: the file is proven to be
                            # both what landed on disk and what the sandbox
                            # listed, and ArtifactClient re-hashes it again
                            # for its own content check.
                            record['artifact'] = await _upload_fetched_artifact(
                                run_id=run_id,
                                runtime=name,
                                remote_path=remote_path,
                                path=final_path,
                                name=final_name,
                                sandbox_sha256=remote_sha,
                                size_bytes=len(data),
                                artifact_folder=artifact_folder,
                                storage_pool=storage_pool,
                            )
                            record['artifact_uploaded'] = True
                            artifact_summary['uploaded'] += 1
                        except Exception as exc:
                            # One exhausted upload (ArtifactClient owns the
                            # transient-retry budget) is a failed upload: it
                            # lands in manifest['errors'] so the hook's
                            # required/optional policy decides what it means
                            # for the run, while the disk copy and manifest
                            # survive.
                            record['artifact_uploaded'] = False
                            record['upload_error'] = f'{type(exc).__name__}: {exc}'
                            artifact_summary['failed'] += 1
                            manifest['errors'].append(f'[{name}] artifact upload {remote_path}: {exc}')
                            logger.error('output_fetch: artifact upload failed for %s: %s', remote_path, exc)
            except Exception as exc:
                record['error'] = f'{type(exc).__name__}: {exc}'
                manifest['errors'].append(f'[{name}] {remote_path}: {exc}')
                logger.exception('output_fetch: fetch failed for %s', remote_path)
            manifest['files'].append(record)

    # Recorded only after the local fetch finished: a broken credential
    # source must never cost the outputs themselves, but it is still an
    # error -- the requested upload backend did not happen -- so it flows
    # through the same errors channel and required/optional policy.
    if credentials_probe_error is not None:
        manifest['errors'].append(f'[artifact_upload] {credentials_probe_error}')
    return _finalize()


@run_hook_registry.register('output_fetch')
class OutputFetchHook(RunHook):
    """Fetch sandbox-produced artifacts to the host before the sandbox dies.

    A plain ``RunHook`` rather than a dedicated ``RunnerConfig`` field: the
    lifecycle contract already guarantees the required timing (task finished,
    sandbox alive, before output callbacks), and the result slot is
    ``RunOutput.hook_results['output_fetch']`` where the composed hook
    collects it.

    Fetch failures propagate instead of being swallowed: the composed hook
    records them in ``RunOutput.hook_errors`` (and, for a required hook, in
    the embedding ``RunHooks.hook_failures``) and applies its required /
    optional policy. The task's own state is not affected -- the runner's
    post-run hook channel keeps it -- so raising reports a lost artifact
    without turning a finished task into a failed one.

    Configuration (``runner.hooks.<key>.args`` in YAML):

    * ``dest_dir`` -- host directory, created if missing.
    * ``globs`` -- shell globs expanded inside the sandbox.
    * ``timeout`` -- per sandbox command, default 120s.
    * ``per_runtime_subdir`` -- file under ``<dest_dir>/<runtime>/``.
    * ``run_id_subdir`` -- also namespace by run when several runs share
      ``dest_dir`` (their manifests would otherwise overwrite each other).
      The online evals pass a per-run ``dest_dir``, so the default is False.
    * ``artifact_upload`` -- also upload verified fetched files to the
      artifact service through the standard ``ArtifactClient``.  The same
      hook serves local and Cloud runs; credentials come from the running
      environment (``$ARTIFACT_TAIHU_PAT`` / key file, plus
      ``$ARTIFACT_SERVICE_URL``), never from the Cloud execution ticket.
      An environment with no Artifact credentials stays disk-only, with the
      skip recorded in the manifest.  Transient retries are owned by
      ``ArtifactClient`` (control-plane 502/503/504 and pre-signed COS
      transfers); one exhausted upload is a failure like any other.
    * ``artifact_folder`` -- folder URI/path under which uploads are placed
      (resolved via ``ArtifactClient``; default no parent folder).
    * ``storage_pool`` -- storage pool hint passed to the upload session.
    * ``max_file_bytes`` -- per-file size cap; a file whose sandbox listing
      size (or, after download, whose decoded size) exceeds it is skipped and
      recorded in the manifest's errors instead of being pulled into memory.
      Default 64 MiB; ``null`` disables.  A fetch holds ~2.3x one file's
      bytes in memory in a process shared with other runs, and each uploaded
      file rides the artifact service's shared bandwidth, so a wide glob must
      not be able to OOM the worker.
    * ``max_files`` -- per-fetch cap on the number of downloaded files
      (across runtimes); excess matches are skipped with an error record.
      Default ``null`` (unlimited).

    Upload failures follow the hook's existing required/optional policy: with
    ``required: false`` they are recorded and never fail the run; with the
    default ``required: true`` they surface through the standard hook failure
    channel.  Set ``required: false`` explicitly when uploads are best-effort.

    Consumers must read fetched paths from ``fetch_manifest.json``
    (``files[].local_path``), never assume the standard basename: names can
    be disambiguated, mismatched copies are quarantined, and a rerun into a
    shared ``dest_dir`` never overwrites what a previous attempt left on disk.
    """

    def __init__(
        self,
        *,
        dest_dir: str,
        globs: list[str],
        timeout: float = 120.0,
        per_runtime_subdir: bool = False,
        run_id_subdir: bool = False,
        artifact_upload: bool = False,
        artifact_folder: str | None = None,
        storage_pool: str | None = None,
        max_file_bytes: int | None = DEFAULT_MAX_FILE_BYTES,
        max_files: int | None = None,
    ) -> None:
        if max_file_bytes is not None and max_file_bytes <= 0:
            raise ValueError('max_file_bytes must be positive (or null to disable)')
        if max_files is not None and max_files <= 0:
            raise ValueError('max_files must be positive (or null to disable)')
        self.dest_dir = dest_dir
        self.globs = list(globs)
        self.timeout = timeout
        self.per_runtime_subdir = per_runtime_subdir
        self.run_id_subdir = run_id_subdir
        self.artifact_upload = artifact_upload
        self.artifact_folder = artifact_folder
        self.storage_pool = storage_pool
        self.max_file_bytes = max_file_bytes
        self.max_files = max_files
        self._result: dict[str, Any] = {}

    async def on_run_output(self, ctx: ExecutionContext, output: RunOutput) -> None:
        # Cache the (partial) fetch record on self._result before any exit so
        # get_result() can still summarize what was fetched, then let every
        # failure propagate: the composed hook owns the required/optional
        # policy and the standard failure channels (hook_errors /
        # hook_failures) that a swallowed return would bypass.
        if ctx.runtime_provider is None:
            self._result = {'errors': ['no runtime provider in context; nothing fetched']}
            raise OutputFetchError('output_fetch: no runtime provider in context', manifest=self._result)
        try:
            self._result = await fetch_runtime_outputs(
                ctx.runtime_provider,
                dest_dir=self.dest_dir,
                globs=self.globs,
                timeout=self.timeout,
                per_runtime_subdir=self.per_runtime_subdir,
                run_id_subdir=self.run_id_subdir,
                run_id=ctx.run_id,
                artifact_upload=self.artifact_upload,
                artifact_folder=self.artifact_folder,
                storage_pool=self.storage_pool,
                max_file_bytes=self.max_file_bytes,
                max_files=self.max_files,
            )
        except Exception as exc:
            partial = getattr(exc, 'manifest', None)
            if isinstance(partial, dict):
                self._result = partial
            else:
                self._result = {'errors': [f'output_fetch failed: {type(exc).__name__}: {exc}']}
                logger.exception('output_fetch hook failed (run_id=%s)', ctx.run_id)
            raise

    def get_result(self) -> dict[str, Any]:
        """Summary for ``RunOutput.hook_results`` (the manifest stays on disk).

        ``artifacts`` carries at most :data:`MAX_ARTIFACT_REFS_IN_RESULT` refs
        (the summary travels inside the run result, which has a hard JSON
        budget); the full per-file record always lives in the on-disk manifest.
        Each entry pairs the artifact service reference (``artifact_id``,
        ``name``, ``size_bytes``, ``status``) with the fetch's own verification
        of the bytes it uploaded (``sha256`` of the sandbox listing and
        ``sha256_match``).  ``name`` is the final saved name and
        ``local_path`` is deliberately omitted: on workers it is a temporary
        disk path with no out-of-run value.
        """
        try:
            files = self._result.get('files', [])
            artifact_records = [f for f in files if f.get('artifact')]
            artifacts = [
                {
                    'artifact_id': f['artifact'].get('artifact_id'),
                    'name': f['artifact'].get('name'),
                    'size_bytes': f['artifact'].get('size_bytes'),
                    'status': f['artifact'].get('status'),
                    'sha256': f.get('sandbox_sha256'),
                    'sha256_match': bool(f.get('sha256_match')),
                    'remote_path': f.get('remote_path'),
                    'disambiguated': f.get('disambiguated'),
                }
                for f in artifact_records[:MAX_ARTIFACT_REFS_IN_RESULT]
            ]
            return {
                'dest_dir': self.dest_dir,
                'n_files': len(files),
                'n_sha256_match': sum(1 for f in files if f.get('sha256_match')),
                'artifact_mode': self._result.get('artifact_upload', {}).get('mode', 'disk_only'),
                'artifacts': artifacts,
                'errors': self._result.get('errors', []),
            }
        except Exception:
            logger.exception('output_fetch hook could not summarize its result')
            return {}


__all__ = ['OutputFetchError', 'OutputFetchHook', 'fetch_runtime_outputs']
