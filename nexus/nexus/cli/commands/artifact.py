"""Artifact service CLI commands for uploading, sharing, and fetching artifacts.

These commands are thin wrappers over :class:`nexus.utils.artifact.client.ArtifactClient`.
Authentication and the server URL follow the same resolution order as the SDK:
explicit flags override environment variables which override the packaged
defaults. The Taihu personal access token is resolved as ``--pat`` >
``ARTIFACT_TAIHU_PAT`` > the local key file (``~/.nexus/auth/taihu``),
so ``nexus artifact login`` can persist it once instead of exporting it for
every command. A dev-only ``--dev-employee`` / ``ARTIFACT_DEV_EMPLOYEE`` header
is also supported for a gateway-less local backend. The server URL comes from
``--server`` or ``ARTIFACT_SERVICE_URL``.

Resources are addressed by human-readable references. Two forms are accepted
everywhere a "path" is expected:

* a bare slash-path (``/models/v1/model.bin``), which always refers to the
  caller's own resources; and
* a full ``artifact:///<owner>/<path>`` URI, which explicitly names the
  owner so a resource shared by another employee can be reached without any
  view flag. Bare ``artifact:///`` (empty path) is reserved for the
  aggregated shared-with-me listing on ``ls`` / ``artifacts`` and is not
  accepted anywhere else.

Listings show names only under a single owner (folders get a trailing ``/``).
The aggregated ``artifact:///`` view prints one full URI per line so two
grantors sharing a same-named file remain distinguishable at a glance; pass
``--show-id`` to also print resource ids, or ``--json`` for the raw payload
augmented with a computed ``uri`` field.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import logging
import sys
import uuid

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import yaml


logger = logging.getLogger(__name__)

T = TypeVar('T')


def _make_client(args: argparse.Namespace) -> Any:
    from nexus.utils.artifact.client import ArtifactClient

    return ArtifactClient(
        base_url=getattr(args, 'server', None),
        taihu_pat=getattr(args, 'pat', None),
        dev_employee=getattr(args, 'dev_employee', None),
    )


def _with_client(args: argparse.Namespace, func: Callable[[Any], Awaitable[T]]) -> T:
    async def _run() -> T:
        async with _make_client(args) as client:
            return await func(client)

    return asyncio.run(_run())


# --- commands ---------------------------------------------------------------


def _format_folder_uri(owner: str, segments: tuple[str, ...] | list[str]) -> str:
    """Build ``artifact:///<owner>/<segments>/`` for use as a listing header."""

    from nexus.utils.artifact.uri import format_artifact_uri

    if not segments:
        return format_artifact_uri(owner, trailing_slash=True)
    return format_artifact_uri(owner, *segments, trailing_slash=True)


def _format_item_uri(owner: str, path_segments: tuple[str, ...] | list[str], item: dict[str, Any]) -> str:
    """Build a URI for one item under an owner's namespace.

    Folders get a trailing ``/`` so users can tell them apart from artifacts
    at a glance.
    """

    from nexus.utils.artifact.uri import format_artifact_uri

    name = str(item.get('name') or '')
    rtype = item.get('resource_type') or ('folder' if item.get('folder_id') else 'artifact')
    trailing = rtype == 'folder'
    return format_artifact_uri(owner, *path_segments, name, trailing_slash=trailing)


def _augment_listing_json(
    items: list[dict[str, Any]], *, owner_context: str | None, path_segments: tuple[str, ...] | list[str] | None
) -> list[dict[str, Any]]:
    """Return a copy of ``items`` with a computed ``uri`` field per row.

    ``owner_context`` is the URI owner when the caller navigated into a
    single owner (or ``None`` for the aggregated ``artifact:///`` view, in
    which case the per-item ``owner_id`` is authoritative).
    """

    augmented: list[dict[str, Any]] = []
    segs = tuple(path_segments or ())
    for item in items:
        row = dict(item)
        owner = owner_context or str(row.get('owner_id') or '')
        if owner and row.get('name'):
            row['uri'] = _format_item_uri(owner, segs, row)
        augmented.append(row)
    return augmented


def artifact_ls(args: argparse.Namespace) -> int:
    """List folders and artifacts.

    With no argument, lists the caller's own top-level resources followed by
    every top-level resource that has been shared with the caller (``view=all``).
    Own items are printed as bare names; items from other owners are printed as
    full ``artifact:///<owner>/...`` URIs so same-named resources from different
    grantors remain distinguishable.

    Pass a bare path to list a subfolder in the caller's own tree, or a URI to
    navigate into another owner's namespace. Three URI shapes have special meaning:

    * ``artifact:///~[/<subfolder>]`` (also accepted as ``artifact://~[/<subfolder>]``,
      two-slash shorthand): self-reference — resolves to the caller's own
      namespace at runtime (no need to know your login in advance).
    * ``artifact:///<owner>/[subfolder]``: navigate into one specific grantor's
      namespace; rows are printed as bare names.
    * ``artifact:///``: aggregated shared-with-me listing only (legacy alias for
      the shared portion of the no-arg view); each row is a full URI.
    """

    from nexus.utils.artifact.uri import SELF_OWNER, format_artifact_uri, parse_artifact_uri

    target = getattr(args, 'target', None)
    aggregate_shared = bool(target) and parse_artifact_uri(target).is_aggregated_shared_view
    is_all_view = not target  # True when no args: server returns view=all

    async def _run(client: Any) -> tuple[Any, str | None, tuple[str, ...] | None]:
        # ``owner_context`` = the URI owner segment when the walk is inside
        # one specific namespace; ``None`` when the listing spans owners
        # (``artifact:///``) or the caller's own top-level (where the owner
        # is implicit).
        if not target:
            my_login = await client.whoami()
            page = await client.list_resources(view='all', limit=args.limit, cursor=args.cursor)
            return page, my_login, ()

        parsed = parse_artifact_uri(target)
        if parsed.is_aggregated_shared_view:
            # ``artifact:///`` -- aggregated top-level shared listing.
            page = await client.list_resources(view='shared', limit=args.limit, cursor=args.cursor)
            return page, None, ()

        if parsed.owner is not None:
            self_login = await client.whoami()
            # ``~`` is the self-reference sentinel: treat as the caller's login.
            effective_owner = self_login if parsed.owner == SELF_OWNER else parsed.owner
            if effective_owner == self_login:
                view = None
                owner_ctx = self_login
            else:
                view = 'shared'
                owner_ctx = effective_owner
        else:
            owner_ctx = await client.whoami()
            effective_owner = owner_ctx
            view = None

        if not parsed.segments:
            page = await client.list_resources(
                parent_folder_id=None,
                view=view,
                limit=args.limit,
                cursor=args.cursor,
            )
            # For a filtered top-level shared listing, apply the owner filter
            # client-side so it matches what resolve_path would see. The
            # server doesn't accept an owner filter, so this is post-hoc.
            if view == 'shared' and effective_owner:
                page.items[:] = [it for it in page.items if str(it.get('owner_id') or '') == effective_owner]
            return page, owner_ctx, parsed.segments

        # Non-empty path: descend to that folder id and list its children.
        # If ``~`` was used, rewrite the URI with the actual login before handing
        # it to resolve_resource_id (the server does not know about ``~``).
        resolve_target = format_artifact_uri(self_login, *parsed.segments) if parsed.owner == SELF_OWNER else target
        parent_id = await client.resolve_resource_id(resolve_target, expect='folder')
        page = await client.list_resources(
            parent_folder_id=parent_id,
            view=None,
            limit=args.limit,
            cursor=args.cursor,
        )
        return page, owner_ctx, parsed.segments

    try:
        page, owner_ctx, path_segments = _with_client(args, _run)
    except Exception as exc:
        _report_error(args, exc)
        return 1

    if args.json:
        # For view=all (no target), each item carries its own owner_id; pass
        # owner_context=None so _augment_listing_json uses per-item owner_id
        # to build URIs (same as the aggregate artifact:/// view).
        json_owner_ctx = None if is_all_view else owner_ctx
        _print_payload(
            {
                'items': _augment_listing_json(
                    list(page.items), owner_context=json_owner_ctx, path_segments=path_segments
                ),
                'next_cursor': page.next_cursor,
            },
            as_json=True,
        )
        return 0

    show_id = getattr(args, 'show_id', False)
    for item in page.items:
        rtype = item.get('resource_type') or ('folder' if item.get('folder_id') else 'artifact')
        name = item.get('name') or ''
        row_owner = str(item.get('owner_id') or '')
        if aggregate_shared:
            # artifact:/// view: always show full URI
            display = _format_item_uri(row_owner, (), item) if row_owner else name
        elif is_all_view and row_owner and row_owner != owner_ctx:
            # view=all: item belongs to another owner → show as URI so the caller
            # can tell it apart from a same-named resource in their own tree.
            display = _format_item_uri(row_owner, (), item)
        else:
            display = f'{name}/' if rtype == 'folder' else name
        if show_id:
            rid = item.get('resource_id') or item.get('folder_id') or item.get('artifact_id') or ''
            print(f'{display}\t{rid}')
        else:
            print(display)
    if page.next_cursor:
        print(f'# next_cursor: {page.next_cursor}')
    return 0


def artifact_artifacts(args: argparse.Namespace) -> int:
    """Flat listing across folders. Owned by default; ``artifact:///`` for shared."""

    from nexus.utils.artifact.uri import parse_artifact_uri

    target = getattr(args, 'target', None)
    aggregate_shared = False
    view = 'owned'
    if target:
        parsed = parse_artifact_uri(target)
        if parsed.is_aggregated_shared_view:
            aggregate_shared = True
            view = 'shared'
        else:
            _report_error(
                args,
                ValueError(
                    '`artifacts` accepts no target, or artifact:/// for the aggregated '
                    'shared-with-me listing. Use `ls` to navigate into a specific owner '
                    'or subfolder.'
                ),
            )
            return 1

    try:
        page = _with_client(
            args,
            lambda c: c.list_artifacts(view=view, limit=args.limit, cursor=args.cursor),
        )
    except Exception as exc:
        _report_error(args, exc)
        return 1

    if args.json:
        # ``artifacts`` is a flat listing across every folder, and the rows do
        # not carry their ancestor path. We therefore cannot build a resolvable
        # ``artifact:///<owner>/<path>`` URI here: forcing an empty path would
        # fabricate a top-level URI (``artifact:///<owner>/<name>``) that may
        # not exist, or worse, silently resolve to a different same-named file
        # that really does sit at the top level. Emit the raw rows unchanged;
        # each already carries ``artifact_id``, a stable ID-bearing reference
        # usable with the ``/v1/artifacts/{id}`` endpoints.
        _print_payload(
            {
                'items': list(page.items),
                'next_cursor': page.next_cursor,
            },
            as_json=True,
        )
        return 0

    show_id = getattr(args, 'show_id', False)
    for item in page.items:
        name = item.get('name') or ''
        size = item.get('size_bytes')
        status = item.get('status') or ''
        size_col = size if size is not None else '-'
        if aggregate_shared:
            row_owner = str(item.get('owner_id') or '')
            display = _format_item_uri(row_owner, (), item) if row_owner else name
        else:
            display = name
        if show_id:
            aid = item.get('artifact_id') or ''
            print(f'{display}\t{status}\t{size_col}\t{aid}')
        else:
            print(f'{display}\t{status}\t{size_col}')
    if page.next_cursor:
        print(f'# next_cursor: {page.next_cursor}')
    return 0


def artifact_mkdir(args: argparse.Namespace) -> int:
    async def _run(client: Any) -> Any:
        # ``path`` is the full folder path to create; its parent must exist.
        parent_id = await client.resolve_parent_folder_id(args.path)
        name = client.leaf_name(args.path)
        return await client.create_folder(name, parent_folder_id=parent_id)

    try:
        folder = _with_client(args, _run)
    except Exception as exc:
        _report_error(args, exc)
        return 1
    if args.json:
        _print_payload(folder.raw or {'folder_id': folder.folder_id}, as_json=True)
    else:
        print(args.path)
    return 0


def artifact_rmdir(args: argparse.Namespace) -> int:
    async def _run(client: Any) -> None:
        folder_id = await client.resolve_resource_id(args.path, expect='folder')
        await client.delete_folder(folder_id)

    try:
        _with_client(args, _run)
    except Exception as exc:
        _report_error(args, exc)
        return 1
    logger.info('Folder deleted: %s', args.path)
    return 0


def artifact_upload(args: argparse.Namespace) -> int:
    return _do_upload(args, multipart=None, concurrency=None)


def artifact_upload_large(args: argparse.Namespace) -> int:
    return _do_upload(args, multipart=True, concurrency=args.concurrency)


def _do_upload(args: argparse.Namespace, *, multipart: bool | None, concurrency: int | None) -> int:
    async def _run(client: Any) -> Any:
        # --folder is a destination folder path (default: root).
        parent_id = None
        if getattr(args, 'folder', None):
            parent_id = await client.resolve_resource_id(args.folder, expect='folder')
        kwargs: dict[str, Any] = {
            'name': args.name,
            'parent_folder_id': parent_id,
            'content_type': args.content_type,
            'storage_pool': args.pool,
        }
        if multipart is not None:
            kwargs['multipart'] = multipart
        if concurrency is not None:
            kwargs['max_concurrency'] = concurrency
        return await client.upload_file(args.path, **kwargs)

    try:
        info = _with_client(args, _run)
    except Exception as exc:
        _report_error(args, exc)
        return 1
    if args.json:
        _print_payload(info.raw or {'artifact_id': info.artifact_id}, as_json=True)
    else:
        label = 'multipart' if multipart else 'single'
        logger.info(
            'Uploaded (%s) %s -> %s (%s bytes, status=%s)',
            label,
            args.path,
            info.name or info.artifact_id,
            info.size_bytes,
            info.status,
        )
        print(info.name or info.artifact_id)
    return 0


def artifact_publish(args: argparse.Namespace) -> int:
    """Publish every file under a local directory into a fresh artifact folder.

    Thin orchestration over``create_folder`` + ``upload_file`` (+ optional
    ``put_grant``): each invocation creates a brand-new, uniquely named folder
    (``<label>-<UTC timestamp>-<short random>``) so every run is a separate
    snapshot of the task artifacts. The folder is private to its creator by default (server-side
    ACL);``--grant`` shares read access with colleagues, inherited by every
    uploaded file.
    """

    local_dir = Path(args.local_dir)
    if not local_dir.exists():
        logger.error('Local path does not exist: %s', local_dir)
        return 1
    if not local_dir.is_dir():
        logger.error('Not a directory: %s', local_dir)
        return 1

    scanned = _scan_files(local_dir)
    if not scanned:
        logger.error('No files to publish under: %s', local_dir)
        return 1

    # The artifact service rejects 0-byte uploads ("size_bytes must be at least
    # 1"), so skip empty files with a warning instead of failing the whole
    # publish on the first one.
    files: list[tuple[Path, Path]] = []
    skipped_empty: list[str] = []
    for abs_path, rel in scanned:
        try:
            size = abs_path.stat().st_size
        except OSError:
            size = 0
        if size > 0:
            files.append((abs_path, rel))
        else:
            skipped_empty.append(rel.as_posix())
    for rel in skipped_empty:
        logger.warning('Skipping empty file (0 bytes; artifact service rejects empty uploads): %s', rel)
    if not files:
        logger.error('Nothing to publish: every file under %s is empty', local_dir)
        return 1

    grantees = _parse_grantees(getattr(args, 'grant', None))
    now = datetime.now(timezone.utc)
    folder_name = _build_folder_name(local_dir, getattr(args, 'task', None), now)

    async def _run(client: Any) -> tuple[Any, list[tuple[str, Any]], list[str]]:
        # Place every publish under /tasks/YYYY/MM/DD/ so tens of thousands of
        # runs don't pile up at the root.  The date folders are shared across
        # runs on the same day and are created idempotently.
        date_folder_id = await _ensure_tasks_date_folder(client, now)
        top = await client.create_folder(folder_name, parent_folder_id=date_folder_id)
        # Map an already-created relative directory to its folder id so shared
        # ancestors are created once. The root maps to the top folder.
        dir_cache: dict[tuple[str, ...], str] = {(): top.folder_id}
        uploaded: list[tuple[str, Any]] = []
        for abs_path, rel in files:
            parent_id = await _ensure_dir_folder(client, dir_cache, rel.parent)
            # follow_symlinks=False: reject a symlink swapped in at the leaf
            # between the scan above and this open, so it cannot redirect the
            # read to a file outside the publish root.
            info = await client.upload_file(
                str(abs_path), parent_folder_id=parent_id, name=rel.name, follow_symlinks=False
            )
            uploaded.append((rel.as_posix(), info))
        for employee in grantees:
            await client.put_grant(top.folder_id, employee, permission='read')
        return top, uploaded, grantees

    try:
        top, uploaded, granted = _with_client(args, _run)
    except Exception as exc:
        _report_error(args, exc)
        return 1

    folder_path = f'/tasks/{now.strftime("%Y/%m/%d")}/{folder_name}'
    if args.json:
        _print_payload(
            {
                'folder_path': folder_path,
                'folder': top.raw or {'folder_id': top.folder_id, 'name': folder_name},
                'uploaded': [
                    {
                        'path': rel,
                        'artifact_id': info.artifact_id,
                        'name': info.name,
                        'size_bytes': info.size_bytes,
                        'status': info.status,
                    }
                    for rel, info in uploaded
                ],
                'grants': [{'grantee_id': employee, 'permission': 'read'} for employee in granted],
                'skipped_empty': skipped_empty,
            },
            as_json=True,
        )
        return 0
    print(folder_path)
    for rel, _info in uploaded:
        print(f'  {rel}')
    logger.info('Published %d file(s) to %s', len(uploaded), folder_path)
    if skipped_empty:
        logger.info('Skipped %d empty file(s)', len(skipped_empty))
    for employee in granted:
        logger.info('Granted read on %s to employee:%s', folder_path, employee)
    return 0


def artifact_download(args: argparse.Namespace) -> int:
    """Download an artifact, or an entire folder tree, to the local filesystem.

    When ``args.path`` resolves to an artifact the command behaves as before:
    the file goes to ``-o`` (defaulting to the artifact's own name in the
    current directory). When it resolves to a folder, the command switches
    to recursive mode and mirrors the folder's contents into ``-o`` (which
    defaults to a directory named after the folder, in the current
    directory). Individual failures inside a recursive walk are collected
    rather than aborting the whole run; the summary is logged at the end
    and the exit code reflects any failures.
    """

    async def _run(client: Any) -> dict[str, Any]:
        item = await client.resolve_path(args.path)
        rtype = item.get('resource_type') or ('folder' if item.get('folder_id') else 'artifact')

        if rtype == 'folder':
            folder_id = item.get('folder_id') or item.get('resource_id')
            folder_name = str(item.get('name') or 'artifact-folder')
            # ``-o`` defaults to a directory named after the folder; users
            # who want the tree merged into an existing directory can pass
            # ``-o some/existing/dir`` explicitly.
            dest = Path(args.output) if args.output else Path.cwd() / folder_name
            successes, failures = await client.download_folder_to_dir(
                folder_id,
                dest,
                concurrency=max(1, args.concurrency),
            )
            return {
                'type': 'folder',
                'dest': str(dest),
                'total': successes + len(failures),
                'ok': successes,
                'failed': len(failures),
                'failures': [
                    {
                        'artifact_id': f.artifact_id,
                        'relative_path': f.relative_path,
                        'error': f.error,
                    }
                    for f in failures
                ],
            }

        # Artifact: preserve the previous single-file behaviour.
        artifact_id = item.get('artifact_id') or item.get('resource_id')
        target = args.output
        if not target:
            target = str(item.get('name') or client.leaf_name(args.path))
        path = await client.download_to_file(artifact_id, target)
        return {'type': 'artifact', 'path': str(path)}

    try:
        result = _with_client(args, _run)
    except Exception as exc:
        _report_error(args, exc)
        return 1

    if result['type'] == 'artifact':
        if args.json:
            _print_payload({'path': result['path']}, as_json=True)
        else:
            logger.info('Downloaded %s -> %s', args.path, result['path'])
            print(result['path'])
        return 0

    # Folder branch. Log a compact summary and enumerate failures on stderr.
    if args.json:
        _print_payload(result, as_json=True)
    else:
        logger.info(
            'Downloaded folder %s -> %s (%d/%d files, %d failed)',
            args.path,
            result['dest'],
            result['ok'],
            result['total'],
            result['failed'],
        )
        for failure in result['failures']:
            logger.error(
                'download failed: %s (artifact_id=%s): %s',
                failure['relative_path'],
                failure['artifact_id'],
                failure['error'],
            )
        print(result['dest'])
    return 1 if result['failed'] else 0


def artifact_url(args: argparse.Namespace) -> int:
    async def _run(client: Any) -> Any:
        artifact_id = await client.resolve_resource_id(args.path, expect='artifact')
        return await client.create_download_url(artifact_id)

    try:
        download = _with_client(args, _run)
    except Exception as exc:
        _report_error(args, exc)
        return 1
    if args.json:
        _print_payload(download.raw or {'url': download.url}, as_json=True)
    else:
        print(download.url)
    return 0


def artifact_info(args: argparse.Namespace) -> int:
    async def _run(client: Any) -> Any:
        artifact_id = await client.resolve_resource_id(args.path, expect='artifact')
        return await client.get_artifact(artifact_id)

    try:
        info = _with_client(args, _run)
    except Exception as exc:
        _report_error(args, exc)
        return 1
    _print_payload(info.raw or {'artifact_id': info.artifact_id}, as_json=args.json)
    return 0


def artifact_rm(args: argparse.Namespace) -> int:
    from nexus.utils.artifact.errors import ArtifactPathError

    async def _run(client: Any) -> None:
        # ``rm`` only ever touches the caller's own resources (the server
        # rejects deletes on grantee-side paths), so we short-circuit any URI
        # naming another owner rather than let it fail with a confusing 404.
        self_login = await client.whoami()
        if client.uri_targets_other_owner(args.path, self_login):
            raise ArtifactPathError(
                f'{args.path!r} names a resource owned by another employee; '
                f'rm only applies to your own artifacts. Ask the owner to delete it.',
            )
        artifact_id = await client.resolve_resource_id(args.path, expect='artifact')
        await client.delete_artifact(artifact_id)

    try:
        _with_client(args, _run)
    except Exception as exc:
        _report_error(args, exc)
        return 1
    logger.info('Artifact delete requested: %s', args.path)
    return 0


def artifact_grants(args: argparse.Namespace) -> int:
    async def _run(client: Any) -> Any:
        resource_id = await client.resolve_resource_id(args.path)
        return await client.list_grants(resource_id)

    try:
        grants = _with_client(args, _run)
    except Exception as exc:
        _report_error(args, exc)
        return 1
    if args.json:
        _print_payload({'items': [g.raw for g in grants]}, as_json=True)
        return 0
    for grant in grants:
        suffix = f' (inherited_from {grant.inherited_from})' if grant.inherited_from else ''
        expires = f' expires={grant.expires_at}' if grant.expires_at else ''
        print(f'{grant.permission}\t{grant.grantee_type}:{grant.grantee_id}{expires}{suffix}')
    return 0


def artifact_grant(args: argparse.Namespace) -> int:
    async def _run(client: Any) -> None:
        resource_id = await client.resolve_resource_id(args.path)
        await client.put_grant(
            resource_id,
            args.grantee_id,
            grantee_type=args.type,
            permission=args.permission,
            expires_at=args.expires,
        )

    try:
        _with_client(args, _run)
    except Exception as exc:
        _report_error(args, exc)
        return 1
    if args.json:
        _print_payload(
            {
                'status': 'granted',
                'path': args.path,
                'grantee_id': args.grantee_id,
                'grantee_type': args.type,
                'permission': args.permission,
                'expires_at': args.expires,
            },
            as_json=True,
        )
    else:
        logger.info(
            'Granted %s on %s to %s:%s',
            args.permission,
            args.path,
            args.type,
            args.grantee_id,
        )
    return 0


def artifact_revoke(args: argparse.Namespace) -> int:
    async def _run(client: Any) -> None:
        resource_id = await client.resolve_resource_id(args.path)
        await client.delete_grant(
            resource_id,
            args.grantee_id,
            grantee_type=args.type,
            permission=args.permission,
        )

    try:
        _with_client(args, _run)
    except Exception as exc:
        _report_error(args, exc)
        return 1
    logger.info(
        'Revoked %s on %s from %s:%s',
        args.permission,
        args.path,
        args.type,
        args.grantee_id,
    )
    return 0


def artifact_login(args: argparse.Namespace) -> int:
    from nexus.utils.artifact.auth import save_taihu_pat

    pat = args.pat
    if not pat:
        try:
            pat = getpass.getpass('Taihu PAT: ')
        except (EOFError, KeyboardInterrupt):
            print('\nAborted.', file=sys.stderr)
            return 1
    if not pat or not pat.strip():
        print('No PAT provided.', file=sys.stderr)
        return 1
    try:
        path = save_taihu_pat(pat)
    except (ValueError, OSError) as exc:
        _report_error(args, exc)
        return 1
    print(f'Saved Taihu PAT to {path}')
    return 0


def artifact_logout(args: argparse.Namespace) -> int:
    from nexus.utils.artifact.auth import clear_taihu_pat, taihu_pat_file

    path = taihu_pat_file()
    try:
        removed = clear_taihu_pat()
    except OSError as exc:
        _report_error(args, exc)
        return 1
    if removed:
        print(f'Removed saved Taihu PAT at {path}')
    else:
        print(f'No saved Taihu PAT at {path}')
    return 0


def handle_artifact_command(args: argparse.Namespace) -> int:
    dispatch: dict[str, Callable[[argparse.Namespace], int]] = {
        'login': artifact_login,
        'logout': artifact_logout,
        'whoami': artifact_whoami,
        'ls': artifact_ls,
        'artifacts': artifact_artifacts,
        'mkdir': artifact_mkdir,
        'rmdir': artifact_rmdir,
        'upload': artifact_upload,
        'upload-large': artifact_upload_large,
        'publish': artifact_publish,
        'download': artifact_download,
        'url': artifact_url,
        'info': artifact_info,
        'rm': artifact_rm,
        'grants': artifact_grants,
        'grant': artifact_grant,
        'revoke': artifact_revoke,
    }
    handler = dispatch.get(args.artifact_command)
    if handler is None:
        return artifact_help(args)
    return handler(args)


# --- parser -----------------------------------------------------------------


def register_artifact_parser(subparsers: argparse._SubParsersAction) -> None:
    artifact_parser = subparsers.add_parser(
        'artifact',
        help='Upload, share, and fetch artifacts from the artifact service',
        description='Commands for the Nexus artifact service (folders, artifacts, grants)',
    )
    artifact_subparsers = artifact_parser.add_subparsers(dest='artifact_command', help='Artifact subcommands')

    client_parent = argparse.ArgumentParser(add_help=False)
    _add_client_args(client_parent)

    login_parser = artifact_subparsers.add_parser(
        'login',
        help='Save a Taihu PAT to ~/.nexus/auth/taihu for later commands',
        description=(
            'Persist a Taihu personal access token locally so subsequent commands '
            'do not need --pat or $ARTIFACT_TAIHU_PAT. Prompts securely when --pat '
            'is omitted. The file is written with owner-only (0600) permissions.'
        ),
    )
    login_parser.add_argument(
        '--pat',
        help='Taihu PAT to save; omit to be prompted securely (recommended to avoid shell history)',
    )

    artifact_subparsers.add_parser(
        'logout',
        help='Remove the saved Taihu PAT (~/.nexus/auth/taihu)',
    )

    whoami_parser = artifact_subparsers.add_parser(
        'whoami',
        parents=[client_parent],
        help='Print the acting employee login (calls GET /v1/whoami)',
    )
    whoami_parser.add_argument('--json', action='store_true', help='Output JSON')

    ls_parser = artifact_subparsers.add_parser(
        'ls',
        parents=[client_parent],
        help='List folders and artifacts',
        description=(
            "With no argument, lists the caller's own top-level resources followed by "
            'every top-level resource shared with the caller (view=all): own items are '
            'shown as bare names, shared items as artifact:///<owner>/... URIs. '
            'artifact:///~ (or artifact:///~/<subfolder>) is a self-reference that resolves '
            "to the caller's own namespace at runtime—useful in scripts where the login "
            "isn't known in advance; the two-slash shorthand artifact://~[/<subfolder>] is "
            "also accepted. A bare path lists a subfolder in the caller's own tree. "
            'artifact:///<owner>/ or artifact:///<owner>/<subfolder> navigates into '
            "another owner's shared namespace. The reserved URI artifact:/// shows "
            'only the shared-with-me top level, one URI per line (legacy alias).'
        ),
    )
    ls_parser.add_argument(
        'target',
        nargs='?',
        help='Optional path or artifact:/// URI (default: your own top-level)',
    )
    ls_parser.add_argument('--limit', type=int, default=50, help='Maximum items per page; default: 50')
    ls_parser.add_argument('--cursor', help='Pagination cursor from a previous next_cursor')
    ls_parser.add_argument('--show-id', dest='show_id', action='store_true', help='Also print resource ids')
    ls_parser.add_argument('--json', action='store_true', help='Output JSON')

    artifacts_parser = artifact_subparsers.add_parser(
        'artifacts',
        parents=[client_parent],
        help='List artifacts (flat, across folders)',
        description=(
            "Flat listing across folders. With no target, lists the caller's own "
            'artifacts. Pass artifact:/// for the aggregated shared-with-me '
            'listing (each row printed as a full URI so same-named files from '
            'different grantors remain distinct). To navigate into a specific owner '
            'or subfolder use `ls` instead.'
        ),
    )
    artifacts_parser.add_argument(
        'target',
        nargs='?',
        help='Optional artifact:/// URI (default: your own artifacts). Only artifact:/// is accepted here.',
    )
    artifacts_parser.add_argument('--limit', type=int, default=50, help='Maximum items per page; default: 50')
    artifacts_parser.add_argument('--cursor', help='Pagination cursor from a previous next_cursor')
    artifacts_parser.add_argument('--show-id', dest='show_id', action='store_true', help='Also print artifact ids')
    artifacts_parser.add_argument('--json', action='store_true', help='Output JSON')

    mkdir_parser = artifact_subparsers.add_parser('mkdir', parents=[client_parent], help='Create a folder')
    mkdir_parser.add_argument('path', help='Folder path to create (for example /models or /models/v1)')
    mkdir_parser.add_argument('--json', action='store_true', help='Output JSON')

    rmdir_parser = artifact_subparsers.add_parser('rmdir', parents=[client_parent], help='Delete an empty folder')
    rmdir_parser.add_argument('path', help='Folder path to delete (for example /models)')

    upload_parser = artifact_subparsers.add_parser('upload', parents=[client_parent], help='Upload a local file')
    upload_parser.add_argument('path', help='Local file path')
    upload_parser.add_argument('--folder', help='Destination folder path (for example /models); default is the root')
    upload_parser.add_argument('--name', help='Artifact name; default is the local file name')
    upload_parser.add_argument('--content-type', dest='content_type', help='Override the guessed content type')
    upload_parser.add_argument('--pool', help='Storage pool id; default is the server default pool')
    upload_parser.add_argument('--json', action='store_true', help='Output JSON')

    upload_large_parser = artifact_subparsers.add_parser(
        'upload-large',
        parents=[client_parent],
        help='Upload a large local file via chunked multipart pre-signed URLs',
    )
    upload_large_parser.add_argument('path', help='Local file path')
    upload_large_parser.add_argument(
        '--folder', help='Destination folder path (for example /models); default is the root'
    )
    upload_large_parser.add_argument('--name', help='Artifact name; default is the local file name')
    upload_large_parser.add_argument('--content-type', dest='content_type', help='Override the guessed content type')
    upload_large_parser.add_argument('--pool', help='Storage pool id; default is the server default pool')
    upload_large_parser.add_argument(
        '--concurrency', type=int, default=5, help='Number of parts uploaded in parallel; default: 5'
    )
    upload_large_parser.add_argument('--json', action='store_true', help='Output JSON')

    publish_parser = artifact_subparsers.add_parser(
        'publish',
        parents=[client_parent],
        help='Publish a local directory of task artifacts into a new folder',
        description=(
            'Create a new folder and upload every file under LOCAL_DIR into it, recreating any '
            'subdirectories. The folder is private to you by default (server-side ACL); pass '
            '--grant to also share read access with colleagues (inherited by every uploaded file). '
            'Each run creates a fresh, uniquely named folder (<label>-<UTC timestamp>-<short random>), '
            'so every run is a separate snapshot of the artifacts.'
        ),
    )
    publish_parser.add_argument('local_dir', help='Local directory whose files are the task artifacts')
    publish_parser.add_argument(
        '--task',
        help='Human-readable label used for the folder name; default is the directory name. '
        'A UTC timestamp and a short random suffix are always appended so each run is a new folder.',
    )
    publish_parser.add_argument(
        '--grant',
        help='Comma-separated employee LoginNames to grant read access (inherited by all uploaded files)',
    )
    publish_parser.add_argument('--json', action='store_true', help='Output JSON')

    _path_or_uri_help = (
        'Artifact path or URI: bare path (e.g. /models/model.bin) for your own resources, '
        'or artifact:///<owner>/<path> for a resource shared with you.'
    )

    download_parser = artifact_subparsers.add_parser(
        'download',
        parents=[client_parent],
        help='Download an artifact, or recursively download a folder tree',
        description=(
            'When the path/URI resolves to an artifact the file is written to `-o` '
            '(default: the artifact name in the current directory). When it resolves '
            "to a folder the command recursively mirrors the folder's contents into "
            '`-o` (default: a directory named after the folder, in the current '
            'directory). Downloads inside a folder run concurrently; individual '
            'failures do not abort the walk but produce a non-zero exit code.'
        ),
    )
    download_parser.add_argument('path', help=_path_or_uri_help)
    download_parser.add_argument(
        '-o',
        '--output',
        help='Output path. For an artifact: the local file name (default: artifact name). '
        'For a folder: the destination directory (default: `./<folder-name>`).',
    )
    download_parser.add_argument(
        '--concurrency',
        type=int,
        default=4,
        help='Max simultaneous file downloads during recursive folder mirroring (default: 4). '
        'Ignored when downloading a single artifact.',
    )
    download_parser.add_argument('--json', action='store_true', help='Output JSON')

    url_parser = artifact_subparsers.add_parser('url', parents=[client_parent], help='Print a pre-signed download URL')
    url_parser.add_argument('path', help=_path_or_uri_help)
    url_parser.add_argument('--json', action='store_true', help='Output JSON')

    info_parser = artifact_subparsers.add_parser('info', parents=[client_parent], help='Show artifact metadata')
    info_parser.add_argument('path', help=_path_or_uri_help)
    info_parser.add_argument('--json', action='store_true', help='Output JSON')

    rm_parser = artifact_subparsers.add_parser('rm', parents=[client_parent], help='Delete an artifact')
    rm_parser.add_argument(
        'path',
        help="Artifact path to delete (for example /models/model.bin). Only the caller's own artifacts can be deleted.",
    )

    grants_parser = artifact_subparsers.add_parser('grants', parents=[client_parent], help='List grants on a resource')
    grants_parser.add_argument('path', help='Folder or artifact path')
    grants_parser.add_argument('--json', action='store_true', help='Output JSON')

    grant_parser = artifact_subparsers.add_parser(
        'grant', parents=[client_parent], help='Grant a permission on a resource'
    )
    _add_grant_target_args(grant_parser)
    grant_parser.add_argument('--expires', help='Optional ISO-8601 expiry, for example 2026-12-31T23:59:59')
    grant_parser.add_argument('--json', action='store_true', help='Output JSON')

    revoke_parser = artifact_subparsers.add_parser(
        'revoke', parents=[client_parent], help='Revoke a permission on a resource'
    )
    _add_grant_target_args(revoke_parser)

    artifact_subparsers.add_parser('help', help='Show artifact command help')


def artifact_whoami(args: argparse.Namespace) -> int:
    """Print the acting employee login (uses ``GET /v1/whoami``).

    Useful when composing ``artifact:///<owner>/...`` URIs: run once to
    confirm which identity the current PAT / dev-employee header maps to.
    """

    async def _run(client: Any) -> str:
        return await client.whoami()

    try:
        login = _with_client(args, _run)
    except Exception as exc:
        _report_error(args, exc)
        return 1
    if args.json:
        _print_payload({'employee_id': login}, as_json=True)
    else:
        print(login)
    return 0


def artifact_help(args: argparse.Namespace) -> int:
    print(
        """Artifact Commands
=================

  login      Save a Taihu PAT locally so later commands need no --pat/env
  logout     Remove the saved Taihu PAT
  whoami     Print the acting employee login (helpful when composing URIs)
  ls         List folders/artifacts (own top-level, a folder, or shared-with-me)
  artifacts  List artifacts flat (own or shared)
  mkdir      Create a folder
  rmdir      Delete an empty folder
  upload     Upload a local file and print its name
  upload-large  Upload a large file via chunked multipart pre-signed URLs
  publish    Publish a local directory of task artifacts into a new folder
  download   Download an artifact to a local file
  url        Print a pre-signed download URL
  info       Show artifact metadata
  rm         Delete an artifact
  grants     List grants on a folder or artifact
  grant      Grant employee read access
  revoke     Revoke employee read access

Resources are addressed by two interchangeable forms:

  * Bare slash-path (e.g. /models/v1/model.bin): the caller's own resource.
  * artifact:///<owner>/<path> URI: an explicit owner (yours or someone else's).
    Same-name conflicts across grantors are impossible to confuse this way.
  * artifact:/// (list commands only): the aggregated shared-with-me view.

Listings default to bare names; the artifact:/// aggregated view prints one full
URI per row so multi-grantor collisions stay distinguishable. --show-id adds
resource ids; --json augments each item with a computed `uri` field.

Auth/server come from flags, env, or a saved key file. The Taihu PAT resolves
as --pat > $ARTIFACT_TAIHU_PAT > ~/.nexus/auth/taihu (run
`nexus artifact login` once to save it). Point $ARTIFACT_SERVICE_URL at the
Taihu gateway. A dev-only --dev-employee header can be used against a
gateway-less development backend.

Examples:

  export ARTIFACT_SERVICE_URL=https://artifact-nexus-agent.mcp.woa.com
  nexus artifact login                 # prompts securely, saves the PAT locally
  nexus artifact mkdir /models
  nexus artifact upload ./model.bin --folder /models
  nexus artifact upload-large ./bigmodel.bin --folder /models --concurrency 8
  nexus artifact publish ./task-out --task my-run --grant alice,bob
  nexus artifact ls /models                # your /models
  nexus artifact download /models/model.bin -o ./model.bin
  nexus artifact grant /models alice
  nexus artifact ls artifact:///                     # aggregate shared-with-me
  nexus artifact ls artifact:///alice/               # alice's shared top-level
  nexus artifact download artifact:///alice/model.bin -o ./model.bin
"""
    )
    return 0


def _add_client_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--server',
        help='Artifact service base URL (Taihu gateway or backend); default: $ARTIFACT_SERVICE_URL',
    )
    parser.add_argument(
        '--pat',
        help='Taihu personal access token, sent as "Authorization: Bearer <PAT>"; default: $ARTIFACT_TAIHU_PAT',
    )
    parser.add_argument(
        '--dev-employee',
        dest='dev_employee',
        help='Dev-only plaintext employee id sent as X-Dev-Employee (gateway-less dev backend); '
        'default: $ARTIFACT_DEV_EMPLOYEE',
    )


def _add_grant_target_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('path', help='Folder or artifact path')
    parser.add_argument('grantee_id', help='Employee LoginName to grant read access to')
    parser.add_argument(
        '--type',
        default='employee',
        choices=['employee'],
        help='Grantee type; only employee is supported',
    )
    parser.add_argument(
        '--permission',
        default='read',
        choices=['read'],
        help='Permission to grant; only read is supported',
    )


def _error_message(exc: Exception) -> str:
    return str(exc) or exc.__class__.__name__


def _report_error(args: argparse.Namespace, exc: Exception) -> None:
    message = _error_message(exc)
    if getattr(args, 'json', False):
        print(message, file=sys.stderr)
    else:
        logger.error('%s', message)


def _print_payload(payload: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).rstrip())


# --- publish helpers --------------------------------------------------------


def _scan_files(local_dir: Path) -> list[tuple[Path, Path]]:
    """Return ``(path, path_relative_to_local_dir)`` for every regular file.

    Symlinks are refused, never followed. ``publish`` targets are frequently
    agent- or task-controlled output directories, so following a symlink such
    as ``result -> ../secret`` (or ``-> /etc/passwd``) would upload a host file
    from *outside* ``local_dir`` as if it were a produced artifact. We therefore
    skip any symlinked entry; ``rglob`` does not descend into symlinked
    directories, so an entire out-of-tree subtree cannot be pulled in either.
    Only real regular files are kept (fifos, sockets and broken symlinks are
    skipped). The actual open is additionally performed no-follow (see
    ``upload_file(follow_symlinks=False)`` in the publish loop) so a symlink
    swapped in *after* this scan cannot redirect the read. The listing is
    sorted for a deterministic upload order.
    """

    files: list[tuple[Path, Path]] = []
    for path in sorted(local_dir.rglob('*')):
        if path.is_symlink():
            logger.warning('Skipping symlink (not followed during publish): %s', path)
            continue
        if path.is_file():
            files.append((path, path.relative_to(local_dir)))
    return files


async def _ensure_dir_folder(client: Any, cache: dict[tuple[str, ...], str], rel_dir: Path) -> str:
    """Return the folder id for ``rel_dir``, creating missing ancestors once.

    ``cache`` maps a relative directory (as a tuple of path parts, ``()`` for the
    root) to its already-created folder id. Intermediate folders are created
    top-down so each segment is nested under the correct parent.
    """

    parent_id = cache[()]
    prefix: tuple[str, ...] = ()
    for part in rel_dir.parts:
        if part in ('', '.'):
            continue
        prefix = (*prefix, part)
        cached = cache.get(prefix)
        if cached is not None:
            parent_id = cached
            continue
        folder = await client.create_folder(part, parent_folder_id=parent_id)
        parent_id = folder.folder_id
        cache[prefix] = parent_id
    return parent_id


async def _ensure_tasks_date_folder(client: Any, dt: datetime) -> str:
    """Lazily create ``/tasks/YYYY/MM/DD`` and return the leaf folder id.

    Each segment is created with a deterministic idempotency key so concurrent
    publishes on the same day get-or-create the same shared date directories
    rather than colliding with a ``name_conflict`` error.  The key is scoped to
    the path only (the server already scopes it to the caller's identity).
    """
    segments = ['tasks', dt.strftime('%Y'), dt.strftime('%m'), dt.strftime('%d')]
    parent_id: str | None = None
    path_so_far = ''
    for seg in segments:
        path_so_far = f'{path_so_far}/{seg}'
        folder = await client.create_folder(
            seg,
            parent_folder_id=parent_id,
            idempotency_key=f'tasks-dir:{path_so_far}',
        )
        parent_id = folder.folder_id
    return parent_id  # type: ignore[return-value]


def _build_folder_name(local_dir: Path, task: str | None, dt: datetime | None = None) -> str:
    """Build ``<label>-<UTC timestamp>-<short random>``; label defaults to the directory name.

    The short random suffix disambiguates two runs that land in the same second
    (folder names must be unique among an owner's siblings).
    """

    label = (task or '').strip() or local_dir.resolve().name
    label = label.strip('/').strip() or 'artifacts'
    ts = dt.strftime('%Y%m%d-%H%M%S') if dt is not None else _utc_timestamp()
    return f'{label}-{ts}-{_short_suffix()}'


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')


def _short_suffix() -> str:
    """Return a short random hex suffix (disambiguates same-second runs)."""

    return uuid.uuid4().hex[:6]


def _parse_grantees(raw: str | None) -> list[str]:
    """Split a comma-separated ``--grant`` value into unique, ordered LoginNames."""

    if not raw:
        return []
    seen: dict[str, None] = {}
    for item in raw.split(','):
        name = item.strip()
        if name:
            seen.setdefault(name, None)
    return list(seen)
