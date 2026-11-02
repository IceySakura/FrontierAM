"""Dataset CLI commands for Nexus dataset authoring and inspection."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import tempfile

from pathlib import Path
from typing import Any

import yaml

from nexus.dataset.cache import sha256_file
from nexus.dataset.item import TaskFilesSpec
from nexus.dataset.layout import create_dataset_template, is_local_dataset_path
from nexus.dataset.reader import (
    DatasetReaderError,
    DatasetVersionCatalog,
    LocalDatasetReader,
    RawDatasetItemReader,
    _is_not_found_error,
    get_item,
    list_item_ids,
    load_items,
    open_dataset,
)
from nexus.dataset.source import parse_dataset_source_uri
from nexus.dataset.uri import DatasetURI, parse_dataset_uri
from nexus.dataset.zip_index import NexusZipError, read_zip_index_from_file, write_nexus_zip_directory_to_file


logger = logging.getLogger(__name__)


def _read_taihu_token(args: argparse.Namespace) -> str | None:
    """Return a Taihu PAT from --taihu-token or --taihu-token-file, or None."""

    token: str | None = getattr(args, 'taihu_token', None)
    token_file: Path | None = getattr(args, 'taihu_token_file', None)
    if token and token_file:
        raise ValueError('--taihu-token and --taihu-token-file are mutually exclusive')
    if token:
        return token.strip()
    if token_file is not None:
        from nexus.dataset.auth import tai_pat_from_file

        return tai_pat_from_file(token_file)
    return None


def _make_dataset_authenticator(args: argparse.Namespace) -> Any:
    """Build an optional explicit authenticator from Dataset CLI arguments."""

    from nexus.dataset.auth import TaiPatAuthenticator

    token = _read_taihu_token(args)
    return TaiPatAuthenticator(pat=token) if token is not None else None


def _make_dataset_client(args: argparse.Namespace) -> Any:
    """Create a DatasetClient with authentication configured from CLI args."""

    from nexus.dataset.client import DatasetClient

    return DatasetClient(authenticator=_make_dataset_authenticator(args))


def _requested_split(dataset: str, explicit_split: str | None) -> str | None:
    if explicit_split is not None:
        if not isinstance(explicit_split, str) or not explicit_split:
            raise DatasetReaderError('dataset_split_invalid', detail='split must be a non-empty string')
        return explicit_split
    if dataset.startswith('cos://'):
        from nexus.dataset.cos import CosDatasetRef

        return CosDatasetRef.parse(dataset).split
    source_uri = parse_dataset_source_uri(dataset)
    if source_uri is not None:
        values = (source_uri.query or {}).get('split', ())
        if len(values) > 1:
            raise ValueError(f'Dataset source URI has multiple split values: {dataset}')
        return values[0] if values else None
    path = Path(dataset).expanduser()
    if is_local_dataset_path(path) or path.is_absolute() or dataset.startswith(('./', '../', '~/', '.\\', '..\\')):
        return None
    return parse_dataset_uri(dataset).split


def _add_auth_arguments(parser: argparse.ArgumentParser) -> None:
    """Add --taihu-token / --taihu-token-file auth options to a subcommand parser."""

    parser.add_argument(
        '--taihu-token',
        help='Taihu personal access token; prefer --taihu-token-file for security',
    )
    parser.add_argument(
        '--taihu-token-file',
        type=Path,
        help='File containing a Taihu personal access token (must be chmod 600)',
    )


def dataset_init(args: argparse.Namespace) -> int:
    try:
        root = create_dataset_template(
            args.path,
            template=args.template,
            item_id=args.id,
            name=args.name,
            force=args.force,
        )
    except Exception as exc:
        logger.error('%s', _error_message(exc))
        return 1
    logger.info('Dataset template created: %s', root)
    return 0


def dataset_show(args: argparse.Namespace) -> int:
    async def _run() -> dict[str, Any]:
        async with open_dataset(
            args.dataset,
            authenticator=_make_dataset_authenticator(args),
        ) as reader:
            return (await reader.describe()).model_dump(mode='json')

    try:
        payload = asyncio.run(_run())
    except Exception as exc:
        logger.error('%s', _error_message(exc))
        return 1
    _print_payload(payload, as_json=args.json)
    return 0


def dataset_versions(args: argparse.Namespace) -> int:
    async def _run() -> dict[str, Any]:
        try:
            reader = open_dataset(
                args.dataset,
                authenticator=_make_dataset_authenticator(args),
            )
        except DatasetReaderError as exc:
            if exc.code == 'dataset_reader_scheme_not_found':
                raise DatasetReaderError('dataset_versions_unsupported', detail=args.dataset) from exc
            raise
        async with reader:
            if isinstance(reader, DatasetVersionCatalog):
                return await reader.list_versions()
            descriptor = await reader.describe()
            raise DatasetReaderError(
                'dataset_versions_unsupported',
                detail=f'Dataset provider {descriptor.provider!r} does not support version catalogs',
            )

    try:
        payload = asyncio.run(_run())
    except Exception as exc:
        logger.error('%s', _error_message(exc))
        return 1
    _print_payload(payload, as_json=args.json)
    return 0


def dataset_items(args: argparse.Namespace) -> int:
    if getattr(args, 'dataset', None) is None:
        if getattr(args, 'details', False):
            logger.error('--details requires a dataset reference')
            return 1

        async def _run_list_assets() -> dict[str, Any]:
            async with _make_dataset_client(args) as client:
                return await client.list_assets(keyword=args.keyword, limit=args.limit)

        try:
            payload = asyncio.run(_run_list_assets())
        except Exception as exc:
            logger.error('%s', exc)
            return 1
        _print_payload(payload, as_json=args.json)
        return 0

    async def _run_items() -> dict[str, Any]:
        selected_split = _requested_split(args.dataset, args.split)
        async with open_dataset(
            args.dataset,
            authenticator=_make_dataset_authenticator(args),
        ) as reader:
            if getattr(args, 'details', False):
                items = await load_items(reader, split=selected_split, limit=args.limit)
                items_payload: list[dict[str, Any]] = []
                for item in items:
                    task_files = item.resolved_task_files()
                    items_payload.append({
                        'id': item.id,
                        'metadata': item.metadata,
                        'files': task_files.model_dump() if task_files is not None else None,
                    })
                return {
                    'dataset': args.dataset,
                    'split': selected_split,
                    'count': len(items),
                    'items': items_payload,
                }

            item_ids = await list_item_ids(reader, split=selected_split, limit=args.limit)
            return {
                'dataset': args.dataset,
                'split': selected_split,
                'count': len(item_ids),
                'items': [{'id': item_id} for item_id in item_ids],
            }

    try:
        payload = asyncio.run(_run_items())
    except Exception as exc:
        logger.error('%s', _error_message(exc))
        return 1

    if args.json:
        _print_payload(payload, as_json=True)
    else:
        for item in payload['items']:
            print(item['id'])
    return 0


def dataset_get(args: argparse.Namespace) -> int:
    async def _run() -> dict[str, Any]:
        selected_split = _requested_split(args.dataset, args.split)
        async with open_dataset(
            args.dataset,
            authenticator=_make_dataset_authenticator(args),
        ) as reader:
            if isinstance(reader, RawDatasetItemReader):
                return await reader.get_raw_item(args.item_id, split=selected_split)
            item = await get_item(reader, split=selected_split, item_id=args.item_id)
            if isinstance(reader, LocalDatasetReader):
                return item.model_dump()
            return item.model_dump(exclude_none=True)

    try:
        payload = asyncio.run(_run())
    except Exception as exc:
        logger.error('%s', _error_message(exc))
        return 1

    _write_item_json(payload, output=args.output)
    return 0


def dataset_fetch(args: argparse.Namespace) -> int:
    try:
        _validate_fetch_limit(getattr(args, 'limit', None))
        if getattr(args, 'split', None) is not None and not args.split:
            raise DatasetReaderError('dataset_split_invalid', detail='split must be a non-empty string')
    except Exception as exc:
        logger.error('%s', _error_message(exc))
        return 1

    try:
        is_source_uri = args.dataset.startswith('cos://') or parse_dataset_source_uri(args.dataset) is not None
    except Exception as exc:
        logger.error('%s', _error_message(exc))
        return 1
    if is_source_uri:

        async def _run_source() -> dict[str, Any]:
            async with open_dataset(
                args.dataset,
                authenticator=_make_dataset_authenticator(args),
            ) as reader:
                return await _fetch_reader_dataset(args, reader)

        try:
            payload = asyncio.run(_run_source())
        except Exception as exc:
            logger.error('%s', _error_message(exc))
            return 1
        _print_payload(payload, as_json=args.json)
        return 0

    explicit_local = _looks_like_explicit_local_path(args.dataset)
    if explicit_local and not is_local_dataset_path(args.dataset):
        logger.error('%s', _error_message(DatasetReaderError('local_dataset_not_found', detail=str(args.dataset))))
        return 1

    if not explicit_local:

        async def _run() -> dict[str, Any]:
            from nexus.dataset.cache import DEFAULT_DATASET_CACHE_DIR, dataset_cache_dir, sha256_file

            parsed = parse_dataset_uri(args.dataset)
            requested_split = args.split if args.split is not None else parsed.split
            for component, label in (
                (parsed.namespace, 'namespace'),
                (parsed.dataset, 'dataset'),
                (parsed.version, 'version'),
            ):
                _validate_fetch_path_component(str(component), label=label)
            if requested_split is not None:
                _validate_fetch_path_component(requested_split, label='split')
            async with _make_dataset_client(args) as client:
                resolved_version = await client.resolve_version(parsed)
                _validate_fetch_path_component(str(resolved_version), label='version')
                # Pin all subsequent reads to the version resolved for this
                # fetch.  Passing the original ``latest``/``dev`` URI to
                # ``load_items`` would resolve it a second time and could put
                # items from a newer version under the older cache directory.
                resolved_parsed = DatasetURI(
                    namespace=parsed.namespace,
                    dataset=parsed.dataset,
                    version=str(resolved_version),
                    split=parsed.split,
                    raw=parsed.raw,
                )
                if requested_split is not None and args.limit == 0:
                    # ``limit=0`` skips item pagination, but an explicit
                    # split still needs metadata validation.  The normal
                    # positive-limit path validates through ``load_items``;
                    # that call is intentionally absent here.
                    list_splits = getattr(client, 'list_splits', None)
                    if callable(list_splits):
                        try:
                            available_splits = await list_splits(parsed, version=resolved_version)
                        except Exception as exc:
                            if _is_not_found_error(exc):
                                raise DatasetReaderError(
                                    'dataset_split_not_found', detail=str(requested_split)
                                ) from exc
                            raise
                        normalized_splits = {str(name) for name in (available_splits or ())}
                        if normalized_splits and requested_split not in normalized_splits:
                            raise DatasetReaderError('dataset_split_not_found', detail=str(requested_split))
                if requested_split is not None:
                    fetch_splits: list[str | None] = [requested_split]
                else:
                    fetch_splits = list(
                        await client.list_splits(
                            parsed,
                            version=resolved_version,
                        )
                    ) or [None]
                cache_root = dataset_cache_dir(
                    namespace=parsed.namespace,
                    dataset=parsed.dataset,
                    version=resolved_version,
                    cache_dir=DEFAULT_DATASET_CACHE_DIR,
                )
                output_arg = Path(args.output).expanduser() if args.output else cache_root
                _reject_fetch_path_symlinks(output_arg)
                output_root = output_arg.resolve()
                cached_items: list[dict[str, str | None]] = []
                fetched_splits: dict[str, list[str]] = {}
                remaining = args.limit
                found_item = False
                for fetch_split in fetch_splits:
                    # A positive global limit is exhausted once the previous split
                    # supplied enough items.  ``limit=0`` is intentionally handled
                    # as an empty export for every selected split without touching
                    # the provider (DatasetClient rejects zero-page requests).
                    if remaining == 0 and args.limit != 0:
                        break
                    if fetch_split is not None:
                        _validate_fetch_path_component(str(fetch_split), label='split')
                    if args.limit == 0:
                        items = []
                    else:
                        try:
                            items = await client.load_items(
                                resolved_parsed,
                                split=fetch_split,
                                item_id=args.id,
                                limit=remaining,
                            )
                        except Exception as exc:
                            if args.id is not None and _is_not_found_error(exc):
                                items = []
                            else:
                                raise
                    if args.id is not None:
                        items = [item for item in items if item.id == args.id]
                        found_item = found_item or bool(items)
                    if remaining is not None:
                        items = items[: max(remaining, 0)]
                    split_name = _local_split_name(fetch_split, items)
                    _validate_fetch_path_component(split_name, label='split')
                    if split_name in fetched_splits:
                        raise DatasetReaderError('dataset_duplicate_split', detail=split_name)
                    split_item_paths: list[str] = []
                    seen_item_ids: set[str] = set()
                    for item in items:
                        _validate_fetch_path_component(item.id, label='item id')
                        if item.id in seen_item_ids:
                            raise DatasetReaderError('dataset_duplicate_item_id', detail=item.id)
                        seen_item_ids.add(item.id)
                        item_rel_path = f'items/{split_name}/{item.id}/item.yaml'
                        item_dir = output_root / 'items' / split_name / item.id
                        _reject_fetch_symlink_components(output_root, item_dir.relative_to(output_root))
                        item_dir.mkdir(parents=True, exist_ok=True)
                        item_payload = _portable_item_payload(
                            item.model_dump(exclude={'source_path', 'remote_ref'}, exclude_none=True)
                        )

                        spec = item.resolved_task_files()
                        files_path: Path | None = None
                        if spec is not None:
                            if spec.remote is None:
                                raise ValueError(
                                    f'Remote dataset item {item.id!r} has files without remote coordinates'
                                )
                            if spec.type != 'zip':
                                raise ValueError(
                                    f'Remote dataset item {item.id!r} has unsupported files type: {spec.type}'
                                )
                            files_path = item_dir / 'files.zip'
                            _reject_fetch_symlink_components(output_root, files_path.relative_to(output_root))
                            if not _cached_file_matches(
                                files_path, expected_sha256=spec.sha256, expected_size=spec.size
                            ):
                                signed_urls = await client.sign_file_urls(
                                    DatasetURI(
                                        namespace=spec.remote.namespace,
                                        dataset=spec.remote.dataset,
                                        version=spec.remote.version,
                                        split=spec.remote.split,
                                    ),
                                    [spec.path],
                                    split=spec.remote.split,
                                )
                                signed_url = signed_urls.get(spec.path)
                                if not signed_url:
                                    raise ValueError(
                                        f'files:sign response did not include a signed URL for {spec.path!r}'
                                    )
                                assert client.io is not None
                                tmp_path = files_path.with_suffix('.zip.tmp')
                                _reject_fetch_symlink_components(output_root, tmp_path.relative_to(output_root))
                                try:
                                    await client.io.download_file(signed_url, tmp_path)
                                    if spec.size is not None and tmp_path.stat().st_size != spec.size:
                                        raise ValueError(
                                            f'Downloaded files size mismatch for item {item.id}: '
                                            f'expected {spec.size}, got {tmp_path.stat().st_size}'
                                        )
                                    try:
                                        read_zip_index_from_file(tmp_path)
                                    except (OSError, NexusZipError) as exc:
                                        raise ValueError(
                                            f'Downloaded task files are not a valid Nexus ZIP for item {item.id}'
                                        ) from exc
                                    downloaded_sha256 = sha256_file(tmp_path)
                                    if spec.sha256 and downloaded_sha256 != spec.sha256:
                                        raise ValueError(
                                            f'Downloaded files sha256 mismatch for item {item.id}: '
                                            f'expected {spec.sha256}, got {downloaded_sha256}'
                                        )
                                    tmp_path.replace(files_path)
                                finally:
                                    tmp_path.unlink(missing_ok=True)
                            file_sha256 = sha256_file(files_path)
                            if spec.sha256 and file_sha256 != spec.sha256:
                                raise ValueError(
                                    f'Downloaded files sha256 mismatch for item {item.id}: '
                                    f'expected {spec.sha256}, got {file_sha256}'
                                )
                            item_payload['files'] = {
                                'type': 'zip',
                                'path': 'files.zip',
                                'sha256': file_sha256,
                                'size': files_path.stat().st_size,
                            }

                        item_path = item_dir / 'item.yaml'
                        _reject_fetch_symlink_components(output_root, item_path.relative_to(output_root))
                        _write_yaml(item_path, item_payload)
                        split_item_paths.append(item_rel_path)
                        cached_items.append({
                            'id': item.id,
                            'split': split_name,
                            'item_rel_path': item_rel_path,
                            'item_path': str(item_path),
                            'files_path': str(files_path) if files_path is not None else None,
                        })
                    fetched_splits[split_name] = split_item_paths
                    if remaining is not None:
                        remaining -= len(items)

                if args.id is not None and args.limit != 0 and not found_item:
                    raise DatasetReaderError('dataset_item_not_found', detail=args.id)

                _reject_fetch_symlink_components(output_root, Path('dataset.yaml'))
                dataset_yaml = _load_dataset_yaml(output_root / 'dataset.yaml')
                raw_splits = dataset_yaml.get('splits')
                splits: dict[str, Any] = raw_splits if isinstance(raw_splits, dict) else {}
                for split_name, split_item_paths in fetched_splits.items():
                    splits[split_name] = {'items': split_item_paths}
                if not fetched_splits:
                    fetched_splits['default'] = []
                default_split = next(iter(fetched_splits), 'default')
                _reject_fetch_symlink_components(output_root, Path('dataset.yaml'))
                _write_yaml(
                    output_root / 'dataset.yaml',
                    {
                        'schema_version': 1,
                        'name': parsed.dataset,
                        'default_split': default_split,
                        'splits': splits,
                    },
                )
                return {
                    'dataset': DatasetURI(
                        namespace=parsed.namespace,
                        dataset=parsed.dataset,
                        version=resolved_version,
                        split=requested_split,
                    ).display,
                    'cache_dir': str(cache_root),
                    'output_dir': str(output_root),
                    'splits': list(fetched_splits),
                    'count': len(cached_items),
                    'items': cached_items,
                }

        try:
            payload = asyncio.run(_run())
        except Exception as exc:
            logger.error('%s', exc)
            return 1

        _print_payload(payload, as_json=args.json)
        return 0

    logger.info('Local dataset is already available: %s', Path(args.dataset).resolve())
    return 0


async def _fetch_reader_dataset(args: argparse.Namespace, reader: Any) -> dict[str, Any]:
    """Export a registered Dataset Reader into the standard local layout."""

    if not args.output:
        raise ValueError('--output is required when fetching a virtual dataset source')
    output_arg = Path(args.output).expanduser()
    _reject_fetch_path_symlinks(output_arg)
    output_root = output_arg.resolve()
    if output_root.exists():
        if not output_root.is_dir():
            raise ValueError(f'Dataset fetch output is not a directory: {output_root}')
        if any(output_root.iterdir()) and not getattr(args, 'force', False):
            raise ValueError(f'Dataset fetch output is not empty: {output_root}; pass --force to replace it')

    descriptor = await reader.describe()
    requested_split = _requested_split(args.dataset, args.split)
    # A zero-limit export deliberately skips item loading, but an explicit
    # split still has to be checked against the Reader's descriptor.  Without
    # this guard, ``--limit 0 --split missing`` would silently create a new
    # empty split and hide a typo in the query.
    if requested_split is not None and descriptor.splits:
        available_splits = {split.name for split in descriptor.splits}
        if requested_split not in available_splits:
            raise DatasetReaderError('dataset_split_not_found', detail=requested_split)
    if requested_split is not None:
        selected_splits: list[str | None] = [requested_split]
    elif descriptor.splits:
        selected_splits = [split.name for split in descriptor.splits]
    elif descriptor.default_split:
        selected_splits = [descriptor.default_split]
    else:
        selected_splits = [None]

    remaining = args.limit
    split_items: list[tuple[str, list[Any]]] = []
    seen_split_names: set[str] = set()
    found_item = False
    for selected_split in selected_splits:
        if remaining == 0 and args.limit != 0:
            break
        if selected_split is not None:
            _validate_fetch_path_component(selected_split, label='split')
        if args.limit == 0:
            items = []
        else:
            try:
                items = await load_items(reader, split=selected_split, item_id=args.id, limit=remaining)
            except DatasetReaderError as exc:
                if args.id is not None and exc.code == 'dataset_item_not_found':
                    items = []
                else:
                    raise
        if args.id is not None:
            items = [item for item in items if item.id == args.id]
            found_item = found_item or bool(items)
        if remaining is not None:
            items = items[: max(remaining, 0)]
        split_name = selected_split or _infer_item_split(items) or descriptor.default_split or 'default'
        _validate_fetch_path_component(split_name, label='split')
        if split_name in seen_split_names:
            raise DatasetReaderError('dataset_duplicate_split', detail=split_name)
        seen_split_names.add(split_name)
        seen_item_ids: set[str] = set()
        for item in items:
            _validate_fetch_path_component(item.id, label='item id')
            if item.id in seen_item_ids:
                raise DatasetReaderError('dataset_duplicate_item_id', detail=item.id)
            seen_item_ids.add(item.id)
        split_items.append((split_name, items))
        if remaining is not None:
            remaining -= len(items)

    if args.id is not None and args.limit != 0 and not found_item:
        raise DatasetReaderError('dataset_item_not_found', detail=args.id)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    transaction_root = Path(tempfile.mkdtemp(prefix=f'.{output_root.name}.fetch-', dir=output_root.parent))
    staging_root = transaction_root / 'new'
    staging_root.mkdir()
    fetched_splits: dict[str, list[str]] = {}
    fetched_items: list[dict[str, str | None]] = []
    try:
        for split_name, items in split_items:
            item_paths: list[str] = []
            for item in items:
                item_rel_path = f'items/{split_name}/{item.id}/item.yaml'
                item_dir = staging_root / 'items' / split_name / item.id
                item_dir.mkdir(parents=True, exist_ok=True)
                item_payload = _portable_item_payload(
                    item.model_dump(exclude={'source_path', 'remote_ref'}, exclude_none=True)
                )
                spec = item.resolved_task_files()
                staged_files_path: Path | None = None
                if spec is not None:
                    staged_files_path = item_dir / 'files.zip'
                    await _write_reader_task_files(
                        spec,
                        staged_files_path,
                        authenticator=_make_dataset_authenticator(args),
                    )
                    item_payload['files'] = {
                        'type': 'zip',
                        'path': 'files.zip',
                        'sha256': sha256_file(staged_files_path),
                        'size': staged_files_path.stat().st_size,
                    }
                    if spec.root_name is not None:
                        item_payload['files']['root_name'] = spec.root_name

                item_path = item_dir / 'item.yaml'
                _write_yaml(item_path, item_payload)
                item_paths.append(item_rel_path)
                relative_item_dir = item_dir.relative_to(staging_root)
                fetched_items.append({
                    'id': item.id,
                    'split': split_name,
                    'item_rel_path': item_rel_path,
                    'item_path': str(output_root / item_rel_path),
                    'files_path': str(output_root / relative_item_dir / 'files.zip')
                    if staged_files_path is not None
                    else None,
                })
            fetched_splits[split_name] = item_paths

        _write_yaml(
            staging_root / 'dataset.yaml',
            {
                'schema_version': 1,
                'name': descriptor.name or output_root.name,
                'default_split': next(iter(fetched_splits), 'default'),
                'splits': {name: {'items': paths} for name, paths in fetched_splits.items()},
            },
        )
        _replace_fetched_dataset(staging_root, output_root, transaction_root=transaction_root)
    finally:
        shutil.rmtree(transaction_root, ignore_errors=True)

    return {
        'dataset': descriptor.resolved_reference,
        'provider': descriptor.provider,
        'output_dir': str(output_root),
        'splits': list(fetched_splits),
        'count': len(fetched_items),
        'items': fetched_items,
    }


async def _write_reader_task_files(
    spec: TaskFilesSpec,
    target: Path,
    *,
    authenticator: Any = None,
) -> None:
    """Copy or archive one Reader item's task files into a portable ZIP."""

    if spec.source is not None:
        from nexus.dataset.access import DataAccessContext
        from nexus.dataset.blob import FileResolver
        from nexus.dataset.item import LocalFileRef

        if spec.type == 'directory' and isinstance(spec.source, LocalFileRef):
            write_nexus_zip_directory_to_file(Path(spec.source.path), target)
        else:
            async with DataAccessContext.from_env() as context:
                context.authenticator = authenticator
                blob = FileResolver.open(spec.source, context, size=spec.size)
                if spec.size is not None and (await blob.stat()).size != spec.size:
                    raise ValueError('Task files size mismatch')
                await blob.download_to(target, sha256=spec.sha256)
        _validate_exported_task_files_zip(target)
        return
    if not isinstance(spec.path, str) or not spec.path.strip():
        raise ValueError('Task files path must be non-empty')
    if spec.remote is not None:
        from nexus.dataset.task_files import task_files_from_spec

        files = task_files_from_spec(spec, authenticator=authenticator)
        if files is None:  # pragma: no cover - guarded by TaskFilesSpec validation.
            raise ValueError('task_files source is empty')
        with tempfile.TemporaryDirectory(prefix='nexus-fetch-files-') as directory:
            # ``TaskFiles.download('**/*')`` intentionally reports a missing
            # match as ``FileNotFoundError``.  An empty remote ZIP is still a
            # valid task-files archive, so inspect the index first and write an
            # empty portable ZIP when there are no entries.
            if await files.list('**/*'):
                await files.download('**/*', directory)
            write_nexus_zip_directory_to_file(Path(directory), target)
        _validate_exported_task_files_zip(target)
        return

    path = Path(spec.path).expanduser()
    if spec.type == 'directory':
        write_nexus_zip_directory_to_file(path, target)
        _validate_exported_task_files_zip(target)
        return
    if spec.type != 'zip':
        raise ValueError(f'Unsupported task files type: {spec.type}')
    try:
        read_zip_index_from_file(path)
    except (OSError, NexusZipError) as exc:
        raise ValueError(f'Invalid task files ZIP: {path}') from exc
    _validate_declared_task_file_identity(spec, path)
    shutil.copyfile(path, target)


def _validate_declared_task_file_identity(spec: TaskFilesSpec, path: Path) -> None:
    actual_size = path.stat().st_size
    if spec.size is not None and actual_size != spec.size:
        raise ValueError(f'Task files size mismatch: expected {spec.size}, got {actual_size}')
    if spec.sha256 is not None:
        actual_sha256 = sha256_file(path)
        if actual_sha256 != spec.sha256:
            raise ValueError(f'Task files sha256 mismatch: expected {spec.sha256}, got {actual_sha256}')


def _validate_exported_task_files_zip(path: Path) -> None:
    try:
        read_zip_index_from_file(path)
    except (OSError, NexusZipError) as exc:
        raise ValueError(f'Invalid exported task files ZIP: {path}') from exc


def _portable_item_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop live host path hints when exporting an item bundle."""

    _sanitize_portable_value(payload)
    return payload


def _sanitize_portable_value(value: Any, *, path_context: bool = False) -> None:
    """Remove serialized host-path hints from nested item payloads.

    Harbor's compiler currently emits ``task_config.task_dir`` and
    ``metadata.source_path`` at known locations, but extension Readers can
    carry equivalent fields deeper in their body.  Only path-bearing field
    names are removed; ordinary absolute paths in task instructions or other
    user data remain untouched.
    """

    if isinstance(value, dict):
        for key in list(value):
            nested = value[key]
            normalized = str(key).lower()
            if normalized in {'task_dir', 'source_path'}:
                value.pop(key, None)
                continue
            if path_context and normalized == 'path' and isinstance(nested, str) and _looks_like_absolute_path(nested):
                value.pop(key, None)
                continue
            if normalized == 'task_files' and _contains_absolute_task_path(nested):
                # The exporter writes the canonical top-level ``files`` entry
                # after this sanitizer.  Never preserve a second host path in
                # the partial RunInput body.
                value.pop(key, None)
                continue
            _sanitize_portable_value(
                nested,
                path_context=path_context or normalized in {'files', 'task_files'},
            )
        return
    if isinstance(value, list):
        for nested in value:
            _sanitize_portable_value(nested, path_context=path_context)
        return
    if isinstance(value, tuple):
        for nested in value:
            _sanitize_portable_value(nested, path_context=path_context)


def _contains_absolute_task_path(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() == 'path' and isinstance(nested, str) and _looks_like_absolute_path(nested):
                return True
            if _contains_absolute_task_path(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_absolute_task_path(nested) for nested in value)
    elif isinstance(value, str):
        return _looks_like_absolute_path(value)
    return False


def _looks_like_absolute_path(value: str) -> bool:
    expanded = value.strip()
    return (
        Path(expanded).expanduser().is_absolute()
        or expanded.startswith(('\\\\', '/'))
        or (len(expanded) >= 3 and expanded[1] == ':' and expanded[2] in {'/', '\\'})
    )


def _infer_item_split(items: list[Any]) -> str | None:
    for item in items:
        remote_ref = getattr(item, 'remote_ref', None)
        split = getattr(remote_ref, 'split', None)
        if split:
            return str(split)
    return None


def _validate_fetch_path_component(value: str, *, label: str) -> None:
    if value in {'', '.', '..'} or '/' in value or '\\' in value or '\x00' in value:
        raise ValueError(f'Dataset Reader returned unsafe {label}: {value!r}')


def _validate_fetch_limit(limit: int | None) -> None:
    if limit is None:
        return
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise DatasetReaderError('dataset_limit_invalid')


def _looks_like_explicit_local_path(value: str) -> bool:
    path = Path(value).expanduser()
    raw = value.strip()
    return path.exists() or path.is_absolute() or raw.startswith(('./', '../', '~/', '.\\', '..\\'))


def _reject_fetch_symlink_components(root: Path, relative: Path) -> None:
    """Reject pre-existing symlinks before writing a fetched remote dataset."""

    current = root
    if current.is_symlink():
        raise ValueError(f'Dataset fetch output path must not contain a symlink: {current}')
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f'Dataset fetch output path must not contain a symlink: {current}')


def _reject_fetch_path_symlinks(path: Path) -> None:
    """Reject symlinks in an output path before resolving or creating it.

    Checking only ``path.is_symlink()`` is insufficient for a not-yet-created
    leaf such as ``/tmp/link/new``: ``link`` may redirect the write outside the
    caller's requested tree. Walk the lexical absolute path without calling
    ``resolve()`` first, so every existing ancestor and the leaf itself is
    checked with symlink semantics.
    """

    expanded = path.expanduser()
    # ``Path.absolute()`` preserves lexical ``..`` components.  Normalizing
    # with ``os.path.abspath`` first could erase ``link/..`` and hide a
    # symlink that was traversed on the way to the final path.
    absolute = expanded.absolute()
    parts = absolute.parts
    current = Path(parts[0]) if parts else absolute
    if current.is_symlink():
        raise ValueError(f'Dataset fetch output path must not contain a symlink: {current}')
    for component in parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f'Dataset fetch output path must not contain a symlink: {current}')


def _replace_fetched_dataset(staging_root: Path, output_root: Path, *, transaction_root: Path) -> None:
    previous_root = transaction_root / 'previous'
    had_previous = output_root.exists()
    if had_previous:
        output_root.replace(previous_root)
    try:
        staging_root.replace(output_root)
    except BaseException:
        if had_previous and previous_root.exists() and not output_root.exists():
            previous_root.replace(output_root)
        raise
    if had_previous:
        shutil.rmtree(previous_root)


def dataset_publish(args: argparse.Namespace) -> int:
    from nexus.dataset.auth import TaiPatAuthenticator
    from nexus.dataset.publisher import publish_local_dataset, resolve_with_parquet, run_publish

    if (
        getattr(args, 'backend', None) is not None
        or getattr(args, 'import_spec', None) is not None
        or getattr(args, 'id_field', None) is not None
        or getattr(args, 'files_field', None) is not None
        or getattr(args, 'files_to_cwd', False)
        or str(args.path).lower().endswith('.jsonl')
        or '://' in str(args.path)
    ):
        return _dataset_publish_reader(args)

    try:
        namespace, dataset_name, release_version, split = _resolve_publish_target(args)
        with_parquet = resolve_with_parquet(args.with_parquet, build_only=args.build_only)
        token = _read_taihu_token(args)
        authenticator = TaiPatAuthenticator(pat=token) if token is not None else None
        result = run_publish(
            publish_local_dataset(
                args.path,
                namespace=namespace,
                dataset_name=dataset_name,
                release_version=release_version,
                split=split,
                output_dir=args.output,
                snapshot=args.snapshot,
                with_parquet=with_parquet,
                build_only=args.build_only,
                expires=args.expires,
                register_asset=args.register_asset,
                upload_concurrency=args.upload_concurrency,
                creator=args.creator,
                admin_list=args.admin or None,
                visibility=args.visibility,
                description=args.description,
                authenticator=authenticator,
            )
        )
    except Exception as exc:
        logger.error('%s', exc)
        return 1

    build = result.build
    payload = {
        'namespace': build.namespace,
        'dataset': build.dataset,
        'snapshot': build.snapshot,
        'version': release_version,
        'output_dir': str(build.output_dir),
        'files_root': str(build.files_root),
        'file_count': len(build.files),
        'item_count': build.item_count,
        'uploaded': result.uploaded,
        'build_only': args.build_only,
        'with_parquet': with_parquet,
        'upload_concurrency': args.upload_concurrency,
    }
    if args.json:
        _print_payload(payload, as_json=True)
    else:
        logger.info(
            'Dataset publish complete: %s/%s snapshot=%s version=%s files=%s uploaded=%s output=%s',
            build.namespace,
            build.dataset,
            build.snapshot,
            release_version,
            len(build.files),
            result.uploaded,
            build.output_dir,
        )
    return 0


def _record_import_options(args: argparse.Namespace) -> dict[str, Any]:
    return {key: getattr(args, key, None) for key in ('import_spec', 'id_field', 'files_field')}


def _dataset_publish_reader(args: argparse.Namespace) -> int:
    from nexus.dataset.access import DataAccessContext
    from nexus.dataset.auth import TaiPatAuthenticator
    from nexus.dataset.build import DatasetBuilder
    from nexus.dataset.cos import CosPublisher
    from nexus.dataset.publishing import TaijiPublisher
    from nexus.dataset.reader import open_dataset

    async def publish() -> dict[str, Any]:
        namespace, name, version, split = _resolve_publish_target(args)
        backend = getattr(args, 'backend', None) or 'taiji'
        if backend == 'cos' and (not args.store or not args.region):
            raise ValueError('COS publish requires --store and --region')
        if args.build_only and not args.output:
            raise ValueError('Reader build-only requires --output')
        with tempfile.TemporaryDirectory(prefix='nexus-dataset-build-') as temporary:
            async with DataAccessContext.from_env() as context:
                if token := _read_taihu_token(args):
                    context.authenticator = TaiPatAuthenticator(pat=token)
                async with open_dataset(
                    args.path,
                    context=context,
                    options=_record_import_options(args),
                ) as reader:
                    descriptor = await reader.describe()
                    name = name or descriptor.name
                    output = Path(args.output) if args.output else Path(temporary) / 'build'
                    build = await DatasetBuilder(files_to_cwd=getattr(args, 'files_to_cwd', False)).build(
                        reader,
                        output,
                        context,
                        split=split,
                    )
                if args.build_only:
                    return {
                        'build_only': True,
                        'output_dir': str(build.root),
                        'item_count': build.item_count,
                        'content_sha256': build.content_sha256,
                    }
                if backend == 'cos':
                    publisher = CosPublisher(store=args.store, region=args.region)
                else:
                    publisher = TaijiPublisher(
                        with_parquet=args.with_parquet,
                        expires=args.expires,
                        upload_concurrency=args.upload_concurrency,
                        creator=args.creator,
                        admin_list=args.admin or None,
                        visibility=args.visibility,
                        description=args.description,
                    )
                result = await publisher.publish(build, f'{namespace}/{name}@{version}', context)
                return result.model_dump(exclude_none=True)

    try:
        payload = asyncio.run(publish())
    except Exception as exc:
        logger.error('%s', exc)
        return 1
    _print_payload(payload, as_json=args.json)
    return 0


def dataset_import(args: argparse.Namespace) -> int:
    from nexus.dataset.access import DataAccessContext
    from nexus.dataset.build import DatasetBuilder
    from nexus.dataset.publishing import export_local_build
    from nexus.dataset.reader import open_dataset

    async def convert() -> dict[str, Any]:
        output = Path(args.output).expanduser().resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError('Import output must be empty')
        async with DataAccessContext.from_env() as context:
            async with open_dataset(args.path, context=context, options=_record_import_options(args)) as reader:
                descriptor = await reader.describe()
                build = await DatasetBuilder(files_to_cwd=args.files_to_cwd).build(
                    reader,
                    output / '_build',
                    context,
                )
            export_local_build(build, output, name=descriptor.name)
        return {'output_dir': str(output), 'item_count': build.item_count, 'content_sha256': build.content_sha256}

    try:
        _print_payload(asyncio.run(convert()), as_json=args.json)
        return 0
    except Exception as exc:
        logger.error('%s', exc)
        return 1


def dataset_namespace(args: argparse.Namespace) -> int:
    async def _run() -> dict[str, Any]:
        async with _make_dataset_client(args) as client:
            if args.namespace_command == 'show':
                permission = await client.get_namespace_permission(args.namespace)
                if permission is None:
                    return {'schema': 'nexus', 'namespace': args.namespace, 'exists': False}
                return {'exists': True, **permission}
            if args.namespace_command == 'create':
                return await client.create_namespace_permission(args.namespace, args.admin)
            if args.namespace_command == 'update':
                return await client.update_namespace_permission(args.namespace, args.admin)
            raise ValueError('Dataset namespace command is required; use show, create, or update')

    try:
        payload = asyncio.run(_run())
    except Exception as exc:
        logger.error('%s', exc)
        return 1
    _print_payload(payload, as_json=args.json)
    return 0


def handle_dataset_command(args: argparse.Namespace) -> int:
    if args.dataset_command == 'import':
        return dataset_import(args)
    if args.dataset_command == 'init':
        return dataset_init(args)
    if args.dataset_command == 'show':
        return dataset_show(args)
    if args.dataset_command == 'versions':
        return dataset_versions(args)
    if args.dataset_command in {'items', 'list'}:
        return dataset_items(args)
    if args.dataset_command == 'get':
        return dataset_get(args)
    if args.dataset_command == 'fetch':
        return dataset_fetch(args)
    if args.dataset_command == 'publish':
        return dataset_publish(args)
    if args.dataset_command == 'namespace':
        return dataset_namespace(args)
    return dataset_help(args)


def _error_message(exc: Exception) -> str:
    return str(exc) or exc.__class__.__name__


def register_dataset_parser(subparsers: argparse._SubParsersAction) -> None:
    dataset_parser = subparsers.add_parser(
        'dataset',
        help='Dataset authoring and inspection commands',
        description='Commands for Nexus run-part datasets',
    )
    dataset_subparsers = dataset_parser.add_subparsers(dest='dataset_command', help='Dataset subcommands')

    init_parser = dataset_subparsers.add_parser('init', help='Create a local dataset template')
    init_parser.add_argument('path', help='Output dataset directory')
    init_parser.add_argument('--template', choices=['generic', 'terminal', 'swe'], default='terminal')
    init_parser.add_argument('--id', help='Create one item with this id; default creates case-001 through case-003')
    init_parser.add_argument('--name', help='Dataset name written to dataset.yaml')
    init_parser.add_argument('--force', action='store_true', help='Overwrite an existing non-empty directory')

    show_parser = dataset_subparsers.add_parser('show', help='Show dataset metadata')
    show_parser.add_argument('dataset', help='Local dataset path or remote dataset URI')
    show_parser.add_argument('--json', action='store_true', help='Output JSON')
    _add_auth_arguments(show_parser)

    versions_parser = dataset_subparsers.add_parser('versions', help='List dataset versions')
    versions_parser.add_argument('dataset', help='Local dataset path or remote dataset URI')
    versions_parser.add_argument('--json', action='store_true', help='Output JSON')
    _add_auth_arguments(versions_parser)

    items_parser = dataset_subparsers.add_parser('items', help='List dataset item ids')
    _add_item_list_args(items_parser)
    _add_auth_arguments(items_parser)
    list_parser = dataset_subparsers.add_parser('list', help='List remote assets or item ids')
    _add_item_list_args(list_parser, dataset_required=False)
    list_parser.add_argument('--keyword', help='Optional remote asset name filter')
    _add_auth_arguments(list_parser)

    get_parser = dataset_subparsers.add_parser('get', help='Get one dataset item')
    get_parser.add_argument('dataset', help='Local dataset path or remote dataset URI')
    get_parser.add_argument('item_id', help='Item id')
    get_parser.add_argument('--split', '-s', help='Split name')
    get_parser.add_argument('-o', '--output', help='Output file path')
    _add_auth_arguments(get_parser)

    fetch_parser = dataset_subparsers.add_parser('fetch', help='Fetch/cache dataset files')
    fetch_parser.add_argument('dataset', help='Local dataset path or remote dataset URI')
    fetch_parser.add_argument('--split', '-s', help='Split name')
    fetch_parser.add_argument('--id', help='Fetch a single item id')
    fetch_parser.add_argument('--limit', type=int, default=None, help='Maximum number of items to fetch')
    fetch_parser.add_argument('-o', '--output', help='Local dataset output directory; default: dataset cache path')
    fetch_parser.add_argument(
        '--force',
        action='store_true',
        help='Replace a non-empty output directory when fetching a virtual dataset source',
    )
    fetch_parser.add_argument('--json', action='store_true', help='Output JSON summary')
    _add_auth_arguments(fetch_parser)

    publish_parser = dataset_subparsers.add_parser('publish', help='Build, upload, commit, and publish a dataset')
    publish_parser.add_argument('--backend', choices=['taiji', 'cos'], default=None)
    publish_parser.add_argument('--store', help='COS publish root: cos://bucket/prefix')
    publish_parser.add_argument('--region', help='COS region, included in published references')
    publish_parser.add_argument('--import-spec', type=Path, help='JSONL record conversion YAML')
    publish_parser.add_argument('--id-field', help='Required top-level JSONL ID field; overrides import-spec')
    publish_parser.add_argument('--files-field', help='Top-level JSONL directory/ZIP field; overrides import-spec')
    publish_parser.add_argument('--files-to-cwd', action='store_true', help='Restore bundles under task pwd')
    publish_parser.add_argument('path', help='Local dataset directory, JSONL file, or dataset source URI')
    publish_parser.add_argument('target', nargs='?', help='Remote target URI, for example research/my_dataset@v1')
    publish_parser.add_argument('--namespace', help='Dataset namespace; required unless target includes namespace')
    publish_parser.add_argument('--name', help='Dataset name; defaults to dataset.yaml name or directory name')
    publish_parser.add_argument('--version', help='Release version to publish, for example v1.0.0')
    publish_parser.add_argument('--split', '-s', help='Publish only one split')
    publish_parser.add_argument(
        '-o',
        '--output',
        help='Build output directory; default: <dataset>/_nexus_out, overwritten if non-empty',
    )
    publish_parser.add_argument('--snapshot', help='Dev snapshot name, default dev-<UTC timestamp>')
    publish_parser.add_argument(
        '--with-parquet',
        action='store_true',
        default=None,
        dest='with_parquet',
        help='Generate parquet files; auto-installs pyarrow if missing',
    )
    publish_parser.add_argument(
        '--no-parquet',
        action='store_false',
        dest='with_parquet',
        help='Skip parquet generation; this is the default',
    )
    publish_parser.add_argument('--build-only', action='store_true', help='Only build files locally; do not upload')
    publish_parser.add_argument('--expires', type=int, default=None, help='Signed upload URL expiration seconds')
    publish_parser.add_argument(
        '--upload-concurrency',
        type=int,
        default=5,
        help='Maximum concurrent file uploads; default: 5',
    )
    publish_parser.add_argument('--register-asset', action='store_true', help=argparse.SUPPRESS)
    publish_parser.add_argument('--creator', help='UC asset creator; defaults to Wedata key subject when registering')
    publish_parser.add_argument('--admin', action='append', default=[], help='UC asset admin; can be repeated')
    publish_parser.add_argument('--visibility', default='PRIVATE', choices=['PRIVATE', 'PUBLIC'], help='UC visibility')
    publish_parser.add_argument('--description', help='UC asset description')
    publish_parser.add_argument('--json', action='store_true', help='Output JSON summary')
    _add_auth_arguments(publish_parser)

    import_parser = dataset_subparsers.add_parser('import', help='Convert records and files to a local dataset')
    import_parser.add_argument('path')
    import_parser.add_argument('--output', '-o', required=True)
    import_parser.add_argument('--import-spec', type=Path)
    import_parser.add_argument('--id-field', help='Required top-level JSONL ID field; overrides import-spec')
    import_parser.add_argument('--files-field', help='Top-level JSONL directory/ZIP field; overrides import-spec')
    import_parser.add_argument('--files-to-cwd', action='store_true')
    import_parser.add_argument('--json', action='store_true')

    namespace_parser = dataset_subparsers.add_parser('namespace', help='Manage dataset namespace permissions')
    namespace_subparsers = namespace_parser.add_subparsers(
        dest='namespace_command',
        help='Namespace permission commands',
    )
    namespace_show_parser = namespace_subparsers.add_parser('show', help='Show namespace permission metadata')
    namespace_show_parser.add_argument('namespace', help='Dataset namespace')
    namespace_show_parser.add_argument('--json', action='store_true', help='Output JSON')
    _add_auth_arguments(namespace_show_parser)
    for command in ('create', 'update'):
        command_parser = namespace_subparsers.add_parser(
            command,
            help=f'{command.capitalize()} namespace permission metadata',
        )
        command_parser.add_argument('namespace', help='Dataset namespace')
        command_parser.add_argument(
            '--admin',
            action='append',
            required=True,
            help='Namespace administrator; can be repeated',
        )
        command_parser.add_argument('--json', action='store_true', help='Output JSON')
        _add_auth_arguments(command_parser)

    dataset_subparsers.add_parser('help', help='Show dataset command help')


def dataset_help(args: argparse.Namespace) -> int:
    print(
        """Dataset Commands
================

  init      Create a local Nexus dataset template
  show      Show local dataset metadata
  versions  List versions (local datasets report "local")
  items     List item ids
  list      Alias for items
  get       Print one item JSON
  fetch     Fetch/cache files when supported
  publish   Build, upload, commit, and publish a local dataset
  namespace Show, create, or update namespace permissions

Examples:

  nexus dataset init ./workspace/datasets/my_terminal
  nexus dataset items ./workspace/datasets/my_terminal --split test
  nexus dataset get ./workspace/datasets/my_terminal case-001
  nexus dataset fetch swebench_verified -o ./workspace/datasets/swebench_verified
  nexus dataset namespace create research --admin alice
  nexus dataset publish ./workspace/datasets/my_terminal research/my_terminal@v1
"""
    )
    return 0


def _add_item_list_args(parser: argparse.ArgumentParser, *, dataset_required: bool = True) -> None:
    if dataset_required:
        parser.add_argument('dataset', help='Local dataset path or remote dataset URI')
    else:
        parser.add_argument('dataset', nargs='?', help='Local dataset path or remote dataset URI')
    parser.add_argument('--split', '-s', help='Split name')
    parser.add_argument('--limit', type=int, default=None, help='Maximum number of items to show')
    parser.add_argument(
        '--details',
        action='store_true',
        help='Load items and include metadata/files instead of listing IDs only',
    )
    parser.add_argument('--json', action='store_true', help='Output JSON')


def _resolve_publish_target(args: argparse.Namespace) -> tuple[str, str | None, str, str | None]:
    namespace = args.namespace
    dataset_name = args.name
    release_version = args.version
    split = args.split

    if args.target:
        if not _publish_target_has_namespace(args.target):
            raise ValueError('Publish target must include namespace, for example research/my_dataset@v1')
        parsed = parse_dataset_uri(args.target)
        if namespace and namespace != parsed.namespace:
            raise ValueError(f'Publish target namespace {parsed.namespace!r} conflicts with --namespace {namespace!r}')
        if dataset_name and dataset_name != parsed.dataset:
            raise ValueError(f'Publish target dataset {parsed.dataset!r} conflicts with --name {dataset_name!r}')
        if parsed.split and split and split != parsed.split:
            raise ValueError(f'Publish target split {parsed.split!r} conflicts with --split {split!r}')

        target_has_version = _publish_target_has_version(args.target)
        if target_has_version:
            if release_version and release_version != parsed.version:
                raise ValueError(
                    f'Publish target version {parsed.version!r} conflicts with --version {release_version!r}'
                )
            release_version = parsed.version
        elif not release_version:
            raise ValueError('Publish target must include @version, or --version must be provided')

        namespace = parsed.namespace
        dataset_name = parsed.dataset
        split = split or parsed.split

    if not namespace:
        raise ValueError('Publish namespace is required; pass namespace/dataset@version or --namespace')
    if not release_version:
        raise ValueError('Publish version is required; pass namespace/dataset@version or --version')
    return namespace, dataset_name, release_version, split


def _publish_target_has_namespace(value: str) -> bool:
    target_path = value.strip().split('?', 1)[0].split('@', 1)[0].strip('/')
    return '/' in target_path


def _publish_target_has_version(value: str) -> bool:
    return '@' in value.strip().split('?', 1)[0]


def _print_payload(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).rstrip())


def _write_item_json(payload: dict[str, Any], *, output: str | None) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if not output:
        print(text)
        return
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + '\n', encoding='utf-8')


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _load_dataset_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, encoding='utf-8') as f:
        payload = yaml.safe_load(f)
    return payload if isinstance(payload, dict) else {}


def _local_split_name(fetch_split: str | None, items: list[Any]) -> str:
    if fetch_split:
        return fetch_split
    for item in items:
        remote_ref = getattr(item, 'remote_ref', None)
        split = getattr(remote_ref, 'split', None)
        if split:
            return str(split)
    return 'default'


def _cached_file_matches(path: Path, *, expected_sha256: str | None, expected_size: int | None) -> bool:
    if not path.is_file():
        return False
    if expected_size is not None and path.stat().st_size != expected_size:
        return False
    if expected_sha256 is not None:
        from nexus.dataset.cache import sha256_file

        if sha256_file(path) != expected_sha256:
            return False
    try:
        read_zip_index_from_file(path)
    except (OSError, NexusZipError):
        return False
    return True
