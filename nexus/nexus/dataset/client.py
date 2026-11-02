"""High-level dataset facade across local layout, UC, and IO services."""

from __future__ import annotations

import asyncio
import logging

from collections.abc import AsyncIterator
from typing import Any

import aiohttp

from nexus.dataset.auth import DatasetAuthenticator
from nexus.dataset.io import DEFAULT_TIMEOUT_SECONDS, DEFAULT_UPLOAD_CONCURRENCY, DatasetIoClient
from nexus.dataset.item import DatasetItem, RemoteDatasetRef
from nexus.dataset.task_files import TaskFiles, task_files_from_spec
from nexus.dataset.uc import UcMetadataClient
from nexus.dataset.uri import DatasetURI, parse_dataset_uri


logger = logging.getLogger(__name__)

_ITEM_FETCH_CONCURRENCY = 16


class DatasetClient:
    """Async facade for remote Taiji Nexus dataset operations."""

    def __init__(
        self,
        *,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        authenticator: DatasetAuthenticator | None = None,
        server_url: str | None = None,
        uc_base_url: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.authenticator = authenticator
        self.server_url = server_url
        self.uc_base_url = uc_base_url
        self.session: aiohttp.ClientSession | None = None
        self.io: DatasetIoClient | None = None
        self.uc: UcMetadataClient | None = None

    async def __aenter__(self) -> DatasetClient:
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        try:
            self.io = DatasetIoClient(
                session=self.session,
                authenticator=self.authenticator,
                server_url=self.server_url,
            )
            self.uc = UcMetadataClient(
                session=self.session,
                authenticator=self.authenticator,
                uc_base_url=self.uc_base_url,
            )
        except BaseException as exc:
            # If client construction fails before __aenter__ returns, __aexit__
            # will never run, so close the just-created session here to avoid
            # leaking the connection (and aiohttp's "Unclosed client session"
            # warning). Reset the half-initialized state before re-raising.
            logger.warning(
                'DatasetClient initialization failed while constructing sub-clients: %s; '
                'closing the session and re-raising.',
                exc,
                exc_info=True,
            )
            await self.session.close()
            self.session = None
            self.io = None
            self.uc = None
            raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.session is not None:
            await self.session.close()

    async def list_assets(self, *, keyword: str | None = None, limit: int | None = None) -> dict[str, Any]:
        assert self.uc is not None
        return await self.uc.list_assets(asset_name=keyword, page_size=limit)

    async def create_namespace_permission(self, namespace: str, admin_list: list[str]) -> dict[str, Any]:
        assert self.uc is not None
        return await self.uc.create_namespace_permission(namespace, admin_list)

    async def get_namespace_permission(self, namespace: str) -> dict[str, Any] | None:
        assert self.uc is not None
        return await self.uc.get_namespace_permission(namespace)

    async def update_namespace_permission(self, namespace: str, admin_list: list[str]) -> dict[str, Any]:
        assert self.uc is not None
        return await self.uc.update_namespace_permission(namespace, admin_list)

    async def show(self, uri: str | DatasetURI) -> dict[str, Any]:
        parsed = parse_dataset_uri(uri) if isinstance(uri, str) else uri
        assert self.uc is not None
        assert self.io is not None
        asset = await self.uc.get_asset(parsed.namespace, parsed.dataset)
        overview = await self.io.get_dataset(parsed.namespace, parsed.dataset)
        return {'asset': asset, 'overview': overview}

    async def versions(self, uri: str | DatasetURI) -> dict[str, Any]:
        parsed = parse_dataset_uri(uri) if isinstance(uri, str) else uri
        assert self.io is not None
        overview = await self.io.get_dataset(parsed.namespace, parsed.dataset)
        return _extract_versions(overview)

    async def load_items(
        self,
        uri: str | DatasetURI,
        *,
        split: str | None = None,
        item_id: str | None = None,
        limit: int | None = None,
    ) -> list[DatasetItem]:
        return [
            item
            async for item in self.iter_items(
                uri,
                split=split,
                item_id=item_id,
                limit=limit,
            )
        ]

    async def resolve_ref(
        self,
        uri: str | DatasetURI,
        *,
        split: str | None = None,
    ) -> RemoteDatasetRef:
        parsed = parse_dataset_uri(uri) if isinstance(uri, str) else uri
        resolved_version = await self.resolve_version(parsed)
        resolved_split = await self.resolve_split(parsed, version=resolved_version, split=split or parsed.split)
        return RemoteDatasetRef(
            namespace=parsed.namespace,
            dataset=parsed.dataset,
            version=resolved_version,
            split=resolved_split,
        )

    async def iter_items(
        self,
        uri: str | DatasetURI,
        *,
        split: str | None = None,
        item_id: str | None = None,
        limit: int | None = None,
        page_size: int = 500,
    ) -> AsyncIterator[DatasetItem]:
        ref = await self.resolve_ref(uri, split=split)
        if item_id is not None:
            assert self.io is not None
            record = await self.io.get_item(
                ref.namespace,
                ref.dataset,
                ref.version,
                item_id,
                split=ref.split,
            )
            yield _item_from_record(record, fallback_id=item_id, remote_ref=ref)
            return
        async for item in self.iter_resolved_items(ref, limit=limit, page_size=page_size):
            yield item

    async def iter_resolved_items(
        self,
        ref: RemoteDatasetRef,
        *,
        limit: int | None = None,
        page_size: int = 500,
    ) -> AsyncIterator[DatasetItem]:
        emitted = 0
        async for records, item_ids in self._iter_item_pages(ref, limit=limit, page_size=page_size):
            if records:
                for record in records:
                    yield _item_from_record(record, fallback_id=str(emitted), remote_ref=ref)
                    emitted += 1
                continue
            async for item in self._iter_items_by_id(ref, item_ids):
                yield item
                emitted += 1

    async def _iter_item_pages(
        self,
        ref: RemoteDatasetRef,
        *,
        limit: int | None,
        page_size: int,
    ) -> AsyncIterator[tuple[list[dict[str, Any]], list[str]]]:
        if page_size < 1:
            raise ValueError('page_size must be positive')
        if limit is not None and limit < 1:
            raise ValueError('limit must be positive')
        assert self.io is not None

        cursor: str | None = None
        seen_cursors: set[str] = set()
        emitted = 0
        while limit is None or emitted < limit:
            if cursor is not None:
                if cursor in seen_cursors:
                    raise ValueError(f'Dataset item pagination returned repeated cursor: {cursor}')
                seen_cursors.add(cursor)
            request_limit = page_size if limit is None else min(page_size, limit - emitted)
            listed = await self.io.list_items(
                ref.namespace,
                ref.dataset,
                ref.version,
                split=ref.split,
                cursor=cursor,
                limit=request_limit,
            )
            records = _extract_item_records(listed)
            if records and _looks_like_item_record(records[0]):
                page_records = records if limit is None else records[: limit - emitted]
                page_ids: list[str] = []
            else:
                page_records = []
                ids = _extract_item_ids(listed)
                page_ids = ids if limit is None else ids[: limit - emitted]
            emitted += len(page_records) + len(page_ids)
            if page_records or page_ids:
                yield page_records, page_ids
            next_cursor = _extract_next_cursor(listed)
            if next_cursor is None:
                if _extract_has_more(listed):
                    raise ValueError('Dataset item pagination reported more items without a next cursor')
                return
            cursor = next_cursor

    async def _iter_items_by_id(
        self,
        ref: RemoteDatasetRef,
        item_ids: list[str],
    ) -> AsyncIterator[DatasetItem]:
        assert self.io is not None
        for start in range(0, len(item_ids), _ITEM_FETCH_CONCURRENCY):
            current_ids = item_ids[start : start + _ITEM_FETCH_CONCURRENCY]
            records = await asyncio.gather(
                *(
                    self.io.get_item(
                        ref.namespace,
                        ref.dataset,
                        ref.version,
                        current_id,
                        split=ref.split,
                    )
                    for current_id in current_ids
                )
            )
            for current_id, record in zip(current_ids, records, strict=True):
                yield _item_from_record(record, fallback_id=current_id, remote_ref=ref)

    async def list_item_ids(
        self,
        uri: str | DatasetURI,
        *,
        split: str | None = None,
        limit: int | None = None,
    ) -> list[str]:
        return [
            item_id
            async for item_id in self.iter_item_ids(
                uri,
                split=split,
                limit=limit,
            )
        ]

    async def iter_item_ids(
        self,
        uri: str | DatasetURI,
        *,
        split: str | None = None,
        limit: int | None = None,
        page_size: int = 500,
    ) -> AsyncIterator[str]:
        """Stream item IDs without materializing complete Dataset Items."""
        ref = await self.resolve_ref(uri, split=split)
        async for records, page_ids in self._iter_item_pages(ref, limit=limit, page_size=page_size):
            for record in records:
                nested_record = record.get('record')
                item_data = nested_record if isinstance(nested_record, dict) else record
                item_id = _first_present(item_data, 'id', 'item_id', 'itemId') or _first_present(
                    record, 'id', 'item_id', 'itemId'
                )
                if item_id is not None:
                    yield str(item_id)
            for item_id in page_ids:
                yield item_id

    async def list_splits(self, uri: str | DatasetURI, *, version: str | None = None) -> list[str]:
        parsed = parse_dataset_uri(uri) if isinstance(uri, str) else uri
        assert self.io is not None
        resolved_version = version or await self.resolve_version(parsed)
        version_data = await self.io.get_version(parsed.namespace, parsed.dataset, resolved_version)
        splits = _extract_splits(version_data)
        if splits:
            return splits
        default_split = _first_present(version_data, 'default_split', 'defaultSplit')
        return [str(default_split)] if default_split else []

    async def get_raw_item(
        self,
        uri: str | DatasetURI,
        item_id: str,
        *,
        split: str | None = None,
    ) -> dict[str, Any]:
        parsed = parse_dataset_uri(uri) if isinstance(uri, str) else uri
        assert self.io is not None
        resolved_version = await self.resolve_version(parsed)
        resolved_split = await self.resolve_split(parsed, version=resolved_version, split=split or parsed.split)
        record = await self.io.get_item(
            parsed.namespace,
            parsed.dataset,
            resolved_version,
            item_id,
            split=resolved_split,
        )
        nested_record = record.get('record')
        return nested_record if isinstance(nested_record, dict) else record

    async def sign_file_urls(
        self,
        uri: str | DatasetURI,
        files: list[str],
        *,
        split: str | None = None,
    ) -> dict[str, str]:
        parsed = parse_dataset_uri(uri) if isinstance(uri, str) else uri
        assert self.io is not None
        resolved_version = await self.resolve_version(parsed)
        resolved_split = await self.resolve_split(parsed, version=resolved_version, split=split or parsed.split)
        signed = await self.io.sign_files(
            parsed.namespace,
            parsed.dataset,
            resolved_version,
            files,
            split=resolved_split,
        )
        return _extract_signed_urls(signed)

    def task_files_for_item(self, item: DatasetItem, *, run_id: str | None = None) -> TaskFiles | None:
        """Build host-side task_files for an item, reusing this client's auth context.

        Remote task_files signing otherwise re-reads process-global auth
        configuration, which can diverge from the authenticator this client was
        constructed with. Passing ``self.authenticator`` keeps signing
        consistent with the metadata/items path (and honors an explicitly
        provided authenticator). When it is ``None``, signing falls back to
        auto-detection, whose precedence mirrors ``build_default_authenticator``.
        """

        return task_files_from_spec(
            item.resolved_task_files(),
            run_id=run_id,
            authenticator=self.authenticator,
        )

    async def upload_file(
        self,
        namespace: str,
        dataset: str,
        snapshot: str,
        local_path: str,
        *,
        part_size: int | None = None,
        expires: int | None = None,
        max_concurrency: int = DEFAULT_UPLOAD_CONCURRENCY,
    ) -> dict[str, Any]:
        """Upload one file to a dev snapshot with automatic COS multipart selection."""

        assert self.io is not None
        return await self.io.upload_local_file(
            namespace,
            dataset,
            snapshot,
            local_path,
            part_size=part_size,
            expires=expires,
            max_concurrency=max_concurrency,
        )

    async def resolve_version(self, parsed: DatasetURI) -> str:
        if parsed.version not in {'latest', 'dev'}:
            return parsed.version
        assert self.io is not None
        overview = await self.io.get_dataset(parsed.namespace, parsed.dataset)
        if parsed.version == 'latest':
            value = _first_present(overview, 'latest_release', 'latestRelease', 'latest', 'release_version')
        else:
            value = _first_present(overview, 'current_dev', 'currentDev', 'dev', 'dev_version')
        if not value:
            raise ValueError(f'Cannot resolve {parsed.version!r} for dataset {parsed.namespace}/{parsed.dataset}')
        return str(value)

    async def resolve_split(self, parsed: DatasetURI, *, version: str, split: str | None = None) -> str | None:
        if split:
            return split
        assert self.io is not None
        version_data = await self.io.get_version(parsed.namespace, parsed.dataset, version)
        default_split = _first_present(version_data, 'default_split', 'defaultSplit')
        if default_split:
            return str(default_split)
        splits = _extract_splits(version_data)
        if len(splits) == 1:
            return splits[0]
        if len(splits) > 1:
            raise ValueError(f'Split is required; available splits: {", ".join(splits)}')
        return None


def _item_from_record(record: dict[str, Any], *, fallback_id: str, remote_ref: RemoteDatasetRef) -> DatasetItem:
    nested_record = record.get('record')
    data: dict[str, Any] = nested_record if isinstance(nested_record, dict) else record
    if 'id' not in data:
        # A transport wrapper may echo the requested ID while the canonical
        # provider record uses an alias such as ``item_id``.  Prefer aliases
        # from the payload being materialized before falling back to wrapper
        # metadata, otherwise a mismatched nested record can be mislabeled as
        # the requested item.  Never synthesize an ordinal ID: doing so turns
        # a malformed provider record into a valid Dataset Item and can make
        # identity/duplicate checks silently operate on the wrong key.
        record_id = _first_present(data, 'item_id', 'itemId')
        if record_id is None:
            record_id = _first_present(record, 'id', 'item_id', 'itemId')
        if record_id is None:
            raise ValueError(f'Remote dataset item {fallback_id!r} is missing a canonical item id')
        data = {**data, 'id': str(record_id)}
    if 'body' not in data:
        raise ValueError(f'Remote dataset item {fallback_id!r} is not a Nexus run-part item: missing body')
    return DatasetItem.from_mapping(data, remote_ref=remote_ref)


def _extract_versions(overview: dict[str, Any]) -> dict[str, Any]:
    releases = _first_present(overview, 'releases', 'versions', 'release_versions') or []
    latest = _first_present(overview, 'latest_release', 'latestRelease', 'latest')
    current_dev = _first_present(overview, 'current_dev', 'currentDev', 'dev')
    return {
        'latest_release': latest,
        'current_dev': current_dev,
        'versions': releases,
    }


def _extract_item_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ('items', 'records', 'data'):
        value = data.get(key)
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value
    return []


def _extract_item_ids(data: dict[str, Any]) -> list[str]:
    for key in ('ids', 'item_ids', 'itemIds'):
        value = data.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    value = data.get('items')
    if isinstance(value, list) and all(not isinstance(item, dict) for item in value):
        return [str(item) for item in value]
    records = _extract_item_records(data)
    ids: list[str] = []
    for record in records:
        value = _first_present(record, 'id', 'item_id', 'itemId')
        if value is not None:
            ids.append(str(value))
    return ids


def _extract_next_cursor(data: dict[str, Any]) -> str | None:
    containers = [data]
    for key in ('pagination', 'page_info', 'pageInfo', 'meta'):
        value = data.get(key)
        if isinstance(value, dict):
            containers.append(value)
    for container in containers:
        for key in ('next_cursor', 'nextCursor', 'next'):
            value = container.get(key)
            if value not in (None, ''):
                return str(value)
    return None


def _extract_has_more(data: dict[str, Any]) -> bool:
    containers = [data]
    for key in ('pagination', 'page_info', 'pageInfo', 'meta'):
        value = data.get(key)
        if isinstance(value, dict):
            containers.append(value)
    for container in containers:
        for key in ('has_more', 'hasMore'):
            if key in container:
                return container[key] is True
    return False


def _extract_signed_urls(data: dict[str, Any]) -> dict[str, str]:
    for key in ('urls', 'signed_urls', 'signedUrls'):
        value = data.get(key)
        if isinstance(value, dict):
            return {str(path): str(url) for path, url in value.items()}
        if isinstance(value, list):
            result = _extract_urls_from_entries(value)
            if result:
                return result
    result: dict[str, str] = {}
    for key in ('files', 'items'):
        value = data.get(key)
        if isinstance(value, list):
            result.update(_extract_urls_from_entries(value))
    return result


def _extract_urls_from_entries(entries: list[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        path = item.get('path') or item.get('file') or item.get('key')
        url = item.get('url') or item.get('signed_url') or item.get('signedUrl')
        if path and url:
            result[str(path)] = str(url)
    return result


def _extract_splits(version_data: dict[str, Any]) -> list[str]:
    value = version_data.get('splits')
    if isinstance(value, dict):
        return [str(key) for key in value]
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                name = _first_present(item, 'name', 'split')
                if name:
                    result.append(str(name))
        return result
    return []


def _looks_like_item_record(record: dict[str, Any]) -> bool:
    return 'body' in record or isinstance(record.get('record'), dict)


def _first_present(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, ''):
            return value
    return None
