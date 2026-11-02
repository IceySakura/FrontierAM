"""Unified read-only dataset handles and reference resolution."""

from __future__ import annotations

import json
import re

from collections.abc import AsyncIterator, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import parse_qsl, urlsplit

import yaml

from pydantic import BaseModel, Field
from typing_extensions import Self

from nexus.dataset.item import DatasetItem, RemoteDatasetRef
from nexus.dataset.uri import DatasetURI, parse_dataset_uri


_SENSITIVE_METADATA_KEY_PARTS = frozenset({
    'accesskey',
    'apikey',
    'authentication',
    'authorization',
    'credential',
    'password',
    'privatekey',
    'secret',
    'signature',
    'signedurl',
    'token',
    'cookie',
    'xtaiidentity',
    'taiidentity',
})
_NON_ALPHANUMERIC_RE = re.compile(r'[^a-z0-9]+')
_SENSITIVE_URL_MARKERS = (
    'presign',
    'signature=',
    'signed',
    'token=',
    'x-amz-signature=',
)


class DatasetReaderError(ValueError, RuntimeError):
    """Stable, coded failure raised at the Dataset Reader boundary.

    It also derives from ``RuntimeError`` so existing callers that only
    distinguish lifecycle failures continue to behave compatibly.
    """

    def __init__(self, code: str, *, detail: str | None = None) -> None:
        self.code = code
        self.detail = detail
        if detail is None:
            default_details = {
                'dataset_limit_invalid': 'limit must be non-negative',
                'dataset_duplicate_item_id': 'Duplicate item id',
                'dataset_reader_scheme_not_found': 'Unknown dataset source scheme',
            }
            detail = default_details.get(code)
            self.detail = detail
        elif code == 'dataset_duplicate_item_id' and not detail.startswith('Duplicate item id'):
            detail = f'Duplicate item id: {detail}'
            self.detail = detail
        elif code == 'dataset_reader_scheme_not_found' and not detail.startswith('Unknown dataset source scheme'):
            detail = f'Unknown dataset source scheme {detail!r}'
            self.detail = detail
        message = f'{code}:{detail}' if detail else code
        super().__init__(message)


class DatasetSplitDescriptor(BaseModel):
    """Description of one dataset split without materializing its items."""

    name: str
    item_count: int | None = None


class DatasetDescriptor(BaseModel):
    """Provider-neutral description of one opened dataset."""

    reference: str
    resolved_reference: str
    provider: str
    name: str
    version: str | None = None
    default_split: str | None = None
    splits: list[DatasetSplitDescriptor] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def source(self) -> str:
        """Backward-compatible alias for the provider name."""

        return self.provider

    @property
    def input_reference(self) -> str:
        """Backward-compatible alias for the input reference."""

        return self.reference


class DatasetReader(Protocol):
    """A context-scoped, read-only handle bound to one dataset reference."""

    async def __aenter__(self) -> Self: ...

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None: ...

    async def describe(self) -> DatasetDescriptor: ...

    def iter_item_ids(
        self,
        *,
        split: str | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[str]: ...

    def iter_items(
        self,
        *,
        split: str | None = None,
        item_id: str | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[DatasetItem]: ...


@runtime_checkable
class DatasetVersionCatalog(Protocol):
    """Optional version-catalog capability implemented by versioned providers."""

    async def list_versions(self) -> dict[str, Any]: ...


@runtime_checkable
class RawDatasetItemReader(Protocol):
    """Optional capability for inspecting provider-native item records."""

    async def get_raw_item(
        self,
        item_id: str,
        *,
        split: str | None = None,
    ) -> dict[str, Any]: ...


class BaseDatasetReader:
    """No-op lifecycle base for readers without external resources."""

    def __init__(self) -> None:
        self._reader_open = False
        self._active_provider_iterators: list[Any] = []

    async def __aenter__(self) -> Self:
        if self._reader_open:
            raise DatasetReaderError('dataset_reader_open', detail='DatasetReader is already open')
        self._reader_open = True
        try:
            await self._open()
        except BaseException:
            self._reader_open = False
            raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            await self._close(exc_type, exc_val, exc_tb)
        finally:
            # Provider generators can hold response bodies or cursors even
            # when the caller leaves one of our public iterators partially
            # consumed.  Close them at the Reader boundary so they cannot
            # outlive the context/session.
            await self._close_active_provider_iterators()
            self._reader_open = False

    def _track_provider_iterator(self, iterator: Any) -> Any:
        self._active_provider_iterators.append(iterator)
        return iterator

    def _untrack_provider_iterator(self, iterator: Any) -> None:
        with suppress(ValueError):
            self._active_provider_iterators.remove(iterator)

    async def _scoped_items(self, iterator: AsyncIterator[DatasetItem]) -> AsyncIterator[DatasetItem]:
        self._require_open()
        self._track_provider_iterator(iterator)
        try:
            async for item in iterator:
                self._require_open()
                yield item
                self._require_open()
        finally:
            self._untrack_provider_iterator(iterator)
            await _close_async_iterator(iterator)

    async def _close_active_provider_iterators(self) -> None:
        active = self._active_provider_iterators
        self._active_provider_iterators = []
        for iterator in active:
            await _close_async_iterator(iterator)

    async def _open(self) -> None:
        return None

    async def _close(self, exc_type, exc_val, exc_tb) -> None:
        return None

    def _require_open(self) -> None:
        if not self._reader_open:
            raise DatasetReaderError(
                'dataset_reader_closed',
                detail='DatasetReader must be used inside async with open_dataset(...)',
            )


class LocalDatasetReader(BaseDatasetReader):
    """Read one existing local ``dataset.yaml`` authoring layout."""

    def __init__(self, reference: str) -> None:
        super().__init__()
        self.reference = reference
        self._dataset: Any | None = None

    async def _open(self) -> None:
        from nexus.dataset.layout import load_local_dataset

        self._dataset = load_local_dataset(self.reference)

    async def _close(self, exc_type, exc_val, exc_tb) -> None:
        self._dataset = None

    @property
    def dataset(self) -> Any:
        self._require_open()
        assert self._dataset is not None
        return self._dataset

    async def describe(self) -> DatasetDescriptor:
        dataset = self.dataset
        splits = [
            DatasetSplitDescriptor(name=name, item_count=len(dataset.item_paths(name))) for name in dataset.splits
        ]
        return DatasetDescriptor(
            reference=self.reference,
            resolved_reference=str(dataset.root),
            provider='local',
            name=str(dataset.raw.get('name') or dataset.root.name),
            default_split=dataset.default_split,
            splits=splits,
            metadata={'consistency': 'live'},
        )

    async def _resolve_split(self, split: str | None) -> str:
        self._require_open()
        _validate_query(split=split)
        available = tuple(self.dataset.splits)
        if split is not None:
            if split not in available:
                raise DatasetReaderError('dataset_split_not_found', detail=split)
            return split
        default = self.dataset.default_split
        if default is not None:
            if default not in available:
                raise DatasetReaderError('dataset_split_not_found', detail=default)
            return default
        if len(available) == 1:
            return available[0]
        raise DatasetReaderError('dataset_split_required')

    async def _iter_paths(self, *, split: str | None, limit: int | None) -> AsyncIterator[Path]:
        self._require_open()
        _validate_query(split=split, limit=limit)
        # Resolve the split even for an empty result.  ``limit=0`` suppresses
        # item enumeration/pagination, but it must not turn an explicitly
        # unknown split into a successful empty query.
        resolved_split = await self._resolve_split(split)
        if limit == 0:
            return
        for index, path in enumerate(self.dataset.item_paths(resolved_split)):
            if limit is not None and index >= limit:
                return
            yield path

    async def iter_item_ids(
        self,
        *,
        split: str | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[str]:
        seen: set[str] = set()
        async for path in self._iter_paths(split=split, limit=limit):
            self._require_open()
            try:
                item_id = _read_local_item_id(path)
            except (OSError, TypeError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
                raise DatasetReaderError('dataset_item_invalid', detail=str(path)) from exc
            if item_id in seen:
                raise DatasetReaderError('dataset_duplicate_item_id', detail=item_id)
            seen.add(item_id)
            yield item_id

    async def iter_items(
        self,
        *,
        split: str | None = None,
        item_id: str | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[DatasetItem]:
        self._require_open()
        _validate_query(split=split, item_id=item_id, limit=limit)
        # Split validation is independent of the requested item count.  Keep
        # the zero-limit fast path free of item parsing while still honoring
        # the explicit-missing-split error contract.
        resolved_split = await self._resolve_split(split)
        if limit == 0:
            return
        seen: set[str] = set()
        if item_id is not None:
            selected: DatasetItem | None = None
            for path in self.dataset.item_paths(resolved_split):
                self._require_open()
                try:
                    item = DatasetItem.from_file(path)
                except (OSError, TypeError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
                    raise DatasetReaderError('dataset_item_invalid', detail=str(path)) from exc
                if item.id in seen:
                    raise DatasetReaderError('dataset_duplicate_item_id', detail=item.id)
                seen.add(item.id)
                if item.id == item_id:
                    selected = item
            if selected is None:
                raise DatasetReaderError('dataset_item_not_found', detail=item_id)
            self._require_open()
            yield selected
            return

        for emitted, path in enumerate(self.dataset.item_paths(resolved_split), start=1):
            self._require_open()
            try:
                item = DatasetItem.from_file(path)
            except (OSError, TypeError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
                raise DatasetReaderError('dataset_item_invalid', detail=str(path)) from exc
            if item.id in seen:
                raise DatasetReaderError('dataset_duplicate_item_id', detail=item.id)
            seen.add(item.id)
            yield item
            if limit is not None and emitted >= limit:
                return


class TaijiDatasetReader(BaseDatasetReader):
    """Bind a Taiji ``DatasetClient`` to one immutable resolved reference."""

    def __init__(
        self,
        reference: str,
        *,
        dataset_client: Any | None = None,
        authenticator: Any | None = None,
        page_size: int = 500,
    ) -> None:
        super().__init__()
        if dataset_client is not None and authenticator is not None:
            raise ValueError('dataset_client and authenticator are mutually exclusive')
        if page_size < 1:
            raise ValueError('page_size must be positive')
        self.reference = reference
        self.parsed = parse_dataset_uri(reference)
        self._client = dataset_client
        self._authenticator = authenticator
        self._owns_client = False
        self._page_size = page_size
        self._resolved_version: str | None = None
        self._descriptor: DatasetDescriptor | None = None
        self._split_catalog_available: bool | None = None
        self._split_names: tuple[str, ...] | None = None

    @property
    def client(self) -> Any:
        self._require_open()
        assert self._client is not None
        return self._client

    async def _open(self) -> None:
        if self._client is None:
            from nexus.dataset.client import DatasetClient

            self._client = DatasetClient(authenticator=self._authenticator)
            await self._client.__aenter__()
            self._owns_client = True

    async def _close(self, exc_type, exc_val, exc_tb) -> None:
        owned_client = self._owns_client
        if owned_client and self._client is not None:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
        self._owns_client = False
        self._descriptor = None
        self._resolved_version = None
        self._split_catalog_available = None
        self._split_names = None
        if owned_client:
            self._client = None

    async def _version(self) -> str:
        if self._resolved_version is None:
            self._resolved_version = await self.client.resolve_version(self.parsed)
        return self._resolved_version

    async def _resolved_split(self, split: str | None) -> str | None:
        _validate_query(split=split)
        requested = split if split is not None else self.parsed.split
        if requested is not None:
            try:
                available = await self._available_splits()
            except Exception as exc:
                if _is_split_not_found_error(exc):
                    raise DatasetReaderError('dataset_split_not_found', detail=requested) from exc
                raise
            if available is not None:
                if requested not in available:
                    raise DatasetReaderError('dataset_split_not_found', detail=requested)
                return requested
            try:
                resolved = await self.client.resolve_split(
                    self.parsed,
                    version=await self._version(),
                    split=requested,
                )
            except Exception as exc:
                if isinstance(exc, DatasetReaderError):
                    raise
                if isinstance(exc, KeyError) or _is_split_not_found_error(exc):
                    raise DatasetReaderError('dataset_split_not_found', detail=requested) from exc
                raise
            if not resolved:
                raise DatasetReaderError('dataset_split_not_found', detail=requested)
            return str(resolved)
        try:
            resolved = await self.client.resolve_split(
                self.parsed,
                version=await self._version(),
                split=requested,
            )
        except ValueError as exc:
            if 'split is required' in str(exc).lower():
                raise DatasetReaderError('dataset_split_required') from exc
            raise
        if resolved is not None:
            return str(resolved)
        # Older/custom clients may return ``None`` even though their optional
        # split catalog is able to identify a sole split.  Preserve the common
        # Reader contract's single-split fallback without hiding an unavailable
        # or multi-split provider response.
        try:
            available = await self._available_splits()
        except Exception as exc:
            if _is_split_not_found_error(exc):
                raise DatasetReaderError('dataset_split_not_found') from exc
            raise
        if available is not None:
            if len(available) == 1:
                return available[0]
            if len(available) > 1:
                raise DatasetReaderError('dataset_split_required')
        return None

    async def _available_splits(self) -> tuple[str, ...] | None:
        """Return the provider split catalog when the client exposes one."""

        if self._split_catalog_available is False:
            return None
        if self._split_catalog_available is True:
            return self._split_names or ()
        list_splits = getattr(self.client, 'list_splits', None)
        if list_splits is None:
            self._split_catalog_available = False
            return None
        version = await self._version()
        names = await list_splits(self._fixed_uri(), version=version)
        self._split_names = tuple(str(name) for name in (names or ()))
        # An empty response is treated as an unavailable catalog.  Some older
        # DatasetClient implementations expose ``list_splits`` but return an
        # empty list while ``resolve_split`` still knows the provider's split.
        if not self._split_names:
            self._split_catalog_available = False
            return None
        self._split_catalog_available = True
        return self._split_names

    def _fixed_uri(self) -> DatasetURI:
        assert self._resolved_version is not None
        return DatasetURI(
            namespace=self.parsed.namespace,
            dataset=self.parsed.dataset,
            version=self._resolved_version,
            raw=self.reference,
        )

    async def describe(self) -> DatasetDescriptor:
        self._require_open()
        if self._descriptor is not None:
            # Keep the cached descriptor private so callers cannot mutate
            # subsequent observations of the same reader handle.
            return self._descriptor.model_copy(deep=True)

        version = await self._version()
        fixed = self._fixed_uri()
        show = getattr(self.client, 'show', None)
        raw_payload = await show(fixed) if show is not None else {}
        payload = _safe_descriptor_metadata(raw_payload)
        try:
            split_names = await self._available_splits()
        except Exception as exc:
            if _is_split_not_found_error(exc):
                raise DatasetReaderError('dataset_split_not_found') from exc
            raise
        try:
            default_split = await self._resolved_split(None)
        except DatasetReaderError as exc:
            if exc.code != 'dataset_split_required':
                raise
            default_split = None
        resolved_reference = _remote_reference(
            self.parsed.namespace,
            self.parsed.dataset,
            version,
            default_split,
        )
        self._descriptor = DatasetDescriptor(
            reference=self.reference,
            resolved_reference=resolved_reference,
            provider='taiji',
            name=self.parsed.dataset,
            version=version,
            default_split=default_split,
            splits=[DatasetSplitDescriptor(name=name) for name in (split_names or ())],
            metadata={
                **payload,
                'namespace': self.parsed.namespace,
                'consistency': 'versioned',
            },
        )
        return self._descriptor.model_copy(deep=True)

    async def resolved_reference(self, *, split: str | None = None) -> str:
        """Return the concrete version/split reference without querying UC metadata."""
        version = await self._version()
        resolved_split = await self._resolved_split(split)
        return _remote_reference(
            self.parsed.namespace,
            self.parsed.dataset,
            version,
            resolved_split,
        )

    async def iter_item_ids(
        self,
        *,
        split: str | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[str]:
        self._require_open()
        _validate_query(split=split, limit=limit)
        resolved_split = await self._resolved_split(split)
        if limit == 0:
            return
        await self._version()
        fixed = self._fixed_uri()
        seen: set[str] = set()
        emitted = 0
        page_iterator = getattr(self.client, '_iter_item_pages', None)
        if page_iterator is not None:
            ref = RemoteDatasetRef(
                namespace=self.parsed.namespace,
                dataset=self.parsed.dataset,
                version=await self._version(),
                split=resolved_split,
            )
            pages = self._track_provider_iterator(aiter(page_iterator(ref, limit=limit, page_size=self._page_size)))
            try:
                while True:
                    self._require_open()
                    try:
                        records, item_ids = await anext(pages)
                    except StopAsyncIteration:
                        return
                    for record in records:
                        self._require_open()
                        if limit is not None and emitted >= limit:
                            return
                        value = _remote_record_id(record)
                        _check_duplicate(value, seen)
                        seen.add(value)
                        emitted += 1
                        yield value
                    for value in item_ids:
                        self._require_open()
                        if limit is not None and emitted >= limit:
                            return
                        value = str(value)
                        _check_duplicate(value, seen)
                        seen.add(value)
                        emitted += 1
                        yield value
            finally:
                await _close_async_iterator(pages)
                self._untrack_provider_iterator(pages)
        else:
            iterator = getattr(self.client, 'iter_item_ids', None)
            if iterator is not None:
                values = self._track_provider_iterator(
                    aiter(
                        iterator(
                            fixed,
                            split=resolved_split,
                            limit=limit,
                            page_size=self._page_size,
                        )
                    )
                )
                try:
                    while True:
                        self._require_open()
                        try:
                            value = await anext(values)
                        except StopAsyncIteration:
                            return
                        value = str(value)
                        if limit is not None and emitted >= limit:
                            return
                        _check_duplicate(value, seen)
                        seen.add(value)
                        emitted += 1
                        yield value
                finally:
                    await _close_async_iterator(values)
                    self._untrack_provider_iterator(values)
            else:
                values = await self.client.list_item_ids(fixed, split=resolved_split, limit=limit)
                for value in values:
                    self._require_open()
                    if limit is not None and emitted >= limit:
                        return
                    value = str(value)
                    _check_duplicate(value, seen)
                    seen.add(value)
                    emitted += 1
                    yield value

    async def iter_items(
        self,
        *,
        split: str | None = None,
        item_id: str | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[DatasetItem]:
        self._require_open()
        _validate_query(split=split, item_id=item_id, limit=limit)
        resolved_split = await self._resolved_split(split)
        if limit == 0:
            return
        await self._version()
        seen: set[str] = set()
        if item_id is not None:
            item = await self._get_item(item_id, resolved_split, lookup_limit=limit)
            self._require_open()
            yield item
            return
        ref = RemoteDatasetRef(
            namespace=self.parsed.namespace,
            dataset=self.parsed.dataset,
            version=await self._version(),
            split=resolved_split,
        )
        iterator = getattr(self.client, 'iter_resolved_items', None)
        if iterator is not None:
            provider_items = self._track_provider_iterator(aiter(iterator(ref, limit=limit, page_size=self._page_size)))
        else:
            provider_items = self._track_provider_iterator(
                aiter(
                    self.client.iter_items(
                        self._fixed_uri(),
                        split=resolved_split,
                        item_id=None,
                        limit=limit,
                        page_size=self._page_size,
                    )
                )
            )
        emitted = 0
        try:
            while True:
                self._require_open()
                try:
                    item = await anext(provider_items)
                except StopAsyncIteration:
                    return
                except Exception as exc:
                    if _is_invalid_item_error(exc):
                        raise DatasetReaderError('dataset_item_invalid') from exc
                    raise
                self._require_open()
                if limit is not None and emitted >= limit:
                    return
                if item.id in seen:
                    raise DatasetReaderError('dataset_duplicate_item_id', detail=item.id)
                seen.add(item.id)
                emitted += 1
                yield item
        finally:
            await _close_async_iterator(provider_items)
            self._untrack_provider_iterator(provider_items)

    async def _get_item(
        self,
        item_id: str,
        resolved_split: str | None,
        *,
        lookup_limit: int | None,
    ) -> DatasetItem:
        ref = RemoteDatasetRef(
            namespace=self.parsed.namespace,
            dataset=self.parsed.dataset,
            version=await self._version(),
            split=resolved_split,
        )
        io = getattr(self.client, 'io', None)
        io_get_item = getattr(io, 'get_item', None) if io is not None else None
        if callable(io_get_item):
            try:
                from nexus.dataset.client import _item_from_record

                record = await io_get_item(
                    ref.namespace,
                    ref.dataset,
                    ref.version,
                    item_id,
                    split=ref.split,
                )
                item = _item_from_record(record, fallback_id=item_id, remote_ref=ref)
                if item.id != item_id:
                    raise DatasetReaderError('dataset_item_not_found', detail=item_id)
                return item
            except Exception as exc:
                if _is_not_found_error(exc):
                    raise DatasetReaderError('dataset_item_not_found', detail=item_id) from exc
                if _is_invalid_item_error(exc):
                    raise DatasetReaderError('dataset_item_invalid', detail=item_id) from exc
                raise
        iterator = getattr(self.client, 'iter_resolved_items', None)
        if iterator is not None:
            # ``limit`` bounds the number of yielded lookup results, not the
            # provider scan: the requested ID may appear after the first page.
            seen: set[str] = set()
            provider_items = self._track_provider_iterator(aiter(iterator(ref, limit=None, page_size=self._page_size)))
            try:
                async for item in provider_items:
                    self._require_open()
                    _check_duplicate(item.id, seen)
                    seen.add(item.id)
                    if item.id == item_id:
                        return item
            except Exception as exc:
                if _is_not_found_error(exc):
                    raise DatasetReaderError('dataset_item_not_found', detail=item_id) from exc
                if _is_invalid_item_error(exc):
                    raise DatasetReaderError('dataset_item_invalid', detail=item_id) from exc
                raise
            finally:
                await _close_async_iterator(provider_items)
                self._untrack_provider_iterator(provider_items)
        else:
            # Legacy clients may ignore ``item_id`` or apply ``limit`` to the
            # provider page rather than to the requested match.  Scan without
            # a collection limit so an exact ID that appears after the first
            # page remains discoverable; the common ``io.get_item`` path above
            # is still a single-record lookup.
            seen = set()
            provider_items = self._track_provider_iterator(
                aiter(
                    self.client.iter_items(
                        self._fixed_uri(),
                        split=resolved_split,
                        item_id=item_id,
                        limit=None,
                        page_size=self._page_size,
                    )
                )
            )
            try:
                async for item in provider_items:
                    self._require_open()
                    _check_duplicate(item.id, seen)
                    seen.add(item.id)
                    if item.id == item_id:
                        return item
            except Exception as exc:
                if _is_not_found_error(exc):
                    raise DatasetReaderError('dataset_item_not_found', detail=item_id) from exc
                if _is_invalid_item_error(exc):
                    raise DatasetReaderError('dataset_item_invalid', detail=item_id) from exc
                raise
            finally:
                await _close_async_iterator(provider_items)
                self._untrack_provider_iterator(provider_items)
        raise DatasetReaderError('dataset_item_not_found', detail=item_id)

    async def list_versions(self) -> dict[str, Any]:
        self._require_open()
        return await self.client.versions(self.parsed)

    async def get_raw_item(
        self,
        item_id: str,
        *,
        split: str | None = None,
    ) -> dict[str, Any]:
        _validate_query(split=split, item_id=item_id)
        await self._version()
        getter = getattr(self.client, 'get_raw_item', None)
        if getter is None:
            item = await self._get_item(item_id, await self._resolved_split(split), lookup_limit=None)
            return item.model_dump(mode='json', exclude_none=True)
        try:
            return await getter(
                self._fixed_uri(),
                item_id,
                split=await self._resolved_split(split),
            )
        except Exception as exc:
            if _is_not_found_error(exc):
                raise DatasetReaderError('dataset_item_not_found', detail=item_id) from exc
            raise


def open_dataset(
    reference: str | Path,
    *,
    dataset_client: Any | None = None,
    authenticator: Any | None = None,
    page_size: int = 500,
    options: Mapping[str, Any] | None = None,
    context: Any | None = None,
) -> DatasetReader:
    """Resolve a source URI, local path, or Taiji reference to one Reader."""
    from nexus.dataset.access import current_data_access

    context = context or current_data_access()
    if dataset_client is not None and authenticator is not None:
        raise ValueError('dataset_client and authenticator are mutually exclusive')

    effective_options = dict(options or {})
    configured_page_size = effective_options.get('page_size', page_size)
    if isinstance(configured_page_size, bool) or not isinstance(configured_page_size, int) or configured_page_size < 1:
        raise DatasetReaderError('dataset_page_size_invalid')

    from nexus.dataset.source import open_dataset_reader, parse_dataset_source_uri

    reference_value = str(reference)
    is_jsonl = Path(reference_value).suffix.lower() == '.jsonl' and '://' not in reference_value
    if not is_jsonl and any(
        effective_options.get(key) is not None for key in ('import_spec', 'id_field', 'files_field')
    ):
        raise ValueError('JSONL import options require a local .jsonl file')
    if reference_value.startswith('cos://'):
        from nexus.dataset.cos import CosDatasetReader

        return CosDatasetReader(reference_value, context=context)
    if is_jsonl:
        from nexus.dataset.records import JsonlRecordReader, RecordAdapter, RecordDatasetReader, resolve_import_spec

        spec = resolve_import_spec(
            effective_options.get('import_spec'),
            id_field=effective_options.get('id_field'),
            files_field=effective_options.get('files_field'),
        )
        return RecordDatasetReader(JsonlRecordReader(reference_value), RecordAdapter(spec))
    source_uri = parse_dataset_source_uri(reference_value)
    if source_uri is not None:
        from nexus.dataset.source import dataset_source_registry

        if source_uri.scheme not in dataset_source_registry:
            raise DatasetReaderError('dataset_reader_scheme_not_found', detail=source_uri.scheme)
        return open_dataset_reader(reference_value, context=context)

    path = Path(reference_value).expanduser()
    if _looks_like_local_path(reference_value, path):
        from nexus.dataset.layout import DATASET_YAML

        root = path.resolve()
        if not root.is_dir():
            raise DatasetReaderError('local_dataset_not_found', detail=str(root))
        if not (root / DATASET_YAML).is_file():
            raise DatasetReaderError('local_dataset_not_found', detail=str(root / DATASET_YAML))
        return LocalDatasetReader(reference_value)

    if dataset_client is None and authenticator is None and context is not None:
        authenticator = context.dataset_authenticator()
        if authenticator is None and not context.allow_local_auth:
            raise ValueError('Taiji credentials are missing for this data access context')
    return TaijiDatasetReader(
        reference_value,
        dataset_client=dataset_client,
        authenticator=authenticator,
        page_size=configured_page_size,
    )


async def list_item_ids(
    reader: DatasetReader,
    *,
    split: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """Collect a Reader's lightweight ID stream."""
    return [item_id async for item_id in reader.iter_item_ids(split=split, limit=limit)]


async def load_items(
    reader: DatasetReader,
    *,
    split: str | None = None,
    item_id: str | None = None,
    limit: int | None = None,
) -> list[DatasetItem]:
    """Collect a Reader's item stream for list-oriented callers."""
    return [
        item
        async for item in reader.iter_items(
            split=split,
            item_id=item_id,
            limit=limit,
        )
    ]


async def get_item(
    reader: DatasetReader,
    *,
    item_id: str,
    split: str | None = None,
) -> DatasetItem:
    """Read exactly one standard Nexus Dataset Item."""
    items = await load_items(reader, split=split, item_id=item_id, limit=1)
    if not items:
        raise DatasetReaderError('dataset_item_not_found', detail=item_id)
    return items[0]


def _validate_query(
    *,
    split: str | None = None,
    item_id: str | None = None,
    limit: int | None = None,
) -> None:
    if split is not None and (not isinstance(split, str) or not split):
        raise DatasetReaderError('dataset_split_invalid', detail='split must be a non-empty string')
    if item_id is not None and (not isinstance(item_id, str) or not item_id):
        raise DatasetReaderError('dataset_item_id_invalid', detail='item_id must be a non-empty string')
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise DatasetReaderError('dataset_limit_invalid')
        if limit < 0:
            raise DatasetReaderError('dataset_limit_invalid')


def _check_duplicate(item_id: str, seen: set[str]) -> None:
    if item_id in seen:
        raise DatasetReaderError('dataset_duplicate_item_id', detail=item_id)


def _is_not_found_error(exc: BaseException) -> bool:
    """Recognize provider/HTTP not-found errors across client versions."""

    if isinstance(exc, KeyError):
        return True
    if getattr(exc, 'http_status', None) == 404:
        return True
    code = str(getattr(exc, 'code', '') or '').lower().replace('-', '_')
    if code in {'not_found', 'item_not_found', 'dataset_item_not_found', 'resource_not_found'}:
        return True
    message = str(exc).lower()
    return 'http 404' in message or 'not found' in message


def _is_split_not_found_error(exc: BaseException) -> bool:
    """Recognize an explicit provider response that the requested split is absent."""

    if _is_not_found_error(exc):
        return True
    message = str(exc).lower()
    if 'split' not in message:
        return False
    return any(
        marker in message
        for marker in (
            'unknown split',
            'split unknown',
            'invalid split',
            'split invalid',
            'unsupported split',
            'split unsupported',
            'no such split',
            'split does not exist',
        )
    )


def _is_invalid_item_error(exc: BaseException) -> bool:
    """Recognize malformed provider item payloads without hiding transport errors."""

    if exc.__class__.__name__ == 'ValidationError':
        return True
    message = str(exc).lower()
    return (
        'missing a canonical item id' in message
        or 'missing body' in message
        or 'must be a nexus run-part item' in message
    )


async def _close_async_iterator(iterator: Any) -> None:
    """Close a provider async iterator without masking the original failure."""

    close = getattr(iterator, 'aclose', None)
    if not callable(close):
        return
    with suppress(Exception):
        await close()


def _remote_record_id(record: Mapping[str, Any]) -> str:
    nested = record.get('record')
    sources = (nested, record) if isinstance(nested, Mapping) else (record,)
    for source in sources:
        assert isinstance(source, Mapping)
        for key in ('id', 'item_id', 'itemId'):
            value = source.get(key)
            if value is not None and str(value):
                return str(value)
    raise DatasetReaderError('dataset_item_invalid', detail='remote item id is missing')


def _read_local_item_id(path: Path) -> str:
    with path.open(encoding='utf-8') as item_file:
        payload = json.load(item_file) if path.suffix == '.json' else yaml.safe_load(item_file)
    if not isinstance(payload, dict):
        raise TypeError(f'Dataset item must be a mapping: {path}')
    item_id = payload.get('id')
    if not isinstance(item_id, str) or not item_id:
        raise ValueError(f'Dataset item must contain a non-empty string id: {path}')
    return item_id


def _looks_like_local_path(reference: str, path: Path) -> bool:
    raw = reference.strip()
    return path.exists() or path.is_absolute() or raw.startswith(('./', '../', '~/', '.\\', '..\\'))


def _remote_reference(namespace: str, dataset: str, version: str, split: str | None) -> str:
    value = f'{namespace}/{dataset}@{version}'
    return f'{value}?split={split}' if split else value


def _safe_descriptor_metadata(value: Any) -> dict[str, Any]:
    """Copy provider metadata while excluding credential-bearing fields.

    ``DatasetClient.show`` combines UC and Dataset Service responses.  Those
    responses are provider-owned and may grow fields such as PATs, API keys,
    credentials, or signed URLs.  Descriptors are exposed to CLI output and
    serialized batch metadata, so retaining arbitrary keys would make the
    Reader boundary a credential exfiltration path.  Keep the useful shape of
    the response, but recursively omit keys whose normalized names identify
    secret material.  Non-mapping provider responses are treated as empty
    metadata rather than being interpolated into the descriptor.
    """

    sanitized = _sanitize_descriptor_value(value)
    return sanitized if isinstance(sanitized, dict) else {}


def _sanitize_descriptor_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[Any, Any] = {}
        for key, nested in value.items():
            if _is_sensitive_metadata_key(key):
                continue
            result[key] = _sanitize_descriptor_value(nested)
        return result
    if isinstance(value, list):
        return [_sanitize_descriptor_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_descriptor_value(item) for item in value)
    if isinstance(value, str) and _is_sensitive_url(value):
        return '<redacted>'
    return value


def _is_sensitive_metadata_key(key: Any) -> bool:
    normalized = _NON_ALPHANUMERIC_RE.sub('', str(key).lower())
    return normalized in {'auth', 'key'} or any(part in normalized for part in _SENSITIVE_METADATA_KEY_PARTS)


def _is_sensitive_url(value: str) -> bool:
    lowered = value.lower()
    if not lowered.startswith(('http://', 'https://')):
        return False
    if any(marker in lowered for marker in _SENSITIVE_URL_MARKERS):
        return True
    try:
        parsed = urlsplit(value)
    except ValueError:
        # A malformed URL should not be treated as safe metadata merely
        # because parsing failed; callers can still retain the raw value when
        # it does not look credential-bearing.
        return False
    if parsed.username is not None or parsed.password is not None:
        return True
    sensitive_query_names = {
        'accesskey',
        'apikey',
        'auth',
        'credential',
        'expires',
        'signature',
        'sig',
        'token',
    }
    return any(
        _NON_ALPHANUMERIC_RE.sub('', key.lower()) in sensitive_query_names
        for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
    )


__all__ = [
    'BaseDatasetReader',
    'DatasetDescriptor',
    'DatasetReader',
    'DatasetReaderError',
    'DatasetSplitDescriptor',
    'DatasetVersionCatalog',
    'LocalDatasetReader',
    'RawDatasetItemReader',
    'TaijiDatasetReader',
    'get_item',
    'list_item_ids',
    'load_items',
    'open_dataset',
]
