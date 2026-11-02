"""Dataset URI parsing."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit


DEFAULT_NAMESPACE = 'nexus-curated'
DEFAULT_VERSION = 'latest'
TAIJI_NEXUS_PREFIX = 'taiji.nexus.'


@dataclass(frozen=True)
class DatasetURI:
    """Parsed remote dataset URI.

    ``version`` may still be the symbolic values ``latest`` or ``dev``. The dataset
    service resolves those against the dataset overview later.
    """

    namespace: str
    dataset: str
    version: str = DEFAULT_VERSION
    split: str | None = None
    raw: str = ''

    @property
    def display(self) -> str:
        value = f'{self.namespace}/{self.dataset}@{self.version}'
        if self.split:
            value = f'{value}?split={self.split}'
        return value


def parse_dataset_uri(value: str, *, default_namespace: str = DEFAULT_NAMESPACE) -> DatasetURI:
    """Parse a Nexus dataset URI.

    Supported forms:

    - ``dataset``
    - ``dataset@version``
    - ``namespace/dataset``
    - ``namespace/dataset@version``
    - ``taiji.nexus.namespace/dataset@version``
    - any of the above with ``?split=name``
    """

    raw = value.strip()
    if not raw:
        raise ValueError('Dataset URI must not be empty')

    split = None
    parsed = urlsplit(raw)
    path = parsed.path or raw
    query = parse_qs(parsed.query, keep_blank_values=True)
    if 'split' in query:
        values = query['split']
        if any(not item for item in values):
            raise ValueError(f'Dataset URI split must not be empty: {raw}')
        if len(values) > 1:
            raise ValueError(f'Dataset URI has multiple split values: {raw}')
        split = values[0] if values else None

    if path.startswith(TAIJI_NEXUS_PREFIX):
        path = path[len(TAIJI_NEXUS_PREFIX) :]

    version = DEFAULT_VERSION
    if '@' in path:
        path, version = path.rsplit('@', 1)
        if not version:
            raise ValueError(f'Dataset URI version must not be empty: {raw}')

    parts = path.strip('/').split('/')
    if len(parts) == 1:
        namespace = default_namespace
        dataset = parts[0]
    elif len(parts) == 2:
        namespace, dataset = parts
    else:
        raise ValueError(f'Invalid dataset URI: {raw}')

    if not namespace or not dataset:
        raise ValueError(f'Invalid dataset URI: {raw}')

    return DatasetURI(namespace=namespace, dataset=dataset, version=version, split=split, raw=raw)
