from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import sys
import tempfile
import uuid

from dataclasses import dataclass
from pathlib import Path

from nexus.distributed.artifacts.store import GenericRepositoryArtifactStore, StoredObject


ROOT = Path(__file__).parents[3]


@dataclass(frozen=True, slots=True)
class StaticCredentials:
    username: str
    token: str

    def read(self) -> tuple[str, str]:
        return self.username, self.token


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.is_file():
        for raw_line in path.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    values.update(os.environ)
    return values


def artifact_credentials(values: dict[str, str]) -> StaticCredentials:
    username = values.get('NEXUS_ARTIFACT_USERNAME') or values.get('NEXUS_DIST_ARTIFACT_WRITER_USERNAME')
    token = values.get('NEXUS_ARTIFACT_TOKEN') or values.get('NEXUS_DIST_ARTIFACT_WRITER_TOKEN')
    if not username or not token:
        raise RuntimeError('NEXUS_ARTIFACT_USERNAME and NEXUS_ARTIFACT_TOKEN are required')
    return StaticCredentials(username=username, token=token)


async def run_contract(
    *,
    base_url: str,
    repository: str,
    credentials: StaticCredentials,
    prefix: str | None = None,
) -> str:
    object_key = f'{prefix or "nexus-cloud-submit-tests"}/{uuid.uuid4().hex}/contract.bin'
    content = f'nexus generic repository contract {uuid.uuid4().hex}\n'.encode()
    digest = hashlib.sha256(content).hexdigest()
    store = GenericRepositoryArtifactStore(
        base_url=base_url,
        repository=repository,
        credentials=credentials,
    )
    stored: StoredObject | None = None
    with tempfile.TemporaryDirectory(prefix='nexus-generic-contract-') as temporary_dir:
        root = Path(temporary_dir)
        source = root / 'source.bin'
        target = root / 'target.bin'
        source.write_bytes(content)
        try:
            stored = await store.put(source, object_key=object_key, sha256=digest, size_bytes=len(content))
            stat = await store.stat(stored)
            if stat.size_bytes != len(content):
                raise AssertionError(f'HEAD size mismatch: {stat.size_bytes}')
            await store.get(stored, target)
            if target.read_bytes() != content:
                raise AssertionError('GET content mismatch')
        finally:
            if stored is not None:
                await store.delete(stored)
                try:
                    await store.stat(stored)
                except FileNotFoundError:
                    pass
                else:
                    raise AssertionError('DELETE did not remove the contract object')
            await store.close()
    return object_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the real Generic Repository artifact-store contract.')
    parser.add_argument('--base-url', default='https://mirrors.tencent.com/repository/generic')
    parser.add_argument('--repository', default='nexus-agent')
    parser.add_argument('--prefix', default='nexus-cloud-submit-tests')
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    values = load_env_file(args.env_file)
    object_key = asyncio.run(
        run_contract(
            base_url=args.base_url,
            repository=args.repository,
            credentials=artifact_credentials(values),
            prefix=args.prefix,
        )
    )
    print(f'generic repository contract passed and cleaned: {object_key}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
