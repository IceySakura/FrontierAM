"""Publish a Bootstrap archive, verify it, then update the Mirrors manifest."""

import argparse
import base64
import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERIC_URL = 'https://mirrors.tencent.com/repository/generic'
DEFAULT_REPO = 'nexus-gui-bootstrap-bundle'


def digest(stream):
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        result.update(chunk)
    return result.hexdigest()


def credentials(token_file):
    """Read the skill-compatible config as data, never execute it or log secrets."""
    config = {}
    if token_file.is_file():
        for line in token_file.read_text(encoding='utf-8-sig').splitlines():
            key, separator, value = line.strip().partition('=')
            if separator and key in {'MIRRORS_USERNAME', 'MIRRORS_TOKEN'}:
                config[key] = value.strip().strip('"\'')
    values = [os.environ.get(key) or config.get(key) for key in ('MIRRORS_USERNAME', 'MIRRORS_TOKEN')]
    if not all(values):
        raise ValueError('Configure MIRRORS_USERNAME and MIRRORS_TOKEN in the environment or token file')
    return values


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        destination = urllib.parse.urlsplit(newurl)
        if req.get_method() != 'GET' or destination.scheme not in {'http', 'https'} or destination.username:
            raise RuntimeError('Refusing an unsafe Mirrors redirect')
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        # Mirrors downloads can redirect to signed internal HTTP storage URLs.
        # Never forward upload credentials there; the caller verifies the bytes.
        if redirected is not None and (
            destination.scheme != 'https' or urllib.parse.urlsplit(req.full_url).netloc != destination.netloc
        ):
            redirected.remove_header('Authorization')
        return redirected


class MirrorsStore:
    def __init__(self, repo, username, token, *, base_url=GENERIC_URL):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', repo):
            raise ValueError('Invalid Mirrors repository name')
        self.root_url = f'{base_url}/{repo}/'
        self.authorization = 'Basic ' + base64.b64encode(f'{username}:{token}'.encode()).decode()
        self.opener = urllib.request.build_opener(SafeRedirect())

    def url(self, path):
        return self.root_url + urllib.parse.quote(path, safe='/')

    def request(self, path, *, data=None, size=None):
        headers = {'Authorization': self.authorization, 'Cache-Control': 'no-cache'}
        if data is not None:
            headers.update({'Content-Length': str(size), 'Content-Type': 'application/octet-stream'})
        request = urllib.request.Request(
            self.url(path), data=data, headers=headers, method='GET' if data is None else 'PUT'
        )
        return self.opener.open(request, timeout=300)

    def checksum(self, path):
        try:
            with self.request(path) as response:
                return digest(response)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise RuntimeError(f'Mirrors read failed (HTTP {error.code})') from None

    def upload(self, path, source):
        try:
            with source.open('rb') as stream:
                with self.request(path, data=stream, size=source.stat().st_size) as response:
                    response.read()
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'Mirrors upload failed (HTTP {error.code})') from None


def validate_bundle(bundle):
    bundle = bundle.resolve(strict=True)
    manifest_path = bundle / 'manifest.json'
    if manifest_path.is_symlink():
        raise ValueError('Manifest must be a regular file')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if any(
        not isinstance(manifest.get(key), str) or not re.fullmatch(r'[a-z0-9][a-z0-9_]*', manifest[key])
        for key in ('os', 'arch')
    ):
        raise ValueError('Manifest must specify safe os and arch identifiers')
    files = manifest.get('files', [])
    if manifest.get('schema_version') != 1 or len(files) != 1 or files[0].get('extract') is not True:
        raise ValueError('Expected a Bootstrap schema 1 manifest with one extractable archive')
    item = files[0]
    path = item.get('path', '')
    if item.get('url') or not re.fullmatch(r'bundles/[0-9A-Za-z][0-9A-Za-z.+_-]*\.tar\.gz', path):
        raise ValueError('Archive must use bundles/{version}.tar.gz without a URL override')
    if path != f'bundles/{manifest.get("version")}.tar.gz':
        raise ValueError('Archive version does not match manifest')
    archive = bundle / path
    if archive.is_symlink() or not archive.resolve(strict=True).is_relative_to(bundle):
        raise ValueError('Archive must be a regular in-bundle file')
    with archive.open('rb') as stream:
        if digest(stream) != item.get('sha256'):
            raise ValueError('Archive SHA-256 does not match manifest')
    return manifest_path, archive, item


def publish(bundle, store):
    manifest, archive, item = validate_bundle(bundle)
    published = json.loads(manifest.read_text(encoding='utf-8'))
    target = f'{published["os"]}-{published["arch"]}'
    archive_key = f'{target}/{item["path"]}'
    manifest_key = f'{target}/manifest.json'
    release_version = archive.name.removesuffix('.tar.gz')
    versioned_key = f'{target}/manifest-{release_version}.json'
    existing = store.checksum(archive_key)
    if existing is not None and existing != item['sha256']:
        raise ValueError('Remote archive name already contains different bytes; publish a new version')
    if existing is None:
        print(f'Uploading {archive_key} ({archive.stat().st_size} bytes)', flush=True)
        store.upload(archive_key, archive)
        print('Verifying remote archive SHA-256 before updating manifest', flush=True)
        existing = store.checksum(archive_key)
    if existing != item['sha256']:
        raise RuntimeError('Remote archive verification failed; manifest was not updated')
    # Mirrors redirects manifests to a content-addressed storage URL. Bootstrap
    # resolves relative paths there, so publish an explicit stable archive URL.
    published['files'][0]['url'] = store.url(archive_key)
    published_manifest = bundle / 'published-manifest.json'
    published_manifest.write_text(json.dumps(published, indent=2), encoding='utf-8')
    with published_manifest.open('rb') as stream:
        manifest_hash = digest(stream)
    existing_manifest = store.checksum(versioned_key)
    if existing_manifest is not None and existing_manifest != manifest_hash:
        raise ValueError('Remote versioned manifest already contains different bytes; publish a new version')
    if existing_manifest is None:
        store.upload(versioned_key, published_manifest)
    if store.checksum(versioned_key) != manifest_hash:
        raise RuntimeError('Remote versioned manifest verification failed; latest was not updated')
    if store.checksum(manifest_key) != manifest_hash:
        store.upload(manifest_key, published_manifest)
    if store.checksum(manifest_key) != manifest_hash:
        raise RuntimeError('Remote manifest verification failed')
    result = {
        'status': 'PUBLISHED',
        'platform': target,
        'manifest_url': store.url(manifest_key),
        'versioned_manifest_url': store.url(versioned_key),
        'runtime_version': release_version,
        'archive_url': store.url(archive_key),
        'archive_bytes': archive.stat().st_size,
        'archive_sha256': item['sha256'],
        'manifest_sha256': manifest_hash,
    }
    (bundle / 'publish.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'build/bundle/windows-x86_64')
    parser.add_argument('--repo', default=DEFAULT_REPO)
    parser.add_argument('--token-file', type=Path, default=Path.home() / '.mirrors_token')
    args = parser.parse_args()
    try:
        username, token = credentials(args.token_file)
        result = publish(args.bundle, MirrorsStore(args.repo, username, token))
    except (ValueError, OSError, RuntimeError) as error:
        # Network errors are deliberately not rendered: redirects can contain signed URLs.
        if isinstance(error, OSError):
            parser.exit(1, 'Publish failed: file or network error; check paths, credentials and connectivity\n')
        parser.exit(1, f'Publish failed: {error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
