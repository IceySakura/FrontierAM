import hashlib
import json
import threading
import urllib.request

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


@pytest.fixture
def publisher():
    spec = spec_from_file_location(
        'bundle_publish', Path(__file__).resolve().parents[2] / 'deployment/bootstrap/publish.py'
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(tmp_path):
    archive = tmp_path / 'bundles/1.0.0.tar.gz'
    archive.parent.mkdir()
    archive.write_bytes(b'archive')
    manifest = {
        'schema_version': 1,
        'version': '1.0.0',
        'os': 'windows',
        'arch': 'x86_64',
        'files': [
            {
                'path': 'bundles/1.0.0.tar.gz',
                'sha256': hashlib.sha256(b'archive').hexdigest(),
                'extract': True,
            }
        ],
    }
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    return tmp_path


class Store:
    def __init__(self):
        self.files = {}
        self.uploads = []

    def checksum(self, path):
        return hashlib.sha256(self.files[path]).hexdigest() if path in self.files else None

    def upload(self, path, source):
        self.uploads.append(path)
        self.files[path] = source.read_bytes()

    def url(self, path):
        return f'https://example.test/{path}'


def test_publish_uploads_archive_first_and_is_idempotent(publisher, bundle):
    store = Store()
    result = publisher.publish(bundle, store)
    assert store.uploads == [
        'windows-x86_64/bundles/1.0.0.tar.gz',
        'windows-x86_64/manifest-1.0.0.json',
        'windows-x86_64/manifest.json',
    ]
    assert result['status'] == 'PUBLISHED'
    published = json.loads(store.files['windows-x86_64/manifest.json'])
    assert published['files'][0]['url'] == 'https://example.test/windows-x86_64/bundles/1.0.0.tar.gz'
    assert 'url' not in json.loads((bundle / 'manifest.json').read_text())['files'][0]
    publisher.publish(bundle, store)
    assert len(store.uploads) == 3


def test_remote_collision_preserves_existing_manifest(publisher, bundle):
    store = Store()
    store.files = {
        'windows-x86_64/bundles/1.0.0.tar.gz': b'old',
        'windows-x86_64/manifest.json': b'old manifest',
    }
    with pytest.raises(ValueError, match='different bytes'):
        publisher.publish(bundle, store)
    assert store.uploads == []
    assert store.files['windows-x86_64/manifest.json'] == b'old manifest'


def test_failed_archive_verification_never_updates_manifest(publisher, bundle):
    store = Store()
    store.upload = lambda path, source: store.files.update({path: b'corrupt'})
    with pytest.raises(RuntimeError, match='manifest was not updated'):
        publisher.publish(bundle, store)
    assert 'windows-x86_64/manifest.json' not in store.files


@pytest.mark.parametrize('path', ['../escape.tar.gz', 'bundles/../../escape.tar.gz', 'https://example.test/a.tar.gz'])
def test_publish_rejects_unsafe_paths_before_network(publisher, bundle, path):
    manifest_path = bundle / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files'][0]['path'] = path
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Archive must use'):
        publisher.publish(bundle, Store())


def test_corrupt_local_archive_is_not_uploaded(publisher, bundle):
    (bundle / 'bundles/1.0.0.tar.gz').write_bytes(b'corrupt')
    store = Store()
    with pytest.raises(ValueError, match='SHA-256'):
        publisher.publish(bundle, store)
    assert not store.uploads


def test_credentials_are_data_and_env_takes_precedence(publisher, tmp_path, monkeypatch):
    config = tmp_path / '.mirrors_token'
    config.write_text('MIRRORS_USERNAME=user\nMIRRORS_TOKEN="file-token"\n')
    monkeypatch.delenv('MIRRORS_USERNAME', raising=False)
    monkeypatch.setenv('MIRRORS_TOKEN', 'env-token')
    assert publisher.credentials(config) == ['user', 'env-token']


@pytest.mark.parametrize(('os_name', 'arch'), [('windows', 'x86_64'), ('linux', 'aarch64'), ('macos', 'aarch64')])
def test_platform_publication_is_isolated(publisher, bundle, os_name, arch):
    manifest_path = bundle / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(os=os_name, arch=arch)
    manifest_path.write_text(json.dumps(manifest))
    store = Store()
    store.files['other-x86_64/manifest.json'] = b'keep'
    result = publisher.publish(bundle, store)
    prefix = f'{os_name}-{arch}'
    assert store.uploads == [
        f'{prefix}/bundles/1.0.0.tar.gz',
        f'{prefix}/manifest-1.0.0.json',
        f'{prefix}/manifest.json',
    ]
    assert result['manifest_url'] == f'https://example.test/{prefix}/manifest.json'
    assert store.files['other-x86_64/manifest.json'] == b'keep'


@pytest.mark.parametrize(('key', 'value'), [('os', '../windows'), ('arch', 'x86_64/..'), ('os', None), ('arch', '')])
def test_invalid_platform_is_rejected_before_network(publisher, bundle, key, value):
    manifest_path = bundle / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest[key] = value
    manifest_path.write_text(json.dumps(manifest))
    store = Store()
    with pytest.raises(ValueError, match='os and arch'):
        publisher.publish(bundle, store)
    assert not store.uploads


def test_download_redirect_never_forwards_credentials_to_storage(publisher):
    request = urllib.request.Request('https://mirrors.tencent.com/a', headers={'Authorization': 'Basic test'})
    handler = publisher.SafeRedirect()
    redirected = handler.redirect_request(request, None, 302, '', {}, 'https://cdn.example.test/a')
    assert not redirected.has_header('Authorization')
    redirected = handler.redirect_request(request, None, 302, '', {}, 'http://storage.example.test/a')
    assert not redirected.has_header('Authorization')
    with pytest.raises(RuntimeError, match='unsafe'):
        handler.redirect_request(request, None, 302, '', {}, 'file:///a')
    upload = urllib.request.Request('https://mirrors.tencent.com/a', data=b'payload', method='PUT')
    with pytest.raises(RuntimeError, match='unsafe'):
        handler.redirect_request(upload, None, 302, '', {}, 'https://storage.example.test/a')


def test_real_http_put_get_with_basic_auth(publisher, bundle):
    files = {}
    uploads = []

    class Handler(BaseHTTPRequestHandler):
        def do_PUT(self):
            if self.headers.get('Authorization') != 'Basic dXNlcjp0b2tlbg==':
                self.send_error(401)
                return
            files[self.path] = self.rfile.read(int(self.headers['Content-Length']))
            uploads.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{}')

        def do_GET(self):
            if self.path not in files:
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(files[self.path])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        store = publisher.MirrorsStore('test', 'user', 'token', base_url=f'http://127.0.0.1:{server.server_port}')
        assert publisher.publish(bundle, store)['status'] == 'PUBLISHED'
        assert uploads == [
            '/test/windows-x86_64/bundles/1.0.0.tar.gz',
            '/test/windows-x86_64/manifest-1.0.0.json',
            '/test/windows-x86_64/manifest.json',
        ]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_versioned_manifest_collision_does_not_change_latest(publisher, bundle):
    store = Store()
    store.files['windows-x86_64/manifest-1.0.0.json'] = b'existing immutable manifest'
    store.files['windows-x86_64/manifest.json'] = b'latest'
    with pytest.raises(ValueError, match='versioned manifest'):
        publisher.publish(bundle, store)
    assert store.files['windows-x86_64/manifest-1.0.0.json'] == b'existing immutable manifest'
    assert store.files['windows-x86_64/manifest.json'] == b'latest'


def test_versioned_manifest_verification_precedes_latest(publisher, bundle):
    store = Store()
    upload = store.upload

    def corrupt_versioned(path, source):
        upload(path, source)
        if path.endswith('manifest-1.0.0.json'):
            store.files[path] = b'corrupt'

    store.upload = corrupt_versioned
    with pytest.raises(RuntimeError, match='latest was not updated'):
        publisher.publish(bundle, store)
    assert 'windows-x86_64/manifest.json' not in store.files
