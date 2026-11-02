"""Unit tests for :mod:`nexus.cli.commands.artifact`.

Resources are addressed by slash paths; the fake client below models a small
in-memory folder tree so path resolution (``resolve_resource_id`` /
``resolve_parent_folder_id`` / ``leaf_name``) behaves like the real SDK.
"""

import argparse
import json

import pytest

from nexus.cli.commands.artifact import (
    artifact_download,
    artifact_grant,
    artifact_grants,
    artifact_login,
    artifact_logout,
    artifact_ls,
    artifact_mkdir,
    artifact_publish,
    artifact_revoke,
    artifact_rm,
    artifact_upload,
    artifact_upload_large,
    artifact_url,
    handle_artifact_command,
    register_artifact_parser,
)
from nexus.utils.artifact.errors import ArtifactPathError
from nexus.utils.artifact.models import ArtifactInfo, DownloadUrl, Folder, Grant


# --- shared fake client -----------------------------------------------------
#
# The tree models: /models (folder fld_1) containing model.bin (artifact art_1).
# Path resolution walks it exactly like the real client.


_TREE = {
    None: [
        {'resource_type': 'folder', 'resource_id': 'fld_1', 'name': 'models', 'owner_id': 'self'},
    ],
    'fld_1': [
        {
            'resource_type': 'artifact',
            'resource_id': 'art_1',
            'name': 'model.bin',
            'artifact_id': 'art_1',
            'owner_id': 'self',
        },
    ],
}


class FakeArtifactClient:
    last: 'FakeArtifactClient | None' = None

    def __init__(self, **kwargs):
        FakeArtifactClient.last = self
        self.kwargs = kwargs
        self.calls: list[tuple] = []
        # ``self`` is a stand-in login: the CLI compares whatever whoami()
        # returns against the URI's owner segment. Tests use ``self`` for the
        # caller and ``alice`` / ``bob`` when they need to model URIs pointing
        # at other employees.
        self._whoami = 'self'

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def whoami(self):
        self.calls.append(('whoami',))
        return self._whoami

    def uri_targets_other_owner(self, uri_or_path, self_login):
        from nexus.utils.artifact.uri import parse_artifact_uri

        parsed = parse_artifact_uri(uri_or_path)
        if parsed.is_aggregated_shared_view:
            return True
        return parsed.owner is not None and parsed.owner != self_login

    # --- path resolution (mirrors the real SDK over the in-memory tree) -----

    def _split_uri(self, uri_or_path):
        # Cheap URI-aware split matching the real ``parse_artifact_uri``
        # behaviour just enough for tests: we only need segments and to
        # remember that ``artifact:///`` URIs exist so aggregated-view and
        # cross-owner references fail explicitly rather than silently.
        from nexus.utils.artifact.uri import parse_artifact_uri

        parsed = parse_artifact_uri(uri_or_path)
        if parsed.is_aggregated_shared_view:
            raise ArtifactPathError('aggregated view cannot address a resource', path=uri_or_path)
        return list(parsed.segments)

    async def _find(self, parent_id, name):
        for item in _TREE.get(parent_id, []):
            if item['name'] == name:
                return item
        return None

    async def resolve_path(self, uri_or_path, *, expect=None):
        segments = self._split_uri(uri_or_path)
        if not segments:
            raise ArtifactPathError('empty path', path=uri_or_path)
        parent = None
        item = None
        for index, seg in enumerate(segments):
            item = await self._find(parent, seg)
            if item is None:
                raise ArtifactPathError(f'segment {seg!r} not found', path=uri_or_path)
            if index != len(segments) - 1:
                if item.get('resource_type') != 'folder':
                    raise ArtifactPathError('not a folder', path=uri_or_path)
                parent = item['resource_id']
        if expect is not None and item.get('resource_type') != expect:
            raise ArtifactPathError(f'not a {expect}', path=uri_or_path)
        # Record the call so tests can assert dispatch behaviour ("did the CLI
        # walk the URI at all?"), independent of whether it then went down
        # the folder or the artifact branch.
        self.calls.append(('resolve_path', uri_or_path, expect))
        return item

    async def resolve_resource_id(self, uri_or_path, *, expect=None):
        item = await self.resolve_path(uri_or_path, expect=expect)
        self.calls.append(('resolve_resource_id', uri_or_path, expect))
        return item['resource_id']

    async def resolve_parent_folder_id(self, uri_or_path):
        segments = self._split_uri(uri_or_path)
        if len(segments) <= 1:
            self.calls.append(('resolve_parent_folder_id', uri_or_path, None))
            return None
        # For test simplicity, reduce to a bare path when calling recursively.
        parent = await self.resolve_path('/'.join(segments[:-1]), expect='folder')
        self.calls.append(('resolve_parent_folder_id', uri_or_path, parent['resource_id']))
        return parent['resource_id']

    def leaf_name(self, uri_or_path):
        return self._split_uri(uri_or_path)[-1]

    # --- endpoints ----------------------------------------------------------

    async def create_folder(self, name, *, parent_folder_id=None, idempotency_key=None):
        self.calls.append(('create_folder', name, parent_folder_id))
        # A name-based id lets tests assert parent/child wiring (publish tree).
        return Folder.from_dict({'folder_id': f'fld::{name}', 'name': name})

    async def delete_folder(self, folder_id):
        self.calls.append(('delete_folder', folder_id))

    async def list_resources(self, *, parent_folder_id=None, view=None, limit=50, cursor=None):
        self.calls.append(('list_resources', parent_folder_id, view, limit, cursor))
        from nexus.utils.artifact.models import Page

        # view=all (no-arg ls): owned items first, then shared items
        if parent_folder_id is None and view == 'all':
            return Page(
                items=[
                    {'resource_type': 'folder', 'resource_id': 'fld_1', 'name': 'models', 'owner_id': 'self'},
                    {
                        'resource_type': 'artifact',
                        'resource_id': 'art_1',
                        'name': 'model.bin',
                        'artifact_id': 'art_1',
                        'owner_id': 'self',
                    },
                    {
                        'resource_type': 'artifact',
                        'resource_id': 'art_a',
                        'name': 'test.sh',
                        'artifact_id': 'art_a',
                        'owner_id': 'alice',
                    },
                ],
                next_cursor=None,
            )
        # For ``artifact:///`` (top-level, view='shared') return two same-named
        # files from different grantors, so tests can assert the URI-prefixed
        # rendering keeps them distinct.
        if parent_folder_id is None and view == 'shared':
            return Page(
                items=[
                    {
                        'resource_type': 'artifact',
                        'resource_id': 'art_a',
                        'name': 'test.sh',
                        'artifact_id': 'art_a',
                        'owner_id': 'alice',
                    },
                    {
                        'resource_type': 'artifact',
                        'resource_id': 'art_b',
                        'name': 'test.sh',
                        'artifact_id': 'art_b',
                        'owner_id': 'bob',
                    },
                ],
                next_cursor=None,
            )
        return Page(
            items=[
                {'resource_type': 'folder', 'resource_id': 'fld_1', 'name': 'models', 'owner_id': 'self'},
                {'resource_type': 'artifact', 'resource_id': 'art_1', 'name': 'model.bin', 'owner_id': 'self'},
            ],
            next_cursor='cursor-2',
        )

    async def list_artifacts(self, *, view='owned', limit=50, cursor=None):
        self.calls.append(('list_artifacts', view, limit, cursor))
        from nexus.utils.artifact.models import Page

        if view == 'shared':
            return Page(
                items=[
                    {
                        'artifact_id': 'art_a',
                        'name': 'test.sh',
                        'size_bytes': 3,
                        'status': 'ready',
                        'owner_id': 'alice',
                    },
                    {'artifact_id': 'art_b', 'name': 'test.sh', 'size_bytes': 4, 'status': 'ready', 'owner_id': 'bob'},
                ],
                next_cursor=None,
            )
        return Page(
            items=[
                {'artifact_id': 'art_1', 'name': 'model.bin', 'size_bytes': 3, 'status': 'ready', 'owner_id': 'self'}
            ],
            next_cursor='cursor-3',
        )

    async def put_grant(
        self,
        resource_id,
        grantee_id,
        *,
        grantee_type='employee',
        permission='read',
        expires_at=None,
        idempotency_key=None,
    ):
        self.calls.append(('put_grant', resource_id, grantee_id, grantee_type, permission, expires_at))

    async def delete_grant(self, resource_id, grantee_id, *, grantee_type='employee', permission='read'):
        self.calls.append(('delete_grant', resource_id, grantee_id, grantee_type, permission))

    async def list_grants(self, resource_id):
        self.calls.append(('list_grants', resource_id))
        return [Grant.from_dict({'grantee_type': 'employee', 'grantee_id': 'alice', 'permission': 'read'})]

    async def upload_file(
        self,
        path,
        *,
        name=None,
        parent_folder_id=None,
        content_type=None,
        storage_pool=None,
        multipart=None,
        max_concurrency=5,
        idempotency_key=None,
        follow_symlinks=True,
    ):
        self.calls.append((
            'upload_file',
            str(path),
            name,
            parent_folder_id,
            content_type,
            storage_pool,
            multipart,
            max_concurrency,
            follow_symlinks,
        ))
        return ArtifactInfo.from_dict({
            'artifact_id': 'art_up',
            'name': name or 'model.bin',
            'size_bytes': 3,
            'status': 'ready',
        })

    async def get_artifact(self, artifact_id):
        self.calls.append(('get_artifact', artifact_id))
        return ArtifactInfo.from_dict({'artifact_id': artifact_id, 'name': 'model.bin'})

    async def delete_artifact(self, artifact_id):
        self.calls.append(('delete_artifact', artifact_id))

    async def create_download_url(self, artifact_id):
        self.calls.append(('create_download_url', artifact_id))
        return DownloadUrl.from_dict({'method': 'GET', 'url': 'http://cos/get', 'expires_in': 900})

    async def download_to_file(self, artifact_id, local_path):
        from pathlib import Path

        self.calls.append(('download_to_file', artifact_id, str(local_path)))
        path = Path(local_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'data')
        return path

    async def download_folder_to_dir(self, folder_id, dest_dir, *, concurrency=4):
        """Mock folder download: write one file per leaf under ``_TREE[folder_id]``.

        Simple enough to unit-test the CLI's dispatch and reporting logic
        without pulling in the real client's asyncio machinery. When
        ``simulate_folder_failure`` is set, mark every artifact as failed
        so tests can exercise the summary + non-zero exit path.
        """
        from pathlib import Path

        # Look up the failure dataclass through the sys.modules cache rather
        # than via ``ArtifactClient`` -- the CLI test monkeypatches the
        # symbol to *this* class, so ``ArtifactClient.DownloadFailure``
        # would resolve back to the fake, not the real dataclass.
        import importlib

        real_module = importlib.import_module('nexus.utils.artifact.client')
        _RealClient = real_module.ArtifactClient
        # ``monkeypatch`` replaces the module attribute, but the class object
        # itself is unchanged; grab it back via the sentinel we stashed on
        # setup (see fixture).
        _RealClient = getattr(real_module, '_ORIG_ArtifactClient', _RealClient)

        self.calls.append(('download_folder_to_dir', folder_id, str(dest_dir), concurrency))
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        successes = 0
        failures: list = []
        for item in _TREE.get(folder_id, []):
            if item.get('resource_type') != 'artifact':
                continue
            name = str(item.get('name') or '')
            if not name:
                continue
            local_target = dest / name
            if getattr(self, 'simulate_folder_failure', False):
                failures.append(
                    _RealClient.DownloadFailure(
                        artifact_id=str(item.get('artifact_id') or item.get('resource_id') or ''),
                        name=name,
                        relative_path=name,
                        local_path=local_target,
                        error='simulated failure',
                    )
                )
                continue
            local_target.write_bytes(b'data')
            successes += 1
        return successes, failures


@pytest.fixture
def fake_client(monkeypatch):
    FakeArtifactClient.last = None
    # Stash the original class so the mock can build real ``DownloadFailure``
    # instances even while the module attribute points at the fake.
    import nexus.utils.artifact.client as _real_client_mod

    monkeypatch.setattr(_real_client_mod, '_ORIG_ArtifactClient', _real_client_mod.ArtifactClient, raising=False)
    monkeypatch.setattr('nexus.utils.artifact.client.ArtifactClient', FakeArtifactClient)
    return FakeArtifactClient


def _ns(**overrides) -> argparse.Namespace:
    values = {
        'server': None,
        'pat': None,
        'dev_employee': None,
        'json': False,
        'show_id': False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _kinds(calls):
    return [c[0] for c in calls]


# --- parser -----------------------------------------------------------------


def test_register_parser_parses_subcommands():
    parser = argparse.ArgumentParser(prog='nexus')
    subparsers = parser.add_subparsers(dest='command')
    register_artifact_parser(subparsers)

    args = parser.parse_args(['artifact', 'upload', './x.bin', '--folder', '/models', '--name', 'n', '--json'])
    assert args.command == 'artifact' and args.artifact_command == 'upload'
    assert args.path == './x.bin' and args.folder == '/models' and args.name == 'n' and args.json

    args = parser.parse_args(['artifact', 'ls', 'artifact:///', '--limit', '10', '--show-id'])
    assert args.artifact_command == 'ls' and args.target == 'artifact:///' and args.limit == 10
    assert args.show_id

    args = parser.parse_args(['artifact', 'grant', '/models', 'alice', '--permission', 'read'])
    assert args.path == '/models' and args.grantee_id == 'alice' and args.type == 'employee'
    assert args.permission == 'read'

    args = parser.parse_args(['artifact', 'download', '/models/model.bin', '-o', './out.bin'])
    assert args.path == '/models/model.bin' and args.output == './out.bin'


@pytest.mark.parametrize(
    'arguments',
    [
        ['artifact', 'grant', '/models', 'app_x', '--type', 'application'],
        ['artifact', 'grant', '/models', 'alice', '--permission', 'write'],
        ['artifact', 'revoke', '/models', 'app_x', '--type', 'application'],
        ['artifact', 'revoke', '/models', 'alice', '--permission', 'write'],
    ],
)
def test_register_parser_rejects_unsupported_grants(arguments):
    parser = argparse.ArgumentParser(prog='nexus')
    subparsers = parser.add_subparsers(dest='command')
    register_artifact_parser(subparsers)

    with pytest.raises(SystemExit):
        parser.parse_args(arguments)


def test_client_server_is_forwarded(fake_client, capsys):
    rc = artifact_mkdir(_ns(path='/models', server='http://host:8080'))
    assert rc == 0
    assert fake_client.last.kwargs == {
        'base_url': 'http://host:8080',
        'taihu_pat': None,
        'dev_employee': None,
    }


def test_taihu_pat_and_dev_employee_forwarded(fake_client):
    rc = artifact_mkdir(
        _ns(
            path='/models',
            server='https://artifact-nexus-agent.mcp.woa.com',
            pat='tai_pat_xyz',
            dev_employee='junlinyin',
        )
    )
    assert rc == 0
    assert fake_client.last.kwargs['base_url'] == 'https://artifact-nexus-agent.mcp.woa.com'
    assert fake_client.last.kwargs['taihu_pat'] == 'tai_pat_xyz'
    assert fake_client.last.kwargs['dev_employee'] == 'junlinyin'


def test_register_parser_parses_taihu_flags():
    parser = argparse.ArgumentParser(prog='nexus')
    subparsers = parser.add_subparsers(dest='command')
    register_artifact_parser(subparsers)

    args = parser.parse_args([
        'artifact',
        'ls',
        '--server',
        'https://artifact-nexus-agent.mcp.woa.com',
        '--pat',
        'tai_pat_xyz',
        '--dev-employee',
        'junlinyin',
    ])
    assert args.pat == 'tai_pat_xyz'
    assert args.dev_employee == 'junlinyin'
    assert args.server == 'https://artifact-nexus-agent.mcp.woa.com'


# --- login / logout ---------------------------------------------------------


def test_register_parser_parses_login_logout():
    parser = argparse.ArgumentParser(prog='nexus')
    subparsers = parser.add_subparsers(dest='command')
    register_artifact_parser(subparsers)

    args = parser.parse_args(['artifact', 'login', '--pat', 'tai_pat_xyz'])
    assert args.artifact_command == 'login' and args.pat == 'tai_pat_xyz'

    args = parser.parse_args(['artifact', 'logout'])
    assert args.artifact_command == 'logout'


def test_login_saves_pat_from_flag(monkeypatch, tmp_path, capsys):
    from nexus.utils.artifact import auth as auth_module

    key_file = tmp_path / 'key'
    monkeypatch.setattr(auth_module, 'ARTIFACT_AUTH_KEY_FILE', key_file)

    rc = artifact_login(argparse.Namespace(pat='Bearer tai_pat_flag', json=False))

    assert rc == 0
    assert key_file.read_text(encoding='utf-8') == 'tai_pat_flag'
    assert str(key_file) in capsys.readouterr().out


def test_login_prompts_when_pat_missing(monkeypatch, tmp_path):
    from nexus.cli.commands import artifact as artifact_cli
    from nexus.utils.artifact import auth as auth_module

    key_file = tmp_path / 'key'
    monkeypatch.setattr(auth_module, 'ARTIFACT_AUTH_KEY_FILE', key_file)
    monkeypatch.setattr(artifact_cli.getpass, 'getpass', lambda *_a, **_k: 'tai_pat_prompted')

    rc = artifact_login(argparse.Namespace(pat=None, json=False))

    assert rc == 0
    assert key_file.read_text(encoding='utf-8') == 'tai_pat_prompted'


def test_login_rejects_empty_pat(monkeypatch, tmp_path):
    from nexus.cli.commands import artifact as artifact_cli
    from nexus.utils.artifact import auth as auth_module

    monkeypatch.setattr(auth_module, 'ARTIFACT_AUTH_KEY_FILE', tmp_path / 'key')
    monkeypatch.setattr(artifact_cli.getpass, 'getpass', lambda *_a, **_k: '   ')

    rc = artifact_login(argparse.Namespace(pat=None, json=False))

    assert rc == 1


def test_logout_removes_saved_pat(monkeypatch, tmp_path, capsys):
    from nexus.utils.artifact import auth as auth_module

    key_file = tmp_path / 'key'
    key_file.write_text('tai_pat_x', encoding='utf-8')
    monkeypatch.setattr(auth_module, 'ARTIFACT_AUTH_KEY_FILE', key_file)

    rc = artifact_logout(argparse.Namespace(json=False))

    assert rc == 0
    assert not key_file.exists()
    assert 'Removed' in capsys.readouterr().out


def test_logout_when_no_saved_pat(monkeypatch, tmp_path, capsys):
    from nexus.utils.artifact import auth as auth_module

    monkeypatch.setattr(auth_module, 'ARTIFACT_AUTH_KEY_FILE', tmp_path / 'key')

    rc = artifact_logout(argparse.Namespace(json=False))

    assert rc == 0
    assert 'No saved' in capsys.readouterr().out


@pytest.mark.parametrize(
    'legacy_args',
    [
        ['--token', 'tok'],
        ['--cmk-file', '/tmp/key'],
        ['--user', 'alice'],
        ['--proxy-user', 'bob'],
        ['--region', 'sg'],
        ['--target', 'artifact-service'],
    ],
)
def test_register_parser_rejects_legacy_tauth_flags(legacy_args):
    parser = argparse.ArgumentParser(prog='nexus')
    subparsers = parser.add_subparsers(dest='command')
    register_artifact_parser(subparsers)

    with pytest.raises(SystemExit):
        parser.parse_args(['artifact', 'ls', *legacy_args])


# --- commands ---------------------------------------------------------------


def test_mkdir_creates_under_resolved_parent(fake_client, capsys):
    rc = artifact_mkdir(_ns(path='/models/checkpoints'))
    assert rc == 0
    # Prints the path, not an id.
    assert capsys.readouterr().out.strip() == '/models/checkpoints'
    # Parent /models resolved to fld_1, leaf name is checkpoints.
    assert ('create_folder', 'checkpoints', 'fld_1') in fake_client.last.calls


def test_mkdir_top_level_has_no_parent(fake_client, capsys):
    rc = artifact_mkdir(_ns(path='/models'))
    assert rc == 0
    assert capsys.readouterr().out.strip() == '/models'
    assert ('create_folder', 'models', None) in fake_client.last.calls


def test_ls_human_output_names_only(fake_client, capsys):
    rc = artifact_ls(_ns(target='/models', limit=50, cursor=None))
    assert rc == 0
    out = capsys.readouterr().out
    # Names only, folders get a trailing slash; no ids.
    assert 'models/' in out
    assert 'model.bin' in out
    assert 'fld_1' not in out
    assert '# next_cursor: cursor-2' in out
    # target path was resolved to a folder id before listing.
    assert ('list_resources', 'fld_1', None, 50, None) in fake_client.last.calls


def test_ls_subfolder_resolves_and_lists_children(fake_client):
    """A path (or URI with segments) resolves to a folder id and lists its children."""
    rc = artifact_ls(_ns(target='/models', limit=50, cursor=None))
    assert rc == 0
    # Path was resolved to a folder id.
    assert ('resolve_resource_id', '/models', 'folder') in fake_client.last.calls
    # Child listing drops the view (parent_folder_id is present).
    assert ('list_resources', 'fld_1', None, 50, None) in fake_client.last.calls


def test_ls_shared_alias_requests_shared_view(fake_client):
    """``artifact:///`` triggers the aggregated shared listing."""
    rc = artifact_ls(_ns(target='artifact:///', limit=50, cursor=None))
    assert rc == 0
    # No path resolution -- the alias is list-only.
    assert 'resolve_resource_id' not in _kinds(fake_client.last.calls)
    assert ('list_resources', None, 'shared', 50, None) in fake_client.last.calls


def test_ls_shared_alias_prints_owner_prefixed_uri(fake_client, capsys):
    """Under artifact:///, same-named files from two grantors stay distinguishable as URIs."""
    rc = artifact_ls(_ns(target='artifact:///', limit=50, cursor=None))
    assert rc == 0
    out = capsys.readouterr().out
    assert 'artifact:///alice/test.sh' in out
    assert 'artifact:///bob/test.sh' in out


def test_ls_owner_scoped_uri_uses_shared_view(fake_client):
    """``artifact:///<other>/`` navigates that owner's namespace via the shared view."""
    rc = artifact_ls(_ns(target='artifact:///alice/', limit=50, cursor=None))
    assert rc == 0
    # Compares URI owner to whoami() to decide the view.
    assert ('whoami',) in fake_client.last.calls
    # Server call uses the shared view; owner filter is applied client-side.
    assert ('list_resources', None, 'shared', 50, None) in fake_client.last.calls


def test_ls_self_uri_uses_owned_view(fake_client):
    """``artifact:///<self>/`` normalises to the owned view without a shared query."""
    rc = artifact_ls(_ns(target='artifact:///self/', limit=50, cursor=None))
    assert rc == 0
    assert ('list_resources', None, None, 50, None) in fake_client.last.calls


def test_ls_show_id_appends_ids(fake_client, capsys):
    rc = artifact_ls(_ns(target=None, limit=50, cursor=None, show_id=True))
    assert rc == 0
    out = capsys.readouterr().out
    # Own items are shown as bare names with id appended
    assert 'models/\tfld_1' in out
    assert 'model.bin\tart_1' in out
    # Shared item is shown as URI with id appended
    assert 'artifact:///alice/test.sh\tart_a' in out


def test_ls_top_level_does_not_resolve(fake_client):
    rc = artifact_ls(_ns(target=None, limit=50, cursor=None))
    assert rc == 0
    # No path resolution when listing the top level.
    assert 'resolve_resource_id' not in _kinds(fake_client.last.calls)
    assert ('list_resources', None, 'all', 50, None) in fake_client.last.calls


def test_ls_no_args_uses_all_view(fake_client):
    """No-arg ls requests view=all from the server."""
    rc = artifact_ls(_ns(target=None, limit=20, cursor=None))
    assert rc == 0
    assert ('list_resources', None, 'all', 20, None) in fake_client.last.calls


def test_ls_no_args_own_items_as_bare_name(fake_client, capsys):
    """Own items in view=all are printed as bare names (folder gets trailing/)."""
    rc = artifact_ls(_ns(target=None, limit=50, cursor=None))
    assert rc == 0
    out = capsys.readouterr().out
    assert 'models/' in out
    assert 'model.bin' in out
    # Own items must NOT be printed as URIs
    assert 'artifact:///self/models' not in out
    assert 'artifact:///self/model.bin' not in out


def test_ls_no_args_shared_items_as_uri(fake_client, capsys):
    """Shared items in view=all are printed as full artifact:/// URIs."""
    rc = artifact_ls(_ns(target=None, limit=50, cursor=None))
    assert rc == 0
    out = capsys.readouterr().out
    assert 'artifact:///alice/test.sh' in out


def test_ls_no_args_json_uses_per_item_owner_id(fake_client, capsys):
    """JSON output for view=all computes URI from per-item owner_id."""
    rc = artifact_ls(_ns(target=None, limit=50, cursor=None, json=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    uris = {item['uri'] for item in payload['items'] if item.get('uri')}
    # Own items use'self' as owner in URI
    assert 'artifact:///self/models/' in uris or any('self' in u for u in uris)
    # Shared item uses 'alice' as owner in URI
    assert 'artifact:///alice/test.sh' in uris


def test_ls_self_uri_uses_owned_view(fake_client):
    """artifact:///~ triggers view=None (own top-level listing, not shared)."""
    rc = artifact_ls(_ns(target='artifact:///~', limit=50, cursor=None))
    assert rc == 0
    # Must NOT call list_resources with view='shared'
    calls = fake_client.last.calls
    list_calls = [c for c in calls if c[0] == 'list_resources']
    assert list_calls, 'list_resources must have been called'
    for call in list_calls:
        _name, parent_folder_id, view, *_ = call
        assert view != 'shared', "artifact:///~ must not use view='shared'"


def test_ls_self_uri_does_not_use_all_view(fake_client):
    """artifact:///~ shows only the caller's own resources (view=None), not view=all."""
    rc = artifact_ls(_ns(target='artifact:///~', limit=50, cursor=None))
    assert rc == 0
    calls = fake_client.last.calls
    list_calls = [c for c in calls if c[0] == 'list_resources']
    views = [c[2] for c in list_calls]
    assert 'all' not in views


def test_ls_self_uri_shows_bare_names(fake_client, capsys):
    """Items returned for artifact:///~ are displayed as bare names (own namespace)."""
    rc = artifact_ls(_ns(target='artifact:///~', limit=50, cursor=None))
    assert rc == 0
    out = capsys.readouterr().out
    # Must contain bare folder/file names, not full URIs
    assert 'models/' in out or 'model.bin' in out
    assert 'artifact:///self/' not in out


def test_ls_self_uri_resolves_without_knowing_login(fake_client):
    """artifact:///~ works even though no explicit login is embedded in the URI."""
    # The fake_client returns 'self' from whoami(); the test verifies that
    # artifact:///~ triggers a whoami() lookup and then lists own resources.
    rc = artifact_ls(_ns(target='artifact:///~', limit=50, cursor=None))
    assert rc == 0
    whoami_calls = [c for c in fake_client.last.calls if c[0] == 'whoami']
    assert whoami_calls, 'whoami() must be called to resolve ~'


def test_ls_json_output(fake_client, capsys):
    rc = artifact_ls(_ns(target='artifact:///', limit=5, cursor='c1', json=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    # Aggregated shared listing returns the two same-named files with
    # per-owner computed URIs so callers can address either one downstream.
    uris = sorted(item['uri'] for item in payload['items'])
    assert uris == ['artifact:///alice/test.sh', 'artifact:///bob/test.sh']


def test_artifacts_human_output_names_only(fake_client, capsys):
    from nexus.cli.commands.artifact import artifact_artifacts

    rc = artifact_artifacts(_ns(target=None, limit=50, cursor=None))
    assert rc == 0
    out = capsys.readouterr().out
    assert 'model.bin\tready\t3' in out
    assert 'art_1' not in out


def test_artifacts_json_omits_fabricated_uri(fake_client, capsys):
    """Flat ``artifacts --json`` must not emit a path-based ``uri``.

    The rows carry no ancestor path, so an ``artifact:///<owner>/<name>`` URI
    built from an empty path segment would be fabricated (it may not exist, or
    may resolve to a different same-named top-level file). We expose the stable
    ``artifact_id`` instead and never a guessed URI.
    """
    from nexus.cli.commands.artifact import artifact_artifacts

    rc = artifact_artifacts(_ns(target=None, limit=50, cursor=None, json=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['next_cursor'] == 'cursor-3'
    assert [item['artifact_id'] for item in payload['items']] == ['art_1']
    # No row may carry a computed/fabricated URI.
    assert all('uri' not in item for item in payload['items'])


def test_artifacts_json_shared_view_omits_fabricated_uri(fake_client, capsys):
    """Even the aggregated shared flat listing must not fabricate per-owner URIs."""
    from nexus.cli.commands.artifact import artifact_artifacts

    rc = artifact_artifacts(_ns(target='artifact:///', limit=50, cursor=None, json=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert {item['artifact_id'] for item in payload['items']} == {'art_a', 'art_b'}
    assert all('uri' not in item for item in payload['items'])


def test_upload_prints_name_and_resolves_folder(fake_client, capsys):
    rc = artifact_upload(_ns(path='./model.bin', folder='/models', name=None, content_type=None, pool=None))
    assert rc == 0
    assert capsys.readouterr().out.strip().endswith('model.bin')
    call = [c for c in fake_client.last.calls if c[0] == 'upload_file'][0]
    # parent_folder_id resolved from /models -> fld_1
    assert call[3] == 'fld_1'


def test_upload_to_root_when_no_folder(fake_client):
    rc = artifact_upload(_ns(path='./model.bin', folder=None, name=None, content_type=None, pool=None))
    assert rc == 0
    call = [c for c in fake_client.last.calls if c[0] == 'upload_file'][0]
    assert call[3] is None


def test_upload_large_delegates_with_multipart(fake_client, capsys):
    rc = artifact_upload_large(
        _ns(path='./big.bin', folder='/models', name=None, content_type=None, pool=None, concurrency=8)
    )
    assert rc == 0
    call = [c for c in fake_client.last.calls if c[0] == 'upload_file'][0]
    assert call[3] == 'fld_1'
    assert call[6] is True  # multipart
    assert call[7] == 8  # max_concurrency


def test_register_parser_parses_upload_large():
    parser = argparse.ArgumentParser(prog='nexus')
    subparsers = parser.add_subparsers(dest='command')
    register_artifact_parser(subparsers)

    args = parser.parse_args(['artifact', 'upload-large', './big.bin', '--folder', '/models', '--concurrency', '4'])
    assert args.artifact_command == 'upload-large'
    assert args.path == './big.bin' and args.folder == '/models' and args.concurrency == 4

    args = parser.parse_args(['artifact', 'upload-large', './big.bin'])
    assert args.concurrency == 5


def test_grant_resolves_path(fake_client):
    rc = artifact_grant(_ns(path='/models', grantee_id='alice', type='employee', permission='read', expires=None))
    assert rc == 0
    assert ('put_grant', 'fld_1', 'alice', 'employee', 'read', None) in fake_client.last.calls


def test_upload_json_error_keeps_stdout_clean(fake_client, capsys, monkeypatch):
    async def fail_upload(*args, **kwargs):
        raise RuntimeError('upload failed')

    monkeypatch.setattr(fake_client, 'upload_file', fail_upload)
    rc = artifact_upload(_ns(path='x.bin', folder=None, name=None, content_type=None, pool=None, json=True))

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ''
    assert captured.err.strip() == 'upload failed'


def test_grant_json_outputs_machine_readable_result(fake_client, capsys):
    rc = artifact_grant(
        _ns(
            path='/models',
            grantee_id='alice',
            type='employee',
            permission='read',
            expires='2026-08-01T00:00:00+08:00',
            json=True,
        )
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == {
        'status': 'granted',
        'path': '/models',
        'grantee_id': 'alice',
        'grantee_type': 'employee',
        'permission': 'read',
        'expires_at': '2026-08-01T00:00:00+08:00',
    }


def test_revoke_resolves_path(fake_client):
    rc = artifact_revoke(_ns(path='/models', grantee_id='alice', type='employee', permission='read'))
    assert rc == 0
    assert ('delete_grant', 'fld_1', 'alice', 'employee', 'read') in fake_client.last.calls


def test_grants_resolves_path(fake_client, capsys):
    rc = artifact_grants(_ns(path='/models'))
    assert rc == 0
    assert 'read\temployee:alice' in capsys.readouterr().out
    assert ('list_grants', 'fld_1') in fake_client.last.calls


def test_url_resolves_artifact_path(fake_client, capsys):
    rc = artifact_url(_ns(path='/models/model.bin'))
    assert rc == 0
    assert capsys.readouterr().out.strip() == 'http://cos/get'
    assert ('create_download_url', 'art_1') in fake_client.last.calls


def test_rm_resolves_artifact_path(fake_client):
    rc = artifact_rm(_ns(path='/models/model.bin'))
    assert rc == 0
    assert ('delete_artifact', 'art_1') in fake_client.last.calls


def test_download_derives_name_from_metadata(fake_client, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = artifact_download(_ns(path='/models/model.bin', output=None, concurrency=4))
    assert rc == 0
    printed = capsys.readouterr().out.strip()
    assert printed.endswith('model.bin')
    kinds = _kinds(fake_client.last.calls)
    # resolve_path returns the item dict with ``name`` already set, so we no
    # longer need a separate ``get_artifact`` round-trip to figure out the
    # local file name.
    assert 'resolve_path' in kinds
    assert 'get_artifact' not in kinds
    assert ('download_to_file', 'art_1', 'model.bin') in fake_client.last.calls


def test_download_missing_path_reports_error(fake_client, capsys):
    rc = artifact_download(_ns(path='/models/nope.bin', output=None, concurrency=4))
    assert rc == 1


def test_download_folder_mirrors_tree(fake_client, tmp_path, monkeypatch):
    """Auto-detect folder: walk the tree, download every leaf, mirror structure."""
    monkeypatch.chdir(tmp_path)
    rc = artifact_download(_ns(path='/models', output=str(tmp_path / 'out'), concurrency=2))
    assert rc == 0
    # Every artifact under /models should have been fetched into out/<tree>.
    assert (tmp_path / 'out' / 'model.bin').exists()
    kinds = _kinds(fake_client.last.calls)
    assert 'download_folder_to_dir' in kinds


def test_download_folder_defaults_output_to_folder_name(fake_client, tmp_path, monkeypatch):
    """Without``-o`` a folder download lands in ``./<folder-name>/``."""
    monkeypatch.chdir(tmp_path)
    rc = artifact_download(_ns(path='/models', output=None, concurrency=1))
    assert rc == 0
    assert (tmp_path / 'models' / 'model.bin').exists()


def test_download_folder_reports_failures_and_nonzero_rc(fake_client, tmp_path, monkeypatch, caplog):
    """Per-file failures don't abort the walk but do set a non-zero exit code."""
    fake_client.simulate_folder_failure = True
    caplog.set_level('ERROR', logger='nexus.cli.commands.artifact')
    monkeypatch.chdir(tmp_path)
    rc = artifact_download(_ns(path='/models', output=str(tmp_path / 'out'), concurrency=1))
    assert rc == 1
    assert any('download failed:' in rec.message for rec in caplog.records)


# --- publish ----------------------------------------------------------------
#
# ``publish`` creates a new timestamped folder and uploads every file under a
# local directory into it, recreating subdirectories, then optionally grants
# read to colleagues. The fake client's ``create_folder`` returns a name-based
# id so we can assert parent/child wiring.


def _fixed_naming(monkeypatch):
    from datetime import datetime, timezone

    from nexus.cli.commands import artifact as artifact_cli

    # Pin the random suffix and freeze datetime.now so both the timestamp in the
    # folder name and the date-hierarchy path(/tasks/2026/08/04/) are stable.
    _frozen = datetime(2026, 8, 4, 0, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(artifact_cli, '_utc_timestamp', lambda: '20260804-000000')
    monkeypatch.setattr(artifact_cli, '_short_suffix', lambda: 'abcdef')

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return _frozen

    monkeypatch.setattr(artifact_cli, 'datetime', _FrozenDatetime)


def test_publish_uploads_directory_tree(fake_client, tmp_path, monkeypatch, capsys):
    _fixed_naming(monkeypatch)
    (tmp_path / 'a.txt').write_text('a')
    sub = tmp_path / 'sub'
    sub.mkdir()
    (sub / 'b.txt').write_text('b')

    rc = artifact_publish(_ns(local_dir=str(tmp_path), task='mytask', grant=None))

    assert rc == 0
    calls = fake_client.last.calls
    # Date-hierarchy folders are created first: tasks / year / month / day.
    assert ('create_folder', 'tasks', None) in calls
    day_id = 'fld::04'  # fake client returns folder_id='fld::{name}'
    # Top folder is created under the day folder (not at root).
    assert ('create_folder', 'mytask-20260804-000000-abcdef', day_id) in calls
    top_id = 'fld::mytask-20260804-000000-abcdef'
    # The subdirectory is recreated under the top folder.
    assert ('create_folder', 'sub', top_id) in calls
    sub_id = 'fld::sub'
    # Each file is uploaded under the folder that mirrors its local directory.
    parents = {c[2]: c[3] for c in calls if c[0] == 'upload_file'}  # name -> parent_folder_id
    assert parents == {'a.txt': top_id, 'b.txt': sub_id}
    out = capsys.readouterr().out
    assert '/tasks/2026/08/04/mytask-20260804-000000-abcdef' in out
    assert 'a.txt' in out
    assert 'sub/b.txt' in out


def test_scan_files_skips_symlink_to_outside_file(tmp_path):
    """A symlink pointing outside the publish root must not be collected."""
    from nexus.cli.commands.artifact import _scan_files

    secret = tmp_path / 'secret.txt'
    secret.write_text('TOP-SECRET-HOST-FILE')
    pub = tmp_path / 'pub'
    pub.mkdir()
    (pub / 'real.txt').write_text('real')
    # Agent-controlled output dir plants a symlink to a host file outside it.
    (pub / 'result').symlink_to(secret)

    scanned = _scan_files(pub)

    rels = sorted(rel.as_posix() for _abs, rel in scanned)
    assert rels == ['real.txt']
    # The out-of-tree target is never referenced by any collected entry.
    assert all('secret.txt' not in str(abs_path) for abs_path, _rel in scanned)


def test_scan_files_skips_symlinked_directory(tmp_path):
    """A symlinked subdirectory must not be descended into."""
    from nexus.cli.commands.artifact import _scan_files

    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'leak.txt').write_text('leak')
    pub = tmp_path / 'pub'
    pub.mkdir()
    (pub / 'keep.txt').write_text('keep')
    (pub / 'linkdir').symlink_to(outside, target_is_directory=True)

    scanned = _scan_files(pub)

    rels = sorted(rel.as_posix() for _abs, rel in scanned)
    assert rels == ['keep.txt']


def test_publish_uploads_no_follow(fake_client, tmp_path, monkeypatch):
    """publish must open each file no-follow (follow_symlinks=False)."""
    _fixed_naming(monkeypatch)
    (tmp_path / 'a.txt').write_text('a')

    rc = artifact_publish(_ns(local_dir=str(tmp_path), task='t', grant=None))

    assert rc == 0
    upload_calls = [c for c in fake_client.last.calls if c[0] == 'upload_file']
    assert upload_calls, 'expected at least one upload_file call'
    # The trailing tuple element is follow_symlinks; every publish upload is no-follow.
    assert all(call[8] is False for call in upload_calls)


def test_publish_creates_date_hierarchy(fake_client, tmp_path, monkeypatch):
    """publish creates /tasks/YYYY/MM/DD before the task folder."""
    _fixed_naming(monkeypatch)
    (tmp_path / 'x.txt').write_text('data')

    rc = artifact_publish(_ns(local_dir=str(tmp_path), task='job', grant=None))

    assert rc == 0
    calls = fake_client.last.calls
    folder_calls = [(c[1], c[2]) for c in calls if c[0] == 'create_folder']
    # First four folders build the date path: tasks / 2026 / 08 / 04
    assert folder_calls[:4] == [
        ('tasks', None),
        ('2026', 'fld::tasks'),
        ('08', 'fld::2026'),
        ('04', 'fld::08'),
    ]


def test_publish_folder_path_includes_date(fake_client, tmp_path, monkeypatch, capsys):
    """Output path includes /tasks/YYYY/MM/DD prefix."""
    _fixed_naming(monkeypatch)
    (tmp_path / 'x.txt').write_text('data')

    rc = artifact_publish(_ns(local_dir=str(tmp_path), task='job', grant=None))

    assert rc == 0
    out = capsys.readouterr().out
    assert '/tasks/2026/08/04/job-20260804-000000-abcdef' in out


def test_publish_defaults_folder_label_to_dir_name(fake_client, tmp_path, monkeypatch):
    _fixed_naming(monkeypatch)
    run_dir = tmp_path / 'runout'
    run_dir.mkdir()
    (run_dir / 'result.json').write_text('{}')

    rc = artifact_publish(_ns(local_dir=str(run_dir), task=None, grant=None))

    assert rc == 0
    # Folder is placed under the day directory, not at root (parent_folder_id != None).
    calls = fake_client.last.calls
    task_folder = [c for c in calls if c[0] == 'create_folder' and c[1] == 'runout-20260804-000000-abcdef']
    assert len(task_folder) == 1
    assert task_folder[0][2] is not None  # parent is the day folder, not root


def test_publish_grants_read_to_each_employee(fake_client, tmp_path, monkeypatch):
    _fixed_naming(monkeypatch)
    (tmp_path / 'a.txt').write_text('a')

    rc = artifact_publish(_ns(local_dir=str(tmp_path), task='t', grant='alice, bob,alice'))

    assert rc == 0
    top_id = 'fld::t-20260804-000000-abcdef'
    grants = [c for c in fake_client.last.calls if c[0] == 'put_grant']
    # Deduplicated, order preserved, all read on the top folder.
    assert grants == [
        ('put_grant', top_id, 'alice', 'employee', 'read', None),
        ('put_grant', top_id, 'bob', 'employee', 'read', None),
    ]


def test_publish_json_output(fake_client, tmp_path, monkeypatch, capsys):
    _fixed_naming(monkeypatch)
    (tmp_path / 'a.txt').write_text('a')

    rc = artifact_publish(_ns(local_dir=str(tmp_path), task='t', grant='alice', json=True))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['folder_path'] == '/tasks/2026/08/04/t-20260804-000000-abcdef'
    assert payload['uploaded'][0]['path'] == 'a.txt'
    assert payload['grants'] == [{'grantee_id': 'alice', 'permission': 'read'}]


def test_publish_folder_name_appends_random_suffix(tmp_path, monkeypatch):
    from nexus.cli.commands import artifact as artifact_cli

    # Only pin the timestamp; the random suffix stays real so we can assert its shape.
    monkeypatch.setattr(artifact_cli, '_utc_timestamp', lambda: '20260804-000000')
    name = artifact_cli._build_folder_name(tmp_path, 'run')

    assert name.startswith('run-20260804-000000-')
    suffix = name.rsplit('-', 1)[-1]
    assert len(suffix) == 6
    assert all(ch in '0123456789abcdef' for ch in suffix)


def test_publish_skips_empty_files(fake_client, tmp_path, monkeypatch, capsys):
    _fixed_naming(monkeypatch)
    (tmp_path / 'good.txt').write_text('data')
    (tmp_path / 'empty.md').write_text('')  # 0 bytes -> rejected by the service

    rc = artifact_publish(_ns(local_dir=str(tmp_path), task='t', grant=None, json=True))

    assert rc == 0
    calls = fake_client.last.calls
    uploaded_names = [c[2] for c in calls if c[0] == 'upload_file']
    # Only the non-empty file is uploaded; the 0-byte file is skipped.
    assert uploaded_names == ['good.txt']
    payload = json.loads(capsys.readouterr().out)
    assert payload['skipped_empty'] == ['empty.md']
    assert [u['path'] for u in payload['uploaded']] == ['good.txt']


def test_publish_all_empty_files_errors(fake_client, tmp_path):
    (tmp_path / 'a.md').write_text('')
    (tmp_path / 'b.md').write_text('')

    rc = artifact_publish(_ns(local_dir=str(tmp_path), task=None, grant=None))

    assert rc == 1
    # No folder is created when every candidate file is empty.
    assert fake_client.last is None


def test_publish_empty_directory_errors(fake_client, tmp_path):
    rc = artifact_publish(_ns(local_dir=str(tmp_path), task=None, grant=None))
    assert rc == 1
    # No client work is attempted when there is nothing to publish.
    assert fake_client.last is None


def test_publish_missing_directory_errors(fake_client, tmp_path):
    rc = artifact_publish(_ns(local_dir=str(tmp_path / 'nope'), task=None, grant=None))
    assert rc == 1
    assert fake_client.last is None


def test_publish_rejects_file_argument(fake_client, tmp_path):
    target = tmp_path / 'a.txt'
    target.write_text('a')
    rc = artifact_publish(_ns(local_dir=str(target), task=None, grant=None))
    assert rc == 1
    assert fake_client.last is None


def test_register_parser_parses_publish():
    parser = argparse.ArgumentParser(prog='nexus')
    subparsers = parser.add_subparsers(dest='command')
    register_artifact_parser(subparsers)

    args = parser.parse_args(['artifact', 'publish', './out', '--task', 'demo', '--grant', 'a,b', '--json'])
    assert args.artifact_command == 'publish'
    assert args.local_dir == './out'
    assert args.task == 'demo'
    assert args.grant == 'a,b'
    assert args.json

    # Only the directory is required; label/grant are optional.
    args = parser.parse_args(['artifact', 'publish', './out'])
    assert args.local_dir == './out'
    assert args.task is None
    assert args.grant is None


# --- shared-with-me addressing (download / url / info / rm) ------------------
#
# The four read/delete commands each resolve a single artifact path or URI.
# ``--view`` no longer exists: the view is inferred from the URI (``artifact://
# /<other>/...`` picks the shared view under the hood, bare paths and
# ``artifact:///<self>/...`` pick owned). What the CLI must do is forward the
# reference *unchanged* to ``resolve_resource_id`` -- the SDK's URI-aware
# resolver already handles view selection.


def test_download_forwards_uri_unchanged(fake_client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = artifact_download(_ns(path='/models/model.bin', output=None, concurrency=4))
    assert rc == 0
    # ``download`` now inspects the resolved item to auto-detect folder vs.
    # artifact, so it calls ``resolve_path`` (returns the whole dict) rather
    # than the id-only variant.
    kinds = _kinds(fake_client.last.calls)
    assert 'resolve_path' in kinds
    assert ('download_to_file', 'art_1', 'model.bin') in fake_client.last.calls


def test_url_forwards_uri_unchanged(fake_client, capsys):
    rc = artifact_url(_ns(path='/models/model.bin'))
    assert rc == 0
    assert ('resolve_resource_id', '/models/model.bin', 'artifact') in fake_client.last.calls


def test_info_forwards_uri_unchanged(fake_client, capsys):
    from nexus.cli.commands.artifact import artifact_info

    rc = artifact_info(_ns(path='/models/model.bin'))
    assert rc == 0
    assert ('resolve_resource_id', '/models/model.bin', 'artifact') in fake_client.last.calls


def test_rm_forwards_uri_unchanged(fake_client):
    rc = artifact_rm(_ns(path='/models/model.bin'))
    assert rc == 0
    assert ('resolve_resource_id', '/models/model.bin', 'artifact') in fake_client.last.calls
    assert ('delete_artifact', 'art_1') in fake_client.last.calls


def test_rm_rejects_uri_for_other_owner(fake_client, caplog):
    """``rm`` refuses cross-owner URIs client-side (writes never target grantees)."""
    caplog.set_level('ERROR', logger='nexus.cli.commands.artifact')
    rc = artifact_rm(_ns(path='artifact:///alice/models/model.bin'))
    assert rc == 1
    # Never even reached the resolver: short-circuited on the owner check.
    assert 'resolve_resource_id' not in _kinds(fake_client.last.calls)
    assert any('owned by another employee' in rec.message for rec in caplog.records)


def test_register_parser_read_commands_have_no_view_flag():
    """``--view`` is gone; the URI in``path`` carries any owner information."""
    parser = argparse.ArgumentParser(prog='nexus')
    subparsers = parser.add_subparsers(dest='command')
    register_artifact_parser(subparsers)

    for sub in ('download', 'url', 'info', 'rm'):
        args = parser.parse_args(['artifact', sub, 'artifact:///alice/models/model.bin'])
        assert args.artifact_command == sub
        assert args.path == 'artifact:///alice/models/model.bin'
        # No ``view`` attribute is set: the flag was removed.
        assert not hasattr(args, 'view')
        # ``--view`` is now rejected outright.
        with pytest.raises(SystemExit):
            parser.parse_args(['artifact', sub, '/x', '--view', 'shared'])


def test_handle_unknown_command_prints_help(fake_client, capsys):
    rc = handle_artifact_command(_ns(artifact_command='bogus'))
    assert rc == 0
    help_text = capsys.readouterr().out
    assert 'Artifact Commands' in help_text
    assert 'employee read access' in help_text
    assert 'application' not in help_text
    assert 'read/write' not in help_text


def test_command_reports_error_as_nonzero(monkeypatch):
    class _Boom:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def resolve_parent_folder_id(self, path):
            return None

        def leaf_name(self, path):
            return path

        async def create_folder(self, *args, **kwargs):
            raise RuntimeError('server down')

    monkeypatch.setattr('nexus.utils.artifact.client.ArtifactClient', _Boom)
    rc = artifact_mkdir(_ns(path='/x'))
    assert rc == 1
