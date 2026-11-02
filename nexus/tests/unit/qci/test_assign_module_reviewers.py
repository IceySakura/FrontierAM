from __future__ import annotations

import importlib.util
import sys

from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[3] / 'deployment' / 'qci' / 'assign_module_reviewers.py'
SPEC = importlib.util.spec_from_file_location('assign_module_reviewers', SCRIPT_PATH)
assert SPEC is not None
reviewers = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reviewers
assert SPEC.loader is not None
SPEC.loader.exec_module(reviewers)


def test_pyproject_changes_are_not_reviewable():
    old = b'[project]\nname = "demo"\nauthors = [{ name = "Alice" }]\n'
    new = b'[project]\nname = "demo"\nauthors = [{ name = "Bob" }]\n'

    assert not reviewers.has_reviewable_change('nexus/extensions/tasks/demo/pyproject.toml', old, None, new)


def test_pep723_metadata_changes_are_not_reviewable():
    old = b"""# /// script
# dependencies = []
#
# [tool.nexus]
# tasks = { demo = "nexus.extensions.tasks.demo" }
# ///

print("same")
"""
    new = b"""# /// script
# dependencies = []
#
# [project]
# name = "demo"
# authors = [{ name = "Alice", email = "alice@example.com" }]
#
# [tool.nexus]
# tasks = { demo = "nexus.extensions.tasks.demo" }
# ///

print("same")
"""

    assert not reviewers.has_reviewable_change('nexus/extensions/tasks/demo.py', old, None, new)


def test_modified_pep723_metadata_change_is_not_a_changed_module(tmp_path):
    path = 'nexus/extensions/tasks/demo.py'
    old = b"""# /// script
# dependencies = []
#
# [tool.nexus]
# tasks = { demo = "nexus.extensions.tasks.demo" }
# ///

print("same")
"""
    new = b"""# /// script
# dependencies = []
#
# [project]
# name = "demo"
# authors = [{ name = "Alice", email = "alice@example.com" }]
#
# [tool.nexus]
# tasks = { demo = "nexus.extensions.tasks.demo" }
# ///

print("same")
"""
    module_file = tmp_path / path
    module_file.parent.mkdir(parents=True)
    module_file.write_bytes(new)

    modules = reviewers.discover_modules(tmp_path)
    changed = reviewers.changed_modules_for_entries(
        [reviewers.DiffEntry(status='M', old_path=None, new_path=path)],
        modules,
        lambda requested_path: old if requested_path == path else None,
        lambda requested_path: new if requested_path == path else None,
    )

    assert changed == []


def test_pep723_code_changes_are_reviewable():
    old = b"""# /// script
# dependencies = []
#
# [tool.nexus]
# tasks = { demo = "nexus.extensions.tasks.demo" }
# ///

print("old")
"""
    new = b"""# /// script
# dependencies = []
#
# [tool.nexus]
# tasks = { demo = "nexus.extensions.tasks.demo" }
# ///

print("new")
"""

    assert reviewers.has_reviewable_change('nexus/extensions/tasks/demo.py', old, None, new)


def test_rename_only_is_not_reviewable():
    content = b'print("same")\n'

    assert not reviewers.has_reviewable_change('old/path.py', content, 'new/path.py', content)


def test_detects_qci_mr_id_environment(monkeypatch):
    monkeypatch.setenv('QCI_MR_ID', '25069348')

    assert reviewers.detect_mr_id() == '25069348'


def test_reviewer_ids_accept_gongfeng_review_shapes():
    values = [
        {'id': 149538899, 'userId': 335821, 'user': {'id': 335821}},
        {'id': 148664, 'username': 'baynejiang'},
        {'user': {'id': 15027}},
    ]

    assert reviewers._reviewer_ids(values) == {335821, 148664, 15027}


def test_minimal_toml_fallback_parses_module_people():
    data = reviewers._load_minimal_module_toml(
        b"""[project]
name = "demo"
authors = [
    { name = "Alice", email = "alice@example.com" },
]

[tool.nexus.tasks]
demo = "nexus.extensions.tasks.demo"
"""
    )

    assert data['project']['name'] == 'demo'
    assert data['project']['authors'] == [{'name': 'Alice', 'email': 'alice@example.com'}]
    assert data['tool']['nexus']


class FakeGongfengClient:
    def __init__(self, users: dict[str, dict], mr: dict | None = None, notes: list[dict] | None = None) -> None:
        self.users = users
        self.mr = mr or {}
        self.notes = notes or []
        self.added_reviewers = []
        self.dismissed_reviewers = []
        self.created_notes = []
        self.updated_notes = []

    def get_merge_request(self, _project: str, _mr_id: str) -> dict:
        return self.mr

    def get_user(self, username: str) -> dict | None:
        return self.users.get(username)

    def add_necessary_reviewers(self, _project: str, _mr_id: str, reviewer_ids: list[int]) -> dict:
        self.added_reviewers.append(list(reviewer_ids))
        reviewers_list = self.mr.setdefault('necessary_reviewers', [])
        existing_ids = reviewers._reviewer_ids(reviewers_list)
        for reviewer_id in reviewer_ids:
            if reviewer_id not in existing_ids:
                reviewers_list.append({'userId': reviewer_id})
        return {}

    def dismiss_reviewer(self, _project: str, _mr_id: str, reviewer_id: int) -> dict:
        self.dismissed_reviewers.append(reviewer_id)
        self.mr['necessary_reviewers'] = [
            item for item in self.mr.get('necessary_reviewers', []) if reviewers._reviewer_id(item) != reviewer_id
        ]
        return {}

    def list_merge_request_notes(self, _project: str, _mr_id: str) -> list[dict]:
        return self.notes

    def create_merge_request_note(self, _project: str, _mr_id: str, body: str) -> dict:
        note = {'id': len(self.notes) + 1, 'body': body}
        self.notes.insert(0, note)
        self.created_notes.append(body)
        return note

    def update_merge_request_note(self, _project: str, _mr_id: str, note_id: int, body: str) -> dict:
        self.updated_notes.append((note_id, body))
        for note in self.notes:
            if note.get('id') == note_id:
                note['body'] = body
                return note
        return {}


class CapturingGongfengClient(reviewers.GongfengClient):
    def __init__(self) -> None:
        super().__init__(api_url='https://git.example.com/api/v3', access_token='token', oauth_token=None)
        self.requests = []

    def request_json(self, method, path, body=None):
        self.requests.append((method, path, body))
        return {'ok': True}


def test_single_module_candidates_prefer_maintainers(tmp_path):
    module_dir = tmp_path / 'nexus' / 'extensions' / 'tasks' / 'demo'
    module_dir.mkdir(parents=True)
    (module_dir / 'pyproject.toml').write_text(
        '[project]\n'
        'name = "demo"\n'
        'authors = [{ name = "Alice", email = "alice@example.com" }]\n'
        'maintainers = [{ name = "Bob", email = "bob@example.com" }]\n'
        '\n'
        '[tool.nexus]\n'
        'tasks = { demo = "nexus.extensions.tasks.demo" }\n',
        encoding='utf-8',
    )

    modules = reviewers.discover_modules(tmp_path)
    changed = reviewers.changed_modules_for_entries(
        [reviewers.DiffEntry(status='M', old_path=None, new_path='nexus/extensions/tasks/demo/task.py')],
        modules,
        lambda path: b'print("old")\n',
        lambda path: b'print("new")\n',
    )
    candidates = reviewers.reviewer_candidates(changed)

    assert [item.module.name for item in changed] == ['demo']
    assert [(item.username, item.modules) for item in candidates] == [
        ('bob', ('demo',)),
    ]


def test_discover_modules_includes_exact_top_level_module_dir(tmp_path):
    module_dir = tmp_path / 'nexus' / 'extensions' / 'harbor_bridge'
    module_dir.mkdir(parents=True)
    (module_dir / 'pyproject.toml').write_text(
        '[project]\n'
        'name = "harbor_bridge"\n'
        'authors = [{ name = "Alice", email = "alice@example.com" }]\n'
        '\n'
        '[tool.nexus]\n'
        'dataset_sources = { t3-harbor = "nexus.extensions.harbor_bridge.dataset_source" }\n',
        encoding='utf-8',
    )

    modules = reviewers.discover_modules(tmp_path)

    assert [(module.name, module.rel_path) for module in modules] == [
        ('harbor_bridge', 'nexus/extensions/harbor_bridge')
    ]


def test_single_module_candidates_fall_back_to_authors(tmp_path):
    module_dir = tmp_path / 'nexus' / 'extensions' / 'tasks' / 'demo'
    module_dir.mkdir(parents=True)
    (module_dir / 'pyproject.toml').write_text(
        '[project]\n'
        'name = "demo"\n'
        'authors = [{ name = "Alice", email = "alice@example.com" }]\n'
        '\n'
        '[tool.nexus]\n'
        'tasks = { demo = "nexus.extensions.tasks.demo" }\n',
        encoding='utf-8',
    )

    modules = reviewers.discover_modules(tmp_path)
    changed = reviewers.changed_modules_for_entries(
        [reviewers.DiffEntry(status='M', old_path=None, new_path='nexus/extensions/tasks/demo/task.py')],
        modules,
        lambda path: b'print("old")\n',
        lambda path: b'print("new")\n',
    )
    candidates = reviewers.reviewer_candidates(changed)

    assert [item.module.name for item in changed] == ['demo']
    assert [(item.username, item.modules) for item in candidates] == [
        ('alice', ('demo',)),
    ]


def test_new_module_candidates_do_not_fall_back_to_authors(tmp_path):
    module_dir = tmp_path / 'nexus' / 'extensions' / 'tasks' / 'demo'
    module_dir.mkdir(parents=True)
    (module_dir / 'pyproject.toml').write_text(
        '[project]\n'
        'name = "demo"\n'
        'authors = [{ name = "Alice", email = "alice@example.com" }]\n'
        '\n'
        '[tool.nexus]\n'
        'tasks = { demo = "nexus.extensions.tasks.demo" }\n',
        encoding='utf-8',
    )

    modules = reviewers.discover_modules(tmp_path)
    changed = reviewers.changed_modules_for_entries(
        [reviewers.DiffEntry(status='A', old_path=None, new_path='nexus/extensions/tasks/demo/task.py')],
        modules,
        lambda path: None,
        lambda path: b'print("new")\n',
    )
    candidates = reviewers.reviewer_candidates(changed)

    assert [item.module.name for item in changed] == ['demo']
    assert changed[0].is_new
    assert candidates == []


def test_new_module_candidates_use_maintainers(tmp_path):
    module_dir = tmp_path / 'nexus' / 'extensions' / 'tasks' / 'demo'
    module_dir.mkdir(parents=True)
    (module_dir / 'pyproject.toml').write_text(
        '[project]\n'
        'name = "demo"\n'
        'authors = [{ name = "Alice", email = "alice@example.com" }]\n'
        'maintainers = [{ name = "Bob", email = "bob@example.com" }]\n'
        '\n'
        '[tool.nexus]\n'
        'tasks = { demo = "nexus.extensions.tasks.demo" }\n',
        encoding='utf-8',
    )

    modules = reviewers.discover_modules(tmp_path)
    changed = reviewers.changed_modules_for_entries(
        [reviewers.DiffEntry(status='A', old_path=None, new_path='nexus/extensions/tasks/demo/task.py')],
        modules,
        lambda path: None,
        lambda path: b'print("new")\n',
    )
    candidates = reviewers.reviewer_candidates(changed)

    assert [item.module.name for item in changed] == ['demo']
    assert changed[0].is_new
    assert [(item.username, item.modules) for item in candidates] == [
        ('bob', ('demo',)),
    ]


def test_reviewer_candidates_skip_multiple_modules():
    first = reviewers.ModuleInfo(
        name='first',
        rel_path='nexus/extensions/tasks/first',
        kind='directory',
        authors=(reviewers.Person(name='alice'),),
    )
    second = reviewers.ModuleInfo(
        name='second',
        rel_path='nexus/extensions/tasks/second',
        kind='directory',
        authors=(reviewers.Person(name='bob'),),
    )
    changed = [
        reviewers.ChangedModule(module=first, paths=('nexus/extensions/tasks/first/task.py',)),
        reviewers.ChangedModule(module=second, paths=('nexus/extensions/tasks/second/task.py',)),
    ]

    assert reviewers.reviewer_candidates(changed) == []


def test_reviewer_candidates_skip_module_and_unmatched_path(tmp_path):
    module_dir = tmp_path / 'nexus' / 'extensions' / 'tasks' / 'demo'
    module_dir.mkdir(parents=True)
    (module_dir / 'pyproject.toml').write_text(
        '[project]\n'
        'name = "demo"\n'
        'authors = [{ name = "Alice", email = "alice@example.com" }]\n'
        '\n'
        '[tool.nexus]\n'
        'tasks = { demo = "nexus.extensions.tasks.demo" }\n',
        encoding='utf-8',
    )

    modules = reviewers.discover_modules(tmp_path)
    reviewable = reviewers.reviewable_changes_for_entries(
        [
            reviewers.DiffEntry(status='M', old_path=None, new_path='nexus/extensions/tasks/demo/task.py'),
            reviewers.DiffEntry(status='M', old_path=None, new_path='README.md'),
        ],
        modules,
        lambda path: b'old\n',
        lambda path: b'new\n',
    )
    candidates = reviewers.reviewer_candidates(
        list(reviewable.modules),
        unmatched_paths=reviewable.unmatched_paths,
    )

    assert [item.module.name for item in reviewable.modules] == ['demo']
    assert reviewable.unmatched_paths == ('README.md',)
    assert candidates == []


def test_resolve_necessary_reviewer_ids_promotes_suggestion_reviewer():
    client = FakeGongfengClient(
        {'bob': {'id': 2, 'state': 'active'}},
        {
            'author': {'id': 1, 'username': 'alice'},
            'suggestion_reviewers': [{'user': {'id': 2}}],
            'necessary_reviewers': [{'user': {'id': 3}}],
        },
    )
    candidates = [reviewers.ReviewerCandidate(username='bob', modules=('demo',))]

    desired_ids, added_ids, skipped = reviewers.resolve_necessary_reviewer_ids(
        client,
        'nexus-agent/nexus',
        '123',
        candidates,
    )

    assert desired_ids == [2]
    assert added_ids == [2]
    assert skipped == []


def test_resolve_necessary_reviewer_ids_allows_mr_author():
    client = FakeGongfengClient(
        {'bob': {'id': 2, 'state': 'active'}},
        {
            'author': {'id': 2, 'username': 'bob'},
            'necessary_reviewers': [],
        },
    )
    candidates = [reviewers.ReviewerCandidate(username='bob', modules=('demo',))]

    desired_ids, added_ids, skipped = reviewers.resolve_necessary_reviewer_ids(
        client,
        'nexus-agent/nexus',
        '123',
        candidates,
    )

    assert desired_ids == [2]
    assert added_ids == [2]
    assert skipped == []


def test_add_necessary_reviewers_uses_review_batch_invite():
    client = CapturingGongfengClient()

    assert client.add_necessary_reviewers('nexus-agent/nexus', '123', [3, 2]) == {'ok': True}
    assert client.requests == [
        (
            'POST',
            '/projects/nexus-agent%2Fnexus/merge_request/123/review/batch_invite',
            {'necessary_reviewer_ids': [3, 2]},
        )
    ]


def test_dismiss_reviewer_uses_review_dismissals():
    client = CapturingGongfengClient()

    assert client.dismiss_reviewer('nexus-agent/nexus', '123', 2) == {'ok': True}
    assert client.requests == [
        (
            'DELETE',
            '/projects/nexus-agent%2Fnexus/merge_request/123/review/dismissals?reviewer_id=2',
            None,
        )
    ]


def test_verify_necessary_reviewer_ids_accepts_added_reviewers():
    client = FakeGongfengClient(
        {},
        {
            'necessary_reviewers': [{'userId': 2}, {'user': {'id': 3}}],
        },
    )

    reviewers.verify_necessary_reviewer_ids(client, 'nexus-agent/nexus', '123', [2, 3])


def test_verify_necessary_reviewer_ids_fails_when_gongfeng_ignores_update():
    client = FakeGongfengClient(
        {},
        {
            'necessary_reviewers': [{'userId': 3}],
        },
    )

    with pytest.raises(reviewers.ReviewerAssignmentError, match=r'\[2\]'):
        reviewers.verify_necessary_reviewer_ids(client, 'nexus-agent/nexus', '123', [2])


def test_reconcile_adds_owner_and_records_marker():
    client = FakeGongfengClient({'bob': {'id': 2, 'state': 'active'}}, {'necessary_reviewers': []})

    result = reviewers.reconcile_necessary_reviewers(
        client,
        'nexus-agent/nexus',
        '123',
        [reviewers.ReviewerCandidate(username='bob', modules=('demo',))],
    )

    assert result.added_ids == (2,)
    assert result.removed_ids == ()
    assert result.managed_ids == (2,)
    assert client.added_reviewers == [[2]]
    assert client.created_notes
    assert reviewers.get_managed_reviewer_state(client, 'nexus-agent/nexus', '123').reviewer_ids == (2,)


def test_reconcile_removes_previous_managed_owner_without_candidates():
    marker = reviewers._managed_reviewer_note_body([2])
    client = FakeGongfengClient(
        {},
        {'necessary_reviewers': [{'userId': 2}, {'userId': 3}]},
        [{'id': 10, 'body': marker}],
    )

    result = reviewers.reconcile_necessary_reviewers(client, 'nexus-agent/nexus', '123', [])

    assert result.added_ids == ()
    assert result.removed_ids == (2,)
    assert result.managed_ids == ()
    assert client.dismissed_reviewers == [2]
    assert reviewers._reviewer_ids(client.mr['necessary_reviewers']) == {3}
    assert client.updated_notes
    assert reviewers.get_managed_reviewer_state(client, 'nexus-agent/nexus', '123').reviewer_ids == ()


def test_reconcile_drops_stale_marker_without_dismiss():
    marker = reviewers._managed_reviewer_note_body([2])
    client = FakeGongfengClient(
        {},
        {'necessary_reviewers': [{'userId': 3}]},
        [{'id': 10, 'body': marker}],
    )

    result = reviewers.reconcile_necessary_reviewers(client, 'nexus-agent/nexus', '123', [])

    assert result.removed_ids == ()
    assert result.managed_ids == ()
    assert client.dismissed_reviewers == []
    assert client.updated_notes
    assert reviewers.get_managed_reviewer_state(client, 'nexus-agent/nexus', '123').reviewer_ids == ()


def test_reconcile_switches_managed_owner_between_modules():
    marker = reviewers._managed_reviewer_note_body([2])
    client = FakeGongfengClient(
        {'carol': {'id': 4, 'state': 'active'}},
        {'necessary_reviewers': [{'userId': 2}]},
        [{'id': 10, 'body': marker}],
    )

    result = reviewers.reconcile_necessary_reviewers(
        client,
        'nexus-agent/nexus',
        '123',
        [reviewers.ReviewerCandidate(username='carol', modules=('other',))],
    )

    assert result.added_ids == (4,)
    assert result.removed_ids == (2,)
    assert result.managed_ids == (4,)
    assert client.dismissed_reviewers == [2]
    assert client.added_reviewers == [[4]]
    assert reviewers._reviewer_ids(client.mr['necessary_reviewers']) == {4}
    assert reviewers.get_managed_reviewer_state(client, 'nexus-agent/nexus', '123').reviewer_ids == (4,)


def test_reconcile_does_not_manage_preexisting_manual_reviewer():
    client = FakeGongfengClient(
        {'bob': {'id': 2, 'state': 'active'}},
        {'necessary_reviewers': [{'userId': 2}]},
    )

    result = reviewers.reconcile_necessary_reviewers(
        client,
        'nexus-agent/nexus',
        '123',
        [reviewers.ReviewerCandidate(username='bob', modules=('demo',))],
    )

    assert result.desired_ids == (2,)
    assert result.added_ids == ()
    assert result.managed_ids == ()
    assert client.created_notes == []
    assert client.dismissed_reviewers == []


def test_code_changed_rename_matches_both_modules(tmp_path):
    old_dir = tmp_path / 'nexus' / 'extensions' / 'tasks' / 'old_demo'
    new_dir = tmp_path / 'nexus' / 'extensions' / 'tasks' / 'new_demo'
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    (old_dir / 'pyproject.toml').write_text(
        '[project]\n'
        'name = "old_demo"\n'
        'authors = [{ name = "Alice", email = "alice@example.com" }]\n'
        '\n'
        '[tool.nexus]\n'
        'tasks = { old_demo = "nexus.extensions.tasks.old_demo" }\n',
        encoding='utf-8',
    )
    (new_dir / 'pyproject.toml').write_text(
        '[project]\n'
        'name = "new_demo"\n'
        'authors = [{ name = "Bob", email = "bob@example.com" }]\n'
        '\n'
        '[tool.nexus]\n'
        'tasks = { new_demo = "nexus.extensions.tasks.new_demo" }\n',
        encoding='utf-8',
    )

    modules = reviewers.discover_modules(tmp_path)
    changed = reviewers.changed_modules_for_entries(
        [
            reviewers.DiffEntry(
                status='R080',
                old_path='nexus/extensions/tasks/old_demo/task.py',
                new_path='nexus/extensions/tasks/new_demo/task.py',
            )
        ],
        modules,
        lambda path: b'print("old")\n',
        lambda path: b'print("new")\n',
    )

    assert [item.module.name for item in changed] == ['new_demo', 'old_demo']
