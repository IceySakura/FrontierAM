#!/usr/bin/env python3
"""Assign Gongfeng MR necessary reviewers from Nexus module ownership metadata.

The script is intended for a lightweight QCI MR-triggered job.  It only treats
actual code text changes as reviewable changes: ``pyproject.toml`` files are
ignored, and single-file PEP 723 metadata blocks are stripped before comparing
content.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODULE_ROOTS: tuple[str, ...] = (
    'extensions',
    'extensions/agents',
    'extensions/tasks',
    'extensions/tools',
    'extensions/projects',
    'extensions/openai_clients',
    'runtime/providers',
    'runtime/executors',
    'trace/processors',
)

DEFAULT_PROJECT = 'nexus-agent/nexus'
DEFAULT_API_URL = 'https://git.woa.com/api/v3'
MANAGED_REVIEWERS_MARKER = 'nexus-module-reviewers'
MANAGED_REVIEWERS_RE = re.compile(r'<!--\s*nexus-module-reviewers\s+(\{.*?\})\s*-->', re.DOTALL)
PEP723_RE = re.compile(r'^# /// script\s*\n((?:#[^\n]*\n)*?)# ///\s*\n?', re.MULTILINE)


@dataclass(frozen=True)
class Person:
    name: str = ''
    email: str = ''

    @property
    def username(self) -> str:
        if self.email and '@' in self.email:
            return self.email.split('@', 1)[0]
        return self.name


@dataclass(frozen=True)
class ModuleInfo:
    name: str
    rel_path: str
    kind: str
    authors: tuple[Person, ...]
    maintainers: tuple[Person, ...] = ()

    @property
    def owners(self) -> tuple[Person, ...]:
        return self.maintainers or self.authors


@dataclass(frozen=True)
class DiffEntry:
    status: str
    old_path: str | None
    new_path: str | None


@dataclass(frozen=True)
class ChangedModule:
    module: ModuleInfo
    paths: tuple[str, ...]
    is_new: bool = False


@dataclass(frozen=True)
class ReviewableChanges:
    modules: tuple[ChangedModule, ...]
    unmatched_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReviewerCandidate:
    username: str
    modules: tuple[str, ...]


@dataclass(frozen=True)
class ManagedReviewerState:
    note_id: int | None = None
    reviewer_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ReviewerReconcileResult:
    desired_ids: tuple[int, ...] = ()
    added_ids: tuple[int, ...] = ()
    removed_ids: tuple[int, ...] = ()
    managed_ids: tuple[int, ...] = ()
    skipped: tuple[str, ...] = ()


class ReviewerAssignmentError(RuntimeError):
    """Expected runtime failure for reviewer assignment."""


def _load_toml_bytes(data: bytes) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            return _load_minimal_module_toml(data)
    return tomllib.loads(data.decode('utf-8'))


def _load_minimal_module_toml(data: bytes) -> dict[str, Any]:
    """Parse just enough TOML for this script when tomllib/tomli is absent."""
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise ReviewerAssignmentError('Module metadata must be UTF-8 TOML') from exc

    result: dict[str, Any] = {}
    project_text = _section_text(text, 'project')
    if project_text:
        project: dict[str, Any] = {}
        name = _string_value(project_text, 'name')
        if name:
            project['name'] = name
        project['authors'] = _people_value(project_text, 'authors')
        project['maintainers'] = _people_value(project_text, 'maintainers')
        result['project'] = project

    if re.search(r'^\[tool\.nexus(?:\]|\.)', text, flags=re.MULTILINE):
        result['tool'] = {'nexus': {'_present': True}}
    return result


def _section_text(text: str, section: str) -> str:
    match = re.search(rf'^\[{re.escape(section)}\]\s*$', text, flags=re.MULTILINE)
    if not match:
        return ''
    start = match.end()
    next_section = re.search(r'^\[[^\]]+\]\s*$', text[start:], flags=re.MULTILINE)
    end = start + next_section.start() if next_section else len(text)
    return text[start:end]


def _string_value(text: str, key: str) -> str:
    match = re.search(rf'(?:^|[,\s]){re.escape(key)}\s*=\s*"([^"]*)"', text, flags=re.MULTILINE)
    return match.group(1) if match else ''


def _people_value(text: str, key: str) -> list[dict[str, str]]:
    match = re.search(rf'^\s*{re.escape(key)}\s*=\s*\[(.*?)\]', text, flags=re.MULTILINE | re.DOTALL)
    if not match:
        return []
    people: list[dict[str, str]] = []
    for raw_entry in re.findall(r'\{([^}]*)\}', match.group(1)):
        entry: dict[str, str] = {}
        for field in ('name', 'email'):
            value = _string_value(raw_entry, field)
            if value:
                entry[field] = value
        if entry:
            people.append(entry)
    return people


def _parse_pep723_block(source: bytes) -> dict[str, Any] | None:
    try:
        text = source.decode('utf-8')
    except UnicodeDecodeError:
        return None
    match = PEP723_RE.search(text)
    if not match:
        return None

    toml_lines: list[str] = []
    for line in match.group(1).splitlines():
        if line.startswith('# '):
            toml_lines.append(line[2:])
        elif line == '#':
            toml_lines.append('')
        else:
            toml_lines.append(line.lstrip('# '))
    return _load_toml_bytes('\n'.join(toml_lines).encode())


def _people(project: dict[str, Any], key: str) -> tuple[Person, ...]:
    values = project.get(key) or []
    if not isinstance(values, list):
        return ()

    people: list[Person] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        name = value.get('name') if isinstance(value.get('name'), str) else ''
        email = value.get('email') if isinstance(value.get('email'), str) else ''
        if name or email:
            people.append(Person(name=name, email=email))
    return tuple(people)


def _module_from_toml(rel_path: str, data: dict[str, Any], *, fallback_name: str, kind: str) -> ModuleInfo | None:
    tool_nexus = data.get('tool', {}).get('nexus', {})
    if not tool_nexus:
        return None
    project = data.get('project', {})
    if not isinstance(project, dict):
        project = {}
    name = project.get('name') if isinstance(project.get('name'), str) else fallback_name
    return ModuleInfo(
        name=name,
        rel_path=rel_path,
        kind=kind,
        authors=_people(project, 'authors'),
        maintainers=_people(project, 'maintainers'),
    )


def discover_modules(repo_root: Path) -> list[ModuleInfo]:
    """Discover Nexus modules without importing the Nexus package."""
    nexus_root = repo_root / 'nexus'
    modules: list[ModuleInfo] = []

    for module_root in MODULE_ROOTS:
        root_dir = nexus_root / module_root
        if not root_dir.is_dir():
            continue
        for child in sorted(root_dir.iterdir()):
            if child.name.startswith(('_', '.')):
                continue
            if child.is_dir():
                pyproject = child / 'pyproject.toml'
                if not pyproject.is_file():
                    continue
                meta = _module_from_toml(
                    child.relative_to(repo_root).as_posix(),
                    _load_toml_bytes(pyproject.read_bytes()),
                    fallback_name=child.name,
                    kind='directory',
                )
            elif child.suffix == '.py':
                data = _parse_pep723_block(child.read_bytes())
                if not data:
                    continue
                meta = _module_from_toml(
                    child.relative_to(repo_root).as_posix(),
                    data,
                    fallback_name=child.stem,
                    kind='single_file',
                )
            else:
                continue
            if meta is not None:
                modules.append(meta)
    return modules


def strip_review_irrelevant_text(path: str, content: bytes | None) -> str:
    """Return the content that counts as code for reviewer assignment."""
    if content is None:
        return ''
    if path.endswith('/pyproject.toml') or path == 'pyproject.toml':
        return ''

    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        return ''

    if path.endswith('.py'):
        text = PEP723_RE.sub('', text, count=1)
    return text


def has_reviewable_change(
    old_path: str | None,
    old_content: bytes | None,
    new_path: str | None,
    new_content: bytes | None,
) -> bool:
    """Compare files after dropping ownership/package metadata."""
    old_text = strip_review_irrelevant_text(old_path or new_path or '', old_content)
    new_text = strip_review_irrelevant_text(new_path or old_path or '', new_content)
    return old_text != new_text


def parse_name_status(output: str) -> list[DiffEntry]:
    entries: list[DiffEntry] = []
    for line in output.splitlines():
        if not line:
            continue
        parts = line.split('\t')
        status = parts[0]
        code = status[0]
        if code in {'R', 'C'} and len(parts) >= 3:
            entries.append(DiffEntry(status=status, old_path=parts[1], new_path=parts[2]))
        elif code == 'D' and len(parts) >= 2:
            entries.append(DiffEntry(status=status, old_path=parts[1], new_path=None))
        elif len(parts) >= 2:
            entries.append(DiffEntry(status=status, old_path=None, new_path=parts[1]))
    return entries


def git(repo_root: Path, *args: str, allow_failure: bool = False) -> bytes:
    result = subprocess.run(
        ['git', *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode and not allow_failure:
        stderr = result.stderr.decode('utf-8', errors='replace').strip()
        raise ReviewerAssignmentError(f'git {" ".join(args)} failed: {stderr}')
    return result.stdout


def git_text(repo_root: Path, *args: str, allow_failure: bool = False) -> str:
    return git(repo_root, *args, allow_failure=allow_failure).decode('utf-8', errors='replace')


def git_blob(repo_root: Path, revision: str, path: str | None) -> bytes | None:
    if path is None:
        return None
    result = subprocess.run(
        ['git', 'show', f'{revision}:{path}'],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode:
        return None
    return result.stdout


def changed_entries(repo_root: Path, base_ref: str, head_ref: str) -> list[DiffEntry]:
    output = git_text(repo_root, 'diff', '--name-status', '-M', base_ref, head_ref)
    return parse_name_status(output)


def matching_module(path: str, modules: list[ModuleInfo]) -> ModuleInfo | None:
    matches: list[ModuleInfo] = []
    for module in modules:
        if module.kind == 'single_file':
            if path == module.rel_path:
                matches.append(module)
        elif path == module.rel_path or path.startswith(f'{module.rel_path}/'):
            matches.append(module)
    if not matches:
        return None
    return max(matches, key=lambda item: len(item.rel_path))


def module_existed_in_base(module: ModuleInfo, old_reader) -> bool:
    if module.kind == 'single_file':
        return old_reader(module.rel_path) is not None
    return old_reader(f'{module.rel_path}/pyproject.toml') is not None


def reviewable_changes_for_entries(
    entries: list[DiffEntry],
    modules: list[ModuleInfo],
    old_reader,
    new_reader,
) -> ReviewableChanges:
    module_paths: dict[ModuleInfo, set[str]] = {}
    new_modules: set[ModuleInfo] = set()
    unmatched_paths: set[str] = set()
    for entry in entries:
        old_path_for_content = entry.old_path or entry.new_path
        old_content = old_reader(old_path_for_content)
        new_content = new_reader(entry.new_path)
        if not has_reviewable_change(old_path_for_content, old_content, entry.new_path, new_content):
            continue

        candidate_paths = [path for path in (entry.new_path, entry.old_path) if path]
        for path in candidate_paths:
            module = matching_module(path, modules)
            if module is None:
                unmatched_paths.add(path)
                continue
            module_paths.setdefault(module, set()).add(path)
            if not module_existed_in_base(module, old_reader):
                new_modules.add(module)

    changed = tuple(
        ChangedModule(module=module, paths=tuple(sorted(paths)), is_new=module in new_modules)
        for module, paths in sorted(module_paths.items(), key=lambda item: item[0].rel_path)
    )
    return ReviewableChanges(modules=changed, unmatched_paths=tuple(sorted(unmatched_paths)))


def changed_modules_for_entries(
    entries: list[DiffEntry],
    modules: list[ModuleInfo],
    old_reader,
    new_reader,
) -> list[ChangedModule]:
    return list(reviewable_changes_for_entries(entries, modules, old_reader, new_reader).modules)


def changed_modules(repo_root: Path, base_ref: str, head_ref: str, modules: list[ModuleInfo]) -> list[ChangedModule]:
    return list(reviewable_changes(repo_root, base_ref, head_ref, modules).modules)


def reviewable_changes(repo_root: Path, base_ref: str, head_ref: str, modules: list[ModuleInfo]) -> ReviewableChanges:
    entries = changed_entries(repo_root, base_ref, head_ref)
    return reviewable_changes_for_entries(
        entries,
        modules,
        lambda path: git_blob(repo_root, base_ref, path),
        lambda path: git_blob(repo_root, head_ref, path),
    )


def reviewer_candidates(
    changed: list[ChangedModule], *, unmatched_paths: tuple[str, ...] = ()
) -> list[ReviewerCandidate]:
    if len(changed) != 1 or unmatched_paths:
        return []

    owner_modules: dict[str, set[str]] = {}
    for item in changed:
        owners = item.module.maintainers if item.is_new else item.module.owners
        for owner in owners:
            username = owner.username.strip()
            if username:
                owner_modules.setdefault(username, set()).add(item.module.name)
    return [
        ReviewerCandidate(username=username, modules=tuple(sorted(modules)))
        for username, modules in sorted(owner_modules.items())
    ]


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _hook_payload() -> dict[str, Any]:
    for name in ('NEXUS_MR_HOOK_DATA', 'QCI_HOOK_DATA', 'HOOK_DATA', 'TGIT_HOOK_DATA', 'GIT_HOOK_DATA'):
        value = os.getenv(name)
        if not value:
            continue
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def detect_mr_id(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    value = _first_env((
        'NEXUS_MR_ID',
        'QCI_MR_ID',
        'MERGE_REQUEST_ID',
        'QCI_MERGE_REQUEST_ID',
        'CI_MERGE_REQUEST_ID',
        'GIT_MERGE_REQUEST_ID',
    ))
    if value:
        return value
    payload = _hook_payload()
    for candidate in (
        _nested(payload, 'object_attributes', 'id'),
        payload.get('merge_request_id'),
        payload.get('mergeRequestId'),
        payload.get('id'),
    ):
        if candidate:
            return str(candidate)
    return None


def detect_project(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    payload = _hook_payload()
    for candidate in (
        _first_env(('NEXUS_GONGFENG_PROJECT', 'GONGFENG_PROJECT', 'CI_PROJECT_PATH')),
        _nested(payload, 'project', 'path_with_namespace'),
        _nested(payload, 'repository', 'full_name'),
    ):
        if candidate:
            return str(candidate)
    return DEFAULT_PROJECT


def detect_base_ref(repo_root: Path, head_ref: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit

    sha = _first_env(('NEXUS_REVIEW_BASE_SHA', 'CI_MERGE_REQUEST_TARGET_SHA', 'MERGE_REQUEST_TARGET_SHA'))
    if sha:
        return sha

    target_branch = _first_env((
        'NEXUS_MR_TARGET_BRANCH',
        'CI_MERGE_REQUEST_TARGET_BRANCH_NAME',
        'QCI_MR_TARGET_BRANCH',
        'MERGE_REQUEST_TARGET_BRANCH',
        'TARGET_BRANCH',
    ))
    payload = _hook_payload()
    target_branch = target_branch or _nested(payload, 'object_attributes', 'target_branch')

    candidates: list[str] = []
    if target_branch:
        git(repo_root, 'fetch', 'origin', str(target_branch), '--depth=200', allow_failure=True)
        candidates.extend([f'origin/{target_branch}', str(target_branch)])
    candidates.extend(['origin/main', 'origin/master', 'main', 'master'])

    for candidate in candidates:
        merge_base = git_text(repo_root, 'merge-base', candidate, head_ref, allow_failure=True).strip()
        if merge_base:
            return merge_base
    raise ReviewerAssignmentError('Cannot determine base ref; pass --base-ref or set target branch env')


class GongfengClient:
    def __init__(self, *, api_url: str, access_token: str | None, oauth_token: str | None) -> None:
        self.api_url = api_url.rstrip('/')
        self.access_token = access_token
        self.oauth_token = oauth_token

    @classmethod
    def from_env(cls) -> GongfengClient:
        return cls(
            api_url=os.getenv('GONGFENG_API_URL', DEFAULT_API_URL),
            access_token=_first_env(('GONGFENG_ACCESS_TOKEN', 'TGIT_ACCESS_TOKEN', 'TGIT_TOKEN')),
            oauth_token=_first_env(('GONGFENG_OAUTH_TOKEN', 'TGIT_OAUTH_TOKEN')),
        )

    @property
    def has_auth(self) -> bool:
        return bool(self.access_token or self.oauth_token)

    def request_json(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        if not self.has_auth:
            raise ReviewerAssignmentError('Missing Gongfeng token; set GONGFENG_ACCESS_TOKEN or GONGFENG_OAUTH_TOKEN')

        data = json.dumps(body).encode('utf-8') if body is not None else None
        headers = {'Content-Type': 'application/json', 'User-Agent': 'nexus-module-reviewer'}
        if self.access_token:
            headers['PRIVATE-TOKEN'] = self.access_token
        elif self.oauth_token:
            headers['OAUTH-TOKEN'] = self.oauth_token

        request = urllib.request.Request(f'{self.api_url}{path}', data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')
            raise ReviewerAssignmentError(f'Gongfeng API {method} {path} failed: {exc.code} {detail}') from exc
        return json.loads(raw.decode('utf-8')) if raw else None

    def get_user(self, username: str) -> dict[str, Any] | None:
        path = f'/users/{urllib.parse.quote(username, safe="")}'
        try:
            user = self.request_json('GET', path)
        except ReviewerAssignmentError as exc:
            if ' 404 ' in str(exc):
                return None
            raise
        if not isinstance(user, dict):
            return None
        if user.get('state') not in (None, '', 'active'):
            return None
        return user

    def get_merge_request(self, project: str, mr_id: str) -> dict[str, Any]:
        project_part = urllib.parse.quote(project, safe='')
        mr_part = urllib.parse.quote(str(mr_id), safe='')
        data = self.request_json('GET', f'/projects/{project_part}/merge_request/{mr_part}')
        return data if isinstance(data, dict) else {}

    def add_necessary_reviewers(self, project: str, mr_id: str, reviewer_ids: list[int]) -> dict[str, Any]:
        if not reviewer_ids:
            return {}
        project_part = urllib.parse.quote(project, safe='')
        mr_part = urllib.parse.quote(str(mr_id), safe='')
        data = self.request_json(
            'POST',
            f'/projects/{project_part}/merge_request/{mr_part}/review/batch_invite',
            {'necessary_reviewer_ids': reviewer_ids},
        )
        return data if isinstance(data, dict) else {}

    def dismiss_reviewer(self, project: str, mr_id: str, reviewer_id: int) -> dict[str, Any]:
        project_part = urllib.parse.quote(project, safe='')
        mr_part = urllib.parse.quote(str(mr_id), safe='')
        query = urllib.parse.urlencode({'reviewer_id': reviewer_id})
        data = self.request_json(
            'DELETE',
            f'/projects/{project_part}/merge_request/{mr_part}/review/dismissals?{query}',
        )
        return data if isinstance(data, dict) else {}

    def list_merge_request_notes(self, project: str, mr_id: str) -> list[dict[str, Any]]:
        project_part = urllib.parse.quote(project, safe='')
        mr_part = urllib.parse.quote(str(mr_id), safe='')
        query = urllib.parse.urlencode({'system': 'false', 'sort': 'created_desc', 'page': 1, 'per_page': 100})
        data = self.request_json('GET', f'/projects/{project_part}/merge_requests/{mr_part}/notes?{query}')
        return data if isinstance(data, list) else []

    def create_merge_request_note(self, project: str, mr_id: str, body: str) -> dict[str, Any]:
        project_part = urllib.parse.quote(project, safe='')
        mr_part = urllib.parse.quote(str(mr_id), safe='')
        data = self.request_json(
            'POST',
            f'/projects/{project_part}/merge_requests/{mr_part}/notes',
            {'body': body, 'notify_enabled': False},
        )
        return data if isinstance(data, dict) else {}

    def update_merge_request_note(self, project: str, mr_id: str, note_id: int, body: str) -> dict[str, Any]:
        project_part = urllib.parse.quote(project, safe='')
        mr_part = urllib.parse.quote(str(mr_id), safe='')
        note_part = urllib.parse.quote(str(note_id), safe='')
        data = self.request_json(
            'PUT',
            f'/projects/{project_part}/merge_requests/{mr_part}/notes/{note_part}',
            {'body': body, 'notify_enabled': False},
        )
        return data if isinstance(data, dict) else {}


def _reviewer_id(reviewer: Any) -> int | None:
    if not isinstance(reviewer, dict):
        return None
    if isinstance(reviewer.get('userId'), int):
        return reviewer['userId']
    user = reviewer.get('user')
    if isinstance(user, dict) and isinstance(user.get('id'), int):
        return user['id']
    if isinstance(reviewer.get('id'), int):
        return reviewer['id']
    return None


def _reviewer_id_list(reviewers: Any) -> list[int]:
    ids: list[int] = []
    if not isinstance(reviewers, list):
        return ids
    for reviewer in reviewers:
        user_id = _reviewer_id(reviewer)
        if user_id is not None and user_id not in ids:
            ids.append(user_id)
    return ids


def _reviewer_ids(reviewers: Any) -> set[int]:
    return set(_reviewer_id_list(reviewers))


def _unique_ids(reviewer_ids: list[int] | tuple[int, ...]) -> list[int]:
    ids: list[int] = []
    for reviewer_id in reviewer_ids:
        if reviewer_id not in ids:
            ids.append(reviewer_id)
    return ids


def _managed_reviewer_note_body(reviewer_ids: list[int] | tuple[int, ...]) -> str:
    payload = {
        'necessary_reviewer_ids': _unique_ids(list(reviewer_ids)),
        'version': 1,
    }
    return f'<!-- {MANAGED_REVIEWERS_MARKER} {json.dumps(payload, separators=(",", ":"), sort_keys=True)} -->'


def _parse_managed_reviewer_note(note: dict[str, Any]) -> ManagedReviewerState | None:
    body = note.get('body')
    if not isinstance(body, str):
        body = note.get('note')
    if not isinstance(body, str):
        return None
    match = MANAGED_REVIEWERS_RE.search(body)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    reviewer_ids = payload.get('necessary_reviewer_ids')
    if not isinstance(reviewer_ids, list):
        return None
    ids = _unique_ids([item for item in reviewer_ids if isinstance(item, int)])
    note_id = note.get('id')
    return ManagedReviewerState(
        note_id=note_id if isinstance(note_id, int) else None,
        reviewer_ids=tuple(ids),
    )


def get_managed_reviewer_state(client: GongfengClient, project: str, mr_id: str) -> ManagedReviewerState:
    for note in client.list_merge_request_notes(project, mr_id):
        state = _parse_managed_reviewer_note(note)
        if state is not None:
            return state
    return ManagedReviewerState()


def save_managed_reviewer_state(
    client: GongfengClient,
    project: str,
    mr_id: str,
    state: ManagedReviewerState,
    reviewer_ids: list[int],
) -> None:
    body = _managed_reviewer_note_body(reviewer_ids)
    if state.note_id is not None:
        client.update_merge_request_note(project, mr_id, state.note_id, body)
        return
    if reviewer_ids:
        client.create_merge_request_note(project, mr_id, body)


def resolve_necessary_reviewer_ids(
    client: GongfengClient,
    project: str,
    mr_id: str,
    candidates: list[ReviewerCandidate],
    *,
    mr: dict[str, Any] | None = None,
) -> tuple[list[int], list[int], list[str]]:
    mr = mr or client.get_merge_request(project, mr_id)
    current_reviewer_ids = _reviewer_id_list(mr.get('necessary_reviewers'))

    desired_ids: list[int] = []
    added_ids: list[int] = []
    skipped: list[str] = []
    for candidate in candidates:
        user = client.get_user(candidate.username)
        if not user:
            skipped.append(f'{candidate.username}: user not found or inactive')
            continue
        user_id = user.get('id')
        if not isinstance(user_id, int):
            skipped.append(f'{candidate.username}: missing user id')
            continue
        if user_id not in desired_ids:
            desired_ids.append(user_id)
        if user_id in current_reviewer_ids:
            skipped.append(f'{candidate.username}: already necessary reviewer')
            continue
        if user_id not in added_ids:
            added_ids.append(user_id)
    return desired_ids, added_ids, skipped


def verify_necessary_reviewer_ids(client: GongfengClient, project: str, mr_id: str, reviewer_ids: list[int]) -> None:
    mr = client.get_merge_request(project, mr_id)
    existing_ids = _reviewer_ids(mr.get('necessary_reviewers'))
    missing_ids = [user_id for user_id in reviewer_ids if user_id not in existing_ids]
    if missing_ids:
        raise ReviewerAssignmentError(f'Gongfeng did not add necessary reviewer ids: {missing_ids}')


def verify_removed_necessary_reviewer_ids(
    client: GongfengClient, project: str, mr_id: str, reviewer_ids: list[int]
) -> None:
    mr = client.get_merge_request(project, mr_id)
    existing_ids = _reviewer_ids(mr.get('necessary_reviewers'))
    remaining_ids = [user_id for user_id in reviewer_ids if user_id in existing_ids]
    if remaining_ids:
        raise ReviewerAssignmentError(f'Gongfeng did not remove necessary reviewer ids: {remaining_ids}')


def reconcile_necessary_reviewers(
    client: GongfengClient,
    project: str,
    mr_id: str,
    candidates: list[ReviewerCandidate],
    *,
    mr: dict[str, Any] | None = None,
) -> ReviewerReconcileResult:
    mr = mr or client.get_merge_request(project, mr_id)
    state = get_managed_reviewer_state(client, project, mr_id)
    current_necessary_ids = _reviewer_ids(mr.get('necessary_reviewers'))
    desired_ids, added_ids, skipped = resolve_necessary_reviewer_ids(
        client,
        project,
        mr_id,
        candidates,
        mr=mr,
    )
    managed_ids = list(state.reviewer_ids)
    removed_ids = [
        reviewer_id
        for reviewer_id in managed_ids
        if reviewer_id not in desired_ids and reviewer_id in current_necessary_ids
    ]
    next_managed_ids = [
        reviewer_id
        for reviewer_id in managed_ids
        if reviewer_id in desired_ids and reviewer_id in current_necessary_ids
    ]

    for reviewer_id in removed_ids:
        client.dismiss_reviewer(project, mr_id, reviewer_id)
    if removed_ids:
        verify_removed_necessary_reviewer_ids(client, project, mr_id, removed_ids)

    if added_ids:
        client.add_necessary_reviewers(project, mr_id, added_ids)
        verify_necessary_reviewer_ids(client, project, mr_id, added_ids)
        next_managed_ids.extend(added_ids)

    next_managed_ids = _unique_ids(next_managed_ids)
    if state.note_id is not None or next_managed_ids:
        save_managed_reviewer_state(client, project, mr_id, state, next_managed_ids)

    return ReviewerReconcileResult(
        desired_ids=tuple(desired_ids),
        added_ids=tuple(added_ids),
        removed_ids=tuple(removed_ids),
        managed_ids=tuple(next_managed_ids),
        skipped=tuple(skipped),
    )


def print_plan(
    changed: list[ChangedModule], candidates: list[ReviewerCandidate], unmatched_paths: tuple[str, ...]
) -> None:
    if not changed and not unmatched_paths:
        print('No reviewable module code changes found.')
        return
    if changed:
        print('Reviewable changed modules:')
        for item in changed:
            owner_names = ', '.join(owner.username for owner in item.module.owners) or '<none>'
            print(f'  - {item.module.name} ({item.module.rel_path}) owners=[{owner_names}]')
            for path in item.paths:
                print(f'    {path}')
    if unmatched_paths:
        print('Reviewable changes outside known modules:')
        for path in unmatched_paths:
            print(f'  - {path}')
    if len(changed) != 1 or unmatched_paths:
        print('Reviewable changes are not limited to one module; no module owner necessary reviewers to add.')
    print('Necessary reviewer candidates:')
    for candidate in candidates:
        print(f'  - {candidate.username}: {", ".join(candidate.modules)}')


def run(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    project = detect_project(args.project)
    mr_id = detect_mr_id(args.mr_id)
    client: GongfengClient | None = None
    mr: dict[str, Any] | None = None
    if args.apply:
        if not mr_id:
            raise ReviewerAssignmentError('Cannot determine MR id; pass --mr-id or set QCI hook env')
        client = GongfengClient.from_env()
        mr = client.get_merge_request(project, mr_id)

    modules = discover_modules(repo_root)
    base_ref = detect_base_ref(repo_root, args.head_ref, args.base_ref)
    reviewable = reviewable_changes(repo_root, base_ref, args.head_ref, modules)
    changed = list(reviewable.modules)
    candidates = reviewer_candidates(changed, unmatched_paths=reviewable.unmatched_paths)
    print_plan(changed, candidates, reviewable.unmatched_paths)

    if args.dry_run:
        return 0

    if client is None:
        client = GongfengClient.from_env()
    result = reconcile_necessary_reviewers(
        client,
        project,
        mr_id,
        candidates,
        mr=mr,
    )
    for item in result.skipped:
        print(f'Skipped reviewer: {item}')
    if result.removed_ids:
        print(f'Removed managed necessary reviewer ids: {list(result.removed_ids)}')
    if result.added_ids:
        print(f'Added necessary reviewer ids: {list(result.added_ids)}')
    else:
        print('No necessary reviewer ids to add after filtering.')
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    parser.add_argument('--base-ref', help='Base commit/ref. Defaults to merge-base with target branch env.')
    parser.add_argument('--head-ref', default='HEAD')
    parser.add_argument('--project', help=f'Gongfeng project id/path. Defaults to {DEFAULT_PROJECT}.')
    parser.add_argument('--mr-id', help='Gongfeng global merge request id.')
    parser.add_argument('--apply', action='store_true', help='Call Gongfeng API and add necessary reviewers.')
    parser.add_argument('--fail-open', action='store_true', help='Log errors and exit 0 for CI friendliness.')
    parser.set_defaults(dry_run=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.dry_run = not args.apply
    try:
        return run(args)
    except Exception as exc:
        if args.fail_open:
            print(f'assign_module_reviewers failed open: {exc}', file=sys.stderr)
            return 0
        raise


if __name__ == '__main__':
    raise SystemExit(main())
