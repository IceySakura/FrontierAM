"""Default compatibility and real repository boundaries for SWE task preparation."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from pathlib import Path

import pytest

from nexus.extensions.tasks.swe.constants import GIT_TRUNCATE_SCRIPT, anti_hack_prompt_for_backend
from nexus.extensions.tasks.swe.infer import get_git_patch


def test_unconfigured_process_uses_original_public_defaults():
    program = """
import json
from nexus.extensions.agents.swe_agent.agent import SWEAgentInput
from nexus.extensions.agents.swe_agent.tools import resolve_bash_backend
from nexus.extensions.tasks.swe.task import SwebenchTaskInput
from nexus.runner.batch.log_capture import _resolve_exempt_size
from nexus.utils.common import _resolve_call_deadline, _resolve_retry_total_timeout
print(json.dumps({
    'backend': resolve_bash_backend(None),
    'call': _resolve_call_deadline(),
    'retry': _resolve_retry_total_timeout(),
    'log': _resolve_exempt_size(10000),
    'anti_hack': SwebenchTaskInput.model_fields['enable_anti_hack_prompt'].default,
    'continue_tool': SWEAgentInput.model_fields['continue_on_tool_timeout'].default,
}))
"""
    keys = {'SWE_AGENT_BASH_BACKEND', 'NEXUS_CALL_DEADLINE', 'NEXUS_RETRY_TOTAL_TIMEOUT', 'NEXUS_LOG_EXEMPT_SIZE'}
    environment = {key: value for key, value in os.environ.items() if key not in keys}
    result = subprocess.run(
        [sys.executable, '-B', '-c', program],
        cwd=Path(__file__).resolve().parents[4],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert json.loads(result.stdout) == {
        'backend': 'terminal',
        'call': 0,
        'retry': 0,
        'log': 10000,
        'anti_hack': False,
        'continue_tool': False,
    }


@pytest.mark.parametrize('backend', ['terminal', 'tmux', 'stateless'])
def test_notice_matches_shell_backend(backend):
    notice = anti_hack_prompt_for_backend(backend)
    assert notice.count('<IMPORTANT>') == notice.count('</IMPORTANT>') == 1
    assert notice.count('Mining the harness') == int(backend != 'stateless')
    assert notice.endswith('Submissions that rely on any of the above are invalid and are discarded.\n</IMPORTANT>')


@pytest.fixture
def repository(tmp_path, monkeypatch):
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', '/dev/null')
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')

    def git(*args, check=True):
        return subprocess.run(
            ['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=check,
            timeout=20,
        )

    git('init', '-q')
    (tmp_path / 'code.py').write_text('original\n')
    git('add', '.')
    git('commit', '-qm', 'baseline')
    return tmp_path, git, git('rev-parse', 'HEAD').stdout.strip()


def test_history_truncation_keeps_ancestor_refs_and_removes_nested_future_tags(repository):
    root, git, base = repository
    git('tag', 'base-lightweight', base)
    git('tag', '-a', 'base-annotated', base, '-m', 'baseline tag')
    git('tag', '-a', 'base-nested', 'base-annotated', '-m', 'nested baseline tag')
    git('update-ref', 'refs/remotes/origin/base', base)
    (root / 'code.py').write_text('future answer\n')
    git('add', '.')
    git('commit', '-qm', 'future')
    future = git('rev-parse', 'HEAD').stdout.strip()
    git('tag', '-a', 'future-annotated', future, '-m', 'future tag')
    git('tag', '-a', 'future-nested', 'future-annotated', '-m', 'nested future tag')
    git('update-ref', 'refs/remotes/origin/future', future)
    git('symbolic-ref', 'refs/remotes/origin/HEAD', 'refs/remotes/origin/future')
    git('checkout', '-q', '--detach', base)
    subprocess.run(
        ['bash', '-c', GIT_TRUNCATE_SCRIPT],
        cwd=root,
        env={**os.environ, 'BASE_SHA': base},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    for ref in ('base-lightweight', 'base-annotated', 'base-nested', 'refs/remotes/origin/base'):
        assert git('rev-parse', ref + '^{commit}').stdout.strip() == base
    for ref in ('refs/tags/future-annotated', 'refs/tags/future-nested', 'refs/remotes/origin/future'):
        assert git('show-ref', '--verify', ref, check=False).returncode != 0
    assert git('rev-list', '--all', '--not', base).stdout.strip() == ''
    assert git('cat-file', '-e', future, check=False).returncode != 0


def test_unreachable_answer_is_pruned_when_no_future_ref_remains(repository):
    root, git, base = repository
    (root / 'code.py').write_text('unreachable answer\n')
    git('add', '.')
    git('commit', '-qm', 'unreachable future')
    future = git('rev-parse', 'HEAD').stdout.strip()
    git('reset', '--hard', base)
    subprocess.run(
        ['bash', '-c', GIT_TRUNCATE_SCRIPT],
        cwd=root,
        env={**os.environ, 'BASE_SHA': base},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert git('cat-file', '-e', future, check=False).returncode != 0
    assert git('rev-parse', 'HEAD').stdout.strip() == base


@pytest.mark.parametrize(
    ('language', 'paths', 'artifact'),
    [
        ('rust', ['Cargo.lock'], 'target/generated'),
        ('php', ['composer.lock', 'vendor/library.php'], '.phpunit.result.cache'),
        ('go', ['vendor/library.go'], 'generated.test'),
    ],
)
@pytest.mark.parametrize('explicit_exclusion', [False, True])
def test_required_new_files_are_submitted_unless_explicitly_excluded(
    repository,
    language,
    paths,
    artifact,
    explicit_exclusion,
):
    root, _, base = repository
    for name in [*paths, artifact]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture content\n')
    patch = get_git_patch(str(root), language, base, paths if explicit_exclusion else None)
    for name in paths:
        assert (f'diff --git a/{name} b/{name}' in patch) is not explicit_exclusion
    assert f'diff --git a/{artifact} b/{artifact}' not in patch
