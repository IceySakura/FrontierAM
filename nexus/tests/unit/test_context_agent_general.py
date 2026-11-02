"""Unit coverage for the path-native ``context_agent_general`` chain.

The tools are exercised for real (temp files, the subprocess REPL, actual BM25
retrieval) rather than mocked, because the whole premise of this chain is that
the agent works against files on disk.
"""

import importlib
import json

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nexus.core import ExecutionContext, agent_registry, task_registry
from nexus.core.module_meta import parse_directory_module
from nexus.core.openai_client import OpenAICompatibleClient, OpenAIConfig
from nexus.dataset.task_files import RemoteZipTaskFiles, task_files_from_spec
from nexus.extensions.agents.base.schema import TerminationReason
from nexus.extensions.agents.context_agent import general_path_privacy as path_privacy
from nexus.extensions.agents.context_agent.general_agent import (
    ContextAgentGeneralInput,
    _resolve_enabled_tools,
    _task_system_text,
    run_context_agent_general,
)
from nexus.extensions.agents.context_agent.general_lang import resolve_lang
from nexus.extensions.agents.context_agent.general_prompts import (
    build_file_manifest,
    build_general_system_prompt,
)
from nexus.extensions.agents.context_agent.tools.general import (
    GENERAL_TOOL_ORDER,
    FileReadError,
    GeneralState,
    Knobs,
    build_tool_schemas,
    dispatch_tool,
    parse_declared_files,
    prepare_workspace,
    stage_declared_files,
    subprocess_env,
)
from nexus.extensions.agents.context_agent.tools.general.runtime_tools import ReadLinesInput, read_lines_tool
from nexus.extensions.agents.context_agent.tools.general.tools import curl as curl_tool
from nexus.extensions.agents.context_agent.tools.general.tools import python_repl, shell
from nexus.extensions.tasks.context_agent import general_task as task_module
from nexus.extensions.tasks.context_agent.eval import EVALUATORS, RULE_BASED, STYLES, complexlong
from nexus.extensions.tasks.context_agent.eval._judge_llm import resolve_judge_llm
from nexus.extensions.tasks.context_agent.general_task import (
    ContextAgentGeneralTaskInput,
    run_context_agent_general_task,
)
from nexus.extensions.tasks.context_agent.scoring import rm_scorer
from nexus.extensions.tasks.context_agent.task_files_auth import bind_task_wedata_key
from nexus.extensions.tasks.context_agent.workspace_artifacts import persist_runtime_workspace
from nexus.runtime.models import coerce_args
from nexus.runtime.providers.local import LocalRuntimeProvider


DOCUMENT = '\n'.join(
    [
        '# Project Apollo — Engineering Handbook',
        '',
        '## 3. Networking and Configuration',
        'Unless overridden in apollo.toml, the default client listening port is 7421.',
        'The admin API listens on 7431.',
        '',
        '## 4. Memory and Eviction',
        'A shard evicts entries with an LRU policy once it reaches its memory limit.',
        'The default per-shard memory limit is 256 MB.',
    ]
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


class _Message:
    """An assistant message whose ``model_dump`` is what the agent must append."""

    def __init__(self, content: str | None = None, tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        dumped: dict[str, Any] = {'role': 'assistant', 'content': self.content}
        if self.tool_calls is not None:
            dumped['tool_calls'] = [
                {
                    'id': tc.id,
                    'type': 'function',
                    'function': {'name': tc.function.name, 'arguments': tc.function.arguments},
                }
                for tc in self.tool_calls
            ]
        if exclude_none:
            dumped = {k: v for k, v in dumped.items() if v is not None}
        return dumped


class _ScriptedClient(OpenAICompatibleClient):
    """Replays scripted messages and records what it was asked."""

    def __init__(self, *messages: _Message, finish_reasons: list[str | None] | None = None):
        super().__init__(OpenAIConfig(client_args={}, request_args={}))
        self._queue = list(messages)
        self._finish_reasons = list(finish_reasons or [])
        self.calls: list[dict[str, Any]] = []

    async def _create_completion(self, messages, model=None, **kwargs):
        raise NotImplementedError

    async def create_completion(self, messages, **kwargs):
        self.calls.append({'messages': [dict(m) for m in messages], 'tools': kwargs.get('tools')})
        message = self._queue.pop(0) if self._queue else _Message(content='fallback answer')
        finish_reason = self._finish_reasons.pop(0) if self._finish_reasons else None
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


def _state(tmp_path, files) -> GeneralState:
    workspace = prepare_workspace(tmp_path / 'rollout')
    return GeneralState(
        workspace=workspace,
        knobs=Knobs(file_search_top_k=5, file_search_chunk_lines=4, python_timeout=60),
        declared_files=files,
    )


def test_rm_concurrency_is_disabled_when_unset(monkeypatch) -> None:
    monkeypatch.delenv('CTXAGENT_RM_MAX_CONCURRENT', raising=False)

    assert rm_scorer._rm_max_concurrent() == 0


def test_rm_concurrency_rejects_invalid_values(monkeypatch) -> None:
    monkeypatch.setenv('CTXAGENT_RM_MAX_CONCURRENT', 'many')

    with pytest.raises(ValueError, match='non-negative integer'):
        rm_scorer._rm_max_concurrent()


def test_registered_runtime_tool_coerces_input_model() -> None:
    args, kwargs = coerce_args(
        read_lines_tool,
        ({'path': 'document.md', 'start': 2, 'end': 5}, object()),
        {},
    )

    assert isinstance(args[0], ReadLinesInput)
    assert args[0].start == 2
    assert args[0].end == 5
    assert kwargs == {}


def test_split_module_metadata_and_registration_names() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    expected_modules = {
        'nexus/extensions/agents/context_agent': (
            'context_agent',
            {
                'agents': {
                    'context_agent_general': 'nexus.extensions.agents.context_agent.general_agent',
                },
                'tools': {
                    'context_agent_general/curl': 'nexus.extensions.agents.context_agent.tools.general.runtime_tools',
                    'context_agent_general/bash': 'nexus.extensions.agents.context_agent.tools.general.runtime_tools',
                    'context_agent_general/python': 'nexus.extensions.agents.context_agent.tools.general.runtime_tools',
                    'context_agent_general/file_search_msearch': (
                        'nexus.extensions.agents.context_agent.tools.general.runtime_tools'
                    ),
                    'context_agent_general/file_search_mclick': (
                        'nexus.extensions.agents.context_agent.tools.general.runtime_tools'
                    ),
                    'context_agent_general/grep': 'nexus.extensions.agents.context_agent.tools.general.runtime_tools',
                    'context_agent_general/read_lines': 'nexus.extensions.agents.context_agent.tools.general.runtime_tools',
                    'context_agent_general/_cleanup': 'nexus.extensions.agents.context_agent.tools.general.runtime_tools',
                },
            },
        ),
        'nexus/extensions/tasks/context_agent': (
            'context_agent',
            {
                'tasks': {
                    'context_agent_general_task': 'nexus.extensions.tasks.context_agent.general_task',
                    'context_agent_enterprisebench_native_task': (
                        'nexus.extensions.tasks.context_agent.enterprisebench_native.task'
                    ),
                }
            },
        ),
    }

    for relative_path, (name, objects) in expected_modules.items():
        meta = parse_directory_module(repo_root / relative_path, repo_root)
        assert meta is not None
        assert meta.name == name
        assert meta.objects == objects

    canonical_agent = importlib.import_module('nexus.extensions.agents.context_agent.general_agent')
    canonical_task = importlib.import_module('nexus.extensions.tasks.context_agent.general_task')
    registered_agent = agent_registry.get('context_agent_general')
    registered_task = task_registry.get('context_agent_general_task')

    assert registered_agent is canonical_agent.run_context_agent_general
    assert registered_task is canonical_task.run_context_agent_general_task
    assert registered_agent.nexus_meta['name'] == 'context_agent_general'
    assert registered_task.nexus_meta['name'] == 'context_agent_general_task'


def test_context_agent_model_dependencies_do_not_leak_into_nexus_packaging() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    module_root = repo_root / 'nexus/extensions/agents/context_agent'
    agent_meta = parse_directory_module(module_root, repo_root)

    assert agent_meta is not None
    assert agent_meta.all_runtime_deps == []
    assert 'context-agent-general' not in (repo_root / 'pyproject.toml').read_text(encoding='utf-8')
    requirements = [
        line
        for line in (module_root / 'requirements-main-image.txt').read_text(encoding='utf-8').splitlines()
        if line and not line.startswith('#')
    ]
    assert requirements == [
        'numpy==1.26.4',
        'pandas==2.3.3',
        'openpyxl==3.1.5',
        'pdfminer.six==20260107',
        'pdfplumber==0.11.10',
        'pypdf==6.17.0',
        'pypdfium2==5.13.0',
        'PyMuPDF==1.28.2',
        'Pillow==12.3.0',
        'python-docx==1.2.0',
        'python-pptx==1.0.2',
        'scipy==1.17.1',
        'pytesseract==0.3.13',
        'pdf2image==1.17.0',
        'fonttools==4.65.0',
        'opencv-python-headless==4.11.0.86',
    ]


def test_main_image_env_combines_runtime_path_cleanup_with_secret_scrubbing(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess_env,
        'get_clean_sandbox_env',
        lambda: {
            'PATH': '/main/bin:/usr/bin',
            'OPENAI_API_KEY': 'secret',
            'CTXAGENT_VISIBLE': 'yes',
        },
    )

    assert subprocess_env.main_image_env() == {
        'PATH': '/main/bin:/usr/bin',
        'CTXAGENT_VISIBLE': 'yes',
    }


def test_main_image_python_prefers_python_and_falls_back_to_python3(tmp_path, monkeypatch) -> None:
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    python3 = bin_dir / 'python3'
    python3.write_text('#!/bin/sh\n', encoding='utf-8')
    python3.chmod(0o755)

    assert subprocess_env.resolve_main_image_python({'PATH': str(bin_dir)}) == str(python3)

    python = bin_dir / 'python'
    python.write_text('#!/bin/sh\n', encoding='utf-8')
    python.chmod(0o755)
    assert subprocess_env.resolve_main_image_python({'PATH': str(bin_dir)}) == str(python)

    monkeypatch.setattr(subprocess_env.shutil, 'which', lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match='(?i)main sandbox image.*python'):
        subprocess_env.resolve_main_image_python({'PATH': str(bin_dir)})


def test_python_repl_spawns_the_main_image_python(tmp_path, monkeypatch) -> None:
    command_env = {'PATH': '/main/bin:/usr/bin', 'CTXAGENT_VISIBLE': 'yes'}
    captured: dict[str, Any] = {}
    sentinel = object()

    monkeypatch.setattr(python_repl, 'main_image_env', lambda: command_env)
    monkeypatch.setattr(python_repl, 'resolve_main_image_python', lambda env: '/main/bin/python')

    def fake_popen(args, **kwargs):
        captured['args'] = args
        captured['kwargs'] = kwargs
        return sentinel

    monkeypatch.setattr(python_repl.subprocess, 'Popen', fake_popen)

    repl = python_repl.PythonREPL(cwd=str(tmp_path))
    assert repl._spawn() is sentinel
    assert captured['args'] == ['/main/bin/python', '-u', str(python_repl._SERVER)]
    assert captured['kwargs']['cwd'] == str(tmp_path)
    assert captured['kwargs']['env'] is command_env


def test_bash_uses_the_main_image_environment(tmp_path, monkeypatch) -> None:
    command_env = {'PATH': '/main/bin:/usr/bin', 'CTXAGENT_VISIBLE': 'yes'}
    captured: dict[str, Any] = {}

    monkeypatch.setattr(shell, 'main_image_env', lambda: command_env)

    def fake_run(command, **kwargs):
        captured['command'] = command
        captured['kwargs'] = kwargs
        return SimpleNamespace(stdout='ok\n', stderr='', returncode=0)

    monkeypatch.setattr(shell, 'run_command_group', fake_run)

    assert shell._run_command_blocking('python -m pip list', 10, 1000, str(tmp_path)) == 'ok'
    assert captured['kwargs']['env'] is command_env


def test_curl_uses_main_image_environment_and_retries_an_unresolvable_proxy(monkeypatch) -> None:
    command_env = {'PATH': '/main/bin:/usr/bin', 'CTXAGENT_VISIBLE': 'yes'}
    calls: list[tuple[list[str], dict[str, Any]]] = []

    monkeypatch.setattr(curl_tool, 'main_image_env', lambda: command_env)

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) == 1:
            return SimpleNamespace(stdout='', stderr='curl: (5) Could not resolve proxy', returncode=5)
        return SimpleNamespace(stdout='200\tapplication/pdf\t123\thttps://example.com/a.pdf', stderr='', returncode=0)

    monkeypatch.setattr(curl_tool, 'run_command_group', fake_run)

    command = ['curl', '-sS', '-L', 'https://example.com/a.pdf']
    assert curl_tool._run_curl(command, 10) == (
        0,
        '200\tapplication/pdf\t123\thttps://example.com/a.pdf',
        '',
    )
    assert calls[0][0] == command
    assert calls[1][0] == ['curl', '--disable', '--proxy', '', '-sS', '-L', 'https://example.com/a.pdf']
    assert all(kwargs['env'] is command_env for _, kwargs in calls)


def test_general_tool_repo_root_survives_agent_colocation() -> None:
    state_module = importlib.import_module('nexus.extensions.agents.context_agent.tools.general.state')
    assert Path(__file__).resolve().parents[2] == state_module.REPO_ROOT


_REQUIRED = object()


def _model_field_defaults(model) -> dict[str, Any]:
    return {
        name: _REQUIRED if field.is_required() else field.get_default(call_default_factory=True)
        for name, field in model.model_fields.items()
    }


def test_agent_input_output_fields_and_defaults_are_stable() -> None:
    assert _model_field_defaults(ContextAgentGeneralInput) == {
        'working_dir': _REQUIRED,
        'user_prompt': _REQUIRED,
        'context_files': [],
        'context_dialogue': None,
        'system_messages': None,
        'language': None,
        'files_base': None,
        'max_iterations': 20,
        'max_tool_rounds': None,
        'max_parallel_calls': 0,
        'file_search_top_k': 5,
        'file_search_chunk_lines': 50,
        'python_timeout': 120,
        'curl_timeout': 120,
        'curl_max_bytes': 200 * 1024 * 1024,
        'system_prompt_lang': None,
        'system_prompt_path': None,
        'enabled_tools': None,
        'mcp_servers': None,
        'tool_search': None,
    }
    assert _model_field_defaults(
        importlib.import_module('nexus.extensions.agents.context_agent.general_agent').ContextAgentGeneralOutput
    ) == {
        'main_trajectory': _REQUIRED,
        'final_answer': None,
        'exit_status': 'SUCCESS',
        'tool_rounds_used': 0,
        'tool_calls_requested': 0,
        'tool_calls_executed': 0,
        'tool_calls_skipped': 0,
        'hit_max_turn_limit': False,
        'forced_finalization': False,
        'finalization_failed': False,
        'response_budget_exhausted': False,
        'tool_name_resolution_error_count': 0,
        'language': 'en',
        'num_files': 0,
        'staged_files': [],
        'tool_schemas': [],
    }


def test_task_input_output_fields_and_defaults_are_stable() -> None:
    assert _model_field_defaults(ContextAgentGeneralTaskInput) == {
        'run_agent': _REQUIRED,
        'context_question': _REQUIRED,
        'context_files': [],
        'context_dialogue': None,
        'dataset_name': None,
        'metadata': {},
        'answer': None,
        'rubrics': None,
        'checklist': None,
        'judge_template': None,
        'evaluate': True,
        'tool_call_bad_pattern_policy': None,
        'zero_score_on_tool_name_resolution_error': False,
        'constraint_rubrics': [],
        'constraint_fail_reward': 0.5,
        'judge_backend': 'auto',
        'reward_system_router_id': None,
        'reward_prompt': None,
        'ref_answer': None,
        'rm_extra': None,
        'WEDATA_KEY': None,
        'artifact_root': None,
        'artifact_max_files': 10000,
        'artifact_max_bytes': 1024 * 1024 * 1024,
        'system_messages': None,
        'language': None,
        'files_base': None,
        'working_dir': None,
        'max_iterations': 20,
        'max_tool_rounds': None,
        'max_parallel_calls': 0,
        'file_search_top_k': 5,
        'file_search_chunk_lines': 50,
        'python_timeout': 120,
        'curl_timeout': 120,
        'curl_max_bytes': 200 * 1024 * 1024,
        'system_prompt_lang': None,
        'system_prompt_path': None,
        'enabled_tools': None,
    }
    assert _model_field_defaults(
        importlib.import_module('nexus.extensions.tasks.context_agent.general_task').ContextAgentGeneralTaskOutput
    ) == {
        'score': 0.0,
        'extra_fields': {},
        'final_answer': None,
        'answer_score': 0.0,
        'constraint_score': None,
        'answer_judge': {},
        'constraint_judge': {},
        'grading_style': None,
        'judge_unavailable': False,
        'evaluation_skipped': False,
        'rationale': None,
        'requirement_status': [],
        'graded_on': None,
        'criteria_passed': None,
        'criteria_total': None,
        'judge_rounds': None,
        'criteria': [],
        'workspace_artifact': None,
        'termination_reason': 'ENV_DONE',
    }


@pytest.mark.asyncio
async def test_context_agent_task_files_prefers_task_scoped_wedata_key(monkeypatch) -> None:
    import nexus.extensions.tasks.context_agent.task_files_auth as auth_module

    calls = []

    async def fake_sign(spec, *, auth):
        calls.append({'spec': spec, 'auth': auth})
        return {'urls': {spec.path: 'https://cos.example/signed'}}

    monkeypatch.setattr(auth_module, '_sign_remote_task_files', fake_sign)
    files = task_files_from_spec(
        {
            'type': 'zip',
            'path': 'blobs/case-001.zip',
            'size': 123,
            'remote': {'namespace': 'ns', 'dataset': 'ds', 'version': 'v1', 'split': 'test'},
        }
    )
    assert isinstance(files, RemoteZipTaskFiles)

    scoped = bind_task_wedata_key(
        files,
        '{"subject":"task-user","key":"task-cmk","id":"task-id"}',
    )
    assert scoped is not None
    session = await scoped.create_session(run_id='run-1')
    signed_url = await scoped._get_signed_url()

    expected_auth = {
        'kind': 'wedata_key',
        'key': {'subject': 'task-user', 'key': 'task-cmk', 'id': 'task-id'},
    }
    assert session.auth == expected_auth
    assert signed_url == 'https://cos.example/signed'
    assert calls == [{'spec': scoped.spec, 'auth': expected_auth}]


def test_context_agent_task_files_without_task_key_keeps_default_behavior() -> None:
    files = task_files_from_spec({
        'type': 'zip',
        'path': 'blobs/case-001.zip',
        'size': 123,
        'remote': {'namespace': 'ns', 'dataset': 'ds', 'version': 'v1', 'split': 'test'},
    })

    assert bind_task_wedata_key(files, None) is files
    assert bind_task_wedata_key(files, '') is files


# ---------------------------------------------------------------------------
# Tool contract
# ---------------------------------------------------------------------------


def test_tool_contract_is_seven_path_native_tools() -> None:
    schemas = build_tool_schemas()
    names = [schema['function']['name'] for schema in schemas]

    assert names == list(GENERAL_TOOL_ORDER)
    assert names == [
        'curl',
        'bash',
        'python',
        'file_search_msearch',
        'file_search_mclick',
        'grep',
        'read_lines',
    ]
    # Path-native: nothing addresses a pre-loaded document set.
    assert 'list_documents' not in names
    assert 'doc_outline' not in names
    read_lines = next(s for s in schemas if s['function']['name'] == 'read_lines')
    assert 'path' in read_lines['function']['parameters']['properties']
    assert 'file' not in read_lines['function']['parameters']['properties']


def test_enabled_tools_whitelist_falls_back_when_it_misses() -> None:
    assert _resolve_enabled_tools(None) == list(GENERAL_TOOL_ORDER)
    assert _resolve_enabled_tools(['grep', 'read_lines']) == ['grep', 'read_lines']
    # Canonical order wins over the caller's ordering.
    assert _resolve_enabled_tools(['read_lines', 'grep']) == ['grep', 'read_lines']
    assert _resolve_enabled_tools(['scan', 'notepad']) == list(GENERAL_TOOL_ORDER)


# ---------------------------------------------------------------------------
# Language: the 'cn' vs 'zh' seam
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('explicit', ['cn', 'zh', 'zh-CN', 'chinese', '中文'])
def test_chinese_aliases_resolve_to_the_zh_label(explicit: str) -> None:
    # HyGeneralToolSystem selects the Chinese template with startswith("zh"),
    # so the context_agent chain's "cn" label must be translated, not passed on.
    assert resolve_lang(explicit, 'irrelevant') == 'zh'


def test_chinese_question_gets_the_chinese_prompt_and_manifest(tmp_path) -> None:
    lang = resolve_lang(None, 'Apollo 的默认监听端口是多少？')
    assert lang == 'zh'

    prompt = build_general_system_prompt(language=lang, max_rounds=20, workspace='./workspace', cwd=str(tmp_path))
    manifest = build_file_manifest(parse_declared_files(['https://example.com/a.pdf']), '端口是多少？', lang)

    assert '你' in prompt
    assert '{max_rounds}' not in prompt
    assert '20' in prompt
    assert '网络链接' in manifest


def test_english_question_gets_the_english_prompt() -> None:
    assert resolve_lang(None, 'What is the default listening port?') == 'en'
    manifest = build_file_manifest(parse_declared_files(['https://example.com/a.pdf']), 'Which port?', 'en')
    assert 'URL' in manifest


# ---------------------------------------------------------------------------
# Manifest / declared files
# ---------------------------------------------------------------------------


def test_manifest_advertises_uris_and_never_the_content(tmp_path) -> None:
    doc = tmp_path / 'handbook.md'
    doc.write_text(DOCUMENT, encoding='utf-8')

    declared = parse_declared_files([{'name': 'handbook.md', 'uri': str(doc)}])
    manifest = build_file_manifest(declared, 'Which port?', 'en')

    assert str(doc) in manifest
    assert 'Which port?' in manifest
    # The whole premise: no content is injected.
    assert '7421' not in manifest
    assert 'NOT included' in manifest


def test_declared_files_expand_relative_paths_and_leave_urls_alone(tmp_path) -> None:
    (tmp_path / 'notes.md').write_text('hi', encoding='utf-8')

    declared = parse_declared_files(['notes.md', 'https://example.com/x.pdf?v=2'], base=tmp_path)

    local, url = declared
    assert local.uri == str(tmp_path / 'notes.md')  # absolute, because bash gets it verbatim
    assert local.is_url is False
    assert url.uri == 'https://example.com/x.pdf?v=2'
    assert url.is_url is True
    assert url.name == 'x.pdf'  # derived from the URL, query stripped
    assert url.local_path is None


def test_staging_copies_into_the_workspace_and_dedupes_basenames(tmp_path) -> None:
    first_dir = tmp_path / 'a'
    second_dir = tmp_path / 'b'
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / 'doc.md').write_text('first', encoding='utf-8')
    (second_dir / 'doc.md').write_text('second', encoding='utf-8')

    declared = parse_declared_files([str(first_dir / 'doc.md'), str(second_dir / 'doc.md')])
    workspace = prepare_workspace(tmp_path / 'rollout')
    restaged, staged = stage_declared_files(declared, workspace)

    assert [f.uri for f in restaged] == ['./workspace/doc.md', './workspace/doc_2.md']
    assert len(staged) == 2
    # Two inputs sharing a basename must not collapse into one file.
    assert (workspace.docs_dir / 'doc.md').read_text(encoding='utf-8') == 'first'
    assert (workspace.docs_dir / 'doc_2.md').read_text(encoding='utf-8') == 'second'


def test_staging_leaves_a_missing_source_pointing_at_what_the_dataset_asked_for(tmp_path) -> None:
    declared = parse_declared_files([str(tmp_path / 'gone.md')])
    workspace = prepare_workspace(tmp_path / 'rollout')

    restaged, staged = stage_declared_files(declared, workspace)

    assert staged == []
    assert restaged[0].uri == str(tmp_path / 'gone.md')


# ---------------------------------------------------------------------------
# The tools, for real
# ---------------------------------------------------------------------------


async def test_read_lines_grep_and_search_work_on_a_real_file(tmp_path) -> None:
    doc = tmp_path / 'handbook.md'
    doc.write_text(DOCUMENT, encoding='utf-8')
    declared = parse_declared_files([str(doc)])
    state = _state(tmp_path, declared)

    try:
        # 0-based, end-exclusive; L0 is physical line 1.
        read = await dispatch_tool(name='read_lines', arguments={'path': str(doc), 'start': 3, 'end': 4}, state=state)
        assert 'L3: Unless overridden' in read
        assert '7421' in read

        # Case-insensitive substring search, 0-based line numbers.
        grep = await dispatch_tool(name='grep', arguments={'pattern': 'LISTENING PORT'}, state=state)
        assert 'L3' in grep
        assert '7421' in grep

        # BM25 retrieval hands back an expandable pointer.
        found = await dispatch_tool(
            name='file_search_msearch',
            arguments={'queries': ['default listening port', 'apollo.toml']},
            state=state,
        )
        assert 'Mclick Target' in found
        pointer = found.split('Mclick Target: "')[1].split('"')[0]

        expanded = await dispatch_tool(name='file_search_mclick', arguments={'pointers': [pointer]}, state=state)
        assert 'expanded with context' in expanded
    finally:
        state.close()


async def test_grep_recovers_a_regex_the_model_forgot_to_flag(tmp_path) -> None:
    doc = tmp_path / 'handbook.md'
    doc.write_text(DOCUMENT, encoding='utf-8')
    state = _state(tmp_path, parse_declared_files([str(doc)]))

    try:
        out = await dispatch_tool(name='grep', arguments={'pattern': 'LRU|eviction'}, state=state)
        # A literal search for "LRU|eviction" matches nothing; the retry as a
        # regex must rescue it and say so instead of reporting an absence.
        assert 'no LITERAL match' in out
        assert 'L7' in out
    finally:
        state.close()


async def test_python_repl_keeps_state_and_bash_shares_its_cwd(tmp_path) -> None:
    state = _state(tmp_path, [])
    try:
        python_executable = await dispatch_tool(
            name='python',
            arguments={'code': 'import os, sys; print(os.path.realpath(sys.executable))'},
            state=state,
        )
        bash_python = await dispatch_tool(
            name='bash',
            arguments={'command': "python -c 'import os, sys; print(os.path.realpath(sys.executable))'"},
            state=state,
        )
        assert python_executable == bash_python

        first = await dispatch_tool(name='python', arguments={'code': 'x = 6 * 7'}, state=state)
        assert '(no output)' in first

        second = await dispatch_tool(name='python', arguments={'code': 'x'}, state=state)
        assert '42' in second

        # The python tool writes into the workspace the shell tool then reads.
        await dispatch_tool(
            name='python',
            arguments={'code': "open('./workspace/out.txt', 'w').write('converted')"},
            state=state,
        )
        listed = await dispatch_tool(name='bash', arguments={'command': 'cat ./workspace/out.txt'}, state=state)
        assert 'converted' in listed
    finally:
        state.close()


async def test_subprocess_tools_do_not_see_credentials(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-should-not-leak')
    monkeypatch.setenv('CTXAGENT_HARMLESS', 'visible')
    state = _state(tmp_path, [])

    try:
        out = await dispatch_tool(
            name='bash',
            arguments={'command': 'echo "[$OPENAI_API_KEY][$CTXAGENT_HARMLESS]"'},
            state=state,
        )
        assert 'sk-should-not-leak' not in out
        assert 'visible' in out
    finally:
        state.close()


async def test_binary_file_is_refused_with_a_conversion_hint(tmp_path) -> None:
    fake_pdf = tmp_path / 'report.pdf'
    fake_pdf.write_bytes(b'%PDF-1.7\n' + bytes(range(256)))
    state = _state(tmp_path, parse_declared_files([str(fake_pdf)]))

    try:
        out = await dispatch_tool(
            name='read_lines', arguments={'path': str(fake_pdf), 'start': 0, 'end': 5}, state=state
        )
        assert 'PDF' in out
        assert 'pypdf' in out  # names the right library, not just "it's binary"

        with pytest.raises(FileReadError):
            state.read_lines(fake_pdf)
    finally:
        state.close()


async def test_search_without_a_path_explains_an_empty_corpus(tmp_path) -> None:
    state = _state(tmp_path, parse_declared_files(['https://example.com/report.pdf']))
    try:
        out = await dispatch_tool(name='grep', arguments={'pattern': 'anything'}, state=state)
        assert 'URLs you have not downloaded' in out
    finally:
        state.close()


async def test_curl_refuses_a_local_path_instead_of_downloading_it(tmp_path) -> None:
    state = _state(tmp_path, [])
    try:
        out = await dispatch_tool(name='curl', arguments={'url': str(tmp_path / 'x.md')}, state=state)
        assert 'only handles http(s)' in out
    finally:
        state.close()


# ---------------------------------------------------------------------------
# The rollout loop
# ---------------------------------------------------------------------------


async def test_rollout_appends_assistant_verbatim_and_answers_every_tool_call(tmp_path) -> None:
    doc = tmp_path / 'handbook.md'
    doc.write_text(DOCUMENT, encoding='utf-8')

    tool_turn = _Message(
        content='looking it up',
        tool_calls=[
            _tool_call('call-a', 'grep', '{"pattern": "listening port"}'),
            _tool_call('call-b', 'read_lines', f'{{"path": "{doc}", "start": 3, "end": 5}}'),
        ],
    )
    client = _ScriptedClient(tool_turn, _Message(content='The default port is 7421.'))

    working_dir = tmp_path / 'runtime'
    async with ExecutionContext(
        openai_clients={'main': client},
        runtime_provider=LocalRuntimeProvider(),
    ):
        output = await run_context_agent_general(
            ContextAgentGeneralInput(
                user_prompt='What is the default listening port?',
                context_files=[{'name': 'handbook.md', 'uri': str(doc)}],
                working_dir=str(working_dir),
                max_tool_rounds=5,
            )
        )

    assert output.final_answer == 'The default port is 7421.'
    assert output.language == 'en'
    assert output.num_files == 1
    assert output.tool_rounds_used == 1
    assert output.tool_calls_executed == 2
    assert output.tool_calls_skipped == 0
    assert output.hit_max_turn_limit is False

    roles = [m['role'] for m in output.main_trajectory]
    assert roles == ['system', 'user', 'assistant', 'tool', 'tool', 'assistant']

    # The assistant turn is what the client produced, not a reconstruction:
    # HyTuner forbids rewriting history and the chat template pairs each
    # tool_call with a tool message BY POSITION.
    assert output.main_trajectory[2] == tool_turn.model_dump(exclude_none=True)
    assert [m['tool_call_id'] for m in output.main_trajectory[3:5]] == ['call-a', 'call-b']
    assert '7421' in output.main_trajectory[3]['content']
    assert 'L3:' in output.main_trajectory[4]['content']

    # The first user turn is the manifest — a name and a path, no content. It
    # advertises the staged copy, since staging is on by default.
    assert './workspace/handbook.md' in output.main_trajectory[1]['content']
    assert '7421' not in output.main_trajectory[1]['content']
    # A path the manifest never advertised still resolves, which is what let the
    # read_lines call above pass the dataset's original absolute path.
    assert 'L3:' in output.main_trajectory[4]['content']

    # Only the runtime workspace is materialized; Nexus owns trajectory/output
    # persistence on the host side.
    assert (working_dir / 'workspace' / 'handbook.md').exists()
    assert not (working_dir / 'file_manifest.md').exists()
    assert not (working_dir / 'final_answer.md').exists()


async def test_rollout_uses_stop_finish_reason_for_final_text(tmp_path) -> None:
    doc = tmp_path / 'handbook.md'
    doc.write_text(DOCUMENT, encoding='utf-8')
    client = _ScriptedClient(_Message(content='The default port is 7421.'), finish_reasons=['stop'])

    async with ExecutionContext(
        openai_clients={'main': client},
        runtime_provider=LocalRuntimeProvider(),
    ):
        output = await run_context_agent_general(
            ContextAgentGeneralInput(
                user_prompt='Which port?',
                context_files=[{'name': 'handbook.md', 'uri': str(doc)}],
                working_dir=str(tmp_path / 'runtime'),
            )
        )

    assert output.final_answer == 'The default port is 7421.'
    assert output.exit_status == TerminationReason.SUCCESS


async def test_rollout_executes_tool_calls_when_finish_reason_is_tool_calls(tmp_path) -> None:
    doc = tmp_path / 'handbook.md'
    doc.write_text(DOCUMENT, encoding='utf-8')
    tool_turn = _Message(tool_calls=[_tool_call('call-a', 'grep', '{"pattern": "port"}')])
    client = _ScriptedClient(
        tool_turn,
        _Message(content='The default port is 7421.'),
        finish_reasons=['tool_calls', 'stop'],
    )

    async with ExecutionContext(
        openai_clients={'main': client},
        runtime_provider=LocalRuntimeProvider(),
    ):
        output = await run_context_agent_general(
            ContextAgentGeneralInput(
                user_prompt='Which port?',
                context_files=[{'name': 'handbook.md', 'uri': str(doc)}],
                working_dir=str(tmp_path / 'runtime'),
            )
        )

    assert output.final_answer == 'The default port is 7421.'
    assert output.exit_status == TerminationReason.SUCCESS
    assert output.tool_calls_executed == 1


async def test_rollout_rejects_tool_finish_reason_without_tool_calls(tmp_path) -> None:
    doc = tmp_path / 'handbook.md'
    doc.write_text(DOCUMENT, encoding='utf-8')
    client = _ScriptedClient(_Message(content='I will inspect it next.'), finish_reasons=['tool_calls'])

    async with ExecutionContext(
        openai_clients={'main': client},
        runtime_provider=LocalRuntimeProvider(),
    ):
        output = await run_context_agent_general(
            ContextAgentGeneralInput(
                user_prompt='Which port?',
                context_files=[{'name': 'handbook.md', 'uri': str(doc)}],
                working_dir=str(tmp_path / 'runtime'),
            )
        )

    assert output.final_answer is None
    assert output.exit_status == TerminationReason.FORMAT_ERROR


async def test_rollout_treats_length_finish_reason_as_output_length_exceeded(tmp_path) -> None:
    doc = tmp_path / 'handbook.md'
    doc.write_text(DOCUMENT, encoding='utf-8')
    client = _ScriptedClient(_Message(content='partial answer'), finish_reasons=['length'])

    async with ExecutionContext(
        openai_clients={'main': client},
        runtime_provider=LocalRuntimeProvider(),
    ):
        output = await run_context_agent_general(
            ContextAgentGeneralInput(
                user_prompt='Which port?',
                context_files=[{'name': 'handbook.md', 'uri': str(doc)}],
                working_dir=str(tmp_path / 'runtime'),
            )
        )

    assert output.final_answer is None
    assert output.exit_status == TerminationReason.OUTPUT_LENGTH_EXCEEDED
    assert output.response_budget_exhausted is True


async def test_rollout_treats_content_filter_finish_reason_as_provider_error(tmp_path) -> None:
    doc = tmp_path / 'handbook.md'
    doc.write_text(DOCUMENT, encoding='utf-8')
    client = _ScriptedClient(_Message(content='filtered'), finish_reasons=['content_filter'])

    async with ExecutionContext(
        openai_clients={'main': client},
        runtime_provider=LocalRuntimeProvider(),
    ):
        output = await run_context_agent_general(
            ContextAgentGeneralInput(
                user_prompt='Which port?',
                context_files=[{'name': 'handbook.md', 'uri': str(doc)}],
                working_dir=str(tmp_path / 'runtime'),
            )
        )

    assert output.final_answer is None
    assert output.exit_status == TerminationReason.PROVIDER_ERROR


async def test_rollout_does_not_execute_tool_calls_when_finish_reason_is_stop(tmp_path) -> None:
    doc = tmp_path / 'handbook.md'
    doc.write_text(DOCUMENT, encoding='utf-8')
    client = _ScriptedClient(
        _Message(
            content='The default port is 7421.',
            tool_calls=[_tool_call('call-a', 'grep', '{"pattern": "port"}')],
        ),
        finish_reasons=['stop'],
    )

    async with ExecutionContext(
        openai_clients={'main': client},
        runtime_provider=LocalRuntimeProvider(),
    ):
        output = await run_context_agent_general(
            ContextAgentGeneralInput(
                user_prompt='Which port?',
                context_files=[{'name': 'handbook.md', 'uri': str(doc)}],
                working_dir=str(tmp_path / 'runtime'),
            )
        )

    assert output.final_answer == 'The default port is 7421.'
    assert output.exit_status == TerminationReason.SUCCESS
    assert output.tool_calls_requested == 0
    assert output.tool_calls_executed == 0


async def test_rollout_stages_local_files_into_its_own_workspace(tmp_path) -> None:
    doc = tmp_path / 'handbook.md'
    doc.write_text(DOCUMENT, encoding='utf-8')
    client = _ScriptedClient(_Message(content='done'))

    working_dir = tmp_path / 'runtime'
    async with ExecutionContext(
        openai_clients={'main': client},
        runtime_provider=LocalRuntimeProvider(),
    ):
        output = await run_context_agent_general(
            ContextAgentGeneralInput(
                user_prompt='Which port?',
                context_files=[{'name': 'handbook.md', 'uri': str(doc)}],
                working_dir=str(working_dir),
            )
        )

    assert output.staged_files == [{'name': 'handbook.md', 'source': str(doc)}]
    assert (working_dir / 'workspace' / 'handbook.md').exists()
    # The manifest points at the copy, so bash cannot clobber the dataset.
    assert './workspace/handbook.md' in output.main_trajectory[1]['content']


async def test_parallel_cap_keeps_one_response_per_generated_tool_call(tmp_path) -> None:
    doc = tmp_path / 'handbook.md'
    doc.write_text(DOCUMENT, encoding='utf-8')
    tool_turn = _Message(
        tool_calls=[
            _tool_call('call-a', 'grep', '{"pattern": "port"}'),
            _tool_call('call-b', 'read_lines', '{"path": "./workspace/handbook.md", "start": 0, "end": 2}'),
        ]
    )
    client = _ScriptedClient(tool_turn, _Message(content='7421'))

    async with ExecutionContext(
        openai_clients={'main': client},
        runtime_provider=LocalRuntimeProvider(),
    ):
        output = await run_context_agent_general(
            ContextAgentGeneralInput(
                user_prompt='Which port?',
                context_files=[{'name': 'handbook.md', 'uri': str(doc)}],
                working_dir=str(tmp_path / 'runtime'),
                max_parallel_calls=1,
            )
        )

    tool_messages = [message for message in output.main_trajectory if message['role'] == 'tool']
    assert [message['tool_call_id'] for message in tool_messages] == ['call-a', 'call-b']
    assert 'was not executed' in tool_messages[1]['content']
    assert output.tool_calls_requested == 2
    assert output.tool_calls_executed == 1
    assert output.tool_calls_skipped == 1


async def test_rollout_forces_a_final_answer_when_the_round_budget_runs_out(tmp_path) -> None:
    doc = tmp_path / 'handbook.md'
    doc.write_text(DOCUMENT, encoding='utf-8')

    spin = _Message(content='still looking', tool_calls=[_tool_call('c1', 'grep', '{"pattern": "port"}')])
    client = _ScriptedClient(spin, spin, _Message(content='Best effort: 7421.'))

    async with ExecutionContext(
        openai_clients={'main': client},
        runtime_provider=LocalRuntimeProvider(),
    ):
        output = await run_context_agent_general(
            ContextAgentGeneralInput(
                user_prompt='Which port?',
                context_files=[{'name': 'handbook.md', 'uri': str(doc)}],
                working_dir=str(tmp_path / 'runtime'),
                max_tool_rounds=2,
            )
        )

    assert output.hit_max_turn_limit is True
    assert output.forced_finalization is True
    assert output.final_answer == 'Best effort: 7421.'
    assert output.tool_rounds_used == 2
    # The fallback turn is a plain user cue, but keeps the same tool schema so
    # HyTuner's rendered prefix remains append-only.
    assert output.main_trajectory[-2]['role'] == 'user'
    assert 'all available tool rounds' in output.main_trajectory[-2]['content']
    assert client.calls[-1]['tools']


async def test_rollout_survives_a_response_budget_overflow(tmp_path) -> None:
    from nexus.core import ContextLengthExceededError

    class _OverflowClient(_ScriptedClient):
        async def create_completion(self, messages, **kwargs):
            raise ContextLengthExceededError('response budget exhausted')

    async with ExecutionContext(
        openai_clients={'main': _OverflowClient()},
        runtime_provider=LocalRuntimeProvider(),
    ):
        output = await run_context_agent_general(
            ContextAgentGeneralInput(
                user_prompt='Which port?',
                working_dir=str(tmp_path / 'runtime'),
                context_files=[],
                max_tool_rounds=3,
            )
        )

    # An overflow must not raise: that would mark the rollout TASK_FAILED and
    # feed a false negative into GRPO. It scores as "no answer" instead.
    assert output.response_budget_exhausted is True
    assert output.final_answer is None
    assert output.forced_finalization is False


async def test_task_persona_is_appended_not_substituted(tmp_path) -> None:
    client = _ScriptedClient(_Message(content='ok'))

    async with ExecutionContext(
        openai_clients={'main': client},
        runtime_provider=LocalRuntimeProvider(),
    ):
        await run_context_agent_general(
            ContextAgentGeneralInput(
                user_prompt='Which port?',
                working_dir=str(tmp_path / 'runtime'),
                context_files=[],
                system_messages=[{'role': 'system', 'content': 'You speak like a lawyer.'}],
                system_prompt_lang='en',
            )
        )

    system_prompt = client.calls[0]['messages'][0]['content']
    assert system_prompt.endswith('You speak like a lawyer.')
    assert 'read_lines' in system_prompt  # the agent's own prompt survived


async def test_rollout_replays_dialogue_with_manifest_on_first_user_turn(tmp_path) -> None:
    doc = tmp_path / 'handbook.md'
    doc.write_text(DOCUMENT, encoding='utf-8')
    client = _ScriptedClient(_Message(content='final answer'))

    async with ExecutionContext(
        openai_clients={'main': client},
        runtime_provider=LocalRuntimeProvider(),
    ):
        output = await run_context_agent_general(
            ContextAgentGeneralInput(
                user_prompt='This standalone question must not be appended.',
                context_files=[{'name': 'handbook.md', 'uri': str(doc)}],
                context_dialogue=[
                    {
                        'role': 'user',
                        'content': (
                            '[Attachment: `handbook.md` at `/old/dataset/handbook.md`]\n\nPlease remember the handbook.'
                        ),
                    },
                    {'role': 'assistant', 'content': 'I will remember it.'},
                    {'role': 'user', 'content': 'Which port does it use?'},
                ],
                working_dir=str(tmp_path / 'runtime'),
            )
        )

    messages = client.calls[0]['messages']
    assert [message['role'] for message in messages] == ['system', 'user', 'assistant', 'user']
    assert 'You have been given 1 file:' in messages[1]['content']
    assert messages[1]['content'].count('./workspace/handbook.md') == 2
    assert 'Please remember the handbook.' in messages[1]['content']
    assert messages[2]['content'] == 'I will remember it.'
    assert messages[3]['content'] == 'Which port does it use?'
    assert all('This standalone question must not be appended.' not in message['content'] for message in messages)
    assert output.final_answer == 'final answer'


@pytest.mark.parametrize(
    'dialogue',
    [
        [
            {'role': 'assistant', 'content': 'Invalid opening.'},
            {'role': 'user', 'content': 'Question from dialogue.'},
        ],
        [
            {'role': 'user', 'content': 'Valid opening.'},
            {'role': 'assistant', 'content': 'Invalid ending.'},
        ],
    ],
)
async def test_invalid_dialogue_falls_back_to_context_question(tmp_path, dialogue, caplog) -> None:
    client = _ScriptedClient(_Message(content='fallback answer'))

    async with ExecutionContext(
        openai_clients={'main': client},
        runtime_provider=LocalRuntimeProvider(),
    ):
        output = await run_context_agent_general(
            ContextAgentGeneralInput(
                user_prompt='Use this fallback question.',
                context_files=[],
                context_dialogue=dialogue,
                working_dir=str(tmp_path / 'runtime'),
            )
        )

    messages = client.calls[0]['messages']
    assert [message['role'] for message in messages] == ['system', 'user']
    assert messages[1]['content'] == 'Use this fallback question.'
    assert 'ignoring dialogue: expected it to start and end with a user turn' in caplog.text
    assert output.final_answer == 'fallback answer'


def test_task_system_text_ignores_non_system_turns() -> None:
    assert _task_system_text([{'role': 'user', 'content': 'nope'}]) == ''
    assert _task_system_text('bare string') == 'bare string'
    assert _task_system_text(None) == ''
    assert (
        _task_system_text([
            {'role': 'system', 'content': 'a'},
            {'role': 'assistant', 'content': 'skip'},
            {'role': 'system', 'content': 'b'},
        ])
        == 'a\n\nb'
    )


async def test_env_pins_the_prompt_language_over_the_sample(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('CONTEXTAGENT_GENERAL_SYSTEM_LANG', 'zh')
    client = _ScriptedClient(_Message(content='ok'))

    async with ExecutionContext(
        openai_clients={'main': client},
        runtime_provider=LocalRuntimeProvider(),
    ):
        output = await run_context_agent_general(
            ContextAgentGeneralInput(
                user_prompt='What is the default port?',
                working_dir=str(tmp_path / 'runtime'),
                context_files=[],
                language='en',
            )
        )

    assert output.language == 'zh'


# ---------------------------------------------------------------------------
# Task scoring
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Path privacy
# ---------------------------------------------------------------------------


def test_scrub_swaps_the_directory_and_keeps_the_basename() -> None:
    real = '/apdcephfs_zwfy6/share_302970870/hunyuan/changzelv/dev/nexus/notes.md'
    scrubbed = path_privacy.scrub(f'reading {real} now')

    assert 'apdcephfs' not in scrubbed
    assert 'changzelv' not in scrubbed
    assert path_privacy.FAKE_ROOT in scrubbed
    # The basename survives, so a file the model is asked to cite keeps its name.
    assert 'notes.md' in scrubbed
    # Bijective: what the model hands back maps to the real filesystem.
    assert path_privacy.restore(scrubbed) == f'reading {real} now'


def test_scrub_leaves_neutral_and_relative_paths_alone() -> None:
    for text in ('./workspace/doc.md', '/tmp/pytest-of-root/x/doc.md', '/usr/bin/curl'):
        assert path_privacy.scrub(text) == text


def test_scrub_is_stable_across_calls_for_one_directory() -> None:
    a = path_privacy.scrub('/apdcephfs_x/hunyuan/proj/one.md')
    b = path_privacy.scrub('/apdcephfs_x/hunyuan/proj/two.md')

    # Same directory -> same fake stand-in, so pointers stay comparable.
    assert a.rsplit('/', 1)[0] == b.rsplit('/', 1)[0]


def test_restore_obj_walks_nested_tool_arguments() -> None:
    scrubbed = path_privacy.scrub('/apdcephfs_q/hunyuan/data/report.pdf')
    args = {'path': [scrubbed], 'nested': {'files': (scrubbed,)}, 'start': 0}

    restored = path_privacy.restore_obj(args)

    assert restored['path'] == ['/apdcephfs_q/hunyuan/data/report.pdf']
    assert restored['nested']['files'] == ('/apdcephfs_q/hunyuan/data/report.pdf',)
    assert restored['start'] == 0


def test_scrub_can_be_disabled_but_restore_stays_on(monkeypatch) -> None:
    real = '/apdcephfs_r/hunyuan/data/x.md'
    scrubbed = path_privacy.scrub(real)
    assert scrubbed != real

    monkeypatch.setenv('CTXAGENT_SCRUB_PATHS', '0')
    assert path_privacy.scrub(real) == real
    # A run that scrubbed earlier must keep restoring, or the model's paths
    # stop resolving mid-rollout.
    assert path_privacy.restore(scrubbed) == real


async def test_rollout_stages_host_paths_before_exposing_them_to_the_model(tmp_path) -> None:
    # The Task/Agent boundary passes a host source path, but the runtime
    # manifest exposes only the staged relative path.
    cluster = tmp_path / 'apdcephfs_zwfy6' / 'hunyuan' / 'changzelv' / 'data'
    cluster.mkdir(parents=True)
    doc = cluster / 'handbook.md'
    doc.write_text(DOCUMENT, encoding='utf-8')

    client = _ScriptedClient(
        _Message(content='reading', tool_calls=[_tool_call('c1', 'grep', '{"pattern": "listening port"}')]),
        _Message(content='7421'),
    )

    async with ExecutionContext(openai_clients={'main': client}, runtime_provider=LocalRuntimeProvider()):
        output = await run_context_agent_general(
            ContextAgentGeneralInput(
                user_prompt='Which port?',
                context_files=[{'name': 'handbook.md', 'uri': str(doc)}],
                working_dir=str(tmp_path / 'runtime'),
                max_tool_rounds=3,
            )
        )

    sent = '\n'.join(str(m.get('content') or '') for call in client.calls for m in call['messages'])
    assert 'apdcephfs' not in sent
    assert 'changzelv' not in sent
    assert './workspace/handbook.md' in sent
    assert 'handbook.md' in sent  # the basename the model must cite survives

    # The grep ran against the staged runtime copy.
    tool_message = next(m for m in output.main_trajectory if m['role'] == 'tool')
    assert '7421' in tool_message['content']
    assert 'apdcephfs' not in tool_message['content']
    assert output.final_answer == '7421'


# ---------------------------------------------------------------------------
# Scoring: the ported evaluator suite
# ---------------------------------------------------------------------------


class _FakeJudge:
    """A judge whose replies are scripted; records the prompts it was sent."""

    def __init__(self, *replies: str):
        self._replies = list(replies)
        self.prompts: list[str] = []

    async def chat(self, messages, **_kwargs):
        self.prompts.append(messages[-1]['content'] if messages else '')
        return self._replies.pop(0) if self._replies else ''


class _ForbiddenJudge:
    async def chat(self, *_args, **_kwargs):
        raise AssertionError('a rule-based benchmark must not call a judge model')


def _agent_returning(answer: str | None, exit_status: str | TerminationReason | None = TerminationReason.SUCCESS):
    async def fake_run_agent(**_kwargs):
        return SimpleNamespace(final_answer=answer, exit_status=exit_status)

    return fake_run_agent


async def _score(monkeypatch, judge, **fields):
    monkeypatch.setattr(task_module, 'resolve_judge_llm', lambda _ctx=None: judge)
    async with ExecutionContext(openai_clients={}, runtime_provider=LocalRuntimeProvider()):
        return await run_context_agent_general_task(ContextAgentGeneralTaskInput(**fields))


def test_the_suite_covers_twenty_three_datasets_in_three_styles() -> None:
    assert len(EVALUATORS) == 23
    assert set(STYLES.values()) == {'rule', 'reference', 'rubric'}
    assert set(STYLES) == set(EVALUATORS)
    # Rule-based benchmarks are deterministic and cost nothing to grade.
    assert {'GraphWalks', 'LongBenchV2', 'MRCR_400', 'MRCR_800', 'oolong'} == RULE_BASED


async def test_rule_based_dataset_is_graded_without_any_judge_model(monkeypatch) -> None:
    output = await _score(
        monkeypatch,
        _ForbiddenJudge(),
        run_agent=_agent_returning('After weighing the options: The correct answer is (D)'),
        context_question='Which option?',
        dataset_name='LongBenchV2',
        answer='D',
    )

    assert output.score == 1.0
    assert output.grading_style == 'rule'
    assert output.judge_unavailable is False


async def test_evaluation_can_be_disabled_without_resolving_a_judge(monkeypatch) -> None:
    def forbidden_resolve(*_args, **_kwargs):
        raise AssertionError('evaluate=false must not resolve a judge')

    async def forbidden_evaluate(*_args, **_kwargs):
        raise AssertionError('evaluate=false must not call an evaluator')

    monkeypatch.setattr(task_module, 'resolve_judge_llm', forbidden_resolve)
    monkeypatch.setattr(task_module, 'evaluate', forbidden_evaluate)

    async with ExecutionContext(openai_clients={}, runtime_provider=LocalRuntimeProvider()):
        output = await run_context_agent_general_task(
            ContextAgentGeneralTaskInput(
                run_agent=_agent_returning('The default port is 7421.'),
                context_question='Which port?',
                dataset_name='GDPpdf',
                rubrics=['mentions 7421'],
                evaluate=False,
            )
        )

    assert output.final_answer == 'The default port is 7421.'
    assert output.score == 0.0
    assert output.grading_style is None
    assert output.judge_unavailable is False
    assert output.evaluation_skipped is True
    assert output.rationale == 'Evaluation disabled'


@pytest.mark.parametrize('exit_status', [TerminationReason.TRUNCATION, TerminationReason.OUTPUT_LENGTH_EXCEEDED])
async def test_token_limit_exit_skips_evaluation(monkeypatch, exit_status: TerminationReason) -> None:
    def forbidden_resolve(*_args, **_kwargs):
        raise AssertionError('token-limit exits must not resolve a judge')

    async def forbidden_evaluate(*_args, **_kwargs):
        raise AssertionError('token-limit exits must not call an evaluator')

    monkeypatch.setattr(task_module, 'resolve_judge_llm', forbidden_resolve)
    monkeypatch.setattr(task_module, 'evaluate', forbidden_evaluate)

    async with ExecutionContext(openai_clients={}, runtime_provider=LocalRuntimeProvider()):
        output = await run_context_agent_general_task(
            ContextAgentGeneralTaskInput(
                run_agent=_agent_returning('partial answer', exit_status),
                context_question='Which port?',
                dataset_name='GDPpdf',
                rubrics=['mentions 7421'],
            )
        )

    assert output.score == 0.0
    assert output.evaluation_skipped is True
    assert output.judge_unavailable is False
    assert exit_status.value in output.rationale


async def test_max_steps_without_final_answer_skips_evaluation(monkeypatch) -> None:
    def forbidden_resolve(*_args, **_kwargs):
        raise AssertionError('empty max-step exits must not resolve a judge')

    async def forbidden_evaluate(*_args, **_kwargs):
        raise AssertionError('empty max-step exits must not call an evaluator')

    monkeypatch.setattr(task_module, 'resolve_judge_llm', forbidden_resolve)
    monkeypatch.setattr(task_module, 'evaluate', forbidden_evaluate)

    async with ExecutionContext(openai_clients={}, runtime_provider=LocalRuntimeProvider()):
        output = await run_context_agent_general_task(
            ContextAgentGeneralTaskInput(
                run_agent=_agent_returning(None, TerminationReason.MAX_ITERATIONS),
                context_question='Create a presentation.',
                dataset_name='GDPval',
                rubrics=['creates the deliverable'],
                judge_backend='deliverable',
            )
        )

    assert output.score == 0.0
    assert output.evaluation_skipped is True
    assert output.judge_unavailable is False
    assert output.rationale == 'Judge skipped because agent exited with MAX_STEPS without a final answer.'


async def test_longbenchv2_uses_the_official_regex_strictly(monkeypatch) -> None:
    # The official extractor accepts "The correct answer is D" but not the
    # looser "The answer is D" — worth pinning, since it changes the number.
    loose = await _score(
        monkeypatch,
        _ForbiddenJudge(),
        run_agent=_agent_returning('The answer is D'),
        context_question='Which option?',
        dataset_name='LongBenchV2',
        answer='D',
    )
    assert loose.score == 0.0


async def test_graphwalks_uses_strict_last_line_and_set_f1(monkeypatch) -> None:
    partial = await _score(
        monkeypatch,
        _ForbiddenJudge(),
        run_agent=_agent_returning("I traversed the graph.\nFinal Answer: [node_a, 'node_c', node_extra]"),
        context_question='Run BFS.',
        dataset_name='GraphWalks',
        answer='node_a, node_b, node_c',
    )
    assert partial.score == 0.666667
    assert partial.grading_style == 'rule'
    assert partial.judge_unavailable is False
    assert partial.rationale.startswith('GraphWalks set F1 = 0.6667')

    marker_not_last = await _score(
        monkeypatch,
        _ForbiddenJudge(),
        run_agent=_agent_returning('Final Answer: [node_a, node_b, node_c]\nDone.'),
        context_question='Run BFS.',
        dataset_name='GraphWalks',
        answer='node_a, node_b, node_c',
    )
    assert marker_not_last.score == 0.0
    assert 'format error' in marker_not_last.rationale


async def test_complexlong_picks_its_comparison_rule_from_the_answer_form(monkeypatch) -> None:
    # The four sets share one judge; only this sentence differs, and it is what
    # decides whether a given answer counts, so it must track metadata.answer_form.
    assert complexlong.compare_rule('ComplexLongTable', '集合') is complexlong.RULE_SET
    assert complexlong.compare_rule('ComplexLongTable', '标量') is complexlong.RULE_SCALAR
    assert complexlong.compare_rule('ComplexLongTree', 'node_sequence') is complexlong.RULE_ORDERED
    assert complexlong.compare_rule('ComplexLongGraph', 'G-A1/IN') is complexlong.RULE_INT
    assert complexlong.compare_rule('ComplexLongGraph', 'G-C2/ADJ') is complexlong.RULE_ADJ
    assert complexlong.compare_rule('ComplexLongDoc', 'llm_judge') is complexlong.RULE_OPEN
    # An unknown shape degrades to the open rule rather than raising.
    assert complexlong.compare_rule('ComplexLongTree', 'not_a_form') is complexlong.RULE_OPEN

    judge = _FakeJudge('【评分依据】：结论一致。\n【评分】：1分')
    output = await _score(
        monkeypatch,
        judge,
        run_agent=_agent_returning('<think>hidden</think>最终答案：qrg.001'),
        context_question='第 4 名的 research_group 是哪个？',
        dataset_name='ComplexLongTable',
        metadata={'dataset_name': 'ComplexLongTable', 'answer_form': '标量'},
        answer='qrg.001',
    )

    assert output.score == 1.0
    assert output.grading_style == 'reference'
    assert output.judge_unavailable is False
    # Reasoning is stripped, so the judge grades the conclusion and not the trace.
    assert 'hidden' not in judge.prompts[-1]
    assert complexlong.RULE_SCALAR in judge.prompts[-1]


async def test_complexlong_reports_a_judge_that_never_returns_a_score(monkeypatch) -> None:
    output = await _score(
        monkeypatch,
        _FakeJudge(*['no score line here'] * complexlong.MAX_RETRIES),
        run_agent=_agent_returning('some answer'),
        context_question='Which one?',
        dataset_name='ComplexLongGraph',
        metadata={'dataset_name': 'ComplexLongGraph', 'answer_form': 'G-A1/IN'},
        answer='12',
    )

    # A judge outage must be visible rather than averaged in as a genuine zero.
    assert output.score == 0.0
    assert output.judge_unavailable is True


async def test_mrcr_yields_a_genuine_float_from_the_prefix_gate(monkeypatch) -> None:
    prefix = 'xk39f '
    graded = await _score(
        monkeypatch,
        _ForbiddenJudge(),
        run_agent=_agent_returning(prefix + 'the quick brown fox jumps'),
        context_question='Reproduce it.',
        dataset_name='MRCR_800',
        answer=prefix + 'the quick brown fox leaps',
        metadata={'random_string_to_prepend': prefix},
    )
    assert 0.0 < graded.score < 1.0

    # Missing the prefix is a hard zero regardless of similarity.
    ungated = await _score(
        monkeypatch,
        _ForbiddenJudge(),
        run_agent=_agent_returning('the quick brown fox jumps'),
        context_question='Reproduce it.',
        dataset_name='MRCR_800',
        answer=prefix + 'the quick brown fox jumps',
        metadata={'random_string_to_prepend': prefix},
    )
    assert ungated.score == 0.0


async def test_rubric_dataset_ignores_its_answer_and_grades_the_criteria(monkeypatch) -> None:
    judge = _FakeJudge(
        '{"Grading Rationale": "all met", "List of Requirement Satisfaction Status": ["yes", "yes"],'
        ' "Overall Score": 1}'
    )
    # SurgeOODV21 ships a non-empty answer AND rubrics; the answer is
    # deliberately ignored, which is why dataset_name has to do the routing.
    output = await _score(
        monkeypatch,
        judge,
        run_agent=_agent_returning('The port is 7421 and eviction is LRU.'),
        context_question='Summarise.',
        dataset_name='SurgeOODV21',
        answer='a completely different answer',
        rubrics=['mentions 7421', 'mentions LRU'],
    )

    assert output.score == 1.0
    assert output.grading_style == 'rubric'
    assert output.requirement_status == ['yes', 'yes']
    # The rubrics reached the judge; the answer did not.
    assert 'mentions LRU' in judge.prompts[0]
    assert 'a completely different answer' not in judge.prompts[0]


async def test_judge_reply_in_a_markdown_fence_is_still_parsed(monkeypatch) -> None:
    judge = _FakeJudge(
        '```json\n{"Grading Rationale": "ok", "List of Requirement Satisfaction Status": ["yes"],'
        ' "Overall Score": 1}\n```'
    )
    output = await _score(
        monkeypatch,
        judge,
        run_agent=_agent_returning('7421'),
        context_question='Which port?',
        dataset_name='CLB',
        rubrics=['mentions 7421'],
    )

    assert output.score == 1.0


async def test_judge_outage_is_flagged_rather_than_averaged_in_as_a_zero(monkeypatch) -> None:
    # The ported judge retries then returns score=0 with a "Judge failed"
    # rationale. For RL that is indistinguishable from a real zero, so the task
    # detects the sentinel and says so.
    judge = _FakeJudge(*['not json at all'] * 12)
    output = await _score(
        monkeypatch,
        judge,
        run_agent=_agent_returning('7421'),
        context_question='Which port?',
        dataset_name='CLB',
        rubrics=['mentions 7421'],
    )

    assert output.score == 0.0
    assert output.judge_unavailable is True
    assert output.rationale.startswith('Judge failed')


async def test_an_evaluator_raising_degrades_instead_of_failing_the_task(monkeypatch) -> None:
    class _Exploding:
        async def chat(self, *_args, **_kwargs):
            raise RuntimeError('gateway down')

    async def boom(*_args, **_kwargs):
        raise RuntimeError('evaluator exploded')

    monkeypatch.setattr(task_module, 'evaluate', boom)
    output = await _score(
        monkeypatch,
        _Exploding(),
        run_agent=_agent_returning('7421'),
        context_question='Which port?',
        dataset_name='CLB',
        rubrics=['mentions 7421'],
    )

    assert output.score == 0.0
    assert output.judge_unavailable is True


async def test_unrouted_row_infers_the_judge_from_what_it_carries(monkeypatch) -> None:
    judge = _FakeJudge('{"answer_1": 1}')
    answer_only = await _score(
        monkeypatch,
        judge,
        run_agent=_agent_returning('7421'),
        context_question='Which port?',
        dataset_name='SomeCorpusWithNoEvaluator',
        answer='7421',
    )

    assert answer_only.score == 1.0
    # No registered style, so nothing claims one.
    assert answer_only.grading_style is None


async def test_row_with_nothing_to_grade_against_is_flagged(monkeypatch) -> None:
    output = await _score(
        monkeypatch,
        _ForbiddenJudge(),
        run_agent=_agent_returning('7421'),
        context_question='Which port?',
    )

    assert output.score == 0.0
    assert output.judge_unavailable is True
    assert 'nothing to grade against' in output.rationale


async def test_judge_backend_can_force_a_style_for_an_unknown_corpus(monkeypatch) -> None:
    judge = _FakeJudge(
        '{"Grading Rationale": "ok", "List of Requirement Satisfaction Status": ["yes"], "Overall Score": 1}'
    )
    output = await _score(
        monkeypatch,
        judge,
        run_agent=_agent_returning('7421'),
        context_question='Which port?',
        dataset_name='LongBenchV2',  # would route to the rule-based extractor
        judge_backend='rubrics',  # ... but the caller forces rubric grading
        rubrics=['mentions 7421'],
        answer='D',
    )

    assert output.score == 1.0
    assert 'mentions 7421' in judge.prompts[0]


async def test_rubric_accuracy_backend_uses_ai_judge_statuses() -> None:
    from nexus.extensions.tasks.context_agent import eval as eval_module

    judge = _FakeJudge(
        '{"Grading Rationale": "two criteria met", '
        '"List of Requirement Satisfaction Status": ["yes", "no", "yes"], '
        '"Overall Score": 0}'
    )
    result = await eval_module.evaluate(
        'candidate answer',
        {
            'metadata': {'dataset_name': 'GDPpdf'},
            'rubrics': ['criterion one', 'criterion two', 'criterion three'],
        },
        judge,
        backend='rubric_accuracy',
    )

    assert result['score'] == pytest.approx(2 / 3)
    assert result['criteria_passed'] == 2
    assert result['criteria_total'] == 3
    assert result['graded_on'] == 'final_message'


async def test_rm_system_failure_is_maskable_reward_error(monkeypatch) -> None:
    async def failed_rm(*args, **kwargs):
        assert kwargs['backend'] == 'rm_system'
        return {
            'score': 0.0,
            'rationale': 'Judge failed (rm_system): unavailable',
            'reward_error': True,
            'graded_on': 'reward_system',
        }

    monkeypatch.setattr(task_module, 'evaluate', failed_rm)
    output = await _score(
        monkeypatch,
        _ForbiddenJudge(),
        run_agent=_agent_returning('answer'),
        context_question='question',
        dataset_name='surge_docs',
        judge_backend='rm_system',
        reward_system_router_id='T1-Omni-0',
        rubrics=['criterion'],
    )

    assert output.score == 0.0
    assert output.judge_unavailable is True
    assert output.termination_reason == 'REWARD_ERROR'
    assert output.graded_on == 'reward_system'


async def test_rm_system_uses_full_reward_prompt_and_router(monkeypatch) -> None:
    from nexus.extensions.tasks.context_agent.eval import rm_system

    captured = {}

    async def score_answer(prompt, answer, **kwargs):
        captured.update(prompt=prompt, answer=answer, **kwargs)
        return {'ok': True, 'final_score': 0.5, 'request_ids': ['req-1']}

    monkeypatch.setattr(rm_system.rm_scorer, 'score_answer', score_answer)
    result = await rm_system.evaluate(
        'candidate',
        {
            'context_question': 'last question',
            'reward_prompt': [
                {'role': 'system', 'content': 'persona'},
                {'role': 'user', 'content': 'last question'},
            ],
            'rubrics': ['one', 'two'],
            'reward_system_router_id': 'ComplexInstruction-L3Reward-1',
        },
    )

    assert result['score'] == 0.5
    assert captured['prompt'][0] == {'role': 'system', 'content': 'persona'}
    assert captured['router_id'] == 'ComplexInstruction-L3Reward-1'
    assert '<RUBRIC_2>' in captured['extra']['checklist']
    assert captured['extra']['checklist_items'] == ['one', 'two']
    assert captured['extra']['weight'] == [1.0, 1.0]


async def test_empty_answer_scores_zero_without_calling_the_judge(monkeypatch) -> None:
    output = await _score(
        monkeypatch,
        _ForbiddenJudge(),
        run_agent=_agent_returning('   '),
        context_question='Which port?',
        dataset_name='CLB',
        rubrics=['mentions 7421'],
    )

    assert output.score == 0.0
    assert output.rationale == 'No model output'


@pytest.mark.parametrize(
    ('exit_status', 'expected', 'evaluation_skipped'),
    [
        (TerminationReason.MAX_ITERATIONS, 'MAX_STEPS', True),
        (TerminationReason.FORMAT_ERROR, 'FORMAT_ERROR', False),
        (TerminationReason.TRUNCATION, 'TRUNCATION', True),
        (TerminationReason.OUTPUT_LENGTH_EXCEEDED, 'TRUNCATION', True),
    ],
)
async def test_task_propagates_agent_termination_reason(
    monkeypatch: pytest.MonkeyPatch,
    exit_status: TerminationReason,
    expected: str,
    evaluation_skipped: bool,
) -> None:
    output = await _score(
        monkeypatch,
        _ForbiddenJudge(),
        run_agent=_agent_returning('', exit_status=exit_status),
        context_question='Which port?',
        dataset_name='CLB',
        rubrics=['mentions 7421'],
    )

    assert output.score == 0.0
    assert output.termination_reason == expected
    assert output.evaluation_skipped is evaluation_skipped
    assert output.judge_unavailable is False


@pytest.mark.parametrize(
    ('exit_status', 'expected'),
    [
        ('SUCCESS', 'ENV_DONE'),
        ('ENV_DONE', 'ENV_DONE'),
        (None, 'ENV_DONE'),
        ('MAX_ITERATIONS', 'MAX_STEPS'),
        (TerminationReason.MAX_ITERATIONS, 'MAX_STEPS'),
        ('TRUNCATION', 'TRUNCATION'),
        ('OUTPUT_LENGTH_EXCEEDED', 'TRUNCATION'),
        ('FORMAT_ERROR', 'FORMAT_ERROR'),
        ('INITIALIZATION_ERROR', 'INITIALIZATION_ERROR'),
        ('ENV_ERROR', 'ENV_ERROR'),
        ('TASK_TIMEOUT', 'TASK_TIMEOUT'),
        ('PROVIDER_ERROR', 'UNHANDLED_ERROR'),
        ('UNHANDLED_ERROR', 'UNHANDLED_ERROR'),
        ('REWARD_ERROR', 'REWARD_ERROR'),
        ('unknown_exit', 'UNHANDLED_ERROR'),
    ],
)
@pytest.mark.parametrize('dict_output', [False, True])
async def test_general_and_native_task_share_agent_termination_mapping(
    monkeypatch,
    exit_status,
    expected,
    dict_output,
) -> None:
    from nexus.extensions.tasks.context_agent.enterprisebench_native.task import (
        _agent_termination_reason as native_reason,
    )

    agent_output = {'final_answer': 'answer', 'exit_status': exit_status}
    if not dict_output:
        agent_output = SimpleNamespace(**agent_output)

    async def run_agent(**_kwargs):
        return agent_output

    output = await _score(
        monkeypatch,
        _ForbiddenJudge(),
        run_agent=run_agent,
        context_question='question',
        evaluate=False,
    )

    assert output.termination_reason == native_reason(agent_output) == expected
    assert output.evaluation_skipped is True
    assert output.score == 0.0


async def test_dataset_name_is_read_from_nested_metadata_too(monkeypatch) -> None:
    output = await _score(
        monkeypatch,
        _ForbiddenJudge(),
        run_agent=_agent_returning('The correct answer is (A)'),
        context_question='Which option?',
        metadata={'dataset_name': 'LongBenchV2'},
        answer='A',
    )

    assert output.score == 1.0
    assert output.grading_style == 'rule'


def test_score_is_clamped_and_unusable_values_become_none() -> None:
    assert task_module._normalize_score(1) == 1.0
    assert task_module._normalize_score(0.42) == 0.42
    assert task_module._normalize_score(5) == 1.0
    assert task_module._normalize_score(-2) == 0.0
    assert task_module._normalize_score(None) is None
    assert task_module._normalize_score('nope') is None
    assert task_module._normalize_score(float('nan')) is None


def test_deliverable_reward_is_rubric_accuracy() -> None:
    from nexus.extensions.tasks.context_agent.eval import deliverable_judge

    assert deliverable_judge._rubric_accuracy(0, 4) == 0.0
    assert deliverable_judge._rubric_accuracy(3, 4) == 0.75
    assert deliverable_judge._rubric_accuracy(4, 4) == 1.0


# ---------------------------------------------------------------------------
# The judge must never be the policy
# ---------------------------------------------------------------------------


async def test_routed_judge_prefers_a_named_judge_client_over_main() -> None:
    policy = _ScriptedClient()
    judge_client = _ScriptedClient()

    async with ExecutionContext(openai_clients={'main': policy, 'judge': judge_client}) as ctx:
        judge = resolve_judge_llm(ctx)

    # Under RL 'main' is the model being trained; grading with it would score
    # the policy against itself and feed that back into its own advantage.
    assert judge.resolve()._client is judge_client
    assert judge.resolve()._client is not policy


async def test_routed_judge_never_falls_back_to_the_policy_client() -> None:
    policy = _ScriptedClient()

    async with ExecutionContext(openai_clients={'main': policy}) as ctx:
        judge = resolve_judge_llm(ctx)

    assert judge.resolve()._client is not policy


async def test_rule_based_grading_never_builds_a_judge_client() -> None:
    policy = _ScriptedClient()

    async with ExecutionContext(
        openai_clients={'main': policy},
        runtime_provider=LocalRuntimeProvider(),
    ) as ctx:
        judge = resolve_judge_llm(ctx)
        output = await run_context_agent_general_task(
            ContextAgentGeneralTaskInput(
                run_agent=_agent_returning('The correct answer is (B)'),
                context_question='Which option?',
                dataset_name='LongBenchV2',
                answer='B',
            )
        )

    assert output.score == 1.0
    # Construction is deferred, so a run of only rule-based benchmarks needs no
    # judge endpoint and no credential at all.
    assert judge.resolved is False


def test_pinned_judges_ignore_the_caller_and_share_one_client(monkeypatch) -> None:
    from nexus.extensions.tasks.context_agent.eval import _judge_client

    monkeypatch.setattr(_judge_client, '_clients', {})
    first = _judge_client.pinned_client('AALCR', 'api_ali_qwen3-235b-a22b')
    second = _judge_client.pinned_client('LongBenchPro', 'api_ali_qwen3-235b-a22b')

    # AALCR and LongBenchPro pin the same model, and their published numbers
    # were produced with it. One client means one concurrency semaphore.
    assert first is second


def test_a_pinned_judge_model_can_be_overridden_by_env(monkeypatch) -> None:
    from nexus.extensions.tasks.context_agent.eval import _judge_client

    monkeypatch.setattr(_judge_client, '_clients', {})
    monkeypatch.setenv('CTXAGENT_JUDGE_MODEL_AALCR', 'some-other-model')
    client = _judge_client.pinned_client('AALCR', 'api_ali_qwen3-235b-a22b')

    assert client._model == 'some-other-model'


# ---------------------------------------------------------------------------
# Frozen config / runtime contracts
# ---------------------------------------------------------------------------


def test_gdppdf500_single_case_parses_without_dataset(monkeypatch) -> None:
    from nexus.runner.compose import compile_single_file

    monkeypatch.setenv('CTXAGENT_GONGFENG_TOKEN', 'gongfeng-secret')
    monkeypatch.setenv('CTXAGENT_POLICY_API_KEY', 'policy-secret')
    monkeypatch.setenv('CTXAGENT_JUDGE_API_KEY', 'judge-secret')
    path = (
        Path(__file__).resolve().parents[2]
        / 'nexus/extensions/tasks/context_agent/configs/examples/gdppdf500_single.yaml'
    )
    data = compile_single_file(path)

    assert data.agent_name == 'context_agent_general'
    assert data.task_name == 'context_agent_general_task'
    cfg = data.task_config
    assert cfg['dataset_name'] == 'GDPpdf'
    assert STYLES['GDPpdf'] == 'rubric'
    assert EVALUATORS['GDPpdf'] is not None
    assert cfg['context_files'][0]['uri'].startswith('https://')
    assert data.runtime_provider_name == 'gongfeng'
    assert data.runtime_configs['main']['worker_venv'] == {'modules': []}
    assert data.task_files is None

    ContextAgentGeneralTaskInput(
        run_agent=_agent_returning('x'),
        context_question=cfg['context_question'],
        context_files=cfg['context_files'],
        dataset_name=cfg['dataset_name'],
        metadata=cfg['metadata'],
        rubrics=cfg['rubrics'],
        language=cfg['language'],
        max_tool_rounds=cfg['max_tool_rounds'],
        curl_timeout=cfg['curl_timeout'],
    )


async def test_task_requires_a_runtime_provider() -> None:
    async with ExecutionContext(openai_clients={}):
        with pytest.raises(RuntimeError, match='requires a runtime provider'):
            await run_context_agent_general_task(
                ContextAgentGeneralTaskInput(
                    run_agent=_agent_returning('unused'),
                    context_question='Which port?',
                )
            )


async def test_runtime_rollout_uses_explicit_working_dir_without_host_workspace(tmp_path, monkeypatch) -> None:
    from nexus.extensions.agents.context_agent import general_agent as agent_module
    from nexus.runtime.providers.base import RuntimeProvider

    class _Provider(RuntimeProvider):
        async def acquire(self, name: str, **params):
            raise AssertionError('the fake staging path must not acquire a runtime')

        async def _release_impl(self, name: str) -> None:
            pass

    class _Executor:
        tool_name_resolution_error_count = 0

        def __init__(self, *, enabled_tool_names, allowed_mcp_tool_names, tool_search_config):
            assert enabled_tool_names
            assert allowed_mcp_tool_names == []
            assert tool_search_config is None

        def get_tool_schemas(self):
            return build_tool_schemas()

        def build_tool_search_reminder(self):
            return None

        async def execute(self, name, arguments):
            raise AssertionError('the scripted model does not call tools')

    expected_working_dir = '/tmp/ctxagent_general/runtime-only'

    async def fake_stage(_ctx, declared_files, _knobs, *, working_dir):
        assert working_dir == expected_working_dir
        return declared_files, [], working_dir

    async def fake_cleanup(_ctx):
        return None

    monkeypatch.setattr(agent_module, 'stage_local_files', fake_stage)
    monkeypatch.setattr(agent_module, 'cleanup_runtime_session', fake_cleanup)
    monkeypatch.setattr(agent_module, 'ContextGeneralToolExecutor', _Executor)

    async with ExecutionContext(
        openai_clients={'main': _ScriptedClient(_Message(content='done'))},
        runtime_provider=_Provider(),
    ):
        output = await run_context_agent_general(
            ContextAgentGeneralInput(
                user_prompt='Which port?',
                working_dir=expected_working_dir,
                context_files=[{'name': 'handbook.pdf', 'uri': 'https://example.com/handbook.pdf'}],
            )
        )

    assert output.final_answer == 'done'


async def test_task_allocates_and_passes_runtime_working_dir(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_run_agent(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(final_answer='The correct answer is (B)')

    async with ExecutionContext(
        openai_clients={},
        runtime_provider=LocalRuntimeProvider(),
        run_id='run-123',
    ):
        output = await run_context_agent_general_task(
            ContextAgentGeneralTaskInput(
                run_agent=fake_run_agent,
                context_question='Which option?',
                dataset_name='LongBenchV2',
                answer='B',
            )
        )

    assert captured['working_dir'] == '/tmp/ctxagent_general/run-123'
    assert output.score == 1.0


async def test_workspace_artifact_persists_only_the_runtime_workspace(tmp_path) -> None:
    runtime_root = tmp_path / 'runtime'
    workspace = runtime_root / 'workspace'
    workspace.mkdir(parents=True)
    (workspace / 'input.txt').write_text('provided', encoding='utf-8')
    (workspace / 'report.xlsx').write_bytes(b'generated workbook')
    (runtime_root / 'outside.txt').write_text('must not be persisted', encoding='utf-8')

    async with ExecutionContext(
        openai_clients={},
        runtime_provider=LocalRuntimeProvider(),
        run_id='run:123',
    ) as ctx:
        artifact = await persist_runtime_workspace(
            ctx,
            working_dir=str(runtime_root),
            artifact_root=tmp_path / 'artifacts',
            context_files=[{'name': 'input.txt', 'uri': 'https://example.com/input.txt'}],
            staged_files=[],
            task_id='task-1',
        )

    host_workspace = Path(artifact['path'])
    assert artifact['available'] is True
    assert host_workspace == tmp_path / 'artifacts' / 'run_123' / 'workspace'
    assert (host_workspace / 'input.txt').read_text(encoding='utf-8') == 'provided'
    assert (host_workspace / 'report.xlsx').read_bytes() == b'generated workbook'
    assert not (host_workspace.parent / 'outside.txt').exists()
    assert artifact['files'] == [
        {'path': 'input.txt', 'size_bytes': 8, 'kind': 'context'},
        {'path': 'report.xlsx', 'size_bytes': 18, 'kind': 'agent'},
    ]

    manifest = json.loads((host_workspace.parent / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['run_id'] == 'run:123'
    assert manifest['task_id'] == 'task-1'
    assert manifest['files'] == artifact['files']


async def test_task_persists_workspace_when_evaluation_is_disabled(tmp_path, monkeypatch) -> None:
    async def fake_run_agent(**kwargs):
        workspace = Path(kwargs['working_dir']) / 'workspace'
        workspace.mkdir(parents=True)
        (workspace / 'input.pdf').write_bytes(b'provided')
        (workspace / 'deliverable.pptx').write_bytes(b'generated')
        return SimpleNamespace(
            final_answer='Created the requested presentation.',
            staged_files=[{'name': 'input.pdf', 'source': '/host/input.pdf'}],
        )

    def forbidden_resolve(*_args, **_kwargs):
        raise AssertionError('evaluate=false must not resolve a judge')

    async def forbidden_evaluate(*_args, **_kwargs):
        raise AssertionError('evaluate=false must not call an evaluator')

    monkeypatch.setattr(task_module, 'resolve_judge_llm', forbidden_resolve)
    monkeypatch.setattr(task_module, 'evaluate', forbidden_evaluate)

    working_dir = tmp_path / 'runtime'
    artifact_root = tmp_path / 'artifacts'
    async with ExecutionContext(
        openai_clients={},
        runtime_provider=LocalRuntimeProvider(),
        run_id='gdpval-run',
    ):
        output = await run_context_agent_general_task(
            ContextAgentGeneralTaskInput(
                run_agent=fake_run_agent,
                context_question='Create a presentation.',
                context_files=[{'name': 'input.pdf', 'uri': 'https://example.com/input.pdf'}],
                dataset_name='GDPval',
                metadata={'task_id': 'dataset-task'},
                rubrics=['produces a PowerPoint'],
                working_dir=str(working_dir),
                artifact_root=str(artifact_root),
                evaluate=False,
            )
        )

    assert output.score == 0.0
    assert output.evaluation_skipped is True
    assert output.rationale == 'Evaluation disabled'
    assert output.workspace_artifact is not None
    assert output.workspace_artifact['run_id'] == 'gdpval-run'
    assert output.workspace_artifact['task_id'] == 'dataset-task'
    assert output.workspace_artifact['files'] == [
        {'path': 'deliverable.pptx', 'size_bytes': 9, 'kind': 'agent'},
        {'path': 'input.pdf', 'size_bytes': 8, 'kind': 'context'},
    ]


def test_runtime_session_rebuilds_general_state(tmp_path) -> None:
    from nexus.core import ToolContext
    from nexus.extensions.agents.context_agent.tools.general.sessions import cleanup_session, get_session
    from nexus.extensions.agents.context_agent.tools.general.workspace import Workspace

    docs = tmp_path / Workspace.DOCS_SUBDIR
    docs.mkdir()
    (docs / 'notes.md').write_text('hello from the sandbox\n', encoding='utf-8')
    ctx = ToolContext(
        run_id='session-test',
        tool_state={
            'working_dir': str(tmp_path),
            'workspace_root': str(tmp_path),
            'cwd': str(tmp_path),
            'docs_dir': str(docs),
            'declared_files': [
                {
                    'name': 'notes.md',
                    'uri': './workspace/notes.md',
                    'local_path': str(docs / 'notes.md'),
                }
            ],
            'knobs': {'file_search_top_k': 3, 'python_timeout': 30},
        },
    )
    try:
        state = get_session(ctx)
        assert state.workspace.cwd == str(tmp_path)
        assert state.knobs.file_search_top_k == 3
        assert state.resolve('./workspace/notes.md').read_text(encoding='utf-8').startswith('hello')
        assert get_session(ctx) is state
    finally:
        assert cleanup_session(ctx) is True
        assert cleanup_session(ctx) is False


async def test_max_tool_rounds_env_overrides_agent_config(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('CONTEXTAGENT_GENERAL_MAX_TOOL_ROUNDS', '1')
    client = _ScriptedClient(
        _Message(content='x', tool_calls=[_tool_call('c1', 'bash', '{"command": "echo hi"}')]),
        _Message(content='forced'),
    )
    async with ExecutionContext(
        openai_clients={'main': client},
        runtime_provider=LocalRuntimeProvider(),
    ):
        output = await run_context_agent_general(
            ContextAgentGeneralInput(
                user_prompt='hi',
                working_dir=str(tmp_path / 'runtime'),
                max_iterations=20,
                max_tool_rounds=20,
            )
        )
    assert output.hit_max_turn_limit is True
    assert output.tool_rounds_used == 1
    assert output.forced_finalization is True
    assert output.final_answer == 'forced'


@pytest.mark.parametrize(
    ('constraint_binary', 'tool_penalty', 'expected'),
    [(1, False, 1.0), (0, False, 0.5), (1, True, 0.0), (0, True, 0.0)],
)
async def test_agentif_constraints_compose_with_optional_tool_name_penalty(
    monkeypatch, constraint_binary, tool_penalty, expected
):
    async def evaluate(*args, **kwargs):
        return {'score': 1.0, 'rationale': 'answer passed'}

    async def constraints(**kwargs):
        assert kwargs['trajectory'] == [{'role': 'assistant', 'content': 'answer'}]
        return {'ok': True, 'binary': constraint_binary, 'verdicts': []}

    async def run_agent(**kwargs):
        return SimpleNamespace(
            final_answer='answer',
            exit_status='ENV_DONE',
            main_trajectory=[{'role': 'assistant', 'content': 'answer'}],
            tool_name_resolution_error_count=2,
        )

    monkeypatch.setattr(task_module, 'evaluate', evaluate)
    monkeypatch.setattr(task_module, 'score_constraints', constraints)
    output = await _score(
        monkeypatch,
        _ForbiddenJudge(),
        run_agent=run_agent,
        context_question='question',
        dataset_name='LongBenchV2',
        constraint_rubrics=[{'constraint': 'use the required format'}],
        zero_score_on_tool_name_resolution_error=tool_penalty,
    )
    assert output.score == expected
    assert output.answer_score == 1.0
    assert output.constraint_score == constraint_binary
    metrics = output.extra_fields['reward_extra_info']
    assert metrics['tool_name_resolution_error_count'] == 2
    assert metrics['tool_name_resolution_zeroed'] == float(tool_penalty)
    assert metrics['extra_reward_metrics'] == {'answer_score': 1.0, 'constraint_score': float(constraint_binary)}
    assert output.extra_fields['agentif_judge']['answer']['score'] == 1.0
