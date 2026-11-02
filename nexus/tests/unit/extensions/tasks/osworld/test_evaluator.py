"""OSWorld evaluator regressions."""

from __future__ import annotations

import base64
import builtins
import contextlib
import datetime as dt
import importlib
import json
import os
import sys

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import nexus.extensions.tasks.os_world.metrics as metrics_module

from nexus.extensions.tasks.os_world.evaluation.evaluator import evaluate_os_world
from nexus.extensions.tasks.os_world.getters import GETTER_REGISTRY
from nexus.extensions.tasks.os_world.getters.runtime import run_bash
from nexus.extensions.tasks.os_world.metrics import METRIC_REGISTRY
from nexus.extensions.tasks.os_world.metrics import simple as simple_metrics
from nexus.extensions.tasks.os_world.metrics.official.chrome import is_expected_bookmarks


@dataclass
class _CommandResponse:
    stdout: str = ''
    stderr: str = ''
    return_code: int = 0


class _FakeRuntime:
    def __init__(self, outputs: dict[str, _CommandResponse | str] | None = None) -> None:
        self.outputs = outputs or {}
        self.commands: list[str] = []

    async def run_command(self, command: str, **kwargs: Any) -> _CommandResponse:
        self.commands.append(command)
        for needle, response in self.outputs.items():
            if needle in command:
                if isinstance(response, _CommandResponse):
                    return response
                return _CommandResponse(stdout=response)
        return _CommandResponse()


def _fail_official_metric_imports(monkeypatch: pytest.MonkeyPatch, *module_names: str) -> None:
    failing_modules = {
        f'nexus.extensions.tasks.os_world.metrics.official.{module_name}' for module_name in module_names
    }
    original_import_module = metrics_module.importlib.import_module

    def import_module(name: str, package: str | None = None) -> Any:
        if name in failing_modules:
            raise ImportError('simulated missing optional dependency')
        return original_import_module(name, package)

    monkeypatch.setattr(metrics_module.importlib, 'import_module', import_module)


def _fail_cv2_import(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def import_module(
        name: str,
        global_vars: dict[str, Any] | None = None,
        local_vars: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == 'cv2':
            raise ImportError('libGL.so.1: cannot open shared object file')
        return original_import(name, global_vars, local_vars, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', import_module)
    sys.modules.pop('nexus.extensions.tasks.os_world.metrics.official.vlc', None)
    sys.modules.pop('nexus.extensions.tasks.os_world.metrics.official.gimp', None)
    sys.modules.pop('nexus.extensions.tasks.os_world.metrics.official.docs', None)
    sys.modules.pop('easyocr', None)


def _write_docx(path: Path, paragraphs: list[str]) -> None:
    from docx import Document

    doc = Document()
    for paragraph in paragraphs:
        doc.add_paragraph(paragraph)
    doc.save(path)


def _official_registry_gaps(root: Path) -> tuple[int, set[str], set[str]]:
    missing_metrics: set[str] = set()
    missing_getters: set[str] = set()
    task_count = 0

    for path in root.rglob('*.json'):
        task_count += 1
        evaluator = json.loads(path.read_text(encoding='utf-8')).get('evaluator') or {}
        funcs = evaluator.get('func')
        for func in funcs if isinstance(funcs, list) else [funcs]:
            if func and func != 'infeasible' and func not in METRIC_REGISTRY:
                missing_metrics.add(str(func))
        for section in ('result', 'expected'):
            value = evaluator.get(section)
            for item in value if isinstance(value, list) else [value]:
                if isinstance(item, dict):
                    getter_type = item.get('type')
                    if getter_type and getter_type not in GETTER_REGISTRY:
                        missing_getters.add(str(getter_type))

    return task_count, missing_metrics, missing_getters


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('stderr', 'return_code', 'expected_score'),
    [
        ("ModuleNotFoundError: No module named 'torch'\n", 1, 0.0),
        ('', 0, 1.0),
        ('UserWarning: optional acceleration is unavailable\n', 0, 1.0),
    ],
)
async def test_vm_import_error_is_task_state_not_a_host_dependency_failure(
    stderr: str, return_code: int, expected_score: float
) -> None:
    # Environment-configuration tasks intentionally grade the VM import result.
    # Do not preinstall packages, change Python environments, or retry the agent
    # merely because this command returned an ImportError on stderr.
    runtime = _FakeRuntime(
        {'from InstructorEmbedding import INSTRUCTOR': _CommandResponse(stderr=stderr, return_code=return_code)}
    )
    result = await evaluate_os_world(
        runtime,
        {
            'func': 'check_include_exclude',
            'result': {
                'type': 'vm_command_error',
                'command': ['python', '-c', 'from InstructorEmbedding import INSTRUCTOR;'],
            },
            'expected': {'type': 'rule', 'rules': {'include': [], 'exclude': ['Error:']}},
        },
        unsupported_policy='fail',
        last_action='DONE',
    )

    assert result['supported'] is True
    assert result['score'] == expected_score
    assert result['actual'] == stderr
    assert len(runtime.commands) == 1
    assert 'pip install' not in runtime.commands[0]


@pytest.mark.asyncio
async def test_evaluate_exact_match_with_vm_command_line() -> None:
    runtime = _FakeRuntime({'cat /tmp/nexus_osworld_smoke_result.txt': 'true'})

    result = await evaluate_os_world(
        runtime,
        {
            'func': 'exact_match',
            'result': {
                'type': 'vm_command_line',
                'command': 'cat /tmp/nexus_osworld_smoke_result.txt',
                'shell': True,
            },
            'expected': {'type': 'rule', 'rules': {'expected': 'true'}},
        },
        unsupported_policy='fail',
        last_action='DONE',
    )

    assert result['supported'] is True
    assert result['score'] == 1.0
    assert result['actual'] == 'true'
    assert result['func'] == 'exact_match'


@pytest.mark.asyncio
async def test_postconfig_stdout_feeds_cache_file_metric() -> None:
    runtime = _FakeRuntime({'ls -R /home/user/Documents': 'report.txt\n'})

    result = await evaluate_os_world(
        runtime,
        {
            'postconfig': [
                {
                    'type': 'execute',
                    'parameters': {
                        'command': ['ls', '-R', '/home/user/Documents'],
                        'stdout': 'documents.ls',
                    },
                }
            ],
            'func': 'check_list',
            'result': {'type': 'cache_file', 'path': 'documents.ls'},
            'expected': {'type': 'rule', 'rules': {'expect': [r'\breport\.txt\b']}},
        },
        unsupported_policy='fail',
        last_action='DONE',
    )

    assert result['supported'] is True
    assert result['score'] == 1.0
    assert Path(result['actual']).name == 'documents.ls'


@pytest.mark.asyncio
async def test_postconfig_download_uses_host_cache_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    import nexus.extensions.tasks.os_world.execution.setup as osworld_setup

    fetches: list[str] = []
    uploads = []

    async def fake_download_url_bytes(url: str) -> bytes:
        fetches.append(url)
        return b'postconfig asset'

    class UploadRuntime(_FakeRuntime):
        async def upload_file(self, files, overwrite=True):
            uploads.append((files, overwrite))
            return {'results': [{'path': file.path, 'success': True, 'message': 'ok'} for file in files]}

    monkeypatch.setattr(osworld_setup, '_download_url_bytes', fake_download_url_bytes)
    runtime = UploadRuntime({'printf true': 'true'})

    result = await evaluate_os_world(
        runtime,
        {
            'postconfig': [
                {
                    'type': 'download',
                    'parameters': {'files': [{'url': 'https://example.test/post.txt', 'path': '/tmp/post.txt'}]},
                }
            ],
            'func': 'exact_match',
            'result': {'type': 'vm_command_line', 'command': 'printf true', 'shell': True},
            'expected': {'type': 'rule', 'rules': {'expected': 'true'}},
        },
        unsupported_policy='fail',
        last_action='DONE',
    )

    assert result['score'] == 1.0
    assert fetches == ['https://example.test/post.txt']
    assert len(uploads) == 1
    assert uploads[0][0][0].path == '/tmp/post.txt'
    assert not any('curl -L' in command for command in runtime.commands)


@pytest.mark.asyncio
async def test_runtime_command_helper_can_skip_user_wrapper() -> None:
    runtime = _FakeRuntime()
    runtime.osworld_command_as_user = False

    await run_bash(runtime, 'printf ok')

    assert runtime.commands == ['printf ok']


@pytest.mark.asyncio
async def test_postconfig_close_window_is_supported() -> None:
    runtime = _FakeRuntime({'cat /tmp/nexus_osworld_close_window_result.txt': 'true'})

    result = await evaluate_os_world(
        runtime,
        {
            'postconfig': [
                {
                    'type': 'close_window',
                    'parameters': {
                        'window_name': 'Thunderbird',
                        'strict': True,
                        'by_class': True,
                    },
                }
            ],
            'func': 'exact_match',
            'result': {
                'type': 'vm_command_line',
                'command': 'cat /tmp/nexus_osworld_close_window_result.txt',
                'shell': True,
            },
            'expected': {'type': 'rule', 'rules': {'expected': 'true'}},
        },
        unsupported_policy='fail',
    )

    assert result['score'] == 1.0
    assert any('wmctrl -xFc Thunderbird' in command for command in runtime.commands)


@pytest.mark.asyncio
async def test_postconfig_chrome_inject_js_executes_script(monkeypatch: pytest.MonkeyPatch) -> None:
    from nexus.extensions.tasks.os_world.evaluation import evaluator as osworld_evaluator

    class FakePage:
        def __init__(self) -> None:
            self.waits: list[tuple[str, int]] = []
            self.scripts: list[str] = []

        async def wait_for_load_state(self, state: str, *, timeout: int) -> None:
            self.waits.append((state, timeout))

        async def evaluate(self, script: str) -> None:
            self.scripts.append(script)

    class FakeBrowser:
        def __init__(self, page: FakePage) -> None:
            self.contexts = [type('Context', (), {'pages': [page]})()]
            self.closed = False

    page = FakePage()
    browser = FakeBrowser(page)
    script = 'if (document.activeElement && document.activeElement.blur) { document.activeElement.blur(); }'

    @contextlib.asynccontextmanager
    async def fake_connect_chromium(runtime: Any):
        try:
            yield browser
        finally:
            browser.closed = True

    monkeypatch.setattr(osworld_evaluator, 'connect_chromium', fake_connect_chromium)
    monkeypatch.setattr(osworld_evaluator, '_CHROME_INJECT_JS_RETRY_ATTEMPTS', 1, raising=False)

    result = await evaluate_os_world(
        _FakeRuntime({'printf true': 'true'}),
        {
            'postconfig': [
                {
                    'type': 'chrome_inject_js',
                    'parameters': {
                        'js': script,
                        'wait_for_load': False,
                    },
                }
            ],
            'func': 'exact_match',
            'result': {'type': 'vm_command_line', 'command': 'printf true', 'shell': True},
            'expected': {'type': 'rule', 'rules': {'expected': 'true'}},
        },
        unsupported_policy='fail',
        last_action='DONE',
    )

    assert result['supported'] is True
    assert result['score'] == 1.0
    assert page.scripts == [script]
    assert page.waits == []
    assert browser.closed is True


@pytest.mark.asyncio
async def test_postconfig_chrome_inject_js_requires_script() -> None:
    result = await evaluate_os_world(
        _FakeRuntime(),
        {
            'postconfig': [{'type': 'chrome_inject_js', 'parameters': {'wait_for_load': False}}],
            'func': 'exact_match',
            'result': {'type': 'vm_command_line', 'command': 'printf true', 'shell': True},
            'expected': {'type': 'rule', 'rules': {'expected': 'true'}},
        },
        unsupported_policy='skip',
    )

    assert result['supported'] is False
    assert result['score'] == 0.0
    assert 'Missing JavaScript for OSWorld postconfig chrome_inject_js action' in result['error']


@pytest.mark.asyncio
async def test_postconfig_launch_adds_proxy_only_to_chrome_launch() -> None:
    runtime = _FakeRuntime({'printf true': 'true'})
    runtime.osworld_use_proxy = True

    result = await evaluate_os_world(
        runtime,
        {
            'postconfig': [
                {'type': 'launch', 'parameters': {'command': ['google-chrome', '--remote-debugging-port=1337']}},
                {'type': 'launch', 'parameters': {'command': ['xterm', '-title', 'Demo']}},
            ],
            'func': 'exact_match',
            'result': {'type': 'vm_command_line', 'command': 'printf true', 'shell': True},
            'expected': {'type': 'rule', 'rules': {'expected': 'true'}},
        },
        unsupported_policy='fail',
    )

    assert result['score'] == 1.0
    assert '--proxy-server=http://127.0.0.1:18888' in runtime.commands[0]
    assert '--proxy-server=http://127.0.0.1:18888' not in runtime.commands[1]


@pytest.mark.asyncio
async def test_chrome_getter_fallback_launch_honors_task_proxy_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    from nexus.extensions.tasks.os_world.getters import chrome as chrome_getters

    commands: list[str] = []

    class FakeRuntime:
        osworld_use_proxy = True

        async def run_command(self, command, timeout=60, cwd=None):
            commands.append(command)
            return _CommandResponse()

    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(chrome_getters.asyncio, 'sleep', fake_sleep)

    await chrome_getters._launch_chrome_for_cdp(FakeRuntime())

    assert len(commands) == 1
    assert 'google-chrome --remote-debugging-port=1337' in commands[0]
    assert 'chromium --remote-debugging-port=1337' in commands[0]
    assert commands[0].count('--proxy-server=http://127.0.0.1:18888') == 4  # Two browsers in both user branches.
    assert '--remote-allow-origins=http://127.0.0.1:9222' in commands[0]


@pytest.mark.asyncio
async def test_multi_metric_and_or_semantics() -> None:
    runtime = _FakeRuntime({'cat /tmp/first': 'yes', 'cat /tmp/second': 'no'})
    base = {
        'func': ['exact_match', 'exact_match'],
        'result': [
            {'type': 'vm_command_line', 'command': 'cat /tmp/first', 'shell': True},
            {'type': 'vm_command_line', 'command': 'cat /tmp/second', 'shell': True},
        ],
        'expected': [
            {'type': 'rule', 'rules': {'expected': 'yes'}},
            {'type': 'rule', 'rules': {'expected': 'yes'}},
        ],
    }

    and_result = await evaluate_os_world(runtime, {**base, 'conj': 'and'}, unsupported_policy='fail')
    or_result = await evaluate_os_world(runtime, {**base, 'conj': 'or'}, unsupported_policy='fail')

    assert and_result['score'] == 0.0
    assert and_result['supported'] is True
    assert or_result['score'] == 1.0
    assert or_result['supported'] is True


@pytest.mark.asyncio
async def test_infeasible_uses_last_action_without_getters() -> None:
    fail_result = await evaluate_os_world(_FakeRuntime(), {'func': 'infeasible'}, last_action={'action_type': 'FAIL'})
    done_result = await evaluate_os_world(_FakeRuntime(), {'func': 'infeasible'}, last_action='DONE')

    assert fail_result['score'] == 1.0
    assert fail_result['supported'] is True
    assert done_result['score'] == 0.0
    assert done_result['supported'] is True


@pytest.mark.asyncio
async def test_unsupported_policy_skip_and_fail() -> None:
    evaluator = {
        'func': 'missing_metric',
        'result': {'type': 'vm_command_line', 'command': 'true', 'shell': True},
        'expected': {'type': 'rule', 'rules': {'expected': ''}},
    }

    skipped = await evaluate_os_world(_FakeRuntime(), evaluator, unsupported_policy='skip')
    assert skipped['supported'] is False
    assert skipped['score'] == 0.0
    assert 'Unknown OSWorld metric function' in skipped['error']

    with pytest.raises(ValueError, match='Unknown OSWorld metric function'):
        await evaluate_os_world(_FakeRuntime(), evaluator, unsupported_policy='fail')


def _official_examples_root() -> Path:
    configured = os.environ.get('OSWORLD_OFFICIAL_EXAMPLES_ROOT')
    candidates = [Path(configured)] if configured else []
    for root in candidates:
        if root.is_dir():
            return root
    pytest.skip('Official 369-task external fixture unavailable; set OSWORLD_OFFICIAL_EXAMPLES_ROOT')


def test_official_osworld_task_evaluator_names_are_registered() -> None:
    root = _official_examples_root()
    task_count, missing_metrics, missing_getters = _official_registry_gaps(root)

    assert task_count == 369
    assert missing_metrics == set()
    assert missing_getters == set()


def test_latest_osworld_task_evaluator_names_are_registered_when_available() -> None:
    root = Path('/tmp/osworld-latest-inspect/evaluation_examples/examples')
    if not root.exists():
        pytest.skip('/tmp/osworld-latest-inspect is not available')

    task_count, missing_metrics, missing_getters = _official_registry_gaps(root)

    assert task_count == 369
    assert missing_metrics == set()
    assert missing_getters == set()


def test_osworld_b7db4d8_getter_names_are_registered() -> None:
    assert {'active_tab_url_and_body_text', 'local_file'} <= set(GETTER_REGISTRY)


def test_osworld_b7db4d8_metric_names_are_registered() -> None:
    expected = {
        'check_continuation_line_indent_no_bullet',
        'compare_docx_paper_records',
        'compare_epub_semantic',
        'compare_pdf_form_fields',
        'compare_pptx_files_robust',
        'compare_pptx_files_tolerant',
        'compare_references_gain',
        'compare_table_records',
    }

    assert expected <= set(METRIC_REGISTRY)


@pytest.mark.parametrize(
    ('metric_name', 'module_name'),
    [
        ('compare_pdf_form_fields', 'chrome'),
        ('compare_references_gain', 'docs'),
        ('compare_docx_paper_records', 'docs'),
        ('compare_epub_semantic', 'others'),
        ('compare_pptx_files_tolerant', 'slides'),
        ('compare_pptx_files_robust', 'slides'),
        ('check_continuation_line_indent_no_bullet', 'slides'),
        ('compare_table_records', 'table'),
    ],
)
def test_osworld_b7db4d8_metrics_lazy_import_official_attr(
    monkeypatch: pytest.MonkeyPatch,
    metric_name: str,
    module_name: str,
) -> None:
    imported: list[str] = []

    def fake_import_module(name: str, package: str | None = None) -> Any:
        imported.append(name)

        def fake_metric(*args: Any, **kwargs: Any) -> float:
            return 0.75

        return type('FakeModule', (), {metric_name: staticmethod(fake_metric)})()

    monkeypatch.setattr(metrics_module.importlib, 'import_module', fake_import_module)

    metric = METRIC_REGISTRY.get(metric_name)
    assert metric is not None
    assert metric.__nexus_official_module__ == module_name
    assert metric.__nexus_official_attr__ == metric_name
    assert metric('result', 'expected', option=True) == 0.75
    assert imported == [f'nexus.extensions.tasks.os_world.metrics.official.{module_name}']


@pytest.mark.parametrize(
    ('metric_name', 'module_name'),
    [
        ('compare_pdf_form_fields', 'chrome'),
        ('compare_references_gain', 'docs'),
        ('compare_docx_paper_records', 'docs'),
        ('compare_epub_semantic', 'others'),
        ('compare_pptx_files_tolerant', 'slides'),
        ('compare_pptx_files_robust', 'slides'),
        ('check_continuation_line_indent_no_bullet', 'slides'),
        ('compare_table_records', 'table'),
    ],
)
def test_osworld_b7db4d8_metric_official_attrs_exist(metric_name: str, module_name: str) -> None:
    module = importlib.import_module(f'nexus.extensions.tasks.os_world.metrics.official.{module_name}')

    assert hasattr(module, metric_name)


@pytest.mark.asyncio
async def test_local_file_getter_copies_file_to_cache(tmp_path: Path) -> None:
    from nexus.extensions.tasks.os_world.getters import file as file_getters
    from nexus.extensions.tasks.os_world.getters.runtime import CACHE_DIR_CONFIG_KEY

    source = tmp_path / 'gold.txt'
    source.write_text('expected text', encoding='utf-8')
    cache_dir = tmp_path / 'cache'

    result = await file_getters.get_local_file(
        _FakeRuntime(),
        {
            'path': str(source),
            'dest': 'copied.txt',
            CACHE_DIR_CONFIG_KEY: str(cache_dir),
        },
    )

    assert result == str(cache_dir / 'copied.txt')
    assert Path(result).read_text(encoding='utf-8') == 'expected text'


@pytest.mark.asyncio
async def test_local_file_getter_copies_directory_to_cache(tmp_path: Path) -> None:
    from nexus.extensions.tasks.os_world.getters import file as file_getters
    from nexus.extensions.tasks.os_world.getters.runtime import CACHE_DIR_CONFIG_KEY

    source = tmp_path / 'fixture'
    (source / 'nested').mkdir(parents=True)
    (source / 'nested' / 'gold.txt').write_text('expected text', encoding='utf-8')
    cache_dir = tmp_path / 'cache'

    result = await file_getters.get_local_file(
        _FakeRuntime(),
        {
            'path': str(source),
            'dest': 'copied-fixture',
            CACHE_DIR_CONFIG_KEY: str(cache_dir),
        },
    )

    assert result == str(cache_dir / 'copied-fixture')
    assert (Path(result) / 'nested' / 'gold.txt').read_text(encoding='utf-8') == 'expected text'


@pytest.mark.asyncio
async def test_local_file_getter_resolves_official_relative_path(
    tmp_path: Path,
) -> None:
    from nexus.extensions.tasks.os_world.getters import file as file_getters
    from nexus.extensions.tasks.os_world.getters.runtime import CACHE_DIR_CONFIG_KEY

    osworld_root = tmp_path / 'OSWorld'
    source = (
        osworld_root
        / 'evaluation_examples'
        / 'examples'
        / 'multi_apps'
        / '7f35355e-02a6-45b5-b140-f0be698bcf85_result_gold.txt'
    )
    source.parent.mkdir(parents=True)
    source.write_text('expected text', encoding='utf-8')
    cache_dir = tmp_path / 'cache'
    runtime = _FakeRuntime()
    runtime.osworld_examples_root = str(osworld_root / 'evaluation_examples' / 'examples')

    result = await file_getters.get_local_file(
        runtime,
        {
            'path': 'evaluation_examples/examples/multi_apps/7f35355e-02a6-45b5-b140-f0be698bcf85_result_gold.txt',
            'dest': 'result_gold.txt',
            CACHE_DIR_CONFIG_KEY: str(cache_dir),
        },
    )

    assert result == str(cache_dir / 'result_gold.txt')
    assert Path(result).read_text(encoding='utf-8') == 'expected text'


@pytest.mark.asyncio
async def test_active_tab_url_and_body_text_reads_matching_tab_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nexus.extensions.tasks.os_world.getters import chrome as chrome_getters

    scripts: list[str] = []

    class FakePage:
        url = 'https://example.test/page'

        async def evaluate(self, script: str) -> str:
            scripts.append(script)
            return 'Rendered body\nSecond line'

    class FakeBrowser:
        def __init__(self) -> None:
            self.contexts = [type('Context', (), {'pages': [FakePage()]})()]
            self.closed = False

    browser = FakeBrowser()

    async def fake_active_url(runtime: Any, config: dict[str, Any]) -> str:
        return 'https://example.test/page'

    @contextlib.asynccontextmanager
    async def fake_connect_chromium(runtime: Any):
        try:
            yield browser
        finally:
            browser.closed = True

    async def fail_launch(runtime: Any) -> None:
        raise AssertionError('matching tab should not require launching Chrome')

    monkeypatch.setattr(chrome_getters, 'get_active_url_from_access_tree', fake_active_url)
    monkeypatch.setattr(chrome_getters, 'connect_chromium', fake_connect_chromium)
    monkeypatch.setattr(chrome_getters, '_launch_chrome_for_cdp', fail_launch)

    result = await chrome_getters.get_active_tab_url_and_body_text(_FakeRuntime(), {})

    assert result == {'url': 'https://example.test/page', 'body': 'Rendered body\nSecond line'}
    assert scripts == ["document.body ? document.body.innerText : ''"]
    assert browser.closed is True


@pytest.mark.asyncio
async def test_active_tab_url_and_body_text_opens_active_url_when_no_tab_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nexus.extensions.tasks.os_world.getters import chrome as chrome_getters

    class ExistingPage:
        url = 'https://example.test/other'

        async def evaluate(self, script: str) -> str:
            raise AssertionError('nonmatching page should not be evaluated')

    class NewPage:
        url = ''

        def __init__(self) -> None:
            self.timeout = None
            self.goto_calls: list[tuple[str, str, int]] = []
            self.closed = False

        def set_default_timeout(self, timeout_ms: int) -> None:
            self.timeout = timeout_ms

        async def goto(self, url: str, *, wait_until: str, timeout: int) -> None:
            self.goto_calls.append((url, wait_until, timeout))
            self.url = url

        async def evaluate(self, script: str) -> str:
            assert script == "document.body ? document.body.innerText : ''"
            return 'Fallback body'

        async def close(self) -> None:
            self.closed = True

    class FakeBrowser:
        def __init__(self, new_page: NewPage) -> None:
            self.contexts = [type('Context', (), {'pages': [ExistingPage()]})()]
            self.new_page_obj = new_page
            self.closed = False

        async def new_page(self) -> NewPage:
            return self.new_page_obj

    new_page = NewPage()
    browser = FakeBrowser(new_page)

    async def fake_active_url(runtime: Any, config: dict[str, Any]) -> str:
        return 'https://example.test/page'

    @contextlib.asynccontextmanager
    async def fake_connect_chromium(runtime: Any):
        try:
            yield browser
        finally:
            browser.closed = True

    monkeypatch.setattr(chrome_getters, 'get_active_url_from_access_tree', fake_active_url)
    monkeypatch.setattr(chrome_getters, 'connect_chromium', fake_connect_chromium)

    result = await chrome_getters.get_active_tab_url_and_body_text(_FakeRuntime(), {})

    assert result == {'url': 'https://example.test/page', 'body': 'Fallback body'}
    assert new_page.timeout == 60000
    assert new_page.goto_calls == [('https://example.test/page', 'load', 60000)]
    assert new_page.closed is True
    assert browser.closed is True


@pytest.mark.asyncio
async def test_recreation_page_expected_getter_returns_official_rules_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nexus.extensions.tasks.os_world.getters import chrome as chrome_getters

    async def fake_load(runtime: Any, config: dict[str, Any]):
        assert config['class'] == 'camp-sortable-column-header'
        return {'expected': {'camp-sortable-column-header': 'Next Available'}}

    async def fail_launch(runtime: Any) -> None:
        raise AssertionError('available CDP connection should not require launching Chrome')

    monkeypatch.setattr(chrome_getters, '_load_recreation_expected_content', fake_load)
    monkeypatch.setattr(chrome_getters, '_launch_chrome_for_cdp', fail_launch)

    result = await chrome_getters.get_goto_recreation_page_and_get_html_content(
        _FakeRuntime(),
        {
            'selector': 'class',
            'class': 'camp-sortable-column-header',
            'order': '2',
        },
    )

    assert result == {'expected': {'camp-sortable-column-header': 'Next Available'}}


@pytest.mark.asyncio
async def test_recreation_expected_from_page_extracts_ordered_class() -> None:
    from nexus.extensions.tasks.os_world.getters import chrome as chrome_getters

    class FakeElement:
        def __init__(self, text: str) -> None:
            self.text = text

        async def text_content(self) -> str:
            return self.text

    class FakePage:
        async def query_selector_all(self, selector: str) -> list[FakeElement]:
            assert selector == '.camp-sortable-column-header'
            return [
                FakeElement('Facility'),
                FakeElement('Location'),
                FakeElement('Next Available'),
            ]

    result = await chrome_getters._recreation_expected_from_page(
        FakePage(),
        {
            'selector': 'class',
            'class': 'camp-sortable-column-header',
            'order': '2',
        },
    )

    assert result == {'expected': {'camp-sortable-column-header': 'Next Available'}}


@pytest.mark.asyncio
async def test_rule_relative_time_supports_week_after_next(monkeypatch: pytest.MonkeyPatch) -> None:
    from nexus.extensions.tasks.os_world.getters import misc as misc_getters

    async def fake_now(runtime: Any) -> dt.datetime:
        return dt.datetime(2026, 7, 23, 12, 0, tzinfo=dt.timezone.utc)

    monkeypatch.setattr(misc_getters, '_vm_now', fake_now)

    result = await misc_getters.get_rule_relative_time(
        _FakeRuntime(),
        {
            'rules': {
                'relativeTime': {
                    'from': 'week after next Saturday',
                    'to': 'week after next Sunday',
                },
                'timezone': 'America/New_York',
                'expected': {
                    'from': '{DoW}, {Month} {Day0D}',
                    'to': '{DoW}, {Month} {Day0D}',
                },
            }
        },
    )

    assert result['expected'] == {'from': 'Sat, Aug 08', 'to': 'Sun, Aug 09'}


@pytest.mark.asyncio
async def test_active_tab_html_parse_input_reads_live_dom_value(monkeypatch: pytest.MonkeyPatch) -> None:
    from nexus.extensions.tasks.os_world.getters import chrome as chrome_getters

    async def fake_active_tab_info(runtime: Any, config: dict[str, Any]) -> dict[str, str]:
        return {
            'url': 'https://appointments.example.test/book',
            'content': '<input id="name"><input id="mail">',
        }

    async def fake_input_values(runtime: Any, active_url: str, config: dict[str, Any]) -> dict[str, str]:
        assert active_url == 'https://appointments.example.test/book'
        assert config['category'] == 'input'
        return {'name': 'James Smith', 'mail': 'james.smith@gmail.com'}

    async def fail_launch(runtime: Any) -> None:
        raise AssertionError('available CDP connection should not require launching Chrome')

    monkeypatch.setattr(chrome_getters, 'get_active_tab_info', fake_active_tab_info)
    monkeypatch.setattr(chrome_getters, '_active_tab_live_input_values', fake_input_values)
    monkeypatch.setattr(chrome_getters, '_launch_chrome_for_cdp', fail_launch)

    result = await chrome_getters.get_active_tab_html_parse(
        _FakeRuntime(),
        {
            'category': 'input',
            'inputObject': {
                '//*[@id="name"]': 'name',
                '//*[@id="mail"]': 'mail',
            },
        },
    )

    assert result == {'name': 'James Smith', 'mail': 'james.smith@gmail.com'}


def test_active_tab_html_parse_class_text_preserves_internal_whitespace() -> None:
    from nexus.extensions.tasks.os_world.getters import chrome as chrome_getters

    result = chrome_getters._parse_html_config(
        '<div class="location-info">Boston Logan Intl Airport,\n\tBOS <span>\n\tPick-Up</span></div>',
        'https://budget.example.test/reservation#/vehicles',
        {
            'category': 'class',
            'class_multiObject': {'location-info': {'0': 'start_location'}},
        },
    )

    assert result == {'start_location': 'Boston Logan Intl Airport,\n\tBOS \n\tPick-Up'}


@pytest.mark.asyncio
async def test_url_path_parse_matches_latest_macys_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nexus.extensions.tasks.os_world.getters import chrome as chrome_getters

    async def fake_active_url(runtime: Any, config: dict[str, Any]) -> str:
        return (
            'https://www.macys.com/shop/men/shirts/'
            'Product_department,Men_regular_size_t,Price_discount_range,Sleeve_length/'
            'Shirts,L,50_PERCENT_%20off%20%26%20more,Short%20Sleeve'
        )

    monkeypatch.setattr(chrome_getters, 'get_active_url_from_access_tree', fake_active_url)

    result = await chrome_getters.get_url_path_parse(
        _FakeRuntime(),
        {
            'parse_keys': [
                'mens_clothing',
                'shirts',
                'Men_regular_size_t',
                'Price_discount_range',
                'short_sleeve',
            ],
        },
    )

    assert result == {
        'mens_clothing': True,
        'shirts': True,
        'short_sleeve': True,
        'Men_regular_size_t': 'L',
        'Price_discount_range': '50_PERCENT_ off & more',
    }


def test_official_osworld_postconfig_action_types_are_supported() -> None:
    root = _official_examples_root()
    supported = {
        'activate_window',
        'chrome_inject_js',
        'close_window',
        'command',
        'download',
        'execute',
        'launch',
        'open',
        'sleep',
    }
    action_types: set[str] = set()

    for path in root.rglob('*.json'):
        evaluator = json.loads(path.read_text(encoding='utf-8')).get('evaluator') or {}
        for action in evaluator.get('postconfig') or []:
            if action_type := action.get('type'):
                action_types.add(str(action_type))

    assert action_types
    assert action_types <= supported


def test_official_metrics_replace_simple_fallbacks() -> None:
    metric = METRIC_REGISTRY.get('exact_match')

    assert metric is not simple_metrics.exact_match
    assert metric.__nexus_official_module__ == 'general'


def test_is_expected_bookmarks_accepts_raw_chrome_bookmarks_schema() -> None:
    raw_bookmarks = {
        'checksum': 'checksum',
        'roots': {
            'bookmark_bar': {
                'children': [
                    {
                        'children': [],
                        'name': 'Favorites',
                        'type': 'folder',
                    }
                ],
                'name': 'Bookmarks bar',
                'type': 'folder',
            },
        },
        'version': 1,
    }

    assert is_expected_bookmarks(raw_bookmarks, {'type': 'bookmark_bar_folders_names', 'names': ['Favorites']}) == 1.0


def test_is_expected_bookmarks_keeps_top_level_bookmark_bar_schema() -> None:
    bookmarks = {
        'bookmark_bar': {
            'children': [
                {
                    'children': [],
                    'name': 'Favorites',
                    'type': 'folder',
                }
            ],
            'name': 'Bookmarks bar',
            'type': 'folder',
        }
    }

    assert is_expected_bookmarks(bookmarks, {'type': 'bookmark_bar_folders_names', 'names': ['Favorites']}) == 1.0


_AUTO_SAVE_REGISTRY = """\
<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry">
  <item oor:path="/org.openoffice.Office.Common/Save/Document">
    <prop oor:name="AutoSaveTimeIntervall"><value>3</value></prop>
  </item>
</oor:items>
"""


@pytest.mark.parametrize(
    ('func_name', 'official_module', 'runtime_outputs', 'evaluator'),
    [
        (
            'is_extension_installed',
            'vscode',
            {'code --list-extensions': 'undefined_publisher.test\n'},
            {
                'func': 'is_extension_installed',
                'result': {'type': 'vm_command_line', 'command': ['code', '--list-extensions']},
                'expected': {'type': 'rule', 'rules': {'type': 'contain', 'expected': 'undefined_publisher.test'}},
            },
        ),
        (
            'check_config_status',
            'gimp',
            {'emit-gimprc': '(theme "Light")\n'},
            {
                'postconfig': [{'type': 'execute', 'parameters': {'command': 'emit-gimprc', 'stdout': 'gimprc'}}],
                'func': 'check_config_status',
                'result': {'type': 'cache_file', 'path': 'gimprc'},
                'expected': {'type': 'rule', 'rules': {'key': 'theme', 'value': '"Light"'}},
            },
        ),
        (
            'check_qt_bgcone',
            'vlc',
            {'emit-vlcrc': 'qt-bgcone=0\n'},
            {
                'postconfig': [{'type': 'execute', 'parameters': {'command': 'emit-vlcrc', 'stdout': 'vlcrc'}}],
                'func': 'check_qt_bgcone',
                'result': {'type': 'cache_file', 'path': 'vlcrc'},
                'expected': {'type': 'rule', 'rules': {'expected_qt_bgcone': 0}},
            },
        ),
        (
            'check_auto_saving_time',
            'slides',
            {'emit-registry': _AUTO_SAVE_REGISTRY},
            {
                'postconfig': [
                    {
                        'type': 'execute',
                        'parameters': {'command': 'emit-registry', 'stdout': 'registrymodifications.xcu'},
                    }
                ],
                'func': 'check_auto_saving_time',
                'result': {'type': 'cache_file', 'path': 'registrymodifications.xcu'},
                'expected': {'type': 'rule', 'rules': {'minutes': 3}},
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_official_import_failure_uses_simple_fallback(
    monkeypatch: pytest.MonkeyPatch,
    func_name: str,
    official_module: str,
    runtime_outputs: dict[str, _CommandResponse | str],
    evaluator: dict[str, Any],
) -> None:
    _fail_official_metric_imports(monkeypatch, official_module)

    result = await evaluate_os_world(_FakeRuntime(runtime_outputs), evaluator, unsupported_policy='fail')

    assert result['supported'] is True
    assert result['score'] == 1.0
    assert result['func'] == func_name


@pytest.mark.asyncio
async def test_vlc_config_metric_works_when_cv2_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_cv2_import(monkeypatch)

    result = await evaluate_os_world(
        _FakeRuntime({'emit-vlcrc': 'global-key-play-pause=Ctrl+Space\n'}),
        {
            'postconfig': [{'type': 'execute', 'parameters': {'command': 'emit-vlcrc', 'stdout': 'vlcrc'}}],
            'func': 'check_global_key_play_pause',
            'result': {'type': 'cache_file', 'path': 'vlcrc'},
            'expected': {'type': 'rule', 'rules': {'expected_global_key_play_pause': 1}},
        },
        unsupported_policy='fail',
    )

    assert result['supported'] is True
    assert result['score'] == 1.0


@pytest.mark.asyncio
async def test_gimp_config_metric_works_when_cv2_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_cv2_import(monkeypatch)

    result = await evaluate_os_world(
        _FakeRuntime({'emit-gimprc': '(theme "Light")\n'}),
        {
            'postconfig': [{'type': 'execute', 'parameters': {'command': 'emit-gimprc', 'stdout': 'gimprc'}}],
            'func': 'check_config_status',
            'result': {'type': 'cache_file', 'path': 'gimprc'},
            'expected': {'type': 'rule', 'rules': {'key': 'theme', 'value': '"Light"'}},
        },
        unsupported_policy='fail',
    )

    assert result['supported'] is True
    assert result['score'] == 1.0


@pytest.mark.asyncio
async def test_docs_docx_metric_works_when_cv2_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _fail_cv2_import(monkeypatch)
    actual_docx = tmp_path / 'actual.docx'
    expected_docx = tmp_path / 'expected.docx'
    _write_docx(actual_docx, ['Nexus OSWorld', 'Writer metric'])
    _write_docx(expected_docx, ['Nexus OSWorld', 'Writer metric'])

    result = await evaluate_os_world(
        _FakeRuntime(),
        {
            'func': 'compare_docx_files',
            'result': {'type': 'rule', 'rules': str(actual_docx)},
            'expected': {'type': 'rule', 'rules': str(expected_docx)},
        },
        unsupported_policy='fail',
    )

    assert result['supported'] is True
    assert result['score'] == 1.0


@pytest.mark.asyncio
async def test_docs_ocr_metric_is_unsupported_when_cv2_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _fail_cv2_import(monkeypatch)
    docs = importlib.import_module('nexus.extensions.tasks.os_world.metrics.official.docs')

    def unavailable_ocr():
        raise ImportError('libGL.so.1: cannot open shared object file')

    monkeypatch.setattr(docs, '_easyocr', unavailable_ocr)
    image_path = tmp_path / 'screen.png'
    image_path.write_bytes(b'not a real png')
    evaluator = {
        'func': 'compare_image_text',
        'result': {'type': 'rule', 'rules': str(image_path)},
        'expected': {'type': 'rule', 'rules': {'type': 'text', 'text': 'target'}},
    }

    skipped = await evaluate_os_world(_FakeRuntime(), evaluator, unsupported_policy='skip')

    assert skipped['supported'] is False
    assert skipped['score'] == 0.0
    assert 'compare_image_text' in skipped['error']
    assert 'libGL.so.1' in skipped['error']

    with pytest.raises(NotImplementedError, match='compare_image_text'):
        await evaluate_os_world(_FakeRuntime(), evaluator, unsupported_policy='fail')


@pytest.mark.asyncio
async def test_vlc_video_metric_is_unsupported_when_cv2_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_cv2_import(monkeypatch)
    evaluator = {
        'func': 'compare_videos',
        'result': {'type': 'rule', 'rules': '/tmp/actual.mp4'},
        'expected': {'type': 'rule', 'rules': '/tmp/expected.mp4'},
    }

    skipped = await evaluate_os_world(_FakeRuntime(), evaluator, unsupported_policy='skip')

    assert skipped['supported'] is False
    assert skipped['score'] == 0.0
    assert 'compare_videos' in skipped['error']
    assert 'libGL.so.1' in skipped['error']

    with pytest.raises(NotImplementedError, match='compare_videos'):
        await evaluate_os_world(_FakeRuntime(), evaluator, unsupported_policy='fail')


@pytest.mark.asyncio
async def test_official_import_failure_without_fallback_is_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = {'func': 'compare_table'}
    _fail_official_metric_imports(monkeypatch, 'table')

    skipped = await evaluate_os_world(_FakeRuntime(), evaluator, unsupported_policy='skip')

    assert skipped['supported'] is False
    assert skipped['score'] == 0.0
    assert 'compare_table' in skipped['error']
    assert 'unavailable official metric dependencies' in skipped['error']

    with pytest.raises(NotImplementedError, match='compare_table'):
        await evaluate_os_world(_FakeRuntime(), evaluator, unsupported_policy='fail')


@pytest.mark.asyncio
async def test_official_import_failure_with_incomplete_fallback_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_official_metric_imports(monkeypatch, 'vscode')
    evaluator = {
        'func': 'is_extension_installed',
        'result': {'type': 'vm_command_line', 'command': ['code', '--list-extensions']},
        'expected': {'type': 'rule', 'rules': {'type': 'unsupported', 'expected': 'undefined_publisher.test'}},
    }

    skipped = await evaluate_os_world(_FakeRuntime(), evaluator, unsupported_policy='skip')

    assert skipped['supported'] is False
    assert skipped['score'] == 0.0
    assert 'Unsupported is_extension_installed rule type' in skipped['error']

    with pytest.raises(NotImplementedError, match='Unsupported is_extension_installed rule type'):
        await evaluate_os_world(_FakeRuntime(), evaluator, unsupported_policy='fail')


@pytest.mark.parametrize(
    ('from_label', 'to_label'),
    [
        ('next Monday split', 'next Friday split'),
        ('week after next Monday split', 'week after next Friday split'),
    ],
)
@pytest.mark.asyncio
async def test_rule_relative_time_split_labels_fill_pickup_and_dropoff_components(
    monkeypatch: pytest.MonkeyPatch,
    from_label: str,
    to_label: str,
) -> None:
    """Every ``split`` phrasing must write pu*/do* parts, not a ``from``/``to`` pair.

    Regression for chrome/1704f00f-79e6-43a7-961b-cedd3724d5fd, whose second
    metric uses the ``week after next`` phrasing and used to raise KeyError.
    """
    import datetime as dt

    from nexus.extensions.tasks.os_world.getters import misc as misc_getters

    now = dt.datetime(2026, 9, 20, 12, 0, tzinfo=dt.timezone.utc)

    async def _fake_now(_runtime: Any) -> dt.datetime:
        return now

    monkeypatch.setattr(misc_getters, '_vm_now', _fake_now)

    config = {
        'type': 'rule_relativeTime',
        'rules': {
            'relativeTime': {'from': from_label, 'to': to_label},
            'timezone': 'Europe/Zurich',
            'expected': {
                'locationName': 'Zürich',
                'filterCriteria_carCategory': 'large',
                'puDay': '{DayD}',
                'puMonth': '{MonthD}',
                'puYear': '{Year}',
                'doDay': '{DayD}',
                'doMonth': '{MonthD}',
                'doYear': '{Year}',
            },
        },
    }

    rules = await misc_getters.get_rule_relative_time(_FakeRuntime(), config)
    expected = rules['expected']

    # placeholders are resolved, and no from/to key is invented
    assert 'from' not in expected
    assert 'to' not in expected
    for key in ('puDay', 'puMonth', 'puYear', 'doDay', 'doMonth', 'doYear'):
        assert expected[key].isdigit(), f'{key} unresolved: {expected[key]!r}'

    # untouched keys survive
    assert expected['locationName'] == 'Zürich'

    # pickup is a Monday, dropoff the Friday of the same week
    pu = dt.date(int(expected['puYear']), int(expected['puMonth']), int(expected['puDay']))
    do = dt.date(int(expected['doYear']), int(expected['doMonth']), int(expected['doDay']))
    assert pu.weekday() == 0
    assert do.weekday() == 4
    assert (do - pu).days == 4
    assert pu > now.date()


@pytest.mark.asyncio
async def test_rule_relative_time_non_split_labels_still_use_from_and_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-split phrasings keep the single from/to contract."""
    import datetime as dt

    from nexus.extensions.tasks.os_world.getters import misc as misc_getters

    async def _fake_now(_runtime: Any) -> dt.datetime:
        return dt.datetime(2026, 9, 20, 12, 0, tzinfo=dt.timezone.utc)

    monkeypatch.setattr(misc_getters, '_vm_now', _fake_now)

    config = {
        'type': 'rule_relativeTime',
        'rules': {
            'relativeTime': {'from': 'next week Friday', 'to': 'next week Sunday'},
            'expected': {'from': '{Year}-{Month0D}-{Day0D}', 'to': '{Year}-{Month0D}-{Day0D}'},
        },
    }

    rules = await misc_getters.get_rule_relative_time(_FakeRuntime(), config)
    expected = rules['expected']

    assert '{' not in expected['from']
    assert '{' not in expected['to']
    assert expected['from'] < expected['to']


def test_is_split_label_matches_only_the_split_suffix_word() -> None:
    from nexus.extensions.tasks.os_world.getters.misc import _is_split_label

    assert _is_split_label('next Monday split')
    assert _is_split_label('week after next Friday split')
    assert not _is_split_label('next week Friday')
    assert not _is_split_label('tomorrow')
    assert not _is_split_label('')
    assert not _is_split_label(None)
    # a label merely containing the word must not be mistaken for the suffix
    assert not _is_split_label('split the booking')


@pytest.mark.asyncio
async def test_os_world_evaluator_scores_command_line_exact_match():
    from nexus.extensions.tasks.os_world.evaluation.evaluator import evaluate_os_world

    class FakeRuntime:
        async def run_command(self, command, timeout=60, cwd=None):
            assert 'printf true' in command
            assert 'runuser -u user' in command
            return SimpleNamespace(return_code=0, stdout='true', stderr='')

        async def gui_get_screen_size(self):
            return SimpleNamespace(success=True, width=1280, height=800)

    result = await evaluate_os_world(
        FakeRuntime(),
        {
            'func': 'exact_match',
            'result': {'type': 'vm_command_line', 'command': 'printf true', 'shell': True},
            'expected': {'type': 'rule', 'rules': {'expected': 'true'}},
        },
    )

    assert result['score'] == 1.0
    assert result['actual'] == 'true'
    assert result['supported'] is True


@pytest.mark.asyncio
async def test_os_world_evaluator_preserves_command_line_newlines_for_official_exact_match():
    from nexus.extensions.tasks.os_world.evaluation.evaluator import evaluate_os_world

    class FakeRuntime:
        async def run_command(self, command, timeout=60, cwd=None):
            assert 'printf "100\\n"' in command
            assert 'runuser -u user' in command
            return SimpleNamespace(return_code=0, stdout='100\n', stderr='')

    result = await evaluate_os_world(
        FakeRuntime(),
        {
            'func': 'exact_match',
            'result': {'type': 'vm_command_line', 'command': 'printf "100\\n"', 'shell': True},
            'expected': {'type': 'rule', 'rules': {'expected': '100\n'}},
        },
    )

    assert result['score'] == 1.0
    assert result['actual'] == '100\n'


@pytest.mark.asyncio
async def test_os_world_evaluator_scores_non_infeasible_fail_as_zero_without_state_check():
    from nexus.extensions.tasks.os_world.evaluation.evaluator import evaluate_os_world

    class FakeRuntime:
        async def run_command(self, command, timeout=60, cwd=None):
            raise AssertionError('FAIL should short-circuit ordinary OSWorld evaluators')

    result = await evaluate_os_world(
        FakeRuntime(),
        {
            'func': 'exact_match',
            'result': {'type': 'vm_command_line', 'command': 'printf true', 'shell': True},
            'expected': {'type': 'rule', 'rules': {'expected': 'true'}},
        },
        unsupported_policy='fail',
        last_action='FAIL',
    )

    assert result['score'] == 0.0
    assert result['supported'] is True
    assert result['last_action'] == 'FAIL'


@pytest.mark.asyncio
async def test_os_world_evaluator_preserves_shell_pipes_in_command_lists():
    from nexus.extensions.tasks.os_world.evaluation.evaluator import evaluate_os_world

    commands: list[str] = []

    class FakeRuntime:
        async def run_command(self, command, timeout=60, cwd=None):
            commands.append(command)
            return SimpleNamespace(return_code=0, stdout='test.py\n', stderr='')

    result = await evaluate_os_world(
        FakeRuntime(),
        {
            'func': 'is_extension_installed',
            'result': {'type': 'vm_command_line', 'command': ['ls', '/home/user/Desktop', '|', 'grep', 'test.py']},
            'expected': {'type': 'rule', 'rules': {'type': 'contain', 'expected': 'test.py'}},
        },
    )

    assert result['score'] == 1.0
    assert len(commands) == 1
    assert 'ls /home/user/Desktop | grep test.py' in commands[0]
    assert 'runuser -u user' in commands[0]


@pytest.mark.asyncio
async def test_os_world_evaluator_enable_do_not_track_uses_official_chrome_preferences_path():
    from nexus.extensions.tasks.os_world.evaluation.evaluator import evaluate_os_world

    commands: list[str] = []

    class FakeRuntime:
        async def run_command(self, command, timeout=60, cwd=None):
            commands.append(command)
            assert '/home/user/.config/google-chrome/Default/Preferences' in command
            assert 'base64 -w0' in command
            assert 'glob' not in command
            assert '.config/*/Default/Preferences' not in command
            return SimpleNamespace(return_code=0, stdout=_base64_text('{"enable_do_not_track": true}'), stderr='')

    result = await evaluate_os_world(
        FakeRuntime(),
        {
            'func': 'exact_match',
            'result': {'type': 'enable_do_not_track'},
            'expected': {'type': 'rule', 'rules': {'expected': 'true'}},
        },
        unsupported_policy='fail',
    )

    assert result['score'] == 1.0
    assert result['actual'] == 'true'
    assert len(commands) == 1


@pytest.mark.asyncio
async def test_os_world_evaluator_covers_selected_official_shapes():
    from nexus.extensions.tasks.os_world.evaluation.evaluator import evaluate_os_world

    auto_save_registry = """\
<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry">
  <item oor:path="/org.openoffice.Office.Common/Save/Document">
    <prop oor:name="AutoSaveTimeIntervall"><value>3</value></prop>
  </item>
</oor:items>
"""

    class FakeRuntime:
        async def run_command(self, command, timeout=60, cwd=None):
            if 'base64 -w0 /home/user/.config/google-chrome/Default/Preferences' in command:
                return SimpleNamespace(return_code=0, stdout=_base64_text('{"enable_do_not_track": true}'), stderr='')
            if 'base64 -w0 /home/user/.config/GIMP/2.10/gimprc' in command:
                return SimpleNamespace(return_code=0, stdout=_base64_text('(theme "Light")\n'), stderr='')
            if 'base64 -w0 /home/user/.config/vlc/vlcrc' in command:
                return SimpleNamespace(return_code=0, stdout=_base64_text('qt-bgcone=0\n'), stderr='')
            if 'base64 -w0 /home/user/.config/libreoffice/4/user/registrymodifications.xcu' in command:
                return SimpleNamespace(return_code=0, stdout=_base64_text(auto_save_registry), stderr='')
            if 'code --list-extensions | grep undefined_publisher.test' in command:
                return SimpleNamespace(return_code=0, stdout='undefined_publisher.test\n', stderr='')
            if 'printf "100\\n"' in command:
                return SimpleNamespace(return_code=0, stdout='100\n', stderr='')
            if 'bash eval.sh' in command:
                return SimpleNamespace(return_code=0, stdout='SUCCESS: The task was completed correctly.\n', stderr='')
            raise AssertionError(f'unexpected command: {command}')

        async def gui_get_screen_size(self):
            return SimpleNamespace(success=True, width=1280, height=800)

    evaluator_cases = [
        {
            'func': 'exact_match',
            'result': {'type': 'enable_do_not_track'},
            'expected': {'type': 'rule', 'rules': {'expected': 'true'}},
        },
        {
            'func': 'check_config_status',
            'result': {'type': 'gimp_config_file', 'file_name': 'gimprc', 'dest': 'gimprc'},
            'expected': {'type': 'rule', 'rules': {'type:': 'key-value', 'key': 'theme', 'value': '"Light"'}},
        },
        {
            'func': 'check_qt_bgcone',
            'result': {'type': 'vlc_config', 'dest': 'vlcrc'},
            'expected': {'type': 'rule', 'rules': {'expected_qt_bgcone': 0}},
        },
        {
            'func': 'check_auto_saving_time',
            'result': {
                'type': 'vm_file',
                'path': '/home/user/.config/libreoffice/4/user/registrymodifications.xcu',
                'dest': 'registrymodifications.xcu',
            },
            'expected': {'type': 'rule', 'rules': {'minutes': 3}},
        },
        {
            'func': 'is_extension_installed',
            'result': {
                'type': 'vm_command_line',
                'command': ['code', '--list-extensions', '|', 'grep', 'undefined_publisher.test'],
            },
            'expected': {'type': 'rule', 'rules': {'type': 'contain', 'expected': 'undefined_publisher.test'}},
        },
        {
            'func': 'exact_match',
            'result': {'type': 'vm_command_line', 'command': 'printf "100\\n"', 'shell': True},
            'expected': {'type': 'rule', 'rules': {'expected': '100\n'}},
        },
        {
            'func': 'check_include_exclude',
            'result': {'type': 'vm_command_line', 'command': 'bash eval.sh', 'shell': True},
            'expected': {
                'type': 'rule',
                'rules': {'include': ['SUCCESS: The task was completed correctly.'], 'exclude': []},
            },
        },
    ]

    for evaluator in evaluator_cases:
        result = await evaluate_os_world(FakeRuntime(), evaluator, unsupported_policy='fail')
        assert result['score'] == 1.0, evaluator
        assert result['supported'] is True


def _base64_text(text: str) -> str:
    return base64.b64encode(text.encode('utf-8')).decode('ascii')
