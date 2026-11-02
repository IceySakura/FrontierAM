"""OSWorld document getters regressions."""

from __future__ import annotations

import asyncio
import base64
import builtins
import shlex
import sys

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from docx import Document

from nexus.extensions.tasks.os_world.evaluation.errors import EvaluatorInfrastructureError
from nexus.extensions.tasks.os_world.evaluation.evaluator import evaluate_os_world
from nexus.extensions.tasks.os_world.getters import chrome
from nexus.extensions.tasks.os_world.getters.vscode import get_vscode_config
from nexus.extensions.tasks.os_world.metrics.official.docs import compare_docx_tables


def _write_tables(path: Path, tables: list[list[list[str]]]) -> Path:
    document = Document()
    for rows in tables:
        table = document.add_table(rows=len(rows), cols=len(rows[0]))
        for row, values in zip(table.rows, rows, strict=True):
            for cell, text in zip(row.cells, values, strict=True):
                cell.text = text
    document.save(path)
    return path


@pytest.mark.parametrize('options', [{}, {'ignore_case_rows': None}, {'ignore_case_rows': []}])
def test_docx_tables_default_remains_case_sensitive(tmp_path: Path, options: dict) -> None:
    actual = _write_tables(tmp_path / 'actual.docx', [[['MODEL', 'SCORE'], ['GPT-4', '0.5000']]])
    expected = _write_tables(tmp_path / 'expected.docx', [[['Model', 'Score'], ['GPT-4', '0.5000']]])

    assert compare_docx_tables(actual, expected, **options) == 0
    assert compare_docx_tables(actual, actual, **options) == 1


def test_docx_tables_ignore_case_only_in_selected_zero_based_rows(tmp_path: Path) -> None:
    actual = _write_tables(tmp_path / 'actual.docx', [[[' MODEL ', 'SCORE'], ['GPT-4', '0.5000']]])
    expected = _write_tables(tmp_path / 'expected.docx', [[['Model', 'Score'], ['GPT-4', '0.5000']]])

    assert compare_docx_tables(actual, expected, ignore_case_rows=[0]) == 1
    assert compare_docx_tables(actual, expected, ignore_case_rows=[1]) == 0


@pytest.mark.parametrize(
    'rows',
    [
        [['Model', 'Score'], ['gpt-4', '0.5000']],  # Unselected row still case-sensitive.
        [['Wrong header', 'Score'], ['GPT-4', '0.5000']],  # Do not skip the header.
        [['Model', 'Score'], ['GPT-4', '0.5']],  # Preserve displayed numeric precision.
        [['Model', 'Score']],  # Missing row.
        [['Model'], ['GPT-4']],  # Missing column.
    ],
)
def test_docx_tables_case_option_does_not_hide_other_mismatches(tmp_path: Path, rows: list[list[str]]) -> None:
    actual = _write_tables(tmp_path / 'actual.docx', [rows])
    expected = _write_tables(tmp_path / 'expected.docx', [[['MODEL', 'SCORE'], ['GPT-4', '0.5000']]])

    assert compare_docx_tables(actual, expected, ignore_case_rows=[0]) == 0


def test_docx_tables_row_indices_restart_for_each_table(tmp_path: Path) -> None:
    actual = _write_tables(tmp_path / 'actual.docx', [[['HEADER'], ['Value']], [['OTHER'], ['Value']]])
    expected = _write_tables(tmp_path / 'expected.docx', [[['header'], ['Value']], [['other'], ['Value']]])
    missing_table = _write_tables(tmp_path / 'missing.docx', [[['header'], ['Value']]])

    assert compare_docx_tables(actual, expected, ignore_case_rows=[0]) == 1
    assert compare_docx_tables(actual, missing_table, ignore_case_rows=[0]) == 0


def test_docx_tables_missing_and_invalid_files_still_score_zero(tmp_path: Path) -> None:
    expected = _write_tables(tmp_path / 'expected.docx', [[['Header']]])
    invalid = tmp_path / 'invalid.docx'
    invalid.write_text('not a DOCX')

    assert compare_docx_tables(None, expected, ignore_case_rows=[0]) == 0
    assert compare_docx_tables(tmp_path / 'missing.docx', expected, ignore_case_rows=[0]) == 0
    assert compare_docx_tables(invalid, expected, ignore_case_rows=[0]) == 0


@pytest.mark.asyncio
async def test_docx_tables_evaluator_forwards_ignore_case_rows(tmp_path: Path) -> None:
    actual = _write_tables(tmp_path / 'actual.docx', [[['MODEL', 'SCORE'], ['GPT-4', '0.5000']]])
    expected = _write_tables(tmp_path / 'expected.docx', [[['Model', 'Score'], ['GPT-4', '0.5000']]])
    result = await evaluate_os_world(
        object(),
        {
            'func': 'compare_docx_tables',
            'result': {'type': 'cache_file', 'path': str(actual)},
            'expected': {'type': 'cache_file', 'path': str(expected)},
            'options': {'ignore_case_rows': [0]},
        },
        unsupported_policy='fail',
        last_action='DONE',
    )

    assert result['score'] == 1.0
    assert result['supported'] is True


@pytest.mark.asyncio
async def test_shortcut_framing_preserves_multiline_contents(monkeypatch):
    contents = '[Desktop Entry]\nName=应用\nExec=/opt/google/chrome --app=https://example.test\nComment=a\tb\n'
    encoded = base64.b64encode(contents.encode()).decode()
    monkeypatch.setattr(
        chrome,
        'run_bash',
        AsyncMock(
            return_value=SimpleNamespace(stdout=f'Example App.desktop\t{encoded}\ninvalid frame\n', return_code=0)
        ),
    )
    assert await chrome.get_shortcuts_on_desktop(None, {}) == {'Example App.desktop': contents}


@pytest.mark.asyncio
@pytest.mark.parametrize('exists', [False, True])
@pytest.mark.skipif(sys.platform != 'linux', reason='Exercise production GNU find/base64 pipeline on Linux')
async def test_linux_shortcut_getter_with_real_files_and_official_metric(monkeypatch, tmp_path, exists):
    from nexus.extensions.tasks.os_world.metrics.official.chrome import is_shortcut_on_desktop

    desktop = tmp_path / 'Desktop with spaces'
    desktop.mkdir()
    text = '[Desktop Entry]\nName=Play Puzzle Game 2048\nExec=google-chrome --app=https://example.test\n'
    if exists:
        (desktop / 'Chrome Example.desktop').write_text(text)

    async def run(_runtime, command, **kwargs):
        # Replace only the sandbox root, execute the actual getter pipeline.
        proc = await asyncio.create_subprocess_exec(
            'bash',
            '-c',
            command.replace('/home/user/Desktop', shlex.quote(str(desktop))),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=kwargs['timeout'])
        assert proc.returncode == 0, stderr.decode()
        return SimpleNamespace(stdout=stdout.decode(), return_code=proc.returncode)

    monkeypatch.setattr(chrome, 'run_bash', run)
    getter = chrome.get_shortcuts_on_desktop
    actual = await getter(None, {})
    assert actual == ({'Chrome Example.desktop': text} if exists else {})
    rule = {'type': 'name', 'name': 'Play Puzzle Game 2048'}
    assert is_shortcut_on_desktop(actual, rule) == float(exists)
    assert is_shortcut_on_desktop(actual, {'type': 'name', 'name': 'Wrong App'}) == 0.0
    assert is_shortcut_on_desktop({'Chrome Example.desktop': str(desktop / 'Chrome Example.desktop')}, rule) == 0.0


class ExtensionRuntime:
    def __init__(self, output, *, produce=True, contents='actual-workspace', returncode=0):
        self.output = output
        self.produce = produce
        self.contents = contents
        self.returncode = returncode
        self.actions = []
        self.downloads = []
        self.elapsed = 0.0
        self.download_error = None

    async def run_python(self, code):
        def sleep(seconds):
            self.elapsed += seconds

        def press(key):
            self.actions.append(('press', key))
            if self.produce:
                self.output.write_text(self.contents)

        modules = {
            'time': SimpleNamespace(sleep=sleep, monotonic=lambda: self.elapsed),
            'pyautogui': SimpleNamespace(
                hotkey=lambda *keys: self.actions.append(('hotkey', keys)),
                write=lambda text, **kwargs: self.actions.append(('write', text)),
                press=press,
            ),
        }

        def import_module(name, *args, **kwargs):
            return modules[name] if name in modules else builtins.__import__(name, *args, **kwargs)

        try:
            exec(code, {'__builtins__': {**vars(builtins), '__import__': import_module}})
        except RuntimeError:
            return {'returncode': 1}
        return {'returncode': self.returncode}

    async def get_file(self, path):
        self.downloads.append(path)
        if self.download_error:
            raise self.download_error
        return Path(path).read_bytes()


def config(tmp_path, command='OpenProject'):
    return {
        'vscode_extension_command': command,
        'path': str(tmp_path / 'extension-output.txt'),
        'dest': 'observed.txt',
        '__nexus_cache_dir': str(tmp_path / 'cache'),
    }


@pytest.mark.asyncio
async def test_queries_extension_before_downloading_current_observation(tmp_path):
    cfg = config(tmp_path)
    runtime = ExtensionRuntime(Path(cfg['path']))
    result = await get_vscode_config(runtime, cfg)
    assert Path(result).read_text() == 'actual-workspace'
    assert runtime.actions == [('hotkey', ('ctrl', 'shift', 'p')), ('write', 'OpenProject'), ('press', 'enter')]
    assert runtime.downloads == [cfg['path']]


@pytest.mark.asyncio
@pytest.mark.parametrize('stale', [False, True])
async def test_missing_or_stale_extension_output_is_infrastructure_error(tmp_path, stale):
    cfg = config(tmp_path)
    if stale:
        Path(cfg['path']).write_text('project-from-yesterday')
    runtime = ExtensionRuntime(Path(cfg['path']), produce=False)
    with pytest.raises(EvaluatorInfrastructureError, match='fresh'):
        await get_vscode_config(runtime, cfg)
    assert runtime.downloads == []  # No historical workspaceStorage fallback.


@pytest.mark.asyncio
async def test_refresh_replaces_stale_data_but_does_not_mutate_existing_file_itself(tmp_path):
    cfg = config(tmp_path)
    Path(cfg['path']).write_text('stale-project')
    runtime = ExtensionRuntime(Path(cfg['path']), contents='different-current-project')
    result = await get_vscode_config(runtime, cfg)
    assert Path(result).read_text() == 'different-current-project'


@pytest.mark.asyncio
@pytest.mark.parametrize('contents', ['', 'wrong-project'])
async def test_valid_negative_observation_is_not_an_infrastructure_failure(tmp_path, contents):
    cfg = config(tmp_path)
    runtime = ExtensionRuntime(Path(cfg['path']), contents=contents)
    result = await get_vscode_config(runtime, cfg)
    assert Path(result).read_text() == contents


@pytest.mark.asyncio
async def test_command_is_data_not_python_code(tmp_path):
    command = 'Query "quoted"; print("not executed")'
    cfg = config(tmp_path, command)
    runtime = ExtensionRuntime(Path(cfg['path']))
    await get_vscode_config(runtime, cfg)
    assert runtime.actions[1] == ('write', command)


@pytest.mark.asyncio
@pytest.mark.parametrize('error', [FileNotFoundError('gone'), ConnectionError('disconnected')])
async def test_failed_download_remains_infrastructure_error(tmp_path, error):
    cfg = config(tmp_path)
    runtime = ExtensionRuntime(Path(cfg['path']))
    runtime.download_error = error
    with pytest.raises(EvaluatorInfrastructureError):
        await get_vscode_config(runtime, cfg)


@pytest.mark.asyncio
@pytest.mark.parametrize('command', [None, '', '   ', 3])
async def test_missing_query_command_rejected(tmp_path, command):
    cfg = config(tmp_path, command)
    runtime = ExtensionRuntime(Path(cfg['path']))
    with pytest.raises(EvaluatorInfrastructureError, match='vscode_extension_command'):
        await get_vscode_config(runtime, cfg)
    assert not runtime.actions
