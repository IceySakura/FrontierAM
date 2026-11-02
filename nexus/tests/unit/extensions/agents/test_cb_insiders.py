import asyncio
import base64
import contextlib
import json
import os
import shlex
import sys
import threading
import time

from pathlib import Path

import psutil
import pytest

from nexus.core import agent_registry
from nexus.core.module_meta import parse_directory_module
from nexus.extensions.agents.cb_insiders import agent, installer, process, runtime_setup
from nexus.extensions.agents.cb_insiders.output import parse_output
from nexus.utils.timeout import async_timeout


def test_cb_insiders_is_discovered_under_the_requested_name(request):
    root = request.config.rootpath
    meta = parse_directory_module(root / 'nexus/extensions/agents/cb_insiders', root)
    assert meta.name == 'cb-insiders'
    assert meta.objects['agents'] == {'cb-insiders': 'nexus.extensions.agents.cb_insiders.agent'}
    assert agent_registry.get('cb-insiders').__name__ == 'run_cb_insiders'


@pytest.mark.parametrize(
    ('machine', 'expected'), [('x86_64', 'x64'), ('amd64', 'x64'), ('aarch64', 'arm64'), ('arm64', 'arm64')]
)
def test_package_selection_uses_the_runtime_architecture(monkeypatch, machine, expected):
    monkeypatch.setattr(installer.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(installer.platform, 'machine', lambda: machine)
    assert installer.runtime_architecture() == expected


@pytest.mark.parametrize(('system', 'machine'), [('Darwin', 'arm64'), ('Linux', 's390x')])
def test_unsupported_platform_fails_before_install(monkeypatch, system, machine):
    monkeypatch.setattr(installer.platform, 'system', lambda: system)
    monkeypatch.setattr(installer.platform, 'machine', lambda: machine)
    with pytest.raises(RuntimeError, match='supports Linux|Unsupported'):
        installer.runtime_architecture()


@pytest.mark.parametrize('version', ['latest', '^5.0.0', 'file:/tmp/app', '5.0.0;touch /tmp/injected', '../5.0.0'])
def test_input_rejects_unpinned_or_executable_package_specs(version):
    with pytest.raises(ValueError, match='exact npm version'):
        agent.CbInsidersInput(user_prompt='test', working_dir='/workspace', version=version)


def test_cli_arguments_keep_prompt_as_data_and_preserve_ide_cli_mode():
    prompt = '--model rogue\n$(touch /tmp/should-not-run) `echo no`'
    cmd = runtime_setup.build_argv(Path('/opt/cb app'), prompt, '/work dir', '/tmp/own-data')
    assert cmd[0] == '/opt/cb app/bin/buddycn-insiders'
    assert cmd[cmd.index('--cli') + 1] == '--'
    assert cmd[-3:] == ['--', prompt, '/work dir']
    assert '--use-inmemory-secretstorage' in cmd
    assert '--password-store=basic' in cmd
    assert cmd[cmd.index('--output-format') + 1] == 'stream-json'
    assert cmd[cmd.index('--permission-mode') + 1] == 'bypassPermissions'


@pytest.mark.parametrize('flag', ['--model=other', '--output-format', '--input-format=json', '--permission-mode', '--'])
def test_extra_arguments_cannot_override_the_agent_protocol(flag):
    with pytest.raises(ValueError, match='managed'):
        agent.CbInsidersInput(user_prompt='test', working_dir='/workspace', extra_cli_args=[flag])


def test_environment_isolates_ide_state_and_gateway_keys(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', '/task/home')
    monkeypatch.setenv('XDG_CONFIG_HOME', '/task/config')
    monkeypatch.setenv('VSCODE_IPC_HOOK_CLI', 'desktop-session')
    monkeypatch.setenv('OPENAI_API_KEY', 'host-only-secret')
    monkeypatch.setenv('CODEBUDDY_IDE_PRODUCT_FILE', '/shared/desktop-product.json')
    monkeypatch.setenv('WORKBUDDY_CONFIG_DIR', '/desktop')
    monkeypatch.setenv('CB_INSIDERS_NPM_TOKEN', 'registry-only-secret')
    env = runtime_setup.build_env(
        Path('/opt/cb'),
        tmp_path,
        'http://127.0.0.1:1234',
        {
            'CODEBUDDY_CONFIG_DIR': '/wrong',
            'ACC_PRODUCT_CONFIG_V3': '{}',
            'HOME': '/wrong',
            'NODE_OPTIONS': '--require /wrong.js',
            'LANG': 'en_US.UTF-8',
        },
    )
    assert 'OPENAI_API_KEY' not in env
    assert 'VSCODE_IPC_HOOK_CLI' not in env
    assert 'WORKBUDDY_CONFIG_DIR' not in env
    assert 'CB_INSIDERS_NPM_TOKEN' not in env
    assert 'NODE_OPTIONS' not in env
    assert env['HOME'] == '/task/home'
    assert env['XDG_CONFIG_HOME'] == '/task/config'
    assert env['CODEBUDDY_CONFIG_DIR'] == str(tmp_path / 'config')
    assert env['CODEBUDDY_REMOTE_CONFIG_DISABLED'] == '1'
    assert env['LANG'] == 'en_US.UTF-8'
    product = json.loads(env['ACC_PRODUCT_CONFIG_V3'])
    assert product['endpoint'] == 'http://127.0.0.1:1234'
    assert 'agents' not in product
    assert 'prompts' not in product
    assert 'tools' not in product
    assert product['authentication']['attributes']['token'] == 'dummy-key-for-proxy'


def test_model_configuration_routes_main_and_helper_requests_to_nexus(tmp_path):
    runtime_setup.write_config(
        tmp_path, 'http://127.0.0.1:9999', 'nexus', max_input_tokens=131072, max_output_tokens=4096
    )
    config = json.loads((tmp_path / 'config/models.json').read_text())
    assert config['availableModels'] == ['nexus']
    model = config['models'][0]
    assert model['url'] == 'http://127.0.0.1:9999/v1/chat/completions'
    assert model['relatedModels'] == {'lite': 'nexus', 'reasoning': 'nexus'}
    assert model['maxOutputTokens'] == 4096
    assert model['maxInputTokens'] == 131072


def test_final_result_wins_over_progress_and_preserves_permissions():
    lines = [
        {'type': 'assistant', 'message': {'content': [{'type': 'text', 'text': 'Working...'}]}},
        {'type': 'user', 'message': {'content': [{'type': 'tool_result', 'content': 'PRIVATE_TOOL_TEXT'}]}},
        {'type': 'assistant', 'message': {'content': [{'type': 'text', 'text': 'Done.'}]}},
        {
            'type': 'result',
            'subtype': 'success',
            'is_error': False,
            'result': 'Done.',
            'session_id': 'session-1',
            'permission_denials': [{'tool_name': 'WebSearch'}],
        },
    ]
    parsed = parse_output('startup noise\n' + '\n'.join(map(json.dumps, lines)), 0)
    assert parsed['success'] is True
    assert parsed['assistant_text'] == 'Done.'
    assert parsed['session_id'] == 'session-1'
    assert parsed['permission_denials'] == [{'tool_name': 'WebSearch'}]


@pytest.mark.parametrize(
    ('event', 'code', 'timeout', 'status', 'stage'),
    [
        ({'type': 'assistant'}, 0, False, 'error', 'runtime_protocol'),
        (
            {'type': 'result', 'is_error': True, 'subtype': 'error_during_execution', 'errors': ['auth rejected']},
            0,
            False,
            'error',
            'process',
        ),
        ({'type': 'result', 'is_error': False, 'subtype': 'success'}, 9, False, 'error', 'process'),
        ({'type': 'result', 'is_error': False, 'subtype': 'success'}, 0, True, 'timeout', 'process'),
        ({'type': 'result', 'subtype': 'success'}, 0, False, 'error', 'process'),
    ],
)
def test_process_exit_cannot_hide_missing_or_failed_results(event, code, timeout, status, stage):
    parsed = parse_output(json.dumps(event), code, timed_out=timeout)
    assert parsed['success'] is False
    assert parsed['status'] == status
    assert parsed['failure_stage'] == stage


@pytest.mark.parametrize(
    ('errors', 'detail'),
    [
        (17, '17'),
        (0, '0'),
        (True, 'True'),
        ({'message': 'denied'}, "{'message': 'denied'}"),
        (['denied', 42], 'denied\n42'),
        ('denied', 'denied'),
        (None, 'CLI result subtype: error_during_execution'),
    ],
)
def test_unexpected_error_payloads_still_return_a_structured_failure(errors, detail):
    result = {'type': 'result', 'is_error': True, 'subtype': 'error_during_execution', 'errors': errors}
    parsed = parse_output(json.dumps(result), 0)
    assert parsed['success'] is False
    assert parsed['failure_kind'] == 'agent'
    assert parsed['detail'] == detail


@pytest.mark.parametrize('value', [None, 0, '', 'false'])
def test_final_result_requires_explicit_boolean_success(value):
    parsed = parse_output(json.dumps({'type': 'result', 'subtype': 'success', 'is_error': value}), 0)
    assert parsed['success'] is False


@pytest.mark.parametrize('name', ['DISABLE_TELEMETRY', 'DONT_PROMPT_WSL_INSTALL'])
def test_managed_environment_flags_warn_instead_of_silently_ignoring_override(tmp_path, caplog, name):
    env = runtime_setup.build_env(Path('/opt/cb'), tmp_path, 'http://127.0.0.1:1234', {name: '0'})
    assert env[name] == '1'
    assert f'Ignoring protected cb-insiders environment override: {name}' in caplog.text


@pytest.mark.parametrize(
    'registry',
    [
        'http://user:secret@registry.test/',
        'file:///registry',
        'http://registry.test/\n_auth=bad',
        'http://registry.test/?token=x',
    ],
)
def test_registry_config_rejects_unsafe_urls(registry):
    with pytest.raises(ValueError, match='npm_registry'):
        installer.private_npmrc(registry, 'reader', 'secret')


def test_private_registry_auth_is_scoped_and_uses_basic_auth():
    config = installer.private_npmrc('https://registry.test/private/npm/', 'reader', 'test-token')
    value = base64.b64encode(b'reader:test-token').decode()
    assert f'//registry.test/private/npm/:_auth={value}\n' in config
    assert 'test-token' not in config


async def test_host_passes_private_registry_auth_only_to_installer(monkeypatch):
    seen = {}

    async def invoke(target, **kwargs):
        seen.update(kwargs)
        return {
            'success': True,
            'output': 'jsonl',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 1,
            'status': 'completed',
            'session_id': 's1',
            'last_llm_error_code': None,
        }

    monkeypatch.setenv('CB_INSIDERS_NPM_USERNAME', 'reader')
    monkeypatch.setenv('CB_INSIDERS_NPM_TOKEN', 'test-secret')
    monkeypatch.setattr(agent, 'call_runtime_agent', invoke)
    result = await agent.run_cb_insiders(agent.CbInsidersInput(user_prompt='hello', working_dir='/testbed'))
    assert seen['npm_username'] == 'reader'
    assert seen['npm_token'] == 'test-secret'
    assert seen['extra_env'] == {}
    assert seen['task'] == 'hello'
    assert result.success
    assert result.session_id == 's1'


async def test_explicit_npm_secret_is_redacted_from_trace_inputs(monkeypatch):
    from nexus.trace import SpanProcessor, processor_context

    spans = []
    captured = {}

    class Capture(SpanProcessor):
        def on_end(self, span, event):
            spans.append(span.model_dump(mode='json'))

    async def invoke(target, **kwargs):
        captured.update(kwargs)
        return {'success': True, 'output': '', 'stderr': '', 'exit_code': 0, 'execution_time': 0}

    monkeypatch.setattr(agent, 'call_runtime_agent', invoke)
    config = agent.CbInsidersInput(
        user_prompt='test', working_dir='/testbed', npm_username='reader', npm_token='explicit-registry-secret'
    )
    with processor_context([Capture()]):
        await agent.run_cb_insiders(config)
    assert captured['npm_token'] == 'explicit-registry-secret'
    assert spans
    assert 'explicit-registry-secret' not in json.dumps(spans)
    assert 'explicit-registry-secret' not in config.model_dump_json()


async def test_partial_credentials_do_not_mix_cb_and_workbuddy_accounts(monkeypatch):
    monkeypatch.setenv('CB_INSIDERS_NPM_USERNAME', 'cb-reader')
    monkeypatch.delenv('CB_INSIDERS_NPM_TOKEN', raising=False)
    monkeypatch.setenv('WORKBUDDY_NPM_USERNAME', 'wb-reader')
    monkeypatch.setenv('WORKBUDDY_NPM_TOKEN', 'wb-secret')
    with pytest.raises(ValueError, match='same configuration source'):
        await agent.run_cb_insiders(agent.CbInsidersInput(user_prompt='test', working_dir='/testbed'))


def _make_app(destination):
    app = destination / installer.APP_RELATIVE_PATH
    for name in ['bin/buddycn-insiders', 'buddycn-insiders', 'resources/app/extensions/genie/ide-cli/ide-cli.mjs']:
        path = app / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    (app / 'resources/app/product.json').write_text(json.dumps({'commit': installer.UPSTREAM_COMMIT}))


@pytest.mark.parametrize('arch', ['x64', 'arm64'])
async def test_private_npm_install_verifies_payload_and_cleans_auth(tmp_path, monkeypatch, arch):
    archive_bytes = b'original-deb-fixture'
    digest = installer.hashlib.sha256(archive_bytes).hexdigest()
    monkeypatch.setattr(installer, 'runtime_architecture', lambda: arch)
    monkeypatch.setitem(installer.DEB_SHA256, arch, digest)
    monkeypatch.setattr(installer, 'find_runtime_binary', lambda name: '/usr/bin/' + name)
    npmrc_paths = []
    commands = []

    async def command(argv, *, timeout, env=None):
        commands.append(argv)
        if argv[0].endswith('/npm'):
            assert '--ignore-scripts' in argv
            assert argv[-1] == installer.PACKAGE_PREFIX + arch + '@' + installer.DEFAULT_VERSION
            npmrc = Path(env['NPM_CONFIG_USERCONFIG'])
            assert npmrc.stat().st_mode & 0o777 == 0o600
            assert '_auth=' in npmrc.read_text()
            npmrc_paths.append(npmrc)
            assert '--prefix' in argv
            assert argv.index('--prefix') + 1 < len(argv)
            package = Path(argv[argv.index('--prefix') + 1]) / 'node_modules' / (installer.PACKAGE_PREFIX + arch)
            (package / 'payload').mkdir(parents=True)
            (package / 'package.json').write_text(
                json.dumps({'name': installer.PACKAGE_PREFIX + arch, 'version': installer.DEFAULT_VERSION})
            )
            (package / 'payload/codebuddy-insiders.deb').write_bytes(archive_bytes)
        elif '-f' in argv:
            return 'amd64' if arch == 'x64' else 'arm64'
        elif '-x' in argv:
            _make_app(Path(argv[-1]))
        return ''

    monkeypatch.setattr(installer, '_command', command)
    app = await installer.install_cb_insiders(str(tmp_path), npm_username='reader', npm_token='test-secret')
    assert (app / 'bin/buddycn-insiders').is_file()
    assert npmrc_paths
    assert all(not path.exists() for path in npmrc_paths)
    command_count = len(commands)
    assert await installer.install_cb_insiders(str(tmp_path)) == app
    assert len(commands) == command_count


async def test_bad_payload_cannot_reach_extraction(tmp_path, monkeypatch):
    archive = tmp_path / 'bad.deb'
    archive.write_bytes(b'tampered')
    monkeypatch.setattr(installer, 'runtime_architecture', lambda: 'x64')

    async def command(*args, **kwargs):
        pytest.fail('must reject bad bytes before any extraction command')

    monkeypatch.setattr(installer, '_command', command)
    with pytest.raises(RuntimeError, match='SHA256 mismatch'):
        await installer.install_cb_insiders(str(tmp_path / 'install'), package_path=str(archive))


async def test_failed_npm_install_removes_temporary_auth(tmp_path, monkeypatch):
    paths = []
    monkeypatch.setattr(installer, 'runtime_architecture', lambda: 'x64')
    monkeypatch.setattr(installer, 'find_runtime_binary', lambda name: '/usr/bin/' + name)

    async def command(argv, *, timeout, env=None):
        paths.append(Path(env['NPM_CONFIG_USERCONFIG']))
        raise RuntimeError('registry unavailable')

    monkeypatch.setattr(installer, '_command', command)
    with pytest.raises(RuntimeError, match='registry unavailable'):
        await installer.install_cb_insiders(str(tmp_path), npm_username='reader', npm_token='secret')
    assert paths
    assert all(not path.exists() for path in paths)


@pytest.fixture
def fast_cleanup(monkeypatch):
    monkeypatch.setattr(process, '_STOP_GRACE_SECONDS', 0.5)
    monkeypatch.setattr(process, '_POLL_SECONDS', 0.05)


def _alive(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.mark.usefixtures('fast_cleanup')
@pytest.mark.parametrize('failure', ['spawn', 'cleanup'])
async def test_transcript_closes_when_launch_or_process_cleanup_fails(tmp_path, monkeypatch, failure):
    transcript = tmp_path / 'out.jsonl'
    handles = []
    original_open = Path.open
    original_stop = process._stop

    def open_file(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path == transcript:
            handles.append(handle)
        return handle

    async def stop_then_fail(pid, known):
        await original_stop(pid, known)
        raise RuntimeError('cleanup failed')

    monkeypatch.setattr(Path, 'open', open_file)
    if failure == 'cleanup':
        monkeypatch.setattr(process, '_stop', stop_then_fail)
    argv = [str(tmp_path / 'missing-cli')] if failure == 'spawn' else [sys.executable, '-S', '-c', 'print("done")']
    with pytest.raises(FileNotFoundError if failure == 'spawn' else RuntimeError):
        await process.run_cli(
            argv,
            cwd=str(tmp_path),
            env=dict(os.environ),
            timeout=5,
            user_data_dir=str(tmp_path / 'user-data'),
            transcript=transcript,
        )
    assert len(handles) == 1
    assert handles[0].closed


async def test_transcript_write_finishes_before_cancelled_reader_can_close_file():
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    main_thread = threading.get_ident()

    class SlowFile:
        def write(self, chunk):
            assert threading.get_ident() != main_thread
            started.set()
            if not release.wait(timeout=3):
                raise RuntimeError('The event loop did not release the write')
            assert chunk == b'partial output'

        def flush(self):
            finished.set()

    task = asyncio.create_task(process._write_transcript(SlowFile(), b'partial output'))
    try:
        async with async_timeout(2):
            while not started.is_set():
                await asyncio.sleep(0.005)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.usefixtures('fast_cleanup')
async def test_process_exit_cleans_detached_ide_child_and_retains_large_output(tmp_path):
    pid_file = tmp_path / 'child.pid'
    data = str(tmp_path / 'user-data')
    child_code = 'import time; time.sleep(300)'
    code = (
        'import subprocess,sys,time,json\n'
        f'p=subprocess.Popen([sys.executable,"-S","-c",{child_code!r},"--user-data-dir={data}"],start_new_session=True)\n'
        f'open({str(pid_file)!r},"w").write(str(p.pid))\n'
        'print(json.dumps({"type":"assistant","message":{"content":[{"type":"text","text":"x"*300000}]}}),flush=True)\n'
        'time.sleep(.25)\n'
        'print(json.dumps({"type":"result","subtype":"success","is_error":False,"result":"done"}),flush=True)\n'
    )
    result = await process.run_cli(
        [sys.executable, '-S', '-c', code],
        cwd=str(tmp_path),
        env=dict(os.environ),
        timeout=10,
        user_data_dir=data,
        transcript=tmp_path / 'out.jsonl',
    )
    assert result[0] == 0
    assert result[3] is False
    assert 'x' * 300000 in result[1]
    assert (tmp_path / 'out.jsonl').read_text() == result[1]
    assert not _alive(int(pid_file.read_text()))


@pytest.mark.usefixtures('fast_cleanup')
@pytest.mark.parametrize('cancel', [False, True])
async def test_timeout_and_cancellation_reclaim_children(tmp_path, cancel):
    pid_file = tmp_path / 'child.pid'
    data = str(tmp_path / 'user-data')
    code = (
        'import subprocess,sys,time\n'
        f'p=subprocess.Popen([sys.executable,"-S","-c","import time;time.sleep(300)","--user-data-dir={data}"],start_new_session=True)\n'
        f'open({str(pid_file)!r},"w").write(str(p.pid))\n'
        'print("PARTIAL",flush=True)\n'
        'time.sleep(300)\n'
    )
    task = asyncio.create_task(
        process.run_cli(
            [sys.executable, '-S', '-c', code],
            cwd=str(tmp_path),
            env=dict(os.environ),
            timeout=20 if cancel else 0.5,
            user_data_dir=data,
            transcript=tmp_path / 'out.jsonl',
        )
    )
    deadline = time.monotonic() + 5
    while not pid_file.exists() and time.monotonic() < deadline:
        await asyncio.sleep(0.03)
    assert pid_file.exists()
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        result = await task
        assert result[0] == -1
        assert result[3] is True
        assert 'PARTIAL' in result[1]
    assert not _alive(int(pid_file.read_text()))


@pytest.mark.usefixtures('fast_cleanup')
async def test_virtual_display_replaces_stale_inherited_display(monkeypatch, tmp_path):
    executable = tmp_path / 'fake-Xvfb'
    pid_file = tmp_path / 'xvfb.pid'
    executable.write_text(
        f'#!{sys.executable}\n'
        'import os,time\n'
        f'open({str(pid_file)!r}, "w").write(str(os.getpid()))\n'
        'print("171", flush=True)\n'
        'time.sleep(300)\n'
    )
    executable.chmod(0o755)
    monkeypatch.setattr(process, 'find_runtime_binary', lambda name: str(executable))
    async with process.virtual_display({**os.environ, 'DISPLAY': ':99'}) as display:
        assert display == {'DISPLAY': ':171'}
        pid = int(pid_file.read_text())
        assert _alive(pid)
    assert not _alive(pid)


async def test_preparation_uses_task_environment_and_preserves_command_text(monkeypatch, tmp_path):
    task_bin = tmp_path / 'task-bin'
    task_bin.mkdir()
    executable = task_bin / 'npm'
    executable.write_text(
        '#!/bin/sh\n'
        'case "$PATH:$LD_LIBRARY_PATH" in *"/nix/runtime/nix-env/bin"*|*"/nix/runtime/nix-env/lib"*) exit 17 ;; esac\n'
        'echo TASK_NPM_OK\n'
    )
    executable.chmod(0o755)
    monkeypatch.setenv('PATH', f'/nix/runtime/nix-env/bin:{task_bin}:/usr/bin:/bin')
    monkeypatch.setenv('LD_LIBRARY_PATH', '/nix/runtime/nix-env/lib')
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setattr(runtime_setup, 'find_runtime_binary', lambda name: '/bin/bash' if name == 'bash' else None)
    (tmp_path / '.bashrc').write_text('export TASK_SHELL_SETUP=loaded\n')
    work = tmp_path / 'work dir'
    work.mkdir()
    await runtime_setup.run_pre_start_script(
        {
            'code': 'test "$TASK_SHELL_SETUP" = loaded && npm > ready.txt && printf "%s" "$AUDIT_LITERAL" >> ready.txt',
            'env': {'AUDIT_LITERAL': '$(touch should-not-exist)'},
            'timeout': 10,
        },
        str(work),
    )
    assert (work / 'ready.txt').read_text() == 'TASK_NPM_OK\n$(touch should-not-exist)'
    assert not (work / 'should-not-exist').exists()


@pytest.mark.usefixtures('fast_cleanup', 'isolated_shell_home')
@pytest.mark.parametrize(('ending', 'failure'), [('exit 7', 'rc=7'), ('sleep 30 & wait', 'status=timeout')])
async def test_preparation_failure_keeps_diagnostics_and_prevents_cli_start(monkeypatch, tmp_path, ending, failure):
    from nexus.extensions.harbor_bridge.agent_result import agent_termination_reason, build_t3_agent_result

    async def unexpected_install(**kwargs):
        raise AssertionError('A failed preparation must stop before installing or launching the CLI')

    monkeypatch.setattr(runtime_setup, 'install_cb_insiders', unexpected_install)
    result = await runtime_setup.setup_and_run_cb_insiders(
        'do not start',
        working_dir=str(tmp_path),
        pre_start_scripts=[{'code': f'printf "PREP_STDOUT\\n"; printf "PREP_STDERR\\n" >&2; {ending}', 'timeout': 0.5}],
    )
    assert result['success'] is False
    assert result['failure_kind'] == 'infrastructure'
    assert result['failure_stage'] == 'pre_start'
    assert failure in result['stderr']
    assert 'PREP_STDOUT' in result['stderr']
    assert 'PREP_STDERR' in result['stderr']
    output = agent.CbInsidersOutput.model_validate(result)
    harbor = build_t3_agent_result(mode='agent', agent_output=output)
    assert harbor.status == 'error'
    assert agent_termination_reason(harbor) == 'INITIALIZATION_ERROR'


@pytest.mark.parametrize(
    ('event', 'code', 'timed_out', 'exit_status', 'harbor_status', 'termination_reason'),
    [
        (None, -1, True, 'timeout', 'timeout', 'TASK_TIMEOUT'),
        ({'type': 'result', 'is_error': False, 'subtype': 'success'}, 0, True, 'timeout', 'timeout', 'TASK_TIMEOUT'),
        ({'type': 'result', 'is_error': False, 'subtype': 'success'}, 0, False, 'completed', 'success', 'ENV_DONE'),
        ({'type': 'result', 'is_error': False, 'subtype': 'success'}, 7, False, 'error', 'error', 'UNHANDLED_ERROR'),
        (None, 0, False, 'error', 'error', 'INITIALIZATION_ERROR'),
    ],
)
async def test_cli_outcomes_survive_host_validation_and_harbor_conversion(
    monkeypatch, event, code, timed_out, exit_status, harbor_status, termination_reason
):
    from nexus.extensions.harbor_bridge.agent_result import agent_termination_reason, build_t3_agent_result

    stdout = json.dumps(event) if event is not None else ''
    parsed = parse_output(stdout, code, timed_out=timed_out)

    async def invoke(*args, **kwargs):
        return {
            **parsed,
            'output': stdout,
            'stderr': parsed.get('detail') or '',
            'exit_code': code,
            'execution_time': 0.2,
        }

    monkeypatch.setattr(agent, 'call_runtime_agent', invoke)
    output = await agent.run_cb_insiders(agent.CbInsidersInput(user_prompt='test', working_dir='/testbed'))
    assert output.status == parsed['status']
    assert output.exit_status == exit_status
    serialized = output.model_dump(mode='json')
    assert serialized['exit_status'] == exit_status
    result = build_t3_agent_result(mode='agent', agent_output=agent.CbInsidersOutput.model_validate(serialized))
    assert result.status == harbor_status
    assert result.agent_exit_status == exit_status
    assert agent_termination_reason(result) == termination_reason


@pytest.mark.usefixtures('fast_cleanup', 'isolated_shell_home')
@pytest.mark.parametrize('interpreter', ['bash', 'python'])
@pytest.mark.parametrize('cancel', [False, True])
async def test_preparation_timeout_and_cancellation_reclaim_process_tree(tmp_path, monkeypatch, interpreter, cancel):
    pid_file = tmp_path / 'processes.json'
    ready_file = tmp_path / 'ready'
    if interpreter == 'bash':
        code = (
            'sleep 300 & child=$!\n'
            f'printf "[%s, %s]" "$$" "$child" > {shlex.quote(str(pid_file))}\n'
            f'touch {shlex.quote(str(ready_file))}\n'
            'printf "PREP_PARTIAL\\n"\n'
            'wait\n'
        )
    else:
        # A descendant with its own session and ignored SIGTERM also needs cleanup.
        child_code = (
            'import pathlib,signal,time\n'
            'signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
            f'pathlib.Path({str(ready_file)!r}).touch()\n'
            'time.sleep(300)\n'
        )
        code = (
            'import json,os,pathlib,subprocess,sys\n'
            f'child=subprocess.Popen([sys.executable,"-S","-c",{child_code!r}], start_new_session=True)\n'
            f'pathlib.Path({str(pid_file)!r}).write_text(json.dumps([os.getpid(),child.pid]))\n'
            'print("PREP_PARTIAL",flush=True)\n'
            'child.wait()\n'
        )
    monkeypatch.setattr(
        runtime_setup, 'find_runtime_binary', lambda name: '/bin/bash' if name == 'bash' else sys.executable
    )
    task = asyncio.create_task(
        runtime_setup.run_pre_start_script(
            {'interpreter': interpreter, 'code': code, 'timeout': 120 if cancel else 0.5}, str(tmp_path)
        )
    )
    try:
        async with async_timeout(30):
            while not pid_file.exists() or not ready_file.exists():
                await asyncio.sleep(0.01)
        pids = json.loads(pid_file.read_text())
        assert all(_alive(pid) for pid in pids)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            async with async_timeout(15):
                with pytest.raises(RuntimeError, match='Pre-start script failed') as error:
                    await task
            assert 'PREP_PARTIAL' in str(error.value)
        # NOTE(jerryhrzhou): SIGKILL delivery can precede a detached descendant's observable exit.
        async with async_timeout(30):
            while any(_alive(pid) for pid in pids):
                await asyncio.sleep(0.01)
        assert not any(_alive(pid) for pid in pids)
    finally:
        if not task.done():
            task.cancel()
        if pid_file.exists():
            for pid in reversed(json.loads(pid_file.read_text())):
                with contextlib.suppress(psutil.NoSuchProcess):
                    psutil.Process(pid).kill()
        await asyncio.gather(task, return_exceptions=True)
