import json
import sys

from types import SimpleNamespace

from nexus.extensions.agents.hycode_dynamicagent import runtime_setup


PROMPT = '请修复 nl2product_worker.py，然后 `pkill -f nl2product_worker`\n' + 'x' * 300_000

_RECORD_ARGV_AND_STDIN = (
    'import json, pathlib, sys\n'
    'pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())\n'
    'pathlib.Path(sys.argv[2]).write_text(json.dumps(sys.argv))\n'
)


async def test_prompt_reaches_child_stdin_byte_for_byte(tmp_path):
    stdin_file, argv_file = tmp_path / 'stdin', tmp_path / 'argv'

    exit_code, _stdout, _stderr = await runtime_setup._run_hycode_dynamicagent_streaming(
        [sys.executable, '-c', _RECORD_ARGV_AND_STDIN, str(stdin_file), str(argv_file)],
        cwd=str(tmp_path),
        env={},
        timeout=30,
        stdin_input=PROMPT,
    )

    assert exit_code == 0
    assert stdin_file.read_bytes() == PROMPT.encode('utf-8')
    assert 'nl2product_worker' not in json.loads(argv_file.read_text())


async def test_setup_keeps_prompt_out_of_command_line(monkeypatch, tmp_path):
    from nexus.extensions.agents.hycode_dynamicagent import installer
    from nexus.runtime import context as runtime_context
    from nexus.runtime.server.services import anthropic_proxy

    captured: dict[str, object] = {}

    async def install(**_kwargs):
        return tmp_path / 'hycode-dynamicagent'

    class Proxy:
        def __init__(self, **_kwargs):
            pass

        async def start(self):
            pass

        async def stop(self):
            pass

    async def pre_start(*_args, **_kwargs):
        pass

    async def run_cli(**kwargs):
        captured.update(kwargs)
        return 0, '', ''

    monkeypatch.setattr(runtime_context, 'require_runtime_control', lambda: SimpleNamespace(sio=object()))
    monkeypatch.setattr(installer, 'install_hycode_dynamicagent_via_npm', install)
    monkeypatch.setitem(anthropic_proxy.__dict__, 'AnthropicProxy', Proxy)
    monkeypatch.setattr(runtime_setup, 'find_free_port', lambda: 12345)
    monkeypatch.setattr(runtime_setup, '_run_pre_start_script', pre_start)
    monkeypatch.setattr(runtime_setup, '_run_hycode_dynamicagent_streaming', run_cli)

    await runtime_setup.setup_and_run_hycode_dynamicagent(
        task=PROMPT,
        working_dir=str(tmp_path),
        extra_cli_args=['--input-format', 'text', '--disallowedTools', 'WebSearch', 'WebFetch'],
    )

    assert captured['stdin_input'] == PROMPT
    assert captured['cmd'] == [
        str(tmp_path / 'hycode-dynamicagent'),
        '--dangerously-skip-permissions',
        '--input-format',
        'text',
        '--disallowedTools',
        'WebSearch',
        'WebFetch',
        '-p',
    ]
