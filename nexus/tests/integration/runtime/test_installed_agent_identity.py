"""Offline identity integration tests; Docker tests use an already available image.

Run: uv run pytest tests/integration/runtime/test_installed_agent_identity.py -q
No users are created on the host. The Compose fixture creates alice only inside
its disposable container, and never downloads an image or npm package.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid

from pathlib import Path
from types import SimpleNamespace

import pytest

from aiohttp import web

from nexus.extensions.agents.claude_code.installer import install_claude_code_via_npm
from nexus.extensions.agents.claude_code.runtime_setup import setup_and_run_claude_code
from nexus.extensions.agents.codex.installer import install_codex_cli
from nexus.extensions.agents.codex.runtime_setup import setup_and_run_codex
from nexus.extensions.agents.installed_common.execution import (
    AgentExecutionTransport,
    ComposeExecutionTarget,
    DirectExecutionTarget,
)
from nexus.runtime.context import execution_user


pwd = pytest.importorskip('pwd')
pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[3]
IMAGE = 'python:3.12-slim-bookworm'


@pytest.fixture
def public_directory():
    directory = Path(tempfile.mkdtemp(prefix='nxidentity-'))
    directory.chmod(0o777)
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


@pytest.mark.skipif(os.geteuid() != 0, reason='Direct user switching requires a root test process')
async def test_direct_agent_files_and_process_use_account_while_worker_stays_root(public_directory):
    account = pwd.getpwnam('nobody')
    transport = AgentExecutionTransport(DirectExecutionTarget(user=account.pw_uid))
    await transport.initialize()
    path = public_directory / 'agent/config.json'
    with execution_user('does-not-exist'):
        await transport.write_text(path, '{"private":true}')
        result = await transport.exec_as_agent(
            ['/bin/sh', '-c', 'id -u; printf "%s\n" "$HOME"; /bin/sh -c "id -u"'],
            check=True,
        )
        root = await transport.exec_as_root(['/bin/sh', '-c', 'id -u'], check=True)
    assert result.stdout.splitlines() == [str(account.pw_uid), account.pw_dir, str(account.pw_uid)]
    assert root.stdout.strip() == '0'
    assert os.geteuid() == 0
    assert path.stat().st_uid == account.pw_uid
    assert path.stat().st_mode & 0o777 == 0o600
    assert await transport.read_text(path) == '{"private":true}'
    async with transport.prepare(
        ['/bin/sh', '-c', 'id -u'],
        working_dir=str(public_directory),
        invocation_id='direct',
        env={},
    ) as prepared:
        process = await asyncio.create_subprocess_exec(
            *prepared.command,
            cwd=prepared.cwd,
            env={**os.environ, **prepared.env},
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
    assert stdout.decode().strip() == str(account.pw_uid)
    await transport.remove(path)
    assert not path.exists()


@pytest.mark.skipif(os.geteuid() != 0, reason='Direct user switching requires a root test process')
@pytest.mark.parametrize('user', ['nobody', 'root'])
async def test_direct_bash_env_runs_only_after_identity_switch(public_directory, user):
    account = pwd.getpwnam(user)
    transport = AgentExecutionTransport(DirectExecutionTarget(user=user))
    await transport.initialize()
    startup = public_directory / 'startup.sh'
    marker = public_directory / 'startup-uids'
    startup.write_text(f'id -u >> {marker}\nexport AGENT_TOOLCHAIN=ready\n')
    startup.chmod(0o644)
    result = await transport.exec_as_agent(
        ['/bin/bash', '-c', 'printf "%s" "$AGENT_TOOLCHAIN"'],
        env={'BASH_ENV': str(startup)},
        check=True,
    )
    assert result.stdout == 'ready'
    assert marker.read_text().splitlines() == [str(account.pw_uid)]
    await transport.exec_as_root(['/bin/bash', '-c', 'true'], env={'BASH_ENV': str(startup)}, check=True)
    assert marker.read_text().splitlines() == [str(account.pw_uid)]
    marker.unlink()
    async with transport.prepare(
        ['/bin/bash', '-c', 'printf "%s" "$AGENT_TOOLCHAIN"'],
        working_dir=str(public_directory),
        invocation_id='startup',
        env={'BASH_ENV': str(startup)},
    ) as prepared:
        process = await asyncio.create_subprocess_exec(
            *prepared.command,
            cwd=prepared.cwd,
            env={**os.environ, **prepared.env},
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        assert process.returncode == 0
    assert stdout == b'ready'
    assert marker.read_text().splitlines() == [str(account.pw_uid)]


@pytest.fixture
def compose_target(public_directory, request):
    if getattr(request, 'param', None) == 'readonly':
        yield request.getfixturevalue('readonly_compose_target')
        return
    if shutil.which('docker') is None:
        pytest.skip('Docker is not installed')
    if subprocess.run(['docker', 'compose', 'version'], capture_output=True).returncode:
        pytest.skip('Docker Compose plugin is not installed')
    inspected = subprocess.run(['docker', 'image', 'inspect', IMAGE], capture_output=True)
    if inspected.returncode:
        pytest.skip(f'Offline test requires local image {IMAGE}')
    project = 'nxidentity-' + uuid.uuid4().hex[:10]
    bridge = public_directory / 'bridge.py'
    source = (ROOT / 'deployment/docker/runtime-bin/nexus-socket-bridge').read_text()
    bridge.write_text('#!/usr/bin/env python3\n' + source.split('\n', 1)[1])
    bridge.chmod(0o755)
    compose = public_directory / 'compose.json'
    bootstrap = (
        'useradd -m -u 1000 alice && '
        'mkdir -p /nix/runtime/nix-env/bin /workspace && '
        'ln -s /usr/local/bin/python3 /nix/runtime/nix-env/bin/python3 && '
        'ln -s /bin/bash /nix/runtime/nix-env/bin/bash && '
        'chown alice:alice /workspace && sleep infinity'
    )
    compose.write_text(
        json.dumps({
            'services': {
                'main': {
                    'image': IMAGE,
                    'pull_policy': 'never',
                    'command': ['/bin/sh', '-c', bootstrap],
                    'volumes': [f'{public_directory}:/transport'],
                }
            }
        })
    )
    command = ['docker', 'compose', '-p', project, '-f', str(compose)]
    try:
        subprocess.run([*command, 'up', '-d', '--pull', 'never'], check=True, capture_output=True)
        for _ in range(60):
            result = subprocess.run(
                [*command, 'exec', '-T', 'main', 'test', '-x', '/nix/runtime/nix-env/bin/bash'], capture_output=True
            )
            if result.returncode == 0:
                break
            import time

            time.sleep(0.1)
        else:
            pytest.fail('Container account did not initialize')
        yield ComposeExecutionTarget(
            project_name=project,
            compose_files=[str(compose)],
            service='main',
            transport_host_dir=str(public_directory),
            transport_container_dir='/transport',
            socket_bridge_host_path=str(bridge),
            socket_bridge_container_path='/transport/bridge.py',
            user='alice',
        )
    finally:
        subprocess.run([*command, 'down', '--volumes', '--remove-orphans'], check=True, capture_output=True)


async def install_fake_npm(transport, directory='/usr/local/bin'):
    source = """#!/usr/local/bin/python3
import json, os, pathlib, sys
args = sys.argv
prefix = pathlib.Path(args[args.index('--prefix') + 1])
package = '@openai/codex' if any(s.startswith('@openai/codex') for s in args) else '@anthropic-ai/claude-code'
version = '0.114.0' if package.startswith('@openai') else ('2.1.185' if any('@2.1.185' in s for s in args) else '2.1.112')
directory = prefix / 'node_modules' / package
directory.mkdir(parents=True, exist_ok=True)
name = 'codex' if package.startswith('@openai') else 'claude'
entry = directory / 'cli.js'
entry.write_text('#!/usr/local/bin/python3\\nprint("offline cli")\\n')
entry.chmod(0o755)
(directory / 'package.json').write_text(json.dumps({'version': version, 'bin': {name: 'cli.js'}}))
if version == '2.1.185':
    native_package = next(s.rsplit('@', 1)[0] for s in args if s.startswith('@anthropic-ai/claude-code-linux-'))
    native = prefix / 'node_modules' / native_package / 'claude'
    native.parent.mkdir(parents=True, exist_ok=True)
    native.write_text('#!/usr/local/bin/python3\\nprint("native fixture")\\n')
    native.chmod(0o755)
link = prefix / 'node_modules/.bin' / name
link.parent.mkdir(parents=True, exist_ok=True)
link.symlink_to(entry)
(prefix / 'install-identity.json').write_text(json.dumps({'uid': os.getuid(), 'home': os.environ['HOME']}))
"""
    payload = json.dumps(source)
    await transport.exec_as_root(
        [
            '/usr/local/bin/python3',
            '-c',
            f'import pathlib; p=pathlib.Path({directory!r}) / "npm"; p.write_text({payload}); p.chmod(0o755)',
        ],
        check=True,
    )
    await transport.exec_as_root(['ln', '-s', '/usr/local/bin/python3', f'{directory}/node'], check=True)


async def test_compose_installs_in_inner_account_and_reuses_only_its_install(compose_target):
    transport = AgentExecutionTransport(compose_target)
    await transport.initialize()
    home = transport.agent_env()['HOME']
    assert home.startswith('/transport/agent-1000/home-')
    await install_fake_npm(transport)
    for name, install, version in (
        ('claude-code', install_claude_code_via_npm, '2.1.112'),
        ('claude-code-native', install_claude_code_via_npm, '2.1.185'),
        ('codex-cli', install_codex_cli, '0.114.0'),
    ):
        directory = transport.install_dir(agent_name=name, default='/root/should-not-be-used')
        path = await install(directory, version=version, transport=transport)
        assert str(path).startswith('/transport/agent-1000/install/')
        identity = json.loads(await transport.read_text(Path(directory) / 'install-identity.json'))
        assert identity == {'uid': 1000, 'home': home}
        assert await install(directory, version=version, transport=transport) == path
        assert not (Path(compose_target.transport_host_dir) / 'install').exists()
    root = await transport.exec_as_root(['id', '-u'], check=True)
    assert root.stdout.strip() == '0'
    numeric = AgentExecutionTransport(compose_target.model_copy(update={'user': 1000}))
    await numeric.initialize()
    assert numeric.agent_env()['HOME'].startswith('/transport/agent-1000/home-')
    assert numeric.agent_env()['HOME'] != home
    assert numeric.install_dir(agent_name='claude-code', default='/unused') == transport.install_dir(
        agent_name='claude-code',
        default='/unused',
    )


_CLAUDE = """#!/usr/local/bin/python3
import json, os, pathlib, sys, urllib.request
if len(sys.argv) > 1 and sys.argv[1] == 'mcp':
    pathlib.Path('/workspace/mcp.json').write_text(sys.argv[-1])
else:
    sys.stdin.read()
    urllib.request.urlopen(os.environ['ANTHROPIC_BASE_URL'] + '/health').read()
    config = pathlib.Path(sys.argv[sys.argv.index('--mcp-config') + 1])
    print(json.dumps({'uid': os.getuid(), 'home': os.environ['HOME'], 'config_uid': config.stat().st_uid,
                      'config_mode': config.stat().st_mode & 511, 'private': json.loads(config.read_text())}))
"""
_CODEX = """#!/usr/local/bin/python3
import json, os, pathlib, re, sys, urllib.request
if sys.argv[1:] == ['--version']:
    print('codex-cli 0.158.0')
    sys.exit(0)
home = pathlib.Path(os.environ['CODEX_HOME'])
config = (home / 'config.toml').read_text()
url = re.search(r'base_url = "([^"]+)"', config).group(1)
urllib.request.urlopen(url + '/health').read()
(home / 'sessions').mkdir()
(home / 'sessions/session.jsonl').write_text('session')
(home / 'auth.json').write_text('secret')
(home / 'last_message.txt').write_text(json.dumps({'uid': os.getuid(), 'home': os.environ['HOME']}))
print('{"type":"turn.completed"}')
"""


@pytest.mark.parametrize('compose_target', ['writable', 'readonly'], indirect=True)
async def test_compose_cc_and_codex_setup_run_proxy_files_and_logs(compose_target, monkeypatch, public_directory):
    application = web.Application()

    async def serve(request):
        payload = {'/claude': _CLAUDE, '/codex': _CODEX}.get(request.path, 'ok')
        return web.Response(body=payload.encode())

    application.router.add_get('/{path:.*}', serve)
    server = web.AppRunner(application)
    await server.setup()
    site = web.TCPSite(server, '0.0.0.0', 0)
    await site.start()
    port = server.addresses[0][1]
    transport = AgentExecutionTransport(compose_target)
    uid = int((await transport.exec_as_agent(['id', '-u'], check=True)).stdout.strip())
    gateway = await transport.exec_as_root(
        ['/bin/sh', '-c', 'awk \'$2 == "00000000" {print $3}\' /proc/net/route'],
        check=True,
    )
    host = '.'.join(str(value) for value in bytes.fromhex(gateway.stdout.strip())[::-1])
    events = []

    class Proxy:
        async def start(self):
            events.append(('start', os.geteuid()))

        async def stop(self):
            events.append(('stop', os.geteuid()))

    monkeypatch.setattr('nexus.runtime.context.require_runtime_control', lambda: SimpleNamespace(sio=object()))
    monkeypatch.setattr('nexus.runtime.server.services.anthropic_proxy.AnthropicProxy', lambda **kwargs: Proxy())
    monkeypatch.setattr('nexus.runtime.server.services.responses_proxy.ResponsesProxy', lambda **kwargs: Proxy())
    monkeypatch.setattr('nexus.extensions.agents.claude_code.runtime_setup.find_free_port', lambda: port)
    monkeypatch.setattr('nexus.extensions.agents.codex.runtime_setup.find_free_port', lambda: port)
    monkeypatch.setattr('nexus.extensions.agents.codex.runtime_setup.get_current_span_id', lambda: 'identity-test')
    strict = public_directory / 'strict.json'
    strict.write_text('{"secret":"test"}')
    strict.chmod(0o600)
    try:
        cc = await setup_and_run_claude_code(
            task='offline',
            working_dir='/workspace',
            execution_target=compose_target.model_dump(),
            binary_url=f'http://{host}:{port}/claude',
            binary_checksum=hashlib.sha256(_CLAUDE.encode()).hexdigest(),
            uploaded_files=[str(strict)],
            extra_cli_args=['--mcp-config', str(strict)],
            config_file_patches=[{'path': '~/.claude/settings.json', 'content': {'test': True}}],
            pre_start_scripts=[{'interpreter': 'bash', 'code': 'id -u > /workspace/pre-start.uid'}],
            mcp_servers=[{'name': 'test', 'command': 'echo', 'args': ['ok']}],
        )
        assert cc['success'], cc
        result = json.loads(cc['output'])
        home = result['home']
        assert home.startswith(f'/transport/agent-{uid}/home-')
        assert result == {
            'uid': uid,
            'home': home,
            'config_uid': uid,
            'config_mode': 0o600,
            'private': {'secret': 'test'},
        }
        assert (await transport.read_text('/workspace/pre-start.uid')).strip() == str(uid)
        assert json.loads(await transport.read_text('/workspace/mcp.json'))['command'] == 'echo'
        logs = public_directory / 'logs'
        codex = await setup_and_run_codex(
            task='offline',
            working_dir='/workspace',
            execution_target=compose_target.model_dump(),
            binary_url=f'http://{host}:{port}/codex',
            logs_dir=str(logs),
        )
        assert codex['success'], codex
        codex_result = json.loads(codex['output'])
        assert codex_result['uid'] == uid
        assert codex_result['home'].startswith(f'/transport/agent-{uid}/home-')
        assert codex_result['home'] != home
        assert (logs / 'sessions/session.jsonl').read_text() == 'session'
        assert not (logs / 'auth.json').exists()
        assert events == [('start', 0), ('stop', 0), ('start', 0), ('stop', 0)]
        assert os.geteuid() == 0
    finally:
        await server.cleanup()


_PREINSTALLED_CODEX = """#!/bin/sh
if [ "$1" = "--version" ]; then
    printf 'codex-cli 0.158.0\\n'
else
    id -u
    printf '%s\\n' "$HOME"
fi
"""


@pytest.mark.parametrize('direct', [False, True])
async def test_preinstalled_codex_uses_local_or_direct_account_without_npm(
    public_directory, monkeypatch, isolated_shell_home, direct
):
    if direct and os.geteuid() != 0:
        pytest.skip('Direct user switching requires a root test process')
    tools = public_directory / 'bin'
    tools.mkdir(mode=0o755)
    binary = tools / 'codex'
    binary.write_text(_PREINSTALLED_CODEX)
    binary.chmod(0o755)
    npm = tools / 'npm'
    npm.write_text('#!/bin/sh\necho "unexpected npm install" >&2\nexit 99\n')
    npm.chmod(0o755)
    monkeypatch.setenv('PATH', f'{tools}:/usr/local/bin:/usr/bin:/bin')
    target = DirectExecutionTarget(user='nobody') if direct else None
    transport = AgentExecutionTransport(target)
    directory = public_directory / 'unused-install'
    assert await install_codex_cli(str(directory), version='0.158.0', transport=transport) == binary
    assert not directory.exists()
    result = await transport.exec_as_agent([str(binary), 'exec'], check=True)
    account = pwd.getpwnam('nobody') if direct else pwd.getpwuid(os.geteuid())
    home = account.pw_dir if direct else str(isolated_shell_home)
    assert result.stdout.splitlines() == [str(account.pw_uid), home]


@pytest.mark.parametrize('user', ['alice', 1000])
async def test_compose_reuses_inner_preinstalled_codex_as_agent_without_npm(compose_target, monkeypatch, user):
    from nexus.extensions.agents import installed_common

    monkeypatch.setattr(
        installed_common,
        'find_runtime_binary',
        lambda name: pytest.fail('Target Codex must not be looked up in the Worker'),
    )
    transport = AgentExecutionTransport(compose_target.model_copy(update={'user': user}))
    await transport.initialize()
    source = (
        'import pathlib; '
        f'p=pathlib.Path("/usr/local/bin/codex"); p.write_text({_PREINSTALLED_CODEX!r}); p.chmod(0o755); '
        'p=pathlib.Path("/usr/local/bin/npm"); '
        'p.write_text("#!/bin/sh\\necho unexpected-npm-install >&2\\nexit 99\\n"); p.chmod(0o755)'
    )
    await transport.exec_as_root(['/usr/local/bin/python3', '-c', source], check=True)
    directory = transport.install_dir(agent_name='codex-cli', default='/root/should-not-be-used')
    binary = await install_codex_cli(directory, version='0.158.0', transport=transport)
    assert binary == Path('/usr/local/bin/codex')
    assert not await transport.file('exists', directory)
    result = await transport.exec_as_agent([str(binary), 'exec'], check=True)
    assert result.stdout.splitlines() == ['1000', transport.agent_env()['HOME']]


_PREINSTALLED_CLAUDE = _PREINSTALLED_CODEX.replace('codex-cli 0.158.0', '2.1.283 (Claude Code)')


@pytest.mark.parametrize('direct', [False, True])
async def test_preinstalled_claude_uses_local_or_direct_account_without_npm(
    public_directory, monkeypatch, isolated_shell_home, direct
):
    if direct and os.geteuid() != 0:
        pytest.skip('Direct user switching requires a root test process')
    tools = public_directory / 'bin'
    tools.mkdir(mode=0o755)
    binary = tools / 'claude'
    binary.write_text(_PREINSTALLED_CLAUDE)
    binary.chmod(0o755)
    npm = tools / 'npm'
    npm.write_text('#!/bin/sh\necho unexpected-npm >&2\nexit 99\n')
    npm.chmod(0o755)
    monkeypatch.setenv('PATH', f'{tools}:/usr/local/bin:/usr/bin:/bin')
    transport = AgentExecutionTransport(DirectExecutionTarget(user='nobody') if direct else None)
    directory = public_directory / 'unused-install'
    assert await install_claude_code_via_npm(str(directory), transport=transport) == binary
    assert not directory.exists()
    result = await transport.exec_as_agent([str(binary), '--print'], check=True)
    account = pwd.getpwnam('nobody') if direct else pwd.getpwuid(os.geteuid())
    home = account.pw_dir if direct else str(isolated_shell_home)
    assert result.stdout.splitlines() == [str(account.pw_uid), home]


@pytest.mark.parametrize('user', ['alice', 1000])
async def test_compose_reuses_inner_preinstalled_claude_without_npm(compose_target, monkeypatch, user):
    from nexus.extensions.agents import installed_common

    monkeypatch.setattr(
        installed_common,
        'find_runtime_binary',
        lambda name: pytest.fail('Do not inspect the outer Worker for Claude'),
    )
    transport = AgentExecutionTransport(compose_target.model_copy(update={'user': user}))
    await transport.initialize()
    source = (
        'import pathlib; '
        f'p=pathlib.Path("/usr/local/bin/claude"); p.write_text({_PREINSTALLED_CLAUDE!r}); p.chmod(0o755); '
        'p=pathlib.Path("/usr/local/bin/npm"); '
        'p.write_text("#!/bin/sh\\necho unexpected-npm >&2\\nexit 99\\n"); p.chmod(0o755)'
    )
    await transport.exec_as_root(['/usr/local/bin/python3', '-c', source], check=True)
    directory = transport.install_dir(agent_name='claude-code', default='/root/do-not-use')
    binary = await install_claude_code_via_npm(directory, transport=transport)
    assert binary == Path('/usr/local/bin/claude')
    assert not await transport.file('exists', directory)
    result = await transport.exec_as_agent([str(binary), '--print'], check=True)
    assert result.stdout.splitlines() == ['1000', transport.agent_env()['HOME']]


@pytest.fixture
def readonly_compose_target(public_directory):
    if shutil.which('docker') is None:
        pytest.skip('Docker is not installed')
    if subprocess.run(['docker', 'compose', 'version'], capture_output=True).returncode:
        pytest.skip('Docker Compose plugin is not installed')
    if subprocess.run(['docker', 'image', 'inspect', IMAGE], capture_output=True).returncode:
        pytest.skip(f'Offline test requires local image {IMAGE}')
    project = 'nxreadonly-' + uuid.uuid4().hex[:10]
    tools = public_directory / 'tools'
    tools.mkdir(mode=0o755)
    (tools / 'python3').symlink_to('/usr/local/bin/python3')
    (tools / 'bash').symlink_to('/bin/bash')
    workspace = public_directory / 'workspace'
    workspace.mkdir()
    workspace.chmod(0o777)
    bridge = public_directory / 'bridge.py'
    source = (ROOT / 'deployment/docker/runtime-bin/nexus-socket-bridge').read_text()
    bridge.write_text('#!/usr/bin/env python3\n' + source.split('\n', 1)[1])
    bridge.chmod(0o755)
    compose = public_directory / 'readonly.json'
    compose.write_text(
        json.dumps({
            'services': {
                'main': {
                    'image': IMAGE,
                    'pull_policy': 'never',
                    'user': 'nobody',
                    'read_only': True,
                    'security_opt': ['no-new-privileges:true'],
                    'command': ['sleep', 'infinity'],
                    'working_dir': '/workspace',
                    'environment': {'PATH': '/transport/tools:/usr/local/bin:/usr/bin:/bin'},
                    'volumes': [
                        f'{public_directory}:/transport',
                        f'{tools}:/nix/runtime/nix-env/bin:ro',
                        f'{workspace}:/workspace',
                    ],
                }
            }
        })
    )
    command = ['docker', 'compose', '-p', project, '-f', str(compose)]
    try:
        subprocess.run([*command, 'up', '-d', '--pull', 'never'], check=True, capture_output=True)
        yield ComposeExecutionTarget(
            project_name=project,
            compose_files=[str(compose)],
            service='main',
            transport_host_dir=str(public_directory),
            transport_container_dir='/transport',
            socket_bridge_host_path=str(bridge),
            socket_bridge_container_path='/transport/bridge.py',
        )
    finally:
        subprocess.run([*command, 'down', '--volumes', '--remove-orphans'], check=True, capture_output=True)


@pytest.mark.parametrize('user', [None, 'nobody', 65534])
async def test_readonly_compose_installs_with_nonexistent_account_home(readonly_compose_target, user):
    target = readonly_compose_target.model_copy(update={'user': user})
    transport = AgentExecutionTransport(target)
    await transport.initialize()
    home = transport.agent_env()['HOME']
    assert home.startswith('/transport/agent-65534/home-')
    probe = await transport.exec_as_agent(
        [
            transport.python_binary,
            '-c',
            'import os, pwd; print(pwd.getpwuid(os.geteuid()).pw_dir); '
            'print(bool(os.statvfs("/").f_flag & os.ST_RDONLY))',
        ],
        check=True,
    )
    assert probe.stdout.splitlines() == ['/nonexistent', 'True']
    await install_fake_npm(transport, directory='/transport/tools')
    for name, install, version in (
        ('claude-code', install_claude_code_via_npm, '2.1.112'),
        ('claude-code-native', install_claude_code_via_npm, '2.1.185'),
        ('codex-cli', install_codex_cli, '0.114.0'),
    ):
        directory = transport.install_dir(agent_name=name, default='/tmp/nexus/agents/' + name)
        binary = await install(directory, version=version, transport=transport)
        identity = json.loads(await transport.read_text(Path(directory) / 'install-identity.json'))
        assert identity == {'uid': 65534, 'home': home}
        result = await transport.exec_as_agent([str(binary)], check=True)
        assert result.stdout.strip() in {'offline cli', 'native fixture'}
        assert await install(directory, version=version, transport=transport) == binary
    config = Path(home) / '.claude/settings.json'
    await transport.write_text(config, '{"private":true}')
    assert json.loads(await transport.read_text(config)) == {'private': True}
    info = await transport.file('exists', '/nonexistent')
    assert not info
    root = AgentExecutionTransport(target.model_copy(update={'user': 'root'}))
    await root.initialize()
    assert root.install_dir(agent_name='claude-code', default='/unused') != transport.install_dir(
        agent_name='claude-code',
        default='/unused',
    )
    assert root.agent_env()['HOME'].startswith('/transport/agent-0/home-')
