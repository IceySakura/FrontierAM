"""Sandbox-side functions for the labelled Claude Code print backend.

Invoked through ``runtime.invoke()``, so this runs inside the sandbox process and
its module-level state persists across calls within one container.

Lifecycle::

    1. setup_env()               install CC, deploy hooks -- once per container
    2. open_session()  * N       one labelled proxy per session
    3. run_turn()      * N       one CC process per turn
    4. write_answer()            inject an answer for a deferred AskUserQuestion
    5. close_session() / teardown_env()

Each session gets its own proxy, holding one label for the proxy's lifetime, which
is what lets concurrent turns keep their attribution apart. Installation, the
AskUserQuestion defer/resume hook, MCP registration and the CC command line follow
the legacy print runtime.
"""

import contextlib
import json
import logging
import os
import shlex
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

_claude_path: str | None = None

# What the container was already configured with, so a second call carrying
# different settings is reported rather than skipped.
_setup_signature: str | None = None

# session_id -> its own proxy. One per session, since a proxy's label is fixed
# when it starts.
_sessions: dict[str, '_SessionProxy'] = {}

_DEFAULT_INSTALL_DIR = '/tmp/claude-code'
_DEFAULT_CC_TURN_TIMEOUT_SECONDS = 7200
_HOOKS_DIR = '/tmp/claude-code/hooks'
_ANSWERS_DIR = '/tmp/claude-code/answers'
_HOOK_SCRIPT_NAME = 'ask_user_hook.py'

# Kept byte-identical to the legacy print runtime's hook: it encodes the
# defer/resume protocol the host side depends on (a missing answer file means
# "defer", a present one means "allow with this input"). Diverging here would
# silently change when the orchestrator gets asked for input.
_ASK_USER_HOOK_SCRIPT_TEMPLATE = '''\
#!{python_path}
"""PreToolUse hook — AskUserQuestion defer/resume."""
import json
import os
import sys
import time
from pathlib import Path

LOG_FILE = Path("/tmp/claude-code/hook.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(LOG_FILE, "a") as _f:
    _f.write(f"{{time.time():.3f}} hook invoked pid={{os.getpid()}}\\n")

ANSWERS_DIR = Path("/tmp/claude-code/answers")

payload = json.load(sys.stdin)
session_id = payload.get("session_id", "unknown")
tool_use_id = payload.get("tool_use_id", "unknown")
tool_name = payload.get("tool_name", "")
tool_input = payload.get("tool_input") or {{}}

with open(LOG_FILE, "a") as f:
    f.write(f"{{time.time():.3f}} tool={{tool_name}} session={{session_id}} tool_use_id={{tool_use_id}}\\n")


def respond(output):
    print(json.dumps({{"hookSpecificOutput": output}}, ensure_ascii=False))
    sys.exit(0)


if tool_name != "AskUserQuestion":
    respond({{
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow",
        "permissionDecisionReason": "Non-AskUserQuestion tool allowed.",
    }})

answer_file = ANSWERS_DIR / session_id / f"{{tool_use_id}}.json"
if not answer_file.exists():
    respond({{
        "hookEventName": "PreToolUse",
        "permissionDecision": "defer",
        "permissionDecisionReason": "AskUserQuestion deferred to orchestrator.",
    }})

with open(answer_file) as f:
    record = json.load(f)
answers = record.get("answers", record)
updated_input = dict(tool_input)
updated_input["answers"] = answers
respond({{
    "hookEventName": "PreToolUse",
    "permissionDecision": "allow",
    "permissionDecisionReason": "Answer injected by orchestrator.",
    "updatedInput": updated_input,
}})
'''


@dataclass
class _SessionProxy:
    """One session's proxy, its port, and the label it stamps."""

    proxy: Any
    port: int
    label: str


async def setup_env(
    *,
    working_dir: str = '/workspace',
    version: str = 'latest',
    npm_registry: str | None = None,
    binary_url: str | None = None,
    binary_checksum: str | None = None,
    mcp_servers: list[dict[str, Any]] | None = None,
    extra_env: dict[str, str] | None = None,
    config_file_patches: list[dict[str, Any]] | None = None,
    pre_start_scripts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Install Claude Code and deploy the hook. Idempotent, once per container.

    No proxy is started here -- proxies are per session, so they belong to
    :func:`open_session`. Everything else matches the legacy setup step,
    including the order: patches, then pre-start scripts, then hook, then MCP.

    A second call with the same settings is a no-op. One with different settings
    raises: everything below the install -- the patches, the scripts, the MCP
    servers -- would be skipped, leaving the container configured for whoever
    called first with nothing said.
    """
    global _claude_path, _setup_signature

    from nexus.extensions.agents.claude_code.installer import (
        install_claude_code_from_url,
        install_claude_code_via_npm,
    )
    from nexus.extensions.agents.claude_code.interactive.runtime import _apply_json_patch
    from nexus.extensions.agents.claude_code.runtime_setup import _run_pre_start_script
    from nexus.extensions.agents.installed_common import build_runtime_path

    signature = json.dumps(
        {
            'working_dir': working_dir,
            'version': version,
            'npm_registry': npm_registry,
            'binary_url': binary_url,
            'binary_checksum': binary_checksum,
            'mcp_servers': mcp_servers,
            'extra_env': extra_env,
            'config_file_patches': config_file_patches,
            'pre_start_scripts': pre_start_scripts,
        },
        sort_keys=True,
        default=str,
    )
    if _claude_path and os.path.exists(_claude_path):
        if signature != _setup_signature:
            raise ValueError(
                'this container was already set up with different settings, and the '
                'install is what makes a second setup a no-op: the patches, the '
                'pre-start scripts and the MCP servers below it would not run. Start a '
                'session with the settings the first one used, or use another container.'
            )
        return {'claude_path': _claude_path}

    if binary_url:
        _claude_path = str(
            await install_claude_code_from_url(
                url=binary_url,
                install_dir=_DEFAULT_INSTALL_DIR,
                checksum=binary_checksum,
            )
        )
    else:
        _claude_path = str(
            await install_claude_code_via_npm(
                install_dir=_DEFAULT_INSTALL_DIR,
                version=version,
                npm_registry=npm_registry,
            )
        )
    logger.info('Claude Code installed at: %s', _claude_path)

    for patch in config_file_patches or []:
        _apply_json_patch(patch['path'], patch.get('content') or {})

    claude_env = {
        'HOME': os.environ.get('HOME', '/root'),
        'PATH': build_runtime_path(_DEFAULT_INSTALL_DIR),
    }
    if extra_env:
        claude_env.update(extra_env)
    for script in pre_start_scripts or []:
        await _run_pre_start_script(script, working_dir, claude_env)

    # After the patches and the scripts, so a hook either of them installed is
    # kept rather than replaced.
    _deploy_ask_user_hook()
    await _register_mcp_servers(_claude_path, working_dir=working_dir, claude_env=claude_env, mcp_servers=mcp_servers)

    _setup_signature = signature
    return {'claude_path': _claude_path}


async def open_session(
    *,
    session_id: str,
    label: str,
    timeout: float = _DEFAULT_CC_TURN_TIMEOUT_SECONDS,
    conversion_log_level: str = 'INFO',
    thinking_preset: str = 'hunyuan',
) -> dict[str, Any]:
    """Start this session's own labelled proxy. Idempotent per session.

    The label is fixed for the proxy's lifetime, which is what makes concurrent
    turns safe: there is no per-turn slot for another session to overwrite.
    """
    from nexus.extensions.agents.installed_common import find_free_port
    from nexus.extensions.tasks.hitl.actor.labelling.proxy import build_labelled_proxy_class
    from nexus.runtime.context import require_runtime_control

    if not session_id:
        raise ValueError('session_id must be non-empty')
    if not label:
        raise ValueError('label must be non-empty; an unlabelled session cannot be attributed')

    existing = _sessions.get(session_id)
    if existing is not None:
        if existing.label != label:
            raise ValueError(f'session {session_id} is already open with label {existing.label!r}, not {label!r}')
        return {'proxy_port': existing.port, 'label': existing.label}

    proxy_class = build_labelled_proxy_class()
    rt = require_runtime_control()
    port = find_free_port()
    proxy = proxy_class(
        sio=rt.sio,
        listen_port=port,
        listen_host='127.0.0.1',
        request_timeout=timeout,
        conversion_log_level=conversion_log_level,
        thinking_config=thinking_preset,
        session_label=label,
    )
    await proxy.start()
    _sessions[session_id] = _SessionProxy(proxy=proxy, port=port, label=label)
    logger.info('Session %s proxy on port %d labelling as %s', session_id, port, label)
    return {'proxy_port': port, 'label': label}


async def run_turn(
    *,
    session_id: str,
    prompt: str,
    resume: bool = False,
    working_dir: str = '/workspace',
    timeout: float = _DEFAULT_CC_TURN_TIMEOUT_SECONDS,
    extra_env: dict[str, str] | None = None,
    extra_cli_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run one CC turn via ``--print``. Returns when the process exits.

    Reads only immutable state, so concurrent turns for different sessions do
    not interfere: each resolves its own proxy port from ``_sessions``.
    """
    from nexus.extensions.agents.installed_common import (
        DEFAULT_PROXY_API_KEY,
        build_runtime_path,
    )
    from nexus.runtime.executors.command import RunCommandRequest, run_command

    if not _claude_path:
        raise RuntimeError('Claude Code not installed. Call setup_env() first.')
    session = _sessions.get(session_id)
    if session is None:
        raise RuntimeError(f'Session {session_id} has no proxy. Call open_session() first.')

    start_time = time.time()

    claude_env = {
        'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{session.port}',
        'ANTHROPIC_API_KEY': DEFAULT_PROXY_API_KEY,
        'HOME': os.environ.get('HOME', '/root'),
        'PATH': build_runtime_path(_DEFAULT_INSTALL_DIR),
        'IS_SANDBOX': '1',
        'CLAUDE_CODE_ENTRYPOINT': 'nexus-print',
        'PWD': working_dir,
    }
    if extra_env:
        claude_env.update(extra_env)

    if not Path(working_dir).is_dir():
        # Created rather than refused, because a task may hand the worker a directory
        # it is meant to make. Logged because the other reason it does not exist is a
        # working_dir nobody configured, and a turn in an empty directory looks like a
        # worker that found nothing rather than a run pointed at the wrong place.
        logger.warning('working_dir %s does not exist in the sandbox; creating it empty', working_dir)
        Path(working_dir).mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        _claude_path,
        '--print',
        '--output-format',
        'stream-json',
        '--verbose',
        '--dangerously-skip-permissions',
    ]
    if resume:
        cmd.extend(['--resume', session_id])
    else:
        cmd.extend(['--session-id', session_id])
    if extra_cli_args:
        cmd.extend(extra_cli_args)
    # An answer-injection turn carries no prompt: the answer file is what CC
    # resumes on, and passing a prompt as well would start a new exchange.
    if prompt:
        cmd.extend(['--', prompt])

    shell_cmd = ' '.join(shlex.quote(c) for c in cmd)
    logger.info('Running CC print turn (%s): %s', session_id, shell_cmd[:300])

    result = await run_command(
        RunCommandRequest(
            command=shell_cmd,
            cwd=working_dir,
            timeout=timeout,
            extra_env=claude_env,
        )
    )

    stdout = result.stdout or ''
    stderr = result.stderr or ''
    raw_exit_code = result.return_code
    # A finished CC subprocess sometimes reports no exit status. The run has
    # produced its full output, so a missing code is success (0) rather than a
    # failure or a None that crashes the int() downstream.
    if raw_exit_code is None:
        logger.warning('CC print turn produced no exit code (session_id=%s); treating as success (0)', session_id)
        exit_code = 0
    else:
        exit_code = raw_exit_code
    execution_time = time.time() - start_time

    logger.info(
        'CC print turn done (%s): exit_code=%s time=%.1fs output=%d chars',
        session_id,
        exit_code,
        execution_time,
        len(stdout),
    )

    return {
        'output': stdout,
        'stderr': stderr,
        'exit_code': exit_code,
        'execution_time': execution_time,
        'session_id': session_id,
    }


async def write_answer(*, session_id: str, tool_use_id: str, answers: dict[str, str]) -> None:
    """Write the answer file the hook picks up on the next resume."""
    answer_dir = Path(_ANSWERS_DIR) / session_id
    answer_dir.mkdir(parents=True, exist_ok=True)
    answer_file = answer_dir / f'{tool_use_id}.json'
    answer_file.write_text(json.dumps({'answers': answers}, ensure_ascii=False, indent=2))
    logger.info('Answer file written: %s', answer_file)


async def close_session(*, session_id: str) -> None:
    """Stop one session's proxy, freeing its port."""
    session = _sessions.pop(session_id, None)
    if session is None:
        return
    with contextlib.suppress(Exception):
        await session.proxy.stop()
    logger.info('Session %s proxy on port %d stopped', session_id, session.port)


async def teardown_env() -> None:
    """Stop every remaining proxy."""
    for session_id in list(_sessions):
        await close_session(session_id=session_id)


def _detect_python_path() -> str:
    """Find a working Python interpreter in the sandbox."""
    import shutil

    path = shutil.which('python3')
    if path:
        return path
    python_version = os.environ.get('PYTHON_VERSION', '3.11')
    nix_path = f'/nix/runtime/nexus/.venv-{python_version}/bin/python'
    if os.path.isfile(nix_path):
        return nix_path
    path = shutil.which('python')
    if path:
        return path
    return '/usr/bin/python3'


def _deploy_ask_user_hook() -> None:
    """Deploy the AskUserQuestion hook and add it to the global settings.

    ExitPlanMode is auto-allowed, matching the legacy runtime: plan mode is not
    an interaction the orchestrator is asked about.

    The two entries are appended to whatever ``PreToolUse`` already holds, and
    every other hook event is left alone. A matcher of the same shape is not added
    twice, so this is safe to call again on a container that already has it.
    """
    from nexus.extensions.agents.claude_code.interactive.runtime import _append_required_hooks

    hooks_dir = Path(_HOOKS_DIR)
    hooks_dir.mkdir(parents=True, exist_ok=True)

    script_content = _ASK_USER_HOOK_SCRIPT_TEMPLATE.format(python_path=_detect_python_path())
    script_path = hooks_dir / _HOOK_SCRIPT_NAME
    script_path.write_text(script_content)
    script_path.chmod(0o755)

    Path(_ANSWERS_DIR).mkdir(parents=True, exist_ok=True)

    settings_dir = Path(os.environ.get('HOME', '/root')) / '.claude'
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_path = settings_dir / 'settings.json'

    _append_required_hooks(
        str(settings_path),
        {
            'hooks': {
                'PreToolUse': [
                    {
                        'matcher': 'AskUserQuestion',
                        'hooks': [{'type': 'command', 'command': str(script_path)}],
                    },
                    {
                        'matcher': 'ExitPlanMode',
                        'hooks': [
                            {
                                'type': 'command',
                                'command': 'echo \'{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow","updatedInput":{}}}\'',
                            }
                        ],
                    },
                ]
            }
        },
    )
    logger.info('AskUserQuestion hook deployed: script=%s settings=%s', script_path, settings_path)


async def _register_mcp_servers(
    claude_path: str,
    *,
    working_dir: str,
    claude_env: dict[str, str],
    mcp_servers: list[dict[str, Any]] | None,
) -> None:
    """Register MCP servers via ``claude mcp add``."""
    from nexus.extensions.tasks.hitl.actor.mcp import resolve_interpreters
    from nexus.runtime.executors.command import RunCommandRequest, run_command

    for mcp_cfg in resolve_interpreters(mcp_servers):
        server_name = mcp_cfg.get('name', '')
        server_cmd = mcp_cfg.get('command', '')
        server_args = mcp_cfg.get('args', [])
        if not server_name or not server_cmd:
            logger.warning('Skipping invalid MCP server config: %s', mcp_cfg)
            continue
        mcp_add_cmd = [
            claude_path,
            'mcp',
            'add',
            '--transport',
            'stdio',
            server_name,
            '--',
            server_cmd,
            *server_args,
        ]
        result = await run_command(
            RunCommandRequest(
                command=' '.join(shlex.quote(c) for c in mcp_add_cmd),
                cwd=working_dir,
                timeout=30,
                extra_env=claude_env,
            )
        )
        if (result.return_code or 0) != 0:
            logger.warning('MCP add %s exited %s: %s', server_name, result.return_code, (result.stderr or '').strip())


__all__ = [
    'close_session',
    'open_session',
    'run_turn',
    'setup_env',
    'teardown_env',
    'write_answer',
]
