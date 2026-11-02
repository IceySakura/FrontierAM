"""OSWorld V2.1 lifecycle on existing Nexus agents/runtime, with fork-owned policy."""

import asyncio
import base64
import json
import logging
import math
import os
import re
import secrets
import signal
import sys
import tempfile

from contextlib import asynccontextmanager, nullcontext, suppress
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from nexus.core import require_context
from nexus.extensions.tasks.os_world_v2.adapter.cad_grader import CADGraderOptions
from nexus.extensions.tasks.os_world_v2.adapter.pdf_grader import PDFGraderOptions
from nexus.extensions.tasks.os_world_v2.adapter.source import COMMIT, RELEASE, task_id, validate_task_assets, verify
from nexus.extensions.tasks.os_world_v2.adapter.transport import Transport, local_gitlab_origin
from nexus.utils.timeout import async_timeout


class Options(BaseModel):
    model_config = ConfigDict(extra='forbid')
    repo_root: str = ''
    source_commit: str = COMMIT
    asset_root: str = ''
    website_host_suffix: str = 'localhost'
    official_timeout: float = Field(default=900, gt=0)
    dependency_root: str = ''
    environment: dict[str, str] = Field(default_factory=dict, repr=False)
    setup_only: bool = False
    website_ready_timeout: float = Field(default=900, gt=0)
    gitlab_ready_timeout: float = Field(default=1800, gt=0)
    gitlab_initial_wait: float = Field(default=600, ge=0)
    cad_grader: CADGraderOptions = Field(default_factory=CADGraderOptions)
    pdf_grader: PDFGraderOptions = Field(default_factory=PDFGraderOptions)


async def wait_local_gitlab(runtime, timeout, *, initial_wait=600):
    """Wait before probing guest GitLab; the deadline includes the initial sleep."""
    if timeout <= initial_wait:
        raise ValueError('gitlab_ready_timeout must exceed gitlab_initial_wait')
    logger = logging.getLogger(__name__)
    deadline = asyncio.get_running_loop().time() + timeout
    logger.info('GitLab setup: waiting %.0fs before readiness probes (total deadline %.0fs)', initial_wait, timeout)
    await asyncio.sleep(initial_wait)
    try:
        async with async_timeout(max(0, deadline - asyncio.get_running_loop().time())):
            while True:
                result = await runtime.run_command(
                    command=(
                        'systemctl show osworld-gitlab.service '
                        '--property=LoadState --property=ActiveState --property=Result'
                    ),
                    timeout=20,
                )
                state = {
                    line.split('=', 1)[0]: line.split('=', 1)[1]
                    for line in (result.stdout or '').splitlines()
                    if '=' in line
                }
                if state.get('ActiveState') == 'failed':
                    raise RuntimeError('AGS osworld-gitlab.service failed; inspect GitLab startup logs')
                if (
                    result.return_code == 0
                    and state.get('ActiveState') == 'active'
                    and state.get('Result') == 'success'
                ):
                    probe = await runtime.run_command(
                        command=(
                            "test -f /run/osworld-gitlab-ready && python3 - <<'PY'\n"
                            'import requests\n'
                            's = requests.Session()\n'
                            's.trust_env = False\n'
                            "r = s.get('http://localhost/users/sign_in', timeout=10)\n"
                            "assert r.status_code == 200 and 'GitLab' in r.text\n"
                            "print('NEXUS_GITLAB_READY')\n"
                            'PY'
                        ),
                        timeout=20,
                    )
                    if probe.return_code == 0 and 'NEXUS_GITLAB_READY' in (probe.stdout or ''):
                        logger.info('GitLab setup: readiness probe passed')
                        return
                await asyncio.sleep(5)
    except TimeoutError as exc:
        raise TimeoutError('AGS GitLab initialization did not become ready before the deadline') from exc


async def wait_local_websites(runtime, timeout):
    """The image's platform probe becomes ready before Docker finishes loading sites."""
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        result = await runtime.run_command(
            command='systemctl show osworld-web.service --property=ActiveState --property=SubState --property=Result',
            timeout=20,
        )
        state = {
            line.split('=', 1)[0]: line.split('=', 1)[1] for line in (result.stdout or '').splitlines() if '=' in line
        }
        if result.return_code == 0 and state.get('ActiveState') == 'active' and state.get('Result') == 'success':
            return
        if state.get('ActiveState') == 'failed':
            raise RuntimeError('AGS osworld-web.service failed; inspect image startup logs')
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(f'AGS mock websites not ready: {state}')
        await asyncio.sleep(5)


async def ensure_image_cdp(runtime, password):
    """Validate the image contract; never replace Chrome or inject a guest proxy."""
    del password
    probe = await runtime.run_command(
        command='systemctl show cdp_proxy.service --property=LoadState --property=ActiveState',
        timeout=20,
    )
    state = {line.split('=', 1)[0]: line.split('=', 1)[1] for line in (probe.stdout or '').splitlines() if '=' in line}
    if probe.return_code != 0 or state.get('LoadState') != 'loaded' or state.get('ActiveState') != 'active':
        raise RuntimeError('Image cdp_proxy.service missing or inactive; refusing to replace it')


class Worker:
    def __init__(self, runtime, timeout):
        self.runtime, self.timeout = runtime, timeout
        self.process = None
        self.home = None
        self.stderr_task = None
        self.stderr_tail = ''
        self.phase_stderr_tail = None
        self.stderr_boundary = None
        self.stderr_end = asyncio.Event()
        self.diagnostic_directory = None
        self.secret_values = []
        self.task_id = None
        self.cad_grader = None
        self.pdf_grader = None

    async def _read_stderr(self):
        process = self.process
        assert process is not None
        assert process.stderr is not None
        pending = b''
        while chunk := await process.stderr.read(4096):
            pending += chunk
            while b'\n' in pending or len(pending) > 65536:
                if b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    line += b'\n'
                else:
                    line, pending = pending[:65536], pending[65536:]
                self._stderr_line(line.decode(errors='replace'))
        if pending:
            self._stderr_line(pending.decode(errors='replace'))

    def _stderr_line(self, text):
        marker = f'NEXUS_WORKER_STDERR:{self.stderr_boundary}:'
        if self.stderr_boundary and text.strip() == marker + 'start':
            self.phase_stderr_tail = ''
            return
        if self.stderr_boundary and text.strip() == marker + 'end':
            self.stderr_end.set()
            return
        self.stderr_tail = (self.stderr_tail + text)[-32768:]
        if self.phase_stderr_tail is not None:
            self.phase_stderr_tail = (self.phase_stderr_tail + text)[-32768:]
        if self.diagnostic_directory:
            with suppress(OSError):
                with (Path(self.diagnostic_directory) / 'worker-stderr.log').open('a') as stream:
                    stream.write(self._redact_stderr(text))

    def _redact_stderr(self, text):
        from nexus.extensions.tasks.os_world_v2.adapter.worker import safe_error_message

        for secret in self.secret_values:
            text = text.replace(secret, '[REDACTED]')
        text = ''.join(line for line in text.splitlines(keepends=True) if 'ENV' not in line.upper())
        # Full log is chunked and redacted; do not truncate to the error-summary size.
        return ''.join(safe_error_message(RuntimeError(text[i : i + 4000])) for i in range(0, len(text), 4000))

    def diagnostic_tail(self):
        from nexus.extensions.tasks.os_world_v2.adapter.worker import safe_error_message

        text = self.phase_stderr_tail if self.phase_stderr_tail is not None else self.stderr_tail
        for secret in self.secret_values:
            text = text.replace(secret, '[REDACTED]')
        # Upstream debug logs may dump environments: never surface those lines.
        text = '\n'.join(line for line in text.splitlines() if 'ENV' not in line.upper())
        return safe_error_message(RuntimeError(text[-4000:]))

    async def __aenter__(self):
        self.home = tempfile.TemporaryDirectory(prefix='nexus-v21-')
        env = {key: os.environ[key] for key in ('PATH', 'LANG') if key in os.environ}
        env.update(HOME=self.home.name, PYTHONDONTWRITEBYTECODE='1', PYTHON_DOTENV_DISABLED='1')
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            '-m',
            'nexus.extensions.tasks.os_world_v2.adapter.worker',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,
            limit=64 * 1024 * 1024,
        )
        self.stderr_task = asyncio.create_task(self._read_stderr())
        return self

    async def __aexit__(self, *args):
        async def cleanup():
            if self.process and self.process.returncode is None:
                with suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(self.process.wait(), 3)
                except TimeoutError:
                    with suppress(ProcessLookupError):
                        os.killpg(self.process.pid, signal.SIGKILL)
                    await self.process.wait()
            if self.stderr_task:
                self.stderr_task.cancel()
                await asyncio.gather(self.stderr_task, return_exceptions=True)
            if self.home:
                self.home.cleanup()

        cleanup_task = asyncio.create_task(cleanup())
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            await cleanup_task
            raise

    async def call(self, op, **args):
        try:
            return await self._call(op, **args)
        except BaseException:
            # A cancelled synchronous worker cannot service teardown. Kill it
            # before releasing its proxy/runtime rather than waiting twice.
            await self.__aexit__(None, None, None)
            raise

    async def _call(self, op, **args):
        if op == 'load':
            self.task_id = args.get('task_id')
            self.secret_values = [
                value
                for key, value in args.get('environment', {}).items()
                if value and any(part in key.upper() for part in ('TOKEN', 'SECRET', 'PASSWORD', 'API_KEY'))
            ]
            if args.get('password'):
                self.secret_values.append(args['password'])
            self.diagnostic_directory = args.get('cache')
        process = self.process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError('Official worker is not running')
        self.stderr_boundary = secrets.token_hex(16)
        self.stderr_end.clear()
        self.phase_stderr_tail = ''
        async with async_timeout(self.timeout):
            process.stdin.write(
                (json.dumps({'op': op, 'args': args, 'stderr_boundary': self.stderr_boundary}) + '\n').encode()
            )
            await process.stdin.drain()
            while True:
                raw = await process.stdout.readline()
                if not raw:
                    raise RuntimeError(f'V2.1 official worker exited; stderr_tail={self.diagnostic_tail()}')
                response = json.loads(raw)
                if response.get('runtime_call'):
                    method, kwargs = response['runtime_call'], response['kwargs']
                    if method == 'pdf_grade':
                        if (
                            op != 'evaluate'
                            or self.pdf_grader is None
                            or kwargs.get('task_id') != '025'
                            or self.task_id != '025'
                        ):
                            raise ValueError('PDF grader RPC is only available to task025 evaluate')

                        async def model_callback(request):
                            assert process.stdin is not None
                            assert process.stdout is not None
                            process.stdin.write((json.dumps({'model_call': request}) + '\n').encode())
                            await process.stdin.drain()
                            async with async_timeout(300):
                                reply = json.loads(await process.stdout.readline())
                            value = reply.get('model_result', {})
                            if not value.get('ok') or not isinstance(value.get('text'), str):
                                raise RuntimeError('Official worker auxiliary model failed')
                            return value['text']

                        self.pdf_grader.model_callback = model_callback
                        result = await self.pdf_grader(**kwargs)
                        process.stdin.write((json.dumps({'ok': True, 'result': result}) + '\n').encode())
                        await process.stdin.drain()
                        continue
                    if method == 'cad_grade':
                        if op != 'evaluate' or self.cad_grader is None or kwargs.get('task_id') != self.task_id:
                            raise ValueError('CAD grader RPC is only available to the selected evaluator')
                        result = await self.cad_grader(**kwargs)
                        process.stdin.write((json.dumps({'ok': True, 'result': result}) + '\n').encode())
                        await process.stdin.drain()
                        continue
                    if method not in {'run_command', 'setup_upload_bytes'}:
                        raise ValueError('Unsupported official runtime operation')
                    if method == 'setup_upload_bytes':
                        kwargs['content'] = base64.b64decode(kwargs['content'])
                    if (
                        op == 'prepare'
                        and self.task_id in {'033', '061'}
                        and method == 'run_command'
                        and kwargs.get('timeout', 0) > 120
                    ):
                        from nexus.extensions.tasks.os_world_v2.adapter.preparation import run_preparation

                        result = await run_preparation(self.runtime, **kwargs)
                    else:
                        result = await getattr(self.runtime, method)(**kwargs)
                    fields = ('return_code', 'stdout', 'stderr', 'status_code', 'text', 'ok')
                    serialized = {key: getattr(result, key) for key in fields if hasattr(result, key)}
                    process.stdin.write((json.dumps({'ok': True, 'result': serialized}) + '\n').encode())
                    await process.stdin.drain()
                    continue
                # stdout and stderr are independent pipes. Wait until the
                # matching marker is consumed before attributing the tail.
                if self.stderr_task is not None:
                    await asyncio.wait_for(self.stderr_end.wait(), timeout=2)
                if not response.get('ok'):
                    self._save_phase_diagnostic(op)
                    raise RuntimeError(
                        f'V2.1 official {op} failed: {response["error_type"]}; missing={response.get("missing_module")}; '
                        f'attribute={response.get("attribute")}; frames={response.get("frames")}; '
                        f'message={response.get("message", "")}; stderr_tail={self.diagnostic_tail()}'
                    )
                self._save_phase_diagnostic(op)
                return response['result']

    def _save_phase_diagnostic(self, op):
        """Bounded, redacted worker tail on successful phases as well as failures."""
        try:
            directory = require_context().tool_state.get('osworld_artifact_directory')
        except RuntimeError:
            return
        if directory:
            with suppress(OSError):
                with (Path(directory) / 'worker-phases.jsonl').open('a') as stream:
                    stream.write(json.dumps({'phase': op, 'stderr_tail': self.diagnostic_tail()}) + '\n')


@asynccontextmanager
async def capture_actions(runtime):
    """Observe completed runtime dispatches, not LLM proposals or reasoning."""
    history = []
    originals = {}
    for name in ('run_command', 'gui_pyautogui'):
        original = getattr(runtime, name, None)
        if original is None:
            continue
        originals[name] = (name in vars(runtime), original)

        def wrap(method, kind):
            async def execute(*args, **kwargs):
                result = await method(*args, **kwargs)
                history.append({
                    'executor': kind,
                    'command': kwargs.get('command') or kwargs.get('code') or (args[0] if args else ''),
                    'return_code': getattr(result, 'return_code', None),
                })
                text = history[-1]['command']
                # CUA encodes the executed Python payload in its shell wrapper.
                # Decode literal payloads for the official action-history guard.
                if isinstance(text, str):
                    for payload in re.findall(r"base64\.b64decode\(['\"]([A-Za-z0-9+/=]+)['\"]\)", text):
                        with suppress(ValueError, UnicodeDecodeError):
                            history[-1].setdefault('executed_payloads', []).append(base64.b64decode(payload).decode())
                return result

            return execute

        setattr(runtime, name, wrap(original, name))
    try:
        yield history
    finally:
        for name, (had, original) in originals.items():
            if had:
                setattr(runtime, name, original)
            else:
                delattr(runtime, name)


def executed_cli_actions(directory: Path, harness: str) -> list:
    """CLI completed command events only; never use full responses as action history."""
    if harness == 'codex_native_bridge':
        path = directory / 'codex_stream.jsonl'
        if not path.is_file():
            raise RuntimeError('V2.1 action history requires Codex executed-event evidence')
        result = []
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            item = event.get('item', {})
            if event.get('type') == 'item.completed' and item.get('type', item.get('item_type')) == 'command_execution':
                result.append({'command': item['command'], 'exit_code': item.get('exit_code')})
        return result
    if harness == 'claude_code_native_bridge':
        path = directory / 'cc_stream.jsonl'
        if not path.is_file():
            raise RuntimeError('V2.1 action history requires CC tool-result evidence')
        pending, result = {}, []
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            content = event.get('message', {}).get('content', [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if event.get('type') == 'assistant' and block.get('type') == 'tool_use':
                    pending[block['id']] = block
                elif event.get('type') == 'user' and block.get('type') == 'tool_result':
                    tool = pending.pop(block.get('tool_use_id'), None)
                    if tool is not None:
                        result.append({
                            'tool': tool['name'],
                            'input': tool.get('input', {}),
                            'is_error': block.get('is_error', False),
                        })
        return result
    raise ValueError('V2.1 task requires executed CLI history; this harness is not supported yet')


def score(value):
    value = value['score'] if isinstance(value, dict) else value
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError('Official V2.1 evaluator returned an invalid score')
    return number


async def run(task_input):
    return await _run(task_input)


async def _run(task_input):
    from nexus.extensions.tasks.os_world_v2.adapter.outputs import run_agent
    from nexus.extensions.tasks.os_world_v2.adapter.prompts import (
        build_os_world_agent_tools,
        build_os_world_system_prompt,
        build_os_world_user_prompt,
    )
    from nexus.extensions.tasks.os_world_v2.outputs import OSWorldJudgeOutput
    from nexus.extensions.tasks.os_world_v2.task import OSWorldV2TaskOutput

    ctx = require_context()
    options = Options.model_validate(task_input.v21)
    if options.source_commit != COMMIT:
        raise ValueError('V2.1 source commit does not match adapter profile')
    selected = task_id(task_input.task_id)
    directory = Path(ctx.tool_state['osworld_artifact_directory'])
    dependency_profile = None
    files = getattr(ctx, 'task_files', None)
    if files is not None:
        from nexus.extensions.tasks.os_world_v2.data.bundle import materialize

        if not task_input.bundle_sha256:
            raise ValueError('Remote V2 task_files requires task_config.bundle_sha256')
        root, assets, dependencies, manifest = await materialize(
            files,
            expected_sha256=task_input.bundle_sha256,
            selected=selected,
            destination=directory / 'input',
            raw_key=task_input.wedata_key,
            require_dependencies=task_input.run_setup,
        )
        options = options.model_copy(
            update={'repo_root': str(root), 'asset_root': str(assets), 'dependency_root': str(dependencies)}
        )
        dependency_profile = manifest.dependency_profile
    elif task_input.bundle_sha256:
        raise ValueError('V2 bundle SHA256 specified without task_files; refusing local fallback')
    else:
        if not options.repo_root or not options.asset_root:
            raise ValueError('V2 requires published task_files (or explicit development checkout and asset_root)')
        root, assets = Path(options.repo_root).resolve(), Path(options.asset_root).resolve()
        await asyncio.to_thread(verify, root)
    await asyncio.to_thread(validate_task_assets, root, assets, selected)
    policy = json.loads((root / 'evaluation_examples/setup_support/policies.json').read_text())
    from nexus.extensions.tasks.os_world_v2.execution.runtime import get_osworld_runtime

    # Task-bound capability assembly, as in V1. The provider retains sandbox ownership.
    # Supply V2's password explicitly rather than the V1 image's "password" default.
    params = ctx.runtime_configs.setdefault(task_input.runtime_name, {})
    params.setdefault('sudo_password', task_input.client_password)
    ctx.tool_state['osworld_output_stage'] = 'runtime'
    runtime = cast(Any, await get_osworld_runtime(ctx, task_input.runtime_name))
    ctx.tool_state['osworld_output_stage'] = 'prepare'
    runtime.osworld_command_as_user = task_input.command_as_user
    runtime.osworld_use_proxy = False
    if options.website_host_suffix == 'localhost':
        await wait_local_websites(runtime, options.website_ready_timeout)
    if task_input.run_setup and selected in {'026', '041'} and local_gitlab_origin(options.environment) is not None:
        await wait_local_gitlab(runtime, options.gitlab_ready_timeout, initial_wait=options.gitlab_initial_wait)
    if task_input.run_setup and selected == '048':
        from nexus.extensions.tasks.os_world_v2.adapter.preparation import prepare_task048

        await prepare_task048(runtime, task_input.client_password)
    if getattr(runtime, 'cdp_proxy_mode', None) == 'image':
        await ensure_image_cdp(runtime, task_input.client_password)
    from nexus.extensions.tools.desktop.observation import (
        DESKTOP_OBSERVATION_RUNTIME_NAME_STATE,
        DESKTOP_OBSERVATION_STEP_PAUSE_STATE,
    )

    ctx.tool_state[DESKTOP_OBSERVATION_RUNTIME_NAME_STATE] = task_input.runtime_name
    ctx.tool_state[DESKTOP_OBSERVATION_STEP_PAUSE_STATE] = task_input.step_pause
    cache = directory / 'official-cache'
    cache.mkdir()
    environment = dict(options.environment)
    if any(not name.startswith(('OSWORLD_', 'GITLAB_', 'WEBSITE_')) for name in environment):
        raise ValueError('Only explicit benchmark environment settings are accepted')
    environment.update(WEBSITE_HOST_SUFFIX=options.website_host_suffix, OSWORLD_FILE_BASE_URL=str(assets))
    phases_output, agent_outputs = [], []
    async with Transport(
        runtime,
        policy['website_tunnel_sites'],
        options.website_host_suffix,
        gitlab_origin=local_gitlab_origin(environment),
        gitlab_pages_origin=local_gitlab_origin(environment, 'GITLAB_PAGES_URL'),
        diagnostic_directory=cache,
    ) as transport:
        async with Worker(runtime, options.official_timeout) as worker:
            isolated_cad = selected in {'103', '104'} and options.cad_grader.backend == 'ags'
            isolated_pdf = selected == '025' and options.pdf_grader.backend == 'ags'
            if isolated_pdf:
                from nexus.extensions.tasks.os_world_v2.adapter.pdf_grader import PDFGrader

                worker.pdf_grader = PDFGrader(
                    ctx.runtime_provider,
                    options.pdf_grader,
                    main_tool_id=options.pdf_grader.tool_id or options.cad_grader.tool_id or params.get('tool_id'),
                    password=task_input.client_password,
                    cache=cache,
                    root=root,
                    assets=assets,
                )
                worker.timeout = max(worker.timeout, options.pdf_grader.timeout + 300)
            if isolated_cad:
                from nexus.extensions.tasks.os_world_v2.adapter.cad_grader import CADGrader

                worker.cad_grader = CADGrader(
                    ctx.runtime_provider,
                    options.cad_grader,
                    main_tool_id=params.get('tool_id'),
                    password=task_input.client_password,
                    cache=cache,
                    root=root,
                    assets=assets,
                )
                worker.timeout = max(worker.timeout, options.cad_grader.timeout + 300)
            metadata = await worker.call(
                'load',
                root=str(root),
                task_id=selected,
                host=transport.host,
                cache=str(cache),
                password=task_input.client_password,
                environment=environment,
                sites=policy['website_tunnel_sites'],
                suffix=options.website_host_suffix,
                isolated_cad=isolated_cad,
                isolated_pdf=isolated_pdf,
            )
            if (
                metadata['user_simulator']
                and ctx.agent_name
                not in {
                    'cua_gpt_agent',
                    'claude_code_multiturn',
                    'codex_native_bridge',
                }
                and not options.setup_only
            ):
                raise ValueError('Selected V2.1 HITL task requires an interactive-capable harness')
            cdp = (
                runtime.task_owned_chromium_cdp(client_password=task_input.client_password)
                if metadata['task_owned_cdp']
                else nullcontext()
            )
            async with cdp:
                primary = None
                try:
                    await worker.call(
                        'prepare',
                        run_setup=task_input.run_setup,
                        dependency_root=options.dependency_root,
                        dependency_profile=dependency_profile,
                        image_cdp=runtime.cdp_proxy_mode == 'image',
                    )
                    phases = metadata['phases'] or [{'instruction': metadata['instruction'], 'name': 'task'}]
                    total = 0.0
                    for index, phase in enumerate(phases):
                        ctx.tool_state['osworld_output_stage'] = 'prepare'
                        ctx.tool_state.pop('osworld_judge_output', None)
                        if task_input.run_setup:
                            setup_result = await worker.call('setup', phase=index)
                            if not metadata['phases'] and isinstance(setup_result, dict):
                                phase['instruction'] = setup_result['instruction']
                                metadata['instruction'] = setup_result['instruction']
                            await asyncio.sleep(
                                task_input.setup_wait_time if index == 0 else phase.get('pause_after_setup_seconds', 5)
                            )
                        if options.setup_only:
                            break
                        phase_dir = directory / f'phase-{index + 1:03d}'
                        phase_dir.mkdir()
                        instruction = phase['instruction']
                        screenshot = await runtime.gui_screenshot() if task_input.include_initial_screenshot else None
                        kwargs = {}
                        if ctx.agent_name == 'codex_native_bridge':
                            kwargs['execution_profile'] = 'osworld_v2'
                        if metadata['user_simulator'] or ctx.agent_name == 'claude_code_multiturn':

                            async def ask(request):
                                if getattr(request, 'source', None) == 'turn':
                                    return None
                                if not metadata['user_simulator']:
                                    return None
                                tool_input = (getattr(request, 'hook', None) or {}).get('tool_input', {})
                                questions = tool_input.get('questions')
                                if isinstance(questions, list):
                                    answers = {}
                                    for question in questions:
                                        text = str(question.get('question', ''))
                                        answers[text] = await worker.call('respond', question=text)
                                    return json.dumps(answers, ensure_ascii=False)
                                return await worker.call(
                                    'respond', question=request.hint if hasattr(request, 'hint') else str(request or '')
                                )

                            kwargs['get_user_input'] = ask
                        ctx.tool_state.pop('os_world_last_action', None)
                        ctx.tool_state['osworld_output_stage'] = 'agent'
                        async with capture_actions(runtime) as actions:
                            agent = await run_agent(
                                task_input.run_agent,
                                system_prompt=build_os_world_system_prompt(
                                    action_space=task_input.action_space, client_password=task_input.client_password
                                ),
                                user_prompt=build_os_world_user_prompt(instruction, screenshot),
                                tools=build_os_world_agent_tools(task_input.action_space),
                                post_iteration_hook='nexus.extensions.tools.desktop.observation::desktop_observation',
                                osworld_task_id=selected,
                                osworld_domain='tasks',
                                task_current_date=metadata['task_current_date'],
                                trajectory_dir=str(phase_dir),
                                task_context={
                                    'osworld_task_id': selected,
                                    'osworld_domain': 'tasks',
                                    'trajectory_dir': str(phase_dir),
                                    'osworld_user_simulator': bool(metadata['user_simulator']),
                                    'osworld_client_password': task_input.client_password,
                                },
                                **kwargs,
                            )
                        if agent.exit_code != 0:
                            raise RuntimeError(f'V2.1 agent failed before scoring: {agent.error}')
                        if metadata['requires_action_history'] and ctx.agent_name in {
                            'codex_native_bridge',
                            'claude_code_native_bridge',
                            'claude_code_multiturn',
                        }:
                            # Supervisor commands can contain proposals/guide text.
                            # Only CLI events paired with execution evidence count.
                            actions = executed_cli_actions(
                                phase_dir,
                                'claude_code_native_bridge'
                                if ctx.agent_name == 'claude_code_multiturn'
                                else ctx.agent_name,
                            )
                        marker = ctx.tool_state.get('os_world_last_action')
                        if agent.exit_status == 'finished':
                            marker = 'DONE'
                        history = [*actions, marker] if marker is not None else actions
                        ctx.tool_state['osworld_output_stage'] = 'judge'
                        value = (
                            await worker.call(
                                'evaluate',
                                instruction=instruction,
                                history=history,
                                phase=index if metadata['phases'] else None,
                            )
                            if task_input.run_evaluator
                            else None
                        )
                        points = score(value) if value is not None else 0.0
                        ctx.tool_state['osworld_judge_output'] = (
                            OSWorldJudgeOutput(
                                exit_status='finished',
                                exit_code=0,
                                result=value if isinstance(value, dict) else {'score': points},
                            )
                            if value is not None
                            else OSWorldJudgeOutput(
                                exit_status='judge_error',
                                exit_code=-1,
                                error='V2 evaluation disabled',
                            )
                        )
                        total += points
                        phases_output.append({'phase': index + 1, 'score': points, 'result': value})
                        agent_outputs.append(agent.model_dump(mode='json') if hasattr(agent, 'model_dump') else agent)
                        (phase_dir / 'agent-output.json').write_text(json.dumps(agent_outputs[-1]))
                        (phase_dir / 'evaluation.json').write_text(json.dumps(phases_output[-1]))
                        if task_input.run_evaluator and (
                            ('gate_min_score' in phase and points < phase['gate_min_score'])
                            or (phase.get('gate') and points <= 0)
                        ):
                            break
                except BaseException as exc:
                    primary = exc
                    raise
                finally:
                    try:
                        if worker.process.returncode is None:
                            await worker.call('teardown')
                    except BaseException:
                        if primary is None:
                            raise
    return OSWorldV2TaskOutput(
        status='success' if task_input.run_evaluator and not options.setup_only else 'judge_error',
        completed=True,
        task_id=selected,
        domain='tasks',
        instruction=metadata['instruction'],
        score=round(min(1, total), 4),
        agent_output=ctx.tool_state.get('osworld_agent_output'),
        judge_output=ctx.tool_state.get('osworld_judge_output')
        or OSWorldJudgeOutput(
            exit_status='judge_error',
            exit_code=-1,
            error='V2 setup-only run has no judge result',
        ),
        error=(None if task_input.run_evaluator and not options.setup_only else 'V2 evaluation disabled or setup-only'),
        evaluator_result={
            'score': round(min(1, total), 4),
            'scored': task_input.run_evaluator and not options.setup_only,
            'phases': phases_output,
            'benchmark_release': RELEASE,
            'source_commit': COMMIT,
            'bundle_sha256': task_input.bundle_sha256,
        },
    )
