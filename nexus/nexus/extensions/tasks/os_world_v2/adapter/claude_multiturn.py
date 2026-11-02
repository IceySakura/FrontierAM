"""Task-owned AGS transport for claude_code_multiturn, including in-turn user hooks."""

import asyncio
import json
import shlex
import time
import uuid

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from nexus.core import UserInputRequest, require_context
from nexus.extensions.agents.claude_code.interactive import hooks
from nexus.extensions.agents.claude_code.native_bridge.adapters.base import multiturn_execution_profile_registry
from nexus.extensions.agents.claude_code.native_bridge.adapters.types import ClaudeInstallSpec, ClaudeRunRequest
from nexus.extensions.agents.claude_code.stream_json import ParentSessionState, parse_transcript
from nexus.extensions.tasks.os_world_v2.adapter.prompts import RESOURCE_POLICY
from nexus.extensions.tasks.os_world_v2.execution.claude_code import (
    OSWorldClaudeNativeBridgeAdapter,
    OSWorldExecutionOptions,
    _ParentDrainState,
    _guide_text,
    _terminal_api_failure,
)
from nexus.extensions.tasks.os_world_v2.execution.guide import render_guide
from nexus.runtime.bridge.adapters.ags_osworld import call_native_bridge_runtime_agent, claude_bridge_options


class V2ParentSessionState(ParentSessionState):
    @staticmethod
    def _marker(text):
        marker = ParentSessionState._marker(text)
        return marker if marker == 'TASK_DONE' else None


class V2ParentDrainState(_ParentDrainState):
    def __init__(self, grace_seconds):
        super().__init__(grace_seconds)
        self.parent = V2ParentSessionState()


def default_guide(version):
    """Adapt the shared stock guide without changing V1's completion contract."""
    text = _guide_text(version)
    text = text.replace(' (or `TASK_INFEASIBLE`)', '')
    text = text.replace(' If the task is genuinely impossible, print `TASK_INFEASIBLE` instead.', '')
    text = text.replace(
        "If the task is genuinely impossible (missing application, contradictory requirements, feature doesn't exist):\n\n"
        '```\nTASK_INFEASIBLE\n```\n\n',
        '',
    )
    return text.replace('either marker', 'the completion marker')


class InteractiveAGSAdapter(OSWorldClaudeNativeBridgeAdapter):
    """Reuse standard launch/poll/cleanup and V1 protections; add only hook transport."""

    _parent_drain_state = V2ParentDrainState

    def __init__(self, runtime, get_user_input, *, hook_timeout=240, user_simulator=True):
        super().__init__(runtime)
        self.get_user_input = get_user_input
        self.hook_timeout = hook_timeout
        self.interactions = []
        self.interaction_dir = None
        self.session_spec = None
        self.session_id = str(uuid.uuid4())
        self.user_simulator = user_simulator

    async def _install_guard(self, request, directory):
        request = await super()._install_guard(request, directory)
        self.interaction_dir = directory + '/interactions'
        result = await self.runtime.run_command(
            command=f'mkdir -m 700 {shlex.quote(self.interaction_dir)}',
            timeout=20,
        )
        if result.return_code != 0:
            raise RuntimeError('Cannot create private V2 interaction directory')
        # The upstream hook module is standalone and already implements atomic
        # request/response files, request IDs, deadlines and AskUserQuestion answers.
        path = directory + '/input-hook.py'
        from nexus.extensions.tasks.os_world_v2.adapter import unattended_hook

        hook_module = hooks if self.user_simulator else unattended_hook
        response = await self.runtime.setup_upload_bytes(
            path, Path(hook_module.__file__).read_bytes(), filename='input-hook.py'
        )
        if response.status_code != 200:
            raise RuntimeError('Cannot upload V2 user interaction hook')
        if self.system_grep or self.process_guard:
            read = await self.runtime.run_command(
                command=f'cat {shlex.quote(directory + "/settings.json")}', timeout=20
            )
            if read.return_code != 0:
                raise RuntimeError('Cannot read CC protection settings')
            settings = json.loads(read.stdout or '')
        else:
            settings = {}
            request = replace(
                request,
                extra_cli_args=[
                    *(request.extra_cli_args or []),
                    '--settings',
                    directory + '/settings.json',
                ],
            )
        entries = settings.setdefault('hooks', {}).setdefault('PreToolUse', [])
        for tool in ('AskUserQuestion', 'ExitPlanMode'):
            entries.append({
                'matcher': tool,
                'hooks': [
                    {
                        'type': 'command',
                        'timeout': self.hook_timeout + 30,
                        'command': f'/usr/bin/python3 -I {shlex.quote(path)} {tool} || exit 2',
                    }
                ],
            })
        response = await self.runtime.setup_upload_bytes(
            directory + '/settings.json',
            json.dumps(settings).encode(),
            filename='settings.json',
        )
        if response.status_code != 200:
            raise RuntimeError('Cannot install V2 interaction settings')
        request = replace(
            request,
            extra_env={
                **(request.extra_env or {}),
                hooks.INTERACTION_DIR_ENV: self.interaction_dir,
                hooks.HOOK_WAIT_TIMEOUT_ENV: str(self.hook_timeout),
            },
        )
        from nexus.extensions.agents.claude_code.native_bridge.adapters.shell import build_claude_code_env
        from nexus.extensions.tasks.os_world_v2.adapter import cc_session

        command = [
            request.claude_executable,
            '--dangerously-skip-permissions',
            '--input-format',
            'stream-json',
            '--permission-prompt-tool',
            'stdio',
            *(request.extra_cli_args or []),
        ]
        spec = {
            'command': command,
            'prompt_file': self._prompt_file,
            'session_id': self.session_id,
            'env': {
                **build_claude_code_env(
                    local_bridge_base_url=request.launch.local_base_url, extra_env=request.extra_env
                ),
                'CLAUDE_CODE_ENTRYPOINT': 'nexus-stream-json',
            },
        }
        self.session_spec = directory + '/session.json'
        for name, content in [
            ('session.py', Path(cc_session.__file__).read_bytes()),
            ('session.json', json.dumps(spec).encode()),
        ]:
            response = await self.runtime.setup_upload_bytes(directory + '/' + name, content, filename=name)
            if response.status_code != 200:
                raise RuntimeError('Cannot upload CC session supervisor')
        return request

    def _build_launch_command(self, request, paths):
        if self.session_spec is None:
            raise RuntimeError('V2 CC session specification was not installed')
        directory = str(Path(self.session_spec).parent)
        inner = (
            f'python3 {shlex.quote(directory + "/session.py")} {shlex.quote(self.session_spec)} '
            f'> {shlex.quote(paths.stdout_log)} 2> {shlex.quote(paths.stderr_log)}; '
            f'echo $? > {shlex.quote(paths.exit_code)}'
        )
        return (
            f'mkdir -p {shlex.quote(paths.dir)}; cd {shlex.quote(request.working_dir)} && '
            f'{{ nohup bash -c {shlex.quote(inner)} </dev/null >/dev/null 2>&1 & '
            f'echo $! > {shlex.quote(paths.pid)}; echo LAUNCHED; }}'
        )

    async def _service_interactions(self, runtime):
        directory = self.interaction_dir
        if directory is None:
            raise RuntimeError('V2 interaction directory is not initialized')
        answered = set()
        while True:
            script = (
                'import json,pathlib; p=pathlib.Path(' + repr(directory) + '); '
                'print(json.dumps([json.loads(f.read_text()) for f in p.glob("*.request.json")]))'
            )
            from nexus.extensions.tasks.os_world_v2.adapter.control_reads import read_control

            result = await read_control(
                runtime,
                'python3 -c ' + shlex.quote(script),
                label='hitl_requests',
                timeout=20,
                diagnostic_dir=getattr(self, 'diagnostic_dir', None),
            )
            if result.return_code != 0:
                raise RuntimeError('Cannot poll V2 user interaction requests')
            for raw in json.loads(result.stdout or '[]'):
                iid = raw['interaction_id']
                if iid in answered:
                    continue
                request = UserInputRequest(
                    source='hook',
                    hint=json.dumps(raw['tool_input'], ensure_ascii=False),
                    hook={'tool_name': raw['tool_name'], 'tool_input': raw['tool_input']},
                    metadata={'interaction_id': iid},
                )
                answer = await asyncio.wait_for(self.get_user_input(request), self.hook_timeout)
                if not isinstance(answer, str) or not answer.strip():
                    raise RuntimeError('V2 user simulator returned no answer; refusing unattended approval')
                key = hooks.interaction_key(iid)
                path = directory + '/' + key + '.response.json'
                response = await runtime.setup_upload_bytes(
                    path + '.tmp',
                    json.dumps({'interaction_id': iid, 'answer': answer}).encode(),
                    filename='answer.json',
                )
                if response.status_code != 200:
                    raise RuntimeError('Cannot send V2 user simulator answer')
                moved = await runtime.run_command(
                    command=f'mv {shlex.quote(path + ".tmp")} {shlex.quote(path)}',
                    timeout=20,
                )
                if moved.return_code != 0:
                    raise RuntimeError('Cannot publish V2 user simulator answer')
                answered.add(iid)
                self.interactions.append({'interaction_id': iid, 'tool_name': raw['tool_name'], 'answered': True})
            await asyncio.sleep(1)

    async def _poll_until_done(self, paths, timeout):
        from nexus.extensions.tasks.os_world_v2.adapter.control_reads import PollReadRuntime

        original = self.runtime
        self.runtime = cast(
            Any, PollReadRuntime(original, self._build_poll_command(paths), getattr(self, 'diagnostic_dir', None))
        )
        try:
            return await self._poll_with_interactions(paths, timeout)
        finally:
            self.runtime = original

    async def _poll_with_interactions(self, paths, timeout):
        if not self.user_simulator:
            return await super()._poll_until_done(paths, timeout)
        interaction_task = asyncio.create_task(self._service_interactions(self.runtime))
        process_task = asyncio.create_task(super()._poll_until_done(paths, timeout))
        try:
            done, _ = await asyncio.wait((interaction_task, process_task), return_when=asyncio.FIRST_COMPLETED)
            if interaction_task in done:
                interaction_task.result()
                raise RuntimeError('User interaction service stopped unexpectedly')
            return await process_task
        finally:
            interaction_task.cancel()
            process_task.cancel()
            await asyncio.gather(interaction_task, process_task, return_exceptions=True)


@multiturn_execution_profile_registry.register('osworld_v2')
class OSWorldV2MultiturnProfile:
    def __init__(self, input_data):
        self.input = input_data
        overrides = {'timeout': input_data.timeout} if 'timeout' in input_data.model_fields_set else {}
        self.options = OSWorldExecutionOptions.model_validate({
            **input_data.execution_options,
            **overrides,
            **{
                k: v
                for k, v in input_data.task_context.items()
                if k in {'osworld_task_id', 'osworld_domain', 'osworld_client_password', 'trajectory_dir'}
            },
            'user_prompt': input_data.user_prompt,
        })

    async def run(self, client):
        options = claude_bridge_options(self.options, client)
        if getattr(client, 'native_request_protocol', None) != 'anthropic':
            # Chat backends may support images but not OpenAI file parts. Reuse
            # the bounded PDF renderer; keep genuine Anthropic documents native.
            options['api_protocol'] = 'osworld-anthropic'
        return await call_native_bridge_runtime_agent(self.execute, **options)

    async def execute(self, *, runtime, launch, timeout):
        options, inp = self.options, self.input
        has_simulator = inp.task_context.get('osworld_user_simulator', False)
        adapter = InteractiveAGSAdapter(runtime, inp.get_user_input, user_simulator=has_simulator)
        adapter.system_grep, adapter.process_guard = options.system_grep, options.process_guard
        adapter.parent_terminal_grace_seconds = options.parent_terminal_grace_seconds
        if options.trajectory_dir:
            adapter.diagnostic_dir = Path(options.trajectory_dir)
        try:
            executable = await adapter.ensure_claude(
                ClaudeInstallSpec(
                    True,
                    inp.npm_registry,
                    options.expected_claude_code_version or 'latest',
                )
            )
        except Exception as exc:
            from nexus.extensions.tasks.os_world_v2.outputs import record_agent_output

            record_agent_output(
                {'success': False, 'error': f'{type(exc).__name__}: {exc}', 'failure_stage': 'install'},
                harness='cc',
            )
            raise
        session = str(uuid.uuid4()) if not inp.session_id else str(uuid.UUID(inp.session_id))
        adapter.session_id = session
        guide = options.guide_text if options.guide_text is not None else default_guide(options.guide_version or 'v2')
        guide = render_guide(guide, options.osworld_client_password)
        prompt = inp.user_prompt
        if isinstance(prompt, list):
            prompt = '\n'.join(p.get('text', '') for p in prompt if p.get('type') == 'text')
        prompt = guide + '\n' + str(prompt)
        started = time.monotonic()
        output, errors, events = [], [], []
        turns, code, ended, budget_reached = 0, 0, False, False
        parent_finished = dialogue_timed_out = False
        try:
            while prompt is not None and turns < inp.max_turns:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    # agent_config.timeout spans the whole dialogue: an agent timeout,
                    # not a sandbox failure.
                    dialogue_timed_out = True
                    break
                args = [
                    '--append-system-prompt',
                    RESOURCE_POLICY,
                    '--model',
                    options.cc_model,
                    '--output-format',
                    'stream-json',
                    '--verbose',
                    '--max-turns',
                    str(options.max_turns),
                ]
                args += ['--session-id' if turns == 0 else '--resume', session]
                disallowed = options.disallowed_tools
                if not has_simulator:
                    disallowed = ','.join(
                        filter(None, [disallowed, 'AskUserQuestion', 'EnterPlanMode', 'ExitPlanMode'])
                    )
                if disallowed:
                    args += ['--disallowedTools', disallowed]
                result = await adapter.run_claude(
                    ClaudeRunRequest(
                        executable,
                        launch,
                        prompt,
                        options.working_dir or '/home/user',
                        remaining,
                        options.extra_env,
                        args,
                    )
                )
                turns += 1
                output.append(result.stdout)
                errors.append(result.stderr)
                code = result.exit_code
                trajectory, final, subtype = parse_transcript(result.stdout)
                events.extend(trajectory)
                parent_finished = parent_drain_finished(
                    result.stdout, code, adapter.diagnostics, options.parent_terminal_grace_seconds
                )
                if parent_finished:
                    require_context().tool_state['os_world_last_action'] = 'DONE'
                    break
                budget_reached = code in {0, 1} and subtype in {'error_max_turns', 'max_turns'}
                budget_reached = budget_reached or (
                    code == -1
                    and adapter.diagnostics.get('budget_stop_confirmed') is True
                    and adapter.diagnostics.get('budget_reason') == 'cc_timeout'
                    and not adapter.diagnostics.get('budget_execution_error')
                )
                if budget_reached and not _terminal_api_failure(result.stdout):
                    break
                if code != 0 or _terminal_api_failure(result.stdout):
                    break
                # A turn boundary is not an AskUserQuestion. The V2 task callback
                # ends completed turns rather than sending final text to the simulator.
                prompt = await inp.get_user_input(UserInputRequest(source='turn', hint=final, session_id=session))
                ended = prompt is None
        except BaseException as exc:
            # The common adapter captured bounded pre-cleanup evidence. Preserve
            # it even when no successful CLI result was returned to this profile.
            if adapter.diagnostics.get('stdout'):
                output.append(adapter.diagnostics['stdout'])
            if adapter.diagnostics.get('stderr'):
                errors.append(adapter.diagnostics['stderr'])
            from nexus.extensions.tasks.os_world_v2.outputs import record_agent_output

            reason = getattr(exc, 'reason', None)
            record_agent_output(
                {
                    'success': False,
                    'output': '\n'.join(output),
                    'stderr': '\n'.join(errors),
                    'error': f'{type(exc).__name__}: {exc}',
                    'exit_code': -1,
                    'failure_kind': 'infrastructure',
                    'runtime_reason': reason if isinstance(reason, str) else None,
                },
                harness='cc',
            )
            raise
        finally:
            if options.trajectory_dir:
                root = Path(options.trajectory_dir)
                root.mkdir(parents=True, exist_ok=True)
                (root / 'cc_stream.jsonl').write_text('\n'.join(output))
                (root / 'cc_stderr.txt').write_text('\n'.join(errors))
                (root / 'cc_interactions.json').write_text(json.dumps(adapter.interactions, indent=2))
        if parent_finished:
            events.append({
                'kind': 'termination',
                'reason': 'parent_terminal_grace',
                'marker': adapter.diagnostics['parent_terminal_marker'],
                'cc_return_code': code,
                'cleanup_confirmed': True,
            })
        # The common adapter stops CC on the agent_config.timeout budget (cc_timeout);
        # that is agent_timeout, distinct from a --max-turns/user-turn limit.
        timed_out = dialogue_timed_out or (budget_reached and adapter.diagnostics.get('budget_reason') == 'cc_timeout')
        turn_limit_reached = (budget_reached and not timed_out) or (
            not parent_finished
            and not ended
            and not dialogue_timed_out
            and prompt is not None
            and turns >= inp.max_turns
            and code == 0
        )
        user_turn_limit_reached = turn_limit_reached and not budget_reached
        mapped = {
            'success': (code == 0 or budget_reached or parent_finished or dialogue_timed_out)
            and not any(_terminal_api_failure(raw) for raw in output),
            'output': '\n'.join(output),
            'stderr': '\n'.join(errors),
            'exit_code': code,
            'cc_return_code': code,
            'exit_status': (
                'finished'
                if parent_finished or (code == 0 and not timed_out and not turn_limit_reached)
                else 'agent_timeout'
                if timed_out
                else 'max_iterations'
                if turn_limit_reached
                else 'tool_error'
            ),
            'termination_reason': 'cc_timeout'
            if timed_out
            else 'max_turns'
            if user_turn_limit_reached
            else adapter.diagnostics.get('budget_reason'),
            'timed_out': timed_out,
            'budget_stop_confirmed': adapter.diagnostics.get('budget_stop_confirmed', False) or dialogue_timed_out,
            'failure_stage': 'cleanup' if adapter.diagnostics.get('budget_execution_error') else None,
            'execution_time': time.monotonic() - started,
            'session_id': session,
            'turns': turns,
            'events': events,
            'ended_by_user': ended,
            'reached_max_turns': turn_limit_reached,
        }
        from nexus.extensions.tasks.os_world_v2.outputs import record_agent_output

        record_agent_output(mapped, harness='cc')
        return mapped


def parent_drain_finished(raw, code, diagnostics, grace):
    """Use the same evidence and non-resettable drain policy as the V1 profile."""
    if (
        code != -1
        or diagnostics.get('budget_reason') != 'parent_terminal_grace'
        or diagnostics.get('budget_stop_confirmed') is not True
        or diagnostics.get('budget_execution_error')
        or diagnostics.get('parent_terminal_marker') != 'TASK_DONE'
        or _terminal_api_failure(raw)
    ):
        return False
    state = V2ParentDrainState(grace or 60)
    state.parent.session_id = diagnostics.get('parent_terminal_session')
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            state.feed(event, 0)
    return state.expired(state.grace_seconds) and state.marker == diagnostics.get('parent_terminal_marker')
