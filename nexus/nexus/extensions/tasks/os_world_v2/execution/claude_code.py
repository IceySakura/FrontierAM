"""OSWorld Claude Code policy: safe stdin prompts and evaluable outcomes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
import time
import uuid

from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nexus.extensions.agents.claude_code.native_bridge.adapters.osworld import (
    OSWorldClaudeNativeBridgeAdapter as BaseOSWorldAdapter,
)
from nexus.extensions.agents.claude_code.native_bridge.adapters.packages import installed_posix_claude_executable
from nexus.extensions.agents.claude_code.native_bridge.adapters.types import ClaudeInstallSpec, ClaudeRunRequest
from nexus.extensions.agents.claude_code.stream_json import ParentSessionState, parse_transcript
from nexus.extensions.tasks.os_world_v2.execution.guide import render_guide
from nexus.extensions.tasks.os_world_v2.execution.runtime import OSWorldTaskRuntime
from nexus.runtime.providers.ags.osworld import (
    AGSOSWorldExecutorRuntime,
    require_ags_osworld_runtime,
)
from nexus.utils.shell import quote_posix


GUIDE_FILES = {'v1': 'computer_use_guide.md', 'v2': 'computer_use_guide_v2.md'}
logger = logging.getLogger(__name__)


class _ParentDrainReachedError(Exception):
    """The root terminal candidate survived its non-resettable drain interval."""


class _ParentDrainState:
    def __init__(self, grace_seconds: float) -> None:
        self.parent = ParentSessionState()
        self.grace_seconds = grace_seconds
        self.deadline: float | None = None
        self.marker: str | None = None
        self.revoked = False

    def feed(self, event: dict[str, Any], now: float) -> None:
        if not self.parent.feed(event):
            return
        if self.deadline is not None:
            # A new root action or contradictory answer revokes early completion,
            # but cannot buy a fresh grace period later in the same invocation.
            if self.parent.failed or (
                event.get('type') in {'assistant', 'result'} and self.parent.terminal_marker != self.marker
            ):
                self.revoked = True
        elif self.parent.session_id and self.parent.terminal_marker:
            self.marker = self.parent.terminal_marker
            self.deadline = now + self.grace_seconds

    def expired(self, now: float) -> bool:
        return (
            self.deadline is not None
            and now >= self.deadline
            and not self.revoked
            and not self.parent.pending_tools
            and not self.parent.failed
            and self.parent.terminal_marker == self.marker
        )


# Used only after a confirmed wall-clock budget expiration, never on normal
# completion. Stop the current CC execution tree before evaluating its desktop.
_STOP_BUDGET_EXECUTION = r"""
import errno,json,os,pathlib,signal,sys,time
def stat(pid):
    try:
        f=pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()
        return dict(pid=pid,start=f[19],ppid=int(f[1]),alive=f[0]!="Z")
    except OSError as e:
        if e.errno in (errno.ENOENT,errno.ESRCH):
            return dict(pid=pid,alive=False)
        raise
root=int(pathlib.Path(sys.argv[1]).read_text().strip())
if root<=1 or root==os.getpid():
    raise RuntimeError("invalid CC wrapper PID")
first=stat(root)
if not first["alive"]:
    raise RuntimeError("CC wrapper disappeared before budget termination")
all_processes=[stat(int(p.name)) for p in pathlib.Path("/proc").iterdir() if p.name.isdecimal()]
selected={root:first}
while True:
    children={p["pid"]:p for p in all_processes if p.get("alive") and p.get("ppid") in selected}
    before=len(selected); selected.update(children)
    if before==len(selected): break
protected={1,os.getpid()}
parent=os.getppid()
while parent>1 and parent not in protected:
    protected.add(parent); parent=stat(parent).get("ppid",1)
if protected.intersection(selected):
    raise RuntimeError("refusing to terminate control process")
fds={}
try:
    for pid,saved in selected.items():
        try: fd=os.pidfd_open(pid)
        except ProcessLookupError: continue
        current=stat(pid)
        if not current.get("alive") or current.get("start")!=saved["start"]:
            os.close(fd)
            if pid==root: raise RuntimeError("CC wrapper identity changed")
            continue
        fds[pid]=fd
    if root not in fds: raise RuntimeError("CC wrapper disappeared")
    for sig in (signal.SIGTERM,signal.SIGKILL):
        for pid,fd in fds.items():
            try: signal.pidfd_send_signal(fd,sig)
            except ProcessLookupError: pass
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            observed=[stat(pid) for pid in fds]
            if not any(p.get("alive") and p.get("start")==selected[p["pid"]]["start"] for p in observed):
                break
            time.sleep(.05)
    remaining=[stat(pid) for pid in fds]
    remaining=[p for p in remaining if p.get("alive") and p.get("start")==selected[p["pid"]]["start"]]
    print(json.dumps(dict(confirmed=not remaining,targets=list(selected.values()),remaining=remaining)))
    if remaining: sys.exit(1)
finally:
    for fd in fds.values(): os.close(fd)
"""


def _terminal_api_failure(raw: str) -> bool:
    """A real error result outranks budget expiry; model text is not evidence."""
    for line in reversed(raw.splitlines()):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get('type') != 'result' or event.get('parent_tool_use_id'):
            continue
        return bool(event.get('api_error_status')) or (
            event.get('is_error') is True and event.get('subtype') not in {'error_max_turns', 'max_turns'}
        )
    return False


_BASH_PROCESS_GUARD = r"""
import hashlib,json,os,pathlib,re,sys,time
try:
    payload=json.load(sys.stdin)
    command=str(payload.get("tool_input",{}).get("command",""))
    text=command.lower()
    end=r"(?=$|[\s'\",;|&()])"
    kill=re.compile(r"(?<![\w.-])kill"+end)
    reason=None
    if re.search(r"(?<![\w.-])(?:pkill|killall)"+end,text):
        reason="broad process-name termination"
    elif "pgrep" in text and kill.search(text):
        reason="pgrep combined with kill"
    elif "/proc" in text and (kill.search(text) or re.search(r"\bos\.kill\s*\(",text)
                              or "sigkill" in text or "sigterm" in text):
        reason="proc-wide process termination"
    record=dict(time=time.time(),tool_use_id=payload.get("tool_use_id"),
                command_sha256=hashlib.sha256(command.encode()).hexdigest(),
                decision="deny" if reason else "continue",reason=reason)
    with (pathlib.Path(__file__).parent/"guard.jsonl").open("a") as f:
        f.write(json.dumps(record)+"\n"); f.flush(); os.fsync(f.fileno())
    if reason:
        print(json.dumps({"hookSpecificOutput":{
            "hookEventName":"PreToolUse","permissionDecision":"deny",
            "permissionDecisionReason":"OSWorld process guard blocked "+reason+
            ". Prefer application Quit/Close. If you started a child, record its PID and "
            "verify its identity before stopping only that child. Do not retry broad matching "
            "or stop the desktop execution service, bridge or harness."}}))
except Exception as exc:
    print("OSWorld process guard failed: "+type(exc).__name__,file=sys.stderr)
    sys.exit(2)
"""

_GNU_GREP_HOOK = r"""
import json,sys
try:
    payload=json.load(sys.stdin)
    tool=payload.get("tool_input",{})
    command=tool.get("command")
    if not isinstance(command,str):
        raise ValueError("Bash command must be a string")
    # Run after CC's shell snapshot has defined its functions. Nested Bash
    # inherits the exported GNU wrapper as well.
    prefix="grep() { /usr/bin/grep \"$@\"; }; export -f grep\n"
    print(json.dumps({"hookSpecificOutput":{"hookEventName":"PreToolUse",
        "updatedInput":{**tool,"command":prefix+command}}}))
except Exception as exc:
    print("OSWorld GNU grep hook failed: "+type(exc).__name__,file=sys.stderr)
    sys.exit(2)
"""

# Observation only: no signal, process migration, permission mutation or policy.
_OBSERVE_COMMAND = r"""
printf '=== processes ===\n'
ps -eo pid,ppid,pgid,sid,lstart,stat,rss,comm
printf '=== memory ===\n'; cat /proc/meminfo /proc/self/cgroup
printf '=== disk ===\n'; df -h / /tmp; df -i / /tmp
printf '=== cgroup counters ===\n'
for f in /sys/fs/cgroup/memory.events /sys/fs/cgroup/memory.max \
 /sys/fs/cgroup/memory/memory.events /sys/fs/cgroup/memory/memory.oom_control \
 /sys/fs/cgroup/memory/memory.max_usage_in_bytes; do
 test ! -r "$f" || { echo "$f"; cat "$f"; }
done
printf '=== service ===\n'
/usr/bin/systemctl show osworld.service --no-pager \
 -p MainPID -p InvocationID -p NRestarts -p ExecMainCode -p ExecMainStatus \
 -p Result -p KillMode -p ControlGroup -p ActiveState -p ExecMainStartTimestamp
printf '=== journal ===\n'
/usr/bin/journalctl -u osworld.service --no-pager --since=-10min -n 150 -o short-iso
"""


class OSWorldExecutionOptions(BaseModel):
    """Task context and arguments forwarded to the standard runner."""

    model_config = ConfigDict(extra='forbid')
    runtime_name: str = 'main'
    user_prompt: Any = ''
    osworld_task_id: str = ''
    osworld_domain: str = ''
    osworld_related_apps: tuple[str, ...] = ()
    osworld_client_password: str = 'osworld-public-evaluation'
    trajectory_dir: str = ''
    guide_text: str | None = None
    guide_version: Literal['v1', 'v2'] | None = None
    cc_model: str = 'claude-opus-4-8'
    expected_claude_code_version: str | None = '2.1.205'
    binary_url: str | None = None
    binary_checksum: str | None = None
    thinking_preset: str | None = 'auto'
    max_turns: int = Field(default=100, gt=0)
    timeout: float = Field(default=1800.0, ge=1, allow_inf_nan=False)
    disallowed_tools: str = ''
    process_guard: bool = False
    system_grep: bool = True
    parent_terminal_grace_seconds: float | None = Field(default=60.0, gt=0)
    """OSWorld root terminal drain deadline; None preserves CLI-only completion."""
    working_dir: str | None = None
    extra_env: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def normalize_timeout(cls, value):
        # Accept existing saved configs only at the input boundary. The standard
        # name (including an explicit agent-level override) takes precedence.
        if isinstance(value, dict) and 'cc_timeout' in value:
            value = dict(value)
            legacy = value.pop('cc_timeout')
            value.setdefault('timeout', legacy)
        return value


def _guide_text(version: str = '', filename: str = '') -> str:
    version, filename = version.strip().lower(), filename.strip()
    if version:
        if version not in GUIDE_FILES:
            raise ValueError('guide version must be v1 or v2')
        if filename and filename != GUIDE_FILES[version]:
            raise ValueError(f'guide version {version} requires {GUIDE_FILES[version]}')
        filename = GUIDE_FILES[version]
    filename = filename or GUIDE_FILES['v1']
    if Path(filename).name != filename:
        raise ValueError('guide filename must name a file under prompts/')
    return (Path(__file__).resolve().parents[1] / 'prompts' / 'claude_code' / filename).read_text(encoding='utf-8')


class OSWorldClaudeExecutionProfile:
    """Select task-local protections without changing the default native runner."""

    input_overrides = frozenset({'timeout'})

    def __init__(self, options: dict[str, Any], task_context: dict[str, Any], user_prompt: Any):
        keys = {
            'osworld_task_id',
            'osworld_domain',
            'osworld_related_apps',
            'osworld_client_password',
            'trajectory_dir',
        }
        context = {key: task_context[key] for key in keys if key in task_context}
        self.options = OSWorldExecutionOptions.model_validate({**options, **context, 'user_prompt': user_prompt})

    def bridge_options(self, client: Any) -> dict[str, Any]:
        from nexus.runtime.bridge.adapters.ags_osworld import claude_bridge_options

        return claude_bridge_options(self.options, client)

    async def run(self, client: Any) -> dict[str, Any]:
        from nexus.runtime.bridge.adapters.ags_osworld import call_native_bridge_runtime_agent

        return await call_native_bridge_runtime_agent(self.execute, **self.bridge_options(client))

    async def execute(self, *, runtime: Any, launch: Any, timeout: float) -> dict[str, Any]:
        from nexus.extensions.agents.claude_code.native_bridge.agent_runtime import (
            setup_and_run_native_bridge_claude_code,
        )

        options = self.options
        adapters = []

        def adapter_factory(runtime):
            adapter = OSWorldClaudeNativeBridgeAdapter(runtime)
            adapter.process_guard = options.process_guard
            adapter.system_grep = options.system_grep
            adapter.parent_terminal_grace_seconds = options.parent_terminal_grace_seconds
            if options.trajectory_dir:
                adapter.diagnostic_dir = Path(options.trajectory_dir)
            adapters.append(adapter)
            return adapter

        runtime = require_ags_osworld_runtime(runtime)
        working_dir = options.working_dir
        if working_dir is None:
            home = await runtime.run_command(command='printf "%s" "$HOME"', timeout=30.0)
            working_dir = (home.stdout or '').strip()
            if home.return_code != 0 or not working_dir.startswith('/'):
                raise RuntimeError('Unable to resolve OSWorld working directory from sandbox HOME')
        guide = options.guide_text if options.guide_text is not None else _guide_text(options.guide_version or '')
        guide = render_guide(guide, options.osworld_client_password)
        # The task also supplies a base64 screenshot for bare CUA agents. The CC
        # CLI captures screenshots via tools; initial image data is not text input.
        instruction = options.user_prompt
        if isinstance(instruction, list):
            instruction = '\n'.join(
                str(part.get('text', ''))
                for part in instruction
                if isinstance(part, dict) and part.get('type') == 'text'
            )
        prompt = f'{guide.rstrip()}\n{str(instruction).rstrip()}\n' if guide else str(instruction)
        args = [
            '--model',
            options.cc_model,
            '--output-format',
            'stream-json',
            '--verbose',
            '--max-turns',
            str(options.max_turns),
        ]
        if options.disallowed_tools:
            args += ['--disallowedTools', options.disallowed_tools]
        result = await setup_and_run_native_bridge_claude_code(
            runtime=runtime,
            launch=launch,
            task=prompt,
            working_dir=working_dir,
            timeout=timeout,
            version=options.expected_claude_code_version or 'latest',
            binary_url=options.binary_url,
            binary_checksum=options.binary_checksum,
            extra_env=options.extra_env,
            extra_cli_args=args,
            adapter_factory=adapter_factory,
        )
        diagnostics = getattr(adapters[-1], 'diagnostics', {}) if adapters else {}
        # The generic runner returns an error envelope after exceptions. Recover
        # partial output without replacing its primary error or exit code.
        if not result['success'] and diagnostics:
            result['output'] = result.get('output') or diagnostics.get('stdout', '')
            captured_stderr = diagnostics.get('stderr', '')
            if captured_stderr and captured_stderr not in (result.get('stderr') or ''):
                result['stderr'] = (result.get('stderr') or '') + '\n[Captured stderr]\n' + captured_stderr
        trajectory, final_text, subtype = parse_transcript(result.get('output') or '')
        exit_status = 'tool_error'
        budget_signal = subtype in {'error_max_turns', 'max_turns'} or (
            subtype in {None, 'success'} and 'reached max turns' in (result.get('stderr') or '').lower()
        )
        terminal_failure = _terminal_api_failure(result.get('output') or '')
        drain_evidence_valid = False
        if diagnostics.get('budget_reason') == 'parent_terminal_grace':
            final_state = _ParentDrainState(options.parent_terminal_grace_seconds or 60)
            final_state.parent.session_id = diagnostics.get('parent_terminal_session')
            for line in (result.get('output') or '').splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    final_state.feed(event, 0)
            drain_evidence_valid = final_state.expired(
                final_state.grace_seconds
            ) and final_state.marker == diagnostics.get('parent_terminal_marker')
        if (
            diagnostics.get('budget_reason') == 'parent_terminal_grace'
            and diagnostics.get('budget_stop_confirmed') is True
            and not diagnostics.get('budget_execution_error')
            and result['exit_code'] == -1
            and not terminal_failure
            and drain_evidence_valid
            and diagnostics.get('parent_terminal_marker') in {'TASK_DONE', 'TASK_INFEASIBLE'}
        ):
            result['success'] = True
            exit_status = 'infeasible' if diagnostics['parent_terminal_marker'] == 'TASK_INFEASIBLE' else 'finished'
        elif result['exit_code'] in (0, 1) and budget_signal and not terminal_failure:
            # A recorded CLI exit at the turn budget is an evaluable outcome,
            # unlike a vanished process, transport error or unconfirmed timeout.
            result['success'] = True
            exit_status = 'max_iterations'
        elif (
            diagnostics.get('budget_reason') == 'cc_timeout'
            and diagnostics.get('budget_stop_confirmed') is True
            and not diagnostics.get('budget_execution_error')
            and result['exit_code'] == -1
            and not terminal_failure
        ):
            # The CLI was explicitly stopped at its wall-clock budget. Its
            # nonzero exit is not a task failure: evaluate the current state.
            result['success'] = True
            exit_status = 'max_iterations'
        elif result['success']:
            exit_status = 'infeasible' if 'TASK_INFEASIBLE' in (final_text or '') else 'finished'
        mapped = {
            **result,
            'main_trajectory': trajectory,
            'exit_status': exit_status,
            'iterations': len(trajectory),
            'final_answer': final_text,
            'cc_return_code': result['exit_code'],
            'termination_reason': diagnostics.get('budget_reason'),
        }
        if diagnostics.get('budget_execution_error'):
            mapped['failure_stage'] = 'cleanup'
        from nexus.extensions.tasks.os_world_v2.outputs import record_agent_output

        record_agent_output(mapped, harness='cc')
        if options.trajectory_dir:
            directory = Path(options.trajectory_dir)
            try:
                directory.mkdir(parents=True, exist_ok=True)
                (directory / 'cc_prompt.txt').write_text(prompt, encoding='utf-8')
                (directory / 'cc_stream.jsonl').write_text(result.get('output') or '', encoding='utf-8')
                (directory / 'cc_stderr.txt').write_text(result.get('stderr') or '', encoding='utf-8')
                (directory / 'cc_trajectory.json').write_text(
                    json.dumps(mapped, ensure_ascii=False, indent=2), encoding='utf-8'
                )
                if diagnostics:
                    (directory / 'cc_diagnostics.json').write_text(
                        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding='utf-8'
                    )
            except OSError:
                logger.warning('Failed to save OSWorld CC diagnostics', exc_info=True)
        return mapped


class OSWorldClaudeNativeBridgeAdapter(BaseOSWorldAdapter):
    """Install and run Claude Code on AGS OSWorld."""

    _parent_drain_state = _ParentDrainState

    def __init__(self, runtime: AGSOSWorldExecutorRuntime) -> None:
        super().__init__(require_ags_osworld_runtime(runtime))
        self.runtime: OSWorldTaskRuntime
        self._prompt_file: str | None = None
        self.diagnostics: dict[str, Any] = {}
        self.diagnostic_dir: Path | None = None
        self.process_guard = False
        self.system_grep = True
        self.parent_terminal_grace_seconds: float | None = 60.0

    async def _read_parent_events(self, paths, offset: int) -> tuple[bytes, int]:
        """Read a bounded stream chunk; never replay a mutating command."""
        script = (
            'import base64,json,os,sys\n'
            'p=sys.argv[1]; offset=int(sys.argv[2])\n'
            'try:\n'
            ' with open(p,"rb") as f:\n'
            '  size=os.fstat(f.fileno()).st_size\n'
            '  if size<offset: raise RuntimeError("CC stream truncated")\n'
            '  f.seek(offset); data=f.read(1048576)\n'
            '  print(json.dumps({"data":base64.b64encode(data).decode(),"size":size}))\n'
            'except FileNotFoundError:\n'
            ' if offset: raise\n'
            ' print(json.dumps({"data":"","size":0}))\n'
        )
        response = await asyncio.wait_for(
            self.runtime.run_command(
                command=f'/usr/bin/python3 -I -c {quote_posix(script)} {quote_posix(paths.stdout_log)} {offset}',
                timeout=5,
            ),
            timeout=6,
        )
        if response.return_code != 0 or str(getattr(response.status, 'value', response.status)) != 'Finished':
            raise RuntimeError('Cannot read parent terminal evidence')
        import base64

        value = json.loads(response.stdout or '')
        return base64.b64decode(value['data'], validate=True), int(value['size'])

    async def _poll_with_parent_drain(self, paths, timeout):
        grace = getattr(self, 'parent_terminal_grace_seconds', None)
        if grace is None:
            return await super()._poll_until_done(paths, timeout)
        state = self._parent_drain_state(grace)
        loop = asyncio.get_running_loop()
        offset = 0

        async def watch():
            nonlocal offset
            buffered = b''
            try:
                while True:
                    data, size = await self._read_parent_events(paths, offset)
                    offset += len(data)
                    buffered += data
                    complete, _, buffered = buffered.rpartition(b'\n') if b'\n' in buffered else (b'', b'', buffered)
                    if len(buffered) > 32 * 1024**2:
                        raise RuntimeError('CC terminal observation event exceeds 32 MiB')
                    for line in complete.splitlines():
                        if not line.strip():
                            continue
                        event = json.loads(line)
                        if isinstance(event, dict):
                            state.feed(event, loop.time())
                    if offset == size and not buffered and state.expired(loop.time()):
                        check = await asyncio.wait_for(
                            self.runtime.run_command(command=self._build_poll_command(paths), timeout=5), timeout=6
                        )
                        status = (check.stdout or '').strip()
                        if check.return_code != 0 or str(getattr(check.status, 'value', check.status)) != 'Finished':
                            raise RuntimeError('Cannot verify CC process before parent drain')
                        if status.startswith('DONE '):
                            return  # Let ordinary polling preserve the CLI result.
                        if status != 'RUNNING':
                            raise RuntimeError('CC process unavailable before parent drain')
                        self.diagnostics.update(
                            budget_reason='parent_terminal_grace',
                            parent_terminal_marker=state.marker,
                            parent_terminal_session=state.parent.session_id,
                            parent_terminal_grace_seconds=grace,
                        )
                        raise _ParentDrainReachedError()
                    await asyncio.sleep(0 if data and offset < size else 1)
            except _ParentDrainReachedError:
                raise
            except Exception as exc:
                # Incomplete observation must never manufacture a terminal result.
                self.diagnostics['parent_terminal_observation_error'] = type(exc).__name__
                self._save_diagnostics()

        poll = asyncio.create_task(super()._poll_until_done(paths, timeout))
        observer = asyncio.create_task(watch())
        try:
            done, _ = await asyncio.wait([poll, observer], return_when=asyncio.FIRST_COMPLETED)
            if poll in done:
                return await poll
            await observer
            return await poll
        finally:
            poll.cancel()
            observer.cancel()
            await asyncio.gather(poll, observer, return_exceptions=True)

    async def _observe(self, key: str) -> None:
        """Bounded best effort; observations never feed the exit classifier."""
        try:
            result = await asyncio.wait_for(self.runtime.run_command(command=_OBSERVE_COMMAND, timeout=5), timeout=6)
            self.diagnostics[key] = {
                'time': time.time(),
                'return_code': result.return_code,
                'stdout': (result.stdout or '')[-65536:],
                'stderr': (getattr(result, 'stderr', '') or '')[-4096:],
            }
        except Exception as exc:
            self.diagnostics[key] = {'error_type': type(exc).__name__}
        self._save_diagnostics()

    async def _poll_until_done(self, paths, timeout):
        # The common runner classifies DONE/GONE and selects the polling interval.
        # The explicit wall budget also bounds the host wait.
        runtime = self.runtime
        adapter = self
        deadline = asyncio.get_running_loop().time() + timeout
        poll_command = self._build_poll_command(paths)

        class ObservedRuntime:
            def __getattr__(self, name):
                return getattr(runtime, name)

            async def run_command(self, *args, **kwargs):
                # Stream observation can carry large base64 chunks. Keep the
                # existing probe journal about process state, not repeated images.
                command = kwargs.get('command', args[0] if args else None)
                if command != poll_command:
                    return await runtime.run_command(*args, **kwargs)
                record = {'time': time.time()}
                try:
                    result = await runtime.run_command(*args, **kwargs)
                    record.update(
                        return_code=result.return_code,
                        status=str(result.status),
                        stdout=(result.stdout or '')[-65536:],
                        stderr=(result.stderr or '')[-4096:],
                    )
                    return result
                except BaseException as exc:
                    record['error_type'] = type(exc).__name__
                    raise
                finally:
                    try:
                        history = adapter.diagnostics.setdefault('probes', [])
                        history.append(record)
                        del history[:-12]
                        directory = getattr(adapter, 'diagnostic_dir', None)
                        if directory is not None:
                            directory.mkdir(parents=True, exist_ok=True)
                            with (directory / 'cc_probes.jsonl').open('a', encoding='utf-8') as f:
                                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    except Exception:
                        logger.warning('Unable to record CC poll evidence', exc_info=True)

        self.runtime = ObservedRuntime()  # type: ignore[assignment]
        try:
            try:
                outcome = await asyncio.wait_for(self._poll_with_parent_drain(paths, timeout), timeout=timeout)
            except _ParentDrainReachedError:
                outcome = (None, True, False)
            except TimeoutError:
                if asyncio.get_running_loop().time() < deadline:
                    raise  # An early transport timeout is not budget exhaustion.
                # A blocked HTTP request is not proof of a model budget stop.
                # Require a fresh, read-only response before classifying it.
                check = await asyncio.wait_for(
                    self.runtime.run_command(command=self._build_poll_command(paths), timeout=5), timeout=6
                )
                status = (check.stdout or '').strip()
                if check.return_code != 0 or str(getattr(check.status, 'value', check.status)) != 'Finished':
                    raise RuntimeError('Unable to verify CC state at time budget') from None
                if status.startswith('DONE '):
                    value = status.split()[-1]
                    if not value.lstrip('-').isdigit():
                        raise RuntimeError('Invalid CC exit code at time budget') from None
                    return int(value), False, False
                if status != 'RUNNING':
                    raise RuntimeError(f'CC execution unavailable at time budget: {status}') from None
                outcome = (None, True, False)
        finally:
            self.runtime = runtime
        if outcome[1]:
            # Stop before reading output/evaluating, not after another long HTTP wait.
            self.diagnostics.setdefault('budget_reason', 'cc_timeout')
            try:
                await self._capture_diagnostics(paths)
                if self.diagnostics.get('budget_reason') == 'parent_terminal_grace':
                    # Refresh all bytes after cancelling the observer. If root
                    # work resumed during diagnostics, fail closed, never score
                    # the stale terminal candidate.
                    raw = await self._read_file(paths.stdout_log)
                    candidate = self._parent_drain_state(self.diagnostics['parent_terminal_grace_seconds'])
                    candidate.parent.session_id = self.diagnostics['parent_terminal_session']
                    for line in raw.splitlines():
                        if not line.strip():
                            continue
                        event = json.loads(line)
                        if isinstance(event, dict):
                            candidate.feed(event, 0)
                    if (
                        not candidate.expired(candidate.grace_seconds)
                        or candidate.marker != self.diagnostics['parent_terminal_marker']
                    ):
                        raise RuntimeError('Parent terminal evidence changed before drain')
                    check = await asyncio.wait_for(
                        runtime.run_command(command=self._build_poll_command(paths), timeout=5), timeout=6
                    )
                    status = (check.stdout or '').strip()
                    if check.return_code != 0 or str(getattr(check.status, 'value', check.status)) != 'Finished':
                        raise RuntimeError('Unable to verify CC state before drain')
                    if status.startswith('DONE ') and status.split()[-1].lstrip('-').isdigit():
                        self.diagnostics.pop('budget_reason', None)
                        return int(status.split()[-1]), False, False
                    if status != 'RUNNING':
                        raise RuntimeError('CC process unavailable before drain')
                command = f'/usr/bin/python3 -I -c {quote_posix(_STOP_BUDGET_EXECUTION)} {quote_posix(paths.pid)}'
                stopped = await asyncio.wait_for(runtime.run_command(command=command, timeout=10), timeout=11)
                self.diagnostics['budget_termination'] = {
                    'return_code': stopped.return_code,
                    'stdout': stopped.stdout,
                    'stderr': stopped.stderr,
                }
                if stopped.return_code != 0 or str(getattr(stopped.status, 'value', stopped.status)) != 'Finished':
                    raise RuntimeError('CC time-budget termination could not be confirmed')
                try:
                    value = json.loads(stopped.stdout or '')
                except (ValueError, TypeError):
                    raise RuntimeError('CC time-budget termination returned invalid evidence') from None
                if not isinstance(value, dict) or value.get('confirmed') is not True:
                    raise RuntimeError('CC time-budget termination could not be confirmed')
                self.diagnostics['budget_stop_confirmed'] = True
            except BaseException:
                self.diagnostics['budget_execution_error'] = True
                raise
            finally:
                self._save_diagnostics()
        return outcome

    async def _read_file(self, path: str) -> str:
        if not getattr(self, 'diagnostics', {}).get('budget_stop_confirmed'):
            return await super()._read_file(path)
        try:
            response = await asyncio.wait_for(
                self.runtime.run_command(command=f'cat -- {quote_posix(path)}', timeout=5), timeout=6
            )
            if response.return_code != 0 or str(getattr(response.status, 'value', response.status)) != 'Finished':
                raise RuntimeError('Unable to read CC output after budget stop')
            return response.stdout or ''
        except BaseException:
            self.diagnostics['budget_execution_error'] = True
            raise

    async def _install_guard(self, request, directory):
        guard = getattr(self, 'process_guard', False)
        grep = getattr(self, 'system_grep', False)
        if not guard and not grep:
            return request
        hooks = []
        files = []
        if guard:
            files.append(('g', _BASH_PROCESS_GUARD))
            hooks.append({
                'type': 'command',
                'timeout': 5,
                'command': f'/usr/bin/python3 -I {quote_posix(directory + "/g")} || exit 2',
            })
        if grep:
            check = await self.runtime.run_command(command='/usr/bin/grep --version', timeout=10)
            if check.return_code != 0 or 'GNU grep' not in (check.stdout or ''):
                raise RuntimeError('OSWorld requires /usr/bin/grep to be GNU grep')
            files.append(('gnu-grep', _GNU_GREP_HOOK))
            hooks.append({
                'type': 'command',
                'timeout': 5,
                'command': f'/usr/bin/python3 -I {quote_posix(directory + "/gnu-grep")} || exit 2',
            })
        settings = {
            'hooks': {
                'PreToolUse': [
                    {
                        'matcher': 'Bash',
                        'hooks': hooks,
                    }
                ]
            }
        }
        for name, content in [*files, ('settings.json', json.dumps(settings))]:
            data = content.encode()
            path = directory + '/' + name
            result = await asyncio.wait_for(self.runtime.setup_upload_bytes(path, data, filename=name), timeout=30)
            if result.status_code != 200:
                raise RuntimeError('Unable to upload explicitly requested CC process guard')
            check = await self.runtime.run_command(
                command=f'chmod 600 {quote_posix(path)} && sha256sum {quote_posix(path)}', timeout=10
            )
            if check.return_code != 0 or (check.stdout or '').split()[:1] != [hashlib.sha256(data).hexdigest()]:
                raise RuntimeError('CC process guard checksum mismatch')
        self.diagnostics['process_guard'] = {'enabled': guard, 'policy': 'broad-process-commands-v1'}
        self.diagnostics['system_grep'] = {'enabled': grep, 'executable': '/usr/bin/grep'}
        return replace(
            request,
            extra_cli_args=[
                *(request.extra_cli_args or []),
                '--settings',
                directory + '/settings.json',
            ],
        )

    async def _capture_guard(self, directory):
        if not getattr(self, 'process_guard', False):
            return
        try:
            result = await asyncio.wait_for(
                self.runtime.run_command(
                    command=f'tail -c 65536 -- {quote_posix(directory + "/guard.jsonl")}', timeout=2
                ),
                timeout=3,
            )
            target = getattr(self, 'diagnostic_dir', None)
            if result.return_code == 0 and target is not None:
                target.mkdir(parents=True, exist_ok=True)
                (target / 'cc_process_guard.jsonl').write_text(result.stdout or '', encoding='utf-8')
        except Exception:
            logger.warning('Unable to capture CC process guard evidence', exc_info=True)

    def _save_diagnostics(self) -> None:
        directory = getattr(self, 'diagnostic_dir', None)
        if directory is None:
            return
        try:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'cc_diagnostics.json').write_text(
                json.dumps(self.diagnostics, ensure_ascii=False, indent=2), encoding='utf-8'
            )
            for key, name in (('stdout', 'cc_stream.jsonl'), ('stderr', 'cc_stderr.txt')):
                if key in self.diagnostics:
                    (directory / name).write_text(self.diagnostics[key], encoding='utf-8')
        except OSError:
            logger.warning('Failed to persist pre-cleanup CC diagnostics', exc_info=True)

    async def _capture_diagnostics(self, paths) -> None:
        """Best-effort, bounded tails before the generic cleanup deletes files."""
        if not hasattr(self, 'diagnostics'):
            self.diagnostics = {}
        self.diagnostics.update(files={}, errors=[], truncated_tails=True)
        for key, path in (
            ('stdout', paths.stdout_log),
            ('stderr', paths.stderr_log),
            ('exit_code', paths.exit_code),
            ('pid', paths.pid),
        ):
            try:
                result = await asyncio.wait_for(
                    self.runtime.run_command(
                        command=f'tail -c 65536 -- {quote_posix(path)}',
                        timeout=2,
                    ),
                    timeout=3,
                )
                self.diagnostics['files'][key] = path
                if result.return_code != 0:
                    self.diagnostics['errors'].append(f'{key}: read failed (rc={result.return_code})')
                else:
                    self.diagnostics[key] = (result.stdout or '')[-65536:]
            except Exception as exc:
                self.diagnostics['errors'].append(f'{key}: {type(exc).__name__}')
        await self._observe('failure_before_cleanup')

    async def _cleanup_session(self, paths, *, terminate: bool) -> None:
        from nexus.extensions.tasks.os_world_v2.data.artifacts import snapshot_sandbox_artifacts

        try:
            await snapshot_sandbox_artifacts(
                self.runtime,
                {
                    'stdout.log': paths.stdout_log,
                    'stderr.log': paths.stderr_log,
                    'exit_code': paths.exit_code,
                    'pid': paths.pid,
                },
                label='cc',
            )
        finally:
            await super()._cleanup_session(paths, terminate=terminate)

    async def _run_cleanup(self, paths, *, terminate: bool) -> None:
        primary = sys.exc_info()[1]
        if getattr(self, 'diagnostics', {}).get('budget_stop_confirmed'):
            # The budget path already stopped the execution tree. Preserve
            # file cleanup without signalling stale PIDs a second time.
            try:
                return await super()._run_cleanup(paths, terminate=False)
            except BaseException:
                self.diagnostics['budget_execution_error'] = True
                raise
        if primary is None and not terminate:
            return await super()._run_cleanup(paths, terminate=False)

        async def capture_and_cleanup():
            try:
                await self._capture_diagnostics(paths)
            except Exception as exc:
                self.diagnostics = {'errors': [f'diagnostics: {type(exc).__name__}']}
            if primary is not None:
                self.diagnostics['primary_error_type'] = type(primary).__name__
            self._save_diagnostics()
            try:
                await super(OSWorldClaudeNativeBridgeAdapter, self)._run_cleanup(paths, terminate=terminate)
            except BaseException as exc:
                self.diagnostics['cleanup_error_type'] = type(exc).__name__
                if primary is None:
                    raise
                logger.warning('CC cleanup failed; preserving %s', type(primary).__name__, exc_info=True)
            finally:
                self._save_diagnostics()

        cleanup = asyncio.create_task(capture_and_cleanup())
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            # Finish bounded capture and the existing cleanup before propagating
            # cancellation; never convert cancellation into an ordinary result.
            try:
                await cleanup
            except BaseException:
                logger.warning('CC cleanup failed during cancellation', exc_info=True)
            raise

    async def run_claude(self, request: ClaudeRunRequest):
        # Upload bytes through the file endpoint, never a shell argv/heredoc.
        # The private directory and exact-byte hash are independent of task text.
        directory = f'/tmp/nexus-cc-input-{uuid.uuid4().hex}'
        prompt_file = f'{directory}/prompt.txt'
        content = request.task.encode('utf-8')
        self.diagnostics = {}
        try:
            prepared = await self.runtime.run_command(
                command=f'mkdir -m 700 {quote_posix(directory)}',
                timeout=30,
            )
            if prepared.return_code != 0:
                raise RuntimeError('Unable to prepare CC prompt directory')
            uploaded = await self.runtime.setup_upload_bytes(prompt_file, content, filename='prompt.txt')
            if uploaded.status_code != 200:
                raise RuntimeError(f'CC prompt upload failed: HTTP {uploaded.status_code}')
            check = await self.runtime.run_command(
                command=f'chmod 600 {quote_posix(prompt_file)} && sha256sum {quote_posix(prompt_file)}',
                timeout=30,
            )
            if (
                check.return_code != 0
                or not (check.stdout or '').split()
                or (check.stdout or '').split()[0] != hashlib.sha256(content).hexdigest()
            ):
                raise RuntimeError('CC prompt upload checksum mismatch')
            self._prompt_file = prompt_file
            request = await self._install_guard(request, directory)
            await self._observe('before_agent')
            result = await super().run_claude(replace(request, task=''))
            if self.diagnostics.get('budget_reason') == 'parent_terminal_grace':
                result = replace(
                    result,
                    stderr=result.stderr.replace(
                        f'[Claude Code timed out after {request.timeout:g}s]',
                        '[Claude Code background drain stopped after parent terminal grace]',
                    ),
                )
            return result
        finally:
            primary = sys.exc_info()[1]
            self._prompt_file = None
            try:
                await self._capture_guard(directory)
                await self.runtime.run_command(command=f'rm -rf {quote_posix(directory)}', timeout=30)
            except BaseException as exc:
                if isinstance(exc, asyncio.CancelledError) and primary is None:
                    raise
                logger.warning('Failed to clean CC prompt file; preserving primary outcome', exc_info=True)

    def _build_launch_command(self, request, paths):
        from nexus.extensions.agents.claude_code.native_bridge.adapters.shell import (
            build_claude_code_command,
            build_claude_code_env,
            shell_join_cmd,
            shell_join_env,
        )

        if not self._prompt_file:
            raise RuntimeError('CC prompt must be uploaded before launch')
        env = build_claude_code_env(local_bridge_base_url=request.launch.local_base_url, extra_env=request.extra_env)
        # The profile leaves this CLI setting to explicit caller configuration.
        if 'CLAUDE_ENABLE_STREAM_WATCHDOG' not in (request.extra_env or {}):
            env.pop('CLAUDE_ENABLE_STREAM_WATCHDOG', None)
        cmd = build_claude_code_command(request.claude_executable, task='', extra_cli_args=request.extra_cli_args)
        if cmd[-2:] == ['--', '']:
            cmd = cmd[:-2]
        inner = (
            f'{shell_join_env(env)} {shell_join_cmd(cmd)} < {quote_posix(self._prompt_file)} '
            f'> {quote_posix(paths.stdout_log)} 2> {quote_posix(paths.stderr_log)}; '
            f'echo $? > {quote_posix(paths.exit_code)}'
        )
        wd = quote_posix(request.working_dir)
        return (
            f'mkdir -p {wd}; mkdir -p {quote_posix(paths.dir)}; cd {wd} && '
            f'{{ nohup bash -c {quote_posix(inner)} </dev/null >/dev/null 2>&1 & '
            f'echo $! > {quote_posix(paths.pid)}; echo LAUNCHED; }}'
        )

    async def ensure_claude(self, spec: ClaudeInstallSpec) -> str:
        # An explicit binary URL is authoritative, matching the Codex adapters.
        if spec.version == 'latest' or spec.binary_url:
            return await super().ensure_claude(spec)
        probe = await self.runtime.run_command(command='command -v claude', timeout=30.0)
        candidates = []
        if probe.return_code == 0 and (probe.stdout or '').strip():
            candidates.append((probe.stdout or '').strip().splitlines()[0])
        candidates.append(installed_posix_claude_executable())
        for executable in dict.fromkeys(candidates):
            result = await self.runtime.run_command(command=f'{quote_posix(executable)} --version', timeout=30.0)
            tokens = (result.stdout or '').strip().split()
            if result.return_code == 0 and tokens and tokens[0] == spec.version:
                logger.info('Reusing Claude Code %s at %s; skipping npm installation', spec.version, executable)
                return executable
        return await super().ensure_claude(spec)


__all__ = ['OSWorldClaudeNativeBridgeAdapter']
