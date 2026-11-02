"""Explicit, task-owned idempotent control reads; never a general executor retry."""

import hashlib
import json
import time

from pathlib import Path

import aiohttp

from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse
from nexus.utils.timeout import async_timeout


async def read_control(runtime, command, *, label, timeout, diagnostic_dir=None):
    """Internal capability: callers must supply only audited control reads.

    Reuse the OSWorld read-only HTTP policy (five attempts, bounded backoff).
    Bypass run_command's separate three-timeout retry to avoid multiplying
    budgets. The entire read, including backoff, shares the caller's deadline.
    """
    started = time.monotonic()
    record = {'label': label, 'command_sha256': hashlib.sha256(command.encode()).hexdigest()}
    try:
        payload = runtime._build_osworld_execute_payload(
            command=command,
            cwd=None,
            timeout=timeout,
            stdin=None,
            extra_env=None,
        )
        async with async_timeout(timeout):
            response = await runtime.read_only_request(
                'POST',
                runtime.handle.url(runtime.server_port, '/execute'),
                json=payload,
                retry_label='V2 ' + label,
                timeout=aiohttp.ClientTimeout(total=min(10, timeout / 5)),
            )
        record['http_status'] = response.status_code
        if response.status_code >= 400:
            raise RuntimeError(f'V2 control read {label} failed: HTTP {response.status_code}')
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f'V2 control read {label} returned invalid JSON') from exc
        if not isinstance(data, dict) or type(data.get('returncode')) is not int:
            raise RuntimeError(f'V2 control read {label} lacks a valid return code')
        rc = data['returncode']
        record['return_code'] = rc
        return RunCommandResponse(
            status=CommandRunStatus.Finished if rc == 0 else CommandRunStatus.Error,
            return_code=rc,
            stdout=data.get('output'),
            stderr=data.get('error'),
            execution_time=time.monotonic() - started,
        )
    except BaseException as exc:
        record['error_type'] = type(exc).__name__
        raise
    finally:
        record['elapsed_s'] = round(time.monotonic() - started, 3)
        if diagnostic_dir:
            try:
                with (Path(diagnostic_dir) / 'control-reads.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
            except OSError:
                pass  # Diagnostics cannot change the control outcome.


class PollReadRuntime:
    """Match the exact adapter-generated PID read, not arbitrary shell text."""

    def __init__(self, runtime, poll_command, diagnostic_dir):
        self.runtime = runtime
        self.poll_command = poll_command
        self.diagnostic_dir = diagnostic_dir

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    async def run_command(self, *args, **kwargs):
        command = kwargs.get('command', args[0] if args else None)
        if command == self.poll_command and len(args) <= 1 and not (set(kwargs) - {'command', 'timeout'}):
            return await read_control(
                self.runtime,
                command,
                label='cc_pid_exit',
                timeout=kwargs.get('timeout', 60),
                diagnostic_dir=self.diagnostic_dir,
            )
        return await self.runtime.run_command(*args, **kwargs)
