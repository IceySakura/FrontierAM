"""Align the large guest with V1's runner, without installing task dependencies."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import shlex

from typing import Any

from nexus.extensions.agents.codex.native_bridge.adapters.osworld import _NODE_DIR, _NODE_URL
from nexus.extensions.tasks.os_world.execution.sandbox_policy import LARGE_PROVIDER, LARGE_TOOL_ID
from nexus.extensions.tasks.os_world.task import OSWorldTaskInput
from nexus.runtime.client.base import Runtime
from nexus.runtime.providers.ags import AGSRuntimeProvider
from nexus.runtime.providers.registry import runtime_provider_registry


logger = logging.getLogger(__name__)
EXPECTED_CC_VERSION = '2.1.205'
# Public benchmark-image login, not an account credential. No V2 checkout needed.
_IMAGE_LOGIN_DEFAULT = 'osworld-public-evaluation'


async def _run(
    runtime: Any, cmd: str, *, timeout: float, env: dict[str, str] | None = None
) -> tuple[int | None, str, str]:
    response = await runtime.run_command(cmd, timeout=timeout, extra_env=env or {})
    data = response if isinstance(response, dict) else response.model_dump()
    return data.get('return_code'), data.get('stdout') or '', data.get('stderr') or ''


async def _prepare_v1_guest(runtime: Runtime) -> None:
    v1_password = OSWorldTaskInput.model_fields['client_password'].default
    # Only the fresh disposable guest, not recursive ownership or task packages.
    root_command = "chown user:user /home/user && printf '%s\\n' " + shlex.quote('user:' + v1_password) + ' | chpasswd'
    response = await runtime.run_command(
        command="sudo -S -k -p '' sh -c " + shlex.quote(root_command),
        stdin=_IMAGE_LOGIN_DEFAULT + '\n',
        timeout=30,
    )
    if response.return_code != 0:
        raise RuntimeError('Large guest home/sudo alignment failed')
    check = await runtime.run_command(
        command="test -w /home/user && sudo -S -k -p '' true",
        stdin=v1_password + '\n',
        timeout=30,
    )
    if check.return_code != 0:
        raise RuntimeError('V1 guest home/sudo validation failed')


BOOTSTRAP = f"""
set -euo pipefail
cd /home/user
if [ ! -x {_NODE_DIR}/bin/node ]; then
  wget -nv --timeout=90 {_NODE_URL} -O /tmp/node.tar.xz
  mkdir -p {_NODE_DIR}
  tar -xf /tmp/node.tar.xz -C {_NODE_DIR} --strip-components=1
fi
export PATH={_NODE_DIR}/bin:$PATH
if ! claude --version 2>/dev/null | grep -q '^{EXPECTED_CC_VERSION} '; then
  npm install -g --registry=https://registry.npmmirror.com --omit=dev --no-audit --no-fund \\
    @anthropic-ai/claude-code@{EXPECTED_CC_VERSION}
fi
printf '\\nPINNED_CC_VERSION='
claude --version
/usr/bin/python3 -c 'import shutil; free=shutil.disk_usage("/home/user").free; print("GUEST_FREE_BYTES="+str(free)); assert free >= 20*1024**3, "Large sandbox has less than 20 GiB free"'
"""

# Codex prepares its own CLI/Node. Keep the same disk and guest permissions
# contract, without making a Codex task depend on Claude Code installation.
CODEX_BOOTSTRAP = """
set -euo pipefail
/usr/bin/python3 -c 'import shutil; free=shutil.disk_usage("/home/user").free; print("GUEST_FREE_BYTES="+str(free)); assert free >= 20*1024**3, "Large sandbox has less than 20 GiB free"'
"""


# The AGS gateway closes an idle sandbox connection after ~120s, so a single
# blocking /execute call cannot carry the multi-minute bootstrap: the pinned
# install measured 92s on a good run and 122s (a hard cut) on every bad one.
# Run the script detached on the guest and poll with short requests instead, so
# no individual request ever approaches the gateway limit.
_BOOTSTRAP_DIR = '/tmp/nexus_osworld_large_bootstrap'
_BOOTSTRAP_SH = f'{_BOOTSTRAP_DIR}/bootstrap.sh'
_BOOTSTRAP_LOG = f'{_BOOTSTRAP_DIR}/bootstrap.log'
_BOOTSTRAP_RC = f'{_BOOTSTRAP_DIR}/bootstrap.rc'
_STEP_TIMEOUT_SECONDS = 60.0
_POLL_INTERVAL_SECONDS = 5.0
_BOOTSTRAP_TIMEOUT_SECONDS = 900.0
_BOOTSTRAP_ATTEMPTS = 3
_BOOTSTRAP_RETRY_BASE_DELAY_SECONDS = 5.0


async def _run_detached(
    runtime: Runtime,
    script: str,
    *,
    overall_timeout: float = _BOOTSTRAP_TIMEOUT_SECONDS,
    poll_interval: float = _POLL_INTERVAL_SECONDS,
) -> tuple[int | None, str, str]:
    """Run *script* on the guest without holding one long HTTP request.

    Returns the same ``(rc, stdout, stderr)`` shape as ``_run`` so the callers'
    validation contract is unchanged.
    """
    encoded = base64.b64encode(script.encode()).decode()
    # A stale sentinel from an earlier attempt must never be mistaken for this run.
    prepare = (
        f'rm -rf {_BOOTSTRAP_DIR} && mkdir -p {_BOOTSTRAP_DIR} && '
        f'printf %s {shlex.quote(encoded)} | base64 -d > {_BOOTSTRAP_SH}'
    )
    rc, _, err = await _run(runtime, prepare, timeout=_STEP_TIMEOUT_SECONDS)
    if rc != 0:
        raise RuntimeError(f'V1 large sandbox bootstrap upload failed rc={rc} {err[:200]}')

    inner = f'bash {_BOOTSTRAP_SH} > {_BOOTSTRAP_LOG} 2>&1; echo $? > {_BOOTSTRAP_RC}'
    launch = f'setsid bash -c {shlex.quote(inner)} > /dev/null 2>&1 & echo LAUNCHED'
    rc, out, err = await _run(runtime, launch, timeout=_STEP_TIMEOUT_SECONDS)
    if rc != 0 or 'LAUNCHED' not in out:
        raise RuntimeError(f'V1 large sandbox bootstrap launch failed rc={rc} {err[:200]}')

    loop = asyncio.get_running_loop()
    deadline = loop.time() + overall_timeout
    while True:
        poll_rc, poll_out, _ = await _run(
            runtime,
            f'if [ -f {_BOOTSTRAP_RC} ]; then printf "RC=%s" "$(cat {_BOOTSTRAP_RC})"; else printf RUNNING; fi',
            timeout=_STEP_TIMEOUT_SECONDS,
        )
        if poll_rc == 0 and poll_out.strip().startswith('RC='):
            script_rc = int(poll_out.strip().removeprefix('RC=').strip() or -1)
            break
        if loop.time() >= deadline:
            raise TimeoutError(f'V1 large sandbox bootstrap exceeded {overall_timeout:g}s')
        await asyncio.sleep(poll_interval)

    _, log_out, _ = await _run(runtime, f'cat {_BOOTSTRAP_LOG} 2>/dev/null', timeout=_STEP_TIMEOUT_SECONDS)
    return script_rc, log_out, ''


@runtime_provider_registry.register(LARGE_PROVIDER)
class OSWorldV1LargeRuntimeProvider(AGSRuntimeProvider):
    def __init__(self, *, cli_harness: str = 'cc', **config: Any) -> None:
        if cli_harness not in {'cc', 'codex', 'bare'}:
            raise ValueError('Large sandbox cli_harness must be cc, codex or bare')
        super().__init__(**config)
        self.cli_harness = cli_harness
        self._bootstrap_locks: dict[str, asyncio.Lock] = {}
        self._bootstrapped: dict[str, Runtime] = {}

    async def acquire(self, name: str, **params: Any) -> Runtime:
        if self.get_params(name, **params).get('tool_id') != LARGE_TOOL_ID:
            raise ValueError('V1 large bootstrap is restricted to the validated large template')
        lock = self._bootstrap_locks.setdefault(name, asyncio.Lock())
        async with lock:
            runtime = await super().acquire(name, **params)
            if self._bootstrapped.get(name) is runtime:
                return runtime
            try:
                await _prepare_v1_guest(runtime)
                await self._bootstrap_with_retries(runtime)
                self._bootstrapped[name] = runtime
                logger.info('V1 large sandbox ready: harness=%s; no task packages installed', self.cli_harness)
                return runtime
            except BaseException:
                with contextlib.suppress(BaseException):
                    await asyncio.shield(self.release(name))
                raise

    async def _bootstrap_with_retries(self, runtime: Runtime) -> None:
        """Install the pinned toolchain, tolerating a transient guest/gateway failure.

        The bootstrap downloads Node and the pinned CC, so it sits close to the
        gateway's idle limit and occasionally loses the connection. Retrying here
        reuses the sandbox that is already running; letting the whole task fail
        instead would re-create it, which is slower and no more likely to pass.
        """
        script = BOOTSTRAP if self.cli_harness == 'cc' else CODEX_BOOTSTRAP
        last_error: BaseException | None = None
        for attempt in range(1, _BOOTSTRAP_ATTEMPTS + 1):
            try:
                rc, out, _ = await _run_detached(runtime, script)
                if self.cli_harness == 'cc':
                    if rc != 0 or f'PINNED_CC_VERSION={EXPECTED_CC_VERSION} (Claude Code)' not in out:
                        raise RuntimeError(f'V1 large sandbox pinned CC bootstrap failed rc={rc}')
                elif rc != 0 or 'GUEST_FREE_BYTES=' not in out:
                    raise RuntimeError(f'V1 large sandbox disk validation failed rc={rc}')
                return
            except (RuntimeError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt == _BOOTSTRAP_ATTEMPTS:
                    break
                delay = _BOOTSTRAP_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    'V1 large sandbox bootstrap attempt %d/%d failed (%s); retrying in %.0fs',
                    attempt,
                    _BOOTSTRAP_ATTEMPTS,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
        assert last_error is not None
        raise last_error

    async def _release_impl(self, name: str) -> None:
        self._bootstrapped.pop(name, None)
        await super()._release_impl(name)
