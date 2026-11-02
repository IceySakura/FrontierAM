"""PTY session driver for interactive Claude Code (framework-agnostic).

Self-contained: knows nothing about nexus, socket.io, or the LLM bridge. Only
spawns a command behind a PTY, types prompts, auto-confirms the workspace-trust
prompt, and detects per-turn completion via a Stop-hook sentinel with turn_id.

Two entry points:
  * in-process — ``start()`` / ``send_turn(prompt)`` / ``close()``
  * CLI (``main``) — single-turn compat wrapper (prompt via stdin)
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import json
import logging
import os
import pathlib
import re
import signal
import struct
import sys
import termios
import time

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .screen import TerminalScreen


if TYPE_CHECKING:
    from .hooks import HookInteractionBridge


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# PtyProcess — async PTY spawner with non-blocking I/O pump.
# --------------------------------------------------------------------------- #


def _set_winsize(fd: int, rows: int, cols: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack('HHHH', rows, cols, 0, 0))


class PtyProcess:
    def __init__(self, proc, master, on_data, on_exit):
        self.proc = proc
        self.master = master
        self.shell_pgid = os.getpgid(proc.pid)
        self._on_data = on_data
        self._on_exit = on_exit
        self._closed = False
        self._write_lock = asyncio.Lock()
        self._write_fut = None
        self._loop = asyncio.get_running_loop()
        self._loop.add_reader(master, self._read)
        self._exit_task = asyncio.create_task(self._watch_exit())

    @classmethod
    async def spawn(
        cls,
        argv: list[str],
        env: dict[str, str],
        rows: int,
        cols: int,
        cwd: str,
        on_data: Callable[[bytes], None],
        on_exit: Callable[[], None],
    ) -> PtyProcess:
        import pty

        master, slave = pty.openpty()
        _set_winsize(master, rows, cols)

        flags = fcntl.fcntl(master, fcntl.F_GETFL)
        fcntl.fcntl(master, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        def _preexec() -> None:
            os.setsid()
            fcntl.ioctl(slave, termios.TIOCSCTTY, 0)

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            preexec_fn=_preexec,
            env=env,
            cwd=cwd,
        )
        os.close(slave)
        return cls(proc, master, on_data, on_exit)

    def _read(self) -> None:
        try:
            data = os.read(self.master, 65536)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            data = b''
        if data:
            self._on_data(data)
        else:
            with contextlib.suppress(ValueError, OSError):
                self._loop.remove_reader(self.master)

    async def _watch_exit(self) -> None:
        await self.proc.wait()
        self._on_exit()

    async def write(self, data: bytes) -> None:
        if not data:
            return
        loop = self._loop
        async with self._write_lock:
            fut = loop.create_future()
            self._write_fut = fut
            mv = memoryview(data)
            state = {'off': 0}

            def _on_writable() -> None:
                try:
                    while state['off'] < len(mv):
                        try:
                            n = os.write(self.master, mv[state['off'] :])
                        except (BlockingIOError, InterruptedError):
                            return
                        except OSError as exc:
                            loop.remove_writer(self.master)
                            if not fut.done():
                                fut.set_exception(exc)
                            return
                        state['off'] += n
                    loop.remove_writer(self.master)
                    if not fut.done():
                        fut.set_result(None)
                except Exception as exc:  # noqa: BLE001
                    loop.remove_writer(self.master)
                    if not fut.done():
                        fut.set_exception(exc)

            loop.add_writer(self.master, _on_writable)
            try:
                await fut
            finally:
                with contextlib.suppress(ValueError, OSError):
                    loop.remove_writer(self.master)
                self._write_fut = None

    def killpg(self, sig: int = signal.SIGKILL) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.shell_pgid, sig)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for remove in (self._loop.remove_reader, self._loop.remove_writer):
            with contextlib.suppress(ValueError, OSError):
                remove(self.master)
        if self._write_fut is not None and not self._write_fut.done():
            self._write_fut.set_exception(OSError('pty closed during write'))
        with contextlib.suppress(OSError):
            os.close(self.master)
        if not self._exit_task.done():
            self._exit_task.cancel()


# --------------------------------------------------------------------------- #
# Screen text cleaning + trust prompt + terminal capability auto-reply.
# --------------------------------------------------------------------------- #

_ANSI_RE = re.compile(
    r'\x1b\[[0-?]*[ -/]*[@-~]'
    r'|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)'
    r'|\x1b[@-Z\\-_]'
)

_TRUST_RE = re.compile(
    r'(yes,?\s*i\s*trust\s*this\s*folder|trust\s*this\s*folder|do you trust)',
    re.IGNORECASE,
)

_OSC_FG_QUERY = b'\x1b]10;?\x07'
_OSC_BG_QUERY = b'\x1b]11;?\x07'
_OSC_FG_REPLY = b'\x1b]10;rgb:e6e6/eded/f3f3\x1b\\'
_OSC_BG_REPLY = b'\x1b]11;rgb:1717/1717/2121\x1b\\'

_DA_QUERY = re.compile(rb'\x1b\[(?:\?|>)?[0-9;]*c')
_DA_REPLY = b'\x1b[?1;2c'
_CPR_QUERY = re.compile(rb'\x1b\[6n')
_DSR_QUERY = re.compile(rb'\x1b\[5n')
_DSR_REPLY = b'\x1b[0n'


def _clean(text: str) -> str:
    return _ANSI_RE.sub('', text).replace('\r', '')


# --------------------------------------------------------------------------- #
# ClaudeInteractiveSession — long-lived, multi-turn PTY driver.
# --------------------------------------------------------------------------- #


class ClaudeInteractiveSession:
    """Drive one long-lived interactive Claude Code TUI across multiple turns."""

    def __init__(
        self,
        *,
        claude_path: str,
        cwd: str,
        env: dict[str, str],
        done_file: str,
        final_file: str | None = None,
        turn_file: str | None = None,
        rows: int = 50,
        cols: int = 200,
        settle_seconds: float = 4.0,
        per_turn_timeout: float = 3600.0,
        snapshot_seconds: float | None = None,
        log_frames: bool = False,
        frames_path: str | None = None,
        extra_args: list[str] | None = None,
        type_prompt: bool = False,
        interaction_bridge: HookInteractionBridge | None = None,
    ) -> None:
        self.claude_path = claude_path
        self.cwd = cwd
        self.env = dict(env)
        self.done_file = done_file
        self.final_file = final_file
        self.turn_file = turn_file
        self.rows = rows
        self.cols = cols
        self.settle_seconds = settle_seconds
        self.per_turn_timeout = per_turn_timeout
        if snapshot_seconds is not None and snapshot_seconds <= 0:
            raise ValueError('snapshot_seconds must be greater than zero when enabled')
        self.snapshot_seconds = snapshot_seconds
        self.log_frames = log_frames
        self.frames_path = frames_path
        self.extra_args = list(extra_args or [])
        self.type_prompt = type_prompt
        self.interaction_bridge = interaction_bridge

        self.pty: PtyProcess | None = None
        self._turn = 0
        self._t0: float | None = None
        self._frame = 0
        self._tail = bytearray()
        self._exited = asyncio.Event()
        self._trust_done = False
        self._closed = False
        self._pending: set[asyncio.Task] = set()
        self._active_turn_id: int | None = None
        self._active_turn_started: float | None = None
        self._active_turn_deadline: float | None = None

        self._terminal = TerminalScreen(rows, cols)

    @property
    def claude_pid(self) -> int | None:
        return self.pty.proc.pid if self.pty is not None else None

    # ----- lifecycle -----

    async def start(self) -> None:
        self._t0 = time.time()
        for path in (self.done_file, self.final_file, self.turn_file):
            if path:
                with contextlib.suppress(OSError):
                    os.unlink(path)

    async def _spawn(self, first_prompt: str) -> None:
        argv = (
            [self.claude_path, first_prompt, *self.extra_args]
            if not self.type_prompt and first_prompt.strip()
            else [self.claude_path, *self.extra_args]
        )
        self.pty = await PtyProcess.spawn(
            argv=argv,
            env=self.env,
            rows=self.rows,
            cols=self.cols,
            cwd=self.cwd,
            on_data=self._on_data,
            on_exit=self._on_exit,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.pty is not None and not self._exited.is_set():
            try:
                await asyncio.wait_for(self.pty.write(b'\x03'), timeout=0.5)
                await asyncio.sleep(0.2)
                await asyncio.wait_for(self.pty.write(b'\x03'), timeout=0.5)
                await asyncio.sleep(0.3)
            except (TimeoutError, OSError):
                pass
        if self.pty is not None:
            self.pty.killpg()
            self.pty.close()
        if self.interaction_bridge is not None:
            with contextlib.suppress(Exception):
                await self.interaction_bridge.close()
        for path in (self.done_file, self.final_file, self.turn_file):
            if path:
                with contextlib.suppress(OSError):
                    os.unlink(path)

    # ----- PTY data pump -----

    def _spawn_write(self, data: bytes) -> None:
        if self.pty is None or not data:
            return
        task = asyncio.create_task(self._safe_write(data))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _safe_write(self, data: bytes) -> None:
        if self.pty is None:
            return
        with contextlib.suppress(OSError):
            await self.pty.write(data)

    def _reply(self, data: bytes) -> None:
        out = b''
        if _OSC_FG_QUERY in data:
            out += _OSC_FG_REPLY
        if _OSC_BG_QUERY in data:
            out += _OSC_BG_REPLY
        if _DA_QUERY.search(data):
            out += _DA_REPLY
        if _DSR_QUERY.search(data):
            out += _DSR_REPLY
        if _CPR_QUERY.search(data):
            out += f'\x1b[{self.rows};1R'.encode()
        if out:
            self._spawn_write(out)

    def _on_data(self, data: bytes) -> None:
        self._reply(data)

        if self.log_frames:
            self._frame += 1
            clean = re.sub(r'\s+', ' ', _clean(data.decode('utf-8', 'replace'))).strip()
            if clean:
                dt = 0.0 if self._t0 is None else (time.time() - self._t0)
                line = f'[frame {self._frame} t+{dt:.2f}s {len(data)}B] {clean[:200]}'
                sys.stderr.write(line + '\n')
                sys.stderr.flush()
                if self.frames_path:
                    with contextlib.suppress(OSError):
                        with open(self.frames_path, 'a', encoding='utf-8') as fh:
                            fh.write(line + '\n')

        self._tail.extend(data)
        if len(self._tail) > 65536:
            del self._tail[:-65536]

        with contextlib.suppress(Exception):
            self._terminal.feed(data)

        if not self._trust_done:
            cleaned = ' '.join(_clean(self._tail.decode('utf-8', 'replace')).split())
            if _TRUST_RE.search(cleaned):
                self._trust_done = True
                task = asyncio.create_task(self._confirm_trust())
                self._pending.add(task)
                task.add_done_callback(self._pending.discard)

    async def _confirm_trust(self) -> None:
        await asyncio.sleep(0.15)
        if self.pty is not None:
            with contextlib.suppress(OSError):
                await self.pty.write(b'\r')

    def _on_exit(self) -> None:
        self._exited.set()

    def screen_tail(self) -> str:
        return _clean(self._tail.decode('utf-8', 'replace'))

    def latest_screen(self) -> str:
        """Current terminal viewport as plain text via pyte emulator."""
        return self._terminal.render()

    # ----- turn driving -----

    def _read_completion(self, expected_turn: int) -> tuple[bool, str | None]:
        if not self.done_file or not os.path.exists(self.done_file):
            return False, None
        try:
            raw = pathlib.Path(self.done_file).read_text(encoding='utf-8')
        except OSError:
            return False, None
        if not raw.strip():
            return False, None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return False, None
        try:
            turn_id = int(data.get('turn_id', -1))
        except (TypeError, ValueError):
            return False, None
        if turn_id != expected_turn:
            return False, None
        msg = data.get('last_assistant_message')
        if not msg and self.final_file:
            with contextlib.suppress(OSError):
                msg = pathlib.Path(self.final_file).read_text(encoding='utf-8')
        return True, msg

    async def _await_input_ready(self) -> None:
        await asyncio.sleep(min(self.settle_seconds, 1.5))

    async def _type_prompt(self, prompt: str) -> None:
        assert self.pty is not None
        await self.pty.write(b'\x1b[200~' + prompt.encode('utf-8') + b'\x1b[201~')
        await asyncio.sleep(0.25)
        await self.pty.write(b'\r')

    def _finish_active_turn(self) -> None:
        self._active_turn_id = None
        self._active_turn_started = None
        self._active_turn_deadline = None

    def _turn_result(
        self,
        *,
        status: str,
        completed: bool,
        reason: str,
        final_message: str = '',
        interaction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = self._active_turn_started or time.time()
        result: dict[str, Any] = {
            'status': status,
            'completed': completed,
            'turn_id': self._active_turn_id,
            'final_message': final_message,
            'screen_tail': self.screen_tail()[-4000:],
            'latest_screen': self.latest_screen(),
            'elapsed': time.time() - started,
            'reason': reason,
            'claude_pid': self.claude_pid,
        }
        if interaction is not None:
            result['interaction'] = interaction
        return result

    async def _wait_for_active_turn(self) -> dict[str, Any]:
        turn_id = self._active_turn_id
        deadline = self._active_turn_deadline
        if turn_id is None or deadline is None:
            raise RuntimeError('No active PTY turn')
        pty = self.pty
        if pty is None:
            raise RuntimeError('PTY process is not running')

        last_snap = 0.0
        while time.time() < deadline:
            ok, msg = self._read_completion(turn_id)
            if ok:
                result = self._turn_result(
                    status='completed',
                    completed=True,
                    reason='stop_hook',
                    final_message=msg or '',
                )
                self._finish_active_turn()
                return result
            if self._exited.is_set():
                result = self._turn_result(status='process_exit', completed=False, reason='process_exit')
                result['exit_code'] = pty.proc.returncode
                self._finish_active_turn()
                return result
            if self.interaction_bridge is not None:
                request = await self.interaction_bridge.wait_request(timeout=0.0)
                if request is not None:
                    return self._turn_result(
                        status='needs_user_input',
                        completed=False,
                        reason='needs_user_input',
                        interaction=request.to_dict(),
                    )
            now = time.time()
            if self.snapshot_seconds is not None and now - last_snap >= self.snapshot_seconds:
                last_snap = now
                snap = self.screen_tail()[-1200:]
                started = self._active_turn_started or now
                logger.debug(
                    '[pty_session snapshot turn=%s t+%.0fs]\n%s\n[/snapshot]',
                    turn_id,
                    now - started,
                    snap,
                )
            await asyncio.sleep(0.5)

        result = self._turn_result(status='timeout', completed=False, reason='timeout')
        self._finish_active_turn()
        return result

    async def send_turn(self, prompt: str, *, timeout: float | None = None) -> dict:
        """Run one user turn; spawns on first call, types into TUI thereafter."""
        if self._closed:
            raise RuntimeError('ClaudeInteractiveSession is closed')
        if self._active_turn_id is not None:
            raise RuntimeError('Cannot send a new prompt while the current PTY turn is waiting for input')
        turn_timeout = self.per_turn_timeout if timeout is None else timeout
        self._turn += 1
        turn_id = self._turn

        if self.turn_file:
            with contextlib.suppress(OSError):
                pathlib.Path(self.turn_file).write_text(str(turn_id), encoding='utf-8')
        for path in (self.done_file, self.final_file):
            if path:
                with contextlib.suppress(OSError):
                    os.unlink(path)

        start = time.time()
        if self.pty is None:
            await self._spawn(prompt)
            if self.type_prompt and prompt.strip():
                await asyncio.sleep(self.settle_seconds)
                await self._type_prompt(prompt)
        else:
            await self._await_input_ready()
            await self._type_prompt(prompt)

        self._active_turn_id = turn_id
        self._active_turn_started = start
        self._active_turn_deadline = start + turn_timeout
        return await self._wait_for_active_turn()

    async def respond_and_wait(
        self,
        interaction_id: str,
        answer: str | None,
    ) -> dict[str, Any]:
        """Answer one Hook request and continue waiting for the same PTY turn."""
        if self._closed:
            raise RuntimeError('ClaudeInteractiveSession is closed')
        if self.interaction_bridge is None:
            raise RuntimeError('This PTY session has no Hook interaction bridge')
        response = await self.interaction_bridge.respond(interaction_id, answer)
        response_status = str(response.get('status') or 'unknown')

        def _reusable(status: str) -> dict[str, Any]:
            """Build a non-fatal, reusable-session result for a stale interaction."""
            result = self._turn_result(status=status, completed=False, reason=status)
            result['session_reusable'] = True
            result['interaction_response_status'] = response_status
            return result

        if response_status == 'unknown':
            return _reusable('interaction_not_found')

        if self._active_turn_id is None:
            return _reusable('interaction_expired' if response_status == 'expired' else 'interaction_not_active')

        result = await self._wait_for_active_turn()
        result['interaction_response_status'] = response_status
        return result


# --------------------------------------------------------------------------- #
# CLI — single-turn compatibility wrapper.
# --------------------------------------------------------------------------- #


async def _drive_single_turn(args: argparse.Namespace, prompt: str) -> dict:
    turn_file = args.turn_file or (args.done_sentinel + '.turn')
    session = ClaudeInteractiveSession(
        claude_path=args.claude,
        cwd=args.cwd,
        env=dict(os.environ),
        done_file=args.done_sentinel,
        final_file=args.final_msg or None,
        turn_file=turn_file,
        rows=args.rows,
        cols=args.cols,
        settle_seconds=args.settle_seconds,
        per_turn_timeout=args.timeout,
        snapshot_seconds=args.snapshot_seconds,
        log_frames=args.log_frames,
        frames_path=args.frames_file or None,
        extra_args=args.extra,
        type_prompt=args.type_prompt,
    )
    await session.start()
    try:
        result = await session.send_turn(prompt)
    finally:
        await session.close()
    return {
        'completed': result['completed'],
        'exit_code': 0 if result['completed'] else 1,
        'final_message': result['final_message'],
        'screen_tail': result['screen_tail'],
        'elapsed': result['elapsed'],
        'reason': result['reason'],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='Black-box PTY runner for Claude Code TUI (single turn)')
    ap.add_argument('--claude', required=True, help='path to the claude binary')
    ap.add_argument('--done-sentinel', required=True, help='file the Stop hook writes the completion JSON to')
    ap.add_argument('--final-msg', default='', help='file the Stop hook writes the last assistant message to')
    ap.add_argument(
        '--turn-file', default='', help='file the driver writes the current turn_id to (default: <done>.turn)'
    )
    ap.add_argument('--cwd', default='/testbed')
    ap.add_argument('--rows', type=int, default=50)
    ap.add_argument('--cols', type=int, default=200)
    ap.add_argument('--settle-seconds', type=float, default=4.0)
    ap.add_argument('--timeout', type=float, default=3600.0)
    ap.add_argument(
        '--snapshot-seconds',
        type=float,
        default=None,
        help='enable debug screen snapshots at this interval (disabled by default)',
    )
    ap.add_argument('--log-frames', action='store_true', help='log every PTY output frame to stderr')
    ap.add_argument('--frames-file', default='', help='also append per-frame lines to this file (tailable live)')
    ap.add_argument(
        '--type-prompt',
        action='store_true',
        help='type the prompt into the TUI after launch instead of passing it as a trailing arg',
    )
    ap.add_argument('extra', nargs='*', help='extra args passed to claude (after --)')
    args = ap.parse_args()

    if args.snapshot_seconds is not None:
        logging.basicConfig(level=logging.DEBUG)

    prompt = sys.stdin.read()

    try:
        result = asyncio.run(_drive_single_turn(args, prompt))
        rc = result['exit_code']
    except Exception as exc:  # noqa: BLE001
        import traceback

        result = {
            'completed': False,
            'exit_code': 2,
            'final_message': '',
            'screen_tail': '',
            'elapsed': 0.0,
            'reason': f'runner_error: {exc}\n{traceback.format_exc()}',
        }
        rc = 2

    print('\n__PTY_RUNNER_RESULT__ ' + json.dumps(result, ensure_ascii=False))
    return rc


__all__ = ['ClaudeInteractiveSession', 'PtyProcess', 'main']


if __name__ == '__main__':
    raise SystemExit(main())
