from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import textwrap
import threading

import pytest

from nexus.utils.log.handlers import SocketIOHandler


class RecordingSio:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def emit(self, _event_name: str, payload: dict) -> None:
        self.payloads.append(payload)


def test_tqdm_stream_handler_requires_tqdm_only_when_emitting() -> None:
    script = textwrap.dedent(
        """
        import builtins
        import logging
        import sys

        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == 'tqdm' or name.startswith('tqdm.'):
                raise ModuleNotFoundError("No module named 'tqdm'", name='tqdm')
            return real_import(name, *args, **kwargs)

        builtins.__import__ = guarded_import

        from nexus.utils.log.handlers import TqdmStreamHandler

        handler = TqdmStreamHandler()
        try:
            handler.emit(logging.LogRecord('test', logging.INFO, '<test>', 1, 'hello', (), None))
        except ModuleNotFoundError as exc:
            assert exc.name == 'tqdm'
        else:
            raise AssertionError('TqdmStreamHandler.emit() did not require tqdm')

        sys.stdout.write('tqdm required at emit time\\n')
        """
    )

    completed = subprocess.run([sys.executable, '-c', script], check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == 'tqdm required at emit time\n'


@pytest.mark.asyncio
async def test_socketio_handler_does_not_wait_for_owner_loop_from_worker_thread() -> None:
    sio = RecordingSio()
    handler = SocketIOHandler(sio)  # type: ignore[arg-type]
    await handler.start_async()
    entered = threading.Event()

    def emit_log() -> None:
        entered.set()
        handler.emit(logging.LogRecord('worker', logging.INFO, __file__, 1, 'hello', (), None))

    thread = threading.Thread(target=emit_log)
    thread.start()
    assert entered.wait(timeout=1)

    # Keep the owner loop paused while the worker emits. A logging handler must
    # not wait for that loop, because the owner may itself be joining the worker.
    thread.join(timeout=0.2)
    completed_while_owner_loop_was_paused = not thread.is_alive()

    await asyncio.sleep(0)
    thread.join(timeout=1)
    await handler.close_async()

    assert completed_while_owner_loop_was_paused is True
    assert sio.payloads[0]['items'][0]['message'] == 'hello'
