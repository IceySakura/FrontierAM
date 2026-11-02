import sys
import types

import pytest


def _stub_fastapi():
    for name, attrs in {
        'fastapi': {'FastAPI': object, 'HTTPException': Exception, 'Request': object},
        'fastapi.responses': {'JSONResponse': object, 'StreamingResponse': object},
        'uvicorn': {'Config': object, 'Server': object},
    }.items():
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules.setdefault(name, mod)


_stub_fastapi()
from nexus.runtime.server.services.rsio_llm_bridge import RSIOLLMBridge  # noqa: E402


class ReplacingSIO:
    """Keeps only the last handler per event, the way python-socketio does.

    Counts registrations separately, so a test can tell "registered once" from
    "registered four times and three were replaced".
    """

    def __init__(self):
        self.handlers = {}
        self.registrations = {}

    def on(self, event, handler=None):
        def deco(fn):
            self.handlers[event] = fn
            self.registrations[event] = self.registrations.get(event, 0) + 1
            return fn

        return deco if handler is None else deco(handler)

    async def emit(self, event, data=None):
        pass


def _respond_to_every_request(sio):
    """Answer each llm_request through whichever handler is registered."""

    async def emit(event, data=None):
        if event != 'llm_request':
            return
        await sio.handlers['llm_response']({
            'request_id': data['request_id'],
            'success': True,
            'response': {'echo': data['request_id']},
            'rsio': {'request_id': data['request_id'], 'kind': 'final'},
        })

    sio.emit = emit


def test_one_connection_registers_one_handler():
    """A second registration would replace the first, leaving its bridge unanswered."""
    sio = ReplacingSIO()
    for _ in range(4):
        RSIOLLMBridge(sio)

    assert sio.registrations['llm_response'] == 1


def test_bridges_on_one_connection_share_a_pending_table():
    """The single handler dispatches by request_id, so every bridge's calls must be in
    the table it reads."""
    sio = ReplacingSIO()
    first, second = RSIOLLMBridge(sio), RSIOLLMBridge(sio)

    assert first is not second
    assert first._pending is second._pending


def test_separate_connections_keep_separate_tables():
    first, second = ReplacingSIO(), ReplacingSIO()

    assert RSIOLLMBridge(first)._pending is not RSIOLLMBridge(second)._pending


@pytest.mark.asyncio
async def test_every_bridge_on_one_connection_resolves_its_own_calls():
    """The failure this pins: a bridge whose registration was replaced never sees its
    response, resends every 5s until the deadline, and the host re-runs the model on
    each resend."""
    sio = ReplacingSIO()
    bridges = [RSIOLLMBridge(sio) for _ in range(4)]
    _respond_to_every_request(sio)

    for bridge in bridges:
        result = await bridge.request({'model': 'm', 'messages': []}, timeout=2)
        assert result['echo']
