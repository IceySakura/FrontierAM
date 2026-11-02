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
from nexus.runtime.server.services.openai_proxy import LLMBridge  # noqa: E402
from nexus.runtime.server.services.rsio_llm_bridge import RSIOLLMBridge, RSIOLLMConfig  # noqa: E402


class FakeSIO:
    def __init__(self):
        self._h = {}
        self.requests = []

    def on(self, event, handler=None):
        def deco(fn):
            self._h.setdefault(event, []).append(fn)
            return fn

        return deco if handler is None else deco(handler)

    async def emit(self, event, data=None):
        if event == 'llm_request':
            self.requests.append(data)
            rid = data['request_id']
            for h in self._h.get('llm_response', []):
                await h({
                    'request_id': rid,
                    'success': True,
                    'response': {'ok': 1},
                    'rsio': {'request_id': rid, 'kind': 'final'},
                })


@pytest.mark.asyncio
async def test_openai_bridge_is_rsio_subclass():
    assert issubclass(LLMBridge, RSIOLLMBridge)


@pytest.mark.asyncio
async def test_bridge_with_config_sends_rsio():
    sio = FakeSIO()
    bridge = LLMBridge(sio, RSIOLLMConfig())
    result = await bridge.request({'model': 'm', 'messages': []}, timeout=2)
    assert result == {'ok': 1}
    assert 'rsio' in sio.requests[0]  # rsio mode → metadata present


@pytest.mark.asyncio
async def test_bridge_without_config_defaults_to_rsio():
    # No config ⇒ default RSIOLLMConfig ⇒ RSIO mode (enable=True default).
    sio = FakeSIO()
    bridge = LLMBridge(sio)
    result = await bridge.request({'model': 'm', 'messages': []}, timeout=2)
    assert result == {'ok': 1}
    assert 'rsio' in sio.requests[0]


@pytest.mark.asyncio
async def test_bridge_enable_false_is_legacy():
    sio = FakeSIO()
    bridge = LLMBridge(sio, RSIOLLMConfig(enable=False))
    result = await bridge.request({'model': 'm', 'messages': []}, timeout=2)
    assert result == {'ok': 1}
    assert 'rsio' not in sio.requests[0]


@pytest.mark.asyncio
async def test_bridge_env_var_disables_rsio(monkeypatch):
    # NEXUS_DISABLE_LLM_RSIO on the runtime forces legacy mode — the escape
    # hatch for an old host driving a new runtime.
    monkeypatch.setenv('NEXUS_DISABLE_LLM_RSIO', '1')
    sio = FakeSIO()
    bridge = LLMBridge(sio, RSIOLLMConfig())
    result = await bridge.request({'model': 'm', 'messages': []}, timeout=2)
    assert result == {'ok': 1}
    assert 'rsio' not in sio.requests[0]

