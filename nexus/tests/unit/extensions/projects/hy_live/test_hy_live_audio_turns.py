"""A scripted audio turn has to reach the policy as a content part, not as literal text.

The speech dataset attaches ``{'type': 'audio', 'audio': <patch>}`` to the opening turn,
and the rollout client only recognizes audio in that shape. A later turn that arrived as a
stringified dict would render as text and silently drop the waveform.
"""

from __future__ import annotations

import copy

from types import SimpleNamespace
from typing import Any

import pytest

from nexus.extensions.projects.hy_live import agent as agent_module
from nexus.trace import observe


FIRST_AUDIO = {'type': 'audio', 'audio': {'patch': '/data/audios/turn1.wav', 'audio_length': 101120}}
SECOND_AUDIO = {'type': 'audio', 'audio': {'patch': '/data/audios/turn2.wav', 'audio_length': 94720}}


class FakePolicyClient:
    _client_key = 'main'

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    @observe('openai_completion', self_attrs=['_client_key'])
    async def create_completion(self, messages: list[dict[str, Any]], **kwargs: Any) -> SimpleNamespace:
        self.calls.append(copy.deepcopy({'messages': messages, **kwargs}))
        return SimpleNamespace(choices=[SimpleNamespace(message=self.responses.pop(0))])

    def record_observation(self, **kwargs):
        return None


class WarmRuntime:
    async def run_command(self, command: str, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(return_code=0, stdout='', stderr='', status='done')


def _install_policy(monkeypatch: pytest.MonkeyPatch, policy: FakePolicyClient, run_id: str) -> None:
    async def get_runtime() -> Any:
        return WarmRuntime()

    context = SimpleNamespace(run_id=run_id, get_openai_client=lambda name: policy, get_runtime=get_runtime)
    monkeypatch.setattr(agent_module, 'require_context', lambda: context)
    monkeypatch.setattr(agent_module, 'get_current_span_id', lambda: None)


def _user_messages(messages: list[dict[str, Any]]) -> list[Any]:
    return [message['content'] for message in messages if message['role'] == 'user']


async def test_audio_script_reaches_the_policy_as_content_parts(monkeypatch) -> None:
    policy = FakePolicyClient([
        {'role': 'assistant', 'content': '听到了，我看看。'},
        {'role': 'assistant', 'content': '好，这次也记下了。'},
    ])
    _install_policy(monkeypatch, policy, 'run-audio')

    result = await agent_module.live_agent(
        agent_module.LiveAgentAgentInput.model_validate({
            'user_simulator': {'type': 'scripted', 'turns': [FIRST_AUDIO, SECOND_AUDIO]},
            'max_user_turns': 2,
        })
    )

    assert _user_messages(policy.calls[0]['messages']) == [[FIRST_AUDIO]]
    assert _user_messages(policy.calls[1]['messages']) == [[FIRST_AUDIO], [SECOND_AUDIO]]
    assert _user_messages(result.main_trajectory) == [[FIRST_AUDIO], [SECOND_AUDIO]]
    assert [turn.user_message for turn in result.conversation.turns] == [[FIRST_AUDIO], [SECOND_AUDIO]]
