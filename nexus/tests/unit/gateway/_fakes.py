"""Shared fakes for gateway unit tests: deterministic tokenizer, endpoints, backends."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from nexus.gateway.trajectory_store import InMemoryTrajectoryRecordStore
from nexus.gateway.types import BackendResult


def memory_store(namespace: str = 'gateway-test') -> InMemoryTrajectoryRecordStore:
    return InMemoryTrajectoryRecordStore(namespace)


class FakeTokenizer:
    """Bijective char-level tokenizer with a deterministic chat template.

    The template wraps every message as ``<role>content[tc:...]</role>\\n`` and appends
    ``<assistant>`` for generation prompts. Re-rendering a history therefore appends
    closing glue (``</assistant>\\n``) that is NOT part of
    ``rendered_prompt + completion_text`` — reproducing the template drift that the
    re-render prompt cache (``ProcessorResult.cache_rendered_text``) exists to absorb.
    """

    eos_token = '§'

    def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> str:
        del tokenize, kwargs
        parts = []
        if tools:
            parts.append(f'[tools:{len(tools)}]\n')
        for message in messages:
            role = message.get('role')
            content = message.get('content') or ''
            tool_calls = message.get('tool_calls') or []
            tool_part = ''
            if tool_calls:
                names = ','.join(call['function']['name'] for call in tool_calls)
                tool_part = f'[tc:{names}]'
            parts.append(f'<{role}>{content}{tool_part}</{role}>\n')
        if add_generation_prompt:
            parts.append('<assistant>')
        return ''.join(parts)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return ''.join(chr(i) for i in token_ids)


class StubToolParser:
    """Minimal tool parser for HyTunerClient: never parses tool calls."""

    think_start_token = '<think>'
    think_end_token = '</think>'

    def __init__(self, tokenizer: FakeTokenizer):
        self._tokenizer = tokenizer

    async def extract_tool_calls(self, token_ids: list[int], request: Any = None) -> tuple[str, str, list[Any]]:
        del request
        return '', self._tokenizer.decode(token_ids), []


class RecordingEndpoint:
    """Duck-typed generation endpoint recording per-call sampling params.

    Satisfies both HyTunerClient's ``server_manager`` protocol and the thin
    ``HyTunerServerBackend`` endpoint protocol. Scripted items are completion texts or
    ``(text, stop_reason)`` tuples, encoded with the fake tokenizer so that
    decode() round-trips exactly.
    """

    def __init__(
        self,
        tokenizer: FakeTokenizer,
        scripted: list[str | tuple[str, str]],
        extra_fields: dict[str, Any] | None = None,
    ):
        self._tokenizer = tokenizer
        self._queue: list[str | tuple[str, str]] = list(scripted)
        self._extra_fields = dict(extra_fields or {})
        self.max_tokens_per_call: list[int | None] = []
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        request_id: str | None = None,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Any = None,
        video_data: Any = None,
        audio_data: Any = None,
        priority: int = 0,
        **kwargs: Any,
    ) -> SimpleNamespace:
        del image_data, video_data, audio_data, kwargs
        self.max_tokens_per_call.append(sampling_params.get('max_tokens'))
        self.calls.append(
            {
                'request_id': request_id,
                'prompt_ids': list(prompt_ids),
                'sampling_params': dict(sampling_params),
                'priority': priority,
            }
        )
        item = self._queue.pop(0)
        text, stop_reason = item if isinstance(item, tuple) else (item, 'stop')
        return SimpleNamespace(
            token_ids=self._tokenizer.encode(text, add_special_tokens=False),
            log_probs=None,
            routed_experts=None,
            stop_reason=stop_reason,
            extra_fields=dict(self._extra_fields),
            num_preempted=0,
        )

    # Thin-backend protocol name (HyTunerServerBackend calls ``generate_tokens``);
    # the legacy HyTunerClient protocol keeps calling ``generate`` above.
    generate_tokens = generate


class ScriptedTokenBackend:
    """Token-level InferenceBackend stub returning scripted completions."""

    def __init__(
        self,
        tokenizer: FakeTokenizer,
        scripted_texts: list[str],
        extra_fields: dict[str, Any] | None = None,
    ):
        self._tokenizer = tokenizer
        self._queue = list(scripted_texts)
        self._extra_fields = dict(extra_fields or {})
        self.calls: list[dict[str, Any]] = []

    async def create_completion(
        self,
        request_id: str,
        *,
        prompt_ids: list[int] | None = None,
        messages: list[dict] | None = None,
        tools: list[dict] | None = None,
        sampling_params: dict[str, Any] | None = None,
        image_data: Any | None = None,
        video_data: Any | None = None,
        audio_data: Any | None = None,
        priority: int = 0,
        request_meta: dict[str, Any] | None = None,
    ) -> BackendResult:
        del messages, tools, image_data, video_data, audio_data, priority
        self.calls.append(
            {
                'prompt_ids': list(prompt_ids or []),
                'sampling_params': dict(sampling_params or {}),
                'request_meta': dict(request_meta or {}),
            }
        )
        text = self._queue.pop(0)
        return BackendResult(
            token_ids=self._tokenizer.encode(text, add_special_tokens=False),
            stop_reason='stop',
            extra_fields=dict(self._extra_fields),
        )
