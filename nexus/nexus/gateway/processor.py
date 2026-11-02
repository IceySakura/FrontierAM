"""Processor: model I/O with two internal paths.

Token-level path (training / vLLM replica):
    apply_chat_template -> tokenize -> backend.create_completion(prompt_ids=...) -> decode -> parse

Message-level path (pure inference / OpenAI API):
    backend.create_completion(messages=...) -> pass-through

Path selection: determined by whether a tokenizer is provided at construction time.

Both paths first resolve media references the backend's consumer cannot read (see
``nexus.gateway.media``), so every backend, current or future, gets host-file handling
from its ``accepted_media_sources`` declaration alone.
"""

import asyncio
import inspect
import logging
import os
import time

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from nexus.core.openai_client import UNPARSED_TOOL_CALL_FIELD
from nexus.gateway.backend import InferenceBackend
from nexus.gateway.media import REMOTE_CONSUMER_SOURCES, MediaResolver
from nexus.gateway.mm_expansion import (
    GatewayMultiModalProtocol,
    partition_expanded_ids,
)
from nexus.gateway.types import (
    TOKEN_ID_DTYPE,
    BackendResult,
    FrozenPrefix,
    ProcessorResult,
    PromptRender,
    RequestMultiModal,
)
from nexus.utils.tool_call_ids import ToolCallIdGenerator


logger = logging.getLogger(__name__)


_PrepareGeneration = Callable[[np.ndarray, np.ndarray, dict[str, Any], bool], dict[str, Any]]


@dataclass(frozen=True)
class _ToolParseRequest:
    tools: list[dict[str, Any]] | None
    chat_template_kwargs: dict[str, Any]


class Processor:
    """Model I/O processor with two paths.

    The path is determined at construction time based on whether a tokenizer is provided.
    Token-level path handles template/tokenize/decode/parse; message-level path is pass-through.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        # Token-level deps (optional — determines path)
        tokenizer=None,
        tool_parser=None,
        template_parser=None,
        apply_chat_template_kwargs: dict[str, Any] | None = None,
        mm_plugin: GatewayMultiModalProtocol | None = None,
        # Shared
        priority: int = 0,
        enable_rollout_routing_replay: bool = False,
    ):
        self._backend = backend
        self._tokenizer = tokenizer
        self._tool_parser = tool_parser
        self._template_parser = template_parser
        self._apply_chat_template_kwargs = {'is_training': True, **(apply_chat_template_kwargs or {})}
        self._mm_plugin: GatewayMultiModalProtocol | None = mm_plugin
        self._priority = priority
        self._enable_rollout_routing_replay = enable_rollout_routing_replay
        # Turn-boundary seam policy; see _build_prompt_render. The default prefers an
        # exact direct extension and may insert a verified closure seam after a divergent
        # frozen stream. GATEWAY_LEGACY_SEAM_COMPAT=1 instead reproduces the legacy
        # HyTunerClient canonical-prefix subtraction for every reused prefix.
        # Delete the switch once the legacy subtraction and its goldens are retired.
        self._legacy_seam_compat = os.environ.get('GATEWAY_LEGACY_SEAM_COMPAT', '') == '1'
        self._is_token_level = tokenizer is not None
        self._tool_call_ids = ToolCallIdGenerator()
        # One processor serves one trajectory, so this cache converts each reference once.
        self._media_resolver = MediaResolver()
        self._accepted_media_sources = getattr(backend, 'accepted_media_sources', REMOTE_CONSUMER_SOURCES)

    @property
    def is_token_level(self) -> bool:
        return self._is_token_level

    async def create_completion(
        self,
        request_id: str,
        messages: list[dict],
        request_kwargs: dict[str, Any] | None = None,
        priority: int | None = None,
        prepare_generation: _PrepareGeneration | None = None,
        uses_default_max_tokens: bool = True,
        request_meta: dict[str, Any] | None = None,
        frozen_prefix: FrozenPrefix | None = None,
    ) -> ProcessorResult:
        """Unified entry point. Dispatches based on is_token_level.

        ``request_meta`` is caller-owned and forwarded to the backend untouched
        (gateway never interprets it) — e.g. HyTuner's ``validate`` flag rides it
        to the generation endpoint.

        ``messages`` is the caller's execution copy; media the backend's consumer cannot
        read is resolved into a new list here, so the caller's recorded messages keep
        their original references.
        """
        execution_request = dict(request_kwargs or {})
        tools = execution_request.get('tools')
        messages = await self._media_resolver.resolve(messages, self._accepted_media_sources)

        if self._is_token_level:
            return await self._token_level_completion(
                request_id,
                messages,
                tools,
                execution_request,
                priority=priority,
                prepare_generation=prepare_generation,
                uses_default_max_tokens=uses_default_max_tokens,
                request_meta=request_meta,
                frozen_prefix=frozen_prefix,
            )
        return await self._message_level_completion(
            request_id,
            messages,
            execution_request,
            request_meta=request_meta,
        )

    # ---- Token-level path ----

    async def _token_level_completion(
        self,
        request_id: str,
        messages: list[dict],
        tools: list[dict] | None,
        request_kwargs: dict[str, Any],
        priority: int | None,
        prepare_generation: _PrepareGeneration | None,
        uses_default_max_tokens: bool,
        request_meta: dict[str, Any] | None = None,
        frozen_prefix: FrozenPrefix | None = None,
    ) -> ProcessorResult:
        """Template -> tokenize -> backend.create_completion(prompt_ids) -> decode -> parse."""
        # 1. Render & tokenize
        raw_prompt_ids, prompt_ids, request_mm, prompt_render = await self.render_and_tokenize(
            messages,
            tools,
            frozen_prefix=frozen_prefix,
        )
        if prepare_generation is not None:
            request_kwargs = prepare_generation(
                raw_prompt_ids,
                prompt_ids,
                request_kwargs,
                uses_default_max_tokens,
            )

        routing_start = None
        if self._enable_rollout_routing_replay:
            routing_start = frozen_prefix.frozen_routing_length if frozen_prefix is not None else 0
            if not 0 <= routing_start < len(prompt_ids):
                raise ValueError(f'Routing start {routing_start} outside expanded prompt length {len(prompt_ids)}')

        # 2. Call backend with prompt_ids
        backend_prompt_ids = raw_prompt_ids.tolist()
        inference_started = time.perf_counter()
        result: BackendResult = await self._backend.create_completion(
            request_id,
            prompt_ids=backend_prompt_ids,
            request_kwargs=request_kwargs,
            image_data=request_mm.multi_modal_data.get('images'),
            priority=self._priority if priority is None else priority,
            request_meta=request_meta,
            routed_experts_prompt_start=routing_start,
        )
        abort_wait_time = float((result.extra_fields or {}).get('_abort_wait_time') or 0.0)
        wall_inference_time = max(0.0, time.perf_counter() - inference_started - abort_wait_time)
        inference_time = result.inference_time if result.inference_time is not None else wall_inference_time
        extra_fields = dict(result.extra_fields)

        # 3. Decode
        tokenizer = self._tokenizer
        assert tokenizer is not None
        if result.token_ids is None:
            raise RuntimeError('Token-level backend returned no token IDs')
        token_ids = list(result.token_ids)
        routed_experts = None
        if self._enable_rollout_routing_replay:
            delta = result.routed_experts
            if delta is None:
                raise ValueError('R3 routed_experts append protocol expected routing; got None')
            assert routing_start is not None
            routing_end = len(prompt_ids) + len(token_ids) - 1
            expected_length = routing_end - routing_start
            if delta.shape[0] != expected_length:
                raise ValueError(
                    f'R3 routed_experts length mismatch: got {delta.shape[0]} rows, expected {expected_length}; '
                    f'request_id={request_id}, prompt_start={routing_start}, '
                    f'expanded_prompt_length={len(prompt_ids)}, generated_tokens={len(token_ids)}'
                )
            routed_experts = delta
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        wire_text = (
            result.wire_text if result.wire_text is not None else tokenizer.decode(token_ids, skip_special_tokens=True)
        )

        # 4. Parse tool calls
        raw_tool = None
        tool_calls = None
        content = wire_text
        if self._tool_parser and token_ids:
            try:
                raw_tool, content, tool_calls = await self._parse_tool_calls(token_ids, wire_text, tools)
            except Exception as exc:
                if result.stop_reason != 'length':
                    raise
                logger.warning(
                    'Tool-call parsing failed after output reached max_tokens; preserving raw partial response: %s: %s',
                    type(exc).__name__,
                    exc,
                )
            else:
                hidden_suffix = text[len(wire_text) :] if text.startswith(wire_text) else ''
                if tool_calls and hidden_suffix and isinstance(content, str) and content.endswith(hidden_suffix):
                    content = content[: -len(hidden_suffix)]

        # 5. Build message
        message: dict[str, Any] = {'role': 'assistant', 'content': content}
        if self._tool_parser:
            message['raw_response'] = wire_text
            if raw_tool is not None:
                message['raw_tool'] = raw_tool
            think_end = getattr(self._tool_parser, 'think_end_token', '</think>')
            if (
                isinstance(content, str)
                and think_end in content
                and not self._prompt_closes_think(backend_prompt_ids, think_end)
            ):
                reasoning, content = content.split(think_end, 1)
                think_start = getattr(self._tool_parser, 'think_start_token', '<think>')
                if reasoning.startswith(think_start):
                    reasoning = reasoning[len(think_start) :]
                message['reasoning_content'] = reasoning or None
                message['content'] = content
        if tool_calls:
            message['tool_calls'] = tool_calls

        choice_provider_specific_fields = result.choice_provider_specific_fields
        if not tool_calls and self._began_tool_call(text):
            # The flag rides on the choice, not extra_fields: extra_fields also feed the token
            # channel to the trainer, and this is a wire-level fact for the host only.
            choice_provider_specific_fields = {
                **(choice_provider_specific_fields or {}),
                UNPARSED_TOOL_CALL_FIELD: True,
            }

        return ProcessorResult(
            message=message,
            token_ids=list(token_ids),
            prompt_ids=prompt_ids,
            raw_prompt_ids=raw_prompt_ids,
            log_probs=result.log_probs,
            routed_experts=routed_experts,
            multi_modal_items=request_mm.multi_modal_items,
            stop_reason=result.stop_reason,
            completion_text=text,
            prompt_render=prompt_render,
            inference_time=inference_time,
            wall_inference_time=wall_inference_time,
            usage=result.usage,
            id=result.id,
            created=result.created,
            model=result.model,
            object=result.object,
            system_fingerprint=result.system_fingerprint,
            service_tier=result.service_tier,
            choice_logprobs=result.choice_logprobs,
            choice_provider_specific_fields=choice_provider_specific_fields,
            extra_fields=extra_fields,
        )

    async def render_and_tokenize(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        add_generation_prompt: bool = True,
        frozen_prefix: FrozenPrefix | None = None,
    ) -> tuple[np.ndarray, np.ndarray, RequestMultiModal, PromptRender]:
        """Build both token views for one request while preserving message ownership.

        Returns the engine's raw prompt IDs and the training prompt IDs, in that order.
        They are the same array when the request carries no media.
        """
        if self._tokenizer is None:
            raise RuntimeError('Token-level prompt preparation requires an injected tokenizer')
        multi_modal_data = await self._extract_multi_modal_data(messages)

        def _render() -> tuple[PromptRender, dict[str, list[Any]]]:
            text = self._render_chat(
                messages,
                tools,
                add_generation_prompt=add_generation_prompt,
            )
            return self._build_prompt_render(
                messages,
                tools,
                text=text,
                add_generation_prompt=add_generation_prompt,
                frozen=frozen_prefix,
                multi_modal_data=multi_modal_data,
            )

        prompt_render, multi_modal_items = await asyncio.to_thread(_render)
        request_mm = RequestMultiModal(
            multi_modal_data=dict(multi_modal_data),
            multi_modal_items=multi_modal_items or None,
        )
        return (
            prompt_render.rollout_token_ids,
            prompt_render.token_ids,
            request_mm,
            prompt_render,
        )

    def _expand_message_token_ids(
        self,
        full_text: str,
        raw_tail_ids: list[np.ndarray],
        generation_prompt_ids: np.ndarray,
        multi_modal_data: Mapping[str, Any],
        frozen_prefix: FrozenPrefix | None,
    ) -> tuple[list[np.ndarray], np.ndarray, dict[str, list[Any]]]:
        """Replace each raw tail span's image placeholder with the processor's training run.

        Runs on the render thread, where the processor call already belongs: it is a
        synchronous family call, so wrapping it in its own to_thread only added a hop.

        A reusable prefix means only the appended suffix goes through the processor. The
        frozen spans were expanded when first sampled and the assembler carries them forward
        verbatim, so re-expanding them could only introduce drift.

        Returns:
            The training tail spans, the generation prompt's own IDs, and the per-item
            processor tensors this call produced. A prefix-reusing request expands only its
            suffix, so the tensors cover the appended images alone.
        """
        mm_plugin = self._mm_plugin
        assert mm_plugin is not None, 'multimodal data implies an injected mm_plugin'
        images = list(multi_modal_data.get('images') or [])
        placeholder_id = int(mm_plugin.get_placeholder_token_ids()['images'])
        if frozen_prefix is None or not frozen_prefix.rollout_message_token_ids:
            prefix_text = ''
            prefix_image_count = 0
        else:
            prefix_text = frozen_prefix.text
            prefix_image_count = sum(
                int(np.count_nonzero(message_ids == placeholder_id))
                for message_ids in frozen_prefix.rollout_message_token_ids
            )

        # NOTE(arlenxu): both slices are length-based, so a prefix that is not really a
        # prefix would misfeed the processor instead of failing. Identity matching in
        # TrajectoryAssembler.frozen_prefix guarantees it a file away, so state it here.
        assert full_text.startswith(prefix_text), 'frozen prefix text is not a prefix of the reconstructed prompt'
        assert prefix_image_count <= len(images), (
            f'frozen prefix claims {prefix_image_count} images, the request carries {len(images)}'
        )

        expansion = mm_plugin.expand_prompt(
            full_text[len(prefix_text) :],
            {'images': images[prefix_image_count:]},
        )
        *tail_ids, expanded_generation_prompt_ids = partition_expanded_ids(
            [*raw_tail_ids, generation_prompt_ids],
            np.asarray(expansion.token_ids, dtype=TOKEN_ID_DTYPE),
            placeholder_id,
        )
        multi_modal_items = {modality: list(items) for modality, items in (expansion.multi_modal_items or {}).items()}
        return tail_ids, expanded_generation_prompt_ids, multi_modal_items

    async def _extract_multi_modal_data(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        mm_plugin = self._mm_plugin
        if mm_plugin is None:
            return {}
        # TODO(petersheng): this re-reads and re-resizes every image of the conversation on
        # each turn, including images in the frozen prefix (whose tokens are reused). Cache
        # decoded media per trajectory, keyed by reference, like gateway.media caches
        # converted references. Performance only: prefix matching keys on the reference
        # string, never on pixels.
        multi_modal_data = await mm_plugin.process_multi_modal_info(messages)
        normalized_multi_modal_data = {key: value for key, value in multi_modal_data.items() if value}
        non_image_modalities = set(normalized_multi_modal_data) - {'images'}
        if non_image_modalities:
            raise NotImplementedError(
                f'prompt expansion encodes images only, got {sorted(non_image_modalities)}; '
                'their payloads would be dropped and their placeholders left unexpanded'
            )
        return normalized_multi_modal_data

    def _build_prompt_render(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        *,
        text: str,
        add_generation_prompt: bool,
        frozen: FrozenPrefix | None = None,
        multi_modal_data: Mapping[str, Any] | None = None,
    ) -> tuple[PromptRender, dict[str, list[Any]]]:
        """Build the exact model prompt, with message spans as best-effort attribution.

        A semantic frozen prefix is immutable physical history. Canonical rendering
        supplies only its continuation; it may never replace the sampled stream.
        """
        start = 0
        anchor = ''
        seam = ''
        frozen_text = ''
        templated_messages: list[str] = []
        message_token_ids: list[np.ndarray] = []
        if frozen is not None:
            if frozen.count > len(messages) or not frozen.text:
                raise ValueError('Frozen prefix does not match the request')
            start = frozen.count
            frozen_text = frozen.text
            templated_messages = list(frozen.templated_messages)
            message_token_ids = list(frozen.message_token_ids)

            if self._legacy_seam_compat:
                if not text.startswith(frozen.text):
                    self._observe_splice_divergence(frozen, text)
                anchor = self._render_chat(messages[: frozen.count], tools, add_generation_prompt=False)
                if not text.startswith(anchor):
                    raise ValueError('Tokenizer template does not expose a legacy-compatible continuation tail')
            elif text.startswith(frozen.text):
                # The canonical request already byte-extends the sampled stream.
                # Its direct suffix includes any closure that follows the frozen tail.
                anchor = frozen.text
            else:
                self._observe_splice_divergence(frozen, text)
                anchor = self._render_chat(messages[: frozen.count], tools, add_generation_prompt=False)
                if not text.startswith(anchor):
                    raise ValueError('Tokenizer template does not expose a legacy-compatible continuation tail')
                seam = self._locate_seam(frozen.text, anchor) or ''

        canonical_tail = text[len(anchor) :]
        continuation = seam + canonical_tail

        structured = None
        if frozen is None:
            structured = self._structured_prompt_spans(
                messages,
                tools,
                text=text,
                add_generation_prompt=add_generation_prompt,
            )
        if structured is not None:
            tail_spans, generation_prompt = structured
        else:
            rendered_inputs = None
            try:
                tail_spans, previous_text = self._discovered_spans(
                    messages,
                    tools,
                    start=start,
                    previous_text=anchor,
                )
                rendered_inputs = previous_text
                if not text.startswith(previous_text):
                    raise ValueError('Tokenizer template does not expose stable per-message prefixes')
                generation_prompt = text[len(previous_text) :]
                if tail_spans:
                    tail_spans[0] = seam + tail_spans[0]
                else:
                    generation_prompt = seam + generation_prompt
                if ''.join(tail_spans) + generation_prompt != continuation:
                    raise ValueError('Attributed spans do not reconstruct the physical continuation')
            except Exception:
                # Generic span ownership is bookkeeping. Preserve the exact
                # generation-prompt suffix when it can still be derived, then
                # conservatively attribute the fused input tail to its last message.
                tail_spans = [''] * (len(messages) - start)
                generation_prompt = continuation
                if tail_spans:
                    if not add_generation_prompt:
                        tail_spans[-1] = continuation
                        generation_prompt = ''
                    else:
                        if rendered_inputs is None:
                            try:
                                rendered_inputs = self._render_chat(
                                    messages,
                                    tools,
                                    add_generation_prompt=False,
                                )
                            except Exception:
                                rendered_inputs = None
                        if (
                            rendered_inputs is not None
                            and text.startswith(rendered_inputs)
                            and rendered_inputs.startswith(anchor)
                        ):
                            input_tail = seam + rendered_inputs[len(anchor) :]
                            candidate_generation_prompt = text[len(rendered_inputs) :]
                            if input_tail + candidate_generation_prompt == continuation:
                                tail_spans[-1] = input_tail
                                generation_prompt = candidate_generation_prompt

        raw_tail_ids, generation_prompt_ids = self._canonical_span_ids(tail_spans, generation_prompt)
        templated_messages += tail_spans
        if not multi_modal_data:
            # Without media the raw and training partitions coincide.
            message_token_ids += raw_tail_ids
            token_ids = np.concatenate((*message_token_ids, generation_prompt_ids))
            return (
                PromptRender(
                    text=frozen_text + continuation,
                    token_ids=token_ids,
                    templated_messages=templated_messages,
                    message_token_ids=message_token_ids,
                    generation_prompt=generation_prompt,
                    generation_prompt_ids=generation_prompt_ids,
                ),
                {},
            )

        # raw_tail_ids is the engine view: _canonical_span_ids tokenizes the template text, where
        # each image is one placeholder. The expansion below derives the training view from it.
        tail_ids, generation_prompt_ids, multi_modal_items = self._expand_message_token_ids(
            frozen_text + continuation,
            raw_tail_ids,
            generation_prompt_ids,
            multi_modal_data,
            frozen,
        )
        # The frozen prefix contributes its own two views; only the appended suffix was
        # derived above.
        raw_prefix_ids = list(frozen.rollout_message_token_ids) if frozen is not None else []
        message_token_ids += tail_ids
        raw_message_token_ids = raw_prefix_ids + raw_tail_ids
        mm_plugin = self._mm_plugin
        assert mm_plugin is not None, 'multimodal data implies an injected mm_plugin'
        placeholder_id = int(mm_plugin.get_placeholder_token_ids()['images'])
        token_ids = np.concatenate((*message_token_ids, generation_prompt_ids))
        return (
            PromptRender(
                text=frozen_text + continuation,
                token_ids=token_ids,
                templated_messages=templated_messages,
                message_token_ids=message_token_ids,
                raw_message_token_ids=raw_message_token_ids,
                generation_prompt=generation_prompt,
                generation_prompt_ids=generation_prompt_ids,
                message_media_counts=[
                    {'images': int(np.count_nonzero(message_ids == placeholder_id))}
                    for message_ids in raw_message_token_ids
                ],
            ),
            multi_modal_items,
        )

    def _discovered_spans(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        *,
        start: int,
        previous_text: str,
    ) -> tuple[list[str], str]:
        """Per-message spans for ``messages[start:]`` by monotone prefix rendering.

        Returns the discovered spans and the final rendered prefix. A moved block
        closure is relocated within the discovered region only; any divergence that
        would reach ``previous_text`` itself fails loudly.
        """
        spans: list[str] = []
        for index in range(start, len(messages)):
            prefix = self._render_chat(
                messages[: index + 1],
                tools,
                add_generation_prompt=False,
            )
            if prefix.startswith(previous_text):
                spans.append(prefix[len(previous_text) :])
            else:
                # A closed block prefix is not a prefix of the full text until its
                # closure relocates past the content appended by later messages.
                closure = self._relocatable_closure(previous_text, prefix, spans)
                if closure is None:
                    raise ValueError('Tokenizer template does not expose stable per-message prefixes')
                spans[-1] = spans[-1][: -len(closure)]
                spans.append(prefix[len(previous_text) - len(closure) :])
            previous_text = prefix
        return spans, previous_text

    # Seam location bounds: how far back a template block closure may reach, and
    # how many shared trailing bytes anchor the frozen tail in the re-render.
    _SEAM_SEARCH_LIMIT = 64
    _SEAM_ANCHOR_LENGTH = 4

    def _locate_seam(self, frozen_text: str, rendered_prefix: str) -> str | None:
        """The block closure the re-render appends after the frozen tail.

        Normalization differences live inside a block, so the block's trailing
        bytes are shared between the sampled stream and the re-render. The
        smallest suffix of the re-rendered prefix whose removal re-exposes the
        frozen tail is the closure owed to the head of the tail.
        """
        for seam_length in range(min(self._SEAM_SEARCH_LIMIT, len(rendered_prefix)) + 1):
            body = rendered_prefix[: len(rendered_prefix) - seam_length]
            shared = 0
            for frozen_char, body_char in zip(reversed(frozen_text), reversed(body), strict=False):
                if frozen_char != body_char:
                    break
                shared += 1
            if shared >= min(len(frozen_text), self._SEAM_ANCHOR_LENGTH):
                return rendered_prefix[len(body) :]
        return None

    def _observe_splice_divergence(self, frozen: FrozenPrefix, text: str) -> None:
        """Record an expected physical divergence without changing splice behavior."""
        offset = 0
        for frozen_char, text_char in zip(frozen.text, text, strict=False):
            if frozen_char != text_char:
                break
            offset += 1
        logger.warning(
            'gateway frozen splice (No Worries): re-render diverges from the sampled stream at byte %d '
            '(sampled %r vs re-render %r); resending sampled bytes',
            offset,
            frozen.text[offset : offset + 24],
            text[offset : offset + 24],
        )

    @staticmethod
    def _relocatable_closure(
        previous_text: str,
        prefix: str,
        templated_messages: list[str],
    ) -> str | None:
        """Smallest block closure the template moved past newly appended content.

        Accepts only ``previous == stable + closure`` versus ``prefix == stable +
        appended + closure`` with the closure wholly owned by the immediately
        preceding message span; any other divergence stays a hard failure at the
        call site. The search starts at the structural minimum (the divergence
        point) so content bytes that merely coincide with the closure's tail are
        never pulled out of the preceding span.
        """
        if not templated_messages or len(prefix) <= len(previous_text):
            return None
        last_span = templated_messages[-1]
        divergence = 0
        for previous_char, prefix_char in zip(previous_text, prefix, strict=False):
            if previous_char != prefix_char:
                break
            divergence += 1
        for length in range(len(previous_text) - divergence, min(len(last_span), len(previous_text)) + 1):
            if length <= 0:
                continue
            closure = previous_text[len(previous_text) - length :]
            if not prefix.endswith(closure):
                continue
            stable_length = len(previous_text) - length
            if len(prefix) - length <= stable_length:
                continue
            if prefix.startswith(previous_text[:stable_length]):
                return closure
        return None

    def _structured_prompt_spans(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        *,
        text: str,
        add_generation_prompt: bool,
    ) -> tuple[list[str], str] | None:
        parser = self._template_parser
        if parser is None or not messages:
            return None

        rendered_messages = parser.inject_tool_prompt(messages, tools) if tools is not None else messages
        if len(rendered_messages) != len(messages):
            return None

        templated_messages: list[str] = []
        for message in rendered_messages:
            render_message = getattr(parser, f'parse_{message["role"]}', None)
            if not callable(render_message):
                return None
            templated_messages.append(render_message(message))

        generation_prompt = getattr(parser, 'generation_prompt', None) if add_generation_prompt else ''
        if not isinstance(generation_prompt, str):
            return None
        # A parser that offers per-role spans has claimed the structured capability;
        # inconsistent spans are a parser bug and must not fall back to prefix probing.
        if not text.endswith(generation_prompt):
            raise ValueError('Template parser spans do not reconstruct the rendered prompt')
        message_text = text[: len(text) - len(generation_prompt)] if generation_prompt else text
        rendered_body = ''.join(templated_messages)
        if not message_text.endswith(rendered_body):
            raise ValueError('Template parser spans do not reconstruct the rendered prompt')
        leading_text = message_text[: len(message_text) - len(rendered_body)] if rendered_body else message_text
        templated_messages[0] = leading_text + templated_messages[0]
        return templated_messages, generation_prompt

    def _render_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        *,
        add_generation_prompt: bool,
    ) -> str:
        tokenizer = self._tokenizer
        assert tokenizer is not None
        render_messages = messages
        if any(
            isinstance(message.get('content'), list)
            and all(isinstance(block, dict) and block.get('type') == 'text' for block in message['content'])
            for message in messages
        ):
            render_messages = []
            for message in messages:
                content = message.get('content')
                if isinstance(content, list) and all(
                    isinstance(block, dict) and block.get('type') == 'text' for block in content
                ):
                    message = {
                        **message,
                        'content': ''.join(block.get('text', '') for block in content if isinstance(block, dict)),
                    }
                render_messages.append(message)
        if self._template_parser:
            msgs = render_messages
            if tools is not None:
                msgs = self._template_parser.inject_tool_prompt(msgs, tools)
            return self._template_parser.parse(
                msgs,
                add_generation_prompt=add_generation_prompt,
                is_first_msg=True,
            )
        return tokenizer.apply_chat_template(
            render_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **self._apply_chat_template_kwargs,
        )

    def _encode_text(self, text: str) -> np.ndarray:
        tokenizer = self._tokenizer
        assert tokenizer is not None
        return np.array(tokenizer.encode(text, add_special_tokens=False), dtype=TOKEN_ID_DTYPE)

    # Enough trailing tokens to hold a think_end and the whitespace a template may append after it.
    _THINK_TAIL_TOKENS = 16

    def _prompt_closes_think(self, prompt_ids: list[int], think_end: str) -> bool:
        """Whether the template closed this turn's think block before generation.

        With thinking disabled the prompt ends ``<think></think>`` (plus whitespace in some
        templates), so a ``think_end`` the model samples belongs to the answer: splitting it off as
        reasoning would make the reply re-render differently from its sampled bytes.
        """
        tokenizer = self._tokenizer
        assert tokenizer is not None
        tail = tokenizer.decode(prompt_ids[-self._THINK_TAIL_TOKENS :], skip_special_tokens=False)
        return tail.rstrip().endswith(think_end)

    def _canonical_span_ids(
        self,
        span_texts: list[str],
        generation_prompt: str,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Tokenize the rendered spans canonically, then attribute ids back to spans.

        Independent per-span encodes can diverge from the whole-text tokenization at
        boundaries without a special-token barrier (e.g. consecutive tool responses the
        template folds into one block): BPE merges across the cut and the engine would
        see a non-canonical token stream. The stream the model sees is the hard
        invariant; span id-grouping is internal bookkeeping. So: encode the whole tail
        once, then split the ids at span text offsets. A token straddling a boundary is
        attributed to the earlier span — its boundary is then not id-aligned, which is
        acceptable because span boundaries carry no authority (FrozenPrefix docstring)
        and histories never split between messages the template fused into one block.
        """
        full = ''.join(span_texts) + generation_prompt
        canonical = self._encode_text(full)
        try:
            per_span = [self._encode_text(text) for text in span_texts]
            generation_ids = self._encode_text(generation_prompt)
            if np.array_equal(np.concatenate((*per_span, generation_ids)), canonical):
                return per_span, generation_ids

            tokenizer = self._tokenizer
            assert tokenizer is not None
            starts: list[int] | None = None
            try:
                enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
                if np.array_equal(enc['input_ids'], canonical):
                    starts = [offset[0] for offset in enc['offset_mapping']]
            except Exception:
                starts = None
            if starts is None:
                # Fallback: byte-level BPE decodes each id to its exact text piece.
                starts = []
                pos = 0
                for token_id in canonical:
                    starts.append(pos)
                    pos += len(tokenizer.decode([int(token_id)]))

            boundaries = []
            acc = 0
            for text in span_texts:
                acc += len(text)
                boundaries.append(acc)
            if len(starts) != len(canonical):
                raise ValueError('token offsets must align with the canonical token stream')
            groups = np.split(canonical, np.searchsorted(np.maximum.accumulate(starts), boundaries, side='left'))
            return groups[:-1], groups[-1]
        except Exception:
            # Canonical encoding is authoritative; span grouping is bookkeeping.
            groups = [np.empty(0, dtype=TOKEN_ID_DTYPE) for _ in span_texts]
            if generation_prompt or not groups:
                return groups, canonical
            groups[-1] = canonical
            return groups, np.empty(0, dtype=TOKEN_ID_DTYPE)

    async def _parse_tool_calls(
        self,
        token_ids: list[int],
        text: str,
        tools: list[dict] | None,
    ) -> tuple[Any, str, list[dict] | None]:
        """Parse tool calls from generated text/tokens."""
        if not self._tool_parser:
            return None, text, None
        extract = getattr(self._tool_parser, 'extract_tool_calls', None)
        if not callable(extract):
            raise RuntimeError('Gateway tool_parser does not expose extract_tool_calls')
        request = _ToolParseRequest(
            tools=tools,
            chat_template_kwargs=dict(self._apply_chat_template_kwargs),
        )
        parameters = inspect.signature(extract).parameters.values()
        accepts_request = any(
            parameter.name == 'request' or parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
        )
        parsed = extract(token_ids, request=request) if accepts_request else extract(token_ids)
        if inspect.isawaitable(parsed):
            parsed = await parsed
        raw_tool = parsed[0] if isinstance(parsed, tuple) and parsed else None
        content, tool_calls = self._normalize_tool_parse_result(
            parsed, text, tool_call_id_factory=self._tool_call_ids.next_id
        )
        return raw_tool, content, tool_calls

    def _began_tool_call(self, completion_text: str) -> bool:
        """Return whether the completion contains the parser's tool-call start marker.

        A parser that rejects a malformed block (a second block after more reasoning, or one
        written before the reasoning end marker) returns the raw text with no tool calls, so the
        marker is the only remaining evidence that the model meant to call a tool. A parser that
        does not name its marker leaves this check off.
        """
        marker = getattr(self._tool_parser, 'tool_calls_start_token', None)
        return isinstance(marker, str) and bool(marker) and marker in completion_text

    @staticmethod
    def _normalize_tool_parse_result(
        parsed: Any,
        fallback_text: str,
        *,
        tool_call_id_factory: Callable[[], str] | None = None,
    ) -> tuple[str, list[dict] | None]:
        """Normalize parser output to (assistant_content, OpenAI tool_calls)."""
        content = fallback_text
        calls = parsed
        if isinstance(parsed, tuple):
            if len(parsed) >= 2:
                content = parsed[1]
            if len(parsed) >= 3:
                calls = parsed[2]

        if not calls:
            return content, None

        next_tool_call_id = tool_call_id_factory or ToolCallIdGenerator().next_id
        normalized = []
        for call in calls:
            if isinstance(call, dict):
                if 'function' in call:
                    normalized.append(call)
                    continue
                name = call.get('name', '')
                arguments = call.get('arguments', '')
            else:
                dumped = call.model_dump() if hasattr(call, 'model_dump') else {}
                name = dumped.get('name', getattr(call, 'name', ''))
                arguments = dumped.get('arguments', getattr(call, 'arguments', ''))

            normalized.append({
                'id': next_tool_call_id(),
                'type': 'function',
                'function': {
                    'name': name,
                    'arguments': arguments,
                },
            })

        return content, normalized

    # ---- Message-level path ----

    async def _message_level_completion(
        self,
        request_id: str,
        messages: list[dict],
        request_kwargs: dict[str, Any],
        request_meta: dict[str, Any] | None = None,
    ) -> ProcessorResult:
        """Forward messages to backend. Backend returns a parsed message."""
        inference_started = time.perf_counter()
        result: BackendResult = await self._backend.create_completion(
            request_id,
            messages=messages,
            request_kwargs=request_kwargs,
            request_meta=request_meta,
        )
        wall_inference_time = time.perf_counter() - inference_started
        inference_time = result.inference_time if result.inference_time is not None else wall_inference_time
        if result.message is None:
            raise RuntimeError('Message-level backend returned no message')

        extra_fields = dict(result.extra_fields)
        return ProcessorResult(
            message=result.message,
            inference_time=inference_time,
            wall_inference_time=wall_inference_time,
            usage=result.usage,
            stop_reason=result.stop_reason,
            id=result.id,
            created=result.created,
            model=result.model,
            object=result.object,
            system_fingerprint=result.system_fingerprint,
            service_tier=result.service_tier,
            choice_logprobs=result.choice_logprobs,
            choice_provider_specific_fields=result.choice_provider_specific_fields,
            extra_fields=extra_fields,
        )
