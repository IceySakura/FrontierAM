"""OpenClaw agent integration for Nexus."""

import json
import logging

from typing import Any

from nexus.core import agent_registry, require_context
from nexus.core.agent import WithFinalAnswer
from nexus.extensions.agents.installed_common import (
    InstalledAgentOutput,
    InstalledNodeCliInput,
    call_runtime_agent,
    coerce_user_prompt,
)

from .runtime_setup import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_LLM_IDLE_TIMEOUT_MS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_ONBOARD_TIMEOUT,
    OpenClawExitStatus,
    classify_openclaw_response,
    parse_openclaw_response,
    visible_openclaw_payload_texts,
)


logger = logging.getLogger(__name__)


class OpenClawInput(InstalledNodeCliInput):
    """Input for the OpenClaw agent."""

    model: str | None = None
    """Backend model id used by the host-side OpenAI-compatible client."""

    version: str = 'latest'
    """OpenClaw npm version to install."""

    context_window: int = DEFAULT_CONTEXT_WINDOW
    """Context window size reported to OpenClaw for the target model."""

    max_tokens: int = DEFAULT_MAX_TOKENS
    """Maximum output tokens reported to OpenClaw for the target model."""

    llm_idle_timeout_ms: int | float | str = DEFAULT_LLM_IDLE_TIMEOUT_MS
    """OpenClaw LLM stream idle timeout in milliseconds."""

    onboard_timeout: float | None = DEFAULT_ONBOARD_TIMEOUT
    """Timeout for the ``openclaw onboard`` setup command in seconds."""

    supports_images: bool = False
    """Whether the model supports image input (multimodal vision)."""


class OpenClawOutput(InstalledAgentOutput, WithFinalAnswer):
    """Execution result for the OpenClaw agent.

    ``final_answer`` is declared here rather than on the shared installed-agent output
    because OpenClaw is so far the only CLI Agent that fills it: only an Agent knows how
    its own CLI frames a finished answer, and a field every installed Agent carries but
    none of the others sets would just be an empty promise in their records.
    """

    transcript: list[dict[str, Any]] | None = None
    """OpenClaw transcript."""

    exit_status: OpenClawExitStatus
    """Normalized completion status derived from the Runtime command outcome."""

    @property
    def recorded_messages(self) -> list[dict[str, Any]]:
        """The conversation this run recorded, for a verifier that asked to see it.

        Derived on demand rather than stored: the transcript above already holds it, so a
        second copy would repeat every message in the same record. Deliberately not named
        ``main_trajectory`` — that is the framework's name for a trajectory in a shape
        readers act on, and these messages keep OpenClaw's own field names instead.
        """
        return extract_recorded_messages(self.transcript)


def extract_final_answer(output: str | None, transcript: list[dict[str, Any]] | None = None) -> str | None:
    """Return the visible answer OpenClaw reports, preferring its ``--json`` envelope.

    The envelope is ``{"payloads": [{"text": ...}], "meta": {...}}``; a turn may split its
    reply across several payloads, so visible non-error payloads are joined. A valid
    envelope containing only reasoning/error payloads has no final answer. The session
    transcript is the fallback only when stdout is not a ReplyPayload envelope, and only
    its visible ``text`` blocks are read.
    """
    if isinstance(output, str):
        payload = parse_openclaw_response(output)
        if isinstance(payload, dict):
            reply_payloads = payload.get('payloads')
            if not isinstance(reply_payloads, list):
                reply_payloads = None
        else:
            reply_payloads = None
        if reply_payloads is not None:
            texts = visible_openclaw_payload_texts(payload)
            return '\n\n'.join(texts) if texts else None

    for entry in reversed(transcript or []):
        if not isinstance(entry, dict):
            continue
        message = entry.get('message', entry)
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        text = _visible_text(message.get('content'))
        if text:
            return text
    return None


def extract_recorded_messages(transcript: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Pick the conversation out of an OpenClaw session transcript.

    The session file wraps each message in an envelope (``{"type": "message", "message":
    {...}}``) and interleaves non-message events such as the session header, so unwrapping
    is what happens here — and all that happens. The messages keep OpenClaw's own field
    names (``toolCallId``, ``toolName``, a ``toolResult`` role), which is what a grader
    written against an OpenClaw session reads; translating them would rewrite the very
    characters such a grader compares.
    """
    messages: list[dict[str, Any]] = []
    for entry in transcript or []:
        if not isinstance(entry, dict):
            continue
        message = entry.get('message', entry)
        if isinstance(message, dict) and message.get('role'):
            messages.append(message)
    return messages


def _visible_text(content: Any) -> str | None:
    """Join the user-visible text blocks of one OpenClaw message."""
    if isinstance(content, str):
        return content.strip() or None
    if not isinstance(content, list):
        return None
    parts = [
        block['text']
        for block in content
        if isinstance(block, dict) and block.get('type') == 'text' and isinstance(block.get('text'), str)
    ]
    joined = '\n'.join(part for part in parts if part.strip()).strip()
    return joined or None


def _resolve_model_name(input_data: OpenClawInput) -> str:
    if input_data.model:
        return input_data.model

    ctx = require_context()
    config = getattr(ctx.openai_client, 'config', None)
    if config is None:
        raise RuntimeError('OpenClaw requires agent_config.model or a configured main OpenAI client model')

    for source in (ctx.openai_client.get_effective_request_args(), getattr(config, 'client_args', {})):
        model = source.get('model')
        if isinstance(model, str) and model.strip():
            return model

    raise RuntimeError('OpenClaw requires agent_config.model or openai_clients.main.config.request_args.model')


@agent_registry.register('openclaw')
async def run_openclaw(input_data: OpenClawInput) -> OpenClawOutput:
    """Run OpenClaw inside the configured runtime and proxy its LLM calls to the host."""
    task = coerce_user_prompt(input_data.user_prompt)
    model = _resolve_model_name(input_data)

    logger.debug('Starting OpenClaw agent with model=%s, supports_images=%s', model, input_data.supports_images)
    from .runtime_setup import setup_and_run_openclaw

    result = await call_runtime_agent(
        setup_and_run_openclaw,
        task=task,
        model=model,
        working_dir=input_data.working_dir,
        timeout=input_data.timeout,
        onboard_timeout=input_data.onboard_timeout,
        version=input_data.version,
        npm_registry=input_data.npm_registry,
        context_window=input_data.context_window,
        max_tokens=input_data.max_tokens,
        llm_idle_timeout_ms=input_data.llm_idle_timeout_ms,
        supports_images=input_data.supports_images,
    )
    output = result.get('output', '')
    transcript = result.get('transcript')
    reported_success = bool(result.get('success', False))
    exit_code = result.get('exit_code', -1)
    if exit_code == 0 and parse_openclaw_response(output) is None:
        stderr_payload = parse_openclaw_response(result.get('stderr', '') or '')
        if stderr_payload is not None:
            output = json.dumps(stderr_payload, ensure_ascii=False)
    parsed_exit_status, reply_failure = classify_openclaw_response(output, exit_code)
    raw_exit_status = result.get('exit_status')
    runtime_exit_status: OpenClawExitStatus | None = (
        raw_exit_status if raw_exit_status in {'completed', 'timeout', 'truncation', 'error'} else None
    )
    timed_out = bool(result.get('timed_out')) or runtime_exit_status == 'timeout'
    exit_status: OpenClawExitStatus = 'timeout' if timed_out else (runtime_exit_status or parsed_exit_status)
    success = reported_success and exit_status == 'completed' and parsed_exit_status == 'completed'
    failure_kind = result.get('failure_kind')
    failure_stage = result.get('failure_stage')
    stderr = result.get('stderr', '') or ''
    if reply_failure and reply_failure not in stderr:
        stderr = f'{stderr}\n{reply_failure}'.strip()
    if reported_success and not success and failure_kind is None:
        failure_kind = 'agent'
        failure_stage = 'process'
    if not success and failure_kind is None:
        parsed_payload = parse_openclaw_response(output)
        has_agent_attempt_evidence = (
            timed_out
            or (isinstance(exit_code, int) and exit_code not in {-1, 0})
            or (exit_code == 0 and parsed_payload is not None)
        )
        if has_agent_attempt_evidence:
            failure_kind = 'agent'
            failure_stage = 'process'
        else:
            # Old Runtime images did not distinguish setup failures from a real
            # OpenClaw attempt. Fail closed only when no process/payload evidence exists.
            failure_kind = 'infrastructure'
            failure_stage = 'runtime_protocol'
            protocol_error = 'Runtime returned an unclassified OpenClaw failure; update the Nexus Runtime image'
            stderr = f'{stderr}\n{protocol_error}'.strip()
    return OpenClawOutput(
        success=success,
        output=output,
        final_answer=extract_final_answer(output, transcript) if success else None,
        stderr=stderr,
        exit_code=exit_code,
        execution_time=result.get('execution_time', 0),
        failure_kind=failure_kind,
        failure_stage=failure_stage,
        exit_status=exit_status,
        transcript=transcript,
    )


__all__ = [
    'OpenClawInput',
    'OpenClawOutput',
    'run_openclaw',
]
