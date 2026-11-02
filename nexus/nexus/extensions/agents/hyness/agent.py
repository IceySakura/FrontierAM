"""Run packaged Hyness Apps through the existing Nexus RL bridge."""

from __future__ import annotations

import base64
import binascii
import json
import logging

from typing import Any, Literal

from pydantic import Field, model_validator

from nexus.core import agent_registry, require_context
from nexus.core.agent import ContentPartParam
from nexus.extensions.agents.claude_code.mcp import mcp_config_servers_from_specs
from nexus.extensions.agents.hycode.agent import HyCodeInput, HyCodeOutput, classify_hycode_result
from nexus.extensions.agents.installed_common import call_runtime_agent, coerce_user_prompt
from nexus.runtime.client.base import RuntimeUnhealthyError
from nexus.trace import RemoteTraceContext, get_current_span_id, update_span

from .installer import normalize_package_specs


logger = logging.getLogger(__name__)

_OPENCC_APP = '@tencent/hyness-opencc/apps/opencc/app.yaml'
_OPENCC_OVERLAY = '@tencent/hyness-opencc/apps/opencc/overlays/training.yaml'


def encode_user_prompt(user_prompt: str | list[ContentPartParam], input_format: Literal['text', 'stream-json']) -> str:
    """Keep legacy text input and explicitly convert multimodal SDK input.

    Only embedded images are accepted: resolving URLs or filesystem paths belongs
    to the caller, so prompt conversion never performs network or file access.
    """
    if input_format == 'text':
        return coerce_user_prompt(user_prompt)
    if input_format != 'stream-json':
        raise ValueError(f'Unsupported Hyness input format: {input_format}')
    content: str | list[dict[str, Any]]
    if isinstance(user_prompt, str):
        content = user_prompt
    else:
        content = []
        for part in user_prompt:
            if part['type'] == 'text':
                content.append({'type': 'text', 'text': part['text']})
            elif part['type'] == 'image_url':
                url = part['image_url']['url']
                header, separator, data = url.partition(',')
                mime = header.removeprefix('data:').removesuffix(';base64')
                if (
                    not separator
                    or not header.startswith('data:')
                    or not header.endswith(';base64')
                    or mime not in {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}
                    or not data
                ):
                    raise ValueError('Hyness stream-json images require a base64 JPEG, PNG, GIF or WebP data URI')
                try:
                    base64.b64decode(data, validate=True)
                except (ValueError, binascii.Error) as error:
                    raise ValueError('Invalid base64 image in Hyness stream-json input') from error
                content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': mime, 'data': data}})
            else:
                raise ValueError(f'Hyness stream-json input does not support content type {part["type"]!r}')
    # A string containing JSON is still user text, never an SDK control event.
    return json.dumps({'type': 'user', 'message': {'role': 'user', 'content': content}}, ensure_ascii=False) + '\n'


class HynessInput(HyCodeInput):
    """Shared task inputs with explicit Hyness packages and App selection."""

    version: str = ''
    input_format: Literal['text', 'stream-json'] = 'text'
    """Use stream-json to attach text and base64 images directly to the first user turn."""
    packages: list[str] = Field(default_factory=list)
    app_config: str = Field(default=_OPENCC_APP, min_length=1)
    app_config_overlay: str | None = Field(default=None, min_length=1)
    harness_id: str | None = Field(default=None, min_length=1)
    """None selects the default Harness declared by the App."""
    max_turns: int | None = Field(default=None, ge=0)
    emit_spans: bool = Field(default=False, exclude=True)
    """HyCode's local span-file import is not part of the Hyness runtime protocol."""
    session_persistence: bool = False
    """Persist tool receipts inside the temporary session directory, removed after execution."""
    preserve_tool_schema: bool = False
    """Forward native JSON Schema verbatim when the selected model supports it."""

    @model_validator(mode='before')
    @classmethod
    def resolve_app_defaults(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        # Preserve old OpenCC rows without leaking its policy into other Apps.
        # Explicit null must survive model_dump()/remote validation as null.
        if data.get('app_config', _OPENCC_APP) == _OPENCC_APP:
            data.setdefault('harness_id', 'opencc')
            if data['harness_id'] == 'opencc':
                data.setdefault('app_config_overlay', _OPENCC_OVERLAY)
        return data

    @model_validator(mode='after')
    def validate_launch(self) -> HynessInput:
        if self.version:
            raise ValueError('Remove the HyCode version and pin Hyness packages explicitly')
        if self.emit_spans:
            raise ValueError('Hyness does not support HyCode emit_spans; leave it disabled')
        if self.packages:
            self.packages = normalize_package_specs(self.packages)
        if (self.extra_env or {}).get('ANTHROPIC_API_PROTOCOL', '').strip():
            raise ValueError('The Nexus Hyness agent uses the native Anthropic transport')
        if self.session_persistence and any(
            arg.split('=', 1)[0] == '--no-session-persistence' for arg in self.extra_cli_args
        ):
            raise ValueError('session_persistence cannot be combined with --no-session-persistence')
        return self


class HynessOutput(HyCodeOutput):
    process_exit_code: int | None = None
    process_output: str | None = None
    process_stderr: str | None = None
    execution_termination: dict[str, Any] | None = None


def _record_diagnostics(span_id: str | None, attributes: dict[str, object]) -> None:
    if span_id:
        try:
            update_span(span_id, attributes=attributes)
        except Exception:
            logger.exception('Failed to record Hyness diagnostics on span %s', span_id)


def classify_hyness_result(result: dict[str, Any]) -> HynessOutput:
    """Use HyCode's control-exit policy after decoding Hyness execution facts."""
    facts = dict(result)
    if not facts.get('result_count') and not facts.get('timed_out') and facts.get('exit_code') == 1:
        # Hyness may fail before model I/O. HyCode's scoreable exit-1 policy
        # cannot turn that startup/protocol failure into a trainable trajectory.
        facts.update(exit_code=-1, failure_kind='infrastructure', failure_stage='entrypoint')
    span_id = get_current_span_id()
    output = HynessOutput(
        **classify_hycode_result(facts, agent_span_id=span_id).model_dump(),
        process_exit_code=result.get('process_exit_code', result.get('exit_code')),
        process_output=result.get('process_output'),
        process_stderr=result.get('process_stderr'),
        execution_termination=result.get('execution_termination'),
    )
    _record_diagnostics(span_id, {'hyness.process_exit_code': output.process_exit_code})
    return output


@agent_registry.register('hyness', meta={'gateway': {'scaffold_policy': 'hycode'}, 'supports_execution_target': True})
async def run_hyness(input_data: HynessInput) -> HynessOutput:
    from .artifacts import runtime_package_spec
    from .runtime_setup import setup_and_run_hyness

    task = encode_user_prompt(input_data.user_prompt, input_data.input_format)
    packages = normalize_package_specs(input_data.packages)
    servers = mcp_config_servers_from_specs(input_data.mcp_servers)
    duplicate_names = sorted(servers.keys() & input_data.mcp_config_servers.keys())
    if duplicate_names:
        raise ValueError(f'Duplicate task MCP server names across inputs: {duplicate_names}')
    servers.update(input_data.mcp_config_servers)
    if servers and any(
        arg.split('=', 1)[0] in {'--mcp-config', '--strict-mcp-config'} for arg in input_data.extra_cli_args
    ):
        raise ValueError('Task MCP injection cannot be combined with explicit MCP CLI arguments')
    agent_span_id = get_current_span_id()
    try:
        context = require_context()
    except RuntimeError:
        context = None
    trace = RemoteTraceContext.from_current_span()
    if context is not None:
        if trace is None:
            raise RuntimeError('Hyness active-time timeout requires an active trace span')
        if context.runtime_command_timeout_id is not None:
            raise RuntimeError('ExecutionContext already has an active runtime command timeout')
        context.runtime_command_timeout_id = trace.parent_span_id
    try:
        packages = [await runtime_package_spec(spec) for spec in packages]
        payload = input_data.model_dump(
            exclude={
                'user_prompt',
                'version',
                'packages',
                'disable_invoke_wait_timeout',
                'mcp_servers',
                'mcp_config_servers',
            }
        )
        result = await call_runtime_agent(
            setup_and_run_hyness,
            **payload,
            packages=packages,
            task=task,
            mcp_config_servers=servers,
            runtime_trace_context=trace,
            disable_invoke_wait_timeout=(
                input_data.disable_invoke_wait_timeout
                if input_data.disable_invoke_wait_timeout is not None
                else context is not None and context.task_config.get('is_validation') is False
            ),
        )
    except RuntimeUnhealthyError as error:
        logger.exception('Hyness runtime became unhealthy; classifying as ENV_ERROR')
        _record_diagnostics(
            agent_span_id,
            {'hytuner.termination_reason': 'ENV_ERROR', 'hycode.termination_source': 'runtime_unhealthy'},
        )
        return HynessOutput(
            success=False,
            output='',
            stderr=f'Runtime unhealthy: {error}',
            exit_code=-1,
            execution_time=0,
            failure_kind='infrastructure',
            failure_stage='runtime_unhealthy',
            termination_reason='ENV_ERROR',
            exit_status='env_error',
        )
    finally:
        if context is not None:
            context.runtime_command_timeout_id = None
    return classify_hyness_result(result)
