"""Import local HyCode spans into the active agent timeline after HyCode exits."""

from __future__ import annotations

import asyncio
import json
import logging
import time

from typing import Any

from nexus.trace import SpanController, SpanEndEvent, SpanStartEvent, get_current_span, phase, update_span
from nexus.trace.types import Status, run_attribution_attributes
from nexus.utils.truncate import truncate_recursive


logger = logging.getLogger(__name__)
_SPAN_FETCH_TIMEOUT_SECONDS = 30.0
# HyCode publishes its buffered spans during shutdown within this budget.
_SPANS_SHUTDOWN_TIMEOUT_SECONDS = 20
# Bound parsing and synchronous replay time; limits only shrink the exported timeline.
_MAX_SPANS_FILE_BYTES = 64 * 1024 * 1024
_MAX_TOOL_SPANS = 10000
_MAX_TOOL_SPAN_STRING_BYTES = 16 * 1024
_TOOL_SPAN_ATTRIBUTES = frozenset({'tool.name', 'input.value', 'execution.success', 'execution.error'})


def _decode_value(value: dict[str, Any]) -> Any:
    if 'arrayValue' in value:
        return [_decode_value(item) for item in value['arrayValue'].get('values', [])]
    if 'kvlistValue' in value:
        return _decode_attributes(value['kvlistValue'].get('values', []))
    if 'intValue' in value:
        return int(value['intValue'])
    return next(iter(value.values()), None)


def _decode_attributes(values: list[dict[str, Any]]) -> dict[str, Any]:
    return {item['key']: _decode_value(item['value']) for item in values}


def build_telemetry_env(path: str, *, collect_tool_args: bool) -> dict[str, str]:
    """Configure OpenCC to write local spans with optional tool arguments."""
    return {
        'HYCODE_SPANS_FILE': path,
        'CLAUDE_CODE_ENABLE_TELEMETRY': '1',
        'CLAUDE_CODE_ENHANCED_TELEMETRY_BETA': '1',
        'ENABLE_BETA_TRACING_DETAILED': '1' if collect_tool_args else '0',
        'OTEL_TRACES_EXPORT_INTERVAL': '1000',
        'OTEL_TRACES_SAMPLER': 'always_on',
        'OTEL_LOG_TOOL_CONTENT': '1' if collect_tool_args else '0',
        'CLAUDE_CODE_OTEL_SHUTDOWN_TIMEOUT_MS': str(_SPANS_SHUTDOWN_TIMEOUT_SECONDS * 1000),
    }


def _tool_span(span: dict[str, Any]) -> dict[str, Any] | None:
    attributes = span.get('attributes', [])
    if _decode_attributes([item for item in attributes if item.get('key') == 'span.type']).get('span.type') != 'tool':
        return None
    return {
        'startTimeUnixNano': span.get('startTimeUnixNano'),
        'endTimeUnixNano': span.get('endTimeUnixNano'),
        'attributes': [
            {'key': item['key'], 'value': truncate_recursive(item['value'], max_size=_MAX_TOOL_SPAN_STRING_BYTES)}
            for item in attributes
            if item.get('key') in _TOOL_SPAN_ATTRIBUTES
        ],
    }


async def fetch_hycode_spans(runtime: Any, path: str) -> dict[str, Any]:
    """Download bounded tool spans from the OTLP/JSON file HyCode publishes atomically at exit."""
    from nexus.runtime.executors.models.file import FileItem

    response = await runtime.download_file([FileItem(path=path)])
    if not response.files:
        errors = [error.message for error in response.errors or []]
        if errors == ['File does not exist']:
            return {'spans': [], 'size_bytes': 0, 'invalid_batches': 0, 'missing': True}
        raise RuntimeError(f'Unable to download HyCode spans from {path}: {errors}')
    content = response.files[0].content or ''
    size_bytes = len(content.encode())
    if size_bytes > _MAX_SPANS_FILE_BYTES:
        return {'spans': [], 'size_bytes': size_bytes, 'invalid_batches': 0, 'oversized': True}
    spans: list[dict[str, Any]] = []
    invalid_batches = 0
    for line in content.split('\n'):
        if not line.strip():
            continue
        try:
            batch = json.loads(line)
            batch_spans = [
                _tool_span(span)
                for resource in batch['resourceSpans']
                for scope in resource.get('scopeSpans', [])
                for span in scope.get('spans', [])
            ]
        except (ValueError, KeyError, TypeError, AttributeError):
            invalid_batches += 1
            continue
        spans.extend(span for span in batch_spans if span is not None)
        if len(spans) >= _MAX_TOOL_SPANS:
            break
    return {'spans': spans[:_MAX_TOOL_SPANS], 'size_bytes': size_bytes, 'invalid_batches': invalid_batches}


async def import_hycode_spans(path: str, *, collect_tool_args: bool, tool_args_min_duration: float) -> None:
    """Fetch optional spans before the agent returns, leaving its result unchanged on failure."""
    from nexus.core import require_context

    async def fetch() -> dict[str, Any]:
        return await fetch_hycode_spans(await require_context().get_runtime(), path)

    result: dict[str, Any] = {}
    transfer: dict[str, Any] = {'status': 'error'}
    started = time.monotonic()
    try:
        with phase('task.agent.fetch_spans', close_previous=False) as fetch_phase:
            try:
                result = await asyncio.wait_for(fetch(), timeout=_SPAN_FETCH_TIMEOUT_SECONDS)
                transfer.update(
                    status='missing' if result.get('missing') else 'oversized' if result.get('oversized') else 'ok',
                    size_bytes=result['size_bytes'],
                    invalid_batches=result['invalid_batches'],
                    span_count=len(result['spans']),
                )
                if transfer['status'] != 'ok':
                    logger.info('Optional HyCode span file was not imported (%s): %s', transfer['status'], path)
            except asyncio.TimeoutError:
                transfer['status'] = 'timeout'
                logger.warning('Optional HyCode span fetch timed out after %s seconds', _SPAN_FETCH_TIMEOUT_SECONDS)
            except Exception:
                logger.warning('Unable to read optional HyCode spans; task result is unchanged', exc_info=True)
            transfer['duration_s'] = time.monotonic() - started
            fetch_phase.set_attributes({'hycode.span_transfer': transfer})
        if transfer['status'] == 'ok':
            replay_hycode_spans(
                result['spans'],
                collect_tool_args=collect_tool_args,
                tool_args_min_duration=tool_args_min_duration,
            )
    except Exception:
        logger.warning('Unable to import optional HyCode spans; task result is unchanged', exc_info=True)


def replay_hycode_spans(
    spans: list[dict[str, Any]],
    *,
    collect_tool_args: bool,
    tool_args_min_duration: float,
) -> None:
    """Replay tool observations and report their combined wall time on the agent span."""
    parent = get_current_span()
    if parent is None:
        logger.warning('Cannot import HyCode spans without an active Nexus span')
        return
    controller = SpanController.get_instance_sync()
    attribution = run_attribution_attributes(
        run_id=parent.attributes.get('_run_id'),
        run_span_id=parent.attributes.get('_run_span_id'),
        run_kind=parent.attributes.get('_run_kind'),
    )
    intervals: list[tuple[int, int]] = []
    for source in spans:
        try:
            attributes = _decode_attributes(source.get('attributes', []))
            start = int(source['startTimeUnixNano'])
            end = int(source['endTimeUnixNano'])
            if start <= 0 or end < start:
                raise ValueError('HyCode span must have ordered, nonzero timestamps')
            duration = (end - start) / 1e9
            success = attributes.get('execution.success')
            nexus_attributes: dict[str, Any] = {
                **attribution,
                'type': 'agent_observation',
                'tool_name': attributes.get('tool.name'),
            }
            if isinstance(success, bool):
                nexus_attributes['hytuner.tool_success'] = success
            if success is False and 'execution.error' in attributes:
                nexus_attributes['hytuner.tool_error'] = attributes['execution.error']
            if (
                collect_tool_args
                and (tool_args_min_duration <= 0 or duration >= tool_args_min_duration)
                and 'input.value' in attributes
            ):
                nexus_attributes['hytuner.tool_args'] = attributes['input.value']
            if success is True:
                status = Status(code='OK')
            elif success is False:
                status = Status(code='ERROR', message=attributes.get('execution.error'))
            else:
                status = Status(code='UNSET')
            start_event = SpanStartEvent(
                trace_id=parent.trace_id,
                parent_span_id=parent.span_id,
                name='agent.observation',
                time_unix_nano=start,
                attributes=nexus_attributes,
                status=status,
            )
            controller.add_event(start_event, origin='replay')
            controller.add_event(SpanEndEvent(span_id=start_event.span_id, time_unix_nano=end), origin='replay')
            intervals.append((start, end))
        except Exception:
            logger.warning('Unable to import optional HyCode span', exc_info=True)
    if intervals:
        update_span(parent.span_id, attributes={'hytuner.observation_time': _merged_seconds(intervals)})


def _merged_seconds(intervals: list[tuple[int, int]]) -> float:
    """Count overlapping tool executions once when reporting agent observation time."""
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = 0
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return (total + end - start) / 1e9
