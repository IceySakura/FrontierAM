"""Record actual host Responses HTTP attempts, not reconstructed OpenCode messages."""

import copy
import inspect
import time

from typing import Any

from nexus.extensions.tasks.context_agent.enterprisebench_native.errors import judge_length_reason
from nexus.extensions.tasks.context_agent.enterprisebench_native.judge_monitor import JudgeMonitor
from nexus.utils.provider_io import redact_provider_io


class ResponsesCapture:
    """Ordered delta journal of native input/output; changes never affect API traffic."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {'format': 'responses_stream_v1', 'stream': []}
        self.tools: dict[str, Any] = {'definitions': [], 'bindings': []}
        self._last_tools: dict[str, int] = {}
        self._history: list[Any] = []
        self._parameters: dict[str, Any] = {}
        self._active_request = 0
        self._count = 0

    def _capture_tools(self, payload: dict[str, Any], number: int, side: str) -> None:
        # Keep missing vs null/empty distinguishable and preserve changed schemas.
        if 'tools' not in payload:
            if side in self._last_tools:
                self.tools['bindings'].append({'request_id': number, 'side': side, 'definition_index': None})
                self._last_tools.pop(side)
            return
        value = payload['tools']
        definitions = self.tools['definitions']
        index = next((i for i, old in enumerate(definitions) if old == value), None)
        if index is None:
            index = len(definitions)
            definitions.append(copy.deepcopy(value))
        if self._last_tools.get(side) != index:
            self.tools['bindings'].append({'request_id': number, 'side': side, 'definition_index': index})
            self._last_tools[side] = index

    def start(self, request: dict[str, Any], endpoint: str) -> int:
        self._count += 1
        number = self._count
        self._active_request = number
        event: dict[str, Any] = {
            'type': 'request',
            'request_id': number,
            'started_at': time.time(),
        }
        if self.data.get('endpoint') != endpoint:
            self.data['endpoint'] = endpoint
            event['endpoint'] = endpoint
        self._capture_tools(request, number, 'request')
        parameters = {k: v for k, v in request.items() if k not in {'input', 'tools'}}
        changed = {k: v for k, v in parameters.items() if k not in self._parameters or self._parameters[k] != v}
        removed = [k for k in self._parameters if k not in parameters]
        if changed:
            event['parameters_set'] = changed
        if removed:
            event['parameters_remove'] = removed
        self._parameters = parameters
        history = request.get('input')
        if isinstance(history, list):
            prefix = 0
            for old, new in zip(self._history, history, strict=False):
                if old != new:
                    break
                prefix += 1
            event['retain_items'] = prefix
            event['input_append'] = history[prefix:]
            self._history = list(history)
        else:
            # Preserve non-list inputs explicitly instead of inventing message items.
            event['input_replace'] = history
            self._history = []
        self.data['stream'].append(event)
        return number

    def response(self, number: int, response: Any, headers: Any) -> None:
        if isinstance(response, dict):
            self._capture_tools(response, number, 'response')
            response = {k: v for k, v in response.items() if k != 'tools'}
        output = response.get('output') if isinstance(response, dict) else None
        append = self._active_request == number and isinstance(output, list)
        self.data['stream'].append({
            'type': 'response',
            'request_id': number,
            'response': response,
            'response_headers': headers,
            'append_output_to_history': append,
        })
        if append and isinstance(output, list):
            self._history.extend(output)

    def error(self, number: int, error: Any) -> None:
        self.data['stream'].append({'type': 'error', 'request_id': number, 'error': error})

    def finish(self, number: int, started: float) -> None:
        self.data['stream'].append({
            'type': 'request_end',
            'request_id': number,
            'finished_at': time.time(),
            'duration_seconds': time.monotonic() - started,
        })


class _ResourceProxy:
    """Delegate untouched SDK operations to the original resource."""

    def __init__(self, resource: Any) -> None:
        self._resource = resource

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resource, name)


class _CapturedRawResponses(_ResourceProxy):
    def __init__(self, resource: Any, request: Any) -> None:
        super().__init__(resource)
        self._request = request

    async def create(self, **kwargs: Any) -> Any:
        raw_response, _ = await self._request(kwargs)
        return raw_response


class _CapturedResponses(_ResourceProxy):
    def __init__(self, resource: Any, request: Any) -> None:
        super().__init__(resource)
        self._request = request
        self.with_raw_response = _CapturedRawResponses(resource.with_raw_response, request)

    async def create(self, **kwargs: Any) -> Any:
        _, response = await self._request(kwargs)
        return response


class _CapturedSDK(_ResourceProxy):
    def __init__(self, resource: Any, request: Any) -> None:
        super().__init__(resource)
        self.responses = _CapturedResponses(resource.responses, request)


def attach_capture(client: Any, records: ResponsesCapture, monitor: JudgeMonitor) -> None:
    """Wrap a Task-owned clone's SDK transport inside the existing retry/deadline loop."""
    original_sdk = client.client
    sensitive = {**client.config.client_args, 'resolved_api_key': original_sdk.api_key}

    def clean(value: Any) -> Any:
        return redact_provider_io(value, sensitive_source=sensitive)

    async def recorded_request(request: dict[str, Any]) -> Any:
        if monitor.failure is not None:
            raise monitor.failure
        started = time.monotonic()
        number = records.start(clean(copy.deepcopy(request)), clean(client.endpoint))
        try:
            raw_response = await original_sdk.responses.with_raw_response.create(**request)
            response = raw_response.parse()
            if inspect.isawaitable(response):
                response = await response
            raw_headers = getattr(getattr(raw_response, 'http_response', None), 'headers', None)
            headers = (
                {'x-ms-region': raw_headers['x-ms-region']} if raw_headers and 'x-ms-region' in raw_headers else {}
            )
            value = (
                response.model_dump(mode='json', exclude_unset=True) if hasattr(response, 'model_dump') else response
            )
            records.response(number, clean(value), clean(headers))
            reason = judge_length_reason(value)
            if reason:
                raise monitor.length_failure(reason)
            return raw_response, response
        except BaseException as exc:
            status = getattr(exc, 'status_code', None) or getattr(exc, 'status', None)
            body = getattr(exc, 'body', None)
            records.error(
                number, clean({'type': type(exc).__name__, 'message': str(exc), 'status_code': status, 'body': body})
            )
            reason = judge_length_reason(body)
            if reason:
                raise monitor.length_failure(reason) from None
            raise
        finally:
            records.finish(number, started)

    client.client = _CapturedSDK(original_sdk, recorded_request)
