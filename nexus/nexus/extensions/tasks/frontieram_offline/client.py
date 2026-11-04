"""Per-run budgets around the native Nexus OpenAI client, without SDK retry multiplication."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import random
import time

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx
import openai

from nexus.core import OpenAIConfig
from nexus.extensions.openai_clients.openai import OpenAIClient


class BoundaryError(RuntimeError):
    """Only fixed category identifiers cross the Responses error boundary."""

    def __init__(self, category: str):
        self.error_code = f'frontieram_{category}'
        super().__init__(self.error_code)


def _gateway_code(body: Any) -> int | None:
    if not isinstance(body, dict):
        return None
    error = body.get('error', body)
    value = error.get('code') if isinstance(error, dict) else None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return None


def _retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            seconds = (when - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


class BoundedOpenAIClient(OpenAIClient):
    """Use Nexus tracing/conversion but bind inference to an immutable wall deadline.

    ``deadline`` is a monotonic timestamp supplied by the task. All actual HTTP
    sends, retries, and response acceptance share it. Native runner time credits
    therefore cannot extend the solver's development window.
    """

    def __init__(
        self,
        config: OpenAIConfig,
        *,
        deadline: float,
        retry_policy: dict[str, Any],
        max_http_attempts: int = 800,
        audit_path: Path | None = None,
        model: str | None = None,
        max_output_tokens: int = 64000,
        output_parameter: str = 'max_tokens',
        provider_timeout_max_seconds: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ):
        if not math.isfinite(deadline) or not 1 <= max_http_attempts <= 800:
            raise ValueError('Invalid immutable inference budget')
        if output_parameter not in {'max_tokens', 'max_completion_tokens'} or not 1 <= max_output_tokens <= 64000:
            raise ValueError('Invalid output budget')
        policy = retry_policy.get('data', {})
        if retry_policy.get('code') != 0 or not isinstance(policy.get('dispositions'), list):
            raise ValueError('A successful frozen retry policy is required')
        self.deadline = deadline
        self.max_http_attempts = max_http_attempts
        self.http_attempts = 0
        self.retry_count = 0
        self.logical_requests = 0
        self.events: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.audit_path = Path(audit_path) if audit_path is not None else None
        if self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            self.audit_path.touch(exist_ok=False)
        self._clock, self._sleep, self._jitter = clock, sleep, jitter
        self._stopped = False
        self._lock = asyncio.Lock()
        self._model = model or config.request_args.get('model')
        if not isinstance(self._model, str) or not self._model:
            raise ValueError('A fixed model is required')
        self._max_output_tokens = max_output_tokens
        self._output_parameter = output_parameter
        self._provider_timeout_max_seconds = provider_timeout_max_seconds
        self._retry_codes = {
            item['code']
            for item in policy['dispositions']
            if isinstance(item, dict) and item.get('is_retryable') is True and isinstance(item.get('code'), int)
        }
        self._retry_statuses = {429, 500, 502, 503, 504} & set(
            policy.get('backoff', {}).get('retryable_http_status', [])
        )
        args = copy.deepcopy(config.client_args)
        if any(key in args for key in ('http_client', 'default_query', 'websocket_base_url')):
            raise ValueError('Unsupported client transport override')
        base_url = httpx.URL(args.get('base_url', ''))
        if base_url.scheme not in {'http', 'https'} or not base_url.host:
            raise ValueError('An explicit model gateway URL is required')
        self._expected_url = str(base_url).rstrip('/') + '/chat/completions'
        self._http = httpx.AsyncClient(
            transport=transport,
            follow_redirects=False,
            trust_env=False,
            event_hooks={'request': [self._before_http], 'response': [self._after_http]},
        )
        # The SDK's default 600 s read timeout cut slow but valid responses and ended whole attempts as protocol
        # errors; every request is bounded by the immutable deadline below, so the transport must not end it first.
        budget = max(1.0, deadline - clock()) + 30.0
        args.update(
            max_retries=1,
            sdk_max_retries=0,
            http_client=self._http,
            timeout=httpx.Timeout(budget, connect=30.0),
        )
        request_args = copy.deepcopy(config.request_args)
        request_args['model'] = self._model
        enforced = copy.deepcopy(config.enforced_request_args)
        enforced.update(model=self._model, stream=False, n=1)
        enforced[self._output_parameter] = max_output_tokens
        super().__init__(OpenAIConfig(client_args=args, request_args=request_args, enforced_request_args=enforced))
        self._client_key = 'main'

    def record(self, kind: str, **metadata: Any) -> None:
        event = {'kind': kind, 'at_monotonic': self._clock(), **metadata}
        self.events.append(event)
        if kind == 'error':
            self.errors.append(event)
        if self.audit_path is not None:
            with self.audit_path.open('a') as stream:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')

    def check_deadline(self) -> float:
        if self._stopped:
            raise BoundaryError('stopped')
        remaining = self.deadline - self._clock()
        if remaining <= 0:
            raise BoundaryError('deadline')
        return remaining

    def stop(self) -> None:
        self._stopped = True

    async def aclose(self) -> None:
        self.stop()
        await self.client.close()

    async def _before_http(self, request: httpx.Request) -> None:
        self.check_deadline()
        if str(request.url) != self._expected_url or request.method != 'POST':
            raise BoundaryError('unexpected_endpoint')
        if self.http_attempts >= self.max_http_attempts:
            raise BoundaryError('http_budget')
        self.http_attempts += 1
        self.record(
            'http_attempt',
            attempt=self.http_attempts,
            payload_sha256=hashlib.sha256(request.content).hexdigest(),
            remaining_seconds=self.deadline - self._clock(),
        )

    async def _after_http(self, response: httpx.Response) -> None:
        self.record('http_response', attempt=self.http_attempts, status=response.status_code)

    async def _create_completion(self, messages, model=None, **kwargs):
        async with self._lock:
            remaining = self.check_deadline()
            if model not in (None, self._model):
                raise BoundaryError('model_override')
            self.logical_requests += 1
            kwargs = copy.deepcopy(kwargs)
            # Codex's own inferred reasoning default must not silently replace
            # the deliberately chosen provider default in this benchmark.
            for key in ('reasoning_effort', 'verbosity', 'max_tokens', 'max_completion_tokens'):
                kwargs.pop(key, None)
            kwargs['model'] = self._model
            kwargs['stream'] = False
            kwargs[self._output_parameter] = self._max_output_tokens
            if self._provider_timeout_max_seconds is not None:
                extra_body = copy.deepcopy(kwargs.get('extra_body') or {})
                extra_body['timeout'] = max(1, min(int(remaining), self._provider_timeout_max_seconds))
                kwargs['extra_body'] = extra_body
            for retry in range(4):
                try:
                    if self.http_attempts >= self.max_http_attempts:
                        raise BoundaryError('http_budget')
                    async with asyncio.timeout(self.check_deadline()):
                        response = await super()._create_completion(messages, **kwargs)
                    self.check_deadline()
                    return response
                except asyncio.CancelledError:
                    self.record('error', category='cancelled')
                    raise
                except TimeoutError:
                    self.record('error', category='deadline')
                    raise BoundaryError('deadline') from None
                except openai.APIStatusError as error:
                    code = _gateway_code(error.body)
                    eligible = error.status_code in self._retry_statuses and code in self._retry_codes
                    if not eligible or retry == 3:
                        self.record('error', category='upstream_http', status=error.status_code, gateway_code=code)
                        raise BoundaryError('upstream_http') from None
                    remaining = self.check_deadline()
                    delay = _retry_after(error.response.headers.get('retry-after'))
                    if delay is None:
                        delay = self._jitter(0.0, float(2**retry))
                    if delay >= remaining or self.http_attempts >= self.max_http_attempts:
                        self.record('error', category='retry_budget')
                        raise BoundaryError('retry_budget') from None
                    self.retry_count += 1
                    self.record(
                        'retry', retry=self.retry_count, status=error.status_code, gateway_code=code, delay=delay
                    )
                    await self._sleep(delay)
                except BoundaryError as error:
                    self.record('error', category=error.error_code.removeprefix('frontieram_'))
                    raise
                except Exception as error:
                    # SDK may wrap a request-hook exception; keep the true
                    # deadline/count boundary without exposing its raw message.
                    self.check_deadline()
                    category = 'http_budget' if self.http_attempts >= self.max_http_attempts else 'upstream_protocol'
                    self.record('error', category=category, exception_type=type(error).__name__)
                    raise BoundaryError(category) from None
        raise AssertionError('Unreachable retry state')
