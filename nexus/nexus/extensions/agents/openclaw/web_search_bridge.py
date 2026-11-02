"""Local Brave-compatible web search bridge backed by Serper."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import aiohttp

from nexus.utils.event_loop import create_detached_task


logger = logging.getLogger(__name__)


DEFAULT_PUBLIC_SERPER_URL = 'https://google.serper.dev/search'
DISTILLATION_SERPER_URL_ENV = 'SEARCH_SERPER_URL'


@dataclass(frozen=True)
class SerperBridgeConfig:
    api_key: str
    api_type: str = 'official'
    search_url: str | None = None

    @property
    def normalized_api_type(self) -> str:
        return self.api_type.strip().lower() or 'official'


@dataclass(frozen=True)
class SerperRequest:
    url: str
    headers: dict[str, str]
    body: dict[str, Any]


class UnsupportedBraveQueryError(ValueError):
    """Raised when a Brave query parameter cannot be represented by Serper."""


_BRAVE_FRESHNESS_TO_SERPER_TBS = {
    'pd': 'qdr:d',
    'pw': 'qdr:w',
    'pm': 'qdr:m',
    'py': 'qdr:y',
}
_BRAVE_DATE_RANGE_RE = re.compile(r'^(?P<start>\d{4}-\d{2}-\d{2})?to(?P<end>\d{4}-\d{2}-\d{2})?$')


def resolve_serper_bridge_config(env: Mapping[str, str] | None = None) -> SerperBridgeConfig | None:
    """Resolve Serper bridge settings from environment variables."""
    source = os.environ if env is None else env
    api_type = (source.get('SERPER_API_TYPE') or 'official').strip().lower()
    api_key = (source.get('SERPER_API_KEY') or '').strip()
    search_url = (source.get('SEARCH_SERPER_URL') or '').strip() or None

    if api_type == 'distillation' and not api_key:
        app_id = (source.get('SEARCH_SERPER_APP_ID') or source.get('SERPER_KEY_ID') or '').strip()
        app_key = (source.get('SEARCH_SERPER_APP_KEY') or '').strip()
        if app_id and app_key:
            api_key = f'{app_id}:{app_key}'

    if api_type == 'distillation' and not search_url:
        logger.warning('Serper bridge distillation mode requires %s', DISTILLATION_SERPER_URL_ENV)
        return None

    if not api_key:
        return None

    return SerperBridgeConfig(
        api_key=api_key,
        api_type=api_type,
        search_url=search_url,
    )


def _coerce_count(count: int | str | None) -> int:
    try:
        parsed = int(count) if count is not None else 10
    except (TypeError, ValueError):
        parsed = 10
    return max(1, min(parsed, 10))


def _normalize_locale(value: str | None) -> str | None:
    normalized = (value or '').strip()
    return normalized or None


def _resolve_serper_language(
    *,
    search_lang: str | None,
    language: str | None,
    ui_lang: str | None,
) -> str | None:
    candidates = [
        _normalize_locale(search_lang),
        _normalize_locale(language),
        _normalize_locale(ui_lang),
    ]
    resolved = next((candidate for candidate in candidates if candidate), None)
    if resolved is None:
        return None

    resolved_prefix = resolved.split('-', 1)[0].lower()
    for candidate in candidates:
        if candidate is None:
            continue
        candidate_prefix = candidate.split('-', 1)[0].lower()
        if candidate.lower() != resolved.lower() and candidate_prefix != resolved_prefix:
            raise UnsupportedBraveQueryError(
                'unsupported Brave query parameters: search_lang/language/ui_lang cannot be mapped to one Serper hl'
            )
    return resolved


def _format_serper_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise UnsupportedBraveQueryError(f'unsupported Brave freshness date {value!r}; expected YYYY-MM-DD') from exc
    return f'{parsed.month}/{parsed.day}/{parsed.year}'


def _map_freshness_to_serper_tbs(freshness: str | None) -> str | None:
    value = (freshness or '').strip()
    if not value:
        return None

    relative = _BRAVE_FRESHNESS_TO_SERPER_TBS.get(value.lower())
    if relative:
        return relative

    match = _BRAVE_DATE_RANGE_RE.fullmatch(value)
    if match is None:
        raise UnsupportedBraveQueryError(
            'unsupported Brave freshness value; expected one of pd, pw, pm, py, or YYYY-MM-DDtoYYYY-MM-DD'
        )

    start = match.group('start')
    end = match.group('end')
    if not start and not end:
        raise UnsupportedBraveQueryError('unsupported Brave freshness value; date range must include a bound')

    parts = ['cdr:1']
    if start:
        parts.append(f'cd_min:{_format_serper_date(start)}')
    if end:
        parts.append(f'cd_max:{_format_serper_date(end)}')
    return ','.join(parts)


def build_serper_request(
    *,
    query: str,
    count: int | str | None,
    config: SerperBridgeConfig,
    search_lang: str | None = None,
    language: str | None = None,
    country: str | None = None,
    ui_lang: str | None = None,
    freshness: str | None = None,
) -> SerperRequest:
    """Build the Serper-compatible upstream request for a Brave-style query."""
    body: dict[str, Any] = {
        'q': query,
        'num': _coerce_count(count),
    }
    serper_language = _resolve_serper_language(search_lang=search_lang, language=language, ui_lang=ui_lang)
    if serper_language:
        body['hl'] = serper_language
    if country:
        body['gl'] = country
    serper_tbs = _map_freshness_to_serper_tbs(freshness)
    if serper_tbs:
        body['tbs'] = serper_tbs

    if config.normalized_api_type == 'distillation':
        if not config.search_url:
            raise ValueError(f'{DISTILLATION_SERPER_URL_ENV} is required for Serper bridge distillation mode')
        return SerperRequest(
            url=config.search_url,
            headers={
                'Authorization': f'Bearer {config.api_key}?provider=serper',
                'Content-Type': 'application/json',
            },
            body=body,
        )

    return SerperRequest(
        url=config.search_url or DEFAULT_PUBLIC_SERPER_URL,
        headers={
            'Content-Type': 'application/json',
            'X-API-KEY': config.api_key,
        },
        body=body,
    )


def map_serper_to_brave(data: Mapping[str, Any]) -> dict[str, Any]:
    """Map Serper search JSON into the subset of Brave Search used by OpenClaw."""
    results: list[dict[str, str]] = []
    organic = data.get('organic')
    if not isinstance(organic, list):
        organic = []

    for entry in organic:
        if not isinstance(entry, Mapping):
            continue
        url = str(entry.get('link') or entry.get('url') or '').strip()
        if not url:
            continue
        results.append({
            'title': str(entry.get('title') or ''),
            'url': url,
            'description': str(entry.get('snippet') or entry.get('description') or ''),
            'age': str(entry.get('date') or entry.get('age') or ''),
        })

    return {'web': {'results': results}}


def _sanitize_detail(detail: str, config: SerperBridgeConfig) -> str:
    bearer = f'Bearer {config.api_key}?provider=serper'
    sanitized = detail.replace(bearer, 'Bearer [redacted]')
    sanitized = sanitized.replace(config.api_key, '[redacted]')
    for secret_part in config.api_key.split(':'):
        if secret_part:
            sanitized = sanitized.replace(secret_part, '[redacted]')
    return sanitized


class SerperBridge:
    """Small local HTTP server that exposes a Brave-compatible search API."""

    def __init__(
        self,
        *,
        config: SerperBridgeConfig,
        listen_port: int,
        listen_host: str = '127.0.0.1',
        request_timeout: float = 30,
    ):
        self.config = config
        self.host = listen_host
        self.port = listen_port
        self.request_timeout = request_timeout
        self._server: Any | None = None
        self._server_task: asyncio.Task | None = None

    @property
    def base_url(self) -> str:
        return f'http://{self.host}:{self.port}'

    def _create_app(self):
        from fastapi import FastAPI, HTTPException, Request

        app = FastAPI(title='Nexus Serper Bridge')

        @app.get('/health')
        async def health() -> dict[str, Any]:
            return {'status': 'ok', 'api_type': self.config.normalized_api_type}

        async def web_search(request) -> dict[str, Any]:
            query = (request.query_params.get('q') or '').strip()
            if not query:
                raise HTTPException(status_code=400, detail='q is required')

            try:
                upstream = build_serper_request(
                    query=query,
                    count=request.query_params.get('count'),
                    config=self.config,
                    search_lang=request.query_params.get('search_lang'),
                    language=request.query_params.get('language'),
                    country=request.query_params.get('country'),
                    ui_lang=request.query_params.get('ui_lang'),
                    freshness=request.query_params.get('freshness'),
                )
            except UnsupportedBraveQueryError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            try:
                async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
                    async with session.post(upstream.url, headers=upstream.headers, json=upstream.body) as response:
                        raw = await response.text()
                        if response.status >= 400:
                            detail = raw or response.reason or 'Serper upstream request failed'
                            raise HTTPException(
                                status_code=response.status,
                                detail=_sanitize_detail(detail, self.config),
                            )
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError as exc:
                            raise HTTPException(
                                status_code=502, detail='Serper upstream returned invalid JSON'
                            ) from exc
            except HTTPException:
                raise
            except Exception as exc:
                logger.warning('Serper bridge request failed: %s', exc)
                raise HTTPException(status_code=502, detail='Serper upstream request failed') from exc

            return map_serper_to_brave(payload)

        web_search.__annotations__['request'] = Request
        app.add_api_route('/res/v1/web/search', web_search, methods=['GET'])

        return app

    async def start(self) -> None:
        import uvicorn

        self._server = uvicorn.Server(
            uvicorn.Config(self._create_app(), host=self.host, port=self.port, log_level='warning', access_log=False)
        )
        self._server_task = create_detached_task(self._server.serve(), name='nexus-openclaw-serper-bridge')

        for _ in range(50):
            if self._server_task.done():
                self._server_task.result()
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=0.2)) as session:
                    async with session.get(f'{self.base_url}/health') as response:
                        if response.status == 200:
                            return
            except Exception:
                await asyncio.sleep(0.1)

        await self.stop()
        raise RuntimeError(f'Serper bridge did not start on {self.base_url}')

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._server_task is not None:
            await self._server_task
            self._server_task = None
        self._server = None
