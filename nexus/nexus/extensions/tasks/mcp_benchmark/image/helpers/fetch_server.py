#!/usr/bin/env python3
from __future__ import annotations

import multiprocessing as mp
import time

from urllib.parse import urlparse

import requests

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ErrorData, TextContent, Tool
from mcp_server_fetch.server import Fetch, extract_content_from_html


DEFAULT_BROWSER_USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
)

REQUEST_HEADERS = {
    'User-Agent': DEFAULT_BROWSER_USER_AGENT,
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Cache-Control': 'no-cache',
    'Pragma': 'no-cache',
}
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 10
TOTAL_TIMEOUT = 45
MAX_DOWNLOAD_BYTES = 2_000_000
NON_TEXT_CONTENT_TYPES = (
    'application/pdf',
    'application/octet-stream',
    'application/zip',
    'application/x-gzip',
    'application/gzip',
)


def _jina_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc == 'r.jina.ai':
        return url
    suffix = url.split('://', 1)[1] if '://' in url else url
    return f'https://r.jina.ai/http://{suffix}'


def _response_to_content(url: str, response: requests.Response, *, raw: bool) -> tuple[str, str]:
    response.raise_for_status()
    page_raw = _read_response_text(response)
    content_type = response.headers.get('content-type', '')

    if urlparse(response.url).netloc == 'r.jina.ai':
        return page_raw, ''

    is_page_html = '<html' in page_raw[:100].lower() or 'text/html' in content_type.lower() or not content_type
    if is_page_html and not raw:
        return extract_content_from_html(page_raw), ''

    return page_raw, f'Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n'


def _should_use_fallback(response: requests.Response, *, candidate: str, fallback: str) -> bool:
    if candidate == fallback:
        return False
    content_type = (response.headers.get('content-type') or '').lower()
    return any(content_type.startswith(prefix) for prefix in NON_TEXT_CONTENT_TYPES)


def _read_response_text(response: requests.Response) -> str:
    deadline = time.monotonic() + TOTAL_TIMEOUT
    chunks: list[bytes] = []
    total_bytes = 0
    try:
        for chunk in response.iter_content(chunk_size=16384, decode_unicode=False):
            if time.monotonic() > deadline:
                raise requests.Timeout(f'total download time exceeded {TOTAL_TIMEOUT}s')
            if not chunk:
                continue
            remaining = MAX_DOWNLOAD_BYTES - total_bytes
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk[:remaining]
            chunks.append(chunk)
            total_bytes += len(chunk)
    finally:
        response.close()
    encoding = response.encoding or response.apparent_encoding or 'utf-8'
    return b''.join(chunks).decode(encoding, errors='replace')


def _fetch_impl(url: str, *, raw: bool) -> tuple[str, str]:
    errors: list[str] = []
    candidate_urls = [url]
    fallback = _jina_url(url)
    if fallback != url:
        candidate_urls.append(fallback)

    for candidate in candidate_urls:
        response: requests.Response | None = None
        try:
            response = requests.get(
                candidate,
                headers=REQUEST_HEADERS,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                allow_redirects=True,
                stream=True,
            )
            if response.status_code in (401, 403) and candidate != fallback:
                errors.append(f'{candidate} -> status {response.status_code}')
                response.close()
                continue
            if _should_use_fallback(response, candidate=candidate, fallback=fallback):
                errors.append(
                    f'{candidate} -> content-type {response.headers.get("content-type", "")} requires jina fallback'
                )
                response.close()
                continue
            return _response_to_content(candidate, response, raw=raw)
        except Exception as exc:
            errors.append(f'{candidate} -> {type(exc).__name__}: {exc}')
            if response is not None:
                response.close()

    raise McpError(
        ErrorData(
            code=INTERNAL_ERROR,
            message=f'Failed to fetch {url}. Attempts: {"; ".join(errors)}',
        )
    )


def _fetch_child(queue: mp.Queue, url: str, raw: bool) -> None:
    try:
        content, prefix = _fetch_impl(url, raw=raw)
    except Exception as exc:  # noqa: BLE001
        queue.put({'ok': False, 'message': str(exc)})
        return
    queue.put({'ok': True, 'content': content, 'prefix': prefix})


def _fetch(url: str, *, raw: bool) -> tuple[str, str]:
    ctx = mp.get_context('spawn')
    queue: mp.Queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_fetch_child, args=(queue, url, raw))
    try:
        process.start()
        process.join(TOTAL_TIMEOUT + 5)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f'Failed to fetch {url}. Hard timeout exceeded {TOTAL_TIMEOUT}s.',
                )
            )

        if queue.empty():
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f'Failed to fetch {url}. Fetch worker exited without a result (exit code {process.exitcode}).',
                )
            )

        result = queue.get()
        if not result.get('ok'):
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=str(result.get('message') or f'Fetch worker failed for {url}.'),
                )
            )
        return str(result['content']), str(result['prefix'])
    finally:
        queue.close()
        queue.join_thread()


def _truncate(content: str, *, max_length: int, start_index: int) -> str:
    if start_index >= len(content):
        return '<error>No more content available.</error>'

    truncated = content[start_index : start_index + max_length]
    if not truncated:
        return '<error>No more content available.</error>'

    next_start = start_index + len(truncated)
    if len(truncated) == max_length and next_start < len(content):
        truncated += (
            f'\n\n<error>Content truncated. Call the fetch tool with a start_index of {next_start} '
            'to get more content.</error>'
        )
    return truncated


async def main() -> None:
    server = Server('nexus-benchmark-fetch')

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name='fetch',
                description=(
                    'Fetches a URL from the internet and extracts its contents as markdown. '
                    'This benchmark variant retries via r.jina.ai when direct fetching is blocked.'
                ),
                inputSchema=Fetch.model_json_schema(),
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name != 'fetch':
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f'Unknown tool: {name}'))

        try:
            args = Fetch(**arguments)
        except ValueError as exc:
            raise McpError(ErrorData(code=INVALID_PARAMS, message=str(exc))) from exc

        content, prefix = _fetch(str(args.url), raw=args.raw)
        content = _truncate(content, max_length=args.max_length, start_index=args.start_index)
        return [TextContent(type='text', text=f'{prefix}Contents of {args.url}:\n{content}')]

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options, raise_exceptions=True)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
