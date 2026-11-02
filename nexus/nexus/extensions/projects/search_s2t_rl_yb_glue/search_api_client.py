#!/usr/bin/env python3
"""
独立的 PromptSearch API 调用脚本 (从 stream_orchestrator_search.py 拆出来)。

只负责调用 vr-mcp-server 的 PromptSearch 接口，不依赖 Thinker / Talker / 音频。
支持三种模式：

  single  : 发起单次请求，pretty-print 状态/耗时/响应摘要 (默认)
  batch   : 从 JSONL 文件读取多条用例，串行或并发执行
  bench   : 固定并发 / 固定时长或固定请求数 的压测，输出 P50/P90/P99/avg

示例：

  # 1. 单次调用 (默认 query = prompt)
  python3 search_api_client.py --prompt "腾讯今天股价怎么样"

  # 2. 单次调用，多条子 query，并保存原始响应
  python3 search_api_client.py \
    --prompt "腾讯今天股价怎么样" \
    --query "腾讯控股 0700.HK 今日股价" \
    --query "Tencent stock price today" \
    --save-raw ./search_raw.json

  # 3. 批量从文件 (JSONL: {"prompt": "...", "queries": ["..."]})
  python3 search_api_client.py --mode batch --cases-file ./cases.jsonl --concurrency 4

  # 4. 压测 30s, 并发 8
  python3 search_api_client.py --mode bench --concurrency 8 --duration 30
"""

import argparse
import asyncio
import contextlib
import json
import os
import statistics
import sys
import time
import uuid

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


DEFAULT_SEARCH_URL = 'https://vr-mcp-server.woa.com/trpc.tsbs.vr_mcp_server.VrMcpServerHttp/PromptSearch'

DEFAULT_SEARCH_SOURCE = 'agentic_search'

DEFAULT_SEARCH_EXT_PARAMS: dict[str, str] = {
    'open_google_api': '0',
    'open_wx_video': '0',
    'filter_level': '0',
    'enable_full_content': '1',
}

DEFAULT_PROMPT = '腾讯今天股价怎么样'


# ---------------------------------------------------------------------------
# Search client
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    raw_text: str = ''
    latency_sec: float = 0.0
    http_status: int = 0
    error: str = ''
    request_uuid: str = ''

    @property
    def ok(self) -> bool:
        return not self.error and 200 <= self.http_status < 300


class SearchClient:
    def __init__(
        self,
        url: str = DEFAULT_SEARCH_URL,
        source: str = DEFAULT_SEARCH_SOURCE,
        ext_params: dict[str, str] | None = None,
        time_range: str = '',
    ):
        self.url = url
        self.source = source
        self.ext_params = dict(ext_params or DEFAULT_SEARCH_EXT_PARAMS)
        # Search-backend time filter. Only the `time_range` field is honored by
        # the backend; the protobuf `start_timestamp`/`end_timestamp` fields are
        # ignored server-side (probed). Accepted values:
        #   - enum: OneDay / OneWeek / OneMonth / OneYear
        #   - interval (inclusive both ends, two dots): "YYYY-MM-DD..YYYY-MM-DD"
        # e.g. to block future docs use "2000-01-01..2025-12-31".
        self.time_range = time_range or ''

    def build_payload(
        self,
        prompt: str,
        queries: list[str],
        messages: list[dict] | None = None,
        request_uuid: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        request_uuid = request_uuid or uuid.uuid4().hex
        payload: dict[str, Any] = {
            'header': {
                'session_id': request_uuid,
                'uuid': request_uuid,
                'trace_id': request_uuid,
                'yb_traffic_type': 1,
            },
            'prompt': prompt,
            'source': self.source,
            'sub_query_info': [{'subq': q, 'search_type': 0} for q in queries],
            'messages': messages or [{'role': 'user', 'content': prompt}],
            'params': {'ext_params': self.ext_params},
        }
        # Inject the time filter only when set, so we never pollute a plain
        # request with an empty key (which the backend would reject).
        if self.time_range:
            payload['params']['time_range'] = self.time_range
        return payload, request_uuid

    async def call(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        queries: list[str],
        messages: list[dict] | None = None,
    ) -> SearchResult:
        payload, request_uuid = self.build_payload(prompt, queries, messages)
        # Dump the exact POST body we're about to send so the operator can
        # see the on-the-wire payload (prompt / sub_query_info / ext_params /
        # trace ids) without having to instrument httpx itself.
        try:
            payload_dbg = json.dumps(payload, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            payload_dbg = repr(payload)
        print(
            f'[SearchClient] POST {self.url}\n'
            f'[SearchClient] request_uuid={request_uuid}\n'
            f'[SearchClient] body={payload_dbg}',
            flush=True,
        )
        start = time.perf_counter()
        try:
            resp = await client.post(self.url, json=payload)
            latency = time.perf_counter() - start
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = e.response.text
                err = json.dumps(
                    {
                        'error': 'search_http_error',
                        'status_code': e.response.status_code,
                        'body': body[:2000],
                    },
                    ensure_ascii=False,
                )
                return SearchResult(
                    raw_text=body,
                    latency_sec=latency,
                    http_status=e.response.status_code,
                    error=err,
                    request_uuid=request_uuid,
                )
            return SearchResult(
                raw_text=resp.text,
                latency_sec=latency,
                http_status=resp.status_code,
                request_uuid=request_uuid,
            )
        except Exception as e:
            latency = time.perf_counter() - start
            err = json.dumps(
                {
                    'error': 'search_exception',
                    'exception_type': type(e).__name__,
                    'message': str(e),
                },
                ensure_ascii=False,
            )
            return SearchResult(
                latency_sec=latency,
                error=err,
                request_uuid=request_uuid,
            )


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------


def try_parse_json(text: str) -> Any | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _walk(obj: Any) -> list[Any]:
    """Yield every dict found inside a nested structure."""
    out: list[Any] = []
    if isinstance(obj, dict):
        out.append(obj)
        for v in obj.values():
            out.extend(_walk(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_walk(v))
    return out


def extract_search_items(raw_text: str, max_items: int = 5) -> list[dict[str, Any]]:
    """
    Best-effort extraction of search-result-like items from the PromptSearch JSON.
    The schema isn't strictly stable, so we look for dicts that have at least a
    title/url/snippet-ish key and return up to `max_items` summaries.
    """
    parsed = try_parse_json(raw_text)
    if parsed is None:
        return []

    title_keys = ('title', 'name', 'headline')
    url_keys = ('url', 'link', 'source_url', 'page_url')
    snippet_keys = ('snippet', 'content', 'summary', 'abstract', 'description', 'text')

    items: list[dict[str, Any]] = []
    seen_urls: set = set()
    for d in _walk(parsed):
        title = next((d[k] for k in title_keys if isinstance(d.get(k), str) and d[k]), None)
        url = next((d[k] for k in url_keys if isinstance(d.get(k), str) and d[k]), None)
        snippet = next((d[k] for k in snippet_keys if isinstance(d.get(k), str) and d[k]), None)
        if not title and not url and not snippet:
            continue
        if not (title or snippet):
            continue
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        items.append({
            'title': title or '',
            'url': url or '',
            'snippet': (snippet or '')[:240],
        })
        if len(items) >= max_items:
            break
    return items


# Mapping for PromptSearch `doc_authority_level` -> human-readable Chinese.
# Probed values seen in the wild: -1, 0, 1, 2, 3, 4.
_AUTHORITY_LEVEL_TEXT = {
    -1: '权威性未知',
    0: '权威性未知',
    1: '权威性较低',
    2: '权威性一般',
    3: '权威性较高',
    4: '权威性很高',
}


def authority_level_text(level: Any) -> str:
    """Render a PromptSearch authority level int as Chinese text."""
    try:
        lv = int(level)
    except (TypeError, ValueError):
        return '权威性未知'
    return _AUTHORITY_LEVEL_TEXT.get(lv, '权威性未知')


def extract_web_docs(raw_text: str, max_docs: int = 5) -> list[dict[str, Any]]:
    """
    Extract the top-N web search docs from a PromptSearch response, using the
    real schema observed at runtime: ``parsed["docs"]`` is a list of dicts with
    fields ``title / url / site_name / recall_query / doc_authority_level /
    content / abstract / rank_score / ...``.

    Returns a list (length <= max_docs) of dicts with the keys needed by the
    ``render_tool_result_template`` template:

        {
          "title", "url", "site_name", "search_query",
          "authority_level", "authority_level_text", "content",
          "rank_score", "doc_id", "publish_time",
        }

    Docs are kept in the upstream backend order (no re-sorting) and
    de-duplicated by URL; the first ``max_docs`` survivors are returned.
    Empty ``content`` is preserved as-is (left blank inside the template).
    ``publish_time`` falls back to a few alternate key spellings observed on
    upstream variants and is normalized to a string ("" when missing).
    """
    parsed = try_parse_json(raw_text)
    if not isinstance(parsed, dict):
        return []
    docs = parsed.get('docs')
    if not isinstance(docs, list):
        return []

    def _score(d: dict[str, Any]) -> float:
        v = d.get('rank_score')
        if v is None:
            return 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    out: list[dict[str, Any]] = []
    seen_urls: set = set()
    for d in docs:
        if not isinstance(d, dict):
            continue
        url = d.get('url') or ''
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)

        level_raw = d.get('doc_authority_level')
        # Upstream PromptSearch usually exposes ``publish_time`` (e.g.
        # "2025/10/20 09:34:00"), but a few variants spell it differently
        # (``publishtime`` / ``pub_time`` / ``publish_date``). We probe each
        # in order and coerce to ``str`` so the LLM-facing template never
        # has to deal with ``None``/numeric edge cases.
        publish_time_raw = (
            d.get('publish_time') or d.get('publishtime') or d.get('pub_time') or d.get('publish_date') or ''
        )
        publish_time = str(publish_time_raw) if publish_time_raw else ''
        # Upstream sometimes returns an empty ``content`` string but still
        # carries the human-readable summary in ``abstract``; fall back to
        # ``abstract`` so the LLM is never handed an empty body.
        content = d.get('content') or d.get('abstract') or ''
        out.append({
            'title': d.get('title') or '',
            'url': url,
            'site_name': d.get('site_name') or '',
            'search_query': d.get('recall_query') or '',
            'authority_level': level_raw,
            'authority_level_text': authority_level_text(level_raw),
            'content': content,
            'rank_score': _score(d),
            'doc_id': d.get('doc_id') or '',
            'publish_time': publish_time,
        })
        if len(out) >= max_docs:
            break
    return out


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

SEP = '=' * 90


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f'{n}B'
    if n < 1024 * 1024:
        return f'{n / 1024:.1f}KB'
    return f'{n / 1024 / 1024:.2f}MB'


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def print_single_result(prompt: str, queries: list[str], result: SearchResult, *, show_raw_chars: int) -> None:
    print(SEP)
    print(f'  Prompt:        {prompt}')
    print(f'  Queries:       {queries}')
    print(f'  HTTP status:   {result.http_status}')
    print(f'  Latency:       {result.latency_sec * 1000:.1f} ms')
    print(f'  Response size: {fmt_bytes(len(result.raw_text))} ({len(result.raw_text)} chars)')
    print(f'  Trace UUID:    {result.request_uuid}')
    print(f'  Error:         {"-" if not result.error else result.error[:300]}')
    print(SEP)
    if result.ok:
        items = extract_search_items(result.raw_text, max_items=5)
        if items:
            print('  Top items (best-effort parsed):')
            for i, it in enumerate(items, 1):
                title = it['title'] or '(no title)'
                print(f'    [{i}] {title}')
                if it['url']:
                    print(f'        url: {it["url"]}')
                if it['snippet']:
                    print(f'        > {it["snippet"]}')
        else:
            print('  Top items: (could not auto-extract; use --save-raw for full body)')
        print(SEP)
    if show_raw_chars > 0 and result.raw_text:
        snippet = result.raw_text[:show_raw_chars]
        more = max(0, len(result.raw_text) - show_raw_chars)
        print(f'  Raw response (first {show_raw_chars} chars):')
        print(snippet)
        if more:
            print(f'...<{more} more chars omitted>')
        print(SEP)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


async def run_single(args: argparse.Namespace) -> int:
    client_obj = SearchClient(args.search_url, args.search_source, args.search_ext_params_dict, args.time_range)
    queries = list(args.query) if args.query else [args.prompt]
    timeout = httpx.Timeout(args.timeout, connect=args.connect_timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        result = await client_obj.call(client, args.prompt, queries)

    print_single_result(
        args.prompt,
        queries,
        result,
        show_raw_chars=args.show_raw_chars,
    )

    if args.save_raw:
        Path(args.save_raw).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_raw).write_text(result.raw_text, encoding='utf-8')
        print(f'  Raw response written to: {args.save_raw}')

    return 0 if result.ok else 1


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    suffix = path.suffix.lower()
    if suffix in ('.jsonl', '.ndjson'):
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f'case must be a JSON object, got: {type(obj).__name__}')
            if 'prompt' not in obj:
                raise ValueError("each case JSONL row must contain 'prompt'")
            cases.append(obj)
    elif suffix == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            data = [data]
        for obj in data:
            if not isinstance(obj, dict) or 'prompt' not in obj:
                raise ValueError("each case in JSON array must be an object with 'prompt'")
            cases.append(obj)
    else:
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            cases.append({'prompt': line})
    return cases


@dataclass
class CaseRecord:
    case_index: int
    prompt: str
    queries: list[str]
    result: SearchResult


async def run_batch(args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.cases_file))
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        print('No cases found.', file=sys.stderr)
        return 2

    client_obj = SearchClient(args.search_url, args.search_source, args.search_ext_params_dict, args.time_range)
    timeout = httpx.Timeout(args.timeout, connect=args.connect_timeout)
    sem = asyncio.Semaphore(max(1, args.concurrency))
    records: list[CaseRecord] = []

    async with httpx.AsyncClient(timeout=timeout) as client:

        async def run_one(idx: int, case: dict[str, Any]) -> CaseRecord:
            async with sem:
                prompt = case['prompt']
                queries = case.get('queries') or [prompt]
                if isinstance(queries, str):
                    queries = [queries]
                t0 = time.perf_counter()
                result = await client_obj.call(client, prompt, queries)
                wall = time.perf_counter() - t0
                if args.verbose:
                    status = 'OK' if result.ok else f'FAIL({result.http_status or "X"})'
                    print(
                        f'[{idx + 1}/{len(cases)}] {status} '
                        f'{wall * 1000:.0f}ms '
                        f'size={fmt_bytes(len(result.raw_text))} '
                        f'prompt={prompt[:40]!r}'
                    )
                return CaseRecord(idx, prompt, queries, result)

        records = await asyncio.gather(*(run_one(i, c) for i, c in enumerate(cases)))

    records.sort(key=lambda r: r.case_index)

    latencies = [r.result.latency_sec for r in records if r.result.ok]
    fail = sum(1 for r in records if not r.result.ok)
    sizes = [len(r.result.raw_text) for r in records if r.result.ok]

    print(SEP)
    print(f'  Cases:         {len(records)}  (success: {len(records) - fail}, fail: {fail})')
    if latencies:
        print(
            f'  Latency (ms):  avg={statistics.mean(latencies) * 1000:.1f} '
            f'p50={percentile(latencies, 0.5) * 1000:.1f} '
            f'p90={percentile(latencies, 0.9) * 1000:.1f} '
            f'p99={percentile(latencies, 0.99) * 1000:.1f} '
            f'min={min(latencies) * 1000:.1f} max={max(latencies) * 1000:.1f}'
        )
    if sizes:
        print(
            f'  Resp size:     avg={fmt_bytes(int(statistics.mean(sizes)))} '
            f'min={fmt_bytes(min(sizes))} max={fmt_bytes(max(sizes))}'
        )
    print(SEP)

    if args.report_csv:
        Path(args.report_csv).parent.mkdir(parents=True, exist_ok=True)
        import csv

        with open(args.report_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                'case_index',
                'prompt',
                'queries',
                'ok',
                'http_status',
                'latency_ms',
                'response_chars',
                'trace_uuid',
                'error',
            ])
            for r in records:
                w.writerow([
                    r.case_index,
                    r.prompt,
                    json.dumps(r.queries, ensure_ascii=False),
                    int(r.result.ok),
                    r.result.http_status,
                    f'{r.result.latency_sec * 1000:.2f}',
                    len(r.result.raw_text),
                    r.result.request_uuid,
                    r.result.error[:500],
                ])
        print(f'  CSV written to: {args.report_csv}')

    return 0 if fail == 0 else 1


@dataclass
class BenchSnapshot:
    completed: int = 0
    success: int = 0
    fail: int = 0
    latencies: list[float] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)
    status_counts: dict[int, int] = field(default_factory=dict)
    error_samples: list[str] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def add(self, result: SearchResult) -> None:
        async with self.lock:
            self.completed += 1
            self.status_counts[result.http_status] = self.status_counts.get(result.http_status, 0) + 1
            if result.ok:
                self.success += 1
                self.latencies.append(result.latency_sec)
                self.sizes.append(len(result.raw_text))
            else:
                self.fail += 1
                if len(self.error_samples) < 5 and result.error:
                    self.error_samples.append(result.error[:300])

    def snapshot(self) -> dict[str, Any]:
        return {
            'completed': self.completed,
            'success': self.success,
            'fail': self.fail,
            'status_counts': dict(self.status_counts),
            'latencies': list(self.latencies),
            'sizes': list(self.sizes),
            'error_samples': list(self.error_samples),
        }


async def run_bench(args: argparse.Namespace) -> int:
    prompts: list[str]
    if args.cases_file:
        cases = load_cases(Path(args.cases_file))
        prompts = [c['prompt'] for c in cases]
    elif args.prompt_pool:
        prompts = list(args.prompt_pool)
    else:
        prompts = [args.prompt]
    if not prompts:
        print('Bench needs at least one prompt.', file=sys.stderr)
        return 2

    client_obj = SearchClient(args.search_url, args.search_source, args.search_ext_params_dict, args.time_range)
    timeout = httpx.Timeout(args.timeout, connect=args.connect_timeout)
    snap = BenchSnapshot()

    stop = asyncio.Event()
    if args.duration > 0:

        async def _stop_after():
            try:
                await asyncio.sleep(args.duration)
            finally:
                stop.set()

        stopper = asyncio.create_task(_stop_after())
    else:
        stopper = None

    counter = {'next': 0}
    counter_lock = asyncio.Lock()
    target_total = args.max_requests if args.max_requests > 0 else None

    async def reserve_slot() -> int | None:
        async with counter_lock:
            if stop.is_set():
                return None
            if target_total is not None and counter['next'] >= target_total:
                return None
            idx = counter['next']
            counter['next'] += 1
            return idx

    async def worker(worker_id: int) -> None:
        async with httpx.AsyncClient(timeout=timeout) as client:
            while True:
                idx = await reserve_slot()
                if idx is None:
                    return
                prompt = prompts[idx % len(prompts)]
                queries = [prompt]
                result = await client_obj.call(client, prompt, queries)
                await snap.add(result)
                if stop.is_set():
                    return

    async def reporter() -> None:
        prev = 0
        prev_t = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(args.report_interval)
            now = time.perf_counter()
            cur = snap.completed
            qps = (cur - prev) / max(1e-6, now - prev_t)
            prev, prev_t = cur, now
            lat_p50 = percentile(snap.latencies, 0.5) * 1000 if snap.latencies else 0.0
            lat_p90 = percentile(snap.latencies, 0.9) * 1000 if snap.latencies else 0.0
            print(
                f'[bench] completed={cur} ok={snap.success} fail={snap.fail} '
                f'qps_5s={qps:.2f} p50={lat_p50:.0f}ms p90={lat_p90:.0f}ms',
                flush=True,
            )

    print(SEP)
    print('  Mode:           bench')
    print(f'  Search URL:     {args.search_url}')
    print(f'  Concurrency:    {args.concurrency}')
    if args.duration > 0:
        print(f'  Duration:       {args.duration}s')
    if target_total is not None:
        print(f'  Max requests:   {target_total}')
    print(f'  Prompts:        {len(prompts)} unique')
    print(SEP)

    started = time.perf_counter()
    workers = [asyncio.create_task(worker(i)) for i in range(args.concurrency)]
    rep = asyncio.create_task(reporter())

    try:
        await asyncio.gather(*workers)
    finally:
        stop.set()
        rep.cancel()
        if stopper:
            stopper.cancel()
        for t in (rep, stopper):
            if t is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t
    elapsed = time.perf_counter() - started

    print(SEP)
    print(f'  Elapsed:        {elapsed:.2f}s')
    print(f'  Completed:      {snap.completed} (ok={snap.success}, fail={snap.fail})')
    if snap.completed:
        print(f'  Throughput:     {snap.completed / elapsed:.2f} req/s')
    if snap.latencies:
        lat = snap.latencies
        print(
            f'  Latency (ms):   avg={statistics.mean(lat) * 1000:.1f} '
            f'p50={percentile(lat, 0.5) * 1000:.1f} '
            f'p90={percentile(lat, 0.9) * 1000:.1f} '
            f'p95={percentile(lat, 0.95) * 1000:.1f} '
            f'p99={percentile(lat, 0.99) * 1000:.1f} '
            f'min={min(lat) * 1000:.1f} max={max(lat) * 1000:.1f}'
        )
    if snap.sizes:
        print(
            f'  Resp size:      avg={fmt_bytes(int(statistics.mean(snap.sizes)))} '
            f'min={fmt_bytes(min(snap.sizes))} max={fmt_bytes(max(snap.sizes))}'
        )
    if snap.status_counts:
        parts = [f'{k}:{v}' for k, v in sorted(snap.status_counts.items())]
        print(f'  Status counts:  {" ".join(parts)}')
    if snap.error_samples:
        print('  Error samples (up to 5):')
        for e in snap.error_samples:
            print(f'    - {e}')
    print(SEP)

    return 0 if snap.fail == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='PromptSearch API 调用客户端 (从 stream_orchestrator_search.py 抽出)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--mode', choices=['single', 'batch', 'bench'], default='single')

    p.add_argument(
        '--search-url',
        default=os.environ.get('SEARCH_URL', DEFAULT_SEARCH_URL),
        help='PromptSearch URL',
    )
    p.add_argument(
        '--search-source',
        default=os.environ.get('SEARCH_SOURCE', DEFAULT_SEARCH_SOURCE),
        help='PromptSearch source 字段',
    )
    p.add_argument(
        '--search-ext-params',
        default=os.environ.get('SEARCH_EXT_PARAMS', ''),
        help='覆盖 params.ext_params 的 JSON 字符串 (留空使用默认)',
    )
    p.add_argument(
        '--timeout',
        type=float,
        default=float(os.environ.get('SEARCH_TIMEOUT', '30')),
        help='每次请求总超时 (秒)',
    )
    p.add_argument(
        '--connect-timeout',
        type=float,
        default=float(os.environ.get('SEARCH_CONNECT_TIMEOUT', '5')),
        help='连接超时 (秒)',
    )

    p.add_argument(
        '--time-range',
        default=os.environ.get('SEARCH_TIME_RANGE', ''),
        help='截止日期过滤，屏蔽该日期之后的 doc。枚举 OneDay/OneWeek/OneMonth/OneYear，'
        '或区间 "YYYY-MM-DD..YYYY-MM-DD"（闭区间，含两端），如 "2000-01-01..2025-12-31"',
    )
    p.add_argument(
        '--prompt',
        default=os.environ.get('SEARCH_PROMPT', DEFAULT_PROMPT),
        help='single/bench 模式的 prompt',
    )
    p.add_argument(
        '--query',
        action='append',
        default=None,
        help='single 模式的子 query (可多次指定)。不传则用 prompt',
    )
    p.add_argument(
        '--prompt-pool',
        action='append',
        default=None,
        help='bench 模式的 prompt 池 (可多次指定)，会按 round-robin 选用',
    )

    p.add_argument(
        '--cases-file',
        default='',
        help='batch 模式的用例文件 (JSONL: {prompt, queries?} / 或纯文本一行一 prompt)',
    )
    p.add_argument(
        '--limit',
        type=int,
        default=0,
        help='batch 模式只跑前 N 条; 0=不限',
    )
    p.add_argument(
        '--concurrency',
        type=int,
        default=int(os.environ.get('SEARCH_CONCURRENCY', '1')),
        help='batch / bench 并发数',
    )
    p.add_argument(
        '--duration',
        type=int,
        default=int(os.environ.get('SEARCH_BENCH_DURATION', '30')),
        help='bench 持续秒数; 0=不按时间停 (需配合 --max-requests)',
    )
    p.add_argument(
        '--max-requests',
        type=int,
        default=int(os.environ.get('SEARCH_BENCH_MAX_REQUESTS', '0')),
        help='bench 最大请求数; 0=不限',
    )
    p.add_argument(
        '--report-interval',
        type=float,
        default=float(os.environ.get('SEARCH_BENCH_REPORT_INTERVAL', '5')),
        help='bench 中间报告打印间隔 (秒)',
    )

    p.add_argument(
        '--save-raw',
        default='',
        help='single 模式: 保存原始响应到此文件',
    )
    p.add_argument(
        '--show-raw-chars',
        type=int,
        default=int(os.environ.get('SHOW_RAW_CHARS', '0')),
        help='single 模式: 在终端打印前 N 个字符的原始响应 (0=不打印)',
    )
    p.add_argument(
        '--report-csv',
        default='',
        help='batch 模式: 把每条用例结果写到这个 CSV',
    )
    p.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='batch 模式: 每条用例都打印一行进度',
    )

    args = p.parse_args(argv)

    if args.search_ext_params:
        try:
            parsed = json.loads(args.search_ext_params)
        except json.JSONDecodeError as e:
            p.error(f'--search-ext-params 不是合法 JSON: {e}')
        if not isinstance(parsed, dict):
            p.error('--search-ext-params 必须是 JSON object')
        args.search_ext_params_dict = {str(k): str(v) for k, v in parsed.items()}
    else:
        args.search_ext_params_dict = dict(DEFAULT_SEARCH_EXT_PARAMS)

    if args.mode == 'batch' and not args.cases_file:
        p.error('--mode batch 需要 --cases-file')
    if args.mode == 'bench' and args.duration <= 0 and args.max_requests <= 0:
        p.error('--mode bench 需要至少设置 --duration 或 --max-requests 之一')

    return args


async def amain(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == 'single':
        return await run_single(args)
    if args.mode == 'batch':
        return await run_batch(args)
    if args.mode == 'bench':
        return await run_bench(args)
    return 2


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(amain(argv))
    except KeyboardInterrupt:
        print('\nInterrupted.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    sys.exit(main())
