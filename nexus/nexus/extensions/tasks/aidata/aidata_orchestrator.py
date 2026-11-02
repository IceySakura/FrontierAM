#!/usr/bin/env python3
"""aidata_orchestrator —— 基于 ``nexus run-single`` 的整轮编排器。

计算侧是唯一的 writer ，以脱离终端的后台守护进程运行；可视化组件
（``monitor`` TUI + ``dashboard`` 本地站点）是独立的 reader 进程，只读
``--out`` 目录，因此可以随时启动/停止而不影响正在跑的 job 。 关闭终端
或 SSH 会话不会影响 ``--daemon`` 模式的运行。

设计（见 canvas plan ）：

- Planner    ：枚举 case × model × rep -> run_key = case__model__repN
- Scheduler  ：受限（AIMD 自适应）并发；每次运行 wall timeout + 进程组
               kill ；结果分类；退避重试
- Compute    ：每次运行都是独立 session 里的 ``uv run nexus run-single`` ；
               per-run 的 gongfeng 沙箱天然做到 mock/state 隔离
- Buffer     ：runs/<run_key>/result.json 是持久的 source of truth
- Delivery   ：单一 writer 把归一化记录流式写入按小时分区
               （stream/dt=.../hour=.../part.jsonl ）并用 _SUCCESS 封存

子命令::

    run [--daemon]   启动（或恢复）一次 campaign
    monitor          终端 TUI reader （若安装 rich 则用之）
    dashboard        本地站点 reader （stdlib http.server ）
    status           一次性打印进度
    stop             优雅停止（SIGTERM 守护进程；封存分区后退出）

只依赖标准库（``rich`` 存在则用之，否则 TUI 回退到普通刷新文本）。
子进程通过 ``uv run nexus`` 启动。
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import random
import re
import shutil
import signal
import sys
import threading
import time

from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# 布局 / 常量
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent  # .../aidata
NEXUS_REPO = SCRIPT_DIR.parents[3]  # .../nexus （仓库根目录）
TEMPLATE = SCRIPT_DIR / 'runtime.template.yaml'  # litellm （默认 / 非 Claude ）
TEMPLATE_ANTHROPIC = SCRIPT_DIR / 'runtime.anthropic.template.yaml'  # vertex_anthropic passthrough （Claude ）
UV = shutil.which('uv') or os.path.expanduser('~/.local/bin/uv')

# 各模型的网关配置 —— 只涉及 SEARCH_OPENAI_* 。 凭据不在这里硬编码：
# GONGFENG_* 、 JUDGE_* 、以及默认的 SEARCH_OPENAI_API_KEY 都从
# nexus/.env （git-ignored ）读取； ``opus-4-8`` 复用其中的 API key 。
#
# 编排器根据模型族（见 `_is_anthropic` ）挑选运行时模板：
# Claude / Anthropic 模型（SEARCH_OPENAI_MODEL 含 'claude'/'anthropic' ）
# 会渲染 runtime.anthropic.template.yaml （vertex_anthropic passthrough ，
# 原生 prefix 缓存）；其余模型渲染 runtime.template.yaml （litellm ）。
MODELS: dict[str, dict[str, str]] = {
    'opus-4-8': {
        'SEARCH_OPENAI_BASE_URL': 'http://trpc-gpt-eval.production.polaris:8080',
        'SEARCH_OPENAI_MODEL': 'anthropic.claude-opus-4-8',
    },
    'hy3': {
        'SEARCH_OPENAI_BASE_URL': 'http://aidata-llm-proxy-pro.polaris:8000/v1',
        'SEARCH_OPENAI_MODEL': 'HY3-Preview-ZW-AIData',
        'SEARCH_OPENAI_API_KEY': 'placeholder',
    },
}

# score_valid=False 但可重试（临时性的基础设施 / harness / judge 问题）。
INFRA_KINDS = {
    'runtime_acquire_failed',
    'service_setup_failed',
    'bootstrap_failed',
    'judge_unavailable',
    'agent_timeout',
}
# score_valid=False 且是确定性错误（case 自身 bug ）-> 隔离，不重试。
CASE_BUG_KINDS = {
    'case_config_error',
    'missing_rubric',
    'missing_user_prompt',
    'reward_script_error',
    'rubric_load_failed',
}

_PENDING_RE = re.compile(r'^pending_upstream:\s*(.+?)\s*$', re.MULTILINE)
_AGENT_TIMEOUT_RE = re.compile(r'^agent_timeout:\s*([0-9]+)\s*$', re.MULTILINE)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(f'[{now_iso()}] {msg}', flush=True)


def sanitize(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]', '_', s)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp-{os.getpid()}-{random.randint(0, 1 << 30)}')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False))


def load_env_file(p: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not p.is_file():
        return out
    for raw in p.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        if line.startswith('export '):
            line = line[len('export ') :].lstrip()
        k, _, v = line.partition('=')
        k = k.strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
            v = v[1:-1]
        out[k] = v
    return out


def manifest_text(case_dir: Path) -> str:
    for name in ('task.yaml', 'case.yaml'):
        p = case_dir / name
        if p.exists():
            with contextlib.suppress(Exception):
                return p.read_text(encoding='utf-8')
    return ''


def is_pending_upstream(text: str) -> bool:
    m = _PENDING_RE.search(text)
    if not m:
        return False
    val = m.group(1).strip().strip('\'"').strip()
    return bool(val)


def case_agent_timeout(text: str) -> int | None:
    m = _AGENT_TIMEOUT_RE.search(text)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# 自适应（AIMD ）并发限流器
# ---------------------------------------------------------------------------


class AdaptiveLimiter:
    """一个可调整大小的并发闸门。 ``limit`` 会按 AIMD 策略随基础设施
    失败率（gongfeng 沙箱饱和度的代理指标）动态收缩/扩张。"""

    def __init__(self, start: int, lo: int, hi: int) -> None:
        self.limit = start
        self.lo = lo
        self.hi = hi
        self.in_flight = 0
        self._cond = asyncio.Condition()

    async def acquire(self) -> None:
        async with self._cond:
            while self.in_flight >= self.limit:
                await self._cond.wait()
            self.in_flight += 1

    async def release(self) -> None:
        async with self._cond:
            self.in_flight -= 1
            self._cond.notify()

    async def set_limit(self, n: int) -> None:
        async with self._cond:
            self.limit = max(self.lo, min(self.hi, n))
            self._cond.notify_all()


# ---------------------------------------------------------------------------
# 实时统计（单个 event-loop 线程写入； reader 只做快照）
# ---------------------------------------------------------------------------


class Stats:
    def __init__(self, total: int) -> None:
        self.total = total
        self.resumed = 0
        self.ok = 0
        self.quarantine = 0
        self.failed = 0
        self.running = 0
        self.retrying = 0
        self.attempts = 0
        self.start_ts = time.time()
        self.completion_ts: deque[float] = deque(maxlen=5000)
        self.attempt_log: deque[tuple[float, bool]] = deque(maxlen=5000)
        self.recent: deque[dict] = deque(maxlen=30)
        self.per_hour: dict[str, int] = {}

    def record_attempt(self, is_infra: bool) -> None:
        self.attempts += 1
        self.attempt_log.append((time.time(), is_infra))

    def finalize(self, outcome: str, rec: dict) -> None:
        if outcome == 'ok':
            self.ok += 1
        elif outcome == 'quarantine':
            self.quarantine += 1
        else:
            self.failed += 1
        self.completion_ts.append(time.time())
        self.recent.appendleft({
            'run_key': rec['run_key'],
            'outcome': outcome,
            'score': rec.get('score'),
            'wall_s': rec.get('wall_time_s'),
            'end_time': rec.get('end_time'),
        })
        et = rec.get('end_time') or now_iso()
        with contextlib.suppress(Exception):
            dt = datetime.fromisoformat(et).astimezone(timezone.utc)
            hk = dt.strftime('%Y-%m-%d %H:00')
            self.per_hour[hk] = self.per_hour.get(hk, 0) + 1

    def infra_rate(self, window_s: float = 30.0) -> tuple[float, int]:
        now = time.time()
        rows = [inf for ts, inf in self.attempt_log if now - ts <= window_s]
        if not rows:
            return 0.0, 0
        return sum(1 for x in rows if x) / len(rows), len(rows)

    def snapshot(self, ctx: Orchestrator) -> dict:
        now = time.time()
        terminal = self.ok + self.quarantine + self.failed
        recent = [ts for ts in self.completion_ts if now - ts <= 600]
        window = min(now - self.start_ts, 600) or 1.0
        tpm = len(recent) / (window / 60.0)
        remaining = max(0, self.total - self.resumed - terminal)
        eta = int(remaining / (tpm / 60.0)) if tpm > 0 else None
        pending = max(0, self.total - self.resumed - terminal - self.running - self.retrying)
        return {
            'updated_at': now_iso(),
            'pid': os.getpid(),
            'out': str(ctx.out),
            'models': ctx.models,
            'reps': ctx.reps,
            'concurrency_max': ctx.limiter.hi,
            'total': self.total,
            'resumed': self.resumed,
            'ok': self.ok,
            'quarantine': self.quarantine,
            'failed': self.failed,
            'running': self.running,
            'retrying': self.retrying,
            'pending': pending,
            'terminal': terminal,
            'valid_rate': (self.ok / terminal) if terminal else 0.0,
            'throughput_per_min': round(tpm, 2),
            'eta_seconds': eta,
            'limit': ctx.limiter.limit,
            'in_flight': ctx.limiter.in_flight,
            'attempts': self.attempts,
            'per_hour': dict(sorted(self.per_hour.items())),
            'recent': list(self.recent),
            'done': self.finished,
        }

    finished = False


# ---------------------------------------------------------------------------
# 按小时流式落盘 sink （单一 writer ；跑在 worker 线程里）
# ---------------------------------------------------------------------------


class Sink:
    def __init__(self, stream_root: Path) -> None:
        self.root = stream_root
        self.files: dict[str, Any] = {}
        self.counts: dict[str, int] = {}
        self.keys: dict[str, list[str]] = {}

    @staticmethod
    def _bucket(end_iso: str) -> tuple[str, str]:
        try:
            dt = datetime.fromisoformat(end_iso)
        except Exception:
            dt = datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        return dt.strftime('%Y-%m-%d'), dt.strftime('%H')

    def write(self, rec: dict) -> None:
        day, hour = self._bucket(rec.get('end_time') or now_iso())
        key = f'{day}/{hour}'
        # 记录以近实时方式到达，因此一旦出现新的小时，之前那些小时的
        # 分区就算完整了，可以封存。
        for k in sorted(self.files):
            if k < key:
                self._seal(k)
        f = self.files.get(key)
        if f is None:
            d = self.root / f'dt={day}' / f'hour={hour}'
            d.mkdir(parents=True, exist_ok=True)
            f = open(d / 'part.jsonl', 'a', encoding='utf-8')  # noqa: SIM115
            self.files[key] = f
            self.counts.setdefault(key, 0)
            self.keys.setdefault(key, [])
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
        self.counts[key] += 1
        self.keys[key].append(rec['run_key'])

    def _seal(self, key: str) -> None:
        f = self.files.pop(key, None)
        if f is not None:
            with contextlib.suppress(Exception):
                f.flush()
                os.fsync(f.fileno())
                f.close()
        day, hour = key.split('/')
        d = self.root / f'dt={day}' / f'hour={hour}'
        part = d / 'part.jsonl'
        sha = hashlib.sha256(part.read_bytes()).hexdigest() if part.exists() else ''
        atomic_write_json(
            d / 'manifest.json',
            {
                'hour': key,
                'count': self.counts.get(key, 0),
                'run_keys': self.keys.get(key, []),
                'sha256': sha,
                'sealed_at': now_iso(),
            },
        )
        (d / '_SUCCESS').write_text('', encoding='utf-8')

    def close_all(self) -> None:
        for k in sorted(self.files):
            self._seal(k)


# ---------------------------------------------------------------------------
# 工作项
# ---------------------------------------------------------------------------


class Item:
    __slots__ = ('case_id', 'model', 'rep', 'run_key', 'run_dir', 'timeout')

    def __init__(self, case_id: str, model: str, rep: int, out: Path, timeout: int) -> None:
        self.case_id = case_id
        self.model = model
        self.rep = rep
        self.run_key = f'{sanitize(case_id)}__{sanitize(model)}__rep{rep}'
        self.run_dir = out / 'runs' / self.run_key
        self.timeout = timeout


# ---------------------------------------------------------------------------
# 一次运行结束后的分类
# ---------------------------------------------------------------------------


def classify(result_path: Path, rc: int, timed_out: bool) -> tuple[str, str | None, Any, dict]:
    """返回 (outcome, failure_kind, score, extra) 。 outcome 属于
    {ok, quarantine, infra} ；调用方会把耗尽重试次数的 ``infra``
    转成 ``failed`` 。"""
    if timed_out:
        return 'infra', 'wall_timeout', None, {}
    if not result_path.exists():
        return 'infra', f'norun_rc{rc}', None, {}
    try:
        raw = json.loads(result_path.read_text(encoding='utf-8'))
    except Exception:
        return 'infra', 'parsefail', None, {}

    to = raw.get('task_output') or {}
    ao = raw.get('agent_output') or {}
    ef = to.get('extra_fields') or {}
    sv = ef.get('aidata.score_valid')
    kind = ef.get('aidata.failure_kind')
    phase = ef.get('aidata.failure_phase')
    score = to.get('score')
    success = bool(raw.get('success'))
    extra = {
        'continuous_score': to.get('continuous_score'),
        'dimension_scores': to.get('dimension_scores'),
        'rubric_version': to.get('rubric_version'),
        'wall_time_s': to.get('wall_time_s'),
        'iterations': ao.get('iterations'),
        'exit_status': ao.get('exit_status'),
        'score_valid': sv,
        'failure_phase': phase,
        'case_metadata': to.get('case_metadata') or {},
        'main_trajectory': ao.get('main_trajectory') or [],
    }

    if success and sv is not False:
        return 'ok', None, score, extra
    if kind in CASE_BUG_KINDS:
        return 'quarantine', kind, score, extra
    if kind in INFRA_KINDS:
        return 'infra', kind, score, extra
    if not success:
        err = raw.get('error') or {}
        return 'infra', (err.get('type') or 'run_failed'), score, extra
    return 'infra', kind or 'unknown_invalid', score, extra


def backoff_seconds(attempt: int) -> float:
    return min(300.0, 15.0 * (2 ** (attempt - 1))) + random.uniform(0, 10)


# ---------------------------------------------------------------------------
# 编排器
# ---------------------------------------------------------------------------


class Orchestrator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.out = Path(args.out).resolve()
        self.cases_root = Path(args.cases_root).resolve()
        self.models = [m.strip() for m in args.models.split(',') if m.strip()]
        self.reps = int(args.reps)
        self._concurrency = int(getattr(args, 'concurrency', 32))
        self.max_attempts = int(args.max_attempts)
        self.judge_retries = int(getattr(args, 'judge_retries', 5))
        self.timeout_default = int(args.timeout_default)
        self.timeout_cap = int(args.timeout_cap)
        self.redo = bool(args.redo)
        self.include_pending = bool(args.include_pending)
        self.limit_n = args.limit
        self.only = {s.strip() for s in (args.only or '').split(',') if s.strip()}

        for m in self.models:
            if m not in MODELS:
                sys.exit(f'error: unknown model {m!r}; known: {sorted(MODELS)}')

        self.base_env = os.environ.copy()
        for envf in (NEXUS_REPO / '.env', self.cases_root.parent / '.env'):
            for k, v in load_env_file(envf).items():
                self.base_env.setdefault(k, v)
        self.template_text = TEMPLATE.read_text(encoding='utf-8')
        self.template_text_anthropic = TEMPLATE_ANTHROPIC.read_text(encoding='utf-8')

        self.stats: Stats
        self.limiter: AdaptiveLimiter
        self.sink: Sink
        self.sink_q: asyncio.Queue
        self._render_cache: dict[str, Path] = {}
        self._stopping = False
        self.run_state: dict[str, dict] = {}
        self._deadletter_lock = threading.Lock()

    # -- 规划 -------------------------------------------------------------

    def discover(self) -> list[tuple[str, str]]:
        found: list[tuple[str, str]] = []
        for root, dirs, _files in os.walk(self.cases_root):
            dirs[:] = [d for d in dirs if not d.startswith(('.', '_'))]
            rp = Path(root)
            text = manifest_text(rp)
            if not text:
                continue
            # 一个 case 根目录不会有子 case 。 剪掉后代目录以（a ）避免
            # 遍历每个 case 打包好的 fixture —— work2_swe_* 系列 case 会
            # 携带整个源码仓库，其中 git-ignored 的 node_modules / build
            # 树 os.walk （不像 ripgrep ）会照样进去，正是这一点让冷缓存
            # discovery 花掉近一分钟；（b ）避免把 bundle 里嵌套的
            # task.yaml 误认成真的 case 。
            dirs[:] = []
            case_id = rp.relative_to(self.cases_root).as_posix()
            if self.only and case_id not in self.only:
                continue
            if not self.include_pending and is_pending_upstream(text):
                continue
            found.append((case_id, text))
        found.sort(key=lambda x: x[0])
        if self.limit_n:
            found = found[: int(self.limit_n)]
        return found

    def build_items(self) -> list[Item]:
        items: list[Item] = []
        for case_id, text in self.discover():
            at = case_agent_timeout(text)
            wall = int(at * 1.5) if at else self.timeout_default
            wall = max(600, min(self.timeout_cap, wall))
            for model in self.models:
                for rep in range(1, self.reps + 1):
                    items.append(Item(case_id, model, rep, self.out, wall))
        return items

    @staticmethod
    def _is_anthropic(model: str) -> bool:
        m = MODELS.get(model, {}).get('SEARCH_OPENAI_MODEL', '').lower()
        return 'claude' in m or 'anthropic' in m

    def render(self, case_id: str, model: str) -> Path:
        # Claude 模型渲染 vertex_anthropic passthrough 模板；其他渲染
        # litellm 模板。 按 (case, template kind) 做缓存。
        anthropic = self._is_anthropic(model)
        kind = 'anthropic' if anthropic else 'litellm'
        key = f'{case_id}\x00{kind}'
        p = self._render_cache.get(key)
        if p is None:
            p = self.out / '.render' / f'{sanitize(case_id)}.{kind}.yaml'
            p.parent.mkdir(parents=True, exist_ok=True)
            text = self.template_text_anthropic if anthropic else self.template_text
            p.write_text(text.replace('{{CASE_ID}}', case_id), encoding='utf-8')
            self._render_cache[key] = p
        return p

    # -- 单次运行 ---------------------------------------------------------

    def _terminal(self, item: Item) -> bool:
        sp = item.run_dir / 'status.json'
        if not sp.exists():
            return False
        try:
            return bool(json.loads(sp.read_text(encoding='utf-8')).get('terminal'))
        except Exception:
            return False

    def _set_state(self, run_key: str, **kw: Any) -> None:
        st = self.run_state.get(run_key)
        if st is not None:
            st.update(kw)
            st['updated'] = now_iso()

    async def _spawn(self, item: Item, yaml_path: Path, attempt: int) -> tuple[int, bool, float]:
        env = dict(self.base_env)
        env.update(MODELS[item.model])
        env['AIDATA_CASES_ROOT'] = str(self.cases_root)
        env['AIDATA_STATE_DUMP_DIR'] = str(item.run_dir / 'state')
        env['NEXUS_JUDGE_MAX_RETRIES'] = str(self.judge_retries)
        out = item.run_dir / 'result.json'
        with contextlib.suppress(FileNotFoundError):
            out.unlink()
        logf = open(item.run_dir / f'attempt{attempt}.log', 'wb')  # noqa: SIM115
        t0 = time.time()
        proc = None
        timed_out = False
        try:
            proc = await asyncio.create_subprocess_exec(
                UV,
                'run',
                'nexus',
                'run-single',
                str(yaml_path),
                '--output',
                str(out),
                cwd=str(NEXUS_REPO),
                env=env,
                stdout=logf,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                rc = await asyncio.wait_for(proc.wait(), timeout=item.timeout)
            except asyncio.TimeoutError:
                timed_out = True
                rc = await self._kill(proc)
        except asyncio.CancelledError:
            if proc is not None:
                await self._kill(proc)
            raise
        finally:
            logf.close()
        return rc, timed_out, time.time() - t0

    @staticmethod
    async def _kill(proc: asyncio.subprocess.Process) -> int:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=15)
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            with contextlib.suppress(Exception):
                await proc.wait()
        return proc.returncode if proc.returncode is not None else -9

    async def _finalize(
        self,
        item: Item,
        outcome: str,
        kind: str | None,
        score: Any,
        extra: dict,
        attempt: int,
        start_iso: str,
        wall: float | None,
    ) -> None:
        end_iso = now_iso()
        extra = extra or {}
        meta = extra.get('case_metadata') or {}
        rec = {
            'run_key': item.run_key,
            'case_id': item.case_id,
            'model': item.model,
            'rep': item.rep,
            'attempt': attempt,
            'outcome': outcome,
            'score': score,
            'continuous_score': extra.get('continuous_score'),
            'score_valid': extra.get('score_valid'),
            'failure_phase': extra.get('failure_phase'),
            'failure_kind': kind,
            'dimension_scores': extra.get('dimension_scores'),
            'rubric_version': extra.get('rubric_version'),
            'wall_time_s': extra.get('wall_time_s') or (round(wall, 1) if wall else None),
            'iterations': extra.get('iterations'),
            'exit_status': extra.get('exit_status'),
            'tags': meta.get('tags') or [],
            'start_time': start_iso,
            'end_time': end_iso,
            'trajectory_path': f'runs/{item.run_key}/rollout.jsonl' if extra.get('main_trajectory') else None,
        }
        await asyncio.to_thread(self._write_outputs, item, outcome, kind, score, extra, attempt, end_iso, rec)
        self.stats.finalize(outcome, rec)
        self._set_state(
            item.run_key, state=outcome, score=score, failure_kind=kind, attempt=attempt, wall_s=rec.get('wall_time_s')
        )
        self.sink_q.put_nowait(rec)

    def _write_outputs(
        self, item: Item, outcome: str, kind: str | None, score: Any, extra: dict, attempt: int, end_iso: str, rec: dict
    ) -> None:
        traj = extra.get('main_trajectory') or []
        if traj:
            with open(item.run_dir / 'rollout.jsonl', 'w', encoding='utf-8') as f:
                for m in traj:
                    f.write(json.dumps(m, ensure_ascii=False) + '\n')
        atomic_write_json(
            item.run_dir / 'status.json',
            {
                'run_key': item.run_key,
                'outcome': outcome,
                'terminal': True,
                'attempt': attempt,
                'score': score,
                'score_valid': extra.get('score_valid'),
                'failure_kind': kind,
                'end_time': end_iso,
            },
        )
        if outcome in ('quarantine', 'failed'):
            # _write_outputs 跑在线程池里；用锁避免并发 append 交错
            # 破坏一行完整的 JSON 记录。
            with self._deadletter_lock, open(self.out / 'deadletter.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    async def run_one(self, item: Item) -> None:
        try:
            if not self.redo and self._terminal(item):
                self.stats.resumed += 1
                with contextlib.suppress(Exception):
                    sd = json.loads((item.run_dir / 'status.json').read_text(encoding='utf-8'))
                    self._set_state(
                        item.run_key,
                        state=sd.get('outcome', 'ok'),
                        score=sd.get('score'),
                        failure_kind=sd.get('failure_kind'),
                        attempt=sd.get('attempt', 0),
                    )
                return
            item.run_dir.mkdir(parents=True, exist_ok=True)
            if not (self.cases_root / item.case_id / 'rubric.yaml').exists():
                await self._finalize(item, 'quarantine', 'missing_rubric', None, {}, 0, now_iso(), 0.0)
                return
            yaml_path = self.render(item.case_id, item.model)
            start_iso = now_iso()
            attempt = 0
            while True:
                if self._stopping:
                    return
                attempt += 1
                await self.limiter.acquire()
                self.stats.running += 1
                self._set_state(item.run_key, state='running', attempt=attempt)
                try:
                    rc, timed_out, wall = await self._spawn(item, yaml_path, attempt)
                finally:
                    self.stats.running -= 1
                    await self.limiter.release()
                outcome, kind, score, extra = classify(item.run_dir / 'result.json', rc, timed_out)
                self.stats.record_attempt(outcome == 'infra')
                if outcome in ('ok', 'quarantine'):
                    await self._finalize(item, outcome, kind, score, extra, attempt, start_iso, wall)
                    return
                if attempt >= self.max_attempts:
                    await self._finalize(item, 'failed', kind, score, extra, attempt, start_iso, wall)
                    return
                self.stats.retrying += 1
                self._set_state(item.run_key, state='retrying', attempt=attempt)
                try:
                    await asyncio.sleep(backoff_seconds(attempt))
                finally:
                    self.stats.retrying -= 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 —— 单个坏 case 不能拖垮整个 batch
            log(f'run_one crashed for {item.run_key}: {type(exc).__name__}: {exc}')
            with contextlib.suppress(Exception):
                await self._finalize(
                    item, 'failed', 'orchestrator_error', None, {'failure_phase': 'orchestrator'}, 0, now_iso(), 0.0
                )

    # -- 后台任务 ---------------------------------------------------------

    async def _sink_writer(self) -> None:
        while True:
            rec = await self.sink_q.get()
            if rec is None:
                break
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self.sink.write, rec)
        await asyncio.to_thread(self.sink.close_all)

    async def _progress_writer(self) -> None:
        while not self.stats.finished:
            with contextlib.suppress(Exception):
                atomic_write_json(self.out / 'progress.json', self.stats.snapshot(self))
                atomic_write_json(self.out / 'runs.json', list(self.run_state.values()))
            await asyncio.sleep(5)
        with contextlib.suppress(Exception):
            atomic_write_json(self.out / 'progress.json', self.stats.snapshot(self))
            atomic_write_json(self.out / 'runs.json', list(self.run_state.values()))

    async def _aimd(self) -> None:
        while not self.stats.finished:
            await asyncio.sleep(30)
            rate, n = self.stats.infra_rate(30.0)
            if n < 6:
                continue
            if rate > 0.30:
                await self.limiter.set_limit(int(self.limiter.limit * 0.7))
                log(f'AIMD: infra_rate={rate:.0%} (n={n}) -> shrink to {self.limiter.limit}')
            elif rate < 0.05:
                await self.limiter.set_limit(self.limiter.limit + 2)

    # -- 主流程 -----------------------------------------------------------

    async def main(self, items: list[Item]) -> None:
        start_conc = self.limiter.hi
        self.sink_q = asyncio.Queue()
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def _request_stop() -> None:
            self._stopping = True
            stop_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, _request_stop)

        sink_task = asyncio.create_task(self._sink_writer())
        prog_task = asyncio.create_task(self._progress_writer())
        aimd_task = asyncio.create_task(self._aimd())

        log(f'start: {len(items)} runs | models={self.models} reps={self.reps} concurrency={start_conc} out={self.out}')
        workers = [asyncio.create_task(self.run_one(it)) for it in items]
        gather = asyncio.gather(*workers, return_exceptions=True)
        stopper = asyncio.create_task(stop_event.wait())
        done, _ = await asyncio.wait({gather, stopper}, return_when=asyncio.FIRST_COMPLETED)

        if stopper in done and not gather.done():
            log('stop requested: cancelling in-flight runs (sealing partitions)...')
            for w in workers:
                w.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await gather
        else:
            stopper.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await stopper

        self.stats.finished = True
        await self.sink_q.put(None)
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await sink_task
        for tk in (prog_task, aimd_task):
            tk.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await tk
        with contextlib.suppress(Exception):
            atomic_write_json(self.out / 'progress.json', self.stats.snapshot(self))
            atomic_write_json(self.out / 'runs.json', list(self.run_state.values()))
        s = self.stats
        log(f'done: ok={s.ok} quarantine={s.quarantine} failed={s.failed} resumed={s.resumed} / total={s.total}')

    def prepare(self, items: list[Item]) -> None:
        (self.out / 'runs').mkdir(parents=True, exist_ok=True)
        (self.out / 'stream').mkdir(parents=True, exist_ok=True)
        (self.out / '.render').mkdir(parents=True, exist_ok=True)
        self.stats = Stats(total=len(items))
        self.run_state = {
            it.run_key: {
                'run_key': it.run_key,
                'case_id': it.case_id,
                'model': it.model,
                'rep': it.rep,
                'state': 'pending',
                'score': None,
                'attempt': 0,
                'failure_kind': None,
                'wall_s': None,
                'updated': now_iso(),
            }
            for it in items
        }
        conc = int(getattr(self, '_concurrency', 32))
        self.limiter = AdaptiveLimiter(start=conc, lo=max(4, conc // 8), hi=conc)
        self.sink = Sink(self.out / 'stream')
        atomic_write_json(
            self.out / 'orchestrator.meta.json',
            {
                'started_at': now_iso(),
                'pid': os.getpid(),
                'out': str(self.out),
                'cases_root': str(self.cases_root),
                'models': self.models,
                'reps': self.reps,
                'total': len(items),
                'concurrency': conc,
                'max_attempts': self.max_attempts,
            },
        )


# ---------------------------------------------------------------------------
# 守护进程化
# ---------------------------------------------------------------------------


def daemonize(log_path: Path, pid_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if os.fork() > 0:
        os._exit(0)
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    logfd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(logfd, 1)
    os.dup2(logfd, 2)
    pid_path.write_text(str(os.getpid()), encoding='utf-8')


# ---------------------------------------------------------------------------
# 子命令： run
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    ctx = Orchestrator(args)
    items = ctx.build_items()
    n_cases = len({it.case_id for it in items})
    if not items:
        sys.exit('error: no cases matched (check --cases-root / --only / --limit)')

    if args.dry_run:
        print(f'cases={n_cases}  models={ctx.models}  reps={ctx.reps}  -> {len(items)} runs')
        print(f'cases_root={ctx.cases_root}')
        print(f'out={ctx.out}')
        print(
            f'concurrency={args.concurrency}  max_attempts={ctx.max_attempts}  '
            f'timeout_default={ctx.timeout_default}  cap={ctx.timeout_cap}'
        )
        already = sum(1 for it in items if not ctx.redo and ctx._terminal(it))
        print(f'resume: {already}/{len(items)} already terminal -> would run {len(items) - already}')
        for it in items[:5]:
            print(f'  sample run_key={it.run_key}  timeout={it.timeout}s')
        return 0

    ctx.out.mkdir(parents=True, exist_ok=True)
    pid_path = ctx.out / 'orchestrator.pid'
    if pid_path.exists():
        with contextlib.suppress(Exception):
            old = int(pid_path.read_text())
            os.kill(old, 0)
            sys.exit(f'error: a run seems active (pid {old}); use `stop` first or a different --out')

    if args.daemon:
        daemonize(ctx.out / 'orchestrator.log', pid_path)
    else:
        pid_path.write_text(str(os.getpid()), encoding='utf-8')

    ctx.prepare(items)
    try:
        asyncio.run(ctx.main(items))
    finally:
        with contextlib.suppress(FileNotFoundError):
            pid_path.unlink()
    return 0


# ---------------------------------------------------------------------------
# 子命令： stop
# ---------------------------------------------------------------------------


def cmd_stop(args: argparse.Namespace) -> int:
    pid_path = Path(args.out).resolve() / 'orchestrator.pid'
    if not pid_path.exists():
        print('no orchestrator.pid found; nothing to stop.')
        return 0
    pid = int(pid_path.read_text())
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f'pid {pid} not running; clearing stale pidfile.')
        with contextlib.suppress(FileNotFoundError):
            pid_path.unlink()
        return 0
    print(f'sent SIGTERM to {pid}; waiting for graceful seal...')
    for _ in range(60):
        time.sleep(1)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            print('stopped cleanly.')
            return 0
    print('still running after 60s; check orchestrator.log.')
    return 1


# ---------------------------------------------------------------------------
# 子命令： status / monitor
# ---------------------------------------------------------------------------


def _read_progress(out: Path) -> dict | None:
    p = out / 'progress.json'
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None


def _split_run_key(rk: str) -> tuple[str, str, int]:
    m = re.match(r'^(.*)__(.*)__rep(\d+)$', rk)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return rk, '', 0


def _scan_status(out: Path) -> list[dict]:
    """从 runs/<key>/status.json 回退拿 per-run 详情（仅 terminal ）。

    当编排器还没写出 runs.json 时（比如是老版本启动的一次运行），
    dashboard 就用这个来兜底。 只能看到已完成的 run ；
    running / pending 只能通过当前编排器写出的实时 runs.json 查看。
    """
    rows: list[dict] = []
    rdir = out / 'runs'
    if not rdir.is_dir():
        return rows
    for sp in rdir.glob('*/status.json'):
        try:
            sd = json.loads(sp.read_text(encoding='utf-8'))
        except Exception:
            continue
        rk = sd.get('run_key') or sp.parent.name
        case, model, rep = _split_run_key(rk)
        rows.append({
            'run_key': rk,
            'case_id': case,
            'model': model,
            'rep': rep,
            'state': sd.get('outcome'),
            'score': sd.get('score'),
            'attempt': sd.get('attempt'),
            'failure_kind': sd.get('failure_kind'),
            'wall_s': None,
            'updated': sd.get('end_time'),
        })
    return rows


def _fmt_eta(sec: int | None) -> str:
    if not sec or sec < 0:
        return '—'
    h, r = divmod(int(sec), 3600)
    m, _ = divmod(r, 60)
    return f'{h}h{m:02d}m' if h else f'{m}m'


def _progress_lines(snap: dict) -> list[str]:
    total = snap.get('total', 0) or 1
    term = snap.get('terminal', 0)
    bar_w = 40
    filled = int(bar_w * (term + snap.get('resumed', 0)) / total)
    bar = '█' * filled + '·' * (bar_w - filled)
    lines = [
        f'aidata run · {snap.get("out", "")}',
        f'models={snap.get("models")} reps={snap.get("reps")} pid={snap.get("pid")} '
        f'updated={snap.get("updated_at", "")[11:19]}',
        f'[{bar}] {term + snap.get("resumed", 0)}/{total}',
        (
            f'ok={snap.get("ok", 0)}  quarantine={snap.get("quarantine", 0)}  failed={snap.get("failed", 0)}  '
            f'resumed={snap.get("resumed", 0)}  running={snap.get("running", 0)}  retrying={snap.get("retrying", 0)}  '
            f'pending={snap.get("pending", 0)}'
        ),
        (
            f'valid_rate={snap.get("valid_rate", 0):.1%}  throughput={snap.get("throughput_per_min", 0)}/min  '
            f'eta={_fmt_eta(snap.get("eta_seconds"))}  conc={snap.get("limit")}/{snap.get("concurrency_max")} '
            f'in_flight={snap.get("in_flight")}'
        ),
    ]
    ph = snap.get('per_hour') or {}
    if ph:
        tail = list(ph.items())[-6:]
        lines.append('by hour: ' + '  '.join(f'{k[11:]}={v}' for k, v in tail))
    recent = snap.get('recent') or []
    if recent:
        lines.append('recent:')
        for r in recent[:8]:
            sc = r.get('score')
            sc = f'{sc:.2f}' if isinstance(sc, (int, float)) else '—'
            lines.append(f'  [{r.get("outcome"):<10}] {r.get("run_key", "")[:60]:<60} score={sc}')
    return lines


def cmd_status(args: argparse.Namespace) -> int:
    snap = _read_progress(Path(args.out).resolve())
    if snap is None:
        print('no progress.json yet.')
        return 0
    print('\n'.join(_progress_lines(snap)))
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    interval = float(args.interval)
    try:
        from rich.live import Live
        from rich.text import Text

        with Live(refresh_per_second=4, screen=False) as live:
            while True:
                snap = _read_progress(out)
                body = '\n'.join(_progress_lines(snap)) if snap else 'waiting for progress.json ...'
                live.update(Text(body))
                time.sleep(interval)
    except ImportError:
        try:
            while True:
                snap = _read_progress(out)
                os.system('clear')  # noqa: S605,S607
                print('\n'.join(_progress_lines(snap)) if snap else 'waiting for progress.json ...')
                time.sleep(interval)
        except KeyboardInterrupt:
            return 0
    except KeyboardInterrupt:
        return 0


# ---------------------------------------------------------------------------
# 子命令： dashboard （本地站点）
# ---------------------------------------------------------------------------


DASH_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>aidata run</title><style>
:root{color-scheme:dark}
body{background:#141414;color:#e4e4e4;font:13px ui-monospace,Menlo,monospace;margin:0;padding:22px}
h1{font-size:16px;margin:0 0 4px} h3{margin:18px 0 6px} .sub{color:#8a8a8a;margin-bottom:14px}
.bar{height:22px;border-radius:6px;overflow:hidden;display:flex;background:#222;margin:10px 0}
.bar>div{height:100%}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0}
.tile{border:1px solid #2a2a2a;border-radius:8px;padding:10px 12px;background:#181818}
.tile .v{font-size:19px} .tile .l{color:#8a8a8a;font-size:11px;margin-top:2px}
.hours{display:flex;gap:6px;align-items:flex-end;height:110px;margin:8px 0}
.hours .col{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end}
.hours .b{width:100%;background:#3f76c0;border-radius:3px 3px 0 0}
.hours .t{color:#8a8a8a;font-size:10px;margin-top:3px}
table{border-collapse:collapse;width:100%}
td,th{border-bottom:1px solid #232323;padding:4px 8px;text-align:left;white-space:nowrap}
th{color:#8a8a8a;font-weight:normal;cursor:pointer;position:sticky;top:0;background:#141414}
.k{color:#8a8a8a}
.dot{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;vertical-align:middle}
.legend{margin:4px 0;color:#8a8a8a}
.ctl{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:6px 0}
.ctl input,.ctl select{background:#181818;color:#e4e4e4;border:1px solid #2a2a2a;border-radius:6px;padding:5px 8px;font:inherit}
.ctl input[type=text]{min-width:240px}
.pill{padding:1px 7px;border-radius:10px;font-size:11px}
.wrap{max-height:480px;overflow:auto;border:1px solid #232323;border-radius:8px}
.count{color:#8a8a8a}
</style></head><body>
<h1>AIDATA run</h1><div class="sub" id="sub">loading…</div>
<div class="bar" id="bar"></div>
<div class="legend">
<span class="dot" style="background:#3fa266"></span>ok
<span class="dot" style="background:#7bafe9"></span>running
<span class="dot" style="background:#f1b467"></span>retrying
<span class="dot" style="background:#d08770"></span>quarantine
<span class="dot" style="background:#d06b6b"></span>failed
<span class="dot" style="background:#666"></span>pending</div>
<div class="tiles" id="tiles"></div>
<h3>按小时完成量</h3><div class="hours" id="hours"></div>
<h3>明细 · task × model × run</h3>
<div class="ctl">
 <input type="text" id="q" placeholder="筛选 case_id / run_key…">
 <select id="fstate"><option value="">state: all</option></select>
 <select id="fmodel"><option value="">model: all</option></select>
 <span class="count" id="cnt"></span>
</div>
<div class="wrap"><table id="detail">
<thead><tr><th data-k="case_id">case_id</th><th data-k="model">model</th><th data-k="rep">rep</th>
<th data-k="state">state</th><th data-k="score">score</th><th data-k="attempt">attempt</th>
<th data-k="failure_kind">kind</th><th data-k="updated">updated</th></tr></thead>
<tbody></tbody></table></div>
<script>
const C={ok:'#3fa266',running:'#7bafe9',retrying:'#f1b467',quarantine:'#d08770',failed:'#d06b6b',pending:'#666'};
let RUNS=[],SORT={k:'updated',dir:-1};
function eta(s){if(!s||s<0)return '—';let h=Math.floor(s/3600),m=Math.floor((s%3600)/60);return h?h+'h'+String(m).padStart(2,'0')+'m':m+'m';}
function seg(bar,v,total,c){if(v<=0)return;const d=document.createElement('div');d.style.width=(100*v/total)+'%';d.style.background=c;bar.appendChild(d);}
function esc(x){return (x==null?'':(''+x)).replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]));}
function _p(n){return String(n).padStart(2,'0');}
function _ud(x){if(!x)return null;let s=(''+x).trim().replace(' ','T');if(!(s.endsWith('Z')||s.slice(-6,-5)==='+'||s.slice(-6,-5)==='-'))s+='Z';const d=new Date(s);return isNaN(d.getTime())?null:d;}
function fmtHMS(x){const d=_ud(x);return d?_p(d.getHours())+':'+_p(d.getMinutes())+':'+_p(d.getSeconds()):'';}
function fmtMDHMS(x){const d=_ud(x);return d?_p(d.getMonth()+1)+'-'+_p(d.getDate())+' '+_p(d.getHours())+':'+_p(d.getMinutes())+':'+_p(d.getSeconds()):'';}
function fmtHourLabel(x){const d=_ud(x);return d?_p(d.getHours())+':00':'';}
function opts(sel,vals){const cur=sel.value;const keep=sel.options[0].outerHTML;
 sel.innerHTML=keep+vals.map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join('');sel.value=cur;}
function renderHeader(s,a){
 document.getElementById('sub').textContent=`out=${s.out} · models=${JSON.stringify(s.models)} reps=${s.reps} · pid=${s.pid} · updated ${fmtHMS(s.updated_at)} 本地${s.done?' · DONE':''}`;
 const bar=document.getElementById('bar');bar.innerHTML='';
 seg(bar,a.ok,a.total,C.ok);seg(bar,a.running,a.total,C.running);seg(bar,a.retrying,a.total,C.retrying);
 seg(bar,a.quarantine,a.total,C.quarantine);seg(bar,a.failed,a.total,C.failed);seg(bar,a.pending,a.total,C.pending);
 const tiles=[['已终态',`${a.done}/${a.total}`],['有效率',(100*a.valid_rate).toFixed(1)+'%'],
  ['quarantine',a.quarantine],['运行中',a.running],['吞吐/min',s.throughput_per_min],
  ['ETA',eta(s.eta_seconds)],['并发',`${s.limit}/${s.concurrency_max}`],['pending',a.pending]];
 document.getElementById('tiles').innerHTML=tiles.map(t=>`<div class="tile"><div class="v">${t[1]}</div><div class="l">${t[0]}</div></div>`).join('');
 const ph=Object.entries(s.per_hour||{});const mx=Math.max(1,...ph.map(x=>x[1]));
 document.getElementById('hours').innerHTML=ph.slice(-12).map(([k,v])=>
   `<div class="col"><div class="b" style="height:${100*v/mx}%"></div><div class="t">${fmtHourLabel(k)}·${v}</div></div>`).join('');
}
function renderTable(){
 const q=document.getElementById('q').value.toLowerCase();
 const fs=document.getElementById('fstate').value, fm=document.getElementById('fmodel').value;
 let rows=RUNS.filter(r=>(!fs||r.state===fs)&&(!fm||r.model===fm)&&
  (!q||(r.case_id||'').toLowerCase().includes(q)||(r.run_key||'').toLowerCase().includes(q)));
 const k=SORT.k,dir=SORT.dir;
 rows.sort((a,b)=>{let x=a[k],y=b[k];if(x==null)x='';if(y==null)y='';return x>y?dir:x<y?-dir:0;});
 document.getElementById('cnt').textContent=`showing ${Math.min(rows.length,800)} / ${RUNS.length}`;
 document.querySelector('#detail tbody').innerHTML=rows.slice(0,800).map(r=>{
  const c=C[r.state]||'#888';const sc=(r.score==null)?'—':(+r.score).toFixed(2);
  return `<tr><td>${esc(r.case_id)}</td><td>${esc(r.model)}</td><td>${esc(r.rep)}</td>`+
   `<td><span class="pill" style="background:${c}22;color:${c}">${esc(r.state)}</span></td>`+
   `<td>${sc}</td><td>${esc(r.attempt)}</td><td class="k">${esc(r.failure_kind)}</td>`+
   `<td class="k">${fmtMDHMS(r.updated)}</td></tr>`;
 }).join('');
}
function aggregate(runs,s){
 const c={ok:0,quarantine:0,failed:0,running:0,retrying:0,pending:0};
 for(const r of runs){if(c[r.state]!==undefined)c[r.state]++;}
 const total=(s&&s.total)||runs.length||1;
 const done=c.ok+c.quarantine+c.failed;
 return {total,done,ok:c.ok,quarantine:c.quarantine,failed:c.failed,running:c.running,retrying:c.retrying,pending:c.pending,valid_rate:done?c.ok/done:0};
}
async function tick(){
 let s={};
 try{s=await (await fetch('progress.json',{cache:'no-store'})).json();}catch(e){}
 try{RUNS=await (await fetch('runs.json',{cache:'no-store'})).json();}catch(e){RUNS=[];}
 const a=aggregate(RUNS,s);
 if(s&&s.total)renderHeader(s,a);
 opts(document.getElementById('fstate'),[...new Set(RUNS.map(r=>r.state).filter(Boolean))].sort());
 opts(document.getElementById('fmodel'),[...new Set(RUNS.map(r=>r.model).filter(Boolean))].sort());
 renderTable();
}
['q','fstate','fmodel'].forEach(id=>document.getElementById(id).addEventListener('input',renderTable));
document.querySelectorAll('#detail th').forEach(th=>th.addEventListener('click',()=>{
 const k=th.dataset.k;SORT.dir=(SORT.k===k)?-SORT.dir:1;SORT.k=k;renderTable();}));
tick();setInterval(tick,5000);
</script></body></html>"""


def cmd_dashboard(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:  # 静默
            pass

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ('/', '/index.html'):
                body = DASH_HTML.encode('utf-8')
                ctype = 'text/html; charset=utf-8'
            elif self.path.split('?')[0] == '/progress.json':
                p = out / 'progress.json'
                body = p.read_bytes() if p.exists() else b'{}'
                ctype = 'application/json'
            elif self.path.split('?')[0] == '/runs.json':
                rj = out / 'runs.json'
                body = rj.read_bytes() if rj.exists() else json.dumps(_scan_status(out)).encode('utf-8')
                ctype = 'application/json'
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer((args.host, int(args.dashboard_port)), Handler)
    print(f'dashboard at http://{args.host}:{args.dashboard_port}  (out={out})  Ctrl-C to stop')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='aidata_orchestrator', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)

    r = sub.add_parser('run', help='start or resume a campaign')
    r.add_argument('--out', required=True)
    r.add_argument('--cases-root', default='/data/workspace/Tasks')
    r.add_argument('--models', default='opus-4-8')
    r.add_argument('--reps', type=int, default=1)
    r.add_argument('--concurrency', type=int, default=32)
    r.add_argument('--max-attempts', type=int, default=5, help='max attempts per run (1 initial + N-1 retries)')
    r.add_argument(
        '--judge-retries', type=int, default=5, help='judge-LLM retries on UNAVAILABLE (NEXUS_JUDGE_MAX_RETRIES)'
    )
    r.add_argument('--timeout-default', type=int, default=3600)
    r.add_argument('--timeout-cap', type=int, default=7200)
    r.add_argument('--daemon', action='store_true', help='detach (survives terminal close)')
    r.add_argument('--redo', action='store_true', help='ignore existing terminal status')
    r.add_argument('--include-pending', action='store_true', help='do not skip pending_upstream cases')
    r.add_argument('--only', default='', help='comma-separated case_id allowlist')
    r.add_argument('--limit', type=int, default=None, help='first N cases (smoke test)')
    r.add_argument('--dry-run', action='store_true', help='print plan and exit')
    r.set_defaults(func=cmd_run)

    for name, fn, helptxt in (
        ('monitor', cmd_monitor, 'terminal TUI reader'),
        ('dashboard', cmd_dashboard, 'local site reader'),
        ('status', cmd_status, 'one-shot progress print'),
        ('stop', cmd_stop, 'graceful stop of a --daemon run'),
    ):
        s = sub.add_parser(name, help=helptxt)
        s.add_argument('--out', required=True)
        if name == 'monitor':
            s.add_argument('--interval', type=float, default=2.0)
        if name == 'dashboard':
            s.add_argument('--dashboard-port', type=int, default=8899)
            s.add_argument('--host', default='127.0.0.1')
        s.set_defaults(func=fn)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == '__main__':
    sys.exit(main())
