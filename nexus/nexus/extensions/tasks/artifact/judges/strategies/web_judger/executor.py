"""Generate evidence-collection scripts from flow contracts and execute them in the runtime."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
import uuid

from pathlib import Path
from typing import Any, Literal, TypedDict

from nexus.extensions.tasks.artifact.agent_setup.claude_config import JUDGE_ROOT


logger = logging.getLogger(__name__)


class Evidence(TypedDict, total=False):
    """Evidence collected for one flow, including observations, screenshots, and errors."""

    # Fields supplied to the judge.
    code: str  # 实际执行的脚本
    note: str  # flow2code 对选择器或检查方式的说明
    observations: list[dict[str, Any]]  # 原始事实列表，格式为 [{label, value}]
    shots: list[dict[str, Any]]  # 截图列表，格式为 [{name, label, b64, path}]
    page_errors: list[str]  # 页面未捕获异常
    console_errors: list[str]  # console 错误
    failed_requests: list[str]  # 加载失败的网络请求
    error: str  # 页面不可达或采集中断时的错误信息

    # Internal collection diagnostics.
    fid: str  # 对应的 flow ID
    kind: Literal['browser', 'source_code']  # 脚本类型
    status: str  # 采集状态，见 OK_STATUSES
    suppressed: list[str]  # 被过滤的判定式 label
    return_code: int | None  # source_code 脚本返回码
    duration_ms: int  # 执行耗时
    shot_calls: int  # shot helper 调用次数
    shot_capture_failures: int  # 容器内截图失败次数
    shot_errors: list[str]  # 容器内截图失败的原因


VERDICT_LABEL = (
    r'(^|_)(has|have|is|are|was|were|should|can|does|did|passed|failed|valid|invalid|correct|'
    r'works|working|ok|success|succeeded|matches|satisfies|verified|conform)($|_)'
)
# Filter verdict-like labels; collection records facts, not judgments.

CHROMIUM_ARGS = [
    '--headless=new',  # 使用 Chromium 原生无头渲染，不依赖 X server。
    '--no-sandbox',
    '--disable-setuid-sandbox',
    '--disable-gpu-sandbox',
    '--disable-dev-shm-usage',
    '--password-store=basic',
    '--use-mock-keychain',
    # Use ANGLE/SwiftShader for WebGL and Vulkan/SwiftShader for WebGPU.
    '--enable-unsafe-swiftshader',
    '--use-angle=swiftshader',
    '--use-vulkan=swiftshader',
    '--enable-features=Vulkan',
    '--enable-unsafe-webgpu',
    '--mute-audio',
    '--hide-scrollbars',
    '--disable-extensions',
    '--disable-background-networking',
]
# Keep native headless flags separate from the X11 screenshot path.

SCRIPTS_FILE = f'{JUDGE_ROOT}/web_judger/scripts.json'
# Read generated scripts from a dedicated file rather than mixed stdout.

FLOW2CODE_TIMEOUT = 4800.0
FLOW2CODE_MAX_TURNS = 40
FLOW2CODE_ATTEMPTS = 2
# Each attempt covers all flows and source inspection; a zero-script attempt is retried once.

ACTION_TIMEOUT_MS = 10000
# Bound Playwright actions so missing elements cannot stall a flow.

SHOT_TIMEOUT_MS = 30000
# Allow more time for screenshot rendering than for element actions.

SETTLE_SHOT_MS = 400
# Wait for entry transitions before capturing the page.

MAX_LIST_FILES = 150


def _flow2code_prompt() -> str:
    """Reload the flow2code prompt on each call."""
    from nexus.extensions.tasks.artifact.judges.capabilities.prompts import load_prompt

    return load_prompt('webjudger_flow2code.md')


# ── 读契约 ────────────────────────────────────────────────────────────────────


def format_file_list(files: list[str]) -> str:
    """Format source paths with stack entrypoints first."""
    if not files:
        return '（无文件列表）'

    def rank(path: str) -> tuple[int, str]:
        lowered = path.lower()
        if lowered.endswith('package.json'):
            return (0, path)
        if lowered.endswith(('index.html', 'index.htm')):
            return (1, path)
        if lowered.endswith(('.html', '.htm')):
            return (2, path)
        return (3, path) if '/' not in path else (4, path)

    ordered = sorted(files, key=rank)
    shown = ordered[:MAX_LIST_FILES]
    head = f'共 {len(files)} 个文件'
    if len(files) > len(shown):
        head += f'（列出前 {len(shown)} 个）'
    return '\n'.join([head, *(f'  {p}' for p in shown)])


# ── flow → code ───────────────────────────────────────────────────────────────


def _parse_scripts(raw: str, flow: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse scripts and return missing flow IDs; execute supplies missing-evidence records."""
    from nexus.extensions.tasks.artifact.judges.capabilities.parse import loads_json_lenient

    data, error = loads_json_lenient(raw)
    if error or not isinstance(data, dict):
        return [], [f['fid'] for f in flow.get('flows') or []]

    wanted = {f['fid']: f for f in flow.get('flows') or []}
    scripts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in data.get('scripts') or []:
        if not isinstance(item, dict):
            continue
        fid = str(item.get('fid') or '')
        code = str(item.get('code') or '')
        if fid not in wanted or fid in seen or not code.strip():
            continue
        seen.add(fid)
        kind = item.get('kind')
        if kind not in {'browser', 'source_code'}:
            # Use the contract check mode when a script omits its kind.
            kind = 'source_code' if wanted[fid].get('check_via') == 'source_code' else 'browser'
        scripts.append({'fid': fid, 'kind': kind, 'note': str(item.get('note') or ''), 'code': code})
    return scripts, sorted(set(wanted) - seen)


async def flow2code(
    flow: dict[str, Any],
    *,
    runtime: Any,
    runtime_name: str,
    archive_b64: str,
    problem_statement: str,
    file_list: list[str],
    openai_client: Any,
    source_dir: str,
    timeout: float = FLOW2CODE_TIMEOUT,
    max_turns: int = FLOW2CODE_MAX_TURNS,
    save_dir: str | None = None,
    model_config: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate collection scripts from the contract and source in the supplied runtime."""
    from nexus.extensions.tasks.artifact.agent_setup.claude_config import default_judge_claude_input
    from nexus.extensions.tasks.artifact.agent_setup.claude_hooks import WEBJUDGER_COLLECT_SKILL_DIR
    from nexus.extensions.tasks.artifact.judges.capabilities.agent import run_agent_and_read_reward

    # Read source in place within the evaluation runtime.
    uploaded, upload_error = len(file_list), None
    if uploaded == 0:
        return [], {
            'agent_error': '',
            'read_error': '',
            'execution_time': 0.0,
            'source_files': uploaded,
            'upload_error': upload_error or 'Archive produced no code files',
            'n_flows': len(flow.get('flows') or []),
            'n_scripts': 0,
            'missing_fids': [str(item.get('fid') or '') for item in flow.get('flows') or []],
            'kinds': {'browser': 0, 'source_code': 0},
        }
    # The harness owns navigation; script generation does not receive a page URL.
    from nexus.extensions.tasks.artifact.judges.capabilities.prompts import render_prompt

    prompt = render_prompt(
        _flow2code_prompt(),
        user_requirement=problem_statement,
        task_type=str(flow.get('task_type') or 'web'),
        source_dir=source_dir,
        n_files=len(file_list),
        flows_json=json.dumps(flow.get('flows') or [], ensure_ascii=False, indent=2),
        file_list=format_file_list(file_list),
        scripts_file=SCRIPTS_FILE,
    )
    claude_input = default_judge_claude_input(
        timeout=timeout,
        max_turns=max_turns,
        with_playwright=False,
        # Install the collection API skill without browser MCP tools.
        extra_skill_dirs=[WEBJUDGER_COLLECT_SKILL_DIR],
    )
    if model_config is not None:
        claude_input = model_config.apply_agent(claude_input)
    # Retry script generation according to FLOW2CODE_ATTEMPTS.
    for attempt in range(FLOW2CODE_ATTEMPTS):
        run = await run_agent_and_read_reward(
            image='',
            task_prompt=prompt,
            openai_client=openai_client,
            claude_input=claude_input,
            reward_file=SCRIPTS_FILE,
            prepared_runtime=runtime,
            prepared_runtime_name=runtime_name,
        )
        scripts, missing = _parse_scripts(run.raw, flow)
        if scripts:
            break
        if attempt + 1 < FLOW2CODE_ATTEMPTS:
            logger.warning('flow2code 零脚本（%.0fs, turns 见 raw），重试第 %d 次', run.cc.execution_time, attempt + 1)
    metadata: dict[str, Any] = {
        'agent_error': run.agent_error,
        'read_error': run.read_error,
        'execution_time': run.cc.execution_time,
        'source_files': uploaded,
        'upload_error': upload_error,
        'n_flows': len(flow.get('flows') or []),
        'n_scripts': len(scripts),
        'n_attempts': attempt + 1,
        'missing_fids': missing,
        'kinds': {k: sum(1 for s in scripts if s['kind'] == k) for k in ('browser', 'source_code')},
    }
    if missing:
        logger.warning('flow2code 漏了 %d 条 flow: %s', len(missing), missing)

    if save_dir:
        root = Path(save_dir) / 'web_judger'
        root.mkdir(parents=True, exist_ok=True)
        (root / 'scripts.json').write_text(json.dumps(scripts, ensure_ascii=False, indent=2), encoding='utf-8')
        (root / 'flow2code_raw.txt').write_text(run.cc.output or run.raw or '', encoding='utf-8')
        (root / 'flow2code_trajectory.json').write_text(
            json.dumps(run.cc.trajectories, ensure_ascii=False, indent=2), encoding='utf-8'
        )
    return scripts, metadata


# ── 容器内 harness ────────────────────────────────────────────────────────────

_BROWSER_HARNESS = '''\
import asyncio, json, os, re, time, traceback

URL = {url!r}
OUT = {out!r}
SHOTS = {shots!r}
SETTLE_MS = {settle_ms}

os.makedirs(SHOTS, exist_ok=True)
observations, shot_meta, console_errors, page_errors, failed_requests = [], [], [], [], []
suppressed = []
shot_attempts, shot_capture_failures = [], []


def dump(status='running', error=''):
    """增量写入当前证据。

    每次记录 observation 后立即写盘，使脚本被超时终止时仍能保留中断前的证据。
    """
    try:
        json.dump({{
            'status': status, 'error': error, 'observations': observations, 'shots': shot_meta,
            'console_errors': console_errors[:20], 'page_errors': page_errors[:20],
            'failed_requests': failed_requests[:20], 'suppressed': suppressed,
            'shot_calls': len(shot_attempts), 'shot_capture_failures': len(shot_capture_failures),
            'shot_errors': shot_capture_failures[:10],
        }}, open(OUT, 'w'), ensure_ascii=False)
    except Exception:
        pass


VERDICT_LABEL_RE = {verdict_label_pattern!r}
DATA_URI_RE = re.compile(r'(data:[^,;]+;base64,)([A-Za-z0-9+/=]+)', re.I)


def sanitize(value):
    """递归省略 data URI 的 base64 内容，保留类型、长度和非图片证据。"""
    if isinstance(value, str):
        def replace(match):
            size = len(match.group(2))
            return (match.group(1) + '<base64 内容已省略: ' + str(size)
                    + ' 字符; 二进制资源本身不表示页面错误>')
        return DATA_URI_RE.sub(replace, value)
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        return {{str(key): sanitize(item) for key, item in value.items()}}
    return value


def record(label, value):
    """记录一条原始事实，不记录通过或失败结论。

    布尔值和带判定语义的 label 会被过滤，并记录到 ``suppressed`` 供内部诊断。
    judge 通过脚本原文和其他 observation 了解实际检查内容。
    """
    import re as _re
    if isinstance(value, bool) or _re.search(VERDICT_LABEL_RE, str(label), _re.I):
        suppressed.append(str(label)[:120])
        dump()
        return
    if not isinstance(value, (str, int, float, type(None), list, dict)):
        value = str(value)
    value = sanitize(value)
    observations.append({{'label': str(label)[:120], 'value': value}})
    dump()


async def main():
    from playwright.async_api import async_playwright
    status, error = 'completed', ''
    async with async_playwright() as pw:
        # 容器使用镜像内固定版本的 Chromium；本地执行在该路径不存在时回退到
        # Playwright 自带的浏览器。
        _exe = {chromium_path!r}
        browser = await pw.chromium.launch(
            args={launch_args!r},
            **({{'executable_path': _exe}} if os.path.exists(_exe) else {{}}),
        )
        # 容器网络可能使用自签名证书链。忽略 HTTPS 证书错误，避免外部 CDN 资源加载失败。
        context = await browser.new_context(
            viewport={{'width': 1920, 'height': 1080}}, ignore_https_errors=True)
        # 动作超时收到 ACTION_TIMEOUT_MS；导航仍按 60s，与下面 goto 显式传的值一致，
        # 避免生成脚本自己 goto 时反而比原先更容易超时。
        context.set_default_timeout({action_timeout_ms})
        context.set_default_navigation_timeout(60000)
        page = await context.new_page()
        def _is_favicon(url):
            """判断请求是否为浏览器自动加载的站点图标。

            仅过滤常见图标路径；其他 console 错误和资源加载失败均保留为证据。
            """
            path = str(url or '').split('?')[0].lower()
            return path.endswith(('/favicon.ico', '/favicon.png', '/favicon.svg', '/apple-touch-icon.png'))

        def _on_console(message):
            if message.type != 'error':
                return
            # 资源加载错误的 URL 位于 console message 的 location 字段中。
            try:
                url = (message.location or {{}}).get('url', '')
            except Exception:
                url = ''
            if _is_favicon(url):
                return
            console_errors.append(message.text[:300])

        page.on('console', _on_console)
        page.on('pageerror', lambda e: page_errors.append(str(e)[:300]))
        page.on(
            'requestfailed',
            lambda r: None if _is_favicon(r.url) else failed_requests.append(r.url[:200]),
        )

        async def shot(name, label=''):
            safe = ''.join(c if c.isalnum() or c in '._-' else '_' for c in str(name))[:60] or 'shot'
            path = os.path.join(SHOTS, safe + '.png')
            shot_attempts.append(safe)
            try:
                # 先等入场动画跑完再截。此前用 animations='disabled' 冻结动画，本意是避开
                # transition 中间态，实际却把滚动入场动画（IntersectionObserver 加 class +
                # transition 显现）冻在了起始的 opacity:0 —— 实测 14/177 个产物（7.9%）
                # 有截图充实度不到同产物最高值的 35%，其中 4 张是纯色。
                # 那种图喂给美学 VLM 等于告诉它"这页是空的"，比截到中间态糟得多。
                await page.wait_for_timeout({settle_shot_ms})
                await page.screenshot(path=path, timeout={shot_timeout_ms})
                shot_meta.append({{'name': safe, 'label': str(label)[:200], 'path': path}})
                dump()
            except Exception as exc:
                shot_capture_failures.append(safe + ': ' + str(exc)[:150])
                dump()

        def _locator(sel):
            return page.locator(sel) if isinstance(sel, str) else sel

        async def _first(sel):
            try:
                loc = _locator(sel).first
                return loc if await loc.count() else None
            except Exception:
                return None

        # 查询 helper 自动记录选择器和结果，确保 agent 未显式调用 record 时仍能保留证据。
        async def text_of(sel):
            loc = await _first(sel)
            if loc is None:
                record('text_of ' + str(sel)[:60], '选择器无匹配元素')
                return None
            try:
                text = (await loc.inner_text())[:600]
                # 区分元素存在但文本为空与元素不存在。
                record('text_of ' + str(sel)[:60], text if text.strip() else '(元素存在但文本为空)')
                return text
            except Exception as exc:
                record('text_of ' + str(sel)[:60], '读取失败: ' + str(exc)[:120])
                return None

        async def count_of(sel):
            try:
                n = await _locator(sel).count()
            except Exception as exc:
                record('count_of ' + str(sel)[:60], '选择器无效: ' + str(exc)[:120])
                return 0
            record('count_of ' + str(sel)[:60], n)
            return n

        async def attr_of(sel, name):
            loc = await _first(sel)
            label = 'attr_of ' + str(sel)[:50] + '@' + str(name)[:20]
            if loc is None:
                record(label, '选择器无匹配元素')
                return None
            try:
                value = await loc.get_attribute(name)
                record(label, value if value is not None else '(元素存在但无此属性)')
                return value
            except Exception as exc:
                record(label, '读取失败: ' + str(exc)[:120])
                return None

        async def eval_js(expr):
            label = 'eval_js ' + str(expr)[:70]
            try:
                value = await page.evaluate(expr)
            except Exception as exc:
                record(label, 'eval_error: ' + str(exc)[:150])
                return 'eval_error: ' + str(exc)[:150]
            record(label, value)
            return value

        async def exists(sel):
            return (await count_of(sel)) > 0

        async def dom_state(sel='body'):
            """获取指定区域的无障碍树快照。

            快照包含可访问控件的状态，并自动排除隐藏元素。复杂页面应传入较小范围的 selector，
            以控制输出长度。
            """
            try:
                tree = await _locator(sel).first.aria_snapshot()
            except Exception as exc:
                record('dom_state ' + str(sel)[:50], '取不到无障碍树: ' + str(exc)[:120])
                return ''
            record('dom_state ' + str(sel)[:50], tree[:4000] if tree.strip() else '(该区域无可访问控件)')
            return tree

        async def appear(sel, timeout=5000):
            """等待元素可见，并自动记录等待结果和耗时。

            使用条件等待而不是固定 sleep，以适应动画和异步渲染耗时。超时记录包含等待上限，
            便于 judge 区分元素缺失、选择器不匹配和页面响应缓慢。
            """
            t0 = time.monotonic()
            try:
                await _locator(sel).first.wait_for(state='visible', timeout=timeout)
                # harness 通过 str.format() 构造，因此此处使用字符串拼接，避免花括号冲突。
                record('appeared:' + str(sel)[:60], str(int((time.monotonic() - t0) * 1000)) + 'ms')
                return True
            except Exception:
                record('appear_timeout:' + str(sel)[:60],
                       str(timeout) + 'ms 内没等到可见元素(可能:未实现 / 选择器不匹配 / 页面卡)')
                return False

        async def run_generated_script():
{body}

        try:
            await page.goto(URL, wait_until='domcontentloaded', timeout=60000)
            await page.wait_for_timeout(SETTLE_MS)
        except Exception as exc:
            # 页面不可达单独记录，避免与生成脚本自身异常混淆。
            status = 'page_unreachable'
            error = str(exc)[:400]
            record('page_load_error', error)

        if status == 'completed':
            try:
                await run_generated_script()
            except Exception:
                # 生成脚本异常时保留 traceback，具体归因由 verifier 完成。
                status = 'script_error'
                error = traceback.format_exc()[-1200:]

        try:
            await context.close()
            await browser.close()
        except Exception:
            pass

    # 记录截图元数据数量和磁盘文件数量，用于区分截图生成失败与下载失败。
    try:
        on_disk = sorted(f for f in os.listdir(SHOTS) if f.endswith('.png')) if os.path.isdir(SHOTS) else []
    except Exception as exc:
        on_disk = ['listdir_failed: ' + str(exc)[:80]]
    if shot_meta or on_disk:
        observations.append({{'label': '_shots_selfcheck',
                             'value': 'shot() 记录 %d 张, 磁盘 %d 张: %s' % (
                                 len(shot_meta), len(on_disk), ','.join(on_disk[:8]))}})

    dump(status, error)


asyncio.run(main())
'''


def _build_browser_script(code: str, url: str, out: str, shots: str, settle_ms: int) -> str:
    """Wrap generated code in the browser evidence harness."""
    body = '\n'.join(' ' * 12 + line for line in (code.splitlines() or ['pass']))
    from nexus.extensions.tasks.artifact.judges.capabilities.screenshot_in_container import (
        CHROMIUM_PATH_IN_CONTAINER,
    )

    return _BROWSER_HARNESS.format(
        url=url,
        out=out,
        shots=shots,
        settle_ms=settle_ms,
        body=body,
        action_timeout_ms=ACTION_TIMEOUT_MS,
        shot_timeout_ms=SHOT_TIMEOUT_MS,
        settle_shot_ms=SETTLE_SHOT_MS,
        # Reuse the Chromium path with this module's headless launch flags.
        chromium_path=CHROMIUM_PATH_IN_CONTAINER,
        launch_args=CHROMIUM_ARGS,
        # Pass label-filtering rules into the harness.
        verdict_label_pattern=VERDICT_LABEL,
    )


def _looks_b64(raw: str) -> bool:
    return isinstance(raw, str) and bool(raw) and not raw.lstrip().startswith('{')


# ── 执行 ──────────────────────────────────────────────────────────────────────


async def _run_browser(
    *,
    runtime: Any,
    script: dict[str, Any],
    url: str,
    settle_ms: int,
    timeout: float,
    save_dir: str | None,
) -> dict[str, Any]:
    from nexus.runtime.executors.command import run_command
    from nexus.runtime.executors.file import download_file
    from nexus.runtime.executors.models.command import RunCommandRequest
    from nexus.runtime.executors.models.file import DownloadFileRequest, FileItem

    fid = script['fid']
    root = f'/tmp/wj_{re.sub(r"[^A-Za-z0-9_]", "_", fid)}_{uuid.uuid4().hex[:6]}'
    body = _build_browser_script(script['code'], url, f'{root}/result.json', f'{root}/shots', settle_ms)
    encoded = base64.b64encode(body.encode()).decode()
    # Clear DISPLAY and LD_LIBRARY_PATH for system Python and headless Chromium.
    command = f'mkdir -p {root} && env -u DISPLAY -u LD_LIBRARY_PATH bash -c "echo {encoded} | base64 -d | python3"'
    evidence: dict[str, Any] = {
        'fid': fid,
        'kind': 'browser',
        'note': script.get('note', ''),
        'code': script.get('code', ''),
        'shots': [],
    }
    started = time.monotonic()
    try:
        run = await runtime.invoke(run_command, RunCommandRequest(command=command, timeout=timeout))
    except Exception as error:
        return {**evidence, 'status': 'invoke_failed', 'error': f'{type(error).__name__}: {error}'}

    killed = ''
    if run.return_code != 0:
        # Collect startup diagnostics from both output streams.
        detail = ((run.stderr or '') + (run.stdout or ''))[-800:]
        # Record explicit timeout diagnostics even when command output is missing.
        killed = (
            f'{timeout:.0f}s 超时被 kill'
            if run.return_code is None and not detail
            else f'rc={run.return_code} {detail}'
        )

    try:
        payload = await runtime.invoke(
            download_file,
            DownloadFileRequest(files=[FileItem(path=f'{root}/result.json', encoding='utf8')]),
        )
        raw = payload.files[0].content if payload.files else ''
        data = json.loads(base64.b64decode(raw) if _looks_b64(raw) else raw)
    except Exception as error:
        # A missing result file means the harness stopped before its first checkpoint.
        return {
            **evidence,
            'status': 'script_failed' if killed else 'no_result',
            'error': killed or f'读不到结果: {type(error).__name__}: {error}',
        }

    if killed:
        # Keep partial evidence while marking the script as failed.
        data['status'], data['error'] = 'script_failed', killed

    evidence.update({
        'status': data.get('status') or 'completed',
        'error': data.get('error') or '',
        'observations': data.get('observations') or [],
        'console_errors': data.get('console_errors') or [],
        'page_errors': data.get('page_errors') or [],
        'failed_requests': data.get('failed_requests') or [],
        'shot_calls': int(data.get('shot_calls') or 0),
        'shot_capture_failures': int(data.get('shot_capture_failures') or 0),
        'shot_errors': data.get('shot_errors') or [],
        'duration_ms': int((time.monotonic() - started) * 1000),
    })
    evidence['shots'] = await _download_shots(runtime, data.get('shots') or [], save_dir, fid)
    return evidence


async def _download_shots(
    runtime: Any, shots: list[dict[str, Any]], save_dir: str | None, fid: str
) -> list[dict[str, Any]]:
    """Download screenshots and optionally persist them locally."""
    if not shots:
        return []
    from nexus.runtime.executors.file import download_file
    from nexus.runtime.executors.models.file import DownloadFileRequest, FileItem

    try:
        payload = await runtime.invoke(
            # Download PNGs as base64, not UTF-8.
            download_file,
            DownloadFileRequest(files=[FileItem(path=s['path'], encoding='base64') for s in shots]),
        )
    except Exception as error:
        logger.warning('[%s] 截图下载失败: %s', fid, error)
        return [{'name': s['name'], 'label': s.get('label', ''), 'b64': '', 'download_failed': True} for s in shots]

    # Check per-file download errors explicitly.
    if getattr(payload, 'errors', None):
        logger.warning('[%s] 截图下载报错: %s', fid, str(payload.errors)[:300])
    got = len(payload.files or [])
    if got < len(shots):
        logger.warning('[%s] 只拉回 %d/%d 张截图', fid, got, len(shots))

    files_by_path = {item.path: item for item in payload.files or []}
    out: list[dict[str, Any]] = []
    for meta in shots:
        item = files_by_path.get(meta['path'])
        if item is None:
            out.append({
                'name': meta['name'],
                'label': meta.get('label', ''),
                'b64': '',
                'download_failed': True,
            })
            continue
        b64 = item.content or ''
        row: dict[str, Any] = {'name': meta['name'], 'label': meta.get('label', ''), 'b64': b64}
        if save_dir and b64:
            folder = Path(save_dir) / 'web_judger' / 'shots' / fid
            folder.mkdir(parents=True, exist_ok=True)
            try:
                target = folder / f'{meta["name"]}.png'
                target.write_bytes(base64.b64decode(b64))
                row['path'] = str(target)
            except Exception as error:
                logger.warning('[%s] 截图落盘失败: %s', fid, error)
        out.append(row)
    return out


async def _run_source(*, runtime: Any, script: dict[str, Any], timeout: float, source_dir: str) -> dict[str, Any]:
    from nexus.runtime.executors.command import run_command
    from nexus.runtime.executors.models.command import RunCommandRequest

    fid = script['fid']
    body = f'cd {source_dir} 2>/dev/null || {{ echo "=== 源码目录不存在 ==="; exit 125; }}\n{script["code"]}\n'
    encoded = base64.b64encode(body.encode()).decode()
    evidence: dict[str, Any] = {
        'fid': fid,
        'kind': 'source_code',
        'note': script.get('note', ''),
        'code': script.get('code', ''),
        'shots': [],
    }
    started = time.monotonic()
    try:
        run = await runtime.invoke(
            run_command, RunCommandRequest(command=f'echo {encoded} | base64 -d | bash', timeout=timeout)
        )
    except Exception as error:
        return {**evidence, 'status': 'invoke_failed', 'error': f'{type(error).__name__}: {error}'}

    # grep exit code 1 is a valid negative observation.
    completed = run.return_code in (0, 1)
    return {
        **evidence,
        'status': 'completed' if completed else 'script_error',
        'error': '' if completed else ((run.stderr or '') + (run.stdout or ''))[-1200:],
        'observations': [{'label': 'shell_output', 'value': (run.stdout or '')[:6000]}],
        'return_code': run.return_code,
        'duration_ms': int((time.monotonic() - started) * 1000),
    }


OK_STATUSES = frozenset({'completed', 'page_unreachable'})
# Page unreachability is an observation; script failures indicate collection errors.


async def execute(
    test_scripts: list[dict[str, Any]],
    *,
    runtime: Any,
    page_url: str,
    flow: dict[str, Any] | None = None,
    settle_ms: int = 2500,
    timeout: float = 300.0,
    concurrency: int = 4,
    save_dir: str | None = None,
    source_dir: str,
) -> list[dict[str, Any]]:
    """Execute collection scripts with bounded browser concurrency.

    Source checks share the candidate source directory; missing scripts produce
    explicit evidence records.
    """
    limit = asyncio.Semaphore(max(1, concurrency))

    async def one(script: dict[str, Any]) -> dict[str, Any]:
        try:
            if script['kind'] == 'source_code':
                return await _run_source(
                    runtime=runtime, script=script, timeout=min(timeout, 120.0), source_dir=source_dir
                )
            async with limit:
                return await _run_browser(
                    runtime=runtime,
                    script=script,
                    url=page_url,
                    settle_ms=settle_ms,
                    timeout=timeout,
                    save_dir=save_dir,
                )
        except Exception as error:
            return {
                'fid': script['fid'],
                'kind': script.get('kind', 'browser'),
                'shots': [],
                'status': 'harness_error',
                'error': f'{type(error).__name__}: {error}',
            }

    evidence = list(await asyncio.gather(*(one(s) for s in test_scripts)))

    # Emit missing_script evidence for every flow without a generated script.
    if flow:
        got = {e['fid'] for e in evidence}
        for item in flow.get('flows') or []:
            if item['fid'] not in got:
                evidence.append({
                    'fid': item['fid'],
                    'kind': item.get('check_via', 'browser'),
                    'shots': [],
                    'status': 'missing_script',
                    'error': 'flow2code 没有为这条 flow 产出代码',
                })
        order = {f['fid']: i for i, f in enumerate(flow.get('flows') or [])}
        evidence.sort(key=lambda e: order.get(e['fid'], 999))

    if save_dir:
        root = Path(save_dir) / 'web_judger'
        root.mkdir(parents=True, exist_ok=True)
        # Store PNGs separately; omit duplicate base64 data from evidence.json.
        (root / 'evidence.json').write_text(
            json.dumps(
                [
                    {**e, 'shots': [{k: v for k, v in s.items() if k != 'b64'} for s in e.get('shots') or []]}
                    for e in evidence
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )
    return evidence


__all__ = [
    'FLOW2CODE_MAX_TURNS',
    'FLOW2CODE_TIMEOUT',
    'OK_STATUSES',
    'SCRIPTS_FILE',
    'execute',
    'flow2code',
    'format_file_list',
]
