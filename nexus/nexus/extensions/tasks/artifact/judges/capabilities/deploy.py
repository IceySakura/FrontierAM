"""Artifact collection, Render Service deployment, and shared screenshot contexts."""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures as futures
import functools
import logging
import mimetypes
import os
import os.path as osp
import shutil
import tempfile
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


ARTIFACT_SAVE_SCREENSHOTS_ENV = 'ARTIFACT_SAVE_SCREENSHOTS_DIR'

# Render Service endpoint and request settings.
RENDER_HOST_ENV = 'RENDER_HOST'
RENDER_DEFAULT_HOST = 'http://render.nexus-agent.woa.com:8088'
RENDER_HOLD_SECONDS = 3600
"""沙盒存活时长。到期即回收、URL 失效 —— 覆盖一次判分加缓存 replay 的余量。"""
RENDER_WAIT_TIMEOUT = 1500.0
"""轮询总超时。超过按 infra 故障处理（maskout），不冒充产物问题。"""
RENDER_POLL_INTERVAL = 15.0
RENDER_TERMINAL_STATES = frozenset({'success', 'failed', 'timeout', 'destroyed', 'expired'})

# Exclude dependencies, build outputs, caches, and version-control files.
_EXCLUDED_DIRS = {
    # deps
    'node_modules',
    'bower_components',
    '.pnpm-store',
    # package manager caches
    '.yarn',
    '.npm',
    '.bun',
    '.bun-cache',
    # framework/build output
    '.next',
    '.nuxt',
    '.output',
    '.svelte-kit',
    '.astro',
    '.angular',
    '.vite',
    '.parcel-cache',
    '.turbo',
    '.vercel',
    '.netlify',
    'dist',
    'build',
    'out',
    'coverage',
    'storybook-static',
    # test/report/cache
    'playwright-report',
    'test-results',
    '.nyc_output',
    '.cache',
    # Exclude browser tool state; extra files affect deployment routing and artifact counts.
    '.playwright-cli',
    # tool/state
    '.git',
    '.hg',
    '.svn',
    '.idea',
    '.vscode',
    '__pycache__',
}
_EXCLUDED_FILES = {'.gitignore'}


# ── Artifact collection (runtime-side tar) ────────────────────────────────────


def _tar_working_dir(
    working_dir: str = '/workspace',
    extra_exclude_paths: list[str] | None = None,
    extra_exclude_extensions: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Archive workspace files as base64 tar.gz with relative paths.

    Extra path-prefix and case-insensitive extension exclusions are combined with OR.
    """
    import io
    import tarfile

    exclude_paths = [p.strip('/') for p in (extra_exclude_paths or []) if p and p.strip('/')]
    exclude_extensions = {f'.{e.strip().lstrip(".").lower()}' for e in (extra_exclude_extensions or []) if e.strip()}

    def _is_excluded(rel_path: str) -> bool:
        if exclude_extensions and os.path.splitext(rel_path)[1].lower() in exclude_extensions:
            return True
        return any(rel_path == prefix or rel_path.startswith(f'{prefix}/') for prefix in exclude_paths)

    buf = io.BytesIO()
    file_list: list[str] = []

    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for root, dirs, files in os.walk(working_dir):
            kept_dirs = []
            for d in dirs:
                if d in _EXCLUDED_DIRS:
                    continue
                if _is_excluded(os.path.relpath(os.path.join(root, d), working_dir)):
                    continue
                kept_dirs.append(d)
            dirs[:] = kept_dirs
            for f in files:
                if f in _EXCLUDED_FILES:
                    continue
                full = os.path.join(root, f)
                rel = os.path.relpath(full, working_dir)
                if _is_excluded(rel):
                    continue
                try:
                    tar.add(full, arcname=rel)
                    file_list.append(rel)
                except Exception:
                    pass

    return base64.b64encode(buf.getvalue()).decode(), file_list


async def collect_project_archive(
    runtime: Any,
    working_dir: str = '/workspace',
    exclude_paths: list[str] | None = None,
    exclude_extensions: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Collect an archive and file list in the runtime using shared exclusion rules."""
    return await runtime.invoke(_tar_working_dir, working_dir, exclude_paths, exclude_extensions)


# ── Archive decode ────────────────────────────────────────────────────────────

# Keep CPU-bound decoding off the default asyncio executor.
_DEPLOY_DECODE_POOL = futures.ThreadPoolExecutor(
    max_workers=min(64, (os.cpu_count() or 8)), thread_name_prefix='deploy-decode'
)


async def run_cpu_bound(fn: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Run CPU-bound decoding or image work in the shared executor pool."""
    call = functools.partial(fn, *args, **kwargs) if kwargs else fn
    return await asyncio.get_running_loop().run_in_executor(_DEPLOY_DECODE_POOL, call, *(() if kwargs else args))


BINARY_PREFIX = 'data:'
# Binary files use data URIs; filter them before scanning source text.


def text_files(files: dict[str, str]) -> dict[str, str]:
    """Filter binary data URIs out of a file-content mapping."""
    return {k: v for k, v in files.items() if not v.startswith(BINARY_PREFIX)}


def _file_item_for(path: str, content: str) -> Any:
    """Restore binary data URIs as base64 file items; keep text as UTF-8."""
    from nexus.runtime.executors.file import FileItem

    if content.startswith(BINARY_PREFIX) and ';base64,' in content:
        payload = content.split(';base64,', 1)[1]
        return FileItem(path=path, content=payload, encoding='base64')
    return FileItem(path=path, content=content, encoding='utf8')


def _load_project_files(local_project_dir: str) -> dict[str, str]:
    """Map relative paths to text or binary data URIs for the deployment protocol."""
    code_content_dict: dict[str, str] = {}
    for root, _dirs, files in os.walk(local_project_dir):
        for file in files:
            abs_path = osp.join(root, file)
            rel_path = osp.relpath(abs_path, local_project_dir).replace('\\', '/')
            try:
                with open(abs_path, encoding='utf-8') as f:
                    code_content_dict[rel_path] = f.read()
            except UnicodeDecodeError:
                mime = mimetypes.guess_type(abs_path)[0] or 'application/octet-stream'
                with open(abs_path, 'rb') as fb:
                    code_content_dict[rel_path] = f'data:{mime};base64,{base64.b64encode(fb.read()).decode()}'
            except OSError as e:
                logger.debug('Skipping unreadable file %s: %s', file, e)
    return code_content_dict


def archive_to_code_content_dict(archive_b64: str) -> tuple[dict[str, str], bool]:
    """Decode an archive into a file map, stripping a common directory prefix when present."""
    import io
    import tarfile

    if not archive_b64:
        return {}, False

    tmp_dir = tempfile.mkdtemp(prefix='judge_archive_')
    try:
        raw = base64.b64decode(archive_b64)
        with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as tar:
            try:
                tar.extractall(tmp_dir, filter='data')  # trusted infer artifact
            except TypeError:
                # Fallback for Python versions without extraction filters.
                tar.extractall(tmp_dir)

        code_content_dict = _load_project_files(tmp_dir)
        has_common_prefix = False

        if code_content_dict:
            keys = list(code_content_dict.keys())
            common = osp.commonprefix(keys)
            common = common[: common.rfind('/') + 1] if '/' in common else ''
            if common:
                has_common_prefix = True
                code_content_dict = {k[len(common) :]: v for k, v in code_content_dict.items()}

        return code_content_dict, has_common_prefix
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def upload_archive_to_runtime(
    runtime: Any,
    archive_b64: str,
    working_dir: str = '/workspace',
) -> tuple[int, str | None]:
    """Upload decoded files and return (file_count, error).

    Require successful per-file results for every input file; missing results fail.
    """
    code_content_dict, _ = await run_cpu_bound(archive_to_code_content_dict, archive_b64)
    if not code_content_dict:
        return 0, 'Archive produced no code files'

    # Decode binary data URIs through _file_item_for.
    file_items = [
        _file_item_for(f'{working_dir}/{rel_path}', content) for rel_path, content in code_content_dict.items()
    ]

    try:
        response = await runtime.upload_file(files=file_items, overwrite=True)
    except Exception as e:
        return 0, f'upload_file failed: {type(e).__name__}: {e}'

    # Accept model or dict upload responses; missing per-file results are failures.
    results = getattr(response, 'results', None)
    if results is None and isinstance(response, dict):
        results = response.get('results')
    if results is None:
        return 0, (
            f'upload_file returned no per-file results ({type(response).__name__}); '
            'cannot confirm the source was written'
        )

    failures: list[str] = []
    for item in results:
        ok = getattr(item, 'success', None)
        path = getattr(item, 'path', None)
        message = getattr(item, 'message', None)
        if isinstance(item, dict):
            ok = item.get('success')
            path = item.get('path')
            message = item.get('message')
        if not ok:
            failures.append(f'{path}: {message or "unknown error"}')

    if failures:
        shown = '; '.join(failures[:5])
        suffix = f' (+{len(failures) - 5} more)' if len(failures) > 5 else ''
        return 0, f'upload_file failed for {len(failures)}/{len(file_items)} file(s): {shown}{suffix}'

    uploaded = len(results)
    if uploaded != len(file_items):
        # Every submitted file must have an upload result.
        return 0, f'upload_file reported {uploaded} result(s) for {len(file_items)} file(s)'

    logger.debug('upload_archive_to_runtime: uploaded %d files to %s', uploaded, working_dir)
    return uploaded, None


# ── Screenshot persistence + bundling ─────────────────────────────────────────


def _save_b64_png(save_dir: str, filename: str, b64: str) -> str | None:
    """Write a base64 PNG, returning its path or None on failure."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        out_path = osp.join(save_dir, filename)
        with open(out_path, 'wb') as f:
            f.write(base64.b64decode(b64))
        return out_path
    except Exception as e:
        logger.warning('failed to save screenshot %s/%s: %s', save_dir, filename, e)
        return None


def resolve_screenshot_save_dir(cfg: Any, instance_id: str) -> str | None:
    """Resolve the configured evidence root and per-instance directory."""
    root = (getattr(cfg, 'save_screenshots_dir', '') or '').strip()
    if not root:
        from nexus.extensions.tasks.artifact import config

        root = config.get().obs.save_screenshots_dir
    if not root:
        return None
    return osp.join(root, instance_id or 'unknown')


@dataclass
class ScreenshotSet:
    """Screenshots shared by judges within one evaluation."""

    sandbox: list[str] = field(default_factory=list)
    improved: str | None = None

    def for_vlm(self) -> list[str]:
        """按优先级返回用于 VLM 评分的截图。"""
        if self.improved:
            return [self.improved]
        return list(self.sandbox)

    def is_empty(self) -> bool:
        return not self.sandbox and not self.improved

    def save(self, save_dir: str, label: str) -> None:
        """Persist available screenshots; log individual write failures."""
        if self.is_empty():
            return
        for i, shot in enumerate(self.sandbox):
            _save_b64_png(save_dir, f'{label}_sandbox_{i}.png', shot)
        if self.improved:
            _save_b64_png(save_dir, f'{label}_improved.png', self.improved)


@dataclass
class SandboxContext:
    """Deployment and screenshot context shared by judges.

    Use sandbox_url for scoring and sandbox_proxy_url for human inspection only.
    Screenshots are supplied by container capture, not generated by this object.
    """

    screenshots: ScreenshotSet = field(default_factory=ScreenshotSet)
    sandbox_url: str | None = None
    sandbox_proxy_url: str | None = None
    deploy_error: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.sandbox_url)

    async def ensure_screenshots(self) -> ScreenshotSet:
        """Return the supplied screenshots, which may be empty."""
        return self.screenshots


def save_screenshot(save_dir: str | None, instance_id: str, shot_b64: str) -> None:
    """Persist a candidate PNG without affecting the verdict."""
    if not (save_dir and shot_b64):
        return
    try:
        out = Path(save_dir) / f'{instance_id}.png'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(shot_b64))
    except Exception as exc:  # noqa: BLE001
        logger.warning('[%s] screenshot save failed: %s', instance_id[:12], exc)


def classify_deploy_error(deploy_error: str) -> tuple[bool, str]:
    """Return (is_artifact_fault, reason). Only the artifact: prefix is a model fault.

    Infrastructure and unprefixed errors must be masked out.
    """
    if deploy_error.startswith('artifact: '):
        return True, deploy_error
    return False, deploy_error


def sandbox_context_from_screenshot(
    *,
    sandbox_url: str,
    screenshot_b64: str,
    sandbox_proxy_url: str | None = None,
) -> SandboxContext:
    """Build a sandbox context from a container-captured screenshot."""
    return SandboxContext(
        screenshots=ScreenshotSet(improved=screenshot_b64) if screenshot_b64 else ScreenshotSet(),
        sandbox_url=sandbox_url or None,
        sandbox_proxy_url=sandbox_proxy_url,
    )


async def prepare_sandbox_context(
    archive_b64: str,
    *,
    instance_id: str = '',
) -> SandboxContext:
    """Upload the archive to COS, submit a render task, and poll to a terminal state.

    frontend_url is the scoring URL; frontend_public_url is for human inspection.
    base_url is a control-plane endpoint, not the artifact. Success requires a
    success status, success flag, and non-empty frontend URL. Only status=failed
    is an artifact fault; other failures use the infra prefix.
    """
    from nexus.extensions.tasks.artifact.agent_setup import cos_io

    if not archive_b64:
        return SandboxContext(deploy_error='infra: archive_b64 is empty')
    gongfeng_token = (
        os.environ.get('GONGFENG_TOKEN', '').strip() or os.environ.get('GONGFENG_RUNTIME_TOKEN', '').strip()
    )
    if not gongfeng_token:
        return SandboxContext(
            deploy_error='infra: GONGFENG_TOKEN or GONGFENG_RUNTIME_TOKEN is required for render deployment'
        )

    # Upload the archive to COS.
    try:
        key = cos_io.build_archive_key(
            instance_id=instance_id or 'unknown',
            archive_name='workspace.tar.gz',
        )
        archive_cos_url = await asyncio.to_thread(cos_io.upload_archive_b64, key, archive_b64)
    except Exception as exc:  # noqa: BLE001
        logger.exception('prepare_sandbox_context: cos upload failed')
        return SandboxContext(deploy_error=f'infra: cos upload failed: {type(exc).__name__}: {exc}')

    # Submit the render task and poll to a terminal state.
    host = os.environ.get(RENDER_HOST_ENV, RENDER_DEFAULT_HOST).rstrip('/')
    body = {
        'archive_cos_url': archive_cos_url,
        'gongfeng_token': gongfeng_token,
        'hold_seconds': RENDER_HOLD_SECONDS,
        # Required by the service even for frontend-only requests.
        'agent_retries': 3,
        # Force a fresh render rather than reuse a cached deployment.
        'force': True,
        'frontend_only': True,
        'business_tag': os.environ.get('RENDER_BUSINESS_TAG', 'frontend rl'),
        'purpose': os.environ.get('RENDER_PURPOSE', 'prod'),
    }

    logger.debug(
        'render POST %s/render archive=%s tag=%s purpose=%s',
        host,
        archive_cos_url,
        body['business_tag'],
        body['purpose'],
    )

    code, resp = await asyncio.to_thread(_render_http, 'POST', f'{host}/render', body)
    if code not in (200, 202) or not isinstance(resp, dict) or not resp.get('task_id'):
        detail = resp.get('_error') or resp.get('_raw') or resp.get('error') or resp
        return SandboxContext(deploy_error=f'infra: render POST failed ({code}): {detail}'[:400])
    task_id = str(resp['task_id'])
    logger.debug('render submitted task_id=%s (reused=%s)', task_id, bool(resp.get('reused')))

    deadline = time.monotonic() + RENDER_WAIT_TIMEOUT
    final: dict[str, Any] = {}
    while True:
        _, final = await asyncio.to_thread(_render_http, 'GET', f'{host}/render/{task_id}')
        status = str(final.get('status') or '?') if isinstance(final, dict) else '?'
        if status in RENDER_TERMINAL_STATES:
            break
        if time.monotonic() >= deadline:
            return SandboxContext(
                deploy_error=f'infra: render poll exceeded {RENDER_WAIT_TIMEOUT:.0f}s (last status={status})'
            )
        await asyncio.sleep(RENDER_POLL_INTERVAL)

    # 3. 终态 → SandboxContext。只有 ``status='failed'`` 是模型侧（服务端已经
    #    跑到「构建/启动失败」这一步）；``timeout`` / ``destroyed`` / ``expired``
    #    都是 evaluator/infra 侧，走 maskout 保护训练信号。
    status = str(final.get('status'))
    frontend_url = str(final.get('frontend_url') or '')
    err = final.get('error')

    if status == 'success' and final.get('success') and frontend_url:
        deploy_error = None
    elif status == 'failed':
        deploy_error = f'artifact: {err or "render failed with no error"}'[:400]
    else:
        # 包含 status='success' 但没给 frontend_url 的畸形响应 —— 判分拿不到
        # 地址就没法评，算评测侧故障。
        deploy_error = f'infra: render status={status} ({err or "no frontend_url"})'[:400]

    logger.info(
        'render done task_id=%s status=%s url=%s timings=%s',
        task_id,
        status,
        frontend_url or '<none>',
        final.get('stage_timings'),
    )

    return SandboxContext(
        sandbox_url=frontend_url or None,
        sandbox_proxy_url=str(final.get('frontend_public_url') or '') or None,
        deploy_error=deploy_error,
    )


def deploy_failure_results(judge_names: tuple[str, ...], error: str) -> list[Any]:
    """Build deployment failure results using the shared artifact/infra classification."""
    from nexus.extensions.tasks.artifact.judges.base import JudgeResult

    is_artifact_fault, reason = classify_deploy_error(error)
    results = []
    for name in judge_names:
        if is_artifact_fault:
            # JudgeResult 强制「error 必带 failure_kind」，所以产物侧故障只在
            # metadata 中记录，保持它是可训练的零分结果。
            result = JudgeResult(
                judge_name=name,
                score=0.0,
                passed=False,
                metadata={'deploy_failed': True, 'deploy_error': reason},
            )
        else:
            result = JudgeResult(
                judge_name=name,
                score=0.0,
                error=f'deploy failed: {reason}',
                failure_kind='no_sandbox',
                metadata={'deploy_error': reason},
            )
        results.append(result)
    return results


def _render_http(
    method: str, url: str, body: dict[str, Any] | None = None, timeout: float = 90.0
) -> tuple[int, dict[str, Any]]:
    """Render Service 的一次 HTTP 调用，返回 ``(status_code, parsed_body)``。

    同步实现，调用方用 :func:`asyncio.to_thread` 包起来，POST / GET 期间不占事件
    循环。用 stdlib urllib 而不是 requests：判分跑在容器 worker venv 里，少一个
    依赖少一处装不上的可能。传输层异常转成 ``(0, {'_error': …})`` 而不是抛出，
    让 :func:`prepare_sandbox_context` 统一打 ``infra:`` 前缀。
    """
    import json as _json
    import urllib.error
    import urllib.request

    data = _json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header('Content-Type', 'application/json')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode(errors='replace')
            try:
                return resp.status, _json.loads(raw)
            except _json.JSONDecodeError:
                return resp.status, {'_raw': raw}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors='replace')
        try:
            return exc.code, _json.loads(raw)
        except _json.JSONDecodeError:
            return exc.code, {'_raw': raw}
    except Exception as exc:  # noqa: BLE001
        return 0, {'_error': str(exc)}


__all__ = [
    'ARTIFACT_SAVE_SCREENSHOTS_ENV',
    'BINARY_PREFIX',
    'RENDER_DEFAULT_HOST',
    'RENDER_HOLD_SECONDS',
    'RENDER_HOST_ENV',
    'SandboxContext',
    'ScreenshotSet',
    'archive_to_code_content_dict',
    'classify_deploy_error',
    'collect_project_archive',
    'deploy_failure_results',
    'prepare_sandbox_context',
    'resolve_screenshot_save_dir',
    'run_cpu_bound',
    'sandbox_context_from_screenshot',
    'save_screenshot',
    'text_files',
    'upload_archive_to_runtime',
]
