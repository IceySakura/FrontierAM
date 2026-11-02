"""rubric 子系统的中心化证据抓取层。

Fetcher 拥有所有来源解析 —— 文件系统读取、 mock 状态切片、
SQL 探针、 sidecar exec、 HTTP 探针。 primitive 通过调用本类
而不是直接触碰 state[...]。

第一阶段 （本文件） 是对现有 _impl.py 辅助函数的薄门面， 因此
将 primitive 从直接的 state 访问切换到 fetcher 调用是一次逐字节
一致的重构。 后续阶段将以增量分支加入 sidecar/HTTP/parse_as/script_probe
； 现有 YAML 表面语义永不改变。
"""

from __future__ import annotations

import asyncio
import json
import shlex

from dataclasses import dataclass
from typing import Any

from ._impl import (
    RubricLimits,
    _gather_source,
    _gather_trajectory,
    _resolve_state_source,
    _text_source_selector,
)


@dataclass
class FileSource:
    """结构化文件源规格 （`from:` 的新 dict 形式）。

    经典字符串形式 `from: "file:/path"` 继续原样通过 _gather_source 路由；
    只有当作者写 `from: {file: /path, service: ..., parse_as: ...}` （新能力） 时
    ， 该 dataclass 才会介入。
    """

    file: str
    service: str | None = None  # None => 主沙箱 （与今天逐字节一致）
    parse_as: str | None = None  # None => 原始字符串 （与今天逐字节一致）


@dataclass
class ProbeSpec:
    """新的 script_probe primitive 的结构化规格。"""

    script: str
    service: str = 'main'
    capture: str = 'stdout'  # stdout | stderr | exit_code
    timeout_sec: int = 15


class Fetcher:
    """拥有所有来源解析。 所有 primitive 都调用本类。"""

    def __init__(self, state: dict, runtime: Any = None, limits: RubricLimits | None = None):
        self.state = state
        self.runtime = runtime
        self.limits = limits

    async def trajectory(self, *, limits: RubricLimits | None = None) -> str:
        """遗留的轨迹文本。 与 _gather_trajectory 逐字节一致。"""
        return _gather_trajectory(self.state, limits=limits or self.limits)

    async def state_slice(self, cfg: dict) -> tuple[bool, Any]:
        """结构化 state 访问， 分发优先级与今天完全一致。

        委托给 _resolve_state_source； 需要 jmespath / SQL /
        state_sql: dispatch 的 primitive 继续直接调用专用的 _eval_* 辅助
        （它们拥有那部分逻辑）。 本方法仅覆盖朴素的
        `from: <namespace>.<path>` 情况。
        """
        source = cfg.get('from') or cfg.get('source', '')
        return _resolve_state_source(self.state, source)

    async def text_source(self, cfg: dict, *, limits: RubricLimits | None = None) -> str:
        """_gather_source 的门面。 与今天逐字节一致。"""
        source = _text_source_selector(cfg)
        source_ref = cfg.get('source_ref') if isinstance(cfg, dict) else None
        return _gather_source(self.state, source, source_ref, limits=limits or self.limits)

    async def file(self, source: FileSource | str) -> str | Any:
        """抓取一个文件， 可选从 sidecar 服务读取， 可选解析。

        遗留字符串形式 （"file:/path"） 保持逐字节一致。
        新的 dict 形式 （from: {file, service?, parse_as?}） 解锁：
          - service: str -> 从 sidecar 容器的运行时读取
          - parse_as: 'json'|'yaml'|'csv' -> 返回解析后的结构
        """
        # 遗留 string-source 路径： 与今天逐字节一致
        if isinstance(source, str):
            source = source[5:] if source.startswith('file:') else source
            return _gather_source(self.state, f'file:{source}', None, limits=self.limits)

        # 结构化 dict 形式
        fs = source
        raw = await self._read_file_bytes(fs)
        if fs.parse_as:
            return _parse_file_content(raw, fs.parse_as)
        return raw

    async def _read_file_bytes(self, fs: FileSource) -> str:
        if fs.service is None or fs.service == 'main':
            # 与今天的 from: file:<path> 逐字节一致
            return _gather_source(self.state, f'file:{fs.file}', None, limits=self.limits)
        # sidecar 情况： 复用 SQL 运行时解析器的模式 ——
        # 在 ExecutionContext 上查找已获取的 sidecar 运行时。
        try:
            from nexus.core import require_context

            ctx = require_context()
        except Exception:
            return ''
        acquired = getattr(ctx, '_runtimes', {}) or {}
        sidecar_runtime = acquired.get(fs.service)
        if sidecar_runtime is None:
            return ''  # 优雅降级： 与 from: file: 字符串形式的 on-missing 行为一致
        try:
            result = await sidecar_runtime.run_command(
                command=f'cat {shlex.quote(fs.file)}',
                timeout=15,
            )
        except Exception:
            return ''
        exit_code = getattr(result, 'return_code', None)
        if exit_code is None:
            exit_code = getattr(result, 'exit_code', None)
        if exit_code not in (0, None):
            return ''
        return getattr(result, 'stdout', '') or ''

    async def probe(self, spec: ProbeSpec) -> tuple[str, int]:
        """运行后执行脚本并捕获输出。 返回 (stdout, exit_code)。

        - service='main'： 在主 agent 沙箱中运行
        - service=<name>： 在 sidecar 服务容器中运行
        - 若 runtime 为 None， 优雅失败 （返回 ('', -1)）。
        """
        if self.runtime is None:
            return ('', -1)
        if spec.service == 'main':
            rt: Any = self.runtime
        else:
            # sidecar 情况： 在 ExecutionContext 上查找已获取的 sidecar 运行时
            # （与 ``_read_file_bytes`` 相同的模式）。
            try:
                from nexus.core import require_context

                ctx = require_context()
            except Exception:
                return ('', -1)
            acquired = getattr(ctx, '_runtimes', {}) or {}
            sidecar = acquired.get(spec.service)
            if sidecar is None:
                return ('', -1)
            rt = sidecar
        try:
            result = await asyncio.wait_for(rt.exec(spec.script), timeout=spec.timeout_sec)
            return (result.stdout or '', getattr(result, 'return_code', getattr(result, 'exit_code', -1)))
        except asyncio.TimeoutError:
            return ('', -2)
        except Exception:
            return ('', -3)


def _parse_file_content(raw: str, parse_as: str) -> Any:
    """将抓取到的原始字符串解析为结构化值。

    缺失时静默返回空 （匹配遗留 from: file: 行为）。 解析出错时返回原始
    字符串， 使下游 jmespath / regex 仍可继续处理。
    """
    if not raw:
        return None if parse_as in ('json', 'yaml') else []
    if parse_as == 'json':
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw
    if parse_as == 'yaml':
        try:
            import yaml

            return yaml.safe_load(raw)
        except Exception:
            return raw
    if parse_as == 'csv':
        import csv
        import io

        try:
            return list(csv.DictReader(io.StringIO(raw)))
        except Exception:
            return raw
    return raw


__all__ = ['Fetcher', 'FileSource', 'ProbeSpec']
