"""aidata 用例的辅助服务容器编排。

一个 *service* 是与主 agent 容器并列运行的第二个 docker/gongfeng 运行时。 该模式与
``mcp_benchmark.task._setup_services`` 一致，存在的原因也相同：某些后端（例如
WooCommerce mock ）应仅通过网络访问，而不通过 agent 的 shell MCP —— 这样即使
agent 技术上可以 `cat` 底层 JSON ，也强制其必须使用 REST API 抽象。

接线：

  0. 用例在 case.yaml 中声明 ``services: [...]`` （见 AidataService 模型）。
  1. ``setup_services`` 对每个服务调用 ``runtime_provider.configure(s.runtime_name, ...)`` ，
     以便后续的 ``ctx.get_runtime(s.runtime_name)`` 使用正确的镜像 + exposed_ports 。
  2. ``ctx.get_runtime(s.runtime_name)`` 触发容器实际启动。 正常情况下，返回的 Runtime 的
     ``.port_mappings`` 字典包含 ``{internal_port: "http://<host>:<auto_assigned_host_port>"}`` ；
     若排队启动导致映射晚于 Runtime 就绪，连接解析前会通过 provider 重新查询。
  3. ``s.startup.command`` 在服务容器 **内部** 运行（绑定至 0.0.0.0:<internal>），
     然后轮询 ``s.ready_check.command`` 直到其以 0 退出或 ``ready_timeout_s`` 到期。
  4. 每个 ``s.connections[]`` 条目会解析为一个具体的 ``host:port`` 字符串 ——
     作为 ``ServiceReplacements`` 包返回，编排器会将其拼接进
     user_prompt / MCP server 规格 / bootstrap env 中。

清理工作依赖于 ``ExecutionContext.__aexit__`` 调用 ``runtime_provider.release_all()``；
此处没有 teardown 钩子。
"""

from __future__ import annotations

import asyncio
import base64
import logging
import shlex

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from nexus.core import require_context
from nexus.runtime.executors.command import CommandRunStatus
from nexus.runtime.executors.file import FileItem

from .models import AidataService, AidataServiceCommand


logger = logging.getLogger(__name__)


@dataclass
class ServiceReplacements:
    """:func:`setup_services` 的输出 —— 编排器需要在下游应用的两类替换。

    ``text`` ： 将每个 connection 的 ``placeholder`` 映射到解析后的 ``host:port``
        字符串。 作为字面量 str.replace 应用于 user_prompt 以及
        ``extra_mcp_servers`` 中的每个 str 字段。
    ``env``  ： ``{env_name: resolved_value}`` —— 运行 ``bootstrap_script`` 时
        作为 extra_env 注入。 只有声明了 ``env_name`` 的连接才会贡献于此。
    """

    text: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


def _runtime_config_for(service: AidataService) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if service.image:
        cfg['image'] = service.image
    if service.exposed_ports:
        cfg['exposed_ports'] = list(service.exposed_ports)
    if service.working_dir:
        cfg['working_dir'] = service.working_dir
    return cfg


def _resolve_connection_values(service: AidataService, runtime: Any) -> tuple[dict[str, str], dict[str, str]]:
    text: dict[str, str] = {}
    env: dict[str, str] = {}
    for conn in service.connections:
        if conn.internal_port not in runtime.port_mappings:
            raise RuntimeError(
                f'service {service.name!r}: missing port mapping for internal_port={conn.internal_port}; '
                f'available={sorted(runtime.port_mappings)}'
            )
        base_url = runtime.port_mappings[conn.internal_port]
        parsed = urlparse(base_url)
        host = parsed.hostname
        port = parsed.port
        if not host or port is None:
            raise RuntimeError(f'service {service.name!r}: invalid port mapping {base_url!r}')
        value = conn.value_template.format(
            base_url=base_url,
            host=host,
            port=port,
            internal_port=conn.internal_port,
        )
        text[conn.placeholder] = value
        if conn.env_name:
            env[conn.env_name] = value
    return text, env


async def _wait_for_connection_port_mappings(service: AidataService, runtime: Any, provider: Any) -> None:
    """Refresh delayed provider mappings needed by ``service.connections``.

    Gongfeng may report a queued runtime as scheduled before its requested port
    mappings are visible. The Runtime object is only a snapshot, so waiting on
    it alone cannot discover mappings that appear later in the provider API.
    """
    required_ports = {conn.internal_port for conn in service.connections}
    missing_ports = required_ports - set(runtime.port_mappings)
    if not missing_ports:
        return

    get_runtime_info = getattr(provider, 'get_runtime_info', None)
    if not callable(get_runtime_info):
        logger.warning(
            '[aidata] service %s: runtime is missing required port mappings %s and provider %s cannot refresh them',
            service.name,
            sorted(missing_ports),
            type(provider).__name__,
        )
        return

    loop = asyncio.get_running_loop()
    timeout_s = max(0.0, float(service.ready_timeout_s))
    poll_interval_s = max(0.1, float(service.ready_poll_interval_s))
    deadline = loop.time() + timeout_s
    attempts = 0
    last_error: Exception | None = None
    logger.warning(
        '[aidata] service %s: runtime is missing required port mappings %s; refreshing for up to %.1fs',
        service.name,
        sorted(missing_ports),
        timeout_s,
    )

    while True:
        attempts += 1
        try:
            runtime_info = await get_runtime_info(service.runtime_name)
            if isinstance(runtime_info, Mapping):
                raw_port_mappings = runtime_info.get('port_mappings') or {}
                if isinstance(raw_port_mappings, Mapping):
                    for internal_port, external_url in raw_port_mappings.items():
                        try:
                            parsed_port = int(internal_port)
                        except (TypeError, ValueError):
                            logger.warning(
                                '[aidata] service %s: ignoring invalid provider port mapping key %r',
                                service.name,
                                internal_port,
                            )
                            continue
                        if not isinstance(external_url, str) or not external_url:
                            logger.warning(
                                '[aidata] service %s: ignoring invalid provider URL for port %s: %r',
                                service.name,
                                parsed_port,
                                external_url,
                            )
                            continue
                        runtime.port_mappings[parsed_port] = external_url
            last_error = None
        except Exception as exc:
            last_error = exc
            logger.debug(
                '[aidata] service %s: failed to refresh runtime port mappings',
                service.name,
                exc_info=True,
            )

        missing_ports = required_ports - set(runtime.port_mappings)
        if not missing_ports:
            logger.debug(
                '[aidata] service %s: required port mappings became available after %d refresh attempt(s)',
                service.name,
                attempts,
            )
            return

        remaining_s = deadline - loop.time()
        if remaining_s <= 0:
            break
        await asyncio.sleep(min(poll_interval_s, remaining_s))

    first_missing_port = min(missing_ports)
    raise RuntimeError(
        f'service {service.name!r}: missing port mapping for internal_port={first_missing_port}; '
        f'available={sorted(runtime.port_mappings)}; required={sorted(required_ports)}; '
        f'missing={sorted(missing_ports)}; waited={timeout_s:.1f}s; '
        f'refresh_attempts={attempts}; last_error={last_error!r}'
    )


_SCRIPT_INTERPRETERS = {
    '.sh': 'bash',
    '.py': 'python3',
    '.bash': 'bash',
}
"""服务脚本的扩展名 → 解释器分发表。 未列出的会直接运行文件
（依赖其 shebang ）。"""


async def _upload_script(
    runtime: Any,
    *,
    cmd: AidataServiceCommand,
    case_dir: Path,
    service_name: str,
    phase: str,
) -> str:
    """将 ``cmd.script`` 上传到服务容器；返回 shell 调用字符串。 无操作契约：
    调用方必须已经验证过 ``cmd.script`` 。
    """
    assert cmd.script is not None
    script_path = (case_dir / cmd.script).resolve()
    if not script_path.is_file():
        raise RuntimeError(
            f'service {service_name!r} {phase!r}: script not found at {script_path} '
            f'(case_dir={case_dir}, script={cmd.script})'
        )
    ext = script_path.suffix.lower() or '.sh'
    remote_path = f'/tmp/aidata-svc-{service_name}-{phase}{ext}'
    content_b64 = base64.b64encode(script_path.read_bytes()).decode('ascii')
    upload = await runtime.upload_file(
        files=[FileItem(path=remote_path, content=content_b64, encoding='base64')],
        overwrite=True,
    )
    failures = [r for r in upload.results if not r.success]
    if failures:
        raise RuntimeError(
            f'service {service_name!r} {phase!r}: failed to upload {cmd.script} -> {remote_path}: {failures[0].message}'
        )
    interpreter = _SCRIPT_INTERPRETERS.get(ext)
    # chmod +x 即使运行时拒绝也是无害的；仍然尝试执行，以便当脚本
    # 有 shebang 时可以回退到直接执行。
    if interpreter:
        return f'chmod +x {remote_path} 2>/dev/null; {interpreter} {remote_path}'
    return f'chmod +x {remote_path} && {remote_path}'


async def _upload_service_files(runtime: Any, service: AidataService, *, case_dir: Path | None) -> None:
    """将 ``services[].files`` 条目上传到服务运行时。"""
    if not service.files:
        return
    if case_dir is None:
        raise RuntimeError(f'service {service.name!r}: files require case_dir so sources can be resolved')

    for file_spec in service.files:
        source_path = (case_dir / file_spec.source).resolve()
        if not source_path.is_file():
            raise RuntimeError(
                f'service {service.name!r}: file source not found at {source_path} '
                f'(case_dir={case_dir}, source={file_spec.source})'
            )
        target = file_spec.target
        target_parent = str(Path(target).parent)
        result = await runtime.run_command(command=f'mkdir -p {shlex.quote(target_parent)}', timeout=10)
        status = getattr(result, 'status', None)
        rc = getattr(result, 'return_code', None)
        if (status is not None and status != CommandRunStatus.Finished) or rc not in (0, None):
            stderr = getattr(result, 'stderr', '') or ''
            raise RuntimeError(
                f'service {service.name!r}: failed to create parent dir {target_parent!r}; '
                f'status={getattr(status, "value", status)}, rc={rc}, stderr head: {stderr[:400]!r}'
            )

        content_b64 = base64.b64encode(source_path.read_bytes()).decode('ascii')
        upload = await runtime.upload_file(
            files=[FileItem(path=target, content=content_b64, encoding='base64')],
            overwrite=True,
        )
        failures = [r for r in upload.results if not r.success]
        if failures:
            raise RuntimeError(
                f'service {service.name!r}: failed to upload {file_spec.source} -> {target}: {failures[0].message}'
            )
        logger.debug('[aidata] service %s: uploaded %s -> %s', service.name, file_spec.source, target)


async def _run_in_service(
    runtime: Any,
    cmd: AidataServiceCommand,
    *,
    case_dir: Path | None = None,
    service_name: str = 'svc',
    phase: str = 'cmd',
) -> tuple[int | None, str, str]:
    if cmd.script:
        if case_dir is None:
            raise RuntimeError(
                f'service {service_name!r} {phase!r}: script mode requires case_dir '
                '(engine invocation bug — setup_services must pass case_dir).'
            )
        resolved_command = await _upload_script(
            runtime,
            cmd=cmd,
            case_dir=case_dir,
            service_name=service_name,
            phase=phase,
        )
    else:
        assert cmd.command is not None  # AidataServiceCommand.__init__ 强制 XOR
        resolved_command = cmd.command

    result = await runtime.run_command(
        command=resolved_command,
        cwd=cmd.cwd,
        timeout=cmd.timeout_s,
        extra_env=cmd.env or {},
    )
    status = getattr(result, 'status', None)
    rc = getattr(result, 'return_code', None)
    stdout = getattr(result, 'stdout', '') or ''
    stderr = getattr(result, 'stderr', '') or ''
    if status is not None and status != CommandRunStatus.Finished:
        raise RuntimeError(
            f'command did not finish: status={getattr(status, "value", status)}; stderr head: {stderr[:400]!r}'
        )
    return rc, stdout, stderr


async def _wait_ready(runtime: Any, service: AidataService, *, case_dir: Path | None) -> None:
    """轮询 ``service.ready_check`` 直到其以 0 退出，或抛出异常。"""
    if service.ready_check is None:
        return

    loop = asyncio.get_running_loop()
    deadline = loop.time() + service.ready_timeout_s
    last_err: Exception | None = None
    attempts = 0

    while True:
        attempts += 1
        try:
            rc, _out, _err = await _run_in_service(
                runtime,
                service.ready_check,
                case_dir=case_dir,
                service_name=service.name,
                phase='ready_check',
            )
            if rc == 0:
                logger.debug('[aidata] service %s: ready after %d attempt(s)', service.name, attempts)
                return
            last_err = RuntimeError(f'ready_check exited {rc}; stderr head: {_err[:200]!r}')
        except Exception as exc:
            last_err = exc
        if loop.time() >= deadline:
            break
        await asyncio.sleep(service.ready_poll_interval_s)

    raise RuntimeError(f'service {service.name!r} failed readiness check after {attempts} attempt(s): {last_err}')


async def setup_services(
    services: list[AidataService],
    *,
    case_dir: Path | None = None,
    mock_env_vars: dict[str, str] | None = None,
) -> ServiceReplacements:
    """启动每个已声明的服务，并返回占位符替换。

    ``case_dir`` 仅在任何服务声明了 ``files`` 或 startup / ready_check 使用了
    ``script:`` 时才必需 —— 引擎会在执行之前，将这些宿主机文件从用例目录
    上传到服务容器中。

    ``mock_env_vars`` （可选）会注入到每个服务 startup 的 extra_env 中，
    使 sidecar mock-server 容器可以接收 LLM 路由环境变量
    （NEXUS_MOCK_LLM_*, NEXUS_MOCK_USER_PROMPT 等），而无需每个 case.yaml
    都显式列出它们。

    - 如果 ``services`` 为空则空操作； aidata 用例若没有辅助后端则零开销。
    - 严重失败则大声报错：如果任何 startup / ready_check 失败，整个任务
      会中止。 未启动的后端否则会在很久之后产生令人困惑的 0 分 rubric 结果。
    - 服务是顺序启动的（简单起见）；典型用例 ≤ 2 个，不值得为并行获取
      引入复杂度。
    """
    replacements = ServiceReplacements()
    if not services:
        return replacements

    ctx = require_context()
    if ctx.runtime_provider is None:
        raise RuntimeError('aidata services require a runtime_provider to be configured')

    names = [s.runtime_name for s in services]
    if 'main' in names:
        raise ValueError('service.runtime_name cannot be "main" (reserved for the agent container)')
    if len(set(names)) != len(names):
        raise ValueError(f'service runtime_name values must be unique; got {names!r}')

    # 如果任何服务需要，前置校验 case_dir 。
    needs_case_dir = any(
        s.files or (s.startup is not None and s.startup.script) or (s.ready_check is not None and s.ready_check.script)
        for s in services
    )
    if needs_case_dir and case_dir is None:
        raise RuntimeError(
            'at least one service declares `files:` or `script:` but setup_services was called '
            'without case_dir — the engine needs the case dir to upload host files.'
        )

    for service in services:
        # 镜像解析： case.yaml::services[].image > runtime_configs.<runtime_name>.image 。
        # 如果两者都未设置， provider 会在 acquire() 时抛出 —— 这是正确的失败模式
        # （aidata runtime 模板故意 **不** 在 provider 层面固定镜像，让每个 runtime
        # 自行选择）。
        resolved_image = service.image or ctx.runtime_configs.get(service.runtime_name, {}).get('image')
        logger.debug(
            '[aidata] starting service %s (runtime=%s, image=%s, ports=%s)',
            service.name,
            service.runtime_name,
            resolved_image or '(unresolved — will fail acquire)',
            service.exposed_ports,
        )
        ctx.runtime_provider.configure(service.runtime_name, **_runtime_config_for(service))
        runtime = await ctx.get_runtime(service.runtime_name)
        await _upload_service_files(runtime, service, case_dir=case_dir)

        if service.startup is not None:
            if mock_env_vars:
                merged_env = dict(mock_env_vars)
                merged_env.update(service.startup.env or {})
                service.startup = AidataServiceCommand(
                    command=service.startup.command,
                    script=service.startup.script,
                    cwd=service.startup.cwd,
                    env=merged_env,
                    timeout_s=service.startup.timeout_s,
                )
            rc, stdout, stderr = await _run_in_service(
                runtime,
                service.startup,
                case_dir=case_dir,
                service_name=service.name,
                phase='startup',
            )
            if rc not in (0, None):
                raise RuntimeError(
                    f'service {service.name!r} startup exited {rc}; '
                    f'stdout head: {stdout[-400:]!r}; stderr head: {stderr[-400:]!r}'
                )
            if stdout.strip():
                logger.debug('[aidata] %s startup stdout head: %s', service.name, stdout[:400])

        await _wait_ready(runtime, service, case_dir=case_dir)
        await _wait_for_connection_port_mappings(service, runtime, ctx.runtime_provider)

        text, env = _resolve_connection_values(service, runtime)
        dup_text = set(text) & set(replacements.text)
        dup_env = set(env) & set(replacements.env)
        if dup_text:
            raise ValueError(
                f'service {service.name!r}: connection placeholder collision with earlier service: {sorted(dup_text)}'
            )
        if dup_env:
            raise ValueError(
                f'service {service.name!r}: connection env_name collision with earlier service: {sorted(dup_env)}'
            )
        replacements.text.update(text)
        replacements.env.update(env)
        logger.debug(
            '[aidata] service %s ready; resolved placeholders: %s',
            service.name,
            sorted(text),
        )

    return replacements


def apply_text_replacements(value: Any, replacements: dict[str, str]) -> Any:
    """在类 JSON 树上递归 str.replace 。 非字符串直接透传。

    用于将 ``{__AIDATA_FOO_URL__: 'http://172.17.0.1:49155'}`` 拼接进
    ``extra_mcp_servers[*].client.{args,env}.*`` 和 ``user_prompt`` 中。
    """
    if not replacements:
        return value
    if isinstance(value, str):
        out = value
        for k, v in replacements.items():
            out = out.replace(k, v)
        return out
    if isinstance(value, list):
        return [apply_text_replacements(v, replacements) for v in value]
    if isinstance(value, dict):
        return {k: apply_text_replacements(v, replacements) for k, v in value.items()}
    return value


__all__ = [
    'ServiceReplacements',
    'apply_text_replacements',
    'setup_services',
]
