#!/usr/bin/env python3
"""供 aidata agents 使用的通用 shell MCP 服务器。

上游 ``cli-mcp-server`` 的一个精简本地替代品。 设计取舍
（为什么不直接沿用 cli-mcp-server ）：

* 命令字符串会 **原样** 传给 ``bash -c``。 没有正则
  切分、 没有算子重解析 —— 因此在引号内嵌算子的 shell
  结构（``python3 -c "a; b"``、 ``echo "a | b"``、 here-doc ）
  都能按人类的直觉工作。 cli-mcp-server 用不遵守引号
  的正则来对 ``;|&`` 切分； 我们通过从不试图理解命令
  字符串来避开这一整类 bug 。
* 该工具永远会执行 —— 不存在命令 / flag 白名单。 aidata
  之前的 cli-mcp-server 配置本来就设置了
  ``ALLOWED_COMMANDS=all`` / ``ALLOWED_FLAGS=all``， 因此没有
  功能回退。 风险提示通过响应中一个非阻断的
  ``warnings`` 字段来暴露， 参考自上游 Claude Code
  ``BashTool`` 的校验流水线（``destructiveCommandWarning`` /
  ``pathValidation``）。 想进一步加固的 case 可以对 trajectory
  做后处理。
* 输出（stdout + stderr ）在 16384 字符处以标记方式截断 —— 与
  上游 BashTool 的预算一致， 让 agent 不小心跑
  ``cat <huge binary>`` 时 trajectory 不至于爆炸。
* 超时会杀掉进程， 并把响应中的 ``interrupted=true`` 翻起，
  这样 agent 可以区分 “跑了但失败” 与 “被杀掉” 。

Env vars ：

* ``ALLOWED_DIR``       —— 每个命令的 cwd （默认为当前 cwd ）。
* ``COMMAND_TIMEOUT``   —— 每次调用的默认超时秒数（默认 180 ）。
                          按调用传入的 ``timeout`` 参数会覆盖它。

并发注意 —— LibreOffice headless （``soffice``）在共享同一个
用户 profile 目录（``~/.config/libreoffice/``）时并发调用是不
安全的。 aidata harness 依赖 ``agent_config`` 中的
``parallel_tool_calls: false`` 来串行化工具调用。 如果之后启用
了并行执行， LibreOffice 命令必须通过
``-env:UserInstallation=file:///tmp/lo-profile-<uuid>``
使用 per-call 的 profile 目录。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import traceback

import mcp.server.stdio

from mcp.server.lowlevel import Server
from mcp.types import TextContent, Tool


ALLOWED_DIR = os.environ.get('ALLOWED_DIR') or os.getcwd()
DEFAULT_TIMEOUT = int(os.environ.get('COMMAND_TIMEOUT', '180'))
# stdout/stderr 的截断预算。 计的是 字符 数（非字节数） —— 见
# ``_truncate`` 的说明。 当输出为纯 ASCII 时与上游 Claude Code
# BashTool 的 16 KiB 预算一致； 多字节 UTF-8 流可能会让 JSON body
# 略大于字面字节数所暗示的大小。
MAX_OUTPUT_CHARS = 16 * 1024

# --- LD_PRELOAD confine 集成 ----------------------------------------
#
# 当 engine 把 confine kit （engine/confine/）注入到沙箱中，
# 并且 /workspace/.aidata/confine/libconfine.so 编译成功时，
# 我们 spawn 出的每一个 bash 子进程都会继承指向该 .so 的 LD_PRELOAD
# ， 加上列出待阻断前缀的 AIDATA_CONFINE_DENY 。 该 .so 在 libc
# 内做真正的 EACCES ； 本文件只负责穿针引线 envp 。
#
# 通过 AIDATA_CONFINE_MODE 提供三种模式（在 mcp_catalog.py 中设置）：
#
#   enforce  默认 —— 若 .so 缺失则拒绝运行（exit 99 ）。
#                       堵住 “kit 缺失就悄悄卸下防护” 这个 footgun 。
#   warn     无 confine 运行， 但在 stderr 打印告警。 在 kit
#                       尚未接通的开发 / 镜像启用期比较有用。
#   off      完全跳过检查。 用于 tests/test_sandbox_confinement.py
#                       的 baseline 运行， 展示未缓解的攻击面。
CONFINE_SO = os.environ.get('AIDATA_CONFINE_SO', '')
CONFINE_DENY = os.environ.get('AIDATA_CONFINE_DENY', '')
CONFINE_MODE = (os.environ.get('AIDATA_CONFINE_MODE', 'enforce') or 'enforce').strip().lower()
# 逗号分隔的主机列表， 用于强制 DIRECT （绕过沙箱代理）， 让
# libconfine 的 getaddrinfo() 阻断真的生效（走代理的请求会解析代理
# 主机而非被拒主机， 于是 hook 根本看不到）。 与 libconfine 的
# getaddrinfo() 阻断使用同一份列表； 这里我们只处理 proxy/no_proxy 那半。
# 见 engine/paths.py::CONFINE_NET_DENY_DEFAULT 与 engine/confine/README.md 。
CONFINE_NET_DENY = os.environ.get('AIDATA_CONFINE_NET_DENY', '')

server = Server('shell')


# ---------------------------------------------------------------------------
# 校验流水线（claw-code bash_validation.rs 的简化 port ）
#
# 我们只发出告警 —— 从不阻断 —— 以保留 agent + task 框架的完全
# 自由度。 模式覆盖 “假阳成本低（agent 读一条提示）而假阴代价高
# （不小心 rm -rf 沙箱、 fork 炸弹、 裸盘写入）” 的场景。
# ---------------------------------------------------------------------------


_DESTRUCTIVE_PATTERNS: list[tuple[str, str]] = [
    ('rm -rf /', 'Recursive forced deletion at root — will destroy the sandbox'),
    ('rm -rf ~', 'Recursive forced deletion of home directory'),
    ('rm -rf *', 'Recursive forced deletion of all files in current directory'),
    ('rm -rf .', 'Recursive forced deletion of current directory'),
    ('mkfs', 'Filesystem creation will destroy existing data'),
    ('dd if=', 'Direct disk write — can overwrite partitions/devices'),
    ('> /dev/sd', 'Writing to a raw disk device'),
    ('chmod -R 777', 'Recursively setting world-writable permissions'),
    ('chmod -R 000', 'Recursively removing all permissions'),
    (':(){ :|:& };:', 'Fork bomb — will exhaust process table'),
]

# 无论参数如何都被视作破坏性的命令。
_ALWAYS_DESTRUCTIVE: set[str] = {'shred', 'wipefs'}

_ENV_ASSIGN_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*=')


def _extract_first_command(command: str) -> str:
    """返回第一个真正的命令 token ， 跳过 ``KEY=val`` 环境变量前缀。

    使用 ``shlex.split``（posix 模式）以保持带引号的值不被拆开。 若
    命令的引号不匹配， 则回退到按空白拆分 —— 我们绝不希望
    warning 提取抛异常从而遮掩真实错误。
    """
    try:
        tokens = shlex.split(command, comments=False, posix=True)
    except ValueError:
        tokens = command.split()

    for tok in tokens:
        if _ENV_ASSIGN_RE.match(tok):
            continue
        return tok
    return ''


def _collect_warnings(command: str) -> list[str]:
    """扫描命令中的风险模式。 从不抛异常， 也从不阻断。"""
    out: list[str] = []

    for needle, msg in _DESTRUCTIVE_PATTERNS:
        if needle in command:
            out.append(f'destructive: {msg}')

    first = _extract_first_command(command)
    if first in _ALWAYS_DESTRUCTIVE:
        out.append(f"destructive: command '{first}' is inherently destructive")

    # 兜底那些未被上表列出的其他 rm -rf 变体。
    if (
        ' rm ' in f' {command} '
        and '-r' in command
        and '-f' in command
        and not any(w.startswith('destructive: Recursive forced deletion') for w in out)
    ):
        out.append('destructive: rm -rf detected — verify the target path')

    if '../' in command:
        out.append("path: directory traversal '../' detected — verify target resolves inside ALLOWED_DIR")

    return out


# ---------------------------------------------------------------------------
# 输出截断
# ---------------------------------------------------------------------------


def _truncate(s: str) -> str:
    """将解码后的文本流夹到 ``MAX_OUTPUT_CHARS`` 个字符以内。

    使用 head+tail 策略： 需要截断时保留预算的前 25 % 与
    后 75 % 。 这既保留了错误信息（通常在 stderr 尾部），
    也仍展示输出的开头以提供上下文。 LibreOffice 之类的
    程序会在真正的错误之前发上百行 fontconfig / Java
    warning —— 老的只裁头策略会丢弃可行动的部分。
    """
    if len(s) <= MAX_OUTPUT_CHARS:
        return s
    head_budget = MAX_OUTPUT_CHARS // 4
    marker = f'\n\n[output truncated — {len(s)} total chars, showing first {head_budget} + last part]\n\n'
    tail_budget = MAX_OUTPUT_CHARS - head_budget - len(marker)
    if tail_budget < 0:
        tail_budget = 0
    return s[:head_budget] + marker + s[-tail_budget:] if tail_budget else s[:MAX_OUTPUT_CHARS]


# ---------------------------------------------------------------------------
# bash -c 执行器
# ---------------------------------------------------------------------------


def _coerce_stream(raw: object) -> str:
    """规范化 subprocess 的 stdout/stderr （可能是 str/bytes/None ）。"""
    if raw is None:
        return ''
    if isinstance(raw, bytes):
        return raw.decode('utf-8', 'replace')
    return str(raw)


def _confine_envp() -> tuple[dict[str, str], str | None]:
    """解析要传给 bash 子进程的环境变量 + 一个可选的
    拒绝消息。

    返回 ``(env_overlay, refusal)`` 。 当 ``refusal`` 非 None 时，
    调用方应短路， 在 stderr 呈现该消息而不是 spawn bash ——
    这就是 CONFINE_MODE=enforce 的 fail-CLOSED 分支。

    ``env_overlay`` 在 .so 可加载时携带 LD_PRELOAD +
    AIDATA_CONFINE_DENY ； 否则为空 dict （warn / off / .so 在
    warn 下缺失）。 调用方会把它合并到 ``os.environ`` 之上，
    因此不相关的环境变量（PATH 、 locale 、 …）会原样透出。
    """
    if CONFINE_MODE == 'off':
        return {}, None

    if CONFINE_SO and os.path.isfile(CONFINE_SO) and os.access(CONFINE_SO, os.R_OK):
        return {
            'LD_PRELOAD': CONFINE_SO,
            'AIDATA_CONFINE_DENY': CONFINE_DENY,
        }, None

    msg = (
        f'[confine] {CONFINE_SO!r} missing or unreadable '
        f'(mode={CONFINE_MODE}). Engine should have run mock_servers/confine/compile.sh '
        f'during task setup; check task.py logs for "[aidata] confine kit setup failed".'
    )
    if CONFINE_MODE == 'enforce':
        return {}, msg
    sys.stderr.write(f'[shell] WARN: {msg}; running unconfined\n')
    sys.stderr.flush()
    return {}, None


def _apply_net_deny_no_proxy(env: dict) -> None:
    """强制 CONFINE_NET_DENY 主机绕过沙箱代理。

    否则一个裸的 ``curl skillhub.cn`` 会通过被注入的 http(s)_proxy
    走隧道， 该主机会在代理侧被解析， 于是 libconfine 的
    getaddrinfo() hook 根本看不到被拒主机。 把这些主机追加到
    ``no_proxy`` 会强制 DIRECT 连接， 使得 getaddrinfo() 阻断真的
    生效。

    就地 修改 ``env`` 。 与继承的 no_proxy 合并 —— 绝不覆盖 ——
    （所以 .woa.com / localhost 保持直连）， 并同时写两种拼写
    以适配不同工具。
    """
    if not CONFINE_NET_DENY:
        return
    deny = [h.strip() for h in CONFINE_NET_DENY.split(',') if h.strip()]
    if not deny:
        return
    existing = env.get('no_proxy') or env.get('NO_PROXY') or ''
    parts = [p.strip() for p in existing.split(',') if p.strip()]
    seen = {p.lower() for p in parts}
    for host in deny:
        if host.lower() not in seen:
            parts.append(host)
            seen.add(host.lower())
    merged = ','.join(parts)
    env['no_proxy'] = merged
    env['NO_PROXY'] = merged


def _blocking_run(command: str, timeout: int) -> dict:
    overlay, refusal = _confine_envp()
    if refusal:
        return {
            'stdout': '',
            'stderr': refusal,
            'exit_code': 99,
            'interrupted': False,
        }

    child_env = dict(os.environ)
    child_env.update(overlay)
    _apply_net_deny_no_proxy(child_env)

    proc = subprocess.Popen(
        ['bash', '-c', command],
        cwd=ALLOWED_DIR,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return {
            'stdout': _truncate(stdout or ''),
            'stderr': _truncate(stderr or ''),
            'exit_code': proc.returncode,
            'interrupted': False,
        }
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
        stdout = _coerce_stream(stdout)
        stderr = _coerce_stream(stderr)
        if stderr and not stderr.endswith('\n'):
            stderr += '\n'
        stderr += f'[killed: exceeded {timeout}s timeout]'
        return {
            'stdout': _truncate(stdout),
            'stderr': _truncate(stderr),
            'exit_code': None,
            'interrupted': True,
        }


async def _run_bash(command: str, timeout: int) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _blocking_run, command, timeout)


# ---------------------------------------------------------------------------
# MCP 管道
# ---------------------------------------------------------------------------


_RUN_COMMAND_DESC = (
    'Run a shell command via `bash -c`. The command string is handed to bash '
    'VERBATIM — standard shell syntax (pipes `|`, `&&`, `||`, `;`, quoting, '
    'here-docs, subshells, env-var expansion) all works the way you expect. '
    'Semicolons inside quoted strings do NOT break (unlike the previous '
    'cli-mcp-server wrapper).\n\n'
    f'Working directory is pinned to `{ALLOWED_DIR}`. Default timeout is '
    f'{DEFAULT_TIMEOUT}s; pass `timeout` (int seconds) to override.\n\n'
    'Returns a JSON object:\n'
    '  {\n'
    '    "stdout":      string,   // truncated at 16384 chars with a marker\n'
    '    "stderr":      string,   // same truncation rule\n'
    '    "exit_code":   int|null, // null when killed by timeout\n'
    '    "interrupted": bool,     // true iff the timeout fired\n'
    '    "warnings":    string[], // non-blocking risk hints; empty when clean\n'
    '  }\n\n'
    'Warnings surface patterns like `rm -rf`, fork bombs, `dd if=`, `mkfs`, '
    'world-writable chmod, and path traversal. They NEVER block execution — '
    'they are advisory only.'
)


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name='run_command',
            description=_RUN_COMMAND_DESC,
            inputSchema={
                'type': 'object',
                'properties': {
                    'command': {
                        'type': 'string',
                        'description': 'Shell command to execute (bash -c).',
                    },
                    'timeout': {
                        'type': 'integer',
                        'description': (
                            f'Max wall-clock seconds. Default {DEFAULT_TIMEOUT}. '
                            'On expiry the process is killed and interrupted=true.'
                        ),
                    },
                },
                'required': ['command'],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name != 'run_command':
            return [TextContent(type='text', text=json.dumps({'error': f'unknown tool {name!r}'}))]

        command = arguments.get('command')
        if not isinstance(command, str) or not command.strip():
            return [
                TextContent(
                    type='text',
                    text=json.dumps({
                        'error': 'command must be a non-empty string',
                    }),
                )
            ]

        timeout = arguments.get('timeout')
        if not isinstance(timeout, int) or timeout <= 0:
            timeout = DEFAULT_TIMEOUT

        warnings = _collect_warnings(command)
        result = await _run_bash(command, timeout)
        result['warnings'] = warnings
        return [TextContent(type='text', text=json.dumps(result, ensure_ascii=False))]

    except Exception as exc:
        return [
            TextContent(
                type='text',
                text=json.dumps({
                    'error': str(exc),
                    'trace': traceback.format_exc()[:500],
                }),
            )
        ]


async def main() -> None:
    os.makedirs(ALLOWED_DIR, exist_ok=True)
    sys.stderr.write(
        f'[shell] init: allowed_dir={ALLOWED_DIR}, default_timeout={DEFAULT_TIMEOUT}s, pid={os.getpid()}\n'
    )
    sys.stderr.flush()
    async with mcp.server.stdio.stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())


if __name__ == '__main__':
    asyncio.run(main())
