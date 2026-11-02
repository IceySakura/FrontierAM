"""Shared launch constants and Claude Code judge input factories.

Infer builders live in harness; hooks and skills live in claude_hooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nexus.extensions.agents.claude_code.agent import ClaudeCodeInput
from nexus.extensions.tasks.artifact import config
from nexus.extensions.tasks.artifact.agent_setup import claude_hooks


# ══════════════════════════════════════════════════════════════════════════════
# 常量 / 片段
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CLAUDE_CODE_VERSION = '2.1.117'
"""Shared default Claude Code version, pinned for the evaluation image."""


DEFAULT_JUDGE_WORKING_DIR = '/workspace'

JUDGE_ROOT = '/tmp/artifact_judge'
"""Keep judge outputs outside the artifact workspace to avoid contaminating evidence."""


def judge_cli_args(max_turns: int | None = None) -> list[str]:
    """Build judge CLI flags using call-time configuration.

    Emit verbose stream-json for diagnostics and apply the configured turn limit.
    The runtime adds permission-bypass flags.
    """
    turns = config.get().judge.max_turns if max_turns is None else max_turns
    return [
        '--print',
        '--output-format',
        'stream-json',
        '--verbose',
        '--allowedTools',
        'ExitPlanMode',
        '--max-turns',
        str(turns),
    ]


# ── 星代理(沙箱出站必须透传,否则 @playwright/mcp 握手挂 30s)──────────────────
_STAR_PROXY_URL = 'http://star-proxy.oa.com:3128'

# Tencent 内网 no_proxy —— 与开发机 ~/.bashrc 的 export 一致。
STAR_PROXY_NO_PROXY = (
    '.polaris,21.86.45.67,21.86.45.217,.woa.com,mirrors.cloud.tencent.com,'
    'tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,'
    'localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,'
    '.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,'
    '.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,'
    '.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,'
    '.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,'
    '.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,'
    '.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,'
    '.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,'
    '.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,'
    '.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,'
    '.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,'
    '.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,'
    '.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,'
    '.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,'
    '.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,'
    '.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,'
    'www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com'
)

STAR_PROXY_ENV: dict[str, str] = {
    'http_proxy': _STAR_PROXY_URL,
    'https_proxy': _STAR_PROXY_URL,
    'HTTP_PROXY': _STAR_PROXY_URL,
    'HTTPS_PROXY': _STAR_PROXY_URL,
    'ftp_proxy': _STAR_PROXY_URL,
    'no_proxy': STAR_PROXY_NO_PROXY,
    'NO_PROXY': STAR_PROXY_NO_PROXY,
}

# Shared CLI environment defaults; infer and judge configurations reuse them.
CLAUDE_CODE_COMMON_ENV: dict[str, str] = {
    'CLAUDE_CODE_ATTRIBUTION_HEADER': '0',
    'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
    'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS': '1',
    'CLAUDE_CODE_SKIP_BEDROCK_AUTH': '1',
    'ENABLE_TOOL_SEARCH': 'false',
}

# Pin the browser MCP package and Chromium version to the evaluation image.
PLAYWRIGHT_MCP_CONFIG_PATH = '/tmp/playwright-mcp-config.json'

# Headless Chromium flags for container isolation and software rendering.
_CHROMIUM_LAUNCH_ARGS: list[str] = [
    '--disable-dev-shm-usage',
    '--password-store=basic',
    '--use-mock-keychain',
    '--ozone-platform=headless',
    '--enable-unsafe-swiftshader',
    '--use-gl=angle',
    '--use-angle=swiftshader-webgl',
    '--mute-audio',
    '--hide-scrollbars',
    '--disable-extensions',
    '--disable-background-networking',
]

# Update the MCP version and browser path together with the evaluation image.
DEFAULT_PLAYWRIGHT_MCP: dict[str, Any] = {
    'name': 'playwright',
    'command': 'npx',
    'args': [
        '-y',
        '@playwright/mcp@0.0.68',
        '--headless',
        '--no-sandbox',
        '--executable-path=/root/.cache/ms-playwright/chromium-1208/chrome-linux64/chrome',
    ],
}


def build_playwright_mcp_config_pre_start() -> dict[str, Any]:
    """Build a pre-start script that writes the browser MCP configuration."""
    import json as _json

    # 注意: 局部名不能叫 config —— 会遮蔽本模块 import 的 config 模块。
    mcp_config = {'browser': {'launchOptions': {'args': _CHROMIUM_LAUNCH_ARGS}}}
    body = _json.dumps(mcp_config, indent=2)
    return {
        'interpreter': 'bash',
        'code': (
            f'set -e\n'
            f'mkdir -p $(dirname {PLAYWRIGHT_MCP_CONFIG_PATH})\n'
            f"cat > {PLAYWRIGHT_MCP_CONFIG_PATH} <<'PLAYWRIGHT_MCP_CFG_EOF'\n"
            f'{body}\n'
            f'PLAYWRIGHT_MCP_CFG_EOF\n'
        ),
        'fail_on_error': True,
    }


# ══════════════════════════════════════════════════════════════════════════════
# judge claude_input 工厂(judge config 的 default_factory + pipeline 直接用)
# ══════════════════════════════════════════════════════════════════════════════


def default_judge_claude_input(
    *,
    working_dir: str | None = None,
    timeout: float | None = None,
    max_turns: int | None = None,
    with_playwright: bool = True,
    with_skill: bool = False,
    extra_skill_dirs: list[Path] | None = None,
) -> ClaudeCodeInput:
    """Build a judge launch configuration with shared sandbox defaults.

    Read omitted budgets and working_dir from current config. Add browser MCP
    and skill-install scripts only when requested.
    """
    judge_cfg = config.get().judge
    working_dir = judge_cfg.working_dir if working_dir is None else working_dir
    timeout = judge_cfg.timeout if timeout is None else timeout

    pre_start_scripts: list[Any] = []
    if with_playwright:
        pre_start_scripts.append(build_playwright_mcp_config_pre_start())
    skill_dirs: list[Path] = []
    if with_skill and claude_hooks.PLAYWRIGHT_SKILL_DIR.is_dir():
        skill_dirs.append(claude_hooks.PLAYWRIGHT_SKILL_DIR)
    skill_dirs.extend(d for d in (extra_skill_dirs or []) if d.is_dir())
    if skill_dirs:
        pre_start_scripts.append(claude_hooks.build_skills_install_script(skill_dirs))

    return ClaudeCodeInput(
        # user_prompt judge 用不到(capabilities.agent dump 时 exclude,judge 自拼 task_prompt);
        # 空占位符满足 pydantic 必填校验。
        user_prompt='',
        working_dir=working_dir,
        timeout=timeout,
        # 显式钉版本(单一真相源),不靠 ClaudeCodeInput 类默认(那是 'latest',会随镜像 npm 漂移
        # 且与 infer 不一致)。infer 侧 generator 也引用同一个 DEFAULT_CLAUDE_CODE_VERSION。
        version=DEFAULT_CLAUDE_CODE_VERSION,
        # Preserve structured thinking and signatures; configure proxy conversion logging.
        conversion_log_level=judge_cfg.conversion_log_level,
        thinking_preset='bedrock-claude',
        extra_env={
            **STAR_PROXY_ENV,
            **CLAUDE_CODE_COMMON_ENV,
            **({'PLAYWRIGHT_MCP_CONFIG': PLAYWRIGHT_MCP_CONFIG_PATH} if with_playwright else {}),
        },
        extra_cli_args=judge_cli_args(max_turns),
        mcp_servers=[dict(DEFAULT_PLAYWRIGHT_MCP)] if with_playwright else [],
        pre_start_scripts=pre_start_scripts,
    )


def default_code_review_claude_input(
    *,
    working_dir: str | None = None,
    timeout: float | None = None,
    max_turns: int | None = None,
) -> ClaudeCodeInput:
    """Build a source-review configuration without browser MCP."""
    return default_judge_claude_input(
        working_dir=working_dir, timeout=timeout, max_turns=max_turns, with_playwright=False
    )


__all__ = [
    'DEFAULT_CLAUDE_CODE_VERSION',
    'STAR_PROXY_ENV',
    'STAR_PROXY_NO_PROXY',
    'CLAUDE_CODE_COMMON_ENV',
    'DEFAULT_PLAYWRIGHT_MCP',
    'DEFAULT_JUDGE_WORKING_DIR',
    'PLAYWRIGHT_MCP_CONFIG_PATH',
    'build_playwright_mcp_config_pre_start',
    'judge_cli_args',
    'default_judge_claude_input',
    'default_code_review_claude_input',
]
