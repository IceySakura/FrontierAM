"""claude_code opt-in 脚手架 —— hooks / plugins / skills 注入（默认全关，基本不碰）。

三块能力，都产出可塞进 claude_code ``agent_config`` 的片段:
  * hooks   —— AskUserQuestion(自动应答) + ExitPlanMode(自动 allow) PreToolUse hook
  * plugins —— 从本地 marketplace 离线装 claude plugin 进沙箱
  * skills  —— 把本地 skill 目录同步进沙箱 ``~/.claude/skills/``；``PLAYWRIGHT_SKILL_DIR``
    是 bundled ``playwright-cli`` skill 的单一定义（教 agent 用 pinned playwright MCP 工具）。

``config_file_patches`` 会 deep-merge 进 ``~/.claude/settings.json``；``pre_start_scripts``
在 claude 二进制启动前依次跑（见 ``nexus.extensions.agents.claude_code`` runtime_setup）。

日常要调的 claude_code 参数不在这里 —— 在 sibling ``config.py``。
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import subprocess

from pathlib import Path
from typing import Any

from nexus.extensions.tasks.artifact import config


# ══════════════════════════════════════════════════════════════════════════════
# Hooks — AskUserQuestion + ExitPlanMode PreToolUse + Stop
# ══════════════════════════════════════════════════════════════════════════════

# ExitPlanMode: 自动 allow。AskUserQuestion: deny + 经 python hook 自动应答。
_EXITPLAN_HOOK_CMD = (
    'echo \'{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow","updatedInput":{}}}\''
)

# AskUserQuestion hook: v1(永远选第一项) vs v2(调 in-sandbox anthropic proxy 让 LLM
# 选上下文相关答案,失败回退 v1)。选哪个看 config 的 claude_code.askuser_hook_version。
# 与 frontend/generator_midtrain 对齐。两个脚本都是独立 py 文件,base64 塞进 pre_start。
_ASKUSER_HOOK_FILENAMES = {'v1': 'askuser_hook.py', 'v2': 'askuser_hook_v2.py'}
_ASKUSER_HOOK_SCRIPT_PATH = '/tmp/_nexus_askuser_hook.py'
_ASKUSER_HOOK_CMD = f'python3 {_ASKUSER_HOOK_SCRIPT_PATH}'

# Stop hook: 经 in-sandbox anthropic proxy 模拟用户回复,让 brainstorm 类 skill 在非交互
# batch 里能过"等用户确认"的点继续。由 config 的 claude_code.stop_hook 决定是否注入 +
# 是否透传 NEXUS_STOP_HOOK_ENABLE(脚本内部还会再判一次,两者现在由同一个开关驱动)。
# persona 见 claude_code.stop_hook_persona(nudger/iterative_user)。
# 注: frontend 的 sub_cc_user persona 走独立 user_agent/ 包(多文件,artifact 不搬),
# 这里只搬自包含的 stop_hook.py(nudger/iterative)。
_STOP_HOOK_SCRIPT_PATH = '/tmp/_nexus_stop_hook.py'
_STOP_HOOK_CMD = f'python3 {_STOP_HOOK_SCRIPT_PATH}'


def _read_script(filename: str) -> str:
    """Read a sibling hook script. Called at build time, not import time, so config edits
    (e.g. switching askuser v2 → v1) take effect without reimporting the module."""
    return (Path(__file__).parent / filename).read_text(encoding='utf-8')


def _askuser_script() -> str:
    version = config.get().claude_code.askuser_hook_version.strip().lower()
    return _read_script(_ASKUSER_HOOK_FILENAMES.get(version, _ASKUSER_HOOK_FILENAMES['v2']))


def build_hook_env() -> dict[str, str]:
    """``NEXUS_*`` env for the in-container hook scripts, per the config switches.

    Only non-empty values are emitted: the scripts already carry their own defaults, so an
    unset knob stays unset rather than being restated here. ``NEXUS_STOP_HOOK_ENABLE`` is
    derived from ``stop_hook`` — the script's second gate and the injection decision are the
    same switch now, which is what made the hook unreachable before.
    """
    cc = config.get().claude_code
    env: dict[str, str] = {}
    if cc.askuser_hook_disable_llm:
        env['NEXUS_ASKUSER_HOOK_DISABLE_LLM'] = '1'
    if cc.askuser_hook_model:
        env['NEXUS_ASKUSER_HOOK_MODEL'] = cc.askuser_hook_model
    if cc.stop_hook:
        env['NEXUS_STOP_HOOK_ENABLE'] = 'true'
        if cc.stop_hook_persona:
            env['NEXUS_STOP_HOOK_PERSONA'] = cc.stop_hook_persona
        if cc.stop_hook_model:
            env['NEXUS_STOP_HOOK_MODEL'] = cc.stop_hook_model
        if cc.stop_hook_max_replies:
            env['NEXUS_STOP_HOOK_MAX_REPLIES'] = str(cc.stop_hook_max_replies)
        if cc.stop_hook_iterative_turns:
            env['NEXUS_STOP_HOOK_ITERATIVE_TURNS'] = str(cc.stop_hook_iterative_turns)
        if cc.stop_hook_show_code:
            env['NEXUS_STOP_HOOK_SHOW_CODE'] = 'true'
    return env


def build_hook_patches(*, with_stop_hook: bool | None = None) -> list[dict[str, Any]]:
    """返回 wiring PreToolUse(+ 可选 Stop) hook 的 ``config_file_patches`` 条目。

    deep-merge 进 settings.json。需配合 :func:`build_askuser_hook_writer` 放进
    ``pre_start_scripts``,hook 脚本才在容器里存在。``with_stop_hook`` 默认跟随 config 的
    ``claude_code.stop_hook``;显式传值只用于测试。
    """
    if with_stop_hook is None:
        with_stop_hook = config.get().claude_code.stop_hook
    hooks: dict[str, Any] = {
        'PreToolUse': [
            {'matcher': 'ExitPlanMode', 'hooks': [{'type': 'command', 'command': _EXITPLAN_HOOK_CMD}]},
            {'matcher': 'AskUserQuestion', 'hooks': [{'type': 'command', 'command': _ASKUSER_HOOK_CMD}]},
        ],
    }
    if with_stop_hook:
        # Stop 事件无 matcher;Claude 每次准备结束回合时触发。
        hooks['Stop'] = [{'hooks': [{'type': 'command', 'command': _STOP_HOOK_CMD}]}]
    return [{'path': '~/.claude/settings.json', 'content': {'hooks': hooks}}]


def build_askuser_hook_writer(*, with_stop_hook: bool | None = None) -> dict[str, Any]:
    """返回一个把 AskUserQuestion(+ 可选 Stop) hook 脚本写进容器的 ``pre_start_scripts`` 条目。

    ``with_stop_hook`` 默认跟随 config 的 ``claude_code.stop_hook``;显式传值只用于测试。
    """
    if with_stop_hook is None:
        with_stop_hook = config.get().claude_code.stop_hook
    askuser_b64 = base64.b64encode(_askuser_script().encode()).decode()
    code = (
        'import base64, pathlib\n'
        f'pathlib.Path({_ASKUSER_HOOK_SCRIPT_PATH!r}).write_text('
        f'base64.b64decode({askuser_b64!r}).decode(), encoding="utf-8")\n'
    )
    if with_stop_hook:
        stop_b64 = base64.b64encode(_read_script('stop_hook.py').encode()).decode()
        code += (
            f'pathlib.Path({_STOP_HOOK_SCRIPT_PATH!r}).write_text('
            f'base64.b64decode({stop_b64!r}).decode(), encoding="utf-8")\n'
        )
    return {'interpreter': 'python', 'code': code}


# ══════════════════════════════════════════════════════════════════════════════
# Plugins — 离线装本地 marketplace plugin
# ══════════════════════════════════════════════════════════════════════════════

_MARKETPLACE_NAME = 'claude-plugins-official'
_MARKETPLACE_ROOT = Path(os.path.expanduser('~/.claude/plugins/marketplaces')) / _MARKETPLACE_NAME
_INSTALLED_PLUGINS_JSON = Path(os.path.expanduser('~/.claude/plugins/installed_plugins.json'))
_PLUGIN_CACHE_ROOT = Path(os.path.expanduser('~/.claude/plugins/cache')) / _MARKETPLACE_NAME


def _load_installed_plugin_entry(plugin_name: str) -> dict[str, Any] | None:
    """从 installed_plugins.json 读某 plugin 的安装元数据(marketplace 目录不存在时用)。"""
    if not _INSTALLED_PLUGINS_JSON.exists():
        return None
    try:
        installed = json.loads(_INSTALLED_PLUGINS_JSON.read_text('utf-8'))
    except Exception:
        return None
    entries = installed.get('plugins', {}).get(f'{plugin_name}@{_MARKETPLACE_NAME}', [])
    if not isinstance(entries, list) or not entries:
        return None
    entry = entries[-1]
    return entry if isinstance(entry, dict) else None


def _resolve_local_plugin_dir(plugin_name: str) -> tuple[Path, dict[str, Any] | None]:
    """三级 fallback 解析 plugin 目录(与 frontend 对齐):
    marketplace/plugins/<name> → installed_plugins.json 的 installPath → cache/<name>/<version>。
    支持 `claude plugin install` 装到 cache 的 plugin(非 marketplace clone)。
    """
    plugin_dir = _MARKETPLACE_ROOT / 'plugins' / plugin_name
    if plugin_dir.is_dir():
        return plugin_dir, None

    installed_entry = _load_installed_plugin_entry(plugin_name)
    install_path = str((installed_entry or {}).get('installPath') or '')
    if install_path:
        cached = Path(os.path.expanduser(install_path)).resolve()
        if cached.is_dir():
            return cached, installed_entry

    cache_dir = _PLUGIN_CACHE_ROOT / plugin_name
    if cache_dir.is_dir():
        for version_dir in sorted(cache_dir.iterdir(), reverse=True):
            if version_dir.is_dir():
                return version_dir, installed_entry

    raise FileNotFoundError(
        f'Plugin "{plugin_name}" not found at {plugin_dir} or in {_INSTALLED_PLUGINS_JSON}. '
        f'Run: claude plugin marketplace add anthropics/{_MARKETPLACE_NAME} && '
        f'claude plugin install {plugin_name}'
    )


def _load_local_plugin(plugin_name: str) -> dict[str, Any]:
    plugin_dir, installed_entry = _resolve_local_plugin_dir(plugin_name)

    files: dict[str, str] = {}
    for fp in sorted(plugin_dir.rglob('*')):
        if any(part in {'.git', '.in_use'} for part in fp.relative_to(plugin_dir).parts):
            continue
        if fp.is_file():
            with contextlib.suppress(UnicodeDecodeError):
                files[str(fp.relative_to(plugin_dir))] = fp.read_text('utf-8')

    if '.claude-plugin/plugin.json' not in files:
        raise FileNotFoundError(f'Plugin "{plugin_name}" missing .claude-plugin/plugin.json')

    plugin_json = json.loads(files['.claude-plugin/plugin.json'])
    version = str((installed_entry or {}).get('version') or plugin_json.get('version') or 'unknown')
    git_sha = str((installed_entry or {}).get('gitCommitSha') or '')

    git_dir = _MARKETPLACE_ROOT / '.git'
    if not git_sha and git_dir.is_dir():
        r = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=_MARKETPLACE_ROOT, capture_output=True, text=True)
        git_sha = r.stdout.strip() if r.returncode == 0 else 'unknown'
        version = git_sha[:12] if git_sha != 'unknown' else version
    if not git_sha:
        git_sha = 'unknown'

    return {'files': files, 'version': version, 'gitCommitSha': git_sha}


def _load_marketplace_json(plugin_data: dict[str, dict[str, Any]] | None = None) -> str:
    """读 marketplace.json:先从本地 marketplace clone,不存在时从缓存的 plugin files 里捞。"""
    mp_json = _MARKETPLACE_ROOT / '.claude-plugin' / 'marketplace.json'
    if not mp_json.exists():
        for data in (plugin_data or {}).values():
            content = data.get('files', {}).get('.claude-plugin/marketplace.json')
            if content:
                return str(content)
        raise FileNotFoundError(
            f'marketplace.json not found at {mp_json}. '
            f'Run: claude plugin marketplace add anthropics/{_MARKETPLACE_NAME}'
        )
    return mp_json.read_text('utf-8')


def build_plugin_install_script(plugins: list[str]) -> dict[str, Any]:
    """返回一个在沙箱里离线装 *plugins* 的 ``pre_start_scripts`` 条目。

    从 host 本地 marketplace clone 读 plugin 文件，emit python 代码写进容器的
    ``~/.claude/plugins/`` 加上 ``enabledPlugins`` settings.json 条目。
    """
    if (_MARKETPLACE_ROOT / '.git').is_dir():
        with contextlib.suppress(Exception):
            subprocess.run(['git', 'pull', '--ff-only'], cwd=_MARKETPLACE_ROOT, capture_output=True, timeout=30)

    plugin_data = {}
    for name in plugins:
        plugin_data[f'{name}@{_MARKETPLACE_NAME}'] = _load_local_plugin(name)

    marketplace_json_content = _load_marketplace_json(plugin_data)

    lines = [
        'import os, json',
        'home = os.environ.get("HOME", "/root")',
        'plugins_dir = os.path.join(home, ".claude", "plugins")',
        'os.makedirs(plugins_dir, exist_ok=True)',
        '',
        'installed = {"version": 2, "plugins": {}}',
        'marketplace_meta = {',
        f'    {_MARKETPLACE_NAME!r}: {{',
        f'        "source": {{"source": "github", "repo": "anthropics/{_MARKETPLACE_NAME}"}},',
        f'        "installLocation": os.path.join(plugins_dir, "marketplaces", {_MARKETPLACE_NAME!r}),',
        '        "lastUpdated": "2026-04-23T16:06:43.701Z",',
        '    }',
        '}',
    ]

    for key, data in plugin_data.items():
        version = data['version']
        git_sha = data['gitCommitSha']
        marketplace = key.split('@')[-1]
        pname = key.split('@')[0]

        lines.append('')
        lines.append(f'cache_dir = os.path.join(plugins_dir, "cache", {marketplace!r}, {pname!r}, {version!r})')

        for rel_path, content in data['files'].items():
            lines.append(f'fp = os.path.join(cache_dir, {rel_path!r})')
            lines.append('os.makedirs(os.path.dirname(fp), exist_ok=True)')
            lines.append(f'open(fp, "w").write({json.dumps(content)})')

        lines.append(f'installed["plugins"][{key!r}] = [{{')
        lines.append('    "scope": "user",')
        lines.append('    "installPath": cache_dir,')
        lines.append(f'    "version": {version!r},')
        lines.append('    "installedAt": "2026-04-23T16:06:43.987Z",')
        lines.append('    "lastUpdated": "2026-04-23T16:06:43.987Z",')
        lines.append(f'    "gitCommitSha": {git_sha!r},')
        lines.append('}]')

    enabled = dict.fromkeys(plugin_data, True)

    lines.append('')
    lines.append(f'mp_dir = os.path.join(plugins_dir, "marketplaces", {_MARKETPLACE_NAME!r}, ".claude-plugin")')
    lines.append('os.makedirs(mp_dir, exist_ok=True)')
    lines.append(f'open(os.path.join(mp_dir, "marketplace.json"), "w").write({json.dumps(marketplace_json_content)})')
    lines.append('')
    lines.append(
        'open(os.path.join(plugins_dir, "installed_plugins.json"), "w").write(json.dumps(installed, indent=2))'
    )
    lines.append(
        'open(os.path.join(plugins_dir, "known_marketplaces.json"), "w").write(json.dumps(marketplace_meta, indent=2))'
    )
    lines.append('')
    lines.append('settings_path = os.path.join(home, ".claude", "settings.json")')
    lines.append('settings = {}')
    lines.append('if os.path.exists(settings_path):')
    lines.append('    try: settings = json.loads(open(settings_path).read())')
    lines.append('    except: pass')
    lines.append(f'settings["enabledPlugins"] = {{**settings.get("enabledPlugins", {{}}), **{enabled!r}}}')
    lines.append(
        f'settings.setdefault("extraKnownMarketplaces", {{}})[{_MARKETPLACE_NAME!r}] = '
        f'{{"source": {{"source": "github", "repo": "anthropics/{_MARKETPLACE_NAME}"}}}}'
    )
    lines.append('open(settings_path, "w").write(json.dumps(settings, indent=2))')

    return {'interpreter': 'python', 'code': '\n'.join(lines)}


# ══════════════════════════════════════════════════════════════════════════════
# Skills — 同步本地 skill 目录进沙箱
# ══════════════════════════════════════════════════════════════════════════════

# Bundled skill: nexus/extensions/tasks/artifact/skills/playwright-cli。
# parents[1] = artifact/ 包根（本文件在 artifact/agent_setup/）。
PLAYWRIGHT_SKILL_DIR = (Path(__file__).resolve().parents[1] / 'skills' / 'playwright-cli').resolve()

# Bundled skill: web_judger 的 flow2code 用，教 agent 按 harness 真实 API 写采集脚本。
WEBJUDGER_COLLECT_SKILL_DIR = (Path(__file__).resolve().parents[1] / 'skills' / 'webjudger-collect').resolve()


def parse_skills_dirs(raw: str) -> list[Path]:
    """把逗号分隔的 skill-dir 列表解析成校验过的 Path 列表。"""
    if not raw:
        return []
    dirs = []
    for p in raw.split(','):
        p = p.strip()
        if not p:
            continue
        path = Path(p).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f'skills dir not found: {path}')
        dirs.append(path)
    return dirs


def build_skills_install_script(skills_dirs: list[Path]) -> dict[str, Any]:
    """返回一个把 skill 文件写进 ``~/.claude/skills/`` 的 ``pre_start_scripts`` 条目。"""
    lines = [
        'import os, base64',
        'home = os.environ.get("HOME", "/root")',
        'skills_root = os.path.join(home, ".claude", "skills")',
        'os.makedirs(skills_root, exist_ok=True)',
    ]

    for skill_dir in skills_dirs:
        skill_name = skill_dir.name
        for fp in sorted(skill_dir.rglob('*')):
            if fp.is_file():
                try:
                    content_bytes = fp.read_bytes()
                except Exception:
                    continue
                rel_path = str(fp.relative_to(skill_dir))
                encoded = base64.b64encode(content_bytes).decode()
                target = f'os.path.join(skills_root, {skill_name!r}, {rel_path!r})'
                lines.append(f'os.makedirs(os.path.dirname({target}), exist_ok=True)')
                lines.append(f'open({target}, "wb").write(base64.b64decode({encoded!r}))')

        lines.append(f'print("Skill installed: {skill_name}")')

    return {'interpreter': 'python', 'code': '\n'.join(lines)}


__all__ = [
    # hooks
    'build_hook_patches',
    'build_askuser_hook_writer',
    # plugins
    'build_plugin_install_script',
    # skills
    'PLAYWRIGHT_SKILL_DIR',
    'WEBJUDGER_COLLECT_SKILL_DIR',
    'parse_skills_dirs',
    'build_skills_install_script',
]
