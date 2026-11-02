"""Tools for SWE Setup Agent (In-Place Mode).

Tools: bash, str_replace_editor, switch_runtime_image, submit_eval.
After submit_eval succeeds, the agent exits and the task is complete.

Note: switch_git_state is an internal function used by _validate_fail2pass inside submit_eval.
It is NOT exposed to the agent as a callable tool.
"""

import asyncio
import json
import logging
import os
import re
import time

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Literal

from openai.types.chat import ChatCompletionContentPartParam
from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, require_context, tool_registry
from nexus.extensions.tools.file_editor import FileEditor, ToolError
from nexus.runtime.executors import (
    CapturePaneRequest,
    CreateTerminalSessionRequest,
    RunCommandInSessionRequest,
    SendKeysToSessionRequest,
)
from nexus.runtime.executors.terminal import (
    capture_pane,
    create_terminal_session,
    run_command_in_session,
    send_keys_to_session,
)


logger = logging.getLogger(__name__)

# ============== Constants ==============

MAX_RESPONSE_LEN = 16000
TRUNCATED_MESSAGE = (
    '<response clipped><NOTE>To save on context only part of this file has been shown to you. '
    'You should retry this tool after you have searched inside the file with `grep -n` in order to find '
    'the line numbers of what you are looking for.</NOTE>'
)


# ============== Utility Functions ==============


_MAX_OUTPUT_CHARS = 16_000  # ~4K tokens; prevents single tool output from flooding context window


def _resolve_mirror_image(new_image: str, current_base_image: str) -> str:
    """Translate a short image name to its mirror registry URL if needed.

    The Gongfeng sandbox can only pull images from the internal mirror registry.
    The initial base_image is always the full mirror URL, e.g.:
      mirrors.tencent.com/hunyuan_yanguan/t3build_repolaunch_midtrain
        .swe_builder_eval_base_ubuntu_git_22_04:latest

    This function applies the same naming convention as the generator's
    _mirror_image():  short_name → safe_name → registry.swe_builder_eval_base_safe:latest

    If new_image is already a full URL (contains a dot-separated hostname prefix),
    it is returned unchanged.
    """
    # Already a fully-qualified image URL — pass through as-is
    parts = new_image.split('/')
    if len(parts) >= 2 and '.' in parts[0]:
        return new_image

    # Derive mirror registry prefix from the current base_image, e.g.
    #   mirrors.tencent.com/hunyuan_yanguan/t3build_repolaunch_midtrain
    # (everything before the first ".swe_builder_eval_base_" token)
    _MARKER = '.swe_builder_eval_base_'
    if _MARKER in current_base_image:
        registry_prefix = current_base_image[: current_base_image.index(_MARKER)]
    else:
        # Fallback: hardcoded default (matches generator constant)
        registry_prefix = 'mirrors.tencent.com/hunyuan_yanguan/t3build_repolaunch_midtrain'

    safe = new_image.lower()
    for ch in ('-', ':', '/', '.'):
        safe = safe.replace(ch, '_')
    resolved = f'{registry_prefix}{_MARKER}{safe}:latest'
    logger.debug(f'[SWITCH_RUNTIME] Translated image: {new_image!r} → {resolved!r}')
    return resolved


def _make_output(text: str) -> list[ChatCompletionContentPartParam | dict]:
    """Helper to create standard output format. Truncates very large outputs."""
    if len(text) > _MAX_OUTPUT_CHARS:
        head = text[:10_000]
        tail = text[-6_000:]
        omitted = len(text) - 10_000 - 6_000
        text = f'{head}\n\n... [OUTPUT TRUNCATED: {omitted:,} chars omitted] ...\n\n{tail}'
    return [{'type': 'text', 'text': text}]


def get_working_dir(context: ToolContext) -> str:
    """Get current working directory from context."""
    return context.tool_state.get('working_dir', '/testbed')


def _apply_context_updates(context: ToolContext, updates: dict) -> ToolContext:
    """Return a new ToolContext with the given updates merged into tool_state."""
    if not updates:
        return context
    new_state = dict(context.tool_state)
    new_state.update(updates)
    return ToolContext(
        run_id=context.run_id,
        tool_configs=dict(context.tool_configs),
        tool_state=new_state,
        trace_id=context.trace_id,
    )


# ============== Mirror Setup ==============

_MIRROR_SETUP = r"""
# === Mirror configuration ===
export PIP_INDEX_URL=https://mirrors.tencent.com/pypi/simple/
export PIP_TRUSTED_HOST=mirrors.tencent.com
export NPM_CONFIG_REGISTRY=https://mirrors.tencent.com/npm/
export GOPROXY=https://goproxy.cn,direct
export GONOSUMCHECK=*
export GOFLAGS=-mod=mod

pip config set global.index-url https://mirrors.tencent.com/pypi/simple/ 2>/dev/null || true
pip3 config set global.index-url https://mirrors.tencent.com/pypi/simple/ 2>/dev/null || true
npm config set registry https://mirrors.tencent.com/npm/ 2>/dev/null || true
yarn config set registry https://mirrors.tencent.com/npm/ 2>/dev/null || true
pnpm config set registry https://mirrors.tencent.com/npm/ 2>/dev/null || true

# Cargo: rust-git uses CARGO_HOME=/usr/local/cargo; others use ~/.cargo
# rsproxy.cn git index returns 404; use sparse protocol (Rust >= 1.68) or USTC git (older Rust)
_cargo_cfg="${CARGO_HOME:-$HOME/.cargo}/config.toml"
mkdir -p "$(dirname $_cargo_cfg)" 2>/dev/null || true
_cargo_ver=$(cargo --version 2>/dev/null | grep -oP '\d+\.\d+' | head -1)
if [ -n "$_cargo_ver" ] && [ "$(echo "$_cargo_ver" | awk -F. '{print ($1*100+$2)}')" -ge 168 ]; then
  # Rust >= 1.68: sparse protocol supported (fastest, no git clone needed)
  cat > "$_cargo_cfg" << 'CARGOEOF'
[source.crates-io]
replace-with = "rsproxy-sparse"
[source.rsproxy-sparse]
registry = "sparse+https://rsproxy.cn/index/"
[net]
git-fetch-with-cli = true
CARGOEOF
else
  # Old Rust (< 1.68): use USTC git index (rsproxy git returns 404, TUNA git also broken)
  cat > "$_cargo_cfg" << 'CARGOEOF'
[source.crates-io]
replace-with = "ustc"
[source.ustc]
registry = "https://mirrors.ustc.edu.cn/crates.io-index"
[net]
git-fetch-with-cli = true
CARGOEOF
fi

# Maven: use Tencent nexus mirror (fastest from K8s, verified 1 MB/s)
# mirrorOf=central only — allows project-custom repos to work directly
mkdir -p ~/.m2 2>/dev/null
cat > ~/.m2/settings.xml << 'MVNEOF'
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0">
  <mirrors>
    <mirror>
      <id>tencent-public</id>
      <mirrorOf>central</mirrorOf>
      <url>https://mirrors.tencent.com/nexus/repository/maven-public/</url>
    </mirror>
  </mirrors>
</settings>
MVNEOF

# Gradle init.d: mirror for plugin/dependency resolution
mkdir -p ~/.gradle/init.d 2>/dev/null
cat > ~/.gradle/init.d/mirrors.gradle << 'GRADLEOF'
allprojects {
    buildscript {
        repositories {
            maven { url "https://maven.aliyun.com/repository/public" }
            maven { url "https://mirrors.tencent.com/nexus/repository/maven-public/" }
            mavenCentral()
        }
    }
    repositories {
        maven { url "https://maven.aliyun.com/repository/public" }
        maven { url "https://mirrors.tencent.com/nexus/repository/maven-public/" }
        mavenCentral()
    }
}
GRADLEOF

# gem/bundler mirror (Ruby)
gem sources --add https://mirrors.tencent.com/rubygems/ --remove https://rubygems.org/ 2>/dev/null || true
bundle config set mirror.https://rubygems.org https://mirrors.tencent.com/rubygems/ 2>/dev/null || true

# Composer mirror (PHP)
composer config -g repo.packagist composer https://mirrors.tencent.com/composer/ 2>/dev/null || true

# NuGet mirror (C#) — only write if dotnet is present
if command -v dotnet >/dev/null 2>&1; then
  mkdir -p /root/.nuget/NuGet 2>/dev/null || true
  cat > /root/.nuget/NuGet/NuGet.Config << 'NUGETEOF'
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" protocolVersion="3" />
  </packageSources>
</configuration>
NUGETEOF
fi

echo "[mirrors configured: pip=tencent, npm/yarn/pnpm=tencent, go=goproxy.cn, cargo=rsproxy-sparse/ustc, maven=tencent, gradle=aliyun+tencent, gem=tencent, composer=tencent, nuget=api.nuget.org]"

# Runtime apt safety net: if image still has aliyun apt source (blocked in k8s), switch to tencent
if grep -q 'mirrors.aliyun.com' /etc/apt/sources.list 2>/dev/null; then
    sed -i 's/mirrors.aliyun.com/mirrors.tencent.com/g' /etc/apt/sources.list
fi

# Refresh apt package lists only if empty (all images start with 0 files; ubuntu-git gets ~21 after update)
# Keep stderr visible so silent network failures don't cause mysterious "Unable to locate package" errors
if [ "$(ls /var/lib/apt/lists/ 2>/dev/null | wc -l)" -le 2 ]; then
    apt-get update -qq 2>&1 | tail -3 || true
fi
"""


# ============== Bash Output Compression ==============

_INSTALL_CMD_PATTERNS = [
    'apt-get install',
    'apt install',
    'pip install',
    'pip3 install',
    'npm install',
    'npm ci',
    'yarn install',
    'yarn add',
    'cargo build',
    'cargo install',
    'mvn install',
    'mvn compile',
    'mvn package',
    'mvn test',
    './gradlew',
    'gradle build',
    'bundle install',
    'gem install',
    'composer install',
    'composer update',
    'go mod download',
    'go get',
    '/tmp/install.sh',
]


def _is_install_command(cmd: str) -> bool:
    """Check if command matches known install/build patterns that produce verbose output."""
    cmd_lower = cmd.lower().strip()
    return any(pat in cmd_lower for pat in _INSTALL_CMD_PATTERNS)


def _compress_install_output(output: str, exit_code: int) -> str:
    """Compress install command output: keep errors/warnings + last 30 lines.

    Install commands (apt-get, pip, npm, etc.) produce huge logs where only
    error lines and the final status matter. This extracts the important parts.
    """
    lines = output.splitlines()
    if len(lines) <= 60:
        # Short enough — return as-is with standard truncation
        if len(output) > 6000:
            return output[:2500] + f'\n\n... [{len(output) - 5000} chars omitted] ...\n\n' + output[-2500:]
        return output

    # Extract important lines: errors, warnings, failures, install markers
    important = []
    _error_pats = re.compile(
        r'error|fail|cannot|unable|not found|no such|denied|fatal|exception'
        r'|E: |Err:|FAILED|Could not|Traceback|ImportError|ModuleNotFoundError'
        r'|already exists|conflict|unmet depend|broken',
        re.IGNORECASE,
    )
    _progress_pats = re.compile(
        r'^\s*(?:Fetching|Downloading|Unpacking|Setting up|Processing|Selecting|Preparing'
        r'|Reading |Building |Get:\d|Hit:\d|Ign:\d|\d+%|\s*\.+$|━|▓|█'
        r'|Already satisfied|Requirement already|Using cached'
        r'|added \d+ package|up to date|audited \d+ package)',
        re.IGNORECASE,
    )

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # Keep error/warning lines with context (1 line before)
        if _error_pats.search(stripped):
            if i > 0 and lines[i - 1].strip() and len(important) > 0 and important[-1] != lines[i - 1]:
                important.append(lines[i - 1])
            important.append(line)
        # Keep install.sh step markers (==[1/N]== etc.)
        elif stripped.startswith('==') or stripped.startswith('+ '):
            important.append(line)

    # Always keep last 30 lines (final status)
    tail_lines = lines[-30:]

    # Build compressed output
    parts = []
    if important:
        # Deduplicate and limit important lines
        seen = set()
        unique_important = []
        for l in important[:50]:
            if l not in seen:
                seen.add(l)
                unique_important.append(l)
        parts.append('\n'.join(unique_important))

    parts.append(f'\n... [{len(lines)} total lines, showing errors + last 30 lines] ...\n')
    parts.append('\n'.join(tail_lines))

    result = '\n'.join(parts)
    # Final size cap
    if len(result) > 6000:
        result = result[:2500] + f'\n\n... [{len(result) - 5000} chars omitted] ...\n\n' + result[-2500:]
    return result


def _compress_bash_output(cmd: str, output: str, exit_code: int) -> str:
    """Compress bash output. Install commands get aggressive filtering; others get head+tail truncation."""
    if _is_install_command(cmd) and len(output) > 3000:
        return _compress_install_output(output, exit_code)

    # General commands: keep first 4000 + last 4000 chars
    if len(output) > 8000:
        omitted = len(output) - 8000
        output = output[:4000] + f'\n\n... [{omitted} chars omitted] ...\n\n' + output[-4000:]
    return output


# ============== Auto Setup (Deterministic, pre-agent) ==============


@dataclass
class AutoSetupResult:
    """Structured result from auto_setup — injected into the user prompt before the agent loop."""

    # explore_repo results
    language: str = ''
    language_version: str = 'unknown'
    selected_image: str = ''
    image_switched: bool = False
    package_files: list[str] = field(default_factory=list)
    test_dirs: list[dict] = field(default_factory=list)
    build_systems: list[str] = field(default_factory=list)
    key_file_previews: dict = field(default_factory=dict)
    version_signals: list[str] = field(default_factory=list)
    ci_install_commands: list[str] = field(default_factory=list)
    top_dirs: list[str] = field(default_factory=list)

    # patch probing results
    golden_patch_status: str = 'empty'  # clean | crlf | hunk_failed | already_exists | empty
    golden_patch_detail: str = ''
    test_patch_status: str = 'empty'  # clean | crlf | hunk_failed | context_mismatch | empty
    test_patch_detail: str = ''
    test_patch_applied_on: str = 'none'  # base | golden | none
    eval_strategy: str = 'standard'  # standard | custom_test | crlf_fix

    # pre_install results
    install_commands_run: list[dict] = field(default_factory=list)
    install_summary: str = ''

    # overall
    setup_duration_s: float = 0.0
    warnings: list[str] = field(default_factory=list)


async def _explore_repo_internal(runtime) -> dict:
    """Run explore_repo logic on the runtime, return parsed JSON dict.

    This is the deterministic core extracted from the explore_repo tool.
    It runs a Python script inside the container that scans /testbed and returns
    structured JSON with: top_dirs, top_files, version_signals, package_files,
    file_previews, build_info, test_dirs, ci_install_commands.
    """
    explore_script = r"""python3 - << 'PYEOF'
import os, re, json

testbed = '/testbed'
result = {}

def read_file(path, limit=0):
    try:
        c = open(path).read()
        if limit and len(c) > limit:
            return c[:limit], len(c)
        return c, len(c)
    except:
        return None, 0

# 1. Top-level structure
try:
    items = os.listdir(testbed)
    dirs = [d for d in items if os.path.isdir(os.path.join(testbed, d)) and not d.startswith('.')]
    files = [f for f in items if os.path.isfile(os.path.join(testbed, f))]
    result['top_dirs'] = sorted(dirs)
    result['top_files'] = sorted(files)
except:
    result['top_dirs'] = []
    result['top_files'] = []

# 2. Version signals
signals = []

# Go
c, _ = read_file(f'{testbed}/go.mod')
if c:
    m = re.search(r'^go (\d+\.\d+)', c, re.M)
    if m: signals.append(f'go.mod: go {m.group(1)}')

# Rust
for tf in ['rust-toolchain.toml', 'rust-toolchain']:
    c, _ = read_file(f'{testbed}/{tf}')
    if c:
        m = re.search(r'channel\s*=\s*["\']([^"\']+)["\']', c)
        ver = m.group(1) if m else c.strip()[:20]
        signals.append(f'{tf}: {ver}')
        break

# Node
for nf in ['.nvmrc', '.node-version']:
    c, _ = read_file(f'{testbed}/{nf}')
    if c: signals.append(f'{nf}: {c.strip()}')

# Ruby
for rf in ['.ruby-version', '.tool-versions']:
    c, _ = read_file(f'{testbed}/{rf}')
    if c: signals.append(f'{rf}: {c.strip()}')

# Python
c, _ = read_file(f'{testbed}/.python-version')
if c: signals.append(f'.python-version: {c.strip()}')

c, _ = read_file(f'{testbed}/setup.py')
if c:
    m = re.search(r'python_requires\s*=\s*["\']([^"\']+)["\']', c)
    if m: signals.append(f'setup.py python_requires: {m.group(1)}')

c, _ = read_file(f'{testbed}/setup.cfg')
if c:
    m = re.search(r'python_requires\s*=\s*(.+)', c)
    if m: signals.append(f'setup.cfg python_requires: {m.group(1).strip()}')

c, _ = read_file(f'{testbed}/pyproject.toml')
if c:
    m = re.search(r'requires-python\s*=\s*["\']([^"\']+)["\']', c)
    if m: signals.append(f'pyproject.toml requires-python: {m.group(1)}')
    m2 = re.search(r'build-backend\s*=\s*["\']([^"\']+)["\']', c)
    if m2: signals.append(f'pyproject.toml build-backend: {m2.group(1)}')

c, _ = read_file(f'{testbed}/tox.ini')
if c:
    m = re.search(r'basepython\s*=\s*(.+)', c)
    if m: signals.append(f'tox.ini basepython: {m.group(1).strip()}')
    m2 = re.search(r'envlist\s*=\s*(.+)', c)
    if m2: signals.append(f'tox.ini envlist: {m2.group(1).strip()[:80]}')

# PHP
c, _ = read_file(f'{testbed}/.php-version')
if c: signals.append(f'.php-version: {c.strip()}')

c, _ = read_file(f'{testbed}/composer.json')
if c:
    try:
        cj = json.loads(c)
        php_req = cj.get('require', {}).get('php', '')
        if php_req: signals.append(f'composer.json require.php: {php_req}')
        req_dev = cj.get('require-dev', {})
        for pkg in ['phpunit/phpunit', 'behat/behat', 'phpspec/phpspec', 'pestphp/pest']:
            if pkg in req_dev:
                signals.append(f'composer.json test-framework: {pkg} {req_dev[pkg]}')
    except: pass

# Java - pom.xml
c, _ = read_file(f'{testbed}/pom.xml')
if c:
    m = re.search(r'<java\.version>(.*?)</java\.version>', c)
    if m: signals.append(f'pom.xml java.version: {m.group(1)}')
    m2 = re.search(r'<source>(\d+)</source>', c)
    if m2: signals.append(f'pom.xml compiler source: {m2.group(1)}')
    m3 = re.search(r'<maven\.compiler\.release>(\d+)</maven\.compiler\.release>', c)
    if m3: signals.append(f'pom.xml maven.compiler.release: {m3.group(1)}')
    m4 = re.search(r'<maven\.compiler\.target>(\d+)</maven\.compiler\.target>', c)
    if m4 and not m2: signals.append(f'pom.xml maven.compiler.target: {m4.group(1)}')

# Java - build.gradle / build.gradle.kts
c, _ = read_file(f'{testbed}/build.gradle')
c2, _ = read_file(f'{testbed}/build.gradle.kts')
gc = c or c2
if gc:
    m = re.search(r"sourceCompatibility\s*=\s*['\"]?(\S+?)['\"]?\s*\n", gc)
    if m: signals.append(f'build.gradle sourceCompatibility: {m.group(1)}')
    if not any('build.gradle' in s for s in signals):
        m = re.search(r'jvmToolchain\s*\((\d+)\)', gc)
        if m: signals.append(f'build.gradle.kts jvmToolchain: {m.group(1)}')
        else:
            m = re.search(r'JavaVersion\.VERSION_(\d+)', gc)
            if m: signals.append(f'build.gradle.kts JavaVersion: {m.group(1)}')

c, _ = read_file(f'{testbed}/gradle.properties')
if c and not any('build.gradle' in s or 'pom.xml' in s for s in signals):
    m = re.search(r'(?:sourceCompatibility|javaVersion|java\.version)\s*=\s*([\d.]+)', c, re.M)
    if m: signals.append(f'gradle.properties javaVersion: {m.group(1)}')

# C#
_global_json_found = False
for _gj_dir in [testbed] + [os.path.join(testbed, d) for d in sorted(os.listdir(testbed))
                              if os.path.isdir(os.path.join(testbed, d)) and not d.startswith('.')][:8]:
    c, _ = read_file(f'{_gj_dir}/global.json')
    if c:
        try:
            gj = json.loads(c)
            ver = gj.get('sdk', {}).get('version', '')
            if ver:
                rel = os.path.relpath(_gj_dir, testbed)
                label = 'global.json sdk' if rel == '.' else f'global.json sdk ({rel})'
                signals.append(f'{label}: {ver}')
                _global_json_found = True
                break
        except: pass

if not _global_json_found:
    _csproj_tf = []
    for _root, _dirs, _files in os.walk(testbed):
        _dirs[:] = [d for d in _dirs if not d.startswith('.') and d not in ('bin','obj','node_modules')]
        for _f in _files:
            if _f.endswith(('.csproj', '.fsproj', '.vbproj')):
                _c, _ = read_file(os.path.join(_root, _f))
                if _c:
                    for _m in re.finditer(r'<TargetFrameworks?\s*>(.*?)</TargetFrameworks?>', _c):
                        for _tf in re.split(r'[;,\s]+', _m.group(1)):
                            _tf = _tf.strip()
                            if _tf and _tf not in _csproj_tf:
                                _csproj_tf.append(_tf)
        if len(_csproj_tf) >= 6:
            break
    if _csproj_tf:
        signals.append(f'.csproj TargetFramework: {", ".join(_csproj_tf[:6])}')

if not _global_json_found and not any('.csproj' in s for s in signals):
    sln = [f for f in os.listdir(testbed) if f.endswith('.sln')]
    if sln: signals.append(f'.sln: {sln[0]}')

# Rust - Cargo.toml features and dev-dependencies
_cargo_content, _ = read_file(f'{testbed}/Cargo.toml')
if _cargo_content:
    _in_features = False
    _features_found = []
    for _line in _cargo_content.splitlines():
        _ls = _line.strip()
        if re.match(r'^\[.*features.*\]', _ls):
            _in_features = True
            continue
        if _in_features:
            if _ls.startswith('['):
                _in_features = False
                continue
            _fm = re.match(r'^(\w[\w-]*)\s*=', _ls)
            if _fm and _fm.group(1) != 'default':
                _features_found.append(_fm.group(1))
    if _features_found:
        signals.append(f'Cargo.toml features: {", ".join(_features_found[:8])}')

# CI workflows
wf_dir = f'{testbed}/.github/workflows'
ci_install_lines = []
if os.path.isdir(wf_dir):
    for wf in sorted(os.listdir(wf_dir))[:5]:
        wf_content, _ = read_file(f'{wf_dir}/{wf}')
        if wf_content:
            for pattern in [r'python-version["\s:]+["\']?([\d.]+)', r'node-version["\s:]+["\']?([\d.]+)', r'java-version["\s:]+["\']?(\d+)', r'dotnet-version["\s:]+["\']?([\d.]+)', r'php-version["\s:]+["\']?([\d.]+)']:
                m = re.search(pattern, wf_content)
                if m: signals.append(f'.github/{wf}: {m.group(0)}')
            for line in wf_content.splitlines():
                stripped = line.strip()
                if any(kw in stripped for kw in ['apt-get install', 'apt install', 'pip install', 'npm install', 'brew install', 'yum install', 'dnf install', 'composer install', 'bundle install']):
                    ci_install_lines.append(f'[{wf}] {stripped}')

result['version_signals'] = signals
result['ci_install_commands'] = ci_install_lines[:20]

# 3. Package manager files
pkg_checks = [
    'requirements.txt','setup.py','pyproject.toml','setup.cfg','Pipfile','tox.ini',
    'package.json','yarn.lock','package-lock.json','pnpm-lock.yaml',
    'go.mod','go.sum',
    'Cargo.toml','Cargo.lock',
    'pom.xml','build.gradle','build.gradle.kts','gradlew','mvnw','settings.gradle','settings.gradle.kts',
    'Gemfile','Gemfile.lock',
    'composer.json','composer.lock',
    'CMakeLists.txt','configure.ac','configure','meson.build','Makefile','GNUmakefile','SConstruct',
    'global.json',
]
result['package_files'] = [f for f in pkg_checks if os.path.exists(f'{testbed}/{f}')]

# 4. Key file previews (truncated)
PREVIEW_LIMIT = 1500
key_files = [
    'requirements.txt', 'setup.py', 'setup.cfg', 'pyproject.toml',
    'go.mod', 'Cargo.toml', 'package.json',
    'pom.xml', 'build.gradle', 'build.gradle.kts',
    'Gemfile', 'composer.json',
    'CMakeLists.txt', 'Makefile', 'meson.build', 'configure.ac',
]
file_previews = {}
for f in key_files:
    fp = f'{testbed}/{f}'
    if os.path.exists(fp):
        c, sz = read_file(fp, limit=PREVIEW_LIMIT)
        if c is not None:
            truncated = sz > PREVIEW_LIMIT
            file_previews[f] = {'size': sz, 'truncated': truncated, 'preview': c}
result['file_previews'] = file_previews

# 5. C/C++ build system deep scan
pkg_files = result['package_files']
is_c_cpp = any(f in pkg_files for f in ['CMakeLists.txt', 'Makefile', 'GNUmakefile', 'configure.ac', 'meson.build', 'SConstruct'])

if is_c_cpp:
    build_info = {}
    build_systems = []
    if 'CMakeLists.txt' in pkg_files: build_systems.append('cmake')
    if any(f in pkg_files for f in ['Makefile', 'GNUmakefile']): build_systems.append('make')
    if 'configure.ac' in pkg_files: build_systems.append('autotools')
    if 'meson.build' in pkg_files: build_systems.append('meson')
    if 'SConstruct' in pkg_files: build_systems.append('scons')
    build_info['build_systems'] = build_systems

    cmake_test_targets = []
    cmake_files_scanned = []
    cmake_scan_dirs = [testbed]
    try:
        cmake_scan_dirs += [os.path.join(testbed, d) for d in sorted(os.listdir(testbed))
                            if os.path.isdir(os.path.join(testbed, d)) and not d.startswith('.')][:15]
    except: pass
    for tdir in ['test', 'tests', 'testing', 'src/test', 'src/tests']:
        td = os.path.join(testbed, tdir)
        if os.path.isdir(td) and td not in cmake_scan_dirs:
            cmake_scan_dirs.append(td)

    for scan_dir in cmake_scan_dirs:
        cm_path = os.path.join(scan_dir, 'CMakeLists.txt')
        cm, _ = read_file(cm_path)
        if cm:
            rel = os.path.relpath(cm_path, testbed)
            cmake_files_scanned.append(rel)
            for mt in re.finditer(r'add_test\s*\(\s*(?:NAME\s+)?(\S+)', cm):
                t = mt.group(1)
                if t not in cmake_test_targets: cmake_test_targets.append(t)
            for mt in re.finditer(r'add_executable\s*\(\s*(\w*[Tt]est\w*)', cm):
                t = mt.group(1)
                if t not in cmake_test_targets: cmake_test_targets.append(t)
            for mt in re.finditer(r'gtest_(?:discover|add)_tests\s*\(\s*(\w+)', cm):
                t = mt.group(1)
                if t not in cmake_test_targets: cmake_test_targets.append(t)
    build_info['cmake_test_targets'] = cmake_test_targets[:20]
    build_info['cmake_files_scanned'] = cmake_files_scanned[:20]

    if 'make' in build_systems:
        make_targets = []
        mk, _ = read_file(f'{testbed}/Makefile', limit=8000)
        if mk:
            for mt in re.finditer(r'^([\w._-]*[Tt]est[\w._-]*)\s*:', mk, re.M):
                t = mt.group(1)
                if t not in make_targets: make_targets.append(t)
            for mt in re.finditer(r'^(check|check-am|check-TESTS)\s*:', mk, re.M):
                t = mt.group(1)
                if t not in make_targets: make_targets.append(t)
        build_info['make_test_targets'] = make_targets[:15]

    if 'meson' in build_systems:
        mc, _ = read_file(f'{testbed}/meson.build', limit=8000)
        meson_tests = []
        if mc:
            for mt in re.finditer(r"test\s*\(\s*'([^']+)'", mc):
                meson_tests.append(mt.group(1))
            for mt in re.finditer(r'executable\s*\(\s*["\'](\w*[Tt]est\w*)["\']', mc):
                t = mt.group(1)
                if t not in meson_tests: meson_tests.append(t)
        build_info['meson_test_targets'] = meson_tests[:15]

    test_frameworks = []
    test_fw_patterns = {
        'gtest': [r'#include\s*[<"]gtest/', r'TEST\s*\(', r'TEST_F\s*\('],
        'catch2': [r'#include\s*[<"]catch', r'TEST_CASE\s*\('],
        'doctest': [r'#include\s*[<"]doctest', r'DOCTEST_'],
        'cunit': [r'#include\s*[<"]CUnit/', r'CU_add_test'],
        'cmocka': [r'#include\s*[<"]cmocka', r'cmocka_unit_test'],
        'check': [r'#include\s*[<"]check\.h', r'ck_assert'],
        'unity': [r'#include\s*[<"]unity\.h', r'TEST_ASSERT'],
        'boost.test': [r'#include\s*[<"]boost/test/', r'BOOST_AUTO_TEST'],
        'criterion': [r'#include\s*[<"]criterion/', r'Test\s*\('],
    }
    all_cmake = ' '.join(file_previews.get('CMakeLists.txt', {}).get('preview', '').lower() for _ in [1])
    for fw, _ in test_fw_patterns.items():
        if fw.lower().replace('.', '') in all_cmake.replace('.', ''):
            test_frameworks.append(fw)
    if not test_frameworks:
        for tdir in ['test', 'tests', 'testing', 'src/test', 'src/tests', '.']:
            td = os.path.join(testbed, tdir)
            if not os.path.isdir(td): continue
            try:
                for fn in sorted(os.listdir(td))[:10]:
                    if fn.endswith(('.c', '.cpp', '.cc', '.cxx', '.h', '.hpp')):
                        tc, _ = read_file(os.path.join(td, fn), limit=2000)
                        if tc:
                            for fw, pats in test_fw_patterns.items():
                                if fw not in test_frameworks:
                                    for p in pats:
                                        if re.search(p, tc):
                                            test_frameworks.append(fw)
                                            break
            except: pass
            if test_frameworks: break
    build_info['test_frameworks'] = test_frameworks

    setup_scripts = []
    for pattern_dir in ['scripts', 'ci', '.ci', 'ci/docker', '.github/scripts', 'build', 'tools']:
        sd = os.path.join(testbed, pattern_dir)
        if os.path.isdir(sd):
            try:
                for fn in sorted(os.listdir(sd))[:10]:
                    fp = os.path.join(sd, fn)
                    if os.path.isfile(fp):
                        rel = os.path.relpath(fp, testbed)
                        if fn.endswith(('.sh', '.bash')) or fn.startswith(('setup', 'install', 'build', 'deps', 'bootstrap')):
                            setup_scripts.append(rel)
                        elif fn in ('Dockerfile', 'Dockerfile.dev', 'Dockerfile.build', 'Dockerfile.ci'):
                            setup_scripts.append(rel)
            except: pass
    for rf in ['Dockerfile', 'Dockerfile.dev', '.circleci/config.yml']:
        if os.path.exists(os.path.join(testbed, rf)):
            setup_scripts.append(rf)
    build_info['setup_scripts'] = setup_scripts[:20]

    cmake_ci_flags = []
    if os.path.isdir(wf_dir):
        for wf in sorted(os.listdir(wf_dir))[:5]:
            wf_content, _ = read_file(f'{wf_dir}/{wf}')
            if wf_content:
                for line in wf_content.splitlines():
                    stripped = line.strip()
                    if 'cmake' in stripped.lower() and ('-D' in stripped or '-B' in stripped or '-G' in stripped):
                        cmake_ci_flags.append(f'[{wf}] {stripped}')
    build_info['cmake_ci_flags'] = cmake_ci_flags[:10]

    result['build_info'] = build_info
else:
    result['build_info'] = {}

# 6. Test directories
test_dirs = []
for td_name in ['test', 'tests', 'testing', 'spec', 'specs', 'test_suite',
                 'src/test', 'src/tests', 'src/testing', '__tests__']:
    td = os.path.join(testbed, td_name)
    if os.path.isdir(td):
        try:
            entries = sorted(os.listdir(td))[:15]
            test_dirs.append({'path': td_name, 'entries': entries})
        except:
            test_dirs.append({'path': td_name, 'entries': []})
result['test_dirs'] = test_dirs

print(json.dumps(result))
PYEOF
"""
    out = await runtime.run_command(explore_script, timeout=30)
    raw = (out.stdout or '') + (out.stderr or '')

    try:
        json_start = raw.find('{')
        json_end = raw.rfind('}') + 1
        data = json.loads(raw[json_start:json_end]) if json_start >= 0 else {}
    except Exception:
        logger.warning(f'[AUTO_SETUP] Failed to parse explore_repo output: {raw[:500]}')
        data = {}

    return data


# Known available images in the mirror registry.
# _determine_image validates against this set; unknown images are rejected
# to avoid destroying the current runtime on a pull that will fail.
_KNOWN_IMAGES: set[str] = {
    # Ubuntu (base)
    'ubuntu-git:20.04',
    'ubuntu-git:22.04',
    # Python (slim)
    'python-git:2.7-slim',
    'python-git:3.5-slim',
    'python-git:3.6-slim',
    'python-git:3.7-slim',
    'python-git:3.8-slim',
    'python-git:3.9-slim',
    'python-git:3.10-slim',
    'python-git:3.11-slim',
    'python-git:3.12-slim',
    'python-git:3.13-slim',
    # Python (bullseye)
    'python-git:3.6-bullseye',
    'python-git:3.7-bullseye',
    'python-git:3.8-bullseye',
    'python-git:3.9-bullseye',
    'python-git:3.10-bullseye',
    'python-git:3.11-bullseye',
    'python-git:3.12-bullseye',
    'python-git:3.13-bullseye',
    # Ruby (slim)
    'ruby-git:2.3-slim',
    'ruby-git:2.4-slim',
    'ruby-git:2.5-slim',
    'ruby-git:2.6-slim',
    'ruby-git:2.7-slim',
    'ruby-git:3.0-slim',
    'ruby-git:3.1-slim',
    'ruby-git:3.2-slim',
    'ruby-git:3.3-slim',
    'ruby-git:3.4-slim',
    # Ruby (bullseye)
    'ruby-git:2.7-bullseye',
    'ruby-git:3.1-bullseye',
    'ruby-git:3.2-bullseye',
    'ruby-git:3.3-bullseye',
    # Node
    'node-git:6-stretch',
    'node-git:8-stretch',
    'node-git:10-buster',
    'node-git:12-bullseye',
    'node-git:14-bullseye',
    'node-git:16-bullseye',
    'node-git:18-bullseye',
    'node-git:20-bullseye',
    'node-git:22-bullseye',
    'node-git:20-slim',
    'node-git:22-bookworm',
    # Go
    'golang-local:1.11',
    'golang-local:1.12',
    'golang-local:1.13',
    'golang-local:1.14',
    'golang-local:1.15',
    'golang-local:1.16',
    'golang-local:1.17',
    'golang-local:1.18',
    'golang-local:1.19',
    'golang-local:1.20',
    'golang-local:1.21',
    'golang-local:1.22',
    'golang-local:1.23',
    'golang-local:1.24',
    'golang-local:1.25',
    'golang-local:1.26',
    # C# / .NET
    'dotnet-git:2.1',
    'dotnet-git:core-2.2',
    'dotnet-git:3.1',
    'dotnet-git:5.0',
    'dotnet-git:6.0',
    'dotnet-git:7.0',
    'dotnet-git:8.0',
    'dotnet-git:9.0',
    'dotnet-git:10.0',
    # C / C++ / Debian / ROS
    'debian-git:bullseye-slim',
    'debian-git:bookworm-slim',
    'ubuntu-cpp:20.04',
    'ubuntu-cpp:22.04',
    'ros-git:noetic',
    'ros-git:humble',
    'ros-git:foxy',
    # PHP
    'php-git:5.6-cli',
    'php-git:7.0-cli',
    'php-git:7.1-cli',
    'php-git:7.2-cli',
    'php-git:7.3-cli',
    'php-git:7.4-cli',
    'php-git:8.0-cli',
    'php-git:8.1-cli',
    'php-git:8.2-cli',
    'php-git:8.3-cli',
    'php-git:8.4-cli',
    # Java
    'java-git:8',
    'java-git:11',
    'java-git:17',
    'java-git:21',
    # Rust
    'rust-git:1.60-slim',
    'rust-git:1.65-slim',
    'rust-git:1.70-slim',
    'rust-git:1.73-slim',
    'rust-git:1.75-slim',
    'rust-git:1.77-slim',
    'rust-git:1.80-slim',
    'rust-git:1.83-slim',
    'rust-git:1.85-slim',
    'rust-git:latest',
}


def _validate_image(image: str) -> bool:
    """Check if a short image name exists in the mirror registry."""
    return image in _KNOWN_IMAGES


def _determine_image(language: str, explore_data: dict) -> tuple[str, str]:
    """Determine the best runtime image based on explore_repo results.

    Returns:
        (image_name, version_string) — image_name is the short name (e.g. 'golang-local:1.21'),
        version_string is the detected version or 'unknown'.
        Returns ('', version) if no switch needed or image not available.
    """
    signals_str = ' '.join(explore_data.get('version_signals', []))
    pkg_files = explore_data.get('package_files', [])

    # Go
    if 'go.mod: go' in signals_str:
        m = re.search(r'go\.mod: go (\d+\.\d+)', signals_str)
        if m:
            ver = m.group(1)
            return f'golang-local:{ver}', ver

    # Rust
    if 'rust-toolchain' in signals_str:
        m = re.search(r'rust-toolchain.*?: ([\d.]+)', signals_str)
        if m:
            ver = m.group(1)[:4]
            return f'rust-git:{ver}-slim', ver
        if 'nightly' in signals_str:
            return 'rust-git:latest', 'nightly'

    # C# — global.json sdk
    if 'global.json sdk' in signals_str:
        m = re.search(r'global\.json sdk[^:]*: ([\d.]+)', signals_str)
        if m:
            ver = m.group(1)[:3]
            return f'dotnet-git:{ver}', m.group(1)
    elif '.csproj TargetFramework' in signals_str:
        tf_m = re.search(r'\.csproj TargetFramework: ([\w.,; ]+)', signals_str)
        if tf_m:
            vers = []
            for tfv in re.findall(r'net(\d+)\.(\d+)', tf_m.group(1)):
                major = int(tfv[0])
                if major >= 5:
                    vers.append(major)
            if vers:
                dv = max(vers)
                return f'dotnet-git:{dv}.0', f'net{dv}.0'

    # Java
    _java_ver_signal = any(
        k in signals_str
        for k in [
            'build.gradle sourceCompatibility',
            'build.gradle.kts jvmToolchain',
            'build.gradle.kts JavaVersion',
            'gradle.properties javaVersion',
            'pom.xml java.version',
            'pom.xml compiler source',
            'pom.xml maven.compiler.release',
            'pom.xml maven.compiler.target',
            'java-version',
        ]
    )
    if _java_ver_signal:
        mj = re.search(
            r'(?:jvmToolchain|JavaVersion\.VERSION_|javaVersion|java\.version|sourceCompatibility'
            r'|compiler (?:source|release|target)|java-version[^=\d]*)[\s=:"\']*(\d+)',
            signals_str,
        )
        if mj:
            jv = int(mj.group(1))
            jdk = '8' if jv <= 8 else ('11' if jv <= 11 else ('17' if jv <= 17 else '21'))
            return f'java-git:{jdk}', str(jv)
    elif any(f in pkg_files for f in ['pom.xml', 'build.gradle', 'build.gradle.kts', 'gradlew']):
        return 'java-git:11', 'unknown'

    # Node
    if '.nvmrc' in signals_str or '.node-version' in signals_str:
        m = re.search(r'(?:\.nvmrc|\.node-version): v?(\d+)', signals_str)
        if m:
            ver = m.group(1)
            return f'node-git:{ver}-bullseye', ver
    if 'package.json' in pkg_files:
        ci_node = re.search(r'node-version["\s:]+["\']?(\d+)', signals_str)
        if ci_node:
            return f'node-git:{ci_node.group(1)}-bullseye', ci_node.group(1)

    # Ruby
    if '.ruby-version' in signals_str:
        m = re.search(r'\.ruby-version: (\d+\.\d+)', signals_str)
        if m:
            ver = m.group(1)
            return f'ruby-git:{ver}-slim', ver

    # PHP
    php_ver = None
    if '.php-version' in signals_str:
        m = re.search(r'\.php-version: (\d+\.\d+)', signals_str)
        if m:
            php_ver = m.group(1)
    if not php_ver and 'composer.json require.php' in signals_str:
        m = re.search(r'composer\.json require\.php: [^\d]*(\d+\.\d+)', signals_str)
        if m:
            php_ver = m.group(1)
    if not php_ver:
        ci_php = re.search(r'php-version["\s:]+["\']?(\d+\.\d+)', signals_str)
        if ci_php:
            php_ver = ci_php.group(1)
    if php_ver:
        return f'php-git:{php_ver}-cli', php_ver

    # Python — only switch when .python-version pins a version != 3.10 (ubuntu-git default)
    _PYTHON_AVAILABLE = {
        '2.7',
        '3.5',
        '3.6',
        '3.7',
        '3.8',
        '3.9',
        '3.10',
        '3.11',
        '3.12',
        '3.13',
    }
    if language.lower().strip() == 'python':
        py_ver = None
        # Priority 1: .python-version (explicit pin)
        m = re.search(r'\.python-version: (\d+\.\d+)', signals_str)
        if m:
            py_ver = m.group(1)
        # Priority 2: CI python-version from workflow yaml
        if not py_ver:
            m = re.search(r'python-version["\s:]+["\']?(\d+\.\d+)', signals_str)
            if m:
                py_ver = m.group(1)
        # Priority 3: tox.ini basepython
        if not py_ver:
            m = re.search(r'basepython: python(\d+\.\d+)', signals_str)
            if m:
                py_ver = m.group(1)
        if py_ver and py_ver in _PYTHON_AVAILABLE and py_ver != '3.10':
            return f'python-git:{py_ver}-slim', py_ver

    # --- Fallback: language is known but no version signal found ---
    # Use safe defaults so the agent doesn't get stuck on ubuntu-git with no runtime binary.
    _LANGUAGE_DEFAULT_IMAGES = {
        'go': ('golang-local:1.21', '1.21'),
        'golang': ('golang-local:1.21', '1.21'),
        'rust': ('rust-git:latest', 'unknown'),
        'javascript': ('node-git:18-bullseye', '18'),
        'typescript': ('node-git:18-bullseye', '18'),
        'ruby': ('ruby-git:3.2-slim', '3.2'),
        'php': ('php-git:8.1-cli', '8.1'),
        'java': ('java-git:11', '11'),
        'c#': ('dotnet-git:8.0', '8.0'),
        'csharp': ('dotnet-git:8.0', '8.0'),
    }
    lang_lower = language.lower().strip()
    if lang_lower in _LANGUAGE_DEFAULT_IMAGES:
        # If explore_repo failed (no package_files at all), trust the dataset language tag
        # and use the default image — better to have the right runtime than none.
        if not pkg_files:
            return _LANGUAGE_DEFAULT_IMAGES[lang_lower]
        # Otherwise, only use fallback if the repo has relevant package files confirming the language
        _lang_pkg_indicators = {
            'go': ['go.mod'],
            'golang': ['go.mod'],
            'rust': ['Cargo.toml'],
            'javascript': ['package.json'],
            'typescript': ['package.json'],
            'ruby': ['Gemfile'],
            'php': ['composer.json'],
            'java': ['pom.xml', 'build.gradle', 'build.gradle.kts', 'gradlew'],
            'c#': [],  # dataset language tag is authoritative; .csproj/.sln not in pkg_checks
            'csharp': [],
        }
        indicators = _lang_pkg_indicators.get(lang_lower, [])
        if not indicators or any(f in pkg_files for f in indicators):
            return _LANGUAGE_DEFAULT_IMAGES[lang_lower]

    # C, C++ — no image switch needed (ubuntu-git is the default)
    return '', 'unknown'


async def _probe_patches(runtime, task_config: dict) -> dict:
    """Probe golden_patch and test_patch with git apply --check.

    Returns dict with: golden_status, golden_detail, test_status, test_detail,
    test_applied_on, eval_strategy.
    """
    from nexus.extensions.tasks.swe_builder.runtime_utils import git_reset_and_clean, run_command

    testbed_dir = task_config.get('testbed_dir', '/testbed')
    base_commit = task_config['base_commit']
    golden_patch = task_config.get('golden_patch', '')
    test_patch = task_config.get('test_patch', '')

    result = {
        'golden_status': 'empty',
        'golden_detail': '',
        'test_status': 'empty',
        'test_detail': '',
        'test_applied_on': 'none',
        'eval_strategy': 'standard',
    }

    def _classify_apply_output(exit_code: int, combined: str) -> tuple[str, str]:
        """Classify git apply --check output into (status, detail)."""
        if exit_code == 0:
            if 'trailing whitespace' in combined.lower():
                return 'crlf', 'CRLF/trailing whitespace detected'
            file_count = combined.lower().count('checking patch')
            return 'clean', f'applies cleanly ({file_count} files)'
        combined_lower = combined.lower()
        if 'already exists in working directory' in combined_lower:
            return 'already_exists', combined[:500].strip()
        if 'does not apply' in combined_lower or 'hunk' in combined_lower:
            return 'hunk_failed', combined[:500].strip()
        return 'hunk_failed', combined[:500].strip()

    async def _run_check(patch_file: str) -> tuple[int, str]:
        """Run git apply --check and return (exit_code, combined_output)."""
        cmd = f'cd {testbed_dir} && cat {patch_file} | git apply --check --verbose 2>&1; echo "EXIT_CODE=$?"'
        out = await runtime.invoke(run_command, **{'command': cmd, 'cwd': testbed_dir, 'timeout': 30})
        combined = out.get('stdout', '') + out.get('stderr', '')
        ec_match = re.search(r'EXIT_CODE=(\d+)', combined)
        ec = int(ec_match.group(1)) if ec_match else out.get('exit_code', 1)
        clean = re.sub(r'EXIT_CODE=\d+\s*$', '', combined).strip()
        return ec, clean

    async def _run_check_ignore_ws(patch_file: str) -> tuple[int, str]:
        """Run git apply --check --ignore-whitespace."""
        cmd = (
            f'cd {testbed_dir} && '
            f'cat {patch_file} | git apply --check --verbose --ignore-whitespace 2>&1; echo "EXIT_CODE=$?"'
        )
        out = await runtime.invoke(run_command, **{'command': cmd, 'cwd': testbed_dir, 'timeout': 30})
        combined = out.get('stdout', '') + out.get('stderr', '')
        ec_match = re.search(r'EXIT_CODE=(\d+)', combined)
        ec = int(ec_match.group(1)) if ec_match else out.get('exit_code', 1)
        clean = re.sub(r'EXIT_CODE=\d+\s*$', '', combined).strip()
        return ec, clean

    async def _run_check_patch_fuzz(patch_file: str) -> tuple[int, str]:
        """Run patch --dry-run --fuzz=5 -p1."""
        cmd = f'cd {testbed_dir} && patch --dry-run --batch --fuzz=5 -p1 < {patch_file} 2>&1; echo "EXIT_CODE=$?"'
        out = await runtime.invoke(run_command, **{'command': cmd, 'cwd': testbed_dir, 'timeout': 30})
        combined = out.get('stdout', '') + out.get('stderr', '')
        ec_match = re.search(r'EXIT_CODE=(\d+)', combined)
        ec = int(ec_match.group(1)) if ec_match else out.get('exit_code', 1)
        clean = re.sub(r'EXIT_CODE=\d+\s*$', '', combined).strip()
        return ec, clean

    async def _probe_with_fallback(patch_file: str) -> tuple[str, str]:
        """Probe a patch using three methods (strict → --ignore-whitespace → patch --fuzz=5).

        Returns (status, detail) with method annotation.
        """
        # Method 1: strict
        ec, combined = await _run_check(patch_file)
        status, detail = _classify_apply_output(ec, combined)
        if status == 'clean':
            return status, detail

        # Keep first error for final fallback
        first_status, first_detail = status, detail

        # Method 2: --ignore-whitespace
        ec2, combined2 = await _run_check_ignore_ws(patch_file)
        status2, detail2 = _classify_apply_output(ec2, combined2)
        if status2 == 'clean':
            return 'clean', f'applies with --ignore-whitespace ({detail2})'
        if status2 == 'crlf':
            return 'crlf', detail2

        # Method 3: patch --fuzz=5
        ec3, combined3 = await _run_check_patch_fuzz(patch_file)
        if ec3 == 0:
            file_count = combined3.lower().count('patching file')
            return 'clean', f'applies with patch --fuzz=5 ({file_count} files)'

        # All failed — return original strict error
        return first_status, first_detail

    async def _reset_testbed():
        await runtime.invoke(git_reset_and_clean, **{'testbed_dir': testbed_dir, 'commit': base_commit})

    # --- Probe golden_patch ---
    if golden_patch.strip():
        result['golden_status'], result['golden_detail'] = await _probe_with_fallback('/tmp/golden_patch.patch')
        await _reset_testbed()
    else:
        result['golden_status'] = 'empty'
        result['golden_detail'] = 'no golden patch'

    # --- Probe test_patch ---
    if test_patch.strip():
        if result['golden_status'] == 'clean':
            # Apply golden first using fallback methods, then check test_patch on top
            async def _apply_golden_with_fallback() -> bool:
                """Try to apply golden_patch using three methods. Returns True if applied."""
                # Method 1: strict
                cmd1 = f'cd {testbed_dir} && git apply --verbose /tmp/golden_patch.patch 2>&1; echo "APPLY_EXIT=$?"'
                out1 = await runtime.invoke(run_command, **{'command': cmd1, 'cwd': testbed_dir, 'timeout': 30})
                c1 = out1.get('stdout', '') + out1.get('stderr', '')
                m1 = re.search(r'APPLY_EXIT=(\d+)', c1)
                if m1 and m1.group(1) == '0':
                    return True

                # Method 2: --ignore-whitespace
                await _reset_testbed()
                cmd2 = (
                    f'cd {testbed_dir} && '
                    'git apply --verbose --ignore-whitespace /tmp/golden_patch.patch 2>&1; echo "APPLY_EXIT=$?"'
                )
                out2 = await runtime.invoke(run_command, **{'command': cmd2, 'cwd': testbed_dir, 'timeout': 30})
                c2 = out2.get('stdout', '') + out2.get('stderr', '')
                m2 = re.search(r'APPLY_EXIT=(\d+)', c2)
                if m2 and m2.group(1) == '0':
                    return True

                # Method 3: patch --fuzz=5
                await _reset_testbed()
                cmd3 = (
                    f'cd {testbed_dir} && '
                    'patch --batch --fuzz=5 -p1 < /tmp/golden_patch.patch 2>&1; echo "APPLY_EXIT=$?"'
                )
                out3 = await runtime.invoke(run_command, **{'command': cmd3, 'cwd': testbed_dir, 'timeout': 30})
                c3 = out3.get('stdout', '') + out3.get('stderr', '')
                m3 = re.search(r'APPLY_EXIT=(\d+)', c3)
                return bool(m3 and m3.group(1) == '0')

            golden_applied = await _apply_golden_with_fallback()

            if golden_applied:
                # Check test_patch on golden state using fallback
                result['test_status'], result['test_detail'] = await _probe_with_fallback('/tmp/test_patch.patch')
                result['test_applied_on'] = 'golden'
                await _reset_testbed()

                if result['test_status'] != 'clean':
                    # Fallback: try test_patch on base
                    base_status, base_detail = await _probe_with_fallback('/tmp/test_patch.patch')
                    if base_status == 'clean':
                        result['test_status'] = base_status
                        result['test_detail'] = base_detail
                        result['test_applied_on'] = 'base'
                    else:
                        result['test_status'] = 'context_mismatch'
                        result['test_detail'] = f'fails on both golden and base: {base_detail[:300]}'
                    await _reset_testbed()
            else:
                # Golden didn't apply with any method, try test_patch on base directly
                await _reset_testbed()
                result['test_status'], result['test_detail'] = await _probe_with_fallback('/tmp/test_patch.patch')
                result['test_applied_on'] = 'base'
                await _reset_testbed()
        else:
            # Golden not clean, try test_patch on base directly
            result['test_status'], result['test_detail'] = await _probe_with_fallback('/tmp/test_patch.patch')
            result['test_applied_on'] = 'base'
            await _reset_testbed()
    else:
        result['test_status'] = 'empty'
        result['test_detail'] = 'no test patch'

    # --- Determine eval_strategy ---
    # Strategy is primarily about what to do with test_patch:
    #   standard   → embed test_patch as heredoc (test_patch applies cleanly)
    #   custom_test → write custom test from golden_patch analysis (test_patch broken or empty)
    #   crlf_fix   → add autocrlf config, then standard (golden has CRLF issues)
    gs, ts = result['golden_status'], result['test_status']
    if gs == 'crlf':
        result['eval_strategy'] = 'crlf_fix'
    elif ts in ('hunk_failed', 'context_mismatch', 'already_exists', 'empty'):
        result['eval_strategy'] = 'custom_test'
    else:
        result['eval_strategy'] = 'standard'

    return result


def _get_pre_install_commands(language: str, explore_data: dict) -> list:
    """Generate language-specific pre-install commands based on explore data.

    Returns a list where each element is either:
    - str: a single command to run
    - list[str]: a fallback group — try in order, stop on first success
    """
    pkg_files = explore_data.get('package_files', [])
    lang = language.lower().strip()
    commands: list = []

    if lang == 'python':
        if 'pyproject.toml' in pkg_files or 'setup.py' in pkg_files or 'setup.cfg' in pkg_files:
            # Fallback group: try with extras first, then plain editable install
            commands.append([
                'cd /testbed && pip install -e ".[dev,test]" 2>&1',
                'cd /testbed && pip install -e . 2>&1',
            ])
            # requirements.txt as separate command (may have additional deps)
            if 'requirements.txt' in pkg_files:
                commands.append('cd /testbed && pip install -r requirements.txt 2>&1')
        elif 'requirements.txt' in pkg_files:
            commands.append('cd /testbed && pip install -r requirements.txt 2>&1')
        if 'tox.ini' in pkg_files:
            commands.append('pip install tox 2>&1')

    elif lang in ('go', 'golang'):
        if 'go.mod' in pkg_files:
            # Inline GOPROXY — config files don't persist env vars across run_command calls
            commands.append(
                'cd /testbed && GOPROXY=https://goproxy.cn,direct GONOSUMCHECK=* GOFLAGS=-mod=mod go mod download 2>&1'
            )

    elif lang == 'rust':
        if 'Cargo.toml' in pkg_files:
            commands.append('cd /testbed && CARGO_HTTP_MULTIPLEXING=false cargo fetch 2>&1')

    elif lang in ('javascript', 'typescript'):
        if 'package-lock.json' in pkg_files:
            commands.append('cd /testbed && npm install --ignore-scripts 2>&1')
        elif 'yarn.lock' in pkg_files:
            commands.append('cd /testbed && yarn install --frozen-lockfile 2>&1')
        elif 'pnpm-lock.yaml' in pkg_files:
            commands.append('cd /testbed && pnpm install --frozen-lockfile 2>&1')
        elif 'package.json' in pkg_files:
            commands.append('cd /testbed && npm install --ignore-scripts 2>&1')

    elif lang == 'java':
        # JVM ignores https_proxy env — inject proxy as JVM system properties
        java_proxy_cmd = (
            'if [ -n "$https_proxy" ]; then'
            " PROXY_HOST=$(echo $https_proxy | sed 's|http://||;s|https://||' | cut -d: -f1);"
            " PROXY_PORT=$(echo $https_proxy | sed 's|http://||;s|https://||' | cut -d: -f2);"
            ' export MAVEN_OPTS="-Dhttps.proxyHost=$PROXY_HOST -Dhttps.proxyPort=$PROXY_PORT'
            ' -Dhttp.proxyHost=$PROXY_HOST -Dhttp.proxyPort=$PROXY_PORT ${MAVEN_OPTS:-}";'
            ' export GRADLE_OPTS="-Dhttps.proxyHost=$PROXY_HOST -Dhttps.proxyPort=$PROXY_PORT'
            ' -Dhttp.proxyHost=$PROXY_HOST -Dhttp.proxyPort=$PROXY_PORT ${GRADLE_OPTS:-}"; fi'
        )
        commands.append(java_proxy_cmd)
        if 'pom.xml' in pkg_files:
            commands.append('cd /testbed && mvn dependency:resolve -q 2>&1')
        elif any(f in pkg_files for f in ['build.gradle', 'build.gradle.kts', 'gradlew']):
            # Redirect Gradle wrapper to local zip (services.gradle.org download is slow, 60s for 50MB)
            commands.append(
                'if [ -f /testbed/gradle/wrapper/gradle-wrapper.properties ] && [ -f /opt/gradle/gradle-latest.zip ]; then '
                'sed -i "s|distributionUrl=.*|distributionUrl=file\\\\:///opt/gradle/gradle-latest.zip|" '
                '/testbed/gradle/wrapper/gradle-wrapper.properties && echo "Gradle wrapper redirected to local zip"; fi'
            )
            commands.append('cd /testbed && chmod +x gradlew 2>/dev/null; ./gradlew dependencies --no-daemon 2>&1')

    elif lang == 'ruby':
        if 'Gemfile' in pkg_files:
            commands.append('cd /testbed && BUNDLE_PATH=/root/bundle bundle install 2>&1')

    elif lang == 'php':
        if 'composer.json' in pkg_files:
            commands.append('cd /testbed && composer install --no-scripts 2>&1')

    elif lang in ('c#', 'csharp'):
        commands.append('cd /testbed && dotnet restore 2>&1')

    elif lang in ('c', 'c++', 'cpp'):
        # Only install apt deps if detected from CI; don't do cmake configure
        ci_cmds = explore_data.get('ci_install_commands', [])
        apt_packages = set()
        for ci_line in ci_cmds:
            m = re.search(r'apt-get install\s+(?:-y\s+)?(.*)', ci_line)
            if m:
                for pkg in m.group(1).split():
                    pkg = pkg.strip()
                    if pkg and not pkg.startswith('-'):
                        apt_packages.add(pkg)
        apt_packages.update(['cmake', 'build-essential'])
        if apt_packages:
            commands.append(f'apt-get install -y {" ".join(sorted(apt_packages))} 2>&1')

    return commands


async def _pre_install(runtime, language: str, explore_data: dict) -> tuple[list[dict], str]:
    """Run pre-install commands, return (commands_run, summary).

    Total budget: 900 seconds across all commands.
    Supports fallback groups (list[str]) — tries in order, stops on first success.
    Detects timeout (stalled) and OOM (exit_code=137) conditions.
    """
    from nexus.extensions.tasks.swe_builder.runtime_utils import run_command

    commands = _get_pre_install_commands(language, explore_data)
    if not commands:
        return [], 'no install commands for this language'

    commands_run = []
    total_elapsed = 0.0
    budget = 900.0

    PER_CMD_CAP = 300.0  # single command cap — prevents one stuck install from eating entire budget

    async def _run_single(cmd: str, remaining_budget: float) -> tuple[dict, float]:
        """Run a single command, return (entry_dict, elapsed_seconds)."""
        cmd_timeout = min(remaining_budget, PER_CMD_CAP)
        t0 = time.monotonic()
        try:
            result = await runtime.invoke(
                run_command, **{'command': cmd, 'cwd': '/testbed', 'timeout': int(cmd_timeout)}
            )
            elapsed = time.monotonic() - t0
            exit_code = result.get('exit_code', -1)

            # Classify status
            if exit_code == 0:
                status = 'ok'
            elif exit_code in (137, -9):
                status = 'oom'
            elif elapsed >= cmd_timeout * 0.95:
                status = 'timeout'
            else:
                status = 'failed'

            entry = {
                'cmd': cmd[:200],
                'status': status,
                'duration_s': round(elapsed, 1),
                'exit_code': exit_code,
            }
            if status != 'ok':
                combined = result.get('stdout', '') + result.get('stderr', '')
                error_lines = [
                    l
                    for l in combined.splitlines()
                    if re.search(r'error|fail|cannot|unable|not found|denied|fatal|killed|oom', l, re.IGNORECASE)
                ]
                entry['error'] = error_lines[0][:300] if error_lines else combined[-300:]
            return entry, elapsed
        except Exception as e:
            elapsed = time.monotonic() - t0
            error_str = str(e)
            status = 'oom' if ('killed' in error_str.lower() or 'oom' in error_str.lower()) else 'error'
            return {
                'cmd': cmd[:200],
                'status': status,
                'duration_s': round(elapsed, 1),
                'error': error_str[:300],
            }, elapsed

    def _skip_entry(label: str) -> dict:
        return {
            'cmd': label[:200],
            'status': 'skipped',
            'duration_s': 0,
            'error': f'budget exhausted ({total_elapsed:.0f}s / {budget:.0f}s)',
        }

    for item in commands:
        remaining = budget - total_elapsed
        if remaining < 10:
            label = item[0] if isinstance(item, list) else item
            commands_run.append(_skip_entry(label))
            continue

        if isinstance(item, list):
            # Fallback group: try in order, stop on first success or OOM
            for cmd in item:
                remaining = budget - total_elapsed
                if remaining < 10:
                    commands_run.append(_skip_entry(cmd))
                    break
                entry, elapsed = await _run_single(cmd, remaining)
                total_elapsed += elapsed
                commands_run.append(entry)
                if entry['status'] in ('ok', 'oom', 'timeout'):
                    break  # success → done; oom/timeout → don't try more
        else:
            entry, elapsed = await _run_single(item, remaining)
            total_elapsed += elapsed
            commands_run.append(entry)

    # Build summary
    ok_count = sum(1 for c in commands_run if c['status'] == 'ok')
    fail_count = sum(1 for c in commands_run if c['status'] in ('failed', 'error'))
    timeout_count = sum(1 for c in commands_run if c['status'] == 'timeout')
    oom_count = sum(1 for c in commands_run if c['status'] == 'oom')
    parts = [f'{ok_count} ok', f'{fail_count} failed']
    if timeout_count:
        parts.append(f'{timeout_count} timeout')
    if oom_count:
        parts.append(f'{oom_count} oom')
    parts.append(f'{total_elapsed:.0f}s total')
    summary = ', '.join(parts)
    return commands_run, summary


async def auto_setup(ctx, runtime, task_config: dict) -> AutoSetupResult:
    """Deterministic setup: explore → image switch → probe patches → pre-install.

    Runs BEFORE the agent loop. Does not consume agent iterations.
    Returns AutoSetupResult to be injected into the user prompt.
    """
    t0 = time.monotonic()
    result = AutoSetupResult()
    language = task_config.get('language', '').lower()
    result.language = language
    result.selected_image = task_config.get('base_image', 'ubuntu-git:22.04')

    logger.debug('=' * 80)
    logger.debug('[AUTO_SETUP] Starting pre-build setup for %s (%s)', task_config.get('instance_id', '?'), language)
    logger.debug('=' * 80)

    # ---- Step 1: Explore repo ----
    logger.debug('[AUTO_SETUP] Step 1: Exploring repository...')
    try:
        explore_data = await _explore_repo_internal(runtime)
    except Exception as e:
        logger.error(f'[AUTO_SETUP] explore_repo failed: {e}')
        explore_data = {}
        result.warnings.append(f'explore_repo failed: {e}')

    result.version_signals = explore_data.get('version_signals', [])
    result.package_files = explore_data.get('package_files', [])
    result.test_dirs = explore_data.get('test_dirs', [])
    result.build_systems = explore_data.get('build_info', {}).get('build_systems', [])
    result.key_file_previews = explore_data.get('file_previews', {})
    result.ci_install_commands = explore_data.get('ci_install_commands', [])
    result.top_dirs = explore_data.get('top_dirs', [])
    logger.debug(
        '[AUTO_SETUP] Step 1 done — pkg_files=%s, version_signals=%d, test_dirs=%d',
        result.package_files[:5],
        len(result.version_signals),
        len(result.test_dirs),
    )

    # ---- Step 2: Determine and switch image ----
    logger.debug('[AUTO_SETUP] Step 2: Determining runtime image...')
    recommended_image, version = _determine_image(language, explore_data)
    result.language_version = version

    _switch_failed = False  # tracks whether switch destroyed the runtime without replacement
    if recommended_image:
        current_image = task_config.get('base_image', '')
        # Check if we already have the right image (compare short name in resolved URL)
        safe_recommended = (
            recommended_image.lower().replace('-', '_').replace(':', '_').replace('/', '_').replace('.', '_')
        )
        if safe_recommended not in current_image.lower().replace('-', '_').replace(':', '_').replace('/', '_').replace(
            '.', '_'
        ):
            # Validate image exists in registry before switching (replace_runtime destroys the old container first)
            if not _validate_image(recommended_image):
                logger.warning(f'[AUTO_SETUP] Image {recommended_image!r} not in known registry, skipping switch')
                result.warnings.append(
                    f'Image {recommended_image} not available in registry — keeping current image. '
                    f'Agent can try switch_runtime_image manually if needed.'
                )
                result.selected_image = current_image
            else:
                logger.debug(f'[AUTO_SETUP] Switching image: {current_image} → {recommended_image}')
                try:
                    resolved_image = _resolve_mirror_image(recommended_image, current_image)
                    new_runtime = await ctx.replace_runtime('main', image=resolved_image)
                    runtime = new_runtime  # use new runtime for subsequent steps

                    # Update task_config
                    task_config['base_image'] = resolved_image
                    ctx.tool_state['base_image_switch_count'] = ctx.tool_state.get('base_image_switch_count', 0) + 1
                    ctx.tool_state.pop('terminal_session_id', None)

                    # Reinitialize testbed (also re-writes /tmp patch files lost with old container)
                    await _reinitialize_testbed(ctx, runtime)

                    result.selected_image = resolved_image
                    result.image_switched = True
                    logger.debug(f'[AUTO_SETUP] Image switched to {resolved_image}')
                except Exception as e:
                    logger.error(f'[AUTO_SETUP] Image switch failed: {e}')
                    result.warnings.append(f'Image switch to {recommended_image} failed: {e}')
                    # replace_runtime already destroyed the old container — runtime is dead.
                    # Skip steps 3-4 (they would all fail with network errors).
                    _switch_failed = True
        else:
            logger.debug(f'[AUTO_SETUP] Image already matches: {recommended_image}')
            result.selected_image = current_image
    else:
        logger.debug('[AUTO_SETUP] No image switch needed')

    # ---- Step 3: Probe patches ----
    if _switch_failed:
        logger.warning('[AUTO_SETUP] Skipping steps 3-4 (runtime dead after failed image switch)')
        result.warnings.append('Steps 3-4 skipped — runtime unavailable after image switch failure')
    else:
        logger.debug('[AUTO_SETUP] Step 3: Probing patches...')
        try:
            probe_result = await _probe_patches(runtime, task_config)
            result.golden_patch_status = probe_result['golden_status']
            result.golden_patch_detail = probe_result['golden_detail']
            result.test_patch_status = probe_result['test_status']
            result.test_patch_detail = probe_result['test_detail']
            result.test_patch_applied_on = probe_result['test_applied_on']
            result.eval_strategy = probe_result['eval_strategy']
        except Exception as e:
            logger.error(f'[AUTO_SETUP] probe_patches failed: {e}')
            result.warnings.append(f'probe_patches failed: {e}')
        logger.debug(
            '[AUTO_SETUP] Step 3 done — golden=%s, test=%s, strategy=%s',
            result.golden_patch_status,
            result.test_patch_status,
            result.eval_strategy,
        )

    if not _switch_failed:
        # ---- Step 3.5: Configure mirrors before pre-install ----
        # Mirror setup (pip/npm/go/cargo/maven/gem/composer/nuget) is normally done when the
        # first bash terminal session is created. But _pre_install uses run_command (direct
        # subprocess), not terminal sessions, so mirrors must be configured explicitly here.
        logger.debug('[AUTO_SETUP] Step 3.5: Configuring package mirrors...')
        try:
            from nexus.extensions.tasks.swe_builder.runtime_utils import run_command as _mirror_run_cmd

            await runtime.invoke(_mirror_run_cmd, **{'command': _MIRROR_SETUP, 'cwd': '/testbed', 'timeout': 120})
            logger.debug('[AUTO_SETUP] Mirror setup complete')
        except Exception as e:
            logger.warning(f'[AUTO_SETUP] Mirror setup failed (pre-install may be slow): {e}')
            result.warnings.append(f'mirror setup failed: {e}')

        # ---- Step 4: Pre-install ----
        logger.debug('[AUTO_SETUP] Step 4: Running pre-install...')
        try:
            commands_run, summary = await _pre_install(runtime, language, explore_data)
            result.install_commands_run = commands_run
            result.install_summary = summary
            # Add warnings for failed/timeout/oom installs
            for entry in commands_run:
                status = entry['status']
                if status in ('failed', 'error'):
                    result.warnings.append(f'install failed: {entry["cmd"][:80]} — {entry.get("error", "")[:100]}')
                elif status == 'timeout':
                    result.warnings.append(
                        f'install timeout (stalled): {entry["cmd"][:80]} — killed after {entry["duration_s"]}s'
                    )
                elif status == 'oom':
                    result.warnings.append(f'install OOM killed: {entry["cmd"][:80]} — process killed (exit 137)')
        except Exception as e:
            logger.error(f'[AUTO_SETUP] pre_install failed: {e}')
            result.warnings.append(f'pre_install failed: {e}')
            result.install_summary = f'error: {e}'

    result.setup_duration_s = round(time.monotonic() - t0, 1)
    _warn_count = len(result.warnings)
    logger.debug('=' * 80)
    logger.debug('[AUTO_SETUP] ✅ Pre-build setup complete in %.1fs', result.setup_duration_s)
    logger.debug('[AUTO_SETUP]   Language : %s %s', result.language, result.language_version)
    logger.debug(
        '[AUTO_SETUP]   Image    : %s%s', result.selected_image[-60:], ' (switched)' if result.image_switched else ''
    )
    logger.debug(
        '[AUTO_SETUP]   Patches  : golden=%s, test=%s → strategy=%s',
        result.golden_patch_status,
        result.test_patch_status,
        result.eval_strategy,
    )
    logger.debug('[AUTO_SETUP]   Install  : %s', result.install_summary or 'no commands')
    if _warn_count:
        logger.info('[AUTO_SETUP]   ⚠️  Warnings: %d — agent Phase 1 will fix', _warn_count)
    else:
        logger.debug('[AUTO_SETUP]   Warnings : none — agent skips to Phase 2')
    logger.debug('=' * 80)

    return result


# ============== Bash Tool (Both Phases) ==============


class BashInput(BaseModel):
    """Input for bash command execution."""

    command: str = Field(description='The bash command to execute.')


@tool_registry.register(
    'swe_setup_agent/bash',
    meta={
        'name': 'bash',
        'description': (
            'Execute a bash command in the runtime environment. '
            'Use this to run shell commands, navigate directories, search files, '
            'run installation scripts, check versions, and execute any shell-based operations. '
            'The command runs in a persistent terminal session, maintaining state across calls. '
            'Exit codes are available in the response metadata to check command success.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def bash(params: BashInput, context: ToolContext) -> ToolResponse:
    """Execute a bash command in the runtime environment."""
    session_id = context.tool_state.get('terminal_session_id')
    context_updates = {}

    # Helper function to initialize session
    async def init_session(new_session_id: str) -> None:
        """Initialize a new terminal session with environment and working directory."""
        # Set up environment variables and mirrors
        await run_command_in_session(
            RunCommandInSessionRequest(
                session_id=new_session_id,
                command='export PAGER=cat MANPAGER=cat LESS=-R PIP_PROGRESS_BAR=off TQDM_DISABLE=1\n'
                'export CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse\n'
                'export CARGO_HTTP_MULTIPLEXING=false\n' + _MIRROR_SETUP,
                timeout=300,
            )
        )

        working_dir = get_working_dir(context)
        await run_command_in_session(
            RunCommandInSessionRequest(
                session_id=new_session_id,
                command=f'cd {working_dir}',
                timeout=300,
            )
        )

    # Create new session if not exists
    if session_id is None:
        logger.debug('[BASH] No session found, creating new terminal session')
        # Retry terminal creation if there's a conflict
        max_retries = 3
        for attempt in range(max_retries):
            try:
                create_response = await create_terminal_session(CreateTerminalSessionRequest())
                session_id = create_response.session_id
                logger.debug(f'[BASH] Created new session: {session_id}')
                break
            except Exception as e:
                if attempt < max_retries - 1 and (
                    'already running' in str(e).lower() or 'failed to start' in str(e).lower()
                ):
                    logger.warning(
                        f'[BASH] Terminal creation conflict (attempt {attempt + 1}/{max_retries}), retrying after delay...'
                    )
                    await asyncio.sleep(2.0 * (attempt + 1))  # Exponential backoff
                else:
                    raise

        # Save session_id BEFORE init_session so it survives an init failure.
        # If init raises, the session still exists in TERMINAL_SESSIONS and the
        # saved ID lets the next bash call reuse (or destroy) it cleanly.
        assert session_id is not None, 'Failed to create terminal session after all retries'
        context_updates['terminal_session_id'] = session_id
        await init_session(session_id)

    # At this point session_id is guaranteed to be a valid string
    assert session_id is not None

    # Execute command with retry on errors
    max_cmd_retries = 3
    last_error = None

    for attempt in range(max_cmd_retries):
        try:
            t0 = time.monotonic()
            run_response = await run_command_in_session(
                RunCommandInSessionRequest(
                    session_id=session_id,
                    command=params.command,
                    timeout=1800,
                )
            )
            elapsed = time.monotonic() - t0

            exit_code = getattr(run_response, 'exit_code', None)
            output_str = _compress_bash_output(
                params.command,
                run_response.output,
                exit_code if exit_code is not None else 0,
            )

            # Add timing header for long-running commands (>30s)
            if elapsed > 30:
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                time_str = f'{mins}m{secs}s' if mins > 0 else f'{secs}s'
                timing_note = f'[Command completed in {time_str}, exit_code={exit_code}]\n'
                output_str = timing_note + output_str

            return ToolResponse(
                output=_make_output(output_str),
                context_updates=context_updates,
                meta={
                    'command': params.command,
                    'exit_code': exit_code,
                    'elapsed_seconds': round(elapsed, 1),
                },
            )

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Handle "session not found" - recreate session
            if 'session' in error_str and 'not found' in error_str:
                logger.warning(f'[BASH] Session {session_id} not found, recreating...')
                try:
                    create_response = await create_terminal_session(CreateTerminalSessionRequest())
                    session_id = create_response.session_id
                    logger.debug(f'[BASH] Recreated session: {session_id}')
                    context_updates['terminal_session_id'] = session_id  # save before init
                    await init_session(session_id)
                    # Retry the command with new session
                    continue
                except Exception as create_error:
                    logger.error(f'[BASH] Failed to recreate session: {create_error}')
                    if attempt >= max_cmd_retries - 1:
                        break

            # Handle "already running" — send C-c to unblock; on final retry, kill and recreate session
            if 'already running' in error_str or 'failed to start' in error_str:
                logger.warning(
                    f'[BASH] Command execution conflict (attempt {attempt + 1}/{max_cmd_retries}), sending C-c...'
                )
                try:
                    await send_keys_to_session(SendKeysToSessionRequest(session_id=session_id, keys='C-c'))
                    await asyncio.sleep(1.0)
                    await send_keys_to_session(SendKeysToSessionRequest(session_id=session_id, keys='C-c'))
                    await asyncio.sleep(1.0)
                except Exception:
                    pass

                if attempt < max_cmd_retries - 1:
                    await asyncio.sleep(2.0 * (attempt + 1))
                    continue

                # Final attempt: kill stale session and create a fresh one
                logger.warning('[BASH] Persistent "already running" — killing stale session and recreating...')
                try:
                    create_response = await create_terminal_session(CreateTerminalSessionRequest())
                    session_id = create_response.session_id
                    logger.debug(f'[BASH] Recreated session after stale lock: {session_id}')
                    context_updates['terminal_session_id'] = session_id  # save before init
                    await init_session(session_id)
                    continue
                except Exception as recreate_err:
                    logger.error(f'[BASH] Failed to recreate session: {recreate_err}')
                break
            break

    # Handle the error (from last attempt)
    if last_error:
        e = last_error
        error_message = 'Error: ' + str(e)
        error_str = str(e).lower()

        if 'failed to start' in error_str or 'already running' in error_str:
            try:
                await send_keys_to_session(SendKeysToSessionRequest(session_id=session_id, keys='C-c'))
                await asyncio.sleep(0.5)
            except Exception:
                pass

            try:
                pane_response = await capture_pane(CapturePaneRequest(session_id=session_id))
                error_message = (
                    'Your bash command failed to start after multiple retries. '
                    'A Ctrl+C has been sent to recover the shell. '
                    'Please fix the syntax errors and try again. This can be the result '
                    'of not adhering to the syntax for multi-line commands. Here is the captured shell content:\n'
                    + pane_response.content
                )
            except Exception:
                pass

        return ToolResponse(
            output=_make_output(error_message),
            context_updates=context_updates,
        )
    # Unreachable: loop always returns on success, and last_error is set on failure
    return ToolResponse(output=_make_output('Error: unexpected state'), context_updates=context_updates)


# ============== String Replace Editor Tool (Both Phases) ==============

_file_editors: dict[str, FileEditor] = {}


def _get_file_editor(context: ToolContext) -> FileEditor:
    """Get or create a FileEditor instance, isolated per run_id to prevent cross-task state pollution."""
    working_dir = context.tool_state.get('working_dir', '/testbed')
    # Use run_id as key to ensure each concurrent task has its own FileEditor instance
    run_id = context.run_id or 'default'
    key = f'{run_id}:{working_dir}'
    if key not in _file_editors:
        _file_editors[key] = FileEditor(workspace_root=working_dir)
    return _file_editors[key]


class StrReplaceEditorInput(BaseModel):
    """Input for the string replacement editor."""

    command: Literal['view', 'create', 'str_replace', 'insert', 'undo_edit'] = Field(
        description='The command to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.'
    )
    path: str = Field(description='Absolute path to file or directory, e.g. `/testbed/file.py` or `/tmp/eval.sh`.')
    file_text: str | None = Field(
        default=None,
        description='Required parameter of `create` command, with the content of the file to be created.',
    )
    old_str: str | None = Field(
        default=None,
        description='Required parameter of `str_replace` command containing the string in `path` to replace.',
    )
    new_str: str | None = Field(
        default=None,
        description=(
            'Optional parameter of `str_replace` command containing the new string '
            '(if not given, no string will be added). Required parameter of `insert` command '
            'containing the string to insert.'
        ),
    )
    insert_line: int | None = Field(
        default=None,
        description=(
            'Required parameter of `insert` command. '
            'The `new_str` will be inserted AFTER the line `insert_line` of `path`.'
        ),
    )
    view_range: list[int] | None = Field(
        default=None,
        description=(
            'Optional parameter of `view` command when `path` points to a file. '
            'If provided, the file will be shown in the indicated line number range, e.g. [11, 20] will show lines 11 and 12. '
            'Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end.'
        ),
    )


@tool_registry.register(
    'swe_setup_agent/str_replace_editor',
    meta={
        'name': 'str_replace_editor',
        'description': (
            'Custom editing tool for viewing, creating and editing files. '
            'State is persistent across command calls. '
            'Use `view` to read files (shows line numbers) or list directory contents. '
            'Use `create` to create new files - use `/tmp` for scripts (eval.sh) and `/testbed` for code. '
            'Use `str_replace` to edit existing files by replacing exact string matches. '
            'Use `insert` to add new lines at specific positions. '
            'Use `undo_edit` to revert the last edit made to a file.\n\n'
            'Notes for using the `str_replace` command:\n'
            '* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. '
            'Be mindful of whitespaces!\n'
            '* If the `old_str` parameter is not unique in the file, the replacement will not be performed. '
            'Make sure to include enough context in `old_str` to make it unique.\n'
            '* The `new_str` parameter should contain the edited lines that should replace the `old_str`.\n\n'
            'If a `view` command generates a long output, it will be truncated and marked with `<response clipped>`. '
            'The `create` command cannot be used if the specified `path` already exists as a file.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def str_replace_editor(params: StrReplaceEditorInput, context: ToolContext) -> ToolResponse:
    """String replacement editor for file operations."""
    working_dir = get_working_dir(context)

    path_str = params.path
    if not path_str.startswith('/'):
        path_str = str(PurePosixPath(working_dir) / path_str)

    editor = _get_file_editor(context)

    try:
        observation = await asyncio.to_thread(
            editor,
            command=params.command,
            path=path_str,
            file_text=params.file_text,
            view_range=params.view_range,
            old_str=params.old_str,
            new_str=params.new_str,
            insert_line=params.insert_line,
        )

        output_text = observation.text

        if observation.image_base64 and observation.image_mime_type:
            return ToolResponse(
                output=[
                    {'type': 'text', 'text': output_text},
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:{observation.image_mime_type};base64,{observation.image_base64}'},
                    },
                ],
                meta={
                    'command': observation.command,
                    'path': observation.path,
                },
            )

        return ToolResponse(
            output=_make_output(output_text),
            meta={
                'command': observation.command,
                'path': observation.path,
            },
        )

    except ToolError as e:
        return ToolResponse(
            output=_make_output(f'Error: {e.message}'),
            meta={'error': str(e)},
        )
    except Exception as e:
        return ToolResponse(
            output=_make_output(f'Error: {e}'),
            meta={'error': str(e)},
        )


# ============== Switch Git State Tool (Both Phases) ==============


class SwitchGitStateInput(BaseModel):
    """Input for switch_git_state tool."""

    target: Literal['base', 'full'] = Field(
        description='Target state: base=base_commit only (clean), full=base_commit+golden_patch'
    )


async def switch_git_state(params: SwitchGitStateInput, context: ToolContext) -> ToolResponse:
    """Switch testbed git state, injecting golden_patch from task_config for full state."""
    ctx = require_context()
    runtime = await ctx.get_runtime('main')

    testbed_dir = ctx.task_config.get('testbed_dir', '/testbed')
    base_commit = ctx.task_config['base_commit']
    golden_patch_content = ctx.task_config.get('golden_patch', '')

    patches_applied = []

    logger.debug(f'[SWITCH_GIT] Switching to {params.target} state')

    from nexus.extensions.tasks.swe_builder.runtime_utils import git_reset_and_clean

    # 1. Reset to base_commit
    reset_result = await runtime.invoke(git_reset_and_clean, **{'testbed_dir': testbed_dir, 'commit': base_commit})

    if reset_result['exit_code'] != 0:
        logger.error(f'[SWITCH_GIT] Git reset failed: {reset_result["stderr"]}')
        return ToolResponse(
            output=_make_output(f'Git reset failed: {reset_result["stderr"]}'),
            meta={'success': False, 'current_state': 'unknown', 'patches_applied': []},
        )

    # 2. For 'full' state: apply golden_patch
    # Note: test_patch is NOT applied here — eval.sh applies it itself via heredoc.
    if params.target == 'full' and golden_patch_content.strip():
        from nexus.extensions.tasks.swe_builder.runtime_utils import git_apply_patch_from_content

        apply_result = await runtime.invoke(
            git_apply_patch_from_content, **{'content': golden_patch_content, 'testbed_dir': testbed_dir}
        )
        if apply_result['exit_code'] == 0:
            patches_applied.append('golden_patch')
            logger.debug('[SWITCH_GIT] Applied golden_patch')
        else:
            logger.warning(f'[SWITCH_GIT] Failed to apply golden_patch: {apply_result["stderr"]}')
            # Store stderr so _validate_fail2pass can surface it to the agent
            return ToolResponse(
                output=_make_output('golden_patch failed to apply'),
                meta={
                    'success': True,
                    'current_state': params.target,
                    'patches_applied': [],
                    'golden_patch_apply_error': (
                        apply_result.get('stderr') or apply_result.get('stdout') or '(no output)'
                    ).strip(),
                },
            )

    logger.debug(
        f'[SWITCH_GIT] ✅ Switched to {params.target} state (applied: {", ".join(patches_applied) if patches_applied else "none"})'
    )

    return ToolResponse(
        output=_make_output(
            f'Successfully switched to {params.target} state (applied: {", ".join(patches_applied) if patches_applied else "none"})'
        ),
        meta={
            'success': True,
            'current_state': params.target,
            'patches_applied': patches_applied,
        },
    )


# ============== Switch Runtime Image Tool ==============


class SwitchRuntimeImageInput(BaseModel):
    """Input for switch_runtime_image tool."""

    new_image: str = Field(description='New base image name (e.g., python:3.11-slim, node:18-bullseye)')
    reason: str = Field(default='', description='Reason for switching (e.g., "Need Python 3.11 pre-installed")')


@tool_registry.register(
    'swe_setup_agent/switch_runtime_image',
    meta={
        'name': 'switch_runtime_image',
        'description': (
            'Switch runtime to a new base image when current image cannot meet requirements.\n\n'
            'WARNING:\n'
            '- Destroys current runtime container\n'
            '- Conversation history is preserved (in host process)\n'
            '- Testbed will be reinitialized (need to reinstall dependencies)\n\n'
            'Use this when:\n'
            '- Current base image lacks essential system dependencies\n'
            '- Need a language-specific image with pre-installed environment\n\n'
            'Max switches: 3'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def switch_runtime_image(params: SwitchRuntimeImageInput, context: ToolContext) -> ToolResponse:
    """Switch runtime to a new base image."""
    ctx = require_context()

    # Check switch limit
    max_switches = ctx.task_config.get('max_base_image_switches', 3)
    switch_count = ctx.tool_state.get('base_image_switch_count', 0)

    if switch_count >= max_switches:
        raise RuntimeError(
            f'Reached maximum base image switch limit ({max_switches}). '
            f'Consider using ubuntu-git:22.04 and manually install required tools.'
        )

    # Get old image name from task_config
    old_image = ctx.task_config.get('base_image', 'unknown')

    # Validate image name before attempting to pull (avoid 1800s timeout on non-existent images)
    if not _validate_image(params.new_image):
        from nexus.extensions.tasks.swe_builder.constants import format_available_images_list

        language = ctx.task_config.get('language', '')
        available = format_available_images_list(language)
        logger.warning(f'[SWITCH_RUNTIME] Image {params.new_image!r} not in known registry, rejecting switch')
        return ToolResponse(
            output=_make_output(
                f'Image "{params.new_image}" does not exist in the registry. '
                f'Current image kept: {old_image}\n\n'
                f'Available images:\n{available}\n\n'
                f'Pick an image from the list above and call switch_runtime_image again.'
            ),
            meta={'success': False, 'error': 'invalid_image', 'requested': params.new_image},
        )

    # Resolve short image name to mirror registry URL if needed.
    # Gongfeng sandboxes can only pull from the internal mirror registry; they have
    # no access to docker.io/ghcr.io.  The initial base_image is always the full
    # mirror URL (e.g. mirrors.tencent.com/hunyuan_yanguan/t3build_repolaunch_midtrain
    # .swe_builder_eval_base_ubuntu_git_22_04:latest).  We derive the registry prefix
    # from that URL and translate short names like "python-git:3.11-slim" the same way
    # the generator's _mirror_image() does.
    resolved_image = _resolve_mirror_image(params.new_image, old_image)

    logger.debug('=' * 80)
    logger.debug('[SWITCH_RUNTIME] Switching base image')
    logger.debug(f'[SWITCH_RUNTIME]   Old: {old_image}')
    logger.debug(f'[SWITCH_RUNTIME]   New (requested): {params.new_image}')
    logger.debug(f'[SWITCH_RUNTIME]   New (resolved):  {resolved_image}')
    logger.debug(f'[SWITCH_RUNTIME]   Reason: {params.reason}')
    logger.debug(f'[SWITCH_RUNTIME]   Remaining: {max_switches - switch_count - 1}')
    logger.debug('=' * 80)

    # Replace runtime
    new_runtime = await ctx.replace_runtime('main', image=resolved_image)

    # Update counter and current base_image (store resolved name for next switch)
    ctx.tool_state['base_image_switch_count'] = switch_count + 1
    ctx.task_config['base_image'] = resolved_image

    # Clear stale terminal session ID — the old container is gone, its tmux sessions no longer exist.
    # This ensures the next bash call creates a fresh session directly instead of hitting
    # "session not found" → retry → potential "failed to start" race on the new container's tmux.
    ctx.tool_state.pop('terminal_session_id', None)

    # Reinitialize testbed
    await _reinitialize_testbed(ctx, new_runtime)

    # Record new runtime_id in span for tracing
    from nexus.trace import get_current_span_id, update_span

    new_runtime_id = ctx.get_runtime_id('main') or 'unknown'
    if new_runtime_id != 'unknown':
        logger.debug(f'[SWITCH_RUNTIME] Recorded new runtime_id: {new_runtime_id}')
        span_id = get_current_span_id()
        if span_id:
            update_span(span_id, attributes={'hytuner.runtime_id': new_runtime_id})

    logger.debug('=' * 80)
    logger.debug('[SWITCH_RUNTIME] ✅ Runtime switched successfully')
    logger.debug(f'[SWITCH_RUNTIME]   New base_url: {getattr(getattr(new_runtime, "config", None), "base_url", "N/A")}')
    logger.debug('[SWITCH_RUNTIME]   Testbed reinitialized')
    logger.debug('=' * 80)

    message = (
        f'Runtime switched to {resolved_image}. '
        f'Testbed reinitialized, previous installations lost. '
        f'Remaining switches: {max_switches - switch_count - 1}'
    )

    return ToolResponse(
        output=_make_output(message),
        meta={
            'success': True,
            'old_image': old_image,
            'new_image': resolved_image,
            'remaining_switches': max_switches - switch_count - 1,
        },
    )


async def _reinitialize_testbed(ctx, runtime):
    """Reinitialize testbed after runtime switch."""
    from nexus.extensions.tasks.swe_builder.runtime_utils import (
        git_clone,
        git_remove_remote,
        git_reset_and_clean,
    )

    repo = ctx.task_config['repo']
    base_commit = ctx.task_config['base_commit']
    testbed_dir = ctx.task_config['testbed_dir']

    logger.debug('[REINIT] Reinitializing testbed')

    # 1. Clone
    repo_url = f'https://github.com/{repo}'
    clone_result = await runtime.invoke(git_clone, **{'repo_url': repo_url, 'dest': testbed_dir})

    if clone_result['exit_code'] != 0:
        raise RuntimeError(f'Git clone failed: {clone_result["stderr"]}')

    # 2. Reset
    reset_result = await runtime.invoke(git_reset_and_clean, **{'testbed_dir': testbed_dir, 'commit': base_commit})

    if reset_result['exit_code'] != 0:
        raise RuntimeError(f'Git reset failed: {reset_result["stderr"]}')

    # 3. Remove remote
    await runtime.invoke(git_remove_remote, **{'testbed_dir': testbed_dir})

    # 4. Re-write /tmp patch files — replace_runtime destroys the old container,
    # so /tmp/golden_patch.patch and /tmp/test_patch.patch are gone.
    # Both _probe_patches (auto_setup Step 3) and agent bash commands need them.
    from nexus.extensions.tasks.swe_builder.runtime_utils import write_file as _write_file

    golden_patch = ctx.task_config.get('golden_patch', '')
    test_patch = ctx.task_config.get('test_patch', '')
    if golden_patch.strip():
        await runtime.invoke(_write_file, **{'path': '/tmp/golden_patch.patch', 'content': golden_patch})
    if test_patch.strip():
        await runtime.invoke(_write_file, **{'path': '/tmp/test_patch.patch', 'content': test_patch})

    logger.debug('[REINIT] ✅ Testbed reinitialized (patch files restored)')


# ============== Submit Eval Tool ==============


def _postprocess_eval_script(content: str) -> str:
    """Post-process eval.sh to fix common issues.

    Aligns with t3build offline evaluator (_postprocess_eval_script in dataset_handler.py).
    Fixes applied:
      1. Remove `set -u` (nounset) — unbound variable errors when heredoc parsing leaves
         variables unset due to special characters in test names.
      2. Inject GRADLE_OPTS/MAVEN_OPTS proxy config so JVM builds reach external repos through proxy
         (JVM ignores https_proxy env var — needs -Dhttps.proxyHost system property).
      3. Inject CARGO_HTTP_MULTIPLEXING=false to avoid flaky Cargo downloads behind proxies.
    (conda/Nix activation is skipped — our images use venv/rbenv/nvm, not conda.)
    """
    # 1. Remove -u from set flags: "set -uxo pipefail" → "set -xo pipefail"
    content = re.sub(r'^(set\s+-\w*)u(\w*)', r'\1\2', content, flags=re.MULTILINE)
    # Clean up bare "set -" left when -u was the only flag: "set -u" → "set -xo pipefail"
    content = re.sub(r'^set\s+-\s*$', 'set -xo pipefail', content, flags=re.MULTILINE)
    content = re.sub(r'^set\s+-\s+', 'set ', content, flags=re.MULTILINE)

    # 2. Inject JVM proxy for Gradle/Maven builds (JVM ignores https_proxy env var)
    java_proxy_snippet = (
        'if [ -n "$https_proxy" ]; then'
        " PROXY_HOST=$(echo $https_proxy | sed 's|http://||;s|https://||' | cut -d: -f1);"
        " PROXY_PORT=$(echo $https_proxy | sed 's|http://||;s|https://||' | cut -d: -f2);"
        ' _JVM_PROXY="-Dhttps.proxyHost=$PROXY_HOST -Dhttps.proxyPort=$PROXY_PORT'
        ' -Dhttp.proxyHost=$PROXY_HOST -Dhttp.proxyPort=$PROXY_PORT";'
        ' export GRADLE_OPTS="${_JVM_PROXY} ${GRADLE_OPTS:-}";'
        ' export MAVEN_OPTS="${_JVM_PROXY} ${MAVEN_OPTS:-}";'
        ' fi\n'
    )
    needs_jvm_proxy = ('gradlew' in content or 'mvn' in content or 'gradle ' in content) and '_JVM_PROXY' not in content
    if needs_jvm_proxy:
        content = re.sub(
            r'^(cd\s+/testbed.*\n)',
            r'\1' + java_proxy_snippet,
            content,
            count=1,
            flags=re.MULTILINE,
        )

    # 3. Inject CARGO_HTTP_MULTIPLEXING=false for Cargo builds (prevents flaky downloads behind proxies)
    if 'cargo' in content and 'CARGO_HTTP_MULTIPLEXING' not in content:
        content = re.sub(
            r'^(cd\s+/testbed.*\n)',
            r'\1' + 'export CARGO_HTTP_MULTIPLEXING=false\n',
            content,
            count=1,
            flags=re.MULTILINE,
        )

    return content


class SubmitEvalInput(BaseModel):
    """Input for submit_eval tool."""

    eval_script_path: str = Field(
        default='/tmp/eval.sh', description='Path to targeted eval script (fail2pass verification)'
    )
    explanation: str = Field(description='Explanation: why this eval script can reproduce fail2pass')


async def _ensure_crane(runtime, run_cmd) -> str:
    """Download crane binary to /tmp/crane_bin/ if not already present.

    /tmp/crane_bin is excluded from the tar layer, so crane won't
    appear in committed images. /nix/ is read-only in Gongfeng sandbox.
    Returns the path to the crane binary.
    """
    crane_path = '/tmp/crane_bin/crane'

    # Check if crane is already available
    check = await runtime.invoke(
        run_cmd,
        **{
            'command': f'test -x {crane_path} && {crane_path} version 2>&1 | head -1',
            'cwd': '/tmp',
            'timeout': 10,
        },
    )
    if check.get('exit_code', -1) == 0:
        logger.debug('[CRANE] Already available at %s', crane_path)
        return crane_path

    # Download crane
    logger.debug('[CRANE] Downloading crane binary...')
    crane_url = 'https://github.com/google/go-containerregistry/releases/download/v0.20.3/go-containerregistry_Linux_x86_64.tar.gz'
    download_cmd = (
        f'mkdir -p /tmp/crane_bin && '
        f'curl -fsSL {crane_url} | tar xz -C /tmp/crane_bin crane && '
        f'chmod +x {crane_path} && '
        f'{crane_path} version'
    )
    dl_result = await runtime.invoke(
        run_cmd,
        **{
            'command': download_cmd,
            'cwd': '/tmp',
            'timeout': 120,
        },
    )
    if dl_result.get('exit_code', -1) != 0:
        stderr = dl_result.get('stderr', '')[:300]
        stdout = dl_result.get('stdout', '')[:300]
        raise RuntimeError(f'Failed to download crane: {stderr} {stdout}')

    logger.debug('[CRANE] Downloaded successfully: %s', dl_result.get('stdout', '').strip()[:100])
    return crane_path


async def _commit_and_push_via_sandbox(
    runtime,
    instance_id: str,
    eval_content: str,
    test_patch: str,
    task_config: dict,
) -> dict:
    """Commit and push image from INSIDE the sandbox via crane (no Docker daemon needed).

    Used when the runtime provider doesn't support host-side Docker operations
    (e.g., GongfengRuntimeProvider on K8s where there is no Docker daemon).

    Flow:
      1. Ensure crane binary is available at /nix/runtime/tmp/crane
      2. crane auth login to registry
      3. Create diff layer tarball (filesystem changes only)
      4. crane append -b <base_image> -f <layer.tar.gz> -t <remote_tag>
      5. Cleanup layer tarball

    Returns:
        Dict with 'uploaded', 'remote_image', 'metadata' keys.
    """
    from datetime import datetime

    from nexus.extensions.tasks.swe_builder.runtime_utils import run_command

    # Registry constants (duplicated from upload_handler to avoid importing docker SDK)
    _REGISTRY_HOST = 'mirrors.tencent.com'
    _REGISTRY_PREFIX = 'mirrors.tencent.com/hunyuan_yanguan/t3build_repolaunch_midtrain'
    _REGISTRY_USERNAME_DEFAULT = ''
    _REGISTRY_PASSWORD_DEFAULT = ''

    # Build remote image tag
    safe_id = instance_id.lower().replace('/', '_').replace('-', '_')
    local_image_name = f'swe_builder_eval_prs_{safe_id}:latest'
    remote_tag = f'{_REGISTRY_PREFIX}.{local_image_name}'

    registry_username = os.getenv('REGISTRY_USERNAME', _REGISTRY_USERNAME_DEFAULT)
    registry_password = os.getenv('REGISTRY_PASSWORD', _REGISTRY_PASSWORD_DEFAULT)

    base_image = task_config.get('base_image', '')
    if not base_image:
        raise RuntimeError('No base_image in task_config — cannot determine crane append base')

    # 1. Ensure crane binary
    crane = await _ensure_crane(runtime, run_command)

    # 2. crane auth login
    logger.debug('[CRANE] Authenticating to %s', _REGISTRY_HOST)
    login_result = await runtime.invoke(
        run_command,
        **{
            'command': f'{crane} auth login {_REGISTRY_HOST} -u {registry_username} -p {registry_password}',
            'cwd': '/tmp',
            'timeout': 30,
        },
    )
    if login_result.get('exit_code', -1) != 0:
        raise RuntimeError(f'crane auth login failed: {login_result.get("stderr", "")[:200]}')

    # 3. Create diff layer tarball
    # Exclude: /proc, /sys, /dev (virtual filesystems — files change size during tar, causing exit 2),
    # /nix (volume mount), /tmp/crane_bin (crane binary + layer itself), .dockerenv
    layer_path = '/tmp/crane_bin/layer.tar.gz'
    logger.debug('[CRANE] Creating diff layer tarball...')
    tar_cmd = (
        f'tar czf {layer_path} '
        f'-C / '
        f'--exclude=./proc --exclude=./sys --exclude=./dev '
        f'--exclude=./nix --exclude=./tmp/crane_bin '
        f'--exclude=./.dockerenv '
        f'.'
    )
    tar_result = await runtime.invoke(
        run_command,
        **{
            'command': tar_cmd,
            'cwd': '/tmp',
            'timeout': 1800,
        },
    )
    # tar may return exit code 1 for "file changed as we read it" — that's OK
    tar_exit = tar_result.get('exit_code', -1)
    if tar_exit not in (0, 1):
        raise RuntimeError(f'tar failed (exit {tar_exit}): {tar_result.get("stderr", "")[:300]}')

    # Check layer size
    size_result = await runtime.invoke(
        run_command,
        **{
            'command': f'ls -lh {layer_path} | awk "{{print \\$5}}"',
            'cwd': '/tmp',
            'timeout': 10,
        },
    )
    layer_size = size_result.get('stdout', '').strip()
    logger.debug('[CRANE] Layer tarball size: %s', layer_size)

    # 4. crane append — append diff layer to base image and push to registry (with retry)
    logger.debug('[CRANE] Appending layer to base %s -> %s', base_image, remote_tag)
    append_cmd = f'{crane} append -b {base_image} -f {layer_path} -t {remote_tag}'
    _max_retries = 3
    for _attempt in range(1, _max_retries + 1):
        append_result = await runtime.invoke(
            run_command,
            **{
                'command': append_cmd,
                'cwd': '/tmp',
                'timeout': 1200,
            },
        )
        if append_result.get('exit_code', -1) == 0:
            break
        stderr = append_result.get('stderr', '')[:500]
        if _attempt < _max_retries:
            _backoff = _attempt * 10
            logger.warning(
                '[CRANE] Append attempt %d/%d failed, retrying in %ds: %s',
                _attempt,
                _max_retries,
                _backoff,
                stderr[:200],
            )
            await asyncio.sleep(_backoff)
        else:
            raise RuntimeError(f'crane append failed after {_max_retries} attempts: {stderr}')
    logger.info('[CRANE] Pushed successfully: %s', remote_tag)

    # 5. Cleanup layer tarball (best-effort, in /nix volume so no image impact)
    await runtime.invoke(
        run_command,
        **{
            'command': f'rm -f {layer_path}',
            'cwd': '/tmp',
            'timeout': 10,
        },
    )

    # 6. Generate metadata (inline, avoids importing upload_handler which requires docker SDK)
    upload_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    metadata = {
        'instance_id': instance_id,
        'language': task_config.get('language', 'python'),
        'repo': task_config.get('repo', ''),
        'base_commit': task_config.get('base_commit', ''),
        'remote_image': remote_tag,
        'upload_time': upload_time,
        'eval_script_content': eval_content,
        'test_patch': test_patch,
    }

    return {
        'uploaded': True,
        'remote_image': remote_tag,
        'metadata': metadata,
    }


@tool_registry.register(
    'swe_setup_agent/submit_eval',
    meta={
        'name': 'submit_eval',
        'description': (
            'Submit eval script, validate fail2pass, and export environment image.\n\n'
            'Workflow (all automatic):\n'
            '1. Static anti-hacking checks\n'
            '2. Run full state (base + golden_patch): eval.sh should PASS\n'
            '   → If FAILS: environment/setup issue — returns detailed output for diagnosis\n'
            '3. Run base state (base only): eval.sh should FAIL\n'
            '   → If PASSES: test does not reproduce the bug (pass2pass) — returns guidance\n'
            '4. LLM anti-hacking validation (only runs if fail2pass passes)\n'
            '5. Export and upload image\n\n'
            '/testbed is reset to base_commit after every call (success or failure).\n'
            'Up to 30 submit_eval attempts per trajectory.\n\n'
            'On failure: read the detailed output, make ONE targeted fix, resubmit.\n'
            'After successful submission, the task is complete.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def submit_eval(params: SubmitEvalInput, context: ToolContext) -> ToolResponse:
    """Submit eval script, validate fail2pass, and export image."""
    from nexus.extensions.tasks.swe_builder.constants import TaskPhase

    ctx = require_context()

    # debug mode comes from task_config only — NOT from LLM-supplied params
    debug: bool = ctx.task_config.get('debug', False)

    # Idempotency check: if already submitted, return success immediately
    if ctx.tool_state.get('task_phase', 0) >= TaskPhase.EVAL_SUBMITTED:
        logger.debug('[SUBMIT_EVAL] Already submitted successfully, skipping duplicate submission')
        return ToolResponse(
            output=_make_output('✅ Eval script already submitted successfully. Task is done.'),
            meta={'success': True, 'already_submitted': True},
        )

    _MAX_SUBMIT_ATTEMPTS = ctx.task_config.get('max_submit_attempts', 30)
    submit_attempt = ctx.tool_state.get('submit_eval_count', 0) + 1
    ctx.tool_state['submit_eval_count'] = submit_attempt
    logger.debug(f'[SUBMIT_EVAL] Attempt {submit_attempt}/{_MAX_SUBMIT_ATTEMPTS}')
    if submit_attempt > _MAX_SUBMIT_ATTEMPTS:
        logger.error(f'[SUBMIT_EVAL] Exceeded max attempts ({_MAX_SUBMIT_ATTEMPTS}), marking trajectory as failed')
        return ToolResponse(
            output=[
                {
                    'type': 'text',
                    'text': (
                        f'❌ submit_eval has been called {submit_attempt - 1} times and still has not passed validation. '
                        f'Maximum attempts ({_MAX_SUBMIT_ATTEMPTS}) exceeded. This trajectory is marked as failed. '
                        'Do NOT retry — the task will be terminated.'
                    ),
                }
            ],
            meta={'success': False, 'error': 'max_attempts_exceeded', 'attempts': submit_attempt - 1},
        )

    runtime = await ctx.get_runtime('main')

    # Get test_patch status for LLM validation context
    test_patch = ctx.task_config.get('test_patch', '')
    has_test_patch = test_patch.strip() != ''

    # 1. Read eval script — try the given path first, then search common alternatives
    from nexus.extensions.tasks.swe_builder.runtime_utils import read_file

    eval_content = ''
    eval_script_actual_path = params.eval_script_path
    eval_script_source = 'provided'  # 'provided' | 'fallback_search'

    try:
        eval_content = await runtime.invoke(read_file, **{'path': params.eval_script_path})
    except Exception:
        eval_content = ''

    if not eval_content or not eval_content.strip():
        # Fallback: search common alternative paths where agent might have written the eval script
        _fallback_paths = [
            '/tmp/eval.sh',
            '/tmp/eval_script.sh',
            '/tmp/eval_final.sh',
            '/tmp/eval_script_final.sh',
            '/tmp/final_eval.sh',
            '/root/eval.sh',
            '/root/eval_script.sh',
            '/home/eval.sh',
            '/home/eval_script.sh',
            '/testbed/eval.sh',
        ]
        # Remove the already-tried path
        _fallback_paths = [p for p in _fallback_paths if p != params.eval_script_path]

        for alt_path in _fallback_paths:
            try:
                alt_content = await runtime.invoke(read_file, **{'path': alt_path})
                if alt_content and alt_content.strip():
                    logger.warning(
                        '[SUBMIT_EVAL] Eval script not found at %s, found at fallback path %s',
                        params.eval_script_path,
                        alt_path,
                    )
                    eval_content = alt_content
                    eval_script_actual_path = alt_path
                    eval_script_source = 'fallback_search'
                    break
            except Exception:
                continue

    if not eval_content or not eval_content.strip():
        logger.error(f'[SUBMIT_EVAL] Eval script is empty at {params.eval_script_path} (fallback search also failed)')
        return ToolResponse(
            output=[
                {
                    'type': 'text',
                    'text': f'❌ Eval script at {params.eval_script_path} is empty\n\nPlease write a valid eval script.',
                }
            ],
            meta={'success': False, 'error': 'empty_file'},
        )

    # Track the actual path and source for diagnostics
    ctx.tool_state['eval_script_actual_path'] = eval_script_actual_path
    ctx.tool_state['eval_script_source'] = eval_script_source
    if eval_script_source == 'fallback_search':
        logger.info(
            '[SUBMIT_EVAL] Using eval from fallback path: %s (agent provided: %s)',
            eval_script_actual_path,
            params.eval_script_path,
        )

    # 1.5 Post-process eval.sh to fix common issues (aligns with t3build offline evaluator):
    #   - Remove set -u (nounset)
    #   - Inject GRADLE_OPTS proxy for Gradle builds
    #   - Inject CARGO_HTTP_MULTIPLEXING=false for Cargo builds
    from nexus.extensions.tasks.swe_builder.runtime_utils import write_file as _write_file_pre

    processed_content = _postprocess_eval_script(eval_content)
    if processed_content != eval_content:
        logger.debug('[SUBMIT_EVAL] Post-processed eval.sh (set-u removal / env injection)')
        await runtime.invoke(_write_file_pre, **{'path': eval_script_actual_path, 'content': processed_content})
        eval_content = processed_content

    # 1.6 Static check: reject if eval.sh references /tmp/test_patch.* or /tmp/golden_patch.*
    # These files are written for agent convenience during setup but deleted before commit.
    # eval.sh must embed test_patch as a heredoc (not reference these files).
    # Referencing /tmp/golden_patch.* in eval.sh would be cheating (applying the fix in eval.sh).
    for bad_ref in [r'/tmp/test_patch\.(diff|patch)', r'/tmp/golden_patch\.(diff|patch)']:
        if re.search(bad_ref, eval_content):
            fname_match = re.search(bad_ref, eval_content)
            fname = fname_match.group(0) if fname_match else bad_ref
            logger.warning(f'[SUBMIT_EVAL] Static check failed: eval.sh references {fname}')
            return ToolResponse(
                output=[
                    {
                        'type': 'text',
                        'text': (
                            f'❌ Eval script references `{fname}` — this file is intentionally deleted before eval.sh runs '
                            f'and will not exist during offline validation.\n\n'
                            f'eval.sh must be self-contained and must NOT reference /tmp/golden_patch.patch or /tmp/test_patch.patch.\n\n'
                            f'To apply test_patch in eval.sh, embed the patch content directly as a heredoc:\n\n'
                            '```bash\n'
                            "git apply --ignore-whitespace << 'PATCH_EOF'\n"
                            '<paste test_patch content here>\n'
                            'PATCH_EOF\n'
                            '```\n\n'
                            f'The test_patch content is available in your prompt above (or at /tmp/test_patch.patch during setup, before submit_eval).\n'
                            f'**Fix**: replace the `{fname}` reference with the inline heredoc.'
                        ),
                    }
                ],
                meta={
                    'success': False,
                    'error': 'patch_file_reference',
                    'llm_check': {
                        'approved': False,
                        'reason': f'References {fname}',
                        'issues': [f'{fname} referenced'],
                    },
                },
            )

    # 1.7 Static check: reject if eval.sh has a literal "OMNIGRIL_EXIT_CODE=0" or "OMNIGRIL_EXIT_CODE=1"
    # without capturing a real test runner exit code via rc=$? first.
    # Pattern: echo "OMNIGRIL_EXIT_CODE=<digit>" NOT preceded by rc=$? anywhere in the script.
    # This catches the case where the agent just hardcodes echo "OMNIGRIL_EXIT_CODE=0" to always pass.
    _omnigril_hardcode_re = re.compile(r'echo\s+["\']?OMNIGRIL_EXIT_CODE=[0-9]+["\']?\s*$', re.MULTILINE)
    if _omnigril_hardcode_re.search(eval_content) and 'rc=$?' not in eval_content:
        logger.warning('[SUBMIT_EVAL] Static check failed: hardcoded OMNIGRIL_EXIT_CODE without rc=$?')
        return ToolResponse(
            output=[
                {
                    'type': 'text',
                    'text': (
                        '❌ Eval script has a hardcoded `echo "OMNIGRIL_EXIT_CODE=0"` without capturing '
                        'a real test runner exit code.\n\n'
                        'The correct pattern is:\n'
                        '```bash\n'
                        '<run your test command>\n'
                        'rc=$?\n'
                        'echo "OMNIGRIL_EXIT_CODE=$rc"\n'
                        'exit $rc\n'
                        '```\n\n'
                        '**Fix**: replace the hardcoded value with `rc=$?` captured from the actual test runner.'
                    ),
                }
            ],
            meta={
                'success': False,
                'error': 'hardcoded_exit_code',
                'llm_check': {
                    'approved': False,
                    'reason': 'Hardcoded OMNIGRIL_EXIT_CODE without rc=$?',
                    'issues': ['hardcoded exit code'],
                },
            },
        )

    # 1.8 Pre-check: run eval.sh on base state to catch pass2pass early.
    # This is much cheaper than the full fail2pass validation (no golden_patch apply needed).
    # If eval.sh PASSES on base (exit_code=0, OMNIGRIL_EXIT_CODE=0), it's pass2pass — reject immediately.
    logger.debug('[SUBMIT_EVAL] Pre-check: running eval.sh on base state...')
    from nexus.extensions.tasks.swe_builder.runtime_utils import (
        run_command as _pre_run_command,
    )
    from nexus.extensions.tasks.swe_builder.runtime_utils import (
        run_script as _pre_run_script,
    )

    _OMNIGRIL_RE_PRE = re.compile(r'OMNIGRIL_EXIT_CODE=(\d+)')

    try:
        # Ensure base state
        from nexus.extensions.tasks.swe_builder.runtime_utils import git_reset_and_clean as _pre_reset

        _pre_testbed = ctx.task_config.get('testbed_dir', '/testbed')
        _pre_base_commit = ctx.task_config['base_commit']
        await runtime.invoke(_pre_reset, **{'testbed_dir': _pre_testbed, 'commit': _pre_base_commit})

        # Stop services before pre-check (must match _STOP_SERVICES_CMD in _validate_fail2pass)
        _pre_stop = (
            'service mysql stop 2>/dev/null; '
            'service postgresql stop 2>/dev/null; '
            'service redis-server stop 2>/dev/null; '
            'service mongod stop 2>/dev/null; '
            'service elasticsearch stop 2>/dev/null; '
            'pkill -f "mysqld|postgres|redis-server|mongod" 2>/dev/null; '
            'true'
        )
        await runtime.invoke(_pre_run_command, **{'command': _pre_stop, 'timeout': 10})  # type: ignore[no-matching-overload]

        # Run eval.sh on base
        _language = ctx.task_config.get('language', '').lower()
        _pre_timeout = 1800 if _language in ('c', 'c++', 'rust', 'java', 'c#') else 600
        pre_result = await runtime.invoke(  # type: ignore[no-matching-overload]
            _pre_run_script, **{'script_path': eval_script_actual_path, 'timeout': _pre_timeout}
        )
        pre_combined = pre_result.get('stdout', '') + pre_result.get('stderr', '')
        pre_omnigril = _OMNIGRIL_RE_PRE.search(pre_combined)
        pre_exit = int(pre_omnigril.group(1)) if pre_omnigril else None

        if pre_exit == 0:
            # Base state passes → pass2pass, reject immediately
            logger.warning('[SUBMIT_EVAL] Pre-check: base state PASSED (pass2pass) — skipping full validation')
            # Track in history
            history_list = ctx.tool_state.get('submit_eval_history', [])
            history_list.append(f'{submit_attempt}: base_state_passes')
            ctx.tool_state['submit_eval_history'] = history_list

            # Reset testbed
            await runtime.invoke(_pre_reset, **{'testbed_dir': _pre_testbed, 'commit': _pre_base_commit})

            detail = (
                f'## submit_eval FAILED: base_state_passes (attempt {submit_attempt}/{_MAX_SUBMIT_ATTEMPTS})\n\n'
                f'### Root Cause\neval.sh PASSES on base_commit (OMNIGRIL_EXIT_CODE=0) — no fail2pass signal.\n\n'
                f'### Suggested Fix\nWrite a custom test from golden_patch that exercises the specific code change '
                f'(fails on base, passes with fix). Do NOT use test_patch heredoc.\n\n'
                f'### History\n[{", ".join(history_list[-10:])}]'
            )
            return ToolResponse(
                output=[{'type': 'text', 'text': detail}],
                meta={'success': False, 'failure_type': 'base_state_passes', 'pre_check': True},
            )

        # Check OMNIGRIL_EXIT_CODE exists
        if pre_omnigril is None:
            logger.warning('[SUBMIT_EVAL] Pre-check: OMNIGRIL_EXIT_CODE not found in base-state output')
            # Not a hard failure — proceed to full validation which may still succeed

        logger.info(f'[SUBMIT_EVAL] Pre-check passed: base state FAILS as expected (OMNIGRIL={pre_exit})')
    except Exception as e:
        logger.warning(f'[SUBMIT_EVAL] Pre-check failed (non-fatal, proceeding to full validation): {e}')

    # 2. Remove /tmp patch files before running eval.sh.
    # These files were written at task init for agent convenience but must NOT be present
    # during validation (prevents eval.sh from referencing them to cheat) and must NOT be
    # committed to the image (offline validation runs in a fresh docker container from the image).
    from nexus.extensions.tasks.swe_builder.runtime_utils import remove_files
    from nexus.extensions.tasks.swe_builder.runtime_utils import write_file as _write_file

    _patch_contents: dict[str, str] = {}
    _golden_patch_content = ctx.task_config.get('golden_patch', '')
    _test_patch_content = ctx.task_config.get('test_patch', '')
    if _golden_patch_content.strip():
        _patch_contents['/tmp/golden_patch.patch'] = _golden_patch_content
    if _test_patch_content.strip():
        _patch_contents['/tmp/test_patch.patch'] = _test_patch_content
    if _patch_contents:
        await runtime.invoke(remove_files, **{'paths': list(_patch_contents.keys())})  # type: ignore[no-matching-overload]
        logger.debug(f'[SUBMIT_EVAL] Removed /tmp patch files before validation: {list(_patch_contents.keys())}')

    # 3. Validate fail2pass — run before LLM validation to avoid wasting an LLM call
    # on the common case where eval.sh has bugs (wrong test path, missing env, etc.).
    # LLM validation only runs if fail2pass passes.
    logger.debug('[SUBMIT_EVAL] Starting fail2pass validation')
    validation_result = await _validate_fail2pass(
        ctx=ctx, runtime=runtime, eval_script_path=eval_script_actual_path, context=context
    )

    if not validation_result['is_fail2pass']:
        # Special case: test_patch is empty AND both states pass → pass2pass.
        # This occurs for refactoring/optimization issues where there is no observable
        # behavioral difference between base and fixed states.  We still upload the image
        # so the environment is preserved, but the task is NOT marked as resolved.
        if not has_test_patch and validation_result.get('pass2pass_detected', False):
            logger.info(
                '[SUBMIT_EVAL] pass2pass case detected (empty test_patch, both states pass) — uploading image without resolved flag'
            )
            ctx.tool_state['pass2pass_only'] = True
        else:
            logger.warning(f'[SUBMIT_EVAL] Fail2pass validation failed: {validation_result["reason"]}')
            # Reset /testbed to base_commit so agent can continue cleanly.
            from nexus.extensions.tasks.swe_builder.runtime_utils import git_reset_and_clean

            _testbed_dir = ctx.task_config.get('testbed_dir', '/testbed')
            _base_commit = ctx.task_config['base_commit']
            await runtime.invoke(git_reset_and_clean, **{'testbed_dir': _testbed_dir, 'commit': _base_commit})
            logger.debug('[SUBMIT_EVAL] /testbed reset to base_commit after fail2pass failure')
            # Restore /tmp patch files so agent can continue using them for debugging
            for _p, _c in _patch_contents.items():
                try:
                    await runtime.invoke(_write_file, **{'path': _p, 'content': _c})
                except Exception as _e:
                    logger.warning(f'[SUBMIT_EVAL] Failed to restore {_p}: {_e}')
            if _patch_contents:
                logger.debug(f'[SUBMIT_EVAL] Restored /tmp patch files for agent use: {list(_patch_contents.keys())}')
            # Build structured error message (<1500 chars target)
            full_res = validation_result.get('full_result') or {}
            test_res = validation_result.get('test_result') or {}
            reason = validation_result['reason']

            # Classify failure type and extract root cause
            failure_type = 'unknown'
            root_cause = ''
            suggested_fix = ''

            full_out = full_res.get('stdout', '') + full_res.get('stderr', '')
            test_out = (test_res.get('stdout', '') + test_res.get('stderr', '')) if test_res else ''
            combined_out = full_out + test_out

            # Check for /tmp patch file reference
            if re.search(r'/tmp/(golden_patch|test_patch)\.(patch|diff).*[Nn]o such file', combined_out):
                failure_type = 'eval_script_references_patch'
                root_cause = 'eval.sh references /tmp patch files (deleted before eval runs)'
                suggested_fix = 'Embed test_patch as heredoc in eval.sh instead of referencing /tmp files'

            elif 'golden_patch could not be applied' in reason or 'golden_patch failed to apply' in reason:
                failure_type = 'golden_patch_apply_failed'
                apply_err = validation_result.get('full_result', {}).get('stderr', reason)[:300]
                root_cause = apply_err
                if 'already exists' in apply_err.lower():
                    suggested_fix = 'install created files in /testbed — use --ignore-scripts (npm) or BUNDLE_PATH=/root/bundle (bundler)'
                else:
                    suggested_fix = 'resubmit (auto-retries with --ignore-whitespace and patch --fuzz=5)'

            elif 'full state' in reason and 'did not pass' in reason:
                failure_type = 'full_state_fails'
                # Extract first error from output
                error_lines = [
                    l
                    for l in full_out.splitlines()
                    if re.search(
                        r'Error|FAIL|Cannot|Unable|not found|denied|fatal|Traceback|ModuleNotFoundError|command not found',
                        l,
                        re.IGNORECASE,
                    )
                ]
                root_cause = error_lines[0][:300] if error_lines else full_out[-300:]
                # Classify error type for targeted suggestion
                rc_lower = root_cause.lower()
                if 'modulenotfounderror' in rc_lower or 'importerror' in rc_lower:
                    pkg = re.search(r"No module named '([^']+)'", root_cause)
                    pkg_name = pkg.group(1) if pkg else '<package>'
                    suggested_fix = f'pip install {pkg_name} in eval.sh before test command'
                elif 'command not found' in rc_lower:
                    suggested_fix = 'source runtime env in eval.sh (nvm/venv/cargo/bundle) or apt-get install the tool'
                elif 'no tests ran' in rc_lower or '0 items' in rc_lower or 'collected 0' in rc_lower:
                    suggested_fix = 'wrong test path or filter — fix the test command'
                elif 'build' in rc_lower and ('fail' in rc_lower or 'error' in rc_lower):
                    suggested_fix = 'add build step before test in eval.sh'
                elif 'connection refused' in rc_lower or 'could not connect' in rc_lower:
                    suggested_fix = 'start required service in eval.sh (e.g. service postgresql start; sleep 2)'
                else:
                    suggested_fix = 'read full_state output, fix the first error in eval.sh'

            elif 'base state passed' in reason or 'pass2pass' in reason:
                failure_type = 'base_state_passes'
                root_cause = 'test passes on base_commit — no fail2pass signal'
                suggested_fix = 'write custom test from golden_patch changes (fails on base, passes with fix)'

            else:
                failure_type = 'other'
                root_cause = reason[:300]
                suggested_fix = 'read the error above, make ONE targeted fix'

            # Track submit_eval history in tool_state
            history_list = ctx.tool_state.get('submit_eval_history', [])
            history_list.append(f'{submit_attempt}: {failure_type}')
            ctx.tool_state['submit_eval_history'] = history_list
            history_str = ', '.join(history_list[-10:])  # last 10 entries

            # Build compact structured output
            detail = (
                f'## submit_eval FAILED: {failure_type} (attempt {submit_attempt}/{_MAX_SUBMIT_ATTEMPTS})\n\n'
                f'### Root Cause\n{root_cause}\n\n'
                f'### Suggested Fix\n{suggested_fix}\n\n'
                f'### History\n[{history_str}]'
            )

            # Append truncated output only for full_state_fails (most actionable)
            if failure_type == 'full_state_fails' and full_out:
                # Show last 500 chars of output for debugging
                tail = full_out[-500:]
                detail += f'\n\n### Output (last 500 chars)\n```\n{tail}\n```'

            # Hard cap at 1500 chars
            if len(detail) > 1500:
                detail = detail[:1400] + '\n... (truncated)'

            return ToolResponse(
                output=[{'type': 'text', 'text': detail}],
                meta={
                    'success': False,
                    'failure_type': failure_type,
                    'validation_result': validation_result,
                    'workdir_clean': False,
                    'image_exported': False,
                },
            )
    else:
        logger.info('[SUBMIT_EVAL] ✅ Fail2pass validation passed')

    # 4. LLM validation — runs only after fail2pass passes (avoids wasting LLM calls on broken evals)
    logger.debug('[SUBMIT_EVAL] Starting LLM validation')
    llm_check = await _llm_validate_eval_script(
        ctx=ctx, eval_content=eval_content, explanation=params.explanation, has_test_patch=has_test_patch
    )

    if not llm_check['approved']:
        logger.warning(f'[SUBMIT_EVAL] LLM validation failed: {llm_check["reason"]}')
        # Restore patch files so agent can continue debugging
        for _p, _c in _patch_contents.items():
            try:
                await runtime.invoke(_write_file, **{'path': _p, 'content': _c})
            except Exception as _e:
                logger.warning(f'[SUBMIT_EVAL] Failed to restore {_p}: {_e}')
        return ToolResponse(
            output=[
                {
                    'type': 'text',
                    'text': (
                        f'❌ eval.sh REJECTED — {llm_check["reason"]}\n\n'
                        '**Action required — call str_replace_editor NOW to rewrite /tmp/eval.sh:**\n'
                        '1. Open /tmp/eval.sh with str_replace_editor\n'
                        '2. Replace the invalid test logic with a real test runner invocation '
                        '(pytest / cargo test / go test / dotnet test / npm test / ctest / etc.)\n'
                        '3. Call submit_eval immediately after writing the file — do NOT output analysis first\n\n'
                        'Common fixes:\n'
                        "- grep on source files → replace with the project's actual test command\n"
                        '- build-only (cmake/make/dotnet build) → add the test runner step after the build\n'
                        '- || true on test runner → remove it so failures are visible\n'
                    ),
                }
            ],
            meta={
                'success': False,
                'validation_result': {'llm_check': llm_check},
                'workdir_clean': False,
                'image_exported': False,
            },
        )

    logger.info('[SUBMIT_EVAL] ✅ LLM validation passed')

    # 4. Check workdir clean — use a pure git reset (no test_patch injection) so that
    # test_patch's new files don't appear as untracked and falsely fail the clean check.
    # This also ensures the committed image contains only the installed environment,
    # not test_patch files.
    logger.debug('[SUBMIT_EVAL] Checking workdir status')
    from nexus.extensions.tasks.swe_builder.runtime_utils import git_reset_and_clean, git_status

    testbed_dir = ctx.task_config['testbed_dir']
    base_commit = ctx.task_config['base_commit']
    await runtime.invoke(git_reset_and_clean, **{'testbed_dir': testbed_dir, 'commit': base_commit})

    workdir_status = await runtime.invoke(git_status, **{'testbed_dir': testbed_dir})

    is_clean = workdir_status.get('is_clean', False)

    if not is_clean:
        logger.warning('[SUBMIT_EVAL] Working directory is not clean')
        # Restore patch files so agent can continue debugging
        for _p, _c in _patch_contents.items():
            try:
                await runtime.invoke(_write_file, **{'path': _p, 'content': _c})
            except Exception as _e:
                logger.warning(f'[SUBMIT_EVAL] Failed to restore {_p}: {_e}')
        message = (
            f'Working directory is not clean:\n'
            f'  Untracked: {workdir_status.get("untracked_files", [])}\n'
            f'  Modified: {workdir_status.get("modified_files", [])}\n'
            f'  Staged: {workdir_status.get("staged_files", [])}\n'
            f'Please clean up testbed before submitting.'
        )
        return ToolResponse(
            output=_make_output(message),
            meta={
                'success': False,
                'validation_result': validation_result,
                'workdir_clean': False,
                'image_exported': False,
            },
        )

    logger.debug('[SUBMIT_EVAL] ✅ Working directory is clean')

    # 5. Delete patch files, eval.sh, and install.sh from runtime before commit.
    # /tmp/golden_patch.patch and /tmp/test_patch.patch were already removed before validation
    # (step 2.5), but delete again here as a safety measure in case they were restored.
    # These files must NOT appear in the committed image — offline validation runs in a
    # fresh docker container and eval.sh must be self-contained without these files.
    logger.debug('[SUBMIT_EVAL] Cleaning up /tmp files before image commit')
    from nexus.extensions.tasks.swe_builder.runtime_utils import remove_files

    files_to_remove = [
        eval_script_actual_path,
        '/tmp/install.sh',
        '/tmp/golden_patch.patch',
        '/tmp/test_patch.patch',
    ]

    await runtime.invoke(remove_files, **{'paths': files_to_remove})  # type: ignore[no-matching-overload]

    # 7. Export and upload image
    instance_id = ctx.task_config['instance_id']
    local_image_name = f'swe_builder_eval_prs_{instance_id.lower().replace("/", "_").replace("-", "_")}:latest'
    remote_image = ''

    if not debug:
        _has_local_docker = hasattr(ctx.runtime_provider, '_user_containers')

        if _has_local_docker:
            # ── Path A: Host-side Docker SDK (local Docker provider) ──────
            logger.debug('[SUBMIT_EVAL] Committing and uploading via host Docker SDK')
            from nexus.extensions.tasks.swe_builder.image_tools import (
                CommitRuntimeImageInput,
                commit_runtime_image,
            )

            # Cache copy-out + clean before commit (best-effort)
            try:
                from nexus.extensions.tasks.swe_builder.cache_manager import copy_out_and_clean

                _cname = getattr(ctx.runtime_provider, '_user_containers', {}).get('main')
                if _cname:
                    await copy_out_and_clean(_cname)
            except Exception as _e:
                logger.warning(f'[CACHE] copy-out skipped: {_e}')

            await commit_runtime_image(
                CommitRuntimeImageInput(
                    runtime_name='main', image_name=local_image_name, message=f'Solved environment for {instance_id}'
                ),
                context,
            )

            logger.debug(f'[SUBMIT_EVAL] ✅ Image committed: {local_image_name}')

            from nexus.extensions.tasks.swe_builder.upload_handler import upload_swe_builder_image

            task_output_mock = type(
                'obj',
                (object,),
                {
                    'resolved': True,
                    'score': 1.0,
                    'instance_with_results': {
                        'instance_id': instance_id,
                        'language': ctx.task_config.get('language', 'python'),
                        'repo': ctx.task_config.get('repo', ''),
                        'base_commit': ctx.task_config.get('base_commit', ''),
                        'problem_statement': ctx.task_config.get('problem_statement', ''),
                        'patch': ctx.task_config.get('golden_patch', ''),
                        'test_patch': test_patch,
                        'eval_script_content': eval_content,
                        'test_results': {'fail2pass': [], 'pass2pass': []},
                    },
                },
            )()

            upload_result = await upload_swe_builder_image(
                instance_id=instance_id,
                image_name=local_image_name,
                task_output=task_output_mock,
                upload_strategy='all_success',
                cleanup_local=True,
            )

            if upload_result.get('uploaded', False):
                remote_image = upload_result.get('remote_image', '')
                logger.info(f'[SUBMIT_EVAL] ✅ Image uploaded: {remote_image}')
            else:
                logger.error(f'[SUBMIT_EVAL] ❌ Image upload failed: {upload_result.get("reason", "unknown")}')
                message = f'Image upload failed: {upload_result.get("reason", "unknown")}'
                return ToolResponse(
                    output=_make_output(message),
                    meta={
                        'success': False,
                        'validation_result': validation_result,
                        'workdir_clean': True,
                        'image_exported': True,
                        'image_name': local_image_name,
                    },
                )
        else:
            # ── Path B: crane append inside sandbox (remote provider, no Docker daemon) ─
            logger.debug('[SUBMIT_EVAL] Committing and uploading via crane (remote provider, no Docker daemon)')
            try:
                upload_result = await _commit_and_push_via_sandbox(
                    runtime,
                    instance_id,
                    eval_content,
                    test_patch,
                    ctx.task_config,
                )
                if upload_result.get('uploaded', False):
                    remote_image = upload_result.get('remote_image', '')
                    logger.info(f'[SUBMIT_EVAL] ✅ Image committed and uploaded via sandbox: {remote_image}')
                else:
                    reason = upload_result.get('reason', 'unknown')
                    logger.error(f'[SUBMIT_EVAL] ❌ Sandbox commit/push failed: {reason}')
                    message = f'Image upload failed (sandbox): {reason}'
                    return ToolResponse(
                        output=_make_output(message),
                        meta={
                            'success': False,
                            'validation_result': validation_result,
                            'workdir_clean': True,
                            'image_exported': False,
                            'image_name': local_image_name,
                        },
                    )
            except Exception as _commit_err:
                logger.error(f'[SUBMIT_EVAL] ❌ Sandbox commit/push error: {_commit_err}')
                message = f'Image commit/push failed (sandbox): {_commit_err}'
                return ToolResponse(
                    output=_make_output(message),
                    meta={
                        'success': False,
                        'validation_result': validation_result,
                        'workdir_clean': True,
                        'image_exported': False,
                        'image_name': local_image_name,
                    },
                )
    else:
        logger.debug('[SUBMIT_EVAL] [DEBUG] Skipping image commit and upload in debug mode')

    # 8. Save eval script to host
    output_dir = ctx.task_config.get('output_dir', './outputs/swe_builder')
    instance_output_dir = os.path.join(output_dir, instance_id)
    eval_output_path = os.path.join(instance_output_dir, 'eval.sh')

    # Run file operations in thread pool to avoid blocking
    def _save_eval_script():
        os.makedirs(instance_output_dir, exist_ok=True)
        with open(eval_output_path, 'w') as f:
            f.write(eval_content)

    await asyncio.to_thread(_save_eval_script)
    logger.info(f'[SUBMIT_EVAL] ✅ Eval script saved: {eval_output_path}')

    # 9. Update task state
    pass2pass_only = ctx.tool_state.get('pass2pass_only', False)
    ctx.tool_state['task_phase'] = TaskPhase.EVAL_SUBMITTED
    ctx.tool_state['eval_script_path'] = eval_output_path
    ctx.tool_state['eval_script_content'] = eval_content
    ctx.tool_state['exported_image'] = remote_image if not debug else local_image_name
    ctx.tool_state['image_uploaded'] = bool(remote_image) if not debug else False
    ctx.tool_state['eval_llm_validation'] = llm_check
    ctx.tool_state['eval_fail2pass_validation'] = validation_result

    if pass2pass_only:
        message = (
            f'✅ Eval script submitted (pass2pass: tests pass in both states, likely refactoring/optimization issue)\n'
            f'✅ Image exported: {remote_image if not debug else local_image_name + " (local only)"}\n'
            f'✅ Eval script saved: {eval_output_path}\n'
            f'ℹ️  Task image uploaded but NOT marked as resolved (no fail2pass behavior detected).'
        )
    else:
        message = (
            f'✅ Eval script submitted and validated\n'
            f'✅ Image exported: {remote_image if not debug else local_image_name + " (local only)"}\n'
            f'✅ Eval script saved: {eval_output_path}\n'
            f'✅ Task complete.'
        )

    return ToolResponse(
        output=_make_output(message),
        meta={
            'success': True,
            'validation_result': validation_result,
            'workdir_clean': True,
            'image_exported': True,
            'image_name': local_image_name,
            'remote_image': remote_image,
            'eval_script_saved': eval_output_path,
        },
    )


async def _llm_validate_eval_script(ctx, eval_content: str, explanation: str, has_test_patch: bool) -> dict:
    """Use LLM to validate eval script."""
    client = ctx.get_openai_client('main')

    test_patch_note = (
        'test_patch is NON-EMPTY — eval.sh should apply it via `git apply` heredoc when possible. '
        'If the agent determined that test_patch tests are skipped or do not expose the bug, '
        'it may write custom test logic instead (without the heredoc). Both approaches are valid.'
        if has_test_patch
        else 'test_patch is EMPTY — eval.sh must contain its own inline test logic to verify the fix.'
    )

    prompt = f"""Check if the following eval script is valid.

**Eval Script:**
```bash
{eval_content}
```

**Agent's Explanation:**
{explanation}

**Context:** {test_patch_note}

**Reject ONLY for these specific patterns:**

1. **grep/awk/sed on source code files as the SOLE "test"** — if the script ONLY uses grep/awk/sed
   to inspect source code text (e.g. `grep -q "newFunc" src/file.py`) WITHOUT invoking any test
   runner or test binary, REJECT it. This is source text inspection, not a behavioral test.
   - grep/awk/sed on test runner OUTPUT is fine (e.g. `pytest ... 2>&1 | grep PASSED`)
   - grep used alongside a real test runner is fine
   - Only reject when grep/awk/sed is the ONLY mechanism determining pass/fail

2. **No real test runner invoked** — the script must invoke at least one real test runner
   (pytest, cargo test, go test, npm test, make test, ctest, dotnet test, vendor/bin/phpunit,
   bundle exec rspec, ./gradlew test, mvn test, etc.) or a compiled test binary.
   A pure build check (cmake/make compile with no test step) is NOT sufficient.

3. **`|| true` on the test runner** — `|| true` directly after the test command hides failures.
   `|| true` on setup/build/install steps is acceptable.

**Do NOT reject for anything else** — output format, build approach, heredoc style, set -x,
rebuild location, golden_patch not being in eval.sh (it is pre-applied), test calling functions
added by golden_patch, etc. Only the 3 rules above matter.

Output JSON:
```json
{{
    "approved": true/false,
    "reason": "brief reason"
}}
```
"""

    response = await client._create_completion(messages=[{'role': 'user', 'content': prompt}])

    content = response.choices[0].message.content
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)

    result = None

    # Try to parse the JSON block
    if json_match:
        try:
            json_str = re.sub(r',\s*([}\]])', r'\1', json_match.group(1).strip())
            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f'[LLM_VALIDATE] Failed to parse JSON block: {e}')

    # Fallback: try raw content
    if result is None:
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            logger.warning('[LLM_VALIDATE] Could not parse LLM response, assuming approved')
            result = {'approved': True, 'reason': 'Could not parse LLM response, assuming approved'}

    result.setdefault('issues', [])
    return result


async def _validate_fail2pass(ctx, runtime, eval_script_path: str, context: ToolContext) -> dict:
    """Validate fail2pass behavior.

    switch_git_state injects golden_patch for the full state. eval.sh applies test_patch
    itself via heredoc (embedded patch content).

    Full state:  base + golden_patch → eval.sh applies test_patch → should PASS
    Base state:  base only           → eval.sh applies test_patch → should FAIL
    """
    from nexus.extensions.tasks.swe_builder.runtime_utils import run_command, run_script

    logger.debug('[VALIDATE] Starting fail2pass validation')

    # Stop any services the agent may have started (MySQL, PostgreSQL, Redis, etc.).
    # The final eval pipeline runs eval.sh in a FRESH container from the committed image —
    # services are not running there.  Stopping them here makes the in-container validation
    # faithfully reproduce the clean-container environment.
    _STOP_SERVICES_CMD = (
        'service mysql stop 2>/dev/null; '
        'service postgresql stop 2>/dev/null; '
        'service redis-server stop 2>/dev/null; '
        'service mongod stop 2>/dev/null; '
        'service elasticsearch stop 2>/dev/null; '
        'pkill -f "mysqld|postgres|redis-server|mongod" 2>/dev/null; '
        'true'
    )

    # Timeout per eval.sh run.
    # C/C++/Rust/Java/C#: 1800s — these languages require compilation before running tests.
    #   C/C++: cmake-configure + full make can take 10-20 min on large projects.
    #   Rust:  cargo test compiles the test binary from scratch on each run (~5-15 min for large crates).
    #   Java:  Maven/Gradle full build + test can exceed 600s on multi-module projects.
    #   C#:    dotnet build + test on multi-target projects can exceed 600s.
    # All others: 600s — Ruby/JS/Go/PHP/Python compile fast enough.
    _language = ctx.task_config.get('language', '').lower()
    EVAL_RUN_TIMEOUT = 1800 if _language in ('c', 'c++', 'rust', 'java', 'c#') else 600

    # 1. Full state: base + golden_patch → eval.sh applies test_patch → should PASS
    full_switch_result = await switch_git_state(SwitchGitStateInput(target='full'), context)

    # Verify golden_patch was actually applied (silent failures cause full==base, masking F2P)
    golden_patch_content = ctx.task_config.get('golden_patch', '')
    if golden_patch_content.strip() and 'golden_patch' not in full_switch_result.meta.get('patches_applied', []):
        logger.error('[VALIDATE] golden_patch failed to apply in full state — cannot run F2P validation')
        apply_error = full_switch_result.meta.get('golden_patch_apply_error', '(no detail)')
        return {
            'is_fail2pass': False,
            'pass2pass_detected': False,
            'full_result': {'exit_code': -1, 'stdout': '', 'stderr': 'golden_patch failed to apply'},
            'test_result': None,
            'reason': (
                'Fail: golden_patch could not be applied (git apply failed).\n\n'
                f'git apply error:\n{apply_error}\n\n'
                + (
                    '⚠️  IMPORTANT: This error occurred when applying golden_patch BEFORE eval.sh was run.\n'
                    'Your eval.sh was NOT executed — modifying eval.sh will NOT fix this error.\n\n'
                    + (
                        'Cause: install.sh created a file inside /testbed that golden_patch also tries to create.\n'
                        'The file already exists in a gitignored path and survives git clean -fd.\n'
                        'Fix: ensure install.sh does NOT create source/config files inside /testbed.\n'
                        '  - npm: use --ignore-scripts\n'
                        '  - bundler: use BUNDLE_PATH=/root/bundle\n'
                        '  - Any other postinstall that writes to /testbed — remove it\n'
                        'Then call submit_eval again.'
                        if 'already exists' in str(apply_error).lower()
                        # "patch does not apply" / "hunk FAILED" → data quality, just retry
                        else (
                            'DATA QUALITY: golden_patch context lines do not match base_commit.\n'
                            'Do NOT modify eval.sh or install.sh — this error is unrelated to your script.\n'
                            '→ Call submit_eval again immediately (retries use --ignore-whitespace fallback, then patch --fuzz=5).\n'
                            '   If the same error appears 30 times in a row, this instance is unfixable — stop.\n\n'
                            'One thing that CAN help: if you have not tried switch_runtime_image yet, switching\n'
                            'to the appropriate image re-clones /testbed, which sometimes resolves context drift.'
                            if any(
                                kw in str(apply_error).lower()
                                for kw in (
                                    'does not apply',
                                    'hunk failed',
                                    'patch failed',
                                    'trailing whitespace',
                                    'corrupt patch',
                                )
                            )
                            else 'Cause unclear — read the git apply error above.\n'
                            'Likely causes:\n'
                            '1. install.sh created or modified files in /testbed (run: git -C /testbed status to check)\n'
                            '2. Data quality — patch context does not match base_commit\n'
                            '→ Call submit_eval again. If same error 5× in a row, check install.sh for /testbed writes.'
                        )
                    )
                )
            ),
        }

    await runtime.invoke(run_command, **{'command': _STOP_SERVICES_CMD, 'timeout': 15})
    logger.debug('[VALIDATE] Services stopped before full-state run')
    full_result = await runtime.invoke(run_script, **{'script_path': eval_script_path, 'timeout': EVAL_RUN_TIMEOUT})

    _OMNIGRIL_RE = re.compile(r'OMNIGRIL_EXIT_CODE=(\d+)')

    def _parse_omnigril(result: dict) -> int | None:
        combined = result.get('stdout', '') + result.get('stderr', '')
        m = _OMNIGRIL_RE.search(combined)
        return int(m.group(1)) if m else None

    full_omnigril = _parse_omnigril(full_result)
    full_passed = full_omnigril == 0

    logger.info(f'[VALIDATE] Full state exit_code={full_result["exit_code"]} OMNIGRIL_EXIT_CODE={full_omnigril}')

    # Early return if full state fails — no need to run base state.
    # Full state failure = environment/setup issue (golden_patch can't fix a broken env).
    if not full_passed:
        combined = full_result.get('stdout', '') + full_result.get('stderr', '')
        logger.warning(f'[VALIDATE] Full state FAILED — skipping base-state run. Output tail:\n{combined[-500:]}')
        omnigril_note = (
            f'OMNIGRIL_EXIT_CODE={full_omnigril}'
            if full_omnigril is not None
            else 'OMNIGRIL_EXIT_CODE not found in output'
        )
        return {
            'is_fail2pass': False,
            'pass2pass_detected': False,
            'full_result': full_result,
            'test_result': None,
            'reason': f'Fail: full state (base+golden_patch) did not pass ({omnigril_note}) — likely environment/setup issue',
        }

    # 2. Base state: base only → eval.sh applies test_patch → should FAIL
    await switch_git_state(SwitchGitStateInput(target='base'), context)
    await runtime.invoke(run_command, **{'command': _STOP_SERVICES_CMD, 'timeout': 15})
    logger.debug('[VALIDATE] Services stopped before base-state run')
    test_result = await runtime.invoke(run_script, **{'script_path': eval_script_path, 'timeout': EVAL_RUN_TIMEOUT})

    test_omnigril = _parse_omnigril(test_result)
    test_failed = (test_omnigril is None) or (test_omnigril != 0)

    logger.info(f'[VALIDATE] Base state exit_code={test_result["exit_code"]} OMNIGRIL_EXIT_CODE={test_omnigril}')

    # 3. Judge
    is_fail2pass = full_passed and test_failed

    if is_fail2pass:
        reason = 'Pass: full state PASS, base state FAIL — fail2pass confirmed'
    elif not test_failed:
        reason = f'Fail: base state passed (OMNIGRIL_EXIT_CODE={test_omnigril}) — test does not reproduce the bug (pass2pass)'
    else:
        reason = 'Unknown'

    logger.info(f'[VALIDATE] Result: {reason}')

    # Detect pass2pass: both states exit 0 — tests pass regardless of fix.
    pass2pass_detected = full_passed and not test_failed
    if pass2pass_detected:
        logger.info('[VALIDATE] pass2pass detected: both states pass — test_patch does not expose the bug')

    return {
        'is_fail2pass': is_fail2pass,
        'pass2pass_detected': pass2pass_detected,
        'full_result': full_result,
        'test_result': test_result,
        'reason': reason,
    }


# ============== Explore Repo Tool ==============


class ExploreRepoInput(BaseModel):
    testbed_dir: str = Field(default='/testbed', description='Path to the testbed directory (default: /testbed)')


async def _explore_repo_legacy(params: ExploreRepoInput, context: ToolContext) -> ToolResponse:
    """Collect structured repository information in a single call."""
    testbed = params.testbed_dir

    explore_script = (
        r"""python3 - << 'PYEOF'
import os, re, json

testbed = '"""
        + testbed
        + r"""'

result = {}

def read_file(path, limit=0):
    try:
        c = open(path).read()
        if limit and len(c) > limit:
            return c[:limit], len(c)
        return c, len(c)
    except:
        return None, 0

# 1. Top-level structure
try:
    items = os.listdir(testbed)
    dirs = [d for d in items if os.path.isdir(os.path.join(testbed, d)) and not d.startswith('.')]
    files = [f for f in items if os.path.isfile(os.path.join(testbed, f))]
    result['top_dirs'] = sorted(dirs)
    result['top_files'] = sorted(files)
except:
    result['top_dirs'] = []
    result['top_files'] = []

# 2. Version signals
signals = []

# Go
c, _ = read_file(f'{testbed}/go.mod')
if c:
    m = re.search(r'^go (\d+\.\d+)', c, re.M)
    if m: signals.append(f'go.mod: go {m.group(1)}')

# Rust
for tf in ['rust-toolchain.toml', 'rust-toolchain']:
    c, _ = read_file(f'{testbed}/{tf}')
    if c:
        m = re.search(r'channel\s*=\s*["\']([^"\']+)["\']', c)
        ver = m.group(1) if m else c.strip()[:20]
        signals.append(f'{tf}: {ver}')
        break

# Node
for nf in ['.nvmrc', '.node-version']:
    c, _ = read_file(f'{testbed}/{nf}')
    if c: signals.append(f'{nf}: {c.strip()}')

# Ruby
for rf in ['.ruby-version', '.tool-versions']:
    c, _ = read_file(f'{testbed}/{rf}')
    if c: signals.append(f'{rf}: {c.strip()}')

# Python — .python-version, setup.py, setup.cfg, pyproject.toml, tox.ini
c, _ = read_file(f'{testbed}/.python-version')
if c: signals.append(f'.python-version: {c.strip()}')

c, _ = read_file(f'{testbed}/setup.py')
if c:
    m = re.search(r'python_requires\s*=\s*["\']([^"\']+)["\']', c)
    if m: signals.append(f'setup.py python_requires: {m.group(1)}')

c, _ = read_file(f'{testbed}/setup.cfg')
if c:
    m = re.search(r'python_requires\s*=\s*(.+)', c)
    if m: signals.append(f'setup.cfg python_requires: {m.group(1).strip()}')

c, _ = read_file(f'{testbed}/pyproject.toml')
if c:
    m = re.search(r'requires-python\s*=\s*["\']([^"\']+)["\']', c)
    if m: signals.append(f'pyproject.toml requires-python: {m.group(1)}')
    # Also detect build backend
    m2 = re.search(r'build-backend\s*=\s*["\']([^"\']+)["\']', c)
    if m2: signals.append(f'pyproject.toml build-backend: {m2.group(1)}')

c, _ = read_file(f'{testbed}/tox.ini')
if c:
    m = re.search(r'basepython\s*=\s*(.+)', c)
    if m: signals.append(f'tox.ini basepython: {m.group(1).strip()}')
    # Detect envlist for test matrix
    m2 = re.search(r'envlist\s*=\s*(.+)', c)
    if m2: signals.append(f'tox.ini envlist: {m2.group(1).strip()[:80]}')

# PHP — composer.json require.php, .php-version
c, _ = read_file(f'{testbed}/.php-version')
if c: signals.append(f'.php-version: {c.strip()}')

c, _ = read_file(f'{testbed}/composer.json')
if c:
    try:
        cj = json.loads(c)
        php_req = cj.get('require', {}).get('php', '')
        if php_req: signals.append(f'composer.json require.php: {php_req}')
        # Detect test framework
        req_dev = cj.get('require-dev', {})
        for pkg in ['phpunit/phpunit', 'behat/behat', 'phpspec/phpspec', 'pestphp/pest']:
            if pkg in req_dev:
                signals.append(f'composer.json test-framework: {pkg} {req_dev[pkg]}')
    except: pass

# Java - pom.xml
c, _ = read_file(f'{testbed}/pom.xml')
if c:
    m = re.search(r'<java\.version>(.*?)</java\.version>', c)
    if m: signals.append(f'pom.xml java.version: {m.group(1)}')
    m2 = re.search(r'<source>(\d+)</source>', c)
    if m2: signals.append(f'pom.xml compiler source: {m2.group(1)}')
    m3 = re.search(r'<maven\.compiler\.release>(\d+)</maven\.compiler\.release>', c)
    if m3: signals.append(f'pom.xml maven.compiler.release: {m3.group(1)}')
    m4 = re.search(r'<maven\.compiler\.target>(\d+)</maven\.compiler\.target>', c)
    if m4 and not m2: signals.append(f'pom.xml maven.compiler.target: {m4.group(1)}')

# Java - build.gradle / build.gradle.kts
c, _ = read_file(f'{testbed}/build.gradle')
c2, _ = read_file(f'{testbed}/build.gradle.kts')
gc = c or c2
if gc:
    m = re.search(r"sourceCompatibility\s*=\s*['\"]?(\S+?)['\"]?\s*\n", gc)
    if m: signals.append(f'build.gradle sourceCompatibility: {m.group(1)}')
    if not any('build.gradle' in s for s in signals):
        m = re.search(r'jvmToolchain\s*\((\d+)\)', gc)
        if m: signals.append(f'build.gradle.kts jvmToolchain: {m.group(1)}')
        else:
            m = re.search(r'JavaVersion\.VERSION_(\d+)', gc)
            if m: signals.append(f'build.gradle.kts JavaVersion: {m.group(1)}')

# Java - gradle.properties
c, _ = read_file(f'{testbed}/gradle.properties')
if c and not any('build.gradle' in s or 'pom.xml' in s for s in signals):
    m = re.search(r'(?:sourceCompatibility|javaVersion|java\.version)\s*=\s*([\d.]+)', c, re.M)
    if m: signals.append(f'gradle.properties javaVersion: {m.group(1)}')

# C# - global.json (root first, then 1 level deep)
_global_json_found = False
for _gj_dir in [testbed] + [os.path.join(testbed, d) for d in sorted(os.listdir(testbed))
                              if os.path.isdir(os.path.join(testbed, d)) and not d.startswith('.')][:8]:
    c, _ = read_file(f'{_gj_dir}/global.json')
    if c:
        try:
            gj = json.loads(c)
            ver = gj.get('sdk', {}).get('version', '')
            if ver:
                rel = os.path.relpath(_gj_dir, testbed)
                label = 'global.json sdk' if rel == '.' else f'global.json sdk ({rel})'
                signals.append(f'{label}: {ver}')
                _global_json_found = True
                break
        except: pass

# C# - .csproj / .fsproj TargetFramework detection (fallback when no global.json)
if not _global_json_found:
    _csproj_tf = []
    for _root, _dirs, _files in os.walk(testbed):
        _dirs[:] = [d for d in _dirs if not d.startswith('.') and d not in ('bin','obj','node_modules')]
        for _f in _files:
            if _f.endswith(('.csproj', '.fsproj', '.vbproj')):
                _c, _ = read_file(os.path.join(_root, _f))
                if _c:
                    for _m in re.finditer(r'<TargetFrameworks?\s*>(.*?)</TargetFrameworks?>', _c):
                        for _tf in re.split(r'[;,\s]+', _m.group(1)):
                            _tf = _tf.strip()
                            if _tf and _tf not in _csproj_tf:
                                _csproj_tf.append(_tf)
        if len(_csproj_tf) >= 6:
            break
    if _csproj_tf:
        signals.append(f'.csproj TargetFramework: {", ".join(_csproj_tf[:6])}')

if not _global_json_found and not any('.csproj' in s for s in signals):
    sln = [f for f in os.listdir(testbed) if f.endswith('.sln')]
    if sln: signals.append(f'.sln: {sln[0]}')

# Rust - Cargo.toml features and dev-dependencies
_cargo_content, _ = read_file(f'{testbed}/Cargo.toml')
if _cargo_content:
    _in_features = False
    _features_found = []
    for _line in _cargo_content.splitlines():
        _ls = _line.strip()
        if re.match(r'^\[.*features.*\]', _ls):
            _in_features = True
            continue
        if _in_features:
            if _ls.startswith('['):
                _in_features = False
                continue
            _fm = re.match(r'^(\w[\w-]*)\s*=', _ls)
            if _fm and _fm.group(1) != 'default':
                _features_found.append(_fm.group(1))
    if _features_found:
        signals.append(f'Cargo.toml features: {", ".join(_features_found[:8])}')
    _in_dev = False
    _dev_deps = []
    for _line in _cargo_content.splitlines():
        _ls = _line.strip()
        if re.match(r'^\[dev-dependencies\]', _ls):
            _in_dev = True
            continue
        if _in_dev:
            if _ls.startswith('['):
                _in_dev = False
                continue
            _dm = re.match(r'^([\w-]+)\s*[=]', _ls)
            if _dm:
                _dev_deps.append(_dm.group(1))
    if _dev_deps:
        signals.append(f'Cargo.toml dev-dependencies: {", ".join(_dev_deps[:8])}')

# CI workflows — extract version patterns AND install commands
wf_dir = f'{testbed}/.github/workflows'
ci_install_lines = []
if os.path.isdir(wf_dir):
    for wf in sorted(os.listdir(wf_dir))[:5]:
        wf_content, _ = read_file(f'{wf_dir}/{wf}')
        if wf_content:
            for pattern in [r'python-version["\s:]+["\']?([\d.]+)', r'node-version["\s:]+["\']?([\d.]+)', r'java-version["\s:]+["\']?(\d+)', r'dotnet-version["\s:]+["\']?([\d.]+)', r'php-version["\s:]+["\']?([\d.]+)']:
                m = re.search(pattern, wf_content)
                if m: signals.append(f'.github/{wf}: {m.group(0)}')
            for line in wf_content.splitlines():
                stripped = line.strip()
                if any(kw in stripped for kw in ['apt-get install', 'apt install', 'pip install', 'npm install', 'brew install', 'yum install', 'dnf install', 'composer install', 'bundle install']):
                    ci_install_lines.append(f'  [{wf}] {stripped}')

result['version_signals'] = signals
result['ci_install_commands'] = ci_install_lines[:20]

# 3. Package manager files present
pkg_checks = [
    'requirements.txt','setup.py','pyproject.toml','setup.cfg','Pipfile','tox.ini',
    'package.json','yarn.lock','package-lock.json','pnpm-lock.yaml',
    'go.mod','go.sum',
    'Cargo.toml','Cargo.lock',
    'pom.xml','build.gradle','build.gradle.kts','gradlew','mvnw','settings.gradle','settings.gradle.kts',
    'Gemfile','Gemfile.lock',
    'composer.json','composer.lock',
    'CMakeLists.txt','configure.ac','configure','meson.build','Makefile','GNUmakefile','SConstruct',
    'global.json',
    'Dockerfile','docker-compose.yml','docker-compose.yaml',
]
result['package_files'] = [f for f in pkg_checks if os.path.exists(f'{testbed}/{f}')]

# 4. Key file paths + truncated previews (show path & size, truncate content for long files)
PREVIEW_LIMIT = 1500
key_files = [
    'requirements.txt', 'setup.py', 'setup.cfg', 'pyproject.toml',
    'go.mod', 'Cargo.toml', 'package.json',
    'pom.xml', 'build.gradle', 'build.gradle.kts',
    'Gemfile', 'composer.json',
    'CMakeLists.txt', 'Makefile', 'meson.build', 'configure.ac',
]
file_previews = {}
for f in key_files:
    fp = f'{testbed}/{f}'
    if os.path.exists(fp):
        c, sz = read_file(fp, limit=PREVIEW_LIMIT)
        if c is not None:
            truncated = sz > PREVIEW_LIMIT
            file_previews[f] = {'size': sz, 'truncated': truncated, 'preview': c}
result['file_previews'] = file_previews

# 5. C/C++ build system deep scan
pkg_files = result['package_files']
is_c_cpp = any(f in pkg_files for f in ['CMakeLists.txt', 'Makefile', 'GNUmakefile', 'configure.ac', 'meson.build', 'SConstruct'])

if is_c_cpp:
    build_info = {}

    # Detect primary build system
    build_systems = []
    if 'CMakeLists.txt' in pkg_files: build_systems.append('cmake')
    if any(f in pkg_files for f in ['Makefile', 'GNUmakefile']): build_systems.append('make')
    if 'configure.ac' in pkg_files: build_systems.append('autotools')
    if 'meson.build' in pkg_files: build_systems.append('meson')
    if 'SConstruct' in pkg_files: build_systems.append('scons')
    build_info['build_systems'] = build_systems

    # Scan CMakeLists.txt in root AND one level deep for test targets
    cmake_test_targets = []
    cmake_files_scanned = []
    cmake_scan_dirs = [testbed]
    try:
        cmake_scan_dirs += [os.path.join(testbed, d) for d in sorted(os.listdir(testbed))
                            if os.path.isdir(os.path.join(testbed, d)) and not d.startswith('.')][:15]
    except: pass
    # Also check test/ and tests/ subdirs
    for tdir in ['test', 'tests', 'testing', 'src/test', 'src/tests']:
        td = os.path.join(testbed, tdir)
        if os.path.isdir(td) and td not in cmake_scan_dirs:
            cmake_scan_dirs.append(td)

    for scan_dir in cmake_scan_dirs:
        cm_path = os.path.join(scan_dir, 'CMakeLists.txt')
        cm, _ = read_file(cm_path)
        if cm:
            rel = os.path.relpath(cm_path, testbed)
            cmake_files_scanned.append(rel)
            for mt in re.finditer(r'add_test\s*\(\s*(?:NAME\s+)?(\S+)', cm):
                t = mt.group(1)
                if t not in cmake_test_targets: cmake_test_targets.append(t)
            for mt in re.finditer(r'add_executable\s*\(\s*(\w*[Tt]est\w*)', cm):
                t = mt.group(1)
                if t not in cmake_test_targets: cmake_test_targets.append(t)
            # gtest_discover_tests / gtest_add_tests
            for mt in re.finditer(r'gtest_(?:discover|add)_tests\s*\(\s*(\w+)', cm):
                t = mt.group(1)
                if t not in cmake_test_targets: cmake_test_targets.append(t)
    build_info['cmake_test_targets'] = cmake_test_targets[:20]
    build_info['cmake_files_scanned'] = cmake_files_scanned[:20]

    # Makefile targets (for non-cmake projects)
    if 'make' in build_systems:
        make_targets = []
        mk, _ = read_file(f'{testbed}/Makefile', limit=8000)
        if mk:
            for mt in re.finditer(r'^([\w._-]*[Tt]est[\w._-]*)\s*:', mk, re.M):
                t = mt.group(1)
                if t not in make_targets: make_targets.append(t)
            # Also look for check target (common in autotools)
            for mt in re.finditer(r'^(check|check-am|check-TESTS)\s*:', mk, re.M):
                t = mt.group(1)
                if t not in make_targets: make_targets.append(t)
        build_info['make_test_targets'] = make_targets[:15]

    # Meson test targets
    if 'meson' in build_systems:
        mc, _ = read_file(f'{testbed}/meson.build', limit=8000)
        meson_tests = []
        if mc:
            for mt in re.finditer(r"test\s*\(\s*'([^']+)'", mc):
                meson_tests.append(mt.group(1))
            for mt in re.finditer(r'executable\s*\(\s*["\'](\w*[Tt]est\w*)["\']', mc):
                t = mt.group(1)
                if t not in meson_tests: meson_tests.append(t)
        build_info['meson_test_targets'] = meson_tests[:15]

    # Detect test frameworks from source
    test_frameworks = []
    # Check common include patterns in first few .c/.cpp/.h files
    test_fw_patterns = {
        'gtest': [r'#include\s*[<"]gtest/', r'TEST\s*\(', r'TEST_F\s*\('],
        'catch2': [r'#include\s*[<"]catch', r'TEST_CASE\s*\('],
        'doctest': [r'#include\s*[<"]doctest', r'DOCTEST_'],
        'cunit': [r'#include\s*[<"]CUnit/', r'CU_add_test'],
        'cmocka': [r'#include\s*[<"]cmocka', r'cmocka_unit_test'],
        'check': [r'#include\s*[<"]check\.h', r'ck_assert'],
        'unity': [r'#include\s*[<"]unity\.h', r'TEST_ASSERT'],
        'boost.test': [r'#include\s*[<"]boost/test/', r'BOOST_AUTO_TEST'],
        'criterion': [r'#include\s*[<"]criterion/', r'Test\s*\('],
    }
    # Quick scan: check CMakeLists for framework hints first
    all_cmake = ' '.join(file_previews.get('CMakeLists.txt', {}).get('preview', '').lower()
                         for _ in [1])
    for fw, _ in test_fw_patterns.items():
        if fw.lower().replace('.', '') in all_cmake.replace('.', ''):
            test_frameworks.append(fw)
    # If no framework from cmake, scan a few test source files
    if not test_frameworks:
        for tdir in ['test', 'tests', 'testing', 'src/test', 'src/tests', '.']:
            td = os.path.join(testbed, tdir)
            if not os.path.isdir(td): continue
            try:
                for fn in sorted(os.listdir(td))[:10]:
                    if fn.endswith(('.c', '.cpp', '.cc', '.cxx', '.h', '.hpp')):
                        tc, _ = read_file(os.path.join(td, fn), limit=2000)
                        if tc:
                            for fw, pats in test_fw_patterns.items():
                                if fw not in test_frameworks:
                                    for p in pats:
                                        if re.search(p, tc):
                                            test_frameworks.append(fw)
                                            break
            except: pass
            if test_frameworks: break
    build_info['test_frameworks'] = test_frameworks

    # Extra build config files
    extra_build_files = []
    for ebf in ['CMakePresets.json', 'conanfile.txt', 'conanfile.py', 'vcpkg.json',
                '.cmake', 'CTestTestfile.cmake', 'toolchain.cmake']:
        if os.path.exists(f'{testbed}/{ebf}'):
            extra_build_files.append(ebf)
    build_info['extra_build_files'] = extra_build_files

    # CI scripts and Dockerfiles (important for C/C++ — authoritative dep list)
    setup_scripts = []
    for pattern_dir in ['scripts', 'ci', '.ci', 'ci/docker', '.github/scripts', 'build', 'tools']:
        sd = os.path.join(testbed, pattern_dir)
        if os.path.isdir(sd):
            try:
                for fn in sorted(os.listdir(sd))[:10]:
                    fp = os.path.join(sd, fn)
                    if os.path.isfile(fp):
                        rel = os.path.relpath(fp, testbed)
                        if fn.endswith(('.sh', '.bash')) or fn.startswith(('setup', 'install', 'build', 'deps', 'bootstrap')):
                            setup_scripts.append(rel)
                        elif fn in ('Dockerfile', 'Dockerfile.dev', 'Dockerfile.build', 'Dockerfile.ci'):
                            setup_scripts.append(rel)
            except: pass
    # Root-level CI-relevant files
    for rf in ['Dockerfile', 'Dockerfile.dev', '.circleci/config.yml']:
        if os.path.exists(os.path.join(testbed, rf)):
            setup_scripts.append(rf)
    build_info['setup_scripts'] = setup_scripts[:20]

    # CMake flags from CI
    cmake_ci_flags = []
    if os.path.isdir(wf_dir):
        for wf in sorted(os.listdir(wf_dir))[:5]:
            wf_content, _ = read_file(f'{wf_dir}/{wf}')
            if wf_content:
                for line in wf_content.splitlines():
                    stripped = line.strip()
                    if 'cmake' in stripped.lower() and ('-D' in stripped or '-B' in stripped or '-G' in stripped):
                        cmake_ci_flags.append(f'  [{wf}] {stripped}')
    build_info['cmake_ci_flags'] = cmake_ci_flags[:10]

    result['build_info'] = build_info
else:
    result['build_info'] = {}

# 6. Scan for test directories (useful for all languages)
test_dirs = []
for td_name in ['test', 'tests', 'testing', 'spec', 'specs', 'test_suite',
                 'src/test', 'src/tests', 'src/testing', '__tests__']:
    td = os.path.join(testbed, td_name)
    if os.path.isdir(td):
        try:
            entries = sorted(os.listdir(td))[:15]
            test_dirs.append({'path': td_name, 'entries': entries})
        except:
            test_dirs.append({'path': td_name, 'entries': []})
result['test_dirs'] = test_dirs

print(json.dumps(result))
PYEOF
"""
    )

    ctx = require_context()
    run_ctx = await ctx.get_runtime('main')
    out = await run_ctx.run_command(explore_script, timeout=30)
    raw = (out.stdout or '') + (out.stderr or '')

    # Try to parse JSON output
    try:
        json_start = raw.find('{')
        json_end = raw.rfind('}') + 1
        data = json.loads(raw[json_start:json_end]) if json_start >= 0 else {}
    except Exception:
        return ToolResponse(output=_make_output(f'[explore_repo] Raw output:\n{raw}'))

    # Format structured report
    lines = [
        '## Repository Exploration',
        '',
        f'**Top-level directories**: {", ".join(data.get("top_dirs", []))}',
        f'**Top-level files**: {", ".join(data.get("top_files", []))}',
        '',
    ]

    # Test directories
    test_dirs = data.get('test_dirs', [])
    if test_dirs:
        lines += ['### Test Directories']
        for td in test_dirs:
            entries_str = ', '.join(td['entries'][:10])
            if len(td['entries']) > 10:
                entries_str += f', ... ({len(td["entries"])} total)'
            lines.append(f'- `{td["path"]}/`: {entries_str}')
        lines.append('')

    # Version signals
    lines.append('### Version Signals')
    for s in data.get('version_signals', []):
        lines.append(f'- {s}')
    if not data.get('version_signals'):
        lines.append('- (none detected)')

    # CI install commands
    ci_installs = data.get('ci_install_commands', [])
    if ci_installs:
        lines += ['', '### CI Install Commands (from .github/workflows)']
        for line in ci_installs:
            lines.append(line)

    # Package manager files
    lines += [
        '',
        '### Package Manager / Build Files Present',
        '  ' + ', '.join(data.get('package_files', ['(none)'])),
    ]

    # Key file previews — path + size + truncated content
    file_previews = data.get('file_previews', {})
    if file_previews:
        lines += ['', '### Key File Previews']
        lines.append('*(truncated — use `bash` to read full content if needed)*')
        for fname, info in file_previews.items():
            sz = info['size']
            trunc = info['truncated']
            size_str = f'{sz:,} bytes' if sz >= 1000 else f'{sz} bytes'
            trunc_str = ' (TRUNCATED)' if trunc else ''
            lines += [f'**{fname}** ({size_str}{trunc_str}):', '```', info['preview'], '```', '']

    # C/C++ build system details
    build_info = data.get('build_info', {})
    if build_info:
        lines += ['### C/C++ Build System Details']
        bs = build_info.get('build_systems', [])
        if bs:
            lines.append(f'**Build systems**: {", ".join(bs)}')

        tf = build_info.get('test_frameworks', [])
        if tf:
            lines.append(f'**Test frameworks**: {", ".join(tf)}')

        cmake_targets = build_info.get('cmake_test_targets', [])
        if cmake_targets:
            lines.append(f'**CMake test targets**: `{"`, `".join(cmake_targets)}`')
            scanned = build_info.get('cmake_files_scanned', [])
            if scanned:
                lines.append(f'  (scanned: {", ".join(scanned)})')

        make_targets = build_info.get('make_test_targets', [])
        if make_targets:
            lines.append(f'**Makefile test targets**: `{"`, `".join(make_targets)}`')

        meson_tests = build_info.get('meson_test_targets', [])
        if meson_tests:
            lines.append(f'**Meson test targets**: `{"`, `".join(meson_tests)}`')

        extra = build_info.get('extra_build_files', [])
        if extra:
            lines.append(f'**Extra build config**: {", ".join(extra)}')
            if 'conanfile.txt' in extra or 'conanfile.py' in extra:
                lines.append('  → Conan detected — run `conan install . --build=missing` before cmake')
            if 'vcpkg.json' in extra:
                lines.append('  → vcpkg detected — run `vcpkg install` or use `-DCMAKE_TOOLCHAIN_FILE=...`')

        scripts = build_info.get('setup_scripts', [])
        if scripts:
            lines.append('**Setup/CI scripts** (read these for deps & build flags):')
            for s in scripts:
                lines.append(f'  - `{s}`')

        cmake_ci = build_info.get('cmake_ci_flags', [])
        if cmake_ci:
            lines.append('**CMake flags from CI**:')
            for fl in cmake_ci:
                lines.append(fl)

        lines.append('')

    # Recommended image
    lines += ['### Recommended Action']
    signals_str = ' '.join(data.get('version_signals', []))
    pkg_files = data.get('package_files', [])
    has_recommendation = False

    # Go
    if 'go.mod: go' in signals_str:
        m = re.search(r'go\.mod: go (\d+\.\d+)', signals_str)
        if m:
            ver = m.group(1)
            lines.append(f'→ **switch_runtime_image** to `golang-local:{ver}` (detected from go.mod)')
            has_recommendation = True

    # Rust
    if 'rust-toolchain' in signals_str:
        m = re.search(r'rust-toolchain.*?: ([\d.]+)', signals_str)
        if m:
            ver = m.group(1)[:4]
            lines.append(f'→ **switch_runtime_image** to `rust-git:{ver}-slim` (detected from rust-toolchain)')
            has_recommendation = True
        elif 'nightly' in signals_str:
            lines.append('→ **switch_runtime_image** to `rust-git:latest` (nightly toolchain)')
            has_recommendation = True
    if 'Cargo.toml features:' in signals_str:
        feat_m = re.search(r'Cargo\.toml features: ([^\n]+)', signals_str)
        if feat_m:
            lines.append(f'→ Rust features: `{feat_m.group(1).strip()}` — use `cargo test --features <name>` if needed')

    # C#
    if 'global.json sdk' in signals_str:
        m = re.search(r'global\.json sdk[^:]*: ([\d.]+)', signals_str)
        if m:
            ver = m.group(1)[:3]
            lines.append(f'→ **switch_runtime_image** to `dotnet-git:{ver}` (detected from global.json)')
            has_recommendation = True
    elif '.csproj TargetFramework' in signals_str:
        tf_m = re.search(r'\.csproj TargetFramework: ([\w.,; ]+)', signals_str)
        if tf_m:
            tf_str = tf_m.group(1)
            vers = []
            for tfv in re.findall(r'net(\d+)\.(\d+)', tf_str):
                major = int(tfv[0])
                if major >= 5:
                    vers.append(major)
            if vers:
                dv = max(vers)
                lines.append(f'→ **switch_runtime_image** to `dotnet-git:{dv}.0` (from .csproj: {tf_str.strip()})')
                has_recommendation = True
            else:
                lines.append(
                    f'→ C# project (.csproj: {tf_str.strip()}) — determine dotnet version, then **switch_runtime_image** to `dotnet-git:X.0`'
                )
                has_recommendation = True
    elif '.sln' in signals_str:
        lines.append(
            '→ C# project (.sln) — check .csproj for TargetFramework, then **switch_runtime_image** to `dotnet-git:X.0`'
        )
        has_recommendation = True

    # Java
    _java_ver_signal = any(
        k in signals_str
        for k in [
            'build.gradle sourceCompatibility',
            'build.gradle.kts jvmToolchain',
            'build.gradle.kts JavaVersion',
            'gradle.properties javaVersion',
            'pom.xml java.version',
            'pom.xml compiler source',
            'pom.xml maven.compiler.release',
            'pom.xml maven.compiler.target',
            'java-version',
        ]
    )
    if _java_ver_signal:
        mj = re.search(
            r'(?:jvmToolchain|JavaVersion\.VERSION_|javaVersion|java\.version|sourceCompatibility|compiler (?:source|release|target)|java-version[^=\d]*)[\s=:"\']*(\d+)',
            signals_str,
        )
        if mj:
            jv = int(mj.group(1))
            jdk = '8' if jv <= 8 else ('11' if jv <= 11 else ('17' if jv <= 17 else '21'))
            lines.append(
                f'→ **switch_runtime_image** to `java-git:{jdk}` (Java {jv} detected — JDK + Maven pre-installed)'
            )
            has_recommendation = True
    elif any(f in pkg_files for f in ['pom.xml', 'build.gradle', 'build.gradle.kts', 'gradlew']):
        lines.append(
            '→ Java project detected but no version signal — **switch_runtime_image** to `java-git:11` (safe default)'
        )
        has_recommendation = True

    # Node
    if '.nvmrc' in signals_str or '.node-version' in signals_str:
        m = re.search(r'(?:\.nvmrc|\.node-version): v?(\d+)', signals_str)
        if m:
            ver = m.group(1)
            lines.append(
                f'→ **switch_runtime_image** to `node-git:{ver}-bullseye` (detected from .nvmrc/.node-version)'
            )
            has_recommendation = True
    elif 'package.json' in pkg_files and not has_recommendation:
        # Node project without explicit version
        # Check CI for version
        ci_node = re.search(r'node-version["\s:]+["\']?(\d+)', signals_str)
        if ci_node:
            lines.append(f'→ **switch_runtime_image** to `node-git:{ci_node.group(1)}-bullseye` (from CI workflow)')
            has_recommendation = True

    # Ruby
    if '.ruby-version' in signals_str:
        m = re.search(r'\.ruby-version: (\d+\.\d+)', signals_str)
        if m:
            ver = m.group(1)
            lines.append(f'→ **switch_runtime_image** to `ruby-git:{ver}-slim` (detected from .ruby-version)')
            has_recommendation = True

    # PHP
    php_ver = None
    if '.php-version' in signals_str:
        m = re.search(r'\.php-version: (\d+\.\d+)', signals_str)
        if m:
            php_ver = m.group(1)
    if not php_ver and 'composer.json require.php' in signals_str:
        # Parse version constraint like ">=7.4", "^8.1", "~8.0"
        m = re.search(r'composer\.json require\.php: [^\d]*(\d+\.\d+)', signals_str)
        if m:
            php_ver = m.group(1)
    if not php_ver:
        ci_php = re.search(r'php-version["\s:]+["\']?(\d+\.\d+)', signals_str)
        if ci_php:
            php_ver = ci_php.group(1)
    if php_ver:
        lines.append(f'→ **switch_runtime_image** to `php-git:{php_ver}-cli` (PHP {php_ver} detected)')
        has_recommendation = True
    elif 'composer.json' in pkg_files and not has_recommendation:
        lines.append('→ PHP project detected — **switch_runtime_image** to `php-git:8.1-cli` (safe default)')
        has_recommendation = True

    # C/C++ — build guidance (no image switch needed, ubuntu-git is default)
    if build_info and build_info.get('build_systems'):
        bs = build_info['build_systems']
        cmake_targets = build_info.get('cmake_test_targets', [])
        make_targets = build_info.get('make_test_targets', [])
        scripts = build_info.get('setup_scripts', [])

        if scripts:
            lines.append(
                f'→ C/C++ project — **read setup scripts first**: `{scripts[0]}` for exact deps and build flags'
            )
        if cmake_targets:
            lines.append(
                f'→ CMake test targets found: build with `cmake -B /tmp/project_build /testbed && cmake --build /tmp/project_build --target {cmake_targets[0]} -j4`'
            )
            lines.append('  **Use `-j4`, NOT `-j$(nproc)`** (nproc returns host CPU count → OOM in container)')
        elif make_targets:
            lines.append(f'→ Makefile test targets found: `make -j4 {make_targets[0]}`')
        elif 'cmake' in bs:
            lines.append(
                '→ CMake project — no test targets found in root CMakeLists.txt; check subdirectories or CI scripts for test targets'
            )
        if not has_recommendation:
            has_recommendation = True

    if not has_recommendation:
        lines.append('→ Current image appears suitable (no version file detected)')

    return ToolResponse(output=_make_output('\n'.join(lines)))
