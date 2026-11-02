"""Builder turn-0 auto-setup for code_mt_agent.

Deterministic env prep, run by the AGENT on its first turn (see
nexus/extensions/agents/code_mt/setup.py::run_turn0_setup):
1. Explore repo (version signals, package files, test dirs, build systems, CI hints)
2. Determine and switch runtime image if needed
3. Probe patches (golden_patch + test_patch apply cleanly?)
4. Pre-install (language-specific dependency install)
5. Write patch files to /tmp for agent reference

Returns inject_vars dict (merged into ctx.task_config) whose keys the agent's
prompt template (instruction.md) references — chiefly `auto_setup_results`.
Prompt/system-prompt/mt_feedback templates are NOT loaded here: the agent owns
its prompt (build_initial_messages) and the task owns the MT feedback prompt.
"""

from __future__ import annotations

import base64
import logging
import re
import time

from typing import Any

from nexus.extensions.projects.code_mt.agent.images import image_alias, resolve_mirror_image, validate_image


logger = logging.getLogger(__name__)


class TestbedInitError(Exception):
    """Raised when /testbed could not be (re)populated after an image switch.

    This is a FAIL-FAST signal: a builder task whose repo checkout failed must
    NOT run the agent against an empty /testbed and reach grading (that silently
    produces empty-patch / reward-hacking rollouts). It is deliberately a
    dedicated type so the switch/turn-0 wrappers can re-raise ONLY this error
    while still swallowing non-fatal setup hiccups (explore/probe/pre-install).
    Propagates through run_turn0_setup -> agent loop -> UNHANDLED_ERROR.
    """


async def reinit_testbed_via_setup_commands(runtime: Any, task_config: dict, log_tag: str = 'REINIT_TESTBED') -> None:
    """Re-run the dataset's setup_commands on `runtime` to rebuild /testbed.

    A runtime image switch goes through ctx.replace_runtime, which RELEASES the
    old container and ACQUIRES a brand-new one from the base image. On 0720
    harbor builder tasks the repo is NOT baked into any image and there is no
    /testbed bind-mount across containers — /testbed exists solely because the
    task's [other].setup_commands (git clone + reset --hard <base_commit>) ran
    in the ORIGINAL container. So after a switch the new container's /testbed is
    empty until these commands are re-run.

    This is the single source of truth for that repopulation, shared by:
      * the turn-0 auto-setup hook (_check_and_switch_image), and
      * the agent-invoked switch_runtime_image tool (tools.py).
    Keeping them on one helper prevents the two paths from drifting (the turn-0
    path historically lacked this step, which emptied /testbed ~97-99% of the
    time whenever an image switch fired).

    FAIL-FAST: a setup command that returns rc!=0 or times out raises
    TestbedInitError. Silently swallowing (the old behavior) let a failed repo
    checkout leave /testbed empty, and the agent then ran to grading against an
    empty tree -> empty-patch / CODE_REWARD_HACKING rollouts (~72% of switched
    builder tasks). The dataset's setup_commands are idempotent shallow fetches
    (git init + fetch --depth=1 <sha>), safe to re-run on a fresh container.
    Timeout matches the init path: task_config.setup_timeout or 900s.
    """
    setup_commands = task_config.get('setup_commands', []) or []
    setup_timeout = float(task_config.get('setup_timeout') or 900.0)
    for cmd in setup_commands:
        try:
            formatted = cmd.format(**task_config)
        except (KeyError, IndexError):
            formatted = cmd
        short = formatted if len(formatted) <= 120 else formatted[:117] + '...'
        try:
            result = await runtime.run_command(command=formatted, timeout=setup_timeout)
        except Exception as e:
            # run_command raises on timeout (or transport failure). This is a
            # hard environment-build failure -> fail fast.
            raise TestbedInitError(f'[{log_tag}] setup cmd raised (timeout={setup_timeout}s): {short} ({e})') from e
        if getattr(result, 'return_code', 0) != 0:
            stderr = (getattr(result, 'stderr', '') or '').strip()
            raise TestbedInitError(f'[{log_tag}] setup cmd rc={result.return_code}: {short} :: {stderr[:400]}')
        logger.debug('[%s] setup cmd ok: %s', log_tag, short)


async def builder_pre_agent(
    ctx: Any,
    runtime: Any,
    agent_input: Any = None,
) -> dict[str, Any]:
    """Turn-0 setup for builder tasks.

    Performs deterministic env prep and returns inject_vars for the prompt
    template. Reads every input from ctx.task_config (the task flattens dataset
    [extra] into it at entry), so it needs nothing from agent_input.
    """
    task_config = ctx.task_config
    language = task_config.get('programming_language', task_config.get('language', '')).lower()
    instance_id = task_config.get('instance_id', '?')

    t0 = time.monotonic()
    logger.debug('=' * 80)
    logger.debug('[BUILDER_PRE_AGENT] Starting for %s (%s)', instance_id, language)
    logger.debug('=' * 80)

    results: list[str] = []
    warnings: list[str] = []

    # Step 0: Write patch files to /tmp for agent reference
    await _write_patch_files(runtime, task_config)

    # Step 1: Explore repo (aligned with swe_builder's explore_repo)
    logger.debug('[BUILDER_PRE_AGENT] Step 1: Exploring repository...')
    explore_data = await _explore_repo(runtime)
    if explore_data:
        results.append(_format_explore_results(explore_data, language))

    # Step 2: Determine and switch image
    logger.debug('[BUILDER_PRE_AGENT] Step 2: Checking runtime image...')
    switch_result = await _check_and_switch_image(ctx, runtime, task_config, language, explore_data)
    if switch_result['switched']:
        runtime = switch_result['runtime']
        results.append(f'**Image switched** to `{switch_result["image"]}`')
    elif switch_result.get('warning'):
        warnings.append(switch_result['warning'])

    # Step 3: Probe patches
    logger.debug('[BUILDER_PRE_AGENT] Step 3: Probing patches...')
    probe_result = await _probe_patches(runtime, task_config)
    if probe_result:
        results.append(_format_probe_results(probe_result))

    # Step 4: Pre-install
    logger.debug('[BUILDER_PRE_AGENT] Step 4: Running pre-install...')
    install_result = await _pre_install(runtime, language, explore_data)
    if install_result:
        results.append(_format_install_results(install_result))

    duration = round(time.monotonic() - t0, 1)
    logger.debug('[BUILDER_PRE_AGENT] Done in %.1fs', duration)

    auto_setup_results = _build_auto_setup_markdown(results, warnings, language, task_config, duration, explore_data)

    # inject_vars merged into ctx.task_config for the agent's turn-0 prompt render.
    return {
        'auto_setup_results': auto_setup_results,
        'base_image': image_alias(task_config.get('base_image', 'unknown')),
        'test_patch_display': _patch_display(task_config.get('test_patch', '')),
        'golden_patch_display': _patch_display(task_config.get('golden_patch', '')),
    }


def _patch_display(patch: str, limit: int = 4000) -> str:
    if not patch:
        return '(empty)'
    if len(patch) <= limit:
        return patch
    return patch[:limit] + '\n... [truncated, see /tmp/*.patch in runtime] ...'


# === Step 0: Write Patch Files ===


async def _write_patch_files(runtime: Any, task_config: dict) -> None:
    """Write golden_patch and test_patch to /tmp for agent reference during setup.

    Swallows runtime errors: a transient sandbox hiccup here must not prevent
    the rest of setup from running.
    """
    golden = task_config.get('golden_patch', '')
    test = task_config.get('test_patch', '')
    # Route the patch bytes through stdin (carried in the request body) instead of
    # embedding them in the `bash -c` argv: a large patch overflows execve ARG_MAX
    # ("Argument list too long"), which silently drops the patch file. base64 keeps
    # the piped payload newline/quote-safe.
    try:
        if golden.strip():
            await runtime.run_command(
                command='base64 -d > /tmp/golden_patch.patch',
                stdin=base64.b64encode(golden.encode('utf-8')).decode('ascii'),
                timeout=30,
            )
        if test.strip():
            await runtime.run_command(
                command='base64 -d > /tmp/test_patch.patch',
                stdin=base64.b64encode(test.encode('utf-8')).decode('ascii'),
                timeout=30,
            )
    except Exception as e:
        logger.warning('[BUILDER_PRE_AGENT] _write_patch_files failed (ignored): %s', e)


# === Step 1: Explore Repo (aligned with swe_builder) ===

_EXPLORE_SCRIPT = r"""bash << 'BASHEOF'
# Portable pure-bash repo explorer. Emits line-prefixed records that
# _format_explore_results parses:
#   TOPDIR: <name>
#   SIG: <version signal>
#   PKG: <filename>
#   TESTDIR: <relpath>
#   CI: [wf.yml] <install command line>
#
# Constraints: GNU coreutils/sed/grep/awk/find (all builder images debian-based).
# No python3 dependency — slim language images (rust/ruby/dotnet/node/php/go)
# don't have python3.

set +e
T='/testbed'
cd "$T" 2>/dev/null || exit 0

# 1. top-level dirs (skip dot-dirs, up to 20)
for d in "$T"/*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    case "$name" in .*) continue;; esac
    echo "TOPDIR: $name"
done | head -n 20

# helpers
_read_head() {
    # read first N bytes of a file if it exists, else empty
    [ -f "$1" ] || return
    head -c "${2:-4000}" "$1" 2>/dev/null
}

# 2. version signals

pv=$(_read_head "$T/.python-version" 200 | tr -d '\r\n' | awk '{$1=$1};1')
[ -n "$pv" ] && echo "SIG: .python-version: $pv"

nv=$(_read_head "$T/.nvmrc" 200 | tr -d '\r\n' | awk '{$1=$1};1')
[ -n "$nv" ] && echo "SIG: .nvmrc: $nv"

rv=$(_read_head "$T/.ruby-version" 200 | tr -d '\r\n' | awk '{$1=$1};1')
[ -n "$rv" ] && echo "SIG: .ruby-version: $rv"

gv=$(_read_head "$T/.go-version" 200 | tr -d '\r\n' | awk '{$1=$1};1')
[ -n "$gv" ] && echo "SIG: .go-version: $gv"

# go.mod line "go 1.NN"
gm=$(_read_head "$T/go.mod" 500)
if [ -n "$gm" ]; then
    line=$(printf '%s\n' "$gm" | grep -oE '^go[[:space:]]+[0-9]+\.[0-9]+' | head -n1 | awk '{print $2}')
    [ -n "$line" ] && echo "SIG: go.mod: go $line"
fi

# rust-toolchain / rust-toolchain.toml — prefer channel=, else "nightly", else X.Y.Z
for rtf in rust-toolchain rust-toolchain.toml; do
    rt=$(_read_head "$T/$rtf" 500)
    [ -z "$rt" ] && continue
    chan=$(printf '%s\n' "$rt" | grep -oE 'channel[[:space:]]*=[[:space:]]*"?[A-Za-z0-9_.\-]+' | head -n1 | sed -E 's/.*=[[:space:]]*"?([A-Za-z0-9_.\-]+).*/\1/')
    if [ -n "$chan" ]; then
        echo "SIG: $rtf: $chan"
    elif printf '%s' "$rt" | grep -q nightly; then
        echo "SIG: $rtf: nightly"
    else
        ver=$(printf '%s' "$rt" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1)
        [ -n "$ver" ] && echo "SIG: $rtf: $ver"
    fi
done

# global.json .NET SDK
gj=$(_read_head "$T/global.json" 500)
if [ -n "$gj" ]; then
    sdk=$(printf '%s' "$gj" | grep -oE '"version"[[:space:]]*:[[:space:]]*"[0-9.]+' | head -n1 | sed -E 's/.*"([0-9.]+)$/\1/')
    [ -n "$sdk" ] && echo "SIG: global.json sdk: $sdk"
fi

# .csproj TargetFramework (depth<=2, first match)
csproj=$(find "$T" -maxdepth 3 -name '*.csproj' -print 2>/dev/null | head -n1)
if [ -n "$csproj" ]; then
    tf=$(head -c 2000 "$csproj" 2>/dev/null | grep -oE '<TargetFrameworks?>[^<]+</TargetFramework' | head -n1 | sed -E 's|<TargetFrameworks?>([^<]+)</TargetFramework|\1|')
    [ -n "$tf" ] && echo "SIG: .csproj TargetFramework: $tf"
fi

# pyproject.toml python version
pp=$(_read_head "$T/pyproject.toml" 3000)
if [ -n "$pp" ]; then
    ver=$(printf '%s\n' "$pp" | grep -oE '(python_requires|requires-python)[[:space:]]*=[[:space:]]*"?>=?[0-9.]+' | head -n1 | grep -oE '[0-9]+\.[0-9]+')
    [ -n "$ver" ] && echo "SIG: pyproject.toml python: >=$ver"
fi

# setup.py python_requires
sp=$(_read_head "$T/setup.py" 3000)
if [ -n "$sp" ]; then
    ver=$(printf '%s\n' "$sp" | grep -oE 'python_requires[[:space:]]*=[[:space:]]*"?>=?[0-9.]+' | head -n1 | grep -oE '[0-9]+\.[0-9]+')
    [ -n "$ver" ] && echo "SIG: setup.py python_requires: >=$ver"
fi

# Cargo.toml features (up to 8)
ct=$(_read_head "$T/Cargo.toml" 3000)
if [ -n "$ct" ]; then
    feats=$(printf '%s\n' "$ct" | grep -oE '^\[features\.[A-Za-z0-9_-]+\]' | sed -E 's/^\[features\.([A-Za-z0-9_-]+)\]/\1/')
    if [ -z "$feats" ]; then
        # fallback: `foo = [` under [features]
        feats=$(printf '%s\n' "$ct" | awk '
            /^\[features\]/ {in_f=1; next}
            /^\[/ {in_f=0}
            in_f && /^[A-Za-z0-9_-]+[[:space:]]*=[[:space:]]*\[/ {
                sub(/[[:space:]]*=.*/, ""); print
            }')
    fi
    if [ -n "$feats" ]; then
        joined=$(printf '%s\n' "$feats" | head -n 8 | paste -sd ',' - | sed 's/,/, /g')
        echo "SIG: Cargo.toml features: $joined"
    fi
fi

# 3. CI workflows (first 5 .yml/.yaml in .github/workflows)
wf_dir="$T/.github/workflows"
if [ -d "$wf_dir" ]; then
    cnt=0
    ci_cnt=0
    for wf in $(ls "$wf_dir" 2>/dev/null | sort); do
        [ "$cnt" -ge 5 ] && break
        wfp="$wf_dir/$wf"
        [ -f "$wfp" ] || continue
        cnt=$((cnt + 1))
        content=$(cat "$wfp" 2>/dev/null)
        [ -z "$content" ] && continue
        for pat in 'python-version["[:space:]:]+["]?[0-9.]+' \
                   'node-version["[:space:]:]+["]?[0-9.]+' \
                   'java-version["[:space:]:]+["]?[0-9]+' \
                   'dotnet-version["[:space:]:]+["]?[0-9.]+' \
                   'php-version["[:space:]:]+["]?[0-9.]+'; do
            match=$(printf '%s\n' "$content" | grep -oE "$pat" | head -n1)
            [ -n "$match" ] && echo "SIG: .github/$wf: $match"
        done
        # Install-command lines (up to 20 total across all workflows)
        while IFS= read -r line; do
            [ "$ci_cnt" -ge 20 ] && break 2
            trimmed=$(printf '%s' "$line" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')
            case "$trimmed" in
                *"apt-get install"*|*"apt install"*|*"pip install"*|*"npm install"*|\
                *"brew install"*|*"yum install"*|*"dnf install"*|*"composer install"*|\
                *"bundle install"*)
                    echo "CI: [$wf] $trimmed"
                    ci_cnt=$((ci_cnt + 1))
                    ;;
            esac
        done <<< "$content"
    done
fi

# 4. Package manager files
for f in requirements.txt setup.py pyproject.toml setup.cfg Pipfile tox.ini \
         package.json yarn.lock package-lock.json pnpm-lock.yaml \
         Cargo.toml Cargo.lock go.mod go.sum Gemfile Gemfile.lock \
         composer.json composer.lock pom.xml build.gradle build.gradle.kts \
         settings.gradle Makefile CMakeLists.txt meson.build global.json; do
    [ -e "$T/$f" ] && echo "PKG: $f"
done

# 5. Test directories (max depth 3, up to 15)
find "$T" -mindepth 1 -maxdepth 4 -type d 2>/dev/null | while IFS= read -r d; do
    base=$(basename "$d")
    case "$base" in
        test|tests|spec|__tests__|test_utils|test_*|*_test)
            rel=${d#"$T/"}
            echo "TESTDIR: $rel"
            ;;
    esac
done | head -n 15
BASHEOF"""


async def _explore_repo(runtime: Any) -> dict[str, Any]:
    """Run the bash explorer and parse its line-prefixed output.

    Output format (see _EXPLORE_SCRIPT header comment):
      TOPDIR: <name>
      SIG: <version signal>
      PKG: <filename>
      TESTDIR: <relpath>
      CI: [wf.yml] <install command>
    """
    try:
        result = await runtime.run_command(command=_EXPLORE_SCRIPT, timeout=60)
        output = result.stdout or ''
    except Exception as e:
        logger.warning('[BUILDER_PRE_AGENT] explore_repo failed: %s', e)
        return {}

    data: dict[str, list[str]] = {
        'top_dirs': [],
        'version_signals': [],
        'package_files': [],
        'test_dirs': [],
        'ci_install_commands': [],
    }
    for raw_line in output.splitlines():
        line = raw_line.rstrip('\r')
        if line.startswith('TOPDIR: '):
            data['top_dirs'].append(line[len('TOPDIR: ') :])
        elif line.startswith('SIG: '):
            data['version_signals'].append(line[len('SIG: ') :])
        elif line.startswith('PKG: '):
            data['package_files'].append(line[len('PKG: ') :])
        elif line.startswith('TESTDIR: '):
            data['test_dirs'].append(line[len('TESTDIR: ') :])
        elif line.startswith('CI: '):
            data['ci_install_commands'].append(line[len('CI: ') :])
    return data


def _format_explore_results(data: dict[str, Any], language: str) -> str:
    parts = []
    if data.get('package_files'):
        parts.append(f'Package files: {", ".join(data["package_files"][:10])}')
    if data.get('test_dirs'):
        parts.append(f'Test dirs: {", ".join(data["test_dirs"][:8])}')
    if data.get('version_signals'):
        parts.append(f'Version signals: {"; ".join(data["version_signals"][:6])}')
    if data.get('ci_install_commands'):
        parts.append('CI install hints:')
        for cmd in data['ci_install_commands'][:5]:
            parts.append(f'  {cmd}')
    return '\n'.join(parts) if parts else ''


# === Step 2: Image Switch ===

_LANGUAGE_IMAGE_MAP = {
    'python': 'python-git:{version}-slim',
    'javascript': 'node-git:{version}-bullseye',
    'typescript': 'node-git:{version}-bullseye',
    'ruby': 'ruby-git:{version}-slim',
    'go': 'golang-local:{version}',
    'rust': 'rust-git:latest',
    'java': 'java-git:{version}',
    'c#': 'dotnet-git:{version}',
    'csharp': 'dotnet-git:{version}',
    'php': 'php-git:{version}-cli',
}

_DEFAULT_VERSIONS = {
    'python': '3.11',
    'javascript': '18',
    'typescript': '18',
    'ruby': '3.2',
    'go': '1.22',
    'rust': 'latest',
    'java': '17',
    'c#': '8.0',
    'csharp': '8.0',
    'php': '8.2',
}


def _detect_version(language: str, explore_data: dict[str, Any]) -> str:
    """Detect version from explore_repo signals."""
    signals_str = ' '.join(explore_data.get('version_signals', []))

    if language == 'python':
        m = re.search(r'\.python-version:\s*(\d+\.\d+)', signals_str)
        if m:
            return m.group(1)
        m = re.search(r'python.*?>=?\s*(\d+\.\d+)', signals_str, re.IGNORECASE)
        if m:
            return m.group(1)

    elif language in ('javascript', 'typescript'):
        m = re.search(r'\.nvmrc:\s*v?(\d+)', signals_str)
        if m:
            return m.group(1)
        m = re.search(r'node-version.*?(\d+)', signals_str)
        if m:
            return m.group(1)

    elif language == 'go':
        m = re.search(r'go\.mod: go (\d+\.\d+)', signals_str)
        if m:
            return m.group(1)

    elif language == 'ruby':
        m = re.search(r'\.ruby-version:\s*(\d+\.\d+)', signals_str)
        if m:
            return m.group(1)

    elif language in ('c#', 'csharp'):
        m = re.search(r'global\.json sdk:\s*([\d.]+)', signals_str)
        if m:
            return m.group(1)[:3]
        m = re.search(r'TargetFramework.*?net(\d+)\.\d+', signals_str)
        if m:
            return f'{m.group(1)}.0'

    elif language == 'java':
        m = re.search(r'java-version.*?(\d+)', signals_str)
        if m:
            return m.group(1)

    elif language == 'rust':
        m = re.search(r'rust-toolchain.*?(\d+\.\d+)', signals_str)
        if m:
            return m.group(1)

    return _DEFAULT_VERSIONS.get(language, '')


def _determine_image(language: str, explore_data: dict[str, Any]) -> str | None:
    template = _LANGUAGE_IMAGE_MAP.get(language)
    if not template:
        return None
    version = _detect_version(language, explore_data)
    if not version:
        return None
    image = template.format(version=version)
    if validate_image(image):
        return image
    default_version = _DEFAULT_VERSIONS.get(language, '')
    if default_version and default_version != version:
        fallback = template.format(version=default_version)
        if validate_image(fallback):
            return fallback
    return None


async def _check_and_switch_image(
    ctx: Any, runtime: Any, task_config: dict, language: str, explore_data: dict
) -> dict[str, Any]:
    # Capability gate: only switch the runtime image if the dataset granted the
    # agent the `switch_runtime_image` tool. Datasets that don't expose this
    # tool (e.g. builder_t3_lite) want the agent constrained to the initial
    # image; setup must respect that intent.
    tools = task_config.get('tools', [])
    if 'code_mt/switch_runtime_image' not in tools:
        logger.debug(
            '[BUILDER_PRE_AGENT] switch_runtime_image tool not granted — skipping image switch',
        )
        return {'switched': False, 'runtime': runtime}

    # Gate: only switch if a recommended image exists AND is registered in
    # KNOWN_IMAGES. `_determine_image` already validates against the registry
    # (returns None when no template/version maps to a known image), so reaching
    # this point means the target is registered.
    recommended = _determine_image(language, explore_data)
    if not recommended:
        logger.debug(
            '[BUILDER_PRE_AGENT] No registered image candidate for language=%s — skipping switch',
            language,
        )
        return {'switched': False, 'runtime': runtime}

    current_image = task_config.get('base_image', '')
    safe_rec = recommended.lower().replace('-', '_').replace(':', '_').replace('/', '_').replace('.', '_')
    safe_cur = current_image.lower().replace('-', '_').replace(':', '_').replace('/', '_').replace('.', '_')

    if safe_rec in safe_cur:
        logger.debug('[BUILDER_PRE_AGENT] Image already matches: %s', recommended)
        return {'switched': False, 'runtime': runtime}

    resolved = resolve_mirror_image(recommended, current_image)
    logger.debug('[BUILDER_PRE_AGENT] Switching image: %s → %s', current_image, resolved)

    try:
        new_runtime = await ctx.replace_runtime('main', image=resolved)
        task_config['base_image'] = resolved
        ctx.tool_state.pop('terminal_session_id', None)

        # replace_runtime released the old container and acquired a fresh one
        # from the base image: /testbed is now EMPTY (no baked-in repo, no
        # cross-container /testbed volume). Rebuild it by re-running the task's
        # setup_commands, exactly as the switch_runtime_image tool does. Without
        # this the repo is gone for the rest of the episode and every probe
        # reports hunk_failed (the historical ~97-99% empty-testbed bug).
        await reinit_testbed_via_setup_commands(new_runtime, task_config, log_tag='BUILDER_PRE_AGENT')

        return {'switched': True, 'runtime': new_runtime, 'image': recommended}
    except TestbedInitError:
        # The image switch SUCCEEDED but /testbed could not be repopulated on the
        # fresh container. Do NOT swallow: an empty /testbed means the agent would
        # run to grading against no repo. Propagate to fail the episode.
        raise
    except Exception as e:
        logger.error('[BUILDER_PRE_AGENT] Image switch failed: %s', e)
        return {'switched': False, 'runtime': runtime, 'warning': f'Switch to {recommended} failed: {e}'}


# === Step 3: Probe Patches ===


async def _probe_patches(runtime: Any, task_config: dict) -> dict[str, Any]:
    golden_patch = task_config.get('golden_patch', '')
    test_patch = task_config.get('test_patch', '')
    result: dict[str, Any] = {}

    if golden_patch.strip():
        result['golden'] = await _probe_single_patch(runtime, golden_patch, 'golden_patch')
    if test_patch.strip():
        result['test'] = await _probe_single_patch(runtime, test_patch, 'test_patch')

    return result


async def _probe_single_patch(runtime: Any, patch: str, name: str) -> dict[str, str]:
    try:
        # Route the patch bytes through stdin (base64) instead of an inline heredoc:
        # a large patch overflows execve ARG_MAX ("[Errno 7] Argument list too long"),
        # which spuriously fails the probe and floods the logs. Same ARG_MAX-safe
        # pattern as _write_patch_files; git apply --check reads the temp file by path.
        probe_file = f'/tmp/{name}_probe.patch'
        cmd = f'base64 -d > {probe_file} && cd /testbed && git apply --check --ignore-whitespace {probe_file}'
        r = await runtime.run_command(
            command=cmd,
            stdin=base64.b64encode(patch.encode('utf-8')).decode('ascii'),
            timeout=60,
        )
        if r.return_code == 0:
            return {'status': 'clean', 'detail': ''}
        stderr = (r.stderr or r.stdout or '')[:300]
        if 'CRLF' in stderr or 'trailing whitespace' in stderr:
            return {'status': 'crlf_issue', 'detail': stderr}
        if 'already exists' in stderr:
            return {'status': 'already_applied', 'detail': stderr}
        return {'status': 'hunk_failed', 'detail': stderr}
    except Exception as e:
        return {'status': 'error', 'detail': str(e)[:200]}


def _format_probe_results(probe: dict[str, Any]) -> str:
    parts = []
    for name, info in probe.items():
        status = info.get('status', 'unknown')
        detail = info.get('detail', '')
        line = f'  {name}: {status}'
        if detail and status != 'clean':
            line += f' — {detail[:100]}'
        parts.append(line)
    return 'Patch probing:\n' + '\n'.join(parts)


# === Step 4: Pre-install ===

_INSTALL_COMMANDS = {
    'python': ['pip install -e ".[dev,test,tests]" 2>/dev/null || pip install -e . 2>/dev/null || true'],
    'javascript': ['npm ci --ignore-scripts 2>/dev/null || npm install --ignore-scripts 2>/dev/null || true'],
    'typescript': ['npm ci --ignore-scripts 2>/dev/null || npm install --ignore-scripts 2>/dev/null || true'],
    'ruby': ['bundle install --jobs=4 2>/dev/null || true'],
    'go': ['go mod download 2>/dev/null || true'],
    'rust': ['cargo fetch 2>/dev/null || true'],
    'java': ['mvn dependency:resolve -q 2>/dev/null || ./gradlew dependencies --quiet 2>/dev/null || true'],
    'csharp': ['dotnet restore 2>/dev/null || true'],
    'c#': ['dotnet restore 2>/dev/null || true'],
    'php': ['composer install --no-interaction --no-scripts 2>/dev/null || true'],
}


async def _pre_install(runtime: Any, language: str, explore_data: dict) -> list[dict[str, Any]]:
    commands = _INSTALL_COMMANDS.get(language, [])
    if not commands:
        return []

    results = []
    for cmd in commands:
        t0 = time.monotonic()
        try:
            r = await runtime.run_command(command=f'cd /testbed && {cmd}', timeout=300)
            duration = round(time.monotonic() - t0, 1)
            status = 'ok' if r.return_code == 0 else 'failed'
            if r.return_code == 137:
                status = 'oom'
            results.append({'cmd': cmd[:80], 'status': status, 'duration_s': duration, 'rc': r.return_code})
        except Exception as e:
            duration = round(time.monotonic() - t0, 1)
            if 'timeout' in str(e).lower():
                results.append({'cmd': cmd[:80], 'status': 'timeout', 'duration_s': duration})
            else:
                results.append({'cmd': cmd[:80], 'status': 'error', 'duration_s': duration, 'error': str(e)[:100]})

    return results


def _format_install_results(installs: list[dict[str, Any]]) -> str:
    parts = []
    for entry in installs:
        line = f'  `{entry["cmd"]}` → {entry["status"]} ({entry["duration_s"]}s)'
        if entry.get('error'):
            line += f' [{entry["error"]}]'
        parts.append(line)
    return 'Pre-install:\n' + '\n'.join(parts)


# === Build final markdown ===


def _build_auto_setup_markdown(
    results: list[str],
    warnings: list[str],
    language: str,
    task_config: dict,
    duration: float,
    explore_data: dict,
) -> str:
    parts = ['## Auto-Setup Results\n']
    current_image = image_alias(task_config.get('base_image', 'unknown'))
    parts.append('| Item | Value |')
    parts.append('|------|-------|')
    parts.append(f'| Language | {language} |')
    parts.append(f'| Image | `{current_image}` |')
    parts.append(f'| Duration | {duration}s |')
    parts.append('')

    for r in results:
        if r.strip():
            parts.append(r)
            parts.append('')

    if warnings:
        parts.append('**Warnings:**')
        for w in warnings:
            parts.append(f'- {w}')
        parts.append('')

    # CI install hints (from explore_repo) — give agent awareness of system deps
    ci_hints = explore_data.get('ci_install_commands', [])
    if ci_hints:
        parts.append('**CI Install Commands (from .github/workflows):**')
        for hint in ci_hints[:8]:
            parts.append(f'  {hint}')
        parts.append('')

    return '\n'.join(parts)
