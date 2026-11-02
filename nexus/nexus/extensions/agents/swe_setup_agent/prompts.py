"""Prompt templates for SWE Setup Agent (In-Place Mode).

The agent sets up the environment in-place, writes eval.sh, verifies fail2pass,
and calls submit_eval. Language-specific installation guides are loaded from
the user_prompts/ directory.
"""

from __future__ import annotations

import re

from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from nexus.extensions.agents.swe_setup_agent.tools import AutoSetupResult


# ============== Constants ==============

MIRROR_REGISTRY = 'mirrors.tencent.com/hunyuan_yanguan/t3build_repolaunch_midtrain'


# ============== System Prompt ==============

SYSTEM_PROMPT = """You are an expert DevOps engineer specializing in setting up reproducible development environments.
Your task is to configure the current runtime environment to reproduce software issues and run tests."""


# ============== Helper Functions ==============


def _safe_format(template: str, **kwargs) -> str:
    """Format a template string, replacing only known {key} placeholders.

    Handles {{ }} escape sequences ({{ → {, }} → }).
    Leaves unknown {placeholder} patterns untouched (e.g. awk '{print $1}').

    This avoids KeyError when templates contain shell/awk code with braces
    that are NOT meant to be Python format variables.
    """
    # Step 1: protect {{ and }} from the regex by substituting unique sentinels
    sentinel_open = '\x01'
    sentinel_close = '\x02'
    result = template.replace('{{', sentinel_open).replace('}}', sentinel_close)

    # Step 2: replace only {key} patterns where key is a known kwarg
    def _replacer(m: re.Match) -> str:
        key = m.group(1)
        return str(kwargs[key]) if key in kwargs else m.group(0)

    result = re.sub(r'\{([^{}]+)\}', _replacer, result)

    # Step 3: restore sentinels to literal braces
    return result.replace(sentinel_open, '{').replace(sentinel_close, '}')


def _load_template(template_dir: str, filename: str) -> str:
    """Load a template file from the specified directory.

    Args:
        template_dir: Directory name (e.g., 'user_prompts')
        filename: Template filename (e.g., 'base.md', 'python.md')

    Returns:
        Template content as string
    """
    current_dir = Path(__file__).parent
    template_path = current_dir / template_dir / filename

    if template_path.exists():
        try:
            with open(template_path, encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f'Warning: Failed to load template {template_path}: {e}')

    return ''


def _load_language_guide(language: str) -> str:
    """Load language-specific installation guide.

    Args:
        language: Programming language name

    Returns:
        Language-specific guide content
    """
    language_map = {
        'python': 'python.md',
        'javascript': 'javascript.md',
        'typescript': 'typescript.md',
        'java': 'java.md',
        'go': 'go.md',
        'golang': 'go.md',
        'rust': 'rust.md',
        'ruby': 'ruby.md',
        'php': 'php.md',
        'c++': 'cpp.md',
        'cpp': 'cpp.md',
        'c': 'c.md',
        'c#': 'csharp.md',
        'csharp': 'csharp.md',
    }

    lang_key = language.lower().strip()
    guide_file = language_map.get(lang_key)

    if not guide_file:
        return ''

    return _load_template('user_prompts', guide_file)


# ============== Patch Display ==============

_PATCH_INLINE_LIMIT = 4_000  # ~1k tokens; larger patches are summarised instead of inlined


def _patch_display(patch: str, name: str) -> str:
    """Return inline patch content, or a file-list summary when the patch is too large.

    Args:
        patch: Raw patch text (may be empty).
        name:  Logical name used in the /tmp path, e.g. 'golden_patch' or 'test_patch'.
    """
    if not patch or not patch.strip():
        return f'(empty — no {name} provided)'
    if len(patch) <= _PATCH_INLINE_LIMIT:
        return patch.strip()

    # Build a compact summary: changed files + diff-stat
    files = re.findall(r'^diff --git a/(.+) b/.+$', patch, re.MULTILINE)
    files = list(dict.fromkeys(files))  # deduplicate, preserve order
    added = len(re.findall(r'^\+[^+]', patch, re.MULTILINE))
    removed = len(re.findall(r'^-[^-]', patch, re.MULTILINE))

    header = (
        f'(patch too large to inline — {len(patch):,} chars / ~{len(patch) // 4:,} tokens)\n'
        f'Full patch written to /tmp/{name}.patch\n\n'
    )

    if not files:
        # No "diff --git" headers (raw unified diff or non-standard format).
        # Fall back to showing the first 2000 chars so agent can see the structure.
        preview = patch.strip()[:2000]
        if len(patch) > 2000:
            preview += f'\n... [{len(patch) - 2000:,} chars omitted — read /tmp/{name}.patch for full content]'
        return header + f'**Diff stat:** +{added} / -{removed} lines total\n\n' + preview

    file_list = '\n'.join(f'  - {f}' for f in files[:30])
    if len(files) > 30:
        file_list += f'\n  ... and {len(files) - 30} more files'

    return header + f'**Changed files ({len(files)}):** +{added} / -{removed} lines total\n' + file_list


# ============== Auto Setup Results Formatting ==============


def format_auto_setup_results(result: AutoSetupResult) -> str:
    """Format AutoSetupResult into a concise markdown section for prompt injection.

    Target: ~2500 chars (~600 tokens).
    """
    lines = ['## Auto Setup Results', '']

    # Environment
    img_note = ' (switched from previous)' if result.image_switched else ''
    lines.append(f'**Image:** `{result.selected_image}`{img_note}')
    lines.append(f'**Language:** {result.language} {result.language_version}')
    if result.package_files:
        lines.append(f'**Package files:** {", ".join(result.package_files[:10])}')
    if result.test_dirs:
        td_strs = []
        for td in result.test_dirs[:5]:
            count = len(td.get('entries', []))
            td_strs.append(f'`{td["path"]}/` ({count} entries)')
        lines.append(f'**Test dirs:** {", ".join(td_strs)}')
    if result.build_systems:
        lines.append(f'**Build systems:** {", ".join(result.build_systems)}')
    if result.version_signals:
        lines.append(f'**Version signals:** {"; ".join(result.version_signals[:5])}')

    # Patch probing — use longer limit for non-clean statuses (hunk details matter)
    lines.append('')
    lines.append('### Patch Probing')
    gp_limit = 200 if result.golden_patch_status == 'clean' else 400
    tp_limit = 200 if result.test_patch_status == 'clean' else 400
    lines.append(f'- golden_patch: **{result.golden_patch_status}** — {result.golden_patch_detail[:gp_limit]}')
    lines.append(
        f'- test_patch: **{result.test_patch_status}** (on {result.test_patch_applied_on}) — {result.test_patch_detail[:tp_limit]}'
    )
    lines.append(f'- Strategy: **{result.eval_strategy}**')

    # Pre-install
    if result.install_commands_run:
        lines.append('')
        lines.append('### Pre-Install')
        lines.append('| # | Command | Status | Time |')
        lines.append('|---|---------|--------|------|')
        for i, entry in enumerate(result.install_commands_run, 1):
            cmd_short = entry['cmd'][:60]
            status = entry['status'].upper()
            dur = f'{entry["duration_s"]}s'
            lines.append(f'| {i} | `{cmd_short}` | {status} | {dur} |')
            # Show error detail for failed/oom/timeout commands
            if entry.get('error') and entry['status'] not in ('ok', 'skipped'):
                lines.append(f'|   | Error: {entry["error"][:120]} | | |')
        lines.append(f'**Summary:** {result.install_summary}')

    # CI install hints (surfaced when pre-install has failures, filtered by language relevance)
    has_install_failure = any(e['status'] in ('failed', 'error') for e in result.install_commands_run)
    if result.ci_install_commands and has_install_failure:
        # Filter hints to language-relevant install commands (apt is always relevant)
        _lang_install_kw: dict[str, list[str]] = {
            'python': ['pip install', 'conda install', 'apt-get install', 'apt install'],
            'rust': ['cargo', 'rustup', 'apt-get install', 'apt install'],
            'go': ['go install', 'go get', 'go mod', 'apt-get install', 'apt install'],
            'golang': ['go install', 'go get', 'go mod', 'apt-get install', 'apt install'],
            'javascript': ['npm install', 'yarn', 'pnpm', 'apt-get install', 'apt install'],
            'typescript': ['npm install', 'yarn', 'pnpm', 'apt-get install', 'apt install'],
            'java': ['mvn ', 'gradle', 'apt-get install', 'apt install'],
            'php': ['composer install', 'apt-get install', 'apt install'],
            'ruby': ['bundle install', 'gem install', 'apt-get install', 'apt install'],
            'c': ['apt-get install', 'apt install', 'make install'],
            'c++': ['apt-get install', 'apt install', 'make install'],
            'cpp': ['apt-get install', 'apt install', 'make install'],
        }
        lang_kws = _lang_install_kw.get(result.language.lower().strip(), [])
        if lang_kws:
            filtered_hints = [cmd for cmd in result.ci_install_commands if any(kw in cmd.lower() for kw in lang_kws)]
        else:
            filtered_hints = result.ci_install_commands
        if filtered_hints:
            lines.append('')
            lines.append('### CI Install Hints (from .github/workflows)')
            for ci_cmd in filtered_hints[:5]:
                lines.append(f'- `{ci_cmd[:120]}`')

    # Warnings
    if result.warnings:
        lines.append('')
        lines.append('### Warnings')
        for w in result.warnings:
            lines.append(f'- {w[:200]}')
    else:
        lines.append('')
        lines.append('### Warnings')
        lines.append('- (none)')

    text = '\n'.join(lines)
    # Hard cap at 3000 chars to stay within token budget
    if len(text) > 3000:
        text = text[:2900] + '\n... (truncated)'
    return text


# ============== User Prompt ==============


def get_user_prompt(
    repo: str,
    base_commit: str,
    language: str,
    testbed_dir: str,
    base_image: str,
    problem_statement: str,
    test_patch: str,
    golden_patch: str,
    available_images: str,
    auto_setup_result: AutoSetupResult | None = None,
) -> str:
    """Generate user prompt for environment setup.

    Args:
        repo: Repository name (e.g., 'owner/repo')
        base_commit: Base commit hash
        language: Programming language
        testbed_dir: Testbed directory path (usually /testbed)
        base_image: Current base image
        problem_statement: Issue description
        test_patch: Test patch content (may be empty)
        golden_patch: Golden patch content (the fix)
        available_images: Available base images list
        auto_setup_result: Results from auto_setup (if run)

    Returns:
        Formatted prompt
    """
    # Load base template
    base_template = _load_template('user_prompts', 'base.md')

    if not base_template:
        # Fallback to inline template if file not found
        base_template = """# Environment Setup Task (In-Place Installation Mode)

**Repository**: {repo}
**Base Commit**: {base_commit}
**Language**: {language}
**Testbed**: {testbed_dir}
**Base Image**: `{base_image}`

Please set up the environment and create an eval script.

{language_guide}
"""

    # Load language-specific installation guide
    language_guide = _load_language_guide(language)

    # Format language guide with instance-specific values.
    # _safe_format handles {{ }} bash escapes and leaves unknown {placeholders} (e.g. awk) untouched.
    if language_guide:
        language_guide = _safe_format(
            language_guide, base_image=base_image, testbed_dir=testbed_dir, reg=MIRROR_REGISTRY
        )

    # Format test_patch / golden_patch display (summarise when too large to inline)
    test_patch_display = _patch_display(test_patch, 'test_patch')
    golden_patch_display = _patch_display(golden_patch, 'golden_patch')

    # Format auto_setup results
    auto_setup_results = ''
    if auto_setup_result is not None:
        auto_setup_results = format_auto_setup_results(auto_setup_result)

    # Format the main template
    return _safe_format(
        base_template,
        repo=repo,
        base_commit=base_commit,
        language=language,
        testbed_dir=testbed_dir,
        base_image=base_image,
        problem_statement=problem_statement,
        language_guide=language_guide,
        test_patch=test_patch_display,
        golden_patch=golden_patch_display,
        available_images=available_images,
        auto_setup_results=auto_setup_results,
        reg=MIRROR_REGISTRY,
    )
