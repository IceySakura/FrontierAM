"""System prompt rendering for the ``context_agent_general`` agent.

The original prompt text lives in ``system_prompts/general_{zh,en}.md`` and is
used whenever no explicit ``system_prompt_path`` is configured. Experimental
prompts are selected by path; ``{lang}`` in that path resolves to ``en`` or
``zh``.

Only rollout-invariant values are substituted here (``{max_rounds}``,
``{workspace}``, ``{cwd}``). The task's file list deliberately does NOT go in
the system prompt — it belongs to the opening user turn. Keeping the system
prompt identical across every task of a run means the gateway prefix cache hits
on it, which matters because this chain has no document preview to amortise.

Placeholders are substituted by targeted string replacement, not ``str.format``,
so literal braces in the Markdown (e.g. ``{'a': 1}`` in a code example) survive.
"""

from __future__ import annotations

from pathlib import Path

from nexus.extensions.agents.context_agent.tools.general.state import REPO_ROOT


_PROMPTS_DIR = Path(__file__).resolve().parent / 'system_prompts'

_DEFAULT_PROMPT_FILES: dict[str, str] = {
    'en': 'general_en.md',
    'zh': 'general_zh.md',
}


def _resolve_prompt_path(system_prompt_path: str | Path | None, language: str) -> Path:
    """Return the prompt file to load; explicit path wins over language."""
    lang = 'zh' if str(language).lower().startswith('zh') else 'en'
    if system_prompt_path:
        path = Path(str(system_prompt_path).replace('{lang}', lang)).expanduser()
        if not path.is_absolute():
            repo_path = (REPO_ROOT / path).resolve()
            packaged_path = _PROMPTS_DIR / path.name
            path = repo_path if repo_path.exists() else packaged_path
        if not path.exists():
            raise FileNotFoundError(f'system prompt file not found: {path}')
        return path

    return _PROMPTS_DIR / _DEFAULT_PROMPT_FILES[lang]


def build_general_system_prompt(
    *,
    language: str = 'en',
    max_rounds: int,
    workspace: str,
    cwd: str,
    system_prompt_path: str | Path | None = None,
) -> str:
    """Render the ``context_agent_general`` system prompt from Markdown.

    An explicit ``system_prompt_path`` may contain a ``{lang}`` placeholder.
    Without a path, the original prompt is loaded.
    """
    path = _resolve_prompt_path(system_prompt_path, language)
    template = path.read_text(encoding='utf-8')

    replacements = {
        '{max_rounds}': str(max_rounds),
        '{workspace}': str(workspace),
        '{cwd}': str(cwd),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


_MANIFEST_TEXT: dict[str, dict[str, str]] = {
    'en': {
        'header_one': 'You have been given 1 file:',
        'header_many': 'You have been given {n} files:',
        'url': 'URL',
        'path': 'local path',
        'note': (
            'Their contents are NOT included above and have not been read. '
            'Inspect them with the tools before deciding what to read.'
        ),
    },
    'zh': {
        'header_one': '你拿到了 1 个文件：',
        'header_many': '你拿到了 {n} 个文件：',
        'url': '网络链接',
        'path': '本地路径',
        'note': ('上面**没有**包含它们的任何内容，也没有人替你读过。先用工具探查，再决定读什么。'),
    },
}


def build_file_manifest(declared_files, question: str, language: str = 'en') -> str:
    """Build the opening user turn: the URI manifest followed by the question.

    This is the whole of what the agent starts with. There is no preview and no
    document content — each file appears only as a name and a URI, and every
    byte it ends up reading is the result of a tool call it chose to make.

    The manifest follows the same language as the system prompt: a Chinese
    system prompt with an English scaffold around the question is a seam the
    model can see, and three of the eval sets are Chinese.

    URIs are printed exactly as the dataset authored them, so the agent can
    copy one straight into a tool call (see ``GeneralState.resolve``).
    """
    lang = 'zh' if str(language).lower().startswith('zh') else 'en'
    text = _MANIFEST_TEXT[lang]

    lines: list[str] = []
    if declared_files:
        n = len(declared_files)
        lines.append(text['header_one'] if n == 1 else text['header_many'].format(n=n))
        lines.append('')
        for i, f in enumerate(declared_files, 1):
            kind = text['url'] if f.is_url else text['path']
            lines.append(f'{i}. {f.name}')
            lines.append(f'   {kind}: {f.uri}')
        lines.append('')
        lines.append(text['note'])
        lines.append('')
        lines.append('---')
        lines.append('')
    lines.append(question)
    return '\n'.join(lines)
