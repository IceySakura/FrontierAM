"""Backward-compat module-path aliasing for code_mt data-side import strings.

code_mt's agent+task packages moved twice:
  1. task package -> agent package (agent/task decoupling), then
  2. agents/code_mt + tasks/code_mt -> projects/code_mt/{agent,task} (this move).

Existing task.toml files (never mutated — data is read-only) still declare
import strings with a STALE prefix, e.g.
    prompt_builder_import = "nexus.extensions.tasks.code_mt.judge_builders::rubric_qa"
    setup_strategy        = "nexus.extensions.agents.code_mt.autosetup::..."

These are resolved via ``import_from_path`` (``module::obj``) which does NOT go
through the loader; a stale prefix would raise ModuleNotFoundError. This helper
rewrites the stale prefix segment to the current projects.code_mt.* location so
the DATA stays untouched. Registry names are unrelated (unchanged by the move).
"""

from __future__ import annotations


# Map each historical package prefix to where the module lives now. Order is
# longest-prefix-first so 'agents.code_mt' matches before a hypothetical
# 'agents' catch-all (there is none, but the sort makes intent explicit).
_PREFIX_MAP: dict[str, str] = {
    # judge_builders / scoring / loader / models etc. live in the TASK subpkg.
    'nexus.extensions.tasks.code_mt': 'nexus.extensions.projects.code_mt.task',
    # autosetup / images / tools live in the AGENT subpkg. Note: the original
    # task-package prefix for autosetup/images/tools is handled by setup.py's
    # own alias map (turn-0 setup path); here we only need the code_mt.* -> new
    # rewrite for judge/parser import strings, which point at the task subpkg.
    'nexus.extensions.agents.code_mt': 'nexus.extensions.projects.code_mt.agent',
}


def canonicalize_import_path(import_path: str | None) -> str | None:
    """Rewrite a stale ``module[::obj]`` string to its current location.

    Idempotent: paths already under projects.code_mt.* pass through unchanged.
    ``None``/empty pass through unchanged.
    """
    if not import_path:
        return import_path
    module_part, sep, obj = import_path.partition('::')
    for stale, current in _PREFIX_MAP.items():
        if module_part == stale or module_part.startswith(stale + '.'):
            module_part = current + module_part[len(stale) :]
            break
    return f'{module_part}{sep}{obj}' if sep else module_part
