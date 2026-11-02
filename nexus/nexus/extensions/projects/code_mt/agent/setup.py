"""Turn-0 environment setup for code_mt_agent.

Env prep (image switch, dependency pre-install, /tmp patch helpers, repo
exploration) is OWNED by the AGENT, not the task — this is what lets
code_mt_task stay agent-agnostic (the task just calls
run_agent(user_prompt, working_dir) and can drive any agent). The setup runs
EXACTLY ONCE, on the first entry to the agent loop; MT retry turns must not
re-run it (the runtime already carries turn-0's state). A flag on the shared
ExecutionContext.tool_state guards this across both call modes (tool_call /
text_parse) and across MT turns, since text_parse_loop keeps no resume state
of its own.

The setup_strategy is a dataset-declared `module::func` import path (e.g. the
builder autosetup). It returns a dict of inject_vars that are merged into
ctx.task_config so the agent's prompt render (build_initial_messages) and the
task's templates can reference them.
"""

from __future__ import annotations

import logging

from typing import Any

from nexus.core import require_context
from nexus.core.registry import import_from_path
from nexus.extensions.projects.code_mt.agent.autosetup import TestbedInitError
from nexus.trace import get_current_span_id, update_span


logger = logging.getLogger(__name__)


# Guard key on the shared ExecutionContext.tool_state. Set after the first
# agent turn's setup completes so MT retry turns skip it.
SETUP_DONE_KEY = '_code_mt_setup_done'


# Backward-compat import-path aliases. autosetup/images/tools have moved twice:
#   1. task package -> agent package (agent/task decoupling), and
#   2. agents/code_mt -> projects/code_mt/agent (this project-mode move).
# Existing task.toml files (144k+ under data/code_mt/0710, plus the 0720/0802
# sets) still declare setup_strategy with a STALE prefix — either the original
# "nexus.extensions.tasks.code_mt.autosetup::..." or the intermediate
# "nexus.extensions.agents.code_mt.autosetup::...". We rewrite BOTH stale
# prefixes to the current location here so the DATA stays untouched. (The
# `cleanup` module referenced by eval_cleanup_strategy lives in the task package
# and moved with it to projects.code_mt.task.cleanup, so its path resolves
# directly and needs no alias.)
_NEW = 'nexus.extensions.projects.code_mt.agent'
_MODULE_PATH_ALIASES = {
    # original task-package prefix
    'nexus.extensions.tasks.code_mt.autosetup': f'{_NEW}.autosetup',
    'nexus.extensions.tasks.code_mt.images': f'{_NEW}.images',
    'nexus.extensions.tasks.code_mt.tools': f'{_NEW}.tools',
    # intermediate agents-package prefix
    'nexus.extensions.agents.code_mt.autosetup': f'{_NEW}.autosetup',
    'nexus.extensions.agents.code_mt.images': f'{_NEW}.images',
    'nexus.extensions.agents.code_mt.tools': f'{_NEW}.tools',
}


def _resolve_import_alias(import_path: str) -> str:
    """Rewrite a stale `module::obj` path to its post-decoupling location."""
    if '::' not in import_path:
        return import_path
    module_path, obj = import_path.split('::', 1)
    new_module = _MODULE_PATH_ALIASES.get(module_path)
    if new_module:
        logger.debug('[TURN0_SETUP] alias %s -> %s', module_path, new_module)
        return f'{new_module}::{obj}'
    return import_path


def _extract_runtime_id(runtime: Any) -> str:
    base_url = getattr(runtime, 'base_url', '')
    if base_url:
        parts = str(base_url).rstrip('/').split('/')
        return parts[-1] if parts else ''
    return ''


async def run_turn0_setup(agent_input: Any) -> None:
    """Run the dataset's setup_strategy once, on the first agent turn.

    No-op when no strategy is configured or setup already ran this task-run.
    Failures are swallowed (logged) — a transient setup hiccup must not abort
    the agent, which can still work from the base image.
    """
    setup_strategy = getattr(agent_input, 'setup_strategy', '') or ''
    if not setup_strategy:
        return

    ctx = require_context()
    if ctx.tool_state.get(SETUP_DONE_KEY):
        return
    ctx.tool_state[SETUP_DONE_KEY] = True

    span_id = get_current_span_id()
    runtime = await ctx.get_runtime()
    before_id = _extract_runtime_id(runtime)

    try:
        fn = import_from_path(_resolve_import_alias(setup_strategy))
        result = await fn(ctx=ctx, runtime=runtime, agent_input=agent_input)
        if isinstance(result, dict):
            ctx.task_config.update(result)
            logger.debug(
                '[TURN0_SETUP] %s returned %d inject var(s): %s',
                setup_strategy,
                len(result),
                list(result.keys())[:8],
            )
            # DYNAMIC injection (hook-owned): fill the single {auto_setup_results}
            # placeholder the convertor baked into the turn-0 instruction. Uses
            # str.replace (NOT str.format) so the instruction's literal braces
            # (e.g. SWEBuilder's {host}) are never treated as fields. This is the
            # ONLY place the agent touches the prompt — a hook filling its own
            # runtime-produced value into a pre-baked slot. If the placeholder is
            # absent (non-builder), replace is a no-op.
            asr = result.get('auto_setup_results')
            if asr is not None and getattr(agent_input, 'user_prompt', None):
                agent_input.user_prompt = agent_input.user_prompt.replace('{auto_setup_results}', str(asr))
    except TestbedInitError:
        # Hard environment-build failure: an image switch left /testbed empty
        # (repo checkout failed). Must NOT be swallowed — otherwise the agent
        # runs to grading against no repo and emits empty-patch / reward-hacking
        # rollouts. Propagate to the agent loop -> UNHANDLED_ERROR (fail the run).
        logger.error('[TURN0_SETUP] setup_strategy %s: testbed init failed, aborting turn-0', setup_strategy)
        raise
    except Exception as e:
        logger.warning('[TURN0_SETUP] setup_strategy %s failed: %s', setup_strategy, e)

    # The strategy may have switched the runtime image (ctx.replace_runtime).
    # Re-emit the (possibly new) runtime id so hytuner captures both the
    # pre-switch and post-switch ids (HytunerProcessor dedup-appends).
    after_runtime = await ctx.get_runtime()
    after_id = _extract_runtime_id(after_runtime)
    if span_id and after_id and after_id != before_id:
        update_span(span_id, attributes={'hytuner.runtime_id': after_id})
