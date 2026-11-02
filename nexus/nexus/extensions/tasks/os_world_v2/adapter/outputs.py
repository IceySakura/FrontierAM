"""V2 stage contract, reusing V1 status semantics without its infeasible marker."""

import json

from typing import Any

from nexus.extensions.tasks.os_world_v2.outputs import (
    OSWorldAgentOutput,
    OSWorldOutputContractHook,
    _parent_result,
    apply_host_llm_evidence,
    compact_captured_agent_output,
    normalize_agent_output,
    run_task_agent,
    summarize_task_output,
)
from nexus.runner.hooks.base import run_hook_registry


def normalize_agent(value: Any, *, harness: str | None = None) -> OSWorldAgentOutput:
    data = value.model_dump(mode='python') if hasattr(value, 'model_dump') else dict(value)
    if not isinstance(value, OSWorldAgentOutput) and harness == 'cc':
        root = _parent_result(data.get('output') or '')
        if not data.get('exit_status'):
            if data.get('termination_reason') == 'cc_timeout' and data.get('budget_stop_confirmed'):
                data.update(success=True, exit_status='budget_exhausted')
            elif (
                data.get('success') is not False
                and data.get('exit_code') == 0
                and root.get('subtype') == 'success'
                and not root.get('is_error')
            ):
                data['exit_status'] = 'finished'
    result = normalize_agent_output(
        value if isinstance(value, OSWorldAgentOutput) else data, harness=harness
    ).model_copy()
    result = apply_host_llm_evidence(result, value)
    if result.conclusion == 'infeasible':
        result.conclusion = None
    return result


async def run_agent(run, **kwargs):
    """V1 provider-error observation plus V2's parent/result interpretation."""
    from nexus.core import require_context

    ctx = require_context()
    ctx.tool_state.pop('osworld_agent_output', None)
    raw = await run_task_agent(run, **kwargs)
    recorded = ctx.tool_state.get('osworld_agent_output')
    result = normalize_agent(
        recorded if recorded is not None else raw,
        harness='cc' if ctx.agent_name in {'claude_code_multiturn', 'claude_code_native_bridge'} else None,
    )
    # The profile recorded its output before the bridge attached host request outcomes.
    result = apply_host_llm_evidence(result, raw)
    ctx.tool_state['osworld_agent_output'] = result
    compact_captured_agent_output(raw, result)
    return result


def summarize(value):
    """Preserve V2 scored flag: existing platform valid rules require it."""
    data = value.model_dump(mode='json') if hasattr(value, 'model_dump') else dict(value)
    evaluator = data.get('evaluator_result') or {}
    # The common projection also sets V1-only setup_results; project a dict so a
    # strict V2 task model is never assigned a nonexistent field.
    result = summarize_task_output(data)
    result.pop('setup_results', None)
    result['evaluator_result'] = {
        **result['evaluator_result'],
        **{key: evaluator[key] for key in ('scored', 'benchmark_release', 'source_commit') if key in evaluator},
    }
    return result


@run_hook_registry.register('osworld_v2_output_contract')
class V2OutputContractHook(OSWorldOutputContractHook):
    task_name = 'os_world_v2'
    summarize = staticmethod(summarize)


def save_agent_evidence(ctx, directory):
    agent = ctx.tool_state.get('osworld_agent_output')
    if agent is not None:
        (directory / 'agent-output.json').write_text(
            json.dumps(agent.model_dump(mode='json'), ensure_ascii=False, indent=2)
        )
