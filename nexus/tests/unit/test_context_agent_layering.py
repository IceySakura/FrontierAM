"""Only General and Native remain as Context Agent serving entry points."""

from pathlib import Path

from nexus.extensions.agents.context_agent import profiles
from nexus.extensions.tasks.context_agent.enterprisebench_native import eval as native_eval
from nexus.extensions.tasks.context_agent.eval import enterprisebench


def test_only_general_profile_is_registered():
    assert {p.agent_name: p.task_name for p in profiles.PROFILES.values()} == {
        'context_agent_general': 'context_agent_general_task',
    }
    assert profiles.get_profile('general') is profiles.GENERAL_PROFILE
    assert profiles.get_profile('context_agent_general') is profiles.GENERAL_PROFILE


def test_enterprise_eval_uses_native_implementation():
    assert enterprisebench.main is native_eval.main
    assert enterprisebench.aggregate is native_eval.aggregate


def test_retired_implementations_are_removed():
    extensions = Path(__file__).resolve().parents[2] / 'nexus/extensions'
    for relative in (
        'agents/context_agent/agent.py',
        'agents/context_agent/sft_eval_agent.py',
        'agents/context_agent/tools/legacy',
        'agents/context_agent/tools/document',
        'tasks/context_agent/task.py',
        'tasks/context_agent/sft_eval_task.py',
        'tasks/context_agent/enterprisebench',
        'projects/context_agent_already_preview',
        'projects/context_agent_before_preview',
        'projects/formal_nexus_context_agent',
    ):
        assert not (extensions / relative).exists()
