"""Registry of hy-live sandbox example cases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


EXAMPLE_DIR = Path(__file__).resolve().parent
RuntimeFeature = Literal['client_questions']
Backend = Literal['codebuddy', 'workbuddy']
"""Working agent the example runs on; it is the hy-live ``backend_adapter`` and the ``work_agent_configs`` key."""
BACKENDS: tuple[Backend, ...] = ('codebuddy', 'workbuddy')
CaseCheck = Literal[
    'question_answered_during_run',
    'steer_replaced_running_run',
    'acp_artifact_presented_and_opened',
    'native_web_search_succeeded',
    'native_web_fetch_succeeded',
    'native_web_fetch_two_sources',
    'tasks_done',
]
"""Case-specific validation on top of the structural checks every example run must pass."""


@dataclass(frozen=True)
class ExampleCase:
    """One sandbox example: frozen data and a review checklist."""

    name: str
    title: str
    spec_relpath: str
    data_relpath: str
    runtime_features: frozenset[RuntimeFeature] = frozenset()
    """Runtime image features the case cannot run without; the CLI refuses it until they are switched on."""
    backends: frozenset[Backend] = frozenset(BACKENDS)
    """Working agents the case is written for; the CLI refuses any other ``--backend``."""
    case_checks: tuple[CaseCheck, ...] = ()
    """Checks of ``validate.validate_run`` that only this case's flow makes true."""

    @property
    def spec_path(self) -> Path:
        return EXAMPLE_DIR / self.spec_relpath

    @property
    def data_path(self) -> Path:
        return EXAMPLE_DIR / self.data_relpath


_CLIENT_QUESTIONS: RuntimeFeature = 'client_questions'
_NEEDS_CLIENT_QUESTIONS = frozenset((_CLIENT_QUESTIONS,))

CASES: dict[str, ExampleCase] = {
    'single_snake': ExampleCase(
        name='single_snake',
        title='权限都允许 + 完成后调速续办',
        spec_relpath='single_snake.md',
        data_relpath='data/single_snake.json',
    ),
    'mixed_task_and_cancel': ExampleCase(
        name='mixed_task_and_cancel',
        title='搜索 + 双任务 + 取消后继续',
        spec_relpath='mixed_task_and_cancel.md',
        data_relpath='data/mixed_task_and_cancel.json',
    ),
    'multi_task_query': ExampleCase(
        name='multi_task_query',
        title='四项工作 + 多次问进度 + 逐项权限',
        spec_relpath='multi_task_query.md',
        data_relpath='data/multi_task_query.json',
    ),
    'ask_user_question': ExampleCase(
        name='ask_user_question',
        title='CodeBuddy 问游戏类型（AskUserQuestion）+ 用户回答后接着做 + 打开产物',
        spec_relpath='ask_user_question.md',
        data_relpath='data/ask_user_question.json',
        runtime_features=_NEEDS_CLIENT_QUESTIONS,
    ),
    'ask_permission_reject': ExampleCase(
        name='ask_permission_reject',
        title='权限通知后用户拒绝',
        spec_relpath='ask_permission_reject.md',
        data_relpath='data/ask_permission_reject.json',
    ),
    'backend_web_research': ExampleCase(
        name='backend_web_research',
        title='后脑联网调研 Typesafe 的 JEV：搜索、抓取来源、生成报告',
        spec_relpath='backend_web_research.md',
        data_relpath='data/backend_web_research.json',
        case_checks=('native_web_search_succeeded', 'native_web_fetch_two_sources', 'tasks_done'),
    ),
    'backend_web_fetch': ExampleCase(
        name='backend_web_fetch',
        title='后脑原生 WebFetch 读取指定官方文档并生成摘要',
        spec_relpath='backend_web_fetch.md',
        data_relpath='data/backend_web_fetch.json',
        case_checks=('native_web_fetch_succeeded', 'tasks_done'),
    ),
    'workbuddy_followup_and_steer': ExampleCase(
        name='workbuddy_followup_and_steer',
        title='WorkBuddy 运行中追问（session/fork）+ 中途改需求（steer）+ ACP 产物展示与打开',
        spec_relpath='workbuddy_followup_and_steer.md',
        data_relpath='data/workbuddy_followup_and_steer.json',
        backends=frozenset(('workbuddy',)),
        case_checks=(
            'question_answered_during_run',
            'steer_replaced_running_run',
            'acp_artifact_presented_and_opened',
        ),
    ),
}


def get_case(name: str) -> ExampleCase:
    try:
        return CASES[name]
    except KeyError as exc:
        known = ', '.join(CASES)
        raise KeyError(f'Unknown example case {name!r}. Known: {known}') from exc
