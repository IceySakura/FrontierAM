"""针对已提交 rubric 导入的沙箱 + mockserver E2E 冒烟测试。

该测试使用真实的 ``run_aidata`` 流水线并搭配一个桩 agent：

* ``resolve_case`` 解析用例清单。
* main 沙箱被获取。
* 声明的 ``services[]`` mockserver sidecar 被启动。
* 加载 MCP 工具并运行 dump_state 回读。
* 加载已提交的 ``rubric.yaml`` 并由 ``run_rubric_async`` 分发。

我们对 LLM judge 打 monkeypatch，避免调用外部模型。 断言并非任务在空
agent 轨迹下能得高分；而是 rubric 能被导入并分发，且不出现 DSL 结构错误。
"""

from __future__ import annotations

import os

from types import SimpleNamespace
from typing import Any

import pytest

from nexus.extensions.tasks.aidata.engine.case_spec import resolve_case
from nexus.extensions.tasks.aidata.engine.models import AidataTaskInput
from nexus.extensions.tasks.aidata.engine.task import run_aidata
from nexus.extensions.tasks.aidata.rubric import legacy_v2
from nexus.extensions.tasks.aidata.rubric.primitives import judge_claim

from .conftest import ALL_CASE_IDS, CASES_ROOT


_DEFAULT_E2E_CASES = (
    # 覆盖 HTTP sidecar mockserver + dump_state 选择器导入。
    'work/poc/dump_state_poc',
    # 覆盖 mock-server sidecar + initial_trajectory + ask_user + scope。
    'wx_test_ride_ambiguous',
)

_IMPORT_FAILURE_VERDICTS = {
    'ERROR',
    'INVALID',
    'UNKNOWN_ASSERT_TYPE',
    'UNKNOWN_KIND',
}


def _selected_case_ids() -> list[str]:
    """选择一个成本可控的默认集合，并提供环境开关以扩大覆盖范围。"""
    available = set(ALL_CASE_IDS)
    raw = os.getenv('AIDATA_E2E_RUBRIC_CASES', '').strip()
    if raw:
        requested = [item.strip() for item in raw.split(',') if item.strip()]
        return [case_id for case_id in requested if case_id in available]

    if os.getenv('AIDATA_E2E_ALL_RUBRICS') in {'1', 'true', 'TRUE', 'yes', 'YES'}:
        return [case_id for case_id in ALL_CASE_IDS if (CASES_ROOT / case_id / 'rubric.yaml').exists()]

    return [case_id for case_id in _DEFAULT_E2E_CASES if case_id in available]


async def _stub_run_agent(**_: Any) -> Any:
    """在保留 run_aidata 契约的同时避免 LLM 花费。"""
    return SimpleNamespace(main_trajectory=[])


@pytest.mark.e2e
@pytest.mark.parametrize('case_id', _selected_case_ids())
async def test_e2e_committed_rubric_imports_with_sandbox_and_mockserver(
    case_id: str,
    per_case_ctx,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not CASES_ROOT.is_dir():
        pytest.skip(f'cases root not found: {CASES_ROOT}')
    if not (CASES_ROOT / case_id / 'rubric.yaml').exists():
        pytest.skip(f'{case_id}: no committed rubric.yaml')

    def _fake_judge(*_: Any, **__: Any) -> tuple[str, str]:
        return 'Y', 'Y'

    monkeypatch.setattr(judge_claim, '_llm_judge', _fake_judge)
    monkeypatch.setattr(legacy_v2, '_llm_judge', _fake_judge)

    task_input = AidataTaskInput(case_id=case_id, run_agent=_stub_run_agent)
    resolve_case(task_input)
    if task_input.pending_upstream:
        pytest.skip(f'{case_id}: pending upstream - {task_input.pending_upstream}')

    output = await run_aidata(task_input)

    assert output.error_message is None, (
        f'{case_id}: run_aidata failed before/during rubric import: {output.error_message!r}. '
        f'Output: {output.model_dump()}'
    )
    assert output.rubric_results, f'{case_id}: committed rubric produced no rubric_results'

    failures = []
    for row in output.rubric_results:
        verdict = row.get('verdict')
        if verdict not in _IMPORT_FAILURE_VERDICTS:
            continue
        failures.append(
            f'{row.get("id") or "<missing id>"} '
            f'verdict={verdict} verify={row.get("assert_type") or row.get("verify")} '
            f'detail={row.get("detail")}'
        )

    assert not failures, f'{case_id}: rubric import/DSL failures:\n' + '\n'.join(failures)
