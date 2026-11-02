"""动态层：每个 case 都能拉起自己的沙箱 + MCP 目录。

我们端到端运行 ``run_aidata``，只做一处改动：``run_agent`` 被打桩为
立即返回空轨迹，因此步骤 7（agent 循环）/ 8（state 下载，空时无害）/
9-11（rubric、评分）都会执行但不会消耗 LLM。 被演练的动态部分
是 agent **之前** 的一切——我们要锁定的契约：

* case.yaml 清单 → ``AidataTaskInput`` 转换
* 辅助 ``services[]`` 容器起来并通过 ``ready_check``
* ``connections[]`` 占位符解析为真实 URL
* 每个 case 的 ``bootstrap_script`` 在 main 内运行且无错
* MCP 目录加载——内置 mock + 每 case 额外项 + wx_agent_hub
  展开，所有 MCP 会话干净初始化
* engine 步骤 1 工作区准备（``rm -rf``、``mkdir -p``、skills
  软链）不会跨运行泄漏状态

这是本可以在真正上线前捕获
``Failed to initialize MCP session: NoneType: None`` 回归的冒烟测试。

每个 case 在自己新鲜的 ExecutionContext + main runtime 中运行，因此
测试套件在 ``pytest-xdist`` 下也是安全的。 并行执行方式::

    uv run pytest nexus/extensions/tasks/aidata/tests/test_cases_startup.py -n 8
"""

from __future__ import annotations

import logging

from types import SimpleNamespace
from typing import Any

import pytest

from nexus.extensions.tasks.aidata.engine.case_spec import resolve_case
from nexus.extensions.tasks.aidata.engine.models import AidataTaskInput
from nexus.extensions.tasks.aidata.engine.task import run_aidata

from .conftest import ALL_CASE_IDS


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 打桩
# ---------------------------------------------------------------------------


async def _stub_run_agent(**kwargs: Any) -> Any:
    """真实 agent 循环的零成本替身。

    返回一个形如 ``AgentRunOutput`` 的对象，其轨迹为空，
    这样 ``state['agent_answer']`` 会解析为 '' 而不抛错，
    rubric 仍会运行（verifier 优雅处理缺失的 state）。
    """
    logger.debug('stub run_agent called with keys=%s', sorted(kwargs.keys()))
    return SimpleNamespace(main_trajectory=[])


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('case_id', ALL_CASE_IDS)
async def test_case_boots_through_mcp_load(case_id: str, per_case_ctx) -> None:
    """每个 case 都能从 ``resolve_case`` 走到 MCP 加载而不出错。

    我们依赖 ``run_aidata`` 驱动流水线，只对
    ``AidataTaskOutput`` 表面进行断言：

    * ``error_message`` 必须为 ``None``——步骤 0-6 的任何失败都会
      把它设为一个稳定字符串，如 ``'service setup failed: ...'`` /
      ``'bootstrap failed: ...'`` / ``'Failed to initialize MCP session:
      ...'``，我们据此检测回归。
    * Rubric 允许失败（无 agent 轨迹 → 大部分检查
      不会通过），但那是 **评分** 问题，不是崩溃。 我们不断言
      分数。
    """
    task_input = AidataTaskInput(
        case_id=case_id,
        run_agent=_stub_run_agent,
    )
    # 在获取 gongfeng runtime 之前先窥视清单——声明了
    # ``pending_upstream`` 的 case 结构上无法通过，
    # 除非其外部依赖发布（未发布的 MockServer
    # tag、组合 docker 镜像……）。 用清晰的原因跳过，
    # 这样动态套件不会在这些 case 上浪费沙箱配额。
    resolve_case(task_input)
    if task_input.pending_upstream:
        pytest.skip(f'{case_id}: pending upstream — {task_input.pending_upstream}')

    output = await run_aidata(task_input)

    assert output.error_message is None, (
        f'{case_id}: run_aidata reported error_message={output.error_message!r}. '
        f'This means one of the pre-agent steps (service setup, bootstrap, '
        f'MCP load) failed. Full output: {output.model_dump()}'
    )
