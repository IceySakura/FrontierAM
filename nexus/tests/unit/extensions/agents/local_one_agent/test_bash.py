"""Tests for local_one_agent bash tool.

bash 现继承 harness v3 ``bash_tool_v2``（输出过大存盘逻辑在 v3），local 层只加
sandbox 包装 + 放宽 description。 因此本测试聚焦 local 增量行为，并通过 patch
v3 模块里的 ``run_command`` 验证委托链路。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from nexus.core.tool.types import ToolContext
from nexus.extensions.agents.local_one_agent.tools.bash import LocalBashInput, bash_tool
from nexus.runtime.executors.command import CommandRunStatus, RunCommandResponse


_PATCH_RUN_CMD = 'nexus.extensions.agents.yb_one_agent.tools.run_command'


@pytest.fixture
def tool_context():
    return ToolContext(run_id='test-run', tool_configs={}, tool_state={})


class TestLocalBashInput:
    def test_has_description_field(self):
        inp = LocalBashInput(command='ls', description='List files')
        assert inp.description == 'List files'

    def test_description_optional_defaults_empty(self):
        inp = LocalBashInput(command='ls')
        assert inp.description == ''

    def test_inherits_timeout(self):
        inp = LocalBashInput(command='ls', timeout=5000)
        assert inp.timeout == 5000


class TestBashDelegation:
    async def test_normal_output(self, tool_context):
        mock_resp = RunCommandResponse(
            status=CommandRunStatus.Finished, return_code=0, stdout='hello world', stderr='', execution_time=0.1
        )
        with patch(_PATCH_RUN_CMD, new_callable=AsyncMock, return_value=mock_resp):
            result = await bash_tool(LocalBashInput(command='echo hello'), tool_context)
        assert 'hello world' in result.output[0]['text']
        assert result.meta['exit_code'] == 0

    async def test_timeout_response(self, tool_context):
        mock_resp = RunCommandResponse(
            status=CommandRunStatus.TimeLimitExceeded, stdout='partial output', stderr='', execution_time=60.0
        )
        with patch(_PATCH_RUN_CMD, new_callable=AsyncMock, return_value=mock_resp):
            result = await bash_tool(LocalBashInput(command='sleep 999'), tool_context)
        assert 'timed out' in result.output[0]['text'].lower()
        assert result.meta.get('timeout') is True


class TestBashLargeOutputViaV3:
    async def test_large_output_saved_to_file(self, tmp_path):
        # 本地 executor 注入 working_dir=workspace.path；落盘应落到该目录下 .tool_output
        ctx = ToolContext(run_id='r', tool_configs={}, tool_state={'working_dir': str(tmp_path)})
        large_stdout = 'line\n' * 10000
        mock_resp = RunCommandResponse(
            status=CommandRunStatus.Finished, return_code=0, stdout=large_stdout, stderr='', execution_time=1.0
        )
        with patch(_PATCH_RUN_CMD, new_callable=AsyncMock, return_value=mock_resp):
            result = await bash_tool(LocalBashInput(command='cat big'), ctx)
        assert 'Output too large' in result.output[0]['text']
        output_file = result.meta.get('output_file')
        assert output_file is not None
        # 关键：落盘进了本地工作区，而非默认 /data/workspace
        assert output_file.startswith(str(tmp_path))

    async def test_normal_output_not_saved(self, tmp_path):
        ctx = ToolContext(run_id='r', tool_configs={}, tool_state={'working_dir': str(tmp_path)})
        mock_resp = RunCommandResponse(
            status=CommandRunStatus.Finished, return_code=0, stdout='hello world', stderr='', execution_time=0.1
        )
        with patch(_PATCH_RUN_CMD, new_callable=AsyncMock, return_value=mock_resp):
            result = await bash_tool(LocalBashInput(command='echo hello'), ctx)
        assert 'Output too large' not in result.output[0]['text']


class TestSandboxWrap:
    async def test_sandbox_wrap_fn_applied(self):
        captured: list[str] = []

        async def mock_run(req):
            captured.append(req.command)
            return RunCommandResponse(
                status=CommandRunStatus.Finished, return_code=0, stdout='ok', stderr='', execution_time=0.1
            )

        def wrap_fn(cmd: str) -> str:
            return f'bwrap -- {cmd}'

        ctx = ToolContext(run_id='r', tool_configs={}, tool_state={'sandbox_wrap_fn': wrap_fn})
        with patch(_PATCH_RUN_CMD, side_effect=mock_run):
            await bash_tool(LocalBashInput(command='ls'), ctx)
        assert captured[0] == 'bwrap -- ls'

    async def test_no_wrap_when_absent(self, tool_context):
        captured: list[str] = []

        async def mock_run(req):
            captured.append(req.command)
            return RunCommandResponse(
                status=CommandRunStatus.Finished, return_code=0, stdout='ok', stderr='', execution_time=0.1
            )

        with patch(_PATCH_RUN_CMD, side_effect=mock_run):
            await bash_tool(LocalBashInput(command='ls'), tool_context)
        assert captured[0] == 'ls'
