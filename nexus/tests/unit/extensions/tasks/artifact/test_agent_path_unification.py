"""容器侧 agent 路径：与宿主侧共用一条代码路径。

`capabilities.agent.run_claude_code_on_prepared_runtime` 是**唯一**的 agent 执行
函数。它内部只在一处分叉：LLM 请求怎么送达。

* 远端 runtime — agent 在另一个进程，请求经 socket.io 到达，
  ``LLMRequestHandler`` 通过 ``runtime.on('llm_request', ...)`` 接收。
* ``LocalRuntime`` — agent 在**本进程**。``LocalRuntime.on()`` 按设计抛
  ``NotImplementedError``（没有 socket.io 通道可订阅），改由本地
  ``AnthropicProxy`` + ``LLMHandlerBridge`` 承接。

其余全部共用：清 reward file、``pre_run_setup``、trajectory 采集、读判决、日志。

**为什么要有这个文件**：容器侧曾经是一份 189 行的替代实现，在运行时覆盖掉本模块的
函数。作为第二份实现它必然漂移，实际漏过 5 件事 —— skill 没装、``extra_env`` 丢
失、``pre_run_setup`` 收下却不执行、playwright MCP 从未注册、trajectory 从未采集。
每一件都让 judge 在少了它索要的东西的情况下照样自信打分。

所以这里钉住的不是"两份实现等价"，而是**只有一份实现**，且分叉只发生在该分叉的
那一处。
"""

from __future__ import annotations

from typing import Any

import pytest

from nexus.extensions.tasks.artifact.agent_setup.claude_config import default_judge_claude_input
from nexus.extensions.tasks.artifact.judges.capabilities import agent as agent_capability


class _LocalLikeRuntime:
    """站位 ``LocalRuntime``：``on()`` 抛错，其余 executor 可用。"""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def on(self, *_args, **_kwargs):
        raise NotImplementedError('LocalRuntime does not support event subscription')

    async def call_executor(self, _name, args):
        self.commands.append(str(args.get('command', '')))
        return {}


@pytest.fixture
def bridge_calls(monkeypatch: pytest.MonkeyPatch):
    """把真实 CLI 打桩，记录 bridge 路径收到的参数。"""
    calls: list[tuple[str, Any]] = []

    async def fake_run_claude_code_with_bridge(**kwargs):
        calls.append(('agent', kwargs))
        return {'success': True, 'output': '<score>3</score>', 'stderr': '', 'exit_code': 0, 'execution_time': 1.0}

    class FakeBridge:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

    monkeypatch.setattr(
        'nexus.extensions.tasks.artifact.judges.capabilities.agent_direct_gateway.run_claude_code_with_bridge',
        fake_run_claude_code_with_bridge,
    )
    monkeypatch.setattr(
        'nexus.extensions.tasks.artifact.judges.capabilities.agent_direct_gateway.LLMHandlerBridge',
        FakeBridge,
    )
    return calls


async def _run(runtime, *, claude_input=None, **kwargs):
    return await agent_capability.run_claude_code_on_prepared_runtime(
        runtime,
        'rt',
        task_prompt=kwargs.pop('task_prompt', 'q'),
        openai_client=object(),
        claude_input=claude_input or default_judge_claude_input(with_playwright=False),
        **kwargs,
    )


# ────────────────────── 分叉只在该分叉的那一处 ──────────────────────


def test_local_runtime_takes_the_bridge_path():
    """``LocalRuntime`` 不支持事件 → 走 bridge。"""
    from nexus.runtime.client.local import LocalRuntime

    assert agent_capability.runtime_supports_events(LocalRuntime()) is False


def test_remote_runtime_takes_the_socketio_path():
    class FakeRemote:
        def on(self, *_a, **_k): ...

    assert agent_capability.runtime_supports_events(FakeRemote()) is True


@pytest.mark.asyncio
async def test_host_path_registers_a_handler_and_invokes_remotely(monkeypatch: pytest.MonkeyPatch):
    """宿主侧必须保持原样：注册 handler → ws 重连 → ``runtime.invoke``。

    合并两条路径最大的风险是把本来正常的宿主侧改坏，所以这条逐项钉住。
    """
    seen: list[str] = []

    class FakeRemote:
        def on(self, *_a, **_k):
            seen.append('on')

        def off(self, *_a, **_k):
            seen.append('off')

        async def ws_disconnect(self):
            seen.append('ws_disconnect')

        async def call_executor(self, _name, args):
            if 'rm -f' in str(args):
                seen.append('rm')
            return {}

        async def invoke(self, _fn, **kwargs):
            seen.append('invoke')
            # 全量转发：新增 ClaudeCodeInput 字段无需改动本模块。
            assert 'working_dir' in kwargs and 'pre_start_scripts' in kwargs
            return {'success': True, 'output': 'ok', 'stderr': '', 'execution_time': 1.0}

    async def bridge_must_not_run(**_kwargs):
        raise AssertionError('宿主侧不得走 bridge')

    monkeypatch.setattr(
        'nexus.extensions.tasks.artifact.judges.capabilities.agent_direct_gateway.run_claude_code_with_bridge',
        bridge_must_not_run,
    )

    result = await _run(FakeRemote(), reward_file='/tmp/r.json')

    # 前两项是能力探测（注册一个空回调再摘掉）；之后才是真实流程。
    assert seen[:2] == ['on', 'off'], f'能力探测应先发生: {seen}'
    assert seen[2:] == ['ws_disconnect', 'on', 'rm', 'invoke', 'off'], seen
    assert result.error is None


# ────────────────────── 容器侧拿到 judge 索要的全部东西 ──────────────────────


@pytest.mark.asyncio
async def test_bridge_path_receives_skills_env_and_mcp(bridge_calls):
    """容器侧必须拿到 skill / 代理 env / MCP —— 这三样都曾静默丢失。

    * skill 打包在 ``pre_start_scripts`` 里（``extra_skill_dirs`` **不是字段**，
      读它永远是空 —— 旧实现正是栽在这里）。
    * ``extra_env`` 是公司代理 + ``CLAUDE_CODE_*`` 开关，丢了容器内所有外发失败。
    * ``mcp_servers`` 是浏览器工具；agent 型 judge 的 prompt 让它"浏览页面"，
      没有 MCP 就是让它在没有浏览器的情况下汇报观察结果。
    """
    claude_input = default_judge_claude_input(with_playwright=True, with_skill=True)
    await _run(_LocalLikeRuntime(), claude_input=claude_input, reward_file='/tmp/r.json')

    kwargs = next(kw for name, kw in bridge_calls if name == 'agent')
    assert kwargs['pre_start_scripts'], 'skill 安装脚本没传到'
    assert 'http_proxy' in (kwargs['extra_env'] or {}), '代理 env 丢了'
    assert [s.get('name') for s in kwargs['mcp_servers']] == ['playwright'], 'MCP 丢了'


@pytest.mark.asyncio
async def test_bridge_path_forwards_launch_knobs(bridge_calls):
    """版本 / registry / thinking preset 不能回落到 bridge 的默认值。

    judge 显式钉了 ``version``（单一真相源，与 infer 一致），回落到 'latest'
    会让判分用一个与实验记录不符的 CLI 版本。
    """
    claude_input = default_judge_claude_input(with_playwright=False)
    await _run(_LocalLikeRuntime(), claude_input=claude_input, reward_file='/tmp/r.json')

    kwargs = next(kw for name, kw in bridge_calls if name == 'agent')
    assert kwargs['version'] == claude_input.version
    assert kwargs['npm_registry'] == claude_input.npm_registry
    assert kwargs['thinking_preset'] == claude_input.thinking_preset
    assert kwargs['timeout'] == claude_input.timeout


# ────────────────────── 共用语义在容器侧同样生效 ──────────────────────


@pytest.mark.asyncio
async def test_pre_run_setup_runs_before_the_agent(bridge_calls):
    """回调必须执行，且排在 agent 之前。

    ``code_quality`` 用它把产物源码上传到 prompt 指名的目录。收下参数却不调用
    （旧实现的行为）意味着 agent 去评一个空目录、或上一次运行的残留，然后对着
    不存在的东西给出一个自信的分数。
    """
    runtime = _LocalLikeRuntime()
    order: list[str] = []

    async def pre_run_setup(handed):
        assert handed is runtime, '回调要拿到调用方持有的 runtime'
        order.append('setup')
        bridge_calls.append(('setup', handed))

    await _run(runtime, reward_file='/tmp/r.json', pre_run_setup=pre_run_setup)

    assert [name for name, _ in bridge_calls] == ['setup', 'agent']


@pytest.mark.asyncio
async def test_sync_pre_run_setup_is_supported(bridge_calls):
    """同步回调也要能用。"""
    ran: list[str] = []
    await _run(
        _LocalLikeRuntime(),
        reward_file='/tmp/r.json',
        pre_run_setup=lambda _rt: ran.append('called'),
    )

    assert ran == ['called']
    assert any(name == 'agent' for name, _ in bridge_calls)


@pytest.mark.asyncio
async def test_failed_preparation_blocks_the_agent(bridge_calls):
    """准备失败 → 不启动 agent，原因收敛到 ``agent_error``。

    否则 agent 在未准备好的工作区上跑完，产出一个「评了错东西」的分数，而失败
    原因已经丢了。

    异常由外层 ``run_agent_and_read_reward`` 捕获（宿主原本的语义，现在两条路径
    共用），所以断言打在那一层而不是内层函数上。
    """

    async def pre_run_setup(_rt):
        raise RuntimeError('code upload failed: connection reset')

    run = await agent_capability.run_agent_and_read_reward(
        image='',
        task_prompt='q',
        openai_client=object(),
        claude_input=default_judge_claude_input(with_playwright=False),
        reward_file='/tmp/r.json',
        pre_run_setup=pre_run_setup,
        prepared_runtime=_LocalLikeRuntime(),
    )

    assert not any(name == 'agent' for name, _ in bridge_calls), '准备失败后不该启动 agent'
    assert 'connection reset' in (run.agent_error or ''), '要保留根因'


@pytest.mark.asyncio
async def test_agent_cwd_matches_the_judge_declared_working_dir(bridge_calls):
    """agent 的 cwd 必须用 judge 声明的目录 —— 那是 prompt 指名、也是
    ``pre_run_setup`` 上传进去的目录。"""
    claude_input = default_judge_claude_input(with_playwright=False, working_dir='/app/candidate')
    await _run(_LocalLikeRuntime(), claude_input=claude_input, reward_file='/tmp/r.json')

    kwargs = next(kw for name, kw in bridge_calls if name == 'agent')
    assert kwargs['working_dir'] == '/app/candidate'


@pytest.mark.asyncio
async def test_reward_file_is_cleared_on_both_paths(bridge_calls):
    """复用容器时，把上一轮的判决当成本轮的读出来比读不到更糟。"""
    runtime = _LocalLikeRuntime()
    await _run(runtime, reward_file='/tmp/verdict.json')

    assert any('rm -f /tmp/verdict.json' in c for c in runtime.commands), runtime.commands


@pytest.mark.asyncio
async def test_trajectory_is_captured_on_the_bridge_path(bridge_calls):
    """容器侧也要采 trajectory —— 旧实现从不采，RL 复盘看不到判分 agent 说了什么。"""
    result = await _run(_LocalLikeRuntime(), reward_file='/tmp/r.json')

    assert result.trajectories is not None
