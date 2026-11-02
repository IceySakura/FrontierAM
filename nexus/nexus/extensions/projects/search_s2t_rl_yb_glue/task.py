"""S2T Search Agentic RL task with the yuanbao_search (agts) reward.

Runs the ``search_s2t_agent_yb_glue`` multi-turn search agent (single vendored
PromptSearch tool, optional speech input), then scores the final answer with the
yuanbao ``agts_closed_qa``-based required-search reward pipeline:

    planner format check → (fail ⇒ -1.0)
    no valid search      → 0.0
    valid search         → max(0.0, closed_qa_score(LLM judge ∈ {0, .5, 1}) - query_cost)

The LLM judge runs against a separate frozen ``judge`` client (never the trained
policy), so judge tokens stay out of the training trajectory.
"""

import logging
import uuid

from functools import partial
from typing import Any, Literal

from pydantic import Field

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    task_registry,
)
from nexus.core.agent import WithMainTrajectory
from nexus.extensions.projects.search_s2t_rl_yb_glue.log_utils import log_trace


logger = logging.getLogger(__name__)
_log_task = partial(log_trace, 'TASK')


class SearchS2TYBTaskInput(TaskRunInput, WithAgentRunner):
    """Input for the S2T yuanbao-glue search task."""

    messages: list[dict[str, Any]] | None = Field(
        default=None, description='多轮对话（policy 可见），可含 <audio> 占位'
    )
    prompt: list[dict[str, Any]] | None = Field(default=None, description='messages 的别名（兼容 agts 数据）')
    audios: list[dict[str, Any]] | None = Field(default=None, description='按 <audio> 出现顺序排列的音频 patch 列表')
    user_prompt: str | None = Field(default=None, description='单轮 fallback 文本')

    ground_truth: str = Field(default='', description='参考答案')
    reward_style: str = Field(default='audio-grm', description='奖励计算方式（Reward_system AudioGRMDS）')
    data_source: str = Field(default='agts_closed_qa', description='数据来源标识（reward 分流）')
    extra_info: dict = Field(default_factory=dict, description='额外信息（question / history / 标签 等）')
    reward_system_user_token: str | None = Field(default=None, description='Reward_system user token')
    reward_system_exp_id: str | None = Field(default=None, description='Reward_system experiment id')
    reward_system_router_id: str = Field(default='AudioGRMDS-SimpleFusion-0', description='Reward_system router id')
    reward_system_timeout: int = Field(default=1800, description='Reward_system request timeout seconds')
    reward_system_max_retries: int = Field(default=3, description='Reward_system max retry attempts')

    max_iterations: int = Field(default=2, description='agent 最大迭代次数')
    max_search_calls: int = Field(default=1, description='最大搜索调用次数（硬上限）')
    max_subqueries_per_call: int = Field(default=3, description='单次搜索允许的最大子查询数（硬上限）')
    max_total_observation_tokens: int = Field(default=10000, description='整条轨迹 tool observation 的 token 总预算')
    system_prompt: str | None = Field(default=None, description='可选 system prompt 覆盖')
    format_type: str = Field(default='ds', description='模型格式类型（保留字段）')
    time_range: str = Field(
        default='', description='per-row 截止日期过滤（parquet time_range 列），传给搜索后端屏蔽未来 doc'
    )
    traj_id: str = Field(default='', description='轨迹唯一标识，由 HyTuner 传入')
    run_oracle: bool = Field(default=False, description='是否直接用 ground_truth 作为回答（调试用）')
    reward_mode: Literal['compute', 'zero'] = Field(
        default='compute',
        description='奖励模式；zero 仅在训练时跳过奖励计算并直接返回 0',
    )
    is_validation: bool = Field(default=False, description='是否为验证阶段，由 HyTuner runner 传入')


class SearchS2TYBTaskOutput(TaskRunOutput, WithScore):
    """Output for the S2T yuanbao-glue search task."""

    score: float | None = None
    response: str = Field(default='', description='模型最终答案')
    final_answer: str = Field(default='', description='提取的最终答案')
    judgment: str = Field(default='', description='LLM judge 判定')
    tool_call_count: int = Field(default=0, description='工具调用次数')
    search_count: int = Field(default=0, description='搜索调用次数')
    valid_search_count: int = Field(default=0, description='至少包含一个非空 query 的有效搜索次数')
    subquery_count: int = Field(default=0, description='硬截断前模型尝试生成的子查询数')
    executed_subquery_count: int = Field(default=0, description='硬截断后实际提交的子查询数')
    rejected_tool_calls: int = Field(default=0, description='被执行层硬限制拒绝的工具调用数')
    overlimit_search_calls: int = Field(default=0, description='超过搜索硬上限而被拒绝的搜索调用数')
    tool_observation_tokens: int = Field(default=0, description='轨迹累计 tool observation token 数')
    extra_fields: dict = Field(default_factory=dict, description='传递给 HyTuner 的额外字段')


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, 'tolist'):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _resolve_question(task_input: SearchS2TYBTaskInput, messages: list[dict[str, Any]]) -> str:
    """Prefer the transcription/question in extra_info; fall back to last user text."""
    extra_info = task_input.extra_info or {}
    question = extra_info.get('question') or extra_info.get('query')
    if question:
        return str(question)
    for msg in reversed(messages or []):
        if msg.get('role') == 'user':
            content = msg.get('content', '')
            if isinstance(content, list):
                content = ' '.join(
                    p.get('text', '') for p in content if isinstance(p, dict) and p.get('type') == 'text'
                )
            if content:
                return str(content)
    return str(task_input.user_prompt or '')


async def _compute_reward(
    *,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
    data_source: str,
) -> dict[str, Any]:
    """Load reward dependencies only when this trajectory needs scoring."""
    from .reward.total_reward import compute_score_async

    return await compute_score_async(
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        data_source=data_source,
    )


@task_registry.register('search_s2t_task_yb_glue')
async def run_search_s2t_task_yb_glue(task_input: SearchS2TYBTaskInput) -> SearchS2TYBTaskOutput:
    """Run the S2T search agent and score it with the yuanbao agts_closed_qa reward."""
    traj_id = task_input.traj_id or uuid.uuid4().hex[:8]
    _log_task(traj_id, 'TASK_START', f'data_source={task_input.data_source}')

    messages = task_input.messages or task_input.prompt or []
    audios = _as_list(task_input.audios) or None
    # Clamp legacy parquet configs to the new project-level hard protocol.
    effective_max_iterations = max(1, min(int(task_input.max_iterations), 2))
    effective_max_search_calls = max(0, min(int(task_input.max_search_calls), 1))
    effective_max_subqueries_per_call = max(1, min(int(task_input.max_subqueries_per_call), 3))
    effective_max_total_observation_tokens = max(
        64,
        min(int(task_input.max_total_observation_tokens), 10000),
    )

    if task_input.run_oracle:
        final_answer = str(task_input.ground_truth)
        tool_call_count = 0
        search_count = 0
        valid_search_count = 0
        web_len = 0
        subquery_count = 0
        executed_subquery_count = 0
        overlimit_subquery_count = 0
        malformed_tool_calls = 0
        rejected_tool_calls = 0
        overlimit_search_calls = 0
        tool_observation_tokens = 0
        truncated_tool_observations = 0
    else:
        _log_task(traj_id, 'AGENT_RUN_START')
        agent_user_prompt = task_input.user_prompt or _resolve_question(task_input, messages)
        agent_output = await task_input.run_agent(
            messages=messages,
            audios=audios,
            user_prompt=agent_user_prompt,
            max_iterations=effective_max_iterations,
            max_search_calls=effective_max_search_calls,
            max_subqueries_per_call=effective_max_subqueries_per_call,
            max_total_observation_tokens=effective_max_total_observation_tokens,
            system_prompt=task_input.system_prompt,
            time_range=task_input.time_range,
        )
        _log_task(traj_id, 'AGENT_RUN_DONE')

        if not isinstance(agent_output, WithMainTrajectory):
            raise RuntimeError('search_s2t_task_yb_glue requires agent output to be a WithMainTrajectory subclass')

        final_answer = getattr(agent_output, 'final_answer', None)
        if not final_answer:
            final_answer = next(
                (
                    msg['content']
                    for msg in reversed(agent_output.main_trajectory)
                    if msg.get('role') == 'assistant' and isinstance(msg.get('content'), str) and msg.get('content')
                ),
                '',
            )
        tool_call_count = int(getattr(agent_output, 'tool_call_count', 0) or 0)
        search_count = int(getattr(agent_output, 'search_count', 0) or 0)
        valid_search_count = int(getattr(agent_output, 'valid_search_count', 0) or 0)
        web_len = int(getattr(agent_output, 'web_len', 0) or 0)
        subquery_count = int(getattr(agent_output, 'subquery_count', 0) or 0)
        executed_subquery_count = int(getattr(agent_output, 'executed_subquery_count', 0) or 0)
        overlimit_subquery_count = int(getattr(agent_output, 'overlimit_subquery_count', 0) or 0)
        malformed_tool_calls = int(getattr(agent_output, 'malformed_tool_calls', 0) or 0)
        rejected_tool_calls = int(getattr(agent_output, 'rejected_tool_calls', 0) or 0)
        overlimit_search_calls = int(getattr(agent_output, 'overlimit_search_calls', 0) or 0)
        tool_observation_tokens = int(getattr(agent_output, 'tool_observation_tokens', 0) or 0)
        truncated_tool_observations = int(getattr(agent_output, 'truncated_tool_observations', 0) or 0)

    final_answer = final_answer or ''

    # Assemble the extra_info the yuanbao reward needs. The judge is text-based:
    # it compares the model answer to ground_truth given the question transcription
    # + history, so audio input turns don't change the reward path.
    reward_extra_info = dict(task_input.extra_info or {})
    reward_extra_info['question'] = _resolve_question(task_input, messages)
    reward_extra_info.setdefault('history', _as_list(reward_extra_info.get('history')))
    reward_extra_info['tool_call_count'] = tool_call_count
    reward_extra_info['search_count'] = search_count
    reward_extra_info['valid_search_count'] = valid_search_count
    reward_extra_info['web_len'] = web_len
    reward_extra_info['subquery_count'] = subquery_count
    reward_extra_info['executed_subquery_count'] = executed_subquery_count
    reward_extra_info['overlimit_subquery_count'] = overlimit_subquery_count
    reward_extra_info['malformed_tool_calls'] = malformed_tool_calls
    reward_extra_info['rejected_tool_calls'] = rejected_tool_calls
    reward_extra_info['overlimit_search_calls'] = overlimit_search_calls
    reward_extra_info['tool_observation_tokens'] = tool_observation_tokens
    reward_extra_info['truncated_tool_observations'] = truncated_tool_observations
    reward_extra_info['max_tool_calls'] = effective_max_search_calls
    reward_extra_info['max_subqueries_per_call'] = effective_max_subqueries_per_call
    reward_extra_info['max_total_observation_tokens'] = effective_max_total_observation_tokens
    reward_extra_info['traj_id'] = traj_id
    reward_extra_info.setdefault('model_type', 'ds')
    reward_extra_info['reward_system_router_id'] = task_input.reward_system_router_id
    reward_extra_info['reward_system_user_token'] = task_input.reward_system_user_token
    reward_extra_info['reward_system_exp_id'] = task_input.reward_system_exp_id
    reward_extra_info['timeout'] = task_input.reward_system_timeout
    reward_extra_info['max_retry_attempts'] = task_input.reward_system_max_retries

    skip_reward = task_input.reward_mode == 'zero' and not task_input.is_validation
    # Reward backends return heterogeneous payloads (score / judgment / gate status ...),
    # so the dict stays Any-valued; only score and judgment are consumed here.
    reward_details: dict[str, Any]
    if skip_reward:
        reward_details = {
            'score': 0.0,
            'judgment': '',
            'reward_mode': task_input.reward_mode,
            'reward_skipped': True,
        }
        _log_task(traj_id, 'REWARD_CALC_SKIPPED', f'reward_mode={task_input.reward_mode}')
    else:
        _log_task(traj_id, 'REWARD_CALC_START')
        reward_details = await _compute_reward(
            solution_str=final_answer,
            ground_truth=task_input.ground_truth,
            extra_info=reward_extra_info,
            data_source=task_input.data_source,
        )
        reward_details.setdefault('reward_mode', task_input.reward_mode)
        reward_details.setdefault('reward_skipped', False)
    score = reward_details.get('score', 0.0)
    judgment = reward_details.get('judgment', '')
    _log_task(traj_id, 'REWARD_CALC_DONE & TASK_DONE', f'score={score}, judgment={judgment}')

    return SearchS2TYBTaskOutput(
        score=score,
        response=final_answer,
        final_answer=final_answer,
        judgment=judgment,
        tool_call_count=tool_call_count,
        search_count=search_count,
        valid_search_count=valid_search_count,
        subquery_count=subquery_count,
        executed_subquery_count=executed_subquery_count,
        rejected_tool_calls=rejected_tool_calls,
        overlimit_search_calls=overlimit_search_calls,
        tool_observation_tokens=tool_observation_tokens,
        extra_fields={
            'reward_extra_info': reward_details,
            'tool_call_counts': tool_call_count,
            'search_count': search_count,
            'valid_search_count': valid_search_count,
            'subquery_count': subquery_count,
            'executed_subquery_count': executed_subquery_count,
            'overlimit_subquery_count': overlimit_subquery_count,
            'rejected_tool_calls': rejected_tool_calls,
            'overlimit_search_calls': overlimit_search_calls,
            'tool_observation_tokens': tool_observation_tokens,
            'truncated_tool_observations': truncated_tool_observations,
        },
    )
