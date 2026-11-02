"""
Rubric-based Judge 模块。

对齐 baseline: General_badcase_RL/code/PE/rubric_judge.py

基于 labeling 阶段生成的场景化 rubrics，对模型回复逐条判定，
输出 per-rubric 判定结果 + 加权归一化 reward ∈ [0, 1]。

LLM 调用使用 nexus ExecutionContext（参考 closed_qa_judge.py），
含 rate limiter 并发控制 + reward_rate_limit_wait_time 统计。
"""

import asyncio
import difflib
import json
import logging
import os
import random
import re

from typing import Any

from nexus.utils.rate_limiter import get_rate_limiter
from nexus.utils.timeout import async_timeout

from .judge_utils import extract_query_solution, retry_backoff_sleep
from .rubric_formatter import format_rubrics


logger = logging.getLogger(__name__)

MAX_RETRIES = int(os.environ.get('RUBRIC_JUDGE_MAX_RETRIES', 10))
RATE_LIMIT = int(os.environ.get('RUBRIC_JUDGE_RATE_LIMIT', 6))
SINGLE_ATTEMPT_TIMEOUT = int(os.environ.get('RUBRIC_JUDGE_TIMEOUT_SINGLE', 200))

# 1% 的数据打印详细 debug 信息（judge 输入、输出、reward）
_DEBUG_SAMPLE_RATE = 0.01


# ============================================================
# Judge Prompt 模板
# ============================================================

JUDGE_SYSTEM_PROMPT = """\
### 角色：基于评估标准的精确打分专家

你是一位严格遵循评估标准（Rubrics）进行逐项判定的 AI 评分专家。你的任务是：
1. 仔细阅读给定的对话上下文和评估标准
2. 对待评估的模型回复，逐条 rubric 判定是否满足
3. 输出结构化的判定结果

### 打分规则

对每条评估标准，你需要给出以下判定之一：
- **满足**：回复完全符合该标准的满足条件
- **部分满足**：回复在该标准上有所体现但不完整，或满足了部分条件
- **不满足**：回复明显违反了该标准的不满足条件，或完全未涉及

### 判定原则
1. **严格对标**：判定必须基于 rubric 中给出的"满足条件"和"不满足条件"，不要引入额外标准
2. **独立判定**：每条 rubric 独立评估，不因其他维度的表现影响当前维度
3. **证据驱动**：reason 中必须引用回复中的具体内容作为判定依据
4. **陷阱标准特别注意**：权重为 -1 的陷阱标准，"满足"表示成功避开了陷阱，"不满足"表示踩中了陷阱

### 输出格式

请严格输出以下 JSON 格式，不要有注释或额外文字：

```json
{
  "rubric_scores": [
    {
      "标题": "与评估标准中的标题完全一致",
      "judgment": "满足 / 部分满足 / 不满足",
      "confidence": "高 / 中 / 低",
      "reason": "基于回复内容的具体判定理由"
    }
  ],
  "overall_comment": "综合评价，总结主要优点和不足",
  "overall_quality": "高 / 中 / 低"
}
```

注意：rubric_scores 数组的长度和顺序必须与给定的评估标准一一对应。
"""


JUDGE_USER_PROMPT_TEMPLATE = """\
## 对话历史
{conversation_history}

## 当前轮次用户提问
{current_prompt}

## 待评估的模型回复
{current_answer}

## 参考答案（满分回复）
{perfect_response}

---

## 评估标准（Rubrics）

{rubrics_text}

---

请严格按照以上评估标准，对"待评估的模型回复"进行逐条判定，输出 JSON 结果。
"""


# ============================================================
# Prompt 构建
# ============================================================


def build_judge_prompt(
    conversation_history: str,
    current_prompt: str,
    current_answer: str,
    rubrics_text: str,
    perfect_response: str = '（未提供参考答案）',
) -> dict[str, str]:
    """构建 judge prompt。"""
    user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        conversation_history=conversation_history or '（这是第一轮对话，无历史记录）',
        current_prompt=current_prompt,
        current_answer=current_answer,
        rubrics_text=rubrics_text,
        perfect_response=perfect_response or '（未提供参考答案）',
    )
    return {
        'system_prompt': JUDGE_SYSTEM_PROMPT,
        'user_prompt': user_prompt,
    }


# ============================================================
# Judge 输出解析
# ============================================================


def parse_judge_response(raw_response: str) -> dict[str, Any]:
    """从 judge LLM 输出中提取 JSON。

    支持纯 JSON 和 ```json...``` 代码块两种格式。

    Returns:
        解析后的 dict；失败返回空 dict
    """
    if not raw_response or not raw_response.strip():
        return {}

    text = raw_response.strip()

    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


# ============================================================
# Reward 计算
# ============================================================

# judgment → 得分系数
_JUDGMENT_COEFF: dict[str, float] = {
    '满足': 1.0,
    '部分满足': 0.5,
    '不满足': 0.0,
}

# 默认权重映射：labeling PE 名义权重 → 实际计算权重
# 拉大核心标准与重要标准的区分度（原 3:1 → 5:1）
DEFAULT_WEIGHT_MAP: dict[int, int] = {3: 5, 2: 2, 1: 1}

# 模糊匹配阈值
_FUZZY_MATCH_THRESHOLD = 0.8


def _align_rubric_scores(
    rubrics: list[dict],
    rubric_scores: list[dict],
) -> list[dict | None]:
    """将 judge 输出的 rubric_scores 与原始 rubrics 对齐。

    策略：
      1. 长度一致 → 直接按下标 zip 对齐
      2. 长度不一致 → 降级为标题匹配（精确 → 模糊），未匹配返回 None
    """
    if len(rubric_scores) == len(rubrics):
        return list(rubric_scores)

    # 长度不一致，降级为标题匹配
    logger.warning(f'rubric_scores 长度({len(rubric_scores)}) != rubrics 长度({len(rubrics)})，降级为标题匹配')

    score_by_title: dict[str, dict] = {}
    for item in rubric_scores:
        title = item.get('标题', '')
        if title:
            score_by_title[title] = item

    aligned: list[dict | None] = []
    score_titles = list(score_by_title.keys())

    for rubric in rubrics:
        title = rubric.get('标题', '')

        # 精确匹配
        if title in score_by_title:
            aligned.append(score_by_title[title])
            continue

        # 模糊匹配
        best_match = None
        best_ratio = 0.0
        for st in score_titles:
            ratio = difflib.SequenceMatcher(None, title, st).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = st

        if best_match and best_ratio >= _FUZZY_MATCH_THRESHOLD:
            logger.debug(f"模糊匹配: '{title}' ↔ '{best_match}' (相似度={best_ratio:.2f})")
            aligned.append(score_by_title[best_match])
        else:
            logger.warning(f"rubric '{title}' 未匹配到 judge 输出，将跳过计分")
            aligned.append(None)

    return aligned


def compute_reward(
    rubrics: list[dict],
    judge_result: dict,
    *,
    weight_map: dict[int, int] | None = None,
) -> dict[str, Any]:
    """根据 rubrics 权重和 judge 判定结果，计算归一化 reward。

    计算公式：
      positive_normalized = raw_score / max_score          [0, 1]
      trap_ratio = trap_penalty / num_traps                [-1, 0]
      reward = clamp(positive_normalized + trap_ratio, 0, 1)
    """
    if weight_map is None:
        weight_map = DEFAULT_WEIGHT_MAP

    rubric_scores = judge_result.get('rubric_scores', [])
    aligned_scores = _align_rubric_scores(rubrics, rubric_scores)

    raw_score = 0.0
    max_score = 0
    trap_penalty = 0.0
    num_traps = 0
    details = []

    for rubric, matched_score in zip(rubrics, aligned_scores, strict=False):
        title = rubric.get('标题', '')
        nominal_weight = rubric.get('权重', 1)

        # 未匹配：跳过计分
        if matched_score is None:
            details.append({
                '标题': title,
                '权重': nominal_weight,
                'judgment': '未匹配',
                'reason': 'judge 输出中未找到对应 rubric，跳过计分',
                'score_contribution': 0.0,
            })
            continue

        judgment = matched_score.get('judgment', '不满足')
        reason = matched_score.get('reason', '')
        coeff = _JUDGMENT_COEFF.get(judgment, 0.0)

        if nominal_weight > 0:
            # 正向标准：应用权重映射
            effective_weight = weight_map.get(nominal_weight, nominal_weight)
            score = effective_weight * coeff
            raw_score += score
            max_score += effective_weight
            details.append({
                '标题': title,
                '权重': nominal_weight,
                'effective_weight': effective_weight,
                'judgment': judgment,
                'reason': reason,
                'score_contribution': round(score, 2),
            })
        else:
            # 陷阱标准 (weight = -1)
            penalty = nominal_weight * (1.0 - coeff)
            trap_penalty += penalty
            num_traps += 1
            details.append({
                '标题': title,
                '权重': nominal_weight,
                'judgment': judgment,
                'reason': reason,
                'score_contribution': round(penalty, 2),
            })

    # 归一化
    positive_normalized = raw_score / max_score if max_score > 0 else 0.0
    trap_ratio = trap_penalty / num_traps if num_traps > 0 else 0.0

    reward = max(0.0, min(1.0, positive_normalized + trap_ratio))

    return {
        'rubric_score': round(reward, 4),
        'rubric_raw_score': round(raw_score, 2),
        'rubric_max_score': max_score,
        'rubric_trap_penalty': round(trap_penalty, 2),
        'rubric_details': json.dumps(details, ensure_ascii=False),
        'rubric_overall_comment': judge_result.get('overall_comment', ''),
        'rubric_overall_quality': judge_result.get('overall_quality', ''),
    }


# ============================================================
# LLM 调用（nexus ExecutionContext 版本，含 rate limiter）
# ============================================================


async def llm_as_a_judge(
    system_prompt: str,
    user_prompt: str,
    traj_id: str = 'unknown',
) -> tuple[dict[str, Any], str]:
    """调用 LLM 进行 rubric 判定。

    使用 nexus ExecutionContext 获取 OpenAI client（参考 closed_qa_judge.py）。
    包含 rate limiter 并发控制和 reward_rate_limit_wait_time 统计。

    Args:
        system_prompt: judge system prompt
        user_prompt: judge user prompt
        traj_id: 追踪 ID

    Returns:
        (parsed_result, raw_response_text) 元组；失败 raise RuntimeError
    """
    from nexus.core import require_context

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
    ]

    rate_limiter = get_rate_limiter('rubric_judge', max_concurrent=RATE_LIMIT)
    rate_limit_wait_time = 0.0
    last_exception: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            logger.debug(f'Rubric judge attempt {attempt + 1}/{MAX_RETRIES}')

            rl_start = asyncio.get_event_loop().time()
            async with rate_limiter:
                rl_wait = asyncio.get_event_loop().time() - rl_start
                rate_limit_wait_time += rl_wait
                if rl_wait > 10:
                    logger.warning(f'[{traj_id}] RATE_LIMIT_WAIT {rl_wait:.2f}s')
                ctx = require_context()
                client = ctx.get_openai_client('judge')

                async with async_timeout(SINGLE_ATTEMPT_TIMEOUT):
                    response = await client.create_completion(
                        messages=messages,
                        max_tokens=10000,
                        temperature=0.3,
                        max_retries=1,
                        enable_thinking=True,
                    )

            result_text = response.choices[0].message.content or ''

            parsed = parse_judge_response(result_text)
            if parsed and 'rubric_scores' in parsed:
                return parsed, result_text

            logger.warning(f'Rubric judge attempt {attempt + 1}/{MAX_RETRIES}: JSON 解析成功但缺少 rubric_scores')
            last_exception = RuntimeError('JSON parsed but missing rubric_scores')
            await retry_backoff_sleep(attempt)

        except Exception as e:
            last_exception = e
            logger.warning(f'Rubric judge attempt {attempt + 1}/{MAX_RETRIES} 失败: {type(e).__name__}: {e}')
            await retry_backoff_sleep(attempt)

    logger.error(f'Rubric judge 在 {MAX_RETRIES} 次重试后失败')
    raise RuntimeError(f'Rubric judge failed after {MAX_RETRIES} retries: {last_exception}')


# ============================================================
# 标准接口：compute_score
# ============================================================


async def compute_score(
    solution_str: str,
    ground_truth: str = '',
    extra_info: dict | None = None,
) -> dict[str, Any]:
    """Rubric-based 评分函数。

    适配 nexus reward 标准接口，内部整合 prompt 构建 → LLM 调用 → reward 计算。

    Args:
        solution_str: 模型生成的完整回复
        ground_truth: 满分参考回复（perfect_response）
        extra_info: 包含 rubrics、rubrics_text、conversation_history、current_prompt

    Returns:
        包含 rubric_score 等字段的字典
    """
    extra_info = extra_info or {}
    traj_id = extra_info.get('traj_id', 'unknown')

    rate_limit_wait_time = 0.0

    # 提取最终答案
    current_answer = extract_query_solution(solution_str, extra_info)

    # 提取 rubrics
    rubrics_raw = extra_info.get('rubrics', '[]')
    if isinstance(rubrics_raw, str):
        try:
            rubrics = json.loads(rubrics_raw)
        except json.JSONDecodeError:
            logger.error(f'[{traj_id}] rubrics JSON 解析失败: {rubrics_raw[:200]}')
            rubrics = []
    else:
        rubrics = rubrics_raw

    if not rubrics:
        logger.warning(f'[{traj_id}] rubrics 为空，返回默认分数')
        return {
            'rubric_score': 0.0,
            'rubric_raw_score': 0.0,
            'rubric_max_score': 0,
            'rubric_trap_penalty': 0.0,
            'rubric_details': '[]',
            'rubric_overall_comment': '',
            'rubric_overall_quality': '',
            'reward_rate_limit_wait_time': 0.0,
        }

    # 优先使用已格式化的 rubrics_text，避免重复格式化
    rubrics_text = extra_info.get('rubrics_text', '')
    if not rubrics_text:
        rubrics_text = format_rubrics(rubrics)

    # 其他字段
    conversation_history = extra_info.get('conversation_history', '')
    current_prompt = extra_info.get('current_prompt', extra_info.get('question', ''))
    perfect_response = ground_truth or ''

    # 构建 prompt
    prompt = build_judge_prompt(
        conversation_history=conversation_history,
        current_prompt=current_prompt,
        current_answer=current_answer,
        rubrics_text=rubrics_text,
        perfect_response=perfect_response,
    )

    # 调用 LLM judge（含 rate limiter）
    rl_start = asyncio.get_event_loop().time()
    judge_result, raw_judge_response = await llm_as_a_judge(
        system_prompt=prompt['system_prompt'],
        user_prompt=prompt['user_prompt'],
        traj_id=traj_id,
    )
    rate_limit_wait_time += asyncio.get_event_loop().time() - rl_start

    # 计算 reward
    reward_result = compute_reward(rubrics, judge_result)

    # 1% 采样打印详细 debug 信息
    should_debug = random.random() < _DEBUG_SAMPLE_RATE
    if should_debug:
        logger.warning(
            f'[rubric_judge][DEBUG_SAMPLE] traj_id={traj_id}\n'
            f'===== JUDGE INPUT (user_prompt) =====\n'
            f'{prompt["user_prompt"][:2000]}\n'
            f'===== JUDGE RAW OUTPUT =====\n'
            f'{raw_judge_response[:2000]}\n'
            f'===== REWARD RESULT =====\n'
            f'rubric_score={reward_result["rubric_score"]}, '
            f'raw={reward_result["rubric_raw_score"]}/{reward_result["rubric_max_score"]}, '
            f'trap_penalty={reward_result["rubric_trap_penalty"]}\n'
            f'details={reward_result["rubric_details"][:1000]}\n'
            f'===== END DEBUG_SAMPLE ====='
        )

    logger.warning(
        f'[{traj_id}] RUBRIC_JUDGE_REWARD '
        f'rubric_score={reward_result["rubric_score"]:.4f}, '
        f'raw={reward_result["rubric_raw_score"]}/{reward_result["rubric_max_score"]}, '
        f'trap_penalty={reward_result["rubric_trap_penalty"]}'
    )

    reward_result['reward_rate_limit_wait_time'] = rate_limit_wait_time
    return reward_result
