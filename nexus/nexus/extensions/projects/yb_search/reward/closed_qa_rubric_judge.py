"""
Closed QA Rubric Judge 评分模块。

针对带 rubric 的封闭性问答，对每个 checkpoint 独立 0/1 打分，
再按主次（must_have / nice_to_have）加权汇总到 [0, 1]。

打分公式（默认 must_weight=0.7, nice_weight=0.3）：
    base_must = sum(must_hit) / num_must
    base_nice = sum(nice_hit) / num_nice  if num_nice > 0 else None

    硬性规则：若 base_must == 0（存在 must_have 但全部未命中）→ final = 0
    其他情况：
        - 仅有 must：     final = base_must
        - 仅有 nice：     final = base_nice
        - must + nice：  final = must_weight * base_must + nice_weight * base_nice

逻辑对齐 rlvr_data_online/code/step7b_rubric_verify.py。
对外暴露的字段名（closed_qa_score、judgment、reason 等）与 closed_qa_judge 保持一致，
以便 total_reward 的下游分支无差别复用。
"""

import asyncio
import json
import logging
import os
import re

from functools import partial
from typing import Any

from nexus.extensions.projects.yb_search.log_utils import log_trace
from nexus.utils.rate_limiter import get_rate_limiter
from nexus.utils.timeout import async_timeout

from .judge_utils import (
    extract_history_from_prompt,
    extract_query_from_prompt,
    extract_query_solution,
    retry_backoff_sleep,
)


logger = logging.getLogger(__name__)

MAX_RETRIES = int(os.environ.get('TAIJI_JUDGE_MAX_RETRIES', 10))
SINGLE_ATTEMPT_TIMEOUT = int(os.environ.get('TAIJI_JUDGE_TIMEOUT_SINGLE', 200))
RATE_LIMIT = int(os.environ.get('TAIJI_JUDGE_RATE_LIMIT', 6))
JUDGE_MODEL = os.environ.get('TAIJI_JUDGE_MODEL', '')

DEFAULT_MUST_WEIGHT = float(os.environ.get('TAIJI_RUBRIC_MUST_WEIGHT', 0.7))
DEFAULT_NICE_WEIGHT = float(os.environ.get('TAIJI_RUBRIC_NICE_WEIGHT', 0.3))

_log_reward = partial(log_trace, 'REWARD')

print(
    f'CLOSED_QA_RUBRIC_JUDGE initialized with: {RATE_LIMIT=}, {MAX_RETRIES=}, '
    f'model={JUDGE_MODEL}, must_weight={DEFAULT_MUST_WEIGHT}, nice_weight={DEFAULT_NICE_WEIGHT}, '
    f'single_attempt_timeout={SINGLE_ATTEMPT_TIMEOUT}'
)


# Rubric verifier prompt（对齐 rlvr_data_online/code/prompts/rlvr_rubric_verify_prompt.txt）
# 替换占位符使用 .replace()，因此 JSON 示例中的 `{{` `}}` 保持原样不会影响功能。
PROMPT_TEMPLATE = """你是一个 RLVR rubric verifier，专业、严格、按点判分。

## 任务说明
给定：
- 用户问题
- 标准答案（Ground Truth，仅作参考依据）
- 模型回答（待评估）
- **rubric checkpoints**：从 GT 拆出的若干检查点，每个 checkpoint 都是**一个原子陈述**，并被标注为 `must_have`（主需）或 `nice_to_have`（次需）

请你对**每一个 checkpoint** 独立判断模型回答是否命中：
- **score = 1**：模型回答**正确表达了该 checkpoint 的核心事实**（允许同义/语序差异，允许合理改写，允许更详细，但**事实必须一致**）
- **score = 0**：模型回答**未提及 / 答错 / 与该 checkpoint 矛盾**

## 判分原则

1. **以 checkpoint 内容为准**：判定时只看模型回答是否覆盖了**该 checkpoint 所表达的事实**。不要把 GT 里其他点的内容拿过来抢功劳。
2. **独立判分**：每个 checkpoint 单独打 0/1，互不影响。
3. **同义等价视为命中**：核心事实正确 + 表达方式不同（语序/措辞/换算/简写/同义词），仍判 1。
4. **数值规则**：
   - 必须精确：版本号、ID、年月日、计数、枚举名 → 精确匹配（允许等价表达，例如 "2010 年" vs "2010"）
   - 允许误差：价格、性能指标、百分比 → 相对误差 < 1% 视为命中
5. **数量规则**：模型回答只要"包含 / 覆盖"该原子事实即可，**不要求模型严格按 GT 字面措辞**。
6. **冲突处理**：
   - 模型给出了与 checkpoint **明确矛盾**的事实 → 该点判 0
   - 模型只是**没提到**该 checkpoint → 也判 0
   - 模型只是补充了 checkpoint 之外的信息 → 不影响该 checkpoint 的判定
7. **verify_hint**：如果该 checkpoint 提供了 `verify_hint`，请把它作为判分时的辅助说明（消歧、同义、数值范围等）。
8. **不要过度推理**：只看模型回答的**字面信息**是否命中 checkpoint，不要替模型补全推断。
9. **整体不评估简洁度 / 风格 / 流畅度**——这一步只看每个 checkpoint 是否被覆盖。

## 输出格式

请严格输出 JSON。`results` 字段必须**与输入 checkpoints 数量、顺序、id 完全一致**：

```json
{{
  "results": [
    {{
      "id": "p1",
      "level": "must_have",
      "score": 1,
      "reason": "模型回答中的「xxx」覆盖了该 checkpoint 的「yyy」"
    }},
    {{
      "id": "p2",
      "level": "nice_to_have",
      "score": 0,
      "reason": "模型回答未提及 zzz"
    }}
  ],
  "overall_reason": "对模型回答整体的简短点评（≤ 80 字）"
}}
```

---

现在请评估以下内容：

**对话上下文**：
{conversation}

**当前问题**：
{question}

**标准答案（Ground Truth，仅供参考）**：
```
{ground_truth}
```

**Rubric Checkpoints**：
```json
{rubric_json}
```

**模型回答**：
```
{model_answer}
```

【模型回复协议说明】
模型答案中可能包含以下技术协议，如果包含，评估时只关注协议标签中的回复内容即可：
- **模型回复协议**：`<answer>回复内容...</answer>`

请输出你的判分结果：
"""


def _coerce_int01(v: Any) -> int:
    """把 LLM 返回的 score 字段安全收敛为 0/1 整数。"""
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        return 1 if v >= 0.5 else 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ('1', 'yes', 'y', 'true', '命中', 'hit', 'pass', 'ok'):
            return 1
        if s in ('0', 'no', 'n', 'false', '未命中', 'miss', 'fail'):
            return 0
        try:
            return _coerce_int01(float(s))
        except ValueError:
            return 0
    return 0


def _compute_adaptive_weights(
    n_must: int,
    n_nice: int,
    min_must_weight: float = 0.7,
    dominance_factor: float = 2.0,
) -> tuple[float, float]:
    """Compute adaptive must/nice total weights based on item counts.

    Guarantees:
      1. must_total_weight >= min_must_weight (must always dominates)
      2. per_must_weight > per_nice_weight (each must item worth more than each nice)
      3. Proportional to item counts with dominance_factor bias toward must

    The formula treats each must item as `dominance_factor` units vs 1 unit for nice:
      must_weight = max(min_must_weight, d*n_must / (d*n_must + n_nice))
    """
    if n_nice == 0:
        return 1.0, 0.0
    if n_must == 0:
        return 0.0, 1.0

    proportional = dominance_factor * n_must / (dominance_factor * n_must + n_nice)
    must_w = max(min_must_weight, proportional)
    nice_w = 1.0 - must_w
    return must_w, nice_w


def aggregate_score(
    results: list[dict[str, Any]],
    must_weight: float = DEFAULT_MUST_WEIGHT,
    nice_weight: float = DEFAULT_NICE_WEIGHT,
) -> tuple[float, float, float | None]:
    """根据 results = [{id, level, score(0/1)}, ...] 计算 (final_score, must_score, nice_score)。

    权重分配采用自适应策略：
    - must 总权重 >= 0.7，且每条 must 的单项权重严格大于每条 nice 的单项权重
    - 使用 dominance_factor=2 使 must 按比例获得更多权重
    - 当只有 must 无 nice 时，must 全对 = 满分 1.0
    - base_must == 0 时硬性记 0（主需不满足则全判零）
    """
    must_scores = [int(r.get('score', 0)) for r in results if r.get('level') == 'must_have']
    nice_scores = [int(r.get('score', 0)) for r in results if r.get('level') == 'nice_to_have']

    n_must = len(must_scores)
    n_nice = len(nice_scores)

    if n_must == 0 and n_nice == 0:
        return 0.0, 0.0, None

    base_must = sum(must_scores) / n_must if n_must else 0.0
    base_nice = sum(nice_scores) / n_nice if n_nice else None

    if n_must > 0 and base_must <= 0.0:
        return 0.0, float(base_must), (float(base_nice) if base_nice is not None else None)

    if n_nice == 0:
        return float(base_must), float(base_must), None

    if n_must == 0:
        return float(base_nice or 0.0), 0.0, float(base_nice or 0.0)

    adaptive_must_w, adaptive_nice_w = _compute_adaptive_weights(n_must, n_nice)
    s = float(adaptive_must_w * base_must + adaptive_nice_w * (base_nice or 0.0))
    s = max(0.0, min(1.0, s))
    return s, float(base_must), float(base_nice) if base_nice is not None else None


def _judgment_from_score(score: float) -> str:
    """把 [0, 1] 的连续分数映射回字符串 judgment（保持与 closed_qa_judge 字段一致）。"""
    if score >= 0.999:
        return '完全匹配'
    if score > 0.0:
        return '部分匹配'
    return '不匹配'


def _build_round_buckets(entries: list[Any]) -> dict[int, dict[str, str | None]]:
    """把 history 收敛成 round → {user, assistant} 桶。

    支持两种入参格式：
    - step7b 风格：每一项含 `round` 字段（int 或 int-str）
    - agts 流水线风格：扁平 [{role, content}, ...]，按出现顺序自动配对成 round（每个 user 触发新 round）
    """
    by_round: dict[int, dict[str, str | None]] = {}
    if not entries:
        return by_round

    has_round = any(isinstance(it, dict) and 'round' in it for it in entries)

    if has_round:
        for item in entries:
            if not isinstance(item, dict):
                continue
            r = item.get('round')
            if r is None:
                continue
            try:
                rk = int(r)
            except (TypeError, ValueError):
                continue
            role = item.get('role')
            raw = item.get('content')
            text = '' if raw is None else str(raw)
            slot = by_round.setdefault(rk, {'user': None, 'assistant': None})
            if role == 'user':
                slot['user'] = text
            elif role == 'assistant':
                slot['assistant'] = text
        return by_round

    cur_round = 0
    cur_slot: dict[str, str | None] | None = None
    for item in entries:
        if not isinstance(item, dict):
            continue
        role = item.get('role')
        raw = item.get('content')
        text = '' if raw is None else str(raw)
        if role == 'user':
            cur_round += 1
            cur_slot = by_round.setdefault(cur_round, {'user': None, 'assistant': None})
            cur_slot['user'] = text
        elif role == 'assistant':
            if cur_slot is None:
                cur_round += 1
                cur_slot = by_round.setdefault(cur_round, {'user': None, 'assistant': None})
            cur_slot['assistant'] = text
    return by_round


def _build_context_from_entries(entries: list[Any], max_rounds: int = 20) -> str:
    """对齐 step7b._build_context_from_entries 的多轮拼接格式。"""
    by_round = _build_round_buckets(entries)
    if not by_round:
        return '无'
    sorted_rounds = sorted(by_round.keys())
    if len(sorted_rounds) > max_rounds:
        sorted_rounds = sorted_rounds[-max_rounds:]
    last_two = set(sorted_rounds[-2:]) if len(sorted_rounds) >= 2 else set(sorted_rounds)
    lines: list[str] = []
    for rk in sorted_rounds:
        slot = by_round[rk]
        u = (slot.get('user') or '').strip()
        a = (slot.get('assistant') or '').strip()
        if u:
            lines.append(f'用户: {u[:1024]}')
        if rk in last_two and a:
            lines.append(f'助手: {a[:2048]}')
    return '\n'.join(lines) if lines else '无'


def _build_conversation(history: list[Any] | None, current_query: str, max_rounds: int = 20) -> str:
    """构建 verifier prompt 中 `conversation` 字段使用的对话上下文（含当前问题）。

    对齐 step7b.get_conversation：多轮按 round 排序、最近 max_rounds 轮、仅最后两轮显示助手，
    末尾追加 `用户（当前问题）: ...`，整体为空时返回 "无"。
    """
    ctx = _build_context_from_entries(history, max_rounds) if history else '无'
    pl = str(current_query or '').strip()
    if pl:
        suffix = f'用户（当前问题）: {pl}'
        ctx = ctx + ('\n' if ctx and ctx != '无' else '') + suffix
        ctx = ctx if ctx.strip() else '无'
    return ctx


def _parse_rubric(extra_info: dict) -> dict | None:
    """从 extra_info 中提取 rubric（包含 checkpoints）。

    支持以下来源（按优先级）：
    - extra_info['rubric']：dict 或 JSON str
    - extra_info['rubric_json']：JSON str
    """
    rubric = extra_info.get('rubric')
    if rubric is None:
        rubric = extra_info.get('rubric_json')
    if rubric is None:
        return None
    if isinstance(rubric, str):
        try:
            rubric = json.loads(rubric)
        except json.JSONDecodeError:
            return None
    if not isinstance(rubric, dict):
        return None
    return rubric


def _make_default_score_dict(
    must_weight: float = DEFAULT_MUST_WEIGHT,
    nice_weight: float = DEFAULT_NICE_WEIGHT,
) -> dict[str, Any]:
    """构造一个默认 score_dict，字段命名同时兼容 closed_qa_judge（closed_qa_score/judgment/...）。"""
    return {
        'closed_qa_score': 0.0,
        'judgment': '不匹配',
        'confidence': '低',
        'reason': '',
        'conciseness': '符合',
        'conciseness_reason': '',
        'rubric_verify_score': 0.0,
        'rubric_verify_must_score': 0.0,
        'rubric_verify_nice_score': None,
        'rubric_verify_results': [],
        'rubric_verify_overall_reason': '',
        'rubric_verify_must_weight': must_weight,
        'rubric_verify_nice_weight': nice_weight,
        'rubric_verify_error': '',
        'rubric_verify_error_type': '',
    }


def _build_results_for_empty_answer(cps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            'id': c.get('id', ''),
            'level': c.get('level', ''),
            'content': c.get('content', ''),
            'score': 0,
            'reason': '模型回答为空',
        }
        for c in cps
    ]


async def llm_as_a_judge(
    prompt: str,
    cps: list[dict[str, Any]],
    traj_id: str = 'unknown',
    must_weight: float = DEFAULT_MUST_WEIGHT,
    nice_weight: float = DEFAULT_NICE_WEIGHT,
) -> dict[str, Any]:
    """
    调用 LLM 做 rubric 级 verifier 打分。

    使用与 closed_qa_judge 一致的：
    - nexus ExecutionContext / OpenAI client（'judge'）
    - rate limiter 并发控制
    - 单次超时 + 指数退避重试
    """
    from nexus.core import require_context

    score_dict = _make_default_score_dict(must_weight, nice_weight)
    last_exception: Exception | None = None

    messages = [{'role': 'user', 'content': prompt}]
    rate_limiter = get_rate_limiter('llm_judge', max_concurrent=RATE_LIMIT)
    rate_limit_wait_time = 0.0

    _log_reward(traj_id, 'LLM_RUBRIC_JUDGE_START', f'n_checkpoints={len(cps)}')

    for attempt in range(MAX_RETRIES):
        try:
            _log_reward(traj_id, 'LLM_RUBRIC_JUDGE_ATTEMPT', f'attempt={attempt + 1}/{MAX_RETRIES}')

            rl_start = asyncio.get_event_loop().time()
            async with rate_limiter:
                rl_wait = asyncio.get_event_loop().time() - rl_start
                rate_limit_wait_time += rl_wait
                if rl_wait > 10:
                    _log_reward(traj_id, 'RATE_LIMIT_WAIT', f'{rl_wait:.2f}s')
                ctx = require_context()
                client = ctx.get_openai_client('judge')

                async with async_timeout(SINGLE_ATTEMPT_TIMEOUT):
                    _t0 = asyncio.get_event_loop().time()
                    response = await client.create_completion(
                        messages=messages,
                        max_tokens=15000,
                        reasoning_effort='high',
                        repetition_penalty=1.05,
                        max_retries=1,
                        enable_thinking=True,
                        query_id=traj_id,
                    )
                    _elapsed = asyncio.get_event_loop().time() - _t0
                    _usage = getattr(response, 'usage', None)
                    _completion_tokens = getattr(_usage, 'completion_tokens', None) if _usage else None
                    _thinking_tokens = getattr(_usage, 'completion_tokens_details', None)
                    _thinking_tokens = getattr(_thinking_tokens, 'reasoning_tokens', None) if _thinking_tokens else None
                    _log_reward(
                        traj_id,
                        'LLM_RUBRIC_JUDGE_CALL',
                        f'elapsed={_elapsed:.2f}s completion_tokens={_completion_tokens} '
                        f'thinking_tokens={_thinking_tokens}',
                    )

            result_text = response.choices[0].message.content
            if not result_text:
                continue

            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            json_content = json_match.group(1).strip() if json_match else result_text

            data = json.loads(json_content)

            if not isinstance(data, dict) or 'results' not in data or not isinstance(data['results'], list):
                raise ValueError('invalid format: missing results array')

            # 把 LLM 返回按输入 cp 顺序对齐：以输入 rubric 的 cps 顺序为准，缺失则补 0
            ret_by_id: dict[str, dict[str, Any]] = {}
            for r in data['results']:
                if not isinstance(r, dict):
                    continue
                rid = str(r.get('id', '')).strip()
                if not rid:
                    continue
                ret_by_id[rid] = r

            aligned: list[dict[str, Any]] = []
            for c in cps:
                cid = c.get('id', '')
                level = c.get('level', 'must_have')
                llm_r = ret_by_id.get(cid, {})
                score01 = _coerce_int01(llm_r.get('score', 0)) if llm_r else 0
                reason = str(llm_r.get('reason', '') if llm_r else 'verifier 未返回该 checkpoint 的判定')
                aligned.append({
                    'id': cid,
                    'level': level,
                    'content': c.get('content', ''),
                    'score': int(score01),
                    'reason': reason,
                })

            final_score, must_score, nice_score = aggregate_score(aligned, must_weight, nice_weight)
            overall_reason = str(data.get('overall_reason', ''))

            score_dict['closed_qa_score'] = float(final_score)
            score_dict['judgment'] = _judgment_from_score(final_score)
            score_dict['confidence'] = '高'
            score_dict['reason'] = overall_reason
            score_dict['rubric_verify_score'] = float(final_score)
            score_dict['rubric_verify_must_score'] = float(must_score)
            score_dict['rubric_verify_nice_score'] = float(nice_score) if nice_score is not None else None
            score_dict['rubric_verify_results'] = aligned
            score_dict['rubric_verify_overall_reason'] = overall_reason
            score_dict['reward_rate_limit_wait_time'] = rate_limit_wait_time
            # print(f'========final_score={final_score:.4f} must_score={must_score:.4f} nice_score={nice_score:.4f}')

            _log_reward(
                traj_id,
                'LLM_RUBRIC_JUDGE_DONE',
                f'score={final_score:.4f} must={must_score:.4f} '
                f'nice={"-" if nice_score is None else f"{nice_score:.4f}"}',
            )
            return score_dict

        except json.JSONDecodeError as e:
            logger.debug(f'[{traj_id}] LLM rubric judge attempt {attempt + 1}/{MAX_RETRIES} JSON decode error: {e}')
            last_exception = e
            await retry_backoff_sleep(attempt)
        except TimeoutError:
            logger.debug(
                f'[{traj_id}] LLM rubric judge attempt {attempt + 1}/{MAX_RETRIES} '
                f'timed out after {SINGLE_ATTEMPT_TIMEOUT}s'
            )
            last_exception = TimeoutError(f'single attempt timeout after {SINGLE_ATTEMPT_TIMEOUT}s')
            await retry_backoff_sleep(attempt)
        except Exception as e:
            logger.debug(
                f'[{traj_id}] LLM rubric judge attempt {attempt + 1}/{MAX_RETRIES} exception: {type(e).__name__}: {e}'
            )
            last_exception = e
            await retry_backoff_sleep(attempt)

    _log_reward(traj_id, 'LLM_RUBRIC_JUDGE_FAILED', f'max_retries={MAX_RETRIES}')
    logger.error(
        f'[{traj_id}] LLM rubric judge failed after {MAX_RETRIES} retries: '
        f'{type(last_exception).__name__ if last_exception else "RuntimeError"}: {last_exception}'
    )
    if last_exception is not None:
        raise last_exception
    raise RuntimeError(f'LLM rubric judge failed after {MAX_RETRIES} retries')


async def compute_score(
    solution_str: str,
    ground_truth: str = '无参考答案',
    extra_info: dict | None = None,
) -> dict[str, Any]:
    """
    计算 Closed QA Rubric 评分。

    Args:
        solution_str: 模型生成的完整回答
        ground_truth: 参考答案（仅作为辅助上下文写入 prompt）
        extra_info: 额外信息（包含 prompt/messages、history、rubric、traj_id 等）

    Returns:
        包含评分结果的字典；字段命名兼容 closed_qa_judge（closed_qa_score / judgment / ...）
    """
    extra_info = extra_info or {}
    traj_id = extra_info.get('traj_id', 'unknown')

    extra_info['model_type'] = extra_info.get('model_type', 'ds')

    answer = extract_query_solution(solution_str, extra_info)

    query = extra_info.get('query', extra_info.get('question', ''))

    prompt_messages = extra_info.get('prompt', extra_info.get('messages', []))

    def has_content(obj: Any) -> bool:
        if obj is None:
            return False
        try:
            return len(obj) > 0
        except (TypeError, ValueError):
            return False

    if hasattr(prompt_messages, 'tolist'):
        prompt_messages = prompt_messages.tolist()

    if not query and has_content(prompt_messages):
        query = extract_query_from_prompt(prompt_messages)

    history = extra_info.get('history', [])
    if hasattr(history, 'tolist'):
        history = history.tolist()

    if not has_content(history) and has_content(prompt_messages):
        history = extract_history_from_prompt(prompt_messages)

    if history and history[0].get('role') == 'system':
        history = history[1:]

    rubric = _parse_rubric(extra_info)
    cps = (rubric.get('checkpoints') or []) if isinstance(rubric, dict) else []

    must_weight = float(extra_info.get('rubric_must_weight', DEFAULT_MUST_WEIGHT))
    nice_weight = float(extra_info.get('rubric_nice_weight', DEFAULT_NICE_WEIGHT))
    # print(f"========extra_info:{list(extra_info.keys())}")
    # print(f"========rubric:{rubric}")
    # rubric 缺失：verifier 没法工作，降级为 0 分并打错误标记
    if not cps:
        score_dict = _make_default_score_dict(must_weight, nice_weight)
        score_dict['rubric_verify_error'] = 'empty_rubric'
        score_dict['rubric_verify_error_type'] = 'empty_rubric'
        score_dict['rubric_verify_overall_reason'] = 'rubric checkpoints 为空'
        score_dict['reason'] = 'rubric checkpoints 为空'
        logger.warning(f'[{traj_id}] rubric checkpoints empty, score=0')
        return score_dict

    # 模型答案为空：真实"答错"，按 0 算（仍走聚合逻辑保持字段一致）
    if not answer.strip():
        empty_results = _build_results_for_empty_answer(cps)
        final_score, must_score, nice_score = aggregate_score(empty_results, must_weight, nice_weight)
        score_dict = _make_default_score_dict(must_weight, nice_weight)
        score_dict['closed_qa_score'] = float(final_score)
        score_dict['judgment'] = _judgment_from_score(final_score)
        score_dict['reason'] = '模型回答为空'
        score_dict['rubric_verify_score'] = float(final_score)
        score_dict['rubric_verify_must_score'] = float(must_score)
        score_dict['rubric_verify_nice_score'] = float(nice_score) if nice_score is not None else None
        score_dict['rubric_verify_results'] = empty_results
        score_dict['rubric_verify_overall_reason'] = '模型回答为空'
        return score_dict

    rubric_json = json.dumps({'checkpoints': cps}, ensure_ascii=False, indent=2)

    # 对齐 step7b：conversation = 多轮历史按 round 拼接 + 末尾追加 `用户（当前问题）: ...`
    conversation = _build_conversation(history if has_content(history) else None, query)

    prompt = (
        PROMPT_TEMPLATE
        .replace('{conversation}', conversation)
        .replace('{question}', str(query))
        .replace('{ground_truth}', str(ground_truth))
        .replace('{rubric_json}', rubric_json)
        .replace('{model_answer}', answer)
    )

    return await llm_as_a_judge(prompt, cps, traj_id=traj_id, must_weight=must_weight, nice_weight=nice_weight)
