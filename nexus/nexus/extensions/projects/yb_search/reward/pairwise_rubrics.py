"""
Pairwise Rubric LLM Judge 评分模块。

针对 pairwise 数据中带 `rubrics` 字段的样本，用 LLM 对候选回答在每条 rubric
检查点上逐项判定命中情况，再按 rubric 类型（误区点 / 必答点 / 答案命中点 / 加分点）
聚合出 [0, 1] 的归一化得分。

判分逻辑与 prompt 对齐 RubricSynthesisPipeline/verifier（eval_rubrics.py + judge_prompts.py）：
    - 任一 误区点 被命中(hit=true)        → score = 0（fatal）
    - 任一 必答点 未命中(hit=false)       → score = 0（fatal）
    - 否则 raw = 必答点基础分 + 答案命中点得分 + 加分点得分 + 分组均衡奖励
    - 归一化 score = raw / max_score（max_score 为理论满分），裁剪到 [0, 1]

LLM 调用基础设施（context / judge client / rate limiter / 重试 / 超时）与
closed_qa_rubric_judge.py 完全一致，复用同一个 'judge' 模型服务。
"""

import asyncio
import json
import logging
import os
import re

from typing import Any

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


def _log_reward(traj_id: str, event: str, message: str = '') -> None:
    logger.warning(f'[{traj_id}] {event} {message}'.rstrip())


# rubric 类型常量（对齐 judge_prompts.py / eval_rubrics.py）
_TYPE_MISTAKE = '误区点'
_TYPE_MUST = '必答点'
_TYPE_HIT = '答案命中点'
_TYPE_BONUS = '加分点'

print(
    f'PAIRWISE_RUBRICS_JUDGE initialized with: {RATE_LIMIT=}, {MAX_RETRIES=}, '
    f'model={JUDGE_MODEL}, single_attempt_timeout={SINGLE_ATTEMPT_TIMEOUT}'
)


# Rubric verifier prompt（对齐 RubricSynthesisPipeline/verifier/judge_prompts.py 的 JUDGE_PROMPT）。
# 使用 .replace() 填充 ${...} 占位符，JSON 示例中的字面 `{` `}` 不受影响。
JUDGE_PROMPT = r"""你是一个**严格客观**的 Rubric 评分器。给定一道题目（含历史对话上下文）、一份 Rubric 检查清单以及一个候选回答，请逐条判断该回答是否**命中**清单中的每一项。

## Rubric 类型说明

Rubric 检查点分为 4 种类型，每种类型的"命中"含义不同：

| 类型 | 含义 | hit=true 表示 | hit=false 表示 |
|---|---|---|---|
| **误区点** | 回答中不应出现的错误/陷阱 | 回答**犯了**该错误（扣分项） | 回答**未犯**该错误（正常） |
| **必答点** | 回答必须满足的基本要求 | 回答**满足**该要求（正常） | 回答**未满足**该要求（致命缺陷） |
| **答案命中点** | 回答中值得加分的内容要点 | 回答**覆盖了**该要点（加分） | 回答**未覆盖**该要点（不加分） |
| **加分点** | 超出基本要求的额外亮点 | 回答**包含**该亮点（额外加分） | 回答**未包含**该亮点（不扣分） |

## 判定规则

- **命中 (hit=true)**：回答的内容**实质满足**该检查点的描述（措辞可以不同，但内容/事实/角度都已覆盖到）
- **未命中 (hit=false)**：回答完全没涉及、明显违背、或仅形式上提及但未实质满足
- 对于**误区点**：判断回答是否**触犯**了该误区（hit=true 表示犯了错误）
- 对于**必答点/答案命中点/加分点**：判断回答是否**满足/覆盖**了该要求

## 注意
- 只看候选回答本身的内容，不要看历史对话中已经说过的内容
- 不要因风格/语气/长短打折，只看是否实质覆盖检查点要求
- 检查点中如果包含具体数值或事实，回答里出现等价表述（数值范围、不同单位等）也算命中
- 严格按检查点本身字面要求判断，宁可严格不可宽松
- 每条检查点已包含足够的事实信息供你判断，请据此独立评判

## 评分逻辑（供你理解各类型的重要性，判定时请逐条独立判断）
- 若任一 `误区点` 被命中（hit=true），则该回答最终得分 = 0
- 若任一 `必答点` 未被命中（hit=false），则该回答最终得分 = 0
- 只有当所有 `必答点` 均命中 且 无 `误区点` 被触犯时，才累加分数
- 最终得分 = 必答点基础分 + 答案命中点得分之和 + 加分点得分之和
- 若评分点与字数相关，仅参考字数统计结果，禁止自行统计字数

## 输入

**基础信息**
- 用户提问日期：${datetime}
- 用户位置：${location}

**历史对话**
```
${history}
```

**当前问题**
```
${query}
```

**候选回答**
```
${answer}
```

**字数统计**
${word_count_block}

**Rubric 检查清单（共 ${n} 项，逐项判断）**
```
${checklist_block}
```

## 输出

请严格按照以下 JSON 格式输出（不要多余解释，直接给 JSON 块）：

```json
{
  "judgements": [
    {"id": "check-1", "hit": true,  "reason": "（不超过 30 字的简要理由）"},
    {"id": "check-2", "hit": false, "reason": "..."}
  ]
}
```

注意：
- judgements 数组的长度必须等于 ${n}
- 每项的 id 必须与检查清单中给定的 id 完全一致
- hit 必须是 JSON 布尔值 true/false（不要写字符串"true"/"false"）
- 对于误区点，hit=true 表示回答犯了该错误；对于其他类型，hit=true 表示回答满足了该要求
"""


# ── 字数统计：对齐 judge_prompts.count_pure_text_len ──────────────────────────
_STRIP_PATTERNS = [
    re.compile(r'\[citation:\d+\]'),
    re.compile(r'\[available:\d+\]'),
    re.compile(r'</?answer>'),
    re.compile(r'^\s{0,3}#{1,6}\s+', flags=re.MULTILINE),
    re.compile(r'\*{1,3}|_{1,3}|~{2}'),
    re.compile(r'^\s*[-*+]\s+', flags=re.MULTILINE),
    re.compile(r'^\s*\d+[.)]\s+', flags=re.MULTILINE),
    re.compile(r'\[([^\]]*)\]\([^)]*\)'),
    re.compile(r'`{1,3}'),
    re.compile(r'^\s*>\s?', flags=re.MULTILINE),
    re.compile(r'^\s*\|.*\|\s*$', flags=re.MULTILINE),
    re.compile(r'^\s*[-:| ]+\s*$', flags=re.MULTILINE),
]

_THINK_TAG_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


def _strip_think(text: str) -> str:
    """移除 <think>...</think> 思考块，返回正文。"""
    result = _THINK_TAG_RE.sub('', text)
    if '</think>' in result:
        result = result.split('</think>', 1)[-1]
    return result.strip()


def count_pure_text_len(text: str) -> int:
    """统计纯文本字数（中文 1 字、连续英文 / 连续数字各算 1 字），去除 markdown / 标签 / 引用。"""
    if not text:
        return 0
    for pat in _STRIP_PATTERNS:
        text = pat.sub(lambda m: m.group(1) if m.lastindex else '', text)
    count = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if '\u4e00' <= ch <= '\u9fff':
            count += 1
            i += 1
        elif ch.isascii() and ch.isalpha():
            count += 1
            i += 1
            while i < len(text) and text[i].isascii() and text[i].isalpha():
                i += 1
        elif ch.isdigit():
            count += 1
            i += 1
            while i < len(text) and text[i].isdigit():
                i += 1
        else:
            i += 1
    return count


def build_word_count_block(answer: str) -> str:
    """构建字数统计块（填充 ${word_count_block}）。"""
    n = count_pure_text_len(answer)
    return f'（候选回答纯文本字数：{n}字。已去除markdown语法、标签、引用标记等，仅统计实际内容文字）'


def _parse_rubrics(extra_info: dict) -> list[dict]:
    """从 extra_info 中提取 rubric checkpoints 列表。

    兼容多种来源：
    - extra_info['rubrics']：list / numpy array / JSON str
    - dict 包装（{'rubrics': [...]} 或 {'checkpoints': [...]}）
    """
    rubrics = extra_info.get('rubrics')
    if rubrics is None:
        return []
    if hasattr(rubrics, 'tolist'):
        rubrics = rubrics.tolist()
    if isinstance(rubrics, str):
        try:
            rubrics = json.loads(rubrics)
        except json.JSONDecodeError:
            return []
    if isinstance(rubrics, dict):
        rubrics = rubrics.get('rubrics') or rubrics.get('checkpoints') or []
    if not isinstance(rubrics, list):
        return []
    return [c for c in rubrics if isinstance(c, dict)]


def _rubric_type(c: dict) -> str:
    """读取 rubric 类型（importance 优先，兼容 rubric_type / type）。"""
    return c.get('importance') or c.get('rubric_type') or c.get('type') or ''


def _coerce_score(v: Any) -> float:
    """把 rubric 的 score 字段安全收敛为 float（缺省 1.0）。"""
    if v is None or v == '':
        return 1.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 1.0


def _criterion_of(c: dict) -> str:
    return str(c.get('criterion', c.get('check_point', '')) or '')


def _group_id_of(c: dict) -> str:
    gid = c.get('group_id', '')
    if gid == 'null' or gid is None:
        return ''
    return str(gid)


def _format_checklist_block(checklist: list[dict]) -> str:
    """渲染 checklist 为可读文本块（对齐 eval_rubrics._format_checklist_block）。"""
    blocks = []
    for c in checklist:
        cid = c.get('id', '')
        rubric_type = _rubric_type(c)
        dim = c.get('dimension', '')
        cp = _criterion_of(c)
        score = c.get('score', '')
        group_id = _group_id_of(c)

        header_parts = [f'id: {cid}']
        if rubric_type:
            header_parts.append(f'类型: {rubric_type}')
        if dim:
            header_parts.append(f'维度: {dim}')
        if score != '' and score is not None:
            header_parts.append(f'分值: {score}')
        if group_id:
            header_parts.append(f'分组: {group_id}')

        blocks.append(f'- {" | ".join(header_parts)}\n  检查点: {cp}')
    return '\n'.join(blocks)


def _format_history(history: list[Any] | None, max_rounds: int = 20) -> str:
    """把对话历史渲染成可读字符串（填充 ${history}）。"""
    if not history:
        return '(无)'
    lines: list[str] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get('role', '')
        content = '' if item.get('content') is None else str(item.get('content'))
        content = content.strip()
        if not content:
            continue
        if role == 'user':
            lines.append(f'用户: {content[:1024]}')
        elif role == 'assistant':
            lines.append(f'助手: {content[:2048]}')
    if len(lines) > max_rounds * 2:
        lines = lines[-max_rounds * 2 :]
    return '\n'.join(lines) if lines else '(无)'


# ── LLM 返回解析 ─────────────────────────────────────────────────────────────
def _loads_lenient(s: str) -> Any:
    """尽量解析 JSON：优先用 json_repair 修复，失败回退到标准 json。"""
    try:
        from json_repair import repair_json  # type: ignore

        return json.loads(repair_json(s))
    except Exception:
        return json.loads(s)


def _coerce_hit(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v >= 0.5
    if isinstance(v, str):
        return v.strip().lower() in ('true', 'yes', '命中', '1', 'hit')
    return False


def _align_judgements(raw: str, cps: list[dict]) -> list[dict]:
    """从 LLM 回复中抽出 judgements，按 cps 的顺序 / id 对齐，缺失项补 hit=False。"""
    parsed = None
    blocks = re.findall(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
    for block in reversed(blocks):
        try:
            parsed = _loads_lenient(block)
            break
        except Exception:
            continue
    if parsed is None:
        try:
            parsed = _loads_lenient(raw)
        except Exception:
            parsed = {}

    judgements = parsed.get('judgements', []) if isinstance(parsed, dict) else []
    by_id: dict[str, dict[str, Any]] = {}
    for j in judgements:
        if isinstance(j, dict) and 'id' in j:
            by_id[str(j['id'])] = j

    aligned: list[dict[str, Any]] = []
    for i, c in enumerate(cps):
        cid = str(c.get('id', f'r{i + 1}'))
        j = by_id.get(cid)
        if j is None and i < len(judgements) and isinstance(judgements[i], dict):
            j = judgements[i]
        j = j or {}
        aligned.append({
            'id': cid,
            'rubric_type': _rubric_type(c),
            'dimension': c.get('dimension', ''),
            'criterion': _criterion_of(c),
            'score': _coerce_score(c.get('score', 1)),
            'group_id': _group_id_of(c),
            'hit': _coerce_hit(j.get('hit', False)),
            'reason': str(j.get('reason', '')),
        })
    return aligned


def _build_empty_answer_judgements(cps: list[dict]) -> list[dict]:
    """模型回答为空时构建全 miss 的判定结果（误区点视为未触犯）。"""
    aligned = []
    for i, c in enumerate(cps):
        aligned.append({
            'id': str(c.get('id', f'r{i + 1}')),
            'rubric_type': _rubric_type(c),
            'dimension': c.get('dimension', ''),
            'criterion': _criterion_of(c),
            'score': _coerce_score(c.get('score', 1)),
            'group_id': _group_id_of(c),
            'hit': False,
            'reason': '模型回答为空',
        })
    return aligned


# ── rubric 得分计算（对齐 eval_rubrics）──────────────────────────────────────
def compute_rubric_score(judgements: list[dict]) -> dict[str, Any]:
    """根据 rubric 判定结果计算原始得分。

    - 任一 误区点 hit=true            → fatal，score = 0
    - 任一 必答点 hit=false           → fatal，score = 0
    - 否则 score = 必答点基础分 + 答案命中点得分 + 加分点得分
    """
    fatal = False
    must_pass_score = 0.0
    hit_score = 0.0
    bonus_score = 0.0

    for j in judgements:
        rtype = j.get('rubric_type', '')
        hit = bool(j.get('hit', False))
        item_score = _coerce_score(j.get('score', 1))

        if rtype == _TYPE_MISTAKE:
            if hit:
                fatal = True
        elif rtype == _TYPE_MUST:
            if hit:
                must_pass_score += item_score
            else:
                fatal = True
        elif rtype == _TYPE_HIT:
            if hit:
                hit_score += item_score
        elif rtype == _TYPE_BONUS and hit:
            bonus_score += item_score

    if fatal:
        return {
            'score': 0.0,
            'fatal': True,
            'must_pass_score': must_pass_score,
            'hit_score': hit_score,
            'bonus_score': bonus_score,
            'balance_bonus': 0.0,
        }

    total = must_pass_score + hit_score + bonus_score
    return {
        'score': total,
        'fatal': False,
        'must_pass_score': must_pass_score,
        'hit_score': hit_score,
        'bonus_score': bonus_score,
        'balance_bonus': 0.0,
    }


def compute_balance_bonus(judgements: list[dict]) -> float:
    """计算分组均衡奖励：各组得分之和 − 最大组得分（少于 2 组则为 0）。"""
    group_scores: dict[str, float] = {}
    for j in judgements:
        if j.get('rubric_type') != _TYPE_HIT:
            continue
        gid = j.get('group_id', '')
        if not gid:
            continue
        if j.get('hit', False):
            group_scores[gid] = group_scores.get(gid, 0.0) + _coerce_score(j.get('score', 1))
        else:
            group_scores.setdefault(gid, 0.0)

    if len(group_scores) < 2:
        return 0.0
    scores = list(group_scores.values())
    return sum(scores) - max(scores)


def compute_max_score(checklist: list[dict]) -> float:
    """计算理论满分 = 必答点 + 答案命中点 + 加分点总分 + 最大可能均衡奖励。"""
    base = 0.0
    group_max: dict[str, float] = {}
    for c in checklist:
        rtype = _rubric_type(c)
        item_score = _coerce_score(c.get('score', 1))
        if rtype in (_TYPE_MUST, _TYPE_HIT, _TYPE_BONUS):
            base += item_score
        if rtype == _TYPE_HIT:
            gid = _group_id_of(c)
            if gid:
                group_max[gid] = group_max.get(gid, 0.0) + item_score

    max_balance = 0.0
    if len(group_max) >= 2:
        scores = list(group_max.values())
        max_balance = sum(scores) - max(scores)
    return base + max_balance


def _aggregate(aligned: list[dict], cps: list[dict]) -> dict[str, Any]:
    """聚合 judgements → 归一化 [0, 1] 得分及明细字段。"""
    score_info = compute_rubric_score(aligned)
    balance_bonus = compute_balance_bonus(aligned)
    score_info['balance_bonus'] = balance_bonus
    if not score_info['fatal']:
        score_info['score'] += balance_bonus

    raw_score = float(score_info['score'])
    max_score = compute_max_score(cps)
    norm = raw_score / max_score if max_score > 0 else 0.0 if score_info['fatal'] else 1.0
    norm = max(0.0, min(1.0, norm))

    n_hit = sum(1 for j in aligned if j.get('hit'))
    return {
        'pairwise_rubric_score': float(norm),
        'pairwise_rubric_raw_score': raw_score,
        'pairwise_rubric_max_score': float(max_score),
        'pairwise_rubric_fatal': bool(score_info['fatal']),
        'pairwise_rubric_must_pass_score': float(score_info['must_pass_score']),
        'pairwise_rubric_hit_score': float(score_info['hit_score']),
        'pairwise_rubric_bonus_score': float(score_info['bonus_score']),
        'pairwise_rubric_balance_bonus': float(balance_bonus),
        'pairwise_rubric_n_hit': int(n_hit),
        'pairwise_rubric_n_total': len(cps),
        'pairwise_rubric_results': aligned,
        'pairwise_rubric_overall_reason': '',
        'pairwise_rubric_error': '',
        'pairwise_rubric_error_type': '',
        'pairwise_rubric_skip': False,
    }


def _make_default_score_dict() -> dict[str, Any]:
    return {
        'pairwise_rubric_score': 0.0,
        'pairwise_rubric_raw_score': 0.0,
        'pairwise_rubric_max_score': 0.0,
        'pairwise_rubric_fatal': False,
        'pairwise_rubric_must_pass_score': 0.0,
        'pairwise_rubric_hit_score': 0.0,
        'pairwise_rubric_bonus_score': 0.0,
        'pairwise_rubric_balance_bonus': 0.0,
        'pairwise_rubric_n_hit': 0,
        'pairwise_rubric_n_total': 0,
        'pairwise_rubric_results': [],
        'pairwise_rubric_overall_reason': '',
        'pairwise_rubric_error': '',
        'pairwise_rubric_error_type': '',
        'pairwise_rubric_skip': True,
    }


async def llm_as_a_judge(prompt: str, cps: list[dict], traj_id: str = 'unknown') -> dict[str, Any]:
    """调用 LLM 做 rubric 级 verifier 打分（复用 'judge' 模型服务）。

    LLM 调用基础设施（context / judge client / rate limiter / 单次超时 + 指数退避重试）
    与 closed_qa_rubric_judge.llm_as_a_judge 完全一致。
    """
    from nexus.core import require_context

    last_exception: Exception | None = None
    messages = [{'role': 'user', 'content': prompt}]
    rate_limiter = get_rate_limiter('llm_judge', max_concurrent=RATE_LIMIT)

    _log_reward(traj_id, 'PAIRWISE_RUBRIC_JUDGE_START', f'n_checkpoints={len(cps)}')

    for attempt in range(MAX_RETRIES):
        try:
            _log_reward(traj_id, 'PAIRWISE_RUBRIC_JUDGE_ATTEMPT', f'attempt={attempt + 1}/{MAX_RETRIES}')

            rl_start = asyncio.get_event_loop().time()
            async with rate_limiter:
                rl_wait = asyncio.get_event_loop().time() - rl_start
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
                        'PAIRWISE_RUBRIC_JUDGE_CALL',
                        f'elapsed={_elapsed:.2f}s completion_tokens={_completion_tokens} '
                        f'thinking_tokens={_thinking_tokens}',
                    )

            result_text = response.choices[0].message.content
            if not result_text:
                continue

            raw = _strip_think(result_text)
            aligned = _align_judgements(raw, cps)

            # 兜底：若所有 hit 均为 False 且无任何 reason，疑似 LLM 未正确返回 → 重试
            if not any(a['reason'] for a in aligned) and not any(a['hit'] for a in aligned):
                raise ValueError('empty judgements (no hit / no reason)')

            score_dict = _aggregate(aligned, cps)

            _log_reward(
                traj_id,
                'PAIRWISE_RUBRIC_JUDGE_DONE',
                f'score={score_dict["pairwise_rubric_score"]:.4f} '
                f'raw={score_dict["pairwise_rubric_raw_score"]:.4f}/'
                f'{score_dict["pairwise_rubric_max_score"]:.4f} '
                f'fatal={score_dict["pairwise_rubric_fatal"]} '
                f'hit={score_dict["pairwise_rubric_n_hit"]}/{score_dict["pairwise_rubric_n_total"]}',
            )
            return score_dict

        except TimeoutError:
            logger.warning(
                f'[{traj_id}] pairwise rubric judge attempt {attempt + 1}/{MAX_RETRIES} '
                f'timed out after {SINGLE_ATTEMPT_TIMEOUT}s'
            )
            last_exception = TimeoutError(f'single attempt timeout after {SINGLE_ATTEMPT_TIMEOUT}s')
            await retry_backoff_sleep(attempt)
        except Exception as e:
            logger.warning(f'Pairwise rubric judge attempt {attempt + 1}/{MAX_RETRIES} warning: {e}')
            last_exception = e
            await retry_backoff_sleep(attempt)

    _log_reward(traj_id, 'PAIRWISE_RUBRIC_JUDGE_FAILED', f'max_retries={MAX_RETRIES}')
    logger.error(f'Pairwise rubric judge failed after {MAX_RETRIES} retries')
    if last_exception is not None:
        raise last_exception
    raise RuntimeError(f'Pairwise rubric judge failed after {MAX_RETRIES} retries')


async def compute_score(
    solution_str: str,
    ground_truth: str = '无参考答案',
    extra_info: dict | None = None,
) -> dict[str, Any]:
    """
    计算 Pairwise Rubric 评分。

    Args:
        solution_str: 模型生成的完整回答
        ground_truth: 参考答案（rubric verifier 仅依据 checkpoints 判定，GT 不直接使用）
        extra_info: 额外信息（包含 prompt/messages、history、rubrics、traj_id 等）

    Returns:
        包含 pairwise_rubric_* 字段的字典；pairwise_rubric_score 为归一化到 [0, 1] 的得分。
    """
    extra_info = extra_info or {}
    traj_id = extra_info.get('traj_id', 'unknown')
    extra_info.setdefault('model_type', 'ds')

    skip_rubrics = extra_info.get('skip_rubrics', False)
    cps = _parse_rubrics(extra_info)

    # rubrics 缺失：verifier 无法工作，降级为 0 分并打错误标记
    if not cps or skip_rubrics:
        score_dict = _make_default_score_dict()
        score_dict['pairwise_rubric_overall_reason'] = 'rubrics 为空'
        logger.warning(f'[{traj_id}] pairwise rubrics empty, score=0')
        return score_dict

    answer = extract_query_solution(solution_str, extra_info)

    query = extra_info.get('query', extra_info.get('question', ''))
    prompt_messages = extra_info.get('prompt', extra_info.get('messages', []))
    if hasattr(prompt_messages, 'tolist'):
        prompt_messages = prompt_messages.tolist()

    def has_content(obj: Any) -> bool:
        if obj is None:
            return False
        try:
            return len(obj) > 0
        except (TypeError, ValueError):
            return False

    if not query and has_content(prompt_messages):
        query = extract_query_from_prompt(prompt_messages)

    history = extra_info.get('history', [])
    if hasattr(history, 'tolist'):
        history = history.tolist()
    if not has_content(history) and has_content(prompt_messages):
        history = extract_history_from_prompt(prompt_messages)
    if has_content(history) and isinstance(history[0], dict) and history[0].get('role') == 'system':
        history = history[1:]

    # 模型答案为空：真实"答错"，按全 miss 走聚合（必答点未命中 → fatal → 0）
    if not answer.strip():
        empty = _build_empty_answer_judgements(cps)
        score_dict = _aggregate(empty, cps)
        score_dict['pairwise_rubric_overall_reason'] = '模型回答为空'
        return score_dict

    prompt = (
        JUDGE_PROMPT
        .replace('${datetime}', str(extra_info.get('current_time', '') or ''))
        .replace('${location}', str(extra_info.get('current_location', '') or ''))
        .replace('${history}', _format_history(history if has_content(history) else None))
        .replace('${query}', str(query or ''))
        .replace('${answer}', answer or '(空)')
        .replace('${word_count_block}', build_word_count_block(answer))
        .replace('${n}', str(len(cps)))
        .replace('${checklist_block}', _format_checklist_block(cps))
    )

    return await llm_as_a_judge(prompt, cps, traj_id=traj_id)
