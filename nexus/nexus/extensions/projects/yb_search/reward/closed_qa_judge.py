"""
Closed QA Judge 评分模块。

针对封闭性问答，使用 LLM 评估模型回答的答案匹配度：
- 完全匹配 (1.0)
- 部分匹配 (0.5)
- 不匹配 (0.0)
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
RATE_LIMIT = int(os.environ.get('TAIJI_JUDGE_RATE_LIMIT', 6))  # 最大并发数
JUDGE_MODEL = os.environ.get('TAIJI_JUDGE_MODEL', '')

_log_reward = partial(log_trace, 'REWARD')

print(
    f'CLOSED_QA_JUDGE initialized with: {RATE_LIMIT=}, {MAX_RETRIES=}, '
    f'model={JUDGE_MODEL}, wsid={os.environ.get("TAIJI_JUDGE_WSID")} '
    f'single_attempt_timeout={SINGLE_ATTEMPT_TIMEOUT}'
)


# 评分 prompt 模板（对齐 baseline new_data_judge.py）
PROMPT_TEMPLATE = """你是一个专业的答案匹配评估专家。你的任务是判断模型回答是否正确回答了用户问题。

## 核心评估原则

**评估核心**：判断模型是否正确回答了用户问题的核心需求，而非机械对比文本差异。

**答案形式要求**：本评估针对**封闭性问题**，参考答案均为简短直接的回答。模型应：
- ✅ 直接给出答案，篇幅与参考答案相当（允许±50%字数差异）
- ❌ 避免冗长解释、背景铺垫、过度扩展等非必要内容

**评估顺序**：
1. 首先识别用户问题的核心意图和关键信息需求
2. 判断模型回答是否满足这个核心需求
3. 再检查参考答案中的补充信息是否被正确处理
4. 最后评估回答是否简洁直接

## 评估标准（三档判定）

### ✅ 【完全匹配】
- **核心问题已正确回答**：模型回答准确满足用户问题的主要需求
- **关键事实一致**：核心数值、实体、概念与参考答案相同或语义等价
- **合理扩展**：额外提供的相关信息正确且有价值（即使参考答案未包含）
- **合理纠错**：当用户问题包含明显错误（错别字、错误实体名）时，模型进行了正确纠正

### ⚠️ 【部分匹配】
- **核心问题基本回答**：主要需求得到满足，但存在以下情况：
  - 数值精度差异（相对误差0.5%-1%）
  - 参考答案中的**核心信息**有轻微遗漏（80%以上已覆盖）
  - 表述略有歧义但事实基本正确

### ❌ 【不匹配】
- **核心问题未正确回答**：用户的主要需求未得到满足
- **关键事实错误**：核心数值、实体、概念明显错误
- **严重幻觉**：编造与参考答案明显矛盾的重要事实

## 信息层级区分

评估时需区分参考答案中的信息层级：

**一级信息（核心必答）**：直接回应用户问题的关键信息
- 这些信息必须正确回答，否则判定为不匹配

**二级信息（补充验证）**：参考答案提供但非用户直接询问的补充信息
- 模型回答正确 → 加分项
- 模型未提及 → 不扣分（只要核心信息正确）
- 模型回答错误 → 酌情扣分（视错误严重程度）

**示例说明**：
- 用户问："全红婵退出比赛了吗？"
  - 一级信息：是否退出（必答）
  - 二级信息：冠军是谁、具体成绩、退出原因等（补充）
- 用户问："另外两本书是什么？"
  - 一级信息：两本书的书名（必答）
  - 二级信息：作者、出版信息等（补充）

## 扩展信息处理规则

模型回答中可能包含参考答案未提及的扩展信息，处理原则：
1. **扩展信息不矛盾**：扩展内容与参考答案不冲突 → 视为合理扩展，不影响判定
2. **扩展信息矛盾**：扩展内容与参考答案明确矛盾 → 判定为幻觉
3. **扩展信息无法验证**：参考答案未涉及的细节信息 → 默认不判定为错误，除非明显不合理

**重要**：不要因为参考答案中没有某个细节，就判定模型的补充为幻觉。只有明确矛盾时才判定为错误。

## 数值判定规则

### 必须精确匹配的场景：
- 版本号、端口号、ID编码、日期、计数

### 允许合理误差的场景：
- 价格、性能指标、百分比等
- 相对误差 < 0.5% → 完全匹配
- 相对误差 0.5%-1% → 部分匹配
- 相对误差 ≥ 1% → 不匹配

计算公式：`相对误差 = |模型值 - 参考值| / 参考值 × 100%`

## 特殊场景处理

**用户问题有误**：
- 如用户问题包含错别字或错误实体名
- 模型正确识别并纠正，给出正确答案
- 应判定为完全匹配，不应纠结于用户问题中的错误

**条件类问题**：
- 条件错误（如问2024年答2023年）→ 不匹配
- 条件正确但内容有瑕疵 → 部分匹配

**枚举类问题**：
- 完全匹配：包含所有核心项（顺序可不同）
- 部分匹配：包含80%以上核心项
- 不匹配：遗漏或错误 > 20%

## 标准示例

**示例1：用户问题有误，模型正确纠错**
- 问题：建业城有位好心的医生...他是谁？（"建业城"为错别字）
- 参考答案：建邺城...陈长寿
- 模型回答：建邺城...陈长寿
- 判断：✅ 完全匹配（模型正确纠正了地名，核心问题回答正确）

**示例2：核心问题已答，补充信息未全答**
- 问题：全红婵退出比赛了吗？
- 参考答案：退出了；冠军陈芋汐412.30分；亚军卢为384.00分
- 模型回答：退出了；冠军陈芋汐412.30分；亚军卢为（未提及分数）
- 判断：✅ 完全匹配（核心问题"是否退出"已正确回答，冠军信息完整，亚军分数为二级信息，未提及不影响核心判定）

**示例3：合理扩展不应判为幻觉**
- 问题：另外两本书是什么？
- 参考答案：《阶级斗争》和《社会主义史》
- 模型回答：《阶级斗争》（考茨基著）和《社会主义史》（柯卡普著）
- 判断：✅ 完全匹配（核心问题正确回答，作者信息为合理扩展，参考答案未提及不代表错误）

**示例4：核心问题未正确回答**
- 问题：Pro版的最大并发数是多少？
- 参考答案：500
- 模型回答：Pro版最大并发数为5000
- 判断：❌ 不匹配（核心数值错误，数量级错误）

**示例 5：完全匹配 - 条件正确且包含对比**
- 问题：2024 版 Pro 套餐的价格是多少？
- 参考答案：99元/月
- 模型回答：2024 版 Pro 套餐的价格是 99 元/月（2023 版是 79 元/月）
- 判断：✅ 完全匹配（目标答案正确，额外对比信息不影响判断）

**示例 6：不匹配 - 版本号必须精确**
- 问题：当前 API 版本号是多少？
- 参考答案：v2.1.0
- 模型回答：版本号是 v2.0.0
- 判断：❌ 不匹配（版本号属于必须精确匹配的场景）

## 简洁度评估（独立维度）

**评估目标**：判断模型是否存在**冗余hack行为**（过度啰嗦、堆砌无关内容、答案埋藏在大段文字中，不直接）。

### ❌ 【不符合】(乘数=0) - 仅惩罚以下情况

出现以下**任一情况**视为hack：
1. **无关背景堆砌**：包含与答案无关的背景知识、原理科普、历史介绍等
   - 例：问"Pro版并发数？" 答"Pro版相比基础版在架构设计上采用了分布式方案，经过多次迭代优化，最大并发数为500..."
2. **过度铺垫客套**：重复用户问题、过度礼貌用语、无意义的开场白
   - 例："感谢您的提问，关于您问到的这个问题，让我来为您详细解答一下..."
3. **重复啰嗦表述**：用多句话反复表达同一个意思
   - 例："不可以。这是不允许的。系统不支持这个操作。"（三句表达同一意思）
4. **明显篇幅失衡**：当参考答案非常简短，用户问题可以直接简单回答时，模型回答不直接、包含上述冗余内容
   - 例：GT"陈长寿"(3字) vs 模型"建邺城有一位好心的医生陈长寿，他在游戏中..."(30+字)

### ✅ 【符合】(乘数=1.0) - 以下情况均可接受

- ✅ 比参考答案更简洁（直接答对核心问题即可）
  - GT："不可以，因为权限不足" → 模型："不可以" ✅
- ✅ 与参考答案篇幅相当
  - GT："500" → 模型："Pro版最大并发数为500" ✅
- ✅ 略长于参考答案，但确实是**必要信息**
  - GT："退出了" → 模型："退出了，冠军是陈芋汐" ✅（合理补充）
  - GT："建邺城陈长寿" → 模型："您说的建业城应该是建邺城，医生叫陈长寿" ✅（必要纠错）
- ✅ 参考答案本身有解释，模型只答核心结论
  - GT："不可以，因为系统限制每日最多3次" → 模型："不可以" ✅

**核心判断原则**：
- 不看字数比例，看**是否存在无效信息**
- 模型更简洁 = 好事，不扣分
- 模型稍长但信息有效 = 可接受
- 模型堆砌无关内容 = hack行为，扣分

**特殊说明**：
- 如果模型回答字数略多但确实是**必要的纠错或澄清**（如纠正用户问题中的错误实体名），仍判定为符合
- 对于枚举类问题，如果参考答案列举3项，模型列举3项+简短说明，可视为符合

**字数计算示例**：
```
参考答案："建邺城的陈长寿" (7字)
✅ 模型回答："建邺城的陈长寿医生" (9字) - 符合
✅ 模型回答："是建邺城的陈长寿" (8字) - 符合
❌ 模型回答："在梦幻西游游戏中，建邺城有一位好心的医生，他经常免费为玩家治疗，他的名字叫陈长寿" (38字) - 不符合
```

```
参考答案："Pro版最大并发数为500" (12字)
✅ 模型回答："500" (3字) - 符合
✅ 模型回答："Pro版的最大并发数是500个" (14字) - 符合
❌ 模型回答："根据您的问题，Pro版本相比基础版在性能上有显著提升，其最大并发数配置为500，这个数字是经过压力测试验证的..." (50+字) - 不符合
```

```
参考答案："退出了；冠军陈芋汐412.30分；亚军卢为384.00分" (26字)
✅ 模型回答："退出了，冠军是陈芋汐，得分412.30分" (18字) - 符合（核心信息完整）
❌ 模型回答："全红婵因伤退出了本次比赛，这对中国跳水队是一个遗憾。最终冠军由陈芋汐获得，她的表现非常出色，得分412.30分..." (55+字) - 不符合
```

## 输出格式

```json
{
  "judgment": "完全匹配/部分匹配/不匹配",
  "confidence": "高/中/低",
  "reason": "判断理由（说明核心问题是否正确回答，关键匹配点或不匹配原因）",
  "conciseness": "符合/不符合",
  "conciseness_reason": "简洁度判断理由（对比参考答案字数，说明是否存在冗余）"
}
```

---

现在请评估以下内容：

**用户历史会话**：${history}

**用户问题**：${question}

**参考答案**：
```
${gold_answer}
```

**模型回答**：
```
${model_response}
```

【模型回复协议说明】
模型答案中可能包含以下技术协议，评估时只关注协议标签中的回复内容即可：
- **模型回复协议**：`<result>回复内容...</result>`

请输出你的评估结果："""


async def llm_as_a_judge(prompt: str, traj_id: str = 'unknown') -> dict[str, Any]:
    """
    调用 LLM 进行评分。

    使用 nexus 的 ExecutionContext 获取 OpenAI 客户端。
    包含 rate limit 并发控制和总体超时保护，防止程序 hang 住。

    Args:
        prompt: 评分 prompt
        traj_id: 轨迹唯一标识，用于日志追踪

    Returns:
        包含评分结果的字典
    """
    from nexus.core import require_context

    judgment2score = {'完全匹配': 1.0, '部分匹配': 0.5, '不匹配': 0.0}
    # 简洁度映射：符合*1，不符合*0（对齐 baseline new_data_judge.py）
    conciseness2multiplier = {'符合': 1.0, '不符合': 0.0}

    score_dict: dict[str, Any] = {
        'closed_qa_score': 0.0,
        'judgment': '不匹配',
        'confidence': '低',
        'reason': '',
        'conciseness': '符合',
        'conciseness_reason': '',
    }
    last_exception: Exception | None = None

    messages = [{'role': 'user', 'content': prompt}]
    rate_limiter = get_rate_limiter('llm_judge', max_concurrent=RATE_LIMIT)
    rate_limit_wait_time = 0.0

    _log_reward(traj_id, 'LLM_JUDGE_START')

    for attempt in range(MAX_RETRIES):
        try:
            _log_reward(traj_id, 'LLM_JUDGE_ATTEMPT', f'attempt={attempt + 1}/{MAX_RETRIES}')

            # rate limit 在重试循环内部，重试等待期间不占用槽位
            rl_start = asyncio.get_event_loop().time()
            async with rate_limiter:
                rl_wait = asyncio.get_event_loop().time() - rl_start
                rate_limit_wait_time += rl_wait
                if rl_wait > 10:
                    _log_reward(traj_id, 'RATE_LIMIT_WAIT', f'{rl_wait:.2f}s')
                ctx = require_context()
                client = ctx.get_openai_client('judge')

                async with async_timeout(SINGLE_ATTEMPT_TIMEOUT):
                    # 使用与 new_data_judge.py 一致的参数
                    # max_retries=1: 不在 TaijiClient 内部重试，由外层控制重试逻辑
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
                        'LLM_JUDGE_CALL',
                        f'elapsed={_elapsed:.2f}s completion_tokens={_completion_tokens} thinking_tokens={_thinking_tokens}',
                    )

            result_text = response.choices[0].message.content

            if not result_text:
                logger.debug(
                    f'[{traj_id}] Closed QA judge attempt {attempt + 1}/{MAX_RETRIES}: empty response from LLM'
                )
                last_exception = RuntimeError('empty response from LLM')
                await retry_backoff_sleep(attempt)
                continue

            # 尝试提取 JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            json_content = json_match.group(1).strip() if json_match else result_text

            data = json.loads(json_content)

            judgment = data.get('judgment', '不匹配')
            confidence = data.get('confidence', '低')
            reason = data.get('reason', '')
            conciseness = data.get('conciseness', '符合')
            conciseness_reason = data.get('conciseness_reason', '')

            # 计算基础分数
            base_score = judgment2score.get(judgment, 0.0)
            # 应用简洁度乘数：符合*1，不符合*0（对齐 baseline）
            conciseness2multiplier.get(conciseness, 1.0)
            # final_score = base_score * conciseness_multiplier
            # 去除简洁度
            final_score = base_score

            score_dict['closed_qa_score'] = final_score
            score_dict['judgment'] = judgment
            score_dict['confidence'] = confidence
            score_dict['reason'] = reason
            score_dict['conciseness'] = conciseness
            score_dict['conciseness_reason'] = conciseness_reason
            score_dict['reward_rate_limit_wait_time'] = rate_limit_wait_time

            _log_reward(traj_id, 'LLM_JUDGE_DONE', f'judgment={judgment}')
            return score_dict

        except json.JSONDecodeError as e:
            logger.debug(f'[{traj_id}] Closed QA judge attempt {attempt + 1}/{MAX_RETRIES} JSON decode error: {e}')
            last_exception = e
            await retry_backoff_sleep(attempt)
        except TimeoutError:
            logger.debug(
                f'[{traj_id}] Closed QA judge attempt {attempt + 1}/{MAX_RETRIES} timed out after {SINGLE_ATTEMPT_TIMEOUT}s'
            )
            last_exception = TimeoutError(f'single attempt timeout after {SINGLE_ATTEMPT_TIMEOUT}s')
            await retry_backoff_sleep(attempt)
        except Exception as e:
            logger.debug(
                f'[{traj_id}] Closed QA judge attempt {attempt + 1}/{MAX_RETRIES} exception: {type(e).__name__}: {e}'
            )
            last_exception = e
            await retry_backoff_sleep(attempt)

    _log_reward(traj_id, 'LLM_JUDGE_FAILED', f'max_retries={MAX_RETRIES}')
    logger.error(
        f'[{traj_id}] LLM as a judge failed after {MAX_RETRIES} retries: '
        f'{type(last_exception).__name__ if last_exception else "RuntimeError"}: {last_exception}'
    )
    if last_exception is not None:
        raise last_exception
    raise RuntimeError(f'LLM as a judge failed after {MAX_RETRIES} retries')


async def compute_score(
    solution_str: str,
    ground_truth: str = '无参考答案',
    extra_info: dict | None = None,
) -> dict[str, Any]:
    """
    计算 Closed QA 评分。

    Args:
        solution_str: 模型生成的完整回答
        ground_truth: 参考答案
        extra_info: 额外信息（包含 prompt, messages, traj_id 等）

    Returns:
        包含评分结果的字典
    """
    extra_info = extra_info or {}
    traj_id = extra_info.get('traj_id', 'unknown')

    # 设置默认模型类型
    extra_info['model_type'] = extra_info.get('model_type', 'ds')

    # 提取最终回答
    answer = extract_query_solution(solution_str, extra_info)

    # 获取问题
    query = extra_info.get('query', extra_info.get('question', ''))

    # 获取 prompt messages
    prompt_messages = extra_info.get('prompt', extra_info.get('messages', []))

    def has_content(obj):
        if obj is None:
            return False
        try:
            return len(obj) > 0
        except (TypeError, ValueError):
            return False

    # 转换 numpy array 为 list
    if hasattr(prompt_messages, 'tolist'):
        prompt_messages = prompt_messages.tolist()

    if not query and has_content(prompt_messages):
        query = extract_query_from_prompt(prompt_messages)

    # 提取对话历史
    history = extra_info.get('history', [])
    if hasattr(history, 'tolist'):
        history = history.tolist()

    if not has_content(history) and has_content(prompt_messages):
        history = extract_history_from_prompt(prompt_messages)

    history_str = json.dumps(history, ensure_ascii=False) if has_content(history) else '无历史对话'

    # 构建 prompt
    prompt = PROMPT_TEMPLATE.replace('${history}', history_str)
    prompt = prompt.replace('${question}', query)
    prompt = prompt.replace('${gold_answer}', ground_truth)
    prompt = prompt.replace('${model_response}', answer)

    # 调用 LLM 评分
    return await llm_as_a_judge(prompt, traj_id=traj_id)
