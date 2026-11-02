"""
相关性评估模块（LLM Judge）。

对齐 baseline: verl/utils/reward_score/agts_relevance_judge.py

使用 LLM 评估模型回答的相关性：
- relevance_score: 相关性评分（0/0.5/1）
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

from .judge_utils import extract_query_solution, retry_backoff_sleep


logger = logging.getLogger(__name__)

MAX_RETRIES = int(os.environ.get('TAIJI_JUDGE_MAX_RETRIES', 10))
SINGLE_ATTEMPT_TIMEOUT = int(os.environ.get('TAIJI_JUDGE_TIMEOUT_SINGLE', 200))
RATE_LIMIT = int(os.environ.get('TAIJI_JUDGE_RATE_LIMIT', 6))  # 最大并发数

_log_reward = partial(log_trace, 'REWARD')

# 相关性评估 prompt 模板（对齐 baseline agts_relevance_judge.py）
PROMPT_TEMPLATE = """你是一个专业的AI模型评估专家，需要对大模型生成的回复进行相关性的评估。请基于以下信息进行综合评估：

【评估维度与标准】

## 一、基础指令遵循

### 1.1 主需指令识别
评估模型是否准确理解并回答了用户的核心问题，有无答非所问。
- 优秀：完全理解主意图，精准回答核心问题
- 良好：基本理解主意图，回答有一定偏差但大方向正确
- 较差：未理解主意图，答非所问或理解错误

### 1.2 次需指令识别
评估模型是否遵循了字数、风格、输出格式、语气等约束要求。
- 优秀：完全遵循所有约束要求
- 良好：遵循了主要约束，个别约束有轻微偏差
- 较差：忽略重要约束或多项约束未遵循

### 1.3 模糊/错误意图理解
当用户问题模糊、有歧义或包含错误时，模型是否能够合理处理。
- 优秀：识别出模糊/错误点，主动澄清或给出合理解释
- 良好：做出了合理推测，但未明确指出模糊性
- 较差：未识别问题，回答可能不符合用户真实需求

## 二、基础回复合理性

### 2.1 信息完整性
评估回答是否存在明显遗漏的场景、维度或关键信息。

**多媒体需求特殊说明**：
- 当用户需求为图片、视频等多媒体内容时：
  * 若模型通过 `<media_info>` 协议提供了相应内容，即视为**核心信息已完整提供**
  * 文本部分无需详细描述多媒体内容本身（用户会看到渲染后的实际图片/视频）
  * **不应因"文本未描述图片细节"而扣分**

- 优秀：信息全面完整，覆盖所有重要维度
- 良好：信息基本完整，有轻微遗漏但不影响整体理解
- 较差：存在明显信息缺失，影响问题解决

### 2.2 内容重复性
评估单轮回答内是否存在明显的重复表达。
- 优秀：无重复，表达简洁高效
- 良好：有轻微重复但不影响阅读
- 较差：明显重复，冗余严重

### 2.3 内容价值
评估回答的实用性、深度和严谨性。
- 优秀：内容价值高，实用且有深度，表达严谨
- 良好：有一定价值，但深度或严谨性略有不足
- 较差：价值低，泛泛而谈或不严谨

### 2.4 时效性
评估回答的信息、数据、案例的时效性是否符合当前时间背景。
- 优秀：信息时效性强，符合当前时间语境
- 良好：时效性可接受，略有过时但不严重
- 较差：信息明显过时，违背时效常识

## 三、多轮效果

### 3.1 多轮记忆
评估是否记住上文的限定条件、背景、角色设定等关键信息。
- 优秀：完全记住上文关键信息，回答连贯一致
- 良好：记住主要信息，个别细节有遗漏
- 较差：遗忘重要上文信息，前后矛盾
- 单轮对话无历史记录，默认没问题

### 3.2 多轮混淆
评估是否受到无关上文的干扰，导致本轮回答出现问题。
- 优秀：不受无关信息干扰，焦点准确
- 良好：轻微受干扰但整体可控
- 较差：明显受无关信息干扰，回答偏离
- 单轮对话或上文无干扰信息，默认没问题

### 3.3 多轮重复
评估多轮间是否存在内容重复。
- 优秀：无跨轮重复，每轮都有新信息
- 良好：有少量重复但必要或篇幅小
- 较差：存在明显跨轮重复，篇幅大
- 单轮对话无法评估，默认没问题

【评估材料】
- 历史对话：
```
history = %s
```

- 当前时间：%s
- 当前地点：%s

- 当前用户问题：%s

- 参考答案/评估注意项：
```
%s
```

- 需评估的模型回复：
```
%s
```

（注意：上述的评估材料中因长度原因没有提供搜索到的网页、图片、视频等内容，会使用特殊协议进行渲染）

【模型回复协议说明】
模型答案中可能包含以下技术协议，评估时请注意：

1. **多媒体引入协议**：`<media_info>[{"type": "image_only/image/video", "description": "", "id": 1},...]</media_info>`
   - 该协议用于向用户展示图片、视频等多媒体内容
   - **协议本身即为有效回答**：当用户需求是图片、视频时，返回该协议即表示已提供相应内容
   - **文本描述非必需**：模型无需在文本中详细描述多媒体的具体内容，协议会直接渲染展示

2. **网页引用协议**：`[citation:网页数组下标]`,`[available:网页数组下标]`

**重要提示**：
- 这些协议是技术格式标记，**协议内容会被渲染为实际的多媒体/功能组件呈现给用户**
- 默认模型答案中引用的内容（图片、视频、网页等）与问题相关
- **对于图片/视频需求，协议的存在即表示已提供内容，不应要求文本重复描述**
- 评估时应聚焦于文本回答内容的相关性、完整性和价值
- 协议的部分内容不会对用户直接展示原始文本，会有相关的渲染，不会影响阅读体验

【评估范围说明】
本评估聚焦于回复的**相关性**，**不验证具体事实内容的准确性**。

**关于事实准确性的核心原则**：
- ⚠️ **你作为评估者，无权也无法验证模型回答中的具体事实是否准确**
- ⚠️ **模型回答已基于搜索结果生成，应假定其引用的信息、数据、案例等内容是合理的**
- ⚠️ **你的任务是评估"回答是否解决了用户问题"，而非"回答的内容是否真实存在"**
- **默认假定**：模型提供的具体信息是基于搜索结果的输出
- **仅在以下情况判定为事实错误**：
  * 明显违背常识（如"太阳从西边升起"）
  * 回答内自相矛盾
- **不应判定为错误的情况**：
  * 你无法独立验证但看起来合理的具体信息（歌词、专业数据、可能发生变化的新闻或事件等）
  * 引用协议标记的内容（这些内容已假定相关且准确）

**参考答案中提供事实准确性验证点的特殊处理**：
- ⚠️ **当参考答案中明确提供了【事实准确性验证点】时，这些事实点必须被验证**
- 这类问题通常为：明确的事实、数值、枚举、定义、历史事件等有标准答案的封闭域问题
- **验证要求**：
  * 模型回复中涉及的相关事实必须与给定验证点一致。
  * 如果模型回复与验证点冲突（非仅是未提及，而是明确说反），应判定为事实错误
  * 如果模型未提及某些验证点，不算事实错误，但可能影响"信息完整性"评分
- **判定影响**：事实错误会直接影响"主需指令识别"、"信息完整性"以及整体的评分

【评估流程】
在进行具体评分前，请先完成以下分析：

0. **评估角色确认**：
   - ⚠️ 你是"相关性评估专家"，不是"事实核查专家"
   - ⚠️ 你的任务是判断"回答是否满足用户需求"，不是"回答是否符合客观事实"
   - ⚠️ 你的知识库可能已过时（截止于训练时间），而模型回答基于实时搜索
   - ⚠️ 对于任何你无法当场100%%确认的具体信息（人名、事件、数据等），默认其基于搜索结果，是合理的
   - ⚠️ **禁止使用你的知识库去"纠正"模型回答中的具体事实**
   - 参考答案中是否包含【事实准确性验证点】部分？
   - 如果包含，逐条检查模型回复中的相关表述：
     * 是否与验证点明确冲突？（如地域错误、特征张冠李戴等）
     * 注意：未提及不算错，但明确说错了必须标记
   - 记录所有事实冲突点，这将直接影响"主需指令识别"和"信息完整性"的档位判定

1. **上下文依赖判断**（仅多轮对话需要）：
   - 当前用户问题是否依赖历史对话？（如包含代词、省略主语、追问等）
   - 如果依赖，准确理解需要哪些上下文信息？
   - 模型回答是否正确关联了这些上下文？

2. **真实意图识别**：
   - 结合历史对话（如有），用户的真实意图是什么？
   - 模型理解的意图与真实意图是否一致？

3. **回答有效性判断**：
   - 在明确真实意图后,模型的回答是否有效解决了用户问题？
   - 即使内容质量尚可，是否因为理解偏差而实际无用？

**重要原则**：在多轮对话中，如果模型因忽视历史上下文而理解错误，即使回答内容本身结构完整、表达流畅，也应在"基础指令遵循"和"基础回复合理性"、"总体等级"上都给予较差档位，因为这样的回答对用户毫无价值。

【输出格式要求】
请严格按照以下JSON格式输出评估结果：
```
{
    "基础指令遵循": {
        "主需指令识别": {
            "原因": "<具体分析>",
            "档位": "优秀/良好/较差",
        },
        "次需指令识别": {
            "原因": "<具体分析>",
            "档位": "优秀/良好/较差",
        },
        "模糊错误意图理解": {
            "原因": "<具体分析>",
            "档位": "优秀/良好/较差",
        },
    },
    "基础回复合理性": {
        "信息完整性": {
            "原因": "<具体分析>",
            "档位": "优秀/良好/较差",
        },
        "内容重复性": {
            "原因": "<具体分析>",
            "档位": "优秀/良好/较差",
        },
        "内容价值": {
            "原因": "<具体分析>",
            "档位": "优秀/良好/较差",
        },
        "时效性": {
            "原因": "<具体分析>",
            "档位": "优秀/良好/较差",
        },
    },
    "多轮效果": {
        "多轮记忆": {
            "原因": "<具体分析>",
            "档位": "优秀/良好/较差",
        },
        "多轮混淆": {
            "原因": "<具体分析>",
            "档位": "优秀/良好/较差",
        },
        "多轮重复": {
            "原因": "<具体分析>",
            "档位": "优秀/良好/较差",
        },
    },
    "overall_critique": "<综合评价，总结主要优点和不足>",
    "overall_grade": "优秀/良好/较差",
}
```
请开始评估。
"""


async def llm_as_a_judge(prompt: str, traj_id: str = 'unknown') -> dict:
    """
    调用 LLM 进行相关性评估。

    对齐 baseline agts_relevance_judge.py 的 llm_as_a_judge。

    Args:
        prompt: 评估 prompt
        traj_id: 轨迹唯一标识

    Returns:
        包含 relevance_score 的字典
    """
    from nexus.core import require_context

    grade2level = {'优秀': 1, '良好': 0.5, '较差': 0}

    score_dict: dict[str, Any] = {
        'relevance_score': 0,
        '基础指令遵循': '{}',
        '基础回复合理性': '{}',
        '多轮效果': '{}',
    }

    messages = [{'role': 'user', 'content': prompt}]
    rate_limiter = get_rate_limiter('relevance_judge', max_concurrent=RATE_LIMIT)
    rate_limit_wait_time = 0.0
    last_exception: Exception | None = None

    _log_reward(traj_id, 'RELEVANCE_JUDGE_START')

    for attempt in range(MAX_RETRIES):
        try:
            rl_start = asyncio.get_event_loop().time()
            async with rate_limiter:
                rl_wait = asyncio.get_event_loop().time() - rl_start
                rate_limit_wait_time += rl_wait
                if rl_wait > 10:
                    _log_reward(traj_id, 'RATE_LIMIT_WAIT', f'{rl_wait:.2f}s')
                ctx = require_context()
                client = ctx.get_openai_client('judge')

                async with async_timeout(SINGLE_ATTEMPT_TIMEOUT):
                    response = await client.create_completion(
                        messages=messages,
                        max_tokens=10000,
                        reasoning_effort='high',
                        repetition_penalty=1.05,
                        max_retries=1,
                        enable_thinking=True,
                    )

            result_text = response.choices[0].message.content
            if not result_text:
                logger.debug(
                    f'[{traj_id}] Relevance judge attempt {attempt + 1}/{MAX_RETRIES}: empty response from LLM'
                )
                last_exception = RuntimeError('empty response from LLM')
                await retry_backoff_sleep(attempt)
                continue

            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            json_content = result_text if not json_match else json_match.group(1).strip()

            data = json.loads(json_content)

            overall_grade = data['overall_grade']
            final_score = grade2level.get(overall_grade, 0)

            score_dict['relevance_score'] = float(final_score)
            for key in ['基础指令遵循', '基础回复合理性', '多轮效果']:
                if key in data and isinstance(data[key], dict):
                    score_dict[key] = json.dumps(data[key], ensure_ascii=False)
                else:
                    score_dict[key] = str(data[key]) if key in data else '{}'

            score_dict['reward_rate_limit_wait_time'] = rate_limit_wait_time

            _log_reward(traj_id, 'RELEVANCE_JUDGE_DONE', f'score={final_score}')
            return score_dict
        except TimeoutError:
            logger.debug(
                f'[{traj_id}] Relevance judge attempt {attempt + 1}/{MAX_RETRIES} timed out after {SINGLE_ATTEMPT_TIMEOUT}s'
            )
            last_exception = TimeoutError(f'single attempt timeout after {SINGLE_ATTEMPT_TIMEOUT}s')
            await retry_backoff_sleep(attempt)
        except Exception as e:
            logger.debug(
                f'[{traj_id}] Relevance judge attempt {attempt + 1}/{MAX_RETRIES} exception: {type(e).__name__}: {e}'
            )
            last_exception = e
            await retry_backoff_sleep(attempt)

    _log_reward(traj_id, 'RELEVANCE_JUDGE_FAILED', f'max_retries={MAX_RETRIES}')
    logger.error(
        f'[{traj_id}] Relevance judge failed after {MAX_RETRIES} retries: '
        f'{type(last_exception).__name__ if last_exception else "RuntimeError"}: {last_exception}'
    )
    raise RuntimeError(f'Relevance judge failed after {MAX_RETRIES} retries')


async def compute_score(
    solution_str: str,
    ground_truth: str = '暂无',
    extra_info: dict | None = None,
) -> dict[str, Any]:
    """
    计算相关性评分。

    对齐 baseline agts_relevance_judge.py 的 compute_score。

    Args:
        solution_str: 模型生成的完整回复
        ground_truth: 参考答案
        extra_info: 额外信息

    Returns:
        包含 relevance_score 的字典
    """
    extra_info = extra_info or {}
    traj_id = extra_info.get('traj_id', 'unknown')

    # 搜图场景跳过相关性检查（对齐 baseline）
    image_class = extra_info.get('图片分类', '')
    if image_class == '搜图':
        return {
            'relevance_score': 0,
        }

    answer = extract_query_solution(solution_str, extra_info)

    current_datetime = extra_info.get('current_time', '')
    current_location = extra_info.get('current_location', '')
    query = extra_info.get('question', '')
    history = extra_info.get('history', [])

    if isinstance(history, list):
        history = json.dumps(history, ensure_ascii=False)

    prompt = PROMPT_TEMPLATE % (history, current_datetime, current_location, query, ground_truth, answer)

    return await llm_as_a_judge(prompt, traj_id=traj_id)
