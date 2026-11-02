"""
逻辑一致性评估模块（LLM Judge）。

对齐 baseline: verl/utils/reward_score/agts_logical_consistency_judge.py

使用 LLM 评估模型回答的逻辑性：
- logic_score: 逻辑评分（0/0.5/1，对应原始 0/1/2 归一化）
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

# 逻辑评估 prompt 模板（对齐 baseline agts_logical_consistency_judge.py）
PROMPT_TEMPLATE = """
# 逻辑性评估系统提示词

你是一个专业的逻辑性评估专家，负责评估大语言模型回答的逻辑质量。请按照以下规则进行综合评估：

---

## 核心评估原则
**⚠️ 评估范围说明**：
- 聚焦形式而非事实：本评估关注答案的**逻辑结构、表达合理性和可操作性**，而非验证具体数据、事实、知识点的真实性或数值计算过程的准确性。即使答案中的某些信息可能不准确，只要其逻辑推理过程合理、表述一致、建议可行，就不应扣分。（看**大逻辑**是否通顺，不抠**小细节**是否精确。）
  **仅评估逻辑性**：推理严密性、前后一致性、概念准确性、论证完整性、因果关系正确性
  **不评估**：事实准确性、知识完整性、专业深度、数值计算正确性

**⭐ History的合理使用**：
- 首先判断History与当前Query的相关性
- **仅当History与Query相关时**，才需要结合上下文理解
- 如果History讨论的是完全不同的话题，应独立评估Summary
- 相关History的情况：Summary中的代词、论据、前提可能来自History

---

## 【评估维度与标准】

### 一、推理逻辑谬误

评估回答中是否存在逻辑谬误，推理链条是否严密可靠。

**评估要点**：
- 是否存在因果倒置、循环论证、偷换概念、以偏概全等谬误
- 推理链条是否完整，有无逻辑跳跃（注意，事实准确性 不需要检测是否有论据支撑）
- 结合History，论据起点是否合理

**常见问题示例**：
- ❌ 因果倒置：因为流鼻涕，所以感冒了
- ❌ 循环论证：这个产品好，因为它是个好产品
- ❌ 偷换概念：计算机很强大，所以人工智能会取代人类
- ❌ 以偏概全：我用过一次不好，所以这个品牌都不好

---

### 二、前后一致性

评估回答内部及与历史对话之间是否存在矛盾或不一致。

**评估要点**：
- Summary内部：开头、中间、结尾的观点是否一致
- Summary内部：数据、时间、数量等细节是否前后对应
- 与History：Summary是否与历史信息（观点/数据/结论）冲突
- 如有变化，是否明确说明修正原因

**常见问题示例**：
- ❌ Summary内部矛盾：需要押金...（前文）→ 完全免费（后文）； A比B高（前文） → A:5, B:9（后文）
- ❌ 数据不一致：开放到17:30（前文）→ 5点后关门（后文）
- ❌ 与History矛盾：History说不需要押金，Summary说需要100元押金，且并未澄清

---

### 三、概念使用准确性

评估专业概念的使用是否准确、清晰，概念关系是否正确。

**评估要点**：
- 关键概念的定义是否准确
- 概念之间的关系（包含/并列/对立）是否正确
- 是否有概念混淆、误用或交叉耦合
- 结合History，概念使用是否与之前保持一致

**常见问题示例**：
- ❌ 概念混淆：人工智能就是机器学习（ML是AI的子集，不能等同）
- ❌ 概念误用：区块链是一种虚拟货币（区块链是技术，比特币才是货币）
- ❌ 层次关系错误：深度学习包括机器学习（应该是ML包括DL）

---

### 四、论证完整性

评估论证链条是否完整，论据是否充分支撑结论。

**评估要点**：（注意，事实准确性 不需要检测是否有论据支撑）
- 结论是否有论据支撑
- 论据是否充分
- 论证链条是否完整，有无断裂
- 结合History，Summary省略的论据是否在History中已建立

**常见问题示例**：
- ❌ 论证断裂：这个东西很好，所以推荐（为什么好？缺失）
- ❌ 论据不足：推荐iPhone，因为性能好（性能好在哪里？）
- ❌ 论据无关：这个产品包装精美，所以质量好（包装与质量无必然联系）

---

### 五、因果关系正确性

评估因果关系的准确性，是否混淆相关性和因果性。

**评估要点**：
- "因为...所以..."的因果方向是否正确
- 是否混淆了相关性和因果性
- 因果链条是否清晰完整
- 结合History，因果推理起点是否来自History的结论

**常见问题示例**：
- ❌ 因果颠倒：因为成绩好，所以聪明（应该是聪明导致成绩好）
- ❌ 相关当因果：因为带书包，所以成绩好（只是相关，非因果）
- ❌ 因果链不清晰：红糖甜，甜不好，所以不能喝（中间环节不清楚）

---

### 六、回答实用性

评估回答是否完整解决用户问题，提出的方案是否严重违背基本常识（严重违背基本常识：新手直接进行开颅手术来学习医学）

**评估要点**：
- 关注方案的**整体逻辑结构**是否合理，是否回答了核心问题，而非细节的准确性
- 判断是否存在**严重的违背常识**，而非吹毛求疵
- 只要回答完整地解决用户问题，提出的方案没有严重违背基本常识。则认为回答实用性没有问题。

**常见问题示例**：
- ❌ 信息空泛：学好英语需要努力、坚持（没有具体方法）
- ❌ 缺少关键信息：需要身份证办理（缺少地点、时间、费用）
- ❌ 答非所问：Query问"怎么办"，Summary只说"是什么"

---

## 【评估流程】

### Step 1：问题理解
针对问题本身进行理解，确保不偏离需求。

### Step 2：History相关性判断
**这是关键步骤，决定后续是否需要结合History评估**
如果判断为**不相关**，则跳过History，独立评估Summary。
判断标准：
- ✅ **相关**：Query是延续之前话题、追问细节、指代History中的内容
- ❌ **不相关**：Query是全新话题、与History讨论的主题完全不同


### Step 3：Summary理解
1. 梳理推理过程：
- 描述推理的步骤，确保每一步都有明确的逻辑，说明如何从已知信息推导出解答。
- 梳理开头、中间、结尾的观点，对比是否有冲突
- 梳理回复中的数据、时间、数量、限制条件等细节，对比是否有冲突
2. 分析不同要素之间的关联性：
识别和分析回答中各个要素之间的关系，明确各个要素（如时间、数量、条件、结论等）之间的相互关系，明确这些关系是如何影响推理过程的。

### Step 4：基于上述信息，按六大维度评估Query对应的Summary：（如果History与当前Query相关，才需要结合上下文理解。如果History讨论的是完全不同的话题，应独立评估Summary）
识别该维度潜在的问题，并记录原因：
- ✅ 表现良好的维度
- ⚠️ 存在轻微问题的维度
- ❌ 存在明显问题的维度

### Step 5：综合评价
- 汇总各维度的评估结果
- 总结主要优点和不足
- 计算最终总分
  **2分标准**（表现良好）:
  > 回答表现优秀，最多1-2个非常轻微的小瑕疵

  **1分标准**（存在轻微问题）:
  > 回答基本合格，但有2-3个轻微问题
  > 整体可用，但有改进空间

  **0分标准**（较差）:
  > 回答存在严重问题，或同时存在多个中度问题
  > 答案价值严重受损

---

## 【输出格式要求】

请严格按照以下JSON格式输出评估结果：

```json
{
    "推理逻辑谬误": {
        "原因": "<具体分析推理是否严密，有无逻辑谬误>",
        "档位": "表现良好/存在轻微问题/较差",
    },
    "前后一致性": {
        "原因": "<具体分析Summary内部及与History的一致性>",
        "档位": "表现良好/存在轻微问题/较差",
    },
    "概念使用准确性": {
        "原因": "<具体分析概念使用是否准确>",
        "档位": "表现良好/存在轻微问题/较差",
    },
    "论证完整性": {
        "原因": "<具体分析论证链条是否完整>",
        "档位": "表现良好/存在轻微问题/较差",
    },
    "因果关系正确性": {
        "原因": "<具体分析因果关系是否正确>",
        "档位": "表现良好/存在轻微问题/较差",
    },
    "回答实用性": {
        "原因": "<具体分析回答是否实用>",
        "档位": "表现良好/存在轻微问题/较差",
    },
    "overall_critique": "<综合评价，总结主要优点和不足，说明打分原因>",
    "overall_score": 2/1/0
}
```

---

# 开始评估

现在请基于以上标准，对提供的待验证答案进行全面评估。记住：
1. **必要时结合History理解**：如果History与当前Query相关，才需要结合上下文理解。如果History讨论的是完全不同的话题，应独立评估Summary
2. **独立评估各维度**：不要因为一个维度差就所有维度都打低分
3. **只看逻辑性**：不要因为事实错误就扣逻辑分（除非影响逻辑推理本身）
4. **原因要具体**：每个维度的原因要明确指出问题点或优点
5. **综合评价要全面**：总结所有维度的表现，说明主要优缺点

---

【评估材料】
**基础信息**
- 日期: %s
- 地点: %s

**对话上下文**
- 历史会话:
```
history = %s
```

- 当前问题: %s

**评估对象**
- 待验证答案:
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
- 这些协议是技术格式标记，不需要评估协议格式本身的相关性
- 默认模型答案中引用的内容（图片、视频、小程序、网页等）与问题相关
- 评估时应聚焦于文本回答内容的逻辑性相关维度检查
- 协议的部分的内容不会对用户直接展示，会有相关的渲染，不会影响阅读体验
- 除了上述协议外，无意义的乱码、重复等影响回复质量、逻辑性的情况需进行识别

【评估范围说明】
本评估聚焦于回复的**逻辑性**，而非验证具体事实内容的准确性。

**关于事实准确性的处理原则**：
- 你作为评估者，无法验证所有具体事实内容（如歌词、特定数据、引用内容等）的准确性
- 你的知识库可能已过时（截止于训练时间），而模型回答基于实时搜索
- 对于任何你无法当场100%%确认的具体信息（人名、事件、数据等），默认其基于搜索结果，是合理的
- **禁止使用你的知识库去"纠正"模型回答中的具体事实**
- **默认假定**：模型提供的具体信息是基于其训练数据的合理输出
- **仅在以下情况判定为事实错误**：
  * 明显违背常识（如"太阳从西边升起"）
  * 回答内自相矛盾
- **不应判定为错误的情况**：
  * 你无法独立验证但看起来合理的具体信息（歌词、专业数据、可能发生变化的新闻或事件等）
  * 引用协议标记的内容（这些内容已假定相关且准确）

请开始评估：
"""


async def llm_as_a_judge(prompt: str, traj_id: str = 'unknown') -> dict:
    """
    调用 LLM 进行逻辑一致性评估。

    对齐 baseline agts_logical_consistency_judge.py 的 llm_as_a_judge。

    Args:
        prompt: 评估 prompt
        traj_id: 轨迹唯一标识

    Returns:
        包含 logic_score 的字典
    """
    from nexus.core import require_context

    score_dict: dict[str, Any] = {
        'logic_score': 0,
    }

    messages = [{'role': 'user', 'content': prompt}]
    rate_limiter = get_rate_limiter('logic_judge', max_concurrent=RATE_LIMIT)
    rate_limit_wait_time = 0.0
    last_exception: Exception | None = None

    _log_reward(traj_id, 'LOGIC_JUDGE_START')

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
                logger.debug(f'[{traj_id}] Logic judge attempt {attempt + 1}/{MAX_RETRIES}: empty response from LLM')
                last_exception = RuntimeError('empty response from LLM')
                await retry_backoff_sleep(attempt)
                continue

            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            json_content = result_text if not json_match else json_match.group(1).strip()

            data = json.loads(json_content)

            # 校验分数范围（对齐 baseline）
            if float(data['overall_score']) > 2 or float(data['overall_score']) < 0:
                logger.debug(
                    f'[{traj_id}] Logic judge attempt {attempt + 1}/{MAX_RETRIES}: '
                    f'score out of range: {data["overall_score"]}'
                )
                last_exception = ValueError(f'score out of range: {data["overall_score"]}')
                await retry_backoff_sleep(attempt)
                continue

            # overall_score 为 0/1/2，归一化到 0/0.5/1
            final_score = float(data['overall_score']) / 2

            score_dict['logic_score'] = final_score
            for key in ['推理逻辑谬误', '前后一致性', '概念使用准确性', '论证完整性', '因果关系正确性', '回答实用性']:
                if key in data and isinstance(data[key], dict):
                    score_dict[key] = json.dumps(data[key], ensure_ascii=False)
                else:
                    score_dict[key] = str(data[key]) if key in data else '{}'

            score_dict['reward_rate_limit_wait_time'] = rate_limit_wait_time

            _log_reward(traj_id, 'LOGIC_JUDGE_DONE', f'score={final_score}')
            return score_dict
        except TimeoutError:
            logger.debug(
                f'[{traj_id}] Logic judge attempt {attempt + 1}/{MAX_RETRIES} timed out after {SINGLE_ATTEMPT_TIMEOUT}s'
            )
            last_exception = TimeoutError(f'single attempt timeout after {SINGLE_ATTEMPT_TIMEOUT}s')
            await retry_backoff_sleep(attempt)
        except Exception as e:
            logger.debug(
                f'[{traj_id}] Logic judge attempt {attempt + 1}/{MAX_RETRIES} exception: {type(e).__name__}: {e}'
            )
            last_exception = e
            await retry_backoff_sleep(attempt)

    _log_reward(traj_id, 'LOGIC_JUDGE_FAILED', f'max_retries={MAX_RETRIES}')
    logger.error(
        f'[{traj_id}] Logic judge failed after {MAX_RETRIES} retries: '
        f'{type(last_exception).__name__ if last_exception else "RuntimeError"}: {last_exception}'
    )
    raise RuntimeError(f'Logic judge failed after {MAX_RETRIES} retries')


async def compute_score(
    solution_str: str,
    ground_truth: str = '暂无',
    extra_info: dict | None = None,
) -> dict[str, Any]:
    """
    计算逻辑一致性评分。

    对齐 baseline agts_logical_consistency_judge.py 的 compute_score。

    Args:
        solution_str: 模型生成的完整回复
        ground_truth: 参考答案
        extra_info: 额外信息

    Returns:
        包含 logic_score 的字典
    """
    extra_info = extra_info or {}
    traj_id = extra_info.get('traj_id', 'unknown')

    # 搜图场景跳过逻辑检查（对齐 baseline）
    image_class = extra_info.get('图片分类', '')
    if image_class == '搜图':
        return {
            'logic_score': 0,
        }

    answer = extract_query_solution(solution_str, extra_info)

    current_datetime = extra_info.get('current_time', '')
    current_location = extra_info.get('current_location', '')
    query = extra_info.get('question', '')
    history = extra_info.get('history', [])

    if isinstance(history, list):
        history = json.dumps(history, ensure_ascii=False)

    prompt = PROMPT_TEMPLATE % (current_datetime, current_location, history, query, answer)

    return await llm_as_a_judge(prompt, traj_id=traj_id)
