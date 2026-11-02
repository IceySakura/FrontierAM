"""
格式协议检查模块（LLM Judge）。

对齐 baseline: verl/utils/reward_score/agts_format_judge.py

使用 LLM 检查模型回答的格式协议和多语言混合问题：
- format_score: 格式协议检查（0 通过, -1 不通过）
- language_consistency_score: 多语言混合检查（0 通过, -1 不通过）
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

# 格式检查 prompt 模板（对齐 baseline agts_format_judge.py）
PROMPT_TEMPLATE = """
你是一个专业的内容质量检查助手。请检查以下回答内容是否存在**格式协议错误**和**多语言混合问题**。

## 一、格式协议检查标准

### ✅ 允许使用的标准协议（仅限以下两种）：

1. **多媒体引入**：
   ```
   <media_info>[{"type": "image/video", "description": "", "id": 1}, {"type": "image/video", "description": "", "id": 2},...]</media_info>
   ```

2. **网页引用**：
   ```
   [citation:数字]
   ```
   示例：`[citation:0]`、`[citation:1][citation:5]`

### ❌ 禁止使用的非标准格式：

- 错误的网页引用（除非论文等需求）：`（参考资料1）`、`(参考资料X)`、`【来源3】`
- 错误的图片/视频格式：`[image:X]`、`[video:X]`、`![图片](url)`
- 错误的引用格式：`[src:X]`、`[citation 1, 2, 6]`、`[ref:X]`
- 其他任何非标准格式

#### ⚠️ 多媒体内容规则：
- 搜索到、引用的多媒体内容禁止在正文中详细具体的描述
- **仅允许**概括性地提及"相关视频/图片素材"的整体主题，因为只为用户展现一个视频，所以禁止用"这些视频"、"几个视频"这样的表述。
- **绝对禁止**在正文中提及任何具体的图片或视频
- **绝对禁止**列举视频标题、时长、作者等具体信息
- **绝对禁止**描述"第一个视频展示了..."、"这段18秒的视频..."等具体指向
- **绝对禁止**为多媒体内容添加引用标记如[citation:X]

#### ❌ 多媒体内容处理示例：

##### 违规类型1：列举式介绍
❌ "以下是一些精选的视频片段：
- 第一个视频展示了...
- 第二个视频是关于..."

##### 违规类型2：具体信息描述
❌ "这段18秒的视频..."
❌ "视频1中可以看到...，视频5中..."

##### 违规类型3：引用多媒体
❌ "如视频[1]所示..."
❌ "根据上述图片..."

##### ✅ 正确处理方式：
- "相关素材展现了[主题]的多个方面"
- "视觉资料提供了直观的参考"
- 然后基于web_search的文字结果详细回答问题

---

## 二、多语言混合检查标准

### ❌ 需要标记的问题：

1. **整体语言切换**：
   - 用户使用中文提问，但回答主要使用其他语言（英文、日文等）

2. **无理由混合回复**：
   - 在普通表述中频繁切换语言
   - 示例：❌ "这个问题涉及...Let me check...我需要搜索..."
   - 示例：❌ "如果您有问题，please let me know，我会帮助你"
   - 示例：❌ "如果您有具体问题或需要帮助，請告訴我..."（简繁混合）

3. **中途无故切换**：
   - 开始用中文回答，中途突然切换为其他语言继续回答完整句子/段落

### ✅ 合理使用其他语言的情况（不应标记）：

1. **专有名词**：品牌名、产品名、公司名、人名、地名
   - 示例：✅ "Apple 公司推出了 iPhone 15"

2. **技术术语**：业界通用的英文术语
   - 示例：✅ "使用 API 接口"、"通过 Machine Learning 技术"

3. **引用原文**：直接引用英文资料、搜索结果
   - 示例：✅ "根据官网描述：'This feature allows...'"

4. **代码相关**：变量名、函数名、命令、代码片段
   - 示例：✅ "使用 `console.log()` 输出结果"

5. **系统协议**：前端约定的固定格式
   - 示例：✅ `<media_info>`、`[citation:1]`

---

## 三、检查输出格式

请按以下格式输出检查结果：

```json
{
    "format_check": {
        "analysis": "格式检查分析说明",
        "pass": true/false,
    },
    "language_check": {
        "analysis": "语言检查分析说明",
        "pass": true/false,
    },
}
```

---

## 待检查内容：
- 用户问题：%s

- 模型回复：
```
%s
```
"""


async def llm_as_a_judge(prompt: str, traj_id: str = 'unknown') -> dict[str, int | float]:
    """
    调用 LLM 进行格式检查。

    对齐 baseline agts_format_judge.py 的 llm_as_a_judge，但使用 nexus 的 ExecutionContext。

    Args:
        prompt: 格式检查 prompt
        traj_id: 轨迹唯一标识

    Returns:
        包含 format_score 和 language_consistency_score 的字典
    """
    from nexus.core import require_context

    score_dict: dict[str, int | float] = {
        'format_score': 0,
        'language_consistency_score': 0,
    }

    messages = [{'role': 'user', 'content': prompt}]
    rate_limiter = get_rate_limiter('format_judge', max_concurrent=RATE_LIMIT)
    rate_limit_wait_time = 0.0
    last_exception: Exception | None = None

    _log_reward(traj_id, 'FORMAT_JUDGE_START')

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
                logger.debug(f'[{traj_id}] Format judge attempt {attempt + 1}/{MAX_RETRIES}: empty response from LLM')
                last_exception = RuntimeError('empty response from LLM')
                await retry_backoff_sleep(attempt)
                continue

            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            json_content = json_match.group(1).strip() if json_match else result_text

            data = json.loads(json_content)

            format_check = data['format_check']['pass']
            language_check = data['language_check']['pass']

            if not format_check:
                score_dict['format_score'] = -1
            if not language_check:
                score_dict['language_consistency_score'] = -1
            score_dict['reward_rate_limit_wait_time'] = rate_limit_wait_time

            _log_reward(traj_id, 'FORMAT_JUDGE_DONE', f'format={format_check}, language={language_check}')
            return score_dict
        except TimeoutError:
            logger.debug(
                f'[{traj_id}] Format judge attempt {attempt + 1}/{MAX_RETRIES} timed out after {SINGLE_ATTEMPT_TIMEOUT}s'
            )
            last_exception = TimeoutError(f'single attempt timeout after {SINGLE_ATTEMPT_TIMEOUT}s')
            await retry_backoff_sleep(attempt)
        except Exception as e:
            logger.debug(
                f'[{traj_id}] Format judge attempt {attempt + 1}/{MAX_RETRIES} exception: {type(e).__name__}: {e}'
            )
            last_exception = e
            await retry_backoff_sleep(attempt)

    _log_reward(traj_id, 'FORMAT_JUDGE_FAILED', f'max_retries={MAX_RETRIES}')
    logger.error(
        f'[{traj_id}] Format judge failed after {MAX_RETRIES} retries: '
        f'{type(last_exception).__name__ if last_exception else "RuntimeError"}: {last_exception}'
    )
    raise RuntimeError(f'Format judge failed after {MAX_RETRIES} retries')


async def compute_score(
    solution_str: str,
    ground_truth: str = '暂无',
    extra_info: dict | None = None,
) -> dict[str, Any]:
    """
    计算格式协议检查分数。

    对齐 baseline agts_format_judge.py 的 compute_score。

    Args:
        solution_str: 模型生成的完整回复
        ground_truth: 参考答案（此处未使用）
        extra_info: 额外信息

    Returns:
        包含 format_score 和 language_consistency_score 的字典
    """
    extra_info = extra_info or {}
    traj_id = extra_info.get('traj_id', 'unknown')

    # 搜图场景跳过格式检查（对齐 baseline）
    image_class = extra_info.get('图片分类', '')
    if image_class == '搜图':
        return {
            'format_score': 0,
            'language_consistency_score': 0,
        }

    answer = extract_query_solution(solution_str, extra_info)
    query = extra_info.get('question', '')

    prompt = PROMPT_TEMPLATE % (query, answer)

    return await llm_as_a_judge(prompt, traj_id=traj_id)
