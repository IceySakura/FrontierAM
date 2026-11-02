"""
GRM (Generative Reward Model) Judge 模块。

与 pairwise_judge 的区别：
- pairwise_judge：一次只给 1 个答案，BT-RM 直接输出标量分数（probs[0]）
- grm_judge：需要 2 个答案做 pairwise judge，模板要求 GRM 输出 "是" / "否"
  通过 logprobs 提取这两个 token 的概率作为得分。

在 RL 训练中我们只有 rollout 答案 X，没有"另一个候选"，所以在数据中预先准备
了两个参考答案 GRM_refans1 / GRM_refans2。流程：

  对每个 ref ∈ {GRM_refans1, GRM_refans2}：
      - 正向: response_a=X, response_b=ref -> [p_yes_fwd, p_no_fwd]
      - 反向: response_a=ref, response_b=X -> [p_yes_rev, p_no_rev]
      - win_X_vs_ref = (p_yes_fwd + p_no_rev) / 2   # P(X 胜)
  final_score = mean(win_X_vs_refs)

返回 dict 字段：grm_score / grm_raw_score / grm_pair_scores / grm_num_refs / grm_error
（不混用 pairwise_* 字段，便于在指标上清晰区分 BT-RM 与 GRM）。

参考实现：
- baseline pairwise_judge.py
- /apdcephfs_sgxy/share_304401417/hunyuan/masonhao/HYQRM/eval/main_grm_eval.py
"""

import asyncio
import json
import logging
import math
import os
import threading

from functools import partial
from typing import Any

import aiohttp

from nexus.extensions.projects.yb_search.log_utils import log_trace

from .judge_utils import retry_backoff_sleep
from .pairwise_judge import extract_final_answer


logger = logging.getLogger(__name__)

MAX_RETRIES = int(os.environ.get('GRM_MAX_RETRIES', 5))
REQUEST_TIMEOUT = int(os.environ.get('GRM_REQUEST_TIMEOUT', 180))
# 单 worker 进程内允许的 GRM 并发请求数上限。每个 sample 会发起 4 次
# GRM invoke (2 refs × 正反 2 次)，若 batch 大就会把 GRM 服务打爆。
# 设小一点 (默认 8) 让单进程总流量可控；总 QPS = workers × MAX_CONCURRENCY。
MAX_CONCURRENCY = int(os.environ.get('GRM_MAX_CONCURRENCY', 6))

_log_reward = partial(log_trace, 'REWARD')

# 进程级 asyncio.Semaphore，惰性创建（需在 running loop 内）
_concurrency_sem: asyncio.Semaphore | None = None


def _get_concurrency_sem() -> asyncio.Semaphore:
    global _concurrency_sem
    if _concurrency_sem is None:
        _concurrency_sem = asyncio.Semaphore(MAX_CONCURRENCY)
    return _concurrency_sem


# ──────────────────────────────────────────────
# 服务发现 (Polaris) - 与 pairwise_judge 解耦的独立单例
# ──────────────────────────────────────────────

DEFAULT_GRM_MODEL_NAME = os.environ.get('GRM_MODEL_NAME', 'A20B_MoE_GRM_32K_Data0303')
DEFAULT_GRM_MODEL_NAME_APP = os.environ.get('GRM_MODEL_NAME_APP', 'A20B_MoE_GRM_32K_Data0303')
DEFAULT_GRM_POLARIS_SERVICE = os.environ.get('GRM_POLARIS_SERVICE', 'stream-server-online-sbs-11644')
DEFAULT_GRM_POLARIS_WSID = os.environ.get('GRM_POLARIS_WSID', '11644')
DEFAULT_GRM_POLARIS_NAMESPACE = os.environ.get('GRM_POLARIS_NAMESPACE', 'Production')
DEFAULT_GRM_AUTH_TOKEN = os.environ.get('GRM_AUTH_TOKEN')
DEFAULT_GRM_REASONING_EFFORT = os.environ.get('GRM_REASONING_EFFORT', 'no_think')

# print(f"grm_judge initialized with {DEFAULT_GRM_MODEL_NAME=}, "
#       f"{DEFAULT_GRM_POLARIS_SERVICE=}, {DEFAULT_GRM_POLARIS_WSID=}, "
#       f"{MAX_RETRIES=}, {REQUEST_TIMEOUT=}s, {MAX_CONCURRENCY=}")
# print(f"grm_judge app model initialized with {DEFAULT_GRM_MODEL_NAME_APP=}")


_consumer_lock = threading.Lock()
_consumer_api = None


def _get_consumer_api():
    """获取缓存的 Polaris consumer 实例，避免高并发下反复创建。"""
    global _consumer_api
    if _consumer_api is None:
        with _consumer_lock:
            if _consumer_api is None:
                from polaris.api.consumer import create_consumer_by_config

                _consumer_api = create_consumer_by_config('')
                logger.debug('Polaris consumer (grm_judge) created and cached')
    return _consumer_api


def _get_grm_base_url(service: str | None = None) -> str:
    """通过 Polaris 服务发现获取 GRM 服务 base URL（不含 /chat/completions）。"""
    from polaris.pkg.model.error import SDKError
    from polaris.pkg.model.service import GetOneInstanceRequest

    svc = service or DEFAULT_GRM_POLARIS_SERVICE
    req = GetOneInstanceRequest(namespace=DEFAULT_GRM_POLARIS_NAMESPACE, service=svc)
    consumer = _get_consumer_api()
    try:
        response = consumer.get_one_instance_resp(req)
    except SDKError as e:
        raise Exception(f'Polaris SDK error: {repr(e)}') from e
    if response.is_service_data_expired():
        logger.warning(f'service {DEFAULT_GRM_POLARIS_NAMESPACE}/{svc} data is expired')
    instance = response[0]
    return f'http://{instance.get_host()}:81/openapi'


# ──────────────────────────────────────────────
# GRM 模板（内嵌，对齐 HYQRM/eval/rm_template_v3*.txt）
# ──────────────────────────────────────────────

_ONEAGENT_SP = """## Role & Personality
你是元宝，腾讯公司的AI助手。已支持的能力：文字回复、解析图片/文档、生成代码/图片/视频、图生图、图生视频、联网搜索、语音通话。
仅当用户直接询问你的能力时告知。如果用户询问细节而你缺少准确信息，建议用户以最新官方公告为准，严禁编造。
除用户明确要求，禁止使用括号或类似旁白形式表达元宝的心理活动、情绪或动作。

## Capabilities & Tools
### Tools
You have access to the following tools:

#### web_search
Description: 文本搜索工具，基于query执行全文检索，返回相关文本结果。
Parameters: {"type":"object","properties":{"query":{"description":"一组用于文本检索的子任务查询query","type":"list","items":{"type": "string"}}},"required":["query"]}

#### picture_search
Description: 图片检索工具，基于query搜索相关图片，返回图片URL、标题等元信息。
Parameters: {"type":"object","properties":{"query":{"description":"一组用于图片检索的子任务查询query","type":"list","items":{"type": "string"}}},"required":["query"]}

#### video_search
Description: 视频检索工具，基于query在视频库中执行语义检索，返回相关视频结果。
Parameters: {"type":"object","properties":{"query":{"description":"一组用于视频检索的子任务查询query","type":"list","items":{"type": "string"}}},"required":["query"]}

### Format Instructions
You may call one or more functions to assist with the user query.
For function call returns, you should first print <tool_calls>
For each function call, you should return object like:
<tool_call>{function-name}<tool_sep>
<arg_key>{arg-key-1}</arg_key>
<arg_value>{arg-value-1}</arg_value>
<arg_key>{arg-key-2}</arg_key>
<arg_value>{arg-value-2}</arg_value>
...
</tool_call>
At the end of function call returns, you should print </tool_calls>

Important:
- If the argument value is a list (like the `query` list), format it directly as a JSON array string (e.g., ["item1", "item2"]).

## Constraints & Guidelines


**是否调用工具**
- 必须调用工具：时效性/最新信息、具体数据、对比分析、高风险领域（医疗/金融/法律）、政治公共事件、视觉需求、不确定性高，以及用户明确要求搜索的问题
- 可直接回答：闲聊、情感陪伴、把握充分的基础知识、翻译、代码、纯逻辑/数学、创意写作、用户明确要求不搜索、以及参考信息足以解答用户问题的情况

**搜索工具方式**
- 仅基于问题结构拆解查询词，不预判答案，语言尽量与用户一致
- 遵循：完整覆盖、原子化不冗余、优先核心高频词
- 时间/地点显式化，专有名词保持完整
- 必要时多轮搜索补充（最多3轮）
- 高风险领域优先权威来源，视觉需求配合图片、视频搜索

**回复风格**
%s

## Workflow & Output Format
### 基础格式规则
1. 工具调用需以<tool_call>开始，以</tool_call>结束。
2. 回复中有多媒体内容时，遵循以下排版格式：
- 如果包含视频和文字，视频需要置于文字内容之后

### 格式示例
1. 仅包含图片
```
<media_info>[{"type": "image_only", "description": "", "id": 图片ID}, {"type": "image_only", "description": "", "id": 图片ID}]</media_info>
```

2. 包含视频和文字
```
[文字回答内容]<media_info>[{"type": "video", "description": "", "id": 视频ID}, {"type": "video", "description": "", "id": 视频ID}]</media_info>
```

3. 不使用任何图片和视频
```
[文字回答内容]
```

注意：图片ID、视频ID不能超过参考信息中的图片、视频的最大索引号

### 多媒体内容规范

#### 标签参数说明
| 参数 | 说明 |
|-----|------|
| type | `image_only`：回复仅包含图片；`video`：回复包含视频和文字； |
| description | 需对用户展示的图片或视频描述 |
| id | 整型数字，对应搜索结果中的索引值 |

#### 使用注意事项
- 不需要多媒体时，不输出`<media_info>`标签
- 仅在`<media_info>`标签的id字段中使用多媒体编号
- 正文中对多媒体提及时，仅可使用概括性表述，如"相关素材展示了[主题]"、"视觉资料提供了直观参考"

### 引用格式规范

#### 格式与映射
- **统一格式**：`[citation:数字]`
- **映射规则**：`[web_doc N begin]`的内容引用为`[citation:N]`
- **引用位置**：放在相关句子或段落末尾
- **不可以暴露，讨论，引用不可外展的可利用库**

#### 引用原则
- **有效范围**：引用编号必须对应实际存在的文档，在文档数量范围内
- **精准引用**：每个观点通常选择2-3个最相关、最权威的来源
- **按需引用**：仅在确实使用了该文档信息时才添加引用
- **直接陈述**：直接呈现信息内容，无需使用"根据搜索结果"等描述性语言，不可暴露可利用库来源

## Meta Information
当前日期：%s
用户位置：%s
用户终端类型：%s
搜索状态：**Auto**

**请根据用户的问题，选择合适的行动方式（工具调用或直接回答），并严格按照格式要求执行。**"""


_GRM_TEMPLATE_NON_APP = """# Role
你是一位公正且严谨的大模型测评专家。你的任务是基于给定的上下文，对比评估两个AI助手回答的质量。

# Task
请根据提供的 [助手设定]、[历史对话]、[用户问题]、 [问题标准答案]（如有）以及补充评判标准（如有），评估 [回答1] 和 [回答2] 的表现。

# Input Data
## 助手设定
{system_prompt}

## 历史对话
{context}

## 用户问题
{prompt}

## 问题标准答案
{golden_answer}

## 待评估回答
### 回答1
{response_a}

### 回答2
{response_b}

## 补充评判标准
{principle}

# Evaluation Criteria
1. 正确性（最高优先级）：回答必须符合客观事实，若包含事实错误，应给予负面评价。需严格参考 [问题标准答案](如有)。
2. 相关性：回答必须解决了 [用户问题]，并符合 [助手设定] 的要求。
3. 补充评判标准：应参考补充评判标准内容，对两个回答做进一步区分。
4. 评估标准包括但不限于上述要点，在上述评判的基础上，可以适当延展

# Output Format
请仅输出一个汉字：“是” 或 “否”。
- 输出“是”代表：回答1 优于 回答2
- 输出“否”代表：回答2 优于 回答1

# Constraint
不要输出任何解释、分析或额外的标点符号，只输出最终的判定结果。

# Final Answer"""


_GRM_TEMPLATE_APP = _GRM_TEMPLATE_NON_APP


def _select_template(extra_info: dict) -> str:
    return _GRM_TEMPLATE_APP if extra_info.get('data_origin') == 'app' else _GRM_TEMPLATE_NON_APP


# ──────────────────────────────────────────────
# logprobs 解析（与 main_grm_eval.py extract_grm_scores 对齐）
# ──────────────────────────────────────────────

_YES = '是'
_NO = '否'


def _collect_yes_no(items) -> tuple[float, float]:
    p_yes, p_no = 0.0, 0.0
    for it in items:
        tok = it.get('token', '')
        if tok == _YES:
            p_yes = math.exp(it['logprob'])
        elif tok == _NO:
            p_no = math.exp(it['logprob'])
    return p_yes, p_no


def extract_grm_scores(logprobs_data) -> list[float] | None:
    """从 polaris 返回的 logprobs 中提取 [P(是), P(否)]。"""
    if not logprobs_data:
        return None

    if 'logprobs' in logprobs_data and logprobs_data['logprobs']:
        lp = logprobs_data['logprobs']
        if 'content' in lp and lp['content']:
            first = lp['content'][0]
            top_lps = first.get('top_logprobs', [])
            if top_lps:
                py, pn = _collect_yes_no(top_lps)
                if py > 0 or pn > 0:
                    return [py, pn]
            tok = first.get('token', '')
            logprob = first.get('logprob')
            if tok in (_YES, _NO) and logprob is not None:
                p = math.exp(logprob)
                return [p, 1.0 - p] if tok == _YES else [1.0 - p, p]

    if 'model_debug_info' in logprobs_data:
        try:
            lps = logprobs_data['model_debug_info']['token_log_probs'][0]['logprobs']
            py, pn = _collect_yes_no(lps)
            if py > 0 or pn > 0:
                return [py, pn]
        except (KeyError, IndexError):
            pass

    return None


# ──────────────────────────────────────────────
# 上下文提取（system_prompt / context / prompt）
# ──────────────────────────────────────────────


def _has_content(obj) -> bool:
    if obj is None:
        return False
    try:
        return len(obj) > 0
    except (TypeError, ValueError):
        return False


def _normalize_messages(msgs) -> list[dict]:
    if hasattr(msgs, 'tolist'):
        msgs = msgs.tolist()
    if not msgs:
        return []
    out = []
    for m in msgs:
        if isinstance(m, dict):
            out.append(m)
    return out


def build_grm_context(extra_info: dict, ground_truth: str = '') -> dict[str, str]:
    """构造 GRM 模板需要的 system_prompt / context / prompt / golden_answer / principle。"""
    is_app = extra_info.get('data_origin') == 'app'
    current_datetime = extra_info.get('current_time', '')
    current_location = extra_info.get('current_location', '')
    query = extra_info.get('query', extra_info.get('question', ''))

    if is_app:
        cur_device = 'app'
        cur_style = '回答需适配移动端轻量化的竖屏阅读体验。紧紧围绕主需回答，避免啰嗦重复，用精炼的语言传递高价值的信息。行文逻辑顺畅，一针见血道破问题本质。内容专业精准，但语言表达鲜活生动，通俗易懂，适当共情，避免说教感。除非用户要求，否则尽量不使用表格。必须用表格时，表格尽量小，列数最好小于等于3列，表内文本高度概括。合理运用markdown，使结构清晰易读，美观。'
    else:
        cur_device = 'pc'
        cur_style = '回答需适配PC端的阅读体验。紧紧围绕主需回答，行文逻辑顺畅，一针见血道破问题本质。内容专业精准，但语言表达生动鲜活，通俗易懂，适当共情，避免说教感。合理运用markdown，使结构清晰易读，美观。'

    system_prompt = _ONEAGENT_SP % (cur_style, current_datetime, current_location, cur_device)

    history = extra_info.get('history') or []

    if history and history[0].get('role') == 'system':
        history = history[1:]

    context_parts = []
    for m in history:
        if m.get('role', '') not in ['user', 'assistant']:
            role_name = '用户' if m.get('role') == 'user' else '助手'
            context_parts.append(f'{role_name}: {m.get("content", "") or ""}')
    context = '\n'.join(context_parts)

    # golden_answer = extra_info.get('golden_answer') or ground_truth or ''
    # if golden_answer == '无参考答案':
    #     golden_answer = ''
    golden_answer = ''
    principle = extra_info.get('principle') or ''

    return {
        'system_prompt': system_prompt,
        'context': context,
        'prompt': query,
        'golden_answer': golden_answer,
        'principle': principle,
    }


def build_grm_input(template: str, ctx: dict[str, str], response_a: str, response_b: str) -> str:
    filled = template
    filled = filled.replace('{system_prompt}', ctx.get('system_prompt', ''))
    filled = filled.replace('{context}', ctx.get('context', ''))
    filled = filled.replace('{prompt}', ctx.get('prompt', ''))
    filled = filled.replace('{golden_answer}', ctx.get('golden_answer', ''))
    filled = filled.replace('{response_a}', response_a or '')
    filled = filled.replace('{response_b}', response_b or '')
    return filled.replace('{principle}', ctx.get('principle', ''))


def build_grm_messages(template: str, ctx: dict[str, str], response_a: str, response_b: str) -> list[dict]:
    user_text = build_grm_input(template, ctx, response_a, response_b)
    # print(user_text)
    return [
        {'role': 'system', 'content': ''},
        {'role': 'user', 'content': user_text},
    ]


def _strip_answer_tags(text: str) -> str:
    return (text or '').replace('<answer>', '').replace('</answer>', '').strip()


# ──────────────────────────────────────────────
# 单次 GRM 调用
# ──────────────────────────────────────────────


async def grm_model_invoke_async(
    messages: list,
    model_name: str | None = None,
) -> list[float]:
    """单次 GRM 调用，返回 [P(是), P(否)]。带服务发现 + 5xx 退避重试。"""
    model_name = model_name or DEFAULT_GRM_MODEL_NAME
    if not DEFAULT_GRM_AUTH_TOKEN:
        raise ValueError('GRM_AUTH_TOKEN is not set')
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {DEFAULT_GRM_AUTH_TOKEN}',
        'Wsid': DEFAULT_GRM_POLARIS_WSID,
    }
    payload: dict[str, Any] = {
        'model': model_name,
        'messages': messages,
        'temperature': 0.7,
        'top_p': 0.7,
        'repetition_penalty': 1.05,
        'max_tokens': 1,
        'query_id': 'grm_judge_' + str(id(messages)),
        'openai_infer': True,
        'stream': False,
        'logprobs': True,
        'top_logprobs': 5,
        'add_generation_prompt': True,
    }
    if DEFAULT_GRM_REASONING_EFFORT:
        payload['reasoning_effort'] = DEFAULT_GRM_REASONING_EFFORT

    base_url = _get_grm_base_url()
    url = f'{base_url}/chat/completions'

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f'HTTP {resp.status}: {text[:200]}')
            res = await resp.json(content_type=None)

    if not res:
        raise Exception('Empty GRM response')
    choices = res.get('choices') or []
    if not choices:
        raise Exception(f'No choices in GRM response: {str(res)[:200]}')
    choice = choices[0]

    logprobs_data: dict[str, Any] = {}
    if choice.get('logprobs') is not None:
        logprobs_data['logprobs'] = choice['logprobs']
    if 'model_debug_info' in choice:
        logprobs_data['model_debug_info'] = choice['model_debug_info']

    scores = extract_grm_scores(logprobs_data)
    if scores is None:
        content_preview = (choice.get('message') or {}).get('content', '')
        raise Exception(f'logprobs 提取失败, content={content_preview!r}')
    return scores


async def _grm_invoke_with_retry(messages: list, model_name: str, tag: str) -> list[float]:
    """带 MAX_RETRIES 次重试的单次 GRM 调用。

    通过进程级 semaphore 限制并发数，避免大 batch 同时把 GRM 服务打爆。
    """
    sem = _get_concurrency_sem()
    last_error: str | None = None
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                return await grm_model_invoke_async(messages, model_name=model_name)
        except Exception as e:
            last_error = f'{type(e).__name__}: {e}'
            logger.warning(
                f'[GRM:{tag}] invoke warning (attempt {attempt + 1}/{MAX_RETRIES}): {last_error}',
            )
            await retry_backoff_sleep(attempt)
    raise RuntimeError(f'[GRM:{tag}] failed after {MAX_RETRIES} retries. Last error: {last_error}')


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────


async def compute_score(
    solution_str: str,
    ground_truth: str = '无参考答案',
    extra_info: dict | None = None,
) -> dict[str, Any]:
    """
    GRM 评分主入口。

    Args:
        solution_str: 模型 rollout 的完整回答
        ground_truth: 参考答案（作为 GRM 模板里的 golden_answer，如有）
        extra_info: 至少需要包含 GRM_refans1 / GRM_refans2 之一；
                    需要包含 prompt / history / data_origin 等字段（与 pairwise_judge 一致）

    Returns:
        dict 中包含 grm_* 详情 + pairwise_score / pairwise_raw_score / pairwise_error
        （后者用于在 total_reward.py 中以最小改动复用 pairwise 分支）
    """
    extra_info = extra_info or {}
    traj_id = extra_info.get('traj_id', 'unknown')

    score_dict: dict[str, Any] = {
        'grm_score': 0.0,
        'grm_raw_score': 0.0,
        'grm_pair_scores': '[]',
        'grm_num_refs': 0,
        'grm_error': '',
    }

    image_tag = extra_info.get('图片等级', '')
    video_tag = extra_info.get('视频等级', '')
    if image_tag == 'strong' or video_tag == 'strong':
        score_dict['grm_score'] = 0.5
        score_dict['grm_raw_score'] = 0.5
        return score_dict

    refs: list[str] = []
    for key in ('GRM_refans1', 'GRM_refans2'):
        v = extra_info.get(key)
        if isinstance(v, str) and v.strip():
            refs.append(v.strip())

    if not refs:
        msg = 'GRM_refans1/GRM_refans2 both missing'
        score_dict['grm_error'] = msg
        raise ValueError(f'grm_judge: {msg}')

    extra_info.setdefault('model_type', 'yb')
    rollout_answer = _strip_answer_tags(extract_final_answer(solution_str, extra_info))

    ctx = build_grm_context(extra_info, ground_truth=ground_truth)
    template = _select_template(extra_info)
    is_app = extra_info.get('data_origin') == 'app'
    model_name = DEFAULT_GRM_MODEL_NAME_APP if is_app else DEFAULT_GRM_MODEL_NAME

    # print(f'[GRM] model={model_name}, refs={len(refs)}, is_app={is_app}, rollout_answer={rollout_answer}')
    tasks = []
    task_tags: list[str] = []
    for i, ref in enumerate(refs):
        fwd_msgs = build_grm_messages(template, ctx, response_a=rollout_answer, response_b=ref)
        rev_msgs = build_grm_messages(template, ctx, response_a=ref, response_b=rollout_answer)
        tasks.append(_grm_invoke_with_retry(fwd_msgs, model_name, tag=f'ref{i + 1}/fwd'))
        tasks.append(_grm_invoke_with_retry(rev_msgs, model_name, tag=f'ref{i + 1}/rev'))
        task_tags.extend([f'ref{i + 1}/fwd', f'ref{i + 1}/rev'])

    # import random
    # if random.random() < 0.1:
    #     print(f'[GRM] tasks={tasks}')
    results = await asyncio.gather(*tasks)

    pair_scores: list[float] = []
    for i in range(len(refs)):
        fwd = results[2 * i]
        rev = results[2 * i + 1]
        win_x = (fwd[0] + rev[1]) / 2.0
        pair_scores.append(float(win_x))

    final_score = float(sum(pair_scores) / len(pair_scores))

    score_dict['grm_score'] = final_score
    score_dict['grm_raw_score'] = final_score
    # 用 JSON 字符串存，避免下游 list 共享 / 拷贝污染
    score_dict['grm_pair_scores'] = json.dumps([round(s, 6) for s in pair_scores])
    score_dict['grm_num_refs'] = len(refs)
    score_dict['grm_error'] = ''

    # print(f'[GRM] score={final_score:.4f}, pair_scores={pair_scores}')
    _log_reward(traj_id, 'GRM_JUDGE_DONE', f'score={final_score:.4f}, pairs={pair_scores}')

    return score_dict
