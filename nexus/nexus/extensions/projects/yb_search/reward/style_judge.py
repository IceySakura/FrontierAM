"""
风格评分模块。

对齐 baseline: verl/utils/reward_score/style_judge.py

使用风格评分模型评估回答风格：
- 评估维度：简单问题简单回答、复杂问题详细回答、答案前置、表格展示、资源查询
- 返回 0/1/2 的综合分，归一化到 0/0.5/1
"""

import json
import logging
import re

from functools import partial
from typing import Any

from nexus.extensions.projects.yb_search.log_utils import log_trace

from .judge_utils import extract_query_solution, retry_backoff_sleep


logger = logging.getLogger(__name__)

MAX_RETRIES = 10

_log_reward = partial(log_trace, 'REWARD')

# System Prompt（对齐 baseline style_judge.py）
STYLE_SYSTEM_PROMPT = """# 角色定义
你是一个专业的大模型评估工具，你的目标是参考**评估维度手册**、**问题query**、**历史对话**、**待验证答案**来写一个完整的评估。你需要分析答案在对应评估维度上的表现，并计算具体得分。

# 输出规范

请严格按照以下JSON格式输出评估结果，注意在问题原因中，字符串描述里面统一最多用一个单引号，不允许嵌套，方便json.loads()成字典格式：

```json
{
  "具体维度名称": 0/1/2
}
```

## 字段说明
- 字典的键必须使用评估维度手册中的标准名称
- 字典的值: 0/1/2 (必须是int格式)

-----

# 注意事项

## 评估范围说明
  - 答案中的"<media_info>...</media_info>"和"<think>...</think>"等XML标签格式内容**不参与评估**，这些是为多模态内容预留的占位符
  - 答案中的"[citation:...][available:...]"等引用标记**不参与评估**，这是答案参考引文的占位符，请不要与"便捷性-网页超链"和"便捷性-资源查询"维度混淆
  - 所有"<xxx>...</xxx>"格式的占位符都应忽略

## 评估宽松度说明
  - 评估时**不要过于严格**，答案能满足query的核心需求即可
  - 其他维度以此类推，评判标准应**严格度适中**，避免过度苛刻

-----

## 📊 本次Query的评估维度和评分细则

请严格依据以下针对当前问题的特定评估维度和详细的0/1/2分评判标准进行评估：

${agentic_search_tag}


-----

# 开始评估

现在请基于以上标准，对提供的待验证答案进行评估。记住：

1.  **分析流程**：
    - 仔细阅读问题query和历史对话
    - 阅读待验证答案
    - 逐一判断每个评估维度是否适用
    - 对每个适用的维度，分析答案的表现并确定分数(0/1/2)
    - 为每个维度撰写详细的评分原因，必须引用答案中的具体内容

2.  **评估原则**：
    - 客观公正，有理有据
    - 严格按照评估维度手册中的标准打分
    - 保持适中的评判严格度，避免过度苛刻

3.  **输出要求**：
    - 严格按照规定的JSON格式输出
    - 确保JSON格式正确，可被json.loads()解析
"""

# 维度描述映射（对齐 baseline style_judge.py）
DIMENSION_DESCRIPTIONS = {
    '丰富度-简单问题简单回答': """# 【维度：简单问题简单回答】

## 一、适用场景
1. **简单封闭事实查询**：需求边界清晰、信息目标单一、推理层级浅、答案可直接验证
2. **简单罗列/宽泛推荐**：预期回复为罗列项/推荐项+简单介绍
3. **简单词语解释**：词义/读音/词性查询
4. **查询诗词/文章原文**：无赏析需求，原文篇幅不计入扣分字数
5. **多个简单问题**：一个prompt中提出多个简单问题

## 二、预期风格
核心信息突出，紧扣主需，做有限扩散和延伸，在丰富度上做一定克制，整体清爽、高效、实用。

## 三、评分标准

### 【2分】详略程度合理，符合简洁回答预期
**单个简单问题（类型1、3、4）**：字数≤350字
**简单罗列/宽泛推荐（类型2）**：罗列项≤5项，每项介绍≤70字
**多个简单问题（类型5）**：每个问题对应回答≤70字

### 【1分】有一定偏差，轻微影响简洁性
**单个简单问题**：字数350-500字
**简单罗列/宽泛推荐**：罗列项6-10项，或每项介绍70-100字
**多个简单问题**：每个问题对应回答70-100字

### 【0分】详略程度存在明显问题
**单个简单问题**：字数>500字
**简单罗列/宽泛推荐**：罗列项>10项，或每项介绍>100字
**多个简单问题**：每个问题对应回答>100字
""",
    '丰富度-复杂问题详细回答': """# 【维度：复杂问题详细回答】

## 一、适用场景

### 中篇幅类
1. **小百科式科普**：Query为纯实体名词
2. **新闻动态说明**：询问热点事件进展、人物动态

### 长篇幅类
1. **明确的深度需求**：Query中明确提出介绍、说明、分析等深度需求
2. **深度分析类**：开放问题、对比类、决策建议类
3. **专业权威类**：涉及医疗、金融、法律、教育等专业领域
4. **混合类问题**：一个prompt中同时包含简单问题和复杂问题

## 二、评分标准

### 【2分】详略程度合理，信息充分
**中篇幅类**：字数300-500字
**长篇幅类**：字数≥600字

### 【1分】详略程度有偏差
**中篇幅类**：字数150-300字或500-700字
**长篇幅类**：字数400-600字

### 【0分】严重问题
**中篇幅类**：<150字或>700字
**长篇幅类**：<400字
""",
    '便捷性-答案前置': """# 【维度：答案前置】

## 定义
主需对应的答案必须放在最前面，优先满足用户的核心需求。先回答主需，后进行分析、解释或延伸次需。
**注意：需要推理、计算才可以得出答案的问题，推理、计算过程也算为主需部分**

## 评分标准

### 【0分】主需答案未前置
- 次需信息在前
- 无答案信息的引导语超过100字
- 答案深埋在中间或后半部分
- 无关背景铺垫过长

### 【1分】主需答案相对靠前，但引导语偏长
- 无答案信息的引导语在70-100字之间

### 【2分】主需答案明确前置
- 前70字内进入主需答案内容
- 答案明确，结构合理
""",
    '便捷性-资源查询': """# 【维度：资源查询】

## 定义
当用户查询网站、资源、API接口、书籍下载、软件下载等可直接访问的资源时，答案中需要直接提供可用的URL链接。

## 评分标准

### 【0分】应提供链接但未提供
- 用户明确要求链接但未提供
- 仅提供网站名称或[citation:X]等引用标记

### 【2分】正确提供了直接可用的链接
- 提供了完整的URL（http://或https://开头）
""",
    '便捷性-表格展示': """# 【维度：表格展示】

## 评分标准
- **【2分】**：回答中包含Markdown表格
- **【0分】**：回答中不包含表格
""",
}


def normalize_style_score(raw_score: int) -> float:
    """将风格分数 0/1/2 归一化到 0/0.5/1"""
    if raw_score <= 0:
        return 0.0
    if raw_score == 1:
        return 0.5
    return 1.0


def calculate_overall_style_score(dim_scores: dict) -> int:
    """
    计算综合风格分。

    公式: 综合分 = min(min(P0维度原值, P1维度+1), 2)

    P0维度（原始分）：简单问题简单回答, 复杂问题详细回答, 答案前置
    P1维度（0分折算为1分）：表格展示, 资源查询

    特殊处理：简单问题简单回答、复杂问题详细回答 的1分按2分计算
    """
    p0_dims = ['简单问题简单回答', '复杂问题详细回答', '答案前置']
    p1_dims = ['表格展示', '资源查询']
    one_to_two_dims = ['简单问题简单回答', '复杂问题详细回答', '答案前置']

    scores = []

    for dim in p0_dims:
        if dim in dim_scores:
            score = dim_scores[dim]
            if dim in one_to_two_dims and score == 1:
                score = 2
            scores.append(score)

    for dim in p1_dims:
        if dim in dim_scores:
            scores.append(min(dim_scores[dim] + 1, 2))

    if not scores:
        return 1

    return min(scores)


def count_words(text: str) -> float:
    """统计字数（汉字 + 英文单词等效汉字数）"""
    stopwords = {'citation', 'available', 'media', 'info', 'layout', 'type'}

    clean_text = text
    clean_text = re.sub(r'\[citation:\d+\]', '', clean_text)
    clean_text = re.sub(r'<media_info>|</media_info>|<answer>|</answer>', '', clean_text)
    clean_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean_text)
    clean_text = re.sub(r'https?://[^\s\)>\]]+', '', clean_text)
    clean_text = re.sub(r'www\.[^\s\)>\]]+', '', clean_text)
    clean_text = re.sub(r'[`*_>#\-\[\]{}!|\\]', '', clean_text)

    chinese_chars = re.findall(r'[\u4e00-\u9fff]', clean_text)
    chinese_count = len(chinese_chars)

    english_words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", clean_text)
    english_words_filtered = [w for w in english_words if w.lower() not in stopwords]

    total_english_letters = sum(len(w) for w in english_words_filtered)
    space_count = max(0, len(english_words_filtered) - 1)
    english_equivalent_chinese = (total_english_letters + space_count) / 4

    punctuations = re.findall(r'[，。！？；：、' "''（）《》【】—…,.!?;:\"'()]", clean_text)
    punctuation_count = len(punctuations)

    return chinese_count + punctuation_count + english_equivalent_chinese


def has_content(obj):
    """安全检查对象是否有内容，兼容 list 和 numpy array"""
    if obj is None:
        return False
    try:
        return len(obj) > 0
    except (TypeError, ValueError):
        return False


def extract_query_from_prompt(prompt_messages):
    """从prompt messages中提取最后一个user的问题"""
    if not prompt_messages:
        return ''

    for msg in reversed(prompt_messages):
        if msg.get('role') == 'user':
            content = msg.get('content', '')
            if '# 用户消息为：' in content:
                parts = content.split('# 用户消息为：')
                if len(parts) > 1:
                    return parts[-1].strip()
            return content
    return ''


def extract_history_from_prompt(prompt_messages):
    """从prompt messages中提取对话历史（不包括system和最后一轮）"""
    if not prompt_messages:
        return []

    history = []
    for msg in prompt_messages:
        role = msg.get('role', '')
        if role == 'system':
            continue
        if role in ['user', 'assistant']:
            content = msg.get('content', '')
            if len(content) > 500:
                content = content[:500] + '...'
            history.append({'role': role, 'content': content})

    if history and history[-1].get('role') == 'user':
        history = history[:-1]

    return history


def build_style_messages(extra_info: dict, answer: str) -> list:
    """构建发送给风格评分模型的 messages"""
    prompt_messages = extra_info.get('prompt', extra_info.get('messages', []))
    if hasattr(prompt_messages, 'tolist'):
        prompt_messages = prompt_messages.tolist()

    query = extra_info.get('question', extra_info.get('query', ''))
    if not query and has_content(prompt_messages):
        query = extract_query_from_prompt(prompt_messages)

    history = extra_info.get('history', [])
    if hasattr(history, 'tolist'):
        history = history.tolist()

    if not has_content(history) and has_content(prompt_messages):
        history = extract_history_from_prompt(prompt_messages)

    if has_content(history):
        history = [{'role': msg.get('role', ''), 'content': msg.get('content', '')[:200]} for msg in history[-2:]]

    if not history or len(history) < 2:
        history_str = '无'
    else:
        history_str = (
            f'用户问题: {history[-2].get("content", "")}\n模型回答: {history[-1].get("content", "")[:200]} ...'
        )

    word_count = count_words(answer)
    word_count_str = f'\n    总字数（等效）: {word_count:.1f}\n    按照总字数参考标准维度表进行打分'

    active_descriptions = []

    is_simple = extra_info.get('简单问题简单回答')
    if is_simple == '是':
        active_descriptions.append(DIMENSION_DESCRIPTIONS['丰富度-简单问题简单回答'])
    elif is_simple == '否':
        active_descriptions.append(DIMENSION_DESCRIPTIONS['丰富度-复杂问题详细回答'])

    if extra_info.get('表格展示') == '是':
        active_descriptions.append(DIMENSION_DESCRIPTIONS['便捷性-表格展示'])

    if extra_info.get('答案前置') == '是':
        active_descriptions.append(DIMENSION_DESCRIPTIONS['便捷性-答案前置'])

    if extra_info.get('资源查询') == '是':
        active_descriptions.append(DIMENSION_DESCRIPTIONS['便捷性-资源查询'])

    tags_text = '\n'.join(active_descriptions) if active_descriptions else '无特定强制维度，请通用评估。'

    system_content = STYLE_SYSTEM_PROMPT.replace('${agentic_search_tag}', tags_text)

    user_content = (
        (
            '**输入信息如下组织**：\n\n'
            '  - 历史对话: ${agentic_search_history}\n\n'
            '  - 问题query: ${agentic_search_query}\n\n'
            '  - 待验证答案: ${agentic_search_answer}\n  \n'
            '  - 待验证答案字数统计: ${agentic_word_count}\n\n'
            '请开始你的评估：'
        )
        .replace('${agentic_search_history}', history_str)
        .replace('${agentic_search_query}', query)
        .replace('${agentic_search_answer}', str(answer))
        .replace('${agentic_word_count}', str(word_count_str))
    )

    return [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': user_content},
    ]


async def style_model_invoke_async(messages: list, traj_id: str = 'unknown') -> dict:
    """
    异步调用风格评分模型。

    使用 nexus 的 ExecutionContext 获取 OpenAI client，
    对齐 baseline style_judge.py 的 style_model_invoke_async。

    Returns:
        dict: 各维度的分数，如 {"简单问题简单回答": 2, "答案前置": 1}
    """
    from nexus.core import require_context

    ctx = require_context()
    # Fall back to 'judge' client if 'style_judge' is not configured separately
    try:
        client = ctx.get_openai_client('style_judge')
    except KeyError:
        client = ctx.get_openai_client('judge')

    response = await client.create_completion(
        messages=messages,
        max_tokens=10000,
        temperature=0.7,
        top_p=0.9,
        max_retries=1,
    )

    content = response.choices[0].message.content
    if not content:
        raise Exception('风格评分模型返回空内容')

    logger.debug(
        f'[{traj_id}] [STYLE_API] Raw response: {content[:500]}...'
        if len(content) > 500
        else f'[{traj_id}] [STYLE_API] Raw response: {content}'
    )

    json_content = content.split('```json')[-1].split('```')[0].split('</think>')[-1]
    dim_scores = json.loads(json_content)

    logger.debug(f'[{traj_id}] [STYLE_API] Parsed scores: {json.dumps(dim_scores, ensure_ascii=False)}')

    return dim_scores


async def compute_score(
    solution_str: str,
    ground_truth: str = '无参考答案',
    extra_info: dict | None = None,
) -> dict[str, Any]:
    """
    风格评分函数。

    对齐 baseline style_judge.py 的 compute_score。

    Args:
        solution_str: 模型生成的完整回答
        ground_truth: 参考答案（风格评分不需要）
        extra_info: 包含 prompt, messages, 以及维度开关

    Returns:
        score_dict: 包含风格评分的字典
    """
    extra_info = extra_info or {}
    traj_id = extra_info.get('traj_id', 'unknown')

    score_dict: dict[str, Any] = {
        'style_score': 0.5,
        'style_raw_score': 1,
        'style_dim_scores': {},
        'style_error': '',
    }

    last_error = None

    extra_info.setdefault('model_type', 'ds')
    answer = extract_query_solution(solution_str, extra_info)

    if not answer:
        logger.warning(f'[{traj_id}] style_judge: No answer found in solution_str')
        raise ValueError('style_judge: No answer found in solution_str')

    messages = build_style_messages(extra_info, answer)

    _log_reward(traj_id, 'STYLE_JUDGE_START')

    for attempt in range(MAX_RETRIES):
        try:
            dim_scores = await style_model_invoke_async(messages, traj_id=traj_id)

            overall_score = calculate_overall_style_score(dim_scores)
            normalized_score = normalize_style_score(overall_score)

            score_dict['style_dim_scores'] = dim_scores
            score_dict['style_raw_score'] = overall_score
            score_dict['style_score'] = normalized_score
            score_dict['style_error'] = ''

            _log_reward(traj_id, 'STYLE_JUDGE_DONE', f'score={normalized_score}')

            return score_dict
        except Exception as e:
            logger.debug(
                f'[{traj_id}] Style judge attempt {attempt + 1}/{MAX_RETRIES} exception: {type(e).__name__}: {e}'
            )
            last_error = str(e)
            await retry_backoff_sleep(attempt)

    _log_reward(traj_id, 'STYLE_JUDGE_FAILED', f'max_retries={MAX_RETRIES}')
    logger.error(f'[{traj_id}] Style model invoke failed after {MAX_RETRIES} retries: {last_error}')
    raise RuntimeError(last_error or f'Style model invoke failed after {MAX_RETRIES} retries')
