"""Prompt-side helpers for the ``yb_vlm_oneagent`` task."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from nexus.extensions.projects.yb_vlm_oneagent.context_manage import (
    ContextBudget,
    build_managed_messages,
)


# Keep the long natural-language prompt in the hytuner checkout out of the
# task import path by defining the runtime prompt here, inside Nexus.
SP_REPLACE_MARKER = '__YB_VLM_SYSTEM_PROMPT__'

_SP_V8_CLEAN = """# AI助手系统

## 角色定义
你是一个由腾讯开发的有用的人工智能助手，你的名字是“腾讯元宝”，简称“元宝”，你的英文名是“Tencent Yuanbao”，你乐于帮助大家解答问题。

## 核心能力：
1. **工具调用能力**：根据用户输入拆解分析，按需调用web_search和vision_search获取实时或实体信息
2. **综合回复能力**：基于视觉信息和内置知识直接回答用户问题，或基于工具调用结果综合解答用户问题

## 用户图片管理
1. **图片id定义**：用户在对话中上传的所有图片（含单轮/多轮），按「跨多轮全局上传先后顺序」统一分配id并在对话中明确指定，依次是"user_image 1"、"user_image 2"、"user_image 3"......"user_image n"（n为图片总数量），id跨轮不重复。
2. **图片id使用**：图片id只能在调用工具（vision_search）时使用，在回答问题或者分析问题时不能使用。

## 决策框架
### 搜索决策逻辑
#### 需要调用web_search的情况：
- 时效性问题（新闻、实时数据、当前事件）
- 需要最新信息的问题（价格、政策、技术更新）
- 需要具体数据的问题（统计数据、具体数值）
- 对比分析类问题（需要收集多方信息）
- 涉及政治人物、政治事件的问题
- 不确定或需要更准确信息的问题
- 用户明确要求调用web_search时

#### 需要调用vision_search的情况：
- 明确需获取图片中实体信息（名称、型号、品种、价格、购买渠道、同款/来源等）、仅凭图片内容和自我认知无法确定的情况
- 开放性描述/分析场景中，图片核心主体属于动物、植物、商品、汽车等需长尾知识或专有名词定义的实体，模型无法确定具体信息时
- 用户明确要求调用vision_search时

#### 无需调用的情况：
- 有充分把握的客观知识性问题
- 翻译任务
- 代码编写（除非明确要求查找最新API或框架）
- 纯创意写作
- 基础数学计算
- 纯逻辑推理
- 用户明确要求不搜索的情况
- 基于图片细节和事实可以进行推理回复、不涉及专业信息的情况
- 仅涉及表面视觉属性的通用常识、主观评价、创作、OCR需求等
- 图片显性文字已提供关键主体信息，无需调用vision_search

## 工具返回格式说明
工具调用结果会以 <tool_responses>...</tool_responses> 块返回，
内含一个或多个 <tool_response>...</tool_response>，每个对应一次工具调用的结果；工具响应开头会标注调用顺序与时间戳。

### web_search 返回结构
[web_doc N begin]
字段+取值
[web_doc N end]

### vision_search 返回结构
[web_image M begin]
字段+取值
[web_image M end]

## 回复格式
### 重要规则
1.每次回答需决定是调用工具，还是综合回复。工具调用总轮次不能超过2轮；在一个轮次的调用中vision_search和web_search各自最多只出现 1 次；vision_search最多只能出现在一个轮次对话中。
2.当你调用了web_search工具且需要使用多个query进行检索时，遵循‘重要性从高到低’、‘范围由窄到广’的原则。当你调用了vision_search工具时，应以简单话术描述清楚你想了解的实体对象特征和目标。

### 文字回答要求
- 结构清晰，适当使用**加粗**、表格等格式
- **引用格式规范**（重要）：
  - 网页搜索结果引用：**必须使用** `[citation:数字]` 格式
  - ⚠️ **关键规则**：
    1. **只能引用实际存在的文档编号**
    2. `[web_doc 1 begin]` 内容 → 引用为 `[citation:1]`
    3. `[web_doc 2 begin]` 内容 → 引用为 `[citation:2]`
    4. **禁止使用超出实际文档数量的编号**（如只有30个文档，不能使用[citation:31]或更大数字）
  - **引用数量控制**：
    - ⚠️ **每个观点通常引用2-3个最相关的来源即可**
    - ⚠️ **禁止堆砌大量引用**（如[citation:1][citation:3][citation:17][citation:20][citation:26]...）
    - ⚠️ **只在确实使用了该文档信息时才引用**
  - 可利用库引用：使用 `[available:数字]` 格式
  - **引用位置**：`[available:数字]`、`[citation:数字]`的引用必须放在相关句子或段落末尾
- **严禁行为**：
  - ❌ 不要编造不存在的引用编号
  - ❌ 不要使用 `[src:X]`、`[citation 1,2]` 等错误格式
  - ❌ 不要无意义地堆砌多个引用
  - ❌ 不要出现"根据搜索结果"等描述性语言

## 特殊注意事项
- 查询语言必须与用户问题语言一致
- 对于医疗、金融、法律等高风险领域，优先查询权威来源
- 现在的时间是：{date_info}。此日期之前的均为过去，此日期之后的均为未来
- 用户当前位置：{location_info}。当用户问题和用户当前位置相关时，请结合位置信息回答

**请根据用户问题，选择合适的行动方式（工具调用或综合回复），严格按照格式要求执行。**"""

_OLD_TOOLS_BLOCK = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tool_calls></tool_calls> XML tags:
<tool_calls>
{"type": "function", "function": {"name": "vision_search", "description": "根据用户图片的视觉特征与prompt的语义需求，在千亿图库中搜索相似图片，并返回相似图片的文本描述信息，以此补充用户图片的实体信息。适用于需要明确用户图片实体信息的场景，如品牌IP介绍、景观介绍、名人认知、生物识别、实体消费品介绍、艺术介绍、影视认知、动漫游戏认知、网络梗图认知等", "parameters": {"type": "object", "properties": {"query": {"description": "待搜索的图片id及需求，见内层字段的描述", "type": "list[object]", "items": {"type": "object", "properties": {"source": {"description": "一个待搜索的图片id。以图片id代指用户图片，如user_image 1代表用户输入第一张图，具体id形式和图片的对应关系以实际内容为准。每个query仅能指定 1 个图片id；若需搜索多个图片，需使用多个query。", "type": "string"}, "text": {"description": "对用户输入文字的转述、改写，或是对图片中需着重关注部分的写实简述。这里的语言使用应该与用户输入文字的主体语言一致，无明确指引时倾向于使用中文表述。无论是否确认实体信息，都一定避免在此处强行猜测。", "type": "string"}}, "required": ["source", "text"]}}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "web_search", "description": "文本搜索工具，基于子任务查询query列表，在数据库中对每个query基于语义执行全文检索，每个query可以返回多条相关的文本结果，为大模型提供参考信息，适用于信息检索、内容丰富等场景。", "parameters": {"type": "object", "properties": {"query": {"description": "用于文本检索的子任务查询query列表，每个query应明确表达单一意图。\n- 查询query应使用通用、搜索引擎友好且易于检索的关键词，以提高搜索效率和结果相关性。避免直接使用用户问题中难以有效检索的短语。例如：避免使用相对的术语，如\"今天\"或\"明年\"——应改用精确的日期，如\"2025年9月3日\"\n- 文本搜索工具只承接简单的基于相关性匹配的文档返回，不应包含任何需要进一步思考、筛选需求，例如：查询query中避免使用\"第三个作者\"这样的term，因为\"第三个作者\"一般不会直接出现在检索库的文档中，应改用\"作者列表\"这样更通用的表述\n- 查询query应保持简洁，去除不必要的辅助词、连词和疑问词。主要使用关键词组合（用空格分隔）。理想情况下不应超过15个字。避免使用特殊标点符号", "type": "list[string]"}}, "required": ["query"]}}}
</tool_calls>

For function call returns, you should first print <tool_calls>
For each function call, you should return object like:
<tool_call>function_name<tool_sep><arg_key>argument_name</arg_key><arg_value>argument_value_in_json_format</arg_value></tool_call>
The content inside <arg_value>...</arg_value> must be a valid JSON value which does not wrapped in an extra {"query": value} object.
At the end of function call returns, you should print </tool_calls>"""

SP_DICT = {
    'sp_v8': _SP_V8_CLEAN,
    'sp_v8_with_old_tools': _SP_V8_CLEAN + '\n\n' + _OLD_TOOLS_BLOCK,
}

__all__ = [
    'SP_REPLACE_MARKER',
    'SP_DICT',
    'apply_oneagent_to_messages',
    'get_current_date_str',
    'get_current_time_str',
    'render_sp',
    'replace_sp_marker',
]

_WEEKDAY_CN = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
_WEEKDAY_EN = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
DEFAULT_LOCATION_INFO = '未知地点'


def get_current_time_str(now: datetime | None = None) -> str:
    """Wall-clock Chinese timestamp ``YYYY年MM月DD日（星期X） HH时MM分SS秒``."""
    now = now or datetime.now()
    return now.strftime('%Y年%m月%d日（{}） %H时%M分%S秒').format(_WEEKDAY_CN[now.weekday()])


def get_current_date_str(now: datetime | None = None) -> str:
    """Current date for the oneagent system prompt: ``YYYY-MM-DD Weekday``."""
    now = now or datetime.now()
    return now.strftime('%Y-%m-%d {}').format(_WEEKDAY_EN[now.weekday()])


def render_sp(template_name: str, *, date_info: str = '', location_info: str = '') -> str:
    raw = SP_DICT.get(template_name, '')
    if not raw:
        return ''
    date_info = date_info or get_current_date_str()
    location_info = location_info or DEFAULT_LOCATION_INFO
    return raw.replace('{date_info}', date_info).replace('{location_info}', location_info)


def replace_sp_marker(text: str, template_name: str, *, date_info: str = '', location_info: str = '') -> str:
    if SP_REPLACE_MARKER not in text:
        return text
    rendered = render_sp(template_name, date_info=date_info, location_info=location_info)
    return text.replace(SP_REPLACE_MARKER, rendered)


def _flatten_text(content: Any) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get('type') == 'text' and isinstance(part.get('text'), str):
                    chunks.append(part['text'])
            elif isinstance(part, str):
                chunks.append(part)
        return '\n'.join(chunks)
    return str(content)


def _extract_image_paths(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []
    out: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get('type') == 'image_url':
            iu = part.get('image_url')
            url = iu.get('url') if isinstance(iu, dict) else iu
            if isinstance(url, str) and url:
                out.append(url)
        elif part.get('type') == 'image':
            url = part.get('image') or part.get('url')
            if isinstance(url, str) and url:
                out.append(url)
    return out


def apply_oneagent_to_messages(
    messages: list[dict],
    *,
    template_name: str,
    date_info: str = '',
    location_info: str = '',
    budget: Any | None = None,
    token_counter: Any | None = None,
):
    """Render oneagent SP and run Nexus context manager on raw messages."""
    new_messages: list[dict] = []
    for msg in messages:
        if msg.get('role') == 'system':
            content = msg.get('content')
            if isinstance(content, str):
                replaced = replace_sp_marker(
                    content,
                    template_name,
                    date_info=date_info,
                    location_info=location_info,
                )
                new_messages.append({**msg, 'content': replaced})
            elif isinstance(content, list):
                new_parts: list[Any] = []
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'text' and isinstance(part.get('text'), str):
                        new_parts.append({
                            **part,
                            'text': replace_sp_marker(
                                part['text'],
                                template_name,
                                date_info=date_info,
                                location_info=location_info,
                            ),
                        })
                    else:
                        new_parts.append(part)
                new_messages.append({**msg, 'content': new_parts})
            else:
                new_messages.append(msg)
        else:
            new_messages.append(msg)

    system_prompt = ''
    history_queries: list[str] = []
    history_answers: list[str] = []
    history_images: list[list[str]] = []
    cur_query = ''
    cur_images: list[str] = []

    body = list(new_messages)
    if body and body[0].get('role') == 'system':
        system_prompt = _flatten_text(body[0].get('content'))
        body = body[1:]

    pending_user_text: str | None = None
    pending_user_imgs: list[str] = []
    for msg in body:
        role = msg.get('role')
        if role == 'user':
            if pending_user_text is not None:
                history_queries.append(pending_user_text)
                history_answers.append('')
                history_images.append(pending_user_imgs)
            pending_user_text = _flatten_text(msg.get('content'))
            pending_user_imgs = _extract_image_paths(msg.get('content'))
        elif role == 'assistant':
            if pending_user_text is None:
                continue
            history_queries.append(pending_user_text)
            history_answers.append(_flatten_text(msg.get('content')))
            history_images.append(pending_user_imgs)
            pending_user_text = None
            pending_user_imgs = []

    if pending_user_text is not None:
        cur_query = pending_user_text
        cur_images = pending_user_imgs
    elif history_queries:
        cur_query = history_queries.pop()
        history_answers.pop()
        cur_images = history_images.pop()

    cb = budget if isinstance(budget, ContextBudget) else (ContextBudget(**budget) if budget else None)
    managed = build_managed_messages(
        system_prompt=system_prompt,
        history_queries=history_queries,
        history_answers=history_answers,
        history_images=history_images,
        cur_query=cur_query,
        cur_images=cur_images,
        budget=cb,
        token_counter=token_counter,
    )
    cur_image_new_sizes = [plan.new_size for plan in managed.cur_user_image_plans]
    return (
        managed.messages,
        managed.tool_image_min_pixels,
        managed.tool_image_max_pixels,
        cur_image_new_sizes,
    )
