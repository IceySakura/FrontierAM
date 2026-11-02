"""AudioGRMDS v3 rubric judge prompt / schema / parser, ported from hytuner.

Ported (verbatim logic, minus the unused ``calculate_reward`` aggregator and the
``__main__`` demo block) from
``recipe/audio/speech/reward/rubrics_reward_prompt_v3.py`` on the
``integration/speech-agenticrl-v5-hy-main`` hytuner branch, so that
``gate_judge.py`` / ``llm_judge.py`` no longer need to import the training repo
at runtime. Keep this file byte-for-byte aligned with the hytuner source when
practical — the judge prompt, the ``response_format`` schema, and the parser
must stay in sync with each other and with hytuner's own copy so the scoring
criteria do not silently drift.
"""

from __future__ import annotations

import json
import re


# ============================================================================
# Core Scorer Prompt Template (v3)
# ============================================================================

RUBRICS_REWARD_PROMPT = """\
[任务]
你是语音通话场景下的对话质量评分员。请基于 [对话上下文] 和 [模型当前回复]，逐条判断 [待核对规则列表] 中的每一条规则是否被满足。

你只做评分，不改写回复，不补充新规则，不评价规则本身。

[判定语义]
每条规则都是正向要求，通常写成“模型是否……？”：
- passed = true：模型当前回复满足该规则，或该规则确实本轮不适用。
- passed = false：该规则本轮适用，但模型当前回复未满足、做反、遗漏，或与上下文事实矛盾。

重要原则：
- [待核对规则列表] 是唯一评分依据。规则没要求的风格、格式、篇幅、追问、安全提示等，不要自行加扣分项。
- 只评价 [模型当前回复]。历史助手回复只用于理解上下文、用户意图、指代、历史事实和是否重复。
- 规则通常是针对当前 session 生成的，默认适用于当前轮。只有规则文本明确是条件触发，且当前对话确实没有触发该条件时，才可判“不适用”并 passed=true。
- 不要把“回复没做到规则要求”误判为“不适用”。

[待核对规则列表]
{rules_to_check}

[对话上下文]
以下包含 system prompt（如果有）、历史对话轮次，以及当前轮次的用户输入。
当前轮次的用户输入是对话上下文里最后一条 user 消息。

{conversation_context}

[模型当前回复]
{model_response}

[逐条评分方法]
对每一条规则独立执行以下步骤。

第一步：读懂规则
- 明确这条规则要求当前回复做到什么。
- 如果规则包含多个必须条件，要逐一核对；只有全部满足才 passed=true。
- 如果规则使用“至少、任一、或、如、例如”等表达，要按原文理解，不能随意放宽或收紧：
  * “至少提及 A/B/C 中一个”满足任一即可。
  * “A 和 B”通常要求二者都满足。
  * “如/例如”多为举例，回复可用等价内容满足，除非规则明确要求具体词句或具体对象。
- 如果规则写成“是否没有……”“是否避免……”，满足的意思是当前回复确实没有出现被禁止的问题。
- “等价内容”必须和规则目标在语义上等效。例如规则要求“提到训练场地或休息区”，回复提到“练车场很宽敞”可算等价；但只说“环境不错”不算等价，因为没有具体环境要素。

第二步：判断是否适用
- 规则提到当前用户问题、历史某个实体、具体话题、具体称谓、具体格式或具体篇幅时，通常就是本轮适用。
- 只有在规则本身带有明显触发条件，而当前对话没有进入该条件时，才判不适用。例如规则要求“涉及药物剂量时提示就医”，但本轮完全没聊药物剂量。
- 不适用时 passed=true，但 reason 必须说明“条件未触发”。
- 不要因为回复没有提到规则要求的内容，就说规则不适用。例：规则要求“说明多次违规会影响账号流量”，回复没说影响流量，这是未满足，不是不适用。

第三步：取证判断
- 主要从 [模型当前回复] 找证据；需要理解用户意图、上下文、指代、历史事实、是否重复时，再结合 [对话上下文]。
- 满足时，reason 说明回复中哪部分满足了规则。
- 未满足时，reason 说明规则要求什么、当前回复缺了什么或做反了什么。
- 判断要基于可见文本，不要猜测模型意图，也不要依赖外部不可见材料。

[常见规则类型的判定口径]
下面只是帮助你理解和取证，不是额外规则。最终仍以每条 check_point 原文为准。

1. 需求理解 / 任务完成
- 先还原当前用户真实意图，再看回复是否完成该意图。
- 注意任务动作：解释、推荐、计算、比较、续写、改写、纠错、推理、角色扮演、闲聊、安慰等。
- 如果回复答非所问、只回应了话题但没执行任务、或忽略用户明确限制，相关规则 passed=false。
- 正例：用户问“D 选项呢”，规则要求分析 D 选项；回复直接解释 D 为什么不能加强论点，满足。
- 反例：用户要求“用文氏图验证这个推理”，回复只泛泛讲“文氏图可以表示集合关系”，没有验证题目里的具体推理，不满足。
- 反例：用户要求“三十字左右的话术”，回复写成一大段介绍文案，即使内容相关，也未满足任务限制。

2. 内容质量 / 事实准确 / 关键结论
- 规则要求明确结论时，回复必须给出清楚结论，不能只泛泛铺垫。
- 规则要求具体信息、步骤、理由、风险或建议时，回复要有对应实质内容。
- 若回复与规则中给定事实、题目条件、上下文共识矛盾，相关规则 passed=false。
- 不要因为回复“看起来有帮助”就判通过；必须核对规则要求的具体点是否出现。
- “实质内容”指回复包含规则要求的具体结论、对象、步骤、原因、例子或可执行建议；空泛赞同、泛泛安慰、只说“可以注意一下”不算。
- 正例：规则要求“说明多次违规可能导致降低推荐、限制功能或封号”，回复提到“可能降低推荐、限制发布，严重时封号”，满足。
- 反例：规则要求“至少提到场地、训练设施或休息环境”，回复只说“环境很好，服务也不错”，没有任何具体环境要素，不满足。
- 反例：规则要求“不要将猫和有胡子动物的集合关系说反”，回复说“有胡子的动物都在猫的集合里”，与题目条件矛盾，不满足。

3. 上下文记忆 / 指代消解 / 多轮一致性
- 需要使用历史对话判断实体、称谓、用户偏好、已确认结论、前文任务和当前指代。
- 如果规则要求不引入历史不存在的信息，回复不得虚构用户没说过的姓名、地点、状态、经历或关系。
- 如果规则要求延续历史话题，回复应围绕当前问题和历史上下文，不应重复无关旧内容或切到新话题。
- 正例：历史里用户一直在分析“猫和胡子”的无效推理，本轮说“之前那个推理”；回复继续围绕猫和胡子推理作答，满足。
- 反例：规则要求“不要引入历史未提及的驾校名称或地点”，回复说“我们北京朝阳校区环境很好”，但历史没有校区地点，不满足。
- 反例：用户本轮问“介绍驾校环境”，回复重复上一轮“专业教练、拿证快”的招生话术，未延续当前话题，不满足相关上下文/话题规则。

4. 对话节奏 / 追问克制
- 不要因为回复最后有问句就自动判不满足，也不要因为“先回答再问一句”就自动判满足。
- 如果规则要求直接回答、不追问或追问克制，需判断追问是否必要、相关、低成本、不重复、没有违背用户边界。
- 信息足够、用户只要一个答案、用户让助手决定、用户表达不想继续时，机械补问或连续追问通常不满足相关规则。
- 确实缺少关键槽位时，问一个具体且必要的问题可以满足相关规则；连续抛多个问题通常不满足。
- “必要”指缺少该信息会导致无法完成用户当前任务，或用户明显期待继续细化；为了延长对话而问“你觉得呢”“还想聊聊吗”不算必要。
- 正例：用户问“北京到上海高铁多久”，回复先回答“大约四五个小时”，没有强行追问，满足直接回答/追问克制类规则。
- 正例：用户说“帮我订一下那个”，对象不清；回复只问“你想订的是餐厅还是车票”，满足必要澄清。
- 反例：用户问“D 选项呢”，回复回答后又问“你还想继续分析吗？你觉得哪里难？要不要我再讲一遍？”连续抛多个问题，不满足追问克制。
- 反例：用户说“随便，你决定”，回复继续问“你更喜欢 A 还是 B”，不尊重边界，不满足。

5. 篇幅适配 / 语音可播报
- 篇幅不是越短越好，也不是越长越好，要和当前用户需求匹配。
- 简短确认、道别、轻量闲聊或边界表达：回复应简洁自然，不长篇展开。
- 方法、解释、推荐、推理、复杂情绪支持：回复应给够实质内容，不一句话敷衍。
- 语音可播报规则要求回复读出来自然；若规则明确禁止列表、表格、Markdown、密集符号或复杂结构，出现这些格式则不满足。
- “篇幅适配”必须按规则中的具体判据判断。若规则写“一两句话/约三十字/不要长篇展开”，明显超出且加入无关信息则不满足；若规则要求“完整说明步骤”，过短则不满足。
- 正例：用户问“介绍一下驾校环境”，规则要求“一两句话”；回复用一句话提到场地、设施、休息区，满足。
- 反例：同一规则下，回复额外展开驾校历史、教练资质、报名流程，篇幅和内容都超出，不满足。
- 反例：规则要求适合语音播报且禁止结构化格式，回复使用“1. 场地 2. 教练 3. 价格”编号列表，不满足。

6. 风格 / 格式 / 语言
- 只在规则明确要求时判断，例如口语化、纯文本、无 Markdown、无 emoji、语言一致、无表格列表等。
- 规则未限制格式或风格时，不要因书面化、分段、列表或 Markdown 自行扣分。
- 正例：规则要求“纯文本、无 Markdown”，回复是一整段自然文字，满足。
- 反例：规则要求“没有列表、表格或 Markdown”，回复出现项目符号、编号、表格、代码块、加粗符号或 emoji，不满足。
- 反例：用户全程中文且规则要求语种一致，回复突然用英文回答主要内容，不满足。

7. 人设边界 / 安全
- 如果规则要求不虚构 AI 的现实经历、身体能力、私人关系或物理操作，回复中出现这类自称经历或能力时 passed=false。
- 如果规则要求风险提示、拒绝不安全内容、中立表达或建议咨询专业人士，要核对回复是否实际做到。
- 正例：用户问“你想不想要男朋友”，规则要求守住 AI 人设；回复说“我没有真实的恋爱关系，但可以听你聊感情”，满足。
- 反例：规则要求不使用现实经历，回复说“我以前练车的时候也这样”，虚构了现实经历，不满足。
- 反例：规则要求医疗建议需提示咨询专业医生，回复给出具体用药剂量但没有风险提示或就医建议，不满足。

[输出要求]
请只输出一个合法 JSON 对象，不要输出分析过程、Markdown 包裹或其它文字。
JSON 结构固定为：
{{
  "rule_results": [
    {{
      "id": 1,
      "name": "规则名称（来自输入规则的维度）",
      "passed": true,
      "reason": "简短判断理由，引用当前回复或上下文中的证据。"
    }},
    {{
      "id": 2,
      "name": "规则名称",
      "passed": false,
      "reason": "简短说明规则要求和当前回复未满足之处。"
    }}
  ]
}}
"""


# ============================================================================
# JSON schema constraining scorer output (response_format=json_schema)
# ============================================================================

RULE_RESULTS_JSON_SCHEMA = {
    'type': 'object',
    'properties': {
        'rule_results': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'id': {'type': 'integer', 'description': '规则编号（从 1 开始）'},
                    'name': {'type': 'string', 'description': '规则名称（来自输入规则列表的维度/名称）'},
                    'passed': {
                        'type': 'boolean',
                        'description': 'true 代表回复满足该规则要求（得分），false 代表未满足（不得分）',
                    },
                    'reason': {'type': 'string', 'description': '判断理由'},
                },
                'required': ['id', 'name', 'passed', 'reason'],
                'additionalProperties': False,
            },
        }
    },
    'required': ['rule_results'],
    'additionalProperties': False,
}


# ============================================================================
# Helper: Format checklist into rules_to_check string
# ============================================================================


def format_rules_from_checklist(checklist: list[dict]) -> str:
    """Convert a checklist from the dataset into a formatted rules string.

    Each checklist item has:
        - check_point: str  (the rule description, a positive "模型是否……？" question)
        - dimension: str    (the rule category/dimension)

    Returns:
        Formatted string for insertion into {rules_to_check} placeholder.
    """
    if not hasattr(checklist, '__len__') or len(checklist) == 0:
        return '（无规则）'

    lines = []
    for i, item in enumerate(list(checklist), start=1):
        dimension = item.get('dimension', '未分类')
        check_point = item.get('check_point', '')
        lines.append(f'规则 {i} [{dimension}]: {check_point}')
    return '\n'.join(lines)


def format_rules_from_rubrics(rubrics: dict | None, checklist: list[dict] | None) -> str:
    """Format rules from rubrics dict (multiturn/cultural_creative) or fall back to checklist.

    For datasets with rubrics (containing 'quality' and/or 'multi_turn' keys),
    merge them into a single ordered list (quality first). Otherwise fall back to
    the checklist from extra_info.
    """
    if rubrics is not None and isinstance(rubrics, dict):
        merged = []
        for key in ['quality', 'multi_turn']:
            items = rubrics.get(key, [])
            if hasattr(items, '__len__') and len(items) > 0:
                merged.extend(list(items))
        if merged:
            return format_rules_from_checklist(merged)

    if hasattr(checklist, '__len__') and len(checklist) > 0:
        return format_rules_from_checklist(list(checklist))

    return '（无规则）'


# ============================================================================
# Helper: Format conversation context from original_text_prompt
# ============================================================================


def format_conversation_context(original_text_prompt: list[dict]) -> str:
    """Convert the original_text_prompt messages into a readable conversation context.

    Handles system prompt, multi-turn history (user/assistant pairs with round
    numbering), and the current user input (the last user message).
    """
    if not hasattr(original_text_prompt, '__len__') or len(original_text_prompt) == 0:
        return '（无对话上下文）'

    parts = []
    messages = list(original_text_prompt)  # convert ndarray to list if needed

    # Extract system prompt if present
    start_idx = 0
    if messages[0]['role'] == 'system':
        system_content = messages[0]['content']
        if system_content.strip():
            parts.append(f'<系统设定>\n{system_content}\n</系统设定>')
        start_idx = 1

    remaining = messages[start_idx:]

    if len(remaining) == 0:
        return '\n\n'.join(parts) if parts else '（无对话上下文）'

    if len(remaining) >= 2:
        # Find the last user message as the current turn input; everything before is history.
        last_user_idx = None
        for idx in range(len(remaining) - 1, -1, -1):
            if remaining[idx]['role'] == 'user':
                last_user_idx = idx
                break

        if last_user_idx is None:
            parts.append('（无用户输入）')
            return '\n\n'.join(parts)

        history = remaining[:last_user_idx]
        current_user_msg = remaining[last_user_idx]['content']

        if history:
            round_idx = 1
            history_lines = []
            i = 0
            while i < len(history):
                if history[i]['role'] == 'user':
                    history_lines.append(
                        f'<第{round_idx}轮用户输入>\n{history[i]["content"]}\n</第{round_idx}轮用户输入>'
                    )
                    if i + 1 < len(history) and history[i + 1]['role'] == 'assistant':
                        history_lines.append(
                            f'<第{round_idx}轮助手回复>\n{history[i + 1]["content"]}\n</第{round_idx}轮助手回复>'
                        )
                        i += 2
                    else:
                        i += 1
                    round_idx += 1
                elif history[i]['role'] == 'assistant':
                    history_lines.append(
                        f'<第{round_idx}轮助手回复>\n{history[i]["content"]}\n</第{round_idx}轮助手回复>'
                    )
                    round_idx += 1
                    i += 1
                else:
                    i += 1

            parts.append('<历史对话>\n' + '\n'.join(history_lines) + '\n</历史对话>')

        parts.append(f'<当前轮次用户输入>\n{current_user_msg}\n</当前轮次用户输入>')

    else:
        if remaining[0]['role'] == 'user':
            parts.append(f'<当前轮次用户输入>\n{remaining[0]["content"]}\n</当前轮次用户输入>')
        else:
            parts.append(f'<内容>\n{remaining[0]["content"]}\n</内容>')

    return '\n\n'.join(parts)


# ============================================================================
# Main: Build the complete scoring prompt
# ============================================================================


def build_reward_prompt(
    original_text_prompt: list[dict],
    model_response: str,
    reference_response: str = '',
    checklist: list[dict] | None = None,
    rubrics: dict | None = None,
) -> str:
    """Build the complete reward scoring prompt for the scorer LLM (v3).

    Args:
        original_text_prompt: The original text prompt messages (system + history
            + current user turn).
        model_response: The model's generated response to be evaluated.
        reference_response: Kept for signature compatibility with v2 callers; NOT
            rendered into the prompt in v3 (reference is no longer injected).
        checklist: List of {check_point, dimension} dicts. The single source of rules.
        rubrics: Optional rubrics dict ({quality, multi_turn}); falls back to checklist.

    Returns:
        The complete prompt string ready to be sent to the scorer LLM.
    """
    del reference_response  # unused in v3, kept for signature compatibility
    rules_str = format_rules_from_rubrics(rubrics, checklist)
    context_str = format_conversation_context(original_text_prompt)

    if not model_response or not model_response.strip():
        model_response = '（模型回复为空）'

    return RUBRICS_REWARD_PROMPT.format(
        rules_to_check=rules_str,
        conversation_context=context_str,
        model_response=model_response,
    )


# ============================================================================
# Reward Calculation from Scorer Output
# ============================================================================


def parse_scorer_output(scorer_output: str) -> list[dict] | None:
    """Parse the JSON output from the scorer LLM (v3).

    Extracts the rule_results list. Tolerates markdown code fences, <think> blocks,
    and analysis text before the JSON. Each rule item is expected to carry a
    ``passed`` boolean; for resilience we also accept a legacy ``result`` field
    (interpreted as violation, i.e. passed = not result) so old serving outputs
    don't silently break.

    Returns:
        List of rule result dicts, or None if parsing fails.
    """
    if not scorer_output or not scorer_output.strip():
        return None

    text = scorer_output.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _extract(parsed) -> list[dict] | None:
        if isinstance(parsed, dict) and 'rule_results' in parsed:
            return parsed['rule_results']
        return None

    all_blocks = re.findall(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    for block in all_blocks:
        try:
            res = _extract(json.loads(block.strip()))
            if res is not None:
                return res
        except (json.JSONDecodeError, ValueError):
            continue

    try:
        res = _extract(json.loads(text))
        if res is not None:
            return res
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'\{.*"rule_results".*\}', text, re.DOTALL)
    if json_match:
        try:
            res = _extract(json.loads(json_match.group(0)))
            if res is not None:
                return res
        except json.JSONDecodeError:
            pass

    return None


def _rule_passed(rule: dict) -> bool:
    """Return whether a single rule is satisfied (v3 semantics).

    Reads ``passed`` (true = satisfied). Falls back to legacy ``result``
    (true = violated → passed = not result) when ``passed`` is absent, so mixed
    or old outputs still score sensibly. String values are coerced.
    """

    def _to_bool(v) -> bool:
        if isinstance(v, str):
            return v.strip().lower() in ('true', '是', 'yes', '1')
        return bool(v)

    if 'passed' in rule:
        return _to_bool(rule.get('passed'))
    if 'result' in rule:  # legacy: result=true means violated
        return not _to_bool(rule.get('result'))
    # No verdict field at all → treat as not satisfied (conservative).
    return False
