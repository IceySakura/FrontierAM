# @author: krezhang
# @contact: krezhang
# @file: judge_score.py
# @time: 2026/3/5 21:35
"""
Description:
    根据评分清单对学生回答进行逐条判定和评分的模块
"""

import asyncio
import json
import os
import re

from collections.abc import Sequence
from typing import Any

from nexus.extensions.tasks.mm_agent.reward_system.taiji_api import get_model_response

from .word_count import count_words_and_chars_all
from .yb_rubric_base import get_judge_request_semaphore, parse_json_block, resolve_n_rounds


def _get_judge_model_name(model_name: str | None = None) -> str:
    return model_name or os.environ.get(
        'YB_RUBRIC_JUDGE_MODEL',
        'Qwen35-397B-A17B-FP8-YB-Rubric-SG',
    )


async def _call_judge_model(messages: list[dict[str, Any]], model_name: str | None = None) -> str:
    system_text = ''
    user_parts: list[str] = []
    for message in messages:
        role = message.get('role')
        content = message.get('content', '')
        if role == 'system':
            system_text = str(content)
        elif role == 'user':
            user_parts.append(str(content))

    async with get_judge_request_semaphore():
        answer = await get_model_response(
            img_path=None,
            input_text='\n\n'.join(user_parts),
            MODEL=_get_judge_model_name(model_name),
            system_text=system_text,
            max_token=int(os.environ.get('YB_RUBRIC_JUDGE_MAX_TOKENS', '8192')),
            max_retry=int(os.environ.get('YB_RUBRIC_JUDGE_MAX_RETRIES', '2')),
            request_timeout=int(os.environ.get('YB_RUBRIC_JUDGE_REQUEST_TIMEOUT', '500')),
            temperature=float(os.environ.get('YB_RUBRIC_JUDGE_TEMPERATURE', '0.7')),
            top_p=float(os.environ.get('YB_RUBRIC_JUDGE_TOP_P', '0.8')),
            top_k=int(os.environ.get('YB_RUBRIC_JUDGE_TOP_K', '20')),
            enable_thinking=False,
        )
    if answer is None:
        raise ValueError('LLM response is empty')
    return answer


SCORE_SYSTEM_PROMPT = """
你是一位评判员，负责评估回答是否满足给定的评分标准。

评分规则：
• 对于 "need" 和 "extension" 类型的条目：如果回答满足 conclusion 中的要求，输出 1；否则输出 0。
• 对于 "pitfall" 类型的条目（极其重要，请务必牢记）：conclusion 描述的是一个缺陷/错误/陷阱。
  - 如果回答**没有**出现该缺陷（即成功避开了陷阱） → 输出 1（好）
  - 如果回答**出现了**该缺陷（即踩中了陷阱） → 输出 0（差）
  - 注意：pitfall 的判定逻辑与 need 相反！conclusion 描述的是"不应该出现的问题"，回答中没有这个问题才是好的（得1分），有这个问题才是差的（得0分）。

过度扩展判定规则（重要）：
• 当评分标准的 conclusion 涉及"内容主题"或"内容完整性"时，你需要结合用户的原始问题来判断回答是否"恰当地"满足了要求。
• 如果用户只是要求写一段简短的内容（如"帮我写一段XX"、"写个XX"），但回答生成了一份完整的正式文档/报告/合同/承诺书等远超用户需求范围的内容，则应判定为不满足（score=0），因为回答没有正确理解用户的需求范围。
• 判断依据：用户是否明确要求了完整文档的形式？如果用户说"写一份XX承诺书"、"起草一份XX合同"，则完整文档是合理的；如果用户只是说"写一段XX"、"帮我写XX"而没有指定文档形式，则过度扩展为完整文档是不恰当的。
• 简言之：回答的形式和范围应与用户的实际需求匹配，而非仅仅主题匹配。

判定宽严度规则（重要）：
「严格」指的是：不得对 conclusion 的条件进行过度延伸解读，不要添加 conclusion 中未提及的额外要求。
「宽松」指的是：不要求回答与 conclusion 逐字匹配，允许语义等价的近义表达。
两者并不矛盾——你既不应该额外加码（添加 conclusion 没要求的东西），也不应该过度抠字眼（拒绝合理的近义表达）。

具体规则如下：
• 当 conclusion 使用"或"连接多个条件时（如"A或B"），只要满足其中任一条件即可判定为满足。
• 当 conclusion 描述的条件包含多个层次时（如"名称中含X概念"），只要回答在任一层次上满足即可。例如：如果 conclusion 要求"世界名称或核心设定中含'风流'概念"，而回答的世界名称明确包含"风流"二字（如"风流篇"），则应判定为满足，不应进一步质疑内容是否"真正体现了风流"。
• 当 conclusion 使用"相关元素"、"相关内容"、"相关信息"等宽泛表述时，不要求回答中必须字面出现原词，只要回答中包含了属于该范畴的具体实例即可判定为满足。例如：conclusion 要求"提及史莱克七怪相关元素"，而回答中出现了唐三、小舞、唐雅、贝贝等史莱克七怪成员，即属于"相关元素"，应判定为满足。
• 当 conclusion 中给出了具体的名称、地点、描述等内容时，如果回答中使用了**语义等价或合理近义的表达**（指向同一事物/同一含义），应判定为满足。
  - 典型等价：操场/运动场/体育场、老师/教师、举办/开展/召开、大操场/运动场等
  - 例如：conclusion 要求"活动地点为'枣阳市阳光学校大操场'"，回答写"枣阳市阳光学校运动场"→ 核心信息一致，应判定为满足。
  - 但如果指向**不同事物**（如要求"操场"但写了"教室"，要求"张三"但写了"李四"），则判定为不满足。
• 推理与结论一致性：你的 reason 中的分析推理必须与最终 score 结论保持逻辑一致。如果你在 reason 中已经识别到回答包含了满足条件的内容，则最终结论必须判定为满足（score=1），不得出现"推理认为满足但结论给0分"的自相矛盾情况。

多模态/图片内容判定规则（极其重要）：
• 你无法看到用户上传的图片，因此不要基于"图片是否存在"或"图片是否可见"来进行判定。
• 当 conclusion 涉及"结合图片内容"、"基于图片"等描述时，你只需要判断模型回答的文本内容是否体现了 conclusion 所描述的行为（如：回答中是否包含了与 conclusion 提到的图片场景/主题相关的内容）。
• 例如：conclusion 要求"回答结合了用户提供的图片内容（蒸汽爆米花工厂场景）进行分析"，你只需检查回答中是否涉及了"蒸汽爆米花工厂"相关的内容，而不是去验证图片是否真的存在于对话中。
• **严禁出现以下推理**：
  - ❌ "由于用户并未提供图片证明是航拍，模型属于无依据地断言" → 你看不到图片，不能判断用户是否提供了图片
  - ❌ "用户没有提供截图证明..." → 你无法确认用户是否提供了截图
  - ❌ "图片中并未出现..." → 你看不到图片，无法判断图片中有什么
• **你的判定依据只有两个**：conclusion 的文字描述 + answer 的文本内容。除此之外不得引入任何关于图片存在性、图片内容的推测。
• 简言之：只基于 conclusion 的文字描述和 answer 的文本内容进行匹配判定，不要引入"图片不存在/不可见/未提供"作为扣分理由。

数值与数量判定规则（重要）：
• 对数字、范围、时间等事实点保持高度敏感，答案须与考点在数值上一致（允许等价表达和合理近似，但不允许实质性数值偏差）。
• 等价表达认可（视为数值一致）：
  - 单位换算：1kg=1000g，100cm=1m，1.5小时=90分钟
  - 格式等价：1999=1,999=一千九百九十九
  - 百分比等价：50%=0.5=五成
  - 近似表达：答案含"约/大约/左右/近/接近"时，若考点精确值落在合理近似范围内视为满足（如考点1999元，答案约2000元→满足）
  - 精度/日期容忍：答案可用更低精度表述（年>月>日），已表述精度内须一致（如2022年6月6日→2022年6月✅，99.7%→100%❌）
• 数量要求宽松原则：当 conclusion 给出数量要求（如"提供2-3个""列举至少2个"），答案数量≥下限即满足（多答不扣分，少于下限才不满足）。
• 边界词语义：
  - "超过/多于/大于X"：答案须 > X
  - "至少/不少于/≥X"：答案须 ≥ X
  - "不到/少于/低于X"：答案须 < X
  - "最多/不超过/≤X"：答案须 ≤ X
  - "X以上"/"X以下"：含X本身
• 区间子集原则：当 conclusion 给出数值范围（如"价格160-220元"），答案区间 ⊆ 考点区间才算满足（如答160-185元✅，答160-230元❌）。

举例/枚举类判定规则：
• 当 conclusion 中以举例形式给出答案（含"如""例如""等"标志词）时，答案不要求与示例完全一致，只要逻辑自洽且合理即可满足。
• 覆盖比例以 conclusion 中明确要求的数量为基准（如"提供2-3个"则基准为2-3），而非 conclusion 中列举的所有示例数量。
• 只要已列举项全部无错误即可判定为满足，不因未覆盖全部示例而扣分。但如果列举项中存在事实错误，则判定为不满足。

名称别称等价规则：
• 当 conclusion 中使用"或称/或/又称/也叫/即/又名"以及括号内列出多个名称时，这些名称互为等价，答案匹配其中任何一个即视为满足。
• 判定时须逐一核对 conclusion 中列出的所有等价名称，不得仅锚定第一个或最显眼的名称而忽略其余。

逻辑矛盾/一致性类判定规则（重要）：
• 当 conclusion 涉及"逻辑矛盾"、"前后矛盾"、"逻辑不一致"、"自相矛盾"等内容一致性检查时，你必须**完整通读模型回答的全部内容**进行分析判断，而不是仅依据 conclusion 中括号内给出的示例来判定。
• conclusion 中括号内的示例（如"如前文判定某设定合理后文又判定不合理、数据前后不一致"等）仅是矛盾类型的举例说明，不是判定的唯一依据。你需要自行在回答全文中寻找是否存在任何形式的前后矛盾或逻辑不一致。
• 判定流程：①完整阅读回答全文 → ②识别回答中的所有论点、判断、数据、结论 → ③检查这些内容之间是否存在相互矛盾或不一致 → ④给出判定结果。
• 如果回答内容较长，不要只看开头和结尾，要关注全文中任何可能的矛盾点。

条件触发类判定规则（重要）：
• 当 conclusion 的结构为"当/如果回答中提及X时，X应该满足Y"这种**先设条件、再提要求**的句式时，表示该条目是**条件触发型**判定。
• 条件触发型的判定逻辑：
  - 回答中**没有提及X**（条件未触发） → 直接给1分，无需核查
  - 回答中**提及了X且满足Y**（条件触发且正确） → 给1分
  - 回答中**提及了X但不满足Y**（条件触发但错误） → 给0分
• 例如：conclusion 说"回答中提及端午节时间时，明确为农历五月初五"——如果回答完全没提端午节时间 → 1分；提了且写的是"农历五月初五" → 1分；提了但写错了 → 0分。
• 注意区分：如果 conclusion 是直接要求"回答中明确提及X"（如"回答中明确提及史莱克学院"），这是**直接要求型**，不是条件触发型。直接要求型必须在回答中找到X才能得分，找不到则为0分。
• 区分标准：conclusion 是否包含"当...时"、"如果...则"、"提及X时，X应该..."等先设条件再提要求的结构。如果只是"回答中包含/提及X"，则是直接要求。

隐式前提类判定规则（极其重要）：
• 当 conclusion 涉及"正确识别/理解某些背景信息"（如聊天记录中的消息发送方归属、对话时间线、角色关系等），但模型的回答是**生成回复/建议**而非**分析报告**时，模型不一定需要显式输出这些背景信息。
• **核心原则：不提及不扣分，提及但搞错才扣分**
• 判定逻辑：
  - 回答中**完全没有提及** conclusion 中列出的具体信息点（如某条消息的发送方、某个时间点等） → 直接给1分（模型不需要显式声明这些信息）
  - 回答中**提及了且正确** → 给1分
  - 回答中**提及了但与 conclusion 描述的事实冲突** → 给0分
• 例如：conclusion 说"回答正确识别'话说，你嫂子叫啥'为用户（男方）发送"——如果回答完全没提到这条消息 → 1分（不提及不算错）；如果回答说"你主动问了嫂子叫啥" → 1分（正确归属）；如果回答说"她问了你嫂子叫啥" → 0分（搞反了发送方）。
• **适用场景识别**：当 conclusion 中包含具体的消息内容引用（如引号括起来的对话原文）、发送方归属、时间戳等细节信息，且模型的任务是"帮写回复/给建议"而非"逐条分析对话"时，应按隐式前提类处理。
• **如何判断任务性质**：查看用户问题（question）——如果用户要求的是"帮我回复/帮我想几条回复/怎么回"等生成类任务，则模型不需要显式输出背景分析；如果用户要求的是"帮我分析这段对话/分析他什么意思/帮我看看这段聊天"等分析类任务，则模型应该显式输出分析内容，此时按**直接要求型**处理（不适用隐式前提规则）。
• **与条件触发类的区别**：条件触发类需要 conclusion 中有明确的"当...时"句式；隐式前提类不需要这种句式，而是根据任务性质判断——如果模型的任务不要求显式输出这些信息，就按隐式前提处理。

禁止过度推理规则（极其重要）：
• 判定时必须基于模型回答的**字面内容**和 conclusion 的**明确要求**，不得进行过度推理或逻辑延伸。
• **什么是过度推理**：从模型的某个表述中推断出"模型一定理解错了某个前提"，但这个推断并非唯一合理的解释。
• 例如：
  - conclusion 说"女方说没去过长沙，导致无法回答当地特色"，模型回复中出现"被姐姐安利的感觉也超棒的" → 不能推断为"模型误以为女方在积极讨论长沙"，因为没去过长沙的人完全可以通过网络/朋友安利了解美食，这是合理的表述。
  - conclusion 说"对方表达了冷淡态度"，模型回复语气热情 → 不能仅因为回复热情就推断"模型没理解对方冷淡"，因为热情回复可能是用户想要的策略（如用热情化解冷淡）。
• **判定原则**：只有当模型的表述与 conclusion 中的事实**直接矛盾**（无法用任何合理方式解释）时，才判定为不满足。如果存在合理的解释路径，应判定为满足。
• 简言之：给模型回答一个合理的解释机会，不要选择最不利的解读。

Pitfall 虚构内容判定规则（极其重要）：
• 对于 pitfall 类型中涉及"虚构不存在的内容"的条目，判定依据**只能是 conclusion 中明确列出的事实**。
• **你无法看到原始图片/截图**，因此不能自行判断"某个内容在截图中是否存在"。你只能依据 conclusion 中描述的事实来判断模型是否虚构。
• 判定逻辑：
  - 如果 conclusion 明确说"截图中不存在X"，而模型回答中声称存在X → 给0分（踩中陷阱）
  - 如果 conclusion **没有提及**某个内容是否存在，而模型回答中提到了该内容 → **不能判定为虚构**，给1分（你无法确认该内容是否在截图中存在）
• 例如：conclusion 说"回答虚构了截图中不存在的内容，例如声称周六有对话发生"。如果模型提到了"龙虾图"、"等你回来安排"等内容，但 conclusion 中并未明确说这些内容不存在于截图中 → 你无法判定这是虚构，应给1分。只有 conclusion 中明确列举为"不存在"的内容，模型提到了才算虚构。
• **严禁的推理方式**："因为我看不到图片/截图，所以模型提到的任何具体对话内容都可能是虚构的" → 这是错误的，你必须以 conclusion 的明确描述为准。

要求：
• 你必须严格按照评分标准进行判定，只考虑评分标准中列出的条件。
• 你不得考虑任何其他因素，包括你自己的观点或外部知识。严禁引入外部世界知识来推翻或补充 conclusion 中的判定条件。
• **判定的唯一依据是 conclusion 的文字描述 + answer 的文本内容**。不得引入任何关于图片、截图、音频等多模态内容的推测。
• JSON 安全规则：所有字段值中的引号统一使用中文引号（""或''），禁止在字段值内部出现英文双引号（"），以确保 JSON 合法可解析。
""".strip()

SCORE_USER_PROMPT_TEMPLATE = """
## 评分标准

{checklist}

## 用户问题

{question}

## 提问时间

{date}

## 历史对话

{history}

## 模型回答

{answer}

---

## 评判指令

请逐条判断模型回答是否满足评分标准中每条的 "conclusion"。
输出一个 JSON 对象，其中 "scored_checklist" 数组的每个元素对应一条评分标准（保持原始顺序），包含以下字段：

```json
{{
  "scored_checklist": [
    {{
      "id": "评分条目 id",
      "statement": "评分条目 statement",
      "conclusion": "评分条目 conclusion（原样返回）",
      "reason": "简要说明回答为何满足或不满足该条标准",
      "score": 1 或 0
    }}
  ]
}}
```

- 对于 "need" 和 "extension" 类型：判断回答是否满足 "conclusion" 字段的要求。满足输出 1，不满足输出 0。
- 对于 "pitfall" 类型（重要）："conclusion" 描述的是一个缺陷/错误/陷阱。判定逻辑与 need 相反：
  - 回答**没有**出现 conclusion 描述的缺陷（成功避开陷阱） → 输出 1
  - 回答**出现了** conclusion 描述的缺陷（踩中陷阱） → 输出 0
- 总结：1 始终表示"好"（满足要求 / 避开陷阱），0 始终表示"差"（未满足要求 / 踩中陷阱）。
- 每条评分标准必须独立判定，即使多条共享相同的 "id"。
- 你必须严格按照评分标准进行判定，只考虑评分标准中列出的条件，不得引入外部知识或个人观点。
- 当 conclusion 涉及图片/多模态内容时，只需基于 conclusion 描述和回答文本内容进行匹配判定，不要以"图片不存在/不可见"为由扣分。
- 当 conclusion 涉及内容主题/完整性时，需结合用户原始问题判断回答的形式和范围是否恰当（过度扩展为完整正式文档 → score=0）。
- 允许语义等价的近义表达（如"运动场"="大操场"），但不允许指向不同事物的替换。
- 数值判定：允许等价表达（单位换算、格式等价、近似表达），但不允许实质性数值偏差。数量要求≥下限即满足，多答不扣分。
- 举例/枚举类：答案不要求与示例完全一致，逻辑合理即可；覆盖比例以要求数量为基准。
- 名称别称：conclusion 中"或称/又称/也叫"及括号内列出的名称互为等价，匹配任一即满足。
- 如果 conclusion 是条件触发型（"当提及X时，X应该..."），回答未提及X则直接给1分。
- 隐式前提类判定：当 conclusion 涉及消息发送方归属、对话细节等背景信息，且用户问题是"帮我回复/怎么回"等生成类任务（而非"帮我分析对话"等分析类任务）时，模型未显式提及这些信息不算错（给1分），只有提及但搞错才给0分。分析类任务则按直接要求型处理。
- 禁止过度推理：不得从模型的某个表述中推断"模型一定理解错了某个前提"。只有当模型的表述与 conclusion 中的事实**直接矛盾**（无法用任何合理方式解释）时才判为不满足。
- Pitfall 虚构判定：对于"虚构不存在的内容"类 pitfall，只有 conclusion 中**明确列举为不存在**的内容，模型提到了才算虚构。conclusion 未提及是否存在的内容，模型提到了不能判为虚构（因为你看不到原始图片/截图）。
- 你的 reason 推理必须与最终 score 结论逻辑一致，不得自相矛盾。
- 仅输出合法的 JSON，不要输出任何额外文字。
""".strip()


# ────────────────────────────── 字数统计 PE ──────────────────────────────

WORD_COUNT_SYSTEM_PROMPT = """
# Role: 专业字数统计工具调用工程师

## Profile

你是一名专业的字数统计工具调用工程师。你的核心任务是分析用户的提问和模型回答，结合提供的字数范围要求，从模型回答中抽取要进行字数核查的内容，组织成规范的函数格式进行字数统计工具的调用。

## Tool Definition (工具定义)
* **Tool Name:** `word_count`
* **Parameters:**
  * `length_range`: 字符串类型，表示字数要求（格式如：`[min, max]`，`[0, 50]`，`[200, 300]`）。
  * `content_segment`: 字符串类型，模型回答中待统计的文本片段。
""".strip()

WORD_COUNT_USER_PROMPT_TEMPLATE = """
## Workflow (工作流)

1. **需求扫描**：详细扫描用户提问、字数范围要求和模型回答。
2. **量化抽取**：严格参考每一个字数范围要求，从模型回答中抽取**对应的**内容片段进行字数核查的内容。注意同一个字数范围要求，可能抽取多个内容片段进行核查。
3. **结构化输出**：填充工具调用参数，生成 JSON 结果。

## Examples (少样本示例)

**Case 1:**

* **用户提问：** "请帮我写一段关于春天的文案，要求100字左右，优美一点。"
* **模型回答：** "好的，这是一段按您的要求写的关于春天的文案：春风拂过，冰雪消融，万物从沉睡中苏醒。枝头抽出新芽，花儿次第绽放，空气中漫着清新与温柔。阳光正好，微风不燥，每一寸土地都在诉说生机，每一缕风都带着希望。不必追赶时光，只需静静感受，这人间最美的时节，正带着温柔与力量，悄悄来到身边。"
* **字数范围要求：** ["[80, 120]"]
* output：
```
{{
  "tool_call": [
    {{
      "tool_name": "word_count",
      "parameters": {{
        "length_range": "[80, 120]",
        "content_segment": "春风拂过，冰雪消融，万物从沉睡中苏醒。枝头抽出新芽，花儿次第绽放，空气中漫着清新与温柔。阳光正好，微风不燥，每一寸土地都在诉说生机，每一缕风都带着希望。不必追赶时光，只需静静感受，这人间最美的时节，正带着温柔与力量，悄悄来到身边。"
      }}
    }}
  ]
}}
```

**Case 2:**

* **用户提问：** "根据图片写一段朋友圈文案，不超过50字"
* **模型回答：** "好的，这是我为您生成的多个朋友圈文案：1. 春风如约，万物温柔。心有暖阳，所见皆是美好。愿日子清欢，岁岁皆安。 2. 春风如约，万物温柔。心有暖阳，日子清亮，不慌不忙，静待花开。🌿 3. 春风如约，万物温柔。心向暖阳，所见皆是美好。愿日子清澈，岁岁常欢愉。 "
* **字数范围要求：** ["[40, 50]"]
* output：
```
{{
  "tool_call": [
    {{
      "tool_name": "word_count",
      "parameters": {{
        "length_range": "[40, 50]",
        "content_segment": "春风如约，万物温柔。心有暖阳，所见皆是美好。愿日子清欢，岁岁皆安。"
      }}
    }},
    {{
      "tool_name": "word_count",
      "parameters": {{
        "length_range": "[40, 50]",
        "content_segment": "春风如约，万物温柔。心有暖阳，日子清亮，不慌不忙，静待花开。🌿"
      }}
    }},
    {{
      "tool_name": "word_count",
      "parameters": {{
        "length_range": "[40, 50]",
        "content_segment": "春风如约，万物温柔。心向暖阳，所见皆是美好。愿日子清澈，岁岁常欢愉。"
      }}
    }}
  ]
}}
```

**Case 3:**

* **用户提问：** "生成开头20字、中间一段30字、结尾10字左右的小作文"
* **模型回答：** "以下是满足您需求的作文：春风拂过人间，万物悄然苏醒焕新颜。\n枝头嫩芽轻摇，繁花次第绽放，暖阳洒下温柔，风里裹着清新气息，每一处都藏着生机与希望。\n心怀暖意，岁岁安然。"
* **字数范围要求：** ["[16, 24]", "[24, 36]", "[8, 12]"]
* output：
```
{{
  "tool_call": [
    {{
      "tool_name": "word_count",
      "parameters": {{
        "length_range": "[16, 24]",
        "content_segment": "春风拂过人间，万物悄然苏醒焕新颜。"
      }}
    }},
    {{
      "tool_name": "word_count",
      "parameters": {{
        "length_range": "[24, 36]",
        "content_segment": "枝头嫩芽轻摇，繁花次第绽放，暖阳洒下温柔，风里裹着清新气息，每一处都藏着生机与希望。"
      }}
    }},
    {{
      "tool_name": "word_count",
      "parameters": {{
        "length_range": "[8, 12]",
        "content_segment": "心怀暖意，岁岁安然。"
      }}
    }}
  ]
}}
```

**Case 4:**

* **用户提问：** "写一段20s的口播，介绍维生素c泡腾片"
* **模型回答：** "每天一片维生素 C 泡腾片，温水一冲快速溶解，酸甜果味超好喝！补充每日所需维 C，增强抵抗力，换季少感冒。上班、上学、居家都能随时喝，方便又省心，全家都能常备～"
* **字数范围要求：** ["[50, 70]"]
* output：
```
{{
  "tool_call": [
    {{
      "tool_name": "word_count",
      "parameters": {{
        "length_range": "[50, 70]",
        "content_segment": "每天一片维生素 C 泡腾片，温水一冲快速溶解，酸甜果味超好喝！补充每日所需维 C，增强抵抗力，换季少感冒。上班、上学、居家都能随时喝，方便又省心，全家都能常备～"
      }}
    }}
  ]
}}
```

## Start Labeling

* **用户提问：** {question}
* **模型回答：** {answer}
* **字数范围要求：** {length_range}

## Output Format (输出格式)

* **仅输出 JSON 格式**
* **不得包含任何解释性文字、注释或 Markdown 标记**（如 ```json）
* JSON 结构必须严格如下：
```
{{
  "tool_call": [
    {{
      "tool_name": "word_count",
      "parameters": {{
        "length_range": "具体字数要求范围",
        "content_segment": "待核查的模型回答的内容片段"
      }}
    }}
  ]
}}
```
""".strip()


# 模型上下文窗口大小（tokens），用于截断保护
# 预留 2000 tokens 给输出，实际可用输入 token 数 = MAX_CONTEXT_TOKENS - OUTPUT_RESERVE_TOKENS
MAX_CONTEXT_TOKENS = 70000
OUTPUT_RESERVE_TOKENS = 2000
# 粗略估算：1个中文字符≈1.5 tokens，1个英文单词≈1.3 tokens
# 保守按 1 字符 ≈ 1.5 tokens 估算
CHARS_PER_TOKEN_ESTIMATE = 1.5


def _estimate_tokens(text: str) -> int:
    """粗略估算文本的 token 数"""
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN_ESTIMATE)


def _truncate_text(text: str, max_chars: int, tail_hint: str = '\n\n...[内容过长，已截断]...') -> str:
    """截断文本到指定字符数，并添加截断提示"""
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars] + tail_hint


# ────────────────────────────── 字数统计相关方法 ──────────────────────────────


def _count_words(text: str) -> int:
    """
    统计文本字数，统一使用 word_count.py 的 count_words_and_chars_all 方法。
    """
    if not text or not text.strip():
        return 0
    return count_words_and_chars_all(text.strip())


def _build_word_count_prompt(question: str, answer: str, length_range: list[str]) -> str:
    """构建字数统计的提示词"""
    return WORD_COUNT_USER_PROMPT_TEMPLATE.format(
        question=question, answer=answer, length_range=json.dumps(length_range, ensure_ascii=False)
    )


async def _call_word_count_llm(
    question: str, answer: str, length_range: list[str], max_retries: int = 3
) -> dict[str, Any]:
    """
    调用 LLM 抽取需要核查字数的内容片段

    Args:
        question: 用户提问
        answer: 模型回答
        length_range: 字数范围要求列表
        max_retries: 最大重试次数

    Returns:
        包含 tool_call 列表的字典
    """
    prompt = _build_word_count_prompt(question, answer, length_range)

    for attempt in range(max_retries):
        try:
            response = await _call_judge_model([
                {'role': 'system', 'content': WORD_COUNT_SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt},
            ])
            content = response

            if not content or not content.strip():
                print(f'[WordCount] 第 {attempt + 1} 次尝试：LLM 返回空响应，重试中...')
                continue

            # 解析 JSON
            parsed = parse_json_block(content)
            if 'tool_call' in parsed:
                return parsed
            print(f'[WordCount] 第 {attempt + 1} 次尝试：返回格式不正确，重试中...')
            continue
        except Exception as e:
            print(f'[WordCount] 第 {attempt + 1} 次尝试失败: {e}')
            if attempt < max_retries - 1:
                continue
            raise

    raise ValueError(f'[WordCount] 经过 {max_retries} 次重试后仍然失败')


async def _check_word_count(question: str, answer: str, length_range: list[str]) -> dict[str, Any]:
    """
    执行字数统计核查流程：
    1. 调用 LLM 抽取内容片段
    2. 对每个片段进行实际字数统计
    3. 判断是否在要求范围内

    Args:
        question: 用户提问
        answer: 模型回答
        length_range: 字数范围要求列表

    Returns:
        字数核查结果字典，包含每个片段的统计结果和总体是否通过
    """
    try:
        llm_result = await _call_word_count_llm(question, answer, length_range)
    except Exception as e:
        return {'error': str(e), 'pass': False, 'details': []}

    tool_calls = llm_result.get('tool_call', [])
    details = []
    all_pass = True

    for call in tool_calls:
        params = call.get('parameters', {})
        range_str = params.get('length_range', '')
        content_segment = params.get('content_segment', '')

        # 统计实际字数
        actual_count = _count_words(content_segment)

        # 解析范围
        try:
            range_match = re.match(r'\[(\d+),\s*(\d+)\]', range_str)
            if range_match:
                min_count = int(range_match.group(1))
                max_count = int(range_match.group(2))
                in_range = min_count <= actual_count <= max_count
            else:
                min_count = 0
                max_count = float('inf')
                in_range = True  # 无法解析范围时默认通过
        except Exception:
            min_count = 0
            max_count = float('inf')
            in_range = True

        if not in_range:
            all_pass = False

        details.append({
            'length_range': range_str,
            'content_segment': content_segment,
            'actual_count': actual_count,
            'expected_range': [min_count, max_count],
            'in_range': in_range,
        })

    return {
        'pass': all_pass,
        'details': details,
        'score': 1 if all_pass else 0,
    }


# ────────────────────────────── 原有评分方法 ──────────────────────────────


def _render_history(messages: Sequence[dict[str, Any]], max_rounds: int | None = None) -> str:
    """
    渲染历史对话

    Args:
        messages: 对话消息列表
        max_rounds: 最多保留最近几轮对话（1轮 = 1组 user+assistant），None 表示不限制
    """
    if not messages:
        return '（无历史对话）'

    # 如果指定了 max_rounds，只保留最近的 N 轮（每轮约2条消息：user + assistant）
    if max_rounds is not None and max_rounds > 0:
        # 过滤掉 system 消息后，保留最近 max_rounds * 2 条
        filtered = [msg for msg in messages if msg.get('role') != 'system']
        keep_count = max_rounds * 2
        if len(filtered) > keep_count:
            filtered = filtered[-keep_count:]
    else:
        filtered = [msg for msg in messages if msg.get('role') != 'system']

    lines = []
    for msg in filtered:
        content = msg.get('content', '')
        if not content:
            continue
        role = msg.get('role', 'user')
        lines.append(f'{role}: {content}')
    return '\n'.join(lines) if lines else '（无历史对话）'


def _build_prompt(
    payload: dict[str, Any],
    checklist: list[dict[str, Any]],
    answer: str,
    max_answer_chars: int | None = None,
    max_history_chars: int | None = None,
) -> str:
    """
    构建评分提示词

    Args:
        payload: 包含用户问题、时间、历史对话等信息
        checklist: 评分清单
        answer: 模型回答
        max_answer_chars: answer 最大字符数限制（None 表示不限制）
        max_history_chars: history 最大字符数限制（None 表示不限制）
    """
    question = payload.get('user_question', '')
    date = payload.get('date') or '未知'
    # 默认只保留最近 5 轮对话
    history = _render_history(payload.get('messages') or [], max_rounds=5)

    # 格式化checklist
    checklist_text = json.dumps(checklist, ensure_ascii=False, indent=2)

    # 截断 history 和 answer（如果指定了限制）
    if max_history_chars is not None:
        history = _truncate_text(history, max_history_chars, '\n\n...[历史对话过长，已截断]...')
    if max_answer_chars is not None:
        answer = _truncate_text(answer, max_answer_chars, '\n\n...[回答内容过长，已截断]...')

    return SCORE_USER_PROMPT_TEMPLATE.format(
        question=question, date=date, history=history, checklist=checklist_text, answer=answer
    )


def _is_context_length_error(error: Exception) -> bool:
    """判断是否是上下文长度超限错误"""
    err_str = str(error)
    return any(
        keyword in err_str
        for keyword in [
            'context length',
            'input_tokens',
            'maximum context length',
            'reduce the length',
            'too many tokens',
            'token limit',
        ]
    )


async def _call_llm(prompt: str, max_retries: int = 3) -> dict[str, Any]:
    """
    调用 LLM 并解析响应，如果遇到空响应则重试直到成功

    Args:
        prompt: 提示词
        max_retries: 最大重试次数，默认 3 次

    Raises:
        _ContextLengthError: 当遇到上下文长度超限错误时抛出，由上层处理截断重试
    """
    last_content = None  # 保存最后一次 LLM 返回的原始内容
    for attempt in range(max_retries):
        try:
            response = await _call_judge_model([
                {'role': 'system', 'content': SCORE_SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt},
            ])
            content = response
            # print('raw_answer:', content)
            # 如果内容为空，重试
            if not content or not content.strip():
                print(f'[JudgeScore] 第 {attempt + 1} 次尝试：LLM 返回空响应，重试中...')
                continue
            last_content = content
            parsed = parse_json_block(content)
            parsed['_raw_content'] = content
            return parsed
        except ValueError as e:
            # 如果是空响应错误，重试
            if 'LLM response is empty' in str(e):
                print(f'[JudgeScore] 第 {attempt + 1} 次尝试：{str(e)}，重试中...')
                continue
            # JSON 解析失败，保存原始内容后重试
            print(f'[JudgeScore] 第 {attempt + 1} 次尝试：解析失败 ({str(e)})，重试中...')
            if attempt < max_retries - 1:
                continue
            # 最后一次仍然失败，返回包含原始内容的错误结果
            print('[JudgeScore] 所有重试均解析失败，保存原始模型结果')
            return {'_raw_content': last_content, '_parse_error': str(e)}
        except Exception as e:
            # 上下文长度超限错误：直接抛出，由上层 run() 方法处理截断重试
            if _is_context_length_error(e):
                print(f'[JudgeScore] 上下文长度超限，将尝试截断后重试: {str(e)[:100]}')
                raise
            # 其他错误（网络错误等），也重试
            print(f'[JudgeScore] 第 {attempt + 1} 次尝试：请求失败 ({str(e)})，重试中...')
            if attempt < max_retries - 1:
                continue
            # 最后一次仍然失败，如果有原始内容则保存
            if last_content:
                print('[JudgeScore] 所有重试均失败，保存原始模型结果')
                return {'_raw_content': last_content, '_parse_error': str(e)}
            raise

    # 所有重试都失败（全部是空响应的情况）
    raise ValueError(f'经过 {max_retries} 次重试后，LLM 仍然返回空响应')


def get_overall_score(checklist: list[dict[str, Any]]) -> float:
    """
    计算总分（扣分制）
    规则：
      - 基础分上限 = 0.9（need 得分率 * 0.9）
      - extension 加分空间 = 0.1（仅当 need 全部满足且未踩坑时生效，按 extension 满足比例分配）
      - 每个踩中的严重 pitfall 扣 0.3
      - 每个踩中的轻微 pitfall 扣 0.1
      - 总分上限为 1.0，下限为 0.0
    这样保证：
      - need 全满 + 无陷阱 + 无延展 = 0.9
      - need 全满 + 无陷阱 + 延展全满 = 1.0
      - pitfall 扣分不可被 extension 补回
    """
    need_items = [item for item in checklist if item.get('kind') == 'need']
    pitfall_items = [item for item in checklist if item.get('kind') == 'pitfall']
    extension_items = [item for item in checklist if item.get('kind') == 'extension']

    # 基础分：need 的加权得分率 * 0.9（基础分上限为 0.9）
    if need_items:
        total_weight = 0.0
        weighted_score = 0.0
        for item in need_items:
            # 字数约束条目（含 word_count_detail 字段）权重为 3，其他为 1
            weight = 3.0 if 'word_count_detail' in item else 1.0
            weighted_score += item.get('score', 0) * weight
            total_weight += weight
        need_score = (weighted_score / total_weight) * 0.9
    else:
        need_score = 0.9

    # pitfall 扣分：score=0 表示踩坑了
    pitfall_penalty = 0.0
    for item in pitfall_items:
        if item.get('score', 1) == 0:  # 踩坑了
            severity = item.get('severity', '')
            if severity == '严重':
                pitfall_penalty += 0.3
            else:
                pitfall_penalty += 0.1

    # extension 加分：仅当 need 全部满足且未踩中任何 pitfall 时才生效
    # 加分空间为 0.1，按 extension 满足比例分配
    extension_bonus = 0.0
    if need_items and need_score == 0.9 and pitfall_penalty == 0.0:
        if extension_items:
            satisfied_count = sum(1 for item in extension_items if item.get('score', 0) == 1)
            extension_bonus = 0.1 * (satisfied_count / len(extension_items))
        else:
            # 没有 extension 条目时，need 全满 + 无陷阱直接给满分
            extension_bonus = 0.1

    final_score = need_score - pitfall_penalty + extension_bonus
    return round(max(0.0, min(1.0, final_score)), 4)


def _is_word_count_rubric(item: dict[str, Any]) -> bool:
    """
    判断一个 rubric 条目是否是字数限制相关的条目。
    判断依据：条目中包含 length_range 字段。
    """
    return 'length_range' in item and item['length_range']


async def _score_word_count_rubrics(
    question: str,
    answer: str,
    word_count_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    对字数限制相关的 rubric 条目进行字数统计评分。

    Args:
        question: 用户提问
        answer: 模型回答
        word_count_items: 字数限制相关的 rubric 条目列表

    Returns:
        带有 score 的评分结果列表（与输入顺序一致）
    """
    # 收集所有字数范围要求
    length_ranges = []
    for item in word_count_items:
        lr = item['length_range']
        if isinstance(lr, list):
            length_ranges.extend(lr)
        else:
            length_ranges.append(lr)

    # 去重
    length_ranges = list(dict.fromkeys(length_ranges))

    # 调用字数统计核查
    try:
        word_count_result = await _check_word_count(question, answer, length_ranges)
    except Exception as e:
        print(f'[JudgeScore] 字数统计核查失败: {e}')
        # 失败时所有字数条目默认得 0 分
        scored_items = []
        for item in word_count_items:
            scored_items.append({
                'id': item.get('id', ''),
                'statement': item.get('statement', ''),
                'reason': f'字数统计核查失败: {e}',
                'score': 0,
                'kind': item.get('kind', 'need'),
                'word_count_detail': {'error': str(e)},
            })
        return scored_items

    # 根据核查结果为每个字数条目打分
    details = word_count_result.get('details', [])
    scored_items = []

    for item in word_count_items:
        lr = item['length_range']
        if isinstance(lr, list):
            # 多个范围要求，每个范围的所有片段都必须满足
            item_pass = True
            item_details = []
            for single_range in lr:
                # 找到对应的 detail
                matched_details = [d for d in details if d.get('length_range') == single_range]
                if matched_details:
                    # 所有片段都必须满足字数要求
                    range_pass = all(d.get('in_range', False) for d in matched_details)
                    if not range_pass:
                        item_pass = False
                    item_details.extend(matched_details)
                else:
                    # 没有找到对应的 detail，默认不通过
                    item_pass = False
                    item_details.append({'length_range': single_range, 'error': '未找到对应的核查结果'})
        else:
            # 单个范围要求：所有片段都必须满足字数要求
            matched_details = [d for d in details if d.get('length_range') == lr]
            if matched_details:
                item_pass = all(d.get('in_range', False) for d in matched_details)
                item_details = matched_details
            else:
                item_pass = False
                item_details = [{'length_range': lr, 'error': '未找到对应的核查结果'}]

        score = 1 if item_pass else 0
        pass_text = '通过' if item_pass else '未通过'
        reason = f'字数核查{pass_text}，详情: {json.dumps(item_details, ensure_ascii=False)}'

        scored_items.append({
            'id': item.get('id', ''),
            'statement': item.get('statement', ''),
            'reason': reason,
            'score': score,
            'kind': item.get('kind', 'need'),
            'word_count_detail': item_details,
        })

    return scored_items


async def run_writing_rubric(
    payload: dict[str, Any],
    checklist: list[dict[str, Any]],
    answer: str,
) -> dict[str, Any]:
    """
    执行评分

    Args:
        payload: 包含用户问题、时间、历史对话等信息的字典
        checklist: 评分清单，包含所有需要评分的考点
        answer: 学生回答

    Returns:
        包含评分结果的字典
    """
    question = payload.get('user_question', '')

    # ── 步骤1：将 checklist 拆分为普通条目和字数限制条目 ──
    normal_items = []  # 普通评分条目
    word_count_items = []  # 字数限制条目
    # 记录原始索引，用于最后合并时恢复顺序
    item_indices = []  # (original_index, "normal" | "word_count")

    for i, item in enumerate(checklist):
        if _is_word_count_rubric(item):
            word_count_items.append(item)
            item_indices.append((i, 'word_count', len(word_count_items) - 1))
        else:
            normal_items.append(item)
            item_indices.append((i, 'normal', len(normal_items) - 1))

    # ── 步骤2：对普通条目调用 LLM 评分（含上下文超限自动截断重试） ──
    normal_scored = []
    raw_content = None  # 保存原始模型结果
    if normal_items:
        # 首先尝试不截断的完整 prompt
        max_answer_chars = None
        max_history_chars = None
        max_truncate_retries = 3  # 最多截断重试 3 次

        for truncate_attempt in range(max_truncate_retries + 1):
            prompt = _build_prompt(
                payload, normal_items, answer, max_answer_chars=max_answer_chars, max_history_chars=max_history_chars
            )
            try:
                parsed = await _call_llm(prompt)
                raw_content = parsed.get('_raw_content')
                # 检查是否解析失败（有原始内容但无法解析为 JSON）
                if '_parse_error' in parsed:
                    print('[JudgeScore] JSON 解析失败，保存原始模型结果')
                    return {
                        'error': f'JSON解析失败: {parsed.get("_parse_error")}',
                        '_raw_content': raw_content,
                    }
                normal_scored = parsed.get('scored_checklist', [])
                break  # 成功，跳出截断重试循环
            except Exception as exc:
                if _is_context_length_error(exc) and truncate_attempt < max_truncate_retries:
                    # 上下文超限，计算截断参数后重试
                    # 估算当前 prompt 的 token 数
                    current_tokens = _estimate_tokens(prompt) + _estimate_tokens(SCORE_SYSTEM_PROMPT)
                    overflow_tokens = current_tokens - (MAX_CONTEXT_TOKENS - OUTPUT_RESERVE_TOKENS)
                    # 需要减少的字符数（按比例多减 20% 作为安全余量）
                    chars_to_reduce = int(overflow_tokens * CHARS_PER_TOKEN_ESTIMATE * 1.2)

                    # 优先截断 answer，其次截断 history
                    answer_len = len(answer)
                    history_text = _render_history(payload.get('messages') or [], max_rounds=5)
                    history_len = len(history_text)

                    if max_answer_chars is None:
                        # 第一次截断：先截断 answer
                        max_answer_chars = max(answer_len - chars_to_reduce, 5000)
                    elif max_history_chars is None:
                        # 第二次截断：再截断 history
                        max_answer_chars = max(int(max_answer_chars * 0.7), 3000)
                        max_history_chars = max(history_len - chars_to_reduce, 2000)
                    else:
                        # 第三次截断：进一步压缩
                        max_answer_chars = max(int(max_answer_chars * 0.6), 2000)
                        max_history_chars = max(int(max_history_chars * 0.5), 1000)

                    print(
                        f'[JudgeScore] 上下文超限，第 {truncate_attempt + 1} 次截断重试: '
                        f'max_answer_chars={max_answer_chars}, max_history_chars={max_history_chars}'
                    )
                    continue
                print(f'[JudgeScore] LLM error: {exc}')
                return {'error': str(exc)}

    # ── 步骤3：对字数限制条目调用字数统计核查 ──
    word_count_scored = []
    if word_count_items:
        word_count_scored = await _score_word_count_rubrics(question, answer, word_count_items)
        # print(f'[JudgeScore] 字数核查结果: {json.dumps(word_count_scored, ensure_ascii=False, indent=2)}')

    # ── 步骤4：按原始顺序合并结果 ──
    merged_scored = []
    for _orig_idx, item_type, sub_idx in item_indices:
        if item_type == 'normal':
            if sub_idx < len(normal_scored):
                merged_scored.append(normal_scored[sub_idx])
            else:
                # 兜底：LLM 返回的条目数不够
                merged_scored.append({
                    'id': normal_items[sub_idx].get('id', ''),
                    'statement': normal_items[sub_idx].get('statement', ''),
                    'reason': 'LLM 未返回该条目的评分',
                    'score': 0,
                })
        else:
            if sub_idx < len(word_count_scored):
                merged_scored.append(word_count_scored[sub_idx])
            else:
                merged_scored.append({
                    'id': word_count_items[sub_idx].get('id', ''),
                    'statement': word_count_items[sub_idx].get('statement', ''),
                    'reason': '字数核查未返回该条目的评分',
                    'score': 0,
                })

    # ── 步骤5：计算总分 ──
    # 为计算总分，需要将 kind 信息补充到 merged_scored 中
    for i, (orig_idx, _item_type, _sub_idx) in enumerate(item_indices):
        if 'kind' not in merged_scored[i]:
            merged_scored[i]['kind'] = checklist[orig_idx].get('kind', 'need')
        if 'severity' not in merged_scored[i]:
            merged_scored[i]['severity'] = checklist[orig_idx].get('severity', '')

    try:
        overall_score = get_overall_score(merged_scored)
    except Exception as e:
        print(f'[JudgeScore] 计算总分失败: {e}')
        overall_score = 0.0

    return {
        'scored_checklist': merged_scored,
        'overall_score': overall_score,
    }


async def writing_rubric_score(obj: dict[str, Any]) -> dict[str, Any]:
    """
    Rubric-based scoring function for the Writing topic.

    Uses the original JudgeScore.run path for all writing rubrics so normal
    rubrics and word-count rubrics share the same merge and scoring flow.

    Runs ``n_rounds`` judge rounds concurrently and averages the valid
    rounds, matching the shared ``yb_rubric_score_impl`` semantics. The
    round count follows the same validation config
    (``YB_RUBRIC_N_ROUNDS`` / ``YB_RUBRIC_N_ROUNDS_VAL``).
    """
    try:
        answer = str(obj.get('response', ''))
        checklist = obj.get('rubrics') or []
        payload = {
            'user_question': obj.get('prompt') or obj.get('question') or '',
            'messages': obj.get('history') or [],
            'date': obj.get('date') or '未知',
        }

        n_rounds = resolve_n_rounds(bool(obj.get('validate', False)))

        round_results = await asyncio.gather(
            *(run_writing_rubric(payload, checklist, answer) for _ in range(n_rounds)),
            return_exceptions=True,
        )

        round_scores: list[float] = []
        last_valid_scored_checklist: list[dict[str, Any]] | None = None
        last_error_result: dict[str, Any] | None = None
        for round_idx, result in enumerate(round_results):
            if isinstance(result, BaseException):
                last_error_result = {'error': f'round failed: {result!r}'}
                print(f'[JudgeScore] round {round_idx + 1}/{n_rounds} raised: {result!r}')
                continue

            scored_checklist = result.get('scored_checklist')
            reward = float(result.get('overall_score', -1.0))
            if 'error' in result or scored_checklist is None or reward < 0.0:
                last_error_result = result
                continue

            round_scores.append(reward)
            last_valid_scored_checklist = scored_checklist

        if not round_scores:
            failed = last_error_result if last_error_result is not None else {'error': 'all writing rounds failed'}
            obj['reward'] = -1.0
            obj['critic_resp'] = json.dumps(failed, ensure_ascii=False)
            obj['model_output'] = obj['critic_resp']
            obj['scoring_mode'] = 'yb_rubric_failed'
            obj['round_scores'] = []
            obj['scored_checklist'] = None
            obj['original_request'] = obj
            return obj

        overall_score = sum(round_scores) / len(round_scores)
        overall_score = max(0.0, min(1.0, overall_score))

        obj['reward'] = overall_score
        obj['critic_resp'] = json.dumps(
            {
                'n_rounds': n_rounds,
                'n_valid_rounds': len(round_scores),
                'round_scores': round_scores,
                'mean_score': overall_score,
                'scored_checklist': last_valid_scored_checklist,
            },
            ensure_ascii=False,
        )
        obj['model_output'] = obj['critic_resp']
        obj['scoring_mode'] = 'yb_rubric'
        obj['overall_score'] = overall_score
        obj['round_scores'] = round_scores
        obj['scored_checklist'] = last_valid_scored_checklist
        obj['original_request'] = obj
        return obj
    except Exception as exc:
        obj['reward'] = -1.0
        obj['critic_resp'] = f'Error in writing_rubric_score: {exc}'
        obj['model_output'] = obj['critic_resp']
        obj['scoring_mode'] = 'yb_rubric_failed'
        obj['scored_checklist'] = None
        obj['original_request'] = obj
        return obj


__all__ = ['writing_rubric_score']
