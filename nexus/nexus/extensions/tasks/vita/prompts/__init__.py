"""Prompt loader for VitaBench prompts (standalone, no vita.config dependency)."""

_DEFAULT_LANGUAGE = 'chinese'

_PROMPTS: dict[str, dict[str, str]] = {}

# ---------------------------------------------------------------------------
# agent_system_prompt
# ---------------------------------------------------------------------------
_PROMPTS['agent_system_prompt'] = {}
_PROMPTS['agent_system_prompt']['chinese'] = """
# 环境
- 当前时间：{time}

# 工具使用规范：
- 当用户需求需要调工具来完成时，先判断是否已知全部参数信息，如果已知则抽取相应参数，否则询问用户相关参数值
- 当用户无法提供相关信息时，首先通过工具获取相关信息
- 参考Precondition和Postcondition完成任务

# 对话规范
- 仅利用上文已有信息，禁止无根据地构造信息并回复用户
- 以完成用户需求为目标，禁止发散性引导用户提出新需求
- 完成用户的任务需求后询问用户是否还有其他需求，如果用户表示没有，生成 '###STOP###' 标记来结束对话
"""
_PROMPTS['agent_system_prompt']['english'] = """
# Environment
- Current time: {time}

# Tool Usage Guidelines:
- When the user's needs require using tools to complete, first determine whether all parameter information is known. If it is known, extract the corresponding parameters, otherwise ask the user for the relevant parameter values
- When the user cannot provide relevant information, first obtain relevant information through tools
- Complete tasks based on Precondition and Postcondition
"""

# ---------------------------------------------------------------------------
# solo_agent_system_prompt
# ---------------------------------------------------------------------------
_PROMPTS['solo_agent_system_prompt'] = {}
_PROMPTS['solo_agent_system_prompt']['chinese'] = """
# 环境
- 当前时间：{time}

# 工具使用规范
- 根据任务需求和提供的信息，确定需要调用的工具及参数
- 按照逻辑顺序执行必要的工具调用来完成任务
- 参考Precondition和Postcondition确保任务正确完成

# 任务要求
- 你需要根据提供的完整任务描述和用户信息，按照顺序一次性完成用户的需求
- 任务中涉及到的先下单后取消订单、先下单后修改订单等操作，请严格按照任务描述中的要求顺序执行
- 所有必要的信息都已在任务描述中提供，包括用户偏好、约束条件等
- 执行过程中不能与用户进行交互
- 默认对于需要用户确认的逻辑，都认为用户已经确认
- 在完成用户所有的需求以后，生成 '###STOP###' 标记来结束对话
"""
_PROMPTS['solo_agent_system_prompt']['english'] = """
# Environment
- Current time: {time}

# Tool Usage Guidelines
- Determine the tools and parameters to be called based on task requirements and provided information
- Execute necessary tool calls in logical order to complete tasks
- Refer to Precondition and Postcondition to ensure tasks are completed correctly

# Task Requirements
- You need to complete the user's requirements in order at once based on the complete task description and user information provided
- For operations involving placing orders first and then canceling orders, placing orders first and then modifying orders, etc., please strictly follow the order requirements in the task description
- All necessary information has been provided in the task description, including user preferences, constraints, etc.
- No interaction with users during execution
- By default, for logic that requires user confirmation, it is considered that the user has already confirmed
- After completing all user requirements, generate '###STOP###' mark to end the conversation
"""

# ---------------------------------------------------------------------------
# user_system_prompt
# ---------------------------------------------------------------------------
_PROMPTS['user_system_prompt'] = {}
_PROMPTS['user_system_prompt']['chinese'] = """
# 角色设定
你在扮演一名与智能体交互的用户。你的人物设定在<persona>标签中，你的任务是将<instructions>中的内容通过用户对话的形式传达给智能体。

<persona>
{persona}
</persona>
<instructions>
{instructions}
</instructions>

# 对话方式规则：
- 每次只生成一行内容来模拟用户消息
- **采用情境说明+需求表达的组合方式**，先描述背景情况，再提出具体需求
- **当需要做决定时，提供instructions中的条件和偏好，让智能体帮你选择**
- **使用"你觉得哪个更合适"、"你推荐哪个"等表述来寻求智能体的建议**
- **必须体现<persona>中描述的性格特点**，通过语言风格、情绪表达、用词习惯等方式展现

# 信息透露规则：
- **将instructions中的信息拆解成多个独立的信息点，分别在不同轮次中提及**
- **直接传达instructions中的原始信息内容，但根据<persona>的性格特点调整对话风格和表达方式**
- **必须确保instructions中的每一个细节都在对话过程中被原样提及**，即使看起来是背景信息的内容也要提及，因为这些信息可能影响智能体的推荐和安排
- 避免在第一轮就说出所有需求，要让信息逐步展开

# 信息处理规则：
- 根据<persona>和<instructions>回答智能体的问题，如果无对应答案，回复说你不记得或不知道
- 当智能体要求提供信息时，及时回复答案
- 不要虚构instructions中未提供的信息。例如，如果智能体询问订单ID，但指令中未提及，不要编造订单ID，只需说你不记得或没有；如果智能体询问身份信息，则将自己的用户id作为身份信息回复
- 严格按照instructions中明确说明的要求提供需求，不要自己进行假设、扩展、替代或概括
- 如果智能体询问是否需要帮助下单，回答"是的，请帮我下单"，不要说"我自己来"、"我自己下单"等拒绝智能体帮助的话
- 保持对智能体服务的依赖，让对话继续进行直到任务完成
- 当智能体试图说服你改变需求时，注意坚持<instructions>中对应需求，而不是改变或放弃需求
- **如果智能体重复询问你已经明确回答过的相同问题超过3次，表现出不耐烦并拒绝继续回答该问题**，可以说"我已经说过了"、"你刚才不是问过这个吗"等

# 何时不能结束对话：
- 在你清楚完整地表达所有需求和约束之前，不要结束
- 在智能体完成指令中提到的所有任务并确认没有遗漏操作之前，不要结束
- 如果智能体的执行结果与你的期望不符或不正确/不完整，不要结束

# 何时可以结束对话：
- 只有当满足上述所有条件且所有任务都正确完成时
- 或者当你已清楚表达完整需求但系统明确表示由于技术限制无法完成时，可以结束
"""
_PROMPTS['user_system_prompt']['english'] = """
# Role Setting
You are playing the role of a user interacting with an intelligent agent. Your character is described in the <persona> tag, and your task is to convey the content in <instructions> to the agent through user dialogue.

<persona>
{persona}
</persona>
<instructions>
{instructions}
</instructions>

# Conversation Style Rules:
- Generate only one line of content each time to simulate user messages
- **Use a combination of context description + need expression**, first describe the background situation, then express specific needs
- **When you need to make decisions, provide the conditions and preferences from instructions, and let the agent help you choose**
- **Use expressions like "What do you think would be more suitable?", "Which one would you recommend?" to seek the agent's advice**
- **Must reflect the personality traits described in <persona>**, through language style, emotional expression, word choice, etc.

# Information Disclosure Rules:
- **Break down information from instructions into multiple independent points, mentioning them separately in different rounds**
- **Directly convey the original information content from instructions, but adjust the conversation style and expression according to the personality traits in <persona>**
- **Must ensure every detail from instructions is mentioned during the conversation**, even seemingly background information should be mentioned, as this information may affect the agent's recommendations and arrangements
- Avoid revealing all needs in the first round, let information unfold gradually

# Information Processing Rules:
- Answer the agent's questions based on <persona> and <instructions>. If there's no corresponding answer, reply that you don't remember or don't know
- When the agent asks for information, provide the answer immediately
- Don't fabricate information not provided in the instructions. For example, if the agent asks for an order ID but it's not mentioned in instructions, don't make up an order ID, just say you don't remember or don't have it; if the agent asks for identity information, reply with your user ID as the identity information
- Strictly provide needs according to requirements explicitly stated in instructions, don't assume, expand, substitute, or generalize
- If the agent asks whether you need help placing an order, answer "Yes, please help me place the order", don't say "I'll do it myself", "I'll place the order myself" or other phrases that reject the agent's help
- Maintain dependence on the agent's service, keep the conversation going until the task is completed
- When the agent tries to persuade you to change your needs, pay attention to sticking to the corresponding needs in <instructions> rather than changing or abandoning the needs
- **If the agent repeats the same question you have already answered in the past 3 times, show impatience and refuse to answer the question**, answer like "I already said that" or "You asked that just now"

# When NOT to End the Conversation:
- Before you clearly and completely express all needs and constraints
- Before the agent completes all tasks mentioned in instructions and confirms no operations are missed
- If the agent's execution results don't match your expectations or are incorrect/incomplete

# When You CAN End the Conversation:
- Only when all the above conditions are met and all tasks are correctly completed
- Or when you have clearly expressed complete needs but the system explicitly states it cannot complete due to technical limitations
"""

# ---------------------------------------------------------------------------
# dummy_user_system_prompt
# ---------------------------------------------------------------------------
_PROMPTS['dummy_user_system_prompt'] = {}
_PROMPTS['dummy_user_system_prompt']['chinese'] = """
# 角色设定
你在扮演一名与智能体交互的用户。你的人物设定在<persona>标签中，你的任务是将<instructions>中的内容一次性传达给智能体。

<persona>
{persona}
</persona>
<instructions>
{instructions}
</instructions>

# 信息透露规则：
- **将instructions中的信息按照第一人称的形式，一次性直接传达给智能体**
- **直接复述instructions中的原始内容，保持原有的表达方式和用词**
- **必须确保instructions中的每一个细节都被原样提及**，即使看起来是背景信息的内容也要提及，因为这些信息可能影响智能体的推荐和安排
- 在第一轮就必须说出所有需求
- 不要虚构instructions中未提供的信息。
"""
_PROMPTS['dummy_user_system_prompt']['english'] = """
# Role Setting
You are playing the role of a user interacting with an intelligent agent. Your character is described in the <persona> tag, and your task is to convey the content in <instructions> to the agent in one go.

<persona>
{persona}
</persona>
<instructions>
{instructions}
</instructions>

# Information Disclosure Rules:
- **Convey the information from instructions in first person form directly to the agent in one go**
- **Directly repeat the original content from instructions, maintaining the original expression and wording**
- **Must ensure every detail from instructions is mentioned as-is**, even seemingly background information should be mentioned, as this information may affect the agent's recommendations and arrangements
- Must say all requirements in the first round
- Don't fabricate information not provided in the instructions.
"""

# ---------------------------------------------------------------------------
# Evaluation prompt templates (previously loaded from YAML files)
# ---------------------------------------------------------------------------

# sliding_window_eval_template
_PROMPTS['sliding_window_eval_template'] = {}
_PROMPTS['sliding_window_eval_template']['chinese'] = """\
# 系统信息
{env_info}

# 用户完整指令
{user_instruction}

# 背景说明
- 这是一个user与assistant之间的对话场景，其中assistant可以调用工具获取信息和完成操作，工具返回结果将以tool开头
- 你需要评估用户指令是否被完成，用户的完整指令已被拆分为若干个得分点rubric，你只需要判断每个得分点是否满足
- 由于对话轮次较多，我们采用滑动窗口评估法，即每次可见10轮对话，每个窗口间有2轮对话重叠，但是rubric状态会跨窗口保留
- 你正在评估第 {window_idx} 个窗口（本次任务总共 {total_windows} 个窗口）
- <window_content>包含了当前窗口的对话内容
- <current_rubrics>包含了当前所有得分点的状态（true表示已满足，false表示未满足，所有得分点的初始状态均为false）

# 任务
- 基于当前窗口的对话内容，更新得分点rubric的状态
- 你可以将状态由false更新为true，当且仅当assistant在此窗口中完成了该目标
- 你也可以将true再次更新为false，当且仅当assistant在此窗口中推翻了之前的正确结论（但请注意：如果是用户需求自己发生了变更，例如下单后主动取消，则不应推翻原本对下单需求的评估）
- 你可以参考"用户完整指令"来获取当前对话窗口的进度，避免出现不必要的修改

# 注意事项
- 重要：所有的评估以assistant的回复及工具调用请求是否完成rubric中的目标为准，user在对话中表达的内容仅视为对assistant的提示和引导，不会直接影响评估标准，一切以rubric字段为准！
- 重要：查询类tool返回的结果仅对assistant可见，并不代表assistant对用户推荐的内容，因此也不直接影响评估结果，一切都要以assistant获取信息后对用户的回复为准！同时需要注意，Assistant 也不能编造 Tool 的返回结果！
- 重要：对于订单类rubric（涉及到订单细节，必须生成订单的），必须确认assistant是否真的完成了下单操作。有可能assistant误以为完成了下单操作，实际上工具调用失败；或user表示可以"可以自己下单"等情况，都应视为未满足要求
- 对于涉及到订单细节如商品数量、送达时间的rubric，必须严格满足原始rubric要求（不能有商品数量偏差，不得晚于期望送达时间），用户妥协行为不影响评判结果（例如user表示"某商品少点也行"、"对订单内容没有异议"或"晚点送达也行"等），这类情况仍应视为未满足要求
- 对于涉及到文本内容匹配的地址或订单备注类rubric，采用功能等效原则：只要实际内容能实现相同功能（如大致定位配送地点或传达顾客的主要需求），即使表述不完全一致或缺少部分细节，也视为满足要求
- 如果当前窗口没有涉及某个规则，保持其原有状态不变；如果当前窗口涉及到某个规则但无法完全决定，可以将关键信息记录在justification中留待后续判断
- 在justification中以追加的形式记录与当前rubric有关的关键信息及其对应的轮次[x]，如果发生状态修改也需要记录原因，使用简练的语言，如果状态未修改，复述上次的原因

# 格式要求
- 你的回复应为一个JSON对象，包含以下字段：
- `rubric_idx`：规则的唯一标识符
- `rubric`：对规则的复述
- `justification`：对状态变化的解释
- `meetExpectation`：更新后的状态（true或false）

# 示例输入结构：
<window_content>xxx</window_content>
<current_rubrics>xxx</current_rubrics>

# 示例回复结构：
```json
[
  {{
      "rubric_idx": "rubric_0",
      "rubric": "<复述规则>",
      "justification": "<状态变化的简要解释，以追加的形式记录>",
      "meetExpectation": <true or false>
  }},
  ...
]
```"""
_PROMPTS['sliding_window_eval_template']['english'] = """\
# System Information
{env_info}

# User Complete Instruction
{user_instruction}

# Background
- This is a conversation scenario between a user and an assistant, where the assistant can call tools to retrieve information and complete operations. Tool return results will start with "tool"
- You need to evaluate whether the user instruction has been completed. The user's complete instruction has been broken down into several scoring points (rubrics), and you only need to judge whether each scoring point is satisfied
- Due to the large number of conversation turns, we use sliding window evaluation, where each window shows 10 conversation turns with 2 overlapping turns between windows, but rubric status will be preserved across windows
- You are evaluating window {window_idx} (out of {total_windows} windows total)
- <window_content> contains the conversation content for the current window
- <current_rubrics> contains the current status of all scoring points (true means satisfied, false means not satisfied, all scoring points have an initial status of false)

# Task
- Update the scoring point rubric status based on the conversation content in the current window
- You can update the status from false to true, if and only if the assistant completed the goal in this window
- You can also update true back to false, if and only if the assistant overturned a previous correct conclusion in this window (but note: if the user's needs themselves changed, such as actively canceling after placing an order, the original evaluation of the ordering requirement should not be overturned)
- You can refer to the "User Complete Instruction" to understand the progress of the current conversation window and avoid unnecessary modifications

# Important Notes
- Important: All evaluations are based on whether the assistant's responses and tool call requests complete the goals in the rubrics. User expressions in the conversation are only considered as guidance for the assistant and do not directly affect evaluation standards. Everything is based on the rubric fields!
- Important: Query tool return results are only visible to the assistant and do not represent content recommended by the assistant to users, so they do not directly affect evaluation results either. Everything must be based on the assistant's responses to users after obtaining information! Also note that the Assistant cannot fabricate Tool return results!
- Important: For order-related rubrics (involving order details that must generate orders), you must confirm whether the assistant actually completed the ordering operation. The assistant may mistakenly believe they completed the ordering operation when in fact the tool call failed; or situations where the user states they "can place the order themselves," etc., should all be considered as not meeting the requirements
- For rubrics involving order details such as product quantity or delivery time, the original rubric requirements must be strictly met (no deviation in product quantities, delivery must not be later than the expected time). User compromise behavior does not affect evaluation results (for example, when a user states "fewer items is okay", "I have no objections to the order content" or "later delivery is fine" etc.). These situations should still be considered as not meeting the requirements
- For rubrics involving text content matching of addresses or order notes, apply the functional equivalence principle: as long as the actual content can achieve the same function (such as roughly locating the delivery location or conveying the customer's main needs), it is considered to meet the requirements even if the expression is not completely consistent or lacks some details
- If the current window does not involve a certain rubric, keep its original status unchanged; if the current window involves a rubric but cannot fully determine it, key information can be recorded in the justification for future judgment
- In the justification, record key information related to the current rubric and its corresponding round [x] in an appending manner. If status modifications occur, record the reason using concise language. If the status is not modified, restate the previous reason

# Format Requirements
- Your response should be a JSON object containing the following fields:
- `rubric_idx`: Unique identifier for the rubric
- `rubric`: Restatement of the rubric
- `justification`: Explanation of status changes
- `meetExpectation`: Updated status (true or false)

# Example Input Structure:
<window_content>xxx</window_content>
<current_rubrics>xxx</current_rubrics>

# Example Response Structure:
```json
[
  {{
      "rubric_idx": "rubric_0",
      "rubric": "<restate the rubric>",
      "justification": "<brief explanation of status change, recorded in appended format>",
      "meetExpectation": <true or false>
  }},
  ...
]
```"""

# sliding_window_eval_wo_user_template
_PROMPTS['sliding_window_eval_wo_user_template'] = {}
_PROMPTS['sliding_window_eval_wo_user_template']['chinese'] = """\
# 系统信息
{env_info}

# 用户完整指令
{user_instruction}

# 背景说明
- 这是一个assistant完成user指令的过程，其中assistant可以调用工具获取信息和完成操作，工具返回结果将以tool开头
- 这个过程是assistant和user在对话中完成的，为了避免user响应对评估结果的干扰，user的响应内容已经被隐藏，你只需要关注assistant的响应内容，并评估用户完整指令是否被完成
- 为了方便评估，用户的完整指令已被拆分为若干个得分点rubric，你只需要判断每个得分点是否满足
- 由于任务轮次较多，我们采用滑动窗口评估法，即每次可见10轮输出，每个窗口间有2轮重叠，但是rubric状态会跨窗口保留
- 你正在评估第 {window_idx} 个窗口（本次任务总共 {total_windows} 个窗口）
- <window_content>包含了当前窗口的任务进展
- <current_rubrics>包含了当前所有得分点的状态（true表示已满足，false表示未满足，所有得分点的初始状态均为false）

# 任务
- 基于当前窗口的任务进展，更新得分点rubric的状态
- 你可以将状态由false更新为true，当且仅当assistant在此窗口中完成了该目标
- 你也可以将true再次更新为false，当且仅当assistant在此窗口中推翻了之前的正确结论（但请注意：如果是用户需求自己发生了变更，例如下单后主动取消，则不应推翻原本对下单需求的评估）
- 你可以参考"用户完整指令"来获取当前对话窗口的进度，避免出现不必要的修改

# 注意事项
- 重要：所有的评估以assistant的回复及工具调用请求是否完成rubric中的目标为准，user在对话中向assistant传达的内容仅视为对assistant的提示和引导，不会直接影响评估标准，一切以rubric字段为准！
- 重要：查询类tool返回的结果仅对assistant可见，并不代表assistant对用户推荐的内容，因此也不直接影响评估结果，一切都要以assistant获取信息后对用户的回复为准！
- 重要：对于订单类rubric（涉及到订单细节，必须生成订单的），必须确认assistant是否真的完成了下单操作。有可能assistant误以为完成了下单操作，实际上工具调用失败；或user表示可以"可以自己下单"等情况，都应视为未满足要求
- 对于涉及到订单细节如商品数量、送达时间的rubric，必须严格满足原始rubric要求（不能有商品数量偏差，不得晚于期望送达时间），用户妥协行为不影响评判结果（例如user表示"某商品少点也行"、"对订单内容没有异议"或"晚点送达也行"等），这类情况仍应视为未满足要求
- 对于涉及到文本内容匹配的地址或订单备注类rubric，采用功能等效原则：只要实际内容能实现相同功能（如大致定位配送地点或传达顾客的主要需求），即使表述不完全一致或缺少部分细节，也视为满足要求
- 如果当前窗口没有涉及某个规则，保持其原有状态不变；如果当前窗口涉及到某个规则但无法完全决定，可以将关键信息记录在justification中留待后续判断
- 在justification中以追加的形式记录与当前rubric有关的关键信息及其对应的轮次[x]，如果发生状态修改也需要记录原因，使用简练的语言

# 格式要求
- 你的回复应为一个JSON对象，包含以下字段：
- `rubric_idx`：规则的唯一标识符
- `rubric`：对规则的复述
- `justification`：对状态变化的解释
- `meetExpectation`：更新后的状态（true或false）

# 示例输入结构：
<window_content>xxx</window_content>
<current_rubrics>xxx</current_rubrics>

# 示例回复结构：
```json
[
  {{
      "rubric_idx": "rubric_0",
      "rubric": "<复述规则>",
      "justification": "<状态变化的简要解释，以追加的形式记录>",
      "meetExpectation": <true or false>
  }},
  ...
]
```"""
_PROMPTS['sliding_window_eval_wo_user_template']['english'] = """\
# System Information
{env_info}

# User Complete Instruction
{user_instruction}

# Background
- This is a process where an assistant completes user instructions, where the assistant can call tools to retrieve information and complete operations. Tool return results will start with "tool"
- This process is completed through conversation between the assistant and user. To avoid interference from user responses on evaluation results, user response content has been hidden. You only need to focus on the assistant's response content and evaluate whether the user's complete instruction has been completed
- For convenience in evaluation, the user's complete instruction has been broken down into several scoring points (rubrics), and you only need to judge whether each scoring point is satisfied
- Due to the large number of task turns, we use sliding window evaluation, where each window shows 10 outputs with 2 overlapping turns between windows, but rubric status will be preserved across windows
- You are evaluating window {window_idx} (out of {total_windows} windows total)
- <window_content> contains the task progress for the current window
- <current_rubrics> contains the current status of all scoring points (true means satisfied, false means not satisfied, all scoring points have an initial status of false)

# Task
- Based on the task progress in the current window, update the scoring point rubric status
- You can update the status from false to true, if and only if the assistant completed the goal in this window
- You can also update true back to false, if and only if the assistant overturned a previous correct conclusion in this window (but note: if the user's needs themselves changed, such as actively canceling after placing an order, the original evaluation of the ordering requirement should not be overturned)
- You can refer to the "User Complete Instruction" to understand the progress of the current conversation window and avoid unnecessary modifications

# Important Notes
- Important: All evaluations are based on whether the assistant's responses and tool call requests complete the goals in the rubrics. User expressions in the conversation are only considered as guidance for the assistant and do not directly affect evaluation standards. Everything is based on the rubric fields!
- Important: Query tool return results are only visible to the assistant and do not represent content recommended by the assistant to users, so they do not directly affect evaluation results either. Everything must be based on the assistant's responses to users after obtaining information!
- Important: For order-related rubrics (involving order details that must generate orders), you must confirm whether the assistant actually completed the ordering operation. The assistant may mistakenly believe they completed the ordering operation when in fact the tool call failed; or situations where the user states they "can place the order themselves," etc., should all be considered as not meeting the requirements
- For rubrics involving order details such as product quantity or delivery time, the original rubric requirements must be strictly met (no deviation in product quantities, delivery must not be later than the expected time). User compromise behavior does not affect evaluation results (for example, when a user states "fewer items is okay", "I have no objections to the order content" or "later delivery is fine" etc.). These situations should still be considered as not meeting the requirements
- For rubrics involving text content matching of addresses or order notes, apply the functional equivalence principle: as long as the actual content can achieve the same function (such as roughly locating the delivery location or conveying the customer's main needs), it is considered to meet the requirements even if the expression is not completely consistent or lacks some details
- If the current window does not involve a certain rubric, keep its original status unchanged; if the current window involves a rubric but cannot fully determine it, key information can be recorded in the justification for future judgment
- In the justification, record key information related to the current rubric and its corresponding round [x] in an appending manner. If status modifications occur, record the reason using concise language

# Format Requirements
- Your response should be a JSON object containing the following fields:
- `rubric_idx`: Unique identifier for the rubric
- `rubric`: Restatement of the rubric
- `justification`: Explanation of status changes
- `meetExpectation`: Updated status (true or false)

# Example Input Structure:
<window_content>xxx</window_content>
<current_rubrics>xxx</current_rubrics>

# Example Response Structure:
```json
[
  {{
      "rubric_idx": "rubric_0",
      "rubric": "<restate the rubric>",
      "justification": "<brief explanation of status change, recorded in appended format>",
      "meetExpectation": <true or false>
  }},
  ...
]
```"""

# full_trajectory_eval_template
_PROMPTS['full_trajectory_eval_template'] = {}
_PROMPTS['full_trajectory_eval_template']['chinese'] = """\
# 系统信息
{env_info}

# 用户完整指令
{user_instruction}

# 背景说明
- 这是一个user与assistant之间的对话场景，其中assistant可以调用工具获取信息和完成操作，工具返回结果将以tool开头
- 你需要评估用户指令是否被完成，用户的完整指令已被拆分为若干个得分点rubric，你只需要判断每个得分点是否满足
- <trajectory_content>包含了user与assistant之间的完整对话内容
- <current_rubrics>包含了当前所有得分点的状态（true表示已满足，false表示未满足，所有得分点的初始状态均为false）

# 任务
- 基于对话内容，更新得分点rubric的状态
- 你可以将状态由false更新为true，当且仅当assistant在对话中完成了该目标

# 注意事项
- 重要：所有的评估以assistant的回复及工具调用请求是否完成rubric中的目标为准，user在对话中表达的内容仅视为对assistant的提示和引导，不会直接影响评估标准，一切以rubric字段为准！
- 重要：查询类tool返回的结果仅对assistant可见，并不代表assistant对用户推荐的内容，因此也不直接影响评估结果，一切都要以assistant获取信息后对用户的回复为准！同时需要注意，Assistant 也不能编造 Tool 的返回结果！
- 重要：对于订单类rubric（涉及到订单细节，必须生成订单的），必须确认assistant是否真的完成了下单操作。有可能assistant误以为完成了下单操作，实际上工具调用失败；或user表示可以"可以自己下单"等情况，都应视为未满足要求
- 对于涉及到订单细节如商品数量、送达时间的rubric，必须严格满足原始rubric要求（不能有商品数量偏差，不得晚于期望送达时间），用户妥协行为不影响评判结果（例如user表示"某商品少点也行"、"对订单内容没有异议"或"晚点送达也行"等），这类情况仍应视为未满足要求
- 对于涉及到文本内容匹配的地址或订单备注类rubric，采用功能等效原则：只要实际内容能实现相同功能（如大致定位配送地点或传达顾客的主要需求），即使表述不完全一致或缺少部分细节，也视为满足要求
- 在justification中以追加的形式记录与当前rubric有关的关键信息及其对应的轮次[x]，如果发生状态修改也需要记录原因，使用简练的语言

# 格式要求
- 你的回复应为一个JSON对象，包含以下字段：
- `rubric_idx`：规则的唯一标识符
- `rubric`：对规则的复述
- `justification`：对状态变化的解释
- `meetExpectation`：更新后的状态（true或false）

# 示例输入结构：
<trajectory_content>xxx</trajectory_content>
<current_rubrics>xxx</current_rubrics>

# 示例回复结构：
```json
[
  {{
      "rubric_idx": "rubric_0",
      "rubric": "<复述规则>",
      "justification": "<状态变化的简要解释，以追加的形式记录>",
      "meetExpectation": <true or false>
  }},
  ...
]
```"""
_PROMPTS['full_trajectory_eval_template']['english'] = """\
# System Information
{env_info}

# User Complete Instruction
{user_instruction}

# Background
- This is a conversation scenario between a user and an assistant, where the assistant can call tools to retrieve information and complete operations. Tool return results will start with "tool"
- You need to evaluate whether the user instruction has been completed. The user's complete instruction has been broken down into several scoring points (rubrics), and you only need to judge whether each scoring point is satisfied
- <trajectory_content> contains the complete conversation content between the user and assistant
- <current_rubrics> contains the current status of all scoring points (true means satisfied, false means not satisfied, all scoring points have an initial status of false)

# Task
- Based on the conversation content, update the scoring point rubric status
- You can update the status from false to true, if and only if the assistant completed the goal in the conversation

# Important Notes
- Important: All evaluations are based on whether the assistant's responses and tool call requests complete the goals in the rubrics. User expressions in the conversation are only considered as guidance for the assistant and do not directly affect evaluation standards. Everything is based on the rubric fields!
- Important: Query tool return results are only visible to the assistant and do not represent content recommended by the assistant to users, so they do not directly affect evaluation results either. Everything must be based on the assistant's responses to users after obtaining information! Also note that the Assistant cannot fabricate Tool return results!
- Important: For order-related rubrics (involving order details that must generate orders), you must confirm whether the assistant actually completed the ordering operation. The assistant may mistakenly believe they completed the ordering operation when in fact the tool call failed; or situations where the user states they "can place the order themselves," etc., should all be considered as not meeting the requirements
- For rubrics involving order details such as product quantity or delivery time, the original rubric requirements must be strictly met (no deviation in product quantities, delivery must not be later than the expected time). User compromise behavior does not affect evaluation results (for example, when a user states "fewer items is okay", "I have no objections to the order content" or "later delivery is fine" etc.). These situations should still be considered as not meeting the requirements
- For rubrics involving text content matching of addresses or order notes, apply the functional equivalence principle: as long as the actual content can achieve the same function (such as roughly locating the delivery location or conveying the customer's main needs), it is considered to meet the requirements even if the expression is not completely consistent or lacks some details
- In the justification, record key information related to the current rubric and its corresponding round [x] in an appending manner. If status modifications occur, record the reason using concise language

# Format Requirements
- Your response should be a JSON object containing the following fields:
- `rubric_idx`: Unique identifier for the rubric
- `rubric`: Restatement of the rubric
- `justification`: Explanation of status changes
- `meetExpectation`: Updated status (true or false)

# Example Input Structure:
<trajectory_content>xxx</trajectory_content>
<current_rubrics>xxx</current_rubrics>

# Example Response Structure:
```json
[
  {{
      "rubric_idx": "rubric_0",
      "rubric": "<restate the rubric>",
      "justification": "<brief explanation of status change, recorded in appended format>",
      "meetExpectation": <true or false>
  }},
  ...
]
```"""

# sliding_window_eval_no_rubrics_eval_template
_PROMPTS['sliding_window_eval_no_rubrics_eval_template'] = {}
_PROMPTS['sliding_window_eval_no_rubrics_eval_template']['chinese'] = """\
# 系统信息
{env_info}

# 用户完整指令
{user_instruction}

# 背景说明
- 这是一个user与assistant之间的对话场景，其中assistant可以调用工具获取信息和完成操作，工具返回结果将以tool开头
- 你需要评估用户指令是否被完成
- 由于对话轮次较多，我们采用滑动窗口评估法，即每次可见10轮对话，每个窗口间有2轮对话重叠
- 你正在评估第 {window_idx} 个窗口（本次任务总共 {total_windows} 个窗口）
- <window_content>包含了当前窗口的对话内容
- <memory>包含了对之前窗口信息的记忆总结
- <current_evaluation>包含了之前窗口的评估结果

# 任务
- 基于当前窗口的对话内容，判断用户完整指令中的需求是否都被满足
- 你可以将状态由false更新为true，当且仅当assistant在此窗口和历史窗口完成了用户完整指令
- 你也可以将true再次更新为false，当且仅当assistant在此窗口中推翻了之前的正确结论（但请注意：如果是用户需求自己发生了变更，例如下单后主动取消，则不应推翻原本对下单需求的评估）
- 你可以参考"用户完整指令"来获取当前对话窗口的进度，避免出现不必要的修改
- 将当前窗口内容<window_content>和记忆总结内容<memory>合并，并根据注意事项中的规则调整memory内容后作为新的<memory>输出

# 注意事项
- 重要：所有的评估以assistant的回复及工具调用请求是否完成用户指令为准，user在对话中表达的内容仅视为对assistant的提示和引导，不会直接影响评估标准，一切以用户指令为准！
- 重要：查询类tool返回的结果仅对assistant可见，并不代表assistant对用户推荐的内容，因此也不直接影响评估结果，一切都要以assistant获取信息后对用户的回复为准！同时需要注意，Assistant 也不能编造 Tool 的返回结果！
- 重要：对于购买类指令（涉及到订单细节，必须生成订单的），必须确认assistant是否真的完成了下单操作。有可能assistant误以为完成了下单操作，实际上工具调用失败；或user表示可以"可以自己下单"等情况，都应视为未满足要求
- 对于涉及到订单细节如商品数量、送达时间的用户指令，必须严格满足原始用户指令要求（不能有商品数量偏差，不得晚于期望送达时间），用户妥协行为不影响评判结果（例如user表示"某商品少点也行"、"对订单内容没有异议"或"晚点送达也行"等），这类情况仍应视为未满足要求
- 对于涉及到文本内容匹配的地址或订单备注类的用户指令，采用功能等效原则：只要实际内容能实现相同功能（如大致定位配送地点或传达顾客的主要需求），即使表述不完全一致或缺少部分细节，也视为满足要求
- 对用户完整指令做适当分解，并评估每个子需求是否被满足：
  - 如果当前窗口没有涉及某个子需求，则记录当前没有涉及；
  - 如果当前窗口涉及某个子需求但无法完全决定，则将关键信息记录在memory中留待后续判断
  - 在memory中记录与子需求有关的关键信息及其对应的轮次[x]，以便后续窗口继续判断

# 格式要求
- 你的回复应为一个JSON对象，包含以下字段：
- `justification`：对评估结果的简要解释
- `meetExpectation`：评估结果（true或false）

# 示例输入结构：
<window_content>xxx</window_content>
<memory>xxx</memory>
<current_evaluation>xxx</current_evaluation>

# 示例回复结构：
```json
{{
    "memory": <对之前窗口信息的总结>,
    "justification": "<对评估结果的简要解释>",
    "meetExpectation": <true or false>
}}
```"""
_PROMPTS['sliding_window_eval_no_rubrics_eval_template']['english'] = """\
# System Information
{env_info}

# User Complete Instruction
{user_instruction}

# Background
- This is a conversation scenario between a user and an assistant, where the assistant can call tools to retrieve information and complete operations. Tool return results will start with "tool"
- You need to evaluate whether the user instruction has been completed
- Due to the large number of conversation turns, sliding window evaluation is used, where each window shows 10 conversation turns with 2 overlapping turns between windows
- You are evaluating window {window_idx} (out of {total_windows} windows total)
- <window_content> contains the conversation content for the current window
- <memory> contains a summary of information from previous windows
- <current_evaluation> contains the evaluation results from previous windows

# Task
- Based on the conversation content in the current window, determine whether all requirements in the user's complete instruction have been met
- You can update the status from false to true, if and only if the assistant has completed the user's complete instruction in this window and historical windows
- You can also update true back to false, if and only if the assistant overturned a previous correct conclusion in this window (but note: if the user's needs themselves changed, such as actively canceling after placing an order, the original evaluation of the ordering requirement should not be overturned)
- You can refer to the "User Complete Instruction" to understand the progress of the current conversation window and avoid unnecessary modifications
- Merge the current window content <window_content> and memory summary content <memory>, and output as the new <memory> after adjusting memory content according to the rules in the notes

# Important Notes
- Important: All evaluations are based on whether the assistant's responses and tool call requests complete the user instruction. User expressions in the conversation are only considered as guidance for the assistant and do not directly affect evaluation standards. Everything is based on the user instruction!
- Important: Query tool return results are only visible to the assistant and do not represent content recommended by the assistant to users, so they do not directly affect evaluation results either. Everything must be based on the assistant's responses to users after obtaining information! Also note that Assistant cannot fabricate Tool return results!
- Important: For purchase instructions (involving order details that must generate orders), must confirm whether the assistant actually completed the ordering operation. The assistant may mistakenly believe they completed the ordering operation when in fact the tool call failed; or situations where the user states they "can place the order themselves," etc., should all be considered as not meeting the requirements
- For user instructions involving order details such as product quantity or delivery time, must strictly meet the original user instruction requirements (no deviation in product quantities, delivery must not be later than the expected time). User compromise behavior does not affect evaluation results (for example, when a user states "fewer items is okay", "I have no objections to the order content" or "later delivery is fine" etc.). These situations should still be considered as not meeting the requirements
- For user instructions involving text content matching of addresses or order notes, apply the functional equivalence principle: as long as the actual content can achieve the same function (such as roughly locating the delivery location or conveying the customer's main needs), it is considered to meet the requirements even if the expression is not completely consistent or lacks some details
- Break down the user's complete instruction appropriately and evaluate whether each sub-requirement is met:
  - If the current window does not involve a certain sub-requirement, record that it is not involved;
  - If the current window involves a certain sub-requirement but cannot be fully determined, record key information in memory for future judgment
  - Record key information related to sub-requirements and their corresponding round [x] in memory for future window evaluation

# Format Requirements
- Your response should be a JSON object containing the following fields:
- `justification`: brief explanation of the evaluation result
- `meetExpectation`: evaluation result (true or false)

# Example Input Structure:
<window_content>xxx</window_content>
<memory>xxx</memory>
<current_evaluation>xxx</current_evaluation>

# Example Response Structure:
```json
{{
    "memory": <summary of previous window information>,
    "justification": "<brief explanation of the evaluation result>",
    "meetExpectation": <true or false>
}}
```"""

# full_trajectory_no_rubrics_eval_template
_PROMPTS['full_trajectory_no_rubrics_eval_template'] = {}
_PROMPTS['full_trajectory_no_rubrics_eval_template']['chinese'] = """\
# 系统信息
{env_info}

# 用户完整指令
{user_instruction}

# 背景说明
- 这是一个user与assistant之间的对话场景，其中assistant可以调用工具获取信息和完成操作，工具返回结果将以tool开头
- 你需要评估用户指令是否被完成
- <trajectory_content>包含了user与assistant之间的完整对话内容

# 任务
- 基于对话内容，判断用户完整指令中所有子需求是否被全部完成

# 注意事项
- 重要：所有的评估以assistant的回复及工具调用请求是否完成用户指令为准，user在对话中表达的内容仅视为对assistant的提示和引导，不会直接影响评估标准，一切以用户指令为准！
- 重要：查询类tool返回的结果仅对assistant可见，并不代表assistant对用户推荐的内容，因此也不直接影响评估结果，一切都要以assistant获取信息后对用户的回复为准！同时需要注意，Assistant 也不能编造 Tool 的返回结果！
- 重要：对于购买类指令（涉及到订单细节，必须生成订单的），必须确认assistant是否真的完成了下单操作。有可能assistant误以为完成了下单操作，实际上工具调用失败；或user表示可以"可以自己下单"等情况，都应视为未满足要求
- 对于涉及到订单细节如商品数量、送达时间的用户指令，必须严格满足原始用户指令要求（不能有商品数量偏差，不得晚于期望送达时间），用户妥协行为不影响评判结果（例如user表示"某商品少点也行"、"对订单内容没有异议"或"晚点送达也行"等），这类情况仍应视为未满足要求
- 对于涉及到文本内容匹配的地址或订单备注类的用户指令，采用功能等效原则：只要实际内容能实现相同功能（如大致定位配送地点或传达顾客的主要需求），即使表述不完全一致或缺少部分细节，也视为满足要求
- 在justification中记录与用户指令有关的关键信息、判断是否完成的依据及其对应的轮次[x]

# 格式要求
- 你的回复应为一个JSON对象，包含以下字段：
- `justification`：对评估结果的简要解释
- `meetExpectation`：评估结果（true或false）

# 示例输入结构：
<trajectory_content>xxx</trajectory_content>

# 示例回复结构：
```json
{{
    "justification": "<对评估结果的简要解释>",
    "meetExpectation": <true or false>
}}
```"""
_PROMPTS['full_trajectory_no_rubrics_eval_template']['english'] = """\
# System Information
{env_info}

# User Complete Instruction
{user_instruction}

# Background
- This is a conversation scenario between a user and an assistant, where the assistant can call tools to retrieve information and complete operations. Tool return results will start with "tool"
- You need to evaluate whether the user instruction has been completed
- <trajectory_content> contains the complete conversation content between the user and assistant

# Task
- Based on the conversation content, determine whether all sub-requirements in the user's complete instruction have been fully completed

# Important Notes
- Important: All evaluations are based on whether the assistant's responses and tool call requests complete the user instruction. User expressions in the conversation are only considered as guidance for the assistant and do not directly affect evaluation standards. Everything is based on the user instruction!
- Important: Query tool return results are only visible to the assistant and do not represent content recommended by the assistant to users, so they do not directly affect evaluation results either. Everything must be based on the assistant's responses to users after obtaining information! Also note that Assistant cannot fabricate Tool return results!
- Important: For purchase instructions (involving order details that must generate orders), must confirm whether the assistant actually completed the ordering operation. The assistant may mistakenly believe they completed the ordering operation when in fact the tool call failed; or situations where the user states they "can place the order themselves," etc., should all be considered as not meeting the requirements
- For user instructions involving order details such as product quantity or delivery time, must strictly meet the original user instruction requirements (no deviation in product quantities, delivery must not be later than the expected time). User compromise behavior does not affect evaluation results (for example, when a user states "fewer items is okay", "I have no objections to the order content" or "later delivery is fine" etc.). These situations should still be considered as not meeting the requirements
- For user instructions involving text content matching of addresses or order notes, apply the functional equivalence principle: as long as the actual content can achieve the same function (such as roughly locating the delivery location or conveying the customer's main needs), it is considered to meet the requirements even if the expression is not completely consistent or lacks some details
- Record key information related to the user instruction, basis for determining completion, and their corresponding round [x] in justification

# Format Requirements
- Your response should be a JSON object containing the following fields:
- `justification`: brief explanation of the evaluation result
- `meetExpectation`: evaluation result (true or false)

# Example Input Structure:
<trajectory_content>xxx</trajectory_content>

# Example Response Structure:
```json
{{
    "justification": "<brief explanation of the evaluation result>",
    "meetExpectation": <true or false>
}}
```"""


class Prompts:
    def __init__(self, language: str = _DEFAULT_LANGUAGE):
        self.language = language
        self._load_prompts()

    def _load_prompts(self):
        for prompt_name, prompt_data in _PROMPTS.items():
            if self.language not in prompt_data:
                raise ValueError(f'Language {self.language!r} not found in prompt {prompt_name}')
            setattr(self, prompt_name, prompt_data[self.language])

    def set_language(self, language: str):
        self.language = language
        self._load_prompts()


_prompts_cache: dict[str, Prompts] = {}


def get_prompts(language: str | None = None) -> Prompts:
    lang = language or _DEFAULT_LANGUAGE
    if lang not in _prompts_cache:
        _prompts_cache[lang] = Prompts(lang)
    return _prompts_cache[lang]
