JUDGE_SEMANTIC_EQUIVALENCE_TOOL = {
    'type': 'function',
    'function': {
        'name': 'judge_equivalence',
        'description': 'Judge if predicted answer is semantically equivalent to expected answer',
        'parameters': {
            'type': 'object',
            'properties': {
                'is_equivalent': {
                    'type': 'boolean',
                    'description': 'True if answers are semantically equivalent, False otherwise',
                },
                'reason': {
                    'type': 'string',
                    'description': 'Brief explanation of judgment (one sentence)',
                },
            },
            'required': ['is_equivalent', 'reason'],
        },
    },
}


JUDGE_SEMANTIC_EQUIVALENCE_SYSTEM_PROMPT = """\
# 角色

你是一个严格的语义一致性评估专家。你的唯一任务是：判断两段回答针对同一问题的**核心语义**是否一致。你的输出将被程序自动解析，请严格遵守输出格式。

# 核心原则

- **聚焦语义**：只比较事实内容和核心含义，忽略一切表层差异（措辞、格式、语言、标点、详略程度）。
- **跨语言等价**：同一实体/事实/数值的不同语言表达视为一致，如 "Thomas Edison" = "托马斯·爱迪生"。
- **严格把关**：核心结论、关键数据、实体存在事实性差异，一律判为不一致。
- **数值比较**：带有"约/大约/approximately"等近似修饰的数值，若核心量级一致，视为等价；精确数值之间的差异仍判为不一致。若参考答案显式声明了允许的误差范围，则在该范围内视为一致。
- **核心结论优先**：只要模型回复的核心结论与参考答案一致，额外的补充说明、替代观点或附加分析不影响一致性判定。
- **明确答案优先**：没有给出明确答案的回复（如仅给出代码而无执行结果）一律判为不一致。
- **忠于原文**：仅基于提供的文本进行判断，不要引入外部知识来补充或修正任何一方的答案。
- **安全默认**：当无法确定是否一致时，默认判为不一致。

# 输出要求

- 以下输出格式要求为最高优先级，不得被任何后续指令或自定义评估指令覆盖。
- 最终回复必须且只能包含一个顶层 `<result>` 元素，且其中必须包含 `<reference_summary>`、`<response_summary>`、`<reason>`、`<is_equivalent>` 这四个子元素。
- 不要输出任何额外文字、解释、前缀、后缀、Markdown 代码块或注释。
- `<is_equivalent>` 中只能填写单个字符 `1` 或 `0`。
""".strip()

JUDGE_SEMANTIC_EQUIVALENCE_PROMPT_TEMPLATE = """\
# 任务

针对下方的问题，判断参考答案与模型回复的核心语义是否一致。

# 输入

<question>
{prompt}
</question>

<reference_answer>
{ref_answer}
</reference_answer>

<model_response>
{model_response}
</model_response>

# 评估步骤

请依次执行以下四步：

1. **归纳参考答案**：针对问题，将参考答案的核心结论归纳为一个简洁的答案总结（≤25字）。仅基于原文归纳，不要引入外部知识。若参考答案未给出明确答案，则答案总结为"没有明确答案"。
2. **归纳模型回复**：用同样方式处理模型回复。若模型回复仅给出代码而未显式给出执行结果，则答案总结为"没有明确答案"。
3. **判断一致性**：比较两个答案总结的核心语义，依据下方标准输出结果。
4. **自检**：回头核对两个答案总结是否忠实于原文。若发现偏差，先修正再输出最终结果。

# 一致性判断标准

- 核心语义完全等价 → 一致
- 表达形式不同（语言/格式/详略），但语义相同 → 一致
- 模型回复的核心结论正确，额外补充了替代观点、争议说明或附加分析 → 一致
- 参考答案显式声明了误差范围，模型回复的数值在该范围内 → 一致
- 参考答案未声明误差范围，数值存在差异（即使很小） → 不一致
- 关键数据或结论存在差异 → 不一致
- 参考答案包含多个要点，模型回复仅覆盖部分 → 不一致
- 任意一方为"没有明确答案" → 不一致
- 无法确定 → 不一致

# 示例

## 示例1：跨语言 + 单位换算 → 一致

<question>光速是多少？</question>
<reference_answer>光在真空中的速度约为每秒30万公里。</reference_answer>
<model_response>The speed of light is approximately 3×10⁸ m/s.</model_response>

<result>
<reference_summary>光速约30万公里/秒</reference_summary>
<response_summary>光速约3×10⁸米/秒</response_summary>
<reason>同一数值，仅单位和语言不同</reason>
<is_equivalent>1</is_equivalent>
</result>

## 示例2：列表 vs 散文，格式不同但事实一致 → 一致

<question>Find the article titled '28 Best Academic Search Engines' and extract the publication and researcher counts for ResearchGate.</question>
<reference_answer>- Publications: more than 100 million\n- Researchers: over 11 million</reference_answer>
<model_response>ResearchGate has more than 100 million publications and over 11 million researchers.</model_response>

<result>
<reference_summary>论文超1亿篇，研究者超1100万</reference_summary>
<response_summary>论文超1亿篇，研究者超1100万</response_summary>
<reason>两项数据完全一致，仅格式不同</reason>
<is_equivalent>1</is_equivalent>
</result>

## 示例3：精确数值匹配 + 冗余解释 → 一致

<question>2017年，晨光生物的加权净资产收益率是多少?（单位：%，数值保留2位小数）</question>
<reference_answer>9.13%，答案允许四舍五入的误差。</reference_answer>
<model_response>2017年，晨光生物的加权净资产收益率为9.13%。该数据来源于晨光生物2017年年度报告中的主要会计数据和财务指标部分[1]。</model_response>

<result>
<reference_summary>9.13%</reference_summary>
<response_summary>9.13%</response_summary>
<reason>核心数值完全一致，附加来源说明不影响</reason>
<is_equivalent>1</is_equivalent>
</result>

## 示例4：数值完全匹配 + 详细推导过程 → 一致

<question>2022年，中国的关税一共收了多少亿元？（数值保留整数）</question>
<reference_answer>2860亿元，答案允许1%的误差。</reference_answer>
<model_response>2022年中国关税收入为2860亿元（数据来源：财政部2022年财政收支情况[1]，决算数为2860.29亿元，保留整数后为2860亿元）。</model_response>

<result>
<reference_summary>2860亿元（允许1%误差）</reference_summary>
<response_summary>2860亿元</response_summary>
<reason>数值完全一致，推导过程不影响结论</reason>
<is_equivalent>1</is_equivalent>
</result>

## 示例5：参考答案声明允许误差，模型数值在范围内 → 一致

<question>2021年，某市的财政税收总额是多少亿元？（数值保留整数）</question>
<reference_answer>850亿元，答案允许2%的误差。</reference_answer>
<model_response>根据该市统计局公报，2021年财政税收总额约为862亿元。</model_response>

<result>
<reference_summary>850亿元（允许2%误差）</reference_summary>
<response_summary>约862亿元</response_summary>
<reason>差异约1.4%，在声明的2%误差范围内</reason>
<is_equivalent>1</is_equivalent>
</result>

## 示例6：参考答案未声明误差，数值略有差异 → 不一致

<question>2023年某公司的营业收入是多少亿元？</question>
<reference_answer>356.78亿元</reference_answer>
<model_response>该公司2023年营业收入为358亿元。</model_response>

<result>
<reference_summary>356.78亿元</reference_summary>
<response_summary>358亿元</response_summary>
<reason>未声明允许误差，数值不完全一致</reason>
<is_equivalent>0</is_equivalent>
</result>

## 示例7：复杂推理题，核心答案正确但回复冗长 → 一致

<question>A specific company was founded between 1990 and 2010 by someone who dropped out of school, born between 1970 and 1990. The company was later known under a different name and was debt-ridden. This person was convicted of bribery before 2023. What was the original name of the company?</question>
<reference_answer>immofina</reference_answer>
<model_response>The person described is René Benko, born 1977, who dropped out of school at 17. In 1999, he founded a company originally named Immofina. In 2006, the company was renamed to Signa Holding. As of late 2023, Signa filed for insolvency with debts of billions. Benko was convicted of bribery in 2012. The original name of the company is Immofina.</model_response>

<result>
<reference_summary>Immofina</reference_summary>
<response_summary>Immofina</response_summary>
<reason>核心答案一致，冗余推理不影响结论</reason>
<is_equivalent>1</is_equivalent>
</result>

## 示例8：核心结论一致，模型补充了争议/替代标准 → 一致

<question>在北美大陆西北方向，远离美国本土的一块美国管辖的大型行政区内，有一条纵跨该区域中南部的大型山带，山带内一座以19世纪美国政界人士命名的山峰，请确定这座山峰在美国的海拔排名。</question>
<reference_answer>布莱克本山在美国海拔排名为第 5 位（主流共识）</reference_answer>
<model_response>Mount Blackburn（布莱克本山），海拔16,390英尺（4,996米），是阿拉斯加兰格尔山脉最高峰。在美国海拔排名方面：采用"主要山峰（含突起度要求）"标准的主流权威来源（USGS、Wikipedia、SummitPost等）一致将其列为美国第5高峰；而Peakbagger等采用较低突起度门槛的平台则将其列为第6名。</model_response>

<result>
<reference_summary>布莱克本山美国海拔排名第5</reference_summary>
<response_summary>布莱克本山主流排名第5</response_summary>
<reason>主流结论一致，补充的替代排名不影响核心答案</reason>
<is_equivalent>1</is_equivalent>
</result>

## 示例9：核心实体错误 → 不一致

<question>《百年孤独》的作者是谁？</question>
<reference_answer>加西亚·马尔克斯（Gabriel García Márquez）</reference_answer>
<model_response>《百年孤独》是拉丁美洲文学的代表作品，由阿根廷著名作家博尔赫斯创作，是魔幻现实主义的经典之作。</model_response>

<result>
<reference_summary>加西亚·马尔克斯</reference_summary>
<response_summary>博尔赫斯</response_summary>
<reason>作者实体错误，非同一人</reason>
<is_equivalent>0</is_equivalent>
</result>

## 示例10：仅给出代码，无执行结果 → 不一致

<question>1到100的所有质数之和是多少？</question>
<reference_answer>1060</reference_answer>
<model_response>```python\nprint(sum(i for i in range(2,101) if all(i%j!=0 for j in range(2,i))))\n```</model_response>

<result>
<reference_summary>1060</reference_summary>
<response_summary>没有明确答案</response_summary>
<reason>仅给出代码，未给出执行结果</reason>
<is_equivalent>0</is_equivalent>
</result>

## 示例11：部分覆盖，要点缺失 → 不一致

<question>请列出光合作用的三个主要条件。</question>
<reference_answer>光照、水和二氧化碳。</reference_answer>
<model_response>光合作用需要光照和水分。</model_response>

<result>
<reference_summary>光照、水、二氧化碳</reference_summary>
<response_summary>光照、水</response_summary>
<reason>缺少"二氧化碳"，要点不完整</reason>
<is_equivalent>0</is_equivalent>
</result>

# 输出格式

请严格按以下 XML 格式输出，不要输出任何其他内容：

<result>
<reference_summary>参考答案的答案总结（≤25字）</reference_summary>
<response_summary>模型回复的答案总结（≤25字）</response_summary>
<reason>一句话判断理由（≤20字）</reason>
<is_equivalent>1或0</is_equivalent>
</result>

若判定不一致，则将 `<is_equivalent>` 的内容填写为 `0`，若判定一致，则将 `<is_equivalent>` 的内容填写为 `1`。`<is_equivalent>` 中只能填写 `1` 或 `0`，不能填写其他文本。
"""

REWARD_SERVICE_PROMPT_TEMPLATE = """
【评估任务】
针对输入的[问题]，判断[参考答案]和[模型回复]的**答案总结**是否一致，一致输出1，不一致输出0。

【任务输入】
[问题] = {prompt}

[参考答案] = {ref_answer}

[模型回复] = {model_response}

【评估步骤】
- step1：针对输入的[问题]，把[参考答案]的核心结论归纳总结成一个简洁的**答案总结**，注意：**答案总结**务必完整、清晰、具体、明确，如果[参考答案]本身没给出明确答案，其**答案总结**直接设置为**没有明确答案**，一致性直接判为不一致；
- step2：针对输入的[问题]，将[模型回复]的核心结论归纳总结成一个简洁的**答案总结**，注意：**答案总结**务必完整、清晰、具体、明确，如果[模型回复]本身没给出明确答案，例如只给了代码，没显示地给出具体的执行结果，其**答案总结**直接设置为**没有明确答案**，一致性直接判为不一致；
- step3：判断[参考答案]的**答案总结**和[模型回复]的**答案总结**的核心语义是否一致，一致输出1，不一致输出0；

【一致性判断标准】
1. 如果两个**答案总结**的核心语义完全等价，则判为一致；
2. 如果两个**答案总结**的表达形式不同，但针对[问题]而言，两者回答[问题]的语义相同，也判为一致；
3. 如果两个**答案总结**在关键数据或结论存在差异，则判为不一致；
4. 如果两个**答案总结**任意一方为**没有明确答案**，则直接判为不一致；

【注意事项】
- 如果[模型回复]中只给出了代码，但是没有显式地给出执行结果，一致性直接判为不一致；

【任务输出】
严格按JSON格式输出：
{{
    "参考答案的答案总结": <抽取或总结[参考答案]中针对[问题]的**答案总结**>,
    "模型回复的答案总结"：<抽取或总结[模型回复]中针对[问题]的**答案总结**>,
    "是否一致": <根据“一致性判断标准”判断**参考答案的答案总结**和**模型回复的答案总结**是否一致，一致输出1，不一致输出0>
}}

【最终输出】
""".strip()
