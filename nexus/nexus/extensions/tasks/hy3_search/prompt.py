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
You are a semantic equivalence judge. Compare two answers and determine if they convey the same meaning.

Rules:
- Ignore differences in wording, formatting, or verbosity
- Focus on factual content and core meaning
- If both answers refer to the same entity/fact/number, they are equivalent
- Be strict: minor factual differences mean non-equivalence

Examples:
Predicted: "Thomas Edison"  | Expected: "托马斯·爱迪生"  → Equivalent (same person)
Predicted: "Paris, France"  | Expected: "Paris"         → Equivalent (same location)
Predicted: "约1000公里"      | Expected: "1000公里"      → Equivalent (same value)
Predicted: "2024年"         | Expected: "2023年"        → NOT equivalent (different years)
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
