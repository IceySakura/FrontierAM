"""Judge prompt templates for Taiji Analysis evaluation."""

TAIJI_JUDGE_SYSTEM_PROMPT = """你是训练任务诊断报告的评审专家。
判断两份报告的根因结论和问题解决方案是否语义一致。

你需要从每份报告中自行提取核心根因结论（通常在"一句话总结"或开头段落），
然后判断两者是否指向同一问题。只看核心结论，忽略格式、详略和辅助信息。"""

TAIJI_JUDGE_PROMPT = """判断两份诊断报告的根因和解法是否一致。

## 用户问题
{question}

## 参考报告（完整）
{report_a}

## 待评估报告（完整）
{report_b}

## 评判标准
- 两份报告指向相同根因（如都说"OOM"或"节点故障"）→ 1
- 待评估报告包含参考的根因 + 额外正确分析 → 1
- 根因不同（如一个说 OOM，另一个说 NCCL hang）→ 0
- 待评估报告无有效结论或纯废话 → 0
- 待评估报告解决方案错误 → 0

请先从两份报告中各自提取根因结论，再做比较。

<result>
<reasoning>一句话理由</reasoning>
<is_consistent>1或0</is_consistent>
</result>"""
