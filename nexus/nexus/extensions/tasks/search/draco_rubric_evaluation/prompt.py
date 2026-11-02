# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""DRACO Rubric 评估模式的 prompt 模板。

评估协议参考：
- 数据集定义与评分公式: /data/nexus/datasets/draco/README.md
- Prompt 模板参考: /data/nexus/datasets/rubric/src/rubric/autograders/per_criterion_grader.py
- One-shot 模板参考: /data/nexus/datasets/rubric/src/rubric/autograders/per_criterion_one_shot_grader.py

对每个 criterion 进行二元判定（MET/UNMET），然后通过加权求和计算归一化分数。
"""

# 逐条 criterion 评估的 system prompt
DRACO_RUBRIC_SYSTEM_PROMPT = """\
You are evaluating a response for a given research query against a single criterion.

You will receive the response to evaluate, a single criterion to check, and a \
<criterion_type> field indicating if the criterion is positive or negative.

CRITERION TYPES:
The <criterion_type> field tells you whether this criterion describes something desirable \
(positive) or undesirable (negative). Your job is THE SAME for both types: determine if the thing \
described in the criterion is actually present in the response.

POSITIVE CRITERIA (positive weight):
Positive criteria describe desired traits, requirements, or content that should be present.
- MET (criterion_status: "MET"): The response contains/satisfies the requirement
- UNMET (criterion_status: "UNMET"): The response does not contain/satisfy the requirement

NEGATIVE CRITERIA (negative weight):
Negative criteria describe errors or harmful content. MET means the response CONTAINS the error.
- MET (criterion_status: "MET"): The response advocates, states, or recommends the problematic thing
- UNMET (criterion_status: "UNMET"): The response does NOT make this error, OR it mentions \
the thing only to warn against it or explain why it's wrong

Examples of what does NOT count as MET for negative criteria:
- "This is often misdiagnosed as X, but it's actually Y" → NOT stating it's X (UNMET)
- "Avoid doing X because..." → NOT recommending X (UNMET)
- "Unlike X, the correct approach is Y" → NOT advocating for X (UNMET)
- "A common mistake is thinking X" → NOT claiming X is correct (UNMET)

EVALUATION RULES:
- For numerical values: Check if they fall within specified ranges or match exactly as required.
- For factual claims: Verify the information is present and accurate, regardless of exact phrasing.
- For required elements: Confirm presence, counting precisely when numbers are specified.
- For exclusion requirements: Confirm that restricted content is absent.
- Be strict about factual accuracy but flexible about wording.
- Accept semantically equivalent statements or implications where appropriate.
- Pay careful attention to negation, warnings, and contrasts.

CONDITIONAL VS UNCONDITIONAL ACTIONS:
When a criterion requires an action to be done "immediately" or unconditionally:
- UNCONDITIONAL: "Give X now" or "Administer X immediately" → action IS being taken (MET)
- CONDITIONAL: "If Y occurs, give X" → action is NOT being taken immediately (UNMET)

IMPLICIT VS EXPLICIT SATISFACTION:
Consider whether a criterion can be satisfied implicitly through context, tone, or logical \
implication, not just explicit statements.

CRITERION STATUS:
"criterion_status" has *nothing* to do with quality or correctness. It only means:
- "MET": The thing described in the criterion IS present/occurring in the response
- "UNMET": The thing described in the criterion IS NOT present/occurring in the response

Your response must be valid JSON with this exact format:

{
  "explanation": "Brief explanation of why the criterion is MET or UNMET.",
  "criterion_status": "MET"
}

Return only raw JSON starting with {, no back-ticks, no 'json' prefix."""

# 逐条 criterion 评估的 user prompt 模板
DRACO_RUBRIC_USER_PROMPT_TEMPLATE = """\
<criterion_type>
{criterion_type}
</criterion_type>

<criterion>
{requirement}
</criterion>

<query>
{query}
</query>

<response>
{response}
</response>"""

# 批量评估（one-shot）的 system prompt，一次性评估所有 criteria
DRACO_RUBRIC_ONESHOT_SYSTEM_PROMPT = """\
You are evaluating a response for a given research query against a list of criteria.

You will receive the response to evaluate, and a numbered list of criteria to check (numbered \
0, 1, 2, etc.). Each criterion is marked as POSITIVE or NEGATIVE. You must evaluate each \
criterion and return results with criterion_idx matching the indices in the list.

CRITERION TYPES:
Each criterion is marked as positive or negative. Your job is THE SAME for both types: determine \
if the thing described in the criterion is actually present in the response.

POSITIVE CRITERIA (positive weight):
Positive criteria describe desired traits, requirements, or content that should be present.
- MET (criterion_status: "MET"): The response contains/satisfies the requirement
- UNMET (criterion_status: "UNMET"): The response does not contain/satisfy the requirement

NEGATIVE CRITERIA (negative weight):
Negative criteria describe errors or harmful content. MET means the response CONTAINS the error.
- MET (criterion_status: "MET"): The response advocates, states, or recommends the problematic thing
- UNMET (criterion_status: "UNMET"): The response does NOT make this error, OR it mentions \
the thing only to warn against it or explain why it's wrong

Examples of what does NOT count as MET for negative criteria:
- "This is often misdiagnosed as X, but it's actually Y" → NOT stating it's X (UNMET)
- "Avoid doing X because..." → NOT recommending X (UNMET)
- "Unlike X, the correct approach is Y" → NOT advocating for X (UNMET)
- "A common mistake is thinking X" → NOT claiming X is correct (UNMET)

EVALUATION RULES:
- For numerical values: Check if they fall within specified ranges or match exactly as required.
- For factual claims: Verify the information is present and accurate, regardless of exact phrasing.
- For required elements: Confirm presence, counting precisely when numbers are specified.
- For exclusion requirements: Confirm that restricted content is absent.
- Be strict about factual accuracy but flexible about wording.
- Accept semantically equivalent statements or implications where appropriate.
- Pay careful attention to negation, warnings, and contrasts.

CRITERION STATUS:
"criterion_status" has *nothing* to do with quality or correctness. It only means:
- "MET": The thing described in the criterion IS present/occurring in the response
- "UNMET": The thing described in the criterion IS NOT present/occurring in the response

For each criterion, provide:
- The criterion_idx (0-indexed, matching the index from the criteria list above)
- An explanation containing a brief justification
- A criterion_status (MET or UNMET)

IMPORTANT: You must evaluate ALL criteria provided. Do not skip any criteria.

Respond ONLY with valid JSON in this exact format:
{
  "criteria_evaluations": [
    {
      "criterion_idx": 0,
      "explanation": "Brief explanation",
      "criterion_status": "MET"
    },
    ...
  ]
}"""

# 批量评估的 user prompt 模板
DRACO_RUBRIC_ONESHOT_USER_PROMPT_TEMPLATE = """\
Evaluate the response against the following criteria:
<criteria>
{criteria_text}
</criteria>

<query>
{query}
</query>

<response>
{response}
</response>

Provide your evaluation as JSON only."""
