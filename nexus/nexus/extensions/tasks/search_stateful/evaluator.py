import json
import logging
import re

from collections.abc import Sequence
from typing import Any

from nexus.core import require_context
from nexus.extensions.agents.base.evaluation import Evaluator
from nexus.utils.reward_score import ReasoningReward
from nexus.utils.reward_system import (
    RewardSystemClient,
    build_user_content,
    resolve_strip_multimodal_config,
)


# Judge 结果里 ``JudgeResult.error`` 字段的前缀约定：
# - 以 ``JUDGE_INFRA_ERROR_PREFIX`` 开头 → judge 基础设施故障（网络/鉴权/5xx/服务端异常等），
#   task 层看到应向上抛异常，避免被静默计入 0 分.
# - 以 ``JUDGE_CONTENT_FILTER_ERROR_PREFIX`` 开头 → judge 请求被内容审核策略拦截，属于合法的 0 分.
JUDGE_INFRA_ERROR_PREFIX = 'infra_error: '
JUDGE_CONTENT_FILTER_ERROR_PREFIX = 'content_filtered: '


_THINK_RE = re.compile(r'<think>.*?</think>\s*', re.DOTALL)
_OSS_MATCH_RE = re.compile(r'<match_result>\s*(consistent|inconsistent)\s*</match_result>', re.IGNORECASE)


logger = logging.getLogger(__name__)


# Official HLE benchmark judge prompt (expects JSON output).
HLE_JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
You must output your judgement in the following JSON format:
```json
{{"extracted_final_answer": "...", "reasoning": "...", "correct": "yes or no", "confidence": 0-100}}
```"""


# General evaluator template with detailed Chinese rules
GENERAL_JUDGE_PROMPT = """
给你一个问题的两个答案（可能不完整,最终答案一般在最后），对于一个问题的每个子问题（可能一个，可能多个），你需要抽取两个答案的精简回答res1和res2，根据以下规则做一些转换，然后依据[答案一致性判断规则]比较两个答案是否一致，并给出简短的判断原因。
0.问题拆分
0.1 若问题为多选题，答案1为多个选项，不要将多选题拆分为多个子问题。若问题是选择题但答案不是选项，精简答案应该为非选项答案对应到题目中的选项。
0.2 若答案为范围或者区间，整个范围或者区间是一个完整的答案（子问题类型为公式），不要拆分成多个子问题。
1. 子问题类型为公式、数值、选项、简答其中之一。化学表达式算是公式的一种。
2. 若回答的精简内容为选项，精简答案的形式为选项的拼接（字符串形式，多个选项用,分割）。举个例子，若答案为B和C，res为"B,C"。
3. 若回答的精简内容为公式，需要将精简答案转化为latex格式，即res1和res2为\\boxed{{...}}的格式。注意res1和res2不要带单位。
4. 若回答的精简内容为数值，需要考虑单位情况，将单位统一后二者的数值分别放到res1和res2。注意res1和res2不要带单位。
4.1 若两个答案都不带单位或者都带相同的单位，直接抽取。
4.2 若两个答案都带单位，但单位不同，需要按照答案1的单位，对答案2的结果进行单位转换，将答案1和答案2的单位统一。
4.3 若答案1没带单位，默认是题目给定的单位；若答案2没带单位，可以结合完整的答案。然后结合4.2看是否需要转化。
5. 若答案2的最终答案在之前被四舍五入过，则抽取四舍五入之前的那个答案，即精度选更高的一个答案
6. 若回答的精简内容为数值，必须将其转化为latex格式，即res1和res2为\\boxed{{...}}的格式。若其中包含分数、根式、指数、对数、科学计数法等，不要计算为小数，也不要进行化简约分，直接放\\boxed{{...}}里面。若原来为小数或者整数，也放\\boxed{{...}}里，数值不要改变。
7. 若回答为空或者无意义内容，或者结尾处被截断，没有有效答案，精简内容为空字符串
8. 若题目中有小问是证明题，抽取答案1和答案2中对应证明题的小问答案，如果没有则精简内容为空字符串，统一打分为0
9. 依据下面给出的[答案一致性判断规则]，比较res1和res2是否一致，若一致分数为1，否则为0，并给出简短的判断原因。
**注意**：这里的一致，指的是res1和res2在实际含义上的一致性，而不是字面上的完全一致。

[答案一致性判断规则]
- 若答案类型为"选项"，则比较两个答案的选项内容是否一致，如果是多选，则忽略选项的顺序
- 如果其中一个答案类型为"数值"，需要根据题目要求，判断进行一致性比较时，是否需要考虑精度
  - 若题目中有明确的精度要求，则需要比较两个数值，或数值与公式的计算结果在给定精度上是否完全一致
    - 比较时，若答案1和答案2的单位不同，则需要先将两者的单位统一，然后再进行精度比较
  - 若题目中没有明确的精度要求，则只需要比较两个数值，或数值与公式的计算结果是否一致时，可以允许一定程度的精度差异，如"2.34"和"2.342"可以认为是相同的，只要符合四舍五入的规则即可。但是要注意，如果答案的有效数字小于3位，不能直接进行四舍五入判断，例如0.6和1，二者差距过大，不能认为是相同的
- 若答案类型为"数值"或"公式"。
  - 需要在单位统一的前提下，比较两个答案的数值或公式内容是否等价，即数值是否相等，公式之间是否可以相互转换。如果一个是数值，另一个是公式，则公式的计算结果是否等于数值
  - 若答案1和答案2只有正负号不一样，根据题意判断正负号是否影响正确性，例如如果题目要求的是距离、长度等非负量，则正负号不影响正确性，认为两者一致；如果题目要求的是位移、功等可能为负的量，则正负号影响正确性，认为两者不一致。
  - 若答案1和答案2都是公式，且答案2是答案1的通解形式，在题目没有要求必须使用特殊解的情况下，则认为两者一致。
- 若答案类型为"简答"，则比较两个答案的内容在语义上是否一致，忽略字面上的差异
- 如果答案属于"化学表达式"，则需要依据化学知识判断它们是否等价，比如化学反应方程式的平衡、化学物质的状态等
- 若答案是范围或者区间，需要按照数学中的区间正确性来判定

你的输出需要是以下json列表[{{"类型": 类型, "res1": res1, "res2": res2, "单位": 单位(可以为空字符串), "分数": score, "判断原因": xxx}}, ...]的格式，其中每一个json表示一个子问题。

问题：
{question}


答案1：
{ref_answer}


答案2：
{answer}
"""


# Olympiad judge prompt — outputs VERDICT: CORRECT/INCORRECT
OLYMPIAD_JUDGE_PROMPT = """You are grading an attempted answer to a science olympiad problem. You will be given the
problem, attempted answer, and reference answer. Evaluate the solution against the provided
reference solution, ensuring it is complete and matches the reference solution. Pay close
attention to detail and grade it strictly, but fairly.
The reference answer is either a single number or expression in latex formatting, a chemical
formula, a compound name, or a phrase referring to a specific name, entity, or method.
Mark the attempted answer as correct if it fully matches the reference answer or is otherwise
equivalent (e.g., an equivalent algebraic expression, a numerical number within 1 decimal
place rounding of the reference answer (e.g., 6.69 ≈ 6.7), an equivalent name for a compound/formula, equivalent when accounting for units, etc.). Mark it as incorrect if it is not
equivalent to the reference answer.
***
The problem: {question}
***
The reference answer: {correct_answer}
***
The attempted answer: {response}
***
First, think step-by-step about whether the attempted answer matches the reference answer.
If the attempted answer is correct, write "VERDICT: CORRECT" in the last line of your
response, with no other text or formatting. If it is incorrect, write "VERDICT: INCORRECT"."""


# Research rubric judge prompt — outputs VERDICT: total_points (out of 10)
RESEARCH_JUDGE_PROMPT = """You are grading a science exam.
You will be given the problem, attempted answer, and a rubric to grade the answer. The rubric
will total up to 10 points.
Evaluate the attemped answer against the provided rubric. Pay close attention to detail and
grade it strictly, but fairly. Only evaluate against the rubric, as you yourself should not make
any judgements (e.g., even if you think the answer is correct but rubric is wrong, you should
treat the rubric as the gold standard). Return the absolute total number of points earned (it can
be a decimal based on the rubric). ***
The problem: {question}
***
The rubric: {correct_answer}
***
The attempted answer: {response}
***
First, think step-by-step about each rubric item. Explain your reasoning for each rubric item.
Then, tally the points up and write VERDICT: total_points in the last line of your response,
no other text. For example, VERDICT: 2.5 or VERDICT: 8."""


# OSS math judge prompt — outputs <match_result>consistent/inconsistent</match_result>
OSS_JUDGE_PROMPT = """You are an expert mathematician and a meticulous result verifier.

You will be provided with a math problem, its correct reference answer, and a candidate solution. Your objective is to determine if the conclusion reached in the candidate solution matches the provided reference answer.

**1. Core Instructions**
*   You must act as a **verifier**, NOT a solver. **Do NOT attempt to solve the math problem or correct errors in the candidate solution.**

**2. Criteria**
*   First, read the original math problem carefully to understand its given conditions and what needs to be solved.
   Then, read the reference answer and the candidate solution carefully to note their final conclusions.
   Only when you fully understand the original problem, the candidate solution, and the reference answer can you conduct further analysis and judgment.
*   The candidate solution is considered consistent if its final conclusion is either identical to the reference answer or has the same meaning. Otherwise, it is inconsistent.
*   For problems with multiple sub-questions, a solution is only considered consistent if the answers to all sub-questions match the reference answer in meaning. If the answer to any sub-question is different or missing, the solution is inconsistent.
*   Differences in form-like those from calculation precision, symbol substitution, or format changes/simplifications-are still considered the same, as long as the result follows mathematical rules (e.g., rounding).

**3. Output Format**
After your analysis is completed, please start a new xml-section '<match_result>' to give your final conclusion and stop generation immediately.

- <match_result>: This section should present the matching result, with **content limited to "consistent" or "inconsistent" **, reflecting whether the solution's final conclusion aligns with the reference answer.

An example of <match_result> format is shown below:
[output format example]
<match_result>
consistent/inconsistent
</match_result>

======================================================================
The following is the math problem and solution.

[Math Problem]
{question}

[Reference Answer]
{correct_answer}

[Solution]
{response}

**Verification Task Reminder**
Your task is to act as a result verifier. Your Response MUST be consisted of <match_result> as described above."""


class JudgeResult:
    """Judge evaluation result with detailed information."""

    def __init__(
        self,
        score: float,
        judge_prompt: str,
        judge_response: str,
        parsed_result: dict | None,
        error: str | None = None,
    ):
        self.score = score
        self.judge_prompt = judge_prompt
        self.judge_response = judge_response
        self.parsed_result = parsed_result
        self.error = error

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'score': self.score,
            'judge_prompt': self.judge_prompt,
            'judge_response': self.judge_response,
            'parsed_result': self.parsed_result,
            'error': self.error,
        }


def _strip_think_blocks(prediction: str) -> str:
    return _THINK_RE.sub('', prediction).strip()


def _log_judge_input(prefix: str, question: str | None, prediction: str, reference: str, prompt: str) -> None:
    logger.debug('=' * 80)
    logger.debug(f'[{prefix}] Input:')
    logger.debug(f'  Question: {question[:200]}...' if question and len(question) > 200 else f'  Question: {question}')
    logger.debug(f'  Prediction: {prediction[:200]}...' if len(prediction) > 200 else f'  Prediction: {prediction}')
    logger.debug(f'  Reference: {reference[:200]}...' if len(reference) > 200 else f'  Reference: {reference}')
    logger.debug(f'[{prefix}] Full Prompt:\n{prompt}')


# 命中该完整原句时说明 judge 请求被 Azure OpenAI 内容审核策略拦截：
# 这类"被过滤"是模型/平台的合法输出，应该记为 0 分，而不是判分基础设施故障.
# 使用完整原句做精确子串匹配（大小写敏感），避免误伤其他 4xx 错误.
CONTENT_FILTER_MARKER = (
    "The response was filtered due to the prompt triggering Azure OpenAI's content management policy"
)


def _is_content_filter_error(exc: BaseException) -> bool:
    """判断异常是否是 Azure OpenAI 的内容策略过滤错误（严格匹配完整原句）."""
    return CONTENT_FILTER_MARKER in str(exc)


def _build_judge_failure_result(prompt: str, error_marker: str) -> 'JudgeResult':
    """基于 ``_safe_call_judge`` 返回的 error_marker 构造 JudgeResult.

    在 ``parsed_result`` 里额外打上结构化标记（``content_filtered`` /
    ``infra_error``），方便下游 / 训练日志一眼识别失败类型，而不用去解析 error
    字符串前缀.
    """
    is_content_filtered = error_marker.startswith(JUDGE_CONTENT_FILTER_ERROR_PREFIX)
    is_infra_error = error_marker.startswith(JUDGE_INFRA_ERROR_PREFIX)
    parsed_marker: dict[str, Any] = {
        'content_filtered': is_content_filtered,
        'infra_error': is_infra_error,
    }
    if is_content_filtered:
        parsed_marker['content_filter_marker'] = CONTENT_FILTER_MARKER
    return JudgeResult(
        score=0.0,
        judge_prompt=prompt,
        judge_response='',
        parsed_result=parsed_marker,
        error=error_marker,
    )


async def _safe_call_judge(prefix: str, prompt: str) -> tuple[str, str | None]:
    """调用 judge 模型并返回 ``(output, error_marker)``.

    - 成功：``(response_text, None)``
    - 命中内容过滤：``('', 'content_filtered: ...')`` —— 调用方据此记为合法的 0 分
    - 其他异常：``('', 'infra_error: ...')`` —— 调用方把 error_marker 写入
      ``JudgeResult.error``，由 task 层（``search_stateful_task``）在最后统一
      判断并抛异常，避免把 judge 基础设施故障静默计入 0 分.

    注意：这里不直接抛异常，因为 ``evaluate_output`` 的通用 ``except Exception``
    会把它吞成 0.0。error_marker 前缀作为 task 层区分故障类型的信号.
    """
    ctx = require_context()
    client = ctx.get_openai_client('judge')
    try:
        response = await client.create_completion(messages=[{'role': 'user', 'content': prompt}])
    except Exception as e:
        if _is_content_filter_error(e):
            logger.warning('[%s] Judge request filtered by content policy: %s', prefix, e)
            return '', f'{JUDGE_CONTENT_FILTER_ERROR_PREFIX}{e}'
        logger.error('[%s] Judge request failed (non content-filter): %s', prefix, e)
        return '', f'{JUDGE_INFRA_ERROR_PREFIX}{e}'
    message = response.choices[0].message
    return message.content or '', None


def _parse_hle_judge_output(output: str) -> dict | None:
    """Parse HLE-style judge output and extract JSON."""
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'\{[^{}]*"correct"[^{}]*\}', output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    correct_match = re.search(r'correct["\s:]+\s*(yes|no)', output, re.IGNORECASE)
    if correct_match:
        return {'correct': correct_match.group(1).lower()}

    return None


class HLEEvaluator(Evaluator):
    """HLE benchmark evaluator using official judge prompt."""

    last_judge_result: JudgeResult | None = None

    async def evaluate(
        self,
        prediction: str,
        reference: str,
        question: str | None = None,
    ) -> float:
        clean_prediction = _strip_think_blocks(prediction)
        prompt = HLE_JUDGE_PROMPT.format(
            question=question,
            response=clean_prediction,
            correct_answer=reference,
        )

        _log_judge_input('HLE Judge', question, clean_prediction, reference, prompt)

        output, error_marker = await _safe_call_judge('HLE Judge', prompt)
        if error_marker is not None:
            logger.warning('[HLE Judge] Judge call failed (%s), returning 0.0', error_marker)
            logger.debug('=' * 80)
            self.last_judge_result = _build_judge_failure_result(prompt, error_marker)
            return 0.0

        logger.debug(f'[HLE Judge] Response:\n{output}')

        judge_response = _parse_hle_judge_output(output)
        if judge_response is not None:
            score = 1.0 if 'yes' in judge_response['correct'].lower() else 0.0
            logger.debug(f'[HLE Judge] Parsed: {judge_response}')
            logger.info(f'[HLE Judge] Score: {score}')
            logger.debug('=' * 80)
            self.last_judge_result = JudgeResult(
                score=score,
                judge_prompt=prompt,
                judge_response=output,
                parsed_result=judge_response,
            )
            return score

        logger.warning('[HLE Judge] Failed to parse output, returning 0.0')
        logger.debug('=' * 80)
        self.last_judge_result = JudgeResult(
            score=0.0,
            judge_prompt=prompt,
            judge_response=output,
            parsed_result=None,
            error='Failed to parse judge output',
        )
        return 0.0

    def get_last_judge_result(self) -> dict | None:
        if self.last_judge_result is None:
            return None
        return self.last_judge_result.to_dict()


class GeneralEvaluator(Evaluator):
    """General evaluator with detailed Chinese rules for sub-question analysis."""

    last_judge_result: JudgeResult | None = None

    async def evaluate(
        self,
        prediction: str,
        reference: str,
        question: str | None = None,
    ) -> float:
        clean_prediction = _strip_think_blocks(prediction)
        prompt = GENERAL_JUDGE_PROMPT.format(question=question, ref_answer=reference, answer=clean_prediction)

        _log_judge_input('General Judge', question, clean_prediction, reference, prompt)

        output, error_marker = await _safe_call_judge('General Judge', prompt)
        if error_marker is not None:
            logger.warning('[General Judge] Judge call failed (%s), returning 0.0', error_marker)
            logger.debug('=' * 80)
            self.last_judge_result = _build_judge_failure_result(prompt, error_marker)
            return 0.0

        logger.debug(f'[General Judge] Response:\n{output}')

        try:
            try:
                ans_check_output = json.loads(output)
            except json.JSONDecodeError:
                ans_check_output = json.loads(output.replace('\\', '\\\\'))

            if isinstance(ans_check_output, list) and len(ans_check_output) > 0:
                score = sum(x['分数'] for x in ans_check_output) / len(ans_check_output)
                logger.debug(f'[General Judge] Parsed: {ans_check_output}')
                logger.info(f'[General Judge] Score: {score}')
                logger.debug('=' * 80)
                self.last_judge_result = JudgeResult(
                    score=score,
                    judge_prompt=prompt,
                    judge_response=output,
                    parsed_result={'sub_questions': ans_check_output},
                )
                return score
            raise ValueError('Invalid output format: expected non-empty list')

        except Exception as e:
            logger.warning(f'[General Judge] Failed to parse output: {e}, returning 0.0')
            logger.debug('=' * 80)
            self.last_judge_result = JudgeResult(
                score=0.0,
                judge_prompt=prompt,
                judge_response=output,
                parsed_result=None,
                error=f'Failed to parse judge output: {str(e)}',
            )
            return 0.0

    def get_last_judge_result(self) -> dict | None:
        if self.last_judge_result is None:
            return None
        return self.last_judge_result.to_dict()


class SearchStatefulEvaluator(Evaluator):
    """Extended evaluator for HLE/Olympiad/Research/OSS judging."""

    last_judge_result: JudgeResult | None = None

    def __init__(self, evaluator_type: str = 'hle'):
        self.evaluator_type = evaluator_type

    async def evaluate(
        self,
        prediction: str,
        reference: str,
        question: str | None = None,
    ) -> float:
        clean_prediction = _strip_think_blocks(prediction)

        if self.evaluator_type == 'olympiad':
            prompt = OLYMPIAD_JUDGE_PROMPT.format(
                question=question,
                response=clean_prediction,
                correct_answer=reference,
            )
        elif self.evaluator_type == 'research':
            prompt = RESEARCH_JUDGE_PROMPT.format(
                question=question,
                response=clean_prediction,
                correct_answer=reference,
            )
        elif self.evaluator_type == 'oss':
            prompt = OSS_JUDGE_PROMPT.format(
                question=question,
                response=clean_prediction,
                correct_answer=reference,
            )
        else:
            prompt = HLE_JUDGE_PROMPT.format(
                question=question,
                response=clean_prediction,
                correct_answer=reference,
            )

        prefix = f'Judge:{self.evaluator_type}'
        _log_judge_input(prefix, question, clean_prediction, reference, prompt)

        output, error_marker = await _safe_call_judge(prefix, prompt)
        if error_marker is not None:
            logger.warning('[%s] Judge call failed (%s), returning 0.0', prefix, error_marker)
            logger.debug('=' * 80)
            self.last_judge_result = _build_judge_failure_result(prompt, error_marker)
            return 0.0

        logger.debug(f'[Judge:{self.evaluator_type}] Response:\n{output}')

        if self.evaluator_type == 'olympiad':
            return self._evaluate_olympiad(output, prompt)
        if self.evaluator_type == 'research':
            return self._evaluate_research(output, prompt)
        if self.evaluator_type == 'oss':
            return self._evaluate_oss(output, prompt)
        return self._evaluate_hle(output, prompt)

    def _evaluate_hle(self, output: str, prompt: str) -> float:
        judge_response = _parse_hle_judge_output(output)
        if judge_response is not None:
            score = 1.0 if 'yes' in judge_response['correct'].lower() else 0.0
            logger.debug(f'[Judge:{self.evaluator_type}] Parsed: {judge_response}')
            logger.info(f'[Judge:{self.evaluator_type}] Score: {score}')
            logger.debug('=' * 80)
            self.last_judge_result = JudgeResult(
                score=score,
                judge_prompt=prompt,
                judge_response=output,
                parsed_result=judge_response,
            )
            return score

        logger.warning(f'[Judge:{self.evaluator_type}] Failed to parse output, returning 0.0')
        logger.debug('=' * 80)
        self.last_judge_result = JudgeResult(
            score=0.0,
            judge_prompt=prompt,
            judge_response=output,
            parsed_result=None,
            error='Failed to parse judge output',
        )
        return 0.0

    def _evaluate_olympiad(self, output: str, prompt: str) -> float:
        try:
            score = self._extract_olympiad_score(output)
        except ValueError as e:
            logger.warning(f'[Judge:olympiad] Extraction failed: {e}, returning 0.0')
            self.last_judge_result = JudgeResult(
                score=0.0,
                judge_prompt=prompt,
                judge_response=output,
                parsed_result=None,
                error=str(e),
            )
            return 0.0

        logger.info(f'[Judge:olympiad] Score: {score}')
        logger.debug('=' * 80)
        self.last_judge_result = JudgeResult(
            score=score,
            judge_prompt=prompt,
            judge_response=output,
            parsed_result={'verdict': 'CORRECT' if score == 1.0 else 'INCORRECT'},
        )
        return score

    def _evaluate_research(self, output: str, prompt: str) -> float:
        try:
            score = self._extract_research_score(output)
        except ValueError as e:
            logger.warning(f'[Judge:research] Extraction failed: {e}, returning 0.0')
            self.last_judge_result = JudgeResult(
                score=0.0,
                judge_prompt=prompt,
                judge_response=output,
                parsed_result=None,
                error=str(e),
            )
            return 0.0

        logger.info(f'[Judge:research] Score: {score}')
        logger.debug('=' * 80)
        self.last_judge_result = JudgeResult(
            score=score,
            judge_prompt=prompt,
            judge_response=output,
            parsed_result={'verdict_score': score},
        )
        return score

    def _evaluate_oss(self, output: str, prompt: str) -> float:
        try:
            score = self._extract_oss_score(output)
        except ValueError as e:
            logger.warning(f'[Judge:oss] Extraction failed: {e}, returning 0.0')
            self.last_judge_result = JudgeResult(
                score=0.0,
                judge_prompt=prompt,
                judge_response=output,
                parsed_result=None,
                error=str(e),
            )
            return 0.0

        logger.info(f'[Judge:oss] Score: {score}')
        logger.debug('=' * 80)
        self.last_judge_result = JudgeResult(
            score=score,
            judge_prompt=prompt,
            judge_response=output,
            parsed_result={'match_result': 'consistent' if score == 1.0 else 'inconsistent'},
        )
        return score

    @staticmethod
    def _extract_olympiad_score(output: str) -> float:
        lines = [line.strip() for line in output.split('\n') if line.strip()]
        conclusion = ''
        for line in reversed(lines):
            if 'VERDICT:' in line:
                conclusion = line
                break
        if not conclusion and lines:
            conclusion = lines[-1]

        if 'INCORRECT' in conclusion:
            return 0.0
        if 'CORRECT' in conclusion:
            return 1.0
        raise ValueError(f"Can't find VERDICT in judge answer: {conclusion}")

    @staticmethod
    def _extract_research_score(output: str) -> float:
        lines = [line.strip() for line in output.split('\n') if line.strip()]
        conclusion = ''
        for line in reversed(lines):
            if 'VERDICT:' in line:
                conclusion = line
                break
        if not conclusion and lines:
            conclusion = lines[-1]

        if 'VERDICT:' not in conclusion:
            raise ValueError(f"Can't find VERDICT in judge answer: {conclusion}")

        score_val = conclusion.split('VERDICT:')[-1].strip()
        match = re.search(r'(\d+\.?\d*)', score_val)
        if not match:
            raise ValueError(f"Can't extract score from VERDICT: {score_val}")
        raw_score = float(match.group(1)) / 10
        return 1.0 if raw_score >= 0.7 else 0.0

    @staticmethod
    def _extract_oss_score(output: str) -> float:
        match = _OSS_MATCH_RE.search(output)
        if match:
            return 1.0 if match.group(1).lower() == 'consistent' else 0.0
        raise ValueError(f"Can't find <match_result> in judge answer: {output[-200:]}")

    def get_last_judge_result(self) -> dict | None:
        if self.last_judge_result is None:
            return None
        return self.last_judge_result.to_dict()


class RewardSystemEvaluator(Evaluator):
    """Evaluator that delegates scoring to the reward system API.

    Parameters are split into two groups:
    - **Client-level** (shared across all samples): ``user_token``, ``exp_id``, ``ci_test``
    - **Per-sample** (vary per evaluation): ``router_id``, ``extra``, ``public_test``, ``images``

    Per-sample fields are passed via the constructor so that each task invocation
    can create an evaluator with the correct per-sample context.

    ``images`` carries the same picture the policy was rolled out with, so a
    multimodal router judges the answer against the image rather than the
    question text alone. ``strip_multimodal_content`` drops them again for the
    text-only routers, using the same config shape as hytuner's
    ``reward.reward_kwargs.strip_multimodal_content``.
    """

    last_judge_result: dict[str, Any] | None = None

    def __init__(
        self,
        user_token: str | None = None,
        exp_id: str = '',
        router_id: str = '',
        extra: dict | None = None,
        public_test: dict | None = None,
        ci_test: bool = False,
        images: Sequence[str] | None = None,
        strip_multimodal_content: bool | list[str] | None = None,
    ):
        strip_all_routers, strip_router_ids = resolve_strip_multimodal_config(strip_multimodal_content)
        self.strip_multimodal = strip_all_routers or router_id in strip_router_ids
        self.images = [] if self.strip_multimodal else list(images or [])
        logger.debug(
            '[RewardSystemEvaluator:init] router_id=%s, exp_id=%s, ci_test=%s, token=%s, extra_count=%d, '
            'public_test_count=%d, image_count=%d, strip_multimodal=%s',
            router_id,
            exp_id or '<empty>',
            ci_test,
            user_token,
            len(extra or {}),
            len(public_test or {}),
            len(self.images),
            self.strip_multimodal,
        )
        self.client = RewardSystemClient(user_token=user_token, exp_id=exp_id)
        self.router_id = router_id
        self.extra = extra or {}
        self.public_test = public_test or {}
        self.ci_test = ci_test

    async def evaluate(
        self,
        prediction: str,
        reference: str,
        question: str | None = None,
    ) -> float:
        clean_prediction = _strip_think_blocks(prediction)
        prompt = question or ''
        messages = [
            {'role': 'system', 'content': ''},
            {'role': 'user', 'content': build_user_content(prompt, self.images)},
            {'role': 'assistant', 'content': clean_prediction},
        ]

        error: str | None = None
        try:
            score = await self.client.get_score(
                messages=messages,
                ref_answer=reference,
                router_id=self.router_id,
                extra_param=self.extra,
                unittests=self.public_test,
                ci_test=self.ci_test,
            )
        except Exception as e:
            logger.exception(
                '[RewardSystemEvaluator:evaluate] failed for router_id=%s with error=%s',
                self.router_id,
                e,
            )
            # Do not re-raise: swallow the exception and surface FAILED sentinel so the
            # task layer can pass it through to hytuner instead of losing the signal
            # in evaluate_output's [0.0, 1.0] clamp.
            score = float(ReasoningReward.FAILED.value)
            error = repr(e)

        self.last_judge_result = {
            'score': score,
            'router_id': self.router_id,
            'ci_test': self.ci_test,
            'extra_keys': sorted(self.extra.keys()),
            'public_test_keys': sorted(self.public_test.keys()),
            'image_count': len(self.images),
            'strip_multimodal': self.strip_multimodal,
            'error': error,
        }

        logger.info(
            '[RewardSystem] router_id=%s, score=%.4f, image_count=%d',
            self.router_id,
            score,
            len(self.images),
        )
        return score

    def get_last_judge_result(self) -> dict | None:
        return self.last_judge_result
