"""OSS math LLM judge evaluator for stem_mas."""

import logging
import re

from nexus.core import require_context
from nexus.extensions.agents.base.evaluation import Evaluator
from nexus.extensions.tasks.stem_mas.evaluator.common import log_judge_input, strip_think_blocks
from nexus.extensions.tasks.stem_mas.evaluator.prompt import JudgePrompt
from nexus.extensions.tasks.stem_mas.evaluator.result import JudgeResult


logger = logging.getLogger(__name__)

_OSS_MATCH_RE = re.compile(r'<match_result>\s*(consistent|inconsistent)\s*</match_result>', re.IGNORECASE)


OSS_PROMPT = JudgePrompt(
    name='oss',
    log_prefix='Judge:oss',
    template="""You are an expert mathematician and a meticulous result verifier.

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
Your task is to act as a result verifier. Your Response MUST be consisted of <match_result> as described above.""",
)


class OSSEvaluator(Evaluator):
    last_judge_result: JudgeResult | None = None

    async def evaluate(
        self,
        prediction: str,
        reference: str,
        question: str | None = None,
    ) -> float:
        clean_prediction = strip_think_blocks(prediction)
        prompt = OSS_PROMPT.format(
            question=question,
            response=clean_prediction,
            correct_answer=reference,
        )
        log_judge_input(OSS_PROMPT.log_prefix, question, clean_prediction, reference, prompt)

        ctx = require_context()
        client = ctx.get_openai_client('judge')
        response = await client.create_completion(messages=[{'role': 'user', 'content': prompt}])
        output = response.choices[0].message.content
        logger.debug(f'[Judge:oss] Response:\n{output}')

        try:
            score = extract_oss_score(output)
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

    def get_last_judge_result(self) -> dict | None:
        if self.last_judge_result is None:
            return None
        return self.last_judge_result.to_dict()


def extract_oss_score(output: str) -> float:
    match = _OSS_MATCH_RE.search(output)
    if match:
        return 1.0 if match.group(1).lower() == 'consistent' else 0.0
    raise ValueError(f"Can't find <match_result> in judge answer: {output[-200:]}")
