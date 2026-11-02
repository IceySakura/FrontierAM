"""Research-rubric LLM judge evaluator for stem_mas."""

import logging
import re

from nexus.core import require_context
from nexus.extensions.agents.base.evaluation import Evaluator
from nexus.extensions.tasks.stem_mas.evaluator.common import log_judge_input, strip_think_blocks
from nexus.extensions.tasks.stem_mas.evaluator.prompt import JudgePrompt
from nexus.extensions.tasks.stem_mas.evaluator.result import JudgeResult


logger = logging.getLogger(__name__)


RESEARCH_PROMPT = JudgePrompt(
    name='research',
    log_prefix='Judge:research',
    template="""You are grading a science exam.
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
no other text. For example, VERDICT: 2.5 or VERDICT: 8.""",
)


class ResearchEvaluator(Evaluator):
    last_judge_result: JudgeResult | None = None

    async def evaluate(
        self,
        prediction: str,
        reference: str,
        question: str | None = None,
    ) -> float:
        clean_prediction = strip_think_blocks(prediction)
        prompt = RESEARCH_PROMPT.format(
            question=question,
            response=clean_prediction,
            correct_answer=reference,
        )
        log_judge_input(RESEARCH_PROMPT.log_prefix, question, clean_prediction, reference, prompt)

        ctx = require_context()
        client = ctx.get_openai_client('judge')
        response = await client.create_completion(messages=[{'role': 'user', 'content': prompt}])
        output = response.choices[0].message.content
        logger.debug(f'[Judge:research] Response:\n{output}')

        try:
            score = extract_research_score(output)
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

    def get_last_judge_result(self) -> dict | None:
        if self.last_judge_result is None:
            return None
        return self.last_judge_result.to_dict()


def extract_research_score(output: str) -> float:
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
