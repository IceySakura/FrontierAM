"""Olympiad LLM judge evaluator for stem_mas."""

import logging

from nexus.core import require_context
from nexus.extensions.agents.base.evaluation import Evaluator
from nexus.extensions.tasks.stem_mas.evaluator.common import log_judge_input, strip_think_blocks
from nexus.extensions.tasks.stem_mas.evaluator.prompt import JudgePrompt
from nexus.extensions.tasks.stem_mas.evaluator.result import JudgeResult


logger = logging.getLogger(__name__)


OLYMPIAD_PROMPT = JudgePrompt(
    name='olympiad',
    log_prefix='Judge:olympiad',
    template="""You are grading an attempted answer to a science olympiad problem. You will be given the
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
response, with no other text or formatting. If it is incorrect, write "VERDICT: INCORRECT".""",
)


class OlympiadEvaluator(Evaluator):
    last_judge_result: JudgeResult | None = None

    async def evaluate(
        self,
        prediction: str,
        reference: str,
        question: str | None = None,
    ) -> float:
        clean_prediction = strip_think_blocks(prediction)
        prompt = OLYMPIAD_PROMPT.format(
            question=question,
            response=clean_prediction,
            correct_answer=reference,
        )
        log_judge_input(OLYMPIAD_PROMPT.log_prefix, question, clean_prediction, reference, prompt)

        ctx = require_context()
        client = ctx.get_openai_client('judge')
        response = await client.create_completion(messages=[{'role': 'user', 'content': prompt}])
        output = response.choices[0].message.content
        logger.debug(f'[Judge:olympiad] Response:\n{output}')

        try:
            score = extract_olympiad_score(output)
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

    def get_last_judge_result(self) -> dict | None:
        if self.last_judge_result is None:
            return None
        return self.last_judge_result.to_dict()


def extract_olympiad_score(output: str) -> float:
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
