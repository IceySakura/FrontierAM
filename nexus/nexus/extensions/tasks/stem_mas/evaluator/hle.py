"""HLE LLM judge evaluator for stem_mas."""

import json
import logging
import re

from nexus.core import require_context
from nexus.extensions.agents.base.evaluation import Evaluator
from nexus.extensions.tasks.stem_mas.evaluator.common import log_judge_input, strip_think_blocks
from nexus.extensions.tasks.stem_mas.evaluator.prompt import JudgePrompt
from nexus.extensions.tasks.stem_mas.evaluator.result import JudgeResult


logger = logging.getLogger(__name__)


HLE_PROMPT = JudgePrompt(
    name='hle',
    log_prefix='HLE Judge',
    template="""Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

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
```""",
)


class HLEEvaluator(Evaluator):
    """HLE benchmark evaluator using official judge prompt."""

    last_judge_result: JudgeResult | None = None

    async def evaluate(
        self,
        prediction: str,
        reference: str,
        question: str | None = None,
    ) -> float:
        clean_prediction = strip_think_blocks(prediction)
        prompt = HLE_PROMPT.format(
            question=question,
            response=clean_prediction,
            correct_answer=reference,
        )

        log_judge_input(HLE_PROMPT.log_prefix, question, clean_prediction, reference, prompt)

        ctx = require_context()
        client = ctx.get_openai_client('judge')
        response = await client.create_completion(messages=[{'role': 'user', 'content': prompt}])
        message = response.choices[0].message
        output = message.content

        logger.debug(f'[HLE Judge] Response:\n{output}')

        judge_response = parse_hle_judge_output(output)
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


def parse_hle_judge_output(output: str) -> dict | None:
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
